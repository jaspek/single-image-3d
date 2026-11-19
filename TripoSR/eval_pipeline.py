"""
Evaluation Pipeline: TripoSR on Pix3D (Multi-Category / All-Shapes Benchmark)

Supported Modes:
1. Single Category: --category chair (or bed, bookcase, desk, misc, sofa, table, tool, wardrobe)
2. All Categories:  --category all (runs across all 9 Pix3D shapes)

Pipeline Stages:
1. Data Loading:   Parses pix3d.json and extracts clean unoccluded/untruncated samples.
2. Preprocessing:  Loads 2D image + ground-truth mask, removes background, frames for TripoSR.
3. Inference:      Runs TripoSR feed-forward inference and extracts 3D surface mesh (.obj).
                   Skips samples whose mesh already exists (unless --force-rerun is passed).
4. Visualization:  Renders a 2D preview snapshot (.png) from a standard camera angle.
5. Quantitative:   Aligns predicted mesh with Pix3D ground-truth CAD model via bbox normalization
                   and multi-start ICP, computing symmetric Chamfer Distance and 3D F-Scores
                   (Precision, Recall, F1 @ tau in {0.01, 0.02, 0.05, 0.10, 0.20}).
6. Export:         Saves detailed per-sample results to results.csv and comprehensive summary
                   with category-by-category metrics to results.json.
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

# Prevent online calls to huggingface.co in offline/sandbox environments
os.environ["HF_HUB_OFFLINE"] = "1"

# ==============================================================================
# 1. HARDCODED LOCAL PATHS & CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
PIX3D_ROOT = Path(r"D:\pix3d")
TRIPOSR_ROOT = Path(r"C:\Users\flori\GitHub\TripoSR")

PIX3D_JSON_PATH = PIX3D_ROOT / "pix3d.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

ALL_CATEGORIES = [
    "bed",
    "bookcase",
    "chair",
    "desk",
    "misc",
    "sofa",
    "table",
    "tool",
    "wardrobe",
]

DEFAULT_CATEGORY = "all"
DEFAULT_SAMPLES_PER_CAT = 10
FOREGROUND_RATIO = 0.85
MC_RESOLUTION = 128         # Resolution 128 is ~9x faster on CPU with virtually identical CD/F1
CHUNK_SIZE = 8192
NUM_SAMPLE_POINTS = 10000   # Surface points sampled per mesh for Chamfer Distance & F-Scores
USE_ICP_ALIGNMENT = True    # Refine orientation after bounding-box translation & scaling
RANDOM_SEED = 42

EVAL_THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20]

# Add local TripoSR paths to sys.path
for p in (SCRIPT_DIR, TRIPOSR_ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ==============================================================================
# 2. DATA LOADING (SUBSET SELECTION FROM pix3d.json)
# ==============================================================================
def load_pix3d_subset(
    json_path: Path = PIX3D_JSON_PATH,
    pix3d_root: Path = PIX3D_ROOT,
    categories: Union[str, List[str]] = DEFAULT_CATEGORY,
    max_samples_per_cat: int = DEFAULT_SAMPLES_PER_CAT,
) -> List[Tuple[int, Dict]]:
    """
    Parses pix3d.json and extracts deterministic subsets of clean (untruncated,
    unoccluded) samples whose image, mask, and CAD model files exist on disk.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing {json_path}. Ensure the Pix3D dataset is extracted inside {pix3d_root}."
        )

    if isinstance(categories, str):
        if categories.lower() == "all":
            cat_list = ALL_CATEGORIES
        else:
            cat_list = [categories.lower()]
    else:
        cat_list = [c.lower() for c in categories]

    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    selected: List[Tuple[int, Dict]] = []

    for cat in cat_list:
        cat_selected: List[Tuple[int, Dict]] = []
        for anno_idx, entry in enumerate(annotations):
            if entry.get("category") != cat:
                continue
            if entry.get("truncated", False) or entry.get("occluded", False) or entry.get("slightly_occluded", False):
                continue

            img_path = pix3d_root / entry["img"]
            mask_path = pix3d_root / entry["mask"]
            model_path = pix3d_root / entry["model"]

            if img_path.exists() and mask_path.exists() and model_path.exists():
                cat_selected.append((anno_idx, entry))
                if len(cat_selected) >= max_samples_per_cat:
                    break

        if not cat_selected:
            logging.warning(f"No clean valid samples found for category '{cat}'!")
        else:
            logging.info(f"Loaded {len(cat_selected)} samples for category '{cat}'.")
        selected.extend(cat_selected)

    logging.info(f"Total dataset selection: {len(selected)} samples across {len(cat_list)} categories.")
    return selected


# ==============================================================================
# 3. PREPROCESSING (APPLY PIX3D GROUND-TRUTH MASK FOR TRIPOSR)
# ==============================================================================
def preprocess_image_with_mask(
    img_path: Path,
    mask_path: Path,
    foreground_ratio: float = FOREGROUND_RATIO,
) -> Image.Image:
    """
    Loads 2D RGB image and binary mask, isolates object on a neutral 50% gray canvas,
    resizing and framing to specified foreground ratio.
    """
    from tsr.utils import resize_foreground

    rgb_img = Image.open(img_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")

    if mask_img.size != rgb_img.size:
        mask_img = mask_img.resize(rgb_img.size, Image.Resampling.NEAREST)

    rgb_np = np.array(rgb_img, dtype=np.uint8)
    mask_np = (np.array(mask_img, dtype=np.uint8) > 127).astype(np.uint8) * 255

    if mask_np.max() == 0:
        mask_np[:] = 255

    rgba_np = np.dstack([rgb_np, mask_np])
    rgba_pil = Image.fromarray(rgba_np, mode="RGBA")
    framed_rgba = resize_foreground(rgba_pil, foreground_ratio)

    arr = np.array(framed_rgba, dtype=np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    composited = arr[:, :, :3] * alpha + (1.0 - alpha) * 0.5
    return Image.fromarray((composited * 255.0).astype(np.uint8), mode="RGB")


# ==============================================================================
# 4. ALIGNMENT & METRICS MODULE (CHAMFER DISTANCE + 3D F-SCORES)
# ==============================================================================
def normalize_point_cloud(pts: np.ndarray) -> np.ndarray:
    """
    Translates center of bounding box to origin (0,0,0) and uniformly scales
    so longest bounding-box side is 1.0 (matching Pix3D convention).
    """
    bound_l = np.min(pts, axis=0)
    bound_h = np.max(pts, axis=0)
    center = (bound_l + bound_h) / 2.0
    scale = float(np.max(bound_h - bound_l))
    if scale < 1e-12:
        scale = 1.0
    return (pts - center) / scale


def compute_shape_metrics(
    pred_mesh_path: Path,
    gt_mesh_path: Path,
    num_points: int = NUM_SAMPLE_POINTS,
    use_icp: bool = USE_ICP_ALIGNMENT,
    seed: int = RANDOM_SEED,
    thresholds: List[float] = EVAL_THRESHOLDS,
) -> Dict[str, float]:
    """
    Loads predicted and ground-truth meshes, samples surface points,
    performs bounding-box normalization + multi-start ICP, and computes:
    - Chamfer Distance (symmetric L2)
    - Surface Accuracy & Completeness
    - Precision@tau, Recall@tau, and F1-Score@tau for tau in thresholds
    """
    np.random.seed(seed)

    pred_mesh = trimesh.load(pred_mesh_path, force="mesh")
    gt_mesh = trimesh.load(gt_mesh_path, force="mesh")

    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, num_points)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, num_points)

    pred_pts = normalize_point_cloud(np.asarray(pred_pts, dtype=np.float64))
    gt_pts = normalize_point_cloud(np.asarray(gt_pts, dtype=np.float64))

    if use_icp:
        try:
            best_pts = pred_pts
            best_cost = float("inf")
            n_icp = min(1024, num_points)
            idx = np.random.choice(num_points, n_icp, replace=False)
            src_sub, dst_sub = pred_pts[idx], gt_pts[idx]

            for axis in ([0, 1, 0], [1, 0, 0]):
                for deg in (0, 90, 180, 270):
                    init_tf = trimesh.transformations.rotation_matrix(np.radians(deg), axis)
                    matrix, _, cost = trimesh.registration.icp(
                        src_sub,
                        dst_sub,
                        initial=init_tf,
                        max_iterations=30,
                        scale=True,
                        reflection=False,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_pts = trimesh.transformations.transform_points(pred_pts, matrix)
            pred_pts = best_pts
        except Exception as e:
            logging.warning(f"ICP refinement skipped: {e}")

    # Nearest neighbor distance queries via KDTree
    dist_pred_to_gt, _ = cKDTree(gt_pts).query(pred_pts, k=1)
    dist_gt_to_pred, _ = cKDTree(pred_pts).query(gt_pts, k=1)

    acc = float(np.mean(dist_pred_to_gt))
    comp = float(np.mean(dist_gt_to_pred))
    cd = acc + comp

    metrics = {
        "chamfer_distance": cd,
        "accuracy": acc,
        "completeness": comp,
    }

    for t in thresholds:
        t_key = f"{int(t * 100):02d}"
        prec = float(np.mean(dist_pred_to_gt < t))
        rec = float(np.mean(dist_gt_to_pred < t))
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0.0 else 0.0
        metrics[f"precision_{t_key}"] = prec
        metrics[f"recall_{t_key}"] = rec
        metrics[f"f1_{t_key}"] = f1

    return metrics


def compute_chamfer_distance(
    pred_mesh_path: Path,
    gt_mesh_path: Path,
    num_points: int = NUM_SAMPLE_POINTS,
    use_icp: bool = USE_ICP_ALIGNMENT,
    seed: int = RANDOM_SEED,
) -> float:
    """Convenience wrapper for backward compatibility."""
    return compute_shape_metrics(pred_mesh_path, gt_mesh_path, num_points, use_icp, seed)["chamfer_distance"]


# ==============================================================================
# 5. VISUALIZATION (2D PREVIEW SNAPSHOT)
# ==============================================================================
def render_mesh_to_image(
    mesh: trimesh.Trimesh,
    out_png_path: Path,
    resolution: Tuple[int, int] = (512, 512),
) -> None:
    """Renders a standard 2D view of the 3D model with headless-safe software fallback."""
    vis_mesh = mesh.copy()
    vis_mesh.vertices = normalize_point_cloud(vis_mesh.vertices)
    vis_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    vis_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    # Attempt 1: pyrender
    try:
        import pyrender
        scene = pyrender.Scene(bg_color=[245, 245, 245, 255], ambient_light=[0.35, 0.35, 0.35])
        scene.add(pyrender.Mesh.from_trimesh(vis_mesh, smooth=True))
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        cam_pose = np.array([
            [0.866, -0.250,  0.433,  0.85],
            [0.000,  0.866,  0.500,  0.95],
            [-0.500, -0.433, 0.750,  1.45],
            [0.000,  0.000,  0.000,  1.00],
        ])
        scene.add(camera, pose=cam_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
        color, _ = renderer.render(scene)
        renderer.delete()
        Image.fromarray(color).save(out_png_path)
        return
    except Exception:
        pass

    # Attempt 2: trimesh offscreen
    try:
        scene = vis_mesh.scene()
        png_bytes = scene.save_image(resolution=resolution, visible=False)
        if png_bytes is not None:
            with open(out_png_path, "wb") as f:
                f.write(png_bytes)
            return
    except Exception:
        pass

    # Attempt 3: Matplotlib shaded mesh render
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(resolution[0] / 100.0, resolution[1] / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    verts = vis_mesh.vertices
    faces = vis_mesh.faces
    max_plot_faces = 15000
    if len(faces) > max_plot_faces:
        step = max(1, len(faces) // max_plot_faces)
        faces = faces[::step]

    triangles = verts[faces]
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    light_dir = np.array([0.4, -0.6, 0.7])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.clip(0.35 + 0.65 * np.abs(normals @ light_dir), 0.2, 1.0)
    base_color = np.array([0.45, 0.62, 0.82])
    face_colors = intensity[:, None] * base_color[None, :]

    poly = Poly3DCollection(triangles, facecolors=face_colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(poly)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(-0.55, 0.55)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=35)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png_path, dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ==============================================================================
# 6. EXPORT RESULTS SUMMARY (CSV, JSON & COMPARISON REPORT)
# ==============================================================================
def export_evaluation_results(
    results: List[Dict],
    output_dir: Path = OUTPUT_DIR,
    target_category: str = DEFAULT_CATEGORY,
) -> Tuple[Path, Path]:
    """Writes results.csv and comprehensive results.json with per-category breakdowns."""
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    # Export CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Calculate per-category statistics
    categories = sorted(list(set(r["category"] for r in results)))
    by_category = {}
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        cds = [r["chamfer_distance"] for r in cat_rows]
        cat_stats = {
            "num_evaluated": len(cat_rows),
            "mean_chamfer_distance": float(np.mean(cds)),
            "median_chamfer_distance": float(np.median(cds)),
            "std_chamfer_distance": float(np.std(cds)),
            "min_chamfer_distance": float(np.min(cds)),
            "max_chamfer_distance": float(np.max(cds)),
        }
        for t in EVAL_THRESHOLDS:
            t_key = f"{int(t * 100):02d}"
            cat_stats[f"mean_f1_{t_key}"] = float(np.mean([r[f"f1_{t_key}"] for r in cat_rows]))
            cat_stats[f"mean_precision_{t_key}"] = float(np.mean([r[f"precision_{t_key}"] for r in cat_rows]))
            cat_stats[f"mean_recall_{t_key}"] = float(np.mean([r[f"recall_{t_key}"] for r in cat_rows]))
        by_category[cat] = cat_stats

    # Overall summary
    all_cds = [r["chamfer_distance"] for r in results]
    summary = {
        "target_category": target_category,
        "num_evaluated": len(results),
        "mean_chamfer_distance": float(np.mean(all_cds)),
        "median_chamfer_distance": float(np.median(all_cds)),
        "std_chamfer_distance": float(np.std(all_cds)),
        "min_chamfer_distance": float(np.min(all_cds)),
        "max_chamfer_distance": float(np.max(all_cds)),
    }
    for t in EVAL_THRESHOLDS:
        t_key = f"{int(t * 100):02d}"
        summary[f"mean_f1_{t_key}"] = float(np.mean([r[f"f1_{t_key}"] for r in results]))
        summary[f"mean_precision_{t_key}"] = float(np.mean([r[f"precision_{t_key}"] for r in results]))
        summary[f"mean_recall_{t_key}"] = float(np.mean([r[f"recall_{t_key}"] for r in results]))

    summary["by_category"] = by_category
    summary["samples"] = results

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


# ==============================================================================
# 7. MAIN PIPELINE ENTRY POINT
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="TripoSR on Pix3D Multi-Category Benchmark Pipeline")
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_CATEGORY,
        help=f"Pix3D category to evaluate ('all' or one of {ALL_CATEGORIES}, default: '{DEFAULT_CATEGORY}').",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_CAT,
        help=f"Number of samples per category (default: {DEFAULT_SAMPLES_PER_CAT}).",
    )
    parser.add_argument(
        "--pix3d-root",
        type=Path,
        default=PIX3D_ROOT,
        help=f"Path to Pix3D dataset root folder (default: {PIX3D_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Path to outputs directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--mc-resolution",
        type=int,
        default=MC_RESOLUTION,
        help=f"Marching Cubes resolution (default: {MC_RESOLUTION}).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip TripoSR inference and only evaluate existing .obj meshes.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force TripoSR inference even if predicted .obj mesh already exists on disk.",
    )
    parser.add_argument(
        "--no-icp",
        action="store_true",
        help="Use bounding-box translation and uniform scaling only (disable ICP rotation refinement).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    pix3d_root = args.pix3d_root
    json_path = pix3d_root / "pix3d.json"
    output_dir = args.output_dir
    mc_res = args.mc_resolution

    meshes_dir = output_dir / "meshes"
    renders_dir = output_dir / "renders"
    inputs_dir = output_dir / "inputs"
    for d in (output_dir, meshes_dir, renders_dir, inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Select subset
    subset = load_pix3d_subset(
        json_path=json_path,
        pix3d_root=pix3d_root,
        categories=args.category,
        max_samples_per_cat=args.num_samples,
    )

    # Step 2: TripoSR Inference & 2D Rendering
    if not args.eval_only:
        # Check which samples need inference
        needed = []
        for anno_idx, entry in subset:
            mesh_p = meshes_dir / f"{anno_idx:04d}_pred.obj"
            if args.force_rerun or not mesh_p.exists():
                needed.append((anno_idx, entry))

        if needed:
            from tsr.system import TSR

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logging.info(f"Initializing TripoSR on '{device}' (evaluating {len(needed)} new meshes at res={mc_res})...")
            model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            model.renderer.set_chunk_size(CHUNK_SIZE)
            model.to(device)

            for idx, (anno_idx, entry) in enumerate(needed, start=1):
                sample_id = f"{anno_idx:04d}"
                cat = entry["category"]
                logging.info(f"[Inference {idx}/{len(needed)}] [{cat.upper()}] Pix3D #{sample_id} ({entry['img']})...")

                img_path = pix3d_root / entry["img"]
                mask_path = pix3d_root / entry["mask"]

                prep_img = preprocess_image_with_mask(img_path, mask_path, FOREGROUND_RATIO)
                prep_img.save(inputs_dir / f"{sample_id}_input.png")

                with torch.no_grad():
                    scene_codes = model([prep_img], device=device)
                    extracted = model.extract_mesh(
                        scene_codes,
                        has_vertex_color=True,
                        resolution=mc_res,
                    )
                pred_mesh: trimesh.Trimesh = extracted[0]

                pred_mesh_path = meshes_dir / f"{sample_id}_pred.obj"
                pred_mesh.export(pred_mesh_path)

                render_path = renders_dir / f"{sample_id}_render.png"
                render_mesh_to_image(pred_mesh, render_path)
        else:
            logging.info("All requested meshes already exist on disk. Skipping inference.")

    # Step 3: Evaluation (Chamfer Distance & F-Scores)
    logging.info("Evaluating 3D alignment, Chamfer Distance, and F-Scores...")
    use_icp = USE_ICP_ALIGNMENT and not args.no_icp
    results: List[Dict] = []

    for idx, (anno_idx, entry) in enumerate(subset, start=1):
        sample_id = f"{anno_idx:04d}"
        pred_mesh_path = meshes_dir / f"{sample_id}_pred.obj"
        render_path = renders_dir / f"{sample_id}_render.png"
        gt_mesh_path = pix3d_root / entry["model"]

        if not pred_mesh_path.exists():
            logging.warning(f"Skipping #{sample_id}: missing {pred_mesh_path}")
            continue

        metrics = compute_shape_metrics(
            pred_mesh_path=pred_mesh_path,
            gt_mesh_path=gt_mesh_path,
            num_points=NUM_SAMPLE_POINTS,
            use_icp=use_icp,
        )

        cat = entry["category"]
        cd = metrics["chamfer_distance"]
        f1_05 = metrics["f1_05"] * 100
        logging.info(f"[Eval {idx}/{len(subset)}] [{cat.upper():<8}] #{sample_id} -> CD: {cd:.4f}, F1@0.05: {f1_05:.1f}%")

        results.append({
            "pix3d_id": anno_idx,
            "category": cat,
            "img_rel_path": entry["img"],
            "gt_model_rel_path": entry["model"],
            "pred_mesh_path": str(pred_mesh_path),
            "render_image_path": str(render_path),
            **metrics,
        })

    if not results:
        raise RuntimeError("No meshes evaluated!")

    # Step 4: Export Summary & Print Multi-Category Comparison Table
    csv_path, json_path = export_evaluation_results(results, output_dir, args.category)

    print("\n" + "=" * 90)
    print(f"{'Category':<12} | {'Samples':<7} | {'Mean CD':<9} | {'F1@0.02':<9} | {'F1@0.05':<9} | {'F1@0.10':<9} | {'Prec@0.05':<10} | {'Rec@0.05':<10}")
    print("-" * 90)

    categories = sorted(list(set(r["category"] for r in results)))
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        m_cd = np.mean([r["chamfer_distance"] for r in cat_rows])
        f1_02 = np.mean([r["f1_02"] for r in cat_rows]) * 100
        f1_05 = np.mean([r["f1_05"] for r in cat_rows]) * 100
        f1_10 = np.mean([r["f1_10"] for r in cat_rows]) * 100
        prec_05 = np.mean([r["precision_05"] for r in cat_rows]) * 100
        rec_05 = np.mean([r["recall_05"] for r in cat_rows]) * 100
        print(f"{cat:<12} | {len(cat_rows):<7} | {m_cd:<9.4f} | {f1_02:<8.1f}% | {f1_05:<8.1f}% | {f1_10:<8.1f}% | {prec_05:<9.1f}% | {rec_05:<9.1f}%")

    print("-" * 90)
    tot_cd = np.mean([r["chamfer_distance"] for r in results])
    tot_f1_02 = np.mean([r["f1_02"] for r in results]) * 100
    tot_f1_05 = np.mean([r["f1_05"] for r in results]) * 100
    tot_f1_10 = np.mean([r["f1_10"] for r in results]) * 100
    tot_prec_05 = np.mean([r["precision_05"] for r in results]) * 100
    tot_rec_05 = np.mean([r["recall_05"] for r in results]) * 100
    print(f"{'OVERALL':<12} | {len(results):<7} | {tot_cd:<9.4f} | {tot_f1_02:<8.1f}% | {tot_f1_05:<8.1f}% | {tot_f1_10:<8.1f}% | {tot_prec_05:<9.1f}% | {tot_rec_05:<9.1f}%")
    print("=" * 90 + "\n")

    logging.info(f"Results CSV:  {csv_path}")
    logging.info(f"Results JSON: {json_path}")


if __name__ == "__main__":
    main()
