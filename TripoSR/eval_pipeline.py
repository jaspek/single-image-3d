"""
Minimum Viable Product (MVP) Evaluation Pipeline: TripoSR on Pix3D (50-image subset)

Workspace (Read/Write): C:\\Users\\flori\\GitHub\\single-image-3d
Pix3D Source (Read):    C:\\Users\\flori\\GitHub\\pix3d
TripoSR Source (Read):  C:\\Users\\flori\\GitHub\\TripoSR
Outputs Directory:      C:\\Users\\flori\\GitHub\\single-image-3d\\outputs\\

Pipeline Stages:
1. Data Loading:   Parses pix3d.json and extracts a fixed subset of ~50 "chair" samples.
2. Preprocessing:  Loads each 2D image and ground-truth mask, removes background, and frames for TripoSR.
3. Inference:      Runs TripoSR to generate one 3D mesh (.obj) per image (skipped if --eval-only is passed).
4. Visualization:  Renders one 2D image (.png) of each generated 3D mesh from a standard camera angle.
5. Evaluation:     Loads each generated .obj mesh and its Pix3D ground-truth CAD model, performs
                   bounding-box alignment (translation + uniform scaling + optional ICP), computes
                   Chamfer Distance (CD), and exports results.csv and results.json to outputs/.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

# ==============================================================================
# 1. HARDCODED LOCAL PATHS & CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
PIX3D_ROOT = Path(r"D:\pix3d")
TRIPOSR_ROOT = Path(r"C:\Users\flori\GitHub\TripoSR")

PIX3D_JSON_PATH = PIX3D_ROOT / "pix3d.json"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Evaluation parameters
TARGET_CATEGORY = "chair"
SUBSET_SIZE = 50
FOREGROUND_RATIO = 0.85
MC_RESOLUTION = 256
CHUNK_SIZE = 8192
NUM_SAMPLE_POINTS = 10000  # Surface points sampled per mesh for Chamfer Distance
USE_ICP_ALIGNMENT = True   # Refine orientation after bounding-box translation & scaling
RANDOM_SEED = 42

# Add TripoSR repo to sys.path (read-only import)
if str(TRIPOSR_ROOT) not in sys.path:
    sys.path.insert(0, str(TRIPOSR_ROOT))


# ==============================================================================
# 2. DATA LOADING (FIXED SUBSET OF ~50 IMAGES FROM pix3d.json)
# ==============================================================================
def load_pix3d_subset(
    json_path: Path = PIX3D_JSON_PATH,
    pix3d_root: Path = PIX3D_ROOT,
    category: str = TARGET_CATEGORY,
    max_samples: int = SUBSET_SIZE,
) -> List[Tuple[int, Dict]]:
    """
    Parses pix3d.json and extracts a deterministic subset of up to `max_samples`
    untruncated, unoccluded samples from `category` whose files exist on disk.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing {json_path}. Ensure the Pix3D dataset is extracted inside {pix3d_root}."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    selected: List[Tuple[int, Dict]] = []
    total_in_category = 0
    clean_in_category = 0
    missing_img = 0
    missing_mask = 0
    missing_model = 0

    for anno_idx, entry in enumerate(annotations):
        if entry.get("category") != category:
            continue
        total_in_category += 1
        if entry.get("truncated", False) or entry.get("occluded", False) or entry.get("slightly_occluded", False):
            continue
        clean_in_category += 1

        img_path = pix3d_root / entry["img"]
        mask_path = pix3d_root / entry["mask"]
        model_path = pix3d_root / entry["model"]

        img_ok = img_path.exists()
        mask_ok = mask_path.exists()
        model_ok = model_path.exists()

        if not img_ok:
            missing_img += 1
        if not mask_ok:
            missing_mask += 1
        if not model_ok:
            missing_model += 1

        if img_ok and mask_ok and model_ok:
            selected.append((anno_idx, entry))
            if len(selected) >= max_samples:
                break

    if not selected:
        raise RuntimeError(
            f"No valid samples found for category '{category}' in {json_path}.\n"
            f"Category statistics: total={total_in_category}, unoccluded={clean_in_category}, "
            f"missing img={missing_img}, missing mask={missing_mask}, missing model={missing_model}.\n"
            f"Verify that img/, mask/, and model/ are completely extracted inside {pix3d_root}."
        )

    logging.info(f"Loaded fixed subset of {len(selected)} '{category}' samples from {json_path}.")
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
    Loads the 2D RGB image and its binary Pix3D mask, removes the background using
    the ground-truth mask, crops/pads the foreground using TripoSR's `resize_foreground`,
    and composites onto a neutral 50% gray background.
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
# 4. ALIGNMENT & METRIC MODULE (TRANSLATION/SCALING + CHAMFER DISTANCE)
# ==============================================================================
def normalize_point_cloud(pts: np.ndarray) -> np.ndarray:
    """
    Basic & robust bounding-box alignment (translation + uniform scaling):
    1. Translates the axis-aligned bounding-box center to the origin (0, 0, 0).
    2. Scales uniformly so the longest bounding-box side equals 1.0.
    Matches the normalization convention in Pix3D's eval/eval.py.
    """
    bound_l = np.min(pts, axis=0)
    bound_h = np.max(pts, axis=0)
    center = (bound_l + bound_h) / 2.0
    scale = float(np.max(bound_h - bound_l))
    if scale < 1e-12:
        scale = 1.0
    return (pts - center) / scale


def compute_chamfer_distance(
    pred_mesh_path: Path,
    gt_mesh_path: Path,
    num_points: int = NUM_SAMPLE_POINTS,
    use_icp: bool = USE_ICP_ALIGNMENT,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Loads a generated TripoSR .obj mesh and its corresponding Pix3D ground-truth .obj model,
    samples surface point clouds, performs bounding-box translation & scaling alignment
    (plus optional multi-start ICP), and returns the symmetric Chamfer Distance (L2).
    """
    np.random.seed(seed)

    pred_mesh = trimesh.load(pred_mesh_path, force="mesh")
    gt_mesh = trimesh.load(gt_mesh_path, force="mesh")

    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, num_points)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, num_points)

    # Step 1: Basic translation & uniform scaling to [-0.5, 0.5]^3 unit bounding box
    pred_pts = normalize_point_cloud(np.asarray(pred_pts, dtype=np.float64))
    gt_pts = normalize_point_cloud(np.asarray(gt_pts, dtype=np.float64))

    # Step 2: Optional multi-start ICP rotation/scale refinement
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
            logging.warning(f"ICP refinement skipped (using bbox translation/scaling only): {e}")

    # Step 3: Symmetric Chamfer Distance = mean(d(pred -> gt)) + mean(d(gt -> pred))
    dist_pred_to_gt, _ = cKDTree(gt_pts).query(pred_pts, k=1)
    dist_gt_to_pred, _ = cKDTree(pred_pts).query(gt_pts, k=1)
    chamfer_dist = float(np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred))
    return chamfer_dist


# ==============================================================================
# 5. VISUALIZATION (ONE 2D PICTURE FROM A STANDARD CAMERA ANGLE)
# ==============================================================================
def render_mesh_to_image(
    mesh: trimesh.Trimesh,
    out_png_path: Path,
    resolution: Tuple[int, int] = (512, 512),
) -> None:
    """
    Renders a single 2D image of the generated 3D model from a standard camera angle.
    Tries pyrender / trimesh first, and falls back to a headless-safe shaded render
    so it never fails on Windows if an OpenGL context is unavailable.
    """
    vis_mesh = mesh.copy()
    vis_mesh.vertices = normalize_point_cloud(vis_mesh.vertices)
    vis_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    vis_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    # Attempt 1: Offscreen render via pyrender
    try:
        import pyrender

        scene = pyrender.Scene(bg_color=[245, 245, 245, 255], ambient_light=[0.35, 0.35, 0.35])
        pyr_mesh = pyrender.Mesh.from_trimesh(vis_mesh, smooth=True)
        scene.add(pyr_mesh)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        cam_pose = np.array([
            [0.866, -0.250,  0.433,  0.85],
            [0.000,  0.866,  0.500,  0.95],
            [-0.500, -0.433, 0.750,  1.45],
            [0.000,  0.000,  0.000,  1.00],
        ])
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=cam_pose)

        renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
        color, _ = renderer.render(scene)
        renderer.delete()
        Image.fromarray(color).save(out_png_path)
        return
    except Exception:
        pass

    # Attempt 2: Offscreen render via trimesh.Scene.save_image
    try:
        scene = vis_mesh.scene()
        png_bytes = scene.save_image(resolution=resolution, visible=False)
        if png_bytes is not None:
            with open(out_png_path, "wb") as f:
                f.write(png_bytes)
            return
    except Exception:
        pass

    # Attempt 3: Headless-safe software shaded mesh render via Matplotlib
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
# 6. EXPORT RESULTS SUMMARY (CSV & JSON)
# ==============================================================================
def export_evaluation_results(
    results: List[Dict],
    output_dir: Path = OUTPUT_DIR,
    category: str = TARGET_CATEGORY,
) -> Tuple[Path, Path]:
    """
    Writes per-sample Chamfer Distance metrics and aggregate summary statistics
    to `outputs/results.csv` and `outputs/results.json`.
    """
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    cd_scores = [r["chamfer_distance"] for r in results]
    summary = {
        "category": category,
        "num_evaluated": len(results),
        "mean_chamfer_distance": float(np.mean(cd_scores)),
        "median_chamfer_distance": float(np.median(cd_scores)),
        "std_chamfer_distance": float(np.std(cd_scores)),
        "min_chamfer_distance": float(np.min(cd_scores)),
        "max_chamfer_distance": float(np.max(cd_scores)),
        "samples": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


# ==============================================================================
# 7. MAIN PIPELINE ENTRY POINT
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="TripoSR on Pix3D (MVP Evaluation Pipeline)")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=SUBSET_SIZE,
        help=f"Number of samples to evaluate (default: {SUBSET_SIZE}).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=TARGET_CATEGORY,
        help=f"Pix3D category to evaluate (default: '{TARGET_CATEGORY}').",
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
        "--eval-only",
        action="store_true",
        help="Skip TripoSR inference and only evaluate existing .obj meshes in outputs/meshes/.",
    )
    parser.add_argument(
        "--no-icp",
        action="store_true",
        help="Use bounding-box translation and uniform scaling only (disable ICP rotation refinement).",
    )
    args = parser.parse_args()

    pix3d_root = args.pix3d_root
    json_path = pix3d_root / "pix3d.json"
    category = args.category
    num_samples = args.num_samples
    output_dir = args.output_dir

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    meshes_dir = output_dir / "meshes"
    renders_dir = output_dir / "renders"
    inputs_dir = output_dir / "inputs"
    for d in (output_dir, meshes_dir, renders_dir, inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Load fixed subset of Pix3D images
    subset = load_pix3d_subset(
        json_path=json_path,
        pix3d_root=pix3d_root,
        category=category,
        max_samples=num_samples,
    )

    # Step 2: Run Preprocessing + TripoSR Inference + 2D Rendering (unless --eval-only)
    if not args.eval_only:
        from tsr.system import TSR

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing TripoSR on '{device}'...")
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(CHUNK_SIZE)
        model.to(device)

        for idx, (anno_idx, entry) in enumerate(subset, start=1):
            sample_id = f"{anno_idx:04d}"
            logging.info(f"[Inference {idx}/{len(subset)}] Pix3D sample #{sample_id} ({entry['img']})...")

            img_path = pix3d_root / entry["img"]
            mask_path = pix3d_root / entry["mask"]

            # 2a. Preprocess image with ground-truth mask
            prep_img = preprocess_image_with_mask(img_path, mask_path, FOREGROUND_RATIO)
            prep_img.save(inputs_dir / f"{sample_id}_input.png")

            # 2b. Generate 3D mesh (.obj) with TripoSR
            with torch.no_grad():
                scene_codes = model([prep_img], device=device)
                extracted = model.extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=MC_RESOLUTION,
                )
            pred_mesh: trimesh.Trimesh = extracted[0]

            pred_mesh_path = meshes_dir / f"{sample_id}_pred.obj"
            pred_mesh.export(pred_mesh_path)

            # 2c. Render one 2D image (.png) from a standard camera angle
            render_path = renders_dir / f"{sample_id}_render.png"
            render_mesh_to_image(pred_mesh, render_path)

    # Step 3: Quantitative Evaluation Module (Load .obj meshes -> Align -> Chamfer Distance)
    logging.info("Running quantitative Chamfer Distance evaluation on generated meshes...")
    use_icp = USE_ICP_ALIGNMENT and not args.no_icp
    results: List[Dict] = []

    for idx, (anno_idx, entry) in enumerate(subset, start=1):
        sample_id = f"{anno_idx:04d}"
        pred_mesh_path = meshes_dir / f"{sample_id}_pred.obj"
        render_path = renders_dir / f"{sample_id}_render.png"
        gt_mesh_path = pix3d_root / entry["model"]

        if not pred_mesh_path.exists():
            logging.warning(f"Skipping sample #{sample_id}: missing generated mesh {pred_mesh_path}")
            continue

        chamfer_dist = compute_chamfer_distance(
            pred_mesh_path=pred_mesh_path,
            gt_mesh_path=gt_mesh_path,
            num_points=NUM_SAMPLE_POINTS,
            use_icp=use_icp,
        )

        logging.info(f"[Eval {idx}/{len(subset)}] Sample #{sample_id} -> Chamfer Distance: {chamfer_dist:.6f}")

        results.append({
            "pix3d_id": anno_idx,
            "category": entry["category"],
            "img_rel_path": entry["img"],
            "gt_model_rel_path": entry["model"],
            "pred_mesh_path": str(pred_mesh_path),
            "render_image_path": str(render_path),
            "chamfer_distance": chamfer_dist,
        })

    if not results:
        raise RuntimeError(f"No meshes were evaluated. Check {meshes_dir} for generated .obj files.")

    # Step 4: Export final results
    csv_path, json_path = export_evaluation_results(results, output_dir, category)

    cd_scores = [r["chamfer_distance"] for r in results]
    logging.info("=" * 60)
    logging.info(f"Evaluation Complete! ({len(results)} samples)")
    logging.info(f"Mean Chamfer Distance:   {np.mean(cd_scores):.6f}")
    logging.info(f"Median Chamfer Distance: {np.median(cd_scores):.6f}")
    logging.info(f"CSV Summary:             {csv_path}")
    logging.info(f"JSON Summary:            {json_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
