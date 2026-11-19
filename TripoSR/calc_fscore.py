"""
3D F-Score Evaluation Tool for TripoSR on Pix3D (Multi-Category Analysis)

Computes and displays Precision@tau, Recall@tau, and F1-Score@tau across
all Pix3D shapes and thresholds tau in {0.01, 0.02, 0.05, 0.10, 0.20}.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "outputs" / "results.json"
if not RESULTS_PATH.exists():
    RESULTS_PATH = SCRIPT_DIR.parent / "outputs" / "results.json"

PIX3D_ROOT = Path(r"D:\pix3d")
THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20]


def normalize_point_cloud(pts: np.ndarray) -> np.ndarray:
    bound_l = np.min(pts, axis=0)
    bound_h = np.max(pts, axis=0)
    center = (bound_l + bound_h) / 2.0
    scale = float(np.max(bound_h - bound_l))
    if scale < 1e-12:
        scale = 1.0
    return (pts - center) / scale


def get_aligned_points(pred_path: Path, gt_path: Path, num_points: int = 5000, seed: int = 42):
    np.random.seed(seed)
    pred_mesh = trimesh.load(pred_path, force="mesh")
    gt_mesh = trimesh.load(gt_path, force="mesh")
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, num_points)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, num_points)
    pred_pts = normalize_point_cloud(np.asarray(pred_pts, dtype=np.float64))
    gt_pts = normalize_point_cloud(np.asarray(gt_pts, dtype=np.float64))

    best_pts = pred_pts
    best_cost = float("inf")
    n_icp = min(1024, num_points)
    idx = np.random.choice(num_points, n_icp, replace=False)
    src_sub, dst_sub = pred_pts[idx], gt_pts[idx]
    for axis in ([0, 1, 0], [1, 0, 0]):
        for deg in (0, 90, 180, 270):
            init_tf = trimesh.transformations.rotation_matrix(np.radians(deg), axis)
            matrix, _, cost = trimesh.registration.icp(
                src_sub, dst_sub, initial=init_tf, max_iterations=30, scale=True, reflection=False
            )
            if cost < best_cost:
                best_cost = cost
                best_pts = trimesh.transformations.transform_points(pred_pts, matrix)
    return best_pts, gt_pts


def display_results(samples: List[Dict]):
    categories = sorted(list(set(s.get("category", "unknown") for s in samples)))

    print("\n" + "=" * 90)
    print("3D F-SCORE BENCHMARK PER PIX3D SHAPE / CATEGORY")
    print("=" * 90)
    print(f"{'Category':<12} | {'Samples':<7} | {'Mean CD':<9} | {'F1@0.02':<9} | {'F1@0.05':<9} | {'F1@0.10':<9} | {'Prec@0.05':<10} | {'Rec@0.05':<10}")
    print("-" * 90)

    for cat in categories:
        cat_samples = [s for s in samples if s.get("category") == cat]
        cds = [s["chamfer_distance"] for s in cat_samples]
        f1_02 = np.mean([s.get("f1_02", 0.0) for s in cat_samples]) * 100
        f1_05 = np.mean([s.get("f1_05", 0.0) for s in cat_samples]) * 100
        f1_10 = np.mean([s.get("f1_10", 0.0) for s in cat_samples]) * 100
        p_05 = np.mean([s.get("precision_05", 0.0) for s in cat_samples]) * 100
        r_05 = np.mean([s.get("recall_05", 0.0) for s in cat_samples]) * 100
        print(f"{cat:<12} | {len(cat_samples):<7} | {np.mean(cds):<9.4f} | {f1_02:<8.1f}% | {f1_05:<8.1f}% | {f1_10:<8.1f}% | {p_05:<9.1f}% | {r_05:<9.1f}%")

    print("-" * 90)
    all_cds = [s["chamfer_distance"] for s in samples]
    tot_f1_02 = np.mean([s.get("f1_02", 0.0) for s in samples]) * 100
    tot_f1_05 = np.mean([s.get("f1_05", 0.0) for s in samples]) * 100
    tot_f1_10 = np.mean([s.get("f1_10", 0.0) for s in samples]) * 100
    tot_p_05 = np.mean([s.get("precision_05", 0.0) for s in samples]) * 100
    tot_r_05 = np.mean([s.get("recall_05", 0.0) for s in samples]) * 100
    print(f"{'OVERALL':<12} | {len(samples):<7} | {np.mean(all_cds):<9.4f} | {tot_f1_02:<8.1f}% | {tot_f1_05:<8.1f}% | {tot_f1_10:<8.1f}% | {tot_p_05:<9.1f}% | {tot_r_05:<9.1f}%")
    print("=" * 90 + "\n")

    print("OVERALL F-SCORE THRESHOLD SWEEP (Tatarchenko et al., CVPR 2019):")
    print("-" * 65)
    print(f"{'Threshold (tau)':<18} | {'Mean Precision':<15} | {'Mean Recall':<15} | {'Mean F1-Score':<15}")
    print("-" * 65)
    for t in THRESHOLDS:
        t_key = f"{int(t * 100):02d}"
        mp = np.mean([s.get(f"precision_{t_key}", 0.0) for s in samples]) * 100
        mr = np.mean([s.get(f"recall_{t_key}", 0.0) for s in samples]) * 100
        mf = np.mean([s.get(f"f1_{t_key}", 0.0) for s in samples]) * 100
        print(f"tau = {t:<12.2f} | {mp:13.2f}% | {mr:13.2f}% | {mf:13.2f}%")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate and compare F-Scores across Pix3D categories")
    parser.add_argument("--recompute", action="store_true", help="Recompute F-Scores directly from meshes")
    parser.add_argument("--results-json", type=Path, default=RESULTS_PATH, help="Path to results.json")
    parser.add_argument("--pix3d-root", type=Path, default=PIX3D_ROOT, help="Path to Pix3D dataset")
    args = parser.parse_args()

    if not args.results_json.exists():
        raise FileNotFoundError(f"Missing {args.results_json}. Run eval_pipeline.py first!")

    with open(args.results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        print("No samples found in results.json!")
        return

    has_fscores = "f1_05" in samples[0]
    if has_fscores and not args.recompute:
        print(f"Loaded {len(samples)} evaluated samples from {args.results_json}.")
        display_results(samples)
        return

    print(f"Computing F-Scores for {len(samples)} samples from CAD meshes...")
    for idx, s in enumerate(samples, start=1):
        pred_p = Path(s["pred_mesh_path"])
        if not pred_p.exists():
            pred_p = SCRIPT_DIR / "outputs" / "meshes" / pred_p.name
        gt_p = args.pix3d_root / s["gt_model_rel_path"]

        p_pts, g_pts = get_aligned_points(pred_p, gt_p, num_points=5000)
        d_p2g, _ = cKDTree(g_pts).query(p_pts, k=1)
        d_g2p, _ = cKDTree(p_pts).query(g_pts, k=1)

        for t in THRESHOLDS:
            t_key = f"{int(t * 100):02d}"
            prec = float(np.mean(d_p2g < t))
            rec = float(np.mean(d_g2p < t))
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            s[f"precision_{t_key}"] = prec
            s[f"recall_{t_key}"] = rec
            s[f"f1_{t_key}"] = f1

        if idx % 10 == 0 or idx == len(samples):
            print(f"Processed {idx}/{len(samples)} samples...")

    display_results(samples)


if __name__ == "__main__":
    main()
