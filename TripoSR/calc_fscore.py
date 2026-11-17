import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "outputs" / "results.json"
if not RESULTS_PATH.exists():
    RESULTS_PATH = SCRIPT_DIR.parent / "outputs" / "results.json"

data = json.load(open(RESULTS_PATH, "r", encoding="utf-8"))
samples = data["samples"]
pix3d_root = Path(r"D:\pix3d")

def normalize_point_cloud(pts):
    bound_l = np.min(pts, axis=0)
    bound_h = np.max(pts, axis=0)
    center = (bound_l + bound_h) / 2.0
    scale = float(np.max(bound_h - bound_l))
    if scale < 1e-12:
        scale = 1.0
    return (pts - center) / scale

def get_aligned_points(pred_path, gt_path, num_points=10000, seed=42):
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

thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
all_f1 = {t: [] for t in thresholds}
all_prec = {t: [] for t in thresholds}
all_rec = {t: [] for t in thresholds}

print(f"Calculating F-Scores for {len(samples)} samples...")
for idx, s in enumerate(samples, 1):
    pred_p = Path(s["pred_mesh_path"])
    if not pred_p.exists():
        pred_p = SCRIPT_DIR / "outputs" / "meshes" / pred_p.name
    gt_p = pix3d_root / s["gt_model_rel_path"]
    p_pts, g_pts = get_aligned_points(pred_p, gt_p, num_points=5000)

    d_p2g, _ = cKDTree(g_pts).query(p_pts, k=1)
    d_g2p, _ = cKDTree(p_pts).query(g_pts, k=1)

    for t in thresholds:
        prec = float(np.mean(d_p2g < t))
        rec = float(np.mean(d_g2p < t))
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        all_prec[t].append(prec)
        all_rec[t].append(rec)
        all_f1[t].append(f1)

print("\n" + "="*60)
print(f"{'Threshold (tau)':<18} | {'Mean Precision':<15} | {'Mean Recall':<15} | {'Mean F1-Score':<15}")
print("-"*60)
for t in thresholds:
    print(f"tau = {t:<12.2f} | {np.mean(all_prec[t])*100:13.2f}% | {np.mean(all_rec[t])*100:13.2f}% | {np.mean(all_f1[t])*100:13.2f}%")
print("="*60)

# Check best and worst sample F1@0.05 and F1@0.10
best_idx = 1492
worst_idx = 1489
print(f"\nF1@0.05 for Top Best (#1492):")
for s, f1_05 in zip(samples, all_f1[0.05]):
    if s["pix3d_id"] in [1492, 1478, 1439, 1365, 1489]:
        print(f"Sample #{s['pix3d_id']}: CD={s['chamfer_distance']:.4f}, F1@0.05={f1_05*100:.2f}%")
