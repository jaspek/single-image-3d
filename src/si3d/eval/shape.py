"""Shape metrics for Pix3D: Chamfer distance and F-score between a prediction (mesh or
point cloud) and the ground-truth CAD model, after normalisation and optional ICP.

Both shapes are centred on their bounding-box centre and scaled so the bounding-box
diagonal is 1, so distances are in units of "object size". Accuracy = mean distance
prediction -> ground truth, completeness = mean distance ground truth -> prediction;
a sparse prediction (symmetry method) is judged mainly by accuracy.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SHAPE_METRIC_NAMES = ["accuracy", "completeness", "chamfer", "fscore"]


def as_points(x: trimesh.Trimesh | np.ndarray, n: int = 10000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if isinstance(x, trimesh.Trimesh):
        try:
            pts, _ = trimesh.sample.sample_surface(x, n, seed=seed)
        except TypeError:  # older trimesh without seed argument
            pts, _ = trimesh.sample.sample_surface(x, n)
        return np.asarray(pts, dtype=np.float64)
    p = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] > n:
        p = p[rng.choice(p.shape[0], n, replace=False)]
    return p


def normalize_points(p: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, float]]:
    lo, hi = p.min(axis=0), p.max(axis=0)
    c = (lo + hi) / 2.0
    s = float(np.linalg.norm(hi - lo))
    s = s if s > 1e-12 else 1.0
    return (p - c) / s, (c, s)


def chamfer(p: np.ndarray, q: np.ndarray, tau: float = 0.05) -> dict[str, float]:
    d_pq = cKDTree(q).query(p)[0]
    d_qp = cKDTree(p).query(q)[0]
    prec = float(np.mean(d_pq < tau))
    rec = float(np.mean(d_qp < tau))
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "accuracy": float(np.mean(d_pq)),
        "completeness": float(np.mean(d_qp)),
        "chamfer": float(np.mean(d_pq) + np.mean(d_qp)),
        "fscore": f,
        "tau": tau,
    }


def _initial_rotations() -> list[np.ndarray]:
    inits = []
    for axis in ([0, 1, 0], [1, 0, 0]):
        for deg in (0, 90, 180, 270):
            inits.append(trimesh.transformations.rotation_matrix(np.radians(deg), axis))
    return inits


def icp_align(src: np.ndarray, dst: np.ndarray, scale: bool = True, max_iter: int = 50,
              multi_start: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Align src points onto dst points. Returns (transformed src, 4x4 matrix, cost)."""
    best = None
    inits = _initial_rotations() if multi_start else [np.eye(4)]
    for T0 in inits:
        M, moved, cost = trimesh.registration.icp(
            src, dst, initial=T0, max_iterations=max_iter, scale=scale, reflection=False)
        if best is None or cost < best[2]:
            best = (np.asarray(moved), np.asarray(M), float(cost))
    return best


def evaluate_shape(pred: trimesh.Trimesh | np.ndarray, gt: trimesh.Trimesh | np.ndarray,
                   n: int = 10000, tau: float = 0.05, align: str = "icp",
                   seed: int = 0) -> dict[str, float]:
    p = as_points(pred, n, seed)
    g = as_points(gt, n, seed)
    p, _ = normalize_points(p)
    g, _ = normalize_points(g)
    if align == "icp":
        p, _, _ = icp_align(p, g)
    elif align != "none":
        raise ValueError(f"unknown align {align!r}")
    out = chamfer(p, g, tau)
    out["n_pred"] = int(p.shape[0])
    return out
