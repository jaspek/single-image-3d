"""Surface normals from depth and normal-error metrics (for shape from shading)."""
from __future__ import annotations

import numpy as np

NORMAL_METRIC_NAMES = ["mean_deg", "median_deg", "rmse_deg", "p11", "p22", "p30"]


def unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1)


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unit normals (H, W, 3) oriented towards the camera; NaN where depth is invalid."""
    P = unproject(depth.astype(np.float64), K)
    dx = np.gradient(P, axis=1)
    dy = np.gradient(P, axis=0)
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(norm, 1e-12, None)
    # orient so that the normal points against the viewing ray
    flip = np.sum(n * P, axis=-1, keepdims=True) > 0
    n = np.where(flip, -n, n)
    bad = ~np.isfinite(depth) | (depth <= 0) | (norm[..., 0] < 1e-12)
    n[bad] = np.nan
    return n


def normal_metrics(pred_n: np.ndarray, gt_n: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = pred_n[mask]
    g = gt_n[mask]
    ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(g), axis=1)
    p, g = p[ok], g[ok]
    if p.shape[0] == 0:
        return {k: float("nan") for k in NORMAL_METRIC_NAMES} | {"n_valid": 0}
    p = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
    g = g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-12, None)
    cos = np.clip(np.sum(p * g, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    return {
        "mean_deg": float(np.mean(ang)),
        "median_deg": float(np.median(ang)),
        "rmse_deg": float(np.sqrt(np.mean(ang ** 2))),
        "p11": float(np.mean(ang < 11.25)),
        "p22": float(np.mean(ang < 22.5)),
        "p30": float(np.mean(ang < 30.0)),
        "n_valid": int(ang.size),
    }
