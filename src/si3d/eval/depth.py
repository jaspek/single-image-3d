"""Monocular depth metrics (the usual NYU / KITTI set).

pred and gt are float arrays of the same shape, depth in metres. `mask` selects the
pixels that count. Relative-depth outputs (Depth Anything V2 relative, shape from shading)
have no absolute scale: align them first with `align_pred`.
"""
from __future__ import annotations

import numpy as np

METRIC_NAMES = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "d1", "d2", "d3"]


def eval_mask(gt: np.ndarray, min_depth: float = 1e-3, max_depth: float = 10.0,
              crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    m = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    if crop is not None:
        r0, r1, c0, c1 = crop
        cm = np.zeros_like(m)
        cm[r0:r1, c0:c1] = True
        m &= cm
    return m


def align_pred(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, mode: str = "none",
               space: str = "depth") -> np.ndarray:
    """Return pred rescaled to gt.

    mode:  none     pred is already metric
           median   one global scale, median(gt) / median(pred)
           affine   least-squares scale and shift
    space: depth    pred is (relative) depth
           inverse  pred is relative inverse depth / disparity (Depth Anything relative);
                    the fit is done in inverse-depth space and the result inverted
    """
    pred = pred.astype(np.float64)
    if mode == "none":
        return pred
    p = pred[mask]
    g = gt[mask].astype(np.float64)
    if space == "inverse":
        g = 1.0 / np.clip(g, 1e-6, None)
    if mode == "median":
        s = np.median(g) / max(np.median(p), 1e-12)
        out = pred * s
    elif mode == "affine":
        A = np.stack([p, np.ones_like(p)], axis=1)
        a, b = np.linalg.lstsq(A, g, rcond=None)[0]
        out = a * pred + b
    else:
        raise ValueError(f"unknown align mode {mode!r}")
    if space == "inverse":
        out = 1.0 / np.clip(out, 1e-6, None)
    return out


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                  min_depth: float = 1e-3, max_depth: float = 10.0) -> dict[str, float]:
    p = np.clip(pred[mask].astype(np.float64), min_depth, max_depth)
    g = gt[mask].astype(np.float64)
    if g.size == 0:
        return {k: float("nan") for k in METRIC_NAMES} | {"n_valid": 0}
    thresh = np.maximum(p / g, g / p)
    err = g - p
    return {
        "abs_rel": float(np.mean(np.abs(err) / g)),
        "sq_rel": float(np.mean(err ** 2 / g)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2))),
        "log10": float(np.mean(np.abs(np.log10(g) - np.log10(p)))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25 ** 2)),
        "d3": float(np.mean(thresh < 1.25 ** 3)),
        "n_valid": int(g.size),
    }
