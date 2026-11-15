"""Shape metrics for Pix3D: Chamfer distance and F-score between a prediction (mesh or
point cloud) and the ground-truth CAD model.

Distances are in units of the bounding-box diagonal of the ground truth ("object size").
Accuracy = mean distance prediction -> ground truth, completeness = mean distance ground
truth -> prediction, Chamfer = their sum, F-score = harmonic mean of the fractions of
points closer than tau in the two directions.

A single image fixes neither the scale nor, for methods with their own output frame, the
pose, so the prediction is aligned first:

  align="icp"     similarity transform with a search over rotations (own output frame)
  align="camera"  one scale factor about the camera centre (prediction and ground truth
                  both in the camera frame; position and orientation are judged as predicted)
  align="none"    bounding-box normalisation only
"""
from __future__ import annotations

import itertools

import numpy as np
import trimesh
from scipy.optimize import minimize_scalar
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


def normalize_points(p: np.ndarray, trim: float = 0.0) -> tuple[np.ndarray, tuple[np.ndarray, float]]:
    """Centre on the bounding box and scale its diagonal to 1. With trim > 0 the box spans
    the [trim, 1 - trim] quantiles per axis, so a few stray points do not set the scale."""
    if trim > 0:
        lo, hi = np.quantile(p, trim, axis=0), np.quantile(p, 1.0 - trim, axis=0)
    else:
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


def _symmetric_distance(p: np.ndarray, q: np.ndarray, q_tree: cKDTree) -> float:
    return float(q_tree.query(p)[0].mean() + cKDTree(p).query(q)[0].mean())


# ------------------------------------------------------------------ similarity alignment

def _initial_rotations() -> list[np.ndarray]:
    """The 24 rotations that map the coordinate axes onto each other, identity first."""
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            R[range(3), perm] = signs
            if np.linalg.det(R) > 0:
                rots.append(R)
    return rots


def _similarity(a: np.ndarray, b: np.ndarray, w: np.ndarray, with_scale: bool):
    """Weighted least-squares s, R with b ~ s R (a - mu_a) + mu_b (Umeyama 1991), no reflection."""
    w = w / w.sum()
    mu_a, mu_b = w @ a, w @ b
    a0, b0 = a - mu_a, b - mu_b
    U, S, Vt = np.linalg.svd((b0 * w[:, None]).T @ a0)
    D = np.array([1.0, 1.0, 1.0 if np.linalg.det(U @ Vt) > 0 else -1.0])
    s = float((S * D).sum() / max(w @ (a0 ** 2).sum(axis=1), 1e-12)) if with_scale else 1.0
    return s, (U * D) @ Vt, mu_a, mu_b


def _icp(src: np.ndarray, dst: np.ndarray, M0: np.ndarray, with_scale: bool, iters: int,
         scale_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    """ICP with nearest neighbours in both directions. With pred -> gt pairs only, a free
    scale lets the prediction shrink onto a small part of the ground truth; the gt -> pred
    pairs pull it back, so the cost is the same symmetric distance the metrics use."""
    tree = cKDTree(dst)
    M = M0.copy()
    cur = src @ M[:3, :3].T + M[:3, 3]
    total = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    prev = np.inf
    for _ in range(iters):
        d_fwd, j = tree.query(cur)
        d_bwd, i = cKDTree(cur).query(dst)
        cost = float(d_fwd.mean() + d_bwd.mean())
        if prev - cost < 1e-6:
            return cur, M, cost
        prev = cost
        # the 5 % worst pairs of each direction do not steer the fit (stray points, holes)
        f = d_fwd <= np.quantile(d_fwd, 0.95)
        b = d_bwd <= np.quantile(d_bwd, 0.95)
        w = np.r_[np.full(f.sum(), 1.0 / f.sum()), np.full(b.sum(), 1.0 / b.sum())]
        s, R, mu_a, mu_b = _similarity(np.r_[cur[f], cur[i[b]]], np.r_[dst[j[f]], dst[b]], w, with_scale)
        s = float(np.clip(total * s, *scale_range)) / total
        total *= s
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = s * R, mu_b - s * R @ mu_a
        M = step @ M
        cur = src @ M[:3, :3].T + M[:3, 3]
    return cur, M, _symmetric_distance(cur, dst, tree)


def icp_align(src: np.ndarray, dst: np.ndarray, scale: bool = True, max_iter: int = 50,
              multi_start: bool = True, scale_range: tuple[float, float] = (0.5, 2.0),
              seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Align src points onto dst points with a similarity transform (rigid if scale=False).
    Returns (transformed src, 4x4 matrix, cost), cost = mean distance in both directions.

    multi_start tries the 24 axis-aligned rotations on a subsample first, because a method
    may reconstruct in its own frame, and refines the three best with all points."""
    if multi_start:
        rng = np.random.default_rng(seed)
        a = src[rng.choice(len(src), min(len(src), 1500), replace=False)]
        b = dst[rng.choice(len(dst), min(len(dst), 1500), replace=False)]
        coarse = []
        for R0 in _initial_rotations():
            M0 = np.eye(4)
            M0[:3, :3] = R0
            _, M, cost = _icp(a, b, M0, scale, 15, scale_range)
            coarse.append((cost, M))
        coarse.sort(key=lambda c: c[0])
        starts = [M for _, M in coarse[:3]]
    else:
        starts = [np.eye(4)]
    best = None
    for M0 in starts:
        moved, M, cost = _icp(src, dst, M0, scale, max_iter, scale_range)
        if best is None or cost < best[2]:
            best = (moved, M, cost)
    return best


# ------------------------------------------------------------------ camera-frame alignment

def camera_scale(src: np.ndarray, dst: np.ndarray, span: float = 2.0) -> tuple[np.ndarray, float]:
    """Scale factor s that brings s * src closest to dst (symmetric mean distance).

    For a reconstruction in the camera frame of a single image the only thing that cannot be
    known is the overall scale: multiplying every point by s moves the object along the
    viewing rays and resizes it, and the image stays the same. Returns (s * src, s)."""
    tree = cKDTree(dst)
    s0 = np.median(np.linalg.norm(dst, axis=1)) / max(np.median(np.linalg.norm(src, axis=1)), 1e-12)
    logs = np.log(s0) + np.linspace(-np.log(span), np.log(span), 25)
    costs = [_symmetric_distance(src * np.exp(v), dst, tree) for v in logs]
    k = int(np.argmin(costs))
    lo, hi = logs[max(k - 1, 0)], logs[min(k + 1, len(logs) - 1)]
    res = minimize_scalar(lambda v: _symmetric_distance(src * np.exp(v), dst, tree),
                          bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    s = float(np.exp(res.x if res.fun <= costs[k] else logs[k]))
    return src * s, s


def evaluate_shape(pred: trimesh.Trimesh | np.ndarray, gt: trimesh.Trimesh | np.ndarray,
                   n: int = 10000, tau: float = 0.05, align: str = "icp",
                   seed: int = 0) -> dict[str, float]:
    """Chamfer distance and F-score of a prediction, see the module docstring for `align`."""
    p = as_points(pred, n, seed)
    g = as_points(gt, n, seed)
    if align == "camera":
        p, _ = camera_scale(p, g)
        g, (centre, size) = normalize_points(g)
        p = (p - centre) / size
    elif align in ("icp", "none"):
        # The prediction's own box is only the starting point of the alignment. For a point
        # cloud it spans the 0.5 % .. 99.5 % quantiles, so that single stray points do not
        # set the initial scale; they still count in the distances.
        p, _ = normalize_points(p, trim=0.0 if isinstance(pred, trimesh.Trimesh) else 0.005)
        g, _ = normalize_points(g)
        if align == "icp":
            p, _, _ = icp_align(p, g)
    else:
        raise ValueError(f"unknown align {align!r}")
    out = chamfer(p, g, tau)
    out["n_pred"] = int(p.shape[0])
    return out
