"""Two-view geometry of a mirror-symmetric object in a single image (Hong et al. 2004).

Camera frame: x right, y down, z forward, camera centre at the origin. The object is
symmetric about the plane  n . X = d  with unit normal n. A surface point X and its mirror
partner

    X' = X - 2 (n . X - d) n

project to the pixels x and x'. Looking at X' with the real camera is the same as looking
at X with a second, mirrored camera, so (x, x') is a stereo correspondence. The mirrored
camera sits at 2 d n, and because the relative pose is a reflection the essential matrix
collapses to E = [n]_x. The epipolar constraint

    x'^T K^-T [n]_x K^-1 x = 0

says that x, x' and the vanishing point v = K n of the plane normal lie on one line. Two
pairs fix n; d only sets the global scale, so we use d = 1 (unit = distance between the
camera centre and the symmetry plane).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def pixel_rays(K: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Back-project pixels (N, 2) to viewing rays (N, 3) with z = 1."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(K).T


def reflect(X: np.ndarray, n: np.ndarray, d: float = 1.0) -> np.ndarray:
    """Mirror points (N, 3) at the plane n . X = d."""
    return X - 2.0 * (X @ n - d)[:, None] * n[None, :]


def epipolar_planes(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Unit normals of the planes spanned by the two rays of every pair."""
    m = np.cross(r1, r2)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def fit_normal(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Least-squares normal: n is perpendicular to the epipolar plane of every pair (two
    pairs are enough). It is the eigenvector of sum m m^T with the smallest eigenvalue."""
    m = epipolar_planes(r1, r2)
    return np.linalg.eigh(m.T @ m)[1][:, 0]


def epipolar_distance(K: np.ndarray, n: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Distance in pixels between a point and the line through its partner and the epipole
    v = K n (the larger of the two directions). n may be (3,) or (H, 3); returns (M,) or (H, M).
    """
    single = np.ndim(n) == 1
    v = np.atleast_2d(n) @ K.T
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    x1h = np.c_[x1, np.ones(len(x1))]
    x2h = np.c_[x2, np.ones(len(x2))]
    num = np.abs(v @ np.cross(x1h, x2h).T)          # |det[v, x1, x2]|

    def line_norm(x: np.ndarray) -> np.ndarray:      # length of the normal of the line v x x
        a = v[:, 1:2] - v[:, 2:3] * x[None, :, 1]
        b = v[:, 2:3] * x[None, :, 0] - v[:, 0:1]
        return np.hypot(a, b)

    dist = num / np.maximum(np.minimum(line_norm(x1), line_norm(x2)), 1e-12)
    return dist[0] if single else dist


def depth_ratio(n: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rays r1, r2 (z = 1) of mirror partners return (rho, den) with depth(X') = rho *
    depth(X) and depth(X) = 2 d / den. n may be (3,) or (H, 3)."""
    n = np.atleast_2d(n)
    a1, a2 = n @ r1.T, n @ r2.T
    # (r1 x n).(r2 x n) = r1.r2 - (r1.n)(r2.n), same for the squared norm of r2 x n
    r12 = np.einsum("ij,ij->i", r1, r2)[None, :]
    r22 = np.einsum("ij,ij->i", r2, r2)[None, :]
    rho = (r12 - a1 * a2) / np.maximum(r22 - a2 ** 2, 1e-12)
    return rho, a1 + rho * a2


def triangulate(K: np.ndarray, n: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                d: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D points of mirror partners. Returns (X, X', valid); X' is the exact mirror of X."""
    r1, r2 = pixel_rays(K, x1), pixel_rays(K, x2)
    rho, den = depth_ratio(n, r1, r2)
    rho, den = rho[0], den[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        z1 = 2.0 * d / den
    z2 = rho * z1
    valid = (rho > 0) & np.isfinite(z1) & (z1 > 0)
    z1 = np.where(valid, z1, 1.0)
    z2 = np.where(valid, z2, 1.0)
    X1, X2 = z1[:, None] * r1, z2[:, None] * r2
    X = 0.5 * (X1 + reflect(X2, n, d))               # average out the residual off the epipolar line
    return X, reflect(X, n, d), valid


# ---------------------------------------------------------------------------- robust fit

@dataclass
class PlaneFit:
    normal: np.ndarray       # unit normal of the symmetry plane n . X = 1, camera frame
    inliers: np.ndarray      # (M,) bool
    residual: np.ndarray     # (M,) epipolar distance in pixels
    support: int             # number of distinct epipolar lines that carry an inlier pair
    chance: float            # support that unrelated pairs would reach by chance
    found: bool              # support clearly above chance


def _distinct_lines(n: np.ndarray, m: np.ndarray, inl: np.ndarray, delta: float) -> np.ndarray:
    """Count the distinct epipolar planes among the inlier pairs of every hypothesis.

    Pairs that lie on one image line (matches sliding along a straight edge) agree with every
    epipole on that line, so they must not count more than once."""
    e = np.eye(3)[np.argmin(np.abs(n), axis=1)]
    b1 = np.cross(n, e)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = np.cross(n, b1)
    psi = np.arctan2(b2 @ m.T, b1 @ m.T) % np.pi     # angle of each plane around the axis n
    bins = np.floor(psi / delta).astype(np.int32)
    bins[~inl] = -1
    bins.sort(axis=1)
    return (np.diff(bins, axis=1) != 0).sum(axis=1) + 1 - (bins[:, 0] == -1)


def line_support(K: np.ndarray, n: np.ndarray, x1: np.ndarray, x2: np.ndarray, bin_px: float = 4.0) -> int:
    """Number of distinct epipolar lines occupied by the given pairs (all taken as inliers)."""
    if len(x1) == 0:
        return 0
    m = epipolar_planes(pixel_rays(K, x1), pixel_rays(K, x2))
    return int(_distinct_lines(n[None, :], m, np.ones((1, len(m)), bool), bin_px / K[0, 0])[0])


def _score(K, n, x1, x2, r1, r2, m, thr, delta):
    res = epipolar_distance(K, n, x1, x2)
    rho, den = depth_ratio(n, r1, r2)
    ok = (res < thr) & (rho > 0)
    pos, neg = ok & (den > 0), ok & (den < 0)        # both points in front of the camera,
    flip = neg.sum(axis=1) > pos.sum(axis=1)         # for n or for -n
    inl = np.where(flip[:, None], neg, pos)
    return _distinct_lines(n, m, inl, delta), inl, res, flip


def chance_support(x1: np.ndarray, x2: np.ndarray, thr: float) -> float:
    """Expected number of unrelated pairs within thr of a fixed epipole: the joining line of
    a random pair has a random direction, it passes the test if that direction is within
    asin(thr / length) of the direction to the epipole."""
    sep = np.linalg.norm(np.asarray(x1) - np.asarray(x2), axis=1)
    return float(np.sum(2.0 / np.pi * np.arcsin(np.minimum(1.0, thr / np.maximum(sep, 1e-9)))))


def refine_normal(K: np.ndarray, n: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                  thr: float) -> np.ndarray:
    """Minimise the epipolar distances over the two degrees of freedom of n."""
    u, _, _ = np.linalg.svd(n.reshape(3, 1))
    basis = u[:, 1:]                                 # tangent plane of the unit sphere at n

    def fun(p: np.ndarray) -> np.ndarray:
        q = n + basis @ p
        return epipolar_distance(K, q / np.linalg.norm(q), x1, x2)

    sol = least_squares(fun, np.zeros(2), loss="soft_l1", f_scale=thr, x_scale=1e-2)
    q = n + basis @ sol.x
    return q / np.linalg.norm(q)


def ransac_plane(K: np.ndarray, x1: np.ndarray, x2: np.ndarray, thr: float = 2.0,
                 n_hyp: int = 3000, bin_px: float = 4.0, min_support: int = 10,
                 seed: int = 0) -> PlaneFit:
    """Symmetry-plane normal from candidate mirror pairs x1[i] <-> x2[i] (pixels).

    Minimal sample: two pairs, n is perpendicular to both epipolar planes. A hypothesis is
    scored by the number of distinct epipolar lines with an inlier pair on them."""
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 2)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 2)
    M = len(x1)
    empty = PlaneFit(np.array([1.0, 0.0, 0.0]), np.zeros(M, bool), np.full(M, np.inf), 0, 0.0, False)
    if M < 2:
        return empty
    r1, r2 = pixel_rays(K, x1), pixel_rays(K, x2)
    m = epipolar_planes(r1, r2)
    delta = bin_px / K[0, 0]

    rng = np.random.default_rng(seed)
    if M * (M - 1) // 2 <= n_hyp:
        ia, ib = np.triu_indices(M, 1)
    else:
        ia = rng.integers(0, M, n_hyp)
        ib = (ia + rng.integers(1, M, n_hyp)) % M
    cand = np.cross(m[ia], m[ib])
    norm = np.linalg.norm(cand, axis=1)
    keep = norm > np.sin(np.radians(0.5))            # the two epipolar planes must differ
    cand = cand[keep] / norm[keep, None]
    if len(cand) == 0:
        return empty

    best_n, best_key = None, (-1, -1)
    for lo in range(0, len(cand), 256):
        chunk = cand[lo:lo + 256]
        sup, inl, _, flip = _score(K, chunk, x1, x2, r1, r2, m, thr, delta)
        count = inl.sum(axis=1)
        j = int(np.lexsort((count, sup))[-1])
        if (sup[j], count[j]) > best_key:
            best_key = (int(sup[j]), int(count[j]))
            best_n = -chunk[j] if flip[j] else chunk[j]

    # local optimisation: least squares on the inliers, then non-linear refinement
    n = best_n
    for _ in range(3):
        sup, inl, _, flip = _score(K, n[None], x1, x2, r1, r2, m, thr, delta)
        if flip[0]:
            n = -n
        if inl[0].sum() < 3:
            break
        cand_n = fit_normal(r1[inl[0]], r2[inl[0]])
        cand_n = cand_n if cand_n @ n > 0 else -cand_n
        cand_n = refine_normal(K, cand_n, x1[inl[0]], x2[inl[0]], thr)
        sup2, inl2, _, _ = _score(K, cand_n[None], x1, x2, r1, r2, m, thr, delta)
        if (sup2[0], inl2[0].sum()) < (sup[0], inl[0].sum()):
            break
        n = cand_n

    sup, inl, res, flip = _score(K, n[None], x1, x2, r1, r2, m, thr, delta)
    if flip[0]:
        n = -n
    chance = chance_support(x1, x2, thr)
    needed = max(min_support, chance + 4.5 * np.sqrt(chance) + 3.0)
    return PlaneFit(n, inl[0], res[0], int(sup[0]), chance, bool(sup[0] >= needed))
