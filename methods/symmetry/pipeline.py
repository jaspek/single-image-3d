"""Single-image reconstruction of a mirror-symmetric object: image, mask, K -> 3D points.

1. crop the object, bring it to a working resolution, replace the background by grey
2. keypoints + descriptors, match every feature against the reflected descriptors
3. RANSAC for the normal n of the symmetry plane (two pairs per sample)
4. guided matching along the epipolar lines, refine n, triangulate the pairs
5. dense search along the epipolar lines; silhouette intersection with the mirrored view

All points are in the camera frame (x right, y down, z forward). The scale is fixed by
placing the symmetry plane at distance 1 from the camera centre.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from methods.symmetry import dense as dense_mod
from methods.symmetry import features as feat_mod
from methods.symmetry import geometry as geo


@dataclass
class Params:
    work_size: int = 640          # longer side of the object crop after resizing, pixels
    max_upscale: float = 3.0
    contrast: float = 0.008       # DoG contrast threshold (image in [0, 1])
    max_keypoints: int = 2500
    ratio: float = 0.85           # Lowe ratio for the mirrored matching
    ori_tol_deg: float = 40.0     # tolerance of the mirrored gradient direction
    thr: float = 2.0              # epipolar inlier threshold, pixels at working resolution
    n_hyp: int = 3000
    guided: bool = True
    dense: bool = True            # dense search along the epipolar lines
    hull: bool = True             # silhouette intersection
    dense_step: int = 3           # pixel grid of the dense search
    dense_max_pixels: int = 12000 # the grid is coarsened for large objects to stay below this
    dense_normals: int = 6        # surface orientations tried by the dense search
    min_ncc: float = 0.8
    min_dense_ratio: float = 0.3  # share of textured pixels with a dense match that confirms a plane
    seed: int = 0


@dataclass
class Result:
    status: str                              # ok | no_object | few_features | no_symmetry
    normal: np.ndarray | None = None         # symmetry plane n . X = 1 in the camera frame
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # matched points + mirror copies
    hull: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))     # silhouette intersection surface
    info: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


def to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32) / 255.0
    return (image[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)) / 255.0


def crop_and_resize(gray: np.ndarray, mask: np.ndarray, K: np.ndarray, work_size: int,
                    max_upscale: float = 3.0, margin: float = 0.06):
    """Cut out the object and scale it so its longer side is `work_size`. Returns the grey
    crop, the mask crop and the intrinsics of the new image (same camera, same rays)."""
    ys, xs = np.nonzero(mask)
    pad = int(round(margin * max(np.ptp(ys), np.ptp(xs)))) + 2
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    h, w = y1 - y0, x1 - x0
    s = min(work_size / max(h, w), max_upscale)
    nh, nw = max(int(round(h * s)), 16), max(int(round(w * s)), 16)
    resample = Image.LANCZOS if s < 1 else Image.BICUBIC
    g = np.asarray(Image.fromarray(np.ascontiguousarray(gray[y0:y1, x0:x1])).resize((nw, nh), resample))
    m = np.asarray(Image.fromarray(mask[y0:y1, x0:x1].astype(np.float32)).resize((nw, nh), Image.BILINEAR)) > 0.5
    sx, sy = nw / w, nh / h
    A = np.array([[sx, 0.0, sx * (0.5 - x0) - 0.5], [0.0, sy, sy * (0.5 - y0) - 0.5], [0.0, 0.0, 1.0]])
    return np.clip(g, 0.0, 1.0), m, A @ K, A


def guided_matches(f: feat_mod.Features, K: np.ndarray, n: np.ndarray, thr: float, min_sep: float,
                   max_dist: float = 0.85, ratio: float = 0.9, ori_tol_deg: float | None = 40.0,
                   scale_tol: float = 1.0, exclusion: float = 6.0) -> np.ndarray:
    """Mirror partners searched only among the features on the epipolar line of each feature.
    The line removes most wrong candidates, so the descriptor test can be looser. A pair is
    kept if both features choose each other."""
    N = len(f)
    xh = np.c_[f.xy, np.ones(N)]
    lines = np.cross(K @ n, xh)                                    # line through the epipole and x_i
    lines /= np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-12)
    ok = np.abs(xh @ lines.T).astype(np.float32) < thr             # [j, i]: x_j close to the line of x_i
    ok &= ok.T
    r = geo.pixel_rays(K, f.xy).astype(np.float32)
    a = r @ n.astype(np.float32)
    rho = (r @ r.T - a[:, None] * a[None, :]) / np.maximum((r * r).sum(1)[None, :] - a[None, :] ** 2, 1e-12)
    ok &= (rho > 0) & (a[:, None] + rho * a[None, :] > 0)          # [i, j]: i is X, j is X'
    px, py = f.xy[:, 0].astype(np.float32), f.xy[:, 1].astype(np.float32)
    dx, dy = px[None, :] - px[:, None], py[None, :] - py[:, None]
    ok &= dx ** 2 + dy ** 2 >= min_sep ** 2
    # the nearer partner is imaged larger: scale_i / scale_j should be depth_j / depth_i = rho
    sr = (f.scale[:, None] / f.scale[None, :]).astype(np.float32)
    ok &= np.abs(np.log2(sr / np.where(rho > 0, rho, 1.0))) < scale_tol
    if ori_tol_deg is not None:
        ang = f.angle.astype(np.float32)
        dev = (ang[:, None] + ang[None, :] - 2 * np.arctan2(dy, dx)) % (2 * np.pi)   # should be pi
        ok &= np.abs(dev - np.pi) <= np.radians(ori_tol_deg)
    sim = np.where(ok, f.desc @ feat_mod.mirror_descriptors(f.desc).T, -1.0)
    best = sim.argmax(axis=1)
    s1 = sim[np.arange(N), best]
    near = (px[None, :] - px[best][:, None]) ** 2 + (py[None, :] - py[best][:, None]) ** 2 < exclusion ** 2
    s2 = np.where(near, -1.0, sim).max(axis=1)
    d1, d2 = np.sqrt(np.maximum(2 - 2 * s1, 0)), np.sqrt(np.maximum(2 - 2 * s2, 0))
    keep = (s1 > -1) & (d1 < max_dist) & (d1 < ratio * d2)
    keep &= f.loc[best[best]] == f.loc                             # mutual choice (by location)
    i, j = np.nonzero(keep)[0], best[keep]
    a_, b_ = np.minimum(f.loc[i], f.loc[j]), np.maximum(f.loc[i], f.loc[j])
    _, first = np.unique(a_ * (f.loc.max() + 1) + b_, return_index=True)
    return np.c_[i[first], j[first]]


def main_box(X: np.ndarray, n: np.ndarray, q: float = 0.02, margin: float = 0.35):
    """Box around the points X and their mirror copies (2 % .. 98 % quantiles), enlarged on
    every side by `margin` times its longest edge. Called with the feature pairs, which are
    the most reliable points: a dense match far outside their box is a gross outlier."""
    both = np.r_[X, geo.reflect(X, n)]
    lo, hi = np.quantile(both, q, axis=0), np.quantile(both, 1.0 - q, axis=0)
    pad = margin * float((hi - lo).max())
    return lo - pad, hi + pad


def in_box(X: np.ndarray, n: np.ndarray, box) -> np.ndarray:
    """True where a point and its mirror image are both inside the box."""
    lo, hi = box
    Xm = geo.reflect(X, n)
    return ((X >= lo) & (X <= hi)).all(axis=1) & ((Xm >= lo) & (Xm <= hi)).all(axis=1)


def rotate_about_axis(n: np.ndarray, deg: float) -> np.ndarray:
    """Turn n about the optical axis: the epipolar lines turn by the same angle in the image."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * n[0] - s * n[1], s * n[0] + c * n[1], n[2]])


def reconstruct(image: np.ndarray, mask: np.ndarray, K: np.ndarray, params: Params | None = None) -> Result:
    p = params or Params()
    t0 = time.time()
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() < 400:
        return Result("no_object", info={"seconds": 0.0})
    gray, m, Kw, A = crop_and_resize(to_gray(image), mask, np.asarray(K, float), p.work_size, p.max_upscale)
    soft = ndi.gaussian_filter(m.astype(np.float32), 1.0)
    gray = gray * soft + 0.5 * (1.0 - soft)                        # background does not take part
    ys, xs = np.nonzero(m)
    min_sep = 0.03 * float(np.hypot(np.ptp(ys), np.ptp(xs)))

    f = feat_mod.detect_and_describe(gray, ndi.binary_dilation(m, iterations=2), contrast=p.contrast,
                                     max_keypoints=p.max_keypoints)
    pairs, _ = feat_mod.mirror_matches(f, min_sep, ratio=p.ratio, ori_tol_deg=p.ori_tol_deg)
    info = {"n_keypoints": len(f), "n_matches": len(pairs), "n_inliers": 0, "support": 0, "chance": 0.0}
    debug = {"gray": gray, "mask": m, "K": Kw, "A": A, "features": f, "pairs": pairs,
             "inliers": np.zeros(len(pairs), bool)}

    def stop(status: str, normal=None) -> Result:
        info["seconds"] = round(time.time() - t0, 2)
        return Result(status, normal=normal, info=info, debug=debug)

    if len(pairs) < 4:
        return stop("few_features")
    fit = geo.ransac_plane(Kw, f.xy[pairs[:, 0]], f.xy[pairs[:, 1]], thr=p.thr, n_hyp=p.n_hyp, seed=p.seed)
    info.update(n_inliers=int(fit.inliers.sum()), support=fit.support, chance=round(fit.chance, 2))
    debug["inliers"] = fit.inliers
    # A plane with strong feature support is accepted as it is; one with weak support has to
    # be confirmed by the dense search below.
    if fit.support < max(6.0, fit.chance + 3.0 * np.sqrt(fit.chance) + 2.0):
        return stop("no_symmetry", fit.normal)

    n = fit.normal
    use = pairs[fit.inliers]
    if p.guided:
        for _ in range(2):
            g = guided_matches(f, Kw, n, p.thr, min_sep, ori_tol_deg=p.ori_tol_deg)
            if len(g) <= len(use):
                break
            n_new = geo.refine_normal(Kw, n, f.xy[g[:, 0]], f.xy[g[:, 1]], p.thr)
            use, n = g, (n_new if n_new @ n > 0 else -n_new)
        info["n_guided"] = len(use)
    debug["normal"] = n

    confirmed = fit.found
    step = max(p.dense_step, int(np.ceil(np.sqrt(m.sum() / p.dense_max_pixels))))
    normals = dense_mod.plane_normals(n, p.dense_normals)
    if p.dense:
        def quick_scan(normal: np.ndarray):
            """Dense search on a coarse grid; share of the textured pixels that find a partner."""
            c = dense_mod.epipolar_scan(gray, m, Kw, normal, step=4 * step,
                                        normals=dense_mod.plane_normals(normal, p.dense_normals),
                                        min_ncc=p.min_ncc, despeckle=False, hull=False)
            return c, len(c.points) / max(c.n_textured, 1)

        # Does the image match its mirror view at all, and which surface orientations occur?
        coarse, ratio = quick_scan(n)
        info["dense_ratio"] = round(ratio, 3)
        if not confirmed and coarse.n_textured >= 30 and ratio >= p.min_dense_ratio:
            # Smooth texture finds "partners" for any plane. Measure that level with two
            # planes that are certainly wrong and ask for clearly more.
            null = max(quick_scan(rotate_about_axis(n, deg))[1] for deg in (25.0, -25.0))
            info["dense_null"] = round(null, 3)
            confirmed = ratio >= 1.5 * null + 0.05
        share = np.bincount(coarse.normal_index, minlength=len(normals)) / max(len(coarse.points), 1)
        keep = share >= 0.1
        keep[0] |= not keep.any()
        normals = normals[keep]
        info["n_normals"] = int(keep.sum())
    if not confirmed:
        return stop("no_symmetry", n)

    X, _, valid = geo.triangulate(Kw, n, f.xy[use[:, 0]], f.xy[use[:, 1]])
    valid &= geo.epipolar_distance(Kw, n, f.xy[use[:, 0]], f.xy[use[:, 1]]) < p.thr
    use, X = use[valid], X[valid]
    half = [X]
    hull = np.zeros((0, 3))
    if p.dense or p.hull:
        d = dense_mod.epipolar_scan(gray, m, Kw, n, step=step, normals=normals, min_ncc=p.min_ncc,
                                    match=p.dense, hull=p.hull)
        if p.dense:
            half.append(d.points)
        debug["dense"] = d
    half = np.concatenate(half)
    keep = np.ones(len(half), bool)
    if len(half) >= 8:
        box = main_box(X if len(X) >= 8 else half, n)
        keep = in_box(half, n, box)
        if p.hull:
            # two silhouettes leave the shape open along the symmetry plane, the more the
            # smaller the view angle; cut it where the matched points end
            hull = d.hull[in_box(d.hull, n, box)]
            hull = np.r_[hull, geo.reflect(hull, n)]
    debug.update(used=use[keep[:len(X)]], sparse=X[keep[:len(X)]], kept=keep[len(X):])
    half = half[keep]
    points = np.r_[half, geo.reflect(half, n)]
    info.update(n_sparse=int(keep[:len(X)].sum()), n_dense=int(keep[len(X):].sum()),
                n_points=len(points), n_hull=len(hull))
    if len(points) == 0:
        return stop("no_symmetry", n)
    centre = points.mean(axis=0)                                   # on the plane: points come in mirror pairs
    info["view_angle_deg"] = round(float(np.degrees(np.arcsin(min(1.0, 1.0 / np.linalg.norm(centre))))), 1)
    info["seconds"] = round(time.time() - t0, 2)
    return Result("ok", n, points, hull, info, debug)
