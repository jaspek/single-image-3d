"""Straight edges of the object and their vanishing directions.

Edges that run across the width of a mirror-symmetric object are parallel to the normal n
of the symmetry plane. Their images meet in the vanishing point v = K n, the same point
through which all lines between mirror partners pass (geometry.py). The normal can
therefore be read from the straight edges alone, without matching any feature.

1. line segments: image gradient -> regions of connected pixels with a similar gradient
   direction (Burns, Hanson, Riseman 1986) -> a line fitted to the edge points of a region
2. vanishing directions: a segment with the end points a, b and a direction d belong
   together if d lies in the plane of the two viewing rays, d . m = 0 with m = r_a x r_b.
   This is the constraint of a mirror pair in geometry.py, so the search is the same:
   two segments give a hypothesis, the segments that agree with it are counted, and the
   directions are taken out one after the other.
3. three perpendicular axes: furniture is mostly box-like, its edges run along its width,
   depth and height. Fitting the three axes together is more accurate than three separate
   directions and tells whether the object is box-like at all.
4. the normal: the horizontal axis that faces the camera least (plane_normal), or no answer.

Everything is in the camera frame of geometry.py (x right, y down, z forward).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from methods.symmetry import geometry as geo


@dataclass
class Segments:
    a: np.ndarray            # (N, 2) first end point, pixels (x, y)
    b: np.ndarray            # (N, 2) second end point
    length: np.ndarray       # (N,) pixels
    rms: np.ndarray          # (N,) distance of the edge points from the fitted line, pixels
    curved: float = 0.0      # total length of the smooth edges that are not straight, pixels

    def __len__(self) -> int:
        return len(self.a)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.a + self.b)

    def __getitem__(self, keep) -> "Segments":
        return Segments(self.a[keep], self.b[keep], self.length[keep], self.rms[keep], self.curved)


def _empty() -> Segments:
    return Segments(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0))


def edge_points(gray: np.ndarray, sigma: float, low: float, region: np.ndarray | None = None):
    """Gradient of the smoothed image and its ridge points: pixels above `low` whose gradient
    magnitude is a maximum along the gradient direction, moved to the maximum of a parabola
    through the three samples. Returns (gx, gy, strong mask, edge x, edge y, magnitude, flat
    pixel index of every edge point)."""
    g = np.asarray(gray, dtype=np.float32)
    gx = ndi.gaussian_filter(g, sigma, order=(0, 1))
    gy = ndi.gaussian_filter(g, sigma, order=(1, 0))
    mag = np.hypot(gx, gy)
    strong = mag >= low
    if region is not None:
        strong &= region
    ys, xs = np.nonzero(strong)
    m0 = mag[ys, xs]
    ux, uy = gx[ys, xs] / m0, gy[ys, xs] / m0
    ahead = ndi.map_coordinates(mag, [ys + uy, xs + ux], order=1, mode="nearest")
    behind = ndi.map_coordinates(mag, [ys - uy, xs - ux], order=1, mode="nearest")
    peak = (m0 >= ahead) & (m0 > behind)
    curv = behind - 2.0 * m0 + ahead
    off = np.clip(0.5 * (behind - ahead) / np.minimum(curv, -1e-9), -0.5, 0.5)
    ex, ey = (xs + off * ux)[peak], (ys + off * uy)[peak]
    return gx, gy, strong, ex, ey, m0[peak], (ys * g.shape[1] + xs)[peak]


def _split(x: np.ndarray, y: np.ndarray, w: np.ndarray, ids: np.ndarray, min_length: float, max_rms: float,
           min_fill: float, depth: int, out: list) -> None:
    """Fit a line to the edge points; if they are not straight, cut them at the point farthest
    from the line through their two ends and try both parts again (`depth` times at most).
    Appends (end a, end b, length, rms, point ids) of every straight part to `out`."""
    if len(x) < 2:
        return
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    dx, dy = x - mx, y - my
    cxx, cxy, cyy = (w * dx * dx).sum() / sw, (w * dx * dy).sum() / sw, (w * dy * dy).sum() / sw
    phi = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
    rms = float(np.sqrt(max(0.5 * (cxx + cyy - np.hypot(cxx - cyy, 2.0 * cxy)), 0.0)))
    c, s = np.cos(phi), np.sin(phi)
    t = dx * c + dy * s
    i0, i1 = int(np.argmin(t)), int(np.argmax(t))
    length = float(t[i1] - t[i0])
    if length < min_length or len(x) < min_fill * length:
        return
    if rms <= max_rms:
        out.append(((mx + t[i0] * c, my + t[i0] * s), (mx + t[i1] * c, my + t[i1] * s), length, rms, ids))
        return
    if depth == 0:
        return
    ux, uy = x[i1] - x[i0], y[i1] - y[i0]
    dist = np.abs((x - x[i0]) * uy - (y - y[i0]) * ux)
    cut = t[int(np.argmax(dist))]
    for part in (t <= cut, t >= cut):
        _split(x[part], y[part], w[part], ids[part], min_length, max_rms, min_fill, depth - 1, out)


def detect_segments(gray: np.ndarray, region: np.ndarray | None = None, sigma: float = 1.2,
                    low: float = 0.012, high: float = 0.03, n_bins: int = 12, min_length: float = 20.0,
                    max_rms: float = 0.5, min_fill: float = 0.5, max_bend: float = 0.03) -> Segments:
    """Straight line segments of a grey image in [0, 1], inside `region` if given.

    The gradient direction is quantised into `n_bins` sectors; connected pixels of one
    sector form a region, and the edge points inside it give one line (principal axis).
    Two quantisations shifted by half a sector are used, so that an edge on a sector border
    of one of them is whole in the other; every edge point votes for the longer of its two
    lines and a line is kept if most of its points voted for it.

    A line is valid if it is at least `min_length` long, its edge points stay within
    `max_rms` of it (rejects curves and texture), they cover at least the share `min_fill`
    of its length, and the gradient reaches `high` somewhere on it. Two edges that meet at
    a shallow angle share a sector and come out as one bent region; such a region is cut at
    its corner and the parts are fitted on their own.

    What then still fails only the straightness test, but stays within `max_bend` times its
    length of the line, is a smooth curve (an arc inside one sector bends by 0.02 of its
    length). Curves are not returned, but their total length is, as `curved`: it tells how
    much of the outline of the object is not straight."""
    gx, gy, strong, ex, ey, ew, eidx = edge_points(gray, sigma, low, region)
    if len(ex) < 2:
        return _empty()
    theta = np.arctan2(gy, gx)
    eight = np.ones((3, 3), bool)

    fits = []
    for shift in (0.0, 0.5):
        sector = np.floor(theta / (2.0 * np.pi) * n_bins + shift).astype(np.int32) % n_bins
        lab = np.zeros(theta.shape, np.int32)
        total = 0
        for k in range(n_bins):
            part, count = ndi.label(strong & (sector == k), structure=eight)
            inside = part > 0
            lab[inside] = part[inside] + total
            total += count
        comp = lab.ravel()[eidx]                              # region of every edge point (>= 1)
        size = total + 1
        idx = np.arange(size)
        cnt = np.bincount(comp, minlength=size)
        sw = np.maximum(np.bincount(comp, ew, size), 1e-12)
        mx, my = np.bincount(comp, ew * ex, size) / sw, np.bincount(comp, ew * ey, size) / sw
        dx, dy = ex - mx[comp], ey - my[comp]
        cxx = np.bincount(comp, ew * dx * dx, size) / sw
        cxy = np.bincount(comp, ew * dx * dy, size) / sw
        cyy = np.bincount(comp, ew * dy * dy, size) / sw
        phi = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)          # direction of the principal axis
        across = 0.5 * (cxx + cyy - np.hypot(cxx - cyy, 2.0 * cxy))
        rms = np.sqrt(np.maximum(across, 0.0))
        c, s = np.cos(phi), np.sin(phi)
        t = dx * c[comp] + dy * s[comp]
        t0, t1 = ndi.minimum(t, comp, idx), ndi.maximum(t, comp, idx)
        length = t1 - t0
        edge = (cnt >= 2) & (length >= min_length) & (cnt >= min_fill * length)
        edge &= ndi.maximum(ew, comp, idx) >= high
        straight = edge & (rms <= max_rms)
        a = np.c_[mx + t0 * c, my + t0 * s][straight]
        b = np.c_[mx + t1 * c, my + t1 * s][straight]
        seg_len, seg_rms = length[straight], rms[straight]
        number = np.full(size, -1)
        number[straight] = np.arange(int(straight.sum()))
        sid = number[comp]                                    # line of every edge point, -1: none
        bent = np.nonzero(edge & ~straight)[0]
        if len(bent):
            order = np.argsort(comp, kind="stable")
            lo, hi = np.searchsorted(comp[order], bent), np.searchsorted(comp[order], bent, side="right")
            parts: list = []
            for r0, r1 in zip(lo, hi):
                ids = order[r0:r1]
                _split(ex[ids], ey[ids], ew[ids], ids, min_length, max_rms, min_fill, 2, parts)
            if parts:
                first_new = len(seg_len)
                a = np.r_[a, np.array([q[0] for q in parts])]
                b = np.r_[b, np.array([q[1] for q in parts])]
                seg_len = np.r_[seg_len, [q[2] for q in parts]]
                seg_rms = np.r_[seg_rms, [q[3] for q in parts]]
                for j, q in enumerate(parts):
                    sid[q[4]] = first_new + j
        # an edge point of a smooth curve stands for its share of the length of the curve
        share = np.where(edge & ~straight & (rms <= max_bend * length), length / np.maximum(cnt, 1), 0.0)[comp]
        fits.append((sid, a, b, seg_len, seg_rms, share))

    (s1, a1, b1, l1, r1, w1), (s2, a2, b2, l2, r2, w2) = fits
    on1, on2 = s1 >= 0, s2 >= 0
    first = np.r_[l1, 0.0][s1] >= np.r_[l2, 0.0][s2]          # the vote of every edge point (-1: no line)
    keep1 = np.bincount(s1[on1], first[on1], len(l1)) > 0.5 * np.bincount(s1[on1], minlength=len(l1))
    keep2 = np.bincount(s2[on2], ~first[on2], len(l2)) > 0.5 * np.bincount(s2[on2], minlength=len(l2))
    used = np.r_[keep1, False][s1] | np.r_[keep2, False][s2]
    curved = float(np.maximum(w1, w2)[~used].sum())
    seg = Segments(np.r_[a1[keep1], a2[keep2]], np.r_[b1[keep1], b2[keep2]],
                   np.r_[l1[keep1], l2[keep2]], np.r_[r1[keep1], r2[keep2]], curved)
    return seg[np.argsort(-seg.length)]


# --------------------------------------------------------------------- vanishing directions

@dataclass
class Direction:
    d: np.ndarray            # unit vector in the camera frame (the sign carries no meaning)
    inliers: np.ndarray      # (N,) bool over the segments
    lines: int               # distinct image lines among the inliers
    length: float            # total length of the inliers, pixels


def tolerance(length: np.ndarray, tol_deg: float, tol_px: float) -> np.ndarray:
    """How far an end point may lie from the line through the vanishing point and the middle
    of the segment: an angle of tol_deg, but not less than tol_px (short segments)."""
    return np.maximum(tol_px, 0.5 * length * np.sin(np.radians(tol_deg)))


def support(K: np.ndarray, d: np.ndarray, seg: Segments, tol: np.ndarray, bin_px: float = 4.0,
            free: np.ndarray | None = None):
    """Segments that point to the vanishing point of each direction in d (H, 3).
    Returns (inliers (H, N), distinct lines (H,), total inlier length (H,), distance (H, N))."""
    res = geo.epipolar_distance(K, d, seg.mid, seg.a)
    inl = res <= tol[None, :]
    if free is not None:
        inl &= free[None, :]
    m = geo.epipolar_planes(geo.pixel_rays(K, seg.a), geo.pixel_rays(K, seg.b))
    lines = geo._distinct_lines(d, m, inl, bin_px / K[0, 0])
    return inl, lines, (inl * seg.length[None, :]).sum(axis=1), res


def vanishing_directions(K: np.ndarray, seg: Segments, max_dirs: int = 4, tol_deg: float = 1.5,
                         tol_px: float = 2.0, n_hyp: int = 2000, min_lines: int = 3,
                         min_total: float = 0.0, bin_px: float = 4.0, seed: int = 0) -> list[Direction]:
    """Directions in space to which many segments point, strongest first.

    A hypothesis is the direction common to two segments, d = m_i x m_j. It is scored by the
    total length of the segments that agree with it and needs `min_lines` distinct lines of
    at least `min_total` pixels in total. The segments of an accepted direction are removed
    before the next one is searched."""
    N = len(seg)
    out: list[Direction] = []
    if N < min_lines:
        return out
    m = geo.epipolar_planes(geo.pixel_rays(K, seg.a), geo.pixel_rays(K, seg.b))
    tol = tolerance(seg.length, tol_deg, tol_px)
    rng = np.random.default_rng(seed)
    free = np.ones(N, bool)
    while len(out) < max_dirs and free.sum() >= min_lines:
        idx = np.nonzero(free)[0]
        M = len(idx)
        if M * (M - 1) // 2 <= n_hyp:
            ia, ib = np.triu_indices(M, 1)
        else:
            w = seg.length[idx] / seg.length[idx].sum()       # long segments are the reliable ones
            ia, ib = rng.choice(M, n_hyp, p=w), rng.choice(M, n_hyp, p=w)
        cand = np.cross(m[idx[ia]], m[idx[ib]])
        norm = np.linalg.norm(cand, axis=1)
        ok = norm > np.sin(np.radians(0.5))                   # two different lines
        cand = cand[ok] / norm[ok, None]
        best, best_score = None, 0.0
        for lo in range(0, len(cand), 256):
            chunk = cand[lo:lo + 256]
            _, lines, length, _ = support(K, chunk, seg, tol, bin_px, free)
            score = np.where(lines >= min_lines, length, 0.0)
            j = int(np.argmax(score))
            if score[j] > best_score:
                best, best_score = chunk[j], float(score[j])
        if best is None or best_score < min_total:
            break
        d = best
        for _ in range(3):                                    # least squares on the inliers
            inl, lines, length, _ = support(K, d[None], seg, tol, bin_px, free)
            w = seg.length[inl[0]] ** 2
            mi = m[inl[0]]
            d_new = np.linalg.eigh((mi * w[:, None]).T @ mi)[1][:, 0]
            _, lines_new, length_new, _ = support(K, d_new[None], seg, tol, bin_px, free)
            if lines_new[0] < min_lines or length_new[0] < length[0]:
                break
            d = d_new
        inl, lines, length, res = support(K, d[None], seg, tol, bin_px, free)
        out.append(Direction(d, inl[0], int(lines[0]), float(length[0])))
        free &= res[0] > 2.0 * tol                            # also drop the near misses
    return out


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two directions without regard to their sign."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


# ------------------------------------------------------------------ three orthogonal axes

@dataclass
class Frame:
    """Three mutually perpendicular directions (width, depth, height of a piece of furniture,
    in no particular order) with the segments that point to each of them."""
    axes: np.ndarray         # (3, 3), one unit direction per row
    inliers: np.ndarray      # (3, N) bool over the segments
    lines: np.ndarray        # (3,) distinct image lines per axis
    length: np.ndarray       # (3,) total length of the inliers per axis, pixels


def signed_distance(K: np.ndarray, d: np.ndarray, seg: Segments) -> np.ndarray:
    """Distance (with sign) of the first end point of every segment from the line through
    its middle and the vanishing point of d."""
    v = K @ d
    c = np.c_[seg.mid, np.ones(len(seg))]
    line = np.cross(v[None, :], c)
    return (line * np.c_[seg.a, np.ones(len(seg))]).sum(axis=1) / np.maximum(np.hypot(line[:, 0], line[:, 1]), 1e-12)


def _assign(K: np.ndarray, axes: np.ndarray, seg: Segments, tol: np.ndarray, bin_px: float):
    """Every segment goes to the axis it agrees with best, if it agrees with one at all."""
    inl, _, _, res = support(K, axes, seg, tol, bin_px)
    inl &= res <= res.min(axis=0, keepdims=True)
    m = geo.epipolar_planes(geo.pixel_rays(K, seg.a), geo.pixel_rays(K, seg.b))
    lines = geo._distinct_lines(axes, m, inl, bin_px / K[0, 0])
    return inl, lines, (inl * seg.length[None, :]).sum(axis=1)


def _rotation(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector w (Rodrigues)."""
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * Kx + (1.0 - np.cos(angle)) * (Kx @ Kx)


def _residuals(K: np.ndarray, axes: np.ndarray, seg: Segments, inl: np.ndarray, tol: np.ndarray):
    """Function of a small rotation w of the frame: for every segment the distance of its end
    from the line to the vanishing point of its axis, in units of its tolerance."""
    parts = [seg[inl[k]] for k in range(3)]
    scale = [tol[inl[k]] for k in range(3)]

    def fun(w: np.ndarray) -> np.ndarray:
        turned = axes @ _rotation(w).T
        return np.concatenate([signed_distance(K, turned[k], parts[k]) / scale[k] for k in range(3)])

    return fun


def refine_frame(K: np.ndarray, axes: np.ndarray, seg: Segments, inl: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """Turn the three axes together so that the segments of each axis point to its vanishing
    point as well as possible (three degrees of freedom, robust least squares)."""
    if inl.sum() < 3:
        return axes
    sol = least_squares(_residuals(K, axes, seg, inl, tol), np.zeros(3), loss="soft_l1", f_scale=1.0, x_scale=1e-2)
    return axes @ _rotation(sol.x).T


def frame_uncertainty(K: np.ndarray, axes: np.ndarray, seg: Segments, inl: np.ndarray,
                      noise_px: float = 1.0) -> np.ndarray:
    """How well the segments hold each axis in place: standard deviation in degrees of the
    direction of every axis, if each segment end were off by `noise_px` pixels (first-order
    propagation through the fit of refine_frame). Large for an object that is seen almost
    in parallel projection and shows edges of only two of the three directions."""
    fun = _residuals(K, axes, seg, inl, np.full(len(seg), noise_px))
    h = 1e-4
    J = np.stack([(fun(h * e) - fun(-h * e)) / (2.0 * h) for e in np.eye(3)], axis=1)
    cov = np.linalg.pinv(J.T @ J, rcond=1e-12)
    if np.linalg.matrix_rank(J.T @ J, tol=1e-9) < 3:
        return np.full(3, 90.0)
    out = np.zeros(3)
    for k in range(3):
        e = axes[k]
        Ex = np.array([[0.0, -e[2], e[1]], [e[2], 0.0, -e[0]], [-e[1], e[0], 0.0]])
        out[k] = np.degrees(np.sqrt(max(float(np.linalg.eigvalsh(Ex @ cov @ Ex.T)[-1]), 0.0)))
    return np.minimum(out, 90.0)


def fit_frame(K: np.ndarray, seg: Segments, dirs: list[Direction], tol_deg: float = 1.5, tol_px: float = 2.0,
              bin_px: float = 4.0, max_first: int = 3) -> Frame | None:
    """Three perpendicular axes that explain as much segment length as possible.

    The first axis is one of the strongest vanishing directions. With it fixed, a single
    further segment gives the second axis (perpendicular to the first, in the plane of the
    segment), and the third follows. All such triples are scored, the best one is refined.
    The perpendicularity is what makes this better than three separate directions: a
    direction with few or short edges is held in place by the other two."""
    if not dirs or len(seg) < 2:
        return None
    m = geo.epipolar_planes(geo.pixel_rays(K, seg.a), geo.pixel_rays(K, seg.b))
    tol = tolerance(seg.length, tol_deg, tol_px)
    best, best_score = None, -1.0
    for first in dirs[:max_first]:
        e1 = first.d
        rest = ~first.inliers
        q = np.cross(e1[None, :], m[rest])
        norm = np.linalg.norm(q, axis=1)
        ok = norm > 0.2                                       # the segment says something about the turn about e1
        if not ok.any():
            continue
        e2 = q[ok] / norm[ok, None]
        e3 = np.cross(e1[None, :], e2)
        _, _, len2, _ = support(K, e2, seg, tol, bin_px, rest)
        _, _, len3, _ = support(K, e3, seg, tol, bin_px, rest)
        j = int(np.argmax(len2 + len3))
        score = first.length + float(len2[j] + len3[j])
        if score > best_score:
            best, best_score = np.stack([e1, e2[j], e3[j]]), score
    if best is None:
        return None
    axes = best
    inl, lines, length = _assign(K, axes, seg, tol, bin_px)
    for _ in range(3):
        new = refine_frame(K, axes, seg, inl, tol)
        inl_new, lines_new, length_new = _assign(K, new, seg, tol, bin_px)
        if length_new.sum() < length.sum():
            break
        axes, inl, lines, length = new, inl_new, lines_new, length_new
    return Frame(axes, inl, lines, length)


@dataclass
class LinePlane:
    """Result of plane_normal: the normal, or the reason why the lines do not give one."""
    normal: np.ndarray | None
    reason: str                  # "" | no_lines | tilted | not_box | few_lines | ambiguous | unconfirmed | uncertain
    segments: Segments
    frame: Frame | None = None
    axis: int = -1               # row of frame.axes that is the normal
    up: int = -1                 # row of frame.axes that is the vertical
    explained: float = 0.0       # share of the edge length that is straight and points to an axis
    sigma_deg: float = 90.0      # uncertainty of the normal (frame_uncertainty)
    agree_deg: float = 90.0      # angle between the normal and the nearest vanishing direction found alone
    gravity_off_deg: float = 90.0
    frame_ok: bool = False       # upright and box-like: the three axes can be used, whatever the normal


def plane_normal(gray: np.ndarray, region: np.ndarray, K: np.ndarray, centre: np.ndarray, size: float,
                 min_length: float = 20.0, tol_deg: float = 1.5, tol_px: float = 2.0, min_explained: float = 0.5,
                 max_sigma_deg: float = 1.5, max_agree_deg: float = 5.0, max_gravity_off_deg: float = 20.0,
                 min_lines: int = 3, min_view_gap_deg: float = 5.0, min_share: float = 0.25, blur: float = 1.0,
                 seed: int = 0) -> LinePlane:
    """Normal of the symmetry plane of an upright, box-like object from its straight edges.

    The edges are explained by three perpendicular axes (fit_frame). One of them is the
    vertical. Of the two horizontal ones the normal is the axis that the camera looks along
    least: furniture is photographed from its front, so the viewing ray (`centre`, the unit
    ray to the middle of the object) is closer to the symmetry plane than to its normal.
    The answer is refused when

      tilted       no axis is close to a possible direction of gravity
      not_box      less than `min_explained` of the edge length is straight and points to one
                   of the three axes (curved edges and edges in other directions count against)
      few_lines    fewer than `min_lines` edges point along the normal
      ambiguous    the camera looks along both horizontal axes equally (their view angles
                   differ by less than `min_view_gap_deg`), front and side cannot be told apart
      unconfirmed  no vanishing direction found without the other two axes lies within
                   `max_agree_deg` of the normal: the axis rests on the perpendicularity alone
      uncertain    one pixel of error at the segment ends moves the normal by more than
                   `max_sigma_deg`

    All limits were chosen on 639 Pix3D images (configs/dev_large_pix3d.txt). `size` is the
    diagonal of the object in pixels. `blur` >= 1 says by how much the image is softer than
    a sharp one, for a photo that was enlarged: the gradient filter gets wider, the gradient
    thresholds lower and the straightness test looser by this factor. The sign of the normal
    is arbitrary."""
    seg = detect_segments(gray, region, sigma=1.2 * blur, low=0.012 / blur, high=0.03 / blur,
                          min_length=min_length, max_rms=0.5 * blur)
    dirs = vanishing_directions(K, seg, tol_deg=tol_deg, tol_px=tol_px, min_total=min_share * size, seed=seed)
    frame = fit_frame(K, seg, dirs, tol_deg=tol_deg, tol_px=tol_px)
    if frame is None:
        return LinePlane(None, "no_lines", seg)
    up, off = vertical_axis(frame.axes)
    view = np.degrees(np.arcsin(np.minimum(1.0, np.abs(frame.axes @ centre))))
    k, other = sorted((j for j in range(3) if j != up), key=lambda j: view[j])
    explained = float(frame.length.sum() / max(seg.length.sum() + seg.curved, 1e-9))
    out = LinePlane(None, "", seg, frame, k, up, explained,
                    float(frame_uncertainty(K, frame.axes, seg, frame.inliers)[k]),
                    min(angle_between(frame.axes[k], d.d) for d in dirs), off)
    out.frame_ok = off <= max_gravity_off_deg and out.explained >= min_explained
    if off > max_gravity_off_deg:
        out.reason = "tilted"
    elif out.explained < min_explained:
        out.reason = "not_box"
    elif frame.lines[k] < min_lines:
        out.reason = "few_lines"
    elif view[other] - view[k] < min_view_gap_deg:
        out.reason = "ambiguous"
    elif out.agree_deg > max_agree_deg:
        out.reason = "unconfirmed"
    elif out.sigma_deg > max_sigma_deg:
        out.reason = "uncertain"
    else:
        out.normal = frame.axes[k].copy()
    return out


def vertical_axis(axes: np.ndarray, pitch_deg: tuple[float, float] = (-15.0, 65.0)) -> tuple[int, float]:
    """Which of the axes is the vertical of the scene, and how far it is (degrees) from the
    nearest direction that gravity can have in the camera frame.

    A camera that is not rolled and looks down by the pitch p sees gravity along
    (0, cos p, sin p). Photos of furniture are taken with p between slightly negative and
    steeply downwards, so the vertical is the axis closest to that arc."""
    p = np.radians(np.linspace(pitch_deg[0], pitch_deg[1], 81))
    arc = np.stack([np.zeros_like(p), np.cos(p), np.sin(p)], axis=1)
    off = np.degrees(np.arccos(np.clip(np.abs(axes @ arc.T).max(axis=1), 0.0, 1.0)))
    k = int(np.argmin(off))
    return k, float(off[k])

