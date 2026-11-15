"""Scale-space keypoints with SIFT-like descriptors (Lowe 2004) in numpy/scipy, and the
mirrored matching that turns one image of a symmetric object into stereo correspondences.

A mirror-symmetric object looks, around a point x', like the reflected neighbourhood of the
partner point x. The descriptor is computed in a frame rotated to the dominant gradient
direction, and in that frame reflecting the image only flips the second axis. The
descriptor of the reflected patch is therefore a fixed permutation of the original one
(`mirror_descriptors`), so no second, flipped image is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

N_SPATIAL, N_ORI = 4, 8              # 4 x 4 cells with 8 orientation bins each = 128 numbers


@dataclass
class Features:
    xy: np.ndarray        # (N, 2) pixel coordinates (x, y)
    scale: np.ndarray     # (N,) blur scale sigma in pixels
    angle: np.ndarray     # (N,) dominant gradient direction in radians
    desc: np.ndarray      # (N, 128) unit-length descriptors
    loc: np.ndarray       # (N,) keypoint index; one keypoint can have two orientations

    def __len__(self) -> int:
        return len(self.xy)


# ------------------------------------------------------------------------------ detector

def _pyramid(gray: np.ndarray, n_octaves: int, n_scales: int, sigma0: float) -> list[np.ndarray]:
    """Gaussian scale space: per octave an array (n_scales + 3, H, W), blur sigma0 * 2^(s/n)."""
    k = 2.0 ** (1.0 / n_scales)
    base = ndi.gaussian_filter(gray, np.sqrt(max(sigma0 ** 2 - 0.5 ** 2, 1e-4)))
    octaves = []
    for _ in range(n_octaves):
        levels = [base]
        for s in range(1, n_scales + 3):
            prev = sigma0 * k ** (s - 1)
            levels.append(ndi.gaussian_filter(levels[-1], prev * np.sqrt(k * k - 1.0)))
        octaves.append(np.stack(levels))
        base = levels[n_scales][::2, ::2]
        if min(base.shape) < 32:
            break
    return octaves


def _derivatives(d: np.ndarray, s: np.ndarray, y: np.ndarray, x: np.ndarray):
    """Gradient (N, 3) and Hessian (N, 3, 3) of the DoG stack at integer positions, order s, y, x."""
    c = d[s, y, x]
    g = 0.5 * np.stack([d[s + 1, y, x] - d[s - 1, y, x],
                        d[s, y + 1, x] - d[s, y - 1, x],
                        d[s, y, x + 1] - d[s, y, x - 1]], axis=1)
    hss = d[s + 1, y, x] + d[s - 1, y, x] - 2 * c
    hyy = d[s, y + 1, x] + d[s, y - 1, x] - 2 * c
    hxx = d[s, y, x + 1] + d[s, y, x - 1] - 2 * c
    hsy = 0.25 * (d[s + 1, y + 1, x] - d[s + 1, y - 1, x] - d[s - 1, y + 1, x] + d[s - 1, y - 1, x])
    hsx = 0.25 * (d[s + 1, y, x + 1] - d[s + 1, y, x - 1] - d[s - 1, y, x + 1] + d[s - 1, y, x - 1])
    hyx = 0.25 * (d[s, y + 1, x + 1] - d[s, y + 1, x - 1] - d[s, y - 1, x + 1] + d[s, y - 1, x - 1])
    H = np.stack([np.stack([hss, hsy, hsx], 1), np.stack([hsy, hyy, hyx], 1),
                  np.stack([hsx, hyx, hxx], 1)], axis=1)
    return c, g, H


def _extrema(dog: np.ndarray, contrast: float, edge_ratio: float, border: int = 4):
    """Local extrema of the difference-of-Gaussians stack, refined to sub-pixel accuracy.
    Returns level s, y, x (floats, octave coordinates) and the contrast |D|."""
    S, H, W = dog.shape
    big = ndi.maximum_filter(dog, size=3, mode="nearest")
    small = ndi.minimum_filter(dog, size=3, mode="nearest")
    cand = ((dog == big) & (dog > 0)) | ((dog == small) & (dog < 0))
    cand &= np.abs(dog) > 0.7 * contrast
    cand[[0, -1]] = False
    cand[:, :border] = cand[:, -border:] = False
    cand[:, :, :border] = cand[:, :, -border:] = False
    s, y, x = (a.astype(np.int64) for a in np.nonzero(cand))

    off = np.zeros((len(s), 3))
    val = np.zeros(len(s))
    edge = np.zeros(len(s), bool)
    done = np.zeros(len(s), bool)
    alive = np.ones(len(s), bool)
    for _ in range(4):                                # move to the neighbour if the peak is there
        idx = np.nonzero(alive & ~done)[0]
        if len(idx) == 0:
            break
        c, g, Hm = _derivatives(dog, s[idx], y[idx], x[idx])
        good = np.abs(np.linalg.det(Hm)) > 1e-14
        o = np.zeros((len(idx), 3))
        o[good] = -np.linalg.solve(Hm[good], g[good][:, :, None])[:, :, 0]
        conv = good & (np.abs(o) <= 0.5).all(axis=1)
        ci = idx[conv]
        off[ci], val[ci] = o[conv], c[conv] + 0.5 * np.einsum("ij,ij->i", g[conv], o[conv])
        tr = Hm[conv, 1, 1] + Hm[conv, 2, 2]
        det = Hm[conv, 1, 1] * Hm[conv, 2, 2] - Hm[conv, 1, 2] ** 2
        edge[ci] = (det <= 0) | (tr ** 2 * edge_ratio >= (edge_ratio + 1) ** 2 * det)
        done[ci] = True
        alive[idx[~good]] = False
        mv = idx[good & ~conv]
        step = np.rint(np.clip(o[good & ~conv], -1, 1)).astype(np.int64)
        s[mv], y[mv], x[mv] = s[mv] + step[:, 0], y[mv] + step[:, 1], x[mv] + step[:, 2]
        inside = ((s[mv] >= 1) & (s[mv] <= S - 2) & (y[mv] >= border) & (y[mv] < H - border)
                  & (x[mv] >= border) & (x[mv] < W - border))
        alive[mv[~inside]] = False
    keep = done & ~edge & (np.abs(val) >= contrast)
    return s[keep] + off[keep, 0], y[keep] + off[keep, 1], x[keep] + off[keep, 2], np.abs(val[keep])


# ---------------------------------------------------------------- orientation, descriptor

def _sample(stack: np.ndarray, level: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    lev = np.broadcast_to(level[:, None], yy.shape)
    return ndi.map_coordinates(stack, [lev, yy, xx], order=1, mode="nearest")


def _orientations(gx, gy, level, y, x, sigma, n_bins: int = 36, peak: float = 0.8):
    """Dominant gradient directions from a Gaussian-weighted orientation histogram.
    Returns (index of the keypoint, angle); a second peak above 80 % gives a second entry."""
    lin = np.linspace(-1.0, 1.0, 13)
    oy, ox = np.meshgrid(lin, lin, indexing="ij")
    inside = oy ** 2 + ox ** 2 <= 1.0
    oy, ox = oy[inside], ox[inside]
    radius = 4.5 * sigma                               # 3 standard deviations of a 1.5 sigma window
    yy, xx = y[:, None] + radius[:, None] * oy, x[:, None] + radius[:, None] * ox
    sx, sy = _sample(gx, level, yy, xx), _sample(gy, level, yy, xx)
    weight = np.hypot(sx, sy) * np.exp(-4.5 * (oy ** 2 + ox ** 2))[None, :]
    bins = np.floor((np.arctan2(sy, sx) % (2 * np.pi)) / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
    N = len(y)
    hist = np.bincount((np.arange(N)[:, None] * n_bins + bins).ravel(), weights=weight.ravel(),
                       minlength=N * n_bins).reshape(N, n_bins)
    for _ in range(2):                                 # circular smoothing
        hist = (np.roll(hist, 1, axis=1) + hist + np.roll(hist, -1, axis=1)) / 3.0
    left, right = np.roll(hist, 1, axis=1), np.roll(hist, -1, axis=1)
    is_peak = (hist > left) & (hist >= right) & (hist >= peak * hist.max(axis=1, keepdims=True))
    is_peak &= hist > 0
    # keep the two strongest peaks per keypoint
    ranked = np.where(is_peak, hist, -1.0)
    order = np.argsort(-ranked, axis=1)[:, :2]
    kp = np.repeat(np.arange(N), 2)
    b = order.ravel()
    ok = ranked[kp, b] > 0
    kp, b = kp[ok], b[ok]
    h0, hl, hr = hist[kp, b], left[kp, b], right[kp, b]
    shift = 0.5 * (hl - hr) / np.where(np.abs(hl - 2 * h0 + hr) > 1e-12, hl - 2 * h0 + hr, 1.0)
    return kp, ((b + 0.5 + shift) * (2 * np.pi / n_bins)) % (2 * np.pi)


def _describe(gx, gy, level, y, x, sigma, angle) -> np.ndarray:
    """4 x 4 x 8 histogram of gradient directions in a window rotated to `angle`.
    Layout: desc[cell row, cell column, orientation bin]."""
    lin = np.arange(-2.4, 2.41, 0.2)                   # window of 5 cells, 5 samples per cell
    cy, cx = (a.ravel() for a in np.meshgrid(lin, lin, indexing="ij"))
    N = len(y)
    out = np.zeros((N, N_SPATIAL * N_SPATIAL * N_ORI), dtype=np.float32)
    for lo in range(0, N, 1024):
        sl = slice(lo, min(N, lo + 1024))
        n = sl.stop - sl.start
        cell = 3.0 * sigma[sl][:, None]                # cell size in pixels
        ca, sa = np.cos(angle[sl])[:, None], np.sin(angle[sl])[:, None]
        yy = y[sl][:, None] + (sa * cx[None] + ca * cy[None]) * cell
        xx = x[sl][:, None] + (ca * cx[None] - sa * cy[None]) * cell
        sx, sy = _sample(gx, level[sl], yy, xx), _sample(gy, level[sl], yy, xx)
        rx, ry = ca * sx + sa * sy, -sa * sx + ca * sy          # gradient in the rotated frame
        weight = np.hypot(rx, ry) * np.exp(-(cx ** 2 + cy ** 2) / 8.0)[None, :]
        bo = (np.arctan2(ry, rx) % (2 * np.pi)) * (N_ORI / (2 * np.pi))
        bx, by = cx + 1.5, cy + 1.5                    # cell centres at 0, 1, 2, 3
        x0, y0, o0 = np.floor(bx).astype(np.int64), np.floor(by).astype(np.int64), np.floor(bo).astype(np.int64)
        fx, fy, fo = bx - x0, by - y0, bo - o0
        acc = np.zeros(n * 128)
        row = np.arange(n)[:, None] * 128
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                yi, xi = y0 + dy, x0 + dx
                valid = (yi >= 0) & (yi < N_SPATIAL) & (xi >= 0) & (xi < N_SPATIAL)
                cellidx = (np.clip(yi, 0, 3) * N_SPATIAL + np.clip(xi, 0, 3)) * N_ORI
                spatial = (wy * wx * valid)[None, :] * weight
                for do, wo in ((0, 1 - fo), (1, fo)):
                    idx = row + cellidx[None, :] + (o0 + do) % N_ORI
                    acc += np.bincount(idx.ravel(), weights=(spatial * wo).ravel(), minlength=n * 128)
        out[sl] = acc.reshape(n, 128)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    out = np.minimum(out, 0.2)                         # limit the influence of single strong edges
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def detect_and_describe(gray: np.ndarray, mask: np.ndarray | None = None, n_octaves: int = 4,
                        n_scales: int = 3, sigma0: float = 1.6, contrast: float = 0.008,
                        edge_ratio: float = 10.0, max_keypoints: int = 2500) -> Features:
    """Keypoints and descriptors of a grey image in [0, 1]; keypoints outside `mask` are dropped."""
    gray = np.asarray(gray, dtype=np.float32)
    parts = []
    for o, octave in enumerate(_pyramid(gray, n_octaves, n_scales, sigma0)):
        s, y, x, c = _extrema(octave[1:] - octave[:-1], contrast, edge_ratio)
        if len(s) == 0:
            continue
        if mask is not None:
            yi = np.clip(np.rint(y * 2 ** o).astype(int), 0, mask.shape[0] - 1)
            xi = np.clip(np.rint(x * 2 ** o).astype(int), 0, mask.shape[1] - 1)
            inside = mask[yi, xi]
            s, y, x, c = s[inside], y[inside], x[inside], c[inside]
        parts.append((o, octave, s, y, x, c))
    if not parts:
        return Features(np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros((0, 128), np.float32),
                        np.zeros(0, np.int64))
    # strongest keypoints over all octaves
    contrast_all = np.concatenate([p[5] for p in parts])
    cut = np.sort(contrast_all)[-max_keypoints] if len(contrast_all) > max_keypoints else 0.0

    xy, scale, angle, desc, loc = [], [], [], [], []
    n_loc = 0
    for o, octave, s, y, x, c in parts:
        keep = c >= cut
        s, y, x = s[keep], y[keep], x[keep]
        if len(s) == 0:
            continue
        gy, gx = np.gradient(octave, axis=(1, 2))
        level = np.clip(np.rint(s), 0, octave.shape[0] - 1)
        sigma = sigma0 * 2.0 ** (s / n_scales)         # in pixels of this octave
        kp, ang = _orientations(gx, gy, level, y, x, sigma)
        desc.append(_describe(gx, gy, level[kp], y[kp], x[kp], sigma[kp], ang))
        xy.append(np.c_[x[kp], y[kp]] * 2 ** o)
        scale.append(sigma[kp] * 2 ** o)
        angle.append(ang)
        loc.append(kp + n_loc)
        n_loc += len(s)
    return Features(np.concatenate(xy), np.concatenate(scale), np.concatenate(angle),
                    np.concatenate(desc), np.concatenate(loc))


# ---------------------------------------------------------------------- mirrored matching

def mirror_descriptors(desc: np.ndarray) -> np.ndarray:
    """Descriptors of the reflected patches: flip the cell rows, negate the gradient angle."""
    d = desc.reshape(-1, N_SPATIAL, N_SPATIAL, N_ORI)
    return d[:, ::-1, :, (-np.arange(N_ORI)) % N_ORI].reshape(desc.shape)


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def orientation_deviation(f: Features, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """A reflection along the line joining two partners maps the gradient direction phi_i to
    2 beta + pi - phi_i, beta being the direction of that line. Returns the deviation from it."""
    dxy = f.xy[j] - f.xy[i]
    beta = np.arctan2(dxy[..., 1], dxy[..., 0])
    return np.abs(_wrap(f.angle[i] + f.angle[j] - 2 * beta - np.pi))


def mirror_matches(f: Features, min_sep: float, ratio: float = 0.85, max_dist: float = 0.75,
                   ori_tol_deg: float | None = 40.0, max_scale_ratio: float = 2.5,
                   exclusion: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Candidate mirror partners (i, j) with their descriptor distance.

    Feature i is matched against the reflected descriptors of all other features. A match
    is kept if it passes Lowe's ratio test, where the second-best candidate has to be more
    than `exclusion` pixels away from the best one."""
    N = len(f)
    if N < 2:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    sim = f.desc @ mirror_descriptors(f.desc).T        # symmetric: reflecting twice is the identity
    px, py = f.xy[:, 0].astype(np.float32), f.xy[:, 1].astype(np.float32)
    dx, dy = px[None, :] - px[:, None], py[None, :] - py[:, None]      # from i (row) to j (column)
    bad = dx ** 2 + dy ** 2 < min_sep ** 2
    sr = (f.scale[:, None] / f.scale[None, :]).astype(np.float32)
    bad |= (sr > max_scale_ratio) | (sr < 1.0 / max_scale_ratio)
    if ori_tol_deg is not None:
        ang = f.angle.astype(np.float32)
        dev = _wrap(ang[:, None] + ang[None, :] - 2 * np.arctan2(dy, dx) - np.pi)
        bad |= np.abs(dev) > np.radians(ori_tol_deg)
    sim[bad] = -1.0
    best = sim.argmax(axis=1)
    s1 = sim[np.arange(N), best]
    near_best = (px[None, :] - px[best][:, None]) ** 2 + (py[None, :] - py[best][:, None]) ** 2 < exclusion ** 2
    s2 = np.where(near_best, -1.0, sim).max(axis=1)
    dist1 = np.sqrt(np.maximum(2.0 - 2.0 * s1, 0.0))
    dist2 = np.sqrt(np.maximum(2.0 - 2.0 * s2, 0.0))
    ok = (s1 > -1.0) & (dist1 < max_dist) & (dist1 < ratio * dist2)
    i, j, dist = np.nonzero(ok)[0], best[ok], dist1[ok]
    # one entry per unordered pair of keypoint locations, the closest descriptors win
    a, b = np.minimum(f.loc[i], f.loc[j]), np.maximum(f.loc[i], f.loc[j])
    order = np.lexsort((dist, b, a))
    first = np.r_[True, (a[order][1:] != a[order][:-1]) | (b[order][1:] != b[order][:-1])]
    sel = order[first]
    return np.c_[i[sel], j[sel]], dist[sel]
