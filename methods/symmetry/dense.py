"""Dense part of the symmetry method: with the symmetry plane known, every object pixel x
has its mirror partner x' somewhere on the line through x and the epipole v = K n. Each
position on that line corresponds to one depth of the point behind x.

Two things are read off that line:

* silhouette: the partner has to fall inside the object mask. The depths for which it does
  form the intersection of the two silhouette cones (real camera and mirrored camera), a
  two-view visual hull. Its surface is where the partner enters or leaves the mask.
* appearance: among the admissible positions the partner is the one whose neighbourhood
  looks like the mirrored neighbourhood of x (normalised cross-correlation of patches).

How the neighbourhood of x appears around x' depends on the orientation of the surface.
Faces that are their own mirror image (front, top, seat, back rest) are perpendicular to
the symmetry plane, so their normal is one of the directions perpendicular to n. A few of
those directions are tried; for each one the patch is put on that plane in 3D, mirrored and
projected, which gives the sampling positions around x'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class DenseResult:
    pixels: np.ndarray        # (N, 2) pixels with an accepted partner
    partners: np.ndarray      # (N, 2) the partner positions
    points: np.ndarray        # (N, 3) their 3D points (mirror copies not included)
    score: np.ndarray         # (N,) correlation of the accepted match
    normal_index: np.ndarray  # (N,) which surface orientation gave that correlation
    hull: np.ndarray          # (M, 3) surface points of the silhouette intersection (one half)
    n_textured: int = 0       # pixels that had enough texture and at least one candidate


def _empty() -> DenseResult:
    z2, z3 = np.zeros((0, 2)), np.zeros((0, 3))
    return DenseResult(z2, z2, z3, np.zeros(0), np.zeros(0, np.int64), z3)


def plane_normals(n: np.ndarray, count: int) -> np.ndarray:
    """`count` unit vectors perpendicular to n, evenly spread over half a turn. The first one
    is the direction closest to the optical axis (the plane that faces the camera most)."""
    b1 = np.array([0.0, 0.0, 1.0]) - n[2] * n
    if np.linalg.norm(b1) < 1e-6:                    # n along the optical axis
        b1 = np.cross(n, [1.0, 0.0, 0.0])
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1)
    w = np.arange(count) * np.pi / count
    return np.cos(w)[:, None] * b1[None, :] + np.sin(w)[:, None] * b2[None, :]


def _despeckle(z: np.ndarray, rel_tol: float, min_support: int = 3) -> np.ndarray:
    """Keep a depth sample if enough of its 5 x 5 neighbours agree with it. z is the depth on
    the pixel grid, NaN where there is no match."""
    H, W = z.shape
    pad = np.pad(z, 2, constant_values=np.nan)
    agree = np.zeros((H, W), dtype=np.int64)
    for dy in range(5):
        for dx in range(5):
            if dy == 2 and dx == 2:
                continue
            nb = pad[dy:dy + H, dx:dx + W]
            with np.errstate(invalid="ignore"):
                agree += np.abs(nb - z) <= rel_tol * z
    return np.isfinite(z) & (agree >= min_support)


class _Scanner:
    """Search along the epipolar lines of arbitrary pixels of one image."""

    def __init__(self, gray, mask, K, n, normals, dt, half, spacing, min_ncc, uniqueness,
                 min_std, min_shift, chunk):
        self.mask, self.K, self.n, self.normals = mask, K, n, normals
        self.Kinv = np.linalg.inv(K)
        self.v = K @ n
        self.dt, self.min_ncc, self.uniqueness = dt, min_ncc, uniqueness
        self.min_std, self.min_shift, self.chunk = min_std, min_shift, chunk
        self.smooth = ndi.gaussian_filter(np.asarray(gray, dtype=np.float32), 1.0)
        offs = np.arange(-half, half + 1) * spacing
        self.oy, self.ox = (a.ravel() for a in np.meshgrid(offs, offs, indexing="ij"))
        # Patches must lie on the object: next to the silhouette they would mostly compare
        # the outline against the grey background, which looks alike for any plane.
        self.inner = ndi.binary_erosion(mask, iterations=int(np.ceil(half * spacing)) + 1)
        my, mx = np.nonzero(mask)
        reach = np.hypot(np.ptp(my), np.ptp(mx)) + 2.0
        self.t = np.arange(-reach, reach + dt, dt)                  # signed shift along the line

    def _lookup(self, img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        H, W = img.shape
        xi, yi = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        out = np.zeros(x.shape, bool)
        out[inside] = img[yi[inside], xi[inside]]
        return out

    def _patches(self, px: np.ndarray, scale) -> np.ndarray:
        """Zero-mean patches (P, S) around the pixels, sample spacing multiplied by `scale`."""
        s = np.broadcast_to(np.asarray(scale, dtype=np.float64), (len(px),))[:, None]
        patch = ndi.map_coordinates(self.smooth, [px[:, 1:2] + s * self.oy[None, :],
                                                  px[:, 0:1] + s * self.ox[None, :]], order=1, mode="nearest")
        return patch - patch.mean(axis=1, keepdims=True)

    def _mirror_map(self, rays: np.ndarray, depth: np.ndarray, partner: np.ndarray, m_dir: np.ndarray):
        """Put the neighbourhood of x on the plane through X with normal m_dir, mirror it and
        project it. Locally that is an affine map of pixel offsets. Returns the images of a
        step of one pixel to the right and of one pixel down, and where the map is usable."""
        mr = rays @ m_dir
        usable = np.abs(mr) > 0.17 * np.linalg.norm(rays, axis=1)   # plane not seen edge-on
        cols = []
        for step_ray in (self.Kinv[:, 0], self.Kinv[:, 1]):
            rk = rays + step_ray
            mk = rk @ m_dir
            usable &= mk * mr > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                Y = (depth * mr / mk)[:, None] * rk                 # neighbour ray meets the plane
                Ym = (Y - 2.0 * (Y @ self.n - 1.0)[:, None] * self.n) @ self.K.T
                cols.append(Ym[:, :2] / Ym[:, 2:3] - partner)
        ax, ay = cols
        det = ax[:, 0] * ay[:, 1] - ax[:, 1] * ay[:, 0]
        with np.errstate(invalid="ignore"):
            usable &= (np.abs(det) > 0.1) & (np.abs(det) < 10.0) & ((ax ** 2 + ay ** 2).sum(axis=1) < 36.0)
        return ax, ay, usable

    def _correlate(self, ref, ref_norm, partner, ax, ay, ox, oy) -> np.ndarray:
        """Correlation of zero-mean reference patches with the patches around the partners,
        sampled through the affine map (ax, ay)."""
        sx = partner[:, 0:1] + ox[None, :] * ax[:, 0:1] + oy[None, :] * ay[:, 0:1]
        sy = partner[:, 1:2] + ox[None, :] * ax[:, 1:2] + oy[None, :] * ay[:, 1:2]
        patch = ndi.map_coordinates(self.smooth, [sy, sx], order=1, mode="nearest")
        patch -= patch.mean(axis=1, keepdims=True)
        return (patch * ref).sum(axis=1) / (np.linalg.norm(patch, axis=1) * ref_norm + 1e-9)

    def verify(self, px, rays, depth, partner, normal_index, half: int = 4, spacing: float = 2.5) -> np.ndarray:
        """Correlation of accepted matches over a window about twice as large, with the same
        mirror map. A small patch of smooth texture finds look-alikes; a large one rarely."""
        offs = np.arange(-half, half + 1) * spacing
        oy, ox = (a.ravel() for a in np.meshgrid(offs, offs, indexing="ij"))
        ref = ndi.map_coordinates(self.smooth, [px[:, 1:2] + oy[None, :], px[:, 0:1] + ox[None, :]],
                                  order=1, mode="nearest")
        ref = ref - ref.mean(axis=1, keepdims=True)
        ref_norm = np.linalg.norm(ref, axis=1)
        out = np.full(len(px), -1.0)
        for mi, m_dir in enumerate(self.normals):
            sel = np.nonzero(normal_index == mi)[0]
            if len(sel) == 0:
                continue
            ax, ay, usable = self._mirror_map(rays[sel], depth[sel], partner[sel], m_dir)
            sel = sel[usable]
            out[sel] = self._correlate(ref[sel], ref_norm[sel], partner[sel], ax[usable], ay[usable], ox, oy)
        return out

    def search(self, px: np.ndarray, hull: bool = False, match: bool = True) -> dict:
        """Best partner of every pixel in px (P, 2). Returns the shift along the line, the
        depth, the correlation, the winning orientation, an accept flag, and optionally the
        depths at which the partner crosses the mask boundary."""
        n, Kinv, t = self.n, self.Kinv, self.t
        P, C, S = len(px), len(t), len(self.ox)
        u = np.c_[self.v[0] - self.v[2] * px[:, 0], self.v[1] - self.v[2] * px[:, 1]]
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)   # towards the epipole
        r = np.c_[px, np.ones(P)] @ Kinv.T                          # ray of the pixel
        q = np.c_[u, np.zeros(P)] @ Kinv.T                          # ray change per pixel of shift
        a1, aq = r @ n, q @ n
        rr, rq, qq = (r * r).sum(1), (r * q).sum(1), (q * q).sum(1)

        def depth_at(sl, shift):
            a2 = a1[sl] + shift * aq[sl]
            r12 = rr[sl] + shift * rq[sl]
            r22 = rr[sl] + 2 * shift * rq[sl] + shift ** 2 * qq[sl]
            rho = (r12 - a1[sl] * a2) / np.maximum(r22 - a2 ** 2, 1e-12)
            den = a1[sl] + rho * a2
            front = (rho > 0) & (den > 1e-12)
            return np.where(front, 2.0 / np.where(front, den, 1.0), 0.0), front, rho

        ref = self._patches(px, 1.0)
        ref_norm = np.linalg.norm(ref, axis=1)
        textured = (ref_norm / np.sqrt(S) >= self.min_std) & self._lookup(self.inner, px[:, 0], px[:, 1])

        out = {"ok": np.zeros(P, bool), "tried": np.zeros(P, bool), "shift": np.zeros(P),
               "score": np.zeros(P), "normal": np.zeros(P, dtype=np.int64), "hull_idx": [], "hull_z": []}
        for lo in range(0, P, self.chunk):
            sl = slice(lo, min(P, lo + self.chunk))
            B = sl.stop - sl.start
            xp = px[sl, 0:1] + t[None, :] * u[sl, 0:1]
            yp = px[sl, 1:2] + t[None, :] * u[sl, 1:2]
            z, front, _ = depth_at((sl, None), t[None, :])
            if hull:
                # surface of the silhouette intersection: the partner crosses the mask boundary
                occ = front & self._lookup(self.mask, xp, yp)
                b, k = np.nonzero(occ[:, 1:] != occ[:, :-1])
                out["hull_idx"].append(lo + b)
                out["hull_z"].append(z[b, np.where(occ[b, k], k, k + 1)])
            if not match:
                continue
            cand = front & self._lookup(self.inner, xp, yp) & (np.abs(t)[None, :] >= self.min_shift)
            b, k = np.nonzero(cand & textured[sl, None])
            if len(b) == 0:
                continue
            out["tried"][lo + np.unique(b)] = True
            rb = r[sl][b]
            zb = z[b, k]
            cb = np.c_[xp[b, k], yp[b, k]]                          # partner position of every candidate
            refb, refn = ref[sl][b], ref_norm[sl][b]
            ncc = np.full(len(b), -1.0)
            which = np.zeros(len(b), dtype=np.int64)
            for mi, m_dir in enumerate(self.normals):
                ax, ay, usable = self._mirror_map(rb, zb, cb, m_dir)
                if not usable.any():
                    continue
                c = self._correlate(refb[usable], refn[usable], cb[usable], ax[usable], ay[usable],
                                    self.ox, self.oy)
                ui = np.nonzero(usable)[0]
                better = c > ncc[ui]
                ncc[ui[better]], which[ui[better]] = c[better], mi
            score = np.full((B, C), -1.0)
            score[b, k] = ncc
            winner = np.zeros((B, C), dtype=np.int64)
            winner[b, k] = which
            kb = score.argmax(axis=1)
            rows = np.arange(B)
            s1 = score[rows, kb]
            window = np.abs(np.arange(C)[None, :] - kb[:, None]) <= max(2, int(round(3 / self.dt)))
            s2 = np.where(window, -1.0, score).max(axis=1)
            # parabola through the neighbours for a sub-sample position
            km, kp = np.clip(kb - 1, 0, C - 1), np.clip(kb + 1, 0, C - 1)
            sm, sp = score[rows, km], score[rows, kp]
            curv = sm - 2 * s1 + sp
            fine = (sm > -1) & (sp > -1) & (curv < -1e-9)
            sub = np.where(fine, 0.5 * (sm - sp) / np.where(fine, curv, 1.0), 0.0)
            out["ok"][sl] = (s1 >= self.min_ncc) & ((1.0 - s1) <= self.uniqueness * (1.0 - s2))
            out["score"][sl], out["normal"][sl] = s1, winner[rows, kb]
            out["shift"][sl] = t[kb] + np.clip(sub, -0.5, 0.5) * self.dt
        out["partner"] = px + out["shift"][:, None] * u
        out["depth"], in_front, rho = depth_at(np.arange(P), out["shift"])
        out["ok"] &= in_front
        out["rays"] = r
        # The same neighbourhood compared without mirroring it: the partner as a shifted copy,
        # resized by the depth ratio. Repeated parts (legs, drawers) match this way.
        copy = self._patches(out["partner"], np.clip(1.0 / np.where(in_front, rho, 1.0), 0.5, 2.0))
        out["copy_score"] = (copy * ref).sum(axis=1) / (np.linalg.norm(copy, axis=1) * ref_norm + 1e-9)
        if hull:
            out["hull_idx"], out["hull_z"] = np.concatenate(out["hull_idx"]), np.concatenate(out["hull_z"])
        return out


def epipolar_scan(gray: np.ndarray, mask: np.ndarray, K: np.ndarray, n: np.ndarray,
                  step: int = 3, dt: float = 1.5, half: int = 2, spacing: float = 3.0,
                  normals: np.ndarray | None = None, min_ncc: float = 0.8,
                  uniqueness: float = 0.8, min_std: float = 0.02, min_shift: float = 4.0,
                  reciprocal: float | None = 2.5, mirror_margin: float | None = 0.05,
                  verify_ncc: float | None = 0.7, despeckle: bool = True, match: bool = True,
                  hull: bool = True, chunk: int = 96) -> DenseResult:
    """Scan the epipolar line of every `step`-th object pixel (plane n . X = 1).

    step        grid spacing of the pixels that are reconstructed
    dt          spacing of the candidate partner positions on the line, pixels
    half        patch half-size in samples, `spacing` pixels apart
    normals     (M, 3) surface orientations tried per candidate, default plane_normals(n, 6)
    min_ncc     minimum correlation of an accepted match
    uniqueness  the best mismatch (1 - ncc) must be below this fraction of the runner-up
    min_shift   ignore partners closer than this (a pixel always matches itself)
    reciprocal  the search started at the partner must come back to within this many pixels
    mirror_margin  the mirrored comparison must beat the unmirrored one by this much
    verify_ncc  minimum correlation of an accepted match over a window twice the size
    despeckle   drop matches whose depth has no support among the neighbours
    match / hull   switch the appearance search / the silhouette intersection off
    """
    H, W = mask.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    on = mask[ys, xs]
    px = np.c_[xs[on], ys[on]].astype(np.float64)
    if len(px) == 0:
        return _empty()
    if normals is None:
        normals = plane_normals(n, 6)
    scan = _Scanner(gray, mask, K, n, normals, dt, half, spacing, min_ncc, uniqueness, min_std,
                    min_shift, chunk)
    fwd = scan.search(px, hull=hull, match=match)
    ok = fwd["ok"]
    if match and reciprocal is not None and ok.any():
        sel = np.nonzero(ok)[0]
        bwd = scan.search(fwd["partner"][sel])                      # does the partner choose x back?
        ok[sel] = bwd["ok"] & (np.linalg.norm(bwd["partner"] - px[sel], axis=1) <= reciprocal)
    if match and mirror_margin is not None:
        ok &= fwd["score"] >= fwd["copy_score"] + mirror_margin     # a mirror image, not a copy
    if match and verify_ncc is not None and ok.any():
        sel = np.nonzero(ok)[0]
        ok[sel] = scan.verify(px[sel], fwd["rays"][sel], fwd["depth"][sel], fwd["partner"][sel],
                              fwd["normal"][sel]) >= verify_ncc
    if match and despeckle:
        zmap = np.full(ys.shape, np.nan)
        zmap[on] = np.where(ok, fwd["depth"], np.nan)
        ok &= _despeckle(zmap, rel_tol=0.01 * step)[on]
    sel = np.nonzero(ok)[0]
    hull_pts = fwd["hull_z"][:, None] * fwd["rays"][fwd["hull_idx"]] if hull else np.zeros((0, 3))
    return DenseResult(px[sel], fwd["partner"][sel], fwd["depth"][sel, None] * fwd["rays"][sel],
                       fwd["score"][sel], fwd["normal"][sel], hull_pts, int(fwd["tried"].sum()))
