"""Controlled test case: mirror-symmetric objects with a symmetric procedural texture,
rendered with a pinhole camera by a small z-buffer rasteriser. Everything is known here
(symmetry plane, depth, shape), so the method can be checked before it sees real photos.

Object frame: symmetric about x = 0, y down, z away from the viewer at azimuth 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage as ndi


def _box(lo, hi) -> trimesh.Trimesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    T = np.eye(4)
    T[:3, 3] = (lo + hi) / 2
    return trimesh.creation.box(extents=hi - lo, transform=T)


def chair() -> trimesh.Trimesh:
    parts = [_box([-0.25, -0.03, -0.25], [0.25, 0.03, 0.25]),            # seat
             _box([-0.25, -0.55, 0.19], [0.25, -0.03, 0.25])]            # back
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sorted([sx * 0.19, sx * 0.25]), sorted([sz * 0.19, sz * 0.25])
            parts.append(_box([x[0], 0.03, z[0]], [x[1], 0.47, z[1]]))   # legs
    return trimesh.util.concatenate(parts)


def table() -> trimesh.Trimesh:
    parts = [_box([-0.6, -0.04, -0.35], [0.6, 0.04, 0.35])]
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sorted([sx * 0.5, sx * 0.58]), sorted([sz * 0.25, sz * 0.33])
            parts.append(_box([x[0], 0.04, z[0]], [x[1], 0.7, z[1]]))
    return trimesh.util.concatenate(parts)


def cabinet() -> trimesh.Trimesh:
    return _box([-0.45, -0.6, -0.22], [0.45, 0.6, 0.22])


OBJECTS = {"chair": chair, "table": table, "cabinet": cabinet}


class SymmetricTexture:
    """Value noise in object space evaluated at (|x|, y, z): the same on both sides.
    mirror=False evaluates it at (x, y, z) instead, a texture without the symmetry."""

    def __init__(self, seed: int = 0, freqs=(7.0, 18.0, 42.0), amps=(0.4, 0.35, 0.25),
                 contrast: float = 2.5, mirror: bool = True):
        rng = np.random.default_rng(seed)
        self.lattices = [rng.random((32, 32, 32)) for _ in freqs]
        self.tint = rng.random((3, 32, 32, 32))
        self.freqs, self.amps, self.contrast, self.mirror = freqs, amps, contrast, mirror

    def __call__(self, P: np.ndarray) -> np.ndarray:
        Q = np.array(P, dtype=np.float64)
        if self.mirror:
            Q[:, 0] = np.abs(Q[:, 0])
        lum = np.zeros(len(Q))
        for lat, f, a in zip(self.lattices, self.freqs, self.amps):
            lum += a * ndi.map_coordinates(lat, (Q * f + 7.3).T, order=1, mode="grid-wrap")
        lum = np.clip(0.5 + self.contrast * (lum - 0.5), 0.05, 1.0)
        tint = np.stack([ndi.map_coordinates(t, (Q * 3.0 + 1.7).T, order=1, mode="grid-wrap")
                         for t in self.tint], axis=1)
        return lum[:, None] * (0.7 + 0.3 * tint)


def pose(azimuth_deg: float, elevation_deg: float = 20.0, distance: float = 2.5,
         shift=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Object-to-camera rotation and translation: X_cam = R X_obj + t."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    Ry = np.array([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return Rx @ Ry, np.array([shift[0], shift[1], distance])


@dataclass
class Render:
    image: np.ndarray        # (H, W, 3) uint8
    mask: np.ndarray         # (H, W) bool
    depth: np.ndarray        # (H, W) z in the camera frame, 0 outside the object
    points: np.ndarray       # (H, W, 3) object coordinates of the visible surface
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    mesh: trimesh.Trimesh    # object frame

    @property
    def plane(self) -> tuple[np.ndarray, float]:
        """Symmetry plane n . X = d in the camera frame, d > 0."""
        n = self.R[:, 0].copy()
        d = float(n @ self.t)
        return (n, d) if d > 0 else (-n, -d)

    def mesh_in_camera(self) -> trimesh.Trimesh:
        m = self.mesh.copy()
        m.vertices = m.vertices @ self.R.T + self.t
        return m

    def project(self, P_obj: np.ndarray) -> np.ndarray:
        x = (P_obj @ self.R.T + self.t) @ self.K.T
        return x[:, :2] / x[:, 2:3]


def _rasterise(V_obj: np.ndarray, V_cam: np.ndarray, faces: np.ndarray, K: np.ndarray, H: int, W: int):
    uvw = V_cam @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    zbuf = np.full((H, W), np.inf)
    pos = np.zeros((H, W, 3))
    fid = np.full((H, W), -1, dtype=np.int64)
    for fi, (a, b, c) in enumerate(faces):
        pa, pb, pc = uv[a], uv[b], uv[c]
        x0, x1 = max(int(np.floor(min(pa[0], pb[0], pc[0]))), 0), min(int(np.ceil(max(pa[0], pb[0], pc[0]))), W - 1)
        y0, y1 = max(int(np.floor(min(pa[1], pb[1], pc[1]))), 0), min(int(np.ceil(max(pa[1], pb[1], pc[1]))), H - 1)
        det = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if x0 > x1 or y0 > y1 or abs(det) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((pb[1] - pc[1]) * (xs - pc[0]) + (pc[0] - pb[0]) * (ys - pc[1])) / det
        w1 = ((pc[1] - pa[1]) * (xs - pc[0]) + (pa[0] - pc[0]) * (ys - pc[1])) / det
        w2 = 1.0 - w0 - w1
        za, zb, zc = V_cam[a, 2], V_cam[b, 2], V_cam[c, 2]
        z = 1.0 / (w0 / za + w1 / zb + w2 / zc)                 # perspective-correct depth
        hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z < zbuf[ys, xs])
        if not hit.any():
            continue
        P = (w0[..., None] * V_obj[a] / za + w1[..., None] * V_obj[b] / zb + w2[..., None] * V_obj[c] / zc) * z[..., None]
        zbuf[ys[hit], xs[hit]] = z[hit]
        pos[ys[hit], xs[hit]] = P[hit]
        fid[ys[hit], xs[hit]] = fi
    return zbuf, pos, fid


def render(mesh: trimesh.Trimesh, K: np.ndarray, R: np.ndarray, t: np.ndarray,
           size: tuple[int, int] = (480, 640), texture: SymmetricTexture | None = None,
           ambient: float = 0.75, background: str = "flat", seed: int = 0, ss: int = 2) -> Render:
    """Render with `ss` x supersampling. background: 'flat' or 'clutter' (asymmetric noise)."""
    H, W = size
    texture = texture or SymmetricTexture(seed)
    S = np.diag([ss, ss, 1.0])
    S[:2, 2] = (ss - 1) / 2.0                                    # pixel centres stay pixel centres
    V_cam = mesh.vertices @ R.T + t
    zbuf, pos, fid = _rasterise(np.asarray(mesh.vertices), V_cam, np.asarray(mesh.faces), S @ K, H * ss, W * ss)
    hit = np.isfinite(zbuf)

    color = np.zeros((H * ss, W * ss, 3))
    normals = np.asarray(mesh.face_normals) @ R.T
    light = np.array([0.35, -0.45, 1.0])
    light /= np.linalg.norm(light)                               # direction the light travels
    shade = ambient + (1.0 - ambient) * np.clip(-(normals @ light), 0.0, 1.0)
    color[hit] = texture(pos[hit]) * shade[fid[hit]][:, None]
    rng = np.random.default_rng(seed + 1)
    if background == "clutter":
        bg = ndi.gaussian_filter(rng.random((H * ss, W * ss, 3)), (4 * ss, 4 * ss, 0))
        bg = np.clip(0.5 + 6.0 * (bg - 0.5), 0, 1)
    else:
        bg = np.full((H * ss, W * ss, 3), 0.85)
    color[~hit] = bg[~hit]

    def down(a: np.ndarray) -> np.ndarray:
        return a.reshape(H, ss, W, ss, *a.shape[2:]).mean(axis=(1, 3))

    cover = down(hit.astype(np.float64))
    mask = cover > 0.5
    zsum, psum = down(np.where(hit, zbuf, 0.0)), down(pos * hit[..., None])
    depth = np.where(mask, zsum / np.maximum(cover, 1e-9), 0.0)
    points = np.where(mask[..., None], psum / np.maximum(cover, 1e-9)[..., None], 0.0)
    image = (np.clip(down(color), 0, 1) * 255).round().astype(np.uint8)
    return Render(image, mask, depth, points, K.copy(), R.copy(), t.copy(), mesh)


def make(name: str = "chair", azimuth: float = 25.0, elevation: float = 20.0, distance: float = 2.5,
         size: tuple[int, int] = (480, 640), focal: float | None = None, seed: int = 0, **kw) -> Render:
    """Render one of the test objects. Intrinsics follow Pix3D: principal point in the centre."""
    H, W = size
    f = focal or 1.1 * W
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])
    R, t = pose(azimuth, elevation, distance)
    return render(OBJECTS[name](), K, R, t, size=size, seed=seed, **kw)
