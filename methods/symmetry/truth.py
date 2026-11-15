"""Symmetry planes of the Pix3D CAD models in the camera frame, from the annotated pose.
Used to judge the detected plane (eval_plane.py) and to run the reconstruction with the
true plane (run.py --true-plane), never by the method itself.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from si3d.datasets import Pix3D
from si3d.datasets.pix3d import CONVENTION, CONVENTIONS
from si3d.eval.shape import as_points

_cache: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}


def model_symmetry(mesh, n: int = 20000, factor: float = 2.5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Symmetry planes of a mesh among the three coordinate planes through its bounding-box
    centre, as (normal, point on the plane). The reflected surface has to be as close to the
    surface as a second sampling of the surface itself, up to `factor`."""
    a, b = as_points(mesh, n, 0), as_points(mesh, n, 1)
    tree = cKDTree(a)
    noise = tree.query(b)[0].mean()                       # distance between two samplings
    centre = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    planes = []
    for axis in np.eye(3):
        mirrored = b - 2.0 * ((b - centre) @ axis)[:, None] * axis
        if tree.query(mirrored)[0].mean() <= factor * noise:
            planes.append((axis, centre))
    return planes


def true_planes(ds: Pix3D, i: int) -> list[tuple[np.ndarray, float]]:
    """Symmetry planes of sample i in the camera frame as (unit normal n, view angle in
    degrees), n pointing away from the camera. The view angle is the angle between the ray
    to the object centre and the plane: 0 is a frontal view, 90 a view from the side.
    Sorted so that the left-right plane of a typical photo comes first: planes with a roughly
    horizontal normal before the others, then by view angle."""
    model = ds.entry(i)["model"]
    if model not in _cache:
        _cache[model] = model_symmetry(ds.mesh(i))
    flip = CONVENTIONS[CONVENTION]
    R, t = ds.pose(i)
    out = []
    for axis, centre in _cache[model]:
        n = flip @ R @ axis
        c = flip @ (R @ centre + t)                       # object centre in the camera frame
        if n @ c < 0:
            n = -n
        view = float(np.degrees(np.arcsin(min(1.0, float(n @ c) / np.linalg.norm(c)))))
        out.append((n, view))
    return sorted(out, key=lambda p: (abs(p[0][1]) > 0.5, p[1]))
