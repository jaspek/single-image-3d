"""Pix3D: 10 069 real images of 395 furniture CAD models with mask, pose and focal length.

Expected layout in the data root (see data/README.md):
    pix3d.json          list of annotations
    img/ mask/ model/   referenced by relative paths inside pix3d.json

Sample ids used everywhere in this repo are 0-based indices into pix3d.json.

Camera model. Pix3D gives `rot_mat` (3x3) and `trans_mat` (3,) that map model coordinates
to camera coordinates, and `focal_length` in mm for a 35 mm-film camera with a 32 mm wide
sensor (the Blender default used to render the ground truth). Pixel focal length is
therefore f_px = focal_length * image_width / 32. The axis convention of the camera frame
is selected by `CONVENTION`; run scripts/check_pix3d_projection.py once after downloading
the data: it reprojects the annotated 3D keypoints onto the 2D keypoints and reports which
convention is right.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io
import trimesh
from PIL import Image

SENSOR_WIDTH_MM = 32.0

# Candidate axis flips applied to camera coordinates before the pinhole projection.
CONVENTIONS = {
    "opencv": np.diag([1.0, 1.0, 1.0]),      # x right, y down, z forward
    "flip_yz": np.diag([1.0, -1.0, -1.0]),   # x right, y up, z backward (OpenGL / Blender)
    "flip_z": np.diag([1.0, 1.0, -1.0]),
    "flip_y": np.diag([1.0, -1.0, 1.0]),
}
# Set after running scripts/check_pix3d_projection.py.
CONVENTION = "opencv"


class Pix3D:
    def __init__(self, root: str | Path = "data/pix3d"):
        self.root = Path(root)
        meta_path = self.root / "pix3d.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing {meta_path}, see data/README.md")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: list[dict] = json.load(f)

    def __len__(self) -> int:
        return len(self.meta)

    def ids(self) -> list[int]:
        return list(range(len(self)))

    def entry(self, i: int) -> dict:
        return self.meta[i]

    # -- images --------------------------------------------------------------
    def image(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / self.meta[i]["img"]).convert("RGB"))

    def mask(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / self.meta[i]["mask"]).convert("L")) > 127

    # -- geometry ------------------------------------------------------------
    def mesh(self, i: int) -> trimesh.Trimesh:
        """Ground-truth CAD model in model coordinates."""
        return trimesh.load(self.root / self.meta[i]["model"], force="mesh")

    def voxel(self, i: int) -> np.ndarray:
        return np.asarray(scipy.io.loadmat(self.root / self.meta[i]["voxel"])["voxel"])

    def pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        e = self.meta[i]
        return (np.asarray(e["rot_mat"], dtype=np.float64),
                np.asarray(e["trans_mat"], dtype=np.float64))

    def intrinsics(self, i: int) -> np.ndarray:
        e = self.meta[i]
        w, h = e["img_size"]
        f = float(e["focal_length"]) * w / SENSOR_WIDTH_MM
        return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])

    def keypoints3d(self, i: int) -> np.ndarray | None:
        v = self.meta[i].get("3d_keypoints")
        if v is None:
            return None
        if isinstance(v, str):
            return np.loadtxt(self.root / v, ndmin=2)
        return np.asarray(v, dtype=np.float64)

    def keypoints2d(self, i: int) -> np.ndarray | None:
        """Mean over annotators, NaN where a keypoint was not annotated (-1 in the json)."""
        v = self.meta[i].get("2d_keypoints")
        if not v:
            return None
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 2:
            a = a[None]
        a = np.where(a < 0, np.nan, a)
        with np.errstate(invalid="ignore"):
            return np.nanmean(a, axis=0)

    def to_camera(self, i: int, points: np.ndarray) -> np.ndarray:
        """Model coordinates -> camera coordinates (before any axis-convention flip)."""
        R, t = self.pose(i)
        return points @ R.T + t

    def project(self, i: int, points: np.ndarray, convention: str = CONVENTION) -> np.ndarray:
        """Model coordinates -> pixel coordinates (u, v)."""
        Xc = self.to_camera(i, points) @ CONVENTIONS[convention].T
        K = self.intrinsics(i)
        z = Xc[:, 2:3]
        uv = (Xc[:, :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
        return uv

    def mesh_in_camera(self, i: int, convention: str = CONVENTION) -> trimesh.Trimesh:
        m = self.mesh(i).copy()
        m.vertices = self.to_camera(i, m.vertices) @ CONVENTIONS[convention].T
        return m
