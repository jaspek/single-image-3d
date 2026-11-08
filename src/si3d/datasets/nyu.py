"""NYU Depth v2, labeled subset (1449 aligned RGB + depth frames, 640x480).

Expected files in the data root (see data/README.md):
    nyu_depth_v2_labeled.mat   images, depths (inpainted, metres), rawDepths, scenes
    splits.mat                 official train (795) / test (654) indices (1-based)

Frame ids used everywhere in this repo are 0-based indices into the .mat file.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io

# Kinect intrinsics from the NYU toolbox (camera_params.m), for the 640x480 frames.
FX, FY = 518.857901, 519.469611
CX, CY = 325.582449, 253.736166
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)

# Standard evaluation crop (Eigen et al.): rows 45:471, cols 41:601, 0-based, end exclusive.
EIGEN_CROP = (45, 471, 41, 601)
MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0
HEIGHT, WIDTH = 480, 640

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class NYUDepthV2:
    def __init__(self, root: str | Path = "data/nyu"):
        self.root = Path(root)
        self.mat_path = self.root / "nyu_depth_v2_labeled.mat"
        self.splits_path = self.root / "splits.mat"
        if not self.mat_path.exists():
            raise FileNotFoundError(f"missing {self.mat_path}, see data/README.md")
        self._f: h5py.File | None = None

    # -- file handling -----------------------------------------------------
    def _file(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.mat_path, "r")
        return self._f

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __len__(self) -> int:
        return int(self._file()["images"].shape[0])

    def ids(self) -> list[int]:
        return list(range(len(self)))

    # -- data access ---------------------------------------------------------
    def image(self, i: int) -> np.ndarray:
        """RGB uint8 array of shape (480, 640, 3)."""
        img = self._file()["images"][i]  # stored as (3, 640, 480)
        return np.ascontiguousarray(np.transpose(img, (2, 1, 0)))

    def depth(self, i: int) -> np.ndarray:
        """Inpainted depth in metres, float32 (480, 640). This is the usual ground truth."""
        d = self._file()["depths"][i]  # stored as (640, 480)
        return np.ascontiguousarray(np.transpose(d, (1, 0)).astype(np.float32))

    def raw_depth(self, i: int) -> np.ndarray:
        """Raw Kinect depth in metres, 0 where the sensor had no reading."""
        d = self._file()["rawDepths"][i]
        return np.ascontiguousarray(np.transpose(d, (1, 0)).astype(np.float32))

    def scene(self, i: int) -> str:
        f = self._file()
        ref = f["scenes"][0][i]
        chars = f[ref][:]
        return "".join(chr(int(c)) for c in np.asarray(chars).ravel())

    def splits(self) -> dict[str, np.ndarray]:
        """Official split as 0-based index arrays: {'train': ..., 'test': ...}."""
        if not self.splits_path.exists():
            raise FileNotFoundError(f"missing {self.splits_path}, see data/README.md")
        s = scipy.io.loadmat(self.splits_path)
        return {
            "train": np.asarray(s["trainNdxs"]).ravel().astype(int) - 1,
            "test": np.asarray(s["testNdxs"]).ravel().astype(int) - 1,
        }

    @staticmethod
    def intrinsics() -> np.ndarray:
        return K.copy()
