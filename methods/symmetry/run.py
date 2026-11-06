"""Symmetry-based reconstruction on the Pix3D subset. Writes outputs/symmetry/pix3d/<id>.npy
(sparse points, camera frame, up to scale).

    python methods/symmetry/run.py --subset configs/subset_pix3d.txt
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from si3d import io
from si3d.datasets import Pix3D


def reconstruct(image: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Return (N, 3) points in the camera frame.

    TODO: features on image and mirrored image, symmetric matches, mirror pose, triangulation
    (Hong et al. 2004).
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset_pix3d.txt")
    ap.add_argument("--data-root", default="data/pix3d")
    a = ap.parse_args()

    ds = Pix3D(a.data_root)
    for i in tqdm(io.read_subset(a.subset)):
        pts = reconstruct(ds.image(i), ds.mask(i), ds.intrinsics(i))
        io.save_shape("symmetry", i, pts)


if __name__ == "__main__":
    main()
