"""Shape from shading on the NYU subset. Writes outputs/sfs/nyu/<id>.npy (relative depth)
and <id>_normals.npy.

    python methods/sfs/run.py --subset configs/subset_nyu.txt [--light-tilt 0 --light-slant 30]
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from si3d import io
from si3d.datasets import NYUDepthV2


def tsai_shah(gray: np.ndarray, tilt_deg: float, slant_deg: float, iters: int = 200) -> np.ndarray:
    """Return relative depth Z (H, W) from a grayscale image in [0, 1].

    TODO: implement the linear-approximation iteration of Tsai & Shah 1994.
    """
    raise NotImplementedError


def normals_from_z(z: np.ndarray) -> np.ndarray:
    p = np.gradient(z, axis=1)
    q = np.gradient(z, axis=0)
    n = np.stack([-p, -q, -np.ones_like(z)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset_nyu.txt")
    ap.add_argument("--data-root", default="data/nyu")
    ap.add_argument("--light-tilt", type=float, default=0.0)
    ap.add_argument("--light-slant", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    ds = NYUDepthV2(a.data_root)
    for i in tqdm(io.read_subset(a.subset)):
        gray = ds.image(i).astype(np.float32).mean(axis=-1) / 255.0
        z = tsai_shah(gray, a.light_tilt, a.light_slant, a.iters)
        io.save_depth("sfs", i, z, normals_from_z(z))


if __name__ == "__main__":
    main()
