"""Find the camera-axis convention of Pix3D by reprojecting the annotated 3D keypoints onto
the annotated 2D keypoints. Run once after downloading the data, then set `CONVENTION` in
src/si3d/datasets/pix3d.py to the winner.

    python scripts/check_pix3d_projection.py [--n 300]
"""
from __future__ import annotations

import argparse

import numpy as np

from si3d.datasets import Pix3D
from si3d.datasets.pix3d import CONVENTIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/pix3d")
    ap.add_argument("--n", type=int, default=300, help="number of annotated samples to use")
    a = ap.parse_args()
    ds = Pix3D(a.data_root)

    errors: dict[str, list[float]] = {c: [] for c in CONVENTIONS}
    used = 0
    for i in range(len(ds)):
        k3, k2 = ds.keypoints3d(i), ds.keypoints2d(i)
        if k3 is None or k2 is None or k3.shape[0] != k2.shape[0]:
            continue
        ok = np.all(np.isfinite(k2), axis=1)
        if ok.sum() < 3:
            continue
        for c in CONVENTIONS:
            uv = ds.project(i, k3[ok], convention=c)
            errors[c].append(float(np.mean(np.linalg.norm(uv - k2[ok], axis=1))))
        used += 1
        if used >= a.n:
            break

    if used == 0:
        raise SystemExit("no samples with both 2D and 3D keypoints found")
    print(f"{used} samples with 2D/3D keypoints")
    print(f"{'convention':12s} {'mean px':>10s} {'median px':>10s} {'best in %':>10s}")
    stacked = np.array([errors[c] for c in CONVENTIONS])
    best = np.argmin(stacked, axis=0)
    for j, c in enumerate(CONVENTIONS):
        e = np.array(errors[c])
        print(f"{c:12s} {e.mean():10.2f} {np.median(e):10.2f} {100 * np.mean(best == j):10.1f}")
    winner = list(CONVENTIONS)[int(np.argmin(np.median(stacked, axis=1)))]
    print(f"\n-> set CONVENTION = {winner!r} in src/si3d/datasets/pix3d.py "
          f"(a median error of a few pixels is expected; tens of pixels means something is off)")


if __name__ == "__main__":
    main()
