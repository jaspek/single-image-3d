"""Check the camera model of the Pix3D loader on the data: project the CAD model of each
sample with the annotated pose and focal length and compare its bounding box with the
annotated one. The annotated 2D keypoints are not used, they are too noisy for this.

    python scripts/check_pix3d_projection.py [--n 300]

The convention set in src/si3d/datasets/pix3d.py should come out on top with a median
overlap close to 1.
"""
from __future__ import annotations

import argparse

import numpy as np

from si3d.datasets import Pix3D
from si3d.datasets.pix3d import CONVENTION, CONVENTIONS


def box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/pix3d")
    ap.add_argument("--n", type=int, default=300, help="number of samples to use")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = Pix3D(a.data_root)

    order = np.random.default_rng(a.seed).permutation(len(ds))
    overlap: dict[str, list[float]] = {c: [] for c in CONVENTIONS}
    vertices: dict[str, np.ndarray] = {}
    for i in order:
        e = ds.entry(int(i))
        if e.get("truncated"):
            continue                                    # the box is cut off by the image border
        if e["model"] not in vertices:
            vertices[e["model"]] = np.asarray(ds.mesh(int(i)).vertices)
        for c in CONVENTIONS:
            uv = ds.project(int(i), vertices[e["model"]], convention=c)
            box = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
            overlap[c].append(box_iou(box, e["bbox"]))
        if len(overlap[CONVENTION]) >= a.n:
            break

    print(f"{len(overlap[CONVENTION])} samples: overlap of the projected model box with the annotated box")
    print(f"{'convention':12s} {'median IoU':>11s} {'IoU > 0.8':>10s}")
    for c, v in sorted(overlap.items(), key=lambda kv: -np.median(kv[1])):
        mark = "  <- CONVENTION" if c == CONVENTION else ""
        print(f"{c:12s} {np.median(v):11.3f} {100 * np.mean(np.array(v) > 0.8):9.0f}%{mark}")
    best = max(overlap, key=lambda c: np.median(overlap[c]))
    if best != CONVENTION or np.median(overlap[best]) < 0.9:
        print(f"\nthe loader does not match the data: best convention is {best!r} with a median "
              f"overlap of {np.median(overlap[best]):.3f}; fix src/si3d/datasets/pix3d.py")
    else:
        print("\nloader and data agree")


if __name__ == "__main__":
    main()
