"""Evaluate the shape predictions of one method on the Pix3D subset.

    python scripts/eval_shape.py --method triposr                       # mesh in its own frame
    python scripts/eval_shape.py --method symmetry --gt-frame camera    # points in camera frame

Reads outputs/<method>/pix3d/<id>.(obj|ply|npy), writes results/shape/<method>.csv and
regenerates results/shape_summary.md. Prediction and ground truth are both normalised
(bounding-box centre and diagonal), then aligned by multi-start ICP unless --align none.
--gt-frame camera transforms the CAD model with the annotated pose first, which is the right
choice for methods that reconstruct in the camera frame.
"""
from __future__ import annotations

import argparse

import numpy as np

from si3d import io, results
from si3d.datasets import Pix3D
from si3d.eval import evaluate_shape
from si3d.eval.shape import SHAPE_METRIC_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="folder name under outputs/")
    ap.add_argument("--data-root", default="data/pix3d")
    ap.add_argument("--subset", default="configs/subset_pix3d.txt")
    ap.add_argument("--gt-frame", default="model", choices=["model", "camera"])
    ap.add_argument("--align", default="icp", choices=["icp", "none"])
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=0.05, help="F-score threshold, units of object size")
    a = ap.parse_args()

    ds = Pix3D(a.data_root)
    ids = io.read_subset(a.subset)
    rows, missing = [], []
    for i in ids:
        pred = io.load_shape(a.method, i)
        if pred is None:
            missing.append(i)
            continue
        gt = ds.mesh_in_camera(i) if a.gt_frame == "camera" else ds.mesh(i)
        m = evaluate_shape(pred, gt, n=a.n_points, tau=a.tau, align=a.align)
        rows.append({"id": i, "category": ds.entry(i)["category"], **m,
                     "align": a.align, "gt_frame": a.gt_frame})

    if not rows:
        raise SystemExit(f"no predictions found under outputs/{a.method}/pix3d/ for {a.subset}")
    results.write_csv("shape", a.method, rows)
    summary = results.summarize("shape")
    print(f"{a.method}: {len(rows)} objects evaluated, {len(missing)} missing")
    for m in SHAPE_METRIC_NAMES:
        print(f"  {m:13s} {np.mean([r[m] for r in rows]):.4f}")
    if missing:
        print(f"  missing ids: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    print(f"summary table: {summary}")


if __name__ == "__main__":
    main()
