"""Evaluate the depth predictions of one method on the NYU subset.

    python scripts/eval_depth.py --method depth_anything_v2                 # metric model
    python scripts/eval_depth.py --method sfs --align affine --normals      # relative depth
    python scripts/eval_depth.py --method da_v2_relative --align affine --space inverse

Reads outputs/<method>/nyu/<id>.npy, writes results/depth/<method>.csv and regenerates
results/depth_summary.md. Metrics follow the usual NYU protocol: Eigen crop, ground truth
between 0.001 m and 10 m, prediction clipped to that range.
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from si3d import io, results
from si3d.datasets import NYUDepthV2
from si3d.datasets.nyu import EIGEN_CROP, K, MAX_DEPTH, MIN_DEPTH
from si3d.eval import align_pred, depth_metrics, eval_mask, normal_metrics, normals_from_depth
from si3d.eval.depth import METRIC_NAMES


def resize_to(pred: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if pred.shape == shape:
        return pred
    im = Image.fromarray(pred.astype(np.float32), mode="F")
    return np.asarray(im.resize((shape[1], shape[0]), Image.BILINEAR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="folder name under outputs/")
    ap.add_argument("--data-root", default="data/nyu")
    ap.add_argument("--subset", default="configs/subset_nyu.txt")
    ap.add_argument("--align", default="none", choices=["none", "median", "affine"])
    ap.add_argument("--space", default="depth", choices=["depth", "inverse"])
    ap.add_argument("--no-crop", action="store_true", help="evaluate on the full frame")
    ap.add_argument("--normals", action="store_true",
                    help="also compare normals (from <id>_normals.npy or derived from depth)")
    a = ap.parse_args()

    ds = NYUDepthV2(a.data_root)
    ids = io.read_subset(a.subset)
    crop = None if a.no_crop else EIGEN_CROP
    rows, missing = [], []
    for i in ids:
        pred = io.load_depth(a.method, i)
        if pred is None:
            missing.append(i)
            continue
        gt = ds.depth(i)
        pred = resize_to(pred, gt.shape)
        mask = eval_mask(gt, MIN_DEPTH, MAX_DEPTH, crop)
        pred_a = align_pred(pred, gt, mask, a.align, a.space)
        row = {"id": i, **depth_metrics(pred_a, gt, mask, MIN_DEPTH, MAX_DEPTH),
               "align": a.align, "space": a.space}
        if a.normals:
            pn = io.load_normals(a.method, i)
            if pn is None:
                pn = normals_from_depth(pred_a, K)
            gn = normals_from_depth(gt, K)
            row.update({f"n_{k}": v for k, v in normal_metrics(pn, gn, mask).items()})
        rows.append(row)

    if not rows:
        raise SystemExit(f"no predictions found under outputs/{a.method}/nyu/ for {a.subset}")
    results.write_csv("depth", a.method, rows)
    summary = results.summarize("depth")
    print(f"{a.method}: {len(rows)} frames evaluated, {len(missing)} missing")
    for m in METRIC_NAMES:
        print(f"  {m:9s} {np.mean([r[m] for r in rows]):.4f}")
    if a.normals:
        print(f"  normals: mean {np.mean([r['n_mean_deg'] for r in rows]):.2f} deg, "
              f"<11.25 deg {np.mean([r['n_p11'] for r in rows]):.3f}")
    if missing:
        print(f"  missing ids: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    print(f"summary table: {summary}")


if __name__ == "__main__":
    main()
