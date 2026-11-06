"""Side-by-side picture of one sample: input, ground truth and the predictions of the
given methods. Saved to outputs/vis/<dataset>_<id>.png.

    python scripts/show_sample.py nyu 12 --methods sfs depth_anything_v2
    python scripts/show_sample.py pix3d 345 --methods symmetry triposr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from si3d import io  # noqa: E402
from si3d.datasets import NYUDepthV2, Pix3D  # noqa: E402
from si3d.eval.shape import as_points, normalize_points  # noqa: E402


def show_nyu(i: int, methods: list[str], root: str) -> plt.Figure:
    ds = NYUDepthV2(root)
    img, gt = ds.image(i), ds.depth(i)
    preds = [(m, io.load_depth(m, i)) for m in methods]
    fig, axes = plt.subplots(1, 2 + len(preds), figsize=(4 * (2 + len(preds)), 3.4))
    axes[0].imshow(img); axes[0].set_title(f"NYU {i}: {ds.scene(i)}")
    axes[1].imshow(gt, cmap="turbo"); axes[1].set_title("ground truth depth [m]")
    for ax, (m, p) in zip(axes[2:], preds):
        if p is None:
            ax.set_title(f"{m}: no prediction")
        else:
            ax.imshow(p, cmap="turbo"); ax.set_title(m)
    for ax in axes:
        ax.axis("off")
    return fig


def show_pix3d(i: int, methods: list[str], root: str) -> plt.Figure:
    ds = Pix3D(root)
    img, mask = ds.image(i), ds.mask(i)
    gt, _ = normalize_points(as_points(ds.mesh(i)))
    preds = [(m, io.load_shape(m, i)) for m in methods]
    fig, axes = plt.subplots(1, 2 + len(preds), figsize=(4 * (2 + len(preds)), 3.6))
    over = img.copy()
    over[~mask] = (0.4 * over[~mask]).astype(np.uint8)
    axes[0].imshow(over); axes[0].set_title(f"Pix3D {i}: {ds.entry(i)['category']}")
    axes[0].axis("off")
    axes[1].scatter(gt[:, 0], gt[:, 1], s=0.3); axes[1].set_title("ground truth (x, y)")
    for ax, (m, p) in zip(axes[2:], preds):
        if p is None:
            ax.set_title(f"{m}: no prediction")
            continue
        q, _ = normalize_points(as_points(p))
        ax.scatter(q[:, 0], q[:, 1], s=0.3); ax.set_title(f"{m} (x, y), own frame")
    for ax in axes[1:]:
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["nyu", "pix3d"])
    ap.add_argument("id", type=int)
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    root = a.data_root or f"data/{a.dataset}"
    fig = show_nyu(a.id, a.methods, root) if a.dataset == "nyu" else show_pix3d(a.id, a.methods, root)
    out = Path("outputs/vis")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.dataset}_{a.id:04d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
