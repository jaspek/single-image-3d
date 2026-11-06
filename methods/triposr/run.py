"""TripoSR on the Pix3D subset. Writes outputs/triposr/pix3d/<id>.obj.

    python methods/triposr/run.py --subset configs/subset_pix3d.txt [--mc-resolution 256]

Needs the official repo under external/TripoSR (see README.md in this folder).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from si3d import io
from si3d.datasets import Pix3D

REPO = Path("external/TripoSR")


def prepare_input(image: np.ndarray, mask: np.ndarray, ratio: float = 0.85) -> Image.Image:
    """Masked object on 50 % grey, square, object filling `ratio` of the side."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = image[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1][..., None]
    crop = crop * m + 127.5 * (1 - m)
    h, w = crop.shape[:2]
    side = int(np.ceil(max(h, w) / ratio))
    canvas = np.full((side, side, 3), 127.5, dtype=np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    return Image.fromarray(canvas.astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset_pix3d.txt")
    ap.add_argument("--data-root", default="data/pix3d")
    ap.add_argument("--mc-resolution", type=int, default=256)
    ap.add_argument("--work", default="outputs/triposr/_work")
    a = ap.parse_args()

    ds = Pix3D(a.data_root)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = io.pred_dir("triposr", "pix3d")
    for i in tqdm(io.read_subset(a.subset)):
        inp = work / f"{i:04d}.png"
        prepare_input(ds.image(i), ds.mask(i)).save(inp)
        run_out = work / f"{i:04d}"
        cmd = [sys.executable, str(REPO / "run.py"), str(inp), "--output-dir", str(run_out),
               "--no-remove-bg", "--model-save-format", "obj",
               "--mc-resolution", str(a.mc_resolution)]
        subprocess.run(cmd, check=True)
        mesh = next(run_out.rglob("mesh.obj"))
        shutil.copy(mesh, out_dir / f"{i:04d}.obj")


if __name__ == "__main__":
    main()
