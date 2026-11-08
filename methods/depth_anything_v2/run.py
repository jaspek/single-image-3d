"""Depth Anything V2 metric depth on the NYU subset. Writes outputs/<method>/nyu/<id>.npy
in metres.

    python methods/depth_anything_v2/run.py --encoder vits [--checkpoint ...] [--method depth_anything_v2]

Needs the official repo under external/Depth-Anything-V2 (see README.md in this folder).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch


from si3d import io
from si3d.datasets import NYUDepthV2

CURRENT_DIR = Path(__file__).resolve().parent.parent
ROOT = CURRENT_DIR.parent
REPO = ROOT / "external" / "Depth-Anything-V2" / "metric_depth"
CHECKPOINTS = REPO.parent / "checkpoints"
CONFIGS = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}



def load_model(encoder: str, checkpoint: Path, max_depth: float, device: str):
    sys.path.insert(0, str(REPO))
    from depth_anything_v2.dpt import DepthAnythingV2 
    model = DepthAnythingV2(**CONFIGS[encoder], max_depth=max_depth)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="configs/subset_nyu.txt")
    ap.add_argument("--data-root", default=f"{ROOT}/data/nyu")
    ap.add_argument("--encoder", default="vits", choices=list(CONFIGS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-depth", type=float, default=20.0, help="20 for the indoor (Hypersim) models")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--method", default="depth_anything_v2", help="output folder name")
    a = ap.parse_args()

    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    ckpt = Path(a.checkpoint or f"{CHECKPOINTS}/depth_anything_v2_metric_hypersim_{a.encoder}.pth")
    model = load_model(a.encoder, ckpt, a.max_depth, device)

    ds = NYUDepthV2(a.data_root)
    for i in tqdm(io.read_subset(a.subset)):
        bgr = np.ascontiguousarray(ds.image(i)[:, :, ::-1])  # the repo's infer_image expects BGR
        with torch.no_grad():
            depth = model.infer_image(bgr, input_size=a.input_size)
        io.save_depth(a.method, i, depth)


if __name__ == "__main__":
    main()
