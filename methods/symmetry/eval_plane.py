"""Compare the detected symmetry planes with the symmetry of the Pix3D CAD models.

    python methods/symmetry/eval_plane.py [--name symmetry]

Reads results/<name>_status.csv (written by run.py) and writes results/<name>_plane.csv:
per image the number of symmetry planes of the CAD model, the true view angle and the
angle between the detected normal and the nearest model plane.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.symmetry.truth import true_planes  # noqa: E402
from si3d.datasets import Pix3D  # noqa: E402

VIEW_BINS = [(0, 10), (10, 25), (25, 45), (45, 75), (75, 90.01)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="symmetry")
    ap.add_argument("--data-root", default="data/pix3d")
    ap.add_argument("--tol-deg", type=float, default=5.0, help="a detected plane within this angle is correct")
    a = ap.parse_args()
    with open(Path("results") / f"{a.name}_status.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ds = Pix3D(a.data_root)

    out = []
    for r in rows:
        planes = true_planes(ds, int(r["id"]))
        o = {"id": int(r["id"]), "category": r["category"], "status": r["status"],
             "model_planes": len(planes), "true_view_deg": "", "plane_err_deg": ""}
        if planes:
            o["true_view_deg"] = round(planes[0][1], 1)           # the left-right plane
            if r.get("nx", "") != "":
                n_est = np.array([float(r["nx"]), float(r["ny"]), float(r["nz"])])
                err = [float(np.degrees(np.arccos(min(1.0, abs(float(n_est @ n)))))) for n, _ in planes]
                k = int(np.argmin(err))
                o["plane_err_deg"], o["true_view_deg"] = round(err[k], 1), round(planes[k][1], 1)
        out.append(o)

    path = Path("results") / f"{a.name}_plane.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    def correct(o: dict) -> bool:
        return o["plane_err_deg"] != "" and o["plane_err_deg"] <= a.tol_deg

    sym = [o for o in out if o["model_planes"] > 0]
    found = [o for o in out if o["status"] == "ok"]
    print(f"{len(out)} images, {len(sym)} with a symmetric CAD model")
    print(f"plane accepted in {len(found)} images, {sum(map(correct, found))} of them within "
          f"{a.tol_deg:g} deg of a symmetry plane of the model")
    print(f"not accepted although the best candidate was within {a.tol_deg:g} deg: "
          f"{sum(correct(o) for o in sym if o['status'] != 'ok')}")
    print(f"\n{'true view angle':>16s} {'images':>7s} {'accepted':>9s} {'correct':>8s}")
    for lo, hi in VIEW_BINS:
        sel = [o for o in sym if o["true_view_deg"] != "" and lo <= o["true_view_deg"] < hi]
        acc = [o for o in sel if o["status"] == "ok"]
        print(f"{lo:7.0f} - {min(hi, 90):3.0f} deg {len(sel):7d} {len(acc):9d} {sum(map(correct, acc)):8d}")
    print(f"\nper image: {path}")


if __name__ == "__main__":
    main()
