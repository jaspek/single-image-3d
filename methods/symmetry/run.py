"""Symmetry-based reconstruction on the Pix3D subset.

    python methods/symmetry/run.py                       # every id in configs/subset_pix3d.txt
    python methods/symmetry/run.py --ids 12 345 --vis    # selected images, with debug pictures
    python methods/symmetry/run.py --demo                # rendered test objects, no data needed

Writes
    outputs/symmetry/pix3d/<id>.npy         matched points and their mirror copies
    outputs/symmetry_hull/pix3d/<id>.npy    silhouette intersection with the mirrored view
    outputs/symmetry/vis/<id>.png           debug picture (--vis)
    results/symmetry_status.csv             one row per image: status and counts

Points are in the camera frame, up to scale. An image without a detected symmetry plane
gets no prediction file; the status file says why.
"""
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from dataclasses import fields
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.symmetry.pipeline import Params, Result, reconstruct  # noqa: E402
from si3d import io  # noqa: E402

STATUS_FILE = Path("results") / "symmetry_status.csv"
STATUS_COLUMNS = ["id", "category", "status", "n_keypoints", "n_matches", "n_inliers", "support",
                  "chance", "n_guided", "dense_ratio", "dense_null", "n_normals", "n_sparse",
                  "n_dense", "n_points", "n_hull", "view_angle_deg", "seconds"]


def save_figure(res: Result, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from methods.symmetry.vis import figure
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(res, title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_status(rows: dict[int, dict]) -> None:
    """Merge the new rows into results/symmetry_status.csv (keyed by image id)."""
    merged: dict[int, dict] = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE, newline="", encoding="utf-8") as f:
            merged = {int(r["id"]): r for r in csv.DictReader(f)}
    merged.update(rows)
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for i in sorted(merged):
            w.writerow({c: merged[i].get(c, "") for c in STATUS_COLUMNS})


def run_dataset(a: argparse.Namespace, params: Params) -> None:
    from si3d.datasets import Pix3D
    try:
        ds = Pix3D(a.data_root)
        ids = a.ids if a.ids else io.read_subset(a.subset)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\n(data/README.md: download Pix3D, then scripts/make_subsets.py; "
                         f"or try --demo)")
    rows: dict[int, dict] = {}
    for i in tqdm(ids):
        try:
            image, mask = ds.image(i), ds.mask(i)
            if mask.shape != image.shape[:2]:               # mask stored at another resolution
                ys = np.arange(image.shape[0]) * mask.shape[0] // image.shape[0]
                xs = np.arange(image.shape[1]) * mask.shape[1] // image.shape[1]
                mask = mask[ys][:, xs]
            res = reconstruct(image, mask, ds.intrinsics(i), params)
        except Exception:                                   # keep going, the failure is a result too
            traceback.print_exc()
            res = Result("error")
        for method, pts in (("symmetry", res.points), ("symmetry_hull", res.hull)):
            old = io.pred_dir(method, "pix3d") / f"{i:04d}.npy"
            if res.status == "ok" and len(pts):
                io.save_shape(method, i, pts)
            elif old.exists():
                old.unlink()                                # no stale prediction from an earlier run
        rows[i] = {"id": i, "category": ds.entry(i)["category"], "status": res.status, **res.info}
        if a.vis:
            save_figure(res, io.OUTPUTS / "symmetry" / "vis" / f"{i:04d}.png",
                        f"Pix3D {i} ({ds.entry(i)['category']})")
    write_status(rows)
    ok = sum(r["status"] == "ok" for r in rows.values())
    print(f"{ok} of {len(rows)} images reconstructed; status per image in {STATUS_FILE}")
    for status in sorted({r["status"] for r in rows.values()}):
        print(f"  {status:13s} {sum(r['status'] == status for r in rows.values())}")


def run_demo(a: argparse.Namespace, params: Params) -> None:
    """Rendered objects with a known symmetry plane and shape."""
    from methods.symmetry import synthetic
    from si3d.eval import evaluate_shape
    print(f"{'object':8s} {'azimuth':>7s} {'status':12s} {'normal err':>10s} {'accuracy':>9s} "
          f"{'complet.':>9s} {'F-score':>8s} {'hull F':>7s}")
    for name in a.demo_objects:
        for az in a.demo_azimuths:
            r = synthetic.make(name, azimuth=az)
            res = reconstruct(r.image, r.mask, r.K, params)
            line = f"{name:8s} {az:7.0f} {res.status:12s}"
            if res.status == "ok":
                n_true, _ = r.plane
                err = np.degrees(np.arccos(min(1.0, abs(float(res.normal @ n_true)))))
                gt = r.mesh_in_camera()
                m = evaluate_shape(res.points, gt, align="camera")
                line += f" {err:8.2f} deg {m['accuracy']:9.4f} {m['completeness']:9.4f} {m['fscore']:8.3f}"
                if len(res.hull):
                    line += f" {evaluate_shape(res.hull, gt, align='camera')['fscore']:7.3f}"
            print(line)
            if a.vis:
                save_figure(res, io.OUTPUTS / "symmetry" / "vis" / f"demo_{name}_{az:.0f}.png",
                            f"{name}, azimuth {az:.0f} deg")
    if a.vis:
        print(f"pictures in {io.OUTPUTS / 'symmetry' / 'vis'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="configs/subset_pix3d.txt")
    ap.add_argument("--data-root", default="data/pix3d")
    ap.add_argument("--ids", type=int, nargs="*", help="run these image ids instead of the subset")
    ap.add_argument("--vis", action="store_true", help="save a debug picture per image")
    ap.add_argument("--demo", action="store_true", help="run on rendered test objects")
    ap.add_argument("--demo-objects", nargs="*", default=["chair", "table", "cabinet"])
    ap.add_argument("--demo-azimuths", type=float, nargs="*", default=[15.0, 30.0, 45.0])
    ap.add_argument("--no-dense", action="store_true", help="feature pairs only")
    ap.add_argument("--no-hull", action="store_true", help="skip the silhouette intersection")
    kinds = {"int": int, "float": float}
    for f in fields(Params):                                # every numeric parameter as a flag
        if f.type in kinds:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=kinds[f.type], default=f.default)
    a = ap.parse_args()
    numeric = {f.name: getattr(a, f.name) for f in fields(Params) if f.type in kinds}
    params = Params(**numeric, dense=not a.no_dense, hull=not a.no_hull)
    (run_demo if a.demo else run_dataset)(a, params)


if __name__ == "__main__":
    main()
