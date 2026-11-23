"""Symmetry-based reconstruction on the Pix3D subset.

    python methods/symmetry/run.py                       # every id in configs/subset_pix3d.txt
    python methods/symmetry/run.py --ids 12 345 --vis    # selected images, with debug pictures
    python methods/symmetry/run.py --demo                # rendered test objects, no data needed
    python methods/symmetry/run.py --demo --demo-plain   # the same objects without texture

The normal of the symmetry plane comes from the straight edges of the object (--plane-from
lines, the default on Pix3D), from mirrored feature pairs (features), or from the edges
with the feature pairs as a fallback (auto, the default of --demo).

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

STATUS_COLUMNS = ["id", "category", "status", "plane_source", "line_reject", "n_segments", "frame_lines",
                  "frame_explained", "normal_sigma_deg", "normal_agree_deg", "box", "box_misfit", "n_planes",
                  "coverage_depth", "coverage_up",
                  "n_keypoints", "n_matches", "n_inliers", "support",
                  "support_firm", "chance", "n_guided", "dense_ratio", "dense_null", "n_normals", "n_sparse",
                  "n_dense", "n_points", "n_hull", "view_angle_deg", "nx", "ny", "nz", "seconds"]


def save_figure(res: Result, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from methods.symmetry.vis import figure
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = figure(res, title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_status(path: Path, rows: dict[int, dict]) -> None:
    """Merge the new rows into the status file (keyed by image id)."""
    merged: dict[int, dict] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            merged = {int(r["id"]): r for r in csv.DictReader(f)}
    merged.update(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for i in sorted(merged):
            w.writerow({c: merged[i].get(c, "") for c in STATUS_COLUMNS})


def run_dataset(a: argparse.Namespace, params: Params) -> None:
    from methods.symmetry.truth import true_planes
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
            normal = None
            if a.true_plane:                                # analysis: plane from the CAD model
                planes = true_planes(ds, i)
                if not planes:
                    rows[i] = {"id": i, "category": ds.entry(i)["category"], "status": "model_not_symmetric"}
                    continue
                normal = planes[0][0]
            res = reconstruct(image, mask, ds.intrinsics(i), params, normal=normal)
        except Exception:                                   # keep going, the failure is a result too
            traceback.print_exc()
            res = Result("error")
        for method, pts in ((a.name, res.points), (a.name + "_hull", res.hull)):
            old = io.pred_dir(method, "pix3d") / f"{i:04d}.npy"
            if res.status == "ok" and len(pts):
                io.save_shape(method, i, pts)
            elif old.exists():
                old.unlink()                                # no stale prediction from an earlier run
        rows[i] = {"id": i, "category": ds.entry(i)["category"], "status": res.status, **res.info}
        if a.vis:
            save_figure(res, io.OUTPUTS / a.name / "vis" / f"{i:04d}.png",
                        f"Pix3D {i} ({ds.entry(i)['category']})")
    status_file = Path("results") / f"{a.name}_status.csv"
    write_status(status_file, rows)
    ok = sum(r["status"] == "ok" for r in rows.values())
    print(f"{ok} of {len(rows)} images reconstructed; status per image in {status_file}")
    for status in sorted({r["status"] for r in rows.values()}):
        print(f"  {status:13s} {sum(r['status'] == status for r in rows.values())}")


def run_demo(a: argparse.Namespace, params: Params) -> None:
    """Rendered objects with a known symmetry plane and shape. The error of the normal is
    measured to the nearest symmetry plane of the object (the table has two, the cabinet three)."""
    from methods.symmetry import synthetic
    from methods.symmetry.truth import model_symmetry
    from si3d.eval import evaluate_shape
    print(f"{'object':8s} {'azimuth':>7s} {'status':12s} {'plane from':10s} {'normal err':>10s} {'accuracy':>9s} "
          f"{'complet.':>9s} {'F-score':>8s} {'hull F':>7s} {'box':>4s}")
    for name in a.demo_objects:
        axes = [axis for axis, _ in model_symmetry(synthetic.OBJECTS[name]())]
        for az in a.demo_azimuths:
            r = synthetic.make(name, azimuth=az, plain=a.demo_plain)
            res = reconstruct(r.image, r.mask, r.K, params)
            line = f"{name:8s} {az:7.0f} {res.status:12s}"
            if res.status == "ok":
                err = min(np.degrees(np.arccos(min(1.0, abs(float(res.normal @ r.R @ axis))))) for axis in axes)
                gt = r.mesh_in_camera()
                source = res.info.get("plane_source", "")
                line += f" {source:10s} " + (f"{err:6.2f} deg" if source != "frame" else f"{'-':>10s}")
                if len(res.points):
                    m = evaluate_shape(res.points, gt, align="camera")
                    line += f" {m['accuracy']:9.4f} {m['completeness']:9.4f} {m['fscore']:8.3f}"
                else:
                    line += f" {'-':>9s} {'-':>9s} {'-':>8s}"
                if len(res.hull):
                    line += (f" {evaluate_shape(res.hull, gt, align='camera')['fscore']:7.3f}"
                             f" {'yes' if res.info.get('box') else 'no':>4s}")
            print(line)
            if a.vis:
                save_figure(res, io.OUTPUTS / a.name / "vis" / f"demo_{name}_{az:.0f}.png",
                            f"{name}, azimuth {az:.0f} deg")
    if a.vis:
        print(f"pictures in {io.OUTPUTS / a.name / 'vis'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="configs/subset_pix3d.txt")
    ap.add_argument("--data-root", default="data/pix3d")
    ap.add_argument("--ids", type=int, nargs="*", help="run these image ids instead of the subset")
    ap.add_argument("--name", default="symmetry", help="method name of the outputs, for a variant "
                    "(writes outputs/<name>, outputs/<name>_hull, results/<name>_status.csv)")
    ap.add_argument("--vis", action="store_true", help="save a debug picture per image")
    ap.add_argument("--demo", action="store_true", help="run on rendered test objects")
    ap.add_argument("--demo-objects", nargs="*", default=["chair", "table", "cabinet"])
    ap.add_argument("--demo-azimuths", type=float, nargs="*", default=[15.0, 30.0, 45.0])
    ap.add_argument("--demo-plain", action="store_true",
                    help="render the test objects without texture: only shaded faces and the outline")
    ap.add_argument("--plane-from", choices=["lines", "features", "auto"], default=None,
                    help="source of the plane normal: straight edges, mirrored feature pairs, or edges "
                         "with the feature pairs as fallback (default: lines for Pix3D, auto for --demo)")
    ap.add_argument("--dense", action=argparse.BooleanOptionalAction, default=None,
                    help="dense search along the epipolar lines (default: on for --demo, off for "
                         "Pix3D, where most of its matches are wrong)")
    ap.add_argument("--true-plane", action="store_true",
                    help="analysis: take the symmetry plane from the CAD model and the annotated "
                         "pose instead of detecting it (use with --name)")
    ap.add_argument("--no-hull", action="store_true", help="skip the silhouette intersection")
    ap.add_argument("--no-box", action="store_true",
                    help="do not cut the silhouette intersection down to the box of the edge frame")
    ap.add_argument("--box-only", action="store_true",
                    help="no symmetry: the shape is the box of the edge frame within the outline "
                         "(use with --name; the mirrored camera is not used and frontal views are kept)")
    ap.add_argument("--more-planes", action="store_true",
                    help="test the other axes of the edge frame as further symmetry planes")
    kinds = {"int": int, "float": float}
    for f in fields(Params):                                # every numeric parameter as a flag
        if f.type in kinds:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=kinds[f.type], default=f.default)
    a = ap.parse_args()
    numeric = {f.name: getattr(a, f.name) for f in fields(Params) if f.type in kinds}
    dense = a.demo if a.dense is None else a.dense
    plane_from = a.plane_from or ("auto" if a.demo else "lines")
    params = Params(**numeric, plane_from=plane_from, dense=dense, hull=not a.no_hull, box=not a.no_box,
                    symmetry=not a.box_only, more_planes=a.more_planes)
    (run_demo if a.demo else run_dataset)(a, params)


if __name__ == "__main__":
    main()
