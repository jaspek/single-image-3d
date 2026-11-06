# Single-image 3D reconstruction

Proseminar project, Imaging Beyond Consumer Cameras (911.422), University of Salzburg,
winter term 2026/27. Two classical methods implemented from scratch against two pretrained
networks, on the same images, with the same metrics. The question is where geometric priors
and learned priors each break down.

|  | classical (from scratch) | deep learning (pretrained) | data | metric |
|---|---|---|---|---|
| scenes | shape from shading (Tsai-Shah 1994) | Depth Anything V2, metric indoor | NYU Depth v2 | depth error, normal error |
| objects | symmetry as a second view (Hong et al. 2004) | TripoSR (2024) | Pix3D | Chamfer distance, F-score |

Every method writes predictions in one shared layout; one evaluation script per dataset
reads them and keeps the result tables up to date. Nobody has to run anybody else's code.

## Layout

```
src/si3d/            shared package: dataset loaders, metrics, prediction I/O, result tables
scripts/             make_subsets, eval_depth, eval_shape, show_sample, check_pix3d_projection
methods/<name>/      one folder per method: README + run.py that writes predictions
configs/             the fixed evaluation subsets (committed)
results/             per-sample CSVs and the summary tables (committed)
data/                datasets (not committed), see data/README.md
outputs/             predictions and visualisations (not committed)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -e .[dev]
pytest                            # metric sanity tests, no data needed
```

Then get the data (`data/README.md`), run `python scripts/make_subsets.py` once and commit
`configs/subset_*.txt`. The learned methods need PyTorch and their official repos under
`external/` (see the method READMEs); the shared package and the evaluation do not.

## Workflow

1. Run a method on the subset: `python methods/<name>/run.py` writes `outputs/<name>/...`.
2. Evaluate: `python scripts/eval_depth.py --method <name>` or `scripts/eval_shape.py`.
   This writes `results/<kind>/<name>.csv` and regenerates `results/<kind>_summary.md`.
3. Look: `python scripts/show_sample.py nyu 12 --methods sfs depth_anything_v2`.
4. Commit the CSV, the summary and the method code. Never commit `data/` or `outputs/`.

The prediction contract (file names, shapes, units) is in `methods/README.md`. Relative
depth is fine, say so in the method README and evaluate with `--align affine`.

## Conventions

- Ids are 0-based indices into `nyu_depth_v2_labeled.mat` and `pix3d.json`.
- Depth in metres, (480, 640) float32 for NYU. Camera frame: x right, y down, z forward.
- NYU evaluation uses the Eigen crop and ground truth between 0.001 m and 10 m.
- Pix3D shapes are normalised to unit bounding-box diagonal before comparison; distances are
  in units of object size. `--gt-frame camera` for methods that reconstruct in the camera frame.
- One method name = one folder under `outputs/` = one row in the summary table. A variant
  (fine-tuned, different resolution) is a new method name, e.g. `depth_anything_v2_nyuft`.

## Timeline

- Next progress meeting: datasets downloaded, subsets committed, every method run on a
  handful of images with one picture and one number each.
- Then: full subsets, the own modification of the learned methods, failure analysis.
- Final presentation at the last proseminar date.
