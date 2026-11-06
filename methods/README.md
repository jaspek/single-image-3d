# Methods

One folder per method. Each folder has a `README.md` (what it is, how to install, how to
run, which evaluation flags to use) and a `run.py` that reads the shared subset file and
writes predictions in the shared layout. Nothing else in the repo needs to know how a method
works, the evaluation scripts only read the prediction files.

## Prediction contract

| dataset | file | content |
|---|---|---|
| NYU | `outputs/<method>/nyu/<id:04d>.npy` | float32 (480, 640) depth in metres, or relative depth (say so in the README and use `--align affine`) |
| NYU | `outputs/<method>/nyu/<id:04d>_normals.npy` | optional float32 (480, 640, 3) unit normals in the camera frame |
| Pix3D | `outputs/<method>/pix3d/<id:04d>.obj` or `.ply` | mesh, any frame and scale |
| Pix3D | `outputs/<method>/pix3d/<id:04d>.npy` | float32 (N, 3) point cloud, any frame and scale |

Use `si3d.io.save_depth` / `si3d.io.save_shape`, they put the files in the right place.
Ids are the ones in `configs/subset_nyu.txt` and `configs/subset_pix3d.txt`. Run on the full
subset even if a method clearly fails on some images; the failures are part of the result.

## Folders

| folder | type | data | evaluation |
|---|---|---|---|
| `sfs/` | classical, from scratch | NYU (+ a few Blender renders) | `eval_depth.py --align affine --normals` |
| `symmetry/` | classical, from scratch | Pix3D | `eval_shape.py --gt-frame camera` |
| `depth_anything_v2/` | pretrained, own fine-tune | NYU | `eval_depth.py` (metric) |
| `triposr/` | pretrained | Pix3D | `eval_shape.py` |

Classical methods are implemented by us. For the learned ones the network is used as
published; the own contribution is the evaluation plus one small change (fine-tune on the
NYU train split, a different loss, resolution or pre-processing) with a before/after number.
