# Data

Nothing in this folder is committed except this file. Download once per machine.

## NYU Depth v2 (scenes) -> `data/nyu/`

| file | size | source |
|---|---|---|
| `nyu_depth_v2_labeled.mat` | 2.8 GB | http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat |
| `splits.mat` | small | http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat |

Project page: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. If the NYU server is
down, the labeled subset is mirrored on Hugging Face (search "nyu_depth_v2"), but keep the
`.mat` layout so the loader works. 1449 frames, 640x480, depth in metres, official split
795 train / 654 test. We evaluate on frames from the test split only.

## Pix3D (objects) -> `data/pix3d/`

| file | size | source |
|---|---|---|
| `pix3d.zip` | 3.6 GB | http://pix3d.csail.mit.edu/data/pix3d.zip |

Unzip so that `data/pix3d/pix3d.json`, `data/pix3d/img/`, `data/pix3d/mask/` and
`data/pix3d/model/` exist. 10 069 images, 395 CAD models, 9 categories, with mask, pose,
focal length and keypoints per image. After unzipping run
`python scripts/check_pix3d_projection.py` once and set the camera convention it reports.

## Then

```
python scripts/make_subsets.py      # writes configs/subset_nyu.txt and configs/subset_pix3d.txt
```

Commit the two subset files. Everybody evaluates on exactly those ids.

## Optional

- `data/renders/`: a few Blender renders (one light, Lambertian, known depth) as the
  controlled case for shape from shading. Document the setup in `methods/sfs/README.md`.
- The datasets are also on the department server; ask the lecturer if a download fails.
