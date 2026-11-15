# Symmetry as a second view (classical)

Hong, Yang, Huang, Ma (2004), "On symmetry and multiple-view geometry: structure, pose, and
calibration from a single image" (IJCV). A mirror-symmetric object seen once is the same as
two views of it: looking at the mirror image of a point with the real camera equals looking
at the point with a second, mirrored camera. Everything here is numpy/scipy, no OpenCV.

## Geometry

Camera frame: x right, y down, z forward. The object is symmetric about the plane
`n . X = d`. A point `X` and its partner `X' = X - 2 (n . X - d) n` project to `x` and `x'`.

- The mirrored camera sits at `2 d n`; the relative pose is a reflection, so the essential
  matrix is just `E = [n]x`. The epipolar constraint `x'^T K^-T [n]x K^-1 x = 0` says that
  `x`, `x'` and the vanishing point `v = K n` lie on one line.
- Two pairs fix `n` (two degrees of freedom). `d` is the global scale, we set `d = 1`.
- With `n` known, a pair gives its two depths in closed form (`geometry.triangulate`).
- Every reconstructed point also gives its mirror image, which is usually the side the
  camera does not see.
- The baseline is `2 d`. A camera in the symmetry plane (frontal view) has no baseline and
  gets no depth; the depth error grows as the view angle to the plane shrinks.

## Pipeline (`pipeline.reconstruct`)

1. Crop the object, resize to 640 px, replace the background by grey (`crop_and_resize`).
2. Scale-space keypoints with SIFT-like descriptors (`features.py`). The descriptor of the
   mirrored patch is a fixed permutation of the descriptor, so features are matched against
   the mirrored descriptors of all other features, with ratio test.
3. RANSAC over two-pair samples for `n` (`geometry.ransac_plane`). The score is the number
   of distinct epipolar lines with an inlier, so matches sliding along one straight edge
   count once. The plane is accepted if that number is clearly above what unrelated pairs
   reach. A weakly supported plane has to be confirmed by step 5 on a coarse grid, against
   two deliberately wrong planes.
4. Guided matching along the epipolar lines, non-linear refinement of `n`, triangulation.
5. Dense search (`dense.py`): for every third object pixel the partner is searched along
   its epipolar line by normalised cross-correlation. The patch is mirrored through 3D for
   a few surface orientations perpendicular to the symmetry plane (front, top, seat: faces
   that are their own mirror image). Checks: uniqueness, search back from the partner,
   mirror image must beat shifted copy, larger window, depth agrees with the neighbours.
6. Silhouette intersection: the partner must fall inside the mask, which bounds the depth of
   every pixel. This is a two-view visual hull with the mirrored camera as second view.

## Run

```
python methods/symmetry/run.py --demo --vis        # rendered test objects, no data needed
python methods/symmetry/run.py --vis               # the Pix3D subset
python methods/symmetry/run.py --ids 12 345 --vis  # single images
python scripts/eval_shape.py --method symmetry --align camera
python scripts/eval_shape.py --method symmetry_hull --align camera
```

| output | content |
|---|---|
| `outputs/symmetry/pix3d/<id>.npy` | feature pairs and dense matches with their mirror copies |
| `outputs/symmetry_hull/pix3d/<id>.npy` | surface of the silhouette intersection, cut to the box of the matched points |
| `outputs/symmetry/vis/<id>.png` | matches, depth of the dense matches, top and front view |
| `results/symmetry_status.csv` | per image: status, counts, view angle, seconds |

Points are in the camera frame, up to scale, so `--align camera` (one scale factor, nothing
else) is the fair evaluation; `--align icp` gives the number under the same protocol as the
methods with their own output frame. Status values: `ok`, `no_symmetry` (no plane with
enough support), `few_features`, `no_object`, `error`. Images without a plane get no
prediction file. `--no-dense` keeps the feature pairs only, `--help` lists the parameters.

## Controlled case

`synthetic.py` renders box-built objects with a mirror-symmetric texture, so the plane and
the shape are known. `run.py --demo` prints (F-score at 5 % of the object size, camera alignment):

| object | azimuth | plane error | accuracy | completeness | F-score | hull F-score |
|---|---|---|---|---|---|---|
| chair | 15 / 30 / 45 / 60 deg | 0.0 / 1.4 / 0.1 / 1.2 deg | 0.007 / 0.010 / 0.007 / 0.007 | 0.023 / 0.025 / 0.038 / 0.050 | 0.98 / 0.97 / 0.92 / 0.82 | 0.45 / 0.45 / 0.57 / 0.57 |
| table | 15 / 30 deg | 0.1 / 2.2 deg | 0.008 / 0.036 | 0.048 / 0.025 | 0.92 / 0.94 | 0.43 / 0.40 |
| table | 45 / 60 deg | no plane found | | | | |
| cabinet | 15 / 30 / 45 deg | 0.0 / 0.0 / 0.5 deg | 0.006 / 0.006 / 0.006 | 0.105 / 0.075 / 0.048 | 0.55 / 0.59 / 0.68 | 0.41 / 0.43 / 0.47 |
| cabinet | 60 deg | no plane found | | | | |

## Where it breaks

- No texture: no features, no dense matches. Only the silhouette intersection is left, and
  it needs the plane.
- Faces whose mirror image is hidden (the side of a cabinet) cannot be reconstructed, so
  completeness stays low for box-like objects.
- Horizontal faces seen at a flat angle from far off the symmetry plane: the mirrored
  texture is strongly sheared and the features stop matching (table at 45 deg and more).
- Repeated parts along the symmetry direction (legs, drawers) can pair with the wrong
  partner and land at a wrong depth.
- The component of `n` along the viewing direction comes from perspective convergence only
  and is the least accurate; an error there stretches the shape in depth.
- The silhouette intersection is open along the symmetry plane; the smaller the view angle,
  the longer it gets.
- Views near the symmetry plane (small baseline) and views from the side (partner faces
  hidden).
