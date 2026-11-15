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
  gets no depth. The view angle (between the ray to the object and the plane) is the
  quantity that decides how well depth is defined.

## Pipeline (`pipeline.reconstruct`)

1. Crop the object, resize to 640 px, replace the background by grey (`crop_and_resize`).
2. Scale-space keypoints with SIFT-like descriptors (`features.py`). The descriptor of the
   mirrored patch is a fixed permutation of the descriptor, so features are matched against
   the mirrored descriptors of all other features, with ratio test.
3. RANSAC over two-pair samples for `n` (`geometry.ransac_plane`). The score is the number
   of distinct epipolar lines with an inlier. The plane is accepted if that number is clearly
   above what unrelated pairs reach, and rejected as `frontal` below 10 deg or `side` above
   75 deg view angle.
4. Guided matching along the epipolar lines, non-linear refinement of `n`, triangulation.
   Only pairs whose features vary along the joining line are triangulated: a pair on one
   straight edge can slide along it and has no defined depth.
5. Dense search (`dense.py`): for every third object pixel the partner is searched along
   its epipolar line by normalised cross-correlation. The patch is mirrored through 3D for
   a few surface orientations perpendicular to the symmetry plane (front, top, seat: faces
   that are their own mirror image). Checks: structure along the line, uniqueness, search
   back from the partner, mirror image must beat shifted copy, larger window, depth agrees
   with the neighbours. On by default only for the rendered objects, see below.
6. Silhouette intersection: the partner must fall inside the mask, which bounds the depth of
   every pixel. This is a two-view visual hull with the mirrored camera as second view.

## Run

```
python methods/symmetry/run.py --demo --vis        # rendered test objects, no data needed
python methods/symmetry/run.py --vis               # the Pix3D subset
python methods/symmetry/run.py --ids 12 345 --vis  # single images
python scripts/eval_shape.py --method symmetry --align camera
python scripts/eval_shape.py --method symmetry_hull --align camera
python methods/symmetry/eval_plane.py              # detected plane against the CAD model

# analysis: the same reconstruction with the true plane from the CAD model and pose
python methods/symmetry/run.py --name symmetry_trueplane --true-plane
python scripts/eval_shape.py --method symmetry_trueplane_hull --align camera
```

| output | content |
|---|---|
| `outputs/symmetry/pix3d/<id>.npy` | matched points with their mirror copies |
| `outputs/symmetry_hull/pix3d/<id>.npy` | surface of the silhouette intersection |
| `outputs/symmetry/vis/<id>.png` | matches, epipolar lines, top and front view |
| `results/symmetry_status.csv` | per image: status, counts, detected normal, view angle |
| `results/symmetry_plane.csv` | per image: error of the detected plane, true view angle |

Points are in the camera frame, up to scale, so `--align camera` (one scale factor, nothing
else) is the fair evaluation. Status values: `ok`, `no_symmetry` (no plane with enough
support), `frontal`, `side`, `few_features`, `no_object`, `error`. Images without an
accepted plane get no prediction file. `--help` lists the parameters.

## Results

Rendered objects with a mirror-symmetric texture (`run.py --demo`, F-score at 5 % of the
object size, camera alignment). Plane and shape are known here.

| object | azimuth | plane error | accuracy | completeness | F-score | hull F-score |
|---|---|---|---|---|---|---|
| chair | 15 / 30 / 45 / 60 deg | 0.0 / 1.4 / 0.1 / 1.2 deg | 0.007 / 0.009 / 0.007 / 0.007 | 0.023 / 0.025 / 0.038 / 0.046 | 0.98 / 0.96 / 0.92 / 0.85 | 0.44 / 0.45 / 0.57 / 0.62 |
| table | 15 / 30 deg | 0.1 / 2.2 deg | 0.008 / 0.033 | 0.048 / 0.029 | 0.91 / 0.90 | 0.36 / 0.41 |
| table | 45 / 60 deg | no plane accepted | | | | |
| cabinet | 15 / 30 / 45 deg | 0.0 / 0.0 / 0.5 deg | 0.006 / 0.006 / 0.006 | 0.109 / 0.074 / 0.049 | 0.55 / 0.59 / 0.67 | 0.43 / 0.41 / 0.47 |
| cabinet | 60 deg | no plane accepted | | | | |

Pix3D subset, 50 photos, 38 of them with a mirror-symmetric CAD model. Parameters were set
on 45 other Pix3D images, not on the subset.

| question | answer |
|---|---|
| best plane candidate within 5 deg of a symmetry plane of the model | 8 of 38 |
| plane accepted | 5 of 50 |
| accepted planes within 5 deg | 0 (one at 5.9 deg) |
| `symmetry`, matched points, 5 images | accuracy 0.095, completeness 0.192, F-score 0.12 |
| `symmetry_hull`, 5 images | accuracy 0.206, completeness 0.038, F-score 0.33 |
| `symmetry_trueplane_hull`, true plane given, 29 images with 10 to 75 deg view angle | accuracy 0.122, completeness 0.040, F-score 0.48 |
| dense matches with the true plane within 3 % of the true depth (other 45 images) | 29 % |

## Where it breaks

On the rendered objects the method works as long as the view is neither frontal nor from
the side. On the photos the plane detection fails, and so does the dense search:

- Mirrored features need texture. Most furniture surfaces have none, so the keypoints sit
  on edges. Matches between points of one straight edge, or of parallel edges such as bed
  slats and door frames, slide along the edge. They agree with any epipole on that line,
  and many parallel edges make RANSAC lock onto their common direction. Where that
  direction happens to be the symmetry normal the plane is right, but for the wrong reason
  and with meaningless depths.
- Real mirror partners look different: one side is lit, the other shaded, cushions and
  books are not arranged symmetrically.
- Repeated parts along the symmetry direction (slats, legs, drawers) give several equally
  good partners on an epipolar line. A patch correlation picks one of them at random; a
  higher correlation does not mean a better match (matches above 0.97 are right less often
  than the rest).
- Many product photos are frontal: 8 of the 38 symmetric subset images have a view angle
  below 10 deg. The plane is easiest to find there and useless for depth.
- Faces whose mirror image is hidden (the side of a cabinet) cannot be matched.
- The silhouette intersection needs only the plane and the mask and is the one part that
  carries over to photos. It is open along the symmetry plane, the more the smaller the
  view angle, and two silhouettes cannot see concavities.
- The component of `n` along the viewing direction comes from perspective convergence only
  and is the least accurate; an error there stretches the shape in depth.
