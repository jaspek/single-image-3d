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
- An edge of the object that runs across its width is parallel to `n`, so its image points
  to the same vanishing point `v`. The normal can be read from such edges without matching
  anything.
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
3. The normal `n`, from one of two sources (`--plane-from`):
   - `lines`: from the straight edges of the object, see the next section.
   - `features`: RANSAC over two-pair samples (`geometry.ransac_plane`). The score is the
     number of distinct epipolar lines with an inlier; the plane is accepted if that number
     is clearly above what unrelated pairs reach.
   - `auto`: the edges, and the feature pairs if the edges give no answer.

   A plane is rejected as `frontal` below 10 deg and as `side` above 75 deg view angle.
4. Guided matching along the epipolar lines, triangulation. Only pairs whose features vary
   along the joining line are triangulated: a pair on one straight edge can slide along it
   and has no defined depth.
5. Dense search (`dense.py`): for every third object pixel the partner is searched along
   its epipolar line by normalised cross-correlation. The patch is mirrored through 3D for
   a few surface orientations perpendicular to the symmetry plane (front, top, seat: faces
   that are their own mirror image). Checks: structure along the line, uniqueness, search
   back from the partner, mirror image must beat shifted copy, larger window, depth agrees
   with the neighbours. On by default only for the rendered objects, see below.
6. Silhouette intersection: the partner must fall inside the mask, which bounds the depth of
   every pixel. This is a two-view visual hull with the mirrored camera as second view.
7. With a normal from the edges: further symmetry planes and the intersection of the
   silhouette cones of all mirrored cameras (`hull.py`), see below.

## Plane from the straight edges (`lines.py`)

1. Line segments. The gradient direction is quantised into sectors, connected pixels of one
   sector form a region, and a line is fitted to the edge points of the region (Burns,
   Hanson, Riseman 1986). A region with a corner is cut there. Curved edges are not used,
   but their length is counted.
2. Vanishing directions. A segment with the end points `a`, `b` and a direction `d` belong
   together if `d . (r_a x r_b) = 0`. This is the constraint of a mirror pair, so the
   search is the same RANSAC with two segments per sample, one direction after the other.
3. Three perpendicular axes (`fit_frame`). The strongest direction is the first axis; with
   it fixed, one more segment gives the second axis and the third follows. The triple that
   explains the most edge length is refined as a whole. An axis with few or short edges is
   held in place by the other two.
4. The vertical is the axis closest to a possible direction of gravity (camera not rolled,
   pitched between -15 and 65 deg). Of the two horizontal axes the normal is the one with
   the most distinct edges: front and back edges, rails, shelves and drawer fronts of
   furniture run across its width.

The answer is refused, and the image gets no reconstruction, when

| reason | test |
|---|---|
| `small` | the object had to be enlarged more than 2 times to reach 640 px |
| `no_lines` | no three axes found |
| `tilted` | no axis within 20 deg of a possible direction of gravity |
| `not_box` | less than 75 % of the edge length is straight and points to one of the axes |
| `few_lines` | fewer than 3 edges on the normal |
| `uncertain` | the edges hold the normal in place only to more than 5 deg (error propagation; happens with a long lens and edges of only two directions) |

Whether the object really is symmetric about the plane is not tested. The edges only say
that the plane is one of the three planes of a box.

## More symmetry planes (`hull.py`)

Two silhouette cones leave the shape open along the symmetry plane. Box-like furniture is
often symmetric front to back and top to bottom as well. Reflections at `k` perpendicular
planes form a group of `2^k` elements, each one a camera, and a point of the object must
project into the silhouette after every one of them.

The normal of a further plane is an axis of the edge frame. Its position is found by a
one-dimensional search: with the right position every pixel of the silhouette still sees a
point of the intersection of all cones. If at least 98 % of the silhouette stays covered the
plane is used, otherwise not (a chair has no second plane, the back has no counterpart in
front). A plane that passes the test keeps the result consistent with the outline. It does
not prove that the object has the symmetry, see "Where it breaks".

## Run

```
python methods/symmetry/run.py --demo --vis               # rendered test objects, no data needed
python methods/symmetry/run.py --demo --demo-plain --vis  # the same objects without texture
python methods/symmetry/run.py --vis                      # the Pix3D subset, plane from the edges
python methods/symmetry/run.py --ids 12 345 --vis         # single images
python scripts/eval_shape.py --method symmetry_hull --align camera
python scripts/eval_shape.py --method symmetry --align camera
python methods/symmetry/eval_plane.py                     # detected plane against the CAD model

# variants for comparison
python methods/symmetry/run.py --plane-from features --name symmetry_features
python methods/symmetry/run.py --no-more-planes --name symmetry_twocones
python methods/symmetry/run.py --true-plane --name symmetry_trueplane   # plane from the CAD model and pose
```

| output | content |
|---|---|
| `outputs/symmetry_hull/pix3d/<id>.npy` | surface of the silhouette intersection |
| `outputs/symmetry/pix3d/<id>.npy` | matched points with their mirror copies |
| `outputs/symmetry/vis/<id>.png` | edges by axis, matches, epipolar lines, top and front view |
| `results/symmetry_status.csv` | per image: status, source of the plane, reason of a refusal, counts, normal, view angle |
| `results/symmetry_plane.csv` | per image: error of the detected plane, true view angle |

Points are in the camera frame, up to scale, so `--align camera` (one scale factor, nothing
else) is the fair evaluation. Status values: `ok`, `no_symmetry` (no plane accepted),
`frontal`, `side`, `few_features`, `no_object`, `error`. Images without an accepted plane
get no prediction file. `--help` lists the parameters.

## Results

F-score at 5 % of the object size, camera alignment, everywhere.

### Rendered objects (`run.py --demo`)

Plane and shape are known here. The error of the normal is measured to the nearest symmetry
plane of the object.

With a mirror-symmetric texture (`auto`). The edges refuse in all images but one
(`not_box`, the texture has edges in all directions) and the feature pairs find the plane:

| object | azimuth | plane error | accuracy | completeness | F-score | hull F-score |
|---|---|---|---|---|---|---|
| chair | 15 / 30 / 45 / 60 deg | 1.1 / 1.4 / 0.1 / 1.2 deg | 0.010 / 0.009 / 0.007 / 0.007 | 0.026 / 0.025 / 0.038 / 0.046 | 0.95 / 0.96 / 0.92 / 0.85 | 0.33 / 0.45 / 0.57 / 0.62 |
| table | 15 / 30 deg | 0.1 / 2.2 deg | 0.008 / 0.033 | 0.048 / 0.029 | 0.91 / 0.90 | 0.36 / 0.41 |
| table | 45 / 60 deg | no plane accepted | | | | |
| cabinet | 15 / 30 / 45 deg | 0.0 / 0.0 / 0.5 deg | 0.006 / 0.006 / 0.006 | 0.109 / 0.074 / 0.049 | 0.55 / 0.59 / 0.67 | 0.43 / 0.41 / 0.47 |
| cabinet | 60 deg | no plane accepted | | | | |

The chair at 15 deg is the one image where the edges are accepted.

Without texture, only shaded faces and the outline (`--demo-plain`). The feature pairs find
no plane in any of the twelve images; the edges do:

| object | azimuth | plane error | planes used | hull F-score |
|---|---|---|---|---|
| chair | 15 / 30 / 45 / 60 deg | 0.0 / 0.1 / 0.1 / 0.1 deg | 2 / 1 / 1 / 1 | 0.34 / 0.38 / 0.39 / 0.55 |
| table | 15 / 30 / 45 / 60 deg | 0.0 / 0.0 / 0.0 / 0.0 deg | 2 / 2 / 2 / 2 | 0.85 / 0.83 / 0.77 / 0.70 |
| cabinet | 15 / 30 / 45 / 60 deg | 0.1 / 0.0 / 0.0 / 0.1 deg | 3 / 3 / 3 / 3 | 0.45 / 0.52 / 0.66 / 0.72 |

The chair has one symmetry plane, the table two, the cabinet three. The second plane of
the chair at 15 deg is one it does not have, see "Where it breaks".

The matched points are of no use on these images (F-score below 0.25): without texture the
dense search has nothing to compare.

### Pix3D

50 photos, 38 of them with a mirror-symmetric CAD model. All thresholds were set on 45
other Pix3D images (`configs/dev_pix3d.txt`), not on the subset.

| | plane from feature pairs | plane from edges |
|---|---|---|
| plane accepted | 5 of 50 | 6 of 50 |
| accepted planes within 5 deg of a symmetry plane of the model | 0 | 4 |
| rejected because the view is frontal or from the side | 10 | 4 |

| row of `results/shape_summary.md` | plane | reconstruction | images | accuracy | completeness | F-score |
|---|---|---|---|---|---|---|
| `symmetry_features` | feature pairs | matched points | 5 | 0.095 | 0.192 | 0.12 |
| `symmetry_features_hull` | feature pairs | two cones | 5 | 0.206 | 0.038 | 0.33 |
| `symmetry_twocones_hull` | edges | two cones | 6 | 0.154 | 0.051 | 0.36 |
| `symmetry_hull` | edges | cones of all accepted planes | 6 | 0.080 | 0.047 | 0.58 |
| `symmetry` | edges | matched points | 3 | 0.027 | 0.226 | 0.06 |
| `symmetry_trueplane_hull` | CAD model (analysis) | two cones | 29 | 0.122 | 0.040 | 0.48 |

Of the six accepted images four use two planes and two use three. On these six images the
true plane with two cones reaches 0.45.

Why the edges refuse 40 of the 50 photos: not box-like 23 (mostly beds, chairs, sofas and
tools), too small 11, no vertical axis 3, uncertain 2, no lines 1. On the 45 development images
the same settings accept 12 planes, 11 of them within 5 deg, with a hull F-score of 0.58.
These numbers are optimistic, because the thresholds were chosen on those images; the
subset also has more small images (11 against 7).

## Where it breaks

Plane from the edges:

- It needs a box-like object: straight edges in three perpendicular directions. Chairs,
  sofas and beds with curved or slanted parts are refused, and so is every round object.
- Seen from the side, the depth of the object shows more edges than its width, and the
  wrong horizontal axis is taken (one of the two wrong planes on the subset, a desk at
  54 deg). For an object that is symmetric both ways this does no harm.
- With a long lens the edges of one direction are almost parallel in the image, and the
  component of `n` along the viewing direction is poorly fixed unless edges of all three
  directions are visible. Most such cases are refused as `uncertain`; the other wrong plane
  of the subset (12 deg off) passed just below the limit.
- Small photos: the edges of an object of 300 pixels are too short and too soft.
- Shadow borders and the inner edges of shelves are straight too, but point elsewhere.
- A frontal product photo gives the plane very precisely and no depth (3 subset images).

Plane from the feature pairs, and the matched points:

- Mirrored features need texture. Most furniture surfaces have none, so the keypoints sit
  on edges. Matches between points of one straight edge, or of parallel edges such as bed
  slats and door frames, slide along the edge. They agree with any epipole on that line,
  and many parallel edges make RANSAC lock onto their common direction.
- Real mirror partners look different: one side is lit, the other shaded, cushions and
  books are not arranged symmetrically.
- Repeated parts along the symmetry direction (slats, legs, drawers) give several equally
  good partners on an epipolar line. A patch correlation picks one of them at random; a
  higher correlation does not mean a better match. With the true plane only 29 % of the
  dense matches are within 3 % of the true depth (development images).
- Faces whose mirror image is hidden (the side of a cabinet) cannot be matched.

Silhouette intersection:

- It needs only the planes and the mask and is the part that carries over to photos, but
  it is an outer bound, not the surface. Cones cannot see concavities, and over a flat face
  a wedge remains between the cones of two cameras that look at it from both sides. A
  cabinet stays at an F-score of about 0.5 even with three planes; a table with two planes
  reaches 0.7 to 0.8.
- The coverage test for a further plane is weak when the camera looks almost along the
  first plane. Points in the empty space between the parts of the object then satisfy all
  cones, and a plane that the object does not have is accepted: the rendered chair at
  15 deg gets a second plane (coverage 0.98). The result is still closer to the chair than
  the two cones alone, which are wide open at this angle (F-score 0.34 against 0.25), but
  its extent in depth is then invented. On the photos the further planes helped also where
  the CAD model is not symmetric that way, because the outline of box-like furniture
  nearly is.
- The F-score of this bound reacts little to a wrong plane (a desk with the plane 89 deg
  off still scores 0.79, because its outline is nearly symmetric that way too). The plane
  error is the number that says whether the plane is right.
