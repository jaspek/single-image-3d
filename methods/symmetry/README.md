# Symmetry as a second view (classical)

Hong, Yang, Huang, Ma (2004), "On symmetry and multiple-view geometry: structure, pose, and
calibration from a single image" (IJCV). A mirror-symmetric object seen once is the same as
two views of it: looking at the mirror image of a point with the real camera equals looking
at the point with a second, mirrored camera. Everything here is numpy/scipy, no OpenCV.

On furniture photos the symmetry plane is read from the straight edges of the object. The
same edges give three perpendicular axes and, with the outline, a box that bounds the shape.

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
   every pixel. This is a two-view visual hull with the mirrored camera as second view
   ("two cones").
7. With a normal from the edges: the intersection is cut down to the box that the outline
   and the three axes give (`hull.py`), see below.

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
   pitched between -15 and 65 deg). Of the two horizontal axes the normal is the one that
   the camera looks along least: furniture is photographed from its front, so the viewing
   ray is closer to the symmetry plane than to its normal.

The answer is refused, and the image gets no reconstruction, when

| reason | test |
|---|---|
| `no_lines` | no three axes found |
| `tilted` | no axis within 20 deg of a possible direction of gravity |
| `not_box` | less than 50 % of the edge length is straight and points to one of the axes |
| `few_lines` | fewer than 3 edges point along the normal |
| `ambiguous` | the camera looks along both horizontal axes equally (within 5 deg): front and side cannot be told apart |
| `unconfirmed` | no vanishing direction found without the other two axes lies within 5 deg of the normal |
| `uncertain` | one pixel of error at the segment ends moves the normal by more than 1.5 deg (error propagation; happens with a long lens and edges of only two directions) |

The rule for the normal and all limits were chosen on 639 Pix3D images
(`configs/dev_large_pix3d.txt`) and checked once on 639 others (`configs/holdout_pix3d.txt`).
Tried on the first list and dropped:

- The axis with the most edges as the normal. Where only one horizontal axis is right, it
  picks that one in 66 % of the images, the front-view rule in 78 % (94 % without desks,
  see "Where it breaks").
- Mirror pairs of line segments as evidence for the normal: 62 %.
- A limit on the size of the object in the photo: small images are not less often right.
- A logistic regression over thirteen measures of the frame instead of the two tests
  `unconfirmed` and `uncertain`: the same precision at the same number of accepted planes.

Whether the object really is symmetric about the plane is not tested. The edges only say
that the plane is one of the planes of a box.

## Box from the outline and the axes (`hull.py`)

Two silhouette cones leave the shape open along the symmetry plane, and over every flat
face a wedge remains between them. The three axes give a bound that needs no symmetry: the
object lies inside the box whose faces are perpendicular to the axes and touch its outline.

A line through the vanishing point of an axis that touches the silhouette is the image of
a box edge along that axis. Seen along the axis the box is a rectangle and the edge one of
its corners, and that corner lies on a line through the camera centre: one homogeneous
equation in two face positions. The three axes give six tangents, six equations for the six
faces up to scale; the symmetry plane through the middle of the box fixes the scale
(`fit_box`). For the outline of a box the six equations agree.

There is no box when a vanishing point lies inside the outline, when the outline shows a
single face (frontal view at the height of the object), when the six tangents disagree
(misfit above 0.2) or when the box does not cover the outline.

Because the box needs no symmetry, it can also stand alone: `--box-only` uses the edge
frame and the outline, no symmetry plane and no mirrored camera. It also works for frontal
views and for objects whose normal was refused.

`--more-planes` tests the other two axes as further symmetry planes (a table is symmetric
front to back as well) and intersects the cones of all mirrored cameras. A plane is used if
98 % of the silhouette stays covered with it. This closes the shape too, but less well than
the box, and together with the box it is worse than the box alone (F-score 0.74 against
0.82 on the first list), so it is off.

## Run

```
python methods/symmetry/run.py --demo --vis               # rendered test objects, no data needed
python methods/symmetry/run.py --demo --demo-plain --vis  # the same objects without texture
python methods/symmetry/run.py --vis                      # the Pix3D subset: plane from the edges, two cones and box
python methods/symmetry/run.py --ids 12 345 --vis         # single images
python scripts/eval_shape.py --method symmetry_hull --align camera
python methods/symmetry/eval_plane.py                     # detected plane against the CAD model

# variants for comparison
python methods/symmetry/run.py --no-box --name symmetry_twocones        # symmetry alone
python methods/symmetry/run.py --box-only --name box                    # edge frame and outline alone
python methods/symmetry/run.py --plane-from features --name symmetry_features
python methods/symmetry/run.py --true-plane --name symmetry_trueplane   # plane from the CAD model and pose

# the held-out list (639 images), results in a table of its own
python methods/symmetry/run.py --subset configs/holdout_pix3d.txt --name holdout_symmetry
python methods/symmetry/eval_plane.py --name holdout_symmetry
python scripts/eval_shape.py --method holdout_symmetry_hull --subset configs/holdout_pix3d.txt --align camera --table shape_holdout
```

| output | content |
|---|---|
| `outputs/symmetry_hull/pix3d/<id>.npy` | surface of the silhouette intersection, cut down to the box |
| `outputs/symmetry/pix3d/<id>.npy` | matched points with their mirror copies |
| `outputs/symmetry/vis/<id>.png` | edges by axis, box, epipolar lines, top and front view |
| `results/symmetry_status.csv` | per image: status, source of the plane, reason of a refusal, box, normal, view angle |
| `results/symmetry_plane.csv` | per image: error of the detected plane, true view angle |

Points are in the camera frame, up to scale, so `--align camera` (one scale factor, nothing
else) is the fair evaluation. Status values: `ok`, `no_symmetry` (no plane accepted),
`frontal`, `side`, `few_features`, `no_object`, `error`; with `--box-only` also `no_frame`
(not upright or not box-like) and `no_box`. Images without a result get no prediction
file. `--help` lists the parameters.

## Results

F-score at 5 % of the object size, camera alignment, everywhere.

### Rendered objects (`run.py --demo`)

Plane and shape are known here. The error of the normal is measured to the nearest symmetry
plane of the object (the chair has one, the table two, the cabinet three).

With a mirror-symmetric texture (`auto`). The edges refuse most images (the texture has
edges in all directions) and the feature pairs find the plane. F-score of the matched
points and of the silhouette intersection:

| object | azimuth | plane from | plane error | matched points | silhouette intersection |
|---|---|---|---|---|---|
| chair | 15 / 30 / 45 / 60 deg | features / edges / features / features | 0.0 / 2.2 / 0.1 / 1.2 deg | 0.98 / 0.93 / 0.92 / 0.85 | 0.44 / 0.77 / 0.57 / 0.62 |
| table | 15 / 30 deg | features / edges | 0.1 / 0.5 deg | 0.91 / 0.82 | 0.36 / 0.89 |
| table | 45 / 60 deg | no plane accepted | | | |
| cabinet | 15 / 30 / 45 deg | features | 0.0 / 0.0 / 0.5 deg | 0.55 / 0.59 / 0.67 | 0.43 / 0.41 / 0.47 |
| cabinet | 60 deg | no plane accepted | | | |

The two images with the plane from the edges are the two with a box.

Without texture, only shaded faces and the outline (`--demo-plain`). The feature pairs find
no plane in any of these images, and the matched points are of no use (F-score below 0.25:
the dense search has nothing to compare). F-score of the silhouette intersection:

| object | azimuth | plane from the edges | two cones | two cones and box | box alone |
|---|---|---|---|---|---|
| chair | 15 / 30 deg | 0.0 / 0.0 deg | 0.21 / 0.38 | 0.74 / 0.77 | 0.73 / 0.76 |
| chair | 45 / 60 deg | refused (ambiguous, unconfirmed) | | | 0.80 / 0.80 |
| table | 15 / 30 / 60 deg | 0.0 / 0.0 / 0.0 deg | 0.42 / 0.35 / 0.47 | 0.91 / 0.89 / 0.86 | 0.91 / 0.90 / 0.90 |
| table | 45 deg | refused (ambiguous) | | | 0.89 |
| cabinet | 15 deg | 0.1 deg | 0.18 | 1.00 | 1.00 |
| cabinet | 30 / 45 / 60 deg | refused (few edges, ambiguous) | | | 1.00 / 1.00 / 1.00 |

### Pix3D, evaluation subset

50 photos, 38 of them with a mirror-symmetric CAD model.

| | plane from feature pairs | plane from edges |
|---|---|---|
| plane accepted | 5 of 50 | 10 of 50 |
| accepted planes within 5 deg of a symmetry plane of the model | 0 | 7 |
| rejected because the view is frontal or from the side | 10 | 4 |

| row of `results/shape_summary.md` | plane | reconstruction | images | accuracy | completeness | F-score |
|---|---|---|---|---|---|---|
| `symmetry_features` | feature pairs | matched points | 5 | 0.095 | 0.192 | 0.12 |
| `symmetry_features_hull` | feature pairs | two cones | 5 | 0.206 | 0.038 | 0.33 |
| `symmetry_twocones_hull` | edges | two cones | 10 | 0.193 | 0.047 | 0.37 |
| `symmetry_hull` | edges | two cones and box | 10 | 0.040 | 0.029 | 0.77 |
| `box_hull` | none | box alone | 31 | 0.049 | 0.033 | 0.73 |
| `symmetry` | edges | matched points | 7 | 0.020 | 0.259 | 0.06 |
| `symmetry_trueplane_hull` | CAD model (analysis) | two cones | 29 | 0.122 | 0.040 | 0.48 |

Why the edges give no plane in 36 photos: unconfirmed 9, uncertain 8, not box-like 7, few
edges 5, tilted 4, ambiguous 2, no lines 1. Four more are frontal views. The box alone has
no result for 19 photos: 12 without a usable frame, 7 without a box.

### Pix3D, held-out list

639 photos that took no part in any choice, 577 of them with a mirror-symmetric CAD model.
Rows of `results/shape_holdout_summary.md`.

| | images | accuracy | completeness | F-score |
|---|---|---|---|---|
| plane from edges accepted | 166 | | | |
| of these within 5 deg of a symmetry plane of the model | 122 | | | |
| `holdout_twocones_hull`: two cones | 166 | 0.159 | 0.044 | 0.40 |
| `holdout_symmetry_hull`: two cones and box | 166 | 0.035 | 0.023 | 0.83 |
| `holdout_box_hull`: box alone | 431 | 0.060 | 0.034 | 0.74 |

By category (images, planes accepted, of these within 5 deg, F-score with two cones, with
two cones and box; images with a result of the box alone and their F-score):

| category | images | accepted | within 5 deg | two cones | two cones and box | box alone: images | box alone: F-score |
|---|---|---|---|---|---|---|---|
| bed | 90 | 32 | 29 | 0.39 | 0.73 | 74 | 0.67 |
| bookcase | 62 | 35 | 29 | 0.44 | 0.93 | 51 | 0.89 |
| chair | 90 | 4 | 2 | 0.48 | 0.69 | 33 | 0.47 |
| desk | 90 | 36 | 17 | 0.43 | 0.81 | 78 | 0.79 |
| misc | 27 | 2 | 0 | 0.36 | 0.68 | 4 | 0.57 |
| sofa | 90 | 14 | 9 | 0.26 | 0.72 | 73 | 0.65 |
| table | 90 | 23 | 19 | 0.42 | 0.82 | 62 | 0.75 |
| tool | 20 | 0 | 0 | | | 2 | 0.08 |
| wardrobe | 80 | 20 | 17 | 0.38 | 0.96 | 54 | 0.93 |
| all | 639 | 166 | 122 | 0.40 | 0.83 | 431 | 0.74 |

By the true view angle: of the 142 planes accepted at 10 to 45 deg, 121 are right; of the
16 accepted beyond 45 deg, one is. 85 further images are recognised as frontal views; the
box alone has a result for 47 of them (F-score 0.77).
What the mirrored camera adds to the box: on the 160 photos that have both results, the
box alone reaches 0.83, two cones and box 0.84, two cones alone 0.41.
On the first list (the one the limits were chosen on) 187 planes are accepted and 149 of
the 178 on symmetric models are within 5 deg.

## Where it breaks

Plane from the edges:

- It needs a box-like object: straight edges in three perpendicular directions. Chairs,
  sofas and round objects are mostly refused (4 of 90 held-out chairs get a plane).
- The front-view rule is wrong for an object photographed from its side: beyond 45 deg the
  depth axis is taken for the normal. For an object that is symmetric both ways (a table)
  this does no harm.
- Desks: many CAD models have drawers on one side and are symmetric front to back only.
  The rule takes the long front edge for the normal, as a person would, and counts as
  wrong (17 of 36 held-out desks right). The outline is nearly symmetric both ways, so the
  shape does not suffer.
- With a long lens the edges of one direction are almost parallel in the image, and the
  component of `n` along the viewing direction is poorly fixed unless edges of all three
  directions are visible. These cases are refused as `uncertain`.
- Shadow borders and the inner edges of shelves are straight too, but point elsewhere.
- A frontal product photo gives the plane very precisely and no depth.

Box:

- It is the bounding box, exact only for a box. A table gets its top and legs from the
  cones inside the box; an armchair is cut to a box it does not fill.
- A round outline touches the tangents where a box would have no corner; the box comes out
  too small. Such objects are usually refused before as not box-like.
- The box is right only if the three axes are. It does not depend on which horizontal
  axis is the normal, so a wrong choice between front and side does not hurt it.
- It is not a result of the symmetry. On the photos the box alone reaches almost the score
  of two cones and box together (0.83 against 0.84 on the same held-out photos); the
  mirrored camera adds little once the box is there.

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
  dense matches are within 3 % of the true depth.
- Faces whose mirror image is hidden (the side of a cabinet) cannot be matched.

Silhouette intersection:

- Two cones alone are an outer bound, not the surface: open along the symmetry plane, the
  more the smaller the view angle, with a wedge over every flat face, and blind to
  concavities. Seen from above, the two cones around a rectangular desk form a rhombus.
- The F-score of this bound reacts little to a wrong plane. The plane error is the number
  that says whether the plane is right.
