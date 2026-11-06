# Symmetry as a second view (classical)

Hong, Yang, Huang, Ma (2004), "On symmetry and multiple-view geometry: structure, pose, and
calibration from a single image" (IJCV). A bilaterally symmetric object seen once is the
same as two views of the object: the mirror image is a second camera. Matching features
between the image and its horizontal flip gives correspondences; with the known focal length
(Pix3D gives it) the symmetry plane and a sparse 3D point set follow by triangulation.

Pipeline: SIFT on the image and on its mirror, ratio-test matches, RANSAC on the symmetry
constraint, estimate the mirror pose, triangulate, keep points inside the mask.
Output: sparse points in the camera frame, up to scale (`.npy`, N x 3). Evaluate with
`--gt-frame camera`; accuracy is the meaningful number, completeness will be low because the
output is sparse.

```
python methods/symmetry/run.py --subset configs/subset_pix3d.txt
python scripts/eval_shape.py --method symmetry --gt-frame camera
```
