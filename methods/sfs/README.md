# Shape from shading (classical)

Tsai and Shah (1994), "Shape from shading using linear approximation": Lambertian surface,
single distant light source with known direction, constant albedo, orthographic camera.
The reflectance equation is linearised around the current depth estimate and solved with a
Jacobi-style iteration; it is a few dozen lines of NumPy.

Inputs: grayscale NYU frame, light direction (tilt, slant). Outputs: relative depth and
normals (the method has no absolute scale, so evaluate with `--align affine`).

Expectation: the assumptions do not hold on NYU rooms (several lights, textures, shadows),
so the numbers will be poor. That is part of the story. To show the method working, add
a few Blender renders of a Lambertian object with one light and known depth as a controlled
case (put them under `data/renders/`, document the setup here).

```
python methods/sfs/run.py --subset configs/subset_nyu.txt
python scripts/eval_depth.py --method sfs --align affine --normals
```
