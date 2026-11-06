# TripoSR (pretrained)

Official repo: https://github.com/VAST-AI-Research/TripoSR (2024, PyTorch, single image to
textured mesh in under a second on a GPU). The model wants an object on a neutral
background; Pix3D ships a mask per image, so no background-removal network is needed.

Setup:

```
git clone https://github.com/VAST-AI-Research/TripoSR external/TripoSR
pip install -r external/TripoSR/requirements.txt      # torch first, cu128 on the newer cards
python methods/triposr/run.py --subset configs/subset_pix3d.txt
python scripts/eval_shape.py --method triposr
```

`run.py` composites the masked object onto 50 % grey with 15 % margin (what the repo's own
pre-processing produces), calls the repo's `run.py` with `--no-remove-bg`, and copies the
resulting mesh to `outputs/triposr/pix3d/<id>.obj`. Check the flag names against the repo
README once; they follow the repo at the time of writing.

Own contribution options: marching-cubes resolution and foreground ratio sweeps, with and
without the mask (letting the built-in rembg do the segmentation), and the failure cases per
category (thin structures such as chair legs and tables are the known weak spot).
