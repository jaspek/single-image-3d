# Depth Anything V2, metric indoor (pretrained)

Official repo: https://github.com/DepthAnything/Depth-Anything-V2 (metric models in
`metric_depth/`). The indoor metric checkpoints were fine-tuned on Hypersim (synthetic) and
are evaluated zero-shot on NYU, so NYU numbers are a fair test.

Setup (GPU with CUDA 12.8 on the newer cards: install torch cu128 first):

```
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/Depth-Anything-V2
pip install -r external/Depth-Anything-V2/requirements.txt
# checkpoint depth_anything_v2_metric_hypersim_vits.pth from the repo's model table -> checkpoints/
python methods/depth_anything_v2/run.py --encoder vits
python scripts/eval_depth.py --method depth_anything_v2
```

Own contribution: fine-tune the ViT-S metric model on the NYU train split (795 frames,
`NYUDepthV2.splits()['train']`) with the repo's `metric_depth/train.py`, write its
predictions as method `depth_anything_v2_nyuft`, and report both rows in the summary table.
Keep the test subset untouched; it comes from the official test split.

Check the model API against the repo README once, the calls in `run.py` follow the
`metric_depth/run.py` of the repo at the time of writing.
