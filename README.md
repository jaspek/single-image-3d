# Single-Image 3D Reconstruction: TripoSR on Pix3D Evaluation Pipeline

## Overview

This repository hosts the **Minimum Viable Product (MVP) 3D Evaluation Pipeline** for benchmarking single-image 3D reconstruction using **TripoSR** against ground-truth CAD models from the **Pix3D** dataset (developed as part of the Proseminar *Imaging Beyond Consumer Cameras (911.422)*, University of Salzburg).

Given a fixed subset of **50 real-world images** (from a single category such as `"chair"`) in the Pix3D dataset, the pipeline executes an end-to-end automated workflow:
1. **Subset Selection (`pix3d.json`)**: Parses the Pix3D metadata and selects 50 untruncated, unoccluded samples.
2. **Foreground Masking & Framing**: Applies the ground-truth 2D segmentation mask from Pix3D to remove background clutter, centers and pads the object to an $85\%$ foreground ratio, and composites it onto a neutral $50\%$ gray canvas expected by TripoSR.
3. **Single-Image 3D Mesh Generation**: Runs feed-forward inference with the local **TripoSR** model and extracts a 3D surface mesh (`.obj`) via Marching Cubes ($256^3$ grid resolution).
4. **3D Alignment & Quantitative Evaluation**: Uniformly samples $10{,}000$ surface points from both the generated TripoSR `.obj` mesh and the Pix3D ground-truth CAD `.obj` model, normalizes both point clouds via bounding-box centering and uniform unit scaling (plus optional rigid ICP refinement), and computes the symmetric **Chamfer Distance (CD)**.
5. **2D Visual Comparison Rendering**: Renders a single $512 \times 512$ 2D snapshot (`.png`) of each reconstructed 3D mesh from a standardized camera viewpoint using `pyrender` / `trimesh` (with an automatic headless-safe software shading fallback).

---

## Directory Structure

The pipeline strictly separates **read-only source repositories/datasets** from the **active target workspace**:

```text
C:\Users\flori\GitHub\
│
├── single-image-3d\                  # [MAIN WORKSPACE - Read/Write]
│   ├── eval_pipeline.py              # Main end-to-end MVP evaluation pipeline script
│   ├── README.md                     # Project documentation
│   ├── outputs\                      # All generated meshes, 2D renders, and metric logs
│   │   ├── inputs\                   # Masked & framed 2D input images (<id>_input.png)
│   │   ├── meshes\                   # Generated 3D TripoSR meshes (<id>_pred.obj)
│   │   ├── renders\                  # Rendered 2D snapshots of 3D meshes (<id>_render.png)
│   │   ├── results.csv               # Per-sample Chamfer Distance table
│   │   └── results.json              # Summary statistics (mean/median/std CD) + sample log
│   ├── src\si3d\                     # Shared package modules (datasets, metrics, I/O)
│   ├── scripts\                      # Auxiliary benchmark scripts
│   └── methods\                      # Method-specific runners
│
├── pix3d\                            # [PIX3D SOURCE DATA - Read-Only]
│   ├── pix3d.json                    # Dataset annotations, paths, and metadata
│   ├── img\                          # Raw 2D RGB images
│   ├── mask\                         # Ground-truth binary 2D object masks
│   └── model\                        # Ground-truth 3D CAD meshes (.obj)
│
└── TripoSR\                          # [TRIPOSR SOURCE REPO - Read-Only]
    ├── tsr\                          # TripoSR model architecture & utilities
    ├── run.py                        # Reference CLI inference script
    └── requirements.txt              # TripoSR package requirements
```

---

## Installation & Setup

### 1. Create and Activate a Virtual Environment

Open **PowerShell** inside the main workspace (`C:\Users\flori\GitHub\single-image-3d`):

```powershell
cd C:\Users\flori\GitHub\single-image-3d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch (with CUDA Support)

Install PyTorch matching your local CUDA version (recommended for fast TripoSR inference and Marching Cubes extraction):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Required `pip` Dependencies

Install the core scientific computing, 3D geometry processing (`trimesh`, `scipy`, `numpy`), TripoSR, and rendering libraries:

```powershell
# Core evaluation, 3D alignment, KD-Tree Chamfer Distance, and 2D rendering packages
pip install numpy scipy trimesh Pillow pyrender matplotlib

# TripoSR model dependencies
pip install omegaconf==2.3.0 einops==0.7.0 transformers huggingface-hub rembg imageio

# TorchMCubes (required by TripoSR for Marching Cubes surface extraction)
pip install git+https://github.com/tatsy/torchmcubes.git
```

### 4. Verify Pix3D Dataset Presence

Ensure the Pix3D dataset is unpacked inside `C:\Users\flori\GitHub\pix3d` such that `C:\Users\flori\GitHub\pix3d\pix3d.json` and the `img\`, `mask\`, and `model\` directories exist.

---

## Usage Instructions

All commands should be executed from the `C:\Users\flori\GitHub\single-image-3d` workspace directory.

### 1. Run the Complete End-to-End Pipeline (Inference + Rendering + Evaluation)

Generates the 50 masked inputs, runs TripoSR to produce 50 `.obj` meshes, renders 50 2D `.png` comparison images, computes the Chamfer Distance for each pair, and exports the summary files:

```powershell
cd C:\Users\flori\GitHub\single-image-3d
python eval_pipeline.py
```

### 2. Run Quantitative Evaluation Only (Skip Inference)

If the `.obj` meshes have already been generated inside `outputs\meshes\`, you can re-run the alignment and Chamfer Distance calculation on its own:

```powershell
python eval_pipeline.py --eval-only
```

### 3. Run Evaluation with Bounding-Box Translation/Scaling Only (No ICP)

To evaluate using strictly bounding-box centering and uniform scaling without rigid ICP rotation refinement:

```powershell
python eval_pipeline.py --eval-only --no-icp
```

---

## Alignment & Evaluation Metric Details

Because TripoSR reconstructs shapes in a viewer-centered canonical frame with arbitrary scale whereas Pix3D ground-truth CAD models reside in object coordinates, `eval_pipeline.py` aligns each pair before computing the metric:

1. **Surface Point Sampling**: Uniformly samples $N = 10{,}000$ points from both the predicted mesh $P$ and ground-truth mesh $Q$ using `trimesh.sample.sample_surface`.
2. **Bounding-Box Normalization (Translation & Scaling)**: Centers each point cloud's axis-aligned bounding box at the origin $(0, 0, 0)$ and scales uniformly so the longest bounding-box edge equals $1.0$ (matching the normalization convention in Pix3D's `eval/eval.py`):
   $$\mathbf{p}' = \frac{\mathbf{p} - \frac{1}{2}(\mathbf{b}_{\min} + \mathbf{b}_{\max})}{\max(\mathbf{b}_{\max} - \mathbf{b}_{\min})}$$
3. **Rigid Orientation Refinement (ICP)**: Runs multi-start Iterative Closest Point (`trimesh.registration.icp`) initialized across $90^\circ$ rotations to align the predicted shape orientation with the ground-truth CAD model.
4. **Symmetric Chamfer Distance (CD)**: Queries nearest neighbors in $\mathcal{O}(N \log N)$ time via `scipy.spatial.cKDTree`:
   $$\text{CD}(P, Q) = \frac{1}{|P|}\sum_{\mathbf{p} \in P} \min_{\mathbf{q} \in Q} \|\mathbf{p} - \mathbf{q}\|_2 + \frac{1}{|Q|}\sum_{\mathbf{q} \in Q} \min_{\mathbf{p} \in P} \|\mathbf{q} - \mathbf{p}\|_2$$

---

## Output Structure

After running `python eval_pipeline.py`, all artifacts are stored inside `C:\Users\flori\GitHub\single-image-3d\outputs\`:

| Path | Description |
| :--- | :--- |
| `outputs/inputs/<id>_input.png` | Preprocessed 2D Pix3D image with background removed via the ground-truth mask and framed on a $50\%$ gray canvas. |
| `outputs/meshes/<id>_pred.obj` | Reconstructed 3D triangle mesh (with vertex colors) generated by TripoSR for sample `<id>`. |
| `outputs/renders/<id>_render.png` | Single $512 \times 512$ 2D rendered image of the generated 3D mesh from a standard $3/4$ isometric camera angle for visual comparison. |
| `outputs/results.csv` | Tabular CSV log containing `pix3d_id`, `category`, `img_rel_path`, `gt_model_rel_path`, `pred_mesh_path`, `render_image_path`, and `chamfer_distance` for each evaluated sample. |
| `outputs/results.json` | JSON summary report containing aggregate statistics (`mean_chamfer_distance`, `median_chamfer_distance`, `std_chamfer_distance`, `min_chamfer_distance`, `max_chamfer_distance`) alongside the per-sample records. |
