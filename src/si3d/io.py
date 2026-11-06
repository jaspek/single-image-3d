"""Prediction file conventions shared by all methods.

outputs/<method>/nyu/<id:04d>.npy            depth, float32 (480, 640), metres (or relative, say so)
outputs/<method>/nyu/<id:04d>_normals.npy    optional, float32 (480, 640, 3), camera frame
outputs/<method>/pix3d/<id:04d>.obj|.ply     mesh, or
outputs/<method>/pix3d/<id:04d>.npy          point cloud, float32 (N, 3)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

OUTPUTS = Path("outputs")


def pred_dir(method: str, dataset: str) -> Path:
    d = OUTPUTS / method / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_depth(method: str, idx: int, depth: np.ndarray,
               normals: np.ndarray | None = None) -> Path:
    d = pred_dir(method, "nyu")
    p = d / f"{idx:04d}.npy"
    np.save(p, np.asarray(depth, dtype=np.float32))
    if normals is not None:
        np.save(d / f"{idx:04d}_normals.npy", np.asarray(normals, dtype=np.float32))
    return p


def load_depth(method: str, idx: int) -> np.ndarray | None:
    p = OUTPUTS / method / "nyu" / f"{idx:04d}.npy"
    return np.load(p) if p.exists() else None


def load_normals(method: str, idx: int) -> np.ndarray | None:
    p = OUTPUTS / method / "nyu" / f"{idx:04d}_normals.npy"
    return np.load(p) if p.exists() else None


def save_shape(method: str, idx: int, shape: trimesh.Trimesh | np.ndarray) -> Path:
    d = pred_dir(method, "pix3d")
    if isinstance(shape, trimesh.Trimesh):
        p = d / f"{idx:04d}.ply"
        shape.export(p)
    else:
        p = d / f"{idx:04d}.npy"
        np.save(p, np.asarray(shape, dtype=np.float32).reshape(-1, 3))
    return p


def load_shape(method: str, idx: int) -> trimesh.Trimesh | np.ndarray | None:
    d = OUTPUTS / method / "pix3d"
    for ext in (".obj", ".ply", ".glb", ".off"):
        p = d / f"{idx:04d}{ext}"
        if p.exists():
            return trimesh.load(p, force="mesh")
    p = d / f"{idx:04d}.npy"
    if p.exists():
        return np.load(p)
    return None


def read_subset(path: str | Path) -> list[int]:
    """Subset files: one sample per line, first token is the id, '#' starts a comment."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(int(line.split()[0]))
    return ids
