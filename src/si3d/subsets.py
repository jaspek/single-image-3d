"""Fixed evaluation subsets, so every method is judged on the same images."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .datasets import NYUDepthV2, Pix3D


def nyu_subset(ds: NYUDepthV2, n: int = 50, seed: int = 0, split: str = "test") -> list[int]:
    ids = ds.splits()[split]
    rng = np.random.default_rng(seed)
    pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
    return sorted(int(i) for i in pick)


def pix3d_subset(ds: Pix3D, n: int = 50, seed: int = 0,
                 categories: tuple[str, ...] | None = None) -> list[int]:
    """Unoccluded, untruncated images, spread as evenly as possible over the categories."""
    rng = np.random.default_rng(seed)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(ds.meta):
        if e.get("truncated") or e.get("occluded") or e.get("slightly_occluded"):
            continue
        if categories is not None and e["category"] not in categories:
            continue
        by_cat[e["category"]].append(i)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])
    chosen: list[int] = []
    cats = sorted(by_cat)
    while len(chosen) < n and any(by_cat[c] for c in cats):
        for c in cats:
            if by_cat[c] and len(chosen) < n:
                chosen.append(by_cat[c].pop())
    return sorted(chosen)
