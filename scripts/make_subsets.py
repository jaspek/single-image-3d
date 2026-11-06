"""Write the fixed evaluation subsets to configs/.

    python scripts/make_subsets.py [--n 50] [--seed 0]

Run once after downloading the data and commit the two files, so everybody evaluates on
exactly the same frames. NYU ids come from the official test split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from si3d.datasets import NYUDepthV2, Pix3D
from si3d.subsets import nyu_subset, pix3d_subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nyu-root", default="data/nyu")
    ap.add_argument("--pix3d-root", default="data/pix3d")
    ap.add_argument("--out", default="configs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    nyu = NYUDepthV2(a.nyu_root)
    ids = nyu_subset(nyu, a.n, a.seed)
    lines = [f"# NYU Depth v2 test-split subset, n={len(ids)}, seed={a.seed}", "# id  scene"]
    lines += [f"{i}  {nyu.scene(i)}" for i in ids]
    (out / "subset_nyu.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'subset_nyu.txt'} ({len(ids)} frames)")

    pix = Pix3D(a.pix3d_root)
    ids = pix3d_subset(pix, a.n, a.seed)
    lines = [f"# Pix3D subset (unoccluded, untruncated), n={len(ids)}, seed={a.seed}",
             "# id  category  image"]
    lines += [f"{i}  {pix.entry(i)['category']}  {pix.entry(i)['img']}" for i in ids]
    (out / "subset_pix3d.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'subset_pix3d.txt'} ({len(ids)} images)")


if __name__ == "__main__":
    main()
