"""Debug picture of one reconstruction: matches, depth of the dense matches, two views of
the point cloud."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from methods.symmetry.pipeline import Result  # noqa: E402


def _subsample(p: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[np.random.default_rng(seed).choice(len(p), n, replace=False)]


def figure(res: Result, title: str = "") -> plt.Figure:
    dbg = res.debug
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    info = res.info
    fig.suptitle(f"{title}  status={res.status}  keypoints={info.get('n_keypoints', 0)}  "
                 f"matches={info.get('n_matches', 0)}  inliers={info.get('n_inliers', 0)}  "
                 f"lines={info.get('support', 0)} (chance {info.get('chance', 0)})  "
                 f"view angle={info.get('view_angle_deg', '-')} deg", fontsize=10)
    for ax in axes[:2]:
        ax.axis("off")
    if "gray" not in dbg:
        return fig
    gray, f, pairs = dbg["gray"], dbg["features"], dbg["pairs"]

    ax = axes[0]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax.set_title("mirror matches (green: used, red: rejected)", fontsize=9)
    used = dbg.get("used")
    for i, j in pairs[~dbg["inliers"]]:
        ax.plot(f.xy[[i, j], 0], f.xy[[i, j], 1], color="tab:red", lw=0.4, alpha=0.6)
    if used is not None:
        for i, j in used:
            ax.plot(f.xy[[i, j], 0], f.xy[[i, j], 1], color="lime", lw=0.7)
        ax.scatter(f.xy[used.ravel(), 0], f.xy[used.ravel(), 1], s=5, color="yellow", zorder=3)

    ax = axes[1]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax.set_title("depth of the dense matches, epipolar lines", fontsize=9)
    n = dbg.get("normal")
    if n is not None:
        H, W = gray.shape
        v = dbg["K"] @ n
        ys, xs = np.nonzero(dbg["mask"])
        for y in np.linspace(ys.min(), ys.max(), 7)[1:-1]:
            x = np.array([xs.mean(), y])
            u = np.array([v[0] - v[2] * x[0], v[1] - v[2] * x[1]])
            u /= np.linalg.norm(u)
            a, b = x - 2 * W * u, x + 2 * W * u
            ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=0.5, ls="--")
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
    d = dbg.get("dense")
    if d is not None and len(d.points):
        kept = dbg.get("kept", np.ones(len(d.points), bool))
        ax.scatter(d.pixels[kept, 0], d.pixels[kept, 1], c=d.points[kept, 2], s=2, cmap="turbo")

    pts, hull = _subsample(res.points, 15000), _subsample(res.hull, 15000, 1)
    for ax, (a, b), name in ((axes[2], (0, 2), "top view (x, z)"), (axes[3], (0, 1), "front view (x, y)")):
        if len(hull):
            ax.scatter(hull[:, a], hull[:, b], s=0.3, color="0.8", label="silhouette intersection")
        if len(pts):
            ax.scatter(pts[:, a], pts[:, b], s=0.5, color="tab:blue", label="matched points + mirror")
        if "sparse" in dbg and len(dbg["sparse"]):
            ax.scatter(dbg["sparse"][:, a], dbg["sparse"][:, b], s=4, color="tab:orange", label="feature pairs")
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    if n is not None and len(res.points):
        c = res.points.mean(axis=0)
        t_dir = np.array([-n[2], 0.0, n[0]])                 # direction of the plane in the top view
        t_dir /= max(np.linalg.norm(t_dir), 1e-9)
        ext = 0.6 * np.linalg.norm(np.ptp(res.points, axis=0))
        seg = np.array([c - ext * t_dir, c + ext * t_dir])
        axes[2].plot(seg[:, 0], seg[:, 2], color="k", lw=1, ls="--", label="symmetry plane")
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend(fontsize=6, markerscale=3, loc="best")
    axes[3].invert_yaxis()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
