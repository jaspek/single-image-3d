import numpy as np
import pytest
import trimesh

from si3d import io
from si3d.eval import (align_pred, chamfer, depth_metrics, eval_mask, evaluate_shape,
                       normal_metrics, normals_from_depth, normalize_points)
from si3d.eval.depth import METRIC_NAMES

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])


def test_depth_metrics_perfect():
    gt = np.random.default_rng(0).uniform(0.5, 9.0, (48, 64))
    m = depth_metrics(gt, gt, np.ones_like(gt, bool))
    for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "log10"):
        assert m[k] == pytest.approx(0.0)
    assert m["d1"] == m["d2"] == m["d3"] == 1.0
    assert m["n_valid"] == gt.size
    assert set(METRIC_NAMES) <= set(m)


def test_depth_metrics_scaled():
    gt = np.full((10, 10), 2.0)
    m = depth_metrics(gt * 1.3, gt, np.ones_like(gt, bool))
    assert m["abs_rel"] == pytest.approx(0.3)
    assert m["d1"] == 0.0 and m["d2"] == 1.0


def test_align_affine_recovers_scale_and_shift():
    gt = np.random.default_rng(1).uniform(1.0, 8.0, (30, 40))
    pred = 0.25 * gt - 0.1
    mask = np.ones_like(gt, bool)
    assert np.allclose(align_pred(pred, gt, mask, "affine"), gt)
    assert np.allclose(align_pred(0.5 * gt, gt, mask, "median"), gt)
    assert np.array_equal(align_pred(pred, gt, mask, "none"), pred)


def test_align_inverse_space():
    gt = np.random.default_rng(2).uniform(1.0, 8.0, (30, 40))
    disp = 3.0 / gt + 0.2  # affine in inverse depth
    out = align_pred(disp, gt, np.ones_like(gt, bool), "affine", space="inverse")
    assert np.allclose(out, gt, atol=1e-6)


def test_eval_mask_crop_and_range():
    gt = np.full((100, 100), 5.0)
    gt[0, 0] = 0.0
    gt[1, 1] = 12.0
    m = eval_mask(gt, 1e-3, 10.0, crop=(10, 90, 20, 80))
    assert m.sum() == 80 * 60
    m2 = eval_mask(gt, 1e-3, 10.0)
    assert m2.sum() == gt.size - 2


def test_normals_of_fronto_parallel_plane():
    depth = np.full((120, 160), 2.0)
    n = normals_from_depth(depth, K)
    inner = n[5:-5, 5:-5]
    assert np.allclose(inner, [0.0, 0.0, -1.0], atol=1e-6)


def test_normals_of_tilted_plane():
    # plane z = z0 + k*y in camera coordinates, normal proportional to (0, -k, 1)
    k, z0 = 0.5, 2.0
    v = np.arange(120)[:, None]
    y_over_z = (v - K[1, 2]) / K[1, 1]
    depth = np.broadcast_to(z0 / (1.0 - k * y_over_z), (120, 160)).copy()
    n = normals_from_depth(depth, K)
    expect = np.array([0.0, -k, 1.0])
    expect /= np.linalg.norm(expect)
    # orientation: must face the camera (negative dot with the viewing ray)
    inner = n[5:-5, 5:-5].reshape(-1, 3)
    cos = np.abs(inner @ expect)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 0.5
    m = normal_metrics(n, n, np.ones(depth.shape, bool))
    assert m["mean_deg"] == pytest.approx(0.0, abs=1e-6) and m["p11"] == 1.0


def test_chamfer_identical_and_shifted():
    p = np.random.default_rng(3).normal(size=(500, 3))
    m = chamfer(p, p, tau=0.05)
    assert m["chamfer"] == pytest.approx(0.0) and m["fscore"] == 1.0
    m2 = chamfer(p + [0.1, 0, 0], p, tau=0.05)
    assert m2["accuracy"] > 0.0 and m2["fscore"] < 1.0


def test_normalize_points():
    p = np.array([[0, 0, 0], [2, 0, 0], [0, 4, 0], [0, 0, 4]], float)
    q, (c, s) = normalize_points(p)
    assert np.allclose(c, [1, 2, 2]) and s == pytest.approx(6.0)
    assert np.allclose(q.max(axis=0) + q.min(axis=0), 0.0)  # centred
    assert np.linalg.norm(q.max(axis=0) - q.min(axis=0)) == pytest.approx(1.0)  # unit diagonal


def test_evaluate_shape_recovers_rigid_transform():
    mesh = trimesh.creation.box(extents=[1.0, 0.4, 0.2])
    moved = mesh.copy()
    T = trimesh.transformations.rotation_matrix(np.radians(35), [0, 1, 0])
    T[:3, 3] = [3.0, -1.0, 2.0]
    moved.apply_transform(T)
    moved.apply_scale(2.5)
    m = evaluate_shape(moved, mesh, n=3000, tau=0.05, align="icp")
    assert m["chamfer"] < 0.02
    assert m["fscore"] > 0.95
    m_none = evaluate_shape(moved, mesh, n=3000, tau=0.05, align="none")
    assert m_none["chamfer"] > m["chamfer"]


def test_io_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "OUTPUTS", tmp_path)
    d = np.random.default_rng(4).uniform(0.5, 5.0, (480, 640)).astype(np.float32)
    io.save_depth("m", 7, d, normals=np.zeros((480, 640, 3), np.float32))
    assert np.array_equal(io.load_depth("m", 7), d)
    assert io.load_normals("m", 7).shape == (480, 640, 3)
    assert io.load_depth("m", 8) is None
    pts = np.random.default_rng(5).normal(size=(100, 3)).astype(np.float32)
    io.save_shape("m", 3, pts)
    assert np.array_equal(io.load_shape("m", 3), pts)
    io.save_shape("m", 4, trimesh.creation.icosphere())
    assert isinstance(io.load_shape("m", 4), trimesh.Trimesh)


def test_read_subset(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("# header\n12  bedroom_0001\n\n7 # comment\n", encoding="utf-8")
    assert io.read_subset(f) == [12, 7]
