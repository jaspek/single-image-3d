"""Silhouette intersection with the mirror images at several symmetry planes."""
import numpy as np
import pytest

from methods.symmetry import hull, synthetic
from methods.symmetry.pipeline import Params, reconstruct
from si3d.eval import evaluate_shape


def planes_of(r):
    """The coordinate planes of a rendered object in the camera frame as (normal, offset):
    width (the symmetry plane of every test object), height, depth."""
    return [(r.R[:, k], float(r.R[:, k] @ r.t)) for k in range(3)]


@pytest.fixture(scope="module")
def table():
    return synthetic.make("table", azimuth=30.0, plain=True)


def test_reflections_form_a_group():
    planes = [(np.array([1.0, 0.0, 0.0]), 0.3), (np.array([0.0, 1.0, 0.0]), -0.2), (np.array([0.0, 0.0, 1.0]), 2.0)]
    X = np.random.default_rng(0).normal(size=(20, 3))
    for count in (1, 2, 3):
        maps = hull.reflections(planes[:count])
        assert len(maps) == 2 ** count - 1
        for A, b in maps:
            Y = X @ A.T + b
            assert np.allclose(Y @ A.T + b, X)                       # every element is its own inverse
            assert not np.allclose(Y, X)
    A, b = hull.reflections(planes[:1])[0]
    Y = X @ A.T + b
    assert np.allclose(Y[:, 0] - 0.3, 0.3 - X[:, 0]) and np.allclose(Y[:, 1:], X[:, 1:])


def test_second_plane_of_a_table_is_found(table):
    r = table
    width, _, depth = planes_of(r)
    d, cover = hull.find_plane(r.mask, r.K, [width], depth[0], depth[1] - 0.4, depth[1] + 0.4, 1.5, 4.0)
    assert cover > 0.98
    assert abs(d - depth[1]) < 0.03                                   # the table top is 1.2 wide


def test_a_chair_has_no_second_plane():
    r = synthetic.make("chair", azimuth=30.0, plain=True)
    width, _, depth = planes_of(r)
    _, cover = hull.find_plane(r.mask, r.K, [width], depth[0], depth[1] - 0.4, depth[1] + 0.4, 1.5, 4.0)
    assert cover < 0.95                                               # the back has no counterpart in front


def test_more_planes_close_the_shape(table):
    r = table
    width, _, depth = planes_of(r)
    gt = r.mesh_in_camera()
    one, cover_one = hull.surface(r.mask, r.K, [width], 1.5, 4.0, max_pixels=4000)
    two, cover_two = hull.surface(r.mask, r.K, [width, depth], 1.5, 4.0, max_pixels=4000)
    assert cover_one == 1.0 and cover_two > 0.98
    f_one = evaluate_shape(one, gt, align="camera")
    f_two = evaluate_shape(two, gt, align="camera")
    assert f_two["accuracy"] < 0.8 * f_one["accuracy"]
    assert f_two["fscore"] > f_one["fscore"]
    # every surface point and all its mirror images project into the silhouette
    x = two @ r.K.T
    u, v = np.rint(x[:, 0] / x[:, 2]).astype(int), np.rint(x[:, 1] / x[:, 2]).astype(int)
    inside = (u >= 0) & (u < r.mask.shape[1]) & (v >= 0) & (v < r.mask.shape[0])
    assert inside.mean() > 0.99
    assert r.mask[v[inside], u[inside]].mean() > 0.97


def test_pipeline_uses_the_second_plane_of_the_table(table):
    r = table
    gt = r.mesh_in_camera()
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False))
    assert res.status == "ok" and res.info["n_planes"] == 2
    assert res.info["coverage_depth"] >= 0.98 > res.info["coverage_up"]
    single = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False, more_planes=False))
    assert single.info["n_planes"] == 1
    assert (evaluate_shape(res.hull, gt, align="camera")["fscore"]
            > evaluate_shape(single.hull, gt, align="camera")["fscore"])
