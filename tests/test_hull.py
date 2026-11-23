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


def test_pipeline_can_use_the_second_plane_of_the_table(table):
    r = table
    gt = r.mesh_in_camera()
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False, box=False, more_planes=True))
    assert res.status == "ok" and res.info["n_planes"] == 2
    assert res.info["coverage_depth"] >= 0.98 > res.info["coverage_up"]
    single = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False, box=False))
    assert single.info["n_planes"] == 1
    assert (evaluate_shape(res.hull, gt, align="camera")["fscore"]
            > evaluate_shape(single.hull, gt, align="camera")["fscore"])


# ------------------------------------------------------------- box from outline and axes

def projected_box(azimuth, elevation, half=(0.5, 0.3, 0.4), distance=4.0, size=(480, 640)):
    """Silhouette of a box of the given half sizes (width, height, depth) and what fit_box
    should find: the axes (normal of the symmetry plane, depth, height) and the faces."""
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    K = np.array([[900.0, 0.0, size[1] / 2], [0.0, 900.0, size[0] / 2], [0.0, 0.0, 1.0]])
    R, t = synthetic.pose(azimuth, elevation, distance)
    half = np.asarray(half)
    corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    px = (corners @ R.T + t) @ K.T
    px = px[:, :2] / px[:, 2:]
    img = Image.new("1", (size[1], size[0]), 0)
    ImageDraw.Draw(img).polygon([tuple(px[i]) for i in ConvexHull(px).vertices], fill=1)
    n, d = R[:, 0], float(R[:, 0] @ t)
    n, d = (n, d) if d > 0 else (-n, -d)
    axes = np.stack([n, R[:, 2], R[:, 1]])
    d = d if d > 1e-9 else 1.0                                      # (a camera in the plane has no unit)
    centre = axes @ t / d                                           # unit: distance of the camera from the plane
    return np.asarray(img, bool), K, axes, centre - half[[0, 2, 1]] / d, centre + half[[0, 2, 1]] / d


@pytest.mark.parametrize("azimuth,elevation,half", [(30.0, 20.0, (0.5, 0.3, 0.4)), (15.0, 10.0, (0.6, 0.25, 0.9)),
                                                     (40.0, 35.0, (0.3, 0.5, 0.2)), (20.0, 2.0, (0.5, 0.3, 0.8))])
def test_box_from_the_outline_of_a_box(azimuth, elevation, half):
    mask, K, axes, lo, hi = projected_box(azimuth, elevation, half)
    box = hull.fit_box(mask, K, axes)
    assert box is not None and box.misfit < 0.01
    assert np.allclose(box.lo, lo, atol=0.02 * np.abs(hi - lo).max())
    assert np.allclose(box.hi, hi, atol=0.02 * np.abs(hi - lo).max())
    assert abs(box.lo[0] + box.hi[0] - 2.0) < 1e-9                  # centred on the symmetry plane
    assert hull.box_covers(mask, K, box) > 0.99
    assert box.contains(box.corners(), margin=1e-9).all()


def test_no_box_from_a_single_face():
    """Seen straight from the front at its own height a box shows one face: no depth."""
    mask, K, axes, _, _ = projected_box(0.0, 0.0)
    assert hull.fit_box(mask, K, axes) is None


def test_box_closes_what_the_cones_leave_open():
    r = synthetic.make("cabinet", azimuth=15.0, plain=True)
    gt = r.mesh_in_camera()
    cones = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False, box=False))
    both = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False))
    assert cones.status == both.status == "ok" and cones.info["box"] == 0 and both.info["box"] == 1
    f_cones = evaluate_shape(cones.hull, gt, align="camera")["fscore"]
    f_both = evaluate_shape(both.hull, gt, align="camera")["fscore"]
    assert f_cones < 0.5 and f_both > 0.95


def test_box_alone_gives_a_shape_for_a_frontal_view():
    r = synthetic.make("chair", azimuth=5.0, plain=True)
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False))
    assert res.status == "frontal"                                  # no baseline for the mirrored camera
    res = reconstruct(r.image, r.mask, r.K, Params(dense=False, symmetry=False))
    assert res.status == "ok" and res.info["box"] == 1 and res.info["n_planes"] == 0 and len(res.points) == 0
    assert evaluate_shape(res.hull, r.mesh_in_camera(), align="camera")["fscore"] > 0.6


def test_box_alone_needs_a_box_like_object():
    r = synthetic.make("cabinet", azimuth=30.0)                     # textured: edges in all directions
    res = reconstruct(r.image, r.mask, r.K, Params(dense=False, symmetry=False))
    assert res.status == "no_frame" and res.info["line_reject"] == "not_box"
