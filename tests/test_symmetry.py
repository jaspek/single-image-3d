"""Symmetry method on synthetic data, where the symmetry plane and the shape are known."""
import numpy as np
import pytest

from methods.symmetry import features, geometry, synthetic
from methods.symmetry.dense import plane_normals
from methods.symmetry.pipeline import Params, crop_and_resize, reconstruct, to_gray
from si3d.eval import evaluate_shape

K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0)))


def mirror_pairs(azimuth, n_pts=60, noise=0.0, n_out=0, seed=0):
    """Random points of an object symmetric about x = 0, their mirror partners, both projected."""
    rng = np.random.default_rng(seed)
    P = rng.uniform([-0.5, -0.5, -0.4], [-0.02, 0.5, 0.4], (n_pts, 3))
    R, t = synthetic.pose(azimuth, 15.0, 3.0)
    X, Xm = P @ R.T + t, (P * [-1, 1, 1]) @ R.T + t
    n = R[:, 0]
    d = float(n @ t)
    n, d = (n, d) if d > 0 else (-n, -d)

    def project(Y):
        y = Y @ K.T
        return y[:, :2] / y[:, 2:]

    x1 = project(X) + rng.normal(0, noise, (n_pts, 2))
    x2 = project(Xm) + rng.normal(0, noise, (n_pts, 2))
    if n_out:
        x1 = np.r_[x1, rng.uniform([150, 80], [490, 400], (n_out, 2))]
        x2 = np.r_[x2, rng.uniform([150, 80], [490, 400], (n_out, 2))]
    return x1, x2, n, d, X, Xm


# ------------------------------------------------------------------------------ geometry

def test_reflect_is_an_involution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    n = np.array([0.6, 0.0, 0.8])
    Xm = geometry.reflect(X, n, 1.3)
    assert np.allclose(geometry.reflect(Xm, n, 1.3), X)
    assert np.allclose((X + Xm) / 2 @ n, 1.3)                    # midpoints lie on the plane


def test_mirror_partners_are_collinear_with_the_epipole():
    x1, x2, n, _, _, _ = mirror_pairs(30.0)
    assert geometry.epipolar_distance(K, n, x1, x2).max() < 1e-8


def test_triangulation_is_exact_for_exact_data():
    x1, x2, n, d, X, Xm = mirror_pairs(25.0)
    A, B, valid = geometry.triangulate(K, n, x1, x2)
    assert valid.all()
    assert np.allclose(A * d, X, atol=1e-9) and np.allclose(B * d, Xm, atol=1e-9)


def test_two_pairs_fix_the_plane():
    x1, x2, n, _, _, _ = mirror_pairs(40.0, n_pts=2, seed=3)
    r1, r2 = geometry.pixel_rays(K, x1), geometry.pixel_rays(K, x2)
    assert angle_deg(geometry.fit_normal(r1, r2), n) < 1e-6


@pytest.mark.parametrize("azimuth", [15.0, 30.0, 60.0])
def test_ransac_finds_the_plane_among_outliers(azimuth):
    x1, x2, n, _, _, _ = mirror_pairs(azimuth, noise=0.5, n_out=200, seed=1)
    fit = geometry.ransac_plane(K, x1, x2, thr=2.0)
    assert fit.found
    assert angle_deg(fit.normal, n) < 2.5
    assert fit.normal @ n > 0                                     # points end up in front of the camera
    assert fit.inliers[:60].sum() >= 45 and fit.inliers[60:].sum() <= 12


def test_ransac_rejects_unrelated_pairs():
    rng = np.random.default_rng(5)
    for seed in range(5):
        a = rng.uniform([150, 80], [490, 400], (300, 2))
        b = rng.uniform([150, 80], [490, 400], (300, 2))
        assert not geometry.ransac_plane(K, a, b, seed=seed).found


def test_matches_sliding_along_two_edges_are_not_a_plane():
    """Pairs on one straight edge agree with any epipole on that edge; two edges give a point
    that many pairs agree with, but they lie on two lines only."""
    rng = np.random.default_rng(2)
    e1 = np.c_[np.linspace(100, 300, 40), np.linspace(100, 160, 40)]
    e2 = np.c_[np.linspace(350, 500, 40), np.linspace(300, 120, 40)]
    x1 = np.r_[e1[:20], e2[:20]] + rng.normal(0, 0.3, (40, 2))
    x2 = np.r_[e1[20:], e2[20:]] + rng.normal(0, 0.3, (40, 2))
    fit = geometry.ransac_plane(K, x1, x2)
    assert fit.inliers.sum() > 15 and fit.support <= 6 and not fit.found


# ------------------------------------------------------------------------------ features

@pytest.fixture(scope="module")
def chair():
    return synthetic.make("chair", azimuth=25.0)


def test_render_is_consistent(chair):
    """The renderer itself: depth, object coordinates and projection agree."""
    ys, xs = np.nonzero(chair.mask)
    pick = np.random.default_rng(0).choice(len(ys), 200, replace=False)
    uv = chair.project(chair.points[ys[pick], xs[pick]])
    assert np.abs(uv - np.c_[xs[pick], ys[pick]]).max() < 1.0
    n, d = chair.plane
    assert d > 0 and abs(np.linalg.norm(n) - 1) < 1e-12


def test_mirror_descriptors():
    d = np.random.default_rng(0).random((5, 128)).astype(np.float32)
    assert np.array_equal(features.mirror_descriptors(features.mirror_descriptors(d)), d)
    assert not np.allclose(features.mirror_descriptors(d), d)


def test_mirrored_descriptor_equals_descriptor_of_the_flipped_image(chair):
    gray = to_gray(chair.image)
    f = features.detect_and_describe(gray, chair.mask)
    g = features.detect_and_describe(gray[:, ::-1].copy(), chair.mask[:, ::-1].copy())
    assert len(f) > 100
    flipped = np.c_[gray.shape[1] - 1 - f.xy[:, 0], f.xy[:, 1]]
    sim = features.mirror_descriptors(f.desc) @ g.desc.T
    nearest = np.linalg.norm(flipped - g.xy[sim.argmax(axis=1)], axis=1)
    assert np.mean(nearest < 1.5) > 0.9          # the best match is the same keypoint, mirrored
    assert np.median(sim.max(axis=1)) > 0.98


def test_mirror_matches_are_mostly_true_partners(chair):
    f = features.detect_and_describe(to_gray(chair.image), chair.mask)
    pairs, _ = features.mirror_matches(f, min_sep=8.0)
    assert len(pairs) >= 30 and np.all(pairs[:, 0] != pairs[:, 1])
    x1, x2 = f.xy[pairs[:, 0]], f.xy[pairs[:, 1]]
    P = chair.points[np.rint(x1[:, 1]).astype(int), np.rint(x1[:, 0]).astype(int)]
    err = np.linalg.norm(chair.project(P * [-1, 1, 1]) - x2, axis=1)
    assert np.mean(err < 3.0) > 0.4
    fit = geometry.ransac_plane(chair.K, x1, x2)
    assert fit.found and angle_deg(fit.normal, chair.plane[0]) < 2.0


# ------------------------------------------------------------------------------ pipeline

def test_crop_keeps_the_viewing_rays(chair):
    gray, mask, Kw, A = crop_and_resize(to_gray(chair.image), chair.mask, chair.K, 500)
    assert max(gray.shape) == 500 and mask.shape == gray.shape
    px = np.array([[300.0, 200.0], [350.0, 260.0]])
    moved = (np.c_[px, np.ones(2)] @ A.T)[:, :2]
    assert np.allclose(geometry.pixel_rays(chair.K, px), geometry.pixel_rays(Kw, moved))


def test_plane_normals_are_perpendicular_to_the_symmetry_normal():
    n = np.array([0.8, 0.1, -0.59])
    n /= np.linalg.norm(n)
    m = plane_normals(n, 6)
    assert np.allclose(m @ n, 0, atol=1e-12) and np.allclose(np.linalg.norm(m, axis=1), 1)
    assert m[0, 2] == pytest.approx(np.abs(m[:, 2]).max())       # first one faces the camera most


@pytest.fixture(scope="module")
def chair_result(chair):
    return reconstruct(chair.image, chair.mask, chair.K, Params(work_size=480, dense_step=4))


def test_reconstruct_chair(chair, chair_result):
    res = chair_result
    n_true, d_true = chair.plane
    assert res.status == "ok"
    assert angle_deg(res.normal, n_true) < 3.0
    assert res.info["n_dense"] > 1000 and len(res.points) == 2 * (res.info["n_sparse"] + res.info["n_dense"])
    # mirror copies: the cloud is symmetric about the estimated plane
    half = len(res.points) // 2
    assert np.allclose(geometry.reflect(res.points[:half], res.normal), res.points[half:])
    m = evaluate_shape(res.points, chair.mesh_in_camera(), align="camera")
    assert m["accuracy"] < 0.03 and m["fscore"] > 0.8
    assert abs(res.info["view_angle_deg"] - np.degrees(np.arcsin(d_true / np.linalg.norm(chair.t)))) < 5.0


def test_points_and_hull_project_into_the_mask(chair, chair_result):
    """Every reconstructed point and its mirror image lie inside the silhouette."""
    for pts in (chair_result.points, chair_result.hull):
        assert len(pts) > 0
        uv = pts @ chair.K.T
        uv = np.rint(uv[:, :2] / uv[:, 2:]).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        grown = np.zeros_like(chair.mask)
        ys, xs = np.nonzero(chair.mask)
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                grown[np.clip(ys + dy, 0, 479), np.clip(xs + dx, 0, 639)] = True
        assert np.mean(grown[uv[inside, 1], uv[inside, 0]]) > 0.98 and inside.mean() > 0.98


def test_no_plane_without_symmetry():
    tex = synthetic.SymmetricTexture(seed=1, mirror=False)
    r = synthetic.make("cabinet", azimuth=30.0, texture=tex)
    res = reconstruct(r.image, r.mask, r.K, Params(work_size=480, dense_step=4))
    assert res.status == "no_symmetry" and len(res.points) == 0


def test_empty_mask():
    res = reconstruct(np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100), bool), K)
    assert res.status == "no_object"


# ------------------------------------------------------- cases found on the real images

def _features(xy, tensor):
    n = len(xy)
    desc = np.zeros((n, 128), np.float32)
    desc[np.arange(n), np.arange(n)] = 1.0                       # nothing looks like anything else
    return features.Features(np.asarray(xy, float), np.full(n, 2.0), np.zeros(n), desc, np.arange(n),
                             np.asarray(tensor, float))


def test_no_matches_is_not_an_error():
    f = _features([[10, 10], [100, 40], [60, 90]], [[0.5, 0, 0.5]] * 3)
    pairs, dist = features.mirror_matches(f, min_sep=5.0)
    assert pairs.shape == (0, 2) and len(dist) == 0


def test_a_pair_on_one_edge_can_slide():
    """Two features on a horizontal edge (gradient along y): joined by a horizontal line they
    are not located along it, joined to a feature above they are."""
    edge, blob = [0.0, 0.0, 1.0], [0.5, 0.0, 0.5]
    f = _features([[10, 50], [90, 50], [10, 10]], [edge, edge, blob])
    along = features.along_line(f, np.array([0, 0]), np.array([1, 2]))
    assert along[0] == pytest.approx(0.0) and along[1] == pytest.approx(0.5)


def test_line_support_counts_lines_not_pairs():
    x1, x2, n, _, _, _ = mirror_pairs(30.0, n_pts=40)
    assert geometry.line_support(K, n, x1, x2) > 15
    same = geometry.line_support(K, n, np.repeat(x1[:1], 10, axis=0), np.repeat(x2[:1], 10, axis=0))
    assert same == 1 and geometry.line_support(K, n, x1[:0], x2[:0]) == 0


def test_model_symmetry_planes():
    from methods.symmetry.truth import model_symmetry
    count = {name: len(model_symmetry(synthetic.OBJECTS[name]())) for name in ("chair", "table", "cabinet")}
    assert count == {"chair": 1, "table": 2, "cabinet": 3}
    axis, centre = model_symmetry(synthetic.chair())[0]
    assert np.allclose(axis, [1, 0, 0]) and abs(centre[0]) < 1e-9


def test_reconstruct_with_a_given_plane(chair):
    n_true, _ = chair.plane
    res = reconstruct(chair.image, chair.mask, chair.K, Params(work_size=480, dense_step=4), normal=n_true)
    assert res.status == "ok" and np.allclose(res.normal, n_true)
    m = evaluate_shape(res.points, chair.mesh_in_camera(), align="camera")
    assert m["fscore"] > 0.85


def test_frontal_view_is_rejected():
    """A camera in the symmetry plane has no baseline: the plane is found, the depth is not."""
    r = synthetic.make("chair", azimuth=3.0)
    res = reconstruct(r.image, r.mask, r.K, Params(work_size=480, dense_step=4))
    assert res.status == "frontal" and len(res.points) == 0
    assert angle_deg(res.normal, r.plane[0]) < 3.0


def test_silhouette_intersection_alone(chair):
    """Without the dense search and without usable feature pairs the hull is still returned."""
    n_true, _ = chair.plane
    res = reconstruct(chair.image, chair.mask, chair.K, Params(work_size=480, dense=False, min_along=2.0),
                      normal=n_true)
    assert res.status == "ok" and len(res.points) == 0 and len(res.hull) > 1000
    assert evaluate_shape(res.hull, chair.mesh_in_camera(), align="camera")["fscore"] > 0.3
