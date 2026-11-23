"""Straight edges, vanishing directions and the plane normal read from them."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from methods.symmetry import lines, synthetic
from methods.symmetry.pipeline import Params, reconstruct

K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
QUAD = [(60.3, 50.2), (250.1, 80.7), (230.4, 190.2), (80.8, 170.6)]


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0)))


def polygon_image(corners, size=(240, 320), ss=8, inside=0.9, outside=0.2):
    """Grey image of a filled polygon with smooth (supersampled) edges."""
    H, W = size
    img = Image.new("F", (W * ss, H * ss), outside)
    # pixel centres are at whole numbers: the point x lies at (x + 0.5) * ss - 0.5 of the fine grid
    ImageDraw.Draw(img).polygon([((x + 0.5) * ss - 0.5, (y + 0.5) * ss - 0.5) for x, y in corners], fill=inside)
    return np.asarray(img.resize((W, H), Image.BOX), dtype=np.float32)


def disc_image(radius=80.0, size=(240, 320)):
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    return (0.2 + 0.7 / (1.0 + np.exp(np.hypot(xx - size[1] / 2, yy - size[0] / 2) - radius))).astype(np.float32)


def box_segments(azimuth, elevation=20.0, counts=(4, 4, 4), noise=0.0, seed=0, K=K, distance=3.0):
    """Image segments of the edges of a box of 1.0 x 0.8 x 0.5. Returns (segments, rotation);
    the columns of the rotation are the box axes in the camera frame. `counts` says how many
    of the four edges of each direction are used."""
    rng = np.random.default_rng(seed)
    R, t = synthetic.pose(azimuth, elevation, distance)
    h = np.array([1.0, 0.8, 0.5]) / 2.0
    a, b = [], []
    for axis in range(3):
        u, v = [k for k in range(3) if k != axis]
        for su, sv in [(-1, -1), (-1, 1), (1, -1), (1, 1)][:counts[axis]]:
            lo = np.zeros(3)
            lo[u], lo[v], lo[axis] = su * h[u], sv * h[v], -h[axis]
            hi = lo.copy()
            hi[axis] = h[axis]
            a.append(lo)
            b.append(hi)

    def project(P):
        x = (np.asarray(P) @ R.T + t) @ K.T
        return x[:, :2] / x[:, 2:] + rng.normal(0, noise, (len(P), 2))

    pa, pb = project(a), project(b)
    return lines.Segments(pa, pb, np.linalg.norm(pa - pb, axis=1), np.zeros(len(pa))), R


# ------------------------------------------------------------------------------- segments

def test_segments_of_a_quadrilateral():
    seg = lines.detect_segments(polygon_image(QUAD), min_length=30.0)
    assert len(seg) == 4
    assert seg.curved == 0.0
    for k in range(4):
        p, q = np.array(QUAD[k]), np.array(QUAD[(k + 1) % 4])
        u = (q - p) / np.linalg.norm(q - p)
        # the segment of this side: parallel to it, on it, and nearly as long
        cos = np.abs(((seg.b - seg.a) / seg.length[:, None]) @ u)
        off = np.abs((seg.mid - p) @ np.array([-u[1], u[0]]))
        j = int(np.argmax(cos - off))
        assert np.degrees(np.arccos(min(cos[j], 1.0))) < 0.3
        assert off[j] < 0.2
        assert seg.length[j] > np.linalg.norm(q - p) - 12.0


def test_a_circle_is_curved_edge_not_segments():
    seg = lines.detect_segments(disc_image(80.0), min_length=30.0)
    assert len(seg) == 0
    assert 0.6 * 2 * np.pi * 80.0 < seg.curved < 1.2 * 2 * np.pi * 80.0


def test_segments_stay_inside_the_region():
    region = np.zeros((240, 320), bool)
    region[:, :160] = True
    seg = lines.detect_segments(polygon_image(QUAD), region, min_length=30.0)
    assert len(seg) >= 1
    assert max(seg.a[:, 0].max(), seg.b[:, 0].max()) < 161.0


# ------------------------------------------------------------------- vanishing directions

def test_vanishing_directions_of_a_box():
    seg, R = box_segments(azimuth=35.0)
    dirs = lines.vanishing_directions(K, seg, min_lines=3)
    assert len(dirs) == 3
    for axis in range(3):
        assert min(angle_deg(d.d, R[:, axis]) for d in dirs) < 0.1
    assert all(d.lines == 4 for d in dirs)


def test_vanishing_directions_with_noise_and_clutter():
    seg, R = box_segments(azimuth=25.0, noise=0.3, seed=1)
    rng = np.random.default_rng(2)
    ca = rng.uniform([150, 100], [490, 380], (12, 2))
    cb = ca + rng.uniform(-80, 80, (12, 2))
    both = lines.Segments(np.r_[seg.a, ca], np.r_[seg.b, cb],
                          np.r_[seg.length, np.linalg.norm(ca - cb, axis=1)], np.zeros(len(seg) + 12))
    dirs = lines.vanishing_directions(K, both, min_lines=3)
    for axis in range(3):
        assert min(angle_deg(d.d, R[:, axis]) for d in dirs) < 2.0


def test_frame_holds_an_axis_with_a_single_edge():
    """One edge alone has no vanishing point; with the other two axes it fixes the third."""
    seg, R = box_segments(azimuth=30.0, counts=(4, 4, 1), noise=0.2, seed=3)
    dirs = lines.vanishing_directions(K, seg, min_lines=3)
    assert len(dirs) == 2
    frame = lines.fit_frame(K, seg, dirs)
    for axis in range(3):
        assert min(angle_deg(e, R[:, axis]) for e in frame.axes) < 1.0
    assert np.allclose(frame.axes @ frame.axes.T, np.eye(3), atol=1e-6)
    assert sorted(frame.lines.tolist()) == [1, 4, 4]


def test_vertical_axis_for_level_and_steep_cameras():
    for elevation in (0.0, 20.0, 55.0):
        R, _ = synthetic.pose(40.0, elevation, 3.0)
        k, off = lines.vertical_axis(R.T)                 # rows: the box axes in the camera frame
        assert k == 1 and off < 1.0
    c = np.sqrt(0.5)                                      # turned by 45 deg about the optical axis:
    rolled = np.array([[c, c, 0.0], [-c, c, 0.0], [0.0, 0.0, 1.0]])
    assert lines.vertical_axis(rolled)[1] > 20.0          # no axis can be the vertical


def test_uncertainty_grows_when_the_perspective_is_weak():
    """Edges of two directions only. With a ten times longer lens from ten times as far the
    edges of one direction are almost parallel in the image and hold the axes less firmly."""
    sigma = []
    for scale in (1.0, 10.0):
        Ks = K.copy()
        Ks[0, 0] = Ks[1, 1] = 700.0 * scale
        seg, _ = box_segments(azimuth=30.0, counts=(4, 4, 0), K=Ks, distance=3.0 * scale)
        frame = lines.fit_frame(Ks, seg, lines.vanishing_directions(Ks, seg, min_lines=3))
        sigma.append(lines.frame_uncertainty(Ks, frame.axes, seg, frame.inliers).max())
    assert sigma[1] > 4.0 * sigma[0]


# ------------------------------------------------------------------- plane of an object

@pytest.fixture(scope="module")
def plain_chair():
    return synthetic.make("chair", azimuth=30.0, plain=True)


def test_plane_from_the_edges_of_an_untextured_object(plain_chair):
    r = plain_chair
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False))
    assert res.status == "ok"
    assert res.info["plane_source"] == "lines"
    assert angle_deg(res.normal, r.plane[0]) < 1.0
    assert res.normal @ r.plane[0] > 0                    # towards the object, like the true normal
    assert len(res.hull) > 1000


def test_features_find_no_plane_on_an_untextured_object(plain_chair):
    r = plain_chair
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="features", dense=False))
    assert res.status != "ok"


def test_edges_refuse_a_textured_object_and_features_take_over():
    r = synthetic.make("cabinet", azimuth=30.0)
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False))
    assert res.status == "no_symmetry" and res.info["line_reject"] == "not_box"
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="auto", dense=False))
    assert res.status == "ok" and res.info["plane_source"] == "features"
    assert angle_deg(res.normal, r.plane[0]) < 3.0


def test_front_and_side_cannot_be_told_apart_at_45_degrees():
    r = synthetic.make("chair", azimuth=45.0, plain=True)
    res = reconstruct(r.image, r.mask, r.K, Params(plane_from="lines", dense=False))
    assert res.status == "no_symmetry" and res.info["line_reject"] == "ambiguous"


def test_round_object_has_no_plane_from_edges():
    img = disc_image(80.0)
    out = lines.plane_normal(img, None, K, np.array([0.0, 0.0, 1.0]), size=160.0)
    assert out.normal is None and out.reason in ("no_lines", "not_box")
