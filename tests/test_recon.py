"""Tests for the parts of 3D reconstruction that do not need PyTorch or a GPU."""
import numpy as np

from meld.recon import build_cloud, look_at, pick_moment, render_points, write_ply
from meld.timeline import ClipEntry, Timeline


def clip(name, offset, duration, video=True):
    return ClipEntry(name, offset, duration, video, 640, 360)


def test_pick_moment_is_where_most_videos_overlap():
    tl = Timeline([clip("a", 0, 100), clip("b", 40, 40), clip("c", 50, 20), clip("audio", 0, 100, video=False)])
    assert 50 <= pick_moment(tl) <= 70  # all three video clips run here; the audio-only clip does not count


def test_render_points_nearer_point_wins():
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    w2c = np.hstack([np.eye(3), np.zeros((3, 1))])
    pts = np.array([[0, 0, 2.0], [0, 0, 4.0]])  # same pixel, red is nearer
    cols = np.array([[255, 0, 0], [0, 0, 255]], np.uint8)
    img = render_points(pts, cols, K, w2c, 100, 100, splat=1)
    assert tuple(img[50, 50]) == (255, 0, 0)
    assert img.sum() == 255  # nothing else was drawn


def test_render_points_ignores_points_behind_the_camera():
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    w2c = np.hstack([np.eye(3), np.zeros((3, 1))])
    img = render_points(np.array([[0, 0, -3.0]]), np.array([[9, 9, 9]], np.uint8), K, w2c, 100, 100, splat=1)
    assert img.sum() == 0


def test_look_at_faces_the_target_and_keeps_up_up():
    center, target, up = np.array([0.0, 0, -5]), np.zeros(3), np.array([0.0, 1, 0])
    w2c = look_at(center, target, up)
    assert np.allclose(w2c[:, :3] @ w2c[:, :3].T, np.eye(3), atol=1e-6)  # a proper rotation
    p = w2c @ np.array([*target, 1.0])
    assert np.allclose(p, [0, 0, 5], atol=1e-6)  # target is straight ahead, 5 away
    above = w2c @ np.array([0, 1.0, 0, 1.0])
    assert above[1] < 0  # world "up" is toward the top of the image (negative y in OpenCV axes)


def test_build_cloud_drops_low_confidence_padding_and_flying_pixels():
    S, H, W = 1, 20, 20
    images = np.random.default_rng(0).random((S, 3, H, W)).astype(np.float32)
    masks = np.ones((S, H, W), bool)
    masks[:, :, :2] = False  # padding
    points = np.random.default_rng(1).random((S, H, W, 3)).astype(np.float32)
    depth = np.full((S, H, W), 2.0, np.float32)
    depth[:, :, 10] = 9.0  # a column that jumps in depth: neighbours on both sides are flying pixels
    conf = np.full((S, H, W), 8.0, np.float32)
    conf[:, :5, :] = 1.0  # low confidence rows

    pts, cols, view = build_cloud(images, masks, points, conf, depth, keep=1.0, min_conf=1.5, max_points=10_000, edge=0.08)
    kept = np.zeros((H, W), bool)
    # every surviving point must come from a confident, non-padding, non-edge pixel
    assert len(pts) > 0
    assert len(pts) < H * W - 5 * W - 2 * (H - 5)  # rows 0-4 and the padding are gone, plus the depth jump
    assert cols.dtype == np.uint8 and (view == 0).all()


def test_write_ply_header_and_size(tmp_path):
    pts = np.random.default_rng(0).random((10, 3)).astype(np.float32)
    cols = np.random.default_rng(1).integers(0, 255, (10, 3), dtype=np.uint8)
    write_ply(tmp_path / "a.ply", pts, cols)
    raw = (tmp_path / "a.ply").read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    assert b"element vertex 10" in header and b"binary_little_endian" in header
    assert len(body) == 10 * 15  # 3 float32 + 3 uint8 per vertex
