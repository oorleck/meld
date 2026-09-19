"""3D reconstruction of one moment of the concert.

The clips are already synced, so the frame at time t of every phone that was filming shows the same instant.
Only the phones that see the same thing as another (views.py) are used. Those frames go through VGGT (a feed-forward multi-view model: camera poses + dense depth in one pass).
Outputs, in out/recon_<t>s/: scene.ply (coloured point cloud), cameras.json, viewer.html (orbit the scene and jump
to each phone's viewpoint), swing.mp4 (a virtual camera swinging around the scene) and the source frames.

Works well on well-lit, sharp photos (VGGT's own samples: median confidence ~12). Dark, saturated, low-resolution
concert video is at the edge of what it can do: on real Wembley footage the model's confidence was flat 1.0 for
every pixel, and classic structure-from-motion (COLMAP) could not verify a single camera pair either. When the
model reports that it is not confident, nothing is written (see reconstruct_frames) rather than a smeared cloud.

Needs the optional extra:  uv sync --extra recon
"""
from __future__ import annotations

import base64
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from .media import ffmpeg_exe, run_ffmpeg
from .project import Project
from .timeline import Timeline
from .videocut import GRID, score_clip

VGGT_REPO = "https://github.com/facebookresearch/vggt"
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_WEIGHTS = "facebook/VGGT-1B"
SIZE = 518  # VGGT's working resolution
GOOD_CONF = 2.0  # a pixel the model is reasonably sure about (VGGT's own samples reach 5-25)
SCAN_STEP = 10.0  # seconds between the moments that are looked at, to see which phones see the same thing


def ensure_vggt_source() -> Path:
    """VGGT's own packaging is broken, so fetch its source at a pinned commit and import it directly."""
    root = Path.home() / ".cache" / "meld" / "vggt"
    if not (root / "vggt" / "models" / "vggt.py").exists():
        root.mkdir(parents=True, exist_ok=True)
        for cmd in (
            ["git", "init", "-q"],
            ["git", "remote", "add", "origin", VGGT_REPO],
            ["git", "fetch", "-q", "--depth", "1", "origin", VGGT_COMMIT],
            ["git", "checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(cmd, cwd=root, check=False, capture_output=True)
        if not (root / "vggt" / "models" / "vggt.py").exists():
            raise RuntimeError(f"Could not fetch VGGT from {VGGT_REPO} into {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# ---------------------------------------------------------------- choosing the moment and the views


def pick_moment(tl: Timeline, margin: float = 0.5) -> float:
    """The time at which the most video clips are running (middle of that stretch)."""
    vids = [c for c in tl.clips if c.has_video]
    if not vids:
        raise SystemExit("None of the aligned clips has a video track.")
    times = np.arange(0.0, tl.end, 0.5)
    counts = np.zeros(len(times), int)
    for c in vids:
        counts += (times >= c.offset + margin) & (times <= c.end - margin)
    return float(np.median(times[counts == counts.max()]))


def select_views(project: Project, tl: Timeline, t: float, max_views: int, margin: float = 0.5) -> list[int]:
    """Indices of the video clips running at t, best-looking first, at most max_views."""
    live = [i for i, c in enumerate(tl.clips) if c.has_video and c.offset + margin <= t <= c.end - margin]
    if len(live) <= max_views:
        return live

    def quality(i: int) -> float:
        s = score_clip(project, tl.clips[i])
        k = int(round((t - tl.clips[i].offset) / GRID))
        return float(np.mean(s[max(0, k - 1): k + 2])) if len(s) else 0.0

    return sorted(sorted(live, key=quality, reverse=True)[:max_views])


def extract_frames(project: Project, tl: Timeline, views: list[int], t: float, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, i in enumerate(views):
        e = tl.clips[i]
        dst = out_dir / f"{n:02d}_{Path(e.file).stem}.jpg"
        run_ffmpeg(["-ss", f"{max(t - e.offset, 0):.3f}", "-i", project.clip_path(e), "-frames:v", "1", "-q:v", "2", dst])
        paths.append(dst)
    return paths


# ---------------------------------------------------------------- model


def preprocess(paths: list[Path]):
    """Fit each frame inside a 518x518 canvas (dimensions multiples of 14, white padding).

    Returns float32 images (S, 3, 518, 518) in [0,1], a validity mask (S, 518, 518) that is False on padding,
    and each frame's placement (x0, y0, w, h) inside the canvas.
    """
    from PIL import Image

    imgs, masks, place = [], [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w >= h:
            nw, nh = SIZE, max(14, round(h * SIZE / w / 14) * 14)
        else:
            nh, nw = SIZE, max(14, round(w * SIZE / h / 14) * 14)
        arr = np.asarray(im.resize((nw, nh), Image.BICUBIC), np.float32) / 255.0
        canvas = np.ones((SIZE, SIZE, 3), np.float32)
        mask = np.zeros((SIZE, SIZE), bool)
        y0, x0 = (SIZE - nh) // 2, (SIZE - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = arr
        mask[y0:y0 + nh, x0:x0 + nw] = True
        imgs.append(canvas.transpose(2, 0, 1))
        masks.append(mask)
        place.append((x0, y0, nw, nh))
    return np.stack(imgs), np.stack(masks), place


def run_vggt(images: np.ndarray, log=print):
    """images (S,3,H,W) float32 -> extrinsic (S,3,4) cam-from-world, intrinsic (S,3,3), points (S,H,W,3), conf (S,H,W), depth (S,H,W)."""
    try:
        import torch
    except ImportError:
        raise SystemExit("3D reconstruction needs the recon extra:  uv sync --extra recon") from None
    ensure_vggt_source()
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    if not torch.cuda.is_available():
        log("WARNING: no CUDA GPU found; this will be very slow.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading VGGT ({VGGT_WEIGHTS}); the first run downloads about 5 GB of weights ...")
    model = VGGT.from_pretrained(VGGT_WEIGHTS).eval().to(device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    x = torch.from_numpy(images).to(device)
    log(f"Reconstructing from {len(x)} views on {device} ...")
    with torch.no_grad():
        with torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
            preds = model(x)
        extr, intr = pose_encoding_to_extri_intri(preds["pose_enc"].float(), x.shape[-2:])
        depth = preds["depth"].float()[0].cpu().numpy()
        conf = preds["depth_conf"].float()[0].cpu().numpy()
    extr, intr = extr[0].cpu().numpy(), intr[0].cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, extr, intr)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return extr, intr, points, conf, depth[..., 0]


# ---------------------------------------------------------------- point cloud


def confidence_summary(conf: np.ndarray, masks: np.ndarray) -> dict:
    c = conf[masks]
    return {"median": float(np.median(c)), "p90": float(np.quantile(c, 0.9)), "good": float((c >= GOOD_CONF).mean())}


def build_cloud(images, masks, points, conf, depth, keep: float, min_conf: float, max_points: int, edge: float = 0.08, seed: int = 0):
    """Keep confident real (non-padding) pixels, drop depth-discontinuity streaks, trim far outliers, subsample."""
    S = len(images)
    colors = (images.transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    ok = masks & np.isfinite(points).all(-1) & (depth > 0)
    if ok.any():
        ok &= conf >= max(min_conf, np.quantile(conf[ok], 1.0 - keep))
    # "flying pixels": depth jumps between neighbours smear foreground into background
    jump = np.zeros_like(ok)
    dy = np.abs(np.diff(depth, axis=1)) / np.maximum(depth[:, 1:], 1e-6) > edge
    dx = np.abs(np.diff(depth, axis=2)) / np.maximum(depth[:, :, 1:], 1e-6) > edge
    jump[:, 1:] |= dy
    jump[:, :-1] |= dy
    jump[:, :, 1:] |= dx
    jump[:, :, :-1] |= dx
    ok &= ~jump
    view_id = np.broadcast_to(np.arange(S)[:, None, None], ok.shape)[ok]
    pts, cols = points[ok].astype(np.float32), colors[ok]
    if len(pts):
        d = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
        near = d <= np.quantile(d, 0.985)
        pts, cols, view_id = pts[near], cols[near], view_id[near]
    if len(pts) > max_points:
        pick = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts, cols, view_id = pts[pick], cols[pick], view_id[pick]
    return pts, cols, view_id


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec = np.empty(len(pts), dt)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


# ---------------------------------------------------------------- rendering a fly-around


def render_points(pts, cols, K, w2c, W, H, splat: int = 3) -> np.ndarray:
    """Z-buffered point splatting. K 3x3 intrinsics, w2c 3x4 or 4x4 camera-from-world. Returns (H,W,3) uint8."""
    P = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = P[:, 2]
    front = z > 1e-3
    P, c, z = P[front], cols[front], z[front]
    u = np.round(K[0, 0] * P[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * P[:, 1] / z + K[1, 2]).astype(np.int64)
    inside = (u >= -splat) & (u < W + splat) & (v >= -splat) & (v < H + splat)
    u, v, c, z = u[inside], v[inside], c[inside], z[inside]
    order = np.argsort(-z)  # far to near: later (nearer) writes win
    u, v, c = u[order], v[order], c[order]
    img = np.zeros((H, W, 3), np.uint8)
    r = splat // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            uu, vv = u + dx, v + dy
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[m], uu[m]] = c[m]
    return img


def look_at(center: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-from-world (3x4, OpenCV axes: x right, y down, z forward) for a camera at `center` looking at `target`."""
    z = target - center
    z = z / np.linalg.norm(z)
    y_down = -up / np.linalg.norm(up)
    x = np.cross(y_down, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows are the camera axes in world coordinates
    return np.hstack([R, (-R @ center)[:, None]])


def rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return v * math.cos(angle) + np.cross(axis, v) * math.sin(angle) + axis * (axis @ v) * (1 - math.cos(angle))


def render_swing(pts, cols, cams: list[dict], ref: int, path: Path, size=(1280, 720), seconds=8.0, swing_deg=14.0, fps=30, log=print):
    """A virtual camera starting at a phone's viewpoint and swinging side to side around the scene."""
    W, H = size
    c = cams[ref]
    w2c = np.array(c["w2c"])
    c2w_R, center = w2c[:, :3].T, -w2c[:, :3].T @ w2c[:, 3]
    up = -c2w_R[:, 1]  # camera y points down
    # focus: median position of the points this camera can see in front of it
    cam_pts = pts @ w2c[:, :3].T + w2c[:, 3]
    ahead = pts[(cam_pts[:, 2] > 0) & (np.abs(cam_pts[:, 0] / np.maximum(cam_pts[:, 2], 1e-6)) < 0.6)]
    focus = np.median(ahead, axis=0) if len(ahead) else np.median(pts, axis=0)
    f = c["fy"] * H / c["h"]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    n = int(seconds * fps)
    proc = subprocess.Popen(
        [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    try:
        for k in range(n):
            theta = math.radians(swing_deg) * math.sin(2 * math.pi * k / n)
            cam_center = focus + rotate_about(center - focus, up, theta)
            frame = render_points(pts, cols, K, look_at(cam_center, focus, up), W, H)
            proc.stdin.write(frame.tobytes())
            if (k + 1) % 60 == 0:
                log(f"  swing {k + 1}/{n}")
    finally:
        proc.stdin.close()
        proc.wait()


# ---------------------------------------------------------------- viewer

_VIEWER = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>meld 3D</title>
<style>
html,body{margin:0;height:100%;background:#0b0b10;color:#ddd;font:13px system-ui,sans-serif;overflow:hidden}
#ui{position:fixed;top:8px;left:8px;width:250px;max-height:calc(100vh - 16px);overflow:auto;background:#000b;padding:10px;border-radius:8px}
#ui h1{font-size:14px;margin:0 0 6px}#ui p{margin:4px 0;color:#aaa}
button{display:block;width:100%;margin:3px 0;text-align:left;background:#20222b;color:#ddd;border:1px solid #444;padding:5px 7px;border-radius:5px;cursor:pointer}
button:hover{background:#2c3040}label{display:block;margin-top:8px}input[type=range]{width:100%}
#pip{position:fixed;right:10px;bottom:10px;max-width:36vw;max-height:36vh;border:2px solid #fff9;border-radius:4px;display:none}
#msg{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);color:#ccc}
</style></head>
<body><div id="msg">Loading 3D viewer...</div>
<div id="ui"><h1>meld: one moment in 3D</h1>
<p>Drag to orbit, scroll to zoom, right-drag to pan. Pick a phone to see the scene from where it stood.</p>
<button id="overview">Overview</button><div id="cams"></div>
<label>Point size <input id="size" type="range" min="1" max="30" value="8"></label></div>
<img id="pip">
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const u8 = s => { const b = atob(s), a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a; };
const pos = new Float32Array(u8(D.pos).buffer), col = u8(D.col);

const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(devicePixelRatio); renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0b10);
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, D.size * 0.001, D.size * 50);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;

// The data is in OpenCV axes (y down, z forward); a half-turn about x makes it three.js-friendly.
const root = new THREE.Group(); root.rotation.x = Math.PI; scene.add(root);
const G = new THREE.Matrix4().makeRotationX(Math.PI);

const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('color', new THREE.BufferAttribute(col, 3, true));
const mat = new THREE.PointsMaterial({size: D.size * 0.0008 * 8, vertexColors: true, sizeAttenuation: true});
root.add(new THREE.Points(geo, mat));

const cams = document.getElementById('cams'), pip = document.getElementById('pip');
D.cams.forEach((c, i) => {
  const g = new THREE.Group(); g.matrixAutoUpdate = false; g.matrix.set(...c.c2w.flat());
  const d = D.size * 0.05, h = d * Math.tan(c.fov * Math.PI / 360), w = h * c.aspect;  // frustum drawn d in front of the phone
  const corners = [[-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]], v = [];
  for (let k = 0; k < 4; k++) { v.push(0,0,0, ...corners[k]); v.push(...corners[k], ...corners[(k+1)%4]); }
  g.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(v, 3)), new THREE.LineBasicMaterial({color: 0xffd23f})));
  root.add(g);
  const b = document.createElement('button'); b.textContent = (i + 1) + '. ' + c.name; b.onclick = () => look(i); cams.appendChild(b);
});

const tmpS = new THREE.Vector3();
function place(m, fov) {
  m.decompose(camera.position, camera.quaternion, tmpS);
  camera.fov = fov; camera.updateProjectionMatrix();
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(fwd, D.size * 0.5); controls.update();
}
function look(i) {
  const c = D.cams[i]; const m = new THREE.Matrix4().set(...c.c2w.flat());
  place(G.clone().multiply(m).multiply(G), c.fov);
  pip.src = 'data:image/jpeg;base64,' + c.thumb; pip.style.display = 'block';
}
function overview() {
  const c = D.cams[0]; const m = G.clone().multiply(new THREE.Matrix4().set(...c.c2w.flat())).multiply(G);
  place(m, 60);
  const back = new THREE.Vector3(0, 0, 1).applyQuaternion(camera.quaternion);
  camera.position.addScaledVector(back, D.size * 0.6);
  const f = new THREE.Vector3(...D.focus); f.y *= -1; f.z *= -1; controls.target.copy(f); controls.update();
  pip.style.display = 'none';
}
document.getElementById('overview').onclick = overview;
document.getElementById('size').oninput = e => { mat.size = D.size * 0.0008 * e.target.value; };
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
overview(); document.getElementById('msg').remove();
(function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();
</script></body></html>
"""


def write_viewer(path: Path, pts, cols, cams: list[dict], frames: list[Path], names: list[str], view_max_points: int = 700_000) -> None:
    from PIL import Image

    if len(pts) > view_max_points:
        pick = np.random.default_rng(1).choice(len(pts), view_max_points, replace=False)
        pts, cols = pts[pick], cols[pick]
    center = np.median(pts, axis=0)
    size = float(np.quantile(np.linalg.norm(pts - center, axis=1), 0.95)) if len(pts) else 1.0
    out_cams = []
    for c, fp, name in zip(cams, frames, names):
        im = Image.open(fp).convert("RGB")
        im.thumbnail((420, 420))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        w2c = np.vstack([np.array(c["w2c"]), [0, 0, 0, 1]])
        out_cams.append({
            "name": name, "c2w": np.linalg.inv(w2c).tolist(), "fov": c["fov_y"], "aspect": c["aspect"],
            "thumb": base64.b64encode(buf.getvalue()).decode("ascii"),
        })
    data = {
        "pos": base64.b64encode(np.ascontiguousarray(pts, np.float32).tobytes()).decode("ascii"),
        "col": base64.b64encode(np.ascontiguousarray(cols, np.uint8).tobytes()).decode("ascii"),
        "cams": out_cams, "size": size, "focus": center.tolist(),
    }
    path.write_text(_VIEWER.replace("__DATA__", json.dumps(data)), encoding="utf-8")


# ---------------------------------------------------------------- driver


def reconstruct_frames(
    frames: list[Path],
    names: list[str],
    out: Path,
    keep: float = 0.5,
    min_conf: float = 1.5,
    max_points: int = 1_000_000,
    video: bool = True,
    size=(1280, 720),
    force: bool = False,
    log=print,
) -> Path | None:
    """Reconstruct the scene shown by these photos (one per camera). Returns the output dir, or None if refused."""
    out.mkdir(parents=True, exist_ok=True)
    images, masks, place = preprocess(frames)
    extr, intr, points, conf, depth = run_vggt(images, log)

    q = confidence_summary(conf, masks)
    log(f"Model confidence: median {q['median']:.2f}, 90th percentile {q['p90']:.2f}, {q['good']:.0%} of pixels >= {GOOD_CONF}")
    if q["good"] < 0.03 and not force:
        log(
            "The model is not confident about this scene, so the 3D result would be unreliable and was NOT written.\n"
            "  Typical causes: dark or blurry footage, low resolution, very different zoom levels between phones, or\n"
            "  too little of the same scenery visible in several views. Try --at with a brighter moment, fewer\n"
            "  --max-views, or pass --force to write it anyway."
        )
        return None

    pts, cols, view_id = build_cloud(images, masks, points, conf, depth, keep, min_conf, max_points)
    log(f"Point cloud: {len(pts):,} points.")
    if not len(pts):
        log("No points survived the confidence filter.")
        return None
    write_ply(out / "scene.ply", pts, cols)

    cams = []
    for k in range(len(frames)):
        x0, y0, w, h = place[k]
        fy = float(intr[k][1, 1])
        cams.append({
            "clip": names[k], "frame": Path(frames[k]).name, "w2c": extr[k].tolist(),
            "fx": float(intr[k][0, 0]), "fy": fy, "cx": float(intr[k][0, 2]), "cy": float(intr[k][1, 2]),
            "w": w, "h": h, "aspect": w / h, "fov_y": math.degrees(2 * math.atan(h / 2 / fy)),
            "points": int((view_id == k).sum()),
        })
    (out / "cameras.json").write_text(json.dumps({"confidence": q, "cameras": cams}, indent=2), encoding="utf-8")
    write_viewer(out / "viewer.html", pts, cols, cams, list(frames), [Path(n).stem for n in names])

    if video:
        landscape = [k for k, c in enumerate(cams) if c["aspect"] >= 1.2] or list(range(len(cams)))
        ref = max(landscape, key=lambda k: cams[k]["points"])
        log(f"Rendering the swing video from view {ref + 1} ({names[ref]}) ...")
        render_swing(pts, cols, cams, ref, out / "swing.mp4", size=size, log=log)
    log(f"3D reconstruction -> {out}")
    log(f"  open {out / 'viewer.html'} in a browser (needs internet for three.js), or {out / 'scene.ply'} in MeshLab/CloudCompare")
    return out


def best_moment(project: Project, tl: Timeline, max_views: int, tries: int = 6, log=print):
    """The moment at which the most phones see the same thing, looked for among those with the most phones filming
    (at most `tries` of them, no two within 20 s of each other). Returns a views.Moment."""
    from . import views as seeing

    crowded = sorted(seeing.candidate_times(tl, step=SCAN_STEP, at_least=2), key=lambda t: (-len(seeing.live_clips(tl, t)), t))
    chosen: list[float] = []
    for t in crowded:
        if all(abs(t - u) >= 20.0 for u in chosen):
            chosen.append(t)
        if len(chosen) >= tries:
            break
    if not chosen:
        raise SystemExit("No moment has two phones filming it.")
    log(f"Looking at which phones see the same thing at {len(chosen)} moments ...")
    return seeing.scan(project, tl, chosen, max_views, log=log)[0]


def reconstruct(
    project: Project,
    at: float | None = None,
    max_views: int = 24,
    keep: float = 0.5,
    min_conf: float = 1.5,
    max_points: int = 1_000_000,
    video: bool = True,
    size=(1280, 720),
    force: bool = False,
    log=print,
    scan: bool = False,
) -> Path | None:
    """Reconstruct one moment. Only the phones that see the same thing (views.py) are used: phones are not picked for how
    sharp they are, as they were, since a wide shot from the crowd and a close-up of the drummer have nothing in common
    and the model then has nothing to go on. `scan` only lists the moments, the most promising first."""
    from . import views as seeing

    tl = Timeline.load(project.timeline_path)
    if scan:
        times = seeing.candidate_times(tl, step=SCAN_STEP)
        if not times:
            raise SystemExit("No moment has three phones filming it.")
        log(f"{len(times)} moments have at least {seeing.MIN_VIEWS} phones filming. Which see the same thing:")
        moments = seeing.scan(project, tl, times, max_views, log=log)
        log("\nThe most promising, best first (use --at SECONDS):")
        for m in moments[:8]:
            log(f"  {m.t:7.1f}s  {len(m.best)} of {len(m.clips)} phones see the same thing, held together by {m.strength} matches")
        return None

    if at is None:
        moment = best_moment(project, tl, max_views, log=log)
    else:
        moment = seeing.look_at(project, tl, at, max_views)
    t = moment.t
    log(f"Moment t={t:.1f}s: {len(moment.clips)} phones filming it, {len(moment.best)} of them see the same thing.")
    if len(moment.best) < seeing.MIN_VIEWS and not force:
        raise SystemExit(
            f"At t={t:.1f}s only {len(moment.best)} of the {len(moment.clips)} phones filming see the same thing as another "
            f"(a 3D model needs at least {seeing.MIN_VIEWS} that do): the rest look at other parts of the scene, from too "
            f"far apart or at a different zoom. Try `--scan` to find a better moment, or `--force` to try anyway."
        )
    views = moment.best if len(moment.best) >= 2 else moment.clips
    out = project.out_dir / f"recon_{t:.0f}s"
    frames = extract_frames(project, tl, views, t, out / "frames")
    return reconstruct_frames(frames, [tl.clips[i].file for i in views], out, keep, min_conf, max_points, video, size, force, log)
