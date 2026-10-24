#!/usr/bin/env python
"""
apply_video_motion.py — drive a rigged GLB with human motion extracted from a video.

    /opt/roblox/venv/bin/python apply_video_motion.py \
        --glb scene6.glb --video opening.mp4 --scan pose_scan.json \
        --start 238 --end 248 --out out.mp4

Pipeline
    pose landmarks (MediaPipe, per frame)
      -> canonical space (Y up, metres, facing +Z)
      -> bone directions (+ body yaw from the shoulder line)
      -> swing-only retarget onto the rig
      -> forward kinematics -> linear blend skinning -> foot grounding
      -> software rasterizer (C via ctypes) + supersampling
      -> composite with the source video -> mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glb as glblib                          # noqa: E402
import retarget as rt                         # noqa: E402
from raster import Camera, render, downscale  # noqa: E402

# rig bone node name -> direction key produced by rt.Retargeter.bone_directions
# NOTE: only ONE bone per chain may consume a given direction target. If both
# spine and torso (or neck and head) are mapped, the child re-applies the same
# swing on top of the parent's and the chain folds double. Deeper bones inherit.
BONE_MAP = {
    "spine_Armature": "spine",
    # No head/neck target on purpose: BlazePose gives no reliable head orientation
    # (nose/ear depth is noisy, and the rig's head bone is +Z while the landmarks are
    # mostly Y). Left at bind, the head inherits the spine tilt and stays stable --
    # which is what you want. Re-enable "head_Armature": "head" if you have real
    # head pose data.
    "arm_l_Armature": "arm_l",
    "forearm_l_Armature": "elbow_l",
    "hand_l_Armature": "hand_l",
    "arm_r_Armature": "arm_r",
    "forearm_r_Armature": "elbow_r",
    "hand_r_Armature": "hand_r",
    "thigh_l_Armature": "thigh_l",
    "tibia_l_Armature": "knee_l",
    "foot_l_Armature": "foot_l",
    "thigh_r_Armature": "thigh_r",
    "tibia_r_Armature": "knee_r",
    "foot_r_Armature": "foot_r",
}

FEET_MESH_HINT = ("Cloth", "Bag")     # meshes that can contain the lowest point



# --------------------------------------------------------------------------- #
# landmark sequence handling
# --------------------------------------------------------------------------- #
def load_scan(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def landmark_sequence(scan: dict, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, 33, 3) landmarks and (T, 33) visibility for [start, end], gaps filled."""
    frames = {r["i"]: r for r in scan["frames"]}
    T = end - start + 1
    raw = np.full((T, 33, 3), np.nan)
    vis = np.zeros((T, 33))
    has = np.zeros(T, bool)
    for k, f in enumerate(range(start, end + 1)):
        r = frames.get(f, {})
        if r.get("det") and r.get("world"):
            raw[k] = np.asarray(r["world"], dtype=np.float64)
            vis[k] = np.asarray(r["vis"], dtype=np.float64)
            has[k] = True

    if not has.any():
        raise SystemExit(f"no pose detections in frames {start}..{end}")

    idx = np.arange(T)
    for j in range(33):
        for c in range(3):
            good = has & np.isfinite(raw[:, j, c])
            if good.sum() == 0:
                raw[:, j, c] = 0.0
            elif good.sum() == 1:
                raw[:, j, c] = raw[good, j, c][0]
            else:
                raw[:, j, c] = np.interp(idx, idx[good], raw[good, j, c])
    return raw, vis


def smooth_landmarks(raw: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(raw) < window:
        return raw
    k = np.ones(window) / window
    pad = window // 2
    out = raw.copy()
    for j in range(raw.shape[1]):
        for c in range(3):
            s = np.pad(raw[:, j, c], pad, mode="edge")
            out[:, j, c] = np.convolve(s, k, mode="valid")
    return out


def root_track(raw: np.ndarray, depth_scale: float = 2.0) -> np.ndarray:
    """Pelvis travel in metres (T, 3); y is always 0 (grounding is done per frame)."""
    can = np.empty_like(raw)
    can[:, :, 0] = raw[:, :, 0]
    can[:, :, 1] = -raw[:, :, 1]
    can[:, :, 2] = -raw[:, :, 2] * depth_scale
    sw = np.linalg.norm(can[:, rt.L_SHOULDER] - can[:, rt.R_SHOULDER], axis=1)
    ok = sw > 1e-6
    sw = np.where(ok, sw, np.median(sw[ok]) if ok.any() else 1.0)
    can = can * (0.36 / sw)[:, None, None]
    hips = 0.5 * (can[:, rt.L_HIP] + can[:, rt.R_HIP])
    hips -= hips[0]
    hips[:, 1] = 0.0
    hips[:, 2] *= 0.35          # depth travel is the least reliable axis
    return hips


def body_yaw(canon_seq: np.ndarray, window: int = 9) -> np.ndarray:
    """Yaw (radians) of the subject about +Y, from the shoulder line.

    forward ~ cross(shoulder_line, up); smoothed and relative to the first frame.
    """
    Y = np.array([0.0, 1.0, 0.0])
    yaw = np.zeros(len(canon_seq))
    for k, c in enumerate(canon_seq):
        s = c[rt.R_SHOULDER] - c[rt.L_SHOULDER]
        s[1] = 0.0
        if np.linalg.norm(s) < 1e-6:
            yaw[k] = yaw[k - 1] if k else 0.0
            continue
        s /= np.linalg.norm(s)
        fwd = np.cross(s, Y)          # right-hand rule: +Z forward when s points +X
        yaw[k] = np.arctan2(fwd[0], fwd[2])
    yaw -= yaw[0]
    yaw = np.unwrap(yaw)
    if window > 1 and len(yaw) >= window:
        k = np.ones(window) / window
        yaw = np.convolve(np.pad(yaw, window // 2, mode="edge"), k, mode="valid")
    return yaw


# --------------------------------------------------------------------------- #
# FK on the rig
# --------------------------------------------------------------------------- #
class Rig:
    def __init__(self, scene: glblib.Scene):
        self.scene = scene
        self.children: dict[int, list[int]] = {}
        for i, n in enumerate(scene.nodes):
            for c in n.get("children", []):
                self.children.setdefault(i, []).append(c)
        self.order: list[int] = []
        self.parent: dict[int, int] = {}
        for r in scene.root_nodes:
            self._walk(r, None)
        self.bind_R = {i: scene.global_bind[i][:3, :3].copy() for i in self.order}
        self.bind_p = {i: scene.global_bind[i][:3, 3].copy() for i in self.order}
        self.name_to_idx = {n.get("name"): i for i, n in enumerate(scene.nodes)}
        self.bone_nodes = {self.name_to_idx[b]: k for b, k in BONE_MAP.items()
                           if b in self.name_to_idx}
        self.root_bone = self.name_to_idx.get("hips_Armature", self.order[0])

    def _walk(self, i: int, parent: int | None) -> None:
        self.order.append(i)
        if parent is not None:
            self.parent[i] = parent
        for c in self.children.get(i, []):
            self._walk(c, i)

    def pose(self, dirs: dict[str, np.ndarray],
             root_offset: np.ndarray | None = None,
             yaw: float = 0.0,
             ground_offset: float | None = None) -> dict[int, np.ndarray]:
        """World-space swing rotations -> global matrices for every node."""
        delta: dict[int, np.ndarray] = {}
        pos: dict[int, np.ndarray] = {}
        out: dict[int, np.ndarray] = {}

        cy, sy = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        pivot = self.bind_p[self.root_bone]

        for i in self.order:
            par = self.parent.get(i)
            d_par = delta[par] if par is not None else np.eye(3)
            p_par = pos[par] if par is not None else np.zeros(3)

            if par is not None:
                p = p_par + d_par @ (self.bind_p[i] - self.bind_p[par])
            else:
                p = self.bind_p[i].copy()

            key = self.bone_nodes.get(i)
            d = d_par
            if key and key in dirs:
                target = dirs[key]
                nxt = self._bone_child(i)
                if nxt is not None:
                    bind_dir = self.bind_p[nxt] - self.bind_p[i]
                    if np.linalg.norm(bind_dir) > 1e-6:
                        cur = d_par @ (bind_dir / np.linalg.norm(bind_dir))
                        d = rt.swing_matrix(cur, target) @ d_par

            if i == self.root_bone:
                if root_offset is not None:
                    p = p + np.asarray(root_offset, dtype=np.float64)
                if abs(yaw) > 1e-9:
                    p = pivot + R_yaw @ (p - pivot)
                    d = R_yaw @ d
                if ground_offset is not None:
                    p = p + np.array([0.0, ground_offset, 0.0])

            delta[i] = d
            pos[i] = p
            M = np.eye(4)
            M[:3, :3] = d @ self.bind_R[i]
            M[:3, 3] = p
            out[i] = M
        return out

    def _bone_child(self, i: int) -> int | None:
        stack = list(self.children.get(i, []))
        while stack:
            c = stack.pop(0)
            if np.linalg.norm(self.bind_p[c] - self.bind_p[i]) > 1e-3:
                return c
            stack.extend(self.children.get(c, []))
        return None


# --------------------------------------------------------------------------- #
# scene props
# --------------------------------------------------------------------------- #
def floor_mesh(size=10.0, y=0.0, squares=10, cells=20):
    """Subdivided checkerboard floor.

    Subdivision matters: the rasterizer rejects a triangle if any vertex is behind
    the camera, so one big quad would vanish as soon as it crossed the near plane.
    """
    n = cells
    xs = np.linspace(-size / 2, size / 2, n + 1)
    verts, uvs = [], []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append([xs[i], y, xs[j]])
            uvs.append([(i / n) * squares, (j / n) * squares])
    idx = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + (n + 1), a + (n + 1) + 1
            idx += [[a, c, b], [b, c, d]]
    verts = np.array(verts, dtype=np.float32)
    normals = np.tile([0.0, 1.0, 0.0], (len(verts), 1)).astype(np.float32)
    return verts, normals, np.array(uvs, dtype=np.float32), np.array(idx, dtype=np.int32).ravel()


def checker_texture(n=512, squares=20, a=(132, 140, 168), b=(92, 100, 126)):
    step = n // squares
    xs = np.arange(n) // step
    img = np.where(((xs[:, None] + xs[None, :]) % 2)[..., None] == 0,
                   np.array(a, np.uint8), np.array(b, np.uint8)).astype(np.uint8)
    return img


def backdrop(width: int, height: int, top=(26, 30, 44), bottom=(58, 64, 86)) -> np.ndarray:
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    return (np.array(top) * (1 - t) + np.array(bottom) * t).astype(np.uint8) * np.ones((1, width, 1), np.uint8)


def composite(base_rgb: np.ndarray, layer_rgb: np.ndarray) -> np.ndarray:
    """Alpha-less compositing: `layer` is BG-agnostic; alpha lives in slot 3."""
    return base_rgb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=30.0)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--yaw", type=float, default=1.0, help="0..1 weight of body yaw tracking")
    ap.add_argument("--layout", choices=["split", "model", "overlay"], default="split")
    args = ap.parse_args()

    t_start = time.time()
    scan = load_scan(args.scan)
    raw, vis = landmark_sequence(scan, args.start, args.end)
    raw_s = smooth_landmarks(raw, args.smooth)
    T = len(raw_s)

    scene = glblib.load(args.glb)
    rig = Rig(scene)
    ret = rt.Retargeter({})

    canon_seq = np.stack([rt.to_canonical(raw_s[k], vis[k])[0] for k in range(T)])
    yaws = body_yaw(canon_seq) * float(np.clip(args.yaw, 0.0, 1.0))
    track = root_track(raw)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")

    W, H = args.width, args.height
    ss = max(1, args.ssaa)
    cam = Camera(eye=(0.0, 0.95, 3.55), target=(0.0, 0.80, 0.0),
                 fov_deg=args.fov, width=W * ss, height=H * ss)

    fverts, fnorm, fuv, fidx = floor_mesh()
    ftex = checker_texture()

    # precompute which meshes can hold the lowest point
    foot_meshes = [m for m in scene.meshes
                   if any(h in scene.materials[m.material].name for h in FEET_MESH_HINT)]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 24.0, (W, H))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open")

    panel_w = W // 2 if args.layout == "split" else W
    times = []

    for k in range(T):
        t0 = time.time()
        dirs = ret.bone_directions(canon_seq[k], vis[k])
        glob = rig.pose(dirs, root_offset=track[k], yaw=float(yaws[k]))

        # tracking shot: keep the character centred as she travels
        cam.eye = np.array([track[k][0], 0.95, 3.55])
        cam.target = np.array([track[k][0], 0.80, 0.0])

        # --- foot grounding -------------------------------------------------
        # translate the whole skeleton vertically so the lowest foot sits on y=0
        lowest = np.inf
        for mesh in foot_meshes:
            skin = scene.skins[scene.nodes[mesh.node]["skin"]]
            jm = np.stack([glob[j] for j in skin.joints])
            v = glblib.skin_mesh(mesh, skin, jm)
            lowest = min(lowest, float(v[:, 1].min()))
        dy = -lowest if np.isfinite(lowest) else 0.0
        if dy:
            for M in glob.values():
                M[1, 3] += dy

        # --- render ---------------------------------------------------------
        img = np.zeros((H * ss, W * ss, 3), np.uint8)
        img[:] = (26, 30, 42) if ss == 1 else (26, 30, 42)
        img[:] = backdrop(W * ss, H * ss)

        o, _ = render(fverts, fnorm, fuv, fidx, ftex, cam, bg=(0, 0, 0),
                      depth_cue=False, cull=False)
        m = o[:, :, 3] > 0
        img[m] = o[:, :, :3][m]

        # contact shadow: a dark blob where the pelvis projects onto the floor
        hip = glob[rig.root_bone][:3, 3]
        view = cam.view_matrix()
        pc = view @ np.array([hip[0], 0.0, hip[2], 1.0])
        if pc[2] < -0.1:
            zc = -pc[2]
            fx = cam.intrinsics()[0, 0]
            u = int(fx * pc[0] / zc + cam.W / 2)
            v = int(-fx * pc[1] / zc + cam.H / 2)
            rx, ry = int(0.34 * fx / zc), int(0.11 * fx / zc)
            if rx > 1 and ry > 1:
                ov = img.copy()
                cv2.ellipse(ov, (u, v), (rx, ry), 0, 0, 360, (14, 16, 24), -1)
                img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)

        for mesh in scene.meshes:
            skin = scene.skins[scene.nodes[mesh.node]["skin"]]
            jm = np.stack([glob[j] for j in skin.joints])
            verts = glblib.skin_mesh(mesh, skin, jm)
            nrm = glblib.pose_normals(mesh, skin, jm)
            tex = scene.textures[scene.materials[mesh.material].base_tex]
            o, _ = render(verts, nrm, mesh.uvs, mesh.indices, tex, cam,
                          bg=(0, 0, 0), depth_cue=False)
            mm = o[:, :, 3] > 0
            img[mm] = o[:, :, :3][mm]

        frame_rgb = downscale(img, ss) if ss > 1 else img
        frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])   # cv2 + VideoWriter are BGR
        times.append(time.time() - t0)

        # --- compose --------------------------------------------------------
        if args.layout == "split":
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start + k)
            ok, src = cap.read()
            src = cv2.resize(src, (panel_w, H)) if ok else np.zeros((H, panel_w, 3), np.uint8)
            right = cv2.resize(frame_bgr, (W - panel_w, H))
            canvas = np.hstack([src, right])
            cv2.line(canvas, (panel_w, 0), (panel_w, H), (96, 104, 130), 2)
            cv2.putText(canvas, "source", (panel_w + 14, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (150, 160, 190), 1, cv2.LINE_AA)
        elif args.layout == "overlay":
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start + k)
            ok, src = cap.read()
            src = cv2.resize(src, (W, H)) if ok else np.zeros((H, W, 3), np.uint8)
            canvas = (0.35 * src.astype(np.float32) + 0.65 * frame_bgr.astype(np.float32)).astype(np.uint8)
        else:
            canvas = np.ascontiguousarray(frame_bgr)

        cv2.putText(canvas, f"MIRACULOUS S6 opening  frame {args.start + k:4d}  "
                            f"({(args.start + k) / 24.0:5.2f}s)",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, "pose -> swing solve -> LBS -> software raster",
                    (18, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 160, 190), 1, cv2.LINE_AA)
        writer.write(canvas)

        if k % 10 == 0 or k == T - 1:
            print(f"  frame {k + 1}/{T}  {times[-1] * 1000:.0f} ms", flush=True)

    writer.release()
    cap.release()
    print(f"wrote {args.out}  ({T} frames, {time.time() - t_start:.1f}s total, "
          f"{1000 * np.mean(times[1:]):.0f} ms/frame avg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
