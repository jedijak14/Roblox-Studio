#!/usr/bin/env python
"""
export_animated_glb.py — bake a retargeted motion into a GLB as a real glTF animation.

    /opt/roblox/venv/bin/python export_animated_glb.py \
        --glb scene.glb --video clip.mp4 --scan pose_scan.json \
        --start 238 --end 248 --out scene_animated.glb

Writes `rotation` (quaternion) tracks for every animated bone plus a `translation`
track on the skeleton root, sampled at the video's frame rate. The result opens in
any glTF viewer (three.js, Blender, Babylon) with the motion already on the timeline.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glb as glblib                          # noqa: E402
import retarget as rt                         # noqa: E402
from apply_video_motion import (Rig, landmark_sequence, smooth_landmarks,  # noqa: E402
                                load_scan, root_track, body_yaw, BONE_MAP)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w), glTF order."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_continuity(qs: np.ndarray) -> np.ndarray:
    """Flip quaternions so consecutive samples stay in the same hemisphere."""
    out = qs.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def local_trs(scene: glblib.Scene, idx: int, parent_pose: np.ndarray,
              pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a node's local transform given its global pose and parent's pose."""
    local = np.linalg.inv(parent_pose) @ pose
    t = local[:3, 3]
    R = local[:3, :3]
    s = np.array([np.linalg.norm(R[:, i]) for i in range(3)])
    R = R / np.where(s < 1e-12, 1.0, s)
    return t, mat_to_quat(R), s


def add_accessor(gltf: dict, blob: list[bytes], arr: np.ndarray, comp_type: int,
                 type_str: str, minmax: bool = True) -> int:
    data = np.ascontiguousarray(arr, dtype=np.float32 if comp_type == 5126 else np.uint32)
    raw = data.tobytes()
    while len(blob[0]) % 4:
        blob[0] += b"\x00"
    offset = len(blob[0])
    blob[0] += raw
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset,
                                "byteLength": len(raw), "target": 34962})
    acc = {"bufferView": len(gltf["bufferViews"]) - 1, "componentType": comp_type,
           "count": len(data), "type": type_str}
    if minmax:
        acc["min"] = data.min(axis=0).tolist()
        acc["max"] = data.max(axis=0).tolist()
    gltf["accessors"].append(acc)
    return len(gltf["accessors"]) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--video", required=True, help="unused; kept for CLI symmetry")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--out", default="animated.glb")
    args = ap.parse_args()

    scan = load_scan(args.scan)
    raw, vis = landmark_sequence(scan, args.start, args.end)
    raw = smooth_landmarks(raw, args.smooth)
    T = len(raw)

    container = glblib.GLB(args.glb)
    scene = glblib.load(args.glb)
    rig = Rig(scene)
    ret = rt.Retargeter({})

    canon_seq = np.stack([rt.to_canonical(raw[k], vis[k])[0] for k in range(T)])
    yaws = body_yaw(canon_seq)
    track = root_track(raw)

    # collect global poses per frame
    frames: list[dict[int, np.ndarray]] = []
    for k in range(T):
        dirs = ret.bone_directions(canon_seq[k], vis[k])
        frames.append(rig.pose(dirs, root_offset=track[k], yaw=float(yaws[k])))

    gltf = container.gltf
    blob = [container._bin]

    # glTF 2.0: "When a node is targeted for animation, matrix must not be present."
    # Every node in this file is authored as a matrix, and viewers silently ignore
    # animation channels on such nodes -- so decompose them all to TRS first.
    for node in gltf["nodes"]:
        if "matrix" not in node:
            continue
        M = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        t = M[:3, 3]
        R = M[:3, :3].copy()
        scl = np.array([np.linalg.norm(R[:, i]) for i in range(3)])
        R = R / np.where(scl < 1e-12, 1.0, scl)
        if np.linalg.det(R) < 0:            # left-handed basis -> flip one axis
            scl[0] *= -1
            R[:, 0] *= -1
        node.pop("matrix")
        node["translation"] = [float(x) for x in t]
        node["rotation"] = [float(x) for x in mat_to_quat(R)]
        node["scale"] = [float(x) for x in scl]

    # animated nodes: every joint that actually moves
    joint_nodes = set()
    for skin in scene.skins:
        joint_nodes.update(skin.joints)
    animated = sorted(joint_nodes)

    # node -> parent (in the scene graph)
    parent_of = scene.parents

    def world_pose(idx: int, g: dict[int, np.ndarray]) -> np.ndarray:
        return g.get(idx, np.eye(4))

    rot_tracks: dict[int, np.ndarray] = {i: [] for i in animated}
    trans_track_root: list[np.ndarray] = []
    root = rig.root_bone

    for k in range(T):
        g = frames[k]
        for i in animated:
            par = parent_of.get(i)
            ppose = world_pose(par, g) if par is not None else np.eye(4)
            t, q, s = local_trs(scene, i, ppose, world_pose(i, g))
            rot_tracks[i].append(q)
            if i == root:
                trans_track_root.append(t)

    times = np.arange(T, dtype=np.float32) / args.fps

    channels, samplers = [], []
    time_acc = add_accessor(gltf, blob, times, 5126, "SCALAR")

    for i in animated:
        qs = quat_continuity(np.stack(rot_tracks[i]))
        q_acc = add_accessor(gltf, blob, qs, 5126, "VEC4")
        samplers.append({"input": time_acc, "output": q_acc, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": i, "path": "rotation"}})

    if trans_track_root:
        ts = np.stack(trans_track_root).astype(np.float32)
        t_acc = add_accessor(gltf, blob, ts, 5126, "VEC3")
        samplers.append({"input": time_acc, "output": t_acc, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": root, "path": "translation"}})

    name = f"retarget_frames_{args.start}_{args.end}"
    anim = {"name": name, "channels": channels, "samplers": samplers}
    gltf.setdefault("animations", []).append(anim)

    # ---- write GLB -------------------------------------------------------
    bin_chunk = bytes(blob[0])
    gltf["buffers"] = [{"byteLength": len(bin_chunk)}]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    while len(bin_chunk) % 4:
        bin_chunk += b"\x00"

    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        fh.write(bin_chunk)

    print(f"wrote {args.out} ({total / 1e6:.2f} MB)")
    print(f"  animation '{name}': {len(channels)} channels, {T} samples @ {args.fps} fps "
          f"({T / args.fps:.2f}s)")


if __name__ == "__main__":
    sys.exit(main())
