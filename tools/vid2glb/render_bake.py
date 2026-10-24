#!/usr/bin/env python
"""
render_bake.py — draw a baked motion cycle (solve_motion.py output) to PNG or MP4.

The bake stores, per frame, every scene node's LOCAL transform, so replaying it is
pure forward kinematics. Lineage:

    bake npz -> global transforms -> linear blend skinning -> raster (C, ctypes)
             -> floor + contact shadow + supersample downscale -> PNG or MP4

usage:
  render_bake.py --bake run_cycle.npz --glb "scene 6.glb" --grid /tmp/cycle.png
  render_bake.py --bake run_cycle.npz --glb "scene 6.glb" --out cycle.mp4 --loops 4
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glb as glblib                                    # noqa: E402
from raster import Camera, render, downscale            # noqa: E402
from apply_video_motion import floor_mesh, checker_texture, backdrop  # noqa: E402


def local_to_globals(scene, loc: np.ndarray) -> np.ndarray:
    """(nnodes, T, 4, 4) local -> (nnodes, T, 4, 4) global, in scene-node order."""
    G = np.zeros_like(loc)
    for i in range(len(loc)):
        p = scene.parent_of.get(i)
        G[i] = G[p] @ loc[i] if p is not None else loc[i]
    return G


def paint(base, layer):
    """Alpha-from-slot-3 composite of one render pass onto an RGB base."""
    m = layer[:, :, 3] > 0
    base[m] = layer[:, :, :3][m]
    return base


def draw_frame(scene, G, cam, floor, ss, shadow=True) -> np.ndarray:
    """Render one posed frame at supersampled size; returns BGR-ish RGB uint8."""
    fverts, fnorm, fuv, fidx, ftex = floor
    W, H = cam.W, cam.H
    img = backdrop(W, H)

    o, _ = render(fverts, fnorm, fuv, fidx, ftex, cam, bg=(0, 0, 0),
                  depth_cue=False, cull=False)
    img = paint(img, o)

    if shadow:
        # contact shadow under the pelvis: grounds the character visually
        hip = G[scene.hip_node][:3, 3]
        view = cam.view_matrix()
        pc = view @ np.array([hip[0], 0.0, hip[2], 1.0])
        if pc[2] < -0.1:
            zc = -pc[2]
            fx = cam.intrinsics()[0, 0]
            u = int(fx * pc[0] / zc + cam.W / 2)
            v = int(-fx * pc[1] / zc + cam.H / 2)
            rx, ry = max(2, int(0.34 * fx / zc)), max(2, int(0.11 * fx / zc))
            ov = img.copy()
            cv2.ellipse(ov, (u, v), (rx, ry), 0, 0, 360, (14, 16, 24), -1)
            img = cv2.addWeighted(ov, 0.50, img, 0.50, 0)

    for mesh in scene.meshes:
        skin = scene.skins[scene.nodes[mesh.node]["skin"]]
        jm = G[skin.joints]
        verts = glblib.skin_mesh(mesh, skin, jm)
        nrm = glblib.pose_normals(mesh, skin, jm)
        tex = scene.textures[scene.materials[mesh.material].base_tex]
        o, _ = render(verts, nrm, mesh.uvs, mesh.indices, tex, cam,
                      bg=(0, 0, 0), depth_cue=False, cull=True)
        img = paint(img, o)

    if ss > 1:
        img = downscale(img, ss)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--grid", default="")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--pre-roll", type=int, default=0)
    ap.add_argument("--cell", type=int, default=300, help="grid cell width")
    ap.add_argument("--cam-offset", default="-1.00,0.86,3.02",
                    help="camera offset from the character, metres")
    ap.add_argument("--fov", type=float, default=30.0)
    args = ap.parse_args()

    bake = np.load(args.bake, allow_pickle=True)
    scene = glblib.load(args.glb)
    fps = float(bake["fps"])
    N = int(bake["N"])
    loc = bake["local"]                                     # (N, nnodes, 4, 4)
    off = bake["off"]

    if not hasattr(scene, "parent_of") or scene.parent_of is None:
        scene.parent_of = {}
        for i, nd in enumerate(scene.nodes):
            for c in nd.get("children", []):
                scene.parent_of[c] = i
    scene.hip_node = scene.parent_of.get(
        next(j for j, nd in enumerate(scene.nodes)
             if (nd.get("name") or "").startswith("hips")), 0) or 0

    G = local_to_globals(scene, loc.transpose(1, 0, 2, 3))  # (nnodes, N, 4, 4)

    fverts, fnorm, fuv, fidx = floor_mesh(size=40.0, y=0.0, squares=80, cells=8)
    floor = (fverts, fnorm, fuv, fidx, checker_texture(512, squares=80))

    offv = np.array([float(x) for x in args.cam_offset.split(",")])
    fwd_x = off[:, 0]

    def camera_for(k: int, w=None, h=None) -> Camera:
        x = float(fwd_x[k])
        return Camera(eye=(x + offv[0], offv[1], offv[2]),
                      target=(x, 0.72, 0.0), fov_deg=args.fov,
                      width=w or args.width, height=h or args.height)

    if args.grid:
        cw = args.cell
        ch = int(cw * 9 / 16)
        tiles = []
        for k in range(N):
            cam = camera_for(k, cw, ch)
            img = np.asarray(draw_frame(scene, G[:, k], cam, floor, 1), np.uint8)
            cv2.putText(img, str(k), (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)
            tiles.append(img[:, :, ::-1])
        cols = min(5, len(tiles))
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(args.grid, np.vstack(rows))
        print(f"wrote {args.grid} ({N} frames)")

    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.out, fourcc, fps, (args.width, args.height))
        total = args.pre_roll + N * args.loops
        for f in range(total):
            k = 0 if f < args.pre_roll else ((f - args.pre_roll) % N)
            cam = camera_for(k)
            img = draw_frame(scene, G[:, k], cam, floor, args.ssaa)
            out.write(np.ascontiguousarray(np.asarray(img, np.uint8)[:, :, ::-1]))
            if (f + 1) % 20 == 0:
                print(f"  {f + 1}/{total}", flush=True)
        out.release()
        print(f"wrote {args.out}: {total} frames @ {fps:g}fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
