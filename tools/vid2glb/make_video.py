#!/usr/bin/env python
"""
make_video.py — the deliverable: full-length split-screen retarget video.

Left panel: the source video, every frame, unmodified.
Right panel: the rigged GLB driven by the cycle baked from that video.

The character's cycle is looped across the whole running time while the source plays
at its own speed, so the output is exactly as long as the input (38.12 s, 915 frames
at 24 fps) rather than an 11-frame slice.

usage:
  make_video.py --bake docs/motion/run_cycle.npz --glb "scene 6.glb" \
                --source "opening.mp4" --out docs/scene6_retargeted.mp4 \
                --width 1280 --height 720 --ssaa 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glb as glblib                                        # noqa: E402
from raster import Camera                                    # noqa: E402
from apply_video_motion import floor_mesh, checker_texture    # noqa: E402
import render_bake as RB                                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", required=True)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--layout", choices=["split", "model"], default="split")
    ap.add_argument("--cam-offset", default="0.0,0.95,2.95")
    ap.add_argument("--fov", type=float, default=30.0)
    ap.add_argument("--frames", type=int, default=0, help="0 = whole source clip")
    ap.add_argument("--start-frame", type=int, default=0)
    args = ap.parse_args()

    bake = np.load(args.bake, allow_pickle=True)
    scene = glblib.load(args.glb)
    fps = float(bake["fps"])
    N = int(bake["N"])
    loc = bake["local"]
    off = bake["off"]

    scene.parent_of = {}
    for i, nd in enumerate(scene.nodes):
        for c in nd.get("children", []):
            scene.parent_of[c] = i
    scene.hip_node = next(j for j, nd in enumerate(scene.nodes)
                          if (nd.get("name") or "").startswith("hips"))
    G = RB.local_to_globals(scene, loc.transpose(1, 0, 2, 3))

    fv, fn, fu, fi = floor_mesh(size=60.0, y=0.0, squares=120, cells=10)
    floor = (fv, fn, fu, fi, checker_texture(512, squares=120))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = args.frames or (src_n - args.start_frame)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    W, H = args.width, args.height
    panel_w = W // 2 if args.layout == "split" else W
    offv = np.array([float(x) for x in args.cam_offset.split(",")])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, src_fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open")

    t0 = time.time()
    for f in range(total):
        src_i = args.start_frame + f
        k = f % N
        x = float(off[k, 0])
        cam = Camera(eye=(x + offv[0], offv[1], offv[2]),
                     target=(x, 0.74, 0.0), fov_deg=args.fov,
                     width=panel_w * args.ssaa, height=H * args.ssaa)
        model = np.asarray(RB.draw_frame(scene, G[:, k], cam, floor, args.ssaa),
                           np.uint8)[:, :, ::-1]

        if args.layout == "split":
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_i)
            ok, src = cap.read()
            if not ok:
                src = np.zeros((H, W - panel_w, 3), np.uint8)
            else:
                src = cv2.resize(src, (W - panel_w, H))
            canvas = np.hstack([src, model])
            cv2.line(canvas, (W - panel_w, 0), (W - panel_w, H), (150, 158, 182), 2)
            cv2.putText(canvas, "source", (W - panel_w + 14, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (196, 202, 220), 1, cv2.LINE_AA)
            label = f"retargeted  {f + 1}/{total}   frame {src_i}"
            cv2.putText(canvas, label, (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (196, 202, 220), 1, cv2.LINE_AA)
        else:
            canvas = model

        writer.write(np.ascontiguousarray(canvas))
        if (f + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {f + 1}/{total}  {el / (f + 1):.2f}s/frame  "
                  f"eta {el / (f + 1) * (total - f - 1) / 60:.1f} min", flush=True)

    writer.release()
    print(f"wrote {args.out}: {total} frames @ {src_fps:g}fps "
          f"({total / src_fps:.2f}s), {time.time() - t0:.0f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
