#!/usr/bin/env python
"""
track_roi.py — re-track a shot on an upscaled crop.

MediaPipe's pose model is trained around a ~256-400 px person. In this opening the
character is ~180 px tall, so wrists/knees/ankles fall below the visibility
threshold and the rig gets dead limbs. Cropping around the subject and upscaling
by 2-3x puts the person in the model's comfort zone without changing the shot.

The crop window is driven by a *smoothed* bbox trajectory so the crop itself never
jitters, and landmarks are mapped back through the exact affine, so output is in
full-frame pixel coordinates, same schema as track_pose.py.

usage:
  track_roi.py --video X.mp4 --start 200 --end 330 --out docs/motion/run.json \
               [--seed docs/motion/tracked_full.json] [--zoom 2.6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def zero_phase_lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    y = x.copy()
    for i in range(1, len(y)):
        y[i] = alpha * y[i] + (1 - alpha) * y[i - 1]
    for i in range(len(y) - 2, -1, -1):
        y[i] = alpha * y[i] + (1 - alpha) * y[i + 1]
    return y


def seed_crops(seed_path: str, start: int, end: int, W: int, H: int,
               zoom: float) -> np.ndarray:
    """Smoothed crop rectangles (cx, cy, side) for frames [start, end]."""
    n = end - start + 1
    boxes = np.full((n, 4), np.nan)          # x0, y0, x1, y1 in full-frame px
    if seed_path and os.path.exists(seed_path):
        d = json.load(open(seed_path))
        for f in d["frames"]:
            i = f["i"]
            if not (start <= i <= end) or not f.get("det"):
                continue
            xy = np.asarray(f["xy"], float)
            vis = np.asarray(f["vis"], float)
            m = (vis > 0.3) & np.isfinite(xy).all(1)
            if m.sum() < 4:
                continue
            pts = xy[m]
            # crop side = body extent * slack, never under 240 px source
            bw = pts[:, 0].ptp(); bh = pts[:, 1].ptp()
            side = max(240.0, 1.9 * max(bw, bh))
            boxes[i - start] = [pts[:, 0].mean() - side / 2, pts[:, 1].mean() - side / 2,
                                pts[:, 0].mean() + side / 2, pts[:, 1].mean() + side / 2]

    idx = np.arange(n)
    out = np.zeros((n, 3))
    for c in range(4):
        good = np.isfinite(boxes[:, c])
        if good.sum() == 0:
            boxes[:, c] = [W / 2 - 200, H / 2 - 200, W / 2 + 200, H / 2 + 200][c]
        else:
            boxes[:, c] = np.interp(idx, idx[good], boxes[good, c])

    # convert to centre + side, smooth, then clamp inside the frame
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    side = np.maximum(boxes[:, 2] - boxes[:, 0], 200.0)
    cx = zero_phase_lowpass(cx, 0.25)
    cy = zero_phase_lowpass(cy, 0.25)
    side = zero_phase_lowpass(side, 0.15)
    for k in range(n):
        s = side[k] / 2
        cx[k] = float(np.clip(cx[k], s, W - s)) if s * 2 < W else W / 2
        cy[k] = float(np.clip(cy[k], s, H - s)) if s * 2 < H else H / 2
    out[:, 0] = cx; out[:, 1] = cy; out[:, 2] = side
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--seed", default="")
    ap.add_argument("--zoom", type=float, default=2.6,
                    help="target upscale of the crop (2-3 works well)")
    ap.add_argument("--cx", type=float, default=1, help="model_complexity")
    args = ap.parse_args()

    import mediapipe as mp

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    crops = seed_crops(args.seed, args.start, args.end, W, H, args.zoom)
    print(f"frame {W}x{H} @ {fps:g}fps; crops side "
          f"{np.median(crops[:, 2]):.0f}px -> x{np.median(crops[:, 2]) and (512/np.median(crops[:, 2])):.2f}")

    pose = mp.solutions.pose.Pose(static_image_mode=False,
                                  model_complexity=args.cx,
                                  smooth_landmarks=True,
                                  min_detection_confidence=0.35,
                                  min_tracking_confidence=0.35)

    frames = []
    n_det = 0
    for k, i in enumerate(range(args.start, args.end + 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            frames.append(dict(i=i, det=False, xy=[], vis=[], world=[]))
            continue
        cx, cy, side = crops[k]
        s = side / 2
        x0, y0 = int(round(cx - s)), int(round(cy - s))
        x1, y1 = int(round(cx + s)), int(round(cy + s))
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            frames.append(dict(i=i, det=False, xy=[], vis=[], world=[]))
            continue
        scale = 512.0 / max(crop.shape[1], crop.shape[0])
        big = cv2.resize(crop, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        res = pose.process(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            frames.append(dict(i=i, det=False, xy=[], vis=[], world=[]))
            continue
        n_det += 1
        src = res.pose_landmarks.landmark
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks.landmark
            world = [[float(p.x), float(p.y), float(p.z)] for p in wl]
        else:
            world = [[0.0, 0.0, 0.0] for _ in src]
        xy = [[float(p.x) * big.shape[1] / scale + x0,
               float(p.y) * big.shape[0] / scale + y0] for p in src]
        vis = [float(p.visibility) for p in src]
        frames.append(dict(i=i, det=True, xy=xy, vis=vis, world=world))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(crops)}", flush=True)
    pose.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(dict(fps=fps, W=W, H=H, start=args.start, zoom=args.zoom,
                   frames=frames), open(args.out, "w"))
    print(f"wrote {args.out}: {n_det}/{len(frames)} frames with a pose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
