#!/usr/bin/env python
"""
scan_pose.py — run MediaPipe BlazePose over every frame of a video.

    python scan_pose.py --video clip.mp4 --out pose_scan.json [--stride 1]

Writes both image-space landmarks (xy, for continuity checks) and MediaPipe's
world landmarks (metres, hip-centred). The world landmarks are what the
retargeter consumes; the image-space ones are useful for finding the best
segment (full-body shots, no cuts).

Requires mediapipe, which needs protobuf<5 and pulls in opencv-contrib-python --
hence the separate venv (see README).
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import mediapipe as mp
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="pose_scan.json")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--complexity", type=int, default=1,
                    help="0=lite, 1=full (bundled), 2=heavy (downloads, may be blocked)")
    ap.add_argument("--min-visibility", type=float, default=0.25,
                    help="detection confidence floor")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=args.complexity,
                        min_detection_confidence=args.min_visibility)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames:
        n = min(n, args.max_frames)

    frames = []
    i = 0
    while i < n:
        ok, fr = cap.read()
        if not ok:
            break
        if i % args.stride:
            i += 1
            continue
        res = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        rec = {"i": i, "det": False}
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            xy = np.array([[l.x * W, l.y * H] for l in lm])
            vis = np.array([l.visibility for l in lm])
            wl = res.pose_world_landmarks
            rec = {
                "i": i, "det": True,
                "vis_mean": float(vis.mean()),
                "xy": xy.round(1).tolist(),
                "vis": vis.round(3).tolist(),
                "world": (np.array([[l.x, l.y, l.z] for l in wl.landmark]).round(4).tolist()
                          if wl else None),
            }
            good = vis > 0.5
            if good.sum() > 4:
                b = xy[good]
                rec["extent"] = float(max((b[:, 0].max() - b[:, 0].min()) / W,
                                          (b[:, 1].max() - b[:, 1].min()) / H))
        frames.append(rec)
        if i % 60 == 0:
            print(f"  {i}/{n}", flush=True)
        i += 1

    with open(args.out, "w") as fh:
        json.dump({"n": n, "fps": fps, "W": W, "H": H, "frames": frames}, fh)

    det = sum(1 for r in frames if r["det"])
    print(f"wrote {args.out}: {det}/{len(frames)} frames with a pose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
