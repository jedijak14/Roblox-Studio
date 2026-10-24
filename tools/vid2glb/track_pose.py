#!/usr/bin/env python
"""
track_pose.py — extract pose landmarks for a WHOLE video using MediaPipe's tracking mode.

The important difference from a per-frame scan: `static_image_mode=False` makes
MediaPipe carry the previous frame's landmarks forward, so it follows one subject
instead of re-detecting (and re-choosing) a person every frame. That is what stops
the landmark identities from swapping between characters in a group shot.

Writes one record per frame: image-space xy, visibility, and hip-centred world xyz.
Frames with no detection are recorded with det=false so gaps stay explicit.
"""
from __future__ import annotations
import argparse, json, sys
import cv2, mediapipe as mp, numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--complexity", type=int, default=1)
    ap.add_argument("--min-visibility", type=float, default=0.25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to the end")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end or n
    end = min(end, n)

    pose = mp.solutions.pose.Pose(static_image_mode=False,          # <- tracking, not per-frame
                                  model_complexity=args.complexity,
                                  min_detection_confidence=args.min_visibility)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frames = []
    for i in range(args.start, end):
        ok, fr = cap.read()
        if not ok:
            break
        r = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        d = {"i": i, "det": bool(r.pose_landmarks)}
        if r.pose_landmarks:
            d["xy"] = np.array([[l.x * W, l.y * H] for l in r.pose_landmarks.landmark]).round(1).tolist()
            d["vis"] = np.array([l.visibility for l in r.pose_landmarks.landmark]).round(3).tolist()
            d["world"] = (np.array([[l.x, l.y, l.z] for l in r.pose_world_landmarks.landmark]).round(4).tolist()
                          if r.pose_world_landmarks else None)
        frames.append(d)
        if i % 100 == 0:
            print(f"  {i}/{end}", flush=True)
    pose.close()

    with open(args.out, "w") as fh:
        json.dump({"fps": fps, "W": W, "H": H, "start": args.start, "frames": frames}, fh)
    print(f"wrote {args.out}: {sum(f['det'] for f in frames)}/{len(frames)} frames with a pose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
