#!/usr/bin/env python
"""
find_best_segment.py — rank frames/segments of a pose scan by retarget-suitability.

    python find_best_segment.py pose_scan.json [--min-len 8]

Score favours: confident landmarks, visible legs/ankles (i.e. a full-body shot,
not a close-up), a plausible standing silhouette, and clip continuity.
"""
from __future__ import annotations
import argparse, json
import numpy as np

ANKLE = [27, 28]; KNEE = [25, 26]; HIP = [23, 24]


def score_frame(r: dict, W: int, H: int) -> float:
    if not r.get("det"):
        return -1.0
    v = np.array(r["vis"]); xy = np.array(r["xy"])
    good = v > 0.5
    if good.sum() < 8:
        return -1.0
    b = xy[good]
    bw = (b[:, 0].max() - b[:, 0].min()) / W
    bh = (b[:, 1].max() - b[:, 1].min()) / H
    aspect = bh / max(bw, 1e-3)
    legs = min(v[ANKLE].min(), v[KNEE].min(), v[HIP].min())    # full body?
    return float(v.mean() * 0.5 + legs * 1.0 + min(aspect / 2.5, 1.0) * 0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    d = json.load(open(args.scan))
    W, H = d["W"], d["H"]
    scores = [(r["i"], score_frame(r, W, H)) for r in d["frames"]]

    top = sorted([s for s in scores if s[1] > 0], key=lambda x: -x[1])[:10]
    print("best individual frames:")
    for i, s in top:
        print(f"  frame {i:5d}  t={i / d['fps']:6.2f}s  score {s:.2f}")

    runs, cur = [], []
    for i, s in scores:
        if s > args.threshold:
            cur.append(i)
        else:
            if len(cur) >= args.min_len:
                runs.append(cur)
            cur = []
    if len(cur) >= args.min_len:
        runs.append(cur)
    print(f"\ncontiguous segments with score > {args.threshold}:")
    for r in runs:
        print(f"  {r[0]:5d}-{r[-1]:5d}  ({(r[-1] - r[0] + 1) / d['fps']:5.2f}s, {len(r)} frames)")
    if runs:
        best = max(runs, key=len)
        print(f"\nsuggested: --start {best[0]} --end {best[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
