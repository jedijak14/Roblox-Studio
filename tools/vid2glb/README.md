# vid2glb — drive a rigged GLB with motion from a video

Turn a human performance in a video into animation on an arbitrary rigged glTF
character, then render it and bake it back out as a glTF animation. CPU only: no
Blender, no OpenGL, no GPU. glTF parsing is hand-rolled (`glb.py`), rasterisation is
a ~200-line C scanline rasteriser (`tiny_raster.c`) loaded with `ctypes`.

## Setup

```bash
bash setup_tool.sh          # creates /opt/roblox/venv, installs mediapipe + deps
```

`mediapipe==0.10.14` is pinned: it is the last wheel that ships the pose models
inside the package, so nothing is downloaded at run time.

## The pipeline

```
                    ┌── track_roi.py ──► run_roi.json
video ──► track_pose.py                  (whole clip, full frame)
                    └── scan_pose.py ──► pose_scan.json

run_roi.json ──► build_cycle.py ──► run_cycle.npz ──┬──► make_video.py ──► .mp4
                                                    └──► export_animated_glb.py ──► .glb
```

| Script | Role |
|---|---|
| `track_roi.py` | Track a shot on a **smoothed crop upscaled 2–3×**, mapped back to full-frame coordinates. The main accuracy lever on small subjects. |
| `track_pose.py` | Tracking-mode (`static_image_mode=False`) pass over a frame range. |
| `scan_pose.py` | Independent per-frame pass. Faster, but swaps subjects on cuts — superseded by tracking mode. |
| `build_cycle.py` | **The solver.** One tracked stride → near-side chain → harmonic fit in phase → mirrored far side → loopable cycle + travel speed. |
| `solve_motion.py` | Axis mapping and helpers (`canonical_side`, `zero_phase_lowpass`, `rigid_yaw`, `circular_lowpass`) plus a whole-clip per-frame solve. `build_cycle.py` imports it. |
| `render_bake.py` | Replay a bake: FK → LBS → raster → floor + contact shadow. `--grid` for an inspection sheet, `--out` for a video. |
| `make_video.py` | The deliverable: source and character side by side, full length. |
| `export_animated_glb.py` | Bake node rotation/translation tracks into a real glTF animation (`--bake`, or the older `--scan`/`--start`/`--end`). |
| `apply_video_motion.py` | The original driver. Still the home of `Rig` (the swing solver), the floor, and the compositor. |

## Use

```bash
V=/opt/roblox/venv/bin/python

# 1. track the shot that actually contains the performance
$V track_roi.py --video in.mp4 --start 195 --end 330 \
      --seed docs/motion/tracked_full.json --out docs/motion/run_roi.json

# 2. one stride -> symmetric loop  (--harmonics 2-3, --head-follow 0.2-0.4)
$V build_cycle.py --tracked docs/motion/run_roi.json --glb model.glb \
      --start 234 --end 247 --near l --frames 10 --out docs/motion/run_cycle.npz

# 3. look at it as a sheet before committing to a render
$V render_bake.py --bake docs/motion/run_cycle.npz --glb model.glb --grid /tmp/c.png

# 4. deliverables
$V make_video.py --bake docs/motion/run_cycle.npz --glb model.glb \
      --source in.mp4 --out docs/out.mp4
$V export_animated_glb.py --glb model.glb --bake docs/motion/run_cycle.npz \
      --total-frames 915 --out docs/out.glb
```

## What it recovers, and what it doesn't

**Recovered:** bone directions for the near side of the body — shoulders, elbows,
hips, knees, ankles — plus stride period, stride length, and travel speed (measured
from the solved rig's own stance foot, so the feet do not slide).

**Not recovered:**

- **Twist.** Not observable from sparse landmarks. The solve is swing-only, so limbs
  point correctly but do not roll. This is why the hands stay in a neutral pose.
- **The far side.** Inferred: mirrored from the near side half a stride later. Correct
  for a gait, wrong for anything asymmetric (a wave, a kick).
- **Depth.** MediaPipe's *z* is the noisiest channel. It is clamped and down-weighted
  (`--depth`), never trusted for the main solve.

## Notes from building this

Bugs worth remembering, because each produced motion that looked "haunted" rather
than obviously broken:

1. **Axis mapping.** `retarget.to_canonical` maps image-x → world-X, which is correct
   for a frontal view. For a *profile* shot image-x is the character's forward axis; use
   `canonical_side` instead or the limbs swing sideways.
2. **Mirror plane.** Reflecting through `x → -x` only reverses a gait if the character
   still faces +Z. Yaw the character first and the sagittal plane moves — apply the yaw
   *rigidly, last*, or the mirror folds the body.
3. **`Rig.pose(yaw=...)` is not a body yaw.** It rotates only the root; every bone with
   a direction target then swings back to its un-yawed target, leaving the skeleton
   half-turned.
4. **`ndarray.T` on a 4-D array reverses all axes**, not the first two. Use
   `transpose(1, 0, 2, 3)`.
5. **Smoothing amplitude.** Zero-phase exponential smoothing applied to a signal whose
   period is ~8 samples flattens it badly; doing it twice turns a run into a shuffle.
   Smooth in phase space, where the cycle is a loop and a `[1,2,1]` kernel is harmless.
6. **Phase-binning needs samples.** Binning 12 samples over 1.5 cycles leaves most bins
   empty and interpolation destroys the extremes. Fit harmonics of the stride frequency
   instead.
7. `export_animated_glb` decomposes every node's `matrix` to TRS before writing, because
   glTF viewers ignore animation on nodes that still carry a `matrix`.
