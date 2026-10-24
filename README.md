# Roblox-Studio

Asset experiments: motion capture from video, retargeted onto rigged glTF characters.

## What's in here

| Path | What it is |
|---|---|
| `scene 6.glb` | Marinette (Miraculous) character — 25k tris, 134-bone rig, 4 textures, **no animations** |
| `MIRACULOUS … 720p24.mp4` | Season 6 opening (fan-made, old SAMG style), 1280×720, 24 fps, 38 s |
| `tools/vid2glb/` | The pipeline that turns video motion into GLB animation |
| `docs/scene6_animated.glb` | **Result** — the character with the video's run baked in as a glTF animation |
| `docs/scene6_retargeted.mp4` | **Result** — 720p24 side-by-side screen recording |
| `docs/pose_scan.json` | Per-frame pose landmarks for the whole video (reusable) |

## The result

`docs/scene6_retargeted.mp4` — source footage on the left, retargeted character on the right:

![preview](docs/preview.png)

`docs/scene6_animated.glb` carries the motion as a standard glTF animation
(`retarget_frames_238_248`, 135 channels, 134 joints), so it plays in three.js,
Babylon, Blender, or any glTF viewer without this toolchain.

## How it works

```
video ─► MediaPipe BlazePose ─► landmarks ─► bone directions ─► swing solve
                                                                    │
              screen recording ◄─ software rasterizer ◄─ skinning ◄─┘
                                  animated GLB ◄─ glTF animation tracks
```

No GPU, no OpenGL, no Blender. Rendering is a small C rasterizer loaded via
`ctypes`; glTF reading/writing is `struct` + `numpy`. See
[`tools/vid2glb/README.md`](tools/vid2glb/README.md) for the full write-up,
including the four non-obvious bugs that had to be fixed.

## Reproduce

```bash
bash tools/vid2glb/setup_tool.sh
V=/opt/roblox/venv/bin/python

$V tools/vid2glb/find_best_segment.py docs/pose_scan.json
$V tools/vid2glb/export_animated_glb.py --glb "scene 6.glb" --scan docs/pose_scan.json \
      --start 238 --end 248 --out docs/scene6_animated.glb
$V tools/vid2glb/apply_video_motion.py --glb "scene 6.glb" \
      --video "MIRACULOUS ｜ 🐞 SEASON 6 - OPENING 🐾 ｜ IN OLD SAMG STYLE [FANMADE]-(720p24).mp4" \
      --scan docs/pose_scan.json --start 238 --end 248 \
      --out docs/scene6_retargeted.mp4 --layout split --ssaa 3
```

## Why frames 238–248

The opening is a montage: close-ups, logo cards, rapid cuts. A full scan found only
446/915 frames with a detectable pose, and `find_best_segment.py` ranked segments by
confidence, full-body coverage and silhouette aspect. 238–248 is the only clean run
of the **same character as the GLB**, in full body, with real locomotion (she runs
across the frame). After frame ~249 she holds still.

## Limits

- Bone **twist** is not recovered (not observable from sparse landmarks) — limbs point
  the right way but don't roll.
- **Head** is deliberately left at bind: BlazePose has no reliable head orientation and
  this rig's head bone points +Z, so an "up" target flips the skull backwards.
- **Legs cross** occasionally, because BlazePose confuses left/right in profile views.
- Motion is baked at 24 fps for a single 0.46 s segment. Longer clips work; the
  bottleneck is pose inference (~170 ms/frame) and the 1.3 s/frame render at 3× SSAA.
