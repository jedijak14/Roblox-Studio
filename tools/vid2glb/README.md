# vid2glb — drive a rigged GLB with motion from a video

Extracts human pose from a video and retargets it onto an arbitrary rigged GLB,
then bakes the result into a real glTF animation.

```
video ──► MediaPipe BlazePose ──► landmarks
                                    │
                     canonical space (Y-up, metres, facing +Z)
                                    │
                            bone directions + body yaw
                                    │
                    swing-only solve onto the rig's bind pose
                                    │
                  forward kinematics ──► linear blend skinning
                                    │
              ┌─────────────────────┴─────────────────────┐
     software rasterizer → mp4 render          glTF animation tracks → .glb
```

No GPU, no OpenGL, no Blender, no `libGL` — the rasterizer is a ~250-line C file
loaded through `ctypes` (`tiny_raster.c`), and the GLB reader is plain `struct` +
`numpy` (`glb.py`).

## Setup

```bash
bash setup_tool.sh          # creates /opt/roblox/venv (mediapipe + opencv-headless)
```

`mediapipe` needs `protobuf<5`, which conflicts with the 4D-Humans virtualenv, and
it drags in `opencv-contrib-python` (links against `libGL`, absent here). Hence a
dedicated venv and `--no-deps` + a pinned headless OpenCV.

## Use

```bash
V=/opt/roblox/venv/bin/python

# 1. scan the video for poses
$V scan_pose.py --video clip.mp4 --out pose_scan.json

# 2. pick the most retargetable segment (full-body, no cuts)
$V find_best_segment.py pose_scan.json

# 3. bake the motion into a new GLB (opens in any glTF viewer)
$V export_animated_glb.py --glb scene.glb --scan pose_scan.json \
     --start 238 --end 248 --out scene_animated.glb

# 4. render a side-by-side screen recording
$V apply_video_motion.py --glb scene.glb --video clip.mp4 --scan pose_scan.json \
     --start 238 --end 248 --out demo.mp4 --layout split --ssaa 3
```

`--layout model|split|overlay`, `--width/--height`, `--ssaa N` (supersampling),
`--yaw 0..1` (weight of body-yaw tracking, 0 disables it).

## What it does and does not recover

| Recovered | Not recovered |
|---|---|
| limb swing (arms, legs, spine), from 3D landmarks | bone twist — not observable from sparse landmarks; left at bind |
| pelvis translation → locomotion, camera follows | finger/hand pose beyond the mitten |
| body yaw from the shoulder line | head orientation (see note below) |
| foot-ground contact (skeleton is re-grounded per frame) | facial expression, prop interaction |

**Head:** deliberately not retargeted. BlazePose gives no reliable head orientation,
and this rig's head bone points **+Z** (towards the cheek) while the available
landmark vectors are mostly **+Y**; feeding an "up" vector rotates the skull ~45°
backwards. Left at bind, the head inherits the spine tilt and stays stable. Set
`"head_Armature": "head"` in `BONE_MAP` if you have real head-pose data.

## Notes from building this

- **One direction target per bone chain.** If both `spine` and `torso` consume the
  `spine` target, the child re-applies the parent's swing and the torso folds double.
- **`torso`'s bind axis is horizontal** (straight up the spine is `spine`'s job), so
  it must not receive the spine direction either.
- **glTF forbids `matrix` on animated nodes.** This asset authors every node as a
  matrix; viewers silently ignore animation channels on those nodes, so
  `export_animated_glb.py` decomposes all matrices to TRS first.
- **Winding differs per asset.** This mesh is CW-from-`-Z`;
  `raster.render(..., cull=True)` assumes CCW. The floor is drawn with `cull=False`.
- **Big triangles must be subdivided.** The rasterizer drops any triangle with a
  vertex behind the camera, so a single floor quad vanishes at the near plane.
- **`cv2.VideoWriter` wants BGR.** Feeding it RGB renders the character blue.
