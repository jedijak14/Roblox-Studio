#!/usr/bin/env bash
# Build the retargeting toolchain in its own venv.
#
# mediapipe needs protobuf<5, which is incompatible with the 4D-Humans env, so this
# lives in a dedicated venv. It also must not install opencv-contrib-python (which
# links against libGL, absent here) -- hence --no-deps and a pinned headless cv2.
set -euo pipefail
VENV=${VENV:-/opt/roblox/venv}
PYVER=3.11

log() { printf '\n=== %s\n' "$*"; }

if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV"
  sudo mkdir -p "$(dirname "$VENV")"
  sudo chown "$(id -u):$(id -g)" "$(dirname "$VENV")"
  python3 -m venv "$VENV"
fi
PIP="$VENV/bin/pip"
"$PIP" -q install --upgrade pip wheel setuptools

log "core deps (numpy 1.x: mediapipe + trimesh pin it)"
"$PIP" -q install "numpy==1.26.4" "opencv-python-headless==4.11.0.86" \
                   "protobuf==4.25.3" absl-py attrs flatbuffers scipy pillow

log "mediapipe 0.10.14 (last release that BUNDLES the pose .tflite models)"
if ! "$VENV/bin/python" - <<'PY' 2>/dev/null
import mediapipe, os
p = os.path.dirname(mediapipe.__file__)
models = [f for r, d, fs in os.walk(p) for f in fs if f.endswith(".tflite")]
raise SystemExit(0 if models else 1)
PY
then
  "$PIP" download --no-deps -d /tmp/mpd "mediapipe==0.10.14"
  "$PIP" install --no-deps /tmp/mpd/mediapipe-0.10.14-*.whl
fi

log "building the C rasterizer"
gcc -O3 -fPIC -shared "$(dirname "$0")/tiny_raster.c" -o /tmp/libtiny_raster.so -lm && echo "  ok"

log "smoke test"
"$VENV/bin/python" - <<'PY'
import cv2, mediapipe as mp, numpy as np
print("  mediapipe", mp.__version__, "| cv2", cv2.__version__, "| numpy", np.__version__)
mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
print("  pose model loads offline")
PY
