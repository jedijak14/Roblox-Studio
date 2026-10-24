#!/usr/bin/env python
"""
solve_motion.py — tracked 2D landmarks -> a clean, loopable skeletal performance.

Three things made the first attempt look wrong, and each has a fix here:

1. **Axis mapping.** retarget.to_canonical assumes a frontal view (image-x ->
   world-X). The only usable full-body shot in this opening is a *profile* run, so
   image-x is the character's FORWARD axis. `canonical_side()` re-maps
   (image-x, up, camera-depth) -> (forward Z, up Y, lateral X).

2. **Depth is not trustworthy.** MediaPipe's z is the noisiest channel and half the
   limbs are self-occluded in profile. Instead of feeding it to the solver, its
   contribution is clamped to +-1 unit and down-weighted (`--depth`), then the
   resulting directions are smoothed. Depth only has to break left/right ties.

3. **One stride is not a cycle.** The shot gives ~13 frames (about one stride).
   A run is antisymmetric, so the other stride is the mirror image: the cycle is
   built as [a..b] + mirror([a+1..b]), which is seamless by construction once
   (a, b) is chosen so that pose(a) ~= mirror(pose(b)).

The root travels at the speed that the solved *rig's own* stance foot gives, so the
feet do not slide.

usage:
  solve_motion.py --tracked docs/motion/run_roi.json --glb "scene 6.glb" \
                  --start 234 --end 247 --out docs/motion/run_cycle.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glb as glblib                     # noqa: E402
from retarget import swing_matrix, _unit  # noqa: E402
from apply_video_motion import Rig        # noqa: E402

# BlazePose landmarks
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_IX, R_IX = 19, 20
L_HI, R_HI = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
L_FT, R_FT = 31, 32

# direction key -> (parent landmark, child landmark).  "L" = MediaPipe's own side
# labelling, remapped to rig side by SIDE.
SEGS_SIDE = {
    "arm": (L_SH, L_EL), "elbow": (L_EL, L_WR), "hand": (L_WR, L_IX),
    "thigh": (L_HI, L_KN), "knee": (L_KN, L_AN), "foot": (L_AN, L_FT),
}

MIN_VIS = 0.40


# ------------------------------------------------------------------ helpers
def zero_phase_lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    """Forward-backward exponential smoothing (no phase shift)."""
    if alpha >= 1.0 or len(x) < 2:
        return np.asarray(x, float).copy()
    y = np.asarray(x, float).copy()
    for i in range(1, len(y)):
        y[i] = alpha * y[i] + (1 - alpha) * y[i - 1]
    for i in range(len(y) - 2, -1, -1):
        y[i] = alpha * y[i] + (1 - alpha) * y[i + 1]
    return y


def circular_lowpass(x: np.ndarray, passes: int = 2) -> np.ndarray:
    """Smooth a periodic (T, ...) sequence with a [1,2,1]/4 kernel, wrapped."""
    y = np.asarray(x, float).copy()
    T = len(y)
    for _ in range(passes):
        z = 0.25 * np.roll(y, 1, axis=0) + 0.5 * y + 0.25 * np.roll(y, -1, axis=0)
        y = z
    return y


def fill_time(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Interpolate invalid rows over time; hold at the ends."""
    T, D = x.shape
    idx = np.arange(T)
    out = np.asarray(x, float).copy()
    for d in range(D):
        good = ok & np.isfinite(out[:, d])
        if good.sum() == 0:
            out[:, d] = 0.0
        elif good.sum() == 1:
            out[:, d] = out[good, d][0]
        else:
            out[:, d] = np.interp(idx, idx[good], out[good, d])
    return out


def canonical_side(xy: np.ndarray, vis: np.ndarray, depth: np.ndarray,
                   shoulder_px: float) -> np.ndarray:
    """Profile-view landmarks -> character space (X lateral, Y up, Z forward).

    xy    (33,2) pixels, y down
    depth (33,)  MediaPipe relative z (arbitrary units, negative = toward camera)
    """
    scale = 0.36 / max(shoulder_px, 1.0)          # metres per pixel
    out = np.empty((len(xy), 3))
    # image +x is the direction the subject faces in this shot -> character +Z
    out[:, 0] = -np.asarray(depth, float) * shoulder_px * 0.55   # lateral, clamped below
    out[:, 1] = -xy[:, 1] * scale
    out[:, 2] = xy[:, 0] * scale
    # clamp the depth term so a bad z can never fold the body through itself
    lim = 0.30
    out[:, 0] = np.clip(out[:, 0], -lim, lim)
    hip = 0.5 * (out[L_HI] + out[R_HI])
    out -= hip
    out[:, 1] += 0.90                              # put the hips near a working height
    return out


def bone_targets(P: np.ndarray, depth_w: float) -> dict[str, np.ndarray]:
    """Canonical joint cloud -> unit bone directions for the rig."""
    d = {}
    d["spine"] = _unit(P[L_SH] + P[R_SH] - P[L_HI] - P[R_HI])
    for side, (sh, el, wr, ix, hi, kn, an, ft) in {
        "l": (L_SH, L_EL, L_WR, L_IX, L_HI, L_KN, L_AN, L_FT),
        "r": (R_SH, R_EL, R_WR, R_IX, R_HI, R_KN, R_AN, R_FT),
    }.items():
        d[f"arm_{side}"] = _unit(P[el] - P[sh])
        d[f"elbow_{side}"] = _unit(P[wr] - P[el])
        d[f"hand_{side}"] = _unit(P[ix] - P[wr])
        d[f"thigh_{side}"] = _unit(P[kn] - P[hi])
        d[f"knee_{side}"] = _unit(P[an] - P[kn])
        d[f"foot_{side}"] = _unit(P[ft] - P[an])
        for k in ("arm", "elbow", "hand", "thigh", "knee", "foot"):
            key = f"{k}_{side}"
            v = d[key].copy()
            v[0] *= depth_w                     # trust lateral (depth-derived) least
            d[key] = _unit(v)
    return d


LIMB_KEYS = ["arm_l", "arm_r", "elbow_l", "elbow_r", "hand_l", "hand_r",
             "thigh_l", "thigh_r", "knee_l", "knee_r", "foot_l", "foot_r"]


def mirror_name(name: str) -> str:
    """Swap the side in a bone node name; returns the name unchanged if mid-line."""
    for a, b in (("_l_", "_r_"), ("_L_", "_R_"), (".l", ".r"), ("_left", "_right"),
                 ("_Left", "_Right"), ("Left", "Right"), ("left", "right")):
        if a in name:
            return name.replace(a, b)
        if b in name:
            return name.replace(b, a)
    return name


def mirror_globals(G: dict[int, np.ndarray], pair: dict[int, int]) -> dict[int, np.ndarray]:
    """Reflect a solved skeleton through the sagittal plane (x -> -x)."""
    S = np.diag([-1.0, 1.0, 1.0, 1.0])
    out = {}
    for i, M in G.items():
        j = pair.get(i, i)
        out[i] = S @ G[j] @ S
    return out


def rigid_yaw(G: dict[int, np.ndarray], pivot: np.ndarray, ang: float) -> dict[int, np.ndarray]:
    """Rotate a whole solved skeleton about the vertical axis.

    Rig.pose()'s own `yaw` argument only turns the root, and any bone that has a
    direction target then swings itself back to the un-yawed target -- so the body
    ends up half-turned. Rotating the finished global matrices is what actually
    works.
    """
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    out = {}
    for i, M in G.items():
        Y = np.eye(4)
        Y[:3, 3] = -pivot
        Y2 = np.eye(4)
        Y2[:3, 3] = pivot
        out[i] = Y2 @ R @ Y @ M
    return out


def mat_angle(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, radians."""
    R = A[:3, :3].T @ B[:3, :3]
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracked", required=True)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--depth", type=float, default=0.25)
    ap.add_argument("--smooth", type=float, default=0.45)
    ap.add_argument("--yaw", type=float, default=90.0)
    ap.add_argument("--min-sep", type=int, default=3)
    ap.add_argument("--max-sep", type=int, default=9)
    args = ap.parse_args()

    data = json.load(open(args.tracked))
    W, H, fps = data["W"], data["H"], data["fps"]
    by_i = {f["i"]: f for f in data["frames"]}
    idx = list(range(args.start, args.end + 1))
    T = len(idx)

    scene = glblib.load(args.glb)
    rig = Rig(scene)

    # ---- canonical cloud per frame --------------------------------------
    Pt = np.zeros((T, 33, 3))
    vis = np.zeros((T, 33))
    raw2d = np.full((T, 33, 2), np.nan)
    for k, i in enumerate(idx):
        f = by_i[i]
        if not f["det"]:
            continue
        xy = np.asarray(f["xy"], float)
        v = np.asarray(f["vis"], float)
        dep = np.asarray(f.get("world") or np.zeros((33, 3)), float)
        if dep.shape != (33, 3):
            dep = np.zeros((33, 3))
        sw = np.linalg.norm(xy[L_SH] - xy[R_SH])
        if not np.isfinite(sw) or sw < 5:
            sw = 90.0
        raw2d[k] = xy
        vis[k] = v
        Pt[k] = canonical_side(xy, v, dep[:, 2] / max(sw, 1e-6), sw)

    # gap-fill whole-body motion where the detector dropped out
    det = (vis[:, L_SH] > 0.1) & (vis[:, R_SH] > 0.1)
    for c in range(3):
        Pt[:, :, c] = fill_time(Pt[:, :, c], det)

    # per-joint temporal smoothing before solving (2D noise lives here)
    for j in range(33):
        Pt[:, j] = zero_phase_lowpass(Pt[:, j], args.smooth)

    # ---- solve globals ---------------------------------------------------
    # Solve in the character's OWN frame (facing +Z) so that the sagittal plane is
    # the YZ plane and a mirror is x -> -x. The yaw is applied rigidly at the end.
    G = [rig.pose(bone_targets(Pt[k], args.depth), root_offset=np.zeros(3))
         for k in range(T)]

    # ---- pick the loop window: pose(a) ~= mirror(pose(b)) ----------------
    pair = {i: rig.name_to_idx.get(mirror_name(n.get("name") or ""), i)
            for i, n in enumerate(scene.nodes)}
    key_nodes = {v: k for k, v in rig.bone_nodes.items()}
    probe = [key_nodes[k] for k in LIMB_KEYS if k in key_nodes]

    best = None
    for a in range(0, max(1, T - args.min_sep)):
        for b in range(a + args.min_sep, min(T - 1, a + args.max_sep) + 1):
            Gm = mirror_globals(G[b], pair)
            cost = float(np.mean([mat_angle(G[a][n], Gm[n]) for n in probe]))
            cost -= 0.004 * (b - a)          # prefer a real stride over a twitch
            if best is None or cost < best[0]:
                best = (cost, a, b)
    if best is None:
        print("no loop window found"); return 1
    cost, a, b = best
    print(f"loop window [{idx[a]}..{idx[b]}]  ({b - a + 1} frames, "
          f"mirror mismatch {np.degrees(cost):.1f} deg)")

    # ---- assemble the cycle: [a..b] + mirror([a+1..b]) -------------------
    first = list(range(a, b + 1))
    second = list(range(a + 1, b + 1))
    seq = [(i, False) for i in first] + [(i, True) for i in second]
    N = len(seq)
    print(f"cycle = {N} frames ({N / fps:.2f}s): "
          f"{len(first)} real + {len(second)} mirrored")

    cyc = []
    for i, mir in seq:
        Gk = mirror_globals(G[i], pair) if mir else G[i]
        cyc.append({n: M.copy() for n, M in Gk.items()})

    # canonical frame: Y up, character faces +Z
    FWD, UP = 2, 1

    # ---- measure the rig's own stance-foot speed (no foot sliding) -------
    foot_nodes = [rig.name_to_idx[n] for n in ("foot_l_Armature", "foot_r_Armature")
                  if n in rig.name_to_idx]
    footY = np.array([[Gk[n][UP, 3] for n in foot_nodes] for Gk in cyc])
    footF = np.array([[Gk[n][FWD, 3] for n in foot_nodes] for Gk in cyc])
    support = np.argmin(footY, axis=1)
    vel = []
    for k in range(N):
        k2 = (k + 1) % N
        if support[k] == support[k2]:
            vel.append(footF[k2, support[k]] - footF[k, support[k]])
    vel = np.asarray(vel) if vel else np.zeros(1)
    speed = float(np.median(vel)) * fps
    print(f"stance-foot speed {speed:+.2f} m/s  ({len(vel)} stance frames of {N})")

    # ---- periodic smoothing: safe because the cycle is a loop ------------
    names = sorted(cyc[0])
    flat = np.array([np.stack([cyc[k][n] for n in names]) for k in range(N)])
    flat = circular_lowpass(flat, passes=1)
    cyc = [{n: flat[k, j] for j, n in enumerate(names)} for k in range(N)]

    # ---- ground + travel -------------------------------------------------
    footY = np.array([[Gk[n][UP, 3] for n in foot_nodes] for Gk in cyc])
    bob = circular_lowpass(footY.min(axis=1)[:, None], passes=1)[:, 0]
    lift = bob.min() - bob                      # lowest contact lands on y=0

    # ---- rigid yaw into the render frame (character now travels along +X) ---
    pivot = rig.bind_p[rig.root_bone].copy()
    ang = np.radians(args.yaw)
    for k in range(N):
        cyc[k] = rigid_yaw(cyc[k], pivot, ang)

    # travel along +X once yawed; keep the vertical lift as measured
    off = np.zeros((N, 3))
    off[:, UP] = lift
    off[:, 0] = speed * np.arange(N) / fps

    loc = np.zeros((N, len(names), 4, 4))
    for k in range(N):
        for j, n in enumerate(names):
            par = rig.parent.get(n)
            loc[k, j] = (np.linalg.inv(cyc[k][par]) @ cyc[k][n]
                         if par is not None else cyc[k][n])

    # travel is applied on the parentless (root) node, in world space
    for j, n in enumerate(names):
        if rig.parent.get(n) is None:
            loc[:, j, UP, 3] += off[:, UP]
            loc[:, j, 0, 3] += off[:, 0]

    np.savez_compressed(
        args.out, fps=fps, N=N, W=W, H=H, speed=speed,
        seq=np.array([[i, int(m)] for i, m in seq]),
        node_names=np.array([scene.nodes[n].get("name") or f"node{n}" for n in names]),
        node_ids=np.array(names), local=loc, off=off,
        src=np.array([args.start, args.end]),
    )
    print(f"wrote {args.out}: {N} frames x {len(names)} nodes, "
          f"travel {off[-1, 0] + speed / fps:.2f} m per cycle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
