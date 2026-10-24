#!/usr/bin/env python
"""
build_cycle.py — one tracked stride -> a symmetric, loopable run cycle.

Why this shape
--------------
In the opening, only frames ~234-247 show the whole body, and in profile the far
arm and far leg are occluded (MediaPipe reports them at ~0.35-0.45 confidence and
their angles jump by 100+ degrees between frames). Feeding those to the solver is
what produced the flailing.

A run is antisymmetric: the far limb is the near limb half a stride later, mirrored.
So this script takes the NEAR (well-observed) chain only, phase-bins it over one
stride, resamples onto a regular cycle, and generates the far chain as the same
signal at phase + 0.5 reflected through the sagittal plane (x -> -x).

Result: no dead limbs, exact left/right symmetry, and a loop that closes exactly
because every channel is a periodic function of phase.

usage:
  build_cycle.py --tracked docs/motion/run_roi.json --glb "scene 6.glb" \
                 --start 234 --end 248 --near l --frames 10 \
                 --out docs/motion/run_cycle.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glb as glblib                        # noqa: E402
from retarget import _unit                  # noqa: E402
from apply_video_motion import Rig          # noqa: E402
from solve_motion import (canonical_side, zero_phase_lowpass, rigid_yaw,
                          circular_lowpass)  # noqa: E402

L = dict(sh=11, el=13, wr=15, ix=19, hi=23, kn=25, an=27, ft=31)
R = dict(sh=12, el=14, wr=16, ix=20, hi=24, kn=26, an=28, ft=32)

SIDE = {"l": L, "r": R}


def _axis_angle(R: np.ndarray) -> tuple[float, np.ndarray]:
    """(angle, unit axis) of a rotation matrix."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    if ang < 1e-9:
        return 0.0, np.array([0.0, 1.0, 0.0])
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-9:                       # 180 deg: ambiguous axis
        return ang, np.array([0.0, 1.0, 0.0])
    return ang, ax / n


def _from_axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation."""
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def chain(P: np.ndarray, S: dict) -> dict[str, np.ndarray]:
    """Unit directions of the six limb bones for one side, in canonical space."""
    return {
        "arm":     _unit(P[S["el"]] - P[S["sh"]]),
        "elbow":   _unit(P[S["wr"]] - P[S["el"]]),
        "thigh":   _unit(P[S["kn"]] - P[S["hi"]]),
        "knee":    _unit(P[S["an"]] - P[S["kn"]]),
        "foot":    _unit(P[S["ft"]] - P[S["an"]]),
    }


def phase_from(sig: np.ndarray, period: float) -> np.ndarray:
    """Phase in [0,1) from a near-sinusoidal 1-D signal and a known period."""
    n = len(sig)
    t = np.arange(n)
    # project onto the stride frequency (and its first harmonic) -> analytic phase
    c1 = np.cos(2 * np.pi * t / period); s1 = np.sin(2 * np.pi * t / period)
    x = sig - sig.mean()
    A = np.stack([c1, s1], 1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    ang = np.arctan2(coef[1], coef[0])
    return (t / period - ang / (2 * np.pi)) % 1.0


def find_period(sig: np.ndarray) -> float:
    x = sig - sig.mean()
    if np.abs(x).max() < 1e-9:
        return 8.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0] + 1e-12
    lo, hi = 3, min(len(ac) - 1, len(x) // 2)
    if hi <= lo:
        return 8.0
    return float(lo + np.argmax(ac[lo:hi]))


def interp_at(ph_src: np.ndarray, vals: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Circular linear interpolation of vals(ph_src) evaluated at ph `query`.

    ph_src, query in [0,1); vals (T, ...).
    """
    order = np.argsort(ph_src)
    p = np.concatenate([ph_src[order], ph_src[order][:1] + 1.0])
    flat = vals[order].reshape(len(vals), -1)
    flat = np.concatenate([flat, flat[:1]], axis=0)
    q = np.asarray(query, float) % 1.0
    out = np.empty((len(q), flat.shape[1]))
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(q, p, flat[:, j])
    return out.reshape((len(q),) + vals.shape[1:])


def resample(ph: np.ndarray, vals: np.ndarray, N: int) -> np.ndarray:
    """Rebin vals(ph) onto a regular N-frame cycle."""
    grid = (np.arange(N) + 0.5) / N
    return interp_at(ph, vals, grid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracked", required=True)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--near", choices=["l", "r"], default="l")
    ap.add_argument("--frames", type=int, default=10, help="frames per stride")
    ap.add_argument("--arm-gain", type=float, default=1.0,
                    help="exaggerate the upper-arm swing about its cycle mean (1 = as measured)")
    ap.add_argument("--head-follow", type=float, default=1.0,
                    help="fraction of the inherited spine rotation the head keeps")
    ap.add_argument("--harmonics", type=int, default=3,
                    help="harmonics of the stride frequency used to model each channel")
    ap.add_argument("--cadence", type=float, default=1.0,
                    help="time stretch; >1 slows the cycle without foot sliding")
    ap.add_argument("--depth", type=float, default=0.20)
    ap.add_argument("--smooth", type=float, default=0.85)
    ap.add_argument("--yaw", type=float, default=90.0)
    args = ap.parse_args()

    data = json.load(open(args.tracked))
    fps = data["fps"]
    by_i = {f["i"]: f for f in data["frames"]}
    idx = [i for i in range(args.start, args.end + 1)
           if by_i.get(i, {}).get("det")]
    T = len(idx)
    if T < 6:
        print("not enough tracked frames"); return 1

    # ---- canonical joint cloud -------------------------------------------
    Pt = np.zeros((T, 33, 3))
    for k, i in enumerate(idx):
        f = by_i[i]
        xy = np.asarray(f["xy"], float); vis = np.asarray(f["vis"], float)
        dep = np.asarray(f.get("world") or np.zeros((33, 3)), float)
        if dep.shape != (33, 3):
            dep = np.zeros((33, 3))
        sw = float(np.linalg.norm(xy[L["sh"]] - xy[R["sh"]]))
        sw = sw if np.isfinite(sw) and sw > 5 else 90.0
        Pt[k] = canonical_side(xy, vis, dep[:, 2] / max(sw, 1e-6), sw)

    for j in range(33):
        Pt[:, j] = zero_phase_lowpass(Pt[:, j], args.smooth)

    # ---- near-side chain, phase-binned -----------------------------------
    S = SIDE[args.near]
    dm = chain(Pt[0], S)
    seq = {k: np.stack([chain(Pt[t], S)[k] for t in range(T)]) for k in dm}
    for k in seq:
        seq[k] = np.stack([zero_phase_lowpass(seq[k][:, c], args.smooth)
                           for c in range(3)], axis=1)

    # ---- pick the stride period, then fit a harmonic model in phase ---------
    # Phase-binning 12 samples over ~1.5 cycles leaves most bins empty and
    # interpolation silently flattens the stride. Fitting 1..K harmonics in phase
    # keeps the amplitude and is periodic by construction.
    ankle_y = -Pt[:, S["an"], 1]
    probe = ["thigh", "knee", "arm", "elbow"]
    K = int(args.harmonics)

    def design(ph):
        cols = [np.ones_like(ph)]
        for kk in range(1, K + 1):
            cols += [np.cos(2 * np.pi * kk * ph), np.sin(2 * np.pi * kk * ph)]
        return np.stack(cols, 1)

    def fit(ph, vals):
        A = design(ph)
        coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
        return coef

    def evaluate(coef, ph):
        return design(ph) @ coef

    best = None
    for period in np.arange(4.0, min(14.0, T - 1) + 0.01, 0.1):
        phc = phase_from(ankle_y, period)
        A = design(phc)
        if np.linalg.matrix_rank(A) < A.shape[1]:
            continue
        err = []
        for k in probe:
            rec = evaluate(fit(phc, seq[k]), phc)
            rec = np.stack([_unit(v) for v in rec])
            err.append(np.mean(np.clip(1 - np.sum(seq[k] * rec, 1), 0, 2)))
        score = float(np.mean(err)) + 0.02 * abs(period - 8.0)
        if best is None or score < best[0]:
            best = (score, float(period), phc)
    score, period, ph = best
    print(f"near={args.near}  frames={T}  stride period ~{period:.1f} src frames "
          f"({period / fps:.2f}s)  fit residual {score:.3f}")

    N = max(6, int(round(args.frames)))
    grid = (np.arange(N) + 0.5) / N
    cyc = {}
    for k in seq:
        co = fit(ph, seq[k])
        v = evaluate(co, grid)
        cyc[k] = np.stack([_unit(x) for x in v])
        a = np.degrees(np.arctan2(seq[k][:, 2], -seq[k][:, 1]))
        b = np.degrees(np.arctan2(cyc[k][:, 2], -cyc[k][:, 1]))
        print(f"  {k:6s} amp {np.ptp(a):5.0f} deg -> {np.ptp(b):5.0f} deg")

    # ---- generate the far chain: same signal, half a cycle later, x -> -x --
    far_key = "r" if args.near == "l" else "l"
    half = N // 2
    for k in list(cyc):
        v = np.roll(cyc[k], -half, axis=0).copy()
        v[:, 0] *= -1.0
        cyc[f"far_{k}"] = v

    # ---- optional exaggeration of the upper-arm swing --------------------
    # The near arm spends much of the shot folded across the chest, so the sagittal
    # signal is small and the arms read as stiff. Scaling the ANGULAR deviation from
    # the cycle mean keeps the measured phase and shape, only its size.
    if abs(args.arm_gain - 1.0) > 1e-6:
        for key in ("arm", f"far_arm"):
            m = _unit(cyc[key].mean(axis=0))
            out = np.empty_like(cyc[key])
            for t in range(len(cyc[key])):
                d = _unit(cyc[key][t])
                c = float(np.clip(np.dot(m, d), -1.0, 1.0))
                v = np.cross(m, d)
                sv = float(np.linalg.norm(v))
                if sv < 1e-9:
                    out[t] = d
                    continue
                ang = np.arctan2(sv, c) * args.arm_gain
                out[t] = _unit(m * np.cos(ang) + (v / sv) * np.sin(ang))
            cyc[key] = out
        print(f"  arm swing exaggerated x{args.arm_gain:g}")

    # ---- keep the cycle genuinely periodic under smoothing ---------------
    for k in list(cyc):
        cyc[k] = circular_lowpass(cyc[k], passes=1)
        cyc[k] = np.stack([_unit(d) for d in cyc[k]])

    # ---- assemble per-frame direction dicts ------------------------------
    print(f"cycle {N} frames ({N / fps * args.cadence:.2f}s per stride)")
    frames_dirs = []
    for t in range(N):
        d = {}
        for k in ("arm", "elbow", "thigh", "knee", "foot"):
            d[f"{k}_{args.near}"] = cyc[k][t]
            d[f"{k}_{far_key}"] = cyc[f"far_{k}"][t]
        frames_dirs.append(d)

    # ---- spine: from data, phase-binned (midline, never mirrored) --------
    spine = np.stack([_unit(Pt[t, L["sh"]] + Pt[t, R["sh"]]
                            - Pt[t, L["hi"]] - Pt[t, R["hi"]]) for t in range(T)])
    spine = np.stack([zero_phase_lowpass(spine[:, c], args.smooth) for c in range(3)], 1)
    spine_cyc = evaluate(fit(ph, spine), grid)
    spine_cyc[:, 0] *= 0.35                      # no lateral lean; keep it small
    spine_cyc = np.stack([_unit(v) for v in spine_cyc])
    for t in range(N):
        frames_dirs[t]["spine"] = spine_cyc[t]

    # ---- solve ------------------------------------------------------------
    scene = glblib.load(args.glb)
    rig = Rig(scene)
    G = [rig.pose(frames_dirs[t], root_offset=np.zeros(3)) for t in range(N)]

    # ---- head stabilisation ----------------------------------------------
    # The head node inherits the whole spine, so a 20 deg running lean makes the
    # character stare at her shoes (and shows scalp instead of face). Animators
    # keep the head level: keep only `--head-follow` of the inherited rotation.
    head_node = rig.name_to_idx.get("head_Armature")
    if head_node is not None and args.head_follow < 1.0:
        f = args.head_follow
        for t in range(N):
            par = rig.parent.get(head_node)
            if par is None:
                continue
            d_parent = (G[t][par][:3, :3] @ np.linalg.inv(rig.bind_R[par]))
            # rotate about the axis of d_parent by a fraction of its angle
            w, v = _axis_angle(d_parent)
            if w > 1e-6:
                d_red = _from_axis_angle(v, w * f)
                G[t][head_node][:3, :3] = d_red @ rig.bind_R[head_node]

    # ---- gravity: lowest foot on the floor, natural bob preserved --------
    foot_nodes = [rig.name_to_idx[n] for n in ("foot_l_Armature", "foot_r_Armature")
                  if n in rig.name_to_idx]
    low = np.array([min(G[t][n][1, 3] for n in foot_nodes) for t in range(N)])
    bob = circular_lowpass(low[:, None], passes=1)[:, 0]
    lift = -bob

    # ---- stance speed: the planted foot's backward drift in the body frame
    # A run has a flight phase, so "the lower foot" is arbitrary there. Only count
    # frames where a foot is actually within a few cm of the floor: on those, the
    # planted foot must be stationary in the world, which pins the travel speed.
    footY = np.array([[G[t][n][1, 3] for n in foot_nodes] for t in range(N)])
    footZ = np.array([[G[t][n][2, 3] for n in foot_nodes] for t in range(N)])
    support = np.argmin(footY, axis=1)
    floor_y = footY.min()
    contact = [t for t in range(N)
               if footY[t, support[t]] < floor_y + 0.035]
    vel = [footZ[(t + 1) % N, support[t]] - footZ[t, support[t]]
           for t in contact if support[t] == support[(t + 1) % N]]
    speed = -float(np.median(vel)) * fps / max(args.cadence, 1e-6) if vel else 0.0
    speed = abs(speed)
    print(f"  contact frames {len(contact)}/{N}, velocity samples {len(vel)}")
    print(f"stance-foot speed {speed:.2f} m/s "
          f"({len(vel)}/{N} frames with a single support foot)")

    # ---- yaw into the render frame, then travel along +X -----------------
    pivot = rig.bind_p[rig.root_bone].copy()
    G = [rigid_yaw(G[t], pivot, np.radians(args.yaw)) for t in range(N)]

    names = sorted(G[0])
    loc = np.zeros((N, len(names), 4, 4))
    root_j = [j for j, n in enumerate(names) if rig.parent.get(n) is None][0]
    for t in range(N):
        for j, n in enumerate(names):
            par = rig.parent.get(n)
            loc[t, j] = (np.linalg.inv(G[t][par]) @ G[t][n]
                         if par is not None else G[t][n])
        loc[t, root_j, 1, 3] += lift[t]
        loc[t, root_j, 0, 3] += speed * t / fps * args.cadence

    off = np.zeros((N, 3))
    off[:, 1] = lift
    off[:, 0] = speed * np.arange(N) / fps * args.cadence

    np.savez_compressed(
        args.out, fps=fps, N=N, local=loc, off=off, speed=speed,
        node_ids=np.array(names),
        node_names=np.array([scene.nodes[n].get("name") or f"node{n}" for n in names]),
        src=np.array([args.start, args.end]), near=args.near,
        stride_src=period, cadence=args.cadence,
    )
    print(f"wrote {args.out}: {N} frames, travel {off[-1, 0]:.2f} m/stride")
    return 0


if __name__ == "__main__":
    sys.exit(main())
