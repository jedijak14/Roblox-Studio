"""
Retarget 2D/3D pose landmarks (MediaPipe BlazePose, 33 points) onto the GLB's rig.

MediaPipe emits landmarks in its own normalized space:
    x, y  in [0, 1] of the image
    z     roughly [-1, 1], in units of image *width*, with -z towards the camera

We convert these into a right-handed Y-up, metres, character-facing-+Z space, then
for each rig bone estimate a rotation that swings the bone's bind direction onto the
observed bone direction, using the shortest-arc (swing-only, no twist) solution.
That is the standard cheap retarget: twist is not observable from sparse landmarks,
so leaving it at bind is both stable and visually fine.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# MediaPipe BlazePose landmark indices
# --------------------------------------------------------------------------- #
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# MediaPipe is left-right "as seen in the image"; the rig is mirrored
# (left bones at +x). Swap sides when retargeting.
MP_TO_RIG = [
    (L_SHOULDER, "arm_l"), (R_SHOULDER, "arm_r"),
    (L_ELBOW, "elbow_l"), (R_ELBOW, "elbow_r"),
    (L_WRIST, "hand_l"), (R_WRIST, "hand_r"),
    (L_HIP, "thigh_l"), (R_HIP, "thigh_r"),
    (L_KNEE, "knee_l"), (R_KNEE, "knee_r"),
    (L_ANKLE, "foot_l"), (R_ANKLE, "foot_r"),
]


def to_canonical(landmarks: np.ndarray, visibility: np.ndarray | None = None,
                 depth_scale: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """MediaPipe normalized landmarks -> (33, 3) canonical Y-up/world-ish, metres.

    axis mapping:  world_x = +x,  world_y = -y,  world_z = -z
    A standing person ends up facing +Z (towards a camera placed at +Z).
    Scale is normalised so shoulder width == 0.36 m.
    """
    p = np.asarray(landmarks, dtype=np.float64).copy()
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N,3) landmarks, got {p.shape}")
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    out[:, 1] = -p[:, 1]
    out[:, 2] = -p[:, 2] * depth_scale

    # anchor: mid-hip at the origin
    hips = 0.5 * (out[L_HIP] + out[R_HIP])
    out -= hips

    sw = np.linalg.norm(out[L_SHOULDER] - out[R_SHOULDER])
    if sw > 1e-6:
        out *= 0.36 / sw
    return out, (visibility if visibility is not None else np.ones(len(out)))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


def swing_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking unit vector `a` to unit vector `b`."""
    a = _unit(np.asarray(a, dtype=np.float64))
    b = _unit(np.asarray(b, dtype=np.float64))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = _unit(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))


class Retargeter:
    """Maps BlazePose landmarks to world-space bone directions for the rig."""

    def __init__(self, bind_positions: dict[str, np.ndarray], fallback_depth: float = 0.12):
        """
        bind_positions: bone name -> bind world position (for direction priors)
        """
        self.bind = bind_positions
        self.fallback_depth = fallback_depth

    def bone_directions(self, canon: np.ndarray, vis: np.ndarray) -> dict[str, np.ndarray]:
        """Return rig-bone-name -> observed world direction (unit vectors)."""
        d: dict[str, np.ndarray] = {}
        # arms: parent joint -> child joint
        segs = {
            "arm_l": (L_SHOULDER, L_ELBOW), "arm_r": (R_SHOULDER, R_ELBOW),
            "elbow_l": (L_ELBOW, L_WRIST), "elbow_r": (R_ELBOW, R_WRIST),
            "hand_l": (L_WRIST, L_INDEX), "hand_r": (R_WRIST, R_INDEX),
            "thigh_l": (L_HIP, L_KNEE), "thigh_r": (R_HIP, R_KNEE),
            "knee_l": (L_KNEE, L_ANKLE), "knee_r": (R_KNEE, R_ANKLE),
            "foot_l": (L_ANKLE, L_FOOT), "foot_r": (R_ANKLE, R_FOOT),
            "spine": (None, None),
        }
        for name, (a, b) in segs.items():
            if a is None:
                midh = 0.5 * (canon[L_HIP] + canon[R_HIP])
                midc = 0.5 * (canon[L_SHOULDER] + canon[R_SHOULDER])
                d[name] = _unit(midc - midh)
                continue
            d[name] = _unit(canon[b] - canon[a])
        # Head: the rig's head bone points *forward* (towards the cheek), not up,
        # so the matching landmark vector is head -> nose, not shoulders -> ears.
        # (Feeding it an "up" vector rotates the skull ~45 deg backwards.)
        ear_mid = 0.5 * (canon[L_EAR] + canon[R_EAR])
        d["head"] = _unit(canon[NOSE] - ear_mid)
        # hips: average of left/right thigh directions, gives pelvic orientation
        d["hips"] = _unit(d["thigh_l"] + d["thigh_r"])
        return d


# --------------------------------------------------------------------------- #
# rig-space solving
# --------------------------------------------------------------------------- #
def solve_local_rotations(bind_global: dict[int, np.ndarray],
                          parent_of: dict[int, int],
                          order: list[int],
                          bone_targets: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Solve per-node local rotations so each bone points along its target direction.

    bind_global: node -> bind world matrix
    order: node indices in parent-before-child order
    bone_targets: node -> unit world direction the bone should point in
    Returns node -> 3x3 local rotation (replacing the bind local rotation).
    """
    local_rot: dict[int, np.ndarray] = {}
    pose_global: dict[int, np.ndarray] = {}

    for idx in order:
        parent = parent_of.get(idx)
        bind_w = bind_global[idx]
        bind_pos = bind_w[:3, 3]

        if parent is not None and parent in pose_global:
            parent_pose = pose_global[parent]
            parent_bind = bind_global[parent]
        else:
            parent_pose = np.eye(4)
            parent_bind = np.eye(4)

        # where this bone sits after the parent has moved
        cur_pos = (parent_pose @ np.linalg.inv(parent_bind) @ bind_w)[:3, 3]

        target = bone_targets.get(idx)
        if target is None:
            R_local = np.eye(3)
        else:
            # current bone direction implied by parent pose
            cur_dir = _unit(cur_pos - pose_global[parent][:3, 3]) if parent in pose_global else None
            if cur_dir is None or np.linalg.norm(cur_pos - pose_global[parent][:3, 3]) < 1e-8:
                cur_dir = _unit(bind_w[:3, 3] - parent_bind[:3, 3])
            R_swing = swing_matrix(cur_dir, target)
            R_local = R_swing

        pose_global[idx] = _compose_pose(parent_pose, parent_bind, bind_w, R_local)
        local_rot[idx] = R_local
    return local_rot


def _compose_pose(parent_pose: np.ndarray, parent_bind: np.ndarray,
                  bind_w: np.ndarray, R_local: np.ndarray) -> np.ndarray:
    """Pose world matrix for a bone, given a *world-space* swing rotation."""
    # rotate about the bone's own origin
    T = np.eye(4)
    T[:3, 3] = bind_w[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, 3] = -bind_w[:3, 3]
    R = np.eye(4)
    R[:3, :3] = R_local
    posed = parent_pose @ np.linalg.inv(parent_bind) @ T @ R @ T_inv @ bind_w
    return posed
