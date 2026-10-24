"""
Minimal, dependency-light glTF 2.0 / GLB loader with skinning support.

Deliberately does not use trimesh/pygltflib: the sandbox has no GL stack and the
only thing we need is (a) geometry, (b) the node/bone hierarchy, (c) skin
weights — all of which are a few hundred lines of struct + numpy.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# low-level accessors
# --------------------------------------------------------------------------- #
_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
         5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _trs_matrix(node: dict) -> np.ndarray:
    """Node local transform from either `matrix` or T*R*S."""
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    T = np.eye(4)
    if "translation" in node:
        T[:3, 3] = node["translation"]
    R = np.eye(4)
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    S = np.eye(4)
    if "scale" in node:
        S[:3, :3] = np.diag(node["scale"])
    return T @ R @ S


@dataclass
class Mesh:
    name: str
    positions: np.ndarray            # (N, 3) float32
    normals: np.ndarray              # (N, 3) float32
    uvs: np.ndarray                  # (N, 2) float32
    joints: np.ndarray               # (N, 4) int32  -> indices into skin.joints
    weights: np.ndarray              # (N, 4) float32
    indices: np.ndarray              # (T*3,) uint32
    material: int | None = None
    node: int = -1

    @property
    def n_verts(self) -> int:
        return len(self.positions)

    @property
    def n_tris(self) -> int:
        return len(self.indices) // 3


@dataclass
class Skin:
    name: str
    joints: list[int]                # node indices
    inverse_bind: np.ndarray         # (J, 4, 4)
    skeleton: int | None = None


@dataclass
class Material:
    name: str
    base_color: np.ndarray           # (4,) linear
    base_tex: int | None = None
    metallic: float = 0.0
    roughness: float = 1.0


@dataclass
class Scene:
    nodes: list[dict]
    meshes: list[Mesh]
    skins: list[Skin]
    materials: list[Material]
    textures: list[np.ndarray | None]
    parents: dict[int, int] = field(default_factory=dict)
    global_bind: dict[int, np.ndarray] = field(default_factory=dict)
    root_nodes: list[int] = field(default_factory=list)

    # -- hierarchy ---------------------------------------------------------
    def chain(self, idx: int) -> list[int]:
        """Node indices from `idx` up to (and including) the scene root."""
        out = [idx]
        while out[-1] in self.parents:
            out.append(self.parents[out[-1]])
        return out

    def node_by_name(self, name: str) -> int | None:
        for i, n in enumerate(self.nodes):
            if n.get("name") == name:
                return i
        return None

    def bind_position(self, idx: int) -> np.ndarray:
        return self.global_bind[idx][:3, 3]

    def skin_joint_positions(self, skin: Skin) -> np.ndarray:
        return np.stack([self.bind_position(j) for j in skin.joints])


def _decode_image(data: bytes) -> np.ndarray | None:
    """Decode PNG/JPEG bytes to RGB uint8 without cv2 at import time."""
    try:
        import cv2
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        # cv2 always decodes to BGR(A); keep the first 3 channels and flip to RGB
        return np.ascontiguousarray(img[:, :, 2::-1])
    except Exception:
        return None


class GLB:
    def __init__(self, path: str):
        with open(path, "rb") as fh:
            magic, version, total = struct.unpack("<III", fh.read(12))
            if magic != 0x46546C67:
                raise ValueError(f"{path}: not a GLB (magic={magic:#x})")
            self.version = version
            self.length = total
            self.gltf: dict = {}
            self._bin = b""
            while fh.tell() < total:
                chunk_len, chunk_type = struct.unpack("<II", fh.read(8))
                payload = fh.read(chunk_len)
                if chunk_type == 0x4E4F534A:      # JSON
                    self.gltf = json.loads(payload)
                elif chunk_type == 0x004E4942:    # BIN
                    self._bin = payload
        self.buffers = [self._bin]

    # -- accessors ---------------------------------------------------------
    def accessor(self, idx: int) -> np.ndarray:
        acc = self.gltf["accessors"][idx]
        bv = self.gltf["bufferViews"][acc["bufferView"]]
        dtype = np.dtype(_COMP[acc["componentType"]])
        ncomp = _NCOMP[acc["type"]]
        base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        count = acc["count"]
        stride = bv.get("byteStride")
        buf = self.buffers[bv.get("buffer", 0)]
        if stride and stride != dtype.itemsize * ncomp:
            raw = np.frombuffer(buf, dtype=np.uint8, count=stride * count, offset=base)
            raw = raw.reshape(count, stride)[:, : dtype.itemsize * ncomp]
            arr = raw.copy().view(dtype).reshape(count, ncomp)
        else:
            arr = np.frombuffer(buf, dtype=dtype, count=count * ncomp,
                                offset=base).reshape(count, ncomp)
        return np.array(arr)  # writable copy

    # -- scene building ----------------------------------------------------
    def load(self) -> Scene:
        g = self.gltf
        nodes = g.get("nodes", [])

        parents: dict[int, int] = {}
        root_nodes: list[int] = []
        for i, n in enumerate(nodes):
            for c in n.get("children", []):
                parents[c] = i
        roots = g.get("scenes", [{}])[g.get("scene", 0)].get("nodes", [])
        root_nodes = list(roots)

        global_bind: dict[int, np.ndarray] = {}

        def walk(i: int, parent_mat: np.ndarray) -> None:
            mat = parent_mat @ _trs_matrix(nodes[i])
            global_bind[i] = mat
            for c in nodes[i].get("children", []):
                walk(c, mat)

        for r in roots:
            walk(r, np.eye(4))

        # textures
        textures: list[np.ndarray | None] = []
        for img in g.get("images", []):
            data = None
            if "bufferView" in img:
                bv = g["bufferViews"][img["bufferView"]]
                off = bv.get("byteOffset", 0)
                data = self.buffers[bv.get("buffer", 0)][off: off + bv["byteLength"]]
            elif "uri" in img and img["uri"].startswith("data:"):
                import base64
                data = base64.b64decode(img["uri"].split(",", 1)[1])
            textures.append(_decode_image(data) if data else None)

        # materials
        materials: list[Material] = []
        for m in g.get("materials", []):
            pbr = m.get("pbrMetallicRoughness", {})
            bcf = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1]), dtype=np.float64)
            bct = None
            if "baseColorTexture" in pbr:
                ti = pbr["baseColorTexture"]["index"]
                bct = g["textures"][ti]["source"] if "textures" in g else None
            materials.append(Material(
                name=m.get("name", ""), base_color=bcf, base_tex=bct,
                metallic=float(pbr.get("metallicFactor", 0.0)),
                roughness=float(pbr.get("roughnessFactor", 1.0)),
            ))

        # skins
        skins: list[Skin] = []
        for s in g.get("skins", []):
            ibm = np.stack([np.eye(4)] * len(s["joints"]))
            if "inverseBindMatrices" in s:
                raw = self.accessor(s["inverseBindMatrices"]).astype(np.float64)
                ibm = raw.reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major -> row-major
            skins.append(Skin(name=s.get("name", ""), joints=list(s["joints"]),
                              inverse_bind=ibm, skeleton=s.get("skeleton")))

        # meshes (one Mesh object per primitive)
        meshes: list[Mesh] = []
        for ni, n in enumerate(nodes):
            if "mesh" not in n:
                continue
            mi = n["mesh"]
            skin = skins[n["skin"]] if "skin" in n else None
            for prim in g["meshes"][mi]["primitives"]:
                attrs = prim["attributes"]
                pos = self.accessor(attrs["POSITION"]).astype(np.float32)
                nrm = (self.accessor(attrs["NORMAL"]).astype(np.float32)
                       if "NORMAL" in attrs else np.zeros_like(pos))
                uv = (self.accessor(attrs["TEXCOORD_0"]).astype(np.float32)
                      if "TEXCOORD_0" in attrs else np.zeros((len(pos), 2), np.float32))
                if "JOINTS_0" in attrs:
                    joints = self.accessor(attrs["JOINTS_0"]).astype(np.int32)
                    weights = self.accessor(attrs["WEIGHTS_0"]).astype(np.float32)
                else:
                    joints = np.zeros((len(pos), 4), np.int32)
                    weights = np.zeros((len(pos), 4), np.float32)
                    weights[:, 0] = 1.0
                idx = (self.accessor(prim["indices"]).astype(np.uint32).ravel()
                       if "indices" in prim else np.arange(len(pos), dtype=np.uint32))
                meshes.append(Mesh(
                    name=g["meshes"][mi].get("name") or f"mesh{mi}",
                    positions=pos, normals=nrm, uvs=uv, joints=joints, weights=weights,
                    indices=idx, material=prim.get("material"), node=ni,
                ))

        return Scene(nodes=nodes, meshes=meshes, skins=skins, materials=materials,
                     textures=textures, parents=parents, global_bind=global_bind,
                     root_nodes=root_nodes)


def load(path: str) -> Scene:
    return GLB(path).load()


# --------------------------------------------------------------------------- #
# skinning
# --------------------------------------------------------------------------- #
def skin_mesh(mesh: Mesh, skin: Skin, joint_mats: np.ndarray) -> np.ndarray:
    """Linear blend skinning.

    joint_mats: (J, 4, 4) global bone matrices for the current pose.
    Returns (N, 3) posed vertices.
    """
    N = mesh.n_verts
    v = np.hstack([mesh.positions, np.ones((N, 1), dtype=np.float32)]).astype(np.float64)

    # (N, 4) influence matrices
    M = joint_mats[mesh.joints] @ skin.inverse_bind[mesh.joints]   # (N,4,4,4)
    M = np.einsum("njab,nj->nab", M, mesh.weights)

    posed = np.einsum("nab,nb->na", M, v)
    w = posed[:, 3:4]
    return (posed[:, :3] / np.where(np.abs(w) < 1e-9, 1.0, w)).astype(np.float32)


def pose_normals(mesh: Mesh, skin: Skin, joint_mats: np.ndarray) -> np.ndarray:
    """Rotate normals by the blended linear part (no translation)."""
    N = mesh.n_verts
    M = joint_mats[mesh.joints][..., :3, :3] @ skin.inverse_bind[mesh.joints][..., :3, :3]
    M = np.einsum("njab,nj->nab", M, mesh.weights)
    n = np.einsum("nab,nb->na", M, mesh.normals.astype(np.float64))
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def global_transforms(scene: Scene, local_rotations: dict[int, np.ndarray] | None = None,
                      root_translation: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Recursive forward kinematics.

    local_rotations: node index -> 3x3 rotation replacing that node's bind rotation.
    root_translation: added translation applied to the skeleton root.
    """
    loc = {i: _trs_matrix(n) for i, n in enumerate(scene.nodes)}
    if local_rotations:
        for idx, R in local_rotations.items():
            L = loc[idx].copy()
            L[:3, :3] = R
            loc[idx] = L
    if root_translation is not None and scene.skins:
        sk = scene.skins[0].skeleton
        if sk is None:
            sk = scene.skins[0].joints[0]
        L = loc[sk].copy()
        L[:3, 3] = L[:3, 3] + np.asarray(root_translation, dtype=np.float64)
        loc[sk] = L

    out: dict[int, np.ndarray] = {}

    def walk(i: int, parent: np.ndarray) -> None:
        m = parent @ loc[i]
        out[i] = m
        for c in scene.nodes[i].get("children", []):
            walk(c, m)

    for r in scene.root_nodes:
        walk(r, np.eye(4))
    return out
