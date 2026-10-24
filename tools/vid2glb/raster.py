"""
ctypes wrapper around tiny_raster.c, plus a small camera model.

Falls back to a (slow) pure-numpy rasterizer if a C compiler is unavailable.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tiny_raster.c")
_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def _build_lib() -> ctypes.CDLL | None:
    if SRC in _LIB_CACHE:
        return _LIB_CACHE[SRC]
    # Prefer a copy in a writable cache dir so the repo stays source-only.
    out_dir = os.environ.get("VID2GLB_CACHE", os.path.join(tempfile.gettempdir(), "vid2glb"))
    os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, "libtiny_raster.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(SRC):
        cc = os.environ.get("CC", "gcc")
        try:
            subprocess.run([cc, "-O3", "-fPIC", "-shared", SRC, "-o", so, "-lm"],
                           check=True, capture_output=True)
        except Exception:
            return None
    lib = ctypes.CDLL(so)
    f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    i32p = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    u8p = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    lib.rasterize.argtypes = [f32p, f32p, f32p, i32p, ctypes.c_int,
                              u8p, ctypes.c_int, ctypes.c_int,
                              f32p, f32p, ctypes.c_int, ctypes.c_int,
                              u8p, f32p, f32p, ctypes.c_float, ctypes.c_int]
    lib.rasterize.restype = None
    _LIB_CACHE[SRC] = lib
    return lib


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
class Camera:
    """Pinhole camera looking down -Z, with look-at."""

    def __init__(self, eye, target=(0.0, 0.75, 0.0), up=(0.0, 1.0, 0.0),
                 fov_deg=32.0, width=1280, height=720):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)
        self.fov = fov_deg
        self.W, self.H = int(width), int(height)

    def view_matrix(self) -> np.ndarray:
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, self.up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        # camera looks down -Z, so +Z_cam = -f
        R = np.stack([r, u, -f])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = -R @ self.eye
        return M

    def intrinsics(self) -> np.ndarray:
        f = 0.5 * self.H / np.tan(np.radians(self.fov) / 2)
        return np.array([[f, 0, self.W / 2.0],
                         [0, f, self.H / 2.0],
                         [0, 0, 1.0]])

    def look_at_point(self, x, y, z):
        self.target = np.array([x, y, z], dtype=np.float64)


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def render(vertices: np.ndarray, normals: np.ndarray, uvs: np.ndarray,
           indices: np.ndarray, texture: np.ndarray | None,
           cam: Camera, bg=(0.06, 0.07, 0.10), light=(0.4, 0.6, 0.7),
           alpha=1.0, depth_cue=True, cull=True) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize into an RGBA uint8 image. Returns (image, depth)."""
    W, H = cam.W, cam.H
    view = cam.view_matrix().astype(np.float32)
    proj = cam.intrinsics().astype(np.float32)

    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[:, :, 0] = int(bg[0] * 255)
    out[:, :, 1] = int(bg[1] * 255)
    out[:, :, 2] = int(bg[2] * 255)
    out[:, :, 3] = 0          # alpha == coverage; set before any blending
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    tex = None if texture is None else np.ascontiguousarray(texture, dtype=np.uint8)
    tw, th = (0, 0) if tex is None else (tex.shape[1], tex.shape[0])
    if tex is None:      # keep argtypes happy with a 1-pixel stand-in
        tex = np.zeros((1, 1, 3), dtype=np.uint8)
        tw = th = 1

    l = np.asarray(light, dtype=np.float32)
    l = l / (np.linalg.norm(l) + 1e-9)

    lib = _build_lib()
    tri = np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int32)
    if lib is not None:
        lib.rasterize(np.ascontiguousarray(vertices, dtype=np.float32),
                      np.ascontiguousarray(normals, dtype=np.float32),
                      np.ascontiguousarray(uvs, dtype=np.float32),
                      tri, len(tri), tex, tw, th,
                      view, proj, W, H, out, zbuf, l, float(alpha), int(bool(cull)))
    else:
        out, zbuf = _render_numpy(vertices, normals, uvs, tri, tex, view, proj,
                                  W, H, out, zbuf, l, alpha)

    if depth_cue:
        out = _apply_depth_cue(out, zbuf)
    return out, zbuf


def _apply_depth_cue(img: np.ndarray, zbuf: np.ndarray) -> np.ndarray:
    """Fade geometry slightly with distance so it reads as 3D on flat bg."""
    out = img.astype(np.float32)
    m = np.isfinite(zbuf)
    if m.any():
        z = zbuf[m]
        lo, hi = np.percentile(z, 2), np.percentile(z, 98)
        t = np.clip((zbuf[m] - lo) / max(hi - lo, 1e-6), 0, 1)
        fade = (1.0 - 0.22 * t)[:, None]
        out[m, :3] = out[m, :3] * fade
    return out.clip(0, 255).astype(np.uint8)


def _render_numpy(vertices, normals, uvs, tri, tex, view, proj, W, H, out, zbuf, l, alpha):
    """Fallback: vectorized barycentric rasterizer (same math, ~100x slower)."""
    V = np.hstack([vertices, np.ones((len(vertices), 1))]).astype(np.float64)
    cam = V @ view.T
    z = -cam[:, 2]
    ok = z > 1e-4
    p = np.zeros((len(V), 2))
    p[ok, 0] = proj[0, 0] * cam[ok, 0] / z[ok] + proj[0, 2]
    p[ok, 1] = -proj[1, 1] * cam[ok, 1] / z[ok] + proj[1, 2]

    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    shade = 0.35 * np.abs(n @ l) + 0.65 * np.abs(n[:, 2])

    tw, th = tex.shape[1], tex.shape[0]
    for t in tri:
        a, b, c = t
        if not (ok[a] and ok[b] and ok[c]):
            continue
        tri_p = p[[a, b, c]]
        x0, y0 = np.floor(tri_p.min(0)).astype(int)
        x1, y1 = np.ceil(tri_p.max(0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        d = ((tri_p[1, 0] - tri_p[0, 0]) * (tri_p[2, 1] - tri_p[0, 1]) -
             (tri_p[1, 1] - tri_p[0, 1]) * (tri_p[2, 0] - tri_p[0, 0]))
        if abs(d) < 1e-9 or d >= 0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((tri_p[1, 0] - gx) * (tri_p[2, 1] - gy) - (tri_p[1, 1] - gy) * (tri_p[2, 0] - gx)) / d
        w1 = ((tri_p[2, 0] - gx) * (tri_p[0, 1] - gy) - (tri_p[2, 1] - gy) * (tri_p[0, 0] - gx)) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = 1.0 / (w0 / z[a] + w1 / z[b] + w2 / z[c])
        sub = zbuf[y0:y1, x0:x1]
        hit = inside & (zz < sub)
        if not hit.any():
            continue
        uw = (w0 * uvs[a, 0] / z[a] + w1 * uvs[b, 0] / z[b] + w2 * uvs[c, 0] / z[c]) * zz
        vw = (w0 * uvs[a, 1] / z[a] + w1 * uvs[b, 1] / z[b] + w2 * uvs[c, 1] / z[c]) * zz
        uu = ((uw[hit] % 1.0) * (tw - 1)).astype(int)
        vv = ((1 - vw[hit] % 1.0) * (th - 1)).astype(int)
        rgb = tex[vv, uu]
        sh = (w0 * shade[a] + w1 * shade[b] + w2 * shade[c])[hit]
        sub[hit] = zz[hit]
        out[y0:y1, x0:x1, :3][hit] = (rgb * sh[:, None]).clip(0, 255).astype(np.uint8)
        out[y0:y1, x0:x1, 3][hit] = 255
    return out, zbuf


def downscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter downscale (used for supersampling)."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // factor * factor, w // factor * factor
    a = img[:h2, :w2].astype(np.float32)
    a = a.reshape(h2 // factor, factor, w2 // factor, factor, -1).mean(axis=(1, 3))
    return a.clip(0, 255).astype(np.uint8)
