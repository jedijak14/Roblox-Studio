/*
 * tiny_raster.c — small, dependency-free software rasterizer for skinned meshes.
 *
 * Built and loaded by raster.py through ctypes. No OpenGL, no GPU, no libGL:
 * the sandbox has none (and no apt to install them).
 *
 * Pipeline per frame:
 *   model-space verts -> world -> camera (view) -> perspective divide -> viewport
 *   backface cull -> scanline raster with 1/z interpolation -> z-buffer
 *   bilinear texture lookup -> 2-light Lambert shading -> RGBA out
 *
 * Compile:  gcc -O3 -march=native -fPIC -shared tiny_raster.c -o libtiny_raster.so -lm
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    float *vbuf;      /* depth (1/z) buffer, W*H */
    uint8_t *cbuf;    /* rgba, W*H*4 */
    int W, H;
} target_t;

/* 4x4 row-major matrix * vec4 */
static inline void m4v(const float *m, const float *v, float *out) {
    out[0] = m[0]*v[0] + m[1]*v[1] + m[2]*v[2]  + m[3]*v[3];
    out[1] = m[4]*v[0] + m[5]*v[1] + m[6]*v[2]  + m[7]*v[3];
    out[2] = m[8]*v[0] + m[9]*v[1] + m[10]*v[2] + m[11]*v[3];
    out[3] = m[12]*v[0] + m[13]*v[1] + m[14]*v[2] + m[15]*v[3];
}

static inline void sample_tex(const uint8_t *tex, int tw, int th, float u, float v,
                              float *rgb) {
    if (!tex) { rgb[0] = rgb[1] = rgb[2] = 1.0f; return; }
    /* glTF UV origin is bottom-left; our image rows go top-down */
    u = u - floorf(u);
    v = 1.0f - (v - floorf(v));
    float x = u * (tw - 1), y = v * (th - 1);
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 + 1 < tw ? x0 + 1 : tw - 1;
    int y1 = y0 + 1 < th ? y0 + 1 : th - 1;
    float fx = x - x0, fy = y - y0;
    const uint8_t *p00 = tex + (y0 * tw + x0) * 3;
    const uint8_t *p10 = tex + (y0 * tw + x1) * 3;
    const uint8_t *p01 = tex + (y1 * tw + x0) * 3;
    const uint8_t *p11 = tex + (y1 * tw + x1) * 3;
    for (int c = 0; c < 3; c++) {
        float a = p00[c] * (1 - fx) + p10[c] * fx;
        float b = p01[c] * (1 - fx) + p11[c] * fx;
        rgb[c] = (a * (1 - fy) + b * fy) / 255.0f;
    }
}

/*
 * Draw one triangle. screen coords in px, depth = view-space -z (metres, >0).
 * Returns 1 if any pixel was written.
 */
static inline int triangle(target_t *t,
                           const float *p0, const float *p1, const float *p2,
                           const float *d0, const float *d1, const float *d2,
                           const float *uv0, const float *uv1, const float *uv2,
                           const float *c0, const float *c1, const float *c2,
                           const uint8_t *tex, int tw, int th,
                           float alpha) {
    float minx = fminf(p0[0], fminf(p1[0], p2[0]));
    float maxx = fmaxf(p0[0], fmaxf(p1[0], p2[0]));
    float miny = fminf(p0[1], fminf(p1[1], p2[1]));
    float maxy = fmaxf(p0[1], fmaxf(p1[1], p2[1]));
    int x0 = (int)floorf(minx), x1 = (int)ceilf(maxx);
    int y0 = (int)floorf(miny), y1 = (int)ceilf(maxy);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > t->W - 1) x1 = t->W - 1;
    if (y1 > t->H - 1) y1 = t->H - 1;
    if (x1 < x0 || y1 < y0) return 0;

    float area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
    if (fabsf(area) < 1e-9f) return 0;
    float inv = 1.0f / area;
    int wrote = 0;

    for (int y = y0; y <= y1; y++) {
        float py = y + 0.5f;
        for (int x = x0; x <= x1; x++) {
            float px = x + 0.5f;
            float w0 = ((p1[0] - px) * (p2[1] - py) - (p1[1] - py) * (p2[0] - px)) * inv;
            if (w0 < 0.0f) continue;
            float w1 = ((p2[0] - px) * (p0[1] - py) - (p2[1] - py) * (p0[0] - px)) * inv;
            if (w1 < 0.0f) continue;
            float w2 = 1.0f - w0 - w1;
            if (w2 < 0.0f) continue;

            /* perspective-correct interpolation via reciprocal depth */
            float iz = w0 / d0[0] + w1 / d1[0] + w2 / d2[0];
            if (iz <= 0.0f) continue;
            float z = 1.0f / iz;
            int idx = y * t->W + x;
            if (z >= t->vbuf[idx]) continue;     /* smaller z == nearer */

            float uw = (w0 * uv0[0] / d0[0] + w1 * uv1[0] / d1[0] + w2 * uv2[0] / d2[0]) * z;
            float vw = (w0 * uv0[1] / d0[0] + w1 * uv1[1] / d1[0] + w2 * uv2[1] / d2[0]) * z;
            float rgb[3];
            sample_tex(tex, tw, th, uw, vw, rgb);

            float sh = (w0 * c0[0] + w1 * c1[0] + w2 * c2[0]) * 0.35f
                     + (w0 * c0[1] + w1 * c1[1] + w2 * c2[1]) * 0.65f;
            if (sh > 1.35f) sh = 1.35f;
            if (sh < 0.0f) sh = 0.0f;

            uint8_t *dst = t->cbuf + idx * 4;
            float a = alpha;
            for (int c = 0; c < 3; c++) {
                float v = rgb[c] * sh;
                if (v > 1.0f) v = 1.0f;
                uint8_t s = (uint8_t)(v * 255.0f + 0.5f);
                if (a >= 0.999f) dst[c] = s;
                else dst[c] = (uint8_t)(dst[c] * (1.0f - a) + s * a);
            }
            dst[3] = 255;             /* coverage: geometry written here */
            t->vbuf[idx] = z;
            wrote = 1;
        }
    }
    return wrote;
}

/*
 * Main entry point.
 *   verts     (nv*3)  world-space positions
 *   norms     (nv*3)  world-space normals
 *   uvs       (nv*2)
 *   tris      (nt*3)  indices
 *   tex       (tw*th*3) uint8 rgb, may be NULL for flat colour
 *   view      4x4 row-major world->camera (camera looks down -z)
 *   proj      3x3 row-major intrinsics (fx, 0, cx / 0, fy, cy / 0,0,1)
 *   out       (W*H*4) rgba, cleared by caller to background
 *   zbuf      (W*H)   float, cleared to +inf by caller
 *   light_dir3 normalised dir TOWARDS the key light
 */
void rasterize(const float *verts, const float *norms, const float *uvs,
               const int32_t *tris, int nt,
               const uint8_t *tex, int tw, int th,
               const float *view, const float *proj,
               int W, int H,
               uint8_t *out, float *zbuf,
               const float *light_dir, float alpha, int cull) {
    target_t t;
    t.vbuf = zbuf; t.cbuf = out; t.W = W; t.H = H;

    float sc[4], lc[3] = {light_dir[0], light_dir[1], light_dir[2]};

    for (int i = 0; i < nt; i++) {
        int ia = tris[i*3], ib = tris[i*3+1], ic = tris[i*3+2];
        float v[3][3], p[3][2], d[3][2], uv[3][2], col[3][2];

        int skip = 0;
        for (int k = 0; k < 3; k++) {
            int vi = (k == 0) ? ia : (k == 1) ? ib : ic;
            float in[4] = {verts[vi*3], verts[vi*3+1], verts[vi*3+2], 1.0f};
            m4v(view, in, sc);
            if (sc[2] > -1e-4f) { skip = 1; break; }   /* behind camera */
            float zc = -sc[2];
            d[k][0] = zc;
            p[k][0] = proj[0] * sc[0] / zc + proj[2];
            /* camera +y is up, image +y is down -> negate */
            p[k][1] = -proj[4] * sc[1] / zc + proj[5];
            uv[k][0] = uvs[vi*2]; uv[k][1] = uvs[vi*2+1];
            float n[3] = {norms[vi*3], norms[vi*3+1], norms[vi*3+2]};
            float nl = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]) + 1e-9f;
            n[0] /= nl; n[1] /= nl; n[2] /= nl;
            float key = fabsf(n[0]*lc[0] + n[1]*lc[1] + n[2]*lc[2]);
            float fill = fabsf(n[2]);
            col[k][0] = key;
            col[k][1] = fill;
        }
        if (skip) continue;

        /* screen-space winding: cull backfaces (CCW front in glTF) */
        float a2 = (p[1][0]-p[0][0]) * (p[2][1]-p[0][1]) - (p[1][1]-p[0][1]) * (p[2][0]-p[0][0]);
        if (cull && a2 >= 0.0f) continue;

        triangle(&t, p[0], p[1], p[2], d[0], d[1], d[2],
                 uv[0], uv[1], uv[2], col[0], col[1], col[2],
                 tex, tw, th, alpha);
    }
}
