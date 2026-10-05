"""tk.tex - procedural, seamlessly tiling PBR texture synthesis in numpy (runs in Blender's Python).

All lengths are in METERS on the surface. A canvas `t` covers one tile of `t.tile = (W, H)` meters
with `t.res = (w, h)` pixels. Every operation wraps around, so results always tile.

A material generator is a file materials/<name>.py:

    from tk.tex import *
    META = {...}
    def build(t):
        ...
        return dict(albedo=rgb_linear, height=height_m, rough=r, metal=m)     # arrays or scalars

Arrays are float32, shape (h, w) or (h, w, 3). Image row 0 is the TOP of the texture (V = 1).
"""
import json
import math
import os
import struct
import subprocess
import zlib
import numpy as np

F = np.float32


# ----------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------

def sat(a):
    return np.clip(a, 0.0, 1.0)


def mix(a, b, t):
    """Linear blend; works for scalars, (h,w) and (h,w,3) (t may be (h,w) against colours)."""
    a = np.asarray(a, dtype=F)
    b = np.asarray(b, dtype=F)
    t = np.asarray(t, dtype=F)
    if t.ndim == 2 and (a.ndim == 3 or b.ndim == 3 or (a.ndim == 1 and a.shape[0] == 3) or (b.ndim == 1 and b.shape[0] == 3)):
        t = t[..., None]
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    d = np.asarray(e1 - e0, dtype=F)
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    t = np.clip((x - e0) / d, 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(F)


def remap(a, lo, hi, out_lo=0.0, out_hi=1.0, clip=True):
    t = (a - lo) / (hi - lo if hi != lo else 1e-9)
    if clip:
        t = np.clip(t, 0.0, 1.0)
    return (out_lo + (out_hi - out_lo) * t).astype(F)


def norm01(a):
    """Rescale to exactly 0..1."""
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo if hi > lo else 1.0)).astype(F)


def level(a, lo_pct=1.0, hi_pct=99.0):
    """Robust rescale to 0..1 using percentiles (ignores outliers)."""
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    return np.clip((a - lo) / (hi - lo if hi > lo else 1.0), 0, 1).astype(F)


def hexlin(h):
    """'#RRGGBB' sRGB -> linear RGB float32 array(3)."""
    if not isinstance(h, str):
        return np.asarray(h, dtype=F)
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(F)


def lin2srgb(a):
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)


def ramp(t, stops):
    """Colour gradient lookup. t: (h,w) in 0..1; stops: [(pos, '#hex' or (r,g,b) linear), ...].
    Interpolates in linear RGB. Returns (h,w,3). The main way to colour things: build a mask or
    noise and map it through 3-6 carefully chosen real-world colours."""
    stops = sorted(stops, key=lambda s: s[0])
    pos = np.array([s[0] for s in stops], dtype=np.float64)
    cols = np.stack([hexlin(s[1]) for s in stops]).astype(np.float64)
    out = np.empty(t.shape + (3,), dtype=F)
    for k in range(3):
        out[..., k] = np.interp(t, pos, cols[:, k])
    return out


def colour(c, shape=None):
    """Constant colour image from '#hex'."""
    c = hexlin(c)
    return c if shape is None else np.broadcast_to(c, tuple(shape) + (3,)).astype(F).copy()


def luminance(rgb):
    return (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(F)


def desaturate(rgb, amount):
    l = luminance(rgb)[..., None]
    a = amount[..., None] if isinstance(amount, np.ndarray) else amount
    return (rgb + (l - rgb) * a).astype(F)


def overlay_tint(rgb, tint, amount):
    """Multiply rgb towards a tint colour by `amount` (0..1 scalar or mask)."""
    tint = hexlin(tint)
    a = amount[..., None] if isinstance(amount, np.ndarray) else amount
    return (rgb * (1 + (tint - 1) * a)).astype(F)


# ----------------------------------------------------------------------------
# cell layouts
# ----------------------------------------------------------------------------

class Layout:
    """Result of a cell layout (bricks / stones / planks / tiles).
      .dist  distance (m) from each pixel to the nearest cell border (0 at the border, grows inwards)
      .id    integer cell id per pixel
      .u/.v  cell-local coordinates in meters, origin at the cell centre (u along the cell's length)
      .n     number of cells
      .rand(k) -> per-pixel array of a per-cell random value in 0..1 (k = channel: different k,
                different numbers) - use for per-stone colour / height / tilt variation
      .tilt(amount_m, k) -> per-cell random planar tilt height field (each stone sits a bit crooked)
    """

    def __init__(self, dist, cid, u, v, n, seed, cw=None, ch=None):
        self.dist, self.id, self.u, self.v, self.n, self.seed = dist.astype(F), cid, u.astype(F), v.astype(F), n, seed
        self.cw, self.ch = cw, ch

    def rand(self, k=0):
        r = np.random.default_rng(self.seed * 7919 + k * 104729 + 13).random(self.n + 1).astype(F)
        return r[self.id]

    def tilt(self, amount=0.003, k=0):
        ax = (self.rand(k + 101) - 0.5) * 2
        ay = (self.rand(k + 102) - 0.5) * 2
        s = max(float(np.percentile(np.abs(self.u), 95)), 1e-6)
        return ((ax * self.u + ay * self.v) / s * amount).astype(F)

    def mask(self, gap=0.004, soft=0.002):
        """1 inside cells, 0 in the joints (joint half-width = gap/2, edge softness = soft)."""
        return smoothstep(gap / 2, gap / 2 + soft, self.dist)

    def rounded(self, t, radius=0.01):
        """Round off the cells' sharp corners: returns a new Layout whose .dist has corners eased
        by about `radius` m (river cobbles, worn bricks, pillowy tiles). `t` is the canvas."""
        cap = radius * 2.5
        d = t.blur(np.minimum(self.dist, cap), radius * 0.6)
        # corners (where two borders are close) lose more distance than straight edges
        d2 = np.where(self.dist < cap, np.minimum(self.dist, d * 1.0 - radius * 0.18), self.dist)
        out = Layout(np.maximum(d2, 0.0), self.id, self.u, self.v, self.n, self.seed, self.cw, self.ch)
        for k in ("f1", "f2"):
            if hasattr(self, k):
                setattr(out, k, getattr(self, k))
        return out

    def dome(self, radius=0.02, power=2.0):
        """Rounded-edge profile 0..1: 0 at the border rising to 1 at `radius` m inside (pillowed
        stones / eased brick arrises). Multiply by a height in meters."""
        t = np.clip(self.dist / max(radius, 1e-6), 0, 1)
        return (1 - (1 - t) ** power).astype(F)


# ----------------------------------------------------------------------------
# Canvas
# ----------------------------------------------------------------------------

class Canvas:
    def __init__(self, res=1024, tile=1.0, seed=1):
        tile = (float(tile), float(tile)) if isinstance(tile, (int, float)) else (float(tile[0]), float(tile[1]))
        if isinstance(res, (int, float)):
            w = int(res)
            h = max(4, int(round(res * tile[1] / tile[0] / 4)) * 4)
            # keep power-of-two friendly sizes when the tile is square or 2:1
            res = (w, h)
        self.res = (int(res[0]), int(res[1]))
        self.tile = tile
        self.seed = int(seed)
        self.w, self.h = self.res
        self.W, self.H = tile
        self.px = self.W / self.w          # meters per pixel (x)
        self.py = self.H / self.h
        xs = (np.arange(self.w, dtype=F) + 0.5) * self.px
        ys = self.H - (np.arange(self.h, dtype=F) + 0.5) * self.py  # row 0 = top (V=1)
        self.X, self.Y = np.meshgrid(xs, ys)
        self._k = 0

    # ---- utilities
    def _seed(self, seed):
        if seed is None:
            self._k += 1
            return self.seed * 1000 + self._k
        return self.seed * 1000 + 500 + int(seed)

    def zeros(self, v=0.0):
        return np.full((self.h, self.w), v, dtype=F)

    def white(self, seed=None):
        """Uniform white noise 0..1 per pixel."""
        return np.random.default_rng(self._seed(seed)).random((self.h, self.w)).astype(F)

    def _freq(self):
        fx = np.fft.fftfreq(self.w, d=self.px)
        fy = np.fft.fftfreq(self.h, d=self.py)
        return np.meshgrid(fx, fy)

    # ---- noise
    def noise(self, scale=0.1, seed=None, stretch=(1.0, 1.0), bandwidth=0.6):
        """Smooth band-limited noise with features about `scale` meters across. Returns 0..1
        (mean 0.5, roughly gaussian). stretch=(sx, sy) elongates features: stretch=(8,1) gives
        streaks 8x longer along X (wood grain, brushed metal, drips with (1,8))."""
        rng = np.random.default_rng(self._seed(seed))
        wn = rng.standard_normal((self.h, self.w))
        FX, FY = self._freq()
        f = np.sqrt((FX * scale * stretch[0]) ** 2 + (FY * scale * stretch[1]) ** 2)
        # band-pass around f = 0.5 cycles per `scale`
        with np.errstate(divide='ignore'):
            lf = np.where(f > 0, np.log2(np.maximum(f, 1e-12) / 0.5), -99)
        filt = np.exp(-0.5 * (lf / bandwidth) ** 2)
        filt[0, 0] = 0
        out = np.fft.ifft2(np.fft.fft2(wn) * filt).real
        s = out.std()
        if s < 1e-12:
            return self.zeros(0.5)
        return np.clip(0.5 + out / s * 0.18, 0, 1).astype(F)

    def fbm(self, scale=0.2, octaves=5, gain=0.5, seed=None, stretch=(1.0, 1.0), lacunarity=2.0):
        """Fractal noise: sum of noise() octaves starting at `scale` m, each half the size.
        Returns 0..1 (re-levelled). gain 0.5 = natural; 0.7 = rough/gritty; 0.35 = soft."""
        sd = self._seed(seed)
        rng = np.random.default_rng(sd)
        wn = np.fft.fft2(rng.standard_normal((self.h, self.w)))
        FX, FY = self._freq()
        f = np.sqrt((FX * stretch[0]) ** 2 + (FY * stretch[1]) ** 2)
        filt = np.zeros_like(f)
        amp = 1.0
        s = scale
        for _ in range(octaves):
            with np.errstate(divide='ignore'):
                lf = np.where(f > 0, np.log2(np.maximum(f * s, 1e-12) / 0.5), -99)
            filt += amp * np.exp(-0.5 * (lf / 0.5) ** 2)
            amp *= gain
            s /= lacunarity
        filt[0, 0] = 0
        out = np.fft.ifft2(wn * filt).real
        return level(out, 0.5, 99.5)

    def ridged(self, scale=0.2, octaves=4, gain=0.5, seed=None, stretch=(1.0, 1.0)):
        """Ridged noise 0..1: sharp crests (veins, cracks when inverted, bark ridges)."""
        out = self.zeros()
        amp, tot, s = 1.0, 0.0, scale
        base = 0 if seed is None else seed * 31
        for o in range(octaves):
            n = self.noise(s, seed=(None if seed is None else base + o), stretch=stretch)
            out += amp * (1 - np.abs(n * 2 - 1))
            tot += amp
            amp *= gain
            s /= 2
        return level(out / tot, 0.5, 99.5)

    # ---- filters
    def blur(self, a, sigma, sigma_y=None):
        """Gaussian blur with sigma in meters (wraps). sigma_y for anisotropic blur."""
        FX, FY = self._freq()
        sy = sigma if sigma_y is None else sigma_y
        g = np.exp(-2 * (math.pi ** 2) * ((FX * sigma) ** 2 + (FY * sy) ** 2))
        if a.ndim == 3:
            return np.stack([np.fft.ifft2(np.fft.fft2(a[..., k]) * g).real for k in range(a.shape[2])], axis=-1).astype(F)
        return np.fft.ifft2(np.fft.fft2(a) * g).real.astype(F)

    def sample(self, a, X, Y):
        """Bilinear, wrapping lookup of image `a` at surface positions X, Y (meters)."""
        fx = X / self.px - 0.5
        fy = (self.H - Y) / self.py - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        tx = (fx - x0).astype(F)
        ty = (fy - y0).astype(F)
        x1 = (x0 + 1) % self.w
        y1 = (y0 + 1) % self.h
        x0 %= self.w
        y0 %= self.h
        if a.ndim == 3:
            tx, ty = tx[..., None], ty[..., None]
        return ((a[y0, x0] * (1 - tx) + a[y0, x1] * tx) * (1 - ty) + (a[y1, x0] * (1 - tx) + a[y1, x1] * tx) * ty).astype(F)

    def warp(self, a, amount=0.01, scale=0.1, seed=None):
        """Distort image `a` by a smooth random vector field (amount, scale in meters). Breaks up
        anything that looks too straight / mechanical."""
        s = self._seed(seed)
        dx = (self.noise(scale, seed=s % 9973 + 1) - 0.5) * 2 * amount * 2.7
        dy = (self.noise(scale, seed=s % 9973 + 2) - 0.5) * 2 * amount * 2.7
        return self.sample(a, self.X + dx, self.Y + dy)

    def shift(self, a, dx=0.0, dy=0.0):
        """Translate by (dx, dy) meters (wraps)."""
        return self.sample(a, self.X - dx, self.Y - dy)

    def slope(self, h):
        """Gradient (dh/dx, dh/dy) of a height field in m/m."""
        gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * self.px)
        gy = (np.roll(h, 1, axis=0) - np.roll(h, -1, axis=0)) / (2 * self.py)
        return gx.astype(F), gy.astype(F)

    def curv(self, h, radius=0.004):
        """Convexity of a height field: >0 on ridges / raised edges, <0 in pits / grooves
        (h - blur(h)), in meters. Use it to put wear on edges and dirt in cavities."""
        return (h - self.blur(h, radius)).astype(F)

    def cavity(self, h, radius=0.004, depth=0.001):
        """0..1 mask of recesses: 1 where the surface is `depth` m below its surroundings."""
        return sat(-self.curv(h, radius) / max(depth, 1e-9))

    def edges(self, h, radius=0.004, height=0.001):
        """0..1 mask of raised edges / high points (exposed to wear)."""
        return sat(self.curv(h, radius) / max(height, 1e-9))

    def ao(self, h, radius=0.01, strength=1.0):
        """Ambient occlusion 0..1 from a height field (multi-scale cavity estimate)."""
        occ = self.zeros()
        for r, wgt in ((radius * 0.35, 0.3), (radius, 0.4), (radius * 3.0, 0.3)):
            d = self.blur(h, r) - h
            occ += wgt * np.clip(d / max(r * 0.6, 1e-6), 0, 1.5)
        return sat(1.0 - occ * strength).astype(F)

    def normal(self, h, strength=1.0):
        """Tangent-space normal map (h,w,3) in -1..1 from a height field in METERS (physically
        correct slopes; strength exaggerates)."""
        gx, gy = self.slope(h)
        nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(gx)
        l = np.sqrt(nx * nx + ny * ny + nz * nz)
        return np.stack([nx / l, ny / l, nz / l], axis=-1).astype(F)

    # ---- patterns
    def stripes(self, period, axis='x', duty=0.5, soft=0.0):
        """Periodic bands 0/1 across `axis` ('x': bands vary along X). period is snapped so it tiles."""
        L = self.W if axis == 'x' else self.H
        n = max(1, round(L / period))
        c = (self.X if axis == 'x' else self.Y) / (L / n)
        f = c - np.floor(c)
        if soft <= 0:
            return (f < duty).astype(F)
        s = soft / (L / n)
        return (smoothstep(0, s, f) * (1 - smoothstep(duty, duty + s, f))).astype(F)

    def wave(self, period, axis='x', phase=0.0):
        """Sine 0..1 along axis with a period snapped to tile."""
        L = self.W if axis == 'x' else self.H
        n = max(1, round(L / period))
        c = (self.X if axis == 'x' else self.Y) / L * n
        return (0.5 + 0.5 * np.sin(2 * math.pi * (c + phase))).astype(F)

    def grid(self, cell_w, cell_h, shift=0.5, jitter_w=0.0, seed=None, shift_rows=True):
        """Brick / plank / tile layout: rows of cells `cell_w` x `cell_h` meters (snapped to tile),
        each row shifted by `shift` cells relative to the previous (0.5 = running bond, 0 = stack,
        'random' = random per row). jitter_w (0..0.9) randomises cell lengths within a row.
        With shift_rows=False the columns are shifted instead (vertical planks).  -> Layout.
        For joints: lay.mask(gap); the joint is at lay.dist == 0."""
        sd = self._seed(seed)
        rng = np.random.default_rng(sd)
        X, Y, W, H = (self.X, self.Y, self.W, self.H) if shift_rows else (self.Y, self.X, self.H, self.W)
        rows = max(1, round(H / cell_h))
        cols = max(1, round(W / cell_w))
        ch, cw = H / rows, W / cols
        r = np.floor(Y / ch).astype(np.int64) % rows
        if shift == 'random':
            offs = rng.random(rows) * cw
        else:
            offs = (np.arange(rows) * shift * cw) % cw
            if rows % 2 == 1 and shift == 0.5 and rows > 1:
                offs = rng.random(rows) * cw  # odd row count can't alternate seamlessly
        # per-row cell boundaries (cols+1 edges across W, wrapping)
        edges = np.zeros((rows, cols + 1))
        for i in range(rows):
            e = np.arange(cols + 1) * cw
            if jitter_w > 0:
                e[1:-1] += (rng.random(cols - 1) - 0.5) * cw * jitter_w
            edges[i] = e
        xs = (X - offs[r]) % W
        c = np.zeros(X.shape, dtype=np.int64)
        x0 = np.zeros(X.shape, dtype=F)
        x1 = np.zeros(X.shape, dtype=F)
        for i in range(rows):
            m = r == i
            ci = np.clip(np.searchsorted(edges[i], xs[m], side='right') - 1, 0, cols - 1)
            c[m] = ci
            x0[m] = edges[i][ci]
            x1[m] = edges[i][ci + 1]
        yb = Y % ch
        du = np.minimum(xs - x0, x1 - xs)
        dv = np.minimum(yb, ch - yb)
        dist = np.minimum(du, dv)
        u = xs - (x0 + x1) / 2
        v = yb - ch / 2
        cid = (r * cols + c).astype(np.int64)
        return Layout(dist, cid, u, v, rows * cols, sd, cw, ch)

    def cells(self, size=0.1, jitter=0.9, seed=None, aspect=1.0):
        """Voronoi cell layout: irregular stones / cobbles / cracks / scales about `size` meters
        across (aspect > 1 stretches cells along X). jitter 0 = regular grid, 1 = fully irregular.
        -> Layout with exact distance-to-border (.dist), ids, local coords (.u, .v) and extra
        fields .f1 (distance to the cell's seed point) for round bumps."""
        sd = self._seed(seed)
        rng = np.random.default_rng(sd)
        nx = max(2, round(self.W / (size * aspect)))
        ny = max(2, round(self.H / size))
        cw, ch = self.W / nx, self.H / ny
        px = (np.arange(nx)[None, :] + 0.5 + (rng.random((ny, nx)) - 0.5) * jitter) * cw
        py = (np.arange(ny)[:, None] + 0.5 + (rng.random((ny, nx)) - 0.5) * jitter) * ch
        gx = np.floor(self.X / cw).astype(np.int64)
        gy = np.floor(self.Y / ch).astype(np.int64)
        big = 1e9
        d1 = np.full(self.X.shape, big, dtype=F)
        d2 = np.full(self.X.shape, big, dtype=F)
        p1x = np.zeros(self.X.shape, dtype=F)
        p1y = np.zeros(self.X.shape, dtype=F)
        p2x = np.zeros(self.X.shape, dtype=F)
        p2y = np.zeros(self.X.shape, dtype=F)
        id1 = np.zeros(self.X.shape, dtype=np.int64)
        R = 2
        for oy in range(-R, R + 1):
            for ox in range(-R, R + 1):
                cx = gx + ox
                cy = gy + oy
                wx = cx % nx
                wy = cy % ny
                qx = (px[wy, wx] + (cx - wx) * cw).astype(F)
                qy = (py[wy, wx] + (cy - wy) * ch).astype(F)
                d = (qx - self.X) ** 2 + (qy - self.Y) ** 2
                closer = d < d1
                second = (~closer) & (d < d2)
                d2 = np.where(closer, d1, np.where(second, d, d2))
                p2x = np.where(closer, p1x, np.where(second, qx, p2x))
                p2y = np.where(closer, p1y, np.where(second, qy, p2y))
                p1x = np.where(closer, qx, p1x)
                p1y = np.where(closer, qy, p1y)
                id1 = np.where(closer, wy * nx + wx, id1)
                d1 = np.where(closer, d, d1)
        # exact distance to the bisector between nearest and second-nearest seeds
        mx, my = (p1x + p2x) / 2 - self.X, (p1y + p2y) / 2 - self.Y
        ex, ey = p2x - p1x, p2y - p1y
        el = np.sqrt(ex * ex + ey * ey) + 1e-9
        dist = np.abs(mx * ex + my * ey) / el
        lay = Layout(dist, id1, self.X - p1x, self.Y - p1y, nx * ny, sd, cw, ch)
        lay.f1 = np.sqrt(d1).astype(F)
        lay.f2 = np.sqrt(d2).astype(F)
        return lay

    def cracks(self, size=0.3, width=0.0015, seed=None, wander=0.02, density=1.0):
        """0..1 mask of thin wandering cracks (Voronoi borders, warped). density < 1 removes a
        fraction of the crack network so cracks start and stop."""
        lay = self.cells(size, 1.0, seed=seed)
        d = self.warp(lay.dist, wander, size * 0.25, seed=(seed or 0) + 77)
        m = 1 - smoothstep(width * 0.5, width * 1.5, d)
        if density < 1.0:
            gate = self.noise(size * 1.5, seed=(seed or 0) + 78)
            m = m * smoothstep(1 - density - 0.05, 1 - density + 0.05, level(gate))
        return m.astype(F)

    def speckle(self, density=2000.0, size=0.002, seed=None, soft=0.5):
        """Random dots: `density` per square meter, about `size` m across. 0..1 mask (pores, pits,
        grit, paint flecks, lichen spots with bigger size)."""
        lay = self.cells(max(1.0 / math.sqrt(max(density, 1e-6)), self.px * 3), 1.0, seed=seed)
        r = size / 2 * (0.5 + lay.rand(3))
        return (1 - smoothstep(r * (1 - soft), r, lay.f1)).astype(F)

    def scratches(self, density=60.0, length=0.05, width=0.0006, seed=None, angle=None, angle_spread=180.0):
        """0..1 mask of fine straight scratches (density per m^2, length/width in m).
        angle (deg, 0 = along X) with +-angle_spread/2 variation; default random directions."""
        rng = np.random.default_rng(self._seed(seed))
        n = int(density * self.W * self.H)
        out = self.zeros()
        n = min(n, 4000)
        for _ in range(n):
            cx, cy = rng.random() * self.W, rng.random() * self.H
            a = math.radians((angle if angle is not None else 0.0) + (rng.random() - 0.5) * angle_spread)
            L = length * (0.3 + rng.random() * 1.2)
            wd = max(width * (0.5 + rng.random()), self.px * 0.7)
            # rasterise into a local window
            hw = int((L / 2 + wd) / self.px) + 2
            hh = int((L / 2 + wd) / self.py) + 2
            ix, iy = int(cx / self.px), int((self.H - cy) / self.py)
            xs = np.arange(ix - hw, ix + hw + 1)
            ys = np.arange(iy - hh, iy + hh + 1)
            Xl = (xs[None, :] + 0.5) * self.px - cx
            Yl = (self.H - (ys[:, None] + 0.5) * self.py) - cy
            al = Xl * math.cos(a) + Yl * math.sin(a)
            ac = -Xl * math.sin(a) + Yl * math.cos(a)
            m = np.clip(1 - np.abs(ac) / wd, 0, 1) * np.clip((L / 2 - np.abs(al)) / (L * 0.15 + 1e-9), 0, 1)
            m *= 0.4 + 0.6 * rng.random()
            yy = ys % self.h
            xx = xs % self.w
            out[np.ix_(yy, xx)] = np.maximum(out[np.ix_(yy, xx)], m.astype(F))
        return out

    def drips(self, scale=0.05, length=8.0, seed=None):
        """Vertical streak noise 0..1 (rain runs, rust runs, soot washes): noise stretched along Y."""
        return self.fbm(scale, 4, 0.55, seed=seed, stretch=(1.0, 1.0 / length))

    def fibres(self, scale=0.004, length=20.0, seed=None, axis='x'):
        """Fine directional fibre / grain noise 0..1 running along `axis`."""
        st = (1.0 / length, 1.0) if axis == 'x' else (1.0, 1.0 / length)
        return self.fbm(scale, 3, 0.6, seed=seed, stretch=st)


# ----------------------------------------------------------------------------
# PNG / preview / save
# ----------------------------------------------------------------------------

def write_png(path, img):
    """img: uint8 (h,w), (h,w,3) or (h,w,4)."""
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        img = img[..., None]
    h, w, c = img.shape
    ct = {1: 0, 3: 2, 4: 6}[c]
    raw = np.zeros((h, w * c + 1), dtype=np.uint8)
    raw[:, 1:] = img.reshape(h, w * c)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw.tobytes(), 3)))
        f.write(chunk(b"IEND", b""))


def _u8(a):
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _full(t, v, ch=None):
    a = np.asarray(v, dtype=F)
    shape = (t.h, t.w) if ch is None else (t.h, t.w, ch)
    if a.shape != shape:
        a = np.broadcast_to(a, shape).astype(F)
    return a


def _down(a, f):
    if f <= 1:
        return a
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    a = a[:h, :w]
    if a.ndim == 3:
        return a.reshape(h // f, f, w // f, f, a.shape[2]).mean(axis=(1, 3))
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _shade(alb, n, r, m, a, L, V, light, amb):
    """GGX shade. alb (..,3), n (..,3), r/m/a (..), L/V (..,3) unit, light (..,3) radiance, amb (3,)."""
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    Hh = L + V
    Hh = Hh / np.maximum(np.linalg.norm(Hh, axis=-1, keepdims=True), 1e-6)
    ndl = np.clip((n * L).sum(-1), 0, 1)
    ndv = np.clip((n * V).sum(-1), 1e-3, 1)
    ndh = np.clip((n * Hh).sum(-1), 0, 1)
    vdh = np.clip((V * Hh).sum(-1), 0, 1)
    r = np.clip(r, 0.04, 1.0)
    al = r * r
    D = al * al / (math.pi * ((ndh * ndh) * (al * al - 1) + 1) ** 2 + 1e-9)
    k = (r + 1) ** 2 / 8
    G = (ndl / (ndl * (1 - k) + k + 1e-9)) * (ndv / (ndv * (1 - k) + k))
    F0 = 0.04 * (1 - m[..., None]) + alb * m[..., None]
    Fr = F0 + (1 - F0) * ((1 - vdh) ** 5)[..., None]
    spec = Fr * (D * G / (4 * ndl * ndv + 1e-3))[..., None]
    diff = alb * (1 - m[..., None]) / math.pi
    col = (diff + spec) * ndl[..., None] * light + (alb * (1 - m[..., None]) + F0 * 0.6) * amb * a[..., None] * (0.6 + 0.4 * n[..., 2:3])
    col = col / (1 + col * 0.35)
    return _u8(lin2srgb(col))


def render_preview(t, albedo, nrm, rough, metal, ao, reps=2, out_w=1024, light=(-0.55, 0.45, 0.42)):
    """Simple lit preview of the tile repeated reps x reps (raking light shows the relief)."""
    f = max(1, (t.w * reps) // out_w)
    alb = np.tile(_down(albedo, f), (reps, reps, 1))
    n = np.tile(_down(nrm, f), (reps, reps, 1))
    r = np.tile(_down(rough, f), (reps, reps))
    m = np.tile(_down(metal, f), (reps, reps))
    a = np.tile(_down(ao, f), (reps, reps))
    L = np.array(light, dtype=F)
    L /= np.linalg.norm(L)
    V = np.array([0.0, -0.35, 0.94], dtype=F)
    V /= np.linalg.norm(V)
    sun = np.array([1.0, 0.95, 0.88], dtype=F) * 3.3
    sky = np.array([0.55, 0.65, 0.8], dtype=F) * 0.55
    return _shade(alb, n, r, m, a, np.broadcast_to(L, n.shape), np.broadcast_to(V, n.shape), sun, sky)


def render_far(t, albedo, nrm, rough, metal, ao, out_w=1024, out_h=576):
    """Perspective view across ~7 repeats of the tile lit by one warm lantern over a dim blue night
    ambient: shows repetition, macro read and how the relief/gloss catch a low light (the town look)."""
    S = float(max(t.W, t.H))
    maps = [albedo, nrm, rough[..., None], metal[..., None], ao[..., None]]
    pyr = [[mp] for mp in maps]
    for _ in range(5):
        for pl in pyr:
            if min(pl[-1].shape[:2]) >= 8:
                pl.append(_down(pl[-1], 2))
    eye = np.array([0.5 * S, -0.9 * S, 0.85 * S], dtype=F)
    tgt = np.array([0.5 * S, 1.7 * S, 0.0], dtype=F)
    fw = tgt - eye
    fw /= np.linalg.norm(fw)
    rt = np.cross(fw, [0, 0, 1])
    rt /= np.linalg.norm(rt)
    up = np.cross(rt, fw)
    fl = 1.15
    xs = (np.arange(out_w, dtype=F) + 0.5) / out_w * 2 - 1
    ys = (1 - (np.arange(out_h, dtype=F) + 0.5) / out_h * 2) * out_h / out_w
    PX, PY = np.meshgrid(xs, ys)
    d = fw[None, None, :] * fl + rt[None, None, :] * PX[..., None] + up[None, None, :] * PY[..., None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    hit = d[..., 2] < -1e-3
    tt = np.where(hit, -eye[2] / np.where(hit, d[..., 2], -1.0), 0.0)
    P = eye[None, None, :] + d * tt[..., None]
    # pixel footprint in texels -> mip level
    foot = tt / fl * (2.0 / out_w) / np.maximum(-d[..., 2], 0.05) * (t.w / t.W)
    lod = np.clip(np.floor(np.log2(np.maximum(foot, 1.0)) + 0.5).astype(int), 0, len(pyr[0]) - 1)
    out = [np.zeros((out_h, out_w, mp.shape[2]), dtype=F) for mp in maps]
    U = P[..., 0] / t.W
    Vv = P[..., 1] / t.H
    for l in range(len(pyr[0])):
        sel = hit & (lod == l)
        if not sel.any():
            continue
        hh, ww = pyr[0][l].shape[:2]
        ix = np.floor((U[sel] % 1.0) * ww).astype(int) % ww
        iy = np.floor((1.0 - (Vv[sel] % 1.0)) * hh).astype(int) % hh
        for o, pl in zip(out, pyr):
            o[sel] = pl[l][iy, ix]
    alb, n, r, m, a = out[0], out[1], out[2][..., 0], out[3][..., 0], out[4][..., 0]
    n = np.where(hit[..., None], n, np.array([0, 0, 1], dtype=F))
    lp = np.array([0.15 * S, 1.6 * S, 0.75 * S], dtype=F)
    Lv = lp[None, None, :] - P
    dist = np.linalg.norm(Lv, axis=-1, keepdims=True)
    Lv = Lv / np.maximum(dist, 1e-6)
    att = (S * S) / np.maximum(dist * dist, 1e-6)
    lamp = np.array([1.0, 0.62, 0.30], dtype=F) * 5.5 * att
    moon = np.array([0.30, 0.40, 0.62], dtype=F) * 0.22
    img = _shade(alb, n, r, m, a, Lv, -d, lamp, moon)
    img[~hit] = (8, 10, 16)
    return img


def save_material(out_dir, name, t, maps, meta):
    """Write albedo.png (sRGB), normal.png (RGB normal, A = height 0..1), orm.png (AO, rough, metal),
    preview.png, sheet.png and mat.json. Returns the mat.json dict."""
    os.makedirs(out_dir, exist_ok=True)
    albedo = np.clip(_full(t, maps["albedo"], 3), 0, 1)
    height = _full(t, maps.get("height", 0.0))
    rough = np.clip(_full(t, maps.get("rough", 0.7)), 0.03, 1)
    metal = np.clip(_full(t, maps.get("metal", 0.0)), 0, 1)
    nrm = t.normal(height, float(meta.get("normal_strength", 1.0)))
    hrange = float(height.max() - height.min())
    if maps.get("ao") is not None:
        ao = np.clip(_full(t, maps["ao"]), 0, 1)
    else:
        ao = t.ao(height, radius=float(meta.get("ao_radius", max(hrange * 2.0, 0.003))), strength=float(meta.get("ao_strength", 1.0)))
    h01 = (height - height.min()) / (hrange if hrange > 1e-9 else 1.0)
    write_png(os.path.join(out_dir, "albedo.png"), _u8(lin2srgb(albedo)))
    write_png(os.path.join(out_dir, "normal.png"), np.concatenate([_u8(nrm * 0.5 + 0.5), _u8(h01)[..., None]], axis=-1))
    write_png(os.path.join(out_dir, "orm.png"), _u8(np.stack([ao, rough, metal], axis=-1)))
    write_png(os.path.join(out_dir, "preview.png"), render_preview(t, albedo, nrm, rough, metal, ao))
    write_png(os.path.join(out_dir, "far.png"), render_far(t, albedo, nrm, rough, metal, ao))
    # channel sheet: albedo | normal | roughness | height | ao
    f = max(1, t.w // 512)
    tiles = [_u8(lin2srgb(_down(albedo, f))), _u8(_down(nrm, f) * 0.5 + 0.5)]
    for g in (rough, h01, ao):
        gg = _u8(_down(g, f))
        tiles.append(np.stack([gg, gg, gg], axis=-1))
    write_png(os.path.join(out_dir, "sheet.png"), np.concatenate(tiles, axis=1))
    mean = albedo.reshape(-1, 3).mean(0)
    info = dict(meta)
    info.update({
        "name": name, "tile": [t.W, t.H], "res": [t.w, t.h], "height_range": round(hrange, 5),
        "preview": [round(float(c), 4) for c in mean],
        "mean_rough": round(float(rough.mean()), 3), "mean_metal": round(float(metal.mean()), 3),
        "maps": {"albedo": "albedo.webp", "normal": "normal.webp", "orm": "orm.webp"},
    })
    # webp for the runtime (png kept for inspection)
    for nm, args in (("albedo", ["-quality", "92"]), ("normal", ["-quality", "96", "-define", "webp:alpha-quality=95"]),
                     ("orm", ["-quality", "92"])):
        src = os.path.join(out_dir, nm + ".png")
        try:
            subprocess.run(["magick", src, "-define", "webp:method=4", "-define", "webp:exact=true"] + args + [os.path.join(out_dir, nm + ".webp")],
                           check=True, capture_output=True, timeout=120)
        except Exception as e:  # pragma: no cover
            info["maps"][nm] = nm + ".png"
            print("WARN webp conversion failed:", e)
    with open(os.path.join(out_dir, "mat.json"), "w") as fjson:
        json.dump(info, fjson, indent=1)
    return info
