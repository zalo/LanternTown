"""Fine packed gravel in dirt: 3-25 mm mixed stone trodden into compacted earth.

Construction: a compacted, gently uneven earth surface; then thousands of individual stones
(sub-angular ellipsoids with a few fracture facets, three size classes) are dropped in and pressed
down so mostly only their caps stand a few mm proud of the fines.  Where the stones thin out the
dirt shows as smooth, slightly darker patches with sand grains.  Every stone has its own rock
colour under a common film of dust; flinty / quartz pebbles are smoother than the rest.
"""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.008, "ao_strength": 1.0}


def _dots(t, density, size, seed, soft=0.6):
    """Fast speckle: one dot per jittered grid cell (3x3 search).  0..1 mask, plus a per-dot random."""
    cell = max(1.0 / np.sqrt(density), t.px * 3)
    nx, ny = max(2, int(round(t.W / cell))), max(2, int(round(t.H / cell)))
    cw, ch = t.W / nx, t.H / ny
    rng = np.random.default_rng(seed)
    jx, jy, rr = rng.random((ny, nx)).astype(F), rng.random((ny, nx)).astype(F), rng.random((ny, nx)).astype(F)
    gx = np.floor(t.X / cw).astype(np.int64)
    gy = np.floor(t.Y / ch).astype(np.int64)
    out = np.zeros(t.X.shape, dtype=F)
    rv = np.zeros(t.X.shape, dtype=F)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = gx + ox, gy + oy
            wx, wy = cx % nx, cy % ny
            dx = (cx + jx[wy, wx]) * cw - t.X
            dy = (cy + jy[wy, wx]) * ch - t.Y
            r = size * 0.5 * (0.5 + rr[wy, wx])
            m = 1 - smoothstep(r * (1 - soft), r, np.sqrt(dx * dx + dy * dy))
            upd = m > out
            out = np.where(upd, m, out)
            rv = np.where(upd, rr[wy, wx], rv)
    return out, rv


def _drop_stones(t, H, ID, Q, stones, id0):
    """Press stones into the height field H (max-composite).  stones: rows of
    (cx, cy, a, b, ang, c, top, pw, f1x, f1y, f1h, lump, lph).  Writes the owner id into ID and the
    normalised radial position (0 centre .. 1 outline) into Q."""
    px, py, w, h = t.px, t.py, t.w, t.h
    for k in range(stones.shape[0]):
        cx, cy, a, b, ang, c, top, pw, fx, fy, fh, lump, lph = stones[k]
        rad = max(a, b) * (1.0 + lump) + px
        nxp, nyp = int(rad / px) + 1, int(rad / py) + 1
        ix, iy = int(cx / px), int((t.H - cy) / py)
        xs = np.arange(ix - nxp, ix + nxp + 1)
        ys = np.arange(iy - nyp, iy + nyp + 1)
        X = (xs[None, :] + 0.5) * px - cx
        Y = (t.H - (ys[:, None] + 0.5) * py) - cy
        ca, sa = np.cos(ang), np.sin(ang)
        xl = (X * ca + Y * sa) / a
        yl = (-X * sa + Y * ca) / b
        th = np.arctan2(yl, xl)
        rho = 1.0 + lump * (np.cos(3 * th + lph) + 0.6 * np.cos(5 * th + 2.3 * lph) + 0.5 * np.cos(2 * th + 4.1 * lph))
        q = (xl * xl + yl * yl) / (rho * rho)
        m = q < 1.0
        if not m.any():
            continue
        z = np.sqrt(np.clip(1.0 - q, 0.0, 1.0)) ** pw                 # 1 at the centre .. 0 at the outline
        z = np.minimum(z, fh + fx * xl + fy * yl)                      # fracture facet
        z = top - c + c * z
        yy, xx = ys % h, xs % w
        sub = H[np.ix_(yy, xx)]
        upd = m & (z > sub)
        if not upd.any():
            continue
        H[np.ix_(yy, xx)] = np.where(upd, z, sub)
        sid = ID[np.ix_(yy, xx)]
        ID[np.ix_(yy, xx)] = np.where(upd, id0 + k, sid)
        sq = Q[np.ix_(yy, xx)]
        Q[np.ix_(yy, xx)] = np.where(upd, q, sq)
    return id0 + stones.shape[0]


def build(t):
    rng = np.random.default_rng(1010)

    # ------------------------------------------------------------------ compacted earth
    macro = t.fbm(0.55, 3, 0.5, seed=11)                               # gravel-rich vs. dirt-rich patches
    und = (t.noise(0.26, seed=12) - 0.5) * 0.008 + (t.noise(0.11, seed=17) - 0.5) * 0.003                 # trodden unevenness
    clod = (t.fbm(0.035, 4, 0.55, seed=13) - 0.5) * 0.0030
    fine = t.fbm(0.004, 3, 0.65, seed=14)
    sandg, sand_r = _dots(t, 90000, 0.0030, 15, 0.7)                   # coarse sand grains in the fines
    dirt_h = und + clod + (fine - 0.5) * 0.0007 + sandg * 0.0007

    # ------------------------------------------------------------------ stones
    thin = smoothstep(-0.0024, 0.0004, und + (t.fbm(0.08, 3, 0.55, seed=16) - 0.5) * 0.0030)   # fines wash into the hollows                                # 0 = bare dirt patch, 1 = full gravel

    def make(n, r_lo, r_hi, proud, keep_lo):
        cx, cy = rng.random(n) * t.W, rng.random(n) * t.H
        loc = thin[np.clip(((t.H - cy) / t.py).astype(int), 0, t.h - 1), np.clip((cx / t.px).astype(int), 0, t.w - 1)]
        keep = rng.random(n) < (keep_lo + (1 - keep_lo) * loc)
        r = r_lo * (r_hi / r_lo) ** (rng.random(n) ** 1.6)              # more small than large
        asp = 1.0 + 0.7 * rng.random(n) ** 1.5
        a, b = r * np.sqrt(asp), r / np.sqrt(asp)
        ang = rng.random(n) * np.pi
        c = b * (0.55 + 0.35 * rng.random(n))
        base = und[np.clip(((t.H - cy) / t.py).astype(int), 0, t.h - 1), np.clip((cx / t.px).astype(int), 0, t.w - 1)]
        top = base + r * proud * (0.25 + 0.75 * rng.random(n)) - 0.0004
        pw = 0.75 + 0.5 * rng.random(n)
        fa = rng.random(n) * 2 * np.pi
        fs = 0.25 + 0.5 * rng.random(n)
        fh = np.where(rng.random(n) < 0.65, 0.55 + 0.35 * rng.random(n), 9.0)   # 65 % carry a broken facet
        lump = 0.03 + 0.07 * rng.random(n)
        st = np.stack([cx, cy, a, b, ang, c, top, pw, fs * np.cos(fa), fs * np.sin(fa), fh, lump, rng.random(n) * 6.28], axis=1)
        return st[keep]

    H = dirt_h.astype(np.float64).copy()
    ID = np.zeros(H.shape, dtype=np.int64)
    Q = np.ones(H.shape, dtype=np.float64)
    A = t.W * t.H
    groups = [make(int(15000 * A), 0.0020, 0.0042, 0.80, 0.42),          # 4-8 mm pea gravel
              make(int(6000 * A), 0.0042, 0.0090, 0.85, 0.28),           # 8-17 mm
              make(int(620 * A), 0.0090, 0.0155, 0.75, 0.30)]            # 17-27 mm
    n = 1
    for g in groups:
        n = _drop_stones(t, H, ID, Q, g, n)
    H = H.astype(F)
    Q = Q.astype(F)
    stone = (ID > 0).astype(F)
    rise = sat((H - dirt_h) / 0.004)                                    # how far the cap stands out of the fines

    # stone surface: fine pitting, more on the rougher rock types
    pr = np.random.default_rng(77)
    tab = pr.random((n + 1, 6)).astype(F)
    tab[0] = 0.5
    s_tone, s_hue, s_gl, s_dust, s_mot, s_pat = [tab[ID, i] for i in range(6)]
    mott = t.fbm(0.006, 3, 0.6, seed=21)
    height = H + stone * (mott - 0.5) * 0.00035 * (1 - s_gl)
    # fines bank up around each stone
    soft = t.blur(stone, 0.0022)
    height = height + (1 - stone) * soft * 0.0010

    # ------------------------------------------------------------------ colour
    dirtc = ramp(fine, [(0.0, "#40372c"), (0.5, "#584d41"), (1.0, "#716555")])
    dirtc = dirtc * (0.86 + 0.28 * t.fbm(0.09, 3, 0.5, seed=22))[..., None]
    grainc = ramp(sand_r, [(0, "#4b4740"), (0.4, "#847b6b"), (0.8, "#a39a88"), (1, "#b9b2a3")])
    dirtc = mix(dirtc, grainc, sandg[..., None] * 0.65)
    damp = smoothstep(0.003, -0.003, und) * 0.10 + (1 - thin) * 0.04     # hollows and bare patches hold moisture
    dirtc = dirtc * (1 - damp)[..., None]

    # mixed local gravel: limestone greys, buff sandstone, brown flint / ironstone, dark chert, pale quartz
    rock = ramp(s_hue, [(0.00, "#8b867b"), (0.22, "#9a927f"), (0.36, "#a39378"), (0.50, "#7a766e"), (0.62, "#85705a"),
                        (0.72, "#6b5a4a"), (0.80, "#5a5853"), (0.90, "#b0a999"), (0.96, "#8e7362"), (1.00, "#9c968a")])
    rock = rock * (0.70 + 0.60 * s_tone)[..., None]
    rock = rock * (0.90 + 0.20 * mott * (0.4 + s_mot))[..., None]
    band = 0.5 + 0.5 * np.sin((t.X * np.cos(s_pat * 6.28) + t.Y * np.sin(s_pat * 6.28)) * (700 + 900 * s_mot))
    rock = rock * (1 - 0.10 * band * smoothstep(0.75, 0.95, s_pat))[..., None]      # a few banded pebbles
    dust = hexlin("#7c705e")
    # dust film: thick near the fines, thin on the exposed cap; some stones cleaner than others
    film = sat(0.30 + 0.45 * s_dust + 0.45 * smoothstep(0.35, 1.0, Q) - 0.20 * rise)
    rock = mix(rock, dust * (0.85 + 0.3 * fine)[..., None], film[..., None] * 0.48)
    albedo = mix(dirtc, rock, stone[..., None])
    # shadowed dirt in the crevices between stones (dirt, not lighting)
    cav = t.cavity(height, 0.004, 0.0012)
    albedo = albedo * (1 - 0.28 * cav)[..., None]
    albedo = albedo * (0.96 + 0.08 * macro)[..., None]

    # ------------------------------------------------------------------ roughness
    r_st = 0.86 - 0.34 * smoothstep(0.72, 0.95, s_gl) * (1 - film * 0.7) - 0.08 * rise + 0.08 * (mott - 0.5)
    rough = mix(0.95 + 0.04 * (fine - 0.5) - 0.05 * damp / 0.2, r_st, stone)
    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0)
