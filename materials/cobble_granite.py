"""Rounded grey-buff granite cobbles laid in irregular courses, bedded in sand.

Construction: every cobble is an individual rounded block (super-ellipsoid) set in a wandering
course, each with its own size, proportions, rotation, crown height and tilt.  Sand is brushed in
between and stands a little below the crowns, so the visible joint is where the stone's flank dives
under the sand (narrow where two flanks meet, a little sand triangle where three or four stones
meet).  Crowns are foot-polished: smoother, a touch cleaner; flanks keep sandy dust.
"""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.022, "ao_strength": 0.9}


def _courses(t, X, Y, rows, mean_w, jit_w, seed):
    """Rows of stones.  rows: course heights (rescaled to the tile), widths random per stone.
    Returns per-pixel (id, u, v, hx, hy, n): cell id, cell-local coords (m), cell half sizes (m)."""
    rng = np.random.default_rng(seed)
    W, H = t.W, t.H
    rows = np.asarray(rows, dtype=np.float64)
    rows = rows * (H / rows.sum())
    ye = np.concatenate([[0.0], np.cumsum(rows)])
    Ym = np.mod(Y, H)
    Xm = np.mod(X, W)
    r = np.clip(np.searchsorted(ye, Ym, side="right") - 1, 0, len(rows) - 1)
    cid = np.zeros(X.shape, dtype=np.int64)
    u = np.zeros(X.shape, dtype=F)
    hx = np.zeros(X.shape, dtype=F)
    v = (Ym - (ye[r] + ye[r + 1]) * 0.5).astype(F)
    hy = (rows[r] * 0.5).astype(F)
    base = 0
    for i in range(len(rows)):
        n = max(2, int(round(W / (mean_w * (0.92 + 0.16 * rng.random())))))
        w = 1.0 + jit_w * (rng.random(n) * 2 - 1)
        w = w / w.sum() * W
        xe = np.concatenate([[0.0], np.cumsum(w)])
        off = rng.random() * W
        m = r == i
        xs = np.mod(Xm[m] - off, W)
        ci = np.clip(np.searchsorted(xe, xs, side="right") - 1, 0, n - 1)
        cid[m] = base + ci
        u[m] = xs - (xe[ci] + xe[ci + 1]) * 0.5
        hx[m] = (xe[ci + 1] - xe[ci]) * 0.5
        base += n
    return cid, u, v, hx, hy, base


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


def build(t):
    rng = np.random.default_rng(4107)

    # ------------------------------------------------------------------ layout
    # courses wander (the pavior follows a string only roughly) and squeeze / widen
    wy = (t.noise(0.55, seed=21) - 0.5) * 0.050 + (t.noise(0.22, seed=22) - 0.5) * 0.020
    wx = (t.noise(0.45, seed=23) - 0.5) * 0.040 + (t.noise(0.20, seed=24) - 0.5) * 0.018
    rows = 0.100 * (1.0 + 0.22 * (rng.random(16) * 2 - 1))
    cid, u, v, hx, hy, n = _courses(t, t.X + wx, t.Y + wy, rows, 0.118, 0.30, seed=77)

    def rnd(k):
        return np.random.default_rng(9000 + k).random(n).astype(F)[cid]

    # each stone: rotated a little, slightly wedge shaped, individually lumpy outline
    ang = (rnd(1) - 0.5) * 0.16
    ca, sa = np.cos(ang), np.sin(ang)
    lx = (t.noise(0.060, seed=31) - 0.5) * 0.015
    ly = (t.noise(0.060, seed=32) - 0.5) * 0.015
    uu = u * ca + v * sa + lx
    vv = -u * sa + v * ca + ly
    uu = uu / (1.0 + (rnd(2) - 0.5) * 0.30 * (vv / hy))          # wedge
    vv = vv / (1.0 + (rnd(3) - 0.5) * 0.22 * (uu / hx))
    a = hx - 0.0005 - 0.0020 * rnd(4)
    b = hy - 0.0005 - 0.0020 * rnd(5)
    ex = 2.8 + 2.4 * rnd(6)                                         # 2.5 = oval ... 4.7 = squarish
    q = np.abs(uu / a) ** ex + np.abs(vv / b) ** ex
    p = sat(1.0 - q)
    prof = p ** (0.30 + 0.16 * rnd(7))                              # 0..1 dome, flat-ish crown

    # ------------------------------------------------------------------ height (metres)
    macro = t.fbm(0.8, 3, 0.5, seed=41)                             # 0.5-1 m patches
    stone_top = 0.027 + 0.006 * rnd(8)                              # stone body above bedding plane
    sink = (rnd(9) - 0.5) * 0.006                                   # some stand proud, some sunk
    tilt = ((rnd(10) - 0.5) * u / hx + (rnd(11) - 0.5) * v / hy) * 0.006
    lumps = (t.fbm(0.045, 3, 0.5, seed=42) - 0.5) * 0.0026          # the stone is not a lathe part
    facet = (t.noise(0.022, seed=43) - 0.5) * 0.0012
    grain = (t.fbm(0.006, 3, 0.6, seed=44) - 0.5) * 0.00045         # crystalline tooth
    pit = _dots(t, 5200, 0.0055, 45, 0.7)[0] * smoothstep(0.45, 0.75, t.noise(0.03, seed=46))
    chip = smoothstep(0.62, 0.80, t.fbm(0.018, 3, 0.55, seed=47)) * smoothstep(0.95, 0.55, prof)   # spalls on the shoulders
    body = prof * stone_top
    # many crowns are worn / split flat: cap the dome with a (slightly inclined) plane, smooth-min
    cap = stone_top * (0.80 + 0.17 * rnd(15)) + ((rnd(16) - 0.5) * u / hx + (rnd(17) - 0.5) * v / hy) * 0.004
    ks = 0.0013
    body = -ks * np.log(np.exp(-body / ks) + np.exp(-cap / ks))
    stone_h = body + sink + tilt + lumps + facet + grain - pit * 0.0011 - chip * 0.0016

    sand_fine = t.fbm(0.005, 3, 0.65, seed=51)
    sand_lvl = 0.0150 + (macro - 0.5) * 0.003 + (t.fbm(0.12, 3, 0.5, seed=52) - 0.5) * 0.004
    grit, grit_r = _dots(t, 26000, 0.0052, 53, 0.8)  # small stones in the joint sand
    sand_h = sand_lvl + (sand_fine - 0.5) * 0.0012 + grit * 0.0013
    # sand banks up softly against the flanks (smooth max)
    k = 0.0016
    d = stone_h - sand_h
    inside = smoothstep(-k, k, d)                                   # 1 = stone visible, 0 = sand
    height = np.maximum(stone_h, sand_h) + k * 0.5 * np.exp(-np.abs(d) / k)

    above = sat((stone_h - sand_h) / 0.0095)                         # 0 at the sand line ... 1 on the crown
    crown = smoothstep(0.45, 0.90, above) * smoothstep(0.45, 0.80, prof)
    worn = 0.45 + 0.55 * smoothstep(0.30, 0.70, t.fbm(0.6, 3, 0.5, seed=60))      # walked lines vs. quiet patches
    crown = crown * (0.60 + 0.40 * smoothstep(0.3, 0.7, t.fbm(0.05, 3, 0.5, seed=61))) * worn * (1 - 0.8 * pit) * (1 - 0.7 * chip)

    # ------------------------------------------------------------------ colour
    tone = rnd(12)
    hue = rnd(13)
    # grey-buff granites: cool grey, warm grey, buff, a few pinkish / darker
    base = ramp(hue, [(0.00, "#7c7973"), (0.22, "#8a8479"), (0.42, "#978c78"), (0.60, "#888580"),
                      (0.78, "#a09480"), (0.90, "#96857b"), (1.00, "#72706c")])
    base = base * (0.70 + 0.52 * tone)[..., None]
    cloud = t.fbm(0.035, 4, 0.55, seed=62)                           # mineral clouding inside a stone
    base = base * (0.88 + 0.24 * cloud)[..., None]
    # crystals: pale feldspar, grey quartz, black biotite - irregular grains of mixed size
    xl = t.fbm(0.0050, 2, 0.6, seed=63)
    xl2 = t.fbm(0.0090, 2, 0.5, seed=66)
    coarse = 0.35 + 0.65 * rnd(14)                                    # some stones coarse grained, some fine
    base = mix(base, base * 1.32 + 0.02, sat(smoothstep(0.60, 0.80, xl) * 0.55 + smoothstep(0.66, 0.82, xl2) * 0.5 * coarse))
    base = mix(base, base * 0.60, sat(smoothstep(0.40, 0.22, xl) * 0.50 + smoothstep(0.34, 0.20, xl2) * 0.4 * coarse))
    mica, mica_r = _dots(t, 22000, 0.0032, 64, 0.7)
    base = mix(base, hexlin("#2a2927"), mica * (0.18 + 0.40 * mica_r))
    base = base * (0.93 + 0.14 * macro)[..., None]
    # lower flanks: damp grime close to the sand, patchy pale sandy dust above it; crowns walked clean
    flank = smoothstep(0.70, 0.05, above)
    dusty = smoothstep(0.35, 0.70, t.fbm(0.030, 3, 0.55, seed=67))
    base = mix(base, base * hexlin("#8a7a62") * 2.2, flank * 0.50)
    base = mix(base, hexlin("#8e8169") * (0.80 + 0.3 * sand_fine)[..., None], sat(flank * dusty * 0.45 + pit * 0.35))
    base = mix(base, base * 0.88, crown * 0.6)
    base = mix(base, base * 0.72, chip * 0.35 * (1 - flank))

    sandc = ramp(sand_fine, [(0.0, "#3f382e"), (0.45, "#5a5041"), (0.8, "#726650"), (1.0, "#8a7d66")])
    deep = sat((sand_lvl - 0.0120) / 0.007)                          # low, damp, dirtier sand is darker
    sandc = sandc * (0.62 + 0.38 * deep)[..., None]
    gcol = ramp(grit_r, [(0, "#4f4b45"), (0.5, "#857e70"), (1, "#a59c8a")])
    sandc = mix(sandc, gcol, grit[..., None] * 0.6)
    albedo = mix(sandc, base, inside)
    # shadowed dirt right at the sand line
    line = np.exp(-np.abs(d) / 0.0022)
    albedo = albedo * (1 - 0.25 * line)[..., None]

    # ------------------------------------------------------------------ roughness
    r_stone = 0.76 + 0.10 * (cloud - 0.5) + 0.12 * flank + 0.10 * pit
    r_stone = mix(r_stone, 0.34 + 0.12 * (xl - 0.5), crown * 0.92)
    rough = mix(0.93 + 0.05 * (sand_fine - 0.5), r_stone, inside)
    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0)
