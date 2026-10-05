"""Dark blue-grey basalt setts, ~100 mm courses in running bond, gritty sand joints.

Construction: every sett is an individually split block (straight sides, slightly out of square,
hand-dressed so no two are the same length) laid tight to a string line.  The top is a split face
(undulating fracture surface with small hackle steps) whose high spots have been ground flat and
glossy by a century of iron tyres and boots; arrises are eased and chipped; joints are packed with
dark grit a few mm below the tops.
"""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.014, "ao_strength": 0.9}


def _courses(t, X, Y, rows, mean_w, jit_w, seed, min_lap=0.0):
    """Rows of blocks.  rows: course heights (rescaled to the tile); block lengths random per block.
    Perpends of neighbouring courses are kept at least `min_lap` apart where possible.
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

    def gap(ja, jb):
        dd = np.abs(ja[:, None] - jb[None, :])
        return np.minimum(dd, W - dd).min()

    base = 0
    prev = first = None
    for i in range(len(rows)):
        best = None
        for _ in range(40):
            n = max(2, int(round(W / (mean_w * (0.94 + 0.12 * rng.random())))))
            w = 1.0 + jit_w * (rng.random(n) * 2 - 1)
            w = w / w.sum() * W
            xe = np.concatenate([[0.0], np.cumsum(w)])
            off = rng.random() * W
            j = np.mod(xe[:-1] + off, W)
            g = 1e9 if prev is None else gap(j, prev)
            if i == len(rows) - 1 and first is not None:
                g = min(g, gap(j, first))
            if best is None or g > best[0]:
                best = (g, n, xe, off, j)
            if g >= min_lap:
                break
        _, n, xe, off, j = best
        prev = j
        if first is None:
            first = j
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
    rng = np.random.default_rng(1203)

    # ------------------------------------------------------------------ layout
    # 12 courses of ~100 mm, setts 150-230 mm long; string lines are nearly straight
    wy = (t.noise(0.50, seed=21) - 0.5) * 0.018 + (t.noise(0.16, seed=22) - 0.5) * 0.007
    wx = (t.noise(0.40, seed=23) - 0.5) * 0.016 + (t.noise(0.16, seed=24) - 0.5) * 0.007
    rows = 0.100 * (1.0 + 0.10 * (rng.random(12) * 2 - 1))
    cid, u, v, hx, hy, n = _courses(t, t.X + wx, t.Y + wy, rows, 0.172, 0.26, seed=311, min_lap=0.040)

    def rnd(k):
        return np.random.default_rng(7000 + k).random(n).astype(F)[cid]

    # every sett slightly rotated / out of square, edges split not sawn
    ang = (rnd(1) - 0.5) * 0.05
    ca, sa = np.cos(ang), np.sin(ang)
    ex = (t.noise(0.045, seed=31) - 0.5) * 0.010 + (t.noise(0.014, seed=33) - 0.5) * 0.0035
    ey = (t.noise(0.045, seed=32) - 0.5) * 0.010 + (t.noise(0.014, seed=34) - 0.5) * 0.0035
    uu = u * ca + v * sa + ex
    vv = -u * sa + v * ca + ey
    uu = uu / (1.0 + (rnd(2) - 0.5) * 0.07 * (vv / hy))
    vv = vv / (1.0 + (rnd(3) - 0.5) * 0.10 * (uu / hx))
    a = hx - 0.0030 - 0.0030 * rnd(4)                               # joint 6-12 mm
    b = hy - 0.0030 - 0.0030 * rnd(5)
    cr = 0.006 + 0.010 * rnd(6)                                     # corner radius
    qx, qy = np.abs(uu) - (a - cr), np.abs(vv) - (b - cr)
    d = cr - (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0))   # >0 inside

    # ------------------------------------------------------------------ height (metres)
    macro = t.fbm(0.7, 3, 0.5, seed=41)
    # split face: undulating fracture with hackle steps
    und = (t.fbm(0.070, 3, 0.5, seed=42) - 0.5) * 0.0050
    hack = t.fbm(0.030, 3, 0.55, seed=43, stretch=(1.6, 1.0))
    step = (np.floor(hack * 6) + smoothstep(0.80, 1.0, hack * 6 - np.floor(hack * 6))) / 6.0
    raw = und + (step - 0.5) * 0.0032 + (t.fbm(0.010, 3, 0.6, seed=44) - 0.5) * 0.0009
    raw = raw + ((rnd(10) - 0.5) * u / hx + (rnd(11) - 0.5) * v / hy) * 0.004
    # traffic grinds the high spots to a common plane (per sett), hollows keep the split texture
    capz = -0.0003 + 0.0014 * (rnd(12) - 0.5) + (t.fbm(0.5, 2, 0.5, seed=45) - 0.5) * 0.0016
    ks = 0.00045
    top = -ks * np.log(np.exp(-raw / ks) + np.exp(-capz / ks))
    polished = 1.0 / (1.0 + np.exp(-(raw - capz) / (ks * 1.3)))     # 1 where ground flat
    pit, _ = _dots(t, 9000, 0.0038, 46, 0.6)                         # vesicles
    pit = pit * smoothstep(0.40, 0.70, t.noise(0.05, seed=47))
    micro = (t.fbm(0.0045, 2, 0.6, seed=48) - 0.5) * 0.00030 * (1 - 0.7 * polished)
    # eased arris + spalled edges / corners
    R = 0.0050 + 0.0065 * rnd(7)
    ease = 1 - (1 - sat(d / R)) ** 2.2
    spall = smoothstep(0.58, 0.78, t.fbm(0.028, 3, 0.55, seed=49)) * smoothstep(0.016, 0.002, d)
    level = (rnd(8) - 0.5) * 0.0055                                  # lipping between setts
    depth = 0.0105                                                   # arris roll-off to the joint bed
    stone_h = level + top + micro - pit * 0.0009 - (1 - ease) * depth - spall * 0.0032

    sand_fine = t.fbm(0.0045, 3, 0.65, seed=51)
    grit, grit_r = _dots(t, 30000, 0.0048, 53, 0.8)
    joint_lvl = -0.0070 + (macro - 0.5) * 0.003 + (t.fbm(0.10, 3, 0.5, seed=52) - 0.5) * 0.004
    joint_h = joint_lvl + (sand_fine - 0.5) * 0.0011 + grit * 0.0012
    k = 0.0009
    dd = stone_h - joint_h
    inside = smoothstep(-k, k, dd) * smoothstep(-0.0010, 0.0006, d)
    height = mix(joint_h, np.maximum(stone_h, joint_h), smoothstep(-0.0010, 0.0006, d)) + 0.0004 * np.exp(-np.abs(dd) / k)

    edge = smoothstep(0.010, 0.001, d)                                # band along the arris

    # ------------------------------------------------------------------ colour
    tone = rnd(13)
    hue = rnd(14)
    base = ramp(hue, [(0.00, "#474c53"), (0.30, "#50555b"), (0.55, "#4a4d51"), (0.75, "#585b5e"),
                      (0.90, "#55514c"), (1.00, "#40454d")])
    base = base * (0.78 + 0.44 * tone)[..., None]
    cloud = t.fbm(0.040, 4, 0.55, seed=61)
    base = base * (0.88 + 0.24 * cloud)[..., None]
    xl = t.fbm(0.0040, 2, 0.6, seed=62)                              # fine-grained groundmass
    base = base * (0.93 + 0.14 * xl)[..., None]
    fleck, fr = _dots(t, 14000, 0.0026, 63, 0.6)                     # sparse pale plagioclase / olivine
    base = mix(base, hexlin("#8d8f8c"), fleck * (0.15 + 0.35 * fr))
    base = base * (0.93 + 0.14 * macro)[..., None]
    # ground-flat tops: cleaner, slightly paler blue-grey sheen;  hollows hold pale dust
    hollow = (1 - polished) * (1 - edge)
    dusty = smoothstep(0.30, 0.70, t.fbm(0.060, 3, 0.55, seed=64))
    base = mix(base, base * 1.12 + hexlin("#0b0d10") * 0.25, polished * 0.55)
    base = mix(base, hexlin("#77705f") * (0.75 + 0.5 * sand_fine)[..., None], sat(hollow * (0.12 + 0.32 * dusty) + pit * 0.4))
    # arris: fresh spalls are a little lighter and greyer, low edge near the joint is grimy
    base = mix(base, base * 1.22 + 0.006, spall * 0.5)
    low = smoothstep(0.0, 1.0, (1 - ease))
    base = mix(base, base * hexlin("#7d705c") * 2.3, low * 0.65)

    sandc = ramp(sand_fine, [(0.0, "#2b2823"), (0.45, "#423c33"), (0.8, "#5b5346"), (1.0, "#746a58")])
    deepj = sat((joint_lvl + 0.0105) / 0.007)
    sandc = sandc * (0.60 + 0.40 * deepj)[..., None]
    gcol = ramp(grit_r, [(0, "#34373a"), (0.5, "#5f5d57"), (0.85, "#857f72"), (1, "#9a9485")])
    sandc = mix(sandc, gcol, grit[..., None] * 0.65)
    albedo = mix(sandc, base, inside)
    albedo = albedo * (1 - 0.22 * np.exp(-np.abs(dd) / 0.0018) * smoothstep(0.004, 0.0, np.abs(d)))[..., None]

    # ------------------------------------------------------------------ roughness
    r_stone = 0.80 + 0.10 * (cloud - 0.5) + 0.08 * pit + 0.08 * low
    r_stone = mix(r_stone, 0.36 + 0.10 * (xl - 0.5) + 0.10 * (cloud - 0.5), polished * 0.92)
    r_stone = mix(r_stone, 0.70, spall * 0.6)
    rough = mix(0.93 + 0.05 * (sand_fine - 0.5), r_stone, inside)
    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0)
