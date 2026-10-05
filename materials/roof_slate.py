"""Coursed blue-grey roof slates, 8 courses x 8 slates per 1.5 x 1.2 m tile (187 x 375 mm slates,
150 mm gauge, half-bond). Eaves at V = 0.

Construction: each slate is a riven ~6 mm slab lying on the course below; the visible surface is the
max over the slabs covering a point, so tails step up, the tight perpends open on to the slate
underneath, and broken corners reveal what is below. Every slate sits a little differently (lift,
cross tilt, pitch, a few slightly slipped) so its cleavage face catches the lantern on its own.
Edges are dressed: a 3-5 mm chipped bevel with occasional bites. The riven face is a gently wavy
cleavage plane with fine grain along the length; on some slates a thin lamina has run out near the
tail or one side, leaving a few short parallel steps.
"""
import math
from tk.tex import *

META = {"normal_strength": 1.0}

ROWS, COLS = 8, 8
T = 0.0065
LEN = 0.375


def lapped(t, rows, cols, T, L, seed, X, Y, en, bn, wjit=0.1, gap=(0.0004, 0.0022), tail_jit=0.004,
           rot=0.012, lift=0.0035, tilt=0.003, pitch=0.0015, slip_frac=0.09, slip=(0.010, 0.018),
           chip_frac=0.07, chip=(0.014, 0.04), nib_frac=0.06, cham_w=(0.003, 0.005), cham_d=0.0013,
           round_r=0.002, wall=4.0):
    """Overlapping slab courses. Returns per-pixel fields of the visible (top) slab:
    u, s (slab-local, s = distance up from the tail), w (slab width), d (distance inside the outline),
    brk (distance from a broken edge), cw (bevel width), h, id."""
    rng = np.random.default_rng(seed)
    W, H = t.W, t.H
    g, cw = H / rows, W / cols
    offs = (np.arange(rows) % 2) * 0.5 * cw + (rng.random(rows) - 0.5) * 0.16 * cw
    edges = np.tile(np.arange(cols + 1) * cw, (rows, 1))
    edges[:, 1:-1] += (rng.random((rows, cols - 1)) - 0.5) * cw * wjit
    R = lambda: rng.random((rows, cols)).astype(F)
    gl, gr = [gap[0] + (gap[1] - gap[0]) * R() ** 1.5 for _ in range(2)]
    toff = (R() - 0.5) * 2 * tail_jit
    slipped = R() < slip_frac
    toff = np.where(slipped, -(slip[0] + (slip[1] - slip[0]) * R()), toff)
    rota = (R() - 0.5) * 2 * rot
    lft = R() ** 1.5 * lift + np.where(slipped, 0.0008, 0.0)
    tlx = (R() - 0.5) * 2 * tilt
    twist = (R() - 0.5) * 2 * tilt * 0.6
    pit = (R() - 0.5) * 2 * pitch
    bow = (R() - 0.3) * 0.0006                       # slight cross camber of the slab
    chipped = R() < chip_frac
    chip_side = np.where(R() < 0.5, -1.0, 1.0)
    chip_sz = chip[0] + (chip[1] - chip[0]) * R() ** 1.3
    chip_ang = np.tan(np.radians(25.0 + 40.0 * R()))
    nib = (R() < nib_frac) | (chipped & (R() < 0.6))  # curved conchoidal bite out of the tail
    nib_u = (R() - 0.5) * 1.1                         # position along the tail (fraction of half width * 2)
    nib_r = 0.010 + 0.022 * R()
    nib_c = 0.25 + 0.5 * R()                          # how far the circle centre sits below the tail
    chw = cham_w[0] + (cham_w[1] - cham_w[0]) * R()

    yr = Y / g
    r0 = np.floor(yr).astype(np.int64)
    best = np.full(X.shape, -1e3, dtype=F)
    out = {k: np.zeros(X.shape, dtype=F) for k in ("u", "s", "w", "d", "brk", "cw")}
    oid = np.zeros(X.shape, dtype=np.int64)
    for k in (-1, 0, 1, 2):
        r = r0 - k
        rm = r % rows
        xs = (X - offs[rm]) % W
        c = np.clip(np.floor(xs / cw).astype(np.int64), 0, cols - 1)
        c = np.where(xs < edges[rm, c], c - 1, c)
        c = np.clip(np.where(xs >= edges[rm, np.clip(c, 0, cols - 1) + 1], c + 1, c), 0, cols - 1)
        x0, x1 = edges[rm, c].astype(F), edges[rm, c + 1].astype(F)
        w = x1 - x0
        u = xs - (x0 + x1) * 0.5
        s_nom = (yr - r) * g
        s = s_nom - toff[rm, c] - rota[rm, c] * u
        d = np.minimum(np.minimum(xs - x0 - gl[rm, c], x1 - xs - gr[rm, c]), np.minimum(s, L - s)) + en
        # breakage: a snapped tail corner (random angle, ragged) and / or a curved bite
        a = chip_ang[rm, c]
        dc = ((w * 0.5 - chip_side[rm, c] * u) * a + s - chip_sz[rm, c]) / np.sqrt(1 + a * a) + bn
        dc = np.where(chipped[rm, c], dc, 1.0)
        rr = nib_r[rm, c]
        dn = np.sqrt((u - nib_u[rm, c] * w * 0.5) ** 2 + (s + rr * nib_c[rm, c]) ** 2) - rr + bn * 0.35
        dn = np.where(nib[rm, c], dn, 1.0)
        brk = np.minimum(dc, dn)
        d = np.minimum(d, brk)
        hw = 0.5 * cw
        z = (T * (1 - s_nom / g) + lft[rm, c] + tlx[rm, c] * (u / hw)
             + twist[rm, c] * (u / hw) * (s_nom / g - 0.5) + pit[rm, c] * (s_nom / g - 0.5)
             - bow[rm, c] * (2 * u / w) ** 2)
        chw_ = chw[rm, c]
        dd = np.clip(d / chw_, 0, 1)
        dr = np.clip(d / round_r, 0, 1)
        z = z - cham_d * (1 - dd) ** 1.4 - round_r * 0.5 * (1 - dr) ** 2 - wall * np.clip(-d, 0, None)
        z = z.astype(F)
        win = z > best
        best = np.where(win, z, best)
        oid = np.where(win, rm * cols + c, oid)
        for key, val in (("u", u), ("s", s), ("w", w), ("d", d), ("brk", brk), ("cw", chw_)):
            out[key] = np.where(win, val, out[key]).astype(F)
    out["h"] = best
    out["id"] = oid
    out["n"] = rows * cols
    out["seed"] = seed
    return out


def dots(t, density, size, seed, soft=0.5):
    """Cheap speckle: jittered grid of dots, 3x3 neighbourhood. Returns (mask, per-dot random)."""
    rng = np.random.default_rng(t.seed * 1000 + 9000 + seed)
    c = max(1.0 / math.sqrt(density), t.px * 3)
    nx, ny = max(2, round(t.W / c)), max(2, round(t.H / c))
    cw, ch = t.W / nx, t.H / ny
    px = (rng.random((ny, nx)) * 0.8 + 0.1).astype(F)
    py = (rng.random((ny, nx)) * 0.8 + 0.1).astype(F)
    rv = rng.random((ny, nx)).astype(F)
    rr = (size / 2 * (0.5 + rng.random((ny, nx)))).astype(F)
    gx, gy = t.X / cw, t.Y / ch
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = (gx - ix).astype(F), (gy - iy).astype(F)
    out = t.zeros()
    orv = t.zeros()
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            jx, jy = (ix + ox) % nx, (iy + oy) % ny
            dx = (ox + px[jy, jx] - fx) * cw
            dy = (oy + py[jy, jx] - fy) * ch
            r = rr[jy, jx]
            m = 1 - smoothstep(r * (1 - soft), r, np.sqrt(dx * dx + dy * dy))
            up = m > out
            out = np.where(up, m, out)
            orv = np.where(up, rv[jy, jx], orv)
    return out.astype(F), orv.astype(F)


def prand(lay, k):
    r = np.random.default_rng(lay["seed"] * 7919 + k * 104729 + 13).random(lay["n"]).astype(F)
    return r[lay["id"]]


def build(t):
    g = t.H / ROWS
    Y = t.Y + (t.noise(0.7, seed=1) - 0.5) * 0.008 + (t.noise(0.2, seed=2) - 0.5) * 0.003
    X = t.X + (t.noise(0.6, seed=3) - 0.5) * 0.006
    # dressed edges: mostly straight, with bites 3-10 mm long where the slater's knife spalled the edge
    bite = smoothstep(0.62, 0.86, t.fbm(0.013, 3, 0.5, seed=4))
    en = -bite * 0.0016 + (t.fbm(0.005, 2, 0.5, seed=5) - 0.5) * 0.0005
    bn = (t.fbm(0.016, 3, 0.6, seed=40) - 0.5) * 0.007           # raggedness of broken edges
    lay = lapped(t, ROWS, COLS, T, LEN, 23, X, Y, en, bn)
    R = lambda k: prand(lay, k)
    d, s, u, w = lay["d"], lay["s"], lay["u"], lay["w"]
    top = smoothstep(-0.0004, 0.0010, d)
    ox, oy = R(20) * t.W, R(21) * t.H
    TS = lambda a: t.sample(a, t.X + ox, t.Y + oy)                 # every slate gets its own piece of a field

    # --- riven surface ------------------------------------------------------------------------
    rivenness = 0.45 + 0.9 * R(1)                                   # some slates split cleaner than others
    undul = TS(t.fbm(0.085, 3, 0.5, seed=7, stretch=(1.0, 2.6)))   # gentle cleavage waviness
    grainw = TS(t.fbm(0.022, 3, 0.5, seed=6, stretch=(1.0, 5.0)))  # shallow flutes along the length
    ripple = TS(t.fbm(0.0055, 3, 0.55, seed=8, stretch=(1.0, 7.0)))  # fine grain along the length
    micro = t.fbm(0.0035, 3, 0.65, seed=9)                          # cleavage tooth
    # lamina run-outs: a few short parallel steps near the tail or one side, on ~40% of slates
    qf = TS(t.fbm(0.065, 3, 0.5, seed=31, stretch=(1.0, 7.0))) + (t.fbm(0.005, 2, 0.6, seed=32) - 0.5) * 0.04
    has = (R(40) < 0.42).astype(F)
    side = np.where(R(43) < 0.5, -1.0, 1.0)
    zone_w = 0.028 + 0.024 * R(42)
    dz = np.where(R(41) < 0.62, s, w * 0.5 - side * u)
    zone = (1 - smoothstep(0.15 * zone_w, 1.7 * zone_w, dz)) * has    # the lamina thins out gradually
    th0 = 0.40 + 0.12 * R(44)
    lev = (qf > th0).astype(F) + (qf > th0 + 0.13) + (qf > th0 + 0.27) * (R(45) < 0.5)
    lev = t.blur(lev.astype(F), 0.0007)
    fr = lev - np.floor(lev)
    riser = sat(4 * fr * (1 - fr)) * zone * top
    step_h = 0.0005 + 0.0006 * R(2)
    steps = -lev * step_h * zone

    bev = sat(d / lay["cw"])                                        # 0 at the arris -> 1 on the face
    cham = (1 - bev) * top
    chipn = t.fbm(0.011, 3, 0.55, seed=10)
    fresh = (1 - smoothstep(0.0, 0.006, lay["brk"])) * top          # freshly broken edge face
    h = lay["h"].copy()
    h += top * (((undul - 0.5) * 0.0019 + (grainw - 0.5) * 0.0007) * rivenness
                + (ripple - 0.5) * 0.00020 + (micro - 0.5) * 0.00010 + steps)
    h -= cham * (chipn - 0.35) * 0.0005                             # conchoidal chips on the bevel
    h -= fresh * (0.0012 + (chipn - 0.5) * 0.0016)                  # shelly fracture face

    # a few cracked slates: a hairline from the tail, the loose piece sagging a fraction of a mm
    has_c = (R(70) < 0.05).astype(F)
    ca = (R(71) - 0.5) * 1.1
    cwand = (TS(t.fbm(0.05, 3, 0.55, seed=41)) - 0.5) * 0.02
    dcr = (u - (R(72) - 0.5) * 0.6 * w - s * np.tan(ca) - cwand) * np.cos(ca)
    crack = np.exp(-(dcr / 0.0008) ** 2) * has_c * top
    h -= (smoothstep(-0.0006, 0.0006, dcr * np.where(R(73) < 0.5, -1.0, 1.0)) * 0.0005 * has_c + crack * 0.0008) * top

    cav = t.cavity(h, 0.008, 0.002)
    cav_w = t.cavity(h, 0.03, 0.0035)
    edge = t.edges(h, 0.004, 0.0008)

    # --- colour ----------------------------------------------------------------------------
    tone = R(5)
    base = ramp(tone, [(0.0, "#40454c"), (0.25, "#464c55"), (0.5, "#4c525b"), (0.75, "#51565f"),
                       (1.0, "#575c64")])
    hue = R(6)
    base = mix(base, base * hexlin("#b9a8c4") * 2.15, smoothstep(0.78, 1.0, hue) * 0.3)   # plum-tinged slates
    base = mix(base, base * hexlin("#a9b8a6") * 1.95, smoothstep(0.22, 0.0, hue) * 0.35)  # greenish ones
    cloud = TS(t.fbm(0.08, 4, 0.55, seed=12))
    base = base * (0.87 + 0.26 * cloud)[..., None]
    streak = TS(t.fbm(0.02, 3, 0.55, seed=30, stretch=(1.0, 10.0)))    # mineral banding / rain-washed streaks
    base = base * (0.92 + 0.16 * streak)[..., None]
    base = base * (1 + 0.20 * (ripple - 0.5) + 0.14 * (micro - 0.5))[..., None]
    # weathering bloom: pale mineral film in drifts, heavier on some slates, thinner where rain sheets off the tail
    wn = t.fbm(0.3, 3, 0.5, seed=13)
    wfine = TS(t.fbm(0.035, 4, 0.6, seed=14, stretch=(1.0, 2.2)))
    bloom = sat((wn - 0.5) * 1.1 + (R(7) - 0.45) * 1.1 + (wfine - 0.5) * 1.0 + 0.15 * sat(s / g)
                + (micro - 0.5) * 0.35 + (ripple - 0.5) * 0.3)
    bloom = bloom * (1 - 0.7 * fresh)
    base = mix(base, hexlin("#757b7d"), bloom * 0.34)
    # freshly broken slate (bevel chips, risers, snapped corners) is paler and dustier
    pale = hexlin("#7c8187")
    base = mix(base, pale, sat(cham * (0.18 + 0.5 * chipn) + fresh * (0.35 + 0.4 * chipn)) * 0.7)
    base = base * (1 + 0.13 * riser)[..., None]
    base = mix(base, base * 1.18 + 0.006, edge * top * 0.3)
    # damp dirt in the laps and perpends
    damp = sat(cav_w * 0.8 + cav * 0.7 + crack * 0.8)
    base = mix(base, hexlin("#24272a"), damp * 0.5)

    # iron (pyrite) weeps: about one slate in 15 - a small rusty spot with a wandering, tapering run
    has_r = (R(50) < 0.075).astype(F)
    u0 = (R(51) - 0.5) * 0.66 * w
    s0 = g * (0.42 + 0.5 * R(52))
    rl = 0.02 + 0.045 * R(53)
    ds = s0 - s
    wand = (TS(t.noise(0.028, seed=15)) - 0.5) * 0.02 * sat(ds / 0.03)
    dxr = u - u0 - wand
    tt = sat(ds / rl)
    wid = 0.0007 + (0.0015 + 0.0017 * R(54)) * (1 - tt) ** 0.8
    rstr = 0.35 + 0.9 * TS(t.fbm(0.012, 2, 0.6, seed=16, stretch=(1.0, 3.0)))
    run = np.exp(-(dxr / wid) ** 2) * (ds > 0) * (1 - tt) ** 1.3 * rstr
    r0 = 0.0016 + 0.0012 * R(55)
    spot = np.exp(-((dxr * dxr + ds * ds) / (r0 * r0)) * (0.7 + 0.6 * chipn))
    halo = np.exp(-((dxr * dxr + ds * ds) / (r0 * r0 * 9)))
    rust = sat(np.maximum(run * 0.75, halo * 0.5)) * has_r * top
    spot = spot * has_r * top
    base = mix(base, hexlin("#6a4a33"), rust * (0.4 + 0.3 * R(56)))
    base = mix(base, hexlin("#3d2a1f"), sat(spot * 1.4) * 0.8)

    # lichen: a few crustose patches (lobed, crisp-rimmed), clustered on a minority of slates
    cone, lrv = dots(t, 160, 0.05, 18, soft=1.0)
    lob = t.fbm(0.014, 4, 0.6, seed=19)
    colony = smoothstep(0.55, 0.72, t.fbm(0.35, 2, 0.5, seed=17)) * (R(60) < 0.22)
    lv = (cone * (0.3 + 0.5 * lrv) + (lob - 0.5) * 1.1 - 0.56) * colony + colony - 1
    lich = smoothstep(0.0, 0.03, lv) * top * (1 - fresh)
    rim = lich * (1 - smoothstep(0.02, 0.1, lv))
    ltx = t.fbm(0.003, 2, 0.6, seed=23)
    lcol = mix(hexlin("#747a70"), hexlin("#7b7446"), (lrv > 0.9).astype(F)[..., None])   # grey-green, rarely ochre
    lcol = lcol * (0.8 + 0.4 * ltx)[..., None]
    base = mix(base, lcol, lich * 0.5)
    base = mix(base, base * 1.15, rim * 0.5)
    h += lich * (0.0002 + 0.00015 * ltx)
    albedo = base

    # --- roughness: cleavage faces have a satin sheen that differs slate to slate ------------------
    rough = 0.38 + 0.24 * R(8) + 0.08 * (cloud - 0.5) + 0.07 * (ripple - 0.5) + 0.06 * (streak - 0.5)
    rough = rough + (0.78 - rough) * sat(bloom * 1.1)
    rough = rough + 0.15 * riser + 0.28 * cham + 0.25 * fresh
    rough = mix(rough, 0.9, damp * 0.9)
    rough = mix(rough, 0.95, lich)
    rough = mix(rough, 0.85, sat(rust + spot))

    hor = t.zeros()
    for dy in (0.002, 0.004, 0.008, 0.014, 0.022):
        n = max(1, int(round(dy / t.py)))
        hor = np.maximum(hor, (np.roll(h, n, axis=0) - h) / (n * t.py))
    lap = sat(hor / 1.2)
    ao = t.ao(h, 0.01, 1.2) * (1 - 0.5 * lap)
    albedo = albedo * (1 - 0.2 * lap)[..., None]
    return dict(albedo=albedo, height=h, rough=sat(rough), metal=0.0, ao=sat(ao))
