"""Hand-split cedar shingles, silvered and cupped: 8 courses (125 mm exposure) of random-width
(55-210 mm) shingles per 1.2 x 1.0 m tile. Each shingle is a 400 mm tapered slab lying on the
courses below (three-ply), the visible surface is the max over the slabs covering a point, so butts
step up, keyways open on to the shingle underneath. Grain runs up the slope (along V) because that
is how a shingle is split; eaves at V = 0.
"""
import math
from tk.tex import *

META = {"normal_strength": 1.0}

ROWS, COLS = 8, 9
T = 0.012
LEN = 0.40


def lapped(t, rows, cols, T, L, seed, X, Y, wjit=0.2, gap=(0.0008, 0.003), tail_jit=0.003, rot=0.01,
           lift=0.002, tilt=0.0015, camber=(0.002, 0.005), cross=(0.0004, 0.0022), slip_frac=0.03,
           slip=0.014, chip_frac=0.1, chip=(0.012, 0.035), round_r=0.003, wall=5.0, edge_noise=None):
    """Overlapping slab courses. Returns dict of per-pixel fields for the visible (top) slab."""
    rng = np.random.default_rng(seed)
    W, H = t.W, t.H
    g, cw = H / rows, W / cols
    offs = rng.random(rows) * cw + np.arange(rows) * 0.37 * cw            # random bond
    edges = np.tile(np.arange(cols + 1) * cw, (rows, 1))
    edges[:, 1:-1] += (rng.random((rows, cols - 1)) - 0.5) * cw * wjit
    R = lambda: rng.random((rows, cols)).astype(F)
    gl, gr = [gap[0] + (gap[1] - gap[0]) * R() ** 1.5 for _ in range(2)]
    toff = (R() - 0.5) * 2 * tail_jit
    slipped = R() < slip_frac
    toff = np.where(slipped, -slip * (0.5 + 0.5 * R()), toff)
    rota = (R() - 0.5) * 2 * rot
    lft = R() * lift + np.where(slipped, 0.001, 0.0)
    tlx = (R() - 0.5) * 2 * tilt
    twist = (R() - 0.5) * 2 * tilt
    cam = camber[0] + (camber[1] - camber[0]) * R()
    crs = (cross[0] + (cross[1] - cross[0]) * R()) * np.where(R() < 0.12, -0.6, 1.0)
    chipped = R() < chip_frac
    chip_side = np.where(R() < 0.5, -1.0, 1.0)
    chip_sz = chip[0] + (chip[1] - chip[0]) * R()
    chip_ang = 0.6 + 0.8 * R()

    yr = Y / g
    r0 = np.floor(yr).astype(np.int64)
    best = np.full(X.shape, -1e3, dtype=F)
    out = {k: np.zeros(X.shape, dtype=F) for k in ("u", "s", "w", "d", "k")}
    oid = np.zeros(X.shape, dtype=np.int64)
    en = 0.0 if edge_noise is None else edge_noise
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
        d = np.minimum(np.minimum(xs - x0 - gl[rm, c], x1 - xs - gr[rm, c]), np.minimum(s, L - s))
        # broken tail corner on some slabs
        ch = chipped[rm, c]
        dc = ((w * 0.5 - chip_side[rm, c] * u) * chip_ang[rm, c] + s - chip_sz[rm, c]) / np.sqrt(1 + chip_ang[rm, c] ** 2)
        d = np.where(ch, np.minimum(d, dc), d) + en
        x = np.clip(s_nom / L, -0.2, 1.2)
        z = (T * (1 - s_nom / g) + lft[rm, c] + tlx[rm, c] * (u / (0.5 * cw))
             + twist[rm, c] * (u / (0.5 * cw)) * (s_nom / g - 0.5)
             + cam[rm, c] * 4 * x * (1 - x) - crs[rm, c] * (2 * u / w) ** 2)
        dd = np.clip(d / round_r, 0, 1)
        z = z - round_r * 0.6 * (1 - dd) ** 2 - wall * np.clip(-d, 0, None)
        z = z.astype(F)
        win = z > best
        best = np.where(win, z, best)
        oid = np.where(win, rm * cols + c, oid)
        for key, val in (("u", u), ("s", s), ("w", w), ("d", d)):
            out[key] = np.where(win, val, out[key]).astype(F)
        out["k"] = np.where(win, F(k), out["k"])
    out["h"] = best
    out["id"] = oid
    out["n"] = rows * cols
    out["seed"] = seed
    return out


def dots(t, density, size, seed, soft=0.5):
    """Cheap speckle: jittered grid of dots, 3x3 neighbourhood only (t.speckle is 25 taps)."""
    rng = np.random.default_rng(t.seed * 1000 + 9000 + seed)
    c = max(1.0 / math.sqrt(density), t.px * 3)
    nx, ny = max(2, round(t.W / c)), max(2, round(t.H / c))
    cw, ch = t.W / nx, t.H / ny
    px = (rng.random((ny, nx)) * 0.8 + 0.1).astype(F)
    py = (rng.random((ny, nx)) * 0.8 + 0.1).astype(F)
    rr = (size / 2 * (0.5 + rng.random((ny, nx)))).astype(F)
    gx, gy = t.X / cw, t.Y / ch
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = (gx - ix).astype(F), (gy - iy).astype(F)
    out = t.zeros()
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            jx, jy = (ix + ox) % nx, (iy + oy) % ny
            dx = (ox + px[jy, jx] - fx) * cw
            dy = (oy + py[jy, jx] - fy) * ch
            r = rr[jy, jx]
            out = np.maximum(out, 1 - smoothstep(r * (1 - soft), r, np.sqrt(dx * dx + dy * dy)))
    return out.astype(F)


def prand(lay, k):
    r = np.random.default_rng(lay["seed"] * 7919 + k * 104729 + 13).random(lay["n"]).astype(F)
    return r[lay["id"]]


def build(t):
    Y = t.Y + (t.noise(0.6, seed=1) - 0.5) * 0.012 + (t.noise(0.15, seed=2) - 0.5) * 0.004
    X = t.X + (t.noise(0.5, seed=3) - 0.5) * 0.008
    # butts are sawn/eroded across the grain: ragged along X only
    en = (t.fbm(0.004, 3, 0.6, seed=4, stretch=(1.0, 20.0)) - 0.5) * 0.004
    lay = lapped(t, ROWS, COLS, T, LEN, 37, X, Y, wjit=0.55, gap=(0.0008, 0.0045), tail_jit=0.006, rot=0.03,
                 lift=0.004, tilt=0.002, camber=(-0.003, 0.001), cross=(-0.0045, -0.0008), slip_frac=0.03,
                 slip=0.018, chip_frac=0.06, chip=(0.01, 0.03), round_r=0.0025, wall=4.5, edge_noise=en)
    R = lambda k: prand(lay, k)
    d, s, u, w = lay["d"], lay["s"], lay["u"], lay["w"]
    top = smoothstep(-0.0005, 0.0015, d)
    ox, oy = R(20) * t.W, R(21) * t.H
    TS = lambda a: t.sample(a, t.X + ox, t.Y + oy)

    # --- height: split face with raised latewood ridges, deeper erosion near the butt ---------
    wav = TS(t.fbm(0.2, 2, 0.4, seed=5))                                     # grain wanders gently
    xs_ = t.X + ox + (wav - 0.5) * 0.012
    GS = lambda a: t.sample(a, xs_, t.Y + oy)
    split = GS(t.fbm(0.03, 3, 0.5, seed=6, stretch=(1.0, 16.7)))             # riven undulation
    grain = GS(t.fbm(0.0055, 2, 0.6, seed=7, stretch=(1.0, 50.0)))           # latewood ridges
    fine = GS(t.fbm(0.002, 2, 0.6, seed=8, stretch=(1.0, 25.0)))
    erode = 0.6 + 0.9 * (1 - smoothstep(0.0, 0.07, s)) + 0.5 * R(1)           # butts weather hardest
    h = lay["h"].copy()
    h += top * ((split - 0.5) * 0.0016 + (grain - 0.5) * 0.0010 * erode + (fine - 0.5) * 0.0003)
    # checks: splits running up from the butt, and a few shingles cracked right through
    chk = GS(t.fbm(0.012, 2, 0.5, seed=9, stretch=(1.0, 83.3)))
    check = smoothstep(0.80, 0.9, chk) * (1 - smoothstep(0.01, 0.03 + 0.07 * R(2), s)) * top
    cu = (R(3) - 0.5) * 0.5 * w + (TS(t.noise(0.12, seed=10, stretch=(0.4, 1.0))) - 0.5) * 0.012
    cw_ = 0.0004 + 0.0012 * sat(1 - s / (0.06 + 0.12 * R(11)))               # opens towards the butt
    crack = (1 - smoothstep(cw_, cw_ + 0.0012, np.abs(u - cu))) * (R(4) < 0.14) * (w > 0.1) * top
    h -= check * 0.0022 + crack * 0.005
    # knots are rare in shingle stock; nail heads are covered. Small worm of moss dirt in keyways
    h = np.maximum(h, -2.3 * T)

    cav = t.cavity(h, 0.01, 0.003)
    cav_w = t.cavity(h, 0.03, 0.004)
    gcav = t.cavity(h, 0.002, 0.00035)                                        # grain furrows
    edge = t.edges(h, 0.004, 0.001)

    # --- colour: UV-silvered cedar, browner where sheltered under the lap, dark damp butts ------
    tone = R(5)
    grey = ramp(tone, [(0.0, "#77726b"), (0.3, "#8b867e"), (0.6, "#9d978e"), (0.85, "#aaa59a"), (1.0, "#948d83")])
    brown = ramp(R(6), [(0.0, "#62493a"), (0.5, "#775a44"), (1.0, "#8a6a4e")])
    shelter = smoothstep(0.065, 0.125, s) * (lay["k"] < 0.5)                  # just below the next butt
    keep = sat(0.7 * shelter + 0.7 * smoothstep(0.55, 0.95, R(7)) + 0.7 * (TS(t.fbm(0.06, 3, 0.5, seed=11, stretch=(1.0, 3.3))) - 0.5))
    base = mix(grey, brown, keep * 0.85)
    base = base * (0.86 + 0.28 * R(13))[..., None]
    lat = GS(t.fbm(0.004, 2, 0.6, seed=12, stretch=(1.0, 50.0)))              # growth-ring streaks
    base = base * (0.78 + 0.44 * lat)[..., None]
    band = GS(t.fbm(0.014, 2, 0.5, seed=31, stretch=(1.0, 33.3)))             # broad early/late wood bands
    base = mix(base, base * hexlin("#8a7a6a") * 1.5, smoothstep(0.5, 0.85, band) * 0.35)
    base = base * (0.92 + 0.16 * split)[..., None]
    base = mix(base, base * 1.22 + 0.01, sat(edge * top * 0.4 + (grain - 0.6) * 1.2 * top) * 0.45)   # bleached ridges
    base = mix(base, hexlin("#3a352f"), sat(gcav * 0.75) * top * 0.6)         # dirt / mildew in furrows
    # black mildew: speckled, strongest at the damp butt and in patches
    mild_n = t.fbm(0.2, 4, 0.55, seed=13)
    butt = 1 - smoothstep(0.0, 0.03 + 0.02 * R(8), s)
    msp = t.fbm(0.004, 3, 0.7, seed=14)
    mild = sat(butt * 1.0 + smoothstep(0.5, 0.8, mild_n) * 0.5 + 0.3 * (R(9) - 0.5)) * smoothstep(0.35, 0.7, msp + 0.25 * butt)
    base = mix(base, hexlin("#2c2925"), mild * 0.62 * top)
    wst = GS(t.fbm(0.015, 3, 0.55, seed=15, stretch=(1.0, 25.0)))             # rain streaks down the grain
    base = base * (0.9 + 0.2 * wst)[..., None]
    base = mix(base, hexlin("#171513"), sat(cav_w * 0.9 + cav * 0.6 + check * 0.8 + crack) * 0.75)
    # sparse moss dots lodged in the laps / keyways
    mossn = t.fbm(0.1, 3, 0.6, seed=16)
    moss = smoothstep(0.6, 0.85, cav * (0.2 + 1.2 * mossn)) * smoothstep(0.5, 0.65, t.fbm(0.35, 2, seed=17))
    mtx = t.fbm(0.005, 3, 0.7, seed=18)
    base = mix(base, ramp(mtx, [(0, "#1d2211"), (0.5, "#38411d"), (1, "#57602c")]), moss * 0.9)
    h += moss * (0.002 + 0.002 * mtx)
    # pale lichen flecks, few
    l1 = dots(t, 500, 0.008, 19, soft=0.4) * smoothstep(0.5, 0.7, t.fbm(0.3, 3, seed=20)) * top
    base = mix(base, hexlin("#9a9d8c"), l1 * 0.45)
    albedo = base * 1.1

    rough = 0.84 + 0.12 * (R(12) - 0.5) + 0.10 * (lat - 0.5) - 0.16 * sat((grain - 0.55) * 2.5) + 0.08 * mild - 0.12 * keep
    rough = mix(rough, 0.96, sat(cav + gcav * 0.5))
    rough = mix(rough, 1.0, moss)

    hor = t.zeros()
    for dy in (0.003, 0.006, 0.011, 0.018, 0.028):
        n = max(1, int(round(dy / t.py)))
        hor = np.maximum(hor, (np.roll(h, n, axis=0) - h) / (n * t.py))
    lap = sat(hor / 1.5)
    ao = t.ao(h, 0.012, 1.2) * (1 - 0.55 * lap)
    albedo = albedo * (1 - 0.25 * lap)[..., None]
    return dict(albedo=albedo, height=h, rough=sat(rough), metal=0.0, ao=sat(ao))
