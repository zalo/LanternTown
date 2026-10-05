"""Hand-made plain clay (peg) tiles, double-lapped: 12 courses x 7 tiles per 1.2 m tile.

Built as real slabs: every tile is a cambered 265 x ~170 x 12 mm slab lying on the course below,
the visible surface is the max over the (up to 4) slabs that cover a point, so tails step up by a
tile thickness, side joints open on to the tile underneath and slipped / chipped tiles reveal it.
Eaves at V = 0 (tails face -Y).
"""
import math
from tk.tex import *

META = {"normal_strength": 1.0}

ROWS, COLS = 12, 7          # 100 mm gauge, 171 mm tiles
T = 0.013                   # tile thickness
LEN = 0.265                 # tile length (2.65 gauges -> double lap)


def lapped(t, rows, cols, T, L, seed, X, Y, wjit=0.2, gap=(0.0008, 0.003), tail_jit=0.003, rot=0.01,
           lift=0.002, tilt=0.0015, camber=(0.002, 0.005), cross=(0.0004, 0.0022), slip_frac=0.03,
           slip=0.014, chip_frac=0.1, chip=(0.012, 0.035), round_r=0.003, wall=5.0, edge_noise=None):
    """Overlapping slab courses. Returns dict of per-pixel fields for the visible (top) slab."""
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
    # roof plane is never dead straight: course lines sag a few mm over the tile
    Y = t.Y + (t.noise(0.6, seed=1) - 0.5) * 0.012 + (t.noise(0.17, seed=2) - 0.5) * 0.004
    X = t.X + (t.noise(0.5, seed=3) - 0.5) * 0.006
    en = (t.fbm(0.012, 3, 0.55, seed=4) - 0.5) * 0.0022          # ragged hand-cut edges
    lay = lapped(t, ROWS, COLS, T, LEN, 11, X, Y, edge_noise=en, gap=(0.0003, 0.0022), tail_jit=0.0045,
                 cross=(0.0008, 0.003), camber=(0.002, 0.006), lift=0.003, chip_frac=0.07, chip=(0.008, 0.024))
    R = lambda k: prand(lay, k)
    d, s, u, w = lay["d"], lay["s"], lay["u"], lay["w"]
    top = smoothstep(-0.0005, 0.0015, d)                          # on a tile face (not in a wall)

    # per-tile texture space so no feature runs across a joint
    ox, oy = R(20) * t.W, R(21) * t.H
    TS = lambda a: t.sample(a, t.X + ox, t.Y + oy)

    # --- height ------------------------------------------------------------------------
    h = lay["h"].copy()
    lump = TS(t.fbm(0.07, 3, 0.5, seed=5))                        # hand-moulded unevenness
    drag = TS(t.fbm(0.012, 3, 0.55, seed=6, stretch=(1.0, 8.3))) # strike / drag marks along length
    grit = t.fbm(0.004, 3, 0.65, seed=7)                          # sand facing
    pits = TS(dots(t, 2600, 0.006, 8, soft=0.7))
    stones = dots(t, 1500, 0.004, 9, soft=0.6)             # grog / pebbles proud of the face
    h += top * ((lump - 0.5) * 0.0022 + (drag - 0.5) * 0.0007 + (grit - 0.5) * 0.00045
                - pits * 0.0009 + stones * 0.0005)
    # frost-spalled tails: a thin lamina lost near the tail of some tiles
    sp_n = TS(t.fbm(0.03, 4, 0.6, seed=10))
    sp_gate = smoothstep(0.86, 0.9, R(5))
    spall = smoothstep(0.62, 0.69, t.blur(sp_n, 0.006) + 0.3 * (1 - sat(s / 0.05))) * sp_gate * top
    h -= spall * (0.0006 + 0.0005 * sp_n)
    # hairline firing crack on a few tiles
    crack = TS(t.cracks(0.3, 0.0012, seed=11, wander=0.006, density=0.45)) * smoothstep(0.9, 0.93, R(6)) * top
    h -= crack * 0.0012

    cav = t.cavity(h, 0.012, 0.003)                               # laps and side joints
    cav_w = t.cavity(h, 0.03, 0.004)
    edge = t.edges(h, 0.005, 0.0012)

    # moss / dirt cushions held in the laps and joints
    mossn = t.fbm(0.09, 4, 0.6, seed=12)
    moss = smoothstep(0.45, 0.75, cav * (0.35 + 1.25 * mossn)) * smoothstep(0.45, 0.6, t.fbm(0.3, 2, seed=13) * 0.4 + mossn * 0.6)
    moss_tx = t.fbm(0.006, 3, 0.7, seed=14)
    h += moss * (0.0025 + 0.003 * moss_tx)

    # --- colour --------------------------------------------------------------------------
    tone = R(1)
    base = ramp(tone, [(0.0, "#664033"), (0.2, "#7b4835"), (0.42, "#8d5239"), (0.62, "#9a5a3b"),
                       (0.82, "#a36541"), (1.0, "#96694d")])
    # kiln flashing: darker, greyer zones within a tile (stacking marks), per tile
    flash = TS(t.fbm(0.11, 3, 0.5, seed=15))
    fl = smoothstep(0.45, 0.8, flash) * (0.25 + 0.75 * R(2))
    base = mix(base, base * hexlin("#8b7a78") * 1.25, fl * 0.75)
    mott = TS(t.fbm(0.025, 4, 0.55, seed=16))
    base = base * (0.84 + 0.32 * mott)[..., None]
    base = base * (0.90 + 0.2 * R(3))[..., None]
    # sand facing: pale and dark grains
    g1 = dots(t, 30000, 0.0028, 17)
    g2 = dots(t, 22000, 0.0030, 18)
    base = mix(base, hexlin("#c9a585"), g1 * 0.32)
    base = mix(base, hexlin("#3b2a24"), g2 * 0.38)
    base = mix(base, hexlin("#b7a48f"), stones[..., None] * 0.35)
    # weather: sooty / algal darkening grows towards the sheltered head of the exposed part,
    # tails are rain-washed and a little paler
    wn = t.fbm(0.16, 4, 0.55, seed=19)
    wn2 = TS(t.fbm(0.05, 3, 0.55, seed=20))
    upslope = smoothstep(0.02, 0.1, s) * (lay["k"] < 0.5)
    soot = sat(0.25 + 0.55 * upslope * wn2 + 0.5 * (wn - 0.5) + 0.35 * (R(4) - 0.5))
    base = mix(base, base * hexlin("#7c6a5c") * 1.05, soot * 0.55)
    base = mix(base, hexlin("#231f1b"), sat(cav_w * 0.8 + cav * 0.5) * 0.7)       # dirt deep in the laps
    # fresh clay on spalled faces and worn arrises
    base = mix(base, hexlin("#a5694a") * (0.8 + 0.4 * mott)[..., None], spall * 0.6)
    base = mix(base, base * 1.25 + 0.01, edge * top * 0.35)
    base = mix(base, hexlin("#1f1a17"), crack * 0.6)
    # lichen: pale grey-green crusts and a few ochre ones, in loose colonies
    colony = smoothstep(0.52, 0.72, t.fbm(0.22, 3, 0.5, seed=21))
    l1 = t.warp(dots(t, 650, 0.013, 22, soft=0.35), 0.003, 0.008, seed=23)
    l2 = t.warp(dots(t, 140, 0.014, 24, soft=0.3), 0.003, 0.01, seed=25)
    l3 = dots(t, 2500, 0.005, 26, soft=0.4)
    lich_g = sat(l1 * (0.15 + 0.85 * colony) + l3 * 0.35 * colony) * top
    lich_o = l2 * smoothstep(0.5, 0.7, t.fbm(0.3, 2, seed=27)) * top
    ltx = 0.8 + 0.4 * t.fbm(0.004, 2, seed=28)
    base = mix(base, hexlin("#85877a") * ltx[..., None], lich_g * 0.55)
    base = mix(base, hexlin("#93844f") * ltx[..., None], lich_o * 0.55)
    h += (lich_g + lich_o) * 0.0004
    # moss
    moss_c = ramp(moss_tx, [(0, "#1c2110"), (0.5, "#343f18"), (1, "#56612a")])
    albedo = mix(base, moss_c, moss * 0.92)

    # --- roughness -------------------------------------------------------------------------
    rough = 0.86 - 0.10 * smoothstep(0.0, 0.25, 0.25 - tone) + 0.07 * (mott - 0.5) - 0.06 * fl
    rough = rough + 0.06 * soot + 0.08 * (lich_g + lich_o) - 0.05 * spall
    rough = mix(rough, 0.95, sat(cav * 0.8))
    rough = mix(rough, 1.0, moss)
    albedo = desaturate(albedo, 0.04)

    # --- AO: generic cavities + the shadowed corner under each tail (occlusion from upslope)
    hor = t.zeros()
    for dy in (0.003, 0.006, 0.011, 0.018, 0.028):
        n = max(1, int(round(dy / t.py)))
        hor = np.maximum(hor, (np.roll(h, n, axis=0) - h) / (n * t.py))
    lap = sat(hor / 1.6)
    ao = t.ao(h, 0.014, 1.2) * (1 - 0.55 * lap)
    albedo = albedo * (1 - 0.22 * lap)[..., None]                 # damp dirt line under the tails
    return dict(albedo=albedo, height=h, rough=sat(rough), metal=0.0, ao=sat(ao))
