"""Large warm-buff York sandstone flags, random rectangular coursing, lipped and cracked.

Construction: riven (split along the bedding) sandstone flags sawn to a 150 mm module - 300 mm
closers up to 900 x 600 mm flags - laid "random rectangular": no continuous courses, joints
broken every few flags, almost no cross joints.  A riven face is the bedding plane the block was
split along: broadly undulating, with a few open, sweeping, sub-parallel delamination edges where
a lamina has torn away (a sharp 1-2.5 mm scarp on one side feathering out over a few centimetres
on the other, each dying out along its length).  Flags have settled individually (lips at the
joints, some rock on their bed and sit tilted), the walked centres are dished and burnished,
several carry hairline cracks, one is cracked right through, arrises are worn and spalled, and
the joints are pointed with gritty sand-lime mortar weathered back a few mm.
"""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.02, "ao_strength": 0.9}

# The bond, on a 16 x 16 grid of 150 mm modules that wraps at the tile edge: (x, y, w, h) per flag.
# 24 flags from 300 x 300 closers to 900 x 450 / 600 x 600, laid "random rectangular" the way a mason
# does it: courses are broken everywhere (2-course-high flags, closers), no four flags meet at a
# point, no straight joint runs further than 1.35 m, several flags lie across the tile seams.
# (Worked out once with a randomised exact-cover search that kept re-laying groups of flags until
# those rules held; stored as data so the bake stays fast.)
_FLAGS = [(0, 14, 6, 3), (6, 0, 3, 3), (9, 0, 4, 4), (13, 13, 3, 4), (13, 1, 4, 3), (7, 3, 2, 4), (9, 4, 6, 3),
          (1, 1, 5, 2), (1, 3, 4, 3), (4, 6, 3, 3), (7, 7, 5, 2), (12, 7, 3, 3), (15, 8, 2, 2), (15, 6, 5, 2),
          (8, 9, 4, 4), (4, 9, 4, 2), (1, 8, 3, 2), (0, 10, 4, 4), (12, 10, 4, 3), (8, 13, 5, 3), (5, 3, 2, 3),
          (4, 11, 4, 3), (6, 14, 2, 2), (15, 4, 2, 2)]


def _layout(N=16):
    grid = -np.ones((N, N), dtype=np.int64)
    for i, (x, y, w, h) in enumerate(_FLAGS):
        grid[np.ix_([(y + k) % N for k in range(h)], [(x + k) % N for k in range(w)])] = i
    assert (grid >= 0).all()
    return grid, list(_FLAGS)


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
    size = max(size, t.px * 1.6)
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


class _Samp:
    """Bilinear wrapping lookup into images of canvas `c` at fixed positions (indices computed once)."""

    def __init__(self, c, X, Y):
        fx = X / c.px - 0.5
        fy = (c.H - Y) / c.py - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        self.tx, self.ty = (fx - x0).astype(F), (fy - y0).astype(F)
        self.x1, self.y1 = (x0 + 1) % c.w, (y0 + 1) % c.h
        self.x0, self.y0 = x0 % c.w, y0 % c.h

    def __call__(self, img):
        x0, x1, y0, y1, tx, ty = self.x0, self.x1, self.y0, self.y1, self.tx, self.ty
        return ((img[y0, x0] * (1 - tx) + img[y0, x1] * tx) * (1 - ty) + (img[y1, x0] * (1 - tx) + img[y1, x1] * tx) * ty).astype(F)


def build(t):
    N = 16
    grid, rects = _layout(N)
    n = len(rects)
    W, H = t.W, t.H
    lrng = np.random.default_rng(77)
    xe = np.arange(N + 1) * (W / N)
    ye = np.arange(N + 1) * (H / N)
    xe[1:N] += (lrng.random(N - 1) - 0.5) * 0.022          # flags are only nominally modular
    ye[1:N] += (lrng.random(N - 1) - 0.5) * 0.022
    xx = np.concatenate([xe[:N], xe[:N] + W, [2 * W]])
    yy = np.concatenate([ye[:N], ye[:N] + H, [2 * H]])
    R = np.array(rects)
    cxs, hxs = (xx[R[:, 0]] + xx[R[:, 0] + R[:, 2]]) * 0.5, (xx[R[:, 0] + R[:, 2]] - xx[R[:, 0]]) * 0.5
    cys, hys = (yy[R[:, 1]] + yy[R[:, 1] + R[:, 3]]) * 0.5, (yy[R[:, 1] + R[:, 3]] - yy[R[:, 1]]) * 0.5
    area = R[:, 2] * R[:, 3]

    # smooth fields are synthesised on a fixed 512 px canvas (same pattern at any bake resolution)
    lo = type(t)(res=512, tile=t.tile, seed=t.seed)
    up = _Samp(lo, t.X, t.Y)

    ew1 = (up(lo.noise(0.9, seed=21)) - 0.5) * 0.008
    ew2 = (up(lo.noise(0.9, seed=22)) - 0.5) * 0.008
    Xw, Yw = np.mod(t.X + ew1, W), np.mod(t.Y + ew2, H)
    ix = np.clip(np.searchsorted(xe, Xw, side="right") - 1, 0, N - 1)
    iy = np.clip(np.searchsorted(ye, Yw, side="right") - 1, 0, N - 1)
    cid = grid[iy, ix]
    u = (np.mod(Xw - cxs[cid] + W * 0.5, W) - W * 0.5).astype(F)
    v = (np.mod(Yw - cys[cid] + H * 0.5, H) - H * 0.5).astype(F)
    hx, hy = hxs[cid].astype(F), hys[cid].astype(F)

    def per(k):
        return np.random.default_rng(5000 + k).random(n).astype(F)

    def rnd(k):
        return per(k)[cid]

    # edges: sawn straight but hand-dressed and worn -> a few mm of wobble, flags not quite square on
    ex = (up(lo.noise(0.12, seed=31)) - 0.5) * 0.008 + (t.noise(0.030, seed=33) - 0.5) * 0.004
    ey = (up(lo.noise(0.12, seed=32)) - 0.5) * 0.008 + (t.noise(0.030, seed=34) - 0.5) * 0.004
    ang = (rnd(1) - 0.5) * 0.010
    uu = u + v * ang + ex
    vv = v - u * ang + ey
    a = hx - 0.0035 - 0.0025 * rnd(4)                               # joints 7-12 mm
    b = hy - 0.0035 - 0.0025 * rnd(5)
    cr = 0.005 + 0.010 * rnd(6)
    qx, qy = np.abs(uu) - (a - cr), np.abs(vv) - (b - cr)
    d = cr - (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0))   # >0 inside

    # ------------------------------------------------------------------ flag-local (rotated) fields
    # every flag is its own piece of rock: its bedding undulation and delamination edges have their
    # own direction, so sample the noise fields in a per-flag rotated / offset frame.
    th = rnd(30) * np.pi
    ct, st = np.cos(th), np.sin(th)
    lu = u * ct + v * st
    lv = -u * st + v * ct
    loc = _Samp(lo, lu + rnd(31) * W, lv + rnd(32) * H)

    wA = loc(lo.noise(0.34, seed=101)) - 0.5
    wB = loc(lo.noise(0.09, seed=102)) - 0.5
    wC = loc(lo.noise(0.26, seed=103)) - 0.5
    und = loc(lo.fbm(0.26, 2, 0.5, seed=104, stretch=(1.0, 1.4))) - 0.5
    rip = loc(lo.fbm(0.15, 2, 0.42, seed=105, stretch=(1.0, 1.3))) - 0.5
    cloud = loc(lo.fbm(0.16, 4, 0.55, seed=106))
    iron = loc(lo.fbm(0.14, 3, 0.5, seed=107, stretch=(0.8, 1.5)))
    rag = (t.fbm(0.022, 3, 0.6, seed=108) - 0.5) * 0.009 + (t.fbm(0.006, 2, 0.6, seed=109) - 0.5) * 0.0018

    # ------------------------------------------------------------------ riven face: delamination edges
    wr = max(0.0022, t.px * 0.9)
    ext = np.abs(hx * ct) + np.abs(hy * st)                          # half extent across the edges
    fl_area = hx * hy * 4
    nst = np.where(fl_area < 0.13, 1.0, 2.0) + np.floor(rnd(40) * np.where(fl_area < 0.13, 2.0, 3.0))   # 2-4 (closers 1-2)
    fsign = np.where(rnd(39) < 0.5, -1.0, 1.0)
    terr = t.zeros()
    foot = t.zeros()
    ltone = t.zeros()
    s_main = lu + wA * 0.24 + wB * 0.036 + rag
    dth = (0.35 + 0.45 * rnd(38)) * np.where(rnd(37) < 0.5, -1.0, 1.0)
    s_min = lu * np.cos(dth) + lv * np.sin(dth) + wC * 0.20 - wB * 0.04 + rag * 1.3
    for k in range(6):
        main = k < 4
        env = loc(lo.noise(0.26 if main else 0.17, seed=120 + k, stretch=(1.0, 2.0)))
        if main:
            act = (k < nst).astype(F)
            pk = ext * (-1 + 2 * (k + 0.5 + (rnd(41 + k) - 0.5) * 0.7) / nst)
            x = (s_main - pk) * np.where(rnd(45 + k) < 0.8, 1.0, -1.0) * fsign
            hk = 0.0018 + 0.0014 * rnd(49 + k)                                   # 1.5-3 mm scarp
            A = act * smoothstep(0.36, 0.56, env)
        else:                                                                    # short secondary flakes
            pk = ext * (rnd(41 + k) - 0.5) * 1.3
            x = (s_min - pk) * np.where(rnd(45 + k) < 0.5, 1.0, -1.0)
            hk = 0.0008 + 0.0008 * rnd(49 + k)
            A = smoothstep(0.50, 0.66, env) * (fl_area > 0.10)
        lam = 0.10 + 0.40 * rnd(53 + k)                                          # how much of the step persists
        L = 0.020 + 0.018 * rnd(57 + k)                                          # feathered back-slope
        rise = smoothstep(-wr, wr, x)
        f = rise * (lam + (1 - lam) * np.exp(-np.maximum(x, 0) / L))
        terr += A * hk * (f - 0.5 * lam)
        foot = np.maximum(foot, A * (1 - rise) * np.exp(-np.maximum(-x, 0) / 0.006) * (hk / 0.0025))
        ltone += A * (f - 0.5 * lam) * (0.25 + 0.75 * rnd(61 + k))

    und_h = und * 0.0100                                                         # bedding undulation
    rip_h = rip * 0.0052                                                         # riven hummocks 30-100 mm
    walked = smoothstep(0.25, 0.75, rnd(17))
    Rr = 0.005 + 0.007 * rnd(7)
    ease = 1 - (1 - sat(d / Rr)) ** 2.0
    # ------------------------------------------------------------------ masks
    foot_fall = smoothstep(0.32, 0.68, up(lo.noise(0.5, seed=71)))                  # where people walk
    rels = terr + und_h + 0.6 * rip_h
    crown = smoothstep(0.0002, 0.0026, rels) * ease
    hollow = smoothstep(-0.0003, -0.0028, rels)
    polish = sat(crown * (0.55 + 0.45 * foot_fall) * (0.7 + 0.3 * walked))

    grain = (t.fbm(0.006, 3, 0.65, seed=46) - 0.5) * 0.00040 + (t.fbm(0.040, 4, 0.6, seed=40) - 0.5) * 0.0017 * (1.0 - 0.55 * polish + 0.25 * hollow)
    pclus = smoothstep(0.50, 0.72, up(lo.noise(0.15, seed=48)))
    pit, _ = _dots(t, 14000, 0.0032, 47, 0.7)
    pit = pit * pclus
    dish = -(0.0012 + 0.0020 * walked) * smoothstep(0.0, 0.17, d) * smoothstep(0.08, 0.20, np.minimum(hx, hy))
    relh = terr + und_h + rip_h                                                  # height above the flag's mean plane
    face = relh + grain - pit * 0.0006 + dish

    # ------------------------------------------------------------------ cracks & broken corners
    crng = np.random.default_rng(909)
    cand = [i for i in np.argsort(-area + crng.random(n)) if area[i] >= 12]
    crng.shuffle(cand)
    kind = np.zeros(n, dtype=np.int64)                 # 1 hairline, 2 through crack, 3 broken corner
    wide_c = [i for i in cand if 12 <= area[i] <= 16]
    if wide_c:
        kind[wide_c[0]] = 2
    left = [i for i in cand if kind[i] == 0]
    for i in left[:3]:
        kind[i] = 1
    for i in left[3:4]:
        kind[i] = 3
    kd = kind[cid]
    wob = (up(lo.noise(0.16, seed=35)) - 0.5) * 0.060 + (t.noise(0.035, seed=36) - 0.5) * 0.012 + rag * 0.5
    cth = (rnd(21) - 0.5) * 1.0 + np.where(hx >= hy, 0.0, np.pi * 0.5)             # across the long span
    s1 = uu * np.cos(cth) + vv * np.sin(cth) - (rnd(22) - 0.5) * 0.6 * np.maximum(hx, hy) + wob
    al = (-uu * np.sin(cth) + vv * np.cos(cth)) / np.minimum(hx, hy)               # -1..1 along the crack
    has1 = (kd == 1).astype(F)
    hasw = (kd == 2).astype(F)
    die = np.where(rnd(18) < 0.6, smoothstep(-0.25, 0.45, al * np.where(rnd(19) < 0.5, -1.0, 1.0) + (rnd(16) - 0.5) * 0.8), 1.0)
    hw = max(0.0008, t.px * 0.75)
    hair = has1 * np.exp(-(s1 / hw) ** 2) * die
    sxs = np.where(rnd(24) < 0.5, -1.0, 1.0)
    sys_ = np.where(rnd(25) < 0.5, -1.0, 1.0)
    s2 = (sxs * uu / hx + sys_ * vv / hy - (1.10 + 0.25 * rnd(26))) * np.minimum(hx, hy) * 0.6 + wob * 0.5
    has2 = (kd == 3).astype(F)
    hair = np.maximum(hair, has2 * np.exp(-(s2 / hw) ** 2))
    ww = 0.0018 + 0.0014 * t.noise(0.06, seed=37)                                  # 3.5-6 mm wide gap
    wide = hasw * (1 - smoothstep(ww * 0.6, ww * 1.25, np.abs(s1)))
    chipn = t.fbm(0.035, 3, 0.55, seed=38)
    chip = hasw * smoothstep(0.52, 0.70, chipn) * (1 - smoothstep(0.004, 0.016, np.abs(s1)))   # chipped margin
    near_w = hasw * np.exp(-(s1 / 0.012) ** 2)
    side = has1 * die * np.tanh(s1 / 0.003) * (rnd(27) - 0.5) * 0.0010             # hairline: barely offset
    side = side + hasw * np.tanh(s1 / 0.003) * (0.0009 + 0.0004 * (vv / hy))        # through crack: 1.5-2 mm step
    side = side + has2 * (0.5 + 0.5 * np.tanh(s2 / 0.003)) * (-(0.0010 + 0.0015 * rnd(28)))   # corner piece sinks

    # ------------------------------------------------------------------ flags: lipping, tilt, arris
    spall = smoothstep(0.62, 0.80, t.fbm(0.055, 3, 0.55, seed=49)) * smoothstep(0.030, 0.003, d)
    lvl = (rnd(8) - 0.5) * 0.012                                                   # lipping +-6 mm
    rock = np.where(rnd(12) < 0.30, 0.0090, 0.0035)                                # ~30 % rock on their bed
    tilt = ((rnd(10) - 0.5) * u / hx + (rnd(11) - 0.5) * v / hy) * rock
    plane = lvl + tilt + side
    depth = 0.0110
    stone_h = plane + face - (1 - ease) * depth - spall * 0.0032 - hair * 0.0011 - wide * 0.0060 - chip * 0.0022

    macro = up(lo.fbm(1.1, 3, 0.5, seed=41))
    sand_fine = t.fbm(0.006, 3, 0.65, seed=51)
    grit, grit_r = _dots(t, 9000, 0.0065, 53, 0.8)
    kk = 320.0                                                                     # mortar follows the lower flag
    settle = -np.log(np.maximum(t.blur(np.exp(-kk * plane), 0.007), 1e-9)) / kk
    jdepth = -0.0055 + (macro - 0.5) * 0.002 + (up(lo.fbm(0.18, 3, 0.5, seed=52)) - 0.5) * 0.005
    joint_h = settle + jdepth + (sand_fine - 0.5) * 0.0013 + grit * 0.0012
    instone = smoothstep(-0.0012, 0.0008, d)
    dd = stone_h - joint_h
    inside = instone * np.maximum(smoothstep(-0.0008, 0.0008, dd), smoothstep(0.012, 0.022, d))
    height = mix(joint_h, mix(np.maximum(stone_h, joint_h), stone_h, smoothstep(0.012, 0.022, d)), instone)

    # ------------------------------------------------------------------ colour
    hue = rnd(14)
    base = ramp(hue, [(0.00, "#ad9d83"), (0.30, "#b0a085"), (0.55, "#aa9d84"), (0.80, "#af9e81"), (1.00, "#a99c85")])
    base = base * (0.945 + 0.11 * rnd(13))[..., None]                                # +-6 % per flag
    base = base * (1.0 + 0.13 * ltone + 0.06 * sat(rels / 0.004) - 0.06 * sat(-rels / 0.004))[..., None]   # upper laminae scuffed pale, lower ones hold the dirt
    base = base * (0.93 + 0.14 * cloud)[..., None]
    # iron staining in soft clouds, faint
    fe = smoothstep(0.40, 0.90, iron) * (0.15 + 0.10 * rnd(15))
    base = mix(base, base * hexlin("#b99a72") * 1.75, fe[..., None])
    # weathered grey skin / soot film in broad patches that ignore the joints
    skin = smoothstep(0.35, 0.75, up(lo.fbm(0.55, 3, 0.5, seed=62)))
    base = mix(base, luminance(base)[..., None] * hexlin("#aaa295") * 1.45, (skin * 0.06)[..., None])
    base = base * (0.95 + 0.10 * macro)[..., None]
    # sand grain
    xl = t.fbm(0.0045, 2, 0.6, seed=63)
    base = base * (0.91 + 0.18 * xl)[..., None]
    speck, sr = _dots(t, 26000, 0.0022, 64, 0.6)
    sclus = smoothstep(0.42, 0.70, up(lo.noise(0.13, seed=65)))
    base = mix(base, hexlin("#57493a"), (speck * sclus * (0.10 + 0.15 * sr))[..., None])
    # dirt: in hollows, at the foot of the scarps, in pits, along the low arris, around the cracks
    dirt = sat(hollow * (0.08 + 0.34 * xl) + foot * 0.16 + pit * 0.22 + (1 - ease) * 0.75 + near_w * 0.25 + hair * 0.15)
    base = mix(base, base * hexlin("#756b5b") * 2.0, dirt[..., None] * 0.60)
    base = base * (1.0 + 0.07 * polish)[..., None]                                  # walked crowns scuffed clean
    base = mix(base, base * 1.14 + 0.01, (sat(spall + chip) * 0.5)[..., None])       # fresher stone in spalls
    base = mix(base, hexlin("#3a342b"), sat(hair * 0.50 + wide * 0.85)[..., None])

    sandc = ramp(sand_fine, [(0.0, "#504a3f"), (0.45, "#70695b"), (0.8, "#8a8373"), (1.0, "#a09886")])
    deepj = sat((jdepth + 0.0085) / 0.007)
    sandc = sandc * (0.55 + 0.45 * deepj)[..., None]
    gcol = ramp(grit_r, [(0, "#4d483f"), (0.5, "#7c7568"), (1, "#a29b8a")])
    sandc = mix(sandc, gcol, grit[..., None] * 0.6)
    albedo = desaturate(mix(sandc, base, inside[..., None]), 0.18) * 0.80

    # ------------------------------------------------------------------ roughness
    r_stone = (0.82 - 0.27 * polish - 0.05 * foot_fall * ease + 0.10 * hollow + 0.10 * foot + 0.08 * dirt
               + 0.07 * (xl - 0.5) * (1 + polish) + 0.05 * (cloud - 0.5) + 0.06 * pit + 0.08 * sat(spall + chip))
    rough = mix(0.95 + 0.04 * (sand_fine - 0.5), r_stone, inside)
    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0)
