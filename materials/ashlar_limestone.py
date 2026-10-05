"""Coursed ashlar: dressed pale limestone blocks, fine lime joints, a century of weather.

Built like the wall: seven courses of differing height in a 2 m lift (each course one height,
as it came off the quarry bed), blocks 0.45-0.9 m long with perpends well clear of those in the
course below. Every block was dressed by a banker mason and finished with the punch: short pecks a
centimetre long lying every way (some blocks rubbed nearly smooth, some coarsely punched, about
half with a chisel-drafted margin) - nothing brushed, nothing in parallel rows. Weather has opened
the faces: lumpy differential erosion, softer beds sunk and pitted, the hard skin scaled off in
shallow sharp-edged patches, arrises knocked and corners lost. Joints are recessed 6-10 mm, the
pointing dark with dirt and weathered out in places, with a dirt wash on the stone beside them and
grey crust sheltered under every bed joint. The stone is a shelly cream limestone laid on its
natural bed; blocks differ visibly in tone (grey, cream, buff) as they came from different beds.
"""
from tk.tex import *

import math

META = {"normal_strength": 1.0, "ao_strength": 1.25}
MEAN = (0.5394, 0.4580, 0.3216)          # the town is balanced on this mean: keep it

COURSES = (0.30, 0.25, 0.36, 0.27, 0.23, 0.33, 0.26)        # metres, bottom to top (sum 2.0)


def coursed(t, seed=11):
    """Coursed-ashlar layout -> Layout plus per-pixel half length / half height of the block."""
    rng = np.random.default_rng(seed)
    k = t.H / sum(COURSES)
    hs = np.array(COURSES) * k
    y_edges = np.concatenate([[0.0], np.cumsum(hs)])
    rows = len(hs)
    # block boundaries per course (absolute x in 0..W), perpends >= 0.14 m from the course below
    bounds = []
    for i in range(rows):
        nb = 3 if hs[i] > 0.28 * k else (4 if rng.random() < 0.6 else 3)
        best, best_clear = None, -1.0
        for _ in range(300):
            ln = 1.0 + (rng.random(nb) - 0.5) * 0.6          # +-30 % on the mean block length
            ln = ln / ln.sum() * t.W
            cuts = (rng.random() * t.W + np.concatenate([[0.0], np.cumsum(ln)[:-1]])) % t.W
            cuts = np.sort(cuts)
            clear = 9.0
            for prev in ([bounds[i - 1]] if i > 0 else []) + ([bounds[0]] if i == rows - 1 else []):
                dd = np.abs(cuts[:, None] - prev[None, :])
                clear = min(clear, float(np.minimum(dd, t.W - dd).min()))
            if clear > best_clear:
                best, best_clear = cuts, clear
            if clear > 0.16:
                break
        bounds.append(best)
    r = np.clip(np.searchsorted(y_edges, t.Y, side='right') - 1, 0, rows - 1)
    cid = np.zeros(t.X.shape, dtype=np.int64)
    u = np.zeros(t.X.shape, dtype=F)
    hw = np.zeros(t.X.shape, dtype=F)
    base = 0
    for i in range(rows):
        m = r == i
        cuts = bounds[i]
        xs = (t.X[m] - cuts[0]) % t.W
        e = np.concatenate([cuts - cuts[0], [t.W]])
        ci = np.clip(np.searchsorted(e, xs, side='right') - 1, 0, len(cuts) - 1)
        u[m] = xs - (e[ci] + e[ci + 1]) / 2
        hw[m] = (e[ci + 1] - e[ci]) / 2
        cid[m] = base + ci
        base += len(cuts)
    hh = (hs[r] / 2).astype(F)
    v = (t.Y - (y_edges[r] + y_edges[r + 1]) / 2).astype(F)
    dist = np.minimum(hw - np.abs(u), hh - np.abs(v))
    return Layout(dist, cid, u, v, base, seed, 0.66, 0.28), hw, hh


def pecks(t, density, length, width, seed, keep=0.6):
    """Punch / peck marks: short elongated dabs lying every way. One candidate per jittered grid
    cell, a fraction `keep` exists. Returns profile 0..1 (soft V section)."""
    cell = max(1.0 / math.sqrt(density), length * 0.75, t.px * 4)
    nx, ny = max(2, int(round(t.W * 0.5 / cell))), max(2, int(round(t.H * 0.5 / cell)))
    cw, ch = t.W * 0.5 / nx, t.H * 0.5 / ny
    rng = np.random.default_rng(t.seed * 1000 + 5100 + seed)
    jx, jy, ra, rl, rk, rw = (rng.random((ny, nx)).astype(F) for _ in range(6))
    X, Y = t.X[:t.h // 2, :t.w // 2], t.Y[:t.h // 2, :t.w // 2]     # periodic in half a tile: compute a quarter
    gx = np.floor(X / cw).astype(np.int64)
    gy = np.floor(Y / ch).astype(np.int64)
    out = np.zeros(X.shape, dtype=F)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = gx + ox, gy + oy
            wx, wy = cx % nx, cy % ny
            dx = X - (cx + jx[wy, wx]) * cw
            dy = Y - (cy + jy[wy, wx]) * ch
            a = ra[wy, wx] * math.pi
            ca, sa = np.cos(a), np.sin(a)
            al = dx * ca + dy * sa
            ac = -dx * sa + dy * ca
            L = length * (0.45 + 0.8 * rl[wy, wx])
            Wd = width * (0.6 + 0.7 * rw[wy, wx])
            u = al / L + 0.5
            prof = smoothstep(0.0, 0.2, u) * (1 - smoothstep(0.35, 1.0, u))
            m = prof * np.clip(1 - np.abs(ac) / (Wd * 0.5), 0, 1) * (rk[wy, wx] < keep) * (0.55 + 0.45 * rl[wy, wx])
            out = np.maximum(out, m)
    return np.tile(out.astype(F), (2, 2))


def build(t):
    lay, hw, hh = coursed(t)
    R = lay.rand
    u, v = lay.u, lay.v

    # ------------------------------------------------------------------ block outlines
    # dressed blocks: straight and square, fractionally out of true; arrises eased by weather
    ang = (R(1) - 0.5) * 0.005
    vv = v + ang * u + (R(2) - 0.5) * 0.0015
    uu = u + (R(3) - 0.5) * 0.002
    gap = 0.0026 + 0.0012 * R(4)                             # half joint: 5 - 7.5 mm joints
    d = np.minimum(hw - np.abs(uu), hh - np.abs(vv)) - gap   # metres inside the block face
    d = d + (t.noise(0.06, seed=1) - 0.5) * 0.002 + (t.noise(0.015, seed=16) - 0.5) * 0.001
    # knocked arrises and the odd lost corner
    corner = (1 - smoothstep(0.0, 0.06, hw - np.abs(uu))) * (1 - smoothstep(0.0, 0.06, hh - np.abs(vv)))
    chipn = t.fbm(0.06, 3, 0.5, seed=2)
    chip = smoothstep(0.68, 0.80, chipn + 0.20 * corner + 0.08 * (R(5) - 0.5))
    near = 1 - smoothstep(0.004, 0.030, d)
    chipped = chip * near
    d_out = d - chip * 0.010 * near                          # outline after damage
    inside = smoothstep(-0.0008, 0.0012, d_out)
    inside_j = smoothstep(-0.0008, 0.0012, d)                # the joint proper (chips are stone, not mortar)
    arris = 0.007 + 0.007 * R(6)                             # weather-rounded arris
    prof = sat((d_out + 0.0005) / arris)
    prof = 1 - (1 - prof) ** 2.4

    # ------------------------------------------------------------------ tooling
    # every block was finished by hand with the punch and then rubbed; the finish differs from
    # block to block:  rubbed (pecks all but gone) / punched (fine pecks) / coarse punched.
    # Pecks lie every way: no furrows, no rows, nothing that can read as brushing or wood grain.
    ox, oy = R(7) * t.W, R(8) * t.H                          # decorrelate neighbouring blocks
    kind = np.floor(R(9) * 10).astype(np.int64)              # 0-3 rubbed  4-7 punched  8-9 coarse punched
    pk_f = np.maximum(pecks(t, 2600, 0.013, 0.0042, 1, keep=0.62), 0.6 * pecks(t, 5200, 0.007, 0.0030, 2, keep=0.45))
    pk_c = np.maximum(pecks(t, 1100, 0.022, 0.0075, 3, keep=0.7), 0.5 * pk_f)
    pk_f = t.sample(t.warp(pk_f, 0.0012, 0.008, seed=40), t.X + ox, t.Y + oy)
    pk_c = t.sample(t.warp(pk_c, 0.0020, 0.012, seed=41), t.X + oy, t.Y + ox)
    peck = np.where(kind <= 3, pk_f * 0.30, np.where(kind <= 7, pk_f * 0.85, pk_c)).astype(F)
    tool = (0.5 - 0.5 * peck).astype(F)
    # the punch bit harder in some patches; weather has softened the work in others
    passes = t.sample(t.noise(0.09, seed=5), t.X + oy, t.Y + ox)
    soften = t.fbm(0.18, 3, 0.5, seed=18)
    tool_amp = (0.55 + 0.6 * passes) * (0.6 + 0.6 * R(10)) * (1.2 - 0.95 * smoothstep(0.38, 0.72, soften))
    # drafted margin: a 22-30 mm band round the arris chiselled smooth
    mw = 0.022 + 0.008 * R(11)
    has_margin = (R(12) > 0.45).astype(F)
    panel = smoothstep(mw - 0.003, mw + 0.004, d + (t.noise(0.025, seed=6) - 0.5) * 0.006)
    panel = mix(1.0, panel, has_margin)

    # ------------------------------------------------------------------ stone surface
    bed = t.sample(t.fbm(0.045, 3, 0.55, seed=8, stretch=(4.0, 1.0)), t.X + ox, t.Y + oy)    # natural bedding: faint, broken
    undul = t.sample(t.fbm(0.20, 3, 0.5, seed=9), t.X + oy, t.Y + ox)       # faces are never truly flat
    grain = t.fbm(0.007, 3, 0.6, seed=10)
    # shell voids, strung out along the softer beds
    softbed = smoothstep(0.58, 0.8, bed)
    vugs = t.speckle(density=260, size=0.009, seed=12) * softbed
    pores = t.speckle(density=5000, size=0.003, seed=14) * smoothstep(0.45, 0.75, bed)
    # shallow weathering scallops where a softer bed has let go
    scal_n = t.fbm(0.12, 2, 0.4, seed=15)
    scallop = smoothstep(0.80, 0.90, scal_n + 0.10 * (bed - 0.5))

    # weathering relief: lumpy differential erosion (1-4 cm) and shallow scaled patches where the
    # hard skin has let go - a crisp ragged lip 0.4-0.9 mm high.  Per block (never across a joint).
    ero = (t.fbm(0.034, 4, 0.55, seed=50) - 0.5) * 0.0024 + (t.fbm(0.012, 3, 0.6, seed=51) - 0.5) * 0.0011
    ero = t.sample(ero, t.X + ox, t.Y + oy)
    sk_n = t.fbm(0.15, 4, 0.55, seed=52) + (t.fbm(0.02, 3, 0.6, seed=53) - 0.5) * 0.16
    sk_n = t.sample(sk_n, t.X + oy, t.Y + ox) + (R(25) - 0.5) * 0.10          # some blocks scale more
    thr = np.percentile(sk_n, 80)
    scaled = smoothstep(thr, thr + 0.012, sk_n)
    sk2 = t.sample(t.fbm(0.055, 3, 0.55, seed=54) + (t.fbm(0.012, 2, 0.6, seed=55) - 0.5) * 0.14, t.X + ox, t.Y + oy)
    scaled2 = smoothstep(np.percentile(sk2, 90), np.percentile(sk2, 90) + 0.012, sk2)
    sc_floor = (t.fbm(0.009, 3, 0.65, seed=56) - 0.5) * 0.0009
    scar = sat(scaled + scaled2 * 0.6)

    lvl = (R(13) - 0.5) * 0.0046 + lay.tilt(0.0022, 2)       # blocks sit a little proud / shy / canted
    belly = (R(19) - 0.35) * 0.0022 * (1 - (uu / hw) ** 2) * (1 - (vv / hh) ** 2)
    face = (lvl + belly + (undul - 0.5) * 0.0030
            + panel * (0.0005 * has_margin + (tool - 0.5) * 0.0020 * tool_amp * (1 - scaled))
            + ero - scaled * (0.0006 + 0.0003 * R(26) + sc_floor) - scaled2 * 0.00035
            - softbed * 0.0006 + (bed - 0.5) * 0.0006 + (grain - 0.5) * 0.0006
            - vugs * 0.0028 - pores * 0.0008
            - scallop * (0.0014 + 0.0010 * grain))
    face = face - chipped * (0.004 + 0.004 * chipn)          # chip scars

    # ------------------------------------------------------------------ joints
    jgrit = t.fbm(0.006, 3, 0.65, seed=20)
    lost = smoothstep(0.50, 0.72, t.fbm(0.30, 3, 0.55, seed=21))     # pointing weathered out in places
    joint_h = -0.0075 - 0.0070 * lost + (jgrit - 0.5) * 0.0014
    height = np.where(inside > 0, mix(joint_h, joint_h + (face - joint_h) * prof, inside), joint_h)

    cav = t.cavity(height, 0.004, 0.0006)
    cav_w = t.cavity(height, 0.014, 0.0014)
    edg = t.edges(height, 0.006, 0.0008)

    # ------------------------------------------------------------------ colour
    # block to block the stone differs visibly: beds of greyer, creamier and buff stone, and each
    # block a little lighter or darker than its neighbours
    tone = R(14)
    stone = ramp(tone, [(0.00, "#aea591"),    # greyer block
                        (0.22, "#c2b69d"),
                        (0.50, "#d1c5ab"),    # the common cream
                        (0.78, "#d8c9a6"),    # warmer
                        (1.00, "#c9b48e")])   # buff, iron-rich bed
    stone = stone * (0.90 + 0.19 * R(27))[..., None]
    mott = t.sample(t.fbm(0.12, 4, 0.55, seed=30), t.X + ox, t.Y + oy)
    fine = t.fbm(0.02, 3, 0.6, seed=31)
    val = 1.0 + (mott - 0.5) * 0.24 + (fine - 0.5) * 0.12 + (bed - 0.5) * 0.08 + (grain - 0.5) * 0.08 + scar * 0.07 - (1 - scar) * 0.02
    stone = stone * val[..., None]
    # iron staining follows the bedding in soft clouds
    iron = smoothstep(0.50, 0.85, t.fbm(0.18, 3, 0.5, seed=32) * 0.65 + bed * 0.35)
    stone = mix(stone, stone * hexlin("#dcb888") * 1.42, iron * 0.5)
    # shell fragments: paler crystalline flecks, a few darker
    shell = t.speckle(density=14000, size=0.004, seed=33)
    stone = mix(stone, hexlin("#ded8c8"), shell * 0.35)
    stone = mix(stone, hexlin("#938a78"), np.roll(shell, (t.h // 3, t.w // 5), axis=(0, 1)) * 0.30)
    # fresh stone in chip scars and scallops is paler and cleaner
    stone = mix(stone, hexlin("#d6ccb8") * (0.9 + 0.2 * grain)[..., None], sat(chipped * 0.6 + scallop * 0.4))
    # bruised stone in the peck marks is a touch paler
    stone = mix(stone, stone * 1.06 + 0.004, sat(peck * panel * 1.2) * 0.5 * (1 - scaled))
    # grey weathering crust: soft patches, and sheltered just under each bed joint
    crust_n = t.fbm(0.35, 4, 0.55, seed=34)
    under = (1 - smoothstep(0.0, 0.09, hh - vv)) * smoothstep(0.3, 0.7, t.drips(0.06, 5.0, seed=35))
    crust = sat(smoothstep(0.42, 0.85, crust_n) * 0.55 + under * 0.75 + 0.25 * (R(20) - 0.5)) * (1 - 0.7 * scallop) * (1 - 0.6 * scar)
    stone = mix(stone, stone * hexlin("#a39d94") * 1.55, crust * 0.5)
    # crustose lichen: small pale grey rosettes, a few ochre, in loose colonies
    lich = t.speckle(density=260, size=0.030, seed=36, soft=0.35)
    lich = lich * smoothstep(0.5, 0.68, t.fbm(0.3, 3, 0.5, seed=37)) * (0.6 + 0.4 * fine)
    lcol = mix(hexlin("#c9c8bc"), hexlin("#b9a468"), smoothstep(0.6, 0.7, t.noise(0.11, seed=38)))
    stone = mix(stone, lcol * (0.85 + 0.3 * grain)[..., None], lich * 0.5)
    # black micro-lichen / soot specks peppering the old skin
    spk = t.speckle(density=9000, size=0.0034, seed=39, soft=0.6)
    spk = spk * smoothstep(0.40, 0.66, t.fbm(0.08, 3, 0.55, seed=42)) * (1 - scar)
    stone = mix(stone, hexlin("#6b6558"), spk * 0.22)
    # dirt lodged in the peck marks, pores and voids; a dirt wash on the stone beside every joint
    # (rain carries it out of the joint and down the arris) - the joints read from across the square
    stone = stone * (1 - 0.15 * cav - 0.08 * cav_w)[..., None]
    wash_n = t.fbm(0.05, 3, 0.55, seed=43)
    wash = (1 - smoothstep(0.0, 0.016 + 0.050 * wash_n, d_out)) ** 1.5 * (0.15 + 0.85 * smoothstep(0.35, 0.7, t.fbm(0.22, 2, 0.5, seed=44)))
    below = (1 - smoothstep(0.0, 0.05 + 0.10 * smoothstep(0.35, 0.75, t.drips(0.05, 6.0, seed=45)), hh - vv))    # runs under the bed joint
    stone = mix(stone, stone * hexlin("#8f8676") * 1.25, sat(wash * 0.50 + below * 0.30))
    stone = mix(stone, stone * 1.06 + 0.01, edg * 0.45)

    mortar = ramp(jgrit, [(0.0, "#5f594d"), (0.5, "#7d7667"), (1.0, "#9a9382")])      # old lime pointing, dirty
    mortar = mortar * (1 - 0.40 * lost)[..., None]            # open joints hold dirt
    albedo = mix(mortar, stone, inside_j)
    albedo = albedo * (np.array(MEAN, dtype=F) / albedo.reshape(-1, 3).mean(0))
    albedo = np.clip(albedo, 0.02, 0.85)

    # ------------------------------------------------------------------ roughness
    rs = (0.76 + (R(15) - 0.5) * 0.14 - 0.08 * smoothstep(0.45, 0.8, soften) + panel * (0.5 - tool) * 0.14 * tool_amp - (1 - panel) * 0.08
          + (fine - 0.5) * 0.10 + 0.10 * crust + 0.10 * cav - 0.10 * edg + 0.10 * chipped + 0.08 * scallop + 0.07 * scar)
    rough = mix(0.93, rs, inside_j)
    return dict(albedo=albedo, height=height, rough=np.clip(rough, 0.3, 1.0), metal=0.0)
