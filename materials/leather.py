"""Worn vegetable-tanned leather.

Construction: the grain side of a hide is a field of small rounded pebbles (2-3 mm, with a finer
~1.3 mm sub-grain) separated by soft valleys.  Use flexes the hide along sub-parallel wrinkle
families: a few deep flex creases 1-2 cm apart that run for 5-15 cm, and between them, in the
loose zones, fine parallel micro-wrinkles a few mm apart; the pebbles there are drawn out along
the wrinkles.  Handling burnishes whatever stands proud (darker, glossier) and leaves the hollows
dry, paler and matt.
"""
from tk.tex import *

META = {"normal_strength": 1.5, "ao_radius": 0.0025, "ao_strength": 0.8}

TARGET = (0.158, 0.071, 0.033)          # mean linear albedo (tan, ~#6f4c33)


def _family(t, kx, ky, seed, bend):
    """Wrinkle family = iso-lines of a warped plane wave (integer wave vector so it tiles).
    Returns phase (cycles) and the metric |grad phase| (cycles / m)."""
    w = (t.noise(0.16, seed=seed, bandwidth=0.5) - 0.5) * bend + (t.noise(0.07, seed=seed + 1) - 0.5) * bend * 0.5
    ph = kx * t.X / t.W + ky * t.Y / t.H + w
    gx, gy = t.slope(w)
    gx = gx + kx / t.W
    gy = gy + ky / t.H
    return ph, np.sqrt(gx * gx + gy * gy)


def _lined(ph, g):
    """Distance in metres to the nearest iso-line of the phase."""
    return np.abs(ph - np.round(ph)) / np.maximum(g, 1e-6)


def _pillow(t, lay, soft):
    """Rounded pebble profile 0..1 from a cell layout: dome over the whole cell (normalised by the
    cell's own in-radius) with a round-bottomed valley, not a plateau cut by hairlines."""
    d = t.blur(lay.dist, soft)
    mx = np.zeros(lay.n, dtype=np.float32)
    np.maximum.at(mx, lay.id.ravel(), d.ravel())
    q = sat(d / np.maximum(mx[lay.id], 1e-6))
    return smoothstep(0.0, 0.7, q) ** 0.85, q


def build(t):
    # ------------------------------------------------------------------ flex zones (macro)
    loose = smoothstep(0.38, 0.66, t.noise(0.12, seed=1, bandwidth=0.45))    # flexed / loose areas, evenly spread
    fam = smoothstep(0.40, 0.60, t.noise(0.15, seed=2, bandwidth=0.45))      # 0: family A (along U), 1: family B

    phA, gA = _family(t, 4, 22, 10, 2.2)                                     # creases ~18 mm apart, running ~along U
    phB, gB = _family(t, 22, -4, 20, 2.2)                                    # the crossing family, ~along V

    # ---- main flex creases: each line fades in and out along its own length (5-15 cm runs)
    def creases(ph, g, seed, st):
        d = _lined(ph, g)
        lid = np.round(ph)
        # per-line offset so neighbouring creases start and stop independently (even wave numbers
        # keep the line parity consistent across the tile seam)
        gate = t.noise(0.034, seed=seed, stretch=st, bandwidth=0.5)
        gate2 = t.noise(0.034, seed=seed + 1, stretch=st, bandwidth=0.5)
        gate = np.where(np.mod(lid, 2) == 0, gate, gate2)
        on = smoothstep(0.42, 0.56, gate)
        depth = on * (0.45 + 0.55 * smoothstep(0.5, 0.75, gate))
        sharp = np.exp(-(d / 0.00045) ** 1.4)                                 # the fold line itself
        soft = np.exp(-(d / 0.0017) ** 2)                                     # dished flanks
        roll = (np.cos(2 * np.pi * ph) * -0.5) * t.blur(depth, 0.004)           # soft rolls, crease in the trough
        return sharp * depth, soft * depth, roll

    a_sh, a_so, a_ro = creases(phA, gA, 30, (3.0, 1.0))
    b_sh, b_so, b_ro = creases(phB, gB, 40, (1.0, 2.6))
    wA = 1 - 0.75 * fam
    wB = 0.25 + 0.75 * fam
    dens = 0.55 + 0.45 * loose
    c_sharp = np.maximum(a_sh * wA, b_sh * wB * 0.9) * dens
    c_soft = np.maximum(a_so * wA, b_so * wB * 0.9) * dens
    bulge = t.blur(c_soft, 0.0045)
    roll = (a_ro * wA + b_ro * wB * 0.9) * dens
    crease_h = -0.00016 * c_sharp - 0.00020 * c_soft + 0.00030 * (t.blur(c_soft, 0.009) - bulge) + 0.00015 * roll

    # ---- micro-wrinkles: fine parallel flex lines inside the loose zones (2.5-4 mm apart)
    def micro(ph, seed, st, mult):
        p = ph * mult + (t.fbm(0.03, 2, 0.5, seed=seed) - 0.5) * 1.6
        c = np.cos(2 * np.pi * p) * 0.5 + 0.5
        runs = smoothstep(0.36, 0.58, t.noise(0.012, seed=seed + 1, stretch=st, bandwidth=0.5))
        return smoothstep(0.35, 1.0, c) ** 1.5 * runs

    mA = micro(phA, 50, (4.0, 1.0), 6)
    mB = micro(phB, 60, (1.0, 4.0), 6)
    mw = (mA * (1 - fam) + mB * fam) * (0.25 + 0.75 * loose)
    micro_h = -0.00011 * mw

    # ------------------------------------------------------------------ grain (meso / micro)
    # Three pebble fields: round (tight hide), drawn out along U, drawn out along V; the loose
    # zones use the one that follows their wrinkle family.
    iso = t.cells(0.0028, jitter=0.95, seed=4)
    alu = t.cells(0.0025, jitter=0.95, seed=5, aspect=1.7)
    alv = t.cells(0.0025 * 1.7, jitter=0.95, seed=6, aspect=1 / 1.7)
    p_iso, q_iso = _pillow(t, iso, 0.0003)
    p_alu, q_alu = _pillow(t, alu, 0.0003)
    p_alv, q_alv = _pillow(t, alv, 0.0003)
    wz = loose * 0.9
    w_u = wz * (1 - fam)
    w_v = wz * fam
    w_i = 1 - w_u - w_v
    p1 = p_iso * w_i + p_alu * w_u + p_alv * w_v
    pr = iso.rand(1) * w_i + alu.rand(1) * w_u + alv.rand(1) * w_v            # pebble-to-pebble random

    fine = t.cells(0.0013, jitter=0.9, seed=8)
    p2, _ = _pillow(t, fine, 0.00015)

    # valleys merge and split: their depth comes and goes, so pebbles fuse into short islands
    merge = t.fbm(0.006, 2, 0.5, seed=9)
    vdepth = 0.45 + 0.55 * smoothstep(0.25, 0.7, merge)
    pp = 1 - (1 - p1) * vdepth
    amp1 = (0.00024 + 0.00007 * loose) * (0.75 + 0.5 * pr)
    grain_h = amp1 * pp + 0.00006 * p2 * (0.35 + 0.65 * pp)
    grain_h = grain_h * (1 - 0.55 * c_sharp)                                 # grain is crushed flat in the fold

    # ------------------------------------------------------------------ macro + micro
    und = (t.fbm(0.11, 2, 0.5, seed=12) - 0.5) * 0.0006                      # the hide never lies flat
    tooth = (t.fbm(0.0012, 2, 0.6, seed=13) - 0.5) * 0.000012
    relief = crease_h + micro_h
    height = und + relief + grain_h + tooth

    # ------------------------------------------------------------------ masks
    proud = t.blur(relief, 0.0012) - t.blur(relief, 0.010)                   # > 0: bulges between creases
    high = smoothstep(-0.00002, 0.00009, proud)
    hollow = smoothstep(0.00002, -0.00012, proud)
    handled = smoothstep(0.30, 0.70, t.noise(0.10, seed=14, bandwidth=0.5))   # where hands / straps rub
    valley = (1 - pp) * 0.75 + (1 - p2) * 0.25 * (0.3 + 0.7 * pp)
    top = smoothstep(0.55, 1.0, pp) * (0.7 + 0.3 * p2)
    burn = sat(high * (0.45 + 0.55 * handled)) * (0.35 + 0.65 * top)          # burnished, oil-darkened
    dry = sat(hollow * 0.8 + (1 - handled) * 0.35)                            # dry, matt, a little paler

    scr = t.scratches(density=140, length=0.03, width=0.0005, seed=16)
    nick = t.scratches(density=420, length=0.006, width=0.0007, seed=17)
    scuff = sat(scr * 0.7 * (0.4 + 0.6 * pp) + nick * 0.5 * (0.3 + 0.7 * dry))

    # ------------------------------------------------------------------ colour
    tone = 0.35 * t.noise(0.09, seed=19, bandwidth=0.5) + 0.65 * t.fbm(0.02, 3, 0.5, seed=18)
    base = ramp(level(tone), [(0.0, "#72492f"), (0.5, "#7a4f33"), (1.0, "#82573a")])
    mott = t.fbm(0.004, 3, 0.55, seed=20)                                    # tannin take-up, pebble scale
    base = base * (0.95 + 0.10 * mott)[..., None]
    base = base * (0.95 + 0.10 * pr)[..., None]
    base = base * (1 - 0.15 * valley)[..., None]                             # a little dirt in the grain valleys
    base = base * (1 - 0.22 * sat(c_sharp * 0.8 + c_soft * 0.45) - 0.07 * mw)[..., None]   # creases hold dirt
    base = base * (1 - 0.13 * burn)[..., None]                               # burnish darkens
    base = mix(base, desaturate(base, 0.2) * 1.10 + 0.003, dry * 0.4)       # dry hollows paler, greyer
    shoulder = sat(bulge * 2.4 - c_soft * 1.6) * top                         # abraded shoulders of the creases
    base = mix(base, base * 1.16 + 0.006, shoulder * 0.45)
    base = mix(base, hexlin("#a98562"), scuff * 0.5)                         # pale flesh-coloured scuffs
    mean = base.reshape(-1, 3).mean(0)
    albedo = np.clip(base * (np.array(TARGET, dtype=np.float32) / mean), 0.012, 0.9)

    # ------------------------------------------------------------------ roughness
    rough = 0.50 - 0.20 * burn + 0.14 * dry + 0.16 * valley - 0.09 * top * handled - 0.08 * (handled - 0.5)
    rough = rough + 0.22 * scuff + 0.10 * sat(c_soft) + 0.05 * (mott - 0.5)
    rough = np.clip(rough, 0.28, 0.92)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
