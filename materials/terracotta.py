"""Unglazed terracotta (biscuit-fired red earthenware).

Construction: an open, sandy clay body tempered with grog - the surface is dry, gritty and
slightly pitted, with pale and dark grains standing just proud and burnt-out pores between them.
The wheel left a few broad, soft finger rings (along U) and, where the potter tooled the wall,
patches of fine rib lines; elsewhere the wall was sponged smooth and they fade out.  Knocks have
spalled off small shell-shaped flakes (sharp lip on one side, feathering out on the other, fresh
clay inside).  Salts have dried on the surface as a thin crusty lime bloom with tide-line rims,
sitting in the pores and ring valleys.
"""
from tk.tex import *

META = {"normal_strength": 1.4, "ao_radius": 0.003, "ao_strength": 0.9}

TARGET = (0.452, 0.205, 0.103)          # mean linear albedo (~#b47c5a), buff-orange, not hot


def _grains(t, cell, frac, seed, margin=0.25):
    """Angular crushed grains: a fraction of the cells of a Voronoi layout, each shrunk inside its
    own polygon (convex, sharp-cornered, never touching).  Returns (mask 0..1, per-grain rand)."""
    lay = t.cells(max(cell, t.px * 4), 1.0, seed=seed)
    mx = np.zeros(lay.n, dtype=np.float32)
    np.maximum.at(mx, lay.id.ravel(), lay.dist.ravel())
    inr = mx[lay.id]                                                     # in-radius of the cell
    keep = (lay.rand(1) < frac).astype(np.float32)
    m = margin + (1 - margin) * 0.6 * lay.rand(2)                        # how much of the cell the grain fills
    q = (lay.dist - m * inr) / np.maximum(t.px * 1.1, 1e-6)
    return (sat(q) * keep).astype(np.float32), lay.rand(5)


def build(t):
    # ------------------------------------------------------------------ wheel marks (meso / macro)
    sponged = smoothstep(0.40, 0.62, t.noise(0.13, seed=1, bandwidth=0.5))   # wiped smooth here
    # ~10 broad finger rings per tile, pitch drifting, each ring wandering a little round the pot
    drift = (t.noise(0.16, seed=2, stretch=(5.0, 1.0)) - 0.5) * 1.5 + (t.noise(0.30, seed=3) - 0.5) * 0.7
    fph = 10 * t.Y / t.H + drift
    finger = np.cos(2 * np.pi * fph + 0.55 * np.sin(2 * np.pi * fph)) * 0.5 + 0.5   # soft ridge, steeper on one flank
    f_amp = 0.35 + 0.65 * smoothstep(0.3, 0.7, t.noise(0.11, seed=4, stretch=(3.0, 1.0), bandwidth=0.5))
    # fine rib / tool lines: only in the tooled patches (< 40 % of the wall), fading at the edges
    rph = 58 * t.Y / t.H + drift * 5.8 + (t.fbm(0.05, 2, 0.5, seed=5, stretch=(8.0, 1.0)) - 0.5) * 1.2
    rib = np.cos(2 * np.pi * rph) * 0.5 + 0.5
    tooled = smoothstep(0.50, 0.64, t.noise(0.085, seed=6, stretch=(2.2, 1.0), bandwidth=0.5)) * (1 - 0.5 * sponged)
    ring_h = (finger - 0.5) * 0.00050 * f_amp + (rib - 0.5) * 0.00019 * tooled

    # ------------------------------------------------------------------ clay body (micro)
    lump = (t.fbm(0.035, 3, 0.5, seed=7) - 0.5) * 0.00045                    # uneven wall
    sandy = t.fbm(0.0030, 3, 0.62, seed=8)                                   # open sandy tooth, 1-3 mm
    fine = t.fbm(0.0012, 2, 0.6, seed=9)
    slick = 1 - 0.45 * sponged                                               # sponging lays the sand down a bit
    body_h = (sandy - 0.5) * 0.00056 * slick + (fine - 0.5) * 0.00014

    grog_a, ga = _grains(t, 0.0034, 0.15, 10)                                # coarse grog 1.5-3 mm
    grog_b, gb = _grains(t, 0.0020, 0.28, 12)                                # fine grit ~1 mm
    grog = np.maximum(grog_a, grog_b * 0.7)
    grnd = np.where(grog_a > grog_b * 0.7, ga, gb)
    pore_s = t.speckle(density=26000, size=0.0014, seed=14, soft=0.6)        # burnt-out organics
    pore_l = t.warp(t.speckle(density=1500, size=0.0032, seed=15, soft=0.55), 0.0012, 0.004, seed=16)
    pore = sat(np.maximum(pore_s * 0.8, pore_l)) * (1 - grog)
    drag = t.scratches(density=150, length=0.022, width=0.0011, seed=17, angle=0, angle_spread=10)  # grit dragged by the rib
    drag = drag * (0.4 + 0.6 * sandy)
    body_h = body_h + grog_a * 0.00017 + grog_b * 0.00008 - pore_s * 0.00020 * (1 - grog) - pore_l * 0.00042 * (1 - grog) \
        - drag * 0.00020

    # ------------------------------------------------------------------ spalls
    # Shell-shaped flakes knocked off: struck at one side (sharp, steep lip, deepest there),
    # running out to a feather edge on the other.
    lay = t.cells(0.0625, jitter=0.85, seed=20)
    present = (lay.rand(1) < 0.30).astype(np.float32)
    R = 0.0012 + 0.0085 * lay.rand(2) ** 1.7                                  # 2-19 mm across
    th = lay.rand(3) * 6.283
    cu, su = np.cos(th), np.sin(th)
    al = (lay.u * cu + lay.v * su) / R                                       # -1 lip ... +1 feather edge
    ac = (-lay.u * su + lay.v * cu) / R
    phi = np.arctan2(ac, al)
    lobes = 1 + 0.16 * np.cos(3 * phi + lay.rand(4) * 6.283) + 0.10 * np.cos(5 * phi + lay.rand(5) * 6.283)
    wob = 1 + 0.3 * (t.fbm(0.007, 2, 0.5, seed=21) - 0.5)
    rr = np.sqrt(al * al + (ac * 1.25) ** 2) / (lobes * wob)                 # warped, non-circular outline
    side = sat(0.5 + 0.75 * al)                                              # 0 at the lip, 1 at the feather
    ew = mix(0.10, 0.95, smoothstep(0.0, 1.0, side))
    bowl = sat((1 - rr) / ew) * present
    bowl = bowl * bowl * (3 - 2 * bowl)
    chip = (1 - smoothstep(0.86, 1.0, rr)) * present                         # footprint
    depth = (0.00045 + 0.00075 * lay.rand(6)) * (0.35 + 0.65 * sat(R / 0.006))
    floor = (t.fbm(0.0022, 3, 0.65, seed=22) - 0.5) * 0.00022                # torn granular floor
    dl = np.sqrt((al + 0.85) ** 2 + ac * ac) * R                             # distance from the point of impact
    shell = np.cos(2 * np.pi * dl / (0.0011 + 0.0012 * lay.rand(7))) * 0.00004   # conchoidal ripples
    chip_h = -bowl * depth + (floor + shell) * sat(bowl * 1.5)
    lip = sat(bowl * 4 * (1 - bowl)) * (1 - side)                            # the steep struck wall

    height = (ring_h + lump + body_h) * (1 - 0.8 * sat(bowl * 1.5)) + chip_h

    # ------------------------------------------------------------------ masks
    hl = t.blur(height, 0.0010)
    cav = t.cavity(hl, 0.004, 0.00022)
    high = t.edges(hl, 0.004, 0.00022) * (1 - chip)
    ringlow = sat(0.5 - finger) * 2 * f_amp
    inchip = sat(bowl * 2.5)

    # ------------------------------------------------------------------ colour
    # fire flash: broad, non-directional buff vs red-brown; plus body mottling at grain scale
    fire = level(0.5 * t.fbm(0.22, 3, 0.5, seed=30) + 0.5 * t.fbm(0.05, 3, 0.55, seed=31))
    base = ramp(0.2 + 0.6 * fire, [(0.0, "#a4633f"), (0.35, "#b06e46"), (0.7, "#ba7a52"), (1.0, "#c38a63")])
    base = base * (0.90 + 0.20 * sandy)[..., None] * (0.97 + 0.06 * fine)[..., None]
    # grog: pale (fired fireclay) and dark (over-fired / ironstone) grains
    gcol = ramp(grnd, [(0.0, "#6b4636"), (0.3, "#8a5a44"), (0.5, "#b47a56"), (0.72, "#c49a72"), (0.9, "#d0b692"), (1.0, "#dccdb4")])
    base = mix(base, gcol * (0.9 + 0.2 * fine)[..., None], grog[..., None] * 0.5)
    iron = t.speckle(density=9000, size=0.0012, seed=32, soft=0.5)           # iron / manganese specks
    base = mix(base, hexlin("#5e3f30"), iron * 0.45)
    base = base * (1 - 0.34 * pore - 0.10 * cav - 0.14 * drag)[..., None]    # darker pore floors
    base = mix(base, base * 1.07 + 0.006, high * 0.45)                        # rubbed high spots, dust-free

    # lime bloom: crusty haze from thresholded fine noise; bright tide-line rim, fainter inside;
    # gathers in pores, ring valleys and the lee of lumps
    bn = t.fbm(0.022, 4, 0.62, seed=33)
    hb = t.blur(height * (1 - chip), 0.003)
    low = sat(-t.curv(hb, 0.012) / 0.00025)
    where = 0.06 * (t.noise(0.06, seed=34, bandwidth=0.5) - 0.5) + 0.07 * low + 0.07 * ringlow
    bn = bn + where
    thr = float(np.percentile(bn, 84.0))
    inner = smoothstep(thr, thr + 0.05, bn)
    rim = np.exp(-((bn - thr - 0.008) / 0.011) ** 2)
    crust = 0.55 + 0.45 * smoothstep(0.35, 0.7, t.fbm(0.003, 2, 0.6, seed=35))
    freck = smoothstep(0.62, 0.8, t.fbm(0.0045, 2, 0.6, seed=36)) * inner
    powder = 0.45 + 0.55 * smoothstep(0.75, 0.3, sandy)                      # powder sits in the sandy tooth
    bloom = sat(rim * 1.0 + inner * 0.40 * crust + freck * 0.5) * powder * (1 - inchip) * (1 - 0.5 * grog) * (1 - 0.5 * high)
    bloom = t.blur(bloom, 0.0006) * (0.75 + 0.5 * fine)                      # powdery, not flaky
    base = mix(base, hexlin("#e0d3bd"), sat(bloom) * 0.48)

    # spall interior: fresh clay, a little more saturated, no bloom; dark under the struck lip
    fresh = ramp(sat(0.5 + (sandy - 0.5) * 1.6), [(0.0, "#b0683e"), (0.5, "#c0774a"), (1.0, "#cc8a5c")])
    fresh = mix(fresh, gcol, grog[..., None] * 0.45)
    base = mix(base, fresh, inchip[..., None] * 0.9)
    base = base * (1 - 0.30 * lip - 0.10 * inchip * (1 - side))[..., None]
    edge = sat(chip - inchip) * (1 - side)                                   # bruised pale rim at the lip
    base = mix(base, base * 1.10 + 0.01, edge * 0.5)

    mean = base.reshape(-1, 3).mean(0)
    albedo = np.clip(base * (np.array(TARGET, dtype=np.float32) / mean), 0.02, 0.85)

    # ------------------------------------------------------------------ roughness
    rough = 0.875 + 0.08 * (sandy - 0.5) + 0.07 * pore + 0.05 * cav - 0.03 * sponged
    rough = rough - 0.17 * high * (1 - bloom) - 0.15 * grog * (1 - bloom) + 0.09 * bloom + 0.05 * inchip
    rough = np.clip(rough, 0.66, 0.97)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
