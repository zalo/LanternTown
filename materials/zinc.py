"""Dull grey hot-dip galvanised / zinc sheet.

The zinc coat freezes into spangle: polygonal crystals 8-25 mm across, each grown from a nucleus
as six feathery sectors whose facets lie at different angles - so every sector has its own
brightness and gloss. Over that: the dull blue-grey carbonate patina that kills the shine in
patches, chalky white rust (bloom) where water has sat, dross pimples, handling scratches and
the slight buckle of thin sheet.
"""
from tk.tex import *

META = {"normal_strength": 1.6}
BLOOM = 0.85


def hash01(a, b):
    x = np.sin(a * 12.9898 + b * 78.233 + 1.2345) * 43758.5453
    return (x - np.floor(x)).astype(F)


def build(t):
    # ---------------------------------------------------------------- spangle crystals
    g = t.cells(0.022, 1.0, seed=1)
    gid = g.id.astype(np.float64)
    a0 = g.rand(1) * (math.pi / 3)
    th = np.arctan2(g.v, g.u) - a0
    sec_f = th / (math.pi / 3)
    k = np.floor(sec_f)
    frac = sec_f - k                                                   # 0..1 across one sector
    kk = np.mod(k, 6)
    s_rand = hash01(gid, kk)                                           # facet brightness per sector
    s_rand2 = hash01(gid + 31.7, kk)
    phi = a0 + (kk + 0.5) * (math.pi / 3)                              # sector axis (dendrite stem)
    al = g.u * np.cos(phi) + g.v * np.sin(phi)
    ac = -g.u * np.sin(phi) + g.v * np.cos(phi)
    # feather: side arms leave the stem at 60 deg on either side -> herringbone streaks
    side = np.where(ac >= 0, 1.0, -1.0)
    fa = al * 0.5 + side * ac * 0.866
    fc = -al * 0.866 * side + ac * 0.5
    ox = g.rand(5) * 0.37 + kk * 0.011
    oy = g.rand(6) * 0.41
    streak_src = t.fbm(0.0030, 2, 0.5, seed=2, stretch=(7.0, 1.0))
    streak = t.sample(streak_src, fa + ox * t.W, fc + oy * t.H)
    stem = np.exp(-(np.abs(ac) / 0.0009) ** 2) * sat(al / 0.004)       # the six primary arms
    gb = 1 - smoothstep(0.0, 0.0010, g.dist)                           # grain boundary
    sb = np.exp(-(np.minimum(frac, 1 - frac) * g.f1 * 1.047 / 0.0006) ** 2) * sat(g.f1 / 0.003)  # sector boundary

    facet = (s_rand - 0.5) * 2                                         # -1..1
    facet = t.blur(facet, 0.0005)
    tone = sat(0.5 + 0.34 * facet + 0.14 * (g.rand(2) - 0.5) * 2 + 0.26 * (streak - 0.5) * 2 + 0.08 * stem)

    # ---------------------------------------------------------------- sheet shape and surface relief
    buckle = (t.fbm(0.22, 2, 0.5, seed=3) - 0.5) * 0.00040              # oil-canning of thin sheet
    dn = t.cells(0.085, 1.0, seed=4)
    dent = (dn.rand(1) < 0.45) * np.exp(-(dn.f1 / (0.010 + 0.012 * dn.rand(2))) ** 2) * (0.4 + 0.6 * dn.rand(3))
    dross = t.speckle(density=2600, size=0.0022, seed=5, soft=0.8)     # dross pimples in the coat
    tears = smoothstep(0.62, 0.9, t.fbm(0.012, 3, 0.5, seed=6, stretch=(1, 0.3))) * \
        smoothstep(0.5, 0.8, t.fbm(0.09, 2, 0.5, seed=7))             # thick drained zinc ("tears")
    pyr = (g.rand(7) - 0.5) * 2 * 0.0075 * np.minimum(g.dist, 0.012)       # each crystal a very flat pyramid
    relief = pyr + facet * 0.000020 + (streak - 0.5) * 0.000028 + stem * 0.000015 - gb * 0.000030 - sb * 0.000012
    scr_l = t.scratches(density=380, length=0.07, width=0.0005, seed=8)
    scr_s = t.scratches(density=1400, length=0.018, width=0.0004, seed=9)
    scr = sat(scr_l + scr_s * 0.7)

    # ---------------------------------------------------------------- weathering: patina haze + white rust
    haze = smoothstep(0.25, 0.80, t.fbm(0.18, 3, 0.5, seed=10))       # dull carbonate film
    wet = t.fbm(0.05, 3, 0.5, seed=11)                                 # where water has sat
    powd = t.fbm(0.0022, 3, 0.65, seed=15)
    cloud = smoothstep(0.56, 0.96, wet + (t.fbm(0.015, 2, 0.5, seed=12) - 0.5) * 0.12)   # soft chalky veil
    spot_b = t.speckle(density=9000, size=0.0016, seed=14, soft=0.9)   # first pin-points of white rust
    near = smoothstep(0.40, 0.70, wet)
    bloom = sat(cloud * (0.22 + 0.50 * powd) + spot_b * 0.5 * (0.1 + 0.9 * near) + gb * 0.25 * cloud)
    bloom = bloom * (1 - scr * 0.6) * BLOOM

    height = buckle - dent * 0.00016 + relief * (1 - 0.6 * bloom) + dross * 0.00010 + tears * 0.00008
    height = height + bloom * (0.00004 + 0.00005 * powd) - scr * 0.000025

    # ---------------------------------------------------------------- colour (linear)
    albedo = ramp(tone, [(0.0, "#868b90"), (0.35, "#a5aaad"), (0.7, "#bfc3c5"), (1.0, "#d8dbdc")])
    albedo = mix(albedo, hexlin("#a0a5a8") * (0.90 + 0.20 * tone)[..., None], haze * 0.45)      # blue-grey patina
    stain = smoothstep(0.5, 0.85, t.fbm(0.07, 4, 0.55, seed=16))
    albedo = overlay_tint(albedo, "#cfc9bd", stain * 0.35)                                       # faint warm water stain
    albedo = mix(albedo, hexlin("#5f6366"), gb * 0.18 * (1 - bloom))
    white = ramp(powd, [(0.0, "#b4b8b8"), (0.5, "#cdd0cf"), (1.0, "#e2e4e2")])
    albedo = mix(albedo, white, bloom * 0.9)
    albedo = mix(albedo, hexlin("#d5d7d8"), scr * 0.55)

    rough = 0.50 + 0.18 * t.blur(s_rand2 - 0.5, 0.0005) + 0.04 * facet + 0.06 * (streak - 0.5)
    rough = mix(rough, 0.60 + 0.14 * (s_rand2 - 0.5), haze * 0.5)
    rough = mix(rough, 0.92, bloom)
    rough = mix(rough, 0.30, scr * 0.7)
    rough = rough + 0.10 * gb * (1 - bloom)

    metal = 1.0 - 0.22 * haze
    metal = mix(metal, 0.0, sat(bloom * 1.1))
    metal = mix(metal, 1.0, scr * 0.7)
    return dict(albedo=albedo, height=height, rough=rough, metal=metal)
