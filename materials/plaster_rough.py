"""Roughcast (wet-dash harling) under old limewash.  Built as it is made: a scratched base coat on
an uneven wall; a gravel-and-lime slurry cast on by the trowel-full so pebbles lie in overlapping
clots, every stone skinned in slurry; limewash over the lot.  Aged moderately: the wash has rubbed
off the pebble tips, peeled in thin scabs, and here and there a small clot of dash has let go,
leaving a crumbled-edged hollow down to the base coat with a few stones still stuck in it."""
from tk.tex import *
import math


def pebbles(t, size, rmin, rmax, keep, seed):
    """Close-packed rounded stones from a Voronoi layout: (dome 0..1, layout)."""
    lay = t.cells(size, jitter=0.95, seed=seed)
    r = rmin + (rmax - rmin) * lay.rand(1)
    q = sat(1 - (lay.f1 / r) ** 2)
    dome = np.sqrt(q) * (0.45 + 0.55 * lay.rand(2)) * (lay.rand(3) < keep)
    return dome.astype(F), lay


def build(t):
    # ------------------------------------------------------------------ height (metres)
    wall = (t.fbm(0.6, 3, 0.5, seed=1) - 0.5) * 0.007
    # the cast: trowel-fulls land as overlapping clots, thick here, thin there
    clot = t.fbm(0.11, 3, 0.5, seed=2)
    p_s, lay_s = pebbles(t, 0.0105, 0.0030, 0.0056, 0.78, 3)
    p_b, lay_b = pebbles(t, 0.0230, 0.0055, 0.0090, 0.50, 4)
    # every stone is skinned in slurry: the pea grit is buried deeper (softer), the big stones stay crisp
    a_s = t.blur(p_s * 0.0030, 0.0014)
    a_b = t.blur(p_b * 0.0048, 0.0009)
    big_on = a_b > a_s
    agg = np.maximum(a_s, a_b) * (0.55 + 0.75 * clot)
    slurry = t.fbm(0.017, 4, 0.6, seed=5)
    grit = t.fbm(0.0045, 3, 0.65, seed=6)
    coat = (clot - 0.5) * 0.0085 + agg + (slurry - 0.5) * 0.0018 + (grit - 0.5) * 0.00026

    # A few small scabs (3-7 cm) where a clot of dash has let go, scattered evenly: the big losses
    # are the engine's wear layer's job.  Each has its own depth, and a crumbled margin where the
    # coat thins out over 5-12 mm instead of a cut step; some stones stay stuck to the floor.
    sc = t.cells(0.30, jitter=0.75, seed=7)
    on = sc.rand(1) < 0.48
    rad = 0.019 + 0.019 * sc.rand(2)
    f1 = t.warp(sc.f1, 0.007, 0.05, seed=16)
    sd = f1 - rad + (t.fbm(0.030, 2, 0.5, seed=17) - 0.5) * 0.020           # ragged outline (m, <0 inside)
    gx, gy = t.slope(sd)
    sd = sd / np.maximum(np.sqrt(gx * gx + gy * gy), 0.6)                    # back to a true distance, so the
    sd = np.where(on, sd, 0.05).astype(F)                                    # margin has an honest width
    rimw = 0.007 + 0.007 * sc.rand(3)
    # the margin crumbles stone by stone: each pebble lets go at its own distance from the edge,
    # so the coat thins in ragged pebble-sized bites instead of along a cut line
    q = sat(-sd / rimw)                                                      # 0 at the edge -> 1 a rim-width in
    rp = np.where(big_on, lay_b.rand(8), lay_s.rand(8))
    crumb = (t.fbm(0.008, 2, 0.6, seed=18) - 0.5) * 0.5
    lost = smoothstep(0.0, 1.0, sat(q * 1.75 - rp * 0.75 + crumb * sat(q * 4)))
    lost = t.blur(lost.astype(F), 0.0013)
    lost = lost * on
    depth = 0.0040 + 0.0050 * sc.rand(4)
    # the base coat underneath: floated, sandy, faintly combed for key
    comb_y = t.Y + (t.noise(0.25, seed=9) - 0.5) * 0.05
    comb = 0.5 + 0.5 * np.sin(2 * math.pi * comb_y / (t.H / 80.0))          # 25 mm comb pitch, tiles
    comb = smoothstep(0.55, 0.95, comb) * (0.4 + 0.6 * t.noise(0.08, seed=10))
    sand = t.fbm(0.006, 3, 0.65, seed=11)
    # stones and clots of slurry that stayed stuck to the base coat
    stay_s = (lay_s.rand(7) < 0.24)
    stay_b = (lay_b.rand(7) < 0.15)
    left = np.maximum(a_s * stay_s, a_b * stay_b) * 0.75 + smoothstep(0.62, 0.80, slurry) * 0.0012
    floor_var = (t.fbm(0.03, 3, 0.55, seed=12) - 0.5) * 0.0030                # the break is never a flat plane
    under = -depth - comb * 0.0005 + (sand - 0.5) * 0.0007 + floor_var + left
    under = np.minimum(under, coat)

    # shrinkage cracks: short, hair-fine, opening and closing along their length
    ckn = t.cracks(size=0.42, width=0.0016, seed=13, wander=0.016, density=0.20)
    ckn = t.warp(ckn, 0.0035, 0.011, seed=28)                  # a crack picks its way between the stones
    ckg = t.noise(0.10, seed=19)
    ck = ckn * smoothstep(0.42, 0.62, ckg) * (0.45 + 0.55 * sat(t.noise(0.03, seed=27) * 1.6 - 0.3)) * (1 - lost)

    # thin scabs where only the limewash skin has peeled
    pn = t.fbm(0.055, 4, 0.55, seed=14)
    peel = smoothstep(0.70, 0.73, pn) * smoothstep(0.45, 0.7, t.fbm(0.3, 3, 0.5, seed=15)) * (1 - lost)

    height = wall + mix(coat - peel * 0.0004 - ck * 0.0012, under, lost)

    # ------------------------------------------------------------------ masks
    cav = t.cavity(height, 0.004, 0.0011)                      # between the stones
    tip = t.edges(height, 0.005, 0.0013)                       # pebble tips
    shade = t.cavity(height, 0.012, 0.003)                     # foot of the broken coat edge
    floor = smoothstep(0.35, 0.95, lost)                       # where the base coat really shows

    # ------------------------------------------------------------------ colour
    cloud = t.fbm(0.45, 4, 0.5, seed=20)
    mid = t.fbm(0.10, 4, 0.55, seed=21)
    tone = sat(0.5 + (cloud - 0.5) * 0.45 + (mid - 0.5) * 0.35 + (clot - 0.5) * 0.25 + (slurry - 0.5) * 0.25)
    wash = ramp(tone, [(0.0, "#bfb7a6"), (0.3, "#cbc4b5"), (0.6, "#d4cec1"), (1.0, "#dcd8cd")])
    wash = overlay_tint(wash, "#e6d8bb", smoothstep(0.5, 0.85, t.fbm(0.3, 4, 0.5, seed=22)) * 0.22)
    wash = overlay_tint(wash, "#dde2e3", smoothstep(0.55, 0.85, t.fbm(0.36, 3, 0.5, seed=23)) * 0.18)
    # older, greyer coat where the top skin has peeled
    wash = mix(wash, wash * hexlin("#d9d4c9") * 1.02, peel * 0.9)
    # wash rubbed and rained off the tips of some stones: the gravel itself shows
    stone_c = mix(ramp(lay_s.rand(5), [(0, "#7d776d"), (0.4, "#9a917f"), (0.7, "#8a8378"), (1, "#a79b86")]),
                  ramp(lay_b.rand(5), [(0, "#77726a"), (0.4, "#958b79"), (0.7, "#a39680"), (1, "#868178")]), big_on)
    rubbed = smoothstep(0.30, 0.70, t.fbm(0.2, 3, 0.5, seed=24))
    bare = tip * smoothstep(0.30, 0.75, mix(lay_s.rand(6), lay_b.rand(6), big_on)) * rubbed
    bare = np.maximum(bare, sat(left / 0.002) * floor * 0.7)   # the stones left in the scabs are bare too
    wash = mix(wash, stone_c, bare * 0.55)
    wash = mix(wash, wash * 1.05, tip * 0.5)
    # dust and soot lodge between the stones
    wash = mix(wash, wash * hexlin("#c2b8a6") * 1.0, sat(cav * 0.85))
    wash = mix(wash, hexlin("#857c6d"), ck * 0.25)
    # the base coat: sandy lime render, a little buffer than the wash, with wash ghosts
    base_c = ramp(sat(0.5 + (sand - 0.5) * 1.5 + (t.fbm(0.07, 3, 0.5, seed=25) - 0.5) * 0.6),
                  [(0.0, "#a99f8b"), (0.5, "#bfb5a2"), (1.0, "#d0c8b8")])
    base_c = mix(base_c, base_c * 0.95, comb * 0.5)
    base_c = mix(base_c, hexlin("#d0c9bb"), smoothstep(0.55, 0.8, t.fbm(0.03, 3, 0.55, seed=26)) * 0.4)   # wash ghosts
    base_c = mix(base_c, stone_c, sat(left / 0.002) * 0.45)
    base_c = mix(base_c, wash, 0.22)
    alb = mix(wash, base_c, floor)
    alb = mix(alb, alb * hexlin("#ddd5c6"), sat(shade * 0.5) * (0.3 + 0.4 * lost))

    # ------------------------------------------------------------------ roughness
    # limewash skin chalk-matt 0.85-0.9; dusty cavities 0.97; rubbed bare pebble tips 0.65-0.72
    rvar = t.fbm(0.14, 4, 0.55, seed=30)
    rough = 0.885 + (rvar - 0.5) * 0.07 + (grit - 0.5) * 0.04 + sat(cav * 1.2) * 0.09
    rough = rough - tip * (0.05 + 0.07 * rubbed) - bare * 0.25 - peel * 0.02
    rough = mix(rough, 0.90 + (sand - 0.5) * 0.10 - comb * 0.03 - sat(left / 0.002) * 0.15, floor)
    return dict(albedo=alb, height=height, rough=np.clip(rough, 0.6, 0.99), metal=0.0)
