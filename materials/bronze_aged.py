"""Dark statuary bronze.

A lost-wax casting reproduces the sculptor's surface exactly, so the bronze still shows how the
wax / clay was worked: the whole skin is a patchwork of overlapping modelling-tool facets and
thumb dabs (15-30 mm), each a slightly dished, slightly tilted plane meeting its neighbours in
soft ridges and creases. Backgrounds were matted afterwards with a punch (dense 1 mm dimples) and
the foundry dressed a few places with a rasp. Over that sits the fine grain of the investment and
a scatter of casting pits.

The foundry then stippled on a hot chocolate-brown patina and waxed it. After a century outdoors:
* the wax survives as a satin sheen on the open planes and has gone dry and matt in the hollows;
* thin grey-green verdigris has crept along the creases between facets, into the matting and the
  pits - connected films in the recesses, never islands on the open planes;
* hands have rubbed the patina off the proud ridges, where dark golden metal glints through.
"""
from tk.tex import *

META = {"normal_strength": 1.5}


def _facets(t, size, aspect, seed, tilt, step, dish):
    lay = t.cells(size, 0.95, seed=seed, aspect=aspect)
    r = lay.f1 / (size * 0.6)
    h = lay.tilt(tilt, 1) + (lay.rand(2) - 0.5) * step + dish * r * r       # tilted, stepped, dished dab
    return h.astype(F), lay


def build(t):
    # ---------------------------------------------------------------- modelled form
    und = (t.fbm(0.30, 2, 0.5, seed=1) - 0.5) * 0.0007
    lump = (t.fbm(0.12, 3, 0.40, seed=2) - 0.5) * 0.0010                  # the sculptor's broad modelling

    # two generations of tool strokes; the later one cuts into the earlier (soft minimum of planes)
    ha, la = _facets(t, 0.028, 1.35, 3, 0.00060, 0.00024, 0.00005)
    hb, lb = _facets(t, 0.022, 0.80, 4, 0.00050, 0.00028, 0.00008)
    hb = hb + 0.00008
    k = 0.00006
    fac = -k * np.log(np.exp(-ha / k) + np.exp(-hb / k))
    fac = t.warp(fac.astype(F), 0.0022, 0.025, seed=31)                    # no tool leaves a straight edge
    fac = t.blur(fac, 0.0011)                                             # wax edges are soft, then cast
    fac = fac - float(fac.mean())

    # matting punched into some zones (backgrounds, drapery): dense 1 mm dimples
    mz = smoothstep(0.56, 0.68, t.fbm(0.15, 3, 0.5, seed=5))
    punch = t.speckle(density=330000, size=0.0015, seed=6, soft=0.7)
    punch = sat(punch + t.speckle(density=200000, size=0.0013, seed=26, soft=0.7) * 0.8)
    matt = punch * mz

    # fettling: bundles of rasp / riffler strokes where the foundry dressed the casting
    rz1 = smoothstep(0.60, 0.74, t.fbm(0.10, 2, 0.5, seed=7)) * (1 - mz)
    rz2 = smoothstep(0.62, 0.76, t.fbm(0.10, 2, 0.5, seed=8)) * (1 - mz)
    rasp = t.scratches(density=5200, length=0.040, width=0.0010, seed=21, angle=28, angle_spread=12) * rz1
    rasp = sat(rasp + t.scratches(density=5200, length=0.036, width=0.0010, seed=22, angle=-52, angle_spread=12) * rz2)

    # casting skin and pits
    grain = t.fbm(0.0042, 3, 0.6, seed=9)
    pclu = smoothstep(0.50, 0.76, t.fbm(0.07, 3, 0.5, seed=10))
    pit_a = t.speckle(density=420, size=0.0036, seed=11, soft=0.7) * (0.15 + 0.85 * pclu)
    pit_b = t.speckle(density=5000, size=0.0018, seed=12, soft=0.6) * (0.05 + 0.95 * pclu)
    pits = sat(pit_a + pit_b * 0.7)
    scr = t.scratches(density=120, length=0.05, width=0.0007, seed=13)

    form = und + lump + fac * (1 - 0.45 * mz)
    height = form + (grain - 0.5) * 0.00010 - matt * 0.00007 - rasp * 0.00015 - pits * 0.00038 - scr * 0.00004

    # ---------------------------------------------------------------- masks from the relief
    hs = t.blur(height, 0.0007)
    crease = t.cavity(hs, 0.0050, 0.00016)                                # foot of each tool step, under ledges
    seam = t.cavity(hs, 0.0026, 0.00007)                                  # the thin line right in the crease
    ridge = t.edges(hs, 0.0030, 0.00015)                                  # proud facet edges a hand touches
    hollow = t.cavity(form, 0.030, 0.0006)                                # broad hollows of the modelling
    proud = t.edges(form, 0.030, 0.0006)
    fine_cav = t.cavity(height, 0.0020, 0.00012)                          # pits, punch marks, rasp grooves

    powd = t.fbm(0.0030, 3, 0.65, seed=14)
    crusty = t.fbm(0.020, 3, 0.5, seed=15)
    gz = smoothstep(0.30, 0.70, t.fbm(0.085, 3, 0.5, seed=16))            # where water lingers
    # verdigris: a thin film that follows the crease lines, thickening where water lingers
    wet = sat(seam * 0.75 + crease * 0.45 + pit_a * 0.8) * (0.35 + 0.65 * gz) * (0.25 + 0.75 * smoothstep(0.30, 0.62, crusty))
    green = smoothstep(0.42, 0.86, wet + (powd - 0.5) * 0.30)
    green = t.blur(green, 0.0005) * 0.6 + green * 0.4                     # the film feathers out, no chip edges
    green = sat(green + matt * mz * 0.20 * smoothstep(0.45, 0.75, crusty) * gz + rasp * 0.08 * gz)
    halo = sat(t.blur(green, 0.0020) * 1.4 - green * 0.5) * (0.4 + 0.6 * powd)   # thin wash bleeding out of the film

    hand = smoothstep(0.40, 0.72, t.fbm(0.10, 3, 0.5, seed=17))
    gold = smoothstep(0.50, 0.80, ridge * (0.30 + 0.70 * hand) + (powd - 0.5) * 0.55 + (grain - 0.5) * 0.25 + proud * 0.08)
    gold = gold * (1 - green) * (1 - pits) * (1 - 0.7 * rasp) * (1 - sat(crease * 2))
    gold = sat(gold + scr * 0.35)

    # ---------------------------------------------------------------- colour (linear)
    stip = t.fbm(0.009, 3, 0.55, seed=18)                                 # stippled-on patina
    dabv = (la.rand(5) - 0.5) * 0.10                                      # each plane took the torch a little differently
    dabv = t.blur(t.warp(dabv.astype(F), 0.0022, 0.025, seed=31), 0.002)
    tone = sat(0.45 + (stip - 0.5) * 0.38 + (t.fbm(0.13, 3, 0.5, seed=19) - 0.5) * 0.18 + (grain - 0.5) * 0.14
               + dabv + ridge * 0.22 - crease * 0.20)                     # patina thinner on the ridges
    albedo = ramp(tone, [(0.0, "#2a1e17"), (0.35, "#3a291d"), (0.65, "#4c3624"), (1.0, "#5f452c")])
    redz = smoothstep(0.5, 0.85, t.fbm(0.05, 3, 0.5, seed=20))
    albedo = overlay_tint(albedo, "#c88f7a", redz * 0.30)                 # cuprite red-brown undertone
    dirt = sat(crease * 0.55 + fine_cav * 0.45 + hollow * 0.15) * (1 - green)
    albedo = mix(albedo, hexlin("#191411"), dirt * 0.75)                  # old wax + soot in the recesses
    albedo = mix(albedo, hexlin("#1d1713"), rasp * 0.35)
    verd = ramp(powd, [(0.0, "#31443a"), (0.55, "#445d50"), (1.0, "#5c7a68")])
    albedo = mix(albedo, hexlin("#35463d"), halo * 0.40)
    albedo = mix(albedo, verd, green * (0.45 + 0.40 * powd))
    bronze = ramp(grain, [(0.0, "#8a6a3a"), (0.6, "#9a7841"), (1.0, "#a88448")])
    albedo = mix(albedo, bronze, gold * 0.6)

    # ---------------------------------------------------------------- PBR
    dryz = smoothstep(0.45, 0.75, t.fbm(0.09, 3, 0.5, seed=23) * 0.6 + hollow * 0.5)   # wax weathered away
    rough = 0.415 + 0.16 * (stip - 0.5) + 0.08 * (grain - 0.5) - 0.05 * ridge
    rough = mix(rough, 0.66, dryz * 0.9)
    rough = mix(rough, 0.72, sat(dirt * 0.8 + rasp * 0.6 + matt * 0.35))
    rough = mix(rough, 0.80, halo * 0.6)
    rough = mix(rough, 0.90, green)
    rough = mix(rough, 0.30, gold)
    metal = 0.12 * smoothstep(0.35, 0.8, tone) * (1 - dryz)
    metal = mix(metal, 0.0, sat(green * 1.2 + halo * 0.5 + dirt))
    metal = mix(metal, 1.0, gold)
    return dict(albedo=albedo, height=height, rough=rough, metal=metal)
