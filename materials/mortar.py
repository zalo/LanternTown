"""Plain sandy lime mortar / render fill (tintable: authored as a mid-light neutral sand-lime).

What it is: coarse sharp sand and a little grit bound in lime, thrown on and rubbed up with a
wooden float. The float leaves a closed, slightly burnished skin with faint sweeping drag marks;
weather has since eaten that skin away in patches so the sand stands out, a few bits of grit
have dropped out leaving sockets, there are unslaked lime lumps, pinholes and the odd pressed-in
trowel dab. Colour variation is kept relative (lighter / darker, slightly warmer) so the
surface still makes sense when the engine tints it.
"""
from tk.tex import *

META = {"normal_strength": 1.0}


def build(t):
    def moved(a, fx, fy):
        return np.roll(a, (int(t.h * fy), int(t.w * fx)), axis=(0, 1))

    # ------------------------------------------------------------------ how the surface was worked
    lie = t.fbm(0.30, 3, 0.5, seed=1)                        # never flat: hand-floated over rough backing
    lie2 = t.fbm(0.09, 3, 0.5, seed=2)
    # float drags: long shallow strokes in two sweeping directions
    da = t.fbm(0.05, 3, 0.5, seed=3, stretch=(7.0, 1.0))
    db = t.fbm(0.05, 3, 0.5, seed=4, stretch=(1.0, 6.0))
    sweep = smoothstep(0.35, 0.65, t.fbm(0.35, 2, 0.5, seed=5))
    wx = (t.fbm(0.25, 2, 0.5, seed=6) - 0.5) * 0.10
    wy = (t.fbm(0.25, 2, 0.5, seed=7) - 0.5) * 0.10
    drag = t.sample(mix(da, db, sweep), t.X + wx, t.Y + wy)  # bent into arcs
    # where the skin has weathered off (patches of a hand or two across, ragged edges)
    wn = t.fbm(0.16, 4, 0.55, seed=8) + (t.fbm(0.02, 3, 0.6, seed=9) - 0.5) * 0.22 + (lie2 - 0.5) * 0.3
    worn = smoothstep(0.33, 0.50, wn)                        # most of it: only islands of skin are left
    worn_soft = smoothstep(0.25, 0.60, wn)

    # ------------------------------------------------------------------ aggregate
    sand = t.fbm(0.0045, 4, 0.7, seed=10)                    # packed sharp sand
    sand2 = t.fbm(0.012, 3, 0.6, seed=11)
    grains = t.speckle(density=42000, size=0.0030, seed=12)  # coarse grains 2-3 mm
    grit = t.speckle(density=3400, size=0.0080, seed=13, soft=0.5)   # small stones 4-8 mm
    gsel = t.noise(0.02, seed=14)
    grit_in = grit * smoothstep(0.38, 0.5, gsel)             # still in place
    grit_out = grit * (1 - smoothstep(0.30, 0.38, gsel))     # dropped out: a socket
    pin = t.speckle(density=9000, size=0.0028, seed=15)      # pinholes / air voids
    pin = pin * smoothstep(0.35, 0.6, t.noise(0.07, seed=16))
    # trowel-pressed dabs: soft-edged lumps a few cm across where it was pushed in and not rubbed flat
    dab = t.fbm(0.035, 3, 0.5, seed=17)
    dab = smoothstep(0.35, 0.75, dab) * smoothstep(0.3, 0.7, t.fbm(0.2, 2, 0.5, seed=18))

    height = ((lie - 0.5) * 0.0040 + (lie2 - 0.5) * 0.0030
              + (drag - 0.5) * 0.0018 * (1 - 0.5 * worn) + dab * 0.0012
              - worn * 0.0005 - worn_soft * 0.0004
              + (sand - 0.5) * mix(0.00035, 0.0011, worn)
              + (sand2 - 0.5) * mix(0.0003, 0.0008, worn)
              + grains * mix(0.00012, 0.00055, worn)
              + grit_in * mix(0.0005, 0.0014, worn)
              - grit_out * 0.0012
              - pin * 0.0009)

    cav = t.cavity(height, 0.0035, 0.0005)
    edg = t.edges(height, 0.004, 0.0006)

    # ------------------------------------------------------------------ colour
    tone = sat(0.5 + (t.fbm(0.22, 3, 0.5, seed=20) - 0.5) * 0.9 + (lie2 - 0.5) * 0.4 + (sand2 - 0.5) * 0.35)
    col = ramp(tone, [(0.0, "#a99d88"), (0.5, "#b9ad98"), (1.0, "#c6bba6")])
    # the worn patches show the sand: a little darker and warmer, speckled
    sandcol = ramp(sand, [(0.0, "#96896f"), (0.5, "#afa189"), (1.0, "#c4b79f")])
    col = mix(col, sandcol * (0.93 + 0.14 * tone)[..., None], worn * 0.8 + 0.2)
    # float-burnished high spots are a shade lighter (lime drawn to the surface)
    col = mix(col, col * 1.07, smoothstep(0.5, 0.75, drag) * (1 - worn) * 0.8)
    # individual grains: dark flint / chert, pale quartz, ochre
    gvis = 0.35 + 0.65 * worn
    col = mix(col, hexlin("#6d675d"), moved(grains, 0.31, 0.57) * 0.55 * gvis)
    col = mix(col, hexlin("#d3cdbf"), grains * 0.40 * gvis)
    col = mix(col, hexlin("#a58f6c"), moved(grains, 0.67, 0.19) * 0.40 * gvis)
    gtone = t.noise(0.011, seed=21)
    gcol = ramp(gtone, [(0.0, "#6b665e"), (0.4, "#8c8474"), (0.7, "#a3937a"), (1.0, "#c0b9aa")])
    col = mix(col, gcol, sat(grit_in * 1.4) * (0.45 + 0.45 * worn))
    # unslaked lime lumps
    lime = moved(grit, 0.43, 0.71) * smoothstep(0.55, 0.7, t.noise(0.03, seed=22))
    col = mix(col, hexlin("#ddd8cb"), sat(lime * 1.3) * 0.7)
    # faint drying / damp tide marks, then dirt in sockets and pinholes
    damp = smoothstep(0.52, 0.8, t.fbm(0.28, 3, 0.55, seed=23))
    col = col * (1 - 0.07 * damp)[..., None]
    dirt = sat(cav * 0.9 + grit_out * 0.7 + pin * 0.6)
    col = mix(col, col * hexlin("#8c8478") * 1.9, dirt * 0.55)
    col = col * (1 + 0.05 * edg)[..., None]
    albedo = np.clip(col, 0.03, 0.85)

    # ------------------------------------------------------------------ roughness
    rough = (0.86 + 0.09 * worn - 0.10 * smoothstep(0.5, 0.8, drag) * (1 - worn)
             + (sand - 0.5) * 0.10 - 0.10 * grains * gvis - 0.14 * sat(grit_in * 1.4)
             + 0.06 * dirt - 0.04 * damp)
    return dict(albedo=albedo, height=height, rough=np.clip(rough, 0.4, 1.0), metal=0.0)
