"""Thickly painted cast iron (tintable, authored as an off-white gloss).

Built in the order the real thing is: a sand-cast skin (granular, with the odd sand pit and
blister), buried under several brushed coats of gloss paint that soften it to an orange peel,
leave vertical brush drag and a few sags / runs, then age: grime settles in the peel, the gloss
dulls in patches, hairline cracks open and brittle flakes chip off through the coats down to the
rust-browned iron with a faint rust weep below.
Down = V 0 (runs and weeps fall towards the bottom of the texture).
"""
from tk.tex import *

META = {"normal_strength": 2.0}


def runs(t, n, seed, width=(0.0022, 0.0045), length=(0.02, 0.085)):
    """Paint runs: long vertical beads with a fat rounded lower end. Returns 0..1 height profile."""
    rng = np.random.default_rng(seed)
    out = t.zeros()
    k = int(math.ceil(math.sqrt(n)))
    for i in range(n):
        gx, gy = i % k, i // k
        cx = (gx + rng.random()) / k * t.W
        cy = (gy + rng.random()) / k * t.H
        w = width[0] + rng.random() * (width[1] - width[0])
        L = length[0] + rng.random() ** 1.6 * (length[1] - length[0])
        amp = 0.45 + 0.55 * rng.random()
        wob = (rng.random() - 0.5) * 0.06
        dy = (t.Y - cy + t.H / 2) % t.H - t.H / 2          # >0 above the tip
        dx = (t.X - cx + t.W / 2) % t.W - t.W / 2
        up = np.maximum(dy, 0.0)
        dx = dx - wob * up
        r = w * (0.5 + 0.5 * np.exp(-up / (2.5 * w)))
        d = np.sqrt(dx * dx + np.minimum(dy, 0.0) ** 2)
        prof = np.sqrt(sat(1 - (d / r) ** 2)) * (0.55 + 0.45 * np.exp(-up / (3 * w)))
        fade = 1 - smoothstep(0.45 * L, L, up)
        out = np.maximum(out, prof * fade * amp)
    return out.astype(F)


def build(t):
    # ---------------------------------------------------------------- cast iron under the paint
    und = (t.fbm(0.12, 2, 0.5, seed=1) - 0.5) * 0.0004                   # pattern / mould waviness
    swell = (t.fbm(0.032, 3, 0.45, seed=21) - 0.5) * 0.00060             # swells / scabs of a sand mould
    sand = t.fbm(0.0048, 3, 0.6, seed=2)                                # moulding-sand grain
    sand2 = t.noise(0.0085, seed=3)
    cast = (sand - 0.5) * 0.00042 + (sand2 - 0.5) * 0.00034
    sandpit = t.speckle(density=1100, size=0.0042, seed=4, soft=0.8)    # sand inclusions (pits)
    blister = t.speckle(density=700, size=0.0050, seed=5, soft=0.9)     # cold shots / scabs (bumps)
    cast = cast - sandpit * 0.00040 + blister * 0.00032

    # ---------------------------------------------------------------- paint: several thick coats
    skin = t.blur(cast, 0.0010) * 0.95 + cast * 0.10                    # paint floods the grain -> orange peel
    brush = t.fbm(0.0030, 3, 0.5, seed=6, stretch=(1.0, 0.03))          # vertical brush drag
    brush = t.warp(brush, 0.002, 0.05, seed=22)
    brush_amt = 0.25 + 0.75 * smoothstep(0.3, 0.7, t.fbm(0.08, 3, 0.5, seed=7, stretch=(1, 0.35)))
    run = runs(t, 18, 8, width=(0.0038, 0.0080), length=(0.04, 0.17))
    run = t.blur(run, 0.0008)
    sagn = t.fbm(0.045, 3, 0.5, seed=9, stretch=(1.8, 1.0))
    sag = smoothstep(0.52, 0.80, sagn) * smoothstep(0.0, 0.5, t.fbm(0.1, 2, 0.5, seed=23))   # fat lazy curtains
    nib = t.speckle(density=4000, size=0.0017, seed=10, soft=0.8)       # dust nibs caught in the wet coat
    paint = swell + skin + (brush - 0.5) * 0.00022 * brush_amt + run * 0.00075 + sag * 0.00030 + nib * 0.00008

    # ---------------------------------------------------------------- ageing: cracks and chips
    crk = t.cracks(0.05, 0.0005, seed=11, wander=0.003, density=0.16)
    zone = smoothstep(0.58, 0.9, t.fbm(0.06, 3, 0.5, seed=12))          # where the paint is failing
    fl = t.cells(0.010, 1.0, seed=13)
    pick = (fl.rand(1) < 0.30 * zone ** 6).astype(F)                     # brittle angular flakes, in groups
    fs = t.cells(0.0042, 1.0, seed=24)
    nick = (fs.rand(1) < 0.0009).astype(F)                               # isolated small knocks
    body = t.blur(pick, 0.0012) + t.blur(nick, 0.0007) * 1.1             # neighbours merge into one chip
    body = t.warp(body, 0.0010, 0.005, seed=14)
    body = body + (t.fbm(0.003, 2, 0.5, seed=15) - 0.5) * 0.10
    e = 0.06 * t.px / 0.0006
    chip_o = smoothstep(0.30, 0.30 + e, body)                            # top coats gone -> older coat
    chip_i = smoothstep(0.66, 0.66 + e, body)                            # through to the iron
    rim = sat(smoothstep(0.12, 0.30, body) - chip_o)                     # lifted, curling edge
    iron_tex = (sand - 0.5) * 0.00030 + (t.fbm(0.0018, 2, 0.6, seed=16) - 0.5) * 0.00010

    height = und + paint * (1 - chip_i) + swell * chip_i + rim * 0.00007 - crk * 0.00010 * (1 - chip_o)
    height = height - chip_o * 0.00018 - chip_i * 0.00030 + chip_i * iron_tex

    # ---------------------------------------------------------------- masks
    peel_cav = t.cavity(skin, 0.0030, 0.00016)                           # grime sits in the peel
    peel_top = t.edges(skin, 0.0030, 0.00016)
    cav = t.cavity(height, 0.0020, 0.00018)
    dull = t.fbm(0.14, 4, 0.5, seed=17)                                  # chalked / weathered gloss
    weep = chip_i.copy()
    for k in range(1, 7):                                                # rust weeps downhill (towards V 0)
        weep = np.maximum(weep, t.shift(t.blur(chip_i, 0.0009 + 0.0003 * k), 0, -0.003 * k) * (1 - 0.15 * k))
    weep = sat(weep * 1.5) * (1 - chip_i) * (0.4 + 0.6 * t.fbm(0.004, 2, 0.5, seed=18, stretch=(1, 0.2)))

    # ---------------------------------------------------------------- colour (linear)
    base = hexlin("#cbc9c2")
    v = 1.0 + 0.035 * (t.fbm(0.18, 3, 0.5, seed=19) - 0.5) * 2          # coat thickness / fading
    v = v + 0.03 * (brush - 0.5) * brush_amt + 0.02 * (peel_top - peel_cav)
    albedo = base[None, None, :] * v[..., None]
    albedo = overlay_tint(albedo, "#e6e0d2", smoothstep(0.4, 0.9, dull) * 0.5)    # slight yellowed chalk
    albedo = overlay_tint(albedo, "#a59d8f", sat(peel_cav * 0.35 + cav * 0.3))    # grime in the lows
    albedo = overlay_tint(albedo, "#857f76", sat(crk * 0.45))
    old = base * hexlin("#c4b69a") * 0.9                                       # older, yellowed coat under the top one
    albedo = mix(albedo, old * (0.92 + 0.16 * sand)[..., None], chip_o)
    albedo = overlay_tint(albedo, "#7d7468", rim * 0.5)
    iron = ramp(sat(sand * 0.7 + t.fbm(0.02, 3, 0.5, seed=20) * 0.4),
                [(0.0, "#2a211d"), (0.4, "#3d2b22"), (0.7, "#553626"), (1.0, "#6e442b")])
    albedo = overlay_tint(albedo, "#b48a62", weep * 0.6)
    albedo = mix(albedo, iron, chip_i)

    rough = 0.24 + 0.14 * dull + 0.14 * peel_cav + 0.04 * (sand2 - 0.5) + 0.07 * (brush - 0.5) * brush_amt
    rough = rough - 0.06 * sat(run + sag * 0.6)                          # fat paint flows out glassier
    rough = mix(rough, 0.55, sat(crk + weep * 0.6))
    rough = mix(rough, 0.48, chip_o)
    rough = mix(rough, 0.80 + 0.1 * (sand - 0.5), chip_i)
    metal = chip_i * (0.25 + 0.3 * smoothstep(0.55, 0.3, sand))
    return dict(albedo=albedo, height=height, rough=rough, metal=metal)
