"""Woven willow basketry (randing): 5-6 mm buff willow weavers passing alternately in front of
and behind 7 mm vertical stakes set 30 mm apart. U is horizontal (weavers run along U).

Construction: every rod is a real cylinder with a depth-varying centreline; the surface is the
z-buffer (max) of all rods, so the over/under occlusion, the dark gaps beside each stake and the
brick-like pattern of bulges come out of the geometry rather than being painted on."""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.006, "ao_strength": 0.9}

NS = 8        # stakes per tile  -> 30 mm spacing (even: over/under alternates seamlessly)
NW = 44       # weaver rows      -> 5.45 mm pitch (even)


def _rnd(n, k):
    return np.random.default_rng(90210 + 7919 * k).random(n).astype(F)


def build(t):
    ps, pw = t.W / NS, t.H / NW
    pi = math.pi

    # --- hand-made irregularity: stakes lean, rows undulate and are rapped down unevenly
    wx = (t.noise(0.09, seed=1, stretch=(1, 3)) - 0.5) * 2 * 0.0030
    wy = (t.noise(0.07, seed=2, stretch=(3, 1)) - 0.5) * 2 * 0.0016
    u = (t.X + wx) / ps
    v = (t.Y + wy) / pw
    iu = np.floor(u)
    iv = np.floor(v)

    # slow thickness change along each rod (willow tapers butt -> tip), independent per row-ish
    taper = t.noise(0.16, seed=3, stretch=(1, 0.035))            # long in X, changes row to row
    knob = t.noise(0.02, seed=4, stretch=(1.6, 0.25))

    r_row = _rnd(NW, 1)            # rod thickness
    c_row = _rnd(NW, 2)            # rod colour
    o_row = _rnd(NW, 3)            # sits proud / pushed back
    y_row = _rnd(NW, 4)            # row position error
    g_row = _rnd(NW, 5)            # hue (greener / redder)
    A_row = _rnd(NW, 6)

    floor_z = -0.0060
    z = t.zeros(floor_z)
    kind = t.zeros(0.0)            # 0 gap, 1 weaver, 2 stake
    acr = t.zeros(0.0)             # -1..1 across the winning rod
    tone = t.zeros(0.5)
    hue = t.zeros(0.5)
    rowid = t.zeros(0.0)
    over = t.zeros(0.0)            # +1 where the weaver is in front of a stake

    # --- stakes: vertical rods on the mid-plane
    r_s = 0.0032 * (0.9 + 0.2 * _rnd(NS, 7))[(iu.astype(np.int64)) % NS]
    du = (u - iu - 0.5) * ps
    ins = np.abs(du) < r_s
    zs = np.where(ins, np.sqrt(np.maximum(r_s * r_s - du * du, 0)) - 0.0008, floor_z)
    z = np.maximum(z, zs)
    kind = np.where(ins, 2.0, kind)
    acr = np.where(ins, du / r_s, acr)
    tone = np.where(ins, 0.25 + 0.3 * _rnd(NS, 8)[(iu.astype(np.int64)) % NS], tone)
    hue = np.where(ins, 0.5, hue)

    # --- weavers: each row weaves in front of / behind alternate stakes
    for k in (-1, 0, 1):
        i = (iv + k)
        ii = i.astype(np.int64) % NW
        par = np.where((i % 2) == 0, 1.0, -1.0)
        r = 0.00305 * (0.82 + 0.32 * r_row[ii]) * (0.90 + 0.22 * taper) * (0.97 + 0.06 * knob)
        cy = (i + 0.5 + (y_row[ii] - 0.5) * 0.22) * pw
        dv = v * pw - cy
        ph = np.cos(pi * (u - 0.5)) * par                         # +1 over a stake, -1 behind
        # bends round the stake, runs straighter between: slightly squared sinusoid
        phs = np.sign(ph) * np.abs(ph) ** 0.8
        amp = 0.0042 * (0.92 + 0.16 * A_row[ii])
        zc = amp * phs + (o_row[ii] - 0.5) * 0.0012
        inside = np.abs(dv) < r
        zz = np.where(inside, zc + np.sqrt(np.maximum(r * r - dv * dv, 0)), -1.0)
        win = zz > z
        z = np.where(win, zz, z)
        kind = np.where(win, 1.0, kind)
        acr = np.where(win, dv / r, acr)
        tone = np.where(win, c_row[ii], tone)
        hue = np.where(win, g_row[ii], hue)
        rowid = np.where(win, ii.astype(F), rowid)
        over = np.where(win, ph, over)

    weav = (kind == 1.0).astype(F)
    stake = (kind == 2.0).astype(F)
    rod = sat(weav + stake)

    # --- surface detail of the rods (peeled willow: fine longitudinal ribs, bud scars, dents)
    row_shift = rowid * 0.0137                                    # decorrelate neighbouring rods
    grain_h = t.sample(t.fbm(0.0016, 3, 0.6, seed=5, stretch=(14, 1)), t.X + row_shift, t.Y)
    grain_v = t.fbm(0.0016, 3, 0.6, seed=6, stretch=(1, 14))
    grain = np.where(stake > 0.5, grain_v, grain_h)
    streak_h = t.sample(t.fbm(0.004, 3, 0.55, seed=7, stretch=(22, 1)), t.X + row_shift, t.Y)
    streak_v = t.fbm(0.004, 3, 0.55, seed=8, stretch=(1, 22))
    streak = np.where(stake > 0.5, streak_v, streak_h)
    buds = t.speckle(density=5200, size=0.0030, seed=9, soft=0.7) * weav * sat(1 - np.abs(acr) * 1.2)
    dents = t.fbm(0.012, 3, 0.5, seed=10)

    height = z + rod * ((grain - 0.5) * 0.00012 + (streak - 0.5) * 0.00018
                        + buds * 0.00035 + (dents - 0.5) * 0.00025)
    gapn = t.fbm(0.004, 3, 0.6, seed=11)
    height = np.where(rod > 0.5, height, floor_z + (gapn - 0.5) * 0.0012)
    # the real surface has no vertical cliffs at 1 px: ease the silhouettes a touch
    height = mix(height, t.blur(height, t.px * 0.8), 0.6)

    # --- masks from the geometry
    crown = sat(1 - acr * acr) ** 0.5                              # lit top of each rod
    depth = sat((height - floor_z) / 0.0125)                       # 0 deep .. 1 proud
    cav = t.cavity(height, 0.0022, 0.0011)
    proud = smoothstep(0.55, 1.0, over) * weav * crown             # bulge over a stake: handled, rubbed

    # --- colour: buff (peeled, boiled) willow, each rod its own tone
    along = t.sample(t.noise(0.10, seed=12, stretch=(1, 0.05)), t.X + row_shift * 3, t.Y)
    tn = sat(0.5 + (tone - 0.5) * 0.50 + (streak - 0.5) * 0.50 + (along - 0.5) * 0.9)
    base = ramp(tn, [(0.0, "#7a5e3a"), (0.25, "#8f7146"), (0.5, "#a28353"),
                     (0.75, "#b19362"), (1.0, "#bfa474")])
    # hue drift between rods: some redder (steamed / bark-stained), some greyer-green (weathered)
    base = mix(base, base * hexlin("#e8bc98") * 1.25, sat((hue - 0.70) * 2.2)[..., None])
    base = mix(base, desaturate(base, 0.30) * 0.97, sat((0.25 - hue) * 3.0)[..., None])
    base = base * (0.90 + 0.20 * grain)[..., None]
    base = mix(base, hexlin("#4a3520"), buds * 0.75)
    # pith-coloured rubbing on the proud bulges, tan lines in the grain
    base = mix(base, base * 1.16 + 0.012, proud * 0.55)
    # stakes are older / dirtier and mostly in shade
    base = mix(base, base * hexlin("#b89a78") * 1.05, stake * 0.6)
    # dirt and dust packed where rods meet, and low in the weave
    grime = sat(cav * 0.9 + (1 - depth) * 0.55) * (0.65 + 0.5 * t.fbm(0.03, 3, seed=13))
    base = mix(base, base * hexlin("#6e5a44") * 1.15, sat(grime) * 0.85)
    # soft patina patches over the panel (kept low: the engine adds its own macro variation)
    macro = t.fbm(0.12, 3, 0.5, seed=14)
    base = base * (0.93 + 0.14 * macro)[..., None]
    gap_col = ramp(gapn, [(0, "#2a2119"), (1, "#3d3226")])
    albedo = mix(gap_col, base, rod)

    # --- roughness: peeled willow has a soft sheen, polished on the bulges, dusty in the joints
    rough = 0.60 - 0.16 * proud - 0.05 * crown + 0.10 * (grain - 0.5) + 0.06 * (macro - 0.5)
    rough = rough + 0.25 * sat(grime) + 0.10 * buds + 0.06 * stake
    rough = mix(0.95, rough, rod)

    ao = sat(1 - (1 - t.ao(height, 0.0035, 1.0)) * 0.75)
    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0, ao=ao)
