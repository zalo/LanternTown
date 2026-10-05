"""Aged brass (tintable; authored as yellow brass).

Rolled / spun brass sheet that was hand-polished with an abrasive cloth and then left in service:

* the sheet is not dead flat - a gentle waviness, a few shallow knocks (irregular dished dents)
  and short sharp nicks whose displaced metal stands up as a burr on one side;
* the polish survives as long fine lines mostly one way (along V, slightly arcing as an arm swings),
  crossed in patches by later passes at other angles and by random handling scratches;
* tarnish is a thin even film that has thickened where the cloth no longer reaches: a fine
  mottle, streaks of build-up lying in the polishing lines, oval finger marks, a few broken
  drying tide-lines, and dark brown oxide in the dents, nicks, pores and deeper scratches;
* where hands and cloth still rub, the film is thin and the metal is brighter and smoother.

Roughness and metalness are driven by the same tarnish thickness: bright metal is smooth and
fully metallic, the film is satin, thick brown oxide is matt and no longer a conductor.
"""
from tk.tex import *

META = {"normal_strength": 2.0}


def build(t):
    A0 = 90.0                                                             # polishing direction (deg): along V

    def st(k):                                                            # stretch along the polishing direction
        return (1.0, float(k))

    # ---------------------------------------------------------------- sheet form
    wavy = (t.fbm(0.10, 2, 0.45, seed=1) - 0.5) * 0.00030                 # ~0.15 mm waviness at 0.1 m
    mesoN = t.fbm(0.042, 3, 0.42, seed=60)                                # hand-worked sheet: never optically flat
    meso = (mesoN - 0.5) * 0.00030
    hollow = smoothstep(0.52, 0.24, mesoN)
    crest = smoothstep(0.48, 0.76, mesoN)

    # shallow knocks: a few irregular dished dents, 8-20 mm, each with its own axis and a sharper side
    dn = t.cells(0.080, 1.0, seed=2)
    sel = (dn.rand(1) < 0.32).astype(F)
    ang = dn.rand(4) * math.pi
    ca, sa_ = np.cos(ang), np.sin(ang)
    du = dn.u * ca + dn.v * sa_
    dv = -dn.u * sa_ + dn.v * ca
    ra = 0.0040 + 0.0050 * dn.rand(2)                                     # semi-axes 4..9 mm
    rb = ra * (0.55 + 0.40 * dn.rand(5))
    rr = np.sqrt((du / ra) ** 2 + (dv / rb) ** 2) * (1.0 + 0.25 * np.tanh(du / ra))   # one steeper side
    dent = sel * np.exp(-rr ** 2.4) * (0.55 + 0.45 * dn.rand(3)) * smoothstep(0.0, 0.006, dn.dist)
    dent = t.warp(dent.astype(F), 0.0016, 0.014, seed=31)                  # irregular outline
    dent = t.blur(dent, 0.0009)

    # sharp nicks with a raised burr thrown up on one side
    nick = t.scratches(density=260, length=0.0045, width=0.00055, seed=3)
    nick = sat(nick + t.scratches(density=120, length=0.0020, width=0.0008, seed=33) * 0.9)
    nsoft = t.blur(nick, 0.0005)
    burr = sat(t.shift(nsoft, 0.0007, 0.0004) * 2.2 - nick * 3.0)

    # casting / rolling porosity: clustered, mixed sizes 0.4-1.5 mm
    pclu = smoothstep(0.50, 0.78, t.fbm(0.045, 3, 0.5, seed=21))
    pore = t.speckle(density=2600, size=0.0011, seed=4, soft=0.45) * (0.08 + 0.92 * pclu)
    pore = sat(pore + t.speckle(density=420, size=0.0018, seed=34, soft=0.5) * (0.05 + 0.95 * pclu))

    # ---------------------------------------------------------------- polishing lines
    lf = t.fbm(0.0009, 2, 0.6, seed=5, stretch=st(45))                    # hairlines
    lc = t.fbm(0.0030, 2, 0.5, seed=35, stretch=st(30))                   # coarser grit tracks
    lm = t.fbm(0.0075, 2, 0.5, seed=37, stretch=st(22))                   # bands left by each stroke of the block
    line = (lf - 0.5) * 0.60 + (lc - 0.5) * 0.60 + (lm - 0.5) * 0.45
    line = t.warp(line, 0.0022, 0.22, seed=36)                            # the arm swings: lines arc gently
    sb = t.scratches(density=9000, length=0.034, width=0.00034, seed=7, angle=A0 - 19, angle_spread=9)
    sc = t.scratches(density=9000, length=0.030, width=0.00034, seed=8, angle=A0 + 27, angle_spread=12)
    zb = smoothstep(0.48, 0.66, t.fbm(0.075, 2, 0.5, seed=9))
    zc = smoothstep(0.50, 0.68, t.fbm(0.075, 2, 0.5, seed=19))
    cross = sat(sb * zb + sc * zc)                                        # later passes at other angles
    scr_f = t.scratches(density=1100, length=0.016, width=0.00035, seed=10)    # handling scuffs
    scr_f = sat(scr_f + t.scratches(density=260, length=0.040, width=0.00032, seed=38) * 0.8)
    scr_d = t.scratches(density=300, length=0.050, width=0.00055, seed=11)     # later, deeper scores
    pl = sat(0.5 + line - cross * 0.30)                                   # 0..1, grooves low

    height = wavy + meso - dent * 0.00034 - nick * 0.00013 + burr * 0.00005 - pore * 0.00007
    height = height + (pl - 0.5) * 0.000085 - scr_f * 0.000012 - scr_d * 0.000040

    # ---------------------------------------------------------------- tarnish thickness 0..1
    def arc_(a):                                                          # follow the same swing as the lines
        return t.warp(a, 0.0022, 0.22, seed=36)
    grain = t.fbm(0.0024, 2, 0.6, seed=13)
    mott = smoothstep(0.42, 0.78, t.fbm(0.010, 3, 0.55, seed=12))         # fine mottled film
    mott2 = smoothstep(0.45, 0.80, t.fbm(0.0065, 2, 0.55, seed=42))
    wipe = arc_(t.fbm(0.022, 3, 0.5, seed=14, stretch=st(8)))             # cloth wipes: long soft bands
    strk = arc_(t.fbm(0.0085, 3, 0.55, seed=43, stretch=st(7)))           # build-up lying along the lines
    fine = arc_(t.fbm(0.0034, 2, 0.55, seed=44, stretch=st(10)))
    large = t.fbm(0.11, 2, 0.5, seed=22)                                  # low-contrast, tile-scale

    # finger marks: 15-25 mm ovals, darker at the rim where the sweat dried
    fg = t.cells(0.045, 1.0, seed=15)
    fa = fg.rand(4) * math.pi
    fu = fg.u * np.cos(fa) + fg.v * np.sin(fa)
    fv = -fg.u * np.sin(fa) + fg.v * np.cos(fa)
    fr = np.sqrt((fu / (0.0080 + 0.0040 * fg.rand(2))) ** 2 + (fv / (0.0058 + 0.0028 * fg.rand(5))) ** 2)
    fr = t.warp(fr.astype(F), 0.0012, 0.008, seed=45)
    fclu = smoothstep(0.45, 0.60, t.fbm(0.085, 2, 0.5, seed=52))            # a hand grips in one place
    fsel = (fg.rand(1) < 0.20 + 0.55 * fclu) * (0.45 + 0.55 * fg.rand(3))
    smudge = fsel * (1 - smoothstep(0.80, 1.05, fr)) * (0.55 + 0.45 * smoothstep(0.45, 1.0, fr))
    smudge = smudge * (0.65 + 0.7 * (t.fbm(0.0016, 2, 0.6, seed=46) - 0.5))

    # drying tide-lines: a handful per tile, wobbly and broken into arcs
    dr = t.cells(0.20, 0.9, seed=16)
    dd = t.warp(dr.f1, 0.0050, 0.022, seed=47) + (t.fbm(0.006, 2, 0.5, seed=48) - 0.5) * 0.0016
    dR = 0.012 + 0.016 * dr.rand(2)
    dw = 0.00045 + 0.0009 * t.fbm(0.012, 2, 0.5, seed=49)
    arc = smoothstep(0.42, 0.62, t.fbm(0.030, 2, 0.5, seed=50))
    ring = np.exp(-((dd - dR) / dw) ** 2) * arc * (0.5 + 0.5 * dr.rand(3))
    inside = (1 - smoothstep(-0.002, 0.0, dd - dR)) * 0.5

    cav = t.cavity(height, 0.0030, 0.00012)
    dent_tarn = smoothstep(0.10, 0.75, dent) * (0.55 + 0.6 * (t.fbm(0.004, 3, 0.6, seed=51) - 0.5) * 2 * 0.5)
    recess = sat(dent_tarn * 0.48 + nsoft * 1.4 + pore * 0.75 + t.blur(pore, 0.0009) * 0.7 + scr_d * 0.55 + cav * 0.35)

    # where cloth and hands still pass the film stays thin: long bands with the polishing direction
    rub = smoothstep(0.50, 0.76, 0.55 * wipe + 0.25 * t.fbm(0.075, 2, 0.5, seed=17, stretch=st(2.5)) + 0.20 * strk)
    film = 0.13 + 0.10 * (large - 0.5) + 0.26 * smoothstep(0.40, 0.80, strk) + 0.14 * smoothstep(0.45, 0.85, fine)
    film = film + 0.15 * mott + 0.08 * mott2 + 0.20 * smudge + 0.32 * ring + 0.05 * inside
    film = film + 0.36 * hollow * (0.35 + 0.65 * mott) * (0.6 + 0.8 * strk)   # the cloth skips the hollows
    film = film * (0.74 + 0.52 * grain)
    film = film * (1 - 0.85 * rub * (0.60 + 0.40 * crest)) * (1 - 0.25 * crest)
    film = film - (pl - 0.5) * 0.16                                       # the film sits in the grooves, crests are rubbed
    film = 0.80 * (1.0 - np.exp(-np.maximum(film, 0.0) * 1.7))            # thick film is rare; it saturates slowly
    tarn = sat(film) * (1 - 0.75 * scr_f) * (1 - 0.3 * cross)             # fresh scuffs cut through it
    tarn = sat(tarn + recess * 0.70)

    # ---------------------------------------------------------------- colour (linear)
    albedo = ramp(tarn, [(0.0, "#dcc686"), (0.22, "#cfb577"), (0.45, "#b3975e"),
                         (0.68, "#8a6c40"), (0.86, "#5c462c"), (1.0, "#3d2f21")])
    albedo = albedo * (1.0 + 0.06 * (pl - 0.5) * 2)[..., None]
    red = smoothstep(0.55, 0.85, t.fbm(0.035, 3, 0.5, seed=18)) * smoothstep(0.25, 0.6, tarn)
    albedo = overlay_tint(albedo, "#e2b9a0", red * 0.30)                  # coppery where the zinc has leached
    albedo = mix(albedo, hexlin("#e2cc8a"), sat(scr_f * 0.40 + burr * 0.5 + scr_d * 0.10))

    # ---------------------------------------------------------------- PBR from the tarnish thickness
    rough = 0.265 + 0.125 * smoothstep(0.10, 0.34, tarn) + 0.36 * smoothstep(0.36, 0.86, tarn)
    rough = rough + 0.07 * (0.5 - pl) * 2 * (1 - 0.5 * tarn) + 0.04 * (grain - 0.5)
    rough = mix(rough, 0.24, sat(scr_f * 0.6 + burr * 0.6))
    metal = 1.0 - 0.40 * smoothstep(0.42, 0.70, tarn) - 0.45 * smoothstep(0.72, 0.95, tarn)
    return dict(albedo=albedo, height=height, rough=rough, metal=metal)
