"""Old hand-made red clay brick, running bond (225 x 75 mm module = 215 x 65 brick + 10 mm lime joint).

Built the way the wall is built: each brick is its own slightly mis-sized, slightly crooked
rounded box (hand-thrown into a mould, so no arris is straight), bedded in a recessed sandy lime
mortar. Clamp firing gives each brick its own colour and a darker / lighter end; frost has
blown the face off a few and chipped the arrises of many.
"""
from tk.tex import *

META = {"normal_strength": 1.0}


def build(t):
    lay = t.grid(0.225, 0.075, shift=0.5, seed=3)          # 4 stretchers x 12 courses per 0.9 m
    R = lay.rand

    # ------------------------------------------------------------------ brick outlines
    # every brick: own size (+-2 mm), own position in its bed, own small rotation
    ang = (R(1) - 0.5) * 0.022
    ca, sa = np.cos(ang), np.sin(ang)
    u0 = lay.u + (R(2) - 0.5) * 0.004
    v0 = lay.v + (R(3) - 0.5) * 0.0025
    u = u0 * ca + v0 * sa
    v = -u0 * sa + v0 * ca
    hx = 0.1075 - 0.0025 * R(4)
    hy = 0.0325 - 0.0020 * R(5)
    # hand-made bricks are slightly banana'd / bellied: the long edges bow by a mm or so
    v = v + (R(6) - 0.5) * 0.004 * (u / 0.11) ** 2
    rad = 0.003 + 0.004 * R(7)                              # corner radius (worn bricks rounder)
    qx = np.abs(u) - (hx - rad)
    qy = np.abs(v) - (hy - rad)
    sd = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - rad
    d = -sd                                                  # metres inside the brick (+) / into the joint (-)
    # wobbly arrises (two scales) + bites out of the edges
    d = d + (t.noise(0.05, seed=11) - 0.5) * 0.0055 + (t.noise(0.014, seed=12) - 0.5) * 0.0016
    chipn = t.fbm(0.022, 3, 0.55, seed=13)
    chip = smoothstep(0.64, 0.78, chipn + 0.14 * (R(8) - 0.5))
    d = d - chip * 0.007 * (1 - smoothstep(0.004, 0.016, d))

    arris = 0.0035 + 0.004 * R(9)                            # eased, worn arris width
    prof = sat((d + 0.0005) / arris)
    prof = 1 - (1 - prof) ** 2.2                             # 0 at the mortar line -> 1 on the face
    inside = smoothstep(-0.0008, 0.0012, d)

    # ------------------------------------------------------------------ brick face height
    lvl = (R(10) - 0.5) * 0.0026 + lay.tilt(0.0011, 1)      # proud / shy / crooked bricks
    bow = (t.noise(0.07, seed=14) - 0.5) * 0.0030            # hand-moulded faces are never flat
    bow = bow + (R(17) - 0.45) * 0.0022 * (1 - (v / 0.033) ** 2) * (1 - 0.5 * (u / 0.11) ** 2)   # bellied / dished
    sandf = t.fbm(0.006, 4, 0.62, seed=15)                   # sand-struck skin
    drag = t.fbm(0.02, 3, 0.55, seed=16, stretch=(5.0, 1.0)) # strike / drag marks along the brick
    crease = sat((t.ridged(0.03, 3, 0.5, seed=17, stretch=(3.0, 1.0)) - 0.80) / 0.2)
    crease = crease * smoothstep(0.5, 0.62, t.noise(0.09, seed=18))          # only on some faces
    pits = t.speckle(density=1100, size=0.006, seed=19)      # burnt-out inclusions, stone pops
    pits = pits * smoothstep(0.35, 0.6, t.noise(0.12, seed=24))
    pores = t.speckle(density=30000, size=0.0022, seed=20)

    def moved(a, fx, fy):                                    # periodic re-use of a dot layer elsewhere
        return np.roll(a, (int(t.h * fy), int(t.w * fx)), axis=(0, 1))
    face = (bow + (sandf - 0.5) * 0.0010 + (drag - 0.5) * 0.0010
            - crease * 0.0013 - pits * 0.0016 - pores * 0.0005)

    # frost spalling: the fired skin blown off a few bricks, leaving a rough granular hollow
    # with a crisp lip; it starts from an arris and eats into the face
    sp_n = t.fbm(0.075, 2, 0.4, seed=21)
    sp_n = t.sample(sp_n, t.X + R(18) * t.W, t.Y + R(19) * t.H)     # every brick fails in its own way
    pick = (R(11) > 0.78).astype(F)                          # about a quarter of the bricks
    edge_bias = 0.13 * (1 - smoothstep(0.0, 0.028, d))
    thr = 0.60 - 0.10 * R(16)
    spall = smoothstep(thr, thr + 0.02, sp_n + edge_bias + (t.noise(0.012, seed=23) - 0.5) * 0.05) * pick * inside
    sp_floor = t.fbm(0.012, 4, 0.65, seed=22)
    face = face * (1 - 0.7 * spall) - spall * (0.0022 + 0.0030 * sp_floor)

    brick_h = lvl + face

    # ------------------------------------------------------------------ mortar bed
    grit = t.fbm(0.0045, 4, 0.7, seed=30)
    agg = moved(pores, 0.37, 0.61)     # sharp sand / small flint showing
    erode = t.fbm(0.16, 3, 0.5, seed=32)                     # some joints weathered back further
    mortar_h = -0.0055 - 0.0045 * smoothstep(0.45, 0.8, erode) + (grit - 0.5) * 0.0016 + agg * 0.0006
    # mortar is struck off against the arrises: it rises a little where it meets the brick
    mortar_h = mortar_h + 0.0012 * smoothstep(-0.004, 0.0, d)

    height = np.where(inside > 0, mix(mortar_h, mortar_h + (brick_h - mortar_h) * prof, inside), mortar_h)

    # ------------------------------------------------------------------ masks from the relief
    cav = t.cavity(height, 0.003, 0.0006)
    edg = t.edges(height, 0.004, 0.0009)

    # ------------------------------------------------------------------ brick colour
    tone = R(12)
    body = ramp(tone, [(0.00, "#684d45"),     # over-burnt, purplish brown
                       (0.14, "#82544a"),
                       (0.34, "#955d49"),     # the common soft red
                       (0.60, "#a0664e"),
                       (0.82, "#a87154"),     # orange place brick
                       (1.00, "#b08a6c")])    # pale, under-fired salmon
    # clamp firing: one end of the brick sat nearer the fire
    flash = (u / 0.11) * (R(13) - 0.5) * 0.42 + (v / 0.033) * (R(14) - 0.5) * 0.10
    mott = t.fbm(0.035, 4, 0.55, seed=40)
    cloud = t.fbm(0.012, 3, 0.6, seed=41)
    val = 1.0 + flash + (mott - 0.5) * 0.40 + (cloud - 0.5) * 0.16 + (sandf - 0.5) * 0.10
    body = body * val[..., None]
    # reduction core / hearts: dusky purple-grey cloud in the middle of some faces
    heart = smoothstep(0.55, 0.8, t.fbm(0.05, 3, 0.5, seed=42)) * smoothstep(0.55, 0.9, R(15)) \
        * smoothstep(0.004, 0.02, d)
    body = mix(body, body * hexlin("#8a7f86") * 1.9, heart * 0.55)
    # iron spots and pale grog / lime nodules
    spots = moved(pores, 0.71, 0.23) * smoothstep(0.4, 0.6, cloud)
    grog = t.speckle(density=9000, size=0.003, seed=44)
    body = mix(body, hexlin("#3d2c28"), spots * 0.6)
    body = mix(body, hexlin("#c2aa8e"), grog * 0.5)
    # exposed interior where the face has spalled: paler, dustier, more orange
    inner = desaturate(body, 0.15) * (0.95 + 0.45 * sp_floor)[..., None] * hexlin("#fff0e0") + hexlin("#2a1a12") * 0.25
    body = mix(body, inner, spall * 0.9)
    # lime bloom: mortar smeared over the arrises and a thin pale wash on some faces
    bloom_n = t.fbm(0.06, 4, 0.55, seed=45)
    smear = (1 - smoothstep(0.0, 0.012, d)) * smoothstep(0.35, 0.7, t.fbm(0.03, 3, 0.55, seed=46))
    bloom = sat(smoothstep(0.58, 0.85, bloom_n) * 0.45 + smear * 0.55) * (1 - spall)
    body = mix(body, desaturate(body, 0.6) * 1.25 + hexlin("#3a362e") * 0.35, bloom * 0.55)
    # a thin grey weathering / soot film clings to parts of the skin
    soot = smoothstep(0.52, 0.85, t.fbm(0.09, 4, 0.6, seed=49)) * (1 - spall)
    body = mix(body, desaturate(body, 0.5) * hexlin("#b4aaa2") * 1.25, soot * 0.6)
    # soot and dirt sit in the pits, creases and sand texture
    body = body * (1 - 0.45 * cav)[..., None]
    body = mix(body, body * 1.12 + 0.012, edg * 0.5)

    # ------------------------------------------------------------------ mortar colour
    mcol = ramp(grit, [(0.0, "#857b68"), (0.45, "#a89e8a"), (1.0, "#c4baa3")])
    mcol = mix(mcol, hexlin("#4d463d"), agg * 0.45)
    mcol = mix(mcol, hexlin("#d2ccbd"), moved(grog, 0.43, 0.29) * 0.5)   # lime lumps
    mcol = mcol * (0.80 + 0.30 * t.fbm(0.11, 3, 0.5, seed=48))[..., None]
    mcol = mcol * (1 - 0.35 * smoothstep(0.45, 0.8, erode))[..., None]      # deep joints hold dirt

    albedo = mix(mcol, body, inside)
    macro = t.fbm(0.45, 2, 0.5, seed=50)
    albedo = albedo * (0.94 + 0.12 * macro)[..., None]
    albedo = np.clip(albedo, 0.015, 0.85)

    # ------------------------------------------------------------------ roughness
    dense = 1 - smoothstep(0.0, 0.3, tone)                   # over-burnt bricks are glassier
    skin = t.fbm(0.05, 3, 0.5, seed=51)                      # patches where the fired skin is tight / worn smooth
    rb = 0.80 - 0.20 * dense - 0.10 * (R(20) - 0.5) + (sandf - 0.5) * 0.16 - 0.16 * smoothstep(0.5, 0.8, skin) \
        + 0.14 * spall + 0.10 * bloom - 0.10 * edg + 0.10 * cav
    rough = mix(0.95 - 0.05 * agg, rb, inside)
    return dict(albedo=albedo, height=height, rough=np.clip(rough, 0.3, 1.0), metal=0.0)
