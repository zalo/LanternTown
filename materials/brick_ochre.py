"""Buff / yellow stock brick, Flemish-type bond with dark flared headers and sooty faces.

Module: stretcher 225 mm, header 112.5 mm, course 75 mm (215 x 102.5 x 65 brick + 10 mm joint).
A 0.9 m tile takes S H S H S per course; successive courses step by 3/4 of a stretcher so every
header sits centred over a stretcher (the Flemish rule, quarter-brick lap) while the direction
of the step varies, so the header pattern wanders instead of forming a countable grid.

Stock bricks are soft-edged, coarse and full of burnt-out ash ("Spanish"): pocked faces, black
clinker specks, colour from pale yellow to dull brown, headers burnt dark grey-brown, and decades
of coal smoke sitting as a black film on the more absorbent faces and in every pit.
"""
from tk.tex import *

META = {"normal_strength": 1.0}

S_, H_ = 0.225, 0.1125
STEPS = (1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1)          # 6 up / 6 down: wraps after 12 courses


def bond(t, seed=5):
    """Flemish-ish bond layout -> (Layout, half_length per pixel, is_header per pixel)."""
    rows = 12
    ch = t.H / rows
    k = t.W / 0.9
    edges = np.array([0.0, S_, S_ + H_, 2 * S_ + H_, 2 * S_ + 2 * H_, 3 * S_ + 2 * H_]) * k
    kinds = np.array([0, 1, 0, 1, 0])
    offs = np.concatenate([[0], np.cumsum(STEPS)[:-1]]) * 0.16875 * k + 0.031 * k
    r = np.floor(t.Y / ch).astype(np.int64) % rows
    xs = (t.X - offs[r]) % t.W
    c = np.clip(np.searchsorted(edges, xs, side='right') - 1, 0, 4)
    x0, x1 = edges[c], edges[c + 1]
    yb = t.Y % ch
    dist = np.minimum(np.minimum(xs - x0, x1 - xs), np.minimum(yb, ch - yb))
    lay = Layout(dist, r * 5 + c, xs - (x0 + x1) / 2, yb - ch / 2, rows * 5, seed, S_, ch)
    return lay, ((x1 - x0) / 2).astype(F), kinds[c].astype(F)


def build(t):
    lay, hcell, header = bond(t)
    R = lay.rand

    # ------------------------------------------------------------------ brick outlines
    ang = (R(1) - 0.5) * 0.024 * (1 + header)
    ca, sa = np.cos(ang), np.sin(ang)
    u0 = lay.u + (R(2) - 0.5) * 0.004
    v0 = lay.v + (R(3) - 0.5) * 0.0025
    u = u0 * ca + v0 * sa
    v = -u0 * sa + v0 * ca
    hx = hcell - 0.005 - 0.0025 * R(4)
    hy = 0.0325 - 0.0022 * R(5)
    un = u / hcell
    v = v + (R(6) - 0.5) * 0.004 * un ** 2                    # banana'd stocks
    rad = 0.004 + 0.005 * R(7)                               # soft, rounded corners
    qx = np.abs(u) - (hx - rad)
    qy = np.abs(v) - (hy - rad)
    sd = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - rad
    d = -sd
    d = d + (t.noise(0.05, seed=11) - 0.5) * 0.006 + (t.noise(0.015, seed=12) - 0.5) * 0.002
    chipn = t.fbm(0.024, 3, 0.55, seed=13)
    chip = smoothstep(0.62, 0.78, chipn + 0.14 * (R(8) - 0.5))
    d = d - chip * 0.008 * (1 - smoothstep(0.004, 0.018, d))

    arris = 0.0045 + 0.0045 * R(9)
    prof = sat((d + 0.0005) / arris)
    prof = 1 - (1 - prof) ** 2.2
    inside = smoothstep(-0.0008, 0.0012, d)

    # ------------------------------------------------------------------ brick face height
    lvl = (R(10) - 0.5) * 0.0030 + lay.tilt(0.0012, 1) * (1 - 0.5 * header)
    bow = (t.noise(0.07, seed=14) - 0.5) * 0.0032
    bow = bow + (R(17) - 0.45) * 0.0022 * (1 - (v / 0.033) ** 2) * (1 - 0.5 * un ** 2)
    skin = t.fbm(0.007, 4, 0.62, seed=15)                    # coarse open texture
    lumps = t.fbm(0.022, 3, 0.55, seed=16)                   # lumpy, badly mixed clay
    crease = sat((t.ridged(0.035, 3, 0.5, seed=17, stretch=(2.5, 1.0)) - 0.80) / 0.2)
    crease = crease * smoothstep(0.48, 0.62, t.noise(0.09, seed=18))
    pocks = t.speckle(density=1500, size=0.007, seed=19)    # burnt-out ash pockets
    pocks = pocks * smoothstep(0.3, 0.55, t.noise(0.10, seed=24))
    fine = t.speckle(density=30000, size=0.0024, seed=20)

    def moved(a, fx, fy):
        return np.roll(a, (int(t.h * fy), int(t.w * fx)), axis=(0, 1))

    face = (bow + (skin - 0.5) * 0.0012 + (lumps - 0.5) * 0.0014
            - crease * 0.0014 - pocks * 0.0022 - fine * 0.0006)

    # a few faces have blown / flaked
    sp_n = t.fbm(0.07, 2, 0.4, seed=21)
    sp_n = t.sample(sp_n, t.X + R(18) * t.W, t.Y + R(19) * t.H)
    pick = (R(11) > 0.84).astype(F)
    edge_bias = 0.13 * (1 - smoothstep(0.0, 0.028, d))
    thr = 0.60 - 0.08 * R(16)
    spall = smoothstep(thr, thr + 0.02, sp_n + edge_bias + (t.noise(0.012, seed=23) - 0.5) * 0.05) * pick * inside
    sp_floor = t.fbm(0.012, 4, 0.65, seed=22)
    face = face * (1 - 0.7 * spall) - spall * (0.0022 + 0.0030 * sp_floor)
    brick_h = lvl + face

    # ------------------------------------------------------------------ mortar bed
    grit = t.fbm(0.0045, 4, 0.7, seed=30)
    agg = moved(fine, 0.37, 0.61)
    erode = t.fbm(0.16, 3, 0.5, seed=32)
    mortar_h = -0.0060 - 0.0045 * smoothstep(0.45, 0.8, erode) + (grit - 0.5) * 0.0016 + agg * 0.0006
    mortar_h = mortar_h + 0.0012 * smoothstep(-0.004, 0.0, d)
    height = np.where(inside > 0, mix(mortar_h, mortar_h + (brick_h - mortar_h) * prof, inside), mortar_h)

    cav = t.cavity(height, 0.003, 0.0006)
    edg = t.edges(height, 0.004, 0.0009)

    # ------------------------------------------------------------------ brick colour
    tone = R(12)
    buff = ramp(tone, [(0.00, "#8c705a"),     # dull brown stock
                       (0.20, "#a5855e"),
                       (0.45, "#b69465"),     # the common yellow-buff
                       (0.70, "#c0a070"),
                       (0.88, "#c6ab85"),     # pale malm
                       (1.00, "#b38a70")])    # pinkish, picked up iron
    dark = ramp(R(21), [(0.0, "#5c5651"),     # flared headers: grey-brown to blue-grey, slightly vitrified
                        (0.5, "#695d50"),
                        (1.0, "#786754")])
    is_dark = header * smoothstep(0.22, 0.62, R(22))         # most headers flared, to varying degrees
    body = mix(buff, dark, is_dark)
    flash = un * (R(13) - 0.5) * 0.36 + (v / 0.033) * (R(14) - 0.5) * 0.10
    mott = t.fbm(0.035, 4, 0.55, seed=40)
    cloud = t.fbm(0.012, 3, 0.6, seed=41)
    val = 1.0 + flash + (mott - 0.5) * 0.36 + (cloud - 0.5) * 0.16 + (skin - 0.5) * 0.10
    body = body * val[..., None]
    # purple-brown scorch clouds on some buff faces
    scorch = smoothstep(0.55, 0.8, t.fbm(0.05, 3, 0.5, seed=42)) * smoothstep(0.5, 0.9, R(15)) \
        * smoothstep(0.004, 0.02, d) * (1 - is_dark)
    body = mix(body, body * hexlin("#9a8580") * 1.7, scorch * 0.6)
    # black clinker / ash specks, pale chalk nodules
    clinker = moved(fine, 0.71, 0.23) * smoothstep(0.35, 0.55, cloud)
    chalk = t.speckle(density=7000, size=0.0032, seed=44)
    body = mix(body, hexlin("#2b2725"), clinker * 0.7)
    body = mix(body, hexlin("#cfc6ae"), chalk * 0.45 * (1 - is_dark))
    inner = desaturate(body, 0.1) * (1.0 + 0.45 * sp_floor)[..., None] + hexlin("#2a2412") * 0.3
    body = mix(body, inner, spall * 0.9)
    # mortar smeared over the arrises
    smear = (1 - smoothstep(0.0, 0.012, d)) * smoothstep(0.35, 0.7, t.fbm(0.03, 3, 0.55, seed=46))
    body = mix(body, desaturate(body, 0.6) * 1.15 + hexlin("#35322c") * 0.3, smear * 0.4 * (1 - spall))
    # coal soot: a black film on the more absorbent bricks, patchy, thickest in the texture
    soot_n = t.fbm(0.16, 3, 0.45, seed=49)
    thirsty = R(23) ** 1.6                                   # most bricks only lightly, a few heavily
    soot = smoothstep(0.25, 0.95, soot_n * 0.6 + 0.75 * thirsty + 0.10 * (0.5 - skin)) * (1 - spall)
    soot = soot * smoothstep(-0.002, 0.012, d) ** 0.5       # rain keeps the arrises a little cleaner
    body = mix(body, hexlin("#4c4138") * (0.85 + 0.3 * mott)[..., None], soot * 0.55 * (1 - 0.5 * is_dark))
    body = body * (1 - 0.50 * cav)[..., None]
    body = mix(body, body * 1.10 + 0.012, edg * 0.5 * (1 - soot))

    # ------------------------------------------------------------------ mortar: grey, ashy lime
    mcol = ramp(grit, [(0.0, "#787265"), (0.45, "#999283"), (1.0, "#b8b09d")])
    mcol = mix(mcol, hexlin("#39352f"), agg * 0.5)
    mcol = mix(mcol, hexlin("#cdc8ba"), moved(chalk, 0.43, 0.29) * 0.45)
    mcol = mcol * (0.80 + 0.30 * t.fbm(0.11, 3, 0.5, seed=48))[..., None]
    mcol = mcol * (1 - 0.35 * smoothstep(0.45, 0.8, erode))[..., None]

    albedo = mix(mcol, body, inside)
    macro = t.fbm(0.45, 2, 0.5, seed=50)
    albedo = albedo * (0.94 + 0.12 * macro)[..., None]
    albedo = np.clip(albedo, 0.015, 0.85)

    # ------------------------------------------------------------------ roughness
    tight = t.fbm(0.05, 3, 0.5, seed=51)
    rb = 0.84 - 0.22 * is_dark - 0.08 * (R(20) - 0.5) + (skin - 0.5) * 0.16 - 0.12 * smoothstep(0.5, 0.8, tight) \
        + 0.12 * spall + 0.08 * soot - 0.08 * edg + 0.10 * cav
    rough = mix(0.95 - 0.05 * agg, rb, inside)
    return dict(albedo=albedo, height=height, rough=np.clip(rough, 0.3, 1.0), metal=0.0)
