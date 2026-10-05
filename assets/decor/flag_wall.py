"""Wall flag on an angled pole. A cast-iron wall plate with a swivel knuckle carries a socket tube set at
~45 degrees; a turned wooden pole with a stop collar sits in the socket, clamped by a thumbscrew, and ends
in a gilt finial. A hemp halyard runs from a horn cleat over a brass truck sheave; the flag's canvas
heading is clipped to it with brass rings through grommets. The cloth is an inextensible sheet hung from
the hoist: it drops under gravity (plus a little breeze), gathers into deep vertical folds, has bound
edges and an appliqued emblem on both faces. Cloth, emblem, grommets and rings form the 'flutter' part.
"""
from math import sin, cos, pi, radians, degrees, sqrt, exp

import bmesh
from mathutils import Vector

from tk import *

META = {"name": "Wall flag on angled pole", "class": "component", "mount": "wall", "mount_height": 3.3,
        "focus": [-0.8, -0.62, 0.38],
        "tri_budget": 15000}        # deeply folded double-sided cloth needs the resolution
PARAMS = {
    "pole_length": {"default": 2.2, "min": 1.4, "max": 3.2, "desc": "pole length from the knuckle to the finial base (m)"},
    "pole_diameter": {"default": 0.042, "min": 0.032, "max": 0.06, "desc": "pole diameter at the socket (m)"},
    "angle": {"default": 45.0, "min": 30.0, "max": 60.0, "desc": "pole elevation above horizontal (deg)"},
    "yaw": {"default": -55.0, "min": -70.0, "max": 70.0, "desc": "swivel of the knuckle sideways from square-to-wall (deg)"},
    "flag_hoist": {"default": 0.9, "min": 0.5, "max": 1.5, "desc": "flag height = edge along the pole (m)"},
    "flag_fly": {"default": 1.35, "min": 0.7, "max": 2.2, "desc": "flag length = edge hanging away from the pole (m)"},
    "pattern": {"default": "plain", "choices": ["plain", "bicolour", "quartered"], "desc": "field division"},
    "colour_a": {"default": "#9a1f26", "desc": "main field colour"},
    "colour_b": {"default": "#e6dcc0", "desc": "second field colour (bicolour / quartered)"},
    "trim": {"default": "#d8ae45", "desc": "colour of the emblem applique and the edge binding"},
    "emblem": {"default": "crown", "choices": ["crown", "star", "sun", "moon", "none"], "desc": "appliqued emblem"},
    "emblem_size": {"default": 0.7, "min": 0.3, "max": 0.85, "desc": "emblem size as a fraction of the hoist"},
    "drape": {"default": 1.0, "min": 0.4, "max": 1.6, "desc": "fold depth / how much the cloth gathers"},
    "breeze": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "how far a light wind along the wall lifts the cloth"},
    "fade": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "sun-fade and soiling of the cloth"},
    "finial": {"default": "ball", "choices": ["ball", "spear", "acorn"], "desc": "gilt finial shape"},
    "finial_size": {"default": 1.0, "min": 0.7, "max": 1.5, "desc": "finial scale"},
    "pole_finish": {"default": "oak", "choices": ["oak", "painted", "tarred"], "desc": "pole finish"},
    "pole_colour": {"default": "#e9e2d0", "desc": "paint colour when pole_finish = painted"},
    "metal": {"default": "#34443b", "desc": "paint colour of the cast-iron bracket"},
    "stay": {"default": False, "desc": "wrought-iron tie rod from a wall eye down to a collar on the pole"},
    "tassels": {"default": True, "desc": "pair of cord tassels hung from the pole head"},
}
VARIANTS = {
    "guild_quartered": {"pattern": "quartered", "colour_a": "#1f4476", "colour_b": "#e6dcc0", "emblem": "star",
                        "pole_length": 2.9, "flag_hoist": 1.15, "flag_fly": 1.7, "yaw": -61, "finial": "spear",
                        "pole_finish": "painted", "stay": True, "breeze": 0.45, "seed": 7},
    "small_bicolour": {"pattern": "bicolour", "colour_a": "#1f5e3d", "colour_b": "#ead9a8", "emblem": "sun",
                       "trim": "#c8452c", "pole_length": 1.6, "pole_diameter": 0.034, "flag_hoist": 0.62, "flag_fly": 0.9,
                       "angle": 36, "yaw": -64, "breeze": 0.7, "finial": "acorn", "finial_size": 0.8, "seed": 13},
    "moon_banner": {"pattern": "plain", "colour_a": "#d29a34", "trim": "#27407c", "emblem": "moon", "emblem_size": 0.75,
                    "pole_length": 2.4, "flag_hoist": 0.85, "flag_fly": 1.8, "angle": 55, "yaw": -58, "drape": 1.3,
                    "breeze": 0.32, "tassels": False, "pole_finish": "tarred", "metal": "#2a1d14", "seed": 21},
    "old_faded": {"pattern": "bicolour", "colour_a": "#a8322a", "colour_b": "#d9b64a", "emblem": "none", "trim": "#8a7a5a",
                  "pole_length": 1.9, "flag_hoist": 0.75, "flag_fly": 1.1, "yaw": -65, "tassels": False, "breeze": 0.65, "drape": 0.7,
                  "fade": 1.0, "finial_size": 1.3, "metal": "#3a2020", "seed": 34},
}


# ------------------------------------------------------------------------------------------- helpers
def _rot_to(v):
    """Euler degrees that turn local +Z onto direction v."""
    e = Vector((0, 0, 1)).rotation_difference(Vector(v).normalized()).to_euler()
    return (degrees(e.x), degrees(e.y), degrees(e.z))


def _tri_fixed(o):
    """Triangulate quads on the v0-v2 diagonal so the emblem applique can follow the exact same facets."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='FIXED', ngon_method='BEAUTY')
    bm.to_mesh(o.data)
    bm.free()


def _clip(poly, tri):
    """Sutherland-Hodgman: clip polygon `poly` by the convex CCW triangle `tri` (2D)."""
    out = poly
    for k in range(3):
        ax, ay = tri[k]
        bx, by = tri[(k + 1) % 3]
        inp, out = out, []
        if len(inp) < 3:
            return []
        prev = inp[-1]
        sp = (bx - ax) * (prev[1] - ay) - (by - ay) * (prev[0] - ax)
        for cur in inp:
            sc = (bx - ax) * (cur[1] - ay) - (by - ay) * (cur[0] - ax)
            if (sc >= 0) != (sp >= 0):
                t = sp / (sp - sc)
                out.append((prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t))
            if sc >= 0:
                out.append(cur)
            prev, sp = cur, sc
    return out


def _area(poly):
    return 0.5 * sum(poly[k][0] * poly[(k + 1) % len(poly)][1] - poly[(k + 1) % len(poly)][0] * poly[k][1]
                     for k in range(len(poly)))


def _emblem_polys(kind):
    """Emblem as 2D polygons in a unit box (x right, y up), each a separate felt piece."""
    def circ(cx, cy, r, n=12):
        return [(cx + r * cos(2 * pi * k / n), cy + r * sin(2 * pi * k / n)) for k in range(n)]

    def star(cx, cy, ro, ri, n=5, a0=0.0):
        return [(cx + (ro if k % 2 == 0 else ri) * sin(a0 + pi * k / n), cy + (ro if k % 2 == 0 else ri) * cos(a0 + pi * k / n))
                for k in range(2 * n)]

    if kind == "star":
        return [star(0, -0.04, 1.0, 0.41)]
    if kind == "sun":
        polys = [circ(0, 0, 0.47, 20)]
        for k in range(12):
            a = 2 * pi * k / 12
            ro = 1.0 if k % 2 == 0 else 0.84
            polys.append([(0.57 * cos(a - 0.17), 0.57 * sin(a - 0.17)), (0.57 * cos(a + 0.17), 0.57 * sin(a + 0.17)),
                          (ro * cos(a), ro * sin(a))])
        return polys
    if kind == "crown":
        polys = [[(-0.8, -0.78), (0.8, -0.78), (0.8, -0.56), (-0.8, -0.56)],
                 [(-0.8, -0.47), (0.8, -0.47), (1.0, 0.36), (0.66, -0.06), (0.5, 0.56), (0.25, -0.06), (0.0, 0.74),
                  (-0.25, -0.06), (-0.5, 0.56), (-0.66, -0.06), (-1.0, 0.36)]]
        for (x, y, r) in ((-1.0, 0.5, 0.085), (1.0, 0.5, 0.085), (-0.5, 0.7, 0.085), (0.5, 0.7, 0.085), (0.0, 0.89, 0.1)):
            polys.append(circ(x, y, r, 10))
        return polys
    if kind == "moon":
        R, r, d = 0.92, 0.74, 0.34
        xi = (d * d + R * R - r * r) / (2 * d)
        yi = sqrt(max(R * R - xi * xi, 1e-6))
        from math import atan2
        a_o = atan2(yi, xi)
        a_i = atan2(yi, xi - d)
        outer = [(R * cos(a_o + (2 * pi - 2 * a_o) * k / 22) - 0.15, R * sin(a_o + (2 * pi - 2 * a_o) * k / 22)) for k in range(23)]
        inner = [(d + r * cos(-a_i - (2 * pi - 2 * a_i) * k / 16) - 0.15, r * sin(-a_i - (2 * pi - 2 * a_i) * k / 16)) for k in range(1, 16)]
        return [outer + inner, star(0.5, 0.02, 0.33, 0.14, a0=0.3)]
    return []


# --------------------------------------------------------------------------------------------- build
def build(p, rng):
    el, yaw = radians(p.angle), radians(p.yaw)
    ROT = (90.0 - p.angle, 0.0, p.yaw)
    K = Vector((0.0, -0.046, 0.0))                       # knuckle centre = foot of the pole axis
    Hd = Vector((sin(yaw), -cos(yaw), 0.0))              # horizontal direction of the pole
    Lx = Vector((cos(yaw), sin(yaw), 0.0))               # pole-frame +X (sideways)
    Ax = Hd * cos(el) + Vector((0, 0, sin(el)))          # pole-frame +Z (pole axis)
    Uy = Ax.cross(Lx)                                    # pole-frame +Y (upper side of the pole)

    def F(x, y, z):
        return K + Lx * x + Uy * y + Ax * z

    def to_pole(o):
        apply_transform(o)
        place(o, loc=tuple(K), rot=ROT)
        return o

    iron = mat("iron_painted", p.metal)
    brass = mat("brass", rough=0.85)
    gilt = mat("metal_plain", "#d4a84b", rough=0.42)
    rope_m = mat("rope", "#cdbf9c")
    if p.pole_finish == "painted":
        wood = mat("wood_painted", p.pole_colour)
    elif p.pole_finish == "tarred":
        wood = mat("wood_oak", "#3a2a1e")
    else:
        wood = mat("wood_oak", "#8a6238")

    Lp = p.pole_length
    r0 = p.pole_diameter / 2
    r1 = r0 * 0.8
    Lt = 0.2                                             # socket tube length
    Ro = r0 + 0.008

    def rz(z):
        return r0 + (r1 - r0) * clamp((z - Lt - 0.1) / max(Lp - Lt - 0.21, 0.1))

    # ================================================================== wall bracket (cast iron)
    body = []
    ear = [(0.036 * cos(radians(a)), -0.118 + 0.036 * sin(radians(a))) for a in range(-90, 41, 26)]
    half = ear + [(0.058, -0.095), (0.068, -0.091), (0.071, -0.082), (0.071, 0.082), (0.068, 0.091), (0.058, 0.095)] + \
        [(x, -z) for (x, z) in reversed(ear)]
    outline = half + [(-x, z) for (x, z) in reversed(half[1:-1])]
    body.append(shape_extrude("wall_plate", outline, depth=0.011, loc=(0, -0.0015, -0.012), rot=(90, 0, 0), bevel=0.0025))
    body.append(shape_extrude("plate_rim", rounded_rect(0.118, 0.156, 0.012, 3), [rounded_rect(0.1, 0.138, 0.008, 3)],
                              depth=0.004, loc=(0, -0.0125, -0.012), rot=(90, 0, 0), bevel=0.0012))
    body.append(box("neck", (0.054, 0.04, 0.124), loc=(0, -0.031, -0.012), bevel=0.005))
    body.append(lathe("knuckle", [(0.0, -0.09), (0.016, -0.09), (0.024, -0.084), (0.03, -0.074), (0.037, -0.07), (0.037, -0.056),
                                  (0.0335, -0.051), (0.0335, 0.03), (0.037, 0.035), (0.037, 0.047), (0.031, 0.053),
                                  (0.02, 0.059), (0.009, 0.062), (0.007, 0.068), (0.011, 0.076), (0.008, 0.084), (0.0, 0.087)],
                      segments=16, loc=tuple(K)))
    # socket tube (pole frame): reinforcing bands, flared mouth
    tube = lathe("socket", [(0.0, 0.0), (Ro, 0.0), (Ro, 0.04), (Ro + 0.0045, 0.045), (Ro + 0.0045, 0.058), (Ro, 0.063),
                            (Ro - 0.002, 0.08), (Ro - 0.003, Lt - 0.034), (Ro + 0.002, Lt - 0.028), (Ro + 0.006, Lt - 0.018),
                            (Ro + 0.006, Lt - 0.006), (Ro + 0.003, Lt), (r0 + 0.0012, Lt), (r0 + 0.0012, Lt - 0.012)], segments=20)
    body.append(to_pole(tube))
    # thumbscrew clamping the pole
    zs_ = Lt - 0.062
    body.append(to_pole(cylinder("screw_boss", 0.0105, 0.012, loc=(Ro + 0.002, 0, zs_), rot=(0, 90, 0), verts=10, bevel=0.0015)))
    scr = [to_pole(cylinder("screw_shaft", 0.0042, 0.03, loc=(Ro + 0.02, 0, zs_), rot=(0, 90, 0), verts=8)),
           to_pole(box("screw_wing", (0.0045, 0.036, 0.017), loc=(Ro + 0.036, 0, zs_), rot=(rng.u(-35, 35), 0, 0), bevel=0.002, segments=1))]
    # gusset web under the tube, in the vertical plane of the pole
    ce, se = cos(el), sin(el)
    gA = (0.012, -0.078)
    gC = (0.186 * ce + (Ro - 0.003) * se, 0.186 * se - (Ro - 0.003) * ce)
    gD = (0.03 * ce + (Ro - 0.003) * se, 0.03 * se - (Ro - 0.003) * ce)
    gQ = (gD[0] * 0.75 + gA[0] * 0.25 + 0.012, gD[1] * 0.75 + gA[1] * 0.25 - 0.004)
    gp = [gA] + [((1 - t) ** 2 * gA[0] + 2 * t * (1 - t) * gQ[0] + t * t * gC[0], (1 - t) ** 2 * gA[1] + 2 * t * (1 - t) * gQ[1] + t * t * gC[1])
                 for t in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)] + [gC, gD, (0.004, 0.0)]
    body.append(extrude_profile("gusset", gp, 0.011, loc=tuple(K + Lx * 0.0055), rot=(90, 0, p.yaw - 90), bevel=0.002))
    for o in body:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.06, seed=p.seed + 3, lo=0.42, hi=0.75), height_ramp(-0.16, 0.12), 0.75),
              grime=mask_mul(height_ramp(-0.16, 0.05), 0.5))
    for o in scr:
        assign(o, brass)
    # fixings: six coach bolts on washers
    bpos = [(0, 0.118), (0, -0.118 - 0.024), (-0.05, 0.05), (0.05, 0.05), (-0.05, -0.074), (0.05, -0.074)]
    br = rng.fork("bolts")
    wb_ = lathe("washer", [(0.0, 0.0), (0.0145, 0.0), (0.0145, 0.0018), (0.0, 0.0018)], segments=10)
    bb_ = bolt("coach_bolt", 0.0085, 0.007, kind="square")
    wm = [trs((x, -0.0125, z + 0.0), (90, 0, 0)) for (x, z) in bpos]
    bm_ = [trs((x, -0.0143, z + 0.0), (90, br.u(0, 90), 0)) for (x, z) in bpos]
    for o in (instances_to_mesh("washers", wb_, wm, keep_base=False), instances_to_mesh("bolts", bb_, bm_, keep_base=False)):
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.03, seed=p.seed + 5, lo=0.3, hi=0.7), 0.8))

    # ================================================================== pole (turned wood)
    zt = Lt
    prof = [(0.0, 0.02), (r0 - 0.001, 0.02), (r0 - 0.001, zt + 0.002), (r0 + 0.006, zt + 0.004), (r0 + 0.0085, zt + 0.012),
            (r0 + 0.006, zt + 0.02), (r0 + 0.001, zt + 0.026), (r0, zt + 0.042), (r0 + 0.003, zt + 0.048), (r0 + 0.003, zt + 0.056),
            (r0, zt + 0.062)]
    nsec = 8
    for k in range(nsec + 1):
        z = zt + 0.1 + (Lp - 0.11 - zt - 0.1) * k / nsec
        prof.append((rz(z), z))
    prof += [(r1 + 0.003, Lp - 0.104), (r1 + 0.003, Lp - 0.096), (r1, Lp - 0.09), (r1 * 0.8, Lp - 0.072), (r1 * 0.78, Lp - 0.012),
             (r1 * 0.6, Lp), (0.0, Lp)]
    pole = lathe("pole", prof, segments=16)
    uv_grain(pole, axis="Z")
    assign(pole, wood)
    to_pole(pole)
    hand_z = F(0, 0, Lt + 0.3).z
    paint(pole, wear=mask_mul(height_ramp(hand_z, hand_z + 0.25), height_ramp(hand_z, hand_z - 0.25), noise_mask(0.05, seed=p.seed + 8), 0.7),
          grime=mask_mul(facing((0, 0, 1)), noise_mask(0.3, seed=p.seed + 9, lo=0.3, hi=0.8), 0.6),
          special=mask_mul(facing((0, 0, 1)), 0.25 * p.fade))

    # ---- gilt finial on a ferrule
    fs = p.finial_size
    rf = r1 * 0.78 + 0.0015
    z0 = Lp - 0.06
    fp = [(rf, z0), (rf + 0.0025, z0 + 0.003), (rf + 0.0025, z0 + 0.05), (rf + 0.007, z0 + 0.056), (rf + 0.007, z0 + 0.064),
          (rf * 0.62, z0 + 0.072), (rf * 0.52, z0 + 0.085)]
    zn = z0 + 0.085
    if p.finial == "ball":
        Rb = 0.041 * fs
        zc = zn + Rb * 0.9
        fp += [(Rb * cos(radians(a)), zc + Rb * sin(radians(a))) for a in range(-64, 86, 15)]
        fp += [(0.007 * fs, zc + Rb + 0.001), (0.0085 * fs, zc + Rb + 0.008 * fs), (0.004 * fs, zc + Rb + 0.013 * fs), (0.0, zc + Rb + 0.014 * fs)]
    elif p.finial == "spear":
        fp += [(0.013 * fs, zn + 0.006), (0.016 * fs, zn + 0.014), (0.009 * fs, zn + 0.024)]
        Ls, Wm = 0.21 * fs, 0.046 * fs
        for k in range(1, 12):
            t = k / 11
            r = Wm * ((t / 0.3) ** 0.75 if t < 0.3 else ((1 - t) / 0.7) ** 0.85)
            fp.append((max(r, 0.0), zn + 0.024 + Ls * t))
    else:
        fp += [(0.024 * fs, zn + 0.005), (0.033 * fs, zn + 0.018 * fs), (0.035 * fs, zn + 0.034 * fs), (0.031 * fs, zn + 0.039 * fs),
               (0.0295 * fs, zn + 0.05 * fs), (0.027 * fs, zn + 0.07 * fs), (0.02 * fs, zn + 0.088 * fs), (0.009 * fs, zn + 0.101 * fs),
               (0.003 * fs, zn + 0.106 * fs), (0.0036 * fs, zn + 0.114 * fs), (0.0, zn + 0.116 * fs)]
    fin = lathe("finial", fp, segments=14)
    if p.finial == "spear":
        zb = zn + 0.024

        def flatten(co):
            k = lerp(1.0, 0.2, smoothstep(zb - 0.004, zb + 0.03, co.z))
            return Vector((co.x * k, co.y, co.z))
        transform_verts(fin, flatten)
    assign(fin, gilt)
    to_pole(fin)

    # ---- truck: brass strap with cheeks and a sheave under the pole head
    z_s = Lp - 0.145
    r_s = rz(z_s)
    y_s = -(r_s + 0.021)
    R_s = 0.0138                                           # rope centre-line radius round the sheave
    truck = [ring_band("truck_band", radius=r_s + 0.0012, width=0.026, thick=0.0026, loc=(0, 0, z_s), segments=18, bulge=0.001),
             cylinder("sheave", 0.0105, 0.0062, loc=(0, y_s, z_s), rot=(0, 90, 0), verts=12, bevel=0.001),
             cylinder("sheave_pin", 0.0024, 0.017, loc=(0, y_s, z_s), rot=(0, 90, 0), verts=6)]
    for sx in (-1, 1):
        truck.append(box("cheek", (0.0022, 0.031, 0.03), loc=(sx * 0.0052, y_s + 0.0065, z_s), bevel=0.0009, segments=1))
    # ---- horn cleat on the underside of the pole
    z_c = Lt + 0.27
    r_c = rz(z_c)
    yh = -(r_c + 0.02)
    truck += [box("cleat_base", (0.02, 0.005, 0.056), loc=(0, -(r_c + 0.0012), z_c), bevel=0.0015, segments=1),
              cylinder("cleat_stem", 0.0062, 0.018, loc=(0, -(r_c + 0.011), z_c), rot=(90, 0, 0), verts=8),
              sweep("cleat_horns", [(0, yh + 0.003, z_c - 0.052), (0, yh, z_c - 0.028), (0, yh - 0.0015, z_c), (0, yh, z_c + 0.028),
                                    (0, yh + 0.003, z_c + 0.052)], 0.0066, sides=7, up=(0, 1, 0), step=0.013,
                    scale_fn=lambda t: 0.5 + 0.5 * sin(pi * t) ** 0.6)]
    for o in truck:
        assign(o, brass)
        to_pole(o)
        paint(o, special=mask_mul(noise_mask(0.02, seed=p.seed + 11, lo=0.4, hi=0.8), 0.5))

    # ---- halyard: up the inner side, over the sheave, down the outer side (the flag is clipped to that run)
    rr = 0.0031
    yA_top, yB_top = y_s - R_s, y_s + R_s
    y_cw = -(r_c + 0.0085)
    zA0 = z_c + 0.03

    def lineA(z):                                          # (x, y) of the outer run at pole height z
        t = clamp((z - zA0) / (z_s - zA0))
        return (0.004 * (1 - t), lerp(y_cw, yA_top, t))

    hr = rng.fork("halyard")
    path = []
    nB = 10
    sagB = hr.u(0.004, 0.012)
    for k in range(nB + 1):
        t = k / nB
        path.append((-0.005 * (1 - t), lerp(y_cw, yB_top, t) - sagB * 4 * t * (1 - t) * (1 - t) * 1.6, lerp(zA0, z_s, t)))
    for a in (30, 60, 90, 120, 150):
        path.append((0.0, y_s + R_s * cos(radians(a)), z_s + R_s * sin(radians(a))))
    for k in range(nB + 1):
        z = lerp(z_s, zA0, k / nB)
        x, y = lineA(z)
        path.append((x, y, z))
    ropes = [sweep("halyard", path, rr, sides=6, up=(0, 1, 0), smooth_path=False)]
    # figure-of-eight turns on the cleat
    wraps = []
    for k in range(31):
        ta = k / 30 * 4 * pi
        wraps.append((0.0105 * sin(2 * ta), y_cw + 0.001 - 0.0022 * (k / 30), z_c + 0.031 * cos(ta)))
    ropes.append(sweep("cleat_turns", wraps, rr, sides=4, up=(0, 1, 0), smooth_path=False))
    for o in ropes:
        assign(o, rope_m)
        to_pole(o)
    # spare line hung over the lower horn in a coil, with a free tail
    Ph = F(0, yh - 0.004, z_c - 0.034) + Vector((0, 0, 0.004))
    for k in range(3):
        a_, b_ = hr.u(0.03, 0.042), hr.u(0.1, 0.14) + 0.012 * k
        tw = radians(hr.u(-28, 28))
        side = Lx * cos(tw) + Hd * sin(tw)
        off = Ax * (0.005 * (k - 1)) + side * hr.u(-0.006, 0.006)
        loop = []
        for q in range(14):
            ta = 2 * pi * q / 14
            loop.append(tuple(Ph + off + side * (a_ * sin(ta) * (0.62 - 0.38 * cos(ta))) + Vector((0, 0, -b_ * (1 - cos(ta)) * 0.5))))
        c = sweep(f"coil{k}", loop, rr, closed=True, sides=5, up=tuple(Ax), smooth_path=False)
        assign(c, rope_m)
        ropes.append(c)
    tail_l = hr.u(0.2, 0.32)
    tx = hr.u(-0.02, 0.02)
    tail = sweep("tail", [tuple(Ph + Vector((tx * t, 0.004 * sin(t * 5), -tail_l * t))) for t in (0, 0.15, 0.35, 0.55, 0.75, 0.9, 1.0)],
                 rr, sides=5, up=(0, 1, 0), smooth_path=False)
    knot = sphere("tail_knot", 0.0068, loc=tuple(Ph + Vector((tx, 0.004 * sin(5), -tail_l + 0.012))), segments=8, rings=5, scale=(1, 1, 1.5))
    for o in (tail, knot):
        assign(o, rope_m)

    # ---- pair of cord tassels looped round the pole neck above the truck
    if p.tassels:
        cord_m = mat("canvas", p.trim, scale=1.6)
        z_n = Lp - 0.118
        o = to_pole(torus("cord_loop", rz(z_n) + 0.0035, 0.0034, loc=(0, 0, z_n), major_seg=12, minor_seg=5))
        assign(o, cord_m)
        tr_ = rng.fork("tassels")
        bc = radians(36.0) * p.breeze * 0.45
        dn = Vector(((1.0 if p.yaw >= 0 else -1.0) * sin(bc), 0, -cos(bc)))
        tas = lathe("tassel", [(0.0, 0.03), (0.007, 0.028), (0.0115, 0.02), (0.0115, 0.012), (0.006, 0.005), (0.0055, 0.0), (0.009, -0.006),
                               (0.0125, -0.03), (0.0155, -0.075), (0.0145, -0.082), (0.0, -0.08)], segments=9)
        tms = []
        for k, sx in enumerate((-1, 1)):
            P0 = F(sx * 0.006, -(rz(z_n) + 0.006), z_n)
            ln = tr_.u(0.26, 0.36) + 0.13 * k
            sway = Lx * (sx * 0.012) + Hd * tr_.u(-0.01, 0.01)
            pts = [tuple(P0 + dn * (ln * t) + sway * sin(t * 1.6)) for t in (0, 0.12, 0.3, 0.5, 0.7, 0.85, 1.0)]
            c = sweep(f"cord{k}", pts, 0.0034, sides=5, up=(0, 1, 0), smooth_path=False)
            assign(c, cord_m)
            e = _rot_to(-dn)
            tms.append(trs(tuple(P0 + dn * (ln + 0.028) + sway * sin(1.6)), (e[0] + tr_.u(-5, 5), e[1], e[2]), 1.0))
        o = instances_to_mesh("tassels", tas, tms, keep_base=False)
        assign(o, cord_m)

    # ================================================================== optional stay (tie rod)
    if p.stay:
        z_st = Lp * 0.56
        r_st = rz(z_st)
        col = [ring_band("stay_collar", radius=r_st + 0.0012, width=0.024, thick=0.003, loc=(0, 0, z_st), segments=18),
               torus("stay_lug", 0.0095, 0.003, loc=(0, r_st + 0.011, z_st), rot=(0, 90, 0), major_seg=12, minor_seg=5)]
        for o in col:
            assign(o, iron)
            to_pole(o)
        E1 = F(0, r_st + 0.013, z_st)
        hz = sqrt(E1.x ** 2 + E1.y ** 2)
        W = Vector((0.0, -0.034, E1.z + 0.27 * hz))
        dv = (W - E1).normalized()
        sidev = dv.cross(Vector((0, 0, 1))).normalized()
        upv = sidev.cross(dv)
        rod = [tuple(E1 + upv * 0.0), tuple(E1 + dv * 0.02 + upv * 0.004)] + \
              [tuple(E1 + dv * ((W - E1).length * t)) for t in (0.08, 0.3, 0.7, 0.92)] + [tuple(W - dv * 0.02 + upv * 0.004), tuple(W)]
        st = [sweep("stay_rod", rod, 0.0042, sides=6, up=tuple(sidev)),
              lathe("turnbuckle", [(0.0045, -0.05), (0.0085, -0.04), (0.0095, -0.03), (0.0095, 0.03), (0.0085, 0.04), (0.0045, 0.05)],
                    segments=6, loc=tuple(E1.lerp(W, 0.5)), rot=_rot_to(dv)),
              sweep("stay_eye", [tuple(W + sidev * 0.0 + Vector((0, 0.011 * cos(a), 0.011 * sin(a))) + Vector((0, -0.002, 0)))
                                 for a in [2 * pi * q / 10 for q in range(10)]], 0.003, closed=True, sides=5, up=(1, 0, 0)),
              lathe("stay_plate", [(0.0, 0.0), (0.034, 0.0), (0.034, 0.004), (0.028, 0.008), (0.016, 0.009), (0.011, 0.014),
                                   (0.008, 0.024), (0.0, 0.026)], segments=14, loc=(0, -0.001, W.z), rot=(90, 0, 0))]
        for o in st:
            assign(o, iron)
            paint(o, special=mask_mul(noise_mask(0.05, seed=p.seed + 13, lo=0.4, hi=0.75), 0.6))
        sb = bolt("stay_bolt", 0.0052, 0.004, kind="dome")
        o = instances_to_mesh("stay_bolts", sb, [trs((0.024 * cos(radians(a)), -0.006, W.z + 0.024 * sin(radians(a))), (90, 0, 0))
                                                 for a in (45, 135, 225, 315)], keep_base=False)
        assign(o, iron)

    # ================================================================== the flag (cloth lattice)
    hoist = min(p.flag_hoist, Lp - 0.8)
    fly = p.flag_fly
    z_h0 = z_s - 0.055                                     # head of the hoist along the pole
    NS = int(clamp(2 * round(hoist / 0.046), 24, 40))
    NF = int(clamp(2 * round(fly / 0.15), 14, 20))
    ds, df = hoist / NS, fly / NF
    wh = min(0.045, df * 0.8)                              # canvas heading width
    wbd = min(0.02, ds * 0.9)                              # edge binding width

    def Hpt(s):                                            # hoist edge hangs just under the outer halyard run
        z = z_h0 - s
        x, y = lineA(z)
        return F(x, y - 0.011, z)

    beta0 = radians(36.0) * p.breeze
    Wd = Vector((1.0 if p.yaw >= 0 else -1.0, 0, 0))       # the breeze blows along the wall
    hang, dirs = [Vector((0, 0, 0))], []
    for j in range(NF + 1):
        b = beta0 * (0.5 + 1.0 * (j / NF))
        d = Wd * sin(b) - Vector((0, 0, cos(b)))
        dirs.append(d)
        if j < NF:
            hang.append(hang[-1] + d * df)
    nrm = [Ax.cross(d).normalized() for d in dirs]

    fr = rng.fork("folds")
    nfold = max(2.0, hoist / 0.27) * fr.u(0.88, 1.12)
    lam1 = hoist / nfold
    lam2 = lam1 * fr.u(0.5, 0.62)
    ph1, ph2, ph3, ph4, ph5, ph6 = [fr.u(0, 2 * pi) for _ in range(6)]
    sk1, sk2 = fr.u(1.0, 2.2) * fr.pick([-1, 1]), fr.u(2.0, 4.0) * fr.pick([-1, 1])
    curl = fr.pick([-1, 1])
    gmax = 0.46 * p.drape
    s_c = hoist * fr.u(0.3, 0.45)
    ng = max(3, int(round(hoist / 0.28)) + 1)              # grommets along the heading
    sg = [0.02 + (hoist - 0.04) * k / (ng - 1) for k in range(ng)]

    def cloth_pt(s, j):
        f = j * df
        G = 1.0 + gmax * (1 - exp(-f / (0.2 * fly + 0.06))) * (0.8 + 0.2 * f / fly)
        lam = lam1 / G
        amp = lam * sqrt(2 * (G * G - 1)) / (2 * pi) * 0.72     # fold depth that keeps the cloth length
        ff = f / fly
        # uneven fold widths (domain warp), folds of different depth, small creases that appear lower down
        sw = s + lam1 * (0.24 * sin(2 * pi * s / hoist * 0.9 + ph3) + 0.11 * sin(2 * pi * s / hoist * 2.3 + ph4))
        t1 = 2 * pi * sw / lam1 + ph1 + sk1 * ff
        env = 0.62 + 0.38 * sin(2 * pi * s / hoist * 0.7 + ph5 + 0.8 * ff) ** 2
        w = env * (sin(t1) + 0.2 * sin(2 * t1 + 0.7)) + (0.15 + 0.4 * ff) * sin(2 * pi * sw / lam2 + ph2 + sk2 * ff)
        # one broad billow across the whole cloth and the free corners curling out of plane
        bil = 0.045 * p.drape * sin(pi * s / hoist * 1.1 + ph6) * ff ** 1.4
        e0, e1_ = clamp(1 - s / (0.16 * hoist)), clamp(1 - (hoist - s) / (0.16 * hoist))
        bil += curl * 0.05 * p.drape * (e0 * e0 - e1_ * e1_) * ff
        se = s_c + (s - s_c) / G
        sc = 0.011 * sin(pi * clamp((s - sg[0]) / (sg[-1] - sg[0])) * (ng - 1)) ** 2 * exp(-f / 0.12)
        return Hpt(se) + hang[j] + nrm[j] * (amp * w + bil) + dirs[j] * sc

    P = [[cloth_pt(i * ds, j) for j in range(NF + 1)] for i in range(NS + 1)]
    N = [[None] * (NF + 1) for _ in range(NS + 1)]
    for i in range(NS + 1):
        for j in range(NF + 1):
            du = P[min(i + 1, NS)][j] - P[max(i - 1, 0)][j]
            dv_ = P[i][min(j + 1, NF)] - P[i][max(j - 1, 0)]
            N[i][j] = du.cross(dv_).normalized()

    def samp(x, y):
        """Point and normal on the triangulated cloth at lattice coordinates (x along the hoist, y along the fly)."""
        i = int(clamp(int(x), 0, NS - 1))
        j = int(clamp(int(y), 0, NF - 1))
        a, b = x - i, y - j
        out = []
        for Q in (P, N):
            if b <= a:
                out.append(Q[i][j] + (Q[i + 1][j] - Q[i][j]) * a + (Q[i + 1][j + 1] - Q[i + 1][j]) * b)
            else:
                out.append(Q[i][j] + (Q[i + 1][j + 1] - Q[i][j + 1]) * a + (Q[i][j + 1] - Q[i][j]) * b)
        return out[0], out[1].normalized()

    col_a = mat("canvas", p.colour_a)
    col_b = mat("canvas", p.colour_b)
    trim = mat("canvas", p.trim, scale=1.6)
    head_m = mat("canvas", "#d6ccb2")
    flag = []
    TH = 0.0026

    def panel(name, i0, i1, j0, j1, m):
        o = surface(name, lambda u, v: tuple(P[i0 + int(round(u * (i1 - i0)))][j0 + int(round(v * (j1 - j0)))]), i1 - i0, j1 - j0)
        _tri_fixed(o)
        solidify(o, TH, offset=0.0)
        assign(o, m)
        flag.append(o)

    hi, hj = NS // 2, NF // 2
    if p.pattern == "plain":
        panel("field", 0, NS, 0, NF, col_a)
    elif p.pattern == "bicolour":
        panel("field_a", 0, hi, 0, NF, col_a)
        panel("field_b", hi, NS, 0, NF, col_b)
    else:
        panel("quarter_1", 0, hi, 0, hj, col_a)
        panel("quarter_2", hi, NS, 0, hj, col_b)
        panel("quarter_3", 0, hi, hj, NF, col_b)
        panel("quarter_4", hi, NS, hj, NF, col_a)

    def strip(name, fn, n, m, th):
        o = surface(name, lambda u, v: tuple(fn(int(round(u * n)), int(round(v)))), n, 1)
        solidify(o, th, offset=0.0)
        assign(o, m)
        flag.append(o)

    strip("binding_top", lambda k, e: P[0][k].lerp(P[1][k], e * wbd / ds), NF, trim, TH + 0.003)
    strip("binding_foot", lambda k, e: P[NS][k].lerp(P[NS - 1][k], e * wbd / ds), NF, trim, TH + 0.003)
    strip("binding_fly", lambda k, e: P[k][NF].lerp(P[k][NF - 1], e * wbd / df), NS, trim, TH + 0.003)
    strip("heading", lambda k, e: P[k][0].lerp(P[k][1], e * wh / df), NS, head_m, TH + 0.0042)

    # ---- emblem: felt pieces sewn on both faces, cut along the cloth facets so they follow every fold
    polys = _emblem_polys(p.emblem)
    if polys:
        Re = 0.5 * p.emblem_size * min(hoist, fly * 0.75)
        cs, cf = hoist * 0.5, fly * 0.5
        ev, ef = [], []
        off = TH * 0.5 + 0.0016
        for poly in polys:
            ip = [((cs + x * Re) / ds, (cf - y * Re) / df) for (x, y) in poly]
            x0, x1 = min(q[0] for q in ip), max(q[0] for q in ip)
            y0, y1 = min(q[1] for q in ip), max(q[1] for q in ip)
            for i in range(max(0, int(x0)), min(NS, int(x1) + 1)):
                for j in range(max(0, int(y0)), min(NF, int(y1) + 1)):
                    for tri in (((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i, j + 1))):
                        c = _clip(ip, tri)
                        if len(c) < 3:
                            continue
                        ar = _area(c)
                        if abs(ar) < 2e-4:
                            continue
                        if ar < 0:
                            c = c[::-1]
                        cx_, cy_ = (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0, (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0
                        pts = []
                        for (x, y) in c:
                            # nudge towards the triangle centre so the sample lands in this facet
                            pts.append(samp(x + (cx_ - x) * 1e-6, y + (cy_ - y) * 1e-6))
                        b0 = len(ev)
                        ev += [tuple(q + n_ * off) for (q, n_) in pts]
                        ef.append(list(range(b0, b0 + len(pts))))
                        b0 = len(ev)
                        ev += [tuple(q - n_ * off) for (q, n_) in pts]
                        ef.append(list(range(b0 + len(pts) - 1, b0 - 1, -1)))
        if ef:
            em = mesh_object("emblem", ev, ef)
            weld(em, 2e-5)
            smooth(em, 60)
            assign(em, trim)
            flag.append(em)

    # ---- grommets in the heading, each clipped to the halyard with a brass ring
    gm_base = lathe("grommet", [(0.0042, -0.0042), (0.0092, -0.0048), (0.0092, 0.0048), (0.0042, 0.0042)], segments=8)
    gms, rings = [], []
    for k, s in enumerate(sg):
        Gp, Gn = samp(s / ds, (wh * 0.55) / df)
        gms.append(trs(tuple(Gp), _rot_to(Gn)))
        z = z_h0 - (s_c + (s - s_c) / 1.0)
        x, y = lineA(z)
        Rp = F(x, y, z)
        e1 = (Gp - Rp)
        dist = e1.length
        e1 = e1 / max(dist, 1e-6)
        e2 = (Gn - e1 * Gn.dot(e1)).normalized()
        Cc = (Gp + Rp) * 0.5 - e1 * 0.001
        rho = dist * 0.5 + 0.0052
        ring = sweep(f"clip_ring{k}", [tuple(Cc + e1 * (rho * cos(2 * pi * q / 12)) + e2 * (rho * sin(2 * pi * q / 12))) for q in range(12)],
                     0.0017, closed=True, sides=4, up=tuple(e1.cross(e2)), smooth_path=False)
        assign(ring, brass)
        rings.append(ring)
    gro = instances_to_mesh("grommets", gm_base, gms, keep_base=False)
    assign(gro, brass)

    # ---- cloth weathering: sun-fade on the faces that see the sky, soiling along the foot of the drape
    zmin = min(q.z for row in P for q in row)
    zmax = max(q.z for row in P for q in row)
    for o in flag:
        paint(o, special=mask_mul(noise_mask(0.4, seed=p.seed + 17, lo=0.25, hi=0.75), 0.3 + 0.7 * p.fade),
              grime=mask_mul(height_ramp(zmin, zmin + 0.4 * (zmax - zmin)), noise_mask(0.16, seed=p.seed + 19, lo=0.1, hi=0.6), 0.4 + 0.6 * p.fade))
    piv = Hpt(hoist * 0.5)
    part(flag + [gro] + rings, "flag", pivot=tuple(piv), motion="flutter", amp=0.03, speed=0.5)
