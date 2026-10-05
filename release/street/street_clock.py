"""Pedestal street clock. A cast-iron pedestal (plinth, ogee base moulding, sunk bolection-moulded panels
between riveted stiles, corner colonnettes, moulded cornice and ogee cap; cast brass maker's plaque in
front, winding door at the back) carries a stout tapering fluted column with an attic base, acanthus
foot, parcel-gilt rings and a leafy capital. A neck and two forged C-scroll brackets cradle a riveted
two-faced drum case. Each face has a brass bezel with hinge and catch, a puttied domed glass and a
back-lit opal dial (lamp hot-spot in the middle, dust towards the rim) with raised numerals, minute
track and blued spade hands. A scroll crest with urn finial (or a spire, or a jeweller's name board)
crowns the head.
"""
from math import sin, cos, tan, radians, degrees, pi, atan2, hypot

import numpy as np
from mathutils import Vector
from tk import *

META = {"name": "Street clock on a post", "class": "street_furniture", "mount": "ground", "focus": 0.8}
PARAMS = {
    "height": {"default": 3.4, "min": 2.6, "max": 4.8, "desc": "height of the dial centre above the paving (m)"},
    "dial": {"default": 0.8, "min": 0.5, "max": 1.2, "desc": "dial diameter (m)"},
    "time": {"default": 10.17, "min": 0.0, "max": 12.0, "desc": "time shown, decimal hours (10.17 = 10:10)"},
    "paint": {"default": "#3b6455", "desc": "paint colour of the ironwork"},
    "lit": {"default": True, "desc": "back-lit opal glass dials (False = enamel dials, no light)"},
    "glow": {"default": 0.6, "min": 0.1, "max": 3.0, "desc": "emissive strength of the lit dials (centre hot-spot)"},
    "intensity": {"default": 120.0, "min": 0.0, "max": 250.0, "desc": "light spilled onto the square by the lit head (cd)"},
    "dial_tint": {"default": "#ffcd8c", "desc": "colour of the dial glass / enamel"},
    "numerals": {"default": "roman", "choices": ["roman", "batons", "arabic"], "desc": "hour marks"},
    "crest": {"default": "scroll", "choices": ["scroll", "spire", "sign"], "desc": "ornament on top of the head"},
    "base": {"default": "square", "choices": ["square", "octagon"], "desc": "plan of the panelled pedestal"},
    "flutes": {"default": 16, "min": 0, "max": 24, "desc": "flutes on the column (0 = plain)"},
    "legend": {"default": "ANNO 1887", "desc": "text on the pedestal plaque and on the name-board crest"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "condition: 0 = freshly painted, 1 = rusty and neglected"},
}
VARIANTS = {
    "station": {"height": 3.9, "dial": 1.05, "numerals": "batons", "crest": "spire", "base": "octagon", "paint": "#262a33",
                "time": 3.67, "flutes": 20, "dial_tint": "#ffd9a6", "legend": "No 4", "seed": 4, "wear": 0.3},
    "jeweller": {"height": 3.1, "dial": 0.7, "crest": "sign", "paint": "#6a1d1b", "legend": "HORLOGER", "time": 1.83,
                 "flutes": 12, "seed": 11},
    "small_blue": {"height": 2.75, "dial": 0.56, "numerals": "arabic", "paint": "#28456b", "flutes": 0, "time": 7.42,
                   "crest": "spire", "legend": "1902", "seed": 23},
    "neglected": {"height": 3.6, "dial": 0.9, "paint": "#4b6152", "wear": 1.0, "time": 4.9, "glow": 0.42, "intensity": 70.0,
                  "dial_tint": "#f2c688", "base": "octagon", "legend": "ANNO 1871", "seed": 31},
}


# ------------------------------------------------------------------ local mesh helpers
def ngon_ring(n, apo, cr, seg=3):
    """Regular n-gon outline (flats facing +-X / +-Y) with rounded corners, CCW."""
    half = pi / n
    cr = min(cr, apo * 0.45)
    pts = []
    for k in range(n):
        ac = half + k * 2 * half
        cd = (apo - cr) / cos(half)
        cx, cy = cd * cos(ac), cd * sin(ac)
        for j in range(seg + 1):
            a = ac - half + 2 * half * j / seg
            pts.append((cx + cr * cos(a), cy + cr * sin(a)))
    return pts


def loft(name, n, prof, cr=0.008, seg=3, cap=True):
    """Stack rounded n-gon rings: prof = [(apothem, z), ...] bottom to top (cast, eased corners)."""
    verts, faces = [], []
    m = n * (seg + 1)
    for (a, z) in prof:
        verts += [(x, y, z) for (x, y) in ngon_ring(n, a, cr, seg)]
    for i in range(len(prof) - 1):
        for j in range(m):
            a0, a1 = i * m + j, i * m + (j + 1) % m
            faces.append((a0, a1, a1 + m, a0 + m))
    if cap:
        faces.append(tuple(range((len(prof) - 1) * m, len(prof) * m)))
    o = mesh_object(name, verts, faces)
    smooth(o, 35)
    uv_box(o)
    return o


def prisms(name, quads, z0, z1, sides=True):
    """Many convex CCW quads (XY) raised from z0 to z1 as one mesh; sides=False = painted marks."""
    verts, faces = [], []
    for q in quads:
        b = len(verts)
        if sides:
            verts += [(x, y, z0) for x, y in q] + [(x, y, z1) for x, y in q]
            faces.append((b + 4, b + 5, b + 6, b + 7))
            for i in range(4):
                j = (i + 1) % 4
                faces.append((b + i, b + j, b + 4 + j, b + 4 + i))
        else:
            verts += [(x, y, z1) for x, y in q]
            faces.append((b, b + 1, b + 2, b + 3))
    o = mesh_object(name, verts, faces)
    uv_box(o)
    return o


def on_dial(quad, theta_deg):
    """Map a glyph-space quad (u = clockwise tangent, v = radius) onto the dial at clock angle theta."""
    th = radians(theta_deg)
    ox, oy, rx, ry = sin(th), cos(th), cos(th), -sin(th)
    return [(u * rx + v * ox, u * ry + v * oy) for (u, v) in quad]


def roman_quads(s, h, v0):
    """Condensed roman numeral: thick / thin strokes between two serif bars, base at radius v0."""
    tw, nw, sf = 0.15 * h, 0.065 * h, 0.07 * h
    adv = {"I": 0.25 * h, "V": 0.52 * h, "X": 0.52 * h}
    total = sum(adv[c] for c in s)
    quads = []

    def bar(x0, x1, w):
        quads.append([(x0 - w / 2, v0), (x0 + w / 2, v0), (x1 + w / 2, v0 + h), (x1 - w / 2, v0 + h)])

    u = -total / 2
    for c in s:
        a = adv[c]
        cx = u + a / 2
        hw = a / 2 - 0.09 * h
        if c == "I":
            bar(cx, cx, tw)
        elif c == "V":
            bar(cx, cx - hw, tw)
            bar(cx, cx + hw, nw)
        else:
            bar(cx + hw, cx - hw, tw)
            bar(cx - hw, cx + hw, nw)
        u += a
    x0, x1 = -total / 2 + 0.01 * h, total / 2 - 0.01 * h
    quads.append([(x0, v0), (x1, v0), (x1, v0 + sf), (x0, v0 + sf)])
    quads.append([(x0, v0 + h - sf), (x1, v0 + h - sf), (x1, v0 + h), (x0, v0 + h)])
    return quads


def scroll_path(cA, rA, aA, sA, cB, rB, aB, sB, turns=1.12, k=0.55, n=11, m=7):
    """2D C-scroll: spiral A (centre out) -> bezier bar -> spiral B (outside in). a = angle of each
    spiral's outer end seen from its centre, s = +1 if that spiral winds outwards counter-clockwise."""
    def spiral(c, r, a_end, s):
        pts = []
        for i in range(n + 1):
            t = i / n
            ph = radians(a_end) - s * turns * 2 * pi * (1 - t)
            rr = r * (0.17 + 0.83 * t ** 0.85)
            pts.append((c[0] + rr * cos(ph), c[1] + rr * sin(ph)))
        return pts

    A, B = spiral(cA, rA, aA, sA), spiral(cB, rB, aB, sB)
    tA = (-sA * sin(radians(aA)), sA * cos(radians(aA)))
    tB = (sB * sin(radians(aB)), -sB * cos(radians(aB)))
    p0, p3 = A[-1], B[-1]
    d = hypot(p3[0] - p0[0], p3[1] - p0[1]) * k
    p1 = (p0[0] + tA[0] * d, p0[1] + tA[1] * d)
    p2 = (p3[0] - tB[0] * d, p3[1] - tB[1] * d)
    mid = []
    for i in range(1, m):
        t = i / m
        w = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3)
        mid.append((w[0] * p0[0] + w[1] * p1[0] + w[2] * p2[0] + w[3] * p3[0],
                    w[0] * p0[1] + w[1] * p1[1] + w[2] * p2[1] + w[3] * p3[1]))
    return A + mid + B[::-1]


def hand_outline(half, L):
    """Symmetric hand outline from the right-hand half [(x, y) fractions of L], tail to tip, CCW."""
    pts = [(0.0, half[0][1] - 0.03)] + list(half)
    left = [(-x, y) for (x, y) in reversed(half[:-1])]
    return [(x * L, y * L) for (x, y) in pts + left]


def fit_text(name, body, h, maxw, center, depth=0.002, font="serif", min_squeeze=1.0):
    """Lettering lying in XY (extruded +Z) with exact glyph height h, condensed / shrunk to fit maxw,
    its bounding box centred on `center`."""
    t = text(name, body, size=h, depth=depth, loc=(0, 0, 0), rot=(0, 0, 0), font=font)
    lo, hi = world_bbox([t])
    w0, h0 = max(hi.x - lo.x, 1e-6), max(hi.y - lo.y, 1e-6)
    f = h / h0
    sx = 1.0
    if w0 * f > maxw:
        sx = max(min_squeeze, maxw / (w0 * f))
        if w0 * f * sx > maxw * 1.001:
            f *= maxw / (w0 * f * sx)
    t.scale = (f * sx, f, 1.0)
    lo, hi = world_bbox([t])
    c = (lo + hi) / 2
    t.location = Vector(t.location) + Vector(center) - c
    return t


def disc(name, radii, z, seg):
    """Flat disc facing +Z with concentric vertex rings (explicit winding: a lathe cannot know which
    way a flat disc should face)."""
    verts = [(0.0, 0.0, z)]
    for r in radii:
        verts += [(r * cos(2 * pi * i / seg), r * sin(2 * pi * i / seg), z) for i in range(seg)]
    faces = [(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)]
    for k in range(len(radii) - 1):
        b0, b1 = 1 + k * seg, 1 + (k + 1) * seg
        faces += [(b0 + i, b1 + i, b1 + (i + 1) % seg, b0 + (i + 1) % seg) for i in range(seg)]
    o = mesh_object(name, verts, faces)
    uv_planar(o)
    return o


def below(z, top, length):
    """Mask: 1 just under height `top`, fading to 0 `length` further down, 0 above."""
    return np.clip(1.0 - (top - z) / length, 0.0, 1.0) * (z <= top + 0.004)


def near(z, z0, w):
    return np.clip(1.0 - np.abs(z - z0) / w, 0.0, 1.0)


def streaks(seed, width=0.03, lo=0.42, hi=0.72):
    """Vertical run / drip noise: value noise stretched 14x along Z."""
    nm = noise_mask(width, seed=seed, lo=lo, hi=hi)

    def fn(co, n):
        c2 = np.array(co, dtype=float, copy=True)
        c2[:, 2] *= 0.07
        return nm(c2, n)
    return fn


# ------------------------------------------------------------------ build
def build(p, rng):
    W = p.wear
    iron = mat("iron_painted", p.paint, rough=0.5, wear=1.0 + 0.8 * W, special=0.8 + 0.7 * W)
    brass = mat("brass", rough=0.8, wear=1.3)
    black = mat("iron_painted", "#0d0e10")
    putty = mat("iron_painted", "#0b0b0c", rough=1.9)
    glass = mat("glass_clear", "#ffffff", opacity=0.2, rough=1.4)
    moss_m = mat("moss")
    if p.lit:
        dial_m = mat("glass_frosted", p.dial_tint, glow=p.glow, opacity=1.0)
    else:
        dial_m = mat("ceramic_glazed", p.dial_tint)

    zc = p.height                      # dial centre
    Rd = p.dial / 2                    # dial radius
    Rc = Rd + 0.075                    # drum case radius
    D = clamp(0.42 * p.dial, 0.28, 0.46)   # drum depth
    h2 = D / 2
    hs = p.dial / 0.8                  # scale of the head ornaments
    cs = hs ** 0.5                     # scale of column / pedestal sections
    octo = p.base == "octagon"
    nb = 8 if octo else 4
    a = (0.3 if octo else 0.275) * cs  # pedestal apothem (face of the stiles)
    ped_h = clamp(0.29 * zc, 0.8, 1.15)
    iron_objs, brass_objs = [], []

    # ---- pedestal: plinth, ogee base moulding, die with sunk panels, cornice, ogee cap
    rec = 0.014                        # depth of the panel recess behind the stiles
    zb = 0.21
    k0 = ped_h - 0.215
    Pp = a + 0.085
    prof = [(Pp, 0.0), (Pp, 0.088), (Pp - 0.008, 0.097), (a + 0.05, 0.1), (a + 0.05, 0.126), (a + 0.044, 0.14), (a + 0.026, 0.156),
            (a + 0.011, 0.175), (a + 0.005, 0.196), (a + 0.002, zb), (a - rec, zb), (a - rec, lerp(zb, k0, 0.33)), (a - rec, lerp(zb, k0, 0.66)),
            (a - rec, k0), (a + 0.002, k0), (a + 0.009, k0 + 0.006), (a + 0.012, k0 + 0.014), (a + 0.008, k0 + 0.022), (a + 0.004, k0 + 0.026),
            (a + 0.008, k0 + 0.04), (a + 0.024, k0 + 0.06), (a + 0.048, k0 + 0.072), (a + 0.058, k0 + 0.076), (a + 0.058, k0 + 0.106),
            (a + 0.05, k0 + 0.114), (a + 0.022, k0 + 0.121), (a - 0.02, k0 + 0.128), (a * 0.83, k0 + 0.143), (a * 0.765, k0 + 0.163),
            (a * 0.745, k0 + 0.172), (a * 0.745, ped_h - 0.004), (a * 0.735, ped_h)]
    ped = loft("pedestal", nb, prof, cr=0.006 if octo else 0.012)
    densify(ped, 0.34 if octo else 0.28)
    iron_objs.append(ped)
    # gilt bead on the corona of the cornice
    kb = k0 + 0.091
    brass_objs.append(loft("cornice_bead", nb, [(a + 0.056, kb - 0.008), (a + 0.063, kb - 0.005), (a + 0.065, kb), (a + 0.063, kb + 0.005),
                                                (a + 0.056, kb + 0.008)], cr=0.006 if octo else 0.012, cap=False))

    fw = 2 * a * tan(pi / nb)          # width of one face
    cq = 0.006 if octo else 0.036 * cs  # corner quirk left free by the stiles
    fwid, fh = fw - 2 * cq, k0 - zb
    stile, rail = (0.03 if octo else 0.058) * cs, 0.062 * cs
    ow, oh = fwid - 2 * stile, fh - 2 * rail      # sunk panel opening
    pzm = (zb + k0) / 2
    sg = 1 if octo else 2
    frame = shape_extrude("frame", rounded_rect(fwid, fh + 0.004, 0.002, 1), [rounded_rect(ow, oh, 0.012, sg)], depth=rec)
    densify(frame, 0.34 if octo else 0.26)
    bolection = shape_extrude("bolection", rounded_rect(ow + 0.03, oh + 0.03, 0.018, sg), [rounded_rect(ow - 0.02, oh - 0.02, 0.008, sg)],
                              depth=rec + 0.007, bevel=0.005, bevel_res=1)
    fieldp = rounded_box("field", (ow - 0.075, oh - 0.075, 0.008), radius=0.008, loc=(0, 0, 0.004), seg=sg, bevel=0.002)
    rs = clamp(ow / 0.3, 0.4, 1.0) * 2.3
    ros = lathe("rosette", [(0.0, 0.0), (0.024 * rs, 0.0), (0.026 * rs, 0.004 * rs), (0.018 * rs, 0.009 * rs), (0.012 * rs, 0.008 * rs),
                            (0.009 * rs, 0.014 * rs), (0.0, 0.017 * rs)], segments=6)
    rv = lathe("rivet", [(0.0075, 0.0), (0.0058, 0.0035), (0.0, 0.0055)], segments=6)
    face_m, rm, bm_ = [], [], []
    for k in range(nb):
        ang = k * 360.0 / nb
        ar = radians(ang)
        face_m.append(trs(((a - rec) * sin(ar), -(a - rec) * cos(ar), pzm), (90, 0, ang)))
        if k not in (0, nb // 2):
            rm.append(trs(((a - rec + 0.008) * sin(ar), -(a - rec + 0.008) * cos(ar), pzm), (90, 0, ang)))
        for sx in (-1, 1):
            for sz in ((1,) if octo else (-1, 1)):
                u, v = sx * (fwid / 2 - stile * 0.5), sz * (fh / 2 - rail * 0.5)
                bm_.append(trs((a * sin(ar) + u * cos(ar), -a * cos(ar) + u * sin(ar), pzm + v), (90, 0, ang), 1.25))
    iron_objs.append(instances_to_mesh("frames", frame, face_m, keep_base=False))
    iron_objs.append(instances_to_mesh("bolections", bolection, face_m, keep_base=False))
    iron_objs.append(instances_to_mesh("fields", fieldp, [m for i, m in enumerate(face_m) if i != nb // 2], keep_base=False))
    iron_objs.append(instances_to_mesh("stile_bolts", rv, bm_))
    brass_objs.append(instances_to_mesh("rosettes", ros, rm, keep_base=False))
    if not octo:
        # three-quarter colonnettes standing in the corner quirks
        cb = lathe("colonnette", [(0.025, zb), (0.025, zb + 0.022), (0.019, zb + 0.03), (0.0205, zb + 0.05), (0.0205, k0 - 0.05),
                                  (0.019, k0 - 0.03), (0.025, k0 - 0.022), (0.025, k0)], segments=8)
        transform_verts(cb, lambda co: Vector((co.x * cs, co.y * cs, co.z)))
        cc = a - 0.02 * cs
        iron_objs.append(instances_to_mesh("colonnettes", cb, [trs((sx * cc, sy * cc, 0)) for sx in (-1, 1) for sy in (-1, 1)], keep_base=False))

    # front: cast brass maker's plaque - raised rim and lettering polished, ground filled with black paint, four screws
    plw = ow - 0.085
    words = p.legend.strip().split()
    lines = [" ".join(words)] if (len(words) < 2 or len(p.legend.strip()) <= 5) else [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
    plh = min(oh - 0.09, plw * (0.62 if len(lines) == 2 else 0.42))
    plate = box("plaque_plate", (plw - 0.004, plh - 0.004, 0.003), loc=(0, 0, 0.0095))
    assign(plate, black)
    instances_to_mesh("plaque_ground", plate, [face_m[0]], keep_base=False)
    pb = [shape_extrude("plaque_rim", rounded_rect(plw - 0.003, plh - 0.003, 0.006, 2), [rounded_rect(plw - 0.02, plh - 0.02, 0.003, 1)], depth=0.0075,
                        loc=(0, 0, 0.008), bevel=0.0015)]
    if words:
        lh = (plh - 0.05) / len(lines) - (0.012 if len(lines) == 2 else 0.0)
        lh = min(lh, plw * 0.3)
        for i, s in enumerate(lines):
            cy = 0.0 if len(lines) == 1 else (0.5 - i) * (lh + 0.016)
            pb.append(fit_text(f"legend{i}", s, lh, plw - 0.06, (0, cy, 0.01225), depth=0.0025, font="serif"))
    scr = lathe("plq_screw", [(0.0058, 0.0), (0.0048, 0.0024), (0.0, 0.0034)], segments=6)
    pb.append(instances_to_mesh("plaque_screws", scr, [trs((sx * (plw / 2 - 0.019), sz * (plh / 2 - 0.019), 0.011)) for sx in (-1, 1) for sz in (-1, 1)],
                                keep_base=False))
    plq = instances_to_mesh("plaque", join(pb, "plaque_b"), [face_m[0]], keep_base=False)
    assign(plq, brass)
    paint(plq, wear=0.9, grime=0.0, special=mask_mul(noise_mask(0.05, seed=p.seed + 12, lo=0.5, hi=0.9), 0.5 * W), mode="set")

    # back: winding door with knuckle hinges and a brass keyhole escutcheon
    dw, dh = ow - 0.06, oh - 0.06
    door = [rounded_box("door", (dw, dh, 0.011), radius=0.006, loc=(0, 0, 0.0055), seg=2, bevel=0.002)]
    for sy in (-1, 1):
        door.append(cylinder("door_hinge", 0.0085, 0.065, loc=(-dw / 2 + 0.002, sy * dh * 0.3, 0.011), rot=(90, 0, 0), verts=10, bevel=0.0015))
    door = join(door, "door")
    iron_objs.append(instances_to_mesh("door_i", door, [face_m[nb // 2]], keep_base=False))
    esc = lathe("escutcheon", [(0.02, 0.0), (0.02, 0.004), (0.013, 0.009), (0.006, 0.009), (0.005, 0.004), (0.0, 0.004)], segments=14)
    brass_objs.append(instances_to_mesh("escutcheon_i", esc, [trs((-(dw / 2 - 0.04), a - rec + 0.011, pzm), (90, 0, 180))], keep_base=False))

    # holding-down bolts on the plinth ledge
    hb = join([cylinder("hd_washer", 0.017, 0.004, loc=(0, 0, 0.002), verts=10), cylinder("hd_nut", 0.012, 0.012, loc=(0, 0, 0.009), verts=6)], "hd_bolt")
    rb_ = (a + 0.066) / cos(pi / nb) - 0.016
    iron_objs.append(instances_to_mesh("hd_bolts", hb, [trs((rb_ * cos(pi / nb + k * 2 * pi / nb), rb_ * sin(pi / nb + k * 2 * pi / nb), 0.098),
                                                             (0, 0, rng.u(0, 60))) for k in range(nb)], keep_base=False))

    # moss tufts in the joint between plinth and paving (more on the damp back, more when neglected)
    mr = rng.fork("moss")
    blob = sphere("moss_tuft", 0.5, segments=7, rings=4)
    mm = []
    for i in range(int(round(3 + 11 * W))):
        k = mr.pick([nb // 2] * 3 + list(range(nb)))
        ar = radians(k * 360.0 / nb)
        u = mr.u(-0.5, 0.5) * 2 * Pp * tan(pi / nb) * 0.92
        r = mr.u(0.035, 0.075)
        zs = mr.u(0.3, 0.5)
        mm.append(trs(((Pp + 0.004) * sin(ar) + u * cos(ar), -(Pp + 0.004) * cos(ar) + u * sin(ar), r * zs * 0.5), (0, 0, k * 360.0 / nb + mr.u(-20, 20)),
                      (r * mr.u(1.2, 2.2), r * 0.8, r * zs)))
    mo = instances_to_mesh("moss", blob, mm, keep_base=False)
    assign(mo, moss_m)

    # ---- column: attic base, stout fluted tapering shaft with a turned band, leafy capital, neck
    z0 = ped_h
    z1 = zc - Rc - 0.42 * cs
    r0, r1 = 0.135 * cs, 0.085 * cs
    L = z1 - z0
    bs = 1.25 * cs

    def rad(z):
        t = clamp((z - z0) / L)
        return r0 + (r1 - r0) * t + 0.006 * sin(pi * t)

    nf = int(p.flutes)
    fz0, fz1 = z0 + 0.36 * cs, z1 - 0.06
    bz = fz0 + (fz1 - fz0) * 0.3
    base_p = [(0.052, 0.0), (0.052, 0.028), (0.042, 0.033), (0.048, 0.044), (0.048, 0.058), (0.038, 0.069), (0.022, 0.076), (0.017, 0.09),
              (0.023, 0.1), (0.028, 0.108), (0.024, 0.12), (0.008, 0.128)]
    zs0 = z0 + 0.14 * bs
    iron_objs.append(lathe("shaft_base", [(r0 + dr * bs, z0 + dz * bs) for (dr, dz) in base_p] + [(rad(zs0), zs0)], segments=24))
    sp = []
    rows = [zs0, fz0 - 0.02, fz0, fz0 + 0.02, fz0 + 0.045, bz - 0.085, bz - 0.06, bz - 0.04, bz + 0.04, bz + 0.06, bz + 0.085,
            fz1 - 0.045, fz1 - 0.02, fz1, z1]
    rows += list(np.arange(bz + 0.2, fz1 - 0.1, 0.25)) + list(np.arange(fz0 + 0.14, bz - 0.13, 0.25))
    last = -1.0
    for z in sorted(rows):
        if z - last > 0.008 and zs0 - 0.001 < z <= z1:
            sp.append((rad(z), float(z)))
            last = z
    shaft = lathe("shaft", sp, segments=(nf * 3 if nf >= 12 else max(32, nf * 4)), smooth_angle=60)

    def flute(co):
        z = co.z
        if nf <= 0 or z < fz0 or z > fz1:
            return co
        e = min(smoothstep(fz0, fz0 + 0.045, z), 1.0 - smoothstep(fz1 - 0.045, fz1, z), smoothstep(0.04, 0.085, abs(z - bz)))
        c = 0.5 + 0.5 * cos(atan2(co.y, co.x) * nf)
        r = hypot(co.x, co.y)
        k = (r - 0.0085 * cs * e * (1.0 - c ** 3)) / r if r > 1e-6 else 1.0
        return Vector((co.x * k, co.y * k, co.z))

    transform_verts(shaft, flute)
    iron_objs.append(shaft)
    rr = rad(bz)
    brass_objs.append(lathe("band", [(rr - 0.002, bz - 0.04), (rr + 0.007, bz - 0.033), (rr + 0.007, bz - 0.024), (rr + 0.017, bz - 0.012), (rr + 0.02, bz),
                                     (rr + 0.017, bz + 0.012), (rr + 0.007, bz + 0.024), (rr + 0.007, bz + 0.033), (rr - 0.002, bz + 0.04)], segments=24))
    # parcel-gilt upper torus of the attic base and astragal under the capital: they catch the lantern light
    brass_objs.append(lathe("base_torus", [(r0 + dr * bs, z0 + dz * bs) for (dr, dz) in ((0.02, 0.097), (0.027, 0.1), (0.0315, 0.108), (0.027, 0.117),
                                                                                         (0.02, 0.121))], segments=24))
    za = fz1 + 0.03
    ra_ = rad(za)
    brass_objs.append(lathe("astragal", [(ra_ - 0.002, za - 0.013), (ra_ + 0.007, za - 0.009), (ra_ + 0.011, za), (ra_ + 0.007, za + 0.009),
                                         (ra_ - 0.002, za + 0.013)], segments=24))

    def leaf_fn(u, v):
        return ((u - 0.5) * 0.088 * (1 - 0.86 * v ** 2.2), -0.006 * sin(v * pi * 0.9) - 0.013 * v ** 4 - 0.013 * cos((u - 0.5) * pi) * (1 - 0.6 * v),
                v * 0.15)

    leaf = surface("leaf", leaf_fn, 3, 3, thickness=0.004)
    realize(leaf)
    rl = rad(zs0) + 0.003
    lm = [trs((rl * sin(radians(k * 45)), -rl * cos(radians(k * 45)), zs0 - 0.012), (0, 0, k * 45), (cs * 1.25, cs * 1.2, cs * (1.3 + 0.12 * (k % 2))))
          for k in range(8)]
    rl = r1 + 0.006
    lm += [trs((rl * sin(radians(k * 45 + 22.5)), -rl * cos(radians(k * 45 + 22.5)), z1 + 0.034 * cs), (0, 0, k * 45 + 22.5), cs * 0.7) for k in range(8)]
    iron_objs.append(instances_to_mesh("leaves", leaf, lm, keep_base=False))

    zn = zc - Rc                       # underside of the drum
    c1 = z1
    capital = lathe("capital", [(r1 + 0.003, c1 - 0.002), (r1 + 0.016, c1 + 0.008 * cs), (r1 + 0.019, c1 + 0.018 * cs), (r1 + 0.014, c1 + 0.028 * cs),
                                (r1 + 0.004, c1 + 0.034 * cs), (r1 + 0.006, c1 + 0.07 * cs), (r1 + 0.022, c1 + 0.115 * cs), (r1 + 0.046, c1 + 0.142 * cs),
                                (r1 + 0.056, c1 + 0.15 * cs), (r1 + 0.056, c1 + 0.172 * cs), (r1 + 0.046, c1 + 0.18 * cs), (0.078 * cs, c1 + 0.19 * cs),
                                (0.066 * cs, c1 + 0.215 * cs), (0.063 * cs, zn - 0.1 * cs), (0.07 * cs, zn - 0.085 * cs), (0.08 * cs, zn - 0.075 * cs),
                                (0.07 * cs, zn - 0.062 * cs), (0.082 * cs, zn - 0.04 * cs), (0.114 * cs, zn - 0.012), (0.124 * cs, zn + 0.03)], segments=20)
    iron_objs.append(capital)
    # saddle casting hugging the underside of the drum
    sad = [((Rc - 0.008) * cos(radians(t)), 0.0, zc + (Rc - 0.008) * sin(radians(t))) for t in np.linspace(270 - 26 / hs ** 0.5, 270 + 26 / hs ** 0.5, 11)]
    iron_objs.append(sweep("saddle", sad, rounded_rect(0.034, D * 0.66, 0.006, 2), up=(0, 1, 0), smooth_path=False))

    # ---- forged C-scroll brackets between the capital and the flanks of the drum
    bar = rounded_rect(0.027 * cs, 0.046 * cs, 0.005, 1)
    rA, rB = 0.056 * cs, 0.048 * cs
    beta = radians(-57.0)
    cB = ((Rc - 0.014 + rB) * cos(beta), zc + (Rc - 0.014 + rB) * sin(beta))
    cA = (0.066 * cs + rA, c1 + 0.18 * cs + rA)
    br2d = scroll_path(cA, rA, -90.0, 1, cB, rB, 0.0, -1)
    eye = cylinder("eye", 0.012 * cs, 0.06 * cs, verts=8)
    eyes = []
    for sx in (-1, 1):
        pts = [(sx * x, 0.0, z) for (x, z) in br2d]
        iron_objs.append(sweep(f"bracket{sx}", pts, bar, up=(0, 1, 0), scale_fn=lambda t: 0.5 + 0.5 * min(1.0, 2.2 * sin(pi * t))))
        eyes += [trs((sx * cA[0], 0, cA[1]), (90, 0, 0)), trs((sx * cB[0], 0, cB[1]), (90, 0, 0))]
        # collar strap tying the lower curl to the neck
        iron_objs.append(box(f"strap{sx}", (0.03 * cs, 0.062 * cs, 0.014), loc=(sx * 0.07 * cs, 0, cA[1]), bevel=0.002))

    # ---- drum case (axis along Y) with moulded rims, mid band and rivets
    half = [(Rd + 0.03, -h2), (Rc - 0.014, -h2), (Rc - 0.004, -h2 + 0.004), (Rc, -h2 + 0.014), (Rc, -h2 + 0.04), (Rc - 0.006, -h2 + 0.046),
            (Rc - 0.008, -h2 + 0.055), (Rc - 0.02, -h2 + 0.064), (Rc - 0.022, -h2 + 0.076), (Rc - 0.022, -0.046), (Rc - 0.013, -0.036), (Rc - 0.011, -0.03)]
    drum = lathe("drum", half + [(r, -z) for (r, z) in reversed(half)], segments=40, loc=(0, 0, zc), rot=(90, 0, 0), smooth_angle=40)
    iron_objs.append(drum)
    rvm = []
    nrv = int(round(18 * hs / 2)) * 2
    for sy in (-1, 1):
        for k in range(nrv):
            t = radians((k + 0.5) * 360.0 / nrv)
            rvm.append(trs((Rc * sin(t), sy * (h2 - 0.027), zc + Rc * cos(t)), (0, degrees(t), 0)))
    iron_objs.append(instances_to_mesh("rivets", rv, rvm, keep_base=False))
    # turned bosses on the flanks
    boss_p = [(0.05, 0.0), (0.05, 0.012), (0.036, 0.02), (0.03, 0.03), (0.042, 0.044), (0.046, 0.058), (0.034, 0.078), (0.014, 0.09), (0.0, 0.094)]
    for sx in (-1, 1):
        iron_objs.append(lathe(f"boss{sx}", [(r * hs, z * hs) for (r, z) in boss_p], segments=16, loc=(sx * (Rc - 0.03), 0, zc), rot=(0, sx * 90, 0)))

    # ---- one clock face, modelled lying in XY facing +Z (z = 0 is the rim of the drum)
    # what is printed on the dial and the hands: matt non-metal near-black paint at every level
    ink = mat("wood_painted", "#0c0c0e", rough=1.8, wear=0.0, special=0.0)
    zd = -0.004                        # dial surface
    gz = 0.014                         # seat of the glass in the bezel rebate
    Rg = Rd + 0.003
    gd = max(0.05 * hs, 0.034)         # rise of the domed glass
    face = []
    dial = disc("dial", [f * Rd for f in (0.2, 0.4, 0.6, 0.78, 0.9, 0.96, 1.0 + 0.005 / Rd)], zd, 36)
    assign(dial, dial_m)
    face.append(dial)
    marks, painted = [], []
    f_nums = []
    ra, rb = 0.86 * Rd, 0.94 * Rd
    for i in range(60):
        if i % 5 == 0:
            marks.append(on_dial([(-0.017 * Rd, ra), (0.017 * Rd, ra), (0.021 * Rd, rb), (-0.021 * Rd, rb)], i * 6))
        else:
            painted.append(on_dial([(-0.006 * Rd, ra), (0.006 * Rd, ra), (0.006 * Rd, rb), (-0.006 * Rd, rb)], i * 6))
    lw = max(0.0035, 0.011 * Rd)
    arabic = p.numerals == "arabic"
    for (rr_, w_) in ((ra - lw / 2, lw), (rb + lw / 2, lw), ((0.535 if arabic else 0.555) * Rd, lw * 0.8)):
        ns = 36
        for i in range(ns):
            t0, t1 = 2 * pi * i / ns, 2 * pi * (i + 1) / ns
            ri, ro = rr_ - w_ / 2, rr_ + w_ / 2
            painted.append([(ri * cos(t0), ri * sin(t0)), (ro * cos(t0), ro * sin(t0)), (ro * cos(t1), ro * sin(t1)), (ri * cos(t1), ri * sin(t1))])
    hN, v0 = 0.225 * Rd, 0.6 * Rd
    if p.numerals == "roman":
        names = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
        for k, s in enumerate(names):
            marks += [on_dial(q, k * 30) for q in roman_quads(s, hN, v0)]
    elif p.numerals == "batons":
        for k in range(12):
            w0, w1 = (0.05, 0.085) if k % 3 == 0 else (0.03, 0.05)
            marks.append(on_dial([(-w0 * Rd, v0), (w0 * Rd, v0), (w1 * Rd, v0 + hN), (-w1 * Rd, v0 + hN)], k * 30))
    else:
        # upright bold arabic numerals, 14 % of the dial diameter tall, centred on the roman radius
        hA, rt = 0.28 * Rd, 0.7 * Rd
        for k in range(12):
            th = radians(k * 30)
            t = fit_text(f"num{k}", str(12 if k == 0 else k), hA, 0.3 * Rd, (rt * sin(th), rt * cos(th), zd + 0.0042), depth=0.0, font="serif",
                         min_squeeze=0.7)
            assign(t, ink)
            face.append(t)
            f_nums.append(t)
    # CLEARANCE RULE: everything printed on the lit dial stands 4 mm or more above it (it was 1.8 / 2 / 3 mm)
    mk = prisms("marks", marks, zd, zd + 0.0045, sides=p.numerals == "batons")
    pt = prisms("painted", painted, zd, zd + 0.004, sides=False)
    assign(mk, ink)
    assign(pt, ink)
    face += [mk, pt]

    # hands: blued-steel spade hour hand and slender minute hand with counterpoise tails
    hour_half = [(0.04, -0.2), (0.062, -0.13), (0.036, -0.06), (0.07, 0.0), (0.036, 0.07), (0.032, 0.5), (0.105, 0.63), (0.082, 0.76), (0.0, 1.0)]
    min_half = [(0.024, -0.24), (0.042, -0.18), (0.022, -0.08), (0.046, 0.0), (0.022, 0.07), (0.02, 0.58), (0.034, 0.64), (0.012, 0.82), (0.0, 1.0)]
    h_ang = (p.time % 12.0) / 12.0 * 360.0
    m_ang = (p.time % 1.0) * 360.0
    hh = shape_extrude("hand_hour", hand_outline(hour_half, 0.56 * Rd), depth=0.003, loc=(0, 0, zd + 0.008), rot=(0, 0, -h_ang), bevel=0.0008)
    mh = shape_extrude("hand_min", hand_outline(min_half, 0.85 * Rd), depth=0.003, loc=(0, 0, zd + 0.014), rot=(0, 0, -m_ang), bevel=0.0008)
    for o in (hh, mh):
        assign(o, ink)
    hub = lathe("hub", [(0.05 * Rd, zd), (0.05 * Rd, zd + 0.019), (0.034 * Rd, zd + 0.024), (0.016 * Rd, zd + 0.024), (0.014 * Rd, zd + 0.029),
                        (0.0, zd + 0.03)], segments=12)
    assign(hub, brass)
    face += [hh, mh, hub]

    # The dial is the identity of the clock: what stands for its parts at lod 1 / lod 2 is modelled here from the
    # same numbers (tk.lod_standin: the full model is not touched, the reducer never simplifies these further).
    #   lod 1: hands as flat fronts, the three hairline rings on 24 segments (minute ticks go); the opal dial, the
    #          hour blocks and the numerals as modelled (they are flat already)
    #   lod 2: a 16-sided opal disc (one inner ring of vertices for the lamp hot-spot), 12 bold hour marks instead
    #          of blocks + numerals, bold hands of the same lengths and angles, 57 mm wide = 1.5 px at 44 m (never
    #          over 0.2 x the dial radius). Layers keep their heights: marks 4.5 mm, hands 11 / 17 mm over the dial.
    f_mats = [trs((0, -h2, zc), (90, 0, 0)), trs((0, h2, zc), (90, 0, 180))]

    def f_inst(nm, parts):
        return instances_to_mesh(nm, join(parts, nm + "_one"), f_mats, uv_jitter=False, keep_base=False)

    def f_poly(nm, outline, ang, z, m):         # a flat outline facing +Z, turned clockwise as rot=(0, 0, -ang) does
        c_, s_ = cos(radians(ang)), sin(radians(ang))
        o = mesh_object(nm, [(x * c_ + y * s_, -x * s_ + y * c_, z) for x, y in outline], [tuple(range(len(outline)))])
        uv_box(o)
        assign(o, m)
        return o

    def f_hands_mid():
        return f_inst("faces_hands_mid", [f_poly("hand_hour_mid", hand_outline(hour_half, 0.56 * Rd), h_ang, zd + 0.011, ink),
                                          f_poly("hand_min_mid", hand_outline(min_half, 0.85 * Rd), m_ang, zd + 0.017, ink)])

    def f_painted_mid():
        qs = []
        for (rr_, w_) in ((ra - lw / 2, lw), (rb + lw / 2, lw), ((0.535 if arabic else 0.555) * Rd, lw * 0.8)):
            for i in range(24):
                t0, t1 = 2 * pi * i / 24, 2 * pi * (i + 1) / 24
                ri, ro = rr_ - w_ / 2, rr_ + w_ / 2
                qs.append([(ri * cos(t0), ri * sin(t0)), (ro * cos(t0), ro * sin(t0)), (ro * cos(t1), ro * sin(t1)), (ri * cos(t1), ri * sin(t1))])
        o = prisms("painted_mid", qs, zd, zd + 0.004, sides=False)
        assign(o, ink)
        return f_inst("faces_painted_mid", [o])

    def f_far():
        d = disc("dial_far", [0.45 * Rd, (Rd + 0.005) / cos(pi / 16)], zd, 16)
        assign(d, dial_m)
        qs = []
        for k in range(12):
            w0 = (0.075 if k % 3 == 0 else 0.05) * Rd
            qs.append(on_dial([(-w0, 0.6 * Rd), (w0, 0.6 * Rd), (w0, rb), (-w0, rb)], k * 30))
        m2 = prisms("marks_far", qs, zd, zd + 0.0045, sides=False)
        assign(m2, ink)
        wb = min(max(0.057, 0.09 * Rd), 0.2 * Rd)
        lh2, lm2 = 0.56 * Rd, 0.85 * Rd
        hb = [(-0.5 * wb, -0.2 * lh2), (0.5 * wb, -0.2 * lh2), (0.6 * wb, 0.6 * lh2), (0.0, lh2), (-0.6 * wb, 0.6 * lh2)]
        mb = [(-0.5 * wb, -0.24 * lm2), (0.5 * wb, -0.24 * lm2), (0.5 * wb, 0.82 * lm2), (0.0, lm2), (-0.5 * wb, 0.82 * lm2)]
        return f_inst("faces_far", [d, m2, f_poly("hand_hour_far", hb, h_ang, zd + 0.011, ink),
                                    f_poly("hand_min_far", mb, m_ang, zd + 0.017, ink)])

    lod_standin(dial, far=f_far)
    lod_standin([mk] + f_nums, far=())
    lod_standin(pt, mid=f_painted_mid, far=())
    lod_standin([hh, mh], mid=f_hands_mid, far=())

    # brass bezel: moulded ring with an inner lip; the domed glass sits in its rebate on a dark putty fillet
    bez = lathe("bezel", [(Rd + 0.052, -0.002), (Rd + 0.052, 0.006), (Rd + 0.045, 0.013), (Rd + 0.034, 0.016), (Rd + 0.028, 0.022), (Rd + 0.018, 0.0275),
                          (Rd + 0.006, 0.0275), (Rd - 0.003, 0.0235)], segments=32)
    assign(bez, brass)

    def dome(r):
        return gz + gd * (1.0 - (r / Rg) ** 2)

    pr = Rd - 0.012
    seat = lathe("putty", [(Rd - 0.003, 0.0235), (Rd - 0.0075, 0.0205), (pr, dome(pr) + 0.0006), (Rd + 0.0035, gz - 0.002),
                           (Rd + 0.0035, zd - 0.001)], segments=32)
    assign(seat, putty)
    gl = lathe("glass", [(f * Rg, dome(f * Rg)) for f in (1.0, 0.9, 0.74, 0.52, 0.27)] + [(0.0, gz + gd)], segments=24)
    assign(gl, glass)
    hinge = cylinder("bez_hinge", 0.0085, 0.085, loc=(-(Rd + 0.058), 0, 0.005), rot=(90, 0, 0), verts=10, bevel=0.0015)
    leafp = box("bez_hinge_leaf", (0.03, 0.07, 0.004), loc=(-(Rd + 0.046), 0, 0.004), bevel=0.001)
    catch = box("bez_catch", (0.03, 0.022, 0.012), loc=(Rd + 0.054, 0, 0.006), bevel=0.002)
    knob = sphere("bez_knob", 0.009, loc=(Rd + 0.068, 0, 0.012), segments=10, rings=6)
    for o in (hinge, leafp, catch, knob):
        assign(o, brass)
    bscr = lathe("bez_screw", [(0.0045, 0.0), (0.0034, 0.002), (0.0, 0.003)], segments=6)
    assign(bscr, brass)
    scm = [trs(((Rd + 0.039) * cos(radians(t)), (Rd + 0.039) * sin(radians(t)), 0.0145)) for t in range(30, 360, 60)]
    face += [bez, seat, gl, hinge, leafp, catch, knob, instances_to_mesh("bez_screws", bscr, scm, keep_base=False)]
    face = join(face, "face")
    faces = instances_to_mesh("faces", face, [trs((0, -h2, zc), (90, 0, 0)), trs((0, h2, zc), (90, 0, 180))], uv_jitter=False, keep_base=False)

    # ageing of the faces, painted in place: lamp hot-spot / dusty rim on the opal, tarnish and runs on the bezels
    dn = noise_mask(0.16, seed=p.seed + 21, lo=0.2, hi=0.8)
    dn2 = noise_mask(0.035, seed=p.seed + 22, lo=0.3, hi=0.8)

    def dial_dark(co, n):
        r = np.hypot(co[:, 0], co[:, 2] - zc) / Rd
        d = np.abs(co[:, 1]) - h2
        on = (np.abs(d - zd) < 0.0008) * 1.0
        g = 0.62 * np.clip((r - 0.12) / 0.83, 0.0, 1.0) ** 1.5 + 0.16 * (dn(co, n) - 0.5) * np.clip(r * 2.0, 0.0, 1.0)
        ring = np.clip((r - (1.0 - 0.034 / Rd)) / (0.02 / Rd), 0.0, 1.0)
        low = np.clip(-(co[:, 2] - zc) / Rd - 0.45, 0.0, 1.0) * 0.25
        return np.clip(g + ring * (0.16 + 0.14 * dn2(co, n)) + low, 0.0, 1.0) * on

    def face_grime(co, n):
        d = np.abs(co[:, 1]) - h2
        up = (co[:, 2] - zc) / Rd
        off = (np.abs(d - zd) >= 0.0008) * 1.0
        return np.clip(0.3 + 0.3 * W + 0.35 * np.clip(-up, 0.0, 1.0) * dn(co, n), 0.0, 1.0) * off

    def face_verd(co, n):
        up = (co[:, 2] - zc) / Rd
        return np.clip((0.25 + 0.75 * W) * dn2(co, n) * (0.4 + 0.6 * np.clip(-up + 0.3, 0.0, 1.0)), 0.0, 1.0)

    paint(faces, grime=lambda co, n: np.maximum(dial_dark(co, n), face_grime(co, n)), special=face_verd)

    # ---- crest
    zt = zc + Rc
    top_sad = [((Rc - 0.012) * cos(radians(t)), 0.0, zc + (Rc - 0.012) * sin(radians(t))) for t in np.linspace(90 - 22, 90 + 22, 9)]
    iron_objs.append(sweep("crest_saddle", top_sad, rounded_rect(0.03, D * 0.5, 0.006, 2), up=(0, 1, 0), smooth_path=False))
    fin_p = [(0.07, 0.0), (0.076, 0.012), (0.06, 0.026), (0.036, 0.036), (0.03, 0.05), (0.05, 0.07), (0.075, 0.1), (0.08, 0.125), (0.066, 0.15),
             (0.03, 0.165), (0.022, 0.18)]
    spike_p = [(0.022, 0.176), (0.036, 0.188), (0.036, 0.196), (0.022, 0.206), (0.017, 0.26), (0.007, 0.34), (0.0, 0.355)]
    if p.crest in ("scroll", "spire"):
        fs = hs * (1.2 if p.crest == "scroll" else 1.5)
        fz = zt - 0.004
        iron_objs.append(lathe("finial", [(r * fs, fz + z * fs) for (r, z) in fin_p], segments=16))
        brass_objs.append(lathe("finial_spike", [(r * fs, fz + z * fs) for (r, z) in spike_p], segments=14))
    if p.crest == "scroll":
        rA2, rB2 = 0.055 * hs, 0.045 * hs
        cA2 = (0.055 * hs + rA2, zt + 0.002 + rA2)
        g = radians(53.0)
        cB2 = ((Rc - 0.014 + rB2) * cos(g), zc + (Rc - 0.014 + rB2) * sin(g))
        cr2d = scroll_path(cA2, rA2, 90.0, -1, cB2, rB2, 0.0, 1)
        bar2 = rounded_rect(0.023 * hs, 0.04 * hs, 0.005, 1)
        for sx in (-1, 1):
            pts = [(sx * x, 0.0, z) for (x, z) in cr2d]
            iron_objs.append(sweep(f"crest_scroll{sx}", pts, bar2, up=(0, 1, 0), scale_fn=lambda t: 0.5 + 0.5 * min(1.0, 2.2 * sin(pi * t))))
            eyes += [trs((sx * cA2[0], 0, cA2[1]), (90, 0, 0), hs / cs * 0.9), trs((sx * cB2[0], 0, cB2[1]), (90, 0, 0), hs / cs * 0.9)]
    elif p.crest == "spire":
        ball_p = [(0.03, 0.0), (0.034, 0.008), (0.02, 0.02), (0.016, 0.03), (0.036, 0.045), (0.042, 0.065), (0.034, 0.085), (0.012, 0.098), (0.0, 0.1)]
        for sx in (-1, 1):
            g = radians(90 - sx * 42)
            iron_objs.append(lathe(f"acroter{sx}", [(r * hs, z * hs) for (r, z) in ball_p], segments=14,
                                   loc=((Rc - 0.012) * cos(g), 0, zc + (Rc - 0.012) * sin(g)), rot=(0, sx * 42, 0)))
    else:
        # jeweller's name board: arched cast panel on two turned standards, lettering both sides
        bw_, bh_ = p.dial * 1.02, 0.25 * hs
        bz0 = zt + 0.085 * hs
        outline = arch_pts(bw_, bh_, spring=bh_ * 0.62, kind="segmental", n=12)
        big = [(x * (1 + 0.05 / bw_ * 2), y * (1 + 0.05 / bh_) - 0.025) for (x, y) in outline]
        board = shape_extrude("board", outline, depth=0.018, loc=(0, 0.009, bz0), rot=(90, 0, 0))
        assign(board, dial_m)
        iron_objs.append(shape_extrude("board_frame", big, [[(x * 0.985, y * 0.97 + 0.003) for (x, y) in outline]], depth=0.036, loc=(0, 0.018, bz0),
                                       rot=(90, 0, 0), bevel=0.003))
        st_p = [(0.016, -0.2), (0.016, 0.0), (0.026, 0.008), (0.026, 0.016), (0.015, 0.026), (0.013, 0.05), (0.022, 0.06), (0.014, 0.07), (0.014, 0.1)]
        for sx in (-1, 1):
            xs_ = sx * bw_ * 0.3
            ztop_d = zc + (Rc * Rc - xs_ * xs_) ** 0.5 - 0.022
            iron_objs.append(lathe(f"standard{sx}", [(r * hs, min(ztop_d + z * hs, bz0 - 0.02 + (0.03 if z > 0.09 else 0))) for (r, z) in st_p],
                                   segments=12, loc=(xs_, 0, 0)))
            brass_objs.append(sphere(f"board_ball{sx}", 0.028 * hs, loc=(sx * (bw_ / 2 + 0.012), 0, bz0 + bh_ * 0.62 + 0.03 * hs), segments=14, rings=8))
        brass_objs.append(lathe("board_finial", [(r * hs * 0.7, bz0 + bh_ + 0.022 + (z - 0.176) * hs * 0.7) for (r, z) in spike_p], segments=14))
        if p.legend.strip():
            for sy, rz in ((-1, 0), (1, 180)):
                t = text(f"board_text{sy}", p.legend, size=bh_ * 0.5, depth=0.0, loc=(0, sy * 0.014, bz0 + bh_ * 0.17), rot=(90, 0, rz), font="serif")
                bb = world_bbox([t])
                tw_ = bb[1].x - bb[0].x
                if tw_ > bw_ * 0.84:
                    f = bw_ * 0.84 / tw_
                    t.scale = (f, f, f)
                    t.location.z = bz0 + bh_ * 0.42 - bh_ * 0.25 * f
                assign(t, black)
    iron_objs.append(instances_to_mesh("eyes", eye, eyes, keep_base=False))

    # ---- materials and art-directed ageing
    for o in iron_objs:
        assign(o, iron)
    for o in brass_objs:
        assign(o, brass)
    sd = p.seed
    blot = noise_mask(0.07, seed=sd + 2, lo=0.3, hi=0.7)
    patch = noise_mask(0.28, seed=sd + 7, lo=0.66 - 0.3 * W, hi=0.88)
    fine = noise_mask(0.025, seed=sd + 4, lo=0.5, hi=0.72)
    run_a, run_b = streaks(sd + 5), streaks(sd + 6, width=0.045, lo=0.35, hi=0.7)
    under = facing((0, 0, -1), 2.0)
    upf = facing((0, 0, 1), 2.0)
    zbt = zs0 - 0.02                     # top of the attic base

    def axial(co):
        return np.hypot(co[:, 0], co[:, 2] - zc)

    def rust(co, n):
        z = co[:, 2]
        # rising damp at the plinth, in the base moulding and round the foot of the column
        foot = np.maximum(below(z, 0.3, 0.3) * 0.9, near(z, z0 + 0.02, 0.07) * 0.8) * blot(co, n)
        # runs below the head, the collar ring, the capital and the cornice
        runs = np.maximum.reduce([below(z, zn - 0.02, 0.5 + 0.5 * W), below(z, bz - 0.035, 0.3 + 0.35 * W), below(z, k0, 0.25 + 0.3 * W) * 0.8,
                                  below(z, zbt, 0.1) * 0.6]) * run_a(co, n)
        # blooms along the riveted rims of the drum (worst underneath), under every moulding, at the bracket joints
        rim = near(np.abs(co[:, 1]), h2 - 0.027, 0.03) * (axial(co) > Rc - 0.03) * (0.35 + 0.65 * np.clip((zc - z) / Rc, 0.0, 1.0)) * blot(co, n)
        joints = near(z, z1 + 0.02, 0.05) * blot(co, n) * 0.7
        r = np.maximum.reduce([foot, runs * (0.55 + 0.45 * W), rim * (0.5 + 0.5 * W), joints, patch(co, n) * W * 0.9,
                               under(co, n) * (0.2 + 0.5 * W) * (0.5 + 0.5 * blot(co, n))])
        return np.clip(r * (0.55 + 0.6 * W), 0.0, 1.0)

    def grime(co, n):
        z = co[:, 2]
        splash = np.maximum(below(z, 0.22, 0.22), below(z, 0.6, 0.6) * 0.55 * blot(co, n))
        drips = np.maximum.reduce([below(z, zn + Rc * 0.5, Rc * 0.7) * (axial(co) > Rc - 0.03) * 0.8, below(z, zn, 0.7) * 0.8,
                                   below(z, bz - 0.035, 0.45) * 0.7, below(z, k0 + 0.07, 0.45) * 0.75]) * run_b(co, n)
        dust = upf(co, n) * 0.35
        return np.clip(np.maximum.reduce([splash, drips, dust, under(co, n) * 0.45]) * (0.6 + 0.4 * W), 0.0, 1.0)

    def chips(co, n):
        z = co[:, 2]
        # kicked and knocked paint up to knee height, rubbed flute crests at hand height
        kick = np.maximum(below(z, 0.75, 0.7), 0.25) * (z < ped_h + 0.2) * fine(co, n)
        c = 0.0
        if nf > 0:
            crest = (0.5 + 0.5 * np.cos(np.arctan2(co[:, 1], co[:, 0]) * nf)) ** 6
            c = crest * (z > fz0) * (z < fz1) * (np.hypot(co[:, 0], co[:, 1]) < r0 + 0.01) * (0.35 + 0.65 * blot(co, n)) * np.clip(1.4 - (z - fz0) / 1.2, 0.3, 1.0)
        return np.clip(np.maximum(kick, c) * (0.45 + 0.55 * W), 0.0, 1.0)

    for o in iron_objs:
        paint(o, grime=grime, special=rust, wear=chips)
    # brass: tarnished in the hollows, verdigris weeping from the undersides, still bright where it is rubbed
    bn = noise_mask(0.06, seed=sd + 9, lo=0.35 - 0.2 * W, hi=0.8)
    for o in brass_objs:
        paint(o, special=lambda co, n: np.clip((0.3 + 0.7 * W) * bn(co, n) * (0.5 + 0.8 * under(co, n)), 0.0, 1.0),
              grime=lambda co, n: np.clip(0.35 + 0.35 * W + 0.3 * blot(co, n) - 0.3 * upf(co, n), 0.0, 1.0))

    if p.lit and p.intensity > 0.01:
        # the lamp sits inside the case: the main light is shadowed by the iron drum, so it only leaves through the two opal
        # dials (pools in front of and behind the clock, a band of shade under the drum); the near plane keeps the hands
        # from throwing giant shadows.
        light("dials", (0, 0, zc), color="#ffc98a", intensity=p.intensity, range=10.0, flicker=0.03, shadow=True, shadow_near=0.7 * Rd)
        # amber bounce off the two lit pools of paving onto the pedestal and the foot of the column. The sockets sit just
        # under the paving so the engine's light halo stays hidden (a fill hung in mid-air shows up as a floating orb).
        for sy, nm in ((-1, "spill_front"), (1, "spill_back")):
            light(nm, (0, sy * 1.6, -0.3), color="#ffc98a", intensity=p.intensity * 0.14, range=6.0, flicker=0.03)
    socket("sign_l", "attach", (-rad(z0 + L * 0.62) - 0.01, 0, z0 + L * 0.62), rot=(0, 0, 90), role="banner")
