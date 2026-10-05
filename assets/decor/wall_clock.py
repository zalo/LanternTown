"""Double-sided drum clock projecting from a wall on a wrought / cast iron bracket.
A moulded cast-iron drum (axis parallel to the wall, one dial each way along the street) is carried by a
twisted square arm, an arched top stay and a bowed lower brace with collared S- and C-scroll infill, all
bolted to the wall through a lobed back plate. Each face has a recessed opal-glass dial lit from inside
(or an enamel dial when unlit) with painted roman numerals / batons, a minute track, spade hands on a
brass hub, a hinged brass bezel and a domed cover glass. A crest (spire / crown / orb) flanked by scrolls
sits on top, a turned drop finial hangs below.
"""
from math import atan2, cos, hypot, pi, radians, sin, sqrt

from tk import *

META = {"name": "Bracket wall clock", "class": "component", "mount": "wall", "mount_height": 3.3,
        "focus": [0.0, -0.7, 0.0]}
PARAMS = {
    "dial": {"default": 0.6, "min": 0.4, "max": 0.9, "desc": "dial diameter (m)"},
    "reach": {"default": 0.36, "min": 0.24, "max": 0.7, "desc": "clear gap between the wall and the drum (m)"},
    "time": {"default": "10:09", "desc": "time shown on both faces, HH:MM"},
    "numerals": {"default": "roman", "choices": ["roman", "batons"], "desc": "dial marking style"},
    "crest": {"default": "spire", "choices": ["spire", "crown", "orb"], "desc": "crest on top of the drum"},
    "paint": {"default": "#25382e", "desc": "case and bracket paint colour"},
    "dial_colour": {"default": "#f8e3b6", "desc": "dial colour (opal glass tint when lit, enamel when unlit; dark = light marks)"},
    "gilt": {"default": "accents", "choices": ["none", "accents", "rich"], "desc": "amount of gilding / brass"},
    "lit": {"default": True, "desc": "dials lit from inside (dark dial: pierced numerals glow instead of the face)"},
    "intensity": {"default": 2.0, "min": 0.3, "max": 8.0, "desc": "light spill per dial (cd, for a 0.6 m dial)"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "rust / verdigris / grime amount"},
}
VARIANTS = {
    "station_large": {"dial": 0.85, "reach": 0.5, "time": "08:43", "paint": "#1c2f4a", "gilt": "rich", "crest": "crown",
                      "seed": 4},
    "jeweller_black": {"dial": 0.45, "reach": 0.28, "time": "04:52", "paint": "#5a1414", "gilt": "rich", "crest": "orb",
                       "numerals": "batons", "dial_colour": "#15161a", "wear": 0.15, "seed": 11},
    "plain_worn": {"dial": 0.55, "reach": 0.6, "time": "01:27", "paint": "#2c2b28", "gilt": "none", "crest": "orb",
                   "dial_colour": "#e6d3a4", "intensity": 1.1, "wear": 0.95, "seed": 23},
    "batons_green": {"dial": 0.7, "reach": 0.3, "time": "07:21", "paint": "#234a3a", "gilt": "accents", "crest": "crown",
                     "numerals": "batons", "dial_colour": "#ffe2b0", "seed": 31},
}

ROMAN = ["I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


# ------------------------------------------------------------------ 2D helpers (y, z) in the bracket plane
def _volute(c, r0, r1, turns, a_end, ccw=True, tol=0.0015):
    """Spiral from the inner eye (r0) out to r1, ending at polar angle a_end. Returns (pts, end tangent).
    Angular steps follow a chord-sag tolerance, so the tight eye does not waste segments."""
    sg = 1.0 if ccw else -1.0
    total = turns * 2.0 * pi
    phis = [0.0]
    while phis[-1] < total - 1e-6:
        r = r0 + (r1 - r0) * (phis[-1] / total) ** 0.8
        phis.append(min(total, phis[-1] + clamp(sqrt(8.0 * tol / r), radians(16), radians(45))))
    if len(phis) > 3 and total - phis[-2] < radians(8):
        del phis[-2]
    pts = []
    for ph in phis:
        r = r0 + (r1 - r0) * (ph / total) ** 0.8
        a = radians(a_end) - sg * (total - ph)
        pts.append((c[0] + r * cos(a), c[1] + r * sin(a)))
    a = radians(a_end)
    return pts, (-sin(a) * sg, cos(a) * sg)


def _bez(p0, p1, p2, p3, n=10):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        out.append(tuple(u * u * u * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t * t * t * p3[k] for k in (0, 1)))
    return out


def _link(A, tan_a, B, tan_b, n=7, pull=0.42):
    """A (ending with direction tan_a) -> bezier -> B (starting with direction tan_b)."""
    e0, e1 = A[-1], B[0]
    d = hypot(e1[0] - e0[0], e1[1] - e0[1]) * pull
    mid = _bez(e0, (e0[0] + tan_a[0] * d, e0[1] + tan_a[1] * d), (e1[0] - tan_b[0] * d, e1[1] - tan_b[1] * d), e1, n)
    return A + mid[1:-1] + B


def _z_at(pts, y):
    """z of a (y, z) polyline at the given y (nearest segment, linear)."""
    best = pts[0][1]
    for (y0, z0), (y1, z1) in zip(pts, pts[1:]):
        lo, hi = min(y0, y1), max(y0, y1)
        if lo <= y <= hi and hi - lo > 1e-9:
            return z0 + (z1 - z0) * (y - y0) / (y1 - y0)
    return best


def _lum(h):
    h = h.lstrip("#")
    try:
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except Exception:
        return 1.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _parse_time(s):
    try:
        hh, mm = str(s).replace(".", ":").split(":")[:2]
        return int(hh) % 12, int(mm) % 60
    except Exception:
        return 10, 9


# ------------------------------------------------------------------ dial printing (flat, dial-local x right / y up)
def _dial_marks(name, R, style, z, level=0):
    """level 0: the full printing. 1: the same without the 48 minute ticks. 2: twelve bold hour marks only
    (the stand-ins of lod 1 / lod 2, see build())."""
    V, F = [], []

    def quad(pts):
        ar = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))
        if ar < 0:
            pts = pts[::-1]
        i0 = len(V)
        V.extend((q[0], q[1], z) for q in pts)
        F.append((i0, i0 + 1, i0 + 2, i0 + 3))

    def pol(th, r, a=0.0):                      # th clockwise from 12 o'clock (rad); a = tangential offset
        return (r * sin(th) + a * cos(th), r * cos(th) - a * sin(th))

    def ring(r0, r1, n=40):
        for k in range(n):
            a0, a1 = 2 * pi * k / n, 2 * pi * (k + 1) / n
            quad([pol(a0, r0), pol(a1, r0), pol(a1, r1), pol(a0, r1)])

    if level >= 2:
        for hr in range(12):
            th = radians(hr * 30.0)
            w = (0.075 if hr % 3 == 0 else 0.05) * R
            quad([pol(th, 0.64 * R, -w), pol(th, 0.64 * R, w), pol(th, 0.985 * R, w), pol(th, 0.985 * R, -w)])
        return mesh_object(name, V, F)
    # minute track: two hairlines, 60 ticks, heavier five-minute blocks
    ring(0.895 * R, 0.905 * R)
    for k in range(60):
        if level == 1 and k % 5:
            continue
        th = 2 * pi * k / 60
        w = 0.014 * R if k % 5 == 0 else 0.0045 * R
        quad([pol(th, 0.905 * R, -w), pol(th, 0.905 * R, w), pol(th, 0.985 * R, w), pol(th, 0.985 * R, -w)])
    ring(0.615 * R, 0.623 * R, 36)

    rn = 0.755 * R
    if style == "roman":
        h = 0.2 * R
        tw, tn, gap, ts = 0.042 * R, 0.014 * R, 0.02 * R, 0.011 * R
        wv = 0.5 * h
        for hr in range(1, 13):
            th = radians(hr * 30.0)
            s = ROMAN[hr - 1]
            widths = [tw if ch == "I" else wv for ch in s]
            total = sum(widths) + gap * (len(s) - 1)
            x = -total / 2

            def P(a, b):
                return pol(th, rn + b, a)

            for ch, w in zip(s, widths):
                cx = x + w / 2
                if ch == "I":
                    quad([P(cx - tw / 2, -h / 2), P(cx + tw / 2, -h / 2), P(cx + tw / 2, h / 2), P(cx - tw / 2, h / 2)])
                elif ch == "V":
                    quad([P(cx - wv / 2, h / 2), P(cx - wv / 2 + tw, h / 2), P(cx + tw * 0.45, -h / 2), P(cx - tw * 0.55, -h / 2)])
                    quad([P(cx + wv / 2 - tn, h / 2), P(cx + wv / 2, h / 2), P(cx + tw * 0.45, -h / 2), P(cx + tw * 0.45 - tn, -h / 2)])
                else:   # X
                    quad([P(cx - wv / 2, h / 2), P(cx - wv / 2 + tw, h / 2), P(cx + wv / 2, -h / 2), P(cx + wv / 2 - tw, -h / 2)])
                    quad([P(cx + wv / 2 - tn, h / 2), P(cx + wv / 2, h / 2), P(cx - wv / 2 + tn, -h / 2), P(cx - wv / 2, -h / 2)])
                x += w + gap
            e = 0.012 * R                         # slab serifs tying the numeral together
            for b0 in (h / 2 - ts * 0.2, -h / 2 - ts * 0.8):
                quad([P(-total / 2 - e, b0), P(total / 2 + e, b0), P(total / 2 + e, b0 + ts), P(-total / 2 - e, b0 + ts)])
    else:
        for hr in range(12):
            th = radians(hr * 30.0)
            big = hr % 3 == 0
            wo, wi = (0.05 * R, 0.03 * R) if big else (0.028 * R, 0.014 * R)
            r0, r1 = (0.64 * R, 0.865 * R) if big else (0.7 * R, 0.865 * R)
            if hr == 0:                           # double baton at twelve
                for off in (-0.045 * R, 0.045 * R):
                    quad([pol(th, r0, off - wi * 0.7), pol(th, r0, off + wi * 0.7), pol(th, r1, off + wo * 0.7), pol(th, r1, off - wo * 0.7)])
            else:
                quad([pol(th, r0, -wi), pol(th, r0, wi), pol(th, r1, wo), pol(th, r1, -wo)])
            d = 0.016 * R                         # lozenge on the inner ring
            quad([pol(th, 0.56 * R, 0), pol(th, 0.56 * R + d, d * 0.6), pol(th, 0.56 * R + 2 * d, 0), pol(th, 0.56 * R + d, -d * 0.6)])
    return mesh_object(name, V, F)


def _rot_cw(pts, deg):
    c, s = cos(radians(deg)), sin(radians(deg))
    return [(x * c + y * s, -x * s + y * c) for x, y in pts]


def _lift(o, dz):
    transform_verts(o, lambda co: Vector((co.x, co.y, co.z + dz)))
    return o


def build(p, rng):
    D = p.dial
    Rd = D / 2
    s = clamp(D / 0.6, 0.75, 1.4)                 # moulding scale
    bs = clamp(0.55 + 0.45 * D / 0.6, 0.8, 1.25)  # bracket bar scale
    Rc = Rd + 0.07 * s                            # case outer radius
    Rb = Rc - 0.02 * s                            # barrel radius between the rim mouldings
    W = 0.10 + 0.22 * D                           # drum depth
    xf = W / 2
    rec = 0.02 * s                                # dial recess
    g = p.reach
    yc = -(g + Rc)
    C = (0.0, yc, 0.0)
    hh, mm = _parse_time(p.time)
    dark = _lum(p.dial_colour) < 0.3

    # ---------------------------------------------------------------- materials
    iron = mat("iron_painted", p.paint, scale=1.8)
    brass = mat("brass", rough=0.85)
    black = mat("iron_painted", "#0c0c0e", wear=0.0)
    gilt = brass if p.gilt != "none" else iron
    gilt2 = brass if p.gilt == "rich" else iron   # extra gilding on the rich version
    if p.lit and not dark:
        dial_m = mat("glass_frosted", p.dial_colour, glow=1.3, opacity=1.0)
    else:
        dial_m = mat("ceramic_glazed", p.dial_colour)
    if dark:
        hand_m = brass if p.gilt != "none" else mat("ceramic_glazed", "#e6dcc2", wear=0.0)
        mark_m = mat("emissive", "#ffd9a0", glow=2.5) if p.lit else hand_m   # pierced dial over opal glass
    else:
        mark_m = hand_m = mat("wood_painted", "#0c0c0e", rough=1.8, wear=0.0, special=0.0)     # matt non-metal near-black, every level
    glass_m = mat("glass_clear", "#f4fff8", opacity=0.1)

    iron_objs, gilt_objs, gilt2_objs = [], [], []

    # ---------------------------------------------------------------- drum case (one turned profile)
    half = [(Rd + 0.001, xf - rec), (Rd + 0.008 * s, xf - 0.004 * s), (Rd + 0.014 * s, xf),
            (Rc - 0.016 * s, xf + 0.003 * s), (Rc - 0.005 * s, xf - 0.001 * s), (Rc, xf - 0.009 * s), (Rc, xf - 0.022 * s),
            (Rc - 0.008 * s, xf - 0.03 * s), (Rc - 0.014 * s, xf - 0.037 * s),
            (Rc - 0.008 * s, xf - 0.045 * s), (Rc - 0.008 * s, xf - 0.054 * s), (Rc - 0.017 * s, xf - 0.06 * s),
            (Rb, xf - 0.07 * s), (Rb, 0.024 * s), (Rb + 0.007 * s, 0.018 * s)]
    prof = [(r, -z) for r, z in half] + [(r, z) for r, z in reversed(half)]
    case = lathe("case", prof, segments=40, loc=C, rot=(90, 0, 90))
    iron_objs.append(case)
    assign(case, iron)
    if p.gilt == "rich":
        # gold-lined bead on each rim and the girdle band
        zb_ = xf - 0.0495 * s
        assign_faces(case, brass, lambda c, n: (abs(abs(c.z) - zb_) < 0.0055 * s and hypot(c.x, c.y) > Rc - 0.0125 * s) or
                     (abs(c.z) < 0.019 * s and hypot(c.x, c.y) > Rb + 0.002 * s))

    # studs round each face
    stud = lathe("stud", [(0.0065 * s, 0.0), (0.0052 * s, 0.003 * s), (0.0, 0.0046 * s)], segments=6)
    sm = []
    ns = 10
    rs = Rd + 0.042 * s
    for side in (1, -1):
        for k in range(ns):
            a = 2 * pi * (k + 0.5) / ns
            sm.append(trs((side * (xf + 0.0015 * s), yc + rs * cos(a), rs * sin(a)), (0, 90 * side, 0)))
    iron_objs.append(instances_to_mesh("studs", stud, sm, keep_base=False))

    # ---------------------------------------------------------------- the two faces
    zd = xf - rec
    R = Rd - 0.004 * s                             # visible dial radius
    hour_deg = (hh + mm / 60.0) * 30.0
    min_deg = mm * 6.0
    for side in (1, -1):
        rot = (90, 0, 90 * side)
        tag = "r" if side > 0 else "l"
        bz = lathe("bezel_" + tag, [(Rd + 0.03 * s, xf - 0.001), (Rd + 0.029 * s, xf + 0.008 * s), (Rd + 0.023 * s, xf + 0.015 * s),
                                    (Rd + 0.009 * s, xf + 0.017 * s), (Rd - 0.004 * s, xf + 0.008 * s),
                                    (Rd - 0.004 * s, zd + 0.0005)], segments=40, loc=C, rot=rot)
        gilt_objs.append(bz)
        dial = lathe("dial_" + tag, [(Rd - 0.003 * s, zd), (0.0, zd)], segments=40, loc=C, rot=rot)
        assign(dial, dial_m)
        # CLEARANCE RULE: the marks stand 4.5 mm above the dial (one of the two glows: the opal dial, or the
        # pierced marks of a dark dial; it was 1.2 mm) and the hands and hub ride HL higher, so the hour hand
        # stays 4.5 mm or more above the marks
        HL = 0.006
        marks = _dial_marks("marks_" + tag, R, p.numerals, zd + 0.0045)
        place(marks, loc=C, rot=rot)
        assign(marks, mark_m)
        # hands
        Lh, Lm = 0.56 * R, 0.88 * R
        w = 0.05 * R
        hour = [(-0.32 * w, -0.2 * Lh), (0.32 * w, -0.2 * Lh), (0.5 * w, 0.0), (0.3 * w, 0.56 * Lh), (1.25 * w, 0.71 * Lh), (0.95 * w, 0.83 * Lh),
                (0.0, Lh), (-0.95 * w, 0.83 * Lh), (-1.25 * w, 0.71 * Lh), (-0.3 * w, 0.56 * Lh), (-0.5 * w, 0.0)]
        minute = [(-0.3 * w, -0.2 * Lm), (0.3 * w, -0.2 * Lm), (0.48 * w, 0.0), (0.42 * w, 0.62 * Lm), (0.6 * w, 0.72 * Lm), (0.1 * w, Lm),
                  (-0.1 * w, Lm), (-0.6 * w, 0.72 * Lm), (-0.42 * w, 0.62 * Lm), (-0.48 * w, 0.0)]
        hd = _lift(shape_extrude("hour_" + tag, _rot_cw(hour, hour_deg), depth=0.0025 * s), zd + 0.004 * s + HL)
        md = _lift(shape_extrude("minute_" + tag, _rot_cw(minute, min_deg), depth=0.0025 * s), zd + 0.0085 * s + HL)
        for o in (hd, md):
            place(o, loc=C, rot=rot)
            assign(o, hand_m)
        hub = lathe("hub_" + tag, [(0.06 * R, zd + 0.001), (0.06 * R, zd + 0.006 * s + HL), (0.04 * R, zd + 0.013 * s + HL), (0.022 * R, zd + 0.016 * s + HL),
                                   (0.0, zd + 0.017 * s + HL)], segments=12, loc=C, rot=rot)
        assign(hub, brass)

        # The dial is the identity of the clock: what stands for its parts at lod 1 / lod 2 is modelled here from
        # the same numbers (tk.lod_standin: the full model is not touched, the reducer never simplifies these).
        #   lod 1: hands as flat fronts, the printing without its 48 minute ticks; the dial as modelled
        #   lod 2: a 16-sided dial, 12 bold hour marks, bold hands of the same lengths and angles, 57 mm wide =
        #          1.5 px at 44 m (never over 0.2 x the dial radius). Layers keep their heights over the dial.
        zh, zm = zd + 0.0065 * s + HL, zd + 0.011 * s + HL          # front planes of the hour / minute hand

        def d_poly(nm, pts, z, m):
            o = mesh_object(nm, [(x, y, z) for x, y in pts], [tuple(range(len(pts)))])
            place(o, loc=C, rot=rot)
            assign(o, m)
            return o

        def d_marks_mid():
            o = _dial_marks("marks_mid_" + tag, R, p.numerals, zd + 0.0045, level=1)
            place(o, loc=C, rot=rot)
            assign(o, mark_m)
            return o

        def d_hands_mid():
            return [d_poly("hour_mid_" + tag, _rot_cw(hour, hour_deg), zh, hand_m),
                    d_poly("minute_mid_" + tag, _rot_cw(minute, min_deg), zm, hand_m)]

        def d_far():
            o = lathe("dial_far_" + tag, [((Rd - 0.003 * s) / cos(pi / 16), zd), (0.0, zd)], segments=16, loc=C, rot=rot)
            assign(o, dial_m)
            mk = _dial_marks("marks_far_" + tag, R, p.numerals, zd + 0.0045, level=2)
            place(mk, loc=C, rot=rot)
            assign(mk, mark_m)
            wb = min(max(0.057, 0.11 * R), 0.2 * R)
            hb = [(-0.5 * wb, -0.2 * Lh), (0.5 * wb, -0.2 * Lh), (0.6 * wb, 0.6 * Lh), (0.0, Lh), (-0.6 * wb, 0.6 * Lh)]
            mb = [(-0.5 * wb, -0.2 * Lm), (0.5 * wb, -0.2 * Lm), (0.5 * wb, 0.82 * Lm), (0.0, Lm), (-0.5 * wb, 0.82 * Lm)]
            return [o, mk, d_poly("hour_far_" + tag, _rot_cw(hb, hour_deg), zh, hand_m),
                    d_poly("minute_far_" + tag, _rot_cw(mb, min_deg), zm, hand_m)]

        lod_standin(dial, far=d_far)
        lod_standin(marks, mid=d_marks_mid, far=())
        lod_standin([hd, md], mid=d_hands_mid, far=())
        gl = lathe("glass_" + tag, [(Rd - 0.003 * s, xf + 0.006 * s), (Rd * 0.66, xf + 0.027 * s),
                                    (0.0, xf + 0.04 * s)], segments=24, loc=C, rot=rot)
        assign(gl, glass_m)
        # bezel hinge (towards the wall) and catch (outer side)
        hy = (Rd + 0.031 * s)
        gilt_objs.append(cylinder("hinge_" + tag, 0.006 * s, 0.05 * s, loc=(side * (xf + 0.008 * s), yc + hy, 0.0), verts=8))
        gilt_objs.append(box("catch_" + tag, (0.012 * s, 0.016 * s, 0.022 * s), loc=(side * (xf + 0.008 * s), yc - hy, 0.0), bevel=0.002, segments=1))
        if p.lit:
            light("dial_" + tag, (side * (xf + D), yc, 0.0), color="#ffd49a", intensity=p.intensity * (D / 0.6) ** 2,
                  range=4.0 + 2.0 * D, flicker=0.04, shadow_near=0.9 * D)

    # ---------------------------------------------------------------- rosette boss on the outer end of the barrel
    ros = lathe("end_rosette", [(0.062 * s, 0.0), (0.062 * s, 0.006 * s), (0.05 * s, 0.012 * s), (0.036 * s, 0.01 * s), (0.03 * s, 0.016 * s),
                                (0.016 * s, 0.022 * s), (0.0, 0.024 * s)], segments=24, loc=(0, yc - Rb - 0.002 * s, 0), rot=(90, 0, 0))

    def _petal(co):
        r = hypot(co.x, co.y)
        if r < 0.034 * s:
            return co
        k = 1.0 - 0.13 * (0.5 + 0.5 * cos(8 * atan2(co.y, co.x)))
        return Vector((co.x * k, co.y * k, co.z))

    transform_verts(ros, _petal)
    gilt_objs.append(ros)

    # ---------------------------------------------------------------- crest
    top = Rb + 0.004 * s
    crest_i = [(0.058 * s, top - 0.02 * s), (0.058 * s, top + 0.008 * s), (0.05 * s, top + 0.016 * s), (0.036 * s, top + 0.022 * s),
               (0.026 * s, top + 0.034 * s), (0.03 * s, top + 0.042 * s), (0.03 * s, top + 0.047 * s)]
    z0 = top + 0.047 * s
    if p.crest == "spire":
        crest_i += [(0.044 * s, z0 + 0.012 * s), (0.05 * s, z0 + 0.03 * s), (0.044 * s, z0 + 0.05 * s), (0.024 * s, z0 + 0.062 * s),
                    (0.016 * s, z0 + 0.07 * s), (0.0, z0 + 0.07 * s)]
        gilt_objs.append(lathe("crest_spire", [(0.015 * s, z0 + 0.068 * s), (0.026 * s, z0 + 0.078 * s), (0.026 * s, z0 + 0.084 * s),
                                               (0.014 * s, z0 + 0.094 * s), (0.02 * s, z0 + 0.108 * s), (0.013 * s, z0 + 0.124 * s),
                                               (0.008 * s, z0 + 0.15 * s), (0.0, z0 + 0.235 * s)], segments=16, loc=C))
    elif p.crest == "orb":
        crest_i += [(0.018 * s, z0 + 0.008 * s), (0.016 * s, z0 + 0.016 * s), (0.0, z0 + 0.016 * s)]
        ro = 0.046 * s
        zo = z0 + 0.012 * s + ro
        orb = [(ro * sin(radians(a)), zo - ro * cos(radians(a))) for a in range(20, 181, 20)]
        orb = orb[:-1] + [(0.012 * s, zo + ro * 0.985), (0.008 * s, zo + ro + 0.012 * s), (0.014 * s, zo + ro + 0.022 * s), (0.0, zo + ro + 0.045 * s)]
        gilt_objs.append(lathe("crest_orb", orb, segments=20, loc=C))
    else:   # crown
        crest_i += [(0.02 * s, z0 + 0.006 * s), (0.0, z0 + 0.006 * s)]
        zc0 = z0 + 0.004 * s
        cr = 0.062 * s
        gilt_objs.append(lathe("crown_band", [(cr - 0.008 * s, zc0), (cr + 0.003 * s, zc0 + 0.002 * s), (cr, zc0 + 0.008 * s), (cr, zc0 + 0.022 * s),
                                              (cr + 0.004 * s, zc0 + 0.027 * s), (cr - 0.008 * s, zc0 + 0.03 * s)], segments=16, loc=C))
        cap = lathe("crown_cap", [(cr - 0.006 * s, zc0 + 0.022 * s), (cr - 0.004 * s, zc0 + 0.05 * s), (cr * 0.6, zc0 + 0.074 * s), (0.0, zc0 + 0.082 * s)],
                    segments=12, loc=C)
        assign(cap, mat("canvas", "#6d1420", grime=1.5))
        for k in range(4):
            a = radians(45 + 90 * k + rng.jitter(0, 2))
            ca, sa = cos(a), sin(a)
            pts = _bez((cr, zc0 + 0.028 * s), (cr + 0.028 * s, zc0 + 0.06 * s), (cr * 0.75, zc0 + 0.102 * s), (0.004 * s, zc0 + 0.094 * s), 6)
            gilt_objs.append(sweep(f"crown_arch{k}", [(r * ca, yc + r * sa, z) for r, z in pts], 0.0055 * s, sides=4, smooth_path=False))
            a2 = a + radians(45)
            gilt_objs.append(cone(f"crown_fleuron{k}", 0.011 * s, 0.0, 0.036 * s, loc=(cr * cos(a2), yc + cr * sin(a2), zc0 + 0.046 * s), verts=6))
        gilt_objs.append(sphere("crown_orb", 0.017 * s, loc=(0, yc, zc0 + 0.106 * s), segments=10, rings=6))
        gilt_objs.append(box("crown_cross_v", (0.006 * s, 0.006 * s, 0.04 * s), loc=(0, yc, zc0 + 0.14 * s)))
        gilt_objs.append(box("crown_cross_h", (0.006 * s, 0.026 * s, 0.006 * s), loc=(0, yc, zc0 + 0.145 * s)))
    iron_objs.append(lathe("crest_base", crest_i, segments=16, loc=C))

    # ---------------------------------------------------------------- flat-bar scrollwork
    t = 0.012 * bs                                 # bar thickness (in the bracket plane)
    bw = 0.03 * bs                                 # bar width (across, along X)
    bar = rounded_rect(t, bw, 0.003 * bs, 1)

    def taper(u):
        return 0.5 + 0.5 * min(1.0, u * 7.0, (1.0 - u) * 7.0)

    def taper_tip(u):
        return 0.5 + 0.5 * min(1.0, u * 6.0)

    def scroll(name, pts, fn=taper):
        return sweep(name, [(0.0, q[0], q[1]) for q in pts], bar, up=(1, 0, 0), smooth_path=False, scale_fn=fn)

    def collar(name, y, z, rot_x=0.0, h=None):
        return box(name, (bw + 0.008 * bs, 0.02 * bs, h or (2 * t + 0.01 * bs)), loc=(0, y, z), rot=(rot_x, 0, 0), bevel=0.0025, segments=1)

    # crest scrolls: slim bar volutes resting on the drum shoulders, tails rising into the crest
    cbar = rounded_rect(0.009 * s, 0.02 * s, 0.0025 * s, 1)
    rv = 0.042 * s
    ra = Rb + 0.007 * s + 0.0045 * s
    Rv = ra + rv
    eu = {"spire": 0.04, "crown": 0.056, "orb": 0.024}[p.crest] * s
    ez = top + {"spire": 0.064, "crown": 0.06, "orb": 0.05}[p.crest] * s
    for sgn in (1, -1):
        ang = 60.0 + rng.jitter(0, 1.5)
        cu, cz = Rv * cos(radians(ang)), Rv * sin(radians(ang))
        vol, _ = _volute((cu, cz), 0.008 * s, rv, 1.45, ang + 180.0, ccw=False, tol=0.002)
        a1 = radians(ang + 9.0)
        p_arc = (ra * cos(a1), ra * sin(a1))
        rise = _bez(p_arc, (p_arc[0] - sin(a1) * 0.035 * s, p_arc[1] + cos(a1) * 0.035 * s), (eu + 0.04 * s, ez - 0.03 * s), (eu, ez), 5)
        pts = [(0.0, yc + sgn * u, z) for u, z in vol + rise]
        gilt2_objs.append(sweep(f"crest_scroll{sgn}", pts, cbar, up=(1, 0, 0), smooth_path=False, scale_fn=taper_tip))
        iron_objs.append(collar(f"crest_clip{sgn}", yc + sgn * cu, 0.0, h=0.012 * bs))
        place(iron_objs[-1], loc=(0, yc + sgn * (ra - 0.003 * s) * cos(radians(ang)), (ra - 0.003 * s) * sin(radians(ang))), rot=(sgn * (90 - ang) * -1.0, 0, 0))

    # ---------------------------------------------------------------- drop finial
    b = -Rb - 0.004 * s
    drop = [(0.012 * s, b - 0.158 * s), (0.008 * s, b - 0.15 * s), (0.016 * s, b - 0.138 * s), (0.032 * s, b - 0.112 * s), (0.046 * s, b - 0.078 * s),
            (0.042 * s, b - 0.05 * s), (0.024 * s, b - 0.036 * s), (0.022 * s, b - 0.028 * s), (0.034 * s, b - 0.022 * s), (0.034 * s, b - 0.016 * s),
            (0.056 * s, b - 0.008 * s), (0.06 * s, b + 0.0), (0.06 * s, b + 0.02 * s)]
    iron_objs.append(lathe("drop", drop, segments=16, loc=C))
    rt = 0.021 * s
    zt_ = b - 0.158 * s - rt * 0.8
    tip = [(0.0, zt_ - rt - 0.022 * s), (0.006 * s, zt_ - rt - 0.006 * s)] + \
          [(rt * sin(radians(a)), zt_ - rt * cos(radians(a))) for a in range(25, 160, 22)] + [(0.011 * s, b - 0.156 * s)]
    gilt_objs.append(lathe("drop_tip", tip, segments=12, loc=C))

    # ---------------------------------------------------------------- bracket: arm, stay, brace, scrolls
    ta = 0.034 * bs                                # square arm
    pt = 0.012 * bs                                # back plate thickness
    yw = -pt                                       # front face of the plate
    # twisted square arm between plain stubs
    y_end = -(g + (Rc - Rb) + 0.012)
    stub = min(0.07 * bs, g * 0.22)
    y0t, y1t = yw - stub, -(g - stub * 0.9)
    Lt = abs(y1t - y0t)
    turns = max(1, int(round(Lt / 0.17)))
    nseg = max(10, min(18, int(Lt / 0.016)))
    sq = rounded_rect(ta, ta, 0.004 * bs, 1)
    iron_objs.append(sweep("arm_twist", [(0.0, y0t + (y1t - y0t) * i / nseg, 0.0) for i in range(nseg + 1)], sq, up=(0, 0, 1),
                           smooth_path=False, twist_deg=360.0 * turns, cap=False))
    iron_objs.append(box("arm_root", (ta, stub + 0.004, ta), loc=(0, (yw + y0t) / 2, 0), bevel=0.004 * bs, segments=1))
    iron_objs.append(box("arm_end", (ta, abs(y_end - y1t), ta), loc=(0, (y_end + y1t) / 2, 0), bevel=0.004 * bs, segments=1))
    for i, yy in enumerate((y0t, y1t)):
        iron_objs.append(box(f"arm_collar{i}", (ta + 0.014 * bs, 0.016 * bs, ta + 0.014 * bs), loc=(0, yy, 0), bevel=0.004 * bs, segments=1))
    # trumpet boss where the arm enters the drum, cushion block against the plate
    iron_objs.append(lathe("arm_boss", [(0.027 * bs, 0.0), (0.034 * bs, 0.006 * bs), (0.034 * bs, 0.014 * bs), (0.029 * bs, 0.02 * bs), (0.031 * bs, 0.03 * bs),
                                        (0.044 * bs, 0.044 * bs), (0.062 * bs, 0.052 * bs), (0.066 * bs, 0.07 * bs)], segments=20,
                           loc=(0, -(g + (Rc - Rb)) + 0.05 * bs, 0), rot=(90, 0, 0)))
    iron_objs.append(box("arm_cushion", (0.07 * bs, 0.014 * bs, 0.07 * bs), loc=(0, yw - 0.006 * bs, 0), bevel=0.005 * bs, segments=1))

    if p.lit:
        # gas / cable feed: a lead pipe out of the wall, clipped along the side of the arm into the boss
        px = ta / 2 + 0.006 * bs
        pipe = [(0.03 * bs, -0.001, 0.05 * bs), (0.03 * bs, yw - 0.02 * bs, 0.05 * bs), (px, yw - 0.05 * bs, 0.012 * bs), (px, yw - 0.08 * bs, 0.0),
                (px, -(g - 0.05 * bs), 0.0), (px * 0.6, -(g - 0.02 * bs), 0.0), (0.012 * bs, -(g + 0.01 * bs), 0.0)]
        fp = sweep("feed_pipe", pipe, 0.0045 * bs, sides=6, step=0.03)
        assign(fp, mat("lead"))
        iron_objs.append(box("pipe_clip", (0.006 * bs, 0.014 * bs, 0.02 * bs), loc=(px + 0.003 * bs, (y0t + y1t) / 2, 0), bevel=0.002, segments=1))

    # arched top stay to the drum
    aq = radians(38.0)
    yq, zq = yc + (Rb - 0.006) * cos(aq), (Rb - 0.006) * sin(aq)
    hu = zq + 0.05 * bs
    Ls = abs(yq - yw)
    stay = _bez((yw + 0.004, hu), (yw - 0.38 * Ls, hu + 0.055 * bs), (yq + 0.3 * Ls, zq + 0.075 * bs), (yq, zq), 14)
    iron_objs.append(scroll("stay", stay, lambda u: 1.0))
    # bowed lower brace up to the arm
    hl = clamp(0.5 * g + 0.3 * bs, 0.34, 0.66)
    yb = -(g - 0.032 * bs)
    bow = _bez((yw + 0.004, -hl), (yb * 0.78, -hl - 0.012), (yb, -hl * 0.6), (yb, -ta / 2 + 0.004), 16)
    iron_objs.append(scroll("bow", bow, lambda u: 1.0))
    iron_objs.append(collar("bow_collar", yb, -ta / 2 - 0.012 * bs, h=0.03 * bs))

    # S-scroll in the upper field: volute A sits on the arm by the wall, volute B hangs under the stay by the drum
    Hf = hu - ta / 2
    rA = 0.225 * Hf * rng.vary(1.0, 0.04)
    rB = 0.2 * Hf * rng.vary(1.0, 0.04)
    cA = (yw - t / 2 - rA, ta / 2 + t / 2 + rA)
    cBy = yq + rB + t
    for _ in range(3):
        cBz = _z_at(stay, cBy) - t - rB
        yd = yc + sqrt(max(Rb * Rb - cBz * cBz, 1e-6))
        cBy = max(yd + rB + t * 0.5 + 0.012 * s, -(g + Rc * 0.6))
    cBy = min(cBy, cA[0] - rA - rB - 0.02)
    cBz = _z_at(stay, cBy) - t - rB
    vA, tA = _volute(cA, 0.011 * bs, rA, 1.4, 90.0, ccw=True)
    vB, tB = _volute((cBy, cBz), 0.011 * bs, rB, 1.4, -90.0, ccw=True)
    s_pts = _link(vA, tA, vB[::-1], (-tB[0], -tB[1]))
    iron_objs.append(scroll("s_scroll", s_pts))
    iron_objs.append(collar("s_collar_a", cA[0], ta / 2 + t / 2 - 0.004 * bs, h=t + 0.012 * bs))
    iron_objs.append(collar("s_collar_b", cBy, cBz + rB + t / 2 + 0.002, h=2 * t + 0.008 * bs))

    # C-scroll in the lower field: big volute under the arm, small one on the bow, both curling to the wall
    rA2 = min(0.2 * hl, (g - 0.1 * bs) / 2) * rng.vary(1.0, 0.04)
    rB2 = 0.62 * rA2
    cA2 = (yw - t / 2 - rA2, -ta / 2 - t / 2 - rA2)
    cB2y = yw - t / 2 - rB2
    cB2 = (cB2y, _z_at(bow, cB2y) + t + rB2)
    vA2, tA2 = _volute(cA2, 0.011 * bs, rA2, 1.4, 180.0, ccw=True)
    vB2, tB2 = _volute(cB2, 0.011 * bs, rB2, 1.3, 180.0, ccw=False)
    c_pts = _link(vA2, tA2, vB2[::-1], (-tB2[0], -tB2[1]), pull=0.3)
    iron_objs.append(scroll("c_scroll", c_pts))
    iron_objs.append(collar("c_collar_a", cA2[0], -ta / 2 - t / 2 + 0.004 * bs, h=t + 0.012 * bs))
    iron_objs.append(collar("c_collar_b", cB2[0], cB2[1] - rB2 - t / 2, h=2 * t + 0.008 * bs))
    # bud between the C-scroll and the bow
    by_ = (cA2[0] - rA2 + yb) / 2
    bz_ = _z_at(bow, max(by_, yb * 0.98)) if False else -hl * 0.52
    gilt_objs.append(lathe("bud", [(0.0, -0.03 * bs), (0.012 * bs, -0.02 * bs), (0.017 * bs, 0.0), (0.012 * bs, 0.016 * bs), (0.005 * bs, 0.026 * bs),
                                   (0.0, 0.04 * bs)], segments=10, loc=(0, yb + t / 2 + 0.017 * bs, bz_), rot=(0, 0, 0)))

    # ---------------------------------------------------------------- back plate with lobed ends, bolts on rosette washers
    wb, rl = 0.046 * bs, 0.068 * bs
    zt_l, zb_l = hu + 0.045 * bs, -hl - 0.045 * bs
    th0 = atan2(sqrt(rl * rl - wb * wb), wb) * 180.0 / pi
    outline = circle_pts(rl, 14, -th0, 180.0 + th0, center=(0, zt_l)) + circle_pts(rl, 14, 180.0 - th0, 360.0 + th0, center=(0, zb_l))
    plate = shape_extrude("plate", outline, depth=pt - 0.002, rot=(90, 0, 0), bevel=0.002)
    iron_objs.append(plate)
    washer = lathe("washer", [(0.03 * bs, 0.0), (0.03 * bs, 0.003 * bs), (0.024 * bs, 0.007 * bs), (0.015 * bs, 0.007 * bs), (0.0, 0.007 * bs)], segments=12)
    bolt_b = bolt("plate_bolt", 0.012 * bs, 0.009 * bs, kind="hex")
    bz_list = [zt_l, zb_l, 0.085 * bs, -0.085 * bs]
    iron_objs.append(instances_to_mesh("washers", washer, [trs((0, yw, z), (90, 0, 0)) for z in bz_list[:2]], keep_base=False))
    r2 = rng.fork("bolts")
    iron_objs.append(instances_to_mesh("plate_bolts", bolt_b, [trs((0, yw - (0.006 * bs if i < 2 else 0.0), z), (90, r2.u(0, 60), 0),
                                                                    1.0 if i < 2 else 0.8) for i, z in enumerate(bz_list)], keep_base=False))

    # ---------------------------------------------------------------- materials + art-directed weathering
    for o in iron_objs:
        if o is not case:
            assign(o, iron)
    for o in gilt_objs:
        assign(o, gilt)
    for o in gilt2_objs:
        assign(o, gilt2)
    wr = p.wear
    sd = p.seed
    under = facing((0, 0, -1))
    upw = facing((0, 0, 1))
    for o in iron_objs + gilt_objs + gilt2_objs:
        is_brass = (o in gilt_objs and gilt is brass) or (o in gilt2_objs and gilt2 is brass)
        if is_brass:
            # verdigris where rain sits, tarnish underneath
            paint(o, special=mask_mul(noise_mask(0.06, seed=sd + 5, lo=0.45, hi=0.8), wr * 0.9),
                  grime=mask_mul(under, 0.35 + 0.4 * wr))
        else:
            # rust bleeding from the undersides and joints, dust and droppings on top, streaks down the plate
            paint(o, special=mask_mul(noise_mask(0.07, seed=sd + 2, lo=0.45 - 0.3 * wr, hi=0.75 - 0.25 * wr), 0.25 + 0.75 * wr,
                                      lambda co, n: 0.3 + 0.7 * under(co, n) ** 2),
                  grime=mask_mul(upw, noise_mask(0.12, seed=sd + 3, lo=0.2, hi=0.8), 0.3 + 0.5 * wr),
                  wear=mask_mul(noise_mask(0.03, seed=sd + 7, lo=0.62, hi=0.9), wr * 0.8))
    paint(plate, grime=mask_mul(height_ramp(zb_l - rl, zb_l + 0.35), 0.5 + 0.4 * wr),
          special=mask_mul(height_ramp(zb_l - rl, zb_l + 0.2), noise_mask(0.05, seed=sd + 9, lo=0.3, hi=0.7), wr))
