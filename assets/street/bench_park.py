"""Victorian park bench: two (or three) cast-iron frames joined by timber slats.

Each end frame is one flat casting drawn in the (depth, height) plane: a cabriole front leg that rolls
over the seat front into the seat rail, a sabre back leg that sweeps up into the reclined back standard,
an arm (volute scroll / serpent / plain strap loop) and scroll infill, standing on bolted-down pad feet.
An inboard bearer flange carries the slats, which are fixed with dome-headed coach bolts (nuts behind);
two tie rods with hex nuts pull the frames together. Slats sag, crook and wear individually from the seed.
"""
from tk import *
from math import acos, exp
import numpy as np

META = {"name": "Cast-iron and timber park bench", "class": "furniture", "mount": "ground",
        "auto_grime": 0.4, "focus": [-0.56, -0.17, 0.52], "lod": "far"}
PARAMS = {
    "length": {"default": 1.6, "min": 1.1, "max": 2.6, "desc": "overall length over the end frames (m)"},
    "slats": {"default": 6, "min": 4, "max": 8, "desc": "number of seat slats"},
    "back_slats": {"default": 5, "min": 2, "max": 7, "desc": "number of back slats"},
    "style": {"default": "scroll", "choices": ["scroll", "serpent", "plain"],
              "desc": "end-frame casting: volute scrolls, serpent arms, or plain strap"},
    "frames": {"default": 0, "choices": [0, 2, 3], "desc": "number of frames (0 = auto: a third centre frame from 2 m)"},
    "paint": {"default": "#22382c", "desc": "paint colour of the ironwork"},
    "wood": {"default": "oak", "choices": ["oak", "painted", "weathered"], "desc": "slat finish"},
    "wood_tone": {"default": "#7c5836", "desc": "stain / paint / tint colour of the slats"},
    "bolts": {"default": "brass", "choices": ["brass", "iron"], "desc": "coach-bolt heads: brass domes or painted iron"},
    "gilt": {"default": True, "desc": "gilded rosettes on the scroll eyes"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "age: slat sag and misalignment, rubbed seats, chipped paint, rust; from 0.6 one slat is a newer replacement"},
}
VARIANTS = {
    "serpent_long": {"style": "serpent", "length": 2.3, "slats": 7, "back_slats": 4, "paint": "#16261d",
                     "wood_tone": "#6e4526", "wear": 0.3, "seed": 7},
    "plain_weathered": {"style": "plain", "length": 1.3, "slats": 5, "back_slats": 3, "wood": "weathered",
                        "wood_tone": "#c9c2b4", "paint": "#23262b", "bolts": "iron", "gilt": False, "wear": 0.9, "seed": 12},
    "painted_green": {"style": "scroll", "length": 1.85, "slats": 8, "back_slats": 6, "wood": "painted",
                      "wood_tone": "#3d5a45", "paint": "#101312", "wear": 0.7, "seed": 23},
    "burgundy_short": {"style": "scroll", "length": 1.15, "slats": 5, "back_slats": 3, "paint": "#4a1616",
                       "wood_tone": "#a87444", "wear": 0.15, "seed": 31},
}

# ------------------------------------------------------------------------------------------------
# 2D helpers: every frame member is a polyline in the (y, z) plane of the casting
# ------------------------------------------------------------------------------------------------


def _cr(ctrl, step=0.005):
    """Dense Catmull-Rom polyline through 2D control points."""
    P = resample_path([(0.0, y, z) for y, z in ctrl], step=step)
    return [(float(q[1]), float(q[2])) for q in P]


def _thin(pts, ang=7.5, dmax=0.055):
    """Curvature-adaptive decimation: keep a point every `ang` degrees of turning or `dmax` metres."""
    out = [pts[0]]
    acc = run = 0.0
    for i in range(1, len(pts) - 1):
        ax, ay = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
        bx, by = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
        run += hypot(ax, ay)
        acc += abs(degrees(atan2(ax * by - ay * bx, ax * bx + ay * by)))
        if acc >= ang or run >= dmax:
            out.append(pts[i])
            acc = run = 0.0
    out.append(pts[-1])
    return out


def _arc(poly):
    P = np.asarray(poly, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def _at(poly, s):
    """Point and unit tangent at arc length s."""
    P = np.asarray(poly, dtype=float)
    cum = _arc(P)
    s = min(max(s, 0.0), cum[-1])
    i = int(min(max(np.searchsorted(cum, s, side="right") - 1, 0), len(P) - 2))
    seg = cum[i + 1] - cum[i]
    t = (s - cum[i]) / seg if seg > 1e-12 else 0.0
    pt = P[i] + (P[i + 1] - P[i]) * t
    d = P[min(i + 2, len(P) - 1)] - P[max(i - 1, 0)]
    d = d / max(np.linalg.norm(d), 1e-12)
    return (float(pt[0]), float(pt[1])), (float(d[0]), float(d[1]))


def _s_of(poly, pt):
    P = np.asarray(poly, dtype=float)
    return float(_arc(P)[int(np.argmin(np.linalg.norm(P - np.asarray(pt), axis=1)))])


def _s_of_z(poly, z):
    """Arc length where a rising member reaches height z (ignores the flat toe)."""
    P = np.asarray(poly, dtype=float)
    d = np.abs(P[:, 1] - z) + np.where(np.arange(len(P)) < 4, 9.0, 0.0)
    return float(_arc(P)[int(np.argmin(d))])


def _cut(poly, s0, s1):
    cum = _arc(poly)
    return [_at(poly, s0)[0]] + [tuple(poly[i]) for i in range(len(poly)) if s0 + 1e-4 < cum[i] < s1 - 1e-4] + [_at(poly, s1)[0]]


def _near(pt, poly):
    """Nearest point on a polyline -> (distance, point)."""
    P = np.asarray(poly, dtype=float)
    pt = np.asarray(pt, dtype=float)
    a, ab = P[:-1], P[1:] - P[:-1]
    t = np.clip(((pt - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    q = a + ab * t[:, None]
    dd = np.linalg.norm(q - pt, axis=1)
    i = int(np.argmin(dd))
    return float(dd[i]), q[i]


def _touch(c, gaps_polys, it=10):
    """Slide a scroll-eye centre until it sits at the given gap from each member polyline (so the
    scroll kisses the members it is cast against). Returns the centre and the contact points."""
    c = np.asarray(c, dtype=float)
    for _ in range(it):
        for gap, poly in gaps_polys:
            d, q = _near(c, poly)
            c = q + (c - q) / max(d, 1e-9) * gap
    return (float(c[0]), float(c[1])), [_near(c, poly)[1] for _, poly in gaps_polys]


def _volute(c, r_out, a0, turns, sense, r_in=0.28, per_turn=17):
    """Spiral from radius r_out (at angle a0) winding inwards over `turns`; sense +1 = CCW."""
    n = max(6, int(round(turns * per_turn)))
    pts = []
    for i in range(n + 1):
        t = i / n
        a = a0 + sense * 2 * pi * turns * t
        r = r_out * (1.0 - (1.0 - r_in) * t ** 0.9)
        pts.append((c[0] + r * cos(a), c[1] + r * sin(a)))
    return pts


def _curl(pts, r, turns, sense, r_in=0.28, per_turn=17):
    """Continue a path with a tangent volute. Returns (path, eye centre, inner end)."""
    (y0, z0), (y1, z1) = pts[-2], pts[-1]
    d = hypot(y1 - y0, z1 - z0)
    ty, tz = (y1 - y0) / d, (z1 - z0) / d
    ny, nz = (-tz, ty) if sense > 0 else (tz, -ty)
    c = (y1 + ny * r, z1 + nz * r)
    v = _volute(c, r, atan2(-nz, -ny), turns, sense, r_in, per_turn)
    return pts + v[1:], c, v[-1]


def _scroll2(A, ra, B, rb, kind="S", side=1, turns=1.1, r_in=0.3):
    """S- or C-scroll between two eyes: volute - common tangent - volute. Returns (path, end A, end B)."""
    d = hypot(B[0] - A[0], B[1] - A[1])
    ux, uz = (B[0] - A[0]) / d, (B[1] - A[1]) / d
    c = (ra + rb) / d if kind == "S" else (ra - rb) / d
    al = acos(max(-1.0, min(1.0, c))) * side
    nx, nz = ux * cos(al) - uz * sin(al), ux * sin(al) + uz * cos(al)
    PA = (A[0] + ra * nx, A[1] + ra * nz)
    PB = (B[0] - rb * nx, B[1] - rb * nz) if kind == "S" else (B[0] + rb * nx, B[1] + rb * nz)
    ln = hypot(PB[0] - PA[0], PB[1] - PA[1])
    dx, dz = (PB[0] - PA[0]) / ln, (PB[1] - PA[1]) / ln
    aA, aB = atan2(PA[1] - A[1], PA[0] - A[0]), atan2(PB[1] - B[1], PB[0] - B[0])
    sB = 1 if (-sin(aB)) * dx + cos(aB) * dz > 0 else -1
    sA = 1 if (-sin(aA)) * (-dx) + cos(aA) * (-dz) > 0 else -1
    va, vb = _volute(A, ra, aA, turns, sA, r_in), _volute(B, rb, aB, turns, sB, r_in)
    k = max(1, int(ln / 0.05))
    mid = [(PA[0] + (PB[0] - PA[0]) * i / (k + 1), PA[1] + (PB[1] - PA[1]) * i / (k + 1)) for i in range(1, k + 1)]
    return va[::-1] + mid + vb, va[-1], vb[-1], PA, PB


def _shade(h, f):
    """Scale a '#rrggbb' colour (f < 1 darker)."""
    return "#" + "".join(f"{max(0, min(255, int(round(int(h[i:i + 2], 16) * f)))):02x}" for i in (1, 3, 5))


def _moulded(w, t, g=0.003):
    """Cast section: flat web with a raised bead along both edges on both faces (a = in-plane, b = along bench)."""
    hw, ht = w / 2, t / 2
    e, bw = min(0.003, t * 0.22), w * 0.17
    return [(ht, -hw + e), (ht, -hw + bw), (ht - g, -hw + bw + g), (ht - g, hw - bw - g), (ht, hw - bw), (ht, hw - e),
            (ht - e, hw), (-ht + e, hw), (-ht, hw - e), (-ht, hw - bw), (-ht + g, hw - bw - g), (-ht + g, -hw + bw + g),
            (-ht, -hw + bw), (-ht, -hw + e), (-ht + e, -hw), (ht - e, -hw)]


def _handrail(w, t):
    """Arm section: crowned (half-round) on the outer / upper side, flat beneath."""
    hw, lo, hi = w / 2, -t * 0.43, t * 0.57
    return [(hi * 0.5, -hw * 0.9), (hi * 0.85, -hw * 0.5), (hi, 0.0), (hi * 0.85, hw * 0.5), (hi * 0.5, hw * 0.9), (hi * 0.12, hw),
            (lo * 0.6, hw), (lo, hw * 0.9), (lo, -hw * 0.9), (lo * 0.6, -hw), (hi * 0.12, -hw)]


def _bar(name, pts, x, prof, scale_fn=None, ang=7.5, dmax=0.055, sm=55.0):
    """Sweep a cast section along a (y, z) polyline in the plane x = const; profile b axis = +X."""
    o = sweep(name, [(x, y, z) for y, z in _thin(pts, ang, dmax)], prof, smooth_path=False, up=(1, 0, 0), scale_fn=scale_fn)
    smooth(o, sm)
    return o


def _slat(name, Ls, w, t, supports, sag, crook, e_top, e_bot):
    """One sawn board along X: eased top arrises, saw-cut ends with a small chamfer, sagging between
    its supports and slightly crooked. Extra cross-sections at the bolts carry the stain masks."""
    pri = [-Ls / 2, -Ls / 2 + 0.004, Ls / 2 - 0.004, Ls / 2]
    for xb in supports:
        pri += [x for x in (xb - 0.024, xb, xb + 0.024) if -Ls / 2 + 0.008 < x < Ls / 2 - 0.008]
    xs = list(pri)
    for x in np.linspace(-Ls / 2, Ls / 2, max(4, int(round(Ls / 0.15)) + 1)):
        if all(abs(x - q) > 0.04 for q in pri):
            xs.append(float(x))
    xs = sorted(xs)
    sup = sorted(supports)

    def dz(x):
        for i in range(len(sup) - 1):
            if sup[i] <= x <= sup[i + 1]:
                return -sag[i] * sin(pi * (x - sup[i]) / (sup[i + 1] - sup[i])) ** 2
        return 0.0

    hw, ht, et, eb = w / 2, t / 2, e_top, e_bot
    prof = [(hw, -ht + eb), (hw, ht - et), (hw - et, ht), (0.0, ht), (-hw + et, ht),
            (-hw, ht - et), (-hw, -ht + eb), (-hw + eb, -ht), (hw - eb, -ht)]
    path = [(x, crook * ((2 * x / Ls) ** 2 - 0.4), dz(x)) for x in xs]
    o = sweep(name, path, prof, smooth_path=False, up=(0, 0, 1),
              scale_fn=lambda u: 0.94 if (u < 1e-5 or u > 1 - 1e-5) else 1.0)
    smooth(o, 50)
    return o


def _dp(pts, tol):
    """Far level: the fewest points of a polyline that keep within `tol` of it (Douglas-Peucker)."""
    if len(pts) < 3:
        return [tuple(q) for q in pts]
    P = np.asarray(pts, dtype=float)
    d = P[-1] - P[0]
    n = float(np.hypot(d[0], d[1]))
    if n < tol:
        dist = np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])
    else:
        dist = np.abs((P[:, 0] - P[0, 0]) * d[1] - (P[:, 1] - P[0, 1]) * d[0]) / n
    i = int(np.argmax(dist))
    if dist[i] <= tol or i in (0, len(pts) - 1):
        return [tuple(pts[0]), tuple(pts[-1])]
    return _dp(pts[:i + 1], tol)[:-1] + _dp(pts[i:], tol)


def _fbar(name, pts, x, w, t, tol=0.016, flat=False):
    """Far level: a cast member as a square-edged bar (flat=True: a three-sided wedge) through the few points
    that keep its line, in the plane x = const. w = width along the bench, t = thickness in the plane."""
    prof = [(-t / 2, -w / 2), (t / 2, 0.0), (-t / 2, w / 2)] if flat else [(t / 2, w / 2), (-t / 2, w / 2), (-t / 2, -w / 2), (t / 2, -w / 2)]
    return sweep(name, [(x, y, z) for y, z in _dp(pts, tol)], prof, smooth_path=False, up=(1, 0, 0))


def _fprism(name, q, x0, x1):
    """Far level: a prism along X over the (y, z) polygon q (counter-clockwise seen from +X)."""
    n = len(q)
    o = mesh_object(name, [(x, y, z) for x in (x0, x1) for y, z in q],
                    [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)])
    uv_box(o)
    return o


def _feye(name, pts, x, w, pad, n=6):
    """Far level: a scroll eye / coil as an n-sided block over the extent of its spiral."""
    ys, zs = [q[0] for q in pts], [q[1] for q in pts]
    c = ((min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    r = (max(max(ys) - min(ys), max(zs) - min(zs)) / 2 + pad) * (1.06 if n > 4 else 1.2)
    return _fprism(name, [(c[0] + r * cos(2 * pi * (k + 0.5) / n), c[1] + r * sin(2 * pi * (k + 0.5) / n)) for k in range(n)], x - w / 2, x + w / 2)


def build(p, rng):
    L, style, W = p.length, p.style, p.wear
    # lod 2 (META "lod": "far", docs/LOD.md) is modelled here: the castings as square-edged bars through a few
    # points of their lines with the scroll eyes as blocks, neighbouring boards of one finish as one plank;
    # fixings, tie rods, bearers and pad feet (all under half a pixel from 44 m) are left out
    far = lod() == 2
    iron = mat("iron_painted", p.paint)
    brass = mat("brass")
    gilt = brass if p.gilt else iron
    bolt_mat = brass if p.bolts == "brass" else iron
    wood_lib = {"oak": "wood_oak", "painted": "wood_painted", "weathered": "wood_weathered"}[p.wood]
    wood = mat(wood_lib, p.wood_tone)
    wood2 = mat(wood_lib, _shade(p.wood_tone, 0.8))      # a second, darker batch of boards
    Lf = L - 0.03                                         # over the frame faces; nuts and pad feet make up `length`
    end_mat = wood if p.wood == "painted" else mat("wood_endgrain", p.wood_tone)
    nfr = int(p.frames) or (3 if L >= 2.0 else 2)

    wm, tm = 0.044, 0.021            # main cast member: width along the bench, thickness in the frame plane
    wa, ta = 0.040, 0.016            # arm support (the arm itself is a wider crowned hand rail)
    ws, ts = 0.034, 0.013            # scroll infill
    wb = 0.036                       # inboard bearer flange that carries the slats
    zt = 0.0105                      # centre height of the toe where it lies on the pad foot
    MOULD, STRAP = _moulded(wm, tm), rounded_rect(tm, wm, 0.0035, 1)
    ARM, ARMFLAT = _handrail(0.052, 0.021), rounded_rect(ta, wa, 0.003, 1)
    SCR = rounded_rect(ts, ws, 0.0028, 1)

    # ---- the two principal lines of the casting, in (y, z): y = depth (front is -y), z = height
    # A: front pad foot -> cabriole front leg -> knee roll -> seat rail -> merges up into the back standard
    A = _cr([(-0.315, zt), (-0.290, zt), (-0.262, 0.030), (-0.240, 0.085), (-0.232, 0.17), (-0.245, 0.27), (-0.268, 0.345),
             (-0.277, 0.385), (-0.264, 0.411), (-0.228, 0.419), (-0.10, 0.410), (0.02, 0.401), (0.12, 0.399), (0.168, 0.406),
             (0.198, 0.432), (0.2145, 0.468), (0.2265, 0.505)])
    # B: back pad foot -> sabre back leg -> reclined back standard
    B = _cr([(0.325, zt), (0.300, zt), (0.268, 0.032), (0.238, 0.10), (0.212, 0.20), (0.200, 0.30), (0.200, 0.385),
             (0.213, 0.45), (0.236, 0.54), (0.256, 0.66), (0.276, 0.78), (0.292, 0.87)])
    # where the slats sit: seat on A (behind the knee roll), back on B
    sa0, sa1 = _s_of(A, (-0.246, 0.417)), _s_of(A, (0.152, 0.402))
    sb0, sb1 = _s_of_z(B, 0.488), _s_of_z(B, 0.866)
    bearer = _cut(A, sa0 - 0.012, _arc(A)[-1]) + _cut(B, _s_of_z(B, 0.512), sb1 + 0.012)

    def by(z):
        return _at(B, _s_of_z(B, z))[0]

    iron_objs, gilt_objs = [], []

    def xcyl(name, x, y, z, r, depth, verts=10):
        o = cylinder(name, r, depth, loc=(x, y, z), rot=(0, 90, 0), verts=verts, bevel=min(0.0015, r * 0.2), segments=1)
        iron_objs.append(o)
        return o

    ros_base = lathe("rosette", [(0.019, 0.0), (0.019, 0.003), (0.012, 0.0062), (0.006, 0.0088), (0.0, 0.0115)],
                     segments=10, smooth_angle=60)

    def petal(co):
        r = hypot(co.x, co.y)
        if r > 0.012:
            k = 1.0 + 0.14 * cos(5 * atan2(co.y, co.x))
            return Vector((co.x * k, co.y * k, co.z))
        return co

    transform_verts(ros_base, petal)
    ros_m = []                       # rosette instance matrices
    collars = []                     # (x, y, z) of cast-on collars where scrolls kiss the members
    snubs = []                       # (x, y, z) snub ends of scrolls
    nut_m = []                       # hex nut instance matrices

    def eye(x, c, r=0.0165, w=wa):
        """Volute eye: a hub the spiral dies into, with a rosette on both faces."""
        xcyl("eye_hub", x, c[0], c[1], r, w, verts=14)
        for s in (-1, 1):
            ros_m.append(trs((x + s * (w / 2 - 0.0005), c[0], c[1]), (0, 90 * s, 0)))

    def frame(xf, inb, kind):
        end = kind == "end"
        ornate = end and style != "plain"
        main = MOULD if style != "plain" else STRAP
        if not end:
            main = rounded_rect(tm, 0.040, 0.0035, 1)
        fo = []
        if far:
            rng.u(0, 60), rng.u(0, 60)                    # the turn of the two holding-down nuts, drawn as at lod 0
            wf = wm if end else 0.040
            knee = (-0.270, 0.392)
            p0, p1 = _at(A, _s_of_z(A, 0.20))[0], _at(B, _s_of_z(B, 0.215))[0]

            def boss(c, w):
                # the gilt rosettes of an eye: one three-sided boss through the casting
                gilt_objs.append(_fprism("rosette", [(c[0] + 0.03 * cos(a), c[1] + 0.03 * sin(a)) for a in (pi / 2, 7 * pi / 6, 11 * pi / 6)],
                                         xf - w / 2 - 0.009, xf + w / 2 + 0.009))

            def infill(ca, ra, cb, rb, kind, side):
                # a scroll of the infill: its two eyes as blocks and the stem between them
                PA, PB = _scroll2(ca, ra, cb, rb, kind, side=side)[3:]
                fo.append(_fbar("scroll", [PA, PB], xf, ws, ts, flat=True))
                for c, r in ((ca, ra), (cb, rb)):
                    fo.append(_fprism("scroll_eye", [(c[0] + r * cos(a), c[1] + r * sin(a)) for a in (0, pi / 2, pi, 3 * pi / 2)],
                                      xf - ws / 2, xf + ws / 2))

            # the bars run out over the pad feet
            fo.append(_fbar("frame_seat", [(A[0][0] - 0.036, zt)] + A, xf, wf, tm))
            Bp = [(B[0][0] + 0.036, zt)] + B
            if not end:
                fo.append(_fbar("frame_back", Bp, xf, wf, tm))
                fo.append(_fbar("stretcher", [p0, p1], xf, 0.030, ts, flat=True))
                iron_objs.extend(fo)
                return
            g = 0.052 + ts / 2 + tm / 2 - 0.002
            under = None
            if style == "scroll":
                fo.append(_fbar("frame_back", Bp, xf, wf, tm))
                vol, top_c, _ = _curl(list(B), 0.034, 1.2, -1)
                fo.append(_feye("back_eye", vol[len(B):], xf, 0.038, tm * 0.45, 4))
                boss(top_c, 0.038)
                arm_c = _cr([(by(0.676)[0] + 0.004, 0.676), (0.14, 0.664), (0.0, 0.646), (-0.13, 0.647), (-0.235, 0.664)])
                vol, ec, _ = _curl(arm_c, 0.052, 1.35, +1, per_turn=20)
                vol = vol[len(arm_c):]
                fo.append(_fbar("arm", arm_c, xf, 0.052, 0.021))
                fo.append(_feye("arm_eye", vol, xf, 0.042, 0.009))
                boss(ec, wa)
                boss((by(0.676)[0], 0.676), wm)
                hit = vol[int(round(len(vol) * 140.0 / (1.35 * 360.0)))]
                fo.append(_fbar("arm_support", _cr([knee, (-0.292, 0.44), (-0.297, 0.50), (-0.288, 0.552), hit]), xf, wa, ta))
                under = (arm_c, 0.009, 0.046)
            elif style == "serpent":
                # the finial rides on the bar; coil, body, head and the support of the serpent arm
                fo.append(_fbar("frame_back", Bp + [(0.2955, 0.89), (0.299, 0.905), (0.315, 0.972)], xf, wf, tm))
                rs_, rb_ = 0.047, 0.0155
                tc, _ = _touch((0.165, 0.600), [(rs_ + rb_ * 0.5 + tm / 2 - 0.003, B)])
                body_c = _cr([(tc[0], tc[1] + rs_), (0.07, tc[1] + rs_ + 0.003), (-0.03, 0.664), (-0.135, 0.645), (-0.232, 0.655),
                              (-0.292, 0.686), (-0.322, 0.697)])
                fo.append(_feye("serpent_coil", _volute(tc, rs_, pi / 2, 1.2, -1, r_in=0.3), xf, 0.03, rb_ * 0.6))
                fo.append(_fbar("serpent_body", body_c, xf, 0.041, 0.031))
                (hy, hz), (ty, tz) = body_c[-1], _at(body_c, _arc(body_c)[-1])[1]
                fo.append(_fbar("serpent_head", [(hy - ty * 0.016, hz - tz * 0.016), (hy + ty * 0.1, hz + tz * 0.1)], xf, 0.06, 0.038))
                near_s = _near((-0.262, 0.64), body_c)[1]
                fo.append(_fbar("arm_support", _cr([knee, (-0.289, 0.45), (-0.283, 0.53), (-0.268, 0.60), (float(near_s[0]), float(near_s[1]))]),
                                xf, wa, ta))
                under = (body_c, rb_, 0.038)
            else:
                fo.append(_fbar("frame_back", Bp + [(0.2955, 0.89), (0.3005, 0.917)], xf, wf, tm))
                fo.append(_fbar("arm", _cr([(by(0.657)[0] + 0.004, 0.657), (0.10, 0.652), (-0.14, 0.650), (-0.245, 0.642), (-0.287, 0.605),
                                            (-0.296, 0.54), (-0.288, 0.45), knee]), xf, wa, ta))
                fo.append(_fbar("stretcher", [p0, p1], xf, ws, ts, flat=True))
            if under:
                gs = under[2] + ts / 2 + tm / 2 - 0.002
                ca, _ = _touch((0.13, 0.48), [(gs, A), (gs, B)])
                cb, _ = _touch((-0.15, 0.59), [(0.040 + ts / 2 + under[1] - 0.002, under[0])])
                infill(ca, under[2], cb, 0.040, "S", 1)
                ca, _ = _touch((-0.19, 0.35), [(g, A[:len(A) // 2]), (g, A[len(A) // 2:])])
                cb, _ = _touch((0.13, 0.34), [(g, B), (g, A[len(A) // 2:])])
                infill(ca, 0.052, cb, 0.052, "C", -1)
            iron_objs.extend(fo)
            return
        # --- principal lines
        tk_ = _s_of(A, (-0.277, 0.385)) / _arc(A)[-1]
        fo.append(_bar("frame_seat", A, xf, main, ang=9.0, dmax=0.07, scale_fn=lambda t: (
            0.9 + 0.1 * smoothstep(0.04, 0.26, t) + 0.2 * exp(-((t - tk_) / 0.075) ** 2) - 0.1 * smoothstep(0.9, 1.0, t))))
        Bp = list(B)
        top_c = None
        if end and style == "scroll":
            Bp, top_c, top_e = _curl(Bp, 0.034, 1.2, -1)
        elif end:
            Bp = Bp + [(0.2955, 0.89), (0.299, 0.905)]
        else:
            Bp = _cut(B, 0.0, sb1 + 0.02)
        tj_ = _s_of_z(B, 0.37) / _arc(Bp)[-1]
        tip_ = 0.25 if (end and style == "scroll") else 0.0
        fo.append(_bar("frame_back", Bp, xf, main, ang=9.0, dmax=0.07, scale_fn=lambda t: (
            0.9 + 0.1 * smoothstep(0.04, 0.24, t) + 0.16 * exp(-((t - tj_) / 0.06) ** 2) - tip_ * smoothstep(0.86, 1.0, t))))
        if end and style == "scroll":
            eye(xf, top_c, 0.0125, 0.038)
        elif end and style == "serpent":
            # turned bud finial on the back standard
            ty, tz = Bp[-1][0] - Bp[-2][0], Bp[-1][1] - Bp[-2][1]
            fo.append(lathe("finial", [(0.0095, -0.004), (0.015, 0.003), (0.0115, 0.010), (0.0125, 0.016), (0.0195, 0.030), (0.0185, 0.042),
                                       (0.011, 0.055), (0.0055, 0.061), (0.006, 0.066), (0.0, 0.070)], segments=14,
                            loc=(xf, Bp[-1][0], Bp[-1][1]), rot=(degrees(atan2(-ty, tz)), 0, 0)))
        else:
            fo.append(cylinder("back_tip", tm * 0.72, wm if end else 0.040, loc=(xf, Bp[-1][0], Bp[-1][1]), rot=(0, 90, 0), verts=12,
                               bevel=0.002, segments=1))
        # --- pad feet with holding-down bolts
        for nm, (fy, _) in (("foot_f", A[0]), ("foot_b", B[0])):
            fo.append(lathe(nm, [(0.036, 0.0), (0.036, 0.012), (0.031, 0.0195), (0.0, 0.0195)], segments=14, loc=(xf, fy, 0.0)))
            sgn = -1 if fy < 0 else 1
            nut_m.append(trs((xf, fy + sgn * 0.004, 0.0195), (0, 0, rng.u(0, 60))))
        if not end:
            # centre frame: plain stretcher between the legs
            p0, p1 = _at(A, _s_of_z(A, 0.20))[0], _at(B, _s_of_z(B, 0.215))[0]
            fo.append(_bar("stretcher", [p0, p1], xf, rounded_rect(ts, 0.030, 0.0028, 1)))
            iron_objs.extend(fo)
            return
        # --- inboard bearer flange for the slats
        fo.append(_bar("bearer", bearer, xf + inb * (wm / 2 + wb / 2 - 0.002), rounded_rect(tm, wb, 0.003, 1)))

        knee = (-0.270, 0.392)

        def under_arm(arm_poly, arm_half, ra, spurs):
            """S-scroll under the arm: lower eye in the seat / back corner, upper eye against the arm;
            leaf spurs branch off its stem and curl into the spandrels it leaves."""
            rb = 0.040
            ca, qa = _touch((0.13, 0.48), [(ra + ts / 2 + tm / 2 - 0.002, A), (ra + ts / 2 + tm / 2 - 0.002, B)])
            cb, qb = _touch((-0.15, 0.59), [(rb + ts / 2 + arm_half - 0.002, arm_poly)])
            s_path, ea, eb, PA, PB = _scroll2(ca, ra, cb, rb, "S", side=1)
            fo.append(_bar("scroll_arm", s_path, xf, SCR))
            snubs.extend([(xf, *ea), (xf, *eb)])
            collars.extend([(xf, *q) for q in qa + qb])
            ln = hypot(PB[0] - PA[0], PB[1] - PA[1])
            d = ((PB[0] - PA[0]) / ln, (PB[1] - PA[1]) / ln)
            nu = (-d[1], d[0]) if d[0] < 0 else (d[1], -d[0])           # stem normal on the upper side
            for f, hs, ns, rr, reach in ((0.60, -1, 1, 0.031, 0.105), (0.36, 1, -1, 0.033, 0.10))[2 - spurs:]:
                M = (PA[0] + (PB[0] - PA[0]) * f, PA[1] + (PB[1] - PA[1]) * f)
                h, n_ = (d[0] * hs, d[1] * hs), (nu[0] * ns, nu[1] * ns)
                sp = _cr([(M[0] + h[0] * a + n_[0] * b, M[1] + h[1] * a + n_[1] * b)
                          for a, b in ((0.0, 0.0), (reach * 0.35, 0.003), (reach * 0.7, 0.014), (reach, 0.034))])
                sp, sc_, se_ = _curl(sp, rr, 1.0, 1 if h[0] * n_[1] - h[1] * n_[0] > 0 else -1, r_in=0.3)
                fo.append(_bar("scroll_spur", sp, xf, rounded_rect(ts * 0.85, ws - 0.004, 0.0025, 1),
                               scale_fn=lambda t: 1.0 - 0.25 * smoothstep(0.35, 1.0, t)))
                snubs.append((xf, *se_))

        def yoke():
            """Lyre yoke under the seat: two eyes in the knee corners joined along the bottom, turned drop."""
            ra = rb = 0.052
            g = ra + ts / 2 + tm / 2 - 0.002
            ca, qa = _touch((-0.19, 0.35), [(g, A[:len(A) // 2]), (g, A[len(A) // 2:])])
            cb, qb = _touch((0.13, 0.34), [(g, B), (g, A[len(A) // 2:])])
            c_path, ea, eb, _, _ = _scroll2(ca, ra, cb, rb, "C", side=-1)
            fo.append(_bar("scroll_yoke", c_path, xf, SCR))
            snubs.extend([(xf, *ea), (xf, *eb)])
            collars.extend([(xf, *q) for q in qa + qb])
            my, mz = (ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2 - ra - ts / 2
            fo.append(lathe("drop", [(0.0, -0.046), (0.004, -0.043), (0.0045, -0.037), (0.011, -0.027), (0.012, -0.019), (0.006, -0.010),
                                     (0.0085, -0.005), (0.0085, 0.001)], segments=10, loc=(xf, my, mz)))

        if style == "scroll":
            arm_c = _cr([(by(0.676)[0] + 0.004, 0.676), (0.14, 0.664), (0.0, 0.646), (-0.13, 0.647), (-0.235, 0.664)])
            arm, ec, ee = _curl(arm_c, 0.052, 1.35, +1, per_turn=20)
            fo.append(_bar("arm", arm, xf, ARM, ang=6.5, scale_fn=lambda t: 1.0 - 0.3 * smoothstep(0.68, 1.0, t)))
            eye(xf, ec)
            # arm support springs from the knee and dies into the underside of the volute
            vol = arm[len(arm_c) - 1:]
            hit = vol[int(round((len(vol) - 1) * 140.0 / (1.35 * 360.0)))]
            sup = _cr([knee, (-0.292, 0.44), (-0.297, 0.50), (-0.288, 0.552), hit])
            fo.append(_bar("arm_support", sup, xf, ARMFLAT))
            ros_m.extend(trs((xf + s * (wm / 2 - 0.0005), by(0.676)[0], 0.676), (0, 90 * s, 0)) for s in (-1, 1))
            under_arm(arm_c, 0.009, 0.046, 2)
            yoke()
        elif style == "serpent":
            # the arm is a snake: tail coiled against the back standard, body along the arm, head over the knee
            rs, rb_ = 0.047, 0.0155
            tc, qt = _touch((0.165, 0.600), [(rs + rb_ * 0.5 + tm / 2 - 0.003, B)])
            coil = _volute(tc, rs, pi / 2, 1.2, -1, r_in=0.3)
            body_c = _cr([(tc[0], tc[1] + rs), (0.07, tc[1] + rs + 0.003), (-0.03, 0.664), (-0.135, 0.645), (-0.232, 0.655),
                          (-0.292, 0.686), (-0.322, 0.697)])
            body = coil[::-1] + body_c[1:]

            def taper(t):
                return (0.3 + 0.7 * smoothstep(0.0, 0.32, t)) * (1.0 - 0.22 * smoothstep(0.88, 1.0, t))

            fo.append(_bar("serpent_body", body, xf, [(rb_ * cos(2 * pi * k / 10), 0.0205 * sin(2 * pi * k / 10)) for k in range(10)],
                           scale_fn=taper, ang=6.5, sm=180))
            collars.extend([(xf, *q) for q in qt])
            hy, hz = body[-1]
            ty, tz = _at(body, _arc(body)[-1])[1]
            phi = atan2(-ty, tz)
            head = lathe("serpent_head", [(0.0, -0.016), (0.012, -0.011), (0.015, 0.0), (0.0215, 0.019), (0.027, 0.040), (0.0245, 0.062),
                                          (0.016, 0.083), (0.010, 0.094), (0.0, 0.100)], segments=14, smooth_angle=180)
            place(head, loc=(xf, hy, hz), rot=(degrees(phi), 0, 0), scale=(1.12, 0.7, 1.0))
            fo.append(head)
            upy, upz = cos(phi), sin(phi)
            for s in (-1, 1):
                e = sphere("serpent_eye", 0.0056, loc=(xf + s * 0.0205, hy + ty * 0.055 + upy * 0.011, hz + tz * 0.055 + upz * 0.011),
                           segments=8, rings=5)
                gilt_objs.append(e)
            near_s = _near((-0.262, 0.64), body)[1]
            sup = _cr([knee, (-0.289, 0.45), (-0.283, 0.53), (-0.268, 0.60), (float(near_s[0]), float(near_s[1]))])
            fo.append(_bar("arm_support", sup, xf, ARMFLAT))
            fo.append(_bar("arm_cup", [(near_s[0] - 0.032, near_s[1] - 0.013), (near_s[0], near_s[1] - 0.017),
                                       (near_s[0] + 0.032, near_s[1] - 0.014)], xf, rounded_rect(0.008, wa, 0.002, 1)))
            under_arm(body_c, rb_, 0.038, 1)
            yoke()
        else:
            # plain strap: arm and support are one bent loop; flat stretcher between the legs
            loop = _cr([(by(0.657)[0] + 0.004, 0.657), (0.10, 0.652), (-0.14, 0.650), (-0.245, 0.642), (-0.287, 0.605),
                        (-0.296, 0.54), (-0.288, 0.45), knee])
            fo.append(_bar("arm", loop, xf, ARMFLAT))
            p0, p1 = _at(A, _s_of_z(A, 0.20))[0], _at(B, _s_of_z(B, 0.215))[0]
            fo.append(_bar("stretcher", [p0, p1], xf, SCR))
            collars.extend([(xf, *p0), (xf, *p1), (xf, by(0.657)[0], 0.657)])
        iron_objs.extend(fo)

    xe = Lf / 2 - wm / 2
    frame(-xe, +1, "end")
    frame(xe, -1, "end")
    if nfr == 3:
        frame(0.0, 0, "mid")

    # ---- cast-on collars and snub ends, stamped as one mesh each
    if collars:
        cb_ = cylinder("collar_b", 0.0085, ws + 0.007, rot=(0, 90, 0), verts=8)
        apply_transform(cb_)
        iron_objs.append(instances_to_mesh("collars", cb_, [trs(c) for c in collars], keep_base=False))
    if snubs:
        sb_ = cylinder("snub_b", 0.0098, ws + 0.003, rot=(0, 90, 0), verts=10)
        apply_transform(sb_)
        iron_objs.append(instances_to_mesh("snubs", sb_, [trs(c) for c in snubs], keep_base=False))
    if ros_m:
        gilt_objs.append(instances_to_mesh("rosettes", ros_base, ros_m, keep_base=False))
    else:
        delete(ros_base)

    # ---- tie rods through the legs, hex nuts and washers on the outer faces
    nut = lathe("nut_b", [(0.0105, 0.0), (0.0105, 0.0062), (0.0086, 0.0078), (0.0, 0.0078)], segments=6, smooth_angle=30)
    for nm, (ry, rz) in (("tie_front", _at(A, _s_of_z(A, 0.352))[0]), ("tie_back", _at(B, _s_of_z(B, 0.405))[0])):
        if far:
            rng.u(0, 60), rng.u(0, 60)                        # the turn of the two nuts, drawn as at lod 0
            iron_objs.append(_fprism(nm, [(ry + 0.009 * cos(a), rz + 0.009 * sin(a)) for a in (pi / 2, 7 * pi / 6, 11 * pi / 6)],
                                     -L / 2 + 0.001, L / 2 - 0.001))
            continue
        iron_objs.append(cylinder(nm, 0.0062, L - 0.002, loc=(0, ry, rz), rot=(0, 90, 0), verts=8))
        for s in (-1, 1):
            nut_m.append(trs((s * (Lf / 2 + 0.0055), ry, rz), (rng.u(0, 60), 90 * s, 0)))
            iron_objs.append(cylinder("washer", 0.0165, 0.011, loc=(s * (Lf / 2 + 0.0005), ry, rz), rot=(0, 90, 0), verts=12))

    # ---- slats
    Ls = Lf - 2 * wm - 0.007
    xb = Lf / 2 - wm - 0.021                                 # bolt line on the bearers
    supports = [-xb, xb] if nfr == 2 else [-xb, 0.0, xb]
    strap = nfr == 2 and L > 1.7                             # light centre strap under long two-frame benches
    bolt_x = supports if (nfr == 3 or not strap) else [-xb, 0.0, xb]
    n_seat, n_back = int(p.slats), int(p.back_slats)
    gap = 0.012
    specs = []
    pitch = (sa1 - sa0) / n_seat
    for i in range(n_seat):
        specs.append((A, sa0 + (i + 0.5) * pitch, pitch - gap, 0.029, False, i, n_seat))
    pitch = (sb1 - sb0) / n_back
    for i in range(n_back):
        specs.append((B, sb0 + (i + 0.5) * pitch, pitch - gap, 0.026, True, i, n_back))
    rs = rng.fork("slats")
    seats = max(1, int(round(Ls / 0.58)))
    seat_x = [(-0.5 + (i + 0.5) / seats) * Ls + rs.u(-0.05, 0.05) for i in range(seats)]
    replaced = rs.pick(range(len(specs))) if W >= 0.6 else -1
    new_wood = mat("wood_oak", "#c9a26e")
    dome = lathe("dome_b", [(0.0092, 0.0), (0.0088, 0.002), (0.0052, 0.0046), (0.0, 0.0056)], segments=7, smooth_angle=180)
    dome_m = []
    fsl = []                                                 # far level: the boards as (back?, finish, section numbers)
    wn1, wn2 = noise_mask(0.06, seed=p.seed + 11, lo=0.25, hi=0.8), noise_mask(0.22, seed=p.seed + 12, lo=0.2, hi=0.8)
    for k, (curve, s, w, t, is_back, i, n) in enumerate(specs):
        r = rs.fork(f"slat{k}")
        fresh = k == replaced
        w = w - r.u(0.0, 0.003)
        t = t + r.u(-0.001, 0.001) + (0.0015 if fresh else 0.0)
        (cy, cz), (ty, tz) = _at(curve, s + r.u(-0.0015, 0.0015))
        ny, nz = -tz, ty
        cy, cz = cy + ny * (tm / 2 + t / 2), cz + nz * (tm / 2 + t / 2)
        th = degrees(atan2(tz, ty))
        span = (supports[1] - supports[0])
        amp = (0.0006 + 0.0042 * W) * (0.25 if fresh else 1.0) * (span / 0.75) ** 2
        sag = [amp * r.u(-0.15, 1.0) * (0.6 if is_back else 1.0) for _ in range(len(supports) - 1)]
        first = (not is_back) and i == 0
        Lk, crook = Ls + r.u(-0.003, 0.0), r.u(-1, 1) * (0.0004 + 0.0012 * W)
        o = None if far else _slat(f"slat_{'b' if is_back else 's'}{i}", Lk, w, t, supports, sag,
                                   crook=crook, e_top=0.0075 if first else 0.0045, e_bot=0.0025)
        lift = r.u(-1, 1) * (0.0004 + 0.0014 * W)
        cy, cz = cy + ny * lift + ty * r.u(-0.001, 0.001), cz + nz * lift + tz * r.u(-0.001, 0.001)
        th += r.u(-1, 1) * (0.4 + 1.4 * W)
        x0, uo = r.u(-0.0015, 0.0015), (r.u(0, 5), r.u(0, 5))
        finish = new_wood if fresh else (wood2 if r.chance(0.4) else wood)
        if far:
            fsl.append((is_back, finish, (cy - ty * w / 2, cz - tz * w / 2), (cy + ty * w / 2, cz + tz * w / 2),
                        (ny * t / 2, nz * t / 2)))
            continue
        place(o, loc=(x0, cy, cz), rot=(th, 0, 0))
        uv_transform(o, offset=uo)
        assign(o, finish)
        assign_faces(o, mat("wood_endgrain", "#c9a26e") if fresh else end_mat, lambda c, nrm: abs(nrm.x) > 0.9)

        # art-directed masks: rubbed where people sit and lean, hands on the top rail, stains under the bolts
        top_edge = is_back and i == n - 1

        def wear_fn(co, nr, ny=ny, nz=nz, is_back=is_back, i=i, n=n, first=first, top_edge=top_edge, cz=cz):
            top = np.clip((nr[:, 1] * ny + nr[:, 2] * nz - 0.15) / 0.6, 0.0, 1.0)
            sit = np.zeros(len(co))
            for sx in seat_x:
                sit = np.maximum(sit, np.exp(-((co[:, 0] - sx) / 0.19) ** 2))
            if is_back:
                u = (i + 0.5) / n
                amt = 0.08 + 0.6 * sit * (0.35 + 0.65 * sin(pi * min(1.0, u * 1.15)))
                if top_edge:
                    amt = np.maximum(amt, 0.6 * (co[:, 2] > cz))
            else:
                amt = 0.12 + 0.85 * sit * (1.0 - 0.45 * i / max(n - 1, 1))
                if first:
                    amt = amt + 0.25
            # NB: in the renderer the wood 'wear' layer reads as a dark, smoother hand-and-coat patina
            # (not pale), so it is kept under ~0.45 where it only darkens along the grain
            return np.clip(top * amt * (0.35 + 0.65 * wn1(co, nr)) * (0.45 + 1.0 * W), 0.0, 0.3 + (0.16 if p.wood == "oak" else 0.03) * W)

        def grime_fn(co, nr, ny=ny, nz=nz, is_back=is_back, cy=cy, cz=cz, ty=ty, tz=tz):
            under = np.clip(-(nr[:, 1] * ny + nr[:, 2] * nz), 0.0, 1.0) * 0.55
            ends = np.clip((np.abs(co[:, 0]) - (Ls / 2 - 0.09)) / 0.09, 0.0, 1.0) * 0.6
            g = np.maximum(under, ends * (0.4 + 0.6 * wn2(co, nr)))
            along = (co[:, 1] - cy) * ty + (co[:, 2] - cz) * tz          # across the board
            for bx in bolt_x:
                st = np.exp(-((co[:, 0] - bx) / 0.014) ** 2)
                sel = (along < 0.004) if is_back else (np.abs(along) < 0.012)
                g = np.maximum(g, st * sel * 0.5)
            return np.clip(g * (0.35 + 0.8 * W), 0.0, 1.0)

        sp = None
        if p.wood == "weathered":
            sp = mask_mul(noise_mask(0.15, seed=p.seed + 14, lo=0.4, hi=0.75), 0.25 + 0.6 * W, facing((0, 0.6, -0.4)))
        elif p.wood == "painted":
            sp = mask_mul(facing((0, -0.3, 1.0)), 0.25 + 0.5 * W, noise_mask(0.09, seed=p.seed + 15, lo=0.3, hi=0.8))
        if fresh:
            paint(o, grime=0.05)
        else:
            paint(o, wear=wear_fn, grime=grime_fn, special=sp)
        for bx in bolt_x:
            for off in ((-0.25, 0.25) if w > 0.085 else (0.0,)):
                ay, az = cy + ny * (t / 2 - 0.0006) + ty * off * w, cz + nz * (t / 2 - 0.0006) + tz * off * w
                dome_m.append(trs((bx, ay, az), (th, 0, r.u(0, 360))))
                if is_back and abs(bx) > 0.01:
                    nut_m.append(trs((bx, ay - ny * (t + tm - 0.001), az - nz * (t + tm - 0.001)), (th + 180, 0, r.u(0, 60))))
    if far:
        delete(dome)
        delete(nut)
        k = 0
        while k < len(fsl):
            j = k
            while j + 1 < len(fsl) and fsl[j + 1][0] == fsl[k][0] and fsl[j + 1][1] is fsl[k][1]:
                j += 1
            (_, finish, e0, _, h0), (_, _, _, e1, h1) = fsl[k], fsl[j]
            o = _fprism(f"slats_{'b' if fsl[k][0] else 's'}{k}", [(e0[0] - h0[0], e0[1] - h0[1]), (e1[0] - h1[0], e1[1] - h1[1]),
                                                                (e1[0] + h1[0], e1[1] + h1[1]), (e0[0] + h0[0], e0[1] + h0[1])], -Ls / 2, Ls / 2)
            assign(uv_grain(o, 'X'), finish)
            k = j + 1
    else:
        domes = instances_to_mesh("coach_bolts", dome, dome_m, keep_base=False)
        assign(domes, bolt_mat)
        paint(domes, grime=0.2 + 0.3 * W, special=mask_mul(noise_mask(0.05, seed=p.seed + 5), 0.2 + 0.6 * W))
        iron_objs.append(instances_to_mesh("nuts", nut, nut_m, keep_base=False))

    if strap and not far:
        # wrought strap directly under the slats (profile shifted off the bearer line to touch them)
        off = -(tm / 2 - 0.003)
        iron_objs.append(_bar("centre_strap", bearer, 0.0, [(a + off, b) for a, b in rounded_rect(0.007, 0.032, 0.002, 1)],
                              ang=10, dmax=0.09))

    # ---- materials and ageing of the ironwork
    for o in iron_objs:
        assign(o, iron)
    for o in gilt_objs:
        assign(o, gilt)
    n1, n2 = noise_mask(0.05, seed=p.seed + 2, lo=0.3, hi=0.75), noise_mask(0.13, seed=p.seed + 3, lo=0.58, hi=0.82)
    n3 = noise_mask(0.04, seed=p.seed + 4, lo=0.3, hi=0.8)

    def rust_fn(co, nr):
        low = np.clip(1.0 - co[:, 2] / 0.24, 0.0, 1.0) ** 1.3
        return np.clip((low * (0.25 + 0.75 * n1(co, nr)) + 0.4 * n2(co, nr)) * (0.15 + 1.0 * W), 0.0, 1.0)

    def dirt_fn(co, nr):
        # street dirt splashed up the feet and lying in the up-facing hollows of the casting
        low = np.clip(1.0 - co[:, 2] / 0.2, 0.0, 1.0) ** 1.5
        return np.clip(low * (0.45 + 0.4 * n1(co, nr)) + 0.18 * np.clip(nr[:, 2], 0.0, 1.0) * n3(co, nr), 0.0, 1.0)

    def chip_fn(co, nr):
        arm = np.clip((co[:, 2] - 0.60) / 0.03, 0.0, 1.0) * np.clip(nr[:, 2], 0.0, 1.0) * (co[:, 1] < 0.12) * (np.abs(co[:, 0]) > Lf / 2 - 0.06)
        toe = np.clip(1.0 - co[:, 2] / 0.07, 0.0, 1.0) * np.clip(-nr[:, 1], 0.0, 1.0)
        return np.clip((arm * 0.75 + toe * 0.5) * (0.4 + 0.6 * n3(co, nr)) * (0.3 + 0.9 * W), 0.0, 1.0)

    paint(iron_objs, special=rust_fn, wear=chip_fn, grime=dirt_fn)
    if not gilt_objs:
        pass                                                 # (far level: rosettes and eyes are left out)
    elif p.gilt:
        paint(gilt_objs, grime=0.2 + 0.3 * W)
    else:
        paint(gilt_objs, special=rust_fn)

    for i, sx in enumerate(seat_x):
        socket(f"seat{i}", "attach", (sx, -0.06, 0.45), role="seat")
