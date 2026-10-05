"""Tall heraldic cloth banner hung flat against a wall: a turned oak pole with brass finials rests in the
cradles of two forged-iron scroll brackets (wall plate, arm, C-scroll brace, coach bolts). The canvas
banner is sewn round the pole with a sleeve (or tabs), has a woven gold border band, rolled side hems,
a gold applique emblem (crown / star / fleur-de-lis / key) that follows the folds, a shaped bottom
(swallow-tail, pointed, dagged, straight) finished with braid and fringe, and a gold cord swag with two
tassels. The cloth is a 'flutter' part, the tassels are 'swing' parts.
"""
import bisect
import bmesh
import numpy as np
from math import sin, cos, pi, acos, sqrt, radians
from mathutils import Vector
from tk import *

# lod "far": lod 0 and lod 1 as built; from 44 m on (lod() == 2) build() models the banner anew from the _far_* stand-ins
META = {"name": "Hanging banner with crest", "class": "component", "mount": "wall", "mount_height": 4.3,
        "focus": 0.72, "lod": "far"}
PARAMS = {
    "width": {"default": 0.85, "min": 0.5, "max": 1.6, "desc": "cloth width (m)"},
    "drop": {"default": 2.1, "min": 1.0, "max": 3.6, "desc": "cloth length from the pole to its lowest point (m)"},
    "cloth": {"default": "#1d3266", "desc": "cloth colour"},
    "trim": {"default": "#dcae4c", "desc": "colour of the emblem, border, fringe and cords (gold)"},
    "emblem": {"default": "crown", "choices": ["crown", "star", "fleur", "key", "none"], "desc": "applique emblem"},
    "emblem_size": {"default": 0.64, "min": 0.3, "max": 0.8, "desc": "emblem width as a fraction of the cloth width"},
    "bottom": {"default": "swallowtail", "choices": ["swallowtail", "pointed", "dagged", "straight"], "desc": "shape of the bottom edge"},
    "tail": {"default": 0.36, "min": 0.1, "max": 0.8, "desc": "depth of the swallow-tail notch / point (m)"},
    "border": {"default": 0.034, "min": 0.0, "max": 0.08, "desc": "width of the woven gold border band (0 = none)"},
    "stripes": {"default": 1, "min": 0, "max": 2, "desc": "thin gold stripes echoing the bottom edge above the border"},
    "accent": {"default": "#8e1b22", "desc": "colour of the small jewel appliques on the crown band"},
    "fringe": {"default": 0.07, "min": 0.0, "max": 0.15, "desc": "length of the bottom fringe (0 = plain rolled hem)"},
    "heading": {"default": "sleeve", "choices": ["sleeve", "tabs"], "desc": "how the cloth is hung on the pole"},
    "tassels": {"default": True, "desc": "gold cord swag with a tassel at each end of the pole"},
    "finial": {"default": "spear", "choices": ["spear", "ball", "acorn"], "desc": "pole finial"},
    "standoff": {"default": 0.2, "min": 0.12, "max": 0.4, "desc": "distance of the pole from the wall (m)"},
    "folds": {"default": 0.7, "min": 0.0, "max": 1.0, "desc": "depth of the hanging folds / billow"},
    "age": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "sun-fade, grime, rust and fraying"},
}
VARIANTS = {
    "crimson_fleur": {"cloth": "#7c1b20", "emblem": "fleur", "bottom": "pointed", "heading": "tabs", "width": 0.7,
                      "drop": 1.75, "tail": 0.3, "finial": "ball", "seed": 4},
    "green_key_long": {"cloth": "#1f4d38", "emblem": "key", "bottom": "straight", "width": 1.0, "drop": 2.9,
                       "fringe": 0.11, "border": 0.05, "finial": "acorn", "trim": "#e0b24a", "seed": 9},
    "purple_star_dagged": {"cloth": "#4b2a66", "emblem": "star", "bottom": "dagged", "width": 1.25, "drop": 1.5,
                           "tail": 0.34, "emblem_size": 0.5, "standoff": 0.28, "trim": "#d8d2c0", "seed": 15},
    "faded_old": {"cloth": "#2a4a6e", "age": 1.0, "folds": 1.0, "fringe": 0.06, "tail": 0.5, "drop": 2.3,
                  "heading": "tabs", "seed": 23},
}


def _qb(a, c, b, n=6):
    """Quadratic bezier a -> b (control c), end point excluded."""
    out = []
    for i in range(n):
        t = i / n
        out.append(((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * c[0] + t * t * b[0],
                    (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * c[1] + t * t * b[1]))
    return out


def _loop(pts, n=36):
    """Smooth closed loop through 2D control points."""
    P = resample_path([(x, y, 0.0) for x, y in pts], n=n, closed=True, smooth_path=True)
    return [(float(q[0]), float(q[1])) for q in P]


def _emblem_pieces(kind):
    """-> (pieces [(outer, holes, layer)], unit width, unit height, y centre) in unit coordinates."""
    mx = lambda pts: [(-x, y) for x, y in reversed(pts)]
    if kind == "crown":
        BL, BR, TR, VR, TC = (-0.40, 0.0), (0.40, 0.0), (0.53, 0.50), (0.21, 0.20), (0.0, 0.76)
        VL, TL = (-VR[0], VR[1]), (-TR[0], TR[1])
        body = (_qb(BL, (0, -0.07), BR, 8) + _qb(BR, (0.41, 0.27), TR) + _qb(TR, (0.37, 0.30), VR)
                + _qb(VR, (0.10, 0.34), TC) + _qb(TC, (-0.10, 0.34), VL) + _qb(VL, (-0.37, 0.30), TL)
                + _qb(TL, (-0.41, 0.27), BL))
        loz = lambda cx, cy, a, b: [(cx + a, cy), (cx, cy + b), (cx - a, cy), (cx, cy - b)]
        holes = [loz(0, 0.17, 0.05, 0.085), loz(0.25, 0.10, 0.035, 0.058), loz(-0.25, 0.10, 0.035, 0.058)]
        band = (_qb((-0.42, -0.045), (0, -0.115), (0.42, -0.045), 8) + [(0.42, -0.045)]
                + _qb((0.43, -0.15), (0, -0.22), (-0.43, -0.15), 8) + [(-0.43, -0.15)])
        pcs = [(body, holes, 0), (band, [], 1)]
        for (jx, jy) in ((0.0, 0.795), (0.548, 0.532), (-0.548, 0.532)):
            pcs.append((circle_pts(0.05, 12, center=(jx, jy)), [], 1))
        for jx in (-0.3, -0.15, 0.0, 0.15, 0.3):
            jy = -0.0975 - 0.035 * (1 - (jx / 0.42) ** 2)
            pcs.append((circle_pts(0.027, 8, center=(jx, jy)), [], 2))
        return pcs, 1.2, 1.07, 0.31
    if kind == "star":
        star = []
        for i in range(10):
            a = radians(90 + i * 36)
            r = 0.5 if i % 2 == 0 else 0.2
            star.append((r * cos(a), r * sin(a)))
        ring = (circle_pts(0.68, 40), [circle_pts(0.615, 40)], 0)
        return [(star, [circle_pts(0.075, 10)], 0), ring], 1.36, 1.36, 0.0
    if kind == "fleur":
        mid = (_qb((0.05, -0.02), (0.06, 0.15), (0.14, 0.28)) + _qb((0.14, 0.28), (0.16, 0.45), (0.0, 0.66))
               + _qb((0.0, 0.66), (-0.16, 0.45), (-0.14, 0.28)) + _qb((-0.14, 0.28), (-0.06, 0.15), (-0.05, -0.02))
               + [(-0.05, -0.02)])
        petal = _loop([(0.09, -0.02), (0.13, 0.16), (0.21, 0.33), (0.34, 0.39), (0.45, 0.31), (0.49, 0.16),
                       (0.44, 0.04), (0.37, 0.03), (0.40, 0.13), (0.38, 0.22), (0.32, 0.27), (0.25, 0.22),
                       (0.21, 0.10), (0.19, -0.02)], 40)
        band = [(-0.23, -0.105), (0.23, -0.105), (0.23, -0.015), (-0.23, -0.015)]
        stem = (_qb((0.055, -0.1), (0.085, -0.25), (0.0, -0.42)) + _qb((0.0, -0.42), (-0.085, -0.25), (-0.055, -0.1))
                + [(-0.055, -0.1)])
        tail = _loop([(0.095, -0.1), (0.18, -0.1), (0.25, -0.17), (0.33, -0.25), (0.27, -0.30), (0.20, -0.24),
                      (0.135, -0.18)], 24)
        pcs = [(mid, [], 0), (petal, [], 0), (mx(petal), [], 0), (band, [], 1), (stem, [], 0), (tail, [], 0),
               (mx(tail), [], 0)]
        return pcs, 1.0, 1.1, 0.12
    if kind == "key":
        ca, sa = cos(radians(38)), sin(radians(38))
        rot = lambda pts: [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]
        shaft = [(-0.035, -0.5), (0.035, -0.5), (0.035, -0.47), (0.21, -0.47), (0.21, -0.40), (0.14, -0.40),
                 (0.14, -0.355), (0.21, -0.355), (0.21, -0.27), (0.035, -0.27), (0.035, 0.19), (-0.035, 0.19)]
        bow = circle_pts(0.18, 28, center=(0, 0.35))
        bow_hole = circle_pts(0.095, 20, center=(0, 0.35))
        col1 = [(-0.075, 0.11), (0.075, 0.11), (0.075, 0.155), (-0.075, 0.155)]
        col2 = [(-0.06, 0.03), (0.06, 0.03), (0.06, 0.055), (-0.06, 0.055)]
        knob = circle_pts(0.05, 12, center=(0, 0.575))
        pcs = [(rot(shaft), [], 0), (rot(bow), [rot(bow_hole)], 1), (rot(col1), [], 1), (rot(col2), [], 1),
               (rot(knob), [], 0)]
        return pcs, 0.95, 1.0, 0.03
    return [], 1.0, 1.0, 0.0


# ---- lod 2 (from 44 m on, about 20 px/m): plain stand-ins in world coordinates. The vertex masks of a level are
# transferred from lod 0, so none is painted here.
FAR_LIFT = 0.008                # the gold band, the stripes and the emblem lie this far in front of the cloth
FAR_TOL, FAR_PIECE = 0.02, 0.0009      # corners of the emblem flatter than this go (m); pieces smaller than this (m2)


def _far_obj(name, verts, faces, m, fix=False, front=False):
    """fix: faces turned outwards (closed shapes); front: every face turned to the street (-Y; sheets on the cloth)"""
    if front:
        out = []
        for f in faces:
            a, b, c = (Vector(verts[i]) for i in f[:3])
            out.append(tuple(reversed(f)) if (b - a).cross(c - a).y > 0 else tuple(f))
        faces = out
    o = mesh_object(name, verts, faces)
    if fix:
        recalc_normals(o)
    smooth(o, 60)
    uv_box(o, scale=1.0)
    assign(o, m)
    return o


def _far_turn(name, rings, m, cap=False):
    """A turned body as a shell: rings of the same number of points, first to last (a ring of one point is a pole);
    cap closes the first ring."""
    v, rows = [], []
    n = max(len(r) for r in rings)
    for r in rings:
        rows.append([len(v) + (i if len(r) > 1 else 0) for i in range(n)])
        v += [tuple(q) for q in r]
    f = [tuple(rows[0])] if cap else []
    for r0, r1 in zip(rows, rows[1:]):
        for i in range(n):
            q = (r0[i], r0[(i + 1) % n], r1[(i + 1) % n], r1[i])
            f.append(tuple(q[k] for k in range(4) if q[k] != q[k - 1]))
    return _far_obj(name, v, f, m, fix=True)


def _far_rod(name, a, b, r, m, r1=None):
    """A cord as one three-sided prism from a (radius r) to b (radius r1)."""
    a, b = Vector(a), Vector(b)
    d = (b - a).normalized()
    x = d.orthogonal().normalized()
    y = d.cross(x)
    v = [tuple(q + (x * cos(t) + y * sin(t)) * rr) for q, rr in ((a, r), (b, r1 or r)) for t in (0.0, 2.0944, 4.1888)]
    return _far_obj(name, v, [(i, (i + 1) % 3, 3 + (i + 1) % 3, 3 + i) for i in range(3)] + [(2, 1, 0), (3, 4, 5)], m)


def _far_bar(name, path, wd, t, m):
    """A flat iron bar (wd wide along X, t thick) bent in the Y / Z plane, as a closed prism of three faces."""
    P = [Vector(q) for q in path]
    v = []
    for i, q in enumerate(P):
        d = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]).normalized()
        n = Vector((0.0, -d.z, d.y))
        v += [tuple(q + Vector((wd * 0.5, 0, 0)) - n * (t * 0.5)), tuple(q - Vector((wd * 0.5, 0, 0)) - n * (t * 0.5)),
              tuple(q + n * (t * 0.5))]
    f = [(3 * i + j, 3 * i + (j + 1) % 3, 3 * i + 3 + (j + 1) % 3, 3 * i + 3 + j) for i in range(len(P) - 1) for j in range(3)]
    k = 3 * (len(P) - 1)
    return _far_obj(name, v, f + [(2, 1, 0), (k, k + 1, k + 2)], m, fix=True)


def _area(pts):
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))) * 0.5


def _far_outline(pts, tol, nmax=10):
    """A closed outline without the corners that stand less than tol off the line of their neighbours (the flattest
    goes first), and of nmax corners at most."""
    pts = list(pts)
    while len(pts) > 3:
        hs = []
        for i in range(len(pts)):
            a, q, b = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            ln = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            hs.append(abs((q[0] - a[0]) * (b[1] - a[1]) - (q[1] - a[1]) * (b[0] - a[0])) / ln if ln > 1e-9 else 0.0)
        i = min(range(len(pts)), key=hs.__getitem__)
        if hs[i] >= tol and len(pts) <= nmax:
            break
        pts.pop(i)
    return pts


def build(p, rng):
    w, drop, s, age = p.width, p.drop, p.standoff, p.age
    hw = w / 2
    rp = 0.019                      # pole radius
    th = 0.003                      # cloth thickness
    navy = mat("canvas", p.cloth)
    goldc = mat("canvas", p.trim, scale=1.5)
    cord = mat("rope", p.trim)
    wood = mat("wood_oak", "#4d3523")
    brass = mat("brass")
    iron = mat("iron_wrought")
    far = lod() == 2

    # ------------------------------------------------------------------ cloth layout (flat, in X / Z)
    tabs = p.heading == "tabs"
    gap_s = 0.055 if tabs else 0.032
    z_top = -(rp + gap_s)           # top edge of the cloth panel (seam under the pole)
    shape = p.bottom
    if shape == "straight":
        tail, slope = 0.0, 0.0
    elif shape == "dagged":
        tail = min(p.tail * 0.6, drop * 0.3, w * 0.4)
        slope = tail / (w / 6)
    else:
        tail = min(p.tail, drop * 0.42)
        slope = tail / hw
    k = sqrt(1 + slope * slope)
    Lmin = drop - tail

    def Lx(x):
        a = min(abs(x) / hw, 1.0)
        if shape == "pointed":
            return drop - tail * a
        if shape == "swallowtail":
            return drop - tail * (1 - a)
        if shape == "dagged":
            f = (abs(x) / (w / 6)) % 2.0
            f = f if f <= 1 else 2 - f
            return drop - tail * f
        return drop

    bw = min(p.border, w * 0.09) if p.border >= 0.012 else 0.0
    inset = 0.022 if bw else 0.0
    a0 = inset + bw
    # thin stripes above the bottom border: (lo, hi) distances from the bottom edge
    bands, o_ = [], a0
    for i in range(int(p.stripes)):
        if Lmin - a0 - k * (o_ + 0.046) < 0.55 * Lmin:      # no room on short banners with deep tails
            break
        bands.append((o_ + 0.03, o_ + 0.046))
        o_ += 0.046
    b1 = k * o_                     # vertical offset of the highest bottom-referenced row

    # folds: one displacement field shared by the cloth, its trimmings and the applique
    r = rng.fork("drape")
    ph1, ph2, ph3 = r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28)
    lam1, lam2 = w / r.u(2.5, 3.3), w / r.u(1.05, 1.45)
    A = 0.044 * p.folds
    bill = 0.045 * p.folds * r.u(0.4, 1.0)
    twist = r.u(-1, 1) * 0.04 * p.folds
    wl = max(0.15, w / 4.6) * r.pick([-1, 1])
    curl_l, curl_r = r.u(-1, 1) * 0.05 * p.folds, r.u(-1, 1) * 0.05 * p.folds

    def drape(x, z):
        t = min(max((z_top - z) / drop, 0.0), 1.3)
        env = (0.42 + 0.58 * smoothstep(0.0, 0.45, t)) * (0.45 + 0.55 * t)
        xn = x / hw
        th1 = 2 * pi * x / lam1 + ph1 + 0.7 * t
        dy = A * env * (0.7 * sin(th1) + 0.22 * sin(2 * th1 + 1.1 + 0.8 * t) + 0.45 * sin(2 * pi * x / lam2 + ph2 - 1.2 * t))
        dy += A * 0.5 * t * t * sin(4.0 * t + ph3 + xn)
        # small diagonal drag wrinkles (also near the heading, where the big folds are still shallow)
        dy += 0.0045 * p.folds * sin(2 * pi * (x / wl + z / 0.55) + ph3) * sin(2 * pi * z / 0.41 + ph2)
        dy -= bill * t ** 1.6
        dy += twist * xn * t * t
        dy += (curl_l if x < 0 else curl_r) * xn * xn * t ** 3
        return x * (1 - 0.085 * p.folds * env), -s + dy, z

    # grid columns (kinks of the bottom edge and the border band fall on columns)
    ncol = int(min(36, max(24, round(w / 0.025 / 12) * 12)))
    req = [-hw, hw]
    if bw:
        req += [-(hw - inset), hw - inset, -(hw - a0), hw - a0]
    if shape == "dagged":
        req += [i * w / 6 for i in range(-2, 3)]
    else:
        req += [0.0]
    xs = list(req)
    for i in range(ncol + 1):
        x = -hw + w * i / ncol
        if all(abs(x - q) > 0.45 * w / ncol for q in req):
            xs.append(x)
    xs.sort()
    if far:                         # the band's columns, the kinks of the bottom edge and one column per fold
        xs = sorted(set(([-hw, hw, -(hw - inset), hw - inset, -(hw - a0), hw - a0] if bw else [-hw, -w / 3, w / 3, hw])
                        + [i * w / 6 for i in (range(-2, 3) if shape == "dagged" else range(-1, 2))]))
    # rows: referenced to the top edge, blending into rows parallel to the shaped bottom edge
    nb = int(min(20, max(8, round((Lmin - a0 - b1) / 0.11))))
    boffs = {0.0}
    if bw:
        boffs |= {k * inset, k * a0}
    for lo, hi in bands:
        boffs |= {k * lo, k * hi}
    nb = int(max(6, min(nb, 950 // (len(xs) - 1) - len(boffs) - (2 if bw else 0))))     # triangle budget
    rows = [("t", 0.0)] + ([("t", inset), ("t", a0)] if bw else [])
    rows += [("m", i / nb) for i in range(1, nb)]
    rows += [("b", v) for v in sorted(boffs, reverse=True)]
    if far:                         # the band's rows and a few between; the thin stripes lie on the cloth as sheets
        nb = int(clamp(round(Lmin / 0.5), 2, 6))
        rows = ([("t", 0.0)] + ([("t", inset), ("t", a0)] if bw else []) + [("m", i / nb) for i in range(1, nb)]
                + ([("b", k * a0), ("b", k * inset)] if bw else []) + [("b", 0.0)])
    tau0 = clamp(1 - 2.2 * tail / max(Lmin, 0.1), 0.15, 0.85)

    def zrow(row, x):
        kind, v = row
        if kind == "t":
            return z_top - v
        if kind == "b":
            return z_top - Lx(x) + v
        zf = z_top - a0 - v * (Lmin - a0 - b1)
        zs = z_top - a0 - v * (Lx(x) - a0 - b1)
        return lerp(zf, zs, smoothstep(tau0, 1.0, v))

    nx, nr = len(xs), len(rows)
    colz = [[zrow(row, x) for row in rows] for x in xs]
    colp = [[drape(x, z) for z in zl] for x, zl in zip(xs, colz)]

    def along(i, z):
        zl, pl = colz[i], colp[i]
        if z >= zl[0]:
            return pl[0]
        for j in range(nr - 1):
            if z >= zl[j + 1]:
                f = (zl[j] - z) / max(zl[j] - zl[j + 1], 1e-9)
                return (lerp(pl[j][0], pl[j + 1][0], f), lerp(pl[j][1], pl[j + 1][1], f))
        return pl[-1]

    def drape_lin(x, z):
        """Drape exactly as the faceted cloth mesh does (so trimmings never dip through it)."""
        i = min(max(bisect.bisect_right(xs, x) - 1, 0), nx - 2)
        f = clamp((x - xs[i]) / (xs[i + 1] - xs[i]))
        a, b = along(i, z), along(i + 1, z)
        return (lerp(a[0], b[0], f), lerp(a[1], b[1], f), z)

    verts = [(x, 0.0, z) for x, zl in zip(xs, colz) for z in zl]
    faces = [(i * nr + j, i * nr + j + 1, (i + 1) * nr + j + 1, (i + 1) * nr + j)
             for i in range(nx - 1) for j in range(nr - 1)]
    if far:                         # the front as it hangs, the back as one face per column
        nv = len(verts)
        verts = [(x, -th * 0.5, z) for x, _, z in verts] + [(x, th * 0.5, zl[j]) for x, zl in zip(xs, colz) for j in (0, nr - 1)]
        faces += [(nv + 2 * i, nv + 2 * i + 2, nv + 2 * i + 3, nv + 2 * i + 1) for i in range(nx - 1)]
        # ... and closed between the two along the sides and the bottom edge: seen from the side or from below
        # nobody looks in between the sheets (the back is the chord of the drape, so the slit is as wide as the sag)
        ie = nx - 1
        faces.append(tuple([nv, nv + 1] + [nr - 1 - j for j in range(nr)]))
        faces.append(tuple([ie * nr + j for j in range(nr)] + [nv + 2 * ie + 1, nv + 2 * ie]))
        faces += [(i * nr + nr - 1, nv + 2 * i + 1, nv + 2 * i + 3, (i + 1) * nr + nr - 1) for i in range(nx - 1)]
    cloth = mesh_object("cloth", verts, faces)
    assign(cloth, navy)
    if bw or bands:
        def in_gold(c, n):
            ds, dt, db = hw - abs(c.x), z_top - c.z, (c.z - (z_top - Lx(c.x))) / k
            d = min(ds, dt, db)
            if bw and inset + 1e-4 < d < a0 - 1e-4:
                return True
            return not far and ds > a0 + 1e-4 and any(lo + 1e-4 < db < hi - 1e-4 for lo, hi in bands)
        assign_faces(cloth, goldc, in_gold)
    if far:
        transform_verts(cloth, lambda co: Vector(drape(co.x, co.z)) + Vector((0.0, co.y, 0.0)))
    else:
        transform_verts(cloth, lambda co: Vector(drape(co.x, co.z)))
        solidify(cloth, th, offset=0.0)
    smooth(cloth, 60)
    uv_planar(cloth, (1, 0, 0), (0, 0, 1))
    soft = [cloth]
    if far and bands:               # the stripes: strips of their own just in front of the cloth
        def far_on(x, z):
            q = drape_lin(x, z)
            return (q[0], q[1] - th * 0.5 - FAR_LIFT, q[2])
        zb = lambda x: z_top - Lx(x)
        gv, gf = [], []
        xa = hw - a0
        for b0, b1_ in bands:
            xl = [-xa] + [x for x in xs if abs(x) < xa - 1e-4] + [xa]
            n0 = len(gv)
            gv += [far_on(x, zb(x) + k * v) for x in xl for v in (b0, b1_)]
            gf += [(n0 + 2 * i, n0 + 2 * i + 1, n0 + 2 * i + 3, n0 + 2 * i + 2) for i in range(len(xl) - 1)]
        soft.append(_far_obj("trim", gv, gf, goldc, front=True))

    # rolled side hems (and bottom hem when there is no fringe)
    left = [drape(-hw, zrow(row, -hw)) for row in rows]
    right = [drape(hw, zrow(row, hw)) for row in rows]
    bottom = [drape(x, z_top - Lx(x)) for x in xs]
    if far:                         # hems left out; braid and fringe as one strip below the edge, front and back
        if p.fringe >= 0.015:
            rf = rng.fork("fringe")
            hf = 0.005 + p.fringe * 0.95 * (1.0 - 0.117 * age)
            xk = [i * w / 6 for i in range(-3, 4)] if shape == "dagged" else [i * w / 4 for i in range(-2, 3)]
            fq = [drape(x, z_top - Lx(x)) for x in xk]
            fv = [(q[0], q[1] + dy, q[2] + dz) for dy in (-0.005, 0.005) for q in fq for dz in (0.0075, -hf)]
            n, nk = 2 * len(xk), len(xk) - 1
            soft.append(_far_obj("fringe", fv, [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(nk)]
                                 + [(n + 2 * i, n + 2 * i + 2, n + 2 * i + 3, n + 2 * i + 1) for i in range(nk)]
                                 + [(2 * i + 1, n + 2 * i + 1, n + 2 * i + 3, 2 * i + 3) for i in range(nk)]      # closed below
                                 + [(0, n, n + 1, 1), (2 * nk, 2 * nk + 1, n + 2 * nk + 1, n + 2 * nk)], cord))   # ... and at the ends
    elif p.fringe >= 0.015:
        for nm, path in (("hem_l", left), ("hem_r", right)):
            h = sweep(nm, path, 0.0034, sides=4, smooth_path=False, up=(0, -1, 0))
            assign(h, navy)
            soft.append(h)
        braid = sweep("braid", bottom, rounded_rect(0.007, 0.015, 0.002, 1), smooth_path=False, up=(0, 0, 1))
        assign(braid, cord)
        soft.append(braid)
        # fringe: individual strands hanging plumb from the braid
        rf = rng.fork("fringe")
        strand = mesh_object("strand", [(0.0028 * cos(a), 0.0028 * sin(a), 0.0) for a in (0.0, 2.1, 4.2)]
                             + [(0.0016 * cos(a), 0.0016 * sin(a), -1.0) for a in (0.0, 2.1, 4.2)],
                             [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5), (3, 4, 5)])
        recalc_normals(strand)
        assign(strand, cord)
        uv_grain(strand, "Z")
        realize(strand)
        elen = w * k
        pitch = max(0.0125, elen / 110)
        n = int(elen / pitch)
        mats, gap = [], 0
        for i in range(n + 1):
            x = -hw + 0.004 + (w - 0.008) * i / n
            if gap > 0:
                gap -= 1
                continue
            if rf.chance(0.05 * age):
                gap = rf.pick([1, 2, 3, 5])
                continue
            q = drape_lin(x, z_top - Lx(x))
            ln = p.fringe * rf.u(0.84, 1.06) * (1.0 - 0.35 * age * rf.u(0, 1) ** 2)
            mats.append(trs((q[0], q[1], q[2] - 0.005), (rf.jitter(0, 7), rf.jitter(0, 6), rf.u(0, 360)), (1, 1, ln)))
        soft.append(instances_to_mesh("fringe", strand, mats, keep_base=False))
    else:
        h = sweep("hem", left + bottom[1:-1] + right[::-1], 0.0034, sides=5, smooth_path=False, up=(0, -1, 0))
        assign(h, navy)
        soft.append(h)

    # ------------------------------------------------------------------ heading: sleeve or tabs round the pole
    d = rp + gap_s
    rs = rp + 0.0035
    phi = acos(rs / d)
    off = th * 0.5 + 0.0013
    ring = [(-off, -d - 0.014), (-off, -d)]
    ring += [(-rs * sin(a), -rs * cos(a)) for a in np.linspace(phi, 2 * pi - phi, 9)]
    ring += [(off, -d), (off, -d - 0.014)]
    nvr = len(ring) - 1
    rw = rng.fork("sleeve")

    def wrap(name, x0, x1, nu, m):
        pa, pb = rw.u(0, 6.28), rw.u(0, 6.28)
        if far:                     # four faces from the top of the pole to the seam, the ends closed
            r4 = 1.3 * rs
            sec = [(-s, r4), (-s - r4, 0.0), (-s, z_top - 0.004), (-s + r4, 0.0)]
            return _far_obj(name, [(x, y, z) for x in (x0, x1) for y, z in sec],
                            [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)] + [(3, 2, 1, 0), (4, 5, 6, 7)],
                            m, fix=True)

        def fn(u, v):
            x = lerp(x0, x1, u)
            j = min(int(round(v * nvr)), nvr)
            ry, rz = ring[j]
            c = sin(pi * v) ** 2
            g = 1.0 + c * (0.05 * sin(x * 31 + pa) + 0.035 * sin(x * 77 + pb))
            q = drape(x, z_top)                                   # the seam follows the gathered cloth
            wv = smoothstep(0.0, d, -rz)
            return (lerp(x, q[0], wv), -s + ry * g + wv * (q[1] + s), rz * (g if rz > -d + 1e-6 else 1.0))
        o = surface(name, fn, nu, nvr, thickness=0.0025)
        assign(o, m)
        return o

    if tabs:
        ntab = max(3, int(round(w / 0.17)))
        tw = 0.07
        for i in range(ntab):
            cx = lerp(-hw + tw / 2 + 0.012, hw - tw / 2 - 0.012, i / (ntab - 1))
            soft.append(wrap(f"tab{i}", cx - tw / 2, cx + tw / 2, 2, goldc))
    else:
        soft.append(wrap("sleeve", -hw, hw, int(clamp(w / 0.09, 8, 11)), navy))

    # ------------------------------------------------------------------ applique emblem following the folds
    sag = min(0.07 + 0.1 * w, 0.17)                         # of the cord swag
    pcs, uw, uh, uy0 = _emblem_pieces(p.emblem)
    if pcs:
        avail = Lmin - a0 - b1
        sc = min(p.emblem_size * w / uw, 0.72 * avail / uh, (w - 2 * a0 - 0.04) / uw)
        clear = (sag - gap_s - rp - a0 + 0.035) if p.tassels else 0.04      # keep clear of the cord swag
        cz = z_top - a0 - min(max(0.4 * avail, sc * uh * 0.5 + max(clear, 0.04)), avail - sc * uh * 0.5 - 0.03)
        jewel = mat("canvas", p.accent, scale=1.5)
        emb = []
        for i, (outer, holes, layer) in enumerate(pcs):
            if far:                 # the outlines in a few corners, without the jewels, the small pieces and the small holes
                if layer == 2 or _area(outer) * sc * sc < FAR_PIECE:
                    continue
                outer = _far_outline(outer, FAR_TOL / sc)
                holes = [_far_outline(hh, FAR_TOL / sc) for hh in holes if _area(hh) * sc * sc >= FAR_PIECE]
            T = lambda pts: [(x * sc, cz + (y - uy0) * sc) for x, y in pts]
            dep = 0.0022 + 0.0011 * layer
            o = shape_extrude(f"emblem{i}", T(outer), [T(hh) for hh in holes], depth=dep, rot=(90, 0, 0))
            apply_transform(o)
            recalc_normals(o)
            # drop the hidden back, then cut along the cloth's grid lines so the patch folds with it
            bm = bmesh.new()
            bm.from_mesh(o.data)
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > (-0.5 if far else 0.5)], context='FACES')
            ex0, ex1 = min(v.co.x for v in bm.verts), max(v.co.x for v in bm.verts)
            ez0, ez1 = min(v.co.z for v in bm.verts), max(v.co.z for v in bm.verts)
            for xc in xs:
                if ex0 + 1e-4 < xc < ex1 - 1e-4:
                    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5,
                                           plane_co=(xc, 0, 0), plane_no=(1, 0, 0))
            for zc in ([] if far else colz[nx // 2]):      # far: cut at the columns only, where the cloth is creased
                if ez0 + 1e-4 < zc < ez1 - 1e-4:
                    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5,
                                           plane_co=(0, 0, zc), plane_no=(0, 0, 1))
            bm.to_mesh(o.data)
            bm.free()

            def lay(co):
                q = drape_lin(co.x, co.z)
                return Vector((q[0], q[1] - th * 0.5 - (FAR_LIFT if far else 0.0014) + co.y, q[2]))
            transform_verts(o, lay)
            assign(o, jewel if layer == 2 else goldc)
            smooth(o, 40)
            uv_planar(o, (1, 0, 0), (0, 0, 1))
            emb.append(o)
        soft += emb

    # ------------------------------------------------------------------ pole and finials
    ext = 0.125
    r4 = 1.3 * rp                   # far: pole and finials four-sided, a corner up and one to the street
    sq = lambda x, r: [(x, -s + r * cos(a), r * sin(a)) for a in (0.0, pi / 2, pi, 3 * pi / 2)]
    if far:
        pole = _far_turn("pole", [sq(sx * (hw + ext - 0.026), r4) for sx in (-1, 1)], wood)
    else:
        pole = cylinder("pole", rp, w + 2 * ext, loc=(0, -s, 0), rot=(0, 90, 0), verts=14)
        uv_grain(pole, "Z")
        smooth(pole, 50)
        assign(pole, wood)
    f0 = [(rp + 0.0012, -0.026), (rp + 0.004, -0.024), (rp + 0.004, -0.004), (rp + 0.0015, 0.0), (0.012, 0.004)]
    if p.finial == "ball":
        fp = f0 + [(0.010, 0.012), (0.016, 0.017), (0.016, 0.022), (0.011, 0.028), (0.022, 0.037), (0.031, 0.05),
                   (0.034, 0.064), (0.031, 0.078), (0.022, 0.09), (0.010, 0.097), (0.008, 0.102), (0.010, 0.107), (0.0, 0.112)]
    elif p.finial == "acorn":
        fp = f0 + [(0.010, 0.012), (0.017, 0.018), (0.012, 0.025), (0.024, 0.031), (0.029, 0.04), (0.030, 0.052),
                   (0.026, 0.057), (0.0265, 0.07), (0.023, 0.088), (0.014, 0.104), (0.005, 0.113), (0.004, 0.12), (0.0, 0.122)]
    else:
        fp = f0 + [(0.010, 0.012), (0.017, 0.019), (0.019, 0.025), (0.017, 0.031), (0.009, 0.038), (0.011, 0.046),
                   (0.019, 0.058), (0.024, 0.074), (0.024, 0.086), (0.019, 0.105), (0.011, 0.128), (0.004, 0.15), (0.0, 0.158)]
    hard = [pole]
    for sx in (-1, 1):
        if far:                     # from the pole's own section over the widest ring to the tip
            fr, fz = max(fp[5:], key=lambda q: q[0])
            hard.append(_far_turn(f"finial{sx}", [sq(sx * (hw + ext - 0.026), r4), sq(sx * (hw + ext + fz), 1.2 * fr),
                                                  [(sx * (hw + ext + fp[-1][1]), -s, 0.0)]], brass))
            continue
        f = lathe(f"finial{sx}", fp, segments=12, loc=(sx * (hw + ext), -s, 0), rot=(0, sx * 90, 0))
        assign(f, brass)
        paint(f, special=mask_mul(noise_mask(0.03, seed=p.seed + 5, lo=0.45, hi=0.8), 0.25 + 0.6 * age))
        hard.append(f)
    if not far:
        paint(pole, grime=0.2 + 0.4 * age, wear=mask_mul(noise_mask(0.08, seed=p.seed + 6), 0.4))

    # ------------------------------------------------------------------ forged wall brackets
    xb = hw + 0.05
    R = rp + 0.005
    r1 = min(0.036, s * 0.18)
    yc = -s * 0.68
    zc = -R - 0.007 - r1
    hbr = clamp(s * 0.95, 0.15, 0.3)
    zA = zc + r1 - hbr
    zt0, zb0 = 0.06, zA - 0.055
    iron_objs = []
    bolts = []
    rb = rng.fork("bracket")
    for sx in (-1, 1):
        x = sx * xb
        lean = rb.jitter(0.0, 0.004)
        if far:                     # the plate as two faces off the wall, arm and brace as bars of three faces, no bolts
            pv = [(x - 0.022, 0.0, zb0 + 0.028), (x + 0.022, 0.0, zb0 + 0.028), (x + 0.022, 0.0, zt0 - 0.028),
                  (x - 0.022, 0.0, zt0 - 0.028), (x, -0.008, zb0), (x, -0.008, zt0)]
            sp = spiral_path(0.007, r1, 1.3, n=18, a0_deg=90 + 360 * 1.3, plane="YZ", center=(x, yc, zc), ccw=False)
            S = sp[-1]
            bz = bezier_path(S, (x, S[1] + (abs(yc) - 0.009) * 0.62, S[2]), (x, -0.012, zA + hbr * 0.6), (x, -0.009, zA), n=10)
            iron_objs += [_far_obj(f"plate{sx}", pv, [(0, 3, 5, 4), (1, 4, 5, 2)], iron, front=True),
                          _far_bar(f"arm{sx}", [(x, -0.004, -R), (x, -s - 1.1 * R, -1.05 * R),
                                                (x, -s - R - 0.016, R * sin(radians(15)) + 0.02)], 0.026, 0.012, iron),
                          _far_bar(f"brace{sx}", [(x, yc - 0.85 * r1, zc - 0.5 * r1), S, bz[5], bz[10]], 0.02, 0.01, iron)]
            rb.u(0, 90), rb.u(0, 90)
            continue
        plate = extrude_profile(f"plate{sx}", [(-0.022, zb0 + 0.028), (0, zb0), (0.022, zb0 + 0.028), (0.022, zt0 - 0.028),
                                               (0, zt0), (-0.022, zt0 - 0.028)], 0.006, loc=(x, 0, 0), rot=(90, 0, 0), bevel=0.0012)
        arm = [(x, -0.004, -R), (x, -s * 0.5, -R)]
        arm += [(x, -s - R * cos(radians(a)), R * sin(radians(a)) + lean * 0) for a in range(-90, 16, 15)]
        arm += [(x, -s - R * cos(radians(15)) - 0.005, R * sin(radians(15)) + 0.012), (x, -s - R - 0.016, R * sin(radians(15)) + 0.02)]
        arm_o = sweep(f"arm{sx}", arm, rounded_rect(0.008, 0.026, 0.002, 1), step=0.016, up=(1, 0, 0),
                      scale_fn=lambda t: 1.0 - 0.45 * smoothstep(0.86, 1.0, t))
        sp = spiral_path(0.007, r1, 1.3, n=18, a0_deg=90 + 360 * 1.3, plane="YZ", center=(x, yc, zc), ccw=False)
        S = sp[-1]
        bz = bezier_path(S, (x, S[1] + (abs(yc) - 0.009) * 0.62, S[2]), (x, -0.012, zA + hbr * 0.6), (x, -0.009, zA), n=10)
        brace = sweep(f"brace{sx}", sp + bz[1:], rounded_rect(0.006, 0.02, 0.0015, 1), smooth_path=False, up=(1, 0, 0),
                      scale_fn=lambda t: 0.5 + 0.5 * smoothstep(0.0, 0.3, t))
        iron_objs += [plate, arm_o, brace]
        bolts += [trs((x, -0.006, zt0 - 0.034), (90, 0, rb.u(0, 90))), trs((x, -0.006, zb0 + 0.034), (90, 0, rb.u(0, 90))),
                  trs((x, -0.012, zA + 0.022), (90, 0, 0), 0.7), trs((x, -(abs(yc) + 0.0), -R + 0.004 - 0.008, ), (180, 0, 0), 0.6)]
    if not far:
        bt = bolt("bolt", 0.0075, 0.0055, kind="square")
        iron_objs.append(instances_to_mesh("bolts", bt, bolts, keep_base=False))
    rust = mask_mul(noise_mask(0.05, seed=p.seed + 3, lo=0.35, hi=0.7), 0.35 + 0.6 * age)
    for o in ([] if far else iron_objs):
        assign(o, iron)
        paint(o, special=rust, grime=0.15 + 0.3 * age, wear=mask_mul(noise_mask(0.04, seed=p.seed + 8, lo=0.4, hi=0.8), 0.5))

    # ------------------------------------------------------------------ cord swag and tassels
    if p.tassels:
        rt = rng.fork("tassel")
        xt = hw + 0.092
        yt = -s - rp - 0.008
        if far:                     # the swag as a strip facing the street; the wraps round the pole are left out
            cp = catenary((-xt, yt, -0.004), (xt, yt, -0.004), sag=sag, n=6)
            _far_obj("swag", [(q[0], q[1], q[2] + dz) for q in cp for dz in (0.006, -0.006)],
                     [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(6)], cord, front=True)
            for _ in range(4):
                rt.jitter(0, 4)
        else:
            swag = sweep("swag", catenary((-xt, yt, -0.004), (xt, yt, -0.004), sag=sag, n=22), 0.0046, sides=6,
                         smooth_path=False)
            assign(swag, cord)
            wraps = []
            for sx in (-1, 1):
                for dx in (-0.0045, 0.0045):
                    wraps.append(torus(f"wrap{sx}", rp + 0.0042, 0.0046, loc=(sx * xt + dx, -s, 0), rot=(0, 90, rt.jitter(0, 4)),
                                       major_seg=10, minor_seg=4))
            for o in wraps:
                assign(o, cord)
        missing = age > 0.85 and rt.chance(0.8)
        for sx in (-1, 1):
            if missing and sx == 1:
                continue
            Lc = rt.u(0.26, 0.44)
            sway = rt.jitter(0, 0.012)
            path = [(sx * xt, yt, 0.0), (sx * xt + sway * 0.3, yt - 0.004, -Lc * 0.5), (sx * xt + sway, yt - 0.002, -Lc)]
            if far:                 # the cord as one prism from its wraps round the pole down, the tassel three-sided
                c = _far_rod(f"cord{sx}", (sx * xt, yt, rp + 0.009), path[2], 0.009, cord, r1=0.004)
                tq = lambda r, z: [(path[2][0] + r * cos(a), path[2][1] + r * sin(a), path[2][2] + z) for a in (0.5236, 2.618, 4.7124)]
                ts = _far_turn(f"tassel{sx}", [tq(0.026, -0.142), tq(0.02, -0.02), [(path[2][0], path[2][1], path[2][2] + 0.006)]],
                               cord, cap=True)
                part([c, ts], "tassel_l" if sx < 0 else "tassel_r", pivot=(sx * xt, yt, 0.0), motion="swing")
                continue
            c = sweep(f"cord{sx}", path, 0.004, sides=6, step=0.04)
            assign(c, cord)
            tp = [(0.0, -0.142), (0.019, -0.14), (0.0215, -0.115), (0.019, -0.075), (0.0135, -0.046), (0.012, -0.04),
                  (0.0135, -0.037), (0.0135, -0.031), (0.010, -0.028), (0.0165, -0.02), (0.0185, -0.011), (0.0165, -0.002),
                  (0.009, 0.004), (0.0, 0.006)]
            ts = lathe(f"tassel{sx}", tp, segments=10, smooth_angle=35)

            def rib(co):
                if co.z < -0.045 and (co.x * co.x + co.y * co.y) > 1e-8:
                    a = np.arctan2(co.y, co.x)
                    g = 1.0 + 0.09 * cos(5 * a) * smoothstep(-0.045, -0.07, co.z)
                    return Vector((co.x * g, co.y * g, co.z))
                return co
            transform_verts(ts, rib)
            uv_transform(ts, rot_deg=90)
            place(ts, loc=(sx * xt + sway, yt - 0.002, -Lc))
            assign(ts, cord)
            part([c, ts], "tassel_l" if sx < 0 else "tassel_r", pivot=(sx * xt, yt, 0.0), motion="swing")

    # ------------------------------------------------------------------ weathering of the cloth
    zlow = z_top - drop
    n1 = noise_mask(0.3, seed=p.seed + 1, lo=0.3, hi=0.75)
    n2 = noise_mask(0.45, seed=p.seed + 2, lo=0.25, hi=0.8)

    def streak(co, nrm, fn, sx=1.0):
        q = co.copy()
        q[:, 0] *= sx
        q[:, 2] *= 0.12             # stretch the noise vertically -> rain / drip streaks
        return fn(q, nrm)

    def grime_fn(co, nrm):
        low = np.clip(1.0 - (co[:, 2] - zlow) / 0.6, 0, 1)
        top = np.clip(1.0 - np.abs(z_top - 0.03 - co[:, 2]) / 0.2, 0, 1) * 0.3
        st = streak(co, nrm, n1, 2.0)
        return np.clip((low * 0.9 + top + 0.1 + 0.3 * st) * (0.35 + 0.65 * n1(co, nrm)) * (0.35 + 0.65 * age), 0, 1)

    def fade_fn(co, nrm):
        return np.clip(age * (0.25 + 0.45 * n2(co, nrm) + 0.4 * streak(co, nrm, n2, 1.5)), 0, 1)

    for o in ([] if far else soft):
        paint(o, grime=grime_fn, special=fade_fn)
    part(soft, "cloth", pivot=(0, -s, 0), motion="flutter")
    socket("wall_l", "attach", (-xb, 0, 0), role="fixing")
    socket("wall_r", "attach", (xb, 0, 0), role="fixing")
