"""Broad stone staircase joining two plaza levels. Every step is a row of individually cut stones with a
bull-nosed, foot-dished tread bedded on a mortar core; the flight runs between solid coursed-ashlar cheek
walls (raked or stepped) on a rock-faced plinth, capped with weathered coping stones and stopped by square
piers with moulded caps. All the masonry is the town's cream limestone, laid in the same courses as the plaza's
ashlar walls (0.23-0.36 m high, stones 0.45-0.9 m long) with knocked corners and spalled arrises. The bottom piers carry small iron lanterns (or lamp-post sockets), an optional
mid landing is paved with flags and an optional forged-iron handrail with scrolled ends runs up the centre.
Origin: centre of the bottom step's front edge; the flight rises towards +Y and ends flush at z = rise.
"""
from tk import *
import bmesh
import colorsys
import numpy as np
from math import sin, cos, radians, degrees, pi, atan, hypot, ceil, floor, sqrt, exp

META = {"name": "Wide plaza stairs", "class": "structure", "mount": "ground", "focus": [1.1, 0.8, 1.15]}
PARAMS = {
    "width": {"default": 4.6, "min": 2.0, "max": 10.0, "desc": "overall width across the cheek walls at the top (m)"},
    "rise": {"default": 1.7, "min": 0.5, "max": 3.5, "desc": "total height climbed (m); top tread is flush at z = rise"},
    "steps": {"default": 10, "min": 3, "max": 24, "desc": "wished number of steps (clamped so risers stay 125-185 mm)"},
    "going": {"default": 0.34, "min": 0.28, "max": 0.42, "desc": "tread depth (m)"},
    "landing": {"default": False, "desc": "flagged landing half way up"},
    "landing_depth": {"default": 1.3, "min": 0.9, "max": 2.0, "desc": "depth of the mid landing (m)"},
    "cheek": {"default": "raked", "choices": ["raked", "stepped"], "desc": "cheek wall top: raked coping or stepped levels"},
    "cheek_height": {"default": 0.7, "min": 0.3, "max": 1.0, "desc": "cheek wall height above the nosing line (m)"},
    "flare": {"default": 0.0, "min": 0.0, "max": 1.2, "desc": "extra width per side at the bottom, walls sweep outwards (m; limited to what the run can carry)"},
    "handrail": {"default": True, "desc": "forged iron handrail up the centre"},
    "lanterns": {"default": "bottom", "choices": ["none", "bottom", "all"], "desc": "pier lanterns (piers without one get a lamp-post socket)"},
    "intensity": {"default": 15.0, "min": 8.0, "max": 80.0, "desc": "light per pier lantern (cd)"},
    "wear": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "foot wear: lanes dished up to ~30 mm, sagging rounded nosings, chips"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "moss / damp in corners, on copings and the wall foot"},
    "stone": {"default": "#c8bca3", "desc": "cheek wall ashlar colour (kept in the cream limestone family: hue pulled to it, saturation capped)"},
    "tread_stone": {"default": "#aea798", "desc": "step stone colour"},
    "trim": {"default": "#c9bea7", "desc": "coping and pier cap stone colour (kept in the cream limestone family like the ashlar)"},
}
VARIANTS = {
    "grand_landing": {"width": 7.6, "rise": 3.0, "steps": 18, "landing": True, "flare": 0.9, "lanterns": "all",
                      "cheek_height": 0.8, "seed": 4},
    "narrow_stepped": {"width": 2.6, "rise": 1.35, "steps": 8, "cheek": "stepped", "handrail": False, "moss": 0.8,
                       "stone": "#a9a294", "trim": "#b3ad9f", "tread_stone": "#8f8d88", "wear": 0.9, "seed": 11},
    "low_flared": {"width": 6.5, "rise": 0.7, "steps": 4, "flare": 1.0, "handrail": False, "cheek_height": 0.45,
                   "stone": "#cbbfa6", "trim": "#d1c5ab", "seed": 23},
    "stepped_tall": {"width": 4.0, "rise": 2.5, "steps": 15, "cheek": "stepped", "landing": True, "landing_depth": 1.1,
                     "cheek_height": 0.75, "lanterns": "all", "stone": "#c4b8a0", "trim": "#cbc0a9", "wear": 0.8, "seed": 31},
}

SQ2 = sqrt(2.0)
GAP = 0.007           # mortar joint
COURSES = (0.30, 0.25, 0.36, 0.27, 0.23, 0.33, 0.26)   # the town's limestone ashlar coursing, bottom up (repeats every 2 m)
CREAM_H, CREAM_S = 41.0 / 360.0, 0.185                 # hue / saturation of that limestone (#c8bca3 .. #d1c5ab)
_BC = ((1, 2), (0, 2), (0, 1))


# ------------------------------------------------------------------ small mesh helpers
def _prism_x(name, poly, x0, x1):
    """Closed prism: polygon [(y, z)] (CCW seen from +X) extruded from x0 to x1."""
    n = len(poly)
    verts = [(x0, y, z) for (y, z) in poly] + [(x1, y, z) for (y, z) in poly]
    faces = [[k, n + k, n + (k + 1) % n, (k + 1) % n] for k in range(n)]
    faces.append(list(range(n - 1, -1, -1)))
    faces.append([n + k for k in range(n)])
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _path_prism(name, sec, pts, xc):
    """Stone of cross-section sec [(x offset, offset along the in-plane normal)] following 2-3 (y, z) points (mitred)."""
    def nrm(a, b):
        dy, dz = b[0] - a[0], b[1] - a[1]
        L = hypot(dy, dz)
        return (-dz / L, dy / L)
    m, k = len(sec), len(pts)
    verts = []
    for i, (y, z) in enumerate(pts):
        if i == 0:
            mv = nrm(pts[0], pts[1])
        elif i == k - 1:
            mv = nrm(pts[-2], pts[-1])
        else:
            n1, n2 = nrm(pts[i - 1], pts[i]), nrm(pts[i], pts[i + 1])
            d = 1.0 + n1[0] * n2[0] + n1[1] * n2[1]
            mv = ((n1[0] + n2[0]) / d, (n1[1] + n2[1]) / d)
        for (a, b) in sec:
            verts.append((xc + a, y + b * mv[0], z + b * mv[1]))
    faces = []
    for i in range(k - 1):
        for j in range(m):
            j1 = (j + 1) % m
            faces.append([i * m + j, i * m + j1, (i + 1) * m + j1, (i + 1) * m + j])
    faces.append(list(range(m)))
    faces.append([(k - 1) * m + j for j in range(m - 1, -1, -1)])
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _offset_poly(pts, d):
    """Offset an open (y, z) polyline by d along its (mitred) upward normals."""
    def nrm(a, b):
        dy, dz = b[0] - a[0], b[1] - a[1]
        L = hypot(dy, dz)
        return (-dz / L, dy / L)
    out = []
    k = len(pts)
    for i, (y, z) in enumerate(pts):
        if i == 0:
            mv = nrm(pts[0], pts[1])
        elif i == k - 1:
            mv = nrm(pts[-2], pts[-1])
        else:
            n1, n2 = nrm(pts[i - 1], pts[i]), nrm(pts[i], pts[i + 1])
            dd = 1.0 + n1[0] * n2[0] + n1[1] * n2[1]
            mv = ((n1[0] + n2[0]) / dd, (n1[1] + n2[1]) / dd)
        out.append((y + d * mv[0], z + d * mv[1]))
    return out


def _tone(col, k, warm=0.0):
    """'#RRGGBB' scaled by k (and shifted a little warmer / cooler) -> '#RRGGBB'."""
    c = col.lstrip("#")
    r, g_, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    r, g_, b = r * k * (1.0 + warm), g_ * k, b * k * (1.0 - warm)
    return "#%02x%02x%02x" % tuple(int(clamp(v, 0, 255)) for v in (r, g_, b))


def _cream(col, pull=0.8, keep=0.3, smax=0.205):
    """Keep a stone colour inside the cream limestone family of the plaza walls: hue pulled towards it, any excess
    saturation mostly taken out, value kept - so no flight turns out orange next to the ashlar it joins."""
    c = col.lstrip("#")
    r, g_, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hh, ss, vv = colorsys.rgb_to_hsv(r, g_, b)
    if ss > 1e-4:
        hh = (hh + pull * (((CREAM_H - hh + 0.5) % 1.0) - 0.5)) % 1.0
    if ss > CREAM_S:
        ss = min(CREAM_S + keep * (ss - CREAM_S), smax)
    return "#%02x%02x%02x" % tuple(int(round(clamp(v, 0.0, 1.0) * 255)) for v in colorsys.hsv_to_rgb(hh, ss, vv))


def _levels(ztop, min_last=0.16):
    """Bed joint heights 0 .. ztop in the town's ashlar courses; the top course takes up a short remainder."""
    out, i = [0.0], 0
    while True:
        z1 = out[-1] + COURSES[i % len(COURSES)]
        if ztop - z1 < min_last:
            out.append(ztop)
            return out
        out.append(z1)
        i += 1


def _joints(a, b, r, hi=1.05):
    """Perpend joints a .. b for coping stones of believable, unequal lengths (about 0.5 - 1.05 m)."""
    L = b - a
    ns = max(1, int(round(L / r.u(0.7, 0.95))))
    if L > hi and ns < 2:
        ns = 2
    return [a] + [a + L * (q + r.u(-0.14, 0.14)) / ns for q in range(1, ns)] + [b]


def _densify(o, lim):
    """densify() with one edge limit per axis (x, y, z), so hidden bed / joint faces stay coarse."""
    for _ in range(8):
        bm = bmesh.new()
        bm.from_mesh(o.data)
        long_edges = []
        for e in bm.edges:
            d = e.verts[0].co - e.verts[1].co
            if d.length > lim[max(range(3), key=lambda i: abs(d[i]))]:
                long_edges.append(e)
        if not long_edges:
            bm.free()
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
        bm.to_mesh(o.data)
        bm.free()
    return o


def _box_spalls(size, r, n, axis, top_only=True, k=1.0):
    """n bites out of the arrises parallel to `axis` of a box centred on the origin, as _dress() spalls:
    (axis, position, half length, arris b, arris c, outward sign b, outward sign c, leg b, leg c)."""
    b, c = _BC[axis]
    L = size[axis]
    out = []
    for _ in range(n):
        rc = r.u(0.03, 0.075) * k
        if L < 2 * rc + 0.2:
            continue
        pos = r.u(-L / 2 + rc + 0.08, L / 2 - rc - 0.08)
        sb = 1.0 if r.chance(0.5) else -1.0
        sc = 1.0 if (top_only or r.chance(0.65)) else -1.0
        if any(abs(pos - q[1]) < rc + q[2] + 0.02 for q in out):
            continue
        out.append((axis, pos, rc, sb * size[b] / 2, sc * size[c] / 2, sb, sc,
                    min(r.u(0.012, 0.03) * k, 0.3 * size[b]), min(r.u(0.012, 0.034) * k, 0.3 * size[c])))
    return out


def _dress(o, r, knocks=0, spalls=(), kd=(0.01, 0.026), tilt=0.0, up=0.7):
    """Arris damage on a convex stone, before it is densified / bevelled (mesh coordinates).
    knocks: that many corners struck off along an oblique plane (the corner furthest out in a random direction;
    tilt = slope of a raked stone, up = share of knocks on the upper corners).
    spalls: bites out of axis-parallel arrises (see _box_spalls) - two facets meeting in a shallow valley."""
    if not knocks and not spalls:
        return o
    bm = bmesh.new()
    bm.from_mesh(o.data)

    def everything():
        return bm.verts[:] + bm.edges[:] + bm.faces[:]

    def near(q):
        v = min(bm.verts, key=lambda w: (w.co - q).length_squared)
        return v if (v.co - q).length < 2e-4 else None
    for (a, pos, rc, eb, ec, sb, sc, db, dc) in spalls:
        b, c = _BC[a]

        def vec(va, vb, vc):
            q = [0.0, 0.0, 0.0]
            q[a], q[b], q[c] = va, vb, vc
            return Vector(q)
        for q in (pos - rc, pos, pos + rc):
            bmesh.ops.bisect_plane(bm, geom=everything(), dist=1e-6, plane_co=vec(q, 0, 0), plane_no=vec(1, 0, 0))
        for end, sg in ((pos - rc, 1.0), (pos + rc, -1.0)):
            lo, hi = min(end, pos), max(end, pos)
            fs = [f_ for f_ in bm.faces if lo < f_.calc_center_median()[a] < hi]
            geom = set(fs)
            for f_ in fs:
                geom.update(f_.edges)
                geom.update(f_.verts)
            bmesh.ops.bisect_plane(bm, geom=list(geom), dist=1e-6, plane_co=vec(end, eb, ec),
                                   plane_no=vec(sg / rc, sb / db, sc / dc).normalized(), clear_outer=True)
        A, B = near(vec(pos - rc, eb, ec)), near(vec(pos + rc, eb, ec))
        T, S = near(vec(pos, eb - sb * db, ec)), near(vec(pos, eb, ec - sc * dc))
        if not any(q is None for q in (A, B, T, S)):
            try:
                bm.faces.new((A, S, T))
                bm.faces.new((B, T, S))
            except ValueError:
                pass
    ct, st = cos(tilt), sin(tilt)
    for _ in range(knocks):
        dx, dy, dz = (r.u(0.3, 1.0) * (1.0 if r.chance(0.5) else -1.0), r.u(0.3, 1.0) * (1.0 if r.chance(0.5) else -1.0),
                      r.u(0.3, 1.0) * (1.0 if r.chance(up) else -1.0))
        d = Vector((dx, dy * ct - dz * st, dy * st + dz * ct)).normalized()
        v = max(bm.verts, key=lambda w: w.co.dot(d))
        bmesh.ops.bisect_plane(bm, geom=everything(), dist=1e-6, plane_co=v.co - d * r.u(kd[0], kd[1]), plane_no=d, clear_outer=True)
        hole = [e for e in bm.edges if e.is_boundary]
        if hole:
            bmesh.ops.holes_fill(bm, edges=hole, sides=0)
    hole = [e for e in bm.edges if e.is_boundary]
    if hole:
        bmesh.ops.holes_fill(bm, edges=hole, sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(o.data)
    bm.free()
    return o


def _tool(o, r):
    """Every stone was dressed on the banker on its own: droving runs either along or across the block
    (the library texture's tooling sits at ~40 deg, so rotate the box UVs by +38 / -52 deg per stone)."""
    uv_box(o)
    realize(o)
    uv_transform(o, rot_deg=(38.0 if r.chance(0.55) else -52.0) + r.u(-5.0, 5.0), offset=(r.u(0.0, 1.0), r.u(0.0, 1.0)))
    uv_keep(o)


def _face_block(name, xa, xb, za, zb, y, c=0.008, depth=0.022):
    """Ashlar facing stone seen from +Y only: dressed face flush at y, chamfered arrises, short joint returns."""
    v = [(xa + c, y, za + c), (xb - c, y, za + c), (xb - c, y, zb - c), (xa + c, y, zb - c),
         (xa, y - c, za), (xb, y - c, za), (xb, y - c, zb), (xa, y - c, zb),
         (xa, y - depth, za), (xb, y - depth, za), (xb, y - depth, zb), (xa, y - depth, zb)]
    f = [[3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
         [4, 5, 9, 8], [5, 6, 10, 9], [6, 7, 11, 10], [7, 4, 8, 11]]
    return mesh_object(name, v, f)


def _stone_row(name, P0, P1, cw, tw, zmin, closing, x0, x1, dx, lanef, chips, und, xscale, fine=True):
    """One step stone swept along X. P0 = fresh profile [(y, z)] (riser foot -> nosing -> tread back), P1 = the same
    points fully worn (sagged tread, rounded-off nosing); lanef(x) in 0..1 blends between them so the nosing line
    dips in the walking lanes. chips = [(x, half width, depth)] spalls bitten out of the nosing; cw = per point
    (push back, drop) chip weights, tw = top-surface undulation weights. closing = hidden points closing the end
    caps. Hidden back / bed faces are not built."""
    c = 0.005
    xs = [x0 + c] if fine else [x0]
    hi_ = x1 - c if fine else x1
    nseg = max(1, int(ceil((x1 - x0 - 2 * c) / dx)))
    mids = [x0 + c + (x1 - x0 - 2 * c) * j / nseg for j in range(1, nseg)]
    for (xc, rc, dc) in chips:
        mids += [xc - rc, xc - 0.5 * rc, xc, xc + 0.5 * rc, xc + rc] if fine else [xc - rc, xc, xc + rc]
    for x in sorted(mids):
        if x - xs[-1] > 0.012 and hi_ - x > 0.012:
            xs.append(x)
    xs = ([x0] + xs + [x1 - c, x1]) if fine else (xs + [x1])
    npt = len(P0)
    m = npt + len(closing)
    verts = []
    for j, x in enumerate(xs):
        end = fine and (j == 0 or j == len(xs) - 1)
        w = lanef(x)
        chp = 0.0
        for (xc, rc, dc) in chips:
            chp += dc * max(0.0, 1.0 - ((x - xc) / rc) ** 2) ** 0.8
        u1 = und[0] * sin(x * und[1] + und[2]) + und[3] * sin(x * und[4] + und[5])
        ring = []
        for k in range(npt):
            y = P0[k][0] + w * (P1[k][0] - P0[k][0])
            z = P0[k][1] + w * (P1[k][1] - P0[k][1]) + tw[k] * u1
            if chp > 0.0 and (cw[k][0] or cw[k][1]):
                y += cw[k][0] * chp
                z = max(z - cw[k][1] * chp, zmin)
            ring.append((y, z))
        ring += list(closing)
        cy = sum(q[0] for q in ring) / m
        cz = sum(q[1] for q in ring) / m
        for (y, z) in ring:
            if end:
                d = hypot(cy - y, cz - z)
                if d > 1e-6:
                    y += (cy - y) / d * c
                    z += (cz - z) / d * c
            verts.append((x * xscale(y), y, max(z, 0.0)))
    faces = []
    for j in range(len(xs) - 1):
        for k in range(npt - 1):
            faces.append([j * m + k, (j + 1) * m + k, (j + 1) * m + k + 1, j * m + k + 1])
    faces.append([k for k in range(m)])
    last = (len(xs) - 1) * m
    faces.append([last + k for k in range(m - 1, -1, -1)])
    o = mesh_object(name, verts, faces)
    smooth(o, 40)
    return o


def _scroll(E, ang_deg, r1, r0, turns, ccw=True, npts=26):
    """Volute in the YZ plane starting at E = (y, z) travelling in direction ang_deg, curling to radius r0."""
    sg = 1.0 if ccw else -1.0
    a0 = radians(ang_deg) - sg * pi / 2
    cy, cz = E[0] - r1 * cos(a0), E[1] - r1 * sin(a0)
    pts = []
    for i in range(npts + 1):
        tt = i / npts
        a = a0 + sg * 2 * pi * turns * tt
        r = r1 + (r0 - r1) * tt
        pts.append((cy + r * cos(a), cz + r * sin(a)))
    return pts


def _lantern(iron, glass, flame, seg=12):
    """Small four-sided pier lantern on a turned stem; origin at the centre of its base plate. ~1.05 m tall."""
    o = []
    o.append(box("l_plate", (0.2, 0.2, 0.012), loc=(0, 0, 0.006), bevel=0.003))
    bt = cylinder("l_bolt", 0.009, 0.007, loc=(0, 0, 0.0035), verts=6)
    apply_transform(bt)
    o.append(instances_to_mesh("l_bolts", bt, [trs((sx * 0.075, sy * 0.075, 0.012), (0, 0, 17.0 * (sx + 2 * sy))) for sx in (-1, 1) for sy in (-1, 1)],
                               keep_base=False))
    o.append(lathe("l_stem", [(0.062, 0.012), (0.064, 0.02), (0.046, 0.032), (0.03, 0.05), (0.024, 0.09), (0.024, 0.108),
                              (0.036, 0.122), (0.041, 0.138), (0.036, 0.154), (0.023, 0.17), (0.019, 0.34), (0.028, 0.356),
                              (0.06, 0.37), (0.066, 0.378), (0.03, 0.392), (0.021, 0.41), (0.018, 0.45),
                              (0.027, 0.468), (0.05, 0.488), (0.056, 0.50), (0.0, 0.50)], segments=seg))
    zb, zc = 0.50, 0.81
    b0, b1 = 0.074, 0.116
    o.append(box("l_tray", (2 * b0 + 0.032, 2 * b0 + 0.032, 0.016), loc=(0, 0, zb + 0.008), bevel=0.003))
    bar = rounded_rect(0.014, 0.014, 0.003, 1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            o.append(sweep("l_bar", [(sx * b0, sy * b0, zb + 0.014), (sx * b1, sy * b1, zc)], bar, smooth_path=False))
    o.append(shape_extrude("l_top", rounded_rect(2 * b1 + 0.03, 2 * b1 + 0.03, 0.004, 1),
                           [rounded_rect(2 * b1 - 0.022, 2 * b1 - 0.022, 0.002, 1)], depth=0.014, loc=(0, 0, zc - 0.006)))
    rr = b1 + 0.04
    roof = lathe("l_roof", [(rr * SQ2, zc + 0.008), (rr * SQ2, zc + 0.02), ((rr - 0.012) * SQ2, zc + 0.03),
                            (b1 * 0.62 * SQ2, zc + 0.075), (0.052 * SQ2, zc + 0.115), (0.04 * SQ2, zc + 0.125),
                            (0.04 * SQ2, zc + 0.15), (0.058 * SQ2, zc + 0.156), (0.058 * SQ2, zc + 0.166),
                            (0.018 * SQ2, zc + 0.19), (0.0, zc + 0.19)], segments=4, rot=(0, 0, 45), smooth_angle=30)
    o.append(roof)
    o.append(lathe("l_finial", [(0.012, zc + 0.185), (0.016, zc + 0.195), (0.008, zc + 0.205), (0.019, zc + 0.222),
                                (0.02, zc + 0.232), (0.012, zc + 0.246), (0.004, zc + 0.262), (0.0, zc + 0.275)], segments=10))
    o.append(cylinder("l_burner", 0.012, 0.085, loc=(0, 0, zb + 0.016 + 0.0425), verts=8))
    o.append(cylinder("l_gallery", 0.02, 0.012, loc=(0, 0, zb + 0.105), verts=10))
    for q in o:
        assign(q, iron)
    # panes held just inside the corner bars
    gv, gf = [], []
    e = 0.004
    for k in range(4):
        a = radians(90 * k)
        ca, sa = cos(a), sin(a)
        base = len(gv)
        for (x, y, z) in ((-b0 + e, -b0 + e, zb + 0.016), (b0 - e, -b0 + e, zb + 0.016), (b1 - e, -b1 + e, zc - 0.004),
                          (-b1 + e, -b1 + e, zc - 0.004)):
            gv.append((x * ca - y * sa, x * sa + y * ca, z))
        gf.append([base, base + 1, base + 2, base + 3])
    panes = mesh_object("l_glass", gv, gf)
    assign(panes, glass)
    mantle = sphere("l_mantle", 0.021, loc=(0, 0, zb + 0.15), segments=10, rings=6, scale=(1, 1, 1.7))
    assign(mantle, flame)
    return o, panes, mantle


# ------------------------------------------------------------------ the asset
def build(p, rng):
    W, rise = p.width, p.rise
    n = int(round(p.steps))
    n = max(int(ceil(rise / 0.185 - 1e-6)), min(n, int(floor(rise / 0.125 + 1e-6))))
    n = max(n, 2)
    h = rise / n
    g = p.going
    s = h / g
    theta = degrees(atan(s))
    wear, moss = p.wear, p.moss
    nose = 0.032
    nt = min(0.055, h * 0.36)                        # nosing thickness
    rr = min(0.02, nt * 0.4)                         # fresh nosing round
    pw = clamp(0.36 + 0.02 * W + 0.045 * rise, 0.42, min(0.6, 0.42 + 0.06 * (W - 2.0)))   # pier die (square)
    capo = 0.045                                     # cap / plinth projection
    t = pw - 0.14                                    # cheek wall thickness
    xw = W / 2 - capo - pw / 2                       # wall + pier centre line
    xin = xw - t / 2                                 # inner wall face
    has_land = bool(p.landing) and n >= 5
    m_land = (n - 1) // 2 if has_land else -1
    Ld = p.landing_depth
    top_d = max(g, pw + capo + 0.06)
    yf, dep = [0.0], []
    for i in range(n):
        d = top_d if i == n - 1 else (Ld if i == m_land else g)
        dep.append(d)
        yf.append(yf[-1] + d)
    Y = yf[n]
    zt = [(i + 1) * h for i in range(n)]
    lod = clamp(sqrt(W * Y / 18.0), 1.0, 2.0)         # coarser tessellation on very large flights (triangle budget)
    raked = p.cheek == "raked"
    ch = p.cheek_height
    ya = pw                       # rear face of the bottom pier
    yb = Y - capo - pw            # front face of the top pier
    top_pier = yb - ya >= 1.0     # short flights: the cheek runs to the back without a top pier
    if not top_pier:
        yb = Y
    emb = 0.03 if top_pier else 0.0
    yw = Y - capo - 0.006         # face of the retaining wall under the top step (= rear face of the top piers)
    cy0, cy1 = pw / 2, Y - capo - pw / 2 - 0.006
    short = n <= 5 or not top_pier                    # low flights: a level plinth-block cheek instead of a rake
    rail = bool(p.handrail) and 2 * xin > 1.5
    lean = lod >= 1.3 or Y * rise > 12.0              # big flights / tall walls: leaner stones (triangle budget)

    # flare: walls sweep outwards towards the bottom (never more than the run can carry; piers stay square)
    Lz = min(Y, max(0.6 * Y, 4 * p.flare))
    fl = max(0.0, min(p.flare, 0.38 * (Lz - ya)))

    fpow = 1.0 if short else 2.0                      # low flights: straight splayed wing blocks, otherwise a sweep

    def f(y):
        tt = clamp((Lz - max(y, ya)) / max(Lz - ya, 1e-6), 0.0, 1.0)
        return fl * tt ** fpow

    def fv(y):
        tt = np.clip((Lz - np.maximum(y, ya)) / max(Lz - ya, 1e-6), 0.0, 1.0)
        return fl * tt ** fpow

    stone_c, trim_c = _cream(p.stone), _cream(p.trim)     # one hue family with the plaza ashlar, whatever is asked for
    m_tread = mat("stone_smooth", _cream(p.tread_stone), rough=1.3)
    m_ash = mat("stone_smooth", stone_c, rough=1.55)
    m_ash2 = mat("stone_smooth", _tone(stone_c, 0.925, -0.008), rough=1.6)   # the odd stone from a slightly greyer bed
    m_rough = mat("stone_rough", stone_c)
    m_trim = mat("stone_smooth", trim_c, rough=1.45)
    m_mortar = mat("mortar", "#5a5449")
    m_iron = mat("iron_wrought")

    # ---------------------------------------------------------------- cheek wall top line T and coping bed U
    ct = 0.12                     # raked coping thickness
    cs = 0.11                     # level coping slab thickness
    secs = []
    slab = short or not raked
    if short:
        ztop = rise + max(0.12, 0.45 * ch)
        T = [(ya - 0.03, ztop), (yb + emb, ztop)]
        secs.append((ya - 0.03, yb + emb, ztop))
        U = [(y, z - cs) for (y, z) in T]
    elif raked:
        kl = 0.22                 # level kneeler stone against each pier

        def T1(y):
            return zt[0] + ch + (y - yf[0]) * s
        if has_land:
            zL = zt[m_land] + ch

            def T2(y):
                return zt[m_land + 1] + ch + (y - yf[m_land + 1]) * s
            if yf[m_land] - (ya + kl) > 0.35:
                T = [(ya - 0.03, T1(ya + kl)), (ya + kl, T1(ya + kl)), (yf[m_land], zL), (yf[m_land + 1] - g, zL)]
            else:
                T = [(ya - 0.03, zL), (yf[m_land + 1] - g, zL)]
        else:
            T2 = T1
            T = [(ya - 0.03, T1(ya + kl)), (ya + kl, T1(ya + kl))]
        ykt = yb + emb - kl - 0.03
        if ykt - T[-1][0] > 0.35:
            T += [(ykt, T2(ykt)), (yb + emb, T2(ykt))]
        else:
            T.append((yb + emb, T2(yb + emb)))
        U = _offset_poly(T, -ct)
    else:
        k = 2 if n <= 7 else (3 if n <= 15 else 4)
        cl = max(0.3, ch - 0.5 * (k - 1) * h)
        T = []
        a = 0
        y0s = ya - 0.03
        while a < n:
            b = min(n, a + k)
            while b < n and (yf[b] + nose - y0s < 0.45):
                b += 1
            if n - b <= 1 or yf[min(b, n - 1)] + nose > yb - 0.3:
                b = n
            ztop = zt[b - 1] + cl
            y1s = yf[b] + nose if b < n else yb + emb
            T += [(y0s, ztop), (y1s, ztop)]
            secs.append((y0s, y1s, ztop))
            y0s = y1s
            a = b
        U = [(y, z - cs) for (y, z) in T]

    # single-valued lookups (vertical jumps nudged)
    def _arr(P):
        ys, zs = [P[0][0] - 2.0], [P[0][1] - 2.0 * ((P[1][1] - P[0][1]) / max(P[1][0] - P[0][0], 1e-6))]
        for (y, z) in P:
            ys.append(max(y, ys[-1] + 1e-7))
            zs.append(z)
        ys.append(ys[-1] + 2.0)
        zs.append(zs[-1] + 2.0 * ((P[-1][1] - P[-2][1]) / max(P[-1][0] - P[-2][0], 1e-6)))
        return np.array(ys), np.array(zs)
    Uy, Uz = _arr(U)

    def Uev(y):
        return float(np.interp(y, Uy, Uz))
    Umax = max(z for _, z in U)
    die_b = T[0][1] + 0.22        # pier dies stand a good hand above the coping that dies into them
    die_t = T[-1][1] + 0.22

    # ---------------------------------------------------------------- weathering (dirt only where it collects)
    GMAX = 0.42                   # the stone shader turns anything denser into black cloud blotches
    seed = int(p.seed)
    nz_big = noise_mask(0.55, seed=seed + 3, lo=0.3, hi=0.7)
    nz_mid = noise_mask(0.16, seed=seed + 5, lo=0.35, hi=0.7)
    nz_fine = noise_mask(0.06, seed=seed + 7, lo=0.3, hi=0.75)
    nz_run = noise_mask(0.075, seed=seed + 11, lo=0.3, hi=0.78)     # rain runs: sampled on the plan position only
    nz_len = noise_mask(0.3, seed=seed + 13, lo=0.25, hi=0.8)
    nz_lich = noise_mask(0.4, seed=seed + 17, lo=0.42, hi=0.75)     # lichen colonies: broad, the shader breaks them up
    PLAN = np.array([1.0, 1.0, 0.025])
    yf_arr = np.array(yf)

    def wall_paint(o, g0, zcap=None, inner=True, reach=0.34):
        """Rain runs of uneven length below the coping / cap drip, splash dirt and green at the wall foot (ground
        outside, treads inside) and in the re-entrant corners where the cheek dies into a pier."""
        def geo(co, nr):
            y, z = co[:, 1], co[:, 2]
            under = np.maximum((np.interp(y, Uy, Uz) if zcap is None else zcap) - z, 0.0)
            pl = co * PLAN
            drip = np.clip(1.0 - under / (reach * (0.45 + 1.5 * nz_len(pl, nr))), 0.0, 1.0) ** 1.3 * (0.25 + 0.75 * nz_run(pl, nr))
            foot = np.clip(1.0 - z / 0.4, 0.0, 1.0)
            cor = np.zeros_like(z)
            if inner:
                tz = np.clip(np.searchsorted(yf_arr, y, side="right"), 0, n) * h
                ins = (np.abs(co[:, 0]) - fv(y)) < xw
                foot = np.where(ins, np.clip(1.0 - (z - tz) / 0.16, 0.0, 1.0) * (z > tz - 0.05), foot)
                dj_ = np.abs(y - ya)
                if top_pier:
                    dj_ = np.minimum(dj_, np.abs(y - yb))
                cor = np.clip(1.0 - dj_ / 0.17, 0.0, 1.0) * np.clip(1.0 - (z - np.where(ins, tz, 0.0)) / 1.1, 0.2, 1.0)
            return drip, foot, cor

        def gr(co, nr):
            drip, foot, cor = geo(co, nr)
            mid = 0.4 + 0.6 * nz_mid(co, nr)
            v = g0 + (0.3 + 0.14 * moss) * foot * mid + 0.36 * drip + 0.24 * cor * mid
            return np.clip(v, 0.0, GMAX)

        def sp(co, nr):
            drip, foot, cor = geo(co, nr)
            mk = 0.45 + 0.55 * moss if moss > 0.02 else 0.0
            v = mk * (1.15 * foot ** 1.2 * (0.5 + 0.5 * nz_mid(co, nr)) + 0.85 * cor * (0.35 + 0.65 * nz_big(co, nr))) + 0.55 * mk * drip * nz_lich(co, nr)
            return np.clip(v, 0.0, 0.72)
        paint(o, grime=gr, special=sp)

    def top_paint(o, g0, soot=None):
        """Copings / caps: rain-washed tops, streaky sides; a soft halo + one run-off streak below a lantern foot."""
        def gr(co, nr):
            up = np.clip(nr[:, 2], 0.0, 1.0)
            v = g0 + 0.3 * (1.0 - up) * nz_fine(co * np.array([1.0, 1.0, 0.2]), nr) + (0.1 + 0.12 * moss) * up * nz_big(co, nr)
            if soot is not None:
                dx_, dy_ = co[:, 0] - soot[0], co[:, 1] - soot[1]
                ux, uy = cos(soot[2]), sin(soot[2])
                al, ac = dx_ * ux + dy_ * uy, -dx_ * uy + dy_ * ux
                ring = np.clip(1.0 - (np.hypot(dx_, dy_) - 0.1) / 0.14, 0.0, 1.0) * up
                streak = (np.exp(-(ac / (0.035 + 0.12 * np.clip(al - 0.1, 0.0, 1.0))) ** 2) * np.clip((al - 0.07) / 0.04, 0.0, 1.0)
                          * (0.65 + 0.35 * nz_fine(co, nr)))
                v = v + 0.13 * ring + 0.3 * streak
                return np.clip(v, 0.0, GMAX + 0.03)
            return np.clip(v, 0.0, GMAX)

        def sp(co, nr):                               # lichen rosettes on the rain-fed upper faces
            up = np.clip(nr[:, 2], 0.0, 1.0)
            return np.clip((0.5 + 0.5 * moss if moss > 0.02 else 0.0) * up * nz_lich(co, nr) * (0.35 + 0.65 * nz_big(co, nr)), 0.0, 0.36)
        paint(o, grime=gr, special=sp)

    def shear(o, sgn):
        if fl > 1e-4:
            apply_transform(o)
            transform_verts(o, lambda co: Vector((co.x + sgn * f(co.y), co.y, co.z)))

    def ash(r):
        return m_ash2 if r.chance(0.3) else m_ash

    # ---------------------------------------------------------------- cheek walls
    absorb = 0.085                # courses run up to this much higher under the coping instead of feather edges
    stage = 0.035 if Umax > 2.0 else 0.0              # tall walls stand on a thicker three-course base outside

    def wall(sgn, r):
        xc = sgn * xw
        x0, x1 = xc - t / 2, xc + t / 2
        # dark mortar just behind both faces (seen only in the joints)
        ye = yb + (0.05 if top_pier else -0.004)
        nsg = max(1, int(ceil((ye - ya + 0.05) / 0.4)))
        ys_ = [ya - 0.05 + (ye - ya + 0.05) * q / nsg for q in range(nsg + 1)]
        for (y, _) in U:
            if ya - 0.05 < y < ye:
                ys_ += [y - 1e-5, y + 1e-5]
        ys_ = sorted(ys_)
        cv_, cf_ = [], []
        for xs_, flip in ((x0 + 0.012, True), (x1 - 0.012, False)):
            b0 = len(cv_)
            for y in ys_:
                cv_ += [(xs_, y, 0.0), (xs_, y, Uev(y) - 0.003)]
            for j in range(len(ys_) - 1):
                if ys_[j + 1] - ys_[j] < 1e-4:
                    continue
                q = [b0 + 2 * j, b0 + 2 * j + 2, b0 + 2 * j + 3, b0 + 2 * j + 1]
                cf_.append(q[::-1] if flip else q)
        core = mesh_object("cheek_core", cv_, cf_)
        assign(core, m_mortar)
        shear(core, sgn)
        paint(core, grime=0.3)
        z0 = 0.0
        ci = 0
        while z0 + absorb < Umax:
            z1 = z0 + COURSES[ci % len(COURSES)]
            if Umax - z1 < 0.02:
                z1 = Umax + 0.05
            la, lb = z0 + absorb, z1 + absorb

            def top(y):
                u = Uev(y)
                if u < la:
                    return z0
                return u if u < lb else z1
            # mandatory joints: wherever the course top is discontinuous or kinked
            J = [ya, yb]
            for i in range(len(U) - 1):
                (y_a, z_a), (y_b, z_b) = U[i], U[i + 1]
                if y_b - y_a > 1e-6:
                    for L in (la, lb):
                        if (z_a - L) * (z_b - L) < 0:
                            J.append(y_a + (L - z_a) / (z_b - z_a) * (y_b - y_a))
            for (y, _) in U[1:-1]:
                lo, hi = sorted((Uev(y - 1e-5), Uev(y + 1e-5)))
                if lo < lb and hi > la:
                    J.append(y)
            J = sorted(q for q in J if ya - 1e-6 <= q <= yb + 1e-6)
            for jx in range(len(J) - 1):
                sa, sb = J[jx], J[jx + 1]
                if sb - sa < 0.03:
                    continue
                if top(0.5 * (sa + sb)) <= z0 + 1e-6:
                    continue
                y = sa
                first = True
                while y < sb - 1e-4:
                    L = r.u(0.46, 0.86) * (0.6 if (first and (ci + jx) % 2) else 1.0)
                    first = False
                    y1 = min(sb, y + L)
                    if sb - y1 < 0.24:
                        y1 = sb
                    ha, hb = top(y + 1e-5), top(y1 - 1e-5)
                    plinth = ci == 0
                    out = (0.03 if plinth else 0.0) + (stage if ci < 3 else 0.0)   # outside face only
                    px0, px1 = (x0, x1 + out) if sgn > 0 else (x0 - out, x1)
                    dj = r.u(-0.003, 0.003)
                    poly = [(y + GAP / 2, z0 + (GAP / 2 if ci else 0.0)), (y1 - GAP / 2, z0 + (GAP / 2 if ci else 0.0)),
                            (y1 - GAP / 2, hb - GAP / 2), (y + GAP / 2, ha - GAP / 2)]
                    blk = _prism_x("cheek_block", poly, px0 + dj, px1 + dj)
                    if plinth:
                        densify(blk, min(0.34, 0.19 * lod))
                        bevel(blk, 0.018, segments=2, angle=25)
                        displace_noise(blk, strength=0.007, scale=0.11, seed=r.u(0, 99))
                        transform_verts(blk, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
                        assign(blk, m_rough)
                    else:
                        if r.chance(0.3 if lean else 0.45):
                            _dress(blk, r, knocks=1 if r.chance(0.75) else 2, kd=(0.008, 0.022))
                        # mask resolution where the weather shows: under the coping drip and at the foot
                        fine_ = max(ha, hb) > Uev(0.5 * (y + y1)) - 0.8 or z0 < 0.6
                        dn = (0.3 if lean else 0.19) if fine_ else 0.42
                        _densify(blk, (t + 0.03, dn, dn))
                        bevel(blk, 0.007, segments=1, angle=25)
                        assign(blk, ash(r))
                        _tool(blk, r)
                    shear(blk, sgn)
                    wall_paint(blk, r.u(0.0, 0.15))
                    y = y1
            z0 = z1
            ci += 1
        # copings: separate stones on a recessed dark bed
        cw = t + 0.09
        c = cw / 2
        if not slab:
            sec = [(-c, -ct), (c, -ct), (c, -0.042), (c - 0.022, -0.013), (c * 0.5, 0.0), (-c * 0.5, 0.0),
                   (-c + 0.022, -0.013), (-c, -0.042)]
            bed = _path_prism("coping_bed", [(-c + 0.014, -ct - 0.004), (c - 0.014, -ct - 0.004), (c - 0.014, -0.03), (-c + 0.014, -0.03)],
                              T, xc)
            densify(bed, 0.4)
            assign(bed, m_mortar)
            shear(bed, sgn)
            paint(bed, grime=0.3)
            segL = [hypot(T[i + 1][0] - T[i][0], T[i + 1][1] - T[i][1]) for i in range(len(T) - 1)]
            cum = [0.0]
            for L in segL:
                cum.append(cum[-1] + L)

            def pt(sv):
                for i in range(len(segL)):
                    if sv <= cum[i + 1] + 1e-9 or i == len(segL) - 1:
                        q = (sv - cum[i]) / segL[i]
                        return (T[i][0] + (T[i + 1][0] - T[i][0]) * q, T[i][1] + (T[i + 1][1] - T[i][1]) * q)
            stones = []                               # point lists; kneelers / corner stones carry the kink in one stone
            prev = 0.0
            nv = len(T)
            for vi in range(1, nv - 1):
                half = min(0.32, 0.45 * segL[vi - 1], 0.45 * segL[vi])
                lo_, hi_ = cum[vi] - half, cum[vi] + half
                if vi == 1 and segL[0] < 0.5:
                    lo_ = 0.0
                if vi == nv - 2 and segL[-1] < 0.5:
                    hi_ = cum[-1]
                if lo_ - prev > 0.02:
                    ns = max(1, int(round((lo_ - prev) / r.u(0.74, 0.96))))
                    for q in range(ns):
                        stones.append([pt(prev + (lo_ - prev) * q / ns + GAP / 2), pt(prev + (lo_ - prev) * (q + 1) / ns - GAP / 2)])
                stones.append([pt(lo_ + GAP / 2), T[vi], pt(hi_ - GAP / 2)])
                prev = hi_
            if cum[-1] - prev > 0.02:
                ns = max(1, int(round((cum[-1] - prev) / r.u(0.74, 0.96))))
                for q in range(ns):
                    stones.append([pt(prev + (cum[-1] - prev) * q / ns + GAP / 2), pt(prev + (cum[-1] - prev) * (q + 1) / ns - GAP / 2)])
            for sp in stones:
                dz = r.u(-0.002, 0.002)
                cp = _path_prism("coping", sec, [(q[0], q[1] + dz) for q in sp], xc + r.u(-0.003, 0.003))
                if len(sp) == 2:
                    _dress(cp, r, knocks=int(r.u(0.0, 2.5)), kd=(0.008, 0.02),
                           tilt=atan((sp[1][1] - sp[0][1]) / max(sp[1][0] - sp[0][0], 1e-6)), up=0.85)
                densify(cp, 0.3 if lean else 0.22)
                bevel(cp, 0.005, segments=1, angle=50)
                smooth(cp, 38)
                assign(cp, m_trim)
                _tool(cp, r)
                shear(cp, sgn)
                top_paint(cp, r.u(0.0, 0.16))
        else:
            for j, (y0s, y1s, ztop) in enumerate(secs):
                ys0 = y0s - (0.05 if j > 0 else 0.0)
                bed = box("coping_bed", (cw - 0.03, y1s - ys0, cs - 0.02), loc=(xc, (ys0 + y1s) / 2, ztop - cs / 2 - 0.012))
                apply_transform(bed)
                densify(bed, 0.4)
                assign(bed, m_mortar)
                shear(bed, sgn)
                paint(bed, grime=0.3)
                cuts = _joints(ys0, y1s, r)
                for q in range(len(cuts) - 1):
                    qa, qb = cuts[q] + GAP / 2, cuts[q + 1] - GAP / 2
                    sz_ = (cw + r.u(-0.004, 0.004), qb - qa, cs)
                    cp = box("coping", sz_)
                    _dress(cp, r, knocks=int(r.u(0.2, 2.9)), spalls=_box_spalls(sz_, r, int(r.u(0.5, 3.4)), 1, top_only=r.chance(0.8), k=1.35),
                           kd=(0.01, 0.026), up=0.85)
                    place(cp, loc=(xc, (qa + qb) / 2, ztop - cs / 2 + r.u(-0.002, 0.002)), rot=(0, 0, r.u(-0.2, 0.2)))
                    apply_transform(cp)
                    dn = 0.2 if lean else 0.13
                    _densify(cp, (dn, dn, 0.2))
                    # weathered top: a slight fall to both drips, never dead flat
                    transform_verts(cp, lambda co, zq=ztop - 0.006: Vector((co.x, co.y, co.z - (0.007 * ((co.x - xc) / c) ** 2
                                                                                                if co.z > zq else 0.0))))
                    bevel(cp, 0.009, segments=2, angle=30)
                    assign(cp, m_trim)
                    _tool(cp, r)
                    shear(cp, sgn)
                    top_paint(cp, r.u(0.0, 0.16))

    # ---------------------------------------------------------------- piers
    def pier(sgn, cy, ztop, r, has_lantern):
        cx = sgn * (xw + f(cy))
        core = box("pier_core", (pw - 0.024, pw - 0.024, ztop - 0.004), loc=(cx, cy, (ztop - 0.004) / 2))
        assign(core, m_mortar)
        paint(core, grime=0.3)
        lv = _levels(ztop)        # the die courses with the cheek walls
        k0 = int(r.u(0, 3))
        a = pw / 2
        for ci in range(len(lv) - 1):
            z, hh = lv[ci], lv[ci + 1] - lv[ci]
            if ci == 0:
                e = 2 * capo * 0.8
                b = box("pier_block", (pw + e, pw + e, hh - GAP / 2),
                        loc=(cx + r.u(-0.003, 0.003), cy + r.u(-0.003, 0.003), z + hh / 2),
                        rot=(0, 0, r.u(-0.5, 0.5)))
                apply_transform(b)
                transform_verts(b, lambda co: Vector((co.x, co.y, max(co.z - GAP / 4, 0.0))))
                densify(b, min(0.34, 0.2 * lod))
                bevel(b, 0.02, segments=2, angle=25)
                displace_noise(b, strength=0.006, scale=0.11, seed=r.u(0, 99))
                transform_verts(b, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
                assign(b, m_rough)
                wall_paint(b, r.u(0.0, 0.2), zcap=ztop, reach=0.3)
                continue
            # bonded die: one through stone, then two stones with the perpend across X, then across Y
            kind = (ci + k0) % 3
            if kind == 0:
                cells = [(-a, a, -a, a)]
            else:
                q = -a + pw * r.u(0.37, 0.63)
                cells = [(-a, q - GAP / 2, -a, a), (q + GAP / 2, a, -a, a)] if kind == 1 else [(-a, a, -a, q - GAP / 2), (-a, a, q + GAP / 2, a)]
            fine_ = lv[ci + 1] > ztop - 0.65 or z < 0.6
            for (xa_, xb_, ya_, yb_) in cells:
                sz_ = (xb_ - xa_, yb_ - ya_, hh - GAP)
                b = box("pier_block", sz_)
                _dress(b, r, knocks=int(r.u(0.0, 2.3 if lod < 1.3 else 1.6)),
                       spalls=_box_spalls(sz_, r, int(r.u(0.0, 2.0)), 0 if r.chance(0.5) else 1, top_only=False), kd=(0.008, 0.024), up=0.6)
                place(b, loc=(cx + (xa_ + xb_) / 2 + r.u(-0.002, 0.002), cy + (ya_ + yb_) / 2 + r.u(-0.002, 0.002), z + hh / 2),
                      rot=(0, 0, r.u(-0.35, 0.35)))
                apply_transform(b)
                densify(b, (0.19 if lod < 1.3 else 0.26) if fine_ else 0.34)
                bevel(b, 0.008, segments=1, angle=25)
                assign(b, m_ash2 if r.chance(0.3) else m_ash)
                _tool(b, r)
                wall_paint(b, r.u(0.0, 0.15), zcap=ztop, reach=0.3)
        prof = [(a - 0.01, -0.002), (a + 0.012, 0.0), (a + 0.012, 0.022), (a + 0.02, 0.034), (a + 0.034, 0.05), (a + capo, 0.06),
                (a + capo, 0.118), (a + capo - 0.012, 0.13), (a + 0.006, 0.145), (0.125, 0.196), (0.125, 0.212), (0.0, 0.212)]
        cap = lathe("pier_cap", [(q * SQ2, zz) for (q, zz) in prof], segments=4, loc=(cx, cy, ztop), rot=(0, 0, 45), smooth_angle=28)
        apply_transform(cap)
        densify(cap, (0.085 if lod < 1.3 else 0.12) if has_lantern else 0.3)
        assign(cap, m_trim)
        _tool(cap, r)
        top_paint(cap, r.u(0.04, 0.15), soot=(cx, cy, radians(90.0 * int(r.u(0, 4)) + r.u(-14, 14))) if has_lantern else None)
        return (cx, cy, ztop + 0.212)

    tops = {}
    for sgn, sn in ((-1, "l"), (1, "r")):
        wall(sgn, rng.fork("wall" + sn))
        tops["b" + sn] = pier(sgn, cy0, die_b, rng.fork("pier_b" + sn), p.lanterns != "none")
        if top_pier:
            tops["t" + sn] = pier(sgn, cy1, die_t, rng.fork("pier_t" + sn), p.lanterns == "all")

    # ---------------------------------------------------------------- steps
    D = 0.007 + 0.024 * wear      # depth of the dish worn into the walking lanes
    rs = rng.fork("steps")

    def lanes(hw):
        """(centre x, sigma, strength) of the walked lanes: either side of the handrail, or up the middle."""
        if rail:
            ln = [(-0.46, 0.3, 1.0), (0.46, 0.3, 1.0)]
            if hw > 1.7:
                xo = 0.46 + 0.6 * (hw - 0.46)
                ln += [(-xo, 0.36, 0.5), (xo, 0.36, 0.5)]
        else:
            ln = [(0.0, min(0.42, 0.42 * hw), 1.0)]
            if hw > 1.6:
                ln += [(-0.52 * hw, 0.36, 0.55), (0.52 * hw, 0.36, 0.55)]
        return ln

    def lane_np(x, ln):
        v = np.zeros_like(x)
        for (xc, sg, am) in ln:
            v = np.maximum(v, am * np.exp(-((x - xc) / sg) ** 2))
        return v

    def step_paint(o, ln, z_top, y_back, y_riser):
        """Dirt only where feet do not reach: against the cheeks, in the riser foot, under the nosing; lanes stay clean."""
        def gr(co, nr):
            x, y, z = co[:, 0], co[:, 1], co[:, 2]
            lane = lane_np(x, ln)
            side = np.clip(1.0 - (xin + fv(y) - np.abs(x)) / 0.3, 0.0, 1.0)
            ontop = z > z_top - 0.034 - D
            back = np.clip(1.0 - (y_back - y) / 0.06, 0.0, 1.0) * ontop
            v = 0.44 * side ** 1.5 + 0.29 * back * (1.0 - 0.7 * lane) + 0.14 * nz_big(co, nr) * (1.0 - lane) ** 2 * ontop
            if y_riser is not None:
                isr = (np.abs(y - y_riser) < 0.006) & (z < z_top - nt + 0.002)
                rfoot = isr * np.clip(1.0 - (z - (z_top - h)) / 0.07, 0.0, 1.0)
                und_n = isr * np.clip(1.0 - ((z_top - nt) - z) / 0.03, 0.0, 1.0)
                v = v + 0.33 * rfoot * (1.0 - 0.6 * lane) + 0.2 * und_n
            return np.clip(v, 0.0, GMAX)

        def wr(co, nr):                                        # feet keep the walking lanes pale and polished
            ontop = co[:, 2] > z_top - 0.034 - D
            return np.clip((0.25 + 0.6 * wear) * lane_np(co[:, 0], ln) ** 1.5 * ontop * (0.78 + 0.22 * nz_big(co, nr)), 0.0, 0.8)

        def sp(co, nr):                                        # green where nobody treads: against the cheeks, in the riser foot
            x, y, z = co[:, 0], co[:, 1], co[:, 2]
            side = np.clip(1.0 - (xin + fv(y) - np.abs(x)) / 0.22, 0.0, 1.0)
            ontop = z > z_top - 0.034 - D
            back = np.clip(1.0 - (y_back - y) / 0.07, 0.0, 1.0) * ontop
            mk = 0.45 + 0.55 * moss if moss > 0.02 else 0.0
            v = mk * (0.95 * side ** 2 * (0.35 + 0.65 * nz_mid(co, nr)) + 0.55 * back * (1.0 - lane_np(x, ln)) * nz_big(co, nr))
            return np.clip(v, 0.0, 0.65)
        paint(o, grime=gr, wear=wr, special=sp)

    def lay_row(name, P0, P1, cw, tw, closing, y_front, z_top, y_back, y_riser, lo=0.9, hi=1.7, jit=True, chip=True):
        hw = xin + f(y_front)
        X = hw + 0.04
        jl = rs.u(-0.05, 0.05)
        ln = [(xc + jl + rs.u(-0.03, 0.03), sg * rs.u(0.92, 1.08), am) for (xc, sg, am) in lanes(hw)]

        def lanef(x):
            v = 0.0
            for (xc, sg, am) in ln:
                v = max(v, am * exp(-((x - xc) / sg) ** 2))
            return v

        def xscale(y):
            return (xin + f(y) + 0.04) / X
        dx = clamp(2 * X / 30, 0.12, 0.26) * lod ** 0.6
        zmin = z_top - nt + 0.008
        x = -X
        first = True
        while x < X - 1e-4:
            L = rs.u(lo, hi) * (rs.u(0.4, 0.75) if first else 1.0)
            first = False
            x1 = min(X, x + L)
            if X - x1 < 0.5:
                x1 = X
            chips = []
            if chip and wear > 0.05:
                for _ in range(int(rs.u(0, 0.9 + 1.7 * wear / lod ** 0.5))):
                    rc = rs.u(0.04, 0.11)
                    xc = rs.u(x + 0.03, x1 - 0.03)
                    if abs(abs(xc) - hw) < rc + 0.02:
                        continue
                    chips.append((xc, rc, min(0.038, rs.u(0.016, 0.02 + 0.03 * wear))))
            und = (rs.u(0.0005, 0.0018), rs.u(3, 7), rs.u(0, 6), rs.u(0.0004, 0.0012), rs.u(9, 17), rs.u(0, 6))
            dzs, dys = (rs.u(-0.002, 0.002), rs.u(-0.003, 0.003)) if jit else (0.0, 0.0)
            q0 = [(y + dys, z + (dzs if z > 0.02 else 0.0)) for (y, z) in P0]
            q1 = [(y + dys, z + (dzs if z > 0.02 else 0.0)) for (y, z) in P1]
            cl_ = [(y + dys, z) for (y, z) in closing]
            st = _stone_row(name, q0, q1, cw, tw, zmin, cl_, x + GAP / 2, x1 - GAP / 2, dx, lanef, chips, und, xscale, fine=not lean)
            assign(st, m_tread)
            _tool(st, rs)
            step_paint(st, ln, z_top, y_back, y_riser)
            x = x1

    sag = 0.9 * D
    Rz = min(rr + 0.008 + 0.014 * wear, nt - sag - 0.012)        # worn nosing: a long soft roll-off instead of the arris
    Ry = rr + 0.018 + 0.03 * wear
    lip = 0.11
    for i in range(n):
        y0, z1 = yf[i], zt[i]
        zb = max(0.0, z1 - h - 0.02)
        zs = z1 - nt
        yr = y0 + nose
        zr1 = min(z1 - h + 0.055, zs - 0.012)
        d = dep[i]
        is_top = i == n - 1
        P0 = [(yr, zb), (yr, zr1), (yr, zs), (y0, zs), (y0, z1 - rr)]
        P1 = [(yr, zb), (yr, zr1), (yr, zs), (y0 + 0.002, zs), (y0 + 0.003, z1 - sag - Rz)]
        for ang in (150, 120, 90):
            ca_, sa_ = cos(radians(ang)), sin(radians(ang))
            P0.append((y0 + rr + rr * ca_, z1 - rr + rr * sa_))
            P1.append((y0 + 0.003 + Ry + Ry * ca_, z1 - sag - Rz + Rz * sa_))
        cw = [(0, 0), (0, 0), (0, 0), (0, 0), (0.0, 1.0), (0.3, 0.75), (0.65, 0.35), (1.0, 0.0)]   # a spall = oblique facet off the arris
        tw = [0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.8, 1.0]
        if is_top:                                             # top stone: dead level at the back, drip lip over the wall
            ys_t = [y0 + 0.3 * d, y0 + 0.62 * d]
            P0 += [(ys_t[0], z1), (ys_t[1], z1), (Y - 0.006, z1), (Y, z1 - 0.006), (Y, z1 - lip), (yw - 0.012, z1 - lip)]
            P1 += [(ys_t[0], z1 - D), (ys_t[1], z1 - 0.5 * D), (Y - 0.006, z1), (Y, z1 - 0.006), (Y, z1 - lip), (yw - 0.012, z1 - lip)]
            tw += [1.0, 0.6, 0.0, 0.0, 0.0, 0.0]
            closing = [(yw - 0.012, zb)]
            y_back = Y + 1.0
        elif i == m_land:
            P0 += [(y0 + 0.45 * g, z1), (y0 + g - 0.007, z1), (y0 + g - 0.003, z1 - 0.004)]
            P1 += [(y0 + 0.45 * g, z1 - D), (y0 + g - 0.007, z1 - 0.6 * D), (y0 + g - 0.003, z1 - 0.004 - 0.6 * D)]
            tw += [1.0, 0.8, 0.8]
            closing = [(y0 + g - 0.003, zb)]
            y_back = Y + 1.0
        else:
            yb_ = y0 + d + nose + 0.016
            P0 += [(y0 + 0.36 * d, z1), (y0 + 0.66 * d, z1), (yb_, z1)]
            P1 += [(y0 + 0.36 * d, z1 - D), (y0 + 0.66 * d, z1 - 0.6 * D), (yb_, z1 - 0.25 * D)]
            tw += [1.0, 1.0, 0.6]
            closing = [(yb_, zb)]
            y_back = y0 + d + nose
        cw += [(0, 0)] * (len(P0) - len(cw))
        if lean:                                               # big flights: leaner section
            keep = [q for q in range(len(P0)) if q not in (1, 5, 8)]
            P0, P1, cw, tw = ([L_[q] for q in keep] for L_ in (P0, P1, cw, tw))
        lay_row("step", P0, P1, cw, tw, closing, y0, z1, y_back, yr, jit=not is_top)
        if i == m_land:                                        # flagged landing behind the nosing stone
            ya_l, yb_l = y0 + g, y0 + d + nose + 0.016
            nr_ = max(1, int(round((yb_l - ya_l) / 0.5)))
            Df = 0.55 * D
            for q in range(nr_):
                qa = ya_l + (yb_l - ya_l) * q / nr_
                qb = ya_l + (yb_l - ya_l) * (q + 1) / nr_
                F0 = [(qa + 0.003, z1 - 0.035), (qa + 0.003, z1 - 0.005), (qa + 0.008, z1), ((qa + qb) / 2, z1), (qb - 0.008, z1),
                      (qb - 0.003, z1 - 0.005), (qb - 0.003, z1 - 0.035)]
                F1 = [F0[0]] + [(y, z - Df) for (y, z) in F0[1:6]] + [F0[6]]
                lay_row("flag", F0, F1, [(0, 0)] * 7, [0.0, 0.8, 1.0, 1.0, 1.0, 0.8, 0.0], [], qa, z1,
                        (y0 + d + nose) if q == nr_ - 1 else Y + 1.0, None, lo=0.55, hi=1.0, chip=False)

    # dark mortar bed under the steps (seen in the joints only)
    cv, cf = [], []

    def quad(a, b, c_, d_):
        k0 = len(cv)
        cv.extend([a, b, c_, d_])
        cf.append([k0, k0 + 1, k0 + 2, k0 + 3])
    for i in range(n):
        yr = yf[i] + nose + 0.012
        yn = (yf[i + 1] + nose + 0.012) if i < n - 1 else yw - 0.02
        zlo = 0.0 if i == 0 else zt[i - 1] - nt
        zhi = zt[i] - nt                                       # below the deepest spall, or the bed shows through
        x = xin + f(yf[i]) + 0.03
        quad((-x, yr, zlo), (x, yr, zlo), (x, yr, zhi), (-x, yr, zhi))
        quad((-x, yr, zhi), (x, yr, zhi), (x, yn, zhi), (-x, yn, zhi))
    score = mesh_object("step_core", cv, cf)
    assign(score, m_mortar)
    paint(score, grime=0.3)

    # ---------------------------------------------------------------- retaining face under the top step (rear)
    rb = rng.fork("back")
    zlip = rise - lip
    xbk = (xw - pw / 2 - 0.004) if top_pier else (xin + 0.004)
    bsheet = mesh_object("back_core", [(xbk, yw - 0.012, 0.0), (-xbk, yw - 0.012, 0.0), (-xbk, yw - 0.012, zlip), (xbk, yw - 0.012, zlip)],
                         [[0, 1, 2, 3]])
    assign(bsheet, m_mortar)
    paint(bsheet, grime=0.3)
    lvb = _levels(zlip, 0.12)
    for ci in range(len(lvb) - 1):
        z0, z1 = lvb[ci], lvb[ci + 1]
        x = -xbk
        first = True
        while x < xbk - 1e-4:
            L = rb.u(0.5, 0.9) * (0.55 if (first and ci % 2) else 1.0)
            first = False
            x1 = min(xbk, x + L)
            if xbk - x1 < 0.26:
                x1 = xbk
            if ci == 0:                                        # rock-faced plinth course like the cheeks
                blk = box("back_block", (x1 - x - GAP, 0.09, z1 - GAP / 2), loc=((x + x1) / 2, yw - 0.015, (z1 - GAP / 2) / 2))
                apply_transform(blk)
                densify(blk, 0.3)
                bevel(blk, 0.018, segments=1, angle=25)
                displace_noise(blk, strength=0.007, scale=0.11, seed=rb.u(0, 99))
                transform_verts(blk, lambda co: Vector((co.x, min(co.y, Y - 0.004), max(co.z, 0.0))))
                assign(blk, m_rough)
            else:
                blk = _face_block("back_block", x + GAP / 2, x1 - GAP / 2, z0 + GAP / 2, z1 - GAP / 2, yw + rb.u(-0.002, 0.002))
                densify(blk, 0.3)
                assign(blk, ash(rb))
                _tool(blk, rb)
            wall_paint(blk, rb.u(0.0, 0.15), zcap=zlip, inner=False, reach=0.4)
            x = x1

    # ---------------------------------------------------------------- moss cushions in the damp corners
    if moss > 0.05:
        rm = rng.fork("moss")
        tb = sphere("tuft", 1.0, segments=6, rings=3, scale=(1, 1, 0.55))
        apply_transform(tb)
        displace_noise(tb, strength=0.22, scale=0.8, seed=seed)
        transform_verts(tb, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
        assign(tb, mat("moss"))
        tm = []
        km = 0.8 / lod
        for i in range(n):
            for sgn in (-1, 1):
                for _ in range(int(rm.u(0, 1.0 + 6.0 * moss * km))):     # along the wall foot on the tread
                    y = min(rm.u(yf[i] + nose + 0.03, yf[i] + dep[i] + nose), Y - 0.1)
                    sc = rm.u(0.018, 0.042)
                    tm.append(trs((sgn * (xin + f(y) - rm.u(0.0, 0.012)), y, zt[i] - 0.004 - 0.2 * D), (0, 0, 90 + rm.u(-20, 20)),
                                  (sc * rm.u(1.0, 2.4), sc, sc * rm.u(0.5, 0.9))))
                if i < n - 1:
                    for _ in range(int(rm.u(0, 0.6 + 5.0 * moss * km))):  # in the riser foot, thinning away from the wall
                        y = yf[i + 1] + nose - rm.u(0.0, 0.012)
                        sc = rm.u(0.016, 0.038)
                        xx = xin + f(y) - 0.02 - abs(rm.u(-1, 1)) * rm.u(0.0, 0.3)
                        tm.append(trs((sgn * xx, y, zt[i] - 0.004 - 0.25 * D), (0, 0, rm.u(-20, 20)), (sc * rm.u(1.0, 2.6), sc, sc * rm.u(0.5, 0.9))))
        for sgn in (-1, 1):                                              # outside, at the foot of the plinth
            for _ in range(int((yb - ya) * rm.u(1.0, 2.0 + 7.0 * moss * km))):
                y = rm.u(ya + 0.05, yb - 0.13)
                sc = rm.u(0.018, 0.045)
                tm.append(trs((sgn * (xw + t / 2 + 0.035 + stage + f(y)), y, 0.0), (0, 0, 90 + rm.u(-15, 15)), (sc * rm.u(1.0, 2.5), sc, sc * rm.u(0.5, 0.9))))
        if tm:
            instances_to_mesh("moss_tufts", tb, tm, keep_base=False)
        else:
            delete([tb])

    # ---------------------------------------------------------------- central handrail (forged iron)
    if rail:
        Hr = 0.92
        off = 0.5 * g + nose / 2
        Hp = off * s + Hr - 0.008
        q_ = 3 if n >= 7 else 2

        def pick(a, b):
            idx = list(range(a, b + 1, q_))
            if idx[-1] != b:
                if b - idx[-1] < 2 and len(idx) > 1:
                    idx[-1] = b
                else:
                    idx.append(b)
            return idx
        posts = []                                   # (y, z base)
        if has_land:
            for i in pick(0, m_land):
                posts.append((yf[i] + off, zt[i]))
            posts.append((yf[m_land + 1] - g + off, zt[m_land]))
            if m_land + 2 <= n - 1:
                for i in pick(m_land + 2, n - 1):
                    posts.append((yf[i] + off, zt[i]))
            else:
                posts.append((yf[n - 1] + off, zt[n - 1]))
        else:
            for i in pick(0, n - 1):
                posts.append((yf[i] + off, zt[i]))
        zk = Hp * 0.5                                # height of the lower rod through the post eyes
        pp = [(0.036, -0.012), (0.038, 0.004), (0.036, 0.012), (0.024, 0.02), (0.021, 0.034), (0.026, 0.044), (0.026, 0.054),
              (0.0155, 0.07), (0.0145, zk - 0.05), (0.02, zk - 0.03), (0.027, zk - 0.012), (0.028, zk), (0.027, zk + 0.012),
              (0.02, zk + 0.03), (0.0145, zk + 0.05), (0.013, Hp - 0.075), (0.018, Hp - 0.06), (0.022, Hp - 0.048),
              (0.018, Hp - 0.036), (0.013, Hp - 0.024), (0.017, Hp - 0.008), (0.017, Hp)]
        post = lathe("rail_post", pp, segments=8 if lod < 1.5 else 6)
        assign(post, m_iron)
        paint(post, grime=mask_mul(height_ramp(0.0, 0.2), 0.4))
        pobj = instances_to_mesh("rail_posts", post, [trs((0, y, z), (0, 0, rng.u(0, 90))) for (y, z) in posts], keep_base=False)
        rail_sec = rounded_rect(0.022, 0.046, 0.008, 3 if lod < 1.5 else 2)
        main = [(0.0, y, z + Hp + 0.008) for (y, z) in posts]
        ca, sa = cos(radians(theta)), sin(radians(theta))
        Eb = (main[0][1] - 0.1 * ca, main[0][2] - 0.1 * sa)
        main_path = [(0.0, Eb[0], Eb[1])] + main
        iron_objs = [pobj]
        iron_objs.append(sweep("rail_top", main_path, rail_sec, smooth_path=False, up=(1, 0, 0)))
        sc = _scroll(Eb, 180 + theta, 0.072, 0.018, 1.2, ccw=True)
        iron_objs.append(sweep("rail_scroll_b", [(0.0, y, z) for (y, z) in sc], rail_sec, smooth_path=False, up=(1, 0, 0),
                               scale_fn=lambda q: 1.0 - 0.45 * q))
        # top end: level out past the last post, then a small curl
        Rl = (main[-1][1], main[-1][2])
        Et = (Rl[0] + 0.17, Rl[1] + 0.06 * s + 0.012)
        ease = bezier_path((0.0, Rl[0], Rl[1]), (0.0, Rl[0] + 0.07, Rl[1] + 0.07 * s), (0.0, Et[0] - 0.07, Et[1]), (0.0, Et[0], Et[1]), n=8)
        iron_objs.append(sweep("rail_ease", ease, rail_sec, smooth_path=False, up=(1, 0, 0)))
        sc2 = _scroll(Et, 0.0, 0.05, 0.014, 1.15, ccw=False, npts=22)
        iron_objs.append(sweep("rail_scroll_t", [(0.0, y, z) for (y, z) in sc2], rail_sec, smooth_path=False, up=(1, 0, 0),
                               scale_fn=lambda q: 1.0 - 0.45 * q))
        # forged C-scroll brackets from post to rail
        ry = [Eb[0]] + [q[1] for q in main] + [Et[0]]
        rz = [Eb[1]] + [q[2] for q in main] + [Et[1]]
        flat = [(-0.004, -0.011), (0.004, -0.011), (0.004, 0.011), (-0.004, 0.011)]
        for bi, (y, z) in enumerate(posts):
            for sd in (-1, 1):
                ye_ = y + sd * 0.2
                if ye_ < ry[0] + 0.03 or ye_ > ry[-1] - 0.01 or (bi == 0 and sd < 0):
                    continue

                def rz_(yy):
                    return float(np.interp(yy, ry, rz)) - 0.0145
                z0_ = z + Hp - 0.3
                e1 = (y + sd * 0.155, rz_(y + sd * 0.155))
                bz = bezier_path((0.0, y + sd * 0.017, z0_), (0.0, y + sd * 0.02, z0_ + 0.16),
                                 (0.0, y + sd * 0.07, rz_(y + sd * 0.07) - 0.035), (0.0, e1[0], e1[1]), n=9)
                slope = degrees(atan((rz_(e1[0] + 0.02) - rz_(e1[0] - 0.02)) / 0.04))
                cur = _scroll(e1, slope if sd > 0 else 180 + slope, 0.024, 0.008, 0.85, ccw=(sd < 0), npts=10)
                path = list(bz) + [(0.0, a_, b_) for (a_, b_) in cur[1:]]
                # root curl at the post
                iron_objs.append(sweep("rail_bracket", path, flat, smooth_path=False, up=(1, 0, 0),
                                       scale_fn=lambda q: 1.0 - 0.35 * max(0.0, (q - 0.75) / 0.25)))
                iron_objs.append(box("rail_collar", (0.03, 0.012, 0.016), loc=(0, y + sd * 0.0175, z0_ + 0.03), bevel=0.002 if lod < 1.5 else 0.0))
        iron_objs.append(sweep("rail_rod", [(0.0, y, z + zk) for (y, z) in posts], 0.0085, sides=8, smooth_path=False))
        for (y, z) in (posts[0], posts[-1]):
            iron_objs.append(sphere("rail_rod_end", 0.016, loc=(0, y, z + zk), segments=10, rings=6))
        for o in iron_objs:
            assign(o, m_iron)

    # ---------------------------------------------------------------- pier lanterns / lamp-post sockets
    lit = [k for k in {"none": [], "bottom": ["bl", "br"], "all": ["bl", "br", "tl", "tr"]}[p.lanterns] if k in tops]
    if lit:
        glass = mat("glass_lantern", glow=0.68)
        flame = mat("emissive", "#ffd29a", glow=6.5)
        io, panes, mantle = _lantern(m_iron, glass, flame, seg=12 if len(lit) <= 2 else 8)
        for o in io:
            if o.name.startswith("l_roof"):
                paint(o, grime=0.35)
        lan = join(io + [panes, mantle], "lantern_base")
        mats = [trs(tops[k], (0, 0, rng.u(-3, 3))) for k in lit]
        instances_to_mesh("pier_lanterns", lan, mats, uv_jitter=False, keep_base=False)
        for k in lit:
            cx, cy, cz = tops[k]
            light("lantern_" + k, (cx, cy, cz + 0.65), color="#ffbf6a", intensity=p.intensity, range=9.0, flicker=0.14,
                  shadow_near=0.12)
    for k in ("bl", "br", "tl", "tr"):
        if k in tops and k not in lit:
            socket("lamp_" + k, "attach", tops[k], role="lantern")
