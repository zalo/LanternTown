"""Cast-iron manhole cover / gully grate seated in a cast frame and bedded in the paving.

Round or square access covers are a dished plate with a raised rim, a founder's lettering ring (or
bands), a tread pattern of individually cast lugs with draft, two lifting keyholes with raised lips
and an optional bronze town emblem on a roundel inside a ringed boss.  The gully grate is a slotted
bar grate with a spine, lettered rails and a knuckle hinge.  The frame is set in a collar of
butt-jointed dressed stone slabs (quarters / halves with a circular cut-out, or a pinwheel of four
rectangular slabs) on a recessed mortar bed, or in a course of setts; a dark, irregular mortar
bedding skirt ties the collar into the surrounding paving.  Traffic polishes the raised iron tops
to bare dark steel, rust and dirt collect in the recesses, the keyholes and the frame gap, and rust
runs stain the stone next to the frame.
"""
from math import sin, cos, pi, radians, degrees, hypot, tan, asin, sqrt, atan2

import numpy as np

from tk import *

META = {"name": "Manhole and drain covers", "class": "prop_small", "mount": "ground",
        # the lettering ring, ~100 individually cast tread lugs and 96-segment rims ARE the object
        "tri_budget": 8500, "focus": [0.0, 0.0, 0.01]}
PARAMS = {
    "shape": {"default": "round", "choices": ["round", "square", "gully"], "desc": "round / square access cover or slotted gully grate"},
    "size": {"default": 0.6, "min": 0.3, "max": 0.9, "desc": "cover diameter / width (m)"},
    "aspect": {"default": 1.0, "min": 0.6, "max": 1.5, "desc": "depth / width of square and gully covers"},
    "pattern": {"default": "chequer", "choices": ["chequer", "diamond", "radial", "studs"],
                "desc": "tread pattern (gully: chequer / studs = straight bars, diamond / radial = herringbone bars)"},
    "text": {"default": "LANTERN TOWN", "desc": "founder's lettering, upper arc / top band / front rail"},
    "text2": {"default": "WATER WORKS 1887", "desc": "second legend, lower arc / bottom band / back rail ('' = none)"},
    "emblem": {"default": "star", "choices": ["star", "crown", "none"], "desc": "polished bronze town emblem on a roundel in a ringed centre boss"},
    "surround": {"default": "stone", "choices": ["stone", "setts", "none"],
                 "desc": "butt-jointed dressed stone collar, course of setts, or bare frame"},
    "stone": {"default": "#81796b", "desc": "stone tint of the surround"},
    "finish": {"default": "cast", "choices": ["cast", "tarred", "painted"],
               "desc": "cover finish: bare weathered casting, bitumen-black coating, or paint in the `paint` colour"},
    "paint": {"default": "#5a1e18", "desc": "paint colour when finish = painted"},
    "wear": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "foot / wheel polish on the raised pattern (also chips paint off the tops)"},
    "rust": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "rust in the recesses, on the frame and bleeding onto the stone"},
    "proud": {"default": 0.002, "min": 0.0, "max": 0.008, "desc": "how far the frame stands above its 6.5 mm bedding (m)"},
    "turn": {"default": 0.0, "min": -180.0, "max": 180.0, "desc": "how a round cover was dropped back into its frame (deg)"},
    "leaves": {"default": 0, "min": 0, "max": 10, "desc": "dead leaves caught on the cover"},
}
VARIANTS = {
    "gully_grate": {"shape": "gully", "size": 0.46, "aspect": 0.8, "pattern": "chequer", "text": "T.T. FOUNDRY", "text2": "",
                    "surround": "setts", "stone": "#77767c", "finish": "cast", "rust": 0.85, "wear": 0.35, "leaves": 6, "seed": 4},
    "hydrant_square": {"shape": "square", "size": 0.44, "aspect": 0.78, "pattern": "diamond", "text": "FIRE", "text2": "HYDRANT",
                       "emblem": "none", "finish": "painted", "paint": "#683029", "surround": "stone", "stone": "#a3967f",
                       "wear": 0.8, "seed": 8},
    "sunburst_crown": {"shape": "round", "size": 0.78, "pattern": "radial", "emblem": "crown", "text": "LANTERN SQUARE",
                       "text2": "GAS LIGHT & COKE Co", "surround": "setts", "stone": "#6f6e75", "wear": 0.9, "rust": 0.25,
                       "turn": 14.0, "seed": 15},
    "coal_hole": {"shape": "round", "size": 0.36, "pattern": "studs", "emblem": "none", "text": "COAL", "text2": "",
                  "surround": "stone", "stone": "#9c9484", "finish": "tarred", "rust": 0.55, "wear": 0.75, "leaves": 2, "seed": 23},
}



# ------------------------------------------------------------------ small local mesh helpers
class Acc:
    """Accumulates many small solids into one mesh."""

    def __init__(self):
        self.v, self.f = [], []

    def add(self, verts, faces):
        b = len(self.v)
        self.v += verts
        self.f += [tuple(b + i for i in f) for f in faces]

    def obj(self, name):
        return mesh_object(name, self.v, self.f) if self.f else None

    def tris(self):
        return sum(len(f) - 2 for f in self.f)


def mix_hex(a, b, t):
    ca = [int(a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    cb = [int(b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(round(lerp(x, y, t))) for x, y in zip(ca, cb))


def outline(hw, hd, r, seg, ns=0):
    """CCW outline of a rounded rectangle (half sizes hw, hd, corner radius r; r = hw = hd is a circle that
    starts at angle 0 with 4 * seg points).  ns = extra points on every straight."""
    corners = [(hw - r, hd - r, 0.0), (-(hw - r), hd - r, 90.0), (-(hw - r), -(hd - r), 180.0), (hw - r, -(hd - r), 270.0)]
    pts = []
    for q in range(4):
        cx, cy, a0 = corners[q]
        arc = [(cx + r * cos(radians(a0 + 90.0 * i / seg)), cy + r * sin(radians(a0 + 90.0 * i / seg))) for i in range(seg + 1)]
        nx_, ny_, a1 = corners[(q + 1) % 4]
        sx, sy = nx_ + r * cos(radians(a1)), ny_ + r * sin(radians(a1))
        if hypot(sx - arc[-1][0], sy - arc[-1][1]) < 1e-6:
            pts += arc[:-1]
        else:
            pts += arc
            pts += [(lerp(arc[-1][0], sx, k / (ns + 1)), lerp(arc[-1][1], sy, k / (ns + 1))) for k in range(1, ns + 1)]
    return pts


def loop_rr(hx, hy, r, step):
    """Dense CCW loop on a rounded rectangle with outward normals: [(x, y, nx, ny)]."""
    corners = [(hx - r, hy - r, 0.0), (-(hx - r), hy - r, 90.0), (-(hx - r), -(hy - r), 180.0), (hx - r, -(hy - r), 270.0)]
    na = max(2, int(pi / 2 * r / step + 0.999))
    pts = []
    for q in range(4):
        cx, cy, a0 = corners[q]
        for i in range(na + 1):
            a = radians(a0 + 90.0 * i / na)
            pts.append((cx + r * cos(a), cy + r * sin(a), cos(a), sin(a)))
        nx_, ny_, _ = corners[(q + 1) % 4]
        a1 = radians(a0 + 90.0)
        ex, ey = cx + r * cos(a1), cy + r * sin(a1)
        sx, sy = nx_ + r * cos(a1), ny_ + r * sin(a1)
        ns = int(hypot(sx - ex, sy - ey) / step)
        for k in range(1, ns + 1):
            t = k / (ns + 1)
            pts.append((lerp(ex, sx, t), lerp(ey, sy, t), cos(a1), sin(a1)))
    out = []
    for q in pts:
        if not out or hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > 1e-6:
            out.append(q)
    if hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out


def ray_rect(a, hx, hy):
    c, s = cos(a), sin(a)
    t = min(hx / max(abs(c), 1e-9), hy / max(abs(s), 1e-9))
    return (t * c, t * s)


def up_faces(verts, faces):
    """Orient faces so their normals point up."""
    out = []
    for f in faces:
        nz = 0.0
        for k in range(len(f)):
            a, b = verts[f[k]], verts[f[(k + 1) % len(f)]]
            nz += (a[0] - b[0]) * (a[1] + b[1])
        out.append(tuple(f) if nz >= 0 else tuple(reversed(f)))
    return out


def stitch(faces, verts, A, B):
    """Triangulate between two open vertex rows of different resolution (shortest diagonal first)."""
    i = j = 0
    d2 = lambda a, b: (verts[a][0] - verts[b][0]) ** 2 + (verts[a][1] - verts[b][1]) ** 2
    while i < len(A) - 1 or j < len(B) - 1:
        if j >= len(B) - 1 or (i < len(A) - 1 and d2(A[i + 1], B[j]) <= d2(A[i], B[j + 1])):
            faces.append((A[i], A[i + 1], B[j]))
            i += 1
        else:
            faces.append((A[i], B[j + 1], B[j]))
            j += 1


def ring_mesh(name, outline_fn, prof, cap=False, off=(0.0, 0.0)):
    """Concentric section: prof = [(inset from the outline, z), ...] ordered along the visible surface;
    outline_fn(inset, k) gives ring k.  A ring may have half the points of the previous one."""
    rings = [outline_fn(ins, k) for k, (ins, _) in enumerate(prof)]
    verts, faces, start = [], [], []
    for (ins, z), ring in zip(prof, rings):
        start.append(len(verts))
        verts += [(x + off[0], y + off[1], z) for x, y in ring]
    for k in range(len(prof) - 1):
        n, m = len(rings[k]), len(rings[k + 1])
        a, b = start[k], start[k + 1]
        if n == m:
            for j in range(n):
                j2 = (j + 1) % n
                faces.append((a + j, a + j2, b + j2, b + j))
        else:
            assert n == 2 * m, (n, m)
            for j in range(m):
                o0, o1, o2 = a + 2 * j, a + 2 * j + 1, a + (2 * j + 2) % n
                i0, i1 = b + j, b + (j + 1) % m
                faces += [(o0, o1, i0), (o1, o2, i1), (o1, i1, i0)]
    if cap:
        faces.append(tuple(start[-1] + j for j in range(len(rings[-1]))))
    return mesh_object(name, verts, faces)


def levels_prof(bounds, zs, zbot, d=0.002):
    """Section of a cast plate: levels zs (outermost first) changing at the insets `bounds`, with draft."""
    pts = [(0.0, zbot), (0.0, zs[0] - 0.001), (0.001, zs[0])]
    for b, za, zb_ in zip(bounds, zs[:-1], zs[1:]):
        pts += [(b - d * 0.5, za), (b + d * 0.5, zb_)]
    return pts


def ccw(pts):
    a = sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))
    return list(pts) if a >= 0 else list(reversed(pts))


def prism(acc, pts, z0, z1, inset=0.0015, dome=0.0, zfn=None, top=None, tilt=(0.0, 0.0)):
    """Cast lug / bar / sett: polygon `pts` (CCW) at z0, top face inset by the draft at z1. No bottom.
    top = another Acc that receives the top face (own vertices) so it can be a different, polished surface."""
    n = len(pts)
    tp = []
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        ax, ay = p1[0] - p0[0], p1[1] - p0[1]
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        la, lb = hypot(ax, ay) or 1e-9, hypot(bx, by) or 1e-9
        n0 = (-ay / la, ax / la)
        n1 = (-by / lb, bx / lb)
        k = max(1.0 + n0[0] * n1[0] + n0[1] * n1[1], 0.35)
        tp.append((p1[0] + inset * (n0[0] + n1[0]) / k, p1[1] + inset * (n0[1] + n1[1]) / k))
    cx = sum(x for x, _ in tp) / n
    cy = sum(y for _, y in tp) / n
    zt = lambda x, y: z1 + (zfn(x, y) if zfn else 0.0) + tilt[0] * (x - cx) + tilt[1] * (y - cy)
    tv = [(x, y, zt(x, y)) for x, y in tp]
    acc.add([(x, y, z0) for x, y in pts] + tv, [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)])
    if dome:
        tv = tv + [(cx, cy, zt(cx, cy) + dome)]
        tf = [(i, (i + 1) % n, n) for i in range(n)]
    else:
        tf = [tuple(range(n))]
    (top if top is not None else acc).add(tv, tf)


def rect_pts(cx, cy, l, w, ang=0.0):
    c, s = cos(radians(ang)), sin(radians(ang))
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in ((-l / 2, -w / 2), (l / 2, -w / 2), (l / 2, w / 2), (-l / 2, w / 2))]


def clip_x(pts, x0, x1):
    """Clip a convex polygon to x0 <= x <= x1."""
    for sign, lim in ((1.0, x0), (-1.0, x1)):
        out = []
        for i in range(len(pts)):
            a, b = pts[i - 1], pts[i]
            da, db = sign * (a[0] - lim), sign * (b[0] - lim)
            if (da >= 0) != (db >= 0):
                t = da / (da - db)
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            if db >= 0:
                out.append(b)
        pts = out
        if len(pts) < 3:
            return []
    return pts


def glyphs(body, cap, depth, font="sans"):
    """Per-letter meshes (bottom caps removed), centred on the cap height: [(x centre, verts, faces)], total width."""
    ref = text("ref", "H", size=cap, depth=depth, rot=(0, 0, 0), font=font)
    ys = [v.co.y for v in ref.data.vertices]
    y0, y1 = min(ys), max(ys)
    delete([ref])
    k = cap / max(y1 - y0, 1e-6)
    cy = (y0 + y1) / 2
    out, x, gap = [], 0.0, cap * 0.2
    for ch in body:
        if ch == " ":
            x += cap * 0.42
            continue
        o = text("g", ch, size=cap, depth=depth, rot=(0, 0, 0), font=font)
        me = o.data
        if len(me.polygons):
            xs = [v.co.x for v in me.vertices]
            gx0 = min(xs)
            w = (max(xs) - gx0) * k
            vs = [((v.co.x - gx0) * k - w / 2, (v.co.y - cy) * k, v.co.z) for v in me.vertices]
            fs = [tuple(pl.vertices) for pl in me.polygons if pl.normal.z > -0.5]
            out.append((x + w / 2, vs, fs))
            x += w + gap
        delete([o])
    return out, max(x - gap, 0.0)


def emit_text(acc, body, cap, depth, z0, avail, where):
    """Lay the legend out; where(x along the line, centred) -> (px, py, angle rad)."""
    body = body.strip()
    if not body:
        return
    gl, total = glyphs(body, cap, depth)
    s = min(1.0, avail / max(total, 1e-6))
    for xc, vs, fs in gl:
        px, py, ang = where((xc - total / 2) * s)
        c, sn = cos(ang), sin(ang)
        acc.add([(px + (x * c - y * sn) * s, py + (x * sn + y * c) * s, z0 + z) for x, y, z in vs], fs)


def leaf_mesh(acc, cx, cy, z, length, yaw, rng):
    w = length * rng.u(0.26, 0.4)
    curl = rng.u(0.002, 0.007)
    fold = rng.u(-0.003, 0.004)
    c, s = cos(yaw), sin(yaw)
    verts, faces = [], []

    def P(t, side):
        hw_ = w * (sin(pi * t) ** 0.8) * (1.2 - 0.5 * t)
        x, y = (t - 0.45) * length, side * hw_
        zz = z + curl * (2 * t - 1) ** 2 + fold * abs(side) + 0.0015 * sin(t * 9 + side)
        return (cx + x * c - y * s, cy + x * s + y * c, zz)

    ts = (0.1, 0.27, 0.5, 0.73, 0.9)
    ns = len(ts)
    verts.append(P(0.0, 0))
    for t in ts:
        verts += [P(t, 1), P(t, 0), P(t, -1)]          # L, M, R
    verts.append(P(1.0, 0))
    L = lambda i: 1 + 3 * i
    M = lambda i: 2 + 3 * i
    R = lambda i: 3 + 3 * i
    faces += [(0, R(0), M(0)), (0, M(0), L(0))]
    for i in range(ns - 1):
        faces += [(M(i), R(i), R(i + 1), M(i + 1)), (M(i), M(i + 1), L(i + 1), L(i))]
    tip = 1 + 3 * ns
    faces += [(M(ns - 1), R(ns - 1), tip), (M(ns - 1), tip, L(ns - 1))]
    acc.add(verts, faces)


# ------------------------------------------------------------------ dressed stone collar
ARRIS = 0.004      # width of the worn, rounded arris
DROP = 0.0045      # how far the stone edge dives below its top (the mortar bed hides the bottom of it)
BED = 0.0017       # mortar joints are recessed this far below the stone tops


def _stone_from_columns(name, cols, coarse, stain_w, ztop, dz, st_mat, stain_mat, stain_ok=None):
    """Stone slab from columns [(pt at the frame side, pt at the outer side)].  Three fine rows next to the
    frame (edge, arris, the wavy limit of the rust stain), five coarser rows across the rest of the stone.
    Returns the object; faces of the rust-stained band get stain_mat."""
    n = len(cols)
    cidx = [i for i in range(n) if coarse[i]]
    verts, faces = [], []
    kinds = []        # per vertex: 0 top, 1 edge sunk in the joint, 2 arris, 3 arris against the frame, 4 limit of the rust stain
    fine = [[0] * n for _ in range(3)]
    crs = [[0] * len(cidx) for _ in range(5)]

    def vert(i, v, ln, low, kind=0):
        (ax, ay), (bx, by) = cols[i]
        x, y = lerp(ax, bx, v / ln), lerp(ay, by, v / ln)
        z = ztop(x, y) + dz[0] + dz[1] * x + dz[2] * y
        verts.append((x, y, max(z - DROP, 0.0) if low else z))
        kinds.append(1 if low else (2 if i in (1, n - 2) and kind == 0 else kind))
        return len(verts) - 1

    for i in range(n):
        (ax, ay), (bx, by) = cols[i]
        ln = max(hypot(bx - ax, by - ay), 1e-6)
        edge = i in (0, n - 1)
        st = min(max(stain_w[i], ARRIS + 0.004), 0.45 * ln)
        for r, v in enumerate((0.0, ARRIS, st)):
            fine[r][i] = vert(i, v, ln, edge or r == 0, (0, 3 if stain_ok is None or stain_w[i] > 0.006 else 2, 4)[r])
        if coarse[i]:
            j = cidx.index(i)
            a_, b_ = min(0.06, 0.5 * ln), ln - 0.016
            for r, v in enumerate((lerp(a_, b_, 0.3), lerp(a_, b_, 0.68), b_, ln - ARRIS, ln)):
                crs[r][j] = vert(i, v, ln, edge or r == 4, 2 if r == 3 else 0)
    stain_faces = []
    for r in range(2):
        for i in range(n - 1):
            if r == 1 and (stain_ok is None or stain_ok(i)):
                stain_faces.append(len(faces))
            faces.append((fine[r][i], fine[r][i + 1], fine[r + 1][i + 1], fine[r + 1][i]))
    stitch(faces, verts, fine[2], crs[0])
    for r in range(4):
        for j in range(len(cidx) - 1):
            faces.append((crs[r][j], crs[r][j + 1], crs[r + 1][j + 1], crs[r + 1][j]))
    st = mesh_object(name, verts, up_faces(verts, faces))
    assign(st, st_mat)
    st.data.materials.append(stain_mat)
    for fi in stain_faces:
        st.data.polygons[fi].material_index = 1
    smooth(st, 62)
    uv_box(st, scale=4.0)          # fine tooled grain: features of 5-20 mm on a 0.3 m slab
    return st, np.array(kinds)


def sector_stone(name, R, Sx, Sy, th0, th1, nseg, jw, wst, ztop, dz, st_mat, stain_mat):
    """Slab with a circular cut-out (radius R) between the butt joints at th0 and th1 (deg, multiples of 90),
    reaching out to the rectangle +-Sx, +-Sy.  The cut-out follows the frame's own tessellation (nseg)."""
    step = 360.0 / nseg
    d0 = (cos(radians(th0)), sin(radians(th0)))
    d1 = (cos(radians(th1)), sin(radians(th1)))
    n0, n1 = (-d0[1], d0[0]), (d1[1], -d1[0])           # normals pointing into the stone
    hs = (jw / 2, jw / 2 + ARRIS, jw / 2 + 0.016)

    def ecol(d, nrm, h):
        sd = Sx if abs(d[0]) > 0.5 else Sy
        c = sqrt(max(R * R - h * h, 1e-9))
        return ((d[0] * c + nrm[0] * h, d[1] * c + nrm[1] * h), (d[0] * sd + nrm[0] * h, d[1] * sd + nrm[1] * h))

    amin = degrees(asin(min(0.9, hs[2] / R))) + 1.0
    cols = [ecol(d0, n0, h) for h in hs]
    coarse = [True] * 3
    for k in range(int(round(th0 / step)), int(round(th1 / step)) + 1):
        if th0 + amin < k * step < th1 - amin:
            a = radians(k * step)
            cols.append(((R * cos(a), R * sin(a)), ray_rect(a, Sx, Sy)))
            coarse.append(k % 2 == 0)
    cols += [ecol(d1, n1, h) for h in reversed(hs)]
    coarse += [True] * 3
    return _stone_from_columns(name, cols, coarse, [wst(c[0][0], c[0][1]) for c in cols], ztop, dz, st_mat, stain_mat)


def slab_stone(name, org, du, dv, L, W, Lf, jw, wst, ztop, dz, st_mat, stain_mat):
    """Rectangular slab of a pinwheel collar: local u runs along the frame edge from the butt joint at u = 0,
    v runs outwards.  Only u < Lf lies against the frame (the rest butts against the next slab)."""
    us = [jw, jw + ARRIS, jw + 0.016]
    ni = max(2, int((L - jw - 0.032) / 0.04))
    us += [lerp(jw + 0.016, L - 0.016, k / ni) for k in range(1, ni)] + [L - 0.016, L - ARRIS, L]
    P = lambda u, v: (org[0] + du[0] * u + dv[0] * v, org[1] + du[1] * u + dv[1] * v)
    cols = [(P(u, 0.0), P(u, W)) for u in us]
    sw_ = []
    for u in us:
        x, y = P(u, 0.0)
        sw_.append(lerp(0.0, wst(x, y), smoothstep(Lf + 0.004, Lf - 0.03, u)))
    return _stone_from_columns(name, cols, [True] * len(us), sw_, ztop, dz, st_mat, stain_mat,
                               stain_ok=lambda i: 0.5 * (us[i] + us[i + 1]) < Lf)


def bed_mesh(name, angles, inner_fn, outer_fn, zfn, fr=(0.0, 0.5, 1.0)):
    """Mortar bed under a collar: a sheet between two loops sampled along rays from the centre."""
    verts, faces = [], []
    n, m = len(angles), len(fr)
    for a in angles:
        pi_, po_ = inner_fn(a), outer_fn(a)
        for f in fr:
            x, y = lerp(pi_[0], po_[0], f), lerp(pi_[1], po_[1], f)
            verts.append((x, y, max(zfn(x, y), 0.0006)))
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(m - 1):
            faces.append((i * m + j, i * m + j + 1, i2 * m + j + 1, i2 * m + j))
    return mesh_object(name, verts, up_faces(verts, faces))


def skirt_mesh(name, hx, hy, r, w0, zin, sr):
    """Irregular mortar / sand bedding skirt around a collar: packed against the stone, feathering out to the paving."""
    lp = loop_rr(hx, hy, r, 0.036)
    n = len(lp)
    ph = [sr.u(0, 6.28) for _ in range(3)]
    verts, faces = [], []
    for i, (x, y, nx, ny) in enumerate(lp):
        s = 2 * pi * i / n
        w = w0 * (1.0 + 0.26 * sin(3 * s + ph[0]) + 0.2 * sin(8 * s + ph[1]) + 0.16 * sin(17 * s + ph[2])) + sr.u(-0.003, 0.004)
        verts += [(x - nx * 0.003, y - ny * 0.003, zin),
                  (x + nx * w * 0.5, y + ny * w * 0.5, zin * 0.8 + sr.u(-0.0003, 0.0004)),
                  (x + nx * w, y + ny * w, 0.0)]
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(2):
            faces.append((i * 3 + j, i * 3 + j + 1, i2 * 3 + j + 1, i2 * 3 + j))
    sk = mesh_object(name, verts, up_faces(verts, faces))
    smooth(sk, 60)
    uv_box(sk)
    return sk


# ------------------------------------------------------------------ the asset
def build(p, rng):
    shape = p.shape
    is_round = shape == "round"
    is_gully = shape == "gully"
    hw = p.size / 2
    hd = hw if is_round else hw * p.aspect
    zf = 0.0065 + p.proud             # frame top
    zc = zf - 0.0005                  # cover top (raised work)
    zp = zc - 0.0035                  # recessed floor of the plate
    zb = 0.0015                       # underside of loose cover parts
    gap = 0.003                       # play between cover and frame
    fw = clamp(p.size * 0.07, 0.028, 0.044) * (1.0 if is_round else 0.85)   # frame flange width
    rc = hw if is_round else 0.010    # cover corner radius
    SEG = 24 if is_round else 3       # 96 segments on the round rims: they are seen from 1.5 m
    NS = 0 if is_round else 4

    # ---------------------------------------------------------------- materials
    finish = p.finish
    bare = mat("iron_painted", "#55524f", rough=2.2, grime=0.8)      # the sand-cast charcoal iron itself
    # trafficked tops: dark gun-metal on a bare casting, rubbed-through rusty iron under paint / tar
    tw = clamp(p.wear * 1.4)          # little traffic: the tops stay dull brown-grey; busy spot: bright gun-metal
    polished = (mat("metal_plain", mix_hex("#2e2925", "#56585d", tw), rough=round(lerp(0.6, 0.37, tw), 2)) if finish == "cast"
                else mat("iron_wrought", rough=0.45, grime=0.4))
    if finish == "painted":
        body = mat("iron_painted", p.paint, rough=1.35, grime=1.2)
    elif finish == "tarred":
        body = mat("iron_painted", "#2a2c30", rough=1.25)
    else:
        body = bare
    frame_mat = body if finish == "cast" else bare
    bronze = mat("brass", "#e2a868", rough=0.8)
    patina = mat("bronze_aged")
    void = mat("plain", "#050505")
    stone_mat = mat("stone_rough", p.stone, grime=0.3)
    stain_mat = mat("stone_rough", mix_hex(str(p.stone), "#6e4a2c", 0.2 + 0.32 * p.rust), grime=0.3)
    mortar_mat = mat("mortar", "#5c544a")

    nm_a = noise_mask(0.12, seed=p.seed + 1, lo=0.25, hi=0.75)
    nm_b = noise_mask(0.06, seed=p.seed + 2, lo=0.3, hi=0.7)
    nm_c = noise_mask(0.3, seed=p.seed + 3, lo=0.2, hi=0.8)
    nm_d = noise_mask(0.025, seed=p.seed + 4, lo=0.3, hi=0.7)
    rmax = max(hw, hd)

    def raised(co):
        return np.clip((co[:, 2] - (zp + 0.0008)) / (zc - zp - 0.0012), 0.0, 1.0)

    def centre_bias(co):
        return 1.0 - 0.45 * np.clip(np.hypot(co[:, 0], co[:, 1]) / rmax, 0.0, 1.0) ** 1.5

    def edge_near(co):
        """1 right at the frame / cover gap, 0 from 30 mm inwards."""
        if is_round:
            d = hw - np.hypot(co[:, 0], co[:, 1])
        else:
            d = np.minimum(hw - np.abs(co[:, 0]), hd - np.abs(co[:, 1]))
        return np.clip(1.0 - d / 0.03, 0.0, 1.0)

    cov = []          # everything that belongs to the loose cover (turned / rocked as one)
    fixed = []
    void_acc = Acc()
    letters = Acc()
    lugs = Acc()
    tops = Acc()      # trafficked top faces of the tread: polished bare iron
    keys = []
    KW = KL = 0.03

    # ---------------------------------------------------------------- frame
    hwF, hdF = hw + gap + fw, hd + gap + fw
    r_out = hwF if is_round else 0.008

    def frame_outline(ins, k=0):
        if is_round:
            return outline(hwF - ins, hdF - ins, hwF - ins, SEG)
        r = lerp(r_out, rc + gap, min(ins / fw, 1.0)) - max(ins - fw, 0.0)
        return outline(hwF - ins, hdF - ins, max(r, 0.0004), SEG, NS)

    frame = ring_mesh("frame", frame_outline,
                      [(0.0, 0.0), (0.0, zf - 0.0012), (0.0012, zf), (fw - 0.001, zf), (fw, zf - 0.001),
                       (fw, 0.0025), (fw + gap + 0.006, 0.0025)])
    smooth(frame, 40)
    assign(frame, frame_mat)
    uv_box(frame)
    fixed.append(frame)

    # ---------------------------------------------------------------- cover
    rim = clamp(p.size * 0.037, 0.013, 0.024) if not is_gully else 0.018
    rin = 0.0
    c_pitch = clamp(p.size * 0.072, 0.03, 0.048)
    txt1, txt2 = str(p.text).strip(), str(p.text2).strip()
    chip = clamp(p.wear * (1.8 if finish == "cast" else 1.3), 0.0, 1.0)     # share of coated lugs worn through on top

    def lug(acc_rng, pts, z0, z1, cx, cy, **kw):
        """One tread lug; its top goes to the polished set when traffic has worn it bare."""
        worn = acc_rng.u() < chip * (1.0 - 0.45 * min(hypot(cx, cy) / rmax, 1.0) ** 2)
        tl = (acc_rng.u(-0.03, 0.03), acc_rng.u(-0.03, 0.03))
        prism(lugs, pts, z0, z1, top=tops if worn else None, tilt=tl, **kw)

    if not is_gully:
        tb = clamp(p.size * (0.105 if is_round else 0.13), 0.036, 0.066)
        ringw = 0.009
        if is_round:
            has_txt = bool(txt1 or txt2)
            if has_txt:
                bounds = [rim, rim + tb, rim + tb + ringw]
                zs = [zc, zp, zc, zp]
            else:
                bounds, zs = [rim], [zc, zp]
            r_field = hw - bounds[-1] - 0.001
            fx0 = fy0 = -r_field
            fx1 = fy1 = r_field
        else:
            bounds, zs = [rim], [zc, zp]
            ix, iy = hw - rim - 0.001, hd - rim - 0.001
            fx0, fx1 = -ix, ix
            fy1 = iy - (tb + ringw if txt1 else 0.0)
            fy0 = -iy + (tb + ringw if txt2 else 0.0)
            r_field = min(ix, iy)
        prof = levels_prof(bounds, zs, zb)
        bl = bounds[-1] + 0.001
        if is_round:
            segs = [SEG] * 5 + [SEG // 2] * (len(prof) - 5)
            rf = hw - bl
            prof += [(bl + 0.33 * rf, zp), (bl + 0.62 * rf, zp), (bl + 0.86 * rf, zp)]
            segs += [SEG // 2, SEG // 4, SEG // 4]
        else:
            segs = [SEG] * len(prof)
            mh = min(hw, hd) - bl
            prof += [(bl + 0.4 * mh, zp), (bl + 0.8 * mh, zp)]
            segs += [SEG, SEG]
        plate = ring_mesh("plate", lambda ins, k: outline(hw - ins, hd - ins, max(rc - ins, 0.0004), segs[k], NS), prof, cap=True)
        smooth(plate, 40)
        cov.append(plate)

        # lifting keyholes with raised lips (dark pocket inside)
        KL = clamp(p.size * 0.09, 0.03, 0.05)
        KW = KL * 0.5
        kx = (r_field if is_round else fx1) - KW / 2 - 0.012
        # centre boss with the emblem
        rb = clamp(p.size * 0.115, 0.035, 0.085)
        fcy = (fy0 + fy1) / 2
        room = min(kx - KW / 2, (fy1 - fy0) / 2) - 0.012
        if p.emblem != "none" and room > 0.03:
            rb = min(rb, room)
            rin = rb + 0.004
            boss = ring_mesh("boss", lambda ins, k: outline(rb - ins, rb - ins, rb - ins, 8),
                             [(0.0, zp - 0.0004), (0.0013, zc), (0.0058, zc), (0.0071, zp - 0.0004)], off=(0.0, fcy))
            smooth(boss, 40)
            cov.append(boss)
            # bronze roundel let into the boss, emblem standing on it
            rr_ = rb - 0.0062
            zr0 = zp + 0.0012
            roundel = ring_mesh("roundel", lambda ins, k: outline(rr_ - ins, rr_ - ins, rr_ - ins, 6),
                                [(0.0, zr0 - 0.0006), (0.0012, zr0), (rr_ * 0.5, zr0 + 0.0003)], cap=True, off=(0.0, fcy))
            smooth(roundel, 40)
            em = Acc()
            ze = zc + 0.0002
            if p.emblem == "star":
                ro = rr_ * 0.9
                pts = [((ro if i % 2 == 0 else ro * 0.42) * cos(radians(90 + i * 36)),
                        (ro if i % 2 == 0 else ro * 0.42) * sin(radians(90 + i * 36)) + fcy) for i in range(10)]
                prism(em, pts, zr0, ze - 0.0008, inset=min(0.0012, ro * 0.05), dome=0.0016)     # faceted, ridged star
            else:
                s = rr_ * 1.42
                oy = fcy - 0.05 * s
                crown = [(-0.4, -0.16), (0.4, -0.16), (0.5, 0.3), (0.26, 0.04), (0.0, 0.4), (-0.26, 0.04), (-0.5, 0.3)]
                prism(em, [(x * s, y * s + oy) for x, y in crown], zr0, ze, inset=0.0012)
                prism(em, rect_pts(0, -0.3 * s + oy, 0.84 * s, 0.11 * s), zr0, ze + 0.0004, inset=0.0012)
                for (bx, by) in ((0.5, 0.36), (0.0, 0.47), (-0.5, 0.36)):
                    prism(em, [(bx * s + 0.065 * s * cos(radians(a)), by * s + oy + 0.065 * s * sin(radians(a))) for a in range(0, 360, 60)],
                          zr0, ze, inset=0.001, dome=0.0008)
            emb = em.obj("emblem")
            for o, m_ in ((roundel, patina), (emb, bronze)):
                assign(o, m_)
                uv_box(o)
                cov.append(o)
        if kx - KW / 2 > rin + 0.008:
            keys = [(kx, fcy), (-kx, fcy)]
        else:
            keys = [(0.0, fcy)] if rin == 0.0 else []
        rimw = min(0.005, KW * 0.3)
        for i, (qx, qy) in enumerate(keys):
            ko = lambda ins, k=0: outline(KW / 2 - ins, KL / 2 - ins, max(KW / 2 - ins, 0.0004), 4)
            lip = ring_mesh(f"key{i}", ko, [(0.0, zp - 0.0004), (0.0013, zc), (rimw - 0.0008, zc), (rimw, zp + 0.0008)], off=(qx, qy))
            smooth(lip, 40)
            cov.append(lip)
            n0 = len(ko(rimw))
            void_acc.add([(x + qx, y + qy, zp + 0.0008) for x, y in ko(rimw)], [tuple(range(n0))])

        # ---- founder's lettering
        depth = zc - zp + 0.0003
        if is_round:
            rt = hw - rim - tb / 2
            cap_h = tb * 0.6
            both = bool(txt1) and bool(txt2)
            avail = rt * radians(148 if both else 300)
            emit_text(letters, txt1, cap_h, depth, zp - 0.0003, avail,
                      lambda x: (rt * cos(pi / 2 - x / rt), rt * sin(pi / 2 - x / rt), -x / rt))
            emit_text(letters, txt2, cap_h, depth, zp - 0.0003, avail,
                      lambda x: (rt * cos(-pi / 2 + x / rt), rt * sin(-pi / 2 + x / rt), x / rt))
            if both:        # separating studs at 3 and 9 o'clock
                for sx in (-1, 1):
                    prism(lugs, [(sx * rt + tb * 0.2 * cos(radians(a)), tb * 0.2 * sin(radians(a))) for a in range(0, 360, 60)],
                          zp - 0.0004, zc, inset=0.002, dome=0.0006)
        else:
            cap_h = tb * 0.62
            for body_, ys, yb in ((txt1, iy - tb / 2, fy1 + ringw / 2), (txt2, -iy + tb / 2, fy0 - ringw / 2)):
                if not body_:
                    continue
                emit_text(letters, body_, cap_h, depth, zp - 0.0003, 2 * ix - 0.02, lambda x, ys=ys: (x, ys, 0.0))
                prism(lugs, rect_pts(0, yb, 2 * ix + 0.002, ringw - 0.002), zp - 0.0004, zc, inset=0.0015)

        # ---- tread pattern
        def inside(x, y, m=0.0):
            r = hypot(x, y - fcy)
            if r < rin + m:
                return False
            if is_round:
                if hypot(x, y) > r_field - m:
                    return False
            elif not (fx0 + m <= x <= fx1 - m and fy0 + m <= y <= fy1 - m):
                return False
            for (qx, qy) in keys:
                if abs(x - qx) < KW / 2 + m and abs(y - qy) < KL / 2 + m:
                    return False
            return True

        zl0 = zp - 0.0004
        jr = rng.fork("lugs")

        def ztop(x, y):
            # the middle of the cover is trodden a little lower; castings are never dead level
            return zc - 0.0005 * p.wear * max(0.0, 1.0 - hypot(x, y) / max(r_field, 1e-3)) - jr.u(0.0, 0.00025)

        c = c_pitch
        ext = int(max(hw, hd) / c) + 2
        if p.pattern == "chequer":
            for i in range(-ext, ext):
                for j in range(-ext, ext):
                    cx, cy = (i + 0.5) * c, (j + 0.5) * c + fcy
                    for o in (-1, 1):
                        if (i + j) % 2 == 0:
                            bx_, by_ = cx, cy + o * c * 0.235
                            bar = rect_pts(bx_, by_, c * 0.84, c * 0.33)
                        else:
                            bx_, by_ = cx + o * c * 0.235, cy
                            bar = rect_pts(bx_, by_, c * 0.33, c * 0.84)
                        if all(inside(x, y, 0.003) for x, y in bar):
                            lug(jr, bar, zl0, ztop(cx, cy), bx_, by_, inset=0.0022)
        elif p.pattern == "diamond":
            g = c * 0.62
            for i in range(-2 * ext, 2 * ext):
                for j in range(-2 * ext, 2 * ext):
                    cx, cy = i * g, j * g + fcy
                    a = 45.0 if (i + j) % 2 == 0 else -45.0
                    l, w = c * 0.66, c * 0.27
                    loz = [(cx + x * cos(radians(a)) - y * sin(radians(a)), cy + x * sin(radians(a)) + y * cos(radians(a)))
                           for x, y in ((l / 2, 0), (0, w / 2), (-l / 2, 0), (0, -w / 2))]
                    if all(inside(x, y, 0.003) for x, y in loz):
                        lug(jr, loz, zl0, ztop(cx, cy), cx, cy, inset=0.0018)
        elif p.pattern == "studs":
            s = c * 0.8
            rs = c * 0.29
            if is_round:
                r = max(rin, 0.0) + s * (0.75 if rin > 0 or keys and keys[0][0] == 0 else 0.0)
                spots = []
                k = 0
                while r < r_field - rs:
                    nring = max(1, int(2 * pi * r / s)) if r > 1e-6 else 1
                    spots += [(r * cos(2 * pi * (q + 0.5 * (k % 2)) / nring), r * sin(2 * pi * (q + 0.5 * (k % 2)) / nring)) for q in range(nring)]
                    r += s
                    k += 1
            else:
                spots = [((i + 0.5 * (j % 2)) * s, j * s * 0.866 + fcy) for i in range(-ext - 1, ext + 2) for j in range(-ext - 1, ext + 2)]
            for (cx, cy) in spots:
                if inside(cx, cy, rs + 0.003):
                    lug(jr, [(cx + rs * cos(radians(a)), cy + rs * sin(radians(a))) for a in range(0, 360, 60)],
                        zl0, ztop(cx, cy) - 0.0005, cx, cy, inset=0.0025, dome=0.0009)
        else:   # radial sunburst: long and short tapering ribs
            nrib = max(12, 2 * int(pi * r_field / (c * 2.6)))
            r0 = max(rin, 0.012) + 0.006
            for k in range(2 * nrib):
                th = 2 * pi * (k + 0.5) / (2 * nrib)
                ct, st = cos(th), sin(th)
                if is_round:
                    r1 = r_field
                else:
                    r1 = min((fx1 if ct > 0 else -fx0) / max(abs(ct), 1e-6),
                             ((fy1 - fcy) if st > 0 else (fcy - fy0)) / max(abs(st), 1e-6))
                rs0 = r0 if k % 2 == 0 else lerp(r0, r1, 0.55)
                run = []
                steps = int((r1 - rs0) / 0.004) + 1
                for q in range(steps + 1):
                    r = rs0 + (r1 - rs0) * q / steps
                    ok = inside(r * ct, r * st + fcy, 0.006)
                    if ok:
                        run.append(r)
                    if (not ok or q == steps) and run:
                        ra, rb_ = run[0], run[-1]
                        run = []
                        if rb_ - ra < 0.022:
                            continue
                        wfn = lambda r: 0.0038 + 0.0052 * (r - r0) / max(r_field - r0, 1e-3)
                        quad = [(ra, -wfn(ra)), (rb_, -wfn(rb_)), (rb_, wfn(rb_)), (ra, wfn(ra))]
                        rm = 0.5 * (ra + rb_)
                        lug(jr, [(u * ct - v * st, u * st + v * ct + fcy) for u, v in quad], zl0, ztop(rm * ct, rm * st),
                            rm * ct, rm * st, inset=0.0018)
            if rin == 0.0 and not (keys and keys[0][0] == 0):
                prism(lugs, [(0.011 * cos(radians(a)), fcy + 0.011 * sin(radians(a))) for a in range(0, 360, 45)], zl0, zc, inset=0.003, dome=0.001)
    else:
        # ---------------------------------------------------------------- gully grate
        band = ring_mesh("grate_rim", lambda ins, k: outline(hw - ins, hd - ins, max(rc - ins, 0.0004), SEG, NS),
                         [(0.0, zb), (0.0, zc - 0.001), (0.001, zc), (rim - 0.0015, zc), (rim + 0.001, zb)])
        smooth(band, 40)
        cov.append(band)
        tr = clamp(p.size * 0.09, 0.03, 0.045)
        zr = zc - 0.003
        xi = hw - rim - 0.001
        ya, yb = -hd + rim + 0.001, hd - rim - 0.001
        e = 0.0035
        for body_, sgn in ((txt1, -1), (txt2, 1)):
            if not body_:
                continue
            yc = sgn * (hd - rim - tr / 2)
            prism(lugs, rect_pts(0, yc - sgn * 0.001, 2 * xi + 2 * e, tr + 0.004), zb, zr, inset=0.0004)
            emit_text(letters, body_, tr * 0.56, zc - zr + 0.0003, zr - 0.0003, 2 * xi - 0.03, lambda x, yc=yc: (x, yc, 0.0))
            if sgn < 0:
                ya += tr
            else:
                yb -= tr
        void_acc.add([(-xi, ya, 0.0012), (xi, ya, 0.0012), (xi, yb, 0.0012), (-xi, yb, 0.0012)], [(0, 1, 2, 3)])
        ym = (ya + yb) / 2
        spw = 0.02
        jr = rng.fork("bars")
        lug(jr, rect_pts(0, ym, 2 * xi + 2 * e, spw), zb, zc - 0.0004, 0.0, ym, inset=0.002)
        nb = max(3, int(round(2 * xi / 0.043)))
        pitch = 2 * xi / nb
        bw = pitch * 0.46
        if p.pattern in ("chequer", "studs"):
            for i in range(1, nb):
                x = -xi + i * pitch + jr.u(-0.0006, 0.0006)
                lug(jr, rect_pts(x, ym, bw, yb - ya + 2 * e, 0), zb, zc - 0.001 - jr.u(0, 0.0004), x, ym, inset=0.0022)
        else:
            for lo, hi, sg in ((ya - e, ym, 1), (ym, yb + e, -1)):
                sl = (hi - lo) * tan(radians(32)) * sg
                for i in range(-3, nb + 4):
                    x = -xi + i * pitch
                    quad = ccw([(x - bw / 2, lo), (x + bw / 2, lo), (x + bw / 2 + sl, hi), (x - bw / 2 + sl, hi)])
                    quad = clip_x(quad, -xi - e, xi + e)
                    if len(quad) >= 3 and max(q[0] for q in quad) - min(q[0] for q in quad) > 0.008:
                        lug(jr, quad, zb, zc - 0.001 - jr.u(0, 0.0004), x, 0.5 * (lo + hi), inset=0.002)
        # knuckle hinge on the back edge: frame - grate - frame barrels on one pin
        hz, hy, hr = zc - 0.001, hd + gap * 0.5, 0.0065
        for sx in (-1, 1):
            cxh = sx * hw * 0.52
            for k, own in ((-1, fixed), (0, cov), (1, fixed)):
                b = cylinder(f"knuckle{sx}{k}", hr, 0.021, loc=(cxh + k * 0.022, hy, hz), rot=(0, 90, 0), verts=10)
                smooth(b, 40)
                own.append(b)
                assign(b, frame_mat if own is fixed else body)
            pin = cylinder(f"pin{sx}", 0.003, 0.074, loc=(cxh, hy, hz), rot=(0, 90, 0), verts=6)
            assign(pin, frame_mat)
            fixed.append(pin)
            prism(lugs, rect_pts(cxh, hd - 0.006, 0.02, 0.014), zb, zc + 0.0012, inset=0.0015)

    # ---------------------------------------------------------------- finish the cover objects
    for o in cov:
        if not o.material_slots:
            assign(o, body)
            uv_box(o)
    lug_obj = lugs.obj("tread")
    if lug_obj:
        assign(lug_obj, body)
        uv_box(lug_obj)
        cov.append(lug_obj)
    top_obj = tops.obj("tread_tops")
    if top_obj:
        assign(top_obj, polished)
        uv_box(top_obj)
        cov.append(top_obj)
    let_obj = letters.obj("lettering")
    if let_obj:
        assign(let_obj, body)
        smooth(let_obj, 30)
        uv_box(let_obj)
        cov.append(let_obj)
    void_obj = void_acc.obj("void")
    if void_obj:
        assign(void_obj, void)
        (fixed if is_gully else cov).append(void_obj)
    if finish == "cast":
        # rim, ribs, boss rings and letter faces are walked on just like the lugs
        for o in cov:
            if o is top_obj or o is void_obj or o.name in ("emblem", "roundel"):
                continue
            assign_faces(o, polished, lambda c, n: n.z > 0.9 and c.z > zc - 0.0011)

    # the loose cover never sits dead true: turned, shifted in its play and rocking a hair
    tr_ = rng.fork("seat")
    turn = (p.turn + tr_.u(-3.0, 3.0)) if is_round else tr_.u(-0.35, 0.35)
    tilt = (tr_.u(-0.14, 0.14) * 0.3 / max(hw, hd), tr_.u(-0.14, 0.14) * 0.3 / max(hw, hd))
    place(cov, loc=(tr_.u(-0.0008, 0.0008), tr_.u(-0.0008, 0.0008), 0.0), rot=(tilt[0], tilt[1], turn))
    ct_, st_ = cos(radians(turn)), sin(radians(turn))
    wkeys = [(qx * ct_ - qy * st_, qx * st_ + qy * ct_) for qx, qy in keys]

    def key_near(co):
        k = np.zeros(len(co))
        for (qx, qy) in wkeys:
            k = np.maximum(k, np.clip(1.0 - (np.hypot(co[:, 0] - qx, co[:, 1] - qy) - KW * 0.5) / 0.03, 0.0, 1.0))
        return k

    # ---- weathering story: traffic polishes the high points (centre first), rust and dirt keep to the recesses,
    #      the keyholes and the edge against the frame
    wgain = 0.15 + 1.4 * p.wear
    rgain = 0.25 + 1.5 * p.rust

    def polish(co, n):
        return np.clip((0.5 + 0.5 * nm_a(co, n)) * centre_bias(co) * wgain, 0.0, 1.0)

    def recess_rust(co, n):
        low = 1.0 - raised(co)
        return np.clip(low * (0.09 + 0.55 * nm_b(co, n) ** 2 * (0.35 + 0.65 * nm_c(co, n)) + 0.5 * edge_near(co) ** 1.5
                              + 0.5 * key_near(co)) * rgain, 0.0, 0.65)

    coated = finish != "cast"
    for o in cov:
        if o is void_obj or o.name in ("emblem", "roundel"):
            continue
        if o is top_obj:
            paint(o, wear=lambda co, n: np.clip(polish(co, n) + (0.35 if coated else 0.0), 0, 1) * 0.75,
                  special=lambda co, n: np.clip(0.5 * edge_near(co) * nm_b(co, n) * rgain + (0.25 * nm_d(co, n) * rgain if coated else 0.0), 0, 0.6),
                  grime=lambda co, n: (1.0 - polish(co, n)) * 0.6)
        elif coated:
            # paint / tar survives in the recesses and the lettering band, chips on the crowns, rust bleeds from below
            paint(o,
                  wear=lambda co, n: np.clip(raised(co) * (0.25 + 0.75 * nm_d(co, n)) * centre_bias(co) * wgain * 0.8, 0, 0.85),
                  special=lambda co, n: np.clip(recess_rust(co, n) * 0.8 + raised(co) * nm_b(co, n) * 0.35 * rgain, 0, 1),
                  grime=lambda co, n: np.clip(0.3 + (1.0 - raised(co)) * (0.25 + 0.3 * nm_c(co, n)), 0, 1))
        else:
            paint(o,
                  wear=lambda co, n: 0.12 + 0.6 * raised(co) * polish(co, n),
                  special=recess_rust,
                  grime=lambda co, n: (1.0 - raised(co)) * (0.15 + 0.25 * nm_c(co, n) + 0.3 * edge_near(co)) + 0.6 * recess_rust(co, n)
                  + raised(co) * (1.0 - polish(co, n)) * 0.5)
    for o in cov:
        if o.name == "emblem":      # rubbed golden on the high points
            paint(o, wear=lambda co, n: np.clip((co[:, 2] - (zp + 0.0016)) / 0.0018, 0, 1) * (0.55 + 0.45 * p.wear),
                  special=lambda co, n: np.clip(1.0 - (co[:, 2] - (zp + 0.001)) / 0.0015, 0, 1) * 0.25)
        elif o.name == "roundel":   # verdigris and dirt in the dished ground
            paint(o, special=lambda co, n: 0.12 + 0.45 * nm_d(co, n), grime=0.2, wear=0.05)
    # the frame: rusty casting, top rubbed dark by traffic, dirt and rust packed in the seating gap
    paint(frame,
          wear=lambda co, n: 0.06 + np.clip((co[:, 2] - (zf - 0.0015)) / 0.0015, 0, 1) * (0.08 + 0.22 * nm_a(co, n)) * (0.25 + 0.75 * p.wear),
          special=lambda co, n: np.clip((0.08 + 0.5 * nm_b(co, n) ** 2 * (0.3 + 0.7 * nm_c(co, n))) * (0.3 + 0.9 * p.rust)
                                        + 0.7 * np.clip(((zf - 0.0005) - co[:, 2]) / 0.002, 0, 1), 0, 1),
          grime=lambda co, n: np.clip(((zf - 0.0015) - co[:, 2]) / 0.003, 0, 1) * 0.8
          + np.clip((co[:, 2] - (zf - 0.0015)) / 0.0015, 0, 1) * (0.15 + 0.3 * (1.0 - nm_a(co, n))))
    for o in fixed:
        if o.name.startswith(("knuckle", "pin")):
            uv_box(o)
            paint(o, special=0.15 + 0.35 * p.rust, wear=0.2, grime=0.35)

    # ---------------------------------------------------------------- surround
    def sd1(x, y):
        """Distance outside the frame outline (m)."""
        qx, qy = abs(x) - (hwF - r_out), abs(y) - (hdF - r_out)
        return hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - r_out

    def sdf(co):
        qx = np.abs(co[:, 0]) - (hwF - r_out)
        qy = np.abs(co[:, 1]) - (hdF - r_out)
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r_out

    sr = rng.fork("surround")
    stones, mortars, stone_kinds = [], [], []
    z_in, z_out = zf - 0.001, min(0.003, zf - 0.001)        # collar top at the frame / at the paving
    zsk = max(z_out - 0.0013, 0.0008)                       # bedding skirt where it meets the collar
    if p.surround == "stone":
        sw = clamp(p.size * 0.2, 0.09, 0.15)
        cl = 0.004                    # gap to the frame, packed with rusty dirt
        jw = sr.u(0.007, 0.009)       # butt joints
        stn = noise_mask(0.16, seed=p.seed + 11, lo=0.35, hi=0.8)
        stn2 = noise_mask(0.22, seed=p.seed + 12, lo=0.3, hi=0.7)

        def wst(x, y):
            q = np.array([[x, y, 0.0]])
            z3 = np.zeros((1, 3))
            return 0.01 + 0.045 * float(stn(q, z3)[0]) ** 1.3 * (0.4 + 0.6 * float(stn2(q, z3)[0])) * (0.4 + 0.9 * p.rust)

        Sx, Sy = hwF + cl + sw, hdF + cl + sw
        dzs = lambda: (sr.u(-0.0006, 0.0005), sr.u(-0.003, 0.003), sr.u(-0.003, 0.003))
        if is_round:
            R = hwF + cl

            def ztop_c(x, y):
                r = hypot(x, y)
                ro = hypot(*ray_rect(atan2(y, x), Sx, Sy)) if r > 1e-6 else Sx
                return lerp(z_in, z_out, smoothstep(0.0, 1.0, clamp((r - R) / max(ro - R, 1e-6))))

            if p.size >= 0.5:
                joints = [0.0, 90.0, 180.0, 270.0]           # four quarters, joints at the mid-sides
            else:
                j0 = 90.0 * int(sr.u(0, 2))
                joints = [j0, j0 + 180.0]                    # two halves
            for k, th in enumerate(joints):
                th1 = joints[k + 1] if k + 1 < len(joints) else joints[0] + 360.0
                st, kd = sector_stone(f"collar{k}", R, Sx, Sy, th, th1, 4 * SEG, jw, wst, ztop_c, dzs(), stone_mat, stain_mat)
                stones.append(st)
                stone_kinds.append(kd)
            angs = [radians(a * 7.5) for a in range(48)]
            bed = bed_mesh("bed", angs, lambda a: ((R - 0.004) * cos(a), (R - 0.004) * sin(a)), lambda a: ray_rect(a, Sx, Sy),
                           lambda x, y: ztop_c(x, y) - BED)
        else:
            X, Y = hwF + cl, hdF + cl

            def ztop_c(x, y):
                return lerp(z_in, z_out, smoothstep(0.0, sw, sd1(x, y) - cl))

            for k, (org, du, dv, L, Lf) in enumerate((((X, -Y), (-1, 0), (0, -1), 2 * X + sw, 2 * X),
                                                      ((-X, -Y), (0, 1), (-1, 0), 2 * Y + sw, 2 * Y),
                                                      ((-X, Y), (1, 0), (0, 1), 2 * X + sw, 2 * X),
                                                      ((X, Y), (0, -1), (1, 0), 2 * Y + sw, 2 * Y))):
                st, kd = slab_stone(f"collar{k}", org, du, dv, L, sw, Lf, jw, wst, ztop_c, dzs(), stone_mat, stain_mat)
                stones.append(st)
                stone_kinds.append(kd)
            angs = sorted({round(radians(a * 15.0), 6) for a in range(24)}
                          | {round(atan2(sy * b, sx * a_) % (2 * pi), 6) for sx in (-1, 1) for sy in (-1, 1)
                             for (a_, b) in ((hwF - 0.004, hdF - 0.004), (Sx, Sy))})
            bed = bed_mesh("bed", angs, lambda a: ray_rect(a, hwF - 0.004, hdF - 0.004), lambda a: ray_rect(a, Sx, Sy),
                           lambda x, y: ztop_c(x, y) - BED)
        mortars.append(bed)
        mortars.append(skirt_mesh("skirt", Sx, Sy, 0.006, sr.u(0.019, 0.023), zsk, sr))
    elif p.surround == "setts":
        sett = Acc()
        dep0 = 0.13
        cl = 0.005
        zs_top = zf - 0.002
        fall = lambda x, y: -(zs_top - 0.0036) * smoothstep(0.0, dep0, sd1(x, y))
        if is_round:
            R = hwF + cl
            n = max(8, int(round(2 * pi * (R + 0.02) / 0.105)))
            a0 = sr.u(0, 6.28)
            cuts = [a0 + 2 * pi * (k + sr.u(-0.12, 0.12)) / n for k in range(n)]
            for k in range(n):
                a, b = cuts[k], cuts[(k + 1) % n] + (2 * pi if k == n - 1 else 0.0)
                dep = dep0 + sr.u(-0.012, 0.012)
                am = 0.5 * (a + b)
                quad = [(R * cos(a), R * sin(a)), ((R + dep) * cos(a), (R + dep) * sin(a)), ((R + dep) / cos(0.5 * (b - a)) * cos(am), (R + dep) / cos(0.5 * (b - a)) * sin(am)),
                        ((R + dep) * cos(b), (R + dep) * sin(b)), (R * cos(b), R * sin(b)), (R * cos(am), R * sin(am))]
                prism(sett, ccw(quad), 0.0, zs_top + sr.u(-0.0012, 0.0008), inset=0.0055, dome=sr.u(0.0006, 0.0018), zfn=fall)
            Ro = R + dep0
            angs = [radians(a * 7.5) for a in range(48)]
            bed = bed_mesh("bed", angs, lambda a: ((R - 0.006) * cos(a), (R - 0.006) * sin(a)), lambda a: ((Ro + 0.012) * cos(a), (Ro + 0.012) * sin(a)),
                           lambda x, y: 0.0016, fr=(0.0, 1.0))
            mortars.append(skirt_mesh("skirt", Ro + 0.004, Ro + 0.004, Ro + 0.004, sr.u(0.019, 0.023), zsk, sr))
        else:
            X, Y = hwF + 0.002, hdF + 0.002
            dep = dep0 * 0.85
            rows = [((-X - dep, -Y - dep), (X + dep, -Y - dep), (0, 1)), ((-X - dep, Y), (X + dep, Y), (0, 1)),
                    ((-X - dep, -Y), (-X - dep, Y), (1, 0)), ((X, -Y), (X, Y), (1, 0))]
            for (a, b, nrm) in rows:
                ln = hypot(b[0] - a[0], b[1] - a[1])
                n = max(2, int(round(ln / 0.11)))
                cuts = [0.0] + [(k + sr.u(-0.13, 0.13)) / n for k in range(1, n)] + [1.0]
                for k in range(n):
                    p0 = (lerp(a[0], b[0], cuts[k]), lerp(a[1], b[1], cuts[k]))
                    p1 = (lerp(a[0], b[0], cuts[k + 1]), lerp(a[1], b[1], cuts[k + 1]))
                    quad = [p0, p1, (p1[0] + nrm[0] * dep, p1[1] + nrm[1] * dep), (p0[0] + nrm[0] * dep, p0[1] + nrm[1] * dep)]
                    prism(sett, ccw(quad), 0.0, zs_top + sr.u(-0.0012, 0.0008), inset=0.0055, dome=sr.u(0.0006, 0.0018), zfn=fall)
            Sx, Sy = X + dep, Y + dep
            angs = sorted({round(radians(a * 15.0), 6) for a in range(24)}
                          | {round(atan2(sy * b, sx * a_) % (2 * pi), 6) for sx in (-1, 1) for sy in (-1, 1)
                             for (a_, b) in ((hwF - 0.004, hdF - 0.004), (Sx - 0.002, Sy - 0.002))})
            bed = bed_mesh("bed", angs, lambda a: ray_rect(a, hwF - 0.004, hdF - 0.004), lambda a: ray_rect(a, Sx - 0.002, Sy - 0.002),
                           lambda x, y: 0.0016, fr=(0.0, 1.0))
            mortars.append(skirt_mesh("skirt", Sx - 0.003, Sy - 0.003, 0.012, sr.u(0.019, 0.023), zsk, sr))
        mortars.append(bed)
        so = sett.obj("setts")
        assign(so, mat("stone_rough", p.stone))
        uv_box(so, scale=3.0)
        g0 = sr.u(0.0, 0.15)
        paint(so,
              grime=lambda co, n: g0 + np.clip(1.0 - sdf(co) / 0.035, 0, 1) * (0.12 + 0.3 * p.rust) * (0.4 + 0.6 * nm_b(co, n)) + 0.12 * nm_c(co, n)
              + 0.5 * (co[:, 2] < 0.001),
              special=lambda co, n: (co[:, 2] < 0.001) * np.clip(nm_a(co, n) * (0.6 + 0.5 * np.clip(co[:, 1] / max(hdF, 0.1), -1, 1)), 0, 1) * 0.7,
              wear=lambda co, n: (co[:, 2] > 0.003) * 0.35 * nm_b(co, n) * p.wear)
        stones.append(so)
    if p.surround == "stone":
        for st, kd in zip(stones, stone_kinds):
            # worn pale arrises, dark damp joints, rusty dirt washed off the frame, a little moss in the shadier joints
            g0 = sr.u(0.0, 0.12)

            def by_kind(vals, co, kd=kd):
                if len(co) != len(kd):
                    print("DBG stone vertex count changed", len(co), len(kd))
                    return np.zeros(len(co))
                return np.array(vals)[kd]

            paint(st,
                  grime=lambda co, n, g0=g0, f=by_kind: np.clip(f((g0, 1.0, 0.3, 0.4 + 0.4 * p.rust, 0.06), co) * (0.75 + 0.25 * nm_b(co, n)), 0, 1),
                  special=lambda co, n, f=by_kind: f((0.0, 0.6, 0.0, 0.0, 0.0), co) * np.clip(nm_a(co, n) * (0.5 + 0.5 * np.clip(co[:, 1] / max(hdF, 0.1), -1, 1)), 0, 1),
                  wear=lambda co, n, f=by_kind: f((0.12 * p.wear, 0.0, 0.35 + 0.4 * p.wear, 0.15, 0.1), co))
    for mo in mortars:
        assign(mo, mortar_mat)
        uv_box(mo)
        paint(mo, grime=lambda co, n: np.clip(0.2 + 0.3 * nm_b(co, n) + np.clip(1.0 - sdf(co) / 0.02, 0, 1) * 0.5, 0, 1),
              special=lambda co, n: 0.18 * nm_a(co, n) * (0.5 + 0.5 * np.clip(co[:, 1] / max(hdF, 0.1), -1, 1)))

    # ---------------------------------------------------------------- dead leaves
    nl = int(p.leaves)
    if nl > 0:
        lr = rng.fork("leaves")
        tints = ["#8a6a30", "#6c4a22"]
        per = {}
        for k in range(nl):
            a, r = lr.u(0, 2 * pi), (hw + fw) * (lr.u(0.0, 1.0) ** 0.6)
            acc = per.setdefault(tints[k % 2], Acc())
            leaf_mesh(acc, r * cos(a), r * sin(a) * (hd / hw), zf + 0.0012, lr.u(0.055, 0.09), lr.u(0, 2 * pi), lr)
        for tint, acc in per.items():
            lo = acc.obj("leaves")
            assign(lo, mat("leaf", tint))
            smooth(lo, 60)
            uv_box(lo)
            paint(lo, grime=lambda co, n: 0.3 + 0.4 * nm_b(co, n))
