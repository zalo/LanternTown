"""Hinged timber A-frame pavement menu board. Two framed leaves (stiles running down to bevel-cut feet,
rails butted between them, a chalkboard panel held in a rebate by a planted bead) are joined at the apex
by brass butt hinges and kept from splaying by a light chain stay on screw eyes at each side. The front
leaf can carry a shaped header board with a hand slot, a chalk ledge with chalk sticks and a wrought
crook with a candle jar that rakes warm light across the hand-chalked menu (title, priced lines with
dotted leaders, border, doodle, footer - all text is parametric and drawn as thin chalk geometry).
"""
from math import sin, cos, radians, degrees, pi, hypot

import bmesh
from mathutils import Matrix, Vector

from tk import *

META = {"name": "A-frame menu board", "class": "prop_small", "mount": "ground",
        "tri_budget": 13000,          # two faces of real chalk lettering + chain links
        "focus": 0.7}
PARAMS = {
    "height": {"default": 1.05, "min": 0.75, "max": 1.45, "desc": "overall height of the open board (m)"},
    "width": {"default": 0.58, "min": 0.42, "max": 0.8, "desc": "overall width (m)"},
    "spread": {"default": 16.0, "min": 10.0, "max": 24.0, "desc": "lean of each leaf from vertical (deg)"},
    "leg": {"default": 0.11, "min": 0.05, "max": 0.3, "desc": "clear leg height below the bottom rail (m)"},
    "top": {"default": "crest", "choices": ["flat", "arched", "crest"], "desc": "header of the front leaf"},
    "frame": {"default": "painted", "choices": ["painted", "oak", "weathered"], "desc": "frame timber finish"},
    "paint": {"default": "#3a6650", "desc": "frame paint colour (stain tint for oak / weathered)"},
    "board_tint": {"default": "#8f8f8f", "desc": "chalkboard tint (grey = darker slate, e.g. #7fae90 green)"},
    "title": {"default": "Menu", "desc": "chalk title, front"},
    "lines": {"default": "Coffee:3|Hot cocoa:4|Apple tart:5|Mulled wine:5",
              "desc": "front menu lines separated by |, optional item:price"},
    "footer": {"default": "open till late", "desc": "small chalk footer line (empty = none)"},
    "back_title": {"default": "Open", "desc": "chalk title on the back leaf"},
    "back_lines": {"default": "Breakfast:7 - 11|Lunch:12 - 3|Supper:6 - late", "desc": "back lines (| separated)"},
    "border": {"default": "double", "choices": ["double", "ticket", "scallop", "none"], "desc": "chalk border"},
    "doodle": {"default": "cup", "choices": ["cup", "wine", "sprig", "none"], "desc": "chalk drawing under the lines"},
    "chalk": {"default": "#f1eee4", "desc": "main chalk colour"},
    "accent": {"default": "#f2c766", "desc": "accent chalk colour (title, footer, doodle)"},
    "chalk_tray": {"default": True, "desc": "chalk ledge with chalk sticks on the front leaf"},
    "candle": {"default": True, "desc": "wrought crook with a candle jar lighting the front"},
    "lamp_intensity": {"default": 0.8, "min": 0.2, "max": 3.0, "desc": "candle jar intensity (cd)"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "scuffing / chipping / grime amount"},
}
VARIANTS = {
    "bistro_oak": {"frame": "oak", "paint": "#8a6238", "top": "arched", "height": 1.22, "width": 0.64, "title": "Bistro",
                   "lines": "Plat du jour:9|Onion soup:6|Cheese board:8|Tarte tatin:5|House red:4|Cider:3",
                   "footer": "bon appetit", "border": "ticket", "doodle": "wine", "accent": "#f08a9a", "seed": 4},
    "tea_green_small": {"height": 0.8, "width": 0.46, "top": "flat", "paint": "#6a1f1c", "board_tint": "#7fae90",
                        "title": "Teas", "lines": "Assam:2|Earl Grey:2|Scones:3", "footer": "", "border": "scallop",
                        "doodle": "cup", "candle": False, "spread": 19.0, "accent": "#ffffff", "seed": 11},
    "weathered_wide": {"frame": "weathered", "paint": "#b8b2a6", "width": 0.78, "height": 0.95, "spread": 21.0,
                       "top": "flat", "title": "Today", "leg": 0.16, "wear": 1.0,
                       "lines": "Fish stew:7|Mussels:8|Crab roll:6|Chowder:5", "footer": "fresh off the quay",
                       "border": "none", "doodle": "sprig", "accent": "#9fd3e6", "chalk_tray": False, "seed": 23},
    "tall_blue": {"height": 1.42, "width": 0.56, "spread": 13.0, "paint": "#2f5688", "top": "crest", "title": "Vins",
                  "lines": "Rouge:4|Blanc:4|Rose:4|Porto:5|Calvados:6|Cognac:7|Kir:5", "footer": "sante !",
                  "border": "ticket", "doodle": "wine", "accent": "#e9a3c9", "leg": 0.2, "seed": 31},
}

T = 0.03          # frame thickness
SW = 0.048        # stile width
RW = 0.058        # top rail height
BRW = 0.072       # bottom rail height
PANEL_SET = 0.009  # panel face set back from the frame face


# ----------------------------------------------------------------------------- chalk drawing
class Chalk:
    """Collects flat chalk marks (ribbons, dots, lettering) drawn in the board plane (x, z) at depth y."""

    def __init__(self, y, r):
        self.y, self.r = y, r
        self.v = {0: [], 1: []}
        self.f = {0: [], 1: []}

    def stroke(self, pts, w=0.003, col=0, closed=False, taper=0.0, wob=0.0005):
        n = len(pts)
        if n < 2:
            return
        ph = [self.r.u(0, 6.28) for _ in range(3)]
        V, F = self.v[col], self.f[col]
        base = len(V)
        s = 0.0
        for i in range(n):
            p = pts[i]
            pa = pts[(i - 1) % n] if (closed or i > 0) else p
            pb = pts[(i + 1) % n] if (closed or i < n - 1) else p
            if i > 0:
                s += hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1])
            t1 = (p[0] - pa[0], p[1] - pa[1])
            t2 = (pb[0] - p[0], pb[1] - p[1])
            l1, l2 = hypot(*t1), hypot(*t2)
            if l1 < 1e-9:
                t1, l1 = t2, l2
            if l2 < 1e-9:
                t2, l2 = t1, l1
            l1, l2 = max(l1, 1e-9), max(l2, 1e-9)
            t1 = (t1[0] / l1, t1[1] / l1)
            t2 = (t2[0] / l2, t2[1] / l2)
            tx, tz = t1[0] + t2[0], t1[1] + t2[1]
            lt = hypot(tx, tz)
            if lt < 1e-6:
                tx, tz, lt = t1[0], t1[1], 1.0
            tx, tz = tx / lt, tz / lt
            nx, nz = -tz, tx
            mitre = 1.0 / max(0.55, nx * -t1[1] + nz * t1[0])
            wi = w * (1.0 + 0.22 * sin(s * 70.0 + ph[0]))
            if taper > 0.0 and not closed:
                wi *= taper + (1.0 - taper) * sin(pi * i / (n - 1)) ** 0.6
            o = wob * sin(s * 38.0 + ph[1]) + 0.5 * wob * sin(s * 91.0 + ph[2])
            hw = wi * 0.5 * mitre
            V.append((p[0] + nx * (o + hw), self.y, p[1] + nz * (o + hw)))
            V.append((p[0] + nx * (o - hw), self.y, p[1] + nz * (o - hw)))
        m = n if closed else n - 1
        for i in range(m):
            a, b = base + 2 * i, base + 2 * ((i + 1) % n)
            F.append((a, a + 1, b + 1, b))

    def line(self, p0, p1, w=0.003, col=0, step=0.04, **kw):
        k = max(1, int(hypot(p1[0] - p0[0], p1[1] - p0[1]) / step))
        self.stroke([(p0[0] + (p1[0] - p0[0]) * i / k, p0[1] + (p1[1] - p0[1]) * i / k) for i in range(k + 1)], w, col, **kw)

    def dot(self, x, z, d=0.002, col=0):
        V, F = self.v[col], self.f[col]
        b = len(V)
        V += [(x - d, self.y, z), (x, self.y, z - d), (x + d, self.y, z), (x, self.y, z + d)]
        F.append((b, b + 1, b + 2, b + 3))

    def text(self, body, size, x, z, col=0, align="C", font="sans", maxw=None, rot=0.0, center=False):
        """Front faces of a text mesh mapped into the board plane. Returns (x_left, x_right)."""
        if not body.strip():
            return (x, x)
        o = text("chalk_txt", body, size=size, depth=0.002, align="LEFT", font=font)
        me = o.data
        vs = [(v.co.x, v.co.y, v.co.z) for v in me.vertices]
        polys = [(tuple(pl.vertices), pl.normal.z) for pl in me.polygons]
        delete([o])
        if not vs:
            return (x, x)
        zmax = max(v[2] for v in vs)
        front = [(idx, nz) for idx, nz in polys if abs(nz) > 0.9 and all(vs[i][2] > zmax * 0.5 for i in idx)]
        used = sorted({i for idx, _ in front for i in idx})
        if not used:
            return (x, x)
        # chalk has no thickness: keep the glyph faces only, simplified (collinear outline points dissolved)
        bm = bmesh.new()
        bv = {i: bm.verts.new((vs[i][0], vs[i][1], 0.0)) for i in used}
        for idx, nz in front:
            try:
                bm.faces.new([bv[i] for i in (idx if nz > 0 else idx[::-1])])
            except ValueError:
                pass
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=size * 0.004)
        bmesh.ops.dissolve_limit(bm, angle_limit=radians(17.0), verts=bm.verts[:], edges=bm.edges[:])
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.verts.index_update()
        gv = [(v.co.x, v.co.y) for v in bm.verts]
        gf = [tuple(v.index for v in fc.verts) for fc in bm.faces if fc.calc_area() > 1e-10]
        bm.free()
        if not gv or not gf:
            return (x, x)
        x0, x1 = min(v[0] for v in gv), max(v[0] for v in gv)
        y0, y1 = min(v[1] for v in gv), max(v[1] for v in gv)
        w = max(x1 - x0, 1e-6)
        s = min(1.0, maxw / w) if maxw else 1.0
        ax = x0 if align == "L" else (x1 if align == "R" else (x0 + x1) * 0.5)
        ay = (y0 + y1) * 0.5 if center else 0.0
        c, sn = cos(radians(rot)), sin(radians(rot))
        V, F = self.v[col], self.f[col]
        base = len(V)
        for gx, gy in gv:
            lx, ly = (gx - ax) * s, (gy - ay) * s
            V.append((x + lx * c - ly * sn, self.y, z + lx * sn + ly * c))
        for fc in gf:
            F.append(tuple(base + i for i in fc))
        return (x + (x0 - ax) * s, x + (x1 - ax) * s)

    def build(self, name, mats):
        out = []
        for k in (0, 1):
            if self.f[k]:
                o = mesh_object(f"{name}_{k}", self.v[k], self.f[k])
                assign(o, mats[k])
                flat(o)
                out.append(o)
        return out


def _arc(cx, cz, rx, rz, a0, a1, n):
    return [(cx + rx * cos(radians(a0 + (a1 - a0) * i / n)), cz + rz * sin(radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def _border(ck, kind, xh, zt, zb, r):
    if kind == "none":
        return

    def rect(d, w):
        pts = []
        cs = [(-xh + d, zt - d), (xh - d, zt - d), (xh - d, zb + d), (-xh + d, zb + d)]
        cs = [cs[0], cs[3], cs[2], cs[1]]            # counter-clockwise
        for i in range(4):
            p0, p1 = cs[i], cs[(i + 1) % 4]
            k = max(1, int(hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.05))
            pts += [(p0[0] + (p1[0] - p0[0]) * j / k, p0[1] + (p1[1] - p0[1]) * j / k) for j in range(k)]
        ck.stroke(pts, w, 0, closed=True)

    if kind == "double":
        rect(0.0, 0.0042)
        rect(0.0105, 0.0018)
        for sx in (-1, 1):
            for z in (zt - 0.026, zb + 0.026):
                ck.dot(sx * (xh - 0.026), z, 0.0045, 1)
                ck.stroke(_arc(sx * (xh - 0.026), z, 0.011, 0.011, 0, 360, 10)[:-1], 0.0016, 0, closed=True, wob=0.0002)
    elif kind == "ticket":
        rc = 0.034
        pts = []
        corners = [(-xh, zt, 270, 360), (-xh, zb, 0, 90), (xh, zb, 90, 180), (xh, zt, 180, 270)]
        # counter-clockwise: start top-left going down
        seq = [corners[0], corners[1], corners[2], corners[3]]
        arcs = []
        for (cx, cz, a0, a1) in seq:
            arcs.append(_arc(cx, cz, rc, rc, a1, a0, 6))
        # arcs: TL runs from (-xh+rc, zt) -> (-xh, zt-rc); then down the left edge ...
        for i in range(4):
            a = arcs[i]
            pts += a
            nxt = arcs[(i + 1) % 4][0]
            p0 = a[-1]
            k = max(1, int(hypot(nxt[0] - p0[0], nxt[1] - p0[1]) / 0.05))
            pts += [(p0[0] + (nxt[0] - p0[0]) * j / k, p0[1] + (nxt[1] - p0[1]) * j / k) for j in range(1, k)]
        ck.stroke(pts, 0.0036, 0, closed=True)
        for sx in (-1, 1):
            for z in (zt - 0.011, zb + 0.011):
                ck.dot(sx * (xh - 0.011), z, 0.005, 1)
    elif kind == "scallop":
        rect(0.0, 0.0022)
        rs = 0.013
        for (p0, p1, nrm) in (((-xh, zt), (xh, zt), (0, -1)), ((xh, zt), (xh, zb), (-1, 0)),
                              ((xh, zb), (-xh, zb), (0, 1)), ((-xh, zb), (-xh, zt), (1, 0))):
            L = hypot(p1[0] - p0[0], p1[1] - p0[1])
            k = max(2, int(round(L / (2 * rs))))
            dx, dz = (p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L
            rr = L / k / 2
            pts = []
            for j in range(k):
                c0 = rr * (2 * j + 1)
                for q in range(5 if j < k - 1 else 6):
                    a = pi * q / 5
                    u, v = c0 - rr * cos(a), rr * 0.8 * sin(a)
                    pts.append((p0[0] + dx * u + nrm[0] * v, p0[1] + dz * u + nrm[1] * v))
            ck.stroke(pts, 0.0026, 1, wob=0.0003)


def _doodle(ck, kind, cx, cz, s, r):
    """Chalk drawing roughly s tall centred on (cx, cz)."""
    w0 = clamp(s * 0.03, 0.0022, 0.0042)

    def P(pts):
        return [(cx + s * a, cz + s * b) for a, b in pts]

    def smooth_pts(pts, n):
        q = resample_path([(a, 0.0, b) for a, b in pts], n=n, smooth_path=True)
        return [(float(v[0]), float(v[2])) for v in q]

    if kind == "cup":
        ck.stroke(P(smooth_pts([(-0.3, 0.1), (-0.27, -0.12), (-0.17, -0.27), (0.0, -0.31), (0.17, -0.27), (0.27, -0.12), (0.3, 0.1)], 16)), w0, 0)
        ck.stroke(P(_arc(0.0, 0.1, 0.3, 0.06, 0, 360, 16)[:-1]), w0 * 0.8, 0, closed=True)
        ck.stroke(P(_arc(0.29, -0.07, 0.15, 0.12, 80, -100, 9)), w0, 0)
        ck.stroke(P(_arc(0.0, -0.33, 0.5, 0.075, 150, 390, 16)), w0, 0, taper=0.4)
        for k, x in enumerate((-0.14, 0.0, 0.14)):
            ph = r.u(0, 1.0)
            ck.stroke(P([(x + 0.035 * sin(ph + q * 1.25), 0.2 + 0.045 * q + (0.03 if k == 1 else 0.0)) for q in range(7)]), w0 * 0.8, 1, taper=0.3)
    elif kind == "wine":
        ck.stroke(P(smooth_pts([(-0.2, 0.42), (-0.22, 0.25), (-0.15, 0.1), (0.0, 0.03), (0.15, 0.1), (0.22, 0.25), (0.2, 0.42)], 14)), w0, 0)
        ck.stroke(P(_arc(0.0, 0.42, 0.2, 0.04, 0, 360, 14)[:-1]), w0 * 0.8, 0, closed=True)
        ck.stroke(P(_arc(0.0, 0.22, 0.205, 0.035, 180, 360, 8)), w0 * 0.8, 1)
        ck.line(P([(0.0, 0.03)])[0], P([(0.0, -0.36)])[0], w0, 0, step=s * 0.1)
        ck.stroke(P(_arc(0.0, -0.39, 0.19, 0.045, 0, 360, 14)[:-1]), w0, 0, closed=True)
        for (a, b, c, d) in ((0.33, 0.47, 0.42, 0.56), (0.38, 0.36, 0.5, 0.38), (-0.33, 0.47, -0.42, 0.56), (-0.38, 0.36, -0.5, 0.38)):
            ck.line(P([(a, b)])[0], P([(c, d)])[0], w0 * 0.8, 1, taper=0.4)
    elif kind == "sprig":
        L = 1.6
        stem = [(-L / 2 + L * i / 16, 0.06 * sin(i / 16 * pi * 2.0)) for i in range(17)]
        ck.stroke(P(stem), w0, 0, taper=0.4)
        for i in range(2, 15, 2):
            x, z = stem[i]
            for sg in (-1, 1):
                ang = radians(sg * (48 + r.jitter(0, 8)))
                ll = 0.16 * (1.0 - abs(i - 8) / 14.0)
                dx, dz = cos(ang), sin(ang)
                tip = (x + dx * ll, z + dz * ll)
                mid = (x + dx * ll * 0.5, z + dz * ll * 0.5)
                bw = ll * 0.26
                leafp = [(x, z), (mid[0] - dz * bw, mid[1] + dx * bw), tip, (mid[0] + dz * bw, mid[1] - dx * bw)]
                ck.stroke(P(leafp), w0 * 0.8, 1, closed=True, wob=0.0002)


def _menu(ck, r, title, lines, footer, xh, zt, zb, border, doodle):
    bi = 0.014
    bx, bt, bb = xh - bi, zt - bi, zb + bi
    _border(ck, border, bx, bt, bb, r)
    m = 0.024 if border != "none" else 0.012
    cx, ct, cb = bx - m, bt - m, bb + m
    width = 2 * cx
    z = ct
    if title.strip():
        ts = min(clamp(width * 0.3, 0.05, 0.14), (ct - cb) * 0.3)
        z -= ts * 0.6
        x0, x1 = ck.text(title, ts, r.jitter(0, 0.006), z, col=1, align="C", font="script", maxw=width * 0.9,
                         rot=r.jitter(0.0, 1.6), center=True)
        z -= ts * 0.6 + 0.014
        hw = min(width * 0.46, (x1 - x0) * 0.5 + 0.05)
        ph = r.u(0, 1)
        ck.stroke([(-hw + 2 * hw * i / 24, z + 0.005 * sin(i / 24 * 2 * pi + ph)) for i in range(25)], 0.0038, 0, taper=0.2)
        ck.line((-hw * 0.6, z - 0.009), (hw * 0.6, z - 0.0095), 0.0016, 0, taper=0.3)
        z -= 0.034
    if footer.strip() and z - cb > 0.16:
        fs = clamp(width * 0.115, 0.024, 0.05)
        ck.text(footer, fs, r.jitter(0, 0.008), cb + fs * 0.55, col=1, align="C", font="script", maxw=width * 0.92,
                rot=r.jitter(0.0, 1.2), center=True)
        cb += fs * 1.1 + 0.022
    rows = [s.strip() for s in lines.split("|") if s.strip()]
    avail = max(z - cb, 0.0)
    rows = rows[:max(1, int(avail / 0.03))]
    n = len(rows)
    dh = 0.0
    if doodle != "none":
        dh = clamp(avail * 0.27, 0.0, max(0.2, width * 0.45))
        if n and (avail - dh) / n < 0.042:
            dh = avail - 0.042 * n
        if dh < 0.075:
            dh = 0.0
    if n:
        pitch = min(clamp(width * 0.2, 0.07, 0.13), (avail - dh) / n)
        ls = clamp(pitch * 0.7, 0.014, 0.085)
        for i, row in enumerate(rows):
            zl = z - pitch * (i + 0.5) + r.jitter(0, 0.0025)
            rot = r.jitter(0.0, 0.9)
            if ":" in row:
                item, price = [s.strip() for s in row.split(":", 1)]
                _, xr = ck.text(item, ls, -cx + r.u(0, 0.006), zl, col=0, align="L", font="serif", maxw=width * 0.66, rot=rot)
                xl, _ = ck.text(price, ls, cx - r.u(0, 0.006), zl, col=1, align="R", font="serif", maxw=width * 0.26, rot=rot)
                xd = xr + 0.014
                while xd < xl - 0.012:
                    ck.dot(xd, zl - ls * 0.3 + r.jitter(0, 0.0008), 0.002, 0)
                    xd += 0.014
            else:
                ck.text(row, ls, r.jitter(0, 0.006), zl, col=0, align="C", font="serif", maxw=width * 0.95, rot=rot)
        z -= pitch * n
    if dh > 0.0:
        hh = min(dh, z - cb)
        _doodle(ck, doodle, 0.0, (z + cb) * 0.5 - (0.03 * hh if doodle == "cup" else 0.0), hh * (0.95 if doodle != "sprig" else min(0.5, width / hh * 0.5)), r)


# ----------------------------------------------------------------------------- joinery
def _wedge(name, x0, x1, y0, y1, ztop, zb):
    """Square timber with its lower end cut on the slope zb(y) so the foot sits flat on the paving
    (extra edge rings just above the foot give the scuff / splash masks something to bite on)."""
    v, f = [], []
    rings = [0.0, 0.03, 0.075, 0.15]
    for d in rings:
        v += [(x0, y0, zb(y0) + d), (x1, y0, zb(y0) + d), (x1, y1, zb(y1) + d), (x0, y1, zb(y1) + d)]
    v += [(x0, y0, ztop), (x1, y0, ztop), (x1, y1, ztop), (x0, y1, ztop)]
    n = len(rings)
    f.append((0, 3, 2, 1))
    for k in range(n):
        b = 4 * k
        for i in range(4):
            j = (i + 1) % 4
            f.append((b + i, b + j, b + 4 + j, b + 4 + i))
    f.append((4 * n, 4 * n + 1, 4 * n + 2, 4 * n + 3))
    o = mesh_object(name, v, f)
    recalc_normals(o)
    return o


def _leaf(p, r, front, a, Hh, W, ch, M, mats, title, lines, footer, border, doodle):
    """One framed leaf in its own frame (hinge line = X axis at the origin, outer face y = -T, hanging to -Z)."""
    sa, ca = sin(a), cos(a)
    wood, slate, endg = mats["wood"], mats["slate"], mats["endgrain"]
    pw = W - 2 * SW
    header = front and ch > 0.0
    zst = -RW if header else 0.0
    wood_objs, other = [], []

    def zbot(y):
        return (y * sa - Hh) / ca

    for sx in (-1, 1):
        x0, x1 = sorted((sx * W / 2, sx * (W / 2 - SW)))
        st = _wedge(f"stile{sx}", x0, x1, -T, 0.0, zst, zbot)
        densify(st, 0.2)
        bevel(st, 0.0032, segments=1)
        uv_grain(st, "Z")
        assign(st, wood)
        if p.frame != "painted":
            assign_faces(st, endg, lambda c, n: abs(n.z) > 0.9)
        wood_objs.append(st)

    if header:
        sh = 0.012

        def top_z(x):
            u = abs(x) / (W / 2)
            if p.top == "arched":
                return sh + (ch - sh) * (1.0 - u * u)
            return sh + (ch - sh) * (1.0 - smoothstep(0.2, 0.8, u))

        outer = [(-W / 2, -RW), (W / 2, -RW)] + [(W / 2 - W * i / 22, top_z(W / 2 - W * i / 22)) for i in range(23)]
        holes = []
        if p.top == "crest":
            zc = ch * 0.45
            holes.append([(x, y + zc) for x, y in rounded_rect(min(0.105, W * 0.2), 0.026, 0.0128, 4)])
        hd = shape_extrude("header", outer, holes, depth=T - 0.004, loc=(0, -0.002, 0), rot=(90, 0, 0), bevel=0.002)
        uv_grain(hd, "X")
        assign(hd, wood)
        wood_objs.append(hd)
    else:
        tr = box("top_rail", (pw, T, RW), loc=(0, -T / 2, -RW / 2), bevel=0.003, segments=1)
        densify(tr, 0.16)
        uv_grain(tr, "X")
        assign(tr, wood)
        wood_objs.append(tr)

    zb0 = -(Hh - min(p.leg, Hh - 0.42)) / ca
    zb1 = zb0 + BRW
    br = box("bottom_rail", (pw, T - r.u(0.0, 0.0015), BRW), loc=(0, -T / 2, (zb0 + zb1) / 2), bevel=0.003, segments=1)
    densify(br, 0.16)
    uv_grain(br, "X")
    assign(br, wood)
    wood_objs.append(br)

    # chalkboard panel housed in the frame, held by a planted bead
    ypf = -T + PANEL_SET
    pz0, pz1 = zb1 - 0.004, -RW + 0.004
    pn = box("panel", (pw + 0.008, 0.010, pz1 - pz0), loc=(0, ypf + 0.005, (pz0 + pz1) / 2))
    densify(pn, 0.22)
    assign(pn, slate)
    assign_faces(pn, wood, lambda c, n: n.y > 0.5)
    other.append(pn)
    bw = 0.011
    bx, bz0, bz1 = pw / 2 - bw / 2, zb1 + bw / 2, -RW - bw / 2
    bead_prof = [(-bw / 2, 0.0), (bw / 2, 0.0), (bw / 2, 0.003), (bw / 2 - 0.003, 0.0062), (-bw / 2 + 0.003, 0.0062), (-bw / 2, 0.003)]
    for sgn in (1, -1):       # one of the two has its moulded side out; the other is hidden in the panel
        bd = sweep(f"bead{sgn}", [(-bx, ypf, bz0), (bx, ypf, bz0), (bx, ypf, bz1), (-bx, ypf, bz1)],
                   [(q[0], q[1] * sgn) for q in (bead_prof if sgn > 0 else bead_prof[::-1])],
                   closed=True, smooth_path=False, up=(0, -1, 0))
        assign(bd, wood)
        wood_objs.append(bd)

    # oak pegs pinning the rail tenons
    peg = cylinder("peg", 0.0042, 0.003, verts=7, radius2=0.0034)
    pm = []
    ztr = (top_z(W / 2) - RW) * 0.5 if header else -RW / 2
    for sx in (-1, 1):
        for zc, dz in ((ztr, 0.014), ((zb0 + zb1) / 2, 0.018)):
            for k in (-1, 1):
                pm.append(trs((sx * (W / 2 - SW * 0.5 + r.jitter(0, 0.003)), -T, zc + k * dz + r.jitter(0, 0.002)), (90, 0, r.u(0, 90))))
    pegs = instances_to_mesh("pegs", peg, pm, keep_base=False)
    assign(pegs, endg)
    other.append(pegs)

    # chalk ledge
    if front and p.chalk_tray:
        tl = pw * 0.72
        tx = r.jitter(0.0, pw * 0.08)
        zl = zb1 - 0.02
        led = box("ledge", (tl, 0.036, 0.011), loc=(tx, -T - 0.018 + 0.0005, zl), bevel=0.002)
        lip = box("ledge_lip", (tl, 0.006, 0.019), loc=(tx, -T - 0.036 + 0.003, zl + 0.004), bevel=0.0015)
        for o in (led, lip):
            uv_grain(o, "X")
            assign(o, wood)
            wood_objs.append(o)
        for sx in (-1, 1):
            cb = extrude_profile("corbel", [(0.0, 0.0), (0.0, -0.03), (-0.008, -0.03), (-0.03, -0.004), (-0.03, 0.0)][::-1], 0.012,
                                 loc=(tx + sx * tl * 0.36 - 0.006, -T, zl - 0.0055), rot=(90, 0, 90))
            assign(cb, wood)
            wood_objs.append(cb)
        dx = tx + tl * 0.5 - 0.062 - r.u(0.0, 0.02)
        dz = r.jitter(0.0, 5.0)
        blk = box("duster_block", (0.085, 0.027, 0.016), loc=(dx, -T - 0.017, zl + 0.0055 + 0.018), rot=(0, 0, dz), bevel=0.003, segments=1)
        assign(blk, endg)
        uv_grain(blk, "X")
        felt = box("duster_felt", (0.081, 0.024, 0.011), loc=(dx, -T - 0.017, zl + 0.0055 + 0.0055), rot=(0, 0, dz), bevel=0.002, segments=1)
        assign(felt, mats["chalk"][0])
        other += [blk, felt]
        xk = tx - tl * 0.4
        for k in range(3):
            ln = r.u(0.03, 0.06)
            cs = cylinder(f"chalk_stick{k}", 0.0052, ln, loc=(xk + ln / 2, -T - 0.0062 - (0.011 if k == 1 else 0.0), zl + 0.0055 + 0.0052),
                          rot=(0, 90, r.jitter(0, 7.0)), verts=8, bevel=0.0008, segments=1)
            assign(cs, mats["chalk"][k % 2])
            smooth(cs, 50)
            other.append(cs)
            xk += ln + r.u(0.015, 0.035)

    # the menu itself
    ck = Chalk(ypf - 0.0007, r.fork("chalk"))
    _menu(ck, r.fork("menu"), title, lines, footer, pw / 2 - bw, -RW - bw, zb1 + bw, border, doodle)
    chalk_objs = ck.build("chalk", mats["chalk"])

    for o in wood_objs + other + chalk_objs:
        o.matrix_basis = M @ o.matrix_basis
    return wood_objs, other, chalk_objs, (zb0, zb1)


def _chain(name, path, link_len=0.03, wire=0.0022):
    P = [Vector(q) for q in resample_path(path, step=link_len * 0.74, smooth_path=False)]
    hl, hw = link_len / 2 - wire, link_len * 0.3 - wire
    base = sweep(name + "_link", [(hl * cos(2 * pi * i / 6), hw * 1.1 * sin(2 * pi * i / 6), 0.0) for i in range(6)], wire,
                 closed=True, sides=3, smooth_path=False)
    ms = []
    xa = Vector((1, 0, 0))
    for i, q in enumerate(P):
        t = (P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]).normalized()
        b = xa.cross(t).normalized()
        nn = xa
        if i % 2:
            nn, b = b, -nn
        ms.append(Matrix(((t.x, b.x, nn.x, q.x), (t.y, b.y, nn.y, q.y), (t.z, b.z, nn.z, q.z), (0, 0, 0, 1))))
    o = instances_to_mesh(name, base, ms, uv_jitter=False, keep_base=False)
    smooth(o, 180)
    return o


# ----------------------------------------------------------------------------- build
def build(p, rng):
    W, H = p.width, p.height
    ch = {"flat": 0.0, "arched": 0.07, "crest": 0.115}[p.top]
    rl = rng.fork("lean")
    a_f = radians(p.spread + rl.jitter(0.0, 0.7))
    a_b = radians(p.spread + rl.jitter(0.0, 0.7))
    Hh = H - T * sin(a_f) - ch * cos(a_f)                      # hinge line height

    if p.frame == "painted":
        wood = mat("wood_painted", p.paint, wear=0.9 + 0.9 * p.wear)
        endc = "#8a6a48"
    elif p.frame == "oak":
        wood = mat("wood_oak", p.paint)
        endc = p.paint
    else:
        wood = mat("wood_weathered", p.paint)
        endc = p.paint
    brass = mat("brass")
    iron = mat("iron_wrought")
    mats = {"wood": wood, "slate": mat("chalkboard", p.board_tint), "endgrain": mat("wood_endgrain", endc),
            "chalk": [mat("plaster_lime", p.chalk), mat("plaster_lime", p.accent)]}

    Mf = Matrix.Translation((0, 0, Hh)) @ Matrix.Rotation(-a_f, 4, "X")
    Mb = Matrix.Rotation(pi, 4, "Z") @ Matrix.Translation((0, 0, Hh)) @ Matrix.Rotation(-a_b, 4, "X")

    wf, of, cf, (zf0, zf1) = _leaf(p, rng.fork("front"), True, a_f, Hh, W, ch, Mf, mats, p.title, p.lines, p.footer, p.border, p.doodle)
    wb, ob, cb_, _ = _leaf(p, rng.fork("back"), False, a_b, Hh, W, 0.0, Mb, mats, p.back_title, p.back_lines or p.lines, "",
                           p.border, "sprig" if p.doodle != "none" else "none")

    # ---- brass strap hinges: knuckle in the apex valley, straps folded over the top edge and down the outer face
    # (the front strap runs up the back of the header board when there is one)
    hl, th, kr = 0.05, 0.002, 0.0052
    hx = W * (0.2 if ch > 0.0 else 0.29)
    xs = [-hx, hx] + ([0.0] if W > 0.7 and ch == 0.0 else [])
    scr = cylinder("screw", 0.0034, 0.0013, loc=(0, 0, 0.00065), verts=7, radius2=0.0022)
    sm = []
    brass_objs = []
    adf, adb = degrees(a_f), degrees(a_b)
    sl = 0.05
    strap = [(-hl / 2, 0.001), (-hl / 2, -sl * 0.5), (-hl * 0.26, -sl * 0.68), (-hl * 0.26, -sl * 0.8), (0.0, -sl),
             (hl * 0.26, -sl * 0.8), (hl * 0.26, -sl * 0.68), (hl / 2, -sl * 0.5), (hl / 2, 0.001)]
    strap_up = [(x, -z * 0.86 + 0.004) for x, z in strap][::-1]
    spts = [(-hl * 0.3, -sl * 0.2), (hl * 0.3, -sl * 0.2), (0.0, -sl * 0.72)]

    def leaf_strap(M, x, rz, ad):
        """Top flap + strap folded down the outer face of a leaf (leaf frame, then moved by M)."""
        fl = box("hinge_flap", (hl, T + th, th), loc=(x, -(T + th) / 2, th / 2), bevel=0.0006, segments=1)
        st = shape_extrude("hinge_strap", [(x + a, b) for a, b in strap], depth=th, loc=(0, -T, 0), rot=(90, 0, 0), bevel=0.0005)
        for o in (fl, st):
            o.matrix_basis = M @ o.matrix_basis
            brass_objs.append(o)
        for a, b in spts:
            sm.append(trs(M @ Vector((x + a, -T - th, b)), (90 - ad, 0, rz)))
        for k in (-1, 1):
            sm.append(trs(M @ Vector((x + k * hl * 0.3, -T * 0.5, th)), (-ad, 0, rz)))

    for x in xs:
        g = 0.0007
        prof = [(0.0, -hl / 2 - 0.003), (0.0032, -hl / 2 - 0.002), (0.0032, -hl / 2)]
        for k in range(3):
            z0 = -hl / 2 + k * hl / 3 + (g if k else 0.0)
            z1 = -hl / 2 + (k + 1) * hl / 3 - (g if k < 2 else 0.0)
            prof += [(kr, z0), (kr, z1)]
            if k < 2:
                prof += [(kr - 0.0014, z1), (kr - 0.0014, z1 + 2 * g)]
        prof += [(0.0032, hl / 2), (0.0032, hl / 2 + 0.002), (0.0, hl / 2 + 0.003)]
        brass_objs.append(lathe("knuckle", prof, segments=7, loc=(x, 0, Hh + kr * 0.6), rot=(0, 90, 0)))
        if ch > 0.0:
            st = shape_extrude("hinge_strap_up", [(x + a, b) for a, b in strap_up], depth=th, loc=(0, th, 0), rot=(90, 0, 0), bevel=0.0005)
            st.matrix_basis = Mf @ st.matrix_basis
            brass_objs.append(st)
            for a, b in spts:
                sm.append(trs(Mf @ Vector((x + a, th, -b * 0.86 + 0.004)), (-adf - 90, 0, 0)))
        else:
            leaf_strap(Mf, x, 0.0, adf)
        leaf_strap(Mb, -x, 180.0, adb)
    brass_objs.append(instances_to_mesh("hinge_screws", scr, sm, keep_base=False))

    # ---- chain stays on screw eyes, one each side
    rc = rng.fork("chain")
    for sx in (-1, 1):
        xe = sx * (W / 2 + 0.0085)
        zc = Hh * rc.u(0.44, 0.5)
        pf = Mf @ Vector((xe, -T / 2, -(Hh - zc) / cos(a_f)))
        pb = Mb @ Vector((-xe, -T / 2, -(Hh - zc) / cos(a_b)))
        for q in (pf, pb):
            brass_objs.append(torus("eye", 0.0052, 0.0015, loc=q, rot=(90, 0, 0), major_seg=8, minor_seg=4))
            brass_objs.append(cylinder("eye_shank", 0.0016, 0.009, loc=(q.x - sx * 0.008, q.y, q.z), rot=(0, 90, 0), verts=6))
        span = (pb - pf).length
        path = catenary(tuple(pf), tuple(pb), sag=span * rc.u(0.06, 0.13), n=20)
        brass_objs.append(_chain(f"stay{sx}", [tuple(q) for q in path]))
    for o in brass_objs:
        assign(o, brass)

    # ---- wrought crook with a candle jar hung in front of the top corner
    iron_objs = []
    if p.candle:
        xr = W / 2 + 0.0045
        yc = -T * 0.62
        lp = [(yc, -0.19), (yc, -0.1), (yc, -0.01), (yc - 0.006, 0.04), (-T / 2 - 0.04, 0.095), (-0.11, 0.118),
              (-0.18, 0.105), (-0.225, 0.065), (-0.235, 0.03), (-0.224, 0.01), (-0.206, 0.012), (-0.198, 0.028)]

        def xo(y):                                              # the arm leans outwards as it reaches forward
            return xr + 0.06 * clamp((-y - T) / 0.2)

        path = [tuple(Mf @ Vector((xo(y), y, z))) for y, z in lp]
        crook = sweep("crook", path, 0.0036, sides=5, scale_fn=lambda t: 1.0 - 0.35 * smoothstep(0.75, 1.0, t))
        iron_objs.append(crook)
        for z in (-0.17, -0.06):
            iron_objs.append(box("clip", (0.004, 0.02, 0.012), loc=Mf @ Vector((xr + 0.003, yc, z)), rot=(-adf, 0, 0), bevel=0.001, segments=1))
        q = Mf @ Vector((xo(-0.224), -0.224, 0.0145))          # hanging point (bottom of the hook curl)
        jr, jh = 0.033, 0.092
        jz = q.z - 0.062 - jh                                    # jar base
        jar = lathe("jar", [(0.0, 0.0), (jr - 0.004, 0.0), (jr, 0.006), (jr, 0.064), (jr - 0.005, 0.076), (jr - 0.005, 0.086),
                            (jr - 0.003, 0.088), (jr - 0.003, jh), (jr - 0.0065, jh), (jr - 0.0065, 0.078), (jr - 0.002, 0.064),
                            (jr - 0.002, 0.008), (0.0, 0.006)], segments=14, loc=(q.x, q.y, jz))
        assign(jar, mat("glass_lantern", glow=0.8, opacity=0.3))
        candle = cylinder("candle", 0.013, 0.03, loc=(q.x, q.y, jz + 0.006 + 0.015), verts=10, bevel=0.002, segments=1)
        assign(candle, mat("wax", "#efe2c4"))
        flame = lathe("flame", [(0.0, 0.0), (0.0042, 0.004), (0.0048, 0.009), (0.0022, 0.017), (0.0, 0.023)], segments=8,
                      loc=(q.x, q.y, jz + 0.039))
        assign(flame, mat("emissive", "#ffb45e", glow=9))
        neck = torus("jar_wire", jr - 0.0035, 0.0012, loc=(q.x, q.y, jz + 0.082), major_seg=14, minor_seg=4)
        bail = sweep("jar_bail", [(q.x, q.y - (jr - 0.003), jz + 0.082), (q.x, q.y - (jr + 0.004), jz + 0.105), (q.x, q.y - 0.014, q.z + 0.001),
                                  (q.x, q.y, q.z + 0.0045), (q.x, q.y + 0.014, q.z + 0.001), (q.x, q.y + (jr + 0.004), jz + 0.105),
                                  (q.x, q.y + (jr - 0.003), jz + 0.082)], 0.0012, sides=3)
        for o in (neck, bail):
            assign(o, iron)
        for o in iron_objs:
            assign(o, iron)
        part([jar, candle, flame, neck, bail], "lamp", pivot=tuple(q), motion="swing", amp=2.5, speed=0.45, axis="Y")
        lpos = (q.x, q.y, jz + 0.05)
        light("candle", lpos, color="#ffb866", intensity=p.lamp_intensity, range=3.0, flicker=0.45, shadow_near=0.06)

    # ---- weathering: scuffed, splashed feet; chipped corners; hand-worn top; faded upward faces
    wr = p.wear
    sd = int(p.seed)
    for o in wf + wb:
        paint(o, grime=mask_mul(height_ramp(0.0, 0.25 + 0.2 * wr), 0.12 + 0.3 * wr),
              wear=mask_mul(height_ramp(0.0, 0.1 + 0.1 * wr), noise_mask(0.03, seed=sd + 1, lo=0.05, hi=0.5), 0.8 + 0.2 * wr),
              special=mask_mul(facing((0, 0, 1)), 0.4 + 0.5 * wr))
        paint(o, wear=mask_mul(height_ramp(0.0, 0.035), 1.0))
        paint(o, wear=mask_mul(noise_mask(0.05, seed=sd + 3, lo=0.56, hi=0.76), 0.25 + 0.6 * wr))
        paint(o, wear=mask_mul(height_ramp(H + ch * 0.2, Hh - 0.07), noise_mask(0.07, seed=sd + 5, lo=0.3, hi=0.7), 0.25 + 0.4 * wr))
    for o in of + ob:
        paint(o, grime=mask_mul(height_ramp(0.0, 0.45), noise_mask(0.12, seed=sd + 7, lo=0.25, hi=0.75), 0.3 + 0.5 * wr))
    for o in brass_objs:
        paint(o, special=mask_mul(noise_mask(0.03, seed=sd + 9, lo=0.45, hi=0.8), 0.25 + 0.5 * wr), wear=0.25)
    for o in iron_objs:
        paint(o, special=mask_mul(noise_mask(0.04, seed=sd + 11, lo=0.4, hi=0.8), 0.2 + 0.4 * wr))

