"""Travel luggage stacked outside the 'Luggage' shop: a canvas-covered steamer trunk with oak slats held by
brass clamps, studded edge binding, brass corner bumpers, lock, draw latches and strap handles; leather
suitcases with a lid lip, stitched corner caps, buckled straps, handle and latches; a round hat box, a
striped carpet bag with a brass frame, an upright case leaning on the trunk and a small vanity case.
Every piece is built flat at the origin, joined per material and then set into the stack with a rigid
transform, so the three stack styles only differ in the layout code. Travel labels are thin paper decals.
"""
from math import sin, cos, pi, radians, tan

import bmesh
import numpy as np
from mathutils import Matrix, Vector

from tk import *

# tri_budget: up to seven separate hero pieces (trunk, three cases, hat box, bag, vanity) in one asset
META = {"name": "Trunks and suitcases", "class": "prop", "mount": "ground", "focus": 0.6, "tri_budget": 18000}
PARAMS = {
    "pieces": {"default": 6, "min": 2, "max": 7,
               "desc": "number of pieces: trunk, big case, second case, hat box, carpet bag, leaning case, vanity case"},
    "palette": {"default": "tan", "choices": ["tan", "oxblood", "green"], "desc": "leather / canvas colour scheme"},
    "style": {"default": "stack", "choices": ["stack", "tower", "display"],
              "desc": "stack = pyramid on the trunk, tower = one tall neat pile, display = trunk stood on end beside the cases"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "0 = shop-new, 1 = scuffed, faded, sloppily stacked"},
    "labels": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "density of hotel / steamer labels"},
    "monogram": {"default": "T.T.", "desc": "owner's initials stencilled on the trunk front ('' = none)"},
}
VARIANTS = {
    "oxblood_tower": {"style": "tower", "palette": "oxblood", "pieces": 6, "wear": 0.25, "labels": 0.35, "monogram": "K.H.", "seed": 4},
    "green_display": {"style": "display", "palette": "green", "pieces": 7, "wear": 0.6, "labels": 0.9, "monogram": "S.S.", "seed": 11},
    "worn_pair": {"style": "stack", "palette": "tan", "pieces": 3, "wear": 1.0, "labels": 1.0, "monogram": "", "seed": 23},
    "oxblood_full": {"style": "stack", "palette": "oxblood", "pieces": 7, "wear": 0.7, "labels": 0.7, "monogram": "A.W.", "seed": 31},
}

PALETTES = {
    "tan": dict(main="#96633a", alt="#c39a63", trim="#5e3c25", trunk="#a59b72", carpet="#6a2a26", carpet2="#94774a", wood="#a87f4f"),
    "oxblood": dict(main="#64241f", alt="#a06a3f", trim="#4a291e", trunk="#7a735c", carpet="#2c4a43", carpet2="#8f7a4a", wood="#9c7448"),
    "green": dict(main="#40644a", alt="#96633a", trim="#4c3322", trunk="#617a66", carpet="#7a3226", carpet2="#a58a58", wood="#a87f4f"),
}

NROT = {(0, 0, 1): (0, 0, 0), (0, 0, -1): (180, 0, 0), (0, -1, 0): (90, 0, 0), (0, 1, 0): (-90, 0, 0),
        (1, 0, 0): (0, 90, 0), (-1, 0, 0): (0, -90, 0)}


def _mx(loc=(0, 0, 0), rx=0.0, ry=0.0, rz=0.0):
    return (Matrix.Translation(Vector(loc)) @ Matrix.Rotation(radians(rz), 4, 'Z') @ Matrix.Rotation(radians(ry), 4, 'Y')
            @ Matrix.Rotation(radians(rx), 4, 'X'))


def _tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def cut_planes(o, xs=(), ys=(), zs=()):
    """Bisect a mesh at axis planes. The exporter bakes AO / grime per vertex, so the panel vertices are put
    on open canvas / leather, clear of the slats, straps and fittings that would otherwise shade them black."""
    inv = o.matrix_basis.inverted()
    m3t = o.matrix_basis.to_3x3().transposed()
    bm = bmesh.new()
    bm.from_mesh(o.data)
    for ax, vals in enumerate((xs, ys, zs)):
        for c in vals:
            co, no = Vector((0, 0, 0)), Vector((0, 0, 0))
            co[ax], no[ax] = c, 1.0
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5,
                                   plane_co=inv @ co, plane_no=m3t @ no)
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    return o


class Piece:
    """One piece of luggage: objects grouped per material key + rivet instances."""

    def __init__(self, name, mats):
        self.name, self.mats, self.groups, self.riv, self.nails = name, mats, {}, [], []
        self.top = 0.0

    def add(self, key, obj):
        assign(obj, self.mats[key])
        self.groups.setdefault(key, []).append(obj)
        return obj

    def rivet(self, loc, n, size=1.0):
        self.riv.append(trs(loc, NROT[tuple(int(round(c)) for c in n)], size))

    def nail(self, loc, n, size=1.0):
        self.nails.append(trs(loc, NROT[tuple(int(round(c)) for c in n)], size))

    def rivet_rot(self, loc, rot, size=1.0):
        self.riv.append(trs(loc, rot, size))

    def finish(self, M, rivet_base, nail_base):
        if self.riv:
            self.add("brass", instances_to_mesh(self.name + "_rivets", rivet_base, self.riv, keep_base=True))
        if self.nails:
            self.add("brass", instances_to_mesh(self.name + "_nails", nail_base, self.nails, keep_base=True))
        out = {}
        for key, objs in self.groups.items():
            o = join(objs, f"{self.name}_{key}")
            o.data.transform(M)
            o.data.update()
            out[key] = o
        return out


def rivet_mesh():
    r, h, n = 0.0042, 0.0022, 5
    V = [(r * cos(2 * pi * k / n), r * sin(2 * pi * k / n), 0.0) for k in range(n)]
    V += [(0.55 * r * cos(2 * pi * k / n), 0.55 * r * sin(2 * pi * k / n), h) for k in range(n)]
    F = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)] + [tuple(range(n, 2 * n))]
    o = mesh_object("rivet", V, F)
    smooth(o, 80)
    return o


def nail_mesh():
    """Upholstery nail head: a low four-sided dome (4 tris, smooth shaded)."""
    r, h = 0.0052, 0.0026
    o = mesh_object("nail", [(r, 0, 0), (0, r, 0), (-r, 0, 0), (0, -r, 0), (0, 0, h)], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    smooth(o, 80)
    return o


def corner_cap(name, c, sg, R, L, sink):
    """Pressed corner protector: a sphere octant of radius R centred on c with three lobes of length L
    running back along the faces; the lobe ends sink into the surface so there is no open rim."""
    k = 0.70710678
    q = 0.57735027
    Rs = R - sink
    base = [(R, 0, 0), (0, R, 0), (0, 0, R), (R * k, R * k, 0), (0, R * k, R * k), (R * k, 0, R * k),
            (Rs, 0, -L), (Rs * k, Rs * k, -L), (0, Rs, -L),
            (-L, Rs, 0), (-L, Rs * k, Rs * k), (-L, 0, Rs),
            (0, -L, Rs), (Rs * k, -L, Rs * k), (Rs, -L, 0),
            (Rs, -L, -L), (-L, Rs, -L), (-L, -L, Rs), (R * q, R * q, R * q)]
    F = [(18, 0, 3), (18, 3, 1), (18, 1, 4), (18, 4, 2), (18, 2, 5), (18, 5, 0),
         (0, 6, 7, 3), (3, 7, 8, 1), (1, 9, 10, 4), (4, 10, 11, 2), (2, 12, 13, 5), (5, 13, 14, 0),
         (0, 14, 15, 6), (1, 8, 16, 9), (2, 11, 17, 12)]
    V = [(c[0] + sg[0] * x, c[1] + sg[1] * y, c[2] + sg[2] * z) for x, y, z in base]
    if sg[0] * sg[1] * sg[2] < 0:
        F = [tuple(reversed(f)) for f in F]
    o = mesh_object(name, V, F)
    smooth(o, 55)
    return o


def lstrip(name, e0, e1, n1, n2, w1, w2, t):
    """L-section strip folded over the box edge e0->e1 (n1, n2 = outward normals of the two faces):
    leg 1 lies on face n1 (width w1), leg 2 on face n2 (width w2), thickness t, eased outer arris."""
    e0, e1, n1, n2 = Vector(e0), Vector(e1), Vector(n1), Vector(n2)
    sec = [(-0.001, -w1), (t, -w1), (t, 0.3 * t), (0.3 * t, t), (-w2, t), (-w2, -0.001), (0.0, 0.0)]
    V = [tuple(e + n1 * a + n2 * b) for e in (e0, e1) for a, b in sec]
    m = len(sec)
    F = [(i, i + 1, m + i + 1, m + i) for i in range(5)]
    F += [(1, 2, 6, 0), (2, 3, 6), (3, 4, 5, 6)]
    F += [tuple(reversed((m + 1, m + 2, m + 6, m + 0))), tuple(reversed((m + 2, m + 3, m + 6))), tuple(reversed((m + 3, m + 4, m + 5, m + 6)))]
    # derived for d parallel to n1 x n2 with end caps facing -d at e0; flip everything otherwise
    if n1.cross(n2).dot(e1 - e0) > 0:
        F = [tuple(reversed(f)) for f in F]
    o = mesh_object(name, V, F)
    smooth(o, 50)
    return o


def decal(name, pts, org, u, v, n, off):
    """Paper label: one flat polygon a hair above the surface (org + u/v frame, normal n)."""
    org, u, v, n = Vector(org), Vector(u), Vector(v), Vector(n)
    V = [tuple(org + u * a + v * b + n * off) for a, b in pts]
    f = tuple(range(len(V)))
    if u.cross(v).dot(n) < 0:
        f = tuple(reversed(f))
    return mesh_object(name, V, [f])


def cyl_decal(name, R, a0, zc, w, h, off, nseg=4):
    V, F = [], []
    for i in range(nseg + 1):
        a = a0 + (i / nseg - 0.5) * w / R
        x, y = (R + off) * sin(a), -(R + off) * cos(a)
        V += [(x, y, zc - h / 2), (x, y, zc + h / 2)]
    for i in range(nseg):
        F.append((2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1))
    return mesh_object(name, V, F)


def ngon(n, rx, ry=None, a0=0.0):
    ry = rx if ry is None else ry
    return [(rx * cos(a0 + 2 * pi * k / n), ry * sin(a0 + 2 * pi * k / n)) for k in range(n)]


def rect(w, h, cx=0.0, cy=0.0):
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def label_layers(kind, col):
    if kind == "hotel":
        return [("cream", rect(1.0, 0.64)), (col, rect(0.86, 0.2, 0, 0.12)), (col, rect(0.86, 0.05, 0, -0.17))]
    if kind == "round":
        return [(col, ngon(14, 0.5)), ("cream", ngon(12, 0.37)), (col, ngon(8, 0.15))]
    if kind == "diamond":
        return [(col, ngon(4, 0.58)), ("cream", rect(0.56, 0.17))]
    if kind == "oval":
        return [("cream", ngon(14, 0.56, 0.37)), (col, ngon(12, 0.44, 0.26)), ("cream", rect(0.5, 0.08))]
    if kind == "pennant":
        return [(col, [(-0.55, -0.3), (0.6, 0.0), (-0.55, 0.3)]), ("cream", rect(0.1, 0.5, -0.36, 0))]
    return [("cream", rect(0.56, 0.9)), (col, rect(0.42, 0.34, 0, 0.2)), (col, rect(0.42, 0.05, 0, -0.2))]


class Labeler:
    KINDS = ["hotel", "round", "diamond", "oval", "pennant", "ticket", "hotel", "round"]

    def __init__(self, rng, density):
        self.rng, self.density = rng, density

    def stick(self, pc, org, u, v, n, s, ang):
        r = self.rng
        u, v = Vector(u), Vector(v)
        ca, sa = cos(ang), sin(ang)
        uu, vv = u * ca + v * sa, v * ca - u * sa
        col = r.pick(["red", "blue"])
        for i, (key, pts) in enumerate(label_layers(r.pick(self.KINDS), col)):
            pc.add(key, decal("label", [(a * s, b * s) for a, b in pts], org, uu, vv, n, 0.0008 + 0.0006 * i))

    def scatter(self, pc, regions, count):
        """regions: [(org, u, v, n, hu, hv)]; count scales with the label density."""
        r = self.rng
        k = int(round(count * self.density))
        placed = []
        for _ in range(k * 6):
            if len(placed) >= k:
                break
            ri = int(r.u(0, len(regions) - 1e-6))
            org, u, v, n, hu, hv = regions[ri]
            s = r.u(0.07, 0.105)
            if hu < s * 0.55 or hv < s * 0.55:
                s = min(hu, hv) / 0.55
                if s < 0.05:
                    continue
            a, b = r.u(-(hu - s * 0.5), hu - s * 0.5), r.u(-(hv - s * 0.5), hv - s * 0.5)
            if any(q[0] == ri and (q[1] - a) ** 2 + (q[2] - b) ** 2 < (0.6 * (s + q[3])) ** 2 for q in placed):
                continue
            placed.append((ri, a, b, s))
            self.stick(pc, Vector(org) + Vector(u) * a + Vector(v) * b, u, v, n, s, radians(r.u(-28, 28)))


def strap_loop(name, x, zc, sy, sz, rad, s, width):
    """Closed leather strap around a body cross-section (in the YZ plane at x)."""
    loop = rounded_rect(sy, sz, rad, 3)
    path = [(x, a, zc + b) for a, b in loop]
    hw = width / 2
    return sweep(name, path, [(-s / 2, -hw), (s / 2, -hw), (s / 2, hw), (-s / 2, hw)], closed=True, smooth_path=False, up=(1, 0, 0))


def buckle(pc, strap_key, x, y, z, s, width, rng):
    """Brass frame buckle on the -Y face at (x, y = strap surface, z) with prong, keeper and strap tail."""
    fw, fh = width + 0.016, 0.034
    fr = shape_extrude("buckle", rounded_rect(fw, fh, 0.006, 2), [rect(fw - 0.011, fh - 0.012)], depth=0.0032,
                       loc=(x, y - 0.0006, z), rot=(90, 0, 0))
    pc.add("brass", fr)
    pc.add("brass", box("prong", (0.0036, 0.003, fh - 0.004), loc=(x, y - 0.0042, z)))
    tilt = rng.jitter(0.0, 4.0)
    pc.add(strap_key, box("strap_tail", (width - 0.003, s, 0.062), loc=(x + 0.0005 * tilt, y - s * 0.5 - 0.0008, z + 0.036),
                          rot=(0, tilt, 0), bevel=0.001, segments=1))
    pc.add(strap_key, box("keeper", (width + 0.008, 2 * s + 0.003, 0.009), loc=(x, y - s * 0.5 + 0.0004, z + 0.047), bevel=0.001, segments=1))


def grip(name, x0, y0, z0, half, out, droop):
    """Flat leather handle arching out of the -Y face between x0 +- half."""
    n = 9
    pts = []
    for i in range(n):
        t = i / (n - 1)
        s_ = sin(pi * t) ** 0.75
        pts.append((x0 - half + 2 * half * t, y0 - 0.007 - out * s_, z0 - droop * s_))
    prof = [(0.0055, -0.008), (0.0055, 0.008), (0.0, 0.012), (-0.0055, 0.008), (-0.0055, -0.008), (0.0, -0.012)]
    return sweep(name, pts, prof, smooth_path=False, up=(0, 0, 1))


# ----------------------------------------------------------------------------------------- trunk
def build_trunk(pc, W, D, H, rng, monogram, lab, flat_handle=False):
    ts = 0.012                      # slat thickness
    tcl = ts + 0.003                # brass slat clamp thickness (the trunk stands on the bottom clamps)
    Wb, Db = W - 2 * ts, D - 2 * ts
    hx, hy = Wb / 2, Db / 2
    zb0, zb1 = tcl, H - tcl
    Hb = zb1 - zb0
    zs = zb0 + Hb * 0.67            # lid seam
    rb, tb, wb, sw = 0.008, 0.0035, 0.032, 0.042

    body = pc.add("canvas", box("trunk_body", (Wb, Db, Hb), loc=(0, 0, (zb0 + zb1) / 2), bevel=rb, segments=2))
    smooth(body, 40)
    # lid and base valance bands either side of the seam (dark shadow line between them)
    for nm, za, zb_ in (("val_base", zs - 0.027, zs - 0.002), ("val_lid", zs + 0.002, zs + 0.025)):
        hh = (zb_ - za) / 2
        path = [(x, y, (za + zb_) / 2) for x, y in rounded_rect(Wb, Db, rb, 2)]
        pc.add("trim", sweep(nm, path, [(-0.003, -hh), (0.003, -hh), (0.003, hh), (-0.003, hh)], closed=True, smooth_path=False, up=(0, 0, 1)))

    # --- edge binding (leather over the arrises) with rows of brass nails
    def nails(p0, p1, n, skip=None, pitch=0.095):
        p0, p1 = Vector(p0), Vector(p1)
        L = (p1 - p0).length
        k = max(2, int(round(L / pitch)) + 1)
        for i in range(k):
            q = p0.lerp(p1, i / (k - 1))
            if skip and skip(q):
                continue
            pc.nail(tuple(q), n)

    xi, xo = W * 0.165, hx - 0.078
    xs = [-xo, -xi, xi, xo]
    near_slat = lambda q: any(abs(q.x - x) < sw / 2 + 0.014 for x in xs)
    near_seam = lambda q: abs(q.z - zs) < 0.034
    fl_ = sw / 2 + 0.02             # panel vertices sit in the bays, beside (never under) slats, binding, lock
    cut_planes(body, xs=[sg * v for sg in (-1, 1) for v in (xo - fl_, xi + fl_, xi - fl_)],
               ys=[-(hy - wb - 0.015), 0.0, hy - wb - 0.015],
               zs=[zb0 + wb + 0.012, zb0 + Hb * 0.25, zs - 0.05, zs + 0.045, zb1 - wb - 0.012])
    cl = 0.036                      # binding stops under the corner caps
    for sy in (-1, 1):
        for zz, nz in ((zb1, 1), (zb0, -1)):
            pc.add("trim", lstrip("bind_x", (-hx + cl, sy * hy, zz), (hx - cl, sy * hy, zz), (0, 0, nz), (0, sy, 0), wb, wb, tb))
        nails((-hx + 0.06, sy * (hy - 0.017), zb1 + tb), (hx - 0.06, sy * (hy - 0.017), zb1 + tb), (0, 0, 1), near_slat)
        nails((-hx + 0.06, sy * (hy + tb), zb1 - 0.017), (hx - 0.06, sy * (hy + tb), zb1 - 0.017), (0, sy, 0), near_slat)
        nails((-hx + 0.06, sy * (hy + tb), zb0 + 0.017), (hx - 0.06, sy * (hy + tb), zb0 + 0.017), (0, sy, 0), near_slat)
    for sx in (-1, 1):
        for zz, nz in ((zb1, 1), (zb0, -1)):
            pc.add("trim", lstrip("bind_y", (sx * hx, -hy + cl, zz), (sx * hx, hy - cl, zz), (0, 0, nz), (sx, 0, 0), wb, wb, tb))
        nails((sx * (hx - 0.017), -hy + 0.06, zb1 + tb), (sx * (hx - 0.017), hy - 0.06, zb1 + tb), (0, 0, 1))
        nails((sx * (hx + tb), -hy + 0.06, zb1 - 0.017), (sx * (hx + tb), hy - 0.06, zb1 - 0.017), (sx, 0, 0))
        nails((sx * (hx + tb), -hy + 0.06, zb0 + 0.017), (sx * (hx + tb), hy - 0.06, zb0 + 0.017), (sx, 0, 0))
        for sy in (-1, 1):
            for za, zb_ in ((zb0 + cl, zs - 0.003), (zs + 0.003, zb1 - cl)):
                pc.add("trim", lstrip("bind_z", (sx * hx, sy * hy, za), (sx * hx, sy * hy, zb_), (sx, 0, 0), (0, sy, 0), wb, wb, tb))
            nails((sx * (hx - 0.017), sy * (hy + tb), zb0 + 0.07), (sx * (hx - 0.017), sy * (hy + tb), zb1 - 0.07), (0, sy, 0), near_seam)
            nails((sx * (hx + tb), sy * (hy - 0.017), zb0 + 0.07), (sx * (hx + tb), sy * (hy - 0.017), zb1 - 0.07), (sx, 0, 0), near_seam)
            for sz, zz in ((-1, zb0), (1, zb1)):
                pc.add("brass", corner_cap("trunk_corner", (sx * (hx - rb), sy * (hy - rb), zz - sz * rb), (sx, sy, sz), rb + 0.0055, 0.05, 0.0065))
                c = (sx * hx, sy * hy, zz)
                if sz > 0:
                    pc.rivet((c[0] - sx * 0.03, c[1] - sy * 0.03, zz + 0.0045), (0, 0, 1), 1.2)
                pc.rivet((c[0] - sx * 0.03, c[1] + sy * 0.0045, zz - sz * 0.03), (0, sy, 0), 1.2)
                pc.rivet((c[0] + sx * 0.0045, c[1] - sy * 0.03, zz - sz * 0.03), (sx, 0, 0), 1.2)

    # --- oak slats: across the lid, down the front and back, under the bottom; brass clamps at every end
    r = rng.fork("slats")
    for i, x in enumerate(xs):
        o = pc.add("wood", box("slat_top", (sw, Db - 0.004, ts), loc=(x, 0, zb1 + ts / 2), bevel=0.003, segments=1))
        uv_grain(o, axis='Y')
        o = pc.add("wood", box("slat_bot", (sw, Db - 0.004, ts), loc=(x, 0, zb0 - ts / 2)))
        uv_grain(o, axis='Y')
        for sy in (-1, 1):
            for za, zb_ in ((zb0, zs - 0.004), (zs + 0.004, zb1)):
                o = pc.add("wood", box("slat_v", (sw, ts, zb_ - za), loc=(x, sy * (hy + ts / 2), (za + zb_) / 2), bevel=0.003, segments=1))
                uv_grain(o, axis='Z')
            hw = sw / 2 + 0.0035
            for zz, nz in ((zb1, 1), (zb0, -1)):
                pc.add("brass", lstrip("clamp", (x - hw, sy * hy, zz), (x + hw, sy * hy, zz), (0, 0, nz), (0, sy, 0), 0.042, 0.042, tcl))
                pc.rivet((x, sy * (hy + tcl), zz - nz * 0.027), (0, sy, 0), 1.25)
                if nz > 0:
                    pc.rivet((x, sy * (hy - 0.027), zz + tcl), (0, 0, 1), 1.25)
            for sgn in (-1, 1):       # slat shoes either side of the seam
                bv = 0.0015 if sy < 0 else 0.0
                pc.add("brass", box("shoe", (sw + 0.007, tcl, 0.022), loc=(x, sy * (hy + tcl / 2), zs + sgn * 0.0145), bevel=bv, segments=1))
                pc.nail((x, sy * (hy + tcl), zs + sgn * 0.0145), (0, sy, 0), 1.0)
    zend = [zb0 + Hb * 0.14, zb0 + Hb * 0.5, (zs + zb1) / 2 + 0.004]
    for sx in (-1, 1):
        for z in zend:
            o = pc.add("wood", box("slat_end", (ts, Db - 2 * wb - 0.01, sw), loc=(sx * (hx + ts / 2), 0, z), bevel=0.003, segments=1))
            uv_grain(o, axis='Y')
            for sy in (-1, 1):
                pc.nail((sx * (hx + ts), sy * (hy - wb - 0.03), z), (sx, 0, 0), 1.0)

    # --- centre lock: plate on the base, hasp from the lid, key barrel
    yf = -hy - 0.003
    pc.add("brass", shape_extrude("lock_plate", rounded_rect(0.092, 0.078, 0.012, 2), [], depth=0.004, loc=(0, yf, zs - 0.041), rot=(90, 0, 0)))
    pc.add("brass", box("lock_hasp", (0.044, 0.005, 0.096), loc=(0, yf - 0.0062, zs + 0.01), bevel=0.0015, segments=1))
    pc.add("brass", cylinder("lock_boss", 0.016, 0.008, loc=(0, yf - 0.0105, zs - 0.018), rot=(90, 0, 0), verts=12))
    pc.add("trim", cylinder("lock_key", 0.005, 0.004, loc=(0, yf - 0.014, zs - 0.018), rot=(90, 0, 0), verts=6))
    for sx in (-1, 1):
        pc.rivet((sx * 0.035, yf - 0.004, zs - 0.068), (0, -1, 0))
        pc.rivet((sx * 0.035, yf - 0.004, zs - 0.014), (0, -1, 0))
        pc.rivet((sx * 0.012, yf - 0.0087, zs + 0.046), (0, -1, 0))
    # --- draw latches between the slats
    xl = (xi + xo) / 2
    for sx in (-1, 1):
        x = sx * xl
        pc.add("brass", box("latch_catch", (0.042, 0.005, 0.03), loc=(x, yf - 0.0025, zs - 0.019), bevel=0.0015, segments=1))
        pc.add("brass", box("latch_plate", (0.042, 0.005, 0.04), loc=(x, yf - 0.0025, zs + 0.027), bevel=0.0015, segments=1))
        pc.add("brass", box("latch_lever", (0.022, 0.006, 0.058), loc=(x, yf - 0.0085, zs + 0.006), rot=(-5, 0, 0), bevel=0.002, segments=1))
        pc.add("brass", cylinder("latch_pin", 0.004, 0.034, loc=(x, yf - 0.008, zs + 0.036), rot=(0, 90, 0), verts=6))
        for dx in (-0.014, 0.014):
            pc.rivet((x + dx, yf - 0.005, zs - 0.026), (0, -1, 0), 0.8)
    # --- hinges on the back
    yb = hy + 0.003
    for x in (-xl, 0.0, xl):
        pc.add("brass", box("hinge", (0.046, 0.004, 0.074), loc=(x, yb + 0.002, zs), bevel=0.0015, segments=1))
        pc.add("brass", cylinder("hinge_pin", 0.006, 0.052, loc=(x, yb + 0.0055, zs), rot=(0, 90, 0), verts=8))
        for dx, dz in ((-0.014, -0.027), (0.014, -0.027), (-0.014, 0.027), (0.014, 0.027)):
            pc.nail((x + dx, yb + 0.004, zs + dz), (0, 1, 0), 0.7)

    # --- strap handles on the ends
    zh = zb0 + Hb * 0.335
    prof = [(-0.016, 0.0), (-0.01, -0.0055), (0.01, -0.0055), (0.016, 0.0), (0.01, 0.0055), (-0.01, 0.0055)]
    for sx in (-1, 1):
        fl = flat_handle and sx < 0
        out, droop = (0.001, 0.0) if fl else (0.02, 0.016)
        pts = []
        for i in range(9):
            t = i / 8
            s_ = sin(pi * t) ** 0.8
            pts.append((sx * (hx + 0.006 + out * s_), -0.092 + 0.184 * t, zh - droop * s_))
        pc.add("trim", sweep("trunk_handle", pts, prof, smooth_path=False, up=(sx, 0, 0)))
        for sy in (-1, 1):
            capt = 0.009 if fl else 0.014
            pc.add("brass", box("handle_cap", (capt, 0.036, 0.044), loc=(sx * (hx + capt / 2), sy * 0.096, zh), bevel=0.002, segments=1))
            for dz in (-0.013, 0.013):
                pc.rivet((sx * (hx + capt), sy * 0.102, zh + dz), (sx, 0, 0), 0.9)

    # --- stencilled initials
    s = (monogram or "").strip()
    if s:
        size = min(0.062, 0.2 / max(1, len(s)) / 0.62)
        t = text("monogram", s, size=size, depth=0.0011, loc=(0, -hy - 0.0002, zb0 + Hb * 0.27), font="sans")
        pc.add("cream", t)

    # --- travel labels between the slats
    bay_c, bay_o = xi - sw / 2 - 0.012, (xo - xi) / 2 - sw / 2 - 0.008
    regs = [((0, 0, zb1), (1, 0, 0), (0, 1, 0), (0, 0, 1), bay_c, hy - 0.06)]
    for sx in (-1, 1):
        regs.append(((sx * xl, 0, zb1), (1, 0, 0), (0, 1, 0), (0, 0, 1), bay_o, hy - 0.06))
        zc = (zb0 + wb + zs - 0.06) / 2
        regs.append(((sx * xl, -hy, zc), (1, 0, 0), (0, 0, 1), (0, -1, 0), bay_o, (zs - 0.06 - zb0 - wb) / 2))
        regs.append(((sx * xl, hy, zc + 0.02), (-1, 0, 0), (0, 0, 1), (0, 1, 0), bay_o, (zs - 0.06 - zb0 - wb) / 2))
    lab.scatter(pc, regs, 11)
    pc.top = H
    pc.dims = (W, D, H)


# ----------------------------------------------------------------------------------------- suitcase
def build_case(pc, W, D, T, rng, lab, main="main", corner="trim", straps=True, n_labels=5, lid_labels=True, tag=False):
    s, gap, rb = 0.0035, 0.002, 0.016
    z0 = s + gap
    zc = z0 + T / 2
    body = pc.add(main, box("case_body", (W, D, T), loc=(0, 0, zc), bevel=rb, segments=2))
    smooth(body, 40)
    # stitched piping round the lid and the bottom panel
    e = rb * 0.2929 - 0.0012
    for zp in (z0 + T - e, z0 + e):
        path = [(x, y, zp) for x, y in rounded_rect(W - 2 * e, D - 2 * e, rb * 0.72, 2)]
        pc.add("trim", sweep("case_piping", path, 0.0026, sides=3, closed=True, smooth_path=False))
    zs = z0 + T * 0.6
    # lid lip overlapping the base
    path = [(x, y, zs + 0.008) for x, y in rounded_rect(W, D, rb, 3)]
    pc.add(main, sweep("case_lip", path, [(-0.0022, -0.008), (0.0022, -0.008), (0.0022, 0.008), (-0.0022, 0.008)], closed=True,
                       smooth_path=False, up=(0, 0, 1)))
    yf = -D / 2
    hw = 0.068
    # straps with buckles on the handle side
    sxp = W * 0.28
    if straps:
        cx = [sg * (sxp + d) for sg in (-1, 1) for d in (-0.04, 0.04)]
    else:
        cx = [-(hw + 0.03), hw + 0.03] if W * 0.31 - 0.03 > hw + 0.03 else [0.0]
    cut_planes(body, xs=cx, ys=[-D * 0.17, D * 0.17], zs=[z0 + T * 0.3, z0 + T * 0.8 + 0.008])
    if straps:
        off = gap + s / 2
        for sx in (-1, 1):
            pc.add("trim", strap_loop("case_strap", sx * sxp, zc, D + 2 * off, T + 2 * off, rb + off, s, 0.03))
            buckle(pc, "trim", sx * sxp, yf - gap - s, z0 + T * 0.3, s, 0.03, rng)
            # leather keeper loops stitched to the lid
            for y in (-D * 0.2, D * 0.24):
                pc.add(main, box("strap_loop", (0.044, 0.012, 0.004), loc=(sx * sxp, y, z0 + T + gap + s + 0.0005), bevel=0.001, segments=1))
    # handle with brass mounts
    zh = zc - 0.004
    pc.add("trim", grip("case_grip", 0.0, yf, zh, hw, 0.03, 0.02))
    for sx in (-1, 1):
        pc.add("brass", box("grip_plate", (0.026, 0.003, 0.038), loc=(sx * hw, yf - 0.0015, zh), bevel=0.001, segments=1))
        pc.add("brass", box("grip_bridge", (0.012, 0.013, 0.03), loc=(sx * hw, yf - 0.0065, zh), bevel=0.002, segments=1))
        pc.rivet((sx * hw, yf - 0.003, zh + 0.0155), (0, -1, 0), 0.7)
        pc.rivet((sx * hw, yf - 0.003, zh - 0.0155), (0, -1, 0), 0.7)
    # spring latches across the lip
    lx = (hw + sxp) / 2 + 0.004 if straps else W * 0.31
    for sx in (-1, 1):
        pc.add("brass", box("latch", (0.03, 0.004, 0.05), loc=(sx * lx, yf - 0.0035, zs + 0.002), bevel=0.0012, segments=1))
        pc.add("brass", box("latch_btn", (0.013, 0.004, 0.02), loc=(sx * lx, yf - 0.007, zs - 0.006), bevel=0.0012, segments=1))
        pc.rivet((sx * lx, yf - 0.0055, zs + 0.02), (0, -1, 0), 0.7)
    if not straps:
        pc.add("brass", box("lockplate", (0.03, 0.004, 0.034), loc=(0, yf - 0.0035, zs + 0.012), bevel=0.0012, segments=1))
        pc.add("trim", cylinder("keyhole", 0.0035, 0.002, loc=(0, yf - 0.0062, zs + 0.012), rot=(90, 0, 0), verts=6))
    # corner caps
    R = rb + 0.0026
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                c = (sx * (W / 2 - rb), sy * (D / 2 - rb), zc + sz * (T / 2 - rb))
                pc.add(corner, corner_cap("case_corner", c, (sx, sy, sz), R, 0.034, 0.0036))
                if corner == "brass":
                    pc.nail((sx * (W / 2 - 0.03), sy * (D / 2 + 0.0012), zc + sz * (T / 2 - 0.028)), (0, sy, 0), 0.7)
                    pc.nail((sx * (W / 2 + 0.0012), sy * (D / 2 - 0.03), zc + sz * (T / 2 - 0.028)), (sx, 0, 0), 0.7)
                    if sz > 0:
                        pc.nail((sx * (W / 2 - 0.03), sy * (D / 2 - 0.03), zc + T / 2 + 0.0012), (0, 0, 1), 0.7)
    # hinge side: brass feet (the case stands on them when carried upright) and two hinges
    for sx in (-1, 1):
        for fz in (0.22, 0.84):
            pc.add("brass", cylinder("foot", 0.009, 0.008, loc=(sx * W * 0.4, D / 2 + 0.003, z0 + T * fz), rot=(90, 0, 0), verts=8, radius2=0.0065))
        pc.add("brass", box("case_hinge", (0.044, 0.003, 0.03), loc=(sx * W * 0.17, D / 2 + 0.0035, zs + 0.004), bevel=0.001, segments=1))
    # labels on the lid
    regs = [((0, 0, z0 + T), (1, 0, 0), (0, 1, 0), (0, 0, 1), W / 2 - 0.045, D / 2 - 0.045)] if lid_labels else []
    zlo, zhi = z0 + 0.024, zs - 0.004
    for sx in (-1, 1):
        regs.append(((sx * W / 2, 0, (zlo + zhi) / 2), (0, sx, 0), (0, 0, 1), (sx, 0, 0), D / 2 - 0.05, (zhi - zlo) / 2))
    lab.scatter(pc, regs, n_labels)
    # a paper luggage tag tied to the handle
    if tag:
        tz = rng.jitter(0.0, 14.0)
        pc.add("cream", box("tag", (0.036, 0.0014, 0.064), loc=(hw + 0.026, yf - 0.0045, zh - 0.036), rot=(4, tz, 0), bevel=0.0005, segments=1))
        pc.add("trim", sweep("tag_string", [(hw + 0.004, yf - 0.012, zh + 0.004), (hw + 0.018, yf - 0.008, zh - 0.004), (hw + 0.026, yf - 0.0052, zh - 0.012)],
                             0.0011, sides=3, smooth_path=False))
    pc.top = z0 + T + (gap + s if straps else 0.0)
    pc.dims = (W, D, pc.top)


# ----------------------------------------------------------------------------------------- hat box
def build_hatbox(pc, R, Hh, rng, lab, main="alt"):
    s, gap = 0.0032, 0.0015
    z0 = s + gap
    zl = z0 + Hh * 0.7              # lower edge of the lid skirt
    zt = z0 + Hh
    prof = [(R - 0.014, z0), (R - 0.003, z0 + 0.001), (R, z0 + 0.006), (R, zl), (R + 0.0045, zl), (R + 0.0045, zt - 0.007),
            (R + 0.002, zt - 0.0015), (R - 0.006, zt), (0.0, zt)]
    pc.add(main, lathe("hat_body", prof, segments=26))
    pc.add("trim", lathe("hat_base_band", [(R - 0.002, z0 + 0.003), (R + 0.0022, z0 + 0.006), (R + 0.0022, z0 + 0.03), (R - 0.001, z0 + 0.032)], segments=26))
    pc.add("trim", lathe("hat_lid_band", [(R + 0.003, zl - 0.0005), (R + 0.0066, zl + 0.001), (R + 0.0066, zl + 0.013), (R + 0.003, zl + 0.0145)], segments=26))
    off = 0.0068 + gap + s / 2
    pc.add("trim", strap_loop("hat_strap", 0.0, z0 + Hh / 2, 2 * (R + off), Hh + 2 * (gap + s / 2), 0.014, s, 0.026))
    buckle(pc, "trim", 0.0, -(R + off + s / 2), z0 + Hh * 0.34, s, 0.026, rng)
    # carrying handle on the lid, over the strap
    zh = zt + gap + s
    n = 9
    pts = [(-0.062 + 0.124 * i / (n - 1), 0.0, zh + 0.004 + 0.032 * sin(pi * i / (n - 1)) ** 0.7) for i in range(n)]
    prof_h = [(-0.011, 0.0), (-0.007, -0.0045), (0.007, -0.0045), (0.011, 0.0), (0.007, 0.0045), (-0.007, 0.0045)]
    pc.add("trim", sweep("hat_handle", pts, prof_h, smooth_path=False, up=(0, 0, 1)))
    for sx in (-1, 1):
        pc.add("trim", box("hat_tab", (0.04, 0.03, 0.0045), loc=(sx * 0.07, 0, zt + 0.002), bevel=0.0012, segments=1))
        pc.rivet((sx * 0.078, 0.0, zt + 0.0042), (0, 0, 1), 0.9)
        pc.rivet((sx * 0.06, 0.008, zt + 0.0042), (0, 0, 1), 0.9)
        pc.rivet((sx * 0.06, -0.008, zt + 0.0042), (0, 0, 1), 0.9)
    # labels: lid top (clear of strap and handle) and one wrapped round the side
    regs = [((sx * R * 0.5, sy * R * 0.5, zt), (1, 0, 0), (0, 1, 0), (0, 0, 1), R * 0.3, R * 0.3) for sx in (-1, 1) for sy in (-1, 1)]
    lab.scatter(pc, regs, 3)
    if lab.density > 0.15:
        a0 = radians(rng.pick([-1, 1]) * rng.u(38, 60))
        col = rng.pick(["red", "blue"])
        zc = z0 + (zl - z0) * 0.55
        pc.add("cream", cyl_decal("hat_label", R, a0, zc, 0.105, 0.066, 0.0008))
        pc.add(col, cyl_decal("hat_label_band", R, a0, zc + 0.012, 0.09, 0.02, 0.0014))
        pc.add(col, cyl_decal("hat_label_rule", R, a0, zc - 0.018, 0.09, 0.006, 0.0014))
    pc.top = zh
    pc.dims = (2 * R, 2 * R, zh)


# ----------------------------------------------------------------------------------------- carpet bag
def build_bag(pc, W, D, Hh, rng):
    ph, ph2 = rng.u(0, 6.28), rng.u(0, 6.28)
    zf = 0.016

    def sec(u, v, off=0.0):
        a = 2 * pi * u
        c, s_ = cos(a), sin(a)
        tuck = smoothstep(0.0, 0.1, v)
        vv = clamp((v - 0.1) / 0.9)
        z = zf * min(v / 0.1, 1.0) ** 2 + (Hh - zf) * vv
        ha = (W / 2) * (1 + 0.05 * sin(pi * vv ** 0.8)) * (1 - 0.12 * vv ** 2.2)
        hb = lerp((D / 2) * (1 + 0.14 * sin(pi * min(vv * 1.15, 1.0) ** 0.7)), 0.0095, vv ** 2.6)
        e = lerp(0.42, 0.3, vv)
        ins = 0.8 + 0.2 * tuck ** 0.5
        x = (ha * ins + off) * (1 if c >= 0 else -1) * abs(c) ** e
        y = (hb * ins + off) * (1 if s_ >= 0 else -1) * abs(s_) ** e
        wob = sin(pi * vv) * (1 - 0.6 * vv)
        y += 0.006 * sin(5 * a + ph) * wob
        x += 0.004 * sin(3 * a + ph2) * wob
        z -= 0.007 * sin(pi * vv) * (0.5 + 0.5 * sin(2 * a + ph))
        return (x, y, z)

    body = pc.add("carpet", surface("bag_body", sec, 28, 10, closed_u=True))
    uv_transform(body, rot_deg=90.0)           # kilim bands run round the bag
    smooth(body, 60)
    top = pc.add("trim", surface("bag_top", lambda u, v: sec(u, 0.9 + 0.1 * v, 0.0022), 28, 1, closed_u=True))
    smooth(top, 60)
    band = pc.add("trim", surface("bag_base", lambda u, v: sec(u, 0.035 + 0.2 * v, 0.003), 28, 2, closed_u=True))
    smooth(band, 60)
    Lf = W * 0.88 + 0.012
    for sy in (-1, 1):
        pc.add("brass", box("bag_frame", (Lf, 0.011, 0.018), loc=(0, sy * 0.0064, Hh + 0.001), bevel=0.0025, segments=1))
    pc.add("brass", box("bag_clasp", (0.04, 0.03, 0.011), loc=(0, 0, Hh + 0.0125), bevel=0.002, segments=1))
    pc.add("brass", cylinder("bag_knob", 0.006, 0.013, loc=(0, -0.004, Hh + 0.023), verts=8, radius2=0.0042))
    for sx in (-1, 1):
        pc.add("brass", cylinder("bag_hinge", 0.0105, 0.03, loc=(sx * Lf / 2, 0, Hh + 0.001), rot=(90, 0, 0), verts=8))
    for sy in (-1, 1):
        hA = rng.u(0.075, 0.105)
        lean = rng.u(0.004, 0.03)
        n = 9
        pts = []
        for i in range(n):
            t = i / (n - 1)
            s_ = sin(pi * t)
            pts.append((-0.072 + 0.144 * t, sy * (0.03 + lean * s_), Hh - 0.034 + (hA + 0.034) * s_ ** 0.7))
        pc.add("trim", sweep("bag_handle", pts, 0.0072, sides=6, smooth_path=False))
        for sx in (-1, 1):
            pc.add("trim", box("bag_tab", (0.026, 0.0045, 0.056), loc=(sx * 0.072, sy * 0.034, Hh - 0.04), rot=(sy * 28, 0, 0), bevel=0.0012, segments=1))
            pc.rivet_rot((sx * 0.072, sy * 0.0405, Hh - 0.046), (-sy * 62, 0, 0), 1.0)
            pc.rivet_rot((sx * 0.072, sy * 0.029, Hh - 0.024), (-sy * 62, 0, 0), 1.0)
    pc.top = Hh + 0.1
    pc.dims = (W * 1.06, D * 1.16, pc.top)


# ----------------------------------------------------------------------------------------- weathering
def weather(o, key, w, seed):
    ground = mask_mul(height_ramp(0.0, 0.2), 0.45 + 0.5 * w)
    if key in ("main", "alt", "trim"):
        paint(o, wear=mask_mul(noise_mask(0.08, seed=seed, lo=0.55 - 0.2 * w, hi=0.85), 0.25 + 0.7 * w), grime=ground)
        paint(o, grime=mask_mul(noise_mask(0.22, seed=seed + 3, lo=0.45, hi=0.8), 0.15 + 0.45 * w))
    elif key == "canvas":
        paint(o, grime=mask_mul(height_ramp(0.0, 0.16), 0.3 + 0.4 * w),
              special=mask_mul(noise_mask(0.3, seed=seed + 1, lo=0.25, hi=0.7), 0.25 + 0.6 * w))
        paint(o, grime=mask_mul(noise_mask(0.2, seed=seed + 4, lo=0.5, hi=0.9), 0.04 + 0.2 * w))
    elif key == "carpet":
        z0 = o_min_z(o)
        paint(o, grime=mask_mul(height_ramp(z0, z0 + 0.12), 0.3 + 0.4 * w),
              special=mask_mul(noise_mask(0.12, seed=seed + 5, lo=0.35, hi=0.75), 0.2 + 0.6 * w))
        # broad woven bands: every third weft group is a paler, sun-bleached yarn
        paint(o, special=lambda co, nrm: (0.22 + 0.2 * w) * (0.5 + 0.5 * np.sin((co[:, 2] - z0) * (2 * pi / 0.092) + seed)))
    elif key == "wood":
        paint(o, grime=ground, wear=mask_mul(noise_mask(0.06, seed=seed + 6, lo=0.5 - 0.15 * w, hi=0.85), 0.1 + 0.45 * w))
        paint(o, grime=mask_mul(noise_mask(0.18, seed=seed + 7, lo=0.4, hi=0.8), 0.15 + 0.4 * w))
    elif key == "brass":
        paint(o, special=mask_mul(noise_mask(0.05, seed=seed + 8, lo=0.55 - 0.1 * w, hi=0.85), 0.05 + 0.45 * w),
              wear=mask_mul(noise_mask(0.04, seed=seed + 9, lo=0.3, hi=0.7), 0.3 - 0.1 * w))   # satin, not mirror
    else:
        paint(o, grime=mask_mul(noise_mask(0.035, seed=seed + 10, lo=0.45, hi=0.8), 0.2 + 0.6 * w),
              special=mask_mul(noise_mask(0.06, seed=seed + 11, lo=0.5, hi=0.8), 0.7 * w))


def o_min_z(o):
    return world_bbox([o])[0].z


# ----------------------------------------------------------------------------------------- build
def build(p, rng):
    w = clamp(float(p.wear))
    n = int(p.pieces)
    pal = PALETTES[p.palette]
    lg = dict(wear=round(0.7 + 0.9 * w, 2), grime=round(0.7 + 0.6 * w, 2))
    mats = {
        "main": mat("leather", pal["main"], **lg),
        "alt": mat("leather", pal["alt"], **lg),
        "trim": mat("leather", pal["trim"], **lg),
        "canvas": mat("canvas", pal["trunk"], grime=round(0.45 + 0.4 * w, 2), special=round(0.6 + 0.6 * w, 2)),
        "carpet": mat("canvas", pal["carpet"], stripe=0.034, tint2=pal["carpet2"], scale=0.6),
        "wood": mat("wood_oak", pal["wood"], wear=round(0.7 + 0.5 * w, 2)),
        "brass": mat("brass", grime=0.3, special=round(0.3 + 0.6 * w, 2), rough=round(2.0 + 0.2 * w, 2)),
        "cream": mat("paper", "#eadfc4"),
        "red": mat("paper", "#b8432f"),
        "blue": mat("paper", "#3f6f8f"),
    }
    rv, nl_ = rivet_mesh(), nail_mesh()
    assign(rv, mats["brass"])
    assign(nl_, mats["brass"])
    lab = Labeler(rng.fork("labels"), float(p.labels) * (0.75 + 0.5 * w))
    done = []                                   # (key, object)
    info = {}

    def emit(pc, M, snap=None):
        res = pc.finish(M, rv, nl_)
        if snap is not None:
            mn, _ = world_bbox(list(res.values()))
            for o in res.values():
                o.data.transform(Matrix.Translation((0, 0, snap - mn.z)))
                o.data.update()
        print("DBG", pc.name, "tris", sum(_tris(o) for o in res.values()))
        done.extend(res.items())

    r = rng.fork("layout")
    sl = 1.0 + w                                # sloppiness of the stacking
    sc = r.vary(1.0, 0.04)
    Wt, Dt, Ht = 0.92 * sc, 0.52 * sc, 0.58 * sc
    dL = (r.vary(0.74, 0.03), r.vary(0.46, 0.02), r.vary(0.205, 0.05))
    dM = (r.vary(0.64, 0.03), r.vary(0.41, 0.02), r.vary(0.178, 0.05))
    dS = (r.vary(0.52, 0.03), r.vary(0.34, 0.02), r.vary(0.15, 0.05))
    dV = (r.vary(0.34, 0.04), r.vary(0.25, 0.03), r.vary(0.2, 0.05))
    Rh, Hh = r.vary(0.165, 0.04), r.vary(0.215, 0.06)
    dB = (r.vary(0.4, 0.04), r.vary(0.18, 0.04), r.vary(0.27, 0.05))
    display, tower = p.style == "display", p.style == "tower"

    # ---- trunk
    trunk = Piece("trunk", mats)
    build_trunk(trunk, Wt, Dt, Ht, rng.fork("trunk"), p.monogram, lab, flat_handle=display)
    yaw_t = r.jitter(0.0, 1.5 * sl)
    if display:
        tcx = -0.03 - Ht / 2
        emit(trunk, _mx((tcx, 0, Wt / 2), rz=yaw_t) @ _mx((Ht / 2, 0, 0), ry=-90), snap=0.0)
    else:
        emit(trunk, _mx(rz=yaw_t))

    # ---- flat suitcases (a list of supports: x, y, top z, W, D)
    sup = dict(x=0.0, y=0.0, z=Ht, W=Wt, D=Dt)
    if display:
        sup = dict(x=0.03 + dL[0] / 2 + 0.02, y=-(Dt - dL[1]) / 2 + 0.015, z=0.0, W=dL[0], D=dL[1])
    flats = []
    if n >= 2:
        flats.append(("case_l", dL, "main", "trim", True, 5))
    if n >= 3:
        flats.append(("case_m", dM, "alt", "trim", True, 4))
    if n >= 6 and tower:
        flats.append(("case_s", dS, "main", "brass", False, 4))
    sgn = r.pick([-1, 1])
    tag_i = int(r.u(0, len(flats) - 1e-6)) if flats else -1
    right_edge = None
    for i, (nm, d, mk, ck, st, nl) in enumerate(flats):
        pc = Piece(nm, mats)
        exposed = i == len(flats) - 1 and (n < 4 or display and n < 4)
        build_case(pc, d[0], d[1], d[2], rng.fork(nm), lab, main=mk, corner=ck, straps=st, n_labels=nl if exposed else 3,
                   lid_labels=exposed, tag=(i == tag_i))
        if i == 0:
            x = sup["x"] + (0.0 if display else r.jitter(-0.03 if not tower else 0.0, 0.02))
            y = sup["y"] + (0.0 if display else r.jitter(-0.005, 0.012))
        else:
            x = sup["x"] + (r.jitter(-0.03, 0.012) if not tower else r.jitter(0.0, 0.02 * sl))
            y = sup["y"] + r.jitter(0.006, 0.01)
        yaw = sgn * r.u(1.5, 2.5 + 3.0 * sl) * (1.6 if tower else 1.0)
        sgn = -sgn
        emit(pc, _mx((x, y, sup["z"]), rz=yaw))
        if i == 0:
            right_edge = (x + d[0] / 2, sup["z"] + pc.top, y)
        sup = dict(x=x, y=y, z=sup["z"] + pc.top, W=d[0], D=d[1])

    # ---- hat box
    if n >= 4:
        pc = Piece("hatbox", mats)
        build_hatbox(pc, Rh, Hh, rng.fork("hat"), lab)
        if tower:
            x, y = sup["x"] + r.jitter(0.0, 0.03), sup["y"] + r.jitter(0.0, 0.015)
        else:
            x, y = sup["x"] - sup["W"] / 2 + Rh + 0.02, sup["y"] + r.jitter(0.004, 0.012)
        emit(pc, _mx((x, y, sup["z"]), rz=r.u(-35, 25)))
        hat_top = sup["z"] + pc.top

    # ---- carpet bag
    if n >= 5:
        pc = Piece("bag", mats)
        build_bag(pc, dB[0], dB[1], dB[2], rng.fork("bag"))
        if tower:
            yaw = r.u(72, 100)
            hxb = 0.5 * (pc.dims[0] * abs(cos(radians(yaw))) + pc.dims[1] * abs(sin(radians(yaw))))
            loc = (Wt / 2 + 0.035 + hxb, r.jitter(-0.06, 0.03), 0.0)
        elif display:
            yaw = r.u(74, 100)
            loc = (tcx + r.jitter(0.0, 0.02), r.jitter(0.0, 0.03), Wt + 0.002)
        else:
            yaw = r.u(68, 80)
            loc = (sup["x"] + sup["W"] / 2 - 0.125, sup["y"] + r.jitter(0.0, 0.01), sup["z"])
        emit(pc, _mx(loc, rz=yaw))

    # ---- small case leaning upright (stack / display)
    if n >= 6 and not tower:
        pc = Piece("case_s", mats)
        build_case(pc, dS[0], dS[1], dS[2], rng.fork("case_s"), lab, main="main", corner="brass", straps=False, n_labels=5)
        if display and right_edge:
            lean = r.u(19, 23)
            px = right_edge[0] + 0.004 + right_edge[1] * tan(radians(lean))
            py = right_edge[2] + r.jitter(0.0, 0.02)
        else:
            lean = r.u(11, 15)
            px = Wt / 2 + 0.006 + (dS[1] + 0.008) * sin(radians(lean))
            py = r.jitter(0.0, 0.03)
        M = _mx((px, py, 0.0), ry=-lean) @ _mx((0, 0, dS[1] / 2 + 0.008)) @ _mx(rz=-90) @ _mx(rx=-90)
        emit(pc, M, snap=0.0)

    # ---- vanity case on the ground in front
    if n >= 7:
        pc = Piece("vanity", mats)
        build_case(pc, dV[0], dV[1], dV[2], rng.fork("vanity"), lab, main="alt" if tower else "main", corner="brass", straps=False, n_labels=3)
        yaw = r.u(8, 20) * r.pick([-1, 1])
        reach = dV[1] / 2 * cos(radians(yaw)) + dV[0] / 2 * abs(sin(radians(yaw))) + 0.03
        if display:
            loc = (0.03 + dL[0] * 0.55, -Dt / 2 + 0.015 - reach, 0.0)
        else:
            loc = (-Wt / 2 + 0.2, -Dt / 2 - reach, 0.0)
        emit(pc, _mx(loc, rz=yaw))

    delete([rv, nl_])

    # ---- centre the footprint on the origin, sit exactly on the paving, then weather in place
    objs = [o for _, o in done]
    mn, mx = world_bbox(objs)
    T = Matrix.Translation((-(mn.x + mx.x) / 2, -(mn.y + mx.y) / 2, -mn.z))
    for o in objs:
        o.data.transform(T)
        o.data.update()
    for i, (key, o) in enumerate(done):
        weather(o, key, w, int(p.seed) * 13 + i)
    print("DBG size", tuple(round(c, 3) for c in (mx - mn)))
