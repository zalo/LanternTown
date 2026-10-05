"""Clock-tower hall: the civic landmark at the head of the plaza. A tall masonry hall (coursed block plinth,
buttressed bays with two-tier traceried windows of leaded stained glass, corbelled eaves, steep tiled roof with
modelled eaves / ridge tiles, gutters, shuttered dormers, coped gables with rose windows and chimneys, optional
round stair turret) and a square clock tower standing forward on the axis: moulded portal with a glazed fanlight and oak doors with grille lights up a
flight of stone steps, clasping corner piers with quoins and weathered set-offs, open-louvred belfry glowing from
within, four lit clock dials (hands from `time`), corbel table, pierced parapet with pinnacles and a slim bell-cast
spire with four lucarnes, hip tiles, gilded ball and a rooster weathervane. Cast-iron lamp standards flank the steps
and stand on the kerbed forecourt terrace in line with the buttresses, short bracket lanterns hang from buttresses,
turrets and the rear, great lanterns on long brackets light the dials. Six lights: landing, one per wing, three dials.
Front wall plane of the tower is y = 0, the body extends to +Y, the steps project < 3 m to -Y, spire tip <= 26 m.
"""
from math import cos, sin, tan, atan, atan2, acos, radians, degrees, pi, sqrt, hypot, floor, ceil

import bmesh
import numpy as np
from mathutils import Matrix, Vector
from tk import *

META = {"name": "Clock-tower hall", "class": "landmark", "mount": "ground", "focus": [0.0, -0.4, 4.0]}
PARAMS = {
    "width": {"default": 26.0, "min": 22.0, "max": 32.0, "desc": "overall width along x (m)"},
    "depth": {"default": 12.0, "min": 10.0, "max": 16.0, "desc": "overall depth along +y (m)"},
    "tower_height": {"default": 16.5, "min": 14.5, "max": 17.5, "desc": "tower height to the top of the cornice (m); the hall eaves follow it"},
    "spire_height": {"default": 7.6, "min": 5.5, "max": 8.2, "desc": "spire height above the cornice (m), clamped so the vane stays under 26 m"},
    "time": {"default": 10.17, "min": 0.0, "max": 12.0, "desc": "time shown by the four dials, in hours (10.17 = 10:10)"},
    "stone_tint": {"default": "#d6c9ad", "desc": "tint of the dressed stone (plinth, quoins, tracery, cornices)"},
    "wall_finish": {"default": "ashlar", "choices": ["ashlar", "brick", "render"], "desc": "walling between the stone dressings"},
    "wall_tint": {"default": "#e2cfa6", "desc": "limewash colour of the walls when wall_finish = render"},
    "roof": {"default": "clay", "choices": ["clay", "slate", "shingle"], "desc": "roof covering of hall, spire and turret"},
    "spire": {"default": "pyramid", "choices": ["pyramid", "octagonal"], "desc": "square pyramid spire or octagonal spire"},
    "arch": {"default": "pointed", "choices": ["pointed", "round"], "desc": "arch form of windows, portal and belfry"},
    "turret": {"default": "left", "choices": ["left", "right", "both", "none"], "desc": "round stair turret(s) at the front corners"},
    "dormers": {"default": 2, "min": 0, "max": 3, "desc": "dormers per wing on the front roof slope"},
    "glass_a": {"default": "#3560e0", "desc": "stained glass: main field colour"},
    "glass_b": {"default": "#f08a2a", "desc": "amber glass: lantern panes, belfry glow behind the louvres, window borders, dormer and slit margins"},
    "glass_c": {"default": "#c2342c", "desc": "stained glass: third colour (lower lights, quarries, rose petals)"},
    "window_glow": {"default": 7.0, "min": 0.5, "max": 10.0, "desc": "brightness of the stained glass and (scaled down) of the amber lantern / belfry glass"},
    "dial_glow": {"default": 7.0, "min": 0.5, "max": 10.0, "desc": "brightness of the lamp-lit opal glass: clock dials, cores of the hall windows, dormers, lucarnes and slits, fanlight, flames"},
    "lantern_intensity": {"default": 90.0, "min": 30.0, "max": 130.0, "desc": "base lamp strength (cd); the six lights (landing, one per wing, three dials) are fixed multiples of it"},
    "door_tint": {"default": "#6a4528", "desc": "stain of the oak doors, shutters and louvres"},
    "inscription": {"default": "ANNO MDCCCXLII", "desc": "gilded lettering on the frieze over the portal"},
    "moss": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of moss / damp weathering"},
}
VARIANTS = {
    "brick_slate_wide": {"width": 32.0, "depth": 14.0, "tower_height": 17.0, "spire_height": 8.2, "wall_finish": "brick",
                         "roof": "slate", "spire": "octagonal", "turret": "both", "dormers": 3, "time": 3.5, "stone_tint": "#e0d6bf",
                         "glass_a": "#2f8f6a", "glass_b": "#e8b040", "glass_c": "#b8345a", "seed": 7},
    "render_compact": {"width": 22.0, "depth": 10.0, "tower_height": 14.5, "spire_height": 6.0, "wall_finish": "render",
                       "wall_tint": "#e9d39b", "arch": "round", "turret": "none", "dormers": 1, "time": 7.75,
                       "glass_a": "#b8402e", "glass_b": "#f2c15a", "glass_c": "#3a78c8", "door_tint": "#3d4a34", "seed": 13},
    "grey_shingle_tall": {"width": 27.0, "depth": 16.0, "tower_height": 17.5, "spire_height": 8.2, "roof": "shingle", "spire": "octagonal",
                          "stone_tint": "#aeb0ad", "turret": "right", "time": 1.9, "glass_a": "#7a4fc0",
                          "glass_b": "#e88a3a", "glass_c": "#2f8f5a", "moss": 0.9, "seed": 21},
    "render_slate_round": {"width": 29.0, "depth": 13.0, "tower_height": 15.5, "spire_height": 8.0, "wall_finish": "render",
                           "wall_tint": "#d9b7a0", "roof": "slate", "arch": "round", "turret": "both", "dormers": 2,
                           "time": 5.4, "stone_tint": "#cfc2a8", "seed": 34},
}


# ----------------------------------------------------------------------------- 2D outlines
def _arch_dims(w, h, kind):
    half = w / 2
    rise = half if kind == "round" else max(half * 1.02, min(0.78 * w, h * 0.6))
    return rise, h - rise


def _arch(w, h, kind="pointed", off=0.0, n=8, base=0.0, cx=0.0):
    """Arched opening outline (x across, z up), CCW from the bottom right. `off` grows jambs and head."""
    half = w / 2
    rise, spring = _arch_dims(w, h, kind)
    spring += base
    pts = [(half + off, base)]
    if kind == "round":
        for i in range(2 * n + 1):
            a = pi * i / (2 * n)
            pts.append(((half + off) * cos(a), spring + (half + off) * sin(a)))
    else:
        c = (rise * rise - half * half) / w
        ro = half + c + off
        amax = acos(c / ro)
        right = [(-c + ro * cos(amax * i / n), spring + ro * sin(amax * i / n)) for i in range(n + 1)]
        pts += right + [(-x, z) for (x, z) in reversed(right[:-1])]
    pts.append((-half - off, base))
    return [(x + cx, z) for x, z in pts]


def _clip_above(poly, z0):
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = a[1] >= z0, b[1] >= z0
        if ina:
            out.append(a)
        if ina != inb:
            t = (z0 - a[1]) / (b[1] - a[1])
            out.append((a[0] + (b[0] - a[0]) * t, z0))
    return out


def _band(w, h, kind, o0, o1, n=8, base=0.0, cx=0.0, clip=None):
    poly = _arch(w, h, kind, o1, n, base, cx) + _arch(w, h, kind, o0, n, base, cx)[::-1]
    return _clip_above(poly, clip) if clip is not None else poly


def _circle(r, n=20, cx=0.0, cz=0.0):
    return [(cx + r * cos(2 * pi * i / n), cz + r * sin(2 * pi * i / n)) for i in range(n)]


def _foil(nl, r, cx=0.0, cz=0.0, seg=6, a0=pi / 2):
    """Trefoil / quatrefoil outline of overall radius ~r (CCW)."""
    d, rl = r * 0.52, r * 0.5
    t = d * cos(pi / nl) + sqrt(max(rl * rl - (d * sin(pi / nl)) ** 2, 0.0))
    pts = []
    for i in range(nl):
        ph = a0 + 2 * pi * i / nl
        lx, lz = d * cos(ph), d * sin(ph)
        pa = (t * cos(ph - pi / nl), t * sin(ph - pi / nl))
        pb = (t * cos(ph + pi / nl), t * sin(ph + pi / nl))
        aa = atan2(pa[1] - lz, pa[0] - lx)
        ab = atan2(pb[1] - lz, pb[0] - lx)
        while ab <= aa:
            ab += 2 * pi
        for k in range(seg):
            a = aa + (ab - aa) * k / seg
            pts.append((cx + lx + rl * cos(a), cz + lz + rl * sin(a)))
    return pts


# ----------------------------------------------------------------------------- mesh helpers
def _prism(name, prof, x0, x1):
    """Profile [(out, z)] (out = towards -Y) extruded along x."""
    n = len(prof)
    vs = [(x0, -o, z) for o, z in prof] + [(x1, -o, z) for o, z in prof]
    fs = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    fs += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    return o


def _extrude_y(name, prof, y0, y1):
    """Profile [(x, z)] extruded along y."""
    n = len(prof)
    vs = [(x, y0, z) for x, z in prof] + [(x, y1, z) for x, z in prof]
    fs = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    fs += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    return o


def _loop(name, poly, prof, closed=True):
    """Run a moulding profile [(out, z)] around a plan polygon [(x, y)] (CCW seen from above) with true mitres."""
    P = [Vector((x, y)) for x, y in poly]
    n = len(P)

    def nrm(a, b):
        d = (P[b] - P[a]).normalized()
        return Vector((d.y, -d.x))

    ms = []
    for i in range(n):
        if closed or 0 < i < n - 1:
            n1, n2 = nrm((i - 1) % n, i), nrm(i, (i + 1) % n)
            ms.append((n1 + n2) / (1.0 + n1.dot(n2)))
        elif i == 0:
            ms.append(nrm(0, 1))
        else:
            ms.append(nrm(n - 2, n - 1))
    k = len(prof)
    vs = [(P[i].x + ms[i].x * o, P[i].y + ms[i].y * o, z) for i in range(n) for (o, z) in prof]
    fs = []
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        for a in range(k):
            b = (a + 1) % k
            fs.append((i * k + a, i * k + b, j * k + b, j * k + a))
    if not closed:
        fs.append(tuple(range(k)))
        fs.append(tuple((n - 1) * k + a for a in range(k))[::-1])
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    return o


def _slab(name, pts, thick):
    """Thin slab under a planar polygon given CCW as seen from outside; UVs: U along pts[0]->pts[1], V up the slope."""
    P = [Vector(q) for q in pts]
    nr = (P[1] - P[0]).cross(P[2] - P[0]).normalized()
    n = len(P)
    vs = [tuple(q) for q in P] + [tuple(q - nr * thick) for q in P]
    fs = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    fs += [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    u = (P[1] - P[0]).normalized()
    uv_planar(o, u_axis=tuple(u), v_axis=tuple(nr.cross(u)))
    return o


def _frustum(name, b0, b1):
    """Box frustum between rectangles b = (x0, x1, y0, y1, z)."""
    vs = []
    for (x0, x1, y0, y1, z) in (b0, b1):
        vs += [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    fs = [(3, 2, 1, 0), (4, 5, 6, 7)] + [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    return o


class _Face:
    """A facade: elements are built facing -Y on the plane y = 0 (x = along, z = up) and put() onto it."""

    def __init__(self, ox, oy, yaw):
        self.ox, self.oy, self.yaw = ox, oy, yaw
        a = radians(yaw)
        self.ux, self.uy, self.nx, self.ny = cos(a), sin(a), sin(a), -cos(a)

    def pt(self, u=0.0, o=0.0, z=0.0):
        return (self.ox + self.ux * u + self.nx * o, self.oy + self.uy * u + self.ny * o, z)

    def put(self, obj):
        apply_transform(obj)
        place(obj, loc=(self.ox, self.oy, 0.0), rot=(0, 0, self.yaw))
        apply_transform(obj)
        return obj

    def M(self, u=0.0, o=0.0, z=0.0, scale=1.0):
        return trs(self.pt(u, o, z), (0, 0, self.yaw), scale)


def _fx(o):
    apply_transform(o)
    return o


def _strip(o, direction, thresh=0.9):
    """Remove faces facing `direction` (hidden backs of wall panels / undersides of roof slabs)."""
    d = Vector(direction).normalized()
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    kill = [f for f in bm.faces if f.normal.dot(d) > thresh]
    if kill:
        bmesh.ops.delete(bm, geom=kill, context='FACES')
    bm.to_mesh(o.data)
    bm.free()
    return o


def _grid_cut(o, dx=None, dy=None, dz=None, dissolve=False, zs=()):
    """Cut the mesh on a regular world grid so weathering masks get clean vertical / horizontal resolution
    (rain streaks need vertex columns) without the triangle soup of a uniform subdivision."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    if dissolve:
        bmesh.ops.dissolve_limit(bm, angle_limit=radians(1.0), verts=bm.verts[:], edges=bm.edges[:])
    for ax, d in enumerate((dx, dy, dz)):
        if not d or not bm.verts:
            continue
        cs = [v.co[ax] for v in bm.verts]
        lo, hi = min(cs), max(cs)
        n = int((hi - lo) / d)
        if n < 1:
            continue
        step = (hi - lo) / (n + 1)
        no = Vector((0.0, 0.0, 0.0))
        no[ax] = 1.0
        for i in range(1, n + 1):
            co = Vector((0.0, 0.0, 0.0))
            co[ax] = lo + step * i
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5, plane_co=co, plane_no=no)
    for z in zs:                                           # crisp starts for run-off streaks under drip lines
        for zz in (z + 0.03, z - 0.012):
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-4, plane_co=Vector((0.0, 0.0, zz)),
                                   plane_no=Vector((0.0, 0.0, 1.0)))
    bm.to_mesh(o.data)
    bm.free()
    return o


GUV = (0.37, 0.61)


def _const_uv(o):
    """Collapse the UVs to one texel: the stained-glass material then gives one clean colour per piece (its own
    random quarry pattern is replaced by modelled leading)."""
    me = o.data
    lay = me.uv_layers[0] if me.uv_layers else me.uv_layers.new(name="UVMap")
    lay.data.foreach_set("uv", np.tile(np.array(GUV, dtype=np.float32), len(me.loops)))
    uv_keep(o)
    return o


def _flat(name, polys, y=0.0):
    """Flat single-sided faces [(x, z), ...] (CCW seen from the front) on the plane y, facing -Y."""
    vs, fs = [], []
    for poly in polys:
        k = len(vs)
        vs += [(x, y, z) for x, z in poly]
        fs.append(tuple(range(k, k + len(poly))))
    return mesh_object(name, vs, fs)


def _inside(poly, x, z):
    n = len(poly)
    for i in range(n):
        ax, az = poly[i]
        bx, bz = poly[(i + 1) % n]
        if (bx - ax) * (z - az) - (bz - az) * (x - ax) < -1e-9:
            return False
    return True


def _clip_seg(x0, z0, x1, z1, rx0, rx1, rz0, rz1):
    dx, dz = x1 - x0, z1 - z0
    t0, t1 = 0.0, 1.0
    for pq, q in ((-dx, x0 - rx0), (dx, rx1 - x0), (-dz, z0 - rz0), (dz, rz1 - z0)):
        if abs(pq) < 1e-12:
            if q < 0:
                return None
        else:
            t = q / pq
            if pq < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
    if t1 - t0 < 1e-6:
        return None
    return (x0 + dx * t0, z0 + dz * t0, x0 + dx * t1, z0 + dz * t1)


def _bar(x0, z0, x1, z1, wd):
    dx, dz = x1 - x0, z1 - z0
    ln = hypot(dx, dz)
    nx, nz = -dz / ln * wd / 2, dx / ln * wd / 2
    return [(x0 - nx, z0 - nz), (x1 - nx, z1 - nz), (x1 + nx, z1 + nz), (x0 + nx, z0 + nz)]


def _lattice(rx0, rx1, rz0, rz1, xa, za, qx, qz, wd=0.011):
    """Lead cames of a diamond-quarry lattice (both diagonals through (xa + k qx, za)) clipped to a rectangle."""
    out = []
    s = qz / qx
    span = (rz1 - rz0) / s + (rx1 - rx0)
    k0 = int(floor((rx0 - span - xa) / qx)) - 1
    k1 = int(ceil((rx1 + span - xa) / qx)) + 1
    for k in range(k0, k1 + 1):
        xk = xa + k * qx
        for sg in (1, -1):
            c = _clip_seg(rx0, za + sg * s * (rx0 - xk), rx1, za + sg * s * (rx1 - xk), rx0, rx1, rz0, rz1)
            if c:
                out.append(_bar(c[0], c[1], c[2], c[3], wd))
    return out


def _ring(outer, inner):
    """Quads between two outlines with the same number of points (both CCW)."""
    n = len(outer)
    return [[outer[i], outer[(i + 1) % n], inner[(i + 1) % n], inner[i]] for i in range(n)]


def _inset_arch(w, h, kind, b, n, base, cx):
    q = _arch(w, h, kind, -b, n, base, cx)
    q[0] = (q[0][0], base + b)
    q[-1] = (q[-1][0], base + b)
    return q


def _tray(name, proj, H=0.3, ch=0.022):
    """One plinth block (unit length along x, projecting `proj` to -Y, chamfered arrises, open at the back)."""
    b, a = 0.03, proj - ch
    vs = [(-0.5, b, 0), (0.5, b, 0), (0.5, b, H), (-0.5, b, H),
          (-0.5, -a, 0), (0.5, -a, 0), (0.5, -a, H), (-0.5, -a, H),
          (-0.5 + ch, -proj, ch), (0.5 - ch, -proj, ch), (0.5 - ch, -proj, H - ch), (-0.5 + ch, -proj, H - ch)]
    fs = [(8, 9, 10, 11), (4, 5, 9, 8), (5, 6, 10, 9), (6, 7, 11, 10), (7, 4, 8, 11), (1, 2, 6, 5), (0, 4, 7, 3), (3, 7, 6, 2)]
    o = mesh_object(name, vs, fs)
    uv_box(o)
    o["tsq_smooth"] = 20.0
    return o


def _tile(name, tw, tl, th):
    """One roofing tile: lower edge on y = 0, running up-slope along +y, thickness along +z."""
    x = tw / 2
    vs = [(-x, 0, 0), (x, 0, 0), (x, tl, 0), (-x, tl, 0), (-x, 0, th), (x, 0, th), (x, tl, th), (-x, tl, th)]
    fs = [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (3, 0, 4, 7), (3, 2, 1, 0)]
    o = mesh_object(name, vs, fs)
    uv_planar(o, u_axis=(1, 0, 0), v_axis=(0, 1, 0))
    o["tsq_smooth"] = 20.0
    return o


def _along(pts, step):
    """Resample a polyline into ~step long pieces: [(mid, dir, length)]."""
    P = [Vector(q) for q in pts]
    segs = [(P[i], P[i + 1]) for i in range(len(P) - 1)]
    tot = sum((b - a).length for a, b in segs)
    n = max(1, int(round(tot / step)))
    d = tot / n

    def at(s):
        for a, b in segs:
            ln = (b - a).length
            if s <= ln + 1e-9:
                return a + (b - a) * (s / ln)
            s -= ln
        return P[-1].copy()

    out = []
    for i in range(n):
        a, b = at(i * d), at((i + 1) * d)
        out.append(((a + b) / 2, (b - a).normalized(), d))
    return out


def _noise1(s, seed):
    i = np.floor(s)
    f = s - i
    f = f * f * (3 - 2 * f)

    def h(k):
        x = np.sin(k * 127.1 + seed * 311.7) * 43758.5453
        return x - np.floor(x)

    return h(i) * (1 - f) + h(i + 1) * f


def _hash1(i, seed):
    x = np.sin(i * 127.1 + seed * 311.7) * 43758.5453
    return x - np.floor(x)


def _runs(tops, length, seed, dens=0.3, w=0.085):
    """Rain run-off: narrow vertical streaks (one vertex column wide) hanging below each drip line in `tops`
    (z heights or fn(co) -> z): random columns, random length and strength, darkest right under the line."""
    def fn(co, n):
        col = np.floor((co[:, 0] + co[:, 1] * 1.37) / w + 0.5)
        out = np.zeros(len(co))
        for k, zt in enumerate(tops):
            ztv = zt(co) if callable(zt) else zt
            on = _hash1(col + 57.0 * k, seed) > 1.0 - dens
            st = 0.45 + 0.55 * _hash1(col + 13.0 * k, seed + 5)
            ln = length * (0.2 + 1.0 * _hash1(col + 31.0 * k, seed + 9) ** 2)
            d = ztv - co[:, 2]
            fall = np.where(d >= -0.005, np.clip(1.0 - d / ln, 0.0, 1.0), 0.0)
            out = np.maximum(out, on * st * fall)
        return out
    return fn


def _marks(face, marks):
    """Run-off streaks at chosen places of a facade (below sill ends, label stops, corbels):
    marks = [(u along the face, z of the drip, length, strength, half width)]."""
    def fn(co, n):
        s = (co[:, 0] - face.ox) * face.ux + (co[:, 1] - face.oy) * face.uy
        near = np.abs((co[:, 0] - face.ox) * face.nx + (co[:, 1] - face.oy) * face.ny) < 0.9
        out = np.zeros(len(co))
        for (u, zt, ln, st, hw) in marks:
            d = zt - co[:, 2]
            fall = np.where(d >= -0.005, np.clip(1.0 - d / ln, 0.0, 1.0), 0.0)
            out = np.maximum(out, st * np.clip(1.0 - np.abs(s - u) / hw, 0.0, 1.0) * fall)
        return out * near
    return fn


# ----------------------------------------------------------------------------- reusable elements (face -Y, plane y = 0)
def _window_geo(w, h, kind):
    rise, spring = _arch_dims(w, h, kind)
    mul, jm = 0.13, 0.09
    lw = (w - 2 * jm - mul) / 2
    lcx = mul / 2 + lw / 2
    ztr = spring * 0.4                                     # transom
    lh = spring + lw * 0.32                                # apex of the upper lights
    fr = (0.25 if kind == "pointed" else 0.27) * rise
    fz = spring + (0.56 if kind == "pointed" else 0.5) * rise
    lights = []
    for sx in (-1, 1):
        lights.append((sx * lcx, 0.05, ztr - 0.07 - 0.05))
        lights.append((sx * lcx, ztr + 0.07, lh - ztr - 0.07))
    return {"rise": rise, "spring": spring, "lw": lw, "lcx": lcx, "ztr": ztr, "lh": lh, "fr": fr, "fz": fz, "lights": lights}


def _window(name, w, h, kind, stone, fine=True):
    """Stonework of a two-tier traceried window: moulded surround, hood mould with stops, sill, plate tracery with
    four lights under a quatrefoil, moulded transom. Origin = middle of the sill."""
    g = _window_geo(w, h, kind)
    spring = g["spring"]
    n = 8 if fine else 6
    bv = 0.012 if fine else 0.0
    objs = [shape_extrude(name + "_surround", _band(w, h, kind, 0.0, 0.2, n), depth=0.13, loc=(0, 0.05, 0), rot=(90, 0, 0), bevel=bv),
            shape_extrude(name + "_hood", _band(w, h, kind, 0.2, 0.31, n, clip=spring - 0.06), depth=0.18, loc=(0, 0.05, 0),
                          rot=(90, 0, 0), bevel=bv)]
    for sx in (-1, 1):
        objs.append(sphere(name + "_stop", 0.085, loc=(sx * (w / 2 + 0.255), -0.09, spring - 0.08), segments=8, rings=5))
    objs.append(box(name + "_sill", (w + 0.62, 0.4, 0.15), loc=(0, -0.04, -0.075), bevel=0.02 if fine else 0.0, segments=1))
    nl = 6 if fine else 5
    holes = [_arch(g["lw"], hh, kind, 0.0, nl, base, cx) for cx, base, hh in g["lights"]]
    holes.append(_foil(4, g["fr"], 0.0, g["fz"], 6 if fine else 4))
    objs.append(shape_extrude(name + "_tracery", _arch(w, h, kind, 0.01, n), holes, depth=0.12, loc=(0, 0.26, 0), rot=(90, 0, 0),
                              bevel=0.006 if fine else 0.0))
    objs.append(_prism(name + "_transom", [(-0.12, g["ztr"] - 0.07), (-0.1, g["ztr"] - 0.085), (-0.07, g["ztr"] - 0.085), (-0.07, g["ztr"] + 0.05),
                                           (-0.12, g["ztr"] + 0.085)], -w / 2 + 0.02, w / 2 - 0.02))
    for o in objs:
        _fx(o)
        assign(o, stone)
    return join(objs, name)


def _pattern(design, up, i, j, nq):
    par = (1 - nq) % 2
    pp, qq = (i - nq + j - par) // 2, (j - (i - nq) - par) // 2
    if design == 0:
        return pp % 2 == 0 and qq % 2 == 0
    if design == 1:
        return (pp % 4 == 0 or qq % 4 == 0) if up else (pp % 2 == 0 and qq % 2 == 0)
    return (pp % 3 == 0 and qq % 3 == 0) or j % 12 == 6


def _window_glass(name, w, h, kind, design, mats):
    """Leaded glazing for _window: coloured border strip, field of diamond quarries with a regular diaper of
    accent quarries, medallions, modelled lead cames and iron saddle bars. mats = (a, b, c, clear, iron);
    every light has a coloured edge strip and band round a warm lamp-lit core of amber quarries (mats[1]), so the
    window reads as lit from inside at a distance; three designs mix the hues differently."""
    g = _window_geo(w, h, kind)
    border, f_up, f_lo, a_up, a_lo, med, medc = ((1, 0, 0, 2, 0, 2, 1), (1, 2, 0, 0, 2, 0, 1), (0, 2, 2, 0, 0, 2, 1))[design]
    L = {}

    def add(mi, layer, poly):
        L.setdefault((mi, layer), []).append(poly)

    add(border, 0, [(-w / 2, 0.0), (w / 2, 0.0), (w / 2, h), (-w / 2, h)])
    lw = g["lw"]
    fw = lw - 0.1
    nq = max(3, int(round(fw / 0.115)))
    qx = fw / nq
    qz = qx * 1.5
    for (cx, base, hh) in g["lights"]:
        up = base > g["ztr"]
        field = _inset_arch(lw, hh, kind, 0.05, 6, base, cx)
        add(f_up if up else f_lo, 1, field)
        bc = 0.05 + qx * 0.62
        core = _inset_arch(lw, hh, kind, bc, 6, base, cx)
        add(3, 2, core)
        for q in _ring(_inset_arch(lw, hh, kind, bc - 0.007, 6, base, cx), _inset_arch(lw, hh, kind, bc + 0.007, 6, base, cx)):
            add(4, 6, q)
        x_l, z0 = cx - fw / 2, base + 0.05
        sp = _arch_dims(lw, hh, kind)[1]
        zc = base + (sp * 0.55 if up else hh * 0.42)
        j0 = int(round((zc - z0) / (qz / 2)))
        if (nq + j0) % 2 == 0:
            j0 += 1
        zc = z0 + j0 * qz / 2
        if up:
            rm = min(0.25, fw * 0.37)
            ring = _circle(rm, 12, cx, zc)
            add(med, 4, ring)
            add(medc, 5, _circle(rm * 0.48, 8, cx, zc))
            for k in range(12):
                add(4, 6, _bar(ring[k][0], ring[k][1], ring[(k + 1) % 12][0], ring[(k + 1) % 12][1], 0.014))
        else:
            rm = qz
            add(med, 4, [(cx, zc - qz), (cx + qx, zc), (cx, zc + qz), (cx - qx, zc)])
            add(medc, 5, [(cx, zc - qz / 2), (cx + qx / 2, zc), (cx, zc + qz / 2), (cx - qx / 2, zc)])
        j = 1
        while z0 + j * qz / 2 < base + hh:
            zq = z0 + j * qz / 2
            for i in range(1, 2 * nq):
                if (i + j) % 2 == 0 or not _pattern(design, up, i, j, nq):
                    continue
                xq = x_l + i * qx / 2
                if hypot(xq - cx, zq - zc) < rm * 0.9:
                    continue
                dm = [(xq, zq - qz / 2), (xq + qx / 2, zq), (xq, zq + qz / 2), (xq - qx / 2, zq)]
                if all(_inside(core, x, z) for x, z in dm):
                    add(a_up if up else a_lo, 3, dm)
            j += 1
        for q in _lattice(cx - lw / 2 - 0.01, cx + lw / 2 + 0.01, base - 0.01, base + hh + 0.01, x_l, z0, qx, qz):
            add(4, 6, q)
        zb = z0 + qz * 2
        while zb < base + sp - 0.04:
            add(4, 7, [(cx - lw / 2 - 0.01, zb - 0.011), (cx + lw / 2 + 0.01, zb - 0.011), (cx + lw / 2 + 0.01, zb + 0.011),
                       (cx - lw / 2 - 0.01, zb + 0.011)])
            zb += qz * 3
    fr, fz = g["fr"], g["fz"]
    add(f_up, 1, _circle(fr * 0.66, 12, 0.0, fz))
    add(medc, 4, _circle(fr * 0.27, 8, 0.0, fz))
    for k in range(4):
        a = pi / 4 + k * pi / 2
        add(4, 6, _bar(fr * 0.27 * cos(a), fz + fr * 0.27 * sin(a), fr * 1.0 * cos(a), fz + fr * 1.0 * sin(a), 0.012))
    objs = []
    for (mi, layer), polys in L.items():
        o = _flat(name + "_g", polys, 0.335 - 0.007 * layer)
        assign(o, mats[mi])
        objs.append(o)
    return _const_uv(join(objs, name))


def _slit(name, w, h, kind, stone, glass, iron, core=None):
    """Small single-light window: stone frame, sill, lit pane, glazing bars. Origin = middle of the sill."""
    fr = _fx(shape_extrude(name + "_fr", _band(w, h, kind, 0.0, 0.11, 5), depth=0.16, loc=(0, 0.09, 0), rot=(90, 0, 0)))
    sill = box(name + "_sill", (w + 0.34, 0.24, 0.09), loc=(0, -0.02, -0.045), bevel=0.0)
    assign(fr, stone)
    assign(sill, stone)
    gl = _const_uv(_flat(name + "_gl", [_arch(w, h, kind, 0.02, 5)], 0.05))
    assign(gl, glass)
    objs = [fr, sill, gl]
    if core is not None:                                   # lamp-lit core behind the amber border
        co = _const_uv(_flat(name + "_core", [_arch(w, h, kind, 0.02 + w * 0.2, 5)], 0.044))
        assign(co, core)
        objs.append(co)
    bars = [[(-0.011, 0.0), (0.011, 0.0), (0.011, h), (-0.011, h)]]
    z = h * 0.3
    while z < h * 0.8:
        bars.append([(-w / 2, z - 0.01), (w / 2, z - 0.01), (w / 2, z + 0.01), (-w / 2, z + 0.01)])
        z += h * 0.3
    br = _const_uv(_flat(name + "_bars", bars, 0.035))
    assign(br, iron)
    return join(objs + [br], name)


def _gablet(name, w, hw, gp, back, kind, M, mode="lit", shutters=True):
    """Dormer / lucarne: gabled stone front with a deeply recessed leaded light (amber glass with a lamp-lit core),
    boarded shutters (open, half closed or shut = dark), label mould, sill, apex ball, tile-hung cheeks and two roof
    slopes. Front plane y = 0, base z = 0, the body runs back to y = back. M = dict of materials."""
    objs = []

    def S(o, m):
        _fx(o)
        assign(o, M[m])
        objs.append(o)
        return o

    hg = w / 2 * tan(radians(gp))
    pent = [(-w / 2, 0), (w / 2, 0), (w / 2, hw), (0, hw + hg), (-w / 2, hw)]
    ww, wh = w * 0.5, hw * 0.84 + hg * 0.22
    zs = hw * 0.14
    spring = _arch_dims(ww, wh, kind)[1]
    S(shape_extrude(name + "_front", pent, [_arch(ww, wh, kind, 0.0, 6, zs)], depth=0.2, loc=(0, 0.2, 0), rot=(90, 0, 0)), "stone")
    S(shape_extrude(name + "_label", _band(ww, wh, kind, 0.04, 0.1, 6, zs, clip=zs + spring - 0.03), depth=0.045, loc=(0, 0.0, 0),
                    rot=(90, 0, 0)), "stone")
    S(box(name + "_sill", (ww + 0.26, 0.22, 0.07), loc=(0, -0.01, zs - 0.035), bevel=0.01, segments=1), "stone")
    S(sphere(name + "_ball", 0.07, loc=(0, 0.06, hw + hg + 0.1), segments=8, rings=5), "stone")
    yg = 0.15
    if mode == "dark":
        S(_flat(name + "_board", [_arch(ww, wh, kind, 0.02, 6, zs - 0.02)], yg), "wood")
    else:
        S(_const_uv(_flat(name + "_pane", [_arch(ww, wh, kind, 0.02, 6, zs - 0.02)], yg)), "glass")
        if mode == "lit":
            cz = zs + wh * 0.4
            S(_flat(name + "_lamp", [[(ww * 0.4 * cos(2 * pi * k / 10), cz + wh * 0.34 * sin(2 * pi * k / 10)) for k in range(10)]],
                    yg - 0.006), "lamp")
        cames = _lattice(-ww / 2, ww / 2, zs, zs + wh, 0.0, zs, 0.105, 0.158, 0.012)
        cames.append([(-0.016, zs), (0.016, zs), (0.016, zs + wh), (-0.016, zs + wh)])
        cames.append([(-ww / 2, zs + spring - 0.016), (ww / 2, zs + spring - 0.016), (ww / 2, zs + spring + 0.016), (-ww / 2, zs + spring + 0.016)])
        S(_const_uv(_flat(name + "_cames", cames, yg - 0.012)), "iron")
    if shutters:
        sh_w, sh_h = ww / 2 - 0.006, spring + 0.02
        for sx in (-1, 1):
            closed = mode == "dark" or (mode == "half" and sx < 0)
            xc, yc = (sx * ww / 4, 0.1) if closed else (sx * (ww / 2 + sh_w / 2 + 0.015), -0.02)
            o = S(box(name + "_shutter", (sh_w, 0.028, sh_h), loc=(xc, yc, zs + sh_h / 2), bevel=0.004, segments=1), "wood")
            uv_grain(o, "Z")
            for zz in (0.14, 0.86):
                o = S(box(name + "_ledge", (sh_w - 0.03, 0.014, 0.05), loc=(xc, yc - 0.021, zs + sh_h * zz)), "wood")
                uv_grain(o, "X")
    body = _extrude_y(name + "_body", [(x * 0.96, z * 0.985) for x, z in pent], 0.19, back)
    assign(body, M["cheek"])
    objs.append(body)
    e, f, th = 0.11, 0.15, 0.06
    ze, zr = hw - e * tan(radians(gp)) + 0.05, hw + hg + 0.05
    for sx in (-1, 1):
        q = [(sx * (w / 2 + e), back, ze), (sx * (w / 2 + e), -f, ze), (0, -f, zr), (0, back, zr)]
        if sx > 0:
            q = [q[1], q[0], q[3], q[2]]
        s = _slab(name + "_roof", q, th)
        assign(s, M["roof"])
        objs.append(s)
    return join(objs, name)


def _hex_glow(name, r0, z0, r1, z1, cx, cy, glass, flame):
    """Lantern glazing: six amber panes (hexagonal frustum, r0 at z0 to r1 at z1, about the vertical through (cx, cy))
    and the lamp flame showing through every pane (a small brighter teardrop set just proud of the glass)."""
    gl = _const_uv(_fx(lathe(name + "_glass", [(r0, z0), (r1, z1)], segments=6, loc=(cx, cy, 0), smooth_angle=20)))
    assign(gl, glass)
    hgt = z1 - z0
    zm, ra = z0 + hgt * 0.4, (r0 + (r1 - r0) * 0.4) * cos(pi / 6) + 0.0046     # CLEARANCE RULE: 4.5 mm proud of the pane (it was 3.95)
    sl = (r1 - r0) / hgt * cos(pi / 6)
    wf, hf = min(r0, r1) * 0.4, hgt * 0.46
    vs, fs = [], []
    for k in range(6):
        ph = pi / 6 + k * pi / 3
        c, sn = cos(ph), sin(ph)
        i0 = len(vs)
        for u, v in ((0.0, -0.5), (0.5, -0.24), (0.4, 0.06), (0.0, 0.5), (-0.4, 0.06), (-0.5, -0.24)):
            r = ra + v * hf * sl
            vs.append((cx + r * c - u * wf * sn, cy + r * sn + u * wf * c, zm + v * hf))
        fs.append(tuple(range(i0, i0 + 6)))
    fl = _const_uv(mesh_object(name + "_flame", vs, fs))
    assign(fl, flame)
    return join([gl, fl], name + "_glow")


def _sect(a, b):
    """Rectangular bar section: a in the plane of the bracket, b across it."""
    return [(a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2), (-a / 2, -b / 2)]


def _lantern(name, reach, iron, glass, flame):
    """Hexagonal lantern hung from a short scrolled wrought-iron wall bracket: bolted back plate on y = 0 (facing -Y),
    44 x 40 mm arm, 40 x 30 mm raking stay and a scroll in the angle. The lantern hangs `reach` out; panes centre on
    (0, -reach, -0.32). Returns (ironwork, glazing)."""
    bar, stay, thin = _sect(0.044, 0.04), _sect(0.04, 0.03), _sect(0.032, 0.024)
    lan = [box(name + "_plate", (0.15, 0.026, 0.8), loc=(0, -0.013, -0.12), bevel=0.006, segments=1),
           box(name + "_boss", (0.1, 0.02, 0.2), loc=(0, -0.034, 0.06), bevel=0.005, segments=1),
           box(name + "_shoe", (0.1, 0.02, 0.16), loc=(0, -0.034, -0.4), bevel=0.005, segments=1)]
    for zz in (0.22, -0.1, -0.47):
        for sx in (-1, 1):
            lan.append(cylinder(name + "_fixing", 0.014, 0.02, loc=(sx * 0.05, -0.03, zz), rot=(90, 0, 0), verts=6))
    arm = bezier_path((0, -0.03, 0.06), (0, -reach * 0.45, 0.36), (0, -reach * 0.96, 0.36), (0, -reach, 0.07), n=10)
    lan.append(sweep(name + "_arm", arm, bar, up=(1, 0, 0), smooth_path=False))
    rs = min(0.19, reach * 0.2)
    sc = spiral_path(0.03, rs, 1.35, n=14, a0_deg=20, plane="YZ", center=(0, -rs * 1.75, -0.08))
    lan.append(sweep(name + "_scroll", sc, thin, up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t))
    lan.append(sweep(name + "_stay", [(0, -0.03, -0.4), (0, -reach * 0.3, -0.26), (0, -reach * 0.66, 0.3)], stay, up=(1, 0, 0),
                     smooth_path=False))
    lan.append(box(name + "_collar", (0.06, 0.07, 0.05), loc=(0, -reach * 0.66, 0.3), rot=(-35, 0, 0), bevel=0.004, segments=1))
    ly, lz = -reach, 0.06
    lan.append(cylinder(name + "_hanger", 0.02, 0.1, loc=(0, ly, lz - 0.01), verts=6))
    lan.append(lathe(name + "_roof", [(0.185, -0.23), (0.19, -0.215), (0.1, -0.12), (0.05, -0.1), (0.035, -0.06), (0.02, -0.045), (0.0, -0.045)],
                     segments=6, loc=(0, ly, lz), smooth_angle=20))
    lan.append(lathe(name + "_tray", [(0.0, -0.61), (0.03, -0.6), (0.05, -0.575), (0.115, -0.565), (0.12, -0.545), (0.0, -0.545)], segments=6,
                     loc=(0, ly, lz), smooth_angle=20))
    lan.append(sphere(name + "_drop", 0.022, loc=(0, ly, lz - 0.62), segments=6, rings=4))
    for k in range(6):
        ak = radians(k * 60)
        # CLEARANCE RULE: square-set glazing bars (faces square to the arris and along it: 30 / 60 degrees to the two
        # glowing panes); the three-sided bars had a face lying 1.4 mm over one pane and 2.3 mm under the other
        lan.append(sweep(name + "_bar", [(0.112 * cos(ak), ly + 0.112 * sin(ak), lz - 0.55), (0.165 * cos(ak), ly + 0.165 * sin(ak), lz - 0.225)],
                         _sect(0.009, 0.013), up=(-sin(ak), cos(ak), 0), smooth_path=False))
    for o in lan:
        assign(_fx(o), iron)
    return join(lan, name), _hex_glow(name, 0.105, lz - 0.55, 0.158, lz - 0.225, 0.0, ly, glass, flame)


def _clock_lamp(name, reach, drop, k, iron, glass, flame, brass):
    """Great lantern (scale k) under a heavy wrought-iron gallows bracket. A 45 mm back plate bolted to the parapet
    face (y = 0, facing -Y) carries a 65 mm king post that rises above the coping; the 75 x 60 mm arm at z = 0.25
    reaches `reach` out, held by a raking tie from the post head and by a quadrant brace springing from a cast shoe
    on the cornice fascia, with a scroll in each angle. The lantern hangs `drop` below the arm on a 50 mm hanger bar
    with a swivel. Returns (ironwork, glazing, lamp centre z, gilt brass fittings)."""
    arm_s, tie_s, brace_s, thin = _sect(0.075, 0.06), _sect(0.05, 0.04), _sect(0.06, 0.045), _sect(0.045, 0.032)
    yp = -0.078                                            # king post axis, clear of the coping nose
    zt = 1.3                                               # post head
    ir = [box(name + "_plate", (0.22, 0.045, 0.72), loc=(0, -0.0225, 0.08), bevel=0.008, segments=1),
          box(name + "_shoe", (0.18, 0.04, 0.3), loc=(0, -0.19, -0.44), bevel=0.008, segments=1),
          box(name + "_post", (0.065, 0.065, zt + 0.28), loc=(0, yp, (zt - 0.28) / 2), bevel=0.006, segments=1),
          box(name + "_post_cap", (0.1, 0.1, 0.05), loc=(0, yp, zt + 0.02), bevel=0.01, segments=1),
          sweep(name + "_arm", [(0, yp, 0.25), (0, -reach * 0.5, 0.275), (0, -reach - 0.16, 0.25)], arm_s, up=(1, 0, 0), smooth_path=False),
          sweep(name + "_tie", bezier_path((0, yp, zt - 0.08), (0, -reach * 0.3, zt - 0.5), (0, -reach * 0.62, 0.52), (0, -reach * 0.9, 0.3), n=8),
                tie_s, up=(1, 0, 0), smooth_path=False),
          sweep(name + "_brace", bezier_path((0, -0.2, -0.5), (0, -reach * 0.42, -0.5), (0, -reach * 0.72, -0.14), (0, -reach * 0.74, 0.22), n=9),
                brace_s, up=(1, 0, 0), smooth_path=False),
          sweep(name + "_scroll", spiral_path(0.035, 0.2, 1.4, n=16, a0_deg=20, plane="YZ", center=(0, -0.42, 0.56)), thin, up=(1, 0, 0),
                smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t),
          sweep(name + "_scroll_b", spiral_path(0.03, 0.15, 1.3, n=14, a0_deg=200, plane="YZ", center=(0, -0.42, 0.04)), thin, up=(1, 0, 0),
                smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t),
          sweep(name + "_curl", spiral_path(0.03, 0.095, 1.2, n=10, a0_deg=-90, plane="YZ", center=(0, -reach - 0.16, 0.345)), thin,
                up=(1, 0, 0), smooth_path=False),
          box(name + "_collar_a", (0.085, 0.1, 0.1), loc=(0, -reach * 0.74, 0.25), bevel=0.006, segments=1),
          box(name + "_collar_b", (0.085, 0.11, 0.11), loc=(0, -reach, 0.25), bevel=0.006, segments=1),
          cylinder(name + "_rod", 0.025, drop, loc=(0, -reach, 0.25 - drop / 2), verts=8, cap=False),
          cylinder(name + "_rod_boss", 0.04, 0.06, loc=(0, -reach, 0.25 - drop * 0.3), verts=8)]
    for zz in (0.36, -0.2):
        for sx in (-1, 1):
            ir.append(cylinder(name + "_fixing", 0.02, 0.03, loc=(sx * 0.075, -0.05, zz), rot=(90, 0, 0), verts=6))
    for sx in (-1, 1):
        ir.append(cylinder(name + "_fixing", 0.018, 0.03, loc=(sx * 0.06, -0.215, -0.33), rot=(90, 0, 0), verts=6))
    ly, lz = -reach, 0.25 - drop + 0.045 * k

    def P(pr):
        return [(r * k, z * k) for r, z in pr]

    ir.append(lathe(name + "_roof", P([(0.185, -0.23), (0.19, -0.215), (0.1, -0.12), (0.05, -0.1), (0.035, -0.06), (0.02, -0.045), (0.0, -0.045)]),
                    segments=6, loc=(0, ly, lz), smooth_angle=20))
    ir.append(lathe(name + "_tray", P([(0.0, -0.61), (0.03, -0.6), (0.05, -0.575), (0.115, -0.565), (0.12, -0.545), (0.0, -0.545)]), segments=6,
                    loc=(0, ly, lz), smooth_angle=20))
    for i in range(6):
        ak = radians(i * 60)
        # CLEARANCE RULE: bars set square to the arris (see _lantern); they lay 14 degrees off a glowing pane, 0.8 mm over it
        ir.append(sweep(name + "_bar", [(0.112 * k * cos(ak), ly + 0.112 * k * sin(ak), lz - 0.55 * k),
                                        (0.165 * k * cos(ak), ly + 0.165 * k * sin(ak), lz - 0.225 * k)], _sect(0.0198, 0.0198),
                        up=(-sin(ak), cos(ak), 0), smooth_path=False))
    for o in ir:
        assign(_fx(o), iron)
    # gilt fittings catch the lamp light, so the bracket reads against the night sky from the plaza
    br = [sphere(name + "_post_ball", 0.062, loc=(0, yp, zt + 0.1), segments=8, rings=5),
          cone(name + "_post_spike", r1=0.022, r2=0.003, depth=0.2, loc=(0, yp, zt + 0.25), verts=6),
          sphere(name + "_arm_ball", 0.05, loc=(0, -reach - 0.16, 0.345), segments=8, rings=5),
          sphere(name + "_swivel", 0.058, loc=(0, -reach, 0.25 - drop * 0.55), segments=8, rings=5),
          cylinder(name + "_sleeve", 0.036, 0.16, loc=(0, -reach, lz - 0.02 * k), verts=8),
          sphere(name + "_drop", 0.03 * k, loc=(0, ly, lz - 0.625 * k), segments=6, rings=4)]
    for o in br:
        assign(_fx(o), brass)
    gl = _hex_glow(name, 0.105 * k, lz - 0.55 * k, 0.158 * k, lz - 0.225 * k, 0.0, ly, glass, flame)
    return join(ir, name), gl, lz - 0.3875 * k, join(br, name + "_gilt")


def _standard(name, h, iron, glass, flame, kb=1.0):
    """Cast-iron lamp standard: moulded octagonal base (foot, torus, panelled die with raised ribs, cap and necking;
    scale kb), tapering shaft with collars, ladder bar and a hexagonal lantern. Origin = foot, lamp centre at
    (0, 0, h). Returns (ironwork, glazing)."""
    def B(pr):
        return [(r * kb, z * kb) for r, z in pr]

    zb = 1.0 * kb
    ir = [lathe(name + "_base", B([(0.2, 0.0), (0.2, 0.06), (0.185, 0.078), (0.158, 0.085), (0.176, 0.115), (0.158, 0.145), (0.136, 0.155),
                                   (0.136, 0.5), (0.156, 0.52), (0.162, 0.56), (0.146, 0.59), (0.1, 0.66), (0.074, 0.8), (0.094, 0.82),
                                   (0.094, 0.86), (0.062, 0.9), (0.05, 1.0)]), segments=8, rot=(0, 0, 22.5), smooth_angle=20),
          cylinder(name + "_shaft", 0.05, h - 0.4 - zb, loc=(0, 0, (h - 0.4 + zb) / 2), verts=8, cap=False, radius2=0.034),
          cylinder(name + "_collar", 0.064, 0.05, loc=(0, 0, zb + (h - 1.0 - zb) * 0.42), verts=8),
          cylinder(name + "_collar2", 0.055, 0.05, loc=(0, 0, h - 0.7), verts=8),
          cylinder(name + "_ladder_bar", 0.013, 0.6, loc=(0, 0, h - 0.62), rot=(0, 90, 0), verts=6),
          lathe(name + "_cup", [(0.034, h - 0.42), (0.06, h - 0.36), (0.135, h - 0.3), (0.14, h - 0.27), (0.0, h - 0.27)], segments=6, smooth_angle=20),
          lathe(name + "_roof", [(0.235, h + 0.15), (0.24, h + 0.17), (0.13, h + 0.3), (0.06, h + 0.33), (0.045, h + 0.4), (0.0, h + 0.44)],
                segments=6, smooth_angle=20),
          sphere(name + "_knob", 0.04, loc=(0, 0, h + 0.46), segments=6, rings=4),
          cone(name + "_spike", r1=0.014, r2=0.002, depth=0.16, loc=(0, 0, h + 0.57), verts=4)]
    for k in range(4):                                     # raised ribs on the cardinal faces of the die
        ak = radians(k * 90)
        ir.append(box(name + "_rib", (0.05 * kb, 0.05 * kb, 0.27 * kb), loc=(0.128 * kb * cos(ak), 0.128 * kb * sin(ak), 0.33 * kb),
                      rot=(0, 0, k * 90 + 90)))
    for sx in (-1, 1):
        ir.append(sphere(name + "_bar_end", 0.024, loc=(sx * 0.3, 0, h - 0.62), segments=6, rings=4))
    for k in range(6):
        ak = radians(k * 60)
        # CLEARANCE RULE: bars set square to the arris (see _lantern)
        ir.append(sweep(name + "_bar", [(0.132 * cos(ak), 0.132 * sin(ak), h - 0.28), (0.21 * cos(ak), 0.21 * sin(ak), h + 0.16)],
                        _sect(0.009, 0.013), up=(-sin(ak), cos(ak), 0), smooth_path=False))
    for o in ir:
        assign(_fx(o), iron)
    return join(ir, name), _hex_glow(name, 0.125, h - 0.275, 0.2, h + 0.155, 0.0, 0.0, glass, flame)


K_PORTAL, K_WING, K_CLOCK = 0.9, 1.3, 0.6                 # light strengths as multiples of lantern_intensity
LAMP = 0.4                                                 # glow of the lamp-lit cores and dials per unit of dial_glow (0.75 clipped the dials to white)
AMB = 0.55                                                  # amber emissive glow per unit of window_glow
SW = 0.1                                                   # vertex column pitch of the facades = width of a run-off streak


def build(p, rng):
    W, D, TH = p.width, p.depth - 1.5, p.tower_height      # D = back wall plane (buttresses and lanterns stay inside depth)
    VANE = 1.36                                            # vane above the spire tip
    SH = min(p.spire_height, 25.95 - VANE - TH)            # nothing rises above 26 m
    kind = p.arch
    seed = int(p.seed)
    moss = p.moss

    # ---- materials (10)
    stone = mat("stone_smooth", p.stone_tint)
    if p.wall_finish == "brick":
        wallm = mat("brick_red")
    elif p.wall_finish == "render":
        wallm = mat("plaster_lime", p.wall_tint)
    else:
        wallm = mat("ashlar_limestone")
    roofm = mat({"clay": "roof_tile_clay", "slate": "roof_slate", "shingle": "wood_shingle"}[p.roof])
    g_a = mat("glass_stained", p.glass_a, glow=p.window_glow)
    g_b = mat("emissive", p.glass_b, glow=p.window_glow * AMB)   # lamp-lit amber glass: window cores, lantern panes, belfry
    g_bel = mat("emissive", p.glass_b, glow=p.window_glow * AMB * 0.3)   # the bell chamber behind the louvres: dim, seen in wide bands
    g_c = mat("glass_stained", p.glass_c, glow=p.window_glow)
    dialm = mat("emissive", "#ffa448", glow=p.dial_glow * LAMP)   # lamp-lit opal glass: dials, window / dormer cores, flames
    iron = mat("iron_wrought")
    wood = mat("wood_oak", p.door_tint)
    brass = mat("brass")
    GM = (g_a, g_b, g_c, dialm, iron)

    groups = {"wall": [], "gable": [], "butt": [], "turret": [], "twall": [], "stone": [], "roof": [], "iron": [], "deck": [], "wood": [],
              "brass": [], "glass": [], "step": [], "patch": [], "pave": []}
    mats = {"wall": wallm, "gable": wallm, "butt": wallm, "turret": wallm, "twall": wallm, "stone": stone, "roof": roofm, "iron": iron,
            "deck": iron, "wood": wood, "brass": brass, "patch": roofm, "pave": stone}

    def A(o, g, m=None):
        assign(o, m if m is not None else mats[g])
        groups[g].append(o)
        return o

    # ---- main dimensions
    TW = clamp(5.6 + (W - 26.0) * 0.08, 5.2, 6.1)          # tower is square
    YH = 2.4                                               # hall front wall plane; the tower stands forward of it
    ZP, WT = 1.0, 0.5                                      # floor / plinth top, wall thickness
    ZE = clamp(TH - 9.2, 6.6, 7.6)                         # hall eaves: a tall single storey hall
    RT = 1.55                                              # turret radius
    t_l, t_r = p.turret in ("left", "both"), p.turret in ("right", "both")
    xl = -W / 2 + (RT + 0.52 if t_l else 1.2)
    xr = W / 2 - (RT + 0.52 if t_r else 1.2)
    Dh, ymid = D - YH, (YH + D) / 2
    TC = TW / 2                                            # tower centre y
    # tower stages
    ZS1 = ZE + 0.3                                         # frieze over the base stage
    ZC0 = TH - 0.9                                         # corbel table
    avail = ZC0 - ZS1
    kc = min(3.3, avail * 0.5)
    kb = min(3.2, avail - kc)
    ZK0 = ZC0 - kc                                         # clock stage
    ZB0 = ZK0 - kb                                         # belfry stage
    midh = ZB0 - ZS1
    # hall roof: 48 deg, truncated to a flat deck (or eased to >= 40 deg) so the ridge stays under the dials
    ov, ZEr = 0.35, ZE + 0.1
    hs = Dh / 2 + ov
    pitch = radians(48.0)
    rise = hs * tan(pitch)
    flat_half = 0.0
    rmax = ZK0 + 0.7 - ZEr
    if rise > rmax:
        flat_half = hs - rmax / tan(pitch)
        if flat_half < 0.75 and atan(rmax / hs) >= radians(40.0):
            pitch, flat_half = atan(rmax / hs), 0.0
        rise = rmax
    ZR = ZEr + rise
    tp = tan(pitch)
    cp_, sp_ = cos(pitch), sin(pitch)
    pdeg = degrees(pitch)
    y_e = YH - ov
    y_top = y_e + rise / tp                                # top of the front slope
    y_topb = D + ov - rise / tp
    slope_len = rise / sp_

    def zroof(y):
        return min(ZR, ZEr + (y - y_e) * tp, ZEr + (D + ov - y) * tp)

    def wall_panel(name, face, outer, holes, thick=WT, du=SW, dz=0.4, grp="wall", zs=()):
        o = shape_extrude(name, outer, holes, depth=thick, loc=(0, thick, 0), rot=(90, 0, 0))
        face.put(o)
        _strip(o, (-face.nx, -face.ny, 0.0))
        _grid_cut(o, du if abs(face.ux) > 0.5 else None, du if abs(face.uy) > 0.5 else None, dz, dissolve=True, zs=zs)
        A(o, grp)
        uv_box(o, offset=None)
        return o

    def rect(u0, u1, z0, z1):
        return [(u0, z0), (u1, z0), (u1, z1), (u0, z1)]

    def G(o, m):
        assign(_const_uv(o), m)
        groups["glass"].append(o)
        return o

    # ======================================================================================= TOWER
    tf = {"f": _Face(0, 0, 0), "r": _Face(TW / 2, TC, 90), "b": _Face(0, TW, 180), "l": _Face(-TW / 2, TC, -90)}
    Wd = 2.6                                               # portal clear width
    Hp = clamp(ZE - 2.75, 3.9, 4.8)                        # portal height to the apex
    Hd = min(3.1, Hp * 0.68)                               # door leaf height
    p_rise, p_spring = _arch_dims(Wd, Hp, kind)
    p_spring += ZP
    bw, bh = 0.8, kb - 0.9                                 # belfry lights
    bx = 0.72
    zbs = ZB0 + 0.45
    bel_holes = [_arch(bw, bh, kind, 0.0, 8, zbs, sx * bx) for sx in (-1, 1)]
    mid_w, mid_h = 0.56, midh - 0.75
    z_hid = zroof(TW)                                      # the hall roof buries the back of the tower up to here
    back_bel = z_hid < zbs + 0.3
    gap_o = (ZS1 - 0.27) - (ZP + Hp + 0.62)                # wall between the portal hood and the frieze
    r_oc = min(0.32, gap_o / 2 - 0.26)
    z_oc = ZP + Hp + 0.62 + gap_o / 2
    for key, f in tf.items():
        holes = [] if (key == "b" and not back_bel) else list(bel_holes)
        z_lo = 0.0
        if key == "f":
            holes.append(_arch(Wd, Hp, kind, 0.44, 10, ZP))
            if mid_h >= 0.6:
                holes.append(_arch(mid_w, mid_h, kind, 0.0, 6, ZS1 + 0.5))
            if r_oc >= 0.2:
                holes.append(_circle(r_oc, 16, 0.0, z_oc))
        elif key == "b":
            z_lo = max(0.0, min(z_hid - 0.5, zbs - 0.4) if back_bel else z_hid - 0.5)
        wall_panel("tower_wall_" + key, f, rect(-TW / 2, TW / 2, z_lo, ZC0 + 0.35), holes, 0.6, SW if key != "b" else 0.3, 0.42, "twall",
                   zs=[z_ for z_ in (ZS1 - 0.27, ZB0 - 0.1, ZK0 - 0.1, ZC0, zbs - 0.14) if z_ > z_lo + 0.1])

    # ---- clasping corner piers with weathered set-offs, plinths and quoins
    q_mats = []
    qr = rng.fork("quoins")

    def quoins(cx, cy, sx, sy, z0, z1):
        z, i = z0, 0
        while z + 0.3 <= z1 + 1e-6:
            lx, ly = (0.56, 0.3) if i % 2 == 0 else (0.3, 0.56)
            q_mats.append(trs((cx - sx * (lx / 2 - 0.028) + qr.jitter(0, 0.004), cy - sy * (ly / 2 - 0.028) + qr.jitter(0, 0.004), z + 0.15),
                              (0, 0, 0 if i % 2 == 0 else 90), (1.0, 1.0, qr.u(0.93, 0.99))))
            z += 0.3
            i += 1

    pr = rng.fork("plinth")
    blk_mats = []                                          # instanced plinth blocks
    courses = [(0.0, 0.34), (0.34, 0.64), (0.64, 0.9)]
    PJ = 0.15

    def course_blocks(poly, closed=True, proj=1.0):
        P = [Vector(q) for q in poly]
        n = len(P)
        for ci, (z0, z1) in enumerate(courses):
            for i in range(n if closed else n - 1):
                a, b = P[i], P[(i + 1) % n]
                d = b - a
                ln_seg = d.length
                d = d / ln_seg
                yaw = degrees(atan2(d.y, d.x))
                s0, s1 = 0.0, ln_seg
                if closed or i > 0:
                    dp = (a - P[i - 1]).normalized()
                    if dp.x * d.y - dp.y * d.x > 1e-6 and (i + ci) % 2 == 1:
                        s0 = -PJ * proj
                if closed or i < n - 2:
                    dn = (P[(i + 2) % n] - b).normalized()
                    if d.x * dn.y - d.y * dn.x > 1e-6 and (i + 1 + ci) % 2 == 0:
                        s1 = ln_seg + PJ * proj
                s = s0
                first = True
                while s < s1 - 1e-6:
                    ln = pr.u(0.8, 1.2) * (0.55 if (first and ci % 2 == 1) else 1.0)
                    if s1 - (s + ln) < 0.5:
                        ln = s1 - s
                    c = a + d * (s + ln / 2)
                    blk_mats.append(trs((c.x, c.y, z0), (0, 0, yaw), (ln - 0.004, proj * pr.u(0.96, 1.05), (z1 - z0) / 0.3)))
                    s += ln
                    first = False

    stages = [(1.15, 0.36, 0.0, ZS1 - 0.25), (1.0, 0.25, ZS1 + 0.3, ZK0 - 0.15), (0.85, 0.14, ZK0 + 0.3, ZC0 + 0.32)]
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        prev = None
        for si, (s, e, z0, z1) in enumerate(stages):
            xo, xi = sx * (TW / 2 + e), sx * (TW / 2 + e - s)
            yo, yi = TC + sy * (TW / 2 + e), TC + sy * (TW / 2 + e - s)
            rc = (min(xo, xi), max(xo, xi), min(yo, yi), max(yo, yi))
            if sy > 0 and si == 0:                         # inside the hall
                prev = rc + (z1,)
                continue
            b = box("pier", (s, s, z1 - z0), loc=((xo + xi) / 2, (yo + yi) / 2, (z0 + z1) / 2))
            apply_transform(b)
            _grid_cut(b, 0.19, 0.19, 0.55)
            A(b, "twall")
            uv_box(b, offset=None)
            if si > 0:                                     # stone weathering between the stages
                g = 0.04
                wz0 = prev[4]
                A(_frustum("pier_weathering", (prev[0] - g, prev[1] + g, prev[2] - g, prev[3] + g, wz0 + 0.06), rc + (z0,)), "stone")
                A(box("pier_drip", (prev[1] - prev[0] + 2 * g, prev[3] - prev[2] + 2 * g, 0.1),
                      loc=((prev[0] + prev[1]) / 2, (prev[2] + prev[3]) / 2, wz0 + 0.01), bevel=0.015, segments=1), "stone")
            else:                                          # plinth of coursed blocks
                g = 0.13
                course_blocks([(rc[0], rc[2]), (rc[1], rc[2]), (rc[1], rc[3]), (rc[0], rc[3])], True, g / PJ)
                A(_frustum("pier_plinth_cap", (rc[0] - g, rc[1] + g, rc[2] - g, rc[3] + g, 0.9), rc + (ZP + 0.02,)), "stone")
            zq0 = (ZP + 0.05) if si == 0 else z0 + 0.02
            if sy > 0:
                zq0 = max(zq0, z_hid - 1.2)
            quoins(xo, yo, sx, sy, zq0, z1 - 0.02)
            prev = rc + (z1,)

    # ---- string courses, frieze, corbel table, cornice
    sq = [(-TW / 2, 0.0), (TW / 2, 0.0), (TW / 2, TW), (-TW / 2, TW)]
    z = ZS1
    fr_prof = [(-0.02, z - 0.27), (0.05, z - 0.27), (0.09, z - 0.22), (0.09, z + 0.2), (0.13, z + 0.22), (0.15, z + 0.27),
               (0.03, z + 0.34), (-0.02, z + 0.34)]
    A(densify(_loop("tower_frieze", sq, fr_prof), 1.4), "stone")

    def string_prof(z):
        return [(-0.02, z - 0.1), (0.06, z - 0.1), (0.1, z - 0.05), (0.1, z + 0.03), (0.02, z + 0.1), (-0.02, z + 0.1)]

    A(_loop("tower_string_k", sq, string_prof(ZK0)), "stone")
    if midh > 0.6:
        A(_loop("tower_string_b", sq, string_prof(ZB0)), "stone")
    a = ZC0 + 0.3
    co_prof = [(-0.05, a), (0.16, a), (0.2, a + 0.1), (0.3, a + 0.16), (0.4, a + 0.3), (0.43, a + 0.3), (0.43, a + 0.42),
               (0.47, a + 0.46), (0.47, a + 0.6), (-0.05, a + 0.6)]
    A(_loop("tower_cornice", sq, co_prof), "stone")
    corbel = _prism("corbel", [(-0.02, 0.0), (0.05, 0.0), (0.09, 0.04), (0.2, 0.2), (0.2, 0.31), (-0.02, 0.31)], -0.085, 0.085)
    cm = []
    ncb = int((TW - 1.5) / 0.44)
    for f in tf.values():
        for i in range(ncb + 1):
            cm.append(f.M((i - ncb / 2) * (TW - 1.5) / ncb, 0.0, ZC0))

    # ---- tower flat, pierced parapet, corner pinnacles
    A(box("tower_flat", (TW + 0.7, TW + 0.7, 0.08), loc=(0, TC, TH)), "stone")
    Lp = TW - 0.2
    nsl = max(3, int((Lp - 0.5) / 0.52))
    slots = [_arch(0.26, 0.5, kind, 0.0, 4, 0.13, (i - (nsl - 1) / 2) * (Lp - 0.5) / nsl) for i in range(nsl)]
    par = shape_extrude("parapet", rect(-Lp / 2, Lp / 2, 0.0, 0.78), slots, depth=0.17, loc=(0, -0.13, 0), rot=(90, 0, 0))
    _fx(par)
    assign(par, stone)
    cop = _prism("parapet_coping", [(0.34, 0.76), (0.34, 0.81), (0.215, 0.89), (0.09, 0.81), (0.09, 0.76)], -Lp / 2, Lp / 2)
    assign(cop, stone)
    par = join([par, cop], "parapet")
    A(instances_to_mesh("parapets", par, [f.M(0, 0, TH) for f in tf.values()], keep_base=False), "stone")
    pin = [box("pin_shaft", (0.5, 0.5, 1.15), loc=(0, 0, 0.575), bevel=0.02, segments=1),
           box("pin_neck", (0.6, 0.6, 0.09), loc=(0, 0, 1.19), bevel=0.02, segments=1),
           cone("pin_spike", r1=0.36, r2=0.03, depth=1.15, loc=(0, 0, 1.8), rot=(0, 0, 45), verts=4),
           sphere("pin_ball", 0.075, loc=(0, 0, 2.4), segments=8, rings=5)]
    for k in range(4):                                     # little gablets on the shaft faces
        o = _prism("pin_gab", [(0.25, 0.8), (0.3, 0.8), (0.3, 1.0), (0.25, 1.23)], -0.2, 0.2)
        place(o, rot=(0, 0, 90 * k))
        pin.append(o)
    for o in pin:
        _fx(o)
        assign(o, stone)
    flat(pin[2])
    pin = join(pin, "pinnacle")
    pm = [trs((sx * (TW / 2 + 0.12), TC + sy * (TW / 2 + 0.12), TH), (0, 0, 0 if sy < 0 else 180)) for sx in (-1, 1) for sy in (-1, 1)]
    A(instances_to_mesh("pinnacles", pin, pm, keep_base=False), "stone")

    # ---- belfry: open oak louvres with the dimly lamp-lit bell chamber glowing between the blades
    lr = rng.fork("louvres")
    bel = []
    b_spring = _arch_dims(bw, bh, kind)[1]
    for sx in (-1, 1):
        o = shape_extrude("bel_surround", _band(bw, bh, kind, 0.0, 0.14, 6, cx=sx * bx), depth=0.12, loc=(0, 0.04, 0), rot=(90, 0, 0), bevel=0.01)
        assign(_fx(o), stone)
        bel.append(o)
        o = shape_extrude("bel_hood", _band(bw, bh, kind, 0.14, 0.23, 6, cx=sx * bx, clip=b_spring - 0.05), depth=0.17,
                          loc=(0, 0.04, 0), rot=(90, 0, 0))
        assign(_fx(o), stone)
        bel.append(o)
        # weathered oak blades (36 mm, 40 deg, shedding outwards, 0.34 m apart) in front of the dimly lamp-lit bell
        # chamber, with the bell frame (a post and its headstock) dark against it. The warm bands between the blades
        # are 12 - 20 cm high from the square and dim: nothing bright here is narrower than a few pixels from across
        # the plaza. (The first belfry showed the chamber as 28 mm lines of full window glow between hit-and-miss
        # boards: under a pixel wide at 30 m, each was drawn or not from one step to the next and the belfry and its
        # streak on the wet paving flashed.)
        pitch_l, zz = 0.34, 0.2
        while zz < bh - 0.1:
            o = box("louvre", (bw + 0.12, 0.3, 0.036), loc=(sx * bx, 0.14, zz), rot=(40 + lr.jitter(0, 2.0), 0, lr.jitter(0, 0.4)))
            uv_grain(_fx(o), "X")
            assign(o, wood)
            bel.append(o)
            zz += pitch_l
        o = box("bel_post", (0.1, 0.02, bh + 0.02), loc=(sx * bx, 0.27, bh / 2))
        uv_grain(_fx(o), "Z")
        assign(o, wood)
        bel.append(o)
        o = box("bel_headstock", (bw + 0.12, 0.02, 0.16), loc=(sx * bx, 0.27, bh * 0.62))
        uv_grain(_fx(o), "X")
        assign(o, wood)
        bel.append(o)
        o = _const_uv(_flat("bel_glow", [rect(sx * bx - bw / 2 - 0.05, sx * bx + bw / 2 + 0.05, -0.02, bh + 0.05)], 0.31))
        assign(o, g_bel)
        bel.append(o)
    o = box("bel_sill", (2 * bx + bw + 0.6, 0.34, 0.14), loc=(0, -0.03, -0.07), bevel=0.02, segments=1)
    assign(o, stone)
    bel.append(o)
    o = box("bel_impost", (0.62, 0.12, 0.1), loc=(0, -0.04, b_spring), bevel=0.015, segments=1)
    assign(o, stone)
    bel.append(o)
    bel = join(bel, "belfry")
    bel_faces = [f for key, f in tf.items() if key != "b" or back_bel]
    groups["stone"].append(instances_to_mesh("belfries", bel, [f.M(0, 0, zbs) for f in bel_faces], uv_jitter=False, keep_base=False))

    # ---- clock dials
    # hands, numerals, ticks and rings: matt non-metal near-black paint at every level (wrought iron, metal, took a
    # highlight from the clock lanterns and read pale brown on the lit dial)
    d_ink = mat("wood_painted", "#0c0c0e", rough=1.8, wear=0.0, special=0.0)
    Rd = min(1.25, kc / 2 - 0.36, TW / 2 - 1.25)
    zd = ZK0 + kc / 2 + 0.03
    dl = []
    o = lathe("dial_surround", [(Rd + 0.3, 0.0), (Rd + 0.3, 0.07), (Rd + 0.26, 0.13), (Rd + 0.18, 0.13), (Rd + 0.14, 0.2), (Rd + 0.06, 0.2),
                                (Rd + 0.01, 0.15), (Rd + 0.01, 0.03)], segments=32, rot=(90, 0, 0))
    assign(_fx(o), stone)
    dl.append(o)
    d_sur = o
    o = _flat("dial_face", [_circle(Rd + 0.02, 32)], -0.05)
    assign(o, dialm)
    dl.append(o)
    d_face = o
    marks = []
    for r0, r1 in ((Rd * 0.955, Rd * 0.99), (Rd * 0.7, Rd * 0.725), (Rd * 0.2, Rd * 0.22)):
        for k in range(32):
            a0_, a1_ = 2 * pi * k / 32, 2 * pi * (k + 1) / 32
            marks.append([(r0 * cos(a0_), r0 * sin(a0_)), (r1 * cos(a0_), r1 * sin(a0_)), (r1 * cos(a1_), r1 * sin(a1_)), (r0 * cos(a1_), r0 * sin(a1_))])
    for i in range(60):
        a_ = radians(i * 6)
        r0, r1 = Rd * (0.865 if i % 5 == 0 else 0.9), Rd * 0.955
        marks.append(_bar(r0 * sin(a_), r0 * cos(a_), r1 * sin(a_), r1 * cos(a_), 0.04 if i % 5 == 0 else 0.018))
    o = _flat("dial_marks", marks, -0.062)
    assign(o, d_ink)
    dl.append(o)
    d_marks, d_nums, d_hands = o, [], []
    rn = Rd * 0.8
    for i, s in enumerate(["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]):
        th = i * 30.0
        o = text("numeral", s, size=Rd * 0.19, depth=0.0, loc=(rn * sin(radians(th)), -0.064, rn * cos(radians(th))), rot=(90, th, 0),
                 font="roman", spacing=0.92)
        assign(_fx(o), d_ink)
        dl.append(o)
        d_nums.append(o)
    hrs = float(p.time) % 12.0
    mins = (hrs * 60.0) % 60.0
    lh_, lm_ = Rd * 0.56, Rd * 0.9
    hour = [(-0.035, -0.2 * lh_), (0.035, -0.2 * lh_), (0.03, 0.0), (0.045, 0.45 * lh_), (0.1, 0.6 * lh_), (0.06, 0.74 * lh_), (0.0, lh_),
            (-0.06, 0.74 * lh_), (-0.1, 0.6 * lh_), (-0.045, 0.45 * lh_), (-0.03, 0.0)]
    minute = [(-0.03, -0.24 * lm_), (0.03, -0.24 * lm_), (0.024, 0.0), (0.032, 0.6 * lm_), (0.055, 0.72 * lm_), (0.0, lm_), (-0.055, 0.72 * lm_),
              (-0.032, 0.6 * lm_), (-0.024, 0.0)]
    for nm, outline, ang, yy in (("hand_hour", hour, hrs * 30.0, -0.085), ("hand_minute", minute, mins * 6.0, -0.105)):
        o = shape_extrude(nm, outline, depth=0.014, loc=(0, yy, 0), rot=(90, ang, 0))
        assign(_fx(o), d_ink)
        dl.append(o)
        d_hands.append(o)
        o = shape_extrude(nm + "_cw", _circle(0.07, 10, 0.0, -0.2 * (lh_ if "hour" in nm else lm_)), depth=0.014, loc=(0, yy, 0), rot=(90, ang, 0))
        assign(_fx(o), d_ink)
        dl.append(o)
        d_hands.append(o)
    o = lathe("dial_boss", [(0.085, 0.0), (0.085, 0.085), (0.06, 0.105), (0.03, 0.125), (0.0, 0.13)], segments=12, rot=(90, 0, 0))
    assign(_fx(o), brass)
    dl.append(o)
    d_boss = o

    # The dial is the identity of the tower: what stands for its parts at lod 1 / lod 2 is modelled here, from the
    # same numbers (tk.lod_standin: the full model is not touched, the reducer never simplifies these further).
    #   lod 1: the hands as flat fronts, the three rings and the 12 hour ticks (minute ticks go); face, numerals,
    #          boss and surround as modelled
    #   lod 2: a 16-sided lit disc, one bold ring, 12 bold hour marks (no numerals), bold hands of the same lengths
    #          and angles (hour 0.17-0.21 m, minute 0.12-0.13 m wide: 3 px or more at 44 m), a flat boss, a plain
    #          round surround. Every layer keeps its plane, so the 12 mm / 37 mm clearances of the full dial hold.
    d_mats = [f.M(0, 0, zd) for f in tf.values()]

    def d_inst(nm, parts):
        return instances_to_mesh(nm, join(parts, nm + "_one"), d_mats, uv_jitter=False, keep_base=False)

    def d_turn(outline, ang):                   # a hand outline turned clockwise as rot=(90, ang, 0) does
        c_, s_ = cos(radians(ang)), sin(radians(ang))
        return [(x * c_ + y * s_, -x * s_ + y * c_) for x, y in outline]

    def d_marks_mid():
        o = _flat("dial_marks_mid", marks[:96] + [m for i, m in enumerate(marks[96:]) if i % 5 == 0], -0.062)
        assign(o, d_ink)
        return d_inst("dials_marks_mid", [o])

    def d_hands_mid():
        parts = []
        for outline, ang, yy, ln in ((hour, hrs * 30.0, -0.099, lh_), (minute, mins * 6.0, -0.119, lm_)):
            o = _flat("hand_mid", [d_turn(outline, ang), d_turn(_circle(0.07, 10, 0.0, -0.2 * ln), ang)], yy)
            assign(o, d_ink)
            parts.append(o)
        return d_inst("dials_hands_mid", parts)

    def d_far():
        parts = []
        o = _flat("dial_face_far", [_circle((Rd + 0.02) / cos(pi / 16), 16)], -0.05)
        assign(o, dialm)
        parts.append(o)
        polys = []
        for k in range(24):
            a0_, a1_ = 2 * pi * k / 24, 2 * pi * (k + 1) / 24
            r0, r1 = Rd * 0.94, Rd * 0.99
            polys.append([(r0 * cos(a0_), r0 * sin(a0_)), (r1 * cos(a0_), r1 * sin(a0_)), (r1 * cos(a1_), r1 * sin(a1_)), (r0 * cos(a1_), r0 * sin(a1_))])
        for i in range(12):
            a_ = radians(i * 30)
            r0, r1 = Rd * 0.7, Rd * 0.94
            polys.append(_bar(r0 * sin(a_), r0 * cos(a_), r1 * sin(a_), r1 * cos(a_), 0.15 if i % 3 == 0 else 0.1))
        o = _flat("dial_marks_far", polys, -0.062)
        assign(o, d_ink)
        parts.append(o)
        hour_b = [(-0.085, -0.2 * lh_), (0.085, -0.2 * lh_), (0.105, 0.62 * lh_), (0.0, lh_), (-0.105, 0.62 * lh_)]
        minute_b = [(-0.06, -0.24 * lm_), (0.06, -0.24 * lm_), (0.065, 0.8 * lm_), (0.0, lm_), (-0.065, 0.8 * lm_)]
        for outline, ang, yy in ((hour_b, hrs * 30.0, -0.099), (minute_b, mins * 6.0, -0.119)):
            o = _flat("hand_far", [d_turn(outline, ang)], yy)
            assign(o, d_ink)
            parts.append(o)
        o = _flat("dial_boss_far", [_circle(0.09, 8)], -0.13)
        assign(o, brass)
        parts.append(o)
        return d_inst("dials_far", parts)

    def d_sur_far():
        o = lathe("dial_surround_far", [(Rd + 0.3, 0.0), (Rd + 0.28, 0.12), (Rd + 0.1, 0.2), (Rd + 0.01, 0.15), (Rd + 0.01, 0.03)],
                  segments=24, rot=(90, 0, 0))
        assign(_fx(o), stone)
        return d_inst("dials_surround_far", [o])

    lod_standin(d_sur, far=d_sur_far)
    lod_standin([d_face, d_boss] + d_nums, far=d_far)
    lod_standin(d_marks, mid=d_marks_mid, far=())
    lod_standin(d_hands, mid=d_hands_mid, far=())
    dl = join(dl, "dial")
    groups["stone"].append(instances_to_mesh("dials", dl, d_mats, uv_jitter=False, keep_base=False))

    # ---- mid-stage light, oculus over the portal and the gilded inscription on the frieze
    if mid_h >= 0.6:
        sl = _slit("tower_light", mid_w, mid_h, kind, stone, g_b, iron, dialm)
        groups["stone"].append(instances_to_mesh("tower_lights", sl, [trs((0, 0.12, ZS1 + 0.5))], uv_jitter=False, keep_base=False))
    if r_oc >= 0.2:
        A(_fx(lathe("oculus_ring", [(r_oc + 0.2, 0.0), (r_oc + 0.2, 0.05), (r_oc + 0.16, 0.1), (r_oc + 0.1, 0.1), (r_oc + 0.06, 0.15),
                                    (r_oc + 0.01, 0.15), (r_oc - 0.02, 0.1), (r_oc - 0.02, -0.2)], segments=24, loc=(0, 0, z_oc), rot=(90, 0, 0))), "stone")
        A(_fx(shape_extrude("oculus_tracery", _circle(r_oc, 20, 0.0, z_oc), [_foil(4, r_oc * 0.86, 0.0, z_oc, 5, pi / 4)], depth=0.08,
                            loc=(0, 0.24, 0), rot=(90, 0, 0))), "stone")
        G(_flat("oculus_glass", [_circle(r_oc + 0.01, 16, 0.0, z_oc)], 0.27), g_b)
        G(_flat("oculus_eye", [_circle(r_oc * 0.36, 10, 0.0, z_oc)], 0.264), g_c)
    if p.inscription.strip():
        size = 0.3
        o = text("inscription", p.inscription, size=size, depth=0.022, loc=(0, -0.09, ZS1 - 0.02), font="roman", spacing=1.08)
        bmin, bmax = world_bbox([o])
        wtxt = bmax.x - bmin.x
        lim = TW - 2 * (1.15 - 0.36) - 0.5
        if wtxt > lim:
            delete([o])
            o = text("inscription", p.inscription, size=size * lim / wtxt, depth=0.022, loc=(0, -0.09, ZS1 - 0.02), font="roman", spacing=1.08)
        A(_fx(o), "brass")

    # ---- spire: slim bell-cast pyramid, hip tiles, four lucarnes, finial and weathervane
    a0, fl = TW / 2 - 0.8, 0.3
    zb, zt = TH + 0.28, TH + SH
    Hs = zt - zb
    ns = 8 if p.spire == "octagonal" else 4               # square pyramid or octagonal spire
    tn, cs_ = tan(pi / ns), cos(pi / ns)
    A(cylinder("spire_drum", (a0 + 0.06) / cs_, 0.34, loc=(0, TC, TH + 0.15), rot=(0, 0, 180.0 / ns), verts=ns), "stone")
    A(cylinder("spire_soffit", (a0 + fl) / cs_, 0.05, loc=(0, TC, zb - 0.02), rot=(0, 0, 180.0 / ns), verts=ns), "stone")
    ts = [0.0, 0.02, 0.05, 0.09, 0.14, 0.2, 0.32, 0.45, 0.58, 0.7, 0.8, 0.9, 0.985]

    def srad(t):
        return a0 * (1.0 - t) + fl * max(0.0, 1.0 - t / 0.2) ** 2

    nc = 4 if ns == 4 else 2
    sl_len = hypot(a0, Hs)
    hip = cylinder("hip_tile", 0.075, 0.4, verts=6, cap=False, radius2=0.055)
    hip_ms = []
    for k in range(ns):
        ca, sa = cos(2 * pi * k / ns), sin(2 * pi * k / ns)
        vs, fs = [], []
        for t in ts:
            r = srad(t)
            hw = r * tn
            for j in range(nc + 1):
                x, y = -hw + 2 * hw * j / nc, -r
                vs.append((x * ca - y * sa, TC + x * sa + y * ca, zb + t * Hs))
        for i in range(len(ts) - 1):
            for j in range(nc):
                q = i * (nc + 1) + j
                fs.append((q, q + 1, q + nc + 2, q + nc + 1))
        o = mesh_object("spire_face", vs, fs)
        uv_planar(o, u_axis=(ca, sa, 0.0), v_axis=(-sa * a0 / sl_len, ca * a0 / sl_len, Hs / sl_len))
        A(o, "roof")
        hp = []
        for t in ts[:-1] + [0.94]:
            r = srad(t) * 1.005
            hp.append((r * tn * ca + r * sa, TC + r * tn * sa - r * ca, zb + t * Hs + 0.012))
        for mid, dr_, ln in _along(hp, 0.36):
            hip_ms.append(Matrix.Translation(mid) @ Vector((0, 0, 1)).rotation_difference(dr_).to_matrix().to_4x4()
                          @ Matrix.Diagonal((1.0, 1.0, ln / 0.4 * 1.16, 1.0)))
    A(instances_to_mesh("spire_hips", hip, hip_ms, keep_base=False), "roof")
    DM = {"stone": stone, "glass": g_b, "lamp": dialm, "cheek": roofm, "roof": roofm, "iron": iron, "wood": wood}
    t_l_ = 0.21
    w_, hw_, gp = 0.66, 1.0, 63.0
    back = (hw_ + w_ / 2 * tan(radians(gp))) * a0 / Hs + 0.45
    luc = _gablet("lucarne", w_, hw_, gp, back, kind, DM, "lit", shutters=False)
    r = a0 * (1.0 - t_l_) + 0.27
    lm = [trs((r * sin(k * pi / 2), TC - r * cos(k * pi / 2), zb + t_l_ * Hs - 0.2), (0, 0, k * 90)) for k in range(4)]
    groups["stone"].append(instances_to_mesh("lucarnes", luc, lm, uv_jitter=False, keep_base=False))
    A(lathe("spire_cap", [(0.34, zt - 0.95), (0.26, zt - 0.6), (0.16, zt - 0.2), (0.12, zt - 0.08), (0.17, zt - 0.02), (0.17, zt + 0.03),
                          (0.09, zt + 0.1), (0.05, zt + 0.22), (0.0, zt + 0.22)], segments=12, loc=(0, TC, 0)), "deck")
    A(cylinder("vane_rod", 0.022, 0.95, loc=(0, TC, zt + 0.6), verts=8), "iron")
    A(sphere("vane_ball", 0.16, loc=(0, TC, zt + 0.38), segments=14, rings=8), "brass")
    zc_ = zt + 0.63
    A(cylinder("vane_arm_x", 0.012, 1.0, loc=(0, TC, zc_), rot=(0, 90, 0), verts=6), "iron")
    A(cylinder("vane_arm_y", 0.012, 1.0, loc=(0, TC, zc_), rot=(90, 0, 0), verts=6), "iron")
    for ch, dx, dy in (("E", 0.56, 0.0), ("W", -0.56, 0.0), ("S", 0.0, -0.56), ("N", 0.0, 0.56)):
        A(_fx(text("vane_" + ch, ch, size=0.17, depth=0.012, loc=(dx, TC + dy, zc_), font="serif")), "brass")
    zv = zt + 0.86
    va = radians(rng.u(0, 360))
    vane = [cylinder("vane_arrow", 0.013, 1.3, loc=(0, 0, 0), rot=(0, 90, 0), verts=6),
            shape_extrude("vane_head", [(0.6, -0.1), (0.86, 0.0), (0.6, 0.1), (0.65, 0.0)], depth=0.008, loc=(0, 0.004, 0), rot=(90, 0, 0)),
            shape_extrude("vane_tail", [(-0.9, -0.16), (-0.52, -0.02), (-0.52, 0.02), (-0.9, 0.16), (-0.78, 0.0)], depth=0.008, loc=(0, 0.004, 0), rot=(90, 0, 0)),
            cylinder("vane_sleeve", 0.035, 0.12, loc=(0, 0, 0), verts=10)]
    cockerel = [(-0.05, 0.0), (0.05, 0.0), (0.06, 0.12), (0.2, 0.2), (0.3, 0.38), (0.3, 0.55), (0.42, 0.52), (0.34, 0.62), (0.36, 0.72),
                (0.3, 0.78), (0.24, 0.74), (0.2, 0.8), (0.16, 0.7), (0.17, 0.52), (0.05, 0.42), (-0.12, 0.42), (-0.22, 0.6), (-0.3, 0.82),
                (-0.42, 0.9), (-0.5, 0.8), (-0.44, 0.78), (-0.38, 0.6), (-0.46, 0.62), (-0.5, 0.5), (-0.4, 0.42), (-0.34, 0.25),
                (-0.2, 0.14), (-0.06, 0.12)]
    vane.append(shape_extrude("vane_cock", [(x * 0.5, z * 0.5 + 0.01) for x, z in cockerel], depth=0.012, loc=(0, 0.006, 0), rot=(90, 0, 0), bevel=0.002))
    for i, o in enumerate(vane):
        _fx(o)
        place(o, loc=(0, TC, zv), rot=(0, 0, degrees(va)))
        apply_transform(o)
        A(o, "brass" if i in (1, 2, 4) else "iron")
    part(vane, "vane", pivot=(0, TC, zv), motion="spin", axis="Z", amp=25, speed=0.08)

    # ======================================================================================= PORTAL, DOORS, STEPS
    A(_fx(shape_extrude("portal_order1", _band(Wd, Hp, kind, 0.22, 0.44, 10, ZP), depth=0.42, loc=(0, 0.6, 0), rot=(90, 0, 0), bevel=0.012)), "stone")
    A(_fx(shape_extrude("portal_order2", _band(Wd, Hp, kind, 0.0, 0.22, 10, ZP), depth=0.24, loc=(0, 0.6, 0), rot=(90, 0, 0), bevel=0.012)), "stone")
    A(_fx(shape_extrude("portal_hood", _band(Wd, Hp, kind, 0.46, 0.62, 10, ZP, clip=p_spring - 0.04), depth=0.14, loc=(0, 0.0, 0),
                        rot=(90, 0, 0), bevel=0.012)), "stone")
    roll = [(x, 0.085, z) for x, z in _arch(Wd, Hp, kind, 0.345, 10, ZP)[1:-1]]
    A(sweep("portal_roll", roll, 0.065, sides=8, step=0.16), "stone")
    roll2 = [(x, 0.27, z) for x, z in _arch(Wd, Hp, kind, 0.13, 10, ZP)[1:-1]]
    A(sweep("portal_roll2", roll2, 0.055, sides=8, step=0.16), "stone")
    cap_prof = [(0.065, -0.06), (0.075, -0.04), (0.07, 0.0), (0.1, 0.06), (0.125, 0.1), (0.125, 0.14), (0.0, 0.14)]
    base_prof = [(0.0, 0.0), (0.115, 0.0), (0.115, 0.05), (0.09, 0.07), (0.1, 0.1), (0.075, 0.14), (0.065, 0.2)]
    for sx in (-1, 1):
        for (xo_, yo_, rr) in ((Wd / 2 + 0.345, 0.085, 0.062), (Wd / 2 + 0.13, 0.27, 0.052)):
            A(cylinder("nook_shaft", rr, p_spring - ZP - 0.2, loc=(sx * xo_, yo_, (p_spring + ZP) / 2), verts=10), "stone")
            A(lathe("nook_cap", cap_prof, segments=10, loc=(sx * xo_, yo_, p_spring - 0.14)), "stone")
            A(lathe("nook_base", base_prof, segments=10, loc=(sx * xo_, yo_, ZP)), "stone")
        A(stone_block("hood_stop", (0.2, 0.18, 0.22), loc=(sx * (Wd / 2 + 0.54), -0.07, p_spring - 0.1), rng=rng.fork("stop"), detail=0.1), "stone")
    A(box("threshold", (Wd + 0.9, 0.62, 0.2), loc=(0, 0.3, ZP - 0.1 + 0.005), bevel=0.01, segments=1), "step", stone)
    # doors: boarded oak leaves, strap hinges, studs, ring handles
    dr = rng.fork("door")
    nb = 6
    lw = Wd / 2 - 0.012
    bwd = lw / nb
    studs = []
    stud_z = [0.18 + 0.44 * k for k in range(int((Hd - 0.28) / 0.44) + 1)]
    dl_w, dl_h, dl_x, dl_z = 0.3, 0.56, lw * 0.5 + 0.012, Hd * 0.75 - 0.175    # glazed grille light in each leaf, between the straps
    for sx in (-1, 1):
        for i in range(nb):
            xc_ = sx * (0.012 + bwd * (i + 0.5))
            A(plank("door_board", Hd - 0.02, bwd - 0.005, 0.055, loc=(xc_, 0.5 + dr.jitter(0, 0.002), ZP + Hd / 2 + 0.01), axis="Z",
                    bevel=0.004, rng=dr), "wood")
            for zz in stud_z:
                if abs(abs(xc_) - dl_x) < dl_w / 2 + 0.07 and abs(zz - dl_z) < dl_h / 2 + 0.08:
                    continue
                studs.append(trs((xc_, 0.47, ZP + zz), (90, 0, 0)))
        cx_, cz_ = sx * dl_x, ZP + dl_z
        for dx_ in (-1, 1):
            A(box("door_light_stile", (0.05, 0.03, dl_h + 0.1), loc=(cx_ + dx_ * (dl_w / 2 + 0.025), 0.459, cz_), bevel=0.005, segments=1), "wood")
            A(box("door_light_rail", (dl_w, 0.03, 0.05), loc=(cx_, 0.459, cz_ + dx_ * (dl_h / 2 + 0.025)), bevel=0.005, segments=1), "wood")
        G(_flat("door_light", [[(cx_ - dl_w / 2, cz_ - dl_h / 2), (cx_ + dl_w / 2, cz_ - dl_h / 2), (cx_ + dl_w / 2, cz_ + dl_h / 2),
                                (cx_ - dl_w / 2, cz_ + dl_h / 2)]], 0.466), dialm)
        # CLEARANCE RULE: the lit pane 4.5 mm in front of the proudest door board (it was 2.5) and 4.5 mm behind the bars
        for k in (-1, 0, 1):
            A(cylinder("door_light_bar", 0.008, dl_h + 0.02, loc=(cx_ + k * dl_w * 0.25, 0.4535, cz_), verts=6), "iron")
        for k in (-1, 1):
            A(cylinder("door_light_tie", 0.007, dl_w + 0.02, loc=(cx_, 0.452, cz_ + k * dl_h / 6), rot=(0, 90, 0), verts=6), "iron")
        for zz in (0.4, Hd / 2 + 0.05, Hd - 0.4):
            ln = lw * 0.82
            o = shape_extrude("door_strap", [(0, -0.035), (ln * 0.8, -0.03), (ln * 0.86, -0.07), (ln, 0.0), (ln * 0.86, 0.07), (ln * 0.8, 0.03), (0, 0.035)],
                              depth=0.012, loc=(sx * (lw + 0.012), 0.4705, ZP + zz), rot=(90, 0 if sx < 0 else 180, 0), bevel=0.002)
            A(_fx(o), "iron")
            for k in range(4):
                studs.append(trs((sx * (lw + 0.012 - 0.08 - k * ln * 0.22), 0.458, ZP + zz), (90, 0, 0), 0.8))
            A(cylinder("door_pin", 0.022, 0.16, loc=(sx * (lw + 0.02), 0.455, ZP + zz), verts=8), "iron")
        A(box("door_plate", (0.09, 0.008, 0.2), loc=(sx * 0.14, 0.468, ZP + 1.12), bevel=0.003, segments=1), "iron")
        A(torus("door_ring", 0.075, 0.011, loc=(sx * 0.14, 0.452, ZP + 1.06), rot=(90, 0, 0), major_seg=16, minor_seg=6), "brass")
        A(sphere("door_ring_boss", 0.022, loc=(sx * 0.14, 0.46, ZP + 1.135), segments=8, rings=6), "brass")
    A(box("door_astragal", (0.06, 0.03, Hd - 0.02), loc=(0, 0.462, ZP + Hd / 2 + 0.01), bevel=0.006, segments=1), "wood")
    uv_grain(groups["wood"][-1], "Z")
    stud = bolt("stud", 0.017, 0.014, kind="pyramid")
    A(instances_to_mesh("door_studs", stud, studs, keep_base=False), "iron")
    zt_ = ZP + Hd
    A(box("transom", (Wd + 0.04, 0.2, 0.2), loc=(0, 0.5, zt_ + 0.1), bevel=0.02, segments=1), "stone")
    A(box("door_backing", (Wd + 0.04, 0.02, Hd), loc=(0, 0.56, ZP + Hd / 2)), "wood")
    tym = _clip_above(_arch(Wd, Hp, kind, 0.005, 10, ZP), zt_ + 0.19)
    tz = zt_ + 0.2 + (ZP + Hp - zt_ - 0.2) * 0.42
    f_main, f_side = _foil(4, 0.36, 0.0, tz), [_foil(3, 0.17, sx * 0.6, tz - 0.2) for sx in (-1, 1)]
    A(_fx(shape_extrude("tympanum", tym, [f_main] + f_side, depth=0.1, loc=(0, 0.56, 0), rot=(90, 0, 0), bevel=0.006)), "stone")
    # CLEARANCE RULE: the lit glass 5 mm behind the back of the portal order (its rim lay in the plane of that back)
    G(_flat("tym_glass", [tym], 0.605), dialm)
    G(_flat("tym_main", [_circle(0.2, 12, 0.0, tz)], 0.594), g_a)
    G(_flat("tym_side", [_circle(0.085, 10, sx * 0.6, tz - 0.2) for sx in (-1, 1)], 0.594), g_c)
    G(_flat("tym_eye", [_circle(0.12, 10, 0.0, tz)], 0.588), g_b)
    A(_const_uv(_flat("tym_cames", _lattice(-0.4, 0.4, tz - 0.4, tz + 0.4, 0.0, tz, 0.16, 0.16, 0.012), 0.582)), "iron")

    # steps: individual stones with staggered joints, landing, cheek walls with copings and ball piers
    sr = rng.fork("steps")
    nr, go, land = 6, 0.3, 0.8
    rz = ZP / nr
    sw = Wd + 2 * 0.44 + 1.0
    for i in range(nr):
        if i < nr - 1:
            yf, dep = -(land + (nr - 1 - i) * go), go + 0.06
        else:
            yf, dep = -land, land + 0.03
        xs = [-sw / 2, -sw / 6 + sr.jitter(0, 0.3), sw / 6 + sr.jitter(0, 0.3), sw / 2]
        for j in range(3):
            ln = xs[j + 1] - xs[j] - 0.006
            b = stone_block("step", (ln, dep + 0.02, rz + 0.03), loc=((xs[j] + xs[j + 1]) / 2, yf + dep / 2 - 0.02 + sr.jitter(0, 0.004),
                                                                     (i + 1) * rz - (rz + 0.03) / 2), chamfer=0.014, rough=0.003, rng=sr, detail=0.26)
            A(b, "step", stone)
    y_foot = land + (nr - 1) * go                          # 2.3
    step_lamps = []
    for sx in (-1, 1):
        x0, x1 = sx * sw / 2, sx * (sw / 2 + 0.42)
        ck = _prism("cheek", [(-0.02, 0.0), (y_foot + 0.02, 0.0), (y_foot + 0.02, 0.5), (land + 0.05, ZP + 0.5), (-0.02, ZP + 0.5)], min(x0, x1), max(x0, x1))
        _grid_cut(ck, None, 0.5, 0.4)
        A(ck, "stone")
        cp = _prism("cheek_coping", [(-0.02, ZP + 0.49), (land + 0.09, ZP + 0.49), (y_foot + 0.02, 0.49), (y_foot + 0.02, 0.6),
                                     (land + 0.07, ZP + 0.6), (-0.02, ZP + 0.6)], min(x0, x1) - 0.05, max(x0, x1) + 0.05)
        _grid_cut(cp, None, 0.6, None)
        A(cp, "stone")
        xp = sx * (sw / 2 + 0.21)
        A(stone_block("step_pier", (0.56, 0.56, 1.05), loc=(xp, -(y_foot + 0.22), 0.525), chamfer=0.02, rng=sr, detail=0.2), "stone")
        A(stone_block("step_pier_cap", (0.68, 0.68, 0.12), loc=(xp, -(y_foot + 0.22), 1.11), chamfer=0.03, rng=sr, detail=0.25), "stone")
        step_lamps.append((xp, -(y_foot + 0.22), 1.17))

    # ---- lighting: six lights in all, none inside a fitting (a light inside a lantern only whites out its cage). The
    # lanterns themselves are amber emissive panes with a brighter flame; each light sits just in front of the lamps
    # it stands for: one between the two step lamps, one by a forecourt standard of each wing, one by each of the
    # three great bracket lanterns over the plaza-facing dials.
    R_L = 0.55
    lan_i, lan_g = _lantern("lantern", R_L, iron, g_b, dialm)
    lan_ms = []
    LI = p.lantern_intensity

    def lantern(face_m):
        lan_ms.append(face_m)

    H_ST = 2.95
    st_i, st_g = _standard("step_lamp", H_ST, iron, g_b, dialm, 0.8)
    ms = [trs(q) for q in step_lamps]
    A(instances_to_mesh("step_lamps", st_i, ms, keep_base=False), "iron")
    groups["glass"].append(instances_to_mesh("step_lamp_glass", st_g, ms, uv_jitter=False, keep_base=False))
    light("portal_lamps", (0.0, -1.5, ZP + 2.1), color="#ffb45e", intensity=LI * K_PORTAL, range=13.0, flicker=0.07,
          shadow_near=0.5)
    posts = []                                             # forecourt lamp standards (x, y)
    wing_lights = []
    Y_PO = 0.2

    # ======================================================================================= HALL
    # ---- bays
    def bay_layout(u0, u1, target=2.75):
        n = max(1, int(round((u1 - u0) / target)))
        bwid = (u1 - u0) / n
        return [u0 + bwid * (i + 0.5) for i in range(n)], bwid

    ZW0 = ZP + 1.0
    Hw = ZE - 0.8 - ZW0
    wings = []
    for sgn, has_t in ((-1, t_l), (1, t_r)):
        xe = xl if sgn < 0 else xr
        inner = sgn * (TW / 2 + 0.75)
        outer = xe - sgn * (RT + 0.35 if has_t else 0.95)
        lo, hi = min(inner, outer), max(inner, outer)
        cs, bwid = bay_layout(lo, hi)
        wings.append((sgn, has_t, cs, bwid, lo, hi, inner, outer))
    bwid_min = min(wg[3] for wg in wings)
    Ww = min(1.9, bwid_min - 1.05)
    f_front = _Face(0.0, YH, 0)
    f_back = _Face(0.0, D, 180)
    f_left = _Face(xl, ymid, -90)
    f_right = _Face(xr, ymid, 90)
    # buttress stages
    zb1, zb2, zb3 = ZP + 1.3, ZP + 0.58 * (ZE - 0.6 - ZP), ZE - 0.62
    z_wl = zb2 + 0.95                                      # bracket lanterns: wall plate on the face of the upper buttress stage
    wm_f, wm_b, bm_ = [], [], []                           # window (fine / rear) and buttress matrices
    pipes = []                                             # downpipe matrices
    for sgn, has_t, cs, bwid, lo, hi, inner, outer in wings:
        xe = xl if sgn < 0 else xr                         # this wing's end
        a_, b_ = (xl, -TW / 2 + 0.3) if sgn < 0 else (TW / 2 - 0.3, xr)
        holes = [_arch(Ww, Hw, kind, 0.0, 8, ZW0, c) for c in cs]
        wall_panel("hall_front", f_front, rect(a_, b_, 0.0, ZE + 0.3), holes, zs=[ZW0 - 0.2, ZE - 0.3])
        wm_f += [f_front.M(c, 0, ZW0) for c in cs]
        ib = [inner + sgn * bwid * i for i in range(1, len(cs))]   # buttresses between the bays, from the tower outwards
        bxs = ib + ([] if has_t else [xe - sgn * 0.3])              # + the corner buttress where there is no turret
        for x_ in bxs:
            bm_.append(f_front.M(x_, 0, 0))
        # lamp standards stand on the forecourt in line with buttresses (never in front of a window): the first and
        # the last one between the bays; any buttress between them carries a short bracket lantern
        px = [ib[0], ib[-1]] if len(ib) >= 2 else (ib[:1] if ib else ([] if has_t else bxs[:1]))
        for x_ in ib:
            if x_ not in px:
                lantern(f_front.M(x_, 0.345, z_wl))
        if has_t:                                          # a lantern on the plaza face of the turret
            lantern(trs((xe, YH - RT - 0.005, z_wl), (0, 0, 0)))
        posts += [(x_, Y_PO) for x_ in px]
        if px:                                             # the wing's light: by the standard nearest the middle of the wing
            wing_lights.append((sgn, "post", max(px, key=lambda x_: (-round(abs(x_ - (lo + hi) / 2), 2), abs(x_)))))
        else:
            wing_lights.append((sgn, "turret", xe))
        pipes.append(f_front.M(sgn * (TW / 2 + 0.62), 0, 0))
        pipes.append(f_front.M(outer + sgn * (0.14 if has_t else 0.08), 0, 0))
    csb, bwb = bay_layout(-(xr - 0.8), -(xl + 0.8), 3.3)   # back wall (u runs towards -x)
    i_ch = len(csb) // 2 if len(csb) >= 3 else -1          # the bay taken by the rear chimney breast
    u_ch = csb[i_ch] if i_ch >= 0 else None
    csw = [c for i, c in enumerate(csb) if i != i_ch]
    wall_panel("hall_back", f_back, rect(-xr, -xl, 0.0, ZE + 0.3), [_arch(Ww, Hw, kind, 0.0, 6, ZW0, c) for c in csw], du=0.15, dz=0.5,
               zs=[ZW0 - 0.2, ZE - 0.3])
    wm_b += [f_back.M(c, 0, ZW0) for c in csw]
    b_us = [-(xr - 0.8) + bwb * i for i in range(len(csb) + 1)]
    for u in b_us:
        bm_.append(f_back.M(u, 0, 0))
    nbk = len(b_us)
    for i in sorted(set([max(1, nbk // 4), nbk - 1 - max(1, nbk // 4)])):
        lantern(f_back.M(b_us[i], 0.345, z_wl))
        pipes.append(f_back.M(b_us[i] + 0.45, 0, 0))
    # gable ends with a rose and (if the hall is deep enough) two windows
    pp = 0.26
    zk = ZEr + ov * tp + pp
    Rr = clamp((ZR - ZE) * 0.2, 0.55, 1.0)
    zrose = ZE + (ZR - ZE) * 0.42
    side_ws = [-Dh * 0.2, Dh * 0.2] if Dh * 0.4 - Ww > 1.3 else [0.0]
    for f, has_t, sg in ((f_left, t_l, 1), (f_right, t_r, -1)):
        outer_ = [(-Dh / 2, 0.0), (Dh / 2, 0.0), (Dh / 2, zk)]
        outer_ += [(flat_half, ZR + pp), (-flat_half, ZR + pp)] if flat_half > 0 else [(0.0, ZR + pp)]
        outer_.append((-Dh / 2, zk))
        us = [u for u in side_ws if not (has_t and sg * u > Dh / 2 - RT - Ww / 2 - 0.4)]
        holes = [_arch(Ww, Hw, kind, 0.0, 8, ZW0, u) for u in us] + [_circle(Rr, 24, 0.0, zrose)]
        wall_panel("hall_gable", f, outer_, holes, 0.45, 0.2, 0.5, "gable", zs=[ZW0 - 0.2, ZE - 0.3])
        wm_f += [f.M(u, 0, ZW0) for u in us]
    gr = rng.fork("glass")
    win = _window("hall_window", Ww, Hw, kind, stone, True)
    groups["stone"].append(instances_to_mesh("hall_windows", win, wm_f, keep_base=False))
    if wm_b:
        win_b = _window("hall_window_rear", Ww, Hw, kind, stone, False)
        groups["stone"].append(instances_to_mesh("hall_windows_rear", win_b, wm_b, keep_base=False))
    gms = [[], [], []]
    for i, m in enumerate(wm_f + wm_b):
        u = gr.u(0.0, 1.0)
        gms[0 if u < 0.4 else (1 if u < 0.75 else 2)].append(m)
    for d_ in range(3):
        if gms[d_]:
            wg_ = _window_glass("hall_glass_%d" % d_, Ww, Hw, kind, d_, GM)
            groups["glass"].append(instances_to_mesh("hall_glazing_%d" % d_, wg_, gms[d_], uv_jitter=False, keep_base=False))
    # rose windows
    ro = []
    o = lathe("rose_ring", [(Rr + 0.24, 0.0), (Rr + 0.24, 0.06), (Rr + 0.2, 0.11), (Rr + 0.12, 0.11), (Rr + 0.08, 0.17), (Rr + 0.02, 0.17),
                            (Rr - 0.02, 0.12), (Rr - 0.02, -0.12)], segments=28, rot=(90, 0, 0))
    assign(_fx(o), stone)
    ro.append(o)
    petals = []
    npet = 8
    for i in range(npet):
        a = 2 * pi * (i + 0.5) / npet
        petals.append(_foil(3, Rr * 0.2, Rr * 0.62 * cos(a), Rr * 0.62 * sin(a), 4, a))
    eye = _circle(Rr * 0.27, 14)
    o = shape_extrude("rose_tracery", _circle(Rr, 28), petals + [eye], depth=0.1, loc=(0, 0.2, 0), rot=(90, 0, 0), bevel=0.006)
    assign(_fx(o), stone)
    ro.append(o)
    rg = [(_flat("rose_back", [_circle(Rr + 0.02, 20)], 0.25), dialm)]
    for par_, gm_ in ((0, g_c), (1, g_a)):
        rg.append((_flat("rose_petal", [_circle(Rr * 0.19, 8, Rr * 0.62 * cos(2 * pi * (i + 0.5) / npet), Rr * 0.62 * sin(2 * pi * (i + 0.5) / npet))
                                        for i in range(npet) if i % 2 == par_], 0.244), gm_))
    rg.append((_flat("rose_eye", [_circle(Rr * 0.3, 12)], 0.244), g_a))
    rg.append((_flat("rose_pupil", [_circle(Rr * 0.13, 8)], 0.238), g_b))
    rg.append((_flat("rose_cames", [_bar(Rr * 0.13 * cos(k * pi / 4), Rr * 0.13 * sin(k * pi / 4), Rr * 0.3 * cos(k * pi / 4), Rr * 0.3 * sin(k * pi / 4), 0.014)
                                    for k in range(8)], 0.232), iron))
    for o, m_ in rg:
        assign(_const_uv(o), m_)
        ro.append(o)
    ro = join(ro, "rose")
    groups["stone"].append(instances_to_mesh("roses", ro, [f_left.M(0, 0, zrose), f_right.M(0, 0, zrose)], uv_jitter=False, keep_base=False))

    # ---- buttresses (walling with stone weatherings and a plinth of three block courses)
    bprof = [(-0.02, 0.0), (0.78, 0.0), (0.78, zb1 - 0.3), (0.55, zb1), (0.55, zb2 - 0.3), (0.34, zb2), (0.34, zb3 - 0.45), (-0.02, zb3)]
    bt = [_prism("buttress", bprof, -0.28, 0.28)]
    _grid_cut(bt[0], 0.094, 0.3, 0.5, zs=[zb1 - 0.36, zb2 - 0.36])
    assign(bt[0], wallm)
    uv_box(bt[0], offset=None)
    for (oa, za, ob, zb_) in ((0.78, zb1 - 0.3, 0.55, zb1), (0.55, zb2 - 0.3, 0.34, zb2), (0.34, zb3 - 0.45, -0.02, zb3)):
        o = _prism("buttress_weathering", [(oa + 0.05, za - 0.05), (oa + 0.05, za + 0.03), (ob + 0.0, zb_ + 0.08), (ob - 0.0, zb_ - 0.0), (oa, za - 0.05)], -0.33, 0.33)
        assign(o, stone)
        bt.append(o)
    for ci, (z0, z1) in enumerate(courses):
        o = box("buttress_plinth", (0.84 + (0.02 if ci == 1 else 0.0), 0.94, z1 - z0 - 0.004), loc=(0, -0.45, (z0 + z1) / 2), bevel=0.02, segments=1)
        assign(o, stone)
        bt.append(o)
    o = _frustum("buttress_plinth_cap", (-0.42, 0.42, -0.92, 0.02, 0.9), (-0.3, 0.3, -0.8, 0.02, ZP + 0.02))
    assign(o, stone)
    bt.append(o)
    bt = join(bt, "buttress")
    groups["butt"].append(instances_to_mesh("buttresses", bt, bm_, uv_jitter=False, keep_base=False))

    # ---- rear chimney breast: a lateral stack climbing the back wall between two buttresses
    gut_gaps = []
    if u_ch is not None:
        ch = []
        zsh = ZE - 1.5
        o = _prism("breast", [(-0.02, 0.0), (0.5, 0.0), (0.5, zsh), (-0.02, zsh)], -0.9, 0.9)
        _grid_cut(o, 0.3, None, 0.5)
        ch.append((o, "wall"))
        ch.append((_frustum("breast_shoulder", (-0.94, 0.94, -0.54, 0.02, zsh - 0.02), (-0.56, 0.56, -0.5, 0.02, zsh + 0.75)), "stone"))
        z_st = ZE + min(4.2, rise * 0.8)
        o = box("breast_stack", (1.12, 0.62, z_st - zsh), loc=(0, -0.19, (z_st + zsh) / 2))
        apply_transform(o)
        _grid_cut(o, 0.3, 0.3, 0.5)
        ch.append((o, "wall"))
        ch.append((box("breast_band", (1.22, 0.72, 0.1), loc=(0, -0.19, z_st - 0.34), bevel=0.015, segments=1), "stone"))
        ch.append((stone_block("breast_cap", (1.3, 0.8, 0.16), loc=(0, -0.19, z_st + 0.08), chamfer=0.03, rng=rng.fork("breast"), detail=0.3), "stone"))
        for i, dx in enumerate((-0.3, 0.3)):
            hpot = 0.5 + 0.12 * i
            ch.append((lathe("breast_pot", [(0.15, 0.0), (0.16, 0.04), (0.125, 0.08), (0.11, hpot - 0.08), (0.14, hpot - 0.05), (0.14, hpot), (0.1, hpot),
                                            (0.1, hpot - 0.2)], segments=10, loc=(dx, -0.19, z_st + 0.16)), "roof" if p.roof == "clay" else "stone"))
        for ci, (z0, z1) in enumerate(courses):
            ch.append((box("breast_plinth", (1.96, 0.66, z1 - z0 - 0.004), loc=(0, -0.31, (z0 + z1) / 2), bevel=0.02, segments=1), "stone"))
        ch.append((_frustum("breast_plinth_cap", (-0.98, 0.98, -0.64, 0.02, 0.9), (-0.9, 0.9, -0.5, 0.02, ZP + 0.02)), "stone"))
        for o, g_ in ch:
            _fx(o)
            place(o, loc=(-u_ch, D, 0.0), rot=(0, 0, 180))
            apply_transform(o)
            A(o, g_)
            if g_ == "wall":
                uv_box(o, offset=None)
        gut_gaps.append(-u_ch)

    # ---- plinth of individual blocks, moulded plinth cap, eaves cornice with corbel table
    foot = [(-TW / 2, 0.0), (TW / 2, 0.0), (TW / 2, YH), (xr, YH), (xr, D), (xl, D), (xl, YH), (-TW / 2, YH)]
    course_blocks(foot, True)
    tray = _tray("plinth_block", PJ)
    A(instances_to_mesh("plinth_blocks", tray, blk_mats, keep_base=False), "stone")
    cap_prof = [(-0.02, 0.9), (0.15, 0.9), (0.15, 0.915), (0.05, ZP + 0.02), (-0.02, ZP + 0.02)]
    A(densify(_loop("plinth_cap", foot, cap_prof), 0.9), "stone")
    ev = [(TW / 2, YH), (xr, YH), (xr, D), (xl, D), (xl, YH), (-TW / 2, YH)]
    ev_prof = [(-0.02, ZE - 0.3), (0.05, ZE - 0.3), (0.085, ZE - 0.25), (0.085, ZE - 0.19), (0.2, ZE - 0.09), (0.2, ZE - 0.05), (0.27, ZE + 0.0),
               (0.27, ZE + 0.06), (-0.02, ZE + 0.06)]
    A(densify(_loop("eaves_cornice", ev, ev_prof, closed=False), 1.2), "stone")
    sill_prof = [(-0.02, ZW0 - 0.2), (0.05, ZW0 - 0.2), (0.08, ZW0 - 0.16), (0.08, ZW0 - 0.1), (-0.02, ZW0 - 0.02)]
    A(densify(_loop("sill_course", ev, sill_prof, closed=False), 1.2), "stone")
    for sgn in (-1, 1):
        a_, b_ = (xl, -TW / 2) if sgn < 0 else (TW / 2, xr)
        n = int((b_ - a_ - 0.5) / 0.46)
        for i in range(n + 1):
            cm.append(f_front.M(a_ + 0.25 + (b_ - a_ - 0.5) * i / max(n, 1), 0.0, ZE - 0.6))
    n = int((xr - xl - 0.5) / 0.46)
    for i in range(n + 1):
        cm.append(f_back.M(-xr + 0.25 + (xr - xl - 0.5) * i / n, 0.0, ZE - 0.6))
    A(instances_to_mesh("corbels", corbel, cm), "stone")

    # ---- half-round iron gutters on brackets, hopper heads and downpipes
    g_c_, g_z, g_R = ov + 0.075, ZEr - 0.1, 0.085
    gut = [(g_c_ + g_R * cos(radians(a_)), g_z + g_R * sin(radians(a_))) for a_ in range(180, 361, 30)]
    gut += [(g_c_ + (g_R - 0.014) * cos(radians(a_)), g_z + (g_R - 0.014) * sin(radians(a_))) for a_ in range(360, 179, -30)]
    runs = [(f_front, -xl - 0.0, -TW / 2 - 0.45), (f_front, TW / 2 + 0.45, xr)]
    if gut_gaps:
        runs += [(f_back, -xr, u_ch - 0.62), (f_back, u_ch + 0.62, -xl)]
    else:
        runs.append((f_back, -xr, -xl))
    runs[0] = (f_front, xl, -TW / 2 - 0.45)
    brk = [box("gutter_strap", (0.03, g_c_ - 0.2, 0.012), loc=(0, -(0.2 + (g_c_ - 0.2) / 2), g_z - g_R - 0.012)),
           box("gutter_lip", (0.03, 0.012, 0.1), loc=(0, -(g_c_ + g_R + 0.006), g_z - 0.04)),
           box("gutter_back", (0.03, 0.012, 0.1), loc=(0, -(g_c_ - g_R - 0.006), g_z - 0.04))]
    brk = join([_fx(o) for o in brk], "gutter_bracket")
    brk_ms = []
    for f, u0, u1 in runs:
        A(_loop("gutter", [f.pt(u0)[:2], f.pt(u1)[:2]], gut, closed=False), "iron")
        nb_ = max(2, int((u1 - u0) / 0.95))
        for i in range(nb_ + 1):
            brk_ms.append(f.M(u0 + 0.12 + (u1 - u0 - 0.24) * i / nb_, 0, 0))
    A(instances_to_mesh("gutter_brackets", brk, brk_ms, keep_base=False), "iron")
    zh = ZE - 0.82                                          # hopper head
    dp = [sweep("pipe_neck", [(0, -g_c_, g_z - g_R + 0.01), (0, -g_c_, ZE - 0.34), (0, -0.14, zh + 0.12), (0, -0.14, zh)], 0.042, sides=6,
                smooth_path=False),
          _frustum("hopper", (-0.085, 0.085, -0.21, -0.07, zh - 0.26), (-0.15, 0.15, -0.29, -0.03, zh + 0.0)),
          box("hopper_rim", (0.33, 0.29, 0.035), loc=(0, -0.16, zh + 0.01)),
          cylinder("downpipe", 0.05, zh - 0.26 - (ZP + 0.2), loc=(0, -0.14, (zh - 0.26 + ZP + 0.2) / 2), verts=8),
          cylinder("downpipe_shoe", 0.05, 0.26, loc=(0, -0.21, ZP + 0.15), rot=(50, 0, 0), verts=8)]
    zz = ZP + 0.7
    while zz < zh - 0.5:
        dp.append(box("pipe_clip", (0.2, 0.05, 0.05), loc=(0, -0.09, zz)))
        zz += 1.55
    dp = join([_fx(o) for o in dp], "downpipe")
    A(instances_to_mesh("downpipes", dp, pipes, keep_base=False), "iron")

    # ---- roof slopes with eaves course, ridge tiles, repair patches, cresting, gable copings and chimneys
    th_r = 0.1
    xa, xb = xl + 0.3, xr - 0.3
    rr_ = rng.fork("roof")
    tw_ = {"clay": 0.2, "slate": 0.3, "shingle": 0.15}[p.roof]
    tl_ = {"clay": 0.3, "slate": 0.4, "shingle": 0.34}[p.roof]
    tile = _tile("roof_tile", tw_ - 0.006, tl_, 0.024 if p.roof != "slate" else 0.012)
    tile_ms = []

    def slope_pt(x, s, front, lift=0.0):
        """Point on the roof surface: s = distance up the slope from the eaves."""
        if front:
            return (x, y_e + s * cp_ + lift * -sp_, ZEr + s * sp_ + lift * cp_)
        return (x, D + ov - s * cp_ + lift * sp_, ZEr + s * sp_ + lift * cp_)

    s_dorm = (0.8 + (1.6 + 0.65 * tan(radians(52.0))) / tp + 0.4) / cp_ + 0.25   # top of the dormer zone up the slope

    def roof_slope(name, u0, u1, y0, front):
        s0 = (y0 - y_e) / cp_ if front else 0.0
        if front:
            q = [slope_pt(u0, s0, True), slope_pt(u1, s0, True), slope_pt(u1, slope_len, True), slope_pt(u0, slope_len, True)]
        else:
            q = [slope_pt(u1, 0.0, False), slope_pt(u0, 0.0, False), slope_pt(u0, slope_len, False), slope_pt(u1, slope_len, False)]
        s = _slab(name, q, th_r)
        nrm = (0.0, -sp_, cp_) if front else (0.0, sp_, cp_)
        _strip(s, (-nrm[0], -nrm[1], -nrm[2]))
        _grid_cut(s, 0.55, 0.38, None)
        displace_noise(s, strength=0.02, scale=1.9, seed=seed + (3 if front else 4))
        A(s, "roof")
        yawz = 0.0 if front else 180.0
        if s0 < 0.05:                                      # projecting eaves course of individual tiles (double course)
            n = max(1, int((u1 - u0) / tw_))
            wd = (u1 - u0) / n
            for row in range(2):
                for i in range(n):
                    x = u0 + wd * (i + 0.5 + (0.5 if row else 0.0))
                    if x > u1 - wd * 0.4:
                        continue
                    loc = slope_pt(x, -0.07 + row * tl_ * 0.62 + rr_.jitter(0, 0.008), front, 0.002 + row * 0.012)
                    tile_ms.append(trs(loc, (pdeg - 2.5 + rr_.jitter(0, 1.2), rr_.jitter(0, 1.5), yawz + rr_.jitter(0, 1.5)), (wd / tw_, 1.0, 1.0)))
        area = (u1 - u0) * (slope_len - s0)
        # re-laid areas above the dormers: ragged outlines stepping along the tile courses
        s_lo = max(s0 + 0.6, (s_dorm if front else 0.9) if (not front or int(p.dormers) > 0) else 0.0)
        for i in range(max(1, int(area / (24.0 if front else 11.0)))):
            pw, ph = rr_.u(1.0, 2.4) * (1.0 if front else 1.3), rr_.u(0.6, 1.5)
            if slope_len - 0.5 - ph < s_lo or u1 - u0 < pw + 1.0:
                continue
            px, ps = rr_.u(u0 + 0.4, u1 - 0.4 - pw), rr_.u(s_lo, slope_len - ph - 0.5)
            rows = []
            nrw = max(2, int(round(ph / 0.3)))
            for r in range(nrw):
                a_ = px + tw_ * int(rr_.u(-2.0, 2.99))
                b_ = px + pw + tw_ * int(rr_.u(-2.0, 2.99))
                sa, sb = ps + r * ph / nrw, ps + (r + 1) * ph / nrw
                if front:
                    q = [slope_pt(a_, sa, True, 0.012), slope_pt(b_, sa, True, 0.012), slope_pt(b_, sb, True, 0.012), slope_pt(a_, sb, True, 0.012)]
                else:
                    q = [slope_pt(b_, sa, False, 0.012), slope_pt(a_, sa, False, 0.012), slope_pt(a_, sb, False, 0.012), slope_pt(b_, sb, False, 0.012)]
                rows.append(_slab("roof_patch_row", q, 0.012))
            pt_ = join(rows, "roof_patch")
            pt_["tsq_patch"] = rr_.u(0.0, 1.0)
            A(pt_, "patch")

    for (u0, u1, y0) in ((xa, -TW / 2 + 0.02, y_e), (TW / 2 - 0.02, xb, y_e), (-TW / 2 + 0.02, TW / 2 - 0.02, TW - 0.02)):
        if y0 < y_top - 0.05:
            roof_slope("roof_front", u0, u1, y0, True)
    roof_slope("roof_back", xa, xb, 0.0, False)
    A(instances_to_mesh("roof_tiles", tile, tile_ms, keep_base=False), "roof")

    # ---- rear slope furniture: a second chimney stack through the slope with a lead apron, the sweep's roof ladder
    # hooked over the ridge beside it, and tile-vent gablets between the dormers
    rb = rng.fork("rear_roof")
    kbr = max(2, int((xr - xl) / 4.4))                     # rear dormer cells (a dormer sits in the middle of each)
    j_st = 1 if rb.u(0.0, 1.0) < 0.5 else kbr - 1
    x_st = xl + (xr - xl) * j_st / kbr
    s_st = slope_len * 0.62
    c_ = slope_pt(x_st, s_st, False)
    z_top = ZR + 1.0
    b = box("rear_stack", (0.95, 0.72, z_top - (c_[2] - 0.6)), loc=(x_st, c_[1], (z_top + c_[2] - 0.6) / 2))
    apply_transform(b)
    _grid_cut(b, 0.24, 0.24, 0.5)
    A(b, "wall")
    uv_box(b, offset=None)
    A(box("rear_stack_band", (1.05, 0.82, 0.1), loc=(x_st, c_[1], z_top - 0.32), bevel=0.015, segments=1), "stone")
    A(stone_block("rear_stack_cap", (1.13, 0.9, 0.16), loc=(x_st, c_[1], z_top + 0.08), chamfer=0.03, rng=rb, detail=0.3), "stone")
    for dx in (-0.24, 0.24):
        hpot = rb.u(0.42, 0.6)
        A(lathe("rear_stack_pot", [(0.15, 0.0), (0.16, 0.04), (0.125, 0.08), (0.11, hpot - 0.08), (0.14, hpot - 0.05), (0.14, hpot), (0.1, hpot),
                                   (0.1, hpot - 0.2)], segments=10, loc=(x_st + dx, c_[1], z_top + 0.16)), "roof" if p.roof == "clay" else "stone")
    A(box("rear_stack_apron", (1.3, 1.25, 0.03), loc=slope_pt(x_st, s_st, False, 0.03), rot=(-pdeg, 0, 0)), "deck")
    x_ld = x_st + (0.95 if x_st < (xl + xr) / 2 else -0.95)
    yr_b = y_topb if flat_half > 0 else ymid
    sl0, sl1 = 0.25, slope_len - 0.03
    for dx in (-0.2, 0.2):
        o = box("roof_ladder_rail", (0.045, sl1 - sl0, 0.05), loc=slope_pt(x_ld + dx, (sl0 + sl1) / 2, False, 0.062), rot=(-pdeg, 0, 0),
                bevel=0.004, segments=1)
        uv_grain(_fx(o), "Y")
        A(o, "wood")
        A(box("roof_ladder_hook", (0.03, 0.46, 0.012), loc=(x_ld + dx, yr_b - 0.1, ZR + 0.118)), "iron")
        A(box("roof_ladder_hook_end", (0.03, 0.012, 0.14), loc=(x_ld + dx, yr_b - 0.33, ZR + 0.054)), "iron")
    rung = box("roof_ladder_rung", (0.47, 0.05, 0.026), bevel=0.003, segments=1)
    uv_grain(rung, "X")
    n = int((sl1 - sl0 - 0.3) / 0.3)
    A(instances_to_mesh("roof_ladder_rungs", rung, [trs(slope_pt(x_ld, sl0 + 0.15 + 0.3 * i, False, 0.1), (-pdeg, 0, 0)) for i in range(n + 1)],
                        keep_base=False), "wood")
    s_v, vw, vh = slope_len * 0.52, 0.62, 0.3
    for j in range(1, kbr):
        xv = xl + (xr - xl) * j / kbr
        if j == j_st or (u_ch is not None and abs(xv + u_ch) < 1.3) or abs(xv - x_ld) < 0.7:
            continue
        f_ = slope_pt(xv, s_v, False, 0.0)
        yf_, zf_ = f_[1] + 0.06, f_[2] - 0.06 * tp
        o = mesh_object("roof_vent", [(xv - vw / 2 - 0.03, yf_, zf_), (xv + vw / 2 + 0.03, yf_, zf_), (xv, yf_, f_[2] + vh),
                                      (xv, f_[1] - vh / tp, f_[2] + vh)], [(0, 1, 2), (1, 3, 2), (0, 2, 3)])
        recalc_normals(o)
        uv_planar(o, u_axis=(1, 0, 0), v_axis=(0, -cp_, sp_))
        A(o, "roof")
        A(_extrude_y("roof_vent_louvre", [(xv - vw * 0.36, f_[2] + 0.015), (xv + vw * 0.36, f_[2] + 0.015), (xv, f_[2] + vh * 0.86)],
                     yf_ - 0.03, yf_ + 0.022), "wood")
        for k in range(2):
            A(box("roof_vent_slat", (vw * (0.5 - 0.22 * k), 0.03, 0.014), loc=(xv, yf_ + 0.03, f_[2] + 0.075 + 0.085 * k), rot=(-35, 0, 0)), "deck")

    cr_spike = [cone("crest_spike", r1=0.022, r2=0.004, depth=0.34, loc=(0, 0, 0.2), verts=4),
                sphere("crest_ball", 0.03, loc=(0, 0, 0.19), segments=5, rings=3),
                torus("crest_loop", 0.08, 0.009, loc=(0.16, 0, 0.1), rot=(90, 0, 0), major_seg=8, minor_seg=3)]
    for o in cr_spike:
        _fx(o)
    cr_spike = join(cr_spike, "crest")
    crm = []
    ridge = cylinder("ridge_tile", 0.115, 0.42, verts=8, cap=False, radius2=0.098)
    ridge_ms = []
    ridge_lines = [y_top, y_topb] if flat_half > 0 else [ymid]
    if flat_half > 0:
        fl_ = _slab("roof_deck", [(xa, y_top - 0.05, ZR + 0.02), (xb, y_top - 0.05, ZR + 0.02), (xb, y_topb + 0.05, ZR + 0.02), (xa, y_topb + 0.05, ZR + 0.02)], 0.12)
        A(fl_, "deck")
        nsm = max(2, int((xb - xa) / 0.9))
        for i in range(1, nsm):                            # standing seams of the metal deck
            A(box("deck_seam", (0.03, y_topb - y_top + 0.06, 0.035), loc=(xa + (xb - xa) * i / nsm, ymid, ZR + 0.035)), "deck")
    for yr in ridge_lines:
        segs = [(xa + 0.3, xb - 0.3)]
        if yr < TW + 0.15:
            segs = [(xa + 0.3, -TW / 2 - 0.2), (TW / 2 + 0.2, xb - 0.3)]
        for (u0, u1) in segs:
            n = max(1, int(round((u1 - u0) / 0.38)))
            for i in range(n):
                ridge_ms.append(trs((u0 + (u1 - u0) * (i + 0.5) / n, yr, ZR - 0.015 + rr_.jitter(0, 0.004)), (0, 90 + rr_.jitter(0, 1.0), rr_.jitter(0, 1.0)),
                                    (1.0, 1.0, (u1 - u0) / n / 0.42 * 1.14)))
            A(box("crest_rail", (u1 - u0, 0.014, 0.03), loc=((u0 + u1) / 2, yr, ZR + 0.17)), "iron")
            n = int((u1 - u0) / 0.32)
            for i in range(n + 1):
                if rr_.u(0.0, 1.0) < 0.04:
                    continue                              # the odd spike has gone missing
                crm.append(trs((u0 + (u1 - u0) * i / n, yr, ZR + 0.12), (rr_.jitter(0, 3.0), rr_.jitter(0, 3.0), 0)))
    A(instances_to_mesh("ridge_tiles", ridge, ridge_ms, keep_base=False), "roof")
    A(instances_to_mesh("cresting", cr_spike, crm, keep_base=False), "iron")
    kr = rng.fork("gable")
    for xe, sgn in ((xl, -1), (xr, 1)):
        c0, c1 = (xe - 0.07, xe + 0.52) if sgn < 0 else (xe - 0.52, xe + 0.07)
        zc1 = ZR + pp
        yk = YH + 0.0
        ytop_f = ymid - flat_half
        A(densify(_slab("gable_coping", [(c0, yk - 0.1, zk - 0.02), (c1, yk - 0.1, zk - 0.02), (c1, ytop_f, zc1), (c0, ytop_f, zc1)], 0.14), 1.0), "stone")
        A(densify(_slab("gable_coping", [(c1, D + 0.1, zk - 0.02), (c0, D + 0.1, zk - 0.02), (c0, ymid + flat_half, zc1), (c1, ymid + flat_half, zc1)], 0.14), 1.0), "stone")
        if flat_half > 0:
            A(box("gable_coping_flat", (0.59, 2 * flat_half + 0.1, 0.14), loc=((c0 + c1) / 2, ymid, zc1 - 0.07), bevel=0.015, segments=1), "stone")
        for yy in (YH - 0.14, D + 0.14):
            A(stone_block("kneeler", (0.66, 0.62, 0.5), loc=((c0 + c1) / 2, yy, zk - 0.33), chamfer=0.03, rng=kr, detail=0.3), "stone")
        # chimney stack on the gable
        xc_ = xe + sgn * -0.42
        ch_h = 2.3
        b = box("chimney", (0.84, 1.3, ch_h + 1.0), loc=(xc_, ymid, ZR + ch_h / 2 - 0.5))
        apply_transform(b)
        _grid_cut(b, 0.3, 0.33, 0.5)
        A(b, "wall")
        uv_box(b, offset=None)
        A(box("chimney_band", (0.94, 1.4, 0.1), loc=(xc_, ymid, ZR + ch_h - 0.32), bevel=0.015, segments=1), "stone")
        A(stone_block("chimney_cap", (1.02, 1.48, 0.16), loc=(xc_, ymid, ZR + ch_h + 0.08), chamfer=0.03, rng=kr, detail=0.3), "stone")
        for dy in (-0.33, 0.33):
            hpot = kr.u(0.42, 0.6)
            A(lathe("chimney_pot", [(0.15, 0.0), (0.16, 0.04), (0.125, 0.08), (0.11, hpot - 0.08), (0.14, hpot - 0.05), (0.14, hpot), (0.1, hpot),
                                    (0.1, hpot - 0.2)], segments=10, loc=(xc_, ymid + dy, ZR + ch_h + 0.16)), "roof" if p.roof == "clay" else "stone")

    # ---- dormers: lamp-lit, half shuttered or shut for the night (seeded), front and rear
    nd = int(p.dormers)
    if nd > 0:
        dw, dhw, dgp = 1.3, 1.6, 52.0
        y_d = y_e + 0.8
        back = (dhw + dw / 2 * tan(radians(dgp))) / tp + 0.4
        spots = []
        for sgn, has_t, cs, bwid, lo, hi, inner, outer in wings:
            k = min(nd, max(1, int((hi - lo) / 2.6)))
            for i in range(k):
                spots.append(trs((lo + (hi - lo) * (i + 0.5) / k, y_d, zroof(y_d) - 0.12), (0, 0, 0)))
        nfr = len(spots)
        kb_ = max(2, int((xr - xl) / 4.4))
        y_db = D + ov - 0.8
        for i in range(kb_):
            x_ = xl + (xr - xl) * (i + 0.5) / kb_
            if u_ch is not None and abs(x_ + u_ch) < 1.5:
                x_ += 1.7 if x_ >= -u_ch else -1.7
            spots.append(trs((x_, y_db, zroof(y_db) - 0.12), (0, 0, 180)))
        dq = rng.fork("dormers")
        modes = []
        for i in range(len(spots)):
            u = dq.u(0.0, 1.0)
            modes.append("lit" if u < 0.56 else ("half" if u < 0.8 else "dark"))
        if nfr >= 2 and not any(m != "lit" for m in modes[:nfr]):
            modes[int(dq.u(0, nfr - 0.001))] = "dark"
        if nfr >= 1 and not any(m == "lit" for m in modes[:nfr]):
            modes[int(dq.u(0, nfr - 0.001))] = "lit"
        for md in ("lit", "half", "dark"):
            ms = [m for m, mm in zip(spots, modes) if mm == md]
            if ms:
                dm = _gablet("dormer_" + md, dw, dhw, dgp, back, kind, DM, md)
                groups["stone"].append(instances_to_mesh("dormers_" + md, dm, ms, uv_jitter=False, keep_base=False))

    # ---- hall quoins at the free corners
    for xe, sgn, has_t in ((xl, -1, t_l), (xr, 1, t_r)):
        if not has_t:
            quoins(xe + sgn * 0.0, YH, sgn, -1, ZP + 0.05, ZE - 0.35)
        quoins(xe, D, sgn, 1, ZP + 0.05, ZE - 0.35)
    qb = box("quoin", (0.56, 0.3, 0.29), bevel=0.014, segments=1)
    A(instances_to_mesh("quoins", qb, q_mats, keep_base=False), "stone")

    # ---- forecourt: a low flagged terrace with a kerb fills the re-entrant between tower and wings; the lamp
    # standards stand on pad stones at its edge, in line with the buttresses
    Y_K, H_K = -0.32, 0.15
    fq = rng.fork("forecourt")
    flag = _fx(box("flag", (1.0, 1.0, 0.15), loc=(0, 0, 0.075), bevel=0.007, segments=1))
    flag_ms = []
    y_f0, y_f1 = Y_K + 0.3, YH - 0.08
    nrow = 4
    rd = (y_f1 - y_f0) / nrow
    for sgn, has_t, cs, bwid, lo, hi, inner, outer in wings:
        xe = xl if sgn < 0 else xr
        xa_ = sgn * (TW / 2 + 0.05)
        xb_ = xe + sgn * (RT + 0.3 if has_t else 0.42)
        u0, u1 = min(xa_, xb_), max(xa_, xb_)
        x = u0
        while x < u1 - 1e-6:                               # kerb stones along the front edge
            ln = fq.u(0.85, 1.35)
            if u1 - (x + ln) < 0.6:
                ln = u1 - x
            A(stone_block("kerb", (ln - 0.008, 0.3, H_K + 0.025), loc=(x + ln / 2, Y_K + 0.15, (H_K + 0.025) / 2), chamfer=0.018, rough=0.002,
                          rng=fq, detail=0.3), "pave")
            x += ln
        y = y_f0
        xk = xb_ - sgn * 0.15
        while y < YH + 0.2:                                # kerb return at the outer end
            ln = min(fq.u(0.8, 1.2), YH + 0.3 - y)
            A(stone_block("kerb", (0.3, ln - 0.008, H_K + 0.025), loc=(xk, y + ln / 2, (H_K + 0.025) / 2), chamfer=0.018, rough=0.002,
                          rng=fq, detail=0.3), "pave")
            y += ln
        f0, f1 = (u0, u1 - 0.3) if sgn > 0 else (u0 + 0.3, u1)
        for r in range(nrow):                              # flags in courses with broken joints
            x, first = f0, True
            while x < f1 - 1e-6:
                ln = fq.u(0.6, 1.15) * (0.55 if (first and r % 2) else 1.0)
                if f1 - (x + ln) < 0.4:
                    ln = f1 - x
                xc_, yc_ = x + ln / 2, y_f0 + rd * (r + 0.5)
                x += ln
                first = False
                if has_t and hypot(xc_ - xe, yc_ - YH) < RT - 0.45:
                    continue
                flag_ms.append(trs((xc_, yc_, 0.0), (0, 0, fq.jitter(0, 0.2)), (ln - 0.008, rd - 0.008, (H_K + fq.jitter(0, 0.004)) / 0.15)))
    A(instances_to_mesh("forecourt_flags", flag, flag_ms, keep_base=False), "pave")
    H_PO, H_PAD = 4.0, H_K + 0.03
    if posts:
        po_i, po_g = _standard("lamp_post", H_PO, iron, g_b, dialm)
        pm_ = [trs((x_, y_, H_PAD), (0, 0, 0)) for x_, y_ in posts]
        A(instances_to_mesh("lamp_posts", po_i, pm_, keep_base=False), "iron")
        groups["glass"].append(instances_to_mesh("lamp_post_glass", po_g, pm_, uv_jitter=False, keep_base=False))
        for x_, y_ in posts:
            A(box("post_pad", (0.62, 0.62, H_PAD), loc=(x_, y_, H_PAD / 2), bevel=0.02, segments=1), "pave")
            for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):   # holding-down bolts of the base
                A(cylinder("post_bolt", 0.014, 0.03, loc=(x_ + dx * 0.125, y_ + dy * 0.125, H_PAD + 0.07), verts=6), "iron")
    for sgn, fit, lx in wing_lights:                       # 1.3 m in front of the lantern, level with its flame
        lp = (lx, Y_PO - 1.3, H_PAD + H_PO) if fit == "post" else (lx, YH - RT - R_L - 1.3, z_wl - 0.3)
        light("wing_lamps_" + ("l" if sgn < 0 else "r"), lp, color="#ffb45e", intensity=LI * K_WING, range=14.0, flicker=0.07,
              shadow_near=0.9)

    # ======================================================================================= TURRETS
    def turret(xc, yc, sgn):
        ZT = ZE + 2.2
        nseg = 28
        A(lathe("turret_shaft", [(RT, ZP - 0.05)] + [(RT, ZP + (ZT - ZP) * i / 20) for i in range(1, 21)], segments=48, loc=(xc, yc, 0)), "turret")
        A(lathe("turret_plinth", [(RT + 0.15, 0.0), (RT + 0.15, 0.32), (RT + 0.125, 0.34), (RT + 0.15, 0.36), (RT + 0.15, 0.62), (RT + 0.125, 0.64),
                                  (RT + 0.15, 0.66), (RT + 0.15, ZP - 0.1), (RT + 0.04, ZP + 0.02), (RT - 0.02, ZP + 0.02)], segments=nseg, loc=(xc, yc, 0)), "stone")
        jb = box("turret_joint", (0.02, 0.012, 0.3))
        jm_ = []
        for ci, (z0, z1) in enumerate(courses):            # perpend joints of the curved plinth blocks
            for i in range(11):
                aj = 2 * pi * (i + 0.5 * (ci % 2) + pr.jitter(0, 0.08)) / 11
                jm_.append(trs((xc + (RT + 0.148) * cos(aj), yc + (RT + 0.148) * sin(aj), (z0 + z1) / 2), (0, 0, degrees(aj) + 90), (1.0, 1.0, (z1 - z0) / 0.3)))
        A(instances_to_mesh("turret_joints", jb, jm_, keep_base=False), "deck")
        for z in (ZW0 - 0.11, ZE - 0.1):
            A(lathe("turret_band", [(RT - 0.02, z - 0.1), (RT + 0.06, z - 0.1), (RT + 0.095, z - 0.05), (RT + 0.095, z + 0.03), (RT - 0.02, z + 0.1)],
                    segments=nseg, loc=(xc, yc, 0)), "stone")
        A(lathe("turret_cornice", [(RT - 0.02, ZT - 0.5), (RT + 0.05, ZT - 0.5), (RT + 0.09, ZT - 0.44), (RT + 0.09, ZT - 0.36), (RT + 0.2, ZT - 0.22),
                                   (RT + 0.2, ZT - 0.16), (RT + 0.3, ZT - 0.06), (RT + 0.3, ZT + 0.04), (RT - 0.02, ZT + 0.04)], segments=nseg, loc=(xc, yc, 0)), "stone")
        tcm = [trs((xc + RT * cos(2 * pi * i / 20), yc + RT * sin(2 * pi * i / 20), ZT - 0.82), (0, 0, degrees(2 * pi * i / 20) + 90)) for i in range(20)]
        A(instances_to_mesh("turret_corbels", corbel, tcm), "stone")
        Hc = 4.1
        prof = [(RT + 0.2, ZT - 0.02), (RT + 0.5, ZT - 0.02), (RT + 0.5, ZT + 0.03), (RT + 0.3, ZT + 0.22), (RT + 0.1, ZT + 0.5)]
        r1, z1 = RT - 0.1, ZT + 0.9
        prof += [(r1 + (0.07 - r1) * i / 7, z1 + (ZT + Hc - z1) * i / 7) for i in range(8)]
        A(lathe("turret_roof", prof, segments=nseg, loc=(xc, yc, 0), smooth_angle=60), "roof")
        zt2 = ZT + Hc
        A(lathe("turret_cap", [(0.26, zt2 - 0.6), (0.16, zt2 - 0.25), (0.08, zt2 - 0.05), (0.12, zt2 + 0.02), (0.12, zt2 + 0.06), (0.05, zt2 + 0.14),
                               (0.03, zt2 + 0.5), (0.0, zt2 + 0.5)], segments=12, loc=(xc, yc, 0)), "deck")
        A(sphere("turret_ball", 0.11, loc=(xc, yc, zt2 + 0.56), segments=12, rings=7), "brass")
        A(cone("turret_spike", r1=0.025, r2=0.003, depth=0.5, loc=(xc, yc, zt2 + 0.88), verts=6), "iron")
        sl = _slit("turret_light", 0.36, 1.15, kind, stone, g_b, iron, dialm)
        step_h = (ZT - 2.4 - (ZP + 1.2)) / 3.0
        azs = [(250, ZP + 1.2), (205, ZP + 1.2 + step_h), (300, ZP + 1.2 + step_h * 1.5), (180, ZP + 1.2 + step_h * 2.3), (325, ZP + 1.2 + step_h * 2.6)]
        # CLEARANCE RULE: the upper stair lights stay below the ring of lights under the cornice (the last one, 5 degrees
        # round from a ring light, ran up into it: two lit windows crossing each other)
        azs = [(a_, min(z_, ZT - 2.05 - 1.4)) for a_, z_ in azs]
        azs += [(a_, ZT - 2.05) for a_ in (140, 185, 230, 275, 320, 5)]
        ms = []
        for az, z in azs:
            if sgn > 0:
                az = 180 - az
            ar = radians(az)
            ms.append(trs((xc + (RT + 0.065) * cos(ar), yc + (RT + 0.065) * sin(ar), z), (0, 0, az + 90)))
        groups["stone"].append(instances_to_mesh("turret_lights", sl, ms, uv_jitter=False, keep_base=False))

    if t_l:
        turret(xl, YH, -1)
    if t_r:
        turret(xr, YH, 1)
    delete([corbel])
    if lan_ms:
        A(instances_to_mesh("lanterns", lan_i, lan_ms, keep_base=False), "iron")
        groups["glass"].append(instances_to_mesh("lantern_glass", lan_g, lan_ms, uv_jitter=False, keep_base=False))
    else:
        delete([lan_i, lan_g])

    # ======================================================================================= LIGHTS, SOCKETS
    # a great lantern on a long bracket from the parapet over each plaza-facing dial; its light sits between the
    # lantern and the dial so the clock stage is washed from above and the lantern keeps its amber glass
    CL_R, CL_D = 1.45, 0.85
    cl_i, cl_g, cl_z, cl_b = _clock_lamp("clock_lamp", CL_R, CL_D, 1.5, iron, g_b, dialm, brass)
    cl_f = [tf[key] for key in ("f", "l", "r")]
    cl_ms = [f.M(0.0, 0.3, TH + 0.3) for f in cl_f]
    A(instances_to_mesh("clock_lamps", cl_i, cl_ms, keep_base=False), "iron")
    A(instances_to_mesh("clock_lamp_gilt", cl_b, cl_ms, keep_base=False), "brass")
    groups["glass"].append(instances_to_mesh("clock_lamp_glass", cl_g, cl_ms, uv_jitter=False, keep_base=False))
    for key, f in zip("flr", cl_f):                        # the light keeps its place over the dial whatever the hanger length
        light("clock_lamp_" + key, f.pt(0.0, 0.3 + CL_R - 0.45, TH - 0.92), color="#ffb866", intensity=LI * K_CLOCK, range=13.0,
              flicker=0.05, shadow_near=0.9, halo_at=f.pt(0.0, 0.3 + CL_R, TH + 0.3 + cl_z))
    socket("banner_l", "attach", (-TW / 2 - 0.37, 0.4, ZS1 + 1.0), rot=(0, 0, 0), role="banner")
    socket("banner_r", "attach", (TW / 2 + 0.37, 0.4, ZS1 + 1.0), rot=(0, 0, 0), role="banner")
    socket("notice", "attach", (Wd / 2 + 0.44 + 0.35, -0.01, ZP + 1.5), rot=(0, 0, 0), role="sign")
    socket("door", "attach", (0.0, 0.0, ZP), rot=(0, 0, 0), role="door")

    # ======================================================================================= WEATHERING
    # directional: narrow run-off streaks under the sill course, eaves, strings, corbel table and every buttress
    # set-off, heavy drips below sill ends, label stops and the belfry sills; an even film of soot that takes the
    # new-stone glare off, block-to-block tone on the dressings, dirt on ledges and set-off tops; a damp band at
    # the foot; moss on the shaded tops of plinth and set-offs (rear and gables worst); worn pale tread centres.
    def shade(co, n):                                      # 1 on the sunless rear, less on the plaza front
        return np.clip(0.45 + 0.55 * (co[:, 1] - YH) / max(D - YH, 1.0), 0.35, 1.0)

    def up(co, n):
        return np.clip(n[:, 2], 0.0, 1.0) ** 0.6

    def damp(co, n):
        return np.clip(1.0 - co[:, 2] / 0.42, 0.0, 1.0) ** 0.7

    def roofline(co):
        y = co[:, 1]
        return np.minimum(ZR, np.minimum(ZEr + (y - y_e) * tp, ZEr + (D + ov - y) * tp)) + 0.2

    splash = height_ramp(ZP, ZP + 1.0)
    w_spring = _arch_dims(Ww, Hw, kind)[1]
    mk = []
    for sgn, has_t, cs, bwid, lo, hi, inner, outer in wings:
        for c in cs:
            for sx in (-1, 1):
                mk.append((c + sx * (Ww / 2 + 0.3), ZW0 - 0.2, 0.8, 0.95, 0.14))             # ends of the window sills
                mk.append((c + sx * (Ww / 2 + 0.31), ZW0 + w_spring - 0.17, 1.7, 0.75, 0.12))  # label stops of the hood moulds
    m_hall = [_marks(f_front, mk)]
    m_tow = []
    for key, f in tf.items():
        mk = [(sx * (bx + bw / 2 + 0.3), zbs - 0.14, 2.1, 0.95, 0.14) for sx in (-1, 1)] + [(0.0, zbs - 0.14, 1.2, 0.7, 0.12)]
        if key == "f":
            mk += [(sx * (Wd / 2 + 0.58), p_spring - 0.2, 1.8, 0.85, 0.14) for sx in (-1, 1)]
            mk += [(sx * (mid_w / 2 + 0.2), ZS1 + 0.46, 0.5, 0.7, 0.1) for sx in (-1, 1)]
        m_tow.append(_marks(f, mk))

    def wall_grime(runs, marks=(), base=0.17, k=0.8):
        def fn(co, n):
            st = runs(co, n)
            for m in marks:
                st = np.maximum(st, m(co, n))
            tone = base + 0.12 * _noise1(co[:, 0] * 0.33 + co[:, 1] * 0.29 + co[:, 2] * 0.21, seed + 17)
            return np.clip(tone + k * st * (1.0 - tone) + 0.3 * splash(co, n) + 0.7 * damp(co, n), 0.0, 1.0)
        return fn

    def wall_moss(co, n):
        low = np.clip(1.0 - (co[:, 2] - ZP) / 1.2, 0.0, 1.0)
        return np.clip((up(co, n) * 0.9 + low * 0.5 * _noise1((co[:, 0] + co[:, 1]) * 0.9, seed + 5)) * shade(co, n) * moss, 0.0, 1.0)

    def ledge(co, n):                                      # dirt lying on set-off tops, sills, copings
        return up(co, n) * np.clip((co[:, 2] - ZP) / 0.4, 0.0, 1.0)

    g_wall = wall_grime(_runs([ZW0 - 0.2, ZE - 0.3, ZR + 2.3 - 0.37], 1.5, seed + 11, dens=0.2, w=SW), m_hall)
    g_gable = wall_grime(_runs([ZW0 - 0.2, ZE - 0.3, roofline], 1.7, seed + 12, dens=0.22, w=0.2))
    g_turret = wall_grime(_runs([ZW0 - 0.21, ZE - 0.2, ZE + 1.7], 1.6, seed + 13, dens=0.25, w=0.2))
    r_butt = wall_grime(_runs([zb1 - 0.36, zb2 - 0.36], 1.6, seed + 14, dens=0.45, w=0.094), base=0.2)
    tops_t = [ZS1 - 0.27, ZK0 - 0.1, ZC0] + ([ZB0 - 0.1] if midh > 0.6 else [])
    g_tow = wall_grime(_runs(tops_t, 2.0, seed + 23, dens=0.2, w=SW), m_tow)

    def g_butt(co, n):
        return np.clip(np.maximum(r_butt(co, n), 0.5 * ledge(co, n)), 0.0, 1.0)

    for grp, fn in (("wall", g_wall), ("gable", g_gable), ("turret", g_turret), ("butt", g_butt), ("twall", g_tow)):
        for o in groups[grp]:
            paint(o, grime=fn, special=wall_moss)

    def stone_grime(co, n):
        blk = _hash1(np.floor(co[:, 2] / 0.3 + 0.05) * 3.0 + np.floor((co[:, 0] * 0.83 + co[:, 1]) / 0.62), seed + 2)   # block-to-block tone
        return np.clip(0.13 + 0.17 * blk + 0.42 * ledge(co, n) + 0.85 * damp(co, n) + 0.3 * np.clip(1.0 - co[:, 2] / 1.0, 0.0, 1.0), 0.0, 1.0)

    def stone_moss(co, n):
        low = np.clip(1.0 - (co[:, 2] - 0.8) / 0.9, 0.0, 1.0)
        return np.clip(up(co, n) * (0.35 + 0.65 * low) * shade(co, n) * moss * 1.5, 0.0, 1.0)

    for o in groups["stone"]:
        paint(o, grime=stone_grime, special=stone_moss)

    def pave_grime(co, n):                                 # trodden paving: flag-to-flag tone, darker along the wall and in the kerb face
        cell = np.floor(co[:, 0] / 0.87) * 7.0 + np.floor((co[:, 1] - Y_K) / 0.6) * 13.0
        back = np.clip((co[:, 1] - (YH - 1.1)) / 1.1, 0.0, 1.0)
        return np.clip(0.3 + 0.26 * _hash1(cell, seed + 3) + 0.25 * back + 0.25 * (1.0 - np.clip(n[:, 2], 0.0, 1.0)), 0.0, 1.0)

    def pave_moss(co, n):
        back = np.clip((co[:, 1] - (YH - 0.9)) / 0.9, 0.0, 1.0)
        return np.clip(back * _noise1(co[:, 0] * 1.3, seed + 6) * moss * 0.9, 0.0, 1.0)

    for o in groups["pave"]:
        paint(o, grime=pave_grime, special=pave_moss)

    def step_dirt(co, nr):                                 # feet keep the tread centres pale; dirt gathers towards the cheek walls
        side = np.clip((np.abs(co[:, 0]) - 0.8) / 0.9, 0.0, 1.0)
        top = np.clip(nr[:, 2], 0.0, 1.0)
        return np.clip(0.16 + side ** 0.8 * (0.5 + 0.34 * top) + (1.0 - top) * 0.22, 0.0, 1.0)

    def step_moss(co, nr):
        return np.clip((np.abs(co[:, 0]) - 1.55) / 0.5, 0.0, 1.0) * np.clip(nr[:, 2], 0.0, 1.0) * moss * 0.9

    for o in groups["step"]:
        paint(o, grime=step_dirt, special=step_moss)

    def roof_moss(co, n):
        s = (co[:, 2] - ZEr) / max(rise, 1.0)
        low = np.clip(1.0 - s / 0.45, 0.0, 1.0)
        nz = _noise1(co[:, 0] * 0.7 + co[:, 2] * 0.4, seed + 9)
        return np.clip(low * (0.25 + 0.75 * nz) * shade(co, n) * moss * 1.3, 0.0, 1.0)

    st_roof = _runs([ZR + 0.3], 3.4, seed + 41, dens=0.3, w=0.55)

    def roof_grime(co, n):
        return np.clip(0.4 * st_roof(co, n) + 0.3 * _noise1(co[:, 0] * 0.45 + co[:, 1] * 0.8, seed + 13), 0.0, 1.0)

    for o in groups["roof"]:
        paint(o, special=roof_moss, grime=roof_grime)
    for o in groups["patch"]:                              # re-laid areas: sooty second-hand tiles or pale new ones
        t = float(o["tsq_patch"])
        del o["tsq_patch"]
        if t >= 0.45:
            paint(o, grime=0.42 + 0.9 * (t - 0.45))
        else:
            paint(o, wear=0.3 + 0.8 * t)
    for o in groups["iron"]:
        paint(o, special=mask_mul(noise_mask(0.3, seed=seed + 7, lo=0.4, hi=0.8), 0.5))
    for o in groups["brass"]:
        paint(o, special=mask_mul(noise_mask(0.25, seed=seed + 8, lo=0.4, hi=0.8), 0.45))
    for o in groups["wood"]:
        paint(o, grime=mask_mul(height_ramp(ZP, ZP + 0.5), 0.3))
