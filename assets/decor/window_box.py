"""Flower window box that hangs below a sill on two forged scroll brackets. Three constructions:
a nailed plank box (separate sawn boards, cleats, lozenge appliques, chipped paint), a wrought-iron
hayrack basket with a coir liner, or a tapered terracotta trough with a swag relief. It is filled with
mounded compost (it shows between the clumps where the planting is not grown in) and planted with
geraniums: rosettes of round scalloped leaves around each crown over a small-leaved under-mass, upright
umbels of 12-18 cupped five-petal florets (three tints, dark throat, pale eye) with drooping buds, and
trailing strands (heart-shaped leaves on petioles at every node, small cupped blooms) that swing out over
the front rim and the ends. Leaves / petals are single double-sided sheets whose normals face the open air
(the baked AO would turn enclosed blades black); the planting scales itself to the triangle cap.
Origin: centre of the back top edge of the box; wall plane y = 0.
"""
from math import sin, cos, pi, radians, exp

import bpy
import numpy as np
from mathutils import Matrix, Vector

from tk import *

META = {"name": "Flower window box", "class": "prop_small", "lod": "far", "mount": "wall", "mount_height": 1.15,
        "tri_budget": 7000,    # geranium umbels of 12-18 real florets; capped in build() well below this
        "focus": [0.0, -0.15, 0.05]}
PARAMS = {
    "width": {"default": 0.9, "min": 0.5, "max": 1.8, "desc": "overall width of the box (m)"},
    "depth": {"default": 0.2, "min": 0.15, "max": 0.28, "desc": "projection of the box from the wall (m)"},
    "height": {"default": 0.18, "min": 0.14, "max": 0.24, "desc": "height of the box (m)"},
    "style": {"default": "plank", "choices": ["plank", "iron_basket", "terracotta"], "desc": "box construction"},
    "finish": {"default": "painted", "choices": ["painted", "oak", "weathered"], "desc": "timber finish of the plank box"},
    "paint": {"default": "#35574a", "desc": "paint colour (plank box) / ironwork colour (basket)"},
    "flower_a": {"default": "#c9242b", "desc": "colour of the upright flower heads"},
    "flower_b": {"default": "#f1e7dc", "desc": "colour of the trailing blooms"},
    "foliage": {"default": "#84b062", "desc": "foliage tint"},
    "fullness": {"default": 0.75, "min": 0.0, "max": 1.0, "desc": "how grown-in the planting is"},
    "trailing": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "amount / length of trailing strands"},
    "blooms": {"default": 0.7, "min": 0.0, "max": 1.0, "desc": "flower density"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "paint chipping, rust and staining"},
}
VARIANTS = {
    "iron_basket": {"style": "iron_basket", "width": 1.1, "paint": "#1c2420", "flower_a": "#7b3fb0", "flower_b": "#f2c230",
                    "fullness": 0.9, "trailing": 1.0, "seed": 4},
    "terracotta": {"style": "terracotta", "width": 0.7, "height": 0.2, "flower_a": "#e25a8f", "flower_b": "#f6efe6",
                   "fullness": 0.6, "trailing": 0.35, "seed": 9},
    "wide_oak": {"finish": "oak", "width": 1.5, "depth": 0.24, "height": 0.21, "flower_a": "#e8822a", "flower_b": "#4a63c8",
                 "fullness": 0.9, "trailing": 0.75, "blooms": 0.85, "seed": 13},
    "sparse_weathered": {"finish": "weathered", "width": 0.6, "depth": 0.17, "height": 0.15, "flower_a": "#d9c23a",
                         "flower_b": "#c9242b", "fullness": 0.25, "trailing": 0.25, "blooms": 0.5, "wear": 0.9, "seed": 21},
}

TB = 0.006      # bracket flat bar thickness
WB = 0.026      # bracket flat bar width
GOLD = 2.39996  # golden angle (rad)


# ----------------------------------------------------------------------------------------- helpers
def _shade(hex_col, k, to_white=0.0):
    h = hex_col.lstrip("#")
    c = [int(h[i:i + 2], 16) * k for i in (0, 2, 4)]
    c = [min(255, max(0, int(v + (255 - v) * to_white))) for v in c]
    return "#%02x%02x%02x" % tuple(c)


WAX = (1.0, 0.9, 0.65)      # hue of the ivory 'wax' base, normalised - divided out of petal tints


def _petal(hex_col, k=1.0, to_white=0.0):
    """Tint that gives the petal material (ivory wax base) the requested flower colour."""
    h = _shade(hex_col, k, to_white).lstrip("#")
    lin = [(int(h[i:i + 2], 16) / 255.0) ** 2.2 / WAX[j] for j, i in enumerate((0, 2, 4))]
    m = max(lin)
    if m > 1.15:
        lin = [c * 1.15 / m for c in lin]
    return "#%02x%02x%02x" % tuple(int(round(min(1.0, c) ** (1 / 2.2) * 255)) for c in lin)


def _mat4(x, y, z, pos, s):
    sx, sy, sz = s if isinstance(s, (tuple, list)) else (s, s, s)
    return Matrix(((x.x * sx, y.x * sy, z.x * sz, pos[0]),
                   (x.y * sx, y.y * sy, z.y * sz, pos[1]),
                   (x.z * sx, y.z * sy, z.z * sz, pos[2]),
                   (0.0, 0.0, 0.0, 1.0)))


def _m_along(pos, ydir, zhint, s):
    """Right-handed matrix whose local +Y runs along ydir and +Z is as close to zhint as possible."""
    y = Vector(ydir)
    y = y.normalized() if y.length > 1e-6 else Vector((0, 0, 1))
    z = Vector(zhint)
    z = z - y * z.dot(y)
    z = z.normalized() if z.length > 1e-4 else y.orthogonal().normalized()
    return _mat4(y.cross(z), y, z, pos, s)


def _m_facing(pos, zdir, s, r):
    """Right-handed matrix whose local +Z faces zdir, random spin around it."""
    z = Vector(zdir)
    z = z.normalized() if z.length > 1e-6 else Vector((0, 0, 1))
    v = Vector((r.u(-1, 1), r.u(-1, 1), r.u(-1, 1)))
    y = v - z * v.dot(z)
    y = y.normalized() if y.length > 1e-3 else z.orthogonal().normalized()
    return _mat4(y.cross(z), y, z, pos, s)


def _set_mask(o, name, vals):
    me = o.data
    a = me.attributes.get("m_" + name)
    if a is None:
        a = me.attributes.new("m_" + name, 'FLOAT', 'POINT')
    a.data.foreach_set("value", np.asarray(vals, dtype=np.float32))


def _sheet(name, verts, faces, uvs, grime=None, angle=75.0, wear=None):
    """Single-sheet mesh (rendered double-sided) with one UV per vertex in 'leaf' tile units (0..1)."""
    o = mesh_object(name, verts, faces)
    me = o.data
    uvl = me.uv_layers[0] if me.uv_layers else me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            u, v = uvs[me.loops[li].vertex_index]
            uvl.data[li].uv = (u * 0.25, v * 0.25)
    uv_keep(o)
    smooth(o, angle)
    if grime is not None:
        _set_mask(o, "grime", grime)
    if wear is not None:
        _set_mask(o, "wear", wear)
    return o


def _stamp(name, base, ms, material, gadd=None, angle=75.0):
    """Instance `base` at the matrices; gadd = per-instance offset on the grime mask (tone variation)."""
    if not ms:
        return None
    nv = len(base.data.vertices)
    o = instances_to_mesh(name, base, ms, uv_jitter=False)
    assign(o, material)
    smooth(o, angle)
    if gadd is not None:
        me = o.data
        k = len(ms)
        vals = np.zeros(nv * k, dtype=np.float32)
        a = me.attributes.get("m_grime")
        if a is None:
            a = me.attributes.new("m_grime", 'FLOAT', 'POINT')
        else:
            a.data.foreach_get("value", vals)
        vals = np.clip(vals.reshape(k, nv) + np.asarray(gadd, dtype=np.float32)[:, None], 0.0, 1.0)
        a.data.foreach_set("value", vals.ravel())
    return o


def _leaf_round(name, n, cup, droop, wave):
    """Geranium leaf blade, about 1 unit across, petiole point at the origin, tip towards +Y, top = +Z:
    round, shallowly scalloped outline with a narrow sinus at the petiole (cordate base), funnel-cupped
    around the petiole point, gently wavy margin, drooping tip. n perimeter verts + centre = n tris."""
    cy = 0.17
    vs, uvs = [(0.0, cy, 0.0)], [(0.5, 0.04 + (cy + 0.3) / 1.02 * 0.92)]
    sinus = radians(17.0)
    for i in range(n):
        if i == 0:
            th, rr = -pi / 2, cy * 0.8
        else:
            th = -pi / 2 + sinus + (2 * pi - 2 * sinus) * (i - 1) / (n - 2)
            rr = 0.49 + 0.014 * cos(5 * (th - pi / 2))
        x, y = rr * cos(th), cy + rr * sin(th)
        z = cup * (rr / 0.5) ** 2 - droop * max(0.0, (y - cy) / 0.5) ** 2
        if i > 0:
            z += wave * (1 if i % 2 else -1) * (0.5 if i in (1, n - 1) else 1.0)
        vs.append((x, y, z))
        uvs.append((0.5 + x / 1.06, 0.04 + (y + 0.3) / 1.02 * 0.92))
    fs = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return _sheet(name, vs, fs, uvs)


def _leaf_heart(name):
    """Trailing leaf: heart-shaped blade (unit length) folded along the midrib, drooping tip, on a short
    petiole that starts at the origin (the stem node) = 7 tris."""
    pl = 0.34
    half = ((0.30, -0.15, 0.035), (0.46, 0.22, 0.085), (0.27, 0.64, 0.04))
    vs = [(0.0, 0.0, 0.0), (0.0, pl, 0.0)]
    vs += [(x, pl + y, z) for (x, y, z) in half]
    vs.append((0.0, pl + 1.0, -0.11))
    vs += [(-x, pl + y, z) for (x, y, z) in reversed(half)]
    vs += [(0.04, pl + 0.03, 0.004), (-0.04, pl + 0.03, 0.004)]
    fs = [(1, 2 + i, 3 + i) for i in range(6)] + [(0, 9, 10)]
    uvs = [(0.5 + x / 0.98, 0.05 + max(0.0, y - pl + 0.16) / 1.2 * 0.92) for (x, y, z) in vs]
    return _sheet(name, vs, fs, uvs)


def _floret(name, ped=0.0, edge=0.3):
    """Five-petal bloom facing +Z: separate obovate petals, cupped upwards and twisted like a pinwheel so
    each overlaps its neighbour; the throat is painted darker (grime mask), the petal tips paler (wear
    mask = bleached) and a small pale eye sits in the centre. ped lifts the flower above the origin
    (stalked blooms). 16 tris; V runs along every petal."""
    vs, fs, uvs, g, w = [], [], [], [], []
    for k, ln in enumerate((1.0, 0.9, 0.98, 0.93, 0.87)):
        a = radians(k * 72.0 + 90.0)
        rx, ry = cos(a), sin(a)
        b = len(vs)
        for (pr, pt, pz) in ((0.05, 0.0, 0.0), (0.56 * ln, -0.41, 0.12), (0.95 * ln, -0.19, 0.23),
                             (0.95 * ln, 0.19, 0.23), (0.56 * ln, 0.41, 0.12)):
            vs.append((rx * pr - ry * pt, ry * pr + rx * pt, pz + 0.05 * pt / 0.41 + ped))
        fs += [(b, b + 1, b + 2), (b, b + 2, b + 3), (b, b + 3, b + 4)]
        uvs += [(0.5, 0.06), (0.1, 0.5), (0.3, 0.94), (0.7, 0.94), (0.9, 0.5)]
        g += [0.5, 0.1, 0.0, 0.0, 0.1]
        w += [0.0, 0.0, edge, edge * 0.7, 0.0]
    b = len(vs)
    z = ped + 0.065
    vs += [(0.14 * cos(radians(90 + 120 * k)), 0.14 * sin(radians(90 + 120 * k)), z) for k in range(3)]
    fs.append((b, b + 1, b + 2))
    uvs += [(0.5, 0.5)] * 3
    g += [0.0] * 3
    w += [1.0] * 3
    return _sheet(name, vs, fs, uvs, grime=g, angle=70.0, wear=w)


def _ribbon(name, nodes, face, hw, taper=0.45, us=(0.47, 0.53)):
    """Thin trailing stem as a double-sided strip through the nodes, its flat side turned towards `face`."""
    vs, uvs, fs = [], [], []
    n = len(nodes)
    face = Vector(face)
    for k, q in enumerate(nodes):
        tg = nodes[min(k + 1, n - 1)] - nodes[max(k - 1, 0)]
        side = tg.cross(face)
        side = side.normalized() if side.length > 1e-6 else Vector((1, 0, 0))
        w = hw * (1.0 - taper * k / (n - 1))
        vs += [tuple(q - side * w), tuple(q + side * w)]
        v = 0.1 + 0.8 * k / (n - 1)
        uvs += [(us[0], v), (us[1], v)]
    for k in range(n - 1):
        a = 2 * k
        fs.append((a, a + 1, a + 3, a + 2))
    return _sheet(name, vs, fs, uvs)


def _pedicel(name, ped):
    vs = [(-0.045, 0, 0), (0.045, 0, 0), (0, 0, ped + 0.02), (0, -0.045, 0), (0, 0.045, 0)]
    return _sheet(name, vs, [(0, 1, 2), (3, 4, 2)], [(0.45, 0.2), (0.55, 0.2), (0.5, 0.8), (0.45, 0.2), (0.55, 0.2)])


def _bud(name):
    """Closed bud drooping from the umbel centre (origin) along +Z: thin pedicel + 3-sided spindle = 7 tris."""
    vs = [(-0.035, 0, 0.12), (0.035, 0, 0.12), (0, 0, 0.66), (0, 0, 0.62), (0, 0, 1.0)]
    for k in range(3):
        a = radians(120 * k + 30)
        vs.append((0.105 * cos(a), 0.105 * sin(a), 0.80))
    fs = [(0, 1, 2)]
    for k in range(3):
        k1 = (k + 1) % 3
        fs += [(3, 5 + k1, 5 + k), (4, 5 + k, 5 + k1)]
    uvs = [(0.5 + x, 0.1 + 0.8 * z) for (x, y, z) in vs]
    return _sheet(name, vs, fs, uvs, grime=[0.5, 0.5, 0.6, 0.75, 0.2, 0.45, 0.45, 0.45], angle=80.0)


def _oriented_sweep(name, path, prof, out_dir, up):
    """sweep() whose profile +a side is guaranteed to face out_dir (mirrors the section if needed)."""
    o = sweep(name, path, prof, smooth_path=False, up=up)
    k = max(range(len(prof)), key=lambda i: prof[i][0])
    d = Vector(o.data.vertices[k].co) - Vector(path[0])
    if d.dot(Vector(out_dir)) < 0:
        delete(o)
        o = sweep(name, path, [(-a, b) for (a, b) in reversed(prof)], smooth_path=False, up=up)
    return o


def _board(name, p0, p1, thick, wide, out_dir, up, n, r, bow=0.0012):
    """Sawn board from p0 to p1: eased arrises on the outer face, a row of verts 9 mm inside each long
    edge and every ~7 cm along the length (so paint chipping / stains have resolution), slight bow."""
    p0, p1, od = Vector(p0), Vector(p1), Vector(out_dir)
    L = (p1 - p0).length
    fr = [0.0, min(0.03, 0.2 * L) / L] + [(i + r.u(-0.25, 0.25)) / n for i in range(1, n)] + [1.0 - min(0.03, 0.2 * L) / L, 1.0]
    fr = sorted(set(fr))
    b = r.jitter(0, bow)
    path = [tuple(p0.lerp(p1, f) + od * (b * sin(pi * f))) for f in fr]
    ht, hh, c, e = thick / 2, wide / 2, 0.003, 0.010
    prof = [(-ht, -hh), (ht - c, -hh), (ht, -hh + c), (ht, -hh + e), (ht, hh - e), (ht, hh - c), (ht - c, hh), (-ht, hh)]
    return _oriented_sweep(name, path, prof, od, up)


def _u_outline(hw, d, rc, yb=-0.005, na=3):
    """U-shaped plan outline open towards the wall: left end, round the front, right end."""
    pts = [(-hw, yb)]
    for i in range(na + 1):
        a = radians(180 + 90 * i / na)
        pts.append((-hw + rc + rc * cos(a), -(d - rc) + rc * sin(a)))
    for i in range(na + 1):
        a = radians(270 + 90 * i / na)
        pts.append((hw - rc + rc * cos(a), -(d - rc) + rc * sin(a)))
    pts.append((hw, yb))
    return pts


def _tris_in_scene():
    return sum(len(pl.vertices) - 2 for o in bpy.data.objects if o.type == 'MESH' for pl in o.data.polygons)


# ----------------------------------------------------------------------------------------- far level (lod 2)
def _hull(name, vs, fs, c, grain=None):
    """A handful of flat faces, each wound to look away from the point c (one point, or one per face)."""
    out = []
    for k, f in enumerate(fs):
        q = [Vector(vs[i]) for i in f]
        ck = Vector(c[k]) if isinstance(c, list) else Vector(c)
        n = (q[1] - q[0]).cross(q[2] - q[0])
        out.append(tuple(f) if n.dot(sum(q, Vector((0, 0, 0))) / len(q) - ck) >= 0 else tuple(reversed(f)))
    o = mesh_object(name, vs, out)
    if grain:
        uv_grain(o, grain)
    else:
        uv_box(o)
    return o


def _bracket_far(xb, H, reach, iron):
    """The bracket from afar: wall leg and arm as one three-sided bar (its flat face outwards, closed at both
    ends) and a loop of three thin bars where the volute curls."""
    leg = clamp(reach * 0.95, 0.16, 0.26)
    w, yo, yr, zo, zr, zl = WB * 0.62, -TB - 0.001, -0.001, -H - TB, -H - 0.0005, -H - leg
    vs = [(xb - w, yo, zl), (xb + w, yo, zl), (xb, yr, zl), (xb - w, yo, zo), (xb + w, yo, zo), (xb, yr, zr),
          (xb - w, -reach, zo), (xb + w, -reach, zo), (xb, -reach, zr)]
    fs = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5), (0, 1, 2), (3, 4, 7, 6), (4, 5, 8, 7), (5, 3, 6, 8), (6, 7, 8)]
    cl, ca = (xb, yo + (yr - yo) / 3, (zl + zo) / 2), (xb, (yo - reach) / 2, zo + (zr - zo) / 3)
    bar = _hull("bracket_bar", vs, fs, [cl] * 4 + [ca] * 4)
    R = 0.42 * min(reach, leg)
    yc, zc = -TB - 0.003 - R, -H - TB - 0.002 - R
    pts = [(yc + R + 0.004, zc), (yc, zc + R + 0.004), (yc - 0.5 * R, zc - 0.5 * R)]      # on the leg, under the arm, the far side of the curl
    my, mz = sum(q[0] for q in pts) / 3, sum(q[1] for q in pts) / 3
    vs, fs, cs = [], [], []
    for k, (y, z) in enumerate(pts):
        n = Vector((y - my, z - mz)).normalized()
        vs += [(xb - 0.008, y + n.x * 0.002, z + n.y * 0.002), (xb + 0.008, y + n.x * 0.002, z + n.y * 0.002), (xb, y - n.x * 0.004, z - n.y * 0.004)]
        a, b = 3 * k, 3 * ((k + 1) % 3)
        fs += [(a, a + 1, b + 1, b), (a + 1, a + 2, b + 2, b + 1), (a + 2, a, b, b + 2)]
        cs += [(xb, (y + pts[(k + 1) % 3][0]) / 2, (z + pts[(k + 1) % 3][1]) / 2)] * 3
    st = _hull("bracket_scroll", vs, fs, cs)
    for o in (bar, st):
        assign(o, iron)
    return [bar, st]


def _heads_far(name, heads, zl):
    """Geranium heads from afar: every umbel a closed eight-faced gem, and for its drooping buds and stalk a thin
    three-sided point from inside the gem down into the foliage just over the compost lid (zl), so no head floats.
    heads: (centre, axis, e1, e2, umbel radius, stalk foot)."""
    vs, fs, uvs = [], [], []
    for (c, ax, e1, e2, ru, foot) in heads:
        b = len(vs)
        t = clamp((c.z - zl - 0.04) / max(c.z - foot.z, 1e-4), 0.4, 1.0)
        vs += [tuple(c + ax * (1.3 * ru))] + [tuple(c + d * (1.45 * ru) - ax * (0.05 * ru)) for d in (e1, e2, -e1, -e2)] + [tuple(c - ax * (1.1 * ru))]
        uvs += [(0.5, 0.92), (0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5), (0.5, 0.08)]
        for k in range(4):
            fs += [(b, b + 1 + k, b + 1 + (k + 1) % 4), (b + 5, b + 1 + (k + 1) % 4, b + 1 + k)]
        tip = c.lerp(foot, t)
        if (c - tip).length > 1.6 * ru:
            b = len(vs)
            vs += [tuple(c - ax * (0.6 * ru) + (e1 * cos(radians(120 * k)) + e2 * sin(radians(120 * k))) * 0.006) for k in range(3)] + [tuple(tip)]
            uvs += [(0.4, 0.3), (0.5, 0.3), (0.6, 0.3), (0.5, 0.08)]
            fs += [(b + 3, b + (k + 1) % 3, b + k) for k in range(3)]
    return _sheet(name, vs, fs, uvs, angle=70.0) if vs else None


def _bloom_far(name, ped):
    """A stalked bloom from afar: one triangle where the flower faces."""
    vs = [(0.84 * cos(radians(120 * k + 90)), 0.84 * sin(radians(120 * k + 90)), ped + 0.12) for k in range(3)]
    return _sheet(name, vs, [(0, 1, 2)], [(0.5, 0.9), (0.15, 0.3), (0.85, 0.3)])


# ----------------------------------------------------------------------------------------- brackets
def _bracket(xb, H, reach, iron, r, wear, seed):
    """Forged shelf bracket: one flat bar bent into wall leg + arm with an upturned retaining lip and
    a pig-tail foot, a volute scroll riveted into the corner, two collars and coach bolts. Rust blooms
    at the fixings, the scroll eye, the foot and the lip; the arm is rubbed bright where the box sits."""
    objs = []
    leg = clamp(reach * 0.95, 0.16, 0.26)
    za = -H - TB / 2            # arm centreline
    y0 = -TB / 2 - 0.001        # leg centreline
    zl = -H - leg
    rb = 0.012
    R = 0.42 * min(reach, leg)
    yc = -TB - 0.003 - R
    zc = -H - TB - 0.002 - R
    zb1, zb2 = -H - TB - 0.028, zc - R * 0.2          # coach bolt heights
    path = []
    rc = 0.013                  # pig-tail curl at the foot
    for i in range(6):
        a = radians(-215 + 215 * i / 5)
        path.append((xb, y0 - rc + rc * cos(a), zl + rc * sin(a)))
    for z in (zb2 - 0.03, zb2, (zb1 + zb2) / 2, zb1):
        path.append((xb, y0, z))
    for i in range(4):
        a = radians(90 * i / 3)
        path.append((xb, y0 - rb + rb * cos(a), za - rb + rb * sin(a)))
    path.append((xb, -reach * 0.5, za))
    for i in range(4):
        a = radians(-90 - 90 * i / 3)
        path.append((xb, -reach + rb + rb * cos(a), za + rb + rb * sin(a)))
    path.append((xb, -reach - 0.002, za + 0.036))
    hb_, ht_ = WB / 2, TB / 2
    bar_prof = [(-ht_, -hb_), (ht_, -hb_), (ht_, hb_), (-ht_, hb_)]
    bar = sweep("bracket_bar", path, bar_prof, smooth_path=False, up=(1, 0, 0))
    objs.append(bar)

    # volute: strap up the leg, round under the arm, then curling in on itself
    vp = [(xb, yc + R, zl + 0.014), (xb, yc + R, (zl + 0.014 + zc) / 2)]
    for i in range(5):
        a = radians(90 * i / 4)
        vp.append((xb, yc + R * cos(a), zc + R * sin(a)))
    turns = 380.0
    n = 14
    for i in range(1, n + 1):
        t = i / n
        a = radians(90 + turns * t)
        rr = R * (1.0 - 0.74 * t ** 0.85)
        vp.append((xb, yc + rr * cos(a), zc + rr * sin(a)))
    tip = vp[-1]
    prof = [(-0.002, -0.008), (0.002, -0.008), (0.002, 0.008), (-0.002, 0.008)]
    objs.append(sweep("bracket_scroll", vp, prof, smooth_path=False, up=(1, 0, 0), scale_fn=lambda t: 1.0 - 0.4 * t * t))
    # collars binding the scroll to arm and leg
    objs.append(box("collar_arm", (WB + 0.007, 0.013, TB + 0.007), loc=(xb, yc, -H - (TB + 0.0075) / 2 - 0.0004)))
    objs.append(box("collar_leg", (WB + 0.007, TB + 0.0085, 0.013), loc=(xb, -(TB + 0.0085) / 2 - 0.0004, zl + 0.034)))
    # coach bolts into the wall
    bt = bolt("coach_bolt", 0.0075, 0.0045, kind="pyramid")
    ms = [trs((xb, -TB - 0.001, zb1), (90, 0, r.u(0, 40))),
          trs((xb, -TB - 0.001, zb2), (90, 0, r.u(0, 40)))]
    objs.append(instances_to_mesh("coach_bolts", bt, ms, keep_base=False))

    hot = np.array([(y0, zb1, 0.05), (y0, zb2, 0.05), (tip[1], tip[2], 0.035), (y0 - rc, zl, 0.034),
                    (-reach - 0.002, za + 0.03, 0.04), (yc, -H - TB, 0.03), (y0, zl + 0.034, 0.03)])
    nz_a = noise_mask(0.03, seed=seed + 31, lo=0.3, hi=0.75)
    nz_b = noise_mask(0.05, seed=seed + 37, lo=0.4, hi=0.8)

    def rust(co, nrm):
        d = np.sqrt((co[:, 1:2] - hot[None, :, 0]) ** 2 + (co[:, 2:3] - hot[None, :, 1]) ** 2)
        bloom = np.clip(1.0 - (d - 0.006) / hot[None, :, 2], 0.0, 1.0).max(axis=1)
        na_ = nz_a(co, nrm)
        return np.clip(np.maximum(bloom * (0.25 + 0.25 * wear + 0.5 * na_), na_ * (0.3 + 0.6 * wear)), 0.0, 1.0)

    def rubbed(co, nrm):         # arm top + outer arrises burnished by the box and by handling
        top = np.clip((co[:, 2] - (za - 0.002)) / 0.004, 0.0, 1.0) * np.clip(nrm[:, 2], 0.0, 1.0)
        return np.clip(top * 0.55 + nz_b(co, nrm) * 0.3, 0.0, 1.0)

    for o in objs:
        assign(o, iron)
        paint(o, special=rust, wear=rubbed, grime=mask_mul(height_ramp(zl - 0.02, zl + 0.12), 0.6))
    return objs


# ----------------------------------------------------------------------------------------- boxes
def _box_plank(p, r, W, D, H, wood, nail_mat):
    t = 0.018
    boards = []
    nfb = 2 if H < 0.2 else 3
    gap = 0.003
    hb = (H - gap * (nfb - 1)) / nfb
    zcs = [-hb / 2 - i * (hb + gap) for i in range(nfb)]
    nseg = max(5, int(round(W / 0.11 / max(1.0, W / 0.9) ** 0.6)))
    for i, zc in enumerate(zcs):
        w_ = W + r.jitter(0, 0.004)
        xo, yo, tz = r.jitter(0, 0.002), -D + t / 2 + r.jitter(0, 0.0012), r.jitter(0, 0.0012)
        boards.append(_board(f"front{i}", (xo - w_ / 2, yo, zc - tz), (xo + w_ / 2, yo, zc + tz), t, hb, (0, -1, 0), (0, 0, 1), nseg, r))
    back = box("back", (W - 0.004, t, H - 0.002), loc=(0, -0.004 - t / 2, -H / 2))
    uv_grain(back, 'X')
    boards.append(back)
    el = D - 2 * t - 0.004
    ye0, ye1 = -0.004 - t - 0.0005, -D + t + 0.0005
    for sx in (-1, 1):
        xe_ = sx * (W / 2 - t / 2 - 0.0015)
        ze = -H / 2 + r.jitter(0, 0.001)
        boards.append(_board("end", (xe_, ye0, ze), (xe_, ye1, ze), t, H - 0.003, (sx, 0, 0), (0, 0, 1), 2, r, bow=0.0006))
    bw = (el - 0.006) / 2
    for k in (-1, 1):
        b = box("bottom", (W - 2 * t - 0.006, bw, t), loc=(0, -(D + 0.004) / 2 + k * (bw / 2 + 0.003), -H + t / 2))
        uv_grain(b, 'X')
        boards.append(b)
    # cleats nailed across the front boards + lozenge appliques between them
    xs = [-(W / 2 - 0.075), W / 2 - 0.075] if W < 1.15 else [-(W / 2 - 0.075), 0.0, W / 2 - 0.075]
    nails = []                                              # (x, y, z)
    for x in xs:
        for zc in zcs:
            nails.append((x + r.jitter(0, 0.006), -D - 0.012, zc + r.jitter(0, hb * 0.15)))
    for sx in (-1, 1):
        for zc in zcs:
            for dz in (-0.27, 0.27):
                nails.append((sx * (W / 2 - t / 2 - 0.001) + r.jitter(0, 0.002), -D, zc + dz * hb + r.jitter(0, 0.003)))
    for i, x in enumerate(xs):
        lean = r.jitter(0, 0.0015)
        c = _board(f"cleat{i}", (x - lean, -D - 0.006, -H + 0.003), (x + lean, -D - 0.006, -0.005), 0.012, 0.042, (0, -1, 0), (1, 0, 0),
                   3, r, bow=0.0005)
        boards.append(c)
    if W >= 0.58:
        s = H * 0.36
        for i in range(len(xs) - 1):
            xm = (xs[i] + xs[i + 1]) / 2
            lz = box(f"lozenge{i}", (s, 0.008, s), loc=(xm, -D - 0.004, -H / 2), rot=(0, 45 + r.jitter(0, 1.5), 0), bevel=0.0025, segments=1)
            uv_grain(lz, 'X')
            boards.append(lz)

    # ---- weathering: paint chipped back to bare wood along the arrises, at the board ends, cleat corners and
    # round the nail heads; drip stains from the drain gap down the lower board; compost spilt over the rim
    painted = p.finish == "painted"
    wk = (0.62 + 0.38 * p.wear) * (1.0 if painted else 0.6)
    edges_z = [0.0]
    for zc in zcs:
        edges_z += [zc + hb / 2, zc - hb / 2]
    edges_z = np.array(edges_z)
    nl = np.array(nails)
    cl = np.array(xs)
    n_fine = noise_mask(0.028, seed=p.seed + 3, lo=0.3, hi=0.7)
    n_patch = noise_mask(0.075, seed=p.seed + 4, lo=0.56, hi=0.8)
    n_dirt = noise_mask(0.09, seed=p.seed + 5, lo=0.2, hi=0.7)
    n_rim = noise_mask(0.05, seed=p.seed + 6, lo=0.35, hi=0.7)
    ph1, ph2, ph3 = r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28)
    z_drain = zcs[-1] + hb / 2                                # top of the lowest board

    def wearf(co, nrm):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        fine = n_fine(co, nrm)
        dz = np.abs(z[:, None] - edges_z[None, :]).min(axis=1)
        arris = np.clip(1.0 - (dz - 0.0035) / 0.005, 0.0, 1.0) * (y > -D - 0.003)      # boards, not the appliques
        ends = np.clip(1.0 - (W / 2 - np.abs(x) - 0.004) / 0.03, 0.0, 1.0)
        dc = np.abs(np.abs(x[:, None] - cl[None, :]) - 0.021).min(axis=1)        # cleat long edges
        cleat = np.clip(1.0 - dc / 0.004, 0.0, 1.0) * (y < -D - 0.003)
        dn = np.sqrt(((co[:, None, :] - nl[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        nail = np.clip(1.0 - (dn - 0.008) / 0.02, 0.0, 1.0)
        w = np.maximum.reduce([arris * (0.3 + 0.7 * fine), ends * (0.25 + 0.6 * fine), cleat * (0.35 + 0.65 * fine), nail * 0.8])
        w = np.maximum(w * wk, n_patch(co, nrm) * (0.2 + 0.8 * p.wear) * (0.75 if painted else 0.4))
        return np.clip(w, 0.0, 1.0)

    def grimef(co, nrm):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        low = np.clip((-H * 0.4 - z) / (H * 0.6), 0.0, 1.0) * n_dirt(co, nrm) * (0.5 + 0.5 * p.wear)
        streak = (0.5 + 0.5 * np.sin(x * 37.0 + ph1)) * (0.5 + 0.5 * np.sin(x * 83.0 + ph2)) + 0.25 * np.sin(x * 19.0 + ph3)
        streak = np.clip(streak * 1.7 - 0.25, 0.0, 1.0)
        drip = streak * np.clip((z_drain + 0.004 - z) / (hb * 0.9), 0.0, 1.0) ** 0.6 * (0.6 + 0.4 * p.wear)
        rim = np.clip(1.0 - (-z) / 0.03, 0.0, 1.0) * n_rim(co, nrm) * 0.7
        return np.clip(np.maximum.reduce([low, drip, rim]), 0.0, 1.0)

    for b in boards:
        assign(b, wood)
        paint(b, wear=wearf, grime=grimef, special=mask_mul(facing((0, -0.5, 1)), 0.3 + 0.5 * p.wear))
    # forged nails: cleats into every front board, front boards into the end boards
    nail = bolt("nail", 0.0042, 0.0022, kind="pyramid")
    no = instances_to_mesh("nails", nail, [trs(q, (90, 0, r.u(0, 90))) for q in nails], keep_base=False)
    assign(no, nail_mat)
    paint(no, special=0.5 + 0.4 * p.wear)
    soil_outline = [(-W / 2 + t, -0.004 - t), (-W / 2 + t, -D + t), (W / 2 - t, -D + t), (W / 2 - t, -0.004 - t)]
    info = dict(front_y=-D - 0.002, soil=soil_outline, soil_z=-0.024, x_half=W / 2 - t - 0.012, reach=D + 0.014, bulge=0.014)
    if lod() == 2:          # from afar: one closed box; the compost lid (_plants) is its top
        lod_drop(boards + [no], above=1)
        vs = [(sx * W / 2, y, z) for z in (0.0, -H) for (sx, y) in ((-1, -0.004), (-1, -D), (1, -D), (1, -0.004))]
        vs += [(sx * (W / 2 - t), y, 0.0) for (sx, y) in ((-1, -0.004 - t), (-1, -D + t), (1, -D + t), (1, -0.004 - t))]
        fs = [(1, 2, 6, 5), (3, 0, 4, 7), (0, 1, 5, 4), (2, 3, 7, 6), (4, 5, 6, 7)] + [(k, (k + 1) % 4, 8 + (k + 1) % 4, 8 + k) for k in range(4)]
        hull = _hull("box_far", vs, fs, (0.0, -D / 2, -H / 2), grain='X')
        assign(hull, wood)
        info["lid"] = vs[8:]
    return info


def _box_terracotta(p, r, W, D, H, clay):
    taper = min(0.03, D * 0.14)
    zb0 = -0.052
    prof = [(taper + 0.007, -H), (taper, -H + 0.007), (0.014, zb0), (0.008, zb0 + 0.005), (0.008, zb0 + 0.011), (0.012, zb0 + 0.015),
            (0.0, -0.033), (0.0, -0.005), (0.004, 0.0), (0.013, 0.0), (0.017, -0.004), (0.019, -0.034)]
    Dd = D - 0.004
    cy = -(D + 0.004) / 2
    rings = []
    for (ins, z) in prof:
        rr = rounded_rect(W - 2 * ins, Dd - 2 * ins, max(0.006, 0.03 - ins), 3)
        rings.append([(x, y + cy, z) for (x, y) in rr])
    m = len(rings[0])
    verts = [v for ring in rings for v in ring]
    faces = [tuple(reversed(range(m)))]
    for j in range(len(rings) - 1):
        for k in range(m):
            k1 = (k + 1) % m
            faces.append((j * m + k, j * m + k1, (j + 1) * m + k1, (j + 1) * m + k))
    body = mesh_object("trough", verts, faces)
    smooth(body, 35)
    densify(body, 0.3)
    displace_noise(body, strength=0.002, scale=0.25, seed=p.seed)
    objs = [body]

    def yf(z):      # outer front face at height z
        t = clamp((z - (-H + 0.007)) / (zb0 - (-H + 0.007)))
        return -D + lerp(taper, 0.014, t)

    # swag relief between rosettes
    nsw = max(2, int(round(W / 0.3)))
    Wf = W - 0.15
    zn = zb0 - 0.022
    sag = min(0.055, (H - 0.09) * 0.75)
    for k in range(nsw):
        xa, xb_ = -Wf / 2 + k * Wf / nsw, -Wf / 2 + (k + 1) * Wf / nsw
        pts = []
        for i in range(8):
            t = i / 7
            z = zn - sag * sin(pi * t)
            pts.append((lerp(xa, xb_, t), yf(z) - 0.0005, z))
        objs.append(sweep(f"swag{k}", pts, 0.0065, sides=6, smooth_path=False, cap=False))
    ros = lathe("rosette", [(0.021, -0.003), (0.021, 0.004), (0.013, 0.007), (0.007, 0.012), (0.0, 0.013)], segments=8, rot=(90, 0, 0))
    apply_transform(ros)
    objs.append(instances_to_mesh("rosettes", ros, [trs((-Wf / 2 + k * Wf / nsw, yf(zn), zn)) for k in range(nsw + 1)], keep_base=False))
    for o in objs:
        assign(o, clay)
        paint(o, grime=mask_mul(height_ramp(-H - 0.01, -H * 0.3), noise_mask(0.1, seed=p.seed + 5, lo=0.2, hi=0.7), 0.5 + 0.5 * p.wear),
              special=mask_mul(height_ramp(-H - 0.02, -0.02), noise_mask(0.07, seed=p.seed + 9, lo=0.4, hi=0.75), 0.3 + 0.7 * p.wear))
    i_in = 0.02
    soil_outline = [(-W / 2 + i_in, -0.004 - i_in), (-W / 2 + i_in, -D + i_in), (W / 2 - i_in, -D + i_in), (W / 2 - i_in, -0.004 - i_in)]
    info = dict(front_y=-D, soil=soil_outline, soil_z=-0.025, x_half=W / 2 - 0.034, reach=D - taper + 0.012, bulge=0.004)
    if lod() == 2:          # from afar: tapered body and upright rim collar, square in plan; the compost lid is its top
        lod_drop(objs, above=1)

        def rr_(ins, z):
            return [(sx * (W / 2 - ins), cy + sy * (Dd / 2 - ins), z) for (sx, sy) in ((-1, 1), (-1, -1), (1, -1), (1, 1))]
        vs = rr_(0.019, 0.0) + rr_(0.0, 0.0) + rr_(0.0, -0.034) + rr_(taper, -H)
        fs = [(j * 4 + k, j * 4 + (k + 1) % 4, (j + 1) * 4 + (k + 1) % 4, (j + 1) * 4 + k) for j in range(3) for k in range(4)]
        hull = _hull("trough_far", vs, fs + [(12, 13, 14, 15)], (0.0, cy, -H / 2))
        assign(hull, clay)
        info["lid"] = vs[:4]
    return info


def _box_basket(p, r, W, D, H, iron, liner_mat):
    def hw(t):
        return W / 2 - 0.045 * t * t + 0.006 * sin(pi * t)

    def dd(t):
        return D * (1.0 + 0.10 * sin(pi * t * 0.9) - 0.42 * t ** 2.2)

    def rc(t):
        return lerp(0.05, 0.032, t)

    def zz(t):
        return -0.009 - (H - 0.014) * t

    objs = []
    # rails: flat top rail, round mid and bottom rails (the bottom one closes along the wall)
    top = [(x, y, zz(0) - 0.001) for (x, y) in _u_outline(hw(0), dd(0), rc(0), yb=-0.002)]
    objs.append(sweep("rail_top", top, rounded_rect(0.005, 0.02, 0.0012, 1), smooth_path=False))
    mid = [(x, y, zz(0.5)) for (x, y) in _u_outline(hw(0.5) + 0.003, dd(0.5) + 0.003, rc(0.5), yb=-0.002)]
    objs.append(sweep("rail_mid", mid, 0.004, sides=5, smooth_path=False))
    bot = [(x, y, zz(1)) for (x, y) in _u_outline(hw(1), dd(1), rc(1), yb=-0.006)]
    objs.append(sweep("rail_bot", bot, 0.0046, sides=5, smooth_path=False, closed=True))
    # bars
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    nf = max(3, int(round((W - 0.1) / 0.058)))
    ne = max(1, int(round((D - 0.05) / 0.07)))

    def bar(fn, name):
        pts = [fn(t) for t in levels]
        pts[0] = (pts[0][0], pts[0][1], 0.004)        # bar tops stand a little proud of the rail
        b = sweep(name, pts, 0.0036, sides=3, smooth_path=False)
        return b

    for k in range(nf):
        fx = -1 + 2 * (k + 0.5) / nf
        objs.append(bar(lambda t, fx=fx: (fx * (hw(t) - rc(t)), -dd(t), zz(t)), f"bar_f{k}"))
    c45 = cos(radians(45))
    for sx in (-1, 1):
        objs.append(bar(lambda t, sx=sx: (sx * (hw(t) - rc(t) + rc(t) * c45), -(dd(t) - rc(t) + rc(t) * c45), zz(t)), "bar_c"))
        for j in range(ne):
            fy = (j + 0.6) / ne
            objs.append(bar(lambda t, sx=sx, fy=fy: (sx * hw(t), -0.006 - fy * (dd(t) - rc(t) - 0.006), zz(t)), "bar_e"))
        # wall lugs with a coach bolt at each end of the top rail
        objs.append(box("lug", (0.03, 0.004, 0.064), loc=(sx * hw(0), -0.002, -0.026)))
    bt = bolt("lug_bolt", 0.007, 0.004, kind="pyramid")
    objs.append(instances_to_mesh("lug_bolts", bt, [trs((sx * hw(0), -0.004, -0.046), (90, 0, 20)) for sx in (-1, 1)], keep_base=False))
    for o in objs:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.05, seed=p.seed + 31, lo=0.35, hi=0.7), 0.3 + 0.7 * p.wear),
              wear=mask_mul(noise_mask(0.04, seed=p.seed + 3, lo=0.5, hi=0.8), p.wear))
    # coir liner inside the bars, closed along the wall under the soil
    lv = [0.02, 0.2, 0.4, 0.6, 0.8, 0.97]
    rows = []
    lr = r.fork("liner")
    for i, t in enumerate(lv):
        ol = _u_outline(hw(t) - 0.006, dd(t) - 0.006, max(0.01, rc(t) - 0.006), yb=-0.003, na=3)
        rows.append([(x, y, zz(t) + (lr.jitter(0.006, 0.005) if i == 0 else 0.0)) for (x, y) in ol])
    rows.append([(x * 0.985, -0.004, zz(1.0) + 0.004) for (x, y, z) in rows[-1]])
    m = len(rows[0])
    verts = [v for row in rows for v in row]
    faces = []
    for j in range(len(rows) - 1):
        for k in range(m - 1):
            faces.append((j * m + k, (j + 1) * m + k, (j + 1) * m + k + 1, j * m + k + 1))
    liner = mesh_object("liner", verts, faces)
    densify(liner, 0.16)
    displace_noise(liner, strength=0.004, scale=0.08, seed=p.seed + 2, axis_mask=(0.3, 1, 0.3))
    transform_verts(liner, lambda co: Vector((co.x, min(co.y, -0.003), co.z)))
    smooth(liner, 60)
    assign(liner, liner_mat)
    paint(liner, grime=mask_mul(height_ramp(-H - 0.01, -H * 0.2), 0.7))
    soil_outline = _u_outline(hw(0.03) - 0.008, dd(0.03) - 0.008, rc(0) - 0.008, yb=-0.004)
    hw0, d0, rc0 = hw(0.03) - 0.008, dd(0.03) - 0.008, rc(0) - 0.008

    def soil_clip(x, y):         # keep the compost inside the rounded front corners of the liner
        cx, cy = hw0 - rc0, -(d0 - rc0)
        if abs(x) > cx and y < cy:
            vx, vy = abs(x) - cx, y - cy
            l = (vx * vx + vy * vy) ** 0.5
            if l > rc0:
                vx, vy = vx * rc0 / l, vy * rc0 / l
            return ((cx + vx) * (1 if x > 0 else -1), cy + vy)
        return (x, y)

    info = dict(front_y=-D - 0.003, soil=soil_outline, soil_z=-0.018, x_half=W / 2 - 0.03, reach=dd(1.0) + 0.016, bulge=0.024,
                soil_clip=soil_clip)
    if lod() == 2:          # from afar: the coir liner as a three-row trough (bottom closed), the top rail as a band round it
        lod_drop(objs + [liner], above=1)

        def row(t, dz=0.0):
            a, b = hw(t) + 0.002, dd(t) + 0.002         # where the bars stand
            return [(-a, -0.003, zz(t) + dz), (-a, -b, zz(t) + dz), (a, -b, zz(t) + dz), (a, -0.003, zz(t) + dz)]
        vs = row(0.02, 0.006) + row(0.5) + row(1.0)
        fs = [(j * 4 + k, j * 4 + k + 1, (j + 1) * 4 + k + 1, (j + 1) * 4 + k) for j in range(2) for k in range(3)]
        ln = _hull("liner", vs, fs + [(8, 9, 10, 11)], (0.0, -D / 2, -H / 2))
        assign(ln, liner_mat)
        a, b = hw(0) + 0.006, dd(0) + 0.006
        rv = [(sx * a, y, zz(0) + dz) for dz in (0.009, -0.011) for (sx, y) in ((-1, -0.002), (-1, -b), (1, -b), (1, -0.002))]
        rail = _hull("rail_top", rv, [(k, k + 1, k + 5, k + 4) for k in range(3)], (0.0, -D / 2, zz(0)))
        assign(rail, iron)
        bv, bf = [], []         # every second front bar and one at each end, as flat strips standing proud of the liner

        def strip(fn, wd):
            b = len(bv)
            for t in (0.0, 0.5, 1.0):
                q = Vector(fn(t))
                bv.extend([tuple(q - wd * 0.006), tuple(q + wd * 0.006)])
            bf.extend([(b, b + 1, b + 3, b + 2), (b + 2, b + 3, b + 5, b + 4)])
        nfar = max(2, nf // 2)
        for k in range(nfar):
            fx = -1 + 2 * (k + 0.5) / nfar
            strip(lambda t, fx=fx: (fx * (hw(t) - rc(t)), -dd(t) - 0.008, zz(t) + (0.013 if t == 0 else 0.0)), Vector((1, 0, 0)))
        for sx in (-1, 1):
            strip(lambda t, sx=sx: (sx * (hw(t) + 0.008), -0.006 - 0.55 * (dd(t) - rc(t) - 0.006), zz(t) + (0.013 if t == 0 else 0.0)),
                  Vector((0, 1, 0)))
        bars = _hull("bars_far", bv, bf, (0.0, -D / 2, -H / 2))
        assign(bars, iron)
        info["lid"] = vs[:4]
    return info


# ----------------------------------------------------------------------------------------- planting
PETAL = "wax"       # soft translucent petal tissue; its streaks run along V = along every petal


def _plants(p, rng, W, D, info):
    full, trail, bloom = p.fullness, p.trailing, p.blooms
    front_y, soil_z, x_half, bulge = info["front_y"], info["soil_z"], info["x_half"], info["bulge"]
    up = Vector((0, 0, 1))
    leaf_a = mat("leaf", p.foliage, rough=1.3)
    leaf_b = mat("leaf", _shade(p.foliage, 0.68), rough=1.45)
    hedge = mat("hedge", _shade(p.foliage, 0.74))
    fl_a = mat(PETAL, _petal(p.flower_a), rough=1.6)                    # three tints of the head colour, mixed
    fl_a2 = mat(PETAL, _petal(p.flower_a, 0.86), rough=1.95)             # floret by floret inside every umbel
    fl_a3 = mat(PETAL, _petal(p.flower_a, 1.0, 0.22), rough=1.3)
    fl_b = mat(PETAL, _petal(p.flower_b), rough=1.6)
    span = 2 * x_half
    dens = (span / 0.83) ** 0.8         # planting counts grow a little slower than the width
    out = []
    far, heads = lod() == 2, []

    # ---- compost: a slightly mounded, lumpy bed 2-3 cm below the rim
    r = rng.fork("soil")
    so = info["soil"]
    sx0, sx1 = min(q[0] for q in so), max(q[0] for q in so)
    syb, syf = max(q[1] for q in so), min(q[1] for q in so)
    clip = info.get("soil_clip") or (lambda x, y: (x, y))
    nxs = max(4, int(round((sx1 - sx0) / 0.075)))
    sv, sf = [], []
    for j in range(4):
        fy = j / 3
        for i in range(nxs + 1):
            fx = i / nxs
            x, y = clip(lerp(sx0, sx1, fx), lerp(syb, syf, fy))
            crown = sin(pi * fy) * min(1.0, 4 * sin(pi * fx))
            sv.append((x, y, soil_z + crown * (0.017 + r.jitter(0, 0.006))))
    for j in range(3):
        for i in range(nxs):
            a = j * (nxs + 1) + i
            sf.append((a, a + nxs + 1, a + nxs + 2, a + 1))
    soil = mesh_object("soil", sv, sf)
    smooth(soil, 60)
    soil_m = mat("soil", rough=1.3, grime=0.4)
    assign(soil, soil_m)
    out.append(soil)
    if far:                 # from afar the compost is one flat lid that closes the box
        lod_drop(soil, above=1)
        lid = _hull("soil_far", info["lid"], [(0, 1, 2, 3)], (0.0, -D / 2, -1.0))
        assign(lid, soil_m)
        out.append(lid)

    # ---- under-mass: lumpy loaf of dense growth below the leaf rosettes; at low fullness it breaks up
    # into separate clumps (and sinks under the compost in between) so the bed shows
    r = rng.fork("mass")
    ph = [r.u(0, 2 * pi) for _ in range(8)]
    xe = x_half + 0.004 + 0.02 * full
    hz = 0.032 + 0.052 * full
    yc = -D * 0.5 - 0.002
    ryb = D * 0.5 - 0.008

    def lump(x):
        return 0.5 + 0.3 * sin(x * 13.0 + ph[0]) + 0.2 * sin(x * 31.0 + ph[1])

    def grown(x):          # 0 = bare compost .. 1 = fully grown-in at this x
        return clamp((lump(x) - (0.62 - 0.6 * full)) * 4.0)

    def spill(x):
        return clamp(0.5 + 0.45 * sin(x * 14.0 + ph[2]) + 0.3 * sin(x * 33.0 + ph[3]))

    def mass_pt(u, v):
        x = lerp(xe, -xe, u)
        e = (1.0 - abs(2 * u - 1) ** 5) ** 0.5
        g = grown(x)
        phi = radians(lerp(128, -30, v))
        rz = hz * (0.7 + 0.45 * lump(x)) * (0.3 + 0.7 * e) * g
        ryf = (D * 0.5 - 0.014 + (0.018 + 0.05 * full) * spill(x)) * (0.6 + 0.4 * e) * (0.35 + 0.65 * g)
        c = cos(phi)
        y = yc - (ryf if c > 0 else ryb * (0.35 + 0.65 * g)) * c
        z = soil_z - 0.016 * (1.0 - g) + rz * sin(phi) + 0.01 * sin(x * 60 + phi * 4 + ph[4]) * e * g
        return (x * (0.93 + 0.07 * e), min(y, -0.008), z)

    def mass_frame(u, v):
        P = Vector(mass_pt(u, v))
        du = Vector(mass_pt(min(u + 0.01, 1), v)) - Vector(mass_pt(max(u - 0.01, 0), v))
        dv = Vector(mass_pt(u, min(v + 0.01, 1))) - Vector(mass_pt(u, max(v - 0.01, 0)))
        n = du.cross(dv)
        n = n.normalized() if n.length > 1e-9 else Vector((0, 0, 1))
        return P, n

    nmu = max(9, int(span / 0.055))
    if far:
        # From afar the loaf stands for the leaf rosettes as well: drawn out to where they reach, a few cross-sections
        # (wall-side foot on the compost lid, back shoulder, crest, front, tucked back to the box front), both ends closed.
        K = max(3, int(round(span / 0.2))) + 1
        zl_ = info["lid"][0][2] - 0.003
        mv, mf, ends = [], [], []
        for i in range(K):
            u = 0.04 + 0.92 * i / (K - 1)
            xs_ = [lerp(xe, -xe, clamp(u + 0.92 / (K - 1) * d)) for d in (-0.4, -0.2, 0.0, 0.2, 0.4)]
            g, lp, sp = sum(grown(x) for x in xs_) / 5, sum(lump(x) for x in xs_) / 5, sum(spill(x) for x in xs_) / 5
            cg = clamp((g - 0.15) / 0.3)        # 1 where plant crowns stand (their leaves spread wide), 0 over bare compost
            ge = max(g, 0.6 * cg)
            e = (1.0 - abs(2 * u - 1) ** 5) ** 0.5
            x = lerp(xe, -xe, u) * (0.93 + 0.07 * e)
            top = max(soil_z - 0.016 * (1.0 - g) + hz * (0.7 + 0.45 * lp) * (0.3 + 0.7 * e) * g + 0.055 * cg, zl_ + 0.045 * cg) - zl_
            ryf = (D * 0.5 - 0.014 + (0.018 + 0.05 * full) * sp + 0.045 * cg) * (0.6 + 0.4 * e) * (0.35 + 0.65 * ge)
            yb, yf = min(yc + ryb * (0.35 + 0.65 * ge) + 0.02 * cg, -0.008), yc - ryf
            mv += [(x, yb, zl_), (x, yb, zl_ + 0.9 * top), (x, yc - 0.3 * ryf, zl_ + top), (x, yf, zl_ + 0.35 * top),
                   (x, max(yf, front_y + 0.005), zl_ + min(0.3 * top, 0.0) - 0.005 - 0.06 * cg)]
            ends.append((x + (0.012 + 0.045 * cg) * (1 if x > 0 else -1), yc - 0.02, zl_ + 0.4 * top))
        for i in range(K - 1):
            for j in range(4):
                mf.append((5 * i + j, 5 * i + j + 5, 5 * i + j + 6, 5 * i + j + 1))
        mv += [ends[0], ends[-1]]
        for j in range(5):
            mf += [(5 * K, j, (j + 1) % 5), (5 * K + 1, 5 * (K - 1) + (j + 1) % 5, 5 * (K - 1) + j)]
        mass = _sheet("foliage_mass", mv, mf, [(0.3 + 0.4 * (k // 5) / (K - 1), 0.3 + 0.1 * (k % 5)) for k in range(5 * K)] + [(0.3, 0.5), (0.7, 0.5)],
                      angle=80.0)
    else:
        mass = surface("foliage_mass", mass_pt, nmu, 8)
        displace_noise(mass, strength=0.01, scale=0.035, seed=p.seed + 11)
        transform_verts(mass, lambda co: Vector((co.x, min(co.y, -0.006), co.z)))
    smooth(mass, 80)
    assign(mass, leaf_a if far else hedge)
    out.append(mass)

    def hidden(pos):       # inside the box, below the rim
        return pos.z < 0.006 and pos.y > front_y + 0.004 and abs(pos.x) < W / 2 + 0.002

    # ---- counts, scaled back together if a very wide / very full box would pass the triangle cap
    lpr = lerp(6.0, 11.0, full)
    nn_avg = (0.12 + 0.36 * trail * 0.72) / 0.0275 + 3

    def counts(k):         # k = 1 .. 0: filler leaves go first, flower heads last
        n_top = span / 0.118 * (0.4 + 0.75 * full)
        n_front = span / 0.094 * (0.3 + 0.85 * full)
        na = (n_top + 0.25 * n_front) * min(1.0, bloom * 1.3) * k
        n_top, n_front = n_top * k ** 0.9, n_front * k
        n_fill = span / 0.016 * full ** 1.5 * k ** 1.3
        nstr = 13 * dens * trail * (0.45 + 0.55 * full) * k
        nb = 11 * dens * bloom * (0.35 + 0.65 * full) * k
        est = 0.88 * ((n_top + n_front) * lpr * 10.3 + n_fill * 10.3 + na * 290         # 0.88: culled / rejected
                      + nstr * (nn_avg * 9.3 + (nn_avg - 3) * 0.4 * bloom * 18 + 18) + nb * 18)
        return est, [max(1, int(round(n_top))), int(round(n_front)), int(round(n_fill)), int(round(na)),
                     int(round(nstr)), int(round(nb))]

    room = info["tri_cap"] - nmu * 16 - nxs * 6
    kcap = 1.0
    while kcap > 0.1 and counts(kcap)[0] > room:
        kcap -= 0.02
    n_top, n_front, n_fill, na, nstr, nb = counts(kcap)[1]
    lscale = clamp(kcap ** -0.3, 1.0, 1.1)        # fewer but slightly larger leaves when capped

    leaf_ms = {"young": ([], []), "cup": ([], []), "old": ([], []), "trail": ([], [])}   # per base: (leaf_a, leaf_b)
    leaf_g = {k: ([], []) for k in leaf_ms}
    fl_ms, fl_g = ([], [], []), ([], [], [])
    bud_ms = []
    bl_ms, bl_g = [], []
    stems = []

    def add_leaf(kind, pos, ydir, zhint, L, wide, rr_, dark=0.38):
        pos = Vector(pos)
        reach_y = pos.y + L * (1.36 if kind == "trail" else 0.74)
        if reach_y > -0.006:
            if reach_y > 0.03:
                return
            pos.y -= reach_y + 0.006
        i = 1 if rr_.chance(dark) else 0
        leaf_ms[kind][i].append(_m_along(pos, ydir, zhint, (L * wide, L, L)))
        leaf_g[kind][i].append(rr_.u(0.0, 0.22))

    def add_bloom(pos, face, rad, rr_):
        pos = Vector(pos)
        pos.y = min(pos.y, -rad * 2.6 - 0.006)
        bl_ms.append(_m_facing(pos, face, rad, rr_))
        bl_g.append(rr_.u(0.0, 0.25))

    # ---- plant crowns: a top row along the crest and a staggered front row that spills over the rim.
    # Every crown carries a rosette of leaves (small cupped young ones inside, big drooping ones outside).
    r = rng.fork("rosettes")
    crowns = []
    for row, n_row, v0, v1 in (("top", n_top, 0.17, 0.36), ("front", n_front, 0.52, 0.8)):
        for i in range(n_row):
            for _ in range(6):
                u = 0.04 + 0.92 * (i + r.u(0.12, 0.88)) / n_row
                if grown(lerp(xe, -xe, u)) > 0.3:
                    break
            else:
                continue
            P, n = mass_frame(u, r.u(v0, v1))
            crowns.append((row, P + n * 0.006, n, grown(P.x)))
    for row, C, n, g in crowns:
        e1 = n.orthogonal().normalized()
        e2 = n.cross(e1)
        nl_ = max(4, int(round(lpr * r.u(0.85, 1.15) * (0.6 + 0.4 * g))))
        ph0 = r.u(0, 2 * pi)
        for k in range(nl_):
            t = (k + 0.5) / nl_
            a = ph0 + k * GOLD + r.jitter(0, 0.25)
            rad = e1 * cos(a) + e2 * sin(a)
            el = radians(lerp(60, 7, t ** 0.75) + r.jitter(0, 8))
            L = lerp(0.03, 0.074, t ** 0.85) * r.u(0.88, 1.1) * (0.9 + 0.1 * full) * lscale
            dist = lerp(0.004, 0.056, t) * (0.8 + 0.3 * full) * lscale
            dirv = rad * cos(el) + n * sin(el)
            pos = C + rad * dist + n * (0.004 + 0.02 * (1 - t) + r.u(0, 0.008))
            if hidden(pos + dirv * (L * 0.25)):
                continue
            top = n * cos(el) - rad * sin(el) + Vector((0, -0.22, 0.3))
            kind = "young" if t < 0.36 else ("old" if r.chance(0.4) else "cup")
            add_leaf(kind, pos, dirv, top, L, r.u(0.9, 1.08), r, dark=0.25 + 0.3 * t)

    # ---- filler leaves lying over the under-mass between the rosettes, shingled outwards / downwards
    for i in range(n_fill):
        u = r.u(0.03, 0.97)
        g = grown(lerp(xe, -xe, u))
        if g < 0.5:
            continue
        crest = i % 3 == 0                      # every third one stands up along the crest (breaks the outline)
        P, n = mass_frame(u, r.u(0.1, 0.3) if crest else r.u(0.25, 0.86))
        if crest:
            dirv = Vector((r.u(-0.7, 0.7), -r.u(0.0, 0.5), 1.0))
            top = Vector((r.u(-0.3, 0.3), -1.0, 0.35))
        else:
            tng = Vector((r.u(-1, 1), r.u(-0.6, 0.1), r.u(-1.0, 0.25)))
            tng = tng - n * tng.dot(n)
            if tng.length < 0.05:
                continue
            dirv = tng.normalized() + n * r.u(0.25, 0.6)
            top = n + Vector((0, -0.15, 0.25))
        L = r.u(0.036, 0.06) * (0.9 + 0.1 * full) * lscale
        pos = P + n * r.u(0.006, 0.018)
        if hidden(pos + dirv.normalized() * (L * 0.3)):
            continue
        add_leaf("cup" if r.chance(0.6) else "young", pos, dirv, top, L, r.u(0.9, 1.08), r, dark=0.5)

    # ---- upright geranium heads: a stalk from the crown carrying an umbel of 12-18 florets + drooping buds
    r = rng.fork("umbels")
    order = [c for c in crowns if c[0] == "top"] + [c for c in crowns if c[0] == "front"][::3]
    step = max(1.0, len(order) / max(na, 1))
    for i in range(min(na, len(order))):
        row, C, n, g = order[min(len(order) - 1, int(i * step))] if na < len(order) else order[i]
        axis = (n * 0.3 + Vector((r.u(-0.28, 0.28), -r.u(0.05, 0.3), 1.0))).normalized()
        Ru = r.u(0.033, 0.043)
        head = C + axis * (r.u(0.075, 0.12) * (0.8 + 0.3 * full) + (0.02 if row == "front" else 0.0))
        head.y = min(head.y, -(Ru * 1.12 + 0.008))
        midp = C.lerp(head, 0.5) + Vector((r.u(-0.008, 0.008), r.u(-0.008, 0.004), 0))
        midp.y = min(midp.y, -0.012)
        base = C - n * 0.02
        base.y = min(base.y, -0.012)
        if not far:
            stems.append(sweep(f"stalk{i}", [tuple(base), tuple(midp), tuple(head - axis * (Ru * 0.25))], 0.0021, sides=3,
                               smooth_path=False, cap=False))
        e1 = axis.orthogonal().normalized()
        e2 = axis.cross(e1)
        heads.append((head.copy(), axis.copy(), e1, e2, Ru, base.copy()))
        nf = int(r.u(12, 18.99))
        ncand = int(nf * 1.5)                  # candidates over the whole cap; the wall side is skipped
        ph0 = r.u(0, 2 * pi)
        cmax = cos(radians(98))
        main = 0 if r.chance(0.6) else 1
        placed = 0
        for k in range(ncand):
            ct = 1.0 - (k + 0.5) / ncand * (1.0 - cmax)
            st = (1.0 - ct * ct) ** 0.5
            a = ph0 + k * GOLD
            dv = axis * ct + (e1 * cos(a) + e2 * sin(a)) * st
            if dv.y > 0.25 or placed >= nf:    # faces the wall - never seen, and would bake dark
                continue
            placed += 1
            rf = Ru * r.u(0.5, 0.62)
            c_ = head + dv * (Ru * r.u(0.8, 1.0))
            face = dv + axis * 0.3 + Vector((r.u(-0.2, 0.2), r.u(-0.4, 0.05), r.u(-0.05, 0.35)))
            M = _m_facing(c_, face, rf, r)
            ti = main if r.chance(0.6) else (2 if r.chance(0.5) else 1 - main)
            fl_ms[ti].append(M)
            fl_g[ti].append(r.u(0.0, 0.14))
        for k in range(int(r.u(2, 3.99))):
            a = r.u(0, 2 * pi)
            th = radians(r.u(108, 140))
            dv = axis * cos(th) + (e1 * cos(a) + e2 * sin(a)) * sin(th)
            if dv.y > 0.3:
                dv.y = -dv.y
            bud_ms.append(_m_facing(head, dv, Ru * r.u(0.95, 1.25), r))

    # ---- trailing strands: over the rim, swinging outwards, then hanging; a leaf on a petiole at every node
    r = rng.fork("trail")
    for i in range(nstr):
        end_side = 0
        if i >= 2 and trail > 0.3 and r.chance(0.16):
            end_side = r.pick([-1, 1])
        if end_side == 0:
            o_ = Vector((0, -1, 0))
            s_ = Vector((1, 0, 0))
            x0 = -x_half + (i + r.u(0.1, 0.9)) * span / nstr
            start = Vector((x0, front_y + 0.04, soil_z + 0.012))
            rim = Vector((x0 + r.jitter(0, 0.012), front_y, 0.0))
            bl = bulge
        else:
            o_ = Vector((end_side, 0, 0))
            s_ = Vector((0, -1, 0))
            y0 = r.u(front_y + 0.06, -0.085)
            start = Vector((end_side * (W / 2 - 0.04), y0, soil_z + 0.012))
            rim = Vector((end_side * (W / 2 + 0.002), y0 + r.jitter(0, 0.008), 0.0))
            bl = 0.004
        Ls = 0.12 + 0.36 * trail * r.u(0.45, 1.0)
        nodes = [start, rim - o_ * 0.009 + up * 0.013, rim + o_ * (0.012 + bl) - up * 0.004]
        pos, s = nodes[-1].copy(), 0.0
        swing, lean, curl = r.u(0.55, 1.0), r.u(-0.13, 0.13), r.u(0.2, 0.6)
        while s < Ls:
            t = s / Ls
            st_ = lerp(0.034, 0.02, t) * r.u(0.9, 1.1)
            outw = swing * exp(-s / 0.04) + curl * max(0.0, t - 0.72) / 0.28
            lean += r.jitter(0, 0.05)
            d = (o_ * outw + s_ * lean - up).normalized()
            pos = pos + d * st_
            s += st_
            nodes.append(pos.copy())
        nn = len(nodes)
        fdir = o_ + s_ * r.pick([-0.7, 0.7]) + up * 0.3
        if far:             # the strand and its leaves as one tapering strip: rim, middle, tip
            stems.append(_ribbon(f"strand{i}", [nodes[1], nodes[2 + (nn - 3) // 2] + o_ * 0.018, nodes[-1] + o_ * 0.01], fdir, 0.013, taper=0.75, us=(0.3, 0.7)))
        else:
            stems.append(_ribbon(f"strand{i}", nodes, fdir, 0.0017))
        sg = r.pick([-1, 1])
        bloomed = False
        for k in range(2, nn):
            t = (k - 2) / max(1, nn - 3)
            q = nodes[k]
            sg = -sg
            L = lerp(0.038, 0.013, t ** 0.9) * r.u(0.9, 1.1)
            if k == nn - 1:                    # growing tip: a last small leaf continuing the strand
                ydir = (nodes[k] - nodes[k - 1]) + o_ * 0.01
            else:
                ydir = s_ * (sg * r.u(0.6, 1.0)) + o_ * r.u(0.3, 0.6) - up * r.u(0.15, 0.55)
            if end_side != 0 and q.y + ydir.normalized().y * L * 1.36 > -0.01:
                ydir = Vector((ydir.x, -abs(ydir.y), ydir.z))
            zh = o_ * 0.8 + up * 0.6 - s_ * (sg * 0.2) + Vector((r.u(-0.15, 0.15), r.u(-0.15, 0.15), 0))
            add_leaf("trail", q, ydir, zh, L, r.u(0.92, 1.12), r, dark=0.3)
            if 2 < k < nn - 1 and (r.chance(0.4 * bloom) or (not bloomed and k == nn - 2 and bloom > 0.3)):
                bloomed = True
                add_bloom(q, o_ * 0.8 + up * r.u(0.2, 0.6) - s_ * (sg * r.u(0.3, 0.7)), r.u(0.0105, 0.015), r)

    # ---- single stalked blooms nestled in the front of the planting
    r = rng.fork("blooms")
    for i in range(nb):
        u = r.u(0.04, 0.96)
        if grown(lerp(xe, -xe, u)) < 0.4:
            continue
        P, n = mass_frame(u, r.u(0.45, 0.92))
        pos = P + n * 0.04
        if hidden(pos + n * 0.02):
            continue
        add_bloom(pos, n + Vector((r.u(-0.2, 0.2), -0.35, 0.4)), r.u(0.012, 0.0165), r)

    # ---- stamp everything
    if far:                 # strand strips, one gem per flower head, every trailing / stalked bloom as a triangle
        for o in stems:
            assign(o, leaf_a)
        out += stems
        hd = _heads_far("florets_a", heads, zl_)
        if hd is not None:
            assign(hd, fl_a)
            out.append(hd)
        b = _bloom_far("bloom", 1.5)
        out.append(_stamp("bloom", b, bl_ms, fl_b, None, 70.0))
        delete(b)
        # the leaves that reach furthest forward, to either side and down hold the bounds: one triangle each
        cand = [(M, 1.36 if kind == "trail" else 0.74) for kind in leaf_ms for part in leaf_ms[kind] for M in part]
        if cand:
            tips = [M @ Vector((0.0, k_, 0.0)) for (M, k_) in cand]
            ids = range(len(tips))
            pick = {min(ids, key=lambda i: tips[i].y), min(ids, key=lambda i: tips[i].x), max(ids, key=lambda i: tips[i].x),
                    min(ids, key=lambda i: tips[i].z)}
            lv, lf, luv = [], [], []
            for i in sorted(pick):
                M, k_ = cand[i]
                lf.append((len(lv), len(lv) + 1, len(lv) + 2))
                lv += [tuple(M @ Vector(q)) for q in ((-0.42, 0.3 * k_, 0.0), (0.42, 0.3 * k_, 0.0), (0.0, k_, 0.0))]
                luv += [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
            lo_ = _sheet("leaves_far", lv, lf, luv)
            assign(lo_, leaf_a)
            out.append(lo_)
        return [o for o in out if o is not None]
    for o in stems:
        assign(o, leaf_b)
    out += stems
    bases = {"young": _leaf_round("leaf_young", 9, 0.17, 0.02, 0.012), "cup": _leaf_round("leaf_cup", 11, 0.12, 0.1, 0.016),
             "old": _leaf_round("leaf_old", 11, 0.06, 0.2, 0.02), "trail": _leaf_heart("leaf_heart")}
    for kind, b in bases.items():
        for j, m_ in enumerate((leaf_a, leaf_b)):
            out.append(_stamp(f"leaves_{kind}", b, leaf_ms[kind][j], m_, leaf_g[kind][j]))
        delete(b)
    def lum(hex_col):
        h = hex_col.lstrip("#")
        return (0.3 * int(h[0:2], 16) + 0.59 * int(h[2:4], 16) + 0.11 * int(h[4:6], 16)) / 255.0

    fb = _floret("floret", 0.0, 0.34 if lum(p.flower_a) < 0.62 else 0.08)
    for j, m_ in enumerate((fl_a, fl_a2, fl_a3)):
        out.append(_stamp("florets_a", fb, fl_ms[j], m_, fl_g[j], 70.0))
    delete(fb)
    b = _bud("bud")
    out.append(_stamp("buds", b, bud_ms, fl_a2))
    delete(b)
    ped = 1.5
    for b, m_, g in ((_floret("bloom", ped, 0.3 if lum(p.flower_b) < 0.62 else 0.06), fl_b, bl_g),
                     (_pedicel("pedicel", ped), leaf_b, None)):
        out.append(_stamp(b.name, b, bl_ms, m_, g, 70.0))
        delete(b)
    return [o for o in out if o is not None]


# ----------------------------------------------------------------------------------------- build
def build(p, rng):
    W, D, H = p.width, p.depth, p.height
    if p.style == "plank":
        wood = {"painted": mat("wood_painted", p.paint, rough=1.3), "oak": mat("wood_oak", "#8a6540"),
                "weathered": mat("wood_weathered")}[p.finish]
        iron = mat("iron_wrought")
        info = _box_plank(p, rng.fork("box"), W, D, H, wood, iron)
    elif p.style == "terracotta":
        iron = mat("iron_wrought")
        info = _box_terracotta(p, rng.fork("box"), W, D, H, mat("terracotta"))
    else:
        iron = mat("iron_painted", p.paint)
        info = _box_basket(p, rng.fork("box"), W, D, H, iron, mat("burlap", "#8a6a48"))

    xb = W / 2 - max(0.11, W * 0.17)
    br = rng.fork("brackets")
    for sx in (-1, 1):
        bo = _bracket(sx * xb, H, info["reach"], iron, br, p.wear, p.seed + (0 if sx < 0 else 7))
        if lod() == 2:
            lod_drop(bo, above=1)
            _bracket_far(sx * xb, H, info["reach"], iron)

    built = lod_memo("box_tris", _tris_in_scene)       # the count of the full model, at the far level too
    info["tri_cap"] = lerp(6500, 6820, clamp((W - 0.9) / 0.6)) - built * 1.06      # planting scales itself to fit
    _plants(p, rng.fork("plants"), W, D, info)
