"""Cluster of hand-thrown terracotta flower pots around a cafe corner: lathe-turned pots in three shapes
(collared flowerpot, Tuscan pot with rolled rim and bead, glazed belly jar) with wall thickness, soil and
drainage saucers, a rectangular herb trough on pot feet, a clipped bay / box feature plant, geraniums,
daisies, petunias, lavender, rosemary, basil, chives and thyme built from instanced leaf / petal meshes,
ivy trailing over the rims onto the paving, and one pot tipped over (spilled soil) or broken (shards).
"""
from tk import *
import math
from math import sin, cos, radians, degrees, pi, atan2, hypot, sqrt, ceil
from mathutils import Vector, Matrix

META = {"name": "Terracotta pot cluster", "class": "prop", "mount": "ground", "tri_budget": 24000,
        "focus": 0.28}
PARAMS = {
    "count": {"default": 5, "min": 3, "max": 7, "desc": "number of round pots (a trough is extra)"},
    "spread": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "0 = tight huddle, 1 = loosely scattered"},
    "size": {"default": 1.0, "min": 0.8, "max": 1.25, "desc": "overall pot size multiplier"},
    "trough": {"default": True, "desc": "include a rectangular herb trough on pot feet"},
    "trough_length": {"default": 0.62, "min": 0.45, "max": 0.9, "desc": "trough length (m)"},
    "feature": {"default": "bay_standard", "choices": ["bay_standard", "box_ball", "none"],
                "desc": "plant in the largest pot: bay lollipop tree, clipped box ball, or ordinary flowers"},
    "planting": {"default": "mixed", "choices": ["mixed", "flowers", "herbs"], "desc": "what grows in the smaller pots"},
    "mishap": {"default": "tipped", "choices": ["tipped", "broken", "both", "none"], "desc": "one pot tipped over and / or broken"},
    "glazed_ratio": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "share of pots that are glazed"},
    "glaze": {"default": "#1f4670", "desc": "glaze colour"},
    "clay": {"default": "#d8ad8e", "desc": "terracotta tint (multiplies the pale fired-clay base)"},
    "saucers": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "share of pots standing in a drainage saucer"},
    "flower_a": {"default": "#c9242b", "desc": "flower colour A (geraniums)"},
    "flower_b": {"default": "#9d7fe6", "desc": "flower colour B (petunias, lavender, chive heads)"},
    "flower_c": {"default": "#f3eee0", "desc": "flower colour C (daisies)"},
    "flower_eye": {"default": "#e9b320", "desc": "flower centre colour"},
    "foliage": {"default": "#5d8a3e", "desc": "leaf colour"},
    "ivy": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "amount of trailing ivy"},
    "spares": {"default": 2, "min": 0, "max": 3, "desc": "stack of nested empty spare pots"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "algae / moss on the clay (age and damp)"},
}
VARIANTS = {
    "glazed_blue": {"count": 4, "glazed_ratio": 1.0, "feature": "box_ball", "planting": "flowers", "mishap": "none",
                    "flower_a": "#f0e9da", "flower_b": "#e9a91c", "flower_c": "#f4f0e6", "spread": 0.15, "trough": False,
                    "saucers": 0.3, "ivy": 0.35, "moss": 0.15, "seed": 4},
    "herb_corner": {"count": 3, "feature": "none", "planting": "herbs", "mishap": "broken", "glazed_ratio": 0.0,
                    "trough_length": 0.85, "saucers": 1.0, "ivy": 0.2, "moss": 0.6, "size": 0.9, "seed": 11},
    "abundant": {"count": 7, "spread": 0.55, "planting": "flowers", "mishap": "both", "glazed_ratio": 0.45,
                 "glaze": "#2f5f47", "flower_a": "#e0568c", "flower_b": "#e8822a", "flower_c": "#f6f0dc",
                 "ivy": 1.0, "size": 1.1, "seed": 23},
    "mossy_old": {"count": 5, "spread": 0.8, "trough": False, "feature": "box_ball", "planting": "mixed", "mishap": "broken",
                  "glazed_ratio": 0.2, "glaze": "#7d6a2a", "clay": "#c9a48c", "moss": 1.0, "saucers": 0.2,
                  "flower_a": "#b02a55", "flower_b": "#5a6fd0", "foliage": "#4c7a3a", "size": 1.2, "seed": 37},
}

T = 0.25                      # 'leaf' texture tile (m): every leaf / petal base mesh spans one tile
Z = Vector((0.0, 0.0, 1.0))


# ------------------------------------------------------------------ low-level helpers
def uv_mesh(name, verts, faces, uvs):
    """Mesh with explicit per-vertex UVs in metres."""
    o = mesh_object(name, verts, faces)
    me = o.data
    layer = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            layer.data[li].uv = uvs[me.loops[li].vertex_index]
    uv_keep(o)
    return o


def frame(loc, ydir, normal, s=1.0):
    """Instance matrix: local +Y along ydir, local +Z towards normal, uniform scale s."""
    y = Vector(ydir)
    if y.length < 1e-6:
        y = Vector((0, 1, 0))
    y.normalize()
    z = Vector(normal)
    z = z - y * z.dot(y)
    if z.length < 1e-5:
        z = y.orthogonal()
    z.normalize()
    x = y.cross(z)
    return Matrix(((x.x * s, y.x * s, z.x * s, loc[0]), (x.y * s, y.y * s, z.y * s, loc[1]),
                   (x.z * s, y.z * s, z.z * s, loc[2]), (0, 0, 0, 1)))


def tubes(name, specs):
    """Many tapered tubes in one mesh. specs: [(path, r0, r1, sides), ...]"""
    verts, faces, uvs = [], [], []
    for path, r0, r1, sides in specs:
        pts = [Vector(q) for q in path]
        n = len(pts)
        if n < 2:
            continue
        base = len(verts)
        dist, prev = 0.0, None
        for i, q in enumerate(pts):
            t = pts[1] - pts[0] if i == 0 else (pts[-1] - pts[-2] if i == n - 1 else pts[i + 1] - pts[i - 1])
            if t.length < 1e-7:
                t = Vector((0, 0, 1))
            t.normalize()
            ref = prev if prev is not None else (Z if abs(t.z) < 0.9 else Vector((1, 0, 0)))
            x = ref - t * ref.dot(t)
            if x.length < 1e-6:
                x = t.orthogonal()
            x.normalize()
            y = t.cross(x)
            prev = x
            if i:
                dist += (pts[i] - pts[i - 1]).length
            r = r0 + (r1 - r0) * i / (n - 1)
            for k in range(sides):
                a = 2 * pi * k / sides
                verts.append(q + (x * cos(a) + y * sin(a)) * r)
                # stay inside the blade drawn in the middle of the leaf tile
                uvs.append((0.08 + (k / sides - 0.5) * 0.02, 0.08 + 0.09 * abs(((dist / 0.09) % 2.0) - 1.0)))
        for i in range(n - 1):
            for k in range(sides):
                a = base + i * sides + k
                b = base + i * sides + (k + 1) % sides
                faces.append((a, b, b + sides, a + sides))
        faces.append(tuple(base + (n - 1) * sides + k for k in range(sides)))
    if not verts:
        return None
    o = uv_mesh(name, verts, faces, uvs)
    recalc_normals(o)
    smooth(o, 70)
    return o


class Bucket:
    """Collects instance matrices (per base-mesh key) and tube specs; merge() re-frames them."""

    def __init__(self):
        self.d = {}
        self.objs = []

    def add(self, key, item):
        self.d.setdefault(key, []).append(item)

    def merge(self, other, xf):
        for k, lst in other.d.items():
            for it in lst:
                if isinstance(it, Matrix):
                    self.add(k, xf @ it)
                else:
                    path, r0, r1, sides = it
                    self.add(k, ([xf @ Vector(q) for q in path], r0, r1, sides))
        for o in other.objs:
            transform_verts(o, lambda co: xf @ co)
            self.objs.append(o)


# ------------------------------------------------------------------ leaf / petal base meshes (one texture tile each)
def base_mesh(kind):
    if kind == "leaf_round":            # scalloped, cupped pelargonium leaf
        out = [(0, 0.0), (0.36, 0.08), (0.5, 0.45), (0.36, 0.86), (0, 1.0), (-0.36, 0.86), (-0.5, 0.45), (-0.36, 0.08)]
        v = [(x * T, y * T, (-0.10 * y * y - 0.06 * abs(x)) * T) for x, y in out] + [(0, 0.45 * T, 0.07 * T)]
        f = [(i, (i + 1) % 8, 8) for i in range(8)]
        uv = [((x + 0.5) * T, y * T) for x, y in out] + [(0.5 * T, 0.45 * T)]
    elif kind == "leaf_oval":           # folded along the midrib
        pts = [(0, 0, 0), (0.2, 0.3, 0.05), (0.2, 0.68, 0.035), (0, 1, -0.08), (-0.2, 0.68, 0.035), (-0.2, 0.3, 0.05)]
        v = [(x * T, y * T, z * T) for x, y, z in pts]
        f = [(0, 1, 2, 3), (0, 3, 4, 5)]
        uv = [((x / 0.42 + 0.5) * T, y * T) for x, y, z in pts]
    elif kind == "needle":
        pts = [(0, 0), (0.09, 0.4), (0, 1), (-0.09, 0.4)]
        v = [(x * T, y * T, 0.03 * T * (1 - abs(x) * 11)) for x, y in pts]
        f = [(0, 1, 2, 3)]
        uv = [((x * 4 + 0.5) * T, y * T) for x, y in pts]
    elif kind == "blade":               # narrow drooping grass blade, both faces modelled (leaf shader is one-sided)
        st = ((0.0, 0.03, 0.0), (0.35, 0.034, 0.0), (0.7, 0.026, -0.035), (1.0, 0.004, -0.12))
        v, uv, f = [], [], []
        for side in (0, 1):
            for yy, w, zz in st:
                for sx in (-1, 1):
                    v.append((sx * w * T * 0.5, yy * T, (zz + (0.004 if side else 0.0)) * T))
                    uv.append(((0.5 + sx * 0.12 - 0.2) * T, (0.15 + 0.7 * yy) * T))
            b = side * 8
            for j in range(3):
                q = (b + 2 * j, b + 2 * j + 1, b + 2 * j + 3, b + 2 * j + 2)
                f.append(q if side else q[::-1])
    elif kind == "ivy":                 # five-lobed ivy leaf
        out = [(0, 0.08), (0.4, 0.0), (0.3, 0.36), (0.5, 0.6), (0.2, 0.66), (0, 1.0), (-0.2, 0.66), (-0.5, 0.6), (-0.3, 0.36), (-0.4, 0.0)]
        v = [(x * T, y * T, -0.07 * abs(x) * T - 0.05 * y * y * T) for x, y in out] + [(0, 0.42 * T, 0.05 * T)]
        f = [(i, (i + 1) % 10, 10) for i in range(10)]
        uv = [((x + 0.5) * T, y * T) for x, y in out] + [(0.5 * T, 0.42 * T)]
    elif kind == "ros":                 # five-petalled cupped floret, facing +Z
        out = []
        for k in range(5):
            a = radians(k * 72.0)
            for da, rr, zz in ((-17, 0.5, 0.07), (17, 0.5, 0.07), (36, 0.2, 0.02)):
                out.append((rr * cos(a + radians(da)), rr * sin(a + radians(da)), zz))
        v = [(x * T, y * T, z * T) for x, y, z in out] + [(0, 0, 0)]
        f = [(i, (i + 1) % 15, 15) for i in range(15)]
        uv = [((x + 0.5) * T, (y + 0.5) * T) for x, y, z in out] + [(0.5 * T, 0.5 * T)]
    elif kind == "daisy":               # ring of nine ray petals
        v, f, uv = [], [], []
        for k in range(9):
            a = radians(k * 40.0 + (k % 2) * 4)
            c, s = cos(a), sin(a)
            zz = -0.03 if k % 2 else 0.01
            for rr, w, z in ((0.09, -0.04, 0.012), (0.09, 0.04, 0.012), (0.5, 0.075, zz), (0.5, -0.075, zz)):
                x, y = rr * c - w * s, rr * s + w * c
                v.append((x * T, y * T, z * T))
                uv.append(((x + 0.5) * T, (y + 0.5) * T))
            f.append((4 * k + 3, 4 * k + 2, 4 * k + 1, 4 * k))
    elif kind == "eye":                 # domed flower centre
        v = [(0.14 * T * cos(radians(k * 60)), 0.14 * T * sin(radians(k * 60)), 0.012 * T) for k in range(6)] + [(0, 0, 0.075 * T)]
        f = [(k, (k + 1) % 6, 6) for k in range(6)]
        uv = [(x + 0.5 * T, y + 0.5 * T) for x, y, z in v]
    elif kind == "spike":               # flower spike along +Y
        # stacked whorls of buds: a three-sided spindle whose rings swell and pinch, each ring twisted 60 deg
        rings = ((0.13, 0.115), (0.27, 0.06), (0.42, 0.125), (0.56, 0.06), (0.7, 0.105), (0.84, 0.05))
        v = [(0, 0, 0)]
        for j, (yy, rr) in enumerate(rings):
            for k in range(3):
                a = radians(90 + j * 60 + k * 120)
                v.append((rr * T * cos(a), yy * T, rr * T * sin(a)))
        v.append((0, T, 0))
        nr = len(rings)
        f = [(0, 1 + k, 1 + (k + 1) % 3) for k in range(3)]
        for j in range(nr - 1):
            a0, b0 = 1 + j * 3, 4 + j * 3
            for k in range(3):
                k1 = (k + 1) % 3
                f.append((a0 + k, b0 + k, a0 + k1))
                f.append((a0 + k1, b0 + k, b0 + k1))
        top = 1 + nr * 3
        f += [(top - 3 + k, top, top - 3 + (k + 1) % 3) for k in range(3)]
        uv = [(x + 0.5 * T, y) for x, y, z in v]
    else:
        raise ValueError(kind)
    o = uv_mesh("base_" + kind, v, f, uv)
    if kind == "spike":
        recalc_normals(o)
    smooth(o, 75)
    return o


# ------------------------------------------------------------------ pots
def pot_profile(style, R, H):
    """Outer profile (bottom -> rim), inner profile (rim -> soil), soil level / radius. z from the pot base."""
    t = 0.007 + 0.02 * R
    if style == "collar":
        rb, e, hc = 0.64 * R, 0.011 + 0.05 * R, 0.17 * H

        def body(z):
            return rb + (R - rb) * z / H

        rc = body(H - hc)
        outer = [(rb - 0.004, 0.0), (rb, 0.004), (rc, H - hc), (rc + e * 0.8, H - hc + 0.004), (rc + e, H - hc + 0.012),
                 (R + e, H - 0.008), (R + e - 0.003, H - 0.002), (R + e - 0.008, H)]
        zs = H - 0.028 - 0.05 * R
        inner = [(R - t * 0.6 + 0.003, H), (R - t * 0.6, H - 0.005), (body(zs) - t, zs)]
    elif style == "rolled":
        rb, rr = 0.6 * R, 0.012 + 0.045 * R
        rn, zt = R - 0.45 * rr, H - 2.0 * rr - 0.001

        def body(z):
            return rb + (rn - rb) * clamp((z - 0.028) / (zt - 0.028))

        zb = 0.7 * H
        outer = [(rb + 0.004, 0.0), (rb + 0.009, 0.005), (rb + 0.009, 0.02), (rb, 0.028),
                 (body(zb - 0.012), zb - 0.012), (body(zb - 0.007) + 0.006, zb - 0.007), (body(zb + 0.003) + 0.006, zb + 0.003),
                 (body(zb + 0.009), zb + 0.009), (rn, zt)]
        for a in (-100, -55, -10, 35, 80, 125, 160):
            outer.append((R + rr * cos(radians(a)), H - rr + rr * sin(radians(a))))
        zs = H - 2.0 * rr - 0.012 - 0.03 * R
        inner = [(body(zs) - t, zs)]
    else:                               # belly jar
        rb, rm, rn = 0.6 * R, 1.2 * R, 0.92 * R
        zm, zn = 0.5 * H, 0.88 * H

        def body(z):
            if z <= zm:
                return rb + (rm - rb) * sin(0.5 * pi * z / zm) ** 0.75
            return rm + (rn - rm) * smoothstep(zm, zn, z)

        outer = [(rb - 0.003, 0.0), (rb + 0.002, 0.004)]
        outer += [(body(H * k), H * k) for k in (0.07, 0.16, 0.27, 0.39, 0.5, 0.61, 0.72, 0.81, 0.88)]
        outer += [(rn + 0.002, 0.91 * H), (R + 0.008, 0.955 * H), (R + 0.012, H - 0.006), (R + 0.010, H - 0.001), (R + 0.004, H)]
        zs = 0.84 * H
        inner = [(R - t + 0.004, H - 0.003), (rn - t, 0.9 * H), (body(zs) - t, zs)]
    return {"outer": outer, "inner": inner, "zs": zs, "rs": inner[-1][0], "t": t, "rb": rb, "body": body}


def wall_fn(outer):
    def f(z):
        best = 0.0
        for (r0, z0), (r1, z1) in zip(outer, outer[1:]):
            lo, hi = min(z0, z1), max(z0, z1)
            if lo - 1e-9 <= z <= hi + 1e-9:
                k = 0.5 if hi - lo < 1e-9 else (z - z0) / (z1 - z0)
                best = max(best, r0 + (r1 - r0) * k)
        if best == 0.0:
            best = outer[0][0] if z < outer[0][1] else outer[-1][0]
        return best
    return f


# ------------------------------------------------------------------ plants (pot-local: axis = Z, soil at zs, radius rs)
def rand_dir(r, el0=5.0, el1=89.0):
    az = r.u(0, 2 * pi)
    sz = r.u(sin(radians(el0)), sin(radians(el1)))
    cz = sqrt(max(0.0, 1 - sz * sz))
    return Vector((cz * cos(az), cz * sin(az), sz)), az


def dome_leaves(B, key, r, n, Rd, Hd, z0, size, droop=0.25, k_lo=0.72, el0=4.0):
    for _ in range(n):
        d, az = rand_dir(r, el0)
        k = r.u(k_lo, 1.0)
        pos = Vector((d.x * Rd * k, d.y * Rd * k, z0 + d.z * Hd * k))
        nrm = Vector((d.x, d.y, d.z + 0.75)) + Vector(r.vec(0.25))
        out = Vector((cos(az), sin(az), -droop + r.u(-0.3, 0.3))) + Vector((-sin(az), cos(az), 0)) * r.u(-0.7, 0.7)
        out.normalize()
        s = r.vary(size, 0.3)
        B.add(key, frame(pos - out * s * 0.5, out, nrm, s / T))


def plant_geranium(B, rs, zs, r, col):
    Rd, Hd = rs * 1.25 + 0.035, rs * 0.9 + 0.06
    dome_leaves(B, "leaf_round", r, int(clamp(rs * 330, 20, 58)), Rd, Hd, zs + 0.03, 0.042 + rs * 0.3)
    for k in range(5):
        a = k * 1.26 + r.u(0, 0.6)
        B.add("stem", ([(0.01 * cos(a), 0.01 * sin(a), zs), (Rd * 0.3 * cos(a), Rd * 0.3 * sin(a), zs + Hd * 0.5),
                        (Rd * 0.75 * cos(a), Rd * 0.75 * sin(a), zs + Hd * 0.75)], 0.004, 0.0025, 3))
    nu = int(clamp(2 + rs * 30, 3, 7))
    for i in range(nu):
        az = (i + r.u(-0.3, 0.3)) * 2 * pi / nu
        el = radians(r.u(38, 80))
        d = Vector((cos(el) * cos(az), cos(el) * sin(az), sin(el)))
        top = Vector((d.x * Rd * 0.9, d.y * Rd * 0.9, zs + 0.03 + d.z * Hd)) + d * r.u(0.045, 0.1)
        root = Vector((d.x * Rd * 0.3, d.y * Rd * 0.3, zs + 0.02 + Hd * 0.3))
        B.add("stem", ([root, (root + top) / 2 + Z * 0.02, top], 0.0024, 0.0018, 3))
        rc = 0.017 + rs * 0.1
        nf = r.pick([9, 10, 11, 12])
        fs = 0.025 + rs * 0.085
        side = d.orthogonal().normalized()
        side2 = d.cross(side)
        for j in range(nf):
            if j == 0:
                dd = d.copy()
            else:
                a2 = j * 2 * pi / (nf - 1) + r.u(-0.3, 0.3)
                dd = (d * r.u(0.15, 0.7) + side * cos(a2) + side2 * sin(a2)).normalized()
            B.add("ros_" + col, frame(top + dd * rc, dd.orthogonal(), dd + Z * 0.3, r.vary(fs, 0.15) / T))


def plant_daisy(B, rs, zs, r, col):
    Rd, Hd = rs * 1.2 + 0.03, rs * 0.8 + 0.07
    dome_leaves(B, "leaf_oval", r, int(clamp(rs * 480, 28, 72)), Rd, Hd, zs + 0.03, 0.036 + rs * 0.12)
    for _ in range(int(clamp(6 + rs * 75, 8, 20))):
        d, az = rand_dir(r, 22)
        pos = Vector((d.x * Rd, d.y * Rd, zs + 0.03 + d.z * Hd)) + d * r.u(0.012, 0.045)
        nrm = (d + Z * 0.6 + Vector(r.vec(0.2))).normalized()
        m = frame(pos, nrm.orthogonal(), nrm, r.vary(0.034 + rs * 0.06, 0.15) / T)
        B.add("daisy_" + col, m)
        B.add("eye", m)
        B.add("stem", ([pos - d * 0.06 - Z * 0.01, pos - nrm * 0.002], 0.0016, 0.0013, 3))


def plant_petunia(B, rs, zs, r, col):
    Rd, Hd = rs * 1.3 + 0.03, rs * 0.55 + 0.045
    dome_leaves(B, "leaf_oval", r, int(clamp(rs * 400, 24, 60)), Rd, Hd, zs + 0.03, 0.042 + rs * 0.15, droop=0.5)
    for _ in range(int(clamp(5 + rs * 55, 6, 16))):
        d, az = rand_dir(r, 10)
        pos = Vector((d.x * Rd, d.y * Rd, zs + 0.035 + d.z * Hd)) + d * r.u(0.0, 0.02)
        nrm = (d + Z * 0.5 + Vector(r.vec(0.25))).normalized()
        s = r.vary(0.042 + rs * 0.08, 0.15) / T
        m = frame(pos, nrm.orthogonal(), nrm, s)
        B.add("ros_" + col, m)
        B.add("eye", frame(pos + nrm * 0.002, nrm.orthogonal(), nrm, s * 0.7))


def plant_lavender(B, rs, zs, r, col):
    # grey-green cushion of narrow leaves, flower stalks standing well clear of it
    Lc = 0.07 + rs * 0.55
    for _ in range(int(clamp(rs * 560, 34, 80))):
        d, az = rand_dir(r, 12)
        base = Vector((d.x * rs * 0.7, d.y * rs * 0.7, zs))
        dv = (Vector((d.x, d.y, 0)) * r.u(0.25, 1.1) + Z * r.u(0.6, 1.2)).normalized()
        pos = base + dv * Lc * r.u(0.25, 1.0)
        B.add("needle", frame(pos, dv + Vector(r.vec(0.35)), Vector((d.x, d.y, 0.7)) + Vector(r.vec(0.3)), r.u(0.045, 0.07) / T))
    L0 = 0.15 + rs * 1.15
    for _ in range(int(clamp(rs * 210, 12, 26))):
        az = r.u(0, 2 * pi)
        lean = radians(r.u(3, 36))
        L = r.vary(L0, 0.2)
        r0 = rs * 0.55 * sqrt(r.u())
        a0 = az + r.u(-1, 1)
        base = Vector((r0 * cos(a0), r0 * sin(a0), zs + 0.01))
        dv = Vector((sin(lean) * cos(az), sin(lean) * sin(az), cos(lean)))
        mid = base + dv * L * 0.5 + Z * 0.012
        top = base + dv * L
        B.add("stem", ([base, mid, top], 0.002, 0.0015, 3))
        B.add("spike_" + col, frame(top - dv * 0.035, dv, dv.orthogonal(), r.u(0.06, 0.09) / T))


def plant_rosemary(B, rs, zs, r, col=None):
    L0 = 0.16 + rs * 1.3
    for _ in range(int(clamp(rs * 190, 11, 20))):
        az = r.u(0, 2 * pi)
        lean = radians(r.u(2, 32))
        L = r.vary(L0, 0.25)
        r0 = rs * 0.5 * sqrt(r.u())
        base = Vector((r0 * cos(az), r0 * sin(az), zs))
        dv = Vector((sin(lean) * cos(az), sin(lean) * sin(az), cos(lean)))
        top = base + dv * L + Vector((cos(az), sin(az), 0)) * L * 0.08
        mid = base.lerp(top, 0.5) + Vector(r.vec(0.012))
        B.add("wood", ([base, mid, top], 0.003, 0.0014, 4))
        side = dv.orthogonal().normalized()
        side2 = dv.cross(side)
        nn = int(L / 0.0085)
        for j in range(nn):
            tt = 0.1 + 0.9 * j / nn
            pos = (base.lerp(mid, tt * 2) if tt < 0.5 else mid.lerp(top, tt * 2 - 1))
            a2 = j * 2.4 + r.u(-0.3, 0.3)
            sd = side * cos(a2) + side2 * sin(a2)
            B.add("needle", frame(pos, dv * 0.9 + sd, sd + Z * 0.8 + Vector((cos(az), sin(az), 0)) * 0.6, r.u(0.032, 0.046) / T))


def plant_basil(B, rs, zs, r, col=None):
    Rd, Hd = rs * 1.15 + 0.03, rs * 0.9 + 0.1
    dome_leaves(B, "leaf_oval", r, int(clamp(rs * 480, 30, 70)), Rd, Hd, zs + 0.03, 0.055 + rs * 0.2, droop=0.1, k_lo=0.5)
    for k in range(4):
        a = k * 1.57 + r.u(0, 0.8)
        B.add("stem", ([(0, 0, zs), (Rd * 0.25 * cos(a), Rd * 0.25 * sin(a), zs + Hd * 0.5),
                        (Rd * 0.4 * cos(a), Rd * 0.4 * sin(a), zs + Hd * 0.9)], 0.003, 0.002, 3))


def plant_chives(B, rs, zs, r, col):
    L0 = 0.17 + rs * 0.9
    for i in range(int(clamp(rs * 420, 24, 54))):
        az = r.u(0, 2 * pi)
        lean = radians(r.u(2, 30))
        L = r.vary(L0, 0.25)
        r0 = rs * 0.45 * sqrt(r.u())
        base = Vector((r0 * cos(az + 1.0), r0 * sin(az + 1.0), zs))
        dv = Vector((sin(lean) * cos(az), sin(lean) * sin(az), cos(lean)))
        hz = Vector((cos(az), sin(az), 0))
        nin = (Z * sin(lean) - hz * cos(lean)).normalized()
        top = base + dv * L - nin * 0.12 * L
        B.add("blade", frame(base, dv, nin, L / T))
        if i % 9 == 0:
            for dd in (Z, hz, -hz, Vector((-hz.y, hz.x, 0)), Vector((hz.y, -hz.x, 0))):
                nn = (dd + Z * 0.3).normalized()
                B.add("ros_" + col, frame(top + nn * 0.008, nn.orthogonal(), nn, 0.026 / T))


def hedge_dome(name, Rd, Hd, z0, r_in, seed, seg=12):
    o = lathe(name, [(r_in, z0 - 0.035), (Rd, z0), (Rd * 0.96, z0 + Hd * 0.4), (Rd * 0.7, z0 + Hd * 0.78),
                     (Rd * 0.35, z0 + Hd * 0.96), (0.0, z0 + Hd)], segments=seg, smooth_angle=80)
    displace_noise(o, strength=min(0.012, Rd * 0.12), scale=max(0.04, Rd * 0.6), seed=seed)
    return o


def plant_thyme(B, rs, zs, r, col=None):
    Rd, Hd = rs + 0.03, rs * 0.45 + 0.03
    B.objs.append(hedge_dome("thyme", Rd, Hd, zs + 0.04, rs * 0.9, r.fork("n").u(0, 999).__int__()))
    dome_leaves(B, "leaf_oval", r, int(clamp(rs * 700, 40, 110)), Rd * 1.05, Hd * 1.07, zs + 0.04, 0.03, droop=0.0, k_lo=1.0, el0=2)


def feature_ball(B, r, zc, Rb, n_leaf, leaf):
    o = sphere("topiary", Rb, loc=(0, 0, zc), segments=16, rings=10)
    displace_noise(o, strength=Rb * 0.07, scale=Rb * 0.55, seed=int(r.u(0, 999)))
    smooth(o, 80)
    B.objs.append(o)
    for _ in range(n_leaf):
        d, az = rand_dir(r, -55)
        pos = Vector((0, 0, zc)) + d * Rb * r.u(0.93, 1.0)
        tg = d.cross(Vector(r.vec(1.0)) + Z * 0.01)
        tg = tg.normalized() if tg.length > 1e-4 else d.orthogonal()
        B.add("leaf_oval", frame(pos, tg + d * r.u(0.25, 0.9), d + Vector(r.vec(0.3)), r.vary(leaf, 0.25) / T))


PLANTS = {"geranium": plant_geranium, "daisy": plant_daisy, "petunia": plant_petunia, "lavender": plant_lavender,
          "rosemary": plant_rosemary, "basil": plant_basil, "chives": plant_chives, "thyme": plant_thyme}
SEQ = {
    "flowers": [("geranium", "a"), ("daisy", "c"), ("petunia", "b"), ("geranium", "b"), ("daisy", "c"), ("petunia", "a")],
    "herbs": [("lavender", "b"), ("rosemary", "b"), ("basil", "b"), ("chives", "b"), ("thyme", "b"), ("lavender", "b")],
    "mixed": [("geranium", "a"), ("lavender", "b"), ("daisy", "c"), ("basil", "b"), ("petunia", "b"), ("rosemary", "b")],
}


# ------------------------------------------------------------------ ivy
def ivy_vine(B, r, anchor, wall, u0, z_soil, r_soil, z_rim, hang, glen, du):
    """A vine from the soil, over the rim, down the wall (anchor(u) -> origin, outward dir; wall(z) -> offset)
    and, if it reaches the paving, wandering away over the ground."""
    o, d = anchor(u0)
    o0 = o.copy()
    wr = wall(z_rim)
    pts = [o + d * r_soil + Z * (z_soil + 0.004), o + d * (wr - 0.008) + Z * (z_rim + 0.011),
           o + d * (wr + 0.009) + Z * (z_rim + 0.002)]
    z, u = z_rim - 0.02, u0
    zend = max(z_rim - hang, 0.0)
    while z > max(zend, 0.014):
        o, d = anchor(u)
        pts.append(o + d * (wall(z) + 0.008) + Z * z)
        z -= 0.03
        u += du * r.u(0.4, 1.6)
    if zend <= 0.0 and glen > 0.02:
        o, d = anchor(u)
        pos = o + d * (wall(0.0) + 0.03)
        heading = atan2(d.y, d.x) + r.u(-0.7, 0.7)
        for _ in range(int(glen / 0.045) + 1):
            pts.append(Vector((pos.x, pos.y, 0.007)))
            pos = pos + Vector((cos(heading), sin(heading), 0)) * 0.045
            heading += r.u(-0.5, 0.5)
    path = [Vector(q) for q in resample_path(pts, step=0.04)]
    B.add("stem", (path, 0.0026, 0.0012, 3))
    n = len(path)
    for i in range(2, n):
        q = path[i]
        tan = (path[i] - path[i - 1]).normalized()
        if q.z < 0.02:
            nrm = Z + Vector(r.vec(0.3))
            sd = Z.cross(tan)
            lift = 0.006
        else:
            hz = Vector((q.x - o0.x, q.y - o0.y, 0))
            hz = hz.normalized() if hz.length > 1e-4 else d
            nrm = hz + Z * 0.45 + Vector(r.vec(0.3))
            sd = hz.cross(tan) if abs(hz.dot(tan)) < 0.95 else Vector((-hz.y, hz.x, 0))
            lift = 0.007
        sgn = 1 if i % 2 else -1
        for k in range(2 if r.chance(0.35) else 1):
            yd = tan * r.u(0.0, 0.7) + sd * sgn * (1 if k == 0 else -1) + Vector(r.vec(0.3))
            s = r.u(0.034, 0.06) * (1.0 - 0.45 * i / n)
            B.add("ivy", frame(q + nrm.normalized() * lift + sd * sgn * 0.006, yd, nrm, s / T))


# ------------------------------------------------------------------ build
def build(p, rng):
    S = p.size
    n = int(p.count)
    terra = mat("terracotta", p.clay)
    glaze = mat("ceramic_glazed", p.glaze)
    soil_m = mat("soil")
    leaf_m = mat("leaf", p.foliage, wear=0.01, grime=0.5)
    hedge_m = mat("hedge", wear=0.01, grime=0.6)
    bark_m = mat("bark")
    # petals use the matte "wax" body: the leaf texture swallows cool tints (violet turned tan)
    fcol = {k: mat("wax", c) for k, c in (("a", p.flower_a), ("b", p.flower_b), ("c", p.flower_c))}
    eye_m = mat("wax", p.flower_eye)
    seq = SEQ[p.planting]
    B = Bucket()
    clay_objs, glazed_objs, soil_objs = [], [], []

    # ---- decide the pots
    tipped_i = (4 if n >= 5 else 2) if p.mishap in ("tipped", "both") else -1
    broken_i = (3 if n >= 4 else 1) if p.mishap in ("broken", "both") else -1
    classes = ["L", "M", "S", "M", "S", "XS", "M"]
    ph_g, ph_s = rng.fork("glz").u(), rng.fork("sau").u()
    pots = []
    for i in range(n):
        r = rng.fork(f"pot{i}")
        c = classes[i]
        R = {"L": r.u(0.185, 0.21), "M": r.u(0.115, 0.145), "S": r.u(0.082, 0.1), "XS": r.u(0.06, 0.072)}[c] * S
        mis = "tipped" if i == tipped_i else ("broken" if i == broken_i else None)
        glazed = (not mis) and ((ph_g + i * 0.618034) % 1.0) < p.glazed_ratio
        if mis:
            style = "collar"
        elif glazed:
            style = "belly" if (c != "L" and r.chance(0.65)) else "rolled"
        else:
            style = "rolled" if c == "L" else r.pick(["collar", "collar", "rolled"])
        H = R * {"collar": r.u(1.65, 1.85), "rolled": r.u(1.6, 1.8), "belly": r.u(1.9, 2.15)}[style]
        prof = pot_profile(style, R, H)
        saucer = (not mis) and ((ph_s + i * 0.618034) % 1.0) < p.saucers
        rmax = max(q[0] for q in prof["outer"])
        rb_out = prof["outer"][0][0] + 0.005
        sau = None
        if saucer:
            rs0 = rb_out + 0.012
            sau = {"r0": rs0, "r1": rs0 + 0.018 + 0.07 * R, "h": 0.02 + 0.06 * R}
        if i == 0 and p.feature != "none":
            plant = (p.feature, "a")
        elif mis == "tipped":
            plant = ("basil", "b") if p.planting == "herbs" else ("geranium", "a")
        else:
            plant = seq[(i - 1) % len(seq)]
        pots.append({"i": i, "R": R, "H": H, "style": style, "prof": prof, "glazed": glazed, "mis": mis, "sau": sau,
                     "rmax": rmax, "fr": max(rmax, sau["r1"] if sau else 0.0) + 0.006, "cls": c, "plant": plant, "r": r})

    # ---- layout: tangent packing, big pots to the back, trough and small pots to the front
    lay = rng.fork("layout")
    gap = 0.004 + 0.2 * p.spread
    placed = []

    def fits(circs):
        return all(hypot(x - a, y - b) >= rr + c + gap for (x, y, rr) in circs for (a, b, c) in placed)

    def search(make_circs, bias, reach, rmin):
        best = None
        for _ in range(240):
            a, b, c = lay.pick(placed)
            ang = lay.u(0, 2 * pi)
            dist = c + rmin + gap + 0.002 + lay.u(0, reach)
            x, y = a + dist * cos(ang), b + dist * sin(ang)
            cs, extra = make_circs(x, y)
            if not fits(cs):
                continue
            cost = hypot(x * 0.72, y) - bias * y + lay.u(0, 0.25 * p.spread)
            if best is None or cost < best[0]:
                best = (cost, x, y, cs, extra)
        return best

    pots[0]["pos"], pots[0]["rot"] = (0.0, 0.0), lay.u(0, 2 * pi)
    placed.append((0.0, 0.0, pots[0]["fr"]))
    tr = None
    if p.trough:
        TL, TW, TH = p.trough_length * (0.9 + 0.1 * S), 0.2 * S, 0.17 * S
        half = TL / 2 - TW / 2
        nc = int(ceil(2 * half / (TW * 0.5))) + 1

        def tr_circs(x, y):
            rot = radians(lay.u(-24, 24))
            return [(x + cos(rot) * (-half + 2 * half * k / (nc - 1)), y + sin(rot) * (-half + 2 * half * k / (nc - 1)), TW / 2 + 0.028)
                    for k in range(nc)], rot

        b = search(tr_circs, -0.55, half, TW / 2 + 0.028)
        tr = {"pos": (b[1], b[2]), "rot": b[4]}
        placed.extend(b[3])
    order = sorted(pots[1:], key=lambda q: (q["mis"] == "tipped", -q["R"]))
    for q in order:
        if q["mis"] == "tipped":
            R, H, rb = q["R"], q["H"], q["prof"]["rb"]

            def tip_circs(x, y, R=R, H=H, rb=rb):
                az = atan2(y, x) + lay.u(-0.6, 0.6)
                cs = [(x + cos(az) * o, y + sin(az) * o, rr) for o, rr in ((0.0, rb + 0.01), (H * 0.5, R), (H, R + 0.03), (H + R, R * 0.8))]
                return cs, az

            b = search(tip_circs, -0.35, 0.05, rb + 0.01)
        else:
            fr = q["fr"]
            bias = clamp((q["R"] / S - 0.105) / 0.09, -0.4, 0.55)
            b = search(lambda x, y, fr=fr: ([(x, y, fr)], lay.u(0, 2 * pi)), bias, 0.03, fr)
        q["pos"], q["rot"] = (b[1], b[2]), b[4]
        placed.extend(b[3])
    spare = None
    if int(p.spares) > 0:
        sr = rng.fork("spares")
        sR = sr.u(0.08, 0.098) * S
        sH = sR * sr.u(1.65, 1.8)
        sprof = pot_profile("collar", sR, sH)
        sfr = max(q_[0] for q_ in sprof["outer"]) + 0.012
        b = search(lambda x, y: ([(x, y, sfr)], 0.0), -0.2, 0.04, sfr)
        placed.extend(b[3])
        spare = {"pos": (b[1], b[2]), "H": sH, "prof": sprof}
    x0 = (min(c[0] - c[2] for c in placed) + max(c[0] + c[2] for c in placed)) / 2
    y0 = (min(c[1] - c[2] for c in placed) + max(c[1] + c[2] for c in placed)) / 2

    def age_clay(o, seed):
        paint(o, grime=mask_mul(height_ramp(0.0, 0.07), 0.3),
              special=mask_mul(height_ramp(0.0, 0.1 + 0.3 * p.moss), noise_mask(0.07, seed=seed, lo=0.3, hi=0.9), min(0.8, 0.15 + 0.65 * p.moss)))
        if p.moss > 0.05:
            paint(o, special=mask_mul(facing((0.35, 1.0, 0.1)), noise_mask(0.13, seed=seed + 5, lo=0.4, hi=0.85), p.moss * 0.7))

    # ---- build every pot
    for q in pots:
        r, R, H, prof, i = q["r"], q["R"], q["H"], q["prof"], q["i"]
        seg = 28 if R > 0.15 else (22 if R > 0.09 else 18)
        zb = 0.007 if q["sau"] else 0.0
        zs, rs = prof["zs"] + zb, prof["rs"]
        mis = q["mis"]
        px, py = q["pos"][0] - x0, q["pos"][1] - y0
        if mis == "tipped":
            rb = prof["outer"][1][0]
            th = min(atan2(zq, rq - rb) for rq, zq in prof["outer"][2:] if rq > rb + 1e-4)
            xf = Matrix.Translation((px, py, rb * sin(th) + 0.001)) @ Matrix.Rotation(q["rot"], 4, 'Z') @ Matrix.Rotation(th, 4, 'Y')
        else:
            xf = Matrix.Translation((px, py, 0.0)) @ Matrix.Rotation(q["rot"], 4, 'Z')
        closed = mis is not None
        pts = [(rr, z + zb) for rr, z in prof["outer"] + prof["inner"]]
        if closed:
            pts = [(0.0, 0.006), (0.009, 0.006), (0.011, 0.0)] + pts + [(prof["rb"] - prof["t"], prof["t"] + 0.004), (0.0, prof["t"] + 0.004)]
        pot = lathe(f"pot{i}", pts, segments=seg, smooth_angle=42)
        if mis == "broken":
            ba = r.u(0, 2 * pi)
            cut = sphere("cut", R * 0.78, loc=(R * 1.0 * cos(ba), R * 1.0 * sin(ba), H * 0.8), segments=7, rings=5,
                         scale=(1.0, 1.0, 1.15))
            displace_noise(cut, strength=R * 0.2, scale=R * 0.45, seed=p.seed + 3)
            boolean(pot, cut)
            smooth(pot, 42)
        local = [pot]
        if q["glazed"]:
            assign(pot, glaze)
            assign_faces(pot, terra, lambda c, nrm: c.z < 0.024)
            glazed_objs.append(pot)
        else:
            assign(pot, terra)
            clay_objs.append(pot)
        # soil (a full root ball when the pot is broken open)
        sp = [(rs - 0.001, zs - 0.006), (rs * 0.75, zs + 0.003), (rs * 0.3, zs + 0.009), (0.0, zs + 0.011)]
        if mis == "broken":
            sp = [(0.0, prof["t"] + 0.006), (prof["rb"] - prof["t"] - 0.003, prof["t"] + 0.006), (rs - 0.003, zs - 0.006)] + sp[1:]
        so = lathe(f"soil{i}", sp, segments=16, smooth_angle=60)
        displace_noise(so, strength=0.004, scale=0.03, seed=p.seed + i)
        assign(so, soil_m)
        local.append(so)
        soil_objs.append(so)
        if q["sau"]:
            s0, s1, hs = q["sau"]["r0"], q["sau"]["r1"], q["sau"]["h"]
            sa = lathe(f"saucer{i}", [(s0 - 0.003, 0.0), (s0, 0.003), (s1, hs - 0.003), (s1 + 0.001, hs), (s1 - 0.004, hs + 0.001),
                                      (s1 - 0.007, hs - 0.002), (s0 - 0.005, 0.008), (s0 * 0.8, 0.007)], segments=seg, smooth_angle=42)
            if q["glazed"] and r.chance(0.6):
                assign(sa, glaze)
                glazed_objs.append(sa)
            else:
                assign(sa, terra)
                clay_objs.append(sa)
            local.append(sa)
        for o in local:
            transform_verts(o, lambda co: xf @ co)

        # ---- planting (pot-local, then re-framed)
        PB = Bucket()
        pr = r.fork("plant")
        kind, col = q["plant"]
        ws = wall_fn([(rr, z + zb) for rr, z in prof["outer"]])
        sau = q["sau"]

        def wall(z, ws=ws, sau=sau):
            w = ws(z)
            if sau and z < sau["h"] + 0.012:
                w = max(w, sau["r1"] + 0.002)
            return w

        if kind == "bay_standard":
            Rb = 0.2 * S * r.u(0.95, 1.08)
            trunk = 0.5 * S * r.u(0.9, 1.15)
            zc = zb + H + trunk + Rb * 0.8
            lx, ly = r.u(-0.02, 0.02), r.u(-0.02, 0.02)
            PB.add("wood", ([(0, 0, zs - 0.01), (lx * 0.6, ly * 0.6, zs + trunk * 0.35), (lx, ly, zs + trunk * 0.7), (lx * 0.5, ly * 0.5, zc - Rb * 0.5)],
                            0.019 * S, 0.012 * S, 8))
            PB.add("wood", ([(lx, ly, zs + trunk * 0.9), (lx + Rb * 0.3, ly + 0.02, zc - Rb * 0.3), (lx + Rb * 0.5, ly, zc)], 0.008, 0.004, 5))
            PB.add("wood", ([(lx, ly, zs + trunk * 0.85), (lx - Rb * 0.3, ly - 0.03, zc - Rb * 0.3), (lx - Rb * 0.45, ly - 0.05, zc)], 0.008, 0.004, 5))
            off = Bucket()
            feature_ball(off, pr, zc, Rb, 260, 0.062)
            PB.merge(off, Matrix.Translation((lx * 0.5, ly * 0.5, 0)))
            dome_leaves(PB, "ivy", pr, 14, rs * 0.9, 0.02, zs + 0.012, 0.045, droop=0.0, k_lo=0.3)
        elif kind == "box_ball":
            Rb = R * r.u(1.08, 1.2)
            PB.add("wood", ([(0, 0, zs - 0.01), (0, 0, zs + Rb * 0.5)], 0.014, 0.012, 6))
            feature_ball(PB, pr, zs + Rb * 0.86, Rb, 200, 0.032)
        elif mis == "tipped":
            PLANTS[kind](PB, rs * 0.8, zs, pr, col)
        else:
            PLANTS[kind](PB, rs, zs, pr, col)
        # ivy trailing from the feature pot (and a little from one other pot)
        nv = 0
        if p.ivy > 0.01 and not mis:
            if i == 0:
                nv = int(round(2 + 5 * p.ivy))
            elif i == 1 and p.ivy > 0.5:
                nv = 2
        ir = r.fork("ivy")
        a_front = -pi / 2 - q["rot"]                         # world -Y in pot-local terms
        for k in range(nv):
            u0 = a_front + ir.u(-1.9, 1.9)
            reach_ground = ir.chance(0.45 + 0.4 * p.ivy)
            ivy_vine(PB, ir, lambda u: (Vector((0, 0, 0)), Vector((cos(u), sin(u), 0))), wall, u0, zs, rs * 0.75, zb + H,
                     (H + 0.1) if reach_ground else ir.u(0.3, 0.85) * H, ir.u(0.1, 0.25 + 0.4 * p.ivy), ir.u(-0.12, 0.12))
        for o in PB.objs:
            assign(o, hedge_m)
        B.merge(PB, xf)

        # ---- mishap dressing
        if mis == "tipped":
            az = q["rot"]
            dv = Vector((cos(az), sin(az), 0))
            Rs, hs = R * 0.95, 0.022 + R * 0.22
            sm = lathe("spill", [(Rs, 0.0), (Rs * 0.82, hs * 0.4), (Rs * 0.5, hs * 0.8), (Rs * 0.2, hs * 0.97), (0.0, hs)], segments=14, smooth_angle=70)
            displace_noise(sm, strength=0.012, scale=0.05, seed=p.seed + 17)
            mc = Vector((px, py, 0)) + dv * (H * 0.98 + Rs * 0.5)
            mx = Matrix.Translation(mc) @ Matrix.Rotation(az, 4, 'Z') @ Matrix.Diagonal((1.35, 1.0, 1.0, 1.0))
            transform_verts(sm, lambda co: mx @ co)
            transform_verts(sm, lambda co: Vector((co.x, co.y, max(co.z, 0.0005))))
            assign(sm, soil_m)
            soil_objs.append(sm)
        if mis == "broken":
            sr = r.fork("shards")
            wa = q["rot"] + ba
            for k in range(sr.pick([3, 4])):
                w = sr.u(0.5, 1.0)
                hh = sr.u(0.5, 1.0) * R
                Rc = R * sr.u(0.85, 1.0)
                sk = sr.u(-0.3, 0.3)

                def fn(u, v, w=w, hh=hh, Rc=Rc, sk=sk):
                    a = (u - 0.5 + sk * v) * w * (1.0 - 0.75 * v ** 1.4)
                    return (Rc * sin(a), v * hh, Rc * cos(a) - Rc * cos(w * 0.5))

                sh = surface(f"shard{k}", fn, 4, 3, thickness=prof["t"])
                realize(sh)
                da = wa + sr.u(-0.9, 0.9)
                dd = fr_ = R * 1.0 + sr.u(0.06, 0.2)
                place(sh, loc=(px + cos(da) * dd, py + sin(da) * dd, 0.0), rot=(sr.u(-6, 6), 180 if sr.chance(0.3) else 0, sr.u(0, 360)))
                apply_transform(sh)
                mn, _mx = world_bbox([sh])
                dz = 0.001 - mn.z
                transform_verts(sh, lambda co, dz=dz: Vector((co.x, co.y, co.z + dz)))
                assign(sh, terra)
                clay_objs.append(sh)

    # ---- stack of nested empty spare pots (each rests with its collar on the rim below)
    if spare:
        prof = spare["prof"]
        t = prof["t"]
        dz = max(0.3 * spare["H"], 5.4 * t)
        pts = [(0.0, 0.006), (0.009, 0.006), (0.011, 0.0)] + prof["outer"] + prof["inner"] + [(prof["rb"] - t, t + 0.004), (0.0, t + 0.004)]
        for k in range(int(p.spares)):
            o = lathe(f"spare{k}", pts, segments=20, smooth_angle=42)
            xf = (Matrix.Translation((spare["pos"][0] - x0 + sr.u(-0.003, 0.003), spare["pos"][1] - y0 + sr.u(-0.003, 0.003), k * dz))
                  @ Matrix.Rotation(sr.u(0, 6.28), 4, 'Z') @ Matrix.Rotation(radians(sr.u(-2.5, 2.5)) if k else 0.0, 4, 'X'))
            transform_verts(o, lambda co: xf @ co)
            assign(o, terra)
            clay_objs.append(o)

    # ---- trough
    if tr:
        r = rng.fork("trough")
        z0 = 0.024
        tp, hc, e, tw = 0.028 * S, 0.2 * TH, 0.011, 0.012
        xf = Matrix.Translation((tr["pos"][0] - x0, tr["pos"][1] - y0, 0.0)) @ Matrix.Rotation(tr["rot"], 4, 'Z')

        def dims(z):
            k = 1.0 - clamp(z / TH)
            return TL - 2 * tp * k, TW - 2 * tp * k

        zs = TH - 0.035
        zbd = 0.4 * TH
        spec = [(0.0, -0.01, 0.03), (0.005, 0.0, 0.035), (zbd, 0.0, 0.035), (zbd + 0.004, 0.009, 0.039), (zbd + 0.013, 0.009, 0.039),
                (zbd + 0.017, 0.0, 0.035), (TH - hc, 0.0, 0.035), (TH - hc + 0.005, 1.6 * e, 0.04), (TH - hc + 0.012, 2 * e, 0.042),
                (TH - 0.006, 2 * e, 0.042), (TH - 0.001, 2 * e - 0.005, 0.04), (TH, 2 * e - 0.012, 0.037),
                (TH, -2 * tw + 0.006, 0.026), (TH - 0.005, -2 * tw, 0.024), (zs - 0.004, -2 * tw, 0.024)]
        verts, faces = [], []
        for (z, off, rad) in spec:
            w, d = dims(z)
            verts += [(x, y, z0 + z) for x, y in rounded_rect(w + off, d + off, rad, 4)]
        m = len(verts) // len(spec)
        for k in range(len(spec) - 1):
            for j in range(m):
                a, b = k * m + j, k * m + (j + 1) % m
                faces.append((a, b, b + m, a + m))
        faces.append(tuple(reversed(range(m))))
        body = mesh_object("trough", verts, faces)
        recalc_normals(body)
        smooth(body, 42)
        w, d = dims(zs)
        ts = extrude_profile("trough_soil", rounded_rect(w - 2 * tw - 0.002, d - 2 * tw - 0.002, 0.022, 4), 0.014, loc=(0, 0, z0 + zs - 0.014))
        densify(ts, 0.05)
        displace_noise(ts, strength=0.004, scale=0.03, seed=p.seed + 40)
        assign(ts, soil_m)
        soil_objs.append(ts)
        feet = [box(f"foot{k}", (0.055, dims(0)[1] - 0.03, z0 + 0.004), loc=(sx * TL * 0.3, 0, (z0 + 0.004) / 2), bevel=0.005) for k, sx in enumerate((-1, 1))]
        for o in [body] + feet:
            assign(o, terra)
            clay_objs.append(o)
        for o in [body, ts] + feet:
            transform_verts(o, lambda co: xf @ co)
        PB = Bucket()
        ns = max(2, int(round(TL / 0.2)))
        tseq = SEQ["flowers"] if p.planting == "flowers" else SEQ["herbs"]
        vr = TW / 2 - tw - 0.012
        for k in range(ns):
            sub = Bucket()
            ti = (k + int(r.u(0, 5.99))) % len(tseq) if k else (2 + p.seed) % len(tseq)
            if k and tseq[ti][0] == kind:           # never two neighbours alike
                ti = (ti + 1) % len(tseq)
            kind, col = tseq[ti]
            PLANTS[kind](sub, vr, z0 + zs, r.fork(f"p{k}"), col)
            for o in sub.objs:
                assign(o, hedge_m)
            PB.merge(sub, Matrix.Translation(((-0.5 + (k + 0.5) / ns) * (TL - 0.1), 0, 0)))
        ir = r.fork("ivy")
        nv = int(round(1 + 4 * p.ivy)) if p.ivy > 0.01 else 0

        def hw(z):
            zz = clamp(z - z0, 0.0, TH)
            return dims(zz)[1] / 2 + (e if zz > TH - hc else (0.0045 if zbd < zz < zbd + 0.017 else 0.0))

        def hl(z):
            zz = clamp(z - z0, 0.0, TH)
            return dims(zz)[0] / 2 + (e if zz > TH - hc else (0.0045 if zbd < zz < zbd + 0.017 else 0.0))

        for k in range(nv):
            reach = ir.chance(0.5 + 0.4 * p.ivy)
            hang = (TH + z0 + 0.1) if reach else ir.u(0.4, 0.9) * TH
            if k == 0:                                         # over one end
                sx = ir.pick([-1, 1])
                ivy_vine(PB, ir, lambda u, sx=sx: (Vector((0, u, 0)), Vector((sx, 0, 0))), hl, ir.u(-0.04, 0.04) * S, z0 + zs, TL / 2 - 0.04,
                         z0 + TH, hang, ir.u(0.15, 0.3 + 0.3 * p.ivy), ir.u(-0.006, 0.006))
            else:
                sy = -1 if ir.chance(0.75) else 1
                xs = (TL / 2 - 0.06) * ir.pick([-1, 1]) * ir.u(0.45, 1.0)
                ivy_vine(PB, ir, lambda u, sy=sy: (Vector((u, 0, 0)), Vector((0, sy, 0))), hw, xs, z0 + zs, TW / 2 - 0.035,
                         z0 + TH, hang, ir.u(0.1, 0.2 + 0.3 * p.ivy), ir.u(-0.012, 0.012))
        B.merge(PB, xf)

    # ---- fallen petals and leaves on the paving
    fr_ = rng.fork("fallen")
    for q in pots:
        if q["plant"][0] in ("geranium", "petunia", "daisy") and not q["mis"]:
            for _ in range(5):
                a = fr_.u(0, 2 * pi)
                dd = q["fr"] + fr_.u(0.015, 0.14)
                loc = Vector((q["pos"][0] - x0 + cos(a) * dd, q["pos"][1] - y0 + sin(a) * dd, 0.0045))
                if all(hypot(loc.x - (c[0] - x0), loc.y - (c[1] - y0)) > c[2] + 0.005 for c in placed):
                    key = ("ros_" + q["plant"][1]) if fr_.chance(0.65) else "leaf_oval"
                    B.add(key, frame(loc, (cos(a * 3), sin(a * 3), 0), Z + Vector(fr_.vec(0.12)), fr_.u(0.022, 0.035) / T))

    # ---- realise the instanced foliage
    def keep_above(o):
        def f(co):
            if co.z < 0.004:
                h = sin(co.x * 91.7 + co.y * 47.3) * 43758.5453
                return Vector((co.x, co.y, 0.004 + 0.007 * (h - math.floor(h))))
            return co
        transform_verts(o, f)

    greens = []
    for key, lst in B.d.items():
        if key in ("stem", "wood"):
            o = tubes(key, lst)
            if o is None:
                continue
            assign(o, leaf_m if key == "stem" else bark_m)
        else:
            kind, _, col = key.partition("_")
            if kind in ("ros", "daisy", "spike"):
                bm_, m_ = base_mesh(kind), fcol[col]
            elif key == "eye":
                bm_, m_ = base_mesh("eye"), eye_m
            else:
                bm_, m_ = base_mesh(key), leaf_m
            assign(bm_, m_)
            o = instances_to_mesh(key, bm_, lst, uv_jitter=False, keep_base=False)
            uv_keep(o)
            if m_ is leaf_m:
                greens.append(o)
        keep_above(o)
    for o in greens:
        paint(o, grime=mask_mul(noise_mask(0.22, seed=p.seed + 9, lo=0.35, hi=0.8), 0.3))

    # ---- weathering story: splash-back and green algae at the foot, damp north sides, chipped glaze
    for k, o in enumerate(clay_objs):
        age_clay(o, p.seed + k * 7)
    for k, o in enumerate(glazed_objs):
        paint(o, grime=mask_mul(height_ramp(0.0, 0.09), 0.7),
              wear=mask_mul(noise_mask(0.03, seed=p.seed + k * 3, lo=0.62, hi=0.8), 0.6))
    for o in soil_objs:
        paint(o, special=mask_mul(noise_mask(0.08, seed=p.seed + 1), 0.5))
