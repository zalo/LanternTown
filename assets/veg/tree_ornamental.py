"""Small ornamental street tree (flowering cherry / crab-apple type) for plaza corners. A slender
grafted stem with root flare, graft knuckle and pruning stubs grows out of a cast-iron tree grate
(or a granite-kerbed soil bed) and is held by one or two driven timber stakes with nailed belt
ties. The crown is grown branch by branch: limbs leave the stem at staggered, swollen unions,
carry secondaries and twigs and fill a round, vase, weeping or clipped-ball envelope; extra
boughs are grown into any gap so the head is balanced. Foliage is built in two layers the way a
real crown is: dense, dark inner leaf masses on the boughs and fine 6-9 cm blades (older dark
leaves inside, light new growth at the shoot tips) on every twig around them. Blossom is small
3-4 cm flowers packed in clusters of 5-12 on the shoots and over the boughs, turning the masses
pale towards full bloom. Petals / leaves litter the ground and small candle lanterns hang on wires
in clear pockets under the crown.
"""
from math import atan2, cos, degrees, exp, hypot, pi, radians, sin, sqrt

import bpy
import numpy as np
from mathutils import Matrix, Vector

from tk import *

META = {"name": "Small ornamental tree", "class": "vegetation", "mount": "ground", "focus": 0.62}
PARAMS = {
    "height": {"default": 4.0, "min": 3.0, "max": 5.0, "desc": "overall height (m)"},
    "crown": {"default": "round", "choices": ["round", "vase", "weeping", "clipped"],
              "desc": "crown shape: natural round head, upright vase, weeping standard, clipped ball (no blossom)"},
    "crown_width": {"default": 2.5, "min": 1.4, "max": 3.6, "desc": "crown diameter (m)"},
    "clear_stem": {"default": 1.9, "min": 1.2, "max": 2.4, "desc": "bare stem height under the crown (m)"},
    "trunk_girth": {"default": 0.11, "min": 0.07, "max": 0.2, "desc": "stem diameter at knee height (m)"},
    "lean": {"default": 2.5, "min": 0.0, "max": 8.0, "desc": "lean of the stem (deg)"},
    "density": {"default": 1.0, "min": 0.4, "max": 1.25, "desc": "foliage density"},
    "leaf_colour": {"default": "#8fa57e", "desc": "leaf tint (greens, autumn colours)"},
    "blossom": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "blossom amount (0 none .. 1 full bloom with few leaves)"},
    "blossom_colour": {"default": "#ffe3dc", "desc": "petal colour"},
    "litter": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "fallen petals / leaves on the ground"},
    "stakes": {"default": 1, "choices": [0, 1, 2], "desc": "number of timber stakes with belt ties"},
    "base": {"default": "grate", "choices": ["grate", "kerb"], "desc": "cast-iron tree grate or stone-kerbed soil bed"},
    "bed_size": {"default": 1.2, "min": 1.0, "max": 1.6, "desc": "side of the square tree pit (m)"},
    "lanterns": {"default": 2, "choices": [0, 1, 2, 3], "desc": "candle lanterns hung in the branches"},
    "lantern_cd": {"default": 5.0, "min": 1.0, "max": 15.0, "desc": "light intensity per lantern (cd)"},
}
VARIANTS = {
    "vase_white": {"crown": "vase", "height": 4.8, "crown_width": 3.0, "blossom": 0.95, "blossom_colour": "#fff6ee",
                   "base": "kerb", "bed_size": 1.4, "stakes": 2, "lanterns": 3, "trunk_girth": 0.13, "seed": 4},
    "weeping_pink": {"crown": "weeping", "height": 3.6, "crown_width": 2.8, "blossom": 0.7, "blossom_colour": "#f8c4d5",
                     "stakes": 1, "lanterns": 1, "lantern_cd": 6.0, "lean": 4.0, "bed_size": 1.3, "seed": 11},
    "clipped_ball": {"crown": "clipped", "height": 3.2, "crown_width": 1.6, "clear_stem": 1.7, "leaf_colour": "#869f7c",
                     "stakes": 0, "lanterns": 1, "lantern_cd": 3.5, "lean": 0.5, "bed_size": 1.0, "litter": 0.15, "seed": 23},
    "autumn_vase": {"crown": "vase", "height": 4.3, "crown_width": 2.6, "blossom": 0.0, "leaf_colour": "#f2803a",
                    "density": 0.8, "stakes": 0, "lanterns": 2, "litter": 1.0, "trunk_girth": 0.15, "lean": 5.0,
                    "base": "kerb", "seed": 37},
}

UP = Vector((0.0, 0.0, 1.0))
TILE = 0.25            # leaf material tile (m): one blade per tile
GA = radians(137.5)
PETAL = "plaster_rough"   # matte white base: takes the petal tint cleanly under amber light
TRI_TARGET = 57500     # vegetation budget is 60k


def _uvmesh(name, verts, faces, loop_uvs):
    o = mesh_object(name, verts, faces)
    o.data.uv_layers.new(name="UVMap").data.foreach_set("uv", np.asarray(loop_uvs, dtype=np.float32).ravel())
    return uv_keep(o)


def _rv(rg):
    return Vector((rg.u(-1, 1), rg.u(-1, 1), rg.u(-1, 1)))


def _unit(rg):
    z = rg.u(-1, 1)
    a = rg.u(0, 2 * pi)
    r = sqrt(max(0.0, 1 - z * z))
    return Vector((r * cos(a), r * sin(a), z))


def _dir(a, e):
    return Vector((cos(a) * cos(e), sin(a) * cos(e), sin(e)))


def _side(tg, ang_deg):
    b0 = tg.cross(UP)
    b0 = b0.normalized() if b0.length > 1e-3 else tg.orthogonal().normalized()
    return Matrix.Rotation(radians(ang_deg), 3, tg) @ b0


def _blen(pts):
    return sum((pts[i + 1] - pts[i]).length for i in range(len(pts) - 1))


def _at(br, t):
    pts, rad = br["pts"], br["rad"]
    n = len(pts) - 1
    f = clamp(t) * n
    i = min(int(f), n - 1)
    k = f - i
    return pts[i].lerp(pts[i + 1], k), (pts[i + 1] - pts[i]).normalized(), lerp(rad[i], rad[i + 1], k)


def _mix_hex(a, b, t):
    if not isinstance(a, str):
        return a
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(lerp(x, y, t))) for x, y in zip(ca, cb))


def _scene_tris():
    n = 0
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            n += sum(len(pl.vertices) - 2 for pl in o.data.polygons)
    return n


class _Tubes:
    """All bark tubes in one mesh; U wraps one bark tile around the limb, V runs along it."""

    def __init__(self):
        self.v, self.f, self.uv = [], [], []

    def tris(self):
        return sum(len(f) - 2 for f in self.f)

    def add(self, pts, radii, sides, rfn=None, cap=True):
        n = len(pts)
        if n < 2:
            return
        tg = [(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized() for i in range(n)]
        ref = UP if abs(tg[0].z) < 0.9 else Vector((1.0, 0.0, 0.0))
        nrm = tg[0].cross(ref).normalized()
        base = len(self.v)
        rm = sum(radii) / n
        vs = clamp(0.6 / (2 * pi * rm), 1.0, 7.0)
        dist, vv = 0.0, []
        for i in range(n):
            if i > 0:
                dist += (pts[i] - pts[i - 1]).length
                ax = tg[i - 1].cross(tg[i])
                if ax.length > 1e-6:
                    nrm = Matrix.Rotation(tg[i - 1].angle(tg[i]), 3, ax.normalized()) @ nrm
                nrm = (nrm - tg[i] * nrm.dot(tg[i])).normalized()
            b = tg[i].cross(nrm)
            vv.append(dist * vs)
            for k in range(sides):
                a = 2 * pi * k / sides
                r = radii[i] * (rfn(i, a) if rfn else 1.0)
                self.v.append(tuple(pts[i] + (nrm * cos(a) + b * sin(a)) * r))
        du = 0.6 / sides
        for i in range(n - 1):
            for k in range(sides):
                a0 = base + i * sides + k
                a1 = base + i * sides + (k + 1) % sides
                self.f.append((a0, a1, a1 + sides, a0 + sides))
                self.uv += [(k * du, vv[i]), ((k + 1) * du, vv[i]), ((k + 1) * du, vv[i + 1]), (k * du, vv[i + 1])]
        if cap:
            tip = len(self.v)
            self.v.append(tuple(pts[-1] + tg[-1] * radii[-1] * 0.8))
            last = base + (n - 1) * sides
            for k in range(sides):
                self.f.append((last + k, last + (k + 1) % sides, tip))
                self.uv += [(k * du, vv[-1]), ((k + 1) * du, vv[-1]), ((k + 0.5) * du, vv[-1] + radii[-1] * vs)]

    def mesh(self, name):
        return _uvmesh(name, self.v, self.f, self.uv)


class _Blobs:
    """Lumpy low-poly ellipsoids in one mesh: the dense inner leaf (or blossom) mass of a bough.
    uvk scales the metre UVs so one leaf-texture tile covers one blade-sized patch."""

    def __init__(self):
        self.v, self.f, self.uv = [], [], []

    def add(self, c, rad, rg, uvk=(1.0, 1.0), segs=7, rings=4):
        b0 = len(self.v)
        rot = rg.u(0, 2 * pi)
        self.v.append((c.x, c.y, c.z + rad[2] * rg.u(0.85, 1.05)))
        for i in range(1, rings):
            ph = pi * i / rings
            for k in range(segs):
                th = rot + 2 * pi * (k + 0.5 * (i % 2)) / segs
                j = rg.u(0.62, 1.22)
                self.v.append((c.x + rad[0] * sin(ph) * cos(th) * j, c.y + rad[1] * sin(ph) * sin(th) * j, c.z + rad[2] * cos(ph) * rg.u(0.8, 1.15)))
        self.v.append((c.x, c.y, c.z - rad[2] * rg.u(0.8, 1.0)))
        last = len(self.v) - 1
        du = 2 * pi * (rad[0] + rad[1]) * 0.5 / segs * uvk[0]
        dv = pi * rad[2] / rings * uvk[1] * 1.3
        uo, vo = rg.u(0, 1), rg.u(0, 1)

        def ring(i, k):
            return b0 + 1 + (i - 1) * segs + k % segs

        for k in range(segs):
            self.f.append((b0, ring(1, k), ring(1, k + 1)))
            self.uv += [(uo + (k + 0.5) * du, vo), (uo + k * du, vo + dv), (uo + (k + 1) * du, vo + dv)]
        for i in range(1, rings - 1):
            for k in range(segs):
                self.f.append((ring(i, k), ring(i + 1, k), ring(i + 1, k + 1), ring(i, k + 1)))
                self.uv += [(uo + k * du, vo + i * dv), (uo + k * du, vo + (i + 1) * dv),
                            (uo + (k + 1) * du, vo + (i + 1) * dv), (uo + (k + 1) * du, vo + i * dv)]
        for k in range(segs):
            self.f.append((last, ring(rings - 1, k + 1), ring(rings - 1, k)))
            self.uv += [(uo + (k + 0.5) * du, vo + rings * dv), (uo + (k + 1) * du, vo + (rings - 1) * dv), (uo + k * du, vo + (rings - 1) * dv)]

    def mesh(self, name):
        return _uvmesh(name, self.v, self.f, self.uv)


def _leaf_base(name, flat=False):
    """Ovate blade, base at the origin, tip at +Y, upper face +Z: curved outline, arched along its
    length with a drooping tip and a slight twist (no centre fold)."""

    def z(y, dz=0.0):
        return 0.0 if flat else 0.1 * sin(2.8 * y) - 0.17 * y * y + dz

    v = [(0, 0, 0), (-0.235, 0.3, z(0.3, 0.022)), (0.235, 0.3, z(0.3, -0.008)), (-0.2, 0.68, z(0.68, 0.014)),
         (0.2, 0.68, z(0.68, -0.014)), (0, 1.0, z(1.0))]
    f = [(0, 2, 1), (1, 2, 4), (1, 4, 3), (3, 4, 5)]
    uv = [((0.5 + v[i][0] / 0.5) * TILE, (0.02 + 0.96 * v[i][1]) * TILE) for tri in f for i in tri]
    return _uvmesh(name, v, f, uv)


def _flower_base(name):
    """Small open five-petalled flower seen as a slightly cupped, ruffled disc; radius 1, faces +Z."""
    zz = (0.2, 0.0, 0.24, 0.06, 0.26)
    v = [(cos(radians(90 + j * 72)), sin(radians(90 + j * 72)), zz[j]) for j in range(5)]
    f = [(0, 1, 2), (0, 2, 3), (0, 3, 4)]
    uv = [((0.5 + 0.4 * v[i][0]) * TILE, (0.5 + 0.4 * v[i][1]) * TILE) for tri in f for i in tri]
    return _uvmesh(name, v, f, uv)


def _core_base(name):
    """Lumpy octahedron, radius ~1: the closed heart of a flower cluster."""
    v = [(1.05, 0.1, 0.0), (0.0, 0.95, 0.12), (-0.9, -0.1, 0.1), (0.1, -1.05, -0.1), (0.1, 0.0, 1.0), (-0.1, 0.1, -0.92)]
    f = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (1, 0, 5), (2, 1, 5), (3, 2, 5), (0, 3, 5)]
    uv = [((0.5 + 0.3 * v[i][0]) * TILE, (0.5 + 0.3 * v[i][1]) * TILE) for tri in f for i in tri]
    return _uvmesh(name, v, f, uv)


def _petal_base(name):
    v = [(0, -0.5, 0), (0.42, 0.1, 0), (0, 0.5, 0), (-0.42, 0.1, 0)]
    f = [(0, 1, 2), (0, 2, 3)]
    uv = [(0.5 * TILE, 0.05 * TILE), (0.85 * TILE, 0.55 * TILE), (0.5 * TILE, 0.95 * TILE),
          (0.5 * TILE, 0.05 * TILE), (0.5 * TILE, 0.95 * TILE), (0.15 * TILE, 0.55 * TILE)]
    return _uvmesh(name, v, f, uv)


def _frame(q, ydir, nd, sy, sx=None):
    """4x4 with local +Y = ydir (length sy), +Z as close to nd as possible."""
    x = ydir.cross(nd)
    if x.length < 1e-4:
        x = ydir.orthogonal()
    x.normalize()
    z = x.cross(ydir).normalized()
    sx = sy if sx is None else sx
    return ((x.x * sx, ydir.x * sy, z.x * sy, q.x), (x.y * sx, ydir.y * sy, z.y * sy, q.y),
            (x.z * sx, ydir.z * sy, z.z * sy, q.z), (0.0, 0.0, 0.0, 1.0))


def _lantern(tag, w=0.1):
    """Small square candle lantern hanging from its top ring (origin), body below. Returns
    (iron objs, glass objs, flame obj, flame z, bottom z)."""
    hb = w * 1.45
    iron, glass = [], []
    iron.append(torus(tag + "_ring", major=0.011, minor=0.0022, loc=(0, 0, -0.011), rot=(90, 0, 0), major_seg=10, minor_seg=4))
    rr = w * 0.7071 + 0.02
    zr0 = -0.022 - 0.064
    iron.append(lathe(tag + "_roof", [(0.0, zr0), (rr, zr0), (rr, zr0 + 0.004), (rr * 0.42, zr0 + 0.036), (0.013, zr0 + 0.046),
                                      (0.016, zr0 + 0.057), (0.0, zr0 + 0.064)], segments=4, rot=(0, 0, 45), smooth_angle=20))
    zb1 = zr0
    zb0 = zb1 - hb
    for sx in (-1, 1):
        for sy in (-1, 1):
            iron.append(box(tag + "_post", (0.007, 0.007, hb), loc=(sx * w / 2, sy * w / 2, (zb0 + zb1) / 2)))
    for k in range(4):
        a = radians(k * 90.0)
        iron.append(box(tag + "_rail", (w, 0.006, 0.006), loc=(sin(a) * w / 2, -cos(a) * w / 2, zb1 - 0.02), rot=(0, 0, k * 90.0)))
        # CLEARANCE RULE: one glowing sheet per side, 4 mm behind the back of the rail (it was a 2 mm box of glass
        # lying inside the rail and against the posts: two sheets fighting each other and the iron)
        glass.append(lod_faces(box(tag + "_pane", (w - 0.008, 0.002, hb - 0.004), loc=(sin(a) * (w / 2 - 0.008), -cos(a) * (w / 2 - 0.008), (zb0 + zb1) / 2),
                                   rot=(0, 0, k * 90.0)), keep=(sin(a), -cos(a), 0), above=-1))
    iron.append(box(tag + "_tray", (w + 0.016, w + 0.016, 0.012), loc=(0, 0, zb0 - 0.006), bevel=0.002))
    iron.append(lathe(tag + "_drop", [(0.0, zb0 - 0.042), (0.006, zb0 - 0.035), (0.004, zb0 - 0.027), (0.012, zb0 - 0.019), (0.012, zb0 - 0.012)], segments=6))
    iron.append(cylinder(tag + "_burner", 0.012, 0.03, loc=(0, 0, zb0 + 0.015), verts=8))
    zf = zb0 + 0.03
    flame = lathe(tag + "_flame", [(0.0, zf), (0.007, zf + 0.007), (0.0085, zf + 0.016), (0.0045, zf + 0.03), (0.0, zf + 0.042)], segments=6)
    return iron, glass, flame, zf + 0.02, zb0 - 0.042


def build(p, rng):
    H = float(p.height)
    shape = p.crown
    seed = int(p.seed)
    weep, clip, vase = shape == "weeping", shape == "clipped", shape == "vase"
    blossom = 0.0 if clip else float(p.blossom)
    dens = float(p.density)
    S = float(p.bed_size)
    grate = p.base == "grate"
    n_lan = int(p.lanterns)
    n_st = int(p.stakes)

    # at most 10 materials per build: the second (new-growth) leaf tone is used when there is room
    p_pale = 0.0 if clip else smoothstep(0.5, 0.88, blossom)      # share of the inner masses that are blossom
    has_green, has_pale = clip or p_pale < 0.999, p_pale > 0.001
    n_mats = (3 + (1 if blossom > 0 else 0) + (1 if (grate or n_lan or n_st) else 0) + (1 if n_lan else 0)
              + (2 if n_st else 0) + (0 if grate else 1) + (1 if has_green else 0))
    bark = mat("bark")
    leaf_a = mat("leaf", p.leaf_colour)
    mass_m = mat("hedge", p.leaf_colour) if has_green else None
    petal_m = mat(PETAL, p.blossom_colour) if blossom > 0 else None
    leaf_b = leaf_a
    if n_mats < 10:
        leaf_b = mat("leaf", _mix_hex(p.leaf_colour, "#e4e8a2", 0.42))
        n_mats += 1
    own_flame = n_mats < 10
    iron = mat("iron_rusty")
    soil_m = mat("soil")

    # ------------------------------------------------------------------ proportions / envelope
    rt = rng.fork("trunk")
    rp = rng.fork("prim")
    r_base = p.trunk_girth * 0.5
    r_cb = r_base * 0.78
    R = p.crown_width * 0.5
    z_low = 0.0
    if clip:
        clear = min(p.clear_stem, 0.6 * H)
        R = min(R, (H - clear) * 0.5)
        rz = R * 0.95
        zc = H - rz
        z_cb = zc - rz * 0.6
        z_top = zc + rz * 0.35
        r_tip = 0.012
    elif weep:
        z_cb = clamp(p.clear_stem * 1.12, 0.52 * H, 0.7 * H)
        zc = z_cb + 0.05
        rz = H - zc
        z_low = max(0.75, min(z_cb - 0.6, 0.3 * H))
        z_top = z_cb + 0.28
        r_tip = r_cb * 0.5
    elif vase:
        z_cb = min(p.clear_stem * 0.85, 0.45 * H)
        zc = lerp(z_cb, H, 0.55)
        rz = (H - z_cb) * 0.5
        z_top = z_cb + rp.u(0.36, 0.5)
        r_tip = r_cb * 0.52
    else:
        z_cb = min(p.clear_stem, 0.55 * H)
        rz = (H - z_cb) * 0.55
        zc = H - rz
        z_top = z_cb + 0.6 * (H - z_cb)
        r_tip = 0.015
    lean_az = rt.u(0, 2 * pi)
    ld = Vector((cos(lean_az), sin(lean_az), 0.0))
    lp = Vector((-ld.y, ld.x, 0.0))
    la = np.tan(radians(p.lean)) * z_cb * 0.64
    wa, wb, wph = rt.u(0.008, 0.02), rt.u(0.005, 0.014), rt.u(0, 6.28)

    def axis_pt(z):
        u = min(z / z_cb, 1.0)
        o = ld * (la * sin(pi / 2 * u)) + lp * (wa * sin(u * 5.2 + wph) * min(1.0, z / 0.4)) + ld * (wb * sin(u * 8.0 + wph * 2) * min(1.0, z / 0.4))
        if z > z_cb:
            o += lp * (0.03 * sin((z - z_cb) * 4.0 + wph)) * min(1.0, (z - z_cb) / 0.3)
        return Vector((o.x, o.y, z))

    top = axis_pt(z_cb)
    Cx, Cy = top.x, top.y
    Cn = Vector((Cx, Cy, zc + (0.2 * rz if weep else 0.0)))
    ph1, ph2 = rt.u(0, 6.28), rt.u(0, 6.28)
    asym = 0.02 if clip else 0.055

    def fenv(q):
        dx, dy, dz = q.x - Cx, q.y - Cy, q.z - zc
        rxy = hypot(dx, dy)
        az = atan2(dy, dx)
        rk = R * (1.0 + asym * sin(2 * az + ph1) + asym * 0.7 * sin(3 * az + ph2))
        if vase:
            t = (q.z - z_cb) / (H - z_cb)
            if t < -0.08 or t > 1.0:
                return 2.0
            tt = clamp(t)
            rv = rk * (0.2 + 0.8 * tt ** 0.6)
            if tt > 0.78:
                rv *= sqrt(max(0.08, 1.0 - ((tt - 0.78) / 0.22) ** 2 * 0.85))
            return rxy / rv
        if weep:
            if q.z >= zc:
                return sqrt((rxy / rk) ** 2 + (dz / rz) ** 2)
            if q.z < z_low:
                return 2.0
            return rxy / (rk * (1.0 - 0.1 * (zc - q.z) / (zc - z_low)))
        return sqrt((rxy / rk) ** 2 + (dz / rz) ** 2)

    def reach(p0, d):
        best, s = 0.0, 0.06
        while s < 5.0:
            if fenv(p0 + d * s) <= 1.0:
                best = s
            elif best > 0.0:
                break
            s += 0.06
        return best

    def radial(q):
        o = Vector((q.x - Cx, q.y - Cy, 0.0))
        if o.length > 1e-5:
            o.normalize()
        return o

    def grow(rg, p0, d0, L, r0, r1, seg, trop, wig, zmin=None, bounce=False, collar=0.0):
        n = max(2, int(round(L / seg)))
        step = L / n
        pts = [p0.copy()]
        d = d0.normalized()
        s = 0.0
        for i in range(n):
            s += step
            d = (d + trop(pts[-1], d, s) * (step / 0.11) + _rv(rg) * wig).normalized()
            q = pts[-1] + d * step
            if i >= 2 and fenv(q) > 1.0:
                if bounce and (vase or q.z > z_low):
                    if weep and q.z >= zc:          # slide over the dome
                        o = Vector(((q.x - Cx) / (R * R), (q.y - Cy) / (R * R), (q.z - zc) / (rz * rz))).normalized()
                        d = d - o * (max(0.0, d.dot(o)) * 1.3)
                        d.z -= 0.12
                    else:
                        o = radial(q)
                        d = d - o * (max(0.0, d.dot(o)) + (0.12 if vase else 0.0))
                        d.z += 0.25 if vase else -0.35
                    d.normalize()
                    q = pts[-1] + d * step
                    if vase and q.z > H - 0.06:
                        break
                else:
                    break
            if zmin is not None and q.z < zmin and i >= 1:
                break
            pts.append(q)
        m = len(pts) - 1
        if m < 1:
            return None
        rad = [lerp(r0, r1, (i / m) ** 0.8) * (1.0 + collar * exp(-i * step / 0.045)) for i in range(m + 1)]
        return {"pts": pts, "rad": rad}

    def no_trop(q, d, s):
        return Vector((0.0, 0.0, 0.0))

    def trop2(q, d, s):
        return UP * (-0.32 if weep else (0.05 if vase else 0.10))

    # ------------------------------------------------------------------ limb plan (before the stem: unions swell it)
    az0 = rp.u(0, 2 * pi)
    plan = []                       # (z0, azimuth, elevation, girth fraction, kind)
    if weep:
        n1 = int(round(clamp(8 + R * 2.2, 9, 13)))
        for i in range(n1):
            high = i % 3 == 2       # every third limb leaves the short leader and climbs before it arches over
            z0 = rp.u(z_cb + 0.07, z_top - 0.03) if high else z_cb - rp.u(0.0, 0.07)
            plan.append((z0, az0 + i * GA + rp.u(-0.3, 0.3), radians(rp.u(60, 82) if high else rp.u(24, 60)),
                         rp.u(0.36, 0.62), (rp.u(0.5, 0.8) if high else rp.u(0.42, 0.88), rp.u(0.4, 0.7) if high else rp.u(0.32, 0.6))))
    elif vase:
        n1 = int(round(clamp(4 + R * 1.3, 5, 7)))
        span = z_top - z_cb - 0.05
        for i in range(n1):
            u = i / (n1 - 1)
            z0 = z_cb - 0.04 + span * clamp(u + rp.u(-0.5, 0.5) / (n1 - 1))
            plan.append((z0, az0 + i * GA + rp.u(-0.3, 0.3), radians(lerp(40, 64, u) + rp.u(-7, 7)),
                         lerp(0.7, 0.44, u) * rp.u(0.82, 1.15), None))
    else:
        n1 = int(round(clamp(7 + R * 3.2, 8, 13)))
        for i in range(n1):
            u = (i + rp.u(0.1, 0.7)) / n1
            plan.append((z_cb + u * (z_top - z_cb) * 0.94, az0 + i * GA + rp.u(-0.28, 0.28),
                         radians(lerp(30, 76, u ** 0.9) + rp.u(-6, 6)), rp.u(0.48, 0.62), None))
    unions = [pl[0] for pl in plan]
    k_un = 0.07 if (clip or not (vase or weep)) else 0.14

    # ------------------------------------------------------------------ stem
    tub = _Tubes()
    z_g = (z_cb - 0.07) if weep else rt.u(0.17, 0.26)
    g_amp = 0.3 if weep else 0.14

    def radius(z):
        if z <= z_cb:
            r = r_base * (1.0 - 0.22 * z / z_cb)
        else:
            r = lerp(r_cb * 0.9, r_tip, clamp((z - z_cb) / max(z_top - z_cb, 1e-3)) ** 0.8)
        r += r_base * (0.7 * exp(-z / 0.05) + 0.25 * exp(-z / 0.2))
        r += g_amp * r_base * exp(-((z - z_g) / 0.05) ** 2)
        for zu in unions:
            r += k_un * r_base * exp(-((z - zu + 0.02) / 0.035) ** 2)
        return r

    zs = [0.0, 0.012, 0.03, 0.055, 0.085, 0.12, 0.17, 0.23]
    while zs[-1] < z_top - 0.16:
        zs.append(zs[-1] + 0.115)
    for zu in unions:
        for dz in (-0.05, -0.018, 0.02):
            if 0.3 < zu + dz < z_top - 0.02 and all(abs(zu + dz - z) > 0.016 for z in zs):
                zs.append(zu + dz)
    zs.sort()
    zs.append(z_top)
    lob_ph, nlob = rt.u(0, 6.28), rt.pick([4, 5, 5])
    bump_ph = rt.u(0, 6.28)

    def trunk_rfn(i, a):
        z = zs[i]
        return (1.0 + 0.42 * exp(-z / 0.055) * (0.5 + 0.5 * sin(nlob * a + lob_ph)) ** 1.5
                + 0.05 * sin(3 * a + z * 7.0 + bump_ph) + 0.03 * sin(5 * a - z * 13.0))

    tub.add([axis_pt(z) for z in zs], [radius(z) for z in zs], 14, rfn=trunk_rfn)
    for k in range(2):              # pruning stubs left on the clean stem
        z = lerp(0.45, 0.9, (k + rt.u(0.1, 0.9)) / 2) * z_cb
        a = rt.u(0, 2 * pi)
        d = Vector((cos(a), sin(a), 0.45)).normalized()
        q = axis_pt(z)
        rr = radius(z)
        tub.add([q + d * (rr * 0.5), q + d * (rr + 0.012), q + d * (rr + 0.03)], [0.019, 0.015, 0.011], 7)

    # ------------------------------------------------------------------ limbs
    prims, secs, twigs = [], [], []

    def weep_trop(f0, g, bend, ph):
        return lambda q, d, s: UP * -(0.03 + g * clamp((fenv(q) - f0) / 0.4) ** 2) + bend * (0.03 * sin(s * 2.0 + ph))

    def vase_trop(bend, ph):
        def f(q, d, s):
            t = (q.z - z_cb) / (H - z_cb)
            return UP * (0.14 * max(0.0, 1.0 - s / 0.85)) + radial(q) * (0.012 + 0.05 * clamp((t - 0.4) / 0.3)) + bend * (0.04 * sin(s * 2.3 + ph))
        return f

    def round_trop(bend, ph):
        return lambda q, d, s: UP * 0.07 + bend * (0.03 * sin(s * 3.0 + ph))

    for (z0, a, e, gf, ex) in plan:
        d0 = _dir(a, e)
        p0 = axis_pt(z0)
        bend = Vector((-sin(a), cos(a), 0.0)) * rp.pick([-1.0, 1.0])
        ph = rp.u(0, 6.28)
        r0 = max(0.008, radius(z0) * gf)
        if weep:
            b = grow(rp, p0, d0, 5.0, r0, 0.004, 0.17, weep_trop(ex[0], ex[1], bend, ph), 0.09,
                     zmin=z_low + rp.u(0.0, 1.0), bounce=True, collar=0.55)
        elif vase:
            b = grow(rp, p0, d0, (H - 0.1 - z0) / sin(radians(74)) * rp.u(0.8, 1.06), r0, 0.005, 0.12, vase_trop(bend, ph), 0.05,
                     bounce=True, collar=0.6)
        else:
            L = reach(p0, d0) * rp.u(0.84, 0.95)
            if L < 0.25:
                continue
            b = grow(rp, p0, d0, L, r0, 0.005, 0.12, round_trop(bend, ph), 0.07, collar=0.45)
        if b:
            prims.append(b)
    # the leader carries on above the last union
    p0 = axis_pt(z_top)
    a = rp.u(0, 2 * pi)
    d0 = _dir(a, radians(rp.u(74, 86)))
    bend = Vector((-sin(a), cos(a), 0.0))
    r0 = radius(z_top) * 0.96
    if weep:
        b = grow(rp, p0, d0, 5.0, r0, 0.004, 0.17, weep_trop(rp.u(0.45, 0.65), 0.6, bend, 0.0), 0.09, zmin=z_low + rp.u(0.3, 1.0), bounce=True)
    elif vase:
        b = grow(rp, p0, d0, (H - 0.12 - z_top) * rp.u(0.92, 1.02), r0, 0.005, 0.12, vase_trop(bend, rp.u(0, 6.28)), 0.05, bounce=True)
    else:
        b = grow(rp, p0, d0, max(0.2, reach(p0, d0) * 0.93), r0, 0.005, 0.12, round_trop(bend, 0.0), 0.06)
    if b:
        prims.append(b)

    rs = rng.fork("sec")
    if not clip:
        for b in prims:
            Lb = _blen(b["pts"])
            n2 = max(2, int(Lb / (0.5 if weep else (0.27 if vase else 0.21))))
            ga = rs.u(0, 360)
            for j in range(n2):
                t = lerp(0.12 if weep else 0.2, 0.94, (j + rs.u(0.1, 0.9)) / n2)
                q, tg, r = _at(b, t)
                side = _side(tg, ga + j * 137.5)
                if weep:
                    d0 = tg * 0.6 + side * 0.75 + UP * 0.2
                    L2, zmin = rs.u(0.6, 1.7), z_low + rs.u(0.0, 0.8)
                else:
                    ang = radians(rs.u(34, 58))
                    out = (q - Cn).normalized()
                    d0 = (tg * cos(ang) + side * sin(ang) + out * 0.32 + UP * 0.1).normalized()
                    L2, zmin = min(reach(q, d0) * 0.93, Lb * (0.3 + 0.6 * (1 - t)) * rs.u(0.8, 1.2)), None
                if L2 < 0.16:
                    continue
                c = grow(rs, q, d0, L2, max(0.0055, r * 0.6), 0.0032, 0.22 if weep else 0.12, trop2, 0.12 if weep else 0.09,
                         zmin=zmin, bounce=weep, collar=0.3)
                if c:
                    secs.append(c)

    # ------------------------------------------------------------------ bough sites: an even shell inside the envelope;
    # where no branch reaches a site a new bough is grown to it, so the head has no holes or detached lobes
    rgf = rng.fork("fill")
    sites = []
    if not clip:
        z_site = (zc - 0.12) if weep else (z_cb + 0.3 * (H - z_cb) if vase else z_cb + 0.12)
        sp_site = 0.33 * clamp(sqrt(R / 1.25), 0.85, 1.2)
        for _ in range(6000):
            q = Vector((Cx + rgf.u(-1.15, 1.15) * R, Cy + rgf.u(-1.15, 1.15) * R, rgf.u(z_site, H)))
            if not (0.5 < fenv(q) < 0.76) or any((q - t).length < sp_site for t in sites):
                continue
            sites.append(q)
            if len(sites) >= 130:
                break
        cloud = [(q, b["rad"][i]) for b in prims + secs for i, q in enumerate(b["pts"])]
        for T in sites:
            best, bd = None, 1e9
            for (q, r) in cloud:
                dd = (q - T).length_squared
                if dd < bd:
                    best, bd = (q, r), dd
            dist = sqrt(bd)
            if dist > 0.24 and best is not None:
                c = grow(rgf, best[0], T - best[0], dist, clamp(best[1] * 0.6, 0.0045, 0.012), 0.003, 0.13, no_trop, 0.05, collar=0.3)
                if c:
                    secs.append(c)

    # ------------------------------------------------------------------ lanterns: hung on a wire from a limb, in a clear pocket
    keep_out = []
    CLR = 0.3

    def clear_pt(q, rad=CLR):
        for (x, y, z0, z1) in keep_out:
            zz = min(max(q.z, z0 + 0.05), z1 - 0.12)
            if (q.x - x) ** 2 + (q.y - y) ** 2 + (q.z - zz) ** 2 < rad * rad:
                return False
        return True

    if n_lan > 0:
        rl = rng.fork("lantern")
        glass_m = mat("glass_lantern", glow=1.6)
        flame_m = mat("emissive", "#ffb45e", glow=9) if own_flame else glass_m
        hosts = [(b, True) for b in prims] + [(b, False) for b in secs]
        cp = [tuple(q) + (4.0 if is_p else 1.0,) for (b, is_p) in hosts for q in b["pts"]] + [tuple(axis_pt(z)) + (4.0,) for z in zs]
        CP = np.array(cp)
        cands = []
        for hi, (b, is_p) in enumerate(hosts):
            for t in (0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76, 0.84):
                q, tg, r = _at(b, t)
                ro = hypot(q.x - Cx, q.y - Cy)
                if clip:                # a low inner limb; the lantern drops out of the underside of the ball
                    if not is_p or ro < 0.3 or ro > 0.78 * R:
                        continue
                    wl = q.z - (zc - rz * sqrt(max(0.05, 1.0 - (ro / R) ** 2))) + 0.05
                    sc = q.z + 0.3 * (q.y - Cy) / R + 3.0 * max(0.0, 0.4 - ro)
                else:
                    if r < 0.006 or ro < 0.4 or abs(tg.z) > 0.96:
                        continue
                    wl = clamp(q.z - r - 2.32, 0.2, 0.5)
                    ztop = q.z - r - 0.004 - wl
                    m = (CP[:, 2] > ztop - 0.29 - CLR) & (CP[:, 2] < ztop + 0.04) & ((CP[:, 0] - q.x) ** 2 + (CP[:, 1] - q.y) ** 2 < CLR * CLR)
                    viol = float(CP[m, 3].sum())
                    if weep:            # out on the shoulder of an arching limb, in a pocket of the skirt
                        sc = (abs(q.z - 2.75) + 1.1 * (q.y - Cy) / R + 1.6 * abs(ro / R - 0.82) + 0.15 * min(viol, 8.0)
                              + (0.0 if is_p else 0.3) + 2.0 * max(0.0, 2.3 - q.z))
                    else:
                        sc = (abs(q.z - 2.78) + 0.28 * (q.y - Cy) / R + (0.3 if ro > 0.75 * R else 0.0) + 0.5 * min(viol, 8.0)
                              + (0.0 if is_p else 0.2) + 0.3 * abs(tg.z) + 2.0 * max(0.0, 2.3 - q.z))
                cands.append((sc, hi, q, tg, r, wl))
        cands.sort(key=lambda c: c[0])
        chosen = []
        for c in cands:
            if len(chosen) >= n_lan:
                break
            a = atan2(c[2].y - Cy, c[2].x - Cx)
            if any(c[1] == o[1] or abs((a - atan2(o[2].y - Cy, o[2].x - Cx) + pi) % (2 * pi) - pi) < radians(75) for o in chosen):
                continue
            chosen.append(c)
        for i, (_, hi, q, tg, r, wl) in enumerate(chosen):
            wl *= rl.u(0.93, 1.0)
            ztop = q.z - r - 0.004 - wl
            eul = tg.to_track_quat('Z', 'Y').to_euler()
            loop = torus(f"lan{i}_loop", major=r + 0.004, minor=0.0022, loc=tuple(q), rot=tuple(degrees(e) for e in eul), major_seg=10, minor_seg=4)
            loop.data.materials.clear()
            assign(loop, iron)
            paint(loop, wear=0.7)
            ir, gl, fl, zf, zbot = _lantern(f"lan{i}")
            ir.append(sweep(f"lan{i}_wire", [(0, 0, wl + 0.002), (0.002, 0, wl * 0.5), (0, 0, -0.002)], 0.0017, sides=4))
            ir.append(torus(f"lan{i}_eye", major=0.007, minor=0.0017, loc=(0, 0, wl + 0.006), rot=(90, 0, 0), major_seg=8, minor_seg=4))
            for o in ir:
                assign(o, iron)
            for o in gl:
                assign(o, glass_m)
            assign(fl, flame_m)
            grp = [join(ir, f"lan{i}_iron"), join(gl, f"lan{i}_glass"), fl]
            yaw = rl.u(0, 90)
            for o in grp:
                place(o, loc=(q.x, q.y, ztop), rot=(0, 0, yaw))
                apply_transform(o)
            paint(grp[0], wear=0.75, special=0.15)
            part(grp, f"lantern{i}", pivot=(q.x, q.y, q.z - r), motion="swing", amp=3.0, speed=0.35)
            light(f"lantern{i}", (q.x, q.y, ztop + zf), color="#ffbf6a", intensity=float(p.lantern_cd), range=5.5, flicker=0.3,
                  shadow_near=0.14)
            keep_out.append((q.x, q.y, ztop + zbot, ztop))
            if i == 0:
                fx("moths", "moths", (q.x, q.y, ztop + zf), size=0.35)
        # boughs that would hang into a lantern are pruned off
        host_ids = {id(hosts[c[1]][0]) for c in chosen}
        print(f"DBG lanterns {len(chosen)} of {n_lan}", [tuple(round(v, 2) for v in k) for k in keep_out])

        def in_pocket(b):
            for q in b["pts"]:
                for (x, y, z0, z1) in keep_out:
                    if z0 - CLR < q.z < z1 + 0.04 and (q.x - x) ** 2 + (q.y - y) ** 2 < CLR * CLR:
                        return True
            return False

        secs = [b for b in secs if id(b) in host_ids or not in_pocket(b)]

    LAN = np.array([(x, y, z0 + 0.1) for (x, y, z0, z1) in keep_out]) if keep_out else np.zeros((0, 3))

    def lan_shade(co):              # foliage right beside a lantern is older, sooted and darker
        co = np.atleast_2d(np.asarray(co, dtype=float))
        if not len(LAN):
            return np.zeros(len(co))
        d = np.sqrt(((co[:, None, :] - LAN[None]) ** 2).sum(-1)).min(1)
        return np.clip(1.0 - (d - 0.3) / 0.55, 0.0, 1.0) * 0.45

    # ------------------------------------------------------------------ inner foliage masses on the bough sites
    rb = rng.fork("blossom")
    pads = []                       # leaf-mass lobes: (centre, (rx, ry, rz))
    masses = []                     # (site, radius) of each compound mass
    pad_k = lerp(0.62, 1.0, clamp((dens - 0.4) / 0.6)) * clamp(sqrt(R / 1.25), 0.85, 1.2)
    flat_k = 0.62 if weep else 0.8
    for T in sites:
        r = rgf.u(0.17, 0.25) * pad_k
        r = min(r, reach(T, (T - Cn).normalized()) - 0.02)
        if r < 0.1 or not has_green or rb.chance(0.85 * p_pale) or not clear_pt(T, CLR + 0.14 + r):
            continue
        masses.append((T, r))
        for k in range(7):          # several small lobes, so the mass has the grain of leaf clumps
            o = _unit(rgf)
            c2 = T + Vector((o.x, o.y, o.z * flat_k)) * (r * (0.0 if k == 0 else rgf.u(0.35, 0.8)))
            r2 = r * (rgf.u(0.6, 0.7) if k == 0 else rgf.u(0.42, 0.6))
            if fenv(c2) + r2 / R < 0.97 and clear_pt(c2, CLR + 0.14 + r2):
                pads.append((c2, (r2, r2, r2 * flat_k)))
    PC = np.array([tuple(pd[0]) for pd in pads]) if pads else np.zeros((0, 3))
    PR = np.array([pd[1] for pd in pads]) if pads else np.ones((0, 3))

    def in_pads(Q, k=0.86):
        if not pads:
            return np.zeros(len(Q), dtype=bool)
        d = (Q[:, None, :] - PC[None]) / PR[None]
        return ((d * d).sum(-1) < k * k).any(1)

    # ------------------------------------------------------------------ twigs
    rtw = rng.fork("twig")

    def add_twig(c):
        if c and all(clear_pt(q) for q in c["pts"]):
            twigs.append(c)

    def twigs_on(b, t0, per):
        Lb = _blen(b["pts"])
        n3 = max(1, int(Lb * (1 - t0) / per))
        ga = rtw.u(0, 360)
        for j in range(n3):
            t = lerp(t0, 1.0, (j + rtw.u(0.1, 0.95)) / n3)
            q, tg, r = _at(b, t)
            side = _side(tg, ga + j * 137.5)
            if weep:
                d0 = tg * 0.4 + side * 0.6 - UP * 0.2
                L3, zmin = rtw.u(0.3, 0.8), z_low + rtw.u(0.0, 0.8)
            else:
                out = (q - Cn).normalized()
                d0 = (tg * 0.55 + side * 0.8 + out * 0.25 + UP * 0.12).normalized()
                L3, zmin = min(rtw.u(0.2, 0.42), reach(q, d0) + 0.05), None
            if L3 < 0.1:
                continue
            add_twig(grow(rtw, q, d0, L3, min(max(r * 0.7, 0.003), 0.0055), 0.0018, 0.3 if weep else 0.2, trop2, 0.16 if weep else 0.12,
                          zmin=zmin, bounce=weep))

    if not clip:
        for b in secs:
            twigs_on(b, 0.15, 0.55 if weep else 0.15)
        for b in prims:
            twigs_on(b, 0.12 if weep else 0.35, 0.42 if weep else 0.19)
        # sprays breaking out of every inner mass, rooted on the bough inside it
        cloud = [q for b in prims + secs for q in b["pts"]]
        for (c, r) in masses:
            rad = (r, r, r * flat_k)
            q0 = min(cloud, key=lambda q: (q - c).length_squared)
            if (q0 - c).length > rad[0] * 1.2:
                continue
            for k in range(rtw.pick([3, 4, 4, 5])):
                n = (_unit(rtw) + (c - Cn).normalized() * 0.7 + UP * 0.15).normalized()
                e = c + Vector((n.x * rad[0], n.y * rad[1], n.z * rad[2])) * rtw.u(1.15, 1.6) - UP * (0.1 if weep else 0.0)
                if not weep and fenv(e) > 1.02:
                    continue
                mid = q0.lerp(e, 0.55) + _rv(rtw) * 0.025 + UP * 0.02
                add_twig({"pts": [q0.copy(), mid, e], "rad": [0.0045, 0.003, 0.0016]})

    # clipped ball: sheared flats on the sphere, and a few shoots that escaped the shears
    flats = []
    escapes = []
    if clip:
        rc = rng.fork("clip")
        for k in range(15):
            flats.append((_unit(rc), rc.u(0.95, 0.985)))

        def shell(n):
            f = 1.0 + 0.022 * sin(3 * atan2(n.y, n.x) + ph1) * sqrt(max(0.0, 1 - n.z * n.z)) + 0.02 * sin(4 * n.z + ph2)
            cut = False
            for (nk, dk) in flats:
                c = n.dot(nk)
                if c > dk:
                    f = min(f, dk / c)
                    cut = True
            return f, cut

        for k in range(int(round(11 * clamp(dens, 0.6, 1.25)))):
            n = (_unit(rc) + UP * 0.45).normalized()
            f, _ = shell(n)
            q = Vector((Cx + n.x * R * f, Cy + n.y * R * f, zc + n.z * rz * f))
            d = (n + _rv(rc) * 0.3 + UP * 0.35).normalized()
            L = rc.u(0.1, 0.25)
            c = {"pts": [q - d * 0.07, q + d * L * 0.5 + _rv(rc) * 0.012, q + d * L], "rad": [0.0035, 0.0026, 0.0013]}
            if all(clear_pt(x) for x in c["pts"]):
                escapes.append(c)

    for b in prims:
        tub.add(b["pts"], b["rad"], 6 if weep else 8)
    for b in secs:
        tub.add(b["pts"], b["rad"], 3 if weep else 4)
    for b in twigs + escapes:
        tub.add(b["pts"], b["rad"], 3, cap=False)
    wood = tub.mesh("wood")
    assign(wood, bark)
    smooth(wood, 70)
    hr, nm = height_ramp(0.03, 1.7), noise_mask(0.13, seed=seed + 3, lo=0.35, hi=0.7)
    paint(wood, grime=mask_mul(height_ramp(0.0, 0.45), 0.7),
          special=lambda co, n: hr(co, n) * nm(co, n) * np.clip(0.3 + 0.7 * n[:, 1], 0.0, 1.0) * 0.85)
    print(f"DBG wood tris {tub.tris()} prims {len(prims)} secs {len(secs)} twigs {len(twigs)} masses {len(masses)} lobes {len(pads)}")

    # ------------------------------------------------------------------ tree pit
    rg = rng.fork("base")
    rh = 0.25
    kw, kh = 0.11, 0.1
    if grate:
        fw = 0.045
        frame = shape_extrude("grate_frame", rounded_rect(S, S, 0.008, 2), [rounded_rect(S - 2 * fw, S - 2 * fw, 0.003, 1)], depth=0.037, bevel=0.003)
        hp = S / 2 - fw - 0.003
        g = 0.003
        outer = [(g, -hp), (hp, -hp), (hp, hp), (g, hp), (g, rh)] + [(rh * cos(radians(a)), rh * sin(radians(a))) for a in np.linspace(80, -80, 13)] + [(g, -rh)]

        def slot(r0, r1, a, w):
            ca, sa, e = cos(a), sin(a), w * 0.5
            return [(r * ca - s * sa, r * sa + s * ca) for (r, s) in
                    ((r0, -e), (r1, -e), (r1 + e * 0.7, 0.0), (r1, e), (r0, e), (r0 - e * 0.7, 0.0))]

        holes = []
        r_in = rh + 0.06
        avail = hp - 0.03 - r_in
        two = avail >= 0.19
        rings = [(r_in, r_in + (avail - 0.05) / 2), (r_in + (avail + 0.05) / 2, r_in + avail)] if two else [(r_in, r_in + avail)]
        for (ra, rb_) in rings:
            n = max(8, int(2 * pi * (ra + rb_) / 2 / 0.085) // 2 * 2)
            for k in range(n):
                holes.append(slot(ra, rb_, radians((k + 0.5) * 360.0 / n), 0.024))
        rc_ = hp + 0.035
        for (da, r0, r1) in ((0.0, rc_, rc_ + 0.13), (-10.0, rc_ - 0.01, rc_ + 0.06), (10.0, rc_ - 0.01, rc_ + 0.06)):
            for qd in (45.0, -45.0):
                holes.append(slot(r0, r1, radians(qd + da), 0.024))
        holes = [h for h in holes if all(x > g + 0.014 and x < hp - 0.014 and abs(y) < hp - 0.014 for (x, y) in h)]
        plate = shape_extrude("grate_plate", outer, holes, depth=0.018, loc=(0, 0, 0.01))
        plate2 = mirror(plate, 'X', "grate_plate_b")
        cast = [frame, plate, plate2]
        # raised cast ribs: collar round the tree hole, a bead between the slot rings, flanges along the split
        cast.append(lathe("grate_collar", [(rh + 0.001, 0.012), (rh + 0.001, 0.036), (rh + 0.006, 0.041), (rh + 0.026, 0.041),
                                           (rh + 0.034, 0.034), (rh + 0.037, 0.0275)], segments=40))
        beads = [r_in + avail + 0.016] + ([r_in + avail / 2] if two else [])
        for bi, rm in enumerate(beads):
            if rm + 0.012 < hp:
                cast.append(lathe(f"grate_bead{bi}", [(rm - 0.011, 0.0275), (rm - 0.006, 0.0335), (rm + 0.006, 0.0335), (rm + 0.011, 0.0275)], segments=48))
        fl_len = hp - rh - 0.045
        for sx in (-1, 1):
            for sy in (-1, 1):
                cast.append(box("grate_flange", (0.016, fl_len, 0.008), loc=(sx * 0.013, sy * (rh + 0.04 + fl_len / 2), 0.031), bevel=0.002))
        bt = bolt("grate_bolt", 0.009, 0.005, kind="square")
        cast.append(instances_to_mesh("grate_bolts", bt, [trs((sx * 0.045, sy * (hp - 0.035), 0.0285), (0, 0, rg.u(0, 90))) for sx in (-1, 1) for sy in (-1, 1)]
                                      + [trs((sx * (hp - 0.03), sy * 0.03, 0.0285), (0, 0, rg.u(0, 90))) for sx in (-1, 1) for sy in (-1, 1)], keep_base=False))
        rust = noise_mask(0.2, seed=seed + 5, lo=0.3, hi=0.7)
        tread = noise_mask(0.35, seed=seed + 6, lo=0.25, hi=0.7)

        def grate_wear(co, n):          # tops polished by feet, most on the raised ribs
            return np.clip(0.22 + 0.7 * np.clip(n[:, 2], 0.0, 1.0) ** 2 * np.clip((co[:, 2] - 0.022) / 0.012, 0.3, 1.0) * (0.4 + 0.6 * tread(co, n)), 0.0, 1.0)

        for o in cast:
            assign(o, iron)
            paint(o, special=mask_mul(rust, 0.45), wear=grate_wear, grime=lambda co, n: np.clip((0.03 - co[:, 2]) / 0.02, 0.0, 1.0) * 0.5)
        soil = grid("soil", size=(S - 2 * fw + 0.01, S - 2 * fw + 0.01), cuts=(2, 2), loc=(0, 0, 0.004))
        mound = lathe("mound", [(rh + 0.02, 0.0045), (rh * 0.8, 0.012), (rh * 0.5, 0.024), (rh * 0.25, 0.031), (0.0, 0.034)], segments=18)
        displace_noise(mound, strength=0.004, scale=0.07, seed=seed)
    else:
        stone = mat("stone_rough", "#a39d92")
        for sd in range(4):
            L = S - kw
            n = 2 if L < 1.15 else 3
            cuts = [0.0] + sorted(clamp((k + 1) / n + rg.u(-0.08, 0.08), 0.15, 0.85) for k in range(n - 1)) + [1.0]
            a = radians(sd * 90.0)
            for k in range(n):
                ln = (cuts[k + 1] - cuts[k]) * L
                cx = -S / 2 + (cuts[k] + cuts[k + 1]) / 2 * L
                cy = -S / 2 + kw / 2 + rg.u(-0.004, 0.004)
                hh = kh * rg.u(0.94, 1.05)
                st = stone_block(f"kerb{sd}_{k}", (ln - rg.u(0.006, 0.012), kw * rg.u(0.95, 1.04), hh),
                                 loc=(cx * cos(a) - cy * sin(a), cx * sin(a) + cy * cos(a), hh / 2 + 0.004),
                                 rot=(0, 0, sd * 90.0 + rg.u(-0.8, 0.8)), rng=rg, chamfer=0.014, rough=0.004, detail=0.12)
                assign(st, stone)
                paint(st, grime=height_ramp(0.0, 0.09),
                      special=mask_mul(noise_mask(0.2, seed=seed + 7, lo=0.3, hi=0.7), lambda co, n: np.clip(0.25 + 0.75 * n[:, 1] + 0.4 * n[:, 2], 0, 1), 0.8))
        soil = grid("soil", size=(S - 2 * kw + 0.03, S - 2 * kw + 0.03), cuts=(12, 12), loc=(0, 0, 0.058))
        displace_noise(soil, strength=0.009, scale=0.16, seed=seed + 2)
        mound = lathe("mound", [(0.34, 0.05), (0.26, 0.068), (0.16, 0.086), (0.08, 0.096), (0.0, 0.1)], segments=18)
        displace_noise(mound, strength=0.006, scale=0.09, seed=seed)
    for o in (soil, mound):
        assign(o, soil_m)
    paint(soil, special=mask_mul(noise_mask(0.3, seed=seed + 9), 0.5))

    # ------------------------------------------------------------------ stakes and belt ties
    if n_st > 0:
        rk = rng.fork("stake")
        wood_m = mat("wood_weathered", "#c9b89a")
        belt_m = mat("leather", "#2a1d15")
        sd_ = 0.175 if grate else 0.24
        sa0 = rk.u(0, 2 * pi)
        nails = []
        for i in range(n_st):
            a = sa0 + i * pi + rk.u(-0.25, 0.25)
            sx, sy = sd_ * cos(a), sd_ * sin(a)
            sh = min(z_cb * 0.78, 1.55) * rk.u(0.92, 1.04)
            sr = rk.u(0.029, 0.034)
            kx, ky = rk.u(-0.02, 0.02), rk.u(-0.02, 0.02)
            prof = [(0.0, 0.0), (sr * 1.02, 0.0)]
            nseg = 6
            for k in range(1, nseg):
                prof.append((sr * rk.u(0.96, 1.04), sh * k / nseg))
            prof += [(sr, sh - 0.02), (sr - 0.013, sh - 0.002), (0.0, sh)]
            st = lathe(f"stake{i}", prof, segments=10, smooth_angle=38)
            transform_verts(st, lambda co, kx=kx, ky=ky, sx=sx, sy=sy: Vector((co.x + sx + kx * co.z + 0.006 * sin(co.z * 3.1), co.y + sy + ky * co.z, co.z)))
            uv_grain(st, axis='Z')
            assign(st, wood_m)
            paint(st, grime=height_ramp(0.0, 0.5), special=mask_mul(height_ramp(0.0, 0.45), noise_mask(0.1, seed=seed + 20 + i), 0.8),
                  wear=mask_mul(height_ramp(sh, sh - 0.12), 0.5))
            # belt tie: loop round stem and stake, pinched by a keeper between them
            zt = sh - rk.u(0.11, 0.16)
            T = axis_pt(zt)
            Sc = Vector((sx + kx * zt + 0.006 * sin(zt * 3.1), sy + ky * zt, zt))
            u = Vector((Sc.x - T.x, Sc.y - T.y, 0.0))
            D = u.length
            u.normalize()
            v = Vector((-u.y, u.x, 0.0))
            r_t = radius(zt) * 1.06 + 0.004
            r_s = sr + 0.004
            M = T + u * (r_t + (D - r_t - r_s) * 0.5)
            path = []
            for ang in np.linspace(55, 305, 10):
                path.append(T + (u * cos(radians(ang)) + v * sin(radians(ang))) * r_t)
            path.append(M - v * 0.008)
            for ang in np.linspace(-125, 125, 9):
                path.append(Sc + (u * cos(radians(ang)) + v * sin(radians(ang))) * r_s + UP * 0.006)
            path.append(M + v * 0.008)
            belt = sweep(f"tie{i}", [tuple(q) for q in path], rounded_rect(0.004, 0.03, 0.0012, 1), closed=True, up=(0, 0, 1))
            keeper = box(f"tie_keeper{i}", (0.028, 0.03, 0.04), loc=tuple(M + UP * 0.002), rot=(0, 0, degrees(atan2(u.y, u.x))), bevel=0.004)
            for o in (belt, keeper):
                assign(o, belt_m)
            az = degrees(atan2(u.y, u.x))
            for dz in (-0.008, 0.009):
                nails.append(trs(tuple(Sc + u * (r_s + 0.0015) + UP * (0.006 + dz)), (0, 90, az)))
        nl = instances_to_mesh("tie_nails", bolt("nail", 0.0055, 0.0035, kind="dome"), nails, keep_base=False)
        assign(nl, iron)
        paint(nl, wear=0.5)

    # ------------------------------------------------------------------ litter: fallen petals (or leaves when not in bloom)
    n_lit = int(460 * p.litter * ((0.35 + 0.65 * blossom) if blossom > 0 else 0.45))
    if n_lit > 0:
        rq = rng.fork("litter")
        half = S / 2
        rmax = max(R * 1.05, half + 0.3)
        mats_l = []
        for i in range(n_lit):
            rr = rmax * sqrt(rq.u(0.01, 1.0))
            a = rq.u(0, 2 * pi)
            x, y = Cx * 0.5 + rr * cos(a), Cy * 0.5 + rr * sin(a)
            m = max(abs(x), abs(y))
            r0 = hypot(x, y)
            if grate:
                if r0 < rh + 0.045:
                    continue
                z = 0.0405 if m > half - 0.048 and m < half else (0.0305 if m < half else 0.004)
            else:
                if m < half + 0.01 and m > half - kw - 0.012:
                    continue
                z = 0.075 if m < half else 0.004
                if r0 < 0.36:
                    continue
            sc = rq.u(0.02, 0.03) if blossom > 0 else rq.u(0.055, 0.085)
            mats_l.append(trs((x, y, z + rq.u(0.0, 0.0015)), (rq.u(-5, 5), rq.u(-5, 5), rq.u(0, 360)), sc))
        if mats_l:
            if blossom > 0:
                lit = instances_to_mesh("litter", _petal_base("petal_base"), mats_l, uv_jitter=False, keep_base=False)
                assign(lit, petal_m)
            else:
                lit = instances_to_mesh("litter", _leaf_base("leaf_flat", flat=True), mats_l, uv_jitter=False, keep_base=False)
                assign(lit, leaf_a)
                paint(lit, grime=0.45)

    # ------------------------------------------------------------------ foliage
    rf = rng.fork("leaf")
    nprs = np.random.RandomState((seed * 7919 + 17) % (2 ** 31 - 1))
    leaf_len = {"round": 0.082, "vase": 0.08, "weeping": 0.08, "clipped": 0.046}[shape]
    droop = 1.0 if weep else 0.4

    def outward(q):
        o = q - Cn
        if o.length < 1e-4:
            return UP.copy()
        o.normalize()
        if weep:
            o.z = max(o.z, -0.35)
            o.normalize()
        return o

    # inner masses of the boughs: dense small-leaf surface, darker than the blades around it
    green = _Blobs()
    for (c, rad) in pads:
        green.add(c, rad, rgf, (0.55, 0.55), 6, 3)
    if green.v:
        gm = green.mesh("leaf_mass")
        assign(gm, mass_m)
        smooth(gm, 38)
        nmg = noise_mask(0.12, seed=seed + 14, lo=0.2, hi=0.8)
        paint(gm, grime=lambda co, n: np.clip(0.45 + 0.3 * nmg(co, n) - 0.2 * np.clip(n[:, 2], 0.0, 1.0) + lan_shade(co), 0.0, 1.0))
    if clip:
        core = sphere("core", 1.0, loc=(Cx, Cy, zc), segments=30, rings=16)
        transform_verts(core, lambda co: Vector((co.x * R, co.y * R, co.z * rz)) * (0.955 * shell(co.normalized())[0]))
        displace_noise(core, strength=0.012, scale=0.15, seed=seed)
        assign(core, mass_m)
        smooth(core, 60)
        paint(core, grime=0.4)

    shoots = []                     # (points, start fraction, blossom sleeve radius): every leaf sits on real wood
    if clip:
        for b in escapes:
            shoots.append((b["pts"][1:], 0.0, 0.0))
    else:
        for b in twigs:
            shoots.append((b["pts"], 0.15, 0.0))
        for b in secs:
            pts = b["pts"]
            shoots.append((pts[int(len(pts) * (0.25 if weep else 0.5)):], 0.0, 0.0))
        for b in prims:
            pts = b["pts"]
            shoots.append((pts[int(len(pts) * (0.15 if weep else 0.75)):], 0.0, 0.0))
    shoots = [s for s in shoots if len(s[0]) >= 2]
    lens = [_blen(s[0]) for s in shoots]
    ltot = max(sum(lens), 1e-3)

    fixed = _scene_tris()
    avail = max(4000, TRI_TARGET - fixed)
    want_leaf = 8200 * dens * (1.0 - 0.7 * blossom) * (0.9 if clip else 1.0)
    want_fl = 9600 * blossom * (0.6 + 0.4 * dens)
    k_fit = min(1.0, avail / (want_leaf * 4 + want_fl * 4.0 + 1.0))     # flower: 3 tris + its share of the cluster core
    n_leaf, n_fl = int(want_leaf * k_fit), int(want_fl * k_fit)
    print(f"DBG fixed tris {fixed} -> leaves {n_leaf} flowers {n_fl} (fit {k_fit:.2f})")

    la_m, lb_m, la_g, lb_g = [], [], [], []      # matrices and per-leaf shade for old / new leaves

    def put_leaf(q, ldir, nd, sz, new, shade):
        m = _frame(q, ldir, nd, sz, sz * rf.u(0.85, 1.12))
        shade = min(0.9, shade + float(lan_shade(tuple(q))[0]))
        if new:
            lb_m.append(m)
            lb_g.append(shade)
        else:
            la_m.append(m)
            la_g.append(shade)

    def depth(q):                   # 0 at the crown surface .. 1 deep inside
        return clamp(1.0 - fenv(q)) if not weep or q.z >= zc else clamp(1.0 - hypot(q.x - Cx, q.y - Cy) / R)

    green_pads = pads
    n_pad_leaf = int(n_leaf * (0.3 if weep else 0.46)) if green_pads else 0
    if clip:
        n_shell = n_leaf - 8 * len(escapes)
        n_shoot_leaf = n_leaf - n_shell
    else:
        n_shell = 0
        n_shoot_leaf = n_leaf - n_pad_leaf

    # leaves along the shoots: spiral phyllotaxis, hanging a little, smaller and lighter towards the tip
    cand = []
    sp = clamp(ltot / max(n_shoot_leaf, 1) / 2.4, 0.012, 0.05)
    if clip:
        sp = 0.022
    for (s, t0, r_s), L in zip(shoots, lens):
        ga = rf.u(0, 360)
        dist = L * t0 + rf.u(0.0, sp)
        k = 0
        seg_i, seg_s = 0, 0.0
        while dist < L:
            while seg_i < len(s) - 2 and dist > seg_s + (s[seg_i + 1] - s[seg_i]).length:
                seg_s += (s[seg_i + 1] - s[seg_i]).length
                seg_i += 1
            tg = (s[seg_i + 1] - s[seg_i]).normalized()
            q = s[seg_i] + tg * (dist - seg_s)
            side = _side(tg, ga + k * 137.5 + rf.u(-20, 20))
            tf = dist / L
            ldir = (tg * 0.5 + side + UP * (-droop * rf.u(0.5, 1.25)) + _rv(rf) * 0.2).normalized()
            nd = (outward(q) + UP * (0.0 if weep else 0.35) + _rv(rf) * 0.45).normalized()
            sz = leaf_len * rf.u(0.74, 1.18) * (1.0 - 0.22 * tf)
            cand.append((q + side * (0.004 + 0.8 * r_s), ldir, nd, sz, tf))
            dist += sp * rf.u(0.7, 1.3)
            k += 1
    if cand:
        mids = np.array([tuple(c[0] + c[1] * (c[3] * 0.5)) for c in cand])
        ok = ~in_pads(mids)
        for i, c in enumerate(cand):
            if not ok[i]:
                continue
            tip = c[0] + c[1] * c[3]
            if not (clear_pt(c[0]) and clear_pt(tip)) or (not weep and not clip and fenv(tip) > 1.04):
                ok[i] = False
        idx = np.nonzero(ok)[0]
        if len(idx) > n_shoot_leaf:
            idx = nprs.choice(idx, n_shoot_leaf, replace=False)
        for i in idx:
            q, ldir, nd, sz, tf = cand[i]
            new = clip or tf > 0.7 or rf.chance(0.06)
            put_leaf(q, ldir, nd, sz, new, 0.05 * rf.u(0, 1) if new else clamp(0.1 + 0.55 * depth(q) + 0.3 * rf.u(0, 1) ** 2, 0.0, 0.85))

    # leaves lying over the inner masses (they close the crown)
    if n_pad_leaf > 0:
        wts = np.array([pd[1][0] ** 2 for pd in green_pads])
        wts = wts / wts.sum()
        picks = nprs.choice(len(green_pads), int(n_pad_leaf * 1.8), p=wts)
        made = 0
        for pi_ in picks:
            if made >= n_pad_leaf:
                break
            c, rad = green_pads[pi_][:2]
            n = _unit(rf)
            if n.dot(outward(c)) < -0.3 and rf.chance(0.75):
                continue
            q = c + Vector((n.x * rad[0], n.y * rad[1], n.z * rad[2]))
            nw = Vector((n.x / rad[0], n.y / rad[1], n.z / rad[2])).normalized()
            if in_pads(np.array([tuple(q)]), 0.93)[0] or not clear_pt(q):
                continue
            tgt = nw.cross(_rv(rf))
            if tgt.length < 1e-3:
                continue
            ldir = (tgt.normalized() + nw * rf.u(0.2, 0.75) - UP * (0.3 * droop / 0.4)).normalized()
            sz = leaf_len * rf.u(0.78, 1.2)
            new = rf.chance(0.14)
            put_leaf(q - nw * 0.012, ldir, (nw + _rv(rf) * 0.35).normalized(), sz, new,
                     0.06 if new else clamp(0.12 + 0.5 * depth(q) + 0.3 * rf.u(0, 1) ** 2, 0.0, 0.8))
            made += 1

    # clipped ball: dense sheared shell of small leaves following the flats
    if clip:
        for i in range(n_shell):
            n = _unit(rf)
            f, cut = shell(n)
            f *= 1.0 + (rf.u(-0.012, 0.014) if cut else rf.u(-0.02, 0.035))
            q = Vector((Cx + n.x * R * f, Cy + n.y * R * f, zc + n.z * rz * f))
            if not clear_pt(q, 0.2):
                continue
            tgt = n.cross(_rv(rf))
            if tgt.length < 1e-3:
                continue
            ldir = (tgt.normalized() + n * (rf.u(0.0, 0.2) if cut else rf.u(0.1, 0.55))).normalized()
            sz = leaf_len * (rf.u(0.6, 0.95) if cut else rf.u(0.8, 1.3))
            put_leaf(q - ldir * sz * 0.4, ldir, (n + _rv(rf) * 0.3).normalized(), sz * 1.0, rf.chance(0.1),
                     clamp(0.1 + 0.5 * rf.u(0, 1) ** 2 + (0.12 if cut else 0.0), 0.0, 0.8))

    for nm_, mm, gg, lm_ in (("leaves", la_m, la_g, leaf_a), ("leaves_new", lb_m, lb_g, leaf_b)):
        if not mm:
            continue
        lv = instances_to_mesh(nm_, _leaf_base(nm_ + "_base"), [np.array(m) for m in mm], uv_jitter=False, keep_base=False)
        assign(lv, lm_)
        smooth(lv, 80)
        ga_ = np.repeat(np.array(gg), 6)

        def leaf_grime(co, n, ga_=ga_):
            return ga_ if len(co) == len(ga_) else np.full(len(co), float(ga_.mean()))

        paint(lv, grime=leaf_grime)
    print(f"DBG leaves old {len(la_m)} new {len(lb_m)}")

    # blossom: tight clusters of 5-12 small flowers round a spur (a small pale core closes each cluster),
    # set evenly along the shoots and over the boughs; fuller and larger towards full bloom
    if n_fl > 0 and shoots:
        fm, cm = [], []
        cum = np.cumsum(lens)
        pad_frac = (0.3 if weep else 0.5) if pads else 0.0
        if pads:
            wts = np.array([pd[1][0] ** 2 for pd in pads])
            wts = wts / wts.sum()
        n_cl = max(1, int(n_fl / 8.1))
        n_sh = max(1, int(n_cl * (1.0 - pad_frac)))

        def cluster(c, ow, kf):
            rc = rb.u(0.034, 0.048) * (1.0 + 0.3 * p_pale)
            if not clear_pt(c, CLR + rc):
                return
            cm.append(trs(tuple(c), (rb.u(0, 360), rb.u(0, 360), rb.u(0, 360)), (rc * rb.u(0.85, 1.15), rc * rb.u(0.85, 1.15), rc * rb.u(0.7, 1.0))))
            for _ in range(kf):
                n = (_unit(rb) + ow * 0.45).normalized()
                nd = (n + _rv(rb) * 0.3).normalized()
                yd = nd.cross(_rv(rb))
                if yd.length > 1e-3:
                    fm.append(_frame(c + n * (rc * rb.u(0.9, 1.3)), yd.normalized(), nd, rb.u(0.015, 0.021)))

        for k in range(n_sh):
            d = (k + rb.u(0, 1)) * cum[-1] / n_sh
            si = min(int(np.searchsorted(cum, d)), len(shoots) - 1)
            sh = shoots[si][0]
            j = int(rb.u(0, len(sh) - 1.001))
            q = sh[j].lerp(sh[j + 1], rb.u(0, 1))
            side = _side((sh[j + 1] - sh[j]).normalized(), rb.u(0, 360))
            ow = outward(q)
            cluster(q + (side + ow * 0.5).normalized() * rb.u(0.02, 0.045), (side + ow).normalized(), rb.pick([5, 6, 7, 8, 8, 9, 10, 12]))
        guard = 0
        while len(cm) < n_cl and pads and guard < n_cl * 3:
            guard += 1
            c, rad = pads[int(nprs.choice(len(pads), p=wts))][:2]
            n = _unit(rb)
            if n.dot(outward(c)) < -0.2:
                continue
            q = c + Vector((n.x * rad[0], n.y * rad[1], n.z * rad[2]))
            nw = Vector((n.x / rad[0], n.y / rad[1], n.z / rad[2])).normalized()
            if in_pads(np.array([tuple(q)]), 0.93)[0]:
                continue
            cluster(q + nw * 0.02, nw, rb.pick([5, 6, 7, 8, 8, 9, 10, 12]))
        flowers = instances_to_mesh("flowers", _flower_base("flower_base"), [np.array(m) for m in fm], uv_jitter=False, keep_base=False)
        assign(flowers, petal_m)
        smooth(flowers, 80)
        nmf = noise_mask(0.05, seed=seed + 12, lo=0.3, hi=0.8)
        paint(flowers, grime=lambda co, n: np.clip(0.14 * nmf(co, n) + 0.7 * lan_shade(co), 0.0, 1.0))
        cores = instances_to_mesh("flower_cores", _core_base("core_base"), cm, uv_jitter=False, keep_base=False)
        assign(cores, petal_m)
        smooth(cores, 80)
        paint(cores, grime=lambda co, n: np.clip(0.2 + 0.15 * nmf(co, n) + 0.7 * lan_shade(co), 0.0, 1.0))
        print(f"DBG flowers {len(fm)} in {len(cm)} clusters")

    socket("garland", "attach", (Cx, Cy, z_cb), role="bunting")
