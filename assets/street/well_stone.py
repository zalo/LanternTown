"""Round stone draw-well. A coursed curb of individually cut, wedge-shaped stone blocks with a cramped
coping ring (worn basin, rope grooves and chipped lip at the drawing side) stands on a ring of sunk
flagstones; the shaft continues down to dark water. Two posts (pegged weathered oak, or forged iron bar
with scrolls) carry a windlass - turned drum on an iron axle with bearing plates, ratchet, crank and
coiled hemp rope - and either a small king-post roof (rafters, battens, hand-split shingles or
beaver-tail clay tiles, barge boards, finials) or an open lintel / iron arch. A staved or galvanised
bucket hangs from the rope; a candle lantern hangs from a bracket. Weathering is built, not only
painted: moss cushions in the joints, at the foot of the curb and under the shingle butts; the damp lip,
drip runs down the curb, bucket mark and splash zone are cut into the stone as soaked (darker, glossier)
faces; posts carry damp feet and iron / tannin stains below every fixing; the water has drip rings.
"""
from tk import *
import bmesh
import numpy as np
from math import sin, cos, tan, pi, radians, degrees, atan2, asin, hypot, ceil, exp
from mathutils import Matrix, Vector
from mathutils import noise as mnoise
from mathutils.bvhtree import BVHTree

META = {"name": "Stone well with roof", "class": "structure", "mount": "ground",
        "allow_below_ground": True, "focus": [-0.15, 0.0, 1.22]}
PARAMS = {
    "diameter": {"default": 1.5, "min": 1.1, "max": 2.2, "desc": "outer diameter of the stone curb (m)"},
    "curb_height": {"default": 0.8, "min": 0.6, "max": 1.0, "desc": "height of the curb incl. coping (m)"},
    "roof": {"default": "shingle", "choices": ["shingle", "tile", "none"], "desc": "roof covering (none = open lintel / arch)"},
    "posts": {"default": "timber", "choices": ["timber", "iron_arch"], "desc": "pegged oak posts or forged iron uprights with an arch"},
    "bucket": {"default": "wood", "choices": ["wood", "zinc", "none"], "desc": "bucket hanging from the rope"},
    "pitch": {"default": 45.0, "min": 35.0, "max": 55.0, "desc": "roof pitch (deg)"},
    "step": {"default": True, "desc": "ring of flagstones round the curb"},
    "water_depth": {"default": 0.26, "min": 0.15, "max": 3.0, "desc": "water level below the coping (m); deeper than the curb goes below ground"},
    "rope_turns": {"default": 9, "min": 4, "max": 14, "desc": "turns of rope coiled on the windlass"},
    "lantern": {"default": True, "desc": "candle lantern on a bracket on the left post"},
    "intensity": {"default": 4.5, "min": 2.0, "max": 40.0, "desc": "lantern intensity (cd)"},
    "moss": {"default": 0.55, "min": 0.0, "max": 1.0, "desc": "amount of moss / algae on stone, timber and roof"},
    "wet": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "wet staining: damp lip, drip runs down the curb, bucket mark, splash zone on the step"},
    "stone": {"default": "#aca18c", "desc": "stone tint"},
    "timber": {"default": "#7a5a3c", "desc": "timber tone (weathered oak frame, drum, shingles)"},
}
VARIANTS = {
    "tiled_wide": {"diameter": 2.0, "roof": "tile", "pitch": 50.0, "bucket": "wood", "stone": "#c2ad8a",
                   "timber": "#5a4230", "moss": 0.3, "wet": 0.8, "seed": 4},
    "iron_arch": {"posts": "iron_arch", "roof": "none", "diameter": 1.35, "curb_height": 0.86, "bucket": "zinc",
                  "stone": "#a9a8a4", "moss": 0.4, "wet": 0.45, "seed": 11},
    "open_mossy": {"roof": "none", "diameter": 1.2, "curb_height": 0.92, "step": False, "moss": 1.0, "wet": 0.9,
                   "timber": "#5e4a38", "rope_turns": 12, "water_depth": 0.3, "seed": 23},
    "iron_tiled": {"posts": "iron_arch", "roof": "tile", "diameter": 1.7, "pitch": 40.0, "bucket": "wood",
                   "stone": "#b8a488", "timber": "#4a3526", "moss": 0.7, "seed": 31},
}


# ------------------------------------------------------------------------------------------ helpers
def _c01(a):
    return np.clip(a, 0.0, 1.0)


def _wrapf(d):
    return (d + pi) % (2 * pi) - pi


def _ang(co):
    """numpy: angle round the well axis, 0 = front (-Y), increasing towards +X."""
    return np.arctan2(co[:, 0], -co[:, 1])


def _lin2hex(c):
    def f(v):
        v = clamp(v)
        s = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
        return int(round(clamp(s) * 255))
    return "#%02x%02x%02x" % (f(c[0]), f(c[1]), f(c[2]))


def _polar_bend(obj, r_mid, a_mid, z_mid):
    """Bend a block modelled flat (x = arc length, y = towards the well axis, z up) onto a ring."""
    def f(co):
        a = a_mid + co.x / r_mid
        r = r_mid - co.y
        return Vector((r * sin(a), -r * cos(a), co.z + z_mid))
    transform_verts(obj, f)


def _ring_course(prefix, r_in, r_out, z0, z1, nominal, r, material, chamfer=0.012, rough=0.005, detail=0.1,
                 joint=0.012, proud=0.006, zjit=0.0, jvar=0.0, rjit=0.0, cvar=0.0):
    """One course of wedge-shaped stones with uneven joints. Returns a list of dicts (object + extents)."""
    rm = (r_in + r_out) / 2
    n = max(5, int(round(2 * pi * rm / nominal)))
    ws = [r.u(0.78, 1.25) for _ in range(n)]
    tot = sum(ws)
    a = r.u(0, 2 * pi)
    out = []
    for i, w in enumerate(ws):
        da = 2 * pi * w / tot
        jw = joint * (1.0 + r.u(-jvar, jvar))
        depth = r_out - r_in + r.u(-rjit, rjit * 0.6)
        h = (z1 - z0) - joint * 0.6
        b = stone_block(f"{prefix}{i}", (da * rm - jw, depth, h), rng=r, chamfer=chamfer * (1.0 + r.u(-cvar, cvar)),
                        rough=rough, detail=detail)
        rmid = r_in + depth / 2 + r.u(-proud, proud)
        zm = (z0 + z1) / 2 + r.u(-zjit, zjit)
        _polar_bend(b, rmid, a + da / 2, zm)
        assign(b, material)
        out.append({"o": b, "a0": a, "a1": a + da, "am": a + da / 2, "r_in": rmid - depth / 2, "r_out": rmid + depth / 2,
                    "z0": zm - h / 2, "z1": zm + h / 2, "g0": r.u(0.0, 1.0)})
        a += da
    return out


def _cull(o, pred):
    """Delete never-seen faces: pred(center, normal) in local space."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    dead = [f for f in bm.faces if pred(f.calc_center_median(), f.normal)]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context='FACES')
    bm.to_mesh(o.data)
    bm.free()


def _bvh(objs):
    verts, polys = [], []
    for o in objs:
        M = world_matrix(o)
        off = len(verts)
        verts += [M @ v.co for v in o.data.vertices]
        polys += [tuple(off + i for i in pl.vertices) for pl in o.data.polygons]
    return BVHTree.FromPolygons(verts, polys)


def _cut_zone(o, wf, sizes):
    """Give a zone wf(world position) > 0 (signed distance, m) a true outline on a mesh: subdivide the faces
    near the outline (one level per entry of sizes = selection distance), then split the edges exactly where
    wf crosses zero and join the cuts, so material / paint can follow the outline instead of block-sized steps."""
    M = world_matrix(o)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    for e in sizes:
        sel = [f for f in bm.faces if abs(wf(M @ f.calc_center_median())) < e]
        if not sel:
            break
        bmesh.ops.subdivide_edges(bm, edges=list({ed for f in sel for ed in f.edges}), cuts=1, use_grid_fill=True)
    wv = {v: wf(M @ v.co) for v in bm.verts}
    new = []
    for ed in list(bm.edges):
        w0, w1 = wv[ed.verts[0]], wv[ed.verts[1]]
        if (w0 > 0.0) != (w1 > 0.0) and abs(w0) < 0.4 and abs(w1) < 0.4:
            t = w0 / (w0 - w1)
            if 0.15 < t < 0.85:
                new.append(bmesh.utils.edge_split(ed, ed.verts[0], t)[1])
    if new:
        bmesh.ops.connect_verts(bm, verts=new)
    bm.to_mesh(o.data)
    bm.free()


def _run_w(runs, s_of, z):
    """Signed distance (> 0 inside) to a set of tapering wobbly runs (drips / stains).
    runs: (key, z_top, length, width, phase, wobble); s_of(key) = sideways distance from the run's axis."""
    best = -1.0
    for key, zt, L, w0, ph, wob in runs:
        d = zt - z
        if d < -0.03 or d > L + 0.03:
            continue
        t = clamp(d / L)
        hwid = 0.5 * w0 * (1.0 - t ** 1.6) * (1.0 + 0.22 * sin(9 * t + ph)) + 0.003
        w = min(hwid - abs(s_of(key) - wob * sin(5 * t + ph) * t), d + 0.004, L - d)
        if w > best:
            best = w
    return best


def _blob(name, cx, cy, z, rx, ry, rot, r, nr=3, ns=16, hole=0.0):
    """Flat irregular patch (or ring, hole = inner radius fraction) in the XY plane, normals up."""
    ph = [r.u(0, 2 * pi) for _ in range(3)]
    am = [r.u(0.08, 0.2), r.u(0.05, 0.14), r.u(0.03, 0.09)]
    cr, sr = cos(rot), sin(rot)
    verts, faces = [], []
    rings = nr + 1 if hole > 0 else nr
    for j in range(rings):
        t = (hole + (1 - hole) * j / nr) if hole > 0 else (j + 1) / nr
        for i in range(ns):
            a = 2 * pi * i / ns
            kk = 1.0 + (am[0] * sin(2 * a + ph[0]) + am[1] * sin(3 * a + ph[1]) + am[2] * sin(5 * a + ph[2])) * (t if hole <= 0 else 0.35)
            x, y = rx * t * kk * cos(a), ry * t * kk * sin(a)
            verts.append((cx + x * cr - y * sr, cy + x * sr + y * cr, z))
    for j in range(rings - 1):
        for i in range(ns):
            i1 = (i + 1) % ns
            faces.append((j * ns + i, (j + 1) * ns + i, (j + 1) * ns + i1, j * ns + i1))
    if hole <= 0:
        c = len(verts)
        verts.append((cx, cy, z))
        for i in range(ns):
            faces.append((c, i, (i + 1) % ns))
    return mesh_object(name, verts, faces)


def _frame(P, n, t, s):
    """4x4: local Z -> n, local X -> t (made perpendicular), scaled by s."""
    n = Vector(n).normalized()
    t = Vector(t)
    t = t - n * t.dot(n)
    if t.length < 1e-6:
        t = n.orthogonal()
    t.normalize()
    b = n.cross(t)
    M = Matrix((t, b, n)).transposed().to_4x4()
    M.translation = Vector(P)
    return M @ Matrix.Diagonal((s[0], s[1], s[2], 1.0))


def _clump_base(r):
    """Unit moss cushion (radius 1, height 1, base a little below z = 0 so it beds into rough stone)."""
    n = 5
    verts, faces = [], []
    for j, (rad, z) in enumerate(((1.0, -0.25), (0.8, 0.5))):
        for i in range(n):
            a = 2 * pi * (i + 0.5 * j) / n
            kk = rad * r.u(0.8, 1.16)
            verts.append((kk * cos(a), kk * sin(a), z + r.u(-0.07, 0.07)))
    verts.append((r.u(-0.15, 0.15), r.u(-0.15, 0.15), 1.0))
    for i in range(n):
        i1 = (i + 1) % n
        faces.append((i, i1, n + i1, n + i))
        faces.append((n + i, n + i1, 2 * n))
    o = mesh_object("moss_base", verts, faces)
    smooth(o, 80)
    return o


def _beam(name, a, b, w, t, bevel=0.006, warp=0.0, rng=None):
    """Timber between two points: w = horizontal width, t = depth in the vertical plane."""
    a, b = Vector(a), Vector(b)
    d = b - a
    L = d.length
    rot = (0.0, -degrees(asin(clamp(d.z / L, -1.0, 1.0))), degrees(atan2(d.y, d.x)))
    return plank(name, L, w, t, loc=tuple((a + b) / 2), rot=rot, bevel=bevel, warp=warp, rng=rng)


def _prism_yz(name, pts, x0, x1, bevel=0.004):
    """Extrude a polygon given in (y, z) along X from x0 to x1."""
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    if area < 0:
        pts = pts[::-1]
    o = extrude_profile(name, pts, abs(x1 - x0), bevel=bevel)
    xm = min(x0, x1)
    transform_verts(o, lambda co: Vector((co.z + xm, co.x, co.y)))
    return o


def _tris(objs):
    return sum(sum(len(pl.vertices) - 2 for pl in o.data.polygons) for o in objs if o is not None and o.type == 'MESH')


# -------------------------------------------------------------------------------------------- build
def build(p, rng):
    D = p.diameter
    Ro = D / 2
    wall = clamp(0.15 * D, 0.2, 0.27)
    Ri = Ro - wall
    Hc = p.curb_height
    cop_h = 0.12
    zc_top = Hc - cop_h                          # top of the coursed curb / bed of the coping
    step_h = 0.14 if p.step else 0.0
    zf = step_h                                  # level the curb visibly rises from
    r_lip, r_cop = Ri - 0.035, Ro + 0.055        # inner / outer radius of the coping
    cope_w = r_cop - r_lip
    iron_posts = p.posts == "iron_arch"
    has_roof = p.roof != "none"
    hw = 0.026 if iron_posts else 0.075          # half width of a post
    xp = Ro + 0.07 + hw                          # post axis
    Zw = Hc + 0.70                               # windlass axis height
    He = Hc + 1.30                               # top of posts / underside of tie beams
    z_water = Hc - p.water_depth
    seed = int(p.seed)

    st_lin = tuple(hex_rgb(p.stone))[:3]
    tm_lin = tuple(hex_rgb(p.timber))[:3]
    stone = mat("stone_rough", p.stone)
    wet_m = mat("stone_rough", _lin2hex([c * 0.3 for c in st_lin]), rough=0.3)          # the same stone soaked: darker, glossier
    timber = mat("wood_weathered", _lin2hex([tm_lin[i] * 0.58 + (0.80, 0.76, 0.70)[i] * 0.42 for i in range(3)]))
    iron = mat("iron_wrought")
    rope_m = mat("rope")
    water_m = mat("water", "#46564a", rough=0.3, flow=0.3)
    # (10 material variants at most: the rare tile + zinc combination beds its moss cushions in mossy stone instead)
    moss_m = mat("moss") if not (p.roof == "tile" and p.bucket == "zinc") else None

    exposed, sheltered, irons = [], [], []       # weather-beaten timber, timber under the roof, ironwork
    stains = []                                  # iron / tannin stains running down the posts

    nz_a = noise_mask(0.4, seed=seed + 11, lo=0.3, hi=0.7)
    nz_b = noise_mask(0.13, seed=seed + 12, lo=0.38, hi=0.72)
    nz_c = noise_mask(0.06, seed=seed + 13, lo=0.3, hi=0.8)
    nz_s = noise_mask(0.5, seed=seed + 14, lo=0.1, hi=0.95)       # broad, soft staining

    # damp side: growth is heaviest to the north-east, creeping round the whole well as `moss` rises
    spread_add = max(0.0, p.moss - 0.5) * 2.2
    K = p.moss ** 1.4                             # moss cushion density

    def w_a(a):
        return clamp(0.38 + 0.75 * (0.45 * sin(a) - 0.89 * cos(a)) + spread_add)

    def w_np(co):
        rr = np.maximum(np.hypot(co[:, 0], co[:, 1]), 1e-4)
        return _c01(0.38 + 0.75 * (0.45 * co[:, 0] + 0.89 * co[:, 1]) / rr + spread_add)

    def w_n(n):
        return _c01(0.38 + 0.75 * (0.45 * n[:, 0] + 0.89 * n[:, 1]) + spread_add)

    rd_ = rng.fork("dish")
    a_d = rd_.u(-0.22, 0.22)                      # where people stand and draw (front)

    # ================================================================= stone curb, coping, step, shaft
    rs = rng.fork("curb")
    dsc = max(1.0, Ro / 0.75) ** 0.8              # bigger wells: same number of facets per block, not more
    n_c = max(2, int(round(zc_top / 0.21)))
    hs = [rs.u(0.85, 1.2) for _ in range(n_c)]
    k = zc_top / sum(hs)
    z = 0.0
    courses = []
    for ci, h in enumerate(hs):
        infos = _ring_course(f"curb{ci}_", Ri, Ro, z, z + h * k, rs.u(0.34, 0.46), rs, stone,
                             chamfer=0.014, rough=0.006, detail=0.135 * dsc, jvar=0.5, cvar=0.3)
        courses.append((z, z + h * k, infos))
        z += h * k
    lining = []
    for ci in range(2 if z_water < -0.05 else 0):  # courses of lining below ground (only the inner faces are ever seen)
        lining += _ring_course(f"lining{ci}_", Ri, Ri + 0.13, -0.27 * (ci + 1), -0.27 * ci, 0.42, rs, stone,
                               chamfer=0.012, rough=0.006, detail=0.2, proud=0.008)
    curb = [i_ for c_ in courses for i_ in c_[2]]

    # hearting / mortar behind the joints + plain shaft lining down to the water
    core = lathe("core", [(Ro - 0.016, zc_top * i / 8) for i in range(9)] +
                 [(Ri + 0.016, zc_top), (Ri + 0.016, (zc_top + max(0.0, z_water)) / 2), (Ri + 0.016, max(0.0, z_water)),
                  (Ri + 0.02, z_water - 0.02), (Ri - 0.03, z_water - 0.38), (0.0, z_water - 0.42)], segments=48)
    cope_core = lathe("cope_core", [(r_cop - 0.02, zc_top + 0.004), (r_cop - 0.02, Hc - 0.02), ((r_cop + r_lip) / 2, Hc - 0.017),
                                    (r_lip + 0.02, Hc - 0.02), (r_lip + 0.02, zc_top + 0.004)], segments=48)
    mortar = [core, cope_core]
    # the water: slow rings spreading from where the bucket drips, so the surface catches the lantern
    n_rp = int(min(Ri, 0.42) / 0.026)
    water = lathe("water", [(Ri + 0.03, z_water), (min(Ri, 0.42) + 0.02, z_water)] +
                  [(i * 0.026, z_water + 0.0024 * cos(2 * pi * i * 0.026 / 0.105) * exp(-i * 0.026 / 0.3) * (1.0 if i else 0.6))
                   for i in range(n_rp, -1, -1)], segments=32)
    smooth(water)
    assign(water, water_m)

    cope = _ring_course("cope", r_lip, r_cop, zc_top, Hc, 0.56, rng.fork("cope"), stone, chamfer=0.02, rough=0.0035,
                        detail=0.06 * dsc ** 0.6, joint=0.011, proud=0.004, zjit=0.003, jvar=0.7, cvar=0.4)
    steps, pads = [], []
    if p.step:
        steps = _ring_course("step", Ro + 0.004, Ro + 0.5, -0.05, step_h, 0.7, rng.fork("step"), stone, chamfer=0.018,
                             rough=0.007, detail=0.11 * dsc, joint=0.016, proud=0.0, zjit=0.007, jvar=0.7, rjit=0.045, cvar=0.4)
        bed = lathe("step_bed", [(Ro + 0.43, step_h - 0.024), (Ro + 0.24, step_h - 0.019), (Ro - 0.01, step_h - 0.024)], segments=48)
        displace_noise(bed, strength=0.005, scale=0.12, seed=seed + 3, along_normal=False, axis_mask=(0, 0, 1))
        mortar.append(bed)
    else:
        for sx in (-1, 1):
            pd = stone_block(f"pad{sx}", (0.38, 0.38, 0.16), loc=(sx * xp, 0, 0.02), rng=rs, chamfer=0.02, rough=0.005, detail=0.1)
            apply_transform(pd)
            assign(pd, stone)
            pads.append(pd)
    z0 = step_h if p.step else 0.1               # level the posts stand on

    # --- the coping is shaped by use: worn basin at the drawing side, rounded / chipped lip, rope grooves
    rch = rng.fork("chips")
    chips_in = [(rch.u(0, 2 * pi), rch.u(0.07, 0.16), rch.u(0.006, 0.018)) for _ in range(int(5 + 4 * Ro))]
    chips_in += [(a_d + rch.u(-0.5, 0.5), rch.u(0.08, 0.14), rch.u(0.008, 0.016)) for _ in range(3)]
    chips_out = [(rch.u(0, 2 * pi), rch.u(0.05, 0.11), rch.u(0.008, 0.022)) for _ in range(int(6 + 5 * Ro))]

    def cope_shape(co):
        r = hypot(co.x, co.y)
        a = atan2(co.x, -co.y)
        zz = co.z
        g = exp(-(_wrapf(a - a_d) / 0.42) ** 2)
        top = smoothstep(Hc - 0.06, Hc - 0.012, zz)
        t_in = clamp((r - r_lip) / cope_w)
        zz -= top * g * 0.010 * sin(pi * t_in)                      # basin where full buckets are set down
        de = hypot(r - r_lip, Hc - co.z)
        kk = clamp(1.0 - de / 0.055)
        if kk > 0:                                                  # inner arris dragged round by rope and buckets
            c = sum(d_ * exp(-(_wrapf(a - ai) / wi) ** 2) for ai, wi, d_ in chips_in)
            amt = kk * kk * (0.007 + 0.012 * g + c)
            r += amt * 0.75
            zz -= amt * 0.75
        de = hypot(r_cop - r, Hc - co.z)
        kk = clamp(1.0 - de / 0.05)
        if kk > 0:                                                  # knocks on the outer arris
            c = sum(d_ * exp(-(_wrapf(a - ai) / wi) ** 2) for ai, wi, d_ in chips_out)
            amt = kk * kk * (0.002 + 0.004 * g + c)
            r -= amt * 0.75
            zz -= amt * 0.75
        return Vector((r * sin(a), -r * cos(a), zz))

    def step_shape(co):
        top = smoothstep(step_h - 0.06, step_h - 0.012, co.z)
        if top <= 0:
            return co
        r = hypot(co.x, co.y)
        a = atan2(co.x, -co.y)
        g = exp(-(_wrapf(a - a_d) * (Ro + 0.25) / 0.3) ** 2)
        dz = 0.013 * g * exp(-((r - (Ro + 0.24)) / 0.15) ** 2)      # hollow trodden where people stand to draw
        ko = clamp(1.0 - (Ro + 0.5 - r) / 0.07)
        dz += ko * ko * (0.004 + 0.009 * g)                         # nosing worn round
        return Vector((co.x, co.y, co.z - top * dz))

    for inf in cope:
        transform_verts(inf["o"], cope_shape)
    for inf in steps:
        transform_verts(inf["o"], step_shape)
    # rope grooves sawn into the lip at the drawing side
    for gi in range(3):
        ag = a_d + (-0.2, 0.04, 0.27)[gi] + rd_.u(-0.05, 0.05)
        el = rd_.u(34, 50)
        for inf in cope:
            if 0.07 < (ag - inf["a0"]) % (2 * pi) < (inf["a1"] - inf["a0"]) - 0.07:
                rg, zg = r_lip + 0.02, Hc - 0.021
                cut = cylinder("groove_cut", rd_.u(0.009, 0.013), 0.2, loc=(rg * sin(ag), -rg * cos(ag), zg),
                               rot=(90 - el, 0, degrees(ag)), verts=10)
                try:
                    boolean(inf["o"], cut)
                except Exception as e:                              # keep the block if the solver balks
                    print("DBG groove boolean failed", e)
                    delete([cut])
                break

    # --- hidden faces out (bed faces, joint faces behind the hearting, undersides, buried parts)
    for inf in curb + lining:
        _cull(inf["o"], lambda c, n: (Ri + 0.045 < hypot(c.x, c.y) < Ro - 0.045)
              or (p.step and c.z < step_h - 0.035 and hypot(c.x, c.y) > Ro - 0.05)
              or (c.z < z_water - 0.04 and hypot(c.x, c.y) < Ri + 0.05))
    for inf in cope:
        _cull(inf["o"], lambda c, n: r_lip + 0.055 < hypot(c.x, c.y) < r_cop - 0.075 and c.z < Hc - 0.04)
    for inf in steps:
        ro_ = inf["r_out"]
        _cull(inf["o"], lambda c, n, ro_=ro_: c.z < -0.004 or (c.z < step_h - 0.045 and hypot(c.x, c.y) < ro_ - 0.04))
    for pd in pads:
        _cull(pd, lambda c, n: c.z < -0.004)

    bvh_cope = _bvh([i_["o"] for i_ in cope])
    bvh_step = _bvh([i_["o"] for i_ in steps]) if steps else None
    up_dir = Vector((0, 0, 1))

    # --- where the stone is wet. wet_w(center, normal) is a signed distance on the stone surface (m, > 0 = wet):
    #     the lip all round, a sheet slopped over the coping at the drawing side, the ring a full bucket left,
    #     runs down the face of the curb (under the spill and under the coping joints), splashes on the flags.
    #     The blocks are subdivided along that outline and the wet faces get the soaked variant of the stone.
    rwt = rng.fork("wet")
    wet_on = p.wet > 0.05
    wph = [rwt.u(0, 2 * pi) for _ in range(4)]
    runs, marks, splashes = [], [], []
    a_b = a_d + rwt.pick((-1, 1)) * rwt.u(0.6, 0.85)                # where the bucket gets parked on the coping
    if wet_on:
        n_run = int(round((4 + 11 * p.wet) * min(Ro / 0.75, 1.2)))
        for i in range(n_run):
            front = i < n_run * 0.55
            a = a_d + rwt.u(-1.0, 1.0) * rwt.u(0.06, 0.5) if front else rwt.pick(cope)["a0"] + rwt.u(-0.03, 0.03)
            L = min(rwt.u(0.1, 0.42) * (0.5 + 0.8 * p.wet), zc_top - zf - 0.03)
            if abs(abs(_wrapf(a)) - pi / 2) < (hw + 0.06) / Ro:      # not behind the post straps
                continue
            runs.append((a, zc_top + (0.035 if front else -0.004), L + (0.035 if front else 0.0), rwt.u(0.02, 0.045),
                         rwt.u(0, 2 * pi), rwt.u(0.003, 0.012)))
        rb_c = r_lip + cope_w * 0.55
        marks.append((rb_c * sin(a_b), -rb_c * cos(a_b), 0.118))
        for _ in range(int(2 + 4 * p.wet)):
            aa, rr = a_b + rwt.u(-0.3, 0.3), r_lip + cope_w * rwt.u(0.3, 0.85)
            marks.append((rr * sin(aa), -rr * cos(aa), rwt.u(0.012, 0.035)))
        if p.step:
            for i in range(int(3 + 7 * p.wet)):
                aa = a_d + (rwt.u(-0.1, 0.1) if i == 0 else rwt.u(-0.75, 0.75))
                rr = Ro + (0.2 if i == 0 else rwt.u(0.08, 0.4))
                rx, ry = (rwt.u(0.17, 0.24), rwt.u(0.09, 0.12)) if i == 0 else (rwt.u(0.03, 0.1), rwt.u(0.02, 0.06))
                rot = aa + rwt.u(-0.4, 0.4)
                splashes.append((rr * sin(aa), -rr * cos(aa), rx, ry, cos(rot), sin(rot)))

    def band_w(a):                                                    # reach of the wet from the lip outwards
        g = exp(-(_wrapf(a - a_d) / 0.38) ** 2)                       # (beyond the width of the coping = down its face)
        w = (0.05 + 0.07 * p.wet) * (1.0 + 0.4 * sin(3 * a + wph[0]) + 0.3 * sin(7 * a + wph[1]) + 0.2 * sin(13 * a + wph[2]))
        return max(w, 0.045) + g * p.wet * (cope_w + 0.07) * 2.0 * (0.9 + 0.2 * sin(17 * a + wph[3]))

    def wet_w(c):
        if not wet_on:
            return -1.0
        r = hypot(c.x, c.y)
        if r < 0.05 or c.z < zf - 0.03:
            return -1.0
        a = atan2(c.x, -c.y)
        jit = 0.012 * mnoise.noise(c * 13.0)
        if c.z > zc_top - 0.006:                                      # coping
            if r < r_lip + 0.03:                                      # the lip itself is always damp
                return 0.05
            if r > r_cop + 0.03:
                return -1.0
            w = band_w(a) - ((r - r_lip) + max(0.0, Hc - 0.012 - c.z))
            for mx, my, mr in marks:
                w = max(w, mr - hypot(c.x - mx, c.y - my) - 3.0 * max(0.0, Hc - 0.02 - c.z))
            if r > r_cop - 0.03:
                w = max(w, _run_w(runs, lambda ai: _wrapf(a - ai) * r, c.z))
            return w + jit
        if p.step and c.z < step_h + 0.03 and r > Ro + 0.02:           # flags
            if c.z < step_h - 0.03:
                return -1.0
            g = exp(-(_wrapf(a - a_d) / 0.45) ** 2)
            jit += 0.03 * mnoise.noise(c * 4.5)                        # ragged, soaked-in outline
            w = p.wet * 0.24 * g - 0.03 - (r - Ro)                    # damp foot where the runs end
            for cx, cy, rx, ry, cr_, sr_ in splashes:
                dx, dy = c.x - cx, c.y - cy
                w = max(w, (1.0 - hypot((dx * cr_ + dy * sr_) / rx, (dy * cr_ - dx * sr_) / ry)) * min(rx, ry))
            return w + jit
        if r > Ro - 0.03:                                             # face of the curb
            return _run_w(runs, lambda ai: _wrapf(a - ai) * r, c.z) + jit * 0.4
        return -1.0

    def wet_halo(co, n):                                              # damp darkening fading out round the wet
        v = np.array([wet_w(Vector(c)) for c in co.tolist()]) if wet_on else np.full(len(co), -1.0)
        return 0.34 * _c01((v + 0.035) / 0.04)

    if wet_on:
        for inf in cope:
            _cut_zone(inf["o"], wet_w, (0.036,))
        for inf in curb:
            if inf["z1"] > zf:
                _cut_zone(inf["o"], wet_w, (0.08, 0.04))
        for inf in steps:
            _cut_zone(inf["o"], wet_w, (0.066, 0.033))

    # --- weathering of the stone: moss lives in the joints, at the foot and on the damp side; every block
    #     has its own tone; dirt gathers at the foot and where feet do not scour it away
    def curb_paint(inf):
        am, ha, zb0, zb1, g0 = inf["am"], (inf["a1"] - inf["a0"]) / 2, inf["z0"], inf["z1"], inf["g0"]

        def edge(co):
            ex = (ha - np.abs(_wrapf(_ang(co) - am))) * Ro
            ez = np.minimum(co[:, 2] - zb0, zb1 - co[:, 2])
            return _c01(1.0 - np.minimum(ex, ez) / 0.04)

        def special(co, n):
            outer = np.hypot(co[:, 0], co[:, 1]) > Ri + wall * 0.5
            low = np.clip(1.2 - (co[:, 2] - zf) / (0.8 * Hc), 0.12, 1.0)
            skirt = _c01(1.0 - (co[:, 2] - zf) / 0.15)
            m = p.moss * w_np(co) * (1.3 * edge(co) * low + 1.5 * skirt * (0.55 + 0.45 * nz_b(co, n)) + 0.3 * nz_a(co, n) * low)
            algae = (0.35 + 0.65 * p.moss) * _c01(1.0 - (co[:, 2] - z_water) / 0.3) * (0.5 + 0.5 * nz_b(co, n))
            return _c01(np.where(outer, m, algae))

        def grime(co, n):
            outer = np.hypot(co[:, 0], co[:, 1]) > Ri + wall * 0.5
            g_out = 0.03 + 0.2 * g0 + 0.1 * nz_s(co, n) + 0.25 * _c01(1.0 - (co[:, 2] - zf) / 0.1)
            g_in = 0.15 + 0.4 * _c01(1.0 - (co[:, 2] - z_water) / max(Hc - z_water, 0.1))
            return _c01(np.maximum(np.where(outer, g_out, g_in), wet_halo(co, n)))
        paint(inf["o"], grime=grime, special=special, wear=lambda co, n: 0.25 * nz_c(co, n))

    for inf in curb + lining:
        curb_paint(inf)

    def cope_paint(inf):
        am, ha, g0 = inf["am"], (inf["a1"] - inf["a0"]) / 2, inf["g0"]

        def special(co, n):
            edge = _c01(1.0 - (ha - np.abs(_wrapf(_ang(co) - am))) * Ro / 0.045)
            up = _c01(n[:, 2])
            lichen = _c01((nz_b(co, n) - 0.3) * 2.2) * up
            under = _c01(-n[:, 2]) * 0.8                             # green under the drip edge
            dry = 1.0 - 0.8 * np.exp(-(_wrapf(_ang(co) - a_d) / 0.5) ** 2)       # nothing grows where it is used
            return _c01(p.moss * w_np(co) * (1.1 * lichen + 1.2 * edge + under) * dry)

        def wear(co, n):                                             # polished by hands, rope and bucket bottoms
            front = np.exp(-(_wrapf(_ang(co) - a_d) / 0.6) ** 2)
            return _c01(_c01(n[:, 2] + 0.3) * (0.2 + 0.7 * front) * (0.5 + 0.5 * nz_b(co, n)))
        paint(inf["o"], grime=lambda co, n: _c01(np.maximum(0.04 + 0.16 * g0 + 0.12 * nz_s(co, n), wet_halo(co, n))),
              wear=wear, special=special)

    for inf in cope:
        cope_paint(inf)

    def step_paint(inf):
        am, ha, ro_, g0 = inf["am"], (inf["a1"] - inf["a0"]) / 2, inf["r_out"], inf["g0"]

        def special(co, n):
            r = np.hypot(co[:, 0], co[:, 1])
            ex = (ha - np.abs(_wrapf(_ang(co) - am))) * r
            e = _c01(1.0 - np.minimum(np.minimum(ex, ro_ - r), (r - Ro) * 0.7) / 0.05)
            return _c01(p.moss * w_np(co) * (1.25 * e + 0.35 * nz_a(co, n)))

        def wear(co, n):
            g = np.exp(-(_wrapf(_ang(co) - a_d) / 0.7) ** 2)
            return _c01(_c01(n[:, 2]) * (0.15 + 0.6 * g) * (0.4 + 0.6 * nz_b(co, n)))

        def grime(co, n):
            r = np.hypot(co[:, 0], co[:, 1])
            g = np.exp(-(_wrapf(_ang(co) - a_d) / 0.7) ** 2)
            dirt = 0.4 * _c01(1.0 - (r - Ro) / 0.09) * (1.0 - 0.6 * g) + 0.3 * _c01(1.0 - co[:, 2] / 0.05)
            return _c01(np.maximum(0.1 + 0.22 * g0 + 0.16 * nz_s(co, n) + dirt * (0.5 + 0.5 * nz_b(co, n)), wet_halo(co, n)))
        paint(inf["o"], grime=grime, wear=wear, special=special)

    for inf in steps:
        step_paint(inf)
    for pd in pads:
        paint(pd, grime=lambda co, n: _c01(0.14 + 0.2 * nz_s(co, n) + 0.3 * _c01(1.0 - co[:, 2] / 0.04)),
              special=lambda co, n: _c01(p.moss * (0.9 * nz_b(co, n) + _c01(1.0 - co[:, 2] / 0.05))))
    if wet_on:
        for inf in cope + curb + steps:
            assign_faces(inf["o"], wet_m, lambda c, n: wet_w(c) > 0.0)

    def mortar_special(co, n):
        inner = (np.hypot(co[:, 0], co[:, 1]) < Ri + wall * 0.5) & (co[:, 2] < zc_top + 0.001)
        low = np.clip(1.25 - (co[:, 2] - zf) / (0.8 * Hc), 0.3, 1.0)
        algae = (0.35 + 0.65 * p.moss) * _c01(1.0 - (co[:, 2] - z_water) / 0.3)
        return _c01(np.where(inner, algae, p.moss * 1.8 * w_np(co) * low))

    for o in mortar:                                                 # packed dirt / old mortar, green where it stays damp
        assign(o, stone)
        paint(o, grime=0.6, special=mortar_special)

    # iron cramps let into the coping across every joint (run in lead), each bedded on its own two stones
    cr = box("cramp", (0.15, 0.022, 0.012), bevel=0.003)
    rcr = Ro - wall * 0.5 + 0.02
    crm = []
    for i_ in cope:
        zh = []
        for da in (-0.045, 0.045):
            aa = i_["a0"] + da / rcr
            hit = bvh_cope.ray_cast(Vector((rcr * sin(aa), -rcr * cos(aa), Hc + 0.1)), -up_dir, 0.3)
            if hit[0] is not None:
                zh.append(hit[0].z)
        crm.append(trs((rcr * sin(i_["a0"]), -rcr * cos(i_["a0"]), (min(zh) if zh else Hc) - 0.0035), (0, 0, degrees(i_["a0"]))))
    irons.append(instances_to_mesh("cramps", cr, crm, keep_base=False))

    # --- duckweed drifting on the water in the shaft
    weeds = []
    rdw = rng.fork("weed")
    for _ in range(int(3 + 14 * p.moss)):
        a = rdw.u(0, 2 * pi)
        rr = (Ri - 0.012) * (1.0 - rdw.u(0, 1) ** 2 * 0.45)
        s_ = rdw.u(0.012, 0.04)
        weeds.append(_blob("duckweed", rr * sin(a), -rr * cos(a), z_water + 0.0025, s_, s_ * rdw.u(0.5, 0.9), rdw.u(0, pi), rdw, nr=1, ns=7))

    # ================================================================= moss cushions (real geometry)
    rm_ = rng.fork("moss")
    clumps = []

    def clump(P, n, t, L, W, H):
        clumps.append(_frame(P, n, t, (L * 0.5, W * 0.5, H)))

    def lowf(zq):
        return clamp(1.15 - (zq - zf) / (0.7 * Hc), 0.12, 1.0)

    if p.moss > 0.02:
        # skirt at the foot of the curb
        for _ in range(int(240 * K * Ro / 0.75)):
            a = rm_.u(0, 2 * pi)
            if rm_.u() > w_a(a) or abs(abs(_wrapf(a)) - pi / 2) < (hw + 0.02) / Ro:
                continue
            rr = Ro + rm_.u(-0.004, 0.018)
            clump((rr * sin(a), -rr * cos(a), zf + 0.002), (0.6 * sin(a), -0.6 * cos(a), 0.8), (cos(a), sin(a), 0),
                  rm_.u(0.035, 0.11), rm_.u(0.02, 0.045), rm_.u(0.012, 0.028))
        # joints of the curb: perpends and beds
        for ci, (cz0, cz1, infos) in enumerate(courses):
            for inf in infos:
                a = inf["a0"]
                za_, zb__ = max(cz0, zf) + 0.02, cz1 - 0.02
                if zb__ <= za_:
                    continue
                for _ in range(3):
                    if rm_.u() < w_a(a) * K * 1.6 * lowf((za_ + zb__) / 2):
                        zz = rm_.u(za_, zb__)
                        rr = Ro - 0.008
                        clump((rr * sin(a), -rr * cos(a), zz), (sin(a), -cos(a), 0), (0, 0, 1),
                              rm_.u(0.03, 0.08), rm_.u(0.016, 0.03), rm_.u(0.01, 0.018))
            if ci >= 1 and cz0 > zf + 0.03:
                for _ in range(int(2 * pi * Ro / 0.1)):
                    a = rm_.u(0, 2 * pi)
                    if rm_.u() < w_a(a) * K * 1.3 * lowf(cz0):
                        rr = Ro - 0.008
                        clump((rr * sin(a), -rr * cos(a), cz0), (sin(a), -cos(a), 0), (cos(a), sin(a), 0),
                              rm_.u(0.04, 0.12), rm_.u(0.016, 0.028), rm_.u(0.01, 0.018))
        # joints of the flags and their outer foot on the paving
        for inf in steps:
            a = inf["a0"]
            for _ in range(4):
                if rm_.u() < w_a(a) * K * 1.5:
                    rr = rm_.u(Ro + 0.04, inf["r_out"] - 0.03)
                    clump((rr * sin(a), -rr * cos(a), step_h - 0.014), (0, 0, 1), (sin(a), -cos(a), 0),
                          rm_.u(0.04, 0.12), rm_.u(0.018, 0.03), rm_.u(0.012, 0.02))
        if steps:
            for _ in range(int(150 * K * (Ro + 0.5) / 1.25)):
                a = rm_.u(0, 2 * pi)
                if rm_.u() > w_a(a):
                    continue
                for inf in steps:
                    if (a - inf["a0"]) % (2 * pi) < inf["a1"] - inf["a0"]:
                        rr = inf["r_out"] + rm_.u(0.0, 0.008)
                        clump((rr * sin(a), -rr * cos(a), 0.002), (0.6 * sin(a), -0.6 * cos(a), 0.8), (cos(a), sin(a), 0),
                              rm_.u(0.035, 0.1), rm_.u(0.018, 0.04), rm_.u(0.01, 0.024))
                        break
        # coping: in the joints and as small lichen cushions on top
        for inf in cope:
            a = inf["a0"]
            for _ in range(2):
                if rm_.u() < w_a(a) * K * 1.2:
                    rr = rm_.u(r_lip + 0.07, r_cop - 0.03)
                    clump((rr * sin(a), -rr * cos(a), Hc - 0.013), (0, 0, 1), (sin(a), -cos(a), 0),
                          rm_.u(0.03, 0.07), rm_.u(0.014, 0.022), rm_.u(0.011, 0.016))
        for _ in range(int(70 * K * Ro / 0.75)):
            a = rm_.u(0, 2 * pi)
            if rm_.u() > w_a(a) or abs(_wrapf(a - a_d)) < 0.5:
                continue
            rr = rm_.u(r_lip + 0.09, r_cop - 0.035)
            s_ = rm_.u(0.018, 0.042)
            clump((rr * sin(a), -rr * cos(a), Hc - 0.007), (0, 0, 1), (1, 0, 0), s_, s_ * rm_.u(0.7, 1.0), rm_.u(0.008, 0.012))

    # ================================================================= posts
    rw_ = rng.fork("timber")
    pegs, bolts_m = [], []                         # matrices for oak pegs and iron bolt heads
    posts = []
    if not iron_posts:
        top = He
        zb0_ = z0 - 0.02
        for sx in (-1, 1):
            po = plank(f"post{sx}", top - zb0_, 2 * hw, 2 * hw, loc=(sx * xp, 0, (top + zb0_) / 2), bevel=0.012, rng=rw_, axis='Z')
            densify(po, 0.075)
            posts.append(po)
            exposed.append(po)
            if p.moss > 0.02:                      # cushions round the damp foot
                for _ in range(int(2 + 9 * K)):
                    fy = rm_.pick((-1, 1, 1))
                    if rm_.chance(0.5):
                        P_ = (sx * xp + rm_.u(-hw, hw), fy * (hw + 0.004), z0 + 0.002)
                        n_ = (0, fy * 0.6, 0.8)
                        t_ = (1, 0, 0)
                    else:
                        fx_ = rm_.pick((-1, 1))
                        P_ = (sx * xp + fx_ * (hw + 0.004), rm_.u(-hw, hw), z0 + 0.002)
                        n_ = (fx_ * 0.6, 0, 0.8)
                        t_ = (0, 1, 0)
                    clump(P_, n_, t_, rm_.u(0.03, 0.08), rm_.u(0.018, 0.035), rm_.u(0.012, 0.024))
    else:
        bar = rounded_rect(0.052, 0.052, 0.006, 2)
        for sx in (-1, 1):
            # base plate bolted down, foot scroll buttress bound to the upright with a collar
            irons.append(box(f"foot{sx}", (0.2, 0.16, 0.02), loc=(sx * (xp + 0.02), 0, z0 + 0.004), bevel=0.003))
            for bx in (-0.06, 0.1):
                for by in (-0.055, 0.055):
                    bolts_m.append(trs((sx * (xp + bx), by, z0 + 0.014), (0, 0, rw_.u(0, 60))))
            r1 = 0.11
            sc = spiral_path(0.014, r1, 1.5, n=40, a0_deg=0 if sx > 0 else 180, plane="XZ",
                             center=(sx * (xp + 0.026 + 0.005 + r1), 0, z0 + 0.014 + r1 + 0.006))
            irons.append(sweep(f"foot_scroll{sx}", sc, rounded_rect(0.009, 0.03, 0.002, 2), up=(0, 1, 0),
                               scale_fn=lambda t: 0.55 + 0.45 * t))
            irons.append(box(f"foot_collar{sx}", (0.08, 0.04, 0.022), loc=(sx * (xp + 0.012), 0, z0 + 0.02 + r1), bevel=0.003))
            for zk in (z0 + 0.36, Zw - 0.2, Zw + 0.2):
                irons.append(box(f"knop{sx}", (0.068, 0.068, 0.03), loc=(sx * xp, 0, zk), bevel=0.006))
        if has_roof:
            for sx in (-1, 1):
                irons.append(sweep(f"upright{sx}", [(sx * xp, 0, z0), (sx * xp, 0, He)], bar, up=(0, 1, 0), smooth_path=False))
                irons.append(box(f"saddle{sx}", (0.15, 0.3, 0.014), loc=(sx * xp, 0, He - 0.007), bevel=0.003))
                for sy in (-1, 1):
                    q0, q3 = (sx * xp, sy * 0.02, He - 0.36), (sx * xp, sy * 0.3, He - 0.012)
                    irons.append(sweep(f"stay{sx}{sy}", bezier_path(q0, (sx * xp, sy * 0.2, He - 0.34), (sx * xp, sy * 0.3, He - 0.2), q3, n=12),
                                       rounded_rect(0.008, 0.03, 0.002, 2), up=(1, 0, 0)))
                    bolts_m.append(trs((sx * xp, sy * 0.12, He - 0.014), (180, 0, 0)))
            zs, ra = He - 0.5, 0.4
            arch = [(xp * cos(pi * i / 28), 0, zs + ra * sin(pi * i / 28)) for i in range(29)]
            irons.append(sweep("arch", arch, rounded_rect(0.034, 0.034, 0.004, 2), up=(0, 1, 0), smooth_path=False))
            apex_z = zs + ra
        else:
            zs = Zw + 0.3
            ra = clamp(xp * 0.8, 0.45, 0.85)
            path = [(xp, 0, z0)] + [(xp * cos(pi * i / 36), 0, zs + ra * sin(pi * i / 36)) for i in range(37)] + [(-xp, 0, z0)]
            irons.append(sweep("arch", path, bar, up=(0, 1, 0), smooth_path=False))
            apex_z = zs + ra
            # crest: twin scrolls and a spiked finial on the crown, hook underneath
            for sx in (-1, 1):
                r1 = 0.085
                sc = spiral_path(0.012, r1, 1.5, n=36, a0_deg=90, plane="XZ", ccw=sx > 0,
                                 center=(sx * (r1 + 0.03), 0, apex_z + 0.026 + r1 - 0.004))
                irons.append(sweep(f"crest{sx}", sc, rounded_rect(0.008, 0.028, 0.002, 2), up=(0, 1, 0),
                                   scale_fn=lambda t: 0.55 + 0.45 * t))
            irons.append(lathe("finial", [(0.0, apex_z + 0.02), (0.03, apex_z + 0.024), (0.034, apex_z + 0.04), (0.016, apex_z + 0.06),
                                          (0.012, apex_z + 0.11), (0.03, apex_z + 0.15), (0.034, apex_z + 0.18), (0.02, apex_z + 0.22),
                                          (0.008, apex_z + 0.3), (0.0, apex_z + 0.36)], segments=12))
            irons.append(torus("apex_ring", 0.022, 0.005, loc=(0, 0, apex_z - 0.05), rot=(90, 0, 0), major_seg=14, minor_seg=6))

    # strap tying each post to the curb just under the coping (let 3 mm into the timber: no coincident faces)
    zs_ = zc_top - 0.09
    rr_run = rng.fork("rust")
    for sx in (-1, 1):
        e = hw + 0.002
        strap = [(sx * (Ro - 0.07), -e, zs_), (sx * (xp + e), -e, zs_), (sx * (xp + e), e, zs_), (sx * (Ro - 0.07), e, zs_)]
        irons.append(sweep(f"strap{sx}", strap, rounded_rect(0.01, 0.045, 0.002, 2), up=(0, 0, 1), smooth_path=False))
        bolts_m.append(trs((sx * (xp + e + 0.005), 0.0, zs_), (0, sx * 90, 0)))
        if not iron_posts:
            for sy in (-1, 1):
                bolts_m.append(trs((sx * xp, sy * (e + 0.005), zs_), (-sy * 90, 0, 0)))
            # rust and black iron-tannin bleeding out from under the strap, on the three exposed faces
            for N in (Vector((0, -1, 0)), Vector((0, 1, 0)), Vector((sx, 0, 0))):
                S = N.cross(Vector((0, 0, -1)))
                for _ in range(rr_run.pick((2, 3, 3))):
                    P0 = Vector((sx * xp, 0, zs_ - 0.018)) + N * hw + S * rr_run.u(-hw + 0.02, hw - 0.02)
                    stains.append((P0, N, S, rr_run.u(0.08, 0.3), rr_run.u(0.02, 0.045), rr_run.u(0, 2 * pi), rr_run.u(0.002, 0.008)))

    # ================================================================= roof frame / lintel
    top_z = He
    if has_roof:
        pr = radians(p.pitch)
        tp, cp, sp = tan(pr), cos(pr), sin(pr)
        plate_y = max(0.5, Ro * 0.8)
        eave_y = plate_y + 0.26
        tie_h, tie_w, pl = 0.15, 0.13, 0.1
        Lr = 2 * (xp + hw + 0.3)                  # roof length, verge to verge
        bt, rd, rwd = 0.022, 0.08, 0.06           # batten thickness, rafter depth / width
        za = He + tie_h + pl - 0.02 + plate_y * tp + (bt + rd) / cp   # apex of the roof plane (top of battens)
        Ls = eave_y / cp
        zr_top = za - bt / cp - 0.035 * tp - 0.006
        for sx in (-1, 1):
            tie = plank(f"tie{sx}", 2 * plate_y + 0.26, tie_w, tie_h, loc=(sx * xp, 0, He + tie_h / 2), bevel=0.01,
                        warp=0.004, rng=rw_, axis='Y')
            exposed.append(tie)
            kb = He + tie_h
            sheltered.append(plank(f"king{sx}", zr_top - 0.13 - kb, 0.1, 0.1, loc=(sx * xp, 0, (zr_top - 0.13 + kb) / 2), bevel=0.008,
                                   rng=rw_, axis='Z'))
            for sy in (-1, 1):
                # raking struts from the tie to the rafters, knee braces from the post to the tie
                ys = plate_y * 0.55
                zr_under = za - ys * tp - (bt + rd) / cp
                sheltered.append(_beam(f"strut{sx}{sy}", (sx * xp, sy * 0.06, kb - 0.01), (sx * xp, sy * ys, zr_under + 0.01), 0.06, 0.06, rng=rw_))
                if not iron_posts:
                    exposed.append(_beam(f"brace{sx}{sy}", (sx * xp, sy * (hw - 0.02), He - 0.44), (sx * xp, sy * (hw + 0.4), He + 0.02),
                                         0.075, 0.09, bevel=0.008, rng=rw_))
                    pegs.append(trs((sx * (xp + 0.0375 * sx), sy * (hw + 0.33), He + 0.07), (0, sx * 90, 0)))
                    pegs.append(trs((sx * xp + sx * hw, sy * 0.02, He - 0.37), (0, sx * 90, 0)))
            pegs.append(trs((sx * xp + sx * tie_w / 2, 0.0, He + 0.07), (0, sx * 90, 0)))
        sheltered.append(plank("ridge", Lr - 0.12, 0.07, 0.14, loc=(0, 0, zr_top - 0.07), bevel=0.006, warp=0.004, rng=rw_))
        apex = Vector((0, 0, za))
        n_raf = max(3, int(round(Lr / 0.46)) + 1)
        for sy in (-1, 1):
            n = Vector((0, sy * sp, cp))
            d = Vector((0, sy * cp, -sp))
            sheltered.append(plank(f"plate{sy}", Lr - 0.12, pl, pl, loc=(0, sy * plate_y, He + tie_h + pl / 2), bevel=0.006,
                                   warp=0.004, rng=rw_))
            u0 = (0.03 + (bt + rd) * sp) / cp
            for i in range(n_raf):
                x = -Lr / 2 + 0.1 + (Lr - 0.2) * i / (n_raf - 1)
                c = Vector((x, 0, 0)) + apex - n * (bt + rd / 2)
                sheltered.append(_beam(f"rafter{sy}_{i}", c + d * u0, c + d * (Ls - 0.015), rwd, rd, bevel=0.004, rng=rw_))
        top_z = za
    elif not iron_posts:
        # open well: lintel across the posts with knee braces and a weathering board
        exposed.append(plank("lintel", 2 * xp + 0.56, 0.16, 0.17, loc=(0, 0, He + 0.085), bevel=0.012, warp=0.003, rng=rw_))
        cap = plank("lintel_cap", 2 * xp + 0.68, 0.24, 0.028, loc=(0, 0, He + 0.17 + 0.014), bevel=0.004, warp=0.006, rng=rw_)
        densify(cap, 0.12)
        exposed.append(cap)
        for sx in (-1, 1):
            exposed.append(_beam(f"brace{sx}", (sx * (xp - hw + 0.02), 0, He - 0.46), (sx * (xp - hw - 0.42), 0, He + 0.02), 0.08, 0.09,
                                 bevel=0.008, rng=rw_))
            for sy in (-1, 1):
                pegs.append(trs((sx * xp + rw_.u(-0.02, 0.02), sy * 0.08, He + 0.085), (-sy * 90, 0, 0)))
            pegs.append(trs((sx * (xp - 0.01), -hw, He - 0.4), (90, 0, 0)))
            pegs.append(trs((sx * (xp - hw - 0.34), -0.08, He + 0.06), (90, 0, 0)))
            irons.append(sweep(f"lintel_strap{sx}", [(sx * xp, -0.082, He - 0.14), (sx * xp, -0.082, He + 0.202), (sx * xp, 0.082, He + 0.202),
                                                     (sx * xp, 0.082, He - 0.14)], rounded_rect(0.05, 0.012, 0.002, 2), up=(1, 0, 0), smooth_path=False))
            for sy in (-1, 1):
                bolts_m.append(trs((sx * xp, sy * 0.088, He - 0.09), (-sy * 90, 0, 0)))
        top_z = He + 0.2

    # ================================================================= roof covering
    roof_objs = []
    if has_roof:
        rr_ = rng.fork("roof")
        tile = p.roof == "tile"
        levels = (0.0, 0.3, 0.5, 0.72, 0.96)           # moss level of a shingle / tile (painted on the base, then instanced)
        if tile:
            tw, tl, tth, gauge, lift = 0.18, 0.29, 0.013, 0.115, 0.036
            cover = mat("terracotta", "#c8775a")
            h_under = lift * (1 - gauge / tl) + tth
        else:
            tw, tl, tth, gauge, lift = 0.105, 0.34, 0.014, 0.125, 0.032
            cover = timber
            h_under = lift * (1 - gauge / tl) + tth * (1 - 0.7 * gauge / tl)

        def make_base(lv):
            if tile:
                pts = [(tw / 2, tl), (-tw / 2, tl)] + [(tw / 2 * cos(radians(a)), 0.05 + 0.05 * sin(radians(a))) for a in (180, 225, 270, 315, 360)]
                b_ = extrude_profile("tile_base", pts, tth, bevel=0.0)
            else:
                v = [(-tw / 2, 0, 0), (tw / 2, 0, 0), (tw / 2, 0, tth), (-tw / 2, 0, tth),
                     (-tw / 2, tl, 0), (tw / 2, tl, 0), (tw / 2, tl, tth * 0.3), (-tw / 2, tl, tth * 0.3)]
                f = [(0, 1, 2, 3), (3, 2, 6, 7), (0, 4, 5, 1), (1, 5, 6, 2), (4, 0, 3, 7)]     # head face never seen
                b_ = mesh_object("shingle_base", v, f)
                uv_grain(b_, 'Y')
                flat(b_)
            assign(b_, cover)
            realize(b_)
            if tile:
                _cull(b_, lambda c, n: n.y > 0.9 and c.y > tl - 0.01)       # head face: always under the course above
            # growth is thickest at the butt, where the lap above keeps the wood wet
            paint(b_, special=lambda co, n: _c01(lv * (1.0 - 0.5 * co[:, 1] / tl) * np.ones(len(co))), mode='set')
            return b_

        bases = [make_base(lv) for lv in levels]
        mats = [[] for _ in levels]
        ov = 0.035
        for sy in (-1, 1):
            R = (Matrix.Rotation(pi, 4, 'Z') if sy > 0 else Matrix.Identity(4)) @ Matrix.Rotation(pr, 4, 'X')
            d_s = Vector((0, sy * cp, -sp))
            n_s = Vector((0, sy * sp, cp))
            side_f = 1.0 if sy > 0 else 0.5      # the north slope stays damp
            # battens (one per course) and the tilting fillet at the eaves
            kc = 0
            while True:
                ub = Ls + ov - kc * gauge             # slope distance apex -> butt of this course
                if ub < 0.07:
                    break
                ln = min(tl, ub - 0.004)
                ubat = ub - tl + 0.04
                if ubat > 0.05:
                    c = apex + d_s * ubat - n_s * (bt / 2)
                    sheltered.append(plank(f"batten{sy}_{kc}", Lr - 0.12, 0.045, bt, loc=tuple(c), rot=(-sy * p.pitch, 0, 0), bevel=0.0,
                                           rng=rw_))
                lowr = 0.3 + 0.7 * ub / Ls
                # moss cushions tucked under the butts of this course
                if kc >= 1 and p.moss > 0.02:
                    for _ in range(int(Lr / 0.1)):
                        if rm_.u() < K * (0.6 if sy > 0 else 0.2) * lowr:
                            Pm = apex + d_s * (ub + rm_.u(0.006, 0.016)) + n_s * (h_under - 0.003) + Vector((rm_.u(-Lr / 2 + 0.04, Lr / 2 - 0.04), 0, 0))
                            clump(Pm, n_s, (1, 0, 0), rm_.u(0.03, 0.09), rm_.u(0.016, 0.03), rm_.u(0.009, 0.017))
                # the course itself
                x = -Lr / 2 + (rr_.u(-0.5, 0.0) * tw if not tile else (-0.5 * tw if kc % 2 else 0.0))
                while x < Lr / 2 - 0.01:
                    w = tw if tile else rr_.u(0.07, 0.135)
                    x0_, x1_ = max(x, -Lr / 2), min(x + w, Lr / 2)
                    wv = x1_ - x0_ - (0.003 if tile else 0.004)
                    x += w
                    if wv < 0.03:
                        continue
                    slip = 0.0
                    if rr_.chance(0.035) and 1 < kc:
                        slip = rr_.u(0.015, 0.05)      # a slipped tile / shingle
                    xm_ = (x0_ + x1_) / 2
                    P = apex + Vector((xm_, sy * cp * (ub + slip), -sp * (ub + slip)))
                    lf = lift * (rr_.u(0.99, 1.04) if tile else rr_.u(0.9, 1.2)) + (0.003 if tile else rr_.u(0.0, 0.008))
                    tilt = asin(min(0.5, lf / tl))
                    M = (Matrix.Translation(P) @ R @ Matrix.Translation((0, 0, lf)) @ Matrix.Rotation(-tilt, 4, 'X')
                         @ Matrix.Rotation(radians(rr_.u(-1.5, 1.5) * (0.6 if tile else 1.6)), 4, 'Z')
                         @ Matrix.Rotation(radians(rr_.u(-2.5, 2.5) * (0.4 if tile else 1.0)), 4, 'Y')
                         @ Matrix.Diagonal((wv / tw, ln / tl * (1.0 if tile else rr_.u(0.97, 1.0)), rr_.u(0.95, 1.1) if tile else rr_.u(0.85, 1.25), 1.0)))
                    cl = 0.5 + 0.5 * mnoise.noise(Vector((xm_ * 1.5, ub * 1.8 + sy * 7.3, seed * 0.131)))
                    mm = p.moss * side_f * lowr * (0.25 + 1.5 * cl) + rr_.u(-0.07, 0.07)
                    li = 0 if mm < 0.2 else 1 if mm < 0.33 else 2 if mm < 0.47 else 3 if mm < 0.62 else 4
                    mats[li].append(M)
                kc += 1
        for li, b_ in enumerate(bases):
            if mats[li]:
                roof_objs.append(instances_to_mesh(f"roof_cover{li}", b_, mats[li], keep_base=False))
            else:
                delete([b_])
        # barge boards with plumb-cut feet, mitred at the apex
        dv = 0.15 / cp
        ye = eave_y + 0.012
        for sx in (-1, 1):
            xb0 = sx * (Lr / 2 - 0.035)
            for sy in (-1, 1):
                poly = [(0.0, za + 0.004), (sy * ye, za + 0.004 - ye * tp), (sy * ye, za - ye * tp - dv * 0.62),
                        (sy * (ye - 0.07), za - (ye - 0.07) * tp - dv), (0.0, za - dv)]
                bb = _prism_yz(f"barge{sx}{sy}", poly, xb0, xb0 - sx * 0.026)
                uv_grain(bb, axis=(0, sy * cp, -sp))
                exposed.append(bb)
            # turned finial / pendant on each gable
            fz = za - dv - 0.1
            fin = lathe(f"gable_finial{sx}", [(0.0, fz), (0.014, fz + 0.004), (0.026, fz + 0.03), (0.016, fz + 0.055), (0.03, fz + 0.075),
                                              (0.03, za + 0.1), (0.022, za + 0.115), (0.036, za + 0.15), (0.03, za + 0.185),
                                              (0.012, za + 0.21), (0.016, za + 0.24), (0.0, za + 0.27)], segments=14,
                        loc=(sx * (Lr / 2 - 0.03), 0, 0))
            uv_transform(fin, rot_deg=90)
            exposed.append(fin)
        # ridge capping
        ridge_objs = []
        if tile:
            rr0, rt = 0.105, 0.015
            prof = [(rr0 * cos(radians(a)), rr0 * sin(radians(a))) for a in range(0, 181, 20)] + \
                   [((rr0 - rt) * cos(radians(a)), (rr0 - rt) * sin(radians(a))) for a in range(180, -1, -20)]
            zc = za + 0.05 / cp - (rr0 - rt) * tp * 0.85
            nseg = int(ceil((Lr + 0.04) / 0.33))
            sl = (Lr + 0.04) / nseg
            for i in range(nseg):
                x0_ = -Lr / 2 - 0.02 + i * sl
                rt_ = sweep(f"ridge_tile{i}", [(x0_ - 0.012, 0, zc + rr_.u(-0.004, 0.004)), (x0_ + sl, 0, zc + 0.012)], prof, up=(0, 0, 1),
                            smooth_path=False, scale_fn=lambda t: 1.0 - 0.09 * t)
                smooth(rt_, 50)
                assign(rt_, cover)
                ridge_objs.append(rt_)
        else:
            for sy in (-1, 1):
                n = Vector((0, sy * sp, cp))
                d = Vector((0, sy * cp, -sp))
                c = apex + d * (0.074 + (0.01 if sy > 0 else 0.0)) + n * (0.05 + (0.02 if sy > 0 else 0.0))
                half = (Lr + 0.03) / 2
                xb_ = rr_.u(-0.3, 0.3)
                for q, (s0, s1) in enumerate(((-half, xb_ - 0.002), (xb_ + 0.002, half))):
                    rb = plank(f"ridge_board{sy}{q}", s1 - s0, 0.16, 0.02, loc=((s0 + s1) / 2, c.y, c.z), rot=(-sy * p.pitch, 0, 0),
                               bevel=0.003, warp=0.004, rng=rw_)
                    assign(rb, cover)
                    ridge_objs.append(rb)

        def roof_grime(co, n):
            streak = nz_b(co * np.array([1.0, 0.12, 0.12]), n)       # rain streaks running down the slope
            return _c01(0.03 + 0.13 * nz_c(co, n) + 0.2 * streak * (0.3 + 0.7 * _c01((za - co[:, 2]) / (eave_y * tp))))

        for o in roof_objs:
            paint(o, grime=roof_grime, wear=lambda co, n: 0.45 * nz_c(co, n))
        for o in ridge_objs:
            paint(o, grime=roof_grime, wear=lambda co, n: 0.45 * nz_c(co, n),
                  special=lambda co, n: _c01(p.moss * 0.9 * nz_b(co, n) * _c01(0.45 + co[:, 1] * 4.0)))
        roof_objs += ridge_objs

    # ================================================================= windlass
    rdr = 0.085                                   # drum radius
    Ld = 2 * (xp - hw) - 0.11
    dprof = [(0.0, -Ld / 2), (0.112, -Ld / 2), (0.114, -Ld / 2 + 0.034), (0.1, -Ld / 2 + 0.04), (rdr, -Ld / 2 + 0.05)]
    for i in range(1, 12):
        dprof.append((rdr - 0.002 * sin(pi * i / 12), -Ld / 2 + 0.05 + (Ld - 0.1) * i / 12))
    dprof += [(rdr, Ld / 2 - 0.05), (0.1, Ld / 2 - 0.04), (0.114, Ld / 2 - 0.034), (0.112, Ld / 2), (0.0, Ld / 2)]
    drum = lathe("drum", dprof, segments=22, loc=(0, 0, Zw), rot=(0, 90, 0))
    uv_transform(drum, rot_deg=90)
    assign(drum, timber)
    for sx in (-1, 1):
        irons.append(ring_band(f"drum_hoop{sx}", 0.114, 0.024, 0.004, loc=(sx * (Ld / 2 - 0.016), 0, Zw), rot=(0, 90, 0), segments=22))
    xa0, xa1 = -(xp + hw + 0.05), xp + hw + 0.11
    irons.append(cylinder("axle", 0.017, xa1 - xa0, loc=((xa0 + xa1) / 2, 0, Zw), rot=(0, 90, 0), verts=12))
    irons.append(lathe("axle_cap", [(0.0, 0.0), (0.03, 0.0), (0.032, 0.008), (0.02, 0.016), (0.0, 0.02)], segments=10,
                       loc=(xa0, 0, Zw), rot=(0, -90, 0)))
    for sx in (-1, 1):
        for side in (-1, 1):                      # bearing plates on both faces of each post, let 4 mm into the timber
            xf = sx * xp + side * (hw + 0.002)
            irons.append(box(f"bearing{sx}{side}", (0.012, min(0.11, 2 * hw + 0.05), 0.16), loc=(xf, 0, Zw), bevel=0.003))
            irons.append(cylinder(f"boss{sx}{side}", 0.034, 0.02, loc=(xf + side * 0.01, 0, Zw), rot=(0, 90, 0), verts=12, bevel=0.003))
            for bz in (-0.06, 0.06):
                bolts_m.append(trs((xf + side * 0.006, 0.0 if iron_posts else rw_.u(-0.012, 0.012), Zw + bz), (0, side * 90, 0)))
            if not iron_posts:                    # ... and down the post below each bearing plate
                N = Vector((side, 0, 0))
                S = N.cross(Vector((0, 0, -1)))
                for _ in range(rr_run.pick((2, 3))):
                    P0 = Vector((sx * xp + side * hw, 0, Zw - 0.074)) + S * rr_run.u(-0.045, 0.045)
                    stains.append((P0, N, S, rr_run.u(0.12, 0.4), rr_run.u(0.022, 0.048), rr_run.u(0, 2 * pi), rr_run.u(0.002, 0.008)))
    # ratchet wheel and pawl inside the left post
    xr = -(xp - hw - 0.03)
    teeth = []
    nt = 14
    for i in range(nt):
        a0, a1 = 2 * pi * i / nt, 2 * pi * (i + 1) / nt
        teeth += [(0.075 * cos(a0), 0.075 * sin(a0)), (0.056 * cos(a1 - 0.03), 0.056 * sin(a1 - 0.03))]
    irons.append(shape_extrude("ratchet", teeth, [circle_pts(0.03, 10)], depth=0.012, loc=(xr, 0, Zw), rot=(0, 90, 0)))
    irons.append(_beam("pawl", (xr + 0.006, 0.02, Zw + 0.17), (xr + 0.006, -0.03, Zw + 0.07), 0.01, 0.022, bevel=0.002))
    irons.append(cylinder("pawl_pin", 0.008, 0.07, loc=(xr - 0.012, 0.02, Zw + 0.17), rot=(0, 90, 0), verts=8))
    # crank: forged bar bent twice, turned grip
    rc = rng.fork("crank")
    ca = radians(rc.u(205, 250))
    xc = xp + hw + 0.085
    hy, hz = 0.25 * cos(ca), Zw + 0.25 * sin(ca)
    irons.append(sweep("crank", [(xc - 0.02, 0, Zw), (xc, 0, Zw), (xc + 0.004, hy * 0.5, (hz + Zw) / 2), (xc, hy, hz), (xc + 0.2, hy, hz)],
                       rounded_rect(0.024, 0.024, 0.005, 2), up=(1, 0, 0), smooth_path=False))
    irons.append(cylinder("crank_hub", 0.032, 0.036, loc=(xc, 0, Zw), rot=(0, 90, 0), verts=12, bevel=0.004))
    irons.append(cylinder("grip_cap", 0.02, 0.008, loc=(xc + 0.2, hy, hz), rot=(0, 90, 0), verts=10))
    grip = lathe("grip", [(0.0, 0.0), (0.019, 0.0), (0.023, 0.012), (0.021, 0.06), (0.024, 0.115), (0.019, 0.13), (0.0, 0.13)],
                 segments=12, loc=(xc + 0.045, hy, hz), rot=(0, 90, 0))
    uv_transform(grip, rot_deg=90)
    assign(grip, timber)
    paint(grip, wear=0.9, grime=0.45)             # dark with hand grease, rubbed smooth

    # rope coiled on the drum, leaving over the front and dropping to the bucket
    rr0_ = 0.0085
    rcoil = rdr + rr0_ - 0.001
    turns = int(p.rope_turns)
    pitch_ = rr0_ * 2.12
    rk = rng.fork("rope")
    xe = clamp(rk.u(-0.12, 0.16), -Ld / 2 + 0.24, Ld / 2 - 0.24)
    xs = max(-Ld / 2 + 0.06, xe - turns * pitch_)
    turns_f = (xe - xs) / pitch_
    npt = int(turns_f * 14) + 1
    coil = []
    for i in range(npt + 1):
        t_ = i / npt
        psi = pi - 2 * pi * turns_f * (1 - t_)
        coil.append((xs + (xe - xs) * t_, rcoil * cos(psi), Zw + rcoil * sin(psi)))
    rope_coil = sweep("rope_coil", coil, rr0_, sides=7, smooth_path=False)
    # a few turns of a second layer riding on the first
    lay2 = []
    t2 = min(3.2, turns_f * 0.4)
    n2 = int(t2 * 14) + 1
    for i in range(n2 + 1):
        t_ = i / n2
        psi = pi * 0.6 + 2 * pi * t2 * t_
        lay2.append((xs + pitch_ * 0.5 + pitch_ * t2 * t_, (rcoil + rr0_ * 1.75) * cos(psi), Zw + (rcoil + rr0_ * 1.75) * sin(psi)))
    rope_lay2 = sweep("rope_lay2", lay2, rr0_, sides=7, smooth_path=False)
    for o in (rope_coil, rope_lay2):
        assign(o, rope_m)
        paint(o, grime=lambda co, n: _c01(0.25 + 0.5 * nz_c(co, n)))
    # the drum is burnished where the rope travels, dark and greasy elsewhere
    paint(drum, grime=lambda co, n: _c01(0.42 + 0.3 * nz_b(co, n)),
          wear=lambda co, n: _c01(0.15 + 0.8 * _c01(1.0 - np.abs(co[:, 0] - (xe + 0.12)) / 0.34) * (0.6 + 0.4 * nz_c(co, n))))

    bx_, by_ = xe, -rcoil
    zb = Hc + 0.01                                # underside of the bucket
    bh = 0.3
    z_knot = zb + bh + 0.19
    swing = []
    if p.bucket == "none":
        z_knot = Hc + 0.22
    drop = sweep("rope_drop", [(bx_, by_, Zw + 0.004), (bx_, by_, z_knot - 0.01)], rr0_, sides=7, smooth_path=False)
    knot = sphere("rope_knot", 0.021, loc=(bx_, by_, z_knot + 0.012), segments=10, rings=6, scale=(1.0, 1.0, 1.25))
    tail = sweep("rope_tail", [(bx_, by_, z_knot + 0.02), (bx_ + 0.03, by_ - 0.012, z_knot + 0.035), (bx_ + 0.05, by_ - 0.02, z_knot - 0.03),
                               (bx_ + 0.055, by_ - 0.022, z_knot - 0.09)], rr0_ * 0.9, sides=6)
    for o in (drop, knot, tail):
        assign(o, rope_m)
        paint(o, grime=lambda co, n: _c01(0.3 + 0.5 * _c01((Zw - co[:, 2]) / 0.6)))
        swing.append(o)
    b_iron, b_wood, b_other = [], [], []
    if p.bucket == "none":
        b_iron.append(sweep("hook", [(bx_, by_, z_knot), (bx_, by_, z_knot - 0.06), (bx_ + 0.03, by_, z_knot - 0.1), (bx_, by_, z_knot - 0.135),
                                     (bx_ - 0.03, by_, z_knot - 0.105)], 0.006, sides=7, scale_fn=lambda t: 1.0 - 0.5 * t))
    else:
        Rt, Rb = (0.155, 0.125) if p.bucket == "wood" else (0.15, 0.108)
        zt = zb + bh
        if p.bucket == "wood":
            ns = 14
            sth = 0.015
            wt = 2 * (Rt) * tan(pi / ns) * 0.985
            st = box("stave", (wt, sth, bh), bevel=0.0025)
            transform_verts(st, lambda co: Vector((co.x * lerp(Rb / Rt, 1.0, co.z / bh + 0.5), co.y, co.z)))
            uv_grain(st, 'Z')
            assign(st, timber)
            realize(st)
            lean = degrees(atan2(Rt - Rb, bh))
            Rm = (Rt + Rb) / 2
            rb_ = rng.fork("bucket")
            sm = []
            for i in range(ns):
                a = 360.0 * i / ns
                ar = radians(a)
                sm.append(trs((bx_ + Rm * sin(ar), by_ - Rm * cos(ar), zb + bh / 2 + rb_.u(-0.004, 0.004)), (lean, 0, a),
                              (1.0, rb_.u(0.9, 1.15), 1.0)))
            b_wood.append(instances_to_mesh("staves", st, sm, keep_base=False))
            bot = cylinder("bucket_bottom", Rb - 0.004, 0.018, loc=(bx_, by_, zb + 0.03), verts=14)
            b_wood.append(bot)
            kcr = 1.0 / cos(pi / ns)
            for hz_ in (0.06, 0.215):
                r0 = (Rb + (Rt - Rb) * (hz_ - 0.02) / bh) * kcr + sth / 2
                r1 = (Rb + (Rt - Rb) * (hz_ + 0.02) / bh) * kcr + sth / 2
                b_iron.append(lathe("hoop", [(r0 - 0.003, zb + hz_ - 0.02), (r0 + 0.003, zb + hz_ - 0.02), (r1 + 0.003, zb + hz_ + 0.02),
                                             (r1 - 0.003, zb + hz_ + 0.02)], segments=28, loc=(bx_, by_, 0)))
            for sx in (-1, 1):                    # ears carrying the bail
                b_wood.append(plank("ear", 0.085, 0.05, 0.016, loc=(bx_ + sx * (Rt + 0.004), by_, zt + 0.02), rot=(0, 0, 90), bevel=0.004, axis='Z'))
            rw_in = Rt - sth - 0.004
            ear_x = Rt + 0.004
        else:
            zinc = mat("zinc")
            pail = lathe("pail", [(0.0, zb + 0.016), (Rb - 0.008, zb + 0.016), (Rb - 0.005, zb), (Rb, zb), (Rb + 0.003, zb + 0.018), (Rb, zb + 0.024),
                                  (lerp(Rb, Rt, 0.33), zb + bh * 0.33), (lerp(Rb, Rt, 0.36) + 0.004, zb + bh * 0.345), (lerp(Rb, Rt, 0.38), zb + bh * 0.36),
                                  (lerp(Rb, Rt, 0.7), zb + bh * 0.7), (lerp(Rb, Rt, 0.72) + 0.004, zb + bh * 0.715), (lerp(Rb, Rt, 0.74), zb + bh * 0.73),
                                  (Rt, zt - 0.012), (Rt + 0.007, zt - 0.008), (Rt + 0.008, zt - 0.002), (Rt + 0.003, zt), (Rt - 0.002, zt - 0.004),
                                  (Rb - 0.002, zb + 0.03), (0.0, zb + 0.03)], segments=28, loc=(bx_, by_, 0))
            assign(pail, zinc)
            paint(pail, grime=lambda co, n: _c01(0.2 + 0.6 * _c01(1 - (co[:, 2] - zb) / 0.1) + 0.4 * nz_c(co, n)),
                  special=lambda co, n: _c01(0.8 * nz_c(co, n) * _c01(1 - (co[:, 2] - zb) / 0.12)), wear=lambda co, n: 0.5 * nz_c(co, n))
            b_other.append(pail)
            for sx in (-1, 1):
                b_iron.append(box("lug", (0.006, 0.04, 0.06), loc=(bx_ + sx * (Rt + 0.006), by_, zt - 0.012), bevel=0.002))
            rw_in = Rt - 0.004
            ear_x = Rt + 0.008
        # bail: forged rod through the ears, eye at the top for the rope
        bail = [(bx_ + ear_x * cos(pi * i / 16), by_, zt + 0.02 + 0.16 * sin(pi * i / 16)) for i in range(17)]
        b_iron.append(sweep("bail", bail, 0.0065, sides=7, smooth_path=False))
        for sx in (-1, 1):
            b_iron.append(cylinder("bail_pin", 0.011, 0.012, loc=(bx_ + sx * (ear_x + 0.01), by_, zt + 0.02), rot=(0, 90, 0), verts=8))
        b_iron.append(torus("bail_eye", 0.016, 0.005, loc=(bx_, by_, z_knot - 0.006), rot=(0, 90, 0), major_seg=12, minor_seg=6))
        wl = zb + bh * 0.78
        bw = lathe("bucket_water", [(lerp(Rb, Rt, 0.78) - (Rt - rw_in), wl), (0.0, wl)], segments=14, loc=(bx_, by_, 0))
        assign(bw, water_m)
        b_other.append(bw)
        for o in b_wood:                          # soaked dark staves, pale rubbed rim
            assign(o, timber)
            paint(o, grime=lambda co, n: _c01(0.55 + 0.35 * _c01(1 - (co[:, 2] - zb) / 0.14)),
                  wear=lambda co, n: _c01(0.5 * nz_c(co, n) + 0.5 * _c01((co[:, 2] - (zt - 0.03)) / 0.03)), special=0.3 * p.moss)
        fx("drip", "drip", (bx_ + Rb * 0.6, by_, zb), rate=0.6)
    for o in b_iron:
        assign(o, iron)
        paint(o, special=lambda co, n: _c01(0.1 + 0.4 * nz_c(co, n) * nz_b(co, n)))
    swing += b_iron + b_wood + b_other
    part(swing, "bucket", pivot=(bx_, by_, Zw), motion="swing", amp=1.2, speed=0.25)

    # coil of spare rope on a peg on the left post
    if not iron_posts:
        xpeg = -(xp + hw)
        exposed.append(cylinder("rope_peg", 0.014, 0.1, loc=(xpeg - 0.045, 0.02, Zw - 0.28), rot=(0, 90, 0), verts=8, bevel=0.003))
        rs_ = rng.fork("spare")
        for i in range(4):
            lp = [(xpeg - 0.03 - 0.014 * i + 0.01 * sin(a_), 0.02 + (0.085 + 0.006 * i) * sin(a_) * rs_.u(0.95, 1.05),
                   Zw - 0.28 + 0.012 - (0.17 + 0.012 * i) * (1 - cos(a_)) * 0.5 * 2.0 * 0.62) for a_ in [2 * pi * j / 18 for j in range(18)]]
            lo = sweep(f"spare_rope{i}", lp, rr0_, sides=6, closed=True)
            assign(lo, rope_m)
            paint(lo, grime=0.4)

    # ================================================================= lantern on a scrolled bracket
    if p.lantern:
        zL = Zw + 0.22
        arm = 0.74                                   # reach of the bracket: keeps the flame well clear of post and coping
        yb0 = -hw
        xl = -xp
        flatb = rounded_rect(0.008, 0.024, 0.002, 2)
        irons.append(box("bracket_plate", (0.045, 0.012, 0.4), loc=(xl, yb0 - 0.002, zL - 0.15), bevel=0.002))
        for bz in (0.03, -0.33):
            bolts_m.append(trs((xl, yb0 - 0.008, zL + bz), (90, 0, 0)))
        irons.append(sweep("bracket_arm", [(xl, yb0 - 0.004, zL), (xl, yb0 - arm, zL)], flatb, up=(1, 0, 0), smooth_path=False))
        irons.append(sweep("bracket_stay", bezier_path((xl, yb0 - 0.006, zL - 0.31), (xl, yb0 - 0.3, zL - 0.31), (xl, yb0 - 0.5, zL - 0.22),
                                                       (xl, yb0 - 0.5, zL - 0.008), n=14), flatb, up=(1, 0, 0)))
        sc = spiral_path(0.008, 0.036, 1.25, n=28, a0_deg=180, plane="YZ", center=(xl, yb0 - arm, zL + 0.036 + 0.004))
        irons.append(sweep("bracket_curl", sc, flatb, up=(1, 0, 0), scale_fn=lambda t: 0.5 + 0.5 * t))
        sc = spiral_path(0.006, 0.05, 1.25, n=24, a0_deg=270, plane="YZ", center=(xl, yb0 - 0.2, zL - 0.012 - 0.05))
        irons.append(sweep("bracket_scroll", sc, rounded_rect(0.006, 0.02, 0.002, 2), up=(1, 0, 0), scale_fn=lambda t: 0.5 + 0.5 * t))
        yl = yb0 - arm + 0.08
        irons.append(torus("bracket_ring", 0.013, 0.0035, loc=(xl, yl, zL - 0.004 - 0.012), rot=(90, 0, 0), major_seg=12, minor_seg=6))
        zt_ = zL - 0.062                             # apex of the lantern roof
        lw, lh = 0.062, 0.19                         # half width, body height
        lan = [torus("lan_loop", 0.017, 0.0035, loc=(xl, yl, zt_ + 0.018), rot=(0, 90, 0), major_seg=12, minor_seg=6)]
        lan.append(lathe("lan_roof", [(0.0, -0.088), (lw * 1.5, -0.088), (lw * 1.52, -0.083), (lw * 1.2, -0.07), (0.04, -0.04), (0.02, -0.022),
                                      (0.016, -0.006), (0.0, 0.0)], segments=4, loc=(xl, yl, zt_), rot=(0, 0, 45), smooth_angle=20))
        zb_ = zt_ - 0.088 - lh
        lan.append(box("lan_base", (lw * 2 + 0.012, lw * 2 + 0.012, 0.012), loc=(xl, yl, zb_ - 0.006), bevel=0.002))
        lan.append(lathe("lan_drop", [(0.0, zb_ - 0.05), (0.008, zb_ - 0.044), (0.012, zb_ - 0.03), (0.006, zb_ - 0.02), (0.02, zb_ - 0.012)],
                         segments=8, loc=(xl, yl, 0)))
        for ix in (-1, 1):
            for iy in (-1, 1):
                lan.append(box("lan_bar", (0.008, 0.008, lh), loc=(xl + ix * (lw - 0.004), yl + iy * (lw - 0.004), zb_ + lh / 2)))
        # candle socket and stub; the flame is a small teardrop, the panes stay dark and clear around it
        lan.append(cylinder("lan_socket", 0.02, 0.014, loc=(xl, yl, zb_ + 0.007), verts=10, bevel=0.003))
        lan.append(cylinder("lan_candle", 0.011, 0.05, loc=(xl, yl, zb_ + 0.039), verts=8))
        panes = []
        glass = mat("glass_lantern", "#4a3520", glow=0.06, opacity=0.2)
        for q in range(4):
            a = radians(90 * q)
            g = box("lan_glass", (2 * lw - 0.012, 0.002, lh - 0.004), loc=(xl + (lw - 0.004) * sin(a), yl - (lw - 0.004) * cos(a), zb_ + lh / 2),
                    rot=(0, 0, 90 * q))
            assign(g, glass)
            panes.append(g)
            for zq in (0.62, 0.3):                   # glazing bars
                lan.append(box("lan_glazing", (2 * lw - 0.012, 0.004, 0.004), loc=(xl + (lw - 0.003) * sin(a), yl - (lw - 0.003) * cos(a), zb_ + lh * zq),
                               rot=(0, 0, 90 * q)))
        fl_z = zb_ + 0.066
        flame = lathe("lan_flame", [(0.0, 0.0), (0.0055, 0.004), (0.0075, 0.012), (0.0045, 0.024), (0.0, 0.037)], segments=8, loc=(xl, yl, fl_z))
        assign(flame, mat("emissive", "#ffb45e", glow=6.0))
        for o in lan:
            assign(o, iron)
            paint(o, special=lambda co, n: _c01(0.05 + 0.3 * nz_c(co, n)), grime=lambda co, n: _c01((co[:, 2] - (zt_ - 0.09)) / 0.08))
        light("lantern", (xl, yl, fl_z + 0.018), color="#ffbf6a", intensity=p.intensity, range=7.0, flicker=0.3, shadow_near=0.16)
        part(lan + panes + [flame], "lantern", pivot=(xl, yl, zL - 0.016), motion="swing", amp=1.5, speed=0.4)

    # ================================================================= fixings, materials, weathering
    if pegs:
        pg = cylinder("peg", 0.011, 0.02, verts=8, bevel=0.002)
        exposed.append(instances_to_mesh("pegs", pg, pegs, keep_base=False))
    if bolts_m:
        bt_ = bolt("bolt", 0.011, 0.008, kind="square")
        irons.append(instances_to_mesh("bolts", bt_, bolts_m, keep_base=False))
    if clumps or weeds:
        if moss_m is None:                        # moss cushions as mossy stone
            moss_m = stone
        for o in weeds:
            assign(o, moss_m)
            paint(o, special=1.0, grime=0.0, mode='set')
    if len(clumps) > 480:                         # triangle budget: thin the cushions evenly on very large, very mossy wells
        clumps = [clumps[int(i * len(clumps) / 480)] for i in range(480)]
    if clumps:
        cb = _clump_base(rm_)
        assign(cb, moss_m)
        realize(cb)
        paint(cb, special=1.0, grime=0.0, mode='set')
        instances_to_mesh("moss", cb, clumps, keep_base=False)

    # exposed timber: silvered on the weather faces and end grain, a dark damp band at the feet,
    # rust / black iron-tannin streaks below the ironwork, green on the damp side of the feet
    def stain_w(c):
        best = -1.0
        for P0, N, S, L, w0, ph, wob in stains:
            if -0.005 < (c - P0).dot(N) < 0.03:
                best = max(best, _run_w(((None, P0.z, L, w0, ph, wob),), lambda k_: (c - P0).dot(S), c.z))
        return best

    def ex_grime(co, n):
        foot = _c01(1.0 - (co[:, 2] - z0) / (0.2 + 0.1 * nz_b(co, n)))
        return _c01(np.maximum(foot ** 0.7 * 0.95, 0.06 + 0.16 * nz_b(co, n)))

    def ex_wear(co, n):
        return _c01(0.1 + 0.6 * _c01(n[:, 2]) + 0.3 * nz_c(co, n))

    def ex_moss(co, n):
        foot = _c01(1.0 - (co[:, 2] - z0) / 0.3)
        return _c01(p.moss * (1.5 * foot * w_n(n) * (0.4 + 0.6 * nz_b(co, n)) + 0.6 * p.moss * _c01(n[:, 2]) * nz_a(co, n)))

    if stains:
        for o in posts:
            _cut_zone(o, stain_w, (0.05,))
    for o in exposed:
        assign(o, timber)
        paint(o, grime=ex_grime, special=ex_moss, wear=ex_wear)
    if stains:
        for o in posts:
            paint(o, grime=lambda co, n: 0.85 * _c01((np.array([stain_w(Vector(c)) for c in co.tolist()]) + 0.004) / 0.012))
    for o in exposed:                             # bleached end grain of ties / lintel
        if o.name.startswith("tie"):
            paint(o, wear=lambda co, n: 0.85 * (np.abs(n[:, 1]) > 0.9))
        elif o.name.startswith("lintel"):
            paint(o, wear=lambda co, n: 0.85 * (np.abs(n[:, 0]) > 0.9))
    for o in sheltered:                           # dry and dark under the roof
        assign(o, timber)
        paint(o, grime=lambda co, n: _c01(0.4 + 0.25 * nz_b(co, n)), wear=lambda co, n: 0.15 * nz_c(co, n))

    def iron_rust(co, n):                         # forge-black; rust only blooms in patches and near the ground
        low = _c01(1.0 - (co[:, 2] - z0) / 0.4)
        return _c01(0.05 + 0.4 * nz_c(co, n) * nz_b(co, n) + 0.25 * low * nz_c(co, n))

    for o in irons:
        assign(o, iron)
        paint(o, special=iron_rust, wear=lambda co, n: 0.3 * nz_c(co, n))

    socket("coping_front", "attach", (0.0, -(Ro - wall * 0.4), Hc), role="prop_top")
    socket("step_right", "attach", (Ro + 0.25, -(Ro * 0.4), step_h), role="prop_top")
    print(f"DBG well D={D:.2f} Ri={Ri:.2f} xp={xp:.2f} top={top_z:.2f} roof={p.roof} posts={p.posts} | tris stone="
          f"{_tris([i_['o'] for i_ in curb + lining + cope + steps] + pads + mortar)} weeds={_tris(weeds)} moss={len(clumps) * 15} "
          f"roof={_tris(roof_objs)} timber={_tris(exposed + sheltered)} iron={_tris(irons)}")
