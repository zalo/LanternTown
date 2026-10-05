"""Hanging flower basket. A forged S-hook carries a gathering ring from which three chains (or hemp
ropes) drop to the rim of either a Victorian wire basket (heavy rim ring, hoop, ribs, scalloped
trim, turned drip finial, pale sphagnum / coir liner bulging between the wires with ragged strands
escaping over the rim) or a woven willow basket. The planting grows out of a small dark root ball:
a fill layer of old leaves, a main layer of stalked leaves radiating outwards, young pale growth,
a skirt spilling over the rim, clustered heart-petalled flowers in two colours and trailing ivy
that arcs over the rim, falls in S-curves, forks and curls up at the tips. The foliage parts
around the hangers. A hooded votive jar with a tea light can hang from the ring between the chains.
Everything below the hook is the 'swing' part.
"""
from tk import *
import math
import numpy as np
from mathutils import Vector, Matrix

# tri_budget: a planted basket is mostly vegetation - every leaf, petal and ivy leaf is real geometry
# lod "far": lod 0 and lod 1 as built; from 44 m on (lod() == 2) build() models the basket anew from the _far_* stand-ins
META = {"name": "Hanging flower basket", "class": "prop_small", "mount": "hang", "tri_budget": 10000,
        "mount_height": 2.55, "focus": [0.0, 0.0, -0.45], "lod": "far"}
PARAMS = {
    "diameter": {"default": 0.36, "min": 0.24, "max": 0.5, "desc": "basket rim diameter (m)"},
    "chain_length": {"default": 0.48, "min": 0.25, "max": 0.9, "desc": "length of each of the three chains / ropes (m)"},
    "style": {"default": "wire", "choices": ["wire", "wicker"], "desc": "moss-lined wire basket or woven willow basket"},
    "hanger": {"default": "chain", "choices": ["chain", "rope"], "desc": "three iron chains or three hemp ropes"},
    "flower_color": {"default": "#cf2748", "desc": "main flower colour"},
    "flower_color2": {"default": "#f4ecee", "desc": "second flower colour"},
    "eye_color": {"default": "#f0a818", "desc": "colour of the flower centres"},
    "flower_density": {"default": 0.75, "min": 0.0, "max": 1.0, "desc": "how many flowers (0 = foliage only)"},
    "fullness": {"default": 1.0, "min": 0.5, "max": 1.4, "desc": "height / lushness of the planted mound"},
    "trailing_length": {"default": 0.45, "min": 0.0, "max": 1.2, "desc": "how far the longest ivy strand hangs below the rim (m)"},
    "trailers": {"default": 9, "min": 0, "max": 14, "desc": "number of trailing ivy strands"},
    "leaf_color": {"default": "#5a8a3c", "desc": "foliage colour of the mound"},
    "ivy_color": {"default": "#4a7d46", "desc": "colour of the trailing ivy"},
    "paint": {"default": "#1c2a22", "desc": "paint colour of the wire basket"},
    "rust": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "rust on the ironwork"},
    "candle": {"default": True, "desc": "votive candle jar hanging from the ring between the chains (needs room above the flowers)"},
    "candle_cd": {"default": 0.25, "min": 0.1, "max": 4.0, "desc": "candle intensity (cd)"},
}
VARIANTS = {
    "wicker_rope": {"style": "wicker", "hanger": "rope", "diameter": 0.32, "chain_length": 0.5, "flower_color": "#ee8a12",
                    "flower_color2": "#f6dc62", "eye_color": "#6a2410", "leaf_color": "#628c38", "trailing_length": 0.22,
                    "trailers": 6, "seed": 4},
    "grand_ivy": {"diameter": 0.46, "chain_length": 0.64, "flower_color": "#6a3fc4", "flower_color2": "#ec6aa8",
                  "leaf_color": "#4a7c50", "trailing_length": 0.95, "trailers": 11, "fullness": 1.2, "paint": "#1a2338", "seed": 11},
    "small_rusty": {"diameter": 0.26, "chain_length": 0.32, "flower_color": "#f4f0e6", "flower_color2": "#3f62d4",
                    "flower_density": 0.5, "fullness": 0.7, "trailing_length": 0.14, "trailers": 5, "rust": 0.95,
                    "paint": "#2a2420", "candle": False, "seed": 23},
}


def _tris(objs):
    return sum(len(f.vertices) - 2 for o in objs for f in o.data.polygons)


def _basis(loc, ydir, zhint, s=1.0):
    """4x4 with local +Y along ydir, local +Z as close to zhint as possible, uniform / xyz scale s."""
    y = Vector(ydir).normalized()
    z = Vector(zhint)
    z = z - y * z.dot(y)
    if z.length < 1e-6:
        z = y.orthogonal()
    z.normalize()
    x = y.cross(z)
    sx, sy, sz = (s, s, s) if isinstance(s, (int, float)) else s
    return Matrix(((x.x * sx, y.x * sy, z.x * sz, loc[0]), (x.y * sx, y.y * sy, z.y * sz, loc[1]),
                   (x.z * sx, y.z * sy, z.z * sz, loc[2]), (0, 0, 0, 1)))


def _mesh(name, verts, faces, uvs, two=False, grime=None, wear=None, under=0.12, sm=70):
    """Mesh with hand-set UVs in tile units of the 0.25 m 'leaf' tile (midrib along U at V = 0.5) and
    per-vertex grime / wear patterns. two = add reversed copies of the faces on their own vertices
    (the engine culls back faces); the undersides get `under` extra wear = paler under-leaf."""
    nv = len(verts)
    verts = [tuple(v) for v in verts]
    faces = [tuple(f) for f in faces]
    uvs = list(uvs)
    g = list(grime) if grime is not None else [0.0] * nv
    w = list(wear) if wear is not None else [0.0] * nv
    if two:
        faces = faces + [tuple(nv + i for i in reversed(f)) for f in faces]
        verts = verts + verts
        uvs = uvs + uvs
        g = g + g
        w = w + [min(1.0, x + under) for x in w]
    o = mesh_object(name, verts, faces)
    me = o.data
    uvl = me.uv_layers[0] if me.uv_layers else me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            u, v = uvs[me.loops[li].vertex_index]
            uvl.data[li].uv = (u * 0.25, v * 0.25)
    uv_keep(o)
    smooth(o, sm)
    ga, wa = np.array(g, dtype=np.float32), np.array(w, dtype=np.float32)
    paint(o, grime=lambda co, n: ga if len(co) == len(ga) else np.zeros(len(co)),
          wear=lambda co, n: wa if len(co) == len(wa) else np.zeros(len(co)), mode="set")
    return o


def _stamp(name, base, mats, grime=None, wear=None):
    """instances_to_mesh + per-instance grime / wear offsets (vertex order is instance-major)."""
    k = len(mats)
    o = instances_to_mesh(name, base, mats, uv_jitter=False, keep_base=False)

    def rep(vals):
        arr = np.asarray(vals, dtype=np.float32)

        def fn(co, n):
            m = len(co) // k
            return np.repeat(arr, m) if m * k == len(co) else np.zeros(len(co))
        return fn

    if grime is not None:
        paint(o, grime=rep(grime), mode="add")
    if wear is not None:
        paint(o, wear=rep(wear), mode="add")
    return o


def _blade(name, sides, droop, cup, two, pet=0.0, edge_g=0.0, bend=0.0, under=0.12, lw=0.9):
    """Leaf blade along +Y (unit length), upper face +Z: curved outline from 2 + 2 * len(sides) rim
    verts, folded along the midrib (cup), tip drooping, slightly sickle-bent, optional leaf stalk."""
    n = len(sides)
    my = [0.0] + [(sides[i][1] + sides[i + 1][1]) / 2 for i in range(n - 1)] + [1.0]

    def P(x, y):
        return (x + bend * y * y, y, -droop * y ** 1.8 + cup * abs(x) * (1.0 - 0.5 * y))

    v = [P(0, y) for y in my] + [P(x, y) for x, y in sides] + [P(-x * lw, y) for x, y in sides]
    uv = [(0.04 + 0.92 * y, 0.5) for y in my] + [(0.04 + 0.92 * y, 0.5 + x) for x, y in sides] + [(0.04 + 0.92 * y, 0.5 - x) for x, y in sides]
    g = [0.0] * n + [edge_g] + [edge_g * (0.45 + 0.55 * i / max(1, n - 1)) for i in range(n)] * 2
    f = []
    r0, l0 = n + 1, 2 * n + 1
    for i in range(n):
        f.append((i, r0 + i, i + 1))
        f.append((i + 1, l0 + i, i))
    for i in range(n - 1):
        f.append((i + 1, r0 + i, r0 + i + 1))
        f.append((i + 1, l0 + i + 1, l0 + i))
    if pet > 0:
        b = len(v)
        zb_ = P(0, 0.06)[2] - 0.012
        v += [(0.0, -pet, 0.03 * pet), (0.028, 0.06, zb_), (-0.028, 0.06, zb_)]
        uv += [(0.04, 0.5), (0.1, 0.5), (0.1, 0.5)]
        g += [0.3, 0.1, 0.1]
        f.append((b, b + 1, b + 2))
    return _mesh(name, v, f, uv, two, grime=g, under=under)


def _ivy(name, half, cy, two, under=0.12):
    """Lobed ivy leaf: fan around the petiole junction, lobes drooping, tip along +Y, with a leaf stalk."""
    v = [(0.0, cy, 0.035), (0.0, 0.05, 0.0)] + list(half) + [(0.0, 1.0, -0.12)] + [(-x * 0.92, y, z) for x, y, z in reversed(half)]
    npm = len(v) - 1
    f = [(0, 1 + i, 1 + (i + 1) % npm) for i in range(npm)]
    g = [0.0] * len(v)
    b = len(v)
    v += [(0.0, -0.35, 0.02), (0.03, 0.1, -0.012), (-0.03, 0.1, -0.012)]
    g += [0.35, 0.1, 0.1]
    f.append((b, b + 1, b + 2))
    uv = [(0.04 + 0.92 * clamp(y), 0.5 + 0.95 * x) for x, y, _ in v]
    return _mesh(name, v, f, uv, two, grime=g, under=under)


def _flower_bases():
    """Five overlapping heart-shaped petals (rounded shoulders, notched tip, cupped from a recessed
    throat, tips recurved, each rolled a little so it laps over its neighbour) + a raised eye."""
    v, f, uv, g = [], [], [], []
    roll = math.radians(9)
    pts = [(0.10, 0.0, -0.11, 0.6), (0.56, -0.45, 0.035, 0.1), (0.97, -0.28, -0.05, 0.0), (0.84, 0.0, -0.005, 0.03),
           (0.97, 0.28, -0.05, 0.0), (0.56, 0.45, 0.035, 0.1)]
    for k in range(5):
        a = math.radians(72 * k)
        ca, sa = math.cos(a), math.sin(a)
        b = len(v)
        for r, t, z, gg in pts:
            t2, z2 = t * math.cos(roll), z + t * math.sin(roll)
            v.append((r * ca - t2 * sa, r * sa + t2 * ca, z2))
            uv.append((0.06 + 0.86 * r, 0.5 + 0.5 * t))
            g.append(gg)
        f += [(b, b + 1, b + 2), (b, b + 2, b + 3), (b, b + 3, b + 4), (b, b + 4, b + 5)]
    petals = _mesh("petal_base", v, f, uv, grime=g, sm=50)
    ev = [(0.21 * math.cos(math.radians(90 * k + 20)), 0.21 * math.sin(math.radians(90 * k + 20)), -0.05) for k in range(4)] + [(0, 0, 0.05)]
    eye = _mesh("eye_base", ev, [(k, (k + 1) % 4, 4) for k in range(4)], [(0.5 + 0.2 * x, 0.5 + 0.2 * y) for x, y, _ in ev],
                grime=[0.45] * 4 + [0.0])
    # green calyx tapering into the flower stalk that runs back into the foliage
    sv = [(0.3 * math.cos(math.radians(120 * k + 90)), 0.3 * math.sin(math.radians(120 * k + 90)), -0.075) for k in range(3)] + [(0, 0, -2.4)]
    stalk = _mesh("stalk_base", sv, [((k + 1) % 3, k, 3) for k in range(3)], [(0.5, 0.5)] * 3 + [(0.06, 0.5)], grime=[0.1, 0.1, 0.1, 0.4])
    return petals, eye, stalk


# ---- lod 2 (from 44 m on, about 20 px/m): plain stand-ins in world coordinates. The vertex masks of a level are
# transferred from lod 0, so none is painted here.
def _far_obj(name, verts, faces, m, fix=False):
    o = mesh_object(name, verts, faces)
    if fix:
        recalc_normals(o)
    smooth(o, 60)
    uv_box(o, scale=1.0)
    assign(o, m)
    return o


def _far_shell(name, rings, m):
    """Rings of points bottom to top, each anticlockwise seen from above (a ring of one point is a pole), skinned
    with the faces looking away from the axis."""
    v, rows = [], []
    n = max(len(r) for r in rings)
    for r in rings:
        rows.append([len(v) + (i if len(r) > 1 else 0) for i in range(n)])
        v += [tuple(q) for q in r]
    f = []
    for r0, r1 in zip(rows, rows[1:]):
        for i in range(n):
            q = (r0[i], r0[(i + 1) % n], r1[(i + 1) % n], r1[i])
            f.append(tuple(q[k] for k in range(4) if q[k] != q[k - 1]))
    return _far_obj(name, v, f, m)


def _far_turn(name, prof, n, m, k=1.0):
    """A turned body as an n-sided shell; prof = [(radius, z)] bottom to top along the outside, radius 0 = pole.
    k widens it (a polygon drawn into the circle looks narrower than the circle)."""
    return _far_shell(name, [[(0.0, 0.0, z)] if r <= 1e-9 else
                             [(k * r * math.cos(2 * math.pi * i / n), k * r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
                             for r, z in prof], m)


def _far_rod(name, a, b, r, m, cap=True):
    """A chain, a rope or a rod as one three-sided prism from a to b."""
    a, b = Vector(a), Vector(b)
    d = (b - a).normalized()
    x = d.orthogonal().normalized()
    y = d.cross(x)
    v = [tuple(q + (x * math.cos(t) + y * math.sin(t)) * r) for q in (a, b) for t in (0.0, 2.0944, 4.1888)]
    f = [(i, (i + 1) % 3, 3 + (i + 1) % 3, 3 + i) for i in range(3)]
    return _far_obj(name, v, f + ([(2, 1, 0), (3, 4, 5)] if cap else []), m)


def _kite(sides, droop, cup, bend=0.0, lw=0.9, pet=0.0, **_):
    """The four corners of a _blade: foot of the stalk, widest point right, tip, widest point left."""
    x, y = max(sides)
    z = -droop * y ** 1.8
    return [(0.0, -pet, 0.03 * pet), (x + bend * y * y, y, z + cup * x * (1.0 - 0.5 * y)), (bend, 1.0, -droop),
            (-x * lw + bend * y * y, y, z + cup * x * lw * (1.0 - 0.5 * y))]


FAR_STEP = 0.13
# mound: share of the leaf tips left outside; card area / flower size squared (a flower stands clear of the leaves and
# shows whole: 2.2 would be its petals alone); streamer half width / leaf size
FAR_Q, FAR_FL, FAR_IVY = 0.05, 3.6, 0.9


def _far_outline(have, leaves, tol=0.008):
    """The footprint and the drop of the basket must stay those of lod 0. have: the points built so far; leaves:
    [(four world corners, material)] of every leaf (and strand tip) that was left out. Where the stand-ins end more than tol short
    of the leaf that reaches furthest to a side or down, that one leaf comes back as a two-sided kite. -> objects"""
    if not leaves:
        return []
    lo = [min(q[i] for q in have) for i in range(3)]
    hi = [max(q[i] for q in have) for i in range(3)]
    back = {}
    for ax, sg in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1)):
        K, m = max(leaves, key=lambda l: max(sg * q[ax] for q in l[0]))
        if max(sg * q[ax] for q in K) > (hi[ax] if sg > 0 else -lo[ax]) + tol:
            back.setdefault(m.name, (m, []))[1].append(K)
            lo = [min([lo[i]] + [q[i] for q in K]) for i in range(3)]
            hi = [max([hi[i]] + [q[i] for q in K]) for i in range(3)]
    out = []
    for m, ks in back.values():
        o = _mesh("leaves_outline", [q for K in ks for q in K], [(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3) for i in range(len(ks))],
                  [(0.04, 0.5), (0.5, 0.8), (0.96, 0.5), (0.5, 0.2)] * len(ks), two=True)
        assign(o, m)
        out.append(o)
    return out


def build(p, rng):
    far = lod() == 2                            # from 44 m on: every part as its plain stand-in, the same random draws
    fv = [(0.0, 0.0, 0.0)]                      # ... and the points those reach (see _far_outline)
    dia = p.diameter
    R = dia / 2
    ks = dia / 0.36
    wicker = p.style == "wicker"
    rope_h = p.hanger == "rope"
    iron = mat("iron_wrought")
    frame_mat = mat("iron_painted", p.paint)
    rust_fn = mask_mul(noise_mask(0.035, seed=p.seed + 3, lo=0.62 - 0.5 * p.rust, hi=0.95 - 0.45 * p.rust), min(1.0, 0.25 + p.rust))
    up = Vector((0, 0, 1))
    swing = []

    # ---------------------------------------------------------------- hook + gathering ring
    rh, r1, r2 = 0.003, 0.016, 0.014
    c1z = -rh - r1
    c2z = -rh - 2 * r1 - r2
    pts = [(r1 * math.cos(math.radians(205 - 295 * i / 9)), 0, c1z + r1 * math.sin(math.radians(205 - 295 * i / 9))) for i in range(10)]
    pts += [(r2 * math.cos(math.radians(90 + 250 * i / 8)), 0, c2z + r2 * math.sin(math.radians(90 + 250 * i / 8))) for i in range(1, 9)]
    if not far:
        hook = sweep("hook", pts, rh, sides=5, up=(0, 1, 0), smooth_path=False, scale_fn=lambda t: min(1.0, 0.4 + 7 * t, 0.4 + 7 * (1 - t)))
        smooth(hook, 60)
        assign(hook, iron)
        paint(hook, special=rust_fn, wear=lambda co, n: ((co[:, 2] > -0.012) * 0.8))
    z_hb = c2z - r2
    rg = 0.013
    ring_cz = z_hb + rh + 0.0025 - rg
    if far:                                     # S-hook and ring as one bar as wide as they are
        hook = _far_turn("hook", [(0.016, ring_cz - rg), (0.016, 0.0)], 4, iron)
    else:
        ring = torus("ring", rg, 0.0025, loc=(0, 0, ring_cz), rot=(0, 90, 0), major_seg=10, minor_seg=4)
        assign(ring, iron)
        paint(ring, special=rust_fn)
        swing.append(ring)
    Tz = ring_cz - rg

    # ---------------------------------------------------------------- hangers
    rim_minor = 0.011 if wicker else 0.005
    L = max(p.chain_length, R * 1.25)
    drop = math.sqrt(L * L - R * R)
    zr = Tz - drop - rim_minor - 0.013          # rim centreline height
    az3 = [math.radians(90 + 120 * k + rng.jitter(0, 3)) for k in range(3)]
    hang = []
    lines = []                                  # hanger centre lines (top, rim end): the planting keeps clear of them
    if not rope_h:
        cs = clamp(p.chain_length / 0.5, 1.0, 1.3)          # longer drops carry a heavier chain
        ll, lw, wire = 0.068 * cs, 0.024 * cs, 0.0038 * cs
        a_, rc = ll / 2 - lw / 2, lw / 2 - wire
        lp = [(rc, 0, -a_), (rc, 0, a_)] + [(rc * math.cos(math.radians(t)), 0, a_ + rc * math.sin(math.radians(t))) for t in (60, 120)]
        lp += [(-rc, 0, a_), (-rc, 0, -a_)] + [(rc * math.cos(math.radians(t)), 0, -a_ + rc * math.sin(math.radians(t))) for t in (240, 300)]
        if not far:
            link = sweep("link", lp, wire, closed=True, sides=3, smooth_path=False, up=(0, 1, 0))
            smooth(link, 80)
            assign(link, iron)
        lm = []
        eyes = []
        for k, a in enumerate(az3):
            er = Vector((math.cos(a), math.sin(a), 0))
            et = Vector((-er.y, er.x, 0))
            T = Vector((0, 0, Tz)) + er * 0.004
            E = er * R + up * (zr + rim_minor + 0.013)
            lines.append((T, E))
            v = E - T
            n = max(2, round(v.length / (ll - 2.3 * wire)))
            pitch = v.length / n
            vz = v.normalized()
            sz = (pitch + 2.3 * wire) / ll
            y0 = vz.cross(et).normalized()
            for i in range(n):
                c = T + vz * ((i + 0.5) * pitch)
                roll = math.radians(rng.jitter(0, 9))
                ya = (y0 * math.cos(roll) - et * math.sin(roll)) if i % 2 == 0 else (et * math.cos(roll) + y0 * math.sin(roll))
                lm.append(_basis(c, ya, vz, (1, 1, sz)))      # link lies in its local XZ plane, long axis Z = chain direction
            if far:                             # a chain is one rod as wide as its links
                hang.append(_far_rod(f"chain{k}", (0, 0, z_hb - 0.002), E, 0.011 * cs, iron))
                continue
            eyes.append(torus(f"eye{k}", 0.0075, 0.0024, loc=tuple(er * R + up * (zr + rim_minor + 0.0065)), rot=(90, 0, math.degrees(a)),
                              major_seg=7, minor_seg=3))
        if not far:
            chains = instances_to_mesh("chains", link, lm, keep_base=False)
            paint(chains, special=rust_fn)
            hang.append(chains)
            for e in eyes:
                assign(e, iron if wicker else frame_mat)
                paint(e, special=rust_fn)
            hang += eyes
    else:
        rp = mat("rope", "#b79a6a")
        for k, a in enumerate(az3):
            er = Vector((math.cos(a), math.sin(a), 0))
            T = Vector((0, 0, Tz - 0.03))
            E = er * (R + rim_minor * 0.4) + up * (zr + rim_minor * 0.9)
            lines.append((T, E))
            mid = (T + E) / 2 + er * rng.u(0.0, 0.008)
            if far:                             # the rope alone, as one rod (wrap, knot and tail are a pixel together)
                rng.u(0, 0.01), rng.u(0, 0.03)
                hang.append(_far_rod(f"rope{k}", (0, 0, z_hb - 0.002), E, 0.007, rp))
                continue
            rpath = [(0, 0, Tz + 0.004), tuple(T), tuple(mid), tuple(E), tuple(E + er * 0.012 - up * 0.014)]
            r_ = sweep(f"rope{k}", rpath, 0.0042, sides=6, step=0.06)
            wrap = torus(f"rope_wrap{k}", rim_minor + 0.0045, 0.0042, loc=tuple(er * R + up * zr), rot=(90, 0, math.degrees(a)),
                         major_seg=10, minor_seg=5)
            knot = sphere(f"rope_knot{k}", 0.0095, loc=tuple(er * (R + rim_minor + 0.007) + up * (zr - 0.012)), segments=7, rings=5,
                          scale=(1, 1, 1.25))
            tail = sweep(f"rope_tail{k}", [tuple(er * (R + rim_minor + 0.008) + up * (zr - 0.018)),
                                           tuple(er * (R + rim_minor + 0.012 + rng.u(0, 0.01)) + up * (zr - 0.06 - rng.u(0, 0.03)))],
                         0.004, sides=5, scale_fn=lambda t: 1.0 + 0.5 * t)
            for o in (r_, wrap, knot, tail):
                assign(o, rp)
                hang.append(o)
        if not far:
            whip = cylinder("rope_whip", 0.0095, 0.03, loc=(0, 0, Tz - 0.02), verts=8)
            uv_cyl(whip)
            assign(whip, rp)
            hang.append(whip)
    swing += hang
    line_top = lines[0][0].z
    line_end = lines[0][1].z

    def line_r(z):                              # radius of the hanger lines at height z
        return R * clamp((line_top - z) / (line_top - line_end))

    def clear(P, m):                            # is P at least m away from every hanger?
        for T, E in lines:
            d = E - T
            t = clamp((P - T).dot(d) / d.length_squared)
            if (P - (T + d * t)).length < m:
                return False
        return True

    def az_gap(a):                              # angular distance to the nearest hanger
        return min(abs((a - ac + math.pi) % (2 * math.pi) - math.pi) for ac in az3)

    # ---------------------------------------------------------------- basket
    n_ribs = max(6, int(round(dia * 28 / 2)) * 2)
    basket = []
    liner_mat = mat("burlap", "#cfc38a")         # pale dried sphagnum / coir
    if not wicker:
        D = dia * 0.46
        zb = zr - D

        def bowl(th, off=0.0):
            return ((R + off) * math.sin(th) ** 0.8, zr - (D + off) * math.cos(th))

        wr = 0.0028                              # 5.6 mm rod
        if far:
            # the liner as an eight-sided bowl; of the wires (the rim lies under the skirt of leaves) the drip finial
            basket.append(_far_turn("liner", [(0.0, zb + 0.004), bowl(0.55), bowl(1.05), (R + 0.004, zr + 0.012)], 8, liner_mat, k=1.03))
            basket.append(_far_turn("boss", [(0.0, zb - 0.036), (0.024, zb - 0.004), (0.0, zb + 0.004)], 3, frame_mat))
        else:
            basket.append(torus("rim", R, rim_minor, loc=(0, 0, zr), major_seg=24, minor_seg=4))
            th1 = math.radians(52)
            r_, z_ = bowl(th1, wr * 2)
            hoop = torus("hoop", r_, wr, loc=(0, 0, z_), major_seg=22, minor_seg=3)
            smooth(hoop, 80)
            basket.append(hoop)
            th0 = math.asin(min(1.0, (0.02 / R) ** 1.25))
            for k in range(n_ribs):
                a = 2 * math.pi * k / n_ribs
                pa = []
                for i in range(5):
                    th = math.pi / 2 - (math.pi / 2 - th0) * i / 4
                    r_, z_ = bowl(th, wr)
                    pa.append((r_ * math.cos(a), r_ * math.sin(a), z_))
                rib = sweep(f"rib{k}", pa, wr, sides=3, up=(-math.sin(a), math.cos(a), 0), cap=False, smooth_path=False)
                smooth(rib, 80)
                basket.append(rib)
            # scalloped wire trim: loops welded to the rim at every rib, hanging proud of the bowl so they show in silhouette
            wave = []
            dth = (math.pi / 2 - th1) * 0.78
            for j in range(n_ribs * 4):
                a = 2 * math.pi * j / (n_ribs * 4)
                fsc = (0.0, 0.72, 1.0, 0.72)[j % 4]
                r_, z_ = bowl(math.pi / 2 - dth * fsc, wr * 2 + 0.014 * fsc)
                wave.append((r_ * math.cos(a), r_ * math.sin(a), z_ - 0.004 * (1 - fsc)))
            sc = sweep("scallop", wave, wr * 0.85, closed=True, sides=3, smooth_path=False)
            smooth(sc, 80)
            basket.append(sc)
            basket.append(lathe("boss", [(0.0, zb - 0.036), (0.0045, zb - 0.032), (0.008, zb - 0.024), (0.0035, zb - 0.014),
                                         (0.023, zb - 0.006), (0.026, zb - 0.001), (0.0, zb + 0.003)], segments=7))
            for o in basket:
                assign(o, frame_mat)
                paint(o, special=mask_mul(rust_fn, lambda co, n: 0.55 + 0.45 * (co[:, 2] < zr - D * 0.5)))
            # liner packed against the wires, bulging in every bay and between rim and hoop, rolled over the rim
            prof = [(0.0, zb + 0.004)]
            for i in range(1, 6):
                prof.append(bowl(math.pi / 2 * i / 5, -0.003))
            prof += [(R + 0.007, zr + 0.013), (R - 0.022, zr + 0.026)]
            liner = lathe("liner", prof, segments=n_ribs * 2, smooth_angle=80)

            def bulge(co):
                rr = math.hypot(co.x, co.y)
                if rr < 1e-5:
                    return co
                a = math.atan2(co.y, co.x)
                dep = smoothstep(0.012, 0.06, rr) * smoothstep(zr + 0.02, zr - 0.01, co.z)
                sn = abs(math.sin(n_ribs * a / 2))
                kk = 1.0 + 0.062 * dep * (sn - 0.12)
                return Vector((co.x * kk, co.y * kk, co.z - 0.006 * dep * sn))

            transform_verts(liner, bulge)
            displace_noise(liner, strength=0.005, scale=0.03, seed=p.seed)
            assign(liner, liner_mat)
            paint(liner, grime=lambda co, n: np.maximum(0.3 * noise_mask(0.045, seed=p.seed + 5)(co, n), 0.38 * height_ramp(zb - 0.01, zb + D * 0.3)(co, n)),
                  special=mask_mul(noise_mask(0.08, seed=p.seed + 6), 0.35))
            basket.append(liner)
            # ragged strands of the liner escaping over the rim and through the wires
            rf_ = rng.fork("fibres")
            fb = _mesh("fibre_base", [(-0.5, 0, 0), (0.5, 0, 0), (0.15, 1.0, 0.12)], [(0, 1, 2)], [(0.0, 0.0), (0.03, 0.0), (0.015, 0.3)], two=True, under=0.0)
            assign(fb, liner_mat)
            fm = []
            for i in range(int(40 * ks)):
                a = rf_.u(0, 2 * math.pi)
                er = Vector((math.cos(a), math.sin(a), 0))
                et = Vector((-er.y, er.x, 0))
                if i % 4:
                    P = er * (R + rf_.u(-0.004, 0.008)) + up * (zr + rf_.u(-0.004, 0.02))
                    d = er * rf_.u(0.25, 1.0) + up * rf_.u(-1.1, 0.5) + et * rf_.jitter(0, 0.7)
                else:
                    th = rf_.u(0.35, 1.35)
                    r_, z_ = bowl(th, 0.006)
                    P = er * r_ + up * z_
                    d = er * rf_.u(0.1, 0.6) - up * rf_.u(0.5, 1.0) + et * rf_.jitter(0, 0.5)
                ln = rf_.u(0.025, 0.06)
                fm.append(_basis(P, d, er + rf_.vec(0.5), (rf_.u(0.003, 0.006), ln, ln)))
            basket.append(_stamp("liner_fibres", fb, fm, grime=[rf_.u(0, 0.3) for _ in fm]))
        z_low = zb - 0.036
    else:
        D = dia * 0.5
        zb = zr - D
        Rb = R * 0.62
        wk = mat("wicker", "#b98d54")
        prof = [(0.0, zb), (Rb - 0.01, zb), (Rb, zb + 0.008)]
        for i in range(1, 7):
            t = i / 7
            prof.append((Rb + (R - 0.004 - Rb) * (t ** 0.75) + 0.008 * math.sin(t * math.pi), zb + D * t))
        prof += [(R - 0.004, zr), (R - 0.016, zr + 0.002), (R - 0.018, zr - 0.03)]
        if far:                                 # body, rim, band and foot as one eight-sided pot
            basket.append(_far_turn("wicker_body", [(0.0, zb), (Rb + 0.004, zb), (Rb + (R - 0.004 - Rb) * 0.6 + 0.012, zb + D * 0.5),
                                                    (R + rim_minor * 0.7, zr + rim_minor * 0.5)], 8, wk, k=1.03))
        else:
            body = lathe("wicker_body", prof, segments=26, smooth_angle=70)
            displace_noise(body, strength=0.003, scale=0.06, seed=p.seed)
            rimw = torus("wicker_rim", R, rim_minor, loc=(0, 0, zr), major_seg=26, minor_seg=6)
            foot = torus("wicker_foot", Rb, 0.008, loc=(0, 0, zb + 0.004), major_seg=20, minor_seg=5)
            zt = zb + D * 0.62
            rt = Rb + (R - 0.004 - Rb) * (0.62 ** 0.75) + 0.008 * math.sin(0.62 * math.pi)
            band = torus("wicker_band", rt + 0.002, 0.0065, loc=(0, 0, zt), major_seg=24, minor_seg=5)
            paint(body, grime=mask_mul(height_ramp(zb - 0.01, zb + D * 0.6), 0.6), wear=mask_mul(noise_mask(0.05, seed=p.seed + 8), 0.5))
            for o in (rimw, foot, band):                      # darker, handled border rods
                paint(o, grime=0.5, wear=mask_mul(noise_mask(0.04, seed=p.seed + 9), 0.6))
            for o in (body, rimw, foot, band):
                assign(o, wk)
                basket.append(o)
        z_low = zb
    swing += basket
    fv += [(R, R, zr), (-R, -R, z_low)]

    # ---------------------------------------------------------------- planted mound
    rp_ = rng.fork("plants")
    Rf = R + 0.018 + 0.018 * ks
    zc = zr + 0.012
    hm = dia * 0.46 * p.fullness
    hd = 0.05 + 0.03 * ks
    lumps = [(rp_.u(0.05, 0.11), rp_.pick([2, 3, 4]), rp_.u(0, 6.28), rp_.u(1.5, 3.0), rp_.u(0, 6.28)) for _ in range(3)]
    lop_a, lop = rp_.u(0, 6.28), rp_.u(0.05, 0.11)
    jar_h, jar_r, hood_r = 0.066, 0.024, 0.031
    d_top = max(0.07, (hood_r + 0.02) / R * (line_top - line_end) + (Tz - line_top))     # hood rim clears the hangers
    z_flame = Tz - d_top - 0.011 - jar_h + 0.03
    dent_c = 0.4
    has_candle = bool(p.candle) and z_flame - (zc + hm * 1.12 * (1 - dent_c) + 0.035) >= 0.105
    dent = dent_c if has_candle else 0.0                       # plants kept trimmed back under the candle jar

    def dome(a, phi, k=1.0, groove=True):
        l = 1.0 + lop * math.cos(a - lop_a)
        for amp, fa, pa, fp, pp in lumps:
            l += amp * math.sin(fa * a + pa) * math.sin(fp * phi + pp)
        s, c = math.sin(phi), math.cos(phi)
        rr = Rf * l * k * s
        if c >= 0:
            z = zc + hm * l * k * (1.0 - dent * max(0.0, 1.0 - phi / 0.8) ** 2) * c
            for ac in (az3 if groove else ()):                 # the mound parts in a groove around every hanger
                da = (a - ac + math.pi) % (2 * math.pi) - math.pi
                w = math.exp(-(da / 0.16) ** 2)
                if w > 0.02:
                    lim = max(0.02, line_r(z) - 0.034)
                    if rr > lim:
                        rr -= w * (rr - lim)
        else:
            z = zc + hd * c
        return Vector((rr * math.cos(a), rr * math.sin(a), z))

    def dome_n(a, phi):
        phi = max(phi, 0.1)
        p0 = dome(a, phi)
        n = (dome(a, phi + 0.03) - p0).cross(dome(a + 0.03, phi) - p0)
        if n.dot(p0 - Vector((0, 0, zc))) < 0:
            n = -n
        return n.normalized()

    leaf_mat = mat("leaf", p.leaf_color, rough=1.9)
    ivy_mat = mat("leaf", p.ivy_color, rough=1.9)
    # small dark root ball / top dressing deep inside the foliage (never the visible surface)
    if not far:
        nu, nv = 10, 4
        kc = 0.66
        cv = [tuple(dome(0, 0, kc))]
        cf = []
        for i in range(1, nv + 1):
            phi = (math.pi / 2 + 0.12) * i / nv
            for j in range(nu):
                cv.append(tuple(dome(2 * math.pi * j / nu, phi, kc)))
        for j in range(nu):
            cf.append((0, 1 + j, 1 + (j + 1) % nu))
        for i in range(nv - 1):
            for j in range(nu):
                a0, a1 = 1 + i * nu + j, 1 + i * nu + (j + 1) % nu
                cf.append((a0, a0 + nu, a1 + nu, a1))
        core = mesh_object("core", cv, cf)
        recalc_normals(core)
        smooth(core, 80)
        uv_box(core, scale=1.0)
        assign(core, mat("rope", "#b79a6a") if (wicker and rope_h) else liner_mat)
        paint(core, grime=1.0)
        swing.append(core)

    # leaf shapes: broad ovate (stalked), brown-edged old broad, lanceolate, small round
    SH = {
        "broad": dict(sides=[(0.25, 0.12), (0.35, 0.43), (0.22, 0.79)], droop=0.2, cup=0.22, pet=0.4, bend=0.05),
        "old": dict(sides=[(0.24, 0.13), (0.33, 0.45), (0.2, 0.8)], droop=0.3, cup=0.12, pet=0.4, bend=-0.07, edge_g=0.75),
        "lance": dict(sides=[(0.15, 0.14), (0.22, 0.44), (0.14, 0.8)], droop=0.3, cup=0.28, pet=0.3, bend=0.1),
        "round": dict(sides=[(0.27, 0.2), (0.25, 0.7)], droop=0.12, cup=0.2, bend=0.03),
    }
    groups = {}

    def add_leaf(shape, two, mi, M, g, w):
        gr = groups.setdefault((shape, two, mi), ([], [], []))
        gr[0].append(M)
        gr[1].append(g)
        gr[2].append(w)

    dens = p.fullness ** 0.7 * ks ** 1.5
    phi_top = math.pi / 2 + 0.05

    def shade(phi):                             # the jar foot shades the crown right under the candle
        return 0.3 * (1 - clamp((phi - 0.2) / 0.45, 0, 1)) if has_candle else 0.0

    def spot(i, n, k, jit=0.3):
        a = i * 2.39996 + rp_.jitter(0, jit)
        phi = math.acos(1 - (i + rp_.u(0.15, 0.85)) / n * (1 - math.cos(phi_top)))
        P = dome(a, phi, k)
        nrm = dome_n(a, phi)
        down = (dome(a, phi + 0.05, k) - P).normalized()
        return a, phi, P, nrm, down, nrm.cross(down).normalized()

    # 1) fill layer: big old leaves lying close over the root ball
    nA = int(min(44, 39 * dens))
    for i in range(nA):
        a, phi, P, nrm, down, t2 = spot(i, nA, rp_.u(0.74, 0.82))
        ang = rp_.jitter(0, 1.5)
        td = down * math.cos(ang) + t2 * math.sin(ang)
        lift = rp_.u(0.05, 0.3)
        s = rp_.u(0.075, 0.1) * (0.75 + 0.25 * ks)
        d = td * math.cos(lift) + nrm * math.sin(lift)
        M = _basis(P - d * s * 0.5, d, nrm + rp_.vec(0.2), (s * rp_.u(1.0, 1.3) * rp_.pick([-1, 1]), s, s))
        add_leaf("round", False, int(rp_.chance(0.3)), M, rp_.u(0.1, 0.3) + shade(phi), 0.0)
    # 2) main layer: stalked leaves radiating out of the mound
    nB = int(min(90, 80 * dens))
    for i in range(nB):
        a, phi, P, nrm, down, t2 = spot(i, nB, rp_.u(0.86, 1.0))
        s = rp_.u(0.055, 0.082) * (0.75 + 0.25 * ks)
        for _try in range(4):
            ang = rp_.jitter(0, 1.25)
            td = down * math.cos(ang) + t2 * math.sin(ang)
            lift = rp_.u(0.3, 1.0)
            d = td * math.cos(lift) + nrm * math.sin(lift)
            if clear(P + d * s * 0.55, 0.024) and clear(P + d * s, 0.018):
                break
        else:
            continue
        zh = nrm * math.cos(lift) - td * math.sin(lift) + rp_.vec(0.3) + up * 0.45
        shape = rp_.weighted([("broad", 0.62), ("lance", 0.24), ("old", 0.14)])
        M = _basis(P, d, zh, (s * rp_.u(0.85, 1.2) * rp_.pick([-1, 1]), s, s))
        yel = rp_.u(0.2, 0.32) if rp_.chance(0.1) else rp_.u(0.0, 0.1)
        add_leaf(shape, False, int(rp_.chance(0.2)), M, rp_.pick([0.0, 0.0, 0.05, 0.1, 0.18]) + shade(phi), yel)
    # 3) young growth: small pale leaves standing up between the big ones
    nD = int(min(46, 42 * dens))
    for i in range(nD):
        a, phi, P, nrm, down, t2 = spot(i, nD, rp_.u(0.98, 1.1), jit=1.0)
        ang = rp_.u(0, 6.28)
        td = down * math.cos(ang) + t2 * math.sin(ang)
        lift = rp_.u(0.5, 1.2)
        d = td * math.cos(lift) + nrm * math.sin(lift)
        s = rp_.u(0.03, 0.046) * (0.75 + 0.25 * ks)
        if not clear(P + d * s * 0.5, 0.02):
            continue
        zh = nrm * math.cos(lift) - td * math.sin(lift) + rp_.vec(0.35) + up * 0.4
        add_leaf("round", False, 0, _basis(P - d * s * 0.3, d, zh, (s * rp_.u(0.85, 1.1), s, s)), shade(phi), rp_.u(0.1, 0.26))
    # 4) skirt: leaves on arching stalks spilling over the rim (seen from below: two-sided)
    nC = int(min(44, 36 * ks ** 1.2 * p.fullness ** 0.5))
    for i in range(nC):
        a = 2 * math.pi * (i + rp_.u(-0.35, 0.35)) / nC
        if az_gap(a) < 0.13:
            continue
        er = Vector((math.cos(a), math.sin(a), 0))
        et = Vector((-er.y, er.x, 0))
        th = rp_.u(0.2, 1.3)
        yaw = rp_.jitter(0, 0.65)
        d = (er * math.cos(yaw) + et * math.sin(yaw)) * math.cos(th) - up * math.sin(th)
        P = er * (R * rp_.u(0.9, 1.0) + rp_.u(0.0, 0.02)) + up * (zr + rp_.u(0.014, 0.045))
        s = rp_.u(0.065, 0.095) * (0.75 + 0.25 * ks)
        zh = er * math.sin(th) + up * math.cos(th) + rp_.vec(0.25)
        shape = rp_.weighted([("lance", 0.45), ("broad", 0.4), ("old", 0.15)])
        add_leaf(shape, th < 0.85, int(rp_.chance(0.25)), _basis(P, d, zh, (s * rp_.u(0.85, 1.15) * rp_.pick([-1, 1]), s, s)),
                 rp_.pick([0.0, 0.05, 0.1, 0.2]), rp_.u(0.0, 0.1))
    leaf_objs = []
    gone = []                                   # lod 2: (four corners, material) of every leaf left out
    if far:
        for (shape, two, mi), (ms, gs, ws) in groups.items():
            K = [Vector(q) for q in _kite(**SH[shape])]
            gone += [([M @ q for q in K], ivy_mat if mi else leaf_mat) for M in ms]
        c0 = Vector((0, 0, zc))

        def far_mound():
            """The foliage as one closed mound through the leaf tips: eight sectors round, rings from under the skirt
            up to the crown. -> the mound, its points, skin(direction)"""
            PH = (0.0, 0.65, 1.2, 1.62, 1.95)
            cell = {}
            for K, _m in gone:
                for q in K[1:]:
                    d = q - c0
                    i = min(range(5), key=lambda i_: abs(PH[i_] - math.acos(clamp(d.z / max(d.length, 1e-6), -1.0, 1.0))))
                    cell.setdefault((i, int((math.atan2(d.y, d.x) % (2 * math.pi)) / (math.pi / 4)) % 8 if i else 0), []).append(d.length)
            az = [math.pi / 4 * (j + 0.5) for j in range(8)]

            def reach(i, j):
                a = az[j]
                u = Vector((math.sin(PH[i]) * math.cos(a), math.sin(PH[i]) * math.sin(a), math.cos(PH[i])))
                c = sorted(cell.get((i, j), ()), reverse=True)
                if len(c) >= 3:
                    return c0 + u * c[int(len(c) * FAR_Q)]
                if i == 4:                      # no skirt here: the mound ends on the rim
                    return Vector(((R + 0.012) * math.cos(a), (R + 0.012) * math.sin(a), zr - 0.02))
                return c0 + u * ((dome(a, min(PH[i], 1.5), 1.0, groove=False) - c0).length + 0.02)

            rings = [[Vector((R * 0.86 * math.cos(a), R * 0.86 * math.sin(a), zr - 0.03)) for a in az]]
            rings += [[reach(i, j) for j in range(8)] for i in (4, 3, 2, 1)] + [[reach(0, 0)]]
            o = _far_shell("mound", rings, leaf_mat)
            tri = [[Vector(o.data.vertices[i].co) for i in (pl.vertices[0], pl.vertices[k_], pl.vertices[k_ + 1])]
                   for pl in o.data.polygons for k_ in range(1, len(pl.vertices) - 1)]

            def skin(u):
                """how far from the middle of the mound its skin is in the direction u"""
                best = 0.0
                for A, B, C in tri:
                    e1, e2 = B - A, C - A
                    h = u.cross(e2)
                    det = e1.dot(h)
                    if abs(det) < 1e-12:
                        continue
                    t_ = c0 - A
                    bu = t_.dot(h) / det
                    qv = t_.cross(e1)
                    bv = u.dot(qv) / det
                    if bu < -1e-6 or bv < -1e-6 or bu + bv > 1.000001:
                        continue
                    best = max(best, e2.dot(qv) / det)
                return best
            return o, [tuple(q) for r in rings for q in r], skin

        mound, mv, skin = far_mound()
        leaf_objs.append(mound)
        fv += mv
    else:
        for (shape, two, mi), (ms, gs, ws) in groups.items():
            b = _blade("leaf_base", two=two, **SH[shape])
            assign(b, ivy_mat if mi else leaf_mat)
            leaf_objs.append(_stamp(f"leaves_{shape}{int(two)}{mi}", b, ms, grime=gs, wear=ws))
    swing += leaf_objs

    # flowers in clumps of one colour, held on short stalks just above the leaves
    if not far:
        petal_b, eye_b, stalk_b = _flower_bases()
        assign(stalk_b, leaf_mat)
        assign(eye_b, mat("paper", p.eye_color))
    n_fl = int(min(50, 58 * p.flower_density * p.fullness ** 0.5 * ks ** 1.5))
    fl = {0: ([], []), 1: ([], [])}
    em = []
    placed = []
    clump = []                                  # (clump, colour, place, size, facing) of every flower
    tries = 0
    while len(placed) < n_fl and tries < 90:
        tries += 1
        ca, cphi = rp_.u(0, 2 * math.pi), math.acos(1 - rp_.u(0.0, 1.0) * (1 - math.cos(1.75)))
        col = 0 if rp_.chance(0.6) else 1
        for _ in range(int(rp_.u(3, 8))):
            a = ca + rp_.jitter(0, 0.42) / max(0.35, math.sin(cphi))
            phi = clamp(cphi + rp_.jitter(0, 0.33), 0.42 if has_candle else 0.05, 1.8)
            n = dome_n(a, phi)
            s = rp_.u(0.02, 0.028) * (1.0 if col == 0 else 0.84) * (0.8 + 0.2 * ks)
            P = dome(a, phi) + n * rp_.u(0.02, 0.045)
            if not clear(P, 0.034) or any((P - q).length < 1.55 * s for q in placed):
                continue
            placed.append(P)
            face = (n + rp_.vec(0.3) + Vector((0, 0, 0.25))).normalized()
            clump.append((tries, col, P, s, face))
            t1 = face.orthogonal().normalized()
            ang = rp_.u(0, 6.28)
            M = _basis(P, t1 * math.cos(ang) + face.cross(t1) * math.sin(ang), face, s)
            fl[col][0].append(M)
            fl[col][1].append(rp_.u(0.0, 0.15))
            em.append(M)
            if len(placed) >= n_fl:
                break
    fl_objs = []
    for col, hexc in ((0, p.flower_color), (1, p.flower_color2)):
        if far:
            # every clump of flowers as one card of its colour, facing as they do, just clear of the mound
            v = []
            for t in sorted({c[0] for c in clump if c[1] == col}):
                cl = [c for c in clump if c[0] == t]
                C = sum((c[2] for c in cl), Vector()) / len(cl)
                u = (C - c0).normalized()
                n = sum((c[4] for c in cl), Vector()).normalized()
                t1 = n.orthogonal().normalized()
                h = math.sqrt(sum(FAR_FL * c[3] ** 2 for c in cl) / 2)
                C = c0 + u * (max((C - c0).length, skin(u)) + 0.014 + 0.2 * h)
                v += [C + t1 * h, C + n.cross(t1) * h, C - t1 * h, C - n.cross(t1) * h]
            if v:
                b = _mesh(f"flowers{col}", v, [(i, i + 1, i + 2, i + 3) for i in range(0, len(v), 4)],
                          [(0.1, 0.5), (0.5, 0.9), (0.9, 0.5), (0.5, 0.1)] * (len(v) // 4))
                assign(b, mat("paper", hexc))
                fl_objs.append(b)
                fv += [tuple(q) for q in v]
        elif fl[col][0]:
            b = duplicate(petal_b, f"petal{col}")
            assign(b, mat("paper", hexc))
            fl_objs.append(_stamp(f"flowers{col}", b, fl[col][0], grime=fl[col][1]))
    if not far:
        delete([petal_b])
        if em:
            fl_objs.append(instances_to_mesh("flower_eyes", eye_b, em, uv_jitter=False, keep_base=False))
            fl_objs.append(instances_to_mesh("flower_stalks", stalk_b, em, uv_jitter=False, keep_base=False))
        else:
            delete([eye_b, stalk_b])
    swing += fl_objs

    # ---------------------------------------------------------------- trailing ivy
    nt = int(p.trailers) if p.trailing_length > 0.04 else 0
    z_tip = zr
    ivy_objs = []
    if nt:
        rt_ = rng.fork("ivy")
        TL = p.trailing_length

        def fall(P0, eo, es, Lh, grow):
            """Hanging run from P0: swings out by `grow`, falls in S-curves and hooks up at the tip."""
            c = min(0.06, 0.3 * Lh)
            m = max(3, int((Lh - c) / 0.04))
            lam_r, lam_t = rt_.u(0.2, 0.36) * (0.7 + 0.5 * TL), rt_.u(0.18, 0.34) * (0.7 + 0.5 * TL)
            ar, at = rt_.u(0.012, 0.03), rt_.u(0.02, 0.045)
            ph1, ph2 = rt_.u(0, 6.28), rt_.u(0, 6.28)
            pull = rt_.u(-0.04, 0.1)
            out = []
            for i in range(1, m + 1):
                s = (Lh - c) * i / m
                env = smoothstep(0.0, 0.1, s) * (0.6 + 0.9 * s / max(Lh, 0.25))
                q = P0 + eo * (grow * (1 - math.exp(-s / 0.035)) - pull * s + ar * env * (math.sin(6.283 * s / lam_r + ph1) - math.sin(ph1) * 0.5)) \
                    + es * (at * env * math.sin(6.283 * s / lam_t + ph2)) - up * s
                if q.z < zr and q.z > zb - 0.03:              # stay clear of the basket
                    rmin = R * math.sqrt(max(0.0, 1 - ((zr - q.z) / (D + 0.03)) ** 2)) + 0.03
                    rq = math.hypot(q.x, q.y)
                    if rq < rmin:
                        q.x, q.y = q.x * rmin / max(rq, 1e-4), q.y * rmin / max(rq, 1e-4)
                out.append(q)
            E = out[-1]
            tg = (E - (out[-2] if len(out) > 1 else P0)).normalized()
            b_ = rt_.u(-1.2, 1.2)
            od = eo * math.cos(b_) + es * math.sin(b_)
            rho = c / 2.3
            for th in (40, 85, 130, 168):
                t_ = math.radians(th)
                out.append(E + tg * (rho * math.sin(t_)) + od * (rho * (1 - math.cos(t_))))
            return out

        strands = []                                           # (points, outward dir, age at start, age at end, skip length)
        lens = [1.0] + [rt_.u(0.16, 0.95) ** 1.25 for _ in range(nt - 1)]
        for k in range(nt):
            a0 = 2 * math.pi * (k + rt_.u(-0.3, 0.3)) / nt + 0.4
            if az_gap(a0) < 0.16:
                a0 += 0.3
            Lk = max(0.07, TL * lens[k])
            er = Vector((math.cos(a0), math.sin(a0), 0))
            et = Vector((-er.y, er.x, 0))
            out = rt_.u(0.045, 0.09) * (0.7 + 0.3 * ks)
            sp = [er * (R * 0.55) + up * (zr + 0.015), er * (R * 0.9) + up * (zr + 0.05 + rt_.u(0, 0.015)),
                  er * (R + out * 0.55) + et * rt_.jitter(0, 0.012) + up * (zr + 0.03)]
            sp += fall(sp[-1], er, et, Lk + 0.03, out * 0.45)
            strands.append((sp, er, 0.0, 1.0, R * 0.42))
            nf = (1 if Lk > 0.16 and rt_.chance(0.65) else 0) + (1 if Lk > 0.5 and rt_.chance(0.6) else 0)
            for _f in range(nf):                               # side shoots
                j = 3 + int(rt_.u(0.1, 0.55) * (len(sp) - 8))
                j = max(3, min(len(sp) - 5, j))
                tj = (zr - sp[j].z) / max(Lk, 0.05)
                sd = rt_.pick([-1, 1])
                side = (et * sd * rt_.u(0.6, 1.0) + er * rt_.u(-0.2, 0.5)).normalized()
                Lb = max(0.06, (Lk - (zr - sp[j].z)) * rt_.u(0.35, 0.8))
                bp = [sp[j], sp[j] + side * 0.022 - up * 0.006, sp[j] + side * 0.045 - up * 0.026]
                bp += fall(bp[-1], side, er, Lb, 0.012)
                strands.append((bp, er, clamp(tj, 0.1, 0.8), 1.0, 0.02))
        z_tip = min(q.z for st in strands for q in st[0])
        tot = sum(path_length([tuple(q) for q in st[0]]) for st in strands)
        cap = int(clamp(48 + 7.5 * tot, 48, 98))

        def lay(scale):
            r2 = rt_.fork("leaves")
            res = []
            for sp, er, t0, t1, skip in strands:
                rs = [Vector(q) for q in resample_path([tuple(q) for q in sp], step=0.006)]
                nn = len(rs)
                Ls = nn * 0.006
                s_next = skip + r2.u(0.0, 0.02)
                side = r2.pick([-1, 1])
                while s_next < Ls - 0.004:
                    i = min(nn - 2, int(s_next / 0.006))
                    t = t0 + (t1 - t0) * s_next / Ls
                    P = rs[i]
                    tg = (rs[min(i + 2, nn - 1)] - rs[max(i - 2, 0)]).normalized()
                    lat = tg.cross(er)
                    if lat.length < 1e-3:
                        lat = Vector((-er.y, er.x, 0))
                    lat = lat.normalized() * side
                    side = -side
                    yd = (lat * r2.u(0.55, 1.0) + tg * r2.u(0.15, 0.7) + er * r2.u(0.0, 0.35) - up * r2.u(0.0, 0.3)).normalized()
                    s = lerp(0.062, 0.023, t ** 1.2) * r2.u(0.82, 1.15) * (0.8 + 0.2 * ks)
                    big = t < 0.55 and r2.chance(0.8)
                    two = P.z < zr - D * 0.7 or math.hypot(P.x, P.y) > R + 0.1
                    M = _basis(P + yd * 0.3 * s, yd, er * 0.75 + r2.vec(0.45) + up * 0.8, (s * r2.pick([-1, 1]), s, s))
                    res.append((big, two, M, r2.pick([0.0, 0.06, 0.12, 0.2]) * (1 - t), smoothstep(0.55, 1.0, t) * 0.22 + r2.u(0, 0.06)))
                    s_next += lerp(0.027, 0.062, t ** 1.1) * scale * r2.u(0.85, 1.2)
            return res

        lv = lay(1.0)
        if len(lv) > cap:
            lv = lay(len(lv) / cap)
        ig = {}
        for big, two, M, g, w in lv:
            gr = ig.setdefault((big, two), ([], [], []))
            gr[0].append(M)
            gr[1].append(g)
            gr[2].append(w)
        if far:
            # every strand with its leaves as one tapering three-sided streamer as wide as its leaves reach, with a
            # station every FAR_STEP so that it swings as the strand does
            v, f = [], []
            for sp, er, t0, t1, skip in strands:
                gone.append((sp[-4:], ivy_mat))                # the tip of a strand hooks up and out: see _far_outline
                q = sp[2 if t0 == 0.0 else 0:]
                q = q[:min(range(len(q)), key=lambda i: q[i].z) + 1]
                cum = [0.0]
                for i in range(1, len(q)):
                    cum.append(cum[-1] + (q[i] - q[i - 1]).length)
                Ls = cum[-1]
                if Ls < 0.03:
                    continue
                ns = max(1, int(round(Ls / FAR_STEP)))
                b = len(v)
                for j in range(ns):
                    s_ = Ls * j / ns
                    i = max(i_ for i_ in range(len(q) - 1) if cum[i_] <= s_ + 1e-9)
                    tg = (q[i + 1] - q[i]).normalized()
                    C = q[i] + tg * (s_ - cum[i])
                    t = t0 + (t1 - t0) * (s_ + 0.5 * Ls / ns) / (Ls + 0.5 * Ls / ns)
                    sl = lerp(0.062, 0.023, t ** 1.2) * (0.8 + 0.2 * ks)
                    w = FAR_IVY * sl
                    lat = tg.cross(er)
                    lat = Vector((-er.y, er.x, 0)) if lat.length < 1e-3 else lat.normalized()
                    v += [C + lat * w, C - lat * w, C + er * (0.4 * w)]
                v.append(q[-1])
                f.append((b, b + 1, b + 2))
                for j in range(ns):
                    r0 = [b + 3 * j + i for i in range(3)]
                    r1 = [b + 3 * j + 3 + (i if j < ns - 1 else 0) for i in range(3)]
                    for i in range(3):
                        fq = (r0[i], r0[(i + 1) % 3], r1[(i + 1) % 3], r1[i])
                        f.append(tuple(fq[k_] for k_ in range(4) if fq[k_] != fq[k_ - 1]))
            if v:
                ivy_objs.append(_far_obj("ivy", v, f, ivy_mat, fix=True))
                fv += [tuple(q) for q in v]
            for (big, two), (ms, gs, ws) in ig.items():
                x, y, z = (0.46, 0.43, -0.07) if big else (0.4, 0.3, -0.05)
                K = [Vector((0.0, -0.35, 0.02)), Vector((x, y, z)), Vector((0.0, 1.0, -0.12)), Vector((-0.92 * x, y, z))]
                gone += [([M @ q for q in K], ivy_mat) for M in ms]
        else:
            for k, (sp, er, t0, t1, skip) in enumerate(strands):
                st = sweep(f"ivy_stem{k}", [tuple(q) for q in sp], 0.0023 * (1.0 - 0.35 * t0), sides=3, step=clamp(0.028 + 0.05 * TL, 0.04, 0.08), cap=False,
                           scale_fn=lambda t: 1.0 - 0.62 * t)
                assign(st, ivy_mat)
                paint(st, grime=0.55)
                ivy_objs.append(st)
            five = [(0.27, -0.02, -0.05), (0.3, 0.26, 0.0), (0.46, 0.43, -0.07), (0.24, 0.62, -0.01)]
            three = [(0.3, 0.02, -0.04), (0.4, 0.3, -0.05), (0.2, 0.56, 0.0)]
            for (big, two), (ms, gs, ws) in ig.items():
                b = _ivy("ivy_base", five if big else three, 0.3 if big else 0.32, two)
                assign(b, ivy_mat)
                ivy_objs.append(_stamp(f"ivy_leaves{int(big)}{int(two)}", b, ms, grime=gs, wear=ws))
    if far:
        ivy_objs += _far_outline(fv, gone)
    swing += ivy_objs

    # ---------------------------------------------------------------- votive candle jar between the chains
    cand = []
    if has_candle:
        z_hood = Tz - d_top
        zj1 = z_hood - 0.011
        zj0 = zj1 - jar_h
        glass = mat("glass_lantern", "#f0a24e", glow=0.8, opacity=0.36)
        if far:
            # jar, flame and hood as low prisms; of the iron inside the jar the cup's plate, which shades the mound
            cand = [_far_turn("votive", [(0.0, zj0), (jar_r, zj0), (jar_r, zj1)], 5, glass, k=1.07),
                    _far_turn("votive_flame", [(0.0, z_flame - 0.005), (0.0055, z_flame + 0.003), (0.0, z_flame + 0.019)], 3,
                              mat("emissive", "#ffb45e", glow=9)),
                    _far_turn("votive_cup", [(0.0, zj0 + 0.014), (0.0205, zj0 + 0.014)], 4, iron),
                    _far_turn("votive_hood", [(0.0, zj1 + 0.011), (hood_r, zj1 + 0.011), (0.0, zj1 + 0.045)], 6, iron, k=1.155),
                    _far_rod("votive_rod", (0, 0, zj1 + 0.043), (0, 0, z_hb - 0.002), 0.002, iron, cap=False)]
        else:
            jar = lathe("votive", [(0.0, zj0), (jar_r - 0.005, zj0), (jar_r - 0.001, zj0 + 0.005), (jar_r + 0.002, zj0 + jar_h * 0.45),
                                   (jar_r - 0.001, zj1 - 0.014), (jar_r - 0.004, zj1 - 0.007), (jar_r - 0.001, zj1), (jar_r - 0.005, zj1)], segments=10)
            wax = lathe("votive_wax", [(0.0105, zj0 + 0.012), (0.0105, zj0 + 0.021), (0.0, zj0 + 0.022)], segments=6)
            for o in (jar, wax):
                assign(o, glass)
            paint(jar, grime=height_ramp(zj1, zj0 + jar_h * 0.45))             # sooted towards the neck
            flame = lathe("votive_flame", [(0.0, z_flame - 0.005), (0.0042, z_flame + 0.001), (0.0032, z_flame + 0.008), (0.0, z_flame + 0.019)], segments=6)
            assign(flame, mat("emissive", "#ffb45e", glow=9))
            cup = lathe("votive_cup", [(0.0, zj0 + 0.003), (0.0135, zj0 + 0.003), (0.0145, zj0 + 0.016), (0.0125, zj0 + 0.016), (0.012, zj0 + 0.011)], segments=7)
            hood = lathe("votive_hood", [(0.0, zj1 + 0.027), (0.011, zj1 + 0.026), (hood_r - 0.003, zj1 + 0.01), (hood_r, zj1 + 0.011), (hood_r, zj1 + 0.013),
                                         (0.012, zj1 + 0.031), (0.0055, zj1 + 0.037), (0.0, zj1 + 0.045)], segments=8)
            rod = sweep("votive_rod", [(0, 0, zj1 + 0.043), (0.001, 0, (Tz + zj1) / 2), (0, 0, Tz - 0.004)], 0.0015, sides=3, smooth_path=False, cap=False)
            eye = torus("votive_eye", 0.0048, 0.0014, loc=(0, 0, Tz - 0.0005), rot=(90, 0, 0), major_seg=6, minor_seg=3)
            iron_bits = [cup, hood, rod, eye]
            for k in range(3):
                a = math.radians(30 + 120 * k)
                ca, sa = math.cos(a), math.sin(a)
                iron_bits.append(sweep(f"votive_stay{k}", [((jar_r - 0.001) * ca, (jar_r - 0.001) * sa, zj1 - 0.007), ((hood_r - 0.004) * ca, (hood_r - 0.004) * sa, zj1 + 0.011)],
                                       0.0012, sides=3, smooth_path=False, cap=False))
            for o in iron_bits:
                assign(o, iron)
                paint(o, special=rust_fn, grime=height_ramp(zj1 + 0.045, zj1))      # soot under the hood
            cand = [jar, wax, flame] + iron_bits
        light("candle", (0, 0, z_flame + 0.03), color="#ffb766", intensity=p.candle_cd, range=3.0, flicker=0.4, shadow_near=0.06)
    swing += cand

    if not far:
        print("DBG tris hook+ring %d hang %d basket %d core %d leaves %d flowers %d ivy %d candle %d total %d" % (
            _tris([hook, ring]), _tris(hang), _tris(basket), _tris([core]), _tris(leaf_objs), _tris(fl_objs), _tris(ivy_objs), _tris(cand),
            _tris([hook] + swing)))
        print("DBG zr %.3f bottom %.3f tip %.3f mound top %.3f flame %.3f leaves %s ivy %d" % (
            zr, z_low, z_tip, zc + hm, z_flame if has_candle else 0.0, {k[0] + str(int(k[1])) + str(k[2]): len(v[0]) for k, v in groups.items()},
            sum(len(o.data.polygons) for o in ivy_objs)))

    part(swing, "basket", pivot=(0, 0, z_hb), motion="swing", amp=2.5, speed=0.3)
    socket("under", "attach", (0, 0, z_low), role="lantern")
