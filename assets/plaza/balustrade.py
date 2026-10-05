"""Classical dressed-limestone balustrade run along X, as on the retaining walls and stair sides of the
plaza. Built the way a mason builds it: a rough-dressed base course, a moulded plinth rail, turned
balusters (lathe; square base block + abacus, engaged half-balusters against the piers), a moulded
handrail coping (ovolo bed-mould, fillet, projecting nose with a drip groove, weathered top) laid in
lengths with lime-mortar joints and iron dog-cramps, and square piers (rough plinth block, moulded base,
fielded-panel die, corniced cap, ball / urn / pyramid / lantern finial). Every stone is its own block,
slightly out of true and tooled in its own direction; arrises carry a few large conchoidal spalls,
the tops carry crustose lichen (grey-white / ochre crusts, pale washed patches), moss cushions grow
in the damp foot and joints, a few balusters are paler replacements and some may be snapped off with
the fallen top lying on the paving.  The run spans exactly x in [-length/2, +length/2].
"""
from tk import *
from math import sqrt, sin, cos, tan, pi, radians, atan2
import numpy as np
import bmesh
from mathutils import Vector, Matrix, noise as mnoise
from mathutils.bvhtree import BVHTree

# auto_wear 0: on these tinted stone materials the `wear` mask renders near-black, not paler - the
# automatic convex-edge wear put a sooty line on every arris and blotches on the cap tops.
META = {"name": "Stone balustrade run", "class": "structure", "mount": "ground", "focus": 0.66, "auto_wear": 0.0, "lod": "far"}
FRESH = [None]          # material of freshly broken stone (set in build)
PARAMS = {
    "length": {"default": 4.2, "min": 1.0, "max": 12.0, "desc": "overall run length along X (m); spans exactly +-length/2"},
    "height": {"default": 1.0, "min": 0.8, "max": 1.25, "desc": "height to the top of the handrail coping (m)"},
    "width": {"default": 0.30, "min": 0.24, "max": 0.42, "desc": "width of the handrail coping; balusters and piers scale with it (m)"},
    "pier_spacing": {"default": 2.0, "min": 1.0, "max": 12.0, "desc": "target distance between pier centres (m); >= length gives one clear bay; a very long run with very close piers is spaced out to stay in budget"},
    "ends": {"default": "both", "choices": ["both", "left", "right", "none"], "desc": "which ends finish in a pier (open ends butt a wall or the next run); very short runs drop piers"},
    "baluster": {"default": "vase", "choices": ["vase", "double", "square"], "desc": "baluster style: single vase, symmetrical double vase, square-section"},
    "pitch": {"default": 1.45, "min": 1.2, "max": 2.2, "desc": "baluster centre spacing as a multiple of the belly diameter"},
    "cap": {"default": "ball", "choices": ["ball", "urn", "flat", "lantern"], "desc": "pier finial; 'lantern' puts gas lanterns on the first and last pier (balls between)"},
    "stone": {"default": "#bdb7a6", "desc": "stone tint"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age: spalled arrises, stones out of true, rain staining"},
    "moss": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "lichen crusts on the tops and moss cushions along the damp foot / joints"},
    "repairs": {"default": 0.08, "min": 0.0, "max": 0.5, "desc": "fraction of balusters that are paler, newer replacements"},
    "broken": {"default": 0, "min": 0, "max": 3, "desc": "number of balusters snapped off (stub + fallen top on the paving)"},
    "cramps": {"default": True, "desc": "iron dog-cramps let into the coping across its joints"},
    "paint": {"default": "#1b2622", "desc": "lantern ironwork paint (cap = lantern)"},
    "intensity": {"default": 9.0, "min": 2, "max": 60, "desc": "lantern intensity (cd, cap = lantern)"},
}
VARIANTS = {
    "terrace_urns": {"length": 6.4, "pier_spacing": 3.0, "baluster": "double", "cap": "urn", "stone": "#c6bfa9",
                     "wear": 0.3, "moss": 0.3, "seed": 4},
    "stair_lanterns": {"length": 3.0, "pier_spacing": 3.0, "cap": "lantern", "height": 1.05, "stone": "#989689",
                       "wear": 0.5, "moss": 0.4, "seed": 9},
    "old_mossy_square": {"length": 5.0, "pier_spacing": 2.4, "baluster": "square", "cap": "flat", "width": 0.34,
                         "pitch": 1.6, "stone": "#a3a294", "wear": 1.0, "moss": 1.0, "repairs": 0.2, "broken": 2,
                         "seed": 16},
    "infill_open": {"length": 2.6, "ends": "none", "pier_spacing": 12.0, "height": 0.9, "width": 0.26, "pitch": 1.7,
                    "stone": "#bdb49f", "wear": 0.25, "moss": 0.2, "repairs": 0.0, "seed": 23},
}

SQUARE_L = [(0.90, 0.0), (0.90, 0.045), (0.78, 0.060), (0.62, 0.095), (0.62, 0.110), (0.84, 0.200), (0.88, 0.270),
            (0.72, 0.480), (0.52, 0.720), (0.50, 0.795), (0.60, 0.808), (0.60, 0.836), (0.50, 0.850), (0.62, 0.925),
            (0.84, 0.972), (0.84, 1.0)]

S2 = sqrt(2.0)
JOINT = 0.008                      # mortar joint width between stones (m)
TOOL = 27.0                        # slant of the tooling lines in the stone_smooth texture (deg from +U)
TRI_TARGET = 63000

# turned profiles: (radius / max radius, height fraction of the turned part)
VASE = [(0.86, 0.0), (0.97, 0.008), (1.0, 0.022), (0.97, 0.036), (0.86, 0.044),                 # base torus
        (0.80, 0.047), (0.80, 0.058), (0.66, 0.068), (0.60, 0.084), (0.63, 0.100),              # fillet + scotia
        (0.70, 0.106), (0.70, 0.116),                                                           # fillet
        (0.84, 0.150), (0.95, 0.195), (1.0, 0.245), (0.985, 0.300), (0.91, 0.360), (0.79, 0.430),
        (0.65, 0.510), (0.53, 0.590), (0.45, 0.670), (0.405, 0.740), (0.39, 0.790),             # belly -> neck
        (0.47, 0.796), (0.53, 0.808), (0.53, 0.820), (0.47, 0.832),                             # astragal
        (0.40, 0.838), (0.40, 0.870), (0.47, 0.900), (0.60, 0.935), (0.76, 0.962), (0.84, 0.975),  # echinus
        (0.88, 0.980), (0.88, 1.0)]
VASE_L = [(0.86, 0.0), (0.98, 0.010), (0.98, 0.034), (0.86, 0.044),
          (0.80, 0.050), (0.64, 0.070), (0.61, 0.090), (0.70, 0.110),
          (0.86, 0.155), (0.98, 0.210), (1.0, 0.250), (0.95, 0.330), (0.80, 0.425), (0.62, 0.530), (0.47, 0.650),
          (0.39, 0.790), (0.53, 0.805), (0.53, 0.822), (0.40, 0.838), (0.42, 0.880), (0.60, 0.935), (0.84, 0.975),
          (0.88, 1.0)]
_DH = [(0.86, 0.0), (0.97, 0.008), (1.0, 0.020), (0.97, 0.032), (0.86, 0.040), (0.80, 0.043), (0.80, 0.052),
       (0.68, 0.062), (0.64, 0.075), (0.70, 0.088), (0.70, 0.096),
       (0.84, 0.125), (0.95, 0.160), (1.0, 0.200), (0.97, 0.245), (0.87, 0.295), (0.72, 0.350), (0.57, 0.400),
       (0.46, 0.440), (0.42, 0.466), (0.50, 0.472), (0.58, 0.484), (0.60, 0.5)]
DOUBLE = _DH + [(r, 1.0 - t) for (r, t) in reversed(_DH[:-1])]
_DL = [(0.86, 0.0), (0.98, 0.010), (0.98, 0.030), (0.86, 0.040), (0.66, 0.068), (0.70, 0.092), (0.90, 0.140),
       (1.0, 0.200), (0.92, 0.270), (0.70, 0.360), (0.44, 0.455), (0.58, 0.478), (0.60, 0.5)]
DOUBLE_L = _DL + [(r, 1.0 - t) for (r, t) in reversed(_DL[:-1])]
VASE_XL = [(0.86, 0.0), (0.98, 0.012), (0.98, 0.034), (0.80, 0.048), (0.62, 0.085), (0.72, 0.112), (0.98, 0.205),
           (0.98, 0.270), (0.80, 0.425), (0.55, 0.590), (0.39, 0.790), (0.53, 0.812), (0.40, 0.838), (0.44, 0.890),
           (0.84, 0.975), (0.88, 1.0)]
_DX = [(0.86, 0.0), (0.98, 0.012), (0.98, 0.030), (0.68, 0.070), (0.72, 0.095), (0.98, 0.170), (0.98, 0.230),
       (0.70, 0.360), (0.44, 0.455), (0.60, 0.5)]
DOUBLE_XL = _DX + [(r, 1.0 - t) for (r, t) in reversed(_DX[:-1])]
SQUARE = [(0.90, 0.0), (0.90, 0.045), (0.78, 0.060), (0.78, 0.075), (0.62, 0.095), (0.62, 0.110), (0.74, 0.135),
          (0.84, 0.200), (0.88, 0.270), (0.84, 0.360), (0.72, 0.480), (0.60, 0.600), (0.52, 0.720), (0.50, 0.795),
          (0.60, 0.808), (0.60, 0.836), (0.50, 0.848), (0.50, 0.880), (0.62, 0.925), (0.80, 0.965), (0.84, 0.972),
          (0.84, 1.0)]


# ------------------------------------------------------------------------------------------------
# small local helpers
# ------------------------------------------------------------------------------------------------

def tint_shift(hexc, f, cool=0.0):
    """Scale an sRGB '#rrggbb' colour (lighter / darker bed of the same quarry), optionally cooler."""
    h = hexc.lstrip("#")
    c = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    g = (f * (1.0 - cool), f, f * (1.0 + cool))
    return "#%02x%02x%02x" % tuple(int(clamp(v * q, 0, 255)) for v, q in zip(c, g))


def arc(cy, cz, r, a0, a1, n, rz=None):
    rz = r if rz is None else rz
    return [(cy + r * cos(radians(a0 + (a1 - a0) * i / n)), cz + rz * sin(radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def mirror_section(right):
    """right half (y >= 0), bottom -> top, last point on the axis -> full CCW (y, z) section."""
    return right + [(-y, z) for (y, z) in reversed(right[:-1])]


def coping_section(a, k, lo=False):
    """Handrail coping, z from 0 (bed) to 0.16 k: bed fillet, ovolo bed-mould, fillet, soffit with a
    drip groove, projecting nose, worn rounded top arris, cambered (weathered) top."""
    yb = a - 0.074 * k
    pts = [(yb, 0.0), (yb, 0.012 * k)]
    na = 2 if lo else 4                 # long runs: the same moulding drawn with fewer facets
    pts += arc(yb, 0.064 * k, 0.040 * k, -90, 0, na, rz=0.052 * k)[1:]                # ovolo bed-mould
    pts += [(yb + 0.040 * k, 0.072 * k),                                               # fillet
            (a - 0.029 * k, 0.072 * k), (a - 0.027 * k, 0.083 * k), (a - 0.013 * k, 0.083 * k), (a - 0.011 * k, 0.072 * k),  # drip
            (a - 0.004 * k, 0.072 * k), (a, 0.076 * k), (a, 0.124 * k)]                # nose
    pts += arc(a - 0.026 * k, 0.124 * k, 0.026 * k, 0, 78, na)[1:]                     # worn top arris
    ye, ze = pts[-1]
    top = 0.160 * k
    for t in ((0.5,) if lo else (0.6, 0.25)):
        pts.append((ye * t, ze + (top - ze) * (1 - t * t)))
    pts.append((0.0, top))
    return mirror_section(pts)


def plinth_section(a, k, lo=False):
    """Plinth rail: upstand, cavetto, fillet, rounded top bed for the balusters. z from 0, 0.08 k tall."""
    pts = [(a - 0.003, 0.0), (a, 0.003), (a, 0.014 * k)]
    pts += arc(a, 0.048 * k, 0.034 * k, 270, 180, 2 if lo else 4)[1:]
    pts += [(a - 0.034 * k, 0.068 * k)]
    pts += arc(a - 0.046 * k, 0.068 * k, 0.012 * k, 0, 90, 1 if lo else 2)[1:]
    ye = pts[-1][0]
    pts += [(ye * 0.5, 0.08 * k), (0.0, 0.08 * k)]
    return mirror_section(pts)


def inset_section(sec, d):
    """Offset a CCW section inwards by d (mitred) - the recessed mortar seen in a joint / end chamfers."""
    m = len(sec)
    out = []
    for i in range(m):
        p0, p1, p2 = Vector(sec[i - 1]), Vector(sec[i]), Vector(sec[(i + 1) % m])
        e1, e2 = (p1 - p0), (p2 - p1)
        if e1.length < 1e-9 or e2.length < 1e-9:
            out.append(tuple(p1))
            continue
        e1.normalize()
        e2.normalize()
        n1, n2 = Vector((-e1.y, e1.x)), Vector((-e2.y, e2.x))
        b = n1 + n2
        if b.length < 1e-6:
            b = n1.copy()
        b.normalize()
        b /= max(b.dot(n1), 0.5)
        out.append(tuple(p1 + b * d))
    return out


def prism_x(name, sec, x0, x1, step=0.15, chamfer=0.0, caps=True, tuck=(False, False)):
    """Extrude a CCW (y, z) section along X with stations every ~step (for masks / displacement).
    chamfer > 0 eases the two end arrises (the moulding arrises are drawn into the section)."""
    n = max(1, int(round((x1 - x0) / step)))
    m = len(sec)
    rows = [(x0 + (x1 - x0) * i / n, sec) for i in range(n + 1)]
    if chamfer > 0 and (x1 - x0) / n > 0.07:        # a tight station by each end keeps contact shading at the joint
        rows = rows[:1] + [(x0 + 0.03, sec)] + rows[1:-1] + [(x1 - 0.03, sec)] + rows[-1:]
    # an end tucked under a pier cap: more stations there, or the baked contact shadow of the overhang
    # is interpolated 15 cm along the top and reads as a soot smear
    for end, sgn, on in ((x0, 1.0, tuck[0]), (x1, -1.0, tuck[1])):
        if on:
            for d_ in (0.062, 0.1, 0.16):
                xs_ = end + sgn * d_
                if all(abs(xs_ - r_[0]) > 0.022 for r_ in rows) and x0 < xs_ < x1:
                    rows.append((xs_, sec))
    rows.sort(key=lambda r_: r_[0])
    if chamfer > 0:
        ins = inset_section(sec, chamfer)
        rows[0] = (x0 + chamfer, sec)
        rows[-1] = (x1 - chamfer, sec)
        rows = [(x0, ins)] + rows + [(x1, ins)]
    verts = [(x, y, z) for (x, s) in rows for (y, z) in s]
    faces = []
    for i in range(len(rows) - 1):
        for j in range(m):
            j2 = (j + 1) % m
            faces.append((i * m + j, i * m + j2, (i + 1) * m + j2, (i + 1) * m + j))
    if caps:
        faces.append(tuple(range(m - 1, -1, -1)))
        faces.append(tuple((len(rows) - 1) * m + j for j in range(m)))
    return mesh_object(name, verts, faces)


def drop_faces(o, pred):
    """Delete faces where pred(center, normal) (local space) - hidden bedding faces."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    dead = [f_ for f_ in bm.faces if pred(f_.calc_center_median(), f_.normal)]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context='FACES')
        bm.to_mesh(o.data)
        o.data.update()
    bm.free()
    return o


def rough_block(name, size, loc, rot, chamfer, rough, seed, max_edge=0.2):
    """A rough-dressed block (cheaper than tk.stone_block: one chamfer segment, coarse grid)."""
    o = box(name, size, bevel=min(chamfer, min(size) * 0.2), segments=1)
    densify(o, max_edge)
    displace_noise(o, strength=rough, scale=max(size) * 0.6, seed=seed)
    displace_noise(o, strength=rough * 0.5, scale=max(size) * 0.15, seed=seed + 7)
    set_loc_rot(o, loc, rot)
    o["tsq_smooth"] = 45.0
    return o


def stone_uv(o, r, modes=(-90, -90, -90, 0, -55, -125)):
    """Box UVs, then turn the texture so this stone's tooling runs its own way (mostly upright
    batting, some along, some diagonal) with its own scale and offset."""
    uv_box(o)
    realize(o)
    ang = TOOL + r.pick(modes) + r.u(-7, 7) + r.pick((0, 180))
    s = r.u(0.85, 1.25)
    uv_transform(o, scale=(s, s), offset=(r.u(0, 1), r.u(0, 1)), rot_deg=ang)
    uv_keep(o)


def turned_uv(o, r, squeeze=0.3):
    """Lathe UVs (U around, V along the profile): squeeze U and turn the texture so the drag lines
    run round the work like lathe rings instead of spiralling."""
    uv_transform(o, scale=(squeeze, 1.0))
    uv_transform(o, rot_deg=TOOL + r.u(-3, 3), offset=(r.u(0, 1), r.u(0, 1)))
    uv_keep(o)


SPALL = {"h": 1.0, "n": 0, "tris": 0, "cap": 1e9}   # facet scale / running cost / budget of the spalls


def spall(obj, p, t, b, length, depth, crng, m=None, width=None):
    """Knock a conchoidal spall off an arris. The fracture is a shallow ellipsoid whose long axis runs
    along the arris (t) and that bites `depth` into the stone along the outward bisector b. No boolean
    (their needle slivers bake as black vertices): the stone is refined locally and the surface inside
    the ellipsoid is driven back onto it. p / t / b are world space; `length` x `depth` is what is lost."""
    if SPALL["tris"] >= SPALL["cap"]:
        return None
    t = Vector(t).normalized()
    b = Vector(b).normalized()
    rb = crng.u(4.0, 6.0) * depth           # a flat lens: the fracture is a shallow facet, not a pit
    rc = (width or crng.u(0.07, 0.1)) * 0.5 + depth
    ta = length * 0.5 / sqrt(1.0 - (1.0 - depth / rb) ** 2)      # `length` = what is missing from the arris
    rm = Matrix.Rotation(radians(crng.u(-24, 24)), 3, t)
    b2 = rm @ b                             # the facet leans towards one of the two faces
    c2 = t.cross(b2).normalized()
    centre = Vector(p) + b2 * (rb - depth) + t * crng.u(-0.004, 0.004)
    basis = Matrix((t * ta, b2 * rb, c2 * rc)).transposed()
    binv = basis.inverted()
    mw = world_matrix(obj)
    mwi = mw.inverted()
    to_q = binv.to_4x4() @ Matrix.Translation(-centre) @ mw      # local -> unit-sphere space of the fracture
    h = min(max(min(length, 2.0 * rc) / 8.0, 0.0045), 0.008) * SPALL["h"]     # facet size inside the spall
    ph = Vector((crng.u(0, 9), crng.u(0, 9), crng.u(0, 9)))
    keep = obj.data.copy()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    t_before = sum(len(f_.verts) - 2 for f_ in bm.faces)

    def crossing():
        qs = {v.index: to_q @ v.co for v in bm.verts}
        out = []
        for e in bm.edges:
            q0, q1 = qs[e.verts[0].index], qs[e.verts[1].index]
            d = q1 - q0
            dd = d.length_squared
            f_ = min(max(-q0.dot(d) / dd, 0.0), 1.0) if dd > 1e-12 else 0.0
            if (q0 + d * f_).length_squared < 1.25:
                out.append(e)
        return out

    bm.verts.index_update()
    hot = crossing()
    ngons = {f_ for e in hot for f_ in e.link_faces if len(f_.verts) > 4}
    if ngons:
        bmesh.ops.triangulate(bm, faces=list(ngons), quad_method='BEAUTY', ngon_method='BEAUTY')
    for _ in range(7):
        bm.verts.index_update()
        long_ = [e for e in crossing() if e.calc_length() > h * 1.45]
        if not long_:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_, cuts=1, use_grid_fill=True, use_single_edge=True)
    bm.normal_update()
    bl = (mwi.to_3x3() @ b).normalized()
    bq = binv @ b
    bq2 = bq.length_squared
    moved = []
    for v in bm.verts:
        q = to_q @ v.co
        if q.length_squared >= 1.0 or v.normal.dot(bl) < -0.25:
            continue
        qb = q.dot(bq)
        s_ = (qb + sqrt(max(qb * qb - bq2 * (q.length_squared - 1.0), 0.0))) / bq2
        w = mw @ v.co
        s_ *= 0.88 + 0.3 * mnoise.noise(w * 55.0 + ph)        # rippled, never a clean scoop
        if s_ > 1e-5:
            moved.append((v, s_))
    faces = {f_ for v, _ in moved for f_ in v.link_faces}
    n0 = {f_: f_.normal.copy() for f_ in faces}
    co0 = [v.co.copy() for v, _ in moved]
    ok = False
    for gain in (1.0, 0.62, 0.38):          # a thin moulding folded over: take a shallower flake instead
        for (v, s_), c_ in zip(moved, co0):
            v.co = c_ - bl * (s_ * gain)
        bm.normal_update()
        if moved and all(f_.normal.dot(n0[f_]) > 0.05 for f_ in faces):
            ok = True
            break
    if not ok:
        bm.free()
        obj.data = keep
        return None
    moved = [(v, s_ * gain) for v, s_ in moved]
    # the fracture face shows fresh, paler stone: its own material slot (also gives the flake a crisp rim)
    if FRESH[0] is not None:
        mats = obj.data.materials
        if len(mats) == 0 and m is not None:
            assign(obj, m)
        idx = next((i for i, mm in enumerate(mats) if mm == FRESH[0]), None)
        if idx is None:
            idx = len(mats)
            assign(obj, FRESH[0], slot=idx)
        deep = {v for v, s_ in moved if s_ > 0.0012}
        for f_ in faces:
            if all(v in deep for v in f_.verts):
                f_.material_index = idx
    SPALL["n"] += 1
    SPALL["tris"] += sum(len(f_.verts) - 2 for f_ in bm.faces) - t_before
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return True


def _bounds(o):
    c = np.empty(len(o.data.vertices) * 3, dtype=np.float32)
    o.data.vertices.foreach_get("co", c)
    c = c.reshape(-1, 3)
    return c.min(axis=0), c.max(axis=0)


def orient_out(o):
    """Make the faces of a (possibly open) lathe shell point away from its axis."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    tot = 0.0
    for f_ in bm.faces:
        c = f_.calc_center_median()
        tot += (f_.normal.x * c.x + f_.normal.y * c.y) * f_.calc_area()
    if tot < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        bm.to_mesh(o.data)
        o.data.update()
    bm.free()
    return o


def seat_ring(o, z_top, r_a, r_b, n=16):
    """Re-mesh the flat finial seat on top of a pier cap (object still centred on its axis): the square
    seat edge -> a circle just outside the finial foot (r_a) -> a circle under the foot (r_b) -> fan.
    The vertices buried under the finial bake black; this keeps that inside the foot's contact line
    instead of letting the big seat triangles smear it over the cap top. Returns False if not possible."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    seat = [f_ for f_ in bm.faces if all(v.co.z > z_top - 1e-4 for v in f_.verts)]
    if not seat:
        bm.free()
        return False
    sset = set(seat)
    bverts = []
    for f_ in seat:
        for e in f_.edges:
            if sum(1 for g in e.link_faces if g in sset) == 1:
                bverts += list(e.verts)
    bverts = list({v.index: v for v in bverts}.values())
    if len(bverts) < 4 or min(v.co.xy.length for v in bverts) < r_a + 0.002:
        bm.free()
        return False
    bmesh.ops.delete(bm, geom=seat, context='FACES')
    bverts = sorted(bverts, key=lambda v: atan2(v.co.y, v.co.x))
    pa = [atan2(v.co.y, v.co.x) for v in bverts]
    a0 = pa[0] + 0.01
    qa = [a0 + 2 * pi * i / n for i in range(n)]
    ring_a = [bm.verts.new((r_a * cos(t), r_a * sin(t), z_top)) for t in qa]
    ring_b = [bm.verts.new((r_b * cos(t), r_b * sin(t), z_top)) for t in qa]
    cen = bm.verts.new((0.0, 0.0, z_top))
    m = len(bverts)
    i = j = 0
    while i < m or j < n:
        nxt_p = pa[i + 1] if i + 1 < m else pa[0] + 2 * pi
        nxt_q = qa[j + 1] if j + 1 < n else qa[0] + 2 * pi
        if j >= n or (i < m and nxt_p <= nxt_q):
            bm.faces.new((bverts[i % m], bverts[(i + 1) % m], ring_a[j % n]))
            i += 1
        else:
            bm.faces.new((bverts[i % m], ring_a[(j + 1) % n], ring_a[j % n]))
            j += 1
    for j in range(n):
        j2 = (j + 1) % n
        bm.faces.new((ring_a[j], ring_a[j2], ring_b[j2], ring_b[j]))
        bm.faces.new((ring_b[j], ring_b[j2], cen))
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.normal_update()
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    return True


def open_seat(o, c, tol=0.004):
    """Drop the hidden centre fan of a seat made by seat_ring (after the booleans, which want a closed mesh)."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    c = world_matrix(o).inverted() @ Vector(c)
    best = min(bm.verts, key=lambda v: (v.co - c).length)
    if (best.co - c).length < tol:
        bmesh.ops.delete(bm, geom=[best], context='VERTS')
        bm.to_mesh(o.data)
        o.data.update()
    bm.free()


def split_run(x0, x1, target, snaps, r, avoid=(), nmin=1):
    """Cut [x0, x1] into stones about `target` long, joints snapped to `snaps` (baluster centres)."""
    n = max(nmin, int(round((x1 - x0) / target)))
    cuts = []
    for j in range(1, n):
        t = x0 + (x1 - x0) * j / n + r.u(-0.1, 0.1)
        cand = [s for s in snaps if x0 + 0.3 < s < x1 - 0.3 and all(abs(s - c) > 0.3 for c in cuts)
                and all(abs(s - c) > 0.05 for c in avoid)]
        if cand:
            t = min(cand, key=lambda s: abs(s - t))
        elif not (x0 + 0.3 < t < x1 - 0.3):
            continue
        cuts.append(t)
    return [x0] + sorted(cuts) + [x1]


def panelled_die(name, s, h, sides, margin, k, edge):
    """Pier die (local, centred): an eased block with a sunk, fielded panel (splayed frame, sunk ground,
    raised field) on the faces listed in `sides`."""
    o = box(name, (s, s, h), bevel=edge, segments=2)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    for f_ in list(bm.faces):
        n = f_.normal.copy()
        if f_.calc_area() < 0.5 * s * h:
            continue
        key = "-y" if n.y < -0.9 else "+y" if n.y > 0.9 else "-x" if n.x < -0.9 else "+x" if n.x > 0.9 else None
        if key not in sides:
            continue
        for thick, push in ((margin, 0.0), (0.012 * k, -0.016 * k), (0.028 * k, 0.0), (0.013 * k, 0.010 * k)):
            bmesh.ops.inset_region(bm, faces=[f_], thickness=thick, depth=0.0, use_even_offset=True, use_boundary=True)
            if push:
                for v in f_.verts:
                    v.co += n * push
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    return o


def fracture_cutter(name, zc, rake, yaw, half, top, frng, mats=None):
    """A block whose underside is a ragged, raked fracture surface (boolean cutter for a snapped stone)."""
    bm = bmesh.new()
    n = 9
    g = [[bm.verts.new(((i / (n - 1) - 0.5) * 2 * half, (j / (n - 1) - 0.5) * 2 * half, 0.0)) for j in range(n)] for i in range(n)]
    for i in range(n - 1):
        for j in range(n - 1):
            bm.faces.new((g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]))
    ret = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    for el in ret["geom"]:
        if isinstance(el, bmesh.types.BMVert):
            el.co.z = top
    step_at = frng.u(-0.3, 0.3) * half
    slope = tan(radians(rake))              # sheared, not tilted: the sides stay upright, nothing is left hanging
    for row in g:
        for v in row:
            v.co.z = (frng.u(-0.007, 0.007) + (0.012 if v.co.y > step_at else -0.004) + 0.05 * sin(v.co.x * 23.0 + v.co.y * 9.0) * half
                      + slope * v.co.y)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    o = from_bmesh(name, bm)
    o.data.transform(Matrix.Translation((0, 0, zc)) @ Matrix.Rotation(radians(yaw), 4, 'Z'))
    if mats:                                # the cut faces take the cutter's slot 1 = fresh stone
        assign(o, mats[0])
        assign(o, mats[1], slot=1)
        o.data.polygons.foreach_set("material_index", [1] * len(o.data.polygons))
    o.data.update()
    return o


def square_shaft(name, pts, c):
    """Square-section shaft from (half side, z) stations with eased corners only (long runs: the full
    arris bevel of the close-up shaft costs twice the triangles). c = 0 leaves plain corners."""
    verts, faces = [], []
    n = 8 if c > 0 else 4
    for hs, z in pts:
        e = min(c, hs * 0.3)
        if c > 0:
            ring = [(hs, -hs + e), (hs, hs - e), (hs - e, hs), (-hs + e, hs), (-hs, hs - e), (-hs, -hs + e), (-hs + e, -hs), (hs - e, -hs)]
        else:
            ring = [(hs, -hs), (hs, hs), (-hs, hs), (-hs, -hs)]
        verts += [(x, y, z) for x, y in ring]
    for i in range(len(pts) - 1):
        for j in range(n):
            j2 = (j + 1) % n
            faces.append((i * n + j, i * n + j2, (i + 1) * n + j2, (i + 1) * n + j))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple((len(pts) - 1) * n + j for j in range(n)))
    o = mesh_object(name, verts, faces)
    o["tsq_smooth"] = 30.0
    return o


def make_baluster(style, R, hbal, seg, lite, brng, nchips, m, closed=False, parts_only=False):
    """One baluster: square base block, turned shaft, square abacus. Bedding faces are left out."""
    k = R / 0.0705
    hbase, habac = 0.055 * k, 0.045 * k
    ht = hbal - hbase - habac
    if style == "square":
        pts = [(rf * R * 0.9 * S2, hbase + tf * ht) for rf, tf in (SQUARE_L if int(lite) >= 2 else SQUARE)]
    else:
        prof = (VASE, VASE_L, VASE_XL)[int(lite)] if style == "vase" else (DOUBLE, DOUBLE_L, DOUBLE_XL)[int(lite)]
        pts = [(rf * R, hbase + tf * ht) for rf, tf in prof]
    pts[0] = (pts[0][0], hbase - 0.003)                 # housed 3 mm into the base block and abacus
    pts[-1] = (pts[-1][0], hbase + ht + 0.003)
    if style == "square" and int(lite) >= 2 and not closed:
        turned = square_shaft("bal_turned", [(r / S2, z) for r, z in pts], 0.006 * k if int(lite) == 2 else 0.0)
        stone_uv(turned, brng, modes=(-90,))
    pts = [(0.0, hbase - 0.003)] + pts + [(0.0, hbase + ht + 0.003)]     # closed: needed for clean normals / booleans
    if style == "square" and int(lite) >= 2 and not closed:
        pass
    elif style == "square":
        turned = lathe("bal_turned", pts, segments=4, rot=(0, 0, 45), smooth_angle=30)
        apply_transform(turned)
        bevel(turned, 0.005 * k, 1)
        realize(turned)
        uv_transform(turned, rot_deg=TOOL - 90 + brng.u(-6, 6), offset=(brng.u(0, 1), brng.u(0, 1)))
        uv_keep(turned)
    else:
        turned = lathe("bal_turned", pts, segments=seg)
        turned_uv(turned, brng)
    displace_noise(turned, strength=0.0014, scale=0.11, seed=brng.randrange(999))
    base = box("bal_base", (2.14 * R, 2.14 * R, hbase), loc=(0, 0, hbase / 2), bevel=0.006 * k, segments=1)
    abac = box("bal_abacus", (2.0 * R, 2.0 * R, habac), loc=(0, 0, hbal - habac / 2), bevel=0.005 * k, segments=1)
    assign([turned, base, abac], m)
    chips = []
    for _ in range(nchips):
        sx, sy = brng.pick((-1, 1)), brng.pick((-1, 1))
        tdir = (-sx, sy, 0.0)
        if brng.chance(0.7):
            chips.append(spall(base, (sx * 1.07 * R, sy * 1.07 * R, hbase), tdir, (sx, sy, 0.9),
                               brng.u(0.045, 0.07) * k, brng.u(0.010, 0.016) * k, brng, m, width=0.03 * k))
        else:
            chips.append(spall(abac, (sx * R, sy * R, hbal - habac), tdir, (sx, sy, -0.9),
                               brng.u(0.035, 0.05) * k, brng.u(0.008, 0.012) * k, brng, m, width=0.025 * k))
    drop_faces(base, lambda c, n: n.z < -0.9 and c.z < -hbase * 0.4)
    drop_faces(abac, lambda c, n: n.z > 0.9 and c.z > habac * 0.4)
    for o in (base, abac):
        stone_uv(o, brng, modes=(-90, 0))
    if parts_only:
        return turned, base, abac, chips
    j = join([turned, base, abac], "baluster")
    uv_keep(j)
    return j


def bisect_half(obj, keep_negative_x=True):
    """Cut a baluster in half on the YZ plane (engaged half-baluster against a pier face)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(0.004, 0, 0), plane_no=(1, 0, 0),
                           clear_outer=keep_negative_x, clear_inner=not keep_negative_x)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def moss_dome(name, seg, rings, seed):
    """Low lumpy cushion: an open dome (no hidden underside), unit radius, rim sunk below z = 0."""
    els = (-22.0, 28.0, 62.0) if rings >= 2 else (-18.0, 42.0)
    verts, faces = [], []
    for j, el in enumerate(els):
        for i in range(seg):
            az = 2 * pi * (i + 0.5 * (j % 2)) / seg
            verts.append((cos(radians(el)) * cos(az), cos(radians(el)) * sin(az), sin(radians(el))))
    verts.append((0.0, 0.0, 1.0))
    for j in range(len(els) - 1):
        for i in range(seg):
            i2 = (i + 1) % seg
            a_, b_, c_, d_ = j * seg + i, j * seg + i2, (j + 1) * seg + i2, (j + 1) * seg + i
            if j % 2 == 0:
                faces += [(a_, b_, d_), (b_, c_, d_)]
            else:
                faces += [(a_, b_, c_), (a_, c_, d_)]
    top = (len(els) - 1) * seg
    for i in range(seg):
        faces.append((top + i, top + (i + 1) % seg, len(verts) - 1))
    o = mesh_object(name, verts, faces)
    displace_noise(o, strength=0.16, scale=0.8, seed=seed)
    return o


def world_tris(objs):
    """World-space vertices / polygons of mesh objects (for the BVH the lichen is draped on)."""
    verts, polys = [], []
    for o in objs:
        realize(o)
        mw = world_matrix(o)
        off = len(verts)
        verts += [mw @ v.co for v in o.data.vertices]
        polys += [tuple(off + i for i in pl.vertices) for pl in o.data.polygons]
    return verts, polys


def crust_mesh(name, bvh, seeds, r, x_lim, nseg=8, kd=None):
    """Crustose lichen: thin irregular lobed patches draped on the stone (ray-cast on to the real,
    displaced surface; rim 0.5 mm proud, centre ~2 mm). seeds: (x, y, z_start, radius).
    kd: KD-tree of the coarse stone vertices - a patch never covers one (a covered vertex bakes
    black and the big faces round it would smear that into a sooty blot)."""
    verts, faces = [], []
    down = Vector((0, 0, -1))
    for (x, y, z0, rad) in seeds:
        if abs(x) + rad * 1.4 > x_lim:
            continue
        loc, nrm, _, _ = bvh.ray_cast(Vector((x, y, z0)), down, 0.6)
        if loc is None or nrm.z < 0.5:
            continue
        if kd is not None:
            near = kd.find_range(loc, rad * 1.5)
            if near:
                # slide away from the nearest vertex once, else give up
                away = (loc - near[0][0])
                away.z = 0.0
                if away.length < 1e-5:
                    continue
                loc2, nrm2, _, _ = bvh.ray_cast(loc + away.normalized() * rad * 1.6 + Vector((0, 0, 0.05)), down, 0.3)
                if loc2 is None or nrm2.z < 0.5 or kd.find_range(loc2, rad * 1.5) or abs(loc2.x) + rad * 1.4 > x_lim:
                    continue
                loc, nrm = loc2, nrm2
        tx = nrm.orthogonal().normalized()
        ty = nrm.cross(tx)
        ph = r.u(0, 6.28)
        ecc, ea = r.u(0.7, 1.0), r.u(0, 3.14)
        lob = [0.72 + 0.28 * sin(3 * (2 * pi * i / nseg) + ph) * r.u(0.3, 1.0) + r.u(-0.14, 0.14) for i in range(nseg)]
        pv, ok = [], True
        for ringf, lift in ((1.0, 0.0005), (0.62, 0.0017)):
            for i in range(nseg):
                ang = 2 * pi * i / nseg
                rr = rad * ringf * lob[i]
                u_, v_ = rr * cos(ang), rr * sin(ang)
                u2 = (u_ * cos(ea) + v_ * sin(ea)) * ecc
                v2 = -u_ * sin(ea) + v_ * cos(ea)
                pnt = loc + tx * (u2 * cos(ea) - v2 * sin(ea)) + ty * (u2 * sin(ea) + v2 * cos(ea))
                h, hn, _, _ = bvh.ray_cast(pnt + nrm * 0.02, -nrm, 0.034)
                if h is None or hn.dot(nrm) < 0.6:
                    ok = False
                    break
                pv.append(h + nrm * lift)
            if not ok:
                break
        if not ok:
            continue
        b0 = len(verts)
        verts += [tuple(v) for v in pv] + [tuple(loc + nrm * 0.0022)]
        for i in range(nseg):
            i2 = (i + 1) % nseg
            faces.append((b0 + i, b0 + i2, b0 + nseg + i2, b0 + nseg + i))
            faces.append((b0 + nseg + i, b0 + nseg + i2, b0 + 2 * nseg))
    if not verts:
        return None
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    uv_box(o)
    smooth(o, 180)
    return o


def trim_to_span(half, snap=()):
    """Clamp every vertex to x in [-half, +half] and z >= 0; vertices of `snap` objects that lie within
    4 mm of an end are put exactly on it, so butted runs meet on one plane."""
    import bpy
    names = {o.name for o in snap}
    for o in list(bpy.context.scene.objects):
        if o.type != 'MESH' or len(o.data.vertices) == 0:
            continue
        mw = np.array(world_matrix(o), dtype=np.float64)
        nv = len(o.data.vertices)
        c = np.empty(nv * 3, dtype=np.float32)
        o.data.vertices.foreach_get("co", c)
        w = c.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
        w2 = w.copy()
        w2[:, 0] = np.clip(w2[:, 0], -half, half)
        w2[:, 2] = np.maximum(w2[:, 2], 0.0)
        if o.name in names:
            w2[:, 0] = np.where(w2[:, 0] > half - 0.004, half, w2[:, 0])
            w2[:, 0] = np.where(w2[:, 0] < -half + 0.004, -half, w2[:, 0])
        if np.abs(w2 - w).max() > 1e-7:
            loc = (w2 - mw[:3, 3]) @ np.linalg.inv(mw[:3, :3]).T
            o.data.vertices.foreach_set("co", loc.astype(np.float32).ravel())
            o.data.update()


# far level (lod 2, seen from 44 m on): the same run out of a few prisms
# ------------------------------------------------------------------------------------------------

FAR_BAL = {"vase": [(0.95, 0.0), (0.42, 0.80), (0.80, 1.0)],          # (half side / R, height fraction)
           "double": [(0.50, 0.0), (0.90, 0.25), (0.45, 0.5), (0.90, 0.75), (0.50, 1.0)],
           "square": [(0.80, 0.0), (0.66, 0.5), (0.80, 1.0)]}


def _far():
    return lod() == 2


def _fprism(name, q, x0, x1, caps=(False, False), skip=()):
    """Prism along X over the counter-clockwise (y, z) polygon q; side i joins q[i] and q[i + 1]."""
    n = len(q)
    verts = [(x0, y, z) for y, z in q] + [(x1, y, z) for y, z in q]
    faces = [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n) if i not in skip]
    if caps[0]:
        faces.append(tuple(range(n - 1, -1, -1)))
    if caps[1]:
        faces.append(tuple(range(n, 2 * n)))
    o = mesh_object(name, verts, faces)
    uv_box(o)
    return o


def _frings(name, xs, prof, n=4, kr=S2, cap=False, skip=(), clip=None):
    """Upright n-sided stacks at every x of xs in one mesh: prof is (radius, z) from the foot up (kr sqrt 2
    and n 4: radius is the half side of a square), a last radius 0 closes to a point, cap closes flat.
    skip leaves sides out (n 4: 0 is +Y, 1 is -X, 2 is -Y, 3 is +X), clip = (lo, hi) clamps x - centre."""
    verts, faces = [], []
    tip = prof[-1][0] <= 0.0
    rings = prof[:-1] if tip else prof
    for cx in xs:
        b = len(verts)
        for r, z in rings:
            for i in range(n):
                an = 2 * pi * (i + 0.5) / n
                dx = r * kr * cos(an)
                if clip:
                    dx = min(max(dx, clip[0]), clip[1])
                verts.append((cx + dx, r * kr * sin(an), z))
        for j in range(len(rings) - 1):
            for i in range(n):
                if i not in skip:
                    i2 = (i + 1) % n
                    faces.append((b + j * n + i, b + j * n + i2, b + (j + 1) * n + i2, b + (j + 1) * n + i))
        t = b + (len(rings) - 1) * n
        if tip:
            verts.append((cx, 0.0, prof[-1][1]))
            faces += [(t + i, t + (i + 1) % n, t + n) for i in range(n)]
        elif cap:
            faces.append(tuple(range(t, t + n)))
    o = mesh_object(name, verts, faces)
    uv_box(o)
    return o


def _ftuft():
    """Moss cushion as a three-sided tent over the footprint of the dome."""
    v = [(1.5 * cos(radians(an)), 1.5 * sin(radians(an)), -0.1) for an in (90, 210, 330)] + [(0.0, 0.0, 0.9)]
    return mesh_object("tuft", v, [(0, 1, 3), (1, 2, 3), (2, 0, 3)])


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------------

def build(p, rng):
    L, H, W = p.length, p.height, p.width
    k = clamp(W / 0.30, 0.85, 1.4)          # moulding scale
    a = W / 2
    wear, moss_amt = p.wear, p.moss
    hb = 0.09 * k                           # top of the base course
    hp = hb + 0.08 * k                      # top of the plinth rail = foot of the balusters
    hc = 0.160 * k                          # coping thickness
    zc = H - hc                             # coping soffit = head of the balusters
    hbal = zc - hp
    R = 0.235 * W                           # baluster belly radius
    S = W + 0.09                            # pier die side
    hsb = S / 2 + 0.045 * k                 # half side of the pier plinth block
    e_cap = S / 2 + 0.070 * k               # half side of the pier cap corona = widest part of a pier
    Wb = W + 0.05                           # base course width
    zd0 = hb + 0.086 * k                    # die springs from the pier base moulding
    zd = H + 0.045 * k                      # top of the pier die
    seat = zd + 0.175 * k                   # top of the pier cap (finial seat)
    f = S / 0.38                            # finial scale
    mis = 0.0012 + 0.0035 * wear            # how far stones sit out of true (m)
    pitch = 2 * R * p.pitch
    lantern_mode = p.cap == "lantern"
    lo2 = _far()                            # far level: the same layout and joints, coarse stones (see FAR_BAL)

    stone_a = mat("stone_smooth", p.stone, scale=1.35)
    stone_b = mat("stone_smooth", tint_shift(p.stone, 1.1, 0.035), scale=1.35)      # paler, cooler bed / newer stone
    stone_r = mat("stone_rough", tint_shift(p.stone, 0.8, 0.045))                   # darker rough plinth course
    mortar = mat("mortar", tint_shift(p.stone, 1.22, -0.01))                        # pale lime mortar (also grey-white lichen)
    moss_m = mat("moss")
    FRESH[0] = mortar                       # broken stone is as pale as the lime: no extra material
    r_tone = rng.fork("tone")

    def pick_stone(r=r_tone, pb=0.3):
        return stone_b if r.chance(pb) else stone_a

    # ---- layout: piers and bays ---------------------------------------------------------------
    left_pier = p.ends in ("both", "left")
    right_pier = p.ends in ("both", "right")

    def clear(lp, rp):
        return L - (e_cap + S / 2) * (int(lp) + int(rp))
    if left_pier and right_pier and clear(True, True) < 2.6 * pitch:
        right_pier = False                              # too short for two piers: keep one
    if (left_pier or right_pier) and clear(left_pier, right_pier) < 2.0 * pitch:
        left_pier = right_pier = False                  # too short for any pier: plain infill
    A = -L / 2 + (e_cap if left_pier else 0.0)
    B = L / 2 - (e_cap if right_pier else 0.0)
    nb = max(1, int(round((B - A) / p.pier_spacing)))
    while nb > 1 and (B - A) / nb < S + 2.6 * pitch:    # never squeeze a bay below ~2 balusters
        nb -= 1

    def lay_out(nb):
        nodes = [A + (B - A) * i / nb for i in range(nb + 1)]
        is_pier = [left_pier] + [True] * (nb - 1) + [right_pier]
        piers = [(x, i) for i, x in enumerate(nodes) if is_pier[i]]
        full_x, halves = [], []             # baluster centres; (x, faces_plus_x) engaged halves
        bays = []
        for i in range(nb):
            pa, pb = is_pier[i], is_pier[i + 1]
            xa = nodes[i] + (S / 2 if pa else 0.0)
            xb = nodes[i + 1] - (S / 2 if pb else 0.0)
            Lb = xb - xa
            if pa and pb:
                n = max(1, int(round(Lb / pitch)))
                xs = [xa + Lb * j / n for j in range(1, n)]
            elif pa or pb:
                n = max(1, int(round(Lb / pitch - 0.5)))
                d = Lb / (n + 0.5)
                xs = [xa + d * j for j in range(1, n + 1)] if pa else [xb - d * j for j in range(n, 0, -1)]
            else:
                n = max(1, int(round(Lb / pitch)))
                xs = [xa + Lb * (j + 0.5) / n for j in range(n)]
            if pa:
                halves.append((xa, True))
            if pb:
                halves.append((xb, False))
            full_x += xs
            bays.append((xa, xb, pa, pb, xs))
        return nodes, is_pier, piers, full_x, halves, bays

    # ---- level of detail from a triangle estimate (the run must stay in budget up to 12 m) -----
    def moss_cap(lod):
        return int(max((340, 300, 240, 190, 150)[lod], L * (60, 50, 40, 32, 26)[lod]))

    def estimate(lod):
        """Triangles of everything but the spalls at this level of detail (calibrated on 12 m builds)."""
        lite_ = 2 if lod >= 4 else 1 if (lod > 0 or nbal >= 12) else 0
        seg_ = (16 if nbal < 12 else 12, 12, 12, 10, 8)[lod]
        if p.baluster == "square":
            bal = (620, 620, 336, 336, 208)[lod]
        else:
            prof = ((VASE, VASE_L, VASE_XL) if p.baluster == "vase" else (DOUBLE, DOUBLE_L, DOUBLE_XL))[lite_]
            bal = 2 * seg_ * (len(prof) + 2) + 84
        pier = (968, 968, 640, 544, 354)[lod] + max(1.0, S / 0.39) * (2718, 2345, 1854, 1221, 790)[lod]
        fs_, fr_ = ((24, 12), (20, 10), (16, 8), (14, 7), (12, 6))[lod]
        ball = 2 * fs_ * (fr_ + 7)
        fin = {"ball": ball, "urn": (1536, 1536, 864, 864, 576)[lod], "flat": 0, "lantern": ball}[p.cap]
        rail = (2190, 2060, 1410, 1290, 1200)[lod] + (85 if p.cramps else 0)
        n_moss = L * (19.0 * moss_amt + 38.0 * clamp((moss_amt - 0.5) * 1.8, 0.0, 0.9) * (moss_amt > 0.55))
        green = min(n_moss, moss_cap(lod)) * (35, 35, 30, 18, 15)[lod] + L * moss_amt * (480, 390, 290, 290, 240)[lod]
        n_lant = min(2, len(piers)) if lantern_mode else 0
        return (bal_eq * bal + len(piers) * (pier + fin) + Lr * rail + green + n_lant * 4700
                + min(int(p.broken), max(0, len(full_x) - 2)) * 1100)

    def spall_cost(h):
        return (n_sp * 320 + bal_eq * min(1.0, wear * 1.025) * 90) * h ** -1.4

    while True:
        nodes, is_pier, piers, full_x, halves, bays = lay_out(nb)
        nbal = len(full_x) + len(halves)
        Lr = max(L - len(piers) * S, 0.5)               # rail length between the piers
        bal_eq = len(full_x) + 0.5 * len(halves)
        n_sp = wear * (4.3 * Lr + 2.7 * len(piers)) + len(piers) * max(2.0 * wear, 1.0 if wear > 0.3 else 0.0)
        if nb <= 1 or estimate(4) + 1500.0 <= TRI_TARGET:
            break
        nb -= 1                             # a pier every metre of a very long run cannot fit: space them out
    lod = 0
    while lod < 4 and estimate(lod) + spall_cost((1.0, 1.25, 1.6, 2.0, 2.4)[lod]) > TRI_TARGET:
        lod += 1
    room = max(TRI_TARGET - estimate(lod), 1500.0)      # what the spalls may cost: coarser facets on long runs
    hs = clamp((spall_cost(1.0) / room) ** (1 / 1.4), 1.0, 2.6)
    SPALL.update(h=hs, n=0, tris=0, cap=room)
    lite = 2 if lod >= 4 else 1 if (lod > 0 or nbal >= 12) else 0
    if p.baluster == "square":
        lite = (0, 0, 2, 2, 3)[lod]         # full arris bevel / eased corners only / plain
    seg = (16 if nbal < 12 else 12, 12, 12, 10, 8)[lod]
    step_c = (0.15, 0.17, 0.3, 0.45, 0.6)[lod]
    fin_seg, fin_rings = ((24, 12), (20, 10), (16, 8), (14, 7), (12, 6))[lod]
    die_edge = (0.1, 0.12, 0.17, 0.25, 0.6)[lod]
    cap_edge = (0.14, 0.16, 0.22, 0.3, 0.6)[lod]
    stones = []                             # (object, kind, chips)
    r_chip = rng.fork("chips")

    def n_chips(length, rate):
        v = wear * rate * length
        return int(v) + (1 if r_chip.chance(v - int(v)) else 0)

    def settle(o, xc, r):
        """Seat a stone slightly out of true about its own centre."""
        for v in o.data.vertices:
            v.co.x -= xc
        o.location = (xc, r.u(-mis, mis), r.u(-mis * 0.5, mis * 0.5))
        o.rotation_euler = (radians(r.u(-1, 1) * mis * 40), 0.0, radians(r.u(-1, 1) * mis * 45))

    # ---- rails, per bay -----------------------------------------------------------------------
    sec_c = [(y, z + zc) for y, z in coping_section(a, k, lod >= 2)]
    sec_p = [(y, z + hb) for y, z in plinth_section(a, k, lod >= 2)]
    mort_c = inset_section(sec_c, 0.0035)
    mort_p = inset_section(sec_p, 0.0035)
    r_rail = rng.fork("rails")
    mortars, cramp_x, base_joints = [], [], []
    end_l = -L / 2 + JOINT / 2              # open ends stop half a joint short; the mortar reaches the end plane
    end_r = L / 2 - JOINT / 2
    for bi, (xa, xb, pa, pb, xs) in enumerate(bays):
        if lo2:                             # the stones of level 0 as plain prisms, neighbours of one stone joined
            x0, x1 = (xa - 0.01 if pa else -L / 2), (xb + 0.01 if pb else L / 2)
            ends = (not pa, not pb)
            yb, yt = a - 0.074 * k, a - 0.045 * k

            def stone_draws(nseed):         # what one stone draws below: noise seeds, settle(), stone_uv()
                for _ in range(nseed):
                    r_rail.randrange(999)
                r_rail.u(-mis, mis), r_rail.u(-mis * 0.5, mis * 0.5), r_rail.u(-1, 1), r_rail.u(-1, 1)
                r_rail.pick((-90, -90, -90, 0, -55, -125)), r_rail.u(-7, 7), r_rail.pick((0, 180))
                r_rail.u(0.85, 1.25), r_rail.u(0, 1), r_rail.u(0, 1)

            def far_run(nm, sec, cuts_, nseed, skip=()):
                ms_ = []
                for _ in range(len(cuts_) - 1):
                    ms_.append(pick_stone())
                    stone_draws(nseed)
                j0 = 0
                for j in range(1, len(ms_) + 1):
                    if j == len(ms_) or ms_[j] is not ms_[j0]:
                        o = _fprism(f"{nm}_{bi}_{j0}", sec, x0 if j0 == 0 else cuts_[j0], x1 if j == len(ms_) else cuts_[j],
                                    (ends[0] and j0 == 0, ends[1] and j == len(ms_)), skip)
                        assign(o, ms_[j0])
                        j0 = j
            cuts = split_run(xa, xb, r_rail.u(0.8, 1.1), xs, r_rail, nmin=2 if xb - xa > 1.0 else 1)
            cramp_x += cuts[1:-1]
            far_run("coping", [(yb, zc), (a, zc + 0.074 * k), (a, zc + 0.128 * k), (0.55 * a, H - 0.002 * k),
                               (-0.55 * a, H - 0.002 * k), (-a, zc + 0.128 * k), (-a, zc + 0.074 * k), (-yb, zc)], cuts, 2)
            mids = [(xs[j] + xs[j + 1]) / 2 for j in range(len(xs) - 1)]
            far_run("plinth", [(a, hb), (a, hb + 0.03 * k), (yt, hp), (-yt, hp), (-a, hb + 0.03 * k), (-a, hb)],
                    split_run(xa, xb, r_rail.u(0.8, 1.1), mids, r_rail, avoid=cuts[1:-1]), 1, skip=(5,))
            o = _fprism(f"base_{bi}", [(Wb / 2, 0.0), (Wb / 2, hb + 0.004), (-Wb / 2, hb + 0.004), (-Wb / 2, 0.0)],
                        nodes[bi] + hsb - 0.004 if pa else -L / 2, nodes[bi + 1] - hsb + 0.004 if pb else L / 2, ends, skip=(3,))
            assign(o, stone_r)
            r_bc = rng.fork(f"base{bi}")
            xa1 = x = (nodes[bi] + hsb + JOINT) if pa else end_l
            xb1 = (nodes[bi + 1] - hsb - JOINT) if pb else end_r
            while x < xb1 - 1e-4:
                ln = r_bc.u(0.42, 0.85)
                if xb1 - (x + ln) < 0.3:
                    ln = xb1 - x
                r_bc.u(-mis, mis), r_bc.u(-0.006, 0.006), r_bc.u(-1, 1), r_bc.randrange(999)
                if x > xa1 + 1e-4:
                    base_joints.append(x)
                x += ln
            continue
        # coping stones, jointed over baluster centres (at least one joint + cramp in a normal bay)
        cuts = split_run(xa, xb, r_rail.u(0.8, 1.1), xs, r_rail, nmin=2 if xb - xa > 1.0 else 1)
        cramp_x += cuts[1:-1]
        for j in range(len(cuts) - 1):
            x0 = cuts[j] + JOINT / 2 if (j > 0 or pa) else end_l
            x1 = cuts[j + 1] - JOINT / 2 if (j < len(cuts) - 2 or pb) else end_r
            m_ = pick_stone()
            o = prism_x(f"coping_{bi}_{j}", sec_c, x0, x1, step=step_c, chamfer=0.004,
                        tuck=(bool(pa) and j == 0, bool(pb) and j == len(cuts) - 2))
            assign(o, m_)
            displace_noise(o, strength=0.0016 + 0.002 * wear, scale=0.33, seed=r_rail.randrange(999))
            displace_noise(o, strength=0.0007, scale=0.05, seed=r_rail.randrange(999))
            settle(o, (x0 + x1) / 2, r_rail)
            ch = []
            for _ in range(n_chips(x1 - x0, 2.2)):
                sy = r_chip.pick((-1, -1, 1))
                ln, dp = r_chip.u(0.065, 0.115), r_chip.u(0.014, 0.022)
                at_end = r_chip.chance(0.4)
                x = r_chip.pick((x0 + ln * 0.3, x1 - ln * 0.3)) if at_end else r_chip.u(x0 + ln, x1 - ln)
                if r_chip.chance(0.8):
                    ch.append(spall(o, (x, sy * (a - 0.007 * k), zc + 0.143 * k), (1, 0, 0), (0, sy, 1), ln, dp, r_chip, m_))
                else:
                    ch.append(spall(o, (x, sy * a, zc + 0.075 * k), (1, 0, 0), (0, sy, -0.75), ln, dp * 0.8, r_chip, m_))
            stone_uv(o, r_rail)
            smooth(o, 38)
            stones.append((o, "coping", ch))
        for x in cuts[1:-1] + ([xa] if pa else []) + ([xb] if pb else []):
            mortars.append(prism_x("mortar_c", mort_c, x - 0.012, x + 0.012, step=1.0, caps=False))
        for x, open_end in ((-L / 2, bi == 0 and not pa), (L / 2, bi == nb - 1 and not pb)):
            if open_end:
                mortars.append(prism_x("mortar_ce", mort_c, min(x, x - np.sign(x) * 0.012), max(x, x - np.sign(x) * 0.012), step=1.0, caps=False))
                mortars.append(prism_x("mortar_pe", mort_p, min(x, x - np.sign(x) * 0.012), max(x, x - np.sign(x) * 0.012), step=1.0, caps=False))

        # plinth rail stones, jointed between balusters
        mids = [(xs[j] + xs[j + 1]) / 2 for j in range(len(xs) - 1)]
        cuts_p = split_run(xa, xb, r_rail.u(0.8, 1.1), mids, r_rail, avoid=cuts[1:-1])
        for j in range(len(cuts_p) - 1):
            x0 = (cuts_p[j] + JOINT / 2) if j > 0 else (xa - 0.02 if pa else end_l)
            x1 = (cuts_p[j + 1] - JOINT / 2) if j < len(cuts_p) - 2 else (xb + 0.02 if pb else end_r)
            m_ = pick_stone()
            o = prism_x(f"plinth_{bi}_{j}", sec_p, x0, x1, step=step_c + 0.03, chamfer=0.003)
            assign(o, m_)
            displace_noise(o, strength=0.0016 + 0.002 * wear, scale=0.3, seed=r_rail.randrange(999))
            settle(o, (x0 + x1) / 2, r_rail)
            ch = []
            for _ in range(n_chips(x1 - x0, 1.0)):
                sy = r_chip.pick((-1, -1, 1))
                ln = r_chip.u(0.05, 0.09)
                ch.append(spall(o, (r_chip.u(x0 + 0.07, x1 - 0.07), sy * (a - 0.036 * k), hp - 0.002), (1, 0, 0), (0, sy, 1),
                                ln, r_chip.u(0.010, 0.015), r_chip, m_, width=0.04))
            stone_uv(o, r_rail)
            smooth(o, 38)
            stones.append((o, "plinth", ch))
        for x in cuts_p[1:-1]:
            mortars.append(prism_x("mortar_p", mort_p, x - 0.012, x + 0.012, step=1.0, caps=False))

        # base course: rough-dressed blocks of random length between the pier plinth blocks
        xa0 = nodes[bi] + (hsb + JOINT if pa else 0.0)
        xb0 = nodes[bi + 1] - (hsb + JOINT if pb else 0.0)
        xa1 = xa0 if pa else end_l
        xb1 = xb0 if pb else end_r
        r_bc = rng.fork(f"base{bi}")
        x = xa1
        j = 0
        while x < xb1 - 1e-4:
            ln = r_bc.u(0.42, 0.85)
            if xb1 - (x + ln) < 0.3:
                ln = xb1 - x
            dy = r_bc.u(-mis, mis) * 1.5
            hz = hb + 0.004
            o = rough_block(f"base_{bi}_{j}", (ln - JOINT * (0.0 if x + ln >= xb1 - 1e-4 and not pb else 1.0), Wb + r_bc.u(-0.006, 0.006), hz),
                            (x + (ln - JOINT * (0.0 if x + ln >= xb1 - 1e-4 and not pb else 1.0)) / 2, dy, hz / 2), (0, 0, r_bc.u(-1, 1) * mis * 30),
                            0.012, 0.003 + 0.003 * wear, r_bc.randrange(999), max_edge=(0.16, 0.18, 0.22, 0.3, 0.4)[lod])
            assign(o, stone_r)
            ch = []
            for _ in range(n_chips(ln, 1.1)):
                sy = r_chip.pick((-1, -1, 1))
                ch.append(spall(o, (x + r_chip.u(0.06, max(0.07, ln - 0.06)), sy * Wb / 2, hb), (1, 0, 0), (0, sy, 0.8),
                                r_chip.u(0.07, 0.12), r_chip.u(0.014, 0.022), r_chip, stone_r))
            drop_faces(o, lambda c, n, hz=hz: n.z < -0.9 and c.z < -hz * 0.4)
            uv_box(o)
            stones.append((o, "base", ch))
            if x > xa1 + 1e-4:
                base_joints.append(x)
            x += ln
            j += 1
        xm0 = xa0 if pa else -L / 2
        xm1 = xb0 if pb else L / 2
        mb = box("mortar_b", (xm1 - xm0, Wb - 0.016, hb - 0.008), loc=((xm0 + xm1) / 2, 0, (hb - 0.008) / 2))
        drop_faces(mb, lambda c, n: abs(n.z) > 0.9)
        mortars.append(mb)
    for m_ in mortars:
        assign(m_, mortar)
        uv_box(m_)

    # ---- balusters ----------------------------------------------------------------------------
    r_bal = rng.fork("balusters")
    hb_inst = hbal + 0.004                   # bedded 2 mm into both rails
    if lo2:
        variants = [r_bal.chance(wear * (0.5 + 0.35 * i)) for i in range(4)][:0]       # the draws, no turned stones
    else:
        variants = [make_baluster(p.baluster, R, hb_inst, seg, lite, r_bal.fork(f"v{i}"),
                                  (1 if r_bal.chance(wear * (0.5 + 0.35 * i)) else 0), stone_b if i == 3 else stone_a)
                    for i in range(4)]
    idx = list(range(len(full_x)))
    r_pick = rng.fork("pick")
    inner = [i for i in idx if all(abs(full_x[i] - c) > pitch * 0.4 for c in cramp_x)] or idx
    broken = []
    for _ in range(min(int(p.broken), max(0, len(idx) - 2))):
        cand = [i for i in inner if i not in broken and all(abs(i - b_) > 1 for b_ in broken)]
        if cand:
            broken.append(r_pick.pick(cand))
    repaired = [i for i in idx if i not in broken and r_pick.chance(p.repairs)]
    if not repaired and p.repairs > 0.03 and len(idx) > 3:
        repaired = [r_pick.pick([i for i in idx if i not in broken])]
    mats_old, mats_new, mats_stub = [], [], []
    for i, x in enumerate(full_x):
        lean = 0.25 + 0.5 * wear
        rot = (r_bal.u(-lean, lean), r_bal.u(-lean, lean), r_bal.pick((0, 90, 180, 270)) + r_bal.u(-2.5, 2.5))
        sc = (r_bal.u(0.985, 1.015), r_bal.u(0.985, 1.015), 1.0)
        M = trs((x + r_bal.u(-0.004, 0.004), r_bal.u(-0.003, 0.003), hp - 0.002), rot, sc)
        if i in broken:
            mats_stub.append((M, x))
        elif i in repaired:
            mats_new.append(M)
        else:
            mats_old.append((M, r_bal.randrange(4)))
    bal_objs = []
    if lo2:                                 # every standing baluster a square prism, one mesh per stone
        zb0, zb1 = hp - 0.002, zc + 0.002
        prof = [(r * R, zb0 + t * (zb1 - zb0)) for r, t in FAR_BAL[p.baluster]]
        which = iter(mats_old)
        pale = [i for i in idx if i not in broken and (i in repaired or next(which)[1] == 3)]
        for nm, m_, xs_ in (("balusters", stone_a, [x for i, x in enumerate(full_x) if i not in broken and i not in pale]),
                            ("balusters_pale", stone_b, [full_x[i] for i in pale])):
            if xs_:
                assign(_frings(nm, xs_, prof), m_)
        for nm, plus in (("balusters_half_a", True), ("balusters_half_b", False)):
            xs_ = [x for (x, pl) in halves if pl == plus]
            if xs_:                         # engaged halves: three faces of a plain taper
                assign(_frings(nm, xs_, [prof[0], (0.5 * R, zb1)], skip=(1,) if plus else (3,),
                               clip=(-0.004, 1.0) if plus else (-1.0, 0.004)), stone_a)
    for vi, v in enumerate(variants):
        ms_ = [M for (M, w) in mats_old if w == vi]
        if ms_:
            bal_objs.append(instances_to_mesh(f"balusters_{vi}", v, ms_, seed=p.seed + vi))
    if mats_new and not lo2:
        fresh = make_baluster(p.baluster, R, hb_inst, seg, lite, r_bal.fork("fresh"), 0, stone_b)
        o = instances_to_mesh("balusters_new", fresh, mats_new, seed=p.seed + 11, keep_base=False)
        stones.append((o, "baluster_new", []))
    if halves and not lo2:
        hv = bisect_half(duplicate(variants[0], "half_baluster"))
        hm = [trs((x, 0, hp - 0.002), (0, 0, 180 if plus else 0)) for (x, plus) in halves]
        bal_objs.append(instances_to_mesh("balusters_half", hv, hm, seed=p.seed + 13, keep_base=False))
    delete(variants)
    for o in bal_objs:
        stones.append((o, "baluster", []))

    # snapped balusters: a ragged raked fracture, the stub left in the rail, the top lying on the paving
    debris = []
    for bi_, (M, bx) in enumerate(mats_stub):
        r_br = rng.fork(f"broken{bi_}")
        kb = R / 0.0705
        hbase, habac = 0.055 * kb, 0.045 * kb
        ht = hb_inst - hbase - habac
        zf = hbase + ht * r_br.u(0.2, 0.42)
        rake, yaw = r_br.u(24, 40), r_br.u(0, 360)
        turned, base, abac, _ = make_baluster(p.baluster, R, hb_inst, 12, True, r_br, 0, stone_a, closed=True, parts_only=True)
        assign(turned, mortar, slot=1)
        top = duplicate(turned, "bal_top")
        fm = (stone_a, mortar)
        boolean(turned, fracture_cutter("frac_a", zf, rake, yaw, 2.2 * R, hb_inst, r_br.fork("f"), fm), op='DIFFERENCE')
        boolean(top, fracture_cutter("frac_b", zf, rake, yaw, 2.2 * R, hb_inst, r_br.fork("f"), fm), op='INTERSECT')
        uv_keep(turned)
        uv_keep(top)
        stub = join([turned, base], f"baluster_stub_{bi_}")
        uv_keep(stub)
        stub.matrix_world = M
        apply_transform(stub)
        stones.append((stub, "baluster", []))
        # the fallen top: lay it on its abacus edge and the broken end, on the paving in front
        piece = join([top, abac], f"baluster_fallen_{bi_}")
        uv_keep(piece)
        me = piece.data
        nv = len(me.vertices)
        c = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", c)
        c = c.reshape(-1, 3).astype(np.float64)
        X, Y, Z = c[:, 2] - hb_inst, c[:, 1].copy(), -c[:, 0] + R          # axis along -X, abacus flat on z = 0
        u_ = -(X + habac)                                                  # distance from the abacus edge towards the break
        far = u_ > 0.02
        th = float(np.min(np.arctan2(Z[far], u_[far]))) if far.any() else 0.0
        u2 = u_ * cos(th) + Z * sin(th)
        Z = -u_ * sin(th) + Z * cos(th)
        X = -u2 - habac
        yw = radians(r_br.u(-35, 35) + r_br.pick((0, 180)))
        Xw, Yw = X * cos(yw) - Y * sin(yw), X * sin(yw) + Y * cos(yw)
        ext = float(max(Xw.max() - Xw.min(), 0.1))
        px = clamp(bx + r_br.u(-0.25, 0.25) - (Xw.max() + Xw.min()) / 2, -L / 2 + 0.02 - Xw.min(), L / 2 - 0.02 - Xw.max()) if ext < L - 0.05 else -(Xw.max() + Xw.min()) / 2
        py = -(Wb / 2 + r_br.u(0.03, 0.12)) - Yw.max()
        out = np.stack([Xw + px, Yw + py, np.maximum(Z - 0.002, 0.0)], axis=1)
        me.vertices.foreach_set("co", out.astype(np.float32).ravel())
        me.update()
        fl = Vector((0.0, 0.0, zf - hb_inst))                               # fracture centre, same transform (shards lie round it)
        fu = -(fl.z + habac)
        fz = -fu * sin(th) + R * cos(th)
        fx_ = -(fu * cos(th) + R * sin(th)) - habac
        fcw = Vector((fx_ * cos(yw) + px, fx_ * sin(yw) + py, fz))
        stones.append((piece, "debris", []))
        for q in range(r_br.pick((2, 3, 4))):                               # shards knocked off in the fall
            s_ = r_br.u(0.012, 0.03)
            sh = sphere(f"shard_{bi_}_{q}", 1.0, segments=6, rings=4, scale=(s_ * r_br.u(1.0, 1.9), s_, s_ * r_br.u(0.45, 0.8)))
            realize(sh)
            displace_noise(sh, strength=s_ * 0.35, scale=s_ * 1.3, seed=r_br.randrange(999))
            sx_ = clamp(fcw.x + r_br.u(-0.18, 0.18), -L / 2 + 0.06, L / 2 - 0.06)
            place(sh, loc=(sx_, fcw.y + r_br.u(-0.12, 0.05), s_ * 0.3), rot=(r_br.u(-12, 12), r_br.u(-12, 12), r_br.u(0, 360)))
            assign(sh, mortar if q % 2 else stone_a)
            uv_box(sh)
            flat(sh)
            paint(sh, grime=0.1)
            debris.append(sh)

    # ---- piers --------------------------------------------------------------------------------
    iron = None
    lantern = None
    lantern_caps = {}
    cap_tops = []                           # (x, kind, ztop, top_z) for the lichen
    snap_objs = []
    for pi_, (x, ni) in enumerate(piers):
        r_p = rng.fork(f"pier{pi_}")
        sides = {"-y", "+y"}
        if ni == 0:
            sides.add("-x")
        if ni == len(nodes) - 1:
            sides.add("+x")
        tw = r_p.u(-1, 1) * mis * 25
        m_pier = pick_stone(r_p, 0.25)
        if lo2:                             # square block, die with the base moulding as a batter, cap, faceted finial
            hbk, h = hb + 0.006, S / 2

            def stone_draws(modes, nplace):  # what a stone of the pier draws below: noise seed, place(), stone_uv()
                r_p.randrange(999)
                for _ in range(nplace):
                    r_p.u(-1, 1)
                r_p.pick(modes), r_p.u(-7, 7), r_p.pick((0, 180)), r_p.u(0.85, 1.25), r_p.u(0, 1), r_p.u(0, 1)
            r_p.randrange(999)
            stone_draws((0, 0, -90), 3)
            stone_draws((-90, -90, 0, -125), 3)
            m_cap = pick_stone(r_p, 0.35)
            stone_draws((0, 0, -90), 2)
            assign(_frings(f"pier_block_{pi_}", [x], [(hsb, 0.0), (hsb, hbk)]), stone_r)
            assign(_frings(f"pier_die_{pi_}", [x], [(hsb - 0.001, hbk - 0.003), (h, zd0), (h, zd + 0.042 * k)]), m_pier)
            cap_kind = p.cap
            if cap_kind == "lantern" and pi_ not in (0, len(piers) - 1):
                cap_kind = "ball"
            cp = [(h - 0.002, 0.040 * k), (h + 0.070 * k, 0.087 * k), (h + 0.070 * k, 0.126 * k)]
            if cap_kind == "flat":
                ztop = 0.139 * k + 0.42 * (h + 0.060 * k)
                cp += [(h + 0.060 * k, 0.139 * k), (0.0, ztop)]
            else:
                ztop = 0.175 * k
                cp += [(0.118 * f, ztop)]
            cap = _frings(f"pier_cap_{pi_}", [x], [(r, z + zd) for r, z in cp], cap=True)
            assign(cap, m_cap)
            if ni in (0, len(nodes) - 1):
                snap_objs.append(cap)
            top_z = zd + ztop
            if cap_kind == "ball":
                Rb = 0.125 * f
                czb = seat + 0.050 * f + sqrt(Rb * Rb - (0.060 * f) ** 2)
                fin = _frings(f"pier_ball_{pi_}", [x], [(0.075 * f, seat - 0.002), (0.93 * Rb, czb - 0.53 * Rb),
                                                        (0.93 * Rb, czb + 0.53 * Rb), (0.0, czb + Rb)], n=6, kr=1.0)
                top_z = czb + Rb
            elif cap_kind == "urn":
                fin = _frings(f"pier_urn_{pi_}", [x], [(r * f, seat + z * f) for r, z in (
                    (0.085, -0.002), (0.04, 0.04), (0.055, 0.10), (0.134, 0.20), (0.136, 0.275), (0.078, 0.325),
                    (0.108, 0.368), (0.03, 0.455), (0.0, 0.492))], n=6, kr=1.0)
                top_z = seat + 0.492 * f
            elif cap_kind == "lantern":
                iron = iron or mat("iron_painted", p.paint)
                hs_ = 0.72
                fin = _frings(f"lantern_ped_{pi_}", [x], [(0.085, seat), (0.034, seat + 0.07), (0.023, seat + 0.60),
                                                          (0.042, seat + 0.705), (0.04, seat + 0.009 + hs_)], kr=1.0, cap=True)
                lantern = lantern or use("lantern_head", width=0.27, height=0.35, paint=p.paint, intensity=p.intensity, seed=p.seed + 1)
                lantern.place((x, 0, seat + 0.008 + hs_), name=f"lantern_{pi_}")
                top_z = seat + hs_
            if cap_kind != "flat":
                assign(fin, iron if cap_kind == "lantern" else pick_stone(r_p, 0.3))
                smooth(fin, 80)
            if cap_kind in ("flat", "ball", "urn"):
                socket(f"pier_top_{pi_}", "attach", (x, 0, top_z), role="prop_top")
            cap_tops.append((x, cap_kind, zd + ztop, top_z))
            continue

        # rough plinth block (same course as the base course under the rail)
        hbk = hb + 0.006
        blk = rough_block(f"pier_block_{pi_}", (2 * hsb, 2 * hsb, hbk), (x, 0, hbk / 2), (0, 0, tw), 0.014,
                          0.003 + 0.003 * wear, r_p.randrange(999), max_edge=(0.14, 0.17, 0.25, 0.3, 0.6)[lod])
        assign(blk, stone_r)
        ch = []
        for _ in range(n_chips(1.0, 1.6)):
            sx, sy = r_chip.pick((-1, 1)), r_chip.pick((-1, -1, 1))
            ch.append(spall(blk, (x + sx * hsb, sy * hsb, hbk), (-sx, sy, 0), (sx, sy, 0.7),
                            r_chip.u(0.07, 0.11), r_chip.u(0.02, 0.032), r_chip, stone_r, width=0.05))
        drop_faces(blk, lambda c, n, hz=hbk: n.z < -0.9 and c.z < -hz * 0.4)
        uv_box(blk)
        stones.append((blk, "pier_block", ch))

        # moulded base: upstand, cavetto, fillet, bead - returns the plinth rail round the pier
        r0 = hsb - 0.006 * k
        bp = [(r0 - 0.02, hb - 0.002), (r0, hb - 0.002), (r0, hb + 0.018 * k), (r0 - 0.004 * k, hb + 0.022 * k)]
        bp += arc(r0 - 0.004 * k, hb + 0.022 * k + 0.033 * k, 0.033 * k, 270, 180, 4)[1:]
        rr_ = r0 - 0.037 * k
        bp += [(rr_, hb + 0.063 * k)]
        bp += arc(rr_, hb + 0.0745 * k, 0.0115 * k, -90, 90, 4)[1:]
        bp += [(S / 2 - 0.02, hb + 0.086 * k)]
        base = lathe(f"pier_base_{pi_}", [(r * S2, z) for r, z in bp], segments=4, rot=(0, 0, 45), smooth_angle=30)
        apply_transform(base)
        orient_out(base)
        assign(base, m_pier)
        bevel(base, 0.005 * k, 1)
        realize(base)
        densify(base, cap_edge)
        displace_noise(base, strength=0.0025, scale=0.3, seed=r_p.randrange(999))
        place(base, loc=(x + r_p.u(-mis, mis), r_p.u(-mis, mis), 0), rot=(0, 0, tw + r_p.u(-1, 1) * mis * 15))
        stone_uv(base, r_p, modes=(0, 0, -90))
        smooth(base, 38)
        stones.append((base, "pier_base", []))

        # die with sunk fielded panels on the free faces, housed 3 cm into base and cap
        z0d, z1d = zd0 - 0.03, zd + 0.03
        die = panelled_die(f"pier_die_{pi_}", S, z1d - z0d, sides, 0.058 * k + 0.03, k, 0.008 * k)
        assign(die, m_pier)
        densify(die, die_edge)
        displace_noise(die, strength=0.0022 + 0.0015 * wear, scale=0.35, seed=r_p.randrange(999))
        dx_, dy_ = x + r_p.u(-mis, mis), r_p.u(-mis, mis)
        place(die, loc=(dx_, dy_, (z0d + z1d) / 2), rot=(0, 0, r_p.u(-1, 1) * mis * 25))
        ch = []
        for _ in range(n_chips(1.0, 1.1)):
            sx, sy = r_chip.pick((-1, 1)), r_chip.pick((-1, -1, 1))
            zz = r_chip.pick((r_chip.u(zd0 + 0.08, zd0 + 0.3), r_chip.u(zd0 + 0.1, zd - 0.12)))
            ch.append(spall(die, (x + sx * S / 2, sy * S / 2, zz), (0, 0, 1), (sx, sy, 0),
                            r_chip.u(0.08, 0.14), r_chip.u(0.015, 0.022), r_chip, m_pier))
        stone_uv(die, r_p, modes=(-90, -90, 0, -125))
        smooth(die, 38)
        stones.append((die, "pier_die", ch))

        # cap: astragal, bed mould, corona, weathered top
        h = S / 2
        cap_kind = p.cap
        if cap_kind == "lantern" and pi_ not in (0, len(piers) - 1):
            cap_kind = "ball"
        cp = [(0.0, -0.004), (h - 0.012, -0.004), (h + 0.004 * k, 0.0), (h + 0.004 * k, 0.007 * k)]
        cp += arc(h + 0.004 * k, 0.019 * k, 0.012 * k, -90, 90, 4)[1:]                    # astragal
        cp += [(h + 0.004 * k, 0.040 * k)]                                                # necking fillet
        cp += arc(h + 0.006 * k, 0.078 * k, 0.038 * k, -90, 0, 4)[1:]                     # ovolo bed mould
        cp += [(h + 0.044 * k, 0.087 * k), (h + 0.070 * k - 0.004, 0.087 * k), (h + 0.070 * k, 0.091 * k),
               (h + 0.070 * k, 0.126 * k), (h + 0.060 * k, 0.139 * k)]                    # corona
        if cap_kind == "flat":
            ztop = 0.139 * k + 0.42 * (h + 0.060 * k)
            cp += [(0.0, ztop)]
        else:
            ztop = 0.175 * k
            r_o, z_o, r_s, z_s = h + 0.060 * k, 0.139 * k, 0.118 * f, ztop - 0.004
            for d_ in (0.085, 0.035):            # rings round the finial seat: keep its contact shadow tight
                if r_s + d_ < r_o - 0.02:
                    cp += [(r_s + d_, z_s + (z_o - z_s) * d_ / (r_o - r_s))]
            cp += [(r_s, z_s), (0.114 * f, ztop), (0.0, ztop)]
        m_cap = pick_stone(r_p, 0.35)
        cap = lathe(f"pier_cap_{pi_}", [(r * S2, z + zd) for r, z in cp], segments=4, rot=(0, 0, 45), smooth_angle=30)
        apply_transform(cap)
        assign(cap, m_cap)
        bevel(cap, 0.006 * k, 1)
        realize(cap)
        densify(cap, cap_edge)
        ringed = False
        if cap_kind in ("ball", "urn"):
            r_ft = (0.093 if cap_kind == "ball" else 0.090) * f
            ringed = seat_ring(cap, zd + ztop, r_ft + 0.006, r_ft - 0.016, n=16 if lod < 2 else 12)
        displace_noise(cap, strength=0.002, scale=0.25, seed=r_p.randrange(999))
        cap_y, cap_rz = r_p.u(-mis, mis), r_p.u(-1, 1) * mis * 20
        place(cap, loc=(x, cap_y, 0), rot=(0, 0, cap_rz))
        ch = []
        for _ in range(max(n_chips(1.0, 2.0), 1 if wear > 0.3 else 0)):
            sx, sy = r_chip.pick((-1, 1)), r_chip.pick((-1, -1, 1))
            if ni == 0 and sx < 0 or ni == len(nodes) - 1 and sx > 0:
                sx = -sx                                 # keep the end plane of the run whole
            e = h + 0.070 * k
            if r_chip.chance(0.65):                      # a corner knocked off the corona
                ch.append(spall(cap, (x + sx * e, sy * e, zd + 0.108 * k), (0, 0, 1), (sx, sy, 0),
                                r_chip.u(0.045, 0.06) * k, r_chip.u(0.02, 0.03), r_chip, m_cap, width=0.07))
            else:                                        # a bite out of the front / back top arris
                ch.append(spall(cap, (x + e * r_chip.u(-0.6, 0.6), sy * (e - 0.004), zd + 0.132 * k), (1, 0, 0), (0, sy, 0.8),
                                r_chip.u(0.07, 0.11), r_chip.u(0.015, 0.022), r_chip, m_cap))
        if ringed:
            open_seat(cap, (x, cap_y, zd + ztop))
        stone_uv(cap, r_p, modes=(0, 0, -90))
        smooth(cap, 38)
        stones.append((cap, "pier_cap", ch))
        if ni in (0, len(nodes) - 1):
            snap_objs.append(cap)
        if cap_kind == "lantern":
            lantern_caps[cap.name] = x
        top_z = zd + ztop

        # finial
        if cap_kind == "ball":
            Rb = 0.125 * f
            fp = [(0.0, -0.004), (0.088 * f, -0.004), (0.093 * f, 0.006 * f), (0.086 * f, 0.016 * f), (0.064 * f, 0.020 * f),
                  (0.052 * f, 0.034 * f), (0.053 * f, 0.044 * f), (0.060 * f, 0.050 * f)]
            t0 = np.arcsin(0.060 * f / Rb)
            czb = 0.050 * f + Rb * cos(t0)
            for i in range(1, fin_rings):
                t = t0 + (pi - t0) * i / fin_rings
                fp.append((Rb * sin(t), czb - Rb * cos(t)))
            t = pi - 0.42 * (pi - t0) / fin_rings
            fp.append((Rb * sin(t), czb - Rb * cos(t)))
            fp.append((0.0, czb + Rb))
            m_fin = pick_stone(r_p, 0.3)
            fin = lathe(f"pier_ball_{pi_}", fp, segments=fin_seg, loc=(x, 0, seat))
            assign(fin, m_fin)
            turned_uv(fin, r_p, squeeze=0.4)
            displace_noise(fin, strength=0.002, scale=0.12, seed=r_p.randrange(999))
            place(fin, rot=(r_p.u(-0.8, 0.8), r_p.u(-0.8, 0.8), r_p.u(0, 360)))
            ch = []
            smooth(fin, 45)
            stones.append((fin, "finial", ch))
            top_z = seat + czb + Rb
        elif cap_kind == "urn":
            up = [(0.0, -0.004), (0.085, -0.004), (0.09, 0.006), (0.085, 0.018), (0.06, 0.024), (0.04, 0.04), (0.034, 0.06),
                  (0.04, 0.075), (0.05, 0.08), (0.04, 0.088), (0.055, 0.10), (0.09, 0.125), (0.118, 0.16), (0.132, 0.20),
                  (0.135, 0.235), (0.128, 0.262), (0.134, 0.268), (0.134, 0.28), (0.118, 0.288), (0.09, 0.30),
                  (0.076, 0.325), (0.08, 0.345), (0.105, 0.36), (0.11, 0.368), (0.105, 0.378), (0.09, 0.384),
                  (0.07, 0.40), (0.045, 0.418), (0.022, 0.43), (0.016, 0.44), (0.026, 0.452), (0.03, 0.466),
                  (0.022, 0.482), (0.0, 0.492)]
            if lod >= 2:
                up = [q for i, q in enumerate(up) if i not in (3, 8, 13, 17, 22, 25, 28, 31)]
            fin = lathe(f"pier_urn_{pi_}", [(r * f, z * f) for r, z in up], segments=(24, 24, 18, 18, 12)[lod])
            assign(fin, pick_stone(r_p, 0.3))
            turned_uv(fin, r_p, squeeze=0.4)
            g0, g1 = 0.098 * f, 0.262 * f

            def gadroon(co, g0=g0, g1=g1):
                if co.z < g0 or co.z > g1:
                    return co
                e = min(smoothstep(g0, g0 + 0.02 * f, co.z), 1.0 - smoothstep(g1 - 0.03 * f, g1, co.z))
                d = 0.09 * e * (1.0 - abs(sin(atan2(co.y, co.x) * 6.0))) ** 2
                return Vector((co.x * (1 - d), co.y * (1 - d), co.z))
            transform_verts(fin, gadroon)
            displace_noise(fin, strength=0.0015, scale=0.1, seed=r_p.randrange(999))
            place(fin, loc=(x, 0, seat), rot=(r_p.u(-0.8, 0.8), r_p.u(-0.8, 0.8), r_p.u(0, 360)))
            smooth(fin, 45)
            stones.append((fin, "finial", []))
            top_z = seat + 0.492 * f
        elif cap_kind == "lantern":
            iron = iron or mat("iron_painted", p.paint)
            hs_ = 0.72                               # cast-iron standard carrying the lantern
            ped = lathe(f"lantern_ped_{pi_}", [(0.0, 0.0), (0.09, 0.0), (0.093, 0.008), (0.085, 0.016), (0.06, 0.022), (0.042, 0.04),
                                              (0.032, 0.08), (0.027, 0.12), (0.035, 0.132), (0.046, 0.146), (0.035, 0.16),
                                              (0.027, 0.172), (0.023, 0.26), (0.02, 0.40), (0.019, 0.52), (0.021, 0.60),
                                              (0.03, 0.612), (0.038, 0.622), (0.03, 0.632), (0.023, 0.644), (0.025, 0.68),
                                              (0.038, 0.70), (0.044, 0.71), (0.04, hs_), (0.0, hs_)],
                        segments=16, loc=(x, 0, seat + 0.009))
            plate = box(f"lantern_plate_{pi_}", (0.2 * f, 0.2 * f, 0.012), loc=(x, 0, seat + 0.005), bevel=0.003, segments=1)
            bt = bolt("ped_bolt", 0.007, 0.006)
            bolts = instances_to_mesh(f"ped_bolts_{pi_}", bt, [trs((x + sx * 0.08 * f, sy * 0.08 * f, seat + 0.011))
                                                              for sx in (-1, 1) for sy in (-1, 1)], keep_base=False)
            assign(plate, iron)
            uv_box(plate)
            for o in (ped, bolts):
                assign(o, iron)
            paint(ped, special=mask_mul(height_ramp(seat, seat + 0.14), noise_mask(0.05, seed=p.seed + pi_, lo=0.3, hi=0.7)))
            lantern = lantern or use("lantern_head", width=0.27, height=0.35, paint=p.paint, intensity=p.intensity, seed=p.seed + 1)
            lantern.place((x, 0, seat + 0.008 + hs_), name=f"lantern_{pi_}")
            top_z = seat + hs_
        if cap_kind in ("flat", "ball", "urn"):
            socket(f"pier_top_{pi_}", "attach", (x, 0, top_z), role="prop_top")
        cap_tops.append((x, cap_kind, zd + ztop, top_z))
        # mortar bed lines of the pier (sides only)
        for zj in (zd0, zd):
            mbp = box("mortar_pier", (S - 0.01, S - 0.01, 0.016), loc=(x, 0, zj))
            drop_faces(mbp, lambda c, n: abs(n.z) > 0.9)
            assign(mbp, mortar)
            uv_box(mbp)

    # ---- iron dog-cramps across the coping joints ---------------------------------------------
    if p.cramps and cramp_x and not lo2:
        r_cr = rng.fork("cramps")
        bar = box("cramp_bar", (0.17, 0.016, 0.012), bevel=0.002, segments=1)
        e1 = box("cramp_e1", (0.03, 0.036, 0.012), loc=(-0.085, 0, 0), bevel=0.002, segments=1)
        e2 = box("cramp_e2", (0.03, 0.036, 0.012), loc=(0.085, 0, 0), bevel=0.002, segments=1)
        cr = join([bar, e1, e2], "cramp")
        cramp_pos = [(x + r_cr.u(-0.006, 0.006), r_cr.u(-0.012, 0.012)) for x in cramp_x]
        cramps = instances_to_mesh("cramps", cr, [trs((x, y, H - 0.0025), (0, 0, r_cr.u(-4, 4))) for (x, y) in cramp_pos], keep_base=False)
        assign(cramps, mat("iron_rusty"))
        uv_grain(cramps, "X")
    else:
        cramp_pos = []

    # ---- moss cushions clustered in the damp foot, at pier bases and in the joints --------------
    max_moss = moss_cap(lod)
    if moss_amt > 0.02:
        r_m = rng.fork("moss")
        tuft = _ftuft() if lo2 else moss_dome("tuft", (7, 7, 6, 6, 5)[lod], 2 if lod < 3 else 1, p.seed)
        mm, mm_a = [], []
        xlim = L / 2 - 0.012

        def cushion(x, y, z, yaw, rx, ry, rz):
            rx = min(rx, 0.11)
            if abs(x) + rx > xlim:
                rx = xlim - abs(x)
                if rx < 0.012:
                    return
            mm.append(trs((x, y, z - 0.001), (r_m.u(-6, 6), r_m.u(-6, 6), yaw), (rx, ry, rz)))
            mm_a.append(rx * ry)

        def foot_y(x, sy):
            near = min([abs(x - px) for px, _ in piers] or [9.0])
            return sy * (hsb if near < hsb else Wb / 2)

        def cluster(x, sy, z, n, big=1.0, ledge=False):
            for _ in range(n):
                xx = x + r_m.u(-0.09, 0.09) * big
                rx = r_m.u(0.022, 0.05) * big
                if ledge:                                    # on the base-course ledge under the plinth rail
                    cushion(xx, sy * (a + r_m.u(0.004, 0.014)), z, r_m.u(-10, 10), rx * 0.8, r_m.u(0.010, 0.016), r_m.u(0.008, 0.016))
                else:
                    y0 = foot_y(xx, sy)
                    cushion(xx, y0 + sy * r_m.u(0.006, 0.02), z, r_m.u(-12, 12), rx * r_m.u(1.0, 1.7), r_m.u(0.026, 0.05) * big, r_m.u(0.024, 0.052) * big)

        def side():
            return 1 if r_m.chance(0.55) else -1            # a little more on the damp back (+Y)

        for px, _ in piers:                                 # re-entrant corners of pier block / base course
            for sx in (-1, 1):
                cx_ = px + sx * (hsb + 0.03)
                if abs(cx_) < L / 2 - 0.06 and r_m.chance(0.35 + 0.65 * moss_amt):
                    cluster(cx_, side(), 0.0, 1 + int(round(4 * moss_amt * r_m.u(0.6, 1.2))), big=1.15)
                    if r_m.chance(moss_amt * 0.7):
                        cluster(cx_ + sx * 0.05, side(), hb, 1 + int(2 * moss_amt), ledge=True)
            if r_m.chance(moss_amt):                         # along the back face of the pier block
                cluster(px + r_m.u(-0.6, 0.6) * hsb, 1, 0.0, 2 + int(2 * moss_amt), big=1.0)
        for xj in base_joints:                              # at the open joints of the base course
            if r_m.chance(0.25 + 0.6 * moss_amt):
                sy = side()
                cluster(xj, sy, 0.0, 1 + int(round(3 * moss_amt * r_m.u(0.5, 1.2))))
                if r_m.chance(0.5 * moss_amt):
                    cluster(xj, sy, hb, 2, ledge=True)
        for _ in range(int(round(moss_amt * L * 5.0))):     # strays
            cluster(r_m.u(-L / 2 + 0.1, L / 2 - 0.1), side(), 0.0, r_m.pick((1, 2, 2, 3)), big=0.85)
        for _ in range(int(moss_amt * L * 1.2)):
            cluster(r_m.u(-L / 2 + 0.1, L / 2 - 0.1), side(), hb, r_m.pick((1, 2, 3)), ledge=True)
        if moss_amt > 0.55:                                 # an unbroken-looking green band along the back foot
            cover = clamp((moss_amt - 0.5) * 1.8, 0.0, 0.9)
            x = -L / 2 + 0.05
            while x < L / 2 - 0.05:
                rx = r_m.u(0.03, 0.06)
                if r_m.chance(cover):
                    cushion(x + rx, foot_y(x + rx, 1) + r_m.u(0.004, 0.018), 0.0, r_m.u(-10, 10), rx * 1.25, r_m.u(0.028, 0.05), r_m.u(0.026, 0.055))
                if r_m.chance(0.85 * cover):
                    cushion(x + rx, foot_y(x + rx, -1) - r_m.u(0.004, 0.016), 0.0, r_m.u(-10, 10), rx * 1.15, r_m.u(0.024, 0.044), r_m.u(0.02, 0.048))
                x += rx * 2 * r_m.u(0.8, 1.15)
        if len(mm) > max_moss:
            keep = sorted(r_m.sample(range(len(mm)), max_moss)) if hasattr(r_m, "sample") else list(range(0, len(mm), 2))[:max_moss]
            mm, mm_a = [mm[i] for i in keep], [mm_a[i] for i in keep]
        if lo2 and mm:                      # the bigger half of the cushions
            mm = [mm[i] for i in sorted(sorted(range(len(mm)), key=lambda i: -mm_a[i])[:(len(mm) + 1) // 2])]
        if mm:
            clumps = instances_to_mesh("moss", tuft, mm, keep_base=False)
            assign(clumps, moss_m)
            uv_box(clumps)
            smooth(clumps, 180)
        else:
            delete(tuft)

    # ---- crustose lichen draped on the weathered tops ------------------------------------------
    if moss_amt > 0.02 and not lo2:         # (the far level leaves the crusts, the cramps and the mortar beds out)
        r_l = rng.fork("lichen")
        targets = [o for (o, kind, _) in stones if kind in ("coping", "pier_cap", "finial")]
        bvh = BVHTree.FromPolygons(*world_tris(targets))
        from mathutils.kdtree import KDTree
        cv, _ = world_tris([o for (o, kind, _) in stones if kind in ("coping", "pier_cap")])
        kd = KDTree(len(cv))
        for i_, v_ in enumerate(cv):
            kd.insert(v_, i_)
        kd.balance()
        seeds_w, seeds_o = [], []                         # grey-white, ochre

        def colony(x, y, z0, big=1.0):
            kind = r_l.weighted([("w", 6), ("o", 4)])
            lst = seeds_w if kind == "w" else seeds_o
            rad = r_l.u(0.014, 0.034) * big
            lst.append((x, y, z0, rad))
            for _ in range(r_l.pick((0, 1, 2, 3))):          # satellites
                ang = r_l.u(0, 2 * pi)
                d_ = rad * r_l.u(1.1, 2.2)
                lst.append((x + d_ * cos(ang), y + d_ * sin(ang), z0, rad * r_l.u(0.35, 0.7)))

        dens = moss_amt * (4.2, 3.6, 2.8, 2.2, 1.8)[lod]
        for (xa, xb, pa, pb, xs) in bays:                  # coping: off the hand-polished centre line
            for _ in range(int(round((xb - xa) * dens))):
                colony(r_l.u(xa + 0.03, xb - 0.03), r_l.pick((-1, 1)) * a * r_l.u(0.3, 0.8), H + 0.08)
        for (px, kind, zt, tz) in cap_tops:
            hh = S / 2 + 0.05 * k
            for _ in range(int(round(moss_amt * (7, 6, 5, 4, 3)[lod]))):
                dx_, dy_ = r_l.u(-hh, hh), r_l.u(-hh, hh)
                if kind != "flat" and sqrt(dx_ * dx_ + dy_ * dy_) < 0.14 * f:
                    continue
                colony(px + dx_, dy_, (zt + 0.03) if kind != "flat" else zt + 0.05, big=0.9)
            if kind in ("ball", "urn"):
                for _ in range(int(round(moss_amt * 3))):
                    ang, d_ = r_l.u(0, 2 * pi), r_l.u(0.0, 0.085 * f)
                    colony(px + d_ * cos(ang), d_ * sin(ang), tz + 0.05, big=0.75)
        lich_m = None if lantern_mode else mat("plaster_rough", "#b9a04a")
        lich_g = mortar if lantern_mode else mat("plaster_rough", "#b3b7a2")      # pale grey-green crust
        if lich_m is None:                  # the lantern variant is at the material limit: one crust colour
            seeds_w += seeds_o
            seeds_o = []
        for nm, sd_, m_ in (("lichen_grey", seeds_w, lich_g), ("lichen_ochre", seeds_o, lich_m)):
            if sd_:
                o = crust_mesh(nm, bvh, sd_, r_l, L / 2 - 0.004, nseg=8 if lod < 2 else 6, kd=kd)
                if o:
                    assign(o, m_)

    # ---- weathering: one story for every stone ------------------------------------------------
    sd = p.seed
    n_patch = noise_mask(0.22, seed=sd + 1, lo=0.42, hi=0.62)
    n_blot = noise_mask(0.09, seed=sd + 2, lo=0.4, hi=0.7)
    n_damp = noise_mask(0.5, seed=sd + 3, lo=0.3, hi=0.7)
    n_big = noise_mask(0.7, seed=sd + 4, lo=0.2, hi=0.8)
    n_str = noise_mask(1.0, seed=sd + 5, lo=0.3, hi=0.7)
    n_ble = noise_mask(0.3, seed=sd + 6, lo=0.38, hi=0.66)
    mg = (0.4 + 0.6 * moss_amt) if moss_amt > 0.01 else 0.0
    cr_xy = np.array(cramp_pos) if cramp_pos else None

    # The stone materials' `special` (moss) tint is a very dark olive and vertex masks smear over the
    # coarse faces of the tops, where it reads as soot - so `special` is used only for the damp green
    # film low on the back / foot; the lichen on the tops is modelled crust patches. `wear` is not painted
    # at all: on these stones it renders near-black (see META).
    def lichen(top_gain=1.0, foot=True):
        def fn(co, n):
            m = np.zeros(len(co))
            if foot:
                north = np.clip(n[:, 1] * 0.8 + 0.2, 0.0, 1.0) * np.clip(1.0 - co[:, 2] / 0.3, 0.0, 1.0) ** 1.3 * (0.22 + 0.3 * n_damp(co, n))
                m = np.maximum(m, north)
            return np.clip(m * mg, 0.0, 0.46)
        return fn

    def dirt(tone, z_over=None, drop=0.5, cx=None, rust=False):
        """Per-stone tone, splash-back at the foot, soot under overhangs, rain curtains below z_over."""
        def fn(co, n):
            z, nz = co[:, 2], n[:, 2]
            g = np.full(len(co), tone)
            g = np.maximum(g, np.clip(1.0 - z / 0.35, 0.0, 1.0) ** 1.5 * (0.25 + 0.45 * n_big(co, n)) * 0.6)
            g = np.maximum(g, np.clip(-nz, 0.0, 1.0) * (0.3 + 0.2 * n_big(co, n)))
            if z_over is not None:
                sco = co * np.array([1.0 / 0.09, 1.0 / 0.09, 1.0 / 0.8])
                fall = np.clip(1.0 - (z_over - z) / drop, 0.0, 1.0) ** 0.7
                st = (0.25 + 0.75 * n_str(sco, n)) * fall * (1.0 - np.abs(nz))
                g = np.maximum(g, st * (0.14 + 0.24 * wear))
            if cx is not None:                      # soot and rust wash under a lantern standard
                d = np.hypot(co[:, 0] - cx, co[:, 1])
                g = np.maximum(g, np.clip(nz, 0.0, 1.0) * (0.2 + 0.3 * np.exp(-(d / 0.17) ** 2)))
            if rust and cr_xy is not None:          # rust weeping from the cramps
                dx = (co[:, None, 0] - cr_xy[None, :, 0]) / 0.11
                dy = (co[:, None, 1] - cr_xy[None, :, 1]) / 0.05
                g = np.maximum(g, np.max(np.exp(-(dx * dx + dy * dy)), axis=1) * 0.3 * np.clip(nz, 0.0, 1.0))
            return np.clip(g, 0.0, 0.7)
        return fn

    for o, kind, ch in stones:
        tone = r_tone.u(0.03, 0.14) + 0.08 * wear
        if kind == "coping":
            paint(o, grime=dirt(tone, H - 0.085 * k, 0.1, rust=True), special=lichen(0.8))
        elif kind == "plinth":
            paint(o, grime=dirt(tone, zc + 0.05, 1.2), special=lichen(1.0))
        elif kind == "base":
            paint(o, grime=dirt(tone * 0.6), special=lichen(1.2))
        elif kind == "baluster":
            paint(o, grime=dirt(tone, zc, 0.3), special=lichen(0.8, foot=False))
        elif kind == "baluster_new":
            paint(o, grime=dirt(0.02), special=lichen(0.3, foot=False))
        elif kind == "debris":
            paint(o, grime=dirt(tone), special=lichen(0.6, foot=False))
        elif kind == "pier_block":
            paint(o, grime=dirt(tone * 0.7), special=lichen(1.2))
        elif kind == "pier_base":
            paint(o, grime=dirt(tone), special=lichen(1.0))
        elif kind == "pier_die":
            paint(o, grime=dirt(tone, zd + 0.02, 0.6), special=lichen(1.0))
        elif kind == "pier_cap":
            paint(o, grime=dirt(tone, cx=lantern_caps.get(o.name)), special=lichen(1.0, foot=False))
        elif kind == "finial":
            paint(o, grime=dirt(tone, seat + 0.2 * f, 0.3), special=lichen(0.9, foot=False))

    # ---- exact span: clamp to +-L/2 and put the end planes exactly on it -------------------------
    trim_to_span(L / 2, snap_objs)

    socket("end_l", "attach", (-L / 2, 0, 0), rot=(0, 0, -90), role="rail_end")
    socket("end_r", "attach", (L / 2, 0, 0), rot=(0, 0, 90), role="rail_end")
