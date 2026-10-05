"""Classical carved stone urn planter. A turned urn (campana krater, shallow tazza or ovoid vase) stands on its
own square sub-base on a stone pedestal: rough plinth course, moulded base, panelled die with paterae, moulded
cap. The bowl is carved the way such urns are, out of one block: a cup of swelling gadroons with rounded ends
under a shoulder bead, a plain bell with laurel festoons in relief (rosette ties, husk drops), an astragal and a
real egg-and-dart ovolo (eggs in shells, darts between) under a fillet and rolled lip, optional reeded scroll
handles ending in volutes or bronze drop rings. It is filled with soil and planted with a clipped box topiary,
trailing flowers or a fern - or left empty with dead leaves. Rain streaks under the rim / cap / sills, grime in
the gadroon valleys, an algae film on up- and north-facing stone and chipped arrises come from the moss / age
params and the seed.
"""
from math import atan2, cos, hypot, pi, radians, sin, sqrt

import numpy as np
from mathutils import Matrix, Vector

from tk import *
import bpy

# tri_budget: all-round carved ornament (egg-and-dart, gadroons, festoons) plus a leaf-card topiary on a hero prop
META = {"name": "Stone urn planter", "class": "prop", "mount": "ground", "focus": 0.56, "tri_budget": 22000}
PARAMS = {
    "height": {"default": 1.15, "min": 0.7, "max": 1.7, "desc": "height from the paving to the urn rim (m)"},
    "width": {"default": 0.56, "min": 0.4, "max": 0.9, "desc": "largest diameter of the urn (m)"},
    "pedestal": {"default": 0.45, "min": 0.0, "max": 0.5, "desc": "fraction of the height taken by the pedestal (small = plinth slab only)"},
    "bowl": {"default": "campana", "choices": ["campana", "tazza", "ovoid"], "desc": "bowl style: bell krater, shallow dish, egg-shaped vase"},
    "gadroons": {"default": 18, "min": 10, "max": 26, "desc": "number of gadroon lobes on the cup"},
    "handles": {"default": "ear", "choices": ["none", "ear", "ring"], "desc": "carved ear handles, bronze drop rings or none"},
    "swags": {"default": True, "desc": "carved festoon swags on the bowl (not on the tazza)"},
    "planting": {"default": "topiary", "choices": ["topiary", "flowers", "fern", "empty"], "desc": "what grows in it"},
    "topiary": {"default": "ball", "choices": ["ball", "cone", "tiers"], "desc": "clipped shape of the topiary"},
    "plant_size": {"default": 1.0, "min": 0.6, "max": 1.4, "desc": "relative size of the planting"},
    "trailing": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "how far the flowers trail over the rim"},
    "foliage": {"default": "#869aa0", "desc": "tint of the foliage (multiplies the leaf green; default deep box green)"},
    "flower": {"default": "#c9364a", "desc": "main flower colour"},
    "flower2": {"default": "#f3e7c6", "desc": "second flower colour"},
    "stone": {"default": "#cfc7b6", "desc": "stone tint"},
    "moss": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "moss and algae on the stone"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: rain streaks, grime, chipped rim, settled blocks"},
}
VARIANTS = {
    "flower_tazza": {"bowl": "tazza", "height": 0.98, "width": 0.82, "pedestal": 0.47, "planting": "flowers", "handles": "none",
                     "stone": "#dccfb4", "gadroons": 24, "moss": 0.2, "age": 0.35, "seed": 4},
    "fern_vase": {"bowl": "ovoid", "height": 1.38, "width": 0.52, "pedestal": 0.4, "planting": "fern", "handles": "ear",
                  "stone": "#aab0ad", "foliage": "#707c68", "gadroons": 14, "moss": 0.6, "age": 0.6, "plant_size": 1.2, "seed": 9},
    "mossy_empty": {"bowl": "campana", "height": 0.74, "width": 0.62, "pedestal": 0.0, "planting": "empty", "handles": "none",
                    "stone": "#b9b5a3", "moss": 1.0, "age": 1.0, "swags": False, "seed": 15},
    "grand_cone": {"bowl": "campana", "height": 1.5, "width": 0.66, "pedestal": 0.48, "planting": "topiary", "topiary": "cone",
                   "handles": "ring", "stone": "#d6b79c", "gadroons": 20, "moss": 0.15, "age": 0.3, "foliage": "#737e6d", "seed": 22},
}

SQ2 = sqrt(2.0)
GRAIN_ROT = 26.6                                # deg: direction of the tooling in the stone_smooth texture

# normalised (radius / R, height / urn height) profiles
STYLES = {
    "campana": dict(
        foot_r=0.60,
        foot=[(0.56, 0.065), (0.585, 0.08), (0.60, 0.098), (0.585, 0.116), (0.52, 0.124), (0.52, 0.136), (0.38, 0.152), (0.27, 0.18),
              (0.215, 0.22), (0.20, 0.245), (0.245, 0.252), (0.275, 0.264), (0.245, 0.277), (0.205, 0.283), (0.23, 0.305)],
        cup=((0.22, 0.295), (0.76, 0.47)),
        upper=[(0.72, 0.458), (0.785, 0.464), (0.805, 0.48), (0.785, 0.496), (0.735, 0.502), (0.735, 0.516), (0.675, 0.56), (0.645, 0.62),
               (0.642, 0.70), (0.67, 0.78), (0.735, 0.845), (0.83, 0.893), (0.85, 0.898), (0.85, 0.908)],
        ovolo=[(0.868, 0.914), (0.925, 0.938), (0.962, 0.966)],
        lip=[(0.962, 0.972), (0.995, 0.978), (1.0, 0.986), (0.995, 0.996), (0.975, 1.0)],
        inner=[(0.905, 1.0), (0.885, 0.99), (0.86, 0.95), (0.80, 0.88), (0.76, 0.82)],
        soil=0.945, swag=(0.80, 0.635),
        handle=dict(B=0.665, anchor=(0.805, 0.48), off=(0.5, 0.85), p1=(2.2, 1.0), rv=0.16), ring_z=0.75),
    "tazza": dict(
        foot_r=0.52,
        foot=[(0.48, 0.065), (0.505, 0.08), (0.52, 0.098), (0.505, 0.116), (0.44, 0.124), (0.44, 0.136), (0.30, 0.16), (0.21, 0.20),
              (0.175, 0.25), (0.20, 0.30), (0.255, 0.35), (0.265, 0.385), (0.22, 0.42), (0.17, 0.45), (0.165, 0.475), (0.215, 0.483),
              (0.24, 0.497), (0.215, 0.51), (0.19, 0.515), (0.21, 0.535)],
        cup=((0.20, 0.525), (0.93, 0.80)),
        upper=[(0.895, 0.786), (0.95, 0.792), (0.968, 0.81), (0.95, 0.828), (0.90, 0.834), (0.90, 0.85), (0.895, 0.862)],
        ovolo=[(0.905, 0.87), (0.95, 0.91), (0.972, 0.955)],
        lip=[(0.972, 0.962), (0.996, 0.968), (1.0, 0.98), (0.996, 0.994), (0.98, 1.0)],
        inner=[(0.915, 1.0), (0.895, 0.985), (0.87, 0.94), (0.80, 0.87), (0.66, 0.79)],
        soil=0.94, swag=None,
        handle=dict(B=0.842, anchor=(0.968, 0.81), off=(0.7, -0.55), p1=(1.3, 0.8), rv=0.12), ring_z=0.845),
    "ovoid": dict(
        foot_r=0.64,
        foot=[(0.60, 0.055), (0.625, 0.068), (0.64, 0.084), (0.625, 0.10), (0.56, 0.107), (0.56, 0.118), (0.40, 0.132), (0.28, 0.158),
              (0.22, 0.195), (0.205, 0.22), (0.25, 0.227), (0.28, 0.239), (0.25, 0.251), (0.21, 0.257), (0.235, 0.278)],
        cup=((0.22, 0.268), (0.95, 0.50)),
        upper=[(0.915, 0.488), (0.975, 0.494), (1.0, 0.512), (0.975, 0.53), (0.93, 0.536), (0.93, 0.55), (0.925, 0.60), (0.88, 0.67),
               (0.79, 0.745), (0.67, 0.81), (0.585, 0.855), (0.555, 0.885), (0.56, 0.905), (0.585, 0.91), (0.585, 0.918)],
        ovolo=[(0.60, 0.923), (0.66, 0.945), (0.70, 0.968)],
        lip=[(0.70, 0.974), (0.735, 0.979), (0.74, 0.987), (0.735, 0.996), (0.72, 1.0)],
        inner=[(0.65, 1.0), (0.63, 0.99), (0.585, 0.95), (0.52, 0.90), (0.50, 0.84)],
        soil=0.955, swag=(0.755, 0.615),
        handle=dict(B=0.87, anchor=(0.88, 0.67), off=(0.85, 0.35), p1=(1.6, 1.4), rv=0.16), ring_z=0.79),
}


# ------------------------------------------------------------------ small local helpers
def _r_at(prof, z):
    """Largest radius of an (r, z) polyline at height z."""
    best = None
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        lo, hi = min(z0, z1), max(z0, z1)
        if lo - 1e-9 <= z <= hi + 1e-9:
            r = r0 if hi - lo < 1e-9 else r0 + (r1 - r0) * (z - z0) / (z1 - z0)
            best = r if best is None else max(best, r)
    if best is None:
        best = prof[0][0] if z < prof[0][1] else prof[-1][0]
    return best


def _cols(loc, x, y, z, s=(1, 1, 1)):
    x, y, z = Vector(x) * s[0], Vector(y) * s[1], Vector(z) * s[2]
    return Matrix(((x.x, y.x, z.x, loc[0]), (x.y, y.y, z.y, loc[1]), (x.z, y.z, z.z, loc[2]), (0, 0, 0, 1)))


def _frame(loc, ydir, zhint=(0, 0, 1), s=(1, 1, 1)):
    """Matrix whose local +Y points along ydir, local +Z as close to zhint as possible."""
    y = Vector(ydir).normalized()
    x = y.cross(Vector(zhint))
    if x.length < 1e-4:
        x = y.cross(Vector((1, 0, 0)))
    x.normalize()
    return _cols(loc, x, y, x.cross(y), s)


def _dome(name, nu=6, nv=2, taper=0.0):
    """Unit half-ellipsoid on z = 0 (x, y in +-0.5, height 1). nv = 1 gives a diamond pyramid (dart)."""
    verts, faces = [], []
    for j in range(nv):
        ph = (j / nv) * pi / 2
        for i in range(nu):
            a = 2 * pi * i / nu
            x, y = 0.5 * cos(a) * cos(ph), 0.5 * sin(a) * cos(ph)
            verts.append((x * (1 + taper * 2 * y), y, sin(ph)))
    verts.append((0, 0, 1))
    for j in range(nv - 1):
        for i in range(nu):
            i1 = (i + 1) % nu
            faces.append((j * nu + i, j * nu + i1, (j + 1) * nu + i1, (j + 1) * nu + i))
    top = (nv - 1) * nu
    for i in range(nu):
        faces.append((top + i, top + (i + 1) % nu, len(verts) - 1))
    return mesh_object(name, verts, faces)


def _boat(name):
    """Carved leaf / husk / petal: pointed oval on z = 0 from the origin along +Y (length 1, width 1) with a raised
    midrib (height 1)."""
    verts = [(0, 0, 0), (-0.5, 0.38, 0), (0.5, 0.38, 0), (0, 1, 0.04), (0, 0.22, 1.0), (0, 0.66, 0.72)]
    faces = [(0, 4, 1), (1, 4, 5), (1, 5, 3), (0, 2, 4), (2, 5, 4), (2, 3, 5)]
    o = mesh_object(name, verts, faces)
    uv_box(o)
    return o


def _grid(name, pts, closed_u=True):
    """Quad surface through pts[i][j] (i around / along, j up) with true arc-length UVs: the rows and columns of
    these grids are deliberately uneven (dense at the carving, pinched at the rain streaks) and surface() spaces
    its UVs evenly per row / column, which smeared the stone texture along the long quads."""
    nu = len(pts) - (0 if closed_u else 1)
    nv = len(pts[0]) - 1
    o = surface(name, lambda u, v: pts[min(len(pts) - 1, int(round(u * nu)))][int(round(v * nv))], nu=nu, nv=nv, closed_u=closed_u)
    P = np.array(pts, dtype=float)
    cu, cv = P.shape[0], P.shape[1]
    me = o.data
    if len(me.vertices) == cu * cv and me.uv_layers:
        Pu = np.concatenate([P, P[:1]], axis=0) if closed_u else P
        U = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pu, axis=0), axis=2).mean(axis=1))])
        V = np.concatenate([np.zeros((Pu.shape[0], 1)), np.cumsum(np.linalg.norm(np.diff(Pu, axis=1), axis=2), axis=1)], axis=1)
        V = 0.5 * (V + V.mean(axis=0)[None, :])            # between per-column and mean length: no stretch, little shear
        uv = me.uv_layers[0].data
        for pl in me.polygons:
            ids = [(vi // cv, vi % cv) for vi in pl.vertices]
            wrap = closed_u and max(i for i, _ in ids) == cu - 1 and min(i for i, _ in ids) == 0 and cu > 2
            for li, (i, j) in zip(pl.loop_indices, ids):
                ii = cu if (wrap and i == 0) else i
                uv[li].uv = (U[ii], V[ii, j])
    return o


def _tint(col, f):
    r, g, b = hex_rgb(col)
    return (r * f, g * f, b * f)


def _streak_fn(centres, amps, lens, sigma, period=None):
    """Thin run-off streaks: fn(x, depth) -> 0..1. x = position across the face (or angle), depth = distance below
    the drip edge; every streak has its own position, strength and length."""
    c, a, l = (np.asarray(v, dtype=np.float64) for v in (centres, amps, lens))

    def f(x, depth):
        d = x[:, None] - c[None, :]
        if period:
            d = (d + period / 2) % period - period / 2
        fall = np.clip(1.0 - depth[:, None] / l[None, :], 0, 1) ** 0.7 * (depth[:, None] >= -0.004)
        return (a[None, :] * np.exp(-(d / sigma) ** 2) * fall).max(axis=1)

    return f


def _leaf_diamond(name, material, fold=0.14):
    """Small leaf card: a diamond folded along the midrib (2 triangles), mapped to the whole 'leaf' tile."""
    T = 0.25
    verts = [(0, 0, 0), (0.5, 0.48, fold), (0, 1, 0), (-0.5, 0.48, fold)]
    o = mesh_object(name, verts, [(0, 1, 2), (0, 2, 3)])
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    for lp in me.loops:
        c = me.vertices[lp.vertex_index].co
        uvl.data[lp.index].uv = ((c.x + 0.5) * T, c.y * T)
    assign(o, material)
    uv_keep(o)
    return o


def _scatter(obj, rng, per_m2, keep=lambda n: True):
    """Random points on the faces of obj (area weighted): [(co, face normal)], plus the sampled area."""
    realize(obj)
    M = world_matrix(obj)
    R3 = M.to_3x3()
    vs = [M @ v.co for v in obj.data.vertices]
    polys = []
    for pl in obj.data.polygons:
        n = (R3 @ pl.normal).normalized()
        if keep(n):
            polys.append(([vs[i] for i in pl.vertices], n, pl.area))
    cum = np.cumsum([a for _, _, a in polys])
    out = []
    for _ in range(int(cum[-1] * per_m2)):
        q, n, _a = polys[min(len(polys) - 1, int(np.searchsorted(cum, rng.u(0, cum[-1]))))]
        u, v = rng.u(0, 1), rng.u(0, 1)
        if len(q) == 4:
            co = (q[0] * (1 - u) + q[1] * u) * (1 - v) + (q[3] * (1 - u) + q[2] * u) * v
        else:
            if u + v > 1:
                u, v = 1 - u, 1 - v
            co = q[0] + (q[1] - q[0]) * u + (q[2] - q[0]) * v
        out.append((co, n))
    return out, float(cum[-1])


def _soil_z(zs, ro, rr):
    return zs + float(np.interp(rr / ro, [0.0, 0.55, 0.9, 1.03], [0.012, 0.009, 0.0, -0.006]))


def _moss_mat():
    return mat("moss", grime=0.2)


def _leaf_base(name, material, tris=4, fold=0.16, droop=0.12, two_sided=False):
    """Pointed blade from the origin along +Y (length 1, width 1), V-folded on the midrib, mapped to the whole
    0.25 m 'leaf' texture tile. tris = 4 (broad leaf) or 2 (pinna)."""
    T = 0.25
    if tris == 4:
        verts = [(0, 0, 0), (0, 0.5, -fold * 0.4), (0, 1, -droop), (-0.5, 0.42, fold), (0.5, 0.42, fold)]
        faces = [(0, 1, 3), (3, 1, 2), (0, 4, 1), (1, 4, 2)]
    else:
        verts = [(0, 0, 0), (0, 1, -droop), (-0.5, 0.38, fold), (0.5, 0.38, fold)]
        faces = [(0, 1, 2), (0, 3, 1)]
    if two_sided:                               # own vertices for the underside (identical faces get validated away)
        nv = len(verts)
        faces = faces + [tuple(nv + i for i in reversed(f)) for f in faces]
        verts = verts + [(x, y, z - 0.0005) for x, y, z in verts]
    o = mesh_object(name, verts, faces)
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    for lp in me.loops:
        c = me.vertices[lp.vertex_index].co
        uvl.data[lp.index].uv = ((c.x + 0.5) * T, c.y * T)
    assign(o, material)
    uv_keep(o)
    return o


def _blossom_base(name, petal_mat, eye_mat, petals=5):
    """Five / six round-tipped petals, slightly cupped, with a raised eye in a second colour."""
    verts = [(0, 0, -0.06)]
    w = 2 * pi / petals
    for i in range(petals):
        a = w * i
        verts.append((0.2 * cos(a - w / 2), 0.2 * sin(a - w / 2), -0.03))
        verts.append((0.5 * cos(a - w * 0.21), 0.5 * sin(a - w * 0.21), 0.07))
        verts.append((0.5 * cos(a + w * 0.21), 0.5 * sin(a + w * 0.21), 0.07))
    n = 3 * petals
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    e0 = len(verts)
    verts.append((0, 0, 0.05))
    for i in range(petals):
        verts.append((0.12 * cos(w * i), 0.12 * sin(w * i), 0.0))
    faces += [(e0, e0 + 1 + i, e0 + 1 + (i + 1) % petals) for i in range(petals)]
    o = mesh_object(name, verts, faces)
    assign(o, petal_mat)
    assign_faces(o, eye_mat, lambda c, nrm: c.z > 0.005 and hypot(c.x, c.y) < 0.1)
    return o


def _surface_points(obj):
    """World-space (co, normal) of the realised vertices of obj."""
    realize(obj)
    M = world_matrix(obj)
    R3 = M.to_3x3()
    return [(M @ v.co, (R3 @ v.normal).normalized()) for v in obj.data.vertices]


def _sq_lathe(name, pts):
    """Square moulding: lathe with 4 segments turned 45 deg; pts are (half-side, z)."""
    o = lathe(name, [(r * SQ2, z) for r, z in pts], segments=4, rot=(0, 0, 45), smooth_angle=25)
    bevel(o, 0.004, segments=2)
    return o


def _perim(half, t):
    """Point on a square outline of half-side `half`, t in 0..1 (starts at the front-left corner, CCW)."""
    t = (t % 1.0) * 4
    k, f = int(t), t - int(t)
    c = [(-half, -half), (half, -half), (half, half), (-half, half)]
    (x0, y0), (x1, y1) = c[k], c[(k + 1) % 4]
    return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f


# ------------------------------------------------------------------ planting
def _sprigs(name, pts, rng, material, count, size, out=0.004, spread=0.7):
    base = _leaf_base(name + "_b", material, two_sided=False)
    ms = []
    for _ in range(count):
        co, n = pts[min(len(pts) - 1, int(rng.u(0, len(pts))))]
        v = Vector(rng.vec(1.0))
        tg = v - n * v.dot(n)
        if tg.length < 1e-4:
            tg = n.orthogonal()
        d = (tg.normalized() + n * spread).normalized()
        l = size * rng.u(0.75, 1.3)
        ms.append(_frame(co + n * out + Vector(rng.vec(size * 0.4)), d, n, (l * 0.62, l, l)))
    return instances_to_mesh(name, base, ms, uv_jitter=False, keep_base=False)


def _ball(name, r, zc, segments, rings, squash):
    """Lathed sphere (the toolkit sphere() does not keep a stable face order between runs, which made the
    scattered leaves differ from build to build)."""
    prof = [(r * sin(pi * i / rings), zc - squash * r * cos(pi * i / rings)) for i in range(rings + 1)]
    prof[0], prof[-1] = (0.0, prof[0][1]), (0.0, prof[-1][1])
    return lathe(name, prof, segments=segments)


def _topiary(p, rng, zs, ro, S, q=1.0):
    hedge = mat("hedge", _tint(p.foliage, 0.75))
    leafm = mat("leaf", p.foliage)
    shells = []
    kind = p.topiary
    if kind == "ball":
        rb = 0.235 * S
        zc = zs + 0.11 * S + rb
        shells.append(_ball("topiary_ball", rb, zc, 22, 12, 0.94))
        ztrunk = zc
    elif kind == "tiers":
        r1, r2 = 0.2 * S, 0.135 * S
        z1 = zs + 0.09 * S + r1
        z2 = z1 + r1 + 0.07 * S + r2
        shells.append(_ball("topiary_ball", r1, z1, 18, 10, 0.9))
        shells.append(_ball("topiary_top", r2, z2, 14, 8, 1.0))
        ztrunk = z2
    else:
        rb, hc = 0.2 * S, 0.78 * S
        zb = zs + 0.07 * S
        prof = [(0.0, zb), (rb * 0.8, zb), (rb, zb + 0.05 * S), (rb * 0.97, zb + hc * 0.12), (rb * 0.88, zb + hc * 0.26), (rb * 0.75, zb + hc * 0.4),
                (rb * 0.6, zb + hc * 0.55), (rb * 0.44, zb + hc * 0.7), (rb * 0.28, zb + hc * 0.84), (rb * 0.13, zb + hc * 0.95), (0.0, zb + hc)]
        shells.append(lathe("topiary_cone", prof, segments=18))
        ztrunk = zb + 0.1
    # clipped surface: lumpy shear marks, then small leaf cards lying on the facets
    samples, area = [], 0.0
    rs = rng.fork("sprigs")
    for k, s in enumerate(shells):
        realize(s)
        displace_noise(s, strength=0.026 * S, scale=0.075, seed=p.seed + 31 + k)
        displace_noise(s, strength=0.009 * S, scale=0.032, seed=p.seed + 37 + k)
        assign(s, hedge)
        smooth(s, 32)
    dens = 2300.0
    tot = 0.0
    for s in shells:
        realize(s)
        tot += sum(pl.area for pl in s.data.polygons)
    grow = max(1.0, sqrt(tot * dens / (1450.0 * q)))
    for s in shells:
        got, a = _scatter(s, rs, dens / (grow * grow), keep=lambda n: n.z > -0.8)
        samples += got
    leaf = _leaf_diamond("topiary_leaf_b", leafm)
    stem = mesh_object("topiary_stem_b", [(0.5, 0, 0), (-0.25, 0, 0.43), (-0.25, 0, -0.43), (0.5, 1, 0), (-0.25, 1, 0.43), (-0.25, 1, -0.43)],
                       [(0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)])
    assign(stem, leafm)
    ms, sm = [], []
    lsz = 0.021 * (0.75 + 0.25 * S) * grow
    for co, n in samples:
        v = Vector(rs.vec(1.0))
        tg = v - n * v.dot(n)
        if tg.length < 1e-4:
            tg = n.orthogonal()
        d = (tg.normalized() + n * rs.u(0.08, 0.5)).normalized()
        l = lsz * rs.u(0.75, 1.25)
        ms.append(_frame(co + n * 0.004, d, n, (l * 0.62, l, l)))
    # a few shoots of new growth the shears missed
    for i in range(int(7 + 4 * S)):
        co, n = samples[int(rs.u(0, len(samples) - 1))]
        ax = (n + Vector(rs.vec(0.45))).normalized()
        L = rs.u(0.035, 0.065) * S
        e1 = ax.orthogonal().normalized()
        e2 = ax.cross(e1)
        sm.append(_frame(co - ax * 0.01, ax, e1, (0.0035, L + 0.01, 0.0035)))
        for j in range(1, 6):
            ps = j * 2.4 + i
            d = ((e1 * cos(ps) + e2 * sin(ps)) * 0.85 + ax * 0.6).normalized()
            l = lsz * rs.u(1.0, 1.3) * (1.1 - 0.08 * j)
            ms.append(_frame(co + ax * (L * j / 5), d, ax, (l * 0.6, l, l)))
    instances_to_mesh("topiary_leaves", leaf, ms, uv_jitter=False, keep_base=False)
    instances_to_mesh("topiary_shoots", stem, sm, keep_base=False)
    # trunk, a little crooked
    r = rng.fork("trunk")
    path = [(0, 0, zs - 0.02)]
    n = 5
    for i in range(1, n + 1):
        t = i / n
        path.append((r.u(-1, 1) * 0.012 * S * (1 - t), r.u(-1, 1) * 0.012 * S * (1 - t), zs - 0.02 + (ztrunk - zs + 0.02) * t))
    tr = sweep("topiary_trunk", path, 0.02 * S, sides=7, scale_fn=lambda t: 1.25 - 0.5 * t, smooth_path=False)
    assign(tr, mat("bark"))
    # clippings lying flat on the soil
    _fallen("clippings", rng.fork("cover"), leafm, zs, ro, int(14 + 8 * S), 0.022)


def _fallen(name, rng, material, zs, ro, count, size):
    base = _leaf_diamond(name + "_b", material, fold=0.05)
    ms = []
    for _ in range(count):
        a = rng.u(0, 2 * pi)
        rr = ro * sqrt(rng.u(0.1, 0.95)) * 0.95
        b = rng.u(0, 2 * pi)
        l = size * rng.u(0.75, 1.3)
        ms.append(_frame((rr * cos(a), rr * sin(a), _soil_z(zs, ro, rr) + 0.0035), (cos(b), sin(b), 0.0),
                         (rng.u(-0.06, 0.06), rng.u(-0.06, 0.06), 1), (l * 0.62, l, l)))
    return instances_to_mesh(name, base, ms, uv_jitter=False, keep_base=False)


def _flowers(p, rng, zs, ro, S, R_rim, z_rim, outer, hu, dq=1.0):
    leafm = mat("leaf", p.foliage)
    f1, f2 = mat("paper", p.flower), mat("paper", p.flower2)
    hm = 0.2 * S
    prof = [(ro * 1.0, zs - 0.012), (min(ro * 1.14, R_rim * 1.02), zs + 0.22 * hm), (ro * 1.02, zs + 0.6 * hm), (ro * 0.7, zs + 0.9 * hm),
            (ro * 0.35, zs + 1.02 * hm), (0.0, zs + 1.05 * hm)]
    mound = lathe("flower_mound", prof, segments=18)
    displace_noise(mound, strength=0.022 * S, scale=0.09, seed=p.seed + 41)
    assign(mound, mat("hedge", p.foliage))
    smooth(mound, 70)
    pts = [q for q in _surface_points(mound) if q[0].z > zs + 0.01]
    _sprigs("flower_leaves", pts, rng.fork("leaves"), leafm, int((170 + 110 * S) * dq), 0.05 * (0.75 + 0.25 * S) / sqrt(dq), out=0.004, spread=0.6)
    bl1 = _blossom_base("blossom1", f1, f2)
    bl2 = _blossom_base("blossom2", f2, f1, petals=6)
    m1, m2 = [], []
    r = rng.fork("bloom")
    thin = dq ** 0.6                              # fewer but larger blooms when the triangle budget is tight
    nb = int((30 + 26 * S) * thin)
    for i in range(nb):
        co, n = pts[min(len(pts) - 1, int(r.u(0, len(pts))))]
        zax = (n + Vector(r.vec(0.45)) + Vector((0, 0, 0.25))).normalized()
        xax = zax.cross(Vector((0.3, 0.9, 0.1))).normalized()
        d = 0.066 * (0.8 + 0.2 * S) * r.u(0.75, 1.25) / sqrt(thin)
        M = _cols(co + n * r.u(0.03, 0.05) * S, xax, zax.cross(xax), zax, (d, d, d))
        (m1 if r.chance(0.68) else m2).append(M)
    # trailing stems over the lip
    lf, lm = _leaf_base("trail_leaf_b", leafm, tris=2, fold=0.12, two_sided=True), []
    stems = []
    r = rng.fork("trail")
    ns = int(round(min(15, 7 + 6 * S) * min(1.0, p.trailing * 2.5) * thin))
    for k in range(ns):
        a = 2 * pi * (k + r.u(-0.35, 0.35)) / max(ns, 1)
        ca, sa = cos(a), sin(a)
        L = max(0.06, p.trailing * r.u(0.35, 1.0) * 0.62 * hu)
        path = [(ro * 0.72, zs + 0.5 * hm), (ro * 1.0, z_rim + 0.035), (R_rim + 0.02, z_rim + 0.012)]
        m = clamp(int(L / 0.045), 3, 9)
        lgrow = sqrt(max(1.0, L / m / 0.045))
        drift, wob = r.u(-0.5, 0.5), r.u(0, 6.28)
        for j in range(1, m + 1):
            z = z_rim + 0.012 - L * j / m
            if z < 0.03:
                break
            rr = max(R_rim + 0.02 - 0.012 * j / m, _r_at(outer, z) + 0.014) + 0.008 * sin(wob + j * 1.3)
            path.append((rr, z))
        pts3 = []
        for j, (rr, z) in enumerate(path):
            aa = a + drift * 0.035 * max(0, j - 2)
            pts3.append((rr * cos(aa), rr * sin(aa), z))
        st = sweep(f"trail{k}", pts3, 0.0034, sides=3, scale_fn=lambda t: 1.0 - 0.6 * t, smooth_path=False)
        assign(st, leafm)
        stems.append(st)
        for j in range(2, len(pts3)):
            q = Vector(pts3[j])
            for side in (-1, 1):
                if r.chance(1.0 - 0.8 * dq):
                    continue
                d = Vector((ca * 0.7 - side * sa * 0.9, sa * 0.7 + side * ca * 0.9, r.u(-0.7, 0.1)))
                l = 0.05 * r.u(0.7, 1.2) * lgrow
                lm.append(_frame(q, d, (ca, sa, 0.4), (l * 0.62, l, l)))
            if r.chance(0.26 * thin) or j == len(pts3) - 1:
                zax = Vector((ca + r.u(-0.4, 0.4), sa + r.u(-0.4, 0.4), r.u(-0.1, 0.5))).normalized()
                xax = zax.cross(Vector((0, 0, 1))).normalized()
                d = 0.058 * r.u(0.75, 1.2)
                M = _cols(q + zax * 0.02, xax, zax.cross(xax), zax, (d, d, d))
                (m1 if r.chance(0.68) else m2).append(M)
    if stems:
        join(stems, "flower_trails")
    if lm:
        instances_to_mesh("trail_leaves", lf, lm, uv_jitter=False, keep_base=False)
    else:
        delete([lf])
    instances_to_mesh("blossoms_a", bl1, m1, keep_base=False)
    if m2:
        instances_to_mesh("blossoms_b", bl2, m2, keep_base=False)
    else:
        delete([bl2])


def _fern(p, rng, zs, ro, S, R_rim, z_rim, q=1.0):
    leafm = mat("leaf", p.foliage)
    pin = _leaf_base("pinna_b", leafm, tris=2, fold=0.1, droop=0.18, two_sided=True)
    pm, stems = [], []
    n = max(9, int((12 + 4 * S) * q))
    for k in range(n):
        r = rng.fork(f"frond{k}")
        young = k % 4 == 3
        az = 2 * pi * (k + r.u(-0.3, 0.3)) / n
        L = S * r.u(0.44, 0.6) * (0.55 if young else 1.0)
        th0 = radians(r.u(80, 88) if young else r.u(58, 78))
        th1 = radians(r.u(15, 45) if young else r.u(-62, -25))
        ns = 9 if young else 15
        curl = r.u(-0.35, 0.35)
        rho, zeta = r.u(0.01, 0.04) * S, zs + 0.005
        pts = []
        for i in range(ns + 1):
            t = i / ns
            if rho > ro * 0.8 and rho < R_rim * 1.1:
                zeta = max(zeta, z_rim + 0.012)
            aa = az + curl * t * t
            pts.append(Vector((rho * cos(aa), rho * sin(aa), zeta)))
            th = th0 + (th1 - th0) * t ** 1.25
            rho += cos(th) * L / ns
            zeta = max(zeta + sin(th) * L / ns, 0.03)
        st = sweep(f"rachis{k}", [tuple(q) for q in pts[:-1:2]] + [tuple(pts[-1])], 0.0042 * S, sides=3, scale_fn=lambda t: 1.0 - 0.75 * t,
                   smooth_path=False)
        assign(st, leafm)
        stems.append(st)
        wmax = 0.095 * S * r.u(0.85, 1.15) * (0.6 if young else 1.0)
        ds = L / ns
        for i in range(2, ns + 1):
            t = i / ns
            w = wmax * (sqrt(t / 0.3) if t < 0.3 else (1 - (t - 0.3) / 0.7) ** 0.8 + 0.06)
            T = (pts[min(i + 1, ns)] - pts[i - 1]).normalized()
            Sd = T.cross(Vector((0, 0, 1)))
            if Sd.length < 1e-3:
                Sd = Vector((-sin(az), cos(az), 0))
            Sd.normalize()
            N = Sd.cross(T)
            if N.z < 0:
                N = -N
            for side in (-1, 1):
                d = Sd * side + T * 0.42 - N * r.u(0.1, 0.35)
                pm.append(_frame(pts[i], d, N, (ds * 1.25, w * r.u(0.88, 1.08), w)))
    join(stems, "fern_stems")
    instances_to_mesh("fern_pinnae", pin, pm, uv_jitter=False, keep_base=False)
    # brown crown at the centre
    cr = _dome("fern_crown", 7, 2)
    transform_verts(cr, lambda c: Vector((c.x * 0.11 * S, c.y * 0.1 * S, zs + c.z * 0.03 * S)))
    assign(cr, _moss_mat())


def _empty(p, rng, zs, ro, pb):
    dead = mat("leaf", "#8a5b2c")
    base = _leaf_base("dead_leaf_b", dead, fold=0.1, droop=0.05)
    ms = []
    r = rng.fork("dead")
    for _ in range(16):
        a, rr = r.u(0, 2 * pi), ro * sqrt(r.u(0.02, 0.85))
        l = r.u(0.05, 0.085)
        b = r.u(0, 6.28)
        ms.append(_frame((rr * cos(a), rr * sin(a), _soil_z(zs, ro, rr) + r.u(0.004, 0.008)), (cos(b), sin(b), r.u(-0.03, 0.03)),
                         (r.u(-0.08, 0.08), r.u(-0.08, 0.08), 1), (l * 0.62, l, l)))
    for _ in range(9):                          # blown against the foot of the plinth
        t = r.u(0, 1)
        x, y = _perim(pb + r.u(0.03, 0.16), t)
        l = r.u(0.05, 0.085)
        b = r.u(0, 6.28)
        ms.append(_frame((x, y, r.u(0.004, 0.007)), (cos(b), sin(b), r.u(0.0, 0.04)), (r.u(-0.08, 0.08), r.u(-0.08, 0.08), 1), (l * 0.62, l, l)))
    instances_to_mesh("dead_leaves", base, ms, uv_jitter=False, keep_base=False)
    # a self-seeded weed
    wl = _leaf_base("weed_b", mat("leaf", p.foliage), two_sided=True)
    wm = []
    a0, rr = r.u(0, 6.28), ro * 0.45
    for i in range(7):
        a = a0 + i * 2.4
        l = r.u(0.06, 0.1)
        wm.append(_frame((rr * cos(a0), rr * sin(a0), zs + 0.004), (cos(a), sin(a), r.u(0.35, 1.2)), (0, 0, 1), (l * 0.45, l, l)))
    instances_to_mesh("weed", wl, wm, uv_jitter=False, keep_base=False)


# ------------------------------------------------------------------ carved parts
def _streak_cols(rng, n, count, lo=2):
    """Pick `count` column indices out of n, at least 3 apart (their neighbours get pulled in to make the streak thin)."""
    out = []
    for _ in range(count * 6):
        c = int(rng.u(lo, n - lo))
        if all(min(abs(c - o), n - abs(c - o)) >= 3 for o in out):
            out.append(c)
        if len(out) >= count:
            break
    return out


def _sq_shell(name, half, z0, z1, rot_deg, rng, per_face=5, cham=0.006, step=0.03, row=0.05, thin=0.007):
    """The four dressed faces of a square block as one quad grid with eased corners. Some columns are pinched
    together so that painted run-off streaks come out thin. Returns (object, streak positions per face)."""
    n = min(16, max(4, int(round(2 * (half - cham) / step))))
    rows = min(6, max(2, int(round((z1 - z0) / row))))
    thin = min(thin, 0.5 * (half - cham) / n)
    cr, sr = cos(radians(rot_deg)), sin(radians(rot_deg))
    pts, streaks = [], []
    for f in range(4):
        ca, sa = cos(f * pi / 2), sin(f * pi / 2)
        qs = [-(half - cham) + 2 * (half - cham) * i / n for i in range(n + 1)]
        sc = _streak_cols(rng, n + 1, per_face) if n >= 8 else []
        for c in sc:
            qs[c - 1], qs[c + 1] = qs[c] - thin, qs[c] + thin
        streaks.append([qs[c] for c in sc])
        for q in qs:
            x, y = q * ca + half * sa, q * sa - half * ca
            x, y = x * cr - y * sr, x * sr + y * cr
            pts.append([(x, y, z0 + (z1 - z0) * j / rows) for j in range(rows + 1)])
    return _grid(name, pts), streaks


def _egg_and_dart(name, bed, ne, pitch, L, hE):
    """Egg-and-dart enrichment of an ovolo: per unit one egg (pointed downwards) lying in a raised U-shaped shell
    and a thin dart between the shells. bed(t) -> ((r, z), (nr, nz)) of the recessed ground, t = 0 bottom .. 1 top."""
    aS, aT, tC = 0.31, 0.46, 0.54
    unit, faces = [], []

    def ell(th, fs, ft):
        c, s = cos(th), sin(th)
        return aS * fs * c * (1.0 - 0.24 * max(0.0, -s)), tC + aT * ft * s

    rings = ((1.0, 0.0), (0.86, 0.5 * hE), (0.52, 0.87 * hE)) if ne <= 48 else ((1.0, 0.0), (0.66, 0.78 * hE))
    for f, h in rings:
        for i in range(8):
            s, t = ell(2 * pi * (i + 0.5) / 8, f, f)
            unit.append((s, t, h))
    unit.append((0.0, tC + 0.02, hE))
    top = 8 * (len(rings) - 1)
    for i in range(8):
        i1 = (i + 1) % 8
        faces += [(j * 8 + i, j * 8 + i1, j * 8 + 8 + i1, j * 8 + 8 + i) for j in range(len(rings) - 1)]
        faces.append((top + i, top + i1, top + 8))
    b0 = len(unit)
    ns = 7
    for i in range(ns):
        th = radians(152 + (388 - 152) * i / (ns - 1))
        end = 0.45 if i in (0, ns - 1) else 1.0
        for fs, ft, h in ((1.10, 1.05, 0.0), (1.33, 1.12, 0.58 * hE * end), (1.58, 1.2, 0.0)):
            s, t = ell(th, fs, ft)
            unit.append((s, t, h))
    for i in range(ns - 1):
        for j in range(2):
            a = b0 + i * 3 + j
            faces.append((a, a + 3, a + 4, a + 1))
    d0 = len(unit)
    unit += [(0.5, 0.57, 0.12 * hE), (0.5, 0.21, 0.62 * hE), (0.5, 0.0, 0.1 * hE), (0.5 - 0.085, 0.23, 0.0), (0.5 + 0.085, 0.23, 0.0)]
    faces += [(d0, d0 + 3, d0 + 1), (d0, d0 + 1, d0 + 4), (d0 + 3, d0 + 2, d0 + 1), (d0 + 1, d0 + 2, d0 + 4)]
    # every face looks outwards (+h)
    fixed = []
    for f in faces:
        a, b, c = (Vector((unit[i][0] * pitch, unit[i][1] * L, unit[i][2])) for i in f[:3])
        fixed.append(f if (b - a).cross(c - a).z >= 0 else tuple(reversed(f)))
    verts, allf = [], []
    nu = len(unit)
    beds = [bed(clamp(t, 0.0, 1.0)) for _, t, _ in unit]
    for kk in range(ne):
        for (s, t, h), ((r, z), (nr, nz)) in zip(unit, beds):
            ang = 2 * pi * (kk + s) / ne
            rr = r + nr * h
            verts.append((rr * cos(ang), rr * sin(ang), z + nz * h))
        allf += [tuple(kk * nu + i for i in f) for f in fixed]
    o = mesh_object(name, verts, allf)
    uv_box(o)
    smooth(o, 62)
    return o


# ------------------------------------------------------------------ build
def build(p, rng):
    H, R = p.height, p.width / 2
    k = R / 0.28                                    # ornament scale
    st = STYLES[p.bowl]
    stone = mat("stone_smooth", p.stone, grime=0.8)
    stone_r = mat("stone_rough", p.stone, grime=0.8)
    bronze = mat("bronze_aged")
    stones, rough_stones, plain = [], [], []

    # ---- pedestal: plinth course, base moulding, panelled die, cap
    a_sub = (st["foot_r"] + 0.04) * R
    hb = 0.09
    # keep the urn itself in proportion: a very tall order on a narrow bowl grows the pedestal instead
    ped_total = max(hb, p.pedestal * H, H - {"campana": 1.5, "tazza": 0.95, "ovoid": 1.9}[p.bowl] * p.width)
    has_die = ped_total - hb >= 0.3
    if not has_die:
        ped_total = hb
    d = a_sub + 0.012
    pb = d + 0.085 if has_die else a_sub + 0.06
    rp = rng.fork("plinth")
    settle = p.age * 1.6
    rough_stones.append(stone_block("plinth", (2 * pb, 2 * pb, hb), loc=(0, 0, hb / 2), rot=(0, 0, rp.u(-1, 1) * settle), chamfer=0.014,
                                    rough=0.004 + 0.003 * p.age, rng=rp, detail=0.2))
    z0 = ped_total
    ledges = [(pb + 0.012, 0.0, 1.0), (pb - 0.028, hb, 1.0)]        # (half-side, z, weight) for moss tufts
    die_q, zd, zc, hd = None, 0.0, 0.0, 0.0
    if has_die:
        hm, hc = 0.065, 0.075
        hd = ped_total - hb - hm - hc
        zb = hb
        stones.append(_sq_lathe("ped_base", [(d + 0.057, zb - 0.002), (d + 0.057, zb + 0.022), (d + 0.049, zb + 0.029), (d + 0.041, zb + 0.031),
                                             (d + 0.03, zb + 0.038), (d + 0.013, zb + 0.052), (d + 0.005, zb + 0.06), (d + 0.002, zb + hm),
                                             (d - 0.012, zb + hm)]))
        zd = zb + hm
        die, die_q = _sq_shell("ped_die", d, zd - 0.004, zd + hd + 0.004, rp.u(-0.5, 0.5) * settle, rng.fork("die_streaks"))
        displace_noise(die, strength=0.0012, scale=0.09, seed=p.seed + 2)
        smooth(die, 30)
        stones.append(die)
        zc = zd + hd
        stones.append(_sq_lathe("ped_cap", [(d - 0.012, zc - 0.002), (d + 0.006, zc), (d + 0.013, zc + 0.006), (d + 0.006, zc + 0.012),
                                            (d + 0.008, zc + 0.017), (d + 0.015, zc + 0.029), (d + 0.031, zc + 0.04), (d + 0.036, zc + 0.042),
                                            (d + 0.036, zc + 0.05), (d + 0.02, zc + 0.05)]))
        ct = hc - 0.044
        stones.append(stone_block("ped_corona", (2 * (d + 0.052), 2 * (d + 0.052), ct), loc=(0, 0, zc + 0.044 + ct / 2),
                                  rot=(0, 0, rp.u(-0.5, 0.5) * settle), chamfer=0.006, rough=0.0015, rng=rp, detail=0.2))
        ledges.append((d + 0.034, z0, 0.35))
        if hd >= 0.2:
            pw, ph = 2 * d - 0.085, hd - 0.075
            frame = shape_extrude("ped_panel", rounded_rect(pw, ph, 0.012, 1), [rounded_rect(pw - 0.04, ph - 0.04, 0.007, 1)], depth=0.007,
                                  bevel=0.002)
            pr = min(pw, ph) * 0.27
            pat = lathe("ped_patera", [(pr, 0.0), (pr * 1.03, 0.005), (pr * 0.86, 0.014), (pr * 0.5, 0.007), (pr * 0.3, 0.012),
                                       (pr * 0.24, 0.022), (0.0, 0.027)], segments=24)

            def petal(co):
                rr = hypot(co.x, co.y)
                if rr < pr * 0.32:
                    return co
                kp = 0.72 + 0.28 * abs(cos(atan2(co.y, co.x) * 3)) ** 0.5
                return Vector((co.x * kp, co.y * kp, co.z))

            transform_verts(pat, petal)
            pm = [trs(((d - 0.001) * sin(radians(q * 90)), -(d - 0.001) * cos(radians(q * 90)), zd + hd / 2), (90, 0, q * 90)) for q in range(4)]
            plain.append(instances_to_mesh("ped_panels", frame, pm, keep_base=False))
            if ph / pw > 0.62:
                plain.append(instances_to_mesh("ped_paterae", pat, pm, keep_base=False))
            else:
                delete([pat])
    hu = H - z0
    z_rim = z0 + hu

    def P(q):
        return (q[0] * R, z0 + q[1] * hu)

    # ---- urn: square sub-base and turned foot
    sub_h = (st["foot"][0][1] + 0.006) * hu
    stones.append(stone_block("urn_base", (2 * a_sub, 2 * a_sub, sub_h), loc=(0, 0, z0 + sub_h / 2), chamfer=0.007, rough=0.002,
                              rng=rng.fork("sub"), detail=0.15))
    ledges.append((a_sub * 0.93, z0 + sub_h, 0.5))
    foot_abs = [P(q) for q in st["foot"]]
    stones.append(lathe("urn_foot", [(0.0, foot_abs[0][1])] + foot_abs + [(0.0, foot_abs[-1][1])], segments=28))

    # ---- gadrooned cup: convex lobes with a crisp V between them and a rounded end under the shoulder bead
    ng = int(p.gadroons)
    (cr0, cz0), (cr1, cz1) = P(st["cup"][0]), P(st["cup"][1])
    Dg = clamp(0.19 * 2 * pi * cr1 / ng, 0.007, 0.016)

    def cup_rz(t):
        return cr0 + (cr1 - cr0) * sin(t * pi / 2) ** 0.85, cz0 + (cz1 - cz0) * (1 - cos(t * pi / 2))

    SJ = [-1.0, -0.9, -0.52, 0.0, 0.52, 0.9]
    HJ = [-0.3, 0.0, 0.84, 1.0, 0.84, 0.0]
    VR = [0.0, 0.14, 0.3, 0.48, 0.66, 0.84, 0.90, 0.94, 0.962, 0.978, 1.0] if ng <= 20 else [0.0, 0.14, 0.36, 0.62, 0.84, 0.91, 0.95, 0.978, 1.0]
    cpts = []
    for kk in range(ng):
        for j in range(6):
            col = []
            for v in VR:
                rb, z = cup_rz(v)
                D = Dg * (0.35 + 0.65 * v) * smoothstep(0.0, 0.16, v)
                if j == 0:
                    s, h = -1.0, HJ[0] * (1.0 - smoothstep(0.84, 0.978, v))
                elif v >= 0.999:
                    s, h = SJ[j], 0.0
                else:
                    x = clamp((v - 0.84) / (0.978 - 0.84), 0.0, 1.0)
                    w = sqrt(max(0.0, 1.0 - x * x))
                    s, h = SJ[j] * max(w, 0.1), HJ[j] * w ** 0.8
                a = 2 * pi * (kk + 0.5 + 0.5 * s) / ng
                r = rb - D * (1.0 - h)
                col.append((r * cos(a), r * sin(a), z))
            cpts.append(col)
    cup_o = _grid("urn_cup", cpts)
    smooth(cup_o, 33)
    stones.append(cup_o)

    # ---- shoulder bead and bell, one skin; the festoons are carved out of the same block (raised relief)
    NU = 72
    up_n = st["upper"]
    sh = [P(q) for q in up_n[:6]]
    c_r, c_z = (sh[0][0] + sh[4][0]) / 2, sh[2][1]
    a_e, b_e = sh[2][0] - c_r, sh[2][1] - sh[0][1]
    sh_rows = [(min(c_r, cr1 - Dg) - 0.012, sh[0][1] + 0.003)]
    sh_rows += [(c_r + a_e * cos(radians(t)), c_z + b_e * sin(radians(t))) for t in (-90, -58, -28, 0, 28, 58)] + [sh[4]]
    bell = [P(q) for q in (up_n[5:-2] if len(up_n) > 9 else up_n[5:])]
    zA, zB = bell[0][1], bell[-1][1]
    if len(bell) >= 3:
        dense = [(float(q[0]), float(q[2])) for q in resample_path([(r, 0.0, z) for r, z in bell], n=48)]
        dense.sort(key=lambda q: q[1])
    else:
        dense = bell
    bz, br = np.array([q[1] for q in dense]), np.array([q[0] for q in dense])

    def rbell(z):
        return float(np.interp(z, bz, br))

    def dbell(z):
        return (rbell(z + 0.004) - rbell(z - 0.004)) / 0.008

    has_swag = bool(p.swags and st["swag"] and len(bell) >= 3)
    NB = 12 if len(bell) >= 3 else 2
    spans = []
    if has_swag:
        zpeg, zsag = z0 + st["swag"][0] * hu, z0 + st["swag"][1] * hu
        Dp = zpeg - zsag
        spans = [(215, 325), (35, 145)] if p.handles != "none" else [(45 + q * 90, 135 + q * 90) for q in range(4)]

    def sw_w(t):                                    # half-width of the garland: thick at the bottom of the sag
        return k * (0.0075 + 0.011 * sin(pi * clamp(t, 0.0, 1.0)) ** 1.3)

    def sw_zc(t):
        return zpeg - Dp * (1.0 - (2 * t - 1) ** 2)

    def sw_fade(t):
        return smoothstep(0.0, 0.06, t) * (1.0 - smoothstep(0.94, 1.0, t))

    def swag_at(adeg):
        best = (0.0, 0.0, None)
        for a0, a1 in spans:
            for off in (-360, 0, 360):
                x = adeg + off
                if a0 <= x <= a1:
                    return 1.0, (x - a0) / (a1 - a0), (a0, a1)
                w = 1.0 - (a0 - x if x < a0 else x - a1) / 12.0
                if w > best[0]:
                    best = (w, 0.0 if x < a0 else 1.0, (a0, a1))
        return best

    rw = rng.fork("weather")
    cols_deg = [360.0 * i / NU for i in range(NU)]
    bell_streaks = _streak_cols(rw, NU, 16, lo=0)
    for c in bell_streaks:
        cols_deg[(c - 1) % NU], cols_deg[(c + 1) % NU] = cols_deg[c] - 1.7, cols_deg[c] + 1.7
    # one level row just above the garland: where the bell flares out to the rim, rows that followed the garland
    # made long skewed quads that shaded as flat panels (one half of every span dark, the other light)
    ztop = min(zpeg + 0.022 * k, zB - 0.02 * k) if has_swag else zA + 0.75 * (zB - zA)
    bpts = []
    for adeg in cols_deg:
        a = radians(adeg)
        zs_ = [zA + (ztop - zA) * j / (NB - 2) for j in range(NB - 1)] + [zB] if NB > 2 else [zA, zB]
        hs_ = [0.0] * NB
        wgt, t, span = swag_at(adeg) if has_swag else (0.0, 0.0, None)
        if wgt > 0:
            zc_, w = sw_zc(t), sw_w(t)
            sl = 4 * Dp * (2 * t - 1) / (rbell(zc_) * radians(span[1] - span[0]))
            wv = min(w * sqrt(1 + sl * sl), 2.0 * w)
            hm_ = 0.8 * w * sw_fade(t)
            zlo, zhi = zc_ - wv - 0.003 * k, zc_ + wv + 0.003 * k
            zw = [zA + (zlo - zA) * j / 3 for j in range(4)] + [zc_ + wv * o for o in (-1, -0.6, 0, 0.6, 1)] + [min(zhi, ztop - 0.004 * k), ztop, zB]
            hw = [0.0] * 4 + [hm_ * c for c in (0.0, 0.8, 1.0, 0.8, 0.0)] + [0.0] * 3
            zs_ = [e + (s_ - e) * wgt for e, s_ in zip(zs_, zw)]
            hs_ = [h * wgt for h in hw]
        col = [(r * cos(a), r * sin(a), z) for r, z in sh_rows]
        for z, h in zip(zs_, hs_):
            r = rbell(z) + h
            col.append((r * cos(a), r * sin(a), z))
        bpts.append(col)
    body = _grid("urn_body", bpts)
    smooth(body, 45)
    stones.append(body)

    # ---- rim: astragal, egg-and-dart ovolo, fillet, rolled lip, inner wall
    rb_t, zb_ = bell[-1]
    Hr = z_rim - zb_
    R_lip = max(q[0] for q in st["lip"]) * R
    hbd = 0.17 * Hr
    z1, z2, z3 = zb_ + 0.22 * Hr, zb_ + 0.76 * Hr, zb_ + 0.80 * Hr
    r_o0, r_o1 = rb_t + 0.035 * R, R_lip - 0.036 * R
    ph0 = radians(35)
    A_, B_ = (r_o1 - r_o0) / (1 - sin(ph0)), (z2 - z1) / cos(ph0)

    def crest(t):
        ph = ph0 + (pi / 2 - ph0) * t
        n = Vector((B_ * sin(ph), -A_ * cos(ph))).normalized()
        return (r_o1 - A_ + A_ * sin(ph), z2 - B_ * cos(ph)), (n.x, n.y)

    cs = [crest(i / 12)[0] for i in range(13)]
    L = sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(cs, cs[1:]))
    r_mid = crest(0.5)[0][0]
    ne = int(clamp(round(2 * pi * r_mid / (0.72 * L)), 20, 56))
    pitch = 2 * pi * r_mid / ne
    rec = min(0.2 * pitch, 0.007 * k)
    hE = min(0.27 * pitch, rec + 0.0035 * k)

    def bed(t):
        (r, z), (nr, nz) = crest(t)
        return (r - nr * rec, z - nz * rec), (nr, nz)

    rcb, rad = rb_t + 0.25 * hbd, hbd / 2
    prof = [(rb_t - 0.004, zb_ - 0.003), (rb_t + 0.001, zb_)]
    prof += [(rcb + rad * cos(radians(t)), zb_ + rad + rad * sin(radians(t))) for t in (-50, 0, 50)]
    prof += [(rcb, zb_ + hbd), (r_o0, z1)]
    prof += [bed(i / 3)[0] for i in range(4)]
    r_f = r_o1 + 0.0015 * k
    prof += [(r_f, z2), (r_f, z3)]
    r_lc, hl = r_f + 0.001, z_rim - z3
    prof += [(r_lc + (R_lip - r_lc) * cos(radians(t)), z3 + hl / 2 + hl / 2 * sin(radians(t))) for t in (-55, -15, 25, 62, 90)]
    n_out = len(prof)
    in_abs = [P(q) for q in st["inner"]]
    rim = lathe("urn_rim", prof + in_abs[:1] + in_abs[2:], segments=48)
    # chipped lip: a couple of knocks out of the arris
    rc = rng.fork("chips")
    chips = [(rc.u(0, 2 * pi), rc.u(0.05, 0.11), rc.u(0.004, 0.011) * p.age) for _ in range(3)]

    def chip(co):
        if co.z < z3:
            return co
        rr = hypot(co.x, co.y)
        if rr < R_lip * 0.955:
            return co
        a = atan2(co.y, co.x)
        dz = 0.0
        for ca, cw, cd in chips:
            da = abs((a - ca + pi) % (2 * pi) - pi)
            if da < cw:
                dz = max(dz, cd * (1 - da / cw))
        kc = 1.0 - dz * 0.5 / max(rr, 1e-6)
        return Vector((co.x * kc, co.y * kc, co.z - dz * (1.0 if co.z > z_rim - 0.4 * hl else 0.3)))

    if p.age > 0.05:
        transform_verts(rim, chip)
    stones.append(rim)
    stones.append(_egg_and_dart("urn_egg_dart", bed, ne, pitch, L, hE))
    R_rim = R_lip
    outer = foot_abs + [cup_rz(i / 8) for i in range(9)] + sh_rows[1:] + [(rbell(zA + (zB - zA) * i / 11), zA + (zB - zA) * i / 11) for i in range(12)] + prof[1:n_out]

    boat = _boat("carve_leaf_b")
    carve = []                                      # carved leaves / husks / petals, one mesh

    def on_bell(adeg, z):
        """Point, outward normal, tangent around, tangent up of the bell surface."""
        a = radians(adeg)
        r, dr = rbell(z), dbell(z)
        e_a = Vector((-sin(a), cos(a), 0.0))
        e_z = Vector((dr * cos(a), dr * sin(a), 1.0)).normalized()
        return Vector((r * cos(a), r * sin(a), z)), e_a.cross(e_z).normalized(), e_a, e_z

    # ---- handles
    metal = []
    if p.handles == "ear":
        hd_ = st["handle"]
        rv = hd_["rv"] * R
        zBh = z0 + hd_["B"] * hu
        Bp = (_r_at(outer, zBh) - 0.008 * k, zBh)
        An = P(hd_["anchor"])
        C = (An[0] + hd_["off"][0] * rv, An[1] + hd_["off"][1] * rv)
        Q = (C[0] + rv, C[1])
        P1 = (Bp[0] + hd_["p1"][0] * rv, Bp[1] + hd_["p1"][1] * rv)
        P2 = (Q[0], Q[1] + 1.5 * rv)
        W, Th = 0.048 * k, 0.030 * k
        sec = [(Th * a, W * b) for a, b in zip((0.26, 0.47, 0.34, 0.5, 0.34, 0.47, 0.26), (-0.5, -0.36, -0.2, 0.0, 0.2, 0.36, 0.5))]
        sec += [(-Th * a, W * b) for a, b in zip((0.3, 0.45, 0.5, 0.45, 0.3), (0.42, 0.25, 0.0, -0.25, -0.42))]
        nsp = 13
        for sx in (-1, 1):
            pts = [tuple(q) for q in bezier_path(*[(sx * r, 0.0, z) for r, z in (Bp, P1, P2, Q)], n=14)]
            for j in range(1, nsp + 1):
                th = -2 * pi * 1.2 * j / nsp
                rr = rv * (1.0 - 0.64 * (j / nsp) ** 0.9)
                pts.append((sx * (C[0] + rr * cos(th)), 0.0, C[1] + rr * sin(th)))
            lens = [0.0]
            for a, b in zip(pts, pts[1:]):
                lens.append(lens[-1] + (Vector(b) - Vector(a)).length)
            ts = lens[14] / lens[-1]

            def hscale(t, ts=ts):
                return float(np.interp(t, [0.0, 0.45 * ts, ts, 1.0], [1.08, 0.8, 1.02, 0.5])) + 0.7 * np.exp(-t / 0.05)

            cen = Vector((sx * (Bp[0] + Q[0] + An[0]) / 3, 0.0, (Bp[1] + Q[1] + An[1]) / 3))
            for flip in (1, -1):
                ho = sweep(f"urn_handle{sx}", pts, [(a * flip, b * flip) for a, b in sec], up=(0, 1, 0), scale_fn=hscale, smooth_path=False)
                vs = ho.data.vertices
                if (vs[7 * 12 + 3].co - cen).length > (vs[7 * 12 + 9].co - cen).length:
                    break
                delete([ho])
            smooth(ho, 40)
            stones.append(ho)
            stones.append(cylinder(f"urn_volute_eye{sx}", radius=0.3 * rv, depth=W * 0.62, loc=(sx * C[0], 0, C[1]), rot=(90, 0, 0), verts=12,
                                   bevel=0.004 * k))
            # two leaves spread on the bell where the handle springs from it
            for sy in (-1, 1):
                root = Vector((sx * (_r_at(outer, zBh) - 0.002), sy * 0.008 * k, zBh + 0.004 * k))
                carve.append(_frame(root, (-sx * 0.22, sy, -0.45), (sx, 0, 0), (0.026 * k, 0.046 * k, 0.007 * k)))
    elif p.handles == "ring":
        zr = z0 + st["ring_z"] * hu
        rr0 = _r_at(outer, zr)
        maj, mnr = 0.17 * R, 0.011 * k + 0.002
        rr1 = _r_at(outer, zr - 1.9 * maj)
        lean = max(0.0, rr1 - rr0) / (1.9 * maj)
        dn = Vector((lean, 0, -1)).normalized()
        boss = _dome("ring_boss_b", 10, 3)
        bm = []
        for sx in (-1, 1):
            bw = 0.2 * R
            bm.append(_cols((sx * (rr0 - 0.004), 0, zr), (0, sx, 0), (0, 0, 1), (sx, 0, 0), (bw, bw, bw * 0.34)))
            c0 = Vector((sx * (rr0 + bw * 0.3), 0, zr - 0.012 * k))
            dd = Vector((sx * dn.x, 0, dn.z))
            nrm = Vector((sx * -dn.z, 0, dn.x))            # outward normal of the ring plane
            M = _cols(c0 + dd * maj + nrm * (mnr + 0.002), (0, 1, 0), dd, Vector((0, 1, 0)).cross(dd))
            ring = torus(f"urn_ring{sx}", maj, mnr, major_seg=24, minor_seg=7)
            transform_verts(ring, lambda co, M=M: M @ co)
            metal.append(ring)
            stp = torus(f"urn_staple{sx}", mnr * 2.1, mnr * 0.8, major_seg=12, minor_seg=6)
            Ms = _cols(Vector((sx * (rr0 + bw * 0.18), 0, zr - 0.006 * k)), (sx, 0, 0), (0, 0, 1), Vector((sx, 0, 0)).cross(Vector((0, 0, 1))))
            transform_verts(stp, lambda co, M=Ms: M @ co)
            metal.append(stp)
        stones.append(instances_to_mesh("urn_ring_boss", boss, bm, keep_base=False))
    for o in metal:
        assign(o, bronze)
        paint(o, special=mask_mul(noise_mask(0.03, seed=p.seed + 5, lo=0.3, hi=0.7), 0.5 + 0.5 * p.age),
              wear=mask_mul(noise_mask(0.02, seed=p.seed + 6, lo=0.2, hi=0.8), 0.7))

    # ---- festoons: laurel leaves laid herringbone on the relief, fruit at the bottom of the sag, rosette ties, husk drops
    if has_swag:
        berry = _dome("swag_berry_b", 6, 2)
        bm = []
        for a0, a1 in spans:
            da = radians(a1 - a0)

            def at(t, off):
                zc_, w = sw_zc(t), sw_w(t)
                pos, nrm, e_a, e_z = on_bell(a0 + (a1 - a0) * t, zc_)
                dv = (e_a * (rbell(zc_) * da) + e_z * (4 * Dp * (2 * t - 1))).normalized()
                perp = nrm.cross(dv)
                hgt = 0.8 * w * sw_fade(t) * sqrt(max(0.0, 1.0 - off * off))
                return pos + perp * (off * w) + nrm * hgt, dv, perp, nrm, w

            path_len = rbell(zsag) * da * 1.12
            t = 0.045
            idx = 0
            while t < 0.485:
                w = sw_w(t)
                step = 1.2 * w / path_len
                for mirror in (False, True):
                    sg = -1.0 if mirror else 1.0
                    tt = 1.0 - t if mirror else t
                    pos, dv, perp, nrm, w = at(tt, 0.0)
                    carve.append(_frame(pos + nrm * 0.0005, dv * sg, nrm, (0.95 * w, 2.3 * w, 0.34 * w)))
                    tt2 = clamp(tt + sg * step * 0.5, 0.0, 1.0)
                    for side in (-1, 1):
                        pos, dv, perp, nrm, w = at(tt2, side * 0.5)
                        carve.append(_frame(pos, dv * sg + perp * (side * 0.55), nrm + perp * (side * 0.5), (0.8 * w, 1.9 * w, 0.3 * w)))
                t += step
                idx += 1
            for dt, off, sc in ((0.0, 0.0, 1.25), (-0.028, 0.5, 0.85), (0.028, 0.5, 0.85), (-0.03, -0.5, 0.85), (0.03, -0.5, 0.85)):
                pos, dv, perp, nrm, w = at(0.5 + dt, off)
                bm.append(_frame(pos - nrm * 0.002, dv, nrm, (sc * w, sc * w, sc * w * 0.62)))
        pr = 0.027 * k
        for adeg in sorted({a % 360 for sp in spans for a in sp}):
            pos, nrm, e_a, e_z = on_bell(adeg, zpeg)
            for i in range(6):
                ps = radians(60 * i + 30)
                carve.append(_frame(pos + nrm * 0.001, e_a * cos(ps) + e_z * sin(ps), nrm, (0.62 * pr, pr, 0.26 * pr)))
            bm.append(_frame(pos, e_a, nrm, (0.75 * pr, 0.75 * pr, 0.42 * pr)))
            z = zpeg - 0.8 * pr
            for f in (1.3, 1.05, 0.8):
                pos, nrm, e_a, e_z = on_bell(adeg, z)
                carve.append(_frame(pos + nrm * 0.0005, -e_z, nrm, (0.56 * f * pr, f * pr, 0.3 * f * pr)))
                z -= 0.82 * f * pr
        sb = instances_to_mesh("urn_swag_fruit", berry, bm, keep_base=False)
        smooth(sb, 60)
        stones.append(sb)
    if carve:
        stones.append(instances_to_mesh("urn_carving", boat, carve, keep_base=False))
    else:
        delete([boat])

    # ---- soil and planting
    empty_pot = p.planting == "empty"
    zs = z0 + (st["soil"] - (0.075 if empty_pot else 0.0)) * hu
    ro = _r_at(in_abs, zs)
    soil = lathe("soil", [(0.0, zs + 0.012), (ro * 0.55, zs + 0.009), (ro * 0.9, zs), (ro * 1.03, zs - 0.006)], segments=20)
    displace_noise(soil, strength=0.003, scale=0.05, seed=p.seed + 3)
    assign(soil, mat("soil"))
    # ---- moss tufts in the ledges, mostly on the damp back (+Y) side
    nm = int(p.moss * 40)
    if nm > 0:
        tuft = _dome("moss_b", 7, 2)
        tm = []
        rm = rng.fork("moss")
        guard = 0
        while len(tm) < nm and guard < 600:
            guard += 1
            half, z, wgt = rm.pick(ledges)
            x, y = _perim(half, rm.u(0, 1))
            if not rm.chance(wgt * (0.3 + 0.7 * smoothstep(-half, half, y))):
                continue
            w = rm.u(0.045, 0.12)
            tm.append(trs((x, y, z - 0.002), (0, 0, rm.u(0, 360)), (w, w * rm.u(0.45, 0.8), rm.u(0.008, 0.015))))
        if empty_pot:
            for _ in range(int(4 + 8 * p.moss)):
                a, rr = rm.u(0, 2 * pi), ro * rm.u(0.5, 0.97)
                w = rm.u(0.04, 0.09)
                tm.append(trs((rr * cos(a), rr * sin(a), zs - 0.004), (0, 0, rm.u(0, 360)), (w, w * rm.u(0.5, 0.9), rm.u(0.014, 0.026))))
        mo = instances_to_mesh("moss_tufts", tuft, tm, keep_base=False)
        assign(mo, _moss_mat())
        smooth(mo, 80)

    # ---- planting: built last, thinned (q) if the carved stone already used most of the triangle budget
    S = (ro / 0.24) ** 0.7 * p.plant_size
    scene = bpy.context.scene

    def tris(objs):
        dg = bpy.context.evaluated_depsgraph_get()
        return sum(len(pl.vertices) - 2 for o in objs if o.type == "MESH" for pl in o.evaluated_get(dg).data.polygons)

    before = set(o.name for o in scene.objects)
    left = META["tri_budget"] - 300 - tris(scene.objects)
    q = 1.0
    for attempt in range(5):
        rpl = rng.fork("plant")
        if p.planting == "topiary":
            _topiary(p, rpl, zs, ro, S, q)
        elif p.planting == "flowers":
            _flowers(p, rpl, zs, ro, S, R_rim, z_rim, outer, hu, q)
        elif p.planting == "fern":
            _fern(p, rpl, zs, ro, S, R_rim, z_rim, q)
        else:
            _empty(p, rpl, zs, ro, pb)
        grown = [o for o in scene.objects if o.name not in before]
        used = tris(grown)
        if used <= left or attempt == 4 or p.planting == "empty":
            break
        delete(grown)
        q *= max(0.45, 0.92 * left / used)

    # ---- weathering of the stonework, driven by the geometry: where water runs, sits and dries last
    n1 = noise_mask(0.17, seed=p.seed + 11, lo=0.3, hi=0.7)
    n2 = noise_mask(0.05, seed=p.seed + 12, lo=0.3, hi=0.75)
    n3 = noise_mask(0.3, seed=p.seed + 13, lo=0.25, hi=0.75)
    st_bell = _streak_fn([radians(cols_deg[c]) % (2 * pi) for c in bell_streaks], [rw.u(0.5, 1.0) for _ in bell_streaks],
                         [rw.u(0.3, 1.0) * (zB - zA) + 0.02 for _ in bell_streaks], radians(1.0), period=2 * pi)
    vamp = np.array([rw.u(0.3, 1.0) for _ in range(ng)])
    st_die = None
    if die_q and sum(len(q) for q in die_q):
        cen, amp, ln = [], [], []
        for f, qf in enumerate(die_q):
            for q in qf:
                cen.append(q + 10.0 * f)
                amp.append(rw.u(0.45, 1.0))
                ln.append(rw.u(0.25, 1.0) * hd)
        st_die = _streak_fn(cen, amp, ln, 0.0042)
    age_k = 0.3 + 0.7 * p.age
    mk = p.moss * 1.25

    def moss_fn(co, n):
        z = co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        vert = 1.0 - np.abs(n[:, 2])
        north = np.clip(n[:, 1] * 0.8 + 0.2, 0, 1)
        low = 1.0 - np.clip(z / H, 0, 1)
        base = np.clip(1.0 - z / 0.16, 0, 1)
        a1_, a3_ = n1(co, n), n3(co, n)
        v = up ** 1.3 * (0.45 + 0.55 * a1_) + 0.9 * north * vert * (0.35 + 0.65 * low) * a1_ + 0.8 * base * (0.4 + 0.6 * a3_)
        v = v + 0.3 * max(0.0, p.moss - 0.45) * vert * a3_ * (0.4 + 0.6 * low)
        return np.clip(v * mk, 0, 1)

    def make_grime(streaks=True):
        def grime_fn(co, n):
            x, y, z = co[:, 0], co[:, 1], co[:, 2]
            rr = np.hypot(x, y) + 1e-9
            ang = np.arctan2(y, x) % (2 * pi)
            down = np.clip(-n[:, 2], 0, 1)
            vert = 1.0 - np.abs(n[:, 2])
            g = 0.3 * down + 0.1 * n1(co, n) + 0.22 * np.clip(1.0 - z / 0.22, 0, 1) ** 1.5 * n3(co, n)
            # the urn
            urn = z > z0 + 0.5 * sub_h
            # dirt washed off the rim runs down the bell in thin streaks
            on_b = urn & (z <= zB + 0.004) & (z >= zA - 0.004)
            g = g + on_b * 0.85 * st_bell(ang, zB - z) * (0.55 + 0.45 * n2(co, n))
            # ... collects in the valleys between the gadroons
            ph = ang * ng / (2 * pi)
            dv = np.abs(ph - np.round(ph))
            on_c = urn & (z > cz0 + 0.1 * (cz1 - cz0)) & (z < cz1 + 0.002) & (rr < cr1 + 0.004)
            g = g + on_c * 0.8 * np.clip(1.0 - dv / 0.07, 0, 1) * vamp[np.round(ph).astype(int) % ng]
            # ... and the soil stains the inside wall above the soil line
            inward = np.clip(-(x * n[:, 0] + y * n[:, 1]) / rr, 0, 1)
            g = g + urn * inward * (z > zs - 0.02) * np.clip(1.0 - (z - zs) / 0.07, 0, 1) * 0.75
            # the pedestal: run-off from the drip edge of the cap down the die
            if st_die is not None and streaks:
                fx = np.abs(n[:, 0]) > np.abs(n[:, 1])
                fid = np.where(fx, np.where(n[:, 0] > 0, 1, 3), np.where(n[:, 1] > 0, 2, 0))
                tc = np.where(fx, np.where(n[:, 0] > 0, y, -y), np.where(n[:, 1] > 0, -x, x))
                on_d = (z > zd - 0.01) & (z < zc + 0.006) & (vert > 0.6)
                g = g + on_d * 0.8 * st_die(tc + 10.0 * fid, zc - z) * (0.55 + 0.45 * n2(co, n))
            g = np.clip(g * age_k, 0, 0.62)
            return g * (1.0 - 0.7 * moss_fn(co, n))
        return grime_fn

    def wear_fn(co, n):
        top = np.clip((co[:, 2] - (z_rim - 0.012)) / 0.012, 0, 1) * np.clip(n[:, 2], 0, 1)
        return np.clip(0.5 * top * (0.4 + 0.6 * p.age) * (0.5 + 0.5 * n2(co, n)), 0, 1)

    for o in stones + plain:
        assign(o, stone)
    for o in rough_stones:
        assign(o, stone_r)
    g_all, g_plain = make_grime(True), make_grime(False)
    for o in stones + rough_stones:
        paint(o, grime=g_all, wear=wear_fn, special=moss_fn)
    for o in plain:
        paint(o, grime=g_plain, wear=wear_fn, special=moss_fn)
    # the tooling / bedding of the 'stone_smooth' texture runs diagonally in UV space: turn it level, so the bed of
    # the stone lies horizontal around the urn and on every face of the pedestal
    for o in stones + plain:
        uv_transform(o, rot_deg=GRAIN_ROT)
        uv_keep(o)

    socket("top", "attach", (0, 0, z_rim), role="prop_top")
