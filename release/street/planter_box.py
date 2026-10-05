"""Timber planter box (Versailles tub / caisse a oranger): four chamfered corner posts standing on their
own feet and topped with moulded caps and turned ball finials, top and bottom rails with a capping board,
panels of individual boards (optionally with a half-lapped, mitred saltire brace or a raised-and-fielded
panel), forged iron strap bands pinned to the posts and nailed through every board, drop-ring lifting
handles, a zinc liner lip and mounded soil. Planted with a clipped standard orange tree (S-curved tapering
trunk with root flare, forking scaffold branches, stake and tie, one irregular clipped head of leaf cards
over a dark inner shell), a topiary cone / ball on a woody stem, or a mound of cupped flowers with
trailing stems. Paint is aged per piece: chipped arrises, flaking damp feet and bottom rail, rust runs
under every strap, nail and pin, algae at the base of the shaded sides, board-to-board tone variation.
"""
from tk import *
from math import cos, sin, pi, radians, degrees, atan2, hypot, sqrt, exp
import colorsys
import numpy as np
from mathutils import Matrix, Vector

META = {"name": "Timber planter box", "class": "prop", "mount": "ground", "focus": [0.0, 0.0, 0.62],
        "tri_budget": 18000}   # tub joinery + a whole plant; the default variant stays under ~15k
PARAMS = {
    "size": {"default": 0.72, "min": 0.45, "max": 1.2, "desc": "outer width of the square tub over the posts (m)"},
    "proportion": {"default": 0.9, "min": 0.65, "max": 1.3, "desc": "body height / width"},
    "paint": {"default": "#4c7659", "desc": "paint colour of the timber"},
    "panel": {"default": "planks", "choices": ["planks", "cross", "fielded"],
              "desc": "panel infill: plain boards, boards with a saltire brace, or a raised-and-fielded panel"},
    "bands": {"default": 2, "min": 0, "max": 3,
              "desc": "iron strap bands per side (on the boards for 'planks'; on the rails, max 2, otherwise)"},
    "finials": {"default": True, "desc": "turned ball finials on the posts (else low pyramid caps)"},
    "handles": {"default": True, "desc": "iron drop-ring lifting handles on the two flanks"},
    "planting": {"default": "tree", "choices": ["tree", "cone", "ball", "flowers"],
                 "desc": "standard orange tree, topiary cone, topiary ball on a stem, or a mound of flowers"},
    "plant_scale": {"default": 1.0, "min": 0.6, "max": 1.5, "desc": "size of the plant relative to normal"},
    "bloom": {"default": "#f08a1e", "desc": "fruit / flower colour"},
    "blooms": {"default": 11, "min": 0, "max": 30, "desc": "fruits on the tree; flower heads = 6x this"},
    "stake": {"default": True, "desc": "gardener's stake and tie beside the stem (tree / ball)"},
    "label": {"default": "VII", "desc": "number on the bare-oak tally board nailed to the front rail ('' = no board)"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: chipped paint, rust runs, damp, algae"},
}
VARIANTS = {
    "topiary_cone": {"planting": "cone", "panel": "fielded", "size": 0.62, "proportion": 1.0, "paint": "#7d8f9a",
                     "handles": False, "label": "IV", "age": 0.3, "seed": 4},
    "flower_tub": {"planting": "flowers", "panel": "cross", "size": 0.95, "proportion": 0.65, "paint": "#8c3a2c",
                   "finials": False, "bloom": "#f4e6dc", "blooms": 22, "label": "", "age": 0.8, "seed": 11},
    "ball_cream": {"planting": "ball", "panel": "cross", "size": 0.56, "proportion": 1.05, "paint": "#cfc7b0",
                   "bands": 1, "plant_scale": 0.9, "label": "XII", "age": 0.55, "seed": 7},
    "grand_orangery": {"planting": "tree", "panel": "planks", "size": 1.05, "proportion": 0.84, "paint": "#4d6678",
                       "bands": 3, "plant_scale": 1.35, "blooms": 22, "label": "XXI", "age": 0.6, "seed": 19},
}

_EDGE = {}     # object name -> per-vertex "arris" weight (1 on chamfers, 0 mid-face) for the paint-chip mask
_GOLD = 2.39996323
_HEDGE = (1.9, 2.0, 1.5)   # the hedge shell lifted to sit just under the value of the leaf cards on top of it


# ------------------------------------------------------------------ small helpers
def _rz(k):
    return Matrix.Rotation(radians(90.0 * k), 4, 'Z')


def _yaw(o, k):
    """Turn an object built for the front (-Y) side onto side k (quarter turns about the tub axis)."""
    apply_transform(o)
    if k:
        place(o, rot=(0, 0, 90.0 * k))
    return o


def _uvmesh(name, verts, faces, uvs):
    o = mesh_object(name, verts, faces)
    me = o.data
    uvl = me.uv_layers.active or me.uv_layers.new(name="UVMap")
    for li, loop in enumerate(me.loops):
        uvl.data[li].uv = uvs[loop.vertex_index]
    uv_keep(o)
    return o


def _tris(objs):
    return sum(len(f.vertices) - 2 for o in objs if o.type == 'MESH' for f in o.data.polygons)


def _dome(name, r, h, seg=6):
    return lathe(name, [(r, 0.0), (r * 0.78, h * 0.66), (0.0, h)], segments=seg)


def _soften(hexcol, sat=0.85, val=0.92):
    """Pull a '#rrggbb' colour's saturation / value down (fruit must not read as emissive at night)."""
    h = hexcol.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    r, g, b = colorsys.hsv_to_rgb(hh, ss * sat, vv * val)
    return '#%02x%02x%02x' % (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))


# ------------------------------------------------------------------ timber
def _strip(name, o, d, s0, s1, hw, yb, yf, c=0.004, nseg=6, planes=(), xf=None, prof=None, r=None, end=0.012):
    """Chamfered timber lying in the XZ plane with its face towards -Y.

    o = (x, z) point on the centre line, d = unit direction (dx, dz), s0..s1 = extent along d, hw = half
    width, yb / yf = back / front plane. `planes` [(a, b, c)] clip every fibre to a*x + b*z <= c, which gives
    true mitred ends (saltire braces into the frame corners, half-laps). Rings are added just inside both
    ends so baked contact shadows stay thin, and a centre column carries mid-face weathering masks.
    xf maps the finished (x, y, z) somewhere else (capping boards). UVs: grain (U) along the length."""
    dx, dz = d
    nx, nz = -dz, dx
    if prof is None:
        prof = [(-hw, yb, 0.3), (-hw, yf + c, 1.0), (-hw + c, yf, 1.0), (0.0, yf, 0.0),
                (hw - c, yf, 1.0), (hw, yf + c, 1.0), (hw, yb, 0.3)]
    uo = r.u(0, 3) if r else 0.0
    vo = r.u(0, 1) if r else 0.0
    nr = nseg + 3
    verts, uvs, edge = [], [], []
    acc = 0.0
    for ci, (v, y, e) in enumerate(prof):
        if ci:
            acc += hypot(v - prof[ci - 1][0], y - prof[ci - 1][1])
        px, pz = o[0] + nx * v, o[1] + nz * v
        a, b = s0, s1
        for pa, pb, pc in planes:
            den = pa * dx + pb * dz
            num = pc - (pa * px + pb * pz)
            if den > 1e-9:
                b = min(b, num / den)
            elif den < -1e-9:
                a = max(a, num / den)
        b = max(b, a + 1e-4)
        L = b - a
        e_ = min(end, L * 0.2)
        st = [a] + [a + e_ + (L - 2 * e_) * i / nseg for i in range(nseg + 1)] + [b]
        for s in st:
            verts.append((px + dx * s, y, pz + dz * s))
            uvs.append((s + uo, acc + vo))
            edge.append(e)
    m = len(prof)
    faces = []
    for ci in range(m - 1):
        for j in range(nr - 1):
            a0 = ci * nr + j
            faces.append((a0, a0 + 1, a0 + nr + 1, a0 + nr))
    V = [Vector(v) for v in verts]
    for j, sgn in ((0, -1.0), (nr - 1, 1.0)):
        ids = [ci * nr + j for ci in range(m)]
        nrm = Vector((0, 0, 0))
        for q in range(m):
            p0, p1 = V[ids[q]], V[ids[(q + 1) % m]]
            nrm += Vector(((p0.y - p1.y) * (p0.z + p1.z), (p0.z - p1.z) * (p0.x + p1.x), (p0.x - p1.x) * (p0.y + p1.y)))
        if nrm.dot(Vector((dx, 0, dz))) * sgn < 0:
            ids.reverse()
        for q in range(1, m - 1):
            faces.append((ids[0], ids[q], ids[q + 1]))
    if xf:
        verts = [xf(*v) for v in verts]
    ob = _uvmesh(name, verts, faces, uvs)
    smooth(ob, 30)
    _EDGE[ob.name] = np.array(edge)
    return ob


def _post(name, cx, cy, pw, zs, c, r):
    """Square post with chamfered arrises: 12-point rings (arris pairs + face centres) at heights zs."""
    hw = pw / 2
    ring = []
    for ca, sa in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        for a, b, e in ((hw, -hw + c, 1.0), (hw, 0.0, 0.0), (hw, hw - c, 1.0)):
            ring.append((a * ca - b * sa, a * sa + b * ca, e))
    ring.append(ring[0])
    uo, vo = r.u(0, 3), r.u(0, 1)
    m = len(ring)
    verts, uvs, edge, faces = [], [], [], []
    for j, z in enumerate(zs):
        acc = 0.0
        for i, (x, y, e) in enumerate(ring):
            if i:
                acc += hypot(x - ring[i - 1][0], y - ring[i - 1][1])
            verts.append((cx + x, cy + y, z))
            uvs.append((z + uo, acc + vo))
            edge.append(e)
    for j in range(len(zs) - 1):
        for i in range(m - 1):
            a0 = j * m + i
            faces.append((a0, a0 + 1, a0 + m + 1, a0 + m))
    ob = _uvmesh(name, verts, faces, uvs)
    smooth(ob, 30)
    _EDGE[ob.name] = np.array(edge)
    return ob


def _weather(o, seed, age, edge_k=1.0, flake_k=0.35, damp=None, fade=0.0, dark=0.0, extra=None, e_const=None):
    """Art-directed paint ageing for one timber piece (painted in the side's own frame, before it is yawed):
    chips along the arrises, flaking patches, damp-dark flaking below damp=(z, range), sun fade on up-facing
    faces plus a per-piece tone offset (fade / dark), optional extra grime fn (rust / dirt runs)."""
    E = _EDGE.get(o.name)
    chips = noise_mask(0.03, seed=seed, lo=0.58 - 0.4 * age, hi=0.9 - 0.32 * age)
    flakes = noise_mask(0.085, seed=seed + 101, lo=0.8 - 0.4 * age, hi=1.0 - 0.32 * age)
    fine = noise_mask(0.028, seed=seed + 202, lo=0.25, hi=0.7)
    tone = noise_mask(0.3, seed=seed + 303, lo=0.2, hi=0.8)
    ka = 0.3 + 0.7 * age

    def low(co):
        return np.clip((damp[0] - co[:, 2]) / damp[1], 0.0, 1.0)

    def wear(co, n):
        e = E if (E is not None and len(E) == len(co)) else (e_const or 0.0)
        w = np.maximum(e * chips(co, n) * edge_k * (0.45 + 0.55 * age), flakes(co, n) * flake_k * ka)
        if damp:
            w = np.maximum(w, low(co) * fine(co, n) * ka)
        return np.clip(w, 0.0, 1.0)

    def grime(co, n):
        g = np.clip(1.0 - co[:, 2] / 0.28, 0.0, 1.0) * (0.3 + 0.5 * age) + dark
        if damp:
            g = np.maximum(g, low(co) * (0.4 + 0.55 * age))
        if extra:
            g = np.maximum(g, extra(co, n))
        return np.clip(g, 0.0, 1.0)

    def special(co, n):
        return np.clip(fade + np.clip(n[:, 2], 0.0, 1.0) * (0.15 + 0.35 * age) + tone(co, n) * 0.16 * ka, 0.0, 1.0)

    paint(o, grime=grime, wear=wear, special=special)
    return o


def _streak(name, x, y, z, L, w, r):
    """Thin tapering rust run on a -Y facing surface, from (x, z) downwards."""
    n = 3
    verts, faces = [], []
    wob = r.u(-0.004, 0.004)
    for j in range(n):
        t = j / n
        hw = w / 2 * (1 - t) ** 0.6
        xx = x + wob * t * t
        verts += [(xx - hw, y, z - L * t), (xx + hw, y, z - L * t)]
    verts.append((x + wob, y, z - L))
    for j in range(n - 1):
        a0 = j * 2
        faces.append((a0 + 2, a0 + 3, a0 + 1, a0))
    faces.append((2 * n, 2 * n - 1, 2 * n - 2))
    return mesh_object(name, verts, faces)


def _patch(name, x, z, y, rx, rz, r, box_, n=9):
    """Irregular low film of algae on a -Y facing surface, kept inside box_ = (x0, x1, z0, z1)."""
    x0, x1, z0, z1 = box_
    cl = lambda px, pz: (min(max(px, x0), x1), min(max(pz, z0), z1))
    px, pz = cl(x, z)
    verts = [(px, y - 0.0026, pz)]
    for i in range(n):
        a = 2 * pi * i / n
        q = r.u(0.5, 1.15)
        px, pz = cl(x + rx * q * cos(a), z + rz * q * sin(a))
        verts.append((px, y - 0.0008, pz))
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return mesh_object(name, verts, faces)


# ------------------------------------------------------------------ foliage helpers
def _leaf_mesh(name, L, W, fold, double):
    """Leaf blade folded along the midrib, +Y = tip, +Z = upper face, mapped onto the whole leaf tile.
    The leaf material is single sided: `double` adds a back for leaves that can be seen from behind."""
    v = [(0, 0, 0), (W / 2, L * 0.45, fold), (0, L, -0.1 * L), (-W / 2, L * 0.45, fold)]
    f = [(0, 1, 2), (0, 2, 3)]
    if double:
        v = v + [(x, y, z - 0.0004) for x, y, z in v]
        f = f + [(4, 6, 5), (4, 7, 6)]
    uvs = [(0.125 + x / W * 0.2, 0.02 + y / L * 0.21) for x, y, z in v]
    return _uvmesh(name, v, f, uvs)


def _mat3(x, y, z, P, s):
    return Matrix(((x.x * s, y.x * s, z.x * s, P.x), (x.y * s, y.y * s, z.y * s, P.y),
                   (x.z * s, y.z * s, z.z * s, P.z), (0, 0, 0, 1)))


def _rvec(r):
    return Vector((r.u(-1, 1), r.u(-1, 1), r.u(-1, 1)))


def _frame(P, n, r, droop=0.3, out=0.6, scale=1.0):
    """Leaf sticking out: blade (local Y) along n with random splay, face (local Z) up / out."""
    y = n * out + _rvec(r) * 0.8 + Vector((0, 0, -droop))
    if y.length < 1e-4:
        y = Vector((1, 0, 0))
    y.normalize()
    z = n + Vector((0, 0, 0.5)) + _rvec(r) * 0.35
    z = z - y * z.dot(y)
    if z.length < 1e-4:
        z = y.orthogonal()
    z.normalize()
    return _mat3(y.cross(z), y, z, P, scale * r.u(0.75, 1.25))


def _shingle(P, n, r, tilt, down=0.6, scale=1.0):
    """Leaf lying on the clipped surface like a shingle: face (local Z) within `tilt` deg of the outward
    normal n, blade (local Y) along a random tangent biased downwards (down < 0: upwards)."""
    t = _rvec(r) + Vector((0, 0, -1.5 * down))
    t = t - n * t.dot(n)
    if t.length < 1e-4:
        t = n.orthogonal()
    t.normalize()
    a = radians(tilt)
    y = t * cos(a) + n * sin(a)
    z = n * cos(a) - t * sin(a)
    x = y.cross(z)
    b = radians(r.u(-28, 28))
    z = z * cos(b) + x * sin(b)
    return _mat3(y.cross(z), y, z, P, scale * r.u(0.75, 1.25))


def _face_out(P, n, r, scale=1.0, jit=0.3, cup=1.0):
    """4x4 with local Z along n (flower heads, fruit); cup scales local Z."""
    z = (n + _rvec(r) * jit).normalized()
    x = z.orthogonal().normalized()
    y = z.cross(x)
    a = r.u(0, 2 * pi)
    x, y = x * cos(a) + y * sin(a), y * cos(a) - x * sin(a)
    s = scale
    return Matrix(((x.x * s, y.x * s, z.x * s * cup, P.x), (x.y * s, y.y * s, z.y * s * cup, P.y),
                   (x.z * s, y.z * s, z.z * s * cup, P.z), (0, 0, 0, 1)))


def _fib(i, n, r, jit=0.6):
    z = 1 - 2 * (i + 0.5) / n
    q = sqrt(max(0.0, 1 - z * z))
    a = i * _GOLD
    return (Vector((q * cos(a), q * sin(a), z)) + _rvec(r) * (jit * sqrt(4 * pi / n))).normalized()


class _Env:
    """Clipped head: ellipsoid (R, R, R*zs) about C with a few low-frequency bulges / hollows of +-amp*R
    (hand-clipped, not turned on a lathe) and a flattened base zb below the centre."""

    def __init__(self, C, R, zs, zb, r, amp=0.07, n=6):
        self.C, self.R, self.zs, self.zb = C, R, zs, zb
        self.L = [(Vector((r.u(-1, 1), r.u(-1, 1), r.u(-0.5, 1))).normalized(), r.u(0.4, 1.0) * amp * (1 if i % 2 else -1))
                  for i in range(n)]

    def rad(self, d):
        return self.R * (1.0 + sum(a * max(0.0, d.dot(b)) ** 2 for b, a in self.L))

    def under(self, d, k=1.0):
        return d.z * self.rad(d) * k * self.zs < -self.zb * k

    def pt(self, d, k=1.0, zb=None):
        q = self.rad(d) * k
        lim = -(self.zb * k if zb is None else zb)
        return self.C + Vector((d.x * q, d.y * q, max(d.z * q * self.zs, lim)))

    def nrm(self, d, k=1.0):
        if self.under(d, k):
            return Vector((d.x * 0.35, d.y * 0.35, -1.0)).normalized()
        return Vector((d.x, d.y, d.z / self.zs)).normalized()


def _shell(name, env, k, zb, seed, material, seg=12, rings=8, rough=0.03, dark=0.0):
    """Inner foliage shell: the envelope shrunk by k, noise-displaced so it never reads as a sphere."""
    o = sphere(name, 1.0, segments=seg, rings=rings)
    transform_verts(o, lambda co: env.pt(co.normalized(), k, zb))
    displace_noise(o, strength=env.R * rough, scale=env.R * 0.3, seed=seed)
    assign(o, material)
    smooth(o, 80)
    if dark:
        paint(o, grime=dark)
    return o


def _head_leaves(env, r, n1, n2, tilt=(8, 42), down=0.6, under_skip=0.6, zmin=-1.0, droop=0.45, out=0.9):
    """Matrices for n1 single-sided shingle leaves covering the envelope and n2 double-sided leaves
    sticking out of it (they make the silhouette)."""
    m1, m2 = [], []
    for i in range(n1):
        d = _fib(i, n1, r)
        if d.z < zmin or (env.under(d) and r.chance(under_skip)):
            continue
        k = 1.0 - 0.15 * r.u(0, 1) ** 2
        m1.append(_shingle(env.pt(d, k), env.nrm(d), r, r.u(*tilt), down))
    for i in range(n2):
        d = _fib(i, n2, r, jit=0.9)
        if d.z < zmin or (env.under(d) and r.chance(under_skip)):
            continue
        m2.append(_frame(env.pt(d, r.u(0.92, 1.03)), env.nrm(d), r, droop=droop, out=out))
    return m1, m2


def _shoots(env, r, n, m2, step=0.045, scale=0.85, zlo=-0.1):
    """A few unclipped shoots standing proud of the head."""
    for _ in range(n):
        d = Vector((r.u(-1, 1), r.u(-1, 1), r.u(zlo, 1))).normalized()
        g = (d + Vector((0, 0, 0.6))).normalized()
        P = env.pt(d, 0.93)
        for j in range(r.pick([2, 3, 3])):
            P = P + g * step * r.u(0.6, 1.0)
            m2.append(_frame(P, g, r, droop=-0.5, out=1.8, scale=scale))


def _emit_leaves(name, m1, m2, L, W, material):
    out = []
    for tag, ms, dbl in (("a", m1, False), ("b", m2, True)):
        if not ms:
            continue
        lf = _leaf_mesh("leaf_base", L, W, W * 0.24, dbl)
        assign(lf, material)
        out.append(instances_to_mesh(f"{name}_{tag}", lf, ms, uv_jitter=False, keep_base=False))
    return out


def _trunk(name, base, top, r, r0, r1, sway, bark, sides=8, flare=0.7):
    """Woody stem from the soil to `top`: S-curve, taper r0 -> r1 and a root flare. Returns (obj, curve(t))."""
    d = top - base
    a = r.u(0, 2 * pi)
    b1 = Vector((cos(a), sin(a), 0)) * sway
    b2 = Vector((-sin(a), cos(a), 0)) * sway * r.u(-0.7, 0.7)

    def curve(t):
        return base + d * t + b1 * sin(2 * pi * t) * (1 - 0.35 * t) + b2 * sin(pi * t)

    ts = [0.0, 0.02, 0.05, 0.1, 0.18] + [0.18 + 0.82 * i / 8 for i in range(1, 9)]
    pts = [tuple(base + Vector((0, 0, -0.04)))] + [tuple(curve(t)) for t in ts]
    tot = path_length(pts)
    t0 = 0.04 / tot

    def sc(t):
        tt = max(0.0, (t - t0) / (1 - t0))
        return lerp(1.0, r1 / r0, tt) + flare * exp(-tt / 0.03)

    o = sweep(name, pts, r0, sides=sides, smooth_path=False, scale_fn=sc, cap=False)
    uv_transform(o, rot_deg=90)
    assign(o, bark)
    paint(o, special=mask_mul(height_ramp(base.z, base.z + 0.25), 0.7), grime=height_ramp(base.z, base.z + 0.12))
    return o, curve


def _limb(name, pts, rad, bark, sides=5, taper=0.4, n=6):
    path = bezier_path(*[tuple(q) for q in pts], n=n)
    o = sweep(name, path, rad, sides=sides, smooth_path=False, scale_fn=lambda t: 1.0 - (1 - taper) * t, cap=False)
    uv_transform(o, rot_deg=90)
    assign(o, bark)
    return o


def _stake(p, r, base, curve, h, wood):
    """Cleft chestnut stake (bare, weathered wood) driven in beside the stem, tied with a figure of eight of wire."""
    a = r.u(0, 2 * pi)
    off = Vector((cos(a), sin(a), 0)) * 0.07
    lean = Vector((r.u(-0.012, 0.012), r.u(-0.012, 0.012), 0))
    b = base + off
    t_ = b + Vector((0, 0, h)) + lean
    st = sweep("stake", [tuple(b + Vector((0, 0, -0.04))), tuple(b.lerp(t_, 0.5) + lean * 0.3), tuple(t_)],
               [(-0.009, -0.007), (0.009, -0.007), (0.009, 0.007), (-0.009, 0.007)], smooth_path=False, scale_fn=lambda t: 1.0 - 0.2 * t)
    recalc_normals(st)
    smooth(st, 30)
    uv_grain(st, "Z")
    assign(st, wood)
    paint(st, wear=1.0, special=0.5, grime=mask_mul(height_ramp(b.z + 0.2, b.z), 0.7))
    tt = 0.8 * h / max(1e-3, (curve(1.0) - base).z)
    c = curve(clamp(tt, 0.1, 0.95))
    s = b.lerp(t_, 0.8)
    q = (c - s)
    q.z = 0
    qn = q.normalized()
    side = Vector((-qn.y, qn.x, 0)) * 0.03
    tie = sweep("tie", [tuple(s - side * 0.4), tuple(c - side + qn * 0.01), tuple(c + qn * 0.032),
                        tuple(c + side + qn * 0.01), tuple(s + side * 0.4), tuple(s - qn * 0.014)],
                0.0028, sides=4, closed=True)
    assign(tie, mat("iron_wrought"))
    paint(tie, special=0.5)
    return st


def _ground_cover(p, r, base, wi, leaf_mat, L=0.05):
    """Moss cushions on the soil, a few weeds and fallen leaves, so the tub does not look freshly potted."""
    for i in range(3):
        a = r.u(0, 2 * pi)
        d = wi * r.u(0.5, 0.82)
        rad = wi * r.u(0.12, 0.2)
        o = sphere(f"moss{i}", rad, loc=(d * cos(a), d * sin(a), base.z - 0.012), segments=7, rings=4, scale=(1, r.u(0.6, 1), 0.3))
        displace_noise(o, strength=rad * 0.12, scale=rad * 0.5, seed=p.seed + 40 + i)
        assign(o, mat("moss"))
        smooth(o, 80)
    ms = []
    for i in range(int(4 + 8 * p.age)):                              # fallen leaves
        a = r.u(0, 2 * pi)
        d = wi * r.u(0.25, 0.9)
        ms.append(trs((d * cos(a), d * sin(a), base.z - 0.012 * (d / wi) ** 2), (r.u(-14, 14), r.u(-14, 14), r.u(0, 360)), r.u(0.8, 1.1)))
    for i in range(3):                                               # weeds
        a = r.u(0, 2 * pi)
        d = wi * r.u(0.45, 0.85)
        P = Vector((d * cos(a), d * sin(a), base.z - 0.02))
        for j in range(5):
            aa = r.u(0, 2 * pi)
            ms.append(_frame(P, Vector((cos(aa) * 0.6, sin(aa) * 0.6, 1)).normalized(), r, droop=-0.2, out=1.5, scale=0.75))
    _emit_leaves("soil_leaves", [], ms, L, L * 0.5, leaf_mat)


# ------------------------------------------------------------------ build
def build(p, rng):
    S = p.size
    half = S / 2
    Hb = clamp(S * p.proportion, 0.34, 1.15)          # top of the capping board
    age = p.age
    ka = 0.3 + 0.7 * age
    nb = int(p.bands)
    pw = clamp(0.042 + S * 0.052, 0.06, 0.1)          # post section
    foot = clamp(0.045 + S * 0.03, 0.05, 0.08)        # clear height under the bottom rail
    tr = 0.034                                        # rail thickness
    hr = clamp(Hb * 0.12, 0.055, 0.1)                 # top rail height
    hrb = hr * 1.2                                    # bottom rail height
    cap_t = 0.022                                     # capping board
    setb = 0.0025                                     # rails sit a hair behind the post faces
    rec = 0.013                                       # boards recessed behind the rail faces
    Lr = S - 2 * pw                                   # clear span between posts
    wi = half - pw                                    # half width of the planting opening
    zt = Hb + 0.03                                    # post tops
    z0p, z1p = foot + hrb, Hb - cap_t - hr            # panel opening
    ph = z1p - z0p
    zc = (z0p + z1p) / 2
    y_rail = -(half - setb)                           # rail face
    y_pl = y_rail + rec                               # board face
    sd = p.seed * 17 + 3

    wood = mat("wood_painted", p.paint, rough=0.8)
    iron = mat("iron_wrought")
    zinc = mat("zinc")
    moss = mat("moss")
    W, I, RUST, ALG = [], [], [], []                  # (weathered) timber / ironwork / rust runs / algae
    shade = (0.25, 0.75, 1.0, 0.7)                     # how damp each side is (front, +X, back, -X)

    # ---- corner posts: square, chamfered arrises, standing on their own feet, moulded cap + finial
    rp = rng.fork("posts")
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        cx, cy = sx * (half - pw / 2), sy * (half - pw / 2)
        h = zt + rp.u(-0.003, 0.003)
        zs_ = [0.0, foot * 0.45, foot] + [foot + (h - foot) * j / 6 for j in range(1, 7)]
        post = _post(f"post{i}", cx, cy, pw, zs_, 0.0075, rp)
        _weather(post, sd + i, age, edge_k=1.0, flake_k=0.3, damp=(foot + 0.045, 0.09), dark=rp.u(0, 0.08), fade=rp.u(0, 0.2))
        capm = box(f"postcap{i}", (pw + 0.016, pw + 0.016, 0.013), loc=(cx, cy, h + 0.0065), bevel=0.004, segments=1)
        _weather(capm, sd + 10 + i, age, e_const=0.85, fade=0.12)
        W += [post, capm]
        zb = h + 0.013
        if p.finials:
            rb = pw * 0.6
            prof = [(pw * 0.42, 0.0), (pw * 0.44, 0.005), (pw * 0.36, 0.012), (pw * 0.21, 0.02), (pw * 0.21, 0.028),
                    (pw * 0.28, 0.035), (pw * 0.2, 0.044)]
            zcb = 0.044 + rb * 0.93
            for j in range(6):
                a = radians(-66 + j * (154 / 5))
                prof.append((rb * cos(a), zcb + rb * sin(a)))
            prof += [(rb * 0.17, zcb + rb * 1.05), (0.0, zcb + rb * 1.12)]
            fin = lathe(f"finial{i}", prof, segments=12, loc=(cx, cy, zb), rot=(0, 0, rp.u(0, 90)), smooth_angle=50)
            _weather(fin, sd + 20 + i, age, e_const=0.6, flake_k=0.7, fade=0.08)
        else:
            rr = (pw + 0.004) / 2 * sqrt(2)
            fin = lathe(f"pyr{i}", [(rr, 0.0), (rr, 0.006), (0.0, 0.006 + pw * 0.32)], segments=4, loc=(cx, cy, zb),
                        rot=(0, 0, 45), smooth_angle=20)
            _weather(fin, sd + 20 + i, age, e_const=0.9, flake_k=0.6, fade=0.1)
        W.append(fin)

    # ---- per side: rails, capping, panel infill, iron straps
    nail = lathe("nail", [(0.0062, 0.0), (0.0, 0.0046)], segments=5, smooth_angle=20)   # forged rose-head nail
    pin = _dome("pin", 0.0095, 0.0075, 6)
    assign(nail, iron)
    assign(pin, iron)
    nail_m, pin_m, peg_m = [], [], []
    sw = clamp(0.024 + S * 0.012, 0.028, 0.04)        # strap width
    st = 0.0045                                       # strap thickness
    xa = half - pw * 0.3
    y_post = -half - st / 2
    for k in range(4):
        rs = rng.fork(f"side{k}")
        sk = sd + 100 * (k + 1)
        side, irons, runs, alg = [], [], [], []
        joggle = p.panel == "planks"
        if joggle:
            rel = {0: [], 1: [0.5], 2: [0.2, 0.8], 3: [0.15, 0.5, 0.85]}[nb]
            zs = [z0p + ph * t + rs.u(-0.003, 0.003) for t in rel]
        else:
            zs = [zq + rs.u(-0.002, 0.002) for zq in [foot + hrb / 2, Hb - cap_t - hr / 2][:nb]]

        top = _strip(f"rail_t{k}", (0, Hb - cap_t - hr / 2), (1, 0), -Lr / 2 + 0.0004, Lr / 2 - 0.0004, hr / 2, y_rail + tr, y_rail, r=rs)
        _weather(top, sk + 1, age, edge_k=0.8, flake_k=0.35, fade=rs.u(0, 0.18), dark=rs.u(0, 0.08))
        yb_ = y_rail + rs.u(-0.0008, 0.0008)
        bot = _strip(f"rail_b{k}", (0, foot + hrb / 2), (1, 0), -Lr / 2 + 0.0004, Lr / 2 - 0.0004, hrb / 2, yb_ + tr, yb_, r=rs)
        _weather(bot, sk + 2, age, edge_k=1.0, flake_k=0.7, damp=(foot + hrb * 1.15, hrb * 1.3), dark=0.1 * shade[k])
        y0c, y1c = -(half + 0.005), -(half - pw) - 0.0015
        cap = _strip(f"cap{k}", (0, (y0c + y1c) / 2), (1, 0), -Lr / 2 + 0.0004, Lr / 2 - 0.0004, (y1c - y0c) / 2, -(Hb - cap_t), -Hb,
                     c=0.005, r=rs, xf=lambda x, y, z: (x, z, -y))
        _weather(cap, sk + 3, age, edge_k=1.0, flake_k=0.6, fade=0.08 + rs.u(0, 0.15))
        side += [top, bot, cap]

        nails = []                                    # (x, z, y of the board face or None) of this side's strap nails
        if p.panel in ("planks", "cross"):
            n = max(3, int(round(Lr / 0.104)))
            ws = [rs.u(0.85, 1.15) for _ in range(n)]
            tot = sum(ws)
            x = -Lr / 2
            for j, wj in enumerate(ws):
                wdt = wj / tot * Lr
                xc = x + wdt / 2
                hwb = wdt / 2
                yf = y_pl + rs.u(-0.0014, 0.0014)
                prof = [(-hwb + 0.0006, yf + 0.009, 0.5), (-hwb + 0.0034, yf, 0.6), (0.0, yf, 0.0),
                        (hwb - 0.0034, yf, 0.6), (hwb - 0.0006, yf + 0.009, 0.5)]
                b = _strip(f"board{k}_{j}", (xc, zc), (0, 1), -(ph / 2 + 0.005), ph / 2 + 0.005, hwb, yf + 0.009, yf, nseg=5, prof=prof,
                           r=rs, end=0.02)
                xn = xc + rs.u(-0.004, 0.004)
                run_k = rs.u(0.5, 1.0)

                def runs_fn(co, nrm, xn=xn, hwb=hwb, run_k=run_k, zq_=tuple(zs) if joggle else ()):
                    g = np.zeros(len(co))
                    cen = 1.0 - 0.6 * np.clip(np.abs(co[:, 0] - xn) / hwb, 0.0, 1.0)
                    for zq in zq_:
                        dzz = (zq - sw / 2) - co[:, 2]
                        g = np.maximum(g, np.where((dzz > -sw * 1.1) & (dzz < 0.12), np.clip(1.0 - dzz / 0.12, 0.0, 1.0), 0.0) * cen)
                    return g * (0.3 + 0.65 * age) * run_k

                _weather(b, sk + 10 + j, age, edge_k=0.45, flake_k=0.3 + 0.3 * rs.u(0, 1), damp=(z0p + 0.035 + 0.06 * rs.u(0, 1), 0.08),
                         fade=rs.u(0.0, 0.26), dark=rs.u(0.0, 0.14), extra=runs_fn)
                side.append(b)
                if joggle:
                    for zq in zs:
                        nails.append((xn, zq + rs.u(-0.003, 0.003), yf))
                x += wdt
        else:
            yg = y_pl + 0.004
            ground = box(f"ground{k}", (Lr + 0.004, 0.016, ph + 0.008), loc=(0, yg + 0.008, zc))
            densify(ground, max_edge=max(0.1, min(Lr, ph) / 3.2))
            uv_grain(ground, "Z")
            m = min(0.055, 0.2 * min(Lr, ph))
            a, b2, ins, d = Lr / 2 - m, ph / 2 - m, min(0.03, m * 0.6), 0.012
            v = [(-a, yg, zc - b2), (a, yg, zc - b2), (a, yg, zc + b2), (-a, yg, zc + b2),
                 (-a + ins, yg - d, zc - b2 + ins), (a - ins, yg - d, zc - b2 + ins),
                 (a - ins, yg - d, zc + b2 - ins), (-a + ins, yg - d, zc + b2 - ins)]
            field = mesh_object(f"field{k}", v, [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            recalc_normals(field)
            bevel(field, 0.002, segments=1)
            densify(field, max_edge=max(0.08, min(Lr, ph) / 4))
            uv_grain(field, "Z")
            e = 0.009
            bead = sweep(f"bead{k}", [(-Lr / 2 + e, yg, z0p + e), (Lr / 2 - e, yg, z0p + e), (Lr / 2 - e, yg, z1p - e),
                                      (-Lr / 2 + e, yg, z1p - e)], 0.0085, closed=True, sides=6, smooth_path=False)
            _weather(ground, sk + 10, age, e_const=0.25, flake_k=0.45, damp=(z0p + 0.06, 0.08), fade=rs.u(0, 0.3))
            _weather(field, sk + 11, age, e_const=0.3, flake_k=0.45, damp=(z0p + 0.1, 0.1), fade=rs.u(0.05, 0.25), dark=rs.u(0, 0.1))
            _weather(bead, sk + 12, age, e_const=0.75, flake_k=0.5, damp=(z0p + 0.05, 0.05))
            side += [ground, field, bead]

        if p.panel == "cross":
            # saltire brace: mitred into the frame corners (2.5 mm clear of rails and posts), half-lapped at the crossing
            hyp = hypot(Lr, ph)
            g = 0.0025
            rect = [(1, 0, Lr / 2 - g), (-1, 0, Lr / 2 - g), (0, 1, z1p - g), (0, -1, -(z0p + g))]
            bw = clamp(0.03 + S * 0.03, 0.04, 0.062) / 2
            flip = 1 if k % 2 == 0 else -1
            dA = (Lr / hyp, flip * ph / hyp)
            dB = (Lr / hyp, -flip * ph / hyp)
            nA = (-dA[1], dA[0])
            off = nA[1] * zc
            yfb = y_pl - 0.0104
            brs = [_strip(f"brace{k}_a", (0, zc), dA, -hyp, hyp, bw, y_pl, yfb, c=0.003, nseg=8, planes=rect, r=rs)]
            for j, sg in enumerate((1, -1)):
                lap = (-sg * nA[0], -sg * nA[1], -sg * off - bw - 0.0012)
                brs.append(_strip(f"brace{k}_b{j}", (0, zc), dB, -hyp, hyp, bw, y_pl, yfb + 0.0006, c=0.003, nseg=4,
                                  planes=rect + [lap], r=rs))
            tone_f, tone_d = rs.u(0.05, 0.35), rs.u(0, 0.08)
            for j, br in enumerate(brs):
                _weather(br, sk + 40 + j, age, edge_k=0.9, flake_k=0.4, damp=(z0p + 0.06, 0.08), fade=tone_f, dark=tone_d)
            side += brs
            irons.append(lathe(f"boss{k}", [(0.021, 0.0), (0.022, 0.003), (0.016, 0.0055), (0.009, 0.0065), (0.008, 0.0105), (0.0, 0.012)],
                               segments=8, loc=(0, yfb, zc), rot=(90, 0, 0)))
            runs.append(_streak(f"run_boss{k}", rs.u(-0.004, 0.004), yfb - 0.0007, zc - 0.021, bw * 1.2 * ka, 0.012, rs))

        # iron strap bands: flat bar pinned to both posts; on plain boards it joggles back and is nailed to each board
        y_in = y_pl - st / 2
        for j, zz in enumerate(zs):
            if joggle:
                path = [(-xa, y_post, zz), (-Lr / 2 + 0.001, y_post, zz), (-Lr / 2 + 0.016, y_in, zz), (0, y_in, zz),
                        (Lr / 2 - 0.016, y_in, zz), (Lr / 2 - 0.001, y_post, zz), (xa, y_post, zz)]
            else:
                path = [(-xa, y_post, zz), (-Lr / 4, y_post, zz), (Lr / 4, y_post, zz), (xa, y_post, zz)]
                nq = max(3, int(round(Lr / 0.13)))
                for q in range(nq):
                    nails.append(((-0.5 + (q + 0.5) / nq) * (Lr - 0.04) + rs.u(-0.004, 0.004), zz + rs.u(-0.002, 0.002), None))
            strap = sweep(f"strap{k}_{j}", path, [(-st / 2, -sw / 2), (st / 2, -sw / 2), (st / 2, sw / 2), (-st / 2, sw / 2)],
                          smooth_path=False, up=(0, 0, 1))
            recalc_normals(strap)
            smooth(strap, 30)
            irons.append(strap)
            for sx in (-1, 1):
                irons.append(lathe(f"eye{k}_{j}", [(sw * 0.62, 0.0), (sw * 0.56, st * 1.2), (0.0, st * 1.3)], segments=7,
                                   loc=(sx * xa, -half, zz), rot=(90, 0, 0)))
                pin_m.append(_rz(k) @ trs((sx * xa, -half - st * 1.2, zz), (90, 0, 0)))
                if rs.chance(0.45 + 0.5 * age):       # rust run down the post under the pin
                    Lrun = min(zz - 0.03, rs.u(0.05, 0.16) * (0.5 + age))
                    if Lrun > 0.012:
                        runs.append(_streak(f"run_pin{k}", sx * xa + rs.u(-0.004, 0.004), -half - 0.0007, zz - sw * 0.64, Lrun, rs.u(0.009, 0.015), rs))
        for xn, zn, yf in nails:
            ys = (y_in - st / 2) if yf is not None else (y_post - st / 2)
            nail_m.append(_rz(k) @ trs((xn, ys, zn), (90, 0, 0)))
            if rs.chance(0.3 + 0.6 * age):            # rust run under the nail (down the board / down the rail under the strap)
                if yf is not None:
                    Lrun = min(zn - sw / 2 - z0p - 0.004, rs.u(0.035, 0.13) * (0.5 + age))
                    ysurf = yf
                else:
                    zlow = foot if zn < zc else z1p
                    Lrun = min(zn - sw / 2 - zlow - 0.006, rs.u(0.02, 0.05))
                    ysurf = y_rail + 0.0009
                if Lrun > 0.012:
                    runs.append(_streak(f"run{k}", xn + rs.u(-0.002, 0.002), ysurf - 0.0007, zn - sw / 2 - 0.0015, Lrun, rs.u(0.006, 0.011), rs))

        # draw-bored oak pegs through the mortise-and-tenon joints (where no strap covers them)
        covered = [] if joggle else zs
        for zj, hj in ((foot + hrb / 2, hrb), (Hb - cap_t - hr / 2, hr)):
            if any(abs(zj - zq) < 0.02 for zq in covered):
                continue
            for sx in (-1, 1):
                for dzp in (-0.24, 0.24):
                    peg_m.append(_rz(k) @ trs((sx * (half - pw * 0.42) + rs.u(-0.003, 0.003), -half, zj + dzp * hj + rs.u(-0.003, 0.003)), (90, 0, 0)))

        # drop-ring lifting handles on the two flanks
        if p.handles and k in (1, 3):
            zh = Hb - cap_t - hr / 2
            rr = clamp(S * 0.062, 0.034, 0.06)
            ys = y_rail - (st + 0.0025 if (not joggle and nb >= 2) else 0.0)
            irons.append(lathe(f"hplate{k}", [(0.026, 0.0), (0.027, 0.004), (0.02, 0.0075), (0.012, 0.008), (0.0, 0.008)], segments=10,
                               loc=(0, ys, zh), rot=(90, 0, 0)))
            irons.append(torus(f"heye{k}", 0.011, 0.0042, loc=(0, ys - 0.012, zh), rot=(0, 90, 0), major_seg=8, minor_seg=5))
            irons.append(torus(f"hring{k}", rr, 0.0055, loc=(0, ys - 0.015, zh - rr + 0.004), rot=(90 + rs.u(3, 8), 0, 0), major_seg=18, minor_seg=6))

        # algae film creeping up the feet and along the bottom rail, heaviest on the damp sides
        dampk = shade[k] * (0.35 + 0.65 * age)
        for q in range(int(round(10 * dampk))):
            if q < 2:                                  # at the feet first
                sx = (-1, 1)[q]
                xp = sx * (half - pw / 2)
                alg.append(_patch(f"alg{k}_{q}", xp + rs.u(-0.012, 0.012), rs.u(0.0, 0.025), -half,
                                  pw * rs.u(0.3, 0.55), rs.u(0.04, 0.08) * (0.6 + dampk), rs,
                                  (xp - pw / 2 + 0.008, xp + pw / 2 - 0.008, 0.002, foot + hrb * 2)))
            else:
                alg.append(_patch(f"alg{k}_{q}", rs.u(-0.46, 0.46) * Lr, foot + hrb * rs.u(0.0, 0.3), y_rail,
                                  rs.u(0.04, 0.1), hrb * rs.u(0.25, 0.5), rs,
                                  (-Lr / 2 + 0.003, Lr / 2 - 0.003, foot + 0.005, foot + hrb - 0.006)))

        for o in side + irons + runs + alg:
            _yaw(o, k)
        W += side
        I += irons
        RUST += runs
        ALG += alg

    # numbered tally board on the front rail (orangery tubs are numbered): bare oak, forged iron numerals, two nails
    if p.label.strip():
        zpl = Hb - cap_t - hr / 2
        on_strap = p.panel != "planks" and nb >= 2
        ys = (-half - st) if on_strap else y_rail
        ph_ = min(hr * 0.66, 0.052)
        pw_ = clamp(ph_ * (1.15 + 0.42 * len(p.label.strip())), 0.05, Lr * 0.6)
        plate = box("tally", (pw_, 0.006, ph_), loc=(0, ys - 0.003, zpl), bevel=0.0015, segments=1)
        uv_grain(plate, "X")
        paint(plate, wear=1.0, special=0.6, grime=mask_mul(noise_mask(0.03, seed=p.seed + 8, lo=0.4, hi=0.9), 0.15 + 0.3 * age))
        W.append(plate)
        num = text("tally_no", p.label.strip(), size=ph_ * 0.6, depth=0.0016, loc=(0, ys - 0.0064, zpl - ph_ * 0.27), font="serif")
        I.append(num)
        for sx in (-1, 1):
            nail_m.append(trs((sx * (pw_ / 2 - 0.006), ys - 0.006, zpl + rng.u(-0.004, 0.004)), (90, 0, 0), 0.7))

    if nail_m:
        I.append(instances_to_mesh("nails", nail, nail_m, keep_base=False))
    else:
        delete(nail)
    if pin_m:
        I.append(instances_to_mesh("pins", pin, pin_m, keep_base=False))
    else:
        delete(pin)
    peg = lathe("peg", [(0.0078, 0.0), (0.0, 0.003)], segments=6)
    if peg_m:
        pegs = instances_to_mesh("pegs", peg, peg_m, keep_base=False)
        paint(pegs, wear=0.35 + 0.5 * age, grime=0.2)
        W.append(pegs)
    else:
        delete(peg)

    # ---- floor boards (seen under the bottom rail), zinc liner lip, soil
    floor = box("floor", (2 * wi + 0.03, 2 * wi + 0.03, 0.022), loc=(0, 0, foot + 0.018))
    paint(floor, grime=0.7 + 0.3 * age, wear=0.4 * ka)
    W.append(floor)
    liner = shape_extrude("liner", rounded_rect(2 * wi + 0.004, 2 * wi + 0.004, 0.004, 2),
                          [rounded_rect(2 * wi - 0.006, 2 * wi - 0.006, 0.003, 2)], depth=0.1, loc=(0, 0, Hb - 0.095))
    assign(liner, zinc)
    paint(liner, special=mask_mul(noise_mask(0.05, seed=p.seed + 9, lo=0.45, hi=0.85), 0.15 + 0.35 * age), grime=0.2)
    z_soil = Hb - 0.05
    soil = grid("soil", size=(2 * wi - 0.004, 2 * wi - 0.004), cuts=(7, 7), loc=(0, 0, z_soil))
    apply_transform(soil)
    displace_noise(soil, strength=0.009, scale=0.09, seed=p.seed + 1, axis_mask=(0, 0, 1))
    transform_verts(soil, lambda co: Vector((co.x, co.y, co.z + 0.028 * max(0.0, 1 - (co.x ** 2 + co.y ** 2) / (wi * wi)))))
    assign(soil, mat("soil"))
    smooth(soil, 60)
    paint(soil, special=mask_mul(noise_mask(0.15, seed=p.seed + 2), 0.6))

    # ---- finishes
    for o in W:
        assign(o, wood)
    for o in I:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.045, seed=p.seed + 3, lo=0.3, hi=0.75), 0.25 + 0.75 * age),
              wear=mask_mul(noise_mask(0.03, seed=p.seed + 6, lo=0.55, hi=0.8), 0.5))
    for o in RUST:
        assign(o, iron)
        paint(o, special=1.0, grime=0.15)
    for o in ALG:
        assign(o, moss)
        paint(o, grime=0.3)

    # ---- planting
    sc = p.plant_scale
    rpl = rng.fork("plant")
    base = Vector((rpl.u(-0.02, 0.02), rpl.u(-0.02, 0.02), z_soil + 0.024))
    if p.planting == "tree":
        _tree(p, rpl, base, sc, wi)
    elif p.planting == "cone":
        _cone(p, rpl, base, sc, wi)
    elif p.planting == "ball":
        _ball(p, rpl, base, sc, wi)
    else:
        _flowers(p, rpl, base, sc, wi, half, Hb)

    socket("top", "attach", (0, 0, Hb), role="prop_top")


# ------------------------------------------------------------------ plants
def _tree(p, r, base, sc, wi):
    """Standard orange tree: clear stem, forking scaffold, one clipped head ~0.8 m across."""
    bark = mat("bark", scale=5.0)
    leafm = mat("leaf", (0.85, 1.05, 0.7), rough=0.6)
    stem_h = 0.6 * sc
    R = 0.395 * sc
    zs = 0.9
    zb = 0.66 * R * zs
    kr = sc ** 0.8
    top = base + Vector((r.u(-0.04, 0.04) * sc, r.u(-0.04, 0.04) * sc, stem_h))
    r0, r1 = 0.0275 * kr, 0.0175 * kr
    _, curve = _trunk("trunk", base, top, r, r0, r1, 0.022 * sc, bark)
    a = 0.0
    for i in range(3):                                # surface roots gripping the soil
        a = r.u(0, 2 * pi) if i == 0 else a + radians(r.u(95, 145))
        dv = Vector((cos(a), sin(a), 0))
        _limb(f"root{i}", [base + dv * r0 * 0.5 + Vector((0, 0, 0.035)), base + dv * r0 * 1.6 + Vector((0, 0, 0.02)),
                           base + dv * r0 * 3.0 + Vector((0, 0, -0.004)), base + dv * r0 * r.u(4.0, 5.5) + Vector((0, 0, -0.03))],
              r0 * 0.5, bark, sides=5, taper=0.35, n=4)
    C = top + Vector((r.u(-0.02, 0.02), r.u(-0.02, 0.02), zb - 0.035 * sc))
    env = _Env(C, R, zs, zb, r, amp=0.075)
    # scaffold: 3-5 limbs forking from the top of the stem, each forking again inside the head
    nbr = r.pick([3, 4, 4, 5])
    a0 = r.u(0, 2 * pi)
    for i in range(nbr):
        az = a0 + i * 2 * pi / nbr + r.u(-0.3, 0.3)
        el = radians(r.u(28, 52))
        dv = Vector((cos(az) * cos(el), sin(az) * cos(el), sin(el)))
        ln = R * r.u(0.72, 0.92)
        tip = top + dv * ln
        mid = top + dv * ln * 0.5 + Vector((0, 0, 0.03 * sc))
        _limb(f"branch{i}", [top + Vector((0, 0, -0.04)), top + Vector((0, 0, 0.035 * sc)) + dv * ln * 0.12, mid, tip], r1 * 0.66, bark, n=6)
        az2 = az + r.pick([-1, 1]) * r.u(0.6, 1.0)
        el2 = radians(r.u(35, 70))
        dv2 = Vector((cos(az2) * cos(el2), sin(az2) * cos(el2), sin(el2)))
        _limb(f"fork{i}", [top + dv * ln * 0.36 + Vector((0, 0, 0.02 * sc)), mid + dv2 * ln * 0.1, mid + dv2 * ln * 0.3, mid + dv2 * ln * 0.55],
              r1 * 0.42, bark, sides=4, n=4)
    _shell("crown_shell", env, 0.8, zb * 0.5, p.seed + 20, mat("moss"), seg=12, rings=8, rough=0.05, dark=0.1)
    dens = sc ** 0.2
    m1, m2 = _head_leaves(env, r, int(1900 * dens), int(450 * dens), tilt=(8, 40), down=0.7, under_skip=0.5, droop=0.8, out=0.45)
    _shoots(env, r, int(4 + 2 * sc), m2, step=0.036 * sc ** 0.6, scale=0.9)
    L = 0.056 * sc ** 0.85
    _emit_leaves("tree_leaves", m1, m2, L, L * 0.52, leafm)
    nf = int(p.blooms)
    if nf > 0:
        fr = sphere("fruit_base", 0.031, segments=7, rings=4, scale=(1, 1, 0.93))
        assign(fr, mat("wax", _soften(p.bloom, 0.85, 0.9), rough=0.9))
        smooth(fr, 80)
        fm = []
        for i in range(nf):
            a = r.u(0, 2 * pi)
            z = r.u(-0.6, 0.4)
            q = sqrt(1 - z * z)
            d = Vector((q * cos(a), q * sin(a), z))
            fm.append(_face_out(env.pt(d, r.u(0.93, 1.0)) + Vector((0, 0, -0.012)), env.nrm(d), r, scale=r.u(0.85, 1.12)))
        if nf > 3 and p.age > 0.3:                    # a windfall on the soil
            a = r.u(0, 2 * pi)
            fm.append(_face_out(Vector((wi * 0.6 * cos(a), wi * 0.6 * sin(a), base.z + 0.012)), Vector((0, 0, 1)), r, scale=0.95))
        instances_to_mesh("fruit", fr, fm, keep_base=False)
    if p.stake:
        _stake(p, r, base, curve, stem_h * 0.93, mat("wood_painted", p.paint, rough=0.8))
    _ground_cover(p, r, base, wi, leafm, L=L)


def _cone(p, r, base, sc, wi):
    """Clipped box cone on a short woody stem: slightly bellied, leaning a touch, hand-clipped lumps."""
    bark = mat("bark", scale=5.0)
    leafm = mat("leaf", (1.1, 1.15, 1.0), rough=0.75)
    z0 = base.z + 0.1 * sc
    hc = 1.12 * sc
    rb = min(wi * 0.98, 0.33 * sc)
    _trunk("cone_stem", base, Vector((base.x + r.u(-0.01, 0.01), base.y + r.u(-0.01, 0.01), z0 + 0.12 * sc)), r, 0.02 * sc ** 0.6, 0.015 * sc ** 0.6,
           0.006, bark, sides=7, flare=0.6)
    lean = Vector((r.u(-0.03, 0.03), r.u(-0.03, 0.03), 0)) * sc
    ph0 = r.u(0, 2 * pi)
    lumps = [(Vector((r.u(-1, 1), r.u(-1, 1), r.u(-0.8, 0.8))).normalized(), r.u(0.45, 1.0) * (1 if i % 2 else -1)) for i in range(7)]

    def prof(v):
        if v < 0.06:
            f = v / 0.06
            return rb * 0.86 * f, z0 + 0.012 * f * f, 0.0
        if v < 0.16:
            a = (v - 0.06) / 0.10 * pi / 2
            return rb * (0.86 + 0.14 * sin(a)), z0 + 0.012 + 0.07 * sc * (1 - cos(a)), 0.0
        t = (v - 0.16) / 0.84
        return rb * ((1 - t) ** 0.88) * (1 + 0.06 * sin(pi * t)) + 0.012 * t, z0 + 0.012 + 0.07 * sc + (hc - 0.082 * sc) * t, t

    def fn(u, v, k=1.0):
        a = 2 * pi * u
        rr, z, t = prof(v)
        d = Vector((cos(a), sin(a), 1.6 * (t - 0.45))).normalized()
        bump = sum(am * max(0.0, d.dot(b)) ** 2 for b, am in lumps) * 0.03 * sc * min(1.0, rr / (rb * 0.5))
        rr = (rr + bump) * k
        ax = lean * t + Vector((cos(ph0), sin(ph0), 0)) * 0.012 * sc * sin(2 * pi * t)
        return (base.x + ax.x + rr * cos(a), base.y + ax.y + rr * sin(a), z)

    shell = surface("cone_shell", lambda u, v: fn(u, v, 0.955), nu=16, nv=15, closed_u=True)
    recalc_normals(shell)
    displace_noise(shell, strength=0.009, scale=0.07, seed=p.seed + 21)
    assign(shell, mat("hedge", _HEDGE))
    smooth(shell, 80)
    slope = atan2(rb, hc)
    m1, m2 = [], []
    dens = sc ** 1.3
    for i in range(int(900 * dens)):
        v = 1 - 0.84 * sqrt(r.u(0, 1)) if r.chance(0.94) else r.u(0.02, 0.16)
        u = r.u(0, 1)
        a = 2 * pi * u
        P = Vector(fn(u, v, 1.0))
        if v < 0.07:
            nn = Vector((cos(a) * 0.3, sin(a) * 0.3, -1)).normalized()
        else:
            nn = Vector((cos(a) * cos(slope), sin(a) * cos(slope), sin(slope)))
        if i % 3:
            m1.append(_shingle(P, nn, r, r.u(12, 48), down=-0.5))
        else:
            m2.append(_frame(P, nn, r, droop=-0.3, out=0.9))
    tipP = Vector(fn(0, 1.0))
    for j in range(4):                                # the leader the shears missed + a few stray shoots
        m2.append(_frame(tipP + Vector((0, 0, 0.02 * j)), Vector((0, 0, 1)), r, droop=-0.6, out=2.0, scale=0.9))
    for _ in range(5):
        u, v = r.u(0, 1), r.u(0.25, 0.85)
        P = Vector(fn(u, v, 1.0))
        g = Vector((cos(2 * pi * u) * 0.6, sin(2 * pi * u) * 0.6, 0.8))
        for j in range(r.pick([2, 3])):
            P = P + g * 0.03
            m2.append(_frame(P, g, r, droop=-0.5, out=1.8, scale=0.85))
    _emit_leaves("cone_leaves", m1, m2, 0.038, 0.021, leafm)
    _ground_cover(p, r, base, wi, leafm, L=0.04)


def _ball(p, r, base, sc, wi):
    """Clipped box ball on a round, slightly crooked stem with a stake."""
    bark = mat("bark", scale=5.0)
    leafm = mat("leaf", (1.1, 1.15, 1.0), rough=0.75)
    stem_h = 0.52 * sc
    R = 0.285 * sc
    zs = 0.93
    zb = 0.74 * R * zs
    top = base + Vector((r.u(-0.03, 0.03), r.u(-0.03, 0.03), stem_h))
    kr = sc ** 0.7
    _, curve = _trunk("ball_stem", base, top + Vector((0, 0, 0.1 * sc)), r, 0.0175 * kr, 0.0135 * kr, 0.014 * sc, bark, sides=8, flare=0.6)
    C = top + Vector((r.u(-0.012, 0.012), r.u(-0.012, 0.012), zb - 0.02))
    env = _Env(C, R, zs, zb, r, amp=0.1)
    for i in range(3):                                # short limbs entering the head, seen from below
        az = r.u(0, 2 * pi)
        dv = Vector((cos(az) * 0.75, sin(az) * 0.75, 0.66))
        _limb(f"ball_limb{i}", [top + Vector((0, 0, -0.03)), top + dv * R * 0.15, top + dv * R * 0.4, top + dv * R * 0.75], 0.009 * kr, bark,
              sides=4, n=4)
    _shell("ball_shell", env, 0.945, zb * 0.9, p.seed + 22, mat("hedge", _HEDGE), seg=16, rings=10, rough=0.035)
    dens = sc ** 1.3
    m1, m2 = _head_leaves(env, r, int(950 * dens), int(210 * dens), tilt=(10, 42), down=-0.3, under_skip=0.5, droop=0.0, out=0.6)
    _shoots(env, r, 7, m2, step=0.03, scale=0.85)
    _emit_leaves("ball_leaves", m1, m2, 0.034, 0.02, leafm)
    if p.stake:
        _stake(p, r, base, curve, stem_h * 0.95, mat("wood_painted", p.paint, rough=0.8))
    _ground_cover(p, r, base, wi, leafm, L=0.04)


def _flower_mesh(name, rad, petals=5):
    """Cupped corolla: notched petals rising from a sunk throat; double sided."""
    n = petals * 2
    v = [(0, 0, 0)]
    for i in range(n):
        a = 2 * pi * i / n
        rr = rad if i % 2 == 0 else rad * 0.8
        v.append((rr * cos(a), rr * sin(a), rad * (0.5 if i % 2 == 0 else 0.36)))
    f = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    nv = len(v)
    v = v + [(x, y, z - 0.0006) for x, y, z in v]
    f = f + [(nv, nv + 1 + (i + 1) % n, nv + 1 + i) for i in range(n)]
    return _uvmesh(name, v, f, [(0.125 + x / rad * 0.04, 0.125 + y / rad * 0.04) for x, y, z in v])


def _flowers(p, r, base, sc, wi, half, Hb):
    """Mound of bedding plants: leafy cushion, cupped flower heads of mixed size and tilt, trailing stems."""
    leafm = mat("leaf", (1.3, 1.4, 1.1), rough=0.75)
    hm = clamp(0.2 * sc, 0.1, 0.34)
    C = Vector((base.x, base.y, base.z - 0.01))
    env = _Env(C, wi * 0.99, hm / (wi * 0.99), 0.012, r, amp=0.12, n=7)
    _shell("mound", env, 0.86, 0.02, p.seed + 23, mat("moss"), seg=12, rings=8, rough=0.04, dark=0.12)
    m1, m2 = _head_leaves(env, r, 2 * 900, 2 * 170, tilt=(10, 40), down=0.3, under_skip=1.0, zmin=0.0, droop=0.45, out=0.35)
    petal = mat("wax", p.bloom, rough=0.9)
    eye_m = mat("wax", "#e8c23c")
    nfl = int(p.blooms) * 6
    fm, em = [], []

    def head(P, n, s):
        M = _face_out(P, n, r, scale=s, jit=0.4, cup=r.u(0.6, 1.5))
        fm.append(M)
        em.append(M @ Matrix.Translation((0, 0, 0.0025)))

    for i in range(nfl):
        a = r.u(0, 2 * pi)
        z = r.u(0.12, 1.0)
        q = sqrt(1 - z * z)
        d = Vector((q * cos(a), q * sin(a), z))
        head(env.pt(d, r.u(1.0, 1.06)) + Vector((0, 0, 0.01)), (env.nrm(d) + Vector((0, 0, 1.2))).normalized(), r.u(0.72, 1.3))
    # trailing stems spilling over the capping board
    for i in range(int(4 + 3 * clamp(sc, 0.6, 1.5))):
        k = r.pick([0, 0, 1, 2, 3, 3])
        xo = r.u(-0.75, 0.75) * wi
        ln = r.u(0.1, 0.26) * sc
        pts = [(xo * 0.8, -wi * 0.72, base.z + 0.03), (xo * 0.95, -half + 0.065, Hb + 0.03), (xo, -half - 0.012, Hb + 0.012),
               (xo + r.u(-0.02, 0.02), -half - 0.022, Hb - ln * 0.5), (xo + r.u(-0.03, 0.03), -half - 0.018 - r.u(0, 0.012), Hb - ln)]
        path = [tuple(q) for q in resample_path(pts, n=9)]
        vine = sweep(f"vine{i}", path, 0.0028, sides=4, smooth_path=False, scale_fn=lambda t: 1.0 - 0.5 * t)
        assign(vine, leafm)
        _yaw(vine, k)
        Rk = _rz(k)
        for j in range(3, len(path)):
            P = Rk @ Vector(path[j])
            out = Rk @ Vector((0, -1, 0))
            for _ in range(2):
                m2.append(_frame(P, (out + Vector((0, 0, -0.3))).normalized(), r, droop=0.5, out=1.0, scale=0.85))
            if r.chance(0.42) or j == len(path) - 1:
                head(P + out * 0.012, (out + Vector((0, 0, 0.2))).normalized(), r.u(0.6, 0.95))
    _emit_leaves("flower_leaves", m1, m2, 0.042, 0.03, leafm)
    if fm:
        fl = _flower_mesh("flower_base", 0.021)
        assign(fl, petal)
        instances_to_mesh("flowers", fl, fm, uv_jitter=False, keep_base=False)
        cen = lathe("flower_eye", [(0.0042, 0.0), (0.0, 0.0045)], segments=5)
        assign(cen, eye_m)
        instances_to_mesh("flower_eyes", cen, em, keep_base=False)
