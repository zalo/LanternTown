"""Old wisteria trained against a wall. Three stems of unequal girth, grown together into one fluted,
furrowed, kinked and leaning trunk (burrs, knuckles and sawn pruning stubs at the old spur points),
rise from a stone-edged planting pit (or a sett-ringed cut-out in the paving) with their root toes
buried in the soil. Iron straps hold the trunk to the wall; at different heights the stems peel off
and taper from wrist-thick limbs to finger-thick runners tied along taut training wires on vine eyes.
Side shoots climb above and drop below the wires, whippy young growth breaks the outline, drooping
pinnate leaves (mature green, darker old fronds, bronze young tips) sit above, and separate pendulous
racemes of pea-flowers - pale banner, darker keel, tight buds at the tip - hang below, lit from
underneath by small jar lanterns on forged brackets.
"""
from tk import *
import colorsys
import numpy as np
from math import sin, cos, pi, exp
from mathutils import Matrix

META = {"name": "Wisteria on a wall", "class": "vegetation", "mount": "wall_ground", "focus": 0.72}
PARAMS = {
    "width": {"default": 4.8, "min": 2.0, "max": 7.0, "desc": "overall width along the wall (m)"},
    "height": {"default": 3.2, "min": 2.3, "max": 5.2, "desc": "height of the top training wire (m)"},
    "tiers": {"default": 2, "min": 1, "max": 3, "desc": "number of horizontal wires / runner tiers"},
    "tier_gap": {"default": 0.62, "min": 0.4, "max": 0.9, "desc": "vertical spacing of the tiers (m)"},
    "trunk_pos": {"default": -0.72, "min": -1.0, "max": 1.0, "desc": "trunk position along the width (-1 left .. 1 right)"},
    "trunk_girth": {"default": 0.23, "min": 0.12, "max": 0.36, "desc": "trunk diameter above the root flare (m)"},
    "bloom": {"default": 0.8, "min": 0.0, "max": 1.0, "desc": "bloom amount: density of flower racemes"},
    "bloom_color": {"default": "#c2aef9", "desc": "flower colour (lilac / white / pink)"},
    "raceme_len": {"default": 0.38, "min": 0.2, "max": 0.7, "desc": "typical raceme length (m); individual tassels vary 0.6x - 1.6x"},
    "leaf_density": {"default": 1.0, "min": 0.25, "max": 1.6, "desc": "amount of foliage"},
    "leaf_color": {"default": "#84965c", "desc": "mature leaf colour"},
    "pods": {"default": 0.0, "min": 0.0, "max": 1.0, "desc": "amount of hanging seed pods (summer / autumn)"},
    "fallen": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "fallen petals on the ground"},
    "bed": {"default": True, "desc": "raised stone-edged planting pit (False: sett-ringed cut-out in the paving)"},
    "lanterns": {"default": 3, "min": 0, "max": 4, "desc": "jar lanterns on wall brackets under the canopy"},
    "intensity": {"default": 2.2, "min": 0.5, "max": 20.0, "desc": "light intensity per lantern (cd)"},
}
VARIANTS = {
    "white_long": {"width": 5.6, "height": 3.5, "tiers": 1, "trunk_pos": 0.15, "bloom": 1.0, "bloom_color": "#f4f1e2",
                   "raceme_len": 0.62, "leaf_density": 0.8, "leaf_color": "#93a465", "lanterns": 3, "seed": 4},
    "ancient_tall": {"width": 4.6, "height": 4.7, "tiers": 3, "tier_gap": 0.78, "trunk_pos": 0.8, "trunk_girth": 0.34,
                     "bloom": 0.75, "bloom_color": "#9f84f0", "raceme_len": 0.44, "lanterns": 4, "intensity": 2.4, "seed": 11},
    "summer_green": {"width": 3.6, "height": 2.9, "tiers": 2, "trunk_pos": -0.9, "bloom": 0.0, "leaf_density": 1.4,
                     "leaf_color": "#74894c", "pods": 0.8, "fallen": 0.0, "lanterns": 2, "seed": 23},
    "pink_small": {"width": 3.0, "height": 2.7, "tiers": 1, "trunk_pos": 0.1, "trunk_girth": 0.16, "bloom": 1.0, "intensity": 1.7,
                   "bloom_color": "#f59ac0", "raceme_len": 0.34, "leaf_density": 0.5, "bed": False, "lanterns": 2, "seed": 33},
}

TILE = 0.25          # 'leaf' material tile (m): one blade is mapped onto the whole tile


# ----------------------------------------------------------------------------- helpers
def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _shade(h, v=1.0, s=1.0, hue=0.0):
    hh, ss, vv = colorsys.rgb_to_hsv(*_rgb(h))
    r, g, b = colorsys.hsv_to_rgb((hh + hue) % 1.0, clamp(ss * s), clamp(vv * v))
    return "#%02x%02x%02x" % (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _frame(d):
    """orthonormal (a, b, d) with a x b = d"""
    d = _unit(d)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = _unit(np.cross(ref, d))
    return a, np.cross(d, a), d


def _M(xa, ya, za, pos, s=1.0):
    return Matrix(((xa[0] * s, ya[0] * s, za[0] * s, pos[0]), (xa[1] * s, ya[1] * s, za[1] * s, pos[1]),
                   (xa[2] * s, ya[2] * s, za[2] * s, pos[2]), (0, 0, 0, 1)))


def _mesh(name, V, F, UV=None, mats=None, midx=None, sm=60.0):
    o = mesh_object(name, V, F)
    me = o.data
    if UV is not None:
        flat_uv = [c for fuv in UV for uv in fuv for c in uv]
        if len(flat_uv) == len(me.loops) * 2:
            me.uv_layers.new(name="UVMap").data.foreach_set("uv", flat_uv)
            uv_keep(o)
    if mats:
        for m in mats:
            me.materials.append(m)
        if midx is not None and len(midx) == len(me.polygons):
            me.polygons.foreach_set("material_index", midx)
    smooth(o, sm)
    return o


def _tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def _rings(pts, radii):
    P = np.asarray(pts, dtype=float)
    n = len(P)
    R = np.broadcast_to(np.asarray(radii, dtype=float), (n,)).copy()
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    N = np.zeros_like(P)
    N[0] = _frame(T[0])[0]
    for i in range(1, n):
        v = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        N[i] = v / max(np.linalg.norm(v), 1e-9)
    B = np.cross(T, N)
    Lc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return P, R, T, N, B, Lc


def _skin(V, n, sides, R, Lc, P, T, cap):
    cm = 2 * pi * float(R.mean())
    F, UV = [], []
    for i in range(n - 1):
        for j in range(sides):
            j1 = (j + 1) % sides
            F.append((i * sides + j, i * sides + j1, (i + 1) * sides + j1, (i + 1) * sides + j))
            UV.append(((j / sides * cm, Lc[i]), ((j + 1) / sides * cm, Lc[i]), ((j + 1) / sides * cm, Lc[i + 1]), (j / sides * cm, Lc[i + 1])))
    base = (n - 1) * sides
    if cap == "flat":                                    # sawn end: eased arris + flat face
        for j in range(sides):
            vj = np.array(V[base + j])
            V.append((P[-1] + (vj - P[-1]) * 0.7 + T[-1] * R[-1] * 0.2).tolist())
        ib = len(V) - sides
        for j in range(sides):
            j1 = (j + 1) % sides
            F.append((base + j, base + j1, ib + j1, ib + j))
            UV.append(((j / sides * cm, Lc[-1]), ((j + 1) / sides * cm, Lc[-1]), ((j + 1) / sides * cm, Lc[-1] + 0.01), (j / sides * cm, Lc[-1] + 0.01)))
        base = ib
    if cap:
        V.append((P[-1] + T[-1] * R[-1] * (0.24 if cap == "flat" else 0.8)).tolist())
        tip = len(V) - 1
        for j in range(sides):
            j1 = (j + 1) % sides
            F.append((base + j, base + j1, tip))
            UV.append(((j / sides * cm, Lc[-1]), ((j + 1) / sides * cm, Lc[-1]), ((j + 0.5) / sides * cm, Lc[-1] + R[-1])))
    return F, UV


def _tube_data(pts, radii, sides=8, cap=True, v0=0.0):
    """Tube along pts with per-point radius. UVs in metres: U around, V along (bark ridges run along V)."""
    P, R, T, N, B, Lc = _rings(pts, radii)
    ang = np.linspace(0.0, 2 * pi, sides, endpoint=False)
    ring = N[:, None, :] * np.cos(ang)[None, :, None] + B[:, None, :] * np.sin(ang)[None, :, None]
    V = (P[:, None, :] + ring * R[:, None, None]).reshape(-1, 3).tolist()
    F, UV = _skin(V, len(P), sides, R, Lc + v0, P, T, cap)
    return V, F, UV


def _clamp_obj(o, zmin=None, ymax=None, xlim=None):
    me = o.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if zmin is not None:
        co[:, 2] = np.maximum(co[:, 2], zmin)
    if ymax is not None:
        co[:, 1] = np.minimum(co[:, 1], ymax)
    if xlim is not None:
        co[:, 0] = np.clip(co[:, 0], -xlim, xlim)
    me.vertices.foreach_set("co", co.ravel())
    me.update()


def _tube(name, pts, radii, sides=8, cap=True, zmin=None, ymax=None):
    V, F, UV = _tube_data(pts, radii, sides, cap)
    o = _mesh(name, V, F, UV, sm=70.0)
    if zmin is not None or ymax is not None:
        _clamp_obj(o, zmin, ymax)
    return o


def _bark_tube(name, pts, radii, sides=12, ridges=4, depth=0.13, squash=0.12, seed=0.0, cap="tip"):
    """Old-wood limb: tube whose section is squashed out of round and cut by wandering furrows between
    broad ridges (relief fades out on thin wood). U around, V along."""
    P, R, T, N, B, Lc = _rings(pts, radii)
    A = np.linspace(0.0, 2 * pi, sides, endpoint=False)[None, :]
    S = Lc[:, None]
    old = np.clip((R - 0.009) / 0.028, 0.0, 1.0)[:, None]
    wander = 0.9 * np.sin(S * 3.1 + seed) + 0.5 * np.sin(S * 7.3 + seed * 1.7)
    w = 0.5 + 0.5 * np.sin(ridges * A + wander + 0.8 * np.sin(2 * A + S * 2.0 + seed * 0.6))
    m = 1.0 + depth * old * (0.4 - 1.6 * (1.0 - w) ** 2.2) + depth * old * 0.45 * np.sin(3 * A + S * 9.0 + seed * 2.3) * np.sin(S * 4.3 + 2 * A + seed)
    e = squash * np.sin(S * 1.3 + seed * 0.9) * old
    ca, sa = np.cos(A) * (1.0 + e), np.sin(A) * (1.0 - e)
    ring = N[:, None, :] * ca[:, :, None] + B[:, None, :] * sa[:, :, None]
    V = (P[:, None, :] + ring * (R[:, None] * m)[:, :, None]).reshape(-1, 3).tolist()
    F, UV = _skin(V, len(P), sides, R, Lc + seed, P, T, cap)
    return _mesh(name, V, F, UV, sm=62.0)


class _Soup:
    """accumulates faces with per-corner UVs, a material index and per-vertex grime / wear masks"""

    def __init__(self):
        self.V, self.F, self.UV, self.MI, self.G, self.W = [], [], [], [], [], []

    def add(self, verts, faces, uvs, mi, g=0.0, w=0.0):
        b = len(self.V)
        nv = len(verts)
        self.V.extend([tuple(float(c) for c in v) for v in verts])
        self.G.extend(list(g) if hasattr(g, "__len__") else [g] * nv)
        self.W.extend(list(w) if hasattr(w, "__len__") else [w] * nv)
        for f, u in zip(faces, uvs):
            self.F.append(tuple(b + i for i in f))
            self.UV.append(u)
            self.MI.append(mi)

    def obj(self, name, mats, sm=75.0, masks=False):
        o = _mesh(name, self.V, self.F, self.UV, mats, self.MI, sm)
        if masks:
            G, Wr = np.array(self.G, dtype=float), np.array(self.W, dtype=float)
            n = len(G)
            paint(o, grime=lambda co, nn: G if len(co) == n else np.zeros(len(co)),
                  wear=lambda co, nn: Wr if len(co) == n else np.zeros(len(co)), mode="set")
        return o


# ----------------------------------------------------------------------------- foliage base meshes
def _leaflet(sp, o, f, n, ln, hw, mi=0, fold=0.35, droop=0.24):
    """pointed blade from o along f, face normal n, folded along the midrib, tip hanging; mapped to the whole leaf tile"""
    w = np.cross(n, f)
    tip = o + f * (ln * 0.97) - n * (droop * ln)
    a1, a2 = o + f * (0.30 * ln), o + f * (0.68 * ln) - n * (droop * 0.3 * ln)
    up = n * (fold * hw)
    v = [o, a1 + w * (hw * 0.97) + up, a2 + w * (hw * 0.72) + up, tip, a2 - w * (hw * 0.72) + up, a1 - w * (hw * 0.97) + up]
    c, e = TILE * 0.5, TILE * 0.44
    uv = [(c, 0.004), (c + e * 0.97, TILE * 0.30), (c + e * 0.72, TILE * 0.68), (c, TILE - 0.004), (c - e * 0.72, TILE * 0.68), (c - e * 0.97, TILE * 0.30)]
    sp.add(v, [(0, 3, 2, 1), (0, 5, 4, 3)], [(uv[0], uv[3], uv[2], uv[1]), (uv[0], uv[5], uv[4], uv[3])], mi)


def _leaf_base(name, r, L, pairs, lf_len, lf_w, material, tone=0.0, fold=0.35, arch=0.75):
    """pinnate wisteria leaf: rachis along +X (normal +Z) arching over and down, `pairs` pairs of leaflets
    hanging relaxed off it + a terminal one. tone = uniform darkening (older fronds)."""
    sp = _Soup()

    def rc(u):
        return np.array([L * u * (1.0 - 0.1 * u * u), 0.0, L * (0.2 * u - arch * u * u)])

    def rd(u):
        return _unit((1.0 - 0.3 * u * u, 0.0, 0.2 - 2.0 * arch * u))

    us = np.linspace(0.0, 1.0, 4)
    V, F, UV = _tube_data([rc(u) for u in us], [0.0024, 0.002, 0.0015, 0.001], sides=3, cap=False)
    UV = [tuple((TILE * 0.5 + (a - 0.003) * 0.3, 0.02 + min(b, 0.2)) for (a, b) in q) for q in UV]
    sp.add(V, F, UV, 0)
    for j in range(pairs):
        u = 0.24 + 0.68 * j / max(pairs - 1, 1)
        t = rd(u)
        nrm = np.array([-t[2], 0.0, t[0]])
        size = 0.78 + 0.3 * sin(pi * (j + 0.6) / (pairs + 0.4))
        for side in (-1.0, 1.0):
            hang = r.u(0.45, 1.05)
            f = _unit(np.array([0.0, side, 0.0]) + t * r.u(0.25, 0.55) - nrm * hang)
            n = _unit(nrm - f * np.dot(nrm, f) + np.array([0.0, side * r.u(-0.25, 0.25), 0.0]))
            n = _unit(n - f * np.dot(n, f))
            _leaflet(sp, rc(u) + f * 0.004, f, n, lf_len * size * r.vary(1.0, 0.14), lf_w * 0.5 * size * r.vary(1.0, 0.1),
                     fold=fold, droop=r.u(0.14, 0.34))
    t = rd(1.0)
    nrm = np.array([-t[2], 0.0, t[0]])
    _leaflet(sp, rc(1.0), t, nrm, lf_len * 1.05, lf_w * 0.5, fold=fold)
    o = sp.obj(name, [material])
    if tone > 0.0:
        paint(o, grime=tone, mode="set")
    return o


def _raceme_base(name, r, L, n_open, n_bud, Rmax, m_pale, m_dark, m_green, edge=0.3):
    """One pendulous flower tassel hanging from the origin along -Z: bare peduncle, then pea-florets
    spiralled tightly round the rachis inside a tapered cone (wide shoulder -> half-open flowers -> tight
    buds at the tip). Each open floret = cupped two-lobed banner (pale throat) + darker keel."""
    sp = _Soup()
    wob, ph = r.u(0.6, 1.4), r.u(0.0, 6.28)
    k = L / 0.4

    def c(t):
        return np.array([0.02 * sin(t * 2.4 * wob + ph) * k * t, 0.014 * sin(t * 3.7 + ph + 1.0) * k * t, -0.025 - t * L])

    def env(t):
        return Rmax * (0.5 + 0.5 * smoothstep(0.0, 0.08, t)) * (0.1 + 0.9 * (1.0 - t ** 1.5))

    ez = np.array([0.0, 0.0, 1.0])
    V, F, UV = _tube_data([np.zeros(3)] + [c(t) for t in np.linspace(0.0, 0.3, 3)], [0.003, 0.0027, 0.0022, 0.002], sides=3, cap=False)
    UV = [tuple((TILE * 0.5 + (a - 0.004) * 0.3, 0.02 + min(b, 0.2) * 0.5) for (a, b) in q) for q in UV]
    sp.add(V, F, UV, 2)
    tc = np.linspace(0.0, 0.98, 8)                       # solid heart of the cluster: florets sit on it like scales
    V, F, UV = _tube_data([c(t) for t in tc], [max(env(t) * 0.7, 0.003) for t in tc], sides=6, cap=True)
    for i in range(len(V) - 1):                          # lumpy, not a smooth carrot
        cc = c(tc[i // 6])
        V[i] = (cc + (np.array(V[i]) - cc) * r.u(0.72, 1.25) + ez * r.u(-0.006, 0.006)).tolist()
    sp.add(V, F, UV, 0, g=[r.u(0.0, 0.2) for _ in V])

    T_open = 0.78
    gt = np.linspace(0.015, T_open, 80)
    cdf = np.cumsum([env(t) + 0.004 for t in gt])
    cdf /= cdf[-1]
    buv = [(0.06, 0.01), (0.115, 0.045), (0.1, 0.1), (0.06, 0.092), (0.02, 0.1), (0.005, 0.045)]
    for i in range(n_open):
        t = float(np.interp((i + 0.5) / n_open, cdf, gt))
        a = i * 2.39996 + r.u(-0.3, 0.3)
        er = np.array([cos(a), sin(a), 0.0])
        R = env(t) * r.u(0.8, 1.05)
        ctr = c(t) + er * R - ez * (0.3 * R)
        s = lerp(0.031, 0.02, t / T_open) * r.vary(1.0, 0.12)
        op = 1.0 - 0.6 * smoothstep(0.6 * T_open, T_open, t)
        pitch, yaw, roll = r.u(-0.8, -0.2), r.u(-0.5, 0.5), r.u(-0.4, 0.4)
        X = er * cos(pitch) + ez * sin(pitch)
        Z = -er * sin(pitch) + ez * cos(pitch)
        Y = np.cross(Z, X)
        X, Y = X * cos(yaw) + Y * sin(yaw), Y * cos(yaw) - X * sin(yaw)
        Y, Z = Y * cos(roll) + Z * sin(roll), Z * cos(roll) - Y * sin(roll)

        def q(x, y, z):
            return ctr + (X * x + Y * y + Z * z) * s

        yo, cl = 0.5 + 0.5 * op, 0.3 * (1.0 - op)
        bv = [q(0.0, 0, -0.14), q(0.17 + cl, 0.52 * yo, 0.2), q(-0.12 * op, 0.37 * yo, 0.8), q(-0.02, 0, 0.72),
              q(-0.12 * op, -0.37 * yo, 0.8), q(0.17 + cl, -0.52 * yo, 0.2)]
        sp.add(bv, [(0, 1, 2, 3), (0, 3, 4, 5)], [tuple(buv[j] for j in (0, 1, 2, 3)), tuple(buv[j] for j in (0, 3, 4, 5))], 0,
               g=[0.0, edge, edge * 1.2, 0.0, edge * 1.2, edge], w=[0.18, 0.0, 0.0, 0.05, 0.0, 0.0])
        kv = [q(0.02, 0, -0.1), q(0.68, 0, -0.42), q(0.36, 0.25 * yo, -0.05), q(0.36, -0.25 * yo, -0.05)]
        sp.add(kv, [(0, 2, 1), (0, 1, 3)], [((0.06, 0.01), (0.1, 0.06), (0.06, 0.11)), ((0.06, 0.01), (0.06, 0.11), (0.02, 0.06))], 1,
               g=[0.0, 0.3, 0.0, 0.0])
    for i in range(n_bud):
        t = lerp(T_open - 0.02, 1.0, ((i + 0.5) / n_bud) ** 0.9)
        a = i * 2.39996 + r.u(-0.4, 0.4)
        er = np.array([cos(a), sin(a), 0.0])
        R = env(t) * r.u(0.5, 1.0)
        s = lerp(0.0125, 0.006, (t - T_open) / (1.0 - T_open)) * r.vary(1.0, 0.15)
        pitch = r.u(-1.15, -0.5)
        X = er * cos(pitch) + ez * sin(pitch)
        Nn = _unit(ez - X * np.dot(ez, X))
        A = np.cross(X, Nn)
        b0 = c(t) + er * (R * 0.3)
        sp.add([b0, b0 + X * s + A * (0.45 * s) - Nn * (0.22 * s), b0 + X * (2.3 * s), b0 + X * s - A * (0.45 * s) - Nn * (0.22 * s)],
               [(0, 1, 2), (0, 2, 3)], [((0.06, 0.01), (0.1, 0.06), (0.06, 0.11)), ((0.06, 0.01), (0.06, 0.11), (0.02, 0.06))], 1,
               g=[0.15, 0.3, 0.6, 0.3])
    return sp.obj(name, [m_pale, m_dark, m_green], sm=65.0, masks=True)


def _pod_cluster(name, r, material):
    """last spring's flower stalk hanging from the origin with 2-4 flat velvety bean pods (12-17 cm, bumps over
    the seeds, beaked tip) dangling from its lower half"""
    sp = _Soup()
    Ls = r.u(0.3, 0.52)
    bx, by = r.u(-0.03, 0.03), r.u(-0.03, 0.0)

    def c(t):
        return np.array([bx * t * t, by * t * t, -Ls * t])

    V, F, UV = _tube_data([c(t) for t in (0.0, 0.35, 0.7, 1.0)], [0.003, 0.0026, 0.0024, 0.002], sides=3, cap=False)
    sp.add(V, F, UV, 0)
    n = r.pick([2, 3, 3, 4])
    a0 = r.u(0.0, 6.28)
    ts = np.linspace(0.0, 1.0, 9)
    for k in range(n):
        t = 1.0 if k == 0 else r.u(0.6, 0.97)
        a = a0 + k * 2.4 + r.u(-0.4, 0.4)
        tilt = r.u(0.08, 0.2) if k == 0 else r.u(0.25, 0.6)
        d = _unit((sin(tilt) * cos(a), sin(tilt) * sin(a), -cos(tilt)))
        e1, e2, e3 = _frame(d)
        Lp, bow = r.u(0.14, 0.2), r.u(-0.025, 0.025)
        pts = [(bow * sin(pi * u), 0.0, 0.012 + Lp * u) for u in ts]
        rad = [0.022 * sin(pi * (0.06 + 0.86 * u)) ** 0.55 * (1.0 + 0.2 * sin(u * 21.0 + k)) * (0.7 + 0.4 * u) for u in ts]
        V, F, UV = _tube_data(pts, rad, sides=6, cap=True)
        b = c(t)
        sp.add([b + e1 * v[0] + e2 * (v[1] * 0.55) + e3 * v[2] for v in V], F, UV, 0)
    return sp.obj(name, [material], sm=70.0)


# ----------------------------------------------------------------------------- lantern on a wall bracket
def _lantern(i, x, zb, reach, iron, intensity, flame_mat):
    """forged scroll bracket fixed to the wall with a small glazed jar lantern hanging from its hook"""
    objs = []
    bar = rounded_rect(0.012, 0.012, 0.003, 1)
    bar2 = rounded_rect(0.009, 0.009, 0.002, 1)
    objs.append(box(f"lb_plate{i}", (0.036, 0.006, 0.24), loc=(x, -0.003, zb - 0.05), bevel=0.0015))
    objs.append(sweep(f"lb_arm{i}", [(x, -0.006, zb), (x, -reach * 0.5, zb + 0.012), (x, -reach, zb + 0.004)], bar, up=(1, 0, 0)))
    objs.append(sweep(f"lb_stay{i}", bezier_path((x, -0.006, zb - 0.15), (x, -0.05, zb - 0.15), (x, -reach * 0.62, zb - 0.1), (x, -reach * 0.72, zb - 0.004), n=9),
                      bar2, up=(1, 0, 0), smooth_path=False))
    sc = spiral_path(0.005, 0.028, 1.35, n=12, a0_deg=270, plane="YZ", center=(x, -reach - 0.004, zb + 0.032))
    objs.append(sweep(f"lb_scroll{i}", sc, bar2, up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.55 + 0.45 * t))
    for dz in (0.045, -0.145):
        objs.append(bolt(f"lb_bolt{i}", 0.006, 0.004, loc=(x, -0.006, zb + dz), rot=(90, 0, 0), kind="dome"))
    yl = -reach + 0.035
    ztop = zb - 0.075                     # top ring of the lantern
    hook = [(x, yl, zb - 0.004), (x, yl, zb - 0.03), (x, yl + 0.008, zb - 0.05), (x, yl, zb - 0.064), (x, yl - 0.008, zb - 0.052)]
    objs.append(sweep(f"lb_hook{i}", hook, 0.0028, sides=5, step=0.012))
    for o in objs:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.05, seed=i + 3, lo=0.35, hi=0.75), 0.55))
    L = (x, yl, ztop)
    body = [torus(f"ln_ring{i}", 0.011, 0.0022, loc=(x, yl, ztop + 0.009), rot=(0, 90, 0), major_seg=10, minor_seg=5),
            sweep(f"ln_bail{i}", [(x - 0.047, yl, ztop - 0.092), (x - 0.04, yl, ztop - 0.045), (x, yl, ztop - 0.004), (x + 0.04, yl, ztop - 0.045),
                                  (x + 0.047, yl, ztop - 0.092)], 0.002, sides=5, step=0.02),
            lathe(f"ln_cap{i}", [(0.046, -0.104), (0.053, -0.101), (0.05, -0.093), (0.02, -0.08), (0.014, -0.071), (0.022, -0.066), (0.02, -0.058),
                                 (0.0, -0.054)], segments=12, loc=L),
            lathe(f"ln_base{i}", [(0.0, -0.252), (0.046, -0.25), (0.053, -0.243), (0.05, -0.235), (0.043, -0.231), (0.0, -0.229)], segments=12, loc=L),
            lathe(f"ln_font{i}", [(0.0, -0.231), (0.022, -0.23), (0.024, -0.215), (0.009, -0.208), (0.006, -0.198), (0.0, -0.197)], segments=8, loc=L)]
    for k in range(4):
        a = pi / 4 + k * pi / 2
        gx, gy = 0.0505 * cos(a), 0.0505 * sin(a)
        body.append(sweep(f"ln_guard{i}_{k}", [(x + gx, yl + gy, ztop - 0.24), (x + gx * 1.04, yl + gy * 1.04, ztop - 0.165), (x + gx, yl + gy, ztop - 0.098)],
                          0.0016, sides=4, smooth_path=False))
    for o in body:
        assign(o, iron)
    glass = lathe(f"ln_glass{i}", [(0.041, -0.232), (0.046, -0.205), (0.047, -0.165), (0.045, -0.125), (0.041, -0.102)], segments=12, loc=L)
    assign(glass, mat("glass_lantern", glow=1.5))
    flame = lathe(f"ln_flame{i}", [(0.0, -0.198), (0.007, -0.191), (0.008, -0.183), (0.004, -0.17), (0.0, -0.158)], segments=8, loc=L)
    assign(flame, flame_mat)
    light(f"lantern{i}", (x, yl, ztop - 0.18), color="#ffcd96", intensity=intensity, range=5.5, flicker=0.3, shadow_near=0.11)
    part(body + [glass, flame], f"lantern{i}", pivot=(x, yl, ztop + 0.01), motion="swing", amp=2.5, speed=0.35)


# ----------------------------------------------------------------------------- build
def build(p, rng):
    W, H, g = p.width, p.height, p.trunk_girth
    xl, xr = -W / 2, W / 2
    yw = -0.11                                           # training wires stand 110 mm off the wall
    tiers = int(p.tiers)
    gap = min(p.tier_gap, (H - 1.6) / (tiers - 1)) if tiers > 1 else 0.0
    zt = [H - i * gap for i in range(tiers)]
    z_top = zt[0]
    tx = p.trunk_pos * max(W / 2 - 0.6, 0.0)
    ty = -(0.52 * g + 0.035)
    s_long = 1 if tx <= 0 else -1
    has_bloom, has_pods = p.bloom > 0.02, p.pods > 0.02
    nlant = int(p.lanterns)
    stat = {}

    # ------------------------------------------------------------------ materials (<= 10 variants)
    bark = mat("bark")
    iron = mat("iron_wrought")
    leaf_a = mat("leaf", p.leaf_color)
    leaf_b = mat("leaf", _shade(p.leaf_color, v=1.12, s=1.55, hue=-0.1))            # bronze-gold young growth
    spare = 10 - (4 + (2 if nlant else 0) + 2 + (2 if has_bloom else 0))
    leaf_c = mat("leaf", _shade(p.leaf_color, v=1.32, s=1.35, hue=-0.055)) if spare >= 1 else leaf_b    # yellowing fronds
    pod_mat = mat("burlap", "#c9b47a") if (has_pods and spare >= 2) else leaf_b
    hh, ss, vv = colorsys.rgb_to_hsv(*_rgb(p.bloom_color))
    white = ss < 0.12
    petal = mat("wax", _shade(p.bloom_color, v=1.08, s=0.9), grime=0.5)
    petal_dk = mat("wax", "#e0e3b2" if white else _shade(p.bloom_color, v=0.95, s=1.32, hue=-0.012 if hh < 0.8 else 0.0), grime=0.5)

    # ------------------------------------------------------------------ planting pit geometry (needed by the roots)
    ba, bb = (0.30 + g, 0.27 + g) if p.bed else (0.17 + g, 0.15 + g)
    h_e, h_c = (0.03, 0.095) if p.bed else (0.012, 0.06)

    def mound(x, y):
        q = ((x - tx) / ba) ** 2 + ((-y - 0.01) / bb) ** 2
        return h_e + (h_c - h_e) * (1.0 - q) if q < 1.0 else 0.0

    def pit_reach(u):
        """distance from the trunk foot to the pit edge (or the wall) in horizontal direction u"""
        c0 = -ty - 0.01
        A = (u[0] / ba) ** 2 + (u[1] / bb) ** 2
        B = c0 * u[1] / bb ** 2
        C = (c0 / bb) ** 2 - 1.0
        t = (B + (B * B - A * C) ** 0.5) / A
        if u[1] > 0.05:
            t = min(t, (-0.03 - ty) / u[1])
        return max(t, 0.02)

    # ------------------------------------------------------------------ trunk: leaning, kinked axis + three unequal fused stems
    rt = rng.fork("trunk")
    hand = rt.pick([-1.0, 1.0])
    kinks = []
    zk, sgn = rt.u(0.3, 0.5), s_long
    while zk < z_top - 0.55:
        kinks.append((zk, sgn * rt.u(0.17, 0.34)))
        sgn = -sgn
        zk += rt.u(0.55, 0.95)
    la, lp = rt.u(0.015, 0.035), rt.u(0.0, 6.28)
    by, bp = rt.u(0.03, 0.07), rt.u(0.0, 6.28)

    def axis(z):
        x = tx + la * sin(2.1 * z + lp) * smoothstep(0.0, 0.6, z)
        for zk_, a_ in kinks:
            d_ = z - zk_
            x += a_ * 0.5 * (d_ + (d_ * d_ + 0.002) ** 0.5)
        y = ty - by * (0.5 + 0.5 * sin(1.7 * z + bp)) * smoothstep(0.1, 0.8, z)
        return np.array([x, y, z])

    p0_ = clamp(7.5 * g, 1.3, 2.6) * rt.u(0.9, 1.3)                       # mean twist pitch; the local pitch wanders
    f1, f2 = rt.u(0.0, 6.28), rt.u(0.0, 6.28)
    zs_tab = np.arange(0.0, z_top + 0.4, 0.02)
    rate = 2 * pi / (p0_ * np.clip(1.0 + 0.6 * np.sin(2.3 * zs_tab + f1) + 0.3 * np.sin(5.3 * zs_tab + f2), 0.5, 2.2))
    th_tab = np.concatenate([[0.0], np.cumsum(rate[:-1] * 0.02)])
    wk = [1.0, 0.8, 0.6]                                                     # relative girth of the three stems
    ph = [-pi / 2 + k * 2.094 + rt.u(-0.3, 0.3) for k in range(3)]
    sph = [rt.u(0.0, 6.28) for _ in range(3)]
    knk = [[rt.u(0.3, max(z_top - 0.4, 0.5)) for _ in range(int(rt.u(2, 5)))] for _ in range(3)]

    def stem_th(k, z):
        return ph[k] + hand * float(np.interp(z, zs_tab, th_tab)) + 0.35 * sin(1.9 * z + sph[k])

    def stem_pt(k, z):
        th = stem_th(k, z)
        rh = g * (0.2 + 0.04 * sin(2.7 * z + sph[k] * 1.7) + 0.26 * exp(-z / 0.12))
        return axis(z) + rh * np.array([cos(th), sin(th), 0.0])

    def stem_r(k, z):
        r = g * 0.275 * wk[k] * lerp(1.06, 0.72, clamp(z / z_top)) * (1.0 + 0.5 * exp(-z / 0.18))
        r *= 1.0 + 0.13 * sin(z * 6.0 + sph[k]) + 0.06 * sin(z * 15.0 + k)
        for zq in knk[k]:
            r *= 1.0 + 0.2 * exp(-((z - zq) / 0.055) ** 2)
        return r

    # runner slots (tier, side, x_end): staggered tier lengths; the three trunk stems feed the long ones first
    slots = []
    for ti in range(tiers):
        for s in (s_long, -s_long):
            full = s * (W / 2 - 0.3)
            span = abs(full - tx)
            if span < 0.6:
                continue
            if ti == 0:
                fr = 1.0
            elif s == s_long:
                fr = rt.u(0.6, 0.8) if ti == 1 else rt.u(0.36, 0.55)
            else:
                fr = rt.u(0.55, 0.9)
            slots.append((ti, s, tx + s * span * fr))
    slots.sort(key=lambda q: (0 if q[1] == s_long else 1, q[0]))
    spare_fr = [0.6, 0.36]
    n_real = len(slots)
    while len(slots) < 3:                                # spare stems twine a shorter way along the main runner
        full = s_long * (W / 2 - 0.3)
        slots.insert(n_real, (0, s_long, tx + (full - tx) * spare_fr[len(slots) - n_real if len(slots) - n_real < 2 else 1]))
    runners = []                                         # (pts, radii, side, tier) of every limb on a wire
    wood = []
    z_step = max(0.05, z_top / 60.0)

    def runner_path(p0, tan, r_d, ti, s, xe, rr):
        """from p0 (heading tan) sweep over to the wire of tier ti and run along it to x = xe, tapering to 2 cm"""
        x0 = axis(zt[ti])[0] + s * rr.u(0.42, 0.7)
        if (xe - x0) * s < 0.35:
            x0 = xe - s * 0.35
        span = abs(xe - x0)
        phw, pw = rr.u(0.0, 6.28), rr.u(0.9, 1.6)
        wa, wb = rr.u(0.02, 0.045), rr.u(0.0, 6.28)
        droop = rr.u(0.1, 0.26)

        def rad_f(f):
            return lerp(r_d * 0.9, 0.010, clamp(f) ** 0.8)

        def wire_pt(dx):
            f = dx / span
            r = rad_f(f)
            psi = phw + 2 * pi * dx / pw
            rw = r * 0.6 + 0.006
            return np.array([x0 + s * dx, yw + rw * cos(psi) - 0.06 * smoothstep(0.8, 1.0, f),
                             zt[ti] + rw * sin(psi) + wa * sin(dx * 2.9 + wb) * smoothstep(0.0, 0.4, dx) - droop * smoothstep(0.7, 1.0, f) ** 2])

        w0 = wire_pt(0.0)
        dist = float(np.linalg.norm(w0 - p0))
        bz = bezier_path(tuple(p0), tuple(p0 + tan * (0.12 + 0.3 * dist)), tuple(w0 - np.array([s * 0.32 * dist, 0.0, 0.03])), tuple(w0), n=9)
        pts = [np.array(q, dtype=float) for q in bz][1:]
        rad = [lerp(r_d, r_d * 0.9, (i + 1) / len(pts)) for i in range(len(pts))]
        for dx in np.arange(0.075, span, 0.075):
            pts.append(wire_pt(dx))
            rad.append(rad_f(dx / span))
        return pts, rad

    z_dep = []
    for k in range(3):
        ti, s, xe = slots[k]
        rr = rt.fork(f"stem{k}")
        z_d = max(zt[ti] - rr.u(0.34, 0.6), 0.9)
        z_dep.append(z_d)
        th0 = stem_th(k, 0.0)
        u0 = np.array([cos(th0), sin(th0), 0.0])
        a0 = axis(0.0)
        D = min(1.7 * g + 0.04, 0.6 * pit_reach(u0))
        r_b = stem_r(k, 0.1)
        pts, rad = [], []
        for dd, zz, rs in ((1.0, 0.012, 0.4), (0.66, 0.024, 0.58), (0.3, 0.055, 0.85)):     # root toe diving into the soil
            pts.append(a0 + u0 * (g * 0.42 + D * dd) + np.array([0.0, 0.0, zz]))
            rad.append(r_b * rs)
        for z in np.arange(0.12, z_d, z_step):
            pts.append(stem_pt(k, z))
            rad.append(stem_r(k, z))
        p0 = stem_pt(k, z_d)
        tan = _unit(p0 - stem_pt(k, z_d - 0.05))
        pts.append(p0)
        rad.append(stem_r(k, z_d))
        n_tr = len(pts)
        rp, rrad = runner_path(p0, tan, stem_r(k, z_d), ti, s, xe, rr)
        pts += rp
        rad += rrad
        for i in range(len(pts)):                        # never through the wall
            pts[i][1] = min(pts[i][1], -(rad[i] * 1.15 + 0.008))
        o = _bark_tube(f"stem{k}", pts, rad, sides=12, ridges=4, depth=0.15, squash=0.14, seed=p.seed * 0.37 + k * 2.1)
        wood.append(o)
        runners.append((np.array(pts[n_tr - 1:]), np.array(rad[n_tr - 1:]), s, ti))
    z_low, z_mid = sorted(z_dep)[0], sorted(z_dep)[1]
    # heartwood between the stems: closes the gaps so the trunk reads as one fused, fluted column
    cz = np.arange(0.05, z_mid + 0.06, z_step * 1.4)
    core = _bark_tube("trunk_core", [axis(z) for z in cz],
                      [g * 0.215 * (1.0 + 0.5 * exp(-z / 0.2)) * (1.0 - 0.5 * smoothstep(z_low - 0.15, z_mid, z)) for z in cz],
                      sides=9, ridges=3, depth=0.1, squash=0.1, seed=p.seed * 0.11)
    wood.append(core)
    for bi, (ti, s, xe) in enumerate(slots[3:]):         # side limbs for the remaining tiers
        rr = rt.fork(f"branch{bi}")
        zs = max(zt[ti] - rr.u(0.4, 0.6), 0.8)
        p0 = axis(zs) + np.array([s * g * 0.12, -g * 0.12, 0.0])
        tan = _unit((s * 0.35, -0.25, 1.0))
        r_d = g * 0.15
        rp, rrad = runner_path(p0 + tan * 0.02, tan, r_d, ti, s, xe, rr)
        pts, rad = [p0] + rp, [r_d * 1.2] + rrad
        for i in range(len(pts)):
            pts[i][1] = min(pts[i][1], -(rad[i] * 1.15 + 0.008))
        wood.append(_bark_tube(f"branch{bi}", pts, rad, sides=9, ridges=3, depth=0.13, squash=0.12, seed=p.seed * 0.23 + bi))
        runners.append((np.array(pts[1:]), np.array(rad[1:]), s, ti))
    # buttress roots between the stem toes, tips buried in the soil
    a0 = axis(0.0)
    for j in range(int(rt.u(2, 4))):
        th = stem_th(j % 3, 0.0) + 1.05 + rt.u(-0.3, 0.3)
        dv = np.array([cos(th), sin(th), 0.0])
        D = min(1.5 * g + 0.03, 0.58 * pit_reach(dv))
        rpts = [axis(0.17) + dv * (g * 0.2), a0 + dv * (g * 0.44 + D * 0.3) + np.array([0, 0, 0.065]),
                a0 + dv * (g * 0.44 + D * 0.7) + np.array([0, 0, 0.026]), a0 + dv * (g * 0.44 + D) + np.array([0, 0, 0.012])]
        rpts = resample_path([tuple(q) for q in rpts], n=7)
        wood.append(_bark_tube(f"root{j}", rpts, np.linspace(g * 0.125, g * 0.055, 7), sides=8, ridges=3, depth=0.12, seed=j * 1.7, cap=None))
    # a young shoot twining up around the old trunk
    rr = rt.fork("twiner")
    tw, twr = [], []
    z_tw = z_low - 0.1
    th_t = rr.u(0.0, 6.28)
    tpw = rr.u(0.0, 6.28)
    for z in np.arange(0.3, z_tw, 0.04):
        th = th_t - hand * 2 * pi * (z + 0.3 * sin(2.3 * z + tpw)) / (p0_ * 0.72)
        q = axis(z) + (g * 0.56 + 0.014 + 0.02 * sin(4.1 * z + tpw)) * np.array([cos(th), sin(th), 0.0])
        q[1] = min(q[1], -0.016)
        tw.append(q)
        twr.append(lerp(0.011, 0.006, z / z_tw))
    if len(tw) > 4:
        wood.append(_tube("twiner", tw, twr, sides=6))
    # burrs with sawn pruning stubs and the odd sprout at old spur points on the trunk
    sprouts = []
    for j in range(int(rt.u(3, 6))):
        z = rt.u(0.3, max(z_low - 0.25, 0.5))
        th = rt.u(3.45, 5.95)
        dv = np.array([cos(th), sin(th), 0.0])
        Rb = g * rt.u(0.11, 0.17)
        c = axis(z) + dv * (g * 0.4)
        b = sphere(f"burr{j}", radius=Rb, loc=tuple(c), segments=9, rings=6, scale=(1.0, 1.0, rt.u(1.1, 1.5)))
        realize(b)
        displace_noise(b, strength=Rb * 0.28, scale=Rb * 0.9, seed=p.seed + j)
        wood.append(b)
        dd = _unit(dv + np.array([0.0, 0.0, rt.u(0.1, 0.8)]))
        if rt.chance(0.7):
            ln = rt.u(0.035, 0.085)
            rs = g * rt.u(0.065, 0.1)
            wood.append(_bark_tube(f"stub{j}", [c + dd * (Rb * 0.5 + ln * f) for f in (0.0, 0.5, 1.0)], [rs * 1.15, rs, rs * 0.95],
                                   sides=8, ridges=3, depth=0.08, seed=j, cap="flat"))
        else:
            sprouts.append((c + dd * Rb * 0.8, dd))
    for pts, rad, s, ti in runners:                      # knuckles with short sawn spurs along the old part of the limbs
        i = int(rt.u(4, 9))
        j = 0
        while i < len(pts) - 2 and rad[i] > 0.016:
            dd = _unit((rt.u(-0.4, 0.4), -rt.u(0.2, 1.0), rt.u(-0.3, 1.0)))
            rs = rad[i] * rt.u(0.34, 0.5)
            wood.append(_bark_tube(f"spurstub{ti}{j}", [pts[i] + dd * (rad[i] * f) for f in (0.3, 1.0, 1.0 + rt.u(0.5, 1.3))], [rs * 1.5, rs * 1.1, rs],
                                   sides=6, ridges=3, depth=0.05, seed=j, cap="flat"))
            i += int(rt.u(5, 11))
            j += 1
    for o in wood:
        displace_noise(o, strength=0.004 + g * 0.012, scale=0.06, seed=p.seed)
        displace_noise(o, strength=g * 0.035, scale=0.28, seed=p.seed + 5)
        _clamp_obj(o, zmin=0.004, ymax=-0.004)
        assign(o, bark)
        paint(o, grime=height_ramp(0.0, 0.45),
              special=mask_mul(height_ramp(0.0, 1.5), noise_mask(0.16, seed=p.seed + 1, lo=0.35, hi=0.7), 0.8))
        paint(o, special=mask_mul(facing((0, 0, 1)), noise_mask(0.22, seed=p.seed + 6, lo=0.45, hi=0.75), 0.5))
    stat["wood"] = sum(_tris(o) for o in wood)

    # extent of every tier (staggered) -> wires, lanterns
    t_ext = []
    for ti in range(tiers):
        xs = [q for pts, rad, s, t2 in runners if t2 == ti for q in (pts[0][0], pts[-1][0])]
        t_ext.append((min(xs), max(xs)) if xs else (tx - 0.3, tx + 0.3))
    cx0, cx1 = min(e[0] for e in t_ext), max(e[1] for e in t_ext)

    # ------------------------------------------------------------------ lantern positions (racemes keep clear of the brackets)
    lant = []
    rlan = rng.fork("lant")
    for i in range(nlant):
        x = lerp(cx0 + 0.3, cx1 - 0.3, (i + 0.5) / nlant) + rlan.u(-0.12, 0.12)
        ti = 0
        for t2 in range(tiers - 1, -1, -1):              # hang it under the lowest tier that reaches this far
            if t_ext[t2][0] - 0.1 <= x <= t_ext[t2][1] + 0.1:
                ti = t2
                break
        zb = max(zt[ti] - (0.13 + 0.5 * p.raceme_len * (1.0 if has_bloom else 0.5)), min(1.95, zt[ti] - 0.2))
        ax_ = axis(zb)[0]
        if abs(x - ax_) < 0.36:
            x = ax_ + (0.38 if (x >= ax_ and ax_ + 0.38 < xr - 0.25) or ax_ - 0.38 < xl + 0.25 else -0.38)
        lant.append((x, zb + (i % 2) * 0.05, ti))

    # ------------------------------------------------------------------ side shoots climbing above / dropping off the wires
    rl_ = rng.fork("laterals")
    laterals = []
    for pts, rad, s, ti in runners:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        d, nxt = 0.0, rl_.u(0.3, 0.7)
        for i in range(1, len(pts) - 3):
            d += seg[i - 1]
            if d < nxt:
                continue
            nxt = d + rl_.u(0.4, 0.95)
            P, r = pts[i], rad[i]
            climb = rl_.chance(0.62 if ti == 0 else 0.42)
            if climb:
                ln = rl_.u(0.4, 1.0)
                dr = _unit((s * rl_.u(-0.3, 0.8), -rl_.u(0.0, 0.2), 1.0))
            else:
                ln = rl_.u(0.35, 0.85)
                dr = _unit((s * rl_.u(-0.2, 0.7), -rl_.u(0.4, 0.8), -0.1))
            n = max(5, int(ln / 0.055))
            path = [P + dr * (r * 0.4)]
            wob = rl_.u(0.0, 6.28)
            for q in range(n):
                e = (q + 1) / n
                if climb:
                    dr = _unit(dr + np.array([0.1 * sin(q * 0.8 + wob), -0.03 - 0.1 * e * e, 0.04 - 0.42 * e ** 2.5]))
                else:
                    dr = _unit(dr + np.array([0.07 * sin(q * 0.8 + wob), 0.05, -0.2]))
                path.append(path[-1] + dr * 0.055)
            for q in path:
                q[1] = min(q[1], -0.03)
                q[0] = clamp(q[0], xl + 0.04, xr - 0.04)
            laterals.append((np.array(path), np.linspace(clamp(r * 0.45, 0.006, 0.015), 0.0028, len(path)), s, ti, climb))
    lat_objs = [_tube(f"lateral{i}", lt[0], lt[1], sides=5, ymax=-0.006) for i, lt in enumerate(laterals)]
    for o in lat_objs:
        assign(o, bark)
    stat["laterals"] = sum(_tris(o) for o in lat_objs)

    # ------------------------------------------------------------------ foliage distribution
    rf = rng.fork("foliage")
    leaves, racs, pods, spurs, shoots = [], [], [], [], []
    up = np.array([0.0, 0.0, 1.0])
    sp_leaf = 0.088 / p.leaf_density
    sp_rac = 0.085 / max(p.bloom, 0.02)
    sp_pod = 0.13 / max(p.pods, 0.02)
    dph = (rf.u(0.0, 6.28), rf.u(0.0, 6.28))

    def dens(x):                                         # clumps and thin patches along the wall
        return 0.5 + 1.0 * (0.5 + 0.5 * sin(x * 2.3 + dph[0]) * sin(x * 0.9 + dph[1]))

    def add_leaf(pos, a, sc, young=False):
        a = np.array(a, dtype=float)
        a[1] = min(a[1], 0.15)
        a = _unit(a)
        pos = np.array(pos, dtype=float)
        pos[1] = min(pos[1], -0.035)
        pos[0] = clamp(pos[0], xl + 0.22, xr - 0.22)
        n0 = up if abs(a[2]) < 0.93 else np.array([0.0, -1.0, 0.0])
        z = _unit(n0 - a * np.dot(n0, a))
        rho = rf.u(-0.7, 0.7)
        z = z * cos(rho) + np.cross(a, z) * sin(rho)
        leaves.append((_M(a, np.cross(z, a), z, pos, sc), young))

    def add_spur(a, b, r0):
        d = np.asarray(b) - np.asarray(a)
        ln = float(np.linalg.norm(d))
        if ln < 0.012:
            return
        e1, e2, e3 = _frame(d)
        mid = (np.asarray(a) + np.asarray(b)) * 0.5
        spurs.append(Matrix(((e1[0] * r0, e2[0] * r0, e3[0] * ln, mid[0]), (e1[1] * r0, e2[1] * r0, e3[1] * ln, mid[1]),
                             (e1[2] * r0, e2[2] * r0, e3[2] * ln, mid[2]), (0, 0, 0, 1))))

    def leafy_shoot(P, d, ln, r0=0.0055, sag=0.27, sc=1.0):
        path, dirs = [np.array(P, dtype=float)], [d]
        nseg = max(5, int(ln / 0.11))
        wob = rf.u(0.0, 6.28)
        for q in range(nseg):
            path.append(path[-1] + d * (ln / nseg))
            d = _unit(d + np.array([0.12 * sin(q * 1.1 + wob), -0.05, -sag * 5.0 / nseg]))
            dirs.append(d)
        for q in path:
            q[1] = min(q[1], -0.02)
            q[0] = clamp(q[0], xl + 0.05, xr - 0.05)
        shoots.append((path, np.linspace(r0, 0.0018, len(path))))
        nl = max(2, int(ln / 0.08))
        for q in range(nl):
            f = (q + 0.7) / nl
            i0 = min(int(f * nseg), nseg - 1)
            pos = lerp(path[i0], path[i0 + 1], f * nseg - i0)
            td = dirs[i0]
            side = _unit(np.cross(td, up)) * (1.0 if q % 2 else -1.0)
            young = q >= nl - 2
            add_leaf(pos, side * 0.9 + td * 0.55 + np.array([0.0, -0.25, 0.25]), sc * rf.u(0.8, 1.15) * (1.0 - 0.25 * f), young)

    def leaf_node(P, r, top):
        if rf.chance(0.3):                               # leafy shoot arching up and out
            d = _unit((rf.u(-0.75, 0.75), -rf.u(0.25, 1.0), rf.u(0.2, 1.0)))
            ln = rf.u(0.25, 0.7)
            if rf.chance(0.24 if top else 0.08):         # strong whippy leaders break the top line unevenly
                d = _unit((d[0] * 0.7, d[1] * 0.4, 1.0))
                ln = rf.u(0.7, 1.45) if top else rf.u(0.5, 0.8)
                leafy_shoot(P + d * (r * 0.5), d, ln, r0=0.007, sag=0.2)
            else:
                leafy_shoot(P + d * (r * 0.5), d, ln)
        else:                                            # short spur with a tuft of leaves
            d = _unit((rf.u(-0.6, 0.6), -rf.u(0.2, 1.0), rf.u(-0.15, 1.0)))
            tip = P + d * (r * 0.6 + rf.u(0.03, 0.1))
            add_spur(P, tip, 0.0042)
            for _ in range(rf.pick([1, 2, 2, 3])):
                add_leaf(tip, d + np.array([rf.u(-0.9, 0.9), rf.u(-0.9, 0.2), rf.u(-0.3, 0.6)]), rf.u(0.8, 1.2))

    def hang_node(P, r, into, ti, sep):
        Q = P + np.array([rf.u(-0.03, 0.03), -(r + rf.u(0.0, 0.21)), -(r * 0.6 + rf.u(0.0, 0.05))])
        Q[1] = min(Q[1], -0.085)
        Q[0] = clamp(Q[0], xl + 0.08, xr - 0.08)
        for O in into:                                   # keep tassels apart so each one keeps its own outline
            if abs(O[0] - Q[0]) < sep and abs(O[2] - Q[2]) < 0.3 and abs(O[1] - Q[1]) < 0.09:
                return
        for lx, lz, lti in lant:                         # ... and off the lantern brackets
            if abs(lx - Q[0]) < 0.075 and Q[2] > lz - 0.1:
                return
        add_spur(P, Q, 0.0034)
        into.append(Q)

    L_tot = 0.0
    ties = []
    for pts, rad, s, ti in runners:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        for i in range(8, int(len(pts) * 0.85), int(rf.u(6, 10))):          # soft-wire ties binding the limb to its wire
            w = np.array([pts[i][0], yw, zt[ti] - 0.002])
            dist = float(np.linalg.norm(pts[i] - w))
            if dist < 0.1:
                ties.append(trs(tuple((pts[i] + w) * 0.5), (rf.u(-12, 12), 90 + rf.u(-10, 10), 0), dist * 0.5 + rad[i] + 0.003))
        L_tot += float(seg.sum())
        nl, nr, npd = rf.u(0.0, sp_leaf), rf.u(0.0, sp_rac) + 0.1, rf.u(0.0, sp_pod) + 0.12
        nw = rf.u(0.5, 1.3)
        d = 0.0
        for i in range(1, len(pts)):
            d += seg[i - 1]
            while d >= nl:
                leaf_node(pts[i].copy(), rad[i], ti == 0)
                nl += sp_leaf * rf.u(0.5, 1.5) / dens(pts[i][0])
            while has_bloom and d >= nr:
                hang_node(pts[i].copy(), rad[i], racs, ti, 0.062)
                nr += sp_rac * rf.u(0.45, 1.9)
            while d >= nw:                               # long pendulous tendril hanging out of the canopy
                nw += rf.u(0.8, 1.5)
                wd = _unit((rf.u(-0.35, 0.35), -rf.u(0.35, 0.8), -0.25))
                wp = [pts[i] + np.array([0.0, -rad[i], 0.0])]
                nst = int(rf.u(9, 17))
                curl = rf.pick([-1.0, 1.0])
                for q in range(nst):
                    wp.append(wp[-1] + wd * 0.055)
                    e = q / nst
                    wd = _unit(wd + np.array([curl * 0.5 * e ** 3, 0.05, -0.3 + 0.75 * e ** 4]))
                for q in wp:
                    q[1] = min(q[1], -0.03)
                    q[0] = clamp(q[0], xl + 0.03, xr - 0.03)
                shoots.append((wp[::2], np.linspace(0.004, 0.0013, len(wp[::2]))))
                for q in range(2, nst, 3):
                    add_leaf(wp[q], np.array([rf.u(-0.8, 0.8), -rf.u(0.2, 0.9), rf.u(-0.5, 0.2)]), rf.u(0.4, 0.62), q > nst - 5)
            while has_pods and d >= npd:
                hang_node(pts[i].copy(), rad[i], pods, ti, 0.1)
                npd += sp_pod * rf.u(0.5, 1.5)
        # whippy young growth reaching beyond the limb's end and trailing down
        d = _unit((s * 0.8, -0.45, 0.3))
        path = [pts[-1].copy()]
        nq = int(rf.u(7, 14))
        for q in range(nq):
            path.append(path[-1] + d * 0.05)
            d = _unit(d + np.array([-s * 0.035 * q, -0.03, -0.22]))
        for q in path:
            q[0] = clamp(q[0], xl + 0.02, xr - 0.02)
        shoots.append((path[::2], np.linspace(0.0055, 0.0015, len(path[::2]))))
        for q in range(2, nq, 2):
            add_leaf(path[q], np.array([s * 0.5, -0.5 if q % 4 else 0.3, 0.3]), rf.u(0.45, 0.7), q > nq - 4)
    for path, rr_, s, ti, climb in laterals:             # leaves alternate up the side shoots, tassels hang off them too
        step = max(1, int(round(sp_leaf * 1.05 / 0.055)))
        n = len(path)
        for q in range(1 + int(rf.u(0, 2)), n, step):
            td = _unit(path[min(q + 1, n - 1)] - path[q - 1])
            side = _unit(np.cross(td, np.array([0.0, -1.0, 0.0]))) * (1.0 if (q // step) % 2 else -1.0)
            add_leaf(path[q], side * 0.8 + td * 0.4 + np.array([0.0, -0.6, 0.1]), rf.u(0.75, 1.1), q >= n - 2 * step)
        for q in range(2, int(n * 0.8), 2):
            if has_bloom and rf.chance(p.bloom * (0.22 if climb else 0.34)):
                hang_node(path[q].copy(), rr_[q], racs, ti, 0.062)
            if has_pods and rf.chance(p.pods * 0.16):
                hang_node(path[q].copy(), rr_[q], pods, ti, 0.1)
    for q in tw[6::8]:                                   # a few leaves on the twiner and sprouts from the burrs
        dq = _unit((q[0] - axis(q[2])[0], q[1] - ty, 0.0))
        add_leaf(q, dq + np.array([0.0, -0.4, 0.35]), rf.u(0.55, 0.85))
    for c, dd in sprouts:
        leafy_shoot(c, _unit(dd + np.array([0.0, -0.3, 0.9])), rf.u(0.3, 0.6), r0=0.005, sag=0.14, sc=0.8)

    k_big = clamp(L_tot / 8.0, 0.75, 1.06)
    cap_leaf = int(215 * k_big * clamp(p.leaf_density, 0.5, 1.15))
    if len(leaves) > cap_leaf:
        keep = len(leaves) / cap_leaf
        leaves = [leaves[int(i * keep)] for i in range(cap_leaf)]

    sh_objs = []
    for i, (path, radii) in enumerate(shoots):
        o = _tube(f"shoot{i}", path, radii, sides=3, ymax=-0.006)
        assign(o, bark)
        sh_objs.append(o)
    stat["shoots"] = sum(_tris(o) for o in sh_objs)
    if spurs:
        sb = cylinder("spur_base", radius=1.0, radius2=0.5, depth=1.0, verts=4, cap=False)
        assign(sb, bark)
        instances_to_mesh("spurs", sb, spurs, keep_base=False)

    # ------------------------------------------------------------------ leaves: mature / dark old / yellowing / bronze young fronds
    if leaves:
        rl = rng.fork("leaf")
        specs = [(0.26, 4, 0.082, 0.038, leaf_a, 0.0, 3.0), (0.32, 5, 0.088, 0.040, leaf_a, 0.12, 3.0), (0.32, 5, 0.088, 0.040, leaf_a, 0.42, 3.0),
                 (0.38, 6, 0.090, 0.040, leaf_a, 0.28, 2.5), (0.38, 6, 0.092, 0.042, leaf_a, 0.6, 2.0), (0.30, 5, 0.082, 0.038, leaf_c, 0.0, 1.2),
                 (0.17, 3, 0.058, 0.025, leaf_b, 0.0, 0.5)]
        bases = [_leaf_base(f"leafbase{vi}", rl, L, pairs, ll, lw, m, tone, fold=0.75 if vi == 6 else 0.35, arch=0.5 if vi == 6 else rl.u(0.6, 0.85))
                 for vi, (L, pairs, ll, lw, m, tone, wt) in enumerate(specs)]
        pick_any = [(vi, sp_[6]) for vi, sp_ in enumerate(specs)]
        groups = [[] for _ in bases]
        for M, young in leaves:
            vi = (6 if rl.chance(0.8) else 0) if young else rl.weighted(pick_any)
            groups[vi].append(M)
        ntl = 0
        for b, ms in zip(bases, groups):
            if ms:
                o = instances_to_mesh("leaves", b, ms, uv_jitter=False, keep_base=False)
                _clamp_obj(o, ymax=-0.008, xlim=W / 2)
                ntl += _tris(o)
            else:
                delete([b])
        stat["leaves"] = ntl

    # ------------------------------------------------------------------ flower racemes: separate tapered tassels of varied length
    n_fall = 0
    if racs:
        rc_ = rng.fork("raceme")
        Lr = p.raceme_len
        lfs = [(0.62, 2.0), (0.8, 3.0), (1.0, 3.0), (1.25, 2.0), (1.6, 1.2)]
        est = sum((6 * clamp(int(115 * Lr * lf), 18, 95) + 2 * int(10 + 24 * Lr * lf) + 110) * wt for lf, wt in lfs) / sum(wt for lf, wt in lfs)
        cap_r = int(22000 * clamp(L_tot / 8.0, 0.65, 0.93) / est)
        if len(racs) > cap_r:
            keep = len(racs) / cap_r
            racs = [racs[int(i * keep)] for i in range(cap_r)]
        variants = []
        for vi, (lf, wt) in enumerate(lfs):
            L = clamp(Lr * lf, 0.16, 1.0)
            variants.append(_raceme_base(f"raceme{vi}", rc_, L, clamp(int(115 * L), 18, 95), int(10 + 24 * L), clamp(0.034 + 0.011 * L / 0.4, 0.04, 0.05),
                                         petal, petal_dk, leaf_a, edge=0.0 if white else 0.3))
        groups = [[] for _ in variants]
        pick_r = [(vi, wt) for vi, (lf, wt) in enumerate(lfs)]
        for Q in racs:
            groups[rc_.weighted(pick_r)].append(trs(tuple(Q), (rc_.u(-6, 6), rc_.u(-6, 6), rc_.u(0, 360)), rc_.u(0.9, 1.1)))
        ntr = 0
        for b, ms in zip(variants, groups):
            if ms:
                o = instances_to_mesh("racemes", b, ms, uv_jitter=False, keep_base=False)
                _clamp_obj(o, ymax=-0.006)
                ntr += _tris(o)
            else:
                delete([b])
        stat["racemes"] = ntr
        stat["n_rac"] = len(racs)
        n_fall = int(p.fallen * len(racs) * 3.5)

    # ------------------------------------------------------------------ seed pods: clusters of velvety beans hanging below the limbs
    if pods:
        rp_ = rng.fork("pods")
        pbs = [_pod_cluster(f"pod_base{vi}", rp_, pod_mat) for vi in range(4)]
        groups = [[] for _ in pbs]
        for Q in pods:
            groups[int(rp_.u(0, 3.999))].append(trs(tuple(Q), (rp_.u(-5, 5), rp_.u(-5, 5), rp_.u(0, 360)), rp_.u(0.88, 1.12)))
        ntp = 0
        for b, ms in zip(pbs, groups):
            if ms:
                o = instances_to_mesh("pods", b, ms, keep_base=False)
                _clamp_obj(o, ymax=-0.006)
                ntp += _tris(o)
            else:
                delete([b])
        stat["pods"] = ntp
        stat["n_pod"] = len(pods)

    # ------------------------------------------------------------------ planting pit: soil mound + raised stone edging / flush sett ring
    rb = rng.fork("bed")
    soil = surface("soil", lambda u, v: (tx + ba * v * cos(pi * u), -(0.01 + bb * v * sin(pi * u)), h_e + (h_c - h_e) * (1.0 - v * v)), 18, 7)
    realize(soil)
    if soil.data.polygons[len(soil.data.polygons) // 2].normal.z < 0:
        soil.data.flip_normals()
    displace_noise(soil, strength=0.012 if p.bed else 0.007, scale=0.09, seed=p.seed + 9, axis_mask=(0, 0, 1))
    _clamp_obj(soil, zmin=0.006, ymax=-0.004)
    assign(soil, mat("soil"))
    stone = mat("stone_rough", "#a39a8a" if p.bed else "#9a948a")
    arc_len = pi * (ba + bb) * 0.5
    K = max(6, int(arc_len / (0.17 if p.bed else 0.115)))
    for i in range(K):
        a = pi * (i + 0.5) / K
        off = 0.04 if p.bed else 0.045
        sx = tx + (ba + off) * cos(a)
        sy = min(-(0.01 + (bb + off) * sin(a)), -0.07)
        hgt = rb.u(0.07, 0.1) if p.bed else rb.u(0.03, 0.042)
        dx_, dy_ = -ba * sin(a), -bb * cos(a)
        b = stone_block(f"edge{i}", (arc_len / K * rb.u(0.84, 0.95), rb.u(0.085, 0.11), hgt), loc=(sx, sy, hgt * 0.5 + 0.004),
                        rot=(rb.u(-3, 3), rb.u(-3, 3), np.degrees(np.arctan2(dy_, dx_)) + rb.u(-5, 5)), rng=rb, chamfer=0.012 if p.bed else 0.008, detail=0.05)
        assign(b, stone)
        paint(b, grime=mask_mul(height_ramp(0.0, 0.1), 1.0 if p.bed else 0.35), special=mask_mul(noise_mask(0.12, seed=p.seed + i, lo=0.3, hi=0.7), 0.75))

    # ------------------------------------------------------------------ fallen petals
    if n_fall > 0:
        rq = rng.fork("fallen")
        a_, b_ = 0.011, 0.014
        pt = _mesh("petal_base", [(-a_, -b_, 0.001), (a_, -b_, 0.0), (a_, b_, 0.002), (-a_, b_, 0.0)], [(0, 1, 2, 3)],
                   [((0.03, 0.02), (0.22, 0.02), (0.22, 0.23), (0.03, 0.23))], [petal])
        ms = []
        for _ in range(n_fall):
            Q = rq.pick(racs)
            x = clamp(Q[0] + rq.u(-0.35, 0.35) + rq.u(-0.25, 0.25), xl, xr)
            y = -(0.03 + abs(rq.u(-0.5, 0.5) + rq.u(-0.45, 0.45)) + abs(Q[1]) * 0.5)
            ms.append(trs((x, y, mound(x, y) + 0.03 if mound(x, y) > 0 else 0.005), (rq.u(-10, 10), rq.u(-10, 10), rq.u(0, 360)), rq.u(0.7, 1.3)))
        instances_to_mesh("fallen_petals", pt, ms, uv_jitter=False, keep_base=False)

    # ------------------------------------------------------------------ training wires, vine eyes, tensioners, trunk ties
    hw_ = []
    eye = join([cylinder("eye_shank", 0.0036, -yw - 0.01, loc=(0, yw * 0.5, 0), rot=(90, 0, 0), verts=6),
                torus("eye_ring", 0.0105, 0.003, loc=(0, yw, 0), rot=(0, 90, 0), major_seg=8, minor_seg=4),
                box("eye_plate", (0.032, 0.004, 0.032), loc=(0, -0.002, 0), rot=(0, 45, 0))], "eye_base")
    ems = []
    for ti, z in enumerate(zt):
        x0 = xl + 0.06 if ti == 0 else max(xl + 0.06, t_ext[ti][0] - 0.3)
        x1 = xr - 0.06 if ti == 0 else min(xr - 0.06, t_ext[ti][1] + 0.3)
        hw_.append(sweep(f"wire{ti}", [(x0, yw, z), ((x0 + x1) / 2, yw, z - 0.004), (x1, yw, z)], 0.0022, sides=5))
        ne = max(2, int(round((x1 - x0) / 1.05)) + 1)
        for i in range(ne):
            ems.append(trs((lerp(x0, x1, i / (ne - 1)), 0, z)))
        tb = x1 - 0.14 if s_long > 0 else x0 + 0.14                             # barrel tensioner near the far eye
        hw_.append(cylinder(f"tens{ti}", 0.0065, 0.075, loc=(tb, yw, z - 0.001), rot=(0, 90, 0), verts=8, bevel=0.0015))
    hw_.append(instances_to_mesh("vine_eyes", eye, ems, keep_base=False))
    if ties:
        hw_.append(instances_to_mesh("wire_ties", torus("tie_base", 1.0, 0.055, major_seg=10, minor_seg=4), ties, keep_base=False))
    for i, f in enumerate((0.3, 0.64)):                                          # iron straps holding the trunk to the wall
        z = z_low * f + 0.25 + rt.u(-0.06, 0.06)
        c = axis(z)
        R = g * 0.53 + 0.012
        path = [(c[0] - R, -0.004, z), (c[0] - R, c[1] * 0.5, z)]
        path += [(c[0] + R * cos(a), c[1] + R * sin(a), z) for a in np.linspace(pi, 2 * pi, 9)]
        path += [(c[0] + R, c[1] * 0.5, z), (c[0] + R, -0.004, z)]
        hw_.append(sweep(f"tie{i}", path, rounded_rect(0.004, 0.026, 0.001, 1), up=(0, 0, 1), smooth_path=False))
        for sx in (-1, 1):
            hw_.append(box(f"tie_lug{i}", (0.05, 0.005, 0.034), loc=(c[0] + sx * (R + 0.02), -0.0025, z), bevel=0.001))
            hw_.append(bolt(f"tie_nail{i}", 0.0055, 0.004, loc=(c[0] + sx * (R + 0.03), -0.005, z), rot=(90, 0, 0), kind="pyramid"))
    for o in hw_:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.06, seed=p.seed + 2, lo=0.3, hi=0.7), 0.7))

    # ------------------------------------------------------------------ jar lanterns under the canopy
    if nlant:
        flame_mat = mat("emissive", "#ffb45e", glow=9)
        for i, (x, zb, ti) in enumerate(lant):
            _lantern(i, x, zb, 0.62, iron, p.intensity, flame_mat)
            socket(f"lantern_hook{i}", "attach", (x, -0.585, zb - 0.06), role="lantern")
