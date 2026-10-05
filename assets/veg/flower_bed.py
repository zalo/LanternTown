"""Raised flower bed for the town square: a kerb of individually laid stones (rock-faced ashlar in
snecked courses on a plinth, lime-pointed, with an overhanging dressed coping and corner piers - or
dry-laid field stones of mixed length and height with corner boulders, pinning stones and earth-packed
joints), filled with a mounded bed of dark soil and planted like a municipal bedding scheme: low
plants in drifts of 2-3 colours (five-petal flowers built petal by petal, daisies, small-flowered
cushions) over three kinds of foliage in three greens (broad ovate, round lobed, narrow lanceolate,
small-leaved sprigs), clumps of hosta and blue grass, foxglove-like spikes of bells along the spine
and ivy-like swags that creep over the coping and hang against the stone face.
Every stone is a light shell (only the faces that can be seen), every leaf / petal a small folded,
smooth-shaded mesh stamped with instances_to_mesh over a dark under-canopy.

Far level (lod 2, seen from 44 m on): the same plan with the same random draws, but nothing is stamped.  The kerb
is one plain section per side (wall face, plinth and coping; over field stones its top and face follow the stones),
piers and corner boulders are boxes; the foliage of the low plants, hostas and spike rosettes is one closed blanket
over the bed in the three greens of the plants below it; every flowering plant carries a low cap in its petal colour
with the area of the flowers it kept; a spike is a three-sided spire, a grass tuft a small core with seven blades, a swag one
ribbon over the coping; fallen petals are one triangle per cluster.  Stems, moss tufts, pointing and pinning stones
are left out.
"""
from collections import defaultdict
from math import asin, atan2, cos, degrees, hypot, pi, radians, sin, sqrt

import numpy as np
from mathutils import Matrix, Vector

from tk import *

META = {"name": "Raised flower bed", "class": "vegetation", "mount": "ground", "focus": 0.42, "lod": "far"}

PALETTES = {
    # (open flowers, daisies, small cushion flowers), drift colours used for the tall spikes
    "cottage": (("#e8799c", "#f3ecd9", "#6f63d6"), (0, 2)),
    "sunset": (("#f0b722", "#e5611f", "#b3202f"), (1, 0)),
    "moonlight": (("#f5f1e4", "#efe089", "#b59be6"), (2, 0)),
    "royal": (("#8a45c4", "#d9408f", "#f1ead6"), (1, 0)),
}

PARAMS = {
    "shape": {"default": "rect", "choices": ["rect", "round"], "desc": "plan shape of the bed"},
    "length": {"default": 2.4, "min": 1.2, "max": 3.2, "desc": "overall length along X, or the diameter of a round bed (m)"},
    "width": {"default": 1.2, "min": 0.8, "max": 1.8, "desc": "overall depth along Y of a rect bed (m)"},
    "kerb_height": {"default": 0.36, "min": 0.18, "max": 0.6, "desc": "height of the stone kerb above the paving (m)"},
    "kerb": {"default": "dressed", "choices": ["dressed", "rustic"], "desc": "dressed ashlar with coping and corner piers / dry-laid field stones"},
    "stone": {"default": "#b9ad97", "desc": "stone tint"},
    "palette": {"default": "cottage", "choices": list(PALETTES), "desc": "flower colour scheme (three drift colours)"},
    "density": {"default": 0.9, "min": 0.3, "max": 1.0, "desc": "planting density (low = freshly planted, soil showing)"},
    "spikes": {"default": 1.0, "min": 0.0, "max": 2.0, "desc": "amount of tall flower spikes"},
    "overgrowth": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "trailing plants spilling over the kerb"},
    "moss": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "damp / moss on the stonework"},
}
VARIANTS = {
    "round_moonlight": {"shape": "round", "length": 2.1, "kerb_height": 0.45, "palette": "moonlight", "spikes": 1.4, "seed": 4},
    "rustic_sunset": {"kerb": "rustic", "length": 3.0, "width": 1.1, "kerb_height": 0.26, "palette": "sunset", "moss": 0.85,
                      "overgrowth": 0.9, "stone": "#a89c8c", "seed": 11},
    "tall_royal": {"length": 1.7, "width": 1.0, "kerb_height": 0.58, "palette": "royal", "density": 1.0, "spikes": 0.6, "seed": 23},
    "round_rustic_young": {"shape": "round", "kerb": "rustic", "length": 1.6, "kerb_height": 0.3, "palette": "sunset",
                           "density": 0.42, "spikes": 0.5, "overgrowth": 0.2, "moss": 0.2, "seed": 31},
}

TRI_TARGET = 29700
# far level: area of a cap per area of the petals it stands for; radius and height of the core of a grass tuft and the
# width of its blades at the foot (per unit of its scale); how far a triangle of fallen petals is drawn in towards its
# petal next to the kerb
FAR_FLOWER = (0.6, 0.28)                                # (open flowers and daisies, the tiny ones half hidden in their sprigs)
FAR_GRASS = (0.1, 0.2, 0.085)
FAR_PETAL = 0.8
FAR_SPIRE = (0.02, 0.0155)                            # radius of the spire of a spike at its foot and half-way up
FAR_RIM = 0.07                                          # how far the blanket of foliage reaches out over the kerb (m)
FAR_LIFT = 0.02                                         # leaf tips above the domes of the plants (m)
FAR_EDGE = 0.07                                         # foliage of the edging plants above the kerb (m)
FKEY = {(0, -1): "-x", (0, 1): "+x", (1, -1): "-y", (1, 1): "+y", (2, -1): "-z", (2, 1): "+z"}


# ---------------------------------------------------------------------------------- helpers
def az_up(ax, ay):
    """Yaw (deg) so that a +Z facing mesh tilted about X leans towards azimuth (ax, ay)."""
    return degrees(atan2(ax, -ay))


def az_fwd(ax, ay):
    """Yaw (deg) so that a mesh growing along +Y points along azimuth (ax, ay)."""
    return degrees(atan2(-ax, ay))


def frame(loc, ydir, zdir, sc=1.0):
    """Instance matrix: local +Y -> ydir (growth), local +Z -> zdir (upper face), at loc."""
    y = Vector(ydir).normalized()
    z = Vector(zdir)
    z = z - y * z.dot(y)
    if z.length < 1e-5:
        z = Vector((0.0, 0.0, 1.0)) if abs(y.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        z = z - y * z.dot(y)
    z.normalize()
    x = y.cross(z)
    s = sc if isinstance(sc, (tuple, list)) else (sc, sc, sc)
    return Matrix(((x.x * s[0], y.x * s[1], z.x * s[2], loc[0]),
                   (x.y * s[0], y.y * s[1], z.y * s[2], loc[1]),
                   (x.z * s[0], y.z * s[1], z.z * s[2], loc[2]),
                   (0.0, 0.0, 0.0, 1.0)))


def mix_hex(a, b, f):
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * f)):02x}" for x, y in zip(ca, cb))


def tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def _co(o):
    a = np.empty(len(o.data.vertices) * 3)
    o.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def _extent(objs):
    cs = [_co(o) for o in objs if o is not None and len(o.data.vertices)]
    return tuple(np.min([c.min(axis=0) for c in cs], axis=0).tolist()), tuple(np.max([c.max(axis=0) for c in cs], axis=0).tolist())


def _far_box(cx, cy, hx, hy, z0, z1, top=True):
    """Four sides (and the top) of a box: verts, faces."""
    vs = [(cx + sx * hx, cy + sy * hy, z) for z in (z0, z1) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    fs = [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
    return vs, fs + ([(4, 5, 6, 7)] if top else [])


def split(length, lo, hi, r):
    n = max(1, int(round(length / (0.5 * (lo + hi)))))
    ws = [r.u(lo, hi) for _ in range(n)]
    k = length / sum(ws)
    e = [0.0]
    for w in ws:
        e.append(e[-1] + w * k)
    return e


def lattice_block(name, size, n_in, bev, faces, r, bulge=0.0, noskirt=(), chips=(), knock=0.0, cham=None, bx=None):
    """Stone as an open shell: only the listed faces ('-y' = outer face, '+z' = top ...) are built.
    Edges between two built faces are rounded, edges towards a missing face get a chamfer running
    back into the joint, interior vertices of a face are pushed out for a rock-faced bulge.
    chips = [(x, depth)] knocks small pieces out of the top outer arris; bx = tighter rounding of the two ends."""
    half = [0.5 * s for s in size]
    b = min(bev, 0.4 * min(half))
    cd = b if cham is None else min(cham, b)
    bs = [min(bx, b) if (bx and a == 0) else b for a in range(3)]
    ax = []
    for a in range(3):
        inner = [float(v) for v in np.linspace(-half[a] + bs[a], half[a] - bs[a], int(n_in[a]) + 1)]
        if a == 0 and chips:
            ex = []
            for x0, _ in chips:
                ex += [x0 - 0.022, x0, x0 + 0.022]
            inner = sorted([inner[0]] + [v for v in inner[1:-1] if all(abs(v - e) > 0.035 for e in ex)] + ex + [inner[-1]])
        ax.append([-half[a]] + inner + [half[a]])
    N = [len(v) - 1 for v in ax]
    index, verts, quads = {}, [], []

    def vid(i, j, k):
        key = (i, j, k)
        if key in index:
            return index[key]
        pos = [ax[0][i], ax[1][j], ax[2][k]]
        ext = [(a, -1 if key[a] == 0 else 1) for a in range(3) if key[a] in (0, N[a])]
        pres = [e for e in ext if FKEY[e] in faces]
        absn = [e for e in ext if FKEY[e] not in faces and FKEY[e] not in noskirt]
        if len(ext) >= 2:
            amt = cd if absn else (b * (0.29 if len(pres) == 2 else 0.42) if len(pres) >= 2 else 0.0)
            for a, s in pres:
                pos[a] -= s * amt * bs[a] / b
        elif len(ext) == 1 and bulge:
            a, s = ext[0]
            oth = [c for c in range(3) if c != a]
            if all(2 <= key[c] <= N[c] - 2 for c in oth):
                f = 1.0
                for c in oth:
                    f *= max(0.0, 1.0 - (pos[c] / half[c]) ** 2)
                pos[a] += s * bulge * (0.3 + 0.7 * sqrt(f)) * r.u(0.35, 1.3)
        index[key] = len(verts)
        verts.append(pos)
        return index[key]

    I, J, K = N
    if "-y" in faces:
        for i in range(I):
            for k in range(K):
                quads.append((vid(i, 0, k), vid(i + 1, 0, k), vid(i + 1, 0, k + 1), vid(i, 0, k + 1)))
    if "+y" in faces:
        for i in range(I):
            for k in range(K):
                quads.append((vid(i, J, k), vid(i, J, k + 1), vid(i + 1, J, k + 1), vid(i + 1, J, k)))
    if "+z" in faces:
        for i in range(I):
            for j in range(J):
                quads.append((vid(i, j, K), vid(i + 1, j, K), vid(i + 1, j + 1, K), vid(i, j + 1, K)))
    if "-x" in faces:
        for j in range(J):
            for k in range(K):
                quads.append((vid(0, j, k), vid(0, j, k + 1), vid(0, j + 1, k + 1), vid(0, j + 1, k)))
    if "+x" in faces:
        for j in range(J):
            for k in range(K):
                quads.append((vid(I, j, k), vid(I, j + 1, k), vid(I, j + 1, k + 1), vid(I, j, k + 1)))

    def chip(i, dep):
        for key, dy, dz in (((i, 0, K), 1.0, 1.0), ((i, 1, K), 0.0, 0.15), ((i, 0, K - 1), 0.15, 0.0)):
            if key in index:
                verts[index[key]][1] += dep * dy
                verts[index[key]][2] -= dep * dz
    for x0, dep in chips:
        i = min(range(len(ax[0])), key=lambda q: abs(ax[0][q] - x0))
        chip(i, dep)
    if knock:
        for cols in ((0, 1), (I - 1, I)):
            if r.chance(knock):
                dep = r.u(0.004, 0.011)
                for i in cols:
                    chip(i, dep)
    return mesh_object(name, verts, quads)


def set_uv(o, uv):
    """Explicit per-vertex UVs (metres; the leaf tile is 0.25 m with its midrib along U at V = 0.125)."""
    me = o.data
    lay = me.uv_layers[0] if len(me.uv_layers) else me.uv_layers.new(name="UVMap")
    for lp in me.loops:
        lay.data[lp.index].uv = uv[lp.vertex_index]
    uv_keep(o)
    return o


def leaf_base(name, L, Wd, nu, nv, m, vm=0.42, base_w=0.3, tip_w=0.3, tip_bow=0.12, base_bow=0.0,
              lift=0.5, droop=0.5, cup=0.25, twist=0.0, wave=0.0, wave_f=9.0, stalk=0.5, shade=0.3):
    """Leaf blade growing along +Y from the origin, upper face +Z: rounded outline (bowed rows), arched
    along its length, folded / cupped on the midrib, optionally twisted / ruffled. The leaf tile is
    mapped with U from base to tip so that the painted midrib runs along the blade."""
    def fn(u, v):
        s2 = 2.0 * (u - 0.5)
        if v < vm:
            wv = base_w + (1.0 - base_w) * sin(v / vm * pi / 2)
        else:
            wv = tip_w + (1.0 - tip_w) * max(cos((v - vm) / (1.0 - vm) * pi / 2), 0.0) ** 0.8
        hw_ = 0.5 * Wd * wv
        y = L * (v - tip_bow * s2 * s2 * v ** 3 + base_bow * s2 * s2 * (1.0 - v) ** 3)
        x = s2 * hw_
        zc = hw_ * (cup * (s2 * s2 - 0.35) * (1.0 - 0.45 * v) + wave * sin(v * wave_f + 1.3 * s2) * s2 * s2)
        tw = radians(twist) * v
        x, zc = x * cos(tw) - zc * sin(tw), x * sin(tw) + zc * cos(tw)
        return (x, y, L * (lift * v - droop * v * v) + zc)
    o = surface(name, lambda a, b: fn(1.0 - b, a), nv, nu, uv_len=(0.25, 0.25))
    assign(o, m)
    smooth(o, 85)
    if shade:
        paint(o, grime=lambda co, n: np.clip(1.0 - co[:, 1] / (stalk * L), 0.0, 1.0) * shade)   # darker towards the stalk
    return o


def lobed_leaf(name, R, lobes, m, per=3, sinus=0.3, notch=0.74, cup=0.3, ripple=0.06):
    """Round, palmately lobed blade (geranium / lady's mantle) on its petiole point: grows along +Y from
    the origin (the basal sinus), shallow funnel with a rippled rim."""
    cy = sinus * R
    no = lobes * per
    vs, uv = [(0.0, cy, 0.0)], [(0.25 * cy / (cy + R), 0.125)]
    for k in range(no):
        j = k % per
        q = (sinus if k == 0 else notch) if j == 0 else 0.86 + 0.14 * sin(pi * j / per)
        a = -pi / 2 + 2 * pi * k / no
        x, y = R * q * cos(a), cy + R * q * sin(a)
        vs.append((x, y, R * (cup * q ** 1.5 + ripple * (1 if j == 0 else -0.6))))
        uv.append((0.25 * clamp(y / (cy + R), 0.02, 0.98), 0.125 + 0.11 * x / R))
    o = mesh_object(name, vs, [(0, 1 + k, 1 + (k + 1) % no) for k in range(no)])
    set_uv(o, uv)
    assign(o, m)
    smooth(o, 85)
    paint(o, grime=lambda co, n: np.clip(1.0 - np.hypot(co[:, 0], co[:, 1] - cy) / (0.6 * R), 0.0, 1.0) * 0.3)
    return o


def sprig_base(name, m, Ls=0.066, pairs=3, Ll=0.026, Wl=0.019):
    """Small-leaved sprig along +Y: pairs of rounded leaflets (folded hexagons) and a terminal one."""
    out = ((0.0, 0.0, 0.0), (0.5, 0.3, 0.14), (0.42, 0.76, 0.1), (0.0, 1.0, -0.06), (-0.42, 0.76, 0.1), (-0.5, 0.3, 0.14))
    vs, fs, uv = [], [], []

    def leaflet(base, ydir, sc):
        y = Vector(ydir).normalized()
        z = Vector((0.0, 0.0, 1.0))
        z = (z - y * z.dot(y)).normalized()
        x = y.cross(z)
        b0 = len(vs)
        for px, py, pz in out:
            vs.append(tuple(Vector(base) + x * (px * Wl * sc) + y * (py * Ll * sc) + z * (pz * Wl * sc)))
            uv.append((0.02 + 0.21 * py, 0.125 + 0.2 * px))
        fs.extend((b0, b0 + i, b0 + i + 1) for i in range(1, 5))
    for i in range(pairs):
        f = (i + 0.55) / (pairs + 0.3)
        for side in (-1, 1):
            leaflet((side * 0.002, Ls * f, Ls * (0.5 * f - 0.38 * f * f)),
                    (side * (0.9 - 0.25 * f), 0.45 + 0.3 * f + 0.1 * side * (i % 2), 0.28 + 0.12 * ((i + (side > 0)) % 2)), 1.0 - 0.3 * f)
    leaflet((0.0, Ls * 0.86, Ls * 0.15), (0.12, 1.0, 0.1), 0.78)
    o = mesh_object(name, vs, fs)
    set_uv(o, uv)
    assign(o, m)
    smooth(o, 85)
    paint(o, grime=lambda co, n: np.clip(1.0 - co[:, 1] / (0.6 * Ls), 0.0, 1.0) * 0.3)
    return o


def corolla(name, R, m, n=5, elev=24.0, bend=42.0, width=1.05, cup=0.3, twist=12.0, heart=True, eye=0.55, eye_dark=0.45, light=False):
    """Flower of n separate petals: each one widens from a narrow claw, is cupped across, rises and then
    recurves towards a heart-shaped / rounded tip, and is pitched like a propeller blade so that it
    overlaps its neighbour at the base. elev / bend (deg) = pitch at the claw / total curl to the tip
    (elev 75 gives a half-open goblet)."""
    e0, bd = radians(elev), max(radians(bend), 1e-3)

    def prof(q):
        return R * (sin(e0) - sin(e0 - bd * q)) / bd, R * (cos(e0 - bd * q) - cos(e0)) / bd
    hwm = R * sin(pi / n) * width
    tw = sin(radians(twist))
    tip = ((-0.6, 1.0, 1.0), (0.0, 0.9, 1.0), (0.6, 1.0, 1.0)) if heart else ((-0.52, 0.95, 1.0), (0.0, 1.0, 1.0), (0.52, 0.95, 1.0))
    if light:                                           # small / distant flowers: one row less
        rows = (((0.0, 0.07, 0.0),), ((-1.0, 0.6, 0.94), (0.0, 0.64, 0.94), (1.0, 0.6, 0.94)), tip)
    else:
        rows = (((0.0, 0.07, 0.0),), ((-1.0, 0.43, 0.68), (0.0, 0.46, 0.68), (1.0, 0.43, 0.68)),
                ((-1.0, 0.78, 1.0), (0.0, 0.82, 1.0), (1.0, 0.78, 1.0)), tip)
    vs, fs = [], []
    for k in range(n):
        a = 2 * pi * k / n + 0.09 * sin(k * 2.3)
        er, et = (cos(a), sin(a)), (-sin(a), cos(a))
        b0 = len(vs)
        for row in rows:
            for s, q, wr in row:
                rho, z = prof(q * (1.0 - 0.06 * (k % 2)))
                lat = s * hwm * wr
                vs.append((er[0] * rho + et[0] * lat, er[1] * rho + et[1] * lat, z + abs(lat) * cup * abs(s) + lat * tw))
        fs += [(b0, b0 + 1, b0 + 2), (b0, b0 + 2, b0 + 3)]
        for r0 in ((1,) if light else (1, 4)):
            fs += [(b0 + r0, b0 + r0 + 3, b0 + r0 + 4, b0 + r0 + 1), (b0 + r0 + 1, b0 + r0 + 4, b0 + r0 + 5, b0 + r0 + 2)]
    o = mesh_object(name, vs, fs)
    assign(o, m)
    smooth(o, 85)
    paint(o, grime=lambda co, nn: np.clip(1.0 - np.hypot(co[:, 0], co[:, 1]) / (eye * R), 0.0, 1.0) * eye_dark)   # deeper throat
    return o


def bell(name, R, Lb, m, sides=5, lobed=True):
    """Tubular bell floret (foxglove / campanula): apex at the origin, mouth of radius R at z = Lb,
    flared and - on the bigger ones - cut into rounded lobes."""
    vs = [(0.0, 0.0, 0.0)]
    for k in range(sides):
        a = 2 * pi * k / sides
        vs.append((0.6 * R * cos(a), 0.6 * R * sin(a), 0.4 * Lb))
    fs = [(0, 1 + (k + 1) % sides, 1 + k) for k in range(sides)]
    o0 = 1 + sides
    for k in range(sides):
        a = 2 * pi * k / sides
        if lobed:
            a2 = a + pi / sides
            vs.append((1.2 * R * cos(a), 1.2 * R * sin(a), Lb))
            vs.append((0.84 * R * cos(a2), 0.84 * R * sin(a2), 0.88 * Lb))
        else:
            vs.append((R * cos(a), R * sin(a), Lb))
    for k in range(sides):
        k1 = (k + 1) % sides
        if lobed:
            A0, A1, T0, N0, T1 = 1 + k, 1 + k1, o0 + 2 * k, o0 + 2 * k + 1, o0 + 2 * k1
            fs += [(A0, A1, N0), (A0, N0, T0), (A1, T1, N0)]
        else:
            fs.append((1 + k, 1 + k1, o0 + k1, o0 + k))
    o = mesh_object(name, vs, fs)
    assign(o, m)
    smooth(o, 85)
    paint(o, grime=lambda co, nn: np.clip(1.0 - co[:, 2] / (0.5 * Lb), 0.0, 1.0) * 0.3)
    return o


def lobed_disc(name, R, lobes, m, per=4, notch=0.5, cup=0.35, reflex=0.0, eye=0.6, eye_dark=0.7):
    """Small corolla as one cupped fan with `lobes` rounded petals (per = rim vertices per petal)."""
    def zf(q):
        return R * (cup * q ** 1.4 - reflex * q ** 5)
    vs = [(0.0, 0.0, 0.0)]
    no = lobes * per
    for k in range(no):
        j = k % per
        q = notch if j == 0 else 0.84 + 0.16 * sin(pi * j / per)
        a = 2 * pi * k / no
        vs.append((R * q * cos(a), R * q * sin(a), zf(q) - (0.07 * R if j == 0 else 0.0)))
    o = mesh_object(name, vs, [(0, 1 + k, 1 + (k + 1) % no) for k in range(no)])
    assign(o, m)
    smooth(o, 85)
    paint(o, grime=lambda co, n: np.clip(1.0 - np.hypot(co[:, 0], co[:, 1]) / (eye * R), 0.0, 1.0) * eye_dark)   # deeper eye
    return o


def daisy_base(m_ray, m_ctr, m_stalk, R=0.031, n=9):
    """Daisy: arched strap-shaped ray petals with blunt tips in two slightly offset rows, domed disc, short stalk."""
    vs, fs = [], []
    for k in range(n):
        odd = k % 2
        a = 2 * pi * k / n + (0.1 if odd else 0.0)
        Rk = R * (0.88 if odd else 1.0)
        w = R * 0.15
        ca, sa = cos(a), sin(a)

        def P(q, side, wk):
            rr = Rk * q
            return (rr * ca - side * wk * sa, rr * sa + side * wk * ca, (0.0025 if odd else 0.0) + R * (0.26 * q - 0.4 * q * q))
        b0 = len(vs)
        vs += [P(0.18, -1, w * 0.45), P(0.18, 1, w * 0.45), P(0.68, -1, w), P(0.68, 1, w), P(1.0, -1, w * 0.5), P(1.0, 1, w * 0.5)]
        fs += [(b0, b0 + 2, b0 + 3, b0 + 1), (b0 + 2, b0 + 4, b0 + 5, b0 + 3)]
    fl = mesh_object("rays", vs, fs)
    assign(fl, m_ray)
    smooth(fl, 85)
    paint(fl, grime=lambda co, nn: np.clip(1.0 - np.hypot(co[:, 0], co[:, 1]) / (0.55 * R), 0.0, 1.0) * 0.45)
    ce = lathe("disc", [(0.25 * R, 0.0005), (0.19 * R, 0.19 * R), (0.0, 0.27 * R)], segments=5)
    assign(ce, m_ctr)
    stalk = cylinder("stalk", 0.0016, 0.09, loc=(0, 0, -0.045), verts=3, cap=False)
    assign(stalk, m_stalk)
    return join([fl, ce, stalk], "fl_daisy")


# ---------------------------------------------------------------------------------- build
def build(p, rng):
    round_ = p.shape == "round"
    dressed = p.kerb == "dressed"
    L = p.length
    W = L if round_ else p.width
    h = p.kerb_height
    t = 0.17 if dressed else 0.21                       # wall thickness
    cop_t = clamp(h * 0.22, 0.055, 0.08) if dressed else 0.0
    cop_over = 0.025 if dressed else 0.0
    hw = h - cop_t                                      # wall height under the coping
    pl = 0.07 if dressed and h >= 0.3 else 0.0         # projecting plinth course
    n_c = 1 if hw - pl <= 0.3 else 2
    ch = (hw - pl) / n_c
    drop = 0.0 if dressed else 0.3                      # field stones: how much lower than the tallest a stone may be
    ms = p.seed * 7 + 3
    far = lod() == 2                                    # far level: same plan and draws, plain stand-ins (see the docstring)

    pal, sp_cols = PALETTES[p.palette]
    st_wall = mat("stone_rough", p.stone)
    cope_tint = mix_hex(p.stone, "#dcdcd6", 0.55)       # paler, cooler dressed stone; also the lime pointing
    # dressed coping, plinth and the lime pointing share one fine sandy material (no directional veining)
    st_cope = mat("mortar", cope_tint, scale=0.35) if dressed else mat("mortar", "#6a5a48")
    m_soil = mat("soil")
    m_la = mat("leaf", "#d2ec6c")                       # lime / fresh yellow-green
    m_lb = mat("leaf", "#6fae62", rough=0.8)            # deep green, matt
    m_lc = mat("leaf", "#bfe0ff", rough=0.9)            # glaucous blue-green
    m_leaf = (m_la, m_lb, m_lc)
    m_pet = [mat("wax", c) for c in pal]               # leaf albedo is green: petals need a white-based material
    m_ctr = mat("wax", "#e3a51c")

    # ------------------------------------------------------------------ kerb layout
    sides = []
    if round_:
        Rc = L / 2 - t / 2 - cop_over
        sides.append({"kind": "arc", "R": Rc, "len": 2 * pi * Rc})
        Ri = Rc - t / 2
        xi = yi = Ri
        area = pi * Ri * Ri
        cp = 0.0
        c = L / 2 - Rc
    else:
        cp = t + (0.08 if dressed else 0.05)             # corner pier / corner boulder size
        c = cp / 2
        x1, y1 = L / 2 - cp + 0.012, W / 2 - cp + 0.012
        xc, yc = L / 2 - c, W / 2 - c
        sides.append({"kind": "line", "P0": (-x1, -yc), "dir": (1, 0), "n": (0, -1), "len": 2 * x1})
        sides.append({"kind": "line", "P0": (xc, -y1), "dir": (0, 1), "n": (1, 0), "len": 2 * y1})
        sides.append({"kind": "line", "P0": (x1, yc), "dir": (-1, 0), "n": (0, 1), "len": 2 * x1})
        sides.append({"kind": "line", "P0": (-xc, y1), "dir": (0, -1), "n": (-1, 0), "len": 2 * y1})
        xi, yi = xc - t / 2, yc - t / 2
        area = 4 * xi * yi
    perim = sum(s["len"] for s in sides)

    def fmap(sd, s, d):
        """(distance along the kerb centreline, offset outwards) -> world x, y."""
        if sd["kind"] == "arc":
            th = s / sd["R"]
            r = sd["R"] + d
            return r * cos(th), r * sin(th)
        return sd["P0"][0] + sd["dir"][0] * s + sd["n"][0] * d, sd["P0"][1] + sd["dir"][1] * s + sd["n"][1] * d

    def fnorm(sd, s):
        if sd["kind"] == "arc":
            th = s / sd["R"]
            return cos(th), sin(th)
        return sd["n"]

    def rand_perim(r, margin=0.12):
        q = r.u(0, perim)
        for sd in sides:
            if q <= sd["len"]:
                return sd, (q if sd["kind"] == "arc" else clamp(q, margin, sd["len"] - margin))
            q -= sd["len"]
        return sides[-1], sides[-1]["len"] * 0.5

    def put(o, sd, sc, do, zc, skew=0.0, sm=50.0, roll=0.0, lean=0.0):
        def f(co):
            zz = max(0.0, zc + co.z + co.x * roll)
            x, y = fmap(sd, sc + co.x, do - co.y + co.x * skew + zz * lean)
            return Vector((x, y, zz))
        transform_verts(o, f)
        uv_box(o)
        smooth(o, sm)
        return o

    def butted(sd, a, b, gap):
        """Stone ends: half a joint towards the neighbour, square and tight where the stone butts a corner pier."""
        line = sd["kind"] == "line"
        first, last = line and a < 1e-6, line and b > sd["len"] - 1e-6
        return (0.0015 if first else a + gap), (sd["len"] - 0.0015 if last else b - gap), \
               (("-x",) if first else ()) + (("+x",) if last else ())

    # ------------------------------------------------------------------ stones
    rk = rng.fork("kerb")
    walls, copes = [], []
    pj = []                                             # plinth joints: (side, s)
    rj = []                                             # rustic joints packed with earth: (side, s, z0, z1)
    mq = []                                             # pointing: (side, s0, s1, z0, z1) flat strips lying in the joints
    cell = 0.075 if perim * n_c < 9.0 else 0.1
    rcell = 0.1 if perim * n_c < 6.5 else 0.13
    for sd in sides:
        arc = sd["kind"] == "arc"
        sd["joints"], sd["tops"], sd["stones"] = [], [], []
        if dressed:
            if n_c == 1:
                hs = [hw - pl]
            else:
                f0 = rk.u(0.53, 0.6)
                hs = [(hw - pl) * f0, (hw - pl) * (1 - f0)]
            zc = pl
            for chh in hs:
                edges = split(sd["len"], 0.24, 0.74, rk)
                ph = rk.u(0, 1) if arc else 0.0
                for a, b in zip(edges[:-1], edges[1:]):
                    ln = max(0.06, b - a - 0.007)
                    parts = [(zc + 0.003, chh - 0.006)]
                    if chh > 0.16 and ln < 0.52 and rk.chance(0.42):       # snecked: two thin stones in one slot
                        h1 = (chh - 0.006) * rk.u(0.4, 0.6)
                        parts = [(zc + 0.003, h1 - 0.0035), (zc + 0.003 + h1 + 0.0035, chh - 0.006 - h1 - 0.0035)]
                        mq.append((sd, a + ph + 0.006, b + ph - 0.006, zc + 0.003 + h1 - 0.0065, zc + 0.003 + h1 + 0.0065))
                    for zb, zh in parts:
                        o = lattice_block("kerb_wall", (ln, t, zh), (max(1, round(ln / cell)), 1, max(1, round(zh / cell))),
                                          0.007, {"-y"}, rk, bulge=rk.u(0.008, 0.018), cham=0.005)
                        walls.append(put(o, sd, 0.5 * (a + b) + ph, rk.u(0.0, 0.0015), zb + zh / 2))
                    sd["joints"].append((a + ph, zc, zc + chh))
                for e in (edges[:-1] if arc else edges):                   # perpends
                    mq.append((sd, max(e + ph - 0.006, 0.0032) if not arc else e + ph - 0.006,
                               min(e + ph + 0.006, sd["len"] - 0.0032) if not arc else e + ph + 0.006, zc + 0.0065, zc + chh - 0.0065))
                cuts = [e + ph for e in edges] if arc else [0.002] + edges[1:-1] + [sd["len"] - 0.002]
                for zj in ([zc, hw] if zc + chh > hw - 1e-4 else [zc]):   # bed joints (and the one under the coping)
                    for a, b in zip(cuts[:-1], cuts[1:]):
                        mq.append((sd, a, b, zj - 0.0065, zj + (0.0065 if zj < hw - 1e-4 else 0.002)))
                zc += chh
            # coping: dressed, overhanging, a few chipped arrises
            edges = split(sd["len"], 0.5, 0.88, rk)
            ph = rk.u(0, 1) if arc else 0.0
            for a, b in zip(edges[:-1], edges[1:]):
                a2, b2, butt = butted(sd, a, b, 0.004)
                ln = b2 - a2
                chips = [(rk.u(-0.3, 0.3) * ln, rk.u(0.012, 0.022))] if rk.chance(0.42) else []
                o = lattice_block("kerb_coping", (ln, t + 2 * cop_over, cop_t), (max(1, round(ln / 0.2)), 1, 1), 0.009,
                                  {"-y", "+y", "+z"}, rk, noskirt=butt, chips=chips, knock=0.0 if butt else 0.35, cham=0.0055)
                copes.append(put(o, sd, 0.5 * (a2 + b2) + ph, rk.u(-0.002, 0.002), hw + cop_t / 2 + rk.u(-0.0025, 0.0015)))
            if pl:
                edges = split(sd["len"], 0.55, 0.98, rk)
                ph = rk.u(0, 1) if arc else 0.0
                for a, b in zip(edges[:-1], edges[1:]):
                    a2, b2, butt = butted(sd, a, b, 0.0035)
                    ln = b2 - a2
                    o = lattice_block("kerb_plinth", (ln, t + 2 * cop_over, pl), (max(1, round(ln / 0.25)), 1, 1), 0.012,
                                      {"-y", "+z"}, rk, noskirt=("-z",) + butt, knock=0.0 if butt else 0.25, cham=0.0055)
                    copes.append(put(o, sd, 0.5 * (a2 + b2) + ph, rk.u(-0.002, 0.002), pl / 2))
                    sd["joints"].append((a + ph, -1.0, pl))
                    if arc or a > 1e-6:
                        pj.append((sd, a + ph))
        else:
            # dry-laid field stones: mixed lengths (18-45 cm) and heights, out of level, staggered in and out,
            # ends fitted closely; the joints are packed with earth and here and there a pinning stone
            for ci in range(n_c):
                top = ci == n_c - 1
                edges = split(sd["len"], 0.18, 0.45, rk)
                ph = rk.u(0, 1) if arc else 0.0
                z0 = ci * ch * 0.97
                row = []
                for a, b in zip(edges[:-1], edges[1:]):
                    ln = max(0.07, b - a - rk.u(0.004, 0.012))
                    zz = ch * (rk.u(1.0 - drop, 1.0) if top else 1.0)
                    dd = t * rk.u(0.84, 1.08)
                    do = rk.u(-0.016, 0.016)
                    lean = rk.u(-0.09, 0.03)
                    faces = {"-y", "+y", "-x", "+x"} | ({"+z"} if top else set())
                    o = lattice_block("kerb_stone", (ln, dd, zz), (max(2, round(ln / rcell)), 1, max(1, round(zz / 0.1))),
                                      rk.u(0.02, 0.034), faces, rk, bulge=rk.u(0.008, 0.02), noskirt=("-z", "+z"), bx=rk.u(0.008, 0.015))
                    displace_noise(o, strength=0.014, scale=0.15, seed=int(rk.u(0, 9999)))
                    put(o, sd, 0.5 * (a + b) + ph, do, z0 + zz / 2, skew=rk.u(-1, 1) * 0.05, sm=60.0,
                        roll=(rk.u(-1, 1) * 0.07 if top else 0.0), lean=lean)
                    paint(o, grime=rk.u(0.0, 0.32), wear=rk.u(0.0, 0.25))        # every stone its own tone
                    walls.append(o)
                    face = dd / 2 + do + min(0.0, lean) * (z0 + zz)
                    row.append((z0 + zz, face))
                    sd["stones"].append((a + ph, b + ph, z0, z0 + zz, face + 0.012))
                    if top:
                        sd["tops"].append((a + ph, b + ph, z0 + zz))
                        sd["joints"].append((a + ph, 0.0, z0 + zz))
                for i, e in enumerate(edges[:-1] if arc else edges[1:-1]):
                    q0, q1 = row[i if arc else i + 1], row[i - 1 if arc else i]
                    zt, fj = min(q0[0], q1[0]) - 0.02, min(q0[1], q1[1]) - 0.014
                    rj.append((sd, e + ph, z0, zt, fj))
                    if top and rk.chance(0.4):                                 # pinning stone wedged into the joint
                        ps_ = (rk.u(0.045, 0.085), rk.u(0.05, 0.075), rk.u(0.035, 0.055))
                        o = lattice_block("kerb_pin", ps_, (1, 1, 1), 0.012, {"-y", "+y", "-x", "+x", "+z"}, rk, noskirt=("-z",))
                        displace_noise(o, strength=0.006, scale=0.06, seed=int(rk.u(0, 9999)))
                        put(o, sd, e + ph + rk.u(-0.01, 0.01), fj + 0.012 - ps_[1] / 2 + rk.u(0.0, 0.012), zt + 0.012 - ps_[2] / 2,
                            skew=rk.u(-0.3, 0.3), sm=60.0, roll=rk.u(-0.2, 0.2))
                        paint(o, grime=rk.u(0.1, 0.4))
                        walls.append(o)
    if not dressed and not round_:
        # corner boulders: one big stone turns each corner (the runs butt against it)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = sx * (L / 2 - cp / 2), sy * (W / 2 - cp / 2)
                zz = hw * rk.u(0.9, 1.0)
                sh = lattice_block("kerb_corner", (cp * rk.u(0.94, 1.0), cp * rk.u(0.94, 1.0), zz), (2, 2, max(2, round(zz / 0.1))),
                                   rk.u(0.03, 0.042), {"-x", "+x", "-y", "+y", "+z"}, rk, bulge=0.014, noskirt=("-z",))
                displace_noise(sh, strength=0.014, scale=0.16, seed=int(rk.u(0, 9999)))
                yaw, rl_ = radians(rk.u(-7, 7)), rk.u(-0.05, 0.05)
                transform_verts(sh, lambda co, cx=cx, cy=cy, zc=zz / 2, yaw=yaw, rl_=rl_: Vector((
                    cx + co.x * cos(yaw) - co.y * sin(yaw), cy + co.x * sin(yaw) + co.y * cos(yaw), max(0.0, co.z + zc + co.x * rl_))))
                uv_box(sh)
                smooth(sh, 60)
                paint(sh, grime=rk.u(0.0, 0.3), wear=rk.u(0.0, 0.25))
                walls.append(sh)
    if dressed and not round_:
        ph_ = h + 0.016
        ALL4 = {"-x", "+x", "-y", "+y"}
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = sx * (L / 2 - cp / 2), sy * (W / 2 - cp / 2)
                z0 = pl + 0.003 if pl else 0.0
                sh = lattice_block("pier", (cp - 0.03, cp - 0.03, ph_ - z0), (2, 2, 2 if ph_ < 0.5 else 3), 0.012, ALL4, rk,
                                   bulge=0.012, noskirt=("-z", "+z"))
                transform_verts(sh, lambda co, cx=cx, cy=cy, zc=z0 + (ph_ - z0) / 2: Vector((co.x + cx, co.y + cy, max(0.0, co.z + zc))))
                uv_box(sh)
                smooth(sh, 50)
                walls.append(sh)
                cap = lattice_block("pier_cap", (cp, cp, 0.06), (2, 2, 1), 0.016, ALL4 | {"+z"}, rk, noskirt=("-z",), knock=0.3)
                # shallow weathered (four-way falling) top
                transform_verts(cap, lambda co, cx=cx, cy=cy: Vector((
                    co.x + cx, co.y + cy,
                    co.z + ph_ + 0.028 + (0.012 * (1 - max(abs(co.x), abs(co.y)) / (cp / 2)) if co.z > 0.02 else 0.0))))
                uv_box(cap)
                smooth(cap, 50)
                copes.append(cap)
                if pl:
                    pb = lattice_block("pier_base", (cp, cp, pl + 0.012), (1, 1, 1), 0.012, ALL4 | {"+z"}, rk, noskirt=("-z",))
                    transform_verts(pb, lambda co, cx=cx, cy=cy: Vector((co.x + cx, co.y + cy, max(0.0, co.z + (pl + 0.012) / 2))))
                    uv_box(pb)
                    smooth(pb, 50)
                    copes.append(pb)

    # bedding / pointing behind the stones: lime mortar recessed ~4 mm (dressed), packed earth (rustic)
    rec = 0.003

    def strip(sd, prof):
        arc = sd["kind"] == "arc"
        n = max(12, int(sd["len"] / (0.34 if far else 0.11))) if arc else 1
        cnt = n if arc else n + 1
        k = len(prof)
        vs, fs = [], []
        for i in range(cnt):
            s = sd["len"] * i / n if arc else -0.02 + (sd["len"] + 0.04) * i / n
            for d, z in prof:
                x, y = fmap(sd, s, d)
                vs.append((x, y, z))
        for i in range(n):
            i1 = (i + 1) % cnt
            for j in range(k - 1):
                fs.append((i * k + j, i1 * k + j, i1 * k + j + 1, i * k + j + 1))
        return mesh_object("bedding", vs, fs)

    # (separate flat / convex pieces with their vertices out in the open: the exporter bakes cavity grime and AO per vertex)
    if dressed:
        wf, ww = t / 2 + cop_over - rec, t / 2 - rec
        profs = [[(ww - 0.012, 0.0), (ww - 0.012, hw)], [(ww - 0.004, hw), (wf, hw), (wf, h - 0.0045), (-wf, h - 0.0045), (-wf, hw - 0.03)]]
        if pl:
            profs.append([(wf - 0.01, 0.0), (wf - 0.01, pl - rec - 0.01), (ww - 0.004, pl - rec - 0.01)])
    else:
        wr = t / 2 - 0.055
        zb = hw - ch * drop - 0.035
        profs = [[(wr, 0.0), (wr, zb), (-wr, zb), (-wr, zb - 0.08)]]
    if far:
        # far level: the stones were laid for their draws and measures; plain sections and boxes stand in for them
        rj, pj, mq = [], [], []
        corner = [(_co(o).min(axis=0), _co(o).max(axis=0)) for o in walls if o.name.startswith("kerb_corner")]
        delete(walls + copes)
        walls, copes, profs = [], [], []
        wo, wi = t / 2 + cop_over, t / 2 + 0.012

        def boxed(lst, name, *a, **k):
            lst.append(mesh_object(name, *_far_box(*a, **k)))
        if dressed:
            profs = [[(wo, hw), (wo, h), (-wo, h), (-wo, h - 0.07)]] + ([[(wo, 0.0), (wo, pl)]] if pl else [])
            walls = [strip(sd, [(wo if pl else wi, pl), (wi, hw)]) for sd in sides]
            if not round_:
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        cx, cy = sx * (L / 2 - cp / 2), sy * (W / 2 - cp / 2)
                        boxed(walls, "pier", cx, cy, cp / 2 - 0.009, cp / 2 - 0.009, pl, h + 0.016, top=False)
                        boxed(copes, "pier_cap", cx, cy, cp / 2, cp / 2, h + 0.014, h + 0.08)
                        if pl:
                            boxed(copes, "pier_base", cx, cy, cp / 2, cp / 2, 0.0, pl + 0.012)
        else:
            zin = hw - ch * drop - 0.045
            for sd in sides:
                st_ = [(0.5 * (a + b), z1 - 0.012, f - 0.018) for a, b, _, z1, f in sd["stones"][-len(sd["tops"]):]]
                if sd["kind"] == "line":
                    st_ = [(-0.01, *st_[0][1:])] + st_ + [(sd["len"] + 0.01, *st_[-1][1:])]
                # (a row of vertices just above the paving keeps the dark foot of the stones at the foot)
                vs = [(*fmap(sd, s, d), z) for s, zt, f in st_
                      for d, z in ((f, 0.0), (f, 0.07), (f, zt), (-0.48 * t, zt), (-0.48 * t, zin))]
                n = len(st_)
                fs = [(i * 5 + j, i1 * 5 + j, i1 * 5 + j + 1, i * 5 + j + 1)
                      for i in range(n if sd["kind"] == "arc" else n - 1) for i1 in ((i + 1) % n,) for j in range(4)]
                if sd["kind"] == "line":                # (closed ends: next to a corner boulder the eye would look into the kerb)
                    for i, sg in ((0, -1.0), (n - 1, 1.0)):
                        nrm = np.cross(np.subtract(vs[i * 5 + 2], vs[i * 5]), np.subtract(vs[i * 5 + 3], vs[i * 5]))
                        flip = sg * (nrm[0] * sd["dir"][0] + nrm[1] * sd["dir"][1]) < 0.0
                        fs += [tuple(i * 5 + k for k in (q[::-1] if flip else q)) for q in ((1, 2, 3), (1, 3, 4), (0, 1, 4))]
                walls.append(mesh_object("kerb_wall", vs, fs))
            for mn_, mx_ in corner:
                boxed(walls, "kerb_corner", 0.5 * (mn_[0] + mx_[0]), 0.5 * (mn_[1] + mx_[1]), 0.5 * (mx_[0] - mn_[0]) - 0.015,
                      0.5 * (mx_[1] - mn_[1]) - 0.015, 0.0, mx_[2] - 0.015)
        for o in walls + copes:
            uv_box(o)
            smooth(o, 50)
    beds = [strip(sd, prof) for sd in sides for prof in profs]
    for sd, e, z0, z1, dj in rj:                         # earth packed into the joints, a little behind the faces
        prof = [(dj, z0), (dj, z1), (-0.06, z1 + 0.004), (-0.06, z1 - 0.07)]
        vs = [(*fmap(sd, e + ds, d), z) for ds in (-0.024, 0.024) for d, z in prof]
        beds.append(mesh_object("packing", vs, [(j, j + 4, j + 5, j + 1) for j in range(3)]))
    for sd, e in pj:
        prof = [(wf, 0.0), (wf, pl - rec), (ww - 0.004, pl - rec)]
        vs = [(*fmap(sd, e + ds, d), z) for ds in (-0.0075, 0.0075) for d, z in prof]
        beds.append(mesh_object("pointing_pl", vs, [(j, j + 3, j + 4, j + 1) for j in range(2)]))
    if mq:
        vs, fs = [], []
        for sd, s0, s1, z0, z1 in mq:
            n = max(1, int((s1 - s0) / 0.1 + 0.999)) if sd["kind"] == "arc" else 1
            for i in range(n):
                sa, sb = s0 + (s1 - s0) * i / n, s0 + (s1 - s0) * (i + 1) / n
                k = len(vs)
                vs += [(*fmap(sd, sa, ww), z0), (*fmap(sd, sb, ww), z0), (*fmap(sd, sb, ww), z1), (*fmap(sd, sa, ww), z1)]
                fs.append((k, k + 1, k + 2, k + 3))
        beds.append(mesh_object("pointing", vs, fs))
    for o in walls:
        assign(o, st_wall)
    for o in copes + beds:
        assign(o, st_cope)

    # where the trailing plants come over the kerb (one or two swags) - the stone below them stays damp
    rt = rng.fork("trail")
    swag_pl = []
    if p.overgrowth >= 0.08:
        for ci in range(2 if p.overgrowth >= 0.6 and perim > 4.5 else 1):
            if ci == 0 and not round_:
                swag_pl.append((sides[0], sides[0]["len"] * rt.u(0.24, 0.76)))
            else:
                swag_pl.append(rand_perim(rt, 0.3))
    swag_xy = [fmap(sd, s0, t / 2 + cop_over) for sd, s0 in swag_pl]

    def near_swag(co, n):
        m = np.zeros(len(co))
        for sx_, sy_ in swag_xy:
            m = np.maximum(m, np.exp(-((co[:, 0] - sx_) ** 2 + (co[:, 1] - sy_) ** 2) / 0.022))
        return m

    wall = join(walls, "kerb_wall")
    mossy = min(1.0, p.moss * 1.3)
    north = facing((0, 1, 0))
    shade = lambda co, n: 0.3 + 0.7 * np.clip(n[:, 1], 0.0, 1.0)             # the side away from the sun stays damp
    foot = mask_mul(height_ramp(0.0, 0.2), noise_mask(0.3, seed=ms + 5, lo=-0.7, hi=0.75))   # splash-back / damp rising from the paving
    drip = mask_mul(height_ramp(hw + 0.01, hw - 0.15), noise_mask(0.26, seed=ms, lo=-0.6, hi=0.75))   # run-off under the overhang
    # damp bloom: soft, low contrast, strongest at the foot and on the shaded (north) side
    paint(wall, special=mask_mul(height_ramp(0.0, 0.26), noise_mask(0.45, seed=ms + 1, lo=0.1, hi=0.95), 0.4 * mossy))
    paint(wall, special=mask_mul(north, noise_mask(0.6, seed=ms + 2, lo=0.05, hi=0.9), 0.2 * mossy))
    paint(wall, grime=mask_mul(foot, 0.5))
    paint(wall, grime=mask_mul(near_swag, height_ramp(h, -0.25 * h), 0.55))
    if dressed:
        paint(wall, grime=mask_mul(drip, 0.6))
    else:
        paint(wall, special=mask_mul(facing((0, 0, 1)), noise_mask(0.5, seed=ms + 3, lo=0.1, hi=0.9), 0.3 * mossy))
    # joints: the pointing / packed earth holds the damp - dark at the foot and under the coping, a green-brown bloom
    # low down that is strongest on the shaded side
    bed = join(beds, "kerb_bedding") if beds else None    # (the far level of field stones has no bedding)
    if bed:
        paint(bed, grime=mask_mul(foot, 0.7))
        paint(bed, special=mask_mul(shade, height_ramp(0.0, max(0.24, 0.75 * h)), noise_mask(0.35, seed=ms + 14, lo=-0.3, hi=0.7), min(1.0, 1.5 * p.moss)))
        if dressed:
            paint(bed, grime=mask_mul(drip, 0.6))
    if copes:
        cope = join(copes, "kerb_coping")
        paint(cope, grime=mask_mul(noise_mask(0.16, seed=ms + 12, lo=0.3, hi=0.9), 0.2))        # faint non-directional mottling
        paint(cope, special=mask_mul(height_ramp(0.0, 0.14), noise_mask(0.45, seed=ms + 1, lo=0.1, hi=0.95), 0.4 * mossy),
              grime=mask_mul(height_ramp(0.0, 0.11), noise_mask(0.3, seed=ms + 5, lo=-0.7, hi=0.75), 0.48))
        paint(cope, special=mask_mul(north, noise_mask(0.6, seed=ms + 2, lo=0.05, hi=0.9), 0.18 * mossy))
        paint(cope, special=mask_mul(facing((0, 0, 1)), noise_mask(0.7, seed=ms + 4, lo=0.2, hi=0.95), 0.2 * mossy),
              wear=mask_mul(facing((0, 0, 1)), noise_mask(0.6, seed=ms + 6, lo=0.2, hi=0.9), 0.35))   # people sit on it
        paint(cope, grime=mask_mul(near_swag, 0.45))
        cope = join([cope, bed], "kerb_coping")
    else:
        cope = bed
    stone_tris = tris(wall) + (tris(cope) if cope else 0)

    # ------------------------------------------------------------------ soil
    if dressed:
        zs = h - 0.04
    else:
        zs = hw - ch * drop - 0.012
    mound = 0.04 + 0.05 * min(W, 1.6)

    def soil_z(x, y):
        if round_:
            f = max(0.0, 1.0 - (hypot(x, y) / Ri) ** 2)
        else:
            f = max(0.0, 1.0 - (x / xi) ** 2) * max(0.0, 1.0 - (y / yi) ** 2)
        return zs + mound * f ** 0.75

    def edge_d(x, y):                                   # distance to the inner face of the kerb
        if round_:
            return Ri - hypot(x, y)
        return min(xi - abs(x), yi - abs(y))

    sx_, sy_ = 2 * xi + 0.05, 2 * yi + 0.05
    soil = grid("soil", size=(sx_, sy_), cuts=lod_exact((max(3, int(sx_ / 0.4)), max(3, int(sy_ / 0.4)))) if far else
                (max(4, int(sx_ / 0.11)), max(4, int(sy_ / 0.11))))

    def soil_f(co):
        x, y = co.x, co.y
        if round_:
            r = hypot(x, y)
            if r > Ri + 0.025:
                k = (Ri + 0.025) / r
                x, y = x * k, y * k
        return Vector((x, y, soil_z(x, y)))
    transform_verts(soil, soil_f)
    displace_noise(soil, strength=0.012, scale=0.09, seed=p.seed + 5, axis_mask=(0, 0, 1))
    assign(soil, m_soil)
    paint(soil, special=mask_mul(noise_mask(0.3, seed=ms + 8), 0.5))

    # ------------------------------------------------------------------ planting plan
    M = defaultdict(list)                               # base mesh key -> instance matrices
    stems = []
    far_sp, far_ho, far_gr, far_sw = [], [], [], []     # far level: spikes, hostas, grasses, swags as planned
    rp = rng.fork("plan")

    def inside(x, y, m):
        return edge_d(x, y) > m

    taken = []                                          # (x, y, radius) of tall / foliage plants

    def free(x, y, rad):
        return all(hypot(x - a, y - b) > rad + rr for a, b, rr in taken)

    def rand_pt(r, fx, fy, margin, rad):
        for _ in range(60):
            if round_:
                a, q = r.u(0, 2 * pi), Ri * fx * sqrt(r.u())
                x, y = q * cos(a), q * sin(a)
            else:
                x, y = r.u(-xi, xi) * fx, r.u(-yi, yi) * fy
            if inside(x, y, margin) and free(x, y, rad):
                return x, y
        return None

    # tall spikes in 1-3 groups along the spine
    n_sp = min(6, int(round(clamp(area * 2.8, 2, 6) * p.spikes)))
    spike_pts, centres = [], []
    if n_sp > 0:
        n_cl = max(1, int(round(n_sp / 3.0)))
        centres = [rand_pt(rp, 0.42 if round_ else 0.8, 0.3, 0.12, 0.0) or (0.0, 0.0) for _ in range(n_cl)]
        for i in range(n_sp):
            cx, cy = centres[i % n_cl]
            for _ in range(30):
                a, q = rp.u(0, 2 * pi), rp.u(0.03, 0.17)
                x, y = cx + q * cos(a), cy + q * sin(a)
                if inside(x, y, 0.1) and all(hypot(x - a2, y - b2) > 0.085 for a2, b2, _ in spike_pts):
                    spike_pts.append((x, y, i % n_cl))
                    break
        for x, y, _ in spike_pts:
            taken.append((x, y, 0.05))
    # foliage clumps
    hostas, grasses = [], []
    for _ in range(min(5, 1 + int(round(area * 0.9)))):
        pt = rand_pt(rp, 0.9, 0.85, 0.16, 0.16)
        if pt:
            hostas.append(pt)
            taken.append((pt[0], pt[1], 0.13))
    for _ in range(min(4, 1 + int(round(area * 0.7)))):
        pt = rand_pt(rp, 0.75, 0.5, 0.14, 0.1)
        if pt:
            grasses.append(pt)
            taken.append((pt[0], pt[1], 0.07))

    # low bedding plants on a jittered hex grid, coloured by drift
    big = clamp((area / 1.55) ** 0.5, 1.0, 1.25)          # big beds: larger plants (triangle budget)
    young = clamp((0.8 - p.density) / 0.4)                # 0 = established mass .. 1 = freshly planted
    sp = 0.168 * big / sqrt(p.density)
    pts = []
    e_off = 0.075                                       # edging row just inside the kerb, then the infill
    if round_:
        n_e = max(6, int(2 * pi * (Ri - e_off) / (sp * 1.05)))
        ring = [((Ri - e_off) * cos(2 * pi * k / n_e), (Ri - e_off) * sin(2 * pi * k / n_e)) for k in range(n_e)]
    else:
        ex, ey = xi - e_off, yi - e_off
        ring = []
        for (ax_, ay_), (bx_, by_) in (((-ex, -ey), (ex, -ey)), ((ex, -ey), (ex, ey)), ((ex, ey), (-ex, ey)), ((-ex, ey), (-ex, -ey))):
            n_e = max(2, int(round(hypot(bx_ - ax_, by_ - ay_) / (sp * 1.05))))
            ring += [(ax_ + (bx_ - ax_) * k / n_e, ay_ + (by_ - ay_) * k / n_e) for k in range(n_e)]
    for x, y in ring:
        x, y = x + rp.u(-0.2, 0.2) * sp, y + rp.u(-0.2, 0.2) * sp
        k = 1.0
        if round_:
            k = min(1.0, (Ri - e_off + 0.015) / max(hypot(x, y), 1e-6))
        else:
            x, y = clamp(x, -ex - 0.015, ex + 0.015), clamp(y, -ey - 0.015, ey + 0.015)
        if free(x * k, y * k, -0.02):
            pts.append((x * k, y * k))
    g_m = e_off + 0.6 * sp
    ny_ = int(yi / (sp * 0.866)) + 2
    nx_ = int(xi / sp) + 2
    for j in range(-ny_, ny_ + 1):
        for i in range(-nx_, nx_ + 1):
            x = (i + 0.5 * (j % 2)) * sp + rp.u(-0.3, 0.3) * sp
            y = j * sp * 0.866 + rp.u(-0.3, 0.3) * sp
            if inside(x, y, g_m) and free(x, y, 0.0):
                pts.append((x, y))
    n_dr = 3 + int(area * 1.3)
    order = [0, 1, 2]
    order.sort(key=lambda _: rp.u())
    drifts = []
    for k in range(n_dr):
        pt = rand_pt(rp, 1.0, 1.0, 0.0, -1.0) or (0.0, 0.0)
        drifts.append((pt[0], pt[1], order[k % 3]))

    def drift_at(x, y):
        return min(drifts, key=lambda d: hypot(x - d[0], y - d[1]))[2]

    def edge_f(x, y):                                   # 0 at the kerb .. 1 in the middle
        return clamp(edge_d(x, y) / max(min(xi, yi), 1e-3))

    # foliage: four kinds of leaf in three greens; every drift mixes two or three of them
    SPECIES = {     # leaf bases, material weights (0 lime, 1 deep green, 2 blue-green), leaves per plant, elevation at crest / rim,
                    # height, height of the dark core, how far from the centre the leaves spring
        "ovate": ((("lf", "ov"), ("lf", "ovB"), ("lf", "ov")), ((1, 4), (0, 1)), (12, 15), (64, 0), 1.0, 0.6, 0.66),
        "lobed": ((("lobe", "a"), ("lobe", "b")), ((0, 4), (1, 1)), (10, 12), (48, -14), 0.9, 0.62, 0.72),
        "lance": ((("lf", "lance"), ("lf", "lanceB")), ((2, 4), (1, 1)), (16, 20), (66, 4), 1.1, 0.55, 0.62),
        "small": ((("sprig", "a"), ("sprig", "b")), ((1, 3), (2, 2)), (10, 12), None, 0.8, 0.78, 0.8),      # tight cushion: sprigs shingle the core
    }
    MIX = ((("ovate", 5), ("lobed", 3), ("small", 1)), (("lance", 5), ("lobed", 2), ("ovate", 2)), (("small", 5), ("lance", 2), ("lobed", 2)))
    rl = rng.fork("low")
    plants = []
    for x, y in pts:
        ci = drift_at(x + rl.u(-0.09, 0.09), y + rl.u(-0.09, 0.09))
        col = ci if not rl.chance(0.07) else int(rl.u(0, 2.999))
        spec = rl.weighted(MIX[ci])
        r = min(sp * 0.72, 0.125 * big * (1.0 - 0.25 * young)) * rl.u(0.84, 1.2)
        H = min(0.2, (0.13, 0.16, 0.11)[ci] * SPECIES[spec][4] * rl.u(0.7, 1.34) * (0.85 + 0.3 * edge_f(x, y)) * (1.0 - 0.3 * young))
        plants.append({"x": x, "y": y, "r": r, "H": H, "kind": ci, "col": col, "spec": spec, "mi": rl.weighted(SPECIES[spec][1])})
    domes = [(q["x"], q["y"], q["r"], SPECIES[q["spec"]][5] * q["H"]) for q in plants]     # the under-canopy sits below the leaves
    domes += [(x, y, 0.17, 0.075) for x, y in hostas] + [(x, y, 0.11, 0.07) for x, y in grasses]
    domes += [(x, y, 0.1, 0.08) for x, y, _ in spike_pts]

    cu_cell = 0.064 * big
    cand = []                                           # (priority, key, matrix): low-plant parts, culled to the triangle budget
    dcand = []                                          # daisies (capped separately)
    top_max = (h if dressed else hw) + 0.004

    # ------------------------------------------------------------------ under-canopy: the dark mass of stems and shade below the leaves
    def canopy(x, y):
        best = 0.0
        for dx, dy, dr, dh in domes:
            d2 = ((x - dx) ** 2 + (y - dy) ** 2) / (dr * dr)
            if d2 < 1.0:
                best = max(best, dh * (1.0 - d2) ** 0.55)
        return best * smoothstep(0.0, 0.06, edge_d(x, y))

    def cush(u, v):
        x, y = (u - 0.5) * 2 * xi, (v - 0.5) * 2 * yi
        if round_:
            r = hypot(x, y)
            if r > Ri - 0.004:
                k = (Ri - 0.004) / r
                x, y = x * k, y * k
        cz = canopy(x, y)
        return (x, y, soil_z(x, y) + (cz if cz > 0.006 else -0.035))
    cu = None
    if not far:
        cu = surface("canopy", cush, max(4, int(2 * xi / cu_cell)), max(4, int(2 * yi / cu_cell)))
        displace_noise(cu, strength=0.008, scale=0.06, seed=p.seed + 9)
        uv_box(cu)
        smooth(cu, 80)
        assign(cu, m_lb)
        realize(cu)
        uv_transform(cu, scale=(3.0, 3.0))               # leaf pattern at leaf size: reads as more foliage in the shade
        uv_keep(cu)
    pxy = np.array([(q["x"], q["y"]) for q in plants] or [(0.0, 0.0)])
    pgr = np.array([0.42 if q["spec"] == "small" else 0.88 for q in plants] or [0.88])

    def shade_g(co, n):                                  # deep shade under the leafy plants, the tight cushions catch the light
        d = (co[:, None, 0] - pxy[None, :, 0]) ** 2 + (co[:, None, 1] - pxy[None, :, 1]) ** 2
        return np.maximum(pgr[np.argmin(d, axis=1)], 0.9 * np.clip(1.0 - (co[:, 2] - zs - mound * 0.5) / 0.07, 0.0, 1.0))
    if cu:
        paint(cu, grime=shade_g)

    # ------------------------------------------------------------------ low plants
    SP_LEN = {"ovate": 0.095, "lobed": 0.06, "lance": 0.12, "small": 0.08}
    for q in plants:
        x, y, r, H, kind, col, spec, mi = (q[k] for k in ("x", "y", "r", "H", "kind", "col", "spec", "mi"))
        shapes, _, (n0, n1), els, _, shf, of = SPECIES[spec]
        z0 = soil_z(x, y)
        ps = r / 0.115
        lsc = ps
        a0 = rl.u(0, 2 * pi)
        a_out = None
        if edge_d(x, y) < 0.13:
            a_out = atan2(y, x) if round_ else (atan2(0.0, x) if xi - abs(x) < yi - abs(y) else atan2(y, 0.0))

        def surf(a, qq, dz=0.0, x=x, y=y, r=r, H=H, z0=z0):
            d = qq * r
            px, py = x + d * cos(a), y + d * sin(a)
            z = z0 + H * (1.0 - qq * qq) ** 0.55 + dz
            if edge_d(px, py) < 0.015:
                z = max(z, top_max + 0.01)
            return (px, py, z)

        # leaves spring from all over the dark core: upright in the heart of the plant, arching out and down towards the rim
        n = int(rl.u(n0, n1 + 0.99))
        for i in range(n):
            qq = min(0.98, sqrt((i + rl.u(0.2, 0.9)) / n))
            a = a0 + i * 2.39996 + rl.u(-0.25, 0.25)
            if a_out is not None and i % 5 in (1, 3):   # edging plants lean out over the kerb
                a = a_out + rl.u(-1.2, 1.2)
            s = lsc * rl.u(0.75, 1.32)
            d0 = of * qq * r
            ox, oy = x + d0 * cos(a), y + d0 * sin(a)
            rho = of * qq
            oz = z0 + shf * H * (1.0 - rho * rho) ** 0.55 - 0.01
            if els is None:                             # lies on the core, a little raised
                oz += 0.016
                el = degrees(atan2(-1.1 * rho * (1.0 - rho * rho) ** -0.45 * shf * H, r)) + rl.u(10, 42) + (30 if i < 2 else 0)
            else:
                el = lerp(els[0], els[1] - 45.0 * clamp((shf * H - 0.05) / 0.1), qq ** 1.3) + rl.u(-10, 10)   # tall plants: the flank is clothed too
            if edge_d(ox, oy) < 0.015:
                oz = max(oz, top_max + 0.01)
            Ls = SP_LEN[spec] * s
            reach = Ls * cos(radians(el))
            if edge_d(ox + reach * cos(a), oy + reach * sin(a)) < 0.012:      # the tip reaches over the kerb: keep it clear of the stone
                el = max(el, degrees(asin(clamp((top_max + 0.014 + 0.2 * Ls - oz) / Ls, -1.0, 0.95))))
            cand.append((rl.u() * (0.3 if i % 4 == 1 else 1.0), (*shapes[i % len(shapes)], mi),
                         trs((ox, oy, oz), (el, rl.u(-18, 18), az_fwd(cos(a), sin(a)) + rl.u(-24, 24)), (s * rl.u(0.85, 1.15), s, s))))
        if kind == 0:                                   # open five-petal flowers (primula / petunia), some half open, some side-on
            nf = int(rl.u(6.0, 9.99))
            for i in range(nf):
                a = a0 + 1.1 + i * 2.39996 + rl.u(-0.5, 0.5)
                side = rl.chance(0.14)
                qq = rl.u(0.8, 0.96) if side else sqrt(rl.u(0.0, 0.66))
                u_ = rl.u()
                shp = "half" if u_ < 0.16 else ("openB" if u_ < 0.7 else "open")
                tilt = (76 + rl.u(-8, 8) if side else 6 + 44 * qq) * (0.45 if shp == "half" else 1.0)
                mtx = trs(surf(a, qq, rl.u(0.02, 0.045) if shp == "half" else rl.u(0.012, 0.03)), (tilt, 0, az_up(cos(a), sin(a))), 1.0) \
                    @ trs((0, 0, 0), (rl.u(-25, 25), rl.u(-25, 25), rl.u(0, 360)), 0.58 + 0.56 * rl.u() ** 0.7)
                cand.append((rl.u() * (0.25 if i < 2 else 0.9), ("fl", shp, col), mtx))
        elif kind == 1:                                 # daisies on stalks above the leaves
            nf = int(rl.u(3.0, 5.99))
            for i in range(nf):
                a, qq = a0 + 1.1 + i * 2.39996 + rl.u(-0.5, 0.5), sqrt(rl.u(0.0, 0.7))
                mtx = trs(surf(a, qq, rl.u(0.02, 0.06)), (5 + 42 * qq, 0, az_up(cos(a), sin(a))), 1.0) \
                    @ trs((0, 0, 0), (rl.u(-22, 22), rl.u(-22, 22), rl.u(0, 360)), rl.u(0.78, 1.2))
                dcand.append((i + rl.u(), ("fl", "daisy", col), mtx))
        else:                                           # low cushion smothered in small flowers (alyssum / lobelia)
            nf = int(rl.u(13, 18))
            for i in range(nf):
                a, qq = rl.u(0, 2 * pi), sqrt(rl.u(0.0, 0.94))
                mtx = trs(surf(a, qq, rl.u(0.006, 0.02)), (70 * qq, 0, az_up(cos(a), sin(a))), 1.0) \
                    @ trs((0, 0, 0), (rl.u(-25, 25), rl.u(-25, 25), rl.u(0, 360)), rl.u(0.8, 1.35))
                cand.append((rl.u() * (0.25 if i < 6 else 0.9), ("fl", "tiny", col), mtx))

    # ------------------------------------------------------------------ tall spikes of bells (foxglove / campanula)
    rs = rng.fork("spikes")
    for x, y, cl in spike_pts:
        col = sp_cols[(cl + (1 if rs.chance(0.25) else 0)) % 2]
        z = soil_z(x, y)
        H = rs.u(0.5, 0.88) * (0.85 + 0.25 * min(1.0, W / 1.4))
        cx, cy = centres[cl]
        la = atan2(y - cy, x - cx) + rs.u(-0.7, 0.7)       # lean away from the middle of the group
        lean, bend = rs.u(0.03, 0.13), rs.u(-0.07, 0.07)

        def P(q, x=x, y=y, z=z, H=H, la=la, lean=lean, bend=bend):
            off, sb = lean * H * q ** 1.6, bend * H * sin(pi * q)
            return (x + cos(la) * off - sin(la) * sb, y + sin(la) * off + cos(la) * sb, z + H * q)
        if not far:
            stems.append(surface("stem", lambda u, v, P=P: tuple(Vector(P(u)) + 0.0042 * (1.0 - 0.6 * u) * Vector((cos(2 * pi * v), sin(2 * pi * v), 0.0))),
                                 7, 5, uv_len=(0.25, 0.25)))
        a0 = rs.u(0, 360)
        nb = int(rs.u(5, 7.99))
        for i in range(nb):                             # basal rosette
            M[("lf", "mid" if i % 2 else "midB", 1)].append(trs((x, y, z + 0.035), (rs.u(40, 66), rs.u(-15, 15), a0 + i * 360.0 / nb + rs.u(-20, 20)),
                                                                rs.u(1.0, 1.4)))
        t0 = rs.u(0.3, 0.38)
        far_sp.append((P, t0, col))
        a = rs.u(0, 360)
        n_sl = int(rs.u(4, 5.99))
        for i in range(n_sl):                           # stem leaves, smaller upwards, right up to the first bells
            a += 137.5 + rs.u(-20, 20)
            f = i / max(n_sl - 1, 1)
            M[("lf", "midB" if i % 2 else "mid", 1)].append(trs(P(lerp(0.07, t0 - 0.015, f)), (rs.u(32, 56), rs.u(-12, 12), a), lerp(0.85, 0.42, f) * rs.u(0.88, 1.12)))
        q = t0
        dens = rs.u(0.95, 1.15)
        while q < 0.992:
            f = (q - t0) / (1.0 - t0)
            size = lerp(1.0, 0.42, f ** 0.8)
            a += 137.5 + rs.u(-14, 14)
            ax, ay = cos(radians(a)), sin(radians(a))
            px, py, pz = P(q)
            rst = 0.0042 * (1.0 - 0.6 * q)
            loc = (px + ax * rst, py + ay * rst, pz)
            if f < 0.42:                                # open bells, nodding
                M[("fl", "bell", col)].append(trs(loc, (lerp(130, 104, f / 0.42) + rs.u(-7, 7), 0, az_up(ax, ay) + rs.u(-12, 12)), size * rs.u(0.9, 1.1)))
            elif f < 0.78:                              # younger, narrower, held more level
                M[("fl", "bells", col)].append(trs(loc, (lerp(102, 60, (f - 0.42) / 0.36) + rs.u(-7, 7), 0, az_up(ax, ay) + rs.u(-12, 12)),
                                                    size / 0.68 * rs.u(0.9, 1.1)))
            else:                                       # coloured buds, then tight green ones at the tip
                g = (f - 0.78) / 0.22
                M[("fl", "budc", col) if g < 0.55 else ("fl", "budg", 0)].append(
                    trs(loc, (lerp(52, 16, g) + rs.u(-8, 8), 0, az_up(ax, ay)), lerp(1.15, 0.6, g)))
            q += 0.0118 * size * dens * (1.0 if f < 0.78 else 1.2) / H

    # ------------------------------------------------------------------ foliage clumps: hosta, ornamental grass
    rf = rng.fork("foliage")
    for x, y in hostas:
        z = soil_z(x, y) + 0.01
        n = int(rf.u(10, 14.99))
        a0, s, mi = rf.u(0, 360), rf.u(0.8, 1.1), rf.pick((2, 2, 1))
        far_ho.append((x, y, s, mi))
        for i in range(n):
            q = i / n
            M[("lf", "hosta" if q < 0.62 else "hostaB", mi)].append(
                trs((x, y, z + 0.03 * q), (rf.u(16, 34) + 44 * q, rf.u(-10, 10), a0 + i * 137.5 + rf.u(-12, 12)), s * rf.u(0.78, 1.15) * (1.1 - 0.3 * q)))
    for x, y in grasses:
        z = soil_z(x, y)
        s, mi = rf.u(0.75, 1.1), rf.pick((2, 2, 0))
        far_gr.append((x, y, z, s, mi, []))
        for i in range(int(rf.u(30, 38))):
            a, rr = rf.u(0, 2 * pi), 0.035 * sqrt(rf.u())
            ax, ay = cos(a), sin(a)
            mtx = trs((x + ax * rr, y + ay * rr, z), (rf.u(54, 86), rf.u(-8, 8), az_fwd(ax, ay) + rf.u(-28, 28)), s * rf.u(0.55, 1.1))
            M[("lf", "blade" if i % 3 else "bladeB", mi)].append(mtx)
            far_gr[-1][5].append((mtx, i % 3 == 0))

    # ------------------------------------------------------------------ trailers: one or two dense swags that creep over the coping and hang on the stone
    dw = t / 2 + cop_over if dressed else t / 2 * 1.12 + 0.012
    top_def = h + 0.004 if dressed else hw - 0.15 * ch

    def spans(sd, s):
        if sd["kind"] == "arc":
            s = s % sd["len"]
            return (s, s - sd["len"], s + sd["len"])
        return (s,)

    def top_at(sd, s):
        if dressed:
            return top_def
        best = None
        for a, b, zt in sd["tops"]:
            if any(a - 0.02 <= q <= b + 0.02 for q in spans(sd, s)):
                best = zt + 0.008 if best is None else min(best, zt + 0.008)
        return top_def if best is None else best

    def face_at(sd, s, z):
        """Outward offset of the stone surface at (s, z)."""
        if dressed:
            return t / 2 + cop_over if (z > hw - 0.004 or z < pl) else t / 2 + 0.015
        best = None
        for a, b, za, zb, fc in sd["stones"]:
            if za - 0.015 <= z <= zb + 0.015 and any(a <= q <= b for q in spans(sd, s)):
                best = fc if best is None else max(best, fc)
        return t / 2 - 0.02 if best is None else best

    fallen_spots = []                                   # (x, y, z, colour, spread, count)
    for sd, s0 in swag_pl:
        arc = sd["kind"] == "arc"
        bx, by = fmap(sd, s0, -dw - 0.12)
        col = drift_at(bx, by)
        fx_, fy_ = fmap(sd, s0 + rt.u(-0.08, 0.08), dw + rt.u(0.04, 0.1))
        fallen_spots.append((fx_, fy_, 0.004, col, 0.06, int(rt.u(5, 8.99))))
        nst = int(clamp(round(3 + 2.5 * p.overgrowth + rt.u(-0.6, 0.6)), 3, 5))
        wsw = rt.u(0.045, 0.07)                         # half width of the swag where it crosses the coping
        lmax = lerp(0.18, 0.44, p.overgrowth) * rt.u(0.9, 1.1)
        skew = rt.u(-0.05, 0.05)
        far_sw.append([])
        for k in range(nst):
            off = clamp(-1.0 + 2.0 * (k + 0.5) / nst + rt.u(-0.12, 0.12), -1.0, 1.0)
            s1 = s0 + off * wsw
            s2 = s0 + off * wsw * 0.4 + skew                 # the strands gather towards the bottom
            zt = top_at(sd, s1)
            hang = max(0.05, min(lmax * (1.0 - 0.6 * abs(off) ** 1.4) * rt.u(0.88, 1.1), zt - 0.03))
            cps = [(s0 + off * 0.03, -dw - 0.09, zs + 0.03), (lerp(s0, s1, 0.5), -dw + 0.012, zt + 0.01), (s1, 0.0, zt + 0.005),
                   (s1, face_at(sd, s1, zt - 0.01) - 0.012, zt + 0.004)]
            nseg = max(3, int(hang / 0.035))
            for j in range(nseg + 1):
                f = j / nseg
                zz = zt - 0.012 - (hang - 0.012) * f
                ss = lerp(s1, s2, f ** 0.8) + 0.006 * sin(7.0 * f + k)
                cps.append((ss, face_at(sd, ss, zz) + 0.004, zz))
            if not arc:
                cps = [(clamp(s, 0.03, sd["len"] - 0.03), d, zz) for s, d, zz in cps]
            far_sw[-1].append((off, cps[0], cps[2], cps[3], cps[-1]))
            path = [Vector((*fmap(sd, s, d), zz)) for s, d, zz in cps]
            path = [Vector(q) for q in resample_path(path, step=0.012, smooth_path=False)]
            if not far:
                stems.append(sweep("trail", path[::4] + ([path[-1]] if (len(path) - 1) % 4 else []), 0.0018, sides=3, smooth_path=False,
                                   cap=False, scale_fn=lambda q: 1.0 - 0.5 * q))
            total = sum((path[i] - path[i - 1]).length for i in range(1, len(path)))
            dist, nxt, side = 0.0, 0.07 + rt.u(0, 0.03), 1
            for i in range(1, len(path)):
                seg = path[i] - path[i - 1]
                dist += seg.length
                if dist < nxt or seg.length < 1e-6:
                    continue
                f = dist / total
                pt = path[i]
                nrm = Vector((*fnorm(sd, cps[-1][0]), 0.0)) if not arc else Vector((pt.x, pt.y, 0.0)).normalized()
                hg = clamp((zt - 0.004 - pt.z) / 0.02)      # 0 on the coping, 1 hanging on the face
                zdir = Vector((0, 0, 1)).lerp(nrm + Vector((0, 0, 0.12)), hg).normalized()
                mi = 0 if (f > 0.86 or rt.chance(0.12)) else 1     # fresh lime growth at the tips
                for _ in range(2 if rt.chance(0.6) else 1):
                    ydir = Matrix.Rotation(radians(side * rt.u(30, 84)), 3, zdir) @ seg.normalized()
                    ydir = ydir + Vector((0, 0, -0.4 * hg)) + zdir * rt.u(0.0, 0.16)
                    s = lerp(1.75, 0.7, f ** 0.7) * rt.u(0.7, 1.25)
                    M[("lf", "heart", mi)].append(frame(pt + zdir * rt.u(0.002, 0.005), ydir, zdir, (s * rt.u(0.85, 1.1), s, s)))
                    side = -side
                side = side if rt.chance(0.8) else -side
                nxt = dist + rt.u(0.017, 0.03) * lerp(1.15, 0.8, f)

    # fallen petals: small dull clusters under the trailers and at the foot of the nearest drift
    rq = rng.fork("petals")
    for _ in range(int(perim * 0.45)):
        sd, s = rand_perim(rq)
        bx, by = fmap(sd, s, -dw - 0.15)
        x, y = fmap(sd, s, dw + rq.u(0.03, 0.09))
        fallen_spots.append((x, y, 0.004, drift_at(bx, by), 0.05, int(rq.u(3, 6.99))))
    if dressed:
        for _ in range(max(1, int(perim * 0.3))):
            sd, s = rand_perim(rq)
            bx, by = fmap(sd, s, -dw - 0.15)
            x, y = fmap(sd, s, rq.u(-0.07, 0.0))
            fallen_spots.append((x, y, h + 0.005, drift_at(bx, by), 0.03, int(rq.u(2, 4.99))))
    for x, y, zf, col, spread, cnt in fallen_spots:
        for _ in range(cnt):
            a, rr = rq.u(0, 2 * pi), spread * sqrt(rq.u())
            M[("fl", "fallen", col)].append(trs((x + rr * cos(a), y + rr * sin(a), zf), (rq.u(-10, 10), rq.u(-10, 10), rq.u(0, 360)), rq.u(0.7, 1.2)))

    # ------------------------------------------------------------------ moss tufts: foot of the kerb, joints, plinth ledge, shaded side
    tri_dbg = defaultdict(int)
    rm = rng.fork("moss")
    mm = []
    d_foot = t / 2 + (cop_over if pl else 0.0)
    for _ in range(int(perim * 4.0 * p.moss)):
        sd, s = rand_perim(rm, 0.05)
        nx, ny = fnorm(sd, s)
        if not rm.chance(0.35 + 0.65 * max(0.0, ny)):
            continue
        x, y = fmap(sd, s, (d_foot if dressed else face_at(sd, s, 0.03)) + rm.u(-0.004, 0.012))
        r = rm.u(0.012, 0.03)
        mm.append(trs((x, y, 0.0), (0, 0, rm.u(0, 360)), (r * rm.u(0.9, 1.9), r, r * rm.u(0.45, 0.75))))
    for sd in sides:
        for s, z0, z1 in sd["joints"]:
            if not rm.chance(min(1.0, 0.25 + 0.7 * p.moss)):
                continue
            nx, ny = fnorm(sd, s)
            nrm = Vector((nx, ny, 0.0))
            r = rm.u(0.007, 0.013)
            if z0 < 0:                                  # joint of the plinth course: tuft on the ledge
                x, y = fmap(sd, s + rm.u(-0.02, 0.02), t / 2 + 0.01)
                mm.append(trs((x, y, pl - 0.004), (0, 0, rm.u(0, 360)), (r * 2.2, r * 1.3, r * 0.8)))
            elif dressed:                               # creeping up a vertical joint
                x, y = fmap(sd, s - 0.0045, t / 2 - 0.006)
                mm.append(frame((x, y, z0 + rm.u(0.0, 0.5) * (z1 - z0)), (0, 0, 1), nrm, (r * 0.75, r * rm.u(1.6, 3.2), r * 0.9)))
            else:                                       # between two field stones
                x, y = fmap(sd, s - 0.004, rm.u(-0.05, 0.05))
                r *= 2.0
                mm.append(trs((x, y, z1 - rm.u(0.035, 0.06)), (0, 0, rm.u(0, 360)), (r * rm.u(1.0, 1.6), r, r * 0.8)))
    mo = None
    if mm:
        for mtx in list(mm):                            # a tuft is two or three unequal cushions grown together
            for _ in range(int(rm.u(1, 2.99))):
                a, k = rm.u(0, 2 * pi), rm.u(0.45, 0.8)
                mm.append(mtx @ trs((0.95 * cos(a), 0.95 * sin(a), 0.0), (0, 0, rm.u(0, 360)), (k, k * rm.u(0.7, 1.0), k * rm.u(0.8, 1.2))))
        if not far:
            dome = lathe("moss_dome", [(1.0, 0.0), (0.8, 0.55), (0.0, 1.0)], segments=4)
            mo = instances_to_mesh("moss", dome, mm, keep_base=False)
            assign(mo, m_lb)
            uv_box(mo)
            paint(mo, grime=0.45)
            tri_dbg["moss"] = tris(mo)

    # ------------------------------------------------------------------ base meshes + stamping
    LEAF = {    # length, width, quads across, quads along, shape
        "ov": (0.095, 0.06, 2, 3, dict(vm=0.45, base_w=0.3, tip_w=0.16, tip_bow=0.1, lift=0.55, droop=0.5, cup=0.3, wave=0.08)),
        "ovB": (0.08, 0.057, 2, 3, dict(vm=0.5, base_w=0.35, tip_w=0.3, tip_bow=0.14, lift=0.35, droop=0.65, cup=-0.18, twist=30, wave=0.15)),
        "lance": (0.12, 0.034, 2, 2, dict(vm=0.5, base_w=0.4, tip_w=0.14, tip_bow=0.05, lift=0.6, droop=0.55, cup=0.4)),
        "lanceB": (0.1, 0.035, 2, 2, dict(vm=0.45, base_w=0.4, tip_w=0.18, tip_bow=0.06, lift=0.5, droop=0.8, cup=0.2, twist=55, wave=0.2)),
        "heart": (0.036, 0.034, 2, 2, dict(vm=0.4, base_w=0.55, tip_w=0.08, tip_bow=0.1, base_bow=-0.28, lift=0.12, droop=0.22, cup=0.22)),
        "mid": (0.19, 0.075, 2, 3, dict(vm=0.4, base_w=0.25, tip_w=0.12, tip_bow=0.04, lift=0.6, droop=0.8, cup=0.3, wave=0.12)),
        "midB": (0.16, 0.07, 2, 3, dict(vm=0.45, base_w=0.3, tip_w=0.16, tip_bow=0.06, lift=0.75, droop=0.6, cup=0.45, twist=28, wave=0.08)),
        "hosta": (0.27, 0.17, 4, 4, dict(vm=0.38, base_w=0.45, tip_w=0.06, tip_bow=0.0, base_bow=-0.12, lift=0.75, droop=1.0, cup=0.35, wave=0.1, wave_f=14.0)),
        "hostaB": (0.2, 0.11, 2, 4, dict(vm=0.4, base_w=0.4, tip_w=0.08, tip_bow=0.0, base_bow=-0.08, lift=0.9, droop=0.5, cup=0.65, twist=20)),
        "blade": (0.52, 0.016, 1, 5, dict(vm=0.2, base_w=0.8, tip_w=0.05, tip_bow=0.0, lift=0.25, droop=0.55, cup=0.6, twist=25, stalk=0.3)),
        "bladeB": (0.4, 0.014, 1, 4, dict(vm=0.2, base_w=0.8, tip_w=0.05, tip_bow=0.0, lift=0.1, droop=0.15, cup=0.5, twist=70, stalk=0.3)),
    }

    def make_base(key):
        fam, shp, i = key
        if fam == "lf":
            Ll, Wl, nu, nv, kw = LEAF[shp]
            return leaf_base("leaf_" + shp, Ll, Wl, nu, nv, m_leaf[i], **kw)
        if fam == "lobe":
            if shp == "a":
                return lobed_leaf("leaf_lobed", 0.046, 5, m_leaf[i], per=3, sinus=0.3, notch=0.72, cup=0.3)
            return lobed_leaf("leaf_lobedB", 0.038, 7, m_leaf[i], per=2, sinus=0.4, notch=0.8, cup=0.48, ripple=0.1)
        if fam == "sprig":
            if shp == "a":
                return sprig_base("sprig", m_leaf[i], Ls=0.08, pairs=2, Ll=0.036, Wl=0.027)
            return sprig_base("sprigB", m_leaf[i], Ls=0.066, pairs=2, Ll=0.04, Wl=0.031)
        if shp == "budg":
            o = lathe("bud", [(0.0, 0.0), (0.003, 0.005), (0.0, 0.0115)], segments=3)
            assign(o, m_la)
            return o
        m = m_pet[i]
        if shp in ("open", "openB", "half"):
            if shp == "open":
                fl = corolla("corolla", 0.022, m, elev=22, bend=40, heart=True)
            elif shp == "openB":
                fl = corolla("corolla", 0.02, m, elev=34, bend=46, width=1.12, cup=0.42, twist=16, heart=False, light=True)
            else:
                fl = corolla("corolla", 0.019, m, elev=76, bend=34, width=0.9, cup=0.5, twist=20, heart=False, eye=0.8, eye_dark=0.5, light=True)
                sk = cylinder("stalk", 0.0015, 0.05, loc=(0, 0, -0.025), verts=3, cap=False)
                assign(sk, m_la)
                return join([fl, sk], "fl_" + shp)
            ce = lathe("eye", [(0.0036, 0.0035), (0.0, 0.008)], segments=4)
            assign(ce, m_ctr)
            sk = mesh_object("stalk", [(-0.0014, 0, 0), (0.0014, 0, 0), (0.001, 0.004, -0.05), (-0.001, 0.004, -0.05)], [(0, 1, 2, 3)])
            uv_box(sk)
            assign(sk, m_la)
            return join([fl, ce, sk], "fl_" + shp)
        if shp == "daisy":
            return daisy_base(m, m_ctr, m_la)
        if shp == "tiny":
            return lobed_disc("fl_tiny", 0.0135, 4, m, per=3, notch=0.4, cup=0.3, eye=0.45, eye_dark=0.5)
        if shp == "bell":
            return bell("fl_bell", 0.014, 0.036, m, sides=4, lobed=True)
        if shp == "bells":
            return bell("fl_bells", 0.0095, 0.026, m, sides=4, lobed=False)
        if shp == "budc":
            o = lathe("budc", [(0.0, 0.0), (0.0036, 0.006), (0.0, 0.014)], segments=3)
            assign(o, m)
            paint(o, grime=0.2)
            return o
        # fallen petal: small cupped oval, dulled
        o = surface("petal", lambda u, v: ((u - 0.5) * 0.013 * (0.5 + 0.5 * sin(pi * (0.15 + 0.7 * v))), (v - 0.5) * 0.018 - 0.004 * (2 * u - 1) ** 2 * (2 * v - 1),
                                           0.0025 * (2 * u - 1) ** 2), 2, 2, uv_len=(0.25, 0.25))
        assign(o, m)
        smooth(o, 85)
        paint(o, grime=0.55)
        return o

    n_inst = 0
    st = None
    if stems:
        st = join(stems, "stems")
        assign(st, m_lb)
        tri_dbg["stems"] = tris(st)
    bases = {}

    def petal_area(o):
        libs = [str(s.material.get("tsq_lib", "")) if s.material else "" for s in o.material_slots]
        return float(sum(f.area for f in o.data.polygons if libs[f.material_index] == "wax"))

    def base(key):
        if key not in bases:
            o = None if far else make_base(key)
            bases[key] = (o, lod_memo(("base_tris",) + tuple(key), lambda: tris(o)),     # (the far level cuts as lod 0 did
                          lod_memo(("base_petal",) + tuple(key), lambda: petal_area(o)))  # and sizes its caps by the petals)
        return bases[key]
    # spend what is left of the triangle budget on the low plants (every part has a priority; the cut thins all plants evenly)
    budget = TRI_TARGET - (lod_memo("fixed_tris", lambda: stone_tris + tris(cu) + tris(soil) + tri_dbg["stems"] + tri_dbg["moss"])
                           + sum(len(v) * base(k)[1] for k, v in M.items()))
    dcand.sort(key=lambda c_: c_[0])
    if dcand:
        dt = base(dcand[0][1])[1]
        for pr, key, mtx in dcand[:int(min(2500.0, max(0.0, budget) * 0.3) // dt)]:
            M[key].append(mtx)
            budget -= base(key)[1]
    cand.sort(key=lambda c_: c_[0])
    kept = 0
    for pr, key, mtx in cand:
        tr = base(key)[1]
        if budget < tr:
            break
        budget -= tr
        M[key].append(mtx)
        kept += 1
    made = [wall, cope, soil, cu, st, mo]
    for key, ms_ in ([] if far else M.items()):
        o = instances_to_mesh("pl_" + "_".join(map(str, key)), base(key)[0], ms_, uv_jitter=False, keep_base=False)
        n_inst += len(ms_)
        tri_dbg[key[1] if key[0] == "fl" else f"{key[0]}{key[2]}"] += tris(o)
        if key[0] != "fl":                              # tonal variation in the foliage
            paint(o, grime=mask_mul(noise_mask(0.22, seed=ms + 11, lo=0.25, hi=0.85), 0.34))
        made.append(o)
    ext0 = lod_memo("extent", lambda: _extent(made))
    if far:
        # ------------------------------------------------------------------ far level
        V = defaultdict(lambda: ([], []))               # (family, material index) -> verts, faces

        def add(k, vs, fs):
            v, f = V[k]
            n = len(v)
            v.extend(tuple(float(c) for c in q) for q in vs)
            f.extend(tuple(i + n for i in q) for q in fs)

        def soil_v(X, Y):
            if round_:
                f = np.clip(1.0 - (X * X + Y * Y) / (Ri * Ri), 0.0, None)
            else:
                f = np.clip(1.0 - (X / xi) ** 2, 0.0, None) * np.clip(1.0 - (Y / yi) ** 2, 0.0, None)
            return zs + mound * f ** 0.75

        # foliage masses: low plants at their full height, hostas, the rosettes of the spikes, the foot of the grasses
        fo = np.array([(q["x"], q["y"], 1.05 * q["r"], q["H"], q["mi"]) for q in plants]
                      + [(x, y, 0.23 * s, 0.14 * s, mi) for x, y, s, mi in far_ho]
                      + [(x, y, 0.11, 0.1, 1) for x, y, _ in spike_pts]
                      + [(x, y, 0.07, 0.06, mi) for x, y, _, _, mi, _ in far_gr] or [(0.0, 0.0, 1e-3, 0.0, 1)])

        def fcan(X, Y):
            d2 = ((X[..., None] - fo[:, 0]) ** 2 + (Y[..., None] - fo[:, 1]) ** 2) / fo[:, 2] ** 2
            return np.max(fo[:, 3] * np.clip(1.0 - d2, 0.0, None) ** 0.55, axis=-1)

        # one blanket over the bed (a square grid, mapped onto the disc of a round bed), its triangles in the green
        # of the plant below; its outer ring of cells is the foliage that leans out over the kerb
        fcell, f_in, f_out = 0.15 * big, 0.06, FAR_RIM

        def lines(half, n):
            return np.r_[-(half + f_out), np.linspace(-(half - f_in), half - f_in, n + 1), half + f_out]
        if round_:
            UA = VB = lines(Ri, max(3, int(round(2 * (Ri - f_in) / fcell)))) / (Ri + f_out)
            A, B = np.meshgrid(UA, VB, indexing="ij")
            wide = np.abs(A) > np.abs(B)
            rr = np.where(wide, A, B)
            ph = np.where(wide, (pi / 4) * B / np.where(A == 0.0, 1.0, A), pi / 2 - (pi / 4) * A / np.where(B == 0.0, 1.0, B))
            GX, GY = (Ri + f_out) * rr * np.cos(ph), (Ri + f_out) * rr * np.sin(ph)
        else:
            UA, VB = lines(xi, max(2, int(round(2 * (xi - f_in) / fcell)))), lines(yi, max(1, int(round(2 * (yi - f_in) / fcell))))
            GX, GY = np.meshgrid(UA, VB, indexing="ij")
        nx, ny = len(UA) - 1, len(VB) - 1
        o3 = 0.33 * fcell
        GH = np.max([fcan(GX + dx * o3, GY + dy * o3) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], axis=0)
        GH = np.where(GH > 0.006, GH + FAR_LIFT, 0.0)
        rim = np.zeros(GX.shape, dtype=bool)
        rim[0, :] = rim[-1, :] = rim[:, 0] = rim[:, -1] = True
        ii, jj = np.meshgrid(np.clip(np.arange(nx + 1), 1, nx - 1), np.clip(np.arange(ny + 1), 1, ny - 1), indexing="ij")
        GH = np.where(rim, GH[ii, jj], GH)              # a rim node is foliage where the node inside it is
        GZ = soil_v(GX, GY) + np.where(GH > 0.006, GH, -0.02)
        GZ = np.where(~rim & (GH > 0.006) & (np.abs(ii - np.arange(nx + 1)[:, None]) + np.abs(jj - np.arange(ny + 1)[None, :]) == 0)
                      & ((ii == 1) | (ii == nx - 1) | (jj == 1) | (jj == ny - 1)), np.maximum(GZ, top_max + FAR_EDGE), GZ)
        # (the rim goes down into the kerb: the flank of the planting stands on the stone, no soil shows below it)
        kerb_lo = h if dressed else min([zt for sd in sides for _, _, zt in sd["tops"]] + [mx_[2] for _, mx_ in corner]) - 0.015

        def kerb_z(x, y):
            """Top of the kerb next to (x, y): of the stones there the lowest."""
            if dressed:
                return h
            if round_:
                sd = sides[0]
                s = (atan2(y, x) % (2 * pi)) * sd["R"]
            else:
                sd = max(sides, key=lambda q: q["n"][0] * (x - q["P0"][0]) + q["n"][1] * (y - q["P0"][1]))
                s = sd["dir"][0] * (x - sd["P0"][0]) + sd["dir"][1] * (y - sd["P0"][1])
                if not 0.0 <= s <= sd["len"]:
                    return kerb_lo                      # (a corner boulder)
            wrap = sd["len"] if round_ else 1e9
            near = [zt for a, b, zt in sd["tops"] if abs((s - 0.5 * (a + b) + 0.5 * wrap) % wrap - 0.5 * wrap) <= 0.5 * (b - a) + 0.15]
            return min(near) - 0.012 if near else kerb_lo
        GZ = np.where(rim, np.vectorize(kerb_z)(GX, GY) - 0.02, GZ)
        diag = GZ[:-1, :-1] + GZ[1:, 1:] >= GZ[1:, :-1] + GZ[:-1, 1:]       # split every quad along its ridge

        def bl_at(x, y):
            """Height of the blanket at (x, y)."""
            if round_:
                r, a = hypot(x, y) / (Ri + f_out), atan2(y, x)
                if abs(a) <= pi / 4:
                    u, v = r, r * a / (pi / 4)
                elif abs(a) >= 3 * pi / 4:
                    u, v = -r, -r * (a - pi if a > 0 else a + pi) / (pi / 4)
                elif a > 0:
                    u, v = r * (pi / 2 - a) / (pi / 4), r
                else:
                    u, v = -r * (-pi / 2 - a) / (pi / 4), -r
            else:
                u, v = x, y
            i, j = int(clamp(np.searchsorted(UA, u, "right") - 1, 0, nx - 1)), int(clamp(np.searchsorted(VB, v, "right") - 1, 0, ny - 1))
            u, v = clamp((u - UA[i]) / (UA[i + 1] - UA[i])), clamp((v - VB[j]) / (VB[j + 1] - VB[j]))
            z00, z10, z11, z01 = GZ[i, j], GZ[i + 1, j], GZ[i + 1, j + 1], GZ[i, j + 1]
            if diag[i, j]:
                return float(z00 + u * (z10 - z00) + v * (z11 - z10) if u >= v else z00 + u * (z11 - z01) + v * (z01 - z00))
            return float(z00 + u * (z10 - z00) + v * (z01 - z00) if u + v <= 1.0 else z11 + (1 - u) * (z01 - z11) + (1 - v) * (z10 - z11))

        node = {}
        for i in range(nx):
            for j in range(ny):
                q = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                for tr_ in ((q[0], q[1], q[2]), (q[0], q[2], q[3])) if diag[i, j] else ((q[0], q[1], q[3]), (q[1], q[2], q[3])):
                    if max(GH[a] for a in tr_) <= 0.006:
                        continue                        # bare soil
                    cx_, cy_ = sum(GX[a] for a in tr_) / 3.0, sum(GY[a] for a in tr_) / 3.0
                    mi = int(fo[np.argmin(((cx_ - fo[:, 0]) ** 2 + (cy_ - fo[:, 1]) ** 2) / fo[:, 2] ** 2), 4])
                    if sum(bool(rim[a]) for a in tr_) >= 2:
                        mi = 1                          # (the foot of the flank: the deep green of the under-canopy)
                    v, f = V[("lf", mi)]
                    ids = []
                    for a in tr_:
                        if (mi, a) not in node:
                            node[(mi, a)] = len(v)
                            v.append((float(GX[a]), float(GY[a]), float(GZ[a])))
                        ids.append(node[(mi, a)])
                    f.append(tuple(ids))

        # flowers: one low cap per plant and colour with the area of the petals of the flowers that plant kept
        caps = defaultdict(lambda: np.zeros(3))
        for key, ms_ in M.items():
            if key[0] == "fl" and key[1] in ("open", "openB", "half", "daisy", "tiny"):
                for mtx in ms_:
                    x, y = mtx[0][3], mtx[1][3]
                    a = FAR_FLOWER[key[1] == "tiny"] * base(key)[2] * mtx.to_scale()[0] ** 2
                    caps[(int(np.argmin((pxy[:, 0] - x) ** 2 + (pxy[:, 1] - y) ** 2)), key[2])] += (a, a * x, a * y)
        n_cap = 0
        for (pi_, col), (a, ax, ay) in sorted(caps.items()):
            q = plants[pi_]
            cx_, cy_ = ax / a, ay / a
            rho = min(sqrt(0.5 * a), 0.95 * q["r"])
            z0 = soil_z(q["x"], q["y"])

            def top(x, y, q=q, z0=z0):
                d2 = ((x - q["x"]) ** 2 + (y - q["y"]) ** 2) / (q["r"] * q["r"])
                z = max(bl_at(x, y), z0 + q["H"] * max(0.0, 1.0 - d2) ** 0.55) + 0.015
                return (x, y, max(z, top_max + 0.012) if edge_d(x, y) < 0.015 else z)
            a0 = 2.39996 * pi_
            ring = [top(cx_ + rho * cos(a0 + 0.5 * pi * k), cy_ + rho * sin(a0 + 0.5 * pi * k)) for k in range(4)]
            apex = top(cx_, cy_)
            apex = (cx_, cy_, max(apex[2] + 0.015, max(v[2] for v in ring) + 0.006))
            # (the narrow face below the cap is its stalks: it ties the cap to the soil through the blanket)
            zr = min(v[2] for v in ring)
            add(("fl", col), ring + [apex, (cx_ - 0.012, cy_, zr), (cx_ + 0.012, cy_, zr), (cx_, cy_, soil_z(cx_, cy_) - 0.05)],
                [(k, (k + 1) % 4, 4) for k in range(4)] + [(5, 6, 7)])
            n_cap += 1

        # spikes: a green shaft with its stem leaves, above it a spire in the colour of the bells
        for k, (Pf, t0, col) in enumerate(far_sp):
            def ring(q, rad, a0=2.4 * k, Pf=Pf):
                c = Pf(q)
                return [(c[0] + rad * cos(a0 + 2 * pi * i / 3), c[1] + rad * sin(a0 + 2 * pi * i / 3), c[2]) for i in range(3)]
            side = [(i, (i + 1) % 3, 3 + (i + 1) % 3, 3 + i) for i in range(3)]
            add(("lf", 1), ring(0.06, 0.08) + ring(t0, 0.03), side)
            c0 = Pf(0.0)
            add(("fl", col), ring(t0, FAR_SPIRE[0]) + ring(0.5 * (t0 + 1.0), FAR_SPIRE[1]) + [Pf(1.0), (c0[0], c0[1], c0[2] - 0.05)],
                side + [(3 + i, 3 + (i + 1) % 3, 6) for i in range(3)] + [(0, 1, 7)])

        # grass tufts: a closed three-sided core (the dense foot) and seven blades: of the blades of the tuft the longest
        # in each seventh of the round, as one triangle from the foot to its tip
        for k, (x, y, z, s, mi, blades) in enumerate(far_gr):
            rc, hc, wb = FAR_GRASS[0] * s, FAR_GRASS[1] * s, 0.5 * FAR_GRASS[2] * s
            vs = [(x + r_ * cos(1.3 * k + 2 * pi * i / 3), y + r_ * sin(1.3 * k + 2 * pi * i / 3), z + zz)
                  for r_, zz in ((0.035, 0.0), (rc, hc)) for i in range(3)]
            fs = [(i, (i + 1) % 3, 3 + (i + 1) % 3, 3 + i) for i in range(3)] + [(3, 4, 5)]
            foot, tip = Vector((x, y, z)), {}
            for m_, short in blades:                    # (tip of the base blade: length, lift less droop)
                q = m_ @ Vector((0.0, 0.4, -0.02) if short else (0.0, 0.52, -0.156))
                sec = int((atan2(q.y - y, q.x - x) % (2 * pi)) / (2 * pi) * 7) % 7
                if sec not in tip or (q - foot).length > (tip[sec] - foot).length:
                    tip[sec] = q
            for _, q in sorted(tip.items()):
                a = atan2(q.y - y, q.x - x)
                n = len(vs)
                vs += [(x + wb * sin(a), y - wb * cos(a), z + 0.04 * s), (x - wb * sin(a), y + wb * cos(a), z + 0.04 * s), tuple(q)]
                fs.append((n + 1, n, n + 2))
            add(("lf", mi), vs, fs)

        # swags: one ribbon from the bed over the coping and down the face, ending in a point at the longest strand
        for (sd, _), sw in zip(swag_pl, far_sw):
            sw = sorted(sw, key=lambda e: e[0])
            lo_, hi_ = sw[0], sw[-1]
            mid = min(sw, key=lambda e: e[4][2])

            def at(c, ds, dd, dz, sd=sd):
                return (*fmap(sd, c[0] + ds, c[1] + dd), c[2] + dz)
            pad = 0.022
            vs = []
            for e, sg in ((lo_, -1.0), (hi_, 1.0)):
                vs += [at(e[1], sg * pad, 0.0, 0.0), at(e[2], sg * pad, 0.0, 0.016), at(e[3], sg * pad, 0.022, 0.016),
                       at(e[4], sg * 0.5 * pad, 0.014, 0.0)]
            fs = [(j, j + 1, j + 5, j + 4) for j in range(3)]
            if mid[4][2] < min(lo_[4][2], hi_[4][2]) - 0.02:
                vs.append(at(mid[4], 0.0, 0.014, 0.0))
                fs.append((3, 8, 7))
            add(("lf", 1), vs, fs)

        # fallen petals: one triangle per cluster (the largest its petals span), drawn in a little
        cur = defaultdict(int)
        fallen = []
        real = []
        for x, y, zf, col, spread, cnt in fallen_spots:
            ms_ = M[("fl", "fallen", col)][cur[col]:cur[col] + cnt]
            cur[col] += cnt
            pt = np.array([(m_[0][3], m_[1][3]) for m_ in ms_])
            real.append(pt)
            if len(pt) < 3:
                pt = np.vstack([pt, pt[:1] + ((0.012, 0.0), (0.0, 0.012))])
            near = int(np.argmax([edge_d(*q) for q in pt])) if zf < 0.01 else 0        # on the paving: the petal next to the kerb stays
            best = max(((near, j, k) for j in range(len(pt)) for k in range(j) if near not in (j, k)),
                       key=lambda t_: abs(float(np.cross(pt[t_[1]] - pt[t_[0]], pt[t_[2]] - pt[t_[0]]))))
            tri_ = pt[list(best)]
            tri_[1:] = tri_[0] + FAR_PETAL * (tri_[1:] - tri_[0])
            d = -edge_d(*tri_[0]) - 0.5 * t
            if zf < 0.01 and d > 0.0:                   # on the paving: that corner goes under the kerb (the petals lie against it)
                x, y = tri_[0]
                if round_:
                    tri_[0] = (x * (1.0 - d / hypot(x, y)), y * (1.0 - d / hypot(x, y)))
                elif xi - abs(x) < yi - abs(y):
                    tri_[0] = (x - (d if x > 0 else -d), y)
                else:
                    tri_[0] = (x, y - (d if y > 0 else -d))
            if float(np.cross(tri_[1] - tri_[0], tri_[2] - tri_[0])) < 0.0:
                tri_ = tri_[[0, 2, 1]]
            fallen.append([col, zf + 0.004, tri_.copy()])
        # petals and moss tufts at the foot of the kerb lie furthest out: where the far level ends short of lod 0, the
        # tuft that reaches there comes back as a pyramid, or a corner of a petal triangle goes out to the last petal
        allv = np.vstack([_co(o) for o in (wall, cope, soil) if o is not None] + [np.array(v) for v, _ in V.values() if v])
        mn_, mx_ = allv.min(axis=0), allv.max(axis=0)
        rp_ = np.vstack(real) if real else np.zeros((0, 2))
        MC = np.array([[tuple(m_ @ Vector(c)) for c in ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1))] for m_ in mm]).reshape(-1, 5, 3)
        for ax_ in (0, 1):
            for sg in (-1.0, 1.0):
                reach = max([sg * (mx_ if sg > 0 else mn_)[ax_]] + [float((sg * f[2][:, ax_]).max()) for f in fallen])
                want = sg * (ext0[1] if sg > 0 else ext0[0])[ax_]
                if reach >= want - 0.006:
                    continue
                r_m = float((sg * MC[:, :4, ax_]).max()) if len(MC) else -1e9
                r_p = float((sg * rp_[:, ax_]).max()) + 0.008 if len(rp_) else -1e9
                if r_m > max(r_p, reach):
                    add(("lf", 1), MC[int(np.argmax((sg * MC[:, :4, ax_]).max(axis=1)))], [(k, (k + 1) % 4, 4) for k in range(4)])
                elif r_p > reach:
                    far_pt = rp_[np.argmax(sg * rp_[:, ax_])]
                    f = fallen[int(np.argmin([np.min(((f[2] - far_pt) ** 2).sum(axis=1)) for f in fallen]))]
                    f[2][int(np.argmax(sg * f[2][:, ax_]))] = far_pt + sg * 0.008 * np.eye(2)[ax_]
        for col, z, tri_ in fallen:
            add(("fl", col), [(x, y, z) for x, y in tri_], [(0, 1, 2)])

        n_far = 0
        for (fam, i), (vs, fs) in sorted(V.items()):
            if not fs:
                continue
            o = mesh_object(f"far_{fam}{i}", vs, fs)
            assign(o, m_leaf[i] if fam == "lf" else m_pet[i])
            uv_box(o)
            if fam == "lf":
                smooth(o, 60)
            n_far += tris(o)
            made.append(o)
        e1 = _extent(made)
        print(f"DBG far: caps={n_cap} blanket={nx}x{ny} planting_tris={n_far} kerb_tris={tris(wall) + (tris(cope) if cope else 0)} "
              f"soil_tris={tris(soil)} d_min={[round(a - b, 3) for a, b in zip(e1[0], ext0[0])]} "
              f"d_max={[round(a - b, 3) for a, b in zip(e1[1], ext0[1])]}")

    socket("lantern", "attach", (0.0, 0.0, soil_z(0.0, 0.0)), role="lantern")
