"""Clipped privet / box hedge run along X. A row of irregularly spaced multi-stemmed shrubs (forking grey
stems leaning out of a planting strip) carries a hand-clipped foliage mass built in three depth layers:
a dark recessed inner shell, a mid layer of tilted leaf-mass flakes and an outer coat of modelled leaf
rosettes with gaps that look into the dark interior, plus escaped shoots on top and drooping sprigs
along the ragged, sagging underside. Each shrub bulges a little so the faces have large-scale form.
Flat, rounded or castellated (flat-topped topiary battlement) top; butt ends are leafy but stay inside
x = +-length/2 so runs join, or rounded free-standing ends; optional dressed-stone kerb or dog-tooth
brick edging around the bed. Leaf density is constant per square metre of face for any length.

Far level (lod() == 2, seen from 20 m and more): same plan and the same random draws. The leaf coat is
one closed coarse skin a little proud of the clip surface (cross-section through the corners of the
clipped profile, stations every ~0.3 m and at every castellation step, ends capped) in the leaf-mass
colours, dark underneath and where the old dark leaves gather;
broad kites lie on it like scales in the leaf colours of the spot (stand-ins for the leaf tufts, browned in
the die-back patches) and a share of the escaped shoots and drooping sprigs stays as single kites so the
top line and the lower edge are not ruled straight. The two thickest stems of each shrub are straight three-sided tapers, the
bed is a coarse closed strip and each kerb run one plain bar. Left out: stem forks and twigs, fallen
leaves, kerb joints.
"""
import math
import numpy as np
from mathutils import Vector, noise as mnoise
from tk import *

# tri_budget: leaf density is constant per m2 of hedge face for every length (a 10 m run must not look
# thinner than a 3 m one): ~9k tris per metre of default section -> the 3 m default is ~27k, 10 m ~85k.
META = {"name": "Clipped hedge run", "class": "vegetation", "mount": "ground", "focus": 0.55, "tri_budget": 100000, "lod": "far"}
PARAMS = {
    "length": {"default": 3.0, "min": 1.0, "max": 10.0, "desc": "run length along X (m); spans exactly +-length/2"},
    "height": {"default": 1.1, "min": 0.5, "max": 2.2, "desc": "clipped height (m)"},
    "width": {"default": 0.6, "min": 0.35, "max": 1.2, "desc": "hedge thickness at the base (m)"},
    "top": {"default": "flat", "choices": ["flat", "rounded", "castellated"], "desc": "clipped top profile"},
    "kerb": {"default": "stone", "choices": ["none", "stone", "brick"], "desc": "edging of the planting strip"},
    "ends": {"default": "butt", "choices": ["butt", "rounded"], "desc": "butt = square ends that join other runs; rounded = free-standing"},
    "skirt": {"default": 0.24, "min": 0.08, "max": 0.4, "desc": "height of the bare-stem gap under the foliage (m)"},
    "shag": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "0 = freshly clipped, 1 = overdue for a trim (belly, lumps, long escaped shoots)"},
    "tint": {"default": "#ffffff", "desc": "foliage tint (multiplies the leaf green: #ffffff privet, #b9d6b4 dark box, #f2efb0 golden)"},
    "leaf_size": {"default": 1.0, "min": 0.55, "max": 1.5, "desc": "leaf scale: 0.6 small-leaved box, 1 privet (3-5 cm), 1.4 laurel"},
    "mulch": {"default": "gravel", "choices": ["gravel", "soil"], "desc": "surface of the planting strip"},
    "dieback": {"default": 0.25, "min": 0.0, "max": 1.0, "desc": "number of distinct browned die-back patches (0 = none, 1 = about one per metre)"},
    "litter": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of fallen leaves on the bed"},
}
VARIANTS = {
    "box_edging_low": {"length": 4.0, "height": 0.55, "width": 0.45, "top": "rounded", "kerb": "brick", "leaf_size": 0.65, "dieback": 0.0, "skirt": 0.1,
                       "shag": 0.08, "tint": "#9ccfa8", "seed": 4},
    "castellated": {"length": 4.0, "height": 1.5, "width": 0.8, "top": "castellated", "kerb": "stone", "shag": 0.15, "dieback": 0.12, "seed": 11},
    "tall_overgrown": {"length": 3.0, "height": 2.0, "width": 0.95, "top": "flat", "kerb": "none", "ends": "rounded",
                       "skirt": 0.1, "shag": 0.95, "dieback": 0.85, "leaf_size": 1.2, "tint": "#e6efc0", "litter": 1.0, "seed": 23},
    "dome_stub": {"length": 1.4, "height": 1.0, "width": 0.9, "top": "rounded", "kerb": "stone", "ends": "rounded",
                  "shag": 0.3, "tint": "#d8f0c8", "leaf_size": 0.8, "seed": 31},
}

TRI_CAP = 94000                     # hard ceiling (only huge tall + long runs reach it and get coarser leaves)
ROS_DENS = 112.0                     # leaf tufts per m2 of visible face at leaf_size 1
FLAKE_DENS = 150.0                  # mid-layer leaf-mass flakes per m2
FAR_COAT = 24.0                     # far level: leaf-coat kites per m2 of visible face (stand-ins for the leaf tufts)
FAR_KITE = 0.16                     # far level: length of a leaf-coat kite (m): one stands for a hand of leaf tufts
FAR_DARK = 0.27                      # far level: a skin face wears the dark leaf mass above this share of dark leaves
FAR_SHOOTS = 0.6                    # far level: share of the escaped shoots / drooping sprigs kept (one kite each)


# ----------------------------------------------------------------------------- small helpers
def _hx(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def _mulhex(a, b):
    ca, cb = _hx(a), _hx(b)
    return "#" + "".join("%02x" % int(round(clamp(x * y) * 255)) for x, y in zip(ca, cb))


def _tris(obj):
    realize(obj)
    return sum(len(f.vertices) - 2 for f in obj.data.polygons)


def _get_co(obj):
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _set_co(obj, co):
    obj.data.vertices.foreach_set("co", np.ascontiguousarray(co, dtype=np.float32).ravel())
    obj.data.update()


def _clamp_mesh(obj, xlim=None, zmin=None):
    realize(obj)
    co = _get_co(obj)
    if xlim is not None:
        co[:, 0] = np.clip(co[:, 0], -xlim, xlim)
    if zmin is not None:
        co[:, 2] = np.maximum(co[:, 2], zmin)
    _set_co(obj, co)


def _soft_limit(obj, xlim, zmin, m=0.045):
    """Keep foliage inside |x| <= xlim without flattening it onto a plane: the last `m` metres are
    compressed smoothly (tanh), so leaves near a butt end keep their relief."""
    realize(obj)
    co = _get_co(obj)
    xe = xlim - m
    ax = np.abs(co[:, 0])
    over = ax > xe
    co[over, 0] = np.sign(co[over, 0]) * (xe + m * np.tanh((ax[over] - xe) / m))
    co[:, 2] = np.maximum(co[:, 2], zmin)
    _set_co(obj, co)


def _outline(hwb, hwt, zb, zt, rb, rt, n):
    """Closed hedge cross-section (y, z), n points evenly spaced, starting at the bottom centre and
    running up the +y side: rounded bottom corners (rb), battered sides, top corners / dome (rt)."""
    pts = [(0.0, zb)]

    def arc(cy, cz, r, a0, a1, k=8):
        for i in range(k + 1):
            a = math.radians(a0 + (a1 - a0) * i / k)
            pts.append((cy + r * math.cos(a), cz + r * math.sin(a)))

    arc(hwb - rb, zb + rb, rb, -90, 0)
    arc(hwt - rt, zt - rt, rt, 0, 90)
    pts.append((0.0, zt))
    poly = np.array(pts + [(-y, z) for (y, z) in reversed(pts[1:-1])])
    seg = np.roll(poly, -1, axis=0) - poly
    sl = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(sl)])
    out = []
    for k in range(n):
        s = cum[-1] * k / n
        i = int(min(np.searchsorted(cum, s, side="right") - 1, len(poly) - 1))
        out.append(poly[i] + seg[i] * ((s - cum[i]) / max(sl[i], 1e-9)))
    return np.array(out), float(cum[-1])


def _leaf_mesh(name, leaves, twig=None, planar=False):
    """leaves: list of (matrix 4x4 np, length, width, oval?) - each a cupped blade lying in its local XY
    plane pointing +Y, upper face +Z: a 6-point oval (4 tris), a 4-point kite (2 tris) or a blunt lobe (3). UVs map the
    blade to the whole 'leaf' tile (0.25 m, midrib along U); planar=True writes metre UVs instead (for
    the leaf-mass flakes in the 'hedge' material)."""
    V, F, UV = [], [], []
    T = 0.25
    for (M, l, w, oval) in leaves:
        c = w * 0.14
        if oval == 2:                                      # blunt 5-point lobe (3 tris)
            loc = [(0, 0, 0), (w * 0.5, l * 0.45, c), (w * 0.24, l * 0.93, 0.5 * c), (-w * 0.24, l * 0.93, 0.5 * c), (-w * 0.5, l * 0.45, c)]
            fs = ((0, 1, 2), (0, 2, 3), (0, 3, 4))
        elif oval:
            loc = [(0, 0, 0), (w * 0.46, l * 0.3, c), (w * 0.43, l * 0.68, c), (0, l, 0.25 * c), (-w * 0.43, l * 0.68, c), (-w * 0.46, l * 0.3, c)]
            fs = ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5))
        else:
            loc = [(0, 0, 0), (w * 0.5, l * 0.44, c), (0, l, 0.3 * c), (-w * 0.5, l * 0.44, c)]
            fs = ((0, 1, 2), (0, 2, 3))
        b = len(V)
        wp = []
        for (x, y, z) in loc:
            q = M @ np.array([x, y, z, 1.0])
            wp.append(q)
            V.append(tuple(q[:3]))
        for f in fs:
            F.append(tuple(b + i for i in f))
            if planar:
                UV.extend((wp[i][0], wp[i][1]) for i in f)
            else:
                UV.extend(((0.04 + loc[i][1] / l * 0.92) * T, (0.5 + loc[i][0] / w * 0.86) * T) for i in f)
    if twig is not None:                                   # 3-sided twig along the given points
        pts, r0, r1 = twig
        b = len(V)
        for k, q in enumerate(pts):
            r = r0 + (r1 - r0) * k / (len(pts) - 1)
            for a in (0, 120, 240):
                V.append((q[0] + r * math.cos(math.radians(a)), q[1] + r * math.sin(math.radians(a)), q[2]))
        for k in range(len(pts) - 1):
            for s in range(3):
                a0, a1 = b + k * 3 + s, b + k * 3 + (s + 1) % 3
                F.append((a0, a1, a1 + 3, a0 + 3))
                UV.extend([(0.2 * T, 0.49 * T), (0.2 * T, 0.51 * T), (0.6 * T, 0.51 * T), (0.6 * T, 0.49 * T)])
    o = mesh_object(name, V, F)
    o.data.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(UV, dtype=np.float32).ravel())
    uv_keep(o)
    smooth(o, 80)
    return o


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _tr(x, y, z):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def _rosette(name, r, n):
    """Tuft at the clipped tip of a twig: n oval leaves at irregular angles around local +Z."""
    leaves = []
    a0 = r.u(0, 6.28)
    for k in range(n):
        az = a0 + k * 6.283 / n + r.jitter(0, 0.85)
        el = math.radians(r.u(10, 68))
        l = r.u(0.034, 0.05)
        leaves.append((_rz(az) @ _tr(r.jitter(0, 0.006), 0.002, r.u(0.0, 0.02)) @ _rx(el), l, l * r.u(0.5, 0.6), True))
    return _leaf_mesh(name, leaves)


def _flake(name, r, n):
    """Mid-layer leaf mass: n broad lobes of different size, tilted, in the 'hedge' material (a dense
    mat of tiny leaves) - these overlap like scales between the core and the outer tufts."""
    lobes = []
    a0 = r.u(0, 6.28)
    for k in range(n):
        az = a0 + k * 6.283 / n + r.jitter(0, 0.5)
        l = r.u(0.055, 0.085)
        lobes.append((_rz(az) @ _tr(0, 0, r.u(0.0, 0.012)) @ _rx(math.radians(r.u(4, 34))), l, l * r.u(0.68, 0.84), 2))
    return _leaf_mesh(name, lobes, planar=True)


def _shoot(name, r, length, twig=True):
    """Escaped shoot along local +Z: opposite leaf pairs (privet) on a thin curved twig."""
    pairs = max(2, int(length / (0.04 if length < 0.17 else 0.054)))     # long escaped shoots: longer internodes
    bend = r.u(0.15, 0.5)
    pts, leaves = [], []
    for k in range(pairs + 1):
        t = k / pairs
        q = (bend * length * t * t * 0.6, 0.0, length * t * (1 - 0.12 * t))
        pts.append(q)
        if k == 0:
            continue
        sc = 1.0 - 0.4 * t
        base_az = (k % 2) * 1.5708 + r.jitter(0, 0.3)
        for side in (0, 3.1416):
            el = math.radians(r.u(25, 55))
            l = r.u(0.036, 0.05) * sc
            leaves.append((_tr(*q) @ _rz(base_az + side) @ _rx(el), l, l * 0.52, length < 0.17 or k >= pairs - 1))
    leaves.append((_tr(*pts[-1]) @ _rx(math.radians(80)), 0.03, 0.015, False))
    tw = pts[::2] if (len(pts) % 2 == 1) else pts[::2] + [pts[-1]]
    return _leaf_mesh(name, leaves, twig=(tw, 0.0028, 0.0012) if twig else None)


def _frames(Pn, D, spin, scale):
    """4x4 matrices taking local +Z to directions D at points Pn, with spin about Z and uniform scale."""
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
    ref = np.where((np.abs(D[:, 2]) < 0.9)[:, None], np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    T = np.cross(ref, D)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    B = np.cross(D, T)
    c, s = np.cos(spin)[:, None], np.sin(spin)[:, None]
    X = (T * c + B * s) * scale[:, None]
    Y = (-T * s + B * c) * scale[:, None]
    M = np.zeros((len(Pn), 4, 4))
    M[:, :3, 0], M[:, :3, 1], M[:, :3, 2], M[:, :3, 3] = X, Y, D * scale[:, None], Pn
    M[:, 3, 3] = 1.0
    return M


def _scatter(Q, QN, e, nrng):
    """Stratified scatter on quads. Q, QN: (m, 4, 3) corner positions / normals, e: expected count per
    quad. Each quad gets floor/ceil(e) points on a jittered sub-grid -> even cover, no clumps or holes."""
    n = np.floor(e).astype(int) + (nrng.random(len(e)) < (e - np.floor(e)))
    qi, uu, vv = [], [], []
    for q in np.nonzero(n)[0]:
        k = int(n[q])
        g = int(math.ceil(math.sqrt(k)))
        cells = nrng.permutation(g * g)[:k]
        qi.extend([q] * k)
        uu.extend((cells % g + nrng.random(k)) / g)
        vv.extend((cells // g + nrng.random(k)) / g)
    if not qi:
        return np.zeros((0, 3)), np.zeros((0, 3))
    qi, u, v = np.array(qi, dtype=int), np.array(uu)[:, None], np.array(vv)[:, None]
    w = ((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v)
    pts = sum(Q[qi, k] * w[k] for k in range(4))
    nor = sum(QN[qi, k] * w[k] for k in range(4))
    nor /= np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
    return pts, nor


# ----------------------------------------------------------------------------- the asset
def build(p, rng):
    L, H, W = float(p.length), float(p.height), float(p.width)
    hl, hw = L / 2, W / 2
    shag = float(p.shag)
    seed = int(p.seed)
    nrng = np.random.default_rng(seed * 7919 + 13)
    off = Vector((seed * 3.17, seed * 1.31, seed * 0.77))
    rounded_ends = p.ends == "rounded"
    castel = p.top == "castellated"
    kerb = p.kerb
    far = lod() == 2
    kw, kh = (0.11, 0.15) if kerb == "stone" else (0.1025, 0.13)
    yk = hw + 0.07 + kw / 2                                  # kerb centre line
    bed_h = 0.085 if kerb != "none" else 0.045               # soil level
    ys = (hw + 0.075) if kerb != "none" else (hw + 0.1)      # half width of the soil strip

    # (leaf albedo 0.06-0.2, dull: brighter, purer greens read as lime, self-lit foliage under a lantern at night)
    m_shell = mat("hedge", _mulhex(p.tint, "#4a5544"))       # dark interior seen through the gaps
    m_flake = mat("hedge", _mulhex(p.tint, "#66735a"))       # mid layer leaf mass
    m_leaf_d = mat("leaf", _mulhex(p.tint, "#4e5f4c"))       # old dark leaves (lower half, interior)
    m_leaf_m = mat("leaf", _mulhex(p.tint, "#62734f"))       # this year's leaves
    m_leaf_p = mat("leaf", _mulhex(p.tint, "#7f8c58"))       # pale new growth (top shoots only)
    m_dead = mat("leaf", "#836852", rough=1.3)               # browned die-back and fallen leaves
    m_bark = mat("wood_weathered", "#8c8072", rough=1.1, scale=2.0)     # smooth grey-buff privet / box stems
    m_soil = mat(p.mulch)
    other = []                                               # non-foliage objects (for the tri count)

    def nz(a, b, c):
        return mnoise.noise(Vector((a, b, c)) + off)

    # ---- planting strip: soil, mounded towards the stems -------------------------------------
    def soil_z(x, y):
        e = clamp(1.0 - (abs(y) / ys) ** 2.2)
        n = mnoise.noise(Vector((x * 5.0, y * 5.0, 0.3)) + off)
        if kerb != "none":
            z = bed_h + 0.025 * e + 0.012 * n
        else:
            z = bed_h * e ** 0.8 * (1.0 + 0.25 * n)
            if rounded_ends:
                z *= smoothstep(0.0, 0.18, hl - abs(x))
        return z

    nu, nv = ((max(3, int(L / 0.7)), 3) if kerb != "none" else (max(6, int(L / 0.2)), 5)) if far else (max(4, int(L / 0.14)), 9)   # spilled earth keeps its mound
    sv, sf = [], []
    for i in range(nu + 1):
        for j in range(nv + 1):
            x, y = -hl + L * i / nu, -ys + 2 * ys * j / nv
            z = soil_z(x, y)
            if kerb == "none":                               # spilled earth: wavy edge on the paving
                y *= 1.0 + 0.16 * mnoise.noise(Vector((x * 2.6, 3.0 if y > 0 else 9.0, 0.0)) + off)
            sv.append((x, y, z))
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            sf.append((a, a + nv + 1, a + nv + 2, a + 1))
    if not rounded_ends:                                     # closed cut ends so butted runs show no hole
        for i, flip in ((0, False), (nu, True)):
            row = [i * (nv + 1) + j for j in range(nv + 1)]
            if kerb != "none":
                x = sv[row[0]][0]
                b = len(sv)
                sv += [(x, ys, 0.0), (x, -ys, 0.0)]
                row += [b, b + 1]
            sf.append(tuple(reversed(row)) if flip else tuple(row))
    soil = mesh_object("soil", sv, sf)
    assign(soil, m_soil)
    smooth(soil, 50)
    paint(soil, special=mask_mul(noise_mask(0.5, seed=seed + 5, lo=0.45, hi=0.8), 0.5))
    other.append(soil)

    # ---- kerb --------------------------------------------------------------------------------
    runs = []                                                # (x0, y0, x1, y1) centre lines
    if kerb != "none":
        xe = hl - (kw if rounded_ends else 0.0)
        runs += [(-xe, -yk, xe, -yk), (-xe, yk, xe, yk)]
        if rounded_ends:
            runs += [(-hl + kw / 2, -yk - kw / 2, -hl + kw / 2, yk + kw / 2), (hl - kw / 2, -yk - kw / 2, hl - kw / 2, yk + kw / 2)]
    if far and kerb != "none":                               # far level: one plain bar per run
        kz = kh if kerb == "stone" else 0.12
        bars = []
        for (bx0, by0, bx1, by1) in runs:
            along_x = abs(bx1 - bx0) > abs(by1 - by0)
            ln = abs(bx1 - bx0) if along_x else abs(by1 - by0)
            bars.append(box("kerb", (ln, kw, kz) if along_x else (kw, ln, kz), loc=((bx0 + bx1) / 2, (by0 + by1) / 2, kz / 2), segments=1))
        kerb_o = join(bars, "kerb") if len(bars) > 1 else bars[0]
        if kerb == "stone":
            assign(kerb_o, mat("stone_rough", "#b3aa98"))
            paint(kerb_o, grime=height_ramp(0.0, 0.12),
                  special=mask_mul(noise_mask(0.14, seed=seed + 2, lo=0.4, hi=0.75), facing((0, 0.35, 1), 1.0), 0.6))
        else:
            assign(kerb_o, mat("terracotta", "#8e6252", special=1.2))
            paint(kerb_o, grime=height_ramp(0.0, 0.1),
                  special=mask_mul(noise_mask(0.2, seed=seed + 2, lo=0.35, hi=0.7), 0.8))
        _clamp_mesh(kerb_o, xlim=hl, zmin=0.0)
        other.append(kerb_o)
    elif kerb == "stone":
        m_stone = mat("stone_rough", "#b3aa98")
        kr = rng.fork("kerb")
        stones = []
        for (x0, y0, x1, y1) in runs:
            along_x = abs(x1 - x0) > abs(y1 - y0)
            run_len = (x1 - x0) if along_x else (y1 - y0)
            n = max(1, int(round(run_len / kr.u(0.48, 0.62))))
            cuts = [0.0] + sorted(clamp((k + kr.jitter(0, 0.16)) / n, 0.02, 0.98) for k in range(1, n)) + [1.0]
            for k in range(n):
                a, b = cuts[k] * run_len, cuts[k + 1] * run_len
                ln = b - a - 0.007
                h = kh + kr.jitter(0, 0.008)
                c = (a + b) / 2
                loc = (x0 + c, y0 + kr.jitter(0, 0.004), h / 2) if along_x else (x0 + kr.jitter(0, 0.004), y0 + c, h / 2)
                rot = (0, 0, kr.jitter(0, 0.6) + (0 if along_x else 90))
                stones.append(stone_block("kerb", (ln, kw, h), loc=loc, rot=rot, chamfer=0.014, rough=0.004, rng=kr, detail=0.14))
        kerb_o = join(stones, "kerb")
        assign(kerb_o, m_stone)
        _clamp_mesh(kerb_o, xlim=hl, zmin=0.0)
        paint(kerb_o, grime=height_ramp(0.0, 0.12),
              special=mask_mul(noise_mask(0.14, seed=seed + 2, lo=0.4, hi=0.75), facing((0, 0.35, 1), 1.0), 0.6))
        other.append(kerb_o)
    elif kerb == "brick":
        # Victorian dog-tooth edging: bricks set on the diagonal, each leaning on the next, half buried
        m_brick = mat("terracotta", "#8e6252", special=1.2)
        kr = rng.fork("kerb")
        brick = box("brick", (0.215, 0.1025, 0.065), bevel=0.004, segments=1)
        pitch = 0.065 / math.sin(math.radians(45)) + 0.003
        mats = []
        for (x0, y0, x1, y1) in runs:
            along_x = abs(x1 - x0) > abs(y1 - y0)
            run_len = (x1 - x0) if along_x else (y1 - y0)
            n = max(1, int(run_len / pitch) + 1)
            for k in range(n + 1):
                s = k * pitch
                if s > run_len + 0.05:
                    break
                loc = (x0 + s, y0 + kr.jitter(0, 0.004), 0.036 + kr.jitter(0, 0.005)) if along_x else \
                      (x0 + kr.jitter(0, 0.004), y0 + s, 0.036 + kr.jitter(0, 0.005))
                mats.append(trs(loc, (kr.jitter(0, 2.5), -45 + kr.jitter(0, 3.0), (0 if along_x else 90) + kr.jitter(0, 1.5))))
        kerb_o = instances_to_mesh("kerb", brick, mats, keep_base=False, seed=seed)
        assign(kerb_o, m_brick)
        _clamp_mesh(kerb_o, xlim=hl, zmin=0.0)
        co = _get_co(kerb_o)                                 # keep the cross returns inside the footprint
        co[:, 1] = np.clip(co[:, 1], -(yk + kw * 0.75), yk + kw * 0.75)
        _set_co(kerb_o, co)
        paint(kerb_o, grime=height_ramp(0.0, 0.1),
              special=mask_mul(noise_mask(0.2, seed=seed + 2, lo=0.35, hi=0.7), 0.8))
        other.append(kerb_o)

    # ---- foliage: the clip surface S (where the shears cut; leaves are scattered on it) ---------
    skirt = float(p.skirt)
    zb0 = bed_h + min(skirt, H * 0.38)                       # nominal underside of the foliage
    zt = H
    inset = (kw + 0.07 if kerb != "none" else 0.07) if rounded_ends else 0.03
    x0, x1 = -hl + inset, hl - inset
    span = x1 - x0
    batter = 0.84                                            # hedges are clipped narrower at the top
    if p.top == "rounded":
        rt = min(hw * batter * 0.98, (zt - zb0) * 0.6)
    else:
        rt = min(0.055 + 0.05 * shag, hw * 0.4)
    rb = min(0.1, hw * 0.4)
    _, perim = _outline(hw, hw * batter, zb0, zt, rb, rt, 8)
    NV = int(clamp(round(perim / 0.125 / 2) * 2, 16, 40))
    ring, _ = _outline(hw, hw * batter, zb0, zt, rb, rt, NV)
    hrel = np.clip((ring[:, 1] - zb0) / (zt - zb0), 0, 1)

    # castellation: flat-topped merlons / crenels ~0.5 m long, merlons centred on the ends so butted
    # runs continue the rhythm; near-vertical step faces (2 stations 4 cm apart at each step)
    n_per = max(1, int(round(L / 1.05)))
    per = L / n_per
    sw = 0.02
    step = min(clamp(0.16 * H, 0.2, 0.25), 0.42 * (zt - zb0))
    cr = rng.fork("castle")
    steps = [-hl + per * (0.25 + 0.5 * k) + cr.jitter(0, 0.015) for k in range(2 * n_per)] if castel else []
    steps = [s for s in steps if x0 + 0.1 < s < x1 - 0.1]

    def merlon(x):
        if not steps:
            return 1.0
        k = int(np.argmin([abs(x - s) for s in steps]))
        d = x - steps[k]
        up_side = -1.0 if k % 2 == 0 else 1.0                # run starts inside a merlon
        if x0 + 0.1 >= -hl + per * 0.25:                     # (a first step was dropped: cannot happen for L >= 1)
            up_side = -up_side
        return clamp(0.5 + up_side * d / (2 * sw))

    R = min(hw * 1.15, span * 0.5)
    rc = 0.03                                                # leafy rounding of the butt corner
    if rounded_ends:
        xs = list(np.linspace(x0, x1, max(6, int(round(span / 0.1))) + 1))
        for f in (0.03, 0.09, 0.18):
            xs += [x0 + R * f, x1 - R * f]
    else:
        xs = [x0, x0 + 0.012, x0 + 0.03, x1 - 0.03, x1 - 0.012, x1]
        xs += list(np.linspace(x0 + 0.075, x1 - 0.075, max(4, int(round((span - 0.15) / 0.1))) + 1))
    step_x = [s + d for s in steps for d in (-sw, sw)]
    xs = [x for x in xs if all(abs(x - s) > 0.034 for s in step_x)] + step_x
    xs = sorted(xs)
    xs = np.array([x for k, x in enumerate(xs) if k == 0 or x - xs[k - 1] > 0.009])
    NU = len(xs)

    belly = 0.15 * shag ** 1.5                               # overgrown hedges swell at mid height
    bulge = 0.03 + 0.045 * shag                              # each shrub bulges: relief at 0.5-0.9 m along X
    sag = 0.11 * clamp(skirt / 0.22, 0.4, 1.0)
    S = np.zeros((NU, NV, 3))
    endfade = np.ones(NU)
    zb_arr = np.zeros(NU)
    sgn = np.sign(ring[:, 0])
    wy = np.clip(np.abs(ring[:, 0]) / (0.6 * hw), 0, 1)
    for i, x in enumerate(xs):
        sy = sz = 1.0
        d_end = min(x - x0, x1 - x)
        if rounded_ends:
            t = clamp(1.0 - d_end / R)
            s = max(math.sqrt(max(1.0 - t * t, 0.0)), 0.3)
            sy, sz = s, 0.66 + 0.34 * s ** 0.8
            ef = 1.0
        else:
            ef = endfade[i] = 0.25 + 0.75 * smoothstep(0.0, 0.3, d_end)   # ends stay nominal so runs line up
        zt_x = zt - (step * (1.0 - merlon(x)) if castel else 0.0)
        zt_x += (0.012 + 0.035 * shag) * H * nz(x * 1.3, 2.3, 5.0) * ef   # the top is not dead level
        zb_x = clamp(zb0 + sag * nz(x * 1.9, 4.4, 1.0) * ef - 0.01 * ef, bed_h + 0.045, zb0 + 0.06)   # sagging underside
        zb_arr[i] = zb_x
        oy = (0.012 + 0.03 * shag) * nz(x * 0.9, 7.1, 0.0) * ef           # the line wanders a little
        z = zb_x + (ring[:, 1] - zb0) * sz * (zt_x - zb_x) / (zt - zb0)
        lump = np.array([nz(x * 2.1, 11.3 if sg > 0 else 27.9, zz * 1.5) for sg, zz in zip(sgn, z)])
        y = ring[:, 0] * (1.0 + belly * np.sin(np.pi * hrel) ** 1.2 * ef) + sgn * wy * (bulge * 2.0 * lump + belly * 0.1) * ef
        S[i, :, 0] = x
        S[i, :, 1] = oy + y * sy
        S[i, :, 2] = z
        if not rounded_ends and d_end < rc:                  # round the butt corner a little (in the y/z plane)
            o = rc * (1.0 - math.sqrt(max(1.0 - (1.0 - d_end / rc) ** 2, 0.0)))
            tg = np.roll(S[i, :, 1:], -1, axis=0) - np.roll(S[i, :, 1:], 1, axis=0)
            n2 = np.stack([tg[:, 1], -tg[:, 0]], axis=1)
            n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)
            S[i, :, 1:] -= n2 * o

    def grid_normals(G):
        tu = np.gradient(G, axis=0)
        tv = np.roll(G, -1, axis=1) - np.roll(G, 1, axis=1)
        n = np.cross(tv, tu)
        return n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-9)

    N = grid_normals(S)
    a1, a2 = 0.012 + 0.035 * shag, 0.008 + 0.012 * shag
    for i in range(NU):
        for j in range(NV):
            q = Vector(S[i, j])
            d = a1 * mnoise.noise(q * 4.2 + off) + a2 * mnoise.noise(q * 10.0 + off * 2.0)
            n = N[i, j].copy()
            if n[2] < -0.35:                                 # ragged, browsed underside
                d += 0.06 * mnoise.noise(Vector((q.x * 7.0, q.y * 7.0, 1.7)) + off) - 0.01
            n[0] *= smoothstep(0.0, 0.1, min(q.x - x0, x1 - q.x)) * (0.3 if castel else 1.0)
            S[i, j] += n * d * endfade[i]
    S[:, :, 0] = np.clip(S[:, :, 0], x0, x1)
    S[:, :, 2] = np.maximum(S[:, :, 2], bed_h + 0.045)
    N = grid_normals(S)

    SV = [S.reshape(-1, 3)]
    SN = [N.reshape(-1, 3)]
    nvert = NU * NV
    quads = []
    for i in range(NU - 1):
        for j in range(NV):
            j2 = (j + 1) % NV
            quads.append((i * NV + j, i * NV + j2, (i + 1) * NV + j2, (i + 1) * NV + j))
    n_body = len(quads)
    for i, sign in ((0, -1), (NU - 1, 1)):                   # end caps
        c = S[i].mean(axis=0)
        prev = [i * NV + j for j in range(NV)]
        if rounded_ends:
            scales = ()
        else:                                                # butt end: leafy relief, noised 0.5-4 cm inward
            scales = (0.7, 0.4)
        for sc in scales:
            rv = c[None, :] + (S[i] - c[None, :]) * sc
            inw = np.array([clamp(0.02 + 0.05 * nz(q[1] * 8.0, q[2] * 8.0, 3.3 * sign), 0.004, 0.04) for q in rv])
            rv[:, 0] = c[0] - sign * inw
            rn = (S[i] - c[None, :]) * np.array([[0, 1, 1]])
            rn = rn / np.maximum(np.linalg.norm(rn, axis=1, keepdims=True), 1e-9) * 0.4 + np.array([[sign, 0, 0]])
            SV.append(rv)
            SN.append(rn / np.linalg.norm(rn, axis=1, keepdims=True))
            cur = [nvert + j for j in range(NV)]
            nvert += NV
            for j in range(NV):
                j2 = (j + 1) % NV
                quads.append((cur[j], prev[j], prev[j2], cur[j2]) if sign > 0 else (cur[j2], prev[j2], prev[j], cur[j]))
            prev = cur
        cc = c.copy()
        cc[0] += sign * (0.05 if rounded_ends else -0.025)
        SV.append(cc[None, :])
        SN.append(np.array([[float(sign), 0, 0]]))
        ci = nvert
        nvert += 1
        for j in range(NV):
            j2 = (j + 1) % NV
            quads.append((ci, prev[j], prev[j2], ci) if sign > 0 else (ci, prev[j2], prev[j], ci))
    SV, SN = np.concatenate(SV), np.concatenate(SN)
    quads = np.array(quads)

    # ---- inner shell: the dark interior, 3-4.5 cm behind the clip surface ---------------------
    rec = np.array([0.034 + 0.02 * mnoise.noise(Vector(q) * 6.0 + off) + 0.03 * smoothstep(0.2, 0.45, mnoise.noise(Vector(q) * 8.5 + off * 5.0))
                    for q in SV])                            # deeper under the gaps in the leaf coat
    shv = SV - SN * rec[:, None]
    if not rounded_ends:
        shv[:, 0] = np.clip(shv[:, 0], x0, x1)
    shv[:, 2] = np.maximum(shv[:, 2], bed_h + 0.04)
    shell_tris = 0
    if not far:
        shell = mesh_object("foliage_core", shv.tolist(), [tuple(q) if q[0] != q[3] else tuple(q[:3]) for q in quads.tolist()])
        assign(shell, m_shell)
        smooth(shell, 70)
        paint(shell, grime=lambda co, n: np.clip(0.45 + np.clip(1.0 - (co[:, 2] - zb0) / 0.3, 0, 1) * 0.5 + np.clip(-n[:, 2], 0, 1) * 0.5, 0, 1))
        shell_tris = _tris(shell)

    def zb_at(x):
        return float(np.interp(x, xs, zb_arr))

    # ---- stems: irregularly spaced shrubs, each 3-5 forking stems leaning outward --------------
    sr = rng.fork("stems")
    stems, twig_tips = [], []
    low = skirt < 0.14                                       # hardly any gap: fewer, simpler stems
    yj = min(0.08, hw * 0.3)
    px = -hl + sr.u(0.1, 0.24)
    k_pl = 0
    while px < hl - 0.09:
        k_pl += 1
        py = sr.jitter(0, yj) + ((-1) ** k_pl * hw * 0.3 if W > 0.8 else 0.0)
        this_x, px = px, px + sr.u(0.24, 0.47)
        if rounded_ends and hl - abs(this_x) < inset + 0.12:
            continue
        gz = soil_z(this_x, py)
        n_st = int(sr.pick([2, 3] if low else [3, 4, 4, 5]))
        a0 = sr.u(0, 6.283)
        for s in range(n_st):
            az = a0 + s * 6.283 / n_st + sr.jitter(0, 0.5)
            r0 = sr.u(0.019, 0.03) if s == 0 else sr.u(0.01, 0.02)      # 2-6 cm thick
            reach = sr.u(0.4, 1.0) * min(hw * 0.75, 0.32)
            ex = clamp(this_x + math.cos(az) * reach * 0.8, -hl + 0.045, hl - 0.045)
            ey = clamp(py + math.sin(az) * reach, -hw * 0.62, hw * 0.62)
            top = zb_at(ex) + sr.u(0.1, 0.17)
            bx, by = this_x + math.cos(az) * sr.u(0.012, 0.035), py + math.sin(az) * sr.u(0.012, 0.035)
            path = [(bx, by, gz - 0.02)]
            for t in (0.3, 0.62, 1.0):
                kink = 0.02 * (1 - t)
                path.append((lerp(bx, ex, t ** 1.5) + sr.jitter(0, kink), lerp(by, ey, t ** 1.5) + sr.jitter(0, kink), lerp(gz, top, t)))
            if not far:
                stems.append(sweep("stem", path, r0, sides=5, step=0.12, cap=False, scale_fn=lambda t: 1.0 - 0.42 * t))
            elif s < 2:                                      # far: the two thickest of each shrub, straight
                stems.append(sweep("stem", [path[0], path[3]], r0, sides=3, step=4.0, smooth_path=False, cap=False,
                                   scale_fn=lambda t: 1.0 - 0.42 * t))
            if low or not sr.chance(0.7):
                continue
            # fork: a thinner branch leaves the stem part-way up and reaches further out
            fk = path[1] if sr.chance(0.5) else path[2]
            faz = az + sr.pick([-1, 1]) * sr.u(0.5, 1.2)
            fr = reach + sr.u(0.06, 0.16)
            fx_, fy_ = clamp(this_x + math.cos(faz) * fr * 0.8, -hl + 0.045, hl - 0.045), clamp(py + math.sin(faz) * fr, -hw * 0.66, hw * 0.66)
            ftop = zb_at(fx_) + sr.u(0.07, 0.14)
            mid = (lerp(fk[0], fx_, 0.45) + sr.jitter(0, 0.01), lerp(fk[1], fy_, 0.45) + sr.jitter(0, 0.01), lerp(fk[2], ftop, 0.6))
            fr0 = r0 * sr.u(0.42, 0.6)
            if not far:
                stems.append(sweep("fork", [fk, mid, (fx_, fy_, ftop)], fr0, sides=4, step=0.13, cap=False,
                                   scale_fn=lambda t: 1.0 - 0.4 * t))
            if sr.chance(0.4):                               # low epicormic twig poking out under the skirt, leaves at its tip
                side = -1.0 if sr.chance(0.5) else 1.0
                tz = lerp(gz, zb_at(fk[0]), sr.u(0.45, 0.85))
                tip = (clamp(fk[0] + sr.jitter(0, 0.1), -hl + 0.06, hl - 0.06), side * hw * sr.u(0.7, 1.0), tz + sr.u(-0.02, 0.05))
                tm = (lerp(fk[0], tip[0], 0.5), lerp(fk[1], tip[1], 0.55), lerp(fk[2], tip[2], 0.5) - 0.015)
                if not far:
                    stems.append(sweep("twig", [fk, tm, tip], 0.0045, sides=3, step=0.2, cap=False, scale_fn=lambda t: 1.0 - 0.55 * t))
                twig_tips.append((tip, (sr.jitter(0, 0.4), side * 0.8, 0.55)))
    if stems:
        stem_o = join(stems, "stems")
        assign(stem_o, m_bark)
        smooth(stem_o, 60)
        _clamp_mesh(stem_o, xlim=hl, zmin=0.0)
        paint(stem_o, grime=mask_mul(height_ramp(0.0, 0.3), 0.6),
              special=mask_mul(noise_mask(0.08, seed=seed + 9, lo=0.45, hi=0.75), 0.45))
        other.append(stem_o)

    # ---- scatter weights ------------------------------------------------------------------------
    Q, QN = SV[quads], SN[quads]
    qc = Q.mean(axis=1)
    qn = np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1])
    area = 0.5 * np.linalg.norm(qn, axis=1)
    qn = QN.mean(axis=1)
    qn /= np.maximum(np.linalg.norm(qn, axis=1, keepdims=True), 1e-9)
    q_zb = np.interp(qc[:, 0], xs, zb_arr)
    q_h = qc[:, 2] - q_zb                                    # height above the local underside
    under = qn[:, 2] < -0.45
    w_face = np.where(under, 0.05, 0.62 + 0.38 * np.clip(q_h / 0.2, 0, 1))   # sparser band in the bottom 20 cm
    vis_area = float(area[~under].sum())
    w_top = area * (0.1 + np.clip(qn[:, 2], 0, 1) * 1.7 + np.clip(q_h, 0, 10) / (zt - zb0) * 0.45) * (~under)
    w_low = area * ((q_h < 0.14) & (np.abs(qn[:, 1]) > 0.35)) * 1.0

    # ---- leaf prototypes + triangle bookkeeping ---------------------------------------------------
    lr = rng.fork("leaves")
    ls = float(p.leaf_size) ** 0.8
    ros_bases = [_rosette(f"ros{k}", lr.fork(f"r{k}"), n) for k, n in enumerate((3, 4, 4, 5))]
    ros_bases += [_shoot(f"sprig{k}", lr.fork(f"s{k}"), 0.075 + 0.03 * k, twig=False) for k in range(2)]
    p_sprig = 0.07 + 0.16 * shag                             # share of short shoots sticking out of the faces
    wsel = np.array([(1 - p_sprig) / 4] * 4 + [p_sprig / 2] * 2)
    ros_t = float(sum(w * _tris(o) for w, o in zip(wsel, ros_bases)))
    flake_bases = [_flake(f"flake{k}", lr.fork(f"f{k}"), n) for k, n in enumerate((3, 3, 4))]
    flake_t = float(np.mean([_tris(o) for o in flake_bases]))
    sl0, sl1 = 0.08 + 0.12 * shag ** 2, 0.12 + 0.25 * shag ** 2    # top shoots: 8-12 cm clipped ... 19-35 cm overgrown
    shoot_bases = [_shoot(f"shoot{k}", lr.fork(f"sh{k}"), lerp(sl0, sl1, k / 3.0)) for k in range(4)]
    shoot_t = float(np.mean([_tris(o) for o in shoot_bases]))
    droop_bases = [_shoot(f"droop{k}", lr.fork(f"d{k}"), 0.09 + 0.035 * k) for k in range(2)]
    droop_t = float(np.mean([_tris(o) for o in droop_bases]))

    n_litter = int(p.litter * L * 2 * ys * 30)
    fixed = shell_tris + sum(_tris(o) for o in other) + n_litter * 2
    d_ros = ROS_DENS / (ls ** 2 if ls > 1 else ls ** 1.2)
    d_flake = FLAKE_DENS / ls
    d_shoot = 8.0 + 22.0 * shag
    d_droop = 26.0 + 30.0 * shag
    wf = float((area * w_face).sum())
    var = wf * (d_ros * ros_t + d_flake * flake_t) + float(w_top.sum()) * d_shoot * shoot_t + float(w_low.sum()) * d_droop * droop_t
    f = min(1.0, (TRI_CAP - fixed) / max(var, 1.0))          # only enormous runs hit the cap: fewer, larger leaves
    grow = min(1.0 / math.sqrt(f), 1.6)

    # die-back: a few distinct brown patches on the long faces
    n_die = int(float(p.dieback) * (0.9 + 1.5 * L) + lr.u())
    cand = np.nonzero((np.abs(qn[:, 1]) > 0.7) & (q_h > 0.12) & (qc[:, 2] < zb0 + 0.85 * (zt - zb0)) & (np.abs(qc[:, 0]) < hl - 0.3))[0]
    die = []
    for _ in range(n_die * 25):
        if len(die) >= n_die or not len(cand):
            break
        c = qc[int(cand[int(lr.u() * len(cand)) % len(cand)])]
        if all(np.linalg.norm(c - d[0]) > 0.6 for d in die):
            die.append((c, lr.u(0.16, 0.28) * clamp(0.75 + 0.25 * (zt - zb0) / 0.8, 0.75, 1.4)))

    def die_dist(pts):
        dd = np.full(len(pts), 9.0)
        for (c, rad) in die:
            dd = np.minimum(dd, np.linalg.norm((pts - c[None, :]) * np.array([[1.0, 0.6, 1.0]]), axis=1) / rad)
        return dd

    def vnoise(pts, fx_, fy_, fz_, o):
        return np.array([mnoise.noise(Vector((q[0] * fx_, q[1] * fy_, q[2] * fz_)) + off * o) for q in pts])

    if far:
        # ---- far level: the leaf coat as one closed coarse skin + a share of the shoots as kites ----
        delete(ros_bases + flake_bases + shoot_bases + droop_bases)
        leaf_mats_far = [m_leaf_d, m_leaf_m, m_leaf_p, m_dead]
        ry, rz = ring[:, 0] / hw, (ring[:, 1] - (zb0 + zt) / 2) / ((zt - zb0) / 2)
        half = NV // 2
        pos = np.arange(1, half)                             # the +y side, bottom centre and top centre excluded
        jb = int(pos[np.argmax((ry - rz)[pos])])             # bottom corner
        jt = int(pos[np.argmax((ry / batter + rz)[pos])])    # top corner / shoulder
        jm = int(pos[np.argmin(np.abs(hrel[pos] - 0.5))])    # mid height (the belly)
        js = {jb, jm, jt}
        if p.top == "rounded":                               # dome: one more on the shoulder and the crown
            ang = np.arctan2(rz, np.maximum(ry, 1e-9))
            js |= {int(pos[np.argmin(np.abs(ang[pos] - math.radians(a)))]) for a in (40, 66)} | {half}
        jj = sorted(js | {(NV - j) % NV for j in js})
        ii = [0]
        for i in range(1, NU - 1):
            d_end = min(xs[i] - x0, x1 - xs[i])
            gap_i = 0.3 * (clamp(d_end / R, 0.25, 1.0) if rounded_ends else 1.0)
            if any(abs(xs[i] - sx) < 1e-6 for sx in step_x) or (xs[i] - xs[ii[-1]] >= gap_i and x1 - xs[i] >= gap_i * 0.5):
                ii.append(i)
        ii.append(NU - 1)
        nI, nJ = len(ii), len(jj)
        push = 0.02 + 0.02 * shag                            # the leaf coat stands this far proud of the clip surface
        G = S[np.ix_(ii, jj)] + N[np.ix_(ii, jj)] * push
        lim = hl if not rounded_ends else hl - (kw if kerb != "none" else 0.0)
        G[:, :, 0] = np.clip(G[:, :, 0], -lim, lim)
        G[:, :, 2] = np.maximum(G[:, :, 2], bed_h + 0.03)
        GN = N[np.ix_(ii, jj)]
        fv = [tuple(q) for q in G.reshape(-1, 3)]
        ff, fm = [], []
        for a in range(nI - 1):
            for b in range(nJ):
                b2 = (b + 1) % nJ
                ff.append((a * nJ + b, a * nJ + b2, (a + 1) * nJ + b2, (a + 1) * nJ + b))
                c = (G[a, b] + G[a, b2] + G[a + 1, b2] + G[a + 1, b]) / 4
                nn = GN[a, b] + GN[a, b2] + GN[a + 1, b2] + GN[a + 1, b]
                nn = nn / max(float(np.linalg.norm(nn)), 1e-9)
                zb_c = zb_at(c[0])
                hh_c = clamp((c[2] - zb0) / (zt - zb0))
                drift = mnoise.noise(Vector((c[0] * 1.5, c[1] * 0.6, c[2] * 0.7)) + off * 3.0)
                p_dark = 0.4 - 0.3 * hh_c + 0.6 * drift + 0.3 * clamp(1.0 - (c[2] - zb_c) / 0.22)
                if nn[2] < -0.45:
                    fm.append(1)                             # the browsed underside: dark interior
                else:
                    fm.append(1 if p_dark > FAR_DARK else 0)
        ff.append(tuple(reversed(range(nJ))))                # end caps
        ff.append(tuple((nI - 1) * nJ + b for b in range(nJ)))
        fm += [0, 0]
        skin = mesh_object("foliage_far", fv, ff)
        assign(skin, m_flake)
        skin.data.materials.append(m_shell)
        for pl, k in zip(skin.data.polygons, fm):
            pl.material_index = k
        smooth(skin, 70)
        paint(skin, grime=lambda co, nn: np.clip(1.0 - (co[:, 2] - np.interp(co[:, 0], xs, zb_arr)) / 0.24, 0, 1) * 0.75)

        def kites(name, pts, d, length, width, m_, spin=None):
            if not len(pts):
                return
            n_ = len(pts)
            base = _leaf_mesh(name + "_k", [(_rx(math.radians(90)), 1.0, width / length, False)])
            M_ = _frames(pts, d, nrng.uniform(0, 6.283, n_) if spin is None else nrng.normal(0, spin, n_), nrng.uniform(0.8, 1.2, n_) * length)
            o = instances_to_mesh(name, base, list(M_), uv_jitter=False, keep_base=False, seed=seed)
            assign(o, m_)
            uv_keep(o)
            flat(o)
            _soft_limit(o, lim, bed_h + 0.03)

        # the leaf coat: broad kites lying on the skin like scales, in the leaf colours of that spot (dark low
        # down and where the old leaves gather, pale on top, browned in the die-back patches)
        pts, nor = _scatter(Q, QN, area * w_face * FAR_COAT * f, nrng)
        if len(pts):
            n_ = len(pts)
            tg = np.array([[0, 0, 1.0]]) - nor * nor[:, 2:3]
            hz = nrng.uniform(0, 6.283, n_)
            tg = np.where((np.linalg.norm(tg, axis=1) < 0.5)[:, None], np.stack([np.cos(hz), np.sin(hz), np.zeros(n_)], axis=1), tg)
            tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
            d = tg * 0.8 + nor * 0.5 + nrng.normal(0, 0.15, (n_, 3))
            hh_p = np.clip((pts[:, 2] - zb0) / (zt - zb0), 0, 1)
            h_low = np.clip(1.0 - (pts[:, 2] - np.interp(pts[:, 0], xs, zb_arr)) / 0.22, 0, 1)
            p_dk = np.clip(0.4 - 0.3 * hh_p + 0.6 * vnoise(pts, 1.5, 0.6, 0.7, 3.0) + 0.3 * h_low, 0.04, 0.9)
            r_ = nrng.random(n_)
            cls = np.where(r_ < p_dk, 0, np.where(r_ > 1.0 - 0.2 * np.clip(nor[:, 2], 0, 1), 2, 1))
            cls = np.where(nrng.random(n_) < np.clip((1.25 - die_dist(pts)) * 4.0, 0, 0.95), 3, cls)
            for k_, m_ in enumerate(leaf_mats_far):
                sel = cls == k_
                kites(f"coat{k_}", pts[sel] + nor[sel] * push * 0.6, d[sel], FAR_KITE * ls ** 0.5, FAR_KITE * 0.75 * ls ** 0.5, m_, spin=0.4)
        pts, nor = _scatter(Q, QN, w_top * d_shoot * f * FAR_SHOOTS, nrng)
        if len(pts):
            n_ = len(pts)
            d = nor + np.array([0, 0, 0.9]) + nrng.normal(0, 0.26, (n_, 3))
            pale = nrng.random(n_) < 0.62
            sl = lerp(sl0, sl1, 0.6) * ls ** 0.7
            kites("shoots_p", pts[pale] + nor[pale] * push * 0.5, d[pale], sl, 0.07 * ls ** 0.7, m_leaf_p)
            kites("shoots_m", pts[~pale] + nor[~pale] * push * 0.5, d[~pale], sl, 0.07 * ls ** 0.7, m_leaf_m)
        pts, nor = _scatter(Q, QN, w_low * d_droop * f * FAR_SHOOTS, nrng)
        if len(pts):
            d = nor * 0.75 + np.array([0, 0, -0.55]) + nrng.normal(0, 0.25, (len(pts), 3))
            kites("droop", pts, d, 0.11 * ls ** 0.7, 0.065 * ls ** 0.7, m_leaf_d)
        print(f"DBG far L={L} skin={_tris(skin)} stations={nI} ring={nJ}")
        return

    by_mat = {}

    def stamp(name, bases, which, M, mi, mats_, jit=False):
        for bi, b in enumerate(bases):
            for k, m in enumerate(mats_):
                sel = (which == bi) & (mi == k)
                if sel.any():
                    o = instances_to_mesh(f"{name}_{bi}_{k}", b, list(M[sel]), uv_jitter=jit, seed=seed + bi * 7 + k)
                    by_mat.setdefault(m.name, (m, []))[1].append(o)
            delete(b)

    leaf_mats = [m_leaf_d, m_leaf_m, m_leaf_p, m_dead]

    # outer coat: rosettes (and a share of short sprigs) on every visible face, with see-through gaps
    pts, nor = _scatter(Q, QN, area * w_face * d_ros * f, nrng)
    n = len(pts)
    gap = vnoise(pts, 8.5, 8.5, 8.5, 5.0)
    dd = die_dist(pts)
    keep = (gap < 0.3) | (dd < 0.9)                          # dead leaves hang on: no gaps inside a brown patch
    pts, nor, dd = pts[keep], nor[keep], dd[keep]
    n_ros = n = len(pts)
    proud = nrng.random(n)
    d = nor + np.array([0, 0, 0.3]) + nrng.normal(0, 0.3, (n, 3))
    d += (np.linalg.norm(d, axis=1) < 0.2)[:, None] * nor
    hh = np.clip((pts[:, 2] - zb0) / (zt - zb0), 0, 1)
    h_low = np.clip(1.0 - (pts[:, 2] - np.interp(pts[:, 0], xs, zb_arr)) / 0.22, 0, 1)
    drift = vnoise(pts, 1.5, 0.6, 0.7, 3.0)                  # each shrub flushes a little differently
    p_dark = np.clip(0.4 - 0.3 * hh + 0.6 * drift + 0.3 * (0.5 - proud) + 0.3 * h_low, 0.04, 0.9)
    p_pale = np.clip(0.14 * np.clip(nor[:, 2], 0, 1) + 0.05 * (hh > 0.85) - 0.2 * drift * (hh > 0.7), 0, 0.22)
    r = nrng.random(n)
    mi = np.where(r < p_dark, 0, np.where(r > 1.0 - p_pale, 2, 1))
    mi = np.where(nrng.random(n) < np.clip((1.25 - dd) * 4.0, 0, 0.95), 3, mi)
    M = _frames(pts + nor * (0.004 + proud * (0.026 + 0.035 * shag))[:, None], d, nrng.uniform(0, 6.283, n),
                nrng.uniform(0.86, 1.14, n) * ls * grow)
    which = nrng.choice(len(ros_bases), size=n, p=wsel)
    dead = np.nonzero(mi == 3)[0]
    if len(dead):                                            # browned patches are matted: a second coat of dead tufts
        M2 = _frames(pts[dead] + nrng.normal(0, 0.022, (len(dead), 3)) - nor[dead] * 0.012, nor[dead] + nrng.normal(0, 0.45, (len(dead), 3)),
                     nrng.uniform(0, 6.283, len(dead)), nrng.uniform(0.9, 1.2, len(dead)) * ls * grow)
        M, mi, which = np.concatenate([M, M2]), np.concatenate([mi, np.full(len(dead), 3)]), np.concatenate([which, which[dead]])
    stamp("ros", ros_bases, which, M, mi, leaf_mats)

    # mid layer: tilted leaf-mass flakes just above the core, overlapping like scales
    pts, nor = _scatter(Q, QN, area * np.where(under, 0.12, w_face) * d_flake * f, nrng)
    n = len(pts)
    keep = (vnoise(pts, 8.5, 8.5, 8.5, 5.0) < 0.44) & ~((die_dist(pts) < 0.95) & (nrng.random(n) < 0.85))
    pts, nor = pts[keep], nor[keep]
    n_flake = n = len(pts)
    d = nor + nrng.normal(0, 0.4, (n, 3))
    M = _frames(pts + nor * nrng.uniform(-0.014, 0.012, (n, 1)), d, nrng.uniform(0, 6.283, n), nrng.uniform(0.8, 1.25, n) * max(ls, 0.8) * grow)
    stamp("flake", flake_bases, nrng.integers(0, len(flake_bases), n), M, np.zeros(n, dtype=int), [m_flake], jit=True)

    # escaped shoots, mostly on the top and the upper shoulders (the pale new growth lives here)
    pts, nor = _scatter(Q, QN, w_top * d_shoot * f, nrng)
    n_shoot = n = len(pts)
    if n:
        d = nor + np.array([0, 0, 0.9]) + nrng.normal(0, 0.26, (n, 3))
        r = nrng.random(n)
        mi = np.where(r < 0.62, 2, 1)
        mi = np.where(nrng.random(n) < np.clip(1.3 - die_dist(pts), 0, 1) * 0.9, 3, mi)
        M = _frames(pts, d, nrng.uniform(0, 6.283, n), nrng.uniform(0.8, 1.2, n) * ls ** 0.7 * (0.5 + 0.5 * np.clip(nor[:, 2] * 1.5, 0, 1)))
        stamp("shoot", shoot_bases, nrng.integers(0, len(shoot_bases), n), M, mi, leaf_mats)
    else:
        delete(shoot_bases)

    # drooping sprigs along the lower edge + leaves on the low twigs: no straight shadow line underneath
    pts, nor = _scatter(Q, QN, w_low * d_droop * f, nrng)
    d = nor * 0.75 + np.array([0, 0, -0.55]) + nrng.normal(0, 0.25, (len(pts), 3))
    if twig_tips:
        pts = np.concatenate([pts, np.array([t[0] for t in twig_tips])])
        d = np.concatenate([d, np.array([t[1] for t in twig_tips])])
    n = len(pts)
    if n:
        M = _frames(pts, d, nrng.uniform(0, 6.283, n), nrng.uniform(0.85, 1.2, n) * ls ** 0.7)
        stamp("droop", droop_bases, nrng.integers(0, len(droop_bases), n), M, np.where(nrng.random(n) < 0.6, 0, 1), leaf_mats)
    else:
        delete(droop_bases)

    lim = hl if not rounded_ends else hl - (kw if kerb != "none" else 0.0)
    leaf_tris = 0
    for k, (mn, (m, objs)) in enumerate(sorted(by_mat.items())):
        o = join(objs, f"leaves{k}") if len(objs) > 1 else objs[0]
        assign(o, m)
        uv_keep(o)
        smooth(o, 80)
        _soft_limit(o, lim, bed_h + 0.03)
        paint(o, grime=lambda co, nn: np.clip(1.0 - (co[:, 2] - np.interp(co[:, 0], xs, zb_arr)) / 0.24, 0, 1) * 0.75)
        leaf_tris += _tris(o)

    # ---- fallen leaves on the bed ------------------------------------------------------------
    if n_litter > 0:
        lit = _leaf_mesh("litter_leaf", [(np.eye(4), 0.044, 0.024, False)])
        mats = []
        for _ in range(n_litter):
            x, y = lr.u(-hl + 0.05, hl - 0.05), lr.u(-ys + 0.03, ys - 0.03)
            z = soil_z(x, y) + 0.004
            if z < 0.012:
                continue
            mats.append(trs((x, y, z), (lr.jitter(0, 14), lr.jitter(0, 14), lr.u(0, 360)), lr.u(0.7, 1.15)))
        lo = instances_to_mesh("litter", lit, mats, uv_jitter=False, keep_base=False)
        assign(lo, m_dead)
        uv_keep(lo)
        flat(lo)

    tot = fixed + leaf_tris
    print(f"DBG L={L} tris~{tot} ({tot / L:.0f}/m, foliage {leaf_tris / max(vis_area, 1e-6):.0f}/m2 of {vis_area:.1f} m2) shell={shell_tris} "
          f"other={fixed - shell_tris} ros={n_ros} flakes={n_flake} shoots={n_shoot} cap_f={f:.2f} die={len(die)}")
