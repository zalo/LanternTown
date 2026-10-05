"""Night watchman's fire brazier in forged iron. Three splayed square-bar legs (twisted between forge
collars, flattened toes, scrolled tops) carry either a riveted strap-iron fire basket (uprights with
curled tips inside three hoops, bar grate) or a hammered fire bowl with rolled rim and punched vent
slits. A stretcher ring ties the legs and holds a sheet-iron ash pan with an uneven ash heap. The fire
is a dark clinker mass packed with individually shaped coal lumps (black crust / grey ash outside,
glowing only in the crevices; a coal basket is heaped to the brim) or short split logs leaning into an
ember bed, with a low crown of overlapping flames, smoke, a shadow-casting fire light inside the heap and
a soft under-glow (burnt down: no flame, ash-grey lumps, a faint ember glow). A poker hangs from a hook
and two drop-ring handles let it be carried.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, degrees, atan2, hypot, sqrt, acos, floor, exp
from mathutils import Vector, Quaternion

META = {"name": "Iron fire brazier", "class": "prop", "mount": "ground", "focus": [0.0, 0.0, 0.66]}
PARAMS = {
    "diameter": {"default": 0.5, "min": 0.35, "max": 0.8, "desc": "diameter of the fire basket / bowl at the rim (m)"},
    "height": {"default": 0.85, "min": 0.55, "max": 1.3, "desc": "height of the rim above the paving (m)"},
    "style": {"default": "basket", "choices": ["basket", "bowl"], "desc": "riveted strap-iron basket or hammered slotted bowl"},
    "fuel": {"default": "coal", "choices": ["coal", "logs"], "desc": "heap of coal lumps or split logs over embers"},
    "intensity": {"default": 55.0, "min": 4, "max": 120, "desc": "fire light (cd); low values = dying embers, more ash, no flame"},
    "bars": {"default": 18, "min": 9, "max": 30, "desc": "number of basket uprights (rounded to a multiple of 3)"},
    "rust": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "amount of rust / fire scale on the ironwork"},
    "poker": {"default": True, "desc": "poker hanging from a hook on the rim"},
    "handles": {"default": True, "desc": "two drop-ring carrying handles"},
}
VARIANTS = {
    "log_bowl": {"style": "bowl", "fuel": "logs", "diameter": 0.72, "height": 0.6, "intensity": 85, "seed": 4},
    "tall_log_basket": {"fuel": "logs", "diameter": 0.4, "height": 1.2, "bars": 12, "intensity": 65, "rust": 0.25, "seed": 11},
    "coal_bowl": {"style": "bowl", "diameter": 0.56, "height": 0.72, "intensity": 45, "rust": 0.7, "seed": 23},
    "dying_embers": {"diameter": 0.6, "height": 0.7, "bars": 21, "intensity": 9, "rust": 0.9, "poker": False, "seed": 31},
}

_ICO_F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
          (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
DARK, ASH, DEEP, DIM, HOT, WOOD, END = range(7)


def _pol(a, r, z, t=0.0):
    """Cylindrical -> xyz; t = offset along the tangent."""
    ca, sa = cos(a), sin(a)
    return (r * ca - t * sa, r * sa + t * ca, z)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _ico(level):
    """Unit icosphere (verts (n,3), tris (m,3)) with outward winding."""
    f = (1 + sqrt(5)) / 2
    V = [np.array(v, dtype=float) / sqrt(1 + f * f) for v in
         [(-1, f, 0), (1, f, 0), (-1, -f, 0), (1, -f, 0), (0, -1, f), (0, 1, f), (0, -1, -f), (0, 1, -f), (f, 0, -1), (f, 0, 1), (-f, 0, -1), (-f, 0, 1)]]
    F = list(_ICO_F)
    for _ in range(level):
        mid, F2 = {}, []

        def m(a, b):
            k = (a, b) if a < b else (b, a)
            if k not in mid:
                v = V[a] + V[b]
                mid[k] = len(V)
                V.append(v / np.linalg.norm(v))
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    V, F = np.array(V), np.array(F)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = np.einsum("ij,ij->i", n, V[F].mean(1)) < 0
    F[flip] = F[flip][:, ::-1]
    return V, F


def _multi(name, V, faces, midx, mats):
    """Mesh with per-face library materials (only the used ones get a slot)."""
    o = mesh_object(name, [tuple(map(float, v)) for v in V], [tuple(map(int, f)) for f in faces])
    midx = np.asarray(midx, dtype=np.int32)
    used = sorted(set(midx.tolist()))
    for s, k in enumerate(used):
        assign(o, mats[k], slot=s)
    if len(o.data.polygons) == len(midx):
        lut = np.zeros(max(used) + 1, dtype=np.int32)
        lut[used] = np.arange(len(used))
        o.data.polygons.foreach_set("material_index", lut[midx])
        o.data.update()
    else:
        print("DBG face count changed in", name)
    return o


def _vmask(arr):
    """Precomputed per-vertex mask as a paint() callable."""
    arr = np.asarray(arr, dtype=np.float32)
    return lambda co, n: arr if len(co) == len(arr) else np.full(len(co), float(arr.mean()), dtype=np.float32)


class _Coals:
    """All coal / ember / cinder lumps of the fire as ONE mesh: every lump is an individually cleaved,
    squashed icosphere; faces turned away from the viewer (into the heap) are dropped; each face is
    classed black crust / grey ash / deep red / red / orange from the lump's heat and how far the face
    is turned out of the heap (outward = cooled crust, crevices and inward faces = glowing)."""

    def __init__(self, seed, h1):
        self.npr = np.random.default_rng(seed)
        self.h1 = h1
        self.tpl = {False: _ico(0), True: _ico(1)}
        self.V, self.F, self.M, self.G, self.n = [], [], [], [], 0

    def add(self, c, rad, heat=0.0, out=(0, 0, 1), fine=True, cull=-0.12, ash=0.3):
        npr = self.npr
        V, F = self.tpl[fine]
        P = V * (1 + npr.uniform(-0.15, 0.13, len(V)))[:, None]
        for _ in range(2 if fine else 1):                      # cleaved flats
            d = npr.normal(size=3)
            d /= np.linalg.norm(d)
            P = P - np.outer(np.maximum(P @ d - npr.uniform(0.45, 0.8), 0.0) * 0.85, d)
        P = P * (np.array([1.1, 0.92, 0.76]) * npr.uniform(0.9, 1.1, 3))
        Q, _ = np.linalg.qr(npr.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        c = np.array(c, dtype=float)
        P = P @ Q.T * rad + c
        out = np.array(out, dtype=float)
        out /= np.linalg.norm(out)
        tri = P[F]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        o2 = out + npr.normal(size=3) * 0.28
        o2 /= np.linalg.norm(o2)
        e = fn @ o2
        keep = (fn @ out) > cull
        # glow only where the lump is turned into the heap (crevices), in ordered bands orange -> red -> deep red
        d = (-0.2 + 0.42 * heat) - e + npr.uniform(-0.025, 0.025, len(F)) if heat > 0.02 else np.full(len(F), -1.0)
        adir = out * 0.55 + np.array([0.0, 0.0, 0.55]) + npr.normal(size=3) * 0.3
        adir /= np.linalg.norm(adir)
        a = ash + 0.45 * (fn @ adir) + npr.uniform(-0.05, 0.05, len(F))
        m = np.where(d > 0.24, HOT if self.h1 > 0.45 else DIM, np.where(d > 0.1, DIM, np.where(d > 0.0, DEEP, np.where(a > 0.6, ASH, DARK))))
        F, m = F[keep], m[keep]
        used = np.unique(F)
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        vn = P[used] - c
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        # soot mask: clean pale ash in the middle of the ash side, sooty towards its edge and on the black crust
        self.G.append(0.22 + 0.7 * (1.0 - _ss(0.5, 0.98, ash + 0.45 * (vn @ adir))))
        self.V.append(P[used])
        self.F.append(remap[F] + self.n)
        self.M.append(m)
        self.n += len(used)

    def build(self, name, mats):
        V, F, M = np.concatenate(self.V), np.concatenate(self.F), np.concatenate(self.M)
        o = _multi(name, V, F, M, mats)
        smooth(o, 33)
        uv_box(o)
        return o, np.concatenate(self.G)


def _quad2(p0, p1, p2, n):
    """Quadratic bezier in 2D (fillet with p1 as the corner), n segments, without p0."""
    out = []
    for i in range(1, n + 1):
        t = i / n
        out.append(((1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t * t * p2[0], (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t * t * p2[1]))
    return out


def _hoop(name, rfn, z, w, th, seg=40):
    """Flat-bar hoop following the (conical) vessel wall: rfn(z) = inner radius of the hoop."""
    r0, r1 = rfn(z - w / 2), rfn(z + w / 2)
    z0, z1 = z - w / 2, z + w / 2
    e = 0.0012
    return lathe(name, [(r0, z0), (r0 + th - e, z0), (r0 + th, z0 + e), (r1 + th, z1 - e), (r1 + th - e, z1), (r1, z1), (r0, z0)], segments=seg, smooth_angle=35)


def _log(name, L, r, rng, zc, split, h1, mats, m_bark):
    """Short firewood billet along local +Z; z = 0 is the end lying in the fire. The first `zc` metres
    are charred: shrunken alligator-cracked char blocks whose cracks (thin rows / columns of faces)
    glow towards the burning end; beyond that bark and pale split faces, end grain on the far end.
    Returns (bark shell, body with char / wood / end grain / embers)."""
    # ---- cross-section loop (CCW seen from +Z) and the kind of the column that starts at each point
    pts, ck = [], []
    dth = 0.0022 / r
    if split:
        span = radians(rng.u(125, 178))
        nb = 7
        cx = -r * 0.38
        cracks = {2, 5} if rng.chance(0.5) else {1, 3, 6}
        pts.append((cx - r * rng.u(0.0, 0.1), rng.jitter(0, r * 0.1)))
        ck.append("split")
        for k in range(nb + 1):
            a = -span / 2 + span * k / nb
            rr = r * rng.u(0.9, 1.1)
            if k in cracks:
                pts += [(cx + rr * cos(a - dth), rr * sin(a - dth)), (cx + rr * cos(a + dth), rr * sin(a + dth))]
                ck += ["crack", "bark"]
            else:
                pts.append((cx + rr * cos(a), rr * sin(a)))
                ck.append("bark" if k < nb else "split")
    else:
        nb = 10
        cracks = {0, 3, 7}
        for k in range(nb):
            a = 2 * pi * k / nb + rng.jitter(0, 0.05)
            rr = r * 0.92 * rng.u(0.92, 1.08)
            if k in cracks:
                pts += [(rr * cos(a - dth), rr * sin(a - dth)), (rr * cos(a + dth), rr * sin(a + dth))]
                ck += ["crack", "bark"]
            else:
                pts.append((rr * cos(a), rr * sin(a)))
                ck.append("bark")
    m = len(pts)
    # ---- rings: char blocks separated by thin crack rows, then raw wood
    zs, rk, blk = [0.0], [], [0]
    z = 0.0
    while True:
        b = rng.u(0.026, 0.046)
        if z + b > zc - 0.014:
            zs.append(zc)
            rk.append("block")
            blk.append(blk[-1])
            break
        z += b
        zs.append(z)
        rk.append("block")
        blk.append(blk[-1])
        z += 0.0035
        zs.append(z)
        rk.append("crack")
        blk.append(blk[-1] + 1)
    nraw = max(1, int(round((L - zc) / 0.075)))
    for i in range(1, nraw + 1):
        zs.append(zc + (L - zc) * i / nraw)
        rk.append("raw")
        blk.append(-1)
    boff = [rng.jitter(0, 0.045) for _ in range(max(blk) + 2)]
    bow = rng.jitter(0, 0.006)
    V, vz = [], []
    for i, z in enumerate(zs):
        t = clamp(z / zc)
        s = 0.72 + 0.28 * t * t * (3 - 2 * t)
        if blk[i] >= 0 and z < zc - 1e-6:
            s *= 1.0 + boff[blk[i]]
        elif blk[i] < 0:
            s *= 1.0 + rng.jitter(0, 0.03)
        ox = bow * sin(pi * z / L) + rng.jitter(0, 0.0015)
        oy = rng.jitter(0, 0.0015)
        for (x, y) in pts:
            V.append((ox + x * s, oy + y * s, z + (rng.jitter(0, 0.004) if i == len(zs) - 1 else 0.0)))
            vz.append(z)
    hotc = HOT if h1 > 0.45 else DIM
    bark_f, body_f, body_m = [], [], []
    for i, kind in enumerate(rk):
        zm = (zs[i] + zs[i + 1]) / 2
        for j in range(m):
            j2 = (j + 1) % m
            q = (i * m + j, i * m + j2, (i + 1) * m + j2, (i + 1) * m + j)
            if kind == "raw":
                if ck[j] == "split":
                    body_f.append(q)
                    body_m.append(WOOD)
                else:
                    bark_f.append(q)
            elif kind == "crack" or ck[j] == "crack":
                gl = (1.0 - zm / (zc * 0.9)) * (0.55 + 0.65 * h1) + rng.jitter(0, 0.2)
                body_f.append(q)
                body_m.append(hotc if gl > 0.72 else DIM if gl > 0.4 else DEEP if gl > 0.12 else DARK)
            else:
                body_f.append(q)
                body_m.append(ASH if (zm < zc * 0.6 and rng.chance(0.14)) else DARK)
    # burning end: glowing rim around a charred heart; far end: end grain
    n0 = len(V)
    for (x, y) in pts:
        V.append((x * 0.72 * 0.6, y * 0.72 * 0.6, -0.004))
        vz.append(0.0)
    for j in range(m):
        j2 = (j + 1) % m
        body_f.append((j2, j, n0 + j, n0 + j2))
        body_m.append(hotc if rng.chance(0.5 * h1) else DIM)
    body_f.append(tuple(n0 + j for j in reversed(range(m))))
    body_m.append(DEEP if rng.chance(0.5) else DARK)
    top = (len(zs) - 1) * m
    body_f.append(tuple(top + j for j in range(m)))
    body_m.append(END)
    V = np.array(V)
    vz = np.array(vz)
    out = []
    for nm, faces, mi in ((name + "_bark", bark_f, None), (name + "_body", body_f, body_m)):
        used = sorted({v for f in faces for v in f})
        remap = {v: k for k, v in enumerate(used)}
        ff = [tuple(remap[v] for v in f) for f in faces]
        if mi is None:
            o = mesh_object(nm, [tuple(v) for v in V[used]], ff)
            assign(o, m_bark)
            smooth(o, 50)
            uv_cyl(o, axis="Z")
        else:
            o = _multi(nm, V[used], ff, mi, mats)
            smooth(o, 42)
            uv_grain(o, axis="Z")
        zz = vz[used]
        # scorched next to the char line, sooty all over
        paint(o, grime=_vmask(0.5 + 0.48 * (1.0 - _ss(zc, zc + 0.7 * (L - zc), zz))))
        out.append(o)
    return out


def build(p, rng):
    D, H = p.diameter, p.height
    R = D / 2
    bowl = p.style == "bowl"
    logs = p.fuel == "logs"
    heat = clamp(p.intensity / 60.0, 0.05, 1.5)
    h1 = min(heat, 1.0)
    burning = heat > 0.3
    brim = not bowl and not logs     # coal basket: heaped to the top hoop, so lumps (not bare fire-lit iron) show inside the rim
    seed = int(p.seed)

    iron = mat("iron_wrought")
    # grey-blue fire scale on the iron that sits against a coal fire (log fires spend that material slot on the wood)
    scale = mat("zinc") if not logs else iron
    g = 0.45 + 0.55 * h1
    m_hot = mat("emissive", "#ff7a1e", glow=round(1.0 * g, 2))
    m_dim = mat("emissive", "#e24a10", glow=round(0.7 * g, 2))
    m_deep = mat("emissive", "#9a2608", glow=round(0.45 * g, 2))
    m_ash = mat("mortar", "#514e4b", scale=2.0)
    m_dark = mat("wood_oak", "#17120f", wear=0.0) if logs else mat("slate", "#2e2e33", scale=2.5)
    # caked soot on the iron that faces the fire. So close to the fire light any lit material (even black) burns out
    # to cream, so the soot is a near-black surface with its own faint red heat instead of a lit one.
    m_soot = mat("emissive", "#6a2409", glow=round(0.14 * g * (0.4 + 0.6 * h1), 3))
    FM = [m_dark, m_ash, m_deep, m_dim, m_hot]

    bt, bw = 0.005, 0.018            # basket strap section
    ls = 0.016 + 0.014 * D           # leg bar (square) size
    vessel, leg_iron, cold_iron, handle_iron = [], [], [], []

    # ------------------------------------------------------------------ vessel shape
    if not bowl:
        hb = clamp(0.8 * D, 0.26, H * 0.52)
        zb = H - hb
        Rb = R * 0.74

        def rv(z):
            return Rb + (R - Rb) * clamp((z - zb) / hb) ** 0.75

        def r_in(z):
            return rv(z) - bt / 2 - 0.003

        def r_out(z):
            return rv(z) + bt / 2

        hug = [(rv(zb + hb * i / 5) + bt / 2 + 0.006 + ls / 2, zb + hb * i / 5) for i in range(6)]
        hug_n = [(1.0, 0.0)] * 6
        zh = H - 0.02
        ro = r_out(zh) + 0.006
    else:
        db = clamp(0.42 * D, 0.15, H * 0.45)
        zb = H - db
        A = radians(84)
        cA, sA = cos(A), sin(A)

        def bp(t):
            f = t * A
            return (R * sin(f) / sA, H - db * (cos(f) - cA) / (1 - cA))

        def bn(t):
            r0, z0_ = bp(t)
            r1, z1_ = bp(t + 0.01)
            tx, tz = r1 - r0, z1_ - z0_
            l = hypot(tx, tz)
            return tz / l, -tx / l

        def r_out(z):
            c = cA + clamp((H - z) / db) * (1 - cA)
            return R * sin(acos(clamp(c, -1, 1))) / sA

        def r_in(z):
            return max(r_out(z) - 0.007, 0.01)

        hug, hug_n = [], []
        for i in range(6):
            t = 0.54 + 0.3 * i / 5
            r0, z0_ = bp(t)
            nx, nz = bn(t)
            hug.append((r0 + nx * (ls / 2 + 0.001), z0_ + nz * (ls / 2 + 0.001)))
            hug_n.append((nx, nz))
        zh = H - 0.05
        ro = r_out(zh) + 0.005

    # ------------------------------------------------------------------ legs
    c = ls / 2 + 0.0005
    Rf = (R * 0.95 + 0.07 * H) if bowl else (R * 1.05 + 0.06 * H)
    Bp = hug[0]
    z_s = clamp(0.42 * zb, 0.11 if bowl else 0.13, zb - 0.1)                 # stretcher ring / ash pan level

    def on(t):
        return (Rf + (Bp[0] - Rf) * t, c + (Bp[1] - c) * t)

    t0 = clamp((z_s - c) / (Bp[1] - c), 0.12, 0.6)
    t1 = t0 + (1 - t0) * 0.68
    Lst = hypot(Bp[0] - Rf, Bp[1] - c)
    turns = max(1, int(round(Lst * (t1 - t0) / 0.15)))
    sq = rounded_rect(ls, ls, 0.003, 1)
    r_s = on(t0)[0]
    leg_ang = [pi / 2 + k * 2 * pi / 3 for k in range(3)]
    rl_ = rng.fork("legs")
    for k, a in enumerate(leg_ang):
        up = (-sin(a), cos(a), 0)
        toe_l = rl_.u(0.06, 0.085)
        segA = [(Rf + toe_l + 0.015, c + 0.011), (Rf + toe_l, c + 0.003), (Rf + toe_l - 0.02, c), (Rf + 0.03, c)]
        segA += _quad2((Rf + 0.03, c), (Rf, c), on(t0 * 0.5), 5) + [on(t0)]
        nB = 8 * turns
        segB = [on(t0 + (t1 - t0) * i / nB) for i in range(nB + 1)]
        segC = [on(t1), on(0.5 + t1 * 0.5)] + _quad2(on(0.5 + t1 * 0.5), Bp, hug[1], 4) + hug[2:]
        if not bowl:
            rT, zT = hug[-1]
            rho = 0.026 + 0.01 * D
            segC.append((rT, zT + 0.02))
            for i in range(1, 12):
                th = radians(180 - 330 * i / 11)
                rr = rho * (1 - 0.55 * i / 11)
                segC.append((rT + rho + rr * cos(th), zT + 0.02 + rr * sin(th)))
            sc = (lambda t: 1.0 if t < 0.8 else 1.0 - 2.4 * (t - 0.8))
        else:
            rT, zT = hug[-1]
            nx, nz = hug_n[-1]
            segC.append((rT - nz * 0.03 + nx * 0.004, zT + nx * 0.03 + nz * 0.004))
            sc = (lambda t: 1.0 if t < 0.85 else 1.0 - 3.0 * (t - 0.85))
        leg_iron.append(sweep(f"leg{k}_foot", [_pol(a, r, z) for r, z in segA], sq, smooth_path=False, up=up,
                              scale_fn=lambda t: 0.8 + 0.2 * min(t * 6, 1.0)))
        leg_iron.append(sweep(f"leg{k}_twist", [_pol(a, r, z) for r, z in segB], sq, smooth_path=False, up=up, twist_deg=360.0 * turns))
        leg_iron.append(sweep(f"leg{k}_top", [_pol(a, r, z) for r, z in segC], sq, smooth_path=False, up=up, scale_fn=sc))
        col = rounded_rect(ls + 0.009, ls + 0.009, 0.003, 1)
        for tc in (t0, t1):
            e = 0.009 / Lst
            leg_iron.append(sweep(f"leg{k}_collar", [_pol(a, *on(tc - e)), _pol(a, *on(tc + e))], col, smooth_path=False, up=up))

    # stretcher ring (inside the legs, at the lower collars) and the ash pan dropped into it
    rr_ = r_s - ls / 2 - 0.005
    leg_iron.append(torus("stretcher", major=rr_, minor=0.007, loc=(0, 0, z_s), major_seg=32, minor_seg=6))
    rp = rr_ - 0.011
    zf_ = z_s + 0.007
    dp = 0.042
    zpt = zf_ + 0.004                 # top of the pan flange
    pan = lathe("ash_pan", [(0, zf_ - dp), (rp * 0.82, zf_ - dp), (rp * 0.96, zf_ - dp + 0.01), (rp, zf_ - 0.006), (rp + 0.005, zf_),
                            (rr_ + 0.016, zf_), (rr_ + 0.018, zf_ + 0.002), (rr_ + 0.016, zf_ + 0.004), (rp + 0.001, zf_ + 0.004),
                            (rp - 0.003, zf_ - dp + 0.012), (rp * 0.82 - 0.002, zf_ - dp + 0.003), (0, zf_ - dp + 0.003)], segments=28, smooth_angle=40)
    displace_noise(pan, 0.002, 0.07, seed=seed + 5)
    transform_verts(pan, lambda co: Vector((co.x, co.y, max(co.z, 0.004))))
    assign(pan, iron)
    # sooty, scaled sheet: dark fire scale and soot patches break up the rust, heaviest inside and on the rim
    paint(pan, grime=mask_mul(noise_mask(0.05, seed=seed + 10, lo=0.25, hi=0.6), 0.95),
          wear=mask_mul(noise_mask(0.035, seed=seed + 11, lo=0.4, hi=0.7), 0.85),
          special=lambda co, n: np.clip(0.25 + 0.3 * p.rust + noise_mask(0.08, seed=seed + 12, lo=0.3, hi=0.7)(co, n) * (0.3 + 0.4 * p.rust), 0, 1)
          * (1.0 - 0.75 * np.clip(n[:, 2] * 2.0 + 1.0 * (np.hypot(co[:, 0], co[:, 1]) < rp) * (n[:, 2] > -0.3), 0, 1)))
    paint(pan, grime=lambda co, n: np.clip(n[:, 2] * 1.5, 0, 1) * 0.8 + (np.hypot(co[:, 0], co[:, 1]) < rp) * np.clip(n[:, 2] + 1.0, 0, 1) * 0.6, mode="add")

    # uneven ash heap: a mound under the grate (offset by the seed), thin towards the pan wall
    ra_ = rng.fork("ash")
    am = _pol(ra_.u(0, 6.28), rp * ra_.u(0.12, 0.3), 0)
    sg = rp * ra_.u(0.3, 0.4)
    am2 = _pol(ra_.u(0, 6.28), rp * ra_.u(0.35, 0.55), 0)
    ah = dp * (0.42 + 0.2 * (1 - h1))
    afloor = zf_ - dp + 0.004

    def ash_z(x, y):
        e = clamp((0.93 - hypot(x, y) / rp) / 0.3)
        e = e * e * (3 - 2 * e)
        mnd = ah * exp(-((x - am[0]) ** 2 + (y - am[1]) ** 2) / (2 * sg * sg)) + 0.4 * ah * exp(-((x - am2[0]) ** 2 + (y - am2[1]) ** 2) / (0.5 * sg * sg))
        return afloor + (0.005 + mnd) * e

    ash = lathe("ash_heap", [(rp * q, 0.0005 * i) for i, q in enumerate((0.93, 0.85, 0.74, 0.62, 0.5, 0.38, 0.26, 0.13, 0.0))], segments=24, smooth_angle=80)
    transform_verts(ash, lambda co: Vector((co.x, co.y, ash_z(co.x, co.y))))
    displace_noise(ash, 0.004, 0.03, seed=seed + 8, along_normal=False, axis_mask=(0.15, 0.15, 1))
    uv_planar(ash)
    assign(ash, m_ash)
    paint(ash, grime=lambda co, n: np.clip(0.3 + _ss(0.5, 0.92, np.hypot(co[:, 0], co[:, 1]) / rp) * 0.7
                                           + noise_mask(0.045, seed=seed + 9, lo=0.3, hi=0.7)(co, n) * 0.5, 0, 1))

    # ------------------------------------------------------------------ rivets
    rivet = lathe("rivet", [(0.0066, 0.0), (0.0048, 0.0038), (0.0, 0.0054)], segments=4)
    riv = []

    def add_rivet(a, r, z, nx=1.0, nz=0.0, s=1.0):
        riv.append(trs(_pol(a, r, z), (0, degrees(atan2(nx, nz)), degrees(a)), s))

    slot_rows = []
    # ------------------------------------------------------------------ basket
    if not bowl:
        N = 3 * max(3, int(round(p.bars / 3)))
        rbar = rng.fork("bars")
        prof_bar = [(-bw / 2, -bt / 2), (bw / 2, -bt / 2), (bw / 2, bt / 2 - 0.0015), (bw / 2 - 0.0015, bt / 2), (-bw / 2 + 0.0015, bt / 2), (-bw / 2, bt / 2 - 0.0015)]
        if N >= 21 or brim:                             # plain section keeps the triangle budget (a 5 mm strap: the chamfer never reads)
            prof_bar = [(-bw / 2, -bt / 2), (bw / 2, -bt / 2), (bw / 2, bt / 2), (-bw / 2, bt / 2)]
        hz = [(zb + 0.018, 0.028, 0.005), (zb + hb * 0.52, 0.026, 0.005), (H - 0.02, 0.034, 0.006)]
        warped = rbar.pick(range(N))
        for k in range(N):
            a = pi / 2 + 2 * pi * k / N + radians(rbar.jitter(0, 0.7))
            bend = rbar.jitter(0, 0.003) + (0.012 if k == warped else 0.0)
            tw = rbar.jitter(0, 0.002)
            pts = []
            for i in range(5):
                z = zb + hb * i / 4
                w = sin(pi * i / 4)
                pts.append(_pol(a, rv(z) + bend * w, z, tw * w))
            rt = rv(H)
            ext = rbar.vary(1.0, 0.22)
            for dr, dz in ((0.0, 0.012), (0.004, 0.023), (0.011, 0.031), (0.019, 0.033), (0.025, 0.028)):
                pts.append(_pol(a, rt + dr * ext, H + dz * ext))
            vessel.append(sweep(f"bar{k}", pts, prof_bar, smooth_path=False, up=(cos(a), sin(a), 0),
                                scale_fn=lambda t: 1.0 if t < 0.82 else 1.0 - 2.4 * (t - 0.82)))
            for (z, w, th) in hz:
                if k % (N // 3) != 0:
                    add_rivet(a, r_out(z) + th, z)
        for i, (z, w, th) in enumerate(hz):
            hp = _hoop(f"hoop{i}", r_out, z, w, th, seg=36)
            displace_noise(hp, 0.0012, 0.1, seed=seed + i, along_normal=False)
            vessel.append(hp)
            # riveted lap joint of the hoop
            al = radians(rng.fork(f"lap{i}").u(110, 190))
            lap = lathe(f"lap{i}", [(r_out(z) + th, z - w / 2 + 0.002), (r_out(z) + th + 0.004, z - w / 2 + 0.003), (r_out(z) + th + 0.004, z + w / 2 - 0.003),
                                    (r_out(z) + th, z + w / 2 - 0.002)], segments=4, angle=degrees(0.07 / r_out(z)), rot=(0, 0, degrees(al)), smooth_angle=30)
            vessel.append(lap)
            for da in (0.012, 0.058):
                add_rivet(al + da / r_out(z), r_out(z) + th + 0.004, z, s=0.9)
            for a in leg_ang:
                add_rivet(a, r_out(z) + th + ls + 0.001, z, s=1.35)
        # bar grate in the bottom
        ng = 7
        gr = []
        for i in range(ng):
            y = Rb * 0.84 * (2 * i / (ng - 1) - 1)
            gr.append(cylinder(f"grate{i}", 0.0055, 2 * sqrt(Rb * Rb - y * y) - 0.004, loc=(0, y, zb + 0.008), rot=(0, 90, 0), verts=6, cap=False))
        gr.append(torus("grate_ring", major=Rb - 0.006, minor=0.006, loc=(0, 0, zb + 0.008), major_seg=32, minor_seg=6))
        for o in gr:
            assign(o, scale)
            paint(o, grime=0.55, special=mask_mul(noise_mask(0.04, seed=seed + 20), 0.8))
    # ------------------------------------------------------------------ bowl
    else:
        nt, seg = 12, 60
        ts0, ts1 = 6, 8                                  # vent slits between these profile rings
        prof = [bp(i / nt) for i in range(nt + 1)]
        prof[0] = (0.0, prof[0][1])
        bo = lathe("bowl", prof, segments=seg, smooth_angle=60)
        zlo, zhi = prof[ts0][1], prof[ts1][1]
        me = bo.data
        keep_f = []
        for f in me.polygons:
            cc = f.center
            if not (zlo < cc.z < zhi and int(floor((atan2(cc.y, cc.x) % (2 * pi)) / (2 * pi / seg))) % 3 == 1):
                keep_f.append(tuple(f.vertices))
        vs_ = [tuple(v.co) for v in me.vertices]
        delete(bo)
        bo = mesh_object("bowl", vs_, keep_f)
        recalc_normals(bo)
        smooth(bo, 60)
        uv_cyl(bo, axis="Z")
        displace_noise(bo, 0.002, 0.06, seed=seed + 2)
        solidify(bo, 0.006, offset=-1.0)
        vessel.append(bo)
        # punched slits get a raised, hammered lip so the wall thickness reads
        lipp = [(-0.004, -0.004), (0.004, -0.004), (0.004, 0.0025), (-0.004, 0.0025)]
        dt = 0.011
        for k in range(1, seg, 3):
            tm = (ts0 + ts1) / (2 * nt)
            rm = bp(tm)[0]
            a0, a1 = k * 2 * pi / seg - 0.0045 / rm, (k + 1) * 2 * pi / seg + 0.0045 / rm
            loop = []
            for a, t in ((a0, ts0 / nt - dt), (a0, tm), (a0, ts1 / nt + dt), (a1, ts1 / nt + dt), (a1, tm), (a1, ts0 / nt - dt)):
                r_, z_ = bp(t)
                nx, nz = bn(t)
                loop.append(_pol(a, r_ + nx * 0.002, z_ + nz * 0.002))
            nx, nz = bn(ts0 / nt)
            am_ = (a0 + a1) / 2
            vessel.append(sweep(f"lip{k}", loop, lipp, closed=True, smooth_path=False, up=(nx * cos(am_), nx * sin(am_), nz)))
        vessel.append(torus("rim", major=R + 0.001, minor=0.009, loc=(0, 0, H), major_seg=seg, minor_seg=6))
        vessel.append(_hoop("band", r_out, zh, 0.034, 0.005, seg=seg))
        for k in range(seg // 3):
            add_rivet((k * 3 + 2.0) * 2 * pi / seg, r_out(zh) + 0.005, zh)
        vessel.append(lathe("boss", [(0.0, zb - 0.014), (0.028 + 0.02 * D, zb - 0.014), (0.04 + 0.02 * D, zb - 0.006), (0.05 + 0.02 * D, bp(0.14)[1] - 0.001)],
                            segments=16, smooth_angle=40))
        for a in leg_ang:
            for i in (1, 4):
                add_rivet(a, hug[i][0] + hug_n[i][0] * ls / 2, hug[i][1] + hug_n[i][1] * ls / 2, hug_n[i][0], hug_n[i][1], s=1.35)

        def slot_soot(co, n):                            # soot licking up the wall above every slit
            u = (np.arctan2(co[:, 1], co[:, 0]) % (2 * pi)) / (2 * pi / seg)
            d = np.abs((u % 3.0) - 1.5)
            up_ = _ss(zlo - 0.01, zhi - 0.005, co[:, 2]) * (1.0 - _ss(zhi + 0.02, zhi + 0.13, co[:, 2]))
            return np.clip(1.75 - d * 1.15, 0, 1) * up_
        paint(bo, grime=slot_soot)

    # ------------------------------------------------------------------ handles
    if p.handles:
        rh = rng.fork("handles")
        for a0 in (0.0, pi):
            a = a0 + radians(rh.jitter(0, 4))
            cold_iron.append(torus("eye", major=0.011, minor=0.004, loc=_pol(a, ro + 0.008, zh), rot=(90, 0, degrees(a)), major_seg=10, minor_seg=5))
            mj = 0.034 + 0.016 * D
            tilt = rh.u(2, 9)
            handle_iron.append(torus("handle", major=mj, minor=0.0055, loc=_pol(a, ro + 0.013 + sin(radians(tilt)) * mj, zh - mj + 0.006),
                                     rot=(0, 90 - tilt, degrees(a)), major_seg=18, minor_seg=6))
            add_rivet(a, ro, zh + 0.012, s=0.9)

    # ------------------------------------------------------------------ fuel: dark clinker mass packed with lumps
    fr = rng.fork("fuel")
    z0 = zb + (0.016 if not bowl else 0.02)
    if bowl:
        fill_top = H - db * (0.12 if not logs else 0.3)
    else:
        fill_top = (H - 0.028) if not logs else zb + hb * 0.5
    # (burnt down, a coal basket is the same full heap under ash)
    zf = z0 + (fill_top - z0) * ((0.95 + 0.05 * h1) if brim else (0.66 + 0.34 * h1))   # heap level at the vessel wall
    hd = (0.12 if brim else (0.15 if not logs else 0.09)) * D * (0.5 + 0.5 * h1)   # crown of the heap above that
    rl = clamp(0.041 * D / 0.5, 0.034, 0.066)            # lump radius against the bars
    # triangle budget: many uprights / long legs are paid for with slightly coarser fuel (less of it shows between close bars anyway)
    rl *= 1.0 + (0.011 * max(N - 18, 0) if not bowl else 0.0) + 0.1 * max(H - 0.85, 0.0) + (1.2 * max(D - 0.66, 0.0) if logs else 0.0)
    st = rl * (0.8 if logs else 0.88)                    # lump radius on top of the heap
    rf = r_in(zf) - st * 0.5

    def S(q):
        return zf + hd * (1 - q * q)

    coals = _Coals(seed * 7919 + 13, h1)

    def heat_of(hot):
        if fr.u() < 0.14 + 0.5 * (1 - h1):
            return 0.0
        return clamp(hot * fr.u(0.5, 1.2) * (0.45 + 0.55 * h1))

    def ash_of(ht):
        return 0.2 + 0.5 * (1 - h1) + 0.22 * ht + fr.jitter(0, 0.15) + (0.08 if logs else 0.0)

    rows = []                                            # (z, hotness, outward normal (r, z), lump radius)
    if not bowl:
        zr0, zr1 = z0 + rl * 0.72, zf - rl * 0.5
        nrow = max(1, int(round((zr1 - zr0) / (rl * (1.6 if brim else 1.42)))) + 1)
        for row in range(nrow):
            f = row / max(nrow - 1, 1)
            rows.append((zr0 + (zr1 - zr0) * f if nrow > 1 else (zr0 + zr1) / 2, 0.95 - 0.35 * f, (1.0, 0.05), rl))
    else:
        tm = (ts0 + ts1) / (2 * nt)
        rsl = min(rl, (zhi - zlo) * 0.62)
        rows.append((bp(tm)[1] + 0.004, 1.0, bn(tm), rsl))
    for row, (z, hot, (nx, nz), rad) in enumerate(rows):
        rr = r_in(z) - rad * 0.9
        cnt = max(6, int(round(2 * pi * rr / (rad * 1.62))))
        a0 = fr.u(0, 6.28)
        for i in range(cnt):
            a = a0 + 2 * pi * (i + 0.5 * (row % 2) + fr.jitter(0, 0.12)) / cnt
            ht = heat_of(hot)
            coals.add(_pol(a, rr + fr.jitter(0, 0.004), z + fr.jitter(0, 0.006)), rad * fr.u(0.94, 1.1), ht,
                      (cos(a) * nx, sin(a) * nx, nz), cull=-0.06, ash=ash_of(ht))
    ntop = max(5, int((1.05 if brim else 1.25) * (rf / st) ** 2))
    for i in range(ntop):
        q = sqrt((i + 0.5) / ntop)
        a = i * 2.39996 + fr.jitter(0, 0.2)
        ht = heat_of(1.0 - 0.5 * q ** 1.5)
        coals.add(_pol(a, rf * q, S(q) - st * 0.2 + fr.jitter(0, 0.006)), st * fr.u(0.88, 1.12), ht,
                  (cos(a) * 0.55 * q, sin(a) * 0.55 * q, 1.0), cull=-0.12, ash=ash_of(ht))
    if not logs:                                         # freshly thrown-on coal: bigger black lumps riding on top
        for i in range(int(3 + 4 * h1)):
            q, a = fr.u(0.15, 0.8), fr.u(0, 6.28)
            coals.add(_pol(a, rf * q, S(q) + st * 0.62), st * 1.12, 0.0, (0, 0, 1), cull=-0.5, ash=0.1 + 0.4 * (1 - h1))
    # clinker and cinders that fell through the grate, half buried in the ash
    for i in range(9):
        q, a = fr.u(0.05, 0.78), fr.u(0, 6.28)
        x, y, _ = _pol(a, rp * q, 0)
        sz = rl * (fr.u(0.42, 0.6) if i < 3 else fr.u(0.18, 0.32))
        coals.add((x, y, ash_z(x, y) + sz * fr.u(-0.15, 0.2)), sz, 0.55 * h1 if fr.chance(0.3) else 0.0, (0, 0, 1), fine=i < 3, cull=-0.3, ash=fr.u(0.2, 0.75))
    # a few lumps and cinders dropped on the paving beside the brazier
    asp = radians(fr.u(20, 70))
    for i in range(int(fr.u(5, 9))):
        a, q = asp + fr.jitter(0, 0.5), Rf + fr.u(0.0, 0.22)
        sz = rl * fr.u(0.28, 0.6)
        coals.add(_pol(a, q, sz * 0.36), sz, 0.0, (0, 0, 1), fine=sz > rl * 0.45, cull=-0.55, ash=fr.u(0.1, 0.5))
    co_o, crust = coals.build("coals", FM)
    paint(co_o, grime=_vmask(crust))

    # the mass behind the lumps: dark, lumpy, with a little deep glow in its hollows - only seen in the gaps
    cpts = []
    for i in range(5):
        z = z0 + (zf - rl * 0.3 - z0) * i / 4
        cpts.append((max(r_in(z) - rl * 1.5, 0.02), z))
    rc0 = cpts[-1][0]
    for qq in (0.75, 0.5, 0.25, 0.0):
        cpts.append((rc0 * qq, S(rc0 * qq / rf) - st * 1.0))
    core = lathe("ember_core", cpts, segments=24, smooth_angle=50)
    displace_noise(core, min(0.011, rl * 0.3), 0.05, seed=seed + 21)
    assign(core, m_dark)

    def core_heat(cc):                                   # soft patches of heat, hotter low down where the draught comes in
        a = atan2(cc.y, cc.x)
        v = 0.5 + 0.5 * sin(a * 5 + cc.z * 37 + seed) * sin(cc.z * 61 - a * 3 + 1.3)
        return (0.35 + 0.65 * v) * (0.45 + 0.55 * h1) * (1.12 - 0.4 * clamp((cc.z - z0) / max(zf - z0, 0.05)))
    assign_faces(core, m_deep, lambda cc, n: core_heat(cc) > 0.5)
    assign_faces(core, m_dim, lambda cc, n: core_heat(cc) > 0.8)
    paint(core, grime=0.7)

    # ------------------------------------------------------------------ logs leaning into the heap
    # fire light: inside the heap, below its surface (deeper under logs, so the wood above is not scorched white)
    zl = max((z0 + 0.75 * (zf - z0)) if brim else ((zf - 0.03) if not logs else (z0 + 0.45 * (zf - z0))), zpt + R + 0.05)
    sn = zl - zpt - 0.015                                # everything above the ash pan is inside shadow_near
    if logs:
        lr = rng.fork("logs")
        nl = 3 if D < 0.45 else (4 if D < 0.62 else 5)
        m_bark = mat("bark")
        LM = FM + [mat("wood_oak", "#8f7452"), mat("wood_endgrain", "#927654")]
        a_off = lr.u(0, 6.28)
        for i in range(nl):
            a = a_off + 2 * pi * (i + lr.jitter(0, 0.15)) / nl
            r = clamp(0.085 * D, 0.03, 0.045) * lr.u(0.88, 1.12)
            Lt = clamp(0.62 * D, 0.25, 0.35) * lr.u(0.92, 1.04)
            zo = H - r * 0.4 + lr.jitter(0, 0.01)
            ro_ = r_in(zo) - r * 1.1
            po = Vector(_pol(a, ro_, zo))                # far end: propped against the wall just under the rim
            tg = _pol(a + pi / 2, r * 1.5, 0)            # aim past the centre (pinwheel): burning ends lie side by side
            dxy = Vector((tg[0] - po.x, tg[1] - po.y)).normalized()
            rise = zo - (S(0.2) + r * 0.35)
            run = min(sqrt(max(Lt * Lt - rise * rise, 0.0036)), ro_ + 0.4 * rf)
            pe = Vector((po.x + dxy.x * run, po.y + dxy.y * run, 0.0))
            pe.z = S(min(hypot(pe.x, pe.y) / rf, 1.0)) + r * 0.35 + i * r * 0.12
            d = po - pe
            L = d.length
            d.normalize()
            eul = (d.to_track_quat("Z", "Y") @ Quaternion((0, 0, 1), lr.u(0, 6.28))).to_euler("XYZ")
            parts = _log(f"log{i}", L, r, lr.fork(f"log{i}"), zc=L * lr.u(0.55, 0.7), split=lr.chance(0.7), h1=h1, mats=LM, m_bark=m_bark)
            place(parts, loc=tuple(pe), rot=(degrees(eul.x), degrees(eul.y), degrees(eul.z)))

    # ------------------------------------------------------------------ poker on a hook (clear of the ash pan, inside shadow_near)
    if p.poker:
        rk_ = rng.fork("poker")
        ap = radians(262 + rk_.jitter(0, 6))
        hook = [(ro + 0.004, zh + 0.012), (ro + 0.004, zh - 0.026), (ro + 0.01, zh - 0.041), (ro + 0.02, zh - 0.046), (ro + 0.029, zh - 0.04), (ro + 0.032, zh - 0.027)]
        cold_iron.append(sweep("hook", [_pol(ap, r, z) for r, z in hook], 0.004, sides=6, step=0.009))
        add_rivet(ap, ro + 0.008, zh + 0.004, s=0.9)
        zc = zh - 0.046 + 0.0075 - 0.019
        rc = ro + 0.02
        handle_iron.append(torus("poker_ring", major=0.019, minor=0.0038, loc=_pol(ap, rc, zc), rot=(0, 90, degrees(ap)), major_seg=16, minor_seg=6))
        zt = zc - 0.019
        ze = max(zt - clamp(H * 0.6, 0.3, 0.6), z_s + 0.13, zb - 0.06)   # a short hand poker: ends beside the vessel bottom
        lo, lt = rk_.u(0.004, 0.014), rk_.jitter(0, 0.012)
        rod = [_pol(ap, rc, zt), _pol(ap, rc + lo * 0.5, (zt + ze) / 2, lt * 0.5), _pol(ap, rc + lo, ze + 0.03, lt), _pol(ap, rc + lo, ze, lt)]
        cold_iron.append(sweep("poker", rod, 0.0055, sides=6, smooth_path=False, scale_fn=lambda t: 1.0 if t < 0.9 else 1.0 - 6.0 * (t - 0.9)))
        spur = [_pol(ap, rc + lo, ze + 0.045, lt), _pol(ap, rc + lo, ze + 0.036, lt + 0.014), _pol(ap, rc + lo, ze + 0.02, lt + 0.024), _pol(ap, rc + lo, ze + 0.008, lt + 0.024)]
        cold_iron.append(sweep("poker_spur", spur, 0.0045, sides=6, smooth_path=False, scale_fn=lambda t: 1.0 - 0.5 * t))

    # ------------------------------------------------------------------ materials + weathering of the ironwork
    # the iron tells its story in zones: grey-blue scale and orange oxide on the basket around the fire,
    # matte soot inside and above the fire line, bare burnished metal where hands and cobbles rub, rust at the feet
    cold_iron.append(instances_to_mesh("rivets", rivet, riv, keep_base=False))
    rs = 0.3 + 0.7 * p.rust
    n_ox = noise_mask(0.07, seed=seed + 1, lo=0.3, hi=0.62)
    n_bl = noise_mask(0.055, seed=seed + 13, lo=0.32, hi=0.62)
    n_fine = noise_mask(0.025, seed=seed + 15, lo=0.3, hi=0.75)

    hv = H - zb

    def inward(co, n):                                   # faces turned towards the fire
        r = np.hypot(co[:, 0], co[:, 1]) + 1e-9
        return np.clip(-(n[:, 0] * co[:, 0] + n[:, 1] * co[:, 1]) / r * 1.6, 0, 1)

    def zone(co, lo0, lo1, hi0, hi1):                    # band of the vessel height, 0 = bottom, 1 = rim
        u = (co[:, 2] - zb) / hv
        return _ss(lo0, lo1, u) * (1.0 - _ss(hi0, hi1, u))

    def soot(co, n):                                     # matte soot inside, on every ledge and above the fire line
        return np.maximum.reduce([_ss(0.52, 0.8, (co[:, 2] - zb) / hv) * 0.95, inward(co, n) * zone(co, -0.1, 0.05, 2, 3) * 0.95,
                                  np.clip(n[:, 2] * 1.8, 0, 1) * zone(co, -0.1, 0.0, 2, 3) * 0.95, 0.22 + 0.4 * n_fine(co, n)])

    def oxide(co, n):                                    # orange-brown burnt oxide low down where the draught makes it hottest
        return np.clip(zone(co, -0.25, -0.02, 0.1, 0.36) * (0.55 + 0.6 * n_ox(co, n)) * rs * 1.25
                       + zone(co, 0.1, 0.3, 0.45, 0.7) * n_ox(co, n) * n_bl(co, n) * rs * 0.9, 0, 1) * (1.0 - 0.7 * inward(co, n))

    def scale_blue(co, n):                               # fire scale flaked to bare grey-blue metal in the middle band
        return zone(co, 0.2, 0.4, 0.5, 0.7) * (0.25 + 0.6 * n_bl(co, n)) * (1.0 - inward(co, n)) * np.clip(1.0 - n[:, 2] * 2.0, 0, 1)

    Lf = Vector((0, 0, zl))

    def sooted(cc, n):                                   # faces looking into the fire (or up out of it): caked matte soot
        d = Lf - cc
        return n.dot(d) > 0.25 * d.length or (n.z > 0.6 and hypot(cc.x, cc.y) < R * 1.05)

    Lf_ = np.array(Lf)

    def clean(co, n):                                    # 0 on the soot-caked faces (their own black shows), 1 elsewhere
        d = Lf_ - co
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        return 1.0 - np.clip(((n * d).sum(axis=1) - 0.05) * 5.0, 0, 1)

    zmid = zb + hv * 0.5
    for o in vessel:
        assign(o, iron)
        if o.name.startswith(("bar", "lip", "boss")):    # heat-scaled where the fire sits against it (seams hidden by the hoops)
            assign_faces(o, scale, (lambda cc, n: cc.z < zmid) if o.name.startswith("bar") else (lambda cc, n: True))
        paint(o, grime=mask_mul(soot, clean), special=mask_mul(oxide, clean), wear=scale_blue)
        if o.matrix_world.translation.length < 1e-6:
            assign_faces(o, m_soot, sooted)
        else:                                            # (rim torus: built around its own origin)
            off = o.matrix_world.translation.copy()
            assign_faces(o, m_soot, lambda cc, n: sooted(cc + off, n))
    for o in cold_iron:
        assign(o, iron)
        paint(o, grime=mask_mul(soot, 0.8), special=lambda co, n: np.maximum(oxide(co, n), n_ox(co, n) * rs * 0.45),
              wear=lambda co, n: np.maximum(scale_blue(co, n), 0.2 + 0.35 * n_fine(co, n)))
    for o in handle_iron:                                # rubbed bright by hands
        assign(o, iron)
        paint(o, wear=lambda co, n: 0.55 + 0.45 * n_fine(co, n), special=mask_mul(n_ox, rs * 0.3))
    for o in leg_iron:
        assign(o, iron)
        paint(o, grime=lambda co, n: np.maximum(soot(co, n) * 0.85, (1.0 - _ss(0.0, 0.1, co[:, 2])) * 0.5),
              special=lambda co, n: np.maximum(np.maximum((1.0 - _ss(0.02, 0.2 + 0.4 * p.rust, co[:, 2])) * n_ox(co, n) * (0.55 + 0.45 * p.rust),
                                                          oxide(co, n) * 0.8), n_ox(co, n) * n_fine(co, n) * rs * 0.5),
              wear=lambda co, n: np.maximum(np.maximum((1.0 - _ss(0.004, 0.03, co[:, 2])) * 0.85,          # toes scraped bright on the cobbles
                                                       _ss(H - 0.01, H + 0.03, co[:, 2]) * (1.0 - inward(co, n)) * (0.35 + 0.6 * n_fine(co, n))),
                                            scale_blue(co, n) * 0.7))

    # ------------------------------------------------------------------ fire: light, flames, smoke
    I = float(p.intensity)
    lcol = "#ff9444" if h1 > 0.7 else ("#ff8238" if burning else "#ff5a2a")
    ztop = S(0.0)
    light("fire", (0, 0, zl), color=lcol, intensity=I * (0.9 if burning else 0.25), range=clamp(4.5 + I / 11.0, 4.5, 12.0), flicker=0.5 if burning else 0.25,
          shadow=True, shadow_near=round(sn, 3))
    light("underglow", (0, 0, (zb + zpt) / 2 if bowl else zb - 0.06), color="#ff5e24", intensity=I * (0.012 if bowl else 0.02) * (1.0 if burning else 0.4), range=2.0, flicker=0.35)
    if burning:
        # a low, broad fire: tight groups of small flames that overlap into licking tongues
        fs = (0.1 + 0.26 * D) * (0.65 + 0.35 * h1)
        zfl = st * 0.15
        if logs:                                         # wood burns with a taller flame than coal, rising from between the logs
            zfl += clamp(0.085 * D, 0.03, 0.045)
            if not bowl:
                fs *= 1.35
        rfl = rng.fork("flames")
        a0 = rfl.u(0, 6.28)
        nfl = 10 if D <= 0.45 else 12
        for k in range(nfl):
            # sunflower spread over ~60 % of the heap: from any side the tongues stand shoulder to shoulder,
            # tallest in the middle, short ones licking up near the edge
            qn = sqrt((k + 0.3) / nfl)
            q = qn * min(0.62, 0.42 * fs / rf)           # (a flame is ~fs / 9 wide: keep neighbours touching)
            x, y, _ = _pol(a0 + k * 2.39996 + rfl.jitter(0, 0.25), rf * q, 0)
            sf = (0.64 - 0.24 * qn) * rfl.u(0.88, 1.12)
            fx("flame" if k == 0 else f"flame{k + 1}", "flame", (x, y, S(q) + zfl), size=round(fs * sf, 3), color="#e06c22")
        fx("smoke", "smoke", (0, 0, ztop + 0.42), size=round(D * 0.45, 2), rate=round(0.2 + (0.1 if logs else 0.0), 2), color="#2b2622")
    else:
        # dying embers: no flame, a faint glow over the ash-covered heap and a thin wisp of smoke
        light("emberglow", (0, 0, ztop + 0.14), color="#ff6a34", intensity=I * 0.007, range=1.0, flicker=0.3)
        fx("smoke", "smoke", (0, 0, ztop + 0.15), size=round(D * 0.35, 2), rate=0.25, color="#3a3632")
    socket("warm_spot", "attach", (0, -(Rf + 0.45), 0), role="stand")
