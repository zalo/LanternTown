"""Granite kerb and sett gutter run. Individually cut, fine-axed grey granite kerb stones (slightly battered
face, foot-polished top and worn bull nose, knocked arrises, a dirt / splash line along the foot) are bedded
with narrow lime-mortar joints that are recessed a few millimetres - the odd one washed out on an old run.
Beside them lies a three-row channel of pillowed setts in running bond with sand-filled joints, laid flush
with the paving on the road side and dished towards the kerb, where water, silt, moss and leaves collect.
A hinged cast-iron gully grate (frame, slotted lid, raised foundry lettering) sits in the two rows next to
the kerb and a cast-iron rainwater outlet box is let into the face of one kerb stone, with a rust stain
below it. Everything is modelled in a straight (s, u, z) space and mapped onto the run's centre line, so the
same generator gives a straight run (x in [-length/2, +length/2], pavement side +Y, road side -Y) or a
quarter-circle corner, an optional dropped cart crossing and an optional flagstone pavement edge.
"""
from tk import *
import math
import numpy as np

META = {"name": "Kerb and gutter run", "class": "ground", "mount": "ground", "focus": [0.3, -0.16, 0.03]}
PARAMS = {
    "length": {"default": 3.0, "min": 1.0, "max": 12.0, "desc": "run length along X (m); straight mode spans exactly x = -length/2 .. +length/2"},
    "kerb_height": {"default": 0.14, "min": 0.08, "max": 0.25, "desc": "kerb upstand above the road (m)"},
    "kerb_width": {"default": 0.2, "min": 0.14, "max": 0.32, "desc": "kerb stone width, front to back (m)"},
    "stone_length": {"default": 0.85, "min": 0.5, "max": 1.4, "desc": "nominal kerb stone length (m); every stone is cut a little different"},
    "gutter_width": {"default": 0.36, "min": 0.28, "max": 0.5, "desc": "width of the three-row sett channel (m)"},
    "curve_radius": {"default": 0.0, "min": 0.0, "max": 8.0, "desc": "0 = straight; otherwise a quarter circle of this centre-line radius (>= 1.2 m), length is then pi/2 * radius"},
    "corner": {"default": "outer", "choices": ["outer", "inner"], "desc": "curved mode: outer = pavement inside the bend, inner = road inside the bend"},
    "grate": {"default": True, "desc": "cast-iron gully grate in the channel"},
    "grate_pos": {"default": 0.6, "min": 0.1, "max": 0.9, "desc": "gully position along the run (fraction)"},
    "casting_text": {"default": "TW 1887", "desc": "foundry lettering cast on the gully lid ('' = none)"},
    "dropped": {"default": 0.0, "min": 0.0, "max": 4.0, "desc": "width of a dropped cart crossing (m, 0 = none; needs a run of 2.5 m or more)"},
    "drop_pos": {"default": 0.35, "min": 0.1, "max": 0.9, "desc": "centre of the dropped crossing along the run (fraction)"},
    "pavement": {"default": 0.0, "min": 0.0, "max": 1.2, "desc": "width of a flagstone pavement edge behind the kerb (m, 0 = none)"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age: rounding of the nose, chips, misalignment, washed-out joints"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "moss band along the kerb foot and in the joints"},
    "wet": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "wetness: gloss of the setts and water standing in the settled channel"},
    "outlet": {"default": True, "desc": "cast-iron rainwater outlet box let into the kerb face (pavement / downpipe drain)"},
    "outlet_pos": {"default": 0.24, "min": 0.05, "max": 0.95, "desc": "which kerb stone carries the outlet (fraction along the run)"},
    "leaves": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "fallen leaves collected along the kerb foot and at the gully"},
    "granite": {"default": "#a9a8a6", "desc": "kerb granite colour"},
    "sett": {"default": "#978b7c", "desc": "channel sett colour"},
    "leaf_tint": {"default": "#8f5f2a", "desc": "leaf colour"},
}
VARIANTS = {
    "corner_pavement": {"curve_radius": 2.0, "corner": "outer", "pavement": 0.7, "grate_pos": 0.3, "seed": 4},
    "cart_crossing": {"length": 6.0, "dropped": 1.7, "drop_pos": 0.42, "pavement": 0.6, "grate_pos": 0.86, "leaves": 0.3, "seed": 11},
    "tall_mossy": {"length": 3.0, "kerb_height": 0.23, "kerb_width": 0.28, "stone_length": 1.2, "gutter_width": 0.45, "moss": 1.0, "wet": 1.0,
                   "wear": 0.9, "granite": "#94928d", "sett": "#82796d", "grate_pos": 0.3, "leaves": 0.8, "seed": 23},
    "inner_bend_dry": {"curve_radius": 3.2, "corner": "inner", "grate": False, "kerb_height": 0.1, "stone_length": 0.6, "gutter_width": 0.3,
                       "wet": 0.0, "moss": 0.1, "leaves": 0.0, "wear": 0.2, "granite": "#b9aea6", "sett": "#9c958b", "outlet": False, "seed": 31},
}

JOINT = 0.008                                       # sand joint between setts (at the foot; ~12 mm at the rim)


def _noise(seed, octaves=3):
    """Cheap smooth pseudo noise (sum of sines) -> roughly -1..1; fn(points (N,3), feature size m)."""
    r = np.random.RandomState(int(seed) & 0x7fffffff)
    D = r.normal(size=(octaves * 3, 3))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    PH = r.uniform(0.0, 6.283, octaves * 3)

    def f(P, scale):
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        out = np.zeros(len(P))
        amp, tot = 1.0, 0.0
        for o in range(octaves):
            for k in range(3):
                i = o * 3 + k
                out += amp * np.sin((P @ D[i]) * (6.283 / scale) * (2 ** o) + PH[i])
            tot += amp * 1.8
            amp *= 0.5
        return out / tot
    return f


class _Path:
    """Centre line of the run. Flat space (s along, u across from the kerb face, z up) -> asset space."""

    def __init__(self, L, R, sig, tf):
        self.L, self.R, self.sig, self.tf = L, R, sig, tf

    def k(self, u):
        return 1.0 if self.R == 0 else (self.R - self.sig * (u + self.tf)) / self.R

    def map(self, P):
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        s, t, z = P[:, 0], P[:, 1] + self.tf, P[:, 2]
        if self.R == 0:
            return np.column_stack([s - 0.5 * self.L, t, z])
        R, sg = self.R, self.sig
        th = s / R
        x = -0.5 * R + R * np.sin(th) - sg * np.sin(th) * t
        y = sg * (-0.5 * R + R * (1.0 - np.cos(th))) + np.cos(th) * t
        return np.column_stack([x, y, z])

    def pt(self, s, u, z):
        return tuple(float(v) for v in self.map([[s, u, z]])[0])


class _Acc:
    """Accumulates many small pieces (flat space) into one mesh with per-piece UVs and vertex masks."""

    def __init__(self, name, seed=0):
        self.name = name
        self.V, self.F, self.UV, self.M, self.n = [], [], [], [], 0
        self.r = np.random.RandomState((int(seed) * 7919 + len(name) * 104729) & 0x7fffffff)

    def add(self, verts, faces, g=0.0, w=0.0, sp=0.0, uvs=None, us=1.0, off=None):
        V = np.asarray(verts, dtype=float).reshape(-1, 3)
        VU = V * np.array([us, 1.0, 1.0])                  # us = arc-length scale on a bend (keeps texels square)
        n = len(V)
        M = np.zeros((n, 3))
        M[:, 0], M[:, 1], M[:, 2] = g, w, sp
        if off is None:
            off = self.r.uniform(0.0, 4.0, 2)
        for fi, f in enumerate(faces):
            self.F.append(tuple(self.n + int(i) for i in f))
            if uvs is not None:
                self.UV.append(np.asarray(uvs[fi], dtype=float))
                continue
            Pf = VU[list(f)]
            nrm = np.zeros(3)
            for i in range(len(Pf)):                       # Newell normal (robust for n-gons)
                a, b = Pf[i], Pf[(i + 1) % len(Pf)]
                nrm[0] += (a[1] - b[1]) * (a[2] + b[2])
                nrm[1] += (a[2] - b[2]) * (a[0] + b[0])
                nrm[2] += (a[0] - b[0]) * (a[1] + b[1])
            an = np.abs(nrm)
            if an[2] >= an[0] and an[2] >= an[1]:
                uv = Pf[:, [0, 1]]
            elif an[1] >= an[0]:
                uv = Pf[:, [0, 2]]
            else:
                uv = Pf[:, [1, 2]]
            self.UV.append(uv + off)
        self.V.append(V)
        self.M.append(M)
        self.n += n

    def build(self, path, material, angle=50.0):
        if self.n == 0:
            return None
        V = np.vstack(self.V)
        M = np.clip(np.vstack(self.M), 0.0, 1.0)
        W = path.map(V)
        obj = mesh_object(self.name, W.tolist(), self.F)
        me = obj.data
        uv = np.vstack(self.UV)
        if len(me.loops) == len(uv):
            layer = me.uv_layers.new(name="UVMap")
            layer.data.foreach_set("uv", uv.astype(np.float32).ravel())
            uv_keep(obj)
        else:
            print("DBG uv fallback", self.name, len(me.loops), len(uv))
        assign(obj, material)

        def col(i):
            def fn(co, nrm):
                if len(co) == len(M):
                    return M[:, i]
                from mathutils import kdtree
                kd = kdtree.KDTree(len(W))
                for j, q in enumerate(W):
                    kd.insert(q, j)
                kd.balance()
                return np.array([M[kd.find(c)[1], i] for c in co])
            return fn
        paint(obj, grime=col(0), wear=col(1), special=col(2))
        if angle:
            smooth(obj, angle)
        else:
            flat(obj)
        self.tris = sum(len(f) - 2 for f in self.F)
        return obj


def _loft(acc, rings, cap0=True, cap1=True, g=0.0, w=0.0, sp=0.0, us=1.0, off=None):
    """rings: equal-length lists of (s,u,z), profile ordered front-bottom -> over the top -> back-bottom."""
    m = len(rings[0])
    V = np.vstack([np.asarray(r, dtype=float) for r in rings])
    F = []
    for j in range(len(rings) - 1):
        for i in range(m - 1):
            a = j * m + i
            F.append((a, a + m, a + m + 1, a + 1))
    if cap0:
        F.append(tuple(range(m)))
    if cap1:
        b = (len(rings) - 1) * m
        F.append(tuple(range(b + m - 1, b - 1, -1)))
    acc.add(V, F, g, w, sp, us=us, off=off)


def _sett(acc, s0, s1, u0, u1, top, crown, ins, cut, ks, g, w, sp, deep=0.010, shear=0.0, rough=0.0):
    """Pillowed sett: octagonal footprint, shoulder rolled up to the rim, domed foot-polished crown (38 tris)."""
    wmin = min(u1 - u0, (s1 - s0) * ks)
    cut = min(cut, 0.3 * wmin)
    ins = min(ins, 0.2 * wmin)

    def octa(i, c):
        a0, a1, b0, b1 = s0 + i / ks, s1 - i / ks, u0 + i, u1 - i
        cs = c / ks
        return np.array([(a0 + cs, b0), (a1 - cs, b0), (a1, b0 + c), (a1, b1 - c), (a1 - cs, b1), (a0 + cs, b1), (a0, b1 - c), (a0, b0 + c)])
    A, C, D = octa(0.0, cut), octa(ins, cut * 0.8), octa(ins + 0.2 * wmin, cut * 0.5)
    zc = top(C[:, 0], C[:, 1])
    zd = top(D[:, 0], D[:, 1]) + crown
    za = np.maximum(top(A[:, 0], A[:, 1]) - deep, 0.0008)
    V = np.vstack([np.column_stack([A, za]), np.column_stack([C, zc]), np.column_stack([D, zd])])
    if rough:
        j = acc.r.uniform(-rough, rough, (8, 2))
        j[:, 0] /= ks
        for k in range(3):
            V[k * 8:(k + 1) * 8, :2] += j * (1.0 if k < 2 else 0.6)
        if not shear:
            V[:, 0] = np.clip(V[:, 0], s0, s1)
    if shear:
        V[:, 0] += shear * (V[:, 1] - 0.5 * (u0 + u1))
    F = []
    for k in range(2):
        for i in range(8):
            j = (i + 1) % 8
            F.append((k * 8 + i, k * 8 + j, (k + 1) * 8 + j, (k + 1) * 8 + i))
    F.append(tuple(range(16, 24)))
    acc.add(V, F, np.repeat(np.asarray(g, float), 8), np.repeat(np.asarray(w, float), 8), np.repeat(np.asarray(sp, float), 8), us=ks)


def _bar(acc, s0, s1, u0, u1, zb, top, ch=0.002, g=(1.0, 0.5, 0.0), w=(0.0, 0.15, 0.95), sp=(0.7, 0.3, 0.0)):
    """Cast-iron bar / frame member lying flat: rectangular plan, chamfered top arris (18 tris)."""
    ch = min(ch, 0.3 * (u1 - u0), 0.3 * (s1 - s0))

    def rect(i):
        return np.array([(s0 + i, u0 + i), (s1 - i, u0 + i), (s1 - i, u1 - i), (s0 + i, u1 - i)])
    A, C = rect(0.0), rect(ch)
    V = np.vstack([np.column_stack([A, np.full(4, zb)]), np.column_stack([A, top(A[:, 0], A[:, 1]) - ch]), np.column_stack([C, top(C[:, 0], C[:, 1])])])
    F = []
    for k in range(2):
        for i in range(4):
            j = (i + 1) % 4
            F.append((k * 4 + i, k * 4 + j, (k + 1) * 4 + j, (k + 1) * 4 + i))
    F.append((8, 9, 10, 11))
    acc.add(V, F, np.repeat(np.asarray(g, float), 4), np.repeat(np.asarray(w, float), 4), np.repeat(np.asarray(sp, float), 4))


def _fbar(acc, s0, s1, z0, z1, ub, uf, ch=0.002, g=0.2, w=0.3, sp=0.3):
    """Cast-iron member standing on a vertical face (front towards -u), chamfered front arris."""
    def rect(i):
        return [(s0 + i, z0 + i), (s1 - i, z0 + i), (s1 - i, z1 - i), (s0 + i, z1 - i)]
    V = [(s, ub, z) for s, z in rect(0.0)] + [(s, uf + ch, z) for s, z in rect(0.0)] + [(s, uf, z) for s, z in rect(ch)]
    F = []
    for k in range(2):
        for i in range(4):
            j = (i + 1) % 4
            F.append((k * 4 + i, k * 4 + j, (k + 1) * 4 + j, (k + 1) * 4 + i))
    F.append((8, 9, 10, 11))
    acc.add(V, F, np.repeat([0.9, g, g * 0.5], 4), np.repeat([0.0, w * 0.3, w], 4), np.repeat([0.9, sp, sp * 0.4], 4))


# inward direction of every kerb-profile point (u, z) - used to recess the joint mortar
_N_IN = np.array([(1, 0), (1, 0), (1, 0), (.866, -.5), (.5, -.866), (0, -1), (0, -1), (0, -1), (-.38, -.92), (-.92, -.38), (-1, 0), (-1, 0)], dtype=float)


def _kerb_profile(kw, h, batter, r, cb, z1, d=0.0):
    """12-point kerb section (u, z): battered face with a splash-line vertex, worn bull nose, flat top, eased back arris."""
    h2 = h - d
    r = max(0.003, min(r - d * 0.5, 0.45 * h2))
    cb = min(cb, 0.3 * h2)

    def fu(z):
        return d + batter * (z / max(h, 1e-4))
    cz = h2 - r
    cu = fu(cz) + r
    za = min(z1, 0.6 * cz)
    pts = [(fu(0.0), 0.0), (fu(za), za)]
    for ang in (180.0, 150.0, 120.0, 90.0):
        pts.append((cu + r * math.cos(math.radians(ang)), cz + r * math.sin(math.radians(ang))))
    ue = kw - d - cb
    pts += [(cu + (ue - cu) * 0.33, h2), (cu + (ue - cu) * 0.66, h2), (ue, h2), (kw - d, h2 - cb), (kw - d, (h2 - cb) * 0.5), (kw - d, 0.0)]
    return np.array(pts)


def _split(A, B, nom, r):
    n = max(1, int(round((B - A) / nom)))
    wts = [r.u(0.72, 1.3) for _ in range(n)]
    tot = sum(wts)
    out, pos = [], A
    for i, wt in enumerate(wts):
        nxt = B if i == n - 1 else pos + (B - A) * wt / tot
        out.append((pos, nxt))
        pos = nxt
    return out


def _fill(A, B, nom, first, r):
    """Lay setts 0.84-1.2 x nominal long from A to B (first one `first` x nominal for the running bond)."""
    out, pos = [], A
    if B - A < 0.05:
        return out
    ln = nom * first
    while True:
        rem = B - pos
        if rem <= 1.25 * nom:
            out.append((pos, B))
            return out
        if rem <= 2.3 * nom:
            mid = pos + rem * (0.5 + r.u(-0.06, 0.06))
            out += [(pos, mid), (mid, B)]
            return out
        out.append((pos, pos + ln))
        pos += ln
        ln = nom * r.u(0.84, 1.18)


def build(p, rng):
    seed = int(p.seed)
    R = 0.0 if p.curve_radius < 0.05 else max(1.2, float(p.curve_radius))
    sig = 1.0 if p.corner == "outer" else -1.0
    L = float(p.length) if R == 0 else 0.5 * math.pi * R
    kw, gw, hk = float(p.kerb_width), float(p.gutter_width), float(p.kerb_height)
    tf = 0.5 * (gw - kw)                           # kerb face offset from the centre line
    path = _Path(L, R, sig, tf)
    wear, mossy, wet = float(p.wear), float(p.moss), float(p.wet)
    pav = float(p.pavement)
    if R > 0 and sig > 0:
        pav = min(pav, R - tf - kw - 0.3)
    if pav < 0.15:
        pav = 0.0
    nz_a, nz_b, nz_c = _noise(seed * 31 + 1), _noise(seed * 31 + 2), _noise(seed * 31 + 3)
    ring_step = 0.15 if R == 0 else min(0.15, 0.06 * R)

    def n1(nz, s, ph, scale):                      # 1D noise along the run
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return nz(np.column_stack([s, np.full(len(s), ph), np.zeros(len(s))]), scale)

    # ---------------------------------------------------------------- dropped crossing: kerb top H(s)
    TR, H_LOW = 0.75, 0.034
    drop = None
    if p.dropped >= 0.4 and L >= 2 * TR + 0.4 + 0.4:
        wd = min(float(p.dropped), L - 2 * TR - 0.4)
        sd = clamp(p.drop_pos * L, wd / 2 + TR + 0.2, L - wd / 2 - TR - 0.2)
        drop = [sd - wd / 2 - TR, sd - wd / 2, sd + wd / 2, sd + wd / 2 + TR]

    def H(s):
        if drop is None:
            return hk + 0.0 * np.asarray(s, dtype=float)
        return np.interp(s, drop, [hk, min(hk, H_LOW), min(hk, H_LOW), hk])

    # ---------------------------------------------------------------- channel shape
    # The paving is z = 0 and nothing may go below it: the road-side sett row meets the paving (its shoulder
    # rolls down to z = 0, no bed skirt) and the channel falls from there towards the low line at the kerb.
    Z_K, Z_LOW, Z_OUT, X_LOW = 0.0075, 0.006, 0.015, 0.15
    rw = (gw - 0.008) / 3.0                        # row pitch
    u_low = -X_LOW * gw

    def dish(u):
        x = np.clip(-np.asarray(u, dtype=float) / gw, 0.0, 1.0)
        return np.where(x < X_LOW, Z_LOW + (Z_K - Z_LOW) * (np.clip(X_LOW - x, 0, 1) / X_LOW) ** 2,
                        Z_LOW + (Z_OUT - Z_LOW) * (np.clip(x - X_LOW, 0, 1) / (1.0 - X_LOW)) ** 1.3)

    # gully
    GL = 0.45
    has_grate = bool(p.grate) and L >= GL + 0.5
    sg = clamp(p.grate_pos * L, GL / 2 + 0.2, L - GL / 2 - 0.2) if has_grate else -99.0
    g0, g1 = sg - GL / 2, sg + GL / 2
    ua_g, ub_g = -0.008 - 2 * rw + JOINT / 2, -0.010
    zga, zgb = float(dish(ua_g)) - 0.0005, float(dish(ub_g)) + 0.0005

    def gt(s, u):                                  # top of the gully frame (follows the cross-fall)
        return zga + (zgb - zga) * (np.asarray(u, dtype=float) - ua_g) / (ub_g - ua_g) + 0.0 * np.asarray(s, dtype=float)

    rs = rng.fork("settle")
    sags = []
    for i in range(max(1, int(round(L / 3.2)))):
        for _ in range(10):
            c = rs.u(0.12, 0.88) * L
            if abs(c - sg) > 0.95 and all(abs(c - q[0]) > 1.2 for q in sags):
                sags.append((c, rs.u(0.0032, 0.0045), rs.u(0.28, 0.5)))
                break

    def sag(s):
        s = np.asarray(s, dtype=float)
        out = np.zeros_like(s)
        for c, dpt, ln in sags:
            out = out + dpt * np.exp(-((s - c) / ln) ** 2)
        return out

    def silt(s, u):                                # 0..1: where fines wash together (low line, settlements, gully)
        s, u = np.asarray(s, dtype=float), np.asarray(u, dtype=float)
        uw = np.exp(-((u - u_low) / (0.3 * gw)) ** 2)
        a = 0.3 + 0.0 * s
        for c, dpt, ln in sags:
            a = a + 0.7 * np.exp(-((s - c) / (1.2 * ln)) ** 2)
        if has_grate:
            a = a + 0.8 * np.exp(-((s - sg) / 0.55) ** 2)
        return np.clip(a * uw, 0.0, 1.0)

    # ---------------------------------------------------------------- kerb stones
    n_kacc = 1
    kerbs = [_Acc("kerb_stones%d" % i, seed) for i in range(n_kacc)]
    fill = _Acc("joint_fill", seed)                # lime mortar in the kerb joints + sand bed between the setts
    iron = _Acc("cast_iron", seed)
    rk = rng.fork("kerb")
    brk = sorted(set([0.0, L] + (drop or [])))
    stones = []
    for A, B in zip(brk[:-1], brk[1:]):
        ramp = drop is not None and (abs(A - drop[0]) < 1e-6 or abs(A - drop[2]) < 1e-6)
        stones += [(A, B)] if ramp else _split(A, B, p.stone_length / path.k(kw / 2), rk)
    jk = 0.005 + 0.004 * wear                      # joint width 5-9 mm
    r_nose = 0.010 + 0.022 * wear
    batter = 0.010
    W_W = np.array([0, .04, .3, .75, .9, .85, .7, .7, .6, .3, 0, 0]) * (0.55 + 0.45 * wear)
    W_G = np.array([.8, .45, .1, .03, .02, .03, .05, .06, .08, .1, .18, .5])
    W_M = np.array([1.0, .5, .02, 0, 0, 0, .03, .05, .08, .2, .6, 1.0])
    NI = [2, 3, 4, 5]
    # rainwater outlet: a real opening cut into the middle of one full-height stone
    fbr, ow = 0.011, 0.12
    outlet = None
    if p.outlet:
        cz_min = hk - r_nose * 1.25 - (0.002 + 0.006 * wear)
        zo1 = clamp(hk * 0.22, 0.022, 0.04) + fbr
        oh = min(0.045, cz_min - 0.006 - fbr - zo1)
        cands = [i for i, (a, b) in enumerate(stones) if b - a > ow + 2 * fbr + 0.14 and float(H(a)) > hk - 1e-4 and float(H(b)) > hk - 1e-4]
        if cands and oh >= 0.018:
            oi = min(cands, key=lambda i: abs(0.5 * (stones[i][0] + stones[i][1]) - p.outlet_pos * L))
            outlet = {"i": oi, "s": 0.5 * (stones[oi][0] + stones[oi][1]), "z1": zo1, "z2": zo1 + oh}
    Rs = []
    c = 0.003
    for si, (a, b) in enumerate(stones):
        acc = kerbs[si % n_kacc]
        a2, b2 = a + jk / 2, b - jk / 2
        sc = 0.5 * (a + b)
        is_out = outlet is not None and outlet["i"] == si
        n = max(2, int(round((b2 - a2) / ring_step)))
        inner = list(np.linspace(a2 + c, b2 - c, n + 1))
        z1 = 0.042
        if is_out:
            sL, sR = outlet["s"] - ow / 2, outlet["s"] + ow / 2
            inner = sorted([s for s in inner if abs(s - sL) > 0.02 and abs(s - sR) > 0.02] + [sL, sR])
            z1 = outlet["z1"]
        ss = [a2] + inner + [b2]
        ins = [c] + [0.0] * len(inner) + [c]
        du = rk.u(-0.001, 0.001 + 0.007 * wear)
        yaw = rk.u(-1, 1) * 0.006 * wear
        dz = rk.u(-1, 1) * (0.002 + 0.006 * wear)
        pitch = rk.u(-1, 1) * 0.008 * wear
        rr = r_nose * rk.u(0.75, 1.25)
        rings = []
        for s, d in zip(ss, ins):
            h = max(0.02, float(H(s)) + dz + pitch * (s - sc))
            pr = _kerb_profile(kw, h, batter, rr, 0.006, z1, d)
            rings.append(np.column_stack([np.full(len(pr), s), pr[:, 0] + du + yaw * (s - sc), pr[:, 1]]))
        Rg = np.array(rings)                               # (nr, 12, 3)
        nr = len(rings)
        # knocked corners at the joints and the odd spall along the nose
        cw = np.array([0.5, 1.0, 1.0, 0.5])
        for end in (0, 1):
            if rk.chance(0.25 + 0.6 * wear):
                dp = rk.u(0.004, 0.014) * (0.5 + wear)
                idx = [0, 1, 2] if end == 0 else [nr - 1, nr - 2, nr - 3]
                for j, f in zip(idx, (1.0, 1.0, 0.45)):
                    Rg[j, NI, 1] += dp * f * cw * 0.8
                    Rg[j, NI, 2] -= dp * f * cw
        for _ in range(int(rk.u(0, 1 + 2 * wear))):
            j = int(rk.u(2, nr - 2))
            dp = rk.u(0.002, 0.005)
            Rg[j, NI, 1] += dp * cw * 0.7
            Rg[j, NI, 2] -= dp * cw
        P = Rg.reshape(-1, 3)
        hrel = np.clip(P[:, 2] / max(1e-3, float(np.max(P[:, 2]))), 0.0, 1.0)
        amp = 0.001 + 0.002 * wear
        P[:, 2] += amp * nz_a(P + si * 3.7, 0.24) * hrel
        P[:, 1] += amp * nz_b(P + si * 5.1, 0.3)
        P[:, 1] = np.maximum(P[:, 1], -0.0035)
        Rg = P.reshape(nr, 12, 3)
        tone = rk.u(0.0, 0.14)
        nm = np.clip(0.55 + 1.1 * nz_c(P, 0.4), 0.0, 1.5)
        splash = np.clip(0.75 + 0.6 * nz_b(P * np.array([1.0, 0.0, 0.0]), 0.3), 0.3, 1.3)
        G = np.tile(W_G, nr) * np.where(np.tile(np.arange(12), nr) == 1, splash, 1.0) + tone
        Wm = np.tile(W_W, nr) * np.clip(0.9 + 0.4 * nz_a(P, 0.2), 0.6, 1.2) * rk.u(0.8, 1.0)
        Mm = np.minimum(np.tile(W_M, nr) * nm * mossy, 0.8)
        endm = np.zeros(nr)
        endm[[0, 1, nr - 2, nr - 1]] = 1.0
        endv = np.repeat(endm, 12)
        G += 0.12 * endv
        Mm += 0.4 * mossy * endv * nm * (1.0 - hrel)
        off = acc.r.uniform(0.0, 4.0, 2)
        Rs.append(Rg.copy())
        if not is_out:
            _loft(acc, list(Rg), True, True, G, Wm, Mm, off=off)
            continue
        # outlet stone: left part, right part, and between them the face split around the opening
        so = outlet["s"]
        G += 0.6 * np.exp(-((P[:, 0] - so) / 0.075) ** 2) * (P[:, 2] < outlet["z1"] + 0.004) * (P[:, 1] < 0.03)   # damp stain below the box
        G2, W2, M2 = G.reshape(nr, 12), Wm.reshape(nr, 12), Mm.reshape(nr, 12)
        iL, iR = ss.index(sL), ss.index(sR)
        mid = Rg[iL:iR + 1]
        t = np.clip((outlet["z2"] - mid[:, 1, 2]) / np.maximum(mid[:, 2, 2] - mid[:, 1, 2], 1e-4), 0.05, 0.95)[:, None]
        Dp = mid[:, 1, :] + t * (mid[:, 2, :] - mid[:, 1, :])
        up = np.concatenate([Dp[:, None, :], mid[:, 2:, :]], axis=1)

        def upm(Mx):
            m_ = Mx[iL:iR + 1]
            return np.concatenate([m_[:, 1:2] + t * (m_[:, 2:3] - m_[:, 1:2]), m_[:, 2:]], axis=1).ravel()
        _loft(acc, list(Rg[:iL + 1]), True, False, G2[:iL + 1].ravel(), W2[:iL + 1].ravel(), M2[:iL + 1].ravel(), off=off)
        _loft(acc, list(Rg[iR:]), False, True, G2[iR:].ravel(), W2[iR:].ravel(), M2[iR:].ravel(), off=off)
        _loft(acc, list(up), False, False, upm(G2), upm(W2), upm(M2), off=off)
        _loft(acc, list(mid[:, 0:2, :]), False, False, G2[iL:iR + 1, 0:2].ravel(), W2[iL:iR + 1, 0:2].ravel(), M2[iL:iR + 1, 0:2].ravel(), off=off)
        ufa = np.concatenate([mid[:, 1, 1], Dp[:, 1]])
        outlet["uf"] = (float(ufa.min()), float(ufa.max()))
        outlet["foot"] = float(mid[:, 0, 1].min())

    # lime mortar pointing: a short plug per joint, recessed 3-6 mm behind the more worn of the two arrises;
    # on an old run the odd joint has washed out and stands open
    rj = rng.fork("joints")
    n_open = 0

    def plug(sa, sb, ringA, ringB, cap0, cap1):
        nonlocal n_open
        dA, dB = (ringA[:, 1:] * _N_IN).sum(1), (ringB[:, 1:] * _N_IN).sum(1)
        base = np.where((dA > dB)[:, None], ringA[:, 1:], ringB[:, 1:])
        hh = float(base[:, 1].max())
        opened = rj.chance(0.45 * max(0.0, wear - 0.35) ** 1.2)
        rec = min(rj.u(0.028, 0.045), 0.35 * hh) if opened else rj.u(0.003, 0.006)
        n_open += int(opened)
        pr = base + rec * _N_IN
        pr[:, 1] = np.maximum(pr[:, 1], 0.002)
        pr[[0, 11], 1] = 0.0
        sm = 0.5 * (sa + sb)
        ms = float(np.clip(0.45 + 1.3 * n1(nz_c, sm, 0.7, 0.6)[0], 0.0, 1.0)) * mossy
        hr = np.clip(pr[:, 1] / max(hh, 1e-3), 0.0, 1.0)
        gg = (0.85 if opened else 0.1) + 0.5 * (1.0 - hr) ** 3
        _loft(fill, [np.column_stack([np.full(12, s), pr]) for s in (sa, sb)], cap0, cap1, np.tile(gg, 2), 0.0 if opened else 0.6, np.tile(np.clip(ms * (1.25 - hr), 0.0, 0.9), 2))
    for i in range(len(stones) - 1):
        b = stones[i][1]
        plug(b - jk / 2 - 0.005, b + jk / 2 + 0.005, Rs[i][-2], Rs[i + 1][1], False, False)
    plug(0.0, jk / 2 + 0.005, Rs[0][1], Rs[0][1], True, False)
    plug(L - jk / 2 - 0.005, L, Rs[-1][-2], Rs[-1][-2], False, True)

    # ---------------------------------------------------------------- sett channel
    n_sacc = 1
    setts = [_Acc("setts%d" % i, seed) for i in range(n_sacc)]
    rt = rng.fork("setts")
    nom = 0.17
    n_setts = 0
    so_ = outlet["s"] if outlet is not None else -99.0
    for r in range(3):
        top_u = -0.008 - r * rw
        bot_u = -0.008 - (r + 1) * rw
        u0 = bot_u + (JOINT / 2 if r < 2 else 0.0)
        u1 = top_u - (JOINT / 2 if r > 0 else 0.0)
        uc = 0.5 * (u0 + u1)
        ks = path.k(uc)
        segs = [(0.0, L)]
        if has_grate and r < 2:
            segs = [(0.0, g0), (g1, L)]
        first = (1.0, 0.5, 0.75)[r] * rt.u(0.9, 1.1)
        for (A, B) in segs:
            for (a, b) in _fill(A, B, nom / ks, first, rt):
                s0, s1 = a + JOINT / 2 / ks, b - JOINT / 2 / ks
                if s1 - s0 < 0.02:
                    continue
                scn = 0.5 * (s0 + s1)
                off = rt.u(-1, 1) * (0.0008 + 0.002 * wear)
                ts, tu = rt.u(-1, 1) * 0.012 * (0.4 + wear), rt.u(-1, 1) * 0.02 * (0.4 + wear)
                odd = rt.u()
                if odd < 0.05 + 0.07 * wear:
                    off -= rt.u(0.003, 0.0055)                # a sunk sett
                elif odd < 0.09 + 0.15 * wear:
                    ts, tu = rt.u(-1, 1) * 0.045, rt.u(-1, 1) * 0.06   # a rocked one
                du0, du1 = rt.u(-0.002, 0.002), rt.u(-0.002, 0.002)

                def top(s, u, off=off, ts=ts, tu=tu, scn=scn, uc=uc):
                    return np.maximum(dish(u) - sag(s) + off + ts * (s - scn) * ks + tu * (u - uc), 0.0018)
                sl = float(silt(scn, uc))
                stain = math.exp(-((scn - so_) / 0.09) ** 2) * (0.6 if r == 0 else 0.3 if r == 1 else 0.0)
                tone = rt.u(0.0, 0.22)
                pol = (0.45 + 0.55 * wear) * rt.u(0.65, 1.0)
                ms = mossy * clamp(0.35 + 1.2 * float(nz_c([[scn, uc, 0.0]], 0.5)[0]) + (0.4 if r == 0 else 0.0), 0.0, 1.0)
                edge = a < 1e-6 or b > L - 1e-6
                acc = setts[0]
                _sett(acc, s0, s1, u0 + (du0 if r < 2 else 0.0), u1 + (du1 if r > 0 else 0.0), top,
                      crown=rt.u(0.003, 0.005), ins=rt.u(0.006, 0.0085), cut=rt.u(0.007, 0.013), ks=ks,
                      g=(1.0, 0.3 + tone + 0.35 * sl + stain, 0.04 + 0.6 * tone + 0.12 * sl + 0.6 * stain),
                      w=(0.0, 0.2 * pol, 0.65 * pol), sp=(ms, ms * 0.3, 0.0),
                      deep=1.0 if (r == 2 or edge) else 0.011,
                      shear=0.0 if edge else rt.u(-1, 1) * 0.02, rough=0.0012 + 0.0016 * wear)
                n_setts += 1
    # sand in the joints: one sheet under the setts, silted up along the low line; on the road side it runs
    # out to z = 0 as the joint to the paving (no skirt), inside the gully frame it drops to the pit
    us = [-gw - 0.006, -gw - 0.001, -0.62 * gw, -0.34 * gw, -0.14 * gw, 0.0, 0.014]
    if has_grate:
        us += [ua_g, ua_g + 0.012, ub_g - 0.012]
    us = sorted(us)
    us = [u for i, u in enumerate(us) if i == 0 or u - us[i - 1] > 0.003]
    sb = list(np.linspace(0.0, L, max(2, int(L / 0.3) + 1)))
    if has_grate:
        sb = [s for s in sb if min(abs(s - q) for q in (g0, g0 + 0.012, g1 - 0.012, g1)) > 0.03] + [g0, g0 + 0.012, g1 - 0.012, g1]
    sb = sorted(sb)
    SB, UB = np.meshgrid(np.array(sb), np.array(us), indexing="ij")
    SL = silt(SB, UB) * np.clip(0.75 + 0.5 * nz_a(np.column_stack([SB.ravel(), UB.ravel(), np.zeros(SB.size)]), 0.35), 0.3, 1.0).reshape(SB.shape)
    ZB = np.maximum(dish(UB) - sag(SB) - (0.0065 - 0.005 * SL), 0.0012)
    ZB = np.where(UB < -gw - 0.01, 0.0005, ZB)
    if has_grate:
        inside = (SB > g0 + 0.006) & (SB < g1 - 0.006) & (UB > ua_g + 0.006) & (UB < ub_g - 0.006)
        ZB = np.where(inside, 0.0006, ZB)
    nsb, nub = len(sb), len(us)
    Vb = np.column_stack([SB.ravel(), UB.ravel(), ZB.ravel()])
    Fb = []
    for j in range(nsb - 1):
        for i in range(nub - 1):
            a = j * nub + i
            Fb.append((a, a + nub, a + nub + 1, a + 1))
    mb = np.clip(0.3 + 1.3 * nz_c(Vb, 0.45), 0.0, 1.0) * mossy * (0.2 + 0.8 * np.exp(np.minimum(Vb[:, 1], 0.0) / 0.07))
    gb = 0.62 + 0.3 * SL.ravel() + 0.4 * np.exp(-((Vb[:, 0] - so_) / 0.09) ** 2) * (Vb[:, 1] > -2 * rw)
    fill.add(Vb, Fb, gb, 0.0, mb)

    # ---------------------------------------------------------------- gully grate
    txt_obj = None
    if has_grate:
        def lid(s, u):
            return gt(s, u) - 0.0012

        def up(fn, dz):
            return lambda s, u: fn(s, u) + dz
        zb = 0.0014
        fb = 0.024
        FR = dict(g=(1.0, 0.5, 0.1), w=(0.0, 0.1, 0.6), sp=(0.8, 0.45, 0.08))
        _bar(iron, g0, g1, ua_g, ua_g + fb, 0.0008, gt, 0.0022, **FR)
        _bar(iron, g0, g1, ub_g - fb, ub_g, 0.0008, gt, 0.0022, **FR)
        _bar(iron, g0, g0 + fb, ua_g + fb, ub_g - fb, 0.0008, gt, 0.0022, **FR)
        _bar(iron, g1 - fb, g1, ua_g + fb, ub_g - fb, 0.0008, gt, 0.0022, **FR)
        i0, i1, ia, ib = g0 + fb + 0.003, g1 - fb - 0.003, ua_g + fb + 0.003, ub_g - fb - 0.003
        rim = 0.012
        _bar(iron, i0, i1, ia, ia + rim, zb, lid)
        _bar(iron, i0, i1, ib - rim, ib, zb, lid)
        _bar(iron, i0, i0 + rim, ia + rim, ib - rim, zb, lid)
        _bar(iron, i1 - rim, i1, ia + rim, ib - rim, zb, lid)
        label = str(p.casting_text).strip()
        band = 0.052 if label and (ib - ia) > 0.15 else 0.0
        if band:
            _bar(iron, i0 + rim, i1 - rim, ia + rim, ia + rim + band, zb, lid, g=(1.0, 0.6, 0.35), w=(0.0, 0.05, 0.1), sp=(0.8, 0.5, 0.2))
        sa, sbb = i0 + rim, i1 - rim                      # slotted field
        fa, fbb = ia + rim + band, ib - rim
        fc = 0.5 * (fa + fbb)
        spine = 0.013
        _bar(iron, sa, sbb, fc - spine / 2, fc + spine / 2, zb, lid)
        bw = 0.013
        ns = max(2, int((sbb - sa + bw) / 0.036))
        slot = (sbb - sa - (ns - 1) * bw) / ns
        for i in range(1, ns):
            x0 = sa + i * slot + (i - 1) * bw
            _bar(iron, x0, x0 + bw, fa, fc - spine / 2, zb, lid, 0.0028)
            _bar(iron, x0, x0 + bw, fc + spine / 2, fbb, zb, lid, 0.0028)
        for hs in (-0.125, 0.125):                         # hinge knuckles on the kerb side, lifting lug opposite
            _bar(iron, sg + hs - 0.022, sg + hs + 0.022, ub_g - fb + 0.003, ub_g - 0.004, zb, up(gt, 0.0045), 0.003)
        _bar(iron, sg - 0.02, sg + 0.02, ua_g + 0.005, ua_g + fb - 0.004, zb, up(gt, 0.003))
        iron.add([(i0, ia, 0.0011), (i1, ia, 0.0011), (i1, ib, 0.0011), (i0, ib, 0.0011)], [(0, 1, 2, 3)], 1.0, 0.0, 0.5)   # the dark pit
        if band:
            txt_obj = text("gully_lettering", label, size=0.034, depth=0.0022, rot=(0, 0, 0), font="sans")
            xs = [v.co.x for v in txt_obj.data.vertices]
            ys = [v.co.y for v in txt_obj.data.vertices]
            wtx = (max(xs) - min(xs)) if xs else 0.0
            cx = 0.5 * (max(xs) + min(xs)) if xs else 0.0
            cy = 0.5 * (max(ys) + min(ys)) if ys else 0.0
            hty = (max(ys) - min(ys)) if ys else 1.0
            k = min(1.0, (i1 - i0 - 2 * rim - 0.03) / max(wtx, 1e-4), (band - 0.012) / max(hty, 1e-4))
            ucen = ia + rim + band / 2

            def tmap(co, k=k, cx=cx, cy=cy, ucen=ucen):
                s_, u_ = sg + (co.x - cx) * k, ucen + (co.y - cy) * k
                return Vector(path.pt(s_, u_, float(lid(s_, u_)) - 0.0003 + co.z))
            transform_verts(txt_obj, tmap)

    # ---------------------------------------------------------------- rainwater outlet box in the kerb face
    if outlet is not None:
        so, zo1, zo2 = outlet["s"], outlet["z1"], outlet["z2"]
        ufn, ufx = outlet["uf"]
        uF, uB = ufn - 0.004, ufx + 0.008                 # frame 4 mm proud of the stone
        x0, x1 = so - ow / 2, so + ow / 2
        _fbar(iron, x0 - fbr, x1 + fbr, zo1 - fbr, zo1 + 0.002, uB, uF, sp=0.75, g=0.4)
        _fbar(iron, x0 - fbr, x1 + fbr, zo2 - 0.002, zo2 + fbr, uB, uF, sp=0.3)
        _fbar(iron, x0 - fbr, x0 + 0.002, zo1 + 0.002, zo2 - 0.002, uB, uF, sp=0.45)
        _fbar(iron, x1 - 0.002, x1 + fbr, zo1 + 0.002, zo2 - 0.002, uB, uF, sp=0.45)
        _fbar(iron, so - 0.005, so + 0.005, zo1 + 0.002, zo2 - 0.002, uB, uF - 0.0 + 0.001, sp=0.45)
        ub_, uf_ = ufn + 0.038, ufn - 0.002               # the box behind it: 38 mm deep, dark
        z1_, z2_ = zo1 - 0.001, zo2 + 0.001
        Vx = [(x0, uf_, z1_), (x1, uf_, z1_), (x1, ub_, z1_), (x0, ub_, z1_), (x0, uf_, z2_), (x1, uf_, z2_), (x1, ub_, z2_), (x0, ub_, z2_)]
        Fx = [(3, 2, 6, 7), (0, 1, 2, 3), (4, 7, 6, 5), (0, 3, 7, 4), (1, 5, 6, 2)]
        iron.add(Vx, Fx, [0.75, 0.75, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0], 0.0, [0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2])
        # rust runs below the frame corners and the mullion, down the face to the setts
        ro = rng.fork("rust")
        zf = float(dish(0.0)) + 0.001
        ust = outlet["foot"] - 0.0012
        zt_ = zo1 - fbr + 0.002
        for sx in (x0 - fbr * 0.6, x0 + ow * ro.u(0.2, 0.4), so + ro.u(-0.006, 0.006), x1 - ow * ro.u(0.15, 0.35), x1 + fbr * 0.6):
            wst, fr = ro.u(0.005, 0.010), ro.u(0.6, 1.0)
            ue, ze = lerp(ufn - 0.0012, ust, fr), lerp(zt_, zf, fr)
            iron.add([(sx - wst / 2, ufn - 0.0012, zt_), (sx + ro.u(-0.003, 0.003), ue, ze), (sx + wst / 2, ufn - 0.0012, zt_)], [(0, 1, 2)], [0.5, 0.9, 0.5], 0.0, [0.85, 0.5, 0.85])

    # ---------------------------------------------------------------- flagstone pavement edge
    flags = _Acc("flags", seed)
    if pav > 0:
        rf = rng.fork("flags")
        nrow = max(1, int(round(pav / 0.42)))
        dep = (pav - 0.010) / nrow
        step = 0.3 if R == 0 else min(0.3, 0.12 * R)

        def Hp(s):
            return np.maximum(H(s) - 0.006, 0.026)
        FW = np.array([0, .1, .3, .5, .5, .3, .1, 0]) * (0.5 + 0.6 * wear)
        FG = np.array([.4, .3, .2, .07, .07, .2, .3, .4])
        FM = np.array([.9, .6, .2, 0, 0, .2, .6, .9])
        for r in range(nrow):
            u0 = kw + 0.010 + r * dep + (JOINT / 2 if r > 0 else 0.0)
            u1 = kw + 0.010 + (r + 1) * dep - (JOINT / 2 if r < nrow - 1 else 0.0)
            ks = path.k(0.5 * (u0 + u1))
            for A, B in zip(brk[:-1], brk[1:]):
                for (a, b) in _split(A, B, rf.u(0.45, 0.75) / ks, rf):
                    a2, b2 = a + JOINT / 2 / ks, b - JOINT / 2 / ks
                    cc = 0.005
                    n = max(1, int(math.ceil((b2 - a2) / step)))
                    ss = [a2] + list(np.linspace(a2 + cc, b2 - cc, n + 1)) + [b2]
                    ins = [cc] + [0.0] * (n + 1) + [cc]
                    dz, tl = rf.u(-1, 1) * (0.001 + 0.004 * wear), rf.u(-1, 1) * 0.006 * wear
                    rings = []
                    for s, d in zip(ss, ins):
                        h = float(Hp(s)) + dz + tl * (s - 0.5 * (a + b))
                        ce = 0.006
                        um = [u0 + d + ce + (u1 - u0 - 2 * d - 2 * ce) * t for t in (0.33, 0.66)]
                        rings.append([(s, u0 + d, 0.0), (s, u0 + d, h - d - ce), (s, u0 + d + ce, h - d), (s, um[0], h - d), (s, um[1], h - d),
                                      (s, u1 - d - ce, h - d), (s, u1 - d, h - d - ce), (s, u1 - d, 0.0)])
                    Pf = np.array(rings).reshape(-1, 3)
                    nrg = len(rings)
                    tone = rf.u(0.0, 0.14)
                    nm = np.clip(0.5 + 1.2 * nz_c(Pf, 0.5), 0.0, 1.4)
                    Pf[:, 2] += 0.002 * nz_a(Pf, 0.35) * (np.tile(np.array([0, 1, 1, 1, 1, 1, 1, 0.0]), nrg))
                    _loft(flags, list(Pf.reshape(nrg, 8, 3)), True, True, np.tile(FG, nrg) + tone, np.tile(FW, nrg), np.tile(FM, nrg) * nm * mossy, us=ks)
        cs2 = sorted(set([0.0, L] + (drop or []) + list(np.linspace(0.0, L, max(2, int(L / step) + 1)))))
        brings = []
        for s in cs2:
            zt = max(float(Hp(s)) - 0.010 - 0.004 * wear, 0.004)
            brings.append([(s, kw - 0.012, 0.0), (s, kw - 0.012, zt), (s, kw + pav - 0.006, zt), (s, kw + pav - 0.006, 0.0)])
        _loft(fill, brings, True, True, 0.35, 0.3, 0.6 * mossy)

    # ---------------------------------------------------------------- standing water in the settled channel
    waters = [_Acc("puddles%d" % i, seed) for i in range(1)]
    if wet > 0.2:
        for (cs_, dpt, ln) in sags:
            zw = Z_LOW + 0.0016 + 0.0016 * wet
            a_r, b_r = ln * (0.9 + 0.5 * wet), 0.42 * gw
            pts = [(cs_, u_low, zw)]
            n = 22
            for i in range(n):
                ang = 2 * math.pi * i / n
                q = 1.0 + 0.28 * float(nz_b([[math.cos(ang) * 2.0, math.sin(ang) * 2.0, cs_]], 1.3)[0])
                s = clamp(cs_ + a_r * q * math.cos(ang), 0.012, L - 0.012)
                u = clamp(u_low + b_r * q * math.sin(ang), -gw + 0.03, 0.007)
                pts.append((s, u, zw))
            F = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
            waters[0].add(pts, F)

    # ---------------------------------------------------------------- moss: bands in the damp joints, cushions at the stone joints
    tufts = _Acc("moss", seed)
    if mossy > 0.05:
        rm = rng.fork("moss")
        cov = 0.3 + 0.7 * mossy

        def band(prof, ph, coverage, skip=None, step=0.07):
            """prof(w01) -> [(u, z_rel), ...] ordered -u -> +u; a strip wherever the noise lets moss grow."""
            n = max(2, int(round(L / step)))
            S = np.linspace(0.0, L, n + 1)
            f = 0.65 * n1(nz_c, S, ph, 0.7) + 0.5 * n1(nz_b, S, ph + 3.0, 0.19)
            wv = np.clip((f + (coverage * 2.0 - 1.0) * 0.9) * 2.2, 0.0, 1.0)
            wv *= np.clip(0.75 + 0.45 * n1(nz_a, S, ph + 7.0, 0.11), 0.4, 1.0)
            if skip is not None:
                wv = np.where((S > skip[0]) & (S < skip[1]), 0.0, wv)
            run = []
            for i in range(n + 2):
                on = i <= n and (wv[i] > 0.03 or (i > 0 and wv[i - 1] > 0.03) or (i < n and wv[i + 1] > 0.03))
                if on:
                    run.append(i)
                    continue
                if len(run) >= 2:
                    rings = []
                    for j in run:
                        base = float(-sag(S[j]))
                        rings.append([(S[j], u, base + z) for (u, z) in prof(max(float(wv[j]), 0.05), S[j])])
                    _loft(tufts, rings, True, True)
                run = []
        zk = float(dish(0.0))
        wk, hk_m = 0.012 + 0.03 * mossy, 0.007 + 0.011 * mossy
        gsk = (g0 - 0.01, g1 + 0.01) if has_grate else None
        # along the kerb foot, climbing a little up the face
        band(lambda w, s: [(-0.004 - wk * w, zk - 0.004), (-0.004 - 0.45 * wk * w, zk - 0.003 + hk_m * w), (0.0075, zk - 0.002 + hk_m * (0.7 + 0.9 * w))], 0.3, cov, gsk)
        if mossy > 0.3:                                    # in the joint between the first and the second row
            uj, zj = -0.008 - rw, float(dish(-0.008 - rw))
            wj = 0.009 + 0.008 * mossy
            band(lambda w, s: [(uj - wj * w, zj - 0.004), (uj, zj - 0.002 + 0.006 * w), (uj + wj * w, zj - 0.004)], 1.7, cov - 0.3, gsk)
        if pav == 0:                                       # behind the kerb, against the paving
            band(lambda w, s: [(kw - 0.008, 0.004 + hk_m * (0.5 + 0.6 * w)), (kw + 0.45 * wk * w, 0.002 + hk_m * 0.8 * w), (kw + 0.004 + wk * w, 0.0006)], 4.1, cov - 0.1)

        def tuft(s, u, z, rs_, ru_, h):
            V, F = [], []
            for k_, (q, zz) in enumerate(((1.0, 0.0), (0.62, 0.72))):
                for i in range(6):
                    ang = math.radians(i * 60 + rm.u(-12, 12))
                    V.append((clamp(s + rs_ * q * math.cos(ang), 0.002, L - 0.002), u + ru_ * q * math.sin(ang) * rm.u(0.8, 1.2), z + h * zz * rm.u(0.8, 1.15)))
            V.append((clamp(s, 0.002, L - 0.002), u, z + h))
            for i in range(6):
                j = (i + 1) % 6
                F.append((i, j, 6 + j, 6 + i))
                F.append((6 + i, 6 + j, 12))
            tufts.add(V, F)
        for (_, b) in stones[:-1]:                         # cushions where the kerb joints meet the channel
            if rm.chance(0.2 + 0.8 * mossy) and not (has_grate and g0 - 0.03 < b < g1 + 0.03):
                tuft(b + rm.u(-0.015, 0.015), -0.006, zk - float(sag(b)) - 0.004, rm.u(0.025, 0.05), rm.u(0.009, 0.015), rm.u(0.012, 0.022) * (0.7 + 0.5 * mossy))
        for _ in range(int(L * 3.0 * mossy)):              # and along the road edge of the channel
            s = rm.u(0.04, L - 0.04)
            tuft(s, -gw - 0.004, 0.0006, rm.u(0.02, 0.05), rm.u(0.006, 0.011), rm.u(0.008, 0.016))

    # ---------------------------------------------------------------- fallen leaves
    leaves = _Acc("leaves", seed)
    if p.leaves > 0.02:
        rl = rng.fork("leaves")
        n_leaf = int(round(p.leaves * (7.0 * L + (36 if has_grate else 0))))
        for i in range(n_leaf):
            at_gully = has_grate and rl.chance(0.62)
            if at_gully:                                   # washed against the upstream side of the gully and onto the lid
                s = clamp(sg - 0.08 + abs(rl.u(-1, 1) * rl.u(0.2, 1)) * 0.5 * (-1 if rl.chance(0.82) else 0.7), 0.05, L - 0.05)
                u = -0.014 - rl.u(0, 1) ** 1.6 * 0.62 * gw
            else:
                s = rl.u(0.05, L - 0.05)
                u = -0.012 - abs(rl.u(-1, 1) * rl.u(0, 1)) * 0.45 * gw
            z = float(dish(u)) - float(sag(s)) + 0.0085
            if has_grate and g0 - 0.01 < s < g1 + 0.01 and ua_g - 0.01 < u:
                z = float(gt(s, u)) + 0.0055
            ln, wd = rl.u(0.05, 0.085), rl.u(0.5, 0.75)
            yaw = rl.u(0, 2 * math.pi)
            lean = rl.u(0.2, 0.9) if u > -0.035 else rl.u(-0.1, 0.1)      # leaning against the kerb face
            curl, fold = rl.u(0.05, 0.3), rl.u(0.1, 0.4)
            cy, sy = math.cos(yaw), math.sin(yaw)
            loc = [(-0.5, 0.0), (-0.3, -0.34 * wd), (0.0, -0.5 * wd), (0.3, -0.3 * wd), (0.5, 0.0), (0.3, 0.3 * wd), (0.0, 0.5 * wd), (-0.3, 0.34 * wd)]
            V, T = [], []
            for (lx, ly) in loc:
                ds, dv = (lx * cy - ly * sy) * ln, (lx * sy + ly * cy) * ln
                zz = z + fold * abs(ly) * ln + curl * max(0.0, lx) ** 2 * ln + max(0.0, dv) * lean + rl.u(0, 0.0015)
                V.append((clamp(s + ds, 0.004, L - 0.004), min(u + dv * (1.0 - 0.5 * abs(lean)), -0.003), zz))
                T.append(((ly / max(wd, 1e-3) + 0.5) * 0.25, (lx + 0.5) * 0.25))
            F = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 6), (0, 6, 7)]
            leaves.add(V, F, g=rl.u(0.0, 1.0) ** 1.5, uvs=[[T[a] for a in f] for f in F])

    # ---------------------------------------------------------------- materials and realisation
    sett_rough = lerp(0.92, 0.38, wet)
    granite = mat("mortar", p.granite, scale=0.6)          # fine speckled grain = fine-axed granite; also the joint fill
    kerbs[0].build(path, granite, 50.0)
    fill.build(path, granite, 60.0)
    setts[0].build(path, mat("stone_smooth", p.sett, rough=sett_rough, scale=1.5), 75.0)
    ironm = mat("iron_wrought")
    iron.build(path, ironm, 40.0)
    if txt_obj is not None:
        assign(txt_obj, ironm)
        paint(txt_obj, wear=0.85, special=0.0, grime=0.0)
    flags.build(path, mat("stone_smooth", "#b3a68e"), 50.0)
    # a thin glossy film: darkens the setts it covers and catches the lantern as a sharp highlight
    waters[0].build(path, mat("glass_clear", "#2c3336", opacity=0.76, rough=0.02), 60.0)
    tufts.build(path, mat("moss"), 70.0)
    leaves.build(path, mat("leaf", p.leaf_tint), 60.0)

    ea, eb = path.pt(0.0, 0.0, 0.0), path.pt(L, 0.0, 0.0)
    end_rot = 0.0 if R == 0 else sig * 90.0
    socket("end_a", "attach", ea, rot=(0, 0, 180), role="kerb_end")
    socket("end_b", "attach", eb, rot=(0, 0, end_rot), role="kerb_end")
    print("DBG kerb_gutter L=%.2f R=%.2f stones=%d setts=%d grate=%s drop=%s pav=%.2f open_joints=%d outlet=%s" % (L, R, len(stones), n_setts, has_grate, drop is not None, pav, n_open, outlet is not None))
