"""Single-arch humpback stone bridge over the canal. A segmental (or, over deep water, stilted round)
arch barrel of jointed ring courses ends on both faces in individually cut voussoirs with a projecting
dated keystone and springs from impost courses on two ashlar abutments; the spandrels are coursed rubble
set in recessed lime mortar, closed by a weathered string course that follows the hump of the deck.
The deck is laid in granite setts with a crowned camber, cross ribs on the steep part and dished stone
gutter channels (or, when the hump is too steep to walk, in stone steps). Above it either a raking
ashlar parapet of 4-5 low courses under a thin saddleback coping with a drip, or a low stone plinth
carrying a wrought-iron railing, stopped by four square end piers that run down to the canal bed as
pilasters. Gas lanterns stand on the pier caps, one hangs on a chain under the crown so the vault
glows over the water, and a small bracket lantern on each canal face, just above the keystone, rakes
its light down over the string course, the date and the voussoir ring.
Origin: deck centre at bank level (z = 0); span along X, canal along Y; abutments at x = +-span/2.
All masonry stays inside `width`; only the two bracket lanterns (param crown_lamps, 0 = none) hang
about 0.23 m further out over the water on each side, clear of the banks and of anything on the deck.
"""
from math import sin, cos, asin, atan, sqrt, radians, degrees, pi, hypot, ceil

import numpy as np

from tk import *

# auto_grime: the automatic ground-splash / ground-plane cavity dirt is meant for things standing ON the ground; here
# everything below the banks hangs over a canal, so it is turned down and the dirt is painted by cause in build().
META = {"name": "Stone canal bridge", "class": "landmark", "mount": "ground", "allow_below_ground": True,
        "auto_grime": 0.3, "focus": [0.5, -1.7, 0.0]}
PARAMS = {
    "span": {"default": 7.0, "min": 4.0, "max": 10.0, "desc": "length along X between the abutments, where the deck meets z=0 (m)"},
    "width": {"default": 3.4, "min": 2.2, "max": 6.0, "desc": "overall width of the masonry along Y, outside of string course / pier caps (m); only the crown_lamps brackets overhang it, by 0.23 m per side"},
    "rise": {"default": 0.8, "min": 0.3, "max": 1.2, "desc": "height of the deck crown above the banks (m); humps steeper than about 1:2.4 are built as steps"},
    "parapet": {"default": "stone", "choices": ["stone", "railing"], "desc": "ashlar parapet wall with coping, or stone plinth with iron railing"},
    "parapet_height": {"default": 0.98, "min": 0.8, "max": 1.2, "desc": "parapet / handrail height above the deck (m)"},
    "water_depth": {"default": 1.5, "min": 0.8, "max": 3.0, "desc": "drop from the bank (z=0) to the canal water surface (m)"},
    "stone": {"default": "#b9ae9b", "desc": "tint of the dressed stone (voussoirs, coping, piers)"},
    "rubble": {"default": "#958d80", "desc": "tint of the rubble spandrel stone and the deck setts"},
    "lamps": {"default": 2, "choices": [0, 2, 4], "desc": "gas lanterns standing on the end piers (unlit piers get a lantern socket)"},
    "arch_lantern": {"default": True, "desc": "lantern hanging on a chain under the crown of the arch"},
    "crown_lamps": {"default": 2, "choices": [0, 1, 2], "desc": "small bracket lanterns on the canal faces above the keystones (1 = -Y face only); each hangs 0.23 m outside `width`"},
    "intensity": {"default": 34.0, "min": 5, "max": 120, "desc": "light intensity per pier lantern (cd)"},
    "moss": {"default": 0.55, "min": 0.0, "max": 1.0, "desc": "amount of moss / algae near the waterline and on ledges"},
    "date": {"default": "1784", "desc": "date in iron numerals on the keystone (empty = none)"},
    "water": {"default": True, "desc": "include the strip of canal water under the arch"},
}
VARIANTS = {
    "railing_wide": {"span": 8.0, "width": 4.4, "rise": 0.9, "parapet": "railing", "lamps": 4, "intensity": 22, "water_depth": 2.0,
                     "crown_lamps": 1, "stone": "#c2b49c", "rubble": "#a39780", "date": "1821", "seed": 4},
    "footbridge_steep": {"span": 4.6, "width": 2.5, "rise": 1.0, "water_depth": 1.9, "lamps": 2, "intensity": 24, "stone": "#a09a8e",
                         "rubble": "#7f7a72", "moss": 0.95, "parapet_height": 0.9, "date": "1702", "seed": 11},
    "long_low": {"span": 9.6, "width": 3.8, "rise": 0.4, "water_depth": 1.1, "lamps": 4, "intensity": 22, "stone": "#cdb893",
                 "rubble": "#b09a78", "moss": 0.25, "parapet_height": 0.85, "date": "1856", "seed": 23},
    "railing_small": {"span": 5.0, "width": 2.8, "rise": 0.6, "parapet": "railing", "lamps": 2, "water_depth": 1.3,
                      "arch_lantern": False, "moss": 0.8, "parapet_height": 1.05, "date": "", "seed": 31},
}

R2 = sqrt(2.0)


def _shade(hx, k):
    """Scale the channels of '#rrggbb' by k = (r, g, b)."""
    h = str(hx).lstrip("#")
    try:
        c = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        c = [185, 174, 155]
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(v * f)))) for v, f in zip(c, k))


def _grid(name, cols, rows):
    """Quad grid through columns [(bottom xyz, top xyz), ...]."""
    verts, faces = [], []
    n = rows + 1
    for b, t in cols:
        for j in range(n):
            f = j / rows
            verts.append((b[0] + (t[0] - b[0]) * f, b[1] + (t[1] - b[1]) * f, b[2] + (t[2] - b[2]) * f))
    for i in range(len(cols) - 1):
        for j in range(rows):
            a = i * n + j
            faces.append((a, a + n, a + n + 1, a + 1))
    return mesh_object(name, verts, faces)


def _sheet(name, xs, ys, zfun):
    """Up-facing height field z = zfun(X, Y) over the lattice xs x ys."""
    X, Y = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float), indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.asarray(zfun(X.ravel(), Y.ravel()), float) + 0 * X.ravel()], 1)
    ny = len(ys)
    F = [(i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1) for i in range(len(xs) - 1) for j in range(ny - 1)]
    return mesh_object(name, V, F)


def _face_to(o, d):
    """Flip the whole (open) mesh if it does not face direction d on average."""
    acc = 0.0
    for poly in o.data.polygons:
        nn = poly.normal
        acc += (nn[0] * d[0] + nn[1] * d[1] + nn[2] * d[2]) * poly.area
    if acc < 0:
        o.data.flip_normals()
    return o


# ------------------------------------------------------------------ cut-stone primitive
def _raw_block(h, c, d, faces, n):
    """Chamfered box (half sizes h) of which only `faces` [(axis, sign), ...] exist. Present neighbours
    are joined by a chamfer strip; towards a missing face the stone ends in a skirt that dives `d`
    into the joint (so walls are built from face trays over a mortar backing, corners from real boxes)."""
    V, F, G, SK = [], [], {}, {}

    def add(pt):
        V.append(tuple(pt))
        return len(V) - 1

    lin = [[(-h[a] + c) + (2 * h[a] - 2 * c) * t / n[a] for t in range(n[a] + 1)] for a in range(3)]
    for (k, s) in faces:
        i, j = [a for a in range(3) if a != k]
        g = [[0] * (n[j] + 1) for _ in range(n[i] + 1)]
        for p_ in range(n[i] + 1):
            for q_ in range(n[j] + 1):
                pt = [0.0, 0.0, 0.0]
                pt[k], pt[i], pt[j] = s * h[k], lin[i][p_], lin[j][q_]
                g[p_][q_] = add(pt)
        G[(k, s)] = g
        for p_ in range(n[i]):
            for q_ in range(n[j]):
                F.append((g[p_][q_], g[p_ + 1][q_], g[p_ + 1][q_ + 1], g[p_][q_ + 1]))

    def row(k, s, a, sa):                    # edge row of face (k, s) on its side (a, sa), running along the third axis
        i, j = [x for x in range(3) if x != k]
        g = G[(k, s)]
        e = -1 if sa > 0 else 0
        return list(g[e]) if a == i else [r[e] for r in g]

    def strip(r1, r2):
        for m in range(len(r1) - 1):
            F.append((r1[m], r1[m + 1], r2[m + 1], r2[m]))

    def end(r, sb):
        return r[-1 if sb > 0 else 0]

    for (k, s) in faces:
        for a in range(3):
            if a == k:
                continue
            for sa in (-1, 1):
                r1 = row(k, s, a, sa)
                if (a, sa) in G:
                    if (k, s) < (a, sa):
                        strip(r1, row(a, sa, k, s))
                else:
                    r2 = []
                    for vi in r1:
                        pt = list(V[vi])
                        pt[a], pt[k] = sa * h[a], s * (h[k] - d)
                        r2.append(add(pt))
                    SK[(k, s, a, sa)] = r2
                    strip(r1, r2)
    for (k, s) in faces:
        i, j = [a for a in range(3) if a != k]
        for si in (-1, 1):
            for sj in (-1, 1):
                fc = G[(k, s)][-1 if si > 0 else 0][-1 if sj > 0 else 0]
                pi_, pj_ = (i, si) in G, (j, sj) in G
                if pi_ and pj_:
                    if (k, s) < (i, si) and (k, s) < (j, sj):
                        F.append((fc, end(row(i, si, k, s), sj), end(row(j, sj, k, s), si)))
                elif not pi_ and not pj_:
                    pt = list(V[fc])
                    pt[i], pt[j], pt[k] = si * h[i], sj * h[j], s * (h[k] - d)
                    x_ = add(pt)
                    F.append((fc, end(SK[(k, s, i, si)], sj), x_))
                    F.append((fc, x_, end(SK[(k, s, j, sj)], si)))
                elif pi_:
                    if (k, s) < (i, si):
                        F.append((fc, end(SK[(k, s, j, sj)], si), end(SK[(i, si, j, sj)], s), end(row(i, si, k, s), sj)))
                else:
                    if (k, s) < (j, sj):
                        F.append((fc, end(SK[(k, s, i, si)], sj), end(SK[(j, sj, i, si)], s), end(row(j, sj, k, s), si)))
    return V, F


_TOPO = {}


def _block(h, faces, n, c, d):
    """-> (verts (N,3) in block space, faces). Topology is cached; positions are linear in h, c, d."""
    key = (tuple(faces), tuple(n))
    t = _TOPO.get(key)
    if t is None:
        A = np.array(_raw_block((1, 1, 1), 0, 0, key[0], key[1])[0], float)
        B = np.array(_raw_block((0, 0, 0), 1, 0, key[0], key[1])[0], float)
        D = np.array(_raw_block((0, 0, 0), 0, 1, key[0], key[1])[0], float)
        F = _raw_block((1, 1, 1), 0.1, 0.15, key[0], key[1])[1]
        V = A + B * 0.1 + D * 0.15
        out = []
        for f in F:
            nn = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
            out.append(tuple(f) if nn.dot(V[list(f)].mean(0)) >= 0 else tuple(reversed(f)))
        t = _TOPO[key] = (A, B, D, out)
    return t[0] * np.asarray(h, float) + t[1] * c + t[2] * d, t[3]


class _Acc:
    """Collects many stones into one mesh with per-stone tone and texture offset."""

    def __init__(self):
        self.V, self.F, self.T, self.U, self.n = [], [], [], [], 0

    def add(self, V, F, tone, uvo):
        n = self.n
        self.F.extend(tuple(i + n for i in f) for f in F)
        self.V.append(np.asarray(V, float))
        self.T.append(np.broadcast_to(np.asarray(tone, float), (len(V),)).copy())
        self.U.append(np.broadcast_to(np.asarray(uvo, float), (len(F), 2)).copy())
        self.n += len(V)

    def obj(self, name, material, smooth_angle=40.0):
        if not self.n:
            return None
        V = np.concatenate(self.V)
        o = mesh_object(name, V, self.F)
        me = o.data
        lens = np.array([len(f) for f in self.F])
        if len(me.polygons) == len(self.F) and len(me.loops) == int(lens.sum()) and len(me.vertices) == len(V):
            lv = np.fromiter((i for f in self.F for i in f), dtype=np.int64, count=int(lens.sum()))
            first = np.concatenate([[0], np.cumsum(lens)[:-1]])
            nn = np.cross(V[lv[first + 1]] - V[lv[first]], V[lv[first + 2]] - V[lv[first]])
            fa = np.repeat(np.argmax(np.abs(nn), axis=1), lens)
            P = V[lv]
            uo = np.repeat(np.concatenate(self.U), lens, axis=0)
            uu = np.where(fa == 0, P[:, 1], P[:, 0]) + uo[:, 0]
            vv = np.where(fa == 2, P[:, 1], P[:, 2]) + uo[:, 1]
            me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.stack([uu, vv], 1).astype(np.float32).ravel())
            uv_keep(o)
            self.tone = np.concatenate(self.T)
        else:                                            # (mesh was cleaned up: fall back to neutral tone)
            self.tone = np.full(len(me.vertices), 0.5)
        assign(o, material)
        smooth(o, smooth_angle)
        self.o = o
        return o


def build(p, rng):
    S, W, H = float(p.span), float(p.width), float(p.rise)
    hs = S / 2
    PW, EM = 0.55, 0.06                      # end pier width, margin to the abutment line
    hc = hs - PW - EM                        # half clear span of the arch
    OUT = 0.10                               # room for string course / pier caps / keystone inside `width`
    YF = W / 2 - OUT                         # structural face of spandrel / parapet
    PT = 0.30                                # parapet thickness
    PP = 0.04                                # how far the pilasters stand proud of the spandrel
    ph = float(p.parapet_height)
    zw = -float(p.water_depth)
    zbot = zw - 0.25
    seed = int(p.seed)
    railing = p.parapet == "railing"
    mk_ = float(p.moss)
    nrs = np.random.RandomState((seed * 7919 + 13) % 2000000)
    GJ = 0.0025                              # half joint
    xe = hc                                  # parapet runs between the piers
    dw = YF - PT                             # half clear width between the parapet faces
    ysw = YF + PP - PW - 0.012               # half width of the sett paving (inside gutters / piers)
    stepped = 3.0 * H / S > 0.42             # steepest part of the hump too steep to walk -> steps

    def deck(x):
        t = min(1.0, abs(2 * x / S))
        return H * (1 - t * t) ** 1.5

    def slope(x):
        return (deck(x + 0.01) - deck(x - 0.01)) / 0.02

    def deck_np(x):
        t = np.clip(np.abs(2 * np.asarray(x, float) / S), 0, 1)
        return H * (1 - t * t) ** 1.5

    n_r = int(ceil(H / 0.165))
    hr = H / n_r
    xk = [hs * sqrt(max(0.0, 1 - ((k - 0.5) * hr / H) ** (2 / 3))) for k in range(1, n_r + 1)]   # riser k at |x| = xk[k-1]

    def walk(x, y):                          # walking surface
        x = np.asarray(x, float)
        yn = np.clip(np.abs(y) / dw, 0, 1)
        if stepped:
            lvl = np.zeros_like(x)
            for q in xk:
                lvl = lvl + (np.abs(x) < q)
            return lvl * hr - 0.02 * yn ** 2
        return deck_np(x) - 0.04 * yn ** 2

    # ---- arch geometry: ring under the crown, springing just above the water
    tr = 0.36 + 0.012 * S                    # ring thickness
    zci = H - tr - 0.2                       # crown of the intrados
    zs = zw + 0.25
    ar = zci - zs
    if ar < 0.14 * 2 * hc:
        zs = max(zci - 0.14 * 2 * hc, zw - 0.5)
        ar = zci - zs
    if ar > hc:                              # deep canal: round arch on stilted jambs
        ar = hc
        zs = zci - ar
    R = (hc * hc + ar * ar) / (2 * ar)
    zc = zci - R
    phi = asin(min(1.0, hc / R))
    Rref = R + tr / 2

    def intr(x, r=None):
        r = R if r is None else r
        return zc + sqrt(max(r * r - x * x, 0.0))

    # ---- materials
    m_dress = mat("stone_smooth", p.stone)
    m_dress2 = mat("stone_smooth", _shade(p.stone, (0.95, 0.93, 0.9)))      # a slightly buffer bed of the same quarry
    m_wet = mat("stone_smooth", _shade(p.stone, (0.6, 0.6, 0.46)), rough=0.45)   # algae-stained, permanently wet courses
    m_rub = mat("stone_rough", p.rubble)
    m_mortar = mat("mortar", "#ddd5c2", grime=0.7)
    m_moss = mat("moss")
    m_iron = mat("iron_wrought")
    m_glass = mat("glass_lantern", glow=0.32)
    m_flame = mat("emissive", "#ffb868", glow=2.3)

    A_d, A_d2, A_wet, A_rub, A_sett, A_mor = _Acc(), _Acc(), _Acc(), _Acc(), _Acc(), _Acc()
    rt = rng.fork("tone")

    QD = min(1.7, max(1.0, (S * W / 24.0) ** 0.8))       # big bridges get coarser facets, so every size stays in budget

    def nd(L, sp):
        return max(1, int(round(L / (sp * QD))))

    def w_shift(cx, cy, cz):
        return lambda V: V + np.array((cx, cy, cz))

    def w_deck(cx, cy, cz):                  # block follows the hump: local z is measured from the deck line
        def f(V):
            X = V[:, 0] + cx
            return np.stack([X, V[:, 1] + cy, deck_np(X) + cz + V[:, 2]], 1)
        return f

    def w_walk(cx, cy, cz):
        def f(V):
            X, Y = V[:, 0] + cx, V[:, 1] + cy
            return np.stack([X, Y, walk(X, Y) + cz + V[:, 2]], 1)
        return f

    def w_cyl(a_c, y_c, r_c):                # local x = arc length, z = radial
        def f(V):
            a = a_c + V[:, 0] / Rref
            r = r_c + V[:, 2]
            return np.stack([r * np.sin(a), V[:, 1] + y_c, zc + r * np.cos(a)], 1)
        return f

    def stone(h, faces, n, warp, kind="dress", zc_=None, c=0.005, d=0.009, chips=0.22, rough=0.0009, bulge=0.0, skew=0.0, chipd=0.004):
        V, F = _block(h, faces, n, c, d)
        k, s = faces[0]
        i, j = [a for a in range(3) if a != k]
        if bulge:
            V[:, k] += s * bulge * np.clip(1 - (V[:, i] / h[i]) ** 2, 0, 1) * np.clip(1 - (V[:, j] / h[j]) ** 2, 0, 1) * (V[:, k] * s > h[k] - 1e-6)
        if skew:
            u, v = V[:, i] / h[i], V[:, j] / h[j]
            q = nrs.uniform(-1, 1, 4)
            V[:, i] += skew * (q[0] * v + q[1] * u * v)
            V[:, j] += skew * (q[2] * u + q[3] * u * v)
        V += nrs.uniform(-rough, rough, V.shape)
        if chips and nrs.rand() < chips:     # knocked corners / spalled arrises
            hh = np.asarray(h, float)
            for _ in range(1 + int(nrs.rand() < 0.3)):
                kk, ss = faces[nrs.randint(len(faces))]
                sg = np.where(nrs.rand(3) < 0.5, -1.0, 1.0)
                sg[kk] = ss
                P = sg * hh
                dv = sg.copy()
                if nrs.rand() < 0.45:
                    a = [x for x in range(3) if x != kk][nrs.randint(2)]
                    P[a] = nrs.uniform(-0.7, 0.7) * hh[a]
                    dv[a] = 0.0
                rc = nrs.uniform(0.035, 0.075)
                w = np.clip(1 - np.linalg.norm(V - P, axis=1) / rc, 0, 1)
                dsp = dv / np.linalg.norm(dv) * (w * rc * 0.5)[:, None]
                for (ka, _s) in faces:       # never dig through to the mortar backing behind a face
                    dsp[:, ka] = np.clip(dsp[:, ka], -chipd, chipd)
                V -= dsp
        if kind == "dress":
            if zc_ is not None and zc_ < zw + 0.16 + 0.2 * rt.u():
                acc = A_wet
            else:
                acc = A_d2 if rt.chance(0.4) else A_d
        elif kind == "sett":
            acc = A_sett
        else:
            acc = A_rub
        acc.add(warp(V), F, rt.u(), (nrs.uniform(0, 3), nrs.uniform(0, 3)))

    def extr(acc, xs, prof, y0, sy, zfn, closed=True, ref=None, chips=0.0):
        """Stone of constant section `prof` [(a outwards, b up), ...] running along x on the line z = zfn(x)."""
        xs = np.asarray(xs, float)
        P = np.array(prof, float)
        m = len(P)
        X = np.repeat(xs, m)
        V = np.stack([X, y0 + sy * np.tile(P[:, 0], len(xs)), zfn(X) + np.tile(P[:, 1], len(xs))], 1)
        F = []
        for i in range(len(xs) - 1):
            for j in range(m if closed else m - 1):
                a, b = i * m + j, i * m + (j + 1) % m
                F.append((a, b, b + m, a + m))
        cen = V.mean(0)
        acc_n = 0.0
        for f in F[:m]:
            nn = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
            acc_n += nn.dot(ref) if ref is not None else nn.dot(V[list(f)].mean(0) * (0, 1, 1) - cen * (0, 1, 1))
        if acc_n < 0:
            F = [tuple(reversed(f)) for f in F]
        V += nrs.uniform(-0.0008, 0.0008, V.shape)
        if chips and nrs.rand() < chips:
            P0 = V[nrs.randint(len(V))].copy()
            rc = nrs.uniform(0.04, 0.08)
            w = np.clip(1 - np.linalg.norm(V - P0, axis=1) / rc, 0, 1)
            dv = (cen - P0) * (0, 1, 1)
            V += dv / max(np.linalg.norm(dv), 1e-6) * (w * rc * 0.45)[:, None]
        acc.add(V, F, rt.u(), (nrs.uniform(0, 3), nrs.uniform(0, 3)))

    def pick():
        return A_d2 if rt.chance(0.4) else A_d

    def pointing(xj, prof, cen, k, y0, sy, zfn, closed=True, ref=None):
        """Mortar in the butt joints of a run of profiled stones (a thin slice of the slightly shrunk section)."""
        cen = np.array(cen)
        pr = [tuple(cen + (np.array(q) - cen) * k) for q in prof]
        for x in xj:
            extr(A_mor, [x - 0.016, x + 0.016], pr, y0, sy, zfn, closed=closed, ref=ref)

    def split(r, a0, a1, lmin, lmax):
        """Random joint positions a0 < ... < a1."""
        out = [a0]
        while a1 - out[-1] > lmax:
            L = r.u(lmin, lmax)
            if a1 - (out[-1] + L) < lmin * 0.7:
                L = (a1 - out[-1]) / 2
            out.append(out[-1] + L)
        out.append(a1)
        return out

    dressed, mortar, iron, glass, flame = [], [], [], [], []

    wall_h = (ph - 0.12) if not railing else 0.24       # masonry height above the deck line at the edges
    ZB = -0.085                                         # parapet courses start this far below the deck line

    # =====================================================================  mortar backing (seen only in the joints)
    for sy in (-1, 1):
        y = sy * (YF - 0.005)
        n = max(12, int(2 * hc / 0.25))
        xs_ = np.linspace(-hc, hc, n + 1)
        cols = [((x, y, intr(x, R + 0.02)), (x, y, deck(x) + wall_h)) for x in xs_]
        mortar.append(_face_to(_grid("core_side", cols, 6), (0, sy, 0)))
        for sx in (-1, 1):
            cols = [((sx * (hs - EM - 0.01), y, zbot), (sx * (hs - EM - 0.01), y, 0.0)), ((sx * hs, y, zbot), (sx * hs, y, 0.0))]
            mortar.append(_face_to(_grid("core_abut", cols, 4), (0, sy, 0)))
        yi = sy * (YF - PT + 0.005)
        cols = [((x, yi, deck(x) - 0.11), (x, yi, deck(x) + wall_h)) for x in xs_]
        mortar.append(_face_to(_grid("core_par_in", cols, 3), (0, -sy, 0)))
        # bed of the arch ring behind the voussoir faces
        na = max(16, int(2 * phi * R / 0.2))
        cols = [(((R + 0.004) * sin(a), sy * (YF + 0.02), zc + (R + 0.004) * cos(a)),
                 ((R + tr - 0.02) * sin(a), sy * (YF + 0.02), zc + (R + tr - 0.02) * cos(a))) for a in np.linspace(-phi, phi, na)]
        mortar.append(_face_to(_grid("ring_bed", cols, 2), (0, sy, 0)))
    Rs = R + 0.006
    sof = surface("soffit_bed", lambda u, v: ((Rs * sin(-phi + v * 2 * phi)), -YF + 0.02 + u * (2 * YF - 0.04), zc + Rs * cos(-phi + v * 2 * phi)),
                  nu=max(4, int(W / 0.6)), nv=max(12, int(2 * R * phi / 0.25)))
    mortar.append(_face_to(sof, (0, 0, -1)))
    for sx in (-1, 1):
        if zs - zbot > 0.05:
            cols = [((sx * (hc + 0.018), y, zbot), (sx * (hc + 0.018), y, zs)) for y in np.linspace(-YF, YF, 7)]
            mortar.append(_face_to(_grid("jamb_bed", cols, max(2, int((zs - zbot) / 0.3))), (-sx, 0, 0)))
        cols = [((sx * (hs - 0.006), y, zbot), (sx * (hs - 0.006), y, 0.0)) for y in np.linspace(-YF, YF, 7)]
        mortar.append(_face_to(_grid("abut_bed", cols, 5), (sx, 0, 0)))

    # =====================================================================  arch barrel: ring courses, voussoirs, keystone
    rv = rng.fork("voussoir")
    ak = 0.26 / Rref                                     # keystone half angle
    nside = max(3, int(round((phi - ak) * Rref / 0.29)))
    wa = (phi - ak) / nside
    courses = [(-ak, ak, -1)]
    for i in range(nside):
        courses.append((ak + i * wa, ak + (i + 1) * wa, i))
        courses.append((-ak - (i + 1) * wa, -ak - i * wa, i))
    for a0, a1, i in courses:
        am = (a0 + a1) / 2
        key = i < 0
        hx = (a1 - a0) / 2 * Rref - GJ
        ends = {}
        for sy in (-1, 1):
            if key:
                dv, proud, r0, r1 = 0.7, 0.075, R - 0.03, R + tr + 0.03
            else:
                dv = (0.62 if i % 2 else 0.4) + rv.u(-0.03, 0.03)
                proud = 0.035 + rv.u(-0.006, 0.008)
                r0 = R + rv.u(-0.003, 0.003)
                r1 = R + tr * (1.0 + (0.14 if i % 2 else 0.0)) + rv.u(-0.012, 0.012)
            dv = min(dv, YF - 0.15)
            ends[sy] = dv
            faces = [(1, sy), (2, -1), (2, 1)]
            if key:
                faces += [(0, -1), (0, 1), (1, -sy)]
            elif i % 2:
                faces += [(0, -1), (0, 1)]
            rc_ = (r0 + r1) / 2
            stone((hx, (dv + proud) / 2, (r1 - r0) / 2), faces, (nd(2 * hx, 0.065), nd(dv, 0.3), nd(r1 - r0, 0.085)),
                  w_cyl(am, sy * (YF + proud - (dv + proud) / 2), rc_), zc_=zc + rc_ * cos(am), c=0.008 if key else 0.006, d=0.022,
                  chips=0.3, chipd=0.03 if key else 0.006)
        ys_ = split(rv, -YF + ends[-1] + 0.005, YF - ends[1] - 0.005, 0.5, 0.95)
        for ya, yb in zip(ys_[:-1], ys_[1:]):
            rj = R + rv.u(-0.004, 0.004)
            stone((hx, (yb - ya) / 2 - GJ, 0.03), [(2, -1)], (1, nd(yb - ya, 0.24), 1), w_cyl(am, (ya + yb) / 2, rj + 0.03),
                  zc_=zc + rj * cos(am), c=0.006, d=0.012, chips=0.05)
    for sy in (-1, 1):
        if str(p.date).strip():
            body = str(p.date).strip()[:6]
            tsz = min(0.185, 0.74 / max(len(body), 1))
            tx = text("date", body, size=tsz, depth=0.008, loc=(0, sy * (YF + 0.075 + 0.0005), zci + 0.11),
                      rot=(90, 0, 0 if sy < 0 else 180), font="serif")
            iron.append(tx)

    # impost courses under the springers, ashlar jambs below
    ri = rng.fork("impost")
    has_impost = zs > zw + 0.1
    jtop = zs - 0.18 if has_impost else zs
    for sx in (-1, 1):
        if has_impost:
            ys_ = split(ri, -YF - 0.02, YF + 0.02, 0.6, 0.95)
            for q, (ya, yb) in enumerate(zip(ys_[:-1], ys_[1:])):
                faces = [(0, -sx), (2, 1), (2, -1)]
                if q == 0:
                    faces.append((1, -1))
                if q == len(ys_) - 2:
                    faces.append((1, 1))
                stone((0.15, (yb - ya) / 2 - GJ, 0.09 - 0.002), faces, (1, nd(yb - ya, 0.25), 1), w_shift(sx * (hc + 0.085), (ya + yb) / 2, zs - 0.09),
                      zc_=zs - 0.09, c=0.016, d=0.02, chips=0.5, chipd=0.03)
        if jtop - zbot > 0.12:
            nj = max(1, int(round((jtop - zbot) / 0.29)))
            hj = (jtop - zbot) / nj
            for kc in range(nj):
                ys_ = split(ri, -YF + PP + 0.0, YF - PP, 0.45, 0.9)
                for ya, yb in zip(ys_[:-1], ys_[1:]):
                    zc_ = zbot + (kc + 0.5) * hj
                    stone((0.03, (yb - ya) / 2 - GJ, hj / 2 - GJ), [(0, -sx)], (1, nd(yb - ya, 0.22), 1), w_shift(sx * (hc + 0.012 + 0.03), (ya + yb) / 2, zc_),
                          zc_=zc_, chips=0.2 if zc_ > zw else 0.0)

    # =====================================================================  rubble spandrels
    rr = rng.fork("rubble")
    z_top_all = H - 0.1
    for sy in (-1, 1):
        z = zs + 0.02
        while z < z_top_all:
            h = rr.u(0.13, 0.24)
            x = -hc - 0.04 - rr.u(0, 0.25)
            while x < hc + 0.04:
                L = rr.u(0.2, 0.52)
                x0, x1 = max(x, -hc + 0.005), min(x + L, hc - 0.005)
                x += L
                if x1 - x0 < 0.1:
                    continue
                z1 = min(z + h, min(deck(x0), deck(x1)) - 0.165)
                if z1 - z < 0.07:
                    continue
                xm_, zm_ = (x0 + x1) / 2, (z + z1) / 2
                ok = True
                for cx, cz in ((x0, z), (x1, z), (x0, z1), (x1, z1)):
                    if hypot(cx, cz - zc) < R + 0.08:
                        ok = False
                if not ok or hypot(xm_, zm_ - zc) < R + tr * 0.75:
                    continue
                pr_ = rr.u(0.0, 0.024)
                stone(((x1 - x0) / 2 - 0.008, 0.03, (z1 - z) / 2 - 0.008), [(1, sy)], (nd(x1 - x0, 0.1), 1, nd(z1 - z, 0.14)),
                      w_shift(xm_, sy * (YF + pr_ - 0.03), zm_), kind="rubble", c=0.012, d=pr_ + 0.018, chips=0.3, rough=0.003, bulge=0.007, skew=0.007)
            z += h

    # stone drain spouts through the spandrel at the haunches
    stains = []                                          # (x, side, z top, half width, length, strength)
    for sy in (-1, 1):
        for sx in (-1, 1):
            x = sx * hc * 0.84
            zsp = deck(x) - 0.27
            if zsp > intr(x, R + tr) + 0.08:
                stone((0.065, 0.06, 0.045), [(1, sy), (2, 1), (2, -1), (0, 1), (0, -1)], (1, 1, 1), w_shift(x, sy * (YF + 0.015), zsp), c=0.01, chips=0.0)
                iron.append(box("spout_lead", (0.07, 0.025, 0.012), loc=(x, sy * (YF + 0.08), zsp + 0.012), rot=(sy * 8, 0, 0)))
                stains.append((x, sy, zsp - 0.03, 0.085, 0.95, 0.6))

    # =====================================================================  string course following the deck
    rs = rng.fork("string")
    prof_s = [(-0.03, 0.0), (0.03, 0.0), (0.079, 0.045), (0.085, 0.052), (0.085, 0.118), (0.078, 0.128), (0.06, 0.16), (-0.03, 0.16)]
    for sy in (-1, 1):
        xj = split(rs, -xe, xe, 0.7, 1.05)
        for x0, x1 in zip(xj[:-1], xj[1:]):
            extr(pick(), np.linspace(x0 + GJ, x1 - GJ, nd(x1 - x0, 0.22) + 1), prof_s, sy * YF, sy, lambda X: deck_np(X) - 0.16, chips=0.3)
        pointing(xj[1:-1], prof_s, np.array(prof_s).mean(0), 0.93, sy * YF, sy, lambda X: deck_np(X) - 0.16)

    # =====================================================================  parapet
    rp = rng.fork("parapet")
    if not railing:
        nC = max(3, int(round((wall_h - ZB) / 0.225)))
        hcr = (wall_h - ZB) / nC
        hw = PT / 2 + 0.035
        prof_c = [(-PT / 2 + 0.01, 0.0), (-hw + 0.005, 0.0), (-hw, 0.005), (-hw, 0.076), (-hw + 0.008, 0.085), (-0.045, 0.12), (0.045, 0.12),
                  (hw - 0.008, 0.085), (hw, 0.076), (hw, 0.005), (hw - 0.005, 0.0), (PT / 2 - 0.01, 0.0)]
        for sy in (-1, 1):
            for fs, yf, sp in ((sy, sy * YF, 0.05), (-sy, sy * (YF - PT), 0.07)):
                for kc in range(nC):
                    xj = split(rp, -xe - (rp.u(0.1, 0.4) if kc % 2 else 0.0), xe, 0.35, 0.7)
                    for x0, x1 in zip(xj[:-1], xj[1:]):
                        x0 = max(x0, -xe)
                        if x1 - x0 < 0.06:
                            continue
                        stone(((x1 - x0) / 2 - GJ, 0.03, hcr / 2 - GJ), [(1, fs)], (nd(x1 - x0, sp), 1, 1),
                              w_deck((x0 + x1) / 2, yf - fs * 0.03, ZB + (kc + 0.5) * hcr), chips=0.25)
            # thin saddleback coping with a drip overhang
            xj = split(rp, -xe, xe, 0.6, 0.95)
            for x0, x1 in zip(xj[:-1], xj[1:]):
                extr(pick(), np.linspace(x0 + GJ, x1 - GJ, nd(x1 - x0, 0.13) + 1), prof_c, sy * (YF - PT / 2), 1, lambda X: deck_np(X) + wall_h + 0.002,
                     closed=False, chips=0.45)
            pointing(xj[1:-1], prof_c, (0.0, 0.04), 0.95, sy * (YF - PT / 2), 1, lambda X: deck_np(X) + wall_h + 0.002, closed=False)
            cols = [((x, sy * (YF - PT + 0.02), deck(x) + wall_h - 0.004), (x, sy * (YF - 0.02), deck(x) + wall_h - 0.004)) for x in np.linspace(-xe, xe, 14)]
            mortar.append(_face_to(_grid("coping_bed", cols, 1), (0, 0, 1)))
    else:
        hp = wall_h
        for si, sy in enumerate((-1, 1)):
            xj = split(rp, -xe, xe, 0.6, 1.0)
            for x0, x1 in zip(xj[:-1], xj[1:]):
                stone(((x1 - x0) / 2 - GJ, PT / 2, (hp - ZB) / 2), [(1, -1), (1, 1), (2, 1)], (nd(x1 - x0, 0.085), 2, 1),
                      w_deck((x0 + x1) / 2, sy * (YF - PT / 2), ZB + (hp - ZB) / 2), c=0.012, chips=0.35)
            cols = [((x, sy * (YF - PT + 0.03), deck(x) + hp - 0.008), (x, sy * (YF - 0.03), deck(x) + hp - 0.008)) for x in np.linspace(-xe, xe, 12)]
            mortar.append(_face_to(_grid("plinth_bed", cols, 1), (0, 0, 1)))
            y = sy * (YF - PT / 2)
            xs_ = np.linspace(-xe - 0.02, xe + 0.02, max(12, int(2 * xe / 0.3)))
            zb_, zt_, zh_ = hp + 0.075, ph - 0.135, ph
            iron.append(sweep("rail_bottom", [(xx, y, deck(xx) + zb_) for xx in xs_], rounded_rect(0.034, 0.012, 0.003, 1)))
            iron.append(sweep("rail_top", [(xx, y, deck(xx) + zt_) for xx in xs_], rounded_rect(0.034, 0.012, 0.003, 1)))
            hr_ = [(-0.027, 0), (0.027, 0), (0.03, 0.01), (0.016, 0.026), (-0.016, 0.026), (-0.03, 0.01)]
            iron.append(sweep("handrail", [(xx, y, deck(xx) + zh_ - 0.026) for xx in xs_], hr_))
            nb = max(8, int(round(2 * xe / 0.125)))
            nst = max(2, int(round(2 * xe / 1.45)))               # bays between heavier standards
            per = max(4, int(round(nb / nst)))
            hb = zt_ - zb_
            bar = box("bal", (0.014, 0.014, hb))               # instance bases stay at the origin
            ring = torus("ring", major=0.049, minor=0.0055, major_seg=8, minor_seg=3)
            hst = ph + 0.07 - hp + 0.04
            std = box("std", (0.03, 0.03, hst), bevel=0.003, segments=1)
            ball = lathe("ball", [(0.0, 0.0), (0.022, 0.0), (0.026, 0.008), (0.018, 0.016), (0.03, 0.035), (0.034, 0.055), (0.024, 0.08), (0.008, 0.095), (0.0, 0.11)], segments=6)
            shoe = lathe("shoe", [(0.045, 0.0), (0.045, 0.012), (0.03, 0.022), (0.024, 0.04)], segments=4, rot=(0, 0, 45))
            mb, mr, ms, mk, mh = [], [], [], [], []
            for i in range(nb + 1):
                xx = -xe + 2 * xe * i / nb
                if i % per == 0 or i == nb:
                    zz = deck(xx) + hp - 0.04
                    ms.append(trs((xx, y, zz + hst / 2), (0, 0, 0)))
                    mk.append(trs((xx, y, zz + hst)))
                    mh.append(trs((xx, y, deck(xx) + hp - 0.004)))
                else:
                    mb.append(trs((xx + rp.u(-0.003, 0.003), y, deck(xx) + zb_ + hb / 2), (rp.u(-0.6, 0.6), rp.u(-0.6, 0.6), 45)))
                if i < nb:
                    xm_ = xx + xe / nb
                    mr.append(trs((xm_, y, deck(xm_) + (zt_ + zh_ - 0.026) / 2), (90, 0, 0)))
            iron.append(instances_to_mesh("balusters", bar, mb, keep_base=False))
            iron.append(instances_to_mesh("rings", ring, mr, keep_base=False))
            iron.append(instances_to_mesh("standards", std, ms, keep_base=False))
            iron.append(instances_to_mesh("finials", ball, mk, keep_base=False))
            iron.append(instances_to_mesh("shoes", shoe, mh, keep_base=False))

    # =====================================================================  end piers (pilasters down to the canal bed)
    rq = rng.fork("pier")
    z_pt = deck(xe) + ph + 0.2
    npc = max(4, int(round((z_pt - zbot) / 0.27)))
    hpc = (z_pt - zbot) / npc
    z_pl = deck(xe) + 0.1                               # centre of the plinth band at bank level
    lit = {(-1, -1): int(p.lamps) >= 2, (1, 1): int(p.lamps) >= 2, (-1, 1): int(p.lamps) >= 4, (1, -1): int(p.lamps) >= 4}
    lamp_i = 0
    guard = lathe("guard", [(0.0, -0.05), (0.15, -0.05), (0.165, 0.06), (0.15, 0.2), (0.11, 0.3), (0.05, 0.36), (0.0, 0.37)], segments=10)
    displace_noise(guard, strength=0.012, scale=0.2, seed=seed)
    gm = []

    def lantern(tag, x, y, zb, w, h, li, gl, fl):
        """Square tapered gas lantern with its floor at zb; returns the mantle position."""
        b, t = w * 0.28, w / 2
        li.append(box(tag + "_floor", (2 * b + 0.024, 2 * b + 0.024, 0.012), loc=(x, y, zb + 0.006), bevel=0.003, segments=1))
        for cx, cy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            li.append(sweep(tag + "_bar", [(x + cx * b, y + cy * b, zb + 0.008), (x + cx * t, y + cy * t, zb + h)], 0.0065, sides=5, smooth_path=False))
        li.append(lathe(tag + "_band", [(t * R2 - 0.014, h - 0.006), (t * R2 + 0.01, h - 0.006), (t * R2 + 0.01, h + 0.014), (t * R2 - 0.014, h + 0.014)],
                        segments=4, loc=(x, y, zb), rot=(0, 0, 45), smooth_angle=30))
        li.append(lathe(tag + "_roof", [(t * R2 + 0.035, h + 0.012), (t * R2 + 0.038, h + 0.022), (t * R2 * 0.6, h + 0.075), (0.07 * w / 0.28, h + 0.1), (0.07 * w / 0.28, h + 0.118), (0.0, h + 0.118)],
                        segments=4, loc=(x, y, zb), rot=(0, 0, 45), smooth_angle=30))
        li.append(lathe(tag + "_vent", [(0.034, h + 0.115), (0.05, h + 0.126), (0.03, h + 0.15), (0.012, h + 0.16), (0.02, h + 0.18), (0.0, h + 0.2)],
                        segments=8, loc=(x, y, zb)))
        li.append(cylinder(tag + "_burner", 0.011, h * 0.36, loc=(x, y, zb + h * 0.18 + 0.01), verts=6))
        gl.append(lathe(tag + "_glass", [(b * R2 * 0.96, 0.014), (t * R2 * 0.96, h - 0.006)], segments=4, loc=(x, y, zb), rot=(0, 0, 45), smooth_angle=30))
        zm_ = zb + h * 0.47
        fl.append(lathe(tag + "_mantle", [(0.0, -0.035), (0.016, -0.03), (0.021, 0.0), (0.016, 0.028), (0.0, 0.036)], segments=8, loc=(x, y, zm_)))
        return (x, y, zm_)

    for sx in (-1, 1):
        for sy in (-1, 1):
            xp = sx * (hs - EM - PW / 2)
            yp = sy * (YF + PP - PW / 2)
            for i in range(npc):
                zc_ = zbot + (i + 0.5) * hpc
                top = zc_ + hpc / 2
                inner = top > deck(xe) - 0.12                 # the face towards the deck is buried below deck level
                sub = 0.065 if zc_ > zw - 0.2 else 0.3
                chip = 0.3 if zc_ > zw else 0.0
                if i % 2 == 0:
                    faces = [(1, sy), (0, -1), (0, 1)] + ([(1, -sy)] if inner else [])
                    stone((PW / 2 + rq.u(-0.002, 0.003), PW / 2 + rq.u(-0.002, 0.003), hpc / 2 - GJ), faces, (nd(PW, sub), nd(PW, sub), 1),
                          w_shift(xp, yp, zc_), zc_=zc_, c=0.006, chips=chip)
                else:
                    dj = rq.u(-0.07, 0.07)
                    ax = (i // 2) % 2                          # split alternately along x / y
                    for sgn in (-1, 1):
                        ln = PW / 2 - sgn * dj - 2 * GJ
                        off = sgn * (PW / 4) + dj / 2
                        if ax == 0:
                            faces = [(1, sy), (0, sgn)] + ([(1, -sy)] if inner else [])
                            stone((ln / 2, PW / 2, hpc / 2 - GJ), faces, (nd(ln, sub), nd(PW, sub), 1), w_shift(xp + off, yp, zc_), zc_=zc_, c=0.006, chips=chip)
                        else:
                            faces = [(0, -1), (0, 1)] + ([(1, sgn)] if (sgn == sy or inner) else [])
                            stone((PW / 2, ln / 2, hpc / 2 - GJ), faces, (nd(PW, sub), nd(ln, sub), 1), w_shift(xp, yp + off, zc_), zc_=zc_, c=0.006, chips=chip)
            mortar.append(box("pier_core", (PW - 0.012, PW - 0.012, z_pt - zbot), loc=(xp, yp, (z_pt + zbot) / 2)))
            # plinth band at bank level and moulded cap
            sides = [(0, -1), (0, 1), (1, -1), (1, 1)]
            stone((PW / 2 + 0.03, PW / 2 + 0.03, 0.12), sides + [(2, 1), (2, -1)], (3, 3, 1), w_shift(xp, yp, z_pl), c=0.02, chips=0.7, chipd=0.025)
            stone((PW / 2 + 0.02, PW / 2 + 0.02, 0.027), sides, (1, 1, 1), w_shift(xp, yp, z_pt + 0.025), c=0.008, chips=0.0)
            stone((PW / 2 + 0.052, PW / 2 + 0.052, 0.05), sides + [(2, 1), (2, -1)], (3, 3, 1), w_shift(xp, yp, z_pt + 0.1), c=0.022, chips=0.7, chipd=0.03)
            rc = (PW + 0.05) / 2 * R2
            zcap = z_pt + 0.15
            if lit[(sx, sy)]:
                # cast-iron base casting covering the cap under the lamp standard
                o = lathe("lamp_base", [(rc * 0.93, 0.0), (rc * 0.93, 0.018), (rc * 0.86, 0.03), (0.19, 0.082), (0.17, 0.09), (0.0, 0.09)], segments=4,
                          loc=(xp, yp, zcap), rot=(0, 0, 45), smooth_angle=30)
                iron.append(o)
                zl = zcap + 0.09
                iron.append(lathe("lamp_std", [(0.105, 0.0), (0.105, 0.018), (0.075, 0.03), (0.05, 0.06), (0.042, 0.075), (0.048, 0.09), (0.03, 0.12),
                                                (0.022, 0.2), (0.019, 0.56), (0.03, 0.575), (0.034, 0.59), (0.03, 0.605), (0.018, 0.62),
                                                (0.03, 0.69), (0.07, 0.725), (0.074, 0.74)], segments=12, loc=(xp, yp, zl)))
                bt = bolt("std_bolt", 0.009, 0.008)
                iron.append(instances_to_mesh("std_bolts", bt, [trs((xp + 0.085 * cos(radians(45 + 90 * q)), yp + 0.085 * sin(radians(45 + 90 * q)), zl + 0.018)) for q in range(4)],
                                              keep_base=False))
                # lamplighter's ladder rest
                iron.append(cylinder("ladder_bar", 0.0075, 0.32, loc=(xp, yp, zl + 0.5), rot=(0, 90, 0), verts=6))
                for sb in (-1, 1):
                    iron.append(sphere("ladder_ball", 0.014, loc=(xp + sb * 0.16, yp, zl + 0.5), segments=6, rings=4))
                lp = lantern(f"lamp{lamp_i}", xp, yp, zl + 0.74, 0.28, 0.34, iron, glass, flame)
                light(f"lamp{lamp_i}", lp, intensity=float(p.intensity), range=10.0, shadow=True, shadow_near=0.13)
                lamp_i += 1
            else:
                o = lathe("pier_top", [(rc, 0.0), (rc, 0.02), (0.05, 0.15), (0.0, 0.155)], segments=4, loc=(xp, yp, zcap), rot=(0, 0, 45), smooth_angle=30)
                bevel(o, 0.008, segments=1)
                assign(o, m_dress)
                dressed.append(o)
                socket(f"lantern_{'w' if sx < 0 else 'e'}{'s' if sy < 0 else 'n'}", "attach", (xp, yp, zcap + 0.155), role="lantern")
            # wheel-guard stone at the inner corner of the pier
            gm.append(trs((sx * (xe - 0.02), sy * (YF + PP - PW - 0.06), deck(xe - 0.02) - 0.02), (rq.u(-4, 4), rq.u(-4, 4), rq.u(0, 360)), rq.u(0.9, 1.1)))
            # ledge stone at the foot of the pier, flush with the abutment end
            stone((EM / 2 + 0.012, PW / 2 - 0.01, 0.07), [(2, 1), (0, sx), (1, sy)], (1, 2, 1), w_shift(sx * (hs - EM / 2 - 0.012), sy * (YF - PW / 2 + 0.01), -0.07), c=0.008)
            # mooring ring on the pilaster, canal side
            zr = zw + 0.62
            if zr < -0.25:
                stains.append((xp, sy, zr - 0.05, 0.045, 0.6, 0.6))
                yq = sy * (YF + PP)
                iron.append(torus("moor_staple", major=0.028, minor=0.008, loc=(xp, yq, zr), rot=(0, 90, 0), major_seg=10, minor_seg=5))
                iron.append(torus("moor_ring", major=0.075, minor=0.009, loc=(xp, yq + sy * 0.016, zr - 0.07), rot=(90 - sy * 8, 0, 0), major_seg=16, minor_seg=5))
    gst = instances_to_mesh("guards", guard, gm, keep_base=False)
    assign(gst, m_rub)

    # abutment end walls (towards the banks)
    ra = rng.fork("abut")
    na_ = max(2, int(round((-0.14 - zbot) / 0.3)))
    ha_ = (-0.14 - zbot) / na_
    for sx in (-1, 1):
        for kc in range(na_):
            ys_ = split(ra, -YF, YF, 0.5, 1.0)
            for ya, yb in zip(ys_[:-1], ys_[1:]):
                zc_ = zbot + (kc + 0.5) * ha_
                stone((0.03, (yb - ya) / 2 - GJ, ha_ / 2 - GJ), [(0, sx)], (1, nd(yb - ya, 0.3), 1), w_shift(sx * (hs - 0.03), (ya + yb) / 2, zc_),
                      zc_=None, chips=0.2)

    # =====================================================================  deck: setts with cross ribs and gutters, or steps
    rd = rng.fork("deck")
    thw = 0.36 if not stepped else max(0.24, hs - xk[0])
    ythr = ysw
    # threshold slabs where the deck meets the banks: level with the bank at the abutment line, and (on a ramped deck)
    # rising with the deck line to the first setts - laid flat they left an open slot under the end of the paving
    w_thr = w_shift if stepped else w_deck
    for sx in (-1, 1):
        ys_ = split(rd, -ythr, ythr, 0.55, 0.95)
        for ya, yb in zip(ys_[:-1], ys_[1:]):
            stone((thw / 2 - 0.002, (yb - ya) / 2 - GJ, 0.07), [(2, 1), (0, 1), (0, -1)], (2, nd(yb - ya, 0.25), 1),
                  w_thr(sx * (hs - thw / 2), (ya + yb) / 2, -0.07 + rd.u(-0.001, 0.003)), c=0.01, chips=0.4, chipd=0.02)

    sett_spans = []                                      # (x0, x1, half width)
    if stepped:
        for sx in (-1, 1):
            for k in range(1, n_r + 1):                  # tread k: top at k * hr, running inwards from riser k
                xo = xk[k - 1]
                crown = k == n_r
                if crown:
                    xi = xo - 0.24 if xo > 0.6 else 0.0
                else:
                    xi = xk[k] - 0.03                    # tucked under the next riser
                if crown and xi == 0.0 and sx > 0:
                    continue
                if crown and xi == 0.0:
                    xo, xi = xo, -xo
                ys_ = split(rd, -dw + 0.004, dw - 0.004, 0.7, 1.3)
                for ya, yb in zip(ys_[:-1], ys_[1:]):
                    faces = [(2, 1), (0, sx)] + ([(0, -sx)] if crown and xi < 0 else [])
                    stone(((xo - xi) / 2, (yb - ya) / 2 - GJ, hr / 2 + 0.012), faces, (nd(xo - xi, 0.3), nd(yb - ya, 0.25), 1),
                          w_shift(sx * (xo + xi) / 2, (ya + yb) / 2, k * hr - hr / 2 - 0.012 + rd.u(-0.002, 0.002)), c=0.012, chips=0.45, chipd=0.02)
        if xk[-1] > 0.6:
            sett_spans.append((-(xk[-1] - 0.244), xk[-1] - 0.244, dw - 0.004))
    else:
        # cross ribs of dressed stone where the ramp is steep (grip for hooves and boots)
        ribs = []
        x = -hs + thw + 0.45
        while x < hs - thw - 0.45:
            if abs(slope(x)) > 0.2:
                ribs.append(x)
                x += rd.u(0.62, 0.78)
            else:
                x += 0.1
        edges = [-hs + thw + 0.004]
        for xr in ribs:
            ys_ = split(rd, -ysw, ysw, 0.7, 1.3)
            for ya, yb in zip(ys_[:-1], ys_[1:]):
                stone((0.062, (yb - ya) / 2 - GJ, 0.05), [(2, 1), (0, 1), (0, -1)], (1, nd(yb - ya, 0.3), 1),
                      w_walk(xr, (ya + yb) / 2, -0.05 + 0.014), c=0.012, chips=0.4, chipd=0.015)
            edges += [xr - 0.066, xr + 0.066]
        edges.append(hs - thw - 0.004)
        for x0, x1 in zip(edges[::2], edges[1::2]):
            sett_spans.append((x0, x1, ysw))
        # dished gutter channels along both parapets, draining to the spouts
        gw = dw - ysw - 0.006
        prof_g = [(0.0, -0.03), (0.0, -0.004), (0.006, 0.0), (0.05, -0.012), (gw * 0.55, -0.02), (gw - 0.04, -0.012), (gw, -0.002)]
        for sy in (-1, 1):
            xj = split(rd, -xe, xe, 0.6, 1.0)
            for x0, x1 in zip(xj[:-1], xj[1:]):
                extr(pick(), np.linspace(x0 + GJ, x1 - GJ, nd(x1 - x0, 0.25) + 1), prof_g, sy * (ysw + 0.006), sy,
                     lambda X: deck_np(X) - 0.04 * (ysw / dw) ** 2 + 0.002, closed=False, ref=(0, 0, 1))
            cols = [((x, sy * (ysw + 0.002), deck(x) - 0.06), (x, sy * dw, deck(x) - 0.06)) for x in np.linspace(-xe, xe, 14)]
            mortar.append(_face_to(_grid("gutter_bed", cols, 1), (0, 0, 1)))

    # granite setts: every stone its own size, height and tilt, laid in rows across the deck
    QS = QD ** 0.5
    for x0, x1, yh in sett_spans:
        cx, cy, hx_, hy_ = [], [], [], []
        x = x0
        while x < x1 - 0.03:
            wr = rd.u(0.105, 0.14) * QS
            if x1 - (x + wr) < 0.08:
                wr = x1 - x
            y = -yh - rd.u(0.0, 0.14)
            while y < yh - 0.01:
                L = rd.u(0.15, 0.24) * QS
                ya, yb = max(y, -yh), min(y + L, yh)
                if yh - yb < 0.07:
                    yb = yh
                y = yb
                if yb - ya < 0.05:
                    continue
                cx.append(x + wr / 2)
                cy.append((ya + yb) / 2)
                hx_.append(wr / 2 - 0.004)
                hy_.append((yb - ya) / 2 - 0.004)
            x += wr
        if not cx:
            continue
        cx, cy, hx_, hy_ = (np.array(a)[:, None] for a in (cx, cy, hx_, hy_))
        ns = len(cx)
        sgx = np.array([-1, 1, 1, -1])[None, :]
        sgy = np.array([-1, -1, 1, 1])[None, :]
        bx, by = cx + sgx * hx_, cy + sgy * hy_
        ins = 0.012
        tx_, ty_ = cx + sgx * (hx_ - ins), cy + sgy * (hy_ - ins)
        off = np.clip(nrs.normal(0, 0.0028, (ns, 1)), -0.006, 0.006)
        bz = walk(bx.ravel(), by.ravel()).reshape(ns, 4) - 0.016
        tz = walk(tx_.ravel(), ty_.ravel()).reshape(ns, 4) + off + nrs.uniform(-0.0018, 0.0018, (ns, 4))
        Vb = np.stack([bx, by, bz], 2)
        Vt = np.stack([tx_, ty_, tz], 2)
        V = np.concatenate([Vb, Vt], 1).reshape(-1, 3)
        F = []
        for q in range(ns):
            b = q * 8
            F += [(b + 4, b + 5, b + 6, b + 7), (b, b + 1, b + 5, b + 4), (b + 1, b + 2, b + 6, b + 5), (b + 2, b + 3, b + 7, b + 6), (b + 3, b, b + 4, b + 7)]
        A_sett.add(V, F, np.repeat(nrs.rand(ns), 8), np.repeat(nrs.uniform(0, 3, (ns, 2)), 5, axis=0))
        bed = _sheet("deck_bed", np.linspace(x0 - 0.004, x1 + 0.004, nd(x1 - x0, 0.22) + 1), np.linspace(-yh - 0.003, yh + 0.003, 7), lambda X, Y: walk(X, Y) - 0.0115)
        assign(bed, m_rub)
        paint(bed, grime=0.8, special=lambda co, n: 0.75 * mk_ * np.clip((np.abs(co[:, 1]) - yh + 0.5) / 0.45, 0, 1))

    # =====================================================================  lantern under the crown
    parts = []
    n_lights = lamp_i
    if p.arch_lantern and zci - 0.95 > zw + 0.3:
        ztop = zci + 0.004
        li, gl, fl = [], [], []
        iron.append(torus("arch_eye", major=0.03, minor=0.008, loc=(0, 0, ztop - 0.03), rot=(90, 0, 0), major_seg=10, minor_seg=5))
        li.append(chain("arch_chain", [(0, 0, ztop - 0.055), (0, 0, ztop - 0.3)], link_len=0.04, wire=0.0045, sides=5))
        li.append(torus("arch_bail", major=0.03, minor=0.006, loc=(0, 0, ztop - 0.325), rot=(90, 0, 0), major_seg=10, minor_seg=5))
        zb = ztop - 0.35 - 0.2 - 0.3
        lp = lantern("archlamp", 0, 0, zb, 0.25, 0.3, li, gl, fl)
        parts.append((li, gl, fl))
        part(li + gl + fl, "arch_lantern", pivot=(0, 0, ztop - 0.03), motion="swing", amp=2.5, speed=0.35)
        light("arch_lantern", lp, intensity=float(p.intensity) * 0.6, range=9.0, shadow_near=0.3, flicker=0.18)
        n_lights += 1
        soot = True
    else:
        soot = False

    # small bracket lanterns on the canal faces above the keystones: they rake down over the string course,
    # the dated keystone and the voussoir ring, and reflect in the water (the only thing outside `width`)
    ncr = max(0, min(int(p.crown_lamps), 6 - n_lights))
    crown_sides = (-1, 1)[:ncr]
    for sy in crown_sides:
        tg = "s" if sy < 0 else "n"
        yw = sy * (YF + 0.002)
        yl = sy * (YF + 0.23)
        hl, wl = 0.2, 0.15
        zf = H + (0.1 if not railing else 0.2)          # flame height
        zb = zf - 0.47 * hl
        li, gl, fl = [], [], []
        bt = bolt("cl_bolt", 0.008, 0.006, kind="dome")
        if not railing:                                  # hanging from a braced arm on the parapet face
            za = zb + hl + 0.2 + 0.09
            iron.append(box("cl_plate", (0.06, 0.008, 0.36), loc=(0, sy * (YF + 0.004), za - 0.12), bevel=0.002, segments=1))
            iron.append(sweep("cl_arm", [(0, yw, za), (0, yl + sy * 0.05, za)], rounded_rect(0.022, 0.009, 0.002, 1), smooth_path=False))
            iron.append(sweep("cl_brace", bezier_path((0, yw + sy * 0.006, za - 0.27), (0, yw + sy * 0.06, za - 0.09), (0, yl - sy * 0.13, za - 0.03),
                                                      (0, yl - sy * 0.02, za - 0.012), n=8), rounded_rect(0.018, 0.007, 0.002, 1)))
            iron.append(sphere("cl_knob", 0.013, loc=(0, yl + sy * 0.055, za), segments=6, rings=4))
            iron.append(instances_to_mesh("cl_bolts", bt, [trs((0, sy * (YF + 0.008), za - 0.12 + dz_), (-90 * sy, 0, 0)) for dz_ in (-0.14, 0.0, 0.14)], keep_base=False))
            iron.append(torus("cl_eye", major=0.014, minor=0.004, loc=(0, yl, za - 0.02), rot=(0, 90, 0), major_seg=8, minor_seg=4))
            ztl = zb + hl + 0.2
            li.append(cylinder("cl_rod", 0.004, max(0.02, za - 0.03 - ztl), loc=(0, yl, (za - 0.03 + ztl) / 2), verts=5))
            lp = lantern("crown" + tg, 0, yl, zb, wl, hl, li, gl, fl)
            part(li + gl + fl, "crown_lantern_" + tg, pivot=(0, yl, za - 0.02), motion="swing", amp=1.5, speed=0.45)
        else:                                            # standing on a short arm bolted to the plinth
            za = zb - 0.03
            iron.append(box("cl_plate", (0.06, 0.008, 0.2), loc=(0, sy * (YF + 0.004), za + 0.02), bevel=0.002, segments=1))
            iron.append(sweep("cl_arm", [(0, yw, za), (0, yl + sy * 0.03, za)], rounded_rect(0.024, 0.012, 0.002, 1), smooth_path=False))
            iron.append(sweep("cl_brace", bezier_path((0, yw + sy * 0.09, za - 0.075), (0, yw + sy * 0.15, za - 0.08), (0, yl - sy * 0.07, za - 0.05),
                                                      (0, yl - sy * 0.01, za - 0.008), n=6), rounded_rect(0.018, 0.007, 0.002, 1)))
            iron.append(instances_to_mesh("cl_bolts", bt, [trs((0, sy * (YF + 0.008), za + 0.02 + dz_), (-90 * sy, 0, 0)) for dz_ in (-0.07, 0.07)], keep_base=False))
            iron.append(lathe("cl_cup", [(0.0, 0.0), (0.02, 0.0), (0.03, 0.012), (0.05, 0.024), (0.0, 0.024)], segments=6, loc=(0, yl, za + 0.006)))
            lp = lantern("crown" + tg, 0, yl, zb, wl, hl, li, gl, fl)
        parts.append((li, gl, fl))
        light("crown_" + tg, lp, intensity=float(p.intensity) * 0.1, range=6.0, shadow=True, shadow_near=0.14, flicker=0.15)
    for li, gl, fl in parts:
        for o in li:
            assign(o, m_iron)
        for o in gl:
            assign(o, m_glass)
        for o in fl:
            assign(o, m_flame)

    # =====================================================================  canal: rippled water strip over a silt bed
    if p.water:
        ye = W / 2 - 0.006
        xw = np.linspace(-hc - 0.005, hc + 0.005, nd(2 * hc, 0.3) + 1)
        yw_ = np.linspace(-ye, ye, int(round(2 * ye / 0.032)) + 1)

        def ripple(X, Y):                                # wavelets running along the canal, crests across it: they
            amp = 0.6 + 0.4 * np.sin(1.1 * X + seed) * np.sin(2 * pi * Y / 1.9 + 0.5 * X)
            return zw + 0.011 * amp * np.sin(2 * pi * Y / 0.2 + 1.0 * np.sin(0.7 * X + seed) + 0.3 * np.sin(1.9 * X + 1.0)) \
                + 0.004 * np.sin(2 * pi * Y / 0.57 + 0.6 * X + 2.0)   # draw every lamp reflection out into a streak
        wt = _sheet("water", xw, yw_, ripple)
        assign(wt, mat("water", rough=0.3))
        smooth(wt, 80)
        bed = _sheet("canal_bed", np.linspace(-hc, hc, 5), np.linspace(-ye, ye, 4), lambda X, Y: zbot + 0.03 + 0 * X)
        assign(bed, m_rub)
        paint(bed, grime=0.85, special=0.5)

    # =====================================================================  moss cushions on the damp ledges and joints
    if mk_ > 0.03:
        rm_ = rng.fork("moss")
        clump = sphere("clump", 0.5, segments=6, rings=4)
        displace_noise(clump, strength=0.09, scale=0.4, seed=seed + 2)
        mm = []

        def moss_run(a0, a1, cover, place):
            t = a0 + rm_.u(0.0, 0.3)
            while t < a1:
                if rm_.chance(cover):
                    e = min(a1, t + rm_.u(0.25, 0.9))
                    while t < e:
                        sz = rm_.u(0.08, 0.16)
                        mm.append(place(t, sz))
                        t += sz * rm_.u(0.5, 0.8)
                else:
                    t += rm_.u(0.2, 0.6)

        for sy in (-1, 1):
            shade = 1.0 if sy > 0 else 0.45                # +Y is the north, damp side
            # top of the string course
            moss_run(-xe + 0.05, xe - 0.05, min(0.9, shade * (0.25 + 0.75 * mk_)),
                     lambda t, sz: trs((t, sy * (YF + 0.05 + rm_.u(-0.012, 0.012)), deck(t) - 0.004), (0, -degrees(atan(slope(t))), rm_.u(-15, 15)),
                                       (sz * 1.5, rm_.u(0.05, 0.075), rm_.u(0.035, 0.055))))
            # foot of the parapet / gutter on the shaded inner side
            moss_run(-xe + 0.1, xe - 0.1, min(0.8, (1.45 - shade) * 0.5 * mk_),
                     lambda t, sz: trs((t, sy * (dw - 0.012), float(walk(t, dw)) + 0.0), (0, -degrees(atan(slope(t))) if not stepped else 0, rm_.u(-10, 10)),
                                       (sz * 1.4, 0.04, rm_.u(0.025, 0.04))))
        for sx in (-1, 1):
            if has_impost:                                  # impost ledges
                moss_run(-YF, YF, min(0.9, 0.3 + 0.7 * mk_),
                         lambda t, sz: trs((sx * (hc - 0.03 + rm_.u(-0.01, 0.01)), t, zs - 0.002), (0, 0, 90 + rm_.u(-12, 12)),
                                           (sz * 1.5, rm_.u(0.045, 0.07), rm_.u(0.035, 0.05))))
            if jtop > zw + 0.05:                            # waterline fringe along the jambs
                moss_run(-YF + PP, YF - PP, min(0.9, 0.35 + 0.6 * mk_),
                         lambda t, sz: trs((sx * (hc + 0.012), t, zw + rm_.u(0.01, 0.05)), (0, 0, 90), (sz * 1.5, 0.035, rm_.u(0.03, 0.06))))
            for sy in (-1, 1):
                shade = 1.0 if sy > 0 else 0.6
                xp = sx * (hs - EM - PW / 2)
                yp = sy * (YF + PP - PW / 2)
                levels = [zw + 0.015] + [zbot + i * hpc for i in range(1, npc) if zw + 0.1 < zbot + i * hpc < zw + 0.8]
                for zj in levels:
                    cov = min(0.95, shade * (1.05 - (zj - zw) / 0.8) * (0.35 + 0.65 * mk_))
                    moss_run(xp - PW / 2 + 0.03, xp + PW / 2 - 0.03, cov,
                             lambda t, sz: trs((t, sy * (YF + PP + 0.002), zj + rm_.u(-0.006, 0.006)), (0, 0, rm_.u(-6, 6)), (sz * 1.5, 0.035, rm_.u(0.025, 0.045))))
                    moss_run(yp - PW / 2 + 0.03, yp + PW / 2 - 0.03, cov,
                             lambda t, sz: trs((xp - sx * (PW / 2 + 0.002), t, zj + rm_.u(-0.006, 0.006)), (0, 0, 90 + rm_.u(-6, 6)), (sz * 1.5, 0.035, rm_.u(0.025, 0.045))))
        if mm:
            ms_ = instances_to_mesh("moss", clump, mm, keep_base=False)
            smooth(ms_, 80)
            assign(ms_, m_moss)
            paint(ms_, grime=lambda co, n: 0.25 + 0.3 * np.clip(1 - (co[:, 2] - zw) / 0.6, 0, 1))
        else:
            delete([clump])

    # =====================================================================  meshes, materials, weathering
    stones = []
    for acc, nm, m_ in ((A_d, "dressed", m_dress), (A_d2, "dressed_b", m_dress2), (A_wet, "wet_courses", m_wet), (A_rub, "rubble", m_rub),
                        (A_sett, "setts", m_rub), (A_mor, "pointing", m_mortar)):
        o = acc.obj(nm, m_, smooth_angle=55.0 if acc is A_sett else 40.0)
        if o is not None:
            stones.append((o, acc))
    for o in mortar:
        assign(o, m_mortar)
    for o in iron:
        assign(o, m_iron)
    for o in glass:
        assign(o, m_glass)
    for o in flame:
        assign(o, m_flame)

    n_big = noise_mask(0.5, seed=seed + 11, lo=0.25, hi=0.75)
    n_fine = noise_mask(0.22, seed=seed + 12, lo=0.42, hi=0.72)
    TAB = np.random.RandomState(seed + 5).rand(5, 4096)

    def streaks(u, dz, salt, p_=0.4, Lmin=0.25, Lmax=0.8, cell=0.24):
        """Run-off streaks: narrow vertical bands at random stations u, fading out below their ledge."""
        out = np.zeros(len(u))
        i0 = np.floor(u / cell).astype(np.int64)
        for di in (-1, 0, 1):
            i = i0 + di
            hsh = (i * 73 + salt * 911) & 4095
            cx = (i + 0.2 + 0.6 * TAB[1][hsh]) * cell
            w = 0.03 + 0.035 * TAB[2][hsh]
            L = Lmin + (Lmax - Lmin) * TAB[3][hsh]
            m = np.exp(-((u - cx) / w) ** 2) * np.clip(1 - dz / L, 0, 1) ** 1.2 * (dz > -0.02) * (TAB[0][hsh] < p_) * (0.5 + 0.5 * TAB[4][hsh])
            out = np.maximum(out, m)
        return out

    def stain(co):                           # run-off below the drain spouts, rust tears below the mooring rings
        out = np.zeros(len(co))
        for x0, sy_, z0, w_, L_, k_ in stains:
            m = np.exp(-((co[:, 0] - x0) / w_) ** 2) * np.clip(1 - (z0 - co[:, 2]) / L_, 0, 1) * (co[:, 2] < z0 + 0.02) * (co[:, 1] * sy_ > YF - 0.06)
            out = np.maximum(out, m * k_)
        return out

    def waterline(co, n):                    # 1 at the water, fading out over 0.4-0.7 m with a noise-broken edge
        z = co[:, 2] - zw
        endw = 1 - 0.65 * ((np.abs(co[:, 0]) > hs - 0.03) & (np.abs(n[:, 0]) > 0.7))      # bank faces are buried
        return np.clip((0.5 + 0.35 * n_big(co, n) - z) / 0.4, 0, 1) ** 0.8 * (z > -0.5) * endw

    def under(dz, k=0.06):
        return np.exp(-np.clip(dz, 0, 9) / k) * (dz > -0.015)

    def g_stone(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        vert = np.clip(1.25 - 2.0 * np.abs(n[:, 2]), 0, 1)
        u = np.where(np.abs(n[:, 1]) > np.abs(n[:, 0]), x, y + 31.7)
        dk = deck_np(x)
        out = np.zeros(len(x))
        mid = np.abs(x) < xe + 0.01
        if not railing:                      # below the coping drip, both faces of the parapet
            dzc = dk + wall_h - z
            par = mid & (np.abs(y) > YF - PT - 0.05)
            out = np.maximum(out, par * (streaks(u + 3.3 * np.sign(y) + 1.7 * np.sign(y) * np.sign(n[:, 1]), dzc, 1, 0.36, 0.3, 0.95) * 0.55 + 0.2 * under(dzc)))
        outer = mid & (np.abs(y) > YF - 0.04)
        dzs = dk - 0.16 - z                  # below the string course
        out = np.maximum(out, outer * (streaks(u + 7.7 * np.sign(y), dzs, 2, 0.22, 0.2, 0.5) * 0.4 + 0.26 * under(dzs, 0.07)) * (dzs < 0.6))
        # below the pier caps and the plinth bands (pilaster faces only)
        pier = (np.abs(x) > hc - 0.02) & (np.abs(x) < hs - EM + 0.02) & (np.abs(y) > YF + PP - PW - 0.03)
        up_ = u + 5.1 * np.sign(x) + 2.9 * np.sign(y)
        out = np.maximum(out, pier * (streaks(up_, z_pt - z, 3, 0.4, 0.25, 0.8) * 0.5 + 0.2 * under(z_pt - z)))
        out = np.maximum(out, pier * (streaks(up_, z_pl - 0.12 - z, 4, 0.45, 0.3, 1.0) * 0.5 + 0.22 * under(z_pl - 0.12 - z)))
        if has_impost:                       # below the impost ledges
            jam = (np.abs(x) < hc + 0.05) & (np.abs(n[:, 0]) > 0.6)
            out = np.maximum(out, jam * (streaks(u, zs - 0.18 - z, 5, 0.5, 0.2, 0.6) * 0.4 + 0.2 * under(zs - 0.18 - z)))
        out = out * vert
        out = np.maximum(out, stain(co))
        return np.maximum(out, 0.65 * waterline(co, n))

    def s_stone(co, n):                      # lichen on weather-struck upward faces, algae rising from the water
        up = np.clip(n[:, 2], 0, 1) ** 2
        shade = 0.5 + 0.5 * (co[:, 1] > 0)
        lich = up * n_fine(co, n) * 0.5 * shade * (0.3 + 0.7 * mk_)
        alg = waterline(co, n) * (0.7 + 0.3 * n_fine(co, n)) * (0.6 + 0.4 * mk_)
        return np.clip(np.maximum(np.maximum(lich, alg), stain(co) * 0.7 * min(1.0, mk_ * 2)), 0, 1)

    def w_stone(co, n):                      # hands polish the coping, feet the thresholds and treads
        x, z = co[:, 0], co[:, 2]
        up = np.clip(n[:, 2], 0, 1)
        cope = 0.4 * up * np.clip((z - deck_np(x) - 0.6) / 0.2, 0, 1) * (np.abs(x) < xe)
        tread = 0.5 * up * (np.abs(co[:, 1]) < dw) * (np.abs(z - walk(x, co[:, 1])) < 0.06) * np.clip(1.15 - np.abs(co[:, 1]) / (dw * 0.8), 0, 1)
        return np.maximum(cope, tread)

    for o, acc in stones:
        t = acc.tone
        if acc is A_sett:
            paint(o, wear=lambda co, n: 0.6 * np.clip(n[:, 2], 0, 1) ** 2 * np.clip(1.1 - np.abs(co[:, 1]) / (ysw * 0.8), 0, 1) + 0.1 * (t < 0.3),
                  grime=lambda co, n: np.maximum(0.4 * np.clip((np.abs(co[:, 1]) - ysw + 0.3) / 0.3, 0, 1), 0.25 * np.clip(t - 0.6, 0, 1)),
                  special=lambda co, n: 0.5 * mk_ * np.clip((np.abs(co[:, 1]) - ysw + 0.3) / 0.3, 0, 1) * n_fine(co, n))
        elif acc is A_wet:
            paint(o, grime=lambda co, n: np.maximum(0.55 * np.clip(1 - (co[:, 2] - zw) / 0.45, 0.35, 1), stain(co)),
                  special=lambda co, n: (0.25 + 0.35 * n_fine(co, n)) * (0.5 + 0.5 * mk_))
        else:
            k_ = 0.6 if acc is A_mor else 1.0
            paint(o, grime=lambda co, n: np.maximum(g_stone(co, n) * k_, 0.2 * np.clip(t - 0.55, 0, 1) / 0.45),
                  special=s_stone,
                  wear=lambda co, n: np.maximum(w_stone(co, n), 0.16 * np.clip(0.35 - t, 0, 1) / 0.35))
    paint(mortar + dressed + [gst], grime=lambda co, n: g_stone(co, n) * 0.6, special=s_stone)
    if soot:                                 # soot on the soffit above the hanging lantern
        for o, acc in stones:
            if acc in (A_d, A_d2):
                paint(o, grime=lambda co, n: 0.7 * np.exp(-(co[:, 0] ** 2 + co[:, 1] ** 2) / 0.32 ** 2) * (co[:, 2] > zci - 0.2) * (n[:, 2] < -0.5))
    io = join(iron, "ironwork")
    paint(io, special=lambda co, n: np.clip(np.maximum(1 - (co[:, 2] - zw) / 1.4, 0.12 + 0.4 * n_fine(co, n)), 0, 1) * 0.75,
          grime=0.3)
