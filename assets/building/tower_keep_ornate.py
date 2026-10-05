"""Ornate Gothic-revival guild tower, built at true size (26-36 m): a square shaft of individually
laid cream ashlar blocks (chamfered, V-jointed courses 0.3-0.4 m) with polychrome bands and long-and-
short quoins of red-brown stone, standing on a battered plinth with a gabled pointed doorcase.
Moulded string courses divide it into stages: door stage (carved shield roundel, slit windows, wall
lanterns), a tall stage with two-light stained-glass lancets under plate tracery, blind arcading
with trefoil lights, and a belfry of quatrefoil roundels.  A corbel table of small arches carries
the machicolated, crenellated parapet; octagonal bartizans are corbelled out at the corners and rise
above it; a low lead roof and a flagpole sit behind the merlons.  All four faces are finished.
"""
from tk import *
import math
import numpy as np

META = {"name": "Ornate battlemented tower", "class": "landmark", "mount": "ground", "focus": 0.27}
PARAMS = {
    "width": {"default": 8.5, "min": 7.0, "max": 10.0, "desc": "side of the square shaft (m)"},
    "height": {"default": 30.0, "min": 26.0, "max": 36.0, "desc": "ground to the top of the merlons (m)"},
    "stages": {"default": 4, "min": 3, "max": 5, "desc": "number of stages between plinth and corbel table"},
    "band_colour": {"default": "#9e573b", "desc": "colour of the banding / quoin / dressing stone"},
    "band_count": {"default": 2, "min": 0, "max": 4, "desc": "polychrome band courses per stage"},
    "stone_tint": {"default": "#f0d6a6", "desc": "tint of the ashlar"},
    "bartizans": {"default": "spire", "choices": ["spire", "battlement", "none"], "desc": "corner turret style"},
    "parapet": {"default": "capped", "choices": ["capped", "swallowtail", "stepped"], "desc": "merlon style"},
    "pierced": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "share of merlons pierced by a loop"},
    "glass": {"default": "royal", "choices": ["royal", "ember", "sea"], "desc": "stained-glass palette"},
    "window_glow": {"default": 2.2, "min": 0.4, "max": 6.0, "desc": "emissive strength of the glass"},
    "lantern_intensity": {"default": 45.0, "min": 0.0, "max": 90.0, "desc": "wall lantern intensity (cd)"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "moss / damp on plinth, ledges and the north face"},
    "pigeons": {"default": 5, "min": 0, "max": 14, "desc": "pigeons roosting on the string courses"},
    "flag": {"default": True, "desc": "flag on the roof pole"},
    "flag_colour": {"default": "#8a1f2a", "desc": "flag colour"},
}
VARIANTS = {
    "slender_ember": {"width": 7.0, "height": 34.0, "stages": 5, "bartizans": "battlement", "parapet": "swallowtail",
                      "glass": "ember", "band_colour": "#7d3f30", "band_count": 3, "flag_colour": "#c98a1e", "seed": 7},
    "squat_plain": {"width": 10.0, "height": 26.0, "stages": 3, "bartizans": "none", "parapet": "stepped",
                    "glass": "sea", "band_colour": "#8f8375", "band_count": 1, "stone_tint": "#d6d0c2", "moss": 0.9,
                    "flag": False, "seed": 13},
    "grand_banded": {"width": 9.5, "height": 36.0, "stages": 5, "bartizans": "spire", "parapet": "capped",
                     "glass": "sea", "band_colour": "#5d4a55", "band_count": 4, "stone_tint": "#efe0c4",
                     "flag_colour": "#1f3f7a", "pierced": 0.8, "seed": 21},
    "battlement_red": {"width": 8.0, "height": 28.0, "stages": 4, "bartizans": "battlement", "parapet": "stepped",
                       "glass": "ember", "band_colour": "#9a4a3a", "band_count": 4, "stone_tint": "#e8cfa6",
                       "flag_colour": "#2a5a3a", "moss": 0.15, "seed": 34},
}

PALETTES = {  # field, accent, warm
    "royal": ("#2f5be0", "#8a45d6", "#ffb14a"),
    "ember": ("#c9321f", "#ff8a2a", "#ffd872"),
    "sea": ("#1f93a8", "#3d62d8", "#ffd27a"),
}
SQ2 = math.sqrt(2.0)


def face_T(A):
    a = math.radians(A)
    c, s = math.cos(a), math.sin(a)
    return lambda x, y, z: (x * c - y * s, x * s + y * c, z)


def local_T(cx, cy, A):
    a = math.radians(A)
    c, s = math.cos(a), math.sin(a)
    return lambda x, y, z: (cx + x * c - y * s, cy + x * s + y * c, z)


def pointed(xc, half, zb, zs, cc, n=10):
    """Pointed-arch outline (x, z), CCW from bottom right; arcs struck from xc -/+ cc on the spring line."""
    Rr = half + cc
    th = math.acos(cc / Rr)
    pts = [(xc + half, zb)]
    if zs - zb > 1e-4:
        pts.append((xc + half, zs))
    for i in range(1, n + 1):
        t = th * i / n
        pts.append((xc - cc + Rr * math.cos(t), zs + Rr * math.sin(t)))
    for i in range(n - 1, 0, -1):
        t = th * i / n
        pts.append((xc + cc - Rr * math.cos(t), zs + Rr * math.sin(t)))
    if zs - zb > 1e-4:
        pts.append((xc - half, zs))
    pts.append((xc - half, zb))
    return pts


def _lobes(xc, zc, cs, rl, a0, a1, n):
    pts = []
    for i in range(n + 1):
        th = a0 + (a1 - a0) * i / n
        ux, uz = math.cos(th), math.sin(th)
        best = 0.0
        for (cx_, cz_) in cs:
            b = cx_ * ux + cz_ * uz
            disc = b * b - (cx_ * cx_ + cz_ * cz_) + rl * rl
            if disc >= 0:
                best = max(best, b + math.sqrt(disc))
        pts.append((xc + best * ux, zc + best * uz))
    return pts


def trefoil(xc, hb, zb, zh, n=30):
    rl = hb / 1.45
    cs = [(hb - rl, 0.0), (-(hb - rl), 0.0), (0.0, 0.95 * rl)]
    return [(xc + hb, zb)] + _lobes(xc, zh, cs, rl, 0.0, math.pi, n) + [(xc - hb, zb)]


def quatrefoil(xc, zc, r, n=47):
    d = 0.48 * r
    return _lobes(xc, zc, [(d, 0), (-d, 0), (0, d), (0, -d)], 0.52 * r, 0.0, 2 * math.pi * n / (n + 1), n)


def ex_circle(xc, zc, R):
    def e(z0, z1):
        if z1 <= zc - R or z0 >= zc + R:
            return None
        dz = 0.0 if z0 <= zc <= z1 else min(abs(z0 - zc), abs(z1 - zc))
        hx = math.sqrt(max(R * R - dz * dz, 0.0))
        return (xc - hx, xc + hx)
    return e


class Acc:
    """Accumulates raw verts / faces (+ a per-vertex tone) for one big hand-built mesh."""

    def __init__(self):
        self.v, self.f, self.g = [], [], []

    def obj(self, name):
        return mesh_object(name, self.v, self.f) if self.f else None


class Prisms:
    """Accumulates closed prisms (dressed blocks, voussoirs, boards) into one mesh, bevelled together."""

    def __init__(self):
        self.v, self.f = [], []

    def poly(self, T, pts, y0, y1):
        n0, m = len(self.v), len(pts)
        for (x, z) in pts:
            self.v.append(T(x, y0, z))
        for (x, z) in pts:
            self.v.append(T(x, y1, z))
        self.f.append(tuple(range(n0, n0 + m)))
        self.f.append(tuple(range(n0 + 2 * m - 1, n0 + m - 1, -1)))
        for i in range(m):
            j = (i + 1) % m
            self.f.append((n0 + i, n0 + m + i, n0 + m + j, n0 + j))

    def box(self, T, x0, x1, y0, y1, z0, z1):
        self.poly(T, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)], y0, y1)

    def xprism(self, T, prof, x0, x1, off, zo=0.0):
        """prof = [(d, z)]: d = distance out of the plane y = -off; extruded along x."""
        n0, m = len(self.v), len(prof)
        for (d, z) in prof:
            self.v.append(T(x0, -(off + d), zo + z))
        for (d, z) in prof:
            self.v.append(T(x1, -(off + d), zo + z))
        self.f.append(tuple(range(n0, n0 + m)))
        self.f.append(tuple(range(n0 + 2 * m - 1, n0 + m - 1, -1)))
        for i in range(m):
            j = (i + 1) % m
            self.f.append((n0 + i, n0 + m + i, n0 + m + j, n0 + j))

    def obj(self, name, bev=0.012):
        if not self.f:
            return None
        o = mesh_object(name, self.v, self.f)
        recalc_normals(o)
        if bev > 0:
            bevel(o, bev, segments=1)
        return o


def skin_block(acc, T, offf, xb0, xb1, xt0, xt1, z0, z1, rng, tone, ch=0.014, jd=0.016):
    """One ashlar block of the wall skin: front face + four chamfers down to the joint line."""
    proud = rng.u(0.0, 0.007)
    back = [(xb0, z0), (xb1, z0), (xt1, z1), (xt0, z1)]
    front = [(xb0 + ch, z0 + ch), (xb1 - ch, z0 + ch), (xt1 - ch, z1 - ch), (xt0 + ch, z1 - ch)]
    n = len(acc.v)
    for (x, z) in back:
        acc.v.append(T(x, -offf(z) + jd, z))
        acc.g.append(0.62)
    for (x, z) in front:
        acc.v.append(T(x, -offf(z) - proud - rng.u(0.0, 0.003), z))
        acc.g.append(tone)
    acc.f.append((n + 4, n + 5, n + 6, n + 7))
    for i in range(4):
        j = (i + 1) % 4
        acc.f.append((n + i, n + j, n + 4 + j, n + 4 + i))


def skin_courses(accfn, T, offf, zs, xext, excl, rng, L0=0.72, xk=None):
    """Lay coursed blocks between the course lines zs over [xext(z)] (or xk(k) per course), leaving the
    excluded spans open.  A cut may be (xl, xr, cw): a closer block of width cw is laid next to it."""
    for k in range(len(zs) - 1):
        z0, z1 = zs[k], zs[k + 1]
        acc = accfn(k)
        if xk is not None:
            a0, b0 = a1, b1 = xk(k)
        else:
            a0, b0 = xext(z0)
            a1, b1 = xext(z1)
        cuts = sorted(c for c in (e(z0, z1) for e in excl) if c)
        segs = []
        left = (a0, a1)
        lcw = 0.0
        for c in cuts:
            cl, cr = c[0], c[1]
            cw = c[2] if len(c) > 2 else 0.0
            if cl - max(left) > 0.1:
                segs.append((left[0], left[1], cl, cl, lcw, cw))
            if cr > max(left):
                lcw = cw
            left = (max(left[0], cr), max(left[1], cr))
        if min(b0, b1) - max(left) > 0.1:
            segs.append((left[0], left[1], b0, b1, lcw, 0.0))
        for (lb, lt, rb, rt, cwl, cwr) in segs:
            xl, xr = max(lb, lt), min(rb, rt)
            xa, xb = xl, xr
            if cwl > 0 and xb - xa > cwl + 0.25:
                xa = xl + cwl
            if cwr > 0 and xb - xa > cwr + 0.25:
                xb = xr - cwr
            Ls = xb - xa
            n = max(1, int(round(Ls / (L0 * rng.u(0.85, 1.2)))))
            if k % 2 and n >= 2:
                ts = [(j + 0.5) / n for j in range(n)]
            else:
                ts = [j / n for j in range(1, n)]
            js = [xa + Ls * clamp(t + rng.u(-0.2, 0.2) / n, 0.12 / max(Ls, 0.2), 1 - 0.12 / max(Ls, 0.2)) for t in ts]
            if xa > xl:
                js = [xa] + js
            if xb < xr:
                js = js + [xb]
            B = [(lb, lt)] + [(x, x) for x in js] + [(rb, rt)]
            for i in range(len(B) - 1):
                if B[i + 1][0] - B[i][0] < 0.06 or B[i + 1][1] - B[i][1] < 0.06:
                    continue
                tone = 0.13 * rng.u() ** 2 + (0.12 if rng.chance(0.06) else 0.0)
                skin_block(acc, T, offf, B[i][0], B[i + 1][0], B[i][1], B[i + 1][1], z0, z1, rng, tone)


def _vn1(t, seed):
    i = np.floor(t)
    f = t - i
    f = f * f * (3 - 2 * f)

    def h(k):
        v = np.sin(k * 127.1 + seed * 31.7) * 43758.5453
        return v - np.floor(v)
    return h(i) * (1 - f) + h(i + 1) * f


def build(p, rng):
    W = float(p.width)
    hw = W / 2
    H = float(p.height)
    seed = int(p.seed)
    nst = int(p.stages)
    bart = p.bartizans != "none"

    # ------------------------------------------------------------------ materials
    pal = PALETTES[p.glass]
    M = {
        "cream": mat("stone_smooth", p.stone_tint),
        "band": mat("stone_smooth", p.band_colour),
        "dress": mat("stone_smooth", "#f0e2c4"),
        "iron": mat("iron_wrought"),
        "oak": mat("wood_oak", "#5d4029"),
        "lead": mat("lead"),
    }
    MG = [mat("glass_stained", c, glow=float(p.window_glow)) for c in pal]

    # ------------------------------------------------------------------ vertical layout
    mer_h, par_h, tab_h, proj = 0.95, 1.15, 1.25, 0.5
    z_sill = H - mer_h                    # crenel sill
    z_par0 = z_sill - par_h               # parapet base / wall-walk level
    z_ct = z_par0 - tab_h                 # underside of the corbel table plate
    z_pl = clamp(0.085 * H, 2.2, 3.0)     # top of the plinth
    z_pt = z_pl - 0.3                     # top of the battered plinth face (moulding above)
    pb = 0.5                              # plinth projection at the ground
    off_p = hw + proj                     # parapet face
    Hs = z_ct - z_pl
    wts = [1.0, 1.7] + [0.95] * (nst - 3) + [0.92]
    kinds = ["door", "lancet"] + ["arcade" if i % 2 == 0 else "twin" for i in range(nst - 3)] + ["belfry"]
    zb = [z_pl]
    for w_ in wts:
        zb.append(zb[-1] + Hs * w_ / sum(wts))
    zb[-1] = z_ct
    strings = zb[1:-1]

    SK = {"cream": Acc(), "band": Acc()}
    PR = {k: Prisms() for k in ("cream", "band", "dress", "oak", "iron")}
    GL = [Acc(), Acc(), Acc()]
    FACES = (0, 90, 180, 270)
    excl = {A: [] for A in FACES}
    holes = {A: [] for A in FACES}
    objs = {k: [] for k in ("cream", "band", "dress", "iron", "oak", "lead")}

    def add(key, o):
        if o is not None:
            objs[key].append(o)
        return o

    def ring4(key, name, prof):
        """Mitred square moulding run round the tower: prof = [(offset, z)] bottom to top, outside on +offset.
        Each side is divided (not a 4-vertex lathe) so vertex weathering / AO has something to hold on to."""
        n = max(4, int(round(W / 1.05)))
        m = 4 * n
        C = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        verts, faces = [], []
        for (r, z) in prof:
            for s_ in range(4):
                a, b = C[s_], C[(s_ + 1) % 4]
                for k in range(n):
                    t = k / n
                    verts.append((r * (a[0] + (b[0] - a[0]) * t), r * (a[1] + (b[1] - a[1]) * t), z))
        for i in range(len(prof) - 1):
            for j in range(m):
                j2 = (j + 1) % m
                faces.append((i * m + j, i * m + j2, (i + 1) * m + j2, (i + 1) * m + j))
        o = mesh_object(name, verts, faces)
        smooth(o, 30)
        return add(key, o)

    # ------------------------------------------------------------------ course lines, band courses, quoin sizes
    r_sk = rng.fork("skin")
    z_end = z_ct + 0.85
    nc = max(4, int(round((z_end - z_pl) / 0.35)))
    hs = [r_sk.u(0.85, 1.15) for _ in range(nc)]
    zc = [z_pl]
    for h_ in hs:
        zc.append(zc[-1] + h_ * (z_end - z_pl) / sum(hs))
    bandset = set()
    for si, kind in enumerate(kinds):
        ks = [k for k in range(nc) if zb[si] + 0.35 <= 0.5 * (zc[k] + zc[k + 1]) < (zb[si + 1] - 0.3 if kind != "belfry" else 1e9)]
        if kind == "belfry":
            bandset.update(ks)
        elif ks and int(p.band_count) > 0:
            bc = int(p.band_count)
            for j in range(bc):
                bandset.add(ks[int((j + 0.5) / bc * len(ks))])

    kz = {round(z, 4): k for k, z in enumerate(zc)}
    r_q = rng.fork("quoins")
    QN = []
    for k in range(nc):
        row = []
        for ci in range(4):
            lx, ly = (0.8, 0.43) if (k + ci) % 2 == 0 else (0.43, 0.8)
            row.append((lx * r_q.u(0.93, 1.07), ly * r_q.u(0.93, 1.07), 0.022 + r_q.u(0, 0.008)))
        QN.append(row)
    nq = max([k + 1 for k in range(nc) if zc[k + 1] <= z_ct + 0.02] or [0])

    # ------------------------------------------------------------------ stained glass
    def glass_fill(T, x0, x1, z0, z1, y, patfn, q, r_):
        qw, qh = q
        ni = int((x1 - x0) / (qw / 2)) + 2
        nj = int((z1 - z0) / (qh / 2)) + 2
        for j in range(nj):
            for i in range(ni):
                if (i + j) % 2:
                    continue
                cx_, cz_ = x0 + i * qw / 2, z0 + j * qh / 2
                acc = GL[patfn(cx_, cz_, r_)]
                n0 = len(acc.v)
                acc.v += [T(cx_ - qw / 2, y, cz_), T(cx_, y, cz_ - qh / 2), T(cx_ + qw / 2, y, cz_), T(cx_, y, cz_ + qh / 2)]
                acc.f.append((n0, n0 + 1, n0 + 2, n0 + 3))

    def pat_small(x, z, r_):
        return r_.weighted([(2, 0.78), (1, 0.14), (0, 0.08)])

    # ------------------------------------------------------------------ pointed window with plate tracery
    def window(A, cx, zb_, w, h, lights, r_, main=True):
        T = face_T(A)
        a = w / 2
        cc = 1.5 * a
        Rr = a + cc
        rise = math.sqrt(Rr * Rr - cc * cc)
        h = max(h, rise + 0.3)
        k0 = max(1, min(range(nc), key=lambda k: abs(zc[k] - zb_)))
        k1 = max(k0 + 1, min(range(nc), key=lambda k: abs(zc[k] - (zb_ + h - rise))))
        zb_, zs_ = zc[k0], zc[k1]
        h = zs_ - zb_ + rise
        if w >= 1.6:
            d1, d2, deep, fw, pd = 0.30, 0.50, 0.44, 0.14, 0.16
        elif w >= 0.8:
            d1, d2, deep, fw, pd = 0.25, 0.40, 0.40, 0.10, 0.12
        else:
            d1, d2, deep, fw, pd = 0.18, 0.29, 0.30, 0.055, 0.08
        yf, yb = -hw - 0.04, -hw + deep
        holes[A].append(pointed(cx, a + 0.02, zb_ - 0.01, zs_, cc, n=8))
        # long-and-short jambs, coursed with the wall so the ashlar butts against every block
        wj = {}
        for k in range(k0, k1):
            for sx in (-1, 1):
                bw = (d2 if k % 2 == 0 else d1) * r_.u(0.93, 1.07)
                wj[(k, sx)] = bw
                x0, x1 = cx + sx * a, cx + sx * (a + bw)
                PR["band"].box(T, min(x0, x1), max(x0, x1), yf - r_.u(0, 0.008), yb, zc[k], zc[k + 1])
        ws = a + d2 + 0.06
        Rc = Rr + 0.4 * d1
        Ro = Rr + d2 * 1.06

        def e(z0, z1, cx=cx, a=a, k0=k0, k1=k1, wj=wj, ws=ws, zs_=zs_, Rc=Rc, Ro=Ro, cc=cc):
            k = kz.get(round(z0, 4))
            if k is None or k < k0 - 1:
                return None
            if k == k0 - 1:
                return (cx - ws + 0.004, cx + ws - 0.004)
            if k < k1:
                return (cx - a - wj[(k, -1)] + 0.004, cx + a + wj[(k, 1)] - 0.004)
            dz = z0 - zs_
            if dz >= math.sqrt(Rc * Rc - cc * cc) - 0.02:
                return None
            hx = math.sqrt(Rc * Rc - dz * dz) - cc
            ho = math.sqrt(max(Ro * Ro - dz * dz, 0.0)) - cc
            return (cx - hx, cx + hx, max(0.12, ho - hx + 0.03))
        excl[A].append(e)
        # voussoirs, alternately long and short on the extrados
        th = math.acos(cc / Rr)
        nv = max(3, int(round(Rr * th / 0.4)))
        for sx in (-1, 1):
            for k in range(nv):
                t = (d1 if k % 2 == 0 else d2) * r_.u(0.94, 1.06)
                t0 = th * k / nv
                t1 = th * (k + 1) / nv
                pin0 = (-cc + Rr * math.cos(t0), Rr * math.sin(t0))
                pout0 = (-cc + (Rr + t) * math.cos(t0), (Rr + t) * math.sin(t0))
                if k < nv - 1:
                    pin1 = (-cc + Rr * math.cos(t1), Rr * math.sin(t1))
                    pout1 = (-cc + (Rr + t) * math.cos(t1), (Rr + t) * math.sin(t1))
                else:
                    pin1 = (0.0, rise)
                    pout1 = (0.0, math.sqrt((Rr + t) ** 2 - cc * cc))
                poly = [pin0, pout0, pout1, pin1]
                if sx < 0:
                    poly = [(-x, z) for (x, z) in reversed(poly)]
                PR["band"].poly(T, [(cx + x, zs_ + z) for (x, z) in poly], yf - r_.u(0, 0.008), yb)
        # sill slab on an apron course
        PR["dress"].box(T, cx - ws, cx + ws, -hw - 0.13, yb, zb_ - 0.15, zb_)
        PR["band"].box(T, cx - ws, cx + ws, yf, -hw + 0.1, zc[k0 - 1], zb_ - 0.15)
        # pale moulded architrave on the face of the arch
        if w >= 0.8:
            add("dress", shape_extrude("architrave", pointed(cx, a + 0.09, zb_, zs_, cc, n=12),
                                       [pointed(cx, a - 0.03, zb_ + 0.01, zs_, cc, n=12)], depth=0.075,
                                       loc=T(0, -hw - 0.035, 0), rot=(90, 0, A), bevel=0.022, bevel_res=2))
        # tracery plate
        back = deep - 0.1
        yg = -hw + back - pd / 2
        oc = None
        if lights == 2:
            lw = (w - 3 * fw) / 2
            xc_ = a - fw - lw / 2
            hl = [pointed(cx + sx * xc_, lw / 2, zb_ + fw, zs_, cc + xc_, n=10) for sx in (-1, 1)]
            Rin, cl, Rl = a + cc - fw, 2 * xc_ + cc, lw / 2 + cc + xc_
            best = (0.0, 0.0)
            for i in range(90):
                zc_ = 0.3 * a + i * (2.0 * a) / 90
                r0 = min(math.hypot(cl, zc_) - Rl - fw * 0.8, Rin - math.hypot(cc, zc_))
                if r0 > best[1]:
                    best = (zc_, r0)
            oc = (cx, zs_ + best[0], best[1])
            hl.append(quatrefoil(oc[0], oc[1], oc[2]))
        else:
            lw = w - 2 * fw
            xc_ = 0.0
            hl = [pointed(cx, a - fw, zb_ + fw, zs_, cc, n=10)]
        add("dress", shape_extrude("tracery", pointed(cx, a, zb_, zs_, cc, n=12), hl, depth=pd,
                                   loc=T(0, -hw + back, 0), rot=(90, 0, A), bevel=0.012))
        # glass: diamond quarries, big figured medallions in the main lights
        P = lw * 1.75

        def pat_main(x, z, rr):
            if oc is not None:
                d = math.hypot(x - oc[0], z - oc[1]) / oc[2]
                if d < 1.05:
                    return 2 if d < 0.42 else (1 if d < 0.8 else 0)
            dx = abs(abs(x - cx) - xc_)
            if dx > lw / 2 - 0.1:
                return 0
            zz = ((z - zb_ - 0.15) % P) - P / 2
            m = dx / (0.42 * lw) + abs(zz) / (0.46 * P)
            if m < 0.42:
                return 2
            if m < 0.82:
                return 1
            if m < 1.02:
                return 2
            return rr.weighted([(0, 0.8), (1, 0.13), (2, 0.07)])
        q = (0.13, 0.19) if w >= 1.6 else ((0.11, 0.16) if w >= 0.8 else (0.08, 0.115))
        glass_fill(T, cx - a, cx + a, zb_, zb_ + h, yg, pat_main if main else pat_small, q, r_)
        # iron saddle bars across the lights
        if w >= 0.8:
            z = zb_ + 0.75
            while z < zs_ + rise * 0.4:
                PR["iron"].box(T, cx - a, cx + a, yg - 0.035, yg - 0.015, z - 0.012, z + 0.012)
                z += 0.78
        return zs_, rise

    # ------------------------------------------------------------------ belfry roundel
    def roundel(A, xc, zc_, r, r_):
        T = face_T(A)
        excl[A].append(ex_circle(xc, zc_, r + 0.2))
        holes[A].append(circle_pts(r + 0.02, 20, center=(xc, zc_)))
        add("dress", lathe("roundel_ring", [(r + 0.31, -0.03), (r + 0.30, 0.05), (r + 0.25, 0.095), (r + 0.16, 0.105), (r + 0.09, 0.08),
                                            (r + 0.045, 0.05), (r, 0.04), (r, -0.34)], segments=28, loc=T(xc, -hw, zc_), rot=(90, 0, A)))
        add("dress", shape_extrude("roundel_plate", circle_pts(r, 28), [quatrefoil(0, 0, r * 0.8)], depth=0.1,
                                   loc=T(xc, -hw + 0.26, zc_), rot=(90, 0, A), bevel=0.01))

        def pat(x, z, rr):
            d = math.hypot(x - xc, z - zc_) / r
            return 1 if d < 0.3 else rr.weighted([(2, 0.82), (0, 0.1), (1, 0.08)])
        glass_fill(T, xc - r, xc + r, zc_ - r, zc_ + r, -hw + 0.2, pat, (0.1, 0.14), r_)

    # ------------------------------------------------------------------ wall lantern on a scrolled bracket
    lights_made = []
    soot = []

    def lantern(tag, A, x, off, z, cd, reach=0.9, rng_=14.0):
        T = face_T(A)
        soot.append(T(x, -off, z + 0.7))
        PR["iron"].box(T, x - 0.055, x + 0.055, -off - 0.014, -off + 0.02, z - 0.32, z + 0.2)
        up = T(1, 0, 0)
        bar = rounded_rect(0.036, 0.014, 0.003, 2)
        path = bezier_path(T(x, -off - 0.01, z - 0.12), T(x, -off - reach * 0.55, z - 0.2), T(x, -off - reach - 0.02, z - 0.16),
                           T(x, -off - reach, z + 0.0), n=12)
        add("iron", sweep("lantern_arm", path, bar, up=up))
        add("iron", sweep("lantern_stay", [T(x, -off - 0.01, z - 0.3), T(x, -off - reach * 0.62, z - 0.17)], 0.009, sides=6))
        add("iron", sweep("lantern_tie", [T(x, -off - 0.01, z + 0.16), T(x, -off - reach * 0.5, z + 0.16), T(x, -off - reach + 0.05, z + 0.5)],
                          0.007, sides=6, smooth_path=False))
        sc = spiral_path(0.012, 0.075, 1.5, n=30, a0_deg=90, plane="YZ", center=(0, 0, 0))
        add("iron", sweep("lantern_scroll", [T(x + q[0], -off - reach * 0.36 + q[1], z - 0.04 + q[2]) for q in sc],
                          rounded_rect(0.024, 0.008, 0.002, 2), up=up, scale_fn=lambda t: 0.55 + 0.45 * t))
        L = T(x, -off - reach, z)
        rz = 45 + A
        add("iron", lathe("lantern_base", [(0.0, -0.04), (0.03, -0.03), (0.04, 0.0), (0.13, 0.015), (0.15, 0.03), (0.15, 0.05), (0.0, 0.05)],
                          segments=4, loc=L, rot=(0, 0, rz), smooth_angle=20))
        g = lathe("lantern_glass", [(0.14, 0.05), (0.21, 0.4)], segments=4, loc=L, rot=(0, 0, rz), smooth_angle=20)
        assign(g, MG[2])
        add("iron", lathe("lantern_roof", [(0.2, 0.39), (0.235, 0.4), (0.235, 0.42), (0.15, 0.49), (0.07, 0.57), (0.035, 0.62), (0.055, 0.65),
                                           (0.03, 0.69), (0.0, 0.71)], segments=4, loc=L, rot=(0, 0, rz), smooth_angle=20))
        for k in range(4):
            an = math.radians(rz + 90 * k)
            c, s = math.cos(an), math.sin(an)
            add("iron", sweep("lantern_bar", [(L[0] + 0.142 * c, L[1] + 0.142 * s, L[2] + 0.04), (L[0] + 0.213 * c, L[1] + 0.213 * s, L[2] + 0.41)],
                              0.008, sides=4))
        if cd > 0.5:
            light("lantern_" + tag, (L[0], L[1], L[2] + 0.22), intensity=cd, range=rng_, shadow_near=0.42)
        lights_made.append(tag)

    # ------------------------------------------------------------------ doorcase (front face, through the plinth)
    def doorcase(r_):
        T = face_T(0)
        a = 0.6
        z0 = 0.17
        cc = a
        Rr = a + cc
        rise = math.sqrt(Rr * Rr - cc * cc)
        zs_ = z0 + 1.42
        zap = zs_ + rise
        yF, yB = -(hw + 0.85), -(hw - 0.05)
        gw = a + 0.54
        ze = zs_ + 0.75
        zga = ze + gw * 1.15

        def e_door(z0_, z1_):
            if z0_ >= zga - 0.25:
                return None
            if z1_ <= zs_ + 0.02:
                return (-(a + 0.326), a + 0.326, 0.2)
            hx = gw if z1_ <= ze else gw * (zga - z1_) / (zga - ze)
            return (-(hx - 0.02), hx - 0.02, 0.25) if hx > 0.2 else None
        excl_pl.append(e_door)
        excl[0].append(e_door)
        nj = 4
        hj = (zs_ - z0) / nj
        for i in range(nj):
            for sx in (-1, 1):
                bw = (0.5 if i % 2 == 0 else 0.33)
                x0, x1 = sx * a, sx * (a + bw)
                PR["band"].box(T, min(x0, x1), max(x0, x1), yF - r_.u(0, 0.008), yB, z0 + i * hj, z0 + (i + 1) * hj)
        th = math.acos(cc / Rr)
        nv = 4
        for sx in (-1, 1):
            for k in range(nv):
                t = 0.3 if k % 2 == 0 else 0.42
                t0, t1 = th * k / nv, th * (k + 1) / nv
                pin0 = (-cc + Rr * math.cos(t0), Rr * math.sin(t0))
                pout0 = (-cc + (Rr + t) * math.cos(t0), (Rr + t) * math.sin(t0))
                if k < nv - 1:
                    pin1 = (-cc + Rr * math.cos(t1), Rr * math.sin(t1))
                    pout1 = (-cc + (Rr + t) * math.cos(t1), (Rr + t) * math.sin(t1))
                else:
                    pin1 = (0.0, rise)
                    pout1 = (0.0, math.sqrt((Rr + t) ** 2 - cc * cc))
                poly = [pin0, pout0, pout1, pin1]
                if sx < 0:
                    poly = [(-x, z) for (x, z) in reversed(poly)]
                PR["band"].poly(T, [(x, zs_ + z) for (x, z) in poly], yF - r_.u(0, 0.008), yB)
        # gable infill behind the arch stones + coping slabs
        arch = pointed(0, a + 0.14, zs_, zs_, cc, n=8)          # right spring ... left spring (CCW)
        infill = [(-gw, zs_)] + list(reversed(arch)) + [(gw, zs_), (gw, ze), (0, zga), (-gw, ze)]
        PR["cream"].poly(T, infill, yF + 0.045, yB)
        sl = (zga - ze) / gw
        for sx in (-1, 1):
            cop = [(sx * (gw + 0.12), ze - 0.12 * sl), (sx * (gw + 0.12), ze - 0.12 * sl + 0.16), (0, zga + 0.16), (0, zga)]
            if sx < 0:
                cop = list(reversed(cop))
            PR["dress"].poly(T, cop, yF - 0.05, yB)
        add("dress", lathe("door_finial", [(0.07, 0.0), (0.085, 0.05), (0.05, 0.09), (0.045, 0.13), (0.1, 0.19), (0.115, 0.26), (0.085, 0.34),
                                           (0.03, 0.4), (0.0, 0.42)], segments=14, loc=T(0, yF + 0.2, zga + 0.1)))
        # threshold step
        PR["dress"].box(T, -(gw + 0.12), gw + 0.12, -(hw + 1.15), -(hw + 0.4), 0.0, z0)
        # oak leaf: separate boards under the arch, strap hinges, studs, ring pull
        nb = 7
        bwd = 2 * a / nb
        yd0, yd1 = -(hw + 0.565), -(hw + 0.51)

        def ztop(x):
            return zs_ + math.sqrt(max(Rr * Rr - (abs(x) + cc) ** 2, 0.0))
        for i in range(nb):
            x0, x1 = -a + i * bwd + 0.003, -a + (i + 1) * bwd - 0.003
            poly = [(x0, z0 + 0.012), (x1, z0 + 0.012), (x1, ztop(x1))]
            if x0 < 0 < x1:
                poly.append((0.0, ztop(0.0)))
            poly.append((x0, ztop(x0)))
            PR["oak"].poly(T, poly, yd0 - r_.u(0, 0.006), yd1)
        studs = []
        for zh in (z0 + 0.42, z0 + 1.22):
            PR["iron"].box(T, -a + 0.02, a - 0.16, yd0 - 0.02, yd0, zh - 0.035, zh + 0.035)
            for i in range(6):
                studs.append(trs(T(-a + 0.08 + i * 0.17, yd0 - 0.02, zh), (90, 0, 0)))
        for zr in (z0 + 0.82, z0 + 1.62):
            for i in range(nb):
                x = -a + (i + 0.5) * bwd
                if zr < ztop(x) - 0.1:
                    studs.append(trs(T(x, yd0 - 0.004, zr), (90, 0, 0)))
        st = bolt("stud", 0.016, 0.012, kind="pyramid")
        add("iron", instances_to_mesh("door_studs", st, studs, keep_base=False))
        add("iron", torus("door_ring", 0.07, 0.009, loc=T(a - 0.2, yd0 - 0.02, z0 + 1.0), rot=(90, 0, 0), major_seg=20, minor_seg=6))
        PR["iron"].box(T, a - 0.25, a - 0.15, yd0 - 0.014, yd0, z0 + 1.0, z0 + 1.1)
        socket("door", "attach", (0, -(hw + 0.85), 0.0), role="door")
        return zga

    # ------------------------------------------------------------------ carved shield in a blind roundel
    def shield(A, cx, zc_, r):
        T = face_T(A)
        add("dress", lathe("crest_ring", [(r + 0.2, -0.02), (r + 0.19, 0.06), (r + 0.13, 0.12), (r + 0.05, 0.125), (r, 0.075),
                                          (r - 0.03, 0.035), (0.0, 0.035)], segments=32, loc=T(cx, -hw, zc_), rot=(90, 0, A)))
        s = r * 0.62
        pts = [(-s, 0.85 * s), (s, 0.85 * s), (s, 0.1 * s)]
        for i in range(1, 9):
            t = i / 9 * math.pi / 2
            pts.append((s * math.cos(t) ** 0.8, 0.1 * s - 1.15 * s * math.sin(t)))
        pts.append((0.0, -1.05 * s))
        for i in range(8, 0, -1):
            t = i / 9 * math.pi / 2
            pts.append((-s * math.cos(t) ** 0.8, 0.1 * s - 1.15 * s * math.sin(t)))
        pts.append((-s, 0.1 * s))
        add("band", shape_extrude("crest_shield", pts, depth=0.09, loc=T(cx, -hw - 0.03, zc_), rot=(90, 0, A), bevel=0.02, bevel_res=2))
        PR["dress"].box(T, cx - 0.1 * s, cx + 0.1 * s, -hw - 0.155, -hw - 0.1, zc_ - 0.72 * s, zc_ + 0.68 * s)
        PR["dress"].box(T, cx - 0.66 * s, cx + 0.66 * s, -hw - 0.155, -hw - 0.1, zc_ + 0.2 * s, zc_ + 0.4 * s)

    # ------------------------------------------------------------------ blind arcade stage
    col_m = []
    col_h = [1.0]

    def arcade(A, z0, z1, r_):
        T = face_T(A)
        na = max(3, int(round((W - 1.6) / 1.3)))
        span = W - 1.5
        bp = span / na
        half = (bp - 0.34) / 2
        cc = half
        rise = math.sqrt((half + cc) ** 2 - cc * cc)
        zb_, zt_ = z0 + 0.2, z1 - 0.16
        zbay = zb_ + 0.34
        zs_ = max(zbay + 0.4, zt_ - 0.5 - rise)
        hl = [pointed(-span / 2 + (i + 0.5) * bp, half, zbay, zs_, cc, n=8) for i in range(na)]
        add("dress", shape_extrude("arcade", [(-span / 2, zb_), (span / 2, zb_), (span / 2, zt_), (-span / 2, zt_)], hl, depth=0.16,
                                   loc=T(0, -hw, 0), rot=(90, 0, A), bevel=0.012))
        for i in range(na):
            xc = -span / 2 + (i + 0.5) * bp
            ring = pointed(xc, half + 0.1, zs_, zs_, cc, n=8) + list(reversed(pointed(xc, half + 0.015, zs_, zs_, cc, n=8)))
            add("band", shape_extrude("archivolt", ring, depth=0.045, loc=T(0, -hw - 0.16, 0), rot=(90, 0, A), bevel=0.012))
        PR["dress"].box(T, -span / 2 - 0.03, span / 2 + 0.03, -hw - 0.34, -hw - 0.1, zbay - 0.11, zbay)
        col_h[0] = zs_ - zbay
        for i in range(na + 1):
            x = -span / 2 + i * bp
            x = clamp(x, -span / 2 + 0.1, span / 2 - 0.1)
            col_m.append(trs(T(x, -hw - 0.235, zbay)))
            PR["dress"].box(T, x - 0.13, x + 0.13, -hw - 0.35, -hw - 0.12, zs_, zs_ + 0.1)
        for i in range(na):
            lit = (i % 2 == 1) if na % 2 else (i in (na // 2 - 1, na // 2))
            if not lit:
                continue
            xc = -span / 2 + (i + 0.5) * bp
            hb = half * 0.5
            zl0 = zbay + min(0.45, (zs_ - zbay) * 0.25)
            zh = zs_ - 0.05
            add("band", shape_extrude("arcade_light", pointed(xc, half, zbay, zs_, cc, n=8), [trefoil(xc, hb, zl0, zh)], depth=0.07,
                                      loc=T(0, -hw - 0.012, 0), rot=(90, 0, A), bevel=0.01))
            glass_fill(T, xc - hb, xc + hb, zl0, zh + 1.36 * hb, -hw - 0.045, pat_small, (0.085, 0.12), r_)

    # ================================================================== programme of the four faces
    excl_pl = []
    r_win = rng.fork("windows")
    zga = doorcase(rng.fork("door"))
    w_main = clamp(0.26 * W, 1.8, 2.6)
    for si, kind in enumerate(kinds):
        z0, z1 = zb[si], zb[si + 1]
        sh = z1 - z0
        for A in FACES:
            if kind == "door":
                zl = z_pl + 0.95
                if A == 0:
                    zc_ = max(zga + 1.2, min(z0 + sh * 0.72, z1 - 1.35))
                    shield(A, 0.0, min(zc_, z1 - 1.1), clamp(sh * 0.14, 0.45, 0.8))
                    lx = min(hw - 1.25, 2.45)
                    for sx in (-1, 1):
                        lantern("door_%s" % ("l" if sx < 0 else "r"), A, sx * lx, hw, zl, float(p.lantern_intensity) * 0.85)
                elif A == 180:
                    for sx in (-1, 1):
                        window(A, sx * hw * 0.42, z_pl + sh * 0.3, 0.42, min(1.9, sh * 0.5), 1, r_win, main=False)
                else:
                    window(A, 0.0, z_pl + sh * 0.3, 0.42, min(1.9, sh * 0.5), 1, r_win, main=False)
            elif kind == "lancet":
                hmain = min(sh - 1.75, 4.3 * w_main)
                if A in (0, 180):
                    window(A, 0.0, z0 + 1.0, w_main, hmain, 2, r_win)
                else:
                    window(A, 0.0, z0 + 1.0, w_main * 0.58, hmain * 0.9, 1, r_win)
            elif kind == "twin":
                for sx in (-1, 1):
                    window(A, sx * W * 0.21, z0 + 0.95, 0.95, min(sh - 1.75, 3.6), 1, r_win)
            elif kind == "arcade":
                arcade(A, z0, z1, r_win)
            elif kind == "belfry":
                nr = max(2, int((W - 2.6) / 1.6))
                sp = (W - 2.2) / nr
                r = min(0.58, sh * 0.15)
                for i in range(nr):
                    roundel(A, (i - (nr - 1) / 2) * sp, z0 + sh * 0.56, r, r_win)
                lantern("upper_%d" % A, A, 0.0, hw, z0 + 0.62, float(p.lantern_intensity) * 1.2, reach=0.95, rng_=16.0)
    if col_m:
        hc = col_h[0]
        col = lathe("colonnette", [(0.0, 0.0), (0.1, 0.0), (0.1, 0.05), (0.085, 0.07), (0.085, 0.1), (0.066, 0.13), (0.06, 0.16), (0.057, hc - 0.2),
                                   (0.07, hc - 0.185), (0.07, hc - 0.16), (0.06, hc - 0.14), (0.085, hc - 0.06), (0.105, hc - 0.02), (0.105, hc)],
                    segments=10)
        add("band", instances_to_mesh("colonnettes", col, col_m, keep_base=False))

    # ================================================================== masonry skin
    def shaft_acc(k):
        return SK["band"] if k in bandset else SK["cream"]
    xs = hw - 0.3
    for fi, A in enumerate(FACES):
        def xk(k, fi=fi):
            if k >= nq:
                return (-xs, xs)
            ql, qr = QN[k][fi], QN[k][(fi + 1) % 4]
            leg = fi % 2
            return (-(hw + ql[2] - ql[leg]) - 0.004, hw + qr[2] - qr[leg] + 0.004)
        skin_courses(shaft_acc, face_T(A), lambda z: hw, zc, None, excl[A], r_sk, xk=xk)
        core = shape_extrude("core", [(-xs - 0.05, z_pl - 0.1), (xs + 0.05, z_pl - 0.1), (xs + 0.05, z_end), (-xs - 0.05, z_end)], holes[A],
                             depth=0.02, loc=face_T(A)(0, -hw + 0.045, 0), rot=(90, 0, A))
        assign(core, M["cream"])
        paint(core, grime=0.8)
    # quoins: long-and-short dressed blocks wrapping every corner
    for k in range(nq):
        for ci, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            lx, ly, pr = QN[k][ci]
            xa, xb_ = sorted((sx * (hw + pr), sx * (hw + pr - lx)))
            ya, yb_ = sorted((sy * (hw + pr), sy * (hw + pr - ly)))
            PR["band"].box(face_T(0), xa, xb_, ya, yb_, zc[k], zc[k + 1])

    # plinth: battered face of big blocks on a vertical foot course, weathered moulding on top
    def off_pl(z):
        return hw + pb - (pb - 0.12) * clamp((z - 0.42) / (z_pt - 0.42), 0.0, 1.0)
    npc = max(3, int(round((z_pt - 0.42) / 0.46)))
    zpc = [0.0, 0.42] + [0.42 + (z_pt - 0.42) * (i + 1) / npc for i in range(npc)]
    r_pl = rng.fork("plinth")
    for A in FACES:
        skin_courses(lambda k: SK["cream"], face_T(A), off_pl, zpc, lambda z: (-off_pl(z), off_pl(z)),
                     excl_pl if A == 0 else [], r_pl, L0=0.95)
    ring4("dress", "plinth_mould", [(hw + 0.09, z_pt - 0.03), (hw + 0.19, z_pt), (hw + 0.215, z_pt + 0.04), (hw + 0.215, z_pt + 0.09),
                                    (hw + 0.17, z_pt + 0.13), (hw + 0.12, z_pt + 0.15), (hw + 0.11, z_pt + 0.2), (hw + 0.035, z_pl + 0.03),
                                    (hw - 0.04, z_pl + 0.05)])
    # string courses between the stages
    for z in strings:
        ring4("dress", "string", [(hw - 0.03, z - 0.17), (hw + 0.04, z - 0.16), (hw + 0.045, z - 0.10), (hw + 0.075, z - 0.07),
                                  (hw + 0.135, z - 0.035), (hw + 0.175, z - 0.02), (hw + 0.185, z + 0.03), (hw + 0.17, z + 0.06),
                                  (hw + 0.05, z + 0.17), (hw - 0.03, z + 0.19)])

    # ================================================================== corbel table, parapet, merlons
    rb = clamp(0.1 * W, 0.74, 0.95)
    cb = hw + 0.05
    St = 2 * (hw - 0.55) if bart else 2 * off_p - 0.004
    nbay = max(4, int(round(St / 0.82)))
    bp = St / nbay
    pw = 0.22
    ra = (bp - pw) / 2
    ys = 0.16
    tpts = [(-St / 2, 0.0)]
    for i in range(nbay):
        xa = -St / 2 + i * bp + pw / 2
        xb_ = xa + 2 * ra
        tpts += [(xa, 0.0), (xa, ys)]
        for k in range(1, 8):
            t = math.pi - math.pi * k / 8
            tpts.append((xa + ra + ra * math.cos(t), ys + ra * math.sin(t)))
        tpts += [(xb_, ys), (xb_, 0.0)]
    tpts += [(St / 2, 0.0), (St / 2, 0.8), (-St / 2, 0.8)]
    cprof = [(-0.04, -0.6), (0.09, -0.57), (0.15, -0.45), (0.15, -0.37), (0.27, -0.31), (0.33, -0.19), (0.33, -0.11), (0.43, -0.07),
             (0.47, 0.0), (-0.04, 0.0)]
    for A in FACES:
        T = face_T(A)
        add("dress", extrude_profile("corbel_table", tpts, proj, loc=T(0, -hw, z_ct), rot=(90, 0, A), bevel=0.012))
        for i in range(nbay + 1):
            x = clamp(-St / 2 + i * bp, -St / 2 + pw / 2, St / 2 - pw / 2)
            PR["dress"].xprism(T, cprof, x - pw / 2 + 0.012, x + pw / 2 - 0.012, hw, z_ct)
    ring4("dress", "cornice", [(off_p - 0.04, z_ct + 0.79), (off_p + 0.035, z_ct + 0.81), (off_p + 0.04, z_ct + 0.88), (off_p + 0.06, z_ct + 0.94),
                               (off_p + 0.11, z_ct + 1.0), (off_p + 0.145, z_ct + 1.03), (off_p + 0.15, z_ct + 1.12), (off_p + 0.13, z_ct + 1.16),
                               (off_p + 0.03, z_par0), (off_p - 0.04, z_par0 + 0.01)])
    r_par = rng.fork("parapet")
    zpp = [z_par0 + par_h * i / 3 for i in range(4)]
    for A in FACES:
        skin_courses(lambda k: SK["band"], face_T(A), lambda z: off_p, zpp, lambda z: (-off_p, off_p), [], r_par, L0=0.62)
    ring4("dress", "parapet_coping", [(off_p - 0.005, z_sill - 0.08), (off_p + 0.04, z_sill - 0.07), (off_p + 0.04, z_sill - 0.02),
                                      (off_p + 0.005, z_sill + 0.005), (off_p - 0.44, z_sill + 0.005), (off_p - 0.44, z_par0 + 0.03),
                                      (0.4, z_par0 + 0.03)])
    for A in FACES:
        T = face_T(A)
        for sx in (-1, 1):
            xsp = sx * (hw * 0.5 + r_par.u(-0.05, 0.05))
            add("lead", sweep("spout", [T(xsp, -off_p + 0.12, z_par0 + 0.13), T(xsp, -off_p - 0.3, z_par0 + 0.1), T(xsp, -off_p - 0.56, z_par0 + 0.02)],
                              rounded_rect(0.13, 0.085, 0.02, 2), up=T(1, 0, 0), smooth_path=False, scale_fn=lambda t: 1.0 - 0.3 * t))
            PR["dress"].xprism(T, [(0.0, -0.2), (0.1, -0.16), (0.22, -0.05), (0.22, 0.0), (0.0, 0.0)], xsp - 0.1, xsp + 0.1, off_p, z_par0 + 0.07)
    Sm = 2 * (hw - 0.6) if bart else 2 * (off_p - 0.97)
    nm = max(2, int(round((Sm - 0.62) / 1.62)))
    sc_ = Sm / (nm * 1.0 + (nm + 1) * 0.62)
    mw, cw = 1.0 * sc_, 0.62 * sc_
    mth = 0.42
    if p.parapet == "capped":
        mh = mer_h - 0.15
        mo = [(-mw / 2, 0), (mw / 2, 0), (mw / 2, mh), (-mw / 2, mh)]
    elif p.parapet == "swallowtail":
        mh = mer_h
        mo = [(-mw / 2, 0), (mw / 2, 0), (mw / 2 + 0.03, mh), (0.0, mh - 0.36 * mw), (-mw / 2 - 0.03, mh)]
    else:
        mh = mer_h - 0.08
        mo = [(-mw / 2, 0), (mw / 2, 0), (mw / 2, mh * 0.56), (mw / 4, mh * 0.56), (mw / 4, mh), (-mw / 4, mh), (-mw / 4, mh * 0.56),
              (-mw / 2, mh * 0.56)]
    slot = [(x, y + mh * 0.34) for (x, y) in rounded_rect(0.085, mh * 0.42, 0.04, 3)]
    m_plain = shape_extrude("merlon_a", mo, depth=mth, bevel=0.012)
    m_loop = shape_extrude("merlon_b", mo, [slot], depth=mth, bevel=0.012)
    subdivide(m_plain, 1)
    subdivide(m_loop, 1)
    mm = {False: [], True: []}
    cap_prof = [(-0.045, 0.0), (mth + 0.045, 0.0), (mth + 0.045, 0.05), (mth / 2, 0.15), (-0.045, 0.05)]
    for A in FACES:
        T = face_T(A)
        for i in range(nm):
            x = -Sm / 2 + cw + mw / 2 + i * (mw + cw)
            mm[r_par.chance(float(p.pierced))].append(trs(T(x, -(off_p - mth), z_sill), (90, 0, A)))
            if p.parapet == "capped":
                PR["dress"].xprism(T, cap_prof, x - mw / 2 - 0.04, x + mw / 2 + 0.04, off_p - mth, z_sill + mh)
            elif p.parapet == "stepped":
                PR["dress"].box(T, x - mw / 4 - 0.035, x + mw / 4 + 0.035, -off_p - 0.035, -(off_p - mth) + 0.035, z_sill + mh, z_sill + mh + 0.08)
    if mm[False]:
        add("band", instances_to_mesh("merlons", m_plain, mm[False], keep_base=False))
    else:
        delete([m_plain])
    if mm[True]:
        add("band", instances_to_mesh("merlons_looped", m_loop, mm[True], keep_base=False))
    else:
        delete([m_loop])

    # ================================================================== bartizans (or corner piers)
    for ci, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        if not bart:
            x0, x1 = sorted((sx * (off_p + 0.02), sx * (off_p - 0.95)))
            y0, y1 = sorted((sy * (off_p + 0.02), sy * (off_p - 0.95)))
            PR["band"].box(face_T(0), x0, x1, y0, y1, z_sill, H - 0.06)
            add("dress", lathe("pier_cap", [(0.74, H - 0.07), (0.75, H - 0.01), (0.68, H + 0.03), (0.12, H + 0.5), (0.16, H + 0.58), (0.1, H + 0.68),
                                            (0.0, H + 0.72)], segments=4, loc=(sx * (off_p - 0.465), sy * (off_p - 0.465), 0), rot=(0, 0, 45),
                               smooth_angle=30))
            continue
        loc = (sx * cb, sy * cb, 0.0)
        zk0, zk1 = z_ct - 2.6, z_ct - 0.35
        hk = zk1 - zk0
        prof = [(0.0, zk0 - 0.32), (0.07, zk0 - 0.29), (0.12, zk0 - 0.2), (0.08, zk0 - 0.1), (0.1, zk0)]
        for k in range(5):
            r1 = 0.1 + (rb + 0.05 - 0.1) * ((k + 1) / 5) ** 0.85
            za = zk0 + hk * k / 5
            prof += [(prof[-1][0] + 0.012, za + 0.04), (r1 - 0.05, za + hk / 5 * 0.5), (r1, za + hk / 5 * 0.66), (r1, za + hk / 5 - 0.03)]
        prof += [(rb + 0.05, zk1 + 0.02), (rb - 0.02, zk1 + 0.07)]
        add("dress", lathe("bartizan_corbel", prof, segments=8, loc=loc, rot=(0, 0, 22.5), smooth_angle=30))
        z_bt = H + 1.25
        sp_ = []
        z = zk1
        while z < z_bt - 0.2:
            hc_ = min(0.37, z_bt - z)
            sp_ += [(rb, z + 0.012), (rb, z + hc_ - 0.012), (rb - 0.013, z + hc_)]
            z += hc_
        add("band", lathe("bartizan_shaft", sp_, segments=8, loc=loc, rot=(0, 0, 22.5), smooth_angle=30))
        add("dress", lathe("bartizan_band", [(rb - 0.02, z_ct + 0.74), (rb + 0.08, z_ct + 0.8), (rb + 0.1, z_ct + 0.95), (rb + 0.1, z_par0 - 0.03),
                                             (rb + 0.03, z_par0 + 0.06), (rb - 0.02, z_par0 + 0.08)], segments=8, loc=loc, rot=(0, 0, 22.5),
                           smooth_angle=30))
        # lit slits on the outward flats of the turret room
        ap = rb * math.cos(math.radians(22.5))
        for (dx, dy) in ((sx, 0), (0, sy), (sx, sy)):
            Ad = math.degrees(math.atan2(dy, dx)) + 90.0
            T = local_T(loc[0], loc[1], Ad)
            s0, s1 = H - 0.35, H + 0.45
            for jx in (-1, 1):
                x0, x1 = sorted((jx * 0.065, jx * 0.15))
                PR["dress"].box(T, x0, x1, -(ap + 0.05), -(ap - 0.03), s0, s1)
            PR["dress"].box(T, -0.19, 0.19, -(ap + 0.07), -(ap - 0.03), s0 - 0.09, s0)
            PR["dress"].poly(T, [(-0.15, s1), (0.15, s1), (0.15, s1 + 0.05), (0.0, s1 + 0.2), (-0.15, s1 + 0.05)], -(ap + 0.05), -(ap - 0.03))
            acc = GL[2]
            n0 = len(acc.v)
            acc.v += [T(-0.07, -(ap + 0.012), s0), T(0.07, -(ap + 0.012), s0), T(0.07, -(ap + 0.012), s1), T(-0.07, -(ap + 0.012), s1)]
            acc.f.append((n0, n0 + 1, n0 + 2, n0 + 3))
        if p.bartizans == "spire":
            tp = [(rb - 0.02, z_bt - 0.14), (rb + 0.07, z_bt - 0.1), (rb + 0.1, z_bt - 0.02), (rb + 0.1, z_bt + 0.05), (rb + 0.03, z_bt + 0.1),
                  (rb - 0.04, z_bt + 0.2)]
            hsp = 2.3
            for i in range(1, 9):
                t = i / 8
                tp.append(((rb - 0.04) * (1 - t) ** 1.25 + 0.075 * t, z_bt + 0.2 + hsp * t))
            zt = z_bt + 0.2 + hsp
            tp += [(0.06, zt + 0.03), (0.13, zt + 0.09), (0.15, zt + 0.17), (0.11, zt + 0.26), (0.05, zt + 0.31), (0.07, zt + 0.37), (0.0, zt + 0.48)]
            add("dress", lathe("bartizan_spire", tp, segments=8, loc=loc, rot=(0, 0, 22.5), smooth_angle=30))
        else:
            add("dress", lathe("bartizan_cornice", [(rb - 0.02, z_bt - 0.14), (rb + 0.07, z_bt - 0.1), (rb + 0.1, z_bt - 0.02), (rb + 0.1, z_bt + 0.06),
                                                    (rb + 0.04, z_bt + 0.09), (rb + 0.04, z_bt + 0.3), (rb - 0.2, z_bt + 0.3), (rb - 0.2, z_bt + 0.12),
                                                    (0.0, z_bt + 0.34)], segments=8, loc=loc, rot=(0, 0, 22.5), smooth_angle=30))
            apm = (rb + 0.04) * math.cos(math.radians(22.5))
            for k in range(8):
                T = local_T(loc[0], loc[1], k * 45.0)
                PR["band"].box(T, -0.17, 0.17, -apm, -(apm - 0.22), z_bt + 0.28, z_bt + 0.68)
                PR["dress"].xprism(T, [(-0.03, 0.0), (0.25, 0.0), (0.25, 0.03), (0.11, 0.1), (-0.03, 0.03)], -0.2, 0.2, apm - 0.22, z_bt + 0.68)

    # ================================================================== roof, flagpole, flag
    hb_ = off_p - 0.44 - 0.3
    z_r0 = z_par0 + 0.03
    rr_ = hb_ * 0.36
    roof = cone("roof", hb_ * SQ2, 0.16 * SQ2, rr_, loc=(0, 0, z_r0 + rr_ / 2), rot=(0, 0, 45), verts=4)
    add("lead", roof)
    for A in FACES:
        T = face_T(A)
        add("lead", sweep("roof_hip", [T(hb_, -hb_, z_r0 + 0.01), T(0.16, -0.16, z_r0 + rr_)], 0.05, sides=6))
        nroll = int(hb_ / 0.85)
        for i in range(-nroll, nroll + 1):
            x = i * 0.85
            if hb_ - abs(x) < 0.7:
                continue
            ye = max(abs(x) + 0.06, 0.2)
            add("lead", sweep("roof_roll", [T(x, -hb_, z_r0 + 0.012), T(x, -ye, z_r0 + 0.012 + rr_ * (1 - (ye - 0.16) / (hb_ - 0.16)))], 0.032, sides=6))
    add("lead", lathe("roof_gutter", [(hb_ + 0.3, z_r0 + 0.1), (hb_ + 0.28, z_r0 + 0.02), (hb_ + 0.02, z_r0), (hb_ - 0.02, z_r0 + 0.02)], segments=4,
                      rot=(0, 0, 45), smooth_angle=30))
    z_ap = z_r0 + rr_
    pole_h = 5.8
    add("iron", cylinder("flagpole", 0.075, pole_h, loc=(0, 0, z_ap + pole_h / 2 - 0.2), verts=10, radius2=0.04))
    add("lead", lathe("pole_collar", [(0.26, z_ap - 0.12), (0.2, z_ap + 0.02), (0.11, z_ap + 0.12), (0.1, z_ap + 0.3), (0.075, z_ap + 0.34)], segments=10))
    z_ft = z_ap + pole_h - 0.2
    add("dress", lathe("pole_ball", [(0.035, z_ft - 0.02), (0.07, z_ft + 0.0), (0.05, z_ft + 0.04), (0.11, z_ft + 0.1), (0.13, z_ft + 0.19),
                                     (0.09, z_ft + 0.28), (0.0, z_ft + 0.31)], segments=12))
    if p.flag:
        FL, FH = 2.5, 1.55

        def flag_fn(u, v):
            wv = math.sin(u * 8.5 + v * 1.7 + seed) * 0.17 * (0.2 + u) + math.sin(u * 3.1 - v * 2.3) * 0.16 * u
            return (0.06 + u * FL * (1 - 0.05 * v), wv, z_ft - 0.12 - v * FH * (1 - 0.06 * u) - 0.55 * u * u * (0.35 + v))
        fl = surface("flag", flag_fn, 22, 10, thickness=0.004)
        assign(fl, mat("canvas", p.flag_colour, stripe=0.52, tint2="#e9dcc0"))
        paint(fl, special=0.25, grime=0.1)
        part([fl], "flag", pivot=(0, 0, z_ft - 0.1), motion="flutter", amp=0.12, speed=0.7)
    # pigeons roosting on the weathered ledges
    npg = int(p.pigeons)
    if npg > 0:
        r_pg = rng.fork("pigeons")
        body = lathe("pg_body", [(0.0, -0.13), (0.03, -0.12), (0.052, -0.07), (0.062, 0.0), (0.056, 0.06), (0.04, 0.105), (0.03, 0.14), (0.0, 0.15)],
                     segments=8, loc=(0, 0, 0.075), rot=(0, 65, 0))
        head = sphere("pg_head", 0.034, loc=(0.14, 0, 0.155), segments=8, rings=5)
        beak = cone("pg_beak", 0.012, 0.0, 0.035, loc=(0.18, 0, 0.15), rot=(0, 90, 0), verts=5)
        tail = box("pg_tail", (0.13, 0.055, 0.014), loc=(-0.16, 0, 0.03), rot=(0, -12, 0))
        pg = join([body, head, beak, tail], "pigeon")
        pm = []
        xlast = None
        for i in range(npg):
            A = FACES[int(r_pg.u(0, 3.999))]
            zs_ = strings[min(len(strings) - 1, int(r_pg.u(0, 0.999) ** 1.6 * len(strings)))]
            if xlast is not None and r_pg.chance(0.4):
                x = xlast[1] + r_pg.u(0.22, 0.4)
                A, zs_ = xlast[0], xlast[2]
            else:
                x = r_pg.u(0.7, hw - 0.9) * (1 if r_pg.chance(0.5) else -1)
            xlast = (A, x, zs_)
            yaw = r_pg.weighted([(0.0, 0.3), (180.0, 0.3), (-90.0, 0.4)]) + r_pg.u(-30, 30)
            sc = r_pg.u(0.9, 1.12)
            pm.append(trs(face_T(A)(x, -(hw + 0.105), zs_ + 0.1), (0, 0, A + yaw), (sc, sc, sc)))
        add("lead", instances_to_mesh("pigeons", pg, pm, keep_base=False))
    socket("banner_front", "attach", (0, -hw - 0.2, zb[1] - 0.4), role="banner")
    socket("banner_back", "attach", (0, hw + 0.2, zb[1] - 0.4), rot=(0, 0, 180), role="banner")

    # ================================================================== realise accumulators, assign, weather
    ledges = list(strings) + [z_ct + 0.3, z_pl]

    def grime_fn(pre=None):
        pre_a = None if pre is None else np.asarray(pre, dtype=np.float64)

        def fn(co, n):
            z = co[:, 2]
            u = co[:, 0] * 0.83 + co[:, 1] * 1.31
            out = np.clip(1.0 - z / 2.4, 0, 1) * 0.5
            st = np.zeros(len(z))
            for zl_ in ledges:
                t = (zl_ - 0.12 - z) / 2.6
                st = np.maximum(st, np.where((t > 0) & (t < 1), (1 - np.clip(t, 0, 1)) ** 1.6, 0.0))
            out = np.maximum(out, st * (0.04 + 0.34 * _vn1(u * 2.3, seed) * _vn1(u * 0.6 + 7.0, seed + 3)))
            for (sx_, sy_, sz_) in soot:
                dh = np.hypot(co[:, 0] - sx_, co[:, 1] - sy_)
                dz = z - sz_
                out = np.maximum(out, 0.42 * np.exp(-(dh / 0.5) ** 2) * np.clip(dz / 0.3 + 1.0, 0, 1) * np.clip(1 - dz / 2.4, 0, 1))
            if pre_a is not None and len(pre_a) == len(z):
                out = out + pre_a
            return np.clip(out, 0, 1)
        return fn
    nz_ = noise_mask(1.1, seed=seed + 5, lo=0.3, hi=0.7)
    nz2 = noise_mask(0.23, seed=seed + 9, lo=0.25, hi=0.75)

    def moss_fn(co, n):
        z = co[:, 2]
        low = np.clip(1.0 - z / 1.7, 0, 1) * 0.75
        north = np.clip(n[:, 1], 0, 1) * np.clip(1.0 - z / (0.5 * H), 0, 1) * 0.8
        up = np.clip(n[:, 2], 0, 1) * (0.25 + 0.5 * np.clip(1.0 - z / (0.4 * H), 0, 1))
        return np.clip(np.maximum(np.maximum(low, north), up) * nz_(co, n) * (0.5 + 0.5 * nz2(co, n)) * float(p.moss) * 1.5, 0, 1)

    for key in ("cream", "band"):
        o = SK[key].obj("ashlar_" + key)
        if o is not None:
            assign(o, M[key])
            paint(o, grime=grime_fn(SK[key].g), special=moss_fn)
    for key, bev in (("cream", 0.012), ("band", 0.012), ("dress", 0.012), ("oak", 0.004), ("iron", 0.0025)):
        add(key, PR[key].obj("blocks_" + key, bev))
    for i in range(3):
        g = GL[i].obj("glass_%d" % i)
        if g is not None:
            assign(g, MG[i])
    for key in ("cream", "band", "dress"):
        for o in objs[key]:
            assign(o, M[key])
            paint(o, grime=grime_fn(), special=moss_fn)
    for o in objs["oak"]:
        assign(o, M["oak"])
        uv_grain(o, axis="Z")
        paint(o, grime=height_ramp(0.0, 0.9), wear=mask_mul(height_ramp(1.3, 0.9), height_ramp(1.3, 1.7), 0.5))
    for o in objs["iron"]:
        assign(o, M["iron"])
        paint(o, special=mask_mul(noise_mask(0.12, seed=seed + 2, lo=0.35, hi=0.75), 0.6))
    for o in objs["lead"]:
        assign(o, M["lead"])
        paint(o, grime=mask_mul(noise_mask(0.8, seed=seed + 4), 0.4))
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    rows = {}
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        try:
            me = o.evaluated_get(dg).to_mesh()
            n = sum(len(pl.vertices) - 2 for pl in me.polygons)
        except Exception as ex:
            n = -1
        k = o.name.split(".")[0]
        rows[k] = rows.get(k, 0) + n
    for k, n in sorted(rows.items(), key=lambda kv: -kv[1])[:28]:
        print("DBG tris", k, n)
