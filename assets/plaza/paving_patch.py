"""Decorative paving inlay for plaza centres and thresholds. Every stone is a separate, individually laid piece
(own outline, eased arrises, slightly domed and tilted top, sand-filled joints a few millimetres below the arrises):
concentric rows of granite setts, sawn flag segments (compass star points, lettered flag ring, medallion), a
dressed kerb ring whose outer edge rolls down to the surrounding paving, and a cast bronze compass boss / stone
medallion / iron drain grate in the middle. Setts are cut to the lines of the star or the rosette petals the way
a paviour does it. Almost flat: everything lies between z = 0 and z = 30 mm.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, degrees, atan2, hypot, sqrt, ceil

META = {"name": "Paving feature patch", "class": "ground", "mount": "ground", "focus": [0.0, -0.35, 0.02]}
PARAMS = {
    "diameter": {"default": 5.0, "min": 2.0, "max": 12.0, "desc": "overall diameter (m); side length of the herringbone square"},
    "pattern": {"default": "compass", "choices": ["rosette", "compass", "concentric", "herringbone"],
                "desc": "compass star of flags / petal rosette of coloured setts / concentric rings / herringbone square"},
    "points": {"default": 8, "min": 4, "max": 16, "desc": "compass star points or rosette petals"},
    "sett": {"default": 0.11, "min": 0.08, "max": 0.2, "desc": "nominal sett size (m); grows a little on very large patches"},
    "joint": {"default": 0.009, "min": 0.004, "max": 0.02, "desc": "joint width (m)"},
    "kerb": {"default": 0.2, "min": 0.12, "max": 0.4, "desc": "width of the kerb ring / border (m)"},
    "centre": {"default": "bronze", "choices": ["bronze", "stone", "grate"], "desc": "centre piece: bronze boss, stone medallion or drain grate"},
    "letters": {"default": "NESW", "desc": "compass letters (north, east, south, west) set in bronze in the flag ring; empty = none"},
    "unevenness": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "settlement: height / tilt differences between stones, sunken setts"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "foot polish on the crowns and along the walking lines"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "moss in the joints and on the damp side"},
    "tint_main": {"default": "#a0978a", "desc": "field setts (granite)"},
    "tint_accent": {"default": "#8c5946", "desc": "accent setts (porphyry)"},
    "tint_dark": {"default": "#4a4a4d", "desc": "dark setts and dark flags (basalt)"},
    "tint_flag": {"default": "#b9ac92", "desc": "light sawn flags (limestone)"},
    "tint_kerb": {"default": "#a9a295", "desc": "kerb stones"},
}
VARIANTS = {
    "rosette": {"pattern": "rosette", "diameter": 4.2, "points": 8, "centre": "stone", "sett": 0.096, "seed": 4},
    "concentric_drain": {"pattern": "concentric", "diameter": 3.0, "centre": "grate", "sett": 0.12, "moss": 0.7, "unevenness": 0.8,
                         "tint_main": "#a39c92", "tint_accent": "#6f7a7c", "seed": 11},
    "herringbone": {"pattern": "herringbone", "diameter": 3.6, "sett": 0.1, "tint_main": "#b87a5c", "tint_accent": "#7d4a3a",
                    "tint_kerb": "#bdb7aa", "joint": 0.007, "moss": 0.2, "seed": 23},
    "grand_compass": {"pattern": "compass", "diameter": 9.0, "points": 16, "sett": 0.13, "kerb": 0.3, "tint_accent": "#3d4a5a",
                      "tint_flag": "#e0d3b0", "wear": 0.8, "moss": 0.15, "seed": 31},
}

ZJ = 0.0135         # nominal top of the sand joint fill
ZMAX = 0.0295       # contract: nothing above 30 mm
ZE = 0.004          # height of the kerb's outer edge where it meets the surrounding paving
KERB_UP = 0.0100    # kerb top above the joint level
TARGET = 44500      # triangle target for the whole patch (levels of detail are chosen to meet it)
KIND = {            # top above joint, crown, chamfer, plan corner radius, resample, height jitter, tilt
    "sett":   (0.0050, 0.0030, 0.0060, 0.012, 0.0, 0.0028, 0.030),
    "cobble": (0.0050, 0.0048, 0.0085, 0.022, 0.0, 0.0032, 0.035),
    "brick":  (0.0050, 0.0015, 0.0040, 0.006, 0.0, 0.0024, 0.022),
    "flag":   (0.0065, 0.0006, 0.0035, 0.008, 0.26, 0.0012, 0.005),
    "flagfix": (0.0065, 0.0004, 0.0035, 0.008, 0.26, 0.0, 0.0),
}
SMALL = ("sett", "cobble", "brick")


# ------------------------------------------------------------------ 2D helpers
def _area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _perim(P):
    E = np.roll(P, -1, 0) - P
    return float(np.hypot(E[:, 0], E[:, 1]).sum())


def _inset(P, d, miter=8.0):
    """Move every edge of a CCW polygon inwards by d (scalar or per vertex; bisector offset, mitre limited)."""
    E = np.roll(P, -1, 0) - P
    L = np.maximum(np.hypot(E[:, 0], E[:, 1]), 1e-9)[:, None]
    T = E / L
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    N0 = np.roll(N, 1, 0)
    k = np.maximum(1.0 + (N0 * N).sum(1), 2.0 / (miter * miter))
    d = np.asarray(d, dtype=np.float64)
    if d.ndim == 1:
        d = d[:, None]
    return P + d * (N0 + N) / k[:, None]


def _dedupe(P, eps=4e-4):
    out = [P[0]]
    for q in P[1:]:
        if abs(q[0] - out[-1][0]) + abs(q[1] - out[-1][1]) > eps:
            out.append(q)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) + abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return np.array(out)


def _round(P, cr, ang=radians(22)):
    """Ease the real corners of the plan outline (two points per corner)."""
    n = len(P)
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        e0, e1 = b - a, c - b
        l0, l1 = hypot(e0[0], e0[1]), hypot(e1[0], e1[1])
        turn = abs(atan2(e0[0] * e1[1] - e0[1] * e1[0], e0[0] * e1[0] + e0[1] * e1[1]))
        if turn < ang or l0 < 1e-6 or l1 < 1e-6:
            out.append(b)
            continue
        q = min(cr, 0.3 * l0, 0.3 * l1)
        out.append(b - e0 / l0 * q)
        out.append(b + e1 / l1 * q)
    return np.array(out)


def _resample(P, max_e):
    n = len(P)
    out = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        out.append(a)
        k = int(hypot(b[0] - a[0], b[1] - a[1]) / max_e)
        for j in range(1, k + 1):
            out.append(a + (b - a) * (j / (k + 1)))
    return np.array(out)


def _spall(P, rng, k):
    """Insert up to k chipped notches on the long edges: returns (P2, depth per vertex; 0 = sound arris)."""
    n = len(P)
    ins = {}
    for _ in range(k):
        i = int(rng.u(0, n)) % n
        a, b = P[i], P[(i + 1) % n]
        l = hypot(b[0] - a[0], b[1] - a[1])
        hw = rng.u(0.012, 0.03)
        if i in ins or l < 0.09 + 2 * hw:
            continue
        ins[i] = (rng.u(0.04 + hw, l - 0.04 - hw), hw, rng.u(0.006, 0.017), l)
    if not ins:
        return P, np.zeros(n)
    out, sp = [], []
    for i in range(n):
        out.append(P[i])
        sp.append(0.0)
        if i in ins:
            s, hw, dep, l = ins[i]
            dv = (P[(i + 1) % n] - P[i]) / l
            for ss, dd in ((s - hw, 0.0), (s + rng.jitter(0.0, hw * 0.3), dep), (s + hw, 0.0)):
                out.append(P[i] + dv * ss)
                sp.append(dd)
    return np.array(out), np.array(sp)


def _clip(P, nx, ny, d):
    """Keep the part of polygon P where nx*x + ny*y <= d."""
    if P is None or len(P) < 3:
        return None
    s = P[:, 0] * nx + P[:, 1] * ny - d
    if (s <= 1e-9).all():
        return P
    if (s >= -1e-9).all():
        return None
    out = []
    n = len(P)
    for i in range(n):
        j = (i + 1) % n
        if s[i] <= 0:
            out.append(P[i])
        if (s[i] < 0 < s[j]) or (s[j] < 0 < s[i]):
            t = s[i] / (s[i] - s[j])
            out.append(P[i] + (P[j] - P[i]) * t)
    return np.array(out) if len(out) >= 3 else None


def _subtract(P, planes):
    """P minus the convex region given by half-planes (nx, ny, d) -> list of convex pieces."""
    inter = P
    for (nx, ny, d) in planes:
        inter = _clip(inter, nx, ny, d)
        if inter is None:
            return [P]
    if abs(_area(inter)) < 1e-6:
        return [P]
    pieces, rem = [], P
    for (nx, ny, d) in planes:
        out = _clip(rem, -nx, -ny, -d)
        if out is not None and abs(_area(out)) > 1e-5:
            pieces.append(out)
        rem = _clip(rem, nx, ny, d)
        if rem is None:
            break
    return pieces


def _arc(r, a0, a1, step=radians(7.0)):
    if abs(a1 - a0) < 1e-6:
        return [(r * cos(a0), r * sin(a0))]
    m = max(1, int(ceil(abs(a1 - a0) / step)))
    return [(r * cos(a0 + (a1 - a0) * i / m), r * sin(a0 + (a1 - a0) * i / m)) for i in range(m + 1)]


class _Noise:
    """Cheap smooth 2D field in about -1..1 (sum of seeded sines); works on scalars and arrays."""

    def __init__(self, rng, scale, n=5):
        self.w = [(2 * pi / (scale * rng.u(0.6, 1.7)), rng.u(0, 2 * pi), rng.u(0, 2 * pi)) for _ in range(n)]

    def __call__(self, x, y):
        s = 0.0
        for k, a, ph in self.w:
            s = s + np.sin(k * (x * cos(a) + y * sin(a)) + ph)
        return s * (1.7 / len(self.w))


def _delta(r, L, rn, beta):
    """Angular half-width at radius r of a star point with its tip at radius L and notches at (rn, +-beta)."""
    if r >= L:
        return 0.0
    if r <= rn:
        return beta
    dx, dy = rn * cos(beta) - L, rn * sin(beta)
    a, b, c = dx * dx + dy * dy, 2 * L * dx, L * L - r * r
    t = (-b - sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    t = min(max(t, 0.0), 1.0)
    return atan2(t * dy, L + t * dx)


# ------------------------------------------------------------------ the paviour
class Paver:
    def __init__(self, p, rng, R, square=False):
        self.p, self.R, self.square = p, R, square
        self.rng = rng.fork("stones")
        self.jw = float(p.joint)
        self.u = float(p.unevenness)
        self.specs = []
        self.kerbs = []
        self.junctions = []
        self.rflat = 0.0
        self.nflag = 0
        self.splay = min(0.075, 0.5 * float(p.kerb))
        self.settle = _Noise(rng.fork("settle"), 1.6)
        self.mossn = _Noise(rng.fork("moss"), 1.1)
        self.wearn = _Noise(rng.fork("wear"), 2.2)
        self.stain = _Noise(rng.fork("stain"), 0.7)
        wr = rng.fork("walk")
        self.walk = [(wr.u(0, pi), wr.jitter(0.0, 0.1 * R)) for _ in range(2)]     # two desire lines across the patch

    # ---- fields
    def zj(self, x, y):
        """Top of the joint fill: follows the settlement of the paving, level round the centre piece."""
        f = np.clip((np.hypot(x, y) - self.rflat) / 0.45, 0.0, 1.0)
        return ZJ + 0.0022 * self.u * self.settle(x, y) * f

    def qedge(self, x, y):
        """Distance to the outer perimeter of the patch."""
        if self.square:
            return self.R - np.maximum(np.abs(x), np.abs(y))
        return self.R - np.hypot(x, y)

    def kerb_top(self, q, zj):
        """Nominal kerb surface: flat top, the outer `splay` rolls down to ZE at the perimeter."""
        t = np.clip(q / self.splay, 0.0, 1.0)
        return ZE + (zj + KERB_UP - ZE) * (1.0 - (1.0 - t) ** 2)

    def mossf(self, x, y):
        rr = np.minimum(np.hypot(x, y) / self.R, 1.2)
        return np.clip((self.mossn(x, y) + 0.75 * rr * rr - 0.35 + 0.25 * y / self.R) / 0.7, 0.0, 1.0) * float(self.p.moss)

    def wearf(self, x, y):
        """Foot traffic, about 0.5..1.5: a little patchy everywhere, strongest along the walking lines."""
        w = 0.0
        for a, off in self.walk:
            dd = (x * cos(a) + y * sin(a) - off) / 0.55
            w = np.maximum(w, np.exp(-dd * dd))
        return np.clip(0.75 + 0.2 * self.wearn(x, y), 0.5, 1.0) + 0.5 * w

    # ---- laying
    def stone(self, key, P_raw, kind="sett", phi=0.0):
        P_raw = _dedupe(np.asarray(P_raw, dtype=np.float64))
        if len(P_raw) < 3 or _area(P_raw) < 2.5e-4:
            return
        P = _dedupe(_inset(P_raw, self.jw * 0.5))
        if len(P) < 3:
            return
        A = _area(P)
        if A < 1.5e-4 or 2 * A / _perim(P) < 0.0035:
            return
        if kind == "flag" and A > 0.045 and self.rng.chance(0.05 + 0.12 * self.u):
            # an old flag cracked right through: two pieces with a hairline between them
            c = P.mean(0) + np.array([self.rng.jitter(0, 0.04), self.rng.jitter(0, 0.04)])
            a = self.rng.u(0, pi)
            nx, ny = cos(a), sin(a)
            dd = nx * c[0] + ny * c[1]
            for Q in (_clip(P, nx, ny, dd), _clip(P, -nx, -ny, -dd)):
                Q = _dedupe(Q, 2e-3) if Q is not None else None
                if Q is not None and len(Q) >= 3 and _area(Q) > 4e-3 and 2 * _area(Q) / _perim(Q) > 0.018:
                    self.specs.append((key, _inset(Q, 0.0009, 2.0), "flagcrack", phi))
            return
        self.specs.append((key, P, kind, phi))
        if kind in SMALL:
            self.junctions.append(P_raw[0])

    def segment(self, r0, r1, in0, in1, out0, out1, L, key, kind="sett", n=None, jit=0.15):
        """One row of stones between radii r0 < r1; the row's ends may be cut on the skew (in*/out* angles)."""
        rng = self.rng
        if n is None:
            n = max(1, int(round(0.5 * ((in1 - in0) * r0 + (out1 - out0) * r1) / L)))
            fr = [0.0] + [(j + rng.jitter(0.0, jit)) / n for j in range(1, n)] + [1.0]
        else:
            fr = [j / n for j in range(n + 1)]
        for j in range(n):
            f0, f1 = fr[j], fr[j + 1]
            oa, ob = out0 + (out1 - out0) * f0, out0 + (out1 - out0) * f1
            ia, ib = in0 + (in1 - in0) * f0, in0 + (in1 - in0) * f1
            poly = _arc(r1, oa, ob) + _arc(r0, ib, ia)
            tm = 0.25 * (oa + ob + ia + ib)
            rm = 0.5 * (r0 + r1)
            k = key(rm, tm) if callable(key) else key
            if k is None:
                continue
            self.stone(k, poly, kind, phi=tm + pi / 2)

    def rows(self, ra, rb, depth, L, key, kind="sett"):
        """Full rings of stones from ra to rb. key(i, n, rm, theta) -> material key."""
        n = max(1, int(round((rb - ra) / depth)))
        rs = [ra] + [ra + (rb - ra) * (i + self.rng.jitter(0.0, 0.1)) / n for i in range(1, n)] + [rb]
        for i in range(n):
            ph = self.rng.u(0, 2 * pi)
            kf = (lambda rm, tm, i=i: key(i, n, rm, tm)) if callable(key) else key
            self.segment(rs[i], rs[i + 1], ph, ph + 2 * pi, ph, ph + 2 * pi, L, kf, kind)
        return n

    def kerb_arc(self, r0, a0, a1):
        """A kerb stone between radius r0 and the perimeter, from angle a0 to a1 (joints included)."""
        R, hj = self.R, self.jw * 0.5
        ri = r0 + hj
        W = R - ri
        rm = ri + 0.5 * W
        b0 = a0 + hj / rm
        Lv = (a1 - a0) * rm - 2 * hj
        if W < 0.03 or Lv < 0.05:
            return

        def pos(u, v):
            a = b0 + v / rm
            return np.stack([(ri + u) * np.cos(a), (ri + u) * np.sin(a)], 1)

        self.kerbs.append((pos, W, Lv, False, min(0.2, sqrt(8 * R * 0.0015))))

    def kerb_quad(self, org, eu, ev, W, Lv, v_outer=False):
        """A straight kerb stone: u runs outwards from the inner joint to the perimeter, v along the kerb."""
        if W < 0.03 or Lv < 0.03:
            return
        org, eu, ev = np.array(org, dtype=np.float64), np.array(eu, dtype=np.float64), np.array(ev, dtype=np.float64)

        def pos(u, v):
            return org[None, :] + u[:, None] * eu[None, :] + v[:, None] * ev[None, :]

        self.kerbs.append((pos, W, Lv, v_outer, 0.22))

    # -------------------------------------------------------------- meshes
    def emit(self, mats):
        p, rng, u = self.p, self.rng, self.u
        nr = np.random.RandomState(int(rng.u(0, 2 ** 30)))
        small = [s for s in self.specs if s[2] in SMALL]
        nbig = len(self.specs) - len(small)
        big_hi = nbig * 105 <= 15000
        fixed = nbig * (95 if big_hi else 70) + len(self.kerbs) * 125 + (11000 if p.pattern == "compass" else 7000)
        lvl = 0
        for lv, t in ((3, 56), (2, 40), (1, 24), (0, 12)):
            if len(small) * t <= TARGET - fixed:
                lvl = lv
                break
        print(f"DBG stones: {len(small)} setts (lvl {lvl}), {nbig} flags (hi={big_hi}), {len(self.kerbs)} kerbs")
        groups = {}
        wear_p = float(p.wear)
        mV, mF, n_col = [], [], 0
        for (key, P, kind, phi) in self.specs:
            crack = kind == "flagcrack"
            if crack:
                kind = "flag"
            up, crown, cham, cr, max_e, hj, tl = KIND[kind]
            is_small = kind in SMALL
            if not is_small or lvl >= 1:
                P = _round(P, cr)
            P = _dedupe(P, 1.5e-3)
            if max_e > 0:
                P = _resample(P, max_e if big_hi else max_e * 1.8)
            sp = None
            if not is_small:
                P, sp = _spall(P, rng, int(rng.u(0, 3.4)) if kind == "flag" else int(rng.u(0, 1.6)))
                phi = rng.u(0, 2 * pi)                       # sawn stone has no lay direction
            lo, hi = P.min(0), P.max(0)

            def inset(e):                                   # chamfer ring, never outside the stone's own outline box
                return np.clip(_inset(P, e, 2.5), lo, hi)

            n = len(P)
            c = P.mean(0)
            A, per = abs(_area(P)), _perim(P)
            cham = min(cham, 0.3 * 2 * A / per)
            zb = float(self.zj(c[0], c[1]))
            top = zb + up + rng.jitter(0.0, hj * u)
            if is_small and rng.chance(0.03 * u):
                top -= rng.u(0.002, 0.004)                  # a sunken sett
            top = min(max(top, zb + 0.0018), ZMAX - crown - 0.0004)
            tx, ty = rng.jitter(0.0, tl * u), rng.jitter(0.0, tl * u)
            g0 = rng.u(0.0, 0.3)
            wf = float(self.wearf(c[0], c[1]))
            w0 = min(1.0, wear_p * 1.1 * rng.u(0.5, 1.0) * wf)
            ms = float(self.mossf(c[0], c[1]))
            mk = rng.u(0.5, 1.0) if is_small else 0.5       # how far the moss creeps onto the top
            g = groups.setdefault(key, {"V": [], "UV": [], "G": [], "W": [], "S": [], "F": [], "nv": 0})
            base = g["nv"]
            rings, zs, gm, wm, sm = [], [], [], [], []

            def ring(Q, z, gg, ww, ss, tilt=True):
                zz = np.zeros(len(Q)) + z
                if tilt:
                    zz = zz + tx * (Q[:, 0] - c[0]) + ty * (Q[:, 1] - c[1])
                    zz = np.clip(zz, zb + 0.0008, ZMAX)
                rings.append(Q)
                zs.append(zz)
                gm.append(np.zeros(len(Q)) + gg)
                wm.append(np.zeros(len(Q)) + ww)
                sm.append(np.zeros(len(Q)) + ss)

            F = g["F"]

            def band(a, b):
                oa, ob = base + a * n, base + b * n
                for j in range(n):
                    j1 = (j + 1) % n
                    F.append((oa + j, oa + j1, ob + j1, ob + j))

            msj = min(0.6, ms * 1.7)
            ring(P, zb - 0.001, 0.6, 0.0, msj, tilt=False)   # foot, just under the joint fill (nothing hidden below)
            if is_small:
                if lvl >= 2:
                    ring(inset(cham * 0.2), top - cham * 0.45, 0.4, 0.2 * w0, msj)
                    R2 = inset(cham)
                    ring(R2, top, g0 * 0.5 + 0.06, w0 * 0.7, min(0.6, ms * mk))
                else:
                    R2 = inset(cham * 0.8)
                    ring(R2, top, g0 * 0.6 + 0.08, w0 * 0.7, min(0.6, ms * mk))
                if lvl >= 3:
                    f = 0.55
                    ring(c + (R2 - c) * f, top + crown * (1.0 - f * f), g0 * 0.4, w0, ms * 0.4 * mk)
                cg, cw, cs_ = g0 * 0.3, w0, ms * 0.25 * mk
            else:
                tone = rng.u(-1.0, 1.0)                      # per-flag tone: dirty / clean and foot-polished
                gf = 0.05 + 0.3 * max(tone, 0.0)
                gf = 0.02 + 0.16 * max(tone, 0.0)
                wfl = min(1.0, (0.1 + 0.3 * max(-tone, 0.0) + 0.25 * wf) * (0.4 + 1.2 * wear_p))
                if key == "dark":
                    wfl = min(1.0, 0.12 + 1.15 * wfl)        # sawn basalt: worn to a grey sheen, never a black hole
                st = 0.12 * np.clip(self.stain(P[:, 0], P[:, 1]), 0.0, 1.0)
                ring(inset(cham * 0.25 + sp * 0.3), top - cham * 0.75 - sp * 0.2, 0.55, 0.15, msj)
                ring(inset(cham * 1.25 + sp), top, 0.3 + gf + st, wfl * 0.45, min(0.6, ms * mk))   # grime gathers at the joints
                if big_hi:
                    R2 = inset(cham * 1.25)
                    und = 0.0 if kind == "flagfix" else 0.0005 * u
                    ring(c + (R2 - c) * 0.74, top + crown * 0.5 + rng.jitter(0.0, und), gf + st * 0.7 + (0.25 if crack else 0.0), wfl, ms * 0.3)
                cg, cw, cs_ = gf * 0.8, wfl, ms * 0.15
            last = len(rings) - 1
            for a in range(last):
                band(a, a + 1)
            oc = base + (last + 1) * n
            ol = base + last * n
            for j in range(n):
                F.append((ol + j, ol + (j + 1) % n, oc))
            nv = (last + 1) * n + 1
            cpt = (c[0], c[1], min(top + crown, ZMAX))
            # moss that has filled the joint round this stone and creeps over its arris
            if is_small and n_col < 44 and ms > 0.05 and rng.chance(min(0.8, ms * 1.2 - 0.05)):
                n_col += 1
                U = _inset(P, 1.0, 2.5) - P
                creep = cham * 1.25 + nr.uniform(0.2, 1.0, n) * (0.005 + 0.016 * min(1.0, ms * 2.0))
                O = P - U * (self.jw * 0.52)
                I = np.clip(P + U * creep[:, None], lo, hi)
                mb = sum(len(v) for v in mV)
                for Q, z, tilt in ((O, zb - 0.001, False), (P, top - cham * 0.3, True), (I, top + crown * 0.7 + 0.0009, True)):
                    zz = np.zeros(n) + z
                    if tilt:
                        zz = np.clip(zz + tx * (Q[:, 0] - c[0]) + ty * (Q[:, 1] - c[1]), zb, ZMAX)
                    mV.append(np.column_stack([Q, zz]))
                for a in range(2):
                    for j in range(n):
                        j1 = (j + 1) % n
                        mF.append((mb + a * n + j, mb + a * n + j1, mb + (a + 1) * n + j1, mb + (a + 1) * n + j))
            XY = np.concatenate(rings)
            Z = np.concatenate(zs)
            # UVs: planar in the stone's own frame with a random offset
            D = XY - c
            cs, sn = cos(phi), sin(phi)
            ou, ov = rng.u(0, 3.0), rng.u(0, 3.0)
            UV = np.stack([cs * D[:, 0] + sn * D[:, 1] + ou, -sn * D[:, 0] + cs * D[:, 1] + ov], 1)
            g["V"].append(np.vstack([np.column_stack([XY, Z]), cpt]))
            g["UV"].append(np.vstack([UV, (ou, ov)]))
            g["G"].append(np.append(np.concatenate(gm), cg))
            g["W"].append(np.append(np.concatenate(wm), cw))
            g["S"].append(np.append(np.concatenate(sm), cs_))
            g["nv"] += nv
        objs = []
        for key, g in groups.items():
            objs.append(_mesh("stones_" + key, np.vstack(g["V"]), g["F"], np.vstack(g["UV"]), mats[key],
                              np.clip(np.concatenate(g["G"]), 0, 1), np.clip(np.concatenate(g["W"]), 0, 1), np.concatenate(g["S"]),
                              angle=27.0 if key.startswith("flag") else 42.0))
        if self.kerbs:
            objs.append(self._emit_kerbs(mats, nr))
        if mV:
            mo = mesh_object("moss_collars", np.vstack(mV), mF)
            assign(mo, mats["moss"])
            uv_box(mo, scale=1.0)
            smooth(mo, 65)
            paint(mo, grime=0.5)
            objs.append(mo)
        print(f"DBG moss collars: {n_col}")
        return objs

    def _emit_kerbs(self, mats, nr):
        """Kerb stones as small grids: eased arrises on the joint sides, rough-dressed top, outer edge rolled
        down to the surrounding paving, short skirt down to the ground."""
        rng, u, sp = self.rng, self.u, self.splay
        c = 0.009
        wear_p = float(self.p.wear)
        Vs, UVs, Gs, Ws, Ss, Fs = [], [], [], [], [], []
        base = 0
        for (pos, W, Lv, v_out, step) in self.kerbs:
            def stations(L, outer):
                a = [0.0, c * 0.3, c * 1.25]
                end = L - (sp if outer else c * 1.25)
                m = max(1, int(ceil((end - c * 1.25) / step)))
                a += [c * 1.25 + (end - c * 1.25) * i / m for i in range(1, m + 1)]
                a += [L - sp * 0.62, L - sp * 0.3, L - sp * 0.1, L, L] if outer else [L - c * 0.3, L]
                return np.array(a)

            us, vs = stations(W, True), stations(Lv, v_out)
            nu, nv = len(us), len(vs)
            Ug, Vg = np.meshgrid(us, vs, indexing="ij")
            skirt = np.zeros((nu, nv), dtype=bool)
            skirt[-1, :] = True
            if v_out:
                skirt[:, -1] = True
            Uf, Vf = Ug.ravel(), Vg.ravel()
            XY = pos(Uf, Vf)
            x, y = XY[:, 0], XY[:, 1]
            cc = pos(np.array([W * 0.5]), np.array([Lv * 0.5]))[0]
            zjc = float(self.zj(cc[0], cc[1]))
            q = W - Uf
            e = np.minimum(Uf, Vf)
            if v_out:
                q = np.minimum(q, Lv - Vf)
            else:
                e = np.minimum(e, Lv - Vf)
            cf = np.interp(e, [0.0, c * 0.3, c * 1.25], [0.0, 0.72, 1.0])
            ft = 1.0 - (1.0 - np.clip(q / sp, 0.0, 1.0)) ** 2
            tx, ty = rng.jitter(0.0, 0.008 * u), rng.jitter(0.0, 0.008 * u)
            ztop = zjc + KERB_UP + rng.jitter(0.0, 0.002 * u) + tx * (x - cc[0]) + ty * (y - cc[1])
            ztop = ztop + nr.uniform(-0.0008, 0.0008, len(x)) * (cf >= 1.0)      # rough-dressed top
            ztop = np.minimum(ZE + (ztop - ZE) * ft, ZMAX)
            nom = ZE + (zjc + KERB_UP - ZE) * ft
            zfoot = np.maximum(0.0002, nom - 0.0085)
            zfoot = np.where(Uf <= 1e-9, np.minimum(zfoot, zjc - 0.004), zfoot)
            z = zfoot + (ztop - zfoot) * cf
            z[skirt.ravel()] = 0.0
            g0 = rng.u(0.08, 0.3)
            w0 = min(1.0, wear_p * 1.1 * rng.u(0.4, 1.0) * float(self.wearf(cc[0], cc[1])))
            ms = self.mossf(x, y)
            Gs.append(np.clip(g0 + 0.5 * (1.0 - cf) + 0.7 * (1.0 - ft) + 0.18 * np.clip(self.stain(x, y), 0, 1), 0.0, 1.0))
            Ws.append(w0 * cf * ft)
            Ss.append(np.clip(ms * (0.45 + 1.1 * (1.0 - ft) + 0.7 * (1.0 - cf)), 0.0, 0.6))
            Vs.append(np.column_stack([XY, z]))
            ou, ov = rng.u(0, 3.0), rng.u(0, 3.0)
            UVs.append(np.stack([Uf + ou, Vf + ov], 1))
            for i in range(nu - 1):
                for j in range(nv - 1):
                    if v_out and i == nu - 2 and j == nv - 2:
                        continue
                    a = base + i * nv + j
                    Fs.append((a, a + nv, a + nv + 1, a + 1))
            base += nu * nv
        return _mesh("stones_kerb", np.vstack(Vs), Fs, np.vstack(UVs), mats["kerb"],
                     np.concatenate(Gs), np.concatenate(Ws), np.concatenate(Ss), angle=40.0)


def _mesh(name, V, F, UV, material, G=None, W=None, S=None, angle=42.0):
    o = mesh_object(name, V, F)
    me = o.data
    li = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", li)
    if len(me.vertices) == len(V):
        layer = me.uv_layers.new(name="UVMap") if not me.uv_layers else me.uv_layers[0]
        layer.data.foreach_set("uv", np.asarray(UV, dtype=np.float32)[li].ravel())
        uv_keep(o)
    assign(o, material)
    smooth(o, angle)

    def arr(a):
        if a is None:
            return None
        return (lambda co, nrm, a=a: a if len(co) == len(a) else float(np.mean(a)))

    paint(o, grime=arr(G), wear=arr(W), special=arr(S))
    return o


# ------------------------------------------------------------------ centre pieces
def _zmask(z0, z1, gain=1.0):
    """Vertex mask rising from 0 at height z0 to `gain` at z1 (either direction)."""
    return lambda co, nrm: np.clip((co[:, 2] - z0) / (z1 - z0), 0.0, 1.0) * gain


def bronze_boss(p, rng, r, br):
    """Cast bronze compass boss: moulded rim, pearl ring, faceted 16-point rose and a domed button. Polished on the
    high points, dark patina in the recesses."""
    k = min(max(r / 0.235, 0.45), 1.35)
    wg = min(1.0, 0.45 + float(p.wear))
    z0 = 0.0155                                             # field / recess floor
    rc = r * 0.72
    prof = [(r, 0.004), (r, 0.0185), (r - 0.003 * k, 0.0210), (r - 0.006 * k, 0.0214), (r - 0.009 * k, 0.0240), (r - 0.013 * k, 0.0256),
            (r - 0.018 * k, 0.0262), (r - 0.023 * k, 0.0256), (r - 0.027 * k, 0.0240), (r - 0.030 * k, 0.0212),
            (r - 0.033 * k, 0.0185), (r - 0.036 * k, z0),
            (rc + 0.002 * k, z0), (rc, 0.0185), (rc - 0.004 * k, 0.0206), (rc - 0.008 * k, 0.0185), (rc - 0.010 * k, z0),
            (0.0, z0)]
    plate = lathe("boss", prof, segments=48, smooth_angle=40)
    ch = (r - 0.036 * k) - (rc + 0.002 * k)                 # pearl channel
    rb = rc + 0.002 * k + 0.5 * ch
    br_ = min(0.011 * k, 0.4 * ch)
    nb = int(min(max(round(2 * pi * rb / (br_ * 2.9)), 12), 40))
    pearl = lathe("pearl", [(1.0, 0.0), (0.8, 0.62), (0.0, 1.0)], segments=6)
    pearls = instances_to_mesh("boss_pearls", pearl, [trs((rb * cos(2 * pi * (i + 0.5) / nb), rb * sin(2 * pi * (i + 0.5) / nb), z0 - 0.0003),
                                                          (0, 0, rng.u(0, 360)), (br_, br_, 0.0062)) for i in range(nb)], keep_base=False)
    # faceted rose: 8 long points over 8 short ones; every point is two facets meeting in a sharp ridge
    V, F, Wm, Gm = [], [], [], []
    rt = rc - 0.014 * k

    def star(n, a_off, r_tip, r_notch, zc, z_tip, z_notch, wl, wd):
        beta = pi / n
        for i in range(n):
            a = a_off + i * 2 * beta
            T = (r_tip * cos(a), r_tip * sin(a))
            N0 = (r_notch * cos(a - beta), r_notch * sin(a - beta))
            N1 = (r_notch * cos(a + beta), r_notch * sin(a + beta))
            for A_, B_, za, zb_, w in ((N0, T, z_notch, z_tip, wl), (T, N1, z_tip, z_notch, wd)):
                b = len(V)
                wa = w * (0.6 if za == z_notch else 1.0)
                wb = w * (0.6 if zb_ == z_notch else 1.0)
                V.extend([(0.0, 0.0, zc), (A_[0], A_[1], za), (B_[0], B_[1], zb_), (A_[0], A_[1], z0 - 0.0005), (B_[0], B_[1], z0 - 0.0005)])
                Wm.extend([w, wa, wb, 0.0, 0.0])
                gg = 0.0 if w == wl else 0.25
                Gm.extend([gg * 0.4, gg, gg, 0.9, 0.9])
                F.append((b, b + 1, b + 2))
                F.append((b + 3, b + 4, b + 2, b + 1))

    star(8, pi / 8, rt * 0.62, rt * 0.27, 0.0255, z0 + 0.0026, z0 + 0.0012, 0.8 * wg, 0.45 * wg)
    star(8, 0.0, rt, rt * 0.33, 0.0283, z0 + 0.0032, z0 + 0.0018, 1.0 * wg, 0.6 * wg)
    V = np.array(V)
    rose = _mesh("boss_rose", V, F, V[:, :2] + 0.37, br, np.array(Gm), np.array(Wm), np.array(Gm) * 0.35, angle=20.0)
    rbt = max(0.17 * r, 0.014)
    button = lathe("boss_button", [(rbt * 1.22, 0.0240), (rbt * 1.22, 0.0262), (rbt * 1.08, 0.0268), (rbt, 0.0268), (rbt * 0.92, 0.0279),
                                   (rbt * 0.7, 0.0288), (rbt * 0.38, 0.0293), (0.0, 0.0295)], segments=20)
    for o in (plate, pearls, button):
        assign(o, br)
        paint(o, wear=mask_mul(_zmask(0.0196, 0.0236), wg), grime=_zmask(0.0200, 0.0160, 0.85),
              special=mask_mul(_zmask(0.0190, 0.0160), noise_mask(0.07, seed=p.seed + 3, lo=0.35, hi=0.75), 0.6))
    return [plate, pearls, rose, button]


def centre_piece(p, rng, r, mats):
    """Round centre piece of radius r, 0..30 mm high."""
    kind = p.centre
    if kind == "bronze":
        return bronze_boss(p, rng, r, mats["bronze"])
    if kind == "stone":
        prof = [(r, 0.004), (r, 0.017), (r - 0.004, 0.0205), (r * 0.80, 0.0212), (r * 0.78, 0.0178), (r * 0.745, 0.0178), (r * 0.725, 0.0214),
                (r * 0.2, 0.0228), (0.0, 0.0228)]
        med = lathe("medallion", prof, segments=56, smooth_angle=35)
        assign(med, mats["flag_l"])
        uv_box(med)
        paint(med, wear=mask_mul(_zmask(0.02, 0.022), float(p.wear)), grime=_zmask(0.0205, 0.0175, 0.8),
              special=mask_mul(noise_mask(0.25, seed=p.seed + 5), float(p.moss) * 0.6))
        # six carved petals (dark stone inlay) and a bronze stud in the eye
        petals = []
        for k in range(6):
            a = k * 60.0 + 30.0
            pts = [(0.0, r * 0.12)] + [(r * 0.105 * sin(pi * t), r * (0.12 + 0.56 * t)) for t in (0.2, 0.4, 0.6, 0.8)] + [(0.0, r * 0.68)]
            pts += [(-r * 0.105 * sin(pi * t), r * (0.12 + 0.56 * t)) for t in (0.8, 0.6, 0.4, 0.2)]
            petals.append(extrude_profile(f"petal{k}", pts, 0.0022, loc=(0, 0, 0.0214), rot=(0, 0, a), bevel=0.0006))
        pj = join(petals, "medallion_petals")
        assign(pj, mats["dark"])
        paint(pj, wear=float(p.wear) * 0.8)
        stud = lathe("medallion_stud", [(r * 0.11, 0.0215), (r * 0.11, 0.0236), (r * 0.095, 0.0242), (r * 0.085, 0.0262), (r * 0.05, 0.0282),
                                        (0.0, 0.029)], segments=20)
        assign(stud, mats["bronze"])
        paint(stud, wear=_zmask(0.0236, 0.027))
        return [med, pj, stud]
    # drain grate: iron frame, slotted plate, dark sump below
    iron = mats["iron"]
    frame = lathe("grate_frame", [(r, 0.004), (r, 0.018), (r - 0.004, 0.0212), (r * 0.87, 0.0212), (r * 0.85, 0.0185), (r * 0.85, 0.012)], segments=48, smooth_angle=35)
    ro = r * 0.855
    holes = []
    pitch = max(0.028, r * 0.2)
    ns = int((ro * 0.86) / pitch)
    for i in range(-ns, ns + 1):
        x = i * pitch
        half = sqrt(max((ro * 0.9) ** 2 - x * x, 0.0))
        for sy in (-1, 1):
            y0, y1 = 0.012, half
            if y1 - y0 < 0.02:
                continue
            w = pitch * 0.5
            rr = rounded_rect(w, y1 - y0, w * 0.45, 3)
            holes.append([(x + qx, sy * (y0 + y1) * 0.5 + qy) for qx, qy in rr])
    plate = shape_extrude("grate_plate", circle_pts(ro, 40), holes, depth=0.0045, loc=(0, 0, 0.015), bevel=0.0008)
    sump = cylinder("grate_sump", ro, 0.002, loc=(0, 0, 0.0142), verts=32)
    for o in (frame, plate):
        assign(o, iron)
        paint(o, wear=mask_mul(_zmask(0.0185, 0.021), float(p.wear)), special=noise_mask(0.07, seed=p.seed + 9, lo=0.3, hi=0.8))
    assign(sump, iron)
    paint(sump, grime=1.0)
    return [frame, plate, sump]


# ------------------------------------------------------------------ patterns
def lay_kerb_ring(pv, R, kw):
    L = min(max(0.22 * R, 0.36), 0.62)
    rng = pv.rng
    n = max(6, int(round(2 * pi * (R - 0.5 * kw) / L)))
    ph = rng.u(0, 2 * pi)
    fr = [0.0] + [(j + rng.jitter(0.0, 0.16)) / n for j in range(1, n)] + [1.0]
    for j in range(n):
        pv.kerb_arc(R - kw, ph + 2 * pi * fr[j], ph + 2 * pi * fr[j + 1])


def _letter(name, ch, H, depth, x0, y0, ang, z, font="serif"):
    """One capital of cap height H lying flat, centred on (x0, y0), its top pointing along `ang` (radians)."""
    t = text(name, ch, size=H * 1.4, depth=depth, loc=(0, 0, 0), rot=(0, 0, 0), font=font)
    me = t.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    lo, hi = co.min(0), co.max(0)
    s = H / max(hi[1] - lo[1], 1e-6)
    x, y = (co[:, 0] - 0.5 * (lo[0] + hi[0])) * s, (co[:, 1] - 0.5 * (lo[1] + hi[1])) * s
    a = ang - pi / 2
    out = np.column_stack([x * cos(a) - y * sin(a) + x0, x * sin(a) + y * cos(a) + y0, co[:, 2] - lo[2] + z])
    me.vertices.foreach_set("co", out.astype(np.float32).ravel())
    me.update()
    print(f"DBG letter {ch}: {len(me.polygons)} faces, {(hi[0] - lo[0]) * s:.3f} x {H:.3f} m")
    return t


def lay_compass(pv, p, rng, R, d, L, mats):
    kw = float(p.kerb)
    Rk = R - kw
    n_ob = 2 if R >= 1.6 else 1
    wf = min(max(0.13 * R, 0.16), 0.36)
    Rf = Rk - n_ob * d - wf
    rh = min(max(0.085 * R + 0.03, 0.11), 0.32)
    rn = max(0.30 * Rf, rh + 0.05)
    n = max(4, int(round(p.points)))
    beta = pi / n
    th = [pi / 2 - k * 2 * beta for k in range(n)]           # clockwise from north, like a compass card
    Ls = []
    for k in range(n):
        if n == 4 or n % 2 == 1:
            f = 1.0
        elif n % 4 == 0 and n >= 16:
            f = 1.0 if k % (n // 4) == 0 else (0.72 if k % (n // 8) == 0 and n % 8 == 0 else 0.52)
        elif n % 4 == 0:
            f = 1.0 if k % (n // 4) == 0 else 0.72
        else:
            f = 1.0 if k % 2 == 0 else 0.72
        Ls.append(max(Rf * f, rn + 0.06))
    # --- star points: each a kite split down its axis into a light and a dark flag, cut into lengths
    rho = rh / cos(beta * 0.5)
    for k in range(n):
        Lk, a = Ls[k], th[k]
        ca, sa = cos(a), sin(a)
        ncut = max(1, int(round((Lk - rh) / 0.6)))
        xs = [rh * 0.5] + [rh + (Lk - rh) * (i / ncut) ** 0.92 + rng.jitter(0, 0.02) for i in range(1, ncut)] + [Lk + 0.1]
        for s in (1, -1):
            loc = [(rho, 0.0), (Lk, 0.0), (rn * cos(beta), s * rn * sin(beta)), (rho * cos(beta), s * rho * sin(beta))]
            if s < 0:
                loc = loc[::-1]
            P = np.array(loc)
            for i in range(ncut):
                Q = _clip(_clip(P, -1.0, 0.0, -xs[i]), 1.0, 0.0, xs[i + 1])
                if Q is None:
                    continue
                W = np.stack([Q[:, 0] * ca - Q[:, 1] * sa, Q[:, 0] * sa + Q[:, 1] * ca], 1)
                pv.stone("flag_l" if s > 0 else "dark", W, "flag", phi=a)
    # --- field: rows of setts between the points, end setts cut to the star's edges
    nrows = max(1, int(round((Rf - rn) / d)))
    rs = [rn] + [rn + (Rf - rn) * (i + rng.jitter(0.0, 0.08)) / nrows for i in range(1, nrows)] + [Rf]
    band_i = min(range(nrows), key=lambda i: abs(0.5 * (rs[i] + rs[i + 1]) - (0.72 * Rf + 0.6 * d)))
    order = sorted(range(n), key=lambda k: th[k])
    for i in range(nrows):
        r0, r1 = rs[i], rs[i + 1]
        blk = [k for k in order if Ls[k] > r0 + 1e-4]
        key = "accent" if (i == band_i and nrows >= 5) else (lambda rm, tm: "main_b" if pv.rng.chance(0.32) else "main_a")
        for bi, a in enumerate(blk):
            b = blk[(bi + 1) % len(blk)]
            ta, tb = th[a], th[b]
            if bi == len(blk) - 1:
                tb += 2 * pi
            in0, in1 = ta + _delta(r0, Ls[a], rn, beta), tb - _delta(r0, Ls[b], rn, beta)
            out0, out1 = ta + _delta(r1, Ls[a], rn, beta), tb - _delta(r1, Ls[b], rn, beta)
            if out1 - out0 < 1e-4:
                continue
            pv.segment(r0, r1, in0, max(in1, in0), out0, out1, L, key, "sett")
    # --- lettered flag ring
    m = max(1, int(round(2 * pi * (Rf + wf * 0.5) / 16 / 0.72)))
    nseg = 16 * m
    ph = pi / 2 + pi / nseg
    sw = 0.014                                             # inlaid bronze strip between field and flag ring
    pv.segment(Rf + sw, Rf + wf, ph, ph + 2 * pi, ph, ph + 2 * pi, 1.0, "flag_l", "flagfix", n=nseg)
    pv.rflat = rh
    up = KIND["flagfix"][0]
    s0, s1 = Rf - pv.jw * 0.5 + 0.0015, Rf + sw + pv.jw * 0.5 - 0.0015
    strip = lathe("strip", [(s0, -0.004), (s0, up - 0.0006), (s0 + 0.0014, up + 0.0006), (s1 - 0.0014, up + 0.0006), (s1, up - 0.0006), (s1, -0.004)],
                  segments=int(min(max(48, Rf * 60), 160)), smooth_angle=30)

    def lift(co):
        co.z += float(pv.zj(co.x, co.y))
        return co

    transform_verts(strip, lift)
    objs = [strip]
    rc = Rf + sw + (wf - sw) * 0.5
    letters = str(p.letters or "")[:4]
    H = min(0.25, (wf - sw) * 0.66)                        # cap height: 200 mm and more on a 5 m rose
    for i, ch in enumerate(letters):
        if ch.strip() == "":
            continue
        a = pi / 2 - i * pi / 2
        x0, y0 = rc * cos(a), rc * sin(a)
        objs.append(_letter(f"letter{i}", ch, H, 0.0022, x0, y0, a, float(pv.zj(x0, y0)) + up - 0.0002))
    rs_ = 0.5 * wf * 0.2
    stud = lathe("stud", [(rs_, -0.002), (rs_, 0.0014), (rs_ * 0.86, 0.0024), (rs_ * 0.62, 0.0048), (rs_ * 0.3, 0.0061),
                          (0.0, 0.0065)], segments=10)
    sm = []
    for k in range(16):
        if k % 4 == 0 and len(letters) > k // 4 and letters[k // 4].strip():
            continue
        a = pi / 2 - k * pi / 8
        rr_ = rc if k % 2 == 0 else rc + 0.0
        sm.append(trs((rr_ * cos(a), rr_ * sin(a), float(pv.zj(rr_ * cos(a), rr_ * sin(a))) + up), (0, 0, rng.u(0, 360)),
                      1.0 if k % 2 == 0 else 0.72))
    studs = instances_to_mesh("studs", stud, sm, keep_base=False)
    objs.append(studs)
    iron = p.centre == "grate"                             # iron inlay when the centre is an iron grate, else bronze
    wg = min(1.0, 0.45 + float(p.wear))
    for o in objs:
        assign(o, mats["iron"] if iron else (mats["bronze"] if o is studs else mats["bronze_d"]))
    paint(strip, wear=0.55 * wg, special=noise_mask(0.1, seed=p.seed + 6, lo=0.45, hi=0.85))
    paint(studs, wear=lambda co, nrm: np.clip(nrm[:, 2], 0.0, 1.0) ** 2 * wg)
    for o in objs[1:-1]:
        paint(o, wear=0.1 * wg, grime=0.4)                 # letters: dark patinated bronze so they read on the pale flags
    # --- dark border rows and the kerb
    pv.rows(Rf + wf, Rk, d, L, "dark", "sett")
    lay_kerb_ring(pv, R, kw)
    return rh, objs


def lay_rosette(pv, p, rng, R, d, L):
    """Flower of porphyry petals, each outlined with a course of dark setts on a light granite field, dark darts
    between the petal tips, accent and dark border rows. The setts are cut to the petal lines."""
    kw = float(p.kerb)
    Rk = R - kw
    rh = min(max(0.09 * R + 0.03, 0.12), 0.36)
    pv.rflat = rh
    n = max(4, int(round(p.points)))
    beta = pi / n
    nt = max(4, int(round((Rk - rh) / d)))
    rs = [rh] + [rh + (Rk - rh) * (i + pv.rng.jitter(0.0, 0.08)) / nt for i in range(1, nt)] + [Rk]
    npet = max(1, nt - 3)                                   # rows carrying the petals
    ra, rp, re = rh, rs[npet] - 0.25 * d, rs[npet]
    rd = ra + 0.5 * (re - ra)                               # the darts start half way out
    ow = max(0.62 * d, 0.05)                                # width of the outline course
    wmax = 0.56 * beta

    def wp(r):
        s = (r - ra) / (rp - ra)
        return wmax * sin(pi * s ** 0.8) ** 0.9 if 0.0 < s < 1.0 else 0.0

    def wd(r):
        if r <= rd:
            return 0.0
        s = min((r - rd) / (re - rd), 1.0)
        return max(0.0, min(0.36 * beta * s, beta - wp(r) - ow / r - 0.45 * L / r))

    def field(rm, tm):
        return "main_b" if pv.rng.chance(0.12) else "main_a"

    cs = sorted(pi / 2 - k * 2 * beta for k in range(n))
    for i in range(nt):
        r0, r1 = rs[i], rs[i + 1]
        if i >= npet:
            ph = pv.rng.u(0, 2 * pi)
            key = "dark" if i == nt - 1 else ("accent" if i == nt - 2 else field)
            pv.segment(r0, r1, ph, ph + 2 * pi, ph, ph + 2 * pi, L, key, "sett")
            continue
        feats = []
        w0, w1 = wp(r0), wp(r1)
        o0 = min((ow if w0 > 0 else ow * 0.4) / r0, beta - w0)
        o1 = min((ow if w1 > 0 else ow * 0.4) / r1, beta - w1)
        h0, h1 = wd(r0), wd(r1)
        for c in cs:
            if max(w0 * r0, w1 * r1) > 0.012:
                pv.segment(r0, r1, c - w0, c + w0, c - w1, c + w1, L, "accent", "sett")
            pv.segment(r0, r1, c - w0 - o0, c - w0, c - w1 - o1, c - w1, L, "dark", "sett", n=1)
            pv.segment(r0, r1, c + w0, c + w0 + o0, c + w1, c + w1 + o1, L, "dark", "sett", n=1)
            feats.append((c, w0 + o0, w1 + o1))
            if h1 * r1 > 0.02:
                cd = c + beta
                pv.segment(r0, r1, cd - h0, cd + h0, cd - h1, cd + h1, L, "dark", "sett")
                feats.append((cd, h0, h1))
        feats.sort()
        m = len(feats)
        for j in range(m):
            ca, ha0, ha1 = feats[j]
            cb, hb0, hb1 = feats[(j + 1) % m]
            if j == m - 1:
                cb += 2 * pi
            i0, i1 = ca + ha0, cb - hb0
            q0, q1 = ca + ha1, cb - hb1
            if (q1 - q0) * r1 < 0.004 and (i1 - i0) * r0 < 0.004:
                continue
            pv.segment(r0, r1, i0, max(i1, i0), q0, max(q1, q0), L, field, "sett")
    lay_kerb_ring(pv, R, kw)
    return rh


def lay_concentric(pv, p, rng, R, d, L):
    kw = float(p.kerb)
    Rk = R - kw
    rh = min(max(0.1 * R + 0.04, 0.14), 0.4)
    pv.rflat = rh
    wf = min(max(0.12 * R, 0.18), 0.34)
    has_flag = (Rk - rh) > 6 * d + wf

    def key(i, nr, rm, tm):
        if i == 0 or i == nr - 1:
            return "dark"
        if i % 4 == 2 and nr >= 5:
            return "accent"
        return "main_b" if pv.rng.chance(0.35) else "main_a"

    def kind_rows(ra, rb):
        n = max(1, int(round((rb - ra) / (d * 1.12))))
        rs = [ra] + [ra + (rb - ra) * (i + pv.rng.jitter(0.0, 0.1)) / n for i in range(1, n)] + [rb]
        for i in range(n):
            ph = pv.rng.u(0, 2 * pi)
            dark = (i == 0 or i == n - 1)
            pv.segment(rs[i], rs[i + 1], ph, ph + 2 * pi, ph, ph + 2 * pi, L * (0.8 if dark else 1.0),
                       (lambda rm, tm, i=i: key(i, n, rm, tm)), "sett" if dark else "cobble")

    if has_flag:
        ra = rh + (Rk - rh - wf) * 0.42
        kind_rows(rh, ra)
        nf = max(8, int(round(2 * pi * (ra + wf * 0.5) / 0.62)))
        ph = pv.rng.u(0, 2 * pi)
        pv.segment(ra, ra + wf, ph, ph + 2 * pi, ph, ph + 2 * pi, 1.0, "flag_l", "flag", n=nf)
        kind_rows(ra + wf, Rk)
    else:
        kind_rows(rh, Rk)
    lay_kerb_ring(pv, R, kw)
    return rh


def lay_herringbone(pv, p, rng, R, d, L):
    kw = float(p.kerb)
    h = R
    jw = pv.jw
    rr = pv.rng
    # kerb: four corner blocks and straight stones between, laid side by side
    hi = h - kw
    side_n = max(1, int(round(2 * hi / 0.55)))
    for q in range(4):
        ca, sa = cos(q * pi / 2), sin(q * pi / 2)
        eu, ev = (ca, sa), (-sa, ca)

        def rot(pts):
            return [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]

        W = h - hi - jw * 0.5
        pv.kerb_quad(rot([(hi + jw * 0.5, hi + jw * 0.5)])[0], eu, ev, W, W, v_outer=True)
        ys = [-hi] + [-hi + 2 * hi * (j + rr.jitter(0.0, 0.1)) / side_n for j in range(1, side_n)] + [hi]
        for j in range(side_n):
            pv.kerb_quad(rot([(hi + jw * 0.5, ys[j] + jw * 0.5)])[0], eu, ev, W, ys[j + 1] - ys[j] - jw)
    # soldier course of dark setts inside the kerb
    ds = d * 1.45
    hs = hi - ds
    sold_n = max(1, int(round(2 * hs / (d * 0.95))))
    for q in range(4):
        ca, sa = cos(q * pi / 2), sin(q * pi / 2)

        def rot(pts):
            return [(x * ca - y * sa, x * sa + y * ca) for x, y in pts]

        pv.stone("dark", rot([(hs, hs), (hi, hs), (hi, hi), (hs, hi)]), "sett", phi=q * pi / 2)
        ys = [-hs] + [-hs + 2 * hs * (j + rr.jitter(0.0, 0.1)) / sold_n for j in range(1, sold_n)] + [hs]
        for j in range(sold_n):
            pv.stone("dark", rot([(hs, ys[j]), (hi, ys[j]), (hi, ys[j + 1]), (hs, ys[j + 1])]), "sett", phi=q * pi / 2)
    # centre medallion (diamond of four flags around the centre piece)
    rh = min(max(0.1 * R + 0.03, 0.12), 0.3)
    md = min(max(0.42 * hs, rh + 0.22), 1.2)
    has_med = hs > md + 0.25
    planes = []
    if has_med:
        pv.rflat = rh
        s2 = sqrt(0.5)
        planes = [(sx * s2, sy * s2, md * s2) for sx in (1, -1) for sy in (1, -1)]
        for q in range(4):
            a0, a1 = q * pi / 2, (q + 1) * pi / 2
            poly = [(md * cos(a0), md * sin(a0)), (md * cos(a1), md * sin(a1))] + _arc(rh, a1, a0, radians(15))
            pv.stone("flag_l", poly, "flag", phi=a0 + pi / 4)
    # herringbone field at 45 degrees, bricks cut at the border and around the medallion
    W = d * 0.92
    N = int(ceil(hs * 1.5 / W)) + 3
    c45 = sqrt(0.5)
    ox, oy = rr.u(0, W), rr.u(0, W)
    sq = [(1.0, 0.0, hs), (-1.0, 0.0, hs), (0.0, 1.0, hs), (0.0, -1.0, hs)]
    for x in range(-N, N + 1):
        for y in range(-N, N + 1):
            m4 = (x - y) % 4
            if m4 == 0:
                rect = [(x, y), (x + 2, y), (x + 2, y + 1), (x, y + 1)]
                phi = pi / 4
            elif m4 == 3:
                rect = [(x, y), (x + 1, y), (x + 1, y + 2), (x, y + 2)]
                phi = 3 * pi / 4
            else:
                continue
            P = np.array([((u * W + ox) * c45 - (v * W + oy) * c45, (u * W + ox) * c45 + (v * W + oy) * c45) for u, v in rect])
            if np.abs(P).min(axis=0).max() > hs + 0.01 and (np.abs(P).min() > hs):
                continue
            for (nx, ny, dd) in sq:
                P = _clip(P, nx, ny, dd)
                if P is None:
                    break
            if P is None:
                continue
            pieces = _subtract(P, planes) if has_med else [P]
            k = "accent" if rr.chance(0.14) else ("main_b" if rr.chance(0.4) else "main_a")
            for Q in pieces:
                pv.stone(k, Q, "brick", phi=phi)
    return rh if has_med else 0.0


# ------------------------------------------------------------------ joint fill
def joint_bed(pv, p, rng, R, square, mats):
    """Sand / grit joint fill a few millimetres below the arrises, rising under the kerb and falling with it to
    the perimeter; damp, dirtier patches and moss on the wet side."""
    kw, jw, sp = float(p.kerb), pv.jw, pv.splay
    bn = _Noise(rng.fork("bed"), 0.45)
    dn = _Noise(rng.fork("damp"), 0.9)
    edge = [kw - jw * 0.5 - 0.006, sp, sp * 0.6, sp * 0.3, 0.012, 0.004]      # distances from the perimeter
    Ri = R - kw
    if square:
        m = int(min(max(Ri / 0.4, 3), 12))
        outer = np.array([R - q for q in edge])
        xs = np.concatenate([-outer[::-1], np.linspace(-Ri, Ri, 2 * m + 1), outer])
        k = len(xs)
        V = np.array([(x, y, 0.0) for y in xs for x in xs])
        F = [(j * k + i, j * k + i + 1, (j + 1) * k + i + 1, (j + 1) * k + i) for j in range(k - 1) for i in range(k - 1)]
    else:
        nr = int(min(max(ceil(Ri / 0.4), 2), 14))
        radii = [Ri * i / nr for i in range(1, nr + 1)] + [R - q for q in edge]
        ns = int(min(max(2 * pi * R / 0.22, 40), 96))
        V = [(0.0, 0.0, 0.0)]
        F = []
        for rr_ in radii:
            V += [(rr_ * cos(2 * pi * j / ns), rr_ * sin(2 * pi * j / ns), 0.0) for j in range(ns)]
        for j in range(ns):
            F.append((0, 1 + j, 1 + (j + 1) % ns))
        for i in range(1, len(radii)):
            a, b = 1 + (i - 1) * ns, 1 + i * ns
            F += [(a + j, b + j, b + (j + 1) % ns, a + (j + 1) % ns) for j in range(ns)]
        V = np.array(V)
    x, y = V[:, 0], V[:, 1]
    q = pv.qedge(x, y)
    zj = pv.zj(x, y)
    z = np.where(q >= kw - jw * 0.5 - 0.003, zj + 0.0007 * bn(x, y), np.maximum(0.0006, pv.kerb_top(q, zj) - 0.0055))
    z[q < 0.006] = 0.0004
    if p.centre == "grate":
        z[np.hypot(x, y) < 0.05] = 0.0115                   # the sump under the grate stays open
    V[:, 2] = z
    bed = mesh_object("joint_fill", V, F)
    assign(bed, mats["sand"])
    uv_box(bed)
    smooth(bed, 60)
    mossf = pv.mossf
    paint(bed, grime=lambda co, nrm: 0.08 + 0.5 * np.clip((dn(co[:, 0], co[:, 1]) - 0.15) / 0.6, 0.0, 1.0),
          special=lambda co, nrm: np.clip(mossf(co[:, 0], co[:, 1]) * 1.3, 0.0, 0.55))
    return bed


# ------------------------------------------------------------------ build
def build(p, rng):
    D = float(p.diameter)
    R = D * 0.5
    square = p.pattern == "herringbone"
    d = max(float(p.sett), 0.0105 * D)                     # keep the stone count sane on very large patches
    L = d * {"compass": 1.38, "herringbone": 1.38, "rosette": 1.02}.get(p.pattern, 1.12)

    def shade(hexc, f, warm=1.0):
        r, g, b = hex_rgb(hexc)
        return (min(r * f * warm, 1.0), min(g * f, 1.0), min(b * f / warm, 1.0))

    SR = 0.6                                               # setts: a little smoother than fresh-split stone

    class Mats(dict):                                      # materials are only created when a pattern uses them
        defs = {
            "main_a": lambda: mat("stone_rough", p.tint_main, rough=SR),
            "main_b": lambda: mat("stone_rough", shade(p.tint_main, 0.66, 1.06), rough=SR),
            "accent": lambda: mat("stone_rough", p.tint_accent, rough=SR, wear=0.7),
            "dark": lambda: mat("stone_rough", p.tint_dark, rough=0.55, wear=0.45),       # worn basalt stays dark, just shinier
            "flag_l": lambda: mat("mortar", p.tint_flag, rough=0.7),        # sawn limestone: fine even grain, no lay direction
            "kerb": lambda: mat("stone_rough", p.tint_kerb),
            "bronze": lambda: mat("brass", "#c9963f", rough=1.2),           # cast boss, studs: rubbed bright
            "bronze_d": lambda: mat("bronze_aged", rough=1.3),              # letters, inlaid strip: dark patina
            "iron": lambda: mat("iron_rusty"),
            "sand": lambda: mat("mortar", "#a39880"),
            "moss": lambda: mat("moss"),
        }

        def __missing__(self, k):
            self[k] = self.defs[k]()
            return self[k]

    mats = Mats()
    pv = Paver(p, rng, R, square)
    if p.pattern == "compass":
        rh, _ = lay_compass(pv, p, rng.fork("compass"), R, d, L, mats)
    elif p.pattern == "rosette":
        rh = lay_rosette(pv, p, rng, R, d, L)
    elif p.pattern == "concentric":
        rh = lay_concentric(pv, p, rng, R, d, L)
    else:
        rh = lay_herringbone(pv, p, rng, R, d, L)
    pv.emit(mats)
    if rh > 0:
        centre_piece(p, rng.fork("centre"), rh - pv.jw * 0.5, mats)
    joint_bed(pv, p, rng, R, square, mats)

    # ---- moss cushions growing out of the joints on the damp side (dark olive, low)
    if p.moss > 0.02 and pv.junctions:
        mr = rng.fork("tufts")
        tuft = lathe("tuft", [(1.0, 0.0), (0.72, 0.62), (0.0, 1.0)], segments=5)
        mm = []
        for q in pv.junctions:
            f = float(pv.mossf(q[0], q[1]))
            if f > 0.04 and mr.chance(min(0.85, f * 2.0)):
                rad = mr.u(0.007, 0.016) * (0.85 + 0.9 * f)
                mm.append(trs((q[0] + mr.jitter(0, 0.004), q[1] + mr.jitter(0, 0.004), float(pv.zj(q[0], q[1])) - 0.002),
                              (0, 0, mr.u(0, 360)), (rad * mr.u(0.8, 1.5), rad, mr.u(0.006, 0.011))))
        if len(mm) > 110:
            step = len(mm) / 110.0
            mm = [mm[int(i * step)] for i in range(110)]
        if mm:
            tf = instances_to_mesh("moss_tufts", tuft, mm, keep_base=False)
            assign(tf, mats["moss"])
            smooth(tf, 70)
            paint(tf, grime=0.5)
        else:
            delete(tuft)
        print(f"DBG moss tufts: {len(mm)}")

    socket("centre", "attach", (0, 0, 0.03), role="prop_top")
