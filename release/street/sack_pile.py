"""Pile of filled jute grain sacks. Every sack is a ring-built soft body: a plump bolster with a flat-sewn
bottom seam and corner ears, a stitched side seam, woven stripes, tension pleats running into a neck that is
tied with hemp rope and ends in a fanned, pleated ruffle. Lying sacks are settled onto a height field of the
pile so far: each one stays a stiff stuffed body that can roll, pitch and bend gently (length preserved),
its underside flattens against whatever it bears on and is kept a few mm clear of it, and its flanks are
squeezed in where a neighbour lies beside it. Standing sacks slump on a slightly irregular foot with
compression folds and lean until they touch the pile. One sack stands open with its rim rolled down, heaped
with kernels (grain / coffee / maize / beans), a tin scoop and a feathered spill on the paving. Optional pallet.
"""
from tk import *
import bpy
import numpy as np
from math import sin, cos, pi, radians, degrees, atan2, hypot, sqrt, tan
from mathutils import Vector, Matrix

META = {"name": "Grain sack pile", "class": "prop", "mount": "ground", "tri_budget": 24000,  # up to 8 soft bodies
        "focus": 0.6}
PARAMS = {
    "count": {"default": 5, "min": 3, "max": 8, "desc": "number of sacks, including the open one"},
    "sack_size": {"default": 0.85, "min": 0.6, "max": 1.1, "desc": "filled sack length, bottom seam to tie (m)"},
    "style": {"default": "heap", "choices": ["heap", "leaning", "stacked"], "desc": "heap = dropped on each other, leaning = stood on end against laid sacks, stacked = crossed layers"},
    "contents": {"default": "grain", "choices": ["grain", "coffee", "corn", "beans"], "desc": "what the open sack shows"},
    "fill": {"default": 0.85, "min": 0.6, "max": 1.0, "desc": "how plump the sacks are"},
    "cloth": {"default": "#dccaa6", "desc": "jute tint"},
    "stripes": {"default": 2, "min": 0, "max": 3, "desc": "woven stripes down the face of each sack (0 = plain)"},
    "stripe_color": {"default": "#3f5f96", "desc": "stripe dye colour (woven in, fades with age)"},
    "label": {"default": "GRAIN", "desc": "stencilled text on the open sack ('' = none)"},
    "pallet": {"default": False, "desc": "board pallet under the laid sacks"},
    "scoop": {"default": True, "desc": "tin scoop stuck in the open sack"},
    "spill": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "amount spilt on the ground by the open sack"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "dirt, damp staining and sun-fade"},
}
VARIANTS = {
    "coffee_stack": {"style": "stacked", "count": 8, "contents": "coffee", "pallet": True, "cloth": "#c9ad80",
                     "stripes": 1, "stripe_color": "#8a2f25", "label": "COFFEE", "sack_size": 0.9, "seed": 4},
    "market_lean": {"style": "leaning", "count": 6, "contents": "corn", "cloth": "#e6dcc4", "stripes": 3,
                    "stripe_color": "#a8392c", "label": "MAIZE", "fill": 0.95, "seed": 11},
    "beans_trio": {"style": "leaning", "count": 3, "contents": "beans", "sack_size": 0.68, "stripes": 0, "cloth": "#b99f78",
                   "label": "", "fill": 0.75, "age": 0.85, "spill": 0.9, "seed": 23},
    "big_heap": {"style": "heap", "count": 8, "sack_size": 1.0, "stripes": 1, "stripe_color": "#3c6b4a", "cloth": "#cdbd9c",
                 "label": "MILL", "pallet": True, "fill": 0.9, "age": 0.7, "seed": 31},
}

CONTENTS = {  # packed-bed tint, kernel material, kernel tint, kernel size (l, w, h), round?
    "grain": ("#a68a55", "plaster_lime", "#ecd6a2", (0.015, 0.0068, 0.005), False),
    "coffee": ("#2e1b10", "ceramic_glazed", "#5a3520", (0.015, 0.011, 0.007), True),
    "corn": ("#b98322", "plaster_lime", "#f7c648", (0.013, 0.0105, 0.006), True),
    "beans": ("#3f1512", "ceramic_glazed", "#63241f", (0.019, 0.0105, 0.008), True),
}

TIN = "#c3c8c7"
US_LY = [0.0, 0.013, 0.04, 0.08, 0.13, 0.2, 0.28, 0.37, 0.46, 0.54, 0.61, 0.67, 0.72, 0.765, 0.805, 0.845, 0.885,
         0.92, 0.95, 0.975, 1.0]
US_ST = [0.075, 0.105, 0.135, 0.165, 0.2, 0.24, 0.29, 0.36, 0.45, 0.54, 0.63, 0.71, 0.78, 0.835, 0.88, 0.915, 0.945,
         0.975, 1.0]
US_OP = [0.115, 0.15, 0.185, 0.22, 0.26, 0.31, 0.38, 0.48, 0.6, 0.72, 0.83, 0.92]
RUF_Q = [0.2, 0.45, 0.72, 1.0]                       # ruffle rings beyond the tie
RUF_IN = [(0.82, 0.86), (0.42, 0.58), (0.1, 0.06)]     # (q, scale): the cloth mouth folding back into the throat


def _ss(a, b, x):
    t = np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _spow(v, e):
    return np.sign(v) * np.abs(v) ** e


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _gauss(v, sig):
    r = int(3 * sig) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    return np.convolve(np.pad(v, r, mode="edge"), k / k.sum(), mode="valid")


class Lump:
    """Cheap smooth 3D pseudo-noise (sum of random plane waves), vectorised, ~[-1, 1]."""

    def __init__(self, rng, scale, n=4):
        self.k = []
        for _ in range(n):
            d = _unit([rng.u(-1, 1), rng.u(-1, 1), rng.u(-1, 1) + 1e-3])
            self.k.append((d * (2 * pi / (scale * rng.u(0.8, 1.5))), rng.u(0, 2 * pi)))

    def __call__(self, P):
        s = 0.0
        for kv, ph in self.k:
            s = s + np.sin(P @ kv + ph)
        return s * (1.5 / len(self.k))


class Field:
    """Height field of the pile (top surface): lying sacks settle on it, standing ones lean against it."""

    def __init__(self, half=2.4, cell=0.015):
        self.half, self.cell = half, cell
        self.n = int(round(2 * half / cell)) + 1
        ax = np.linspace(-half, half, self.n)
        self.X, self.Y = np.meshgrid(ax, ax)
        self.H = np.zeros((self.n, self.n))
        self.base = np.zeros((self.n, self.n))

    def window(self, cx, cy, r):
        i0 = max(0, int((cx - r + self.half) / self.cell))
        i1 = min(self.n, int((cx + r + self.half) / self.cell) + 2)
        j0 = max(0, int((cy - r + self.half) / self.cell))
        j1 = min(self.n, int((cy + r + self.half) / self.cell) + 2)
        return slice(j0, j1), slice(i0, i1)

    def sample(self, A, x, y):
        fx = np.clip((np.asarray(x) + self.half) / self.cell, 0, self.n - 1.001)
        fy = np.clip((np.asarray(y) + self.half) / self.cell, 0, self.n - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        return (A[iy, ix] * (1 - tx) * (1 - ty) + A[iy, ix + 1] * tx * (1 - ty)
                + A[iy + 1, ix] * (1 - tx) * ty + A[iy + 1, ix + 1] * tx * ty)

    def clearance(self, gap):
        """Lowest allowed height for the next sack: a few mm clear of whatever already lies there."""
        return np.where(self.H > self.base + 0.004, self.H + gap, self.base + 0.0015)

    @staticmethod
    def _f3(W, fn):
        Wp = np.pad(W, 1, mode="edge")
        out = W.copy()
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                out = fn(out, Wp[dy:dy + W.shape[0], dx:dx + W.shape[1]])
        return out

    def splat(self, P, sr=8, sc=6):
        """Rasterise the ring surface P[nr, N, 3] (closed around N) into the height field."""
        Pc = np.concatenate([P, P[:, :1]], axis=1)
        A, B, C, E = Pc[:-1, :-1, None, :], Pc[1:, :-1, None, :], Pc[:-1, 1:, None, :], Pc[1:, 1:, None, :]
        U, V = np.meshgrid((np.arange(sr) + 0.5) / sr, (np.arange(sc) + 0.5) / sc, indexing="ij")
        U, V = U.reshape(1, 1, -1, 1), V.reshape(1, 1, -1, 1)
        S = (A * (1 - U) * (1 - V) + B * U * (1 - V) + C * (1 - U) * V + E * U * V).reshape(-1, 3)
        S = np.concatenate([S, P.reshape(-1, 3)], axis=0)
        ix = np.clip(np.rint((S[:, 0] + self.half) / self.cell).astype(int), 0, self.n - 1)
        iy = np.clip(np.rint((S[:, 1] + self.half) / self.cell).astype(int), 0, self.n - 1)
        np.maximum.at(self.H, (iy, ix), S[:, 2])
        j0, j1 = max(0, iy.min() - 2), min(self.n, iy.max() + 3)
        i0, i1 = max(0, ix.min() - 2), min(self.n, ix.max() + 3)
        W = self.H[j0:j1, i0:i1]
        self.H[j0:j1, i0:i1] = np.maximum(W, self._f3(self._f3(W, np.maximum), np.minimum))   # close pinholes

    def stats(self, cx, cy, yaw, ha, hb, hi=1.0):
        """Under a sack footprint: covered fraction, mean / std height, fraction above hi, base unevenness."""
        sy, sx = self.window(cx, cy, max(ha, hb) + 0.04)
        X, Y = self.X[sy, sx] - cx, self.Y[sy, sx] - cy
        c, s = cos(yaw), sin(yaw)
        m = (np.abs((X * c + Y * s) / ha) ** 3 + np.abs((-X * s + Y * c) / hb) ** 3) < 1.0
        rel = (self.H[sy, sx] - self.base[sy, sx])[m]
        if rel.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return float((rel > 0.05).mean()), float(rel.mean()), float(rel.std()), float((rel > hi).mean()), float(self.base[sy, sx][m].std())


class Dim:
    """One sack's dimensions and its private irregularities."""

    def __init__(self, L, fill, rng, n0):
        self.L = L
        self.sz = L / 0.75
        self.fill = fill
        self.Wd = 0.56 * L * rng.vary(1.0, 0.04)
        T = L * (0.15 + 0.14 * fill) * rng.vary(1.0, 0.05)           # 0.85 m sack: ~0.23 m thick lying down
        self.Tc, self.Lw = 0.57 * T, 0.43 * T                          # above / below the side seam
        self.rn = max(0.03, 0.055 * L)
        self.Lr = 0.15 * L * rng.vary(1.0, 0.15)
        self.k = 5 if n0 >= 28 else 4                                   # gathers round the neck
        self.ph0 = rng.u(0, 2 * pi)
        self.ph = [rng.u(0, 2 * pi) for _ in range(12)]
        self.lump = Lump(rng, 0.46 * self.sz)
        self.lump2 = Lump(rng, 0.55 * self.sz, n=3)
        self.lamp = rng.u(0.03, 0.08)                                  # a few broad bulges, different per sack
        self.fold_amp = (0.011 + 0.035 * (1 - fill)) * self.sz * rng.u(0.7, 1.3)
        self.fold_r = (0.045 + 0.09 * (1 - fill)) * rng.u(0.8, 1.25)
        self.Rx = 0.5 * self.Wd
        self.Ry = 0.39 * self.Wd * (0.8 + 0.25 * fill)
        self.Ht = 0.84 * L
        self.flop = _unit([rng.u(-1, 1), rng.u(-1, 1)])
        self.flop_amt = rng.u(0.15, 0.5)

    def pleat(self, th):
        return (0.62 * np.cos(self.k * th + self.ph0 + 0.6 * np.sin(2 * th + self.ph[11])) + 0.25 * np.cos(3 * th + self.ph[1])
                + 0.3 * np.cos((self.k + 2) * th + self.ph[0]))

    def pleat_amp(self, u, q, ruf, d):
        """Depth of the gathers: tension wrinkles fading down the shoulders, pinched flat under the rope,
        opening into deep pleats in the ruffle. d = distance from the tie (m)."""
        A = 0.2 * np.exp(-(1 - u) / 0.05) + 0.13 * np.exp(-(1 - u) / 0.22)
        A = np.where(ruf, 0.2 + 0.08 * q, A)
        return A * (1 - 0.55 * np.exp(-(d / (0.016 * self.sz)) ** 2))

    def base_folds(self, th, z):
        """Compression folds sagging round the foot of a standing sack (radial factor)."""
        sz = self.sz
        off = 0.02 * sz * np.sin(2 * th + self.ph[7]) + 0.012 * sz * np.sin(3 * th + self.ph[8])
        win = _ss(0.03 * sz, 0.07 * sz, z) * (1 - _ss(0.17 * sz, 0.25 * sz, z))
        x = 2 * pi * (z - off[None, :]) / (0.08 * sz)
        return 1 + self.fold_r * (np.sin(x) + 0.3 * np.sin(2 * x + 1.0)) * win * (0.55 + 0.45 * np.sin(th + self.ph[9]))[None, :]

    def foot(self, th, u):
        """Slumped, slightly irregular footprint."""
        return 1 + (0.05 * np.sin(2 * th + self.ph[5]) + 0.03 * np.sin(3 * th + self.ph[6]))[None, :] * (1 - _ss(0.0, 0.4, u))


def _columns(n0, stripes):
    """Angles around the sack (radians) with narrow extra columns on the face for the woven stripes."""
    base = [k * 360.0 / n0 for k in range(n0)]
    offs, flags = {0: ([], []), 1: ([-9, -4, 4, 9], [0, 1, 0]), 2: ([-10, -6, -2, 2, 6, 10], [0, 1, 0, 1, 0]),
                   3: ([-12, -9, -7, -2.5, 2.5, 7, 9, 12], [0, 1, 0, 1, 0, 1, 0])}[stripes]
    if offs:
        lim = max(offs) + 3
        base = sorted([a for a in base if abs(a - 90) >= lim] + [90 + o for o in offs])
    st = np.zeros(len(base), dtype=bool)
    for j, a in enumerate(base):
        b = base[(j + 1) % len(base)]
        if offs and b > a:
            mid = (a + b) / 2 - 90
            for qi, fl in enumerate(flags):
                if fl and offs[qi] <= mid <= offs[qi + 1]:
                    st[j] = True
    return np.radians(base), st


# ------------------------------------------------------------------------------------------------ sack shapes
def lying_shape(D, th):
    """Section rings of a lying sack in its own frame: station t along the length, b across, c up."""
    L, Lr, rn, sz = D.L, D.Lr, D.rn, D.sz
    ts = [-0.016] + [u * L for u in US_LY] + [L + q * Lr for q in RUF_Q] + [L + q * Lr for q, _ in RUF_IN]
    sc = [1.0] * (1 + len(US_LY) + len(RUF_Q)) + [s for _, s in RUF_IN]
    t = np.array(ts)[:, None]
    s = np.array(sc)[:, None]
    u = np.clip(t / L, 0, 1)
    q = np.clip((t - L) / Lr, 0, 1)
    ruf = t > L + 1e-6
    g = np.sqrt(1 - (1 - np.clip(u / 0.16, 0, 1)) ** 2)            # blunt pillow end: pinches only at the seam
    sN = np.clip((u - 0.77) / 0.225, 0, 1) ** 1.8                  # full round shoulders, gathered late into the neck
    pin = 1 - 0.1 * np.exp(-((t - L) / (0.022 * sz)) ** 2)        # waist under the rope
    wb = D.Wd / 2 * (0.95 + 0.05 * g) * (1 - sN) + rn * pin * sN
    tc = (0.005 + (D.Tc - 0.005) * g) * (1 - sN) + rn * pin * sN
    lw = (0.004 + (D.Lw - 0.004) * g) * (1 - sN) + rn * pin * sN
    rr = rn * (0.92 + 0.6 * q ** 0.85)                             # the fanned ruffle, sagging a little flat
    wb = np.where(ruf, rr * 1.15, wb)
    tc = np.where(ruf, rr * (0.9 - 0.25 * q), tc)
    lw = np.where(ruf, rr * (0.85 - 0.3 * q), lw)
    lip = t < 0
    tc = np.where(lip, 0.004, tc)
    lw = np.where(lip, 0.004, lw)
    body = (1 - sN) * g * (~ruf)
    cs, sn = np.cos(th)[None, :], np.sin(th)[None, :]
    up = sn >= 0
    nl = 2.0 + 1.2 * body
    nexp = np.where(up, 2.0 + 0.6 * body, nl)                      # plump rounded flanks, flatter underside
    ce, se = _spow(cs, 2 / nexp), _spow(sn, 2 / nexp)
    A = D.pleat_amp(u, q, ruf, t - L) * (t > 0.35 * L)
    m = 1 + A * D.pleat(th)[None, :]
    b = wb * ce * m * s
    c = np.where(up, tc, lw) * se * m * s
    pk = (1 - _ss(0.0, 0.2, u)) * (t >= 0)                         # puckers pulled by the bottom seam
    c = c * (1 + 0.14 * pk * np.cos(5.0 * pi * ce + D.ph[2]))
    wg = (1 - _ss(0.0, 0.1, u)) * _ss(0.6, 1.0, np.abs(ce))       # corner ears
    da = -0.035 * sz * wg
    b = b + np.sign(ce) * 0.03 * sz * wg
    P0 = np.stack([t - L / 2 + 0 * b, b, c], -1)
    lwt = np.sin(pi * u) ** 0.7 * (s >= 1) * (~ruf)
    c = c * (1 + D.lamp * D.lump(P0) * lwt)
    b = b * (1 + 0.4 * D.lamp * D.lump2(P0) * lwt)
    zone = _ss(0.4, 0.54, u) * (1 - _ss(0.74, 0.87, u)) * (~ruf)   # slack cloth folding behind the shoulders
    off = 0.03 * sz * np.sin(2 * th + D.ph[7]) + 0.02 * sz * np.sin(3 * th + D.ph[8])
    c = c + D.fold_amp * np.sin(2 * pi * (t - off[None, :]) / (0.15 * sz)) * zone * np.clip(sn + 0.3, 0, 1)
    da = da + np.where(ruf, Lr * q * (0.22 * np.sin(th + D.ph[3]) + 0.1 * np.sin(3 * th + D.ph[4]))[None, :], 0.0)
    b[:, 0] += 0.006 * ((t >= 0) & (t <= 0.9 * L))[:, 0] * s[:, 0]  # side seam ridge
    clow = lw * (1 - np.clip(np.abs(b) / np.maximum(wb, 1e-4), 0, 1) ** nl) ** (1 / nl)
    n_us = len(US_LY)
    return {"t": t[:, 0], "b": b, "c": c, "da": da + 0 * b, "hb": np.clip(c + clow, 0, None), "low": lw[:, 0],
            "wb": wb[:, 0], "ce": ce, "und": (sn < -0.05), "bodyw": (body[:, 0] > 0.3).astype(float),
            "info": {"tie": n_us, "seam0": 1, "seam1": 1 + US_LY.index(0.845), "outer": 1 + n_us + len(RUF_Q)}}


def settle(D, S, F, G, cx, cy, yaw, roll_max, stiff):
    """Rest the stuffed sack on the clearance field G. Each section stays rigid (it rolls and pitches), the
    centre line is the smooth upper envelope of what the sections need (so the sack bends only gently and
    bridges gaps), flanks are squeezed in beside taller neighbours, and finally the underside is flattened
    against the support wherever it would otherwise sink in."""
    t, b0, c0, da0, low, wbr = S["t"], S["b"], S["c"], S["da"], S["low"], S["wb"]
    nr = len(t)
    co, si = cos(yaw), sin(yaw)
    tg = np.linspace(t.min(), t.max(), 72)
    dt = tg[1] - tg[0]
    order = np.argsort(t)
    inner = np.abs(S["ce"]) < 0.88
    und = S["und"]
    sq = (0.2 * low + 0.003)[:, None] * und                       # how far the underside may flatten
    h = np.zeros(nr)
    rho = np.zeros(nr)
    pit = np.zeros(nr)
    a = t - D.L / 2
    shr = np.ones((nr, 2))
    dtt = np.abs(t[:, None] - t[None, :])
    for it in range(7):
        bs = b0 * np.where(b0 > 0, shr[:, :1], shr[:, 1:])
        cr_, sr_ = np.cos(rho)[:, None], np.sin(rho)[:, None]
        br = bs * cr_ - c0 * sr_
        cr = bs * sr_ + c0 * cr_
        cp, sp = np.cos(pit)[:, None], np.sin(pit)[:, None]
        al = a[:, None] + da0 * cp - cr * sp
        zl = da0 * sp + cr * cp
        x = cx + al * co - br * si
        y = cy + al * si + br * co
        g = F.sample(G, x, y)
        if it == 6:
            break
        need = np.where(inner, g - zl - sq, -1e9).max(axis=1)
        ng = np.interp(tg, t[order], need[order])
        hg = ng.copy()
        for _ in range(8):
            hg = np.maximum(ng, _gauss(hg, stiff / dt))
        pg = np.arcsin(np.clip(np.gradient(hg, dt), -0.45, 0.45))
        ag = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(pg[1:]) + np.cos(pg[:-1])) * dt)])
        ag -= np.interp(D.L / 2, tg, ag)                           # arc length kept: the sack does not stretch
        h, pit, a = np.interp(t, tg, hg), np.interp(t, tg, pg), np.interp(t, tg, ag)
        # roll: follow the cross slope of the support under the section
        w = (inner & und).astype(float)
        gm = (g * w).sum(1) / w.sum(1)
        bm = (br * w).sum(1) / w.sum(1)
        cov = ((g - gm[:, None]) * (br - bm[:, None]) * w).sum(1)
        var = ((br - bm[:, None]) ** 2 * w).sum(1)
        tgt = np.arctan(cov / np.maximum(var, 1e-6))
        K = np.exp(-0.5 * (dtt / (0.3 * D.L)) ** 2) * (wbr ** 2 * S["bodyw"])[None, :]
        rho = 0.5 * rho + 0.5 * np.clip((K * tgt[None, :]).sum(1) / K.sum(1), -roll_max, roll_max)
        # flanks: squeeze in where a taller neighbour lies beside this sack
        fs = np.linspace(0.74, 1.0, 9)
        for col, sd in ((0, 1.0), (1, -1.0)):
            bb = sd * wbr[:, None] * fs[None, :]
            xr, zr = bb * np.cos(rho)[:, None], bb * np.sin(rho)[:, None]
            xx = cx + a[:, None] * co - xr * si
            yy = cy + a[:, None] * si + xr * co
            bad = (F.sample(G, xx, yy) > h[:, None] + zr + 0.03) & (wbr[:, None] > 0.1)
            first = np.where(bad.any(1), bad.argmax(1), len(fs))
            f = np.where(first >= len(fs), 1.0, fs[np.clip(first - 1, 0, len(fs) - 1)])
            fm = np.where(dtt < 0.07, f[None, :], 1.0).min(axis=1)
            K2 = np.exp(-0.5 * (dtt / 0.05) ** 2)
            shr[:, col] = np.minimum((K2 * fm[None, :]).sum(1) / K2.sum(1), 1.0)
    z = np.maximum(h[:, None] + zl, g + 0.45 * S["hb"])
    bd = S["bodyw"] > 0
    return np.stack([x, y, z], -1), {"bend": float(pit[bd].max() - pit[bd].min()), "tilt": float(abs(pit[bd].mean())),
                                     "roll": float(np.abs(rho[bd]).max())}


def standing_shape(D, th):
    Ht, Lr, rn, sz = D.Ht, D.Lr, D.rn, D.sz
    zb = [0.0, 0.0, 0.012 * sz, 0.035 * sz]
    rho = [0.45, 0.9, 0.99, 1.0]
    zs = zb + [u * Ht for u in US_ST] + [Ht + q * Lr for q in RUF_Q] + [Ht + q * Lr for q, _ in RUF_IN]
    rh = rho + [1.0] * (len(US_ST) + len(RUF_Q)) + [s for _, s in RUF_IN]
    z = np.array(zs)[:, None]
    rh = np.array(rh)[:, None]
    u = np.clip(z / Ht, 0, 1)
    q = np.clip((z - Ht) / Lr, 0, 1)
    ruf = z > Ht + 1e-6
    belly = 1 + 0.09 * np.exp(-((u - 0.2) / 0.17) ** 2) - 0.06 * _ss(0.42, 0.75, u)   # slumps: widest low down
    st = np.clip((u - 0.73) / 0.265, 0, 1) ** 1.8
    pin = 1 - 0.1 * np.exp(-((z - Ht) / (0.022 * sz)) ** 2)
    Rx = belly * D.Rx * (1 - st) + rn * pin * st
    Ry = belly * D.Ry * (1 - st) + rn * pin * st
    rr = rn * (0.92 + 0.6 * q ** 0.85)
    Rx = np.where(ruf, rr, Rx)
    Ry = np.where(ruf, rr * 0.85, Ry)
    A = D.pleat_amp(u, q, ruf, z - Ht)
    nexp = 2.7 - 0.7 * _ss(0.3, 0.9, u)
    cs, sn = np.cos(th)[None, :], np.sin(th)[None, :]
    ce, se = _spow(cs, 2 / nexp), _spow(sn, 2 / nexp)
    m = (1 + A * D.pleat(th)[None, :]) * D.foot(th, u) * D.base_folds(th, z)
    x = Rx * rh * ce * m
    y = Ry * rh * se * m
    zz = z + 0 * x
    wg = (1 - _ss(0.0, 0.11, u)) * _ss(0.65, 1.0, np.abs(ce)) * (rh > 0.85)       # bottom corner ears
    x = x + np.sign(ce) * 0.05 * sz * wg
    y = y * (1 - 0.45 * wg)
    P0 = np.stack([x, y, zz], -1)
    lwt = np.sin(pi * u) ** 0.6 * (~ruf)
    k = 1 + 0.8 * D.lamp * D.lump(P0) * lwt
    x, y = x * k, y * k
    y = y - 0.02 * sz * np.sin(pi * np.clip(u / 0.75, 0, 1)) * (~ruf) * rh         # belly sags to the back face
    x[:, 0] += 0.006 * ((u > 0.03) & (u < 0.86) & (~ruf)).astype(float)[:, 0]      # side seam ridge
    curl = (q ** 2 * (0.5 + 0.5 * np.sin(2 * th + D.ph[10]))[None, :]) * ruf       # one side of the mouth lolls open
    x = x * (1 + 0.3 * curl)
    y = y * (1 + 0.3 * curl)
    zz = zz - 0.22 * Lr * curl
    nk = _ss(0.68, 1.0, u)                                                          # the neck is never dead centre
    x = x + D.flop[0] * 0.035 * sz * nk
    y = y + D.flop[1] * 0.035 * sz * nk
    fq = q ** 1.6 * D.flop_amt                                                      # the ruffle flops over
    x = x + D.flop[0] * 0.6 * Lr * fq
    y = y + D.flop[1] * 0.6 * Lr * fq
    zz = zz - 0.3 * Lr * q ** 2 * D.flop_amt + np.where(ruf, Lr * q * (0.15 * np.sin(th + D.ph[3]) + 0.08 * np.sin(3 * th + D.ph[4]))[None, :], 0.0)
    nb = len(zb) + len(US_ST)
    return {"x": x, "y": y, "z": zz, "z0": z, "tw": np.where(ruf, 1.0, _ss(0.06, 0.55, u)),
            "info": {"tie": nb - 1, "seam0": 3, "seam1": len(zb) + US_ST.index(0.835), "body": (2, nb), "outer": nb + len(RUF_Q)}}


def open_shape(D, th):
    sz = D.sz
    Hr0 = 0.66 * D.L
    rr0 = 0.027 * sz
    Rx0, Ry0 = 0.47 * D.Wd, 0.42 * D.Wd
    zb = [0.0, 0.0, 0.012 * sz, 0.035 * sz]
    rho = [0.45, 0.9, 0.99, 1.0]
    al = [215, 270, 330, 30, 90, 150]
    nb = len(zb) + len(US_OP)
    nr = nb + 1 + len(al) + 2
    N = len(th)
    rrv = rr0 * (1 + 0.2 * np.sin(2 * th + D.ph[3]) + 0.12 * np.sin(5 * th + D.ph[4]))
    Hrv = Hr0 + 0.016 * sz * np.sin(th + D.ph[5]) + 0.007 * sz * np.sin(3 * th + D.ph[6])
    ztop = Hrv - 1.57 * rrv                          # where the body disappears under the roll
    x, y, z = np.zeros((nr, N)), np.zeros((nr, N)), np.zeros((nr, N))
    cs, sn = np.cos(th), np.sin(th)

    def ring(u, rh, dr, zz):
        belly = 1 + 0.08 * np.exp(-((u - 0.25) / 0.22) ** 2) - 0.07 * _ss(0.5, 1.0, u)
        nexp = 2.5 - 0.4 * _ss(0.2, 0.8, u)
        ce, se = _spow(cs, 2 / nexp), _spow(sn, 2 / nexp)
        mm = (D.foot(th, np.array([[u]])) * D.base_folds(th, np.array([[zz]])))[0] if zz is not None else 1.0
        rx, ry = belly * Rx0 * rh * mm, belly * Ry0 * rh * mm
        nx, ny = ce * ry, se * rx
        nl = np.sqrt(nx * nx + ny * ny) + 1e-9
        return rx * ce + dr * nx / nl, ry * se + dr * ny / nl, ce

    i = 0
    for zz, rh in zip(zb, rho):
        xx, yy, ce = ring(zz / Hr0, rh, 0.0, zz)
        wg = (1 - _ss(0.0, 0.12, zz / Hr0)) * _ss(0.65, 1.0, np.abs(ce)) * (rh > 0.85)
        x[i], y[i], z[i] = xx + np.sign(ce) * 0.05 * sz * wg, yy * (1 - 0.45 * wg), zz
        i += 1
    for u in US_OP + [1.0]:
        zz = u * ztop
        xx, yy, ce = ring(u, 1.0, 0.0, float(u * Hr0))
        wg = (1 - _ss(0.0, 0.12, u)) * _ss(0.65, 1.0, np.abs(ce))
        P0 = np.stack([xx, yy, zz], -1)
        k = 1 + 0.8 * D.lamp * D.lump(P0) * np.sin(pi * min(u, 0.95)) ** 0.6
        x[i], y[i], z[i] = xx * k + np.sign(ce) * 0.05 * sz * wg, yy * k * (1 - 0.45 * wg), zz
        i += 1
    for a in al:
        dr = 0.85 * rrv + rrv * cos(radians(a))
        xx, yy, ce = ring(1.0, 1.0, dr, None)
        x[i], y[i], z[i] = xx, yy, Hrv - rrv + rrv * sin(radians(a))
        i += 1
    xx, yy, ce = ring(1.0, 1.0, -0.005, None)
    x[i], y[i], z[i] = xx, yy, Hrv - 1.5 * rrv
    i_in = i
    i += 1
    xx, yy, ce = ring(1.0, 1.0, -0.007, None)
    x[i], y[i], z[i] = xx, yy, Hr0 - 3.4 * rr0
    x[:, 0] += 0.006 * ((z[:, 0] > 0.04) & (np.arange(nr) < nb))
    return {"x": x, "y": y, "z": z, "z0": z, "tw": 0.0 * z,
            "info": {"inner": i_in, "seam0": 3, "seam1": nb - 1, "body": (2, nb + 4), "rim": Hr0, "rr": rr0, "nb": nb, "outer": nr}}


def upright_pose(S, cx, cy, yaw, lean_dir, phi):
    """Stand the sack at (cx, cy) leaning by phi towards lean_dir: sheared centre line (the base stays
    planted), rings tilting progressively up the body."""
    x, y, zz, z0 = S["x"], S["y"], S["z"], S["z0"]
    co, si = cos(yaw), sin(yaw)
    dl = np.array([lean_dir[0] * co + lean_dir[1] * si, -lean_dir[0] * si + lean_dir[1] * co])
    tau = phi * S["tw"]
    rpar = x * dl[0] + y * dl[1]
    sh = tan(phi) * z0
    x2 = x + dl[0] * (rpar * (np.cos(tau) - 1) + sh)
    y2 = y + dl[1] * (rpar * (np.cos(tau) - 1) + sh)
    z2 = zz - rpar * np.sin(tau)
    return np.stack([cx + x2 * co - y2 * si, cy + x2 * si + y2 * co, np.maximum(z2, 0.0015)], -1)


# ------------------------------------------------------------------------------------------------ mesh helpers
def ring_mesh(name, P, st_cols, mats):
    """Closed tube mesh from rings P[nr, N, 3] with fan caps, fabric UVs in metres, stripe columns."""
    nr, N, _ = P.shape
    V = np.concatenate([P.reshape(-1, 3), P[0].mean(axis=0)[None, :], P[-1].mean(axis=0)[None, :]], axis=0)
    c0, c1 = nr * N, nr * N + 1
    i0 = np.arange(nr - 1)[:, None]
    j0 = np.arange(N)[None, :]
    j1 = (j0 + 1) % N
    quads = np.stack([i0 * N + j0 + 0 * i0, i0 * N + j1, (i0 + 1) * N + j1, (i0 + 1) * N + j0], axis=2).reshape(-1, 4)
    jj = np.arange(N)
    jn = (jj + 1) % N
    tris = np.concatenate([np.stack([np.full(N, c0), jn, jj], 1),
                           np.stack([np.full(N, c1), (nr - 1) * N + jj, (nr - 1) * N + jn], 1)], axis=0)
    per = np.linalg.norm(np.roll(P, -1, axis=1) - P, axis=2)            # (nr, N)
    im = int(np.argmax(per.sum(axis=1)))
    U = np.concatenate([[0.0], np.cumsum(per[im])])
    Vv = np.concatenate([[0.0], np.cumsum(np.linalg.norm(P[1:] - P[:-1], axis=2).mean(axis=1))])
    U0 = np.broadcast_to(U[:-1][None, :], (nr - 1, N))
    U1 = np.broadcast_to(U[1:][None, :], (nr - 1, N))
    V0 = np.broadcast_to(Vv[:-1][:, None], (nr - 1, N))
    V1 = np.broadcast_to(Vv[1:][:, None], (nr - 1, N))
    uvq = np.stack([np.stack([U0, V0], -1), np.stack([U1, V0], -1), np.stack([U1, V1], -1), np.stack([U0, V1], -1)],
                   axis=2).reshape(-1, 2)
    um = (U[:-1] + U[1:]) / 2
    uvt = np.concatenate([np.stack([np.stack([um, np.full(N, -0.02)], -1), np.stack([U[1:], np.zeros(N)], -1),
                                    np.stack([U[:-1], np.zeros(N)], -1)], axis=1).reshape(-1, 2),
                          np.stack([np.stack([um, np.full(N, Vv[-1] + 0.02)], -1), np.stack([U[:-1], np.full(N, Vv[-1])], -1),
                                    np.stack([U[1:], np.full(N, Vv[-1])], -1)], axis=1).reshape(-1, 2)], axis=0)
    nq, nt = len(quads), len(tris)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.astype(np.float32).ravel())
    me.loops.add(nq * 4 + nt * 3)
    me.loops.foreach_set("vertex_index", np.concatenate([quads.ravel(), tris.ravel()]).astype(np.int32))
    me.polygons.add(nq + nt)
    me.polygons.foreach_set("loop_start", np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nt) * 3]).astype(np.int32))
    for mm in mats:
        me.materials.append(mm)
    mi = np.concatenate([np.tile(st_cols.astype(np.int32), nr - 1), np.zeros(nt, dtype=np.int32)])
    me.polygons.foreach_set("material_index", mi)
    me.update(calc_edges=True)
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.concatenate([uvq, uvt]).astype(np.float32).ravel())
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    uv_keep(o)
    smooth(o, 180)
    return o


def _basis(loc, X, Y, Z):
    return Matrix(((X[0], Y[0], Z[0], loc[0]), (X[1], Y[1], Z[1], loc[1]), (X[2], Y[2], Z[2], loc[2]), (0, 0, 0, 1)))


def add_stitches(M, pts, cross, nrm, spacing, length, skew, rng, lift=0.0007):
    """Whip stitches of jute twine across a seam line (polyline pts with per-point cross / normal vectors)."""
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(cum[-1] / spacing)
    for qi in range(n):
        d = (qi + 0.5) * cum[-1] / n
        i = int(np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1))
        f = (d - cum[i]) / max(seg[i], 1e-9)
        p = pts[i] * (1 - f) + pts[i + 1] * f
        cr = _unit(cross[i] * (1 - f) + cross[i + 1] * f)
        nn = _unit(nrm[i] * (1 - f) + nrm[i + 1] * f)
        tg = _unit(pts[i + 1] - pts[i])
        a = radians(skew + rng.jitter(0.0, 9.0))
        Z = _unit(cr * cos(a) + tg * sin(a))
        X = _unit(np.cross(nn, Z))
        Y = np.cross(Z, X)
        ln = length * rng.vary(1.0, 0.15)
        M.append(_basis(p + nn * lift - Z * ln * 0.5, X, Y, Z * ln))


def side_seam(M, P, i0, i1, spacing, rng):
    N = P.shape[1]
    idx = list(range(i0, i1 + 1))
    pts = P[idx, 0]
    cross = P[idx, 1] - P[idx, N - 1]
    out = pts - P[idx].mean(axis=1)
    nrm = []
    for k in range(len(idx)):
        tg = pts[min(k + 1, len(idx) - 1)] - pts[max(k - 1, 0)]
        nn = _unit(np.cross(tg, cross[k]))
        if np.dot(nn, out[k]) < 0:
            nn = -nn
        nrm.append(nn)
    add_stitches(M, pts, cross, np.array(nrm), spacing, 0.017, 32.0, rng)


def tie_rope(name, P, it, hint, rr, rng, npt, sides=5):
    """Two turns of hemp rope round the gathered neck, a knot and two loose ends."""
    C = P[it].mean(axis=0)
    ax = _unit(P[it + 1].mean(axis=0) - P[it - 1].mean(axis=0))
    rad = np.linalg.norm(P[it] - C, axis=1)
    R = 0.55 * rad.mean() + 0.45 * rad.max() + rr * 0.6
    h = np.asarray(hint, dtype=float)
    e1 = h - ax * np.dot(h, ax)
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(ax, (1.0, 0.0, 0.0))
    e1 = _unit(e1)
    e2 = np.cross(ax, e1)
    turns = 2
    pts = []
    for k in range(npt + 1):
        ps = 2 * pi * turns * k / npt
        off = (k / npt - 0.5) * turns * rr * 1.9
        pts.append(tuple(C + ax * off + (e1 * cos(ps) + e2 * sin(ps)) * R * (1 + 0.04 * sin(3 * ps + 1.0))))
    out = [sweep(name + "_coil", pts, rr, sides=sides, smooth_path=False)]
    K = C + e1 * (R + rr * 0.7)
    out.append(sphere(name + "_knot", rr * 2.0, loc=tuple(K), segments=6, rings=4, scale=(1.0, 1.0, 0.85)))
    g = np.array((0.0, 0.0, -1.0))
    for sgn in (-1.0, 1.0):
        ln = rng.u(0.06, 0.1)
        d2 = e2 * sgn
        q0 = K
        q1 = K + d2 * ln * 0.4 + e1 * ln * 0.3
        q2 = K + d2 * ln * 0.8 + e1 * ln * 0.4 + g * ln * 0.2
        q3 = K + d2 * ln * 0.95 + e1 * ln * 0.45 + g * ln * 0.85 + ax * rng.u(-0.02, 0.02)
        path = [(float(v[0]), float(v[1]), max(float(v[2]), rr + 0.002)) for v in bezier_path(tuple(q0), tuple(q1), tuple(q2), tuple(q3), n=5)]
        out.append(sweep(name + "_end", path, rr * 0.9, sides=sides, smooth_path=False,
                         scale_fn=lambda t: 1.0 + 0.7 * max(0.0, (t - 0.8) / 0.2)))
    return out, C


# ------------------------------------------------------------------------------------------------ build
def build(p, rng):
    n = int(p.count)
    L0 = float(p.sack_size)
    sz = L0 / 0.75
    age = float(p.age)
    style = p.style
    F = Field()
    rp = rng.fork("plan")
    lod = 0 if n <= 5 else (1 if n == 6 else 2)              # eight soft bodies share the same triangle budget

    cloth = np.array(hex_rgb(p.cloth))
    dye = np.array(hex_rgb(p.stripe_color))
    dye = dye + (dye.mean() - dye) * 0.3                      # woven-in dye: desaturated and faded into the jute
    ks = 0.74 - 0.3 * age
    m_a = mat("burlap", p.cloth)
    m_b = mat("burlap", tuple(float(v) for v in cloth * np.array((0.74, 0.7, 0.64))))   # darker lot + seam twine
    m_s = mat("burlap", tuple(float(v) for v in cloth * (1 - ks) + dye * ks))
    m_rope = mat("rope")
    base_tint, k_lib, k_tint, k_size, k_round = CONTENTS[p.contents]
    m_fill = mat("plaster_rough", base_tint, scale=3.0)
    m_kern = mat(k_lib, k_tint)

    n0 = (32, 28, 24)[lod]
    th, st_cols = _columns(n0, int(p.stripes))
    N = len(th)
    n_closed = n - 1
    W0 = 0.56 * L0
    T0 = L0 * (0.15 + 0.14 * float(p.fill))

    # ---- pallet footprint (the laid sacks rest on it)
    pal = None
    if p.pallet:
        if style == "stacked":
            pw = pd = max(1.16 * L0, 2.1 * W0) + 0.04
        elif style == "heap":
            pw, pd = 1.75 * L0 + 0.1, 1.45 * L0 + 0.1
        else:
            pw, pd = 1.2 * L0 + 0.12, (1.25 if n_closed <= 5 else 2.2) * W0 + 0.1
        pal = (pw, pd, 0.097)
        m = (np.abs(F.X) < pw / 2) & (np.abs(F.Y) < pd / 2)
        F.base[m] = pal[2]
        F.H[m] = pal[2]

    sacks = []          # dicts: kind, D, P, info, ...

    def new_dim(i):
        r = rng.fork(f"sack{i}")
        return Dim(L0 * r.vary(1.0, 0.05), float(p.fill) * r.vary(1.0, 0.08), r, n0), r

    # ---- plan the laid sacks
    lying = []          # (cx, cy, yaw)
    if style == "stacked":
        base_yaw = radians(rp.jitter(0, 3))
        free_side = -1                                        # long layers point their necks away from this side
        n_lean = 1 if n_closed >= 7 else 0
        n_ly = n_closed - n_lean
        i, layer = 0, 0
        while i < n_ly:                                       # two abreast, each layer crossed over the last
            yl = base_yaw + (pi / 2) * (layer % 2)
            mrow = min(2, n_ly - i)
            for k in range(mrow):
                off = (k - (mrow - 1) / 2) * (W0 * 1.05 + 0.012)
                sl = rp.jitter(0, 0.02)
                lying.append((-sin(yl) * off + cos(yl) * sl, cos(yl) * off + sin(yl) * sl,
                              yl + radians(rp.jitter(0, 3.5)) + pi * (((k + layer // 2) % 2) if layer % 2 else (free_side > 0))))
                i += 1
            layer += 1
        n_stand = n_lean
    elif style == "leaning":
        base_yaw = radians(rp.u(-22, 22))
        n_ly = 1 if n_closed <= 2 else (2 if n_closed <= 5 else 3)
        if n_ly == 1:
            lying.append((-0.3 * L0, 0.04 * L0, base_yaw * 0.5 + radians(14)))
        elif n_ly == 2:
            lying.append((0.0, 0.0, base_yaw * 0.4))
            lying.append((rp.jitter(0, 0.05), rp.jitter(0.02, 0.03), base_yaw * 0.4 + pi + radians(rp.jitter(0, 12))))
        else:
            for k in (-1, 1):
                lying.append((rp.jitter(0, 0.04), k * (W0 * 0.53 + 0.006), base_yaw * 0.4 + (pi if k > 0 else 0) + radians(rp.jitter(0, 4))))
            lying.append((rp.jitter(0, 0.05), rp.jitter(0, 0.04), base_yaw * 0.4 + radians(rp.jitter(0, 14))))
        n_stand = n_closed - n_ly
    else:
        base_yaw = radians(rp.u(-7, 7) if pal else rp.u(-22, 22))
        n_stand = 0
        n_ly = n_closed

    for i in range(n_ly):
        D, r = new_dim(i)
        S = lying_shape(D, th)
        G = F.clearance(0.008)
        roll_max, stiff = (0.07 if style == "stacked" else 0.5), 0.15 * D.L
        if style == "heap" and i > 0:
            # G = on the ground snug against the others, S = half on / half off, T = thrown on top where it
            # lies steadily, U = the one crowning the heap. Candidates are ranked on the height field, then the
            # best few are really settled and the one that lies most naturally (least bent / tilted) wins.
            kind = {2: "GS", 3: "GGT", 4: "GGST", 5: "GGGTT", 6: "GGGSTT", 7: "GGGGTTU"}[n_closed][i]
            ha, hb = (D.L + 0.6 * D.Lr) * 0.5, D.Wd * 0.5
            cen = np.mean([[s_["c"][0], s_["c"][1]] for s_ in sacks], axis=0)
            cands = []
            if pal and kind == "G":
                cen = np.zeros(2)                             # fill the pallet before anything lands beside it
            for it in range(420 if kind == "G" else 170):
                ang = rp.u(0, 2 * pi)
                rad = {"G": rp.u(0.3, 1.35), "S": rp.u(0.15, 0.95), "T": rp.u(0.0, 0.9), "U": rp.u(0.0, 0.45)}[kind] * L0
                x, y = cen[0] + rad * cos(ang), cen[1] + rad * sin(ang)
                yw = base_yaw + (rp.pick([0, 0, 1, 2, 2, 3]) * pi / 2 + radians(rp.jitter(0, 8 if pal and kind == "G" else 20)) if kind in "GS" else rp.u(0, 2 * pi))
                xo, yo = x + 0.3 * D.Lr * cos(yw), y + 0.3 * D.Lr * sin(yw)          # footprint incl. the neck
                cover, hm, hs, fhi, bstd = F.stats(xo, yo, yw, ha * 0.95, hb * 0.92, 1.25 * T0)
                d = hypot(x - cen[0], y - cen[1]) / L0
                if kind == "G":
                    touch = F.stats(xo, yo, yw, ha + 0.02, hb + 0.03)[0]
                    near = F.stats(xo, yo, yw, ha + 0.1, hb + 0.1)[0]
                    cost = 40 * touch + 2.0 * d + (1.5 if near < 0.02 else 0.0) + 12 * bstd
                elif kind == "S":
                    cost = 4 * abs(cover - 0.5) + 0.5 * d + 6 * fhi
                elif kind == "T":
                    cost = 5 * (1 - cover) + 2.0 * hs / T0 + 0.4 * d + 8 * fhi
                else:
                    cost = 5 * (1 - cover) + 2.0 * hs / T0 + 0.8 * d
                cands.append((cost, x, y, yw))
            cands.sort()
            best = None
            for cost, x, y, yw in cands[:1 if kind == "G" else 16]:
                P, q = settle(D, S, F, G, x, y, yw, roll_max, stiff)
                c2 = cost + 6 * q["bend"] + 2.0 * q["roll"] + (4 * max(0.0, q["tilt"] - 0.4) if kind == "S" else 5 * q["tilt"])
                if best is None or c2 < best[0]:
                    best = (c2, x, y, yw, P)
            _, cx, cy, yaw, P = best
        else:
            cx, cy, yaw = ((-0.2 * L0, -0.53 * W0, base_yaw) if pal else (0.0, 0.0, base_yaw)) if style == "heap" else lying[i]
            P, _q = settle(D, S, F, G, cx, cy, yaw, roll_max, stiff)
        F.splat(P[:S["info"]["outer"]])
        sacks.append({"kind": "lying", "D": D, "P": P, "info": S["info"], "c": (cx, cy), "rng": r, "i": i})

    # ---- standing sacks (closed ones first, the open one last so it ends up in front)
    def raster_standing(P, i0, i1, yaw):
        co, si = cos(yaw), sin(yaw)
        for i in range(i0, i1):
            c = P[i].mean(axis=0)
            d = P[i, :, :2] - c[:2]
            rx = float(np.abs(d[:, 0] * co + d[:, 1] * si).max())
            ry = float(np.abs(-d[:, 0] * si + d[:, 1] * co).max())
            sy, sx = F.window(c[0], c[1], max(rx, ry) + 0.03)
            X, Y = F.X[sy, sx] - c[0], F.Y[sy, sx] - c[1]
            mk = ((X * co + Y * si) / max(rx, 1e-3)) ** 2 + ((-X * si + Y * co) / max(ry, 1e-3)) ** 2 < 1.0
            Hw = F.H[sy, sx]
            Hw[mk] = np.maximum(Hw[mk], float(P[i, :, 2].max()))

    aa = np.linspace(0, 2 * pi, 28, endpoint=False)

    def foot_free(c, yaw, rx, ry):
        co, si = cos(yaw), sin(yaw)
        for f in (0.35, 0.72, 1.07):
            lx, ly = f * (rx + 0.03 * sz) * np.cos(aa), f * ry * np.sin(aa)
            if float(F.sample(F.H, c[0] + lx * co - ly * si, c[1] + lx * si + ly * co).max()) > 0.006:
                return False
        return True

    org = np.mean([s_["c"] for s_ in sacks], axis=0) if style == "heap" else np.zeros(2)   # middle of the laid sacks

    def find_spot(ang, rx, ry, extra):
        d = np.array([cos(ang), sin(ang)])
        yaw = ang + pi / 2
        r = 0.05
        while r < 2.2 and not foot_free(org + d * r, yaw, rx, ry):
            r += 0.02
        c = org + d * (r + extra)
        sy, sx = F.window(c[0], c[1], 0.75 * sz)             # lean towards the mass of the pile nearby
        X, Y = F.X[sy, sx] - c[0], F.Y[sy, sx] - c[1]
        near = np.hypot(X, Y) < 0.75 * sz
        w = np.clip(F.H[sy, sx] - 0.16 * sz, 0, None) ** 2 * near
        if w.sum() < 1e-4:
            w = F.H[sy, sx] * near
        ld = _unit([float((X * w).sum()), float((Y * w).sum())]) if w.sum() > 1e-3 else -d
        return c, ld

    def fit_lean(S, c, yaw, ld, lo, hi):
        """Largest lean that still clears everything in the height field (the sack just touches the pile)."""
        def pen(deg):
            Q = upright_pose(S, c[0], c[1], yaw, ld, radians(deg))[:S["info"]["outer"]]
            Hs = F.sample(F.H, Q[..., 0], Q[..., 1])
            return float(np.where(Hs > 0.006, Hs + 0.005 - Q[..., 2], -1.0).max())
        if pen(lo) > 0:
            return None, False
        deg = lo
        while deg + 1.0 <= hi:
            if pen(deg + 1.0) > 0:
                for _ in range(3):
                    if pen(deg + 0.25) > 0:
                        break
                    deg += 0.25
                return deg, True
            deg += 1.0
        return deg, False

    def stand(S, ang, rx, ry, extra, lo, hi, free_lean, front=0.0):
        tw = radians(rp.jitter(0, 14))
        for ex in (extra, 0.6 * extra, 0.3 * extra, 0.01):   # far enough out to lean visibly, closer if it cannot reach
            c, ld = find_spot(ang, rx, ry, ex)
            fd = _unit(-np.asarray(ld) * (1 - front) + np.array((-0.4, -1.0)) * front)   # which way the face looks
            yaw = atan2(-fd[1], -fd[0]) + pi / 2 + tw
            deg, hit = None, False
            for _try in range(14):
                deg, hit = fit_lean(S, c, yaw, ld, lo, hi)
                if deg is not None:
                    break
                c = c - ld * 0.025                            # no room even upright: step back from the pile
            if hit:
                break
        if deg is None:
            deg = lo
        elif not hit:
            deg = free_lean                                   # nothing within reach: just a tired slouch
        return c, ld, yaw, upright_pose(S, c[0], c[1], yaw, ld, radians(deg))

    stand_angs = {"leaning": [-60, -120, -18, -162, 82, 104, -90], "stacked": [(0 if style != "stacked" or free_side > 0 else 180) + rp.jitter(0, 16)], "heap": []}[style]
    for k in range(n_stand):
        D, r = new_dim(n_ly + k)
        S = standing_shape(D, th)
        ang = radians(stand_angs[k % len(stand_angs)] + rp.jitter(0, 8))
        c, ld, yaw, P = stand(S, ang, D.Rx, D.Ry, rp.u(0.06, 0.14) * sz, 2.0, 21.0, rp.u(2, 6))
        raster_standing(P, S["info"]["body"][0], S["info"]["body"][1], yaw)
        sacks.append({"kind": "standing", "D": D, "P": P, "info": S["info"], "c": (c[0], c[1]), "rng": r, "i": n_ly + k, "ld": ld})

    D, r = new_dim(n - 1)
    S = open_shape(D, th)
    # to the right of the pile and a little forward: seen from front-left or front-right it never hides the pile
    ang = radians({"leaning": -90 if n_stand != 1 else -118, "stacked": -27, "heap": -33}[style] + rp.jitter(0, 8))
    c, ld, yaw, P = stand(S, ang, 0.47 * D.Wd, 0.42 * D.Wd, rp.u(0.005, 0.03), 0.0, 6.0, rp.u(0, 2.5), front=0.85)
    raster_standing(P, S["info"]["body"][0], S["info"]["body"][1], yaw)
    open_s = {"kind": "open", "D": D, "P": P, "info": S["info"], "c": (c[0], c[1]), "rng": r, "i": n - 1, "ld": ld, "yaw": yaw}
    sacks.append(open_s)

    # ---- recentre on the footprint
    allp = np.concatenate([s["P"].reshape(-1, 3) for s in sacks], axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    if pal:
        lo = np.minimum(lo, (-pal[0] / 2, -pal[1] / 2, 0))
        hi = np.maximum(hi, (pal[0] / 2, pal[1] / 2, 0))
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, 0.0])
    for s in sacks:
        s["P"] = s["P"] + shift
        s["c"] = (s["c"][0] + shift[0], s["c"][1] + shift[1])

    def pile_h(x, y):
        return F.sample(F.H, np.asarray(x) - shift[0], np.asarray(y) - shift[1])

    # ---- meshes
    stitch_M = []
    ropes = []
    bodies = []
    necks = []          # tie centres: where hands grab
    rr = 0.0052 * max(sz, 0.85)
    sp_st = (0.026, 0.03, 0.036)[lod]
    for s in sacks:
        P, info, D, r = s["P"], s["info"], s["D"], s["rng"]
        dark = r.chance(0.3)
        o = ring_mesh(f"sack{s['i']}_{s['kind']}", P, st_cols, [m_b if dark else m_a, m_s])
        bodies.append(o)
        side_seam(stitch_M, P, info["seam0"], info["seam1"], sp_st, r)
        if s["kind"] == "lying":
            up = [j for j in range(N) if sin(th[j]) > 0.2]
            pts = 0.5 * (P[0, up] + P[1, up])
            cross = P[1, up] - P[0, up]
            add_stitches(stitch_M, pts, cross, np.tile((0.0, 0.0, 1.0), (len(up), 1)), sp_st * 0.9, 0.017, 25.0, r, lift=0.0045)
            hint = (r.u(-0.6, 0.6), r.u(-0.6, 0.6), 1.0)
        elif s["kind"] == "standing":
            ld = s["ld"]
            hint = (-ld[0] + r.u(-0.7, 0.7), -ld[1] + r.u(-0.7, 0.7), 0.0)
        if s["kind"] != "open":
            objs, C = tie_rope(f"tie{s['i']}", P, info["tie"], hint, rr, r, (20, 18, 16)[lod], (5, 5, 4)[lod])
            ropes += objs
            necks.append(C)
    for o in ropes:
        assign(o, m_rope)
    st = cylinder("stitch", radius=0.0016, depth=1.0, loc=(0, 0, 0.5), verts=3, cap=False)
    apply_transform(st)
    assign(st, m_a)
    smooth(st, 180)
    instances_to_mesh("stitches", st, stitch_M, keep_base=False)

    # ---- the open sack: heaped contents, kernels, scoop, stencil
    P, info, D, r = open_s["P"], open_s["info"], open_s["D"], open_s["rng"]
    ring_in = P[info["inner"]]
    cen = ring_in.mean(axis=0)
    z_edge = float(ring_in[:, 2].min()) - 0.004
    dome = 0.07 * D.sz
    lump3 = Lump(r, 0.12)
    pk_off = np.array([r.jitter(0, 0.02), r.jitter(0, 0.02)]) * D.sz

    def mound_z(xy, f):
        f = np.clip(f, 0, 1)
        pos = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
        return z_edge + dome * (1 - f ** 1.7) + 0.007 * lump3(pos) * (1 - f ** 2)

    fr = [1.03, 0.9, 0.75, 0.58, 0.4, 0.2]
    mv = []
    for f in fr:
        ring = cen + f * (ring_in - cen)
        ring[:, :2] += pk_off * (1 - min(f, 1.0))
        ring[:, 2] = mound_z(ring[:, :2], np.full(N, f)) if f < 1 else z_edge - 0.004
        mv.append(ring)
    top = np.array([[cen[0] + pk_off[0], cen[1] + pk_off[1], 0.0]])
    top[0, 2] = mound_z(top[:, :2], np.zeros(1))[0]
    MV = np.concatenate(mv + [top], axis=0)
    faces = []
    for a in range(len(fr) - 1):
        for j in range(N):
            j1 = (j + 1) % N
            faces.append((a * N + j, a * N + j1, (a + 1) * N + j1, (a + 1) * N + j))
    cidx = len(fr) * N
    for j in range(N):
        faces.append(((len(fr) - 1) * N + j, (len(fr) - 1) * N + (j + 1) % N, cidx))
    mound = mesh_object("contents", [tuple(v) for v in MV], faces)
    assign(mound, m_fill)
    uv_planar(mound)
    smooth(mound, 80)

    # kernels: low domes (open underneath), packed over the heap on a jittered sunflower pattern
    kl, kw, kh = k_size
    if k_round:
        kv = [(0.5 * cos(2 * pi * q / 6), 0.5 * sin(2 * pi * q / 6), 0.0) for q in range(6)] + [(0.0, 0.0, 1.0)]
        kf = [(q, (q + 1) % 6, 6) for q in range(6)]
        nk = (1000, 800, 500)[lod]
    else:
        kv = [(-0.5, 0, 0), (0, -0.5, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 1.0)]
        kf = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
        nk = (1500, 1200, 680)[lod]
    kern = mesh_object("kernel", kv, kf)
    assign(kern, m_kern)
    smooth(kern, 180)
    KM = []
    rk = r.fork("kernels")
    thx = np.concatenate([th, [2 * pi]])
    for q in range(nk):
        f = min(0.985, sqrt((q + 0.5) / nk) * 0.99 + rk.jitter(0, 0.012))
        an = (q * 2.39996 + rk.jitter(0, 0.25)) % (2 * pi)
        jf = float(np.interp(an, thx, np.arange(N + 1)))
        j = int(jf) % N
        rp_ = ring_in[j] * (1 - (jf - int(jf))) + ring_in[(j + 1) % N] * (jf - int(jf))
        pos = cen[:2] + f * (rp_[:2] - cen[:2]) + pk_off * (1 - f)
        zk = float(mound_z(pos[None, :], np.array([f]))[0])
        sc = rk.vary(1.0, 0.22)
        KM.append(trs((pos[0], pos[1], zk - kh * 0.25), (rk.jitter(0, 24), rk.jitter(0, 24), rk.u(0, 360)), (kl * sc, kw * sc, kh * sc * 1.3)))

    # spill: a thin scatter of single kernels trailing off across the paving, thickest at the foot of the sack
    if p.spill > 0.03:
        out_d = -np.asarray(open_s["ld"], dtype=float)
        sd_a = radians(r.jitter(0, 50))
        out_d = np.array([out_d[0] * cos(sd_a) - out_d[1] * sin(sd_a), out_d[0] * sin(sd_a) + out_d[1] * cos(sd_a)])
        side = np.array([-out_d[1], out_d[0]])
        oc = np.array(open_s["c"])
        r_foot = 0.46 * D.Wd
        reach = (0.07 + 0.2 * p.spill) * sz
        for q in range(int((80 + 400 * p.spill) * (1.0, 0.85, 0.6)[lod])):
            dd = -reach * np.log(max(rk.u(0, 1), 1e-4)) * 0.55
            if dd > 3.2 * reach:
                continue
            lat = rk.jitter(0, 1.0) * (0.045 * sz + 0.42 * dd)
            pos = oc + out_d * (r_foot + dd) + side * lat
            if float(pile_h(pos[0], pos[1])) > 0.004:
                continue
            heap = max(0.0, 1 - dd / (0.35 * reach)) * max(0.0, 1 - abs(lat) / (0.07 * sz))    # a few lie on each other
            sc = rk.vary(1.0, 0.22)
            KM.append(trs((pos[0], pos[1], 0.0006 + heap * rk.u(0, 1.6) * kh), (rk.jitter(0, 10 + 25 * heap), rk.jitter(0, 10 + 25 * heap), rk.u(0, 360)),
                          (kl * sc, kw * sc, kh * sc)))
    instances_to_mesh("kernels", kern, KM, keep_base=False)

    if p.scoop:
        m_tin = mat("iron_painted", TIN)                      # dull tinned sheet (bare metals render black by day)
        m_oak = mat("wood_oak", "#b98c58")
        rs = 0.043 * max(sz, 0.9)
        # worn tin scoop: thin-walled tube with a slanted mouth, soldered back plate, ferrule and turned handle
        sb = lathe("scoop_bowl", [(0.0, 0.1482), (rs - 0.0016, 0.1482), (rs - 0.0016, 0.0), (rs - 0.0003, -0.0012), (rs + 0.0012, 0.0),
                                  (rs, 0.004), (rs, 0.05), (rs, 0.1), (rs, 0.15), (0.6 * rs, 0.15), (0.02, 0.15), (0.0, 0.15)], segments=16)

        def cut(co):
            z0 = 0.05 * (1 + co.y / max(hypot(co.x, co.y), 1e-6)) if hypot(co.x, co.y) > 1e-5 else 0.05
            return Vector((co.x, co.y, z0 + (co.z / 0.15) * (0.15 - z0)))

        transform_verts(sb, cut)
        recalc_normals(sb)
        fe = lathe("scoop_ferrule", [(0.0, 0.15), (0.017, 0.15), (0.0175, 0.153), (0.016, 0.156), (0.016, 0.168), (0.0125, 0.17)], segments=10)
        hd = lathe("scoop_handle", [(0.0115, 0.168), (0.0125, 0.185), (0.0155, 0.215), (0.0165, 0.24), (0.0145, 0.262),
                                    (0.009, 0.272), (0.0, 0.274)], segments=10)
        assign([sb, fe], m_tin)
        assign(hd, m_oak)
        uv_grain(hd, axis='Z')
        ds = np.array([cos(radians(r.u(-15, 40))), sin(radians(r.u(-15, 40)))])
        cs_ = cen[:2] + pk_off - ds * 0.035
        loc = (cs_[0], cs_[1], float(mound_z((cs_ - 0 * ds)[None, :], np.array([0.25]))[0]) - 0.04)
        place([sb, fe, hd], loc=loc, rot=(-r.u(36, 48), r.jitter(0, 10), degrees(atan2(ds[1], ds[0])) - 90))
        paint([sb, fe], wear=0.0, grime=mask_mul(noise_mask(0.05, seed=p.seed + 43, lo=0.4, hi=0.8), 0.22))
        paint(hd, grime=0.25, wear=0.3)

    if p.label:
        # stencil: flat lettering wrapped on the face of the open sack just under the roll
        co, si = cos(open_s["yaw"]), sin(open_s["yaw"])
        nb = info["nb"]
        face_w = 2 * 0.47 * D.Wd
        size = min(0.085 * D.sz, 0.62 * face_w / max(len(p.label), 1) / 0.72)
        tx = text("stencil", p.label, size=size, depth=0.002, font="sans_narrow", rot=(90, 0, 0))
        realize(tx)
        me = tx.data
        # body rings as a height -> (centre, half width, face distance) lookup in the sack's own frame
        rows = list(range(3, nb + 1))
        zc = np.array([P[i, :, 2].mean() for i in rows])
        cxy = np.array([P[i, :, :2].mean(axis=0) for i in rows])
        loc_x = np.array([[(P[i, j, 0] - cxy[k, 0]) * co + (P[i, j, 1] - cxy[k, 1]) * si for j in range(N)] for k, i in enumerate(rows)])
        loc_y = np.array([[-(P[i, j, 0] - cxy[k, 0]) * si + (P[i, j, 1] - cxy[k, 1]) * co for j in range(N)] for k, i in enumerate(rows)])
        face = [j for j in range(N) if sin(th[j]) > 0.25]
        face.sort(key=lambda j: loc_x[-1, j])
        z_mid = float(zc[-1]) * 0.56
        for v in me.vertices:
            lx, lz = -v.co.x, z_mid + v.co.y - size * 0.5          # text local: x across, y up, z towards the reader
            k = int(np.clip(np.searchsorted(zc, lz) - 1, 0, len(zc) - 2))
            f = float(np.clip((lz - zc[k]) / max(zc[k + 1] - zc[k], 1e-6), 0, 1))
            xs = loc_x[k, face] * (1 - f) + loc_x[k + 1, face] * f
            ys = loc_y[k, face] * (1 - f) + loc_y[k + 1, face] * f
            ly = float(np.interp(lx, xs, ys)) + (0.0028 if v.co.z > 1e-4 else 0.0008)
            c2 = cxy[k] * (1 - f) + cxy[k + 1] * f
            v.co = Vector((c2[0] + lx * co - ly * si, c2[1] + lx * si + ly * co, lz))
        me.update()
        set_loc_rot(tx, (0, 0, 0), (0, 0, 0))
        recalc_normals(tx)
        assign(tx, mat("burlap", "#2b2c31"))
        flat(tx)

    # ---- pallet
    if pal:
        pw, pd, ph_ = pal
        m_wood = mat("wood_weathered")
        rpal = rng.fork("pallet")
        boards = []
        for k in (-1, 0, 1):
            boards.append(plank(f"bearer{k}", pd - 0.02, 0.075, 0.072, loc=(k * (pw / 2 - 0.07) + shift[0], shift[1], 0.036),
                                axis='Y', bevel=0.0, warp=0.0, rng=rpal))
        nbd = max(4, int(pd / 0.14))
        gap = (pd - nbd * 0.1) / (nbd - 1)
        nails = []
        for k in range(nbd):
            yb = -pd / 2 + 0.05 + k * (0.1 + gap)
            wd = rpal.vary(0.1, 0.06)
            boards.append(plank(f"deck{k}", pw + rpal.jitter(0, 0.015), wd, 0.024, loc=(shift[0] + rpal.jitter(0, 0.008), yb + shift[1], 0.072 + 0.0125),
                                rot=(0, 0, rpal.jitter(0, 0.7)), axis='X', bevel=0.003, warp=0.004, rng=rpal))
            for kx in (-1, 0, 1):
                for dy in (-0.022, 0.024):
                    nails.append(trs((kx * (pw / 2 - 0.07) + shift[0] + rpal.jitter(0, 0.008), yb + shift[1] + dy + rpal.jitter(0, 0.005), 0.0965)))
        assign(boards, m_wood)
        paint(boards, grime=mask_mul(noise_mask(0.3, seed=p.seed + 5, lo=0.3, hi=0.8), 0.3 + 0.5 * age), wear=mask_mul(facing((0, 0, 1)), 0.35))
        nl = lathe("nail", [(0.0048, 0.0), (0.0, 0.002)], segments=5)
        assign(nl, mat("iron_painted", TIN))
        instances_to_mesh("nails", nl, nails, keep_base=False)

    # ---- weathering: damp wicking up from the paving, handling grime at the necks and the rolled rim,
    #      blotchy dust, sun-fade on the tops (which also breaks up the woven stripes), rubbed bulges
    NK = np.array(necks) + shift if necks else np.zeros((0, 3))
    rim_c = np.array([cen[0], cen[1], float(ring_in[:, 2].mean()) + 0.03 * sz])
    rim_r = 0.47 * D.Wd
    for k, o in enumerate(bodies):
        wick = noise_mask(0.16, seed=p.seed + 17 + k, lo=0.2, hi=0.8)
        blot = noise_mask(0.3, seed=p.seed + 3 * k, lo=0.4, hi=0.95)
        fine = noise_mask(0.07, seed=p.seed + 5 * k + 1, lo=0.3, hi=0.75)
        tone = 0.05 + 0.3 * age * sacks[k]["rng"].u(0.0, 1.0)          # some sacks have simply been around longer

        def grime(co, nrm, wick=wick, blot=blot, fine=fine, tone=tone):
            flat_co = co * np.array((1.0, 1.0, 0.25))
            top = (0.03 + (0.09 + 0.26 * age) * wick(flat_co, nrm)) * sz
            damp = np.clip(1 - co[:, 2] / top, 0, 1) ** 0.8 * (0.22 + 0.5 * age)
            hand = np.zeros(len(co))
            for C in NK:
                hand = np.maximum(hand, np.exp(-(np.linalg.norm(co - C, axis=1) / (0.1 * sz)) ** 2))
            dr = np.hypot(co[:, 0] - rim_c[0], co[:, 1] - rim_c[1])
            hand = np.maximum(hand, np.exp(-((co[:, 2] - rim_c[2]) / (0.05 * sz)) ** 2) * np.exp(-((dr - rim_r) / (0.09 * sz)) ** 2))
            hand = hand * (0.15 + 0.4 * age) * (0.55 + 0.45 * fine(co, nrm))
            dirt = blot(co, nrm) * (0.08 + 0.3 * age)
            return np.clip(np.maximum(np.maximum(damp, hand), dirt) + tone, 0, 1)

        paint(o, grime=grime,
              special=mask_mul(facing((0, 0, 1)), noise_mask(0.3, seed=p.seed + 7 + k, lo=0.2, hi=0.7), 0.3 + 0.65 * age),
              wear=mask_mul(noise_mask(0.12, seed=p.seed + 11 + k, lo=0.55, hi=0.85), 0.3))
    paint(mound, grime=lambda co, nrm: np.clip((z_edge + 0.012 - co[:, 2]) / 0.02, 0, 1) * 0.5)
