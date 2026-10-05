"""Civic bronze statue on a dressed-stone pedestal. The figure (a robed, cloaked traveller holding a
lantern aloft, a star-lantern on a pilgrim staff, or reading an open book by a lantern held low) is
sculpted as one continuous casting: a signed-distance field of body volumes and heavy drapery (sharp
crested pipe folds, a trailing cloak, thick bell sleeves with rolled cuffs, rope girdle, cloth
puddled on the self-base) is meshed with narrow-band surface nets, projected onto the true surface and
decimated by curvature; head, hands and shoes are meshed finer and let into collar, cuffs and hem.
The pedestal is real masonry lofted round a chamfered plan: jointed step slabs, a moulded base
(plinth, torus, scotia), an ashlar die with V joints, and a projecting dentilled cornice with corona
and cyma. It carries a bolted bronze plaque with gilt letters or a stone tablet with cut letters.
Patina, run-off stains, moss, pigeon droppings and votive candles tell its age.
"""
from math import sin, cos, pi, radians, degrees, atan2, sqrt, tan, hypot

import numpy as np
from mathutils import Matrix, Vector, Euler
from mathutils.bvhtree import BVHTree

from tk import *

META = {"name": "Bronze statue on pedestal", "class": "structure", "mount": "ground", "focus": 0.76}
PARAMS = {
    "pose": {"default": "lantern", "choices": ["lantern", "star", "book"],
             "desc": "lantern held aloft / star-lantern on a staff / reading an open book by a lantern held low"},
    "pedestal_height": {"default": 1.8, "min": 1.1, "max": 2.8, "desc": "ground to top of the pedestal cap (m)"},
    "plan": {"default": "square", "choices": ["square", "octagon"], "desc": "plan of the pedestal: square with eased corners, or octagonal"},
    "steps": {"default": 2, "min": 1, "max": 3, "desc": "number of stone steps under the pedestal"},
    "figure_scale": {"default": 1.15, "min": 1.0, "max": 1.35, "desc": "figure size relative to life (1.78 m)"},
    "hood": {"default": "down", "choices": ["down", "up"], "desc": "hood thrown back as a cowl, or worn up"},
    "beard": {"default": True, "desc": "bearded elder (True) or clean-shaven"},
    "inscription": {"default": "HE CARRIED|THE LIGHT|HOME", "desc": "inscription, lines separated by |"},
    "plaque": {"default": "bronze", "choices": ["bronze", "carved"],
               "desc": "bolted bronze plaque with gilt raised letters, or a stone tablet with cut, metal-filled letters"},
    "verdigris": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of green patina on the bronze (1 = gone green all over)"},
    "pigeons": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "pigeon droppings on head, shoulders, lantern and cornice"},
    "stone": {"default": "#b9ad98", "desc": "pedestal stone colour"},
    "candles": {"default": 4, "min": 0, "max": 7, "desc": "votive candles left on the top step"},
    "intensity": {"default": 22.0, "min": 5.0, "max": 120.0, "desc": "lantern / star light intensity (cd)"},
}
VARIANTS = {
    "star_pilgrim": {"pose": "star", "plan": "octagon", "pedestal_height": 2.3, "steps": 3, "hood": "up", "beard": False,
                     "verdigris": 0.8, "stone": "#a3a19a", "inscription": "STELLA|DUCE|1842", "plaque": "carved",
                     "candles": 0, "pigeons": 0.8, "seed": 7},
    "scholar_low": {"pose": "book", "pedestal_height": 1.25, "steps": 1, "figure_scale": 1.05, "verdigris": 0.15,
                    "stone": "#c6b392", "inscription": "TO OUR|TEACHER", "candles": 7, "pigeons": 0.25,
                    "intensity": 16.0, "seed": 12},
    "green_colossus": {"pose": "lantern", "hood": "up", "pedestal_height": 2.7, "steps": 3, "figure_scale": 1.32,
                       "verdigris": 1.0, "stone": "#8f8a80", "inscription": "LUX|IN|TENEBRIS", "candles": 2, "pigeons": 1.0,
                       "seed": 23},
    "scholar_hooded": {"pose": "book", "plan": "octagon", "hood": "up", "pedestal_height": 2.0, "verdigris": 0.6,
                       "stone": "#b3a58c", "inscription": "LEGE|ET|LUCE", "plaque": "carved", "candles": 3,
                       "intensity": 18.0, "seed": 31},
}

ZB = 0.05          # thickness of the bronze self-base (life-size metres)


# ------------------------------------------------------------------ signed distance helpers (numpy)
def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _smax(a, b, k):
    return -_smin(-a, -b, k)


def _sph(P, c, r):
    return np.linalg.norm(P - np.asarray(c, float), axis=1) - r


def _cap(P, a, b, ra, rb=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    d = np.linalg.norm(P - a - t[:, None] * ab, axis=1)
    return d - (ra + ((ra if rb is None else rb) - ra) * t)


def _ell(P, c, r):
    r = np.asarray(r, float)
    q = (P - np.asarray(c, float)) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return np.where(k1 > 1e-3, k0 * (k0 - 1.0) / np.maximum(k1, 1e-3), -float(r.min()))


def _ellxy(x, y, a, b):
    qx = x / a
    qy = y / b
    k0 = np.sqrt(qx * qx + qy * qy)
    k1 = np.sqrt((qx / a) ** 2 + (qy / b) ** 2)
    return np.where(k1 > 1e-3, k0 * (k0 - 1.0) / np.maximum(k1, 1e-3), -np.minimum(a, b))


def _rbox(P, c, hs, r):
    q = np.abs(P - np.asarray(c, float)) - (np.asarray(hs, float) - r)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0) - r


def _foldp(x, e=1.2):
    """Drapery fold profile, 0..1: a sharp-ish crest at x = 0 (mod 2 pi), a round shadowed valley between."""
    return 1.0 - np.abs(np.sin(0.5 * x)) ** e


def _bell(P, A, B, sp, ph):
    """Hanging bell sleeve along the arm segment A-B: a thick cloth cone whose sections hang plumb
    below the arm, with tube folds running to a curved hem, a hollow mouth and a rolled cuff."""
    hv = np.array([B[0] - A[0], B[1] - A[1]])
    Lh = max(float(np.hypot(hv[0], hv[1])), 1e-3)
    hv = hv / Lh
    rx = P[:, 0] - A[0]
    ry = P[:, 1] - A[1]
    u = rx * hv[0] + ry * hv[1]
    s = -rx * hv[1] + ry * hv[0]
    t = u / Lh
    w = np.clip(t, 0.0, 1.0) ** sp["pw"]
    q = A[2] + (B[2] - A[2]) * np.clip(t, 0.0, 1.2) - P[:, 2]         # depth below the arm
    a = sp["a0"] + (sp["a1"] - sp["a0"]) * w
    b = sp["b0"] + (sp["b1"] - sp["b0"]) * w
    qq = q - (b - sp["rtop"])
    phi = np.arctan2(s, qq)
    low = _ss(-0.6, 0.5, np.cos(phi))
    fo = _foldp(sp["nf"] * phi + ph + 0.5 * np.sin(sp["nf"] * phi + ph))
    do = _ellxy(s, qq, a, b) - (0.006 + sp["fold"] * w) * (0.2 + 0.8 * low) * (fo - 0.42)
    tm = sp["t1"] - sp["curl"] * np.clip(q / (2.0 * sp["b1"] - sp["rtop"]), 0.0, 1.3) ** 2
    sm = (t - tm) * Lh                                                 # > 0 beyond the mouth
    d = _smax(do, sm, 0.02)
    cav = np.maximum(do + sp["th"], -(sm + sp["depth"]))
    d = _smax(d, -cav, 0.012)
    d = _smin(d, np.sqrt((do + 0.5 * sp["th"]) ** 2 + sm * sm) - sp["rim"], 0.012)      # rolled cuff
    return _smax(d, -(u + 0.07), 0.03)


def _hang(P, A, B, sp, ph):
    """Sleeve of a raised arm, slid back to the elbow: the doubled cloth hangs plumb below the upper
    arm A-B as a thick curtain whose hem curves down from the armpit to the mouth, breaks into leaning
    pipe folds, and opens at the elbow into a hollow bell with a rolled cuff."""
    hv = np.array([B[0] - A[0], B[1] - A[1]])
    Lh = max(float(np.hypot(hv[0], hv[1])), 1e-3)
    hv = hv / Lh
    rx = P[:, 0] - A[0]
    ry = P[:, 1] - A[1]
    u = rx * hv[0] + ry * hv[1]
    s = -rx * hv[1] + ry * hv[0]
    t = u / Lh
    q = A[2] + (B[2] - A[2]) * np.clip(t, -0.3, 1.3) - P[:, 2]         # depth below the arm axis
    w = np.clip(t, 0.0, 1.0) ** sp["pw"]
    xf = 2.0 * pi * (u - sp["lean"] * q) / sp["lam"] + ph
    fo = np.sin(xf + 0.35 * np.sin(xf))
    hem = (sp["d0"] + (sp["d1"] - sp["d0"]) * w) * (1.0 + 0.05 * fo) * (1.0 - sp["lift"] * _ss(0.55, 1.05, t) ** 2)
    gq = _ss(0.1, 0.95, q / hem)
    T = (sp["T0"] + (sp["T1"] - sp["T0"]) * w) * (0.86 + 0.26 * np.sin(pi * np.clip(q / hem, 0.0, 1.0)) + 0.1 * gq * fo)
    s0 = sp["fold"] * gq * (0.3 + 0.7 * w) * fo
    qs = np.clip(q, 0.0, np.maximum(hem - T, 0.0))
    do = np.sqrt((s - s0) ** 2 + (q - qs) ** 2) - T
    f1 = fo * np.tanh((s - s0) / 0.03)                                 # > 0 where this face of the cloth bulges
    do = do - sp["pipe"] * gq * (0.3 + 0.7 * w) * (np.clip(f1, 0.0, 1.0) ** 1.5 - 0.35)
    tm = sp["t1"] - sp["curl"] * np.clip(q / sp["d1"], 0.0, 1.2) ** 2
    sm = (t - tm) * Lh + sp["slant"] * (s - s0) * (1.0 if hv[0] < 0 else -1.0)      # > 0 beyond the mouth (cut askew, open to the front)
    R = sp["corner"]                                                   # the cuff turns round into the hem
    ca = sm + R
    cb = q - hem + R
    smr = np.sqrt(np.maximum(ca, 0.0) ** 2 + np.maximum(cb, 0.0) ** 2) + np.minimum(np.maximum(ca, cb), 0.0) - R
    smr = sm + (smr - sm) * _ss(-2.4 * R, -1.3 * R, sm)
    d = _smax(do, smr, 0.02)
    cav = np.maximum(do + sp["th"], -(smr + sp["depth"]))
    d = _smax(d, -cav, 0.012)
    d = _smin(d, np.sqrt((do + 0.5 * sp["th"]) ** 2 + smr * smr) - sp["rim"], 0.012)    # rolled cuff
    return _smax(d, -(u + 0.07), 0.03)


def _table(zs, vs, w=5):
    zt = np.linspace(-0.3, 1.9, 221)
    v = np.interp(zt, zs, vs)
    k = np.ones(w) / w
    for _ in range(3):
        v = np.convolve(np.pad(v, w // 2, mode="edge"), k, mode="valid")
    return zt, v


def _eval(field, P, chunk=300000):
    if len(P) <= chunk:
        return np.asarray(field(P), float)
    return np.concatenate([np.asarray(field(P[i:i + chunk]), float) for i in range(0, len(P), chunk)])


def _nets(field, lo, hi, h, smooth_it=1, band=None, project=1):
    """Surface-nets mesher: one vertex per sign-changing cell, one quad per sign-changing edge. With
    `band` the field is sampled on a half-resolution grid first and refined only near the surface;
    the vertices are finally pulled onto the true zero set so creases stay crisp."""
    n = [int(np.ceil((hi[i] - lo[i]) / h)) + 1 for i in range(3)]
    ax = [lo[i] + h * np.arange(n[i]) for i in range(3)]
    if band:
        nc_ = [(n[i] + 1) // 2 for i in range(3)]
        axc = [lo[i] + 2.0 * h * np.arange(nc_[i]) for i in range(3)]
        X, Y, Z = np.meshgrid(axc[0], axc[1], axc[2], indexing="ij")
        Fc = _eval(field, np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)).reshape(nc_)
        F = Fc[np.ix_(np.arange(n[0]) // 2, np.arange(n[1]) // 2, np.arange(n[2]) // 2)].copy()
        I = np.nonzero(np.abs(F) < band * h)
        F[I] = _eval(field, np.stack([ax[0][I[0]], ax[1][I[1]], ax[2][I[2]]], axis=1))
    else:
        X, Y, Z = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
        F = _eval(field, np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)).reshape(n)
    nc = (n[0] - 1, n[1] - 1, n[2] - 1)
    ncell = nc[0] * nc[1] * nc[2]
    sums = np.zeros((ncell, 3))
    cnt = np.zeros(ncell)
    quads = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        s0 = [slice(1, -1)] * 3
        s1 = [slice(1, -1)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        F0 = F[tuple(s0)]
        F1 = F[tuple(s1)]
        idx = np.nonzero((F0 < 0) != (F1 < 0))
        if len(idx[0]) == 0:
            continue
        I = [idx[0].copy(), idx[1].copy(), idx[2].copy()]
        I[b] += 1
        I[c] += 1
        f0 = F0[idx]
        f1 = F1[idx]
        t = f0 / (f0 - f1)
        pos = np.stack([ax[0][I[0]], ax[1][I[1]], ax[2][I[2]]], axis=1)
        pos[:, a] += t * h
        ids = []
        for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
            J = list(I)
            J[b] = I[b] - db
            J[c] = I[c] - dc
            fid = (J[0] * nc[1] + J[1]) * nc[2] + J[2]
            ids.append(fid)
            for k in range(3):
                sums[:, k] += np.bincount(fid, weights=pos[:, k], minlength=ncell)
            cnt += np.bincount(fid, minlength=ncell)
        q = np.stack(ids, axis=1)
        flip = f0 >= 0
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    q = np.concatenate(quads)
    act = cnt > 0
    vid = np.cumsum(act) - 1
    V = sums[act] / cnt[act][:, None]
    Fq = vid[q]
    e0 = np.concatenate([Fq[:, i] for i in range(4)])
    e1 = np.concatenate([Fq[:, (i + 1) % 4] for i in range(4)])
    src = np.concatenate([e0, e1])
    dst = np.concatenate([e1, e0])
    deg = np.maximum(np.bincount(src, minlength=len(V)), 1)[:, None]
    for _ in range(smooth_it):
        for lam in (0.5, -0.52):
            avg = np.stack([np.bincount(src, weights=V[dst, k], minlength=len(V)) for k in range(3)], axis=1) / deg
            V = V + lam * (avg - V)
    e = 0.3 * h
    for _ in range(project):
        d = _eval(field, V)
        g = np.stack([(_eval(field, V + e * np.eye(3)[k]) - _eval(field, V - e * np.eye(3)[k])) / (2 * e) for k in range(3)], axis=1)
        st = d[:, None] * g / np.maximum((g * g).sum(axis=1), 0.3)[:, None]
        L = np.linalg.norm(st, axis=1)
        V = V - st * np.minimum(1.0, 0.6 * h / np.maximum(L, 1e-9))[:, None]
    return V, Fq


# ------------------------------------------------------------------ the figure (life-size, feet at z = 0)
def _sleeve(**kw):
    sp = {"pw": 1.3, "a0": 0.084, "a1": 0.11, "b0": 0.09, "b1": 0.18, "rtop": 0.085, "nf": 7.0, "fold": 0.038,
          "t1": 1.0, "curl": 0.3, "th": 0.04, "depth": 0.13, "rim": 0.026}
    sp.update(kw)
    return sp


def _hsleeve(**kw):
    sp = {"pw": 0.8, "d0": 0.13, "d1": 0.37, "T0": 0.08, "T1": 0.118, "lam": 0.088, "lean": 0.3, "fold": 0.036, "pipe": 0.015,
          "t1": 0.92, "curl": 0.24, "th": 0.04, "depth": 0.17, "rim": 0.03, "lift": 0.3, "slant": 0.26, "corner": 0.075}
    sp.update(kw)
    return sp


def _pose(pose):
    S_r = np.array([-0.2, 0.0, 1.435])
    S_l = np.array([0.2, 0.0, 1.435])
    C = {"raise": pose != "book"}
    left_book = {"S": S_l, "E": np.array([0.30, 0.03, 1.14]), "W": np.array([0.17, -0.165, 1.085]), "kind": "fore", "cuff": 0.74,
                 "sp": _sleeve(b1=0.15, a1=0.105, curl=0.3, nf=7.0)}
    if pose == "lantern":
        C["arms"] = [{"S": S_r, "E": np.array([-0.43, -0.14, 1.58]), "W": np.array([-0.53, -0.37, 1.775]), "kind": "upper",
                      "sp": _hsleeve()}, left_book]
        C["head"] = (-20.0, 5.0)
    elif pose == "star":
        C["arms"] = [{"S": S_r, "E": np.array([-0.39, -0.03, 1.31]), "W": np.array([-0.415, -0.1675, 1.5175]), "kind": "upper",
                      "sp": _hsleeve(d1=0.27, d0=0.11, lam=0.075, curl=0.2)}, left_book]
        C["head"] = (-8.0, -3.0)
    else:
        C["arms"] = [{"S": S_r, "E": np.array([-0.32, -0.045, 1.17]), "W": np.array([-0.34, -0.325, 1.30]), "kind": "fore", "cuff": 0.5,
                      "sp": _sleeve(b1=0.165, a1=0.11, curl=0.42)},
                     {"S": S_l, "E": np.array([0.275, 0.0, 1.15]), "W": np.array([0.145, -0.215, 1.125]), "kind": "fore", "cuff": 0.76,
                      "sp": _sleeve(b1=0.15, a1=0.105, curl=0.3, nf=7.0)}]
        C["head"] = (7.0, 22.0)
    if pose == "book":
        C["tLz"] = [0.1, 1.0, 1.25, 1.4, 1.5]
        C["tL"] = [1.2, 1.08, 0.95, 0.22, 0.0]
        C["tRz"] = C["tLz"]
        C["tR"] = [-1.15, -1.1, -0.95, -0.22, 0.0]
    else:
        C["tLz"] = [0.1, 1.0, 1.25, 1.4, 1.5]
        C["tL"] = [0.85, 0.68, 0.5, 0.15, 0.0]
        C["tRz"] = [0.1, 1.0, 1.3, 1.42, 1.5]
        C["tR"] = [-2.0, -2.1, -2.2, -0.9, 0.0]
    # shoes: (x, y of the ball of the foot, splay in degrees); the right one leads
    C["shoes"] = [(-0.125, -0.232, 9.0), (0.112, -0.196, -13.0)]
    return C


def _body_field(P, C):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ph = C["ph"]
    zt = C["zt"]
    a = np.interp(z, zt, C["ra"])
    b = np.interp(z, zt, C["rb"])
    xr = x - 0.022 * np.sin((z - 0.2) * 2.4)
    de = _ellxy(xr, y, a, b)
    th = np.arctan2(xr, -y)
    rho = np.sqrt(xr * xr + y * y)
    ath = np.abs(th)

    # ---- gown: sharp-crested pipe folds deepening towards the hem, breaking where the cloth meets the base
    fl = _ss(0.26, 0.0, z)
    thw = th + 0.22 * np.sin(2 * th + ph[0]) + 0.12 * np.sin(3 * th + ph[1]) + 0.10 * np.sin(2.6 * z + ph[2]) + 0.1 * fl * np.sin(5 * th + ph[11])
    xf = C["nf"] * thw
    g = _foldp(xf + 0.5 * np.sin(xf + ph[3]))
    A = (0.010 + 0.012 * _ss(1.1, 0.98, z) + 0.036 * _ss(1.0, 0.2, z)) * (0.78 + 0.22 * np.sin(3 * th + ph[4]))
    kn = np.exp(-((x + 0.105) ** 2 + (y + 0.15) ** 2 + (z - 0.52) ** 2) / 0.03)
    A = A * (1.0 - 0.6 * kn)
    d = de - A * (g - 0.42) - 0.12 * A * np.cos(2 * xf + ph[3] + 2.0 * z)
    dgown = d
    front = _ss(0.05, -0.02, y)
    if C["raise"]:      # tension folds radiating from the raised shoulder across the chest
        rx, rz = x + 0.17, z - 1.40
        rr = np.sqrt(rx * rx + rz * rz)
        m = _ss(0.07, 0.16, rr) * (1.0 - _ss(0.32, 0.5, rr)) * _ss(1.06, 1.14, z) * front
        d = d - 0.014 * (_foldp(11 * np.arctan2(rz, rx) + ph[4], 1.0) - 0.4) * m
    else:               # V folds falling to the girdle
        m = _ss(1.1, 1.16, z) * (1.0 - _ss(1.34, 1.44, z)) * front
        d = d - 0.012 * (_foldp(52 * (np.abs(x) * 0.8 + (z - 1.1) * 0.6) + ph[4], 1.0) - 0.4) * m
    d = _smax(d, z - 1.50, 0.04)
    d = _smin(d, _cap(P, (-0.18, 0.005, 1.425), (0.18, 0.005, 1.425), 0.074), 0.05)
    d = _smin(d, _cap(P, (0, 0.02, 1.42), (0, 0.0, 1.62), 0.056, 0.052), 0.03)
    leg = np.minimum(_cap(P, (-0.09, -0.01, 0.93), (-0.105, -0.15, 0.52), 0.088, 0.068),
                     _cap(P, (-0.105, -0.15, 0.52), (-0.12, -0.06, 0.08), 0.064, 0.05))
    d = _smin(d, leg, 0.07)

    # ---- rope girdle, knot and two hanging ends with tassels
    zg = 1.085
    aw = float(np.interp(zg, zt, C["ra"])) + 0.006
    bw = float(np.interp(zg, zt, C["rb"])) + 0.006
    dew = _ellxy(xr, y, aw, bw)
    d = _smin(d, np.sqrt(dew * dew + (z - zg) ** 2) - 0.017, 0.008)
    tk_ = C["knot"]

    def on_robe(t_, z_, out=0.012):
        return np.array([(float(np.interp(z_, zt, C["ra"])) + out) * sin(t_), -(float(np.interp(z_, zt, C["rb"])) + out) * cos(t_), z_])
    d = _smin(d, _sph(P, on_robe(tk_, zg, 0.014), 0.03), 0.008)
    for k, (dt, zl) in enumerate(((-0.07, 0.56), (0.06, 0.68))):
        pts = [on_robe(tk_, zg, 0.016), on_robe(tk_ + dt, 0.9, 0.02), on_robe(tk_ + dt * 1.3, zl, 0.03)]
        for i in range(2):
            d = _smin(d, _cap(P, pts[i], pts[i + 1], 0.0135), 0.006)
        d = _smin(d, _ell(P, pts[2] - np.array([0, 0, 0.03]), (0.022, 0.022, 0.04)), 0.008)

    # ---- cloak: a heavier outer layer hanging from the shoulders, open in front, clasped at the throat,
    # trailing to the base behind where its hem breaks and spreads
    flc = _ss(0.32, 0.0, z)
    ac = np.interp(z, zt, C["ca"]) + 0.03 * flc * flc
    bc = np.interp(z, zt, C["cb"]) + 0.035 * flc * flc
    dc = _ellxy(xr, y, ac, bc)
    thc = th + 0.18 * np.sin(2 * th + ph[5]) + 0.08 * np.sin(2.2 * z + ph[6]) + 0.13 * flc * np.sin(4 * th + ph[9])
    xc = C["nfc"] * thc
    gc = _foldp(xc + 0.5 * np.sin(xc + ph[9]))
    Ac = 0.012 + 0.046 * _ss(1.42, 0.5, z)
    dc = dc - Ac * (gc - 0.42) - 0.1 * Ac * np.cos(2 * xc + ph[10] + 3.0 * z)
    tL = np.interp(z, C["tLz"], C["tL"])
    tR = np.interp(z, C["tRz"], C["tR"])
    gap = np.minimum(th - tR, tL - th) * rho
    dc = dc + 0.03 * _ss(-0.11, 0.0, gap) * _ss(1.45, 1.2, z)
    dc = _smax(dc, gap, 0.02)
    dlow = np.minimum(dgown, dc)
    hem = 0.022 + 0.12 * _ss(2.55, 1.5, ath) + 0.014 * np.cos(xc + 1.0)
    dc = _smax(dc, hem - z, 0.015)
    dc = _smax(dc, z - 1.52, 0.03)
    d = _smin(d, dc, 0.012)
    d = _smin(d, _ell(P, (0, -0.128, 1.462), (0.03, 0.016, 0.03)), 0.006)      # clasp

    # ---- cloth puddled on the base: two crumpled rolls where gown and cloak hems break
    r1 = 0.021 + 0.011 * np.sin(5 * th + ph[5]) * np.sin(3 * th + ph[6]) + 0.005 * np.sin(11 * th + ph[7])
    o1 = 0.004 + 0.012 * np.sin(4 * th + ph[8])
    roll = np.sqrt((dlow - o1) ** 2 + (z - 0.72 * r1) ** 2) - r1
    r2 = (0.013 + 0.008 * np.sin(7 * th + ph[1])) * (0.45 + 0.55 * _ss(0.9, 1.7, ath))
    o2 = o1 + r1 + 0.008 + 0.012 * np.sin(3 * th + ph[2])
    roll = _smin(roll, np.sqrt((dlow - o2) ** 2 + (z - 0.7 * r2) ** 2) - r2, 0.012)
    d = _smin(d, roll, 0.02)
    for sx, sy, _ in C["shoes"]:        # the hem rides up over each shoe
        d = _smax(d, -_ell(P, (sx, sy + 0.035, 0.0), (0.066, 0.112, 0.088)), 0.02)

    # ---- cowl round the neck (and the hood bag down the back when it is thrown off)
    ring = _ellxy(x, y - 0.025, 0.105, 0.10)
    cowl = np.sqrt(ring * ring + (z - (1.495 + 0.35 * (y - 0.025))) ** 2) - 0.04
    cowl = cowl - 0.005 * np.cos(9 * np.arctan2(x, y - 0.025) + ph[7])
    if not C["hood_up"]:
        bag = _ell(P, (0, 0.135, 1.40), (0.10, 0.055, 0.125))
        bag = bag - 0.006 * (_foldp(26 * x + 4 * np.sin(9 * z + ph[8]), 1.0) - 0.4) * _ss(1.5, 1.38, z)
        cowl = _smin(cowl, bag, 0.04)
    d = _smin(d, cowl, 0.03)

    # ---- arms: thick bell sleeves with rolled cuffs, forearm emerging from the hollow mouth
    for k, arm in enumerate(C["arms"]):
        S, E, W = arm["S"], arm["E"], arm["W"]
        f = (W - E) / np.linalg.norm(W - E)
        up = _cap(P, S, E, 0.082, 0.078)
        if arm["kind"] == "upper":      # arm raised: the sleeve has slid back and hangs from the elbow
            sl = _smin(up, _hang(P, S, E, arm["sp"], ph[8 + k]), 0.03)
            d = _smin(d, sl, 0.03)
            fa = _cap(P, E, W, 0.05, 0.037)                               # close under-sleeve
            M = E + (W - E) * 0.84
            fa = _smin(fa, _cap(P, M - 0.022 * f, M + 0.012 * f, 0.043, 0.042), 0.004)   # its turned-back cuff
            fa = fa - 0.0025 * np.cos(95 * ((P - E) @ f)) * _ss(0.8, 0.7, ((P - E) @ f) / np.linalg.norm(W - E))
            d = _smin(d, fa, 0.01)
        else:
            M = E + (W - E) * arm["cuff"]
            sl = _smin(up, _cap(P, E, M, 0.08, 0.072), 0.02)
            sl = _smin(sl, _bell(P, E, M, arm["sp"], ph[8 + k]), 0.025)
            d = _smin(d, sl, 0.03)
            d = _smin(d, _cap(P, E, W, 0.046, 0.034), 0.008)

    # ---- self-base: a low hammered bronze plinth cast in one with the figure
    bx, by = C["base_hs"]
    qx = np.abs(x + 0.02) - (bx - 0.09)
    qy = np.abs(y + 0.015) - (by - 0.09)
    d2 = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0.0) - 0.09
    d2 = np.maximum(d2, (np.abs(x + 0.02) + np.abs(y + 0.015)) * 0.7071 - C["base_diag"])
    base = _smax(d2, z - 0.004 * np.sin(11 * x + ph[2]) * np.sin(9 * y + ph[3]), 0.012)
    d = _smin(d, base, 0.018)
    return np.maximum(d, -(z + ZB))


def _shoe_field(P):
    """Soft leather shoe with a welted sole: toe towards -Y, sole on z = 0, origin under the ball of the foot."""
    Q = P * np.array([1.0 / 1.2, 1.0, 1.0])
    up = _smin(_cap(Q, (0, 0.15, 0.05), (0, 0.05, 0.05), 0.043, 0.044), _cap(Q, (0, 0.05, 0.05), (0, -0.02, 0.036), 0.044, 0.035), 0.02)
    up = _smin(up, _cap(Q, (0, -0.02, 0.036), (0.004, -0.088, 0.024), 0.035, 0.02), 0.015)
    up = up - 0.0035 * np.exp(-((Q[:, 1] + 0.012) / 0.006) ** 2) * (Q[:, 2] > 0.03)          # vamp seam
    up = np.maximum(up, 0.011 - P[:, 2])
    F = Q * np.array([1.0, 1.0, 0.0])
    fp = np.minimum(np.minimum(_cap(F, (0, 0.15, 0), (0, 0.05, 0), 0.043, 0.044), _cap(F, (0, 0.05, 0), (0, -0.02, 0), 0.044, 0.035)),
                    _cap(F, (0, -0.02, 0), (0.004, -0.088, 0), 0.035, 0.02)) - 0.0055
    sole = _smax(fp, np.abs(P[:, 2] - 0.0075) - 0.0075, 0.003)
    return np.maximum(np.minimum(up, sole), P[:, 1] - 0.075)


def _head_field(P, hood_up, beard, ph):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    axx = np.abs(x)
    Pm = np.stack([axx, y, z], axis=1)
    d = _smin(_ell(P, (0, 0.018, 0.03), (0.076, 0.094, 0.09)), _ell(P, (0, -0.02, -0.045), (0.064, 0.073, 0.09)), 0.03)
    d = _smin(d, _ell(P, (0, -0.056, -0.1), (0.036, 0.03, 0.03)), 0.025)                    # chin
    d = _smin(d, _cap(P, (-0.04, -0.084, 0.014), (0.04, -0.084, 0.014), 0.0105), 0.02)        # brow
    d = _smin(d, _ell(Pm, (0.047, -0.058, -0.036), (0.024, 0.024, 0.022)), 0.03)             # cheekbones
    d = _smax(d, -_sph(Pm, (0.032, -0.098, -0.008), 0.0165), 0.012)                          # eye sockets
    d = _smin(d, _sph(Pm, (0.032, -0.074, -0.009), 0.0135), 0.004)                           # lids
    d = _smin(d, _cap(P, (0, -0.09, 0.01), (0, -0.116, -0.04), 0.0085, 0.0125), 0.012)       # nose
    d = _smin(d, _sph(Pm, (0.0125, -0.103, -0.046), 0.0085), 0.006)
    d = _smin(d, _cap(P, (-0.021, -0.095, -0.071), (0.021, -0.095, -0.071), 0.0085), 0.01)   # lips
    d = _smax(d, -_cap(P, (-0.024, -0.105, -0.0715), (0.024, -0.105, -0.0715), 0.0032), 0.004)
    d = _smin(d, _cap(P, (0, 0.02, -0.07), (0, 0.04, -0.2), 0.054, 0.058), 0.03)             # neck
    if beard:
        bd = _ell(P, (0, -0.05, -0.115), (0.054, 0.046, 0.075))
        bd = bd - 0.003 * np.cos(250 * x + 3.0 * np.sin(55 * z + ph))
        bd = _smax(bd, z - (-0.079 + 0.85 * np.maximum(axx - 0.02, 0.0)), 0.006)
        d = _smin(d, bd, 0.008)
        d = _smin(d, _cap(Pm, (0.004, -0.108, -0.06), (0.037, -0.094, -0.083), 0.009, 0.006), 0.005)
    if hood_up:
        out = _ell(P, (0, 0.03, 0.012), (0.114, 0.13, 0.134))
        out = _smin(out, _ell(P, (0, 0.105, 0.095), (0.05, 0.06, 0.05)), 0.05)
        out = _smin(out, _ell(P, (0, 0.04, -0.17), (0.14, 0.125, 0.085)), 0.05)
        out = out - 0.0035 * np.cos(34 * z + 22 * axx + ph) * _ss(-0.02, 0.05, y)
        cav = _ell(P, (0, -0.045, -0.008), (0.088, 0.15, 0.114))
        d = np.minimum(d, _smax(out, -cav, 0.012))
    else:
        ang = np.arctan2(x, y - 0.02)
        strands = np.cos(20 * ang + 1.5 * np.sin(16 * z + ph))
        hair = _ell(P, (0, 0.024, 0.03), (0.088, 0.106, 0.1)) - 0.0035 * strands * _ss(0.11, 0.05, z)
        low = _ell(P, (0, 0.045, -0.06), (0.087, 0.076, 0.078)) - 0.004 * strands
        hair = _smin(hair, low, 0.02)
        yh = -0.05 + 0.036 * _ss(0.03, -0.03, z)
        hair = _smax(hair, -np.maximum(y - yh, z - 0.072), 0.008)
        d = _smin(d, hair, 0.006)
    return d


def _hand_field(P, fist):
    d = _rbox(P, (0, 0, 0.05), (0.041, 0.0135, 0.046), 0.012)
    d = _smin(d, _cap(P, (0, 0.002, 0.0), (0, 0.004, -0.06), 0.029, 0.033), 0.015)
    d = _smin(d, _ell(P, (0.026, -0.012, 0.035), (0.02, 0.014, 0.03)), 0.01)
    curls = (60.0, 80.0, 55.0) if fist else (10.0, 14.0, 12.0)
    xs = (-0.030, -0.010, 0.0105, 0.031)
    Ls = (0.060, 0.074, 0.082, 0.073)
    for i in range(4):
        p = np.array([xs[i], 0.0, 0.092 + (0.004 if i in (1, 2) else 0.0)])
        ang = 0.0
        for j, fr in enumerate((0.45, 0.30, 0.25)):
            ang += radians(curls[j] * (1.0 + (0.0 if fist else 0.25 * (1.5 - i))))
            q = p + np.array([0.0 if fist else 0.006 * (i - 1.5) * (j == 0), -sin(ang), cos(ang)]) * Ls[i] * fr
            d = _smin(d, _cap(P, p, q, 0.0098 - 0.0008 * j, 0.0092 - 0.0008 * j), 0.004)
            p = q
    if fist:
        tp = [(0.034, -0.006, 0.028), (0.052, -0.030, 0.060), (0.028, -0.058, 0.088)]
    else:
        tp = [(0.034, -0.004, 0.028), (0.058, -0.012, 0.062), (0.064, -0.018, 0.094)]
    d = _smin(d, _cap(P, tp[0], tp[1], 0.0135, 0.0115), 0.006)
    d = _smin(d, _cap(P, tp[1], tp[2], 0.0115, 0.0098), 0.004)
    return d


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame_rot(xa, ya, za):
    e = Matrix((tuple(xa), tuple(ya), tuple(za))).transposed().to_euler("XYZ")
    return (degrees(e.x), degrees(e.y), degrees(e.z))


def _sdf_mesh(name, V, Fq, target_tris, flip=False):
    faces = Fq[:, ::-1] if flip else Fq
    o = mesh_object(name, [tuple(v) for v in V.tolist()], [tuple(f) for f in faces.tolist()])
    tris = 2 * len(Fq)
    if tris > target_tris * 1.05:
        m = o.modifiers.new("dec", "DECIMATE")
        m.ratio = target_tris / tris
    smooth(o, 80)
    realize(o)
    _fig_uv(o)
    return o


def _fig_uv(o):
    """UVs for the casting: wrapped round the vertical axis and stretched 2.5x in height, so the
    chasing marks of the bronze run down the forms instead of tiling like orange peel."""
    me = o.data
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    nl = len(me.loops)
    li = np.empty(nl, dtype=np.int32)
    me.loops.foreach_get("vertex_index", li)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    p = co.reshape(-1, 3)[li]
    ls = np.empty(len(me.polygons), dtype=np.int32)
    lt = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    lp = np.repeat(np.arange(len(ls)), lt)
    nn = np.empty(len(me.polygons) * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", nn)
    nz = nn.reshape(-1, 3)[lp, 2]
    ang = np.arctan2(p[:, 0], -p[:, 1])
    pmax = np.full(len(ls), -1e9, dtype=np.float32)
    np.maximum.at(pmax, lp, ang)
    ang = np.where(pmax[lp] - ang > pi, ang + 2 * pi, ang)
    flat_ = np.abs(nz) > 0.8
    u = np.where(flat_, p[:, 0], ang * 0.3)
    v = np.where(flat_, p[:, 1], p[:, 2] * 0.6)
    me.uv_layers[0].data.foreach_set("uv", np.stack([u, v], axis=1).astype(np.float32).ravel())
    uv_keep(o)


# ------------------------------------------------------------------ lit accessories (world units)
def _lantern(x0, y0, z_top, u, bronze, pane_m, flame_m, intensity, ring_dir=0.0, throw=0.3):
    """Hexagonal carrying lantern hanging from its ring; z_top = top of the vent cap."""
    K = 1.0 / cos(pi / 6)
    r = 0.088 * u
    hb = 0.22 * u
    zb = z_top - 0.131 * u - hb
    ze = zb + hb + 0.016 * u
    zt = ze + 0.115 * u
    out = []

    def hexl(name, prof):
        return lathe(name, [(pr * K, pz) for pr, pz in prof], segments=6, loc=(x0, y0, 0), smooth_angle=35)
    out.append(hexl("lan_base", [(0.0, zb - 0.028 * u), (r * 0.55, zb - 0.026 * u), (r + 0.004 * u, zb - 0.016 * u), (r + 0.012 * u, zb - 0.01 * u),
                                 (r + 0.012 * u, zb), (r - 0.006 * u, zb + 0.004 * u), (r - 0.016 * u, zb + 0.004 * u)]))
    out.append(hexl("lan_rail", [(r - 0.014 * u, ze - 0.02 * u), (r + 0.005 * u, ze - 0.018 * u), (r + 0.007 * u, ze - 0.002 * u), (r - 0.004 * u, ze)]))
    out.append(hexl("lan_roof", [(r - 0.01 * u, ze - 0.002 * u), (r + 0.024 * u, ze), (r + 0.026 * u, ze + 0.006 * u), (r + 0.012 * u, ze + 0.012 * u),
                                 (r * 0.6, ze + 0.04 * u), (r * 0.33, ze + 0.066 * u), (0.024 * u, ze + 0.086 * u), (0.0, ze + 0.086 * u)]))
    out.append(lathe("lan_vent", [(0.016 * u, ze + 0.08 * u), (0.022 * u, ze + 0.088 * u), (0.032 * u, ze + 0.094 * u), (0.03 * u, ze + 0.1 * u),
                                  (0.016 * u, ze + 0.109 * u), (0.007 * u, zt), (0.0, zt + 0.003 * u)], segments=12, loc=(x0, y0, 0)))
    post = box("lan_post", (0.012 * u, 0.012 * u, hb + 0.004 * u), bevel=0.0015 * u)
    bar = box("lan_bar", (0.005 * u, 0.005 * u, 2 * r * tan(pi / 6)), bevel=0.0)
    vbar = box("lan_vbar", (0.005 * u, 0.005 * u, hb), bevel=0.0)
    pm, bm, vm = [], [], []
    for k in range(6):
        a = radians(60 * k)
        pm.append(trs((x0 + (r * K - 0.004 * u) * cos(a), y0 + (r * K - 0.004 * u) * sin(a), zb + hb / 2), (0, 0, 60 * k + 30)))
        af = radians(60 * k + 30)
        bm.append(trs((x0 + r * cos(af), y0 + r * sin(af), zb + hb * 0.62), (90, 0, 60 * k + 30)))
        vm.append(trs((x0 + r * cos(af), y0 + r * sin(af), zb + hb / 2), (0, 0, 60 * k + 30)))
    out.append(instances_to_mesh("lan_posts", post, pm, keep_base=False))
    out.append(instances_to_mesh("lan_bars", bar, bm, keep_base=False))
    out.append(instances_to_mesh("lan_vbars", vbar, vm, keep_base=False))
    out.append(lathe("lan_drop", [(0.0, zb - 0.072 * u), (0.008 * u, zb - 0.067 * u), (0.017 * u, zb - 0.053 * u), (0.008 * u, zb - 0.041 * u),
                                  (0.013 * u, zb - 0.03 * u), (0.02 * u, zb - 0.024 * u)], segments=10, loc=(x0, y0, 0)))
    ring = torus("lan_ring", 0.03 * u, 0.0052 * u, loc=(x0, y0, zt + 0.03 * u), rot=(90, 0, ring_dir), major_seg=18, minor_seg=6)
    out.append(ring)
    for o in out:
        assign(o, bronze)
    # horn panes: translucent amber sheets that glow evenly, hottest where the flame stands behind them
    # CLEARANCE RULE: bars (inner faces 2.5 mm inside r) / flame tongues 5 mm behind those / panes 5 mm behind the tongues
    # (the tongues lay 1.5 mm in front of the panes, in the plane of the bars' inner faces); base and rail lips follow
    r_pane, r_flame = r - 0.0125 * u, r - 0.0075 * u
    pw_ = 2 * r_pane * tan(pi / 6)
    nx_, nz_ = 5, 9
    vs, fc = [], []
    for k in range(6):
        af = radians(60 * k + 30)
        cx, cy = x0 + r_pane * cos(af), y0 + r_pane * sin(af)
        tx, ty = -sin(af), cos(af)
        b0 = len(vs)
        for j in range(nz_):
            for i in range(nx_):
                e = (i / (nx_ - 1) - 0.5) * pw_
                vs.append((cx + tx * e, cy + ty * e, zb + hb * j / (nz_ - 1)))
        for j in range(nz_ - 1):
            for i in range(nx_ - 1):
                q = b0 + j * nx_ + i
                fc.append((q, q + 1, q + nx_ + 1, q + nx_))
    gl = mesh_object("lan_glass", vs, fc)
    assign(gl, pane_m)
    # the flame itself, seen as a bright tongue through each pane
    prof = [(0.0, 0.0), (0.011, 0.012), (0.014, 0.03), (0.009, 0.052), (0.003, 0.07), (0.0, 0.076)]
    vs, fc = [], []
    for k in range(6):
        af = radians(60 * k + 30)
        cx, cy = x0 + r_flame * cos(af), y0 + r_flame * sin(af)
        tx, ty = -sin(af), cos(af)
        b0 = len(vs)
        for e, hz in prof + [(-e_, h_) for e_, h_ in prof[-2:0:-1]]:
            vs.append((cx + tx * e * u, cy + ty * e * u, zb + 0.2 * hb + hz * u))
        fc.append(tuple(range(b0, len(vs))))
    fl = mesh_object("lan_flame", vs, fc)
    assign(fl, flame_m)
    # the light socket stands off the lantern, away from the figure, so the bronze beside it is not burnt out
    dn = max(hypot(x0, y0), 1e-3)
    light("lantern", (x0 + x0 / dn * throw * u, y0 + y0 / dn * throw * u, zb + 0.06 * u), intensity=intensity, range=10.0, flicker=0.14,
          shadow_near=(throw + 0.16) * u)
    return out, (x0, y0, ze + 0.1 * u, zt + 0.03 * u)


def _star(x0, y0, zc, u, bronze, gilt, intensity):
    """Five-pointed faceted star of amber glass leaded into bronze cames (lit from within), on the staff head."""
    R, r, t = 0.25 * u, 0.1 * u, 0.062 * u
    rim = []
    for k in range(10):
        a = radians(90 + 36 * k)
        rr = R if k % 2 == 0 else r
        rim.append((x0 + rr * cos(a), y0, zc + rr * sin(a)))
    verts = rim + [(x0, y0 - t, zc), (x0, y0 + t, zc)]
    for par, tint, glow in ((0, "#ffa040", 0.9), (1, "#e87a24", 0.5)):
        faces = []
        for k in range(10):
            k1 = (k + 1) % 10
            if k % 2 == par:
                faces.append((10, k1, k))
            else:
                faces.append((11, k, k1))
        st = mesh_object(f"star_glass{par}", verts, faces)
        recalc_normals(st)
        flat(st)
        assign(st, mat("emissive", tint, glow=glow))
    bars = [sweep("star_rim", rim, 0.0075 * u, closed=True, sides=6, smooth_path=False)]
    for k in range(10):
        for s in (-1, 1):
            bars.append(sweep("star_rib", [(x0, y0 + s * (t + 0.002 * u), zc), rim[k]], 0.0048 * u, sides=5, smooth_path=False))
    sf = join(bars, "star_frame")
    assign(sf, bronze)
    orn = [sphere("star_boss", 0.016 * u, loc=(x0, y0 - t, zc), segments=10, rings=6),
           sphere("star_boss", 0.016 * u, loc=(x0, y0 + t, zc), segments=10, rings=6)]
    tips = sphere("star_tip", 0.011 * u, segments=8, rings=5)
    orn.append(instances_to_mesh("star_tips", tips, [trs(rim[k]) for k in range(0, 10, 2)], keep_base=False))
    so = join(orn, "star_orn")
    assign(so, gilt)
    paint(so, wear=0.3, special=0.15)
    light("star", (x0 * 1.25, y0 - t - 0.3 * u, zc), color="#ffc06a", intensity=intensity, range=11.0, flicker=0.08, shadow_near=0.5 * u)
    return sf


# ------------------------------------------------------------------ masonry lofted round a chamfered plan
def _plan(R, c):
    """Square of half-width R with corners cut back by c: 8 corners, counter-clockwise from the front-left."""
    return [(-R + c, -R), (R - c, -R), (R, -R + c), (R, R - c), (R - c, R), (-R + c, R), (-R, R - c), (-R, -R + c)]


def _fracs(L, step):
    n = max(1, int(round(L / step)))
    return [i / n for i in range(1, n)]


def _loft(name, R, c, prof, z0, step, grooves=(), bv=0.006, sm=28.0):
    """Skin a moulding profile [(offset from the die face, z), ...] round the plan. `step` = spacing of the
    vertex columns (for weathering masks); grooves = [(edge, fraction, z_lo, z_hi)] V-cut perpend joints;
    every arris gets a small eased face instead of a razor edge."""
    P0 = _plan(R, c)
    fr = []
    for e in range(8):
        L0 = float(np.hypot(P0[(e + 1) % 8][0] - P0[e][0], P0[(e + 1) % 8][1] - P0[e][1]))
        gs = [g[1] for g in grooves if g[0] == e]
        fr.append([f for f in _fracs(L0, step) if all(abs(f - gf) > 0.022 / L0 for gf in gs)])
    verts, faces = [], []
    n = 0
    for o, z in prof:
        Pn = _plan(R + o, max(c + 0.586 * o, 3.0 * bv))
        ring = []
        for e in range(8):
            ax_, ay_ = Pn[e]
            bx_, by_ = Pn[(e + 1) % 8]
            L = float(np.hypot(bx_ - ax_, by_ - ay_))
            tx, ty = (bx_ - ax_) / L, (by_ - ay_) / L
            cs = [(bv / L, 0.0)] + [(f, 0.0) for f in fr[e]] + [(1.0 - bv / L, 0.0)]
            for ge, gf, g0, g1 in grooves:
                if ge == e:
                    push = 0.006 if g0 + 1e-4 < z < g1 - 1e-4 else 0.0
                    cs += [(gf - 0.008 / L, 0.0), (gf, push), (gf + 0.008 / L, 0.0)]
            cs.sort()
            for f, push in cs:
                ring.append((ax_ + tx * L * f - ty * push, ay_ + ty * L * f + tx * push, z0 + z))
        n = len(ring)
        verts += ring
    for i in range(len(prof) - 1):
        for j in range(n):
            a = i * n + j
            b = i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    o = mesh_object(name, verts, faces)
    smooth(o, sm)
    return o


# ------------------------------------------------------------------ weathering
def _patina_fns(Ff, to_fig, vg, seed, rubs=(), soot=None):
    """Mask functions for the cast figure, driven by its own geometry (Ff = signed distance in figure
    space): verdigris on rain-washed up-facing forms, pooled in the fold valleys, running down below
    every overhanging lip; sheltered hollows stay dark, fold crests and handled spots are rubbed bright."""
    e = 0.028
    ns_f = noise_mask(0.065, seed=seed + 11, lo=0.3, hi=0.7)
    nl_f = noise_mask(0.3, seed=seed + 12, lo=0.3, hi=0.7)
    nm_f = noise_mask(0.1, seed=seed + 13, lo=0.3, hi=0.7)
    upv = np.array([0.0, 0.0, 1.0])
    sq = np.array([1.0, 1.0, 0.07])
    g_ = vg ** 0.7
    memo = {}

    def terms(co, n):
        key = (len(co), round(float(co[:, 0].sum()), 4), round(float(co[:, 2].sum()), 4))
        if key in memo:
            return memo[key]
        p = to_fig(co)
        cav = np.clip((e - Ff(p + n * e)) / e * 1.3, 0.0, 1.0)
        crest = np.clip((e + Ff(p - n * e)) / e * 1.6, 0.0, 1.0)
        run = np.zeros(len(p))
        for h in (0.035, 0.07, 0.12, 0.18, 0.26, 0.35, 0.45):
            run = np.maximum(run, np.clip((0.012 - Ff(p + n * 0.016 + upv * h)) / 0.012, 0.0, 1.0) * (1.0 - h / 0.52))
        sh = np.zeros(len(p))
        for h in (0.09, 0.2, 0.34):
            sh = np.maximum(sh, np.clip(-Ff(p + n * 0.075 + upv * h) / 0.01, 0.0, 1.0))
        nz = n[:, 2]
        ns = ns_f(co * sq, n)
        nl = nl_f(co, n)
        nm = nm_f(co, n)
        up_exp = np.clip(nz, 0.0, 1.0) ** 1.4 * (1.0 - 0.85 * sh)
        v = g_ * (1.3 * up_exp + 0.85 * cav * (0.45 + 0.55 * nl) * (1.0 - 0.5 * sh)
                  + 1.25 * run * (0.25 + 0.75 * ns) * (1.0 - 0.5 * crest) + 0.4 * nl * ns * (1.0 - sh)
                  + 0.3 * np.clip(n[:, 1], 0.0, 1.0) * (0.4 + 0.6 * ns))            # the damp north side turns first
        v = v * (1.0 - 0.6 * (p[:, 2] < 0.012) * (1.0 - nl) - 0.25 * (p[:, 2] < 0.012))      # trodden, dirty self-base
        if vg > 0.55:       # the whole casting turning; only rubbed high points and deep shelter stay brown
            v = v + (vg - 0.55) / 0.45 * (0.8 - 0.75 * crest * (0.35 + 0.65 * nm) - 0.3 * sh * nl - 0.25 * (1.0 - ns) * (1.0 - nl))
        sp = _ss(0.27, 0.74, v)
        edge = np.exp(-((v - 0.45) / 0.14) ** 2)
        gr = sp * (0.46 + 0.24 * cav + 0.14 * (1.0 - nl)) + 0.4 * edge + (1.0 - sp) * (0.3 + 0.3 * nl) + 0.3 * sh + 0.25 * np.clip(-nz, 0.0, 1.0)
        st = np.zeros(len(co))
        if soot is not None:                # the bronze nearest the light is kept waxed and dark: bare metal, no bloom
            dl = np.linalg.norm(co - np.asarray(soot[0], float), axis=1) / soot[1]
            st = np.clip(1.6 * np.exp(-dl ** 2), 0.0, 1.0)
            gr = gr * (1.0 - 0.9 * np.exp(-(dl / 1.7) ** 2))
            sp = sp * (1.0 - st)
        gr = np.clip(gr, 0.0, 1.0)
        we = np.clip(2.2 * (crest - 0.62), 0.0, 1.0) * (0.2 + 0.8 * nm) * (1.0 - 0.75 * sp) * 0.3
        # the brown of the patina itself: handled, rain-washed convex forms are a warmer, lighter brown than the hollows
        we = we + (0.07 + 0.16 * nl + 0.1 * crest * nm) * (1.0 + 0.7 * np.clip(n[:, 1], 0.0, 1.0)) * (1.0 - 0.8 * cav) * (1.0 - 0.6 * sh) * (1.0 - sp)
        for c, rr, s in rubs:
            we = np.maximum(we, s * np.exp(-(np.linalg.norm(co - np.asarray(c, float), axis=1) / rr) ** 2))
        we = we * (1.0 - 0.7 * st)
        sp = sp * (1.0 - 0.8 * np.clip(we * 1.6, 0.0, 1.0))
        memo[key] = (0.9 * sp, gr, we)
        return memo[key]

    return (lambda co, n: terms(co, n)[0]), (lambda co, n: terms(co, n)[1]), (lambda co, n: terms(co, n)[2])


def _guano(name, objs, spots, rng, material):
    """Pigeon droppings: small domed splats, some with a run, traced over the real surfaces below `spots`
    [(x, y, radius, z_from, count)] by ray casting and surface walking."""
    vs, ps = [], []
    for o in objs:
        realize(o)
        mw = world_matrix(o)
        base = len(vs)
        vs += [mw @ v.co for v in o.data.vertices]
        ps += [tuple(base + i for i in pl.vertices) for pl in o.data.polygons]
    if not ps:
        return None
    bvh = BVHTree.FromPolygons(vs, ps)
    V, F = [], []
    down = Vector((0.0, 0.0, -1.0))

    def tri(a, b, c, nrm):
        if (V[b] - V[a]).cross(V[c] - V[a]).dot(nrm) < 0:
            b, c = c, b
        F.append((a, b, c))

    for cx, cy, rad, z_from, count in spots:
        for _ in range(int(count)):
            a = rng.u(0, 2 * pi)
            rr = rad * sqrt(rng.u())
            hit = bvh.ray_cast(Vector((cx + rr * cos(a), cy + rr * sin(a), z_from)), down, 3.0)
            if hit[0] is None or hit[1].z < 0.12:
                continue
            loc, nor = hit[0], hit[1].normalized()
            t1 = nor.cross(Vector((0.3, 1.0, 0.2))).normalized()
            t2 = nor.cross(t1)
            rs_ = rng.u(0.009, 0.024)
            c0 = len(V)
            V.append(loc + nor * 0.0045)
            m = 7
            a0 = rng.u(0, 2 * pi)
            for i in range(m):
                aa = a0 + 2 * pi * i / m
                q = loc + (t1 * cos(aa) + t2 * sin(aa)) * rs_ * rng.u(0.55, 1.3)
                nq = bvh.find_nearest(q, 0.05)
                V.append((nq[0] + nq[1] * 0.0016) if nq[0] is not None else q)
            for i in range(m):
                tri(c0, c0 + 1 + i, c0 + 1 + (i + 1) % m, nor)
            # the run: follow the surface downhill while it is steep enough
            pos, nrm = loc.copy(), nor
            w = rs_ * rng.u(0.3, 0.5)
            prev = None
            for stp in range(rng.pick([0, 3, 5, 7, 10])):
                g = down - nrm * down.dot(nrm)
                if g.length < 0.3:
                    break
                g.normalize()
                nq = bvh.find_nearest(pos + g * 0.014, 0.04)
                if nq[0] is None or nq[1].z < -0.3:
                    break
                pos, nrm = nq[0], nq[1].normalized()
                side = nrm.cross(g).normalized() * w
                i0 = len(V)
                V.append(pos - side + nrm * 0.002)
                V.append(pos + side + nrm * 0.002)
                if prev is None:
                    tri(c0, i0, i0 + 1, nrm)
                else:
                    tri(prev, i0, i0 + 1, nrm)
                    tri(prev, i0 + 1, prev + 1, nrm)
                prev = i0
                w *= 0.86
    if not F:
        return None
    o = mesh_object(name, [tuple(v) for v in V], F)
    smooth(o, 60)
    assign(o, material)
    return o


def _tris(objs):
    t = 0
    for o in objs:
        if o.type == "MESH":
            t += sum(len(pl.vertices) - 2 for pl in o.data.polygons)
    return t


def _letters(lines, size, y, zc, cap, depth, name):
    """Lines of roman capitals centred on x = 0 around height zc (each line re-centred by its real bounds)."""
    out = []
    n = len(lines)
    for i, ln in enumerate(lines):
        zt = zc + ((n - 1) / 2 - i) * cap * 1.7
        t = text(f"{name}{i}", ln, size=size, depth=depth, loc=(0, y, zt), font="roman")
        m = t.modifiers.new("dis", "DECIMATE")
        m.decimate_type = "DISSOLVE"
        m.angle_limit = radians(7.0)
        realize(t)
        mn, mx = world_bbox([t])
        t.location.x += -(mn.x + mx.x) / 2
        t.location.z += zt - (mn.z + mx.z) / 2
        out.append(t)
    return join(out, name) if len(out) > 1 else out[0]


# ------------------------------------------------------------------ build
def build(p, rng):
    import bpy
    import bmesh
    fs = float(p.figure_scale)
    H = float(p.pedestal_height)
    seed = int(p.seed)
    vg = float(p.verdigris)
    kk = fs / 1.15
    bronze = mat("bronze_aged", rough=1.45)
    gilt = mat("brass", rough=0.85)
    stone = mat("stone_smooth", p.stone)
    stone_step = mat("stone_rough", p.stone, rough=0.95)
    flame_m = mat("emissive", "#ffbe70", glow=5)
    pane_m = mat("emissive", "#ff9230", glow=0.75)
    rs = rng.fork("stone")
    octa = p.plan == "octagon"
    hw = 0.42 * fs                              # half-width of the die
    c0 = 0.45 * hw if octa else 0.07 * kk       # corner cut of the plan
    m = clamp(H / 1.8, 0.8, 1.2)                # scale of the moulding heights
    base_h, cap_h = 0.329 * m, 0.36 * m
    nst = int(p.steps)
    rise = 0.165
    while nst > 1 and H - nst * rise - base_h - cap_h < 0.34:
        nst -= 1
    z_s = nst * rise
    z_d0 = z_s + base_h
    z_c0 = H - cap_h
    die_h = z_c0 - z_d0
    op = 0.16 * kk                              # projection of the plinth beyond the die face
    stones = []

    # ---- steps: individually dressed slabs laid as a ring (hidden core left out), settled and out of true
    mossm = []
    rmo = rng.fork("moss")
    for k in range(nst):
        hs = hw + op + 0.2 + 0.27 * (nst - 1 - k)
        inner = (hs - 0.27 - 0.06) if k < nst - 1 else hw + op - 0.05
        n = max(3, int(round(2 * hs / 0.62)))
        cuts = [-hs] + [(-hs + 2 * hs * (i / n)) + rs.jitter(0, 0.06) for i in range(1, n)] + [hs]
        for i in range(n):
            for j in range(n):
                x0, x1, y0, y1 = cuts[i], cuts[i + 1], cuts[j], cuts[j + 1]
                if x0 > -inner and x1 < inner and y0 > -inner and y1 < inner:
                    continue
                sink = rs.pick([0.0, 0.0, 0.0, 0.004, 0.007]) if k > 0 else 0.0
                dz = rs.jitter(0, 0.003)
                tl = 0.14 if k == 0 else 0.4
                b = stone_block(f"step{k}_{i}_{j}", (x1 - x0 - rs.u(0.004, 0.008), y1 - y0 - rs.u(0.004, 0.008), rise + dz),
                                loc=((x0 + x1) / 2, (y0 + y1) / 2, k * rise + (rise + dz) / 2 - sink + (0.002 if k == 0 else 0.0)),
                                rot=(rs.jitter(0, tl), rs.jitter(0, tl), rs.jitter(0, 0.3)), rng=rs, chamfer=0.014, rough=0.005, detail=0.3)
                assign(b, stone_step)
                stones.append(b)
        # moss cushions in the angle at the foot of each riser, thickest on the damp north side
        for sx, sy, dens in ((0, 1, 3.2), (1, 0, 1.3), (-1, 0, 1.6), (0, -1, 0.5)):
            for _ in range(int(2 * hs * dens)):
                t = rmo.u(-hs, hs)
                px, py = (t, sy * (hs + rmo.u(0.0, 0.02))) if sx == 0 else (sx * (hs + rmo.u(0.0, 0.02)), t)
                s = rmo.u(0.018, 0.05)
                mossm.append(trs((px, py, k * rise + s * 0.1), (0, 0, rmo.u(0, 360)), (s * rmo.u(0.9, 1.9), s, s * rmo.u(0.3, 0.5))))
    ms_ = sphere("moss_base", 1.0, segments=5, rings=3)
    moss = instances_to_mesh("moss", ms_, mossm, keep_base=False)
    transform_verts(moss, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
    smooth(moss, 80)
    assign(moss, stone_step)
    paint(moss, special=1.0, grime=0.35, mode="set")

    def arc(fn, n=6):
        return [fn(radians(90.0 * i / n)) for i in range(1, n + 1)]

    card = (0, 2, 4, 6)
    # ---- moulded base: plinth blocks, torus, fillet, scotia, astragal
    ot = 0.10 * kk
    prof = [(op - 0.008, 0.0), (op, 0.008), (op, 0.135 * m), (op - 0.01, 0.145 * m), (ot, 0.145 * m)]
    prof += [(ot + 0.036 * m * cos(radians(a)), 0.181 * m + 0.036 * m * sin(radians(a))) for a in range(-72, 91, 18)]
    prof += [(ot - 0.004, 0.217 * m), (ot - 0.004, 0.229 * m)]
    r0, r1 = ot - 0.004, 0.022
    prof += arc(lambda t: (r1 + (r0 - r1) * (1 - sin(t)), 0.229 * m + 0.07 * m * (1 - cos(t))), 5)
    prof += [(0.022, 0.311 * m), (0.008, 0.319 * m), (0.0, base_h)]
    gro = [(e, rs.u(0.3, 0.7), -1.0, 0.14 * m) for e in card]
    base = _loft("ped_base", hw, c0, prof, z_s, 0.17, gro)

    # ---- die: ashlar courses of uneven height with V joints and staggered perpends
    ncr = max(2, int(round(die_h / 0.33)))
    chs = [1.0 + rs.jitter(0, 0.13) for _ in range(ncr)]
    zc_ = [0.0]
    for h_ in chs:
        zc_.append(zc_[-1] + h_ * die_h / sum(chs))
    prof = [(0.0, 0.0)]
    gro = []
    fpar = {(e, q): 0.5 + (0.17 if (q + e // 2) % 2 else -0.17) + rs.jitter(0, 0.05) for e in card for q in (0, 1)}
    for c in range(ncr):
        a, b = zc_[c] + (0.006 if c else 0.0), zc_[c + 1] - (0.006 if c < ncr - 1 else 0.0)
        nn = max(1, int(round((b - a) / 0.1)))
        if c:
            prof.append((0.0, a))
        prof += [(0.0, a + (b - a) * i / nn) for i in range(1, nn + 1)]
        if c < ncr - 1:
            prof.append((-0.006, zc_[c + 1]))
        for e in card:
            gro.append((e, fpar[(e, c % 2)], zc_[c] if c else -1.0, zc_[c + 1] if c < ncr - 1 else 99.0))
    die = _loft("ped_die", hw, c0, prof, z_d0, 0.085, gro)

    # ---- cornice: astragal, cavetto, dentil course, ovolo, corona with drip, cyma, weathered top, blocking
    od, od2, o3, oc = 0.040 * kk, 0.082 * kk, 0.126 * kk, 0.152 * kk
    prof = [(0.0, 0.0), (0.012, 0.004 * m), (0.02, 0.014 * m), (0.016, 0.026 * m), (0.006, 0.032 * m)]
    prof += arc(lambda t: (0.006 + (od - 0.006) * (1 - cos(t)), 0.032 * m + 0.05 * m * sin(t)), 4)
    prof += [(od, 0.138 * m), (od2, 0.138 * m), (od2, 0.150 * m)]
    prof += arc(lambda t: (od2 + (o3 - od2) * sin(t), 0.150 * m + 0.046 * m * (1 - cos(t))), 5)
    prof += [(0.140 * kk, 0.196 * m), (0.143 * kk, 0.188 * m), (oc - 0.003, 0.188 * m), (oc, 0.191 * m), (oc, 0.262 * m), (0.158 * kk, 0.266 * m), (0.158 * kk, 0.276 * m)]
    prof += [(0.158 * kk + 0.036 * kk * (0.5 - 0.5 * cos(pi * s / 5)), 0.276 * m + 0.046 * m * s / 5) for s in range(1, 6)]
    prof += [(0.198 * kk, 0.324 * m), (0.198 * kk, 0.334 * m), (0.193 * kk, 0.339 * m), (0.112 * kk, 0.351 * m), (0.104 * kk, cap_h - 0.005), (0.10 * kk, cap_h),
             (0.02, cap_h)]
    gro = [(e, 0.5 + rs.jitter(0, 0.12), -1.0, 99.0) for e in (2, 6)]
    capo = _loft("ped_cap", hw, c0, prof, z_c0, 0.12, gro)
    dn = box("dentil", (0.045 * kk, 0.035 * kk + 0.004, 0.046 * m), bevel=0.003)
    Pd = _plan(hw + od, c0 + 0.586 * od)
    dm = []
    for e in range(8):
        ax_, ay_ = Pd[e]
        bx_, by_ = Pd[(e + 1) % 8]
        L = float(np.hypot(bx_ - ax_, by_ - ay_))
        tx, ty = (bx_ - ax_) / L, (by_ - ay_) / L
        nd = int((L + 0.03 * kk) / (0.075 * kk))
        for i in range(nd):
            s = L / 2 + (i - (nd - 1) / 2) * 0.075 * kk
            dm.append(trs((ax_ + tx * s + ty * 0.0155 * kk, ay_ + ty * s - tx * 0.0155 * kk, z_c0 + 0.115 * m), (0, 0, degrees(atan2(ty, tx)))))
    dent = instances_to_mesh("ped_dentils", dn, dm, keep_base=False)
    stones += [base, die, capo, dent]
    for i, o in enumerate((base, die, capo)):
        displace_noise(o, 0.0022, 0.35, seed=seed + 40 + i)
    for o in (base, die, capo, dent):
        assign(o, stone)

    # ---- inscription: sized to the face (cap height >= 70 mm wherever the wording allows)
    lines = [s.strip() for s in str(p.inscription).split("|") if s.strip()]
    carved = p.plaque == "carved"
    face_w = 2 * (hw - c0)
    pw = face_w - 0.05 if octa else min(0.72 * 2 * hw, face_w - 0.05)
    mg = 0.045 if carved else 0.07
    zc = z_d0 + die_h * 0.52
    plq = []
    if lines:
        s0 = 0.1
        pr = text("probe", "H", size=s0, depth=0.004, font="roman")
        mn, mx = world_bbox([pr])
        capr = max((mx.z - mn.z) / s0, 0.2)
        delete([pr])

        def widest(ls):
            ws = []
            for ln in ls:
                t = text("probe", ln, size=s0, depth=0.004, font="roman")
                mn, mx = world_bbox([t])
                ws.append((mx.x - mn.x) / s0)
                delete([t])
            return ws
        ws = widest(lines)
        hmax = die_h - 0.12 - 2 * mg
        while True:
            cap_w = (pw - 2 * mg) / max(ws) * capr
            if cap_w < 0.07 and pw < face_w - 0.06:
                pw = min(face_w - 0.06, 0.07 * max(ws) / capr + 2 * mg)
                continue
            i = int(np.argmax(ws))
            if cap_w >= 0.07 or len(lines) >= 5 or " " not in lines[i] or (len(lines) + 1) * 1.7 * 0.07 > hmax:
                break
            wd = lines[i].split(" ")
            j = min(range(1, len(wd)), key=lambda q: abs(len(" ".join(wd[:q])) - len(" ".join(wd[q:]))))
            lines[i:i + 1] = [" ".join(wd[:j]), " ".join(wd[j:])]
            ws = widest(lines)
        if cap_w < 0.07:        # last resort on a narrow face: a tighter margin around the lettering
            mg = max(0.03 if carved else 0.055, min(mg, (pw - 0.0705 * max(ws) / capr) / 2))
            cap_w = (pw - 2 * mg) / max(ws) * capr
        nl_ = len(lines)
        cap = min(cap_w, 0.115, hmax / (1.0 + 1.7 * (nl_ - 1)))
        size = cap / capr
        phh = cap * (1.0 + 1.7 * (nl_ - 1)) + 2 * mg + 0.03
        print(f"DBG inscription cap height {cap * 1000:.0f} mm, plate {pw:.2f} x {phh:.2f} m on a {face_w:.2f} m face, {nl_} lines")
        if carved:      # stone tablet let into the die, letters cut through its face and filled with dark metal
            y1 = -hw - 0.016
            body = box("tablet", (pw, 0.02, phh), loc=(0, y1 + 0.01, zc), bevel=0.003)
            fill = box("tablet_fill", (pw - 0.05, 0.002, phh - 0.05), loc=(0, y1 - 0.001, zc))
            face = box("tablet_face", (pw - 0.03, 0.007, phh - 0.03), loc=(0, y1 - 0.0055, zc))
            rim = shape_extrude("tablet_rim", rounded_rect(pw + 0.03, phh + 0.03, 0.006, 2), [rounded_rect(pw - 0.04, phh - 0.04, 0.004, 2)],
                                depth=0.031, loc=(0, -hw + 0.002, zc), rot=(90, 0, 0), bevel=0.004)
            ok = False
            try:
                cut = _letters(lines, size, y1 + 0.004, zc, cap, 0.03, "cut")
                boolean(face, cut, "DIFFERENCE")
                bm = bmesh.new()
                bm.from_mesh(face.data)
                ok = len(bm.faces) > 30
                bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > 0.9], context="FACES")
                bm.to_mesh(face.data)
                bm.free()
            except Exception as ex:
                print("DBG letter cut failed:", ex)
            if not ok:
                lt = _letters(lines, size, y1 - 0.009, zc, cap, 0.002, "letters")
                assign(lt, bronze)
            assign(fill, bronze)
            paint(fill, grime=0.5, special=0.35 * vg)
            for o in (body, face, rim):
                assign(o, stone)
                stones.append(o)
        else:           # cast bronze plaque bolted to the die, raised gilt letters
            yf = -hw - 0.002
            plate = shape_extrude("plaque", rounded_rect(pw, phh, 0.012, 3), depth=0.012, loc=(0, yf, zc), rot=(90, 0, 0), bevel=0.003)
            frame = shape_extrude("plaque_frame", rounded_rect(pw - 0.012, phh - 0.012, 0.01, 3), [rounded_rect(pw - 0.05, phh - 0.05, 0.006, 3)],
                                  depth=0.007, loc=(0, yf - 0.012, zc), rot=(90, 0, 0), bevel=0.0025)
            bt = bolt("plaque_bolt", 0.011, 0.007, kind="dome")
            bolts = instances_to_mesh("plaque_bolts", bt, [trs((sx * (pw / 2 - 0.016), yf - 0.019, zc + sz * (phh / 2 - 0.016)), (90, 0, 0))
                                                           for sx in (-1, 1) for sz in (-1, 1)], keep_base=False)
            plq = [plate, frame, bolts]
            for o in plq:
                assign(o, bronze)
            lt = _letters(lines, size, yf - 0.012, zc, cap, 0.006, "letters")
            assign(lt, gilt)
            vgl = noise_mask(0.05, seed=seed + 21, lo=0.35, hi=0.8)
            paint(lt, grime=lambda co, n: 0.1 * vgl(co, n), wear=lambda co, n: 0.7 + 0.3 * vgl(co * 3.0, n))

    # ---- weathering of the stone
    streak = noise_mask(0.07, seed=seed + 3, lo=0.4, hi=0.72)
    blot = noise_mask(0.35, seed=seed + 4, lo=0.3, hi=0.7)
    fine = noise_mask(0.12, seed=seed + 5, lo=0.3, hi=0.7)
    sqz = np.array([1.0, 1.0, 0.06])
    joints = [z_s, z_s + 0.14 * m, z_d0, z_c0] + [z_d0 + z for z in zc_[1:-1]]
    rr_ = rng.fork("runs")
    runs = [(radians(a + rr_.jitter(0, 10)), rr_.u(0.03, 0.055), (cap_h + rr_.u(0.2, 0.5)) * (0.55 + 0.45 * vg), 0.75 + 0.25 * vg) for a in (45, 135, -45, -135)]
    runs += [(rr_.u(-pi, pi), rr_.u(0.02, 0.04), (cap_h + rr_.u(0.0, 0.4)) * (0.5 + 0.5 * vg), rr_.u(0.5, 0.9)) for _ in range(int(3 + 4 * vg))]

    def runm(co):
        phi = np.arctan2(co[:, 0], -co[:, 1])
        rxy = np.hypot(co[:, 0], co[:, 1])
        z = co[:, 2]
        out = np.zeros(len(co))
        for a0, sig, ln, st in runs:
            dphi = (phi - a0 + pi) % (2 * pi) - pi
            dist = dphi * rxy + 0.018 * np.sin(z * 19.0 + a0 * 7.0)
            fall = np.clip(1.0 - (H - z) / ln, 0.0, 1.0)
            out = np.maximum(out, st * np.exp(-(dist / (sig * (0.55 + 0.45 * fall))) ** 2) * fall ** 0.6)
        return out * (z < H + 0.002)

    def st_grime(co, n):
        z = co[:, 2]
        nz = n[:, 2]
        g = 0.5 * np.clip(1.0 - z / 0.4, 0, 1) ** 1.5                                         # splash-back at the pavement
        on_die = (z > z_d0 - 0.01) & (z < z_c0 + 0.01)
        under = np.clip(1.0 - (z_c0 - z) / 0.65, 0, 1)
        g = g + 0.85 * streak(co * sqz, n) * (0.3 + 0.7 * blot(co, n)) * under * on_die        # rain streaks below the drip
        bed = (z >= z_c0 - 0.005) & (z < z_c0 + 0.19 * m)
        g = g + 0.42 * bed * (0.55 + 0.45 * fine(co, n)) + 0.3 * np.clip(-nz, 0, 1)            # soot in the unwashed bed mouldings
        for zj in joints:
            g = g + 0.5 * np.exp(-np.abs(z - zj) / 0.022)                                    # damp, dirty joints
        g = g + 0.5 * runm(co)
        g = g + (z > z_c0 + 0.3 * m) * np.clip(nz, 0, 1) * (0.15 + 0.3 * blot(co * 2.0, n))
        return np.clip(g, 0, 1)

    def st_moss(co, n):
        y, z = co[:, 1], co[:, 2]
        up = np.clip(n[:, 2], 0, 1) ** 2
        north = np.clip(n[:, 1], 0, 1) * np.clip(1.0 - z / (z_s + 0.3), 0, 1)
        arris = np.clip((1.0 - np.abs(n).max(axis=1)) * 3.4, 0, 1) * (z < z_s + 0.02)
        wet = 0.35 + 0.65 * np.clip(y * 0.6 + 0.5, 0, 1)
        mo = (0.45 * up * wet * (z < z_s + 0.03) + 0.6 * north + 0.85 * arris * wet) * (0.25 + 0.95 * blot(co * 2.2, n))
        mo = mo + 0.5 * up * (z > z_c0 + 0.3 * m) * blot(co * 3.0, n)                         # lichen on the weathered top
        return np.clip(np.maximum(mo, np.clip(runm(co) * 1.45 - 0.3, 0, 1)), 0, 1)

    for o in stones:
        paint(o, grime=st_grime, special=st_moss)

    # ---- votive candles huddled on the top step: stubs of every height, guttered wax, some burnt out
    nca = int(p.candles)
    yc0 = -(hw + op + 0.1)
    if nca > 0:
        rc = rng.fork("candles")
        wax = mat("wax", "#e9dcc0")
        xc = rc.pick([-1, 1]) * hw * 0.5
        pts = []
        tries = 0
        while len(pts) < nca and tries < 400:
            tries += 1
            q = (xc + rc.jitter(0, 0.045 + 0.028 * nca), yc0 + rc.jitter(0, 0.055))
            if all((q[0] - a[0]) ** 2 + (q[1] - a[1]) ** 2 > 0.058 ** 2 for a in pts):
                pts.append(q)
        cs, fl, wk = [], [], []
        lit = []
        n_out = (1 if nca >= 4 else 0) + (1 if nca >= 6 else 0)
        for i, (cx, cy) in enumerate(pts):
            rr = rc.u(0.013, 0.026)
            out_ = i < n_out
            hh = rc.u(0.022, 0.05) if out_ else rc.u(0.05, 0.25) * (0.6 if rr > 0.021 else 1.0)
            rot = (rc.jitter(0, 5.0), rc.jitter(0, 5.0), rc.u(0, 360))
            dip = rc.u(0.004, 0.012)
            body = lathe(f"candle{i}", [(0.0, 0.0), (rr * rc.u(1.5, 2.3), 0.0), (rr * 1.6, 0.004), (rr * 1.12, 0.009), (rr, 0.022), (rr * rc.u(0.95, 1.03), hh * 0.6),
                                        (rr * 1.02, hh - 0.003), (rr * 0.9, hh), (rr * 0.55, hh - dip * 0.6), (0.0, hh - dip)], segments=8)
            parts = [body]
            for _ in range(rc.pick([1, 2, 2, 3])):       # wax runs frozen down the side
                a = rc.u(0, 2 * pi)
                ln = min(hh - 0.01, rc.u(0.015, 0.09))
                rd = rc.u(0.003, 0.0055)
                path = [((rr + rd * 0.3) * cos(a + 0.12 * t * rc.jitter(0, 1)), (rr + rd * 0.3) * sin(a), hh - 0.002 - ln * t) for t in (0.0, 0.35, 0.7, 1.0)]
                parts.append(sweep(f"drip{i}", path, rd, sides=4, smooth_path=False, scale_fn=lambda t: 0.75 + 0.6 * t * t))
            cj_ = join(parts, f"candle{i}") if len(parts) > 1 else body
            place(cj_, loc=(cx, cy, z_s), rot=rot)
            cs.append(cj_)
            tip = Vector((cx, cy, z_s)) + Euler((radians(rot[0]), radians(rot[1]), radians(rot[2]))).to_matrix() @ Vector((0, 0, hh - dip))
            wk.append(cylinder(f"wick{i}", 0.0014, 0.011, loc=(tip.x, tip.y, tip.z + 0.004), rot=(rc.jitter(0, 14), rc.jitter(0, 14), 0), verts=4))
            if not out_:
                fl.append(lathe(f"flame{i}", [(0.0, 0.0), (0.0034, 0.003), (0.0046, 0.009), (0.0026, 0.018), (0.0, 0.027)], segments=6, loc=(tip.x, tip.y, tip.z + 0.007)))
                fx(f"flame{i}", "flame", (tip.x, tip.y, tip.z + 0.014), size=0.018)
                lit.append(tip)
        cj = join(cs, "candles")
        assign(cj, wax)
        paint(cj, grime=lambda co, n: 0.2 + 0.5 * np.clip(1.0 - (co[:, 2] - z_s) / 0.03, 0, 1))
        wj = join(wk, "wicks")
        assign(wj, bronze)
        paint(wj, grime=1.0)
        if fl:
            fj = join(fl, "flames")
            assign(fj, flame_m)
            ctr = sum(lit, Vector((0, 0, 0))) / len(lit)
            light("candles", (ctr.x, ctr.y - 0.05, ctr.z + 0.07), color="#ffb060", intensity=0.5 + 0.2 * len(lit), range=3.5, flicker=0.5, shadow_near=0.1)
    socket("offering", "attach", (0, yc0, z_s), role="prop_top")

    # ---- the figure
    C = _pose(p.pose)
    rf = rng.fork("folds")
    C["ph"] = [rf.u(0, 2 * pi) for _ in range(12)]
    C["nf"] = int(rf.pick([10, 11, 12]))
    C["nfc"] = int(rf.pick([9, 10, 11]))
    C["knot"] = -0.5 if p.pose != "book" else rf.pick([-0.25, 0.25])
    C["hood_up"] = p.hood == "up"
    C["zt"], C["ra"] = _table([0.0, 0.06, 0.18, 0.5, 0.92, 1.08, 1.2, 1.32, 1.42, 1.5], [0.325, 0.3, 0.265, 0.228, 0.208, 0.176, 0.198, 0.218, 0.212, 0.135])
    _, C["rb"] = _table([0.0, 0.06, 0.18, 0.5, 0.92, 1.08, 1.2, 1.32, 1.42, 1.5], [0.27, 0.248, 0.215, 0.178, 0.16, 0.132, 0.144, 0.15, 0.124, 0.08])
    _, C["ca"] = _table([0.0, 0.2, 0.5, 0.9, 1.3, 1.42, 1.5, 1.56], [0.35, 0.335, 0.305, 0.285, 0.276, 0.266, 0.18, 0.1])
    _, C["cb"] = _table([0.0, 0.2, 0.5, 0.9, 1.3, 1.42, 1.5, 1.56], [0.3, 0.28, 0.248, 0.22, 0.202, 0.172, 0.127, 0.09])
    C["base_hs"] = (0.48, 0.45)
    ob = 0.10 * kk
    C["base_diag"] = ((2 * (hw + ob) - (c0 + 0.586 * ob)) * 0.7071 - 0.03) / fs

    def fw(v):          # figure space -> world
        v = np.asarray(v, float)
        return (v + np.array([0.0, 0.0, ZB])) * fs + np.array([0.0, 0.0, H])

    def to_fig(co):     # world -> figure space
        return (co - np.array([0.0, 0.0, H])) / fs - np.array([0.0, 0.0, ZB])

    fig = []
    fields = [lambda P: _body_field(P, C)]
    rubs = []

    # head: finer mesh, neck let into the cowl
    yaw, pitch = C["head"]
    HC = np.array([0.0, -0.012, 1.668])
    Rh = np.array(Matrix.Rotation(radians(yaw), 3, "Z") @ Matrix.Rotation(radians(pitch), 3, "X"))
    up = C["hood_up"]
    hlo, hhi = ((-0.165, -0.15, -0.27), (0.165, 0.2, 0.175)) if up else ((-0.115, -0.15, -0.23), (0.115, 0.155, 0.155))
    phh_ = C["ph"][10]
    beard = bool(p.beard)
    V, Fq = _nets(lambda P: _head_field(P, up, beard, phh_), hlo, hhi, 0.0058, project=1)
    head = _sdf_mesh("fig_head", fw(V @ Rh.T + HC), Fq, 11500 if up else 9000)
    fig.append(head)
    fields.append(lambda P: _head_field((P - HC) @ Rh, up, beard, phh_))
    rubs.append((fw(Rh @ np.array([0.0, -0.126, -0.04]) + HC), 0.016 * fs, 0.75))

    # hands: fine meshes let into the cuffs
    def hand(name, W, xh, yh, zh, fist, mirror):
        V, Fq = _nets(lambda P: _hand_field(P, fist), (-0.062, -0.078, -0.066), (0.082, 0.032, 0.195), 0.0044, smooth_it=1, project=1)
        sx = np.array([-1.0, 1.0, 1.0]) if mirror else np.ones(3)
        Rm = np.column_stack([xh, yh, zh])
        o = _sdf_mesh(name, fw((V * sx) @ Rm.T + W), Fq, 2200, flip=mirror)
        fig.append(o)
        fields.append(lambda P: _hand_field(((P - W) @ Rm) * sx, fist))
        if fist:
            rubs.append((fw(W + zh * 0.095 - yh * 0.02), 0.03 * fs, 0.6))
        return o

    # shoes: tapered, soled, showing under the lifted hem
    Vs, Fs = _nets(_shoe_field, (-0.075, -0.125, -0.006), (0.075, 0.085, 0.105), 0.004, project=1)
    for i, (sx_, sy_, splay) in enumerate(C["shoes"]):
        Rz = np.array(Matrix.Rotation(radians(-splay), 3, "Z"))
        T = np.array([sx_, sy_, 0.0])
        fig.append(_sdf_mesh(f"fig_shoe{i}", fw(Vs @ Rz.T + T), Fs, 800))
        fields.append(lambda P, Rz=Rz, T=T: _shoe_field((P - T) @ Rz))
        rubs.append((fw(Rz @ np.array([0.0, -0.08, 0.04]) + T), 0.035 * fs, 0.7))

    acc = []
    gspots = []         # where the pigeons sit: (x, y, radius, count weight)
    zup = np.array([0.0, 0.0, 1.0])
    if p.pose in ("lantern", "star"):
        ar = C["arms"][0]
        f = _unit(ar["W"] - ar["E"])
        if p.pose == "lantern":
            zh = f
            yh = _unit(zup - (zup @ f) * f)
            xh = np.cross(yh, zh)
        else:
            xh = zup
            zh = _unit(f - (f @ xh) * xh)
            yh = np.cross(zh, xh)
        hand("fig_hand_r", ar["W"], xh, yh, zh, True, False)
        G = fw(ar["W"] - 0.026 * yh + 0.092 * zh)
        if p.pose == "lantern":
            lan, lt_ = _lantern(G[0], G[1], G[2] - 0.062 * fs, fs, bronze, pane_m, flame_m, p.intensity, ring_dir=degrees(atan2(xh[1], xh[0])))
            soot = ((lt_[0], lt_[1], lt_[2] - 0.14 * fs), 0.45 * fs)
            acc += lan
            gspots.append((lt_[0], lt_[1], 0.085 * fs, 3))
            rubs.append(((lt_[0], lt_[1], lt_[3]), 0.04 * fs, 0.8))
            sm_ = fw((ar["S"] + ar["E"]) / 2)
            gspots.append((sm_[0], sm_[1], 0.1 * fs, 3))
        else:
            zb0 = H + ZB * fs * 0.6
            zs_ = float(fw((0, 0, 2.36))[2])        # star centre
            ztop = zs_ - 0.1 * fs
            rs_ = 0.0165 * fs
            sp = [(0.0, zb0), (rs_ * 1.5, zb0), (rs_ * 1.6, zb0 + 0.03 * fs), (rs_ * 1.15, zb0 + 0.05 * fs), (rs_ * 1.25, zb0 + 0.11 * fs), (rs_, zb0 + 0.125 * fs)]
            for zz in (0.62, 1.25, 1.9):
                zk = float(fw((0, 0, zz))[2])
                sp += [(rs_, zk - 0.03 * fs), (rs_ * 1.35, zk - 0.02 * fs), (rs_ * 1.1, zk), (rs_ * 1.35, zk + 0.02 * fs), (rs_, zk + 0.03 * fs)]
            sp += [(rs_, ztop - 0.16 * fs), (rs_ * 1.5, ztop - 0.145 * fs), (rs_ * 2.1, ztop - 0.115 * fs), (rs_ * 1.2, ztop - 0.085 * fs), (rs_ * 1.1, ztop - 0.05 * fs),
                   (rs_ * 2.0, ztop - 0.02 * fs), (rs_ * 2.3, ztop + 0.01 * fs), (rs_ * 1.2, ztop + 0.03 * fs), (0.0, ztop + 0.035 * fs)]
            acc.append(lathe("staff", sp, segments=12, loc=(G[0], G[1], 0)))
            acc.append(_star(G[0], G[1], zs_, fs, bronze, gilt, p.intensity))
            soot = ((G[0], G[1], zs_), 0.42 * fs)
        # left hand flat on a closed book held against the body
        al = C["arms"][1]
        zh = _unit((-0.80, -0.52, -0.30))
        yh = _unit(np.array([0.55, -0.83, 0.0]) - (np.array([0.55, -0.83, 0.0]) @ zh) * zh)
        xh = np.cross(yh, zh)
        hand("fig_hand_l", al["W"], xh, yh, zh, False, True)
        bc_ = al["W"] + zh * 0.055 - yh * (0.0145 + 0.024) + xh * 0.02
        rot = _frame_rot(zh, yh, np.cross(zh, yh))
        bw_, bt_, bh_ = 0.17 * fs, 0.048 * fs, 0.235 * fs
        acc.append(box("book_pages", (bw_ - 0.012 * fs, bt_ - 0.012 * fs, bh_ - 0.014 * fs), loc=fw(bc_), rot=rot, bevel=0.002))
        for s in (-1, 1):
            acc.append(box("book_board", (bw_, 0.007 * fs, bh_), loc=fw(bc_ + yh * s * (bt_ / fs / 2 - 0.0035)), rot=rot, bevel=0.002))
        acc.append(box("book_spine", (0.012 * fs, bt_, bh_), loc=fw(bc_ + zh * (bw_ / fs / 2 - 0.004)), rot=rot, bevel=0.005))
    else:
        ar = C["arms"][0]
        zh = _unit(ar["W"] - ar["E"])
        yh = _unit(zup - (zup @ zh) * zh)
        xh = np.cross(yh, zh)
        hand("fig_hand_r", ar["W"], xh, yh, zh, True, False)
        G = fw(ar["W"] - 0.026 * yh + 0.092 * zh)
        lan, lt_ = _lantern(G[0], G[1], G[2] - 0.062 * fs, 0.92 * fs, bronze, pane_m, flame_m, p.intensity, ring_dir=degrees(atan2(xh[1], xh[0])))
        soot = ((lt_[0], lt_[1], lt_[2] - 0.14 * fs), 0.45 * fs)
        acc += lan
        gspots.append((lt_[0], lt_[1], 0.08 * fs, 2))
        rubs.append(((lt_[0], lt_[1], lt_[3]), 0.04 * fs, 0.8))
        # open book lying on the left palm and forearm, tilted up to the reader
        nb = _unit((0.0, 0.34, 0.94))
        ub = _unit((0.0, -0.94, 0.34))
        al = C["arms"][1]
        zh = _unit(np.array([-0.35, 0, 0]) + ub)
        yh = -nb
        xh = np.cross(yh, zh)
        hand("fig_hand_l", al["W"], xh, yh, zh, False, True)
        ex = np.array([1.0, 0.0, 0.0])
        bc_ = al["W"] + 0.016 * nb + 0.06 * zh + ub * 0.035 - ex * 0.05
        rot = _frame_rot(ex, ub, nb)
        acc.append(box("book_cover", (0.34 * fs, 0.245 * fs, 0.008 * fs), loc=fw(bc_ + nb * 0.004), rot=rot, bevel=0.002))
        for s_ in (-1, 1):
            tilt = Matrix.Rotation(radians(-5.0 * s_), 3, tuple(ub))
            nx = np.array(tilt @ Vector(tuple(ex)))
            nz = np.array(tilt @ Vector(tuple(nb)))
            acc.append(box("book_leaf", (0.156 * fs, 0.225 * fs, 0.024 * fs), loc=fw(bc_ + ex * s_ * 0.081 + nb * (0.008 + 0.012 + 0.006)),
                           rot=_frame_rot(nx, ub, nz), bevel=0.004))
        bk = fw(bc_)
        gspots.append((bk[0], bk[1], 0.12 * fs, 2))

    # body last: it takes what is left of the triangle budget (curvature-driven decimation keeps the
    # folds, sleeve mouths and puddled hem, and thins the broad planes)
    armz = max(float(a["W"][2]) for a in C["arms"])
    lo = (-0.64, -0.5, -ZB - 0.03)
    hi = (0.54, 0.5, max(1.62, armz + 0.09))
    V, Fq = _nets(fields[0], lo, hi, 0.009, smooth_it=1, band=None, project=2)
    used = _tris([o for o in bpy.data.objects])
    target = int(clamp(69000 - used - 900 * float(p.pigeons), 22000, 33000))
    body = _sdf_mesh("fig_body", fw(V), Fq, target)
    fig.insert(0, body)
    print(f"DBG tris: others {used}, body {2 * len(Fq)} -> {_tris([body])}")
    for o in acc + fig:
        assign(o, bronze)

    # ---- patina, driven by the geometry of the casting itself
    def Ff(P):
        d = fields[0](P)
        for fn in fields[1:]:
            d = np.minimum(d, fn(P))
        return d

    sp_f, gr_f, we_f = _patina_fns(Ff, to_fig, vg, seed, rubs, soot)
    for o in fig:
        paint(o, special=sp_f, grime=gr_f, wear=we_f)
    a_ns = noise_mask(0.06, seed=seed + 14, lo=0.3, hi=0.7)
    a_nl = noise_mask(0.25, seed=seed + 15, lo=0.3, hi=0.7)
    g_ = vg ** 0.7

    def acc_sp(co, n):
        v = g_ * (1.3 * np.clip(n[:, 2], 0, 1) ** 1.4 + 0.75 * a_ns(co * np.array([1.0, 1.0, 0.1]), n) * (0.3 + 0.7 * a_nl(co, n)) + 0.15)
        v = v + max(vg - 0.55, 0.0) / 0.45 * (0.25 + 0.7 * a_nl(co * 2.0, n))
        return 0.9 * _ss(0.32, 0.62, v)

    def acc_we(co, n):
        w = np.zeros(len(co))
        for c, rr, s in rubs:
            w = np.maximum(w, s * np.exp(-(np.linalg.norm(co - np.asarray(c, float), axis=1) / rr) ** 2))
        return w

    for o in acc + plq:
        paint(o, special=lambda co, n: acc_sp(co, n) * (1.0 - acc_we(co, n)), grime=lambda co, n: 0.45 * acc_sp(co, n) + 0.25 * np.clip(-n[:, 2], 0, 1) + 0.14 + 0.25 * a_nl(co, n),
              wear=acc_we)

    # ---- pigeons
    pg = float(p.pigeons)
    if pg > 0.01:
        rg = rng.fork("pigeons")
        ht = fw(HC + np.array([0.0, 0.02, 0.0]))
        gspots = [(ht[0], ht[1], 0.085 * fs, 7)] + gspots
        for sx in (-1, 1):
            sh_ = fw((sx * 0.2, 0.01, 1.45))
            gspots.append((sh_[0], sh_[1], 0.075 * fs, 3))
        gspots.append((0.0, 0.07 * fs, 0.14 * fs, 2))
        for _ in range(4):          # the cornice ledge and the self-base
            a = rg.u(0, 2 * pi)
            rr = (hw + 0.15 * kk) / max(abs(cos(a)), abs(sin(a))) * rg.u(0.9, 1.0)
            gspots.append((rr * cos(a), rr * sin(a), 0.06, 2))
        ztop = float(fw((0, 0, 2.2))[2])
        spots = [(x_, y_, r_, ztop, int(round(n_ * pg * 1.5 + rg.u(0, 0.8)))) for x_, y_, r_, n_ in gspots]
        gu = _guano("droppings", fig + acc + [capo], spots, rg, mat("plaster_lime", "#d5d1c2"))
        if gu is not None:
            paint(gu, grime=lambda co, n: 0.12 + 0.3 * a_nl(co * 4.0, n))
