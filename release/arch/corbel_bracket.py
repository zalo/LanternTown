"""Carved oak jetty bracket. A short wall post (stopped chamfers, moulded cyma foot) carries a projecting
jetty joist with a moulded, end-grain nose; between them sits the bracket proper, tenoned into both and
held by draw-bored oak pegs that stand proud of the post and joist cheeks. Three bracket types:
'curved' (hollow knee brace, open spandrel, lamb's-tongue chamfer stops, sunk side panels with a carved
vine-and-leaf trail on an old painted ground), 'scroll' (solid console with volute rolls, sunk panels
and an acanthus leaf down the front) and 'strut' (plain straight strut, four stopped chamfers).
A turned acorn / square pendant drop or a forged lantern ring hangs under the joist end.

Origin: on the wall plane at the TOP of the joist (the jetty bressummer bears on z = 0); the bracket
hangs down to z = -height and projects to y = -projection.
"""
from tk import *
import math
import numpy as np

META = {"name": "Carved timber bracket", "class": "prop_small", "mount": "wall", "mount_height": 2.75,
        "focus": 0.55}
PARAMS = {
    "height": {"default": 0.9, "min": 0.5, "max": 1.4, "desc": "overall height, joist top to the foot of the wall post (m)"},
    "projection": {"default": 0.65, "min": 0.35, "max": 1.0, "desc": "projection of the joist nose from the wall (m)"},
    "thickness": {"default": 0.13, "min": 0.09, "max": 0.2, "desc": "thickness of the bracket across its face (m)"},
    "style": {"default": "curved", "choices": ["curved", "scroll", "strut"], "desc": "curved knee brace / scrolled console / plain straight strut"},
    "carving_depth": {"default": 0.012, "min": 0.0, "max": 0.025, "desc": "depth of the sunk panels and leaf carving (m); 0 = uncarved"},
    "pendant": {"default": "acorn", "choices": ["acorn", "square", "hook", "none"], "desc": "drop under the joist end: turned acorn, square pendant, forged lantern ring, nothing"},
    "pendant_length": {"default": 0.19, "min": 0.1, "max": 0.32, "desc": "length of the pendant drop (m)"},
    "strap": {"default": True, "desc": "forged iron strap nailed round the wall post below the bracket"},
    "finish": {"default": "oak", "choices": ["oak", "weathered", "painted"], "desc": "oiled oak / silvered bare timber / brush-painted joinery (colour = stain)"},
    "stain": {"default": "#96704a", "desc": "oak stain tone, or the paint colour of a painted bracket"},
    "paint": {"default": "#6a2a22", "desc": "old paint left in the carved grounds ('none' = bare oak)"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: grime, bare silvered arrises, flaked paint, moss, out-of-true"},
}
VARIANTS = {
    "scroll_console": {"style": "scroll", "height": 1.0, "projection": 0.62, "thickness": 0.15, "carving_depth": 0.016,
                       "pendant": "none", "finish": "painted", "stain": "#d9d2bf", "paint": "#2c5a4a", "strap": False,
                       "age": 0.35, "seed": 4},
    "plain_strut": {"style": "strut", "height": 1.15, "projection": 0.85, "thickness": 0.11, "carving_depth": 0.0,
                    "pendant": "hook", "finish": "weathered", "stain": "#8d8474", "paint": "none", "age": 0.85, "seed": 11},
    "small_square": {"height": 0.6, "projection": 0.42, "thickness": 0.1, "carving_depth": 0.008, "pendant": "square",
                     "pendant_length": 0.14, "paint": "none", "stain": "#b89a70", "strap": False, "age": 0.9, "seed": 23},
    "grand_blue": {"height": 1.32, "projection": 0.95, "thickness": 0.17, "carving_depth": 0.02, "pendant": "acorn",
                   "pendant_length": 0.28, "paint": "#2b4a78", "stain": "#5a3d28", "age": 0.3, "seed": 31},
}



# ---------------------------------------------------------------- small 2D helpers (side plane = y, z)
def _ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bez(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    s = 1.0 - t
    return (s ** 3) * np.asarray(p0, float) + 3 * s * s * t * np.asarray(p1, float) \
        + 3 * s * t * t * np.asarray(p2, float) + (t ** 3) * np.asarray(p3, float)


def _resample(pts, n):
    pts = np.asarray(pts, float)
    acc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s = np.linspace(0.0, acc[-1], n)
    return np.stack([np.interp(s, acc, pts[:, 0]), np.interp(s, acc, pts[:, 1])], 1), s


def _tangents(pts):
    t = np.gradient(np.asarray(pts, float), axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def _fill(keys, gap):
    ks = sorted(set(round(float(k), 5) for k in keys))
    out = [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if b - a < 2e-4:
            continue
        n = int(math.ceil((b - a) / gap))
        out += [a + (b - a) * j / n for j in range(1, n + 1)]
    return out


def _window(x, lo, hi, ramp):
    """1 inside [lo+ramp, hi-ramp], 0 outside [lo, hi] - the run of a stopped chamfer."""
    return _ss(lo, lo + ramp, x) * (1.0 - _ss(hi - ramp, hi, x))


def _orient(verts, faces, ref):
    """Wind every face so its normal agrees with ref(center, normal) > 0."""
    V = np.asarray(verts, float)
    out = []
    for f in faces:
        a, b, c = V[f[0]], V[f[1]], V[f[-1]]
        nrm = np.cross(b - a, c - a)
        if len(f) == 4 and np.dot(nrm, nrm) < 1e-18:
            nrm = np.cross(V[f[2]] - a, c - a)
        cen = V[list(f)].mean(axis=0)
        out.append(tuple(f) if ref(cen, nrm) >= 0 else tuple(reversed(f)))
    return out


def _loft(name, S, B, W, cs, cb, panel=None, cap0=False, cap1=False, flute=None):
    """Timber of width W (along X) whose side profile is swept between the soffit curve S and the back
    curve B (lists of (y, z)). cs / cb: chamfer of the two soffit / two back arrises per station (this is
    what makes stopped chamfers). panel: per-station (soffit border, back border, depth) of a sunk panel
    in both cheeks."""
    S = np.asarray(S, float)
    B = np.asarray(B, float)
    n = len(S)
    h = W * 0.5
    verts, m = [], 0
    for i in range(n):
        d = B[i] - S[i]
        L = float(np.linalg.norm(d))
        d = d / max(L, 1e-9)
        c0 = min(float(cs[i]), L * 0.3, h * 0.5)
        c1 = min(float(cb[i]), L * 0.3, h * 0.5)
        half = [(h - c0, S[i]), (h, S[i] + d * c0)]
        if panel is not None:
            bs, bb, dp = (float(v) for v in panel[i])
            bs = min(max(bs, c0 + 0.002), L * 0.34)
            bb = min(max(bb, c1 + 0.002), L * 0.34)
            if L - bs - bb < 2 * dp + 0.012:
                dp = 0.0
            e = max(dp, 0.0006)
            half += [(h, S[i] + d * bs), (h - dp, S[i] + d * (bs + e)), (h - dp, B[i] - d * (bb + e)), (h, B[i] - d * bb)]
        half += [(h, B[i] - d * c1), (h - c1, B[i])]
        ring = [(x, q[0], q[1]) for x, q in half] + [(-x, q[0], q[1]) for x, q in reversed(half)]
        if flute is not None:
            fdep, xh, nfl, land = flute          # stopped sunk channels (reeds) along the soffit
            wf = (2 * xh - (nfl - 1) * land) / nfl
            fd = float(fdep[i])
            sl_ = min(max(fd, 0.0005), wf * 0.3)
            for kf in range(nfl):
                xa = -xh + kf * (wf + land)
                for xx, dd in ((xa, 0.0), (xa + sl_, fd), (xa + wf - sl_, fd), (xa + wf, 0.0)):
                    q = S[i] + d * dd
                    ring.append((xx, q[0], q[1]))
        m = len(ring)
        verts += ring
    faces = []
    for i in range(n - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            faces.append((a, a + m, b + m, b))
    if cap0:
        faces.append(tuple(range(m)))
    if cap1:
        faces.append(tuple(range((n - 1) * m, n * m)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


class _Carve:
    """Collects relief carving (raised ridges / leaves) standing on the plane x = x0, facing sx."""

    def __init__(self):
        self.v, self.f = [], []

    def ridge(self, pts, wid, hgt, x0, sx, vein=0.0):
        pts = np.asarray(pts, float)
        n = len(pts)
        tan = _tangents(pts)
        nor = np.stack([-tan[:, 1], tan[:, 0]], 1)
        prof = [(-1.0, -0.12), (0.0, 1.0), (1.0, -0.12)] if vein <= 0 else \
            [(-1.0, -0.12), (-0.55, 0.85), (-0.3, 1.0), (0.0, vein), (0.3, 1.0), (0.55, 0.85), (1.0, -0.12)]
        k = len(prof)
        base = len(self.v)
        for i in range(n):
            w = max(float(wid[i]), 0.0004)
            for a, b in prof:
                q = pts[i] + nor[i] * (a * w)
                self.v.append((x0 + sx * max(b * float(hgt[i]), -0.0012), q[0], q[1]))
        fs = []
        for i in range(n - 1):
            for j in range(k - 1):
                a = base + i * k + j
                fs.append((a, a + 1, a + k + 1, a + k))
        self.f += _orient(self.v, fs, lambda c, nr: nr[0] * sx)

    def boss(self, c, R, hgt, x0, sx, n=10):
        prof = [(1.0, -0.12), (0.88, 0.5), (0.55, 0.9), (0.0, 1.0)]
        base = len(self.v)
        for (r, hh) in prof[:-1]:
            for j in range(n):
                a = 2 * math.pi * j / n
                self.v.append((x0 + sx * max(hh * hgt, -0.0012), c[0] + R * r * math.cos(a), c[1] + R * r * math.sin(a)))
        self.v.append((x0 + sx * hgt, c[0], c[1]))
        apex = len(self.v) - 1
        fs = []
        for i in range(len(prof) - 2):
            for j in range(n):
                a, b = base + i * n + j, base + i * n + (j + 1) % n
                fs.append((a, b, b + n, a + n))
        i = len(prof) - 2
        for j in range(n):
            fs.append((base + i * n + j, base + i * n + (j + 1) % n, apex))
        self.f += _orient(self.v, fs, lambda c_, nr: nr[0] * sx)

    def obj(self, name):
        return mesh_object(name, self.v, self.f) if self.f else None


def _trail(cv, S, D, L, bs, bb, dpv, dep, h, phase):
    """Spandrel carving in the sunk panels: a six-petalled rose on the widest part of the ground and a
    vine running from it into both tails of the panel, a veined leaf filling every hollow of the stem."""
    idx = np.where(dpv > 0.9 * dep)[0]
    if len(idx) < 6:
        return
    sl = slice(idx[0], idx[-1] + 1)
    fw = L[sl] - bs[sl] - bb[sl] - 2 * dep
    cen = S[sl] + D[sl] * (bs[sl] + dep + fw * 0.5)[:, None]
    acc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cen, axis=0), axis=1))])
    Lt = acc[-1]
    if Lt < 0.1:
        return

    def at(sv, off=0.0):
        sv = np.clip(sv, 0.0, Lt)
        c = np.stack([np.interp(sv, acc, cen[:, 0]), np.interp(sv, acc, cen[:, 1])], -1)
        dv = np.stack([np.interp(sv, acc, D[sl][:, 0]), np.interp(sv, acc, D[sl][:, 1])], -1)
        dv = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-9)
        f = np.interp(sv, acc, fw)
        return c + dv * (np.asarray(off) * f)[..., None], f

    ipk = int(np.argmax(fw))
    spk, fpk = float(acc[ipk]), float(fw[ipk])
    Rr = min(0.44 * fpk, 0.3 * Lt)
    cpk, _ = at(np.array(spk))
    npet = 6
    u = np.linspace(0.0, 1.0, 6)
    for sx in (1, -1):
        x0 = sx * (h - dep)
        for k in range(npet):
            a = phase + 2 * math.pi * k / npet
            axis = cpk + (Rr * (0.16 + 0.84 * u))[:, None] * np.array([math.cos(a), math.sin(a)])
            wp = Rr * 0.46 * (0.3 + 0.7 * np.sin(math.pi * u ** 1.25)) * np.sqrt(1.0 - u ** 6)
            cv.ridge(axis, wp, dep * 0.95 * (0.75 + 0.25 * np.sin(math.pi * u)), x0, sx, vein=0.45)
        cv.boss(cpk, 0.26 * Rr, dep * 1.05, x0, sx)
        for (sa, sb, dirn) in ((0.012, spk - Rr - 0.006, -1), (spk + Rr + 0.006, Lt - 0.012, 1)):
            Ln = sb - sa
            if Ln < 0.07:
                continue
            fm = float(np.mean(at(np.linspace(sa, sb, 8))[1]))
            nh = max(1, int(round(Ln / max(1.25 * fm, 0.07))))
            lam = Ln / nh
            r = np.linspace(0.0, Ln, max(8, int(Ln / 0.012)))

            def s_of(rv):
                return sa + rv if dirn > 0 else sb - rv

            def wav(rv):
                env = np.clip(rv / (0.3 * lam), 0, 1) * np.clip((Ln - rv) / (0.3 * lam), 0, 1)
                return 0.25 * np.sin(math.pi * rv / lam) * env

            stem, fs_ = at(s_of(r), wav(r))
            tap = 1.0 - 0.55 * r / Ln
            cv.ridge(stem, 0.06 * fs_ * tap + 0.002, dep * 0.68 * (0.6 + 0.4 * tap), x0, sx)
            for k in range(nh):
                rk = (k + 0.5) * lam
                sg = 1.0 if k % 2 == 0 else -1.0
                r0 = rk - 0.4 * lam
                q0, _ = at(np.array(s_of(r0)), float(wav(np.array(r0))))
                q1, f1 = at(np.array(s_of(rk + 0.02 * lam)), -sg * 0.24)
                q2, _ = at(np.array(s_of(min(rk + 0.5 * lam, Ln + 0.2 * lam))), -sg * 0.04)
                ctrl = 2.0 * q1 - 0.5 * (q0 + q2)
                a = np.linspace(0.0, 1.0, 8)[:, None]
                axis = (1 - a) ** 2 * q0 + 2 * a * (1 - a) * ctrl + a ** 2 * q2
                a = a[:, 0]
                wp = 0.21 * float(f1) * np.sin(math.pi * a ** 0.72) ** 0.75 * (1.0 + 0.16 * np.cos(a * math.pi * 5.0))
                hp = dep * 0.98 * (0.35 + 0.65 * np.sin(math.pi * np.clip(a * 1.08, 0, 1)) ** 0.6)
                cv.ridge(axis, wp, hp, x0, sx, vein=0.4)


def build(p, rng):
    H, P, W = float(p.height), float(p.projection), float(p.thickness)
    dep = float(p.carving_depth)
    if dep < 0.003:
        dep = 0.0
    age = float(p.age)
    style = p.style
    seed = int(p.seed)
    painted = dep > 0 and str(p.paint).startswith("#")

    oak = mat({"oak": "wood_oak", "weathered": "wood_weathered", "painted": "wood_painted"}[p.finish], p.stain)
    endg = mat("wood_endgrain", p.stain) if p.finish != "painted" else oak
    sr, sg_, sb = hex_rgb(p.stain)
    pegm = mat("wood_endgrain", (min(1.0, sr * 1.7 + 0.03), min(1.0, sg_ * 1.7 + 0.03), min(1.0, sb * 1.7 + 0.03))
               if p.finish != "painted" else "#b08a5e")
    iron = mat("iron_wrought")
    pnt = mat("wood_painted", p.paint) if painted else None

    # ---- principal dimensions
    pd = 0.085                                   # wall post stands this far proud of the wall plane
    Wp = W + 0.034                               # post and joist are wider than the bracket tenoned into them
    bh = clamp(0.1 + 0.07 * P, 0.12, 0.17)       # joist depth
    foot = min(0.12, 0.18 * H)                   # moulded foot of the post
    nl = 0.6 * bh                                # length of the moulded nose
    ft = 0.3 * bh                                # top fillet of the nose
    ease = 0.0035                                # every arris is at least eased
    C = clamp(0.2 * W, 0.016, 0.034)             # stopped chamfer
    pr = clamp(W * 0.36, 0.032, 0.055)           # pendant radius
    reserve = {"acorn": 2 * pr + 0.03, "square": 2 * pr + 0.03, "hook": 0.085, "none": 0.03}[p.pendant]
    yN = -(P - nl)                               # underside start of the nose
    yF = yN + reserve                            # front of the bracket under the joist
    zL = -H + foot + 0.075                       # foot of the bracket on the post
    yPend = yN + 0.012 + pr
    solid = []                                   # timber lofts (oak)

    # ---- bracket side profile: soffit curve S, back curve B
    NS = 40
    sh = clamp(0.06 * min(H, P) + 0.012, 0.03, 0.055)      # square shoulders at foot and head of the bracket
    s0 = np.array([-pd + 0.006, zL])
    s3 = np.array([yF, -bh + 0.006])
    dy, dz = abs(s3[0] - s0[0]), s3[1] - s0[1]
    R2 = clamp(0.25 * min(dy, dz), 0.04, 0.1)
    R1 = 0.64 * R2
    c1 = np.array([-(pd + R1 - 0.005), zL + R1])
    c2 = np.array([yF + R2, -bh - R2 + 0.005])
    if style == "curved":
        a0, a3 = np.array([-pd - sh, zL]), np.array([yF, -bh - sh])
        ddy, ddz = abs(a3[0] - a0[0]), a3[1] - a0[1]
        raw = _bez(a0, a0 + np.array([-0.2 * ddy, 0.46 * ddz]), a3 + np.array([0.5 * ddy, -0.18 * ddz]), a3, 64)
        lpath = np.array([[-pd + 0.02, zL], [-pd + 0.02, -bh + 0.02], [yF, -bh + 0.02]])
    elif style == "strut":
        raw = np.stack([s0, s3])
        lpath = None
    else:
        pb = c1 + 0.97 * R1 * np.array([math.cos(radians(172)), math.sin(radians(172))])
        pa = c2 + 0.97 * R2 * np.array([math.cos(radians(285)), math.sin(radians(285))])
        dist = float(np.linalg.norm(pa - pb))
        raw = _bez(pb, pb + 0.34 * dist * np.array([0.14, 0.99]), pa + 0.36 * dist * np.array([0.5, -0.866]), pa, 64)
        raw[:, 0] = np.minimum(raw[:, 0], -pd - 0.06)
        lpath = np.array([[-pd + 0.02, c1[1] - 0.3 * R1], [-pd + 0.02, -bh + 0.02], [c2[0] + 0.3 * R2, -bh + 0.02]])
    S, arc = _resample(raw, NS)
    Ls = arc[-1]
    tan = _tangents(S)
    nrm = np.stack([tan[:, 1], -tan[:, 0]], 1)          # towards the post / joist corner
    t01 = arc / Ls
    if lpath is not None:
        # solid spandrel: the back of the bracket is housed 20 mm into post and joist
        yb_, z0_, zt_, ye_ = lpath[0][0], lpath[0][1], lpath[1][1], lpath[2][0]
        lpo, lbe = zt_ - z0_, yb_ - ye_
        qs = []
        for i in range(NS):                      # pair every soffit station with the point its normal hits
            sp, nn = S[i], nrm[i]
            qh = None
            if nn[0] > 1e-6:
                zz = sp[1] + (yb_ - sp[0]) / nn[0] * nn[1]
                if zz <= zt_:
                    qh = clamp(zz - z0_, 0.0, lpo)
            if qh is None:
                qh = lpo + (clamp(yb_ - (sp[0] + (zt_ - sp[1]) / nn[1] * nn[0]), 0.0, lbe) if nn[1] > 1e-6 else 0.0)
            wq = math.sin(math.pi * t01[i]) ** 0.6
            qs.append(lerp(t01[i] * (lpo + lbe), qh, wq))
        qs = np.maximum.accumulate(np.array(qs))
        B = np.array([(yb_, z0_ + qv) if qv <= lpo else (yb_ - (qv - lpo), zt_) for qv in qs])
        Bu = B.copy()
        out = np.full(NS, -1.0)
        kc = 1.0 if style == "curved" else 0.55
        stop = min(0.07, Ls * 0.14) if style == "curved" else 0.035
        win = _window(arc, stop, Ls - stop, 0.045)
        cs_b = ease + (C * kc - ease) * win
        cb_b = np.full(NS, 0.002)
        b_s = np.full(NS, C * kc + 0.012)
        b_b = np.full(NS, 0.02 + 0.026)
        Lfan = np.linalg.norm(B - S, axis=1)
        pan_dep = dep * win * _ss(0.0, 0.035, Lfan - b_s - b_b - 2 * dep - 0.012)
    else:
        bd = clamp(0.08 + 0.05 * P, 0.09, 0.13)
        Bu = S + nrm * bd
        B = Bu.copy()
        B[:, 0] = np.minimum(B[:, 0], -pd + 0.02)
        B[:, 1] = np.minimum(B[:, 1], -bh + 0.02)
        out = np.minimum(-pd - Bu[:, 0], -bh - Bu[:, 1])       # > 0 where the back edge is clear of post and joist
        stop = min(0.075, Ls * 0.16)
        win = _window(arc, stop, Ls - stop, 0.045)
        cs_b = ease + (C - ease) * win
        cb_b = ease + (C - ease) * _ss(0.03, 0.075, out)
        pan_dep = np.zeros(NS)
        b_s = np.full(NS, C + 0.014)
        b_b = np.full(NS, C + 0.014)
    Dv = B - S
    Lv = np.linalg.norm(Dv, axis=1)
    Dv = Dv / np.maximum(Lv, 1e-9)[:, None]
    panel = np.stack([b_s, b_b, pan_dep], 1) if (dep > 0 and style != "strut") else None
    flute = None
    if style == "curved" and dep > 0:
        xh = W * 0.5 - C - 0.007
        if xh > 0.022:
            flute = (min(0.6 * dep, 0.009) * _window(arc, stop + 0.03, Ls - stop - 0.03, 0.03), xh, 2, 0.012)
    brace = _loft("brace", S, B, W, cs_b, cb_b, panel, cap0=style == "curved", cap1=style == "curved", flute=flute)
    chord = Vector((0.0, float(S[-1][0] - S[0][0]), float(S[-1][1] - S[0][1]))).normalized()
    carve = _Carve()
    if panel is not None:
        _trail(carve, S, Dv, Lv, b_s, b_b, pan_dep, dep, W * 0.5, rng.fork("trail").u(0.0, 1.0))

    # ---- where the open spandrel is (for the stopped chamfers of post and joist)
    clear = np.where(out > 0.0)[0]
    if len(clear) > 2:
        z_gap0 = float(Bu[clear[0], 1])           # brace leaves the post here
        y_gap1 = float(Bu[clear[-1], 0])          # ... and reaches the joist here
    else:
        z_gap0, y_gap1 = -bh, -pd
    pz = (z_gap0 + 0.045, -bh - 0.04) if (-bh - z_gap0) > 0.2 else None
    bz = (y_gap1 + 0.045, -pd - 0.04) if (-pd - y_gap1) > 0.2 else None

    # ---- wall post with moulded foot
    ztop, zf = -bh + 0.012, -H + foot
    keys = [ztop, zf]
    if pz:
        keys += [pz[0], pz[0] + 0.02, pz[0] + 0.04, pz[1] - 0.04, pz[1] - 0.02, pz[1]]
    zs = _fill(keys, 0.1)[::-1]
    Sp = [(-pd, z) for z in zs]
    csp = [ease + ((C - ease) * float(_window(z, pz[0], pz[1], 0.04)) if pz else 0.0) for z in zs]
    g = 0.007
    hh = foot - 3 * g
    ftp = [(-pd + g, zf - g), (-pd + g, zf - 2 * g), (-pd, zf - 3 * g)]
    ftp += [tuple(q) for q in _bez((-pd, zf - 3 * g), (-pd - 0.004, zf - 3 * g - 0.55 * hh), (-0.022, -H + 0.6 * hh), (-0.022, -H), 9)[1:]]
    Sp += ftp
    csp += [ease] * len(ftp)
    post = _loft("post", Sp, [(0.0, q[1]) for q in Sp], Wp, csp, [0.002] * len(Sp), cap1=True)
    solid.append((post, "Z"))

    # ---- jetty joist with moulded nose
    keys = [0.0, yN, -pd]
    if bz:
        keys += [bz[0], bz[0] + 0.02, bz[0] + 0.04, bz[1] - 0.04, bz[1] - 0.02, bz[1]]
    ys = _fill(keys, 0.12)[::-1]
    Sb = [(y, -bh) for y in ys]
    csj = [ease + ((C - ease) * float(_window(y, bz[0], bz[1], 0.04)) if bz else 0.0) for y in ys]
    q = 0.012
    nose = [(yN - 0.008, -bh + 0.008)]
    nose += [tuple(v) for v in _bez((yN - 0.008, -bh + 0.008), (yN - 0.008 - 0.62 * (nl - q), -bh + 0.008),
                                    (-P + q, -bh + 0.5 * (bh - ft)), (-P + q, -ft - q), 8)[1:]]
    nose += [(-P, -ft)]
    Sb += nose
    csj += [ease] * len(nose)
    Bb = [(v[0], 0.0) for v in Sb]
    beam = _loft("joist", Sb, Bb, Wp, csj, [ease] * len(Sb), cap0=True, cap1=True)
    solid.append((beam, "Y"))

    # ---- hand-made irregularity, materials, UVs
    wob = 0.0012 + 0.002 * age
    for k, (o, ax) in enumerate(solid):
        displace_noise(o, strength=wob, scale=0.22, seed=seed + 3 * k)
        transform_verts(o, lambda co: Vector((co.x, min(co.y, -0.0004), co.z)))
        assign(o, oak)
        uv_grain(o, axis=ax)
        smooth(o, 38)
    assign_faces(beam, endg, lambda c, n: n.y < -0.5 and abs(n.x) < 0.5 and c.y < yN)
    displace_noise(brace, strength=wob, scale=0.2, seed=seed + 17, along_normal=False, axis_mask=(0, 1, 1))
    assign(brace, oak)
    if pnt is not None and panel is not None:
        hx = W * 0.5 - dep
        assign_faces(brace, pnt, lambda c, n: abs(n.x) > 0.9 and abs(abs(c.x) - hx) < 0.0015)
    uv_grain(brace, axis=chord)
    smooth(brace, 38)
    carved = [brace]

    # ---- scroll console extras: volute rolls with spiral fillets and eyes, acanthus leaf down the front
    if style == "scroll":
        Wr = W + 0.006
        rh = max(dep * 0.6, 0.004)
        for (c, R, a0, dirn) in ((c1, R1, 172.0, 1.0), (c2, R2, 285.0, -1.0)):
            roll = cylinder("volute", R, Wr, loc=(0, c[0], c[1]), rot=(0, 90, 0), verts=22, bevel=0.004, segments=1)
            assign(roll, oak)
            uv_grain(roll, axis=chord)
            smooth(roll, 38)
            carved.append(roll)
            k = np.linspace(0.0, 1.0, 25)
            ang = radians(a0) + dirn * k * 1.55 * 2 * math.pi
            rr = R * (0.86 - 0.68 * k)
            sp = np.stack([c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang)], 1)
            for sx in (1, -1):
                carve.ridge(sp, R * 0.085 * (1.0 - 0.5 * k), rh * (1.0 - 0.3 * k), sx * Wr * 0.5, sx)
                eye = lathe("eye", [(0.0, -0.002), (R * 0.2, -0.002), (R * 0.2, rh * 0.5), (R * 0.15, rh * 1.3), (R * 0.07, rh * 1.9), (0.0, rh * 2.0)],
                            segments=10, loc=(sx * Wr * 0.5, c[0], c[1]), rot=(0, 90 * sx, 0))
                assign(eye, oak)
                uv_grain(eye, axis=chord)
                carved.append(eye)
        if dep > 0:
            # acanthus leaf lying on the hollow front, stalk tucked under the top roll, tip pointing down
            nu, nv = 8, 18
            hwid = W * 0.5 - C * 0.55 - 0.005
            onrm = -nrm
            fidx = np.arange(NS, dtype=float)
            lv, lf = [], []
            nlobe = 3
            for j in range(nv + 1):
                v = j / nv
                t = lerp(0.985, 0.36, v) * (NS - 1)
                cy, cz = np.interp(t, fidx, S[:, 0]), np.interp(t, fidx, S[:, 1])
                oy, oz = np.interp(t, fidx, onrm[:, 0]), np.interp(t, fidx, onrm[:, 1])
                ph = (v * nlobe) % 1.0                                   # saw-tooth lobes: swell out, cut back sharply
                lobe = 0.62 + 0.38 * math.sin(math.pi * min(ph * 1.25, 1.0)) ** 0.7
                taper = (0.72 + 0.28 * min(v * 4.0, 1.0)) * (1.0 - v ** 2.6)
                wv = max(lobe * taper, 0.02)
                gv = (0.6 + 0.4 * math.sin(math.pi * min(v * 1.1, 1.0))) * (0.85 + 0.15 * lobe)
                for i in range(nu + 1):
                    u = 2.0 * i / nu - 1.0
                    e = float(np.interp(abs(u), [0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.34, 0.74, 0.42, -0.25]))
                    hgt = max(dep * 1.2 * e * gv, -0.002)
                    lv.append((u * wv * hwid, cy + oy * hgt, cz + oz * hgt))
            for j in range(nv):
                for i in range(nu):
                    a = j * (nu + 1) + i
                    lf.append((a, a + 1, a + nu + 2, a + nu + 1))
            ref_c = (c2 + np.array([R2 * 2.5, -R2 * 2.0]))
            lf = _orient(lv, lf, lambda cen, nr: -(nr[1] * (ref_c[0] - cen[1]) + nr[2] * (ref_c[1] - cen[2])))
            leaf = mesh_object("acanthus", lv, lf)
            assign(leaf, oak)
            uv_grain(leaf, axis=chord)
            smooth(leaf, 60)
            carved.append(leaf)

    cobj = carve.obj("carving")
    if cobj is not None:
        assign(cobj, oak)
        uv_grain(cobj, axis=chord)
        smooth(cobj, 50)
        carved.append(cobj)

    # ---- draw-bored oak pegs: two through the post, two through the joist, proud and never quite flush
    rp = rng.fork("pegs")
    cpost = clamp((z_gap0 - zL) if len(clear) > 2 else 0.5 * dz, 0.1, 0.34)
    cbeam = clamp((y_gap1 - yF) if len(clear) > 2 else 0.5 * dy, 0.08, 0.3)
    if style == "scroll":
        cpost, cbeam = min(0.6 * dz, 0.4), min(0.6 * dy, 0.34)
    spots = [(-pd + 0.03, zL + 0.28 * cpost), (-pd + 0.046, zL + 0.74 * cpost),
             (yF + 0.3 * cbeam, -bh + 0.034), (yF + 0.76 * cbeam, -bh + 0.05)]
    pegb = cylinder("peg", 0.0115, 0.04, verts=8, radius2=0.0095)
    pm = []
    for (py, pz_) in spots:
        head = rp.pick([1, -1])
        for sx in (1, -1):
            prot = rp.u(0.007, 0.015) if sx == head else rp.u(0.002, 0.006)
            pm.append(trs((sx * (Wp * 0.5 + prot - 0.02), py + rp.jitter(0, 0.004), pz_ + rp.jitter(0, 0.004)),
                          (rp.jitter(0, 5), 90 * sx + rp.jitter(0, 5), 0), (rp.u(0.9, 1.1), rp.u(0.9, 1.1), 1.0)))
    pegs = instances_to_mesh("pegs", pegb, pm, keep_base=False)
    assign(pegs, pegm)

    # ---- pendant drop / lantern ring under the joist end
    irons = []
    Lp = float(p.pendant_length)
    if p.pendant == "acorn":
        prof = [(0.0, 0.0), (0.2, 0.012), (0.3, 0.045), (0.22, 0.08), (0.5, 0.125), (0.86, 0.22), (1.0, 0.34), (0.95, 0.46),
                (0.74, 0.545), (0.78, 0.56), (1.06, 0.585), (1.08, 0.64), (0.8, 0.665), (0.52, 0.72), (0.46, 0.79),
                (0.62, 0.84), (0.9, 0.875), (0.96, 0.92), (0.96, 1.0)]
        pend = lathe("pendant", [(r * pr, -Lp + f * Lp) for r, f in prof], segments=20, loc=(0, yPend, -bh + 0.004))
        assign(pend, oak)
        uv_grain(pend, axis="Z")
        carved.append(pend)
    elif p.pendant == "square":
        k = 1.25
        prof = [(0.0, 0.0), (0.16, 0.0), (0.34, 0.07), (0.3, 0.1), (0.52, 0.16), (0.9, 0.5), (0.9, 0.56), (0.74, 0.6), (0.74, 0.64),
                (1.0, 0.68), (1.0, 0.84), (0.86, 0.88), (0.86, 0.92), (1.0, 0.95), (1.0, 1.0)]
        pend = lathe("pendant", [(r * pr * k, -Lp + f * Lp) for r, f in prof], segments=4, loc=(0, yPend, -bh + 0.004),
                     rot=(0, 0, 45), smooth_angle=30)
        bevel(pend, 0.004, segments=1)
        assign(pend, oak)
        uv_grain(pend, axis="Z")
        carved.append(pend)
    elif p.pendant == "hook":
        yh = yN + 0.04
        irons.append(box("hook_plate", (0.05, 0.06, 0.005), loc=(0, yh, -bh - 0.0025), bevel=0.001, segments=1))
        st = [(0, yh - 0.016, -bh), (0, yh - 0.016, -bh - 0.02), (0, yh - 0.01, -bh - 0.03), (0, yh, -bh - 0.034),
              (0, yh + 0.01, -bh - 0.03), (0, yh + 0.016, -bh - 0.02), (0, yh + 0.016, -bh)]
        irons.append(sweep("hook_staple", st, 0.0045, sides=7, up=(1, 0, 0)))
        rr = 0.03
        zc = -bh - 0.034 + 0.0045 - rr + 0.0045
        irons.append(torus("hook_ring", rr, 0.0048, loc=(0, yh, zc), rot=(90, 0, 8), major_seg=22, minor_seg=6))
        socket("lantern", "attach", (0, yh, zc - rr), role="lantern")

    # ---- forged strap round the post under the bracket, clout-nailed
    if p.strap:
        zs_ = zL - 0.036 + rng.jitter(0, 0.004)
        tk_, sw = 0.004, 0.034
        tilt = rng.jitter(0, 1.5)
        irons.append(box("strap_f", (Wp + 2 * tk_, tk_, sw), loc=(0, -pd - tk_ * 0.5, zs_), rot=(0, tilt, 0), bevel=0.001, segments=1))
        nb = bolt("nail", 0.0075, 0.0045, kind="pyramid")
        nm = [trs((sxn * Wp * 0.26, -pd - tk_, zs_ + rng.jitter(0, 0.004)), (90, 0, 0)) for sxn in (-1, 1)]
        for sx in (-1, 1):
            zz = zs_ - sx * math.tan(radians(tilt)) * Wp * 0.5
            irons.append(box("strap_s", (tk_, pd - 0.001, sw), loc=(sx * (Wp + tk_) * 0.5, -pd * 0.5 - 0.0005, zz), bevel=0.001, segments=1))
            nm += [trs((sx * (Wp * 0.5 + tk_), -pd * f + rng.jitter(0, 0.004), zz + rng.jitter(0, 0.004)), (0, 90 * sx, 0)) for f in (0.3, 0.72)]
        irons.append(instances_to_mesh("nails", nb, nm, keep_base=False))
    for o in irons:
        assign(o, iron)
        if not o.name.startswith(("hook_staple", "hook_ring")):
            smooth(o, 35)

    # ---- weathering: dust and soot gathered in the sheltered corner, rain-bleached nose and soffit,
    #      paint flaking from the grounds, green at the foot where the drip runs down the post
    nz1 = noise_mask(0.13, seed=seed + 5, lo=0.3, hi=0.7)
    nz2 = noise_mask(0.05, seed=seed + 9, lo=0.35, hi=0.7)
    hx = W * 0.5 - dep

    def grime_fn(co, n):
        dc = np.hypot(co[:, 1] + pd, co[:, 2] + bh)
        g = np.clip(1.0 - dc / (0.3 + 0.25 * P), 0.0, 1.0) ** 1.4
        under = np.clip(-n[:, 2], 0.0, 1.0) * 0.3
        return np.clip((0.85 * g + under) * (0.35 + 0.65 * age), 0.0, 1.0)

    def wear_fn(co, n):
        front = np.clip(-n[:, 1] * 0.8 - n[:, 2] * 0.35 + 0.1, 0.0, 1.0)
        outer = np.clip(-co[:, 1] / P, 0.0, 1.0)
        w = front * (0.3 + 0.7 * outer) * nz1(co, n) * (0.25 + 1.0 * age)
        if painted:
            floor = (np.abs(np.abs(co[:, 0]) - hx) < 0.002) & (np.abs(n[:, 0]) > 0.8)
            w = np.where(floor, np.maximum(w, nz2(co, n) * (0.25 + 0.7 * age)), w)
        return np.clip(w, 0.0, 1.0)

    moss = mask_mul(height_ramp(-H, -H + 0.3), noise_mask(0.08, seed=seed + 2, lo=0.4, hi=0.75), 0.75 * age)
    for o in [post, beam] + carved:
        paint(o, grime=grime_fn, wear=wear_fn, special=moss)
    paint(pegs, grime=0.15 + 0.3 * age, wear=0.15 * age)
    for o in irons:
        paint(o, special=mask_mul(noise_mask(0.04, seed=seed + 7, lo=0.25, hi=0.7), 0.35 + 0.6 * age), grime=0.2 * age)

    socket("bearer", "attach", (0, -P * 0.5, 0), role="prop_top")
