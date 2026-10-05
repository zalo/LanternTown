"""Retaining wall of the upper terrace, running along X with the earth on +Y. Built the way such walls
are built: a rock-faced plinth course with a weathered (sloped) plinth cap, a slightly battered face of
individually modelled coursed blocks (or rubble / brick panels between dressed stone), then either a
blind arcade of relieving arches (voussoir rings with keystones springing from impost blocks on flush
piers, the bays recessed and filled with rock-faced walling) or projecting two-stage buttress piers with
sloped set-offs, a moulded string course, a blocking course and a throated coping whose flat top (z =
height) carries the balustrade. Lead drain spouts in dressed outlet stones leave long damp, mossy streaks,
forged tie-rod anchor plates bleed rust, moss cushions sit on every ledge.
"""
from tk import *
from math import sin, cos, pi, sqrt, acos, asin, atan2, degrees, radians, ceil
import numpy as np

META = {"name": "Terrace retaining wall", "class": "structure", "mount": "ground", "focus": 0.56}
PARAMS = {
    "length": {"default": 8.0, "min": 2.0, "max": 16.0, "desc": "run length along X (m); spans exactly -length/2..+length/2"},
    "height": {"default": 2.4, "min": 0.6, "max": 3.5, "desc": "height of the flat coping top = upper terrace level (m)"},
    "facing": {"default": "ashlar", "choices": ["ashlar", "rubble", "brick"], "desc": "walling between the dressed stone members"},
    "pier_spacing": {"default": 2.6, "min": 1.4, "max": 5.0, "desc": "target centre spacing of piers / arch bays (m)"},
    "arches": {"default": True, "desc": "blind arcade of relieving arches (needs ~1.5 m of height), else buttress piers"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "amount of moss / damp growth"},
    "grime": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of soot, damp staining and run-off streaks"},
    "batter": {"default": 0.05, "min": 0.0, "max": 0.08, "desc": "how far the face leans out at the foot (m)"},
    "depth": {"default": 0.55, "min": 0.45, "max": 0.7, "desc": "wall thickness towards the earth side +Y (m)"},
    "ends": {"default": "half", "choices": ["half", "full"], "desc": "half piers on the end planes (runs butt into whole piers) or full end piers"},
    "spout_every": {"default": 2, "min": 0, "max": 4, "desc": "a drain spout in every Nth bay (0 = none)"},
    "tie_plates": {"default": True, "desc": "forged iron tie-rod anchor plates on the piers"},
    "lanterns": {"default": True, "desc": "small iron wall lanterns in the bays that have no spout"},
    "lantern_cd": {"default": 8.0, "min": 3.0, "max": 45.0, "desc": "intensity per wall lantern (cd)"},
    "inscription": {"default": "ANNO 1786", "desc": "lead-filled date on a tablet in the blocking course ('' = none)"},
    "stone": {"default": "#c6b9a1", "desc": "tint of the dressed stone"},
}
VARIANTS = {
    "buttress_rubble": {"length": 10.0, "height": 3.2, "facing": "rubble", "arches": False, "pier_spacing": 3.2,
                        "moss": 0.6, "ends": "full", "inscription": "", "stone": "#b7ab97", "seed": 4},
    "brick_arcade": {"length": 9.0, "height": 2.9, "facing": "brick", "pier_spacing": 2.2, "moss": 0.2, "grime": 0.7,
                     "ends": "full", "stone": "#cdc3b0", "seed": 11},
    "low_garden": {"length": 6.0, "height": 0.95, "arches": False, "pier_spacing": 2.0, "moss": 0.85, "batter": 0.02,
                   "spout_every": 1, "stone": "#bdb39c", "seed": 23},
    "mossy_buttress": {"length": 12.0, "height": 1.9, "arches": False, "pier_spacing": 3.0, "moss": 1.0, "grime": 0.75,
                       "spout_every": 2, "stone": "#a9a18f", "seed": 31},
}


def _shade(h, f, warm=0.0):
    h = h.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    r, g, b = min(1.0, r * f * (1 + warm)), min(1.0, g * f), min(1.0, b * f * (1 - warm))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


class _Acc:
    """Accumulates many small patches into one mesh (one object per material)."""

    def __init__(self):
        self.v, self.f, self.t = [], [], []

    def add(self, verts, faces, tone=0.0):
        b = len(self.v)
        self.v.extend(verts)
        self.f.extend(tuple(i + b for i in f) for f in faces)
        self.t.extend([tone] * len(verts))

    def obj(self, name, material, sm=35.0):
        if not self.f:
            return None
        o = mesh_object(name, self.v, self.f)
        assign(o, material)
        uv_box(o)
        smooth(o, sm)
        return o


def _patch(acc, f, nu, nv, yf, su=1.0, sv=1.0, proud=0.0, ch=0.007, rd=0.007, tilt=(0.0, 0.0), bulge=0.0,
           ph=(0.0, 0.0), soffit=None, tone=0.0):
    """A stone face: grid f(u, v) -> (x, z) facing -Y at depth yf(z) - proud, with a chamfered rim
    going back to yf + rd (ch = 0: plain grid). soffit = y to which the v = 0 edge is returned."""
    nu, nv = max(1, int(nu)), max(1, int(nv))
    eu = min(0.4, ch / max(su, 1e-3)) if ch > 0 else 0.0
    ev = min(0.4, ch / max(sv, 1e-3)) if ch > 0 else 0.0
    V, F = [], []
    us = [eu + (1 - 2 * eu) * j / nu for j in range(nu + 1)]
    vs = [ev + (1 - 2 * ev) * k / nv for k in range(nv + 1)]
    for j in range(nu + 1):
        for k in range(nv + 1):
            u, v = us[j], vs[k]
            x, z = f(u, v)
            y = yf(z) - proud + tilt[0] * (u - 0.5) + tilt[1] * (v - 0.5)
            if bulge:
                b = (max(sin(pi * (j / nu)), 0.0) * max(sin(pi * (k / nv)), 0.0)) ** 0.45 if (0 < j < nu and 0 < k < nv) else 0.0
                y -= bulge * (b * (0.7 + 0.3 * sin(7 * u + ph[0]) * sin(6 * v + ph[1])) +
                              0.25 * sin(11 * u + ph[1]) * sin(9 * v + ph[0]))
            V.append((x, y, z))
    idx = lambda j, k: j * (nv + 1) + k
    for j in range(nu):
        for k in range(nv):
            F.append((idx(j, k), idx(j + 1, k), idx(j + 1, k + 1), idx(j, k + 1)))
    if ch > 0:
        B = [(j, 0) for j in range(nu)] + [(nu, k) for k in range(nv)] + [(j, nv) for j in range(nu, 0, -1)] + \
            [(0, k) for k in range(nv, 0, -1)]
        r0 = len(V)
        for (j, k) in B:
            u = 0.0 if j == 0 else 1.0 if j == nu else us[j]
            v = 0.0 if k == 0 else 1.0 if k == nv else vs[k]
            x, z = f(u, v)
            V.append((x, yf(z) + rd, z))
        m = len(B)
        for i in range(m):
            a, b = B[i], B[(i + 1) % m]
            F.append((r0 + i, r0 + (i + 1) % m, idx(*b), idx(*a)))
        if soffit is not None:
            s0 = len(V)
            for i in range(nu + 1):
                x, _, z = V[r0 + i]
                V.append((x, soffit, z))
            for i in range(nu):
                F.append((s0 + i, s0 + i + 1, r0 + i + 1, r0 + i))
    acc.add(V, F, tone)


def _rect_f(x0, x1, z0, z1, lo=None, hi=None):
    def f(u, v):
        x = x0 + (x1 - x0) * u
        a = z0 if lo is None else max(z0, lo(x))
        b = z1 if hi is None else min(z1, hi(x))
        if b < a + 0.004:
            b = a + 0.004
        return x, a + (b - a) * v
    return f


def _intervals(x0, x1, za, zb, lo, hi, hmin):
    if lo is None and hi is None:
        return [(x0, x1)]
    n = max(2, int((x1 - x0) / 0.02))
    res, start, prev = [], None, x0
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        a = max(za, lo(x)) if lo else za
        b = min(zb, hi(x)) if hi else zb
        ok = (b - a) >= hmin
        if ok and start is None:
            start = x
        if not ok and start is not None:
            res.append((start, prev))
            start = None
        prev = x
    if start is not None:
        res.append((start, x1))
    return [(a, b) for a, b in res if b - a > 0.07]


def _courses(z0, z1, target, rr, var=0.1):
    n = max(1, int(round((z1 - z0) / target)))
    hs = [rr.u(1 - var, 1 + var) for _ in range(n)]
    s = sum(hs)
    zl = [z0]
    for h in hs:
        zl.append(zl[-1] + h * (z1 - z0) / s)
    zl[-1] = z1
    return zl


def _lay_blocks(pick, x0, x1, zl, rr, yf, lo=None, hi=None, lmin=1.4, lmax=2.7, cell=0.1, proud=(0.0, 0.006),
                bulge=0.0, rd=0.007, joint=0.007, hmin=0.05, tones=(0.0, 0.13), ch=0.007):
    """Coursed blocks with random lengths; lo(x) / hi(x) clip them against an arch."""
    j2 = joint * 0.5
    lo2 = (lambda x: lo(x) + j2) if lo else None
    hi2 = (lambda x: hi(x) - j2) if hi else None
    for c in range(len(zl) - 1):
        za, zb = zl[c], zl[c + 1]
        h = zb - za
        for (xa, xb) in _intervals(x0, x1, za, zb, lo, hi, hmin):
            cuts, x, first = [xa], xa, True
            while True:
                bl = h * rr.u(lmin, lmax) * (rr.u(0.45, 1.0) if first else 1.0)
                first = False
                if x + bl > xb - 0.6 * h * lmin:
                    break
                x += bl
                cuts.append(x)
            cuts.append(xb)
            for i in range(len(cuts) - 1):
                a, b = cuts[i] + j2, cuts[i + 1] - j2
                if b - a < 0.03:
                    continue
                f = _rect_f(a, b, za + j2, zb - j2, lo2, hi2)
                _patch(pick(rr), f, ceil((b - a) / cell), ceil(h / cell), yf, b - a, h, proud=rr.u(*proud),
                       rd=rd, ch=ch, tilt=(rr.jitter(0, 0.003), rr.jitter(0, 0.003)), bulge=bulge * rr.u(0.6, 1.2),
                       ph=(rr.u(0, 6.3), rr.u(0, 6.3)), tone=rr.u(*tones))


def _prism_x(name, prof, x0, x1, bev=0.005, nx=1):
    """Extrude a (y, z) section along X between x0 and x1 (end caps included)."""
    n = len(prof)
    area = sum(prof[i][0] * prof[(i + 1) % n][1] - prof[(i + 1) % n][0] * prof[i][1] for i in range(n))
    if area < 0:
        prof = prof[::-1]
    V, F = [], []
    for r in range(nx + 1):
        x = x0 + (x1 - x0) * r / nx
        V.extend((x, y, z) for (y, z) in prof)
    for r in range(nx):
        a, b = r * n, (r + 1) * n
        F.extend((a + i, a + (i + 1) % n, b + (i + 1) % n, b + i) for i in range(n))
    F.append(tuple(range(n - 1, -1, -1)))
    F.append(tuple(range(nx * n, nx * n + n)))
    o = mesh_object(name, V, F)
    if bev > 0:
        bevel(o, bev, segments=2)
    uv_box(o)
    smooth(o, 35)
    return o


def _cuts(x0, x1, lmin, lmax, rr):
    xs, x = [x0], x0
    while True:
        s = rr.u(lmin, lmax)
        if x + s > x1 - 0.6 * lmin:
            break
        x += s
        xs.append(x)
    xs.append(x1)
    return xs


def build(p, rng):
    L, H = float(p.length), float(p.height)
    X0, X1 = -L / 2, L / 2
    D = float(p.depth)
    bat = min(float(p.batter), 0.03 * H + 0.01)
    moss_k, grime_k = float(p.moss), float(p.grime)
    facing = p.facing

    # ---- vertical zoning
    hc = clamp(0.10 + 0.022 * H, 0.11, 0.17)          # coping
    zt = H - hc                                        # coping bed
    string = H >= 1.15
    hs = 0.11 if string else 0.0                       # string course
    hb = clamp(0.1 * H, 0.18, 0.28) if string else 0.0  # blocking course
    zs = zt - hb - hs                                  # top of the main face
    hp = clamp(0.15 * H, 0.13, 0.4)                    # plinth (incl. cap)
    capH = min(0.07, 0.4 * hp)
    hf = zs - hp
    pp = 0.045                                         # plinth projection
    pw = clamp(0.36 + 0.08 * H, 0.4, 0.62)             # pier width
    ang = degrees(atan2(bat, zt))
    cell = clamp(sqrt(L * H / 2600.0), 0.07, 0.13)

    def yf(z):
        return -bat * (1.0 - min(max(z, 0.0), zt) / zt)

    # ---- bays
    e = pw / 2 if p.ends == "full" else 0.0
    span = L - 2 * e
    n = max(1, int(round(span / float(p.pier_spacing))))
    while n > 1 and span / n - pw < 0.75:
        n -= 1
    bay = span / n
    xs = [X0 + e + i * bay for i in range(n + 1)]
    cxs = [(xs[i] + xs[i + 1]) / 2 for i in range(n)]
    wa = bay - pw

    # ---- arcade geometry (segmental / round relieving arches)
    arc = bool(p.arches) and hf >= 0.95 and wa >= 0.5
    rec = min(0.17, D - 0.3)
    if arc:
        t_nom = clamp(0.11 * wa, 0.18, 0.27)
        zci = zs - 0.08 - t_nom                        # intrados crown
        jamb_min = max(0.3, 0.47 * (zci - hp))
        rise = min(wa / 2, zci - hp - jamb_min)
        if rise < 0.13 * wa:
            rise = 0.13 * wa
        if zci - hp - rise < 0.2:
            arc = False
    if arc:
        z_sp = zci - rise
        R = (wa * wa / 4 + rise * rise) / (2 * rise)
        zc = zci - R
        a_half = asin(min(1.0, (wa / 2) / R))
        t = min(t_nom, sqrt((wa / 2 + pw / 2 - 0.012) ** 2 + (z_sp - zc) ** 2) - R)
        xe = sqrt(max((R + t) ** 2 - (z_sp - zc) ** 2, 0.0))
        imp_h = 0.11

    # ---- materials
    stA = mat("stone_smooth", p.stone)
    stB = mat("stone_smooth", _shade(p.stone, 0.86, 0.035))
    rgP = mat("stone_rough", _shade(p.stone, 0.7))
    mort = mat("mortar", "#cdc2ab")
    rub = mat("brick_red") if facing == "brick" else mat("rubble_fieldstone")
    lead = mat("lead")
    iron = mat("iron_wrought")
    mossm = mat("moss", "#7c9450")
    panel_mat = rub
    textured = facing != "ashlar"
    rd = 0.03 if textured else 0.007                   # how far dressed blocks' rims go back
    poff = 0.012 if facing == "rubble" else 0.006      # textured panel sits this far behind the face plane

    accA, accB, accR1, accR2, accM, accT, accK = _Acc(), _Acc(), _Acc(), _Acc(), _Acc(), _Acc(), _Acc()
    pickS = lambda r: accA if r.chance(0.55) else accB
    pickR = lambda r: accR1
    stones = []                                        # every stone object, for the weathering pass
    streaks = []                                       # (x, z_top, width, grime, moss) run-off stains
    clumps = []                                        # moss cushion matrices
    soots = []                                         # (x, z) lamp soot above the lanterns
    rr = rng.fork("masonry")

    dq = clamp((n + 1) * H / 13.0, 1.0, 2.6)          # many tall piers: coarser stone tessellation keeps the budget
    cell *= min(dq, 1.5)

    def block(name, size, loc, material, rgn, rot=None, chamfer=0.012, rough=0.003, detail=0.11):
        o = stone_block(name, size, loc=loc, rot=rot, chamfer=chamfer, rough=rough, rng=rgn, detail=detail * dq)
        assign(o, material)
        stones.append(o)
        return o

    def pier_stack(cx, w, z0, z1, pj, yback, rgn, split_ok=True):
        """Pier / buttress stage: one or two dressed stones per course, front at yf - pj."""
        if z1 - z0 < 0.06:
            return
        zl = _courses(z0, z1, 0.3, rgn, 0.08)
        xa, xb = max(X0 + 0.003, cx - w / 2), min(X1 - 0.003, cx + w / 2)
        for c in range(len(zl) - 1):
            h = zl[c + 1] - zl[c]
            zm = (zl[c] + zl[c + 1]) / 2
            parts = [(xa, xb)]
            if split_ok and (xb - xa) > 0.46 and c % 2 == 1:
                xm = xa + (xb - xa) * rgn.u(0.36, 0.64)
                parts = [(xa, xm), (xm, xb)]
            for (a, b) in parts:
                yfr = yf(zm) - pj - rgn.u(0.0, 0.005)
                d = yback - yfr
                block("pier", (b - a - 0.008, d, h - 0.008), ((a + b) / 2, yfr + d / 2, zm),
                      stA if rgn.chance(0.6) else stB, rgn, rot=(-ang, 0, 0))

    # =========================================================== main face
    if arc:
        ra = rng.fork("arcade")
        for i in range(n):
            cx = cxs[i]
            xl, xr = cx - wa / 2, cx + wa / 2
            intr = lambda x, cx=cx: zc + sqrt(max(R * R - (x - cx) ** 2, 0.0))
            # recessed bay: rock-faced coursed walling (or rubble / brick panel)
            if not textured:
                _lay_blocks(pickR, xl - 0.01, xr + 0.01, _courses(hp + 0.015, zci + 0.02, 0.21, ra, 0.16), ra,
                            lambda z: rec, hi=lambda x: intr(x) + 0.02, lmin=1.3, lmax=2.6, cell=cell,
                            proud=(0.0, 0.012), bulge=0.014, rd=0.014, ch=0.01, joint=0.012, tones=(0.0, 0.2))
                _patch(accM, _rect_f(xl - 0.03, xr + 0.03, hp - 0.02, zci + 0.04, hi=lambda x: intr(x) + 0.04),
                       ceil(wa / 0.2), ceil((zci - hp) / 0.3), lambda z: rec + 0.011, ch=0)
            else:
                _patch(accT, _rect_f(xl - 0.03, xr + 0.03, hp - 0.02, zci + 0.04, hi=lambda x: intr(x) + 0.04),
                       ceil(wa / cell), ceil((zci - hp) / cell), lambda z: rec + poff, ch=0)
            # voussoir ring with a proud keystone, springers bedded flat on the imposts
            nv_ = max(5, int(round(2 * a_half * (R + t / 2) / 0.23)))
            if nv_ % 2 == 0:
                nv_ += 1
            phis = [-a_half + 2 * a_half * k / nv_ for k in range(nv_ + 1)]
            bed = lambda rho: acos(clamp((z_sp + 0.004 - zc) / rho, -1.0, 1.0))
            arcw = 2 * a_half / nv_ * (R + t / 2)
            for k in range(nv_):
                key = (k == nv_ // 2)
                R1 = R + t - 0.005 + (0.055 if key else 0.0)
                fa = (lambda rho: -bed(rho)) if k == 0 else (lambda rho, q=phis[k]: q + 0.0035 / rho)
                fb = (lambda rho: bed(rho)) if k == nv_ - 1 else (lambda rho, q=phis[k + 1]: q - 0.0035 / rho)

                def vf(u, v, fa=fa, fb=fb, R1=R1, cx=cx):
                    rho = R + (R1 - R) * v
                    q = fa(rho) + (fb(rho) - fa(rho)) * u
                    return cx + rho * sin(q), zc + rho * cos(q)
                _patch(accA if (key or k % 2 == 0) else accB, vf, max(2, ceil(arcw / cell)), max(2, ceil(t / cell)), yf,
                       arcw, t, proud=(0.03 if key else 0.008 + ra.u(0, 0.004)), rd=rd, soffit=rec + 0.02,
                       tilt=(ra.jitter(0, 0.002), 0.0), tone=ra.u(0.0, 0.1))
            # mortar soffit liner behind the voussoir joints
            m = max(8, ceil(2 * a_half * R / 0.12))
            V, F = [], []
            for k in range(m + 1):
                q = -a_half + 2 * a_half * k / m
                x, z = cx + (R + 0.007) * sin(q), zc + (R + 0.007) * cos(q)
                V += [(x, yf(z) + 0.004, z), (x, rec + 0.03, z)]
            for k in range(m):
                F.append((2 * k + 1, 2 * k + 3, 2 * k + 2, 2 * k))
            accM.add(V, F)
            # sloped sill on the plinth inside the bay
            yfh = yf(hp)
            for sa, sb in zip(*(lambda c: (c[:-1], c[1:]))(_cuts(xl + 0.004, xr - 0.004, 0.7, 1.2, ra))):
                o = _prism_x("sill", [(yfh - 0.006, hp - 0.03), (rec + 0.05, hp - 0.03), (rec + 0.05, hp + 0.032),
                                      (yfh - 0.006, hp + 0.004)], sa + 0.003, sb - 0.003, bev=0.004, nx=max(1, ceil((sb - sa) / 0.25)))
                assign(o, stA)
                stones.append(o)

        def lo_sp(x):
            i = int(clamp((x - xs[0]) / bay, 0, n - 1e-6))
            dx = abs(x - cxs[i])
            return zc + sqrt((R + t) ** 2 - dx * dx) if dx < xe else z_sp

        def lo_back(x):
            i = int(clamp((x - xs[0]) / bay, 0, n - 1e-6))
            dx = abs(x - cxs[i])
            return zc + sqrt((R + 0.03) ** 2 - dx * dx) if dx < wa / 2 else z_sp - 0.02

        fsp = lambda u, v: (X0 + L * u, lerp(lo_back(X0 + L * u), zs + 0.03, v))
        if not textured:
            _lay_blocks(pickS, X0, X1, _courses(z_sp, zs, 0.28, rr), rr, yf, lo=lo_sp, cell=cell)
            _patch(accM, fsp, ceil(L / 0.12), 3, lambda z: yf(z) + 0.005, ch=0)
        else:
            _patch(accT, fsp, ceil(L / cell), ceil((zs - z_sp) / cell), lambda z: yf(z) + poff, ch=0)
        # piers with moulded imposts
        for i, cx in enumerate(xs):
            rp = rng.fork("pier%d" % i)
            pier_stack(cx, pw, hp, z_sp - imp_h, 0.004, rec + 0.06, rp)
            for (ex, pj, za_, zb_, chf) in ((0.014, 0.016, z_sp - imp_h, z_sp - 0.066, 0.012), (0.034, 0.038, z_sp - 0.066, z_sp, 0.016)):
                xa, xb = max(X0 + 0.003, cx - pw / 2 - ex), min(X1 - 0.003, cx + pw / 2 + ex)
                yfr = yf(z_sp) - pj
                block("impost", (xb - xa, rec + 0.06 - yfr, zb_ - za_ - 0.004), ((xa + xb) / 2, (yfr + rec + 0.06) / 2, (za_ + zb_) / 2),
                      stA, rp, chamfer=chf, rough=0.0015, detail=0.1)
    else:
        if not textured:
            _lay_blocks(pickS, X0, X1, _courses(hp, zs, 0.29, rr), rr, yf, cell=cell)
        else:
            _patch(accT, _rect_f(X0, X1, hp - 0.03, zs + 0.03), ceil(L / cell), ceil(hf / cell), lambda z: yf(z) + poff, ch=0)

    # mortar backing of the face plane (also closes the wall behind the plinth stones)
    zb0 = 0.0
    zb1 = zt + 0.01
    if arc:
        _patch(accM, _rect_f(X0, X1, 0.0, hp + 0.02), ceil(L / 0.5), 1, lambda z: yf(z) + 0.03, ch=0)
        zb0 = zs - 0.02
    if (not textured) or arc:
        if zb1 - zb0 > 0.05 and not (arc and not string):
            _patch(accM, _rect_f(X0, X1, zb0, zb1), ceil(L / 0.5), max(1, ceil((zb1 - zb0) / 0.4)), lambda z: yf(z) + 0.005, ch=0)
    else:
        _patch(accM, _rect_f(X0, X1, 0.0, hp), ceil(L / 0.5), 1, lambda z: yf(z) + 0.03, ch=0)
        if string:
            _patch(accM, _rect_f(X0, X1, zs - 0.02, zb1), ceil(L / 0.5), 1, lambda z: yf(z) + 0.005, ch=0)

    # blocking course above the string
    if string:
        _lay_blocks(pickS, X0, X1, [zs + hs, zt], rr, yf, lmin=2.4, lmax=3.8, cell=cell)

    # =========================================================== buttress piers
    plate_pts = []
    lantern_pts = []
    if not arc:
        pj1 = max(0.03, 0.2 - bat - pp - 0.012)
        pj2 = pj1 * 0.5
        ztop = (zs - 0.04) if string else (zt - 0.03)
        two = hf >= 1.0
        so = clamp(0.1 * hf, 0.08, 0.16)               # set-off (weathering) height
        z_a = hp + (0.56 * hf if two else (ztop - hp - so))
        for i, cx in enumerate(xs):
            rp = rng.fork("pier%d" % i)
            xa, xb = max(X0 + 0.003, cx - pw / 2), min(X1 - 0.003, cx + pw / 2)
            if xb - xa < 0.08:
                continue
            # plinth breaks forward round the buttress
            xpa, xpb = max(X0 + 0.009, cx - pw / 2 - 0.035), min(X1 - 0.009, cx + pw / 2 + 0.035)
            ypf = yf(0) - pj1 - pp
            block("pier_plinth", (xpb - xpa, pp + pj1 + 0.08, hp - capH - 0.004), ((xpa + xpb) / 2, ypf + (pp + pj1 + 0.08) / 2, (hp - capH) / 2 + 0.004),
                  rgP, rp, chamfer=0.016, rough=0.006, detail=0.12)
            yph = yf(hp) - pj1
            o = _prism_x("pier_cap", [(ypf - 0.006, hp - capH), (yph + 0.08, hp - capH), (yph + 0.08, hp + 0.003), (yph - 0.004, hp + 0.003),
                                      (ypf - 0.006, hp - capH * 0.45)], xpa - 0.004, xpb + 0.004, bev=0.004)
            if xpa <= X0 + 0.01 or xpb >= X1 - 0.01:
                transform_verts(o, lambda co: Vector((clamp(co.x, X0 + 0.002, X1 - 0.002), co.y, co.z)))
            assign(o, stA)
            stones.append(o)
            # lower stage, set-off, upper stage, top weathering
            pier_stack(cx, pw, hp + 0.004, z_a, pj1, yf(hp) + 0.07, rp)
            stages = [(z_a, pj1, pj2 if two else 0.0)]
            if two:
                z_b = ztop - so
                pier_stack(cx, pw - 0.02, z_a + so, z_b, pj2, yf(z_a) + 0.07, rp, split_ok=False)
                stages.append((z_b, pj2, 0.0))
                if xb - xa > 0.36:
                    plate_pts.append((cx, yf((z_a + so + z_b) / 2) - pj2 - 0.004, (z_a + so + z_b) / 2 + 0.05))
            elif xb - xa > 0.36:
                plate_pts.append((cx, yf((hp + z_a) / 2) - pj1 - 0.004, hp + 0.62 * (z_a - hp)))
            for (z0, pa, pb) in stages:
                y0, y1 = yf(z0), yf(z0 + so)
                o = _prism_x("setoff", [(y0 - pa - 0.012, z0 + 0.002), (y0 + 0.05, z0 + 0.002), (y1 + 0.05, z0 + so),
                                        (y1 - pb - 0.002, z0 + so), (y0 - pa - 0.012, z0 + 0.045)], xa - 0.006, xb + 0.006, bev=0.005)
                if xa <= X0 + 0.004 or xb >= X1 - 0.004:
                    transform_verts(o, lambda co: Vector((clamp(co.x, X0 + 0.002, X1 - 0.002), co.y, co.z)))
                assign(o, stA)
                stones.append(o)
                for _ in range(int(2 * moss_k + rp.u(0, 1))):
                    tt = rp.u(0.2, 0.9)
                    clumps.append(trs((rp.u(xa + 0.04, xb - 0.04), lerp(y0 - pa, y1 - pb, tt) - 0.004, lerp(z0 + 0.045, z0 + so, tt) - 0.004),
                                      (-50, 0, rp.u(0, 360)), (rp.u(0.02, 0.045), rp.u(0.02, 0.04), rp.u(0.01, 0.02))))
            if H > 1.9:
                lantern_pts.append((cx, yf(min(H - 0.6, 2.2)) - pj2, min(H - 0.6, 2.2)))
    else:
        for i, cx in enumerate(xs):
            xa, xb = max(X0 + 0.003, cx - pw / 2), min(X1 - 0.003, cx + pw / 2)
            zpl = z_sp + 0.58 * (zs - z_sp)
            if zs - z_sp > 0.5 and (xb - xa) > 0.36:
                plate_pts.append(((xa + xb) / 2, yf(zpl) - 0.004, zpl))
            if H > 1.9 and (xb - xa) > 0.3:
                lantern_pts.append(((xa + xb) / 2, yf(hp + 0.5 * (z_sp - hp)), z_sp - imp_h - 0.12))

    # =========================================================== plinth: rock-faced course + weathered cap
    rpl = rng.fork("plinth")
    yp = yf(0) - pp
    pc = _cuts(X0 + 0.003, X1 - 0.003, 0.6, 1.15, rpl)
    for a, b in zip(pc[:-1], pc[1:]):
        dd = pp + 0.11 + rpl.u(0, 0.006)
        block("plinth", (b - a - 0.008, dd, hp - capH - 0.004), ((a + b) / 2, yp - rpl.u(0, 0.006) + dd / 2, (hp - capH) / 2 + 0.004),
              rgP, rpl, chamfer=0.018, rough=0.007, detail=0.12)
    yph = yf(hp)
    cc = _cuts(X0 + 0.002, X1 - 0.002, 0.8, 1.4, rpl)
    for a, b in zip(cc[:-1], cc[1:]):
        dy = rpl.jitter(0, 0.002)
        o = _prism_x("plinth_cap", [(yp - 0.008 + dy, hp - capH), (yph + 0.06, hp - capH), (yph + 0.06, hp + 0.004), (yph - 0.003, hp + 0.004),
                                    (yp - 0.008 + dy, hp - capH * 0.42)], a + 0.0025, b - 0.0025, bev=0.004, nx=max(1, ceil((b - a) / 0.3)))
        assign(o, stA if rpl.chance(0.6) else stB)
        stones.append(o)

    # =========================================================== string course
    if string:
        rs = rng.fork("string")
        y0 = yf(zs)
        sc = _cuts(X0 + 0.002, X1 - 0.002, 0.9, 1.5, rs)
        for a, b in zip(sc[:-1], sc[1:]):
            dy = rs.jitter(0, 0.002)
            prof = [(y0 + 0.07, zs), (y0 + 0.07, zs + hs), (y0 - 0.004 + dy, zs + hs), (y0 - 0.044 + dy, zs + hs - 0.03),
                    (y0 - 0.052 + dy, zs + hs - 0.036), (y0 - 0.052 + dy, zs + 0.046), (y0 - 0.044 + dy, zs + 0.04),
                    (y0 - 0.036 + dy, zs + 0.024), (y0 - 0.02 + dy, zs + 0.012), (y0 - 0.014 + dy, zs)]
            o = _prism_x("string", prof, a + 0.0025, b - 0.0025, bev=0.003, nx=max(1, ceil((b - a) / 0.3)))
            assign(o, stA if rs.chance(0.65) else stB)
            stones.append(o)

    # =========================================================== coping (flat top exactly at z = H)
    rc = rng.fork("coping")
    ov = 0.07
    ff = 0.56 * hc
    kc = _cuts(X0, X1, 0.8, 1.25, rc)
    ny = 5
    for a, b in zip(kc[:-1], kc[1:]):
        dy = rc.jitter(0, 0.0025)
        top = [(D - (D + ov - dy) * k / ny, H) for k in range(ny + 1)]
        prof = [(D, zt)] + top + [(-ov + dy, H - ff), (-ov + 0.016 + dy, H - ff - 0.02), (-ov + 0.022 + dy, zt),
                                   (-ov + 0.028, zt), (-ov + 0.031, zt + 0.009), (-ov + 0.041, zt + 0.009), (-ov + 0.044, zt)]
        xa = a + (0.003 if a > X0 + 1e-6 else 0.0)
        xb = b - (0.003 if b < X1 - 1e-6 else 0.0)
        o = _prism_x("coping", prof, xa, xb, bev=0.006, nx=max(1, ceil((b - a) / (0.22 * dq))))
        assign(o, stA if rc.chance(0.7) else stB)
        stones.append(o)
    core = _prism_x("coping_bed", [(D - 0.006, zt + 0.002), (D - 0.006, H - 0.006), (-ov + 0.008, H - 0.006), (-ov + 0.008, H - ff + 0.004),
                                   (-ov + 0.03, zt + 0.014), (0.0, zt + 0.014), (0.0, zt + 0.002)], X0 + 0.002, X1 - 0.002, bev=0.0)
    assign(core, mort)

    # =========================================================== body: earth-side face and end faces (rubble backing)
    ny_, nz_ = 3, max(2, ceil(zt / 0.3))
    nxb = max(2, ceil(L / 0.3))
    V, F = [], []
    for j in range(nxb + 1):
        for k in range(nz_ + 1):
            V.append((X0 + L * j / nxb, D - 0.02, zt * k / nz_))
    ib = lambda j, k: j * (nz_ + 1) + k
    for j in range(nxb):
        for k in range(nz_):
            F.append((ib(j, k), ib(j, k + 1), ib(j + 1, k + 1), ib(j + 1, k)))
    accK.add(V, F)
    for sx, xe_ in ((-1, X0), (1, X1)):
        V, F = [], []
        for j in range(ny_ + 1):
            for k in range(nz_ + 1):
                z = zt * k / nz_
                V.append((xe_, lerp(yf(z) + 0.009, D - 0.02, j / ny_), z))
        for j in range(ny_):
            for k in range(nz_):
                q = (ib(j, k), ib(j, k + 1), ib(j + 1, k + 1), ib(j + 1, k))
                F.append(q if sx < 0 else q[::-1])
        accK.add(V, F)

    # =========================================================== drain spouts
    spouts = []
    ev = int(p.spout_every)
    # bays alternate outwards from the middle one: lantern, spout, lantern ... (a single bay gets the spout)
    spout_bays = [i for i in range(n) if ev > 0 and (n == 1 or (i - n // 2) % ev == ev // 2)]
    if ev > 0:
        for i in spout_bays:
            cx = cxs[i] + rng.fork("sp%d" % i).jitter(0, 0.06 if arc else 0.25 * wa)
            if arc:
                zsp_, yface = hp + 0.62 * (zci - hp), rec
            else:
                zsp_ = hp + 0.62 * hf
                yface = yf(zsp_)
            spouts.append((cx, yface, zsp_))
    rsp = rng.fork("spout")
    for k, (cx, yface, z) in enumerate(spouts):
        bh = min(0.22, hf * 0.5)
        block("spout_stone", (0.27, 0.1, bh), (cx, yface - 0.028 + 0.05, z), stA, rsp, chamfer=0.016, rough=0.002, detail=0.07)
        ln = clamp((yface - 0.028 + 0.192) / 0.985, 0.07, 0.24)
        zp_ = z - 0.02
        pipe = lathe("spout_pipe", [(0.037, 0.0), (0.037, ln - 0.014), (0.043, ln - 0.01), (0.043, ln), (0.031, ln),
                                    (0.031, 0.025), (0.0, 0.025)], segments=16, loc=(cx, yface - 0.028, zp_), rot=(100, 0, 0))
        assign(pipe, lead)
        collar = lathe("spout_collar", [(0.037, 0.0), (0.062, 0.0), (0.064, 0.004), (0.056, 0.008), (0.045, 0.011), (0.037, 0.012)],
                       segments=16, loc=(cx, yface - 0.028, zp_), rot=(100, 0, 0))
        assign(collar, lead)
        tip = (cx, yface - 0.028 - ln * 0.985, zp_ - ln * 0.174 - 0.04)
        fx("drip%d" % k, "drip", tip, rate=0.4, size=0.012)
        streaks.append((cx, z - 0.06, 0.055, 1.0, 1.0))
        rm = rng.fork("spmoss%d" % k)
        for _ in range(int(1 + 7 * moss_k)):
            dz = rm.u(0.12, max(0.2, z - hp)) ** 1.0
            zz = z - dz
            if zz < hp + 0.03:
                continue
            ypl = (rec if arc else yf(zz)) - 0.008
            s = rm.u(0.016, 0.04)
            clumps.append(trs((cx + rm.jitter(0, 0.03 + 0.05 * dz), ypl, zz), (90, 0, rm.u(-30, 30)), (s, s * rm.u(1.0, 2.0), s * 0.3)))

    # =========================================================== wall lanterns in the bays without a spout
    lamp_bays = [i for i in range(n) if i not in spout_bays and (i - n // 2) % 2 == 0][:6] if (p.lanterns and zs >= 1.45) else []
    if lamp_bays:
        glass = mat("glass_lantern", glow=0.7)
        flame_m = mat("emissive", "#ffb45e", glow=4.5)
        iparts, gparts, fparts = [], [], []
        for k, i in enumerate(lamp_bays):
            x = cxs[i]
            if arc:                                    # the recessed bays leave room for a longer arm
                z = hp + 0.86 * (zci - hp)
                wy = rec - 0.03
                q, rch = 1.0, 0.17
            else:                                      # flat face: the lantern shrinks to stay inside the 0.2 m projection
                z = clamp(0.72 * zs, 1.0, 2.2)
                wy = yf(z) - 0.008
                q = clamp((0.198 + wy - 0.012) / 0.2, 0.7, 1.0)
                rch = 0.1 * q + 0.012
            block("lamp_stone", (0.2, 0.09, 0.4), (x, wy + 0.045, z - 0.12), stA, rsp, chamfer=0.014, rough=0.002, detail=0.09)
            ya = wy - rch                              # lantern axis
            iparts.append(box("ln_plate", (0.066, 0.006, 0.3), loc=(x, wy - 0.003, z - 0.12), bevel=0.002))
            bt = bolt("ln_bolt", 0.0065, 0.004, kind="dome")
            iparts.append(instances_to_mesh("ln_bolts", bt, [trs((x + sx * 0.02, wy - 0.006, z - 0.12 + sz * 0.128), (90, 0, 0))
                                                             for sx in (-1, 1) for sz in (-1, 1)], keep_base=False))
            sq = rounded_rect(0.012, 0.012, 0.002, 2)
            iparts.append(sweep("ln_arm", [(x, wy - 0.004, z - 0.03), (x, wy - 0.04, z + 0.004), (x, ya, z + 0.002), (x, ya - 0.016, z - 0.012)],
                                sq, up=(1, 0, 0)))
            iparts.append(sweep("ln_stay", bezier_path((x, wy - 0.005, z - 0.2), (x, wy - 0.05, z - 0.19), (x, wy - 0.012, z - 0.06), (x, wy - rch * 0.66, z - 0.006), n=12),
                                rounded_rect(0.012, 0.004, 0.001, 2), up=(1, 0, 0)))
            iparts.append(torus("ln_ring", major=0.011, minor=0.0028, loc=(x, ya, z - 0.018), rot=(0, 90, 0), major_seg=12, minor_seg=6))
            zr = z - 0.03                              # roof apex
            zg1 = zr - 0.058 * q                       # top of the glazing
            zg0 = zg1 - 0.17 * q                       # bottom of the glazing
            iparts.append(lathe("ln_roof", [(0.0, zg1 - 0.002), (0.1 * q, zg1 - 0.002), (0.1 * q, zg1 + 0.005), (0.072 * q, zg1 + 0.02 * q), (0.032 * q, zr - 0.016),
                                            (0.016, zr - 0.008), (0.013, zr), (0.0, zr)], segments=4, loc=(x, ya, 0), smooth_angle=30))
            iparts.append(lathe("ln_tray", [(0.0, zg0 - 0.022), (0.012, zg0 - 0.02), (0.03, zg0 - 0.012), (0.066 * q, zg0 - 0.004), (0.066 * q, zg0 + 0.004), (0.0, zg0 + 0.004)],
                                segments=4, loc=(x, ya, 0), smooth_angle=30))
            iparts.append(lathe("ln_rim", [(0.079 * q, zg1 - 0.012), (0.087 * q, zg1 - 0.012), (0.087 * q, zg1 - 0.002), (0.079 * q, zg1 - 0.002)], segments=4,
                                loc=(x, ya, 0), smooth_angle=30))
            for sx in (-1, 1):
                for sy in (-1, 1):
                    iparts.append(sweep("ln_bar", [(x + sx * 0.062 * q, ya + sy * 0.062 * q, zg0), (x + sx * 0.083 * q, ya + sy * 0.083 * q, zg1)], 0.004, sides=6))
            iparts.append(cylinder("ln_burner", 0.007, 0.05 * q, loc=(x, ya, zg0 + 0.028 * q), verts=8))
            gparts.append(lathe("ln_glass", [(0.061 * q, zg0), (0.082 * q, zg1 - 0.004)], segments=4, loc=(x, ya, 0), smooth_angle=30))
            fparts.append(sphere("ln_flame", 0.013 * q, loc=(x, ya, zg0 + 0.075 * q), segments=10, rings=6, scale=(1, 1, 1.7)))
            # the light sits just in front of the cage so the stone right behind the lamp is not burnt out
            light("lantern%d" % k, (x, ya - 0.12, zg0 + 0.075 * q), color="#ffbf6a", intensity=float(p.lantern_cd), range=6.5, flicker=0.15, shadow_near=0.24)
            soots.append((x, z + 0.0))
        for o in iparts:
            assign(o, iron)
            paint(o, special=mask_mul(noise_mask(0.04, seed=p.seed + 6, lo=0.35, hi=0.75), 0.5))
        for o in gparts:
            assign(o, glass)
        for o in fparts:
            assign(o, flame_m)

    # =========================================================== date tablet with lead-filled lettering in the blocking course
    if string and str(p.inscription).strip() and L >= 3.0:
        zm = zs + hs + hb / 2
        th = hb - 0.05
        size = min(0.1, th * 0.5)
        yt = yf(zm) - 0.016
        tx = text("inscription", str(p.inscription).strip(), size=size, depth=0.003, loc=(0, yt, zm - size * 0.45), font="roman")
        bb = world_bbox([tx])
        tw = bb[1].x - bb[0].x
        dz = zm - (bb[0].z + bb[1].z) / 2
        if abs(dz) > 1e-4 or abs((bb[0].x + bb[1].x) / 2) > 1e-4:
            place(tx, loc=(-(bb[0].x + bb[1].x) / 2, yt, zm - size * 0.45 + dz))
        assign(tx, lead)
        wt = min(L - 0.4, tw + 0.22)
        block("tablet", (wt, 0.07, th), (0, yt + 0.035, zm), stA, rng.fork("tablet"), chamfer=0.012, rough=0.0015, detail=0.1)

    # =========================================================== tie-rod anchor plates (forged cross + square nut)
    if p.tie_plates and hf > 0.8 and plate_pts:
        bar = box("tp_bar", (0.3, 0.042, 0.011), loc=(0, 0, 0.0055), bevel=0.003)
        transform_verts(bar, lambda co: Vector((co.x, co.y * (1.0 - 0.45 * min(1.0, abs(co.x) / 0.15) ** 2), co.z)))
        bar2 = duplicate(bar)
        place(bar2, loc=(0, 0, 0.015), rot=(0, 0, 90))
        boss = lathe("tp_boss", [(0.0, 0.0), (0.042, 0.0), (0.042, 0.02), (0.035, 0.027), (0.0, 0.027)], segments=14)
        nut = bolt("tp_nut", radius=0.021, height=0.018, loc=(0, 0, 0.027), kind="square")
        rod = cylinder("tp_rod", 0.009, 0.026, loc=(0, 0, 0.052), verts=10)
        base = join([bar, bar2, boss, nut, rod], "tp_base")
        rt = rng.fork("ties")
        plates = instances_to_mesh("tie_plates", base, [trs(pt, (90, rt.pick([45, 45, 0]) + rt.jitter(0, 4), 0)) for pt in plate_pts], keep_base=False)
        assign(plates, iron)
        paint(plates, special=mask_mul(noise_mask(0.05, seed=p.seed + 5, lo=0.25, hi=0.7), 0.8), grime=0.3)
        for pt in plate_pts:
            streaks.append((pt[0], pt[2] - 0.03, 0.028, 0.3, 0.0))

    # =========================================================== moss cushions on the ledges and at the foot
    rmo = rng.fork("moss")
    for _ in range(int(L * 5 * moss_k)):
        x = rmo.u(X0 + 0.05, X1 - 0.05)
        s = rmo.u(0.018, 0.05)
        if rmo.chance(0.55):       # on the plinth cap slope
            tt = rmo.u(0.2, 1.0)
            clumps.append(trs((x, lerp(yp, yph, tt) + 0.004, lerp(hp - capH * 0.42, hp, tt) - 0.004), (-35, 0, rmo.u(0, 360)), (s, s * rmo.u(0.7, 1.3), s * 0.4)))
        elif string and rmo.chance(0.5):   # on the weathered top of the string course
            clumps.append(trs((x, yf(zs) - 0.02, zs + hs - 0.012), (-35, 0, rmo.u(0, 360)), (s * 0.8, s * 0.7, s * 0.35)))
        else:                      # along the foot of the wall
            clumps.append(trs((x, yp - rmo.u(0.0, 0.02), s * 0.3), (0, 0, rmo.u(0, 360)), (s * 1.3, s, s * 0.45)))
    if clumps:
        cb = sphere("moss_base", 1.0, segments=7, rings=4)
        displace_noise(cb, strength=0.18, scale=0.8, seed=p.seed)
        mo = instances_to_mesh("moss_clumps", cb, clumps[:int(260 / dq)], keep_base=False)
        transform_verts(mo, lambda co: Vector((clamp(co.x, X0, X1), co.y, max(co.z, 0.0))))
        assign(mo, mossm)
        smooth(mo, 80)

    # =========================================================== realise the accumulated faces
    made = []
    for acc, nm, m_ in ((accA, "ashlar_a", stA), (accB, "ashlar_b", stB), (accR1, "rock_a", rgP)):
        o = acc.obj(nm, m_)
        if o:
            made.append((o, np.array(acc.t)))
    om = accM.obj("mortar_bed", mort, 60)
    ot = accT.obj("panel", panel_mat, 60)
    if ot:
        displace_noise(ot, strength=0.012 if facing == "rubble" else 0.003, scale=0.16, seed=p.seed + 3, along_normal=False, axis_mask=(0, 1, 0))
    ok = accK.obj("backing", rub, 60)
    displace_noise(ok, strength=0.012, scale=0.25, seed=p.seed + 9, along_normal=False, axis_mask=(0, 1, 0))

    # =========================================================== weathering story
    sd = int(p.seed)
    n_big = noise_mask(0.9, seed=sd + 11, lo=0.3, hi=0.75)
    n_mid = noise_mask(0.28, seed=sd + 12, lo=0.35, hi=0.7)
    n_run = noise_mask(0.06, seed=sd + 13, lo=0.3, hi=0.75)
    n_moss = noise_mask(0.22, seed=sd + 14, lo=0.4 - 0.25 * moss_k, hi=0.8 - 0.25 * moss_k)
    n_damp = noise_mask(0.45, seed=sd + 15, lo=0.2, hi=0.8)
    gk = 0.4 + 0.8 * grime_k
    lowk = clamp(H / 2.4, 0.45, 1.0)                   # low walls: the damp foot zone shrinks with the wall

    def streak_fields(co):
        x, z = co[:, 0], co[:, 2]
        g = np.zeros(len(co))
        s = np.zeros(len(co))
        for (sx, sz, w0, gs, ms) in streaks:
            dz = sz - z
            w = w0 + 0.045 * np.clip(dz, 0, None)
            wob = 0.018 * np.sin(z * 8.0 + sx * 3.1)
            m = np.exp(-((x - sx - wob) / w) ** 2) * np.clip(1.0 - 0.45 * dz / max(sz, 0.3), 0.0, 1.0) * (dz > -0.03)
            g = np.maximum(g, gs * m)
            s = np.maximum(s, ms * m * m)
        return g, s

    def G(co, nrm):
        z = co[:, 2]
        run = n_run(co * np.array([1.0, 0.0, 0.07]), nrm)                    # vertical run-off streaks
        g = 0.4 * np.clip(1.0 - z / (0.5 * lowk), 0.0, 1.0) ** 1.4            # splash-back / rising damp
        g = np.maximum(g, 0.12 * n_big(co, nrm))                              # broad sooty patches
        if string:
            g = np.maximum(g, 0.2 * (0.35 + 0.65 * run) * np.clip(1.0 - (zs - z) / 0.4, 0.0, 1.0) ** 1.5 * (z < zs + 0.02))
        g = np.maximum(g, 0.17 * (0.35 + 0.65 * run) * np.clip(1.0 - (zt - z) / 0.25, 0.0, 1.0) * (z < zt + 0.01) * (z > zs + hs - 0.02))
        sg, _ = streak_fields(co)
        g = np.maximum(g, 0.55 * sg * (0.6 + 0.4 * run))
        for (sx, sz) in soots:                                                # lamp soot above the lanterns
            g = np.maximum(g, 0.55 * np.exp(-((co[:, 0] - sx) / 0.11) ** 2) * np.clip(1.0 - (z - sz) / 0.5, 0.0, 1.0) * (z > sz - 0.04))
        return np.clip(g * gk, 0.0, 1.0)

    def S(co, nrm):
        z = co[:, 2]
        up = np.clip(nrm[:, 2], 0.0, 1.0) ** 1.5
        s = up * n_moss(co, nrm) * (0.55 + 0.45 * (z < H - 0.01))             # ledges: plinth cap, set-offs, string, coping
        s = np.maximum(s, 0.8 * np.clip(1.0 - z / (lowk * (0.15 + 0.4 * n_damp(co, nrm))), 0.0, 1.0) * n_mid(co, nrm))   # green foot
        _, ss = streak_fields(co)
        s = np.maximum(s, 0.7 * ss)
        if arc:   # damp lower corners of the recessed bays
            s = np.maximum(s, 0.7 * (co[:, 1] > rec - 0.05) * np.clip(1.0 - (z - hp) / 0.45, 0.0, 1.0) * n_mid(co, nrm))
        return np.clip(s * (0.25 + 0.95 * moss_k) * (moss_k > 0), 0.0, 0.85)

    for o in stones:
        paint(o, grime=G, special=S)
    for o, tones in made:
        def Gt(co, nrm, tones=tones):
            g = G(co, nrm)
            return np.clip(g + tones, 0.0, 1.0) if len(tones) == len(g) else g
        paint(o, grime=Gt, special=S)
    if ot:
        paint(ot, grime=G, special=S)
    if om:
        paint(om, special=S)
    paint(ok, grime=mask_mul(height_ramp(zt, zt - 0.6), 0.5))

    # =========================================================== sockets: balustrade line on the coping, pier tops, lanterns
    yc = D / 2 - ov / 2
    socket("balustrade", "attach", (0, yc, H), role="balustrade", length=L)
    socket("balustrade_start", "attach", (X0, yc, H), role="balustrade_end")
    socket("balustrade_end", "attach", (X1, yc, H), role="balustrade_end")
    for i, cx in enumerate(xs):
        socket("pier_top_%d" % i, "attach", (clamp(cx, X0 + pw / 4, X1 - pw / 4), yc, H), role="prop_top")
    for i, pt in enumerate(lantern_pts):
        socket("lantern_%d" % i, "attach", pt, role="lantern")
