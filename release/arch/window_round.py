"""Round oculus window for gables and towers. The surround is built piece by piece: moulded stone voussoirs bedded in
lime mortar with wedge keystones (sunk panels, dominant crown key) and an optional hood mould with label stops;
pegged, moulded oak felloes under a lead drip; or gauged brick headers in staggered courses. Inside it sits a painted
fixed frame with a horizontally pivoting sash and radial / cross glazing bars, or a moulded quatrefoil cut from four
jointed stones with pierced spandrel lights. Glazing is plain (drawn curtain behind), leaded quarries or stained
glass leaded in small pieces. `diameter` is the clear wall opening; the origin is its centre on the wall plane.
"""
from math import asin, atan2, ceil, cos, degrees, hypot, pi, radians, sin, sqrt

import numpy as np

from tk import *

META = {"name": "Round oculus window", "class": "component", "mount": "wall", "mount_height": 2.3, "focus": 0.5}
PARAMS = {
    "diameter": {"default": 0.9, "min": 0.5, "max": 2.0, "desc": "clear diameter of the wall opening inside the surround (m)"},
    "bars": {"default": "radial", "choices": ["radial", "quatrefoil", "cross"], "desc": "glazing bar / tracery pattern"},
    "spokes": {"default": 8, "min": 5, "max": 16, "desc": "number of radial bars (bars = radial)"},
    "surround": {"default": "stone", "choices": ["stone", "timber", "brick"], "desc": "construction of the ring surround"},
    "surround_width": {"default": 0.17, "min": 0.1, "max": 0.32, "desc": "radial width of the surround ring (m)"},
    "projection": {"default": 0.12, "min": 0.08, "max": 0.2, "desc": "how far the surround stands proud of the wall (m)"},
    "keystones": {"default": 4, "choices": [0, 1, 2, 4], "desc": "key blocks: 1 = crown, 2 = crown + foot, 4 = all quarters"},
    "hood": {"default": False, "desc": "hood mould (stone label / lead drip on timber) over the upper half"},
    "glazing": {"default": "plain", "choices": ["plain", "leaded", "stained"], "desc": "plain panes, leaded diamond quarries or stained glass"},
    "lit": {"default": True, "desc": "room behind is lit (warm glass + spill light) or dark"},
    "frame_color": {"default": "#2e4638", "desc": "paint colour of the timber frame, sash and bars"},
    "stone_color": {"default": "#b9ad98", "desc": "tint of the dressed stone"},
    "glass_tint": {"default": "#ffc37a", "desc": "interior light colour"},
    "intensity": {"default": 2.4, "min": 0.0, "max": 30.0, "desc": "spill light through the glass (cd)"},
    "weathering": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of run-off staining, moss and paint wear"},
}
VARIANTS = {
    "quatrefoil_stained": {"bars": "quatrefoil", "glazing": "stained", "diameter": 1.2, "keystones": 0, "hood": True,
                           "surround_width": 0.2, "stone_color": "#c4b08c", "seed": 4},
    "timber_cross": {"surround": "timber", "bars": "cross", "glazing": "leaded", "diameter": 0.7, "keystones": 0, "hood": True,
                     "surround_width": 0.13, "frame_color": "#4a1c1b", "seed": 11},
    "brick_dark": {"surround": "brick", "spokes": 6, "diameter": 0.8, "keystones": 1, "lit": False, "surround_width": 0.215,
                   "frame_color": "#d8cfb8", "weathering": 0.8, "seed": 23},
    "tower_rose": {"diameter": 1.8, "spokes": 12, "glazing": "stained", "keystones": 4, "hood": True, "surround_width": 0.26,
                   "projection": 0.16, "frame_color": "#20232a", "intensity": 6.0, "seed": 31},
}

FD = 0.046      # front of the fixed frame (m proud of the wall plane)
SD = 0.040      # front of the pivoting sash
DG = 0.012      # glass plane
DB = DG - 0.0045  # back of the glazing bars: they run through the glass, so that no timber face lies on the lit glass plane
#                 CLEARANCE RULE: a glowing sheet keeps >= 4 mm to whatever lies parallel behind or in front of it
# pot-metal palette: deep ruby, cobalt, olive, amber, pale quarry + (draw calls permitting) a greener blue and a green-white
PAL = ("#7c1c22", "#2a469a", "#5c6a2c", "#c08a2a", "#e6dcb8", "#2f6a8c", "#c3cca4")


# ---------------------------------------------------------------------------------------------- geometry helpers
def _pt(r, d, a):
    return (r * cos(a), -d, r * sin(a))


def face_front(o):
    """Flip an open strip so that it faces the street."""
    me = o.data
    if sum(pl.normal.y * pl.area for pl in me.polygons) > 0:
        me.flip_normals()
    return o


def off(prof, dr=0.0, dd=0.0):
    """Shift a (radius, projection) section; points on the wall plane stay on it."""
    return [(r + dr, d + dd if d > 1e-6 else d) for r, d in prof]


def inset_sec(prof, t):
    """Offset an open (radius, projection) section into the solid by t (mortar bed behind the dressed face)."""
    n = len(prof)
    out = []
    for i, (r, d) in enumerate(prof):
        nx = ny = 0.0
        for a, b in ((i - 1, i), (i, i + 1)):
            if a < 0 or b >= n:
                continue
            dr, dd = prof[b][0] - prof[a][0], prof[b][1] - prof[a][1]
            l = hypot(dr, dd) or 1.0
            nx += -dd / l
            ny += dr / l
        l = hypot(nx, ny) or 1.0
        out.append((r + nx / l * t, max(0.0, d + ny / l * t) if d > 1e-6 else 0.0))
    return out


def thin_sec(prof, step=0.01):
    """Drop section points closer than `step` to the previous kept one (cheap section for the pointing)."""
    out = [prof[0]]
    for q in prof[1:-1]:
        if hypot(q[0] - out[-1][0], q[1] - out[-1][1]) >= step:
            out.append(q)
    out.append(prof[-1])
    return out


def ring_seg(name, prof, a0=0.0, a1=360.0, steps=6, cham=0.0):
    """Revolve an open (radius, projection) section around the window axis from a0 to a1 degrees (0 = right,
    90 = top, seen from the front). Section order: outer radius first, round the front, inner radius last.
    Partial segments get end faces, with `cham` easing the arrises at both ends (the joint)."""
    n = len(prof)
    closed = (a1 - a0) >= 359.999
    A0, A1 = radians(a0), radians(a1)
    loops = []
    if closed:
        loops = [(A0 + (A1 - A0) * i / steps, prof) for i in range(steps)]
    elif cham > 0:
        cr = sum(r for r, _ in prof) / n
        cd = sum(d for _, d in prof) / n
        small = []
        for r, d in prof:
            vx, vy = cr - r, cd - d
            l = hypot(vx, vy) or 1.0
            small.append((r + vx / l * cham, d + vy / l * cham if d > 1e-6 else 0.0))
        da = min(cham / cr, (A1 - A0) * 0.2)
        loops.append((A0, small))
        loops += [(A0 + da + (A1 - A0 - 2 * da) * i / steps, prof) for i in range(steps + 1)]
        loops.append((A1, small))
    else:
        loops = [(A0 + (A1 - A0) * i / steps, prof) for i in range(steps + 1)]
    verts = [_pt(r, d, a) for a, pp in loops for r, d in pp]
    L = len(loops)
    faces = []
    for i in range(L if closed else L - 1):
        i2 = (i + 1) % L
        for j in range(n - 1):
            faces.append((i * n + j, i2 * n + j, i2 * n + j + 1, i * n + j + 1))
    if not closed:
        faces.append(tuple(range(n)))
        faces.append(tuple(reversed(range((L - 1) * n, L * n))))
    return mesh_object(name, verts, faces)


def bar(name, p0, p1, sec):
    """Straight moulded bar between two (x, z) points on the window plane; sec = [(side, projection), ...]."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    l = hypot(dx, dz) or 1.0
    tx, tz = dx / l, dz / l
    n = len(sec)
    verts = [(px - s * tz, -d, pz + s * tx) for (px, pz) in (p0, p1) for s, d in sec]
    faces = [(j, (j + 1) % n, n + (j + 1) % n, n + j) for j in range(n)] + [tuple(range(n)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    uv_grain(o, axis=(tx, 0.0, tz))
    return o


def psector(r0, r1, a0, a1, center=(0.0, 0.0)):
    """Annular sector (disc sector when r0 = 0) as a CCW (x, z) polygon; angles in degrees."""
    n = max(1, int(ceil(abs(a1 - a0) / 9.0)))
    cx, cz = center
    angs = [radians(a0 + (a1 - a0) * i / n) for i in range(n + 1)]
    pts = [(cx + r1 * cos(a), cz + r1 * sin(a)) for a in angs]
    if r0 <= 1e-6:
        pts.append((cx, cz))
    else:
        pts += [(cx + r0 * cos(a), cz + r0 * sin(a)) for a in reversed(angs)]
    return pts


def clip_poly(pts, nx, nz):
    """Keep the part of a polygon on the side nx*x + nz*z >= 0."""
    out = []
    n = len(pts)
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        dp = p[0] * nx + p[1] * nz
        dq = q[0] * nx + q[1] * nz
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


def clip_seg(p, q, nx, nz):
    dp = p[0] * nx + p[1] * nz
    dq = q[0] * nx + q[1] * nz
    if dp < 0 and dq < 0:
        return None
    if dp >= 0 and dq >= 0:
        return p, q
    t = dp / (dp - dq)
    m = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
    return (p, m) if dp >= 0 else (m, q)


def rot2(p, a):
    ca, sa = cos(a), sin(a)
    return (p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca)


def lobe_pt(c, th, r, a):
    """Point on the circle of radius r around the lobe centre (c out along th); a = angle from the lobe axis."""
    return (c * cos(th) + r * cos(th + a), c * sin(th) + r * sin(th + a))


def lobe_phi(c, r):
    """Half span of a lobe arc of radius r up to where it meets its neighbour on the diagonal (the cusp / mitre)."""
    t = c * cos(pi / 4) + sqrt(max(r * r - (c * sin(pi / 4)) ** 2, 1e-9))
    return atan2(t * sin(pi / 4), t * cos(pi / 4) - c), t


def eye_outline(c, rl, re, na=5, nf=3):
    """Spandrel light between two lobes (axes 0 and 90 deg) and the outer ring: a curved triangle with sharp
    corners, returned as (base point, unit normal) samples so that base + e * normal is the outline grown by e
    (round joins at the corners). rl = lobe-side radius, re = outer radius of the sharp triangle."""
    if rl * rl <= c * c / 2:
        return None
    tb = c * cos(pi / 4) + sqrt(rl * rl - c * c / 2)
    cb = (tb * cos(pi / 4), tb * sin(pi / 4))
    ct = (re * re + c * c - rl * rl) / (2 * re * c)
    if tb >= re - 0.01 or abs(ct) >= 1.0:
        return None
    t1 = atan2(sqrt(1 - ct * ct), ct)
    if t1 >= pi / 4 - 0.03:
        return None
    crr = (re * cos(t1), re * sin(t1))
    fb = atan2(cb[1], cb[0] - c)
    fr = atan2(crr[1], crr[0] - c)
    arcs = []
    # outer arc (centre origin), right corner -> left corner
    arcs.append([((re * cos(a), re * sin(a)), (cos(a), sin(a))) for a in [t1 + (pi / 2 - 2 * t1) * i / (na + 2) for i in range(na + 3)]])
    # lobe at 90 deg: left corner -> bottom corner (mirror image of the lobe-0 arc), then lobe 0: bottom -> right
    l0 = [((c + rl * cos(a), rl * sin(a)), (-cos(a), -sin(a))) for a in [fb + (fr - fb) * i / na for i in range(na + 1)]]
    arcs.append([((b[1], b[0]), (n[1], n[0])) for b, n in reversed(l0)])
    arcs.append(l0)
    out = []
    mids = []
    for i, arc in enumerate(arcs):
        mids.append(len(out) + len(arc) // 2)
        out += arc
        nxt = arcs[(i + 1) % 3][0]
        a0 = atan2(arc[-1][1][1], arc[-1][1][0])
        a1 = atan2(nxt[1][1], nxt[1][0])
        da = (a1 - a0) % (2 * pi)
        for j in range(1, nf):
            a = a0 + da * j / nf
            out.append((arc[-1][0], (cos(a), sin(a))))
    cen = ((cb[0] + crr[0] + crr[1]) / 3, (cb[1] + crr[1] + crr[0]) / 3)
    return out, mids, cen, re - tb


def keystone(name, a_deg, ks, panel=0.008, diamond=False, nb=4, sunk=None):
    """Wedge key block: tapered sides, leaning face, chamfered arrises and a sunk panel cut all round."""
    r_in, r_out, wi, wo, d_bot, d_top = ks["r_in"], ks["r_out"], ks["wi"], ks["wo"], ks["d_bot"], ks["d_top"]
    H = r_out - r_in
    a0 = radians(a_deg)
    ch = 0.006
    border = 0.014 + 0.11 * wi

    def P(uu, vv, d):
        r = r_in + vv * H
        t = uu * (wi + vv * (wo - wi)) / 2
        return _pt(r, d, a0 - t / r)

    def face(vv):
        return d_bot + vv * (d_top - d_bot)

    def loop(inset, dfn):
        pts = []
        for vv, rev in ((inset / H, False), (1 - inset / H, True)):
            w = (wi + vv * (wo - wi)) / 2
            u1 = 1 - inset / w
            us = [-u1 + 2 * u1 * i / nb for i in range(nb + 1)]
            if rev:
                us.reverse()
            pts += [P(u, vv, dfn(vv)) for u in us]
        return pts

    loops = [loop(0.0, lambda v: 0.0), loop(0.0, lambda v: face(v) - ch), loop(ch, face), loop(border, face),
             loop(border + panel * 1.3, lambda v: face(v) - panel)]
    n = 2 * (nb + 1)
    verts = [q for lp in loops for q in lp]
    faces = []
    for li in range(len(loops) - 1):
        for i in range(n):
            i2 = (i + 1) % n
            faces.append((li * n + i, li * n + i2, (li + 1) * n + i2, (li + 1) * n + i))
    last = (len(loops) - 1) * n
    for i in range(nb):
        faces.append((last + i, last + i + 1, last + n - 2 - i, last + n - 1 - i))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    face_front(o)
    smooth(o, 30)
    out = [o]
    if sunk is not None:
        sunk += loops[-1]
    if diamond:
        # raised lozenge left standing in the panel
        su = 0.9 * (1 - (border + panel * 1.3) / ((wi + wo) / 4))
        sv = 0.5 - (border + panel * 1.3 + 0.002) / H
        dz = face(0.5) - panel - 0.001
        dv = [P(-su, 0.5, dz), P(0, 0.5 - sv, face(0.5 - sv) - panel - 0.001), P(su, 0.5, dz), P(0, 0.5 + sv, face(0.5 + sv) - panel - 0.001),
              P(0, 0.5, face(0.5) + 0.006)]
        d = mesh_object(name + "_lozenge", dv, [(i, (i + 1) % 4, 4) for i in range(4)])
        recalc_normals(d)
        face_front(d)
        out.append(d)
    return out


def key_half(ks, ra, rb):
    """Smallest half angle (degrees) covered by the key's tapered sides between radii ra and rb."""
    best = 1e9
    for i in range(9):
        r = ra + (rb - ra) * i / 8
        vv = clamp((r - ks["r_in"]) / (ks["r_out"] - ks["r_in"]))
        best = min(best, (ks["wi"] + vv * (ks["wo"] - ks["wi"])) / 2 / r)
    return degrees(best)


# ---------------------------------------------------------------------------------------------- weathering masks
def m_streak(seed, freq=34.0, bias=0.36):
    """Vertical run-off streaks: 1D noise across x, constant down the wall."""
    r = np.random.RandomState(int(seed) % 100000)
    ph = r.uniform(0, 6.283, 4)
    fr = freq * np.array([1.0, 2.3, 4.1, 7.7])
    am = np.array([1.0, 0.6, 0.4, 0.25])

    def fn(co, n):
        x = np.asarray(co)[:, 0]
        s = np.zeros(len(x))
        for f, q, a in zip(fr, ph, am):
            s += a * np.sin(x * f + q)
        return np.clip(bias + 0.45 * s, 0.0, 1.0)
    return fn


def m_near(pts, tol=0.004):
    """1 on vertices lying within tol of any of the given points."""
    P_ = np.array(pts, dtype=float).reshape(-1, 3)

    def fn(co, n):
        c_ = np.asarray(co, dtype=float)
        if not len(P_):
            return np.zeros(len(c_))
        d = np.min(np.linalg.norm(c_[:, None, :] - P_[None, :, :], axis=2), axis=1)
        return (d < tol).astype(float)
    return fn


def m_max(*fns):
    def fn(co, n):
        out = np.zeros(len(co))
        for f in fns:
            out = np.maximum(out, f(co, n) if callable(f) else f)
        return out
    return fn


def build(p, rng):
    D = p.diameter
    R = D / 2
    k = p.surround_width
    Ro = R + k
    W = p.weathering
    lit = bool(p.lit)
    kind = p.surround
    tim = kind == "timber"
    nkeys = int(p.keystones)
    hood = bool(p.hood)
    stained = p.glazing == "stained"
    quatre = p.bars == "quatrefoil"
    sd = int(p.seed)

    key_angles = {0: [], 1: [90], 2: [90, 270], 4: [0, 90, 180, 270]}[nkeys]
    use_stone = kind == "stone" or (kind == "brick" and (bool(key_angles) or hood)) or (quatre and not tim)
    use_lead = (hood and tim) or stained or p.glazing == "leaded" or quatre
    use_framep = (not quatre) or tim
    # fine-grained freestone: the sandy, isotropic library face (no tooling streaks), tinted per stone colour
    stone = mat("mortar", p.stone_color, scale=1.2) if use_stone else None
    mortar = mat("mortar", "#e4dcc9", scale=2.0, grime=0.4) if not tim else None
    lead = mat("lead") if use_lead else None
    framep = mat("wood_painted", p.frame_color, rough=1.2) if use_framep else None
    oak = mat("wood_oak", "#5d4432") if tim else None
    n_other = sum([use_stone, not tim, use_lead, use_framep, tim * 2, not quatre, kind == "brick"])

    # ------------------------------------------------------------------ surround sections (radius, projection)
    ph = 0.0
    if kind == "stone":
        pr = p.projection
        sec = [(Ro, 0.0), (Ro, pr * 0.50), (Ro - 0.005, pr * 0.50 + 0.005), (Ro - k * 0.10, pr * 0.56),
               (Ro - k * 0.13, pr * 0.58), (Ro - k * 0.19, pr * 0.63), (Ro - k * 0.25, pr * 0.72), (Ro - k * 0.285, pr * 0.82),
               (Ro - k * 0.29, pr * 0.875), (Ro - k * 0.32, pr * 0.875),
               (Ro - k * 0.33, pr * 0.95), (Ro - k * 0.37, pr * 1.0), (Ro - k * 0.43, pr * 1.0), (Ro - k * 0.47, pr * 0.95),
               (Ro - k * 0.48, pr * 0.885), (Ro - k * 0.76, pr * 0.885), (Ro - k * 0.79, pr * 0.85),
               (R + 0.014, FD + 0.014), (R, FD + 0.008), (R, 0.0)]
    elif tim:
        # architrave section: back band, ovolo, fascia, quirked bead and an ovolo-moulded inner edge
        pr = max(0.10, p.projection * 0.9)
        ph = pr * 0.5
        sec = [(Ro, 0.0), (Ro, ph), (Ro - 0.005, ph + 0.005), (Ro - k * 0.13, ph + 0.005), (Ro - k * 0.15, pr * 0.66),
               (Ro - k * 0.19, pr * 0.80), (Ro - k * 0.25, pr * 0.90), (Ro - k * 0.31, pr * 0.94), (Ro - k * 0.32, pr),
               (Ro - k * 0.63, pr), (Ro - k * 0.65, pr * 0.94), (Ro - k * 0.675, pr * 0.975), (Ro - k * 0.705, pr * 0.99),
               (Ro - k * 0.735, pr * 0.975), (Ro - k * 0.76, pr * 0.94), (Ro - k * 0.775, pr * 0.9), (Ro - k * 0.85, pr * 0.865),
               (Ro - k * 0.925, pr * 0.78), (Ro - k * 0.98, pr * 0.66), (R, pr * 0.58), (R, 0.0)]
    else:
        pr = max(0.06, p.projection * 0.55)
        sec = [(Ro, 0.0), (Ro, pr - 0.003), (Ro - 0.003, pr), (R + 0.003, pr), (R, pr - 0.003), (R, 0.0)]
    hw = 0.055 + 0.1 * k                    # hood mould width
    prh = pr + 0.022                        # hood mould projection
    Rm = (R + Ro) / 2

    # ------------------------------------------------------------------ key blocks at the quarters
    kw = 0.1 + 0.07 * D                     # key width at mid radius
    rk = rng.fork("keys")
    kspec = {}
    for a in key_angles:
        crown, foot = a == 90, a == 270
        sc = 1.2 if crown else (1.0 if foot else 0.86)
        if crown:
            ext = (hw + 0.024) if (hood and not tim) else 0.045 + 0.03 * k
        else:
            ext = (0.03 + 0.03 * k) if foot else (0.016 + 0.02 * k)
        w_mid = kw * sc * rk.vary(1.0, 0.04)
        d_top = pr + (0.032 if crown else 0.024 if foot else 0.015) + rk.jitter(0, 0.002)
        kspec[a] = {"r_in": R + 0.004, "r_out": Ro + ext, "wi": w_mid * 0.72, "wo": w_mid * 1.32,
                    "d_bot": d_top - (0.012 if crown else 0.008), "d_top": d_top}
    keys = []
    sunk = []                               # floors of the sunk key panels (dirt gathers there)
    for a in key_angles:
        for o in keystone(f"key{a}", a, kspec[a], diamond=(a in (90, 270)), sunk=sunk):
            if tim:
                uv_grain(o, axis=(cos(radians(a)), 0, sin(radians(a))))
            keys.append(o)

    # free arcs between the key blocks (the ring runs a little way into the tapered keys)
    if key_angles:
        arcs = []
        for i, a in enumerate(key_angles):
            b = key_angles[(i + 1) % len(key_angles)]
            b = b if b > a else b + 360
            arcs.append((a + key_half(kspec[a], R, Ro) - 0.3, b - key_half(kspec[b % 360], R, Ro) + 0.3))
    else:
        arcs = None

    def joint_cuts(a, b, n, r, rs_, jit=0.01, gap=0.004):
        """n pieces between a and b: cut angles (seeded small length variation) and half joint widths in degrees."""
        cuts = [a + (b - a) * i / n for i in range(n + 1)]
        for i in range(1, n):
            cuts[i] += degrees(rs_.jitter(0, jit) / r)
        gs = [degrees((gap + rs_.jitter(0, gap * 0.2)) / r) for _ in range(n + 1)]
        if arcs is None:
            gs[n] = gs[0]
        else:
            gs[0] = gs[n] = 0.0
        return cuts, gs

    # ------------------------------------------------------------------ the ring itself
    rs = rng.fork("ring")
    ring = []
    stone_objs = []
    mortar_objs = []
    if kind == "stone":
        target = 0.2 + 0.07 * D
        if arcs is None:
            n = max(6, round(2 * pi * Rm / target))
            a_start = 90 - 180.0 / n
            arcs_s = [(a_start, a_start + 360.0, n)]
        else:
            arcs_s = [(a, b, max(1, round(radians(b - a) * Rm / target))) for a, b in arcs]
        msec = thin_sec(inset_sec(sec, 0.002))
        for a, b, n in arcs_s:
            cuts, gs = joint_cuts(a, b, n, Rm, rs)
            for i in range(n):
                s0, s1 = cuts[i] + gs[i], cuts[i + 1] - gs[i + 1]
                o = ring_seg("voussoir", off(sec, rs.jitter(0, 0.001), rs.jitter(0, 0.0016)), s0, s1,
                             steps=max(2, round((s1 - s0) / 8.0)), cham=0.0012)
                smooth(o, 38)
                ring.append(o)
            # lime mortar pointing, 2 mm back from the dressed face, in every radial joint
            for i in range(n + 1):
                if gs[i] > 0 and not (arcs is None and i == n):
                    w = gs[i] + degrees(0.0015 / Rm)
                    mortar_objs.append(ring_seg("joint", msec, cuts[i] - w, cuts[i] + w, steps=1))
        stone_objs += ring
    elif tim:
        nf = max(6, 2 * round(2 * pi * Rm / 0.5 / 2))
        spans = arcs if arcs else [(90 - 180.0 / nf, 90 - 180.0 / nf + 360.0)]
        peg = cylinder("peg", 0.0065, 0.012, verts=7, bevel=0.0015, segments=1)
        pm = []
        for a, b in spans:
            n = max(1, round((b - a) / (360.0 / nf)))
            for i in range(n):
                s0 = a + (b - a) * i / n
                s1 = a + (b - a) * (i + 1) / n
                gdeg = degrees(0.0008 / Rm)
                o = ring_seg("felloe", off(sec, rs.jitter(0, 0.001), rs.jitter(0, 0.002)), s0 + gdeg, s1 - gdeg,
                             steps=max(3, round((s1 - s0) / 7.0)), cham=0.0025)
                am = radians((s0 + s1) / 2)
                uv_grain(o, axis=(-sin(am), 0.0, cos(am)))
                smooth(o, 38)
                ring.append(o)
                for e, sgn in ((s0, 1), (s1, -1)):
                    for rr in (Ro - k * 0.40, Ro - k * 0.55):
                        ap = radians(e) + sgn * 0.045 / rr
                        pm.append(trs((rr * cos(ap) + rs.jitter(0, 0.002), -(pr - 0.003), rr * sin(ap) + rs.jitter(0, 0.002)),
                                      (90, 0, rs.u(0, 90))))
        pegs = instances_to_mesh("pegs", peg, pm, keep_base=False)
        assign(pegs, mat("wood_endgrain", "#7a5c40"))
        for o in ring:
            assign(o, oak)
    else:
        # gauged brick headers: every brick is its own wedge and shows ONE brick of the fired-clay texture
        brick = mat("brick_red")
        jt = 0.009
        courses = 2 if k >= 0.2 else 1
        cl = (k - jt * (courses - 1)) / courses
        spans = arcs if arcs else [(0.0, 360.0)]
        bricks = []
        su = min(1.0, 0.15 / (cl + 0.6 * pr))
        g_run = mask_mul(m_streak(sd + 3), height_ramp(-Ro - 0.05, -R * 0.2), 0.2 + 0.5 * W)
        g_moss = mask_mul(facing((0, 0, 1), 2.0), noise_mask(0.1, seed=sd + 5, lo=0.3, hi=0.7), 0.9 * W)
        for c in range(courses):
            r0 = R + c * (cl + jt)
            r1 = r0 + cl
            rmid = (r0 + r1) / 2
            for a, b in spans:
                n = max(2, round(radians(b - a) * r0 / 0.076))
                dstep = (b - a) / n
                if c % 2 == 0:
                    cuts = [a + dstep * i for i in range(n + 1)]
                elif arcs is None:
                    cuts = [a + dstep * (i + 0.5) for i in range(n + 1)]        # break bond with the course below
                else:
                    cuts = [a] + [a + dstep * (i + 0.5) for i in range(n)] + [b]
                for i in range(1, len(cuts) - 1):
                    cuts[i] += degrees(rs.jitter(0, 0.0025) / r0)
                g = degrees(jt / 2 / r0)
                for i in range(len(cuts) - 1):
                    s0, s1 = cuts[i] + g, cuts[i + 1] - g
                    spall = rs.chance(0.04 + 0.08 * W)
                    pb = pr + rs.jitter(0, 0.0008) - (rs.u(0.001, 0.002) if spall else 0.0)
                    bsec = [(r1, 0.0), (r1, pb - 0.002), (r1 - 0.002, pb), (r0 + 0.002, pb), (r0, pb - 0.002), (r0, 0.0)]
                    o = ring_seg("brick", bsec, s0, s1, steps=1, cham=0.002)
                    if spall:
                        displace_noise(o, strength=0.0025, scale=0.03, seed=sd + i, axis_mask=(0, 1, 0))
                    am = radians((s0 + s1) / 2)
                    row = int(rs.u(0, 11.999))
                    uc = (0.1125 if row % 2 == 0 else 0.0) + 0.225 * int(rs.u(0, 3.999))
                    vc = 0.0375 + 0.075 * row
                    uax = (su * cos(am), -0.6 * su, su * sin(am))
                    vax = (-0.42 * sin(am), -0.2, 0.42 * cos(am))
                    uv_planar(o, uax, vax, offset=(uc - su * (rmid + 0.3 * pr), vc - 0.1 * pr))
                    assign(o, brick)
                    dark = rs.u(0.2, 0.45) if rs.chance(0.1 + 0.1 * W) else 0.0     # over-burnt / sooted header
                    paint(o, grime=m_max(g_run, dark), special=g_moss)
                    bricks.append(o)
        mortar_objs.append(ring_seg("mortar_bed", [(Ro - 0.004, 0.0), (Ro - 0.004, pr - 0.0035), (R + 0.004, pr - 0.0035), (R + 0.004, 0.0)],
                                    steps=56))
        join(bricks, "bricks")

    # ------------------------------------------------------------------ hood mould
    crown = kspec.get(90)
    if hood:
        if tim:
            # dressed lead drip welted over the back band at the crown of the ring
            b0 = ph + 0.005
            lsec = [(Ro + 0.004, 0.0), (Ro + 0.004, b0 + 0.012), (Ro + 0.001, b0 + 0.015), (Ro - 0.016, b0 + 0.015),
                    (Ro - 0.018, b0 + 0.012), (Ro - 0.018, b0 + 0.001), (Ro - 0.015, b0 + 0.001),
                    (Ro - 0.015, b0 + 0.009), (Ro + 0.001, b0 + 0.009), (Ro + 0.001, 0.0)]
            kc = key_half(crown, Ro - 0.02, Ro + 0.005) + 0.2 if crown else 0.0
            spans_h = [(22.0, 90 - kc), (90 + kc, 158.0)] if crown else [(22.0, 158.0)]
            for a, b in spans_h:
                o = ring_seg("lead_drip", lsec, a, b, steps=max(4, round((b - a) / 5.0)), cham=0.001)
                transform_verts(o, lambda co: Vector((co.x, co.y, co.z + 0.0015 * sin(co.x * 55.0))))
                assign(o, lead)
                smooth(o, 40)
        else:
            Rh = Ro + hw
            hsec = [(Rh, 0.0), (Rh, prh * 0.5), (Rh - hw * 0.12, prh * 0.86), (Rh - hw * 0.34, prh), (Rh - hw * 0.62, prh),
                    (Rh - hw * 0.72, prh - 0.012), (Rh - hw * 0.8, prh - 0.03), (Ro + 0.004, prh - 0.036), (Ro - 0.002, pr * 0.5 + 0.004),
                    (Ro - 0.002, 0.0)]
            hmsec = thin_sec(inset_sec(hsec, 0.002))
            rh_ = rng.fork("hood")
            side = kspec.get(0)
            h0 = (degrees((side["wo"] / 2) / side["r_out"]) + 3.0) if side else -7.0
            h1 = 180.0 - h0
            kc = key_half(crown, Ro, Rh) - 0.3 if crown else 0.0
            spans_h = [(h0, 90 - kc), (90 + kc, h1)] if crown else [(h0, h1)]
            for a, b in spans_h:
                n = max(1, round(radians(b - a) * Rh / (0.3 + 0.08 * D)))
                cuts = [a + (b - a) * i / n for i in range(n + 1)]
                g = degrees(0.004 / Rh)
                for i in range(n):
                    s0 = cuts[i] + (g if i > 0 else 0.0)
                    s1 = cuts[i + 1] - (g if i < n - 1 else 0.0)
                    o = ring_seg("label", off(hsec, rh_.jitter(0, 0.001), rh_.jitter(0, 0.0016)), s0, s1,
                                 steps=max(2, round((s1 - s0) / 8.0)), cham=0.0012)
                    smooth(o, 38)
                    stone_objs.append(o)
                for i in range(1, n):
                    w = g + degrees(0.0015 / Rh)
                    mortar_objs.append(ring_seg("joint", hmsec, cuts[i] - w, cuts[i] + w, steps=1))
            # square label stops closing the ends of the hood mould
            ssec = [(Rh + 0.014, 0.0), (Rh + 0.014, prh + 0.004), (Rh + 0.004, prh + 0.014), (Ro - 0.006, prh + 0.014),
                    (Ro - 0.014, prh + 0.004), (Ro - 0.014, 0.0)]
            sa = degrees(0.04 / (Ro + hw / 2))
            for e, sgn in ((h0, -1), (h1, 1)):
                o = ring_seg("label_stop", ssec, min(e, e + sgn * 2 * sa), max(e, e + sgn * 2 * sa), steps=1, cham=0.006)
                smooth(o, 30)
                stone_objs.append(o)
    if not tim:
        stone_objs += keys
    else:
        for o in keys:
            assign(o, oak)
            paint(o, grime=0.25 * W, wear=mask_mul(facing((0, 0, 1), 1.5), 0.6 * W))

    def stone_weather(o, top, bot):
        """Run-off streaks at the foot, green growth on the upward ledges, washed pale stone at the crown."""
        h = top - bot
        paint(o,
              grime=m_max(mask_mul(m_streak(sd + 3), height_ramp(bot - 0.03, bot + 0.42 * h), 0.3 + 0.7 * W),
                          mask_mul(m_near(sunk), 0.37 + 0.16 * W)),
              special=m_max(mask_mul(facing((0, 0, 1), 1.3), noise_mask(0.09, seed=sd + 5, lo=0.2, hi=0.55), 1.6 * W),
                            mask_mul(m_streak(sd + 8, 21.0, 0.2), height_ramp(bot - 0.03, bot + 0.2 * h), 0.7 * W)),
              wear=mask_mul(height_ramp(top + 0.05, top - 0.45 * h), facing((0, -1, 0), 1.0), 0.8 * W))

    if stone_objs:
        for o in stone_objs:
            assign(o, stone)
            displace_noise(o, strength=0.0007, scale=0.06, seed=sd + 17, axis_mask=(1, 0, 1))
        sj = join(stone_objs, "stonework")
        stone_weather(sj, Ro + (hw if hood else 0.0), -Ro)
    if mortar_objs:
        for o in mortar_objs:
            assign(o, mortar)
        mj = join(mortar_objs, "pointing")
        paint(mj, grime=mask_mul(height_ramp(-Ro - 0.05, 0.0), 0.5 * W), special=mask_mul(facing((0, 0, 1), 1.0), 0.6 * W))
    if tim and ring:
        tj = join(ring, "felloes")
        paint(tj, grime=mask_mul(m_streak(sd + 3), height_ramp(-Ro - 0.05, -R * 0.2), 0.3 + 0.5 * W),
              wear=mask_mul(facing((0, 0, 1), 1.5), noise_mask(0.08, seed=sd + 6, lo=0.3, hi=0.8), 0.9 * W),
              special=mask_mul(facing((0, 0, 1), 3.0), noise_mask(0.1, seed=sd + 5, lo=0.35, hi=0.65), W))

    # ------------------------------------------------------------------ glass materials
    rgl = rng.fork("glass")
    if stained:
        gl = 1.8 if lit else 0.3
        ncol = max(3, min(7, 10 - n_other))
        remap = {7: [0, 1, 2, 3, 4, 5, 6], 6: [0, 1, 2, 3, 4, 5, 4], 5: [0, 1, 2, 3, 4, 1, 4], 4: [0, 1, 1, 3, 4, 1, 4],
                 3: [0, 1, 1, 4, 4, 1, 4]}[ncol]
        gcache = {}

        def gmat(ci):
            ci = remap[ci]
            if ci not in gcache:
                gcache[ci] = mat("window_lit", PAL[ci], glow=gl * (0.9 if ci in (4, 6) else 1.0), curtain=1.0)
            return gcache[ci]
        glass_plain = None
    elif lit:
        glass_plain = mat("window_lit", p.glass_tint, glow=2.2, curtain=1.0, rough=2.5)
    else:
        glass_plain = mat("window_lit", "#8ea2c4", glow=0.5, curtain=1.0)       # dark room: drawn curtain barely caught by the sky
    cells = []                               # stained pieces
    leads = []
    cw, ch = 0.0055 + 0.0015 * D, 0.003      # lead came width / height

    def piece(pts, ci):
        """One hand-cut piece of pot-metal glass: its own patch (and streak direction) of the glass texture."""
        if len(pts) < 3:
            return
        if rgl.chance(0.05):
            ci = rgl.pick([4, 3, 2])           # old repairs in whatever glass was to hand
        elif ci in (1, 4) and rgl.chance(0.35):
            ci = 5 if ci == 1 else 6           # no two sheets of pot metal are quite the same hue
        o = mesh_object("quarry", [(x, -DG, z) for x, z in pts], [tuple(range(len(pts)))])
        cx = sum(q[0] for q in pts) / len(pts)
        cz = sum(q[1] for q in pts) / len(pts)
        t = rgl.u(0, pi)
        ua = (cos(t), 0.0, sin(t))
        va = (-sin(t), 0.0, cos(t))
        uv_planar(o, ua, va, offset=(rgl.u(0.25, 0.75) - (cx * ua[0] + cz * ua[2]), rgl.u(0.25, 0.75) - (cx * va[0] + cz * va[2])))
        assign(o, gmat(ci))
        paint(o, grime=rgl.u(0.0, 0.55))       # value jitter from piece to piece
        cells.append(o)

    def came_line(p0, p1):
        dx, dz = p1[0] - p0[0], p1[1] - p0[1]
        l = hypot(dx, dz)
        if l < 1e-4:
            return
        nx, nz = -dz / l * cw / 2, dx / l * cw / 2
        verts = []
        for q in (p0, p1):
            verts += [(q[0] + nx, -(DG - 0.001), q[1] + nz), (q[0], -(DG + ch), q[1]), (q[0] - nx, -(DG - 0.001), q[1] - nz)]
        leads.append(face_front(mesh_object("came", verts, [(0, 1, 4, 3), (1, 2, 5, 4)])))

    def came_rad(r0, r1, adeg, center=(0.0, 0.0)):
        a = radians(adeg)
        came_line((center[0] + r0 * cos(a), center[1] + r0 * sin(a)), (center[0] + r1 * cos(a), center[1] + r1 * sin(a)))

    def came_arc(r, a0=0.0, a1=360.0, center=(0.0, 0.0)):
        full = (a1 - a0) >= 359.999
        n = max(3, int(ceil((a1 - a0) / 360.0 * max(14, min(40, r * 120)))))
        m = n if full else n + 1
        verts = []
        for i in range(m):
            a = radians(a0 + (a1 - a0) * i / n)
            for rr, d in ((r + cw / 2, DG - 0.001), (r, DG + ch), (r - cw / 2, DG - 0.001)):
                verts.append((center[0] + rr * cos(a), -d, center[1] + rr * sin(a)))
        faces = []
        for i in range(n):
            i2 = (i + 1) % m
            faces += [(3 * i, 3 * i + 1, 3 * i2 + 1, 3 * i2), (3 * i + 1, 3 * i + 2, 3 * i2 + 2, 3 * i2 + 1)]
        leads.append(face_front(mesh_object("came_arc", verts, faces)))

    def lattice(rg):
        s = 0.075 + 0.02 * D
        for ang in (58.0, 122.0):
            t = (cos(radians(ang)), sin(radians(ang)))
            nrm = (-t[1], t[0])
            m = int(rg / s) + 1
            for i in range(-m, m + 1):
                c_ = (i + 0.5) * s
                if abs(c_) >= rg - 0.01:
                    continue
                h = sqrt(rg * rg - c_ * c_)
                came_line((c_ * nrm[0] - h * t[0], c_ * nrm[1] - h * t[1]), (c_ * nrm[0] + h * t[0], c_ * nrm[1] + h * t[1]))

    def plain_glass(rg):
        go = mesh_object("glass", [(rg * cos(radians(a)), -DG, rg * sin(radians(a))) for a in range(0, 360, 8)], [tuple(range(45))])
        s = min(1.0, 0.66 / (2 * rg))            # keep to the evenly curtained middle of the texture
        uv_planar(go, (1, 0, 0), (0, 0, 1), scale=s, offset=(0.54, 0.5))
        assign(go, glass_plain)
        if not lit:
            # the pane itself, a hair in front of the dark room: cool sky reflection and a specular glint
            pane = mesh_object("pane", [(rg * cos(radians(a)), -(DG + 0.0045), rg * sin(radians(a))) for a in range(0, 360, 8)],
                               [tuple(range(45))])
            uv_planar(pane, (1, 0, 0), (0, 0, 1), scale=1.0, offset=(0.5, 0.5))
            assign(pane, mat("glass_clear", "#b9cdee", opacity=0.3, rough=0.6))
        return go

    tp = 0.1 + 0.012 * D                     # target size of a piece of stained glass
    frame_objs = []
    brass_objs = []
    if quatre:
        # -------------------------------------------------------------- moulded quatrefoil in four jointed stones
        tr_mat = framep if tim else stone
        c, rf = 0.48 * R, 0.375 * R
        tf = FD + 0.006                       # tracery face
        ms = min(0.022 + 0.02 * D, 0.145 * R - 0.01)     # width of the lobe moulding
        es = 0.012 + 0.006 * D                # width of the moulding round the spandrel lights
        rho = 0.008 + 0.006 * D               # radius easing the corners of the spandrel lights
        # (distance into the stone from the daylight edge, projection): face, hollow splay, roll, fillet, glazing rebate
        tsec = [(ms, tf), (ms * 0.82, tf - 0.006), (ms * 0.62, tf - 0.013), (ms * 0.5, tf - 0.016), (ms * 0.45, tf - 0.011),
                (ms * 0.37, tf - 0.007), (ms * 0.28, tf - 0.0065), (ms * 0.2, tf - 0.0095), (ms * 0.15, tf - 0.016),
                (ms * 0.08, tf - 0.021), (0.0, tf - 0.024), (0.0, DG)]
        esec = [(es, tf), (es * 0.5, tf - 0.007), (es * 0.2, tf - 0.0105), (0.0, tf - 0.017), (0.0, DG)]
        gap = 0.0006 if tim else 0.003
        eye = eye_outline(c, rf + ms + 0.008 + 0.01 * D + es + rho, R - es - 0.016 - rho)
        if eye and eye[3] < 0.03:
            eye = None
        nu = 8
        tr_objs = []
        for q in range(4):
            th = q * pi / 2
            # the moulding runs round half of lobe q, mitres at the cusp and runs back along half of lobe q + 1
            loops = [(th, 1, i / nu) for i in range(nu + 1)] + [(th + pi / 2, -1, 1 - i / nu) for i in range(1, nu + 1)]
            verts = []
            n = len(tsec)
            for lth, sgn, u in loops:
                for s, d in tsec:
                    rr = rf + s
                    g = gap / rr
                    phi, _ = lobe_phi(c, rr)
                    x, z = lobe_pt(c, lth, rr, sgn * (g + (phi - g) * u))
                    verts.append((x, -d, z))
            faces = []
            for i in range(len(loops) - 1):
                for j in range(n - 1):
                    faces.append((i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1))
            o = face_front(mesh_object("tracery", verts, faces))
            smooth(o, 35)
            tr_objs.append(o)
            # flat face of the stone out to the ring, pierced by the spandrel light
            g_o = degrees(gap / R)
            outer = [(R * cos(radians(a)), R * sin(radians(a))) for a in [q * 90 + g_o + (90 - 2 * g_o) * i / 12 for i in range(13)]]
            rr = rf + ms
            phi, _ = lobe_phi(c, rr)
            g = gap / rr
            inner = [lobe_pt(c, th + pi / 2, rr, -(g + (phi - g) * i / nu)) for i in range(nu + 1)]
            inner += [lobe_pt(c, th, rr, g + (phi - g) * (1 - i / nu)) for i in range(1, nu + 1)]
            holes = []
            if eye:
                esamp, emids, ecen, _ = eye
                holes.append([rot2((b[0] + nv[0] * (rho + es), b[1] + nv[1] * (rho + es)), th) for b, nv in esamp])
            pl = shape_extrude("tracery_face", outer + inner, holes, depth=tf - DG - 0.0045, loc=(0, -(DG + 0.0045), 0), rot=(90, 0, 0))
            tr_objs.append(pl)
            if eye:
                n2 = len(esec)
                verts = []
                for b, nv in esamp:
                    for s, d in esec:
                        x, z = rot2((b[0] + nv[0] * (rho + s), b[1] + nv[1] * (rho + s)), th)
                        verts.append((x, -d, z))
                L = len(esamp)
                faces = [(i * n2 + j, ((i + 1) % L) * n2 + j, ((i + 1) % L) * n2 + j + 1, i * n2 + j + 1) for i in range(L) for j in range(n2 - 1)]
                o = face_front(mesh_object("eye_moulding", verts, faces))
                smooth(o, 35)
                tr_objs.append(o)
            if not tim:
                # mortar in the joint through the lobe tip
                jm = [(R - c - rf, tf - 0.003)] + [(s, d - 0.003) for s, d in tsec]
                verts = []
                for sgn in (-1, 1):
                    for s, d in jm:
                        x, z = lobe_pt(c, th, rf + s, sgn * (gap + 0.003) / (rf + s))
                        verts.append((x, -d, z))
                n3 = len(jm)
                mo = face_front(mesh_object("tracery_joint", verts, [(j, n3 + j, n3 + j + 1, j + 1) for j in range(n3 - 1)]))
                assign(mo, mortar)
        for t in tr_objs:
            assign(t, tr_mat)
            if tim:
                uv_grain(t, axis="X")
        trj = join(tr_objs, "tracery")
        if tim:
            paint(trj, wear=mask_mul(noise_mask(0.05, seed=sd + 8, lo=0.45, hi=0.75), height_ramp(-R, R * 0.2), W), special=0.4 * W)
        else:
            stone_weather(trj, R, -R)

        rg = R - 0.004
        if stained:
            cmed = rgl.pick([0, 3])
            cgr = rgl.pick([1, 2]) if cmed == 0 else 1
            cbo = 3 if cmed == 0 else 0
            a_in = 225.0 - degrees(asin(min(1.0, c / (rf * sqrt(2)))))
            r_a, r_b, r_c = 0.3 * rf, 0.66 * rf, rf + 0.008
            m1 = max(4, 2 * round(2 * pi * (r_a + r_b) / 2 / tp / 2))
            m2 = max(8, 4 * round(2 * pi * (r_b + rf) / 2 / (tp * 0.92) / 4))
            _, tcusp = lobe_phi(c, rf)
            for q in range(4):
                th = q * pi / 2
                ctr = (c * cos(th), c * sin(th))
                n1 = (cos(th - pi / 4), sin(th - pi / 4))
                n2 = (cos(th + pi / 4), sin(th + pi / 4))
                thd = degrees(th)
                piece(psector(0, r_a, 0, 360, ctr)[:-1], cmed)
                came_arc(r_a, center=ctr)
                came_arc(r_b, center=ctr)
                for j in range(m1):
                    a0 = thd + (j - 0.5) * 360.0 / m1
                    piece(psector(r_a, r_b, a0, a0 + 360.0 / m1, ctr), cgr if j % 2 == 0 else 4)
                    came_rad(r_a, r_b, a0, ctr)
                for j in range(m2):
                    a0 = thd + (j - 0.5) * 360.0 / m2
                    # outer edge: r_c, in the glazing rebate of the tracery; but only rf where the lobe faces the centre
                    # lights, whose glass starts there (two glowing sheets in one plane fight each other)
                    u0, u1 = (j - 0.5) * 360.0 / m2, (j + 0.5) * 360.0 / m2
                    cuts = [u0] + [a for a in (a_in, 360.0 - a_in) if u0 < a < u1] + [u1]
                    pts = []
                    for u, v in zip(cuts, cuts[1:]):
                        rr = rf if a_in <= (u + v) / 2 <= 360.0 - a_in else r_c
                        nn = max(1, int(ceil((v - u) / 9.0)))
                        pts += [(ctr[0] + rr * cos(radians(thd + u + (v - u) * i / nn)), ctr[1] + rr * sin(radians(thd + u + (v - u) * i / nn)))
                                for i in range(nn + 1)]
                    nn = max(1, int(ceil((u1 - u0) / 9.0)))
                    pts += [(ctr[0] + r_b * cos(radians(thd + u0 + (u1 - u0) * i / nn)), ctr[1] + r_b * sin(radians(thd + u0 + (u1 - u0) * i / nn)))
                            for i in reversed(range(nn + 1))]
                    pts = clip_poly(clip_poly(pts, *n1), *n2)
                    piece(pts, 4 if j % 2 == 0 else (cbo if j % 4 == 1 else cgr))
                    a = radians(a0)
                    sg = clip_seg((ctr[0] + r_b * cos(a), ctr[1] + r_b * sin(a)), (ctr[0] + r_c * cos(a), ctr[1] + r_c * sin(a)), *n1)
                    sg = clip_seg(sg[0], sg[1], *n2) if sg else None
                    if sg:
                        came_line(*sg)
                # centre: four pieces between the lobes, leaded along the axes and the diagonals
                na = 5
                star = [(0.0, 0.0)]
                star += [lobe_pt(c, th, rf, radians(180.0 - (180.0 - a_in) * i / na)) for i in range(na + 1)]
                star += [lobe_pt(c, th + pi / 2, rf, radians(360.0 - a_in - (180.0 - a_in) * i / na)) for i in range(1, na + 1)]
                piece(star, cbo)
                came_arc(rf, a_in + thd, 360.0 - a_in + thd, ctr)
                came_rad(0.0, c - rf, thd)
                came_rad(hypot(*star[na + 1]), tcusp + 0.004, thd + 45.0)
                if eye:
                    eg = [rot2((b[0] + nv[0] * (rho + 0.006), b[1] + nv[1] * (rho + 0.006)), th) for b, nv in esamp]
                    ec = rot2(ecen, th)
                    L = len(eg)
                    for i in range(3):
                        i0, i1 = emids[i], emids[(i + 1) % 3]
                        idx = list(range(i0, i1 + 1)) if i1 > i0 else list(range(i0, L)) + list(range(0, i1 + 1))
                        piece([ec] + [eg[j] for j in idx], 4 if i == 1 else cgr)
                        came_line(ec, eg[i0])
        else:
            plain_glass(rg)
            if p.glazing == "leaded":
                lattice(rg)
            else:
                came_arc(0.3 * rf)
                for q in range(4):
                    came_rad(0.3 * rf, c + rf, q * 90)
                    came_rad(0.3 * rf, 0.5 * R, q * 90 + 45)
    else:
        # -------------------------------------------------------------- fixed frame + pivoting sash + glazing bars
        ff = 0.028 + 0.006 * D
        sf = 0.034 + 0.008 * D
        rso = R - ff - 0.004
        Rd = rso - sf                         # daylight radius
        fsec = [(R + 0.003, 0.0), (R + 0.003, FD - 0.002), (R + 0.001, FD), (R - ff + 0.008, FD), (R - ff + 0.002, FD - 0.003),
                (R - ff, FD - 0.008), (R - ff, 0.0)]
        ssec = [(rso, 0.004), (rso, SD - 0.002), (rso - 0.002, SD), (Rd + sf * 0.5, SD), (Rd + sf * 0.34, SD - 0.003),
                (Rd + sf * 0.16, SD - 0.011), (Rd + 0.004, SD - 0.016), (Rd, SD - 0.019), (Rd, DG)]
        qs = 10 if D < 1.3 else 12
        for q in range(4):
            am = radians(q * 90 + 90)
            for nm, s_, gp in (("frame", fsec, 0.03), ("sash", ssec, 0.05)):
                o = ring_seg(nm, s_, q * 90 + 45 + gp, q * 90 + 135 - gp, steps=qs, cham=0.0006)
                uv_grain(o, axis=(-sin(am), 0.0, cos(am)))
                smooth(o, 40)
                frame_objs.append(o)
        bw = 0.017 + 0.006 * D
        bf = DG + 0.02 + 0.006 * D            # bar front
        bsec = [(bw / 2, DB), (bw / 2, DG + 0.006), (bw * 0.24, bf - 0.004), (bw * 0.17, bf), (-bw * 0.17, bf), (-bw * 0.24, bf - 0.004),
                (-bw / 2, DG + 0.006), (-bw / 2, DB)]

        def bar_ring(r, name="bar_ring"):
            o = ring_seg(name, [(r + bw / 2, DB), (r + bw / 2, DG + 0.006), (r + bw * 0.24, bf - 0.004), (r + bw * 0.17, bf), (r - bw * 0.17, bf),
                                (r - bw * 0.24, bf - 0.004), (r - bw / 2, DG + 0.006), (r - bw / 2, DB)], steps=max(20, min(40, int(r * 90))))
            smooth(o, 40)
            frame_objs.append(o)

        def spoke(r0, r1, adeg):
            a = radians(adeg)
            frame_objs.append(bar("bar", (r0 * cos(a), r0 * sin(a)), (r1 * cos(a), r1 * sin(a)), bsec))

        rg = Rd + 0.008
        big = Rd > 0.5
        r_mid = Rd * 0.62
        if p.bars == "radial":
            ns = int(p.spokes)
            a_off = 90.0
            hub_ring = Rd > 0.26
            rh = max(0.07, Rd * 0.26) if hub_ring else max(0.024, bw * 0.62 / sin(pi / ns))
            for i in range(ns):
                spoke(rh - (0.0 if hub_ring else 0.01), Rd + 0.003, a_off + i * 360.0 / ns)
            if hub_ring:
                bar_ring(rh, "hub_ring")
            else:
                # low turned roundel where the bars meet
                h0_ = bf - DG
                frame_objs.append(lathe("boss", [(rh, 0.0), (rh, h0_ - 0.002), (rh * 0.9, h0_ + 0.003), (rh * 0.72, h0_ + 0.003),
                                                 (rh * 0.62, h0_ + 0.0065), (rh * 0.3, h0_ + 0.008), (0.0, h0_ + 0.008)],
                                        segments=18, loc=(0, -DG, 0), rot=(90, 0, 0)))
            if big:
                bar_ring(r_mid, "mid_ring")
                for i in range(ns):
                    spoke(r_mid, Rd + 0.003, a_off + (i + 0.5) * 360.0 / ns)
            zones = [(rh, r_mid, ns), (r_mid, rg, 2 * ns)] if big else [(rh, rg, ns)]
        else:
            a_off = 0.0
            hub_ring = False
            rh = 0.0
            frame_objs.append(bar("bar_v", (0, -Rd - 0.003), (0, Rd + 0.003), bsec))
            for sx in (-1, 1):
                frame_objs.append(bar("bar_h", (sx * bw * 0.5, 0), (sx * (Rd + 0.003), 0), bsec))
            if big:
                bar_ring(r_mid, "mid_ring")
            # small square boss planted on the crossing, set lozenge-wise, with a pin
            bs = max(0.03, bw * 1.5)
            frame_objs.append(box("boss", (bs, 0.007, bs), loc=(0, -(bf + 0.0025), 0), rot=(0, 45, 0), bevel=0.0025, segments=1))
            frame_objs.append(cylinder("boss_pin", 0.004, 0.004, loc=(0, -(bf + 0.007), 0), rot=(90, 0, 0), verts=8))
            zones = [(0.0, r_mid, 4), (r_mid, rg, 4)] if big else [(0.0, rg, 4)]

        # ---- glass
        if stained:
            wb = 0.045 + 0.012 * D              # border strip against the sash
            grounds = rgl.pick([[1, 0, 2], [0, 1, 2], [1, 2, 0], [0, 2, 1]])
            cacc = rgl.pick([3, 0])
            cbor = 0 if cacc == 3 else 3
            if hub_ring:
                if rh >= 0.13:
                    rc = rh * 0.42
                    mh = 8 if rh > 0.17 else 6
                    piece(psector(0, rc, 0, 360)[:-1], 3)
                    came_arc(rc)
                    for j in range(mh):
                        a0 = a_off + (j - 0.5) * 360.0 / mh
                        piece(psector(rc, rh, a0, a0 + 360.0 / mh), grounds[1] if j % 2 == 0 else 4)
                        came_rad(rc, rh - bw * 0.3, a0)
                else:
                    piece(psector(0, rh, 0, 360)[:-1], 3)
            bi = 0
            for zi, (ra, rb, nsect) in enumerate(zones):
                last = zi == len(zones) - 1
                rb_f = rb - wb if last else rb
                bands = []
                if rb_f - ra > 0.05:
                    nb_ = max(1, round((rb_f - ra) / tp))
                    bands = [(ra + (rb_f - ra) * i / nb_, ra + (rb_f - ra) * (i + 1) / nb_, False) for i in range(nb_)]
                else:
                    rb_f = ra
                if last:
                    bands.append((rb_f, rb, True))
                dsec = 360.0 / nsect
                for bn, (r0, r1, border) in enumerate(bands):
                    m = max(1, round(radians(dsec) * (r0 + r1) / 2 / (0.085 if border else tp)))
                    if bn > 0 and r0 > 0.01:
                        came_arc(r0)
                    ground = grounds[(bi // 2) % 3]
                    second = grounds[(bi // 2 + 1) % 3]
                    for i in range(nsect):
                        a0 = a_off + i * dsec
                        for j in range(m):
                            idx = j if i % 2 == 0 else m - 1 - j
                            if border:
                                ci = [4, cbor][idx % 2] if m > 1 else [4, cbor][i % 2]
                            elif bi % 2 == 0:
                                ci = [ground, 4, second, 4][idx % 4] if m > 1 else [ground, second][i % 2]
                            else:
                                ci = [4, cacc][idx % 2] if m > 1 else [4, cacc][i % 2]
                            piece(psector(r0, r1, a0 + dsec * j / m, a0 + dsec * (j + 1) / m), ci)
                            if j > 0:
                                came_rad(max(r0, 0.012), min(r1, Rd + 0.002), a0 + dsec * j / m)
                    if not border:
                        bi += 1
        else:
            plain_glass(rg)
            if p.glazing == "leaded":
                lattice(rg)

        # ---- brass furniture: pivots at 3 and 9 o'clock, turn-button at the crown, pull ring at the foot
        rp = R - ff - 0.002
        for sx in (-1, 1):
            brass_objs.append(box("pivot_plate", (ff * 0.8, 0.003, 0.034), loc=(sx * (R - ff * 0.52), -(FD + 0.0015), 0), bevel=0.0008))
            brass_objs.append(cylinder("pivot_pin", 0.0075, 0.012, loc=(sx * rp, -(FD - 0.001), 0), rot=(90, 0, 0), verts=12, bevel=0.0015, segments=1))
            for sz in (-1, 1):
                brass_objs.append(cylinder("screw", 0.0026, 0.002, loc=(sx * (R - ff * 0.45), -(FD + 0.004), sz * 0.011), rot=(90, 0, 0), verts=8))
        brass_objs.append(box("turn_button", (0.016, 0.004, ff + 0.022), loc=(0.0, -(FD + 0.002), R - ff * 0.5 - 0.012),
                              rot=(0, rng.jitter(0, 6), 0), bevel=0.0012))
        brass_objs.append(cylinder("button_screw", 0.005, 0.004, loc=(0.0, -(FD + 0.005), R - ff * 0.5), rot=(90, 0, 0), verts=10, bevel=0.001, segments=1))
        zr = -(Rd + sf * 0.55)
        brass_objs.append(cylinder("ring_staple", 0.006, 0.006, loc=(0.0, -(SD + 0.003), zr), rot=(90, 0, 0), verts=10, bevel=0.001, segments=1))
        brass_objs.append(torus("pull_ring", 0.014, 0.0024, loc=(0.0, -(SD + 0.0075), zr - 0.012), rot=(90 - 12, 0, 0), major_seg=16, minor_seg=6))

    # ------------------------------------------------------------------ finish: materials and weathering
    if frame_objs:
        for o in frame_objs:
            assign(o, framep)
        fj = join(frame_objs, "joinery")
        # paint fails first where water sits: the bottom rail of the sash and the foot of the frame
        paint(fj, wear=m_max(mask_mul(noise_mask(0.035, seed=sd + 21, lo=0.38, hi=0.62), height_ramp(-R, -R * 0.45), 1.5 * W),
                             mask_mul(noise_mask(0.05, seed=sd + 22, lo=0.6, hi=0.85), 0.35 * W)),
              special=mask_mul(height_ramp(R, -R * 0.2), 0.6 * W),
              grime=mask_mul(height_ramp(-R, -R * 0.5), 0.6 * W))
    if brass_objs:
        brs = mat("brass")
        for o in brass_objs:
            assign(o, brs)
        bj = join(brass_objs, "furniture")
        paint(bj, special=mask_mul(noise_mask(0.03, seed=sd + 2, lo=0.4, hi=0.7), 0.6 * W), wear=0.3)
    if leads:
        for o in leads:
            assign(o, lead)
        paint(join(leads, "cames"), grime=0.8)
    if cells:
        join(cells, "stained")

    if lit and p.intensity > 0.01:
        ld = pr + 0.5
        # the fitting must not shadow its own spill: the light stands for the room behind the glass
        light("spill", (0.06 * D, -ld, 0.05 * D), color=p.glass_tint, intensity=p.intensity, range=3.5 + 2.0 * D, flicker=0.05,
              shadow_near=hypot(Ro + 0.2, ld) + 0.05)
