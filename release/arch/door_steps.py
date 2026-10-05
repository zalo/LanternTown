"""Door stoop: a short flight of solid, foot-worn stone steps climbing to a landing at a house door.
Every course is its own stone (moulded bullnose nosing returned at the free ends, dished tread centre,
butt joints, knocked arrises) stacked on dressed side blocks. The sides carry either wrought-iron
handrails (square posts leaded into the treads, twisted balusters, flat handrail ending in a forged
volute) or coursed stone cheek walls with ramped copings and capped piers. A boot scraper leaded into
its own stone stands at the foot; moss cushions sit in the damp joints.
"""
from tk import *
import numpy as np
from math import sin, cos, radians, pi, atan, atan2, exp, ceil

META = {"name": "Door stoop steps", "class": "component", "mount": "wall_ground", "focus": [0.25, -0.75, 0.5]}
PARAMS = {
    "steps": {"default": 3, "min": 2, "max": 5, "desc": "number of risers; the top one is the landing (z = steps x rise)"},
    "width": {"default": 1.4, "min": 0.9, "max": 2.6, "desc": "clear width of the treads (m); cheek walls / splayed ends add outside it"},
    "landing": {"default": 0.9, "min": 0.5, "max": 1.6, "desc": "depth of the top landing from the wall (m)"},
    "rise": {"default": 0.17, "min": 0.15, "max": 0.19, "desc": "height of one step (m)"},
    "going": {"default": 0.30, "min": 0.26, "max": 0.38, "desc": "tread depth riser to riser (m)"},
    "rail": {"default": "iron", "choices": ["iron", "cheek", "none"], "desc": "wrought-iron handrail, stone cheek wall, or open sides"},
    "rail_side": {"default": "both", "choices": ["both", "left", "right"], "desc": "which side(s) get the rail / cheek"},
    "balusters": {"default": 2, "min": 0, "max": 3, "desc": "twisted iron balusters per tread (0 = posts and rails only)"},
    "cheek_height": {"default": 0.4, "min": 0.25, "max": 0.7, "desc": "height of the cheek wall above the landing / pitch line (m)"},
    "pier_cap": {"default": "pyramid", "choices": ["pyramid", "ball"], "desc": "cap of the cheek wall piers"},
    "plan": {"default": "straight", "choices": ["straight", "splayed"], "desc": "straight flight, or steps returned round three sides (ignored with cheeks)"},
    "nosing": {"default": "moulded", "choices": ["moulded", "square"], "desc": "bullnose-and-cove nosing or plain square-edged slabs"},
    "stone": {"default": "smooth", "choices": ["smooth", "rough"], "desc": "finish of the side blocks / cheek walls"},
    "tint": {"default": "#c6baa2", "desc": "stone tint"},
    "wear": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "foot wear: depth of the dished tread centres and polish"},
    "moss": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "moss in the joints and at the foot"},
    "scraper": {"default": "left", "choices": ["right", "left", "none"], "desc": "side of the wrought-iron boot scraper"},
    "lantern": {"default": True, "desc": "a lit candle lantern left standing on the landing / cheek coping"},
    "lantern_cd": {"default": 0.9, "min": 0.3, "max": 6.0, "desc": "candle lantern intensity (cd)"},
}
VARIANTS = {
    "cheek_walls": {"steps": 5, "width": 1.7, "landing": 1.1, "rail": "cheek", "pier_cap": "ball", "tint": "#d9c9a8",
                    "scraper": "right", "moss": 0.4, "seed": 4},
    "low_plain": {"steps": 2, "width": 1.15, "landing": 0.7, "rail": "none", "nosing": "square", "stone": "rough",
                  "tint": "#b2b0a8", "moss": 0.95, "wear": 0.95, "seed": 11},
    "one_rail_tall": {"steps": 4, "width": 1.05, "landing": 0.8, "rail": "iron", "rail_side": "right", "balusters": 0,
                      "stone": "rough", "tint": "#b39f86", "scraper": "right", "seed": 23},
    "splayed": {"steps": 3, "width": 1.5, "landing": 0.8, "plan": "splayed", "rail": "none", "tint": "#cdb898",
                "going": 0.33, "scraper": "none", "moss": 0.6, "seed": 31},
}


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _axis(a, b, cell, joints=(), extra=()):
    """Grid lines from a to b: regular cells, a 3-line open joint at every butt joint, plus extra lines.
    -> xs, joint flag, stone id."""
    spec = []
    for x in joints:
        spec += [(x - 0.004, 0), (x, 1), (x + 0.004, 0)]
    for x in sorted(extra):
        if a + 0.006 < x < b - 0.006 and all(abs(x - s[0]) > 0.007 for s in spec):
            spec.append((x, 0))
    n = max(1, int(ceil((b - a) / cell - 1e-6)))
    tol = min(0.02, cell * 0.3)
    pts = [(a + (b - a) * i / n, 0) for i in range(n + 1)]
    pts = [q for k, q in enumerate(pts) if k in (0, n) or all(abs(q[0] - s[0]) > tol for s in spec)]
    allp = sorted(pts + spec)
    xs = [q[0] for q in allp]
    fl = [q[1] for q in allp]
    js = sorted(joints)
    seg = [sum(1 for j in js if j < x - 1e-6) for x in xs]
    return xs, fl, seg


def _slab(name, x0, x1, yf, yb, ztop, h, ret_l, ret_r, kind, cell, jx, jy, lipf, tiltf, dish, wearx, rng,
          hole=None, ex_x=(), ex_y=(), pen_s=-0.0015):
    """One course of the stair: top grid (dished, open butt joints, lipped) + nosing profile run round the
    free edges. x0 / x1 / yf are the riser planes; the nosing oversails them. kind = moulded | square | rock
    (rock = rough-hewn: tight arris, drafted margin, rock-faced riser). The riser foot is let `pen` into the
    tread below so no light or view passes under it. Own UVs: every stone gets its own texture orientation."""
    arc = lambda r: [(r * sin(radians(a)), -r * (1 - cos(radians(a))), "a") for a in (30, 60, 90)]
    pen = 0.012
    if kind == "moulded":
        r, ov, tn, r2, cv = 0.02, 0.03, 0.052, 0.006, 0.012
        prof = arc(r) + [(r, -(tn - r2), "n"), (r - r2, -tn, "n"), (r - ov + cv, -tn, "c"),
                         (r - ov + cv * 0.293, -tn - cv * 0.293, "c"), (r - ov, -tn - cv, "r"),
                         (r - ov, -h + 0.03, "r"), (r - ov, -h, "b")]
    elif kind == "square":
        r, ov = 0.014, 0.0
        prof = arc(r) + [(r, -0.05, "n"), (r, -h + 0.03, "r"), (r, -h, "b")]
    else:
        r, ov = 0.008, 0.0
        prof = arc(r) + [(r, -0.026, "n"), (r, -h * 0.42, "k"), (r, -h * 0.7, "k"), (r, -h + 0.022, "r"), (r, -h, "b")]
    gx0 = x0 - (ov - r) if ret_l else x0
    gx1 = x1 + (ov - r) if ret_r else x1
    gy0 = yf - (ov - r)
    # bites knocked out of the nosing: each gets its own three grid lines so it stays 25-60 mm wide
    chips = []
    for _ in range(int(round((gx1 - gx0) * rng.u(1.2, 2.4)))):
        c = rng.u(gx0 + 0.07, gx1 - 0.07)
        if all(abs(c - q) > 0.07 for q in list(jx) + list(ex_x)) and all(abs(c - q[0]) > 0.1 for q in chips):
            chips.append((c, rng.u(0.01, 0.026) * (1.3 if kind == "rock" else 1.0), rng.u(0.012, 0.032), rng.u(0.012, 0.032)))
    ex = list(ex_x) + ([gx0 + 0.016] if ret_l else []) + ([gx1 - 0.016] if ret_r else [])
    for (c, d, a, b) in chips:
        ex += [c - a, c, c + b]
    xs, fx, sx = _axis(gx0, gx1, cell, jx, ex)
    ys, fy, _ = _axis(gy0, yb, cell * 1.2, jy, [gy0 + 0.016] + list(ex_y))
    nx, ny = len(xs), len(ys)
    vid = lambda i, j: j * nx + i
    V, TW = [], []
    for j in range(ny):
        for i in range(nx):
            V.append([xs[i], ys[j], ztop - 0.009 * max(fx[i], fy[j])])
            TW.append(1.0)
    per = []
    if ret_l:
        per += [(0, j, -1, 0) for j in range(ny - 1, 0, -1)] + [(0, 0, -1, -1)]
    else:
        per.append((0, 0, 0, -1))
    per += [(i, 0, 0, -1) for i in range(1, nx - 1)]
    if ret_r:
        per += [(nx - 1, 0, 1, -1)] + [(nx - 1, j, 1, 0) for j in range(1, ny)]
    else:
        per.append((nx - 1, 0, 0, -1))
    npr = len(per)
    chip = [0.0] * npr

    def knock(k, d, spread):
        chip[k] = max(chip[k], d)
        for dk in (-1, 1):
            if 0 <= k + dk < npr:
                chip[k + dk] = max(chip[k + dk], d * spread)

    for k, (i, j, ox, oy) in enumerate(per):
        if ox != 0 and oy != 0:
            if rng.chance(0.75):
                knock(k, rng.u(0.014, 0.034), 0.3)              # knocked corner of the return
        elif ox == 0:
            for (c, d, a, b) in chips:
                if abs(xs[i] - c) < 1e-6:
                    chip[k] = max(chip[k], d)
    for _ in range(int(npr * (0.2 if kind == "rock" else 0.1))):  # small bruises along the arris
        knock(int(rng.u(0, npr - 0.01)), rng.u(0.002, 0.008), 0.35)
    base = len(V)
    m = len(prof)
    for k, (i, j, ox, oy) in enumerate(per):
        corner = ox != 0 and oy != 0
        front = ox == 0
        g = 0 if corner else (fx[i] if front else fy[j])
        wr = wearx(xs[i]) if front else 0.0
        V[vid(i, j)][2] -= chip[k] * 0.3
        for (o, dz, kd) in prof:
            tw = clamp(1 + dz / 0.05)
            off = o * (0.62 if corner and o > 0 else 1.0)
            zz = ztop + dz
            if kd in "an":
                f = 1.0 if kd == "a" else 0.55
                off = max(off - chip[k] * f - wr * 0.007 * (o / r), min(off, 0.002))      # knocked / worn-back nose
                zz -= (chip[k] * 0.7 + wr * 0.009 * (o / r)) * tw                            # ... and worn down
            elif kd == "k" and not (g or corner):
                off += rng.u(0.0, 0.016)                                                    # rock face
            elif kd == "b":
                zz -= pen if front else pen_s
            off -= 0.004 * g
            zz -= 0.009 * g * tw
            V.append([xs[i] + ox * off, ys[j] + oy * off, zz])
            TW.append(tw)
    for q in range(len(V)):
        x, y = V[q][0], V[q][1]
        V[q][2] = max(0.0, V[q][2] + TW[q] * (lipf(x) - float(dish(x, y))) + tiltf(x))
    # UVs (metres): each stone between butt joints is its own piece of rock with its own orientation
    tfm = {}

    def tuv(key, u, v):
        if key not in tfm:
            # the library stone has a ~25 deg diagonal vein: lay it along the tread, stand it up as tooling on the riser
            a = radians((rng.pick([-25, 155]) + rng.jitter(0, 8)) if key[0] == "t" else (rng.pick([-95, 85]) + rng.jitter(0, 7)))
            tfm[key] = (cos(a), sin(a), rng.u(0, 5), rng.u(0, 5))
        c, s, ou, ov_ = tfm[key]
        return (c * u - s * v + ou, s * u + c * v + ov_)

    seg = lambda x: sum(1 for q in jx if q < x)
    UL = [0.0]
    for k in range(1, npr):
        a, b = per[k - 1], per[k]
        UL.append(UL[-1] + abs(xs[b[0]] - xs[a[0]]) + abs(ys[b[1]] - ys[a[1]]))
    VL = [0.0]
    q0 = (0.0, 0.0)
    for (o, dz, kd) in prof:
        VL.append(VL[-1] + ((o - q0[0]) ** 2 + (dz - q0[1]) ** 2) ** 0.5)
        q0 = (o, dz)
    F, FU = [], []
    for j in range(ny - 1):
        for i in range(nx - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if hole is not None and hole[0] < cx < hole[1] and cy > hole[2]:
                continue
            f = (vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1))
            key = ("t", seg(cx) + 7 * sum(1 for q in jy if q < cy))
            F.append(f)
            FU.append([tuv(key, V[v][0], V[v][1]) for v in f])
    rv = lambda k, l: vid(per[k][0], per[k][1]) if l == 0 else base + k * m + (l - 1)
    for k in range(npr - 1):
        key = ("r", seg((xs[per[k][0]] + xs[per[k + 1][0]]) / 2))
        for l in range(m):
            F.append((rv(k, l), rv(k, l + 1), rv(k + 1, l + 1), rv(k + 1, l)))
            FU.append([tuv(key, UL[kk], -VL[ll]) for kk, ll in ((k, l), (k, l + 1), (k + 1, l + 1), (k + 1, l))])
    for kk, rev in (((0, True),) if ret_l else ()) + (((npr - 1, False),) if ret_r else ()):
        f = tuple(rv(kk, l) for l in (range(m, -1, -1) if rev else range(m + 1)))
        F.append(f)
        FU.append([tuv(("c", 0), V[v][0], V[v][2]) for v in f])
    used = sorted({v for f in F for v in f})
    remap = {v: i for i, v in enumerate(used)}
    o = mesh_object(name, [V[v] for v in used], [tuple(remap[v] for v in f) for f in F])
    look = {}
    for f, uv in zip(F, FU):
        nf = [remap[v] for v in f]
        look[tuple(sorted(nf))] = dict(zip(nf, uv))
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        d = look.get(tuple(sorted(poly.vertices)))
        if d:
            for li in poly.loop_indices:
                uvl.data[li].uv = d[me.loops[li].vertex_index]
    uv_keep(o)
    smooth(o, 28 if kind == "rock" else 50)
    displace_noise(o, strength=0.001, scale=0.3, seed=int(rng.u(0, 999)))
    transform_verts(o, lambda co: Vector((co.x, min(co.y, 0.0), max(co.z, 0.0))))
    return o


class _Strips:
    """Broken, lumpy runs of built-up material (moss cushions, silt) along joints and internal angles."""

    def __init__(self, r):
        self.V, self.F, self.r = [], [], r

    def line(self, P0, P1, out, up, cover, wr, hr, angle=True, step=0.03, sink=0.003, zf=None, wt=None):
        """angle=True: fillet in the corner between the surface (normal `up`) and a face behind (`out` points
        away from that face). angle=False: a cushion lying over a joint in a flat face (`out` = across it)."""
        r = self.r
        P0, P1, out, up = Vector(P0), Vector(P1), Vector(out), Vector(up)
        L = (P1 - P0).length
        if L < 0.04 or cover <= 0:
            return
        d = (P1 - P0) / L
        prof = [(1.0, 0.0), (0.6, 0.66), (0.18, 1.0), (-0.08, 0.86)] if angle else [(-1.0, 0.0), (-0.42, 0.9), (0.4, 1.0), (1.0, 0.0)]
        mid = out * (prof[2][0] - prof[1][0]) + up * (prof[2][1] - prof[1][1])
        flip = d.cross(mid).dot(up + out if angle else up) < 0
        ccw = out.cross(up).dot(d) * (1 if angle else -1) > 0
        t = r.u(-0.08, 0.05)
        while t < L:
            Pm = P0 + d * clamp(t, 0.0, L)
            c = clamp(cover * (float(wt(Pm.x, Pm.y)) if wt else 1.0), 0.0, 0.95)
            if c < 0.05:
                t += 0.12
                continue
            run = r.u(0.05, 0.3) * (0.5 + c)
            a, b = max(t, 0.0), min(t + run, L)
            t += run + max(0.012, run * (1 - c) / c * r.u(0.5, 1.5))
            if b - a < 0.035:
                continue
            n = max(2, int(round((b - a) / step)))
            w0, h0 = r.u(*wr), r.u(*hr)
            base = len(self.V)
            for i in range(n + 1):
                s = i / n
                env = 0.2 + 0.8 * sin(pi * s) ** 0.6
                w, h = w0 * env * r.u(0.72, 1.12), h0 * env * r.u(0.7, 1.1)
                P = P0 + d * (a + (b - a) * s)
                if zf:
                    P.z = float(zf(P.x, P.y))
                for (pa, pb) in prof:
                    self.V.append(P + out * (pa * w) + up * (pb * (h + sink) - sink))
            for i in range(n):
                for l in range(3):
                    q = (base + i * 4 + l, base + (i + 1) * 4 + l, base + (i + 1) * 4 + l + 1, base + i * 4 + l + 1)
                    self.F.append(q[::-1] if flip else q)
            e0, e1 = (base, base + 1, base + 2, base + 3), tuple(base + n * 4 + l for l in range(4))
            self.F.append(e0[::-1] if ccw else e0)
            self.F.append(e1 if ccw else e1[::-1])

    def build(self, name, material, angle=70):
        if not self.F:
            return None
        o = mesh_object(name, [tuple(v) for v in self.V], self.F)
        transform_verts(o, lambda co: Vector((co.x, min(co.y, 0.0), max(co.z, 0.0))))
        assign(o, material)
        smooth(o, angle)
        return o


def _clip(poly, a, b, c):
    """Clip a convex polygon [(y, z)] to a*y + b*z <= c."""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        dp, dq = a * P[0] + b * P[1] - c, a * Q[0] + b * Q[1] - c
        if dp <= 0:
            out.append(P)
        if (dp < 0 < dq) or (dq < 0 < dp):
            t = dp / (dp - dq)
            out.append((P[0] + (Q[0] - P[0]) * t, P[1] + (Q[1] - P[1]) * t))
    res = []
    for q in out:
        if not res or abs(q[0] - res[-1][0]) + abs(q[1] - res[-1][1]) > 1e-4:
            res.append(q)
    if len(res) > 1 and abs(res[0][0] - res[-1][0]) + abs(res[0][1] - res[-1][1]) <= 1e-4:
        res.pop()
    return res


def _area(poly):
    n = len(poly)
    return abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2 if n > 2 else 0.0


def _prism_x(name, poly, xa, xb, bev=0.006, seg=1):
    """Stone cut to a side profile [(y, z)] and extruded across X, arrises chamfered."""
    n = len(poly)
    V = [(xa, y, z) for y, z in poly] + [(xb, y, z) for y, z in poly]
    F = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    F += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    o = mesh_object(name, V, F)
    recalc_normals(o)
    if bev > 0:
        bevel(o, bev, segments=seg)
    return o


def _scroll(E, a, turn, R0, R1, turns, n):
    """2D volute leaving point E with heading a (rad), curling to the left (turn=1) or right (-1)."""
    dx, dy = cos(a), sin(a)
    nx, ny = -turn * dy, turn * dx
    C = (E[0] + R0 * nx, E[1] + R0 * ny)
    a0 = atan2(-ny, -nx)
    pts = []
    for i in range(n + 1):
        t = i / n
        ph = a0 + turn * 2 * pi * turns * t
        R = R0 + (R1 - R0) * t ** 0.85
        pts.append((C[0] + R * cos(ph), C[1] + R * sin(ph)))
    return pts


def build(p, rng):
    N, H, G, W, Ld = int(p.steps), p.rise, p.going, p.width, p.landing
    mp = H / G                                   # pitch
    th = atan(mp)
    ztop = N * H
    yf0 = -(Ld + (N - 1) * G)                    # riser plane of the bottom step
    moulded = p.nosing == "moulded"
    rough = p.stone == "rough"
    kind = "moulded" if moulded else ("rock" if rough else "square")
    ov = 0.03 if moulded else 0.0
    rk = {"moulded": 0.02, "square": 0.014, "rock": 0.008}[kind]
    on = {-1: p.rail_side in ("both", "left"), 1: p.rail_side in ("both", "right")}
    cheek = {s: p.rail == "cheek" and on[s] for s in (-1, 1)}
    ironr = {s: p.rail == "iron" and on[s] for s in (-1, 1)}
    splay = p.plan == "splayed" and p.rail != "cheek"
    lan_s = -1 if p.scraper == "right" else 1    # the lantern stands on this side ...
    damp = -lan_s                                # ... the other one stays dark and damp

    st_m = mat("stone_smooth", p.tint)
    sub_m = mat("stone_rough", p.tint) if rough else st_m
    slab_m = sub_m if kind == "rock" else st_m
    iron_m = mat("iron_wrought", wear=2.2)
    lead_m = mat("lead")
    dirt_m = mat("mortar", "#8a7862")            # silt in the angles, bedding seen through the open joints
    rust_m = mat("mortar", "#7a5232")            # rust washed out of the post sockets
    stones, subs, irons, leads = [], [], [], []

    cx0 = rng.jitter(0.0, W * 0.04)              # the line people walk
    aw = min(0.30 * W, 0.5)                      # half width of the worn hollow: the middle ~60 %
    depth = 0.004 + 0.03 * p.wear
    profx = lambda x, cx: np.exp(-(np.abs(x - cx) / aw) ** 2.6)

    def dish_for(yfront, ynext, cx, landing):
        def f(x, y):
            wy = 0.5 + 0.5 * np.exp(-np.maximum(0.0, y - yfront) / 0.12)       # deepest at the nosing
            if landing:
                wy = np.maximum(wy, 0.75 * np.exp(-((y + 0.25) / 0.28) ** 2))  # where people stand at the door
            return depth * profx(x, cx) * wy * _ss(0.02, 0.09, ynext - y)       # nobody treads in the internal angle
        return f

    # ------------------------------------------------------------------ the courses
    load = N * (W + (N * G if splay else 0.0)) / 4.2          # big flights get a coarser tread grid (triangle budget)
    cell = max(0.065 if kind == "rock" else (0.12 if splay else 0.1), W / 18) \
        * (clamp(load, 1.0, 2.6) ** 0.75 if (ironr[-1] or ironr[1]) else clamp(load / (2.0 if splay else 1.65), 1.0, 2.2))
    nbal = (int(ironr[-1]) + int(ironr[1])) * int(p.balusters) * ((N - 1) + Ld / G)
    tstep = 0.042 * clamp(nbal / 16.0, 1.0, 2.5)
    ovl = 0.035
    tc = 0.22                                     # cheek wall thickness
    C = []                                        # per course: geometry facts the weathering needs
    side_joints = []                              # (P0, P1, across, normal) open joints of the side blocks
    for j in range(N):
        r = rng.fork(f"course{j}")
        top = (j + 1) * H
        yf = -(Ld + (N - 1 - j) * G)
        hw = W / 2 + ((N - 1 - j) * G if splay else 0.0)
        last = j == N - 1
        yb = 0.0 if (last or splay) else yf + G + ovl
        ynext = 0.0 if last else yf + G
        x0 = -hw - (0.02 if cheek[-1] else 0.0)
        x1 = hw + (0.02 if cheek[1] else 0.0)
        wj = x1 - x0
        npc = max(1, int(round(wj / 0.95)))
        jx = [x0 + wj * k / npc + r.jitter(0.0, 0.16 * wj / npc) for k in range(1, npc)]
        jy, hole, ex_x, ex_y = [], None, [], ([-0.026] if last else [ynext - 0.024, ynext])
        if splay and not last:
            hwu = hw - G
            jx = [x for x in jx if abs(x) < hwu - 0.12]
            span = -(yf + G)
            nj = int(span / 0.9)
            jy = [yf + G + span * k / (nj + 1) + r.jitter(0.0, 0.08) for k in range(1, nj + 1)]
            hole = (-(hwu - 0.03), hwu - 0.03, ynext + 0.03)
            ex_x = [e * (hwu + d) for e in (-1, 1) for d in (-0.03, 0.0, 0.024)]
            ex_y = [ynext - 0.024, ynext, ynext + 0.03, -0.026]
        elif last and Ld > 1.15:
            jy = [-Ld * r.u(0.42, 0.58)]
        jx.sort()
        lips = [r.jitter(0.0, 0.0025) for _ in range(len(jx) + 1)]            # stones never bed dead flush
        lipf = lambda x, jx=jx, lips=lips: lips[sum(1 for q in jx if q < x - 1e-6)]
        if j == 0:                                                            # the bottom course has settled at one end
            ta, tsg = r.u(0.003, 0.0065), r.pick([-1, 1])
            tiltf = lambda x, ta=ta, tsg=tsg, hw=hw: -ta * clamp(0.5 + tsg * x / (2 * hw))
        else:
            tiltf = lambda x: 0.0
        cx = cx0 + r.jitter(0.0, 0.04)
        dish = dish_for(yf, ynext, cx, last)
        o = _slab(f"course{j}", x0, x1, yf, yb, top, H, not cheek[-1], not cheek[1], kind, cell, jx, jy, lipf, tiltf,
                  dish, lambda x, cx=cx: float(profx(x, cx)) * p.wear, r, hole, ex_x, ex_y, pen_s=0.012 if splay else -0.0015)
        assign(o, slab_m)
        C.append({"o": o, "top": top, "yf": yf, "hw": hw, "ynext": ynext, "last": last, "jx": jx, "dish": dish, "cx": cx,
                  "zf": lambda x, y, top=top, lipf=lipf, tiltf=tiltf, dish=dish: top + lipf(x) + tiltf(x) - float(dish(x, y))})
        # solid core: bedding seen through the open joints, and nothing (light, view) passes under a riser
        subs_core = box(f"core{j}", (2 * (hw - 0.035), -(yf + 0.035), top - 0.05), loc=(0.0, (yf + 0.035) / 2, (top - 0.05) / 2))
        assign(subs_core, dirt_m)
        # dressed blocks closing the open sides under the upper courses, laid with open joints
        for s in (-1, 1):
            if splay or last or cheek[s]:
                continue
            y = yb
            k = 0
            z0, z1 = (0.0 if j == 0 else top - H + 0.0015), top - 0.0015
            if j > 0:
                side_joints.append(((s * hw, yb, top - H), (s * hw, -0.01, top - H), (0, 0, 1), (s, 0, 0)))
            while y < -0.01:
                y2 = y + r.u(0.34, 0.6) * (0.6 if (k == 0 and j % 2) else 1.0)
                if y2 > -0.2:
                    y2 = 0.0
                bx = 0.15
                loc = (s * (hw - bx / 2 + r.jitter(0.0, 0.0025)), (y + y2) / 2, (z0 + z1) / 2)
                size = (bx, y2 - y - 0.003, z1 - z0)
                if rough:
                    b = stone_block(f"side{j}_{s}_{k}", size, loc=loc, rng=r, chamfer=0.007, rough=0.005, detail=0.1 if N <= 2 else 0.2)
                    transform_verts(b, lambda co, zz=(z1 - z0) / 2: Vector((co.x, co.y, max(co.z, -zz))))
                else:
                    b = box(f"side{j}_{s}_{k}", size, loc=loc, bevel=0.005, segments=1)
                subs.append(b)
                if y2 < -0.01:
                    side_joints.append(((s * hw, y2, z0), (s * hw, y2, z1), (0, 1, 0), (s, 0, 0)))
                y = y2
                k += 1

    # ------------------------------------------------------------------ stone cheek walls
    ps = tc + 0.06                                # pier side
    ypf = yf0 - 0.10                              # pier front face
    pier_top = {}

    def build_cheek(s, r):
        tv = 0.075                                # coping thickness
        xin, xout = s * W / 2, s * (W / 2 + tc)
        zc = ztop + p.cheek_height - tv           # top of the walling under the level coping
        ypb = ypf + ps
        bot = lambda y: zc + min(0.0, y + Ld) * mp
        hp = bot(ypb) + tv + 0.07
        nc = max(2, int(round(zc / 0.2)))
        hc = zc / nc
        for c in range(nc):
            z0, z1 = c * hc, (c + 1) * hc
            y_end = max(ypb - 0.03, -Ld + (z0 - zc) / mp)
            ys = [0.0]
            first = True
            while True:
                L = r.u(0.38, 0.62) * (0.55 if (first and c % 2) else 1.0)
                first = False
                yn = ys[-1] - L
                if yn - 0.22 < y_end:
                    ys.append(y_end)
                    break
                ys.append(yn)
            for k in range(len(ys) - 1):
                ya, yb_ = ys[k + 1], ys[k]
                poly = _clip([(ya, z0), (yb_, z0), (yb_, z1), (ya, z1)], -mp, 1.0, zc + mp * Ld)
                if _area(poly) < 0.003:
                    continue
                xa, xb = sorted((xin + s * r.jitter(0.0, 0.0015), xout + s * r.jitter(0.0, 0.004)))
                subs.append(_prism_x(f"cheek{s}_{c}_{k}", poly, xa, xb, bev=0.007))
        # copings: level over the landing, ramped down the flight, dying into the pier
        xa, xb = sorted((xin - s * 0.03, xout + s * 0.03))
        runs = [(0.0, -Ld, max(1, int(round(Ld / 0.75))))]
        Ls = -Ld - (ypb - 0.02)
        if Ls > 0.05:
            runs.append((-Ld, ypb - 0.02, max(1, int(round(Ls / 0.7)))))
        for (ya, yb_, n) in runs:
            for k in range(n):
                a_, b_ = ya + (yb_ - ya) * k / n, ya + (yb_ - ya) * (k + 1) / n
                dz = r.jitter(0.0, 0.002)
                poly = [(b_, bot(b_) + dz), (a_, bot(a_) + dz), (a_, bot(a_) + tv + dz), (b_, bot(b_) + tv + dz)]
                stones.append(_prism_x(f"coping{s}_{k}", poly, xa, xb, bev=0.014, seg=2))
        # pier with a moulded cap
        xc, yc = s * (W / 2 + tc / 2), ypf + ps / 2
        pier = stone_block(f"pier{s}", (ps, ps, hp), loc=(xc, yc, hp / 2), rng=r, chamfer=0.012, rough=0.0025, detail=0.16)
        transform_verts(pier, lambda co: Vector((co.x, co.y, max(co.z, -hp / 2))))
        subs.append(pier)
        R = (ps / 2 + 0.032) * 1.4142
        if p.pier_cap == "ball":
            cap = lathe(f"pier_cap{s}", [(0.0, 0.0), (R, 0.0), (R, 0.04), (R - 0.022, 0.055), (R - 0.03, 0.075), (0.0, 0.085)],
                        segments=4, loc=(xc, yc, hp), rot=(0, 0, 45), smooth_angle=25)
            ball = lathe(f"pier_ball{s}", [(0.05, 0.08), (0.04, 0.095), (0.038, 0.11), (0.06, 0.125), (0.082, 0.16), (0.09, 0.2),
                                           (0.082, 0.24), (0.06, 0.272), (0.03, 0.288), (0.0, 0.292)], segments=18, loc=(xc, yc, hp))
            stones.extend([cap, ball])
            pier_top[s] = (xc, yc, hp + 0.29)
        else:
            cap = lathe(f"pier_cap{s}", [(0.0, 0.0), (R, 0.0), (R, 0.04), (R - 0.022, 0.058), (0.05, 0.135), (0.0, 0.14)],
                        segments=4, loc=(xc, yc, hp), rot=(0, 0, 45), smooth_angle=25)
            stones.append(cap)
            pier_top[s] = (xc, yc, hp + 0.14)

    for s in (-1, 1):
        if cheek[s]:
            build_cheek(s, rng.fork(f"cheek{s}"))

    # ------------------------------------------------------------------ wrought-iron handrails
    hsec = rounded_rect(0.038, 0.014, 0.005, 1)
    lsec = rounded_rect(0.024, 0.009, 0.003, 1)
    sq = lambda h: [(-h, -h), (h, -h), (h, h), (-h, h)]

    post_feet = []                                # (x, y, z) of every post leaded into a tread

    def build_rail(s, r):
        xr = s * (W / 2 - 0.085)
        hr = 0.95
        zl = ztop + hr
        ytp, ybp = -Ld + 0.10, yf0 + 0.10
        zr = lambda y: zl + min(0.0, y - ytp) * mp
        zlo = lambda y: zr(y) - (hr - 0.17)
        d3 = lambda q: (xr, q[0], q[1])
        # handrail: wall -> level -> eased knee -> down the pitch -> past the newel -> volute
        kf = 0.07
        A, K, B = (ytp + kf, zl), (ytp, zl), (ytp - kf * cos(th), zl - kf * sin(th))
        path = [(0.0, zl), A]
        for t in (0.2, 0.4, 0.6, 0.8):
            path.append(((1 - t) ** 2 * A[0] + 2 * t * (1 - t) * K[0] + t * t * B[0],
                         (1 - t) ** 2 * A[1] + 2 * t * (1 - t) * K[1] + t * t * B[1]))
        yE = ybp - 0.09
        path += [B, (yE, zr(yE))]
        irons.append(sweep(f"handrail{s}", [d3(q) for q in path], hsec, smooth_path=False))
        sc = _scroll((yE, zr(yE)), pi + th, 1, 0.07, 0.014, 1.3, 21)
        irons.append(sweep(f"volute{s}", [d3(q) for q in sc], hsec, smooth_path=False, scale_fn=lambda t: 1.0 - 0.5 * t))
        # lower rail, ending in a small up-curled scroll in front of the newel
        yL = ybp - 0.05
        irons.append(sweep(f"lowrail_a{s}", [d3((0.0, zlo(0.0))), d3((ytp, zlo(ytp)))], lsec, smooth_path=False))
        irons.append(sweep(f"lowrail_b{s}", [d3((ytp, zlo(ytp))), d3((yL, zlo(yL)))], lsec, smooth_path=False))
        sc = _scroll((yL, zlo(yL)), pi + th, -1, 0.048, 0.011, 1.2, 16)
        irons.append(sweep(f"lowscroll{s}", [d3(q) for q in sc], lsec, smooth_path=False, scale_fn=lambda t: 1.0 - 0.45 * t))
        # wall fixings
        irons.append(cylinder(f"rosette{s}", 0.042, 0.008, loc=(xr, -0.004, zl), rot=(90, 0, 0), verts=12, bevel=0.002, segments=1))
        irons.append(cylinder(f"rosette_lo{s}", 0.026, 0.008, loc=(xr, -0.004, zlo(0.0)), rot=(90, 0, 0), verts=10, bevel=0.002, segments=1))
        # posts leaded into the stone
        posts = [(ybp, H, 0.027), (ytp, ztop, 0.027)]
        if -ytp > 1.05:
            posts.append((ytp * 0.5, ztop, 0.022))
        nfl = int((ytp - ybp) / 0.95)
        for k in range(1, nfl + 1):
            y = ybp + (ytp - ybp) * k / (nfl + 1)
            posts.append((y, H * (1 + round((y - ybp + 0.10 - 0.5 * G) / G)), 0.022))
        riv = []
        for k, (y, zb, sz) in enumerate(posts):
            post_feet.append((xr, y, zb))
            zt = zr(y) - 0.005
            irons.append(box(f"post{s}_{k}", (sz, sz, zt - zb + 0.02), loc=(xr, y, (zt + zb - 0.02) / 2), bevel=0.002, segments=1))
            irons.append(box(f"post_foot{s}_{k}", (sz + 0.014, sz + 0.014, 0.018), loc=(xr, y, zb + 0.026), bevel=0.004, segments=1))
            irons.append(box(f"post_collar{s}_{k}", (sz + 0.01, sz + 0.01, 0.014), loc=(xr, y, zlo(y)), bevel=0.003, segments=1))
            leads.append(box(f"lead{s}_{k}", (sz + 0.034, sz + 0.034, 0.014), loc=(xr, y, zb + 0.001), bevel=0.004, segments=1))
            riv.append(trs((xr, y, zr(y) + 0.0065), (th * 180 / pi if y < ytp - 0.01 else 0.0, 0, 0)))
        rb = bolt("rail_rivet", 0.0065, 0.004, kind="dome")
        irons.append(instances_to_mesh(f"rail_rivets{s}", rb, riv, keep_base=False))
        # twisted square balusters between the rails
        nb = int(p.balusters)
        if nb > 0:
            sp = G / nb
            ys = []
            for (a_, b_) in ((ybp, ytp), (ytp, 0.0)):
                taken = [q[0] for q in posts if a_ < q[0] < b_]
                n = max(0, int(round((b_ - a_) / sp)) - 1)
                for k in range(1, n + 1):
                    y = a_ + (b_ - a_) * k / (n + 1)
                    if all(abs(y - t) > 0.06 for t in taken):
                        ys.append(y)
            for k, y in enumerate(ys):
                z0, z1 = zlo(y) + 0.003, zr(y) - 0.005
                za, zb = z0 + (z1 - z0) * 0.24, z0 + (z1 - z0) * 0.66
                q = sq(0.0065)
                irons.append(sweep(f"bal{s}_{k}a", [(xr, y, z0), (xr, y, za)], q, up=(0, 1, 0), smooth_path=False))
                irons.append(sweep(f"bal{s}_{k}b", [(xr, y, za), (xr, y, zb)], q, up=(0, 1, 0), step=tstep, twist_deg=360, cap=False))
                irons.append(sweep(f"bal{s}_{k}c", [(xr, y, zb), (xr, y, z1)], q, up=(0, 1, 0), smooth_path=False))

    for s in (-1, 1):
        if ironr[s]:
            build_rail(s, rng.fork(f"rail{s}"))

    # ------------------------------------------------------------------ boot scraper
    hw0 = W / 2 + ((N - 1) * G if splay else 0.0)
    xside = {s: hw0 + (tc if cheek[s] else 0.0) for s in (-1, 1)}            # outer face of the flight at the foot
    scr_irons, scr_box = [], None
    if p.scraper != "none":
        # a long kerb-like stone bedded tight against the side of the flight, sunk into the paving
        s = 1 if p.scraper == "right" else -1
        r = rng.fork("scraper")
        bw, bd, bh, sunk = 0.2, 0.46, 0.05, 0.028
        xc = s * (xside[s] + 0.005 + bw / 2)
        yc = min((ypf + ps + 0.015 if cheek[s] else yf0 + 0.04) + bd / 2, -(0.012 + bd / 2))
        base = stone_block("scraper_base", (bw, bd, bh + sunk), loc=(xc, yc, (bh - sunk) / 2), rng=r, chamfer=0.012, rough=0.003, detail=0.1)
        transform_verts(base, lambda co: Vector((co.x, co.y, max(co.z, -(bh - sunk) / 2))))
        subs.append(base)
        scr_box = (xc - bw / 2, yc - bd / 2, xc + bw / 2, yc + bd / 2)
        zu = bh + 0.2
        fsec = rounded_rect(0.008, 0.017, 0.002, 1)
        for e in (-1, 1):
            yu = yc + e * 0.145
            scr_irons.append(box(f"scr_post{e}", (0.017, 0.017, zu - bh + 0.03), loc=(xc, yu, (zu + bh - 0.03) / 2), bevel=0.002, segments=1))
            sc = _scroll((yu, zu - 0.01), pi / 2, -e, 0.034, 0.008, 1.2, 14)
            scr_irons.append(sweep(f"scr_scroll{e}", [(xc, q[0], q[1]) for q in sc], fsec, smooth_path=False, up=(1, 0, 0),
                                   scale_fn=lambda t: 1.0 - 0.45 * t))
            leads.append(box(f"scr_lead{e}", (0.045, 0.045, 0.012), loc=(xc, yu, bh), bevel=0.004, segments=1))
            scr_irons.append(bolt(f"scr_rivet{e}", 0.007, 0.004, loc=(xc, yu + e * 0.0085, bh + 0.105), rot=(-e * 90, 0, 0), kind="dome"))
        blade = box("scr_blade", (0.006, 0.29, 0.055), loc=(xc, yc, bh + 0.105), bevel=0.0012, segments=1)
        transform_verts(blade, lambda co: Vector((co.x, co.y, co.z - (0.007 * (1 - (co.y / 0.145) ** 2) if co.z > 0 else 0.0))))
        scr_irons.append(blade)

    # ------------------------------------------------------------------ candle lantern left by the door
    glass_objs, wax_objs, flame_objs = [], [], []
    if p.lantern:
        r = rng.fork("lantern")
        s = lan_s
        if cheek[s]:
            lx, ly, lz = s * (W / 2 + tc / 2), -r.u(0.3, 0.42), ztop + p.cheek_height
        else:
            open_ = p.rail == "none"                 # on an open stoop it stands near the edge and lights the flight
            lx, ly, lz = s * (W / 2 - (0.27 if ironr[s] else 0.2)), -(Ld - r.u(0.18, 0.24) if open_ else min(Ld - 0.16, r.u(0.3, 0.4))), ztop - 0.003
        a = r.u(-25, 25)
        ca, sa = cos(radians(a)), sin(radians(a))
        at = lambda dx, dy, dz: (lx + dx * ca - dy * sa, ly + dx * sa + dy * ca, lz + dz)
        w, hb, hg = 0.13, 0.02, 0.17
        irons.append(box("lan_base", (w + 0.022, w + 0.022, hb), loc=at(0, 0, hb / 2), rot=(0, 0, a), bevel=0.004, segments=1))
        for ex in (-1, 1):
            for ey in (-1, 1):
                irons.append(box(f"lan_post{ex}{ey}", (0.009, 0.009, hg), loc=at(ex * (w / 2 - 0.0045), ey * (w / 2 - 0.0045), hb + hg / 2), rot=(0, 0, a)))
        irons.append(box("lan_head", (w + 0.01, w + 0.01, 0.012), loc=at(0, 0, hb + hg + 0.006), rot=(0, 0, a), bevel=0.002, segments=1))
        R = (w / 2 + 0.016) * 1.4142
        zr0 = hb + hg + 0.012
        irons.append(lathe("lan_roof", [(R, 0.0), (R, 0.005), (0.034, 0.06), (0.026, 0.064), (0.026, 0.08), (0.036, 0.084), (0.03, 0.094), (0.0, 0.098)],
                           segments=4, loc=at(0, 0, zr0), rot=(0, 0, a + 45), smooth_angle=25))
        irons.append(torus("lan_ring", 0.032, 0.0035, loc=at(0, 0, zr0 + 0.118), rot=(90, 0, a + r.u(20, 70)), major_seg=14, minor_seg=5))
        # CLEARANCE RULE: the glowing glass has no ends (they lay 1 mm over the base and 1 mm under the head)
        glass_objs.append(lod_faces(box("lan_glass", (w - 0.012, w - 0.012, hg - 0.002), loc=at(0, 0, hb + hg / 2), rot=(0, 0, a)),
                                    drop=[(0, 0, 1), (0, 0, -1)], above=-1, cos=0.9))
        wax_objs.append(cylinder("lan_candle", 0.015, 0.075, loc=at(0, 0, hb + 0.0375), verts=10, bevel=0.003, segments=1))
        flame_objs.append(sphere("lan_flame", 0.01, loc=at(0, 0, hb + 0.075 + 0.022), segments=8, rings=6, scale=(1, 1, 2.0)))
        light("stoop_lantern", at(0, 0, hb + 0.16), color="#ffbf6a", intensity=p.lantern_cd, range=4.0, flicker=0.4, shadow_near=0.22)

    # ------------------------------------------------------------------ moss and silt: broken lines in the damp angles
    wt_walk = lambda x, y: (0.18 + 0.82 * float(_ss(0.3, 0.85, abs(x - cx0) / (W / 2)))) * (1.25 if x * damp > 0 else 0.8)
    wt_wall = lambda x, y: (0.2 + 0.8 * float(_ss(0.25, 0.8, abs(x) / (W / 2)))) * (1.25 if x * damp > 0 else 0.8)
    wt_foot = lambda x, y: (0.55 + 0.45 * float(_ss(0.3, 0.85, abs(x - cx0) / (W / 2)))) * (1.2 if x * damp > 0 else 0.8)
    wt_side = lambda x, y: (1.25 if x * damp > 0 else 0.75) * (0.65 + 0.35 * float(_ss(yf0, 0.0, y)))
    wt_dirt = lambda x, y: 0.4 + 0.6 * float(_ss(0.2, 0.7, abs(x - cx0) / (W / 2)))
    Z = (0, 0, 1)
    ms, ds = _Strips(rng.fork("moss")), _Strips(rng.fork("silt"))
    mc = 1.6 * p.moss
    mstep = 0.03 * clamp(N * W / 4.2, 1.0, 1.8)
    mw, mh = (0.03, 0.062), (0.009, 0.016)                  # cushion width / height
    dw, dh = (0.012, 0.024), (0.004, 0.008)
    for j in range(N):
        c = C[j]
        if j < N - 1:
            hn, yn = C[j + 1]["hw"], c["ynext"]
            a_, b_ = ((-hn if not cheek[-1] else -W / 2) + 0.012, yn, 0), ((hn if not cheek[1] else W / 2) - 0.012, yn, 0)
            ms.line(a_, b_, (0, -1, 0), Z, mc, mw, mh, zf=c["zf"], wt=wt_walk, step=mstep)
            ds.line(a_, b_, (0, -1, 0), Z, 0.6, dw, dh, zf=c["zf"], wt=wt_dirt, step=0.04)
            if splay:
                for s in (-1, 1):
                    ms.line((s * hn, yn, 0), (s * hn, -0.012, 0), (s, 0, 0), Z, mc, mw, mh, zf=c["zf"], wt=wt_side)
                    ds.line((s * hn, yn, 0), (s * hn, -0.012, 0), (s, 0, 0), Z, 0.5, dw, dh, zf=c["zf"], step=0.04)
                    ms.line((s * (hn + 0.01), 0, 0), (s * (c["hw"] - 0.01), 0, 0), (0, -1, 0), Z, mc, mw, mh, zf=c["zf"], wt=wt_side)
        else:
            a_, b_ = (-W / 2 + 0.012, 0, 0), (W / 2 - 0.012, 0, 0)
            ms.line(a_, b_, (0, -1, 0), Z, mc, mw, mh, zf=c["zf"], wt=wt_wall, step=mstep)
            ds.line(a_, b_, (0, -1, 0), Z, 0.45, dw, dh, zf=c["zf"], wt=wt_wall, step=0.04)
        for xj in c["jx"]:                                   # butt joints of the tread stones
            ya, yb_ = c["yf"] + 0.04, (c["ynext"] if j < N - 1 else 0.0) - 0.015
            ms.line((xj, ya, 0), (xj, yb_, 0), (1, 0, 0), Z, mc * 0.9, (0.007, 0.012), (0.005, 0.008), angle=False,
                    zf=lambda x, y, c=c: c["zf"](x, y) - 0.005, wt=lambda x, y, c=c: 0.3 + 0.7 * float(_ss(c["yf"] + 0.03, c["ynext"] - 0.03, y)))
    # paving line round the foot
    ms.line((-hw0, yf0, 0), (hw0, yf0, 0), (0, -1, 0), Z, mc, (0.045, 0.09), mh, wt=wt_foot, step=mstep)
    for s in (-1, 1):
        ys_ = (ypf + ps) if cheek[s] else yf0
        ms.line((s * xside[s], ys_, 0), (s * xside[s], -0.012, 0), (s, 0, 0), Z, mc, (0.04, 0.08), mh, wt=wt_side, step=mstep)
    for (P0, P1, across, nrm) in side_joints:               # open joints of the side blocks
        ms.line(P0, P1, across, nrm, mc * (0.75 if nrm[0] * damp > 0 else 0.4), (0.006, 0.011), (0.007, 0.012), angle=False, sink=0.004)
    if scr_box:                                             # mossy joint round the sunk scraper stone
        xa, ya, xb, yb_ = scr_box
        xo, e = (xb, 1) if xa > 0 else (xa, -1)
        ms.line((xa, ya, 0), (xb, ya, 0), (0, -1, 0), Z, mc * 1.2, mw, mh)
        ms.line((xa, yb_, 0), (xb, yb_, 0), (0, 1, 0), Z, mc * 1.2, mw, mh)
        ms.line((xo, ya, 0), (xo, yb_, 0), (e, 0, 0), Z, mc * 1.2, mw, mh)
    ms.build("moss_cushions", mat("moss"))
    ds.build("silt", dirt_m)

    # ------------------------------------------------------------------ rust washed out of the post sockets onto the stone
    if post_feet:
        r = rng.fork("rust")
        SV, SF = [], []
        for (px, py, pz) in post_feet:
            c = C[clamp(int(round(pz / H)) - 1, 0, N - 1)]
            ylim = c["yf"] - (ov - rk) + 0.008
            xlim = W / 2 + (ov - rk) - 0.008
            b0 = len(SV)
            cyb = py - 0.014
            SV.append((px, cyb, c["zf"](px, cyb) + 0.0018))
            n = 15
            for i in range(n):
                a = 2 * pi * i / n
                rad = (0.04 + 0.07 * max(0.0, -sin(a)) ** 6) * r.u(0.6, 1.2)         # drawn out down the fall of the tread
                x, y = clamp(px + rad * cos(a) * r.u(0.85, 1.15), -xlim, xlim), max(cyb + rad * sin(a), ylim)
                SV.append((x, y, c["zf"](x, y) + 0.0018))
                SF.append((b0, b0 + 1 + i, b0 + 1 + (i + 1) % n))
        so = mesh_object("rust_stains", SV, SF)
        assign(so, rust_m)
        smooth(so, 80)

    # ------------------------------------------------------------------ materials and the weathering story
    for o in stones:
        assign(o, st_m)
    for o in subs:
        assign(o, sub_m)
    for o in irons + scr_irons:
        assign(o, iron_m)
    for o in leads:
        assign(o, lead_m)
    for o in glass_objs:
        assign(o, mat("glass_lantern", glow=0.6))
    for o in wax_objs:
        assign(o, mat("wax", "#efe2c4"))
    for o in flame_objs:
        assign(o, mat("emissive", "#ffb45e", glow=5))

    n1 = noise_mask(0.09, seed=p.seed + 3, lo=0.3, hi=0.7)
    n2 = noise_mask(0.45, seed=p.seed + 5, lo=0.25, hi=0.75)
    n3 = noise_mask(0.05, seed=p.seed + 9, lo=0.35, hi=0.75)

    def paint_course(j, c):
        top, zb, ynext, last = c["top"], c["top"] - H, c["ynext"], c["last"]
        hn = C[j + 1]["hw"] if (splay and not last) else None
        jxs = np.array(c["jx"] or [1e6])

        def parts(co, n):
            x, y, z = co[:, 0], co[:, 1], co[:, 2]
            up, side = np.clip(n[:, 2], 0, 1), 1 - np.abs(n[:, 2])
            da = ynext - y                                   # distance out of the internal angle (riser above / wall)
            if hn is not None:
                da = np.minimum(np.maximum(da, np.abs(x) - hn), -y)
            ont = up * (z > top - 0.045)
            djx = np.min(np.abs(x[:, None] - jxs[None, :]), axis=1)
            return x, y, z, up, side, da, ont, djx

        def grime(co, n):
            x, y, z, up, side, da, ont, djx = parts(co, n)
            ang = ont * (1 - _ss(0.003, 0.024, da))          # only the last 15-25 mm in front of the riser
            foot = side * (1 - _ss(0.004, 0.034, z - zb))
            jt = ont * (1 - _ss(0.002, 0.012, djx))
            return np.clip(0.1 * (1 - _ss(0.0, 0.25, z)) + (0.2 * ang + 0.15 * foot) * (0.2 + 0.8 * n3(co, n))
                           + 0.1 * side * n2(co, n) + 0.22 * jt, 0, 1)

        def moss(co, n):
            x, y, z, up, side, da, ont, djx = parts(co, n)
            ang = ont * (1 - _ss(0.0, 0.05, da))
            foot = side * (1 - _ss(0.0, 0.05, z - zb)) * (1.0 if j == 0 else 0.5)
            jt = 0.7 * ont * (1 - _ss(0.003, 0.03, djx))
            offc = (0.2 + 0.8 * _ss(0.3, 0.85, np.abs(x - cx0) / (W / 2))) * np.where(x * damp > 0, 1.2, 0.7)
            a = n1(co, n)
            return np.clip(p.moss * 1.3 * (np.maximum(np.maximum(ang, foot), jt) * offc * (0.35 + 0.65 * a)
                                           + 0.12 * ont * a * n2(co, n) * offc), 0, 1)

        def wear(co, n):
            x, y = co[:, 0], co[:, 1]
            d = c["dish"](x, y) / depth                      # the polished hollow, brightest on the worn nose
            return np.clip((0.3 + 0.6 * p.wear) * d ** 0.7 * np.clip(n[:, 2] * 1.5, 0, 1) * (0.6 + 0.4 * n2(co, n)), 0, 1)

        paint(c["o"], grime=grime, wear=wear, special=moss)

    for j, c in enumerate(C):
        paint_course(j, c)

    def g_fn(co, n):
        z = co[:, 2]
        return np.clip(0.12 * (1 - _ss(0.0, 0.25, z)) + 0.12 * (1 - np.abs(n[:, 2])) * n2(co, n), 0, 1)

    def m_fn(co, n):
        x, z = co[:, 0], co[:, 2]
        side = 1 - np.abs(n[:, 2])
        low = side * (1 - _ss(0.0, 0.09, z))
        return np.clip(p.moss * (0.7 * low * (0.35 + 0.65 * n1(co, n)) + 0.3 * side * n1(co, n) * n2(co, n))
                       * np.where(x * damp > 0, 1.2, 0.7), 0, 1)

    paint(stones + subs, grime=g_fn, special=m_fn)
    for o in stones:
        if o.name.startswith(("coping", "pier_")):        # hands and sleeves polish the cheek copings
            paint(o, wear=mask_mul(facing((0, 0, 1), 2.0), n2, 0.6))

    surf = lambda y: H * np.clip(np.floor((y + Ld) / G + 1e-6) + N, 0, N)       # tread height under the rail at y

    def rust_fn(co, n):
        y, z = co[:, 1], co[:, 2]
        hz = z - surf(y)
        return np.clip((1 - _ss(0.0, 0.22, hz)) * (0.3 + 0.6 * n1(co, n)) + 0.25 * n1(co, n) * n2(co, n), 0, 1)

    paint(irons, special=rust_fn)
    for o in irons:
        if o.name.startswith(("handrail", "volute")):     # the top of the rail is kept bright by hands
            paint(o, wear=lambda co, n: np.clip(0.55 + 0.6 * n[:, 2], 0, 1) * (0.7 + 0.3 * n2(co, n)))
    if scr_irons:
        paint(scr_irons, special=lambda co, n: np.clip((1 - _ss(0.05, 0.3, co[:, 2])) * (0.4 + 0.6 * n1(co, n)), 0, 1))
        paint(blade, wear=lambda co, n: _ss(0.164, 0.175, co[:, 2]))     # boots rub the top edge bright
    paint(leads, grime=0.35)

    # ------------------------------------------------------------------ sockets
    socket("door", "attach", (0.0, 0.0, ztop), role="door")
    for s, nm_ in ((-1, "l"), (1, "r")):
        if s in pier_top:
            socket(f"pier_{nm_}", "attach", pier_top[s], role="prop_top")
        elif not ironr[s] or W > 1.3:
            socket(f"pot_{nm_}", "attach", (s * (W / 2 - 0.26), -0.2, ztop), role="prop_top")
