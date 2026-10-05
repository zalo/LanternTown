"""Post-and-chain barrier fencing a monument forecourt: a run of low posts linked by swags of heavy
forged chain. Posts are either monolithic dressed stone (square pier with plinth course, chamfered
arrises, necking and cornice cap; or a turned round post with a mushroom cap and an iron collar) or
cast iron (turned baluster or square-die / octagonal shaft) bolted through a flange plate onto a
stone setting pad. Every post carries a finial (ball, acorn, pyramid, spear). The chain is made of
real interlocked oval links laid along a catenary() between forged eye bolts; at the end posts (and
at an opened span) it ends in a forged hook dropped into a staple so it can be lifted off. An
opened span leaves its chain hanging from the post and coiled on the paving.
The run lies along X and spans exactly [-length/2, +length/2]; the end posts stand just inside both ends
(their pads / plinths finish flush with the ends, and they only ever lean inwards) so runs butt together.
The links are stamped one by one along catenary() rather than with tk.chain(): the link count per span
has to be odd and exactly pitched so the first and last link thread the eye bolts, every link gets its own
forged irregularity, and the link resolution is chosen to keep a 10 m run inside the triangle budget.
"""
import numpy as np
from math import ceil, copysign
from mathutils import Matrix
from tk import *

META = {"name": "Post and chain barrier", "class": "prop", "mount": "ground", "focus": [-0.16, 0.0, 0.62]}
PARAMS = {
    "length": {"default": 4.4, "min": 1.5, "max": 10.0, "desc": "overall run length along X (m); posts stand at both ends"},
    "posts": {"default": 0, "min": 0, "max": 6, "desc": "number of posts, 2-6 (0 = as many as length / spacing asks for)"},
    "spacing": {"default": 2.1, "min": 1.0, "max": 4.0, "desc": "target post spacing when posts = 0 (m)"},
    "post_material": {"default": "stone", "choices": ["stone", "iron"], "desc": "post material: dressed stone or painted cast iron"},
    "post_shape": {"default": "square", "choices": ["square", "round"],
                   "desc": "stone: square panelled pier / round turned post with iron collar; "
                           "iron: square die with octagonal shaft / round turned baluster"},
    "finial": {"default": "ball", "choices": ["ball", "acorn", "pyramid", "spear", "none"], "desc": "finial on every post"},
    "post_height": {"default": 0.95, "min": 0.55, "max": 1.25, "desc": "post height to the top of the finial (m)"},
    "girth": {"default": 1.0, "min": 0.8, "max": 1.3, "desc": "post thickness multiplier"},
    "sag": {"default": 0.12, "min": 0.02, "max": 0.32, "desc": "chain sag at mid-span as a fraction of the span"},
    "rows": {"default": 1, "min": 1, "max": 2, "desc": "number of chains, one above the other"},
    "emblem": {"default": "star", "choices": ["star", "roundel", "none"],
               "desc": "relief on the front and back of each post (stone piers: inside a raised panel; iron dies: on the panel)"},
    "link": {"default": 0.1, "min": 0.05, "max": 0.12, "desc": "chain link length (m)"},
    "chain_finish": {"default": "wrought", "choices": ["wrought", "rusty", "painted", "galvanised"], "desc": "chain finish"},
    "stone": {"default": "#b3a78f", "desc": "stone colour (stone posts, setting pads)"},
    "paint": {"default": "#1f2a22", "desc": "paint colour of cast-iron posts (and a painted chain)"},
    "trim": {"default": "gilt", "choices": ["gilt", "painted", "none"], "desc": "picked-out finials and beads on iron posts"},
    "trim_paint": {"default": "#d6ceb8", "desc": "trim colour when trim = painted"},
    "base": {"default": "pad", "choices": ["pad", "plate", "none"],
             "desc": "pad = stone setting pad (iron posts bolted to it through a flange plate); plate = flange plate / plinth "
                     "straight on the paving; none = set into the paving"},
    "end_hooks": {"default": True, "desc": "chain ends in forged hooks dropped into staples on the two end posts"},
    "open_span": {"default": 0, "min": 0, "max": 5, "desc": "span left open, its chain unhooked and coiled by its post (0 = none)"},
    "rust": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "rust / staining amount"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "moss and damp on stone"},
    "lean": {"default": 1.0, "min": 0.0, "max": 5.0, "desc": "largest post lean (deg); direction and amount per post from the seed"},
}
VARIANTS = {
    "iron_gilt_double": {"post_material": "iron", "post_shape": "round", "finial": "acorn", "rows": 2, "sag": 0.05,
                         "trim": "gilt", "length": 4.0, "posts": 3, "post_height": 1.0, "link": 0.1, "paint": "#294537", "seed": 5},
    "octagon_red_spear": {"post_material": "iron", "post_shape": "square", "finial": "spear", "paint": "#5a1414", "trim": "painted",
                          "chain_finish": "painted", "sag": 0.16, "length": 3.6, "posts": 3, "post_height": 1.1,
                          "base": "plate", "rust": 0.45, "seed": 11},
    "round_granite_low": {"post_material": "stone", "post_shape": "round", "finial": "none", "stone": "#7d7a76", "post_height": 0.66,
                          "girth": 1.2, "sag": 0.26, "link": 0.115, "chain_finish": "rusty", "emblem": "roundel", "length": 5.0,
                          "posts": 4, "moss": 0.8, "rust": 0.8, "lean": 2.5, "base": "none", "seed": 17},
    "open_gate": {"post_material": "stone", "post_shape": "square", "finial": "pyramid", "length": 5.2, "posts": 4, "open_span": 2,
                  "stone": "#c0b192", "sag": 0.13, "seed": 23},
    "short_blue_pair": {"post_material": "iron", "post_shape": "round", "finial": "ball", "paint": "#21406b", "trim": "painted",
                        "emblem": "none", "length": 1.8, "posts": 2, "sag": 0.22, "chain_finish": "galvanised", "end_hooks": False,
                        "post_height": 0.85, "seed": 31},
}

RE, RE_W = 0.019, 0.006            # eye ring: major radius, wire radius (rescaled to the chain bar in build())
RH, RH_W = 0.0125, 0.0045          # hook eye ring
BAR_W = 0.006                      # staple wire radius
BAR_OUT = 0.042                    # staple bar centre, out from the post face


def _arc(rc, zc, rad, a0, a1, n):
    return [(rc + rad * cos(radians(a0 + (a1 - a0) * i / n)), zc + rad * sin(radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def _polygonize(obj, n, segs, wfn, rot_deg=0.0, crn=0.3):
    """Turn a lathe (built with `segs` segments) into an n-gon with eased corners; profile radius =
    distance to the flats. wfn(z) -> 0 (round) .. 1 (polygon)."""
    m = segs // n
    da = 2 * pi / segs
    h = pi / n
    rot = radians(rot_deg)

    def f(co):
        r = hypot(co.x, co.y)
        if r < 1e-6:
            return co
        th = atan2(co.y, co.x)
        k = int(round(th / da)) % segs
        j = k % m
        if j == 0:
            off = 0.0
            rp = 0.5 * (1.0 / cos(h) + cos(da * crn) / cos(h - da * crn))
        else:
            off = da * crn if j == 1 else (m - crn) * da if j == m - 1 else j * da
            rp = 1.0 / cos(off - h)
        w = wfn(co.z)
        rr = r * (1.0 + (rp - 1.0) * w)
        ang = (k - j) * da + (j * da) * (1 - w) + off * w + rot
        return Vector((rr * cos(ang), rr * sin(ang), co.z))

    transform_verts(obj, f)


_TOL = [0.0]                        # profile simplification tolerance (m); > 0 on long runs of many posts


def _rdp(pts, tol):
    """Ramer-Douglas-Peucker on a (r, z) profile."""
    if len(pts) < 3:
        return list(pts)
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    ln = hypot(dx, dy) or 1e-9
    dm, im = 0.0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dx * (pts[i][1] - y0) - dy * (pts[i][0] - x0)) / ln
        if d > dm:
            dm, im = d, i
    if dm <= tol:
        return [pts[0], pts[-1]]
    return _rdp(pts[:im + 1], tol)[:-1] + _rdp(pts[im:], tol)


def _lathe(name, profile, **kw):
    """tk.lathe with the normals verified (a short open piece can come out inside-out)."""
    if _TOL[0] > 0:
        profile = _rdp(profile, _TOL[0])
    o = lathe(name, profile, **kw)
    me = o.data
    tot = sum((q.normal.x * q.center.x + q.normal.y * q.center.y) * q.area for q in me.polygons)
    if tot < 0:
        me.flip_normals()
    return o


def _tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def _finial(kind, rn, rb):
    """Profile [(r, dz)] of a finial standing on a neck of radius rn (dz = 0); rb = bulb radius."""
    if kind == "ball":
        zc = 0.36 * rb + sqrt(max(rb * rb - rn * rn, 1e-8))
        a0 = degrees(atan2(-sqrt(max(rb * rb - rn * rn, 1e-8)), rn))
        pr = [(rn, 0.0), (rn, 0.08 * rb), (rn * 1.14, 0.15 * rb), (rn * 1.14, 0.25 * rb), (rn, 0.33 * rb)]
        pr += _arc(0, zc, rb, a0, 76, 8)
        zt = zc + rb * sin(radians(76))
        pr += [(0.25 * rb, zt + 0.035 * rb), (0.24 * rb, zt + 0.11 * rb), (0.15 * rb, zt + 0.18 * rb), (0.0, zt + 0.2 * rb)]
    elif kind == "acorn":
        pr = [(rn, 0.0), (rn, 0.1 * rb), (rn * 1.15, 0.16 * rb), (0.8 * rb, 0.3 * rb), (0.98 * rb, 0.5 * rb), (1.04 * rb, 0.78 * rb),
              (1.0 * rb, 0.9 * rb), (0.9 * rb, 0.93 * rb), (0.9 * rb, 1.15 * rb), (0.84 * rb, 1.5 * rb), (0.66 * rb, 1.9 * rb),
              (0.4 * rb, 2.2 * rb), (0.17 * rb, 2.38 * rb), (0.1 * rb, 2.46 * rb), (0.09 * rb, 2.56 * rb), (0.0, 2.62 * rb)]
    elif kind == "spear":
        pr = [(rn, 0.0), (rn, 0.12 * rb), (0.95 * rb, 0.22 * rb), (1.0 * rb, 0.32 * rb), (0.95 * rb, 0.42 * rb), (0.5 * rb, 0.52 * rb),
              (0.42 * rb, 0.75 * rb), (0.62 * rb, 1.05 * rb), (0.95 * rb, 1.45 * rb), (1.05 * rb, 1.9 * rb), (0.92 * rb, 2.5 * rb),
              (0.62 * rb, 3.2 * rb), (0.3 * rb, 3.8 * rb), (0.08 * rb, 4.2 * rb), (0.0, 4.3 * rb)]
    elif kind == "pyramid":
        pr = [(rn, 0.0), (rn, 0.12 * rb), (1.2 * rb, 0.24 * rb), (1.25 * rb, 0.3 * rb), (1.25 * rb, 0.46 * rb), (1.15 * rb, 0.52 * rb),
              (0.6 * rb, 1.4 * rb), (0.1 * rb, 2.14 * rb), (0.0, 2.2 * rb)]
    else:
        pr = [(rn, 0.0), (rn * 1.05, 0.06 * rb), (rn * 0.8, 0.16 * rb), (0.0, 0.22 * rb)]
    return pr


def _link_base(name, ll, w, lw, nseg, sides):
    hl, hw = ll / 2 - w, lw / 2 - w
    pts = []
    for i in range(nseg):
        a = 2 * pi * i / nseg
        c, s = cos(a), sin(a)
        pts.append((hl * copysign(abs(c) ** 0.6, c), hw * copysign(abs(s) ** 0.6, s), 0.0))
    return sweep(name, pts, w, closed=True, sides=sides, smooth_path=False)


class _Path:
    """Polyline with arc-length lookup."""

    def __init__(self, pts):
        self.P = np.asarray(pts, dtype=np.float64)
        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.S = np.concatenate([[0.0], np.cumsum(d)])
        self.L = float(self.S[-1])

    def at(self, s):
        s = min(max(s, 0.0), self.L)
        i = int(np.searchsorted(self.S, s, side="right")) - 1
        i = min(max(i, 0), len(self.P) - 2)
        f = (s - self.S[i]) / max(self.S[i + 1] - self.S[i], 1e-9)
        return self.P[i] * (1 - f) + self.P[i + 1] * f

    def tan(self, s, h=0.012):
        d = self.at(s + h) - self.at(s - h)
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _link_matrix(pos, t, ang_deg):
    ref = np.array([0.0, 1.0, 0.0]) if abs(t[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b1 = ref - ref.dot(t) * t
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(t, b1)
    a = radians(ang_deg)
    y = cos(a) * b1 + sin(a) * b2
    z = np.cross(t, y)
    return Matrix(((t[0], y[0], z[0], pos[0]), (t[1], y[1], z[1], pos[1]), (t[2], y[2], z[2], pos[2]), (0, 0, 0, 1)))


def _hook(name, C, t, n, wh, forged):
    """Forged hook whose bend bears on a bar at C; t = unit vector from the bar towards the chain,
    n = unit vector across the hook (its plane is t-n). Returns the centre of its eye ring."""
    C, t, n = np.asarray(C, float), np.asarray(t, float), np.asarray(n, float)
    rh = BAR_W + wh + 0.0008
    ls = 0.05
    pts = [C + n * rh * 1.5 + t * 0.026, C + n * rh * 1.12 + t * 0.012]
    for i in range(9):
        ph = radians(180 - 180 * i / 8)
        pts.append(C + rh * (-n * cos(ph) - t * sin(ph)))
    pts += [C - n * rh + t * ls * 0.42, C - n * rh * 0.82 + t * ls * 0.6, C - n * rh * 0.3 + t * ls * 0.82, C + t * (ls + 0.002)]
    hk = sweep(name, [tuple(q) for q in pts], wh, sides=5, smooth_path=False,
               scale_fn=lambda u: 0.55 + 0.45 * smoothstep(0.0, 0.14, u), up=tuple(np.cross(t, n)))
    smooth(hk, 60)
    forged.append(hk)
    S = C + t * (ls + RH)
    ring = torus(name + "_eye", RH, RH_W, major_seg=8, minor_seg=4)
    b = np.cross(t, n)
    M = Matrix(((t[0], n[0], b[0], S[0]), (t[1], n[1], b[1], S[1]), (t[2], n[2], b[2], S[2]), (0, 0, 0, 1)))
    transform_verts(ring, lambda co: M @ co)
    forged.append(ring)
    return S


def build(p, rng):
    L = float(p.length)
    H = float(p.post_height)
    g = float(p.girth)
    R = float(p.rust)
    seed = int(p.seed)
    stone_post = p.post_material == "stone"
    style = {("stone", "square"): "stone_square", ("stone", "round"): "stone_round",
             ("iron", "round"): "iron_baluster", ("iron", "square"): "iron_octagonal"}[(p.post_material, p.post_shape)]
    v = clamp(H / 0.95, 0.7, 1.15)
    kind = p.finial

    # ---- how many posts, where
    hp = 0.05 if p.base == "pad" else 0.0                      # setting pad height
    if style == "stone_square":
        base_r = (0.105 + 0.028) * g
    elif style == "stone_round":
        base_r = (0.115 + 0.037) * g
    else:
        base_r = 0.12 * g if p.base != "none" else 0.112 * g
    pad_w = 2 * base_r + 0.11
    end_r = pad_w / 2 + 0.006 if hp > 0 else base_r + 0.004
    n = int(p.posts)
    if n <= 0:
        n = int(round((L - 2 * end_r) / float(p.spacing))) + 1
    n = max(n, int(ceil((L - 2 * end_r) / 3.6 - 1e-6)) + 1)            # a chain never spans more than 3.6 m
    n = int(clamp(n, 2, 6))
    while n > 2 and (L - 2 * end_r) / (n - 1) < 0.62:           # never closer than a post and a hand's width
        n -= 1
    xs = [-(L / 2 - end_r) + (L - 2 * end_r) * i / (n - 1) for i in range(n)]
    span = xs[1] - xs[0]
    open_span = int(p.open_span)
    if open_span > n - 1 or n < 3:
        open_span = 0 if n < 3 else min(open_span, n - 1)
    segs = 24 if n <= 3 else 20 if n == 4 else 16
    _TOL[0] = 0.0 if n <= 4 else 0.0016
    if not stone_post:
        segs = 20 if n <= 2 else 16 if n <= 4 else 12            # slender castings need fewer segments

    # ---- materials
    stone_m = mat("stone_smooth", p.stone, rough=1.1, special=0.6 + 0.9 * p.moss)
    pad_m = mat("stone_rough", p.stone, rough=1.25, special=0.6 + 0.9 * p.moss)
    body_m = mat("iron_painted", p.paint, special=1.0 + 0.6 * R, wear=0.8 + 0.8 * R)
    if p.trim == "gilt":
        trim_m = mat("brass", rough=0.72)
    elif p.trim == "painted":
        trim_m = mat("iron_painted", p.trim_paint, special=1.0 + 0.6 * R)
    else:
        trim_m = body_m
    forged_m = mat("iron_wrought")
    chain_m = {"wrought": forged_m, "rusty": mat("iron_rusty"), "galvanised": mat("zinc", rough=1.1),
               "painted": mat("iron_painted", p.paint, special=1.2 + 0.6 * R, wear=1.6)}[p.chain_finish]

    w = clamp(float(p.link) * 0.073, 0.0036, 0.0085)            # chain bar radius
    ll = float(p.link)
    lw = ll * 0.62
    wh = w * 1.05                                              # hook bar radius
    global RE, RE_W
    RE, RE_W = 0.012 + 1.1 * w, max(0.006, 0.94 * w)           # the eye is forged from the same bar as the links

    # ---- post profiles (built on the origin, then leaned and moved)
    zb = hp
    if style == "stone_square":
        a = 0.105 * g
        at = a * 0.93
        rn = 0.043 * g
        rb = 0.074 * g
        if kind in ("pyramid", "spear"):
            top = [(a + 0.024 * g, 0.105 * v), (a * 0.5, 0.105 * v + (0.5 * a) * (1.5 if kind == "pyramid" else 2.6)),
                   (0.0, 0.105 * v + a * (1.5 if kind == "pyramid" else 2.6))]
        elif kind == "none":
            top = [(a + 0.024 * g, 0.105 * v), (a * 0.72, 0.126 * v), (a * 0.3, 0.137 * v), (0.0, 0.14 * v)]
        else:
            top = [(a + 0.024 * g, 0.105 * v), (rn + 0.018 * g, 0.128 * v), (rn + 0.004 * g, 0.134 * v)]
            top += [(r, 0.136 * v + dz) for r, dz in _finial(kind, rn, rb)]
        cap = [(at, 0.0), (at + 0.009 * g, 0.006 * v), (at + 0.009 * g, 0.022 * v), (at, 0.028 * v), (at, 0.04 * v),
               (at + 0.007 * g, 0.052 * v), (a + 0.018 * g, 0.061 * v), (a + 0.03 * g, 0.066 * v), (a + 0.03 * g, 0.098 * v)] + top
        zs1 = H - cap[-1][1]
        zs0 = zb + 0.13 * v
        zc = zs1 - 0.07 * v
    elif style == "stone_round":
        a = 0.115 * g
        at = a * 0.9
        rn = 0.045 * g
        rb = 0.07 * g
        if kind == "none":
            top = [(a * 0.86, 0.1 * v), (a * 0.55, 0.126 * v), (a * 0.2, 0.139 * v), (0.0, 0.141 * v)]
        elif kind in ("pyramid", "spear"):
            top = [(a * 0.9, 0.09 * v), (a * 0.45, 0.09 * v + a * (0.7 if kind == "pyramid" else 1.3)),
                   (0.0, 0.09 * v + a * (1.4 if kind == "pyramid" else 2.5))]
        else:
            top = [(a * 0.82, 0.103 * v), (rn + 0.02 * g, 0.127 * v), (rn + 0.004 * g, 0.134 * v)]
            top += [(r, 0.136 * v + dz) for r, dz in _finial(kind, rn, rb)]
        cap = [(at, 0.0), (at - 0.009 * g, 0.007 * v), (at - 0.009 * g, 0.021 * v), (at + 0.004 * g, 0.03 * v),
               (a + 0.016 * g, 0.041 * v), (a + 0.025 * g, 0.054 * v), (a + 0.024 * g, 0.066 * v), (a + 0.012 * g, 0.082 * v)] + top
        zs1 = H - cap[-1][1]
        zs0 = zb + 0.115 * v
        zc = zs1 - 0.085 * v
    else:
        rn = 0.025 * g
        rb = 0.058 * g if kind != "spear" else 0.039 * g
        fin = _finial(kind, rn, rb)
        cap_h = 0.064 * v
        zs1 = H - cap_h - fin[-1][1]
        pt = 0.014 if p.base != "none" else 0.0                # flange plate thickness
        z0 = zb + pt
        zs0 = z0 + (0.31 if style == "iron_baluster" else 0.2) * v
        zc = zs1 - 0.062 * v
        a = (0.05 if style == "iron_baluster" else 0.06) * g
        at = (0.041 if style == "iron_baluster" else 0.048) * g
    Ls = max(zs1 - zs0, 0.05)

    def rad(z):
        t = clamp((z - zs0) / Ls)
        return a + (at - a) * t + (0.004 * g * sin(pi * t) if style in ("stone_round", "iron_baluster") else 0.0)

    rf = rad(zc)                                               # face (flat) radius where the chain fixes
    if style == "stone_round":
        rf += 0.006                                            # iron collar
    elif not stone_post:
        rf += 0.005 * g                                        # belt
    row_z = [zc]
    d2 = min(0.25 * v, zc - zs0 - 0.035)
    if int(p.rows) > 1 and d2 >= 0.13:
        row_z.append(zc - d2)
    xr = rf + 0.011 + RE                                       # eye ring centre, out from the axis

    def fitting(j, sx):
        if (j == 0 and sx < 0) or (j == n - 1 and sx > 0):
            return None
        if p.end_hooks and ((j == 0 and sx > 0) or (j == n - 1 and sx < 0)):
            return "staple"
        if open_span and j == open_span and sx < 0:
            return "staple"
        return "eye"

    nzs = max(2, int(round(Ls / 0.11)))
    shaft_z = [zs0 + Ls * i / nzs for i in range(nzs + 1)]

    def make_post(j, rs):
        grp = {"stone": [], "body": [], "trim": [], "forged": [], "pad": []}
        sides = [(sx, fitting(j, sx)) for sx in (-1, 1)]
        if hp > 0:
            st = stone_block("pad", (pad_w + rs.u(-0.012, 0.006), pad_w + rs.u(-0.012, 0.012), hp + rs.u(-0.004, 0.004)),
                             loc=(0, 0, hp / 2), rot=(0, 0, rs.u(-2.5, 2.5)), rng=rs, chamfer=0.008, rough=0.0025,
                             detail=0.12 if n <= 4 else 0.4)
            transform_verts(st, lambda co: Vector((co.x, co.y, max(co.z, -hp / 2))))
            grp["pad"].append(st)
        if style == "stone_square":
            prof = [(a + 0.028 * g, zb), (a + 0.028 * g, zb + 0.05 * v), (a + 0.028 * g, zb + 0.095 * v), (a + 0.021 * g, zb + 0.106 * v),
                    (a + 0.007 * g, zb + 0.119 * v)]
            prof += [(rad(z), z) for z in shaft_z] + [(r, zs1 + dz) for r, dz in cap[1:]]
            ztop = zs1 + 0.1 * v
            sq = kind in ("pyramid", "spear")
            post = _lathe("post", prof, segments=segs, smooth_angle=38)
            _polygonize(post, 4, segs, (lambda z: 1.0) if sq else (lambda z: 1.0 - smoothstep(ztop + 0.004 * v, ztop + 0.03 * v, z)), 45.0,
                        crn=0.34)
            displace_noise(post, 0.0016, 0.07, seed=seed * 7 + j, along_normal=True)
            uv_box(post, rot90=True)                            # stone laid on its natural bed: veining runs level
            grp["stone"].append(post)
        elif style == "stone_round":
            prof = [(a + 0.03 * g, zb), (a + 0.03 * g, zb + 0.058 * v)] + _arc(a + 0.02 * g, zb + 0.076 * v, 0.016 * g, -50, 75, 3)
            prof += [(a + 0.005 * g, zb + 0.104 * v)] + [(rad(z), z) for z in shaft_z] + [(r, zs1 + dz) for r, dz in cap[1:]]
            post = _lathe("post", prof, segments=segs, smooth_angle=48)
            displace_noise(post, 0.0022, 0.09, seed=seed * 7 + j, along_normal=True)
            uv_transform(post, rot_deg=90.0)                    # bedding level round the turned post
            grp["stone"].append(post)
            # forged collar clasped round the neck, bolted at the back
            rc = rf - 0.006
            grp["forged"].append(_lathe("collar", [(rc - 0.001, zc - 0.024), (rc + 0.006, zc - 0.021), (rc + 0.006, zc + 0.021),
                                                  (rc - 0.001, zc + 0.024)], segments=segs, smooth_angle=30))
            for sy in (-1, 1):
                grp["forged"].append(box("lug", (0.012, 0.03, 0.036), loc=(sy * 0.009, rc + 0.016, zc), bevel=0.002, segments=1))
            grp["forged"].append(cylinder("clasp_bolt", 0.0055, 0.046, loc=(0, rc + 0.02, zc), rot=(0, 90, 0), verts=6))
        else:
            if pt > 0:
                pw = 0.24 * g
                grp["body"].append(box("plate", (pw, pw, pt), loc=(0, 0, zb + pt / 2), bevel=0.003, segments=1))
                bb = cylinder("bolt", 0.0105, 0.008, loc=(0, 0, 0.004), verts=6, radius2=0.0092)
                ms = [trs((sx * (pw / 2 - 0.022), sy * (pw / 2 - 0.022), zb + pt), (0, 0, rs.u(0, 60))) for sx in (-1, 1) for sy in (-1, 1)]
                grp["forged"].append(instances_to_mesh("bolts", bb, ms, keep_base=False, uv_jitter=False))
            if style == "iron_baluster":
                bp = [(.082, 0), (.082, .020), (.090, .027), (.093, .037), (.090, .047), (.080, .053), (.066, .060), (.060, .070),
                      (.060, .078), (.066, .084), (.074, .100), (.079, .125), (.080, .150), (.076, .180), (.066, .210), (.054, .238),
                      (.046, .258), (.043, .270), (.050, .276), (.052, .284), (.050, .292), (.043, .298)]
                prof = [(r * 1.22 * g, z0 + z * v) for r, z in bp] + [(rad(z), z) for z in shaft_z]
                grp["body"].append(_lathe("post", prof, segments=segs, smooth_angle=42))
                for tt, hh in ((0.3, 0.0065), (0.36, 0.004)):
                    z = zs0 + Ls * tt
                    grp["trim"].append(_lathe("ring", [(rad(z) - 0.001 + hh * g * cos(radians(q)), z + hh * v * sin(radians(q)))
                                                       for q in (-90, -30, 30, 90)], segments=segs, smooth_angle=65))
                belt = _lathe("belt", [(rf - 0.007 * g, zc - 0.03 * v), (rf, zc - 0.023 * v), (rf, zc + 0.023 * v), (rf - 0.007 * g, zc + 0.03 * v)],
                              segments=segs, smooth_angle=35)
                grp["body"].append(belt)
                capp = [(at, 0.0), (at + 0.006 * g, 0.005 * v), (at + 0.007 * g, 0.012 * v), (at + 0.001 * g, 0.018 * v), (at - 0.004 * g, 0.026 * v),
                        (at + 0.004 * g, 0.036 * v), (at + 0.015 * g, 0.043 * v), (at + 0.017 * g, 0.05 * v), (at + 0.012 * g, 0.056 * v),
                        (rn + 0.004 * g, 0.062 * v), (rn, 0.064 * v)]
                grp["body"].append(_lathe("cap", [(r, zs1 + dz) for r, dz in capp], segments=segs, smooth_angle=45))
            else:
                dp = [(.080, 0), (.080, .028), (.072, .036), (.066, .040), (.066, .135), (.072, .142), (.075, .152), (.071, .161), (.050, .19)]
                die = _lathe("die", [(r * 1.2 * g, z0 + z * v) for r, z in dp], segments=segs, smooth_angle=30)
                _polygonize(die, 4, segs, lambda z: 1.0, 45.0)
                grp["body"].append(die)
                # raised panel on each die face
                pn = box("panel", (0.004, 0.09 * g, 0.064 * v), bevel=0.0015, segments=1)
                rd = 0.066 * 1.2 * g
                ms = [trs((rd * cos(radians(q)), rd * sin(radians(q)), z0 + 0.088 * v), (0, 0, q)) for q in (0, 90, 180, 270)]
                grp["body"].append(instances_to_mesh("panels", pn, ms, keep_base=False, uv_jitter=False))
                sh = _lathe("shaft", [(rad(z), z) for z in [z0 + 0.18 * v] + shaft_z], segments=24, smooth_angle=30)
                _polygonize(sh, 8, 24, lambda z: 1.0, 22.5)
                grp["body"].append(sh)
                belt = _lathe("belt", [(rf - 0.007 * g, zc - 0.03 * v), (rf, zc - 0.023 * v), (rf, zc + 0.023 * v), (rf - 0.007 * g, zc + 0.03 * v)],
                              segments=24, smooth_angle=30)
                _polygonize(belt, 8, 24, lambda z: 1.0, 22.5)
                grp["body"].append(belt)
                capp = [(at, 0.0), (at + 0.006 * g, 0.005 * v), (at + 0.007 * g, 0.012 * v), (at + 0.001 * g, 0.018 * v), (at - 0.002 * g, 0.026 * v),
                        (at + 0.006 * g, 0.036 * v), (at + 0.016 * g, 0.043 * v), (at + 0.018 * g, 0.05 * v), (at + 0.012 * g, 0.056 * v),
                        (rn + 0.004 * g, 0.062 * v), (rn, 0.064 * v)]
                cp = _lathe("cap", [(r, zs1 + dz) for r, dz in capp], segments=24, smooth_angle=30)
                _polygonize(cp, 8, 24, lambda z: 1.0 - smoothstep(zs1 + 0.056 * v, zs1 + 0.064 * v, z), 22.5)
                grp["body"].append(cp)
            fz = zs1 + cap_h
            fo = _lathe("finial", [(r, fz + dz) for r, dz in fin], segments=segs if kind != "pyramid" else 24, smooth_angle=50)
            if kind == "spear":
                zsp = fz + 0.6 * rb
                transform_verts(fo, lambda co: Vector((co.x, co.y * (1.0 - 0.58 * smoothstep(zsp, zsp + 0.5 * rb, co.z)), co.z)))
            elif kind == "pyramid":
                _polygonize(fo, 4 if style == "iron_octagonal" else 8, 24, lambda z: smoothstep(fz + 0.1 * rb, fz + 0.24 * rb, z),
                            45.0 if style == "iron_octagonal" else 22.5)
            grp["trim"].append(fo)
            if p.base == "none":                               # run-lead joint where the casting enters the paving
                ld = _lathe("lead", [(base_r + 0.016, -0.0005), (base_r + 0.013, 0.005), (base_r - 0.004, 0.008)], segments=segs, smooth_angle=40)
                displace_noise(ld, 0.002, 0.03, seed=seed + j, along_normal=False, axis_mask=(1, 1, 0.3))
                transform_verts(ld, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
                grp["pad"].append(ld)

        # relief emblem (and, on the stone pier, a raised panel frame) on the front and back faces
        if p.emblem != "none" and style != "iron_baluster":
            em = []
            if style == "stone_square":
                fw, fz0, fz1, ft, d = at * 1.5, zs0 + 0.035 * v, zs1 - 0.03 * v, 0.014 * g, 0.008
                for sx in (-1, 1):
                    em.append(box("frame", (ft, d, fz1 - fz0), loc=(sx * (fw - ft) / 2, -d / 2, (fz0 + fz1) / 2)))
                for zz in (fz0 + ft / 2, fz1 - ft / 2):
                    em.append(box("frame", (fw - 2 * ft, d, ft), loc=(0, -d / 2, zz)))
                ez, er = fz0 + (fz1 - fz0) * 0.66, min(0.05 * g, (fz1 - fz0) * 0.3)
                rface = None
            elif style == "stone_round":
                ez, er, d = zs0 + Ls * 0.5, min(0.05 * g, Ls * 0.3), 0.006
                rface = "round"
            else:
                ez, er, d = z0 + 0.088 * v, 0.024 * g, 0.0075
                rface = 0.08 * g + 0.002
            if p.emblem == "star":
                sp = [(er * (1.0 if q % 2 == 0 else 0.42) * sin(radians(q * 36)), er * (1.0 if q % 2 == 0 else 0.42) * cos(radians(q * 36)))
                      for q in range(10)]
                # faceted relief star: every ray ridged up to a raised centre, on a 3 mm upstand
                vs = [(x, 0.002, ez + y) for x, y in sp] + [(x * 0.97, -(d * 0.45), ez + y * 0.97) for x, y in sp] + [(0.0, -(d + 0.006), ez)]
                fs = [(q, (q + 1) % 10, 10 + (q + 1) % 10, 10 + q) for q in range(10)] + [(10 + q, 10 + (q + 1) % 10, 20) for q in range(10)]
                star = mesh_object("star", vs, fs)
                recalc_normals(star)
                flat(star)
                em.append(star)
            else:
                em.append(cylinder("roundel", er, d + 0.002, loc=(0, -(d + 0.002) / 2, ez), rot=(90, 0, 0), verts=16, radius2=er * 0.86))
                em.append(cylinder("roundel_boss", er * 0.34, 0.006, loc=(0, -(d + 0.004), ez), rot=(90, 0, 0), verts=10, radius2=er * 0.2))
            ej = join(em, "emblem")
            eb = duplicate(ej, "emblem_b")

            def seat(co, sgn=1.0):
                if rface == "round":
                    rr = rad(co.z) - 0.0015 - co.y
                    th = co.x / max(rad(co.z), 1e-4)
                    return Vector((sgn * rr * sin(th), -sgn * rr * cos(th), co.z))
                r0_ = (rad(co.z) if rface is None else rface) - 0.0015
                return Vector((sgn * co.x, sgn * (co.y - r0_), co.z))

            transform_verts(ej, seat)
            transform_verts(eb, lambda co: seat(co, -1.0))
            grp["stone" if stone_post else "trim"] += [ej, eb]

        # chain fixings
        for sx, fk in sides:
            if not fk:
                continue
            for z in row_z:
                rz = rad(z) + (rf - rad(zc) if z == zc else 0.0)
                if stone_post and not (style == "stone_round" and z == zc):
                    bw = 0.03 + 2.2 * RE
                    grp["forged"].append(box("backplate", (0.007, bw, bw), loc=(sx * (rz + 0.0025), 0, z), bevel=0.0015, segments=1))
                    xs_ = rz + 0.006
                    for qy in (-1, 1):                          # four forged studs leaded into the stone
                        for qz in (-1, 1):
                            cy, cz, e = qy * (bw / 2 - 0.009), z + qz * (bw / 2 - 0.009), 0.0058
                            grp["forged"].append(mesh_object("stud", [(sx * xs_, cy - e, cz - e), (sx * xs_, cy + e, cz - e), (sx * xs_, cy + e, cz + e),
                                                                      (sx * xs_, cy - e, cz + e), (sx * (xs_ + 0.0045), cy, cz)],
                                                             [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)] if sx > 0 else [(1, 0, 4), (2, 1, 4), (3, 2, 4), (0, 3, 4)]))
                elif not stone_post:
                    grp["body"].append(_lathe("boss", [(0.021, -0.006), (0.021, 0.003), (0.015, 0.008), (0.009, 0.014)], segments=10 if n <= 4 else 8,
                                              loc=(sx * (rz - 0.001), 0, z), rot=(0, sx * 90, 0), smooth_angle=50))
                if fk == "eye":
                    x1 = xr - RE + 0.002
                    grp["forged"].append(cylinder("shank", 0.0068, x1 - rz + 0.006, loc=(sx * (rz + x1 - 0.006) / 2, 0, z), rot=(0, 90, 0),
                                                  verts=6, cap=False))
                    grp["forged"].append(torus("eye", RE, RE_W, loc=(sx * xr, 0, z), rot=(90, 0, 0), major_seg=12 if n <= 4 else 8, minor_seg=5 if n <= 4 else 4))
                else:
                    xo = rf + BAR_OUT if z == zc else rz + BAR_OUT
                    pts = [(rz - 0.006, -0.021, z), (xo - 0.02, -0.021, z), (xo - 0.006, -0.015, z), (xo, 0.0, z), (xo - 0.006, 0.015, z),
                           (xo - 0.02, 0.021, z), (rz - 0.006, 0.021, z)]
                    grp["forged"].append(sweep("staple", [(sx * q[0], q[1], q[2]) for q in pts], BAR_W, sides=5, cap=False, smooth_path=False))
        return grp

    # ---- build, weather and place the posts
    lean_r = rng.fork("lean")
    posts = []
    out = {"stone": [], "body": [], "trim": [], "forged": [], "pad": []}
    for j in range(n):
        rs = rng.fork("post%d" % j)
        grp = make_post(j, rs)
        la = radians(lean_r.u(0, 360))
        tl = tan(radians(p.lean * lean_r.u(0.15, 1.0)))
        kx, ky = tl * cos(la), tl * sin(la) * 0.6
        if j == 0 or j == n - 1:                                # end posts only ever lean inwards (pulled by the chain): clean ends
            kx = abs(kx) if j == 0 else -abs(kx)
        x0 = xs[j]
        posts.append((x0, kx, ky))
        ps = seed * 13 + j * 5
        n_fine = noise_mask(0.045, seed=ps + 2, lo=0.6 - 0.3 * R, hi=0.9 - 0.3 * R)
        n_big = noise_mask(0.16, seed=ps + 3, lo=0.3, hi=0.75)
        n_chip = noise_mask(0.03, seed=ps + 4, lo=0.55, hi=0.8)
        n_moss = noise_mask(0.07, seed=ps + 5, lo=0.42, hi=0.7)
        fit_sides = [sx for sx in (-1, 1) if fitting(j, sx)]
        ka = radians(rs.u(0, 360))

        def stone_grime(co, nn):
            z = co[:, 2]
            gr = np.clip(1.0 - (z - zb) / 0.22, 0, 1) * 0.55
            gr += np.clip(-nn[:, 2], 0, 1) * 0.5                               # under the cap mouldings
            for sx in fit_sides:                                              # rust weeping from the iron fixings
                for zz in row_z:
                    gr += (np.exp(-(co[:, 1] / 0.024) ** 2) * (sx * co[:, 0] > 0.02) * np.clip(1.0 - (zz - z) / (0.2 + 0.25 * R), 0, 1)
                           * (z < zz + 0.02) * (0.35 + 0.65 * R))
            return np.clip(gr, 0, 1)

        def stone_moss(co, nn):
            z = co[:, 2]
            up = np.clip(nn[:, 2], 0, 1) ** 1.5 * (0.35 + 0.65 * n_moss(co, nn))
            north = np.clip(nn[:, 1], 0, 1) * np.clip(1.0 - (z - zb) / (0.25 + 0.5 * p.moss), 0, 1) * n_moss(co, nn)
            return np.clip((up + north) * (0.25 + 0.9 * p.moss), 0, 1)

        def stone_wear(co, nn):
            z = co[:, 2]
            return np.clip((z - (H - 0.09)) / 0.07, 0, 1) * 0.5 + n_chip(co, nn) * 0.25 * np.exp(-((z - 0.3) / 0.12) ** 2)

        def iron_rust(co, nn):
            z = co[:, 2]
            rise = np.clip(1.0 - (z - zb) / (0.14 + 0.2 * R), 0, 1) ** 1.3
            ledge = np.clip(nn[:, 2], 0, 1) ** 2 * 0.7
            drip = np.exp(-((z - (zc - 0.05)) / 0.04) ** 2) * 0.6 + np.exp(-((z - (zs0 + 0.01)) / 0.03) ** 2) * 0.5
            allover = R * R * 0.6 * n_big(co, nn)
            return np.clip((rise + ledge + drip) * n_fine(co, nn) * (0.55 + 0.6 * R) + allover * n_fine(co, nn) ** 0.5, 0, 1)

        def iron_wear(co, nn):
            z = co[:, 2]
            side = np.clip(nn[:, 0] * cos(ka) + nn[:, 1] * sin(ka), 0, 1)
            knock = np.exp(-((z - 0.3) / 0.1) ** 2) * side * n_chip(co, nn) * (0.5 + 0.5 * R)
            hands = np.clip((z - (H - 0.07 * g)) / 0.05, 0, 1) * 0.5
            return np.clip(knock + hands, 0, 1)

        def iron_grime(co, nn):
            z = co[:, 2]
            return np.clip(np.clip(1.0 - (z - zb) / 0.3, 0, 1) * 0.55 + np.clip(-nn[:, 2], 0, 1) * 0.3 * n_big(co, nn), 0, 1)

        def lean(co, kx=kx, ky=ky, x0=x0):
            d = max(co.z - zb, 0.0)
            return Vector((co.x + kx * d + x0, co.y + ky * d, co.z))

        for key, objs in grp.items():
            if not objs:
                continue
            o = join(objs, "%s%d" % (key, j))
            if key == "stone":
                assign(o, stone_m)
                paint(o, grime=stone_grime, special=stone_moss, wear=stone_wear)
            elif key == "pad":
                if stone_post or p.base == "pad":
                    assign(o, pad_m)
                    rr_ = (rf if stone_post else 0.12 * g) * 1.6
                    paint(o, grime=lambda co, nn: np.clip(1.3 - np.hypot(co[:, 0], co[:, 1]) / rr_, 0, 1) * (0.5 + 0.4 * R),
                          special=mask_mul(facing((0, 0, 1)), n_moss, lambda co, nn: np.clip(0.4 + co[:, 1] * 3.0, 0, 1), 0.3 + 0.7 * p.moss))
                else:
                    assign(o, mat("lead"))
                    paint(o, grime=0.35)
            elif key == "body":
                assign(o, body_m)
                paint(o, grime=iron_grime, wear=iron_wear, special=iron_rust)
            elif key == "trim":
                assign(o, trim_m)
                paint(o, grime=iron_grime, special=lambda co, nn: iron_rust(co, nn) * 0.7, wear=lambda co, nn: 0.25 * n_chip(co, nn))
            else:
                assign(o, forged_m)
                paint(o, wear=0.3, special=lambda co, nn: np.clip(0.2 + R * 0.7, 0, 1) * noise_mask(0.02, seed=ps + 6, lo=0.3, hi=0.7)(co, nn))
            transform_verts(o, lean)
            out[key].append(o)
    post_tris = sum(_tris(o) for objs in out.values() for o in objs)

    def anchor(j, sx, z):
        """World position of the fixing on post j, side sx, at height z (eye centre / staple bar centre)."""
        x0, kx, ky = posts[j]
        rz = rad(z) + (rf - rad(zc) if z == zc else 0.0)
        fk = fitting(j, sx)
        off = xr if fk == "eye" else ((rf if z == zc else rz) + BAR_OUT)
        d = z - zb
        return np.array([x0 + sx * off + kx * d, ky * d, z]), fk

    # ---- chains
    hl, hw = ll / 2 - w, lw / 2 - w
    pitch = ll - 4 * w + 0.0006
    crng = rng.fork("chain")
    forged = []
    runs = []                                                  # (path, n_links, lying_from_s, end_angle)
    floor_boxes = [(x0, pad_w / 2 + 0.01) for x0, _, _ in posts] if hp > 0 else []

    def floor(x, y):
        for x0, hwp in floor_boxes:
            if abs(x - x0) < hwp and abs(y) < hwp:
                return hp
        return 0.0

    for i in range(n - 1):
        for z in row_z:
            (A, fa), (B, fb) = anchor(i, 1, z), anchor(i + 1, -1, z)
            sp = B[0] - A[0]
            sag_abs = min(float(p.sag) * sp * crng.u(0.88, 1.12), max(min(A[2], B[2]) - 0.08, 0.02))
            if open_span == i + 1:
                # unhooked: hangs from the left post's eye, the rest coiled on the paving beside it
                Lc = sp * (1.0 + 2.667 * (sag_abs / sp) ** 2) - 0.1
                x_e, y_e = A[0], A[1]
                top = A + np.array([0, 0, -(RE - RE_W - w)])
                pts = [top]
                # the hanging length rests against the plinth / base mouldings on its way down
                clear = max(0.0, base_r + lw / 2 + 0.008 - (x_e - posts[i][0]))
                x_f = x_e + clear
                zf = floor(x_f, y_e) + w
                drop = top[2] - zf
                nv = max(4, int(drop / 0.035))
                for k in range(1, nv):
                    zz = top[2] - (drop - 0.035) * k / (nv - 1)
                    pts.append(np.array([x_e + clear * smoothstep(zs0 + 0.14, zs0 + 0.01, zz), y_e, zz]))
                x_e = x_f
                cs = np.array([x_e + 0.19 + crng.u(0, 0.03), y_e - 0.035 - crng.u(0, 0.03)])
                r0 = hypot(x_e - cs[0], y_e - cs[1])
                th = atan2(y_e - cs[1], x_e - cs[0])
                dirn = -1.0
                rem = Lc - drop
                r = r0
                dth = 0.09
                first = True
                while rem > 0 and r > 0.055:
                    x, y = cs[0] + r * cos(th), cs[1] + r * sin(th)
                    if not first:
                        pts.append(np.array([x, y, floor(x, y) + w]))
                    first = False
                    th += dirn * dth
                    rem -= r * dth
                    r -= 0.062 * dth / (2 * pi) * (1.0 + 0.3 * sin(th * 2.3))
                path = _Path(pts)
                nl = int((path.L - hl - 0.02) / pitch)
                nl -= nl % 2                                   # even: the last link stands upright through the hook's flat eye
                runs.append((path, nl, drop + 0.03, pitch, None))
                s_end = hl + (nl - 1) * pitch + hl - w + RH - RH_W
                pe, te = path.at(s_end), path.tan(s_end)
                te = np.array([te[0], te[1], 0.0])
                te /= np.linalg.norm(te)
                Cb = pe + te * (0.05 + RH)
                Cb[2] = floor(pe[0], pe[1]) + wh
                _hook("hook_open", Cb, -te, np.array([te[1], -te[0], 0.0]), wh, forged)
                continue
            th_ = atan2(4 * sag_abs, sp)
            ends = []
            for (C, fk, sx) in ((A, fa, 1), (B, fb, -1)):
                t = np.array([sx * cos(th_), 0.0, -sin(th_)])
                if fk == "staple":
                    nrm = np.array([sx * sin(th_), 0.0, cos(th_)])
                    S = _hook("hook", C, t, nrm, wh, forged)
                    ends.append(S + t * (RH - RH_W - w))
                else:
                    ends.append(C + t * (RE - RE_W - w))
            path = _Path(catenary(tuple(ends[0]), tuple(ends[1]), sag=sag_abs * 0.94, n=40))
            nl = int(round((path.L - 2 * hl) / pitch)) + 1
            if nl % 2 == 0:
                nl += 1 if (path.L - 2 * hl) / pitch + 1 > nl else -1
            nl = max(nl, 3)
            runs.append((path, nl, None, (path.L - 2 * hl) / (nl - 1), 0.0))

    total_links = sum(r[1] for r in runs)
    budget = 15700 - post_tris - sum(_tris(o) for o in forged)
    nseg, sides = 6, 3
    for ns_, sd_ in ((10, 6), (8, 6), (8, 5), (8, 4), (6, 4), (6, 3)):
        if total_links * ns_ * sd_ * 2 <= budget:
            nseg, sides = ns_, sd_
            break
    base = _link_base("link", ll, w, lw, nseg, sides)
    mats = []
    for path, nl, lying_s, pt_, _ in runs:
        for k in range(nl):
            s = hl + k * pt_
            pos = path.at(s).copy()
            t = path.tan(s)
            end = k == 0 or k == nl - 1
            ang = (k % 2) * 90.0 + crng.u(-1, 1) * (3.0 if end else 11.0)
            if lying_s is not None and s > lying_s:
                ang = crng.u(-8, 8) if k % 2 == 0 else crng.pick((-1, 1)) * crng.u(38, 62)
                if k == nl - 1:
                    ang = 90.0
                pos[2] = floor(pos[0], pos[1]) + w + abs(sin(radians(ang))) * hw
                t = np.array([t[0], t[1], 0.0])
                t /= max(np.linalg.norm(t), 1e-9)
            elif lying_s is not None:                          # hanging links never dip through the paving
                pos[2] = max(pos[2], floor(pos[0], pos[1]) + w + abs(t[2]) * hl)
            mats.append(_link_matrix(pos, t, ang))
    ch = instances_to_mesh("chain", base, mats, keep_base=False, uv_jitter=True, seed=seed)
    smooth(ch, 180)
    assign(ch, chain_m)
    n_ch = noise_mask(0.05, seed=seed + 40, lo=0.3, hi=0.75)
    paint(ch, wear=lambda co, nn: np.clip(0.42 + 0.3 * n_ch(co, nn) + 0.35 * np.clip(nn[:, 2], 0, 1), 0, 1),
          special=lambda co, nn: np.clip((0.15 + 0.75 * R) * (0.4 + 0.6 * noise_mask(0.03, seed=seed + 41, lo=0.3, hi=0.7)(co, nn)), 0, 1),
          grime=lambda co, nn: np.clip(1.0 - co[:, 2] / 0.12, 0, 1) * 0.6)
    if forged:
        fj = join(forged, "hooks")
        assign(fj, forged_m)
        paint(fj, wear=0.4, special=lambda co, nn: np.clip(0.2 + 0.6 * R, 0, 1) * n_ch(co, nn))

    socket("end_l", "attach", (-L / 2, 0, zc), rot=(0, 0, -90), role="chain")
    socket("end_r", "attach", (L / 2, 0, zc), rot=(0, 0, 90), role="chain")
    print("DBG posts=%d span=%.2f links=%d (%dx%d) post_tris=%d chain_tris=%d open=%d rows=%d"
          % (n, span, total_links, nseg, sides, post_tris, _tris(ch), open_span, len(row_z)))
