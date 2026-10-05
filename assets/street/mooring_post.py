"""Canal-side mooring furniture in three kinds. 'pile': a stout tarred timber pile, hewn with the adze into
ten-odd uneven flats, its head dressed round for a forged iron band (lapped, riveted, clout-nailed, sunk
nearly flush), chamfered and mushroomed by the maul, split by drying checks that run into the end grain;
an iron cross pin, a hemp line hitched round it in a groove it has worn, falling down the timber over the
collar of granite setts to a loosely flaked two-layer coil on the quay; runs of tar and rust below the band.
'bollard': a cast-iron mushroom bollard with bolted, gusseted base flange, turned beads and a cast date,
on a granite pad. 'ring': a forged ring on a staple leaded into a stone - a wall stone (mount = wall) or
a free-standing granite mooring stone (mount = ground) - with a hank of line hung through the ring.
"""
from math import sin, cos, pi, radians, degrees, asin, acos, atan2, ceil, hypot, sqrt, exp

import bpy
import numpy as np
from mathutils import Vector, Euler

from tk import *

META = {"name": "Mooring post and ring", "class": "prop_small", "mount": "ground", "focus": 0.72,
        "mount_height": 1.0, "tri_budget": 8000}   # budget: the rope coils are the bulk of the triangles
PARAMS = {
    "kind": {"default": "pile", "choices": ["pile", "bollard", "ring"], "desc": "timber pile / cast-iron mushroom bollard / iron ring on a staple"},
    "height": {"default": 1.0, "min": 0.5, "max": 1.6, "desc": "overall height (m); bollards stop at 1.1, the ring's mooring stone is 0.7 x this"},
    "girth": {"default": 0.27, "min": 0.2, "max": 0.4, "desc": "diameter of the pile / bollard column (m)"},
    "rope": {"default": True, "desc": "hemp line: hitched round the post with a coil at its foot, or a hank hung in the ring (a pile without one shows the grooves old lines have cut)"},
    "rope_turns": {"default": 5, "min": 2, "max": 8, "desc": "turns of line round the post (ring: loops in the hank)"},
    "rope_dia": {"default": 0.024, "min": 0.018, "max": 0.04, "desc": "rope diameter (m)"},
    "rope_tint": {"default": "#9c8d72", "desc": "rope tint (#9c8d72 weathered hemp, #d9c7a2 new, #6a5c4a tarred)"},
    "head": {"default": "band", "choices": ["band", "painted", "capped"], "desc": "pile head: iron band on bare tar / painted head / lead cap"},
    "head_paint": {"default": "#b9b3a2", "desc": "colour of a painted pile head"},
    "tar": {"default": "#3e352e", "desc": "tint of the tarred timber (multiplies silver-grey weathered wood)"},
    "paint": {"default": "#4d665b", "desc": "paint colour of the cast-iron bollard"},
    "mark": {"default": "1887", "desc": "cast date / foundry mark on the bollard ('' = none)"},
    "ring_size": {"default": 0.2, "min": 0.12, "max": 0.3, "desc": "outer diameter of the mooring ring (m)"},
    "lean": {"default": 2.0, "min": 0.0, "max": 6.0, "desc": "how far the timber pile has been pulled out of plumb (deg)"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: checks in the timber, rust, moss, grime"},
    "mount": {"default": "ground", "choices": ["ground", "wall"], "desc": "wall = ring on a wall stone (kind 'ring' only)"},
}
VARIANTS = {
    "bollard": {"kind": "bollard", "height": 0.64, "girth": 0.3, "rope_turns": 3, "rope_dia": 0.028, "seed": 4},
    "wall_ring": {"kind": "ring", "mount": "wall", "rope_turns": 3, "seed": 8},
    "tall_capped": {"kind": "pile", "height": 1.45, "girth": 0.33, "head": "capped", "rope": False, "lean": 4.5,
                    "age": 0.95, "tar": "#3a322b", "seed": 13},
    "painted_tarred_line": {"kind": "pile", "height": 0.8, "girth": 0.3, "head": "painted", "head_paint": "#6b3d36",
                            "rope_turns": 7, "rope_dia": 0.021, "rope_tint": "#75664f", "lean": 1.0, "seed": 21},
    "mooring_stone": {"kind": "ring", "mount": "ground", "height": 1.1, "ring_size": 0.24, "rope_turns": 2, "rope_tint": "#7d6f59",
                      "age": 0.85, "seed": 30},
}

TAU = 2 * pi
WHIP = "#3d3024"


def _lin(e0, e1, x):
    return clamp((x - e0) / (e1 - e0))


def _zband(z0, half):
    """Mask: 1 at height z0, fading to 0 `half` metres above / below."""
    return lambda co, n: np.clip(1.0 - np.abs(co[:, 2] - z0) / half, 0.0, 1.0)


def _line(name, pts, rr, m, whip=None, closed=False, sides=7, step=None, max_rings=260):
    """Evenly sampled rope (the light hank lines)."""
    step = max(step or max(0.03, rr * 2.2), path_length(pts, closed=closed) / max_rings)   # bounded triangle cost
    path = resample_path(pts, step=step, closed=closed)
    o = sweep(name, path, rr, closed=closed, sides=sides, smooth_path=False)
    assign(o, m)
    out = [o]
    if whip is not None and not closed:
        for k, seg in enumerate((path[1:4], path[-4:-1])):
            w = sweep(f"{name}_whip{k}", [tuple(q) for q in seg], rr * 1.13, sides=sides, smooth_path=False)
            assign(w, whip)
            out.append(w)
    return out


def _rope(name, pts, rr, m, whip, sides=7, ang=0.3, smax=0.08, max_rings=262):
    """Heavy line swept along `pts`, sampled by curvature: rings where it bends, long segments where it hangs
    straight. Both ends are whipped. Returns (objects, dense path)."""
    dense = np.asarray(resample_path(pts, step=0.005), dtype=np.float64)
    d = np.diff(dense, axis=0)
    ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    t = d / ln[:, None]
    turn = np.arccos(np.clip((t[1:] * t[:-1]).sum(1), -1.0, 1.0))
    smin = rr * 0.9
    while True:
        keep, aa, al = [0], 0.0, 0.0
        for i in range(1, len(dense) - 1):
            al += ln[i - 1]
            aa += turn[i - 1]
            if (aa >= ang and al >= smin) or al >= smax:
                keep.append(i)
                aa = al = 0.0
        if len(dense) - 1 - keep[-1] < 2 and len(keep) > 1:
            keep.pop()
        keep.append(len(dense) - 1)
        if len(keep) <= max_rings:
            break
        ang *= 1.1
        smax *= 1.06
    o = sweep(name, [tuple(q) for q in dense[keep]], rr, sides=sides, smooth_path=False)
    assign(o, m)
    out = [o]
    for k, seg in enumerate((dense[3:11:2], dense[-11:-3:2])):          # whippings 15 - 50 mm from the ends
        w = sweep(f"{name}_whip{k}", [tuple(q) for q in seg], rr * 1.14, sides=sides, smooth_path=False)
        assign(w, whip)
        out.append(w)
    return out, dense


def _fray(name, end, dirv, rr, m, r, floor=None, strands=3, fibres=4):
    """The unlaid end of a line beyond its whipping: three strands opening out and a few loose yarns."""
    e = np.asarray(end, dtype=np.float64)
    d = np.asarray(dirv, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-9)
    s1 = np.cross(d, (0.0, 0.0, 1.0))
    if np.linalg.norm(s1) < 1e-3:
        s1 = np.array((1.0, 0.0, 0.0))
    s1 /= np.linalg.norm(s1)
    s2 = np.cross(s1, d)
    out = []

    def fl(q, rad):
        if floor is not None:
            q = np.array((q[0], q[1], max(q[2], floor + rad)))
        return tuple(q)

    a0 = r.u(0, TAU)
    for k in range(strands):
        a = a0 + TAU * k / strands + r.u(-0.3, 0.3)
        rd = cos(a) * s1 + sin(a) * s2
        ln = r.u(2.6, 4.0) * rr
        sp = r.u(0.25, 0.65)
        cu = r.u(-0.3, 0.3) * ln
        pts = [fl(e - d * 0.006 + rd * rr * 0.42, rr * 0.5), fl(e + d * ln * 0.4 + rd * (rr * 0.5 + sp * ln * 0.2), rr * 0.45),
               fl(e + d * ln * 0.75 + rd * (rr * 0.5 + sp * ln * 0.55) + s1 * cu * 0.4, rr * 0.4),
               fl(e + d * ln + rd * (rr * 0.5 + sp * ln) + s1 * cu, rr * 0.3)]
        o = sweep(f"{name}_s{k}", pts, rr * 0.5, sides=5, smooth_path=False, scale_fn=lambda t: 1.0 - 0.5 * t)
        out.append(o)
    for k in range(fibres):
        a = r.u(0, TAU)
        rd = cos(a) * s1 + sin(a) * s2
        ln = r.u(3.0, 5.5) * rr
        sp = r.u(0.3, 0.9)
        pts = [fl(e + rd * rr * 0.5, 0.002), fl(e + d * ln * 0.5 + rd * (rr * 0.6 + sp * ln * 0.35), 0.002),
               fl(e + d * ln + rd * (rr * 0.6 + sp * ln * 0.6) + s2 * r.u(-0.4, 0.2) * ln, 0.002)]
        out.append(sweep(f"{name}_f{k}", pts, 0.0013, sides=3, smooth_path=False, cap=False))
    for o in out:
        assign(o, m)
    return out


def _hitch(radfn, z_hi, turns, rr, r, a0, L=lambda q: q):
    """Line hitched round a post: a short whipped tail crossing over the turns, then `turns` round the
    post working downwards. Returns (points, end angle, end height)."""
    pitch = 2 * rr * 1.04
    ph = r.u(0, TAU)
    z_lo = z_hi - pitch * turns

    def P(a, z, off):
        rad = radfn(a, z) + off
        return L((rad * cos(a), rad * sin(a), z))

    pts = [P(a0 - 0.95, z_lo - 0.13, rr * 1.5), P(a0 - 0.84, z_lo - 0.04, rr * 2.3),
           P(a0 - 0.62, (z_hi + z_lo) / 2, rr * 3.2), P(a0 - 0.34, z_hi - pitch * 0.45, rr * 3.0),
           P(a0 - 0.14, z_hi - pitch * 0.15, rr * 2.0)]
    n = int(turns * 26)
    for i in range(n + 1):
        t = i / n
        th = a0 + TAU * turns * t
        z = z_hi - pitch * turns * t + 0.003 * sin(th * 0.5 + ph) - 0.008 * t * (0.5 + 0.5 * sin(th + ph))
        pts.append(P(th, z, rr * (1.04 + 0.12 * (0.5 + 0.5 * sin(th * 1.7 + ph)))))
    return pts, a0 + TAU * turns, z_lo


def _drape(xy, rr, passes=3):
    """Heights of a line laid along the plan path `xy` (dense, ~rr spacing) on flat ground: where it crosses
    or crowds a part laid earlier it rides up over it, ramping on and off like a stiff rope."""
    P = np.asarray(xy, dtype=np.float64)[:, :2]
    n = len(P)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    z = np.full(n, rr)
    for _ in range(passes):
        for i in range(n):
            j = int(np.searchsorted(s, s[i] - 7 * rr))
            if j <= 0:
                continue
            dd = np.hypot(P[:j, 0] - P[i, 0], P[:j, 1] - P[i, 1])
            m = dd < 2 * rr * 0.995
            z[i] = max(rr, float(np.max(z[:j][m] + np.sqrt((2 * rr) ** 2 - dd[m] ** 2)))) if m.any() else rr
        for k in range(1, n):
            z[k] = max(z[k], z[k - 1] - 0.4 * (s[k] - s[k - 1]))
        for k in range(n - 2, -1, -1):
            z[k] = max(z[k], z[k + 1] - 0.4 * (s[k + 1] - s[k]))
    zs = np.convolve(np.pad(z, 3, mode="edge"), np.ones(7) / 7.0, mode="valid")
    return np.maximum(z, zs)


def _rings_left(sides=7, lo=110, hi=262):
    """How many rope rings the triangle budget still allows after everything built so far."""
    try:
        used = sum(len(f.vertices) - 2 for o in bpy.data.objects if o.type == "MESH" for f in o.data.polygons)
    except Exception:
        return hi
    return int(clamp((META["tri_budget"] - 560 - used) / (2 * sides), lo, hi))


def _rope_dirt(z_ground, z_damp, seed):
    """Grime on a line: ground dirt where it lies, damp and tar on the lower turns / the fall, blotches along it."""
    nz = noise_mask(0.13, seed=seed, lo=0.25, hi=0.8)

    def fn(co, n):
        z = co[:, 2]
        ground = np.clip(1.0 - (z - z_ground) / 0.1, 0, 1) * 0.7
        damp = np.clip((z_damp - z) / 0.07, 0, 1) * 0.5
        return np.clip(np.maximum(ground, damp) + 0.3 * nz(co, n), 0, 1)
    return fn


# ------------------------------------------------------------------------------------------ timber pile
def _pile(p, rng):
    H, D, age = p.height, p.girth, p.age
    rb, rr = D / 2, p.rope_dia / 2
    capped, painted = p.head == "capped", p.head == "painted"
    tar = mat("wood_weathered", p.tar, rough=0.3, scale=0.7)
    iron = mat("iron_wrought")
    r = rng.fork("pile")
    ph = [r.u(0, TAU) for _ in range(10)]
    amp = 0.5 + age

    # ---- heights: band, cross pin, the turns of line (or the grooves old lines have cut)
    bw, tb = 0.05, 0.005
    hp = 0.21                                                    # depth of a painted head
    zband = H - (0.15 if capped else 0.09)
    zb0, zb1 = zband - bw / 2, zband + bw / 2
    zpin = min(H * 0.8, zband - 0.085)
    if painted:
        zpin = min(zpin, H - hp - 0.05)
    zrag = H - hp
    pitch = 2 * rr * 1.04
    z_hi = zpin - 0.03 - rr
    turns = max(0, min(int(p.rope_turns), int((z_hi - 0.25) / pitch))) if p.rope else 0
    z_lo = z_hi - pitch * turns
    g0, g1, gd = z_lo - rr, z_hi + rr, 0.003 + 0.003 * age
    grooves = []
    if turns < 1:
        zg = zpin - 0.06
        for _ in range(3 + int(r.chance(0.6))):
            if zg > 0.3:
                grooves.append((zg, r.u(0.012, 0.016), r.u(0.004, 0.0065)))
            zg -= r.u(0.042, 0.06)
    At = r.u(0.008, 0.015) if grooves else 0.0
    zgl, zgh = (grooves[-1][0], grooves[0][0]) if grooves else (0.0, 0.0)

    def tilt(a, z):                                              # a slack line never bears square to the post
        if not grooves:
            return 0.0
        return At * sin(a + ph[7]) * smoothstep(zgl - 0.11, zgl - 0.03, z) * (1.0 - smoothstep(zgh + 0.02, zgh + 0.055, z))

    def gcut(zeta):
        g = 0.0
        if turns >= 1:
            g = gd * min(_lin(g0 - 0.014, g0, zeta), 1.0 - _lin(g1, g1 + 0.014, zeta))
        for zg, hw, dp in grooves:
            g = max(g, dp * clamp((hw - abs(zeta - zg)) / (0.65 * hw)))
        return g

    def Rn(z):
        return rb * (1.0 - 0.1 * clamp(z / H)) * (1.0 + 0.06 * clamp(1 - z / 0.15) ** 2)

    Rseat = Rn(zband) * 0.945                                    # the head is dressed round to take the band

    def dress(z):
        return 0.25 + 0.75 * smoothstep(0.03, 0.3, zb0 - z)

    # ---- hewn section: nf uneven flats whose arrises wander up the stem, each flat scooped by the adze
    nf = int(clamp(round(D / 0.028), 8, 13))
    wts = [r.u(0.8, 1.2) for _ in range(nf)]
    b0 = r.u(0, TAU)
    bnd = [b0 + TAU * sum(wts[:k]) / sum(wts) for k in range(nf)]
    wnd = [(r.u(0, TAU), r.u(2.0, 5.0), r.u(0.012, 0.035)) for _ in range(nf)]
    scp = [(r.u(0, pi), r.u(0.15, 0.26), r.u(0.0015, 0.0035) * amp) for _ in range(nf)]
    e_ = 0.08

    def corner(k, z):
        kk = k % nf
        p0, w, am = wnd[kk]
        a = bnd[kk] + am * sin(w * z + p0) * dress(z) + (TAU if k >= nf else 0.0)
        return a, Rn(z) * (1.0 + (0.035 * sin(2 * a + ph[0] + 0.8 * z) + 0.02 * sin(3 * a + ph[1] + 2.2 * z)) * amp * dress(z))

    # drying checks: narrow V splits from the head down (and a shake or two lower in old timber)
    order = sorted(range(nf), key=lambda k: r.u())
    ntop, nsh = int(round(3 + 2 * age)), int(round(2 * age))
    chord = 2 * Rn(H * 0.6) * sin(pi / nf)
    checks, chk_of = [], {}
    for i, k in enumerate(order[:min(nf, ntop + nsh)]):
        top = i < ntop
        if top:
            z1, z0 = H + 1.0, max(0.1, H - r.u(0.16, 0.4))
        else:
            z1 = r.u(0.45, 0.72) * (zb0 - 0.05)
            z0 = max(0.08, z1 - r.u(0.2, 0.4))
        chk_of[k] = len(checks)
        checks.append({"z0": z0, "z1": z1, "top": top, "d": r.u(0.007, 0.012) * (0.7 + 0.6 * age), "s": r.u(0.27, 0.73),
                       "ds": r.u(0.0013, 0.002) / chord})

    def cdepth(ci, z):
        c = checks[ci]
        if z <= c["z0"] or z > c["z1"]:
            return 0.0
        if c["top"]:
            return c["d"] * clamp((z - c["z0"]) / (H - c["z0"])) ** 0.6
        return c["d"] * sin(pi * (z - c["z0"]) / (c["z1"] - c["z0"])) ** 0.6

    cols = []                                                    # (facet, position along its chord, kind, check)
    for k in range(nf):
        cols += [(k, 0.0, "arris", None), (k, e_, "edge", None)]
        ci = chk_of.get(k)
        if ci is None:
            cols.append((k, 0.5, "mid", None))
        else:
            s, ds = checks[ci]["s"], checks[ci]["ds"]
            cols += [(k, s - ds, "side", None), (k, s, "chk", ci), (k, s + ds, "side", None)]
        cols.append((k, 1 - e_, "edge", None))
    nc = len(cols)

    def colpt(c, z, check=True):
        k, s, kind, ci = cols[c]
        a0_, q0 = corner(k, z)
        if kind == "arris":
            a, q = a0_, q0 - 0.0018
        else:
            a1_, q1 = corner(k + 1, z)
            x, y = lerp(q0 * cos(a0_), q1 * cos(a1_), s), lerp(q0 * sin(a0_), q1 * sin(a1_), s)
            a, q = atan2(y, x), hypot(x, y)
            sp, sl, sd = scp[k]
            q -= sd * abs(sin(pi * z / sl + sp)) ** 0.6 * clamp((min(s, 1 - s) - e_) / (0.5 - e_)) * dress(z)
        g = gcut(z - tilt(a, z))
        if g > 0.0:                                              # the line has rubbed the arrises round
            q = lerp(q, min(q, Rn(z) * 0.985), min(1.0, g / 0.002)) - g
        q = lerp(Rseat, q, 1.0 - _lin(zb0 - 0.03, zb0 - 0.004, z))
        if check and ci is not None:
            q -= cdepth(ci, z)
        return a, q

    scols = [c for c in range(nc) if cols[c][2] != "chk"]
    ring_cache = {}

    def surf(a, z):
        """Radius of the pile's surface (checks bridged) in direction a at height z."""
        z = min(z, zb1)
        key = round(z, 5)
        un = ring_cache.get(key)
        if un is None:
            pts = [colpt(c, z, False) for c in scols]
            base = pts[0][0]
            un = [(base + ((ca - base) % TAU), cq) for ca, cq in pts]
            un.append((base + TAU, pts[0][1]))
            if len(ring_cache) > 4000:
                ring_cache.clear()
            ring_cache[key] = un
        t = un[0][0] + ((a - un[0][0]) % TAU)
        for (a_i, q_i), (a_j, q_j) in zip(un, un[1:]):
            if a_i <= t <= a_j:
                dj = a_j - a_i
                if dj < 1e-6:
                    return q_i
                return q_i * q_j * sin(dj) / (q_i * sin(t - a_i) + q_j * sin(a_j - t))
        return un[0][1]

    # ---- rows of the stem (finer where grooves, the paint line and the shoulder need them)
    marks = {}

    def mark(z, pr):
        if 0.016 <= z <= zb0 - 0.004 + 1e-9:
            marks[z] = max(pr, marks.get(z, 0))

    mark(0.016, 9), mark(zb0 - 0.03, 9), mark(zb0 - 0.004, 9)
    for z in (0.1, 0.17, 0.235):
        if z < zb0 - 0.07:
            mark(z, 1)
    if turns >= 1:
        for z in (g0 - 0.014, g0, g1, g1 + 0.014):
            mark(z, 5)
    for zg, hw, dp in grooves:
        for z in (zg - hw, zg - 0.35 * hw, zg + 0.35 * hw, zg + hw):
            mark(z, 6)
    if painted:
        mark(zrag - 0.03, 7), mark(zrag, 8), mark(zrag + 0.03, 7)
    zs = []
    for z in sorted(marks):
        if zs and z - zs[-1] < 0.007:
            if marks[z] > marks[zs[-1]]:
                zs[-1] = z
        else:
            zs.append(z)
    rows = [zs[0]]
    for za, zc in zip(zs, zs[1:]):
        n = max(1, int(ceil((zc - za) / (0.078 * max(1.0, H) ** 0.8) - 0.3)))   # rows: ~8 cm, coarser on tall piles (budget)
        rows += [za + (zc - za) * i / n for i in range(1, n + 1)]
    irag = min(range(len(rows)), key=lambda j: abs(rows[j] - zrag)) if painted else None

    def ragf(a):                                                 # ragged brushed edge of a painted head
        return 0.011 * sin(5 * a + ph[8]) + 0.006 * sin(13 * a + ph[9])

    verts, faces, fmat = [], [], []
    for j, zeta in enumerate(rows):
        for c in range(nc):
            z = zeta
            a, _ = colpt(c, z)
            if j == irag:
                z += ragf(a) + r.u(-0.004, 0.004)
            z += tilt(a, z)
            a, q = colpt(c, z)
            verts.append((q * cos(a), q * sin(a), z))

    # ---- head: band seat, then a slightly mushroomed, nicked rim, an axe chamfer and the weathered end grain
    ahead = [colpt(c, zb0)[0] for c in range(nc)]
    hjit = [r.u(-1, 1) for _ in range(nc)]
    edge_cols = [c for c in range(nc) if cols[c][2] == "edge"]
    nicks = {r.pick(edge_cols) for _ in range(2 + int(2 * age))}

    def lip(c, a):
        return 0.002 * sin(3 * a + ph[4]) + 0.0015 * hjit[c] - (0.006 if c in nicks else 0.0)

    def chw(a):
        return 0.004 * sin(2 * a + ph[5]) + 0.003 * sin(5 * a + ph[6])

    if capped:
        head = [(zb1 + 0.003, lambda c, a: Rseat + 0.0005), (H - 0.022, lambda c, a: Rseat + 0.001),
                (H - 0.005, lambda c, a: Rseat - 0.012 + 0.3 * chw(a))]
    else:
        head = [(zb1 + 0.003, lambda c, a: Rseat + 0.0008), (H - 0.037, lambda c, a: Rseat + 0.002 + 0.5 * lip(c, a)),
                (H - 0.021, lambda c, a: Rseat + 0.0045 + lip(c, a)), (H - 0.004, lambda c, a: Rseat - 0.013 + chw(a) + 0.5 * lip(c, a))]
    rtop = []
    for (z, fn) in head:
        for c in range(nc):
            a = ahead[c]
            q = fn(c, a)
            ci = cols[c][3]
            if ci is not None:
                q -= cdepth(ci, min(z, H)) * 1.2
            verts.append((q * cos(a), q * sin(a), z + (0.002 * sin(4 * a + ph[3]) if z > H - 0.01 and not capped else 0.0)))
            if z > H - 0.01:
                rtop.append(q)
    zt = H - (0.005 if capped else 0.004)
    for f in (0.7, 0.36):
        for c in range(nc):
            a = ahead[c]
            z = zt + 0.005 * (1 - f * f) + (0.0 if capped else 0.0015 * sin(3 * a + 5 * f + ph[2]))
            ci = cols[c][3]
            if ci is not None and checks[ci]["top"]:
                z -= checks[ci]["d"] * 0.9 * f ** 0.7
            verts.append((rtop[c] * f * cos(a), rtop[c] * f * sin(a), z))
    verts.append((0.0, 0.0, zt + 0.005))
    nr = len(rows) + len(head) + 2
    for j in range(nr - 1):
        mi = 1 if j >= len(rows) + len(head) - 1 else (2 if painted and j >= irag else 0)
        for i in range(nc):
            i2 = (i + 1) % nc
            faces.append((j * nc + i, j * nc + i2, (j + 1) * nc + i2, (j + 1) * nc + i))
            fmat.append(mi)
    for i in range(nc):
        faces.append(((nr - 1) * nc + i, (nr - 1) * nc + (i + 1) % nc, len(verts) - 1))
        fmat.append(1)
    pile = mesh_object("pile", verts, faces)
    if not capped:                                               # the saw cut is never quite square
        tl = (r.u(-0.03, 0.03), r.u(-0.03, 0.03))
        transform_verts(pile, lambda co: Vector((co.x, co.y, co.z + (co.x * tl[0] + co.y * tl[1]) * smoothstep(H - 0.03, H, co.z))))
    paintm = mat("wood_painted", p.head_paint, rough=1.35) if painted else None
    assign(pile, tar)
    assign(pile, paintm if painted else mat("wood_endgrain", "#4d433a"), slot=1)
    if painted:
        assign(pile, paintm, slot=2)
    if len(pile.data.polygons) == len(fmat):
        pile.data.polygons.foreach_set("material_index", np.array(fmat, dtype=np.int32))
    smooth(pile, 38)
    uv_grain(pile, axis="Z")
    leaned = [pile]

    def surf_m(a, z):
        """Surface of the mesh itself (straight between the rows of the stem) - what stains and runs must lie on."""
        if z >= rows[-1] or z <= rows[0]:
            return surf(a, z)
        j = int(np.searchsorted(rows, z)) - 1
        t = (z - rows[j]) / max(rows[j + 1] - rows[j], 1e-6)
        return lerp(surf(a, rows[j]), surf(a, rows[j + 1]), t)

    def strip(name, a, z0, length, width, bead, m, wander=0.03, bulb=True, taper=0.5):
        """A run of tar / paint, or a rust stain, down the surface of the pile: thin where it starts, a neck and a
        hanging drop where it stopped (bulb)."""
        length = min(length, z0 - 0.07)
        if length < 0.03:
            return None
        n = max(3, int(ceil(length / (0.022 if bulb else 0.03))))
        ts = [i / n for i in range(n + 1)]
        if bulb:
            ts = sorted(set(t for t in ts if (1 - t) * length > 0.034) | {1 - d / length for d in (0.03, 0.02, 0.012, 0.005, 0.0)})
        vs, fs = [], []
        pa = r.u(0, TAU)
        for i, t in enumerate(ts):
            z = z0 - length * t
            d = (1 - t) * length
            ac = a + wander * sin(2.6 * t + pa) * t
            w = width * 0.5 * (1.0 - taper * t)
            hb = bead * (0.7 + 0.3 * t)
            if bulb:
                w *= (0.4 + 0.6 * min(1.0, t * length / 0.02)) * (1.0 + 0.25 * sin(9 * t + pa))
                w *= (1.0 + 1.3 * exp(-((d - 0.011) / 0.007) ** 2)) * (1.0 - 0.4 * exp(-((d - 0.03) / 0.008) ** 2))
                hb *= 1.0 + 1.0 * exp(-((d - 0.011) / 0.008) ** 2)
                if d < 0.002:
                    w *= 0.25
                    hb *= 0.4
            elif i == len(ts) - 1:
                w *= 0.3
                hb *= 0.4
            q0 = surf_m(ac, z)
            for s_, off in ((-1, 0.0005), (0, hb), (1, 0.0005)):
                aa = ac + s_ * w / q0
                q = surf_m(aa, z) + off
                vs.append((q * cos(aa), q * sin(aa), z))
            if i:
                b_, c_ = 3 * (i - 1), 3 * i
                fs += [(b_, c_, c_ + 1, b_ + 1), (b_ + 1, c_ + 1, c_ + 2, b_ + 2)]
        o = mesh_object(name, vs, fs)
        assign(o, m)
        smooth(o, 60)
        return o

    # ---- iron cap band: forged strap driven on to the dressed head, lapped and riveted, clout-nailed
    rband = Rseat + 0.0003
    band = ring_band("cap_band", rband, bw, tb, loc=(0, 0, zband), segments=36, bulge=0.0012)
    al = r.u(0, TAU)
    lap = box("band_lap", (0.08, 0.005, bw * 0.98), loc=((rband + tb + 0.0012) * cos(al), (rband + tb + 0.0012) * sin(al), zband),
              rot=(0, 0, degrees(al) + 90), bevel=0.0015)
    transform_verts(lap, lambda co: Vector((co.x, co.y - 0.9 * co.x * co.x / rband, co.z)))   # follow the hoop
    rv = bolt("rivet", 0.0075, 0.0045, kind="dome")
    rm = [trs(((rband + tb + 0.0035) * cos(al + s), (rband + tb + 0.0035) * sin(al + s), zband), (0, 90, degrees(al + s)))
          for s in (-0.02 / rband, 0.02 / rband)]
    nl = bolt("nail", 0.006, 0.004, kind="pyramid")
    nn = 5
    nail_a = [al + TAU * (k + 0.5 + r.u(-0.12, 0.12)) / nn for k in range(nn)]
    nm = [trs(((rband + tb + 0.0006) * cos(a), (rband + tb + 0.0006) * sin(a), zband + r.u(-0.008, 0.008)), (0, 90, degrees(a))) for a in nail_a]
    irons = [band, lap, instances_to_mesh("rivets", rv, rm, keep_base=False), instances_to_mesh("nails", nl, nm, keep_base=False)]

    # ---- cross pin (keeps a line from riding up and off the head)
    ap = r.u(0, 180)
    apr = radians(ap)
    lp = D * 0.9 + 0.2
    irons.append(cylinder("pin", 0.0125, lp, loc=(0, 0, zpin), rot=(0, 90, ap), verts=10))
    for s, qa in ((1, apr), (-1, apr + pi)):
        irons.append(cylinder("pin_head", 0.019, 0.016, loc=(s * (lp / 2) * cos(apr), s * (lp / 2) * sin(apr), zpin),
                              rot=(0, 90, ap), verts=10, bevel=0.004))
        qc = surf(qa, zpin) + 0.002
        irons.append(cylinder("pin_collar", 0.024, 0.008, loc=(qc * cos(qa), qc * sin(qa), zpin), rot=(0, 90, ap), verts=10, bevel=0.002))
    for o in irons:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.05, seed=p.seed + 3, lo=0.32, hi=0.72), (0.3 + 0.45 * age) * (1.0 if o.name.startswith(("cap_band", "band_lap")) else 0.45)), wear=0.35)
    leaned += irons

    def clear_pin(a, lim=0.3):
        return min(abs(((a - apr + pi / 2) % pi) - pi / 2), 9.0) > lim

    # ---- rust weeping from the band, runs of tar (or of head paint) down the stem
    rust = mat("iron_rusty", rough=1.2)
    stains = []
    for k, a in enumerate(nail_a + [al - 0.3, al + 0.3]):
        if r.chance(0.35 + 0.6 * age):
            ln = r.u(0.05, 0.17) * (0.6 + 0.8 * age)
            if not clear_pin(a, 0.2):
                ln = min(ln, zb0 - zpin - 0.03)
            stains.append(strip(f"rust{k}", a + r.u(-0.03, 0.03), zb0 - 0.0005, ln, r.u(0.007, 0.015), 0.0007, rust,
                                wander=0.02, bulb=False, taper=0.85))
    stains = [o for o in stains if o is not None]
    for o in stains:
        paint(o, grime=lambda co, n: np.clip((zb0 - 0.03 - co[:, 2]) / 0.12, 0, 0.85))
    gloss = mat("ceramic_glazed", "#2b231e", rough=1.1, wear=0.0)
    runs = []
    ztar = (zrag - 0.05) if painted else (zb0 - 0.006)
    for k in range(10 + int(r.u(0, 3))):
        a = r.u(0, TAU)
        ln = r.u(0.07, 0.3)
        z0 = ztar - (0.0 if k < 4 and not painted else r.u(0.0, 0.25) * H)
        if turns >= 1 and z0 > g1:
            ln = min(ln, z0 - g1 - 0.012)                        # stops short of the turns
        if z0 > zpin and not clear_pin(a):
            ln = min(ln, z0 - zpin - 0.03)
        if turns >= 1 and g0 - 0.02 < z0 <= g1:
            z0 = g0 - 0.02 - r.u(0, 0.1)
        if ln > 0.035 and z0 - ln > 0.1:
            runs.append(strip(f"tar_run{k}", a, z0, ln, r.u(0.0045, 0.008), r.u(0.002, 0.0032), gloss, taper=0.45))
    if painted:
        for k in range(6):
            a = r.u(0, TAU)
            ln = r.u(0.025, 0.085)
            if not clear_pin(a):
                continue
            runs.append(strip(f"paint_run{k}", a, zrag + ragf(a) + 0.008, ln + 0.008, r.u(0.006, 0.01), r.u(0.0012, 0.002), paintm, wander=0.015))
    leaned += stains + [o for o in runs if o is not None]

    # ---- lead cap: one sheet dressed down over the head, the surplus gathered in pleats, lapped seam, clout nails
    if capped:
        rc = r.fork("cap")
        Rt, t0 = Rseat + 0.001, 0.003
        npl = 5 + int(rc.chance(0.5))
        asm = rc.u(0, TAU)
        pl = [(asm + TAU * (k + 0.5 + rc.u(-0.2, 0.2)) / npl, rc.u(0.075, 0.115), rc.u(0.006, 0.011), rc.pick((-1, 1))) for k in range(npl)]

        def adiff(a, b):
            return ((a - b + pi) % TAU) - pi

        ang = [TAU * i / 24 for i in range(24)]
        ang = [a for a in ang if all(abs(adiff(a, pa)) > hw * 1.3 for pa, hw, _, _ in pl) and abs(adiff(a, asm)) > 0.07]
        for pa, hw, hh, sg in pl:
            ang += [pa - hw, pa + sg * hw * 0.4, pa + hw]
        ang += [asm - 0.013, asm + 0.013]
        ang = sorted(a % TAU for a in ang)

        def seam(a):
            d = adiff(a, asm)
            return 0.0045 * clamp(1.0 + d / 0.35) if d <= 0 else 0.0

        def off(a, v):
            o = t0 + seam(a)
            for pa, hw, hh, sg in pl:
                d, pk = adiff(a, pa), sg * hw * 0.4
                if -hw < d < hw:
                    u = (d + hw) / (pk + hw) if d < pk else (hw - d) / (hw - pk)
                    o += hh * u * (1.0 - 0.9 * v)
            return o

        na = len(ang)
        skz = [H - 0.092 + 0.005 * sin(3 * a + ph[4]) + rc.u(-0.003, 0.003) for a in ang]
        cv, cf = [], []
        zsh = H - 0.022
        spec = [(0.0, 0.0006, None), (0.0, None, 0.0), (0.45, None, 0.45), (1.0, None, 1.0)]
        for (v, fixed, vo) in spec:
            for i, a in enumerate(ang):
                q = Rt + (fixed if fixed is not None else off(a, vo))
                z = skz[i] + (zsh - skz[i]) * v + (0.003 if fixed is not None else 0.0)
                cv.append((q * cos(a), q * sin(a), z))
        for a in ang:
            q = Rseat - 0.012 + 0.3 * chw(a) + t0 * 0.8 + seam(a) * 0.5
            cv.append((q * cos(a), q * sin(a), H - 0.005 + t0 + seam(a) * 0.6))
        for f in (0.62, 0.28):
            for a in ang:
                q = (Rseat - 0.012) * f
                cv.append((q * cos(a), q * sin(a), H + 0.005 * (1 - f * f) + t0 + 0.0005 + seam(a) * 0.8))
        cv.append((0.0, 0.0, H + 0.005 + t0 + 0.0015))
        ncr = len(spec) + 3
        for j in range(ncr - 1):
            for i in range(na):
                i2 = (i + 1) % na
                cf.append((j * na + i, j * na + i2, (j + 1) * na + i2, (j + 1) * na + i))
        for i in range(na):
            cf.append(((ncr - 1) * na + i, (ncr - 1) * na + (i + 1) % na, len(cv) - 1))
        cap = mesh_object("lead_cap", cv, cf)
        displace_noise(cap, strength=0.0012, scale=0.045, seed=p.seed + 5)
        assign(cap, mat("lead", rough=1.1))
        smooth(cap, 42)
        uv_cyl(cap, axis="Z")
        nzc = noise_mask(0.05, seed=p.seed + 7, lo=0.25, hi=0.75)
        paint(cap, grime=lambda co, n: np.clip(0.08 + 0.3 * nzc(co, n) + np.clip((H - 0.06 - co[:, 2]) / 0.04, 0, 1) * 0.15, 0, 0.6))
        cn = bolt("cap_nail", 0.0062, 0.0032, kind="dome")
        cm = []
        for k in range(9):
            a = TAU * (k + rc.u(-0.15, 0.15)) / 9
            if all(abs(adiff(a, pa)) > hw * 1.15 for pa, hw, _, _ in pl):
                cm.append(trs(((Rt + off(a, 0.3) - 0.0006) * cos(a), (Rt + off(a, 0.3) - 0.0006) * sin(a), H - 0.07 + rc.u(-0.004, 0.004)), (0, 90, degrees(a))))
        for dz in (-0.078, -0.04):                               # the lap of the seam is nailed twice
            a = asm - 0.06
            cm.append(trs(((Rt + off(a, 0.5) - 0.0006) * cos(a), (Rt + off(a, 0.5) - 0.0006) * sin(a), H + dz), (0, 90, degrees(a))))
        nails = instances_to_mesh("cap_nails", cn, cm, keep_base=False)
        assign(nails, iron)
        paint(nails, special=0.2)
        leaned += [cap, nails]

    # ---- pull it out of plumb
    az = r.u(0, TAU)
    ex, ey = p.lean * cos(az), p.lean * sin(az)
    M = Euler((radians(ex), radians(ey), 0.0), "XYZ").to_matrix()
    ab, Ab = r.u(0, TAU), 0.009 * H * (0.5 + age)             # a grown stem is never straight

    def bow(z):
        k = Ab * sin(pi * 0.85 * clamp(z / H)) ** 2
        return Vector((k * cos(ab), k * sin(ab), 0.0))

    for o in leaned:
        apply_transform(o)
        transform_verts(o, lambda co: co + bow(co.z))
        place(o, rot=(ex, ey, 0))

    def L(q):
        v = M @ (Vector(q) + bow(q[2]))
        return (v.x, v.y, v.z)

    # ---- collar of granite setts wedged round the foot
    rs = rng.fork("collar")
    n = 8 if D < 0.31 else 9
    wts = [rs.u(0.8, 1.2) for _ in range(n)]
    a = rs.u(0, TAU)
    r_out = rb + 0.13
    stone = mat("stone_rough", "#8f8b84")
    setts = []
    for k in range(n):
        span = TAU * wts[k] / sum(wts)
        g = 0.007 / (rb + 0.06)
        a0, a1, am = a + g, a + span - g, a + span / 2
        ri, ro = rb * 0.93, rb + rs.u(0.095, 0.13)
        pts = [(ro * cos(a0), ro * sin(a0)), (ro * 1.03 * cos(am), ro * 1.03 * sin(am)), (ro * cos(a1), ro * sin(a1)),
               (ri * cos(a1), ri * sin(a1)), (ri * 0.94 * cos(am), ri * 0.94 * sin(am)), (ri * cos(a0), ri * sin(a0))]
        hs = rs.u(0.034, 0.056)
        s = extrude_profile(f"sett{k}", pts, hs, bevel=0.008, segments=1)
        assign(s, stone)
        paint(s, grime=0.3 + 0.25 * age, special=mask_mul(noise_mask(0.08, seed=p.seed + k, lo=0.35, hi=0.7), 0.3 + 0.6 * age))
        setts.append((a, a + span, hs, ro))
        a += span

    def sett_at(a, margin=0.0):
        """(top height, outer radius) of the collar under direction a."""
        best = None
        for (s0, s1, hs, ro) in setts:
            for d in (-margin, 0.0, margin):
                if ((a + d - s0) % TAU) <= (s1 - s0) and (best is None or hs > best[0]):
                    best = (hs, ro)
        return best or (setts[0][2], setts[0][3])

    # ---- the line: hitched round the pile under the pin, hanging down the timber, over the collar, to a coil
    zw = None
    if turns >= 1:
        rp = mat("rope", p.rope_tint, grime=1.1)
        whip = mat("rope", WHIP)
        rr_ = rng.fork("rope")
        pts, ae, _ = _hitch(surf, z_hi, turns, rr, rr_, rr_.u(0, TAU), L)
        zw = (z_hi + z_lo) / 2

        def G(a, dist, z):
            return (dist * cos(a), dist * sin(a), z)

        def S(da, off, z):
            return L(G(ae + da, surf(ae + da, z) + rr * off, z))

        af = ae + 0.56
        hs, ro_s = sett_at(af + 0.07, rr / (rb + 0.06))
        zfoot = hs + rr + 0.085                                  # where the hanging part starts to bend out
        drop = min(0.14, max(0.03, (z_lo - zfoot) * 0.7))
        for da, fz in ((0.16, 0.045), (0.32, 0.16), (0.45, 0.36), (0.53, 0.64), (0.56, 1.0)):
            pts.append(S(da, 1.1, z_lo - drop * fz))
        z = z_lo - drop - 0.09
        sway = rr_.u(0, TAU)
        while z > zfoot + 0.03:                                  # dead weight: it hangs against the timber
            pts.append(S(0.56 + 0.012 * sin(z * 9 + sway), 1.08 + 0.1 * (0.5 + 0.5 * sin(z * 13 + sway)), z))
            z -= 0.1
        q0 = surf(af, hs + 0.05)
        pts += [S(0.56, 1.12, zfoot), S(0.565, 1.45, hs + rr + 0.04),
                G(af + 0.015, q0 + rr * 2.5, hs + rr + 0.011), G(af + 0.03, q0 + rr * 4.3, hs + rr + 0.001),
                G(af + 0.06, max(q0 + rr * 5.5, ro_s - 0.014), hs + rr),
                G(af + 0.085, ro_s + rr * 0.95, hs * 0.62 + rr + 0.003)]
        # plan of the part lying on the quay: a short run with a kink, a flat coil, a second layer flaked back over it
        sg = rr_.pick((-1, 1))
        rc0 = 0.125 + rr
        ac = af + sg * rr_.u(0.75, 1.0)
        dc = r_out + 0.06 + rc0
        C = np.array((dc * cos(ac), dc * sin(ac)))
        E = np.array(G(af + 0.11, ro_s + rr * 2.4 + 0.02, 0.0)[:2])
        v = E - C
        f0 = atan2(v[1], v[0]) + sg * acos(clamp(rc0 / max(np.linalg.norm(v), 1e-6), -1.0, 1.0))
        T = C + rc0 * np.array((cos(f0), sin(f0)))
        side = np.array((-(T - E)[1], (T - E)[0])) / max(np.linalg.norm(T - E), 1e-6)
        plan = [E, E + (T - E) * 0.4 + side * rr_.u(0.012, 0.022) * sg, E + (T - E) * 0.72 - side * rr_.u(0.006, 0.014) * sg]
        nb_, nt_ = rr_.u(2.15, 2.45), rr_.u(1.45, 1.75)
        pb, pt = 2 * rr * 1.12, 2 * rr * 1.22
        j = [rr_.u(0, TAU) for _ in range(5)]
        m = int(nb_ * 30)
        for i in range(m + 1):
            t = i / m
            f = f0 + sg * TAU * nb_ * t
            q = rc0 - pb * nb_ * t + (0.005 * sin(f + j[0]) + 0.0042 * sin(2.3 * f + j[1])) * min(1.0, 6 * t)
            plan.append(C + q * np.array((cos(f), sin(f))))
        f1, q1 = f0 + sg * TAU * nb_, rc0 - pb * nb_
        dr = 0.014 * np.array((cos(j[4]), sin(j[4])))
        m = int(nt_ * 30)
        for i in range(1, m + 1):
            t = i / m
            f = f1 + sg * TAU * nt_ * t
            q = q1 + 0.8 * rr * min(1.0, 6 * t) + pt * nt_ * t + (0.007 * sin(1.7 * f + j[2]) + 0.004 * sin(3.1 * f + j[3])) * min(1.0, 4 * t)
            plan.append(C + dr * t + q * np.array((cos(f), sin(f))))
        f2, q2 = f1 + sg * TAU * nt_, q1 + 0.8 * rr + pt * nt_
        for u in (0.3, 0.6, 0.85, 1.0):                          # the end flaked out across the coil on to the setts
            f = f2 + sg * 0.55 * u
            q = q2 + (rc0 + 4.2 * rr - q2) * u ** 1.3
            plan.append(C + dr + q * np.array((cos(f), sin(f))))
        plan = resample_path([(q[0], q[1], 0.0) for q in plan], step=rr, smooth_path=False)
        zc = _drape(plan, rr)
        pts += [(plan[i][0], plan[i][1], float(zc[i])) for i in range(len(plan))]
        objs, dense = _rope("line", pts, rr, rp, whip, max_rings=_rings_left())
        objs += _fray("fray_a", dense[0], dense[0] - dense[3], rr, rp, rr_)
        objs += _fray("fray_b", dense[-1], dense[-1] - dense[-4], rr, rp, rr_, floor=0.0)
        dirt = _rope_dirt(0.0, z_lo + 1.6 * pitch, p.seed + 11)
        for o in objs:
            paint(o, grime=dirt)

    # ---- weathering: damp and algae skirt at the quay, worn timber where lines bear, chipped paint, weathered head
    nt1 = noise_mask(0.09, seed=p.seed + 1, lo=0.2, hi=0.8)
    nt2 = noise_mask(0.16, seed=p.seed + 2, lo=0.35, hi=0.8)
    nt3 = noise_mask(0.05, seed=p.seed + 9, lo=0.3, hi=0.62)
    nt4 = noise_mask(0.025, seed=p.seed + 4, lo=0.5, hi=0.68)
    nt5 = noise_mask(0.07, seed=p.seed + 6, lo=0.3, hi=0.7)
    nt6 = noise_mask(0.075, seed=p.seed + 10, lo=0.45, hi=0.78)
    nt7 = noise_mask(0.13, seed=p.seed + 12, lo=0.25, hi=0.8)
    zsk = 0.15 + 0.03 * age

    def grime_f(co, n):
        z = co[:, 2]
        skirt = np.clip(1.0 - (z - (zsk + 0.06 * (nt1(co, n) - 0.5))) / 0.06, 0, 1) * (0.72 + 0.25 * age)
        under = np.clip(1.0 - np.abs(z - (zb0 - 0.035)) / 0.045, 0, 1) * 0.45
        return np.maximum.reduce([skirt, nt2(co, n) * 0.3, under])

    def algae_f(co, n):
        z = co[:, 2]
        return np.clip(1.0 - np.abs(z - (zsk - 0.03)) / 0.12, 0, 1) * nt5(co, n) * (0.45 + 0.5 * age)

    def wear_f(co, n):
        z = co[:, 2]
        wside = np.clip(0.45 + 0.9 * n[:, 1], 0, 1)                              # the weather side has gone grey
        w = np.maximum(nt6(co, n) * (0.24 + 0.2 * age),                         # old tar weathers off in patches
                       (0.22 + 0.78 * nt7(co, n)) * wside * (0.46 + 0.3 * age)) * np.clip((z - zsk) / 0.1, 0, 1)
        if zw is not None:
            w = np.maximum(w, np.clip(1.7 - 1.7 * np.abs(z - zw) / ((z_hi - z_lo) / 2 + 0.04), 0, 1) * nt3(co, n) * 0.6)
        for zg, hw, dp in grooves:
            w = np.maximum(w, np.clip(1.3 - np.abs(z - zg) / hw, 0, 1) * 0.55)
        if grooves:
            w = np.maximum(w, np.clip(1.0 - np.abs(z - (zgl + zgh) / 2) / ((zgh - zgl) / 2 + 0.05), 0, 1) * (0.2 + 0.25 * nt3(co, n)))
        if painted:
            chips = np.maximum.reduce([np.clip((z - (H - 0.045)) / 0.03, 0, 1) * 0.55 + 0.25,
                                       np.clip(1.0 - np.abs(z - zb0) / 0.016, 0, 1) * 0.6,
                                       np.clip(1.0 - np.abs(z - zb1) / 0.02, 0, 1) * 0.6,
                                       np.clip(1.0 - (z - zrag) / 0.028, 0, 1) * 0.4])
            w = np.where(z > zrag - 0.04, np.maximum(w, chips * (0.35 + 0.65 * nt4(co, n)) * (0.6 + 0.5 * age)), w)
        elif not capped:
            w = np.maximum(w, np.clip((z - (H - 0.04)) / 0.025, 0, 1) * 0.5 * nt5(co, n))
        return w

    def special_f(co, n):
        s = algae_f(co, n)
        if painted:                                              # sun-faded head
            s = np.where(co[:, 2] > zrag + 0.024, 0.2 + 0.25 * nt2(co, n), s)
        return s

    paint(pile, grime=grime_f, wear=wear_f, special=special_f)
    socket("line", "attach", L((0, 0, zpin)), role="mooring")


# ---------------------------------------------------------------------------------- cast-iron bollard
def _bollard(p, rng):
    H, D, age = min(p.height, 1.1), p.girth, p.age
    rc = D / 2
    rr = p.rope_dia / 2
    iron = mat("iron_painted", p.paint, rough=0.8)
    zp = 0.05
    h = H - zp
    rf, rh = rc * 1.75, rc * 1.6
    half = max(rc * 2.15, rf + 0.07)

    pad = stone_block("pad", (half * 2, half * 2, zp), loc=(0, 0, zp / 2), rot=(0, 0, rng.u(-5, 5)), chamfer=0.012,
                      rough=0.003, rng=rng.fork("pad"), detail=0.06)
    assign(pad, mat("stone_rough", "#8f8b84"))
    npd = noise_mask(0.05, seed=p.seed + 8, lo=0.25, hi=0.75)
    paint(pad, grime=lambda co, n: np.clip(1.0 - (np.hypot(co[:, 0], co[:, 1]) - rf) / 0.06, 0, 1) * (0.2 + 0.25 * npd(co, n)) + 0.12,
          special=mask_mul(noise_mask(0.07, seed=p.seed, lo=0.45, hi=0.75), 0.25 + 0.4 * age))

    def bead(r0, z, s=1.0):
        return [(r0, z - 0.013 * s), (r0 + 0.008 * s, z - 0.008 * s), (r0 + 0.012 * s, z), (r0 + 0.008 * s, z + 0.008 * s), (r0, z + 0.013 * s)]

    zb2 = 0.65 * h
    prof = [(rf, 0.0), (rf, 0.024), (rf - 0.008, 0.032), (rc * 1.3, 0.037), (rc * 1.14, 0.056), (rc * 1.05, 0.095), (rc, 0.125)]
    prof += bead(rc, 0.15)
    zc0, zc1 = 0.175, zb2 - 0.02
    for k in range(1, 6):
        t = k / 6
        prof.append((rc * (1.0 - 0.07 * sin(pi * t) - 0.04 * t), zc0 + (zc1 - zc0) * t))
    prof += bead(rc * 0.96, zb2)
    prof += [(rc * 0.98, 0.69 * h), (rc * 1.08, 0.735 * h), (rc * 1.3, 0.785 * h), (rc * 1.52, 0.82 * h), (rh, 0.84 * h), (rh + 0.004, 0.855 * h),
             (rh + 0.004, 0.875 * h)]
    for k in range(1, 7):
        a = radians(90 * k / 7)
        prof.append((rh * cos(a) ** 0.85, 0.88 * h + 0.105 * h * sin(a)))
    prof += [(0.0, 0.985 * h)]
    prof = [(q, z + zp) for (q, z) in prof]
    body = lathe("bollard", prof, segments=30)
    objs = [body]

    def rad(a, z):
        for (q0, z0), (q1, z1) in zip(prof, prof[1:]):
            if z0 <= z <= z1 and z1 > z0:
                return q0 + (q1 - q0) * (z - z0) / (z1 - z0)
        return rc

    # cast webs between the holding-down bolts, bolts on square washers
    web = extrude_profile("web", [(rc * 1.0, 0.03), (rc * 1.6, 0.03), (rc * 1.6, 0.04), (rc * 1.02, 0.135)], 0.016, bevel=0.003, segments=1)
    wm, bm = [], []
    a0 = rng.u(0, 60)
    for k in range(6):
        a = radians(a0 + k * 60)
        wm.append(trs((0.008 * sin(a), -0.008 * cos(a), zp), (90, 0, degrees(a))))
        ab = a + radians(30)
        bm.append(trs((rc * 1.53 * cos(ab), rc * 1.53 * sin(ab), zp + 0.033), (0, 0, rng.u(0, 60))))
    objs.append(instances_to_mesh("webs", web, wm, keep_base=False))
    washer = box("washer", (0.044, 0.044, 0.005), loc=(0, 0, 0.0015))
    nut = bolt("nut", 0.0125, 0.012, loc=(0, 0, 0.004))
    hd = join([washer, nut], "hd")
    objs.append(instances_to_mesh("hd_bolts", hd, bm, keep_base=False))

    # cast date / foundry mark standing proud on the crown of the head
    if str(p.mark).strip():
        body_txt = str(p.mark).strip()[:10]
        size = min(0.085, rh * 1.5 / max(2.5, len(body_txt) * 0.8))
        t = text("mark", body_txt, size=size, depth=0.006, rot=(0, 0, 0), font="serif")
        apply_transform(t)
        lo, hi = world_bbox([t])
        cxm, cym, zlo = (lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z
        ztop = zp + 0.88 * h

        def crown(co):
            x, y = co.x - cxm, co.y - cym
            c = clamp(hypot(x, y) / rh, 0.0, 0.98) ** (1 / 0.85)
            return Vector((x, y, ztop + 0.105 * h * sqrt(1 - c * c) - 0.0025 + (co.z - zlo)))

        transform_verts(t, crown)
        place(t, rot=(0, 0, rng.u(0, 360)))
        objs.append(t)
    for o in objs:
        assign(o, iron)
    zrim = zp + 0.86 * h
    # paint worn bright where lines bear (rim of the head, waist, top of the flange), a little rust and dirt at the foot
    paint(body, grime=mask_mul(height_ramp(zp, zp + 0.12), 0.3),
          special=lambda co, n: np.maximum(np.clip(1 - (co[:, 2] - zp) / 0.12, 0, 1) * 0.6,
                                           np.clip(1 - np.abs(co[:, 2] - (zp + 0.74 * h)) / (0.1 * h), 0, 1) * 0.8)
          * noise_mask(0.06, seed=p.seed + 2, lo=0.3, hi=0.7)(co, n) * (0.35 + 0.65 * age),
          wear=lambda co, n: np.maximum.reduce([np.clip(1 - np.abs(co[:, 2] - zrim) / 0.03, 0, 1) * 0.8,
                                                np.clip(1 - np.abs(co[:, 2] - (zp + 0.5 * h)) / (0.2 * h), 0, 1)
                                                * noise_mask(0.05, seed=p.seed + 6, lo=0.4, hi=0.7)(co, n) * 0.6,
                                                np.clip(1 - np.abs(co[:, 2] - (zp + 0.03)) / 0.012, 0, 1) * 0.45]))
    nw = noise_mask(0.03, seed=p.seed + 4, lo=0.35, hi=0.7)
    paint(objs[1], special=mask_mul(nw, 0.3 + 0.4 * age), grime=0.1,
          wear=lambda co, n: np.clip((n[:, 2] - 0.15) * 2.0, 0, 1) * 0.55)          # the sloping backs of the webs
    paint(objs[2], special=mask_mul(nw, 0.2 + 0.3 * age), grime=0.1, wear=lambda co, n: np.clip((co[:, 2] - (zp + 0.042)) / 0.006, 0, 1) * 0.6)

    # line: a few turns round the waist under the head, falling down the column, over the flange, round the foot
    if p.rope:
        rp = mat("rope", p.rope_tint, grime=1.1)
        pitch = 2 * rr * 1.04
        z_hi = zp + zb2 - 0.02 - rr
        turns = min(int(p.rope_turns), int((z_hi - zp - 0.21) / pitch))
        if turns >= 1:
            rr_ = rng.fork("rope")
            af = radians(a0 + 15 + 60 * int(rr_.u(0, 5.99)))             # it comes down between a web and a bolt
            pts, ae, z_lo = _hitch(rad, z_hi, turns, rr, rr_, af - 0.54)

            def G(a, dist, z):
                return (dist * cos(a), dist * sin(a), z)

            zbd = zp + 0.15
            drop = min(0.09, max(0.03, (z_lo - zbd - 0.03) * 0.8))
            for da, fz in ((0.16, 0.05), (0.31, 0.17), (0.43, 0.38), (0.51, 0.66), (0.54, 1.0)):
                z = z_lo - drop * fz
                pts.append(G(ae + da, rad(0, z) + rr * 1.1, z))
            rbead = rc + 0.012 + rr * 1.06
            zl = zp + rr
            pts += [G(af, rbead, zbd + 0.016), G(af, rbead + 0.001, zp + 0.105),
                    G(af, rc * 1.22 + 0.62 * rr, zp + 0.0465 + 0.8 * rr),
                    G(af + 0.015, rc * 1.3 + 0.025, zp + 0.036 + rr), G(af + 0.05, rf - 0.014, zp + 0.033 + rr),
                    G(af + 0.11, rf + 0.75 * rr, zp + 0.022 + 0.8 * rr), G(af + 0.2, rf + 1.7 * rr, zl + 0.003),
                    G(af + 0.32, rf + 2.2 * rr, zl)]
            arc = rr_.u(1.5, 2.3)
            rmax = half - rr - 0.016
            for i in range(1, 11):
                t = i / 10
                pts.append(G(af + 0.32 + arc * t, min(rmax, rf + rr * 2.3 + 0.008 * sin(t * 6 + ae) * min(1.0, 3 * t) + 0.02 * t * t), zl))
            ro, dense = _rope("line", pts, rr, rp, mat("rope", WHIP), max_rings=_rings_left(hi=215))
            ro += _fray("fray_a", dense[0], dense[0] - dense[3], rr, rp, rr_)
            ro += _fray("fray_b", dense[-1], dense[-1] - dense[-4], rr, rp, rr_, floor=zp)
            dirt = _rope_dirt(zp, z_lo + 1.6 * pitch, p.seed + 11)
            for o in ro:
                paint(o, grime=dirt)
    socket("line", "attach", (0, 0, zp + 0.5 * h), role="mooring")


# -------------------------------------------------------------------------- ring on a staple in a stone
def _ring(p, rng):
    wall = p.mount == "wall"
    age = p.age
    iron = mat("iron_wrought")
    R = p.ring_size / 2
    minor = clamp(R * 0.115, 0.009, 0.016)
    R -= minor                                         # ring_size is the outer diameter
    rrh = clamp(p.rope_dia * 0.38, 0.007, 0.011)       # the hank is a lighter line
    st = mat("stone_rough", "#8f8b84")
    rs = rng.fork("stone")
    if wall:
        zs = 0.0
        w, d, h = 2 * R + 0.24, 0.05, 2 * R + 0.22
        ztop = zs + 0.1
        stone = stone_block("stone", (w, d, h), loc=(0, -d / 2 - 0.008, ztop - h / 2), chamfer=0.014, rough=0.004, rng=rs, detail=0.035)
        yf = -d - 0.008
        zbot = ztop - h
    else:
        h = clamp(p.height * 0.7, 0.55, 0.95)
        w, d = 0.36, 0.32
        stone = stone_block("stone", (w, d, h), loc=(0, 0, h / 2), chamfer=0.022, rough=0.007, rng=rs, detail=0.05)
        apply_transform(stone)
        lx, ly = rs.u(-0.03, 0.03), rs.u(-0.02, 0.02)

        def k(z):
            return 1.0 - 0.17 * clamp(z / h) ** 1.6

        def shape(co):
            s = k(co.z)
            x, y = co.x * s, co.y * s
            u = clamp((co.z - 0.84 * h) / (0.16 * h))
            s2 = 1.0 - 0.62 * u ** 1.5                       # weathered four-way top
            x, y = x * s2, y * s2
            return Vector((x + lx * co.z / h, y + ly * co.z / h, max(0.0, co.z - 0.012 * u * u)))

        transform_verts(stone, shape)
        # base slab: finely divided, so the damp and moss gathering round the foot of the stone is a soft, broken fringe
        pad = stone_block("pad", (w + 0.2, d + 0.2, 0.055), loc=(0, 0, 0.0275), rot=(0, 0, rs.u(-4, 4)), chamfer=0.012, rough=0.003,
                          rng=rng.fork("pad"), detail=0.034)
        assign(pad, st)
        nf1 = noise_mask(0.03, seed=p.seed + 5, lo=0.2, hi=0.8)
        nf2 = noise_mask(0.09, seed=p.seed + 6, lo=0.25, hi=0.75)

        def foot(co, n):
            dx, dy = np.abs(co[:, 0]) - w / 2, np.abs(co[:, 1]) - d / 2
            dist = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
            return np.clip(1.0 - dist / (0.03 + 0.04 * nf2(co, n)), 0, 1) * (0.35 + 0.65 * nf1(co, n))

        paint(pad, grime=lambda co, n: 0.1 + 0.3 * foot(co, n), special=lambda co, n: foot(co, n) * (0.3 + 0.35 * age))
        zs = h * 0.74
        yf = -d / 2 * k(zs) + ly * zs / h
        zbot = 0.0
    assign(stone, st)
    smooth(stone, 50)
    if not wall:                                       # the ring (and its hank) must clear the quay
        R = max(0.045, min(R, (zs - 0.05 - (0.24 if p.rope else 0.03)) / 2))
        minor = clamp(R * 0.13, 0.008, minor)

    # staple: round bar bent to a U, both legs leaded into the stone
    g, bar, ext = 0.027 + minor * 0.6, 0.009, 0.036
    sp = [(0, yf + 0.014, zs + g), (0, yf - ext * 0.5, zs + g), (0, yf - ext, zs + g)]
    for i in range(1, 8):
        a = radians(90 - 180 * i / 8)
        sp.append((0, yf - ext - g * cos(a), zs + g * sin(a)))
    sp += [(0, yf - ext, zs - g), (0, yf - ext * 0.5, zs - g), (0, yf + 0.014, zs - g)]
    staple = sweep("staple", sp, bar, sides=8, smooth_path=False, cap=False)
    leads = [sphere("lead", 0.5, loc=(0, yf - 0.001, zs + s * g), segments=10, rings=6,
                    scale=(rs.u(0.042, 0.052), 0.012, rs.u(0.04, 0.05))) for s in (-1, 1)]
    for o in leads:
        assign(o, mat("lead"))

    y_top = yf - ext - g + bar + minor + 0.004
    z_top = zs - g + bar + minor
    y_bot = yf - minor - ((3.2 * rrh + 0.004) if p.rope else 0.0015)
    t = asin(clamp((y_bot - y_top) / (2 * R), -0.6, 0.6))
    ctr = (0, y_top + R * sin(t), z_top - R * cos(t))
    ring = torus("ring", R, minor, loc=ctr, rot=(degrees(t) - 90, 0, 0), major_seg=36, minor_seg=10)
    # forge-weld swelling on the ring
    aw = radians(rs.u(200, 250))
    weld = sphere("weld", 0.5, loc=(ctr[0] + R * cos(aw), ctr[1] - R * sin(aw) * sin(t), ctr[2] + R * sin(aw) * cos(t)), segments=10, rings=6,
                  scale=(minor * 3.4, minor * 2.5, minor * 2.5))
    place(weld, rot=(0, degrees(aw) + 90, 0))
    for o in (staple, ring, weld):
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.04, seed=p.seed + 2, lo=0.35, hi=0.75), 0.35 + 0.4 * age),
              wear=mask_mul(_zband(z_top, 0.035), 0.9))
    zb = z_top - 2 * R * cos(t)

    # stone: rust weeping from the staple, the arc the ring has rubbed, moss on the weather faces
    def rust(co, n):
        dz = zs + g - co[:, 2]
        return (np.exp(-(co[:, 0] / (0.03 + 0.06 * np.clip(dz, 0, 1))) ** 2) * np.clip(dz * 40, 0, 1) * np.clip(1 - dz / (0.3 + 0.2 * age), 0, 1)
                * (n[:, 1] < -0.4) * (0.55 + 0.4 * age))

    def rub(co, n):
        dist = np.hypot(co[:, 0], co[:, 2] - ctr[2])
        return np.clip(1 - np.abs(dist - R) / 0.02, 0, 1) * (co[:, 2] < ctr[2] + R * 0.2) * (n[:, 1] < -0.4) * 0.4

    moss = mask_mul(facing((0, 0, 1), 1.5), noise_mask(0.16, seed=p.seed, lo=0.25, hi=0.75), 0.4 + 0.6 * age)
    if not wall:
        low = mask_mul(height_ramp(0.0, 0.3), noise_mask(0.09, seed=p.seed + 1, lo=0.3, hi=0.7), 0.4 + 0.6 * age)
        m0 = moss
        moss = lambda co, n: np.maximum(m0(co, n), low(co, n))
    paint(stone, grime=lambda co, n: np.maximum(rust(co, n), np.clip(1 - (co[:, 2] - zbot) / 0.22, 0, 1) * (0.0 if wall else 0.4)),
          wear=rub, special=moss)

    # hank of light line hung through the ring: loops over the wire, frapped under it, bights hanging
    if p.rope:
        rp = mat("rope", p.rope_tint, grime=1.2)
        rh = rng.fork("hank")
        loops = int(clamp(p.rope_turns, 2, 4))
        yb = y_bot
        zn = zb - minor - rrh * 5.0
        max_len = max(0.1, zn - zbot - 0.04) if not wall else 0.42
        o_ = minor + rrh
        ropes = []
        for kk in range(loops):
            off = kk - (loops - 1) / 2
            x0 = off * rrh * 2.0
            Lh = min(max_len, rh.u(0.26, 0.38))
            wd = rh.u(0.036, 0.06)
            yk = yb + off * rrh * 1.5
            sx = rh.u(-0.012, 0.012)
            pts = [(x0, yb - o_, zb), (x0, yb - o_ * 0.72, zb + o_ * 0.72), (x0, yb, zb + o_), (x0, yb + o_ * 0.72, zb + o_ * 0.72), (x0, yb + o_, zb),
                   (x0 * 0.6 - rrh * 0.4, yb + rrh * 1.0, zn + rrh * 2.5), (x0 * 0.5 - rrh * 0.6, yb + rrh * 0.9, zn - rrh * 3.0),
                   (x0 - wd * 0.7, yk, zn - Lh * 0.22), (x0 - wd + sx, yk, zn - Lh * 0.55), (x0 - wd * 0.72 + sx, yk, zn - Lh * 0.88),
                   (x0 + sx, yk, zn - Lh), (x0 + wd * 0.72 + sx, yk, zn - Lh * 0.88), (x0 + wd + sx, yk, zn - Lh * 0.55),
                   (x0 + wd * 0.7, yk, zn - Lh * 0.22),
                   (x0 * 0.5 + rrh * 0.6, yb - rrh * 0.9, zn - rrh * 3.0), (x0 * 0.6 + rrh * 0.4, yb - rrh * 1.0, zn + rrh * 2.5)]
            ropes += _line(f"hank{kk}", pts, rrh, rp, closed=True, sides=6, step=0.017)
        # frapping turns round the neck, finished with a whipped, fraying end hanging
        ex, ey = (0.5 * (loops - 1) + 1.5) * rrh + rrh, 2.9 * rrh
        fp = []
        ft = 3
        for i in range(ft * 14 + 1):
            a = TAU * i / 14
            fp.append((ex * cos(a + 0.5), yb + ey * sin(a + 0.5), zn + rrh * (ft - 0.5) - 2.05 * rrh * i / 14))
        fp += [(ex * 0.9, yb - ey * 0.7, zn - rrh * (ft + 1.5)), (ex * 0.8 + 0.01, yb - ey * 0.8, zn - rrh * (ft + 1.5) - 0.06)]
        ropes += _line("frap", fp, rrh, rp, whip=mat("rope", WHIP), sides=6, step=0.017)
        ropes += _fray("fray", fp[-1], np.subtract(fp[-1], fp[-2]), rrh, rp, rh, strands=3, fibres=3)
        nh = noise_mask(0.1, seed=p.seed + 12, lo=0.25, hi=0.8)
        zlo_h = zn - max_len
        for o in ropes:                                     # the bights that hang lowest are the dampest
            paint(o, grime=lambda co, n: np.clip(0.45 * nh(co, n) + np.clip(1.0 - (co[:, 2] - zlo_h) / 0.2, 0, 1) * 0.45, 0, 1))
    socket("line", "attach", (0, y_bot - minor - 0.01, zb), role="mooring")


def build(p, rng):
    if p.kind == "bollard":
        _bollard(p, rng)
    elif p.kind == "ring":
        _ring(p, rng)
    else:
        _pile(p, rng)
