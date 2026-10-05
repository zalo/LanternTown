"""Open oak staircase running up along a wall to a first-floor landing. A dressed stone bottom step carries
two deep closed stringers with housed plank treads (open risers), tied by iron rods; the outer stringer is
tenoned into a newel at the foot and into the landing corner post at the head. The landing is a boarded
deck on joists between a wall ledger and an outer beam framed into full-height stop-chamfered posts on
stone pads, with pegged arch braces. Balustrade (turned / flat-sawn / square balusters) with a moulded
handrail, wall handrail on forged brackets, optional shingled lean-to roof and hanging candle lanterns.
Built climbing towards +X and mirrored for direction = left.
"""
import bpy
import numpy as np
from math import sin, cos, tan, atan, atan2, radians, degrees, hypot, pi, ceil
from mathutils import Matrix
from tk import *

META = {"name": "Timber external staircase", "class": "structure", "mount": "wall_ground",
        "focus": [0.9, -0.9, 3.0]}
PARAMS = {
    "rise": {"default": 2.9, "min": 2.2, "max": 3.8, "desc": "floor height: ground to landing floor (m)"},
    "width": {"default": 1.05, "min": 0.8, "max": 1.6, "desc": "overall width, wall to outer stringer face (m)"},
    "direction": {"default": "right", "choices": ["right", "left"], "desc": "side the stair climbs towards, seen from the front"},
    "going": {"default": 0.26, "min": 0.22, "max": 0.30, "desc": "tread going (m); the riser is ~0.18"},
    "landing": {"default": 1.5, "min": 1.0, "max": 3.0, "desc": "landing length along the wall (m)"},
    "balusters": {"default": "turned", "choices": ["turned", "flat", "square"], "desc": "baluster style (also sets the newel caps)"},
    "roof": {"default": True, "desc": "shingled lean-to roof over the landing"},
    "roof_pitch": {"default": 30.0, "min": 20.0, "max": 45.0, "desc": "lean-to roof pitch (deg)"},
    "wood": {"default": "#8a6848", "desc": "wood tone (stain of the oak frame)"},
    "paint": {"default": "", "desc": "paint colour of balusters and handrails ('' = bare oak)"},
    "lanterns": {"default": 2, "choices": [0, 1, 2], "desc": "0 none, 1 on a bracket at the stair head, 2 also under the landing"},
    "intensity": {"default": 12.0, "min": 6.0, "max": 50.0, "desc": "light of the stair-head lantern (cd); the one under the landing is dimmer"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: warp, foot wear, moss"},
}
VARIANTS = {
    "open_flat_left": {"direction": "left", "balusters": "flat", "roof": False, "rise": 3.1, "landing": 2.0,
                       "wood": "#6b4a30", "paint": "#3d5a4a", "seed": 4},
    "narrow_square_low": {"rise": 2.3, "width": 0.85, "going": 0.24, "landing": 1.1, "balusters": "square", "roof": False,
                          "wood": "#4a3626", "lanterns": 1, "age": 0.9, "seed": 11},
    "grand_gallery": {"rise": 3.5, "width": 1.45, "going": 0.29, "landing": 2.9, "balusters": "turned", "roof": True,
                      "roof_pitch": 24.0, "wood": "#9a7850", "paint": "#6e2a22", "age": 0.25, "seed": 23},
    "pale_left_roofed": {"direction": "left", "rise": 2.7, "width": 1.2, "landing": 1.6, "balusters": "flat", "roof": True,
                         "roof_pitch": 40.0, "wood": "#b89870", "lanterns": 1, "age": 0.7, "seed": 31},
}

HR = [(-0.03, -0.0275), (0.03, -0.0275), (0.034, -0.02), (0.034, -0.008), (0.04, -0.002), (0.04, 0.012), (0.03, 0.024),
      (0.012, 0.0275), (-0.012, 0.0275), (-0.03, 0.024), (-0.04, 0.012), (-0.04, -0.002), (-0.034, -0.008), (-0.034, -0.02)]


def hexmix(a, b, t):
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(pa[i] + (pb[i] - pa[i]) * t)) for i in range(3))


def seg(p0, p1, step=0.35):
    k = max(1, int(ceil(hypot(hypot(p1[0] - p0[0], p1[1] - p0[1]), p1[2] - p0[2]) / step)))
    return [tuple(p0[j] + (p1[j] - p0[j]) * i / k for j in range(3)) for i in range(k + 1)]


def prism(name, pts, depth, loc, plane="XZ", bevel=0.004):
    """Extruded 2D outline standing up. XZ: pts (x, z), extruded towards -Y. YZ: pts (y, z), extruded towards +X."""
    return extrude_profile(name, pts, depth, loc=loc, rot=(90, 0, 0) if plane == "XZ" else (90, 0, 90), bevel=bevel)


def post(name, cx, cy, z0, z1, s, chamf=(), c_big=None):
    """Square post with eased arrises and stop-chamfered lengths (list of (za, zb))."""
    c_small = 0.005
    c_big = c_big or s * 0.16
    keys = [(z0, c_small)]
    for za, zb in sorted(chamf):
        if zb - za < 0.22 or za < keys[-1][0] + 0.01 or zb > z1 - 0.01:
            continue
        keys += [(za, c_small), (za + 0.05, c_big), (zb - 0.05, c_big), (zb, c_small)]
    keys.append((z1, c_small))
    rings = [keys[0]]
    for (za, ca), (zb, cb) in zip(keys[:-1], keys[1:]):
        k = max(1, int(ceil((zb - za) / 0.35)))
        for i in range(1, k + 1):
            t = i / k
            rings.append((za + (zb - za) * t, ca + (cb - ca) * t))
    hs = s / 2
    verts, faces = [], []
    for z, c in rings:
        verts += [(cx + hs - c, cy - hs, z), (cx + hs, cy - hs + c, z), (cx + hs, cy + hs - c, z), (cx + hs - c, cy + hs, z),
                  (cx - hs + c, cy + hs, z), (cx - hs, cy + hs - c, z), (cx - hs, cy - hs + c, z), (cx - hs + c, cy - hs, z)]
    for k in range(len(rings) - 1):
        for i in range(8):
            j = (i + 1) % 8
            faces.append((k * 8 + i, k * 8 + j, (k + 1) * 8 + j, (k + 1) * 8 + i))
    faces.append(tuple(reversed(range(8))))
    faces.append(tuple(range((len(rings) - 1) * 8, len(rings) * 8)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    uv_grain(o, "Z")
    return o


def brace_pts(a, wd, sgn=1, c=1.7, n=7):
    """Arch brace outline; the post/beam corner is (0, 0), post face u = 0 going down, beam soffit z = 0."""
    def arc(a_):
        cc = c * a_
        r = hypot(cc, a_ - cc)
        t0, t1 = atan2(cc - a_, -cc), atan2(cc, a_ - cc)
        return [(cc + r * cos(t0 + (t1 - t0) * i / n), -cc + r * sin(t0 + (t1 - t0) * i / n)) for i in range(n + 1)]
    pts = arc(a) + list(reversed(arc(a - wd * 1.4142)))
    if sgn < 0:
        pts = [(-u, z) for u, z in reversed(pts)]
    return pts


def build(p, rng):
    d = 1.0 if p.direction == "right" else -1.0
    age = float(p.age)
    R, W, g, L = float(p.rise), float(p.width), float(p.going), float(p.landing)
    n = max(8, int(round(R / 0.182)))          # risers
    h = R / n
    run = (n - 1) * g
    x0 = -(run + L) / 2                        # face of the bottom (stone) step
    xl = x0 + run                              # landing edge = head of the flight
    x1 = x0 + run + L
    s = h / g
    tt, nose, ts = 0.045, 0.03, 0.07           # tread thickness, nosing, stringer thickness
    TOP = 0.05
    D = TOP + h + tt + 0.07                    # vertical depth of the stringer
    yW = -0.03                                 # wall-side face of the timberwork
    yF = yW - W                                # outer face of stringer / beam
    PS = 0.16                                  # landing post section
    yP = yF + 0.035                            # post / balustrade centre line
    ymid = (yW - ts + yF + ts) / 2
    Wc = W - 2 * ts
    painted = isinstance(p.paint, str) and p.paint.startswith("#") and len(p.paint) == 7
    print(f"DBG steps={n} riser={h:.3f} run={run:.2f} total={run + L:.2f}")

    oak = mat("wood_oak", p.wood)
    board = mat("wood_weathered", hexmix(p.wood, "#d6d0c6", 0.52))
    endg = mat("wood_endgrain", hexmix(p.wood, "#c8b090", 0.35))
    stone = mat("stone_rough", "#a8a091")
    iron = mat("iron_wrought")
    trim = mat("wood_painted", p.paint) if painted else oak

    def zn(x):                                 # nosing line
        return h + (x - x0) * s

    def zt(x):                                 # stringer top edge
        return zn(x) + TOP

    def zr(x):                                 # flight handrail centre line
        return zn(x) + 0.90

    frame, trims, irons, stones, pegs_m, nails_m = [], [], [], [], [], []
    low_grime = height_ramp(0.0, 0.7)
    moss_low = mask_mul(height_ramp(0.0, 0.45), noise_mask(0.18, seed=p.seed + 3, lo=0.3, hi=0.7), 0.3 + 0.7 * age)

    # ------------------------------------------------------------------ stone bottom step (tread 1) and pads
    rs = rng.fork("stone")
    sx0, sx1 = x0 - 0.02, x0 + g + 0.34
    sy0, sy1 = yF - 0.16, -0.02
    split = lerp(sy0, sy1, rs.u(0.38, 0.62))
    for k, (ya, yb) in enumerate(((sy0, split - 0.004), (split + 0.004, sy1))):
        b = stone_block(f"step_stone{k}", (sx1 - sx0, yb - ya, h), loc=((sx0 + sx1) / 2, (ya + yb) / 2, h / 2), rng=rs,
                        chamfer=0.016, rough=0.005)
        stones.append(b)

    def pad(name, cx, cy, sz=0.3, hh=0.14):
        b = stone_block(name, (rs.vary(sz, 0.06), rs.vary(sz, 0.06), hh), loc=(cx, cy, hh / 2), rot=(0, 0, rs.jitter(0, 3)),
                        rng=rs, chamfer=0.022, rough=0.005)
        stones.append(b)
        return hh - 0.006

    # ------------------------------------------------------------------ stringers
    xf = x0 + g - 0.11                          # plumb cut at the foot
    xb = x0 + (D - TOP) / s                     # where the soffit meets the stone step
    spts = [(xf, h), (xb, h), (xl, zt(xl) - D), (xl, R), (xl - TOP / s, R), (xf, zt(xf))]
    ca, sa = cos(atan(s)), sin(atan(s))
    for nm, yy in (("stringer_wall", yW), ("stringer_out", yF + ts)):
        st = prism(nm, spts, ts, (0, yy, 0), bevel=0.006)
        densify(st, 0.3)
        if age > 0.05:
            displace_noise(st, strength=0.002 * age, scale=0.6, seed=p.seed + len(frame))
        uv_grain(st, axis=(ca, sa, 0))
        frame.append(st)

    # ------------------------------------------------------------------ treads (housed, open risers)
    rt = rng.fork("treads")
    treads = []
    for i in range(2, n):
        xa, xbk = x0 + (i - 1) * g - nose, x0 + i * g + 0.012
        t = plank(f"tread{i}", length=Wc + 0.03, width=xbk - xa, thick=tt, axis="Y", bevel=0.007,
                  loc=((xa + xbk) / 2 + rt.jitter(0, 0.003), ymid, i * h - tt / 2 - rt.u(0, 0.003 * age)),
                  rot=(rt.jitter(0, 0.5 * age), rt.jitter(0, 0.35 * age), rt.jitter(0, 0.3 * age)))
        densify(t, 0.2)
        treads.append(t)
        for dx in (-0.06, 0.07):
            nails_m.append(trs(((xa + xbk) / 2 + dx + rt.jitter(0, 0.008), yF, i * h - tt / 2 + rt.jitter(0, 0.004)), (90, 0, 0)))
    treads = join(treads, "treads")
    assign(treads, board)

    def foot_wear(lo_x=None):
        def fn(co, nr):
            yc = (co[:, 1] - ymid) / (Wc * 0.5)
            return np.clip(nr[:, 2], 0, 1) * np.exp(-(yc * 1.5) ** 2) * (0.35 + 0.6 * age)
        return fn

    def edge_dirt(co, nr):
        yc = np.abs((co[:, 1] - ymid) / (Wc * 0.5))
        return np.clip(nr[:, 2], 0, 1) * np.clip((yc - 0.55) * 2.2, 0, 1) * (0.4 + 0.4 * age)

    paint(treads, wear=mask_mul(foot_wear(), noise_mask(0.22, seed=p.seed + 5, lo=0.15, hi=0.7)), grime=edge_dirt,
          special=mask_mul(edge_dirt, noise_mask(0.12, seed=p.seed + 6, lo=0.45, hi=0.8), age))

    # iron tie rods with plate washers and nuts
    if n >= 9:
        for f in (0.3, 0.68):
            xr = lerp(xb, xl, f)
            zr_ = zt(xr) - D + 0.075
            irons.append(cylinder("tie_rod", 0.007, W + 0.02, loc=(xr, (yW + yF) / 2, zr_), rot=(90, 0, 0), verts=8))
            irons.append(box("tie_plate", (0.06, 0.006, 0.06), loc=(xr, yF - 0.003, zr_), rot=(0, rng.jitter(0, 6), 0), bevel=0.0015))
            irons.append(bolt("tie_nut", 0.012, 0.011, loc=(xr, yF - 0.006, zr_), rot=(90, 0, 0), kind="square"))

    # ------------------------------------------------------------------ bottom newel
    cap_style = p.balusters

    def cap(cx, cy, z, sq):
        ob = [box("cap_abacus", (sq + 0.034, sq + 0.034, 0.028), loc=(cx, cy, z + 0.014), bevel=0.007)]
        if cap_style == "turned":
            ob.append(lathe("cap_ball", [(sq * 0.34, 0.028), (sq * 0.36, 0.036), (sq * 0.24, 0.046), (sq * 0.2, 0.06), (sq * 0.3, 0.074),
                                         (sq * 0.43, 0.098), (sq * 0.47, 0.132), (sq * 0.42, 0.166), (sq * 0.27, 0.19), (sq * 0.1, 0.202),
                                         (0.0, 0.204)], segments=16, loc=(cx, cy, z)))
        elif cap_style == "flat":
            r = (sq / 2 + 0.004) * 1.4142
            ob.append(lathe("cap_pyr", [(r, 0.028), (r, 0.04), (r * 0.82, 0.05), (r * 0.16, 0.115), (0.0, 0.118)], segments=4,
                            loc=(cx, cy, z), rot=(0, 0, 45), smooth_angle=20))
        else:
            ob.append(box("cap_block", (sq - 0.02, sq - 0.02, 0.03), loc=(cx, cy, z + 0.04), bevel=0.012))
        for o in ob:
            uv_grain(o, "X")
        trims.extend(ob)

    xN = xf - 0.06
    zN = zr(xN) + 0.17
    frame.append(post("newel_foot", xN, yP, h - 0.004, zN, 0.12, chamf=[(h + 0.3, zr(xN) - 0.14)]))
    cap(xN, yP, zN, 0.12)
    for dz in (0.05, 0.11):
        pegs_m.append(trs((xN + 0.01, yP - 0.06, h + dz + 0.02), (90, 0, 0)))

    # ------------------------------------------------------------------ landing frame
    zdeck = R - 0.04                            # top of joists
    zbeam = zdeck - 0.13                        # top of beam / ledger
    zsoff = zbeam - 0.20                        # beam soffit
    pxs = [xl + PS / 2, x1 - PS / 2]
    if L > 2.3:
        pxs.insert(1, (xl + x1) / 2)
    z_eaves = R + 2.05
    lant_post = p.lanterns >= 1
    for k, px in enumerate(pxs):
        zb_ = pad(f"pad{k}", px, yP, 0.32)
        if p.roof:
            ztop = z_eaves
        else:
            ztop = R + (1.86 if (k == 0 and lant_post) else 1.2)
        a_x = min(0.6, (pxs[1] - pxs[0] - PS) * 0.42)
        ch = [(0.5, zsoff - a_x - 0.1), (R + 0.2, R + 0.86)]
        if ztop > R + 1.6:
            ch.append((R + 1.14, ztop - (0.45 if p.roof else 0.5)))
        frame.append(post(f"post{k}", px, yP, zb_, ztop, PS, chamf=ch))
        if not p.roof:
            cap(px, yP, ztop, PS)
        if k == 0:                              # stringer tenon pegs
            for dz in (-0.07, -0.17):
                pegs_m.append(trs((px - 0.03, yP - PS / 2, R + dz), (90, 0, 0)))
        for dz in (-0.045, 0.045):              # beam tenon pegs
            pegs_m.append(trs((px + rng.jitter(0, 0.008), yP - PS / 2, zbeam - 0.10 + dz), (90, 0, 0)))

    rj = rng.fork("joists")
    ledger = plank("ledger", length=L, width=0.08, thick=0.18, loc=((xl + x1) / 2, yW - 0.04, zbeam - 0.09), bevel=0.006)
    densify(ledger, 0.35)
    frame.append(ledger)
    hexb = bolt("ledger_bolt", 0.013, 0.01, kind="hex")
    nb = max(2, int(round(L / 0.55)))
    irons.append(instances_to_mesh("ledger_bolts", hexb, [trs((xl + (i + 0.5) * L / nb, yW - 0.08, zbeam - 0.09 + rj.jitter(0, 0.01)),
                                                             (90, rj.u(0, 60), 0)) for i in range(nb)], keep_base=False))
    for k in range(len(pxs) - 1):
        xa, xbm = pxs[k] + PS / 2, pxs[k + 1] - PS / 2
        bm_ = plank(f"beam{k}", length=xbm - xa + 0.02, width=0.13, thick=0.20, loc=((xa + xbm) / 2, yF + 0.065, zbeam - 0.10),
                    bevel=0.008)
        densify(bm_, 0.3)
        if age > 0.05:
            displace_noise(bm_, strength=0.002 * age, scale=0.5, seed=p.seed + 40 + k)
        frame.append(bm_)
        a = min(0.6, (xbm - xa) * 0.42)
        for sg, xc in ((1, xa), (-1, xbm)):
            br = prism("brace_x", brace_pts(a, 0.11, sg), 0.08, (xc, yF + 0.105, zsoff), bevel=0.005)
            uv_grain(br, axis=(sg * 0.7071, 0.7071, 0))
            frame.append(br)
            pegs_m.append(trs((xc - sg * PS / 2, yP - PS / 2, zsoff - a + 0.085), (90, 0, 0)))
            pegs_m.append(trs((xc + sg * (a - 0.1), yF, zsoff + 0.05), (90, 0, 0)))
    # trimmers at the posts, common joists between
    ypost_in = yP + PS / 2
    trim_x = [(xl + 0.05, 0.10, 0.22), (x1 - 0.07, 0.09, 0.20)]
    if len(pxs) == 3:
        trim_x.append((pxs[1], 0.08, 0.16))
    a_y = clamp((ypost_in and (yW - ypost_in)) * 0.62, 0.3, 0.6)
    for k, (tx, tw, td) in enumerate(trim_x):
        j = plank(f"trimmer{k}", length=yW - ypost_in + 0.02, width=tw, thick=td, axis="Y",
                  loc=(tx, (yW + ypost_in) / 2, zdeck - td / 2), bevel=0.006)
        densify(j, 0.35)
        frame.append(j)
        br = prism("brace_y", brace_pts(a_y, 0.09, 1), 0.07, (tx - 0.035, ypost_in, zdeck - td), plane="YZ", bevel=0.005)
        uv_grain(br, axis=(0.7071, 0.7071, 0))
        frame.append(br)
        xface = x1 if k == 1 else (xl + PS if k == 0 else tx + PS / 2)
        pegs_m.append(trs((xface, ypost_in - 0.04, zdeck - td - a_y + 0.08), (0, 90, 0)))
        pegs_m.append(trs((xface, ypost_in - 0.04, zdeck - td / 2), (0, 90, 0)))
    jx0, jx1 = xl + 0.10, x1 - 0.115
    nj = max(1, int(ceil((jx1 - jx0) / 0.42)) - 1)
    for i in range(nj):
        jx = jx0 + (i + 1) * (jx1 - jx0) / (nj + 1)
        j = plank(f"joist{i}", length=yW - (yF - 0.014), width=0.065, thick=0.13, axis="Y", bevel=0.005,
                  loc=(jx + rj.jitter(0, 0.01), (yW + yF - 0.014) / 2, zdeck - 0.065), end_mat=endg)
        densify(j, 0.35)
        frame.append(j)

    # deck boards along the wall
    rd = rng.fork("deck")
    tot = (yW - 0.005) - (yF + 0.012)
    nbd = max(3, int(round(tot / 0.155)))
    bw = tot / nbd
    deck = []
    for i in range(nbd):
        b = plank(f"deck{i}", length=x1 - 0.006 - (xl - nose) + rd.jitter(0, 0.006), width=bw - rd.u(0.004, 0.008), thick=0.04,
                  loc=((xl - nose + x1 - 0.006) / 2, yF + 0.012 + (i + 0.5) * bw, R - 0.02 - rd.u(0, 0.002 * age)),
                  rot=(rd.jitter(0, 0.4 * age), 0, 0), bevel=0.005, end_mat=endg)
        densify(b, 0.3)
        deck.append(b)
    deck = join(deck, "deck")
    assign(deck, board)
    paint(deck, wear=mask_mul(foot_wear(), noise_mask(0.3, seed=p.seed + 8, lo=0.2, hi=0.75)), grime=edge_dirt)

    # wall half-newel closing the end balustrade
    xE = x1 - 0.07
    frame.append(post("newel_wall", xE, yW - 0.045, R - 0.002, R + 1.14, 0.09))
    trims.append(box("cap_wall", (0.115, 0.115, 0.03), loc=(xE, yW - 0.05, R + 1.155), bevel=0.01))

    # ------------------------------------------------------------------ mid prop under the flight
    if run > 2.8:
        xm = lerp(xb, xl, 0.5)
        zbt = zt(xm) - D + 0.045
        bearer = plank("bearer", length=(-0.004) - (yF - 0.07), width=0.10, thick=0.12, axis="Y",
                       loc=(xm, (-0.004 + yF - 0.07) / 2, zbt - 0.06), bevel=0.006, end_mat=endg)
        frame.append(bearer)
        zb_ = pad("pad_prop", xm, yP, 0.27, 0.12)
        frame.append(post("prop", xm, yP, zb_, zbt - 0.118, 0.11, chamf=[(0.4, zbt - 0.3)]))
        br = prism("brace_prop", brace_pts(0.3, 0.07, 1), 0.06, (xm - 0.03, yP + 0.055, zbt - 0.12), plane="YZ", bevel=0.004)
        uv_grain(br, axis=(0.7071, 0.7071, 0))
        frame.append(br)
        pegs_m.append(trs((xm, yP - 0.055, zbt - 0.17), (90, 0, 0)))

    # ------------------------------------------------------------------ balustrade
    style = p.balusters
    Lb0 = 0.87
    many = (xl - x0 + L + W) / 0.118 > 66        # long galleries: lighter turnings keep the triangle budget
    if style == "turned":
        bl = lathe("bal_turn", [(0.0, 0.1), (0.021, 0.1), (0.016, 0.122), (0.026, 0.146), (0.016, 0.178), (0.022, 0.208),
                                (0.031, 0.29), (0.026, 0.39), (0.017, 0.5), (0.0135, 0.622), (0.021, 0.652),
                                (0.0145, 0.682), (0.022, 0.73), (0.021, 0.772), (0.0, 0.772)], segments=6 if many else 8)
        b0 = box("bal_b0", (0.046, 0.046, 0.108), loc=(0, 0, 0.054), bevel=0.003, segments=1)
        b1 = box("bal_b1", (0.046, 0.046, 0.104), loc=(0, 0, Lb0 - 0.052), bevel=0.003, segments=1)
        base = join([bl, b0, b1], "bal_base")
        pitch = 0.118
    elif style == "flat":
        half = [(0.054, 0.0), (0.054, 0.105), (0.046, 0.118), (0.03, 0.132), (0.027, 0.15), (0.034, 0.172), (0.047, 0.2), (0.053, 0.24),
                (0.054, 0.285), (0.05, 0.335), (0.04, 0.4), (0.03, 0.47), (0.023, 0.54), (0.02, 0.6), (0.022, 0.635), (0.032, 0.655),
                (0.037, 0.672), (0.032, 0.69), (0.022, 0.705), (0.021, 0.725), (0.03, 0.745), (0.046, 0.757), (0.054, 0.77), (0.054, Lb0)]
        outline = half + [(-u, z) for u, z in reversed(half)]
        base = extrude_profile("bal_base", outline, 0.024, loc=(0, 0.012, 0), rot=(90, 0, 0), bevel=0.003, segments=1)
        uv_grain(base, "Y")
        realize(base)
        apply_transform(base)
        pitch = 0.128
    else:
        base = box("bal_base", (0.036, 0.036, Lb0), loc=(0, 0, Lb0 / 2), rot=(0, 0, 45), bevel=0.003, segments=1)
        uv_grain(base, "Z")
        realize(base)
        apply_transform(base)
        pitch = 0.108
    rb = rng.fork("bal")
    bm = []
    # flight
    nbf = max(1, int(round((xl - xf) / pitch)) - 1)
    for i in range(nbf):
        x = xf + (i + 1) * (xl - xf) / (nbf + 1)
        zb0, zb1 = zt(x) - 0.025, zr(x) - 0.005
        bm.append(trs((x, yP, zb0), (rb.jitter(0, 0.5 * age), 0, rb.jitter(0, 1.5 * age)), (1, 1, (zb1 - zb0) / Lb0)))
    # landing bays and the end return
    zlb0, zlb1 = R + 0.115, R + 0.995
    runs = [((pxs[k] + PS / 2, yP), (pxs[k + 1] - PS / 2, yP)) for k in range(len(pxs) - 1)]
    runs.append(((xE, yP + PS / 2), (xE, yW - 0.09)))
    for (ax, ay), (bx, by) in runs:
        ln = hypot(bx - ax, by - ay)
        nbl = max(1, int(round(ln / pitch)) - 1)
        rz = 0 if abs(by - ay) < 1e-6 else 90
        for i in range(nbl):
            t = (i + 1) / (nbl + 1)
            bm.append(trs((lerp(ax, bx, t), lerp(ay, by, t), zlb0), (0, 0, rz + rb.jitter(0, 1.5 * age)), (1, 1, (zlb1 - zlb0) / Lb0)))
        horiz = "X" if rz == 0 else "Y"
        trims.append(plank("rail_bottom", length=ln + 0.02, width=0.06, thick=0.05, axis=horiz,
                           loc=((ax + bx) / 2, (ay + by) / 2, R + 0.10), bevel=0.006))
        e = ((bx - ax) / ln * 0.025, (by - ay) / ln * 0.025)
        hr = sweep("handrail_l", seg((ax - e[0], ay - e[1], R + 1.0), (bx + e[0], by + e[1], R + 1.0)), HR, smooth_path=False)
        trims.append(hr)
    bal = instances_to_mesh("balusters", base, bm, keep_base=False)
    assign(bal, trim)
    paint(bal, grime=mask_mul(noise_mask(0.3, seed=p.seed + 12, lo=0.3, hi=0.8), 0.25 + 0.3 * age),
          wear=mask_mul(noise_mask(0.1, seed=p.seed + 13, lo=0.5, hi=0.85), 0.3 + 0.5 * age) if painted else None)
    hr = sweep("handrail_f", seg((xf - 0.03, yP, zr(xf - 0.03)), (xl + 0.03, yP, zr(xl + 0.03))), HR, smooth_path=False)
    trims.append(hr)

    # wall handrail on forged brackets
    xa, xbw = x0 + g * 0.8, xl - 0.03
    yR = -0.085
    wr = sweep("handrail_w", seg((xa, yR, zr(xa)), (xbw, yR, zr(xbw))), 0.021, sides=12, smooth_path=False)
    wall_rail = [wr, sphere("rail_knob0", 0.027, loc=(xa, yR, zr(xa)), segments=12, rings=8),
                 sphere("rail_knob1", 0.027, loc=(xbw, yR, zr(xbw)), segments=12, rings=8)]
    nbk = max(2, int(round((xbw - xa) / 1.1)) + 1)
    for i in range(nbk):
        x = xa + 0.12 + i * (xbw - xa - 0.24) / (nbk - 1)
        z = zr(x)
        irons.append(cylinder("wr_plate", 0.034, 0.006, loc=(x, -0.003, z - 0.075), rot=(90, 0, 0), verts=12, bevel=0.0015))
        irons.append(sweep("wr_arm", [(x, -0.005, z - 0.075), (x, -0.05, z - 0.078), (x, -0.08, z - 0.06), (x, yR, z - 0.018)], 0.0075, sides=6, step=0.02))
        irons.append(box("wr_saddle", (0.07, 0.03, 0.005), loc=(x, yR, z - 0.021), rot=(0, -degrees(atan(s)), 0), bevel=0.0015))

    # ------------------------------------------------------------------ lean-to roof
    roof_objs = []
    if p.roof:
        th = radians(p.roof_pitch)
        ct, tn = cos(th), tan(th)
        rxa, rxb = xl - 0.24, x1 + 0.24
        zpt = z_eaves + 0.14
        plate = plank("eaves_plate", length=rxb - rxa + 0.1, width=0.12, thick=0.14, loc=((rxa + rxb) / 2, yP, z_eaves + 0.07),
                      bevel=0.008, end_mat=endg)
        densify(plate, 0.35)
        frame.append(plate)
        for k, px in enumerate(pxs):
            for sg in (1, -1):
                br = prism("brace_r", brace_pts(0.34, 0.075, sg), 0.06, (px + sg * PS / 2, yP + 0.03, z_eaves), bevel=0.004)
                uv_grain(br, axis=(sg * 0.7071, 0.7071, 0))
                frame.append(br)
                pegs_m.append(trs((px + sg * (PS / 2 + 0.26), yP - 0.06, z_eaves + 0.05), (90, 0, 0)))
        rdp = 0.10
        y_top, y_bot = -0.065, yF - 0.035 - 0.32

        def zc(y, q=0.0):                       # rafter centre plane (+ q perpendicular offset)
            return zpt + (rdp / 2 + q) / ct + (y - yP) * tn

        slope_len = (y_top - y_bot) / ct
        nr_ = max(2, int(ceil((rxb - rxa) / 0.5)) + 1)
        rr = rng.fork("roof")
        for i in range(nr_):
            x = rxa + i * (rxb - rxa) / (nr_ - 1)
            ym = (y_top + y_bot) / 2
            rf = plank(f"rafter{i}", length=slope_len, width=0.06, thick=rdp, axis="Y", loc=(x, ym, zc(ym)), rot=(degrees(th), 0, 0),
                       bevel=0.005, end_mat=endg)
            densify(rf, 0.35)
            frame.append(rf)
        for xv in (rxa - 0.055, rxb + 0.055):
            bg = plank("barge", length=slope_len + 0.03, width=0.024, thick=0.15, axis="Y", loc=(xv, (y_top + y_bot) / 2, zc((y_top + y_bot) / 2, -0.005)),
                       rot=(degrees(th), 0, 0), bevel=0.004, end_mat=endg)
            densify(bg, 0.35)
            frame.append(bg)
        wp_top = zc(-0.07, -rdp)
        frame.append(plank("wall_plate", length=rxb - rxa + 0.06, width=0.075, thick=0.12, loc=((rxa + rxb) / 2, yW - 0.04, wp_top - 0.06),
                           bevel=0.006, end_mat=endg))
        nbt = max(3, int(slope_len / 0.3) + 1)
        for i in range(nbt):
            y = y_bot + 0.03 + i * (y_top - y_bot - 0.08) / (nbt - 1)
            frame.append(plank(f"batten{i}", length=rxb - rxa + 0.12, width=0.045, thick=0.022, loc=((rxa + rxb) / 2, y, zc(y, 0.011)),
                               rot=(degrees(th), 0, 0), bevel=0.002))
        # split shingles, course by course from the eaves
        sl, ex, st_ = 0.36, 0.15, 0.012
        sh = box("shingle", (0.12, sl, st_))
        uv_grain(sh, "Y")
        realize(sh)
        tilt = degrees(th) - degrees(atan(st_ / ex))
        sm = []
        k = 0
        while True:
            u0 = -0.05 + k * ex
            if u0 > slope_len - 0.12:
                break
            ln = min(sl, slope_len - u0 + 0.01)
            uc = u0 + ln / 2
            y = y_bot + uc * ct
            x = rxa - 0.07 - (0.06 if k % 2 else 0.0) - rr.u(0, 0.03)
            while x < rxb + 0.07:
                w = rr.u(0.085, 0.15)
                if x + w > rxb + 0.09:
                    w = max(0.05, rxb + 0.09 - x)
                slip = rr.u(-0.012, 0.004) * (0.5 + age)
                sm.append(trs((x + w / 2, y + slip * ct, zc(y, 0.022 + st_ * 1.15) + slip * sin(th)),
                              (tilt + rr.jitter(0, 0.8), rr.jitter(0, 1.2 * age), rr.jitter(0, 1.0)), ((w - 0.004) / 0.12, ln / sl, rr.u(0.8, 1.3))))
                x += w
            k += 1
        shingles = instances_to_mesh("shingles", sh, sm, keep_base=False, seed=p.seed)
        assign(shingles, board)
        paint(shingles, grime=mask_mul(noise_mask(0.35, seed=p.seed + 20, lo=0.25, hi=0.8), 0.35 + 0.4 * age),
              special=mask_mul(facing((0, 0, 1)), noise_mask(0.28, seed=p.seed + 21, lo=0.4, hi=0.75), 0.25 + 0.75 * age))
        lead = mat("lead")
        yfl = y_top - 0.05
        fl = box("flashing", (rxb - rxa + 0.16, 0.2, 0.004), loc=((rxa + rxb) / 2, yfl, zc(yfl, 0.022 + st_ * 3.2)), rot=(degrees(th), 0, 0), bevel=0.001)
        up = box("flashing_up", (rxb - rxa + 0.16, 0.004, 0.16), loc=((rxa + rxb) / 2, -0.004, zc(-0.004, 0.022 + st_ * 3.2) + 0.06), bevel=0.001)
        for o in (fl, up):
            densify(o, 0.2)
            displace_noise(o, strength=0.003, scale=0.15, seed=p.seed + 22, axis_mask=(0, 0, 1))
            assign(o, lead)
            roof_objs.append(o)

    # ------------------------------------------------------------------ lanterns
    glass = mat("glass_lantern", glow=0.7)
    flame_m = mat("emissive", "#ffb45e", glow=6)

    def lantern(tag, hp, drop, cd, snear=0.3):
        x, y, z = hp
        ob, gl, fl_ = [], [], []
        ob.append(torus(f"{tag}_eye", 0.012, 0.003, loc=(x, y, z - 0.012), rot=(90, 0, 0), major_seg=12, minor_seg=6))
        if drop > 0.05:
            ob.append(chain(f"{tag}_chain", [(x, y, z - 0.022), (x, y, z - drop + 0.012)], link_len=0.04, wire=0.0032, sides=4))
        zt_ = z - drop
        w = 0.16
        ob.append(torus(f"{tag}_ring", 0.012, 0.003, loc=(x, y, zt_ + 0.006), rot=(90, 0, 90), major_seg=12, minor_seg=6))
        ob.append(lathe(f"{tag}_roof", [(0.0, -0.082), (w * 0.8, -0.082), (w * 0.82, -0.076), (w * 0.5, -0.045), (w * 0.3, -0.032),
                                        (w * 0.2, -0.03), (w * 0.2, -0.016), (w * 0.26, -0.012), (w * 0.1, -0.004), (0.0, 0.0)],
                        segments=4, loc=(x, y, zt_), rot=(0, 0, 45), smooth_angle=20))
        z1_ = zt_ - 0.082
        z0_ = z1_ - 0.22
        rt_, rb_ = w * 0.66, w * 0.46            # corner radii top / bottom (tapering body)
        gl.append(lod_faces(lathe(f"{tag}_glass", [(0.0, z0_), (rb_ - 0.003, z0_), (rt_ - 0.003, z1_), (0.0, z1_)], segments=4, loc=(x, y, 0),
                                  rot=(0, 0, 45), smooth_angle=20), drop=[(0, 0, 1), (0, 0, -1)], above=-1, cos=0.9))   # no caps: they lay on the roof / in the pan
        for k in range(4):
            a = radians(45 + 90 * k)
            ob.append(sweep(f"{tag}_bar{k}", [(x + rb_ * cos(a), y + rb_ * sin(a), z0_), (x + rt_ * cos(a), y + rt_ * sin(a), z1_)],
                            rounded_rect(0.009, 0.009, 0.002, 1), smooth_path=False))
        ob.append(lathe(f"{tag}_pan", [(0.0, z0_ - 0.045), (0.008, z0_ - 0.04), (0.012, z0_ - 0.03), (0.006, z0_ - 0.022), (rb_ * 0.6, z0_ - 0.014),
                                       (rb_ + 0.008, z0_ - 0.012), (rb_ + 0.01, z0_), (rb_ + 0.004, z0_ + 0.007), (0.0, z0_ + 0.007)],
                        segments=4, loc=(x, y, 0), rot=(0, 0, 45), smooth_angle=20))
        ob.append(lathe(f"{tag}_top", [(rt_ + 0.004, z1_ - 0.008), (rt_ + 0.009, z1_ - 0.003), (rt_ + 0.006, z1_ + 0.003)], segments=4,
                        loc=(x, y, 0), rot=(0, 0, 45), smooth_angle=20))
        ob.append(lathe(f"{tag}_font", [(0.0, z0_ + 0.007), (0.024, z0_ + 0.007), (0.026, z0_ + 0.028), (0.012, z0_ + 0.036), (0.008, z0_ + 0.052),
                                        (0.0, z0_ + 0.052)], segments=10, loc=(x, y, 0)))
        fl_.append(sphere(f"{tag}_flame", 0.013, loc=(x, y, z0_ + 0.078), segments=10, rings=6, scale=(1, 1, 1.9)))
        for o in ob:
            assign(o, iron)
            paint(o, special=mask_mul(noise_mask(0.04, seed=p.seed + 30, lo=0.45, hi=0.8), 0.5))
        for o in gl:
            assign(o, glass)
        for o in fl_:
            assign(o, flame_m)
        part(ob + gl + fl_, tag, pivot=(d * x, y, z - 0.012), motion="swing", amp=2.0, speed=0.35)
        light(tag, (d * x, y, z0_ + 0.09), intensity=cd, range=9.0, flicker=0.3, shadow_near=snear)

    if p.lanterns >= 1:
        # forged bracket on the front face of the head post; the lantern hangs clear outside the balustrade
        px = pxs[0]
        yq = yP - PS / 2
        za = R + 1.58
        Rc = 0.04
        reach = 0.7
        path = [(px, yq + 0.002, za), (px, yq - reach * 0.5, za), (px, yq - reach, za)]
        for i in range(1, 25):
            ph = i / 24 * 1.3 * 2 * pi
            r = lerp(Rc, 0.01, i / 24)
            path.append((px, yq - reach - r * sin(ph), za + Rc - r * cos(ph)))
        irons.append(sweep("lb_arm", path, rounded_rect(0.016, 0.016, 0.003, 1), up=(1, 0, 0),
                           scale_fn=lambda t: 1.0 - 0.5 * smoothstep(0.45, 1.0, t)))
        stay = bezier_path((px, yq - 0.004, za - 0.38), (px, yq - 0.34, za - 0.38), (px, yq - 0.53, za - 0.26), (px, yq - 0.53, za - 0.008), n=14)
        irons.append(sweep("lb_stay", stay, rounded_rect(0.02, 0.007, 0.002, 1), up=(1, 0, 0)))
        curl = [(px, yq - 0.22 - (0.012 + 0.06 * t) * cos(t * 2.6 * pi), za - 0.105 - (0.012 + 0.06 * t) * sin(t * 2.6 * pi)) for t in
                [i / 22 for i in range(23)]]
        irons.append(sweep("lb_curl", curl, rounded_rect(0.016, 0.006, 0.002, 1), up=(1, 0, 0), scale_fn=lambda t: 0.55 + 0.45 * t))
        irons.append(box("lb_plate", (0.045, 0.007, 0.5), loc=(px, yq - 0.0035, za - 0.17), bevel=0.002))
        for dz in (0.05, -0.19, -0.39):
            irons.append(bolt("lb_bolt", 0.009, 0.007, loc=(px, yq - 0.007, za + dz), rot=(90, 0, 0), kind="pyramid"))
        lantern("lantern_head", (px, yq - reach + 0.06, za - 0.008), 0.09, float(p.intensity), 0.95)
    if p.lanterns >= 2:
        hx = (pxs[0] + pxs[1]) / 2
        drop = clamp(zsoff - 0.33 - 2.0, 0.06, 0.3)
        irons.append(box("ul_plate", (0.07, 0.05, 0.006), loc=(hx, yF + 0.065, zsoff - 0.003), bevel=0.0015))
        lantern("lantern_under", (hx, yF + 0.065, zsoff - 0.004), drop, float(p.intensity) * 0.55)

    # ------------------------------------------------------------------ fixings, materials, weathering
    peg = cylinder("peg", 0.011, 0.018, verts=7)
    pegs = instances_to_mesh("pegs", peg, pegs_m, keep_base=False)
    assign(pegs, endg)
    nail = bolt("nail", 0.0075, 0.005, kind="pyramid")
    irons.append(instances_to_mesh("tread_nails", nail, nails_m, keep_base=False))

    for o in frame:
        assign(o, oak)
        paint(o, grime=mask_mul(low_grime, 0.5 + 0.4 * age), special=moss_low)
    for o in trims:
        assign(o, trim)
        if painted:
            paint(o, wear=mask_mul(noise_mask(0.12, seed=p.seed + 14, lo=0.45, hi=0.8), 0.35 + 0.5 * age), special=0.2 + 0.4 * age)
        else:
            paint(o, wear=mask_mul(facing((0, 0, 1)), noise_mask(0.25, seed=p.seed + 14, lo=0.2, hi=0.7), 0.5 + 0.3 * age))
    for o in wall_rail:
        assign(o, oak)
        paint(o, wear=mask_mul(noise_mask(0.3, seed=p.seed + 15, lo=0.1, hi=0.6), 0.75))
    for o in irons:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.05, seed=p.seed + 16, lo=0.4, hi=0.8), 0.35 + 0.5 * age))
    for o in stones:
        assign(o, stone)
        paint(o, grime=height_ramp(0.0, 0.3), special=mask_mul(noise_mask(0.16, seed=p.seed + 17, lo=0.35, hi=0.7), 0.2 + 0.7 * age),
              wear=mask_mul(foot_wear(), 0.8) if o.name.startswith("step_stone") else None)

    # ------------------------------------------------------------------ sockets
    socket("door", "attach", (d * (xl + x1) / 2, 0.0, R), role="door")
    socket("under", "attach", (d * (xl + x1) / 2, (yW + yF) / 2, 0.0), role="prop")
    socket("sign", "attach", (d * pxs[-1], yP - PS / 2, R + 0.55), role="sign")

    # ------------------------------------------------------------------ mirror for a left-hand stair
    if d < 0:
        S = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))
        for o in list(bpy.context.scene.objects):
            if o.type != "MESH":
                continue
            realize(o)
            apply_transform(o)
            o.data.transform(S)
            o.data.flip_normals()
            o.data.update()
