"""Timber garden pergola / arbour over a path or bench. Stop-chamfered posts stand in wrought-iron
shoes on two-course stone pads; two long beams with shaped tails are tenoned and pegged onto the
post heads, curved (or straight) knee braces stiffen every post both ways, cross ties and rafters
with shaped tails are notched over the beams and carry thin battens. A woody climber is rooted in
small kerbed planting pockets beside some of the pads, twines up the posts and spreads over the roof
in mounded clumps of folded leaves joined by bare runners, spilling over the edges; wisteria racemes
(or roses) hang from it. A four-sided wrought-iron lantern hangs on a chain from a twisted iron bar
between two rafters in every bay (or lantern sockets are left for the town to fill). Without a vine
a lashed canvas shade can be laid over the battens.
length / width / height are the overall extents of the timber frame, centred on the origin.
"""
from math import cos, sin, pi, radians, degrees, atan2, hypot, sqrt, acos, asin, exp, copysign

import numpy as np
from mathutils import Vector

from tk import *

META = {"name": "Timber pergola / arbour", "class": "structure", "mount": "ground", "focus": [-1.25, -0.9, 2.45]}
PARAMS = {
    "length": {"default": 4.4, "min": 2.2, "max": 8.0, "desc": "overall length along X, beam tip to beam tip (m)"},
    "width": {"default": 2.8, "min": 1.6, "max": 4.5, "desc": "overall width along Y, rafter tip to rafter tip (m)"},
    "height": {"default": 2.8, "min": 2.45, "max": 3.6, "desc": "overall height to the top of the roof battens (m)"},
    "bays": {"default": 2, "min": 1, "max": 3, "desc": "bays along the length (posts = 2 x (bays + 1), i.e. 4-8)"},
    "post_size": {"default": 0.14, "min": 0.11, "max": 0.19, "desc": "post section (m); beams scale with it"},
    "ends": {"default": "ogee", "choices": ["ogee", "cove", "bevel"], "desc": "shape cut on the beam and rafter tails"},
    "braces": {"default": "curved", "choices": ["curved", "straight", "none"], "desc": "knee brace style"},
    "trellis": {"default": False, "desc": "diagonal lath trellis panels between the back posts"},
    "finish": {"default": "oak", "choices": ["oak", "weathered", "painted", "tarred"],
               "desc": "timber finish (tarred = brown-black pine tar, ignores tone)"},
    "tone": {"default": "#8f6f4c", "desc": "wood stain tone / paint colour"},
    "vine_density": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "how much climber covers the roof (0 = bare frame)"},
    "leaf_tint": {"default": "#6c8a3c", "desc": "leaf colour (greens, autumn reds)"},
    "flowers": {"default": "wisteria", "choices": ["none", "wisteria", "roses"], "desc": "blossom on the climber"},
    "bloom": {"default": "#a88fdc", "desc": "blossom colour"},
    "shade": {"default": "none", "choices": ["none", "canvas"], "desc": "canvas shade cloths lashed over the battens"},
    "shade_tint": {"default": "#c4b48e", "desc": "canvas colour"},
    "lanterns": {"default": True, "desc": "hang a chain lantern in every bay (else only lantern sockets)"},
    "intensity": {"default": 13.0, "min": 5, "max": 80, "desc": "light intensity per lantern (cd)"},
}
VARIANTS = {
    "painted_trellis": {"length": 6.4, "width": 2.6, "height": 2.7, "bays": 3, "finish": "painted", "tone": "#b3b09c", "intensity": 8,
                        "ends": "cove", "trellis": True, "flowers": "roses", "bloom": "#c8405f", "vine_density": 0.75,
                        "leaf_tint": "#5f8a3a", "seed": 4},
    "small_arbour": {"length": 2.4, "width": 1.9, "height": 2.55, "bays": 1, "finish": "weathered", "tone": "#9c968a", "intensity": 9,
                     "ends": "bevel", "braces": "straight", "flowers": "none", "vine_density": 1.0, "post_size": 0.12,
                     "leaf_tint": "#55803a", "seed": 11},
    "bare_tarred": {"length": 5.2, "width": 3.6, "height": 3.2, "bays": 2, "finish": "tarred", "vine_density": 0.0,
                    "shade": "canvas", "post_size": 0.17, "intensity": 20, "seed": 23},
    "autumn_green_paint": {"length": 3.6, "width": 2.4, "height": 2.7, "bays": 1, "finish": "painted", "tone": "#3d5a45",
                           "ends": "cove", "flowers": "none", "leaf_tint": "#c0612a", "vine_density": 0.85, "intensity": 11,
                           "trellis": True, "seed": 31},
}


# ------------------------------------------------------------------------------------- mask helpers
def _ss(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def _lighten(hx, f):
    """Scale a '#rrggbb' colour (clamped)."""
    h = hx.lstrip("#")
    return "#" + "".join("%02x" % int(clamp(int(h[i:i + 2], 16) * f, 0, 255)) for i in (0, 2, 4))


def mmax(*fs):
    """Union of masks (floats or callables)."""
    return lambda co, n: np.maximum.reduce([f(co, n) if callable(f) else np.full(len(co), float(f)) for f in fs])


def powm(fn, e):
    return lambda co, n: np.clip(fn(co, n), 0.0, 1.0) ** e


def streaky(scale, seed, lo=0.4, hi=0.8, stretch=14.0):
    """Noise stretched vertically: rain streaks / drips running down a face."""
    nm = noise_mask(scale, seed=seed, lo=lo, hi=hi)
    s = np.array([1.0, 1.0, 1.0 / stretch])
    return lambda co, n: nm(co * s, n)


def sides_only(co, n):
    return np.clip(1.0 - np.abs(n[:, 2]) * 1.6, 0.0, 1.0)


# ------------------------------------------------------------------------------------- timber helpers
def tail_profile(L, h, style, tail, step=None):
    """Side elevation of a beam of length L / depth h with a shaped tail at both ends (CCW).
    step adds points along the straight top and bottom edges so weathering masks have resolution."""
    hx = L / 2
    if style == "ogee":
        yn = 0.58 * h
        rc = [(hx - tail + tail * s, yn * (0.5 - 0.5 * cos(pi * s))) for s in [i / 8 for i in range(9)]]
    elif style == "cove":
        r = min(tail - 0.03, 0.62 * h)
        y0 = 0.08 * h
        rc = [(hx - r - 0.03, 0.0), (hx - r - 0.03, y0)]
        rc += [(hx - r * cos(radians(a)), y0 + r * sin(radians(a))) for a in range(0, 91, 10)][1:]
    else:
        rc = [(hx - tail * 0.8, 0.0), (hx, 0.62 * h)]
    rc.append((hx, h))
    left = [(-x, y) for (x, y) in reversed(rc)]
    if not step:
        return list(rc) + left
    n = max(1, int(round(L / step)))
    x_b = rc[0][0]
    top_e = [(hx - 2 * hx * i / n, h) for i in range(1, n)]
    bot_e = [(-x_b + 2 * x_b * i / n, 0.0) for i in range(1, n)]
    return list(rc) + top_e + left + bot_e


def post_mesh(name, a, z0, z1, c_small, c_big, stop_lo, stop_hi):
    """Square post, half-size a, with stopped chamfers (lamb's tongue) between a square foot and head.
    Rings are close together at the foot and head so the splash-back / rain-streak masks are soft."""
    zs = []

    def seg(za, zb, c, step):
        n = max(1, int(round((zb - za) / step)))
        zs.extend((za + (zb - za) * i / n, c) for i in range(n))

    seg(z0, z0 + stop_lo, c_small, 0.07)
    zs += [(z0 + stop_lo, c_small), (z0 + stop_lo + 0.03, c_big * 0.7)]
    seg(z0 + stop_lo + 0.07, z1 - stop_hi - 0.07, c_big, 0.17)
    zs += [(z1 - stop_hi - 0.07, c_big), (z1 - stop_hi - 0.03, c_big * 0.7)]
    seg(z1 - stop_hi, z1, c_small, 0.105)
    zs.append((z1, c_small))
    verts, faces = [], []
    for (z, c) in zs:
        b = a - c
        verts += [(a, -b, z), (a, b, z), (b, a, z), (-b, a, z), (-a, b, z), (-a, -b, z), (-b, -a, z), (b, -a, z)]
    for i in range(len(zs) - 1):
        for k in range(8):
            k2 = (k + 1) % 8
            faces.append((i * 8 + k, i * 8 + k2, (i + 1) * 8 + k2, (i + 1) * 8 + k))
    top = (len(zs) - 1) * 8
    faces.append(tuple(range(top, top + 8)))
    o = mesh_object(name, verts, faces)
    uv_grain(o, 'Z')
    smooth(o, 30)
    return o


def brace_profile(d, tw, style):
    """Knee brace in the (u = away from post, v = up, 0 = soffit) plane, foot at (0,-d), head at (d,0)."""
    if style == "straight":
        k = tw * sqrt(2)
        return [(0.0, -d), (d, 0.0), (d - k, 0.0), (0.0, -d + k)]
    c = 1.7 * d
    R = hypot(c, c - d)
    R2 = R + tw
    t0, t1 = atan2(c - d, -c), atan2(c, d - c)
    ta, tb = acos(-c / R2), pi - asin(c / R2)
    n = 9
    inner = [(c + R * cos(t0 + (t1 - t0) * i / n), -c + R * sin(t0 + (t1 - t0) * i / n)) for i in range(n + 1)]
    outer = [(c + R2 * cos(tb + (ta - tb) * i / n), -c + R2 * sin(tb + (ta - tb) * i / n)) for i in range(n + 1)]
    return inner + outer


# ------------------------------------------------------------------------------------ foliage bases
def blade_base(name, top, faces, width, grime=0.0, fade=0.0):
    """A small folded blade (unit length along +Y, +Z = upper face) mapped to the whole leaf tile."""
    o = mesh_object(name, top, faces)
    uvl = o.data.uv_layers.new(name="UVMap")
    for li, loop in enumerate(o.data.loops):
        x, y, _ = top[loop.vertex_index]
        uvl.data[li].uv = ((x / width + 0.5) * 0.25, (0.03 + y * 0.94) * 0.25)
    uv_keep(o)
    smooth(o, 80)
    if grime > 0 or fade > 0:
        paint(o, grime=grime, special=fade)
    return o


def leaf_base(grime=0.0):
    """Ovate leaflet with a centre fold, drooping tip and slightly raised margins."""
    top = [(0.0, 0.0, 0.0), (0.25, 0.3, 0.085), (0.2, 0.68, 0.04), (0.0, 1.0, -0.1), (-0.2, 0.68, 0.04), (-0.25, 0.3, 0.085)]
    return blade_base("leaf_base", top, [(0, 1, 2, 3), (0, 3, 4, 5)], 0.56, grime)


def petal_base(grime=0.0, fade=0.0):
    """Cupped, keeled petal / floret: a broad kite folded along its axis."""
    top = [(0.0, 0.0, 0.0), (0.4, 0.52, 0.2), (0.0, 1.0, 0.1), (-0.4, 0.52, 0.2)]
    return blade_base("petal_base", top, [(0, 1, 2), (0, 2, 3)], 0.84, grime, fade)


# ------------------------------------------------------------------------------------------ lantern
def link_chain(name, x, y, z_top, z_bot, link=0.036, wire=0.0034):
    """Vertical chain of oval links (alternating 90 deg), light on triangles."""
    hl, hw = link / 2 - wire, link * 0.31 - wire
    base = sweep(name + "_l", [(hl * cos(2 * pi * k / 6), hw * sin(2 * pi * k / 6), 0.0) for k in range(6)], wire, closed=True,
                 sides=3, smooth_path=False)
    n = max(2, int(round((z_top - z_bot) / (link * 0.74))))
    mats = [trs((x, y, z_top - (i + 0.5) * (z_top - z_bot) / n), (0, 90, 90 * (i % 2))) for i in range(n)]
    return instances_to_mesh(name, base, mats, uv_jitter=False, keep_base=False)


def hang_lantern(i, x, y, z_att, zb, iron, p, r):
    """Four-sided wrought-iron lantern (0.2 m across the roof, 0.32 m tall) on a chain: tapered glazed
    cage with corner bars, solid swept roof with a vent cap that shades the rafters, drop finial,
    S-hook and ring. z_att = underside of the fixed eye, zb = lowest point of the lantern."""
    k2 = sqrt(2)
    wt, wb, hb = 0.18 * r.u(0.97, 1.03), 0.124, 0.185
    zf = zb + 0.03                                # underside of the floor
    z0 = zf + 0.01                                # floor top = foot of the glass
    z1 = z0 + hb                                  # top frame / roof eave
    zr = z1 + 0.062                               # roof top = vent base
    zc = zr + 0.058 + 0.014                       # centre of the top ring
    sw = []
    sw.append(lathe("lant_roof", [(0.0, 0.008), (0.092 * k2, 0.0), (0.103 * k2, -0.005), (0.106 * k2, 0.002), (0.074 * k2, 0.02),
                                  (0.047 * k2, 0.042), (0.034 * k2, 0.062), (0.0, 0.062)], segments=4, loc=(x, y, z1), rot=(0, 0, 45),
                    smooth_angle=15))
    sw.append(lathe("lant_vent", [(0.031, -0.006), (0.031, 0.022), (0.022, 0.026), (0.022, 0.032), (0.045, 0.037), (0.047, 0.041),
                                  (0.027, 0.052), (0.01, 0.056), (0.007, 0.062), (0.0, 0.062)], segments=8, loc=(x, y, zr),
                    smooth_angle=35))
    sw.append(torus("lant_ring", 0.012, 0.003, loc=(x, y, zc), rot=(90, 0, 0), major_seg=10, minor_seg=4))
    hook = [(0.011 * cos(radians(t)), 0.014 + 0.011 * sin(radians(t))) for t in range(150, 391, 40)]
    hook += [(0.011 * cos(radians(t)), 0.036 + 0.011 * sin(radians(t))) for t in range(210, -31, -40)]
    sw.append(sweep("lant_hook", [(x + hx, y, zc + hz) for (hx, hz) in hook], 0.0028, sides=5, smooth_path=False))
    z_hook = zc + 0.047
    sw.append(link_chain("lant_chain", x, y, z_att + 0.008, z_hook - 0.006))
    # cage: floor, corner bars, top frame, drop finial, burner
    sw.append(box("lant_floor", (wb + 0.016, wb + 0.016, 0.01), loc=(x, y, zf + 0.005), bevel=0.002, segments=1))
    b = 0.0042
    sq = [(-b, -b), (b, -b), (b, b), (-b, b)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            sw.append(sweep("lant_bar", [(x + sx * wb / 2, y + sy * wb / 2, z0 - 0.002), (x + sx * wt / 2, y + sy * wt / 2, z1 + 0.002)], sq,
                            smooth_path=False))
        sw.append(box("lant_rail", (0.009, wt + 0.012, 0.013), loc=(x + sx * wt / 2, y, z1 - 0.0065), bevel=0.0015, segments=1))
        sw.append(box("lant_rail", (wt + 0.012, 0.009, 0.013), loc=(x, y + sx * wt / 2, z1 - 0.0065), bevel=0.0015, segments=1))
    sw.append(lathe("lant_finial", [(0.0, -0.03), (0.006, -0.028), (0.009, -0.022), (0.005, -0.015), (0.012, -0.009), (0.026, -0.003),
                                    (0.032, 0.0)], segments=8, loc=(x, y, zf)))
    sw.append(lathe("lant_burner", [(0.024, 0.0), (0.027, 0.01), (0.02, 0.019), (0.009, 0.022), (0.008, 0.03), (0.011, 0.032), (0.011, 0.036), (0.0, 0.036)], segments=8,
                    loc=(x, y, z0)))
    for o in sw:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.04, seed=p.seed + i, lo=0.35, hi=0.75), 0.55), wear=0.15)
    # glass panes set just inside the bars, grimy towards the foot and the corners
    gb, gt = wb / 2 - 0.003, wt / 2 - 0.003
    cs = [(1, -1), (1, 1), (-1, 1), (-1, -1)]
    gv = [(x + cx * gb, y + cy * gb, z0 + 0.001) for (cx, cy) in cs] + [(x + cx * gt, y + cy * gt, z1 - 0.012) for (cx, cy) in cs]
    glass = mesh_object("lant_glass", gv, [(k, (k + 1) % 4, 4 + (k + 1) % 4, 4 + k) for k in range(4)])
    subdivide(glass, 2)
    uv_box(glass)
    assign(glass, mat("glass_lantern", glow=0.55, opacity=0.4))
    paint(glass, grime=mmax(mask_mul(powm(height_ramp(z0, z0 + 0.09), 1.3), 0.8),
                            mask_mul(noise_mask(0.05, seed=p.seed + 40 + i, lo=0.3, hi=0.8), 0.45)))
    zfl = z0 + 0.037
    flame = lathe("lant_flame", [(0.0, 0.0), (0.008, 0.007), (0.011, 0.019), (0.007, 0.036), (0.0, 0.052)], segments=8, loc=(x, y, zfl))
    assign(flame, mat("emissive", "#ffb45e", glow=9))
    part(sw + [glass, flame], f"lantern_{i}", pivot=(x, y, z_att), motion="swing", amp=2.0, speed=0.3)
    # the light sits in the flame: floor, finial and glass are inside shadow_near (no black pool under the lantern),
    # the solid roof 0.13 m above it is not, so it shades the rafters and throws the light down and out
    light(f"lantern_{i}", (x, y, zfl + 0.023), intensity=p.intensity, range=7.5, flicker=0.3, shadow=True, shadow_near=0.105)


# ------------------------------------------------------------------------------------------- build
def build(p, rng):
    L, W, H = p.length, p.width, p.height
    ps = p.post_size
    a = ps / 2
    BH, BW = ps * 1.3, ps * 0.86                 # main beams
    RH, RW, TW = 0.14, 0.06, 0.09                # rafters, cross ties
    BT, BWD = 0.035, 0.05                        # battens
    notch = 0.03
    z_raf_top = H - BT
    z_raf_bot = z_raf_top - RH
    z_beam_top = z_raf_bot + notch
    Hb = z_beam_top - BH                         # beam soffit = post head
    ovx = clamp(L * 0.11, 0.32, 0.6)
    ovy = clamp(W * 0.15, 0.28, 0.55)
    span = L - 2 * ovx                           # keep bays between ~1.1 m and ~3.9 m whatever was asked for
    bays = max(1, min(int(p.bays), int(span / 1.1)), min(3, int(span / 3.9) + 1))
    bay = span / bays
    xp = [-L / 2 + ovx + i * bay for i in range(bays + 1)]
    yp = W / 2 - ovy
    posts = [(x, sy * yp) for x in xp for sy in (-1, 1)]

    tarred = p.finish == "tarred"
    painted = p.finish in ("painted", "tarred")  # coated: no separate end grain, special = sun-fade, wear = chipped coat
    if tarred:
        wood = mat("wood_painted", "#2a2018", rough=0.8, wear=1.5)
    else:
        wood = mat({"oak": "wood_oak", "weathered": "wood_weathered", "painted": "wood_painted"}[p.finish], p.tone,
                   **({} if painted else {"wear": 1.6}))
    endm = wood if painted else mat("wood_endgrain", p.tone)
    stone = mat("stone_rough", "#b1a998", grime=0.55)
    iron = mat("iron_wrought")
    d = p.vine_density
    r_t = rng.fork("timber")
    S = p.seed
    top = facing((0, 0, 1))
    broad = noise_mask(0.55, seed=S + 21, lo=0.25, hi=0.9)      # soft, member-sized tone variation

    def end_grain(o, axis=0):
        if not painted:
            assign_faces(o, endm, lambda c, n: abs(n[axis]) > 0.6)

    # ---- stone pads, iron shoes, posts
    z_pad = 0.24
    z_foot = z_pad + 0.01
    padh = (ps + 0.26) / 2
    r_s = rng.fork("stone")
    bolt_m, shoe_objs = [], []
    for i, (px, py) in enumerate(posts):
        rz = r_s.jitter(0, 4)
        s0 = stone_block("pad_base", (ps + 0.26, ps + 0.26, 0.14), loc=(px, py, 0.07), rot=(0, 0, rz), rng=r_s, chamfer=0.016, detail=0.11)
        s1 = stone_block("pad_die", (ps + 0.13, ps + 0.13, z_pad - 0.14), loc=(px, py, (z_pad + 0.14) / 2), rot=(0, 0, rz * 0.5),
                         rng=r_s, chamfer=0.03, detail=0.2)
        for s in (s0, s1):                       # damp, mossy lower third fading out softly; a little lichen on the weathering
            assign(s, stone)
            paint(s, grime=mask_mul(powm(height_ramp(0.0, 0.12), 1.4), 0.4),
                  special=mmax(mask_mul(height_ramp(0.0, 0.15), noise_mask(0.16, seed=S + i, lo=0.1, hi=0.9), 0.8),
                               mask_mul(top, noise_mask(0.1, seed=S + 60 + i, lo=0.4, hi=0.9), 0.3 + 0.3 * d)))
        shoe_objs.append(box("shoe_plate", (ps + 0.03, ps + 0.03, 0.01), loc=(px, py, z_pad + 0.005), bevel=0.002, segments=1))
        for sx in (-1, 1):
            shoe_objs.append(box("shoe_strap", (0.006, 0.05, 0.22), loc=(px + sx * (a + 0.003), py, z_foot + 0.105), bevel=0.0015, segments=1))
            for zz in (0.07, 0.16):
                bolt_m.append(trs((px + sx * (a + 0.006), py, z_foot + zz), (0, sx * 90, 0)))
        po = post_mesh("post", a, z_foot, Hb, 0.004, ps * 0.2, 0.34, 0.42)
        place(po, (px, py, 0), (0, 0, r_t.jitter(0, 0.5)))
        assign(po, wood)
        # soft splash-back at the foot, rain streaks running down from the head, hand polish, sun-bleached patches
        g_post = mmax(mask_mul(powm(height_ramp(z_foot, z_foot + 0.6), 1.6), 0.5),
                      mask_mul(height_ramp(Hb, Hb - 0.8), streaky(0.03, S + 11 + i, 0.4, 0.85), sides_only, 0.5))
        hand = mask_mul(height_ramp(1.2, 1.55), height_ramp(1.2, 0.85), noise_mask(0.08, seed=S + 7, lo=0.2, hi=0.8), 0.4)
        if painted:
            paint(po, grime=g_post, wear=mmax(mask_mul(hand, 0.6), mask_mul(powm(height_ramp(z_foot, z_foot + 0.3), 2.0), 0.3)),
                  special=mask_mul(broad, 0.4))
        else:
            paint(po, grime=g_post, wear=mmax(hand, mask_mul(broad, 0.3)),
                  special=mask_mul(powm(height_ramp(z_foot, z_foot + 0.5), 1.5), noise_mask(0.3, seed=S + 9, lo=0.1, hi=0.9), 0.2 + 0.3 * d))
        nk = box("post_neck", (ps + 0.034, ps + 0.034, 0.036), loc=(px, py, Hb - 0.36), bevel=0.011, segments=1)
        uv_grain(nk, 'X')
        assign(nk, wood)
        paint(nk, grime=mask_mul(top, 0.4), wear=0.15)
    shoe_objs.append(instances_to_mesh("shoe_bolts", bolt("b", 0.01, 0.008, kind="pyramid"), bolt_m, keep_base=False))
    for o in shoe_objs:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.05, seed=S + 3, lo=0.3, hi=0.7), 0.8), grime=0.3)

    # ---- main beams (along X) on the post heads
    beams, rafters, battens = [], [], []
    tail_b = min(1.35 * BH, ovx - a - 0.03)
    for sy in (-1, 1):
        bm = extrude_profile("beam", tail_profile(L, BH, p.ends, tail_b, step=0.13), BW, bevel=0.005)
        uv_grain(bm, 'X')
        place(bm, (0, sy * yp + BW / 2, Hb), (90, 0, 0))
        assign(bm, wood)
        end_grain(bm)
        beams.append(bm)

    # ---- cross ties (over the posts) and rafters, notched over the beams
    xs = []
    for i in range(bays):
        nsub = max(1, round(bay / 0.5))
        xs += [(xp[i] + j * bay / nsub, j == 0) for j in range(nsub)]
    xs.append((xp[-1], True))
    if ovx >= 0.4:
        xs = [(-L / 2 + 0.13, False)] + xs + [(L / 2 - 0.13, False)]
    tail_r = min(1.3 * RH, ovy - BW / 2 - 0.03)
    raf = []                                     # (x, width) of every rafter / tie as built
    for (x, tie) in xs:
        w = TW if tie else RW
        Lr = W - (0.0 if tie else r_t.u(0.0, 0.03))
        rf = extrude_profile("tie" if tie else "rafter", tail_profile(Lr, RH, p.ends, tail_r, step=0.3), w, bevel=0.004 if tie else 0.0)
        uv_grain(rf, 'X')
        jx = 0.0 if tie else r_t.jitter(0, 0.012)
        place(rf, (x + jx - w / 2, 0, z_raf_bot), (90 + (0 if tie else r_t.jitter(0, 0.8)), 0, 90))
        assign(rf, wood)
        end_grain(rf)
        rafters.append(rf)
        raf.append((x + jx, w))
    raf_x = [x for (x, _) in raf]

    # ---- battens along the top: thin chamfered laths, slightly bowed
    nb = max(3, round(W / 0.42))
    bat_y = [-(W / 2 - 0.13) + i * (W - 0.26) / (nb - 1) for i in range(nb)]
    hw_, hh_, ch_ = BWD / 2, BT / 2, 0.005
    bprof = [(-hw_ + ch_, -hh_), (hw_ - ch_, -hh_), (hw_, -hh_ + ch_), (hw_, hh_ - ch_), (hw_ - ch_, hh_), (-hw_ + ch_, hh_),
             (-hw_, hh_ - ch_), (-hw_, -hh_ + ch_)]
    for y in bat_y:
        Lb = L - 0.12 - r_t.u(0, 0.1)
        x0 = -Lb / 2 + r_t.jitter(0, 0.03)
        bow, ph = r_t.jitter(0, 0.006), r_t.u(0, 3)
        nseg = max(4, int(Lb / 0.3))
        path = [(x0 + Lb * k / nseg, y + bow * sin(pi * k / nseg + ph), z_raf_top + BT / 2 - 0.001) for k in range(nseg + 1)]
        bt = sweep("batten", path, bprof, smooth_path=False)
        uv_grain(bt, 'X')
        smooth(bt, 30)
        assign(bt, wood)
        end_grain(bt)
        battens.append(bt)

    # ---- knee braces + oak pegs
    peg_m, braces = [], []

    def peg(loc, e):
        peg_m.append(trs(loc, (0, 90, degrees(atan2(e[1], e[0])))))

    if p.braces != "none":
        bt_ = min(0.07, ps * 0.5)
        tw = ps * 0.72
        dx = clamp(min(0.62, 0.36 * (bay - ps)), 0.22, 0.62)
        dy = clamp(min(0.56, 0.36 * (2 * yp - ps)), 0.2, 0.56)

        def brace(px, py, dirx, diry, dd, ztop, host_half):
            phi = atan2(diry, dirx)
            e = (sin(phi), -cos(phi))
            o = extrude_profile("brace", brace_profile(dd, tw, p.braces), bt_)
            uv_grain(o, (1, 1, 0))
            place(o, (px + a * dirx - e[0] * bt_ / 2, py + a * diry - e[1] * bt_ / 2, ztop), (90, 0, degrees(phi)))
            assign(o, wood)
            braces.append(o)
            for s in (-1, 1):                    # pegs through the post and through the beam / tie
                peg((px + (a - 0.035) * dirx + s * e[0] * (a + 0.001), py + (a - 0.035) * diry + s * e[1] * (a + 0.001),
                     ztop - dd + tw * 0.75), (s * e[0], s * e[1]))
                peg((px + (a + dd - tw * 0.7) * dirx + s * e[0] * (host_half + 0.001),
                     py + (a + dd - tw * 0.7) * diry + s * e[1] * (host_half + 0.001), ztop + 0.04), (s * e[0], s * e[1]))

        for i, x in enumerate(xp):
            for sy in (-1, 1):
                if i < bays:
                    brace(x, sy * yp, 1, 0, dx, Hb, BW / 2)
                if i > 0:
                    brace(x, sy * yp, -1, 0, dx, Hb, BW / 2)
                brace(x, sy * yp, 0, -sy, dy, z_raf_bot, TW / 2)
    for (px, py) in posts:                       # tenon pegs: beam onto post head
        for s in (-1, 1):
            for ox in (-0.028, 0.028):
                peg((px + ox, py + s * (BW / 2 + 0.001), Hb + 0.05 + (0.02 if ox > 0 else 0)), (0, s))
    pg = cylinder("peg", 0.011, 0.02, verts=5)
    pegs = instances_to_mesh("pegs", pg, peg_m, keep_base=False)
    assign(pegs, endm)

    # ---- trellis panels on the back (+Y) side
    trel = []
    if p.trellis:
        r_l = rng.fork("trellis")
        z0 = z_foot + 0.3
        z1 = Hb - 0.72
        ty = yp
        for i in range(bays):
            x0, x1 = xp[i] + a, xp[i + 1] - a
            for z in (z0, z1):
                rl = plank("trellis_rail", x1 - x0, 0.045, 0.06, loc=((x0 + x1) / 2, ty, z), rng=r_l, end_mat=None)
                assign(rl, wood)
                trel.append(rl)
            za, zb = z0 + 0.03, z1 - 0.03
            sp = 0.15 * sqrt(2)
            laths = []
            for sgn in (1, -1):
                c = (x0 - zb) if sgn > 0 else (x0 + za)
                cmax = (x1 - za) if sgn > 0 else (x1 + zb)
                c += r_l.u(0, sp * 0.5)
                while c < cmax:
                    if sgn > 0:
                        xa, xb = max(x0, za + c), min(x1, zb + c)
                        pa, pb = (xa, xa - c), (xb, xb - c)
                    else:
                        xa, xb = max(x0, c - zb), min(x1, c - za)
                        pa, pb = (xa, c - xa), (xb, c - xb)
                    ln = hypot(pb[0] - pa[0], pb[1] - pa[1])
                    if xb - xa > 0.05:
                        lt = plank("lath", ln + 0.02, 0.03, 0.009, loc=((pa[0] + pb[0]) / 2, ty + sgn * 0.0048, (pa[1] + pb[1]) / 2),
                                   rot=(90, -45 * sgn, 0), bevel=0.0, rng=r_l)
                        laths.append(lt)
                    c += sp
            if laths:
                lj = join(laths, "trellis_laths")
                assign(lj, wood)
                trel.append(lj)
        for o in trel:
            paint(o, grime=mask_mul(powm(height_ramp(z0 - 0.05, z0 + 0.5), 1.5), 0.35), special=mask_mul(broad, 0.35) if painted else None,
                  wear=None if painted else mask_mul(broad, 0.3))

    # ---- weathering of the roof timbers: the story is rain and sun from above
    rx_arr = np.array(raf_x)

    def beam_drips(co, n):                       # dark runs down the beam faces under every rafter seat
        dxm = np.min(np.abs(co[:, 0:1] - rx_arr[None, :]), axis=1)
        return np.exp(-(dxm / 0.045) ** 2) * np.clip(np.abs(n[:, 1]) * 1.5 - 0.3, 0, 1) * (0.3 + 0.7 * np.clip((co[:, 2] - Hb) / BH, 0, 1))

    def tail_wet(co, n):                         # rafter / beam tails stick out into the rain: streaked, darker ends
        ty_ = np.clip((np.abs(co[:, 1]) - (W / 2 - 0.42)) / 0.3, 0, 1)
        tx_ = np.clip((np.abs(co[:, 0]) - (L / 2 - 0.5)) / 0.36, 0, 1)
        return np.maximum(ty_, tx_)

    bleach = mask_mul(powm(top, 0.5), mmax(0.6, noise_mask(0.4, seed=S + 31, lo=0.2, hi=0.85)))
    algae = mask_mul(top, noise_mask(0.3, seed=S + 5, lo=0.3, hi=0.8), 0.15 + 0.4 * d)
    st_r = streaky(0.035, S + 15, 0.35, 0.8)
    for o in beams:
        g_ = mmax(mask_mul(beam_drips, 0.6), mask_mul(tail_wet, st_r, sides_only, 0.55), mask_mul(top, 0.25))
        if painted:
            paint(o, grime=g_, special=mmax(mask_mul(bleach, 0.7), mask_mul(broad, 0.3)), wear=mask_mul(tail_wet, 0.2))
        else:
            paint(o, grime=g_, wear=mmax(mask_mul(bleach, 0.55), mask_mul(broad, 0.3)), special=algae)
    for o in rafters:
        g_ = mmax(mask_mul(tail_wet, st_r, sides_only, 0.6), mask_mul(top, 0.2))
        if painted:
            paint(o, grime=g_, special=mmax(mask_mul(bleach, 0.7), mask_mul(broad, 0.3)), wear=mask_mul(tail_wet, 0.2))
        else:
            paint(o, grime=g_, wear=mmax(mask_mul(bleach, 0.7), mask_mul(broad, 0.3)), special=algae)
    for o in battens:                            # the most exposed members: sun-bleached pale on top
        if painted:
            paint(o, special=mmax(mask_mul(bleach, 0.95), 0.25), wear=mask_mul(top, noise_mask(0.12, seed=S + 33, lo=0.45, hi=0.8), 0.45),
                  grime=mask_mul(sides_only, 0.2))
        else:
            paint(o, wear=mmax(mask_mul(bleach, 1.0), 0.2), special=mask_mul(algae, 0.5), grime=mask_mul(sides_only, 0.25))
    for o in braces:
        if painted:
            paint(o, special=mask_mul(broad, 0.35), grime=mask_mul(top, 0.25))
        else:
            paint(o, wear=mask_mul(broad, 0.3), grime=mask_mul(top, 0.25))

    # ---- lanterns / lantern sockets on the centre line, one per bay, hung from a twisted iron bar between two rafters
    rs = sorted(raf)
    z_bar = z_raf_top - 0.04
    lant_xy = []
    for i in range(bays):
        xc = (xp[i] + xp[i + 1]) / 2
        k = min(range(len(rs) - 1), key=lambda j: abs((rs[j][0] + rs[j + 1][0]) / 2 - xc))
        xl, xr = rs[k][0] + rs[k][1] / 2, rs[k + 1][0] - rs[k + 1][1] / 2
        xm = (xl + xr) / 2
        ly = rng.jitter(0, 0.03)
        z_att = z_bar - 0.006 - 0.024
        if p.lanterns:
            fixed = [sweep("lant_hanger", [(xl + 0.004, ly, z_bar), (xr - 0.004, ly, z_bar)], [(-0.006, -0.006), (0.006, -0.006),
                                                                                             (0.006, 0.006), (-0.006, 0.006)],
                           smooth_path=False, step=0.022, twist_deg=540)]
            nail_m = []
            for (xf, sx) in ((xl, 1), (xr, -1)):
                fixed.append(box("lant_flange", (0.012, 0.03, 0.075), loc=(xf + sx * 0.002, ly, z_bar - 0.018), bevel=0.002, segments=1))
                for zz in (0.008, -0.042):
                    nail_m.append(trs((xf + sx * 0.008, ly, z_bar + zz), (0, sx * 90, 0)))
            fixed.append(instances_to_mesh("lant_nails", bolt("n", 0.006, 0.005, kind="pyramid"), nail_m, keep_base=False))
            fixed.append(torus("lant_eye", 0.012, 0.0032, loc=(xm, ly, z_bar - 0.006 - 0.011), rot=(90, 0, 90), major_seg=10, minor_seg=4))
            for o in fixed:
                assign(o, iron)
                paint(o, special=mask_mul(noise_mask(0.05, seed=S + 50 + i, lo=0.3, hi=0.75), 0.6))
            zb = clamp(Hb - 0.47, 2.0, 2.35)     # body hangs clear below the beam soffit, bottom about 2 m up
            zb = min(zb, z_att - 0.05 - 0.411)
            hang_lantern(i, xm, ly, z_att, zb, iron, p, rng.fork(f"lantern{i}"))
            lant_xy.append((xm, ly))
        else:
            socket(f"lantern_{i}", "attach", (xm, 0, z_raf_bot), role="lantern")
        for sy in (-1, 1):
            socket(f"lantern_beam_{i}_{'f' if sy < 0 else 'b'}", "attach", (xc, sy * yp, Hb - 0.03), role="lantern")
            for o in (cylinder("hook_plate", 0.02, 0.005, loc=(xc, sy * yp, Hb - 0.0025), verts=8),
                      torus("hook_eye", 0.012, 0.0032, loc=(xc, sy * yp, Hb - 0.016), rot=(90, 0, 90), major_seg=8, minor_seg=4)):
                assign(o, iron)
    socket("bench", "attach", (0, 0, 0), role="seat")

    g = dict(L=L, W=W, H=H, Hb=Hb, BH=BH, a=a, BW=BW, yp=yp, xp=xp, z_raf_bot=z_raf_bot, raf=raf, bat_y=bat_y, z_pad=z_pad, padh=padh,
             lant_xy=lant_xy, stone=stone)
    # ---- canvas shade cloths
    if p.shade == "canvas":
        build_shade(p, rng.fork("shade"), g)
    # ---- the climber
    if d > 0.02:
        build_vine(p, rng.fork("vine"), posts, g)


# ------------------------------------------------------------------------------------- canvas shade
def build_shade(p, r, g):
    """One sailcloth per bay laid over the battens: bellies between them, lashed to every batten at
    both ends through a roped hem."""
    L, W, H, xp, bat_y = g["L"], g["W"], g["H"], g["xp"], g["bat_y"]
    cv = mat("canvas", p.shade_tint)
    ropem = mat("rope")
    sp = bat_y[1] - bat_y[0]
    y0, y1 = bat_y[0] - sp / 4, bat_y[-1] + sp / 4
    nv = 4 * (len(bat_y) - 1) + 2
    lash = []
    for i in range(len(xp) - 1):
        x0, x1 = xp[i] + 0.12 + r.u(0, 0.06), xp[i + 1] - 0.12 - r.u(0, 0.06)
        ph, slack = r.u(0, 6.28), r.u(0.75, 1.0)

        def f(u, v, x0=x0, x1=x1, ph=ph, slack=slack):
            x, y = lerp(x0, x1, u), lerp(y0, y1, v)
            q = (y - bat_y[0]) / sp
            taut = 0.45 + 0.55 * sin(pi * u) ** 0.5          # pulled tight at the lashed ends
            if 0.0 <= q <= len(bat_y) - 1:
                fr = q % 1.0
                sag = sin(pi * fr) ** 2 * 0.032 * slack * taut * (0.7 + 0.3 * sin(x * 2.7 + ph + 1.9 * int(q)))
                sag -= 0.004 * sin(x * 21 + ph + int(q)) * sin(pi * fr) * taut
            else:
                e = min(abs(y - bat_y[0]), abs(y - bat_y[-1])) / (sp / 4)
                sag = 0.045 * e ** 1.4 * (0.7 + 0.3 * sin(x * 5 + ph))
            return (x, y, H + 0.006 - sag)

        o = surface("shade_canvas", f, nu=max(8, int((x1 - x0) / 0.15)), nv=nv, thickness=0.003)
        assign(o, cv)
        paint(o, grime=lambda co, n: np.clip((H + 0.004 - co[:, 2]) / 0.035, 0, 1) * 0.6,
              special=mask_mul(noise_mask(0.5, seed=p.seed + 70 + i, lo=0.2, hi=0.9), 0.7))
        for u in (0.0, 1.0):
            hem = sweep("shade_hem", [tuple(np.add(f(u, k / nv), (0, 0, 0.004))) for k in range(nv + 1)], 0.006, sides=5, smooth_path=False)
            assign(hem, ropem)
            xe = lerp(x0, x1, u)
            for y in bat_y:
                lash.append(trs((xe + (0.012 if u == 0 else -0.012) + r.jitter(0, 0.004), y, H - 0.0175 + 0.004), (r.jitter(0, 8), 90, 0)))
    lb = torus("lash", 0.037, 0.0045, major_seg=8, minor_seg=4)
    lo = instances_to_mesh("shade_lashings", lb, lash, uv_jitter=False, keep_base=False)
    assign(lo, ropem)


# ------------------------------------------------------------------------------------------ climber
def build_vine(p, r, posts, g):
    L, W, H, Hb, a, BW, padh = g["L"], g["W"], g["H"], g["Hb"], g["a"], g["BW"], g["padh"]
    bat_y, raf = g["bat_y"], g["raf"]
    d = p.vine_density
    S = p.seed
    bark = mat("bark")
    leafm = mat("leaf", p.leaf_tint, rough=1.45)
    corem = mat("hedge", _lighten(p.leaf_tint, 1.25), scale=0.6)
    ex, ey = L / 2 - 0.05, W / 2 - 0.05

    def drape(x, y, z, out=0.13, fall=1.5):
        """Whatever grows out past the roof edge folds down into a hanging curtain (<= 0.1 m outside the frame)."""
        ox, oy = max(0.0, abs(x) - ex), max(0.0, abs(y) - ey)
        if ox > 0:
            x = copysign(ex + out * (1 - exp(-ox / 0.1)), x)
        if oy > 0:
            y = copysign(ey + out * (1 - exp(-oy / 0.1)), y)
        o = hypot(ox, oy)
        return x, y, z - fall * max(0.0, o - 0.04), o

    lm, lm_dark = [], []                         # leaf matrices (two tones)

    def leaf(loc, rot, sc):
        (lm_dark if r.chance(0.27) else lm).append(trs(loc, rot, sc))

    # ---- roots: a kerbed planting pocket beside the pad, a flared trunk that twines up the post and over the beam
    order = sorted(posts, key=lambda _: r.u())
    front = sorted([q for q in order if q[1] < 0], key=lambda q: abs(q[0]))   # first plant: front row, nearest the middle
    if front:
        order.remove(front[0])
        order.insert(0, front[0])
    nroot = int(clamp(round(d * len(posts) * 0.55), 1, len(posts)))
    roots = order[:1]
    while len(roots) < nroot:                    # spread the plants out along the pergola
        roots.append(max((q for q in order if q not in roots),
                         key=lambda q: min(hypot(q[0] - c[0], (q[1] - c[1]) * 0.6) for c in roots) + r.u(0, 0.3)))
    entries, woody = [], []
    for k, (px, py) in enumerate(roots):
        sy = 1 if py > 0 else -1
        cx, cy = px + r.jitter(0, 0.015), py + sy * (padh + 0.01)
        ryk = clamp(W / 2 - abs(py) - padh - 0.03, 0.13, 0.22) * r.u(0.97, 1.05)   # wide, shallow bed: stays in the footprint
        rxk = r.u(0.215, 0.25)
        nk = int(round((rxk + ryk) * 1.62 / 0.1))
        kerb = []
        for i in range(nk):
            th = radians(9 + 162 * i / (nk - 1)) + r.jitter(0, 0.03)
            sl, sh = r.u(0.088, 0.108), r.u(0.07, 0.095)
            tang = degrees(atan2(sy * ryk * cos(th), -rxk * sin(th)))
            st = box("kerb", (sl, r.u(0.058, 0.072), sh), loc=(cx + rxk * cos(th), cy + sy * ryk * sin(th), sh / 2 + 0.009),
                     rot=(r.jitter(0, 3), r.jitter(0, 3), tang + r.jitter(0, 7)), bevel=0.013, segments=1)
            displace_noise(st, strength=0.004, scale=0.05, seed=S + 100 + 10 * k + i)
            kerb.append(st)
        kj = join(kerb, "pocket_kerb")
        assign(kj, g["stone"])
        smooth(kj, 50)
        paint(kj, grime=mask_mul(height_ramp(0.0, 0.07), 0.45),
              special=mmax(mask_mul(height_ramp(0.0, 0.1), 0.6), mask_mul(noise_mask(0.08, seed=S + 90 + k, lo=0.3, hi=0.8), 0.6)))
        so = surface("pocket_soil", lambda u, v: (cx + (rxk - 0.015) * u * cos(pi * (v if sy > 0 else 1 - v)),
                                                  cy + sy * (ryk - 0.015) * u * sin(pi * (v if sy > 0 else 1 - v)),
                                                  0.036 + 0.04 * (1 - u * u)), nu=3, nv=8)
        displace_noise(so, strength=0.007, scale=0.06, seed=S + 80 + k)
        realize(so)
        uv_transform(so, scale=(3.0, 3.0))
        uv_keep(so)
        assign(so, bark)                         # bark-mulch over the earth
        paint(so, grime=mask_mul(noise_mask(0.12, seed=S + 85 + k, lo=0.2, hi=0.8), 0.7), special=0.0)
        # trunk
        r0 = r.u(0.035, 0.041)                   # 7-8 cm stem, flaring to ~11 cm at the soil
        bx_, by_ = cx + r.jitter(0, 0.05), cy + sy * ryk * r.u(0.4, 0.52)
        rho_s = a + 0.045
        dirn = r.pick([-1, 1])
        sxo = (-1 if px > 0 else 1) if abs(px) > 0.05 else r.pick([-1, 1])
        z_s0, z_s1 = 0.64, Hb - 0.8
        path = [(bx_, by_, 0.012), (bx_ + r.jitter(0, 0.012), by_ - sy * 0.005, 0.15),
                (lerp(bx_, px, 0.6), py + sy * (padh + 0.02), 0.33), (px + dirn * 0.01, py + sy * (a + 0.075), 0.5)]
        nsp = 12
        for i in range(nsp + 1):
            t = i / nsp
            ang = sy * pi / 2 + dirn * 2 * pi * t
            path.append((px + rho_s * cos(ang), py + rho_s * sin(ang), lerp(z_s0, z_s1, t)))
        n_spiral = len(path)
        y_e = py + sy * (BW / 2 + 0.045)
        for _ in range(6):                       # come up between two battens
            if min(abs(y_e - yb) for yb in bat_y) < 0.055:
                y_e -= sy * 0.03
        path += [(px + sxo * 0.03, py + sy * (a + 0.04), Hb - 0.55), (px + sxo * 0.06, py + sy * (a + 0.045), Hb - 0.3),
                 (px + sxo * 0.1, py + sy * (a + 0.035), Hb - 0.02), (px + sxo * 0.13, py + sy * max(a + 0.03, BW / 2 + 0.03), Hb + g["BH"] * 0.7),
                 (px + sxo * 0.155, y_e, H - 0.07), (px + sxo * 0.17, y_e - sy * 0.03, H + 0.012), (px + sxo * 0.2, y_e - sy * 0.16, H + 0.02)]
        tr = sweep("vine_trunk", path, r0, sides=7, step=0.095,
                   scale_fn=lambda t: 0.48 + 0.52 * (1 - t) ** 1.4 + 0.5 * exp(-t * 48))
        displace_noise(tr, strength=0.005, scale=0.07, seed=S + 120 + k)
        woody.append(tr)
        # a second, thinner stem twining round the trunk up to the knee braces
        P = np.asarray(resample_path(path[:n_spiral], step=0.065))
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1)[:, None]
        tw = []
        ph = r.u(0, 6.28)
        for j in range(len(P)):
            N = np.cross(T[j], (0.0, 0.0, 1.0))
            if np.linalg.norm(N) < 0.2:
                N = np.cross(T[j], (1.0, 0.0, 0.0))
            N /= np.linalg.norm(N)
            B = np.cross(T[j], N)
            t = j / (len(P) - 1)
            rr = r0 * (0.75 + 0.25 * (1 - t)) * 0.8 + 0.012
            an = ph + j * 0.065 * 2 * pi / 0.34
            q = P[j] + rr * (cos(an) * N + sin(an) * B)
            tw.append((q[0], q[1], max(q[2], 0.012)))
        t2 = sweep("vine_twine", tw, 0.013, sides=4, smooth_path=False, scale_fn=lambda t: 1.0 - 0.5 * t)
        woody.append(t2)
        entries.append((path[-1][0], path[-1][1]))
        # a few leafy side shoots on the trunk
        for _ in range(4):
            j = int(r.u(0.3, 0.98) * (len(P) - 1))
            q = P[j]
            yaw0 = degrees(atan2(q[1] - py, q[0] - px))
            for _ in range(r.pick([3, 4, 5])):
                leaf((q[0] + r.jitter(0, 0.04), q[1] + r.jitter(0, 0.04), q[2] + r.jitter(0, 0.06)),
                     (r.jitter(-30, 30), r.jitter(0, 30), yaw0 - 90 + r.jitter(0, 60)), r.u(0.05, 0.08))

    # ---- mounded clumps over the roof (corners first), joined to the trunks by bare woody runners
    def near_entry(x, y):
        return min(hypot(x - q[0], y - q[1]) for q in entries)

    n_cl = int(clamp(round(d * L * W / 0.37), 1, 36))
    clumps = []

    def add_clump(x, y, big=False):
        rx = r.u(0.4, 0.5) if big else r.u(0.3, 0.46)
        clumps.append(dict(x=x, y=y, rx=rx, ry=rx * r.u(0.72, 0.95), hz=r.u(0.14, 0.26) * (0.8 + 0.2 * d) * (1.1 if big else 1.0),
                           rot=r.u(0, pi)))

    corners = sorted([(sx * (L / 2 - 0.14), sy * (W / 2 - 0.14)) for sx in (-1, 1) for sy in (-1, 1)], key=lambda c: near_entry(*c))
    for c in corners[:int(clamp(round(d * 4.4), 1, 4))]:
        if len(clumps) < n_cl:
            add_clump(c[0] + r.jitter(0, 0.03), c[1] + r.jitter(0, 0.03), True)
    tries = 0
    while len(clumps) < n_cl and tries < 2500:   # the canopy grows outwards from the trunks as overlapping mounds
        tries += 1
        x, y = r.u(-L / 2 + 0.12, L / 2 - 0.12), r.u(-W / 2 + 0.12, W / 2 - 0.12)
        dn = min(hypot(x - c["x"], y - c["y"]) for c in clumps)
        if dn < 0.4 or min(dn, near_entry(x, y)) > 0.74:
            continue
        add_clump(x, y)
    nodes = [(q[0], q[1], 0) for q in entries]
    for c in sorted(clumps, key=lambda c: near_entry(c["x"], c["y"])):
        n0 = min(nodes, key=lambda q: hypot(q[0] - c["x"], q[1] - c["y"]))
        dist = hypot(c["x"] - n0[0], c["y"] - n0[1])
        nodes.append((c["x"], c["y"], n0[2] + 1))
        if dist < 0.12:
            continue
        ux, uy = (c["x"] - n0[0]) / dist, (c["y"] - n0[1]) / dist
        amp, ph = r.u(0.05, 0.13) * min(1.0, dist), r.u(0, 6.28)
        r0 = (0.02, 0.014, 0.01)[min(n0[2], 2)]
        n = max(3, int(dist / 0.22))
        pts = []
        for i in range(n + 1):
            t = i / n
            wv = amp * sin(t * pi) * sin(ph + t * 5.0)
            pts.append((n0[0] + ux * dist * t - uy * wv, n0[1] + uy * dist * t + ux * wv, H + r0 * 0.8 + 0.004 + r.u(0, 0.012)))
        woody.append(sweep("vine_runner", pts, r0, sides=5, step=0.17, scale_fn=lambda t: 1.0 - 0.45 * t))
        for q in resample_path(pts, step=0.16)[1:-1]:             # the odd leaf on the bare runner
            if r.chance(0.55):
                leaf((q[0] + r.jitter(0, 0.03), q[1] + r.jitter(0, 0.03), q[2] + r.u(0.01, 0.04)),
                     (r.jitter(5, 30), r.jitter(0, 35), r.u(0, 360)), r.u(0.05, 0.08))

    # bare shoots exploring the still uncovered battens, with a tuft of young leaves at the tip
    for _ in range(3 + int(round(5 * d))):
        for _try in range(12):
            c = r.pick(clumps)
            an = r.u(0, 2 * pi)
            ln = c["rx"] + r.u(0.45, 1.0)
            x1, y1 = clamp(c["x"] + ln * cos(an), -L / 2 + 0.2, L / 2 - 0.2), clamp(c["y"] + ln * sin(an), -W / 2 + 0.2, W / 2 - 0.2)
            if all(hypot(x1 - q["x"], y1 - q["y"]) > q["rx"] + 0.12 for q in clumps):
                break
        else:
            continue
        ph = r.u(0, 6.28)
        dd_ = hypot(x1 - c["x"], y1 - c["y"])
        nx_, ny_ = -(y1 - c["y"]) / dd_, (x1 - c["x"]) / dd_
        pts = []
        for i in range(7):
            t = i / 6
            wv = 0.07 * sin(t * pi) * sin(ph + t * 6.0)
            pts.append((lerp(c["x"], x1, t) + nx_ * wv, lerp(c["y"], y1, t) + ny_ * wv, H + 0.012 + r.u(0, 0.012)))
        woody.append(sweep("vine_runner", pts, 0.009, sides=4, step=0.17, scale_fn=lambda t: 1.0 - 0.6 * t))
        for q in resample_path(pts, step=0.11)[3:]:
            if r.chance(0.5):
                leaf((q[0] + r.jitter(0, 0.03), q[1] + r.jitter(0, 0.03), q[2] + r.u(0.01, 0.04)),
                     (r.jitter(-5, 30), r.jitter(0, 35), r.u(0, 360)), r.u(0.05, 0.085))
        for _ in range(7):
            leaf((x1 + r.jitter(0, 0.06), y1 + r.jitter(0, 0.06), H + r.u(0.015, 0.07)), (r.jitter(-15, 30), r.jitter(0, 35), r.u(0, 360)),
                 r.u(0.05, 0.085))

    # clump cores (a lumpy, deeply shaded body) + folded leaves standing off them all over
    area = sum(pi * c["rx"] * c["ry"] for c in clumps)
    dens = min(400.0, (1300 + 85 * L * W) / max(area, 0.1))
    cores, drape_pts, top_pts = [], [], []
    for k, c in enumerate(clumps):
        cr, sr = cos(c["rot"]), sin(c["rot"])
        edge = abs(c["x"]) + c["rx"] > ex or abs(c["y"]) + c["ry"] > ey
        co_ = sphere("vine_clump", 1.0, segments=12 if edge else 9, rings=6 if edge else 5)

        def shape(v, c=c, cr=cr, sr=sr):
            zz = c["hz"] * 0.62 * max(v.z, 0.0) ** 0.7 - 0.045 * max(-v.z, 0.0)
            lx, ly = v.x * c["rx"] * 0.72, v.y * c["ry"] * 0.72
            return Vector((c["x"] + lx * cr - ly * sr, c["y"] + lx * sr + ly * cr, H - 0.005 + zz))

        transform_verts(co_, shape)
        displace_noise(co_, strength=0.055, scale=0.14, seed=S + 200 + k)
        # the body stays on the roof; only leaves and tendrils hang over the edge
        transform_verts(co_, lambda v: Vector((clamp(v.x, -ex - 0.02, ex + 0.02), clamp(v.y, -ey - 0.02, ey + 0.02), v.z)))
        cores.append(co_)
        for _ in range(int(dens * pi * c["rx"] * c["ry"])):
            rho, phi = sqrt(r.u(0, 1.12)), r.u(0, 2 * pi)
            lx, ly = rho * cos(phi) * c["rx"], rho * sin(phi) * c["ry"]
            h = c["hz"] * max(0.0, 1 - min(rho, 1.0) ** 2) ** 0.35 + r.u(-0.01, 0.05) - (0.04 if rho > 1 else 0.0)
            wx, wy = c["x"] + lx * cr - ly * sr, c["y"] + lx * sr + ly * cr
            if rho < 0.9 and abs(wx) < ex and abs(wy) < ey and r.chance(0.13):
                # shoots poking down between the battens: nothing flat to look at from underneath
                if min(abs(wy - yb) for yb in bat_y) > 0.04 and all(abs(wx - rx_) > w_ / 2 + 0.015 for (rx_, w_) in raf):
                    leaf((wx, wy, H - r.u(0.045, 0.14)), (-r.u(35, 95), r.jitter(0, 30), r.u(0, 360)), r.u(0.055, 0.09))
                continue
            X, Y, Z, o = drape(wx, wy, H + h)
            if o > 0.06:                         # hanging over the edge: leaves point down and outwards
                e_ang = (0.0 if wx > 0 else 180.0) if abs(wx) - ex > abs(wy) - ey else (90.0 if wy > 0 else -90.0)
                rot = (-r.u(40, 85), r.jitter(0, 35), e_ang - 90 + r.jitter(0, 55))
                if o > 0.12:
                    drape_pts.append((X, Y, Z))
                if r.chance(0.45):               # the curtain is two leaves deep
                    leaf((X + r.jitter(0, 0.035), Y + r.jitter(0, 0.035), Z - r.u(0.02, 0.09)),
                         (-r.u(50, 95), r.jitter(0, 35), e_ang - 90 + r.jitter(0, 70)), r.u(0.055, 0.09))
            else:
                rot = (-50 * rho ** 2 + r.jitter(-6, 32), r.jitter(0, 36), degrees(atan2(wy - c["y"], wx - c["x"])) - 90 + r.jitter(0, 150))
                if r.chance(0.24):               # inner leaves between the shell and the shaded body, at any angle
                    Z = H + max(0.03, h * r.u(0.45, 0.85))
                    rot = (r.jitter(-20, 60), r.jitter(0, 50), r.u(0, 360))
                elif rho < 0.8:
                    top_pts.append((X, Y, Z))
            leaf((X, Y, Z), rot, r.u(0.058, 0.098))
    if cores:
        cj = join(cores, "vine_clumps")
        uv_box(cj)
        smooth(cj, 180)
        assign(cj, corem)
        paint(cj, grime=mmax(0.05, mask_mul(height_ramp(H + 0.03, H + 0.17), 0.6)))

    # ---- leafy tendrils hanging from the clumps on the edges
    edge_cl = [c for c in clumps if abs(c["x"]) + c["rx"] > ex + 0.1 or abs(c["y"]) + c["ry"] > ey + 0.1]
    nh = min(16, int(round(d * (2 * L + 2 * W) * 0.75))) if edge_cl else 0
    for _ in range(nh):
        c = r.pick(edge_cl)
        drop = r.u(0.28, 0.6) * (0.65 + 0.35 * d)
        j = r.jitter(0, 0.06)
        if abs(c["y"]) + c["ry"] - ey > abs(c["x"]) + c["rx"] - ex:
            sy = 1 if c["y"] > 0 else -1
            x = clamp(c["x"] + r.jitter(0, c["rx"] * 0.8), -ex, ex)
            path = [(x, sy * (W / 2 - 0.22), H + 0.07), (x + j, sy * (W / 2 + 0.0), H + 0.05), (x + j * 1.5, sy * (W / 2 + 0.07), H - 0.14),
                    (x + j * 2 + r.jitter(0, 0.04), sy * (W / 2 + 0.065 + r.jitter(0, 0.025)), H - 0.14 - drop)]
        else:
            sx = 1 if c["x"] > 0 else -1
            y = clamp(c["y"] + r.jitter(0, c["ry"] * 0.8), -ey, ey)
            path = [(sx * (L / 2 - 0.22), y, H + 0.07), (sx * (L / 2 + 0.0), y + j, H + 0.05), (sx * (L / 2 + 0.07), y + j * 1.5, H - 0.14),
                    (sx * (L / 2 + 0.065 + r.jitter(0, 0.025)), y + j * 2 + r.jitter(0, 0.04), H - 0.14 - drop)]
        woody.append(sweep("vine_tendril", path, 0.0045, sides=3, step=0.13, scale_fn=lambda t: 1.0 - 0.6 * t))
        for q in resample_path(path, step=0.042):
            if q[2] > H - 0.05:
                continue
            leaf((q[0] + r.jitter(0, 0.015), q[1] + r.jitter(0, 0.015), q[2]), (r.jitter(-58, 24), r.jitter(0, 30), r.u(0, 360)),
                 r.u(0.048, 0.08))
        drape_pts.append((path[-1][0], path[-1][1], path[-1][2] + 0.12))

    for o in woody:
        realize(o)
        uv_transform(o, rot_deg=90)
        uv_keep(o)
        assign(o, bark)
    paint(woody, special=mask_mul(height_ramp(0.0, 0.8), noise_mask(0.15, seed=S + 6, lo=0.2, hi=0.8), 0.7))
    for (mm, gr) in ((lm, 0.0), (lm_dark, 0.42)):
        if mm:
            lb = leaf_base(gr)
            assign(lb, leafm)
            lv = instances_to_mesh("leaves", lb, mm, uv_jitter=False, keep_base=False)
            assign(lv, leafm)

    # ---- blossom
    if p.flowers == "none" or not clumps:
        return
    bloomm = mat("canvas", p.bloom, rough=1.25)
    fl = ([], [], [])                            # petals in three tones (open -> bud): grime darkens towards the raceme tip
    if p.flowers == "wisteria":
        rcores = []
        placed = []

        def raceme(x, y, z, ln, z_from=None):
            R0 = r.u(0.05, 0.06) * (0.8 + 0.5 * ln)
            bend, bph = r.u(0.0, 0.05), r.u(0, 6.28)

            def axis(t):
                return (x + bend * t * t * cos(bph), y + bend * t * t * sin(bph), z - ln * t)

            def rho_n(t):
                return (0.3 + 0.7 * _ss(0.0, 0.17, t)) * (1.0 - 0.84 * t ** 1.2)

            rcores.append(sweep("raceme_core", [axis(t) for t in (0.0, 0.09, 0.22, 0.45, 0.72, 1.0)], R0 * 0.62, sides=4, smooth_path=False,
                                scale_fn=rho_n))
            st = sweep("raceme_stalk", [(x, y, z_from or z + 0.05), (x, y, z - 0.02)], 0.003, sides=3, smooth_path=False)
            assign(st, bark)
            nr = max(6, int(ln / 0.02))
            th = r.u(0, 6.28)
            for i in range(nr):
                t = (i + 0.5) / nr
                rr = R0 * rho_n(t)
                kf = max(3, int(round(2 * pi * rr / 0.04)))
                q = axis(t)
                tone = 0 if t < 0.42 else (1 if t < 0.74 else 2)
                for j in range(kf):
                    th += 2 * pi / kf + r.jitter(0, 0.25)
                    sc = lerp(0.024, 0.014, t) * r.u(0.85, 1.15)
                    fl[tone if r.chance(0.8) else min(2, tone + 1)].append(
                        trs((q[0] + rr * 0.72 * cos(th), q[1] + rr * 0.72 * sin(th), q[2] + 0.008 + r.jitter(0, 0.007)),
                            (-r.u(48, 100), r.jitter(0, 28), degrees(th) - 90 + r.jitter(0, 25)), sc))
                th += 1.1

        def clear(x, y, rad=0.11):
            if any(hypot(x - q[0], y - q[1]) < rad for q in placed) or any(hypot(x - q[0], y - q[1]) < 0.42 for q in g["lant_xy"]):
                return False
            placed.append((x, y))
            return True

        n = min(30, int(round(d * L * W * 3.4)))
        n_edge = min(len(drape_pts), int(n * 0.6))
        k = tries = 0
        while k < n_edge and tries < 300:
            tries += 1
            q = r.pick(drape_pts)
            if q[2] < H - 0.42 or not clear(q[0], q[1]):
                continue
            raceme(q[0], q[1], q[2] - 0.02, r.u(0.25, 0.4))
            k += 1
        tries = 0
        while k < n and tries < 1500:
            tries += 1
            c = r.pick(clumps)
            rr_, ph = sqrt(r.u(0, 0.8)), r.u(0, 6.28)
            x, y = c["x"] + rr_ * c["rx"] * cos(ph), c["y"] + rr_ * c["ry"] * sin(ph)
            if abs(x) > L / 2 - 0.16 or abs(y) > W / 2 - 0.16:
                continue
            if any(abs(x - rx_) < w_ / 2 + 0.06 for (rx_, w_) in raf) or min(abs(y - yy) for yy in bat_y) < 0.068:
                continue
            if abs(abs(y) - g["yp"]) < BW / 2 + 0.06 or not clear(x, y):
                continue
            raceme(x, y, g["z_raf_bot"] + r.u(-0.02, 0.05), r.u(0.27, 0.45), H - 0.03)
            k += 1
        if rcores:
            cj = join(rcores, "raceme_cores")
            displace_noise(cj, strength=0.006, scale=0.03, seed=S + 300)
            smooth(cj, 180)
            assign(cj, bloomm)
            paint(cj, grime=0.3)
    else:
        def rose(x, y, z, s):
            M = trs((x, y, z), (r.jitter(0, 35), r.jitter(0, 35), r.u(0, 360)), s)
            for k in range(5):
                fl[0].append(M @ trs((0, 0, 0), (32, 0, k * 72 + 10), 1.0))
            for k in range(3):
                fl[1].append(M @ trs((0, 0, 0.1), (60, 0, k * 120 + 40), 0.68))
            for k in range(2):
                fl[2].append(M @ trs((0, 0, 0.16), (80, 0, k * 180 + 80), 0.45))

        n = min(90, int(d * L * W * 6), len(top_pts))
        for _ in range(n):
            q = r.pick(top_pts)
            rose(q[0] + r.jitter(0, 0.03), q[1] + r.jitter(0, 0.03), q[2] + r.u(0.02, 0.05), r.u(0.036, 0.052))
        for q in drape_pts[::3][:24]:
            rose(q[0], q[1], q[2], r.u(0.032, 0.045))
    for (mm, gr, fd) in zip(fl, (0.0, 0.12, 0.4), (0.45, 0.0, 0.0)):
        if mm:
            fb = petal_base(gr, fd)
            assign(fb, bloomm)
            blo = instances_to_mesh("blossom", fb, mm, uv_jitter=False, keep_base=False)
            assign(blo, bloomm)
