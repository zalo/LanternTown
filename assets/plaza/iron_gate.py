"""Pair of wrought-iron carriage gates hung on dressed-stone (or brick) piers with ball caps.
Each leaf is built the way a smith makes it: square hinge and meeting stiles, pierced flat rails, square
(optionally twisted) bars with forged spear finials, short dog bars with arrow heads in the bottom
panel, and a lock band of S-scrolls or rings bound with collars. The leaves turn on heel pins in iron
cups on a stone sill and in strap hinges bolted to the piers. An arched overthrow (ring-frieze lintel,
elliptical arch bar, scroll infill) carries a hanging gas lantern and a painted, gilded crest.
"""
from math import atan2, cos, degrees, hypot, pi, radians, sin, sqrt

import numpy as np
from tk import *

META = {"name": "Wrought-iron gate with piers", "class": "structure", "mount": "ground", "focus": 0.7}
PARAMS = {
    "width": {"default": 3.0, "min": 1.6, "max": 4.6, "desc": "clear opening between the piers (m)"},
    "height": {"default": 2.2, "min": 1.5, "max": 3.0, "desc": "leaf height to the top rail at the hinge stile (m)"},
    "pier_style": {"default": "ashlar", "choices": ["ashlar", "rusticated", "brick"], "desc": "pier construction"},
    "pier_width": {"default": 0.62, "min": 0.45, "max": 0.9, "desc": "pier shaft width (m, square)"},
    "pier_cap": {"default": "ball", "choices": ["ball", "urn", "lantern"], "desc": "what stands on the pier cornice"},
    "overthrow": {"default": True, "desc": "arched iron overthrow spanning the piers"},
    "lantern": {"default": True, "desc": "gas lantern hanging in the overthrow"},
    "lantern_roof": {"default": "ogee", "choices": ["ogee", "pyramid", "dome"], "desc": "lantern roof style"},
    "open_angle": {"default": 0.0, "min": 0.0, "max": 90.0, "desc": "how far both leaves stand open, inwards (deg)"},
    "top": {"default": "flat", "choices": ["flat", "rising", "dip"], "desc": "line of the leaf top rail"},
    "ornament": {"default": "scrolls", "choices": ["scrolls", "rings", "plain"], "desc": "lock band / overthrow infill"},
    "twist": {"default": True, "desc": "alternate upper bars are twisted square bar"},
    "gilding": {"default": True, "desc": "gilded finials, crest and collars"},
    "paint": {"default": "#1c2a24", "desc": "paint colour of the ironwork"},
    "crest_paint": {"default": "#7a1414", "desc": "field colour of the crest shield"},
    "monogram": {"default": "T", "desc": "letter(s) on the crest shield"},
    "stone": {"default": "#a89d89", "desc": "stone tint of the piers"},
    "rust": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of rust / weathering on the iron"},
    "intensity": {"default": 42.0, "min": 10, "max": 120, "desc": "light intensity per lantern (cd)"},
}
VARIANTS = {
    "ajar_brick": {"pier_style": "brick", "open_angle": 38, "ornament": "rings", "paint": "#1b2538", "width": 2.6,
                   "height": 2.0, "lantern_roof": "dome", "crest_paint": "#1d3a5c", "monogram": "S", "seed": 4},
    "garden_small": {"width": 1.8, "height": 1.7, "overthrow": False, "pier_cap": "lantern", "top": "rising",
                     "pier_style": "rusticated", "pier_width": 0.5, "paint": "#1f3324", "open_angle": 26,
                     "ornament": "rings", "gilding": False, "rust": 0.9, "intensity": 40, "seed": 11},
    "grand_rusticated": {"width": 4.2, "height": 2.7, "pier_style": "rusticated", "pier_width": 0.8, "pier_cap": "urn",
                         "top": "rising", "paint": "#101214", "lantern_roof": "pyramid", "monogram": "K",
                         "crest_paint": "#1f4a2c", "intensity": 55, "open_angle": 12, "seed": 23},
    "dip_plain": {"width": 2.4, "height": 2.3, "overthrow": False, "pier_cap": "lantern", "top": "dip",
                  "ornament": "plain", "twist": False, "paint": "#3a1512", "stone": "#c9b79a", "pier_width": 0.56,
                  "intensity": 45, "rust": 0.3, "seed": 31},
}

SQ2 = sqrt(2.0)


# ------------------------------------------------------------------------------------------ 2D scroll helpers
def _spiral(E, d, s, R, turns=1.3, k=0.28, per_turn=18):
    """Scroll end: starts at E heading d, curls to side s (+1 left, -1 right) from radius R down to k*R."""
    nx, ny = -d[1] * s, d[0] * s
    cx, cy = E[0] + nx * R, E[1] + ny * R
    a0 = atan2(-ny, -nx)
    n = max(8, int(per_turn * turns))
    pts = []
    for i in range(1, n + 1):
        u = i / n
        th = a0 + s * u * turns * 2 * pi
        r = R * (1 - (1 - k) * u ** 1.2)
        pts.append((cx + r * cos(th), cy + r * sin(th)))
    return pts


def _bez(P0, P1, P2, P3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
        out.append((a * P0[0] + b * P1[0] + c * P2[0] + d * P3[0], a * P0[1] + b * P1[1] + c * P2[1] + d * P3[1]))
    return out


def _scroll(A, B, dA, dB, RA, RB, sA, sB, turns=1.3, tension=0.42, n=12, k=0.28, per_turn=18):
    """Scroll path: spine from A (leaving along dA) to B (arriving along dB) with a spiral eye at each end.
    sA is the curl side of the start eye seen when walking backwards out of A (heading -dA)."""
    L = hypot(B[0] - A[0], B[1] - A[1]) * tension
    spine = _bez(A, (A[0] + dA[0] * L, A[1] + dA[1] * L), (B[0] - dB[0] * L, B[1] - dB[1] * L), B, n)
    head = _spiral(A, (-dA[0], -dA[1]), sA, RA, turns, k, per_turn) if RA > 0 else []
    tail = _spiral(B, dB, sB, RB, turns, k, per_turn) if RB > 0 else []
    return list(reversed(head)) + spine + tail


def _taper(t):
    return 0.55 + 0.45 * smoothstep(0.0, 0.14, min(t, 1.0 - t))


def _rect(a, b, chamfer=True):
    if chamfer:
        return rounded_rect(a, b, min(a, b) * 0.22, 1)
    return [(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)]


def _flat_scroll(name, pts2, thick=0.009, depth=0.022, y=0.0, taper=True, chamfer=True):
    """Sweep flat bar (thin in the plane of the scroll, `depth` deep) along an (x, z) path."""
    return sweep(name, [(x, y, z) for (x, z) in pts2], _rect(thick, depth, chamfer), up=(0, 1, 0),
                 smooth_path=False, scale_fn=_taper if taper else None)


def _ring_base(name, r, thick=0.008, depth=0.02, seg=14, chamfer=False):
    return sweep(name, [(r * cos(2 * pi * k / seg), 0, r * sin(2 * pi * k / seg)) for k in range(seg)],
                 _rect(thick, depth, chamfer), closed=True, up=(0, 1, 0), smooth_path=False)


def _spear(name, seg=8):
    return lathe(name, [(0.0105, 0.0), (0.016, 0.004), (0.018, 0.012), (0.013, 0.02), (0.0085, 0.027), (0.0085, 0.034),
                        (0.022, 0.062), (0.024, 0.078), (0.019, 0.105), (0.009, 0.14), (0.0, 0.16)], segments=seg)


# ------------------------------------------------------------------------------------------ one gate leaf
def _leaf(p, tag, Lw, ztop, zhs, rng, lock, inside):
    """Leaf in its own frame: hinge axis = Z through the origin, leaf runs along +X, y = 0 is its plane."""
    G = {"iron": [], "gilt": [], "black": [], "brass": []}
    I = G["iron"]
    H = p.height
    sh, sm = 0.045, 0.04
    x0, x1 = sh / 2, Lw - sm / 2
    Li = x1 - x0
    xm = (x0 + x1) / 2
    zb = 0.13
    orn = p.ornament
    band = {"scrolls": clamp(0.135 * H, 0.24, 0.32), "rings": 0.112, "plain": 0.0}[orn]
    zl1 = clamp(0.42 * H, 0.7, 1.0)
    zl2 = zl1 + band + 0.022 if band else zl1
    zd = zb + (zl1 - zb) * 0.58

    # stiles
    zt0, zt1 = ztop(0.0) + 0.04, ztop(1.0) + 0.04
    I.append(box(tag + "stile_h", (sh, sh, zt0 - 0.065), loc=(0, 0, (zt0 + 0.065) / 2), bevel=0.003))
    I.append(box(tag + "stile_m", (sm, sm, zt1 - 0.06), loc=(Lw, 0, (zt1 + 0.06) / 2), bevel=0.003))
    I.append(cylinder(tag + "heel_pin", 0.015, 0.045, loc=(0, 0, 0.05), verts=12))
    for zh in zhs:
        I.append(cylinder(tag + "journal", 0.03, 0.075, loc=(0, 0, zh), verts=16, bevel=0.003))
    fin_big = G["gilt"]
    fin_big.append(lathe(tag + "ball_fin", [(0.02, 0.0), (0.028, 0.006), (0.028, 0.014), (0.014, 0.024), (0.012, 0.034),
                                            (0.03, 0.05), (0.037, 0.07), (0.03, 0.09), (0.013, 0.104), (0.0, 0.108)],
                         segments=14, loc=(0, 0, zt0)))
    sp = _spear(tag + "spear_m", 10)
    place(sp, loc=(Lw, 0, zt1), scale=(1.3, 0.75, 1.3))
    fin_big.append(sp)

    # rails (flat bar, pierced for the bars)
    I.append(box(tag + "rail_b", (Li, 0.04, 0.028), loc=(xm, 0, zb), bevel=0.003))
    I.append(box(tag + "rail_d", (Li, 0.032, 0.016), loc=(xm, 0, zd), bevel=0.002))
    I.append(box(tag + "rail_l1", (Li, 0.04, 0.022), loc=(xm, 0, zl1), bevel=0.003))
    if band:
        I.append(box(tag + "rail_l2", (Li, 0.04, 0.022), loc=(xm, 0, zl2), bevel=0.003))
    ntop = 2 if p.top == "flat" else 18
    tp = [(x0 + Li * k / (ntop - 1), 0, ztop((x0 + Li * k / (ntop - 1)) / Lw)) for k in range(ntop)]
    I.append(sweep(tag + "rail_top", tp, rounded_rect(0.04, 0.026, 0.003, 1), up=(0, 0, 1), smooth_path=False))

    fz = 0.112 if orn == "scrolls" else 0.0
    if fz:
        I.append(sweep(tag + "rail_f", [(x, y, z - fz) for x, y, z in tp], _rect(0.034, 0.018), up=(0, 0, 1), smooth_path=False))

    # vertical bars
    nb = max(2, int(round(Li / 0.125)) - 1)
    spc = Li / (nb + 1)
    xs = [x0 + spc * (i + 1) for i in range(nb)]
    bar = box("bar", (0.018, 0.018, 1.0), bevel=0.002, segments=1)
    spear = _spear("spear", 6)
    BM, FM = [], []
    zlo = zl1 if orn == "scrolls" else zl2
    r = rng.fork(tag + "bars")
    for i, x in enumerate(xs):
        zt = ztop(x / Lw) + 0.035
        if zlo > zb:
            BM.append(trs((x, 0, (zb + zlo) / 2), (0, 0, 0), (1, 1, zlo - zb)))
        za, zc = zl2, zt
        tw = p.twist and i % 2 == 1 and (zc - za - fz) > 0.5
        if tw:
            t0, t1 = za + 0.1, zc - 0.14 - fz
            turns = max(1, round((t1 - t0) / 0.24))
            n = max(6, int((t1 - t0) / (0.045 * clamp(p.width * p.height / 7.5, 1.0, 1.7))))
            h = 0.009
            I.append(sweep(f"{tag}twist{i}", [(x, 0, t0 + (t1 - t0) * k / n) for k in range(n + 1)],
                           [(-h, -h), (h, -h), (h, h), (-h, h)], up=(0, 1, 0), smooth_path=False,
                           twist_deg=turns * 360.0, cap=False))
            BM.append(trs((x, 0, (za + t0) / 2), (0, 0, 0), (1, 1, t0 - za + 0.004)))
            BM.append(trs((x, 0, (t1 + zc) / 2), (0, 0, 0), (1, 1, zc - t1 + 0.004)))
        else:
            BM.append(trs((x, r.jitter(0, 0.001), (za + zc) / 2), (0, 0, r.jitter(0, 3)), (1, 1, zc - za)))
        hs = (1.0 if i % 2 == 0 else 0.72) * r.vary(1.0, 0.03)
        FM.append(trs((x, 0, zt), (r.jitter(0, 2.5), r.jitter(0, 2.5), r.jitter(0, 5)), (1.0, 0.55, hs)))
    I.append(instances_to_mesh(tag + "bars", bar, BM, keep_base=False))
    G["gilt"].append(instances_to_mesh(tag + "spears", spear, FM, keep_base=False))

    # dog bars with arrow heads
    xd = [(a + b) / 2 for a, b in zip([x0] + xs, xs + [x1])]
    dog = box("dog", (0.013, 0.013, 1.0), bevel=0.0015, segments=1)
    arrow = lathe("arrow", [(0.008, 0.0), (0.012, 0.004), (0.008, 0.01), (0.017, 0.03), (0.014, 0.05), (0.0, 0.085)], segments=6)
    ztip = zd + 0.055
    I.append(instances_to_mesh(tag + "dogs", dog, [trs((x, 0, (zb + ztip) / 2), (0, 0, 0), (1, 1, ztip - zb)) for x in xd],
                               keep_base=False))
    I.append(instances_to_mesh(tag + "arrows", arrow, [trs((x, 0, ztip), (r.jitter(0, 3), r.jitter(0, 3), 0), (1, 0.6, r.vary(1.0, 0.06))) for x in xd],
                               keep_base=False))

    if fz:
        rr = min((fz - 0.022) / 2, (spc - 0.018) / 2) - 0.0045
        I.append(instances_to_mesh(tag + "frings", _ring_base("fring", rr, 0.008, 0.018),
                                   [trs((x, 0, ztop(x / Lw) - fz / 2)) for x in xd], keep_base=False))

    # lock band infill
    zc_band = (zl1 + zl2) / 2
    lockw = 0.135
    CM, PM = [], []
    if orn == "scrolls":
        t = 0.0105
        zlo_c, zhi_c = zl1 + 0.011 + t / 2, zl2 - 0.011 - t / 2
        hb = zhi_c - zlo_c
        xa, xb = x0 + t / 2 + 0.001, x1 - lockw
        I.append(box(tag + "lock_bar", (0.018, 0.018, band), loc=(xb, 0, zc_band), bevel=0.002))
        xb -= 0.009 + t / 2 + 0.001
        ns = max(1, int(round((xb - xa) / (2.25 * hb))))
        cell = (xb - xa) / ns
        R = min(hb / 2, cell * 0.46)
        for i in range(ns):
            c0 = xa + i * cell
            up = 1 if i % 2 == 0 else -1
            A = (c0 + R, zc_band + up * R)
            B = (c0 + cell - R, zc_band - up * R)
            pts = _scroll(A, B, (1, 0), (1, 0), R, R, up, up, turns=1.45, n=9, k=0.24)
            I.append(_flat_scroll(f"{tag}s{i}", pts, t, 0.024, chamfer=False))
            PM += [trs((pts[0][0], 0, pts[0][1]), (90, 0, 0)), trs((pts[-1][0], 0, pts[-1][1]), (90, 0, 0))]
            CM.append(trs((c0 + R, 0, zc_band + up * (R + 0.002)), (0, 0, 0)))
            CM.append(trs((c0 + cell - R, 0, zc_band - up * (R + 0.002)), (0, 0, 0)))
            if i > 0:
                CM.append(trs((c0, 0, zc_band), (0, 90, 0)))
    elif orn == "rings":
        rr = min(band / 2, (spc - 0.018) / 2) - 0.004
        ring = _ring_base("ring", rr)
        I.append(instances_to_mesh(tag + "rings", ring, [trs((x, 0, zc_band)) for x in xd[:-1]], keep_base=False))
    if CM:
        collar = box("collar", (0.03, 0.03, 0.013), bevel=0.002)
        G["gilt"].append(instances_to_mesh(tag + "collars", collar, CM, keep_base=False))
        penny = cylinder("penny", 0.0105, 0.023, verts=10, bevel=0.002)
        I.append(instances_to_mesh(tag + "pennies", penny, PM, keep_base=False))

    # lock / keep / drop bolt at the meeting stile
    zlk = zc_band if orn == "scrolls" else zl1 - 0.1
    K = G["black"]
    if lock:
        K.append(box(tag + "lockbox", (0.105, 0.052, 0.15), loc=(x1 - 0.0525, 0, zlk), bevel=0.004))
        for sy in (-1, 1):
            G["brass"].append(cylinder(tag + "rose", 0.019, 0.006, loc=(x1 - 0.062, sy * 0.028, zlk + 0.03), rot=(90, 0, 0), verts=14))
            G["brass"].append(cylinder(tag + "escut", 0.012, 0.005, loc=(x1 - 0.04, sy * 0.028, zlk - 0.035), rot=(90, 0, 0), verts=10))
            hp = bezier_path((x1 - 0.062, sy * 0.03, zlk + 0.03), (x1 - 0.062, sy * 0.062, zlk + 0.03),
                             (x1 - 0.08, sy * 0.066, zlk + 0.03), (x1 - 0.165, sy * 0.062, zlk + 0.022), n=10)
            G["brass"].append(sweep(tag + "lever", hp, 0.007, sides=6, scale_fn=lambda u: 1.0 - 0.25 * u))
            G["brass"].append(sphere(tag + "lever_knob", 0.01, loc=(x1 - 0.168, sy * 0.062, zlk + 0.022), segments=10, rings=6))
    else:
        K.append(box(tag + "keep", (0.04, 0.052, 0.12), loc=(x1 - 0.02, 0, zlk), bevel=0.004))
        yb = inside * 0.03
        K.append(cylinder(tag + "dropbolt", 0.008, 0.6, loc=(Lw, yb, 0.345), verts=8))
        K.append(cylinder(tag + "dropbolt_h", 0.007, 0.06, loc=(Lw, yb + inside * 0.025, 0.63), rot=(90, 0, 0), verts=8))
        for z in (0.2, 0.52):
            K.append(box(tag + "staple", (0.034, 0.03, 0.022), loc=(Lw, yb, z), bevel=0.003))
    return G


# ------------------------------------------------------------------------------------------ build
def build(p, rng):
    W, H, pw = p.width, p.height, p.pier_width
    iron = mat("iron_painted", p.paint)
    black = mat("iron_wrought")
    brass = mat("brass", rough=0.8)
    gilt = mat("metal_plain", "#d4a84b", rough=0.5) if p.gilding else iron
    stone = mat("stone_rough" if p.pier_style == "rusticated" else "stone_smooth", p.stone)
    MATS = {"iron": iron, "gilt": gilt, "black": black, "brass": brass}
    rust = p.rust

    amp = clamp(0.11 * W, 0.14, 0.4) if p.top == "rising" else (-clamp(0.09 * W, 0.12, 0.3) if p.top == "dip" else 0.0)

    def ztop(t):
        return H + amp * (1 - (1 - t) ** 2)

    top_max = H + max(amp, 0.0) + 0.04 + 0.21
    hoff = 0.08
    Lw = W / 2 - hoff - 0.006 - 0.02
    zhs = [0.42, H - 0.28]
    zl = top_max + 0.06                         # overthrow lower lintel bar
    bh = 0.15
    zst = zl + bh + 0.2 if p.overthrow else H + 0.32   # top of the pier shaft
    zlatch = clamp(0.42 * H, 0.7, 1.0) + 0.1

    stones, bricks, fixed, fixed_gilt, fixed_black = [], [], [], [], []

    # ---- stone sill with heel cups and centre stop
    sill_h = 0.035
    nsl = max(1, int(round(W / 1.1)))
    rs = rng.fork("sill")
    for i in range(nsl):
        wx = W / nsl
        stones.append(stone_block(f"sill{i}", (wx - 0.004, 0.26, sill_h), loc=(-W / 2 + wx * (i + 0.5), 0, sill_h / 2 + 0.002),
                                  rng=rs, chamfer=0.008, rough=0.002, detail=0.3))
    fixed_black.append(box("stop", (0.13, 0.022, 0.05), loc=(0, -0.034, sill_h + 0.024), bevel=0.004))
    fixed_black.append(box("stop_plate", (0.16, 0.1, 0.008), loc=(0, 0.0, sill_h + 0.004), bevel=0.002))

    # ---- piers
    zcap = {}
    for sx in (-1, 1):
        r = rng.fork(f"pier{sx}")
        cx = sx * (W / 2 + pw / 2)
        ph = 0.3
        stones.append(stone_block(f"plinth{sx}", (pw + 0.08, pw + 0.08, ph), loc=(cx, 0, ph / 2 + 0.002), rng=r, chamfer=0.022, rough=0.004, detail=0.2))
        n = max(3, int(round((zst - ph) / 0.34)))
        ch = (zst - ph) / n
        if p.pier_style == "brick":
            b = box(f"brick{sx}", (pw - 0.012, pw - 0.012, zst - ph), loc=(cx, 0, (zst + ph) / 2), bevel=0.004)
            densify(b, 0.12)
            displace_noise(b, 0.0025, 0.25, seed=p.seed + sx)
            bricks.append(b)
            for zc in zhs + [zst - 0.11]:
                hh = 0.24 if zc < zst - 0.2 else 0.2
                zc = min(zc, zst - hh / 2)
                stones.append(stone_block(f"band{sx}", (pw + 0.012, pw + 0.012, hh), loc=(cx, 0, zc), rng=r, chamfer=0.01, detail=0.2))
        else:
            for i in range(n):
                zc = ph + ch * (i + 0.5)
                if p.pier_style == "rusticated":
                    e = 0.03 if i % 2 == 0 else 0.0
                    stones.append(stone_block(f"course{sx}_{i}", (pw + e, pw + e, ch - 0.004), loc=(cx, 0, zc), rng=r, chamfer=0.028, rough=0.006, detail=0.27))
                else:
                    f = r.u(0.38, 0.62)
                    if i % 2 == 0:
                        stones.append(stone_block(f"course{sx}_{i}a", (pw * f, pw, ch), loc=(cx - pw / 2 + pw * f / 2, 0, zc), rng=r, chamfer=0.007, rough=0.003, detail=0.2))
                        stones.append(stone_block(f"course{sx}_{i}b", (pw * (1 - f), pw, ch), loc=(cx + pw / 2 - pw * (1 - f) / 2, 0, zc), rng=r, chamfer=0.007, rough=0.003, detail=0.2))
                    else:
                        stones.append(stone_block(f"course{sx}_{i}a", (pw, pw * f, ch), loc=(cx, -pw / 2 + pw * f / 2, zc), rng=r, chamfer=0.007, rough=0.003, detail=0.2))
                        stones.append(stone_block(f"course{sx}_{i}b", (pw, pw * (1 - f), ch), loc=(cx, pw / 2 - pw * (1 - f) / 2, zc), rng=r, chamfer=0.007, rough=0.003, detail=0.2))
        # plinth moulding, necking astragal and moulded cornice with weathered top (square lathes)
        hw = pw / 2
        pm = [(hw + 0.04, -0.006), (hw + 0.04, 0.012), (hw + 0.03, 0.022), (hw + 0.012, 0.05), (hw + 0.008, 0.062), (hw - 0.004, 0.066)]
        pmo = lathe(f"plinthmould{sx}", [(a * SQ2, z) for a, z in pm], segments=4, loc=(cx, 0, ph), rot=(0, 0, 45), smooth_angle=20)
        bevel(pmo, 0.004, 2, angle=25)
        stones.append(pmo)
        nk = [(hw - 0.004, -0.1), (hw + 0.012, -0.094), (hw + 0.018, -0.08), (hw + 0.012, -0.066), (hw - 0.004, -0.06)]
        stones.append(lathe(f"necking{sx}", [(a * SQ2, z) for a, z in nk], segments=4, loc=(cx, 0, zst), rot=(0, 0, 45), smooth_angle=20))
        cp = [(hw - 0.01, -0.004), (hw + 0.014, 0.008), (hw + 0.014, 0.03), (hw + 0.026, 0.04), (hw + 0.05, 0.068), (hw + 0.07, 0.078),
              (hw + 0.07, 0.14), (hw + 0.052, 0.156), (hw * 0.62, 0.2), (hw * 0.62, 0.225), (0.0, 0.225)]
        cor = lathe(f"cornice{sx}", [(a * SQ2, z) for a, z in cp], segments=4, loc=(cx, 0, zst), rot=(0, 0, 45), smooth_angle=20)
        bevel(cor, 0.006, 2, angle=25)
        stones.append(cor)
        zc0 = zst + 0.225
        s = pw / 0.62
        if p.pier_cap == "ball":
            rb = 0.168 * s
            prof = [(0.135 * s, 0.0), (0.135 * s, 0.03 * s), (0.105 * s, 0.045 * s), (0.072 * s, 0.07 * s), (0.08 * s, 0.086 * s)]
            zc = 0.086 * s + rb * 0.94
            for k in range(15):
                a = radians(-68 + (158 * k / 14))
                prof.append((max(0.0, rb * cos(a)), zc + rb * sin(a)))
            prof[-1] = (0.0, zc + rb)
            stones.append(lathe(f"ball{sx}", prof, segments=28, loc=(cx, 0, zc0)))
            zcap[sx] = zc0 + zc + rb
        elif p.pier_cap == "urn":
            up = [(0.1, 0), (0.1, 0.03), (0.055, 0.045), (0.042, 0.075), (0.06, 0.09), (0.115, 0.125), (0.158, 0.19), (0.17, 0.255),
                  (0.15, 0.305), (0.108, 0.33), (0.122, 0.345), (0.126, 0.36), (0.08, 0.39), (0.035, 0.425), (0.04, 0.445),
                  (0.022, 0.47), (0.0, 0.485)]
            stones.append(lathe(f"urn{sx}", [(a * s, z * s) for a, z in up], segments=28, loc=(cx, 0, zc0)))
            zcap[sx] = zc0 + 0.485 * s
        else:
            lc = [(hw + 0.074, 0.128), (hw + 0.076, 0.142), (hw + 0.056, 0.16), (hw * 0.62 + 0.004, 0.204), (hw * 0.62 + 0.004, 0.229), (0.0, 0.229)]
            ld = lathe(f"leadcap{sx}", [(a * SQ2, z) for a, z in lc], segments=4, loc=(cx, 0, zst), rot=(0, 0, 45), smooth_angle=20)
            assign(ld, black)
            paint(ld, grime=0.45, special=0.15 + 0.3 * rust)
            st_h = 0.4
            fixed.append(lathe(f"lbase{sx}", [(0.11, 0.0), (0.11, 0.018), (0.066, 0.034), (0.042, 0.08), (0.03, 0.12), (0.027, st_h - 0.13),
                                              (0.04, st_h - 0.1), (0.044, st_h - 0.08), (0.03, st_h - 0.05), (0.05, st_h - 0.012),
                                              (0.05, st_h), (0.0, st_h)], segments=18, loc=(cx, 0, zc0 + 0.004)))
            lh = use("lantern_head", width=0.3, height=0.38, roof=p.lantern_roof, paint=p.paint, intensity=p.intensity * 0.6,
                     seed=p.seed + 5 + sx)
            lh.place((cx, 0, zc0 + st_h))
            zcap[sx] = zc0 + st_h + lh.size.z

        # hinge ironwork fixed to the pier: bolted plate, lug and strap eye round the stile journal
        hx = sx * (W / 2 - hoff)
        bt = bolt("hbolt", 0.009, 0.007, kind="dome")
        HB = []
        for zh in zhs:
            fixed.append(box(f"hplate{sx}", (0.014, 0.12, 0.2), loc=(sx * (W / 2 - 0.007), 0, zh), bevel=0.003))
            fixed.append(box(f"hlug{sx}", (hoff - 0.035, 0.036, 0.036), loc=(sx * (W / 2 - 0.012 - (hoff - 0.035) / 2), 0, zh), bevel=0.003))
            fixed.append(ring_band(f"hring{sx}", radius=0.036, width=0.042, thick=0.009, loc=(hx, 0, zh), segments=20))
            for dy in (-0.04, 0.04):
                for dz in (-0.07, 0.07):
                    HB.append(trs((sx * (W / 2 - 0.014), dy, zh + dz), (0, -90 * sx, 0)))
        fixed.append(instances_to_mesh(f"hbolts{sx}", bt, HB, keep_base=False))
        fixed_black.append(lathe(f"heelcup{sx}", [(0.05, 0.0), (0.05, 0.016), (0.04, 0.03), (0.022, 0.03), (0.022, 0.014), (0.0, 0.014)],
                                 segments=16, loc=(hx, 0, sill_h)))

    # ---- leaves
    for sx, tag in ((-1, "L_"), (1, "R_")):
        r = rng.fork("leaf" + tag)
        G = _leaf(p, tag, Lw, ztop, zhs, r, lock=(sx < 0), inside=(1 if sx < 0 else -1))
        ang = clamp(p.open_angle + r.u(-0.5, 0.8), 0.0, 90.0)
        sag = r.u(0.05, 0.22)
        hx = sx * (W / 2 - hoff)
        rz = ang if sx < 0 else 180.0 - ang
        for key, objs in G.items():
            if not objs:
                continue
            o = join(objs, f"leaf{tag}{key}")
            assign(o, MATS[key])
            if key in ("iron", "black"):
                paint(o, wear=lambda co, nr: 0.75 * np.exp(-((co[:, 0] - Lw + 0.04) / 0.1) ** 2) * np.exp(-((co[:, 2] - zlatch) / 0.2) ** 2))
            place(o, loc=(hx, 0, 0), rot=(0, sag, rz))
            apply_transform(o)
            if key in ("iron", "black"):
                paint(o, special=lambda co, nr: np.clip(np.maximum(height_ramp(0.0, 0.9)(co, nr) * (0.5 + 0.7 * rust),
                                                                   (0.15 + 0.45 * rust) * noise_mask(0.3, seed=p.seed + 13, lo=0.45, hi=0.75)(co, nr))
                                                        * noise_mask(0.06, seed=p.seed + 3, lo=0.25, hi=0.7)(co, nr), 0, 1),
                      grime=mask_mul(height_ramp(0.0, 0.35), 0.7))
            if key == "brass":
                paint(o, wear=0.6)
            # a leaf left standing open drifts a little on its hinges (vertical axis through the hinge line); a shut one is latched
            part(o, "leaf_l" if sx < 0 else "leaf_r", pivot=(hx, 0, 0), motion="swing" if ang > 3.0 else None, axis="Z",
                 amp=0.7, speed=0.17 if sx < 0 else 0.13)

    # ---- overthrow
    if p.overthrow:
        O, OG = fixed, fixed_gilt
        zs = zl + bh
        for z in (zl, zs):
            O.append(box("lintel", (W + 0.1, 0.045, 0.03), loc=(0, 0, z), bevel=0.003))
        for sx in (-1, 1):
            O.append(box(f"lplate{sx}", (0.014, 0.13, bh + 0.13), loc=(sx * (W / 2 - 0.007), 0, zl + bh / 2), bevel=0.003))
            bt = bolt("lbolt", 0.009, 0.007, kind="dome")
            O.append(instances_to_mesh(f"lbolts{sx}", bt, [trs((sx * (W / 2 - 0.014), dy, zl + bh / 2 + dz), (0, -90 * sx, 0))
                                                           for dy in (-0.045, 0.045) for dz in (-0.045, 0.045)], keep_base=False))
        # ring frieze in the lintel
        rr = (bh - 0.03) / 2
        nr = max(4, int(round((W - 0.03) / (2 * rr + 0.004))))
        spc = (W - 0.03) / nr
        rr = min(rr, spc / 2) - 0.004
        O.append(instances_to_mesh("frieze", _ring_base("fr", rr, 0.008, 0.022, 12),
                                   [trs((-(W - 0.03) / 2 + spc * (i + 0.5), 0, zl + bh / 2)) for i in range(nr)], keep_base=False))
        # arch: outer and inner bar with a bead of balls between them
        a = W / 2 - 0.02
        Rz = clamp(0.37 * W, 0.98, 1.2) if p.lantern else clamp(0.27 * W, 0.55, 0.95)
        d = 0.078
        ai, Ri = a - d, Rz - d
        na = 44
        O.append(sweep("arch", [(a * cos(pi * k / na), 0, zs + Rz * sin(pi * k / na)) for k in range(na + 1)],
                       _rect(0.03, 0.048), up=(0, 1, 0), smooth_path=False))
        O.append(sweep("arch_in", [(ai * cos(pi * k / na), 0, zs + Ri * sin(pi * k / na)) for k in range(na + 1)],
                       _rect(0.02, 0.04), up=(0, 1, 0), smooth_path=False))
        nbz = max(8, int(round(pi * sqrt((a * a + Rz * Rz) / 2) / 0.15)))
        bead = sphere("bead", d / 2 - 0.017, segments=8, rings=4)
        OG.append(instances_to_mesh("beads", bead, [trs(((a - d / 2) * cos(pi * (k + 0.5) / nbz), 0, zs + (Rz - d / 2) * sin(pi * (k + 0.5) / nbz)))
                                                    for k in range(nbz)], keep_base=False))

        def zin(x, m=0.0):
            return zs + (Ri - m) * sqrt(max(0.0, 1.0 - (x / (ai - m)) ** 2))

        def fits(xc, zc, rad, m=0.03):
            for k in range(16):
                th = 2 * pi * k / 16
                x, z = xc + rad * cos(th), zc + rad * sin(th)
                if abs(x) >= ai - m or z > zin(x, m):
                    return False
            return True

        # lantern hanging in a frame at the centre
        t = 0.012
        dep = 0.03
        zbase = zs + 0.015 + t / 2
        xf = 0.0
        zap = zs + Ri - 0.01
        if p.lantern:
            lw = clamp(0.3 * Rz, 0.28, 0.34)
            head = use("lantern_head", mount="hang", width=lw, height=lw * 1.25, roof=p.lantern_roof, paint=p.paint,
                       intensity=p.intensity, seed=p.seed + 1)
            drop = -head.min.z
            room = zap - zs - 0.06
            sc = min(1.0, (room - 0.06) / drop)
            zeye = zap - max(0.045, (room - drop * sc) * 0.4)
            head.place((0, 0, zeye), scale=sc)
            O.append(cylinder("lhook", 0.007, zap - zeye + 0.02, loc=(0, 0, (zap + zeye) / 2 + 0.006), verts=8))
            O.append(torus("leye", 0.016, 0.005, loc=(0, 0, zeye + 0.006), rot=(90, 0, 0), major_seg=14, minor_seg=6))
            xf = max(head.size.x, head.size.y) * sc / 2 + 0.08
            for sx in (-1, 1):
                zt = zin(xf) - 0.004
                O.append(box(f"frame{sx}", (0.022, 0.022, zt - zs), loc=(sx * xf, 0, (zt + zs) / 2), bevel=0.002))
                # little scrolled brackets tying the frame to the lintel
                br = _scroll((xf - 0.011 - t * 0.4, zs + 0.2), (xf - 0.1, zbase), (0, -1), (-0.3, -1), 0.0, 0.035, 1, 1, turns=1.1)
                O.append(_flat_scroll(f"fbr{sx}", [(sx * x, z) for x, z in br], 0.008, 0.02))
        else:
            O.append(box("kingbar", (0.022, 0.022, zap - zs), loc=(0, 0, (zap + zs) / 2), bevel=0.002))

        # infill each side
        CM, PM = [], []
        x_in = xf + 0.011 + t / 2 + 0.002
        if p.ornament == "plain":
            nv = max(1, int((ai - x_in) / 0.14))
            VB = []
            for sx in (-1, 1):
                for i in range(nv):
                    x = x_in + (ai - x_in) * (i + 1) / (nv + 1)
                    hz = zin(x) - zs
                    VB.append(trs((sx * x, 0, zs + hz / 2), (0, 0, 0), (1, 1, hz)))
            O.append(instances_to_mesh("obars", box("obar", (0.018, 0.018, 1.0), bevel=0.002, segments=1), VB, keep_base=False))
        else:
            # big S-scroll: large eye by the lantern frame, small eye towards the springing, then a ring
            r1 = min(0.3 * Rz, (zin(xf) - zs) * 0.36)
            while r1 > 0.05 and not fits(x_in + r1, zbase + r1, r1):
                r1 *= 0.95
            xc1 = x_in + r1
            r2 = r1 * 0.78
            while r2 > 0.04 and not fits(xc1 + r1 + r2 * 0.9, zbase + r2, r2):
                r2 *= 0.95
            xc2 = xc1 + r1 + r2 * 0.9
            r3 = r2 * 0.7
            while r3 > 0.03 and not fits(xc2 + r2 + r3 + 0.004, zbase + r3, r3, 0.014):
                r3 *= 0.95
            xc3 = xc2 + r2 + r3 + 0.004
            # C-scroll riding under the arch above the S
            mC = 0.01 + 0.003 + t / 2
            am, Rm = ai - mC, Ri - mC
            rA = min(0.17, 0.4 * (zin((x_in + xc1) / 2) - (zbase + 2 * r1)))
            xA, xB = x_in + rA, xc2 - 0.15 * r2
            Cp = None
            if rA > 0.04 and xB - xA > 0.15 and xB < am * 0.97:
                rB = clamp(0.4 * (zin(xB) - (zbase + 2 * r2)), 0.035, rA * 0.9)
                thA, thB = np.arccos(xA / am), np.arccos(xB / am)
                nsp = 12
                spine = [(am * cos(thA + (thB - thA) * k / nsp), zs + Rm * sin(thA + (thB - thA) * k / nsp)) for k in range(nsp + 1)]

                def tang(th):
                    tx, tz = am * sin(th), -Rm * cos(th)
                    L = hypot(tx, tz)
                    return (tx / L, tz / L)
                tA, tB = tang(thA), tang(thB)
                Cp = list(reversed(_spiral(spine[0], (-tA[0], -tA[1]), 1, rA, 1.35, 0.24))) + spine + _spiral(spine[-1], tB, -1, rB, 1.35, 0.24)
                thM = (thA + thB) / 2
            for sx in (-1, 1):
                def M(pts):
                    return [(sx * x, z) for x, z in pts]
                S = _scroll((xc1, zbase + 2 * r1), (xc2, zbase), (1, 0), (1, 0), r1, r2, 1, 1, turns=1.6, k=0.2, per_turn=20)
                O.append(_flat_scroll(f"oS{sx}", M(S), t, dep))
                PM += [trs((sx * S[0][0], 0, S[0][1]), (90, 0, 0)), trs((sx * S[-1][0], 0, S[-1][1]), (90, 0, 0))]
                CM.append(trs((sx * xc1, 0, zbase - t / 2 + 0.001)))
                CM.append(trs((sx * xc2, 0, zbase - t / 2 + 0.001)))
                CM.append(trs((sx * (x_in - 0.004), 0, zbase + r1), (0, 90, 0)))
                if r3 > 0.035:
                    ring = _ring_base(f"oR{sx}", r3 - t / 2, t, dep - 0.004, 20, chamfer=True)
                    place(ring, loc=(sx * xc3, 0, zbase + r3 - t / 2))
                    O.append(ring)
                    CM.append(trs((sx * xc3, 0, zbase - t / 2 + 0.001)))
                if Cp:
                    O.append(_flat_scroll(f"oC{sx}", M(Cp), t * 0.9, dep - 0.004))
                    PM += [trs((sx * Cp[0][0], 0, Cp[0][1]), (90, 0, 0)), trs((sx * Cp[-1][0], 0, Cp[-1][1]), (90, 0, 0))]
                    CM.append(trs((sx * (am + 0.006) * cos(thM), 0, zs + (Rm + 0.006) * sin(thM)), (0, sx * (degrees(thM) - 90), 0)))
        if CM:
            collar = box("ocollar", (0.036, dep + 0.01, 0.015), bevel=0.002, segments=1)
            OG.append(instances_to_mesh("ocollars", collar, CM, keep_base=False))
        if PM:
            penny = cylinder("openny", 0.015, dep + 0.004, verts=10, bevel=0.002)
            O.append(instances_to_mesh("opennies", penny, PM, keep_base=False))

        # crest on the apex: moulded base, painted shield with gilt rim and monogram, crown, supporter scrolls
        za = zs + Rz + 0.015
        O.append(lathe("crest_base", [(0.03, -0.01), (0.055, 0.0), (0.055, 0.012), (0.034, 0.022), (0.024, 0.04), (0.036, 0.05), (0.036, 0.06), (0.0, 0.06)],
                       segments=16, loc=(0, 0, za)))
        sw, shh = 0.31, 0.36
        z0 = za + 0.052
        sh_pts = [(0.0, 0.0)]
        for k in range(1, 9):
            th = radians(-90 + 90 * k / 8)
            sh_pts.append((sw / 2 * cos(th) ** 0.75, shh * 0.6 * (1 + sin(th))))
        sh_pts += [(sw / 2, shh), (sw * 0.2, shh - 0.02), (0.0, shh), (-sw * 0.2, shh - 0.02), (-sw / 2, shh)]
        for k in range(7, 0, -1):
            th = radians(-90 + 90 * k / 8)
            sh_pts.append((-sw / 2 * cos(th) ** 0.75, shh * 0.6 * (1 + sin(th))))
        shield = shape_extrude("shield", sh_pts, depth=0.016, loc=(0, 0.008, z0), rot=(90, 0, 0), bevel=0.0015)
        assign(shield, mat("iron_painted", p.crest_paint))
        paint(shield, special=mask_mul(noise_mask(0.08, seed=p.seed + 9), 0.3 * rust))
        OG.append(sweep("shield_rim", [(x, 0, z0 + z) for x, z in sh_pts], 0.0105, sides=8, closed=True, smooth_path=False, up=(0, 1, 0)))
        if p.monogram:
            for sy, rz in ((-1, 0), (1, 180)):
                OG.append(text(f"mono{sy}", str(p.monogram)[:3], size=shh * 0.44, depth=0.007, loc=(0, sy * 0.008, z0 + shh * 0.36),
                               rot=(90, 0, rz), font="serif"))
        # coronet in the round: moulded circlet, eight leaning points (alternately tall) with ball tips
        cr = 0.082
        zc = z0 + shh + 0.03
        O.append(box("crown_neck", (0.034, 0.014, 0.05), loc=(0, 0, zc - 0.012)))
        OG.append(lathe("circlet", [(0.0, 0.0), (cr - 0.012, 0.0), (cr, 0.005), (cr, 0.012), (cr - 0.007, 0.017), (cr - 0.007, 0.034), (cr + 0.004, 0.04),
                                    (cr + 0.004, 0.05), (cr - 0.012, 0.052), (cr - 0.02, 0.036), (0.0, 0.03)], segments=16, loc=(0, 0, zc)))
        spike = lathe("cspike", [(0.012, 0.0), (0.016, 0.022), (0.006, 0.07), (0.0, 0.082)], segments=6)
        cball = sphere("cball", 0.012, segments=8, rings=5)
        SM, CB = [], []
        tilt = 16.0
        for k in range(8):
            th = 2 * pi * (k + 0.5) / 8
            hs = 1.0 if k % 2 == 0 else 0.62
            bx, by = (cr - 0.006) * cos(th), (cr - 0.006) * sin(th)
            SM.append(trs((bx, by, zc + 0.046), (0, tilt, degrees(th)), (1, 1, hs)))
            L = 0.082 * hs + 0.008
            CB.append(trs((bx + L * sin(radians(tilt)) * cos(th), by + L * sin(radians(tilt)) * sin(th), zc + 0.046 + L * cos(radians(tilt)))))
        OG.append(instances_to_mesh("cspikes", spike, SM, keep_base=False))
        OG.append(instances_to_mesh("cballs", cball, CB, keep_base=False))
        for sx in (-1, 1):
            xs_ = min(0.5, a * 0.45)
            zs_ = zs + Rz * sqrt(1 - (xs_ / a) ** 2) + 0.015 + t / 2
            Sp = _scroll((xs_, zs_ + 0.075), (sw / 2 + 0.012 + t / 2, z0 + shh * 0.62), (-0.75, 0.66), (-0.25, 0.97), 0.075, 0.0, -1, 1, turns=1.45, k=0.22)
            O.append(_flat_scroll(f"supp{sx}", [(sx * x, z) for x, z in Sp], t * 0.85, dep - 0.006))
            OG.append(instances_to_mesh(f"suppc{sx}", box("sc", (0.015, dep + 0.006, 0.034), bevel=0.002, segments=1),
                                        [trs((sx * (sw / 2 + 0.01), 0, z0 + shh * 0.6))], keep_base=False))
            O.append(cylinder(f"suppp{sx}", 0.014, dep, loc=(sx * Sp[0][0], 0, Sp[0][1]), rot=(90, 0, 0), verts=10))

    # ---- materials and weathering
    for o in fixed:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.09, seed=p.seed + 6, lo=0.45, hi=0.8), 0.25 + 0.5 * rust))
    for o in fixed_gilt:
        assign(o, gilt)
    for o in fixed_black:
        assign(o, black)
        paint(o, special=0.3 + 0.5 * rust, grime=0.5)
    for o in bricks:
        assign(o, mat("brick_red"))
        paint(o, grime=mask_mul(height_ramp(0.0, 0.9), 0.8), special=mask_mul(height_ramp(0.2, 1.2), noise_mask(0.3, seed=p.seed + 8), 0.6))
    streak = zst - 0.02
    for o in stones:
        assign(o, stone)
        paint(o, grime=lambda co, nr: np.maximum(np.clip(1.0 - co[:, 2] / 0.5, 0, 1) * 0.55,
                                                  np.clip(1.0 - np.abs(co[:, 2] - streak + 0.16) / 0.2, 0, 1) * 0.22),
              special=mask_mul(facing((0, 0, 1)), noise_mask(0.25, seed=p.seed + 7, lo=0.35, hi=0.75), 0.5))

    # ---- sockets for adjoining walls / railings and the passage
    socket("wall_l", "attach", (-(W / 2 + pw), 0, 0), rot=(0, 0, -90), role="wall")
    socket("wall_r", "attach", (W / 2 + pw, 0, 0), rot=(0, 0, 90), role="wall")
    socket("passage", "attach", (0, 0, 0), role="door")
