"""Town notice board: two pegged oak posts carrying a boarded panel between mortised rails, a titled
header board and a small pitched cap (pent roof against a wall, gabled when freestanding; split
shingles or lead sheet with wood-cored rolls). The panel is covered in layered, tacked paper bills -
wanted posters, notices, playbills, tear-off notes - with printed ink, curled corners and torn edges.
A small oil lantern on a wrought bracket lights the notices at night. Freestanding boards sit on sleeper
feet with raking braces, wall boards are tied back with iron holdfasts.
"""
import random
from math import sin, cos, tan, asin, atan2, radians, degrees, pi, hypot, ceil

import bmesh
import bpy
import numpy as np
from tk import *

META = {"name": "Notice board with posters", "class": "prop", "mount": "wall_ground", "focus": 0.62,
        "mount_height": 1.4, "tri_budget": 22000}
PARAMS = {
    "width": {"default": 1.6, "min": 1.0, "max": 2.6, "desc": "overall width over the posts (m)"},
    "board_height": {"default": 1.0, "min": 0.7, "max": 1.3, "desc": "height of the boarded panel (m)"},
    "posters": {"default": 11, "min": 0, "max": 20, "desc": "number of pinned posters / notices (a freestanding board gets a third as many again on its back)"},
    "mount": {"default": "wall_ground", "choices": ["wall_ground", "ground", "wall"],
              "desc": "posts against a wall / freestanding on sleeper feet / hung on the wall"},
    "roof": {"default": True, "desc": "small pitched cap over the board"},
    "roofing": {"default": "shingle", "choices": ["shingle", "lead"], "desc": "covering of the cap"},
    "palette": {"default": 1, "min": 0, "max": 999, "desc": "poster palette seed (paper tints)"},
    "tatter": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "how curled / torn / stained the papers are"},
    "title": {"default": "NOTICES", "desc": "lettering on the header board"},
    "headlines": {"default": "WANTED,NOTICE,MARKET,LOST,FAIR,REWARD", "desc": "comma separated poster headlines"},
    "timber": {"default": "oak", "choices": ["oak", "painted", "weathered"], "desc": "timber finish"},
    "colour": {"default": "#c4a27c", "desc": "stain / paint colour of the timber"},
    "lantern": {"default": True, "desc": "oil lantern on a bracket lighting the board"},
    "intensity": {"default": 4.5, "min": 1.0, "max": 30.0, "desc": "lantern intensity (cd)"},
}
VARIANTS = {
    "freestanding_lead": {"mount": "ground", "width": 2.0, "roofing": "lead", "timber": "painted", "colour": "#3d5a47",
                          "posters": 14, "palette": 4, "title": "TOWN NOTICES", "seed": 7},
    "wall_small": {"mount": "wall", "width": 1.1, "board_height": 0.8, "roof": False, "posters": 5, "colour": "#a07a55",
                   "palette": 12, "tatter": 0.3, "title": "NEWS", "seed": 12},
    "tattered_wide": {"mount": "ground", "width": 2.5, "board_height": 1.15, "tatter": 1.0, "posters": 14,
                      "timber": "weathered", "colour": "#e6e0d4", "palette": 9, "title": "PUBLIC NOTICES", "seed": 23},
    "bare_painted": {"roof": False, "timber": "painted", "colour": "#35557c", "posters": 3, "tatter": 0.15,
                     "palette": 31, "width": 1.3, "title": "PARISH", "seed": 31},
}

PS = 0.10          # post section
RH = 0.08          # rail height
HH = 0.17          # header board height
INK = 0.001        # ink sits this far proud of the paper mid-surface


def _beam_yz(name, x, y0, z0, y1, z1, wx, t, bev=0.005):
    """Timber running from (y0, z0) to (y1, z1) in the plane x = const."""
    return plank(name, hypot(y1 - y0, z1 - z0), wx, t, loc=(x, (y0 + y1) / 2, (z0 + z1) / 2),
                 rot=(degrees(atan2(z1 - z0, y1 - y0)), 0, 0), axis='Y', bevel=bev)


def _fit_text(name, body, max_w, size, depth, font):
    t = text(name, body, size=size, depth=depth, font=font)
    mn, mx = world_bbox([t])
    if mx.x - mn.x > max_w:
        sc = max_w / (mx.x - mn.x)
        delete([t])
        t = text(name, body, size=size * sc, depth=depth, font=font)
        mn, mx = world_bbox([t])
    return t, mn, mx


def build(p, rng):
    mode = p.mount
    wallish = mode != "ground"
    hang = mode == "wall"
    W = float(p.width)
    bh = float(p.board_height)
    T = clamp(float(p.tatter))
    yc = -(PS / 2 + 0.035) if wallish else 0.0      # post centre line
    yf = yc - PS / 2                                # front face of posts and rails
    xp = W / 2 - PS / 2
    iw = W - 2 * PS
    zb0 = 0.88
    zb1 = zb0 + bh
    zmid = (zb0 + zb1) / 2
    zh0 = zb1 + RH
    zh1 = zh0 + HH
    zt = zh1 + 0.03                                 # post tops
    ybf = yc - 0.022                                # front face of the boarding

    wood = mat({"oak": "wood_oak", "painted": "wood_painted", "weathered": "wood_weathered"}[p.timber], p.colour)
    iron = mat("iron_wrought")
    cream = mat("wood_painted", "#2b2622" if p.timber == "weathered" else "#e9dfc4")   # painted cut-out letters
    stone = mat("stone_rough")
    timber, ironwork = [], []

    # ------------------------------------------------------------------ posts, feet, rails
    zpb = 0.14 if mode == "ground" else (0.07 if mode == "wall_ground" else zb0 - RH - 0.15)
    r_st = rng.fork("stone")
    for sx in (-1, 1):
        timber.append(plank("post", zt - zpb, PS, PS, loc=(sx * xp, yc, (zt + zpb) / 2), axis='Z', bevel=0.007))
        if mode == "ground":
            timber.append(plank("foot", 0.76, 0.11, 0.09, loc=(sx * xp, 0, 0.095), axis='Y', bevel=0.009))
            for sy in (-1, 1):
                pad = stone_block("pad", (0.21, 0.18, 0.05), loc=(sx * xp, sy * 0.27, 0.025), rng=r_st, chamfer=0.01, detail=0.11)
                assign(pad, stone)
                paint(pad, grime=0.5, special=mask_mul(noise_mask(0.12, seed=p.seed + 3), 0.8))
                timber.append(_beam_yz("brace", sx * xp, sy * 0.33, 0.135, sy * (PS / 2 - 0.012), 0.47, 0.06, 0.055))
        elif mode == "wall_ground":
            pad = stone_block("pad", (0.23, 0.165, 0.07), loc=(sx * xp, -0.09, 0.035), rng=r_st, chamfer=0.012, detail=0.11)
            assign(pad, stone)
            paint(pad, grime=0.5, special=mask_mul(noise_mask(0.12, seed=p.seed + 3), 0.8))
        else:   # hung on the wall: stiles end in a square drop
            timber.append(lathe("drop", [(0.0, -0.085), (0.03, -0.06), (0.024, -0.042), (0.06, -0.016), (0.074, -0.012),
                                         (0.074, 0.0)], segments=4, loc=(sx * xp, yc, zpb), rot=(0, 0, 45), smooth_angle=20))
        if not p.roof:
            timber.append(lathe("cap", [(0.072, 0.0), (0.08, 0.006), (0.08, 0.02), (0.066, 0.028), (0.066, 0.038),
                                        (0.0, 0.105)], segments=4, loc=(sx * xp, yc, zt), rot=(0, 0, 45), smooth_angle=20))

    for z in (zb0 - RH / 2, zb1 + RH / 2):
        timber.append(plank("rail", iw + 0.02, 0.07, RH, loc=(0, yf + 0.035, z), axis='X', bevel=0.005))
    peg = cylinder("peg", 0.0075, 0.01, verts=8)
    pm = [trs((sx * (xp + dx), yf - 0.001, z + dz), (90, 0, 0))
          for sx in (-1, 1) for z in (zb0 - RH / 2, zb1 + RH / 2) for dx, dz in ((-0.014, 0.018), (0.014, -0.018))]
    timber.append(instances_to_mesh("pegs", peg, pm, keep_base=False))

    # ------------------------------------------------------------------ boarding, beads, back ledges
    r_b = rng.fork("boards")
    nb = max(3, int(round(iw / 0.13)))
    bw = iw / nb
    boards, nails = [], []
    nail = bolt("nail", 0.0042, 0.0025, kind='pyramid')
    for i in range(nb):
        x = -iw / 2 + (i + 0.5) * bw
        dy = r_b.jitter(0, 0.0008)
        b = box("board", (bw - 0.0035, 0.02, bh + 0.02 - r_b.u(0, 0.004)), loc=(x, ybf + 0.01 + dy, zmid), bevel=0.002, segments=1)
        uv_grain(b, axis='Z')
        boards.append(b)
        for z in (zb0 + 0.06, zb1 - 0.06):
            nails.append(trs((x + r_b.jitter(0, bw * 0.2), ybf + dy, z + r_b.jitter(0, 0.008)), (90, r_b.u(0, 90), 0)))
    ironwork.append(instances_to_mesh("nails", nail, nails, keep_base=False))
    for z in (zb0 + 0.011, zb1 - 0.011):
        bd = box("bead", (iw, 0.014, 0.022), loc=(0, ybf - 0.007, z), bevel=0.004, segments=1)
        uv_grain(bd, axis='X')
        timber.append(bd)
    for sx in (-1, 1):
        bd = box("bead", (0.022, 0.014, bh - 0.044), loc=(sx * (iw / 2 - 0.011), ybf - 0.007, zmid), bevel=0.004, segments=1)
        uv_grain(bd, axis='Z')
        timber.append(bd)
    timber.append(plank("sill", iw - 0.004, 0.06, 0.022, loc=(0, yf - 0.012, zb0 - 0.004), rot=(-14, 0, 0), axis='X', bevel=0.004))
    if mode == "ground":                # double-sided: the back is beaded like the front
        for z in (zb0 + 0.011, zb1 - 0.011):
            bd = box("bead", (iw, 0.014, 0.022), loc=(0, ybf + 0.027, z), bevel=0.004, segments=1)
            uv_grain(bd, axis='X')
            timber.append(bd)
        for sx in (-1, 1):
            bd = box("bead", (0.022, 0.014, bh - 0.044), loc=(sx * (iw / 2 - 0.011), ybf + 0.027, zmid), bevel=0.004, segments=1)
            uv_grain(bd, axis='Z')
            timber.append(bd)
    else:                               # ledges nailed across the back of the boarding
        for k in (-1, 1):
            timber.append(plank("ledge", iw, 0.018, 0.07, loc=(0, ybf + 0.029, zmid + k * bh * 0.27), axis='X', bevel=0.003))

    # ------------------------------------------------------------------ header board and title
    hy = yf + 0.008                                   # header face, set back from the posts
    if p.roof:
        timber.append(plank("header", iw + 0.02, 0.035, HH, loc=(0, hy + 0.0175, zh0 + HH / 2), axis='X', bevel=0.004))
    else:
        hw = iw / 2 + 0.01
        pts = [(-hw, 0.0), (hw, 0.0)] + [(hw * cos(radians(a)), HH * 0.72 + 0.1 * sin(radians(a))) for a in range(0, 181, 12)]
        hd = extrude_profile("header", pts, 0.035, loc=(0, hy + 0.035, zh0), rot=(90, 0, 0), bevel=0.004)
        uv_grain(hd, axis='X')
        timber.append(hd)
    title = str(p.title).strip()
    if title:
        t, mn, mx = _fit_text("title", title, iw * 0.74, min(0.1, HH * 0.6), 0.005, "roman")
        bm = bmesh.new()                              # drop the hidden back faces of the letters
        bm.from_mesh(t.data)
        bmesh.ops.delete(bm, geom=[fc for fc in bm.faces if fc.normal.z < -0.5], context='FACES')
        bm.to_mesh(t.data)
        bm.free()
        place(t, loc=(-(mn.x + mx.x) / 2, hy, zh0 + HH / 2 + (0.0 if p.roof else 0.012) - (mn.z + mx.z) / 2))
        assign(t, cream)
        paint(t, wear=mask_mul(noise_mask(0.04, seed=p.seed + 9, lo=0.45, hi=0.7), 0.2 + 0.25 * T), grime=0.2)

    # ------------------------------------------------------------------ iron: corner plates, holdfasts
    L = [(-0.017, -0.017), (0.115, -0.017), (0.115, 0.017), (0.017, 0.017), (0.017, 0.115), (-0.017, 0.115)]
    nm = []
    for sx in (-1, 1):
        for sz, z in ((1, zb0 - RH / 2), (-1, zb1 + RH / 2)):
            pts = [(-sx * a, sz * b) for a, b in L]
            if -sx * sz < 0:
                pts = pts[::-1]
            ironwork.append(shape_extrude("corner_plate", pts, depth=0.003, loc=(sx * xp, yf, z), rot=(90, 0, 0), bevel=0.0008))
            for a, b in ((0.0, 0.0), (0.095, 0.0), (0.0, 0.095), (0.05, 0.0)):
                nm.append(trs((sx * xp - sx * a, yf - 0.003, z + sz * b), (90, rng.u(0, 90), 0), 1.5))
    nail2 = bolt("stud", 0.0042, 0.003, kind='pyramid')
    ironwork.append(instances_to_mesh("studs", nail2, nm, keep_base=False))
    if wallish:
        bar = rounded_rect(0.005, 0.032, 0.001, 1)
        bm_ = []
        hb = box("hf_bolt", (0.015, 0.015, 0.006), loc=(0, 0, 0.003), bevel=0.0012, segments=1)
        for sx in (-1, 1):
            xo = sx * (W / 2 + 0.0028)
            for z in (zb1 + RH / 2 + 0.07, zb0 - RH / 2 - (0.07 if not hang else 0.0)):
                path = [(xo, yc - 0.03, z), (xo, -0.022, z), (xo + sx * 0.004, -0.010, z), (xo + sx * 0.014, -0.0045, z), (xo + sx * 0.085, -0.0045, z)]
                ironwork.append(sweep("holdfast", path, bar, up=(0, 0, 1), smooth_path=False))
                bm_.append(trs((xo + sx * 0.0025, yc - 0.01, z), (0, sx * 90, 0)))
                bm_.append(trs((xo + sx * 0.058, -0.007, z), (90, 0, 0)))
        ironwork.append(instances_to_mesh("hf_bolts", hb, bm_, keep_base=False))

    # ------------------------------------------------------------------ pitched cap
    roof_objs = []
    if p.roof:
        a = radians(32.0 if wallish else 40.0)
        ta, ca, sa = tan(a), cos(a), sin(a)
        Lr = W + 0.30
        zr0 = zt + 0.06
        y_e = yf - 0.30 if wallish else -0.31
        y_top = -0.006 if wallish else 0.0
        run = y_top - y_e
        Ls = run / ca
        ze = zr0 - 0.02 + 0.06 / ca - 0.05 * ta
        zr = ze + run * ta
        ty0 = y_e + 0.05
        ty1 = -0.004 if wallish else -ty0
        sides = (1,) if wallish else (1, -1)
        rx = [-xp, xp] + ([0.0] if W > 1.75 else [])
        for sx in (-1, 1):
            timber.append(plank("tie", ty1 - ty0, 0.08, 0.06, loc=(sx * xp, (ty0 + ty1) / 2, zt + 0.03), axis='Y', bevel=0.006))
            if not wallish:
                timber.append(plank("king", zr - 0.085 - zr0, 0.05, 0.05, loc=(sx * xp, 0, (zr - 0.085 + zr0) / 2), axis='Z', bevel=0.004))
            # curved knee braces from the post up to the tie
            r = min(0.2, (yf - ty0) - 0.015)
            kp = [(0.0, -0.03)] + [(r - r * cos(radians(q)), r * sin(radians(q))) for q in range(10, 81, 10)] + [(r, r), (0.0, r)]
            for s in sides:
                kb = extrude_profile("knee", kp, 0.04, loc=(sx * xp + s * 0.02, s * yf, zt - r), rot=(90, 0, -90 * s), bevel=0.004, segments=1)
                timber.append(kb)
        for x in rx:
            for s in sides:
                ya, yb_ = y_e + 0.012, y_top - (0.05 if wallish else 0.012)
                timber.append(_beam_yz("rafter", x, s * (ya + 0.03 * sa), ze + (ya - y_e) * ta - 0.03 * ca,
                                       s * (yb_ + 0.03 * sa), ze + (yb_ - y_e) * ta - 0.03 * ca, 0.05, 0.06))
        if wallish:
            timber.append(plank("wall_plate", Lr - 0.08, 0.04, 0.07, loc=(0, -0.026, ze + (-0.046 - y_e) * ta - 0.037), axis='X', bevel=0.004))
        else:
            timber.append(plank("ridge", Lr - 0.08, 0.045, 0.07, loc=(0, 0, zr - 0.056), axis='X', bevel=0.004))

        shingle = p.roofing == "shingle"
        roofm = mat("wood_weathered", "#b9b2a6") if shingle else mat("lead")
        for s in sides:
            rs = rng.fork(f"slope{s}")
            sark, cover = [], []
            bwid = Ls / 3
            for k in range(3):
                sark.append(plank("sark", Lr - 0.04, bwid - 0.004, 0.016, loc=(rs.jitter(0, 0.006), (k + 0.5) * bwid, 0.008), axis='X', bevel=0.002))
            if shingle:
                nc = int(ceil(Ls / 0.112))
                e = Ls / nc
                th = 0.009
                tilt = degrees(asin(th / e))
                base = box("sh", (0.09, 0.2, th))
                uv_grain(base, axis='Y')
                realize(base)
                sm = []
                for c in range(nc):
                    butt = c * e - 0.022
                    ln = min(0.21, Ls - butt - 0.012)
                    x = -Lr / 2 - 0.012 - rs.u(0, 0.05)
                    while x < Lr / 2 + 0.012:
                        w = rs.u(0.062, 0.125)
                        x0, x1 = max(x, -Lr / 2 - 0.012), min(x + w, Lr / 2 + 0.012)
                        if x1 - x0 > 0.03 and not (T > 0.6 and c > 0 and rs.chance(0.04 * T)):
                            l2 = ln * rs.u(0.95, 1.0)
                            bt = butt + rs.jitter(0, 0.006)
                            tk_ = th * rs.u(0.8, 1.15)
                            zc = 0.0165 + tk_ / 2 + 0.5 * l2 * sin(radians(tilt))
                            sm.append(trs(((x0 + x1) / 2, bt + l2 / 2, zc), (-tilt + rs.jitter(0, 0.8), rs.jitter(0, 1.0), rs.jitter(0, 1.5)),
                                          ((x1 - x0 - 0.003) / 0.09, l2 / 0.2, tk_ / th)))
                        x += w
                cover.append(instances_to_mesh("shingles", base, sm, keep_base=False, seed=p.seed + s))
                cover.append(plank("ridge_board", Lr + 0.03, 0.105, 0.016, loc=(0, Ls - 0.0535, 0.047), rot=(-3, 0, 0), axis='X', bevel=0.003))
            else:
                sh = box("lead_sheet", (Lr + 0.012, Ls + 0.02, 0.004), loc=(0, Ls / 2 - 0.01, 0.0185))
                densify(sh, 0.13)
                displace_noise(sh, 0.0012, 0.14, seed=p.seed + s + 5, axis_mask=(0, 0, 1))
                cover.append(sh)
                cover.append(box("lead_drip", (Lr + 0.016, 0.004, 0.034), loc=(0, -0.0205, 0.004), bevel=0.0012))
                for sx in (-1, 1):
                    cover.append(box("lead_verge", (0.004, Ls + 0.02, 0.034), loc=(sx * (Lr / 2 + 0.007), Ls / 2 - 0.01, 0.004), bevel=0.0012))
                nr = max(2, int(round(Lr / 0.55))) + 1
                for k in range(nr):
                    x = -Lr / 2 + 0.07 + (Lr - 0.14) * k / (nr - 1)
                    rl = cylinder("lead_roll", 0.019, Ls + 0.012, loc=(x, Ls / 2 - 0.012, 0.021), rot=(90, 0, 0), verts=10)
                    smooth(rl, 50)
                    cover.append(rl)
                cover.append(box("lead_apron", (Lr + 0.016, 0.11, 0.004), loc=(0, Ls - 0.055, 0.0235), bevel=0.0012))
            rot = (degrees(a), 0, 0 if s > 0 else 180)
            for group, m, nm_ in ((sark, wood, "sarking"), (cover, roofm, "roofing")):
                for o in group:
                    assign(o, m)
                g = join(group, f"{nm_}{s}")
                place(g, loc=(0, s * y_e, ze), rot=rot)
                (timber if m is wood else roof_objs).append(g)
        if not shingle:
            if wallish:
                roof_objs.append(box("lead_upstand", (Lr + 0.016, 0.004, 0.11), loc=(0, -0.0045, zr + 0.045), bevel=0.0012))
            else:
                rr = cylinder("lead_ridge", 0.03, Lr + 0.03, loc=(0, 0, zr + 0.03), rot=(0, 90, 0), verts=12)
                smooth(rr, 50)
                roof_objs.append(rr)
        for o in roof_objs:
            assign(o, roofm)
            if shingle:
                paint(o, special=mask_mul(noise_mask(0.16, seed=p.seed + 11, lo=0.45, hi=0.8), 0.35 + 0.45 * T),
                      grime=mask_mul(noise_mask(0.6, seed=p.seed + 12, lo=0.3, hi=0.9), 0.3))
            else:
                paint(o, grime=mask_mul(noise_mask(0.3, seed=p.seed + 12), 0.45), wear=mask_mul(noise_mask(0.1, seed=p.seed + 13, lo=0.5, hi=0.8), 0.4))

    # ------------------------------------------------------------------ lantern on a wrought bracket
    if p.lantern:
        fb = [(-0.0035, -0.011), (0.0035, -0.011), (0.0035, 0.011), (-0.0035, 0.011)]
        fs = [(-0.0035, -0.006), (0.0035, -0.006), (0.0035, 0.006), (-0.0035, 0.006)]
        bb = cylinder("bk_bolt", 0.0075, 0.006, loc=(0, 0, 0.003), verts=6)
        ey = yf - 0.45
        if p.roof:
            # flat bar bolted along the inner face of the tie beam, reaching out past the eave
            zA = zt + 0.03
            xa = xp - 0.044
            ex = xa - 0.05
            ironwork.append(sweep("arm", [(xa, yf - 0.02, zA), (xa, y_e + 0.03, zA), (xa - 0.012, y_e - 0.02, zA), (ex, ey, zA)], fb, up=(0, 0, 1), smooth_path=False))
            sc = spiral_path(0.006, 0.034, 1.35, n=18, a0_deg=-90, plane="YZ", center=(ex + 0.022, ey + 0.075, zA - 0.046))
            ironwork.append(sweep("scroll", sc, fs, up=(1, 0, 0), scale_fn=lambda q: 0.55 + 0.45 * q))
            ironwork.append(instances_to_mesh("bk_bolts", bb, [trs((xa - 0.0035, yf - 0.07, zA), (0, -90, 0)), trs((xa - 0.0035, y_e + 0.085, zA), (0, -90, 0))], keep_base=False))
        else:
            # scrolled bracket on the face of the post
            zA = zt - 0.07
            ex = xp - 0.09
            ironwork.append(sweep("arm", [(xp, yf, zA), (ex, ey, zA)], fb, up=(0, 0, 1), smooth_path=False))
            t6 = 0.62
            ax_, ay_ = xp + (ex - xp) * t6, yf + (ey - yf) * t6
            stay = bezier_path((xp, yf - 0.004, zA - 0.21), (xp - 0.01, yf - 0.15, zA - 0.21), (ax_ + 0.005, ay_ + 0.03, zA - 0.12), (ax_, ay_, zA - 0.012), n=9)
            ironwork.append(sweep("stay", stay, fb, up=(0, 0, 1)))
            sc = spiral_path(0.006, 0.034, 1.35, n=18, a0_deg=-90, plane="YZ", center=(xp - 0.02, yf - 0.1, zA - 0.07))
            ironwork.append(sweep("scroll", sc, fs, up=(1, 0, 0), scale_fn=lambda q: 0.55 + 0.45 * q))
            ironwork.append(box("back_plate", (0.038, 0.005, 0.3), loc=(xp, yf - 0.0025, zA - 0.1), bevel=0.0012))
            ironwork.append(instances_to_mesh("bk_bolts", bb, [trs((xp, yf - 0.005, zA + 0.03), (90, 0, 0)), trs((xp, yf - 0.005, zA - 0.235), (90, 12, 0))], keep_base=False))
        ironwork.append(sphere("arm_ball", 0.011, loc=(ex - 0.002, ey - 0.011, zA), segments=10, rings=6))
        drop = 0.075 if p.roof else 0.0                 # short chain keeps the flame clear of the eave
        zl = zA - 0.03 - drop                           # lantern top ring
        ironwork.append(cylinder("hook", 0.0028, 0.03, loc=(ex, ey, zA - 0.02), verts=6))
        if drop:
            ironwork.append(chain("l_chain", [(ex, ey, zA - 0.03), (ex, ey, zl - 0.004)], link_len=0.024, wire=0.0024, sides=4))
        ironwork.append(torus("ring", 0.011, 0.0026, loc=(ex, ey, zl - 0.008), rot=(90, 0, 30), major_seg=12, minor_seg=5))
        ironwork.append(lathe("l_cap", [(0.005, -0.02), (0.012, -0.024), (0.014, -0.036), (0.022, -0.042), (0.05, -0.062), (0.054, -0.067), (0.046, -0.069)],
                              segments=12, loc=(ex, ey, zl)))
        ironwork.append(lathe("l_base", [(0.0, -0.186), (0.026, -0.186), (0.044, -0.178), (0.047, -0.168), (0.04, -0.164)], segments=12, loc=(ex, ey, zl)))
        rod = cylinder("l_rod", 0.0022, 0.1, verts=5)
        ironwork.append(instances_to_mesh("l_rods", rod, [trs((ex + 0.044 * cos(radians(q)), ey + 0.044 * sin(radians(q)), zl - 0.117)) for q in (45, 135, 225, 315)], keep_base=False))
        globe = lathe("l_glass", [(0.036, -0.165), (0.041, -0.13), (0.039, -0.095), (0.034, -0.068)], segments=12, loc=(ex, ey, zl))
        assign(globe, mat("glass_frosted", "#ffd08a", glow=2.6))
        light("lantern", (ex, ey, zl - 0.118), color="#ffbf6a", intensity=float(p.intensity), range=5.0, flicker=0.2, shadow_near=0.085)

    # ------------------------------------------------------------------ posters
    pr = random.Random(1000 + int(p.palette))
    tints = [pr.choice(["#eadfc6", "#dccba4", "#e3e0d2"])] + pr.sample(
        ["#c98f7e", "#93adb6", "#d6b55c", "#a8b287", "#b8604a", "#c7a6b8", "#8fa3c2", "#d99a55"], 2)
    papers = [mat("paper", c) for c in tints]
    inkm = mat("paper", "#2a2019")
    rp = rng.fork("posters")
    ax = iw / 2 - 0.036
    az0, az1 = zb0 + 0.036, zb1 - 0.036
    KINDS = [(("wanted", 0.30, 0.42), 3), (("notice", 0.26, 0.36), 3), (("bill", 0.22, 0.46), 2), (("land", 0.40, 0.27), 2),
             (("tabs", 0.21, 0.28), 1.5), (("note", 0.15, 0.11), 2), (("note", 0.13, 0.17), 2), (("notice", 0.21, 0.29), 2.5)]
    KINDS = [k for k in KINDS if k[0][1] < 2 * ax - 0.03 and k[0][2] < az1 - az0 - 0.03]
    heads = [s.strip() for s in str(p.headlines).split(",") if s.strip()]
    n_text = 0
    iv, ifc, tacks = [], [], []
    sheets = [[], [], []]
    scraps = []

    def pin_up(count, rp, back):
        """Lay out and model `count` papers on the front (or, mirrored, on the back) of the boarding."""
        nonlocal n_text
        ymir = 2 * (ybf + 0.01)

        def xf(c):
            return (-c[0], ymir - c[1], c[2]) if back else c

        chosen = [rp.weighted(KINDS) for _ in range(count)] if KINDS else []
        chosen.sort(key=lambda k: -(k[1] * k[2]) * rp.u(0.6, 1.4))
        placed = []
        for kind, w, h in chosen:
            w *= rp.u(0.93, 1.05)
            h *= rp.u(0.93, 1.05)
            th = radians(rp.jitter(0, (5 + 8 * T) if kind == "note" else (1.5 + 3.5 * T)))
            hx = w / 2 * cos(th) + h / 2 * abs(sin(th))
            hz = h / 2 * cos(th) + w / 2 * abs(sin(th))
            best = None
            for _ in range(22):
                cx = rp.u(-ax + hx, ax - hx)
                cz = rp.u(az0 + hz, az1 - hz)
                sc = 9.0
                for q in placed:
                    sc = min(sc, max(abs(cx - q["cx"]) / (hx + q["hx"]), abs(cz - q["cz"]) / (hz + q["hz"])))
                sc = min(sc, 0.93) + rp.u(0, 0.02)
                if best is None or sc > best[0]:
                    best = (sc, cx, cz)
            _, cx, cz = best
            layer = 0
            for q in placed:
                if abs(cx - q["cx"]) < hx + q["hx"] and abs(cz - q["cz"]) < hz + q["hz"]:
                    layer = max(layer, q["layer"] + 1)
            placed.append(dict(kind=kind, w=w, h=h, th=th, hx=hx, hz=hz, cx=cx, cz=cz, layer=layer))

        for i, q in enumerate(placed):
            kind, w, h, th, cx, cz = q["kind"], q["w"], q["h"], q["th"], q["cx"], q["cz"]
            ct, st = cos(th), sin(th)
            yb = ybf - 0.0032 - q["layer"] * 0.003
            later = placed[i + 1:]

            def covered(lx, lz, _l=later, _ct=ct, _st=st, _cx=cx, _cz=cz):
                X, Z = _cx + lx * _ct + lz * _st, _cz - lx * _st + lz * _ct
                return any(abs(X - o["cx"]) < o["hx"] + 0.03 and abs(Z - o["cz"]) < o["hz"] + 0.03 for o in _l)

            small = kind == "note"
            nu, nv = (4, 4) if small else ((5, 5) if back else (6, 7))
            calm = False
            style = rp.weighted([("one", 3), ("top2", 2)]) if small else rp.weighted([("four", 3.2 - 2.4 * T), ("top2", 0.8 + T), ("three", 1 + 2 * T)])
            pinned = {(-1, -1): True, (1, -1): True, (1, 1): True, (-1, 1): True}
            if style == "one":
                pinned = {k: False for k in pinned}
            elif style == "top2":
                pinned[(-1, -1)] = pinned[(1, -1)] = False
            elif style == "three":
                pinned[(rp.pick([-1, 1]), 1 if rp.chance(0.25 * T) else -1)] = False
            # tearing
            tear = None
            if not small and kind != "tabs" and rp.chance(0.1 + 0.5 * T):
                nu, nv = (8, 5) if back else (10, 7)
                if rp.chance(0.6):
                    tear = ("corner", rp.pick([-1, 1]), rp.u(0.3, 0.75), rp.u(0.15, 0.42))
                    pinned[(tear[1], -1)] = None
                else:
                    tear = ("bottom", rp.u(0, 6.28), rp.u(1.5, 4.0), rp.u(0.12, 0.3))
                    pinned[(-1, -1)] = pinned[(1, -1)] = None
            xs = [-w / 2 + w * k / nu for k in range(nu + 1)]
            zbot = []
            for x in xs:
                tz = 0.0
                if tear and tear[0] == "corner":
                    tz = h * tear[3] * max(0.0, 1.0 - ((x * tear[1] + w / 2) / w) / tear[2])
                elif tear:
                    tz = h * tear[3] * (0.75 + 0.25 * sin(tear[1] + tear[2] * x / w * 6.28))
                if tz > 0.004:
                    tz = max(0.0, tz + rp.jitter(0, 0.009))
                zbot.append(-h / 2 + tz)
            # tear-off tabs
            tab = None
            if kind == "tabs":
                ntab = 6
                tabh = h * 0.3
                gone = [rp.chance(0.3 + 0.4 * T) for _ in range(ntab)]
                gone[rp.pick(range(ntab))] = True
                gone[rp.pick([k for k in range(ntab) if not gone[k]] or [0])] = False
                tl = [rp.u(0.0, 0.01 + 0.022 * T) for _ in range(ntab)]
                xs, zbot = [-w / 2], [-h / 2 + (tabh if gone[0] else 0.0)]
                for k in range(1, ntab):
                    c = -w / 2 + w * k / ntab
                    xs += [c - 0.0012, c + 0.0012]
                    zbot += [-h / 2 + (tabh if gone[k - 1] else 0.0), -h / 2 + (tabh if gone[k] else 0.0)]
                xs.append(w / 2)
                zbot.append(-h / 2 + (tabh if gone[-1] else 0.0))
                nu = len(xs) - 1
                nv = 10
                tab = (ntab, tabh, tl)
                pinned[(-1, -1)] = pinned[(1, -1)] = None
            curls = []
            for (kx, kz), pin in pinned.items():
                if pin is False and not covered(kx * w / 2, kz * h / 2) and not covered(kx * w * 0.3, kz * h * 0.3):
                    curls.append((kx, kz, rp.u(0.4, 1.0) * (0.008 + 0.034 * T), rp.u(0.28, 0.5) * min(w, h)))
            lift = 0.0
            if style in ("top2", "one") or (tear and tear[0] == "bottom"):
                lift = rp.u(0.3, 1.0) * (0.004 + 0.02 * T)
                if covered(0, -h * 0.4) or covered(-w * 0.4, -h * 0.45) or covered(w * 0.4, -h * 0.45):
                    lift *= 0.08
            if tear is None and tab is None and not curls and lift < 0.003:
                calm = True
                nu, nv = (2, 2) if small else (3, 4)
                xs = [-w / 2 + w * k / nu for k in range(nu + 1)]
                zbot = [-h / 2] * (nu + 1)
            bulge = rp.u(0.0005, 0.0016)
            wr = 0.0008 * T
            ph1, ph2 = rp.u(0, 6.28), rp.u(0, 6.28)

            def world_raw(x, z, extra=0.0, _w=w, _h=h, _c=curls, _l=lift, _b=bulge, _wr=wr, _p1=ph1, _p2=ph2, _tab=tab,
                      _ct=ct, _st=st, _cx=cx, _cz=cz, _yb=yb):
                u, v = clamp(x / _w + 0.5), clamp(z / _h + 0.5)
                d = _b * sin(pi * u) * sin(pi * v) + _wr * sin(7.3 * u + _p1) * sin(5.1 * v + _p2) + _l * (1.0 - v) ** 2.5
                ox = oz = 0.0
                for kx, kz, s, r in _c:
                    t = max(0.0, 1.0 - hypot(x - kx * _w / 2, z - kz * _h / 2) / r)
                    if t > 0:
                        d += s * t * t
                        ox -= kx * s * 0.7 * t * t
                        oz -= kz * s * 0.7 * t * t
                if _tab:
                    ztop = -_h / 2 + _tab[1]
                    if z < ztop:
                        k = min(_tab[0] - 1, max(0, int((x / _w + 0.5) * _tab[0])))
                        d += _tab[2][k] * ((ztop - z) / _tab[1]) ** 2
                x2, z2 = x + ox, z + oz
                return (_cx + x2 * _ct + z2 * _st, _yb - d - extra, _cz - x2 * _st + z2 * _ct)

            def world(x, z, extra=0.0, _f=world_raw):
                return xf(_f(x, z, extra))

            def fn(u, v, _xs=xs, _zb=zbot, _nu=nu, _h=h, _world=world):
                k = int(round(u * _nu))
                z0 = _zb[k]
                return _world(_xs[k], z0 + (_h / 2 - z0) * v)

            sheet = surface("poster", fn, nu, nv, thickness=0.0 if calm else 0.0006)
            pi_ = rp.weighted([(0, 4.6), (1, 2.7), (2, 2.7)])
            assign(sheet, papers[pi_])
            paint(sheet, grime=mask_mul(noise_mask(0.1, seed=p.seed + i * 3, lo=0.4, hi=0.8), 0.25 + 0.55 * T),
                  special=mask_mul(noise_mask(0.16, seed=p.seed + 40 + i * 3, lo=0.45, hi=0.75), 0.25 + 0.7 * T))
            sheets[pi_].append(sheet)

            def bot(x, _xs=xs, _zb=zbot):
                return float(np.interp(x, _xs, _zb))

            def rect(x0, x1, z0, z1):
                n = 1 if calm else max(1, int(ceil((x1 - x0) / 0.07)))
                for k in range(n):
                    xa, xb = x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n
                    if z0 < bot(xa) + 0.012 or z0 < bot(xb) + 0.012:
                        continue
                    b = len(iv)
                    iv.extend([world(xa, z0, INK), world(xb, z0, INK), world(xb, z1, INK), world(xa, z1, INK)])
                    ifc.append((b, b + 1, b + 2, b + 3))

            def line(x0, x1, zc, t=0.0045, words=True):
                if not words:
                    rect(x0, x1, zc - t / 2, zc + t / 2)
                    return
                x = x0
                while x < x1 - 0.008:
                    wl = min(rp.u(0.012, 0.05), x1 - x)
                    rect(x, x + wl, zc - t / 2, zc + t / 2)
                    x += wl + rp.u(0.005, 0.009)

            m = 0.085 * w + 0.008
            lft, rgt, top = -w / 2 + m, w / 2 - m, h / 2 - m
            low = -h / 2 + m + (tab[1] if tab else 0.0)

            def headline(size, font):
                nonlocal n_text
                if heads and n_text < 6:
                    body = heads[n_text % len(heads)]
                    n_text += 1
                    t, mn, mx = _fit_text("headline", body, rgt - lft, size, 0.0, font)
                    hgt = mx.z - mn.z
                    lx, lz = -(mn.x + mx.x) / 2, top - hgt / 2 - (mn.z + mx.z) / 2
                    o = world_raw(0.0, top - hgt / 2)
                    place(t, loc=xf((cx + lx * ct + lz * st, o[1] - 0.0005, cz - lx * st + lz * ct)), rot=(90, degrees(th), 180 if back else 0))
                    assign(t, inkm)
                    return top - hgt - 0.014
                rect(lft, rgt, top - size * 0.8, top)
                return top - size * 0.8 - 0.014

            if kind == "wanted":
                z = headline(0.058, "roman")
                line(lft, rgt, z, 0.003, False)
                pw, ph = (rgt - lft) * 0.62, h * 0.3
                z1 = z - 0.012
                z0 = z1 - ph
                for a, b in ((z0, z0 + 0.004), (z1 - 0.004, z1)):
                    rect(-pw / 2, pw / 2, a, b)
                rect(-pw / 2, -pw / 2 + 0.004, z0, z1)
                rect(pw / 2 - 0.004, pw / 2, z0, z1)
                rect(-pw * 0.36, pw * 0.36, z0 + 0.004, z0 + ph * 0.2)          # shoulders
                rect(-pw * 0.27, pw * 0.27, z0 + ph * 0.2, z0 + ph * 0.3)
                rect(-pw * 0.09, pw * 0.09, z0 + ph * 0.3, z0 + ph * 0.4)       # neck
                for f, a, b in ((0.14, 0.4, 0.48), (0.19, 0.48, 0.7), (0.15, 0.7, 0.8), (0.24, 0.8, 0.84), (0.13, 0.84, 0.9)):
                    rect(-pw * f, pw * f, z0 + ph * a, z0 + ph * b)             # head and hat
                z = z0 - 0.02
                line(lft, rgt, z)
                z -= 0.026
                if z - 0.012 > low:
                    rect(-(rgt - lft) * 0.34, (rgt - lft) * 0.34, z - 0.011, z + 0.011)
                    z -= 0.028
                while z > low:
                    line(lft + 0.01, rgt - 0.01, z)
                    z -= 0.015
            elif kind == "notice":
                z = headline(0.036, "serif") - 0.004
                line(lft, rgt, z + 0.008, 0.0025, False)
                z -= 0.008
                left_in_par = int(rp.u(3, 6))
                while z > low + 0.03:
                    last = left_in_par == 1
                    line(lft, rgt if not last else lft + (rgt - lft) * rp.u(0.3, 0.8), z)
                    z -= 0.0145
                    left_in_par -= 1
                    if left_in_par == 0:
                        left_in_par = int(rp.u(3, 6))
                        z -= 0.01
                line(rgt - (rgt - lft) * 0.38, rgt, low + 0.006, 0.007, False)
            elif kind == "bill":
                z = headline(0.05, "sans_narrow")
                line(lft, rgt, z + 0.004, 0.005, False)
                z -= 0.016
                seq = [(0.016, 0.8), (0.005, 0.5), (0.005, 0.62), (0.024, 0.95), (0.005, 0.45), (0.012, 0.7), (0.005, 0.55), (0.005, 0.6),
                       (0.018, 0.85), (0.005, 0.4), (0.005, 0.66), (0.012, 0.75), (0.005, 0.5)]
                for t, f in seq:
                    if z - t < low + 0.012:
                        break
                    hwid = (rgt - lft) * f / 2
                    line(-hwid, hwid, z - t / 2, t, t < 0.008)
                    z -= t + 0.011
                line(lft, rgt, low + 0.003, 0.005, False)
            elif kind == "land":
                z = headline(0.04, "serif")
                line(lft, rgt, z + 0.006, 0.003, False)
                z -= 0.008
                mid = (rgt - lft) * 0.03
                rect(lft, -mid - 0.02, low, z)                                   # woodcut block
                while z > low:
                    line(mid, rgt, z)
                    z -= 0.0145
            elif kind == "tabs":
                z = headline(0.03, "sans")
                while z > low + 0.004:
                    line(lft, rgt, z)
                    z -= 0.0145
                for k in range(tab[0]):
                    if not gone[k]:
                        c = -w / 2 + w * (k + 0.5) / tab[0]
                        rect(c - 0.0035, c + 0.0035, -h / 2 + 0.014, -h / 2 + tab[1] - 0.012)
            else:
                z = top - 0.004
                while z > low:
                    line(lft, lft + (rgt - lft) * rp.u(0.55, 1.0), z, 0.004)
                    z -= 0.017

            def tack(lx, lz):
                o = world(lx, lz, 0.0004)
                tacks.append(trs(o, (-90 if back else 90, rp.u(0, 360), 0), rp.u(0.85, 1.25)))

            if style == "one":
                tack(rp.jitter(0, 0.01), h / 2 - 0.014)
            for (kx, kz), pin in pinned.items():
                if pin:
                    tack(kx * (w / 2 - 0.013 - rp.u(0, 0.006)), kz * (h / 2 - 0.013 - rp.u(0, 0.006)))

        # stray tacks and pinned scraps of older notices
        for k in range(int(round((1 + 5 * T) * (0.5 if back else 1.0))) if count > 0 else 0):
            sx_, sz_ = rp.u(-ax + 0.03, ax - 0.03), rp.u(az0 + 0.06, az1 - 0.03)
            if rp.chance(0.7):
                s = rp.u(0.025, 0.055)
                a0, sk = radians(rp.jitter(0, 35)), rp.u(-0.4, 0.4)

                def sfn(u, v, _s=s, _a=a0, _sk=sk, _x=sx_, _z=sz_):
                    lx = (u - 0.5) * _s * (1.0 - 0.75 * v) + _sk * _s * v * v
                    lz = 0.012 - v * _s
                    return xf((_x + lx * cos(_a) + lz * sin(_a), ybf - 0.0026 - 0.006 * v * v, _z - lx * sin(_a) + lz * cos(_a)))

                sc_ = surface("scrap", sfn, 2, 3, thickness=0.0006)
                assign(sc_, papers[0])
                paint(sc_, grime=0.4, special=0.5)
                scraps.append(sc_)
            tacks.append(trs(xf((sx_, ybf - 0.0032, sz_)), (-90 if back else 90, 0, 0)))

    pin_up(int(p.posters), rp, False)
    if mode == "ground":                # a freestanding board is used from both sides
        pin_up(int(p.posters) // 3, rng.fork("posters_back"), True)
    if iv:
        ink = mesh_object("ink", iv, ifc)
        assign(ink, inkm)
    if tacks:
        tb = lathe("tack", [(0.0046, 0.0), (0.004, 0.0019), (0.0, 0.0031)], segments=5)
        tk_o = instances_to_mesh("tacks", tb, tacks, keep_base=False)
        assign(tk_o, iron)

    # ------------------------------------------------------------------ materials and weathering
    splash = 0.0 if hang else 1.0
    for o in timber:
        assign(o, wood)
        paint(o, grime=mask_mul(height_ramp(0.0, 0.55), splash),
              wear=mask_mul(noise_mask(0.09, seed=p.seed + 21, lo=0.5, hi=0.8), 0.35 + 0.4 * T),
              special=mask_mul(height_ramp(0.0, 0.4), noise_mask(0.15, seed=p.seed + 22), 0.7 * splash))
    for o in boards:
        assign(o, wood)
        paint(o, grime=mask_mul(noise_mask(0.2, seed=p.seed + 23), 0.5),
              wear=mask_mul(noise_mask(0.14, seed=p.seed + 24, lo=0.42, hi=0.62), 0.55 + 0.4 * T))
    for o in ironwork:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.05, seed=p.seed + 25, lo=0.35, hi=0.75), 0.35 + 0.4 * T))

    socket("board", "attach", (0, ybf - 0.02, zmid), role="sign", width=iw, height=bh)

    if hang:   # wall-hung: origin at the centre of the panel on the wall plane
        for o in list(bpy.context.scene.objects):
            if o.parent is None:
                o.location.z -= zmid
