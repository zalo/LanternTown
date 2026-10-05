"""Dressed stone gate pier. A rough two-stone footing course and an ashlar plinth die carry a moulded
plinth cap; the shaft is laid in individually cut stones (rock-faced rustication with V-channelled
joints, sunk moulded panels between corner stiles, or plain bonded ashlar with a bullnose band); a
necking astragal, frieze and a one-piece cornice cap (bed mould, dentils, corona with drip, cymatium,
weathered top) close it, and a finial block carries a ball, a gadrooned urn, a pineapple or a gas lantern
on an iron stem. Optional brass tablet, wrought-iron gate pintles leaded into one flank and a scrolled
bracket lantern (composed wall_lantern) fixed to the front and / or back face, which lights the shaft.
"""
import numpy as np
from math import sin, cos, pi, radians, atan2, hypot, sqrt, ceil

from tk import *

META = {"name": "Stone gate pier", "class": "structure", "mount": "ground", "focus": 0.72}
PARAMS = {
    "height": {"default": 2.6, "min": 1.4, "max": 4.5, "desc": "height to the top of the cornice cap, finial excluded (m)"},
    "width": {"default": 0.72, "min": 0.45, "max": 1.2, "desc": "shaft width across the front (m)"},
    "depth_ratio": {"default": 1.0, "min": 0.7, "max": 1.4, "desc": "shaft depth as a fraction of the width"},
    "shaft": {"default": "rusticated", "choices": ["rusticated", "panelled", "plain"], "desc": "shaft treatment"},
    "finial": {"default": "ball", "choices": ["ball", "urn", "pineapple", "lantern", "none"],
               "desc": "what stands on the cap ('none' = bare block with a lantern socket)"},
    "course": {"default": 0.3, "min": 0.2, "max": 0.45, "desc": "nominal course height of the shaft stones (m)"},
    "dentils": {"default": True, "desc": "dentil course under the corona of the cornice"},
    "moss": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "moss / lichen on ledges, the foot and the damp back"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "knocked arrises, settled stones, rain staining"},
    "stone": {"default": "#b9ab92", "desc": "stone colour"},
    "plaque": {"default": "1786", "desc": "raised lettering of a brass tablet on the front ('' = none)"},
    "pintles": {"default": "none", "choices": ["none", "left", "right"], "desc": "wrought-iron gate pintles on one flank"},
    "iron": {"default": "#1b2622", "desc": "paint colour of the lantern ironwork"},
    "intensity": {"default": 24.0, "min": 8, "max": 120, "desc": "lantern light intensity (cd)"},
    "bracket": {"default": "front", "choices": ["none", "front", "back", "both"],
                "desc": "wrought bracket lantern fixed to the shaft face(s)"},
    "bracket_cd": {"default": 4.5, "min": 1, "max": 40, "desc": "bracket lantern intensity (cd)"},
}
VARIANTS = {
    "urn_panelled": {"shaft": "panelled", "finial": "urn", "bracket": "none", "height": 2.3, "width": 0.84, "stone": "#c9bda4",
                     "moss": 0.12, "age": 0.25, "plaque": "", "seed": 4},
    "ball_tall_plain": {"shaft": "plain", "finial": "ball", "bracket": "none", "height": 3.5, "width": 0.78, "course": 0.36,
                        "pintles": "right", "stone": "#a9a39a", "plaque": "No 4", "moss": 0.2, "seed": 9},
    "pineapple_mossy": {"shaft": "rusticated", "finial": "pineapple", "bracket": "none", "height": 2.0, "width": 0.6,
                        "course": 0.25, "moss": 0.95, "age": 0.95, "stone": "#a89c86", "dentils": False, "plaque": "",
                        "pintles": "left", "seed": 17},
    "grand_lantern": {"shaft": "panelled", "finial": "lantern", "bracket": "none", "height": 3.9, "width": 1.1,
                      "depth_ratio": 0.8, "course": 0.4, "stone": "#b3987a", "plaque": "TOWN GATE", "intensity": 32.0,
                      "moss": 0.35, "seed": 23},
}


# ------------------------------------------------------------------------------------------ helpers
def shade(hexc, f):
    h = hexc.lstrip("#")
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(int(h[i:i + 2], 16) * f)))) for i in (0, 2, 4))


def arc2(c, r, a0, a1, n=6):
    return [(c[0] + r * cos(radians(a0 + (a1 - a0) * i / n)), c[1] + r * sin(radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def ogee(p0, p1, n=8):
    """S-curve between two profile points (vertical tangents at both ends)."""
    return [(lerp(p0[0], p1[0], 0.5 - 0.5 * cos(pi * i / n)), lerp(p0[1], p1[1], i / n)) for i in range(n + 1)]


def ring_mould(name, prof, hx, hy, seg=0.085, cap_bot=False, cap_top=False):
    """Rectangular 'lathe': the profile [(offset, z), ...] (bottom -> top along the visible outside)
    runs round a 2hx x 2hy rectangle with true mitred corners and subdivided sides."""
    nx = max(2, int(ceil(2 * hx / seg)))
    ny = max(2, int(ceil(2 * hy / seg)))
    ring = [(-1 + 2 * i / nx, -1) for i in range(nx)] + [(1, -1 + 2 * i / ny) for i in range(ny)] \
        + [(1 - 2 * i / nx, 1) for i in range(nx)] + [(-1, 1 - 2 * i / ny) for i in range(ny)]
    n = len(ring)
    verts = [(sx * (hx + off), sy * (hy + off), z) for (off, z) in prof for (sx, sy) in ring]
    faces = []
    for j in range(len(prof) - 1):
        for i in range(n):
            i2 = (i + 1) % n
            faces.append((j * n + i, j * n + i2, (j + 1) * n + i2, (j + 1) * n + i))
    if cap_bot:
        faces.append(tuple(reversed(range(n))))
    if cap_top:
        faces.append(tuple((len(prof) - 1) * n + i for i in range(n)))
    o = mesh_object(name, verts, faces)
    uv_box(o)
    smooth(o, 38)
    return o


def perim(hx, hy, off, side, u):
    """Point on the rectangle outline grown by `off`; side 0 front, 1 right, 2 back, 3 left; u in -1..1."""
    if side == 0:
        return (u * (hx + off), -(hy + off))
    if side == 1:
        return (hx + off, u * (hy + off))
    if side == 2:
        return (-u * (hx + off), hy + off)
    return (-(hx + off), -u * (hy + off))


# -------------------------------------------------------------------------------------------- build
def build(p, rng):
    H, w = p.height, p.width
    d = w * p.depth_ratio
    hx, hy = w / 2, d / 2
    wm = min(w, d)
    age, moss = p.age, p.moss
    k = clamp(0.5 * w / 0.72 + 0.5 * H / 2.6, 0.75, 1.45)       # scale of plinth / cornice members
    k = min(k, 0.58 * H / 0.92)
    rust = p.shaft == "rusticated"

    st_a = mat("stone_rough" if rust else "stone_smooth", p.stone)
    st_b = mat("stone_rough" if rust else "stone_smooth", shade(p.stone, 0.9))
    st_m = mat("stone_smooth", shade(p.stone, 1.03)) if rust else st_a    # mouldings, frieze, finial
    st_base = mat("stone_rough", shade(p.stone, 0.9))
    stones = []                                                  # everything that gets the stone weathering

    # ------------------------------------------------------------------ a single cut stone
    def cut_stone(name, size, loc, r, chamfer, material, rough=0.003, bulge=0.0, exposed=("+x", "-x", "+y", "-y"),
                  flat_bottom=False, chips=True, detail=0.09, dens=0.0, chip_h=True):
        sx, sy, sz = size
        o = stone_block(name, size, chamfer=chamfer, rough=rough, rng=r, detail=detail)
        if bulge > 0:
            if dens > 0:
                densify(o, dens)
            mg = chamfer * 1.15
            amp = {e: r.vary(bulge, 0.35) for e in exposed}

            def rock(co):
                mz = sz / 2 - abs(co.z)
                for e in exposed:
                    ax = 0 if e[1] == "x" else 1
                    sg = 1.0 if e[0] == "+" else -1.0
                    half = (sx, sy)[ax] / 2
                    if sg * co[ax] > half - chamfer * 0.9:
                        m = min(mz, (sy, sx)[ax] / 2 - abs(co[1 - ax]))
                        f = smoothstep(mg, mg + 0.035, m) * (0.55 + 0.45 * smoothstep(mg + 0.03, mg + 0.13, m))
                        co[ax] += sg * amp[e] * f
                return co

            transform_verts(o, rock)
            displace_noise(o, strength=bulge * 0.32, scale=0.05, seed=int(r.u(0, 9999)))
        if chips:
            # knocked arrises: scoop the stone back round a few points on its exposed edges
            for _ in range(3):
                if not r.chance((0.12 + 0.55 * age) * (1.0 if bulge > 0 else 0.6)):
                    continue
                e = r.pick(list(exposed))
                ax = 0 if e[1] == "x" else 1
                sg = 1.0 if e[0] == "+" else -1.0
                pt = [0.0, 0.0, 0.0]
                dr = [0.0, 0.0, 0.0]
                pt[ax] = sg * (sx, sy)[ax] / 2
                dr[ax] = -sg
                if r.chance(0.6) or not chip_h:
                    s2 = r.pick([-1.0, 1.0])
                    pt[1 - ax] = s2 * (sy, sx)[ax] / 2
                    dr[1 - ax] = -s2
                    pt[2] = r.u(-sz / 2, sz / 2)
                else:
                    s2 = r.pick([-1.0, 1.0])
                    pt[2] = s2 * sz / 2
                    dr[2] = -s2
                    pt[1 - ax] = r.u(-0.5, 0.5) * (sy, sx)[ax]
                R = r.u(0.04, 0.085) * (0.7 + 0.6 * age) * (1.0 if bulge > 0 else 0.6)
                dep = R * r.u(0.3, 0.5)
                pv, dv = Vector(pt), Vector(dr).normalized()

                def chip(co, pv=pv, dv=dv, R=R, dep=dep):
                    q = (co - pv).length
                    if q < R:
                        co += dv * dep * (1.0 - (q / R) ** 2)
                    return co

                transform_verts(o, chip)
        if flat_bottom:
            transform_verts(o, lambda co: Vector((co.x, co.y, max(co.z, -sz / 2))))
        j = 0.0025 * age
        place(o, (loc[0] + r.jitter(0, j), loc[1] + r.jitter(0, j), loc[2]), rot=(0, 0, r.jitter(0, 0.3 * age)))
        assign(o, material)
        stones.append(o)
        return o

    def course(name, sx, sy, zc, h, split, r, mats, chamfer, rough=0.003, bulge=0.0, gap=0.005, flat_bottom=False,
               plain_front=False, dens=0.0, chip_h=True):
        """One course: a through-stone (split None) or two stones with the perpend on the front/back ('x')
        or on the flanks ('y')."""
        hh = h - gap
        if split is None:
            return [cut_stone(name, (sx, sy, hh), (0, 0, zc), r, chamfer, r.pick(mats), rough, bulge, flat_bottom=flat_bottom, dens=dens, chip_h=chip_h)]
        out = []
        if split == "x":
            j = r.jitter(0.0, 0.08 * sx)
            a, b = sx / 2 + j - gap / 2, sx / 2 - j - gap / 2
            out.append(cut_stone(name + "a", (a, sy, hh), (-sx / 2 + a / 2, 0, zc), r, chamfer, r.pick(mats), rough, bulge,
                                 exposed=("-x", "+y", "-y"), flat_bottom=flat_bottom, dens=dens, chip_h=chip_h))
            out.append(cut_stone(name + "b", (b, sy, hh), (sx / 2 - b / 2, 0, zc), r, chamfer, r.pick(mats), rough, bulge,
                                 exposed=("+x", "+y", "-y"), flat_bottom=flat_bottom, dens=dens, chip_h=chip_h))
        else:
            j = r.jitter(0.0, 0.08 * sy)
            a, b = sy / 2 + j - gap / 2, sy / 2 - j - gap / 2
            out.append(cut_stone(name + "a", (sx, a, hh), (0, -sy / 2 + a / 2, zc), r, chamfer, r.pick(mats), rough,
                                 0.0 if plain_front else bulge, exposed=("+x", "-x", "-y"), flat_bottom=flat_bottom, dens=dens, chip_h=chip_h))
            out.append(cut_stone(name + "b", (sx, b, hh), (0, sy / 2 - b / 2, zc), r, chamfer, r.pick(mats), rough, bulge,
                                 exposed=("+x", "-x", "+y"), flat_bottom=flat_bottom, dens=dens, chip_h=chip_h))
        return out

    def mould(name, prof, mhx=hx, mhy=hy, cap_bot=True, cap_top=True, bev=0.0035, seed=0):
        o = ring_mould(name, prof, mhx, mhy, cap_bot=cap_bot, cap_top=cap_top)
        displace_noise(o, strength=0.0016, scale=0.22, seed=p.seed + seed)
        bevel(o, width=bev, segments=2, angle=32)
        assign(o, st_m)
        stones.append(o)
        return o

    # ------------------------------------------------------------------ plinth
    hb, hd, hm = 0.16 * k, 0.25 * k, 0.1 * k
    e0, e1 = 0.12 * k, 0.078 * k
    rp = rng.fork("plinth")
    course("footing", w + 2 * e0, d + 2 * e0, hb / 2 + 0.0025, hb, "x", rp, [st_base], 0.016 * k, rough=0.006, bulge=0.007 * k,
           flat_bottom=True, chip_h=False)
    course("die", w + 2 * e1, d + 2 * e1, hb + hd / 2, hd, "y", rp, [st_m], 0.006 * k, chip_h=False)
    zpm = hb + hd
    zp = zpm + hm
    pm = [(e1 - 0.04, zpm), (e1 + 0.008 * k, zpm), (e1 + 0.008 * k, zpm + 0.022 * k)]
    pm += ogee((e1 + 0.008 * k, zpm + 0.022 * k), (0.024 * k, zp - 0.034 * k), 8)[1:]
    pm += arc2((0.024 * k, zp - 0.019 * k), 0.015 * k, -90, 90, 6)[1:]
    pm += [(0.014 * k, zp - 0.004 * k), (0.014 * k, zp), (-0.04, zp)]
    mould("plinth_cap", pm, seed=1)

    # ------------------------------------------------------------------ levels of the cap
    kc = 1.22 * k                      # scale of the cornice members
    hf, hc = 0.165 * k, 0.2 * kc
    zc0 = H - hc                       # underside of the cornice stone
    zf0 = zc0 - hf                     # underside of the frieze = top of the shaft
    sh = zf0 - zp
    rs = rng.fork("shaft")

    def courses(z0, z1, target):
        n = max(1, int(round((z1 - z0) / target)))
        hs = [rs.vary(1.0, 0.07) for _ in range(n)]
        s = sum(hs)
        out, z = [], z0
        for h in hs:
            h = h * (z1 - z0) / s
            out.append((z + h / 2, h))
            z += h
        return out

    # bracket lanterns are measured first: the tablet has to sit clear below the back plate
    zl_ = min(H * 0.74, 2.35)
    brackets = []
    for nm_, sg_, rz_ in (("front", -1, 0), ("back", 1, 180)):
        if p.bracket in (nm_, "both"):
            wl = use("wall_lantern", reach=clamp(0.42 * k, 0.38, 0.6), style="scroll", carry="stand", roof="ogee",
                     lantern_width=0.23, paint=p.iron, intensity=p.bracket_cd, seed=p.seed + 5 + rz_)
            zl_ = clamp(H - 0.03 - wl.max.z, zp + 0.4 * sh, 2.6)      # lantern finial just under the top of the cap
            brackets.append((wl, nm_, sg_, rz_))
    zeye, tablet_room = 1.5, True
    if p.bracket in ("front", "both"):
        zeye = min(zeye, zl_ + brackets[0][0].min.z - 0.17)
        tablet_room = zeye > zp + 0.25 * sh - 0.05          # a very low pier has no face left under the bracket
    zeye = clamp(zeye, zp + 0.25 * sh, zp + 0.75 * sh)
    plaque_z, plaque_y = None, -hy
    rd = 0.0

    if p.shaft in ("rusticated", "plain"):
        cs = courses(zp, zf0, p.course)
        i_eye = min(range(len(cs)), key=lambda i: abs(cs[i][0] - zeye))
        plaque_z = cs[i_eye][0]
        want_tablet = bool(p.plaque.strip())
        for i, (zc, h) in enumerate(cs):
            split = "y" if (i - i_eye) % 2 == 0 else "x"       # the tablet course shows one long stone to the front
            if rust:
                course(f"c{i}", w, d, zc, h, split, rs, [st_a, st_a, st_b], 0.021 * k, rough=0.004, bulge=0.013 * k,
                       plain_front=(i == i_eye and want_tablet), dens=clamp(0.052 * k, 0.05, 0.07))
            else:
                course(f"c{i}", w, d, zc, h, split, rs, [st_a, st_a, st_b], 0.0055 * k, rough=0.0025)
        if not rust and len(cs) >= 3:
            # bullnose band over a bed joint, about two thirds up
            ib = min(range(1, max(2, len(cs) - 1)), key=lambda i: abs((cs[i][0] - cs[i][1] / 2) - (zp + 0.6 * sh)))
            zb = cs[ib][0] - cs[ib][1] / 2
            if abs(zb - plaque_z) < cs[i_eye][1] * 0.5 + 0.06 * k and ib + 1 < len(cs):
                zb = cs[ib + 1][0] - cs[ib + 1][1] / 2
            bp = [(-0.03, zb - 0.046 * k), (0.007 * k, zb - 0.046 * k), (0.007 * k, zb - 0.03 * k)]
            bp += arc2((0.007 * k, zb), 0.03 * k, -90, 90, 10)[1:]
            bp += [(0.007 * k, zb + 0.046 * k), (-0.03, zb + 0.046 * k)]
            mould("band", bp, seed=2)
        core = box("core", (w - 0.05 * k, d - 0.05 * k, sh), loc=(0, 0, zp + sh / 2))
        assign(core, st_base)
    else:
        # sunk panels: every course is one stone worked with corner stiles and a recessed field
        rd = 0.042 * k
        sw = clamp(0.19 * wm, 0.085, 0.2)
        rb, rt = 0.17 * k, 0.14 * k
        cut_stone("rail_bot", (w, d, rb - 0.005), (0, 0, zp + rb / 2), rs, 0.0055 * k, st_a)
        cut_stone("rail_top", (w, d, rt - 0.005), (0, 0, zf0 - rt / 2), rs, 0.0055 * k, st_b)
        z0, z1 = zp + rb, zf0 - rt
        cs = courses(z0, z1, p.course * 1.5)
        for i, (zc, h) in enumerate(cs):
            m_ = rs.pick([st_a, st_a, st_b])
            cut_stone(f"field{i}", (w - 2 * rd, d - 2 * rd, h - 0.005), (0, 0, zc), rs, 0.004 * k, m_, chips=False, detail=0.12)
            for qx in (-1, 1):
                for qy in (-1, 1):
                    ex = (("+x" if qx > 0 else "-x"), ("+y" if qy > 0 else "-y"))
                    cut_stone(f"stile{i}", (sw, sw, h - 0.005), (qx * (hx - sw / 2), qy * (hy - sw / 2), zc), rs, 0.0055 * k, m_,
                              exposed=ex, detail=0.11)
        ph = z1 - z0
        fr = [(0.004, rd + 0.0015), (-0.009 * k, rd + 0.0015)]
        fr += [(-0.009 * k - 0.02 * k * sin(radians(a)), rd - 0.0185 * k + 0.02 * k * cos(radians(a))) for a in (15, 30, 45, 60, 75, 90)]
        fr += [(-0.029 * k, rd - 0.025 * k), (-0.036 * k, rd - 0.025 * k), (-0.036 * k, -0.004)]
        for (pw_, yy, rz) in ((w - 2 * sw, hy - rd, 0), (d - 2 * sw, hx - rd, 90), (w - 2 * sw, hy - rd, 180), (d - 2 * sw, hx - rd, 270)):
            f = ring_mould("panel_mould", fr, pw_ / 2, ph / 2, seg=0.12)
            bevel(f, width=0.0025, segments=1, angle=32)
            a = radians(rz)
            place(f, (yy * sin(a), -yy * cos(a), (z0 + z1) / 2), rot=(90, 0, rz))
            assign(f, st_m)
            stones.append(f)
        plaque_z = clamp(zeye + 0.1, z0 + 0.3 * ph, z0 + 0.72 * ph)
        plaque_y = -(hy - rd)
        core = box("core", (w - 2 * rd - 0.03, d - 2 * rd - 0.03, sh - 0.02), loc=(0, 0, zp + sh / 2))
        assign(core, st_base)

    # ------------------------------------------------------------------ necking, frieze, cornice cap
    ast = [(-0.03, zf0 - 0.006 * k), (0.008 * k, zf0 - 0.006 * k), (0.008 * k, zf0 + 0.004 * k)]
    ast += arc2((0.01 * k, zf0 + 0.019 * k), 0.015 * k, -90, 90, 8)
    ast += [(0.008 * k, zf0 + 0.034 * k), (0.008 * k, zf0 + 0.044 * k), (-0.03, zf0 + 0.05 * k)]
    mould("astragal", ast, bev=0.002, seed=3)
    cut_stone("frieze", (w, d, hf + 0.004), (0, 0, zf0 + hf / 2), rs, 0.004, st_m, rough=0.002, chips=False)

    pr = 0.225 * wm                      # half size of the finial block
    rise = 0.1 * k
    cp = [(-0.03 / kc, 0.0), (0.008, 0.0), (0.008, 0.012)]
    if p.dentils:
        cp += [(0.014, 0.012), (0.014, 0.060), (0.040, 0.060)]
        cp += arc2((0.040, 0.085), 0.025, -90, 0, 6)[1:]
        z_bed = 0.085
    else:
        cp += ogee((0.008, 0.012), (0.062, 0.082), 9)[1:]
        cp += [(0.065, 0.082), (0.065, 0.085)]
        z_bed = 0.085
    cp += [(0.065, z_bed + 0.004), (0.086, z_bed + 0.004), (0.089, z_bed + 0.012), (0.095, z_bed + 0.012), (0.098, z_bed + 0.002),
           (0.105, z_bed + 0.002), (0.105, 0.14), (0.110, 0.14), (0.110, 0.147)]
    cp += ogee((0.110, 0.147), (0.134, 0.188), 8)[1:]
    cp += [(0.134, 0.2)]
    cprof = [(o * kc, zc0 + z * kc) for (o, z) in cp]
    o_in = -(wm / 2 - pr) + 0.01
    o_top = 0.126 * kc
    P_out = 0.134 * kc

    def slope_z(o):
        return H + 0.004 * k + rise * (o_top - o) / (o_top - o_in)

    for i in range(6):
        o = lerp(o_top, o_in, i / 5)
        cprof.append((o, slope_z(o)))
    mould("cornice", cprof, bev=0.003, seed=4)

    if p.dentils:
        dw, dh, dp = 0.03 * kc, 0.04 * kc, 0.024 * kc
        fb = 0.014 * kc
        dent = box("dentil", (dw, dp + 0.006, dh), bevel=0.003 * kc, segments=1)
        mats_ = []
        rdn = rng.fork("dentils")
        zd = zc0 + 0.016 * kc + dh / 2
        for side in range(4):
            L = (hx if side % 2 == 0 else hy) + fb + dp - dw / 2
            hh_x, hh_y = (hx, hy)
            n = max(3, int(round(2 * L / (0.056 * kc))) + 1)
            for i in range(n):
                if side % 2 == 1 and i in (0, n - 1):
                    continue                      # the corner dentils belong to the front / back rows
                u = -L + 2 * L * i / (n - 1)
                if side == 0:
                    loc, rz = (u, -(hy + fb + dp / 2 - 0.003)), 0
                elif side == 2:
                    loc, rz = (u, hy + fb + dp / 2 - 0.003), 0
                elif side == 1:
                    loc, rz = (hx + fb + dp / 2 - 0.003, u), 90
                else:
                    loc, rz = (-(hx + fb + dp / 2 - 0.003), u), 90
                mats_.append(trs((loc[0], loc[1], zd + rdn.jitter(0, 0.001)), (0, 0, rz + rdn.jitter(0, 0.8)),
                                 (1.0, 1.0, rdn.vary(1.0, 0.03))))
        dn = instances_to_mesh("dentils", dent, mats_, keep_base=False)
        assign(dn, st_m)
        stones.append(dn)

    zt = H + rise + 0.07 * k              # top of the finial block
    pbx, pby = pr + hx - wm / 2, pr + hy - wm / 2
    cut_stone("finial_block", (2 * pbx, 2 * pby, 0.07 * k + 0.012), (0, 0, zt - (0.07 * k + 0.012) / 2), rng.fork("fb"), 0.008 * k,
              st_m, rough=0.002, chips=False)

    # carved paterae on the frieze, one to each face
    rpat = 0.37 * hf
    pat = lathe("patera", [(0.0, 0.0), (rpat, 0.0), (rpat, 0.005), (rpat * 0.9, 0.011), (rpat * 0.72, 0.008), (rpat * 0.5, 0.017),
                           (rpat * 0.3, 0.012), (rpat * 0.2, 0.02), (0.0, 0.024)], segments=32)

    def petals(co):
        r_ = hypot(co.x, co.y)
        if r_ < rpat * 0.25 or r_ > rpat * 0.95:
            return co
        f = 0.55 + 0.45 * abs(cos(4 * atan2(co.y, co.x)))
        return Vector((co.x, co.y, co.z * f))

    transform_verts(pat, petals)
    pm_ = [trs((0, -hy + 0.002, zf0 + hf * 0.52), (90, 0, 0)), trs((hx - 0.002, 0, zf0 + hf * 0.52), (90, 0, 90)),
           trs((0, hy - 0.002, zf0 + hf * 0.52), (90, 0, 180)), trs((-hx + 0.002, 0, zf0 + hf * 0.52), (90, 0, 270))]
    pat = instances_to_mesh("paterae", pat, pm_, keep_base=False)
    assign(pat, st_m)
    smooth(pat, 40)
    stones.append(pat)

    lead_top = p.finial in ("lantern", "none")
    if lead_top:
        ld = mat("lead")
        lp = [(P_out - 0.004, H - 0.016 * kc), (P_out + 0.0045, H - 0.016 * kc), (P_out + 0.0045, H + 0.003)]
        for i in range(6):
            o = lerp(o_top + 0.004, o_in - 0.004, i / 5)
            lp.append((o, slope_z(o) + 0.0045))
        lp += [(o_in - 0.004, slope_z(o_in) + 0.045 * k), (o_in - 0.012, slope_z(o_in) + 0.045 * k)]
        sheet = ring_mould("lead_sheet", lp, hx, hy, seg=0.11)
        displace_noise(sheet, strength=0.0015, scale=0.08, seed=p.seed + 51)
        bevel(sheet, width=0.003, segments=1, angle=32)
        lead_objs = [sheet]
        for qx in (-1, 1):                 # wood-cored rolls up the four hips
            for qy in (-1, 1):
                c0 = (qx * (hx + P_out - 0.004), qy * (hy + P_out - 0.004), H + 0.006)
                c1 = (qx * (hx + o_in + 0.004), qy * (hy + o_in + 0.004), slope_z(o_in) + 0.012)
                lead_objs.append(sweep("lead_roll", [c0, c1], 0.013, sides=8, smooth_path=False))
        for o in lead_objs:
            assign(o, ld)
            paint(o, grime=mask_mul(noise_mask(0.12, seed=p.seed + 52, lo=0.2, hi=0.8), 0.55 + 0.4 * age))

    # ------------------------------------------------------------------ finial
    fin = p.finial
    rf = rng.fork("finial")
    if fin == "ball":
        rb_ = 0.275 * wm
        pf = [(0.0, 0.0), (0.66, 0.0), (0.69, 0.035), (0.69, 0.075), (0.66, 0.11), (0.54, 0.135), (0.44, 0.17), (0.385, 0.23),
              (0.365, 0.31), (0.385, 0.38), (0.45, 0.415), (0.49, 0.45), (0.49, 0.485), (0.45, 0.52), (0.37, 0.545)]
        zc_ = 0.545 + sqrt(1 - 0.37 ** 2)
        a0 = -atan2(sqrt(1 - 0.37 ** 2), 0.37)
        for i in range(1, 19):
            a = a0 + (pi / 2 - a0) * i / 18
            pf.append((max(cos(a), 0.0), zc_ + sin(a)))
        ball = lathe("ball", [(r * rb_, z * rb_) for (r, z) in pf], segments=44, loc=(0, 0, zt))
        displace_noise(ball, strength=0.002, scale=0.2, seed=p.seed + 31)
        assign(ball, st_m)
        stones.append(ball)
        top_z = zt + (zc_ + 1) * rb_
    elif fin == "urn":
        u_ = 0.3 * wm
        pf = [(0.0, 0.0), (0.56, 0.0), (0.585, 0.03), (0.585, 0.07), (0.56, 0.1), (0.44, 0.13), (0.3, 0.18), (0.22, 0.27), (0.2, 0.35),
              (0.23, 0.41), (0.31, 0.44), (0.33, 0.47), (0.31, 0.5), (0.25, 0.53), (0.27, 0.56)]
        body = [(0.27, 0.56), (0.5, 0.64), (0.74, 0.8), (0.9, 1.0), (0.985, 1.2), (1.0, 1.32)]
        bs = resample_path([(r, 0.0, z) for (r, z) in body], n=22)
        pf += [(float(q[0]), float(q[2])) for q in bs][1:]
        pf += [(1.035, 1.335), (1.045, 1.37), (1.035, 1.405), (1.0, 1.42)]
        neck = [(1.0, 1.42), (0.9, 1.5), (0.72, 1.62), (0.62, 1.78), (0.635, 1.92), (0.73, 2.03)]
        ns_ = resample_path([(r, 0.0, z) for (r, z) in neck], n=14)
        pf += [(float(q[0]), float(q[2])) for q in ns_][1:]
        pf += [(0.79, 2.05), (0.81, 2.08), (0.79, 2.115), (0.72, 2.13), (0.68, 2.16)]
        lid = [(0.68, 2.16), (0.6, 2.25), (0.44, 2.36), (0.26, 2.44), (0.15, 2.5), (0.11, 2.58)]
        ls = resample_path([(r, 0.0, z) for (r, z) in lid], n=12)
        pf += [(float(q[0]), float(q[2])) for q in ls][1:]
        pf += [(0.15, 2.62), (0.19, 2.68), (0.2, 2.74), (0.17, 2.81), (0.1, 2.86), (0.0, 2.88)]
        ng = 14
        urn = lathe("urn", [(max(r, 0.0) * u_, z * u_) for (r, z) in pf], segments=ng * 6, loc=(0, 0, zt))

        def gad(co):
            z = co.z / u_
            e = min(smoothstep(0.58, 0.7, z), 1.0 - smoothstep(1.2, 1.31, z))        # gadrooned bowl
            e2 = min(smoothstep(2.17, 2.24, z), 1.0 - smoothstep(2.4, 2.5, z))       # and lid
            if e <= 0 and e2 <= 0:
                return co
            a = atan2(co.y, co.x)
            c = abs(cos(a * ng / 2))
            f = 1.0 + 0.085 * e * (sqrt(1.0 - c * c) - 0.45) + 0.07 * e2 * (sqrt(1.0 - c * c) - 0.45)
            return Vector((co.x * f, co.y * f, co.z))

        transform_verts(urn, gad)
        displace_noise(urn, strength=0.0015, scale=0.2, seed=p.seed + 32)
        assign(urn, st_m)
        stones.append(urn)
        top_z = zt + 2.88 * u_
    elif fin == "pineapple":
        q_ = 0.27 * wm
        pf = [(0.0, 0.0), (0.7, 0.0), (0.73, 0.035), (0.73, 0.075), (0.68, 0.11), (0.5, 0.15), (0.38, 0.22), (0.35, 0.3), (0.39, 0.38),
              (0.5, 0.43), (0.53, 0.47), (0.5, 0.5)]
        zb0, zb1 = 0.52, 3.15
        nb = 46
        for i in range(nb + 1):
            t = i / nb
            pf.append((sin(pi * (0.085 + 0.865 * t)) ** 0.72, zb0 + (zb1 - zb0) * t))
        pf += [(0.16, zb1 + 0.03), (0.0, zb1 + 0.03)]
        nsc, rows = 9, 6
        pine = lathe("pineapple", [(r * q_, z * q_) for (r, z) in pf], segments=nsc * 8, loc=(0, 0, zt), smooth_angle=70)

        def scales(co):
            t = (co.z / q_ - zb0) / (zb1 - zb0)
            if t <= 0.0 or t >= 1.0:
                return co
            a = atan2(co.y, co.x)
            e = min(smoothstep(0.0, 0.07, t), 1.0 - smoothstep(0.93, 1.0, t))
            b = (0.5 + 0.5 * cos(nsc * a + 2 * pi * rows * t)) * (0.5 + 0.5 * cos(nsc * a - 2 * pi * rows * t))
            f = 1.0 + 0.13 * e * (b ** 0.55 - 0.42)
            return Vector((co.x * f, co.y * f, co.z))

        transform_verts(pine, scales)
        assign(pine, st_m)
        stones.append(pine)

        def leaf_mesh(name, ln, wd, curl, th):
            lf = surface(name, lambda u, v: ((u - 0.5) * wd * (1.0 - v ** 1.7) * (0.75 + 0.5 * sin(v * pi * 0.6)),
                                             -curl * v ** 2.2 - 0.12 * wd * cos((u - 0.5) * pi) + 0.06 * wd,
                                             v * ln * (1.0 - 0.12 * v)), 5, 7, thickness=th)
            realize(lf)
            return lf

        lf = leaf_mesh("crown_leaf", 0.95 * q_, 0.36 * q_, 0.3 * q_, 0.035 * q_)
        mm = []
        zl = zt + (zb1 - 0.06) * q_
        for (cnt, rad_, tilt, sc, ph) in ((5, 0.07, 9, 1.0, 10), (7, 0.15, 30, 0.9, 35), (8, 0.2, 52, 0.72, 0)):
            for i in range(cnt):
                a = 360.0 * i / cnt + ph + rf.jitter(0, 6)
                ar = radians(a)
                mm.append(trs((rad_ * q_ * sin(ar), -rad_ * q_ * cos(ar), zl - 0.04 * q_ * (tilt / 50.0)),
                              (tilt + rf.jitter(0, 4), 0, a), sc * rf.vary(1.0, 0.08)))
        crown = instances_to_mesh("crown", lf, mm, keep_base=False)
        cup = leaf_mesh("cup_leaf", 0.85 * q_, 0.46 * q_, 0.22 * q_, 0.04 * q_)
        mm = []
        for i in range(9):
            a = 360.0 * i / 9 + 20
            ar = radians(a)
            mm.append(trs((0.4 * q_ * sin(ar), -0.4 * q_ * cos(ar), zt + 0.45 * q_), (43, 0, a), rf.vary(1.0, 0.06)))
        cupm = instances_to_mesh("cup", cup, mm, keep_base=False)
        for o in (crown, cupm):
            assign(o, st_m)
            smooth(o, 50)
            stones.append(o)
        top_z = zt + (zb1 + 0.9) * q_
    elif fin == "lantern":
        iron = mat("iron_painted", p.iron)
        Lw = clamp(0.5 * wm, 0.3, 0.46)
        hs = 0.42 + 0.1 * (Lw - 0.3) / 0.16
        io = []
        io.append(box("lamp_plate", (0.21, 0.21, 0.012), loc=(0, 0, zt + 0.004), bevel=0.002))
        bt = bolt("hd_bolt", 0.009, 0.007)
        io.append(instances_to_mesh("hd_bolts", bt, [trs((sx_ * 0.082, sy_ * 0.082, zt + 0.01)) for sx_ in (-1, 1) for sy_ in (-1, 1)],
                                    keep_base=False))
        zk = hs * 0.52                      # mid knop
        sp = [(0.0, 0.008), (0.082, 0.008), (0.084, 0.018), (0.06, 0.026), (0.042, 0.042), (0.03, 0.07), (0.027, 0.09), (0.04, 0.105),
              (0.048, 0.12), (0.04, 0.135), (0.027, 0.15), (0.024, zk - 0.03), (0.034, zk - 0.016), (0.038, zk), (0.034, zk + 0.016),
              (0.023, zk + 0.03), (0.021, hs - 0.085), (0.03, hs - 0.07), (0.04, hs - 0.05), (0.044, hs - 0.03),
              (0.036, hs - 0.012), (0.036, hs), (0.0, hs)]
        io.append(lathe("lamp_stem", sp, segments=20, loc=(0, 0, zt)))
        hw_ = Lw * 0.56 * 0.5
        flat_bar = rounded_rect(0.02, 0.007, 0.002, 1)
        for q in range(4):
            a = radians(45 + q * 90)
            ca, sa = cos(a), sin(a)
            pth = [(0.03, 0.075), (0.13, 0.11), (0.14, hs - 0.14), (hw_ * 1.414 + 0.012, hs + 0.062)]
            pts = bezier_path(*[(r_ * ca, r_ * sa, zt + z_) for (r_, z_) in pth], n=11)
            io.append(sweep(f"stay{q}", pts, flat_bar, up=(-sa, ca, 0)))
            sc = spiral_path(0.006, 0.036, 1.4, n=18, a0_deg=200, plane="XZ", center=(0.098, 0, zt + 0.07))
            s = sweep(f"scroll{q}", sc, flat_bar, scale_fn=lambda t: 0.55 + 0.45 * t, up=(0, 1, 0))
            place(s, rot=(0, 0, 45 + q * 90))
            io.append(s)
        for o in io:
            assign(o, iron)
            paint(o, special=mask_mul(height_ramp(zt, zt + 0.16), noise_mask(0.05, seed=p.seed + 5, lo=0.3, hi=0.7), 0.8),
                  grime=height_ramp(zt, zt + 0.25))
        head = use("lantern_head", width=Lw, height=Lw * 1.25, roof="ogee", paint=p.iron, intensity=p.intensity, seed=p.seed + 1)
        head.place((0, 0, zt + hs))      # brings its own light; the cap rightly shades the shaft below
        top_z = zt + hs + head.size.z
    else:
        top_z = zt
        socket("lantern_top", "attach", (0, 0, zt), role="lantern")
    socket("top", "attach", (0, 0, top_z), role="prop_top")

    # ------------------------------------------------------------------ bronze tablet
    txt = p.plaque.strip()
    if txt and plaque_z is not None and tablet_room:
        bronze = mat("brass")
        th = clamp(0.1 * k, 0.07, 0.13)
        maxw = (w - (2 * clamp(0.19 * wm, 0.085, 0.2) if p.shaft == "panelled" else 0.0)) - 0.16 * k
        t = text("plaque_text", txt, size=th, depth=0.004, loc=(0, 0, 0), font="roman")
        bmin, bmax = world_bbox([t])
        tw, tht = bmax.x - bmin.x, bmax.z - bmin.z
        s = min(1.0, (maxw - 0.07) / max(tw, 1e-3))
        pwid, phei = tw * s + 0.075, tht * s + 0.07
        y0 = plaque_y - 0.009
        place(t, ((-(bmin.x + bmax.x) / 2) * s, y0, plaque_z - (bmin.z + bmax.z) / 2 * s), scale=(s, s, s))
        plate = shape_extrude("plaque_plate", rounded_rect(pwid, phei, 0.012, 3), depth=0.013, bevel=0.002,
                              loc=(0, y0, plaque_z), rot=(90, 0, 0))
        rim = shape_extrude("plaque_rim", rounded_rect(pwid - 0.008, phei - 0.008, 0.01, 3),
                            [rounded_rect(pwid - 0.024, phei - 0.024, 0.006, 3)], depth=0.003, bevel=0.001,
                            loc=(0, y0 - 0.0005, plaque_z), rot=(90, 0, 0))
        bt = bolt("pl_bolt", 0.0055, 0.004, kind="dome")
        bl = instances_to_mesh("plaque_bolts", bt, [trs((sx_ * (pwid / 2 - 0.022), y0, plaque_z + sz_ * (phei / 2 - 0.022)), (90, 0, 0))
                                                    for sx_ in (-1, 1) for sz_ in (-1, 1)], keep_base=False)
        for o in (t, plate, rim, bl):
            assign(o, bronze)
        paint(t, wear=1.0)
        paint(rim, wear=0.5)
        paint(plate, special=mask_mul(noise_mask(0.05, seed=p.seed + 7, lo=0.4, hi=0.9), 0.5), grime=0.55)

    # ------------------------------------------------------------------ gate pintles
    stains = []
    if p.pintles != "none":
        sg = 1.0 if p.pintles == "right" else -1.0
        wi = mat("iron_wrought")
        yb = -hy * 0.45
        emb = 0.03
        for q, zq in enumerate((zp + 0.14 * sh, zp + 0.8 * sh)):
            bar = box("pintle_bar", (0.085 + emb, 0.034, 0.034), loc=(sg * (hx + (0.085 - emb) / 2), yb, zq), bevel=0.004)
            pin = cylinder("pintle_pin", 0.0115, 0.085, loc=(sg * (hx + 0.064), yb, zq + 0.017 + 0.04), verts=12, bevel=0.003)
            col = cylinder("pintle_collar", 0.02, 0.008, loc=(sg * (hx + 0.064), yb, zq + 0.021), verts=12, bevel=0.002)
            lead_ = box("pintle_lead", (0.012, 0.07, 0.07), loc=(sg * (hx + 0.012 * (1 if rust else 0.3)), yb, zq), bevel=0.004)
            assign(lead_, mat("lead"))
            for o in (bar, pin, col):
                assign(o, wi)
                paint(o, special=mask_mul(noise_mask(0.03, seed=p.seed + 40 + q, lo=0.2, hi=0.7), 0.5 + 0.5 * age), wear=0.15)
            stains.append((sg * hx, yb, zq))
        socket("gate", "attach", (sg * (hx + 0.064), yb, 0.0), rot=(0, 0, 90 * sg), role="gate")

    # ------------------------------------------------------------------ weathering of the stonework
    sd = p.seed
    n_streak = noise_mask(0.06, seed=sd + 11, lo=0.3, hi=0.75)
    n_big = noise_mask(0.55, seed=sd + 12, lo=0.3, hi=0.8)
    n_m1 = noise_mask(0.2, seed=sd + 13, lo=0.55 - 0.4 * moss, hi=1.05 - 0.3 * moss)
    n_m2 = noise_mask(0.35, seed=sd + 14, lo=0.62 - 0.45 * moss, hi=0.9 - 0.35 * moss)
    squash = np.array([1.0, 1.0, 0.06])

    def grime_fn(co, n):
        z = co[:, 2]
        streak = n_streak(co * squash, n)
        splash = np.clip(1.0 - z / 0.42, 0, 1) ** 1.4 * 0.75
        under = np.where(z < zc0 + 0.09 * k, np.clip(1.0 - (zc0 - z) / (0.45 + 0.7 * age), 0, 1), 0.0) * streak * (0.45 + 0.5 * age)
        ledge = np.where(z < zpm, np.clip(1.0 - (zpm - z) / 0.25, 0, 1), 0.0) * streak * 0.5
        top = np.clip((z - H) / (0.1 * k), 0, 1) * (0.35 + 0.3 * age) * (0.4 + 0.6 * n_big(co * 3.0, n))
        g = np.maximum(np.maximum(splash, under), np.maximum(ledge, top))
        g = np.maximum(g, n_big(co, n) * 0.4 * age)
        for (sx_, sy_, sz_) in stains:                 # rust weeping down from the pintles
            dz = sz_ - z
            lat = np.hypot(co[:, 1] - sy_, (co[:, 0] - sx_) * 0.6)
            g = np.maximum(g, np.where((dz > -0.03) & (dz < 0.5), np.clip(1.0 - lat / (0.035 + 0.05 * np.clip(dz, 0, 1)), 0, 1)
                                       * np.clip(1.0 - dz / 0.5, 0, 1), 0.0) * 0.9)
        return np.clip(g, 0, 1)

    def moss_fn(co, n):
        z = co[:, 2]
        up = np.clip(n[:, 2], 0, 1) ** 0.8
        north = 0.3 + 0.7 * np.clip(n[:, 1], 0, 1)                     # the back (+Y) stays damp
        low = np.clip(1.0 - z / (0.2 + 1.1 * moss), 0, 1)
        v = np.maximum(up * n_m1(co, n) * clamp((moss - 0.15) * 1.7, 0.0, 1.0) * 1.2, np.maximum(low, 0.22 * north) * north * n_m2(co, n) * 1.2)
        return np.clip(v * min(1.0, 0.15 + 1.6 * moss), 0, 1)

    def wear_fn(co, n):
        z = co[:, 2]
        band = np.clip(1.0 - np.abs(z - 0.75) / 0.6, 0, 1)             # carts, shoulders and hands
        return np.clip(band * n_big(co * 2.3, n) * (0.25 + 0.5 * age), 0, 1)

    for o in stones:
        paint(o, grime=grime_fn, special=moss_fn if moss > 0.01 else None, wear=wear_fn)

    # ------------------------------------------------------------------ moss cushions on the ledges
    if moss > 0.4:
        rm = rng.fork("moss")
        clump = lathe("moss_clump", [(1.0, 0.0), (0.95, 0.16), (0.76, 0.34), (0.42, 0.46), (0.0, 0.5)], segments=7)
        mm = []
        for _ in range(int(5 + 16 * (moss - 0.4) / 0.6)):          # colonies, each a handful of cushions
            side = rm.weighted([(0, 0.12), (1, 0.2), (2, 0.48), (3, 0.2)])
            u0 = rm.u(-0.95, 0.95)
            where = rm.weighted([("ground", 0.34), ("foot", 0.3), ("pcap", 0.1), ("top", 0.26)])
            for _c in range(int(rm.u(3, 8))):
                u = clamp(u0 + rm.jitter(0, 0.09 / max(hx, hy)), -1, 1)
                s = rm.u(0.014, 0.042) * (0.7 + 0.5 * moss)
                if where == "ground":
                    x, y = perim(hx, hy, e0 + rm.u(0.0, 0.012), side, u)
                    z = 0.001
                elif where == "foot":
                    x, y = perim(hx, hy, rm.u(e1 + 0.01, e0 - 0.014), side, u)
                    z = hb - 0.005
                    s *= 0.75
                elif where == "pcap":
                    x, y = perim(hx, hy, e1 * rm.u(0.45, 0.7), side, u)
                    z = zpm + hm * 0.45
                    s *= 0.5
                else:
                    o_ = rm.u(0.0, 0.11 * kc)
                    x, y = perim(hx, hy, o_, side, u)
                    z = slope_z(o_) - 0.003
                    s *= 0.8
                mm.append(trs((x, y, z), (0, 0, rm.u(0, 360)), (s * rm.u(0.8, 1.7), s * rm.u(0.8, 1.3), s * rm.u(0.6, 1.1))))
        mo = instances_to_mesh("moss_clumps", clump, mm, keep_base=False)
        assign(mo, mat("moss"))
        smooth(mo, 60)

    # ------------------------------------------------------------------ sockets
    for wl, nm_, sg_, rz_ in brackets:
        wl.place((0, sg_ * (hy - rd + 0.012 * k), zl_), rot=(0, 0, rz_))
    for nm_, sg_, rz_ in (("front", -1, 0), ("back", 1, 180)):
        if p.bracket not in (nm_, "both"):
            socket("lantern_" + nm_, "attach", (0, sg_ * (hy + 0.012 * k), zl_), rot=(0, 0, rz_), role="lantern")
    socket("rail_right", "attach", (hx, 0, min(1.0, zp + 0.6 * sh)), rot=(0, 0, 90), role="railing")
    socket("rail_left", "attach", (-hx, 0, min(1.0, zp + 0.6 * sh)), rot=(0, 0, -90), role="railing")
