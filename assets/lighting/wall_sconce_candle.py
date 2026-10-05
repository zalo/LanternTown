"""Candle wall sconce: a forged strap back plate screwed to the wall carries a turned boss from which 1-3
curled arms spring (twisted square bar in wrought iron, round rod with knops in brass / bronze), each with a
back-scroll held by a collar and a spun drip pan with a candle cup. Behind the flames sits a dished, beaded
sheet-metal reflector (oval or shield, or none). Tallow candles burn down unevenly, lean a little, run with
wax drips that overflow the pans, and soot the reflector above every flame.
"""
from tk import *
import numpy as np

META = {"name": "Candle wall sconce", "class": "prop_small", "mount": "wall", "mount_height": 1.75,
        "focus": [0.0, -0.1, 0.11], "tri_budget": 8000}     # three arms with pans, candles and drips need a little more than 6k
PARAMS = {
    "arms": {"default": 2, "choices": [1, 2, 3], "desc": "number of candle arms"},
    "metal": {"default": "iron", "choices": ["iron", "brass", "bronze"], "desc": "metal of plate, arms and pans"},
    "reflector": {"default": "oval", "choices": ["oval", "shield", "none"], "desc": "reflector back plate shape"},
    "plate": {"default": "brass", "choices": ["brass", "tin", "copper"], "desc": "sheet metal of the reflector"},
    "candle_height": {"default": 0.16, "min": 0.06, "max": 0.28, "desc": "height of a fresh candle (m)"},
    "reflector_height": {"default": 0.27, "min": 0.18, "max": 0.38, "desc": "reflector height (m)"},
    "reach": {"default": 0.17, "min": 0.12, "max": 0.24, "desc": "arm length from the boss to the candle (m)"},
    "spread": {"default": 42.0, "min": 25.0, "max": 65.0, "desc": "splay angle of the side arms (deg)"},
    "drop": {"default": 0.05, "min": 0.02, "max": 0.09, "desc": "depth of the arm's belly below the pan (m)"},
    "twist": {"default": 1.5, "min": 0.0, "max": 4.0, "desc": "turns of twist in the square iron bar (iron only)"},
    "burn": {"default": 0.3, "min": 0.0, "max": 0.7, "desc": "how unevenly the candles have burnt down (0 = all fresh)"},
    "drips": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "amount of wax drips and overflow"},
    "soot": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "soot on the reflector above the flames"},
    "wax": {"default": "#efe2c2", "desc": "wax colour"},
    "intensity": {"default": 0.24, "min": 0.15, "max": 2.5, "desc": "light per candle (cd); the wall is only a hand away, so keep it low"},
}
VARIANTS = {
    "brass_trio": {"arms": 3, "metal": "brass", "reflector": "shield", "plate": "brass", "candle_height": 0.2,
                   "reflector_height": 0.32, "reach": 0.19, "spread": 50, "burn": 0.2, "seed": 4},
    "iron_single": {"arms": 1, "reflector": "none", "candle_height": 0.13, "twist": 3.0, "drop": 0.07, "reach": 0.2,
                    "drips": 0.9, "seed": 11},
    "bronze_tall": {"arms": 2, "metal": "bronze", "plate": "copper", "reflector": "oval", "reflector_height": 0.36,
                    "candle_height": 0.26, "spread": 32, "wax": "#d9a648", "burn": 0.12, "seed": 23},
    "passage_stub": {"arms": 1, "reflector": "shield", "plate": "tin", "reflector_height": 0.22, "candle_height": 0.08,
                     "reach": 0.13, "drop": 0.03, "burn": 0.5, "drips": 1.0, "soot": 1.0, "twist": 0.0, "seed": 37},
}

Y0 = 0.012          # arms spring from this far in front of the wall (inside the boss)
PAN_R = 0.04        # drip pan radius
CUP_H = 0.022       # candle cup height above the pan floor
CR = 0.0105         # candle radius
BAR = 0.0085        # arm bar thickness


def _bez2(p0, p1, p2, p3, n):
    out = []
    for k in range(n):
        t = k / n
        s = 1.0 - t
        out.append(tuple(s ** 3 * p0[i] + 3 * s * s * t * p1[i] + 3 * s * t * t * p2[i] + t ** 3 * p3[i] for i in (0, 1)))
    return out                                            # end point excluded


def _outline(shape, w, h, n):
    """n outline points (x, z), counter-clockwise seen from the front, centred on the origin."""
    if shape == "oval":
        return [(0.5 * w * cos(2 * pi * k / n - pi / 2), 0.5 * h * sin(2 * pi * k / n - pi / 2)) for k in range(n)]
    q = n // 2
    nb, ns = q // 2, max(2, q // 5)
    nt = q - nb - ns
    side0, ear, peak = (0.5 * w, 0.0), (0.53 * w, 0.44 * h), (0.0, 0.5 * h)
    half = _bez2((0.0, -0.5 * h), (0.3 * w, -0.4 * h), (0.5 * w, -0.24 * h), side0, nb)
    half += [(lerp(side0[0], ear[0], k / ns), lerp(side0[1], ear[1], k / ns)) for k in range(ns)]
    half += _bez2(ear, (0.38 * w, 0.37 * h), (0.14 * w, 0.4 * h), peak, nt)
    return half + [peak] + [(-x, z) for (x, z) in reversed(half[1:])]


def _shear(obj, base, lx, ly):
    """Stand a part built on the z axis at `base`, leaning by the slopes lx, ly."""
    bx, by, bz = base
    transform_verts(obj, lambda co: Vector((co.x + lx * co.z + bx, co.y + ly * co.z + by, co.z + bz)))
    return obj


def build(p, rng):
    n_arms = int(p.arms)
    round_rod = p.metal != "iron"
    if p.metal == "iron":
        metal = mat("iron_wrought")
    elif p.metal == "brass":
        metal = mat("brass", rough=0.9)
    else:
        metal = mat("bronze_aged")
    plate_tint = {"brass": "#ffffff", "tin": "#e6d9c0", "copper": "#f2c9a0"}[p.plate]
    refl_mat = mat("brass", plate_tint)
    wax = mat("wax", p.wax)
    flame_mat = mat("emissive", "#ffb45e", glow=9)
    wick_mat = mat("plain", "#17120e")
    seed = int(p.seed)
    metal_objs, wax_objs, flames = [], [], []

    # ---- arm layout: (angle from straight ahead, reach, pan height)
    R = p.reach
    if n_arms == 1:
        layout = [(0.0, R, 0.03)]
    elif n_arms == 2:
        layout = [(-p.spread, R, 0.03), (p.spread, R, 0.03)]
    else:
        # keep the side pans clear of the centre one on short arms
        sp_min = degrees(np.arccos(clamp((2.21 - (2.0 * PAN_R + 0.012) ** 2 / (R * R)) / 2.2, -1.0, 1.0)))
        sp3 = max(p.spread, sp_min)
        layout = [(-sp3, R, 0.03), (0.0, R * 1.1, 0.06), (sp3, R, 0.03)]

    bar_prof = rounded_rect(BAR, BAR, 0.0018, 1)
    rod_r = 0.0048
    ft, fw = BAR * 0.3, BAR * 0.55                      # forged flat bar with crowned faces: thin radially, wide across
    flat_prof = [(-ft, -fw * 0.6), (0.0, -fw), (ft, -fw * 0.6), (ft, fw * 0.6), (0.0, fw), (-ft, fw * 0.6)]
    stem_prof = [(0.0, -0.036), (0.006, -0.0355), (0.0095, -0.031), (0.0085, -0.026), (0.0055, -0.022), (0.0055, -0.013),
                 (0.011, -0.008), (0.02, -0.004)]
    pan_prof = [(0.012, -0.0062), (0.022, -0.0045), (PAN_R - 0.005, -0.002), (PAN_R - 0.001, 0.003), (PAN_R + 0.0008, 0.008),
                (PAN_R - 0.0008, 0.0085), (PAN_R - 0.004, 0.003), (0.019, 0.0012), (0.0155, 0.002), (0.0148, CUP_H - 0.002),
                (0.0165, CUP_H), (0.013, CUP_H), (CR - 0.001, CUP_H - 0.002)]
    flame_prof = [(0.0, 0.0), (0.0035, 0.002), (0.0062, 0.008), (0.0052, 0.016), (0.0024, 0.026), (0.0, 0.034)]

    for ai, (ang, reach, zp) in enumerate(layout):
        r = rng.fork(f"arm{ai}")
        th = radians(ang + r.jitter(0.0, 2.0))
        sn, cs = sin(th), cos(th)
        nrm = (cs, sn, 0.0)                               # normal of the vertical plane the arm lies in

        def P3(rr, z, sn=sn, cs=cs):
            return (rr * sn, -Y0 - rr * cs, z)

        ze = zp - 0.034 + r.jitter(0.0, 0.003)            # arm end = foot of the pan stem
        zbel = min(ze, 0.0) - p.drop
        rz = [(0.004, 0.0), (0.03, -0.004 + 0.12 * zbel), (reach * 0.4, zbel * 0.7), (reach * 0.66, zbel),
              (reach * 0.9, lerp(zbel, ze, 0.3)), (reach, lerp(zbel, ze, 0.72)), (reach, ze)]
        dense = resample_path([(a, 0.0, b) for a, b in rz], n=64)
        # -- main arm
        path = [P3(q[0], q[2]) for q in dense]
        if round_rod:
            arm = sweep(f"arm{ai}", path, rod_r, sides=8, step=0.011, smooth_path=False, up=nrm,
                        scale_fn=lambda t: 1.12 - 0.3 * t)
        else:
            arm = sweep(f"arm{ai}", path, bar_prof, step=0.0065 if p.twist > 0.05 else 0.009, smooth_path=False, up=nrm,
                        scale_fn=lambda t: 1.08 - 0.22 * t, twist_deg=360.0 * p.twist)
        metal_objs.append(arm)
        # -- back scroll riding on the descending arm, held by a collar
        i0 = next(i for i, q in enumerate(dense) if q[0] >= reach * 0.34)
        a0, a1 = dense[i0], dense[i0 + 2]
        al = atan2(a1[2] - a0[2], a1[0] - a0[0])
        nx, nz = -sin(al), cos(al)
        off = (BAR if not round_rod else rod_r * 2.0) * 0.98
        r1 = clamp(reach * 0.15, 0.018, 0.032)
        cx, cz = a0[0] + nx * (off + r1), a0[2] + nz * (off + r1)
        turns = 1.35
        sp = spiral_path(0.0035, r1, turns, n=18, a0_deg=degrees(al) - 90.0 - 360.0 * turns, plane="XZ", center=(cx, 0, cz))
        tail = []
        for k in (3, 6, 9):
            q = dense[min(i0 + k, len(dense) - 1)]
            tail.append((q[0] + nx * off, 0.0, q[2] + nz * off))
        spath = [P3(q[0], q[2]) for q in sp + tail]
        if round_rod:
            sc = sweep(f"scroll{ai}", spath, rod_r * 0.85, sides=5, smooth_path=False, up=nrm,
                       scale_fn=lambda t: 0.4 + 0.6 * min(1.0, t * 1.5))
        else:
            sc = sweep(f"scroll{ai}", spath, flat_prof, smooth_path=False, up=nrm,
                       scale_fn=lambda t: 0.4 + 0.6 * min(1.0, t * 1.5))
        metal_objs.append(sc)
        qc = dense[min(i0 + 7, len(dense) - 1)]
        cc = P3(qc[0] + nx * off * 0.5, qc[2] + nz * off * 0.5)
        metal_objs.append(box(f"collar{ai}", (off * 1.75, 0.011, off * 2.5), loc=cc, rot=(-degrees(al), 0, degrees(th)), bevel=0.0012))
        # tight forged curl tucked under the pan on the outside of the arm
        zc2 = lerp(zbel, ze, 0.5)
        r2 = clamp(p.drop * 0.3, 0.008, 0.02)
        sp2 = spiral_path(0.003, r2, 1.2, n=12, a0_deg=180.0 + 360.0 * 1.2, plane="XZ", center=(reach + off * 0.9 + r2, 0, zc2), ccw=False)
        sp2 += [(reach + off * 0.9, 0.0, zc2 + 0.006), (reach + off * 0.9, 0.0, min(ze - 0.002, zc2 + 0.014))]
        s2path = [P3(q[0], q[2]) for q in sp2]
        metal_objs.append(sweep(f"curl{ai}", s2path, rod_r * 0.8 if round_rod else flat_prof,
                                sides=6, smooth_path=False, up=nrm, scale_fn=lambda t: 0.4 + 0.6 * min(1.0, t * 1.6)))
        if round_rod:
            # turned knops on the rod
            for f in (0.56,):
                q = dense[int(f * (len(dense) - 1))]
                metal_objs.append(sphere(f"knop{ai}", rod_r * 1.75, loc=P3(q[0], q[2]), segments=8, rings=5))

        # -- drip pan + candle cup
        base = P3(reach, zp)
        pan = lathe(f"pan{ai}", pan_prof, segments=20, loc=base, rot=(0, 0, r.u(0, 360)))
        metal_objs.append(lathe(f"stem{ai}", stem_prof, segments=8, loc=base))
        paint(pan, grime=mask_mul(facing((0, 0, 1)), 0.7), special=mask_mul(noise_mask(0.03, seed=seed + ai, lo=0.4, hi=0.75), 0.6))
        metal_objs.append(pan)

        # -- candle: uneven burn, slight lean, melted rim
        h = max(0.035, p.candle_height * (1.0 - p.burn * r.u(0.0, 1.0)))
        top = 0.006 + h
        lx, ly = r.jitter(0.0, 0.035), r.jitter(0.0, 0.03)
        melt_a = r.u(0, 2 * pi)
        melt = 0.0015 + 0.004 * p.drips
        cprof = [(CR, 0.006), (CR * 1.02, 0.006 + h * 0.45), (CR, top - 0.006), (CR - 0.0002, top - 0.0015), (CR - 0.0016, top),
                 (CR - 0.004, top - 0.0018), (0.003, top - 0.0045), (0.0, top - 0.0045)]
        cand = lathe(f"candle{ai}", cprof, segments=12)

        def melt_fn(co, top=top, melt_a=melt_a, melt=melt):
            if co.z > top - 0.008 and (co.x * co.x + co.y * co.y) > 1e-8:
                a = atan2(co.y, co.x)
                co = Vector((co.x, co.y, co.z - melt * (0.5 + 0.5 * cos(a - melt_a))))
            return co

        transform_verts(cand, melt_fn)
        _shear(cand, base, lx, ly)
        paint(cand, grime=mask_mul(height_ramp(base[2] + top + 0.002, base[2] + top - 0.03), 0.55))
        wax_objs.append(cand)

        def C(x, y, z, base=base, lx=lx, ly=ly):
            return (base[0] + x + lx * z, base[1] + y + ly * z, base[2] + z)

        if p.drips > 0.02:
            # wax runs down the candle from the low side of the rim
            nd = int(round(p.drips * r.u(2.5, 5.5)))
            for k in range(nd):
                a = melt_a + r.jitter(0.0, 1.5)
                ln = min(top - CUP_H - 0.004, r.u(0.018, 0.03 + 0.09 * p.drips))
                if ln < 0.01:
                    continue
                rr = CR + 0.0009
                zt = top - melt * (0.5 + 0.5 * cos(a - melt_a)) - 0.0005
                da = r.jitter(0.0, 0.12)
                pts = [C((rr - 0.003) * cos(a), (rr - 0.003) * sin(a), zt + 0.0002), C(rr * cos(a), rr * sin(a), zt - 0.003),
                       C(rr * cos(a + da * 0.5), rr * sin(a + da * 0.5), zt - ln * 0.55), C(rr * cos(a + da), rr * sin(a + da), zt - ln)]
                w = r.u(0.0017, 0.0026)
                wax_objs.append(sweep(f"drip{ai}_{k}", pts, w, sides=5, step=ln / 4.0,
                                      scale_fn=lambda t: 0.7 + 0.75 * sin(pi * t ** 2.2)))
            # collar of set wax over the cup rim and a puddle in the pan
            if p.drips > 0.25:
                sk = lathe(f"skirt{ai}", [(CR - 0.001, CUP_H - 0.003), (CR + 0.0042, CUP_H - 0.0015), (CR + 0.0062, CUP_H + 0.0015),
                                           (CR + 0.003, CUP_H + 0.006), (CR - 0.0005, CUP_H + 0.011)], segments=12)
                displace_noise(sk, 0.0011, 0.011, seed=seed + ai)
                _shear(sk, base, lx, ly)
                wax_objs.append(sk)
                pa = melt_a + r.jitter(0.0, 0.6)
                wax_objs.append(sphere(f"puddle{ai}", 0.0095 * (0.6 + 0.6 * p.drips), loc=C(0.024 * cos(pa), 0.024 * sin(pa), 0.0022),
                                       segments=8, rings=4, scale=(1.0, r.u(0.7, 1.3), 0.22)))
            # overflow: wax that crept over the pan rim and froze as it fell
            for k in range(int(round(p.drips * r.u(0.4, 2.6)))):
                a = melt_a + r.jitter(0.0, 1.1)
                ln = r.u(0.008, 0.012 + 0.03 * p.drips)
                ca, sa = cos(a), sin(a)
                pts = [(base[0] + (PAN_R - 0.009) * ca, base[1] + (PAN_R - 0.009) * sa, base[2] + 0.0035),
                       (base[0] + (PAN_R - 0.001) * ca, base[1] + (PAN_R - 0.001) * sa, base[2] + 0.0092),
                       (base[0] + (PAN_R + 0.0022) * ca, base[1] + (PAN_R + 0.0022) * sa, base[2] + 0.004),
                       (base[0] + (PAN_R + 0.0012) * ca, base[1] + (PAN_R + 0.0012) * sa, base[2] - ln * 0.5),
                       (base[0] + (PAN_R + 0.001) * ca, base[1] + (PAN_R + 0.001) * sa, base[2] - ln)]
                wax_objs.append(sweep(f"overflow{ai}_{k}", pts, 0.0021, sides=4, step=(ln + 0.02) / 6.0,
                                      scale_fn=lambda t: 1.15 - 0.75 * t ** 1.5))

        # -- wick, flame mesh, effect + light
        wk = cylinder(f"wick{ai}", 0.0009, 0.011, loc=(0, 0, top + 0.0005), verts=5)
        _shear(wk, base, lx + 0.12, ly)
        assign(wk, wick_mat)
        fl = lathe(f"flame{ai}", flame_prof, segments=6)
        fpos = C(0.0, 0.0, top + 0.003)
        _shear(fl, fpos, 0.0, 0.0)
        assign(fl, flame_mat)
        fc = (fpos[0], fpos[1], fpos[2] + 0.016)
        flames.append(fc)
        fx(f"flame{ai}", "flame", (fpos[0], fpos[1], fpos[2] + 0.002), size=0.03)

    # ---- strap back plate with spear ends, boss, screws
    zb = -0.115
    has_ref = p.reflector != "none"
    rh = p.reflector_height
    flame_z = sum(f[2] for f in flames) / len(flames)
    # one light for the whole fitting, a hand in front of the flames (the wall is too close for a light per candle)
    light("candles", (0.0, min(f[1] for f in flames) - 0.26, flame_z + 0.05), color="#ffb45e",
          intensity=p.intensity * n_arms ** 0.7, range=4.5, flicker=0.5, shadow_near=0.35)
    zc = clamp(flame_z - 0.015, 0.5 * rh + 0.04, 0.5 * rh + 0.11)
    zt = zc if has_ref else 0.12
    hw = 0.016
    right = [(0.0, zb), (hw * 0.55, zb + 0.016), (hw * 1.05, zb + 0.036), (hw * 0.5, zb + 0.05), (hw, zb + 0.064), (hw, zt - 0.064),
             (hw * 0.5, zt - 0.05), (hw * 1.05, zt - 0.036), (hw * 0.55, zt - 0.016), (0.0, zt)]
    outline = right + [(-x, z) for (x, z) in reversed(right[1:-1])]
    strap = shape_extrude("strap", outline, depth=0.004, rot=(90, 0, 0), bevel=0.0008)
    displace_noise(strap, 0.0006, 0.02, seed=seed)
    metal_objs.append(strap)
    boss = lathe("boss", [(0.027, 0.0), (0.0285, 0.003), (0.0225, 0.0065), (0.0195, 0.0105), (0.0225, 0.016), (0.0215, 0.024),
                          (0.015, 0.031), (0.006, 0.0345), (0.0, 0.035)], segments=16, loc=(0, -0.0035, 0), rot=(90, 0, 0))
    metal_objs.append(boss)
    screw_z = [zb + 0.07] + ([] if has_ref else [zt - 0.07])
    sb = bolt("screw", 0.0052, 0.003, kind="dome")
    metal_objs.append(instances_to_mesh("screws", sb, [trs((0, -0.004, z), (90, 0, 0)) for z in screw_z], keep_base=False))

    # ---- reflector: dished sheet with a pressed ring, rolled rim bead and a fixing rivet
    if has_ref:
        w = rh * (0.66 if p.reflector == "oval" else 0.74)
        NU, NV = (40 if p.reflector == "oval" else 32), 6
        ol = _outline(p.reflector, w, rh, NU)
        rads = [0.0, 0.3, 0.36, 0.43, 0.72, 0.92, 1.0]
        dish = 0.024 * rh / 0.27
        ridge = [0.0, 0.0, 0.0035, 0.0, 0.0, 0.0, 0.0015]
        yb = 0.0058

        def ref_y(j):
            rr = max(rads[j], rads[1])
            return -(yb + dish * (rr * rr - rads[1] ** 2) + ridge[j])

        # the oval is gadrooned: radial flutes pressed into the outer band; the shield is planished plain
        n_fl = NU // 4 if p.reflector == "oval" else 0
        fl_amp = [0.0, 0.0, 0.0, 0.0, 0.0032, 0.0022, 0.0]

        def ref_fn(u, v):
            i = (NU - int(round(u * NU))) % NU
            j = int(round(v * NV))
            fy = fl_amp[j] * cos(2 * pi * n_fl * i / NU) if n_fl else 0.0
            return (ol[i][0] * rads[j], ref_y(j) - fy, zc + ol[i][1] * rads[j])

        ref = surface("reflector", ref_fn, NU, NV, closed_u=True, thickness=0.0016)
        weld(ref, 1e-5)
        if p.reflector == "shield":
            displace_noise(ref, 0.0012, 0.03, seed=seed + 9, axis_mask=(0, 1, 0))     # planished by hand
        uv_planar(ref, (1, 0, 0), (0, 0, 1))
        smooth(ref, 50)
        y_rim = ref_y(NV)
        rim = sweep("rim", [(x, y_rim, zc + z) for (x, z) in ol], 0.0032, closed=True, sides=5, smooth_path=False, up=(0, 1, 0))
        rivet = bolt("rivet", 0.0065, 0.004, loc=(0, ref_y(0), zc), rot=(90, 0, 0), kind="dome")
        # hanging loop riveted to the top of the sheet, on a forged nail
        ztop = zc + max(z for (_, z) in ol)
        loop = torus("hang_loop", 0.008, 0.0022, loc=(0, y_rim + 0.001, ztop + 0.0065), rot=(90, 0, 0), major_seg=12, minor_seg=5)
        assign(loop, refl_mat)
        metal_objs.append(bolt("hang_nail", 0.0042, 0.004, loc=(0, y_rim - 0.0035, ztop + 0.0105), rot=(90, 0, 0), kind="pyramid"))
        metal_objs.append(cylinder("hang_shank", 0.0017, -y_rim + 0.0035, loc=(0, (y_rim - 0.0035) * 0.5, ztop + 0.0105), rot=(90, 0, 0), verts=5, cap=False))
        fl_arr = np.array(flames)
        soot = float(p.soot)

        def soot_fn(co, nrm_):
            g = np.zeros(len(co))
            for f in fl_arr:
                dx = (co[:, 0] - f[0]) / 0.045
                up = np.clip((co[:, 2] - (f[2] - 0.01)) / 0.07, 0.0, 1.0)
                fade = np.clip(1.0 - (co[:, 2] - f[2]) / 0.3, 0.0, 1.0)
                g = np.maximum(g, np.exp(-dx * dx) * up * fade)
            return np.clip(g * soot * 1.2, 0.0, 1.0)

        def polish_fn(co, nrm_):
            d = np.hypot(co[:, 0] / (0.5 * w), (co[:, 2] - zc) / (0.5 * rh))
            return 0.35 * np.clip(1.0 - d * 0.9, 0.0, 1.0)

        for o in (ref, rim, rivet):
            assign(o, refl_mat)
            paint(o, grime=soot_fn, wear=polish_fn,
                  special=mask_mul(height_ramp(zc - 0.5 * rh, zc - 0.1 * rh), noise_mask(0.035, seed=seed + 5, lo=0.35, hi=0.7), 0.55))

    # ---- materials + ageing
    for o in metal_objs:
        assign(o, metal)
        if o.name.startswith("pan"):
            continue
        if p.metal == "iron":
            # rust where water sits: joints, the foot of the plate, the belly of the arms
            paint(o, special=mask_mul(noise_mask(0.03, seed=seed + 2, lo=0.35, hi=0.7), height_ramp(zb - 0.02, 0.06), 0.8),
                  wear=mask_mul(facing((0, -0.4, 1)), noise_mask(0.02, seed=seed + 3), 0.5))
        else:
            # verdigris in the low, damp spots, hands and cloths polish the tops
            paint(o, special=mask_mul(noise_mask(0.03, seed=seed + 2, lo=0.4, hi=0.75), height_ramp(zb - 0.02, 0.05), 0.7),
                  wear=mask_mul(facing((0, -0.5, 1)), 0.7), grime=mask_mul(facing((0, 0, -1)), 0.4))
    for o in wax_objs:
        assign(o, wax)
