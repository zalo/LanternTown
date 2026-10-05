"""Old cast-iron pillar fire hydrant / standpipe. A flanged foot bolted down onto a two-piece stone
collar, a slightly tapering (plain or fluted) barrel with a bead band and raised cast lettering, a
bolted flange joint with a gasket line, a bulged nozzle section carrying one to three hose outlets
(brass threaded nozzles under lugged iron caps kept on short chains), and a ribbed dome bonnet with
a stuffing gland and the bare pentagon operating nut. Thick chipped paint, rust weeping from the
joint, the outlets and the base.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, sqrt
from mathutils import Matrix, Vector

META = {"name": "Fire hydrant standpipe", "class": "prop_small", "mount": "ground", "focus": 0.66,
        "tri_budget": 10000}   # three chained outlets + cast lettering + 24 fixings on a close-up hero prop
PARAMS = {
    "height": {"default": 0.80, "min": 0.64, "max": 1.15, "desc": "overall height to the top of the operating nut (m)"},
    "outlets": {"default": 2, "choices": [1, 2, 3], "desc": "1 = front outlet, 2 = two side hose outlets, 3 = sides + large front pumper outlet"},
    "paint": {"default": "#8f1f15", "desc": "body paint colour"},
    "cap_paint": {"default": "", "desc": "paint colour of bonnet and outlet caps ('' = same as the body)"},
    "barrel": {"default": 0.175, "min": 0.14, "max": 0.22, "desc": "barrel diameter (m)"},
    "style": {"default": "plain", "choices": ["plain", "fluted"], "desc": "barrel casting"},
    "bolts": {"default": 6, "min": 4, "max": 8, "desc": "bolts per flange"},
    "lettering": {"default": "F.H", "desc": "raised cast lettering on the barrel ('' = none)"},
    "font": {"default": "sans", "choices": ["sans", "sans_narrow", "serif", "roman"], "desc": "lettering face"},
    "wear": {"default": 0.55, "min": 0.0, "max": 1.0, "desc": "condition: 0 = freshly painted, 1 = neglected (chips, rust runs)"},
    "pad": {"default": True, "desc": "two-piece dressed stone collar set into the paving"},
    "chains": {"default": True, "desc": "cap retaining chains"},
    "badge": {"default": True, "desc": "riveted enamel number plate on the nozzle section"},
    "number": {"default": "7", "desc": "hydrant number on the enamel plate"},
    "letter_paint": {"default": "#d6cbaa", "desc": "paint picking out the cast lettering and the plate ('' = body colour)"},
    "missing_cap": {"default": False, "desc": "one outlet cap is gone: bare brass thread, chain left dangling"},
}
VARIANTS = {
    "green_pumper": {"outlets": 3, "height": 0.92, "style": "fluted", "paint": "#1f4a33", "cap_paint": "#b58a22",
                     "lettering": "W.W", "number": "12", "bolts": 7, "wear": 0.45, "seed": 4},
    "squat_yellow": {"outlets": 1, "height": 0.66, "barrel": 0.21, "paint": "#b9861c", "cap_paint": "#7d1a12",
                     "lettering": "W", "number": "3", "pad": False, "wear": 0.4, "seed": 9},
    "tall_standpipe": {"outlets": 2, "height": 1.1, "barrel": 0.15, "style": "fluted", "paint": "#2b4a68",
                       "cap_paint": "#c8bfa4", "lettering": "S.P", "number": "21", "bolts": 5, "wear": 0.3, "seed": 15},
    "neglected": {"outlets": 2, "height": 0.74, "paint": "#8c9087", "cap_paint": "#6a2a1e", "wear": 1.0, "missing_cap": True,
                  "lettering": "F.H", "number": "4", "letter_paint": "", "seed": 23},
}


def _link_chain(name, pts, link_len=0.030, wire=0.0032, rng=None):
    """Light chain (48 tris / link): oval links alternating 90 deg along the path."""
    P = [Vector(q) for q in resample_path(pts, step=link_len * 0.72, smooth_path=False)]
    hl, hw = link_len / 2 - wire, link_len * 0.3 - wire
    loop = [(hl, 0, 0), (hl * 0.5, hw, 0), (-hl * 0.5, hw, 0), (-hl, 0, 0), (-hl * 0.5, -hw, 0), (hl * 0.5, -hw, 0)]
    base = sweep(name + "_link", loop, wire, closed=True, sides=4, smooth_path=False)
    mats = []
    n = len(P)
    for i in range(n):
        t = (P[min(i + 1, n - 1)] - P[max(i - 1, 0)]).normalized()
        b = Vector((0, 0, 1)).cross(t)
        if b.length < 1e-3:
            b = Vector((0, 1, 0)).cross(t)
        b.normalize()
        nn = t.cross(b)
        if i % 2:
            b, nn = nn, -b
        if rng is not None:                      # links never hang perfectly square
            a = radians(rng.jitter(0.0, 14.0))
            b, nn = b * cos(a) + nn * sin(a), nn * cos(a) - b * sin(a)
        mats.append(Matrix(((t.x, b.x, nn.x, P[i].x), (t.y, b.y, nn.y, P[i].y), (t.z, b.z, nn.z, P[i].z), (0, 0, 0, 1))))
    o = instances_to_mesh(name, base, mats, uv_jitter=False, keep_base=False)
    smooth(o, 60)
    return o


def _hex(name, r, h, stud=0.0):
    """Cheap chamfered hex head / nut (optionally with the stud end showing), base on z=0."""
    prof = [(r, 0.0), (r, h * 0.82), (r * 0.84, h)]
    if stud > 0:
        prof += [(r * 0.48, h), (r * 0.48, h + stud * 0.8), (r * 0.36, h + stud), (0.0, h + stud)]
    else:
        prof += [(0.0, h)]
    return lathe(name, prof, segments=6, smooth_angle=30)


def build(p, rng):
    H = p.height
    R = p.barrel / 2
    W = p.wear
    n_out = int(p.outlets)
    nb = int(p.bolts)
    body_m = mat("iron_painted", p.paint)
    cap_m = mat("iron_painted", p.cap_paint or p.paint)
    brass_m = mat("brass")
    letter_m = mat("iron_painted", p.letter_paint or p.paint)
    enamel_m = mat("ceramic_glazed", p.letter_paint or "#d6cbaa")
    stain_m = mat("iron_painted", "#7c4a25", rough=1.7)      # rust-stained paint film (iron_rusty reads black at night)
    bare_m = mat("iron_wrought")
    rust_m = mat("iron_rusty")
    stone_m = mat("stone_rough", "#a39c8e")

    body, caps, brass, bare, rusty, stones, letters = [], [], [], [], [], [], []

    # ------------------------------------------------------------------ stone collar in the paving
    zb = 0.0
    Rf = R + 0.060                                   # foot flange radius
    if p.pad:
        zb = 0.045
        ps = 2 * Rf + 0.17
        r = rng.fork("pad")
        for i, sx in enumerate((-1, 1)):
            hgt = zb - (0.0005 if i == 0 else r.u(0.002, 0.005))
            st = stone_block(f"pad{i}", (ps / 2 - 0.004, ps + r.jitter(0.0, 0.012), hgt),
                             loc=(sx * (ps / 4 + 0.001), r.jitter(0.0, 0.006), hgt / 2), rot=(0, 0, r.jitter(0.0, 0.8)),
                             chamfer=0.01, rough=0.003, rng=r, detail=0.16)
            realize(st)
            apply_transform(st)
            transform_verts(st, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
            stones.append(st)

    # ------------------------------------------------------------------ levels
    zb0 = zb + 0.108                                 # barrel starts above the foot mouldings
    zd1 = H - 0.070                                  # dome top / gland seat
    zd0 = H - 0.150                                  # bonnet rim
    zj = H - 0.340                                   # flange joint (gasket plane)
    zo = zj + 0.108                                  # outlet axis height
    zb1 = zj - 0.040                                 # barrel ends below the joint cove
    Lb = max(zb1 - zb0, 0.05)
    Rj = R + 0.044                                   # joint flange radius
    Rh = R + 0.024                                   # nozzle section radius
    Rt = R - 0.006                                   # barrel radius at the top (cast taper)

    def rb(z):
        return R + (Rt - R) * clamp((z - zb0) / Lb)

    # ------------------------------------------------------------------ foot + barrel + lower flange
    f0 = zb - 0.004 if p.pad else 0.0
    prof = [(Rf - 0.004, f0), (Rf, f0 + 0.005), (Rf, zb + 0.019), (Rf - 0.004, zb + 0.023),
            (R + 0.034, zb + 0.024), (R + 0.026, zb + 0.031), (R + 0.022, zb + 0.046),
            (R + 0.026, zb + 0.052), (R + 0.028, zb + 0.060), (R + 0.026, zb + 0.068), (R + 0.015, zb + 0.076),
            (R + 0.004, zb + 0.092), (R, zb0)]
    tz = clamp(Lb * 0.5, 0.045, 0.10)                # plain drum that carries the lettering
    z_band = zb1 - tz
    lo_top = z_band - 0.012
    fluted = p.style == "fluted" and lo_top - zb0 > 0.08
    if fluted:                                       # dense rows only where the flutes run out
        zs = [zb0 + 0.012, zb0 + 0.028]
        nmid = max(1, int(round((lo_top - zb0 - 0.056) / 0.07)))
        zs += [zb0 + 0.028 + (lo_top - zb0 - 0.056) * i / nmid for i in range(1, nmid)]
        zs += [lo_top - 0.028, lo_top - 0.012, lo_top]
    else:
        nrow = max(2, int(round((lo_top - zb0) / 0.045)))
        zs = [zb0 + (lo_top - zb0) * i / nrow for i in range(1, nrow + 1)]
    for z in zs:
        prof.append((rb(z), z))
    prof += [(rb(z_band) + 0.006, z_band - 0.007), (rb(z_band) + 0.0085, z_band), (rb(z_band) + 0.006, z_band + 0.007),
             (rb(z_band + 0.012), z_band + 0.012)]
    prof += [(rb((z_band + 0.012 + zb1) / 2), (z_band + 0.012 + zb1) / 2), (Rt, zb1),
             (Rt + 0.004, zj - 0.030), (Rt + 0.013, zj - 0.023), (Rj - 0.003, zj - 0.021), (Rj, zj - 0.018),
             (Rj, zj - 0.004), (Rj - 0.005, zj - 0.002), (Rj - 0.005, zj + 0.001)]
    nf = 8
    barrel = lathe("barrel", prof, segments=32 if fluted else 24, smooth_angle=45)
    if fluted:
        fz0, fz1 = zb0 + 0.002, lo_top - 0.002

        def flute(co):
            if co.z < fz0 or co.z > fz1:
                return co
            e = min(smoothstep(fz0, fz0 + 0.026, co.z), 1.0 - smoothstep(fz1 - 0.026, fz1, co.z))
            a = atan2(co.y, co.x)
            c = 0.5 + 0.5 * cos(a * nf)
            d = 0.0065 * e * (1.0 - c ** 3)
            rr = hypot(co.x, co.y)
            k = (rr - d) / rr if rr > 1e-6 else 1.0
            return Vector((co.x * k, co.y * k, co.z))

        transform_verts(barrel, flute)
    body.append(barrel)

    # ------------------------------------------------------------------ nozzle section + bonnet + gland
    zh0, zh1 = zj + 0.042, zd0 - 0.014
    hp = [(Rj - 0.005, zj + 0.001), (Rj - 0.005, zj + 0.003), (Rj, zj + 0.005), (Rj, zj + 0.020), (Rj - 0.003, zj + 0.023),
          (R + 0.017, zj + 0.025), (R + 0.009, zj + 0.032), (R + 0.006, zh0)]
    for i in range(1, 6):
        t = i / 5
        hp.append((R + 0.006 + (Rh - R - 0.006) * sin(min(1.0, t * 2.2) * pi / 2), lerp(zh0, zh1, t)))
    head = lathe("head", hp, segments=24, smooth_angle=45)
    body.append(head)
    dome_r0, dome_r1 = Rh - 0.004, 0.031

    def dome_pt(t):                                   # meridian of the dome, t 0..1
        a = t * pi / 2 * 0.93
        return (dome_r1 + (dome_r0 - dome_r1) * cos(a) / 1.0, zd0 + 0.008 + (zd1 - zd0 - 0.008) * sin(a) / sin(pi / 2 * 0.93))

    bp = [(Rh, zh1), (Rh + 0.008, zd0 - 0.010), (Rh + 0.0105, zd0 - 0.004), (Rh + 0.008, zd0 + 0.002), (Rh, zd0 + 0.006)]
    bp += [dome_pt(i / 5) for i in range(6)]
    bp += [(0.035, zd1 + 0.003), (0.037, zd1 + 0.009), (0.037, zd1 + 0.017), (0.028, zd1 + 0.020), (0.026, zd1 + 0.029),
           (0.030, zd1 + 0.032), (0.030, zd1 + 0.036), (0.0, zd1 + 0.036)]
    bonnet = lathe("bonnet", bp, segments=24, smooth_angle=45)
    caps.append(bonnet)
    # cast ribs up the dome
    rib_prof = [(-0.007, -0.006), (0.007, -0.006), (0.004, 0.0075), (-0.004, 0.0075)]
    a_rib = rng.u(0, 60)
    for k in range(6):
        a = radians(a_rib + k * 60)
        ca, sa = cos(a), sin(a)
        path = []
        for i in range(7):
            rr, z = dome_pt(0.04 + 0.9 * i / 6)
            path.append(((rr + 0.0005) * ca, (rr + 0.0005) * sa, z))
        caps.append(sweep(f"rib{k}", path, rib_prof, up=(ca, sa, 0), smooth_path=False, scale_fn=lambda t: 1.0 - 0.3 * t))
    # pentagon operating nut, rubbed bare by the hydrant key
    nut = extrude_profile("op_nut", circle_pts(0.0185, 5, a0=rng.u(0, 72)), 0.034, loc=(0, 0, H - 0.034), bevel=0.002, segments=1)
    transform_verts(nut, lambda co: Vector((co.x * (1 - 2.2 * co.z), co.y * (1 - 2.2 * co.z), co.z)))   # slight taper
    bare.append(nut)

    # ------------------------------------------------------------------ fixings
    base_nut = _hex("base_nut", 0.0115, 0.011, stud=0.006)
    a0 = rng.u(0, 360 / nb)
    rbolt = R + 0.047
    rusty.append(instances_to_mesh("base_nuts", base_nut,
                                   [trs((rbolt * cos(radians(a0 + k * 360 / nb)), rbolt * sin(radians(a0 + k * 360 / nb)), zb + 0.0225),
                                        (rng.jitter(0, 2), rng.jitter(0, 2), rng.u(0, 60))) for k in range(nb)], keep_base=False))
    j_head = _hex("joint_head", 0.0105, 0.009)
    j_nut = _hex("joint_nut", 0.0105, 0.010)
    a1 = a0 + 180 / nb
    rj = Rj - 0.0135
    mh, mn = [], []
    for k in range(nb):
        a = radians(a1 + k * 360 / nb)
        mh.append(trs((rj * cos(a), rj * sin(a), zj + 0.0225), (0, 0, rng.u(0, 60))))
        mn.append(trs((rj * cos(a), rj * sin(a), zj - 0.0195), (180, 0, rng.u(0, 60))))
    rusty.append(instances_to_mesh("joint_heads", j_head, mh, keep_base=False))
    rusty.append(instances_to_mesh("joint_nuts", j_nut, mn, keep_base=False))

    # ------------------------------------------------------------------ outlets
    if n_out == 1:
        outs = [(-90.0, 0.043, 0.070)]
    elif n_out == 2:
        outs = [(0.0, 0.040, 0.072), (180.0, 0.040, 0.072)]
    else:
        outs = [(0.0, 0.038, 0.072), (180.0, 0.038, 0.072), (-90.0, 0.050, 0.060)]
    miss = len(outs) - 1 if n_out != 3 else 1         # which cap is gone (if any)
    for k, (az, rn, ln) in enumerate(outs):
        r = rng.fork(f"outlet{k}")
        d = Vector((cos(radians(az)), sin(radians(az)), 0.0))
        capped = not (p.missing_cap and k == miss)
        d0 = sqrt(max(Rh * Rh - (rn + 0.016) ** 2, 1e-4)) - 0.004
        e = Rh + ln                                   # end of the iron nozzle
        neck = lathe("neck", [(rn + 0.016, d0), (rn + 0.015, Rh + 0.003), (rn + 0.008, Rh + 0.012), (rn + 0.004, Rh + 0.026),
                              (rn + 0.004, e - 0.021), (rn + 0.009, e - 0.018), (rn + 0.009, e - 0.008), (rn + 0.004, e - 0.006),
                              (rn - 0.004, e - 0.006)], segments=14, smooth_angle=45)
        assign(neck, body_m)
        tl = 0.018 if capped else 0.034               # visible length of brass thread
        tp = [(rn + 0.002, e - 0.008)]
        nth = 3 if capped else 5
        for i in range(nth):
            z0 = e - 0.006 + tl * i / nth
            tp += [(rn + 0.002, z0), (rn - 0.0015, z0 + tl / nth * 0.5)]
        tp += [(rn + 0.0015, e - 0.006 + tl)]
        if not capped:
            tp += [(rn - 0.006, e - 0.006 + tl), (rn - 0.007, e - 0.03), (0.0, e - 0.03)]
        thread = lathe("thread", tp, segments=14, smooth_angle=30)
        assign(thread, brass_m)
        parts_b, parts_c = [neck], []
        rc = rn + 0.010
        c0 = e + 0.011
        if capped:
            cap = lathe("cap", [(rn + 0.001, c0), (rc, c0 + 0.003), (rc, c0 + 0.027), (rc - 0.004, c0 + 0.032), (rn * 0.58, c0 + 0.037),
                                (rn * 0.5, c0 + 0.042), (0.0, c0 + 0.042)], segments=14, smooth_angle=45)
            spin = r.u(0, 90)
            cnut = extrude_profile("cap_nut", circle_pts(0.015, 5, a0=spin), 0.020, loc=(0, 0, c0 + 0.0415), bevel=0.0015, segments=1)
            parts_c += [cap, cnut]
            for i in range(4):                        # spanner lugs
                a = radians(spin + 45 + i * 90)
                parts_c.append(box("lug", (0.013, 0.015, 0.024), loc=((rc + 0.003) * cos(a), (rc + 0.003) * sin(a), c0 + 0.015),
                                   rot=(0, 0, degrees(a)), bevel=0.0025, segments=1))
            if p.chains:                              # chain eye under the cap (local +X is down after placing)
                eye = torus("cap_eye", 0.0075, 0.0026, loc=(rc + 0.005, 0, c0 + 0.014), rot=(90, 0, 0), major_seg=8, minor_seg=4)
                parts_c.append(eye)
            for o in parts_c:
                assign(o, cap_m)
        M = (0, 90, az)
        for grp, lst, nm in ((parts_b, body, "nozzle"), ([thread], brass, "thread"), (parts_c, caps, "cap")):
            if not grp:
                continue
            for o in grp:
                realize(o)
                apply_transform(o)
            j = join(grp, f"{nm}{k}") if len(grp) > 1 else grp[0]
            place(j, loc=(0, 0, zo), rot=M)
            lst.append(j)
        if p.chains:
            # cast ear on the joint flange rim with a ring, chain up to the eye under the cap
            ear = box(f"ear{k}", (0.016, 0.012, 0.016), loc=(d.x * (Rj + 0.002), d.y * (Rj + 0.002), zj + 0.013), rot=(0, 0, az), bevel=0.003, segments=1)
            body.append(ear)
            ring = torus(f"ear_ring{k}", 0.0085, 0.0026, loc=(d.x * (Rj + 0.013), d.y * (Rj + 0.013), zj + 0.008), rot=(90, 0, az), major_seg=8, minor_seg=4)
            bare.append(ring)
            pb = Vector((d.x * (Rj + 0.015), d.y * (Rj + 0.015), zj + 0.001))
            if capped:
                pc = Vector((d.x * (c0 + 0.014), d.y * (c0 + 0.014), zo - rc - 0.012))
                pts = catenary(pb, pc, sag=r.u(0.035, 0.06), n=20)
            else:                                     # cap stolen: the chain hangs slack against the barrel
                ln_c = r.u(0.10, 0.15)
                pts = [pb + Vector((0, 0, -ln_c * i / 8)) + d * (0.004 * sin(i * 0.7)) for i in range(9)]
            bare.append(_link_chain(f"chain{k}", [tuple(q) for q in pts], rng=r))

    # ------------------------------------------------------------------ raised cast lettering wrapped onto the drum
    d_top = zb1 - 0.004
    d_bot = z_band + 0.014
    if p.lettering.strip() and d_top - d_bot > 0.03:
        size = min(0.056, (d_top - d_bot) * 0.74)
        zt = (d_top + d_bot) / 2
        rt = rb(zt)
        t = text("lettering", p.lettering.strip()[:6], size=size, depth=0.0065, loc=(0, 0, zt), font=p.font)
        realize(t)

        def wrap(co):                                  # local: x across, y up, z out of the face
            a = co.x / rt
            rr = rt - 0.0015 + co.z
            return Vector((rr * sin(a), co.y, rr * cos(a)))

        transform_verts(t, wrap)
        letters.append(t)

    # ------------------------------------------------------------------ enamel number plate riveted to the nozzle section
    plates, numbers, stains = [], [], []
    if p.badge:
        back = n_out != 2                              # the front is taken by an outlet
        sg = -1.0 if back else 1.0
        zc = zo + 0.022
        plate = lathe("badge", [(0.031, 0.0), (0.0305, 0.0025), (0.029, 0.0035), (0.0, 0.0035)], segments=16, smooth_angle=40)

        def wrap_b(co):
            a = co.x * 1.38 / Rh
            rr = Rh - 0.001 + co.z
            return Vector((sg * rr * sin(a), -sg * rr * cos(a), zc + co.y))

        transform_verts(plate, wrap_b)
        plates.append(plate)
        rv = bolt("rivet", 0.0038, 0.0028, kind="dome")
        rusty.append(instances_to_mesh("badge_rivets", rv, [
            trs((sg * (Rh + 0.002) * sin(sx * 0.034 / Rh), -sg * (Rh + 0.002) * cos(sx * 0.034 / Rh), zc),
                (90, 0, degrees(sx * 0.034 / Rh) + (180 if back else 0))) for sx in (-1, 1)], keep_base=False))
        if p.number.strip():
            num = text("number", p.number.strip()[:3], size=0.03 if len(p.number.strip()) < 3 else 0.022, depth=0.001,
                       loc=(0, 0, 0), rot=(0, 0, 0), font="sans")
            realize(num)
            transform_verts(num, lambda co: wrap_b(Vector((co.x, co.y, co.z + 0.0034))))
            numbers.append(num)

    # ------------------------------------------------------------------ rust runs
    ncol = 32 if fluted else 24
    colw = 2 * pi / ncol

    def snap(a):                                       # runs follow a column of barrel vertices
        return round(a / colw) * colw

    rr_ = rng.fork("rust")
    runs = []                                          # (azimuth, z_top, length, strength)
    for k in range(nb):                                # weeping from the joint bolts
        if rr_.chance(0.3 + 0.6 * W):
            runs.append((snap(radians(a1 + k * 360 / nb)), zj - 0.018, rr_.u(0.06, 0.2) * (0.5 + W), rr_.u(0.6, 1.0)))
    for (az, rn, ln) in outs:                          # drips under every outlet, on past the chain ear
        runs.append((snap(radians(az)), zo - rn - 0.004, 0.10, 1.0))
        runs.append((snap(radians(az)), zj - 0.018, rr_.u(0.12, 0.22) * (0.6 + W), 1.0))
    for k in range(3):                                 # from under the bonnet rim
        runs.append((snap(rr_.u(0, 2 * pi)), zd0 - 0.012, rr_.u(0.04, 0.08), 0.8))
    # the painted-iron shader only darkens, so the runs below the joint also get a thin skin of real rust
    pz = np.array([q[1] for q in prof])
    pr = np.array([q[0] for q in prof])
    z_lo = (lo_top + 0.003) if fluted else (zb0 + 0.01)
    for i, (a, zt, ln, s_) in enumerate(runs):
        if abs(zt - (zj - 0.018)) > 1e-6 or W < 0.08:
            continue
        z1 = zb1 + 0.007
        z0 = max(z1 - ln * (0.55 + 0.45 * s_), z_lo)
        if z1 - z0 < 0.03:
            continue
        zs_ = sorted(set([z0, z1] + [float(z) for z in pz if z0 < z < z1] + [z0 + (z1 - z0) * j / 6 for j in range(1, 6)]), reverse=True)
        w0 = rr_.u(0.005, 0.009) * (0.7 + 0.5 * W)
        ph = rr_.u(0, 6.28)
        vs, fs = [], []
        for j, z in enumerate(zs_):
            t = (z1 - z) / (z1 - z0)
            hw_ = w0 * (1.0 - t) ** 0.75 * (1.0 + 0.2 * sin(ph + t * 9.0)) + 0.0004
            rz = float(np.interp(z, pz, pr)) + 0.0007
            ac = a + 0.012 * t * sin(ph * 2 + t * 5.0)
            for c in (-1, 0, 1):
                aa = ac + c * hw_ / rz
                vs.append((rz * cos(aa) * (1 + 0.004 * (c == 0)), rz * sin(aa) * (1 + 0.004 * (c == 0)), z))
            if j:
                b0 = (j - 1) * 3
                for c in (0, 1):
                    fs.append((b0 + c, b0 + 3 + c, b0 + 4 + c, b0 + 1 + c))
        stain = mesh_object(f"rust_run{i}", vs, fs)
        smooth(stain, 80)
        stains.append(stain)

    # ------------------------------------------------------------------ materials
    for o in body:
        assign(o, body_m)
    for o in caps:
        assign(o, cap_m)
    for o in brass:
        assign(o, brass_m)
    for o in bare:
        assign(o, bare_m)
    for o in rusty:
        assign(o, rust_m)
    for o in stones:
        assign(o, stone_m)
    for o in letters:
        assign(o, letter_m)
    for o in plates:
        assign(o, enamel_m)
    for o in numbers:
        assign(o, body_m)
    for o in stains:
        assign(o, stain_m)

    # ------------------------------------------------------------------ weathering story
    nz_blotch = noise_mask(0.07, seed=p.seed + 11, lo=0.3, hi=0.7)
    nz_fine = noise_mask(0.022, seed=p.seed + 12, lo=0.5, hi=0.72)
    nz_clu = noise_mask(0.11, seed=p.seed + 14, lo=0.42, hi=0.62)

    def rust(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        ang = np.arctan2(y, x)
        rad = np.hypot(x, y)
        m = np.clip(1.0 - (z - zb) / 0.2, 0, 1) ** 1.4 * nz_blotch(co, n)             # creeping up from the foot
        m = np.maximum(m, (rad > Rj + 0.004) * (z > zj) * np.clip(-n[:, 2], 0, 1) * 0.9)  # undersides of the nozzles
        for (a, zt, ln, s_) in runs:
            da = np.abs((ang - a + pi) % (2 * pi) - pi)
            w = np.maximum(np.clip(1.0 - da / (colw * 0.95), 0, 1), 0.35 * np.clip(1.0 - da / (colw * 2.4), 0, 1))
            t = (zt - z) / ln
            m = np.maximum(m, w * s_ * (t > -0.04) * np.clip(1.0 - t, 0, 1) ** 0.7)
        return np.clip(m * (0.4 + 0.8 * W), 0, 1)

    def chips(co, n):
        z = co[:, 2]
        m = nz_fine(co, n) * nz_clu(co, n) * (0.3 + 1.0 * W)                            # clustered flaking
        m = np.maximum(m, np.clip(1.0 - (z - zb) / 0.05, 0, 1) * nz_blotch(co, n) * (0.3 + 0.6 * W))   # kicked foot flange
        return np.clip(m, 0, 1)

    def grime(co, n):
        z = co[:, 2]
        g = np.clip(1.0 - (z - zb) / 0.3, 0, 1) * 0.8
        g = np.maximum(g, np.clip(n[:, 2], 0, 1) * 0.4)                                # dust on ledges
        return g * (0.45 + 0.55 * W)

    for o in body + letters + numbers:
        paint(o, grime=grime, wear=chips, special=rust)
    for o in plates:                                   # enamel chipped round the rivets and the rim
        paint(o, grime=lambda co, n: 0.25 + 0.4 * W * nz_blotch(co, n), wear=lambda co, n: np.clip(nz_clu(co, n) * (0.2 + 0.9 * W), 0, 1))
    for o in caps:                                     # caps and bonnet get knocked by spanners and keys
        paint(o, grime=grime, special=rust,
              wear=lambda co, n: np.clip(chips(co, n) + 0.3 * nz_blotch(co, n) * W, 0, 1))
    for o in stains:                                   # blotchy, darker towards the source
        paint(o, grime=lambda co, n: 0.1 + 0.45 * nz_blotch(co, n), special=lambda co, n: 0.2 + 0.5 * nz_clu(co, n))
    for o in brass:
        paint(o, special=0.1 + 0.3 * W, wear=0.6)
    for o in bare:
        paint(o, wear=0.55 if o.name.startswith("op_nut") else 0.15, special=0.15 + 0.6 * W)
    for o in rusty:
        paint(o, grime=0.3)
    for o in stones:
        def wash(co, n):                               # rust wash on the stone around the foot
            rr = np.hypot(co[:, 0], co[:, 1])
            return np.clip(1.0 - (rr - Rf) / 0.07, 0, 1) * 0.8 + 0.15

        paint(o, grime=wash, special=mask_mul(facing((0, 0, 1)), noise_mask(0.12, seed=p.seed + 5, lo=0.4, hi=0.75), 0.3 + 0.6 * W))
