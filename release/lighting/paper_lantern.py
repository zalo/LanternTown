"""Hanging paper lantern (chochin type): oiled-paper gores pasted over split-bamboo hoops (each hoop is a raised ridge and
a darker line when lit, the paper sags between them, the gores overlap in darker vertical paste seams), closed by turned,
lacquered wooden rings top and bottom. The glow falls off from the flame towards the collars. It hangs from a forged
S-hook either by three hemp cords gathered in a knot or by a wire bail pivoting on two studs; a candle on an iron pricket
stands on the bottom board and a silk tassel on a turned bead hangs underneath. Everything below the hook swings.
"""
from math import acos, atan2, copysign, cos, hypot, pi, radians, sin, tan

import numpy as np

import bmesh

from tk import *

META = {"name": "Paper lantern", "class": "prop_small", "mount": "hang", "focus": 0.52, "mount_height": 2.7}
PARAMS = {
    "shape": {"default": "barrel", "choices": ["globe", "barrel", "gourd"], "desc": "shade profile"},
    "diameter": {"default": 0.32, "min": 0.18, "max": 0.6, "desc": "largest shade diameter (m)"},
    "aspect": {"default": 1.0, "min": 0.8, "max": 1.3, "desc": "height stretch of the shade"},
    "colour": {"default": "cream", "choices": ["cream", "red", "amber"], "desc": "paper colour"},
    "motif": {"default": "bands", "choices": ["plain", "bands", "stripe"], "desc": "painted decoration: dark collars at the rings / one belt"},
    "accent": {"default": "#3a1512", "desc": "colour of the painted bands / belt"},
    "label": {"default": "", "desc": "letters painted down the front, one below the other (max 5, empty = none)"},
    "label_font": {"default": "serif", "choices": ["serif", "roman", "sans", "script"], "desc": "lettering style"},
    "ribs": {"default": 16, "min": 8, "max": 22, "desc": "number of bamboo hoop intervals"},
    "hanger": {"default": "cord", "choices": ["cord", "bail"], "desc": "three hemp cords to a knot, or a wire bail"},
    "tassel": {"default": 0.24, "min": 0.0, "max": 0.5, "desc": "tassel drop below the bottom ring (m), 0 = none"},
    "tassel_colour": {"default": "#7a1a13", "desc": "tassel silk colour"},
    "ring_paint": {"default": "#191411", "desc": "lacquer colour of the wooden rings"},
    "age": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "out-of-round, sag, soot, damp staining, pasted repairs"},
    "intensity": {"default": 24.0, "min": 0.0, "max": 60.0, "desc": "light intensity (cd), 0 = unlit"},
}
VARIANTS = {
    "red_globe": {"shape": "globe", "colour": "red", "diameter": 0.4, "motif": "plain", "hanger": "bail", "tassel": 0.3,
                  "tassel_colour": "#c9992e", "ring_paint": "#14100e", "ribs": 15, "intensity": 26.0, "seed": 4},
    "amber_gourd": {"shape": "gourd", "colour": "amber", "diameter": 0.3, "motif": "stripe", "accent": "#7a1c12",
                    "tassel": 0.2, "tassel_colour": "#3a2a1c", "ring_paint": "#4a1410", "ribs": 18, "seed": 9},
    "inn_sign": {"shape": "barrel", "colour": "cream", "diameter": 0.38, "aspect": 1.12, "motif": "bands", "accent": "#1b1512",
                 "label": "INN", "tassel": 0.0, "hanger": "bail", "ribs": 18, "intensity": 30.0, "age": 0.3, "seed": 13},
    "small_globe": {"shape": "globe", "diameter": 0.22, "motif": "stripe", "accent": "#28324a", "tassel": 0.14,
                    "tassel_colour": "#28324a", "ribs": 11, "intensity": 10.0, "age": 0.15, "seed": 17},
    "tall_red_old": {"shape": "barrel", "colour": "red", "diameter": 0.36, "aspect": 1.28, "motif": "bands",
                     "accent": "#161210", "tassel": 0.42, "tassel_colour": "#b5892a", "age": 0.95, "ribs": 20,
                     "intensity": 30.0, "seed": 27},
}

# paper tints: level with the flame / further away / at the collars (doubled paper - paste seams, mends - is one level down)
COLOURS = {
    "cream": {"tints": ("#f4be7c", "#eeb06a", "#e4a15a"), "light": "#ffc27a", "glow": 0.54},
    "red": {"tints": ("#cf3a20", "#c22f1a", "#b02615"), "light": "#ff8650", "glow": 1.15},
    "amber": {"tints": ("#ee9c34", "#e88f2a", "#de8222"), "light": "#ffae52", "glow": 0.8},
}
LEVELS = (1.0, 0.8, 0.64)                              # glow of the three paper levels


def _shade_curve(shape, R, rt, rb, H):
    """Nominal shade outline [(r, z)] from the bottom ring (z = -H) to the top ring (z = 0)."""
    pts = []
    if shape == "globe":
        tb, tt = acos(min(rb / R, 0.95)), acos(min(rt / R, 0.95))
        raw = [(R * cos(-tb + (tb + tt) * i / 48), R * sin(-tb + (tb + tt) * i / 48)) for i in range(49)]
        z0, z1 = raw[0][1], raw[-1][1]
        pts = [(r, (z - z1) / (z1 - z0) * H) for r, z in raw]
    elif shape == "barrel":
        e = 2.0 / 2.6
        for i in range(65):
            a = -pi / 2 + pi * i / 64
            s = copysign(abs(sin(a)) ** e, sin(a))
            ro = lerp(rb, rt, (s + 1) / 2)
            pts.append((ro + (R - ro) * abs(cos(a)) ** e, (s - 1) / 2 * H))
    else:  # gourd: big lower lobe, pinched waist, smaller upper lobe
        ctrl = [(rb / R, 0.0), (0.74, 0.035), (0.96, 0.13), (1.0, 0.27), (0.9, 0.41), (0.63, 0.525), (0.47, 0.6),
                (0.52, 0.665), (0.66, 0.75), (0.7, 0.84), (0.6, 0.93), (rt / R, 1.0)]
        path = resample_path([(c[0] * R, 0.0, (c[1] - 1.0) * H) for c in ctrl], n=90)
        pts = [(q[0], q[2]) for q in path]
    return pts


def _face_out(obj, outward=True):
    """Make the (open) surface of revolution face outwards / inwards."""
    me = obj.data
    s = sum(f.normal.x * f.center.x + f.normal.y * f.center.y for f in me.polygons)
    if (s < 0) == outward:
        me.flip_normals()


def _strap(name, pts, phi, w, t, face, edge, w_fn=None, closed=False, apex=0):
    """Flat bar / tape following pts = [(rho, z)] in the vertical plane through the lantern axis at angle phi: w across in
    that plane, t thick. The light sits on the axis, so it only rakes the broad faces (material `face`), the narrow edges
    get material `edge`. apex = +1 / -1 draws the bar out to a knife edge on the left / right of the path: put that side
    towards the flame and no face stares straight into it."""
    n = len(pts)
    cx, sx = cos(phi), sin(phi)
    ring = ((apex, 0), (-apex, 1), (-apex, -1)) if apex else ((1, 1), (1, -1), (-1, -1), (-1, 1))
    m = len(ring)
    verts, faces = [], []
    for i, (x, z) in enumerate(pts):
        a = pts[i - 1] if (i > 0 or closed) else pts[i]
        b = pts[(i + 1) % n] if (i < n - 1 or closed) else pts[i]
        tx, tz = b[0] - a[0], b[1] - a[1]
        ln = hypot(tx, tz) or 1.0
        hw = w / 2 * (w_fn(i / max(n - 1, 1)) if w_fn else 1.0)
        nx, nz = -tz / ln * hw, tx / ln * hw
        for ns, ts in ring:
            rho = x + ns * nx
            verts.append((rho * cx - ts * sx * t / 2, rho * sx + ts * cx * t / 2, z + ns * nz))
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        for q in range(m):
            faces.append((i * m + q, i * m + (q + 1) % m, j * m + (q + 1) % m, j * m + q))
    if not closed:
        e = (n - 1) * m
        faces += [tuple(range(m))[::-1], tuple(e + q for q in range(m))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    assign(o, face)
    if edge is not face:
        assign_faces(o, edge, lambda c, nrm: abs(nrm.y * cx - nrm.x * sx) < 0.5)
    smooth(o, 40)
    return o


def build(p, rng):
    col = COLOURS[p.colour]
    lit = p.intensity > 0.01
    age = p.age
    d = p.diameter
    R = d / 2
    k = clamp(d / 0.32, 0.8, 1.45)                     # hardware scale (hook, cords, ring sections)
    glow = col["glow"] * clamp(0.75 + p.intensity / 80.0, 0.75, 1.4) if lit else 0.0

    papers = [mat("paper", col["tints"][i], glow=glow * LEVELS[i]) for i in range(3)]
    paper_acc = mat("paper", p.accent, glow=glow * 0.2)            # painted collars / belt / lettering
    lacquer = mat("wood_painted", p.ring_paint, rough=0.6)         # proud mouldings: rubbed through to the wood
    lacquer_rim = mat("wood_painted", p.ring_paint, rough=1.0, wear=0.0)   # faces the flame can see: unworn, or they sparkle
    iron = mat("iron_wrought", wear=0.0)
    # tarred hemp. The cords hang in the unshadowed light that leaves the top opening, where every lit fibre blooms like a
    # fairy light: they are drawn as what they are at night, a dim silhouette (flat unlit colour)
    hemp = mat("emissive", "#2a1e13", glow=1.0)
    silk = mat("canvas", tuple(c * 0.4 for c in hex_rgb(p.tassel_colour)[:3]), scale=2.0)
    flame_m = mat("emissive", "#ffcf8a", glow=10)

    swing = []                                         # everything that hangs from the hook
    yaw = rng.fork("yaw").u(-35, 35)

    # ---- forged S-hook (static), hammered flat: top of the upper loop is the suspension point
    hk = []
    c1, c2, hr = -0.012 * k, -0.040 * k, 0.010 * k
    for a in range(200, -46, -35):
        hk.append((hr * cos(radians(a)), c1 + hr * sin(radians(a))))
    for a in range(135, 386, 36):
        hk.append((hr * cos(radians(a)), c2 + hr * sin(radians(a))))
    phi_h = radians(yaw + 90)
    hook = _strap("s_hook", hk, phi_h, 0.0038 * k, 0.002 * k, iron, hemp, w_fn=lambda t: min(1.0, 0.35 + 7.0 * min(t, 1.0 - t)))
    paint(hook, special=mask_mul(noise_mask(0.02, seed=p.seed), 0.5 + 0.5 * age))
    pivot_z = -0.046 * k

    # ---- shade proportions
    if p.shape == "globe":
        rt = rb = R * 0.5
        H = 2 * R * sin(acos(0.5)) * p.aspect
    elif p.shape == "barrel":
        rt = rb = R * 0.62
        H = 1.32 * d * p.aspect
    else:
        rt, rb = R * 0.34, R * 0.42
        H = 1.72 * d * p.aspect
    rt, rb = max(rt, 0.034), max(rb, 0.034)
    ring_h = 0.03 * k
    cord = p.hanger == "cord"
    zk = -0.072 * k                                    # where the three cords are gathered
    zap = -0.046 * k                                   # apex of the bail, in the hook
    z_rt = (zk - max(0.085, 0.4 * d)) if cord else (zap - max(0.08, 0.4 * d))     # top face of the top ring
    z_top = z_rt - ring_h + 0.006 * k                  # where the paper meets the top ring
    z_bot = z_top - H

    # ---- nominal outline of the shade
    curve = np.array(_shade_curve(p.shape, R, rt, rb, H)) + np.array([0.0, z_top])
    seg = np.hypot(*(curve[1:] - curve[:-1]).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    Ls = float(cum[-1])

    def at(s, off=0.0):
        s = clamp(s, 0.0, Ls)
        r, z = float(np.interp(s, cum, curve[:, 0])), float(np.interp(s, cum, curve[:, 1]))
        s0, s1 = max(s - 0.004, 0.0), min(s + 0.004, Ls)
        tr = float(np.interp(s1, cum, curve[:, 0]) - np.interp(s0, cum, curve[:, 0]))
        tz = float(np.interp(s1, cum, curve[:, 1]) - np.interp(s0, cum, curve[:, 1]))
        ln = hypot(tr, tz) or 1.0
        return (r + off * tz / ln, z - off * tr / ln, tz / ln, -tr / ln)

    # ---- bamboo hoops: hand-wound, so the spacing wanders and every hoop sits a little out of level
    N = int(p.ribs)
    pitch = Ls / N
    rr = rng.fork("ribs")
    wob = 0.6 + 0.9 * age
    s_h = [i * pitch + (rr.u(-1, 1) * min(0.0022, pitch * 0.08) * wob if 0 < i < N else 0.0) for i in range(N + 1)]
    tilt = [(rr.u(0.001, 0.0026) * wob if 0 < i < N else 0.0) for i in range(N + 1)]       # m of tip at the full radius
    tph = [rr.u(0, 2 * pi) for _ in range(N + 1)]
    ridge_h = min(0.0013 * k, pitch * 0.06)             # the hoop stands proud under the paper
    sag0 = min(0.0034, pitch * 0.115) * (0.85 + 0.5 * age)
    sag_i = [sag0 * rr.u(0.7, 1.3) for _ in range(N)]   # the paper pulls in between two hoops, never the same twice
    sph = [(rr.u(0, 2 * pi), rr.u(0, 2 * pi), rr.pick([2, 3, 3, 4])) for _ in range(N)]

    # out of round and a little squashed, more so with age (the ends stay true on the rings)
    wr = rng.fork("warp")
    ph = [wr.u(0, 2 * pi) for _ in range(3)]
    amp = R * (0.008 + 0.03 * age)

    def warp(co):
        t = clamp((co.z - z_bot) / H)
        m = sin(pi * t) ** 0.6
        a = atan2(co.y, co.x)
        dr = amp * m * (0.6 * sin(2 * a + ph[0] + 1.5 * t) + 0.4 * sin(3 * a + ph[1] - 3.0 * t))
        dz = -amp * 0.5 * m * sin(a + ph[2])
        r = hypot(co.x, co.y) or 1.0
        f = (r + dr) / r
        return Vector((co.x * f, co.y * f, co.z + dz * r / R))

    def shade_pt(s, a, lift=0.0, relief=True):
        """Point of the paper surface at arc length s (bottom to top) and angle a."""
        s = clamp(s, 0.0, Ls)
        i = min(max(int(np.searchsorted(s_h, s, side="right")) - 1, 0), N - 1)
        u = clamp((s - s_h[i]) / (s_h[i + 1] - s_h[i]))
        if relief:
            p0, p1, nw = sph[i]
            wrinkle = 0.8 + 0.3 * sin(nw * a + p0) + (0.1 + 0.25 * age) * sin(5 * a + p1)
            o = ridge_h * max(0.0, 1.0 - min(u, 1.0 - u) / 0.12) - sag_i[i] * max(wrinkle, 0.3) * 4.0 * u * (1.0 - u)
        else:
            o = ridge_h
        r, z, _, _ = at(s, o + lift)
        dz = lerp(tilt[i] * cos(a - tph[i]), tilt[i + 1] * cos(a - tph[i + 1]), u) * r / R
        return warp(Vector((r * cos(a), r * sin(a), z + dz)))

    f_light = 0.36 if p.shape == "gourd" else 0.5       # height fraction of the flame in the shade
    f_stripe, w_stripe = (0.6, 0.085) if p.shape == "gourd" else (0.5, 0.12)   # the gourd wears its belt on the waist
    nb = max(1, int(round(N * 0.13)))

    def bright(t):
        """Light reaching the paper: hot level with the flame, about half of that at the far collar."""
        x = abs(t - f_light) / max(f_light, 1.0 - f_light)
        return 1.0 - 0.5 * x ** 1.35

    def zone(i):
        zfr = (at((s_h[i] + s_h[i + 1]) / 2)[1] - z_bot) / H
        if p.motif == "bands" and (i < nb or i >= N - nb):
            return 4
        if p.motif == "stripe" and abs(zfr - f_stripe) < w_stripe:
            return 4
        b = bright(zfr)
        return 0 if b >= 0.84 else (1 if b >= 0.67 else 2)

    zones = [zone(i) for i in range(N)]

    nl, s_lo, z_waist = 6, 0.0, None
    if p.shape == "gourd":                              # only the upper lobe can be seen through the top ring
        sel = (curve[:, 1] > z_bot + 0.5 * H) & (curve[:, 1] < z_bot + 0.72 * H)
        iw = int(np.argmin(np.where(sel, curve[:, 0], 9.0)))
        s_lo, z_waist = float(cum[iw]), float(curve[iw, 1])
    # the gourd is not convex: only from the middle of the waist does the (unshadowed) light see no outside face
    z_light = z_waist if z_waist is not None else z_bot + f_light * H

    # ---- turned, lacquered wooden rings. The lantern's light casts no shadow on the lantern itself, so the mouldings are
    # cut the way the flame rakes them: every outer face leans just away from it, except a narrow chamfer next to the paper
    # that catches the glow (the warm rim of the lacquer). Stepped tiers, the last one a rounded bead.
    def moulding(r0, z_a, z_far):
        sg = 1.0 if z_far > z_a else -1.0
        pts = [(r0, z_a)]

        def run(dz, ddeg):
            r, z = pts[-1]
            th = atan2(r, abs(z + sg * dz / 2 - z_light)) + radians(ddeg)
            pts.append((r + dz * tan(th), z + sg * dz))

        def groove():
            r, z = pts[-1]
            pts.append((r - 0.0036 * k * r / abs(z - z_light), z + sg * 0.0006 * k))

        lux = max(p.intensity, 10.0) / (r0 ** 2 + (z_a - z_light) ** 2)
        soft = clamp(450.0 / lux, 0.5, 1.2)            # the closer the flame, the less the chamfer may open towards it
        run(0.0026 * k, 4.2 * soft)                    # the lit chamfer
        run(0.003 * k, 1.2 * soft)
        run(0.0042 * k, -7.0)                          # from here on the tiers turn their back on the flame
        groove()
        run(0.0042 * k, -7.0)
        groove()
        run(0.0042 * k, -7.0)
        r, z = pts[-1]
        pts += [(r, z_far - sg * 0.0032 * k), (r - 0.0012 * k, z_far - sg * 0.001 * k), (r - 0.0038 * k, z_far)]
        return pts

    SR = 18
    ri_t = rt - 0.012 * k                              # the opening
    out_t = moulding(rt + 0.0008 * k, z_top - 0.001 * k, z_rt)
    top_ring = lathe("ring_top", [(rt - 0.002 * k, z_top - 0.004 * k)] + out_t
                     + [(ri_t + 0.002 * k, z_rt), (ri_t, z_rt - 0.002 * k), (ri_t, z_top - 0.002 * k)], segments=SR, smooth_angle=35)
    zb0 = z_bot + 0.006 * k - ring_h                   # underside of the bottom ring
    zboss = zb0 - 0.013 * k
    out_b = moulding(rb + 0.0008 * k, z_bot + 0.001 * k, zb0)
    bot_ring = lathe("ring_bottom", [(0.0, zb0)] + out_b[::-1]
                     + [(rb - 0.002 * k, z_bot + 0.004 * k), (rb - 0.009 * k, z_bot + 0.004 * k), (rb - 0.011 * k, zb0 + 0.011 * k),
                        (0.0, zb0 + 0.011 * k)], segments=SR, smooth_angle=35)
    boss = lathe("ring_boss", [(0.0, zboss), (0.0075 * k, zboss + 0.004 * k), (0.0075 * k, zboss + 0.009 * k), (0.013 * k, zb0 + 0.001 * k)],
                 segments=8)
    for o in (top_ring, bot_ring, boss):
        assign(o, lacquer)
    # only the bead, which stands well clear of the flame's rake, is rubbed through to the wood; the chamfers and tiers, the
    # inside of the opening and the board the candle stands on stay unworn (bare chips there would sparkle)
    zbt, zbb = out_t[-4][1] - 0.0003 * k, out_b[-4][1] + 0.0003 * k
    assign_faces(top_ring, lacquer_rim, lambda c, n: hypot(c.x, c.y) < ri_t + 0.0015 * k or c.z < zbt)
    assign_faces(bot_ring, lacquer_rim, lambda c, n: c.z > zbb or (c.z > zb0 + 0.005 * k and hypot(c.x, c.y) < rb - 0.004 * k))
    chips = noise_mask(0.012, seed=p.seed + 3, lo=0.35, hi=0.75)
    for o, prof in ((top_ring, out_t), (bot_ring, out_b)):
        r_out = max(q[0] for q in prof) - 0.0018 * k
        # lacquer rubbed through to the wood on the proud tiers only; dust and soot on the upward faces
        paint(o, wear=lambda co, n, r_out=r_out: (np.hypot(co[:, 0], co[:, 1]) > r_out) * chips(co, n) * (0.3 + 0.4 * age),
              grime=mask_mul(facing((0, 0, 1)), 0.3 + 0.5 * age))
    swing += [top_ring, bot_ring, boss]

    # ---- hanger. Everything is flat stock set in planes through the axis: seen from the flame it is edge-on, so it stays
    # a dark silhouette with a thin warm rim instead of blooming in the light that leaves the top opening
    if cord:
        ra = rt - 0.004 * k
        loop = _strap("cord_loop", [(0.0105 * k * cos(radians(a)), -0.0566 * k + 0.0105 * k * sin(radians(a))) for a in range(0, 360, 36)],
                      phi_h + pi / 2, 0.003 * k, 0.0015 * k, hemp, hemp, closed=True)
        # whipping that gathers the three tapes: a bell, wider below, so that it turns its back on the flame
        whip = lathe("cord_whip", [(0.0054 * k, zk - 0.008 * k), (0.0042 * k, zk + 0.004 * k), (0.0028 * k, zk + 0.0085 * k),
                                   (0.0, zk + 0.0095 * k)], segments=7)
        assign(whip, hemp)
        hang, eyes = [loop, whip], []
        for i in range(3):
            a = radians(90 + i * 120)
            tape = _strap(f"cord{i}", [(0.0025 * k, zk - 0.004 * k), (ra, z_rt + 0.0035 * k)], a, 0.0042 * k, 0.0007 * k, hemp, hemp, apex=-1)
            hang.append(tape)
            eyes.append(_strap(f"cord_eye{i}", [(ra - 0.003 * k, z_rt - 0.002 * k), (ra - 0.003 * k, z_rt + 0.0032 * k), (ra, z_rt + 0.0056 * k),
                                                (ra + 0.003 * k, z_rt + 0.0032 * k), (ra + 0.003 * k, z_rt - 0.002 * k)], a, 0.0015 * k, 0.0015 * k, iron, hemp))
        for o in eyes:
            paint(o, grime=0.6)
        swing += hang + eyes
    else:
        zs = z_rt - 0.0105 * k                          # stud height: in a groove of the ring
        r_stud = float(np.interp(zs, [q[1] for q in out_t], [q[0] for q in out_t])) - 0.0005 * k
        x0 = r_stud + 0.0045 * k
        half = [(x0, zs - 0.007 * k), (x0, zs + 0.004 * k)]
        for i in range(1, 8):
            a = radians(90 * i / 8)
            half.append((x0 * cos(a) ** 0.85 + 0.0005, zs + 0.01 * k + (zap - 0.007 * k - zs - 0.01 * k) * sin(a) ** 1.15))
        half += [(0.011 * k, zap - 0.0075 * k), (0.005 * k, zap - 0.002 * k)]
        bail = _strap("bail", half + [(0.0, zap)] + [(-q[0], q[1]) for q in reversed(half)], 0.0, 0.0036 * k, 0.001 * k, iron, hemp, apex=1)
        studs = [bolt(f"bail_stud{sd}", 0.0055 * k, 0.005 * k, loc=(sd * r_stud, 0, zs), rot=(0, sd * 90, 0), kind="dome") for sd in (-1, 1)]
        for o in studs:
            assign(o, iron)
        for o in [bail] + studs:
            paint(o, special=mask_mul(noise_mask(0.03, seed=p.seed + 5), 0.3 + 0.6 * age))
        for o in studs:
            smooth(o)
        swing += [bail] + studs

    # ---- gores: the paper is pasted on in vertical panels; where two overlap the paper is double (darker when lit)
    n_patch = int(round(max(0.0, age - 0.45) * 5.5))
    label = str(p.label or "").strip()[:5]
    gr = rng.fork("gores")
    G = int(clamp(round(2 * pi * R / 0.13) - (1 if gr.chance(0.4) else 0), 6, 8))
    fixed = 1650 + (480 if p.tassel >= 0.06 else 0) + (230 if p.hanger == "cord" else 150) + 250 * len(label) + 60 * n_patch
    U_ROWS, wide = (0.0, 0.14, 0.86), 2
    for rows_try, wide_try in (((0.0, 0.12, 0.5, 0.88), 3), ((0.0, 0.12, 0.5, 0.88), 2), ((0.0, 0.14, 0.86), 3)):
        if fixed + 2 * len(rows_try) * N * G * (1 + wide_try) <= 5980:
            U_ROWS, wide = rows_try, wide_try
            break
    seam_w = 0.0085 * k / R                             # angular width of a paste seam
    a0 = -pi / 2 + pi / G + gr.u(-0.1, 0.1)             # no seam straight behind the lettering
    edges = [a0 + 2 * pi * (g + (gr.u(-0.07, 0.07) if g else 0.0)) / G for g in range(G)]
    wander = [(gr.u(0.006, 0.016), gr.u(1.0, 2.2), gr.u(0, 2 * pi)) for _ in range(G)]     # hand-pasted: not plumb
    cols = []                                           # (angle, gore index if it bounds a seam else -1, face is seam)
    for g in range(G):
        e0, e1 = edges[g], (edges[(g + 1) % G] + (2 * pi if g == G - 1 else 0.0))
        cols.append((e0 - seam_w / 2, g, True))
        for j in range(wide):
            cols.append((lerp(e0 + seam_w / 2, e1 - seam_w / 2, j / wide), g if j == 0 else -1, False))
    C = len(cols)

    def col_angle(c, s):
        a, g, _ = cols[c]
        if g >= 0:
            w, fq, ph0 = wander[g]
            a += w * sin(pi * s / Ls) * sin(2 * pi * fq * s / Ls + ph0)
        return a

    rows = [(lerp(s_h[i], s_h[i + 1], u), i, u) for i in range(N) for u in U_ROWS] + [(Ls, N - 1, 1.0)]
    nr = len(rows)
    Ru = 0.85 * R
    verts, v_t, v_ang, v_ridge, v_zone, v_seam = [], [], [], [], [], []
    ang = [[col_angle(c, s) for c in range(C)] for (s, _, _) in rows]
    for j, (s, i, u) in enumerate(rows):
        tnom = (at(s)[1] - z_bot) / H
        for c in range(C):
            verts.append(shade_pt(s, ang[j][c]))
            v_t.append(tnom)
            v_ang.append(ang[j][c])
            v_ridge.append(u == 0.0 or u == 1.0)
            v_zone.append(zones[i])
            v_seam.append(cols[c][2] or cols[c - 1][2])
    faces, f_mat, f_uv = [], [], []
    for j in range(nr - 1):
        i = rows[j][1]
        for c in range(C):
            c1 = (c + 1) % C
            wrap = 2 * pi if c1 == 0 else 0.0
            faces.append((j * C + c, j * C + c1, (j + 1) * C + c1, (j + 1) * C + c))
            f_mat.append(3 if zones[i] == 4 else (min(zones[i] + 1, 2) if cols[c][2] else zones[i]))   # doubled paper: one level down
            f_uv += [(ang[j][c] * Ru, rows[j][0]), ((ang[j][c1] + wrap) * Ru, rows[j][0]),
                     ((ang[j + 1][c1] + wrap) * Ru, rows[j + 1][0]), (ang[j + 1][c] * Ru, rows[j + 1][0])]
    shade = mesh_object("shade", verts, faces)
    me = shade.data
    for m in papers + [paper_acc]:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", f_mat)
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", np.array(f_uv, dtype=np.float32).ravel())
    me.update()
    uv_keep(shade)
    smooth(shade, 60)

    # glow falloff + bamboo showing through + age, all through the grime mask (it dims the lit paper as well)
    v_t, v_ang, v_ridge, v_zone, v_seam = np.array(v_t), np.array(v_ang), np.array(v_ridge), np.array(v_zone), np.array(v_seam)
    lv = np.array(LEVELS + (1.0, 1.0))[v_zone]
    cr = rng.fork("candle")
    c_dir, c_amt = cr.u(0, 2 * pi), cr.u(0.04, 0.09)   # the candle never stands dead centre
    age_f = clamp((age - 0.3) / 0.6)
    n_soot = noise_mask(0.05, seed=p.seed + 1, lo=0.2, hi=0.8)
    n_dust = noise_mask(0.07, seed=p.seed + 4, lo=0.2, hi=0.8)
    n_mott = noise_mask(0.035, seed=p.seed + 6, lo=0.1, hi=0.9)
    n_damp = noise_mask(0.06, seed=p.seed + 2, lo=0.3, hi=0.7)
    n_tide = noise_mask(0.14, seed=p.seed + 8, lo=0.1, hi=0.9)
    xb = np.abs(v_t - f_light) / max(f_light, 1.0 - f_light)
    v_b = (1.0 - 0.5 * xb ** 1.35) * (1.0 + c_amt * np.cos(v_ang - c_dir)) / (1.0 + c_amt)
    t_base = (nb / N + 0.03) if p.motif == "bands" else 0.06     # rain collects in the bottom of the shade and wicks up
    G_IN, G_OUT = [0.5, 0.58, 0.75, 0.88, 0.95, 1.0], [0.71, 0.57, 0.43, 0.29, 0.14, 0.0]     # lit paper left -> grime needed

    def grime_fn(co, n):
        if len(co) != len(v_b):
            return np.zeros(len(co))
        soot = (0.12 + 0.6 * age) * np.clip((v_t - 0.7) / 0.3, 0, 1) ** 1.4 * (0.5 + 0.5 * n_soot(co, n))
        dust = (0.1 + 0.4 * age) * np.clip(n[:, 2], 0, 1) ** 1.5 * n_dust(co, n)
        mott = (0.04 + 0.1 * age) * n_mott(co, n)
        tide = t_base + 0.11 * n_tide(co, n)
        mark = 0.34 * age_f * np.exp(-((v_t - tide) / 0.02) ** 2)       # the dried edge of a damp patch is its darkest line
        seam = np.where(v_seam & (v_zone == 2), 0.88, 1.0)
        rho = np.clip(v_b * (1.0 - soot) * (1.0 - dust) * (1.0 - mott) * (1.0 - mark) * seam / lv, 0.45, 1.0)
        g = np.where(v_zone == 4, 0.25 + 0.5 * soot + mark, np.interp(rho, G_IN, G_OUT))
        return np.where(v_ridge, np.maximum(g, 0.66), g)                # the bamboo shows through as a darker line

    def damp_fn(co, n):
        if len(co) != len(v_b) or age_f <= 0.0:
            return np.zeros(len(co))
        tide = t_base + 0.11 * n_tide(co, n)
        low = np.clip((tide - v_t) / 0.04, 0, 1) * (v_zone != 4)
        run = np.clip((v_t - 0.62) / 0.38, 0, 1) * np.clip(np.sin(3.0 * v_ang + ph[0]) * np.sin(7.0 * v_ang + ph[1]) * 2.2 - 0.8, 0, 1)
        return np.clip(np.maximum(low, 0.8 * run) * (0.35 + 0.65 * n_damp(co, n)) * age_f * 0.55, 0, 1)

    paint(shade, grime=grime_fn, special=damp_fn)

    # inner face of the paper (seen through the open top ring), well clear of the sagging outer skin
    lin = [at(Ls - (Ls - s_lo) * j / nl, -0.011)[:2] for j in range(nl, -1, -1)] + [(ri_t + 0.0006 * k, z_top + 0.001 * k)]
    liner = lathe("shade_inside", lin, segments=16)
    _face_out(liner, outward=False)
    assign(liner, papers[1])
    transform_verts(liner, warp)

    # ---- painted lettering down the front, wrapped onto the paper (over the hoops)
    def s_of_z(z):
        return float(np.interp(z, curve[:, 1], cum))

    if label:
        zf_, zc_ = {"globe": (0.56, 0.5), "barrel": (0.66, 0.5), "gourd": (0.33, 0.27)}[p.shape]
        cell = min(H * zf_ / len(label), 0.5 * d)
        size = cell * 0.8

        def wrap(co):
            r0 = float(np.interp(co.z, curve[:, 1], curve[:, 0]))
            return shade_pt(s_of_z(co.z), -pi / 2 + co.x / r0, lift=0.0007 + max(0.0, -co.y), relief=False)

        for i, ch in enumerate(label):
            if ch == " ":
                continue
            zl = z_bot + H * zc_ + ((len(label) - 1) / 2 - i) * cell
            t = text(f"label{i}", ch, size=size, depth=0.0006, loc=(0, 0, zl), font=p.label_font)
            apply_transform(t)
            vs = np.array([v.co[:] for v in t.data.vertices])
            lo, hi = vs.min(0), vs.max(0)
            sc = size / max(hi[2] - lo[2], 1e-4)        # glyph to the exact cap height, centred on its cell
            sc = min(sc, 0.42 * d / max(hi[0] - lo[0], 1e-4))
            xc, zc0 = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
            transform_verts(t, lambda co: Vector(((co.x - xc) * sc, co.y, (co.z - zc0) * sc + zl)))
            bm = bmesh.new()                            # painted on: only the face of the letter is needed
            bm.from_mesh(t.data)
            bmesh.ops.delete(bm, geom=[fc for fc in bm.faces if fc.normal.y > 0.5], context="FACES")
            bm.to_mesh(t.data)
            bm.free()
            densify(t, max_edge=0.03)
            transform_verts(t, wrap)
            assign(t, paper_acc)
            swing.append(t)

    # ---- old lanterns get mended: torn pieces of paper pasted over holes. They lie on the skin itself (over the hoops, into
    # the sag) and, being a second layer of paper, let a good third less light through
    def facet_pt(j, a, lift):
        row = ang[j]
        a = row[0] + (a - row[0]) % (2 * pi)
        c = max(0, int(np.searchsorted(row, a, side="right")) - 1)
        c1 = (c + 1) % C
        f = clamp((a - row[c]) / max(row[c1] + (2 * pi if c1 == 0 else 0.0) - row[c], 1e-6))
        q = verts[j * C + c].lerp(verts[j * C + c1], f)
        rad = hypot(q.x, q.y) or 1.0
        return Vector((q.x * (1.0 + lift / rad), q.y * (1.0 + lift / rad), q.z))

    pr = rng.fork("patch")
    pa0 = pr.u(0, 2 * pi)
    for i in range(n_patch):
        pa = pa0 + 2 * pi * (i + pr.u(-0.18, 0.18)) / n_patch
        if label and abs((pa + pi / 2 + pi) % (2 * pi) - pi) < 0.9:     # not over the lettering
            pa += 1.5
        s_c = Ls * pr.u(0.3, 0.7)
        pw, phh, slant = pr.u(0.075, 0.11) * k, pr.u(0.06, 0.09) * k, pr.u(-0.35, 0.35)
        js = [j for j, rw in enumerate(rows) if abs(rw[0] - s_c) <= phh / 2 and zones[rw[1]] != 4]
        if len(js) < 3 or js[-1] - js[0] != len(js) - 1:
            continue
        jm = js[len(js) // 2]
        r_m = max(at(rows[jm][0])[0], 0.02)
        base = ang[jm][0]
        inner = [c for c in range(C) if 0.012 < (ang[jm][c] - pa + pw / 2 / r_m) % (2 * pi) < pw / r_m - 0.012]
        inner.sort(key=lambda c: (ang[jm][c] - pa + pi) % (2 * pi))
        pv, pf, p_rib = [], [], []
        nq = len(inner) + 2
        for jr, j in enumerate(js):
            sj = rows[j][0]
            y = (sj - s_c) / (phh / 2)
            r0 = max(at(sj)[0], 0.02)
            taper = 1.0 - 0.5 * abs(y) ** 2.5                           # torn, not cut: ragged sides, rounded corners
            mid = pa + slant * (sj - s_c) / r0
            lo, hi = mid - pw / 2 * taper * pr.u(0.72, 1.15) / r0, mid + pw / 2 * taper * pr.u(0.72, 1.15) / r0
            for a in [lo] + [clamp(pa + (ang[j][c] - pa + pi) % (2 * pi) - pi, lo, hi) for c in inner] + [hi]:
                pv.append(facet_pt(j, a, 0.0008))
                p_rib.append(rows[j][2] == 0.0)
            if jr:
                for q in range(nq - 1):
                    b0, b1 = (jr - 1) * nq + q, jr * nq + q
                    pf.append((b0, b0 + 1, b1 + 1, b1))
        pch = mesh_object(f"patch{i}", pv, pf)
        lvl = min(zones[rows[jm][1]] + 1, 2)
        assign(pch, papers[lvl])
        smooth(pch, 60)
        p_rib = np.array(p_rib)
        n_p = noise_mask(0.03, seed=p.seed + 20 + i, lo=0.2, hi=0.9)

        def patch_grime(co, n, p_rib=p_rib, lvl=lvl, n_p=n_p):
            xp = np.abs((co[:, 2] - z_bot) / H - f_light) / max(f_light, 1.0 - f_light)
            g = np.interp(np.clip((1.0 - 0.5 * xp ** 1.35) * (0.62 + 0.08 * n_p(co, n)) / LEVELS[lvl], 0.45, 1.0), G_IN, G_OUT)
            return np.where(p_rib, np.maximum(g, 0.7), g) if len(co) == len(p_rib) else g

        paint(pch, grime=patch_grime)
        swing.append(pch)
    swing += [shade, liner]

    # ---- candle on an iron pricket stand (visible through the top ring) + the light
    zc = zb0 + 0.011 * k
    hc = min(0.07, H * 0.2)
    z_fl = z_bot + f_light * H - 0.012                  # base of the flame: level with the hot belt of the shade
    z_cb = max(zc + 0.012 * k, z_fl - hc)               # underside of the candle
    stand = lathe("pricket", [(0.0, zc), (0.02 * k, zc), (0.02 * k, zc + 0.003 * k), (0.005 * k, zc + 0.008 * k), (0.003 * k, z_cb - 0.008 * k),
                              (0.017 * k, z_cb - 0.004 * k), (0.02 * k, z_cb + 0.003 * k), (0.012 * k, z_cb), (0.0, z_cb)], segments=8)
    assign(stand, iron)
    paint(stand, grime=0.6)
    candle = cylinder("candle", 0.0105, hc, loc=(0, 0, z_cb + hc / 2), verts=8, bevel=0.0015, segments=1)
    assign(candle, papers[2])                           # tallow candle dyed like the paper, glows a little near the flame
    swing += [stand, candle]
    zf = z_cb + hc
    if lit:
        fl = lathe("flame", [(0.0, zf + 0.002), (0.0045, zf + 0.006), (0.006, zf + 0.012), (0.0035, zf + 0.022), (0.0, zf + 0.032)], segments=8)
        assign(fl, flame_m)
        swing.append(fl)
        light("lantern", (0, 0, z_light), color=col["light"], intensity=p.intensity, range=clamp(4.0 + p.intensity * 0.12, 4.0, 9.0),
              flicker=0.3, shadow_near=round(max(H * 0.75 + p.tassel, R * 1.6) + 0.05, 2))

    # ---- tassel: cord, turned bead, wrapped head and a skirt of silk threads
    if p.tassel >= 0.06:
        tr_ = rng.fork("tassel")
        z_end = zb0 - p.tassel
        zh = min(zboss - 0.03 * k, z_end + 0.1 * k)     # top of the tassel head
        zh = max(zh, z_end + 0.05 * k)
        tc = sweep("tassel_cord", [(0, 0, zboss + 0.003 * k), (0, 0, zh - 0.002 * k)], 0.002 * k, sides=5, smooth_path=False)
        assign(tc, hemp)
        zbd = (zboss + zh) / 2
        bead = lathe("tassel_bead", [(0.0025 * k, zbd - 0.013 * k), (0.0062 * k, zbd - 0.009 * k), (0.0078 * k, zbd - 0.002 * k), (0.0062 * k, zbd + 0.006 * k),
                                     (0.0025 * k, zbd + 0.014 * k)], segments=8)
        assign(bead, lacquer_rim)
        assign_faces(bead, hemp, lambda c, n: n.z > 0.1)     # the cord is whipped over the top of the bead
        zs0 = zh - 0.046 * k                            # skirt starts under the wrapped neck
        # steep-shouldered head (bound over a wooden core) with a thread-wrapped neck
        head = lathe("tassel_head", [(0.0105 * k, zs0 - 0.004 * k), (0.012 * k, zs0), (0.012 * k, zs0 + 0.008 * k), (0.0102 * k, zs0 + 0.0095 * k),
                                     (0.0132 * k, zs0 + 0.013 * k), (0.0132 * k, zs0 + 0.02 * k), (0.0085 * k, zs0 + 0.033 * k),
                                     (0.0045 * k, zs0 + 0.043 * k), (0.0022 * k, zs0 + 0.046 * k)], segments=10)
        assign(head, silk)
        assign_faces(head, hemp, lambda c, n: n.z > 0.15)    # the shoulders are whipped with the same tarred cord
        ns = 12
        jit = [tr_.u(-0.07, 0.05) for _ in range(2 * ns)]
        Lk = zs0 - z_end
        r_top, r_bot = 0.0108 * k, 0.0102 * k              # hangs plumb

        def skirt_fn(u, v):
            j = int(round(v * 2 * ns)) % (2 * ns)
            uu = min(u / 0.85, 1.0)
            fold = smoothstep(0.85, 1.0, u)
            r = lerp(r_top, r_bot, uu ** 0.7) * (1.0 + (0.15 if j % 2 else -0.15) * min(1.0, uu * 5.0)) * (1.0 - fold)
            a = 2 * pi * v
            return (r * cos(a), r * sin(a), zs0 - Lk * (1.0 + jit[j]) * uu + 0.004 * fold)

        skirt = surface("tassel_skirt", skirt_fn, nu=5, nv=2 * ns, closed_v=True)
        _face_out(skirt, outward=True)
        assign(skirt, silk)
        smooth(skirt, 70)
        swing += [tc, bead, head, skirt]

    part(swing, "lantern", pivot=(0, 0, pivot_z), motion="swing", amp=3.5, speed=0.3)
