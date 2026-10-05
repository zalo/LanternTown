"""Traditional timber wheelbarrow: two straight ash shafts converge on an iron-tyred, spoked wooden wheel
(turned nave with iron bands, felloes in segments, axle pin through strapped shaft ends) and end in turned
hand grips. A tapering, flared plank tray (separate bottom, side, head and tail boards, outside cleats, nails
and forged corner irons) sits on the shafts; two raked legs with iron shoes, coach bolts, flat-bar stays and
a stretcher carry it at rest. Loads: soil with flowering plants, stacked logs, or empty. An optional forged
crook bolted to a cleat carries a small candle lantern over the load.
"""
from tk import *
from math import sin, cos, tan, pi, radians, degrees, atan, atan2, hypot
import numpy as np
from mathutils import Matrix, Vector

META = {"name": "Wooden wheelbarrow", "class": "prop", "mount": "ground", "focus": [0.12, -0.2, 0.55]}
PARAMS = {
    "length": {"default": 1.55, "min": 1.3, "max": 1.85, "desc": "overall length, wheel front to handle tips (m)"},
    "width": {"default": 0.6, "min": 0.5, "max": 0.75, "desc": "width across the handle grips (m)"},
    "wheel_dia": {"default": 0.46, "min": 0.36, "max": 0.58, "desc": "wheel diameter over the iron tyre (m)"},
    "spokes": {"default": 8, "min": 6, "max": 12, "desc": "number of spokes (rounded to even; two per felloe)"},
    "handle_height": {"default": 0.5, "min": 0.42, "max": 0.62, "desc": "height of the grips when resting (m)"},
    "tray_depth": {"default": 0.22, "min": 0.14, "max": 0.32, "desc": "height of the side boards at the rear (m)"},
    "flare": {"default": 14.0, "min": 4.0, "max": 26.0, "desc": "outward flare of the side boards (deg)"},
    "load": {"default": "plants", "choices": ["plants", "logs", "empty"], "desc": "what is in the tray"},
    "load_amount": {"default": 0.7, "min": 0.2, "max": 1.0, "desc": "how full the tray is"},
    "flower": {"default": "#c8322a", "desc": "flower colour of the plants"},
    "wood_tone": {"default": "#a8825c", "desc": "stain / tone of the bare timber"},
    "scheme": {"default": "tray", "choices": ["tray", "full", "bare"], "desc": "painted tray only, tray + undercarriage, or all bare wood"},
    "paint": {"default": "#4a8c74", "desc": "paint colour of the tray boards"},
    "trim": {"default": "#9c3a28", "desc": "paint colour of wheel, shafts and legs (scheme = full)"},
    "wear": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "age: chipped paint, rust, grime, open board joints"},
    "tailboard": {"default": True, "desc": "low tail board closing the rear of the tray"},
    "tool": {"default": True, "desc": "a tool left in the barrow (trowel in the soil, spade in an empty tray)"},
    "lantern": {"default": "front", "choices": ["front", "side", "none"], "desc": "forged crook with a candle lantern: on the head board lighting the way ahead, on a side board, or none"},
    "lantern_side": {"default": "right", "choices": ["right", "left"], "desc": "which side of the barrow the lantern crook is fixed to"},
    "crook": {"default": 0.36, "min": 0.26, "max": 0.7, "desc": "height of the lantern crook above the tray rim (m)"},
}
VARIANTS = {
    "logs_bare": {"load": "logs", "scheme": "bare", "wood_tone": "#a08060", "wear": 0.7, "length": 1.72,
                  "wheel_dia": 0.54, "spokes": 10, "tray_depth": 0.17, "flare": 8.0, "load_amount": 1.0, "tailboard": False, "lantern": "side", "lantern_side": "left", "crook": 0.3, "seed": 7},
    "empty_red": {"load": "empty", "scheme": "full", "paint": "#a43a2a", "trim": "#c9b684", "wear": 0.2, "tray_depth": 0.29,
                  "flare": 21.0, "crook": 0.3, "handle_height": 0.56, "lantern_side": "left", "seed": 11},
    "garden_blue": {"load": "plants", "paint": "#5a86b8", "flower": "#e6b23a", "length": 1.4, "wheel_dia": 0.4, "spokes": 6,
                    "load_amount": 1.0, "lantern": "none", "width": 0.54, "seed": 23},
    "old_weathered": {"load": "plants", "scheme": "bare", "wood_tone": "#86705a", "flower": "#d8cfe6", "wear": 1.0,
                      "load_amount": 0.4, "tray_depth": 0.19, "wheel_dia": 0.5, "crook": 0.52, "tool": False, "seed": 41},
}

X = Vector((1, 0, 0))
Z = Vector((0, 0, 1))


def slab(name, org, U, V, poly, thick, bev=0.003):
    """Board cut to a 2D outline `poly` [(a, b)] in the plane org + a*U + b*V, `thick` along U x V."""
    org, U, V = Vector(org), Vector(U).normalized(), Vector(V).normalized()
    N = U.cross(V).normalized()
    n = len(poly)
    front = [org + U * a + V * b + N * thick for a, b in poly]
    back = [org + U * a + V * b for a, b in poly]
    faces = [tuple(range(n)), tuple(reversed(range(n, 2 * n)))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, n + i, n + j, j))
    o = mesh_object(name, [tuple(v) for v in front + back], faces)
    recalc_normals(o)
    if bev:
        bevel(o, bev, 1)
    smooth(o, 35)
    uv_grain(o, axis=tuple(U))
    return o


def aim(pt, n, s=1.0):
    """Matrix placing a +Z-pointing stamp at pt with +Z turned to n."""
    q = Vector(n).normalized().to_track_quat('Z', 'Y')
    return Matrix.Translation(Vector(pt)) @ q.to_matrix().to_4x4() @ Matrix.Scale(s, 4)


def leaf_mesh(name, length, wid, droop, cup, nv=3, tip=0.8):
    """Double-sided leaf blade along +Y (base at the origin), UVs filling one 0.25 m tile."""
    verts, uvs = [], []
    for j in range(nv + 1):
        t = j / nv
        wp = sin(pi * min(1.0, 0.06 + 0.9 * t ** tip))
        for i in (-1, 0, 1):
            x = i * wid * 0.5 * wp
            verts.append((x, t * length, -droop * length * t * t + cup * abs(x)))
            uvs.append(((0.5 + 0.46 * i * wp) * 0.25, (0.03 + 0.94 * t) * 0.25))
    nvt = len(verts)
    faces = []
    for j in range(nv):
        for i in range(2):
            a = j * 3 + i
            faces.append((a, a + 1, a + 4, a + 3))
    verts2 = verts + [(x, y, z - 0.0004) for x, y, z in verts]
    faces2 = faces + [tuple(reversed([k + nvt for k in f])) for f in faces]
    o = mesh_object(name, verts2, faces2)
    uvl = o.data.uv_layers.new(name="UVMap")
    for li, lp in enumerate(o.data.loops):
        uvl.data[li].uv = uvs[lp.vertex_index % nvt]
    uv_keep(o)
    smooth(o, 80)
    return o


def build(p, rng):
    R = p.wheel_dia / 2
    L = p.length
    hh = max(p.handle_height, R + 0.14)
    th = atan2(hh - R, L - R)                 # slope of the shafts at rest
    Ls = hypot(hh - R, L - R)                 # axle -> grip tip along the shaft
    ya = -L / 2 + R
    Mt = Matrix.Translation((0, ya, R)) @ Matrix.Rotation(th, 4, 'X')   # "bed" frame (origin = axle) -> world

    def W(pt):
        return Mt @ Vector(pt)

    wear = p.wear
    seed = int(p.seed)
    bed = []                                  # objects modelled in the bed frame (tilted at the end)
    frame_wood, tray_wood, floor_wood, wheel_wood, grips, irons = [], [], [], [], [], []

    # ------------------------------------------------------------------ shafts with turned grips
    sw, sd, hl = 0.045, 0.06, 0.07
    xa = hl + 0.014 + sw / 2
    xh = p.width / 2 - 0.022
    kx = (xh - xa) / Ls

    def xs(v):
        return xa + kx * v

    vg = Ls - 0.2
    t_tap = (vg - 0.17 + 0.075) / (vg + 0.075)
    grip_tab = [(0.0, 1.22), (0.05, 1.25), (0.1, 1.0), (0.18, 0.84), (0.3, 0.9), (0.52, 1.0), (0.74, 0.9), (0.84, 0.84),
                (0.9, 0.95), (0.95, 1.14), (0.985, 1.05), (1.0, 0.7)]

    def grip_r(t):
        for (t0, s0), (t1, s1) in zip(grip_tab, grip_tab[1:]):
            if t <= t1:
                return lerp(s0, s1, (t - t0) / (t1 - t0))
        return grip_tab[-1][1]

    for sx in (-1, 1):
        pts = [(sx * xs(v), v, 0.0) for v in (-0.075, -0.06, vg - 0.17, vg - 0.06, vg)]
        sh = sweep(f"shaft{sx}", pts, rounded_rect(sw, sd, 0.008, 2), smooth_path=False,
                   scale_fn=lambda t: (0.8 + 0.2 * smoothstep(0.0, 0.012, t)) * (1.0 - 0.28 * smoothstep(t_tap, 1.0, t)))
        frame_wood.append(sh)
        bed.append(sh)
        gp = [(sx * xs(v), v, 0.0) for v in [lerp(vg - 0.012, Ls, t) for t, _ in grip_tab]]
        g = sweep(f"grip{sx}", gp, 0.0195, sides=12, smooth_path=False, scale_fn=grip_r)
        grips.append(g)
        bed.append(g)
        # iron strap sleeve where the axle pin passes through the shaft end
        sl = box(f"axle_strap{sx}", (sw + 0.009, 0.06, sd + 0.009), loc=(sx * xs(0.0), 0.0, 0.0),
                 rot=(0, 0, degrees(-sx * atan(kx))), bevel=0.002)
        irons.append(sl)
        bed.append(sl)

    # ------------------------------------------------------------------ tray geometry (bed frame)
    wb = sd / 2                                # top of the shafts
    tb = 0.022                                 # board thickness
    v0 = R + 0.05
    v1 = Ls - 0.45
    D = p.tray_depth
    phi = radians(p.flare)
    gap = 0.0025 + 0.004 * wear
    jit = 0.0006 + 0.0028 * wear
    rb = rng.fork("boards")

    def xb(v):
        return xs(v) + sw / 2 + 0.018

    for i, v in enumerate((v0 + 0.03, v1 - 0.07)):
        cb = box(f"crossbar{i}", (2 * xs(v) - sw + 0.012, 0.042, 0.046), loc=(0, v, -0.004), bevel=0.004)
        uv_grain(cb, axis='X')
        frame_wood.append(cb)
        bed.append(cb)

    nb = 4
    for k in range(nb):
        f0, f1 = k / nb, (k + 1) / nb
        va, vb = v0 - 0.006 + rb.u(-0.004, 0.0), v1 + 0.006 + rb.u(0.0, 0.004)

        def xe(v, f):
            return -xb(v) + 2 * xb(v) * f

        poly = [(va, -(xe(va, f0) + gap / 2)), (vb, -(xe(vb, f0) + gap / 2)), (vb, -(xe(vb, f1) - gap / 2)), (va, -(xe(va, f1) - gap / 2))]
        o = slab(f"bottom{k}", (0, 0, wb + rb.u(0, jit)), (0, 1, 0), (-1, 0, 0), poly, tb)
        floor_wood.append(o)
        bed.append(o)

    hr = D / cos(phi)                          # slant height of the sides at the rear ...
    hf = hr * 1.32                             # ... and at the front
    af_t, ar_t = tan(radians(26)), tan(radians(12))
    nail_m = []                                # nail heads (bed frame)
    stud_m = []                                # bigger forged studs on the irons (bed frame)
    sides = {}
    for sx in (-1, 1):
        B0 = Vector((sx * xb(v0), v0, wb))
        B1 = Vector((sx * xb(v1), v1, wb))
        U = (B1 - B0).normalized()
        Lsd = (B1 - B0).length
        nh = Vector((U.y, -U.x, 0)) * sx
        Vv = (Z * cos(phi) + nh * sin(phi)).normalized()
        No = U.cross(Vv) * sx                  # outward normal of the side
        bs = hr * 0.5

        def af(b):
            return -b * af_t

        def ar(b, Lsd=Lsd):
            return Lsd + b * ar_t

        def top(t, ar=ar):
            return (lerp(af(hf), ar(hr), t), hr + (hf - hr) * (1 - smoothstep(0.05, 0.7, t)))

        b2 = bs + gap
        lower = [(af(0), 0), (ar(0), 0), (ar(bs), bs), (af(bs), bs)]
        upper = [(af(b2), b2), (ar(b2), b2)] + [top(1 - i / 10) for i in range(11)]
        lo = slab(f"side_lo{sx}", B0 + No * rb.u(0, jit), U, Vv, lower, sx * tb)
        up = slab(f"side_up{sx}", B0 + No * rb.u(0, jit), U, Vv, upper, sx * tb)
        tray_wood += [lo, up]
        bed += [lo, up]
        # outside cleats tying the two boards together
        cl_info = []
        for ci, fa in enumerate((0.17, 0.83)):
            a_c = fa * Lsd
            bt = top(fa)[1] - 0.012
            cw = 0.021
            cpoly = [(a_c - cw, 0.004), (a_c + cw, 0.004), (a_c + cw, bt - 0.006), (a_c + cw - 0.008, bt), (a_c - cw + 0.008, bt), (a_c - cw, bt - 0.006)]
            c = slab(f"cleat{sx}_{ci}", B0 + No * tb, U, Vv, cpoly, sx * 0.017)
            tray_wood.append(c)
            bed.append(c)
            for j, bn in enumerate((bs * 0.28, bs * 0.78, b2 + (bt - b2) * 0.25, b2 + (bt - b2) * 0.8)):
                an = a_c + (0.007 if j % 2 else -0.007) + rb.u(-0.002, 0.002)
                nail_m.append(aim(B0 + U * an + Vv * bn + No * (tb + 0.017), No, rb.u(0.85, 1.15)))
            cl_info.append((a_c, bt))
        sides[sx] = dict(B0=B0, U=U, Vv=Vv, No=No, Lsd=Lsd, af=af, ar=ar, top=top, cleats=cl_info)

    S = sides[1]
    # head board: raked forward over the wheel, capping the side-board ends, arched top
    Bm = Vector((0, v0, wb))
    T0 = S["B0"] + S["U"] * S["af"](hf) + S["Vv"] * hf
    Vh = Vector((0, T0.y - v0, T0.z - wb)).normalized()
    Hh = hypot(T0.y - v0, T0.z - wb)
    Nh = X.cross(Vh)                            # forward (and a little down)
    h_org = Bm - Nh * 0.004

    def xw(b):
        b_s = b * Vh.z / S["Vv"].z
        return (S["B0"] + S["U"] * S["af"](b_s) + S["Vv"] * b_s + S["No"] * tb).x + 0.005

    bh = Hh * 0.48
    crown = 0.03 + 0.02 * (D / 0.22)
    lo = slab("head_lo", h_org + Nh * rb.u(0, jit), X, Vh, [(-xw(0), 0), (xw(0), 0), (xw(bh), bh), (-xw(bh), bh)], tb)
    b2 = bh + gap
    arch = [((1 - 2 * i / 12) * xw(Hh), Hh + crown * (1 - (1 - 2 * i / 12) ** 2)) for i in range(13)]
    up = slab("head_up", h_org + Nh * rb.u(0, jit), X, Vh, [(-xw(b2), b2), (xw(b2), b2)] + arch, tb)
    tray_wood += [lo, up]
    bed += [lo, up]
    for bn in (bh * 0.25, bh * 0.75, b2 + (Hh - b2) * 0.3, b2 + (Hh - b2) * 0.82):
        for sx in (-1, 1):
            nail_m.append(aim(h_org + X * sx * (xw(bn) - 0.016) + Vh * bn + Nh * tb, Nh, rb.u(0.85, 1.15)))

    # tail board: low, raked back
    Bm1 = Vector((0, v1, wb))
    T1 = S["B0"] + S["U"] * S["ar"](hr) + S["Vv"] * hr
    Vt = Vector((0, T1.y - v1, T1.z - wb)).normalized()
    Ht = 0.56 * hypot(T1.y - v1, T1.z - wb)
    Nt = (-X).cross(Vt)                         # rearward
    t_org = Bm1 - Nt * 0.004

    def xwt(b):
        b_s = b * Vt.z / S["Vv"].z
        return (S["B0"] + S["U"] * S["ar"](b_s) + S["Vv"] * b_s + S["No"] * tb).x + 0.005

    if p.tailboard:
        tl = slab("tail", t_org + Nt * rb.u(0, jit), -X, Vt,
                  [(-xwt(0), 0), (xwt(0), 0), (xwt(Ht - 0.012), Ht - 0.012), (xwt(Ht) - 0.03, Ht), (-xwt(Ht) + 0.03, Ht), (-xwt(Ht - 0.012), Ht - 0.012)], tb)
        tray_wood.append(tl)
        bed.append(tl)
        for bn in (Ht * 0.25, Ht * 0.72):
            for sx in (-1, 1):
                nail_m.append(aim(t_org + X * sx * (xwt(bn) - 0.016) + Vt * bn + Nt * tb, Nt, rb.u(0.85, 1.15)))
    else:
        Ht = 0.0

    # forged corner irons: flat bar wrapping head board -> side, two heights, studded
    bar = rounded_rect(0.0036, 0.024, 0.001, 1)
    straps = []
    for sx in (-1, 1):
        Sd = sides[sx]
        for bn in (Hh * 0.2, Hh * 0.8):
            pa = h_org + X * sx * (xw(bn) - 0.085) + Vh * bn + Nh * (tb + 0.0012)
            pb = h_org + X * sx * (xw(bn) + 0.0035) + Vh * bn + Nh * (tb + 0.0012)
            s1 = sweep(f"corner_h{sx}", [pa, pb], bar, smooth_path=False, up=tuple(Vh))
            b_s = bn * Vh.z / Sd["Vv"].z
            a0 = Sd["af"](b_s)
            qa = Sd["B0"] + Sd["U"] * (a0 - 0.026) + Sd["Vv"] * b_s + Sd["No"] * (tb + 0.0012)
            qb = Sd["B0"] + Sd["U"] * (a0 + 0.09) + Sd["Vv"] * b_s + Sd["No"] * (tb + 0.0012)
            s2 = sweep(f"corner_s{sx}", [qa, qb], bar, smooth_path=False, up=tuple(Sd["Vv"]))
            straps += [s1, s2]
            bed += [s1, s2]
            for f in (0.25, 0.78):
                stud_m.append(aim(pa.lerp(pb, f * 0.85) + Nh * 0.0018, Nh))
                stud_m.append(aim(qa.lerp(qb, 0.3 + f * 0.65) + Sd["No"] * 0.0018, Sd["No"]))

    # ------------------------------------------------------------------ legs, shoes, stays, stretcher (world frame)
    v_leg = v1 - 0.13
    bolt_w = []                                 # square coach-bolt heads / nuts (world frame)
    leg_c = {}
    for sx in (-1, 1):
        xt = xs(v_leg) + sw / 2 + 0.02
        topp = W((sx * xt, v_leg, wb - 0.002))
        foot = Vector((topp.x + sx * 0.03, topp.y + 0.05, 0.0))
        d = (topp - foot).normalized()

        def cut(co, d=d):
            return co - d * (co.z / d.z) if co.z < 0 else co

        leg = sweep(f"leg{sx}", [foot - d * 0.05, foot + d * 0.2, topp], rounded_rect(0.04, 0.056, 0.006, 2), smooth_path=False, up=(0, 1, 0))
        transform_verts(leg, cut)
        frame_wood.append(leg)
        shoe = sweep(f"shoe{sx}", [foot - d * 0.05, foot + d * 0.055], rounded_rect(0.048, 0.064, 0.006, 2), smooth_path=False, up=(0, 1, 0))
        transform_verts(shoe, cut)
        irons.append(shoe)
        for dv, dw in ((-0.013, 0.012), (0.013, -0.014)):
            bolt_w.append(aim(W((sx * (xt + 0.02), v_leg + dv, dw)), X * sx))

        def lc(z, foot=foot, d=d):
            return foot + d * (z / d.z)

        leg_c[sx] = lc
        # flat-bar stay from the leg up to the underside of the shaft
        vs = v_leg - 0.25
        off = Vector((0, -0.0306, 0))
        path = [lc(0.085) + off, lc(0.17) + off, W((sx * (xs(vs) + 0.004), vs, -sd / 2 - 0.003)),
                W((sx * (xs(vs - 0.075) + 0.004), vs - 0.075, -sd / 2 - 0.003))]
        st = sweep(f"stay{sx}", path, rounded_rect(0.005, 0.026, 0.001, 1), smooth_path=False, up=(1, 0, 0))
        irons.append(st)
        bolt_w.append(aim(lc(0.125) + Vector((0, -0.033, 0)), (0, -1, 0), 0.8))
    c = leg_c[1](0.15)
    stz = box("stretcher", (2 * (c.x - 0.02) + 0.014, 0.028, 0.042), loc=(0, c.y, 0.15), bevel=0.004)
    uv_grain(stz, axis='X')
    frame_wood.append(stz)

    # ------------------------------------------------------------------ wheel (world frame, centred on the axle)
    wc = Vector((0, ya, R))
    ns = max(6, 2 * int(round(p.spokes / 2)))
    rw = rng.fork("wheel")
    phase = rw.u(0, 360.0 / ns)
    fd, fw = 0.044, 0.042
    rm = R - 0.007 - fd / 2
    nave = lathe("nave", [(0.0, -hl), (0.027, -hl), (0.031, -hl + 0.004), (0.035, -0.046), (0.046, -0.036), (0.053, -0.02), (0.055, 0.0),
                          (0.053, 0.02), (0.046, 0.036), (0.035, 0.046), (0.031, hl - 0.004), (0.027, hl), (0.0, hl)],
                 segments=20, loc=tuple(wc), rot=(0, 90, 0))
    uv_transform(nave, rot_deg=90)
    wheel_wood.append(nave)
    for sx in (-1, 1):
        irons.append(ring_band(f"nave_band{sx}", radius=0.0335, width=0.017, thick=0.003, loc=(sx * 0.0575, ya, R), rot=(0, 90, 0), segments=20))
    spoke = sweep("spoke", [(0, 0, 0.04), (0, 0, rm - fd / 2 + 0.006)], rounded_rect(0.034, 0.024, 0.009, 2), smooth_path=False,
                  up=(1, 0, 0), scale_fn=lambda t: 1.0 - 0.2 * t)
    sp_m = [Matrix.Translation(wc) @ Matrix.Rotation(radians(phase + i * 360.0 / ns - 90.0), 4, 'X') for i in range(ns)]
    wheel_wood.append(instances_to_mesh("spokes", spoke, sp_m, keep_base=False, seed=seed))
    nf = ns // 2
    for k in range(nf):
        a0 = phase + (2 * k - 0.5) * 360.0 / ns + 0.7
        a1 = phase + (2 * k + 1.5) * 360.0 / ns - 0.7
        npt = max(6, int(40 / nf))
        arc = [(0.0, rm * cos(radians(lerp(a0, a1, i / npt))), rm * sin(radians(lerp(a0, a1, i / npt)))) for i in range(npt + 1)]
        fe = sweep(f"felloe{k}", arc, rounded_rect(fd, fw, 0.004, 1), smooth_path=False, up=(1, 0, 0))
        place(fe, loc=tuple(wc))
        wheel_wood.append(fe)
    tyre = sweep("tyre", [(0.0, (R - 0.004) * cos(2 * pi * i / 48), (R - 0.004) * sin(2 * pi * i / 48)) for i in range(48)],
                 rounded_rect(0.008, 0.046, 0.0015, 1), closed=True, smooth_path=False, up=(1, 0, 0))
    place(tyre, loc=tuple(wc))
    smooth(tyre, 50)
    axle_l = 2 * (xa + sw / 2 + 0.02)
    irons.append(cylinder("axle", 0.0105, axle_l, loc=tuple(wc), rot=(0, 90, 0), verts=10))
    nut = bolt("axle_nut", 0.017, 0.012, kind="hex")
    irons.append(instances_to_mesh("axle_nuts", nut, [aim((sx * (xa + sw / 2 + 0.0045), ya, R), X * sx) for sx in (-1, 1)], keep_base=False))

    # ------------------------------------------------------------------ lantern crook
    lan = []
    glass = flame = None
    if p.lantern != "none":
        rc, rh = 0.125, 0.018
        clips = []
        sxl = -1 if p.lantern_side == "left" else 1
        if p.lantern == "side":
            Sd = sides[sxl]
            fa = 0.34
            a_c, bt = fa * Sd["Lsd"], Sd["top"](fa)[1]
            c_org, c_u, c_v, c_n = Sd["B0"] + Sd["U"] * a_c, Sd["U"], Sd["Vv"], Sd["No"]
            dh = Vector((sxl, 0, 0))
        else:
            xc = sxl * 0.62 * xw(Hh)
            bt = Hh + crown * (1 - (xc / xw(Hh)) ** 2)
            c_org, c_u, c_v, c_n = h_org + X * xc, X, Vh, Nh
            dh = Vector((0.5 * sxl, -0.866, 0))

        def Pc(b):
            return W(c_org + c_v * b + c_n * (tb + 0.0066))

        rim = Pc(bt)
        ztop = rim.z + p.crook
        top_xy = Vector((rim.x, rim.y, 0)) + dh * 0.014

        def at(d, z):
            return top_xy + dh * d + Z * z

        path = [Pc(0.02), Pc(bt * 0.5), Pc(bt - 0.005), at(0, rim.z + 0.08), at(0, ztop - rc - 0.04)]
        path += [at(rc - rc * cos(radians(a)), ztop - rc + rc * sin(radians(a))) for a in range(0, 181, 20)]
        cu = at(2 * rc + rh, ztop - rc - 0.03)
        path += [cu - dh * (rh * cos(radians(a))) + Z * (rh * sin(radians(a))) for a in range(0, -211, -30)]
        rod = sweep("crook", path, 0.0062, sides=6, step=0.022, scale_fn=lambda t: 1.0 - 0.45 * smoothstep(0.93, 1.0, t))
        irons.append(rod)
        for bcl in (0.04, bt - 0.055):
            clip = slab("crook_clip", c_org - c_u * 0.02 + c_v * bcl + c_n * tb, c_u, c_v, [(0, 0), (0.04, 0), (0.04, 0.018), (0, 0.018)], 0.0142, bev=0.0015)
            irons.append(clip)
            bed.append(clip)
            for da in (-0.014, 0.014):
                stud_m.append(aim(c_org + c_u * da + c_v * (bcl + 0.009) + c_n * (tb + 0.0142), c_n, 0.7))
        Hp = cu - Z * rh
        ring = torus("lan_ring", major=0.013, minor=0.0021, major_seg=14, minor_seg=5)
        ring.matrix_basis = aim(Hp - Z * 0.0155, dh)
        lan.append(ring)
        zl = Hp.z - 0.028
        lo3 = (Hp.x, Hp.y, zl)
        lan.append(lathe("lan_cap", [(0.0, -0.07), (0.046, -0.07), (0.052, -0.067), (0.05, -0.062), (0.03, -0.037), (0.017, -0.031),
                                     (0.015, -0.017), (0.006, -0.013), (0.0085, -0.006), (0.006, 0.0), (0.0, 0.0)], segments=14, loc=lo3))
        lan.append(lathe("lan_base", [(0.0, -0.218), (0.049, -0.218), (0.053, -0.21), (0.05, -0.202), (0.041, -0.187), (0.039, -0.168), (0.0, -0.168)],
                         segments=14, loc=lo3))
        glass = lathe("lan_glass", [(0.037, -0.168), (0.044, -0.146), (0.046, -0.118), (0.043, -0.09), (0.036, -0.07)], segments=14, loc=lo3)
        wire = cylinder("lan_wire", 0.0017, 0.126, verts=5, cap=False)
        lan.append(instances_to_mesh("lan_wires", wire, [Matrix.Translation((Hp.x + 0.05 * cos(radians(45 + k * 90)), Hp.y + 0.05 * sin(radians(45 + k * 90)), zl - 0.126))
                                                         for k in range(4)], keep_base=False))
        lan.append(cylinder("lan_burner", 0.011, 0.034, loc=(Hp.x, Hp.y, zl - 0.151), verts=8))
        flame = sphere("lan_flame", 0.0105, loc=(Hp.x, Hp.y, zl - 0.116), segments=8, rings=6, scale=(1, 1, 1.8))
        light("lantern", (Hp.x, Hp.y, zl - 0.116), color="#ffbf6a", intensity=3.4, range=5.0, flicker=0.3, shadow_near=0.15)

    # ------------------------------------------------------------------ loads
    tan_h = -Vh.y / Vh.z
    tan_t = Vt.y / Vt.z
    tph = tan(phi)
    rl = rng.fork("load")
    amt = p.load_amount
    soil = None
    leaves = flowers = None
    logs_bark, logs_end = [], []
    h_tail = Ht * Vt.z                          # tail board top above wb
    if p.load == "plants":
        he = max(tb + 0.03, min(0.03 + 0.62 * amt * D, h_tail - 0.018))
        mound = 0.03 + 0.05 * amt
        vf, vr = v0 - he * tan_h - 0.005, v1 + he * tan_t + 0.005

        def soil_fn(u, v):
            vv = lerp(vf, vr, v)
            x = (2 * u - 1) * (xb(vv) + he * tph + 0.005)
            m = max(0.0, sin(pi * u) * sin(pi * v)) ** 0.6
            return (x, vv, wb + he + mound * m)

        soil = surface("soil", soil_fn, 14, 18)
        displace_noise(soil, strength=0.011, scale=0.06, seed=seed, along_normal=False, axis_mask=(0, 0, 1))
        bed.append(soil)
        # plants (world frame so they grow upright)
        leafA = leaf_mesh("leafA", 0.085, 0.074, 0.3, 0.22, nv=3, tip=0.7)
        leafB = leaf_mesh("leafB", 0.3, 0.024, 0.55, 0.3, nv=4, tip=0.5)
        petal = leaf_mesh("petal", 0.034, 0.034, 0.3, 0.2, nv=2, tip=0.6)
        la, lb, pm = [], [], []
        stems = []
        npl = 4 + int(round(3 * amt))
        pts = []
        tries = 0
        while len(pts) < npl and tries < 400:
            tries += 1
            u, v = rl.u(0.2, 0.8), rl.u(0.14, 0.84)
            if all(hypot((u - a) * 0.45, (v - b) * 0.6) > 0.15 for a, b in pts):
                pts.append((u, v))
        ngrass = 0
        for i, (u, v) in enumerate(pts):
            pb = W(soil_fn(u, v)) + Vector((0, 0, 0.004))
            size = rl.u(0.85, 1.2)
            if i % 3 == 2 and ngrass < 2:
                ngrass += 1
                for j in range(16):
                    az, pitch = rl.u(0, 360), rl.u(48, 84)
                    M = Matrix.Translation(pb + Vector((rl.u(-0.012, 0.012), rl.u(-0.012, 0.012), 0))) @ Matrix.Rotation(radians(az), 4, 'Z') \
                        @ Matrix.Rotation(radians(pitch), 4, 'X') @ Matrix.Scale(size * rl.u(0.6, 1.05), 4)
                    lb.append(M)
                continue
            nl = int(15 * size)
            for j in range(nl):
                t = j / (nl - 1)
                az = j * 137.5 + rl.u(-20, 20)
                pitch = lerp(4, 62, t) + rl.u(-9, 9)
                ro = lerp(0.055, 0.008, t) * size
                dirv = Vector((-sin(radians(az)), cos(radians(az)), 0))
                pos = pb + dirv * ro + Vector((0, 0, lerp(0.02, 0.13, t) * size))
                la.append(Matrix.Translation(pos) @ Matrix.Rotation(radians(az), 4, 'Z') @ Matrix.Rotation(radians(pitch), 4, 'X')
                          @ Matrix.Rotation(radians(rl.u(-25, 25)), 4, 'Y') @ Matrix.Scale(size * rl.u(0.75, 1.15), 4))
            for j in range(rl.pick([2, 3, 3])):
                hc = pb + Vector((rl.u(-0.06, 0.06), rl.u(-0.06, 0.06), rl.u(0.17, 0.25) * size))
                mid = pb.lerp(hc, 0.5) + Vector((rl.u(-0.015, 0.015), rl.u(-0.015, 0.015), 0.01))
                stems.append(sweep("stem", [pb + Vector((0, 0, 0.01)), mid, hc], 0.0024, sides=4, step=0.05, cap=False))
                for k in range(12):
                    az, el = k * 137.5 + rl.u(-15, 15), lerp(2, 78, k / 11.0) + rl.u(-8, 8)
                    pm.append(Matrix.Translation(hc) @ Matrix.Rotation(radians(az), 4, 'Z') @ Matrix.Rotation(radians(el), 4, 'X')
                              @ Matrix.Translation((0, 0.003, 0)) @ Matrix.Scale(rl.u(0.8, 1.15) * (1.0 - 0.3 * k / 11.0), 4))
        if p.tool:
            tbase = W(soil_fn(0.2, 0.87))
            ax = Vector((-0.28, 0.34, 1.0)).normalized()
            sd_ = ax.cross(Vector((0.3, 1, 0))).normalized()
            irons.append(slab("trowel_blade", tbase - ax * 0.07 - sd_.cross(ax) * 0.001, ax, sd_,
                              [(0, -0.012), (0.05, -0.034), (0.135, -0.037), (0.15, -0.02), (0.15, 0.02), (0.135, 0.037), (0.05, 0.034), (0, 0.012)], 0.0025, bev=0.0008))
            irons.append(sweep("trowel_neck", [tbase + ax * 0.07, tbase + ax * 0.1 + sd_.cross(ax) * 0.012, tbase + ax * 0.135 + sd_.cross(ax) * 0.014],
                               0.0048, sides=6, smooth_path=False))
            th_ = lathe("trowel_handle", [(0.0, 0.0), (0.0125, 0.0), (0.014, 0.006), (0.0125, 0.012), (0.0135, 0.04), (0.0165, 0.085), (0.0155, 0.11), (0.009, 0.122), (0.0, 0.124)], segments=10)
            th_.matrix_basis = aim(tbase + ax * 0.13 + sd_.cross(ax) * 0.014, ax)
            uv_transform(th_, rot_deg=90)
            grips.append(th_)
        lv = [instances_to_mesh("leavesA", leafA, la, uv_jitter=False, keep_base=False)]
        if lb:
            lv.append(instances_to_mesh("leavesB", leafB, lb, uv_jitter=False, keep_base=False))
        else:
            delete([leafB])
        leaves = lv + stems
        flowers = instances_to_mesh("petals", petal, pm, uv_jitter=False, keep_base=False)
    elif p.load == "logs":
        r0 = 0.044
        v_s = v0 + 0.035
        layer, placed = 0, 0
        n0 = max(2, int((2 * (xb(v_s) + r0 * tph)) / (2 * r0 * 1.03)))
        total = max(2, int(round(amt * (n0 + max(n0 - 1, 0) + max(n0 - 2, 0)))))
        while placed < total and layer < 3:
            nrow = n0 - layer
            if nrow <= 0:
                break
            wc_l = wb + tb + r0 + layer * r0 * 1.74
            for i in range(nrow):
                if placed >= total:
                    break
                r = r0 * rl.u(0.84, 1.0)
                dz = wc_l - r0 - wb               # underside height above wb
                room = (v1 + dz * tan_t) - v_s - 0.02
                over = dz > h_tail + 0.004
                ln = min(room, rl.u(0.42, 0.56)) if not over else rl.u(0.5, 0.76)
                x = (i - (nrow - 1) / 2) * 2 * r0 * 1.03 + rl.u(-0.004, 0.004)
                vst = v_s + rl.u(0.0, 0.035) - (dz * tan_h * 0.6 if layer else 0.0)
                yaw = rl.u(-2.5, 2.5) if not over else rl.u(-6, 6)
                prof = [(r * 0.985, 0.0), (r, 0.008)] + [(r * (1 + 0.05 * sin(6.0 * j / 5 + rl.u(0, 6))), ln * j / 5) for j in range(1, 5)] + [(r, ln - 0.008), (r * 0.985, ln)]
                lg = lathe(f"log{placed}", prof, segments=10, loc=(x, vst, wc_l - (r0 - r)), rot=(-90 + rl.u(-1.5, 1.5), 0, yaw))
                logs_bark.append(lg)
                for e in (0.0015, ln - 0.0015):
                    cp = cylinder(f"logend{placed}", r * 0.99, 0.004, verts=10)
                    cp.matrix_basis = lg.matrix_basis @ Matrix.Translation((0, 0, e))
                    logs_end.append(cp)
                placed += 1
            layer += 1
        bed += logs_bark + logs_end

    elif p.tool:
        tip = Vector((-0.06, v0 + 0.06, wb + tb + 0.012))
        over_t = Vector((0.07, v1 + h_tail * tan_t + 0.012, wb + max(h_tail, tb) + 0.02))
        dr = (over_t - tip).normalized()
        neck = tip + dr * 0.36
        end = neck + dr * 0.7
        sp_sh = sweep("spade_shaft", [neck - dr * 0.03, end], 0.0165, sides=8, smooth_path=False)
        sp_t = sweep("spade_grip", [end - X * 0.07, end + X * 0.07], 0.0165, sides=8, smooth_path=False,
                     scale_fn=lambda t: 0.85 + 0.25 * sin(pi * t))
        grips += [sp_sh, sp_t]
        sock = sweep("spade_socket", [neck - dr * 0.11, neck - dr * 0.02, neck + dr * 0.06], 0.021, sides=8, smooth_path=False,
                     scale_fn=lambda t: 1.12 - 0.22 * t)
        sdv = dr.cross(Z).normalized()
        bn_ = (-dr).cross(sdv)
        blade = slab("spade_blade", neck - dr * 0.09 - bn_ * 0.016, -dr, sdv,
                     [(0, -0.092), (0.012, -0.095), (0.2, -0.086), (0.245, -0.062), (0.27, -0.026), (0.27, 0.026), (0.245, 0.062), (0.2, 0.086), (0.012, 0.095), (0, 0.092)],
                     0.0032, bev=0.001)
        irons += [sock, blade]
        bed += [sp_sh, sp_t, sock, blade]

    # ------------------------------------------------------------------ fixings
    nail = lathe("nail", [(0.0046, 0.0), (0.0036, 0.0016), (0.0, 0.0024)], segments=6)
    nails = instances_to_mesh("nails", nail, nail_m, keep_base=False, seed=seed)
    stud = lathe("stud", [(0.0066, 0.0), (0.0058, 0.0022), (0.003, 0.0038), (0.0, 0.0042)], segments=7)
    studs = instances_to_mesh("studs", stud, stud_m, keep_base=False, seed=seed)
    cb = bolt("coach", 0.0095, 0.006, kind="square")
    irons.append(instances_to_mesh("coach_bolts", cb, bolt_w, keep_base=False, seed=seed))
    irons.append(nails)
    straps.append(studs)
    bed += [nails, studs]

    # ------------------------------------------------------------------ tilt the bed-frame objects into place
    for o in bed:
        o.matrix_basis = Mt @ o.matrix_basis

    # ------------------------------------------------------------------ materials and weathering
    oak = mat("wood_oak", p.wood_tone)
    iron = mat("iron_wrought")
    if p.scheme == "bare":
        m_tray = mat("wood_weathered", p.wood_tone)
        m_frame = m_wheel = oak
    elif p.scheme == "full":
        m_tray = mat("wood_painted", p.paint)
        m_frame = m_wheel = mat("wood_painted", p.trim)
    else:
        m_tray = mat("wood_painted", p.paint)
        m_frame = m_wheel = oak

    chips = noise_mask(0.045, seed=seed + 3, lo=0.7 - 0.45 * wear, hi=1.0 - 0.3 * wear)
    dirt = noise_mask(0.12, seed=seed + 5, lo=0.3, hi=0.8)
    rustn = noise_mask(0.04, seed=seed + 7, lo=0.62 - 0.5 * wear, hi=1.0 - 0.35 * wear)
    low = height_ramp(0.0, 0.32)

    def grime_fn(co, n):
        return np.clip(0.55 * low(co, n) + (0.14 + 0.36 * wear) * dirt(co, n), 0, 1)

    def fade_fn(co, n):
        return np.clip((0.2 + 0.5 * wear) * np.clip(n[:, 2], 0, 1) + 0.35 * wear * dirt(co, n), 0, 1)

    for o in tray_wood:
        assign(o, m_tray)
        paint(o, wear=mask_mul(chips, 0.55 + 0.45 * wear), grime=grime_fn, special=fade_fn)
    for o in floor_wood:
        assign(o, oak if p.scheme != "bare" else m_tray)
        paint(o, grime=0.45 + 0.4 * wear, wear=mask_mul(dirt, 0.4))
    for o in frame_wood:
        assign(o, m_frame)
        paint(o, wear=mask_mul(chips, 0.3 + 0.6 * wear), grime=grime_fn)
    for o in wheel_wood:
        assign(o, m_wheel)
        paint(o, wear=mask_mul(chips, 0.4 + 0.6 * wear), grime=lambda co, n: np.clip(0.3 + 0.5 * wear * dirt(co, n) + 0.4 * low(co, n), 0, 1))
    for o in grips:
        assign(o, oak)
        paint(o, wear=0.3, grime=mask_mul(dirt, 0.25))
    for o in irons:
        assign(o, iron)
        paint(o, special=mask_mul(rustn, 0.5 + 0.5 * wear), grime=mask_mul(low, 0.5))
    assign(tyre, iron)
    for o in straps:
        assign(o, iron)
        paint(o, special=0.55 + 0.4 * wear, wear=0.25)

    def tread(co, n):
        r = np.hypot(co[:, 1] - ya, co[:, 2] - R)
        return np.clip((r - (R - 0.003)) / 0.002, 0, 1) * 0.8

    paint(tyre, wear=tread, special=mask_mul(rustn, 0.6 + 0.4 * wear), grime=mask_mul(dirt, 0.6))

    if lan:
        brass = mat("brass", rough=0.9)
        for o in lan:
            assign(o, brass)
            paint(o, grime=mask_mul(facing((0, 0, 1)), 0.5), special=mask_mul(rustn, 0.3))
        assign(glass, mat("glass_lantern", glow=1.6))
        assign(flame, mat("emissive", "#ffb45e", glow=9))
        part(lan + [glass, flame], "lantern", pivot=tuple(Hp), motion="swing", amp=2.0, speed=0.35)
    if soil is not None:
        assign(soil, mat("soil"))
        m_leaf = mat("leaf", "#6f9a4a")
        for o in leaves:
            assign(o, m_leaf)
        assign(flowers, mat("leaf", p.flower))
    if logs_bark:
        m_bark, m_end = mat("bark"), mat("wood_endgrain", "#d9b98a")
        for o in logs_bark:
            assign(o, m_bark)
            paint(o, special=mask_mul(facing((0, 0, 1)), dirt, 0.4))
        for o in logs_end:
            assign(o, m_end)
            paint(o, grime=mask_mul(dirt, 0.5))

    socket("load", "attach", tuple(W((0, (v0 + v1) / 2, wb + tb + D))), role="prop_top")
