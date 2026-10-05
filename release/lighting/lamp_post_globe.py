"""Parisian electric candelabra lamp post. A kerb of individually dressed stones carries a stepped octagonal
cast-iron plinth (panelled die, rosettes, wiring hatch on the back), a gadrooned vase and a tapering fluted column
with a cast acanthus collar and beaded bands. A leafy capital carries an urn hub from which tapering swan-neck
arms spring out of leaf husks, braced by riveted flat-bar scroll stays; every arm and the centre stem end in a
tulip fitter with thumb screws and a petal gallery clamping an opal glass globe, crowned by a vent cap and finial.
1, 3 or 5 globes. Light: the opal globes are self-luminous; one small rim light in the head keeps the paint colour
of the ironwork readable and the soft pool lights inside the beaded bands carry the output down to the ground.
"""
from tk import *
from math import asin, acos, sqrt, sin, cos, pi, radians, degrees, atan2, hypot, exp
from mathutils import Matrix, Vector
import numpy as np
import bmesh

META = {"name": "Electric globe lamp post", "class": "street_furniture", "mount": "ground", "focus": 0.86, "lod": "far"}
PARAMS = {
    "height": {"default": 4.2, "min": 3.2, "max": 5.6, "desc": "overall height to the tip of the top finial (m)"},
    "globes": {"default": 3, "choices": [1, 3, 5], "desc": "number of globes: 1 = on the post top, 3 / 5 = centre globe + 2 / 4 swan-neck arms"},
    "globe_diameter": {"default": 0.34, "min": 0.24, "max": 0.5, "desc": "glass globe diameter (m)"},
    "spread": {"default": 0.5, "min": 0.36, "max": 0.8, "desc": "arm reach from the column axis (m)"},
    "flutes": {"default": 16, "min": 0, "max": 24, "desc": "number of flutes on the column (0 = plain)"},
    "stone_step": {"default": 0.14, "min": 0.0, "max": 0.4, "desc": "height of the dressed stone kerb under the iron base (0 = none, > 0.2 = two stepped courses)"},
    "paint": {"default": "#33503f", "desc": "paint colour of the cast iron"},
    "gilt": {"default": True, "desc": "gilded accents (rosettes, beading, galleries, rivets); False = painted over"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "condition: 0 = freshly painted, 1 = chipped, scuffed and rust-streaked"},
    "intensity": {"default": 34.0, "min": 8, "max": 150, "desc": "nominal light output per globe (cd)"},
    "light_color": {"default": "#ffcf94", "desc": "colour of the light (warm tungsten)"},
    "number": {"default": "27", "desc": "post number on the enamel plate on the front of the plinth (max 3 characters, empty = no plate)"},
}
VARIANTS = {
    "single_tall": {"globes": 1, "height": 4.6, "globe_diameter": 0.46, "paint": "#2c3a42", "flutes": 20, "intensity": 55, "wear": 0.7,
                    "number": "8", "seed": 4},
    "five_grand": {"globes": 5, "height": 5.0, "globe_diameter": 0.32, "spread": 0.62, "paint": "#1e3324", "stone_step": 0.28,
                   "intensity": 18, "wear": 0.35, "number": "112", "seed": 8},
    "boulevard_plain": {"globes": 3, "height": 3.7, "globe_diameter": 0.28, "spread": 0.42, "flutes": 0, "gilt": False,
                        "paint": "#4a2a1c", "stone_step": 0.0, "wear": 1.0, "number": "", "seed": 15},
    "wide_blue": {"globes": 3, "height": 4.4, "globe_diameter": 0.4, "spread": 0.74, "paint": "#1b2c44", "flutes": 12,
                  "wear": 0.08, "intensity": 36, "seed": 23},
}


# ------------------------------------------------------------------------------------------------ helpers
def np_clip(a):
    return np.clip(a, 0.0, 1.0)


def _streaks(seed, width=0.03, length=0.5, lo=0.45, hi=0.75):
    """Mask: noise stretched along Z -> vertical runs (rain streaks, rust bleeding)."""
    f = noise_mask(width, seed=seed, lo=lo, hi=hi)
    k = width / length

    def m(co, n):
        c2 = np.array(co, dtype=float, copy=True)
        c2[:, 2] *= k
        return f(c2, n)
    return m


def _near(pts, rad):
    """Mask: 1 at any of the points, fading to 0 at distance rad."""
    P = np.array(pts, dtype=float)

    def m(co, n):
        d = np.sqrt(((np.asarray(co)[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
        return np_clip(1.0 - d / rad)
    return m


def _below(z):
    return lambda co, n: (np.asarray(co)[:, 2] < z) * 1.0


def _above(z):
    return lambda co, n: (np.asarray(co)[:, 2] > z) * 1.0


def _msum(*fns):
    def m(co, n):
        t = 0.0
        for f in fns:
            t = t + (f(co, n) if callable(f) else f)
        return np_clip(t)
    return m


def _prof_r(prof, z):
    """Radius of a lathe profile [(r, z), ...] (z ascending) at height z."""
    if z <= prof[0][1]:
        return prof[0][0]
    for (ra, za), (rb, zb) in zip(prof, prof[1:]):
        if zb >= z and zb > za:
            return ra + (rb - ra) * (z - za) / (zb - za)
    return prof[-1][0]


def _drop_faces(obj, pred):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    dead = [f for f in bm.faces if pred(f.calc_center_median(), f.normal)]
    if dead:
        bmesh.ops.delete(bm, geom=dead, context='FACES')
    bm.to_mesh(obj.data)
    bm.free()


def _dome(name, r, h, seg=6):
    """Cheap rivet / screw head sitting on z=0 pointing +Z."""
    return lathe(name, [(r, -0.0015), (r, h * 0.35), (r * 0.62, h * 0.85), (0.0, h)], segments=seg, smooth_angle=70)


def _orbit(o, a):
    """Bake the object's transform, then rotate it about the column axis."""
    apply_transform(o)
    place(o, rot=(0, 0, a))
    return o


def _scroll(A, C, ro, ri, turns, n=22, cw=True):
    """XZ-plane S-stay: leaves A horizontally, lands on top of a circle (centre C, radius ro), then spirals inwards."""
    top = (C[0], C[1] + ro)
    dx = top[0] - A[0]
    lead = bezier_path((A[0], 0, A[1]), (A[0] + dx * 0.45, 0, A[1]), (top[0] - dx * 0.45, 0, top[1]), (top[0], 0, top[1]), n=7)
    pts = list(lead)
    for i in range(1, n + 1):
        t = i / n
        a = radians(90.0) + (-1 if cw else 1) * 2 * pi * turns * t
        r = ro + (ri - ro) * t
        pts.append((C[0] + r * cos(a), 0, C[1] + r * sin(a)))
    return pts


_LEAF_C = (-1.0, -0.8, -0.45, -0.15, 0.0, 0.15, 0.45, 0.8, 1.0)


def _leaf_ring(name, n, zb, hl, rfun, arc, t0, curl, sink, phase=0.0, alt=1.0, nv=10, cols=_LEAF_C):
    """A ring of cast acanthus leaves in relief around a turned body. Every leaf is a closed solid swelling that
    grows out of the body (its rim is sunk `sink` below the surface, so there is no gap behind it), with a raised
    midrib, lobed sides and a tip that turns `curl` outwards."""
    verts, faces = [], []
    nc = len(cols)
    rows = [(0.0, -0.006, True)] + [(j / nv, 0.0, False) for j in range(nv + 1)] + [(1.0, 0.006, True)]
    for k in range(n):
        phi0 = radians(phase + k * 360.0 / n)
        hk = hl * (alt if k % 2 else 1.0)
        base = len(verts)
        for s, dz, closing in rows:
            z = zb + hk * s + dz
            rs = rfun(z)
            lobe = 0.68 + 0.32 * abs(sin(2.5 * pi * s)) ** 0.7
            w = 0.5 * arc * (1.0 - 0.7 * s ** 2.2) * lobe * (0.8 + 0.2 * smoothstep(0.0, 0.2, s))
            t = t0 * (1.0 - 0.25 * s)
            out = curl * smoothstep(0.55, 1.0, s) ** 1.5
            for c in cols:
                g = 0.74 * (1.0 - abs(c) ** 3) + 0.26 * exp(-(c / 0.2) ** 2)
                r = rs - sink if closing else rs - sink * abs(c) ** 4 + (t + out) * g
                ph = phi0 + c * w / max(rs, 1e-4)
                verts.append((r * cos(ph), r * sin(ph), z - (0.3 * out * g if not closing else 0.0)))
        for j in range(len(rows) - 1):
            for i in range(nc - 1):
                a = base + j * nc + i
                faces.append((a, a + 1, a + 1 + nc, a + nc))
    o = mesh_object(name, verts, faces)
    smooth(o, 50)
    return o


def _stone(name, size, loc, rz, r, detail=0.085):
    """One dressed kerb stone: out-of-true block with knocked corners and chipped arrises, bottom face removed."""
    sb = stone_block(name, size, loc=loc, rot=(0, 0, rz), rng=r, chamfer=r.u(0.008, 0.014), rough=0.006, detail=detail)
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    chips = []
    for _ in range(3):
        if r.chance(0.5):
            pt = (r.pick((-1, 1)) * hx, r.u(-hy, hy), hz)
        else:
            pt = (r.u(-hx, hx), r.pick((-1, 1)) * hy, hz)
        chips.append((Vector(pt), r.u(0.045, 0.08), r.u(0.008, 0.018)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            chips.append((Vector((sx * hx, sy * hy, hz)), r.u(0.04, 0.065), r.u(0.004, 0.014)))
    if r.chance(0.75):                             # one corner knocked right off by a cart wheel
        chips.append((Vector((r.pick((-1, 1)) * hx, r.pick((-1, 1)) * hy, hz)), r.u(0.09, 0.13), r.u(0.022, 0.034)))

    def chip(co):
        for c, rad_, dep in chips:
            d = (co - c).length
            if d < rad_:
                dv = Vector((-c.x, -c.y, -hz * 2.0))
                dv.normalize()
                co = co + dv * ((1.0 - d / rad_) ** 2 * dep)
        return co
    transform_verts(sb, chip)
    _drop_faces(sb, lambda c, nrm: nrm.z < -0.7 and c.z < -hz * 0.8)
    return sb


def _course(tag, A, z_lo, h, r, flip, detail, joint=0.007):
    """A square kerb course of side A: four stones laid as a pinwheel with open joints."""
    a = A * r.u(0.57, 0.62)
    b = A - a - joint
    out = []
    for k in range(4):
        ang = radians(90.0 * k)
        cx, cy = (-A * 0.5 + b * 0.5, -A * 0.5 + a * 0.5) if flip else (-A * 0.5 + a * 0.5, -A * 0.5 + b * 0.5)
        x = cx * cos(ang) - cy * sin(ang) + r.jitter(0, 0.005)
        y = cx * sin(ang) + cy * cos(ang) + r.jitter(0, 0.005)
        hk = h - r.u(0.0, 0.013)
        size = (b, a, hk) if flip else (a, b, hk)
        out.append(_stone(f"{tag}{k}", size, (x, y, z_lo + hk * 0.5), 90.0 * k + r.jitter(0, 1.1), r.fork(f"s{k}"), detail))
    return out


def _globe_far(tag, loc, D, tall, mats, q, rng):
    """lod 2 of _globe_unit (same draws, same return): the tulip fitter as one six-sided cup closed underneath, the
    petal gallery as a crown of flat gilt petals standing clear of the glass, the globe as ONE glowing sphere of
    8 x 4, vent cap and finial as one four-sided turning. Thumb screws, clear skin and smoky caps are left out."""
    iron, gilt, core_m, shell_m, shade_m = mats
    R = D * 0.5
    s = D / 0.34
    ss = sqrt(s)
    c = max(0.075, 0.095 * s)
    rn = 0.02 * ss
    rim = 0.6 * R
    zc = c + 0.8 * R
    cup = lathe(tag + "_cup", lod_exact([(0.0, -0.05 * ss), (rn + 0.014 * ss, -0.05 * ss), (rn + 0.004, 0.2 * c), (rim + 0.006, 0.8 * c),
                                         (rim + 0.006, c + 0.003)]), segments=lod_exact(6), smooth_angle=75)
    assign(cup, iron)
    rng.u(0, 120)                                  # (thumb screws: the draw of the full model)
    th0, dth = asin(0.6) - 0.05, 0.46
    rg = 1.01 * R + 0.008                          # every petal edge stays 5 mm or more outside the glowing sphere

    def pv(az, t):
        return (rg * sin(t) * cos(az), rg * sin(t) * sin(az), zc - rg * cos(t))
    vs, fs = [], []
    if tall:
        # The centre globe houses the head light, and the petals are what shades the arms and the column from it:
        # every petal of the full model stays, in its place, as base, shoulder and tip (five corners).
        npet = 8 if q >= 1.0 else 6
        w = pi / npet * 0.9
        for k in range(npet):
            az = 2 * pi * k / npet - pi / 2
            i = len(vs)
            vs += [pv(az - w, th0), pv(az + w, th0), pv(az + 0.75 * w, th0 + 0.55 * dth), pv(az - 0.75 * w, th0 + 0.55 * dth),
                   pv(az, th0 + 1.03 * dth)]
            fs += [(i, i + 1, i + 2), (i, i + 2, i + 3), (i + 3, i + 2, i + 4)]
    else:                                          # an arm globe: a crown of six (five) flat petals
        npet = 6 if q >= 1.0 else 5
        vs = [pv(pi * k / npet, th0) for k in range(2 * npet)] + [pv(pi * (2 * j + 1) / npet, th0 + 0.85 * dth) for j in range(npet)]
        for j in range(npet):
            a, b, d, t = 2 * j, 2 * j + 1, (2 * j + 2) % (2 * npet), 2 * npet + j
            fs += [(a, b, t), (b, d, t)]
    gal = mesh_object(tag + "_gallery", vs, fs)
    uv_box(gal)
    assign(gal, gilt)
    core = sphere(tag + "_opal", R * 0.98, loc=(0, 0, zc), segments=lod_exact(8), rings=lod_exact(4))
    assign(core, core_m)
    smooth(core, 80)
    ct = zc + R
    fh = (0.2 if tall else 0.13) * ss
    # (the centre globe's vent cap shades as the round one does: six sides, drawn a little outside its rim)
    rc_ = (0.41 if tall else 0.38) * R
    cap = lathe(tag + "_cap", lod_exact([(rc_ + 0.04 * R, zc + 0.72 * R), (rc_, zc + 0.93 * R), (0.2 * R, ct + 0.004), (0.028 * ss, ct + 0.5 * fh),
                                         (0.0, ct + fh)]), segments=lod_exact(6 if tall else 4), smooth_angle=95)
    assign(cap, iron)
    tilt = (rng.jitter(0, 0.7), rng.jitter(0, 0.7), rng.u(0, 360))
    for o in (cup, gal, core, cap):
        apply_transform(o)
        place(o, loc=loc, rot=tilt)
    return zc, fh, [cup, cap]


def _globe_unit(tag, loc, D, tall, mats, q, rng):
    """Moulded boss + tulip fitter + petal gallery + opal globe + vent cap and finial, base at `loc`.
    Returns (globe centre z above the base, finial height, [iron objects])."""
    if lod() == 2:
        return _globe_far(tag, loc, D, tall, mats, q, rng)
    iron, gilt, core_m, shell_m, shade_m = mats
    R = D * 0.5
    s = D / 0.34
    ss = sqrt(s)
    c = max(0.075, 0.095 * s)                      # fitter height
    rn = 0.02 * ss
    rim = 0.6 * R
    zc = c + 0.8 * R                               # globe centre above the fitter base
    seg = 16 if q >= 1.0 else 10
    out, irons = [], []
    cup = lathe(tag + "_cup", [(rn * 0.7, -0.052 * ss), (rn + 0.012 * ss, -0.044 * ss), (rn + 0.017 * ss, -0.03 * ss),
                                (rn + 0.01 * ss, -0.016 * ss), (rn + 0.003, -0.01 * ss),
                                (rn + 0.009 * ss, 0.08 * c), (rn + 0.002, 0.22 * c),
                                (0.3 * R, 0.47 * c), (0.5 * R, 0.63 * c), (rim - 0.002, 0.76 * c), (rim + 0.009, 0.83 * c),
                                (rim + 0.005, 0.88 * c), (rim + 0.005, 0.96 * c),
                                (rim + 0.009, c + 0.004), (rim - 0.004, c + 0.002)], segments=seg)
    assign(cup, iron)
    out.append(cup)
    irons.append(cup)
    nr = seg // 2                                  # fluted husk on the tulip

    def rib(co):
        if co.z < 0.22 * c or co.z > 0.76 * c:
            return co
        e = sin(pi * (co.z - 0.22 * c) / (0.54 * c))
        k = 1.0 + 0.1 * e * (0.5 + 0.5 * cos(atan2(co.y, co.x) * nr))
        return Vector((co.x * k, co.y * k, co.z))
    transform_verts(cup, rib)
    # three thumb screws clamp the globe neck in the fitter
    scr = _dome("ts", 0.0065 * ss, 0.016 * ss)
    a0 = rng.u(0, 120)
    sm = [trs(((rim + 0.004) * cos(radians(a0 + k * 120)), (rim + 0.004) * sin(radians(a0 + k * 120)), 0.92 * c), (0, 90, a0 + k * 120)) for k in range(3)]
    th = instances_to_mesh(tag + "_screws", scr, sm, keep_base=False)
    assign(th, gilt)
    out.append(th)
    # petal gallery hugging the lower bowl of the globe
    npet = 8 if q >= 1.0 else 6
    th0 = asin(0.6) - 0.05
    dth = 0.46
    wmax = pi / npet * 0.9

    def petal(u, v):
        t = th0 + v * dth
        w = wmax * (1.0 - 0.9 * v ** 1.7)
        ph = (u - 0.5) * 2.0 * w
        rr = R + 0.004 + 0.016 * s * v ** 2.6 + 0.003 * cos((u - 0.5) * pi)
        return (rr * sin(t) * sin(ph), -rr * sin(t) * cos(ph), zc - rr * cos(t))
    pet = surface("pet", petal, 2, 3, thickness=0.003)
    realize(pet)
    gal = instances_to_mesh(tag + "_gallery", pet, [trs((0, 0, 0), (0, 0, k * 360.0 / npet)) for k in range(npet)], keep_base=False)
    assign(gal, gilt)
    out.append(gal)
    # opal globe: a self-luminous body (unlit, so the neighbouring lamps cannot draw wedges on it) under a thin
    # clear skin that gives the glass its highlight and sheen; two smoky inner caps shade the bowl towards the
    # lamp holder and the neck
    gs = 20 if q >= 1.0 else 16
    core = sphere(tag + "_opal", R * 0.96, loc=(0, 0, zc), segments=gs, rings=gs // 2)
    assign(core, core_m)
    smooth(core, 80)
    out.append(core)
    sk = sphere(tag + "_skin", R, loc=(0, 0, zc), segments=gs, rings=gs // 2)
    assign(sk, shell_m)
    smooth(sk, 80)
    out.append(sk)
    glass = [core, sk]                             # what the reduced levels swap for one sphere (end of this function)
    # CLEARANCE RULE: the smoky caps are the opal body's own facets pushed out (copies of the core mesh, so every
    # cap facet is parallel to the glowing facet under it at every level), the inner one 4.3 mm or more clear of the
    # core. They were lathes of another segment count at 0.972 R / 0.985 R: 2 mm nominal, and cutting through it.
    r1 = min(max(0.985 * R, 0.96 * R + 0.0043), R - 0.0003)
    for rr, tmax in ((r1, 66.0), ((r1 + R) / 2, 52.0))[:2 if q >= 1.0 else 1]:
        sh = lod_faces(duplicate(core, tag + "_shade"), keep=(0, 0, -1), above=-1, cos=cos(radians(tmax)))

        def cap(co, rr=rr, tm=radians(tmax)):
            d = co.normalized()
            if acos(max(-1.0, min(1.0, -d.z))) > tm + 1e-4:            # the rim ring comes down to the cap's edge
                h = hypot(d.x, d.y)
                d = Vector((d.x / h * sin(tm), d.y / h * sin(tm), -cos(tm)))
            return d * rr
        transform_verts(sh, cap)
        assign(sh, shade_m)
        smooth(sh, 80)
        out.append(sh)
        glass.append(sh)
    # vent cap and finial on the crown of the globe
    ct = zc + R
    fh = (0.2 if tall else 0.13) * ss
    cap = lathe(tag + "_cap", [(0.36 * R, zc + 0.922 * R), (0.385 * R, zc + 0.94 * R), (0.33 * R, zc + 0.965 * R),
                                (0.2 * R, ct + 0.004), (0.011 * ss, ct + 0.16 * fh), (0.011 * ss, ct + 0.24 * fh),
                                (0.021 * ss, ct + 0.29 * fh), (0.012 * ss, ct + 0.35 * fh),
                                (0.022 * ss, ct + 0.44 * fh), (0.03 * ss, ct + 0.54 * fh), (0.024 * ss, ct + 0.64 * fh), (0.011 * ss, ct + 0.7 * fh),
                                (0.014 * ss, ct + 0.74 * fh), (0.0, ct + fh)],
                segments=10 if q >= 1.0 else 8)
    assign(cap, iron)
    out.append(cap)
    irons.append(cap)
    tilt = (rng.jitter(0, 0.7), rng.jitter(0, 0.7), rng.u(0, 360))
    for o in out:
        apply_transform(o)
        place(o, loc=loc, rot=tilt)
    paint(cup, grime=0.25)
    paint(cap, grime=0.6)                          # dust-caked crown
    paint(gal, grime=0.3)

    # Reduced levels (tk.lod_standin; the full model is not touched): each globe is ONE glowing sphere. Opal body,
    # clear skin and smoky caps are three shells 1-7 mm apart; simplified separately they cut through each other and
    # the lamp flickers. The sphere has the opal body's radius, so the glow is as clear of the petal gallery as in lod 0.
    def one_globe(sg):
        def make():
            g = sphere(tag + "_globe_lod", R * 0.96, loc=(0, 0, zc), segments=sg, rings=sg // 2)
            assign(g, core_m)
            smooth(g, 80)
            apply_transform(g)
            place(g, loc=loc, rot=tilt)
            apply_transform(g)
            return [g]
        return make
    lod_standin(glass, mid=one_globe(gs - 4), far=one_globe(12 if q >= 1.0 else 10))
    return zc, fh, irons


# ------------------------------------------------------------------------------------------------ build
def build(p, rng):
    W = p.wear
    iron = mat("iron_painted", p.paint, wear=0.5 + 1.7 * W, special=0.55 + 1.0 * W, grime=0.8 + 0.5 * W)
    gilt = mat("brass", "#c9ad7c", rough=0.85) if p.gilt else iron
    D = p.globe_diameter
    R = D * 0.5
    n = int(p.globes)
    lit = clamp((p.intensity / 34.0) ** 0.3, 0.75, 1.35)
    core_m = mat("neon", "#ffeedb", glow=0.8 * lit)
    shell_m = mat("glass_clear", "#f4f6f8", opacity=0.2, rough=0.35)
    shade_m = mat("glass_clear", "#3a3128", opacity=0.2, rough=1.5)
    mats = (iron, gilt, core_m, shell_m, shade_m)
    q = 1.0 if n < 5 else 0.74
    H = p.height
    zs = p.stone_step if p.stone_step > 0.03 else 0.0
    ps = clamp(sqrt(H / 4.2), 0.88, 1.14)          # girth follows the height
    objs = []                                      # painted iron castings
    # lod 2 (META "lod": "far", docs/LOD.md) is modelled here: kerb, octagonal plinth, the turned column with its
    # bands and capital, the swan-neck arms with their scroll stays, and the globes on their fitters. Relief
    # (panels, rosettes, flutes, leaves, beading), fixings and the hatch are under a pixel from 20 m and left out:
    # `gone` collects what the full model builds and the far level does not show, `fk` what was made for it.
    far = lod() == 2
    gone, fk = [], []

    # ---- head dimensions first: everything above the column is sized by the globes
    s = D / 0.34
    cfit = max(0.075, 0.095 * s)
    unit_c = cfit + 1.8 * R + 0.2 * sqrt(s)        # height of the centre unit (tall finial)
    cap_h = 0.172
    S = max(p.spread, D * 0.9 + 0.1)
    kS = S / 0.5
    rise = 0.1 + 0.25 * S
    dzc = 0.62 * D + 0.1
    if n == 1:
        head_h = cap_h + 0.24 + unit_c
    else:
        head_h = cap_h + 0.11 + rise + dzc + unit_c

    # ---- stone kerb: four dressed stones per course laid as a pinwheel, open joints over a dark bed
    A = 0.76 * ps
    if zs > 0:
        st = mat("stone_rough", "#8b8578")
        r = rng.fork("step")
        two = zs > 0.2
        h1 = zs * 0.52 if two else zs
        A1 = A + (0.2 if two else 0.0)
        det = 0.15 if two else 0.1
        stones = _course("kerb_a", A1, 0.0, h1, r.fork("a"), False, 0.2 if two else det)
        beds = [box("kerb_bed_a", (A1 - 0.05, A1 - 0.05, h1 - 0.012), loc=(0, 0, (h1 - 0.012) * 0.5), segments=1)]
        if two:
            stones += _course("kerb_b", A, h1, zs - h1, r.fork("b"), True, det)
            beds.append(box("kerb_bed_b", (A - 0.05, A - 0.05, zs - h1 - 0.012), loc=(0, 0, h1 + (zs - h1 - 0.012) * 0.5), segments=1))
        if far:                                    # each course one square block: no joints, no bed, no chips
            lod_drop(stones + beds, above=1)
            hA, hB = A1 * 0.5, A * 0.5
            kr = [(hA, 0.0), (hA, h1)] + ([(hB + 0.03, h1), (hB, h1), (hB, h1 + 0.03), (hB, zs)] if two else [])
            kv = [(sx * q_, sy * q_, z) for q_, z in kr for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            kf = [(4 * k + i, 4 * k + (i + 1) % 4, 4 * k + 4 + (i + 1) % 4, 4 * k + 4 + i) for k in range(len(kr) - 1) for i in range(4)]
            kb = mesh_object("kerb_far", kv, kf + [tuple(range(len(kv) - 4, len(kv)))])
            uv_box(kb)
            stones, beds = [kb], []
        damp = noise_mask(0.16, seed=p.seed + 31, lo=0.3, hi=0.7)
        rfoot = 0.3 * ps
        for sb in stones:
            assign(sb, st)
            paint(sb,                              # every stone weathers to its own tone
                  grime=_msum(r.u(0.0, 0.32), mask_mul(height_ramp(0.0, 0.11), 0.75),
                              mask_mul(noise_mask(0.09, seed=p.seed + 32, lo=0.4, hi=0.7), 0.4 + 0.3 * W),
                              # rust washed off the iron base on to the stone
                              mask_mul(lambda co, nn: np_clip(1.0 - (np.hypot(co[:, 0], co[:, 1]) - rfoot) / 0.1), facing((0, 0, 1)), 0.25 + 0.5 * W)),
                  special=_msum(mask_mul(height_ramp(0.0, 0.1), damp, 1.0),
                                mask_mul(facing((0, 0, 1)), noise_mask(0.2, seed=p.seed, lo=0.55, hi=0.8), 0.15 + 0.3 * W)),
                  wear=mask_mul(facing((0, 0, 1)), noise_mask(0.07, seed=p.seed + 33, lo=0.5, hi=0.8), 0.5))
        for bd in beds:
            assign(bd, st)
            paint(bd, grime=1.0, special=0.6)

    # ---- stepped octagonal plinth with a panelled die
    dh = clamp(H * 0.095, 0.3, 0.48)
    op = [(0.345, 0.0), (0.345, 0.046), (0.337, 0.056), (0.3, 0.056), (0.3, 0.1), (0.292, 0.108), (0.262, 0.112), (0.245, 0.13),
          (0.235, 0.15), (0.235, 0.15 + dh), (0.245, 0.158 + dh), (0.262, 0.165 + dh), (0.262, 0.177 + dh), (0.203, 0.205 + dh),
          (0.0, 0.205 + dh)]
    if far:                                        # flange, step, die and cornice; the top stays closed under the vase
        op = lod_exact([(0.345, 0.0), (0.345, 0.056), (0.3, 0.082), (0.235, 0.15), (0.235, 0.15 + dh), (0.262, 0.17 + dh),
                        (0.203, 0.205 + dh), (0.0, 0.205 + dh)])
    ped = lathe("plinth", [(r_ * ps, z) for r_, z in op], segments=8, loc=(0, 0, zs), rot=(0, 0, 22.5), smooth_angle=25)
    apply_transform(ped)
    if not far:
        densify(ped, 0.12)
    objs.append(ped)
    fk.append(ped)
    c225, s225 = cos(radians(22.5)), sin(radians(22.5))
    apo = 0.235 * ps * c225
    fw = 2 * 0.235 * ps * s225
    zpm = zs + 0.15 + dh * 0.5
    pw, phh = fw * 0.78, dh - 0.06
    panel = shape_extrude("panel", rounded_rect(pw, phh, 0.014, 1), [rounded_rect(pw - 0.034, phh - 0.034, 0.008, 1)], depth=0.007)
    ros = lathe("rosette", [(0.0, 0.0), (0.03, 0.0), (0.033, 0.004), (0.022, 0.009), (0.012, 0.014), (0.0, 0.018)], segments=8)
    transform_verts(ros, lambda co: Vector((co.x * (1 + 0.16 * cos(4 * atan2(co.y, co.x))), co.y * (1 + 0.16 * cos(4 * atan2(co.y, co.x))), co.z))
                    if hypot(co.x, co.y) > 0.02 else co)
    pm, rm = [], []
    for k in range(8):
        a = k * 45.0
        cx, cy = apo * sin(radians(a)), -apo * cos(radians(a))
        pm.append(trs((cx, cy, zpm), (90, 0, a)))
        if k != 4 and not (k == 0 and p.number.strip()):
            rm.append(trs((cx, cy, zpm), (90, 0, a)))
    objs.append(instances_to_mesh("panels", panel, pm, keep_base=False))
    base_parts = [ped, objs[-1]]                   # reduced levels: see lod_standin below the holding-down bolts
    i_base = len(objs)
    rosm = instances_to_mesh("rosettes", ros, rm, keep_base=False)
    assign(rosm, gilt)
    paint(rosm, grime=0.2 + 0.3 * W)
    # wiring hatch on the back face: plate, two hinge knuckles, a slotted quarter-turn screw
    door = box("hatch", (pw - 0.05, 0.006, phh - 0.05), loc=(0, apo + 0.004, zpm), bevel=0.0015)
    objs.append(door)
    for sz in (-1, 1):
        objs.append(cylinder("hatch_hinge", 0.006, 0.04, loc=(-(pw - 0.05) * 0.5, apo + 0.008, zpm + sz * (phh * 0.28)), verts=8, segments=1))
    lk = bolt("hatch_lock", 0.009, 0.006, loc=((pw - 0.05) * 0.5 - 0.016, apo + 0.007, zpm), rot=(-90, 0, 0), kind="dome")
    assign(lk, gilt)
    # enamel number plate screwed inside the front panel
    num = p.number.strip()[:3]
    if num:
        ea, eb = min(fw * 0.3, 0.062), 0.04
        plate = extrude_profile("num_plate", [(ea * cos(2 * pi * i / 16), eb * sin(2 * pi * i / 16)) for i in range(16)], 0.004,
                                loc=(0, -apo, zpm), rot=(90, 0, 0))
        assign(plate, mat("ceramic_glazed", "#1c3c70"))
        paint(plate, wear=mask_mul(noise_mask(0.03, seed=p.seed + 9, lo=0.55, hi=0.8), 0.3 + 0.7 * W), grime=0.15 + 0.3 * W)
        rimp = sweep("num_rim", [(ea * cos(2 * pi * i / 16), -apo - 0.004, zpm + eb * sin(2 * pi * i / 16)) for i in range(16)], 0.003, closed=True, sides=4)
        assign(rimp, gilt)
        tsz = 0.042 if len(num) < 3 else 0.032
        tx = text("num_text", num, size=tsz, depth=0.0012, loc=(0, -apo - 0.004, zpm - tsz * 0.5), font="serif")
        assign(tx, mat("ceramic_glazed", "#eee7d6"))
        if far:                                    # the enamel plate is a blue patch on the die: one face, 8 mm proud
            gone += [plate, rimp, tx]
            pfar = mesh_object("num_plate_far", [(ea * cos(2 * pi * i / 6), -apo - 0.008, zpm + eb * sin(2 * pi * i / 6)) for i in range(6)],
                               [tuple(range(6))])
            uv_box(pfar)
            assign(pfar, mat("ceramic_glazed", "#1c3c70"))
    # holding-down bolts through the bottom flange
    rbolt = (0.345 + 0.3) * 0.5 * ps * c225
    hb = join([cylinder("nut", 0.012, 0.011, loc=(0, 0, 0.0085), verts=6, segments=1), cylinder("washer", 0.018, 0.003, loc=(0, 0, 0.0015), verts=6, segments=1),
               cylinder("stud", 0.006, 0.006, loc=(0, 0, 0.016), verts=4, segments=1)], "hd_bolt")
    bolt_pts = [(rbolt * cos(radians(k * 45)), rbolt * sin(radians(k * 45)), zs + 0.056) for k in range(8)]
    objs.append(instances_to_mesh("hd_bolts", hb, [trs(bp, (0, 0, rng.u(0, 60))) for bp in bolt_pts], keep_base=False))
    if far:                                        # panel frames, rosettes, hatch, hinges, lock and bolts
        gone += [o for o in objs if o is not ped] + [rosm, lk]

    # Reduced levels (tk.lod_standin; the full model is not touched): the plinth as one clean octagonal shell, so
    # the level does not collapse the densified casting, the panel frames, the hatch and the bolts into one block.
    #   lod 1: the full moulding profile (not densified) + the eight panel frames; hatch, hinges, lock, bolts go
    #   lod 2: flange, step, die and cornice on ten profile points, nothing else
    # Rosettes, number plate and its lettering stay with the reducer.
    def plinth_lod(prof, panels):
        def make():
            o = lathe("plinth_lod", [(r_ * ps, z) for r_, z in prof], segments=8, loc=(0, 0, zs), rot=(0, 0, 22.5), smooth_angle=25)
            apply_transform(o)
            assign(o, iron)
            if not panels:
                return [o]
            pn = shape_extrude("panel_lod", rounded_rect(pw, phh, 0.014, 1), [rounded_rect(pw - 0.034, phh - 0.034, 0.008, 1)], depth=0.007)
            pl = instances_to_mesh("panels_lod", pn, pm, keep_base=False)
            assign(pl, iron)
            return [o, pl]
        return make
    op_far = [(0.345, 0.0), (0.345, 0.056), (0.3, 0.056), (0.3, 0.108), (0.235, 0.15), (0.235, 0.15 + dh), (0.262, 0.165 + dh),
              (0.262, 0.177 + dh), (0.203, 0.205 + dh), (0.0, 0.205 + dh)]
    lod_standin(base_parts + objs[i_base:] + [lk], mid=plinth_lod(op, True), far=plinth_lod(op_far, False))

    # ---- turned vase: torus, scotia, gadrooned bulb, necking
    z_oct = zs + 0.205 + dh
    r0, r1 = 0.08 * ps, 0.054 * ps
    rp = [(0.19, 0.0), (0.207, 0.01), (0.212, 0.025), (0.205, 0.042), (0.185, 0.052), (0.152, 0.075), (0.148, 0.095), (0.158, 0.105),
          (0.158, 0.115), (0.168, 0.128), (0.178, 0.158), (0.172, 0.19), (0.158, 0.218), (0.137, 0.25), (0.116, 0.275),
          (0.112, 0.285), (0.123, 0.292), (0.123, 0.305), (0.108, 0.312)]
    vase_h = 0.33
    ng = 10
    vase = lathe("vase", [(r_ * ps, z) for r_, z in rp] + [(r0 + 0.03, vase_h)], segments=ng * (4 if q >= 1.0 else 3), loc=(0, 0, z_oct))

    def gadroon(co):
        if co.z < 0.118 or co.z > 0.28:
            return co
        e = smoothstep(0.118, 0.15, co.z) * (1.0 - smoothstep(0.24, 0.28, co.z))
        cc = 0.5 + 0.5 * cos(atan2(co.y, co.x) * ng)
        k = 1.0 + 0.055 * e * (cc - 0.5)
        return Vector((co.x * k, co.y * k, co.z))
    transform_verts(vase, gadroon)
    smooth(vase, 75)
    objs.append(vase)

    # ---- column: tapering, fluted between turned bands
    z0 = z_oct + vase_h
    z1 = max(H - head_h, z0 + 0.45)
    L = z1 - z0
    nf = int(p.flutes)

    def rad(z):
        return r0 + (r1 - r0) * clamp((z - z0) / L)
    fz0, fz1 = z0 + 0.15, z1 - 0.05
    # one or two beaded bands ("bagues"); each also houses a small light, see the end of build()
    bands = [fz0 + (fz1 - fz0) * f for f in ((0.2, 0.63) if L > 1.4 else (0.3,) if L > 0.9 else ())]
    zl = {z0 + 0.07, fz0, fz0 + 0.025, fz0 + 0.05, fz1 - 0.05, fz1 - 0.025, fz1}
    for band in bands:
        zl |= {band - 0.085, band - 0.052, band + 0.052, band + 0.085}
    zq = fz0 + 0.2
    while zq < fz1 - 0.08:                        # a ring every 0.3 m carries the weathering masks
        zl.add(zq)
        zq += max(0.3, L / 6.5)
    zz = []
    for z in sorted(zl):
        if not zz or z - zz[-1] > 0.012:
            zz.append(z)
    sp = [(r0 + 0.03, z0 - 0.002), (r0 + 0.035, z0 + 0.012), (r0 + 0.024, z0 + 0.03), (r0 + 0.007, z0 + 0.045)]
    sp += [(rad(z), z) for z in zz]
    shaft = lathe("shaft", sp + [(r1, z1)], segments=max(32, nf * 3), smooth_angle=60)   # 3 per flute: arris, two floor verts
    fd = 0.0058 * ps

    def flute(co):
        z = co.z
        if nf <= 0 or z < fz0 or z > fz1:
            return co
        e = min(smoothstep(fz0, fz0 + 0.05, z), 1.0 - smoothstep(fz1 - 0.05, fz1, z))
        for band in bands:
            e = min(e, smoothstep(0.052, 0.085, abs(z - band)))
        cc = 0.5 + 0.5 * cos(atan2(co.y, co.x) * nf)
        r_ = hypot(co.x, co.y)
        kk = (r_ - fd * e * (1.0 - cc ** 3)) / r_ if r_ > 1e-6 else 1.0
        return Vector((co.x * kk, co.y * kk, co.z))
    transform_verts(shaft, flute)
    objs.append(shaft)
    bead = sphere("bead", 0.0085, segments=6, rings=3)
    nb = 16
    bmats = []
    for band in bands:
        rr = rad(band)
        objs.append(lathe("band", [(rr - 0.002, band - 0.05), (rr + 0.007, band - 0.042), (rr + 0.01, band - 0.03), (rr + 0.022, band - 0.017),
                                   (rr + 0.022, band + 0.017), (rr + 0.01, band + 0.03), (rr + 0.007, band + 0.042), (rr - 0.002, band + 0.05)],
                          segments=20))

        # Reduced levels (tk.lod_standin; the full model is not touched): the band houses a light. As a declared
        # part it and the shaft wall behind it stay whole, so that light stays shut in the column as in the full
        # model. lod 1: the profile on 12 sides; lod 2: one bevelled ring on 8.
        def band_lod(prof, sg, band=band):
            def make():
                o = lathe("band_lod", [(r_, band + dz_) for r_, dz_ in prof], segments=sg)
                assign(o, iron)
                return [o]
            return make
        lod_standin(objs[-1], mid=band_lod([(rr - 0.002, -0.05), (rr + 0.007, -0.042), (rr + 0.01, -0.03), (rr + 0.022, -0.017), (rr + 0.022, 0.017),
                                            (rr + 0.01, 0.03), (rr + 0.007, 0.042), (rr - 0.002, 0.05)], 12),
                    far=band_lod([(rr - 0.002, -0.05), (rr + 0.022, -0.017), (rr + 0.022, 0.017), (rr - 0.002, 0.05)], 8))
        bmats += [trs(((rr + 0.0225) * cos(2 * pi * i / nb), (rr + 0.0225) * sin(2 * pi * i / nb), band)) for i in range(nb)]
    if bmats:
        bm = instances_to_mesh("band_beads", bead, bmats, keep_base=False)
        smooth(bm, 80)
        assign(bm, gilt)
        paint(bm, grime=0.25 + 0.35 * W)
        gone.append(bm)
    # cast acanthus collar at the foot of the column: relief leaves hugging the shaft, tips turning out
    sink = (fd if nf > 0 else 0.0) + 0.0025
    lh = 0.21 * ps
    objs.append(_leaf_ring("leaves_foot", 8, z0 + 0.022, lh, rad, 2 * pi * r0 / 8 * 0.97, 0.0145 * ps, 0.02, sink, phase=22.5, alt=0.8,
                           nv=10 if q >= 1.0 else 8))

    # ---- capital: astragal, leafy bell, moulded abacus with beading
    cp = [(r1 - 0.002, -0.02), (r1 + 0.004, -0.012), (r1 + 0.016, -0.004), (r1 + 0.018, 0.008), (r1 + 0.016, 0.02), (r1 + 0.004, 0.028), (r1 + 0.002, 0.04),
          (r1 + 0.006, 0.08), (r1 + 0.022, 0.115), (r1 + 0.046, 0.134), (r1 + 0.06, 0.14), (r1 + 0.066, 0.147), (r1 + 0.066, 0.162), (r1 + 0.058, 0.168),
          (0.04, cap_h), (0.0, cap_h)]
    objs.append(lathe("capital", cp, segments=20, loc=(0, 0, z1)))
    bell = cp[5:10]
    objs.append(_leaf_ring("leaves_cap", 8, z1 + 0.014, 0.118, lambda z: _prof_r(bell, z - z1), 2 * pi * (r1 + 0.012) / 8 * 0.98, 0.009, 0.016, 0.003,
                           phase=22.5, nv=6, cols=(-1.0, -0.75, -0.3, 0.0, 0.3, 0.75, 1.0)))
    bead2 = sphere("bead2", 0.0075, segments=6, rings=3)
    nb2 = 16
    ab = instances_to_mesh("abacus_beads", bead2, [trs(((r1 + 0.067) * cos(2 * pi * i / nb2), (r1 + 0.067) * sin(2 * pi * i / nb2), z1 + 0.1545))
                                                    for i in range(nb2)], keep_base=False)
    smooth(ab, 80)
    assign(ab, gilt)
    gone.append(ab)
    zc = z1 + cap_h
    # far: vase, column, leaf collar, bands, capital and the hub / stem above it are ONE six-sided turning (a closed
    # shell, so the lights inside the bands stay shut in as in the full model); its profile is finished in the head
    tp = [(0.205 * ps, z_oct), (0.15 * ps, z_oct + 0.085), (0.178 * ps, z_oct + 0.158), (r0 + 0.032, z0),
          (rad(z0 + 0.022 + lh * 0.92) + 0.026, z0 + 0.022 + lh * 0.92), (rad(z0 + 0.034 + lh), z0 + 0.034 + lh)]
    for band in bands:
        tp += [(rad(band - 0.05), band - 0.05), (rad(band) + 0.026, band), (rad(band + 0.05), band + 0.05)]
    tp += [(r1 + 0.012, z1 + 0.03), (r1 + 0.04, z1 + 0.115), (r1 + 0.066, z1 + 0.147), (r1 + 0.066, z1 + 0.165), (0.04, zc)]

    # ---- head
    bt, bw = 0.013, 0.032                          # flat bar of the stays: 13 x 32 mm
    flat_bar = [(-bt * 0.5, -bw * 0.5), (bt * 0.5, -bw * 0.5), (bt * 0.5, bw * 0.5), (-bt * 0.5, bw * 0.5)]
    tri_bar = [(0.016, 0.0), (-0.008, 0.0139), (-0.008, -0.0139)]      # far: the same bar with three sides
    centres = []                                   # (x, y, z) of every globe centre
    rivets = []                                    # matrices of the gilt rivets fixing the stays
    joints = []                                    # where rust starts: collars, lugs, clamps
    gr = rng.fork("globes")
    if n == 1:
        objs.append(lathe("stem", [(0.04, 0.0), (0.047, 0.01), (0.04, 0.02), (0.034, 0.04), (0.03, 0.09), (0.04, 0.13), (0.052, 0.16), (0.044, 0.19),
                                   (0.03, 0.21), (0.024, 0.245)], segments=20, loc=(0, 0, zc)))
        zb = zc + 0.24
        rimx = 0.6 * R + 0.012
        for k in range(4):
            a = 45 + k * 90
            Rz = Matrix.Rotation(radians(a), 4, 'Z')
            pts = bezier_path((0.034, 0, zc + 0.06), (0.15, 0, zc + 0.03), (0.6 * R + 0.1, 0, zb - 0.02), (rimx, 0, zb + cfit * 0.84), n=14)
            st_ = sweep(f"stay{k}", pts, flat_bar, up=(0, 1, 0), scale_fn=lambda t: 1.0 - 0.15 * t)
            objs.append(_orbit(st_, a))
            if far:                                # the stay as a three-sided bar in three runs; no scroll under it
                sf = sweep(f"stay{k}", lod_exact([pts[i] for i in (0, 4, 9, 13)]), tri_bar, up=(0, 1, 0), smooth_path=False)
                objs.append(_orbit(sf, a))
                fk.append(sf)
            sc = spiral_path(0.007, 0.045, 1.35, n=18, a0_deg=215, plane="XZ", center=(0.105, 0, zc + 0.098))
            so = sweep(f"scroll{k}", sc, flat_bar, up=(0, 1, 0), scale_fn=lambda t: 0.6 + 0.4 * t)
            objs.append(_orbit(so, a))
            for px, pz in ((pts[1][0], pts[1][2]), (pts[-2][0], pts[-2][2])):
                for sy in (-1, 1):
                    rivets.append(Rz @ trs((px, sy * bw * 0.5, pz), (-90 * sy, 0, 0)))
            joints.append((0.06 * cos(radians(a)), 0.06 * sin(radians(a)), zc + 0.06))
        zcg, fh, ir = _globe_unit("g0", (0, 0, zb), D, True, mats, q, gr)
        objs += ir
        centres.append((0, 0, zb + zcg))
        tp += [(0.03, zc + 0.09), (0.052, zc + 0.16), (0.024, zc + 0.245), (0.0, zc + 0.245)]
    else:
        za = zc + 0.11
        zarm = za + rise                           # arm fitter base
        zcb = zarm + dzc                           # centre fitter base
        top = zcb - zc
        m = 0.22 + (top - 0.22) * 0.52
        hubp = [(0.04, 0.0), (0.047, 0.01), (0.04, 0.02), (0.05, 0.04), (0.072, 0.075), (0.079, 0.11), (0.071, 0.14), (0.046, 0.17),
                (0.034, 0.19), (0.041, 0.2), (0.041, 0.21), (0.03, 0.22), (0.027, m - 0.035), (0.04, m - 0.016), (0.045, m),
                (0.04, m + 0.016), (0.027, m + 0.035), (0.023, top - 0.012), (0.023, top + 0.004)]
        objs.append(lathe("hub", hubp, segments=20 if q >= 1.0 else 16, loc=(0, 0, zc)))
        hb_ring = torus("hub_ring", 0.079, 0.006, loc=(0, 0, zc + 0.11), major_seg=20, minor_seg=5)
        assign(hb_ring, gilt)
        na = n - 1
        tube = 0.0185
        aseg = 12 if q >= 1.0 else 8
        npt = int(22 * q)
        pend = [(0.0, -0.085), (0.006, -0.075), (0.013, -0.058), (0.007, -0.04), (0.016, -0.028), (0.022, -0.008), (0.014, 0.004), (0.0, 0.004)]
        # leaf husk the arm grows out of (local +Z = along the arm)
        husk_p = [(0.02, -0.018), (0.034, -0.005), (0.037, 0.004), (0.031, 0.011), (0.03, 0.016), (0.038, 0.03), (0.041, 0.046), (0.038, 0.064),
                  (0.032, 0.082), (0.0275, 0.094), (0.032, 0.102), (0.031, 0.107), (0.02, 0.109)]

        def arm_scale(t):
            return 0.84 + 0.16 * (1.0 - t) + 0.45 * max(0.0, 1.0 - t / 0.24) ** 2 + 0.14 * smoothstep(0.9, 1.0, t)
        r_b = 0.066 * kS + (0.014 - 0.066 * kS) * (0.5 / 1.4)    # radius of the scroll where it comes round to its lowest point
        for k in range(na):
            a = k * 360.0 / na
            ca, sa = cos(radians(a)), sin(radians(a))
            Rz = Matrix.Rotation(radians(a), 4, 'Z')
            dz = gr.jitter(0, 0.004)
            pts = bezier_path((0.05, 0, za), (S * 0.5, 0, za - 0.26 * kS), (S * 1.1, 0, za - 0.22 * kS), (S, 0, zarm + dz - 0.03), n=npt)
            arm = sweep(f"arm{k}", pts, tube, sides=10 if q >= 1.0 else 8, smooth_path=False, scale_fn=arm_scale)
            objs.append(_orbit(arm, a))
            if far:                                # the swan neck as a three-sided tube in six runs
                ta_ = tube * 1.2
                af = sweep(f"arm{k}", lod_exact([pts[i] for i in sorted({round(j * (len(pts) - 1) / 6) for j in range(7)})]),
                           [(ta_, 0.0), (-0.5 * ta_, 0.866 * ta_), (-0.5 * ta_, -0.866 * ta_)], smooth_path=False, scale_fn=arm_scale)
                objs.append(_orbit(af, a))
                fk.append(af)
            # ball boss where the arm enters the urn, then the ribbed leaf husk it grows out of
            i3 = max(2, round(0.15 * (len(pts) - 1)))
            tx_, tz_ = pts[i3][0] - pts[0][0], pts[i3][2] - pts[0][2]
            tl = hypot(tx_, tz_)
            tx_, tz_ = tx_ / tl, tz_ / tl
            bx, bz = 0.05 + tx_ * 0.032, za + tz_ * 0.032
            boss = sphere(f"arm_boss{k}", 0.039, loc=(bx, 0, bz), segments=aseg, rings=6)
            objs.append(_orbit(boss, a))
            husk = lathe(f"arm_husk{k}", husk_p, segments=aseg, loc=(bx + tx_ * 0.02, 0, bz + tz_ * 0.02), rot=(0, degrees(atan2(tx_, tz_)), 0))
            nl = aseg // 2
            transform_verts(husk, lambda co: Vector((co.x * (1 + 0.1 * cos(nl * atan2(co.y, co.x))), co.y * (1 + 0.1 * cos(nl * atan2(co.y, co.x))), co.z))
                            if 0.02 < co.z < 0.09 else co)
            objs.append(_orbit(husk, a))
            joints.append((bx * ca, bx * sa, bz))
            # gilt pendant under the lowest point of the swan neck
            low = min(pts, key=lambda pt: pt[2])
            pd = lathe(f"pendant{k}", pend, segments=8, loc=(low[0], 0, low[2] - tube * arm_scale(pts.index(low) / (len(pts) - 1)) * 0.9))
            _orbit(pd, a)
            assign(pd, gilt)
            paint(pd, grime=0.6)
            if far:                                # the gilt pendant as a three-sided drop, its top inside the arm
                gone.append(pd)
                pdf = lathe(f"pendant{k}", lod_exact([(0.0, -0.085), (0.02, -0.012), (0.0, 0.014)]), segments=lod_exact(3),
                            loc=(low[0], 0, low[2] - tube * arm_scale(pts.index(low) / (len(pts) - 1)) * 0.9), smooth_angle=95)
                _orbit(pdf, a)
                assign(pdf, gilt)
            # flat-bar scroll stay: riveted on to a cast lug on the urn shoulder, curling down on to a clamp on the arm
            im = int(len(pts) * 0.47)
            mid = pts[im]
            ta = tube * arm_scale(im / (len(pts) - 1))
            ro = 0.066 * kS
            C = (mid[0], mid[2] + ta + 0.007 + bt * 0.5 + r_b)
            Az = C[1] + ro + 0.07 * kS
            ru = _prof_r(hubp, Az - zc)
            sc = sweep(f"stay{k}", _scroll((ru - 0.012, Az), C, ro, 0.014, 1.4, n=npt), flat_bar, up=(0, 1, 0), smooth_path=False,
                       scale_fn=lambda t: 1.0 - 0.35 * smoothstep(0.7, 1.0, t))
            objs.append(_orbit(sc, a))
            if far:                                # the scroll stay stays in the outline: seven runs of a three-sided bar
                sp_ = _scroll((ru - 0.012, Az), C, ro, 0.014, 1.4, n=7)
                sf = sweep(f"stay{k}", lod_exact([sp_[i] for i in (0, 3, 6, 7, 8, 9, 10, 11)]), tri_bar, up=(0, 1, 0), smooth_path=False,
                           scale_fn=lambda t: 1.0 - 0.3 * smoothstep(0.7, 1.0, t))
                objs.append(_orbit(sf, a))
                fk.append(sf)
            lug = box(f"lug{k}", (0.064, 0.036, 0.022), loc=(ru + 0.016, 0, Az - bt * 0.5 - 0.011), bevel=0.004, segments=1)
            objs.append(_orbit(lug, a))
            for rx in (0.014, 0.036):
                rivets.append(Rz @ trs((ru + rx, 0, Az + bt * 0.5), (0, 0, 0)))
            joints.append((ru * ca, ru * sa, Az))
            # clamp: a strap round the arm and a saddle block the scroll is riveted to
            strap = cylinder(f"strap{k}", ta + 0.0045, 0.032, loc=(mid[0], 0, mid[2]), rot=(0, 90, 0), verts=aseg, bevel=0.0015, segments=1)
            objs.append(_orbit(strap, a))
            sad = box(f"saddle{k}", (0.03, bw + 0.008, 0.024), loc=(mid[0], 0, mid[2] + ta + 0.004), bevel=0.003, segments=1)
            objs.append(_orbit(sad, a))
            for sy in (-1, 1):
                rivets.append(Rz @ trs((mid[0], sy * (bw * 0.5 + 0.004), mid[2] + ta + 0.006), (-90 * sy, 0, 0)))
            joints.append((mid[0] * ca, mid[0] * sa, mid[2]))
            zcg, fh, ir = _globe_unit(f"g{k + 1}", (S * ca, S * sa, zarm + dz), D, False, mats, q, gr)
            objs += ir
            fk += ir
            centres.append((S * ca, S * sa, zarm + dz + zcg))
        zcg, fh, ir = _globe_unit("g0", (0, 0, zcb), D, True, mats, q, gr)
        objs += ir
        centres.insert(0, (0, 0, zcb + zcg))
        gone.append(hb_ring)
        tp += [(0.079, zc + 0.11), (0.036, zc + 0.2), (0.025, zc + top + 0.004), (0.0, zc + top + 0.004)]
    rv = instances_to_mesh("rivets", _dome("rivet", 0.007, 0.005), rivets, keep_base=False)
    assign(rv, gilt)
    if far:
        fk += ir
        trunk = lathe("shaft", lod_exact(tp), segments=lod_exact(6), smooth_angle=75)
        objs.append(trunk)
        fk.append(trunk)
        gone += [o for o in objs if not any(o is f for f in fk)] + [rv]

    for o in objs:
        assign(o, iron)

    # ---- art-directed ageing (everything scales with `wear`; edge chipping also via the material gains above)
    jrust = _near(joints, 0.07)
    run_a = _streaks(p.seed + 41, 0.022, 0.5, 0.42, 0.7)
    run_b = _streaks(p.seed + 42, 0.03, 0.7, 0.4, 0.72)
    blot = noise_mask(0.06, seed=p.seed + 2, lo=0.4, hi=0.68)
    scuff = noise_mask(0.035, seed=p.seed + 3, lo=0.5, hi=0.72)
    brun = [mask_mul(run_a, height_ramp(zb_ - 0.03, zb_ - 0.7), _below(zb_ - 0.01), _above(zb_ - 0.72), 0.2 + 1.0 * W) for zb_ in bands]
    for o in objs:
        nm = o.name
        if nm.startswith("vase"):
            # rust in the scotia and bleeding down from the necking, chipped paint on the shoulder, dust on top
            paint(o, special=_msum(mask_mul(height_ramp(z_oct + 0.07, z_oct + 0.13), height_ramp(z_oct + 0.1, z_oct + 0.04), 0.3 + 0.6 * W),
                                   mask_mul(run_a, height_ramp(z_oct + 0.3, z_oct + 0.12), 1.1 * W)),
                  wear=mask_mul(facing((0, 0, 1)), height_ramp(z_oct + 0.2, z_oct + 0.31), scuff, 0.2 + 1.0 * W),
                  grime=mask_mul(facing((0, 0, 1)), 0.4))
        elif nm.startswith(("plinth", "panels", "hatch")):
            # rust weeping from the holding-down bolts and along the foot of the flange, runs down the die,
            # boots and bicycle pedals scuffing the lowest 0.4 m
            paint(o, special=_msum(mask_mul(_near(bolt_pts, 0.085), 0.55 + 0.6 * W), mask_mul(height_ramp(zs, zs + 0.05), blot, 0.4 + 0.7 * W),
                                   mask_mul(run_b, height_ramp(zs + 0.2 + dh, zs + 0.12), 1.15 * W)),
                  grime=_msum(mask_mul(height_ramp(zs, zs + 0.35), 0.45), mask_mul(facing((0, 0, 1)), 0.3)),
                  wear=mask_mul(height_ramp(zs + 0.03, zs + 0.45), scuff, 0.15 + 1.0 * W))
        elif nm.startswith("hd_bolts"):
            paint(o, special=0.45 + 0.55 * W, grime=0.4)
        elif nm.startswith(("shaft", "band", "leaves_foot")):
            # rust bleeding down from the beaded band, rust in the collar, rain streaks below the capital,
            # hands polishing the paint at grip height
            paint(o, special=_msum(*brun,
                                   mask_mul(height_ramp(z0 + 0.02, z0 + 0.12), blot, 0.3 + 0.6 * W),
                                   mask_mul(noise_mask(0.1, seed=p.seed + 6, lo=0.55, hi=0.8), 0.6 * W)),
                  grime=_msum(mask_mul(run_b, height_ramp(z1 - 0.02, z1 - 0.9), 0.45 + 0.45 * W), mask_mul(height_ramp(z0, z0 + 0.3), 0.3)),
                  wear=mask_mul(height_ramp(1.3, 1.8), height_ramp(1.3, 0.8), scuff, 0.2 + 0.7 * W))
        else:                                      # capital and head ironwork
            paint(o, special=_msum(mask_mul(jrust, 0.3 + 0.7 * W),
                                   mask_mul(facing((0, 0, -1)), noise_mask(0.07, seed=p.seed + 7, lo=0.35, hi=0.7), 0.15 + 0.8 * W)),
                  grime=_msum(mask_mul(facing((0, 0, 1)), 0.35), 0.12))

    lod_drop(gone, above=1)

    # ---- lights and sockets
    # The opal globes are self-luminous (unlit glass). The engine has no near-field roll-off, so a light of the full
    # output inside a globe bleaches every casting within arm's reach. The output is therefore split over three small
    # lights for any number of globes: one in the centre globe rims the head, and the pool on the ground is carried by
    # one light inside each beaded band of the column, far enough below the head not to graze it.
    lc = p.light_color
    pool = p.intensity * (0.4 + 0.2 * (n - 1))
    head = clamp(0.13 * p.intensity, 2.5, 4.5)
    light("globe0", centres[0], color=lc, intensity=head, range=6.0, flicker=0.02, halo=head + pool, halo_at=centres,
          halo_r=p.globe_diameter * 0.5)
    hosts = [(bands[0], 0.68), (bands[1], 0.32)] if len(bands) > 1 else [(bands[0] if bands else z0 + L * 0.5, 1.0)]
    for j, (z, f) in enumerate(hosts):
        light(f"pool{j}", (0, 0, z), color=lc, intensity=pool * f, range=11.0, flicker=0.02, shadow_near=H + 1.0)
    socket("banner_l", "attach", (-rad(z0 + L * 0.8) - 0.01, 0, z0 + L * 0.8), rot=(0, 0, 90), role="banner")
    socket("basket_r", "attach", (rad(z0 + L * 0.62) + 0.01, 0, z0 + L * 0.62), rot=(0, 0, -90), role="planter")
