"""Four-sided tapered gas lantern head (Victorian street lantern).

Construction: cast corner astragals with glazing rebates, bottom / top (and optional mid) rails,
3.5 mm near-clear seeded panes (own bow, tilt and ripple per pane) bedded in hand-run putty with
soot films in the top third and dust in the bottom corners, a framed door leaf on the front with
two hinges, turn latch and knob, a moulded cornice carrying a continuous pierced fleur cresting and
turned corner finials, a sheet-iron roof in satin paint (ogee / pyramid / dome) with hip rolls,
eave rivets and rust runs, a formed square-to-round collar under the stepped vent chimney, a sooty
enamel reflector closing the roof underside and a brass burner (union, stopcock with by-pass lever
and pull rods, gallery, glowing mantle). Soot, dust, rust, chipped paint, bent cresting, a slight
lean and pane damage are driven by `age`, `broken` and the seed.

Component: origin = centre of the bottom collar (sits on a post top / bracket frog, four cradle
arms under the tray). With mount='hang' the collar is replaced by a turned drip pendant and the
lantern hangs from a four-strap crown; the origin is then the bearing point of the suspension eye.
"""
from math import atan2, cos, hypot, pi, radians, sin, sqrt

import numpy as np
from mathutils import Matrix, Vector

from tk import *

META = {"name": "Gas lantern head", "class": "component", "mount": "ground", "tri_budget": 9000,
        "focus": 0.45, "lod": "far"}
PARAMS = {
    "width": {"default": 0.36, "min": 0.22, "max": 0.6, "desc": "width across the top of the glazed body (m)"},
    "height": {"default": 0.46, "min": 0.28, "max": 0.8, "desc": "height of the glazed body (m)"},
    "taper": {"default": 0.56, "min": 0.4, "max": 1.0, "desc": "bottom width / top width"},
    "roof": {"default": "ogee", "choices": ["ogee", "pyramid", "dome"], "desc": "roof profile"},
    "mount": {"default": "post", "choices": ["post", "hang"], "desc": "sits on a post top, or hangs from an eye"},
    "paint": {"default": "#1c2420", "desc": "paint colour of the ironwork"},
    "crown": {"default": True, "desc": "pierced cresting along the roof eaves"},
    "intensity": {"default": 55.0, "min": 5, "max": 200, "desc": "light intensity (cd)"},
    "lit": {"default": True, "desc": "burner alight"},
    "mid_rail": {"default": 0.76, "min": 0.0, "max": 0.85,
                 "desc": "height fraction of a horizontal glazing bar across every side (0 = single pane)"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0,
            "desc": "condition: 0 = freshly painted and plumb, 1 = sooty, rusty, leaning, bent cresting"},
    "broken": {"default": "none", "choices": ["none", "cracked", "missing"], "desc": "damage to one seeded pane"},
}
VARIANTS = {
    "hanging": {"mount": "hang", "roof": "dome", "width": 0.3, "height": 0.4, "seed": 3, "paint": "#1e3550",
                "mid_rail": 0.0, "age": 0.3, "intensity": 10.0},      # small passage lantern on a low by-pass flame
    "wide": {"width": 0.48, "height": 0.5, "taper": 0.7, "roof": "pyramid", "paint": "#1d5230", "seed": 7,
             "mid_rail": 0.0, "age": 0.15},
    "oxblood_worn": {"width": 0.34, "height": 0.52, "taper": 0.5, "paint": "#6e1814", "age": 0.95,
                     "broken": "cracked", "mid_rail": 0.7, "seed": 11},
    "plain_missing": {"width": 0.32, "height": 0.58, "taper": 0.64, "roof": "pyramid", "crown": False,
                      "paint": "#565c5e", "age": 0.8, "broken": "missing", "mid_rail": 0.0, "seed": 16},
}

S2 = sqrt(2.0)


def _loft(name, rings, closed=True, loop=False):
    """Quad skin between consecutive rings of 3D points (all rings the same length)."""
    m, nr = len(rings[0]), len(rings)
    verts = [tuple(q) for r in rings for q in r]
    faces = []
    for i in range(nr if loop else nr - 1):
        i1 = (i + 1) % nr
        for k in range(m if closed else m - 1):
            k1 = (k + 1) % m
            faces.append((i * m + k, i * m + k1, i1 * m + k1, i1 * m + k))
    return mesh_object(name, verts, faces)


def _orient(o, want):
    """Open sheets: flip the (consistently wound) faces so face 0 looks along want(centre)."""
    me = o.data
    me.update()
    f = me.polygons[0]
    if f.normal.dot(want(f.center)) < 0:
        me.flip_normals()
        me.update()
    return o


def _xf(o, M):
    """Bake a 4x4 into the vertices (kit parts are modelled in a local frame, then placed)."""
    apply_transform(o)
    transform_verts(o, lambda co: M @ co)
    return o


def _rot_to(v):
    return Vector((0, 0, 1)).rotation_difference(Vector(v).normalized()).to_matrix().to_4x4()


def _T(x, y, z):
    return Matrix.Translation((x, y, z))


def _build_far(p, rng):
    """lod 2 (from 44 m on the whole head is 6-10 pixels): the tray with its collar or pendant, four corner bars, the
    cornice, one pane a side, roof + vent + finial as one closed turning, the crown of a hung lantern as four flat
    straps under a boss, the eaves cresting as one band a side, the burner as one turning and the mantle as a gem.
    CLEARANCE RULE: the glass stands 8.8 mm behind the nominal face as at lod 0 and nothing lies parallel to it any
    closer - the corner bars are set on the diagonal, so no face of theirs is parallel to a pane. Same dimensions,
    materials, lean, light and socket as build(); only forks of rng are drawn from, as there."""
    age = float(p.age)
    lit = bool(p.lit)
    lin = hex_rgb(p.paint)
    lifted = tuple(min(1.0, c * 1.8 + 0.05) for c in lin[:3])
    iron = mat("iron_painted", lifted, scale=5.0, rough=3.5)
    sheet = mat("plain", tuple(min(1.0, c * 1.15) for c in lifted), rough=0.45)
    forged = mat("iron_wrought")
    brass = mat("brass")
    glass = mat("glass_clear", "#2a2927", opacity=0.1)
    mantle_m = mat("emissive", "#ffa850", glow=5.0 if lit else 0.0)
    wt, wb, h = p.width, p.width * p.taper, p.height
    ht, hb = wt / 2, wb / 2
    z0 = 0.07
    z1 = z0 + h
    R = hb * S2
    zb = z0 - 0.012
    cr = ht * S2
    Rr = cr + 0.034
    Fh = Rr / S2
    zt = z1 + 0.02
    dome = p.roof == "dome"
    re = cr + 0.008
    rh = wt * (0.42 if p.roof != "pyramid" else 0.36)
    rtop = re * 0.26
    rvs = rtop * 0.84
    zr = zt + 0.008
    zc = zr + rh
    ztop = zc + 0.19
    X4 = lod_exact(4)
    objs = []

    def add(o, m):
        assign(o, m)
        objs.append(o)
        return o

    def mesh(name, vs, fs, m):
        o = mesh_object(name, vs, fs)
        uv_box(o)
        return add(o, m)

    blotch = noise_mask(0.07, seed=p.seed + 9, lo=0.45, hi=0.8)
    # tray, and what carries it
    tray = add(lathe("tray", [(0.0, zb), (R + 0.012, zb - 0.005), (R + 0.016, z0 - 0.002), (R + 0.01, z0 + 0.008), (R + 0.011, z0 + 0.022),
                              (R - 0.016, z0 + 0.022), (R - 0.016, z0 + 0.006), (0.0, z0 + 0.006)],
                     segments=X4, rot=(0, 0, 45), smooth_angle=20), iron)
    paint(tray, grime=0.5, special=mask_mul(blotch, 0.2 + 0.7 * age))
    if p.mount == "post":
        add(lathe("collar", [(0.0, 0.0), (0.034, 0.0), (0.026, 0.03), (0.048, 0.057)], segments=lod_exact(5)), iron)
    else:
        add(lathe("pendant", [(0.0, zb - 0.078), (0.015, zb - 0.04), (0.012, zb - 0.02), (0.035, zb + 0.001)], segments=X4), iron)
    # corner bars on the diagonal, cornice
    e = 0.017                                   # (seen square on a little less than bead + glazing fins, on the diagonal a little more)
    vs, fs = [], []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        b = len(vs)
        for q, z in ((hb, z0 + 0.004), (ht, z1 - 0.006)):
            vs += [(sx * q + dx, sy * q + dy, z) for dx, dy in ((e, 0), (0, e), (-e, 0), (0, -e))]
        fs += [(b + i, b + (i + 1) % 4, b + 4 + (i + 1) % 4, b + 4 + i) for i in range(4)]
    bars = mesh("bars", vs, fs, iron)
    paint(bars, grime=0.35, special=mask_mul(height_ramp(z0, z0 + 0.1 + 0.1 * age), blotch, 0.3 + 0.7 * age))
    top = add(lathe("toprail", [(cr - 0.014, z1 - 0.02), (cr + 0.004, z1 - 0.02), (cr + 0.006, z1 - 0.008), (cr + 0.03, z1 + 0.008), (Rr, zt), (cr - 0.02, zt),
                                (cr - 0.014, z1 - 0.02)],       # closed on the inside too: it is looked at from below, through the glass
                    segments=X4, rot=(0, 0, 45), smooth_angle=20), iron)
    paint(top, special=mask_mul(blotch, 0.1 + 0.25 * age), grime=lambda co, n: np.clip(-n[:, 2] * 1.5, 0, 1) * 0.9)
    zq = None
    if p.mid_rail >= 0.05:
        # the mid rail: a ridge round the body, proud of the face - a closed bar that straddles the glazing like the
        # rail of lod 0 (its back is seen through the panes from the far side, and a lamp inside must not shine out
        # behind it). Its inner wall stands 12.5 mm inside the face: 8.4 mm behind the glowing pane a host fits 4.1 mm
        # inside the face, on the same taper. The panes stop short of it above and below.
        zq = z0 + h * clamp(p.mid_rail, 0.3, 0.85)
        rq = lerp(hb, ht, (zq - z0) / h) * S2
        ri0, ri1 = ((lerp(hb, ht, (z - z0) / h) - 0.0125) * S2 for z in (zq - 0.009, zq + 0.009))
        mid = add(lathe("rail_mid", [(ri0, zq - 0.009), (rq - 0.004, zq - 0.009), (rq + 0.009, zq), (rq - 0.004, zq + 0.009), (ri1, zq + 0.009),
                                     (ri0, zq - 0.009)], segments=X4, rot=(0, 0, 45), smooth_angle=20), iron)
        paint(mid, grime=0.35)
    a2 = (cr - 0.02) / S2
    # the ceiling of the lantern, where lod 0 has the enamel reflector: it faces DOWN - from the street one looks up
    # through the glass at it, and a face turned the other way is a hole into the roof
    throat = mesh("throat", [(-a2, -a2, zt - 0.002), (a2, -a2, zt - 0.002), (a2, a2, zt - 0.002), (-a2, a2, zt - 0.002)], [(3, 2, 1, 0)], iron)
    paint(throat, grime=1.0)
    # glazing: one pane a side, from the tray to the soffit of the top rail and with a gap where the mid rail is - both
    # are closed bars (the hole gate looks in through the glass), and a pane must not run into a face of theirs
    hwg = lambda z: lerp(hb, ht, (z - z0) / h) - 0.0088
    spans = [(z0 + 0.008, z1 - 0.021)] if zq is None else [(z0 + 0.008, zq - 0.01), (zq + 0.01, z1 - 0.021)]
    for k in range(4):
        for j_, (za, zb_) in enumerate(spans):
            g = mesh(f"glass{k}_{j_}", [(-hwg(za), -hwg(za), za), (hwg(za), -hwg(za), za), (hwg(zb_), -hwg(zb_), zb_), (-hwg(zb_), -hwg(zb_), zb_)],
                     [(0, 1, 2, 3)], glass)
            _xf(g, Matrix.Rotation(radians(90 * k), 4, 'Z'))
    # roof, vent and finial: one closed turning (square under an ogee or pyramid roof)
    if p.roof == "ogee":
        main = [(re, 0.0), (re * 0.62, rh * 0.3), (re * 0.3, rh * 0.8), (rtop, rh)]
    elif dome:
        rdm_ = Fh - 0.016
        main = [(rdm_, 0.0)] + [(rdm_ * cos(radians(t)), rh * 0.95 * sin(radians(t))) for t in (30, 60)] + [(rvs, rh)]
    else:
        main = [(re, 0.0), (re * 0.9, rh * 0.075), (rtop, rh)]
    kv = 1.0 if dome else 1.12                  # a square stack of the round one's girth
    prof = [(main[0][0] - 0.03, zt), (main[0][0] + 0.005, zt)] + [(r, zr + z) for r, z in main]
    prof += [(rvs * kv * f, zc + dz) for f, dz in ((1.0, 0.03), (1.25, 0.037), (0.85, 0.05), (0.85, 0.062), (1.55, 0.074), (0.9, 0.098),
                                                  (0.33, 0.112), (0.5, 0.15), (0.0, 0.19))]
    roof = add(lathe("roof", prof, segments=lod_exact(8) if dome else X4, rot=None if dome else (0, 0, 45), smooth_angle=30), sheet)
    rf_ = rng.fork("finial")
    bx, by = rf_.jitter(0.0, 0.11 * age), rf_.jitter(0.0, 0.11 * age)   # knocked finial
    zf = zc + 0.105
    transform_verts(roof, lambda co: Vector((co.x + bx * max(0.0, co.z - zf), co.y + by * max(0.0, co.z - zf), co.z)))
    chips = noise_mask(0.03, seed=p.seed + 5, lo=0.5, hi=0.72)
    paint(roof, special=mask_mul(height_ramp(zc + 0.02, zt - 0.05), 0.35 * (0.3 + 0.7 * age)),
          grime=lambda co, n: np.clip(np.maximum(1 - np.abs(co[:, 2] - (zc + 0.054)) / 0.035, (co[:, 2] - (zc - rh * 0.5)) / (rh * 0.5 + 0.01)), 0, 1)
          * (0.55 + 0.4 * age) * (co[:, 2] < zc + 0.09),
          wear=mask_mul(height_ramp(zt, zt + 0.035), chips, 0.25 + 0.6 * age))
    # cresting: one band a side
    if p.crown:
        Hc = clamp(wt * 0.14, 0.04, 0.06) * 0.7
        ac = Fh - 0.013
        vs = [(sx * ac, sy * ac, z) for z in (zt + 0.004, zt + 0.004 + Hc) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        fs = [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)]
        cb = mesh("crest", vs + vs, fs + [(8 + d, 8 + c, 8 + b_, 8 + a) for a, b_, c, d in fs], iron)
        paint(cb, special=mask_mul(blotch, 0.15 + 0.6 * age))
    # burner and mantle
    zf0 = z0 + 0.004
    zg = z0 + h * 0.35
    mh = min(0.09, h * 0.2)
    zm = zg + 0.017
    add(lathe("burner", [(0.016, zf0), (0.007, zf0 + 0.02), (0.007, zg), (0.02, zg + 0.014), (0.0, zg + 0.023)], segments=X4), brass)
    add(lathe("mantle", [(0.0, zm + 0.002), (0.018, zm + mh * 0.3), (0.0165, zm + mh * 0.7), (0.0, zm + mh)], segments=X4), mantle_m)
    # suspension (hang): four flat straps from the vent cap to a boss under the eye
    z_org = 0.0
    if p.mount == "hang":
        rcap, zcap = rvs * 1.5, zc + 0.073
        vs, fs = [], []
        for k in range(4):
            a = radians(45 + 90 * k)
            ca, sa = cos(a), sin(a)
            b = len(vs)
            for r, z in ((rcap * 0.9, zcap + 0.002), (rcap, ztop - 0.03), (0.01, ztop + 0.036)):
                vs += [(r * ca + sa * w, r * sa - ca * w, z) for w in (-0.011, 0.011)]
            vs += vs[b:b + 6]
            fs += [(b, b + 1, b + 3, b + 2), (b + 2, b + 3, b + 5, b + 4), (b + 7, b + 6, b + 8, b + 9), (b + 9, b + 8, b + 10, b + 11)]
        mesh("crown_straps", vs, fs, forged)
        ze = ztop + 0.056 + 0.0145
        add(lathe("crown_boss", [(0.0, ztop + 0.026), (0.015, ztop + 0.04), (0.017, ze), (0.0, ze + 0.0207)], segments=X4), forged)
        z_org = ze + 0.0165 - 0.0042
    # seeded lean, light and socket as in build()
    rlean = rng.fork("lean")
    ang = radians(rlean.u(0.5, 1.5) * (0.35 + 0.65 * age) * (1.0 if p.mount == "post" else 0.6))
    az = rlean.u(0, 2 * pi)
    G = Matrix.Rotation(ang, 4, Vector((cos(az), sin(az), 0.0))) @ _T(0, 0, -z_org)
    for o in objs:
        _xf(o, G)
    if lit:
        lp_ = G @ Vector((0, 0, zm + mh * 0.5))
        light("lamp", tuple(lp_), intensity=p.intensity, range=11.0, flicker=0.14,
              shadow_near=max(p.width, p.height) * 0.95)
    socket("mount", "attach", (0, 0, 0), role="lantern_mount")


def build(p, rng):
    if lod() == 2:
        return _build_far(p, rng)
    age = float(p.age)
    lit = bool(p.lit)
    lin = hex_rgb(p.paint)
    lifted = tuple(min(1.0, c * 1.8 + 0.05) for c in lin[:3])        # dark paints keep form in shade
    iron = mat("iron_painted", lifted, scale=5.0, rough=3.5)          # fine orange-peel, satin not wet
    sheet = mat("plain", tuple(min(1.0, c * 1.15) for c in lifted), rough=0.45)                           # brush-painted sheet metal
    forged = mat("iron_wrought")
    brass = mat("brass")
    glass = mat("glass_clear", "#2a2927", opacity=0.1)                 # near-clear; a dark body keeps it from hazing white
    putty = mat("plaster_lime", "#2f2c27")                             # old linseed putty, skinned over and sooty
    enamel = mat("ceramic_glazed", "#4a463c")                          # old enamel reflector, yellowed and sooty
    mantle_m = mat("emissive", "#ffa850", glow=5.0 if lit else 0.0)

    wt, wb, h = p.width, p.width * p.taper, p.height
    ht, hb = wt / 2, wb / 2
    z0 = 0.07                      # bottom of the glazed body above the collar
    z1 = z0 + h
    objs = []

    def add(o, m=None):
        if m is not None:
            assign(o, m)
        objs.append(o)
        return o

    # ---- face frames: local (u across, v up the slope, d outwards) for each of the four sides
    dxs = ht - hb
    Ls = hypot(h, dxs)

    def wv(v):
        return hb + dxs * v / Ls

    F = Matrix(((1, 0, 0, 0), (0, -dxs / Ls, -h / Ls, -hb), (0, h / Ls, -dxs / Ls, z0), (0, 0, 0, 1)))
    M = [Matrix.Rotation(radians(90 * k), 4, 'Z') @ F for k in range(4)]
    c = 0.0015                     # cast arris chamfer
    BI, GB = 0.012, 0.0088         # bead depth, back of the glass rebate
    vm = 0.0 if p.mid_rail < 0.05 else clamp(p.mid_rail, 0.3, 0.85) * Ls

    # ---- collar / pendant, bottom tray with drip bead
    R = hb * S2
    zb = z0 - 0.012
    tray = lathe("tray", [(0.0, zb), (R + 0.004, zb), (R + 0.008, zb - 0.006), (R + 0.014, zb - 0.004),
                          (R + 0.016, z0 - 0.008), (R + 0.016, z0 - 0.002), (R + 0.01, z0 + 0.008),
                          (R - 0.012, z0 + 0.008), (R - 0.016, z0 + 0.004), (0.0, z0 + 0.004)],
                 segments=4, rot=(0, 0, 45), smooth_angle=20)
    add(tray, iron)
    rivet = lathe("rivet", [(0.0046, 0.0), (0.0038, 0.0018), (0.0, 0.003)], segments=5)
    rv_m = []                      # every rivet in the asset is stamped from this one base
    if p.mount == "post":
        collar = lathe("collar", [(0.0, 0.0), (0.032, 0.0), (0.036, 0.008), (0.03, 0.02), (0.024, 0.032), (0.03, 0.044),
                                  (0.044, 0.05), (0.05, 0.058), (0.046, 0.066), (0.0, 0.066)], segments=16)
        add(collar, iron)
        for sx in (-1, 1):         # set screws clamping the collar on the post spigot
            add(bolt("setscrew", 0.0055, 0.007, loc=(sx * 0.033, 0, 0.014), rot=(0, sx * 90, 0), kind="pyramid"), forged)
        # frog: four forged cradle arms from the collar out to the tray corners
        r3 = max(R - 0.02, 0.046)
        bar = rounded_rect(0.012, 0.0045, 0.001, 1)
        for k in range(4):
            a = radians(45 + 90 * k)
            ca, sa = cos(a), sin(a)
            rz = [(0.027, 0.031), (lerp(0.03, r3, 0.3), 0.026), (lerp(0.03, r3, 0.68), 0.036), (r3, zb - 0.0025)]
            arm = sweep(f"cradle{k}", resample_path([(r * ca, r * sa, z) for r, z in rz], n=7), bar, up=(0, 0, 1), smooth_path=False)
            add(arm, forged)
            rv_m.append(trs(((r3 - 0.007) * ca, (r3 - 0.007) * sa, zb - 0.0045), (180, 0, 0)))
    else:
        drip = lathe("pendant", [(0.0, zb - 0.078), (0.004, zb - 0.072), (0.0065, zb - 0.064), (0.004, zb - 0.057),
                                 (0.012, zb - 0.046), (0.0175, zb - 0.035), (0.014, zb - 0.025), (0.008, zb - 0.019),
                                 (0.02, zb - 0.012), (0.033, zb - 0.006), (0.035, zb + 0.001)], segments=12)
        add(drip, iron)

    # ---- corner astragals: proud bead + two glazing fins (rebate) running up each arris
    o_, fe = 0.004, 0.026
    sec = [(-o_ + c, -o_), (BI - c, -o_), (BI, -o_ + c), (BI, GB), (fe, GB), (fe, BI), (BI, BI),
           (BI, fe), (GB, fe), (GB, BI), (-o_ + c, BI), (-o_, BI - c), (-o_, -o_ + c)]
    rb = rng.fork("bars")
    nb = 4
    for sx in (-1, 1):
        for sy in (-1, 1):
            rings = []
            for i in range(nb + 1):
                z = lerp(z0 - 0.003, z1 + 0.002, i / nb)
                w_ = hb + dxs * (z - z0) / h
                jx = rb.jitter(0.0, 0.0014 * age) if 0 < i < nb else 0.0     # knocks and slight bows
                jy = rb.jitter(0.0, 0.0014 * age) if 0 < i < nb else 0.0
                rings.append([(sx * (w_ - e) + jx, sy * (w_ - f) + jy, z) for e, f in sec])
            b = _loft(f"bar{sx}{sy}", rings)
            recalc_normals(b)
            smooth(b, 30)
            add(b, iron)

    # ---- rails (L-section with rebate) in the face frame
    def rail(name, sec_vd, MM, inset=BI, nu=3):
        rings = []
        for i in range(nu):
            s = -1 + 2 * i / (nu - 1)
            rings.append([(s * (wv(v) - inset), v, d) for v, d in sec_vd])
        o = _loft(name, rings)
        recalc_normals(o)
        smooth(o, 30)
        return _xf(o, MM)

    v0, v1 = 0.004, 0.022
    bot_sec = [(v0, 0.004), (v1 - c, 0.004), (v1, 0.004 - c), (v1, -GB), (v1 + 0.011, -GB), (v1 + 0.011, -BI), (v0, -BI)]
    top_sec = [(Ls - v, d) for v, d in bot_sec]
    mh_ = 0.0065
    mid_sec = [(vm - mh_, -BI), (vm - mh_, 0.003 - c), (vm - mh_ + c, 0.003), (vm + mh_ - c, 0.003), (vm + mh_, 0.003 - c), (vm + mh_, -BI)]
    for k in range(4):
        add(rail(f"rail_b{k}", bot_sec, M[k]), iron)
        add(rail(f"rail_t{k}", top_sec, M[k]), iron)
        if vm:
            add(rail(f"rail_m{k}", mid_sec, M[k]), iron)

    # ---- glazing: thick seeded panes with their own bow / tilt / ripple, soot and dust, putty lines
    def pane(name, va, vb, r, nu=4, nv=5):
        tg = 0.0035
        tv, tu = r.jitter(0.0, 0.0013), r.jitter(0.0, 0.0013)
        bow = r.u(0.0004, 0.0013) * r.pick((-1, 1))
        ph, rip, fr = r.u(0, 6.28), r.u(0.0002, 0.0005), r.u(28, 45)

        def dfun(s, t):
            return (-0.0052 + bow * sin(pi * t) * (1 - s * s) + tv * (2 * t - 1) + tu * s
                    + rip * sin(fr * lerp(va, vb, t) + ph + 2.0 * s))
        vs = []
        for back in (0.0, -tg):
            for j in range(nv + 1):
                t = j / nv
                v = lerp(va, vb, t)
                W = wv(v) - 0.0123
                for i in range(nu + 1):
                    s = -1 + 2 * i / nu
                    vs.append((s * W, v, dfun(s, t) + back))
        n1 = (nu + 1) * (nv + 1)

        def ix(i, j, b):
            return b * n1 + j * (nu + 1) + i

        fs = []
        for j in range(nv):
            for i in range(nu):
                fs.append((ix(i, j, 0), ix(i + 1, j, 0), ix(i + 1, j + 1, 0), ix(i, j + 1, 0)))
        for i in range(nu):
            fs.append((ix(i, 0, 1), ix(i + 1, 0, 1), ix(i + 1, 0, 0), ix(i, 0, 0)))
            fs.append((ix(i, nv, 0), ix(i + 1, nv, 0), ix(i + 1, nv, 1), ix(i, nv, 1)))
        for j in range(nv):
            fs.append((ix(0, j, 0), ix(0, j + 1, 0), ix(0, j + 1, 1), ix(0, j, 1)))
            fs.append((ix(nu, j, 1), ix(nu, j + 1, 1), ix(nu, j + 1, 0), ix(nu, j, 0)))
        g = mesh_object(name, vs, fs)
        recalc_normals(g)
        smooth(g, 30)
        return g, dfun

    def glass_masks(g, k, soot, dust):
        Mi = np.array(M[k].inverted())
        nz = noise_mask(0.05, seed=p.seed + 31 + k, lo=0.2, hi=0.8)

        def grime(co, n):
            q = co @ Mi[:3, :3].T + Mi[:3, 3]
            u, v = q[:, 0], q[:, 1]
            t = v / Ls
            top = np.clip((t - 0.52) / 0.42, 0, 1) ** 1.4 * soot * (0.45 + 0.55 * nz(co, n))
            dc = np.hypot((hb + dxs * t) - np.abs(u) - 0.012, v - 0.022)
            dst = np.clip(1 - dc / 0.09, 0, 1) ** 1.2 * dust + np.clip(1 - (v - 0.022) / 0.035, 0, 1) * dust * 0.45
            return np.clip(np.maximum(top, dst), 0, 1)
        paint(g, grime=grime)

    def putty_ring(name, va, vb, inset, d_out, r, MM):
        cs = [(-1, va), (1, va), (1, vb), (-1, vb)]
        per = []
        for a in range(4):
            (s0, va_), (s1, vb_) = cs[a], cs[(a + 1) % 4]
            for i in range(3):
                per.append((lerp(s0, s1, i / 3), lerp(va_, vb_, i / 3)))
        outer, inner = [], []
        for s, v in per:
            pw = r.u(0.0035, 0.0058)                                   # hand-run putty, never even
            vi = clamp(v, va + pw, vb - pw)
            Wi = wv(vi) - inset - pw
            outer.append((s * (wv(v) - inset), v, d_out))
            inner.append((clamp(s * (wv(v) - inset), -Wi, Wi), vi, -0.0049))
        o = _loft(name, [outer, inner])
        _orient(o, lambda cc: Vector((0, 0, 1)))
        smooth(o, 30)
        return _xf(o, MM)

    # door leaf transform: hung on the left arris of the front, a little ajar and dropped with age
    rd_ = rng.fork("door")
    A = Vector((-hb - 0.001, 0.0, 0.0095))
    axis = Vector((-dxs, Ls, 0.0)).normalized()
    ajar = radians(rd_.u(0.25, 1.0) * (0.3 + age))
    D = _T(*A) @ Matrix.Rotation(-ajar, 4, axis) @ Matrix.Rotation(radians(-0.25 * age), 4, 'Z') @ _T(*(-A))
    MD = M[0] @ D

    rdm = rng.fork("damage")
    kd = rdm.pick([3, 1, 2]) if p.broken != "none" else -1             # the door pane is never the broken one
    rp_ = rng.fork("panes")
    cols = [-1 + 2 * i / 6 for i in range(7)]
    soot_m = mat("glass_clear", "#17120c", opacity=0.1)
    dust_m = mat("glass_clear", "#625c4e", opacity=0.13)
    vg0, vg1 = v1 + 0.0002, Ls - v1 - 0.0002
    spans = [(vg0, vm - mh_), (vm + mh_, vg1)] if vm else [(vg0, vg1)]
    for k in range(4):
        r = rp_.fork(f"side{k}")
        soot = clamp(r.u(0.35, 0.8) + 0.5 * age, 0.0, 1.0)
        dust = clamp(r.u(0.25, 0.6) + 0.4 * age, 0.0, 1.0)
        MM = MD if k == 0 else M[k]
        for j, (va, vb) in enumerate(spans):
            lower = j == 0
            if k == 0:                                                 # bedded in the door leaf's own rebate
                add(putty_ring(f"putty{k}{j}", max(va, 0.027) if lower else va, vb if (lower and vm) else min(vb, Ls - 0.027),
                               0.0165, 0.004, r, MM), putty)
            else:
                add(putty_ring(f"putty{k}{j}", va, vb, BI, -0.0008, r, MM), putty)
            if k == kd and lower and p.broken == "missing":
                # two shards left in the rebate
                for (s_, vv, sz) in ((-1, va, r.u(0.05, 0.08)), (1, vb, r.u(0.035, 0.06))):
                    W = wv(vv) - 0.0125
                    dv = sz * 1.3 * (1 if vv == va else -1)
                    tri = [(s_ * W, vv), (s_ * (W - sz), vv), (s_ * (wv(vv + dv) - 0.0125), vv + dv)]
                    sh = _loft("shard", [[(x, y, -0.0052) for x, y in tri], [(x, y, -0.0087) for x, y in tri]])
                    me = sh.data
                    sh2 = mesh_object("shard_c", [tuple(v_.co) for v_ in me.vertices],
                                      [tuple(f_.vertices) for f_ in me.polygons] + [(0, 1, 2), (5, 4, 3)])
                    delete([sh])
                    recalc_normals(sh2)
                    add(_xf(sh2, MM), glass)
                continue
            g, dfun = pane(f"pane{k}{j}", va, vb, r, nv=5 if (lower or not vm) else 2)
            _xf(g, MM)
            assign(g, glass)
            glass_masks(g, k, soot, dust)
            objs.append(g)

            # soot and dust lie on the inside face as stacked thin films (the stack fakes the gradient)
            def film(name, bottom, topr, off, m, va=va, vb=vb, dfun=dfun, MM=MM):
                rows = [[(s_ * (wv(v_) - 0.0125), v_, dfun(s_, (v_ - va) / (vb - va)) - 0.0035 - off) for s_, v_ in row]
                        for row in (bottom, topr)]
                o = _loft(name, rows, closed=False)
                _orient(o, lambda cc: Vector((0, 0, 1)))
                return add(_xf(o, MM), m)

            base = 0.8 - 0.17 * soot
            p1, p2, lop = r.u(0, 6.28), r.u(0, 6.28), r.jitter(0.0, 0.03)
            for i in range(6):
                e = [Ls * (base + 0.033 * i - 0.07 * s_ * s_ + lop * s_ + 0.012 * sin(2.3 * s_ + p1 + 0.5 * i)
                           + 0.006 * sin(5.1 * s_ + p2 + 0.9 * i)) for s_ in cols]
                if min(e) > vb - 0.006:
                    continue
                film(f"soot{k}{j}{i}", [(s_, clamp(ev, va, vb)) for s_, ev in zip(cols, e)], [(s_, vb) for s_ in cols],
                     0.0003 * (i + 1), soot_m)
            if lower:
                dl, dr = r.u(0.6, 1.3), r.u(0.6, 1.3)                       # drift is never the same in both corners
                for i in range(3):
                    e = [va + Ls * ((0.02, 0.011, 0.004)[i] + (0.15, 0.1, 0.055)[i] * dust * (dl if s_ < 0 else dr) * abs(s_) ** 2.5)
                         for s_ in cols]
                    film(f"dust{k}{j}{i}", [(s_, va) for s_ in cols], [(s_, min(ev, vb)) for s_, ev in zip(cols, e)],
                         0.0003 * (i + 1), dust_m)
            if k == kd and lower and p.broken == "cracked":
                # star crack from a knock low on the pane: bright scattering lines on the glass face
                crack_m = enamel
                u0, vv0 = r.u(-0.3, 0.3) * wv(va), lerp(va, vb, r.u(0.22, 0.4))
                nray = 5
                a0 = r.u(0, 360)
                for q in range(nray):
                    a = radians(a0 + q * 360 / nray + r.jitter(0, 18))
                    pts = [(u0, vv0)]
                    reach = r.u(0.5, 1.0) * (vb - va)
                    for stp in range(1, 4):
                        a += radians(r.jitter(0, 16))
                        uu = pts[-1][0] + cos(a) * reach / 3
                        vv = pts[-1][1] + sin(a) * reach / 3
                        vv = clamp(vv, va + 0.002, vb - 0.002)
                        uu = clamp(uu, -(wv(vv) - 0.014), wv(vv) - 0.014)
                        pts.append((uu, vv))
                    left, right = [], []
                    for i_, (uu, vv) in enumerate(pts):
                        pa, pb = pts[max(i_ - 1, 0)], pts[min(i_ + 1, len(pts) - 1)]
                        tx, ty = pb[0] - pa[0], pb[1] - pa[1]
                        ln = hypot(tx, ty) or 1.0
                        wd = 0.0011 * (1.0 - 0.75 * i_ / (len(pts) - 1))
                        left.append((uu - ty / ln * wd, vv + tx / ln * wd, -0.0034))
                        right.append((uu + ty / ln * wd, vv - tx / ln * wd, -0.0034))
                    ck = _loft(f"crack{q}", [left, right], closed=False)
                    _orient(ck, lambda cc: Vector((0, 0, 1)))
                    add(_xf(ck, MM), crack_m)

    # ---- door leaf: flat-bar frame overlapping the astragals, hinges, latch and brass knob
    d0, d1 = 0.003, 0.0095
    va_o, vb_o, va_i, vb_i = 0.008, Ls - 0.008, 0.026, Ls - 0.026
    Oc = [(-wv(va_o), va_o), (wv(va_o), va_o), (wv(vb_o), vb_o), (-wv(vb_o), vb_o)]
    Ic = [(-(wv(va_i) - 0.0165), va_i), (wv(va_i) - 0.0165, va_i), (wv(vb_i) - 0.0165, vb_i), (-(wv(vb_i) - 0.0165), vb_i)]
    rings = []
    for (ox, oy), (ix_, iy) in zip(Oc, Ic):
        def lp(t, ox=ox, oy=oy, ix_=ix_, iy=iy):
            return (lerp(ox, ix_, t), lerp(oy, iy, t))
        rings.append([(*lp(0), d0), (*lp(0), d1 - c), (*lp(0.09), d1), (*lp(0.91), d1), (*lp(1), d1 - c), (*lp(1), d0)])
    leaf = _loft("door_leaf", rings, loop=True)
    recalc_normals(leaf)
    smooth(leaf, 30)
    add(_xf(leaf, MD), iron)
    if vm:
        lm = 0.0075
        add(rail("door_mid", [(vm - lm, d0), (vm - lm, d1 - c), (vm - lm + c, d1), (vm + lm - c, d1), (vm + lm, d1 - c), (vm + lm, d0)],
                 MD, inset=0.0165), iron)
    edge = Vector((-dxs, Ls, 0.0)).normalized()
    for t in (0.2, 0.8):
        vh = Ls * t
        uh = -(wv(vh) + 0.002)
        kn = cylinder("hinge_knuckle", 0.0042, 0.034, verts=8)
        add(_xf(kn, MD @ _T(uh, vh, 0.0095) @ _rot_to(edge)), forged)
        for e_ in (-1, 1):                                              # pin heads
            rv_m.append(MD @ _T(uh + e_ * edge.x * 0.017, vh + e_ * 0.017, 0.0095) @ _rot_to(edge * e_) @ Matrix.Scale(0.8, 4))
        strap = extrude_profile("hinge_strap", [(0.0, -0.0075), (0.016, -0.0075), (0.023, 0.0), (0.016, 0.0075), (0.0, 0.0075)], 0.0022)
        add(_xf(strap, MD @ _T(uh + 0.001, vh, d1 - 0.0004)), forged)
        rv_m.append(MD @ _T(uh + 0.011, vh, d1 + 0.0016) @ Matrix.Scale(0.75, 4))
    vl = Ls * (0.42 if (not vm or abs(vm - 0.42 * Ls) > 0.04) else 0.27)
    uk = wv(vl) - 0.0085
    rose = lathe("knob", [(0.0, 0.0), (0.011, 0.0), (0.0115, 0.0015), (0.009, 0.003), (0.0045, 0.0035), (0.0045, 0.009),
                          (0.0085, 0.013), (0.0095, 0.017), (0.007, 0.0205), (0.0, 0.022)], segments=10)
    add(_xf(rose, MD @ _T(uk, vl, d1)), brass)
    tongue = box("latch", (0.027, 0.0075, 0.003), loc=(0.0085, 0, 0.0045), bevel=0.0007, segments=1)
    add(_xf(tongue, MD @ _T(uk, vl, d1) @ Matrix.Rotation(radians(-14), 4, 'Z')), forged)
    keeper = box("keeper", (0.006, 0.015, 0.011), loc=(wv(vl) + 0.0015, vl - 0.003, 0.0085), bevel=0.001, segments=1)
    add(_xf(keeper, M[0]), forged)

    # ---- moulded cornice (top rail) carrying the roof and the cresting
    cr = ht * S2
    Rr = cr + 0.034
    Fh = Rr / S2
    zt = z1 + 0.02
    def sqring(a, z, n):
        out = []
        for k in range(4):
            ck, sk = cos(radians(90 * k)), sin(radians(90 * k))
            for i in range(n):
                x, y = (-1 + 2 * i / n) * a, -a
                out.append((x * ck - y * sk, x * sk + y * ck, z))
        return out

    top = _loft("toprail", [sqring(r / S2, z, 6) for r, z in
                            [(cr - 0.014, z1 - 0.008), (cr + 0.006, z1 - 0.008), (cr + 0.019, z1 + 0.002),
                             (cr + 0.03, z1 + 0.008), (cr + 0.034, z1 + 0.013), (Rr, zt), (cr - 0.02, zt)]], loop=True)
    recalc_normals(top)
    smooth(top, 20)
    add(top, iron)

    # ---- roof: riveted sheet, one panel per side, hip rolls over the seams
    dome = p.roof == "dome"
    re = cr + 0.008
    rh = wt * (0.42 if p.roof != "pyramid" else 0.36)
    rtop = re * 0.26
    rvs = rtop * 0.84                                                   # round stack radius
    zr = zt + 0.008
    if p.roof == "ogee":
        main, nvr = [(re, 0.0), (re * 0.86, rh * 0.12), (re * 0.62, rh * 0.3), (re * 0.42, rh * 0.56), (re * 0.3, rh * 0.8), (rtop, rh)], 8
    elif dome:
        rdm_ = Fh - 0.016
        main, nvr = [(rdm_, 0.0)] + [(rdm_ * cos(radians(t)), rh * 0.95 * sin(radians(t))) for t in (14, 30, 46, 60, 72)] + [(rvs, rh)], 8
    else:
        main, nvr = [(re, 0.0), (re * 0.9, rh * 0.075), (rtop, rh)], 6
    rs = resample_path([(r, 0.0, z) for r, z in main], n=nvr + 1, smooth_path=True)
    prof = [(main[0][0] + 0.004, zt), (main[0][0] + 0.006, zt + 0.004)] + [(float(q[0]), zr + float(q[2])) for q in rs]
    zc = zr + rh

    def rp(k, s, r, z):
        if dome:
            a = radians(s * 45.0)
            x, y = r * sin(a), -r * cos(a)
        else:
            x, y = s * r / S2, -r / S2
        ck, sk = cos(radians(90 * k)), sin(radians(90 * k))
        return (x * ck - y * sk, x * sk + y * ck, z)

    roofs = []
    if dome:
        nur = 6
        o = _loft("roof", [[rp(k, -1 + 2 * i / nur, r, z) for k in range(4) for i in range(nur)] for r, z in prof])
        _orient(o, lambda cc: Vector((cc.x, cc.y, 0.02)))
        smooth(o, 50)
        roofs.append(o)
    else:
        nur = 10
        for k in range(4):
            o = _loft(f"roof{k}", [[rp(k, -1 + 2 * i / nur, r, z) for i in range(nur + 1)] for r, z in prof], closed=False)
            _orient(o, lambda cc: Vector((cc.x, cc.y, 0.02)))
            smooth(o, 50)
            roofs.append(o)
    for o in roofs:
        add(o, sheet)
    hips = []
    for k in range(4):
        hp = sweep(f"hip{k}", [rp(k, 1, r + 0.001, z + 0.0015) for r, z in prof[1:]], 0.0055, sides=5, smooth_path=False)
        hips.append(add(hp, iron))
        if not p.crown:
            add(sphere(f"hip_knob{k}", 0.0095, loc=rp(k, 1, prof[1][0] + 0.002, prof[1][1] + 0.004), segments=8, rings=5), iron)
    # rivet row along the eave
    (ra, za), (rb2, zb2) = prof[2], prof[3]
    re_, ze_ = lerp(ra, rb2, 0.45), lerp(za, zb2, 0.45)
    nl = hypot(zb2 - za, rb2 - ra)
    n_r, n_z = (zb2 - za) / nl, -(rb2 - ra) / nl
    nrv = max(3, int(round((2 * re_ / S2 if not dome else re_ * pi / 2) / 0.062)))
    for k in range(4):
        for i in range(nrv):
            s = -1 + (i + 0.5) * 2 / nrv
            px, py, pz = rp(k, s, re_, ze_)
            if dome:
                rr_ = hypot(px, py)
                nrm = (px / rr_ * n_r, py / rr_ * n_r, n_z)
            else:
                ck, sk = cos(radians(90 * k)), sin(radians(90 * k))
                nrm = (n_r * sk, -n_r * ck, n_z)
            rv_m.append(_T(px, py, pz) @ _rot_to(nrm))

    # ---- formed square-to-round collar, stepped vent chimney with louvre fin, onion finial
    def ring16(r, z, square):
        out = []
        for i in range(16):
            a = radians(45 + 22.5 * i)
            rr_ = (r / S2) / max(abs(cos(a)), abs(sin(a))) if square else r
            out.append((rr_ * cos(a), rr_ * sin(a), z))
        return out

    zs_ = [zr + float(q[2]) for q in rs]
    rs_ = [float(q[0]) for q in rs]
    r_sk = float(np.interp(zc - 0.016, zs_, rs_))                      # roof radius where the skirt lands
    if dome:
        rk = [ring16(r_sk + 0.008, zc - 0.016, False), ring16(rvs + 0.005, zc + 0.002, False), ring16(rvs, zc + 0.03, False)]
    else:
        rk = [ring16(r_sk + 0.009, zc - 0.016, True), ring16(rtop + 0.005, zc + 0.002, True), ring16(rvs, zc + 0.03, False)]
    skirt = _loft("vent_collar", rk)
    recalc_normals(skirt)
    smooth(skirt, 35)
    add(skirt, sheet)
    # rust runs: thin tapered films of rust creeping down the sheet from under the collar
    rr_ = rng.fork("rust")
    runs = []
    for k in range(4):
        for q in range(1 + int(rr_.u(0.2, 2.4) * (0.4 + age))):
            x = rr_.u(-0.75, 0.75) * r_sk / S2
            z_s = zc - 0.019
            drop = rr_.u(0.22, 0.75) * rh
            wmax = rr_.u(0.0011, 0.0026)
            left, right = [], []
            for i_, wf in enumerate((0.45, 1.0, 0.9, 0.65, 0.4, 0.12)):
                z_ = max(z_s - drop * i_ / 5, zr + 0.004)
                r_ = float(np.interp(z_, zs_, rs_))
                x += rr_.jitter(0.0, 0.0012)
                un = (0.785 * r_) if dome else (r_ / S2)
                s_c, s_w = clamp(x / un, -0.9, 0.9), wmax * wf / un
                left.append(rp(k, s_c - s_w, r_ + 0.0011, z_ + 0.0011))
                right.append(rp(k, s_c + s_w, r_ + 0.0011, z_ + 0.0011))
            o = _loft(f"rustrun{k}{q}", [left, right], closed=False)
            _orient(o, lambda cc: Vector((cc.x, cc.y, 0.02)))
            smooth(o, 50)
            runs.append(add(o, iron))
    rc = rvs
    vent = lathe("vent", [(rc, zc + 0.029), (rc * 1.25, zc + 0.034), (rc * 1.25, zc + 0.04), (rc * 0.85, zc + 0.046),
                          (rc * 0.85, zc + 0.052), (rc * 0.97, zc + 0.054), (rc * 0.85, zc + 0.056),
                          (rc * 0.85, zc + 0.062), (rc * 1.5, zc + 0.07), (rc * 1.55, zc + 0.078), (rc * 0.9, zc + 0.098),
                          (rc * 0.35, zc + 0.108), (rc * 0.3, zc + 0.125), (rc * 0.55, zc + 0.14), (rc * 0.5, zc + 0.158),
                          (rc * 0.18, zc + 0.172), (0.0, zc + 0.19)], segments=14)
    rf_ = rng.fork("finial")
    bx, by = rf_.jitter(0.0, 0.11 * age), rf_.jitter(0.0, 0.11 * age)   # knocked finial
    zf = zc + 0.105
    transform_verts(vent, lambda co: Vector((co.x + bx * max(0.0, co.z - zf), co.y + by * max(0.0, co.z - zf), co.z)))
    add(vent, sheet)
    ztop = zc + 0.19

    # ---- cresting: continuous pierced strip per eave (rail + linked fleurs) and turned corner finials
    crest = []
    if p.crown:
        rc_ = rng.fork("crest")
        Lc = 2 * (Fh - 0.013)
        nun = max(2, int(round(Lc / 0.062)))
        P = Lc / nun
        Hc = clamp(wt * 0.14, 0.04, 0.06)
        yb, yw = 0.006, 0.16
        unit = [(0.20, 0.50), (0.30, 0.62), (0.24, 0.76), (0.10, 0.66), (0.13, 0.82), (0.0, 1.0), (-0.13, 0.82), (-0.10, 0.66),
                (-0.24, 0.76), (-0.30, 0.62), (-0.20, 0.50)]
        snapped = [(0.20, 0.50), (0.30, 0.62), (0.24, 0.76), (0.10, 0.66), (-0.03, 0.75), (-0.10, 0.66),
                   (-0.24, 0.76), (-0.30, 0.62), (-0.20, 0.50)]
        hole = [(-0.2, 0.03), (0.2, 0.03), (0.14, 0.24), (0.0, 0.42), (-0.14, 0.24)]
        for k in range(4):
            outer = [(-Lc / 2, 0.0), (Lc / 2, 0.0), (Lc / 2, yb + yw * Hc)]
            holes = []
            for j in range(nun - 1, -1, -1):
                cx = -Lc / 2 + (j + 0.5) * P
                un = snapped if rc_.chance(0.3 * max(0.0, age - 0.5)) else unit
                hs = rc_.vary(1.0, 0.05)
                outer += [(cx + x * P, yb + y * Hc * hs) for x, y in un]
                outer.append((cx - 0.5 * P, yb + yw * Hc))
                holes.append([(cx + x * P, yb + y * Hc) for x, y in hole])
            strip = shape_extrude(f"crest{k}", outer, holes, depth=0.003)
            xs = [-Lc / 2 + (j + 0.5) * P for j in range(nun)]
            lj = [rc_.jitter(0.0, 0.03 + 0.22 * age) for _ in range(nun)]      # each fleur bent in or out
            tj = [rc_.jitter(0.0, 0.02 + 0.1 * age) for _ in range(nun)]       # ... and a little sideways

            def bend(co, xs=xs, lj=lj, tj=tj):
                t = max(0.0, co.y - yb - yw * Hc)
                return Vector((co.x + float(np.interp(co.x, xs, tj)) * t, co.y, co.z + float(np.interp(co.x, xs, lj)) * t))
            transform_verts(strip, bend)
            C = Matrix.Rotation(radians(90 * k), 4, 'Z') @ _T(0, -(Fh - 0.0055) + 0.0015, zt - 0.0005) @ Matrix.Rotation(radians(90), 4, 'X')
            crest.append(add(_xf(strip, C), iron))
        Ha = clamp(wt * 0.22, 0.07, 0.09)
        fin = lathe("acro", [(0.0085, 0.0), (0.0095, 0.004), (0.005, 0.008), (0.0042, 0.016), (0.0095, 0.028), (0.008, 0.04),
                             (0.0032, 0.052), (0.0026, 0.062), (0.0, 0.08)], segments=5)
        am = []
        for k in range(4):
            a = radians(45 + 90 * k)
            rr_ = (Fh - 0.0085) * S2
            am.append(trs((rr_ * cos(a), rr_ * sin(a), zt - 0.0005), (rc_.jitter(0, 1 + 7 * age), rc_.jitter(0, 1 + 7 * age), rc_.u(0, 360)), Ha / 0.08))
        crest.append(add(instances_to_mesh("acroteria", fin, am, keep_base=False), iron))

    # ---- reflector closing the roof underside: enamel pyramid up to a sooty flue throat
    def sq8(a, z):
        return [(-a, -a, z), (0, -a, z), (a, -a, z), (a, 0, z), (a, a, z), (0, a, z), (-a, a, z), (-a, 0, z)]

    a0, a2 = ht - 0.013, min(0.035, ht * 0.3)
    refl = _loft("reflector", [sq8(a0, z1 - 0.005), sq8(lerp(a0, a2, 0.55), z1 + 0.016), sq8(a2, z1 + 0.03)])
    _orient(refl, lambda cc: Vector((0, 0, -1)))
    smooth(refl, 20)
    assign(refl, enamel)
    soot_r = 0.35 + 0.55 * age
    paint(refl, grime=lambda co, n: np.clip(1.15 - (np.hypot(co[:, 0], co[:, 1]) - a2) / (a0 * 0.75), 0, 1) ** 1.6 * soot_r)
    objs.append(refl)
    throat = mesh_object("throat", [(-a2, -a2, z1 + 0.03), (a2, -a2, z1 + 0.03), (a2, a2, z1 + 0.03), (-a2, a2, z1 + 0.03)], [(0, 3, 2, 1)])
    _orient(throat, lambda cc: Vector((0, 0, -1)))
    add(throat, iron)
    paint(throat, grime=1.0)

    # ---- burner: union, supply pipe, stopcock with by-pass lever and pull rods, gallery, mantle
    zf0 = z0 + 0.004
    zg = z0 + h * 0.35
    add(lathe("union", [(0.016, zf0), (0.016, zf0 + 0.006), (0.0095, zf0 + 0.012), (0.0065, zf0 + 0.02)], segments=10), brass)
    add(cylinder("pipe", 0.006, zg - zf0, loc=(0, 0, (zg + zf0) / 2), verts=8), brass)
    zs = z0 + h * 0.17
    add(cylinder("cock", 0.0085, 0.022, loc=(0, 0, zs), rot=(0, 90, 0), verts=8, bevel=0.001, segments=1), brass)
    rl = rng.fork("lever")
    ll = min(0.11, (hb + dxs * 0.17) * 1.15)
    la = rl.jitter(0.0, 14.0)
    add(box("lever", (0.0045, ll, 0.0028), loc=(0.0125, 0, zs), rot=(la, 0, 0), bevel=0.0006, segments=1), brass)
    rod_l = min(0.045, h * 0.1)
    for sy in (-1, 1):
        ye, ze = sy * ll * 0.47 * cos(radians(la)), zs + sy * ll * 0.47 * sin(radians(la))
        add(cylinder("pull_rod", 0.0011, rod_l, loc=(0.0125, ye, ze - rod_l / 2), verts=4, cap=False), forged)
        add(sphere("pull_drop", 0.0042, loc=(0.0125, ye, ze - rod_l), segments=6, rings=4, scale=(1, 1, 1.5)), brass)
    add(lathe("gallery", [(0.006, zg), (0.016, zg + 0.006), (0.02, zg + 0.014), (0.02, zg + 0.022), (0.017, zg + 0.023),
                          (0.013, zg + 0.017), (0.0, zg + 0.017)], segments=12), brass)
    zm = zg + 0.017
    mh = min(0.09, h * 0.2)
    # (open underneath: a glowing bottom cap lay 0-2 mm over the brass gallery floor - CLEARANCE RULE, >= 4 mm or nothing)
    mantle = lathe("mantle", [(0.011, zm + 0.002), (0.0155, zm + mh * 0.3), (0.0145, zm + mh * 0.7),
                              (0.008, zm + mh * 0.95), (0.0, zm + mh)], segments=12)
    add(mantle, mantle_m)
    add(sweep("mantle_fork", resample_path([(0.0195, 0, zg + 0.02), (0.022, 0, zm + mh * 0.6), (0.016, 0, zm + mh + 0.009),
                                            (0.0, 0, zm + mh + 0.012)], n=7), 0.0013, sides=4, cap=False, smooth_path=False), forged)

    # ---- suspension (hang): four forged straps from the vent cap to a boss and a 40 mm eye
    z_org = 0.0
    if p.mount == "hang":
        rcap, zcap = rc * 1.5, zc + 0.073
        strap_p = rounded_rect(0.011, 0.0038, 0.001, 1)
        for k in range(4):
            a = radians(45 + 90 * k)
            ca, sa = cos(a), sin(a)
            rz = [(rcap * 0.88, zcap + 0.002), (rcap * 1.0, zcap + 0.03), (rcap * 0.98, ztop - 0.03), (rcap * 0.6, ztop + 0.018), (0.01, ztop + 0.036)]
            add(sweep(f"crown_strap{k}", resample_path([(r * ca, r * sa, z) for r, z in rz], n=10), strap_p, up=(ca, sa, 0),
                      smooth_path=False), forged)
            rv_m.append(_T(rcap * 0.93 * ca, rcap * 0.93 * sa, zcap + 0.012) @ _rot_to((ca, sa, 0.25)))
        add(lathe("crown_boss", [(0.0, ztop + 0.026), (0.013, ztop + 0.029), (0.0155, ztop + 0.037), (0.011, ztop + 0.045),
                                 (0.006, ztop + 0.049), (0.006, ztop + 0.056), (0.0, ztop + 0.056)], segments=10), forged)
        ze = ztop + 0.056 + 0.0145
        add(torus("eye", 0.0165, 0.0042, loc=(0, 0, ze), rot=(90, 0, 0), major_seg=16, minor_seg=6), forged)
        z_org = ze + 0.0165 - 0.0042

    add(instances_to_mesh("rivets", rivet, rv_m, keep_base=False), iron)

    # ---- art-directed ageing
    sd = p.seed
    ph1, ph2 = rng.fork("streak").u(0, 6.28), rng.fork("streak2").u(0, 6.28)

    def streaks(lo=0.6, hi=0.9):
        def fn(co, n):
            x, y = co[:, 0], co[:, 1]
            if dome:
                q = np.arctan2(y, x) * 0.16
            else:
                q = np.where(np.abs(x) > np.abs(y), y + 3.1 * np.sign(x), x + 7.7 * np.sign(y))
            v = 0.5 + 0.27 * np.sin(q * 190 + ph1) + 0.23 * np.sin(q * 83 + ph2)
            return np.clip((v - lo) / (hi - lo), 0, 1)
        return fn

    def radial(co, n):
        return (n[:, 0] * co[:, 0] + n[:, 1] * co[:, 1]) / (np.hypot(co[:, 0], co[:, 1]) + 1e-6)

    def outside(co, n):                      # weather only reaches the faces turned away from the flame
        return np.clip(radial(co, n) * 4 - 0.2, 0, 1)

    lamp_c = np.array([0.0, 0.0, zm + mh * 0.5])

    def inside(co, n):                       # lamp-black on everything that looks at the burner
        d = lamp_c[None, :] - co
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        return np.clip((n * d).sum(axis=1) * 3.0, 0, 1)

    chips = noise_mask(0.03, seed=sd + 5, lo=0.5, hi=0.72)
    blotch = noise_mask(0.07, seed=sd + 9, lo=0.45, hi=0.8)
    for o in roofs + hips:
        paint(o,
              special=mask_mul(streaks(), height_ramp(zc + 0.02, zt - 0.05), 0.3 + 0.7 * age),      # rust runs from the vent
              grime=mask_mul(height_ramp(zc + 0.01, zc - rh * 0.5), 0.55 + 0.4 * age),               # soot fall-out round the flue
              wear=mask_mul(height_ramp(zt, zt + 0.035), chips, 0.25 + 0.6 * age))                   # chipped eave
    paint(runs, special=0.8, grime=0.4, wear=0.0)
    paint([skirt, vent], grime=lambda co, n: np.clip(1 - np.abs(co[:, 2] - (zc + 0.054)) / 0.035, 0, 1) * (0.7 + 0.3 * age),
          special=mask_mul(blotch, 0.2 + 0.6 * age))
    paint(top, special=mask_mul(streaks(0.6, 0.95), blotch, outside, 0.2 + 0.5 * age),
          wear=mask_mul(chips, outside, 0.2 + 0.5 * age),
          grime=lambda co, n: np.clip(np.maximum(-n[:, 2] * 1.5, inside(co, n)), 0, 1) * 0.9)     # sooty soffit
    paint(tray, grime=0.5, special=mask_mul(blotch, 0.2 + 0.7 * age))
    for o in objs:
        if o.name.startswith(("bar", "rail", "door")):
            paint(o, special=mask_mul(height_ramp(z0, z0 + 0.1 + 0.1 * age), blotch, outside, 0.3 + 0.7 * age),   # rust creeping up from the tray
                  wear=mask_mul(chips, outside, 0.15 + 0.55 * age), grime=mask_mul(inside, 0.9))
        elif o.name.startswith("putty"):
            paint(o, grime=mask_mul(inside, 0.8))
    for o in crest:
        paint(o, special=mask_mul(blotch, 0.15 + 0.6 * age), wear=mask_mul(chips, 0.2 + 0.6 * age))

    # ---- seeded lean (post settles 0.5-1.5 deg out of plumb; a hung lantern cants on its eye)
    rlean = rng.fork("lean")
    ang = radians(rlean.u(0.5, 1.5) * (0.35 + 0.65 * age) * (1.0 if p.mount == "post" else 0.6))
    az = rlean.u(0, 2 * pi)
    G = Matrix.Rotation(ang, 4, Vector((cos(az), sin(az), 0.0))) @ _T(0, 0, -z_org)
    for o in objs:
        _xf(o, G)
    if lit:
        lp_ = G @ Vector((0, 0, zm + mh * 0.5))
        light("lamp", tuple(lp_), intensity=p.intensity, range=11.0, flicker=0.14,
              shadow_near=max(p.width, p.height) * 0.95)
    socket("mount", "attach", (0, 0, 0), role="lantern_mount")
