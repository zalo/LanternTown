"""Cast-iron gas street lamp post, built the way the foundries made them: a bolted-down octagonal pedestal
with moulded panels, a hinged meter door (lock, maker's plate) and a cornice, a gadrooned bulb and an acanthus
collar carrying a fluted tapering shaft with a turned mid band, the lamplighter's ladder bar, a capital with a
brass gas tap and four S-scroll stays cradling a four-sided glazed lantern (clear panes in beaded frames,
enamel reflector, burner and mantle, hipped roof with vent cap and finial).  2-3 lanterns ride on swan-neck
arms with their own cradles.  Condition (lean, chips, rust runs, soot, cracked pane, dented roof, dead
mantle) is driven by params and the seed.  Optional dressed-stone plinth.
"""
from tk import *
import bpy
import bmesh
import numpy as np
from math import cos, sin, radians, degrees, pi, atan, atan2, hypot, sqrt, exp
from mathutils import Vector, Matrix

META = {"name": "Gas lamp post", "class": "street_furniture", "mount": "ground", "tri_budget": 30000,
        "focus": 0.86, "lod": "far"}
PARAMS = {
    "height": {"default": 3.6, "min": 2.6, "max": 5.5, "desc": "height to the lantern base (m)"},
    "arms": {"default": 1, "choices": [1, 2, 3], "desc": "1 = lantern on the post top; 2 / 3 = lanterns on swan-neck arms"},
    "spread": {"default": 0.55, "min": 0.35, "max": 0.9, "desc": "arm reach from the shaft axis (m)"},
    "flutes": {"default": 12, "min": 0, "max": 20, "desc": "number of flutes on the shaft (0 = plain)"},
    "plinth": {"default": 0.0, "min": 0.0, "max": 0.8, "desc": "height of a dressed stone plinth under the post (0 = none)"},
    "ladder_bar": {"default": True, "desc": "lamplighter's ladder rest below the lantern"},
    "paint": {"default": "#2a3a33", "desc": "paint colour of the ironwork"},
    "lantern": {"default": "ogee", "choices": ["ogee", "pyramid", "dome"], "desc": "lantern roof style"},
    "lantern_width": {"default": 0.4, "min": 0.28, "max": 0.6, "desc": "lantern width (m)"},
    "intensity": {"default": 60.0, "min": 10, "max": 200, "desc": "light intensity per lantern (cd)"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age of the paint: chips, rust runs, dirt splash, soot (0 = fresh coat, 1 = long neglected)"},
    "lean": {"default": 0.3, "min": 0.0, "max": 1.5, "desc": "lean of the post in degrees (direction from the seed)"},
    "pane": {"default": "auto", "choices": ["auto", "sound", "sooted", "cracked", "missing"],
             "desc": "state of one lantern pane (auto = from wear and seed)"},
    "dent": {"default": 0.0, "min": 0.0, "max": 1.0, "desc": "dent knocked into the lantern roof by a ladder (0 = none)"},
    "unlit": {"default": 0, "min": 0, "max": 2, "desc": "number of arm lanterns whose mantle is out (never all of them)"},
    "gilt": {"default": False, "desc": "gilded bands, collar and finials (smart civic posts)"},
    "maker": {"default": "HOLLIS & SON", "desc": "foundry name cast on the door plate ('/' = line break, empty = no plate)"},
}
VARIANTS = {
    "twin": {"arms": 2, "height": 3.9, "spread": 0.6, "lantern": "dome", "lantern_width": 0.34, "paint": "#15171a", "gilt": True,
             "intensity": 45.0, "wear": 0.3, "lean": 0.15, "seed": 5},
    "triple": {"arms": 3, "height": 4.4, "spread": 0.7, "lantern_width": 0.36, "paint": "#1f3d2c", "flutes": 16, "gilt": True,
               "intensity": 42.0, "wear": 0.45, "lean": 0.2, "unlit": 1, "seed": 9},
    "plinth": {"plinth": 0.55, "height": 3.2, "lantern": "pyramid", "paint": "#1d3250", "wear": 0.6, "lean": 0.0, "seed": 12},
    "short_plain": {"height": 2.8, "flutes": 0, "ladder_bar": False, "lantern_width": 0.34, "paint": "#4d2119", "wear": 0.75,
                    "lean": 0.8, "pane": "sooted", "seed": 21},
    "worn_green": {"height": 3.4, "paint": "#5f7d62", "flutes": 10, "lantern": "pyramid", "wear": 1.0, "lean": 1.5,
                   "pane": "cracked", "dent": 0.9, "seed": 33},
}


def _ramp(z, a, b):
    """1 at a -> 0 at b (clipped, linear)."""
    return np.clip((z - b) / (a - b), 0.0, 1.0)


def build(p, rng):
    zp, H, n = float(p.plinth), float(p.height), int(p.arms)
    W = float(p.lantern_width)
    wr = float(p.wear)
    wk = 0.3 + 0.9 * wr                           # gain of the art-directed weathering
    rc = rng.fork("condition")
    sd = int(p.seed)
    s2 = sqrt(2.0)
    # lod 2 (META "lod": "far", docs/LOD.md) is modelled here: the post's outline, the lantern cage, its roof and
    # the glowing mantle; relief, fixings and scrollwork that are under a pixel from 20 m are left out
    far = lod() == 2
    tiny = []                                     # relief / fixings: not in the far level

    iron = mat("iron_painted", p.paint, rough=1.8)
    brass = mat("brass", rough=0.9)
    gilt = mat("brass", rough=1.0) if p.gilt else iron
    glass = mat("glass_clear", "#0e0e0e", opacity=0.16)
    body, head, bits, gilded, stones, lamps = [], [], [], [], [], []

    # the whole post leans a little about its foot (settled paving); sockets and lights follow
    az = rc.u(0.0, 2 * pi)
    M = Matrix.Translation((0, 0, zp)) @ Matrix.Rotation(radians(p.lean), 4, Vector((cos(az), sin(az), 0.0))) @ Matrix.Translation((0, 0, -zp))

    def Lp(pt):
        v = M @ Vector(pt)
        return (v.x, v.y, v.z)

    # ---- stone plinth (two dressed courses with a chamfered cap)
    if zp > 0.01:
        st = mat("stone_smooth", "#b9ad98")
        r = rng.fork("plinth")
        if far:
            # one square turning. The arrises of base and cap stay as chamfers, and a ring of vertices sits just off
            # every inside corner: the baked corner shade is per vertex, and without the rings it spreads over the
            # whole face of the die and the top of the cap (three plain boxes come out an eighth darker than lod 0)
            rings = ((0.41, 0.0), (0.41, zp * 0.3 - 0.025), (0.385, zp * 0.3), (0.33, zp * 0.3), (0.33, zp * 0.3 + 0.06),
                     (0.33, zp * 0.82 - 0.06), (0.33, zp * 0.82), (0.38, zp * 0.82), (0.38, zp - 0.035), (0.345, zp), (0.27, zp))
            vs = [(sx * q, sy * q, z) for q, z in rings for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            fs = [(4 * k + i, 4 * k + (i + 1) % 4, 4 * k + 4 + (i + 1) % 4, 4 * k + 4 + i) for k in range(len(rings) - 1) for i in range(4)]
            b0 = mesh_object("plinth_base", vs, fs + [tuple(range(len(vs) - 4, len(vs)))])
            uv_box(b0)
            blocks = (b0,)
        else:
            b0 = stone_block("plinth_base", (0.82, 0.82, zp * 0.3), loc=(0, 0, zp * 0.15), rng=r, chamfer=0.015)
            b1 = stone_block("plinth_die", (0.66, 0.66, zp * 0.52), loc=(0, 0, zp * 0.3 + zp * 0.26), rng=r, chamfer=0.012)
            b2 = stone_block("plinth_cap", (0.76, 0.76, zp * 0.18), loc=(0, 0, zp * 0.91), rng=r, chamfer=0.03)
            blocks = (b0, b1, b2)
        for b in blocks:
            assign(b, st)
            paint(b, grime=mask_mul(height_ramp(0.0, 0.22), noise_mask(0.2, seed=sd + 9, lo=0.2, hi=0.9), 0.45),
                  special=mask_mul(facing((0, 0, 1)), noise_mask(0.25, seed=sd), 0.6))
            stones.append(b)

    # ---- pedestal: octagonal casting - bolted flange, base mouldings, panelled die, cornice, ogee cap
    rb = 0.205
    ped_h = min(0.95, H * 0.27)
    d0, d1 = 0.15, ped_h * 0.74
    rn = 0.112
    prof = [(rb + 0.036, -0.02), (rb + 0.05, 0.0), (rb + 0.05, 0.034), (rb + 0.045, 0.04), (rb + 0.016, 0.042),
            (rb + 0.016, 0.066), (rb + 0.011, 0.074), (rb + 0.019, 0.082), (rb + 0.019, 0.094), (rb + 0.008, 0.104),
            (rb + 0.002, 0.125), (rb, d0), (rb, d1), (rb + 0.004, d1 + 0.012), (rb + 0.02, d1 + 0.026),
            (rb + 0.028, d1 + 0.04), (rb + 0.028, d1 + 0.056), (rb + 0.014, d1 + 0.066)]
    c0 = d1 + 0.066
    for t, k in ((0.2, 0.36), (0.45, 0.66), (0.72, 0.88), (1.0, 1.0)):
        prof.append((lerp(rb + 0.014, rn, k), c0 + (ped_h - c0) * t))
    prof.append((0.0, ped_h))
    if far:                                       # flange, base, die, cornice, cap
        prof = lod_exact([(rb + 0.05, 0.0), (rb + 0.05, 0.036), (rb, 0.1), (rb, d1), (rb + 0.028, d1 + 0.045), (rn, ped_h), (0.0, ped_h)])
    ped = lathe("pedestal", prof, segments=8, loc=(0, 0, zp), rot=(0, 0, 22.5), smooth_angle=25)
    if not far:
        densify(ped, 0.065)
        bevel(ped, 0.004, segments=1, angle=35)
    body.append(ped)
    apo = rb * cos(radians(22.5))                 # distance to a flat
    fw = 2 * rb * sin(radians(22.5))              # width of a flat
    ph0, ph1 = d0 + 0.035, d1 - 0.03
    pw, phh = fw * 0.76, ph1 - ph0
    zc = zp + (ph0 + ph1) / 2
    frame = shape_extrude("panel", rounded_rect(pw, phh, 0.014, 1), [rounded_rect(pw - 0.032, phh - 0.032, 0.008, 1)],
                          depth=0.007, bevel=0.0015)
    field = box("field", (pw - 0.052, phh - 0.052, 0.004), loc=(0, 0, 0.002), bevel=0.0015, segments=1)
    boss = lathe("boss", [(0.037, 0.0), (0.037, 0.004), (0.027, 0.009), (0.02, 0.009), (0.015, 0.015), (0.0, 0.019)],
                 segments=4, smooth_angle=30)
    pm, bm = [], []
    for k in range(1, 8):                         # face 0 (front, -Y) carries the door
        a = k * 45.0
        cx, cy = apo * sin(radians(a)), -apo * cos(radians(a))
        pm.append(trs((cx, cy, zc), (90, 0, a)))
        bm.append(trs((cx, cy, zc), (90, 0, a)))
        if phh > 0.34:
            for sgn in (-1, 1):
                bm.append(trs((cx, cy, zc + sgn * phh * 0.3), (90, 0, a), 0.55))
    body.append(instances_to_mesh("panels", frame, pm, keep_base=False))
    body.append(instances_to_mesh("fields", field, pm, keep_base=False))
    gilded.append(instances_to_mesh("bosses", boss, bm, keep_base=False))
    tiny.extend([body[-2], body[-1], gilded[-1]])
    if far and p.gilt:                            # gilded bosses are colour: one low pyramid per face stays
        gilded.append(instances_to_mesh("bosses", lathe("boss", [(0.04, 0.003), (0.0, 0.016)], segments=lod_exact(4)), pm, keep_base=False))
    # meter / stop-cock door: arched plate on two knuckle hinges, brass lock and maker's plate
    door = extrude_profile("door", arch_pts(pw, phh, kind="segmental", n=8), 0.009, loc=(0, -apo, zp + ph0), rot=(90, 0, 0), bevel=0.002)
    body.append(door)
    tiny.append(door)
    hinge_z = [zp + ph0 + phh * 0.2, zp + ph0 + phh * 0.76]
    for hz in hinge_z:
        hk = cylinder("hinge", 0.0075, 0.052, loc=(-pw / 2 - 0.003, -apo - 0.008, hz), verts=8)
        smooth(hk, 40)
        body.append(hk)
        body.append(box("hinge_strap", (0.042, 0.005, 0.02), loc=(-pw / 2 + 0.02, -apo - 0.0105, hz), bevel=0.0015, segments=1))
        tiny.extend([hk, body[-1]])
    bits.append(lathe("lock", [(0.0, 0.0), (0.014, 0.0), (0.015, 0.003), (0.011, 0.006), (0.006, 0.006), (0.006, 0.013), (0.0, 0.013)],
                      segments=10, loc=(pw / 2 - 0.022, -apo - 0.009, zc - 0.03), rot=(90, 0, 0)))
    tiny.append(bits[-1])
    lines = [s.strip() for s in str(p.maker).split("/") if s.strip()][:2]
    if lines:
        ptw, pth = pw - 0.026, 0.018 + 0.017 * len(lines)
        zpl = zp + ph0 + phh * 0.7
        bits.append(box("maker_plate", (ptw, 0.004, pth), loc=(0, -apo - 0.011, zpl), bevel=0.0015, segments=1))
        lod_faces(bits[-1], keep=(0, -1, 0), above=1)      # far: the brass plate is a glint on the door - its front only
        for i, s in enumerate(lines):
            tx = text("maker_txt", s, size=0.0105, depth=0.0022, loc=(0, -apo - 0.013, zpl + (len(lines) - 1) * 0.0085 - i * 0.017 - 0.005),
                      font="sans_narrow")
            mn, mx = world_bbox([tx])
            if mx.x - mn.x > ptw - 0.01:
                kx = (ptw - 0.01) / (mx.x - mn.x)
                tx.scale = (kx, kx, kx)
            bm_ = bmesh.new()
            bm_.from_mesh(tx.data)
            bmesh.ops.dissolve_limit(bm_, angle_limit=radians(12), verts=bm_.verts[:], edges=bm_.edges[:])
            bm_.to_mesh(tx.data)
            bm_.free()
            bits.append(tx)
            tiny.append(tx)
    # holding-down bolts on the base flange
    bt = bolt("hd_bolt", 0.012, 0.01)
    bolt_xy = [((rb + 0.031) * cos(radians(22.5 + k * 90)), (rb + 0.031) * sin(radians(22.5 + k * 90))) for k in range(4)]
    body.append(instances_to_mesh("hd_bolts", bt, [trs((x, y, zp + 0.041), (0, 0, rc.u(0, 60))) for x, y in bolt_xy], keep_base=False))

    # ---- gadrooned bulb between pedestal and shaft
    zt0 = zp + ped_h
    tb = 0.21
    bp = [(0.106, 0.0), (0.110, 0.008), (0.106, 0.018), (0.088, 0.026), (0.084, 0.036), (0.100, 0.05), (0.128, 0.072),
          (0.138, 0.095), (0.130, 0.118), (0.108, 0.14), (0.09, 0.155), (0.086, 0.165), (0.094, 0.172), (0.094, 0.182),
          (0.086, 0.19), (0.082, 0.21)]
    ng = 12
    if far:
        bp = [(0.106, 0.0), (0.086, 0.03), (0.138, 0.095), (0.088, 0.16), (0.082, 0.21)]
    bulb = lathe("bulb", [(r, zt0 + z) for r, z in bp], segments=lod_pick(ng * 4, ng * 4, 8), smooth_angle=60)

    def gad(co):
        z = co.z - zt0
        e = smoothstep(0.042, 0.07, z) * (1.0 - smoothstep(0.13, 0.156, z))
        if e <= 0.0:
            return co
        a = atan2(co.y, co.x)
        k = 1.0 - 0.11 * e * (1.0 - abs(cos(a * ng / 2)))
        return Vector((co.x * k, co.y * k, co.z))

    if not far:                                   # (eight sides cannot carry twelve gadroons)
        transform_verts(bulb, gad)
    body.append(bulb)

    # ---- shaft: tapering, fluted between turned bands
    z0 = zt0 + tb - 0.004
    z1 = zp + H - 0.34
    r0, r1 = 0.072, 0.046
    zs0, zs1 = z0 + 0.05, z1 - 0.07

    def rad_z(z):
        return r0 + (r1 - r0) * clamp((z - zs0) / max(zs1 - zs0, 1e-3))

    nf = int(p.flutes)
    fz0 = z0 + 0.27
    fz1 = z1 - (0.27 if p.ladder_bar else 0.13)
    if fz1 - fz0 < 0.3:
        nf = 0
    bz = fz0 + (fz1 - fz0) * 0.36
    keys = [fz0, fz0 + 0.05, fz1 - 0.05, fz1, bz - 0.06, bz - 0.03, bz + 0.03, bz + 0.06]
    nzr = max(10, int((zs1 - zs0) / 0.1))
    zs = sorted([zs0 + (zs1 - zs0) * i / nzr for i in range(1, nzr)] + [k for k in keys if zs0 + 0.02 < k < zs1 - 0.02])
    zz = []
    for z in zs:
        if not zz or z - zz[-1] > 0.02:
            zz.append(z)
    sp = [(0.083, z0 - 0.004), (r0 + 0.017, z0 + 0.008), (r0 + 0.017, z0 + 0.02), (r0 + 0.005, z0 + 0.032), (r0, zs0)]
    sp += [(rad_z(z), z) for z in zz]
    sp += [(r1, zs1), (r1 + 0.008, z1 - 0.058), (r1 + 0.02, z1 - 0.04), (r1 + 0.024, z1 - 0.028), (r1 + 0.012, z1 - 0.012), (r1 + 0.004, z1)]
    zsplit = fz1 + 0.03
    seg_s = lod_pick(max(32, nf * 4 if nf <= 12 else nf * 3), 32, 6)
    shaft = lathe("shaft", [q for q in sp if q[1] < zsplit - 0.01] + [(rad_z(zsplit), zsplit)], segments=seg_s, smooth_angle=60)
    head.append(lathe("shaft_top", [(rad_z(zsplit), zsplit)] + [q for q in sp if q[1] > zsplit + 0.01], segments=lod_pick(32, 32, 6),
                      smooth_angle=60))

    def flute(co):
        z = co.z
        if nf <= 0 or z < fz0 or z > fz1:
            return co
        # flutes fade in/out at the ends and stop either side of the mid band
        e = min(smoothstep(fz0, fz0 + 0.05, z), 1.0 - smoothstep(fz1 - 0.05, fz1, z), smoothstep(0.03, 0.06, abs(z - bz)))
        c = 0.5 + 0.5 * cos(atan2(co.y, co.x) * nf)
        d = 0.006 * e * (1.0 - c ** 3)
        r = hypot(co.x, co.y)
        k = (r - d) / r if r > 1e-6 else 1.0
        return Vector((co.x * k, co.y * k, co.z))

    if not far:
        transform_verts(shaft, flute)
    body.append(shaft)
    rr = rad_z(bz)
    band = lathe("band", [(rr - 0.002, bz - 0.034), (rr + 0.006, bz - 0.028), (rr + 0.006, bz - 0.02), (rr + 0.016, bz - 0.01), (rr + 0.019, bz),
                          (rr + 0.016, bz + 0.01), (rr + 0.006, bz + 0.02), (rr + 0.006, bz + 0.028), (rr - 0.002, bz + 0.034)],
                 segments=lod_pick(32, 32, 6))
    (gilded if p.gilt else body).append(band)
    ra = rad_z(fz1 + 0.03)
    gilded.append(lathe("astragal", [(ra - 0.002, fz1 + 0.016), (ra + 0.007, fz1 + 0.022), (ra + 0.009, fz1 + 0.03), (ra + 0.007, fz1 + 0.038),
                                   (ra - 0.002, fz1 + 0.044)], segments=lod_pick(28, 28, 6)))
    if not p.gilt:
        tiny.append(gilded[-1])

    # ---- acanthus collar: broad lobed leaves hugging the shaft foot, tips rolled over; a shorter row laps over them
    def leaf_fn(rs, hl, w, curl, lobes):
        def fn(u, v):
            s = (u - 0.5) * 2.0
            out = (0.6 + 0.4 * sin(min(v / 0.4, 1.0) * pi / 2)) * (1.0 - 0.4 * smoothstep(0.45, 1.0, v))
            out *= 1.0 - 0.14 * abs(sin(v * pi * lobes))
            out *= sqrt(1.0 - 0.55 * clamp((v - 0.86) / 0.14) ** 2)
            t = clamp((v - 0.72) / 0.28)
            phi = t * radians(165)
            ro_ = 0.004 + 0.006 * sin(v * pi * 0.75) + curl * (1.0 - cos(phi)) + 0.0045 * (1.0 - abs(s)) ** 0.7
            z = hl * min(v, 0.72) + curl * sin(phi)
            ang = 0.5 * w * out * s / rs
            r = rs - 0.012 * min(v, 0.72) * hl / 0.14 * 0.25 + ro_
            return (r * sin(ang), -r * cos(ang), z)
        return fn

    zlf = z0 + 0.026
    for (nm, rs, hl, w, curl, lobes, off) in (("leaves_tall", r0 + 0.003, 0.27, 0.09, 0.016, 3.0, 0.0),
                                               ("leaves_short", r0 + 0.012, 0.15, 0.072, 0.013, 2.0, 22.5)):
        lm = [trs((0, 0, zlf), (0, 0, off + k * 45), (1.0, 1.0, rc.vary(1.0, 0.07))) for k in range(8)]
        if far:
            continue
        lf = surface(nm, leaf_fn(rs, hl, w, curl, lobes), 4, 9 if hl > 0.2 else 7, thickness=0.004)
        realize(lf)
        body.append(instances_to_mesh(nm, lf, lm, keep_base=False))
    if far:                                       # the two rows of leaves as one flared cup around the shaft foot
        # (its rim ends inside the six-sided shaft all round: nobody looks down into the cup)
        body.append(lathe("leaf_cup", [(r0 + 0.012, zlf), (r0 + 0.017, zlf + 0.15), (r0 + 0.04, zlf + 0.2),
                                       (rad_z(zlf + 0.215) * 0.8, zlf + 0.215)], segments=lod_exact(8), smooth_angle=60))

    # ---- ladder bar with ball ends
    zb = z1 - 0.19
    if p.ladder_bar:
        tl = rc.jitter(0.0, 1.2 * wk)
        lb = cylinder("ladder_bar", 0.011, 0.72, loc=(0, 0, zb), rot=(0, 90 + tl, 0), verts=12)
        smooth(lb, 40)
        head.append(lb)
        rl = rad_z(zb)
        head.append(lathe("ladder_boss", [(rl + 0.001, zb - 0.024), (rl + 0.011, zb - 0.014), (rl + 0.013, zb), (rl + 0.011, zb + 0.014), (rl + 0.001, zb + 0.024)], segments=24))
        tiny.append(head[-1])
        for sx in (-1, 1):
            sb = sphere("ladder_ball", 0.021, loc=(sx * 0.36 * cos(radians(tl)), 0, zb - sx * 0.36 * sin(radians(tl))), segments=12, rings=8)
            smooth(sb, 40)
            head.append(sb)
            tiny.append(sb)

    # ---- lantern: four tapered glazed sides in beaded frames, enamel reflector, burner + mantle, hipped roof, vent cap
    def pane_mask(cxy, tdir, zb0, hbody, wbot, wtop, soot, dust, base, seed):
        nz = noise_mask(0.06, seed=seed, lo=0.2, hi=0.8)

        def fn(co, nn):
            v = np.clip((co[:, 2] - zb0) / hbody, 0.0, 1.0)
            half = 0.5 * (wbot + (wtop - wbot) * v)
            s = np.clip(np.abs((co[:, 0] - cxy[0]) * tdir[0] + (co[:, 1] - cxy[1]) * tdir[1]) / half, 0.0, 1.0)
            top = np.clip((v - 0.45) / 0.55, 0.0, 1.0) ** 1.7
            corner = np.clip(1.0 - v / 0.32, 0.0, 1.0) * s ** 2
            edge = np.clip((s - 0.7) / 0.3, 0.0, 1.0) * 0.3
            return np.clip(base + (soot * (top + edge) + dust * corner) * (0.5 + 0.5 * nz(co, nn)), 0.0, 1.0)
        return fn

    def lantern(cx, zl, idx, lit, state, kp, dent, sn):
        wb = W * 0.56
        hb = W * 1.05
        zb0 = zl + 0.03
        zt = zb0 + hb
        tilt = degrees(atan((W - wb) * 0.5 / hb))
        # bottom frame + floor plate
        ro, ri = (wb / 2 + 0.014) * s2, (wb / 2 - 0.016) * s2
        if far:                                   # frame and floor plate as one closed slab
            head.append(lathe("lt_base", [(0.0, 0.0), (ro, 0.0), (ro, 0.03), (0.0, 0.03)],
                              segments=4, loc=(cx, 0, zl), rot=(0, 0, 45), smooth_angle=30))
        else:
            head.append(lathe("lt_base", [(ri, 0.004), (ro - 0.005, 0.0), (ro, 0.005), (ro, 0.022), (ro - 0.007, 0.03), (ri, 0.03), (ri, 0.004)],
                              segments=4, loc=(cx, 0, zl), rot=(0, 0, 45), smooth_angle=30))
            head.append(lathe("lt_floor", [(0.0, 0.008), (ri + 0.003, 0.008), (ri + 0.003, 0.015), (0.0, 0.015)],
                              segments=4, loc=(cx, 0, zl), rot=(0, 0, 45), smooth_angle=30))
        # corner bars (far: a little stouter - they stand for the glazing beads too; closed like the full ones, their
        # slanted ends show under the top frame)
        dia = [(0.0125, 0.0), (0.0, 0.0125), (-0.0125, 0.0), (0.0, -0.0125)]
        if far:
            dia = [(0.017, 0.0), (-0.0085, 0.0147), (-0.0085, -0.0147)]
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            head.append(sweep("lt_post", [(cx + sx * wb / 2, sy * wb / 2, zb0 - 0.002), (cx + sx * W / 2, sy * W / 2, zt + 0.002)], dia,
                              up=(sx, sy, 0), smooth_path=False))
        # glazing beads that hold each pane
        Ls = hb / cos(radians(tilt))
        ob, ot = wb / 2 - 0.006, W / 2 - 0.006
        if not far:
            fr = shape_extrude("lt_bead", [(-ob, 0), (ob, 0), (ot, Ls), (-ot, Ls)],
                               [[(-ob + 0.011, 0.011), (ob - 0.011, 0.011), (ot - 0.011, Ls - 0.011), (-ot + 0.011, Ls - 0.011)]], depth=0.006)
            ms = []
            for k in range(4):
                a = radians(90 * k)
                ms.append(trs((cx + sin(a) * (wb / 2 - 0.007), -cos(a) * (wb / 2 - 0.007), zb0), (90 + tilt, 0, 90 * k)))
            head.append(instances_to_mesh("lt_beads", fr, ms, keep_base=False))
        # panes
        soot = (0.22 + 0.3 * wr) * (1.0 if lit else 1.6)
        for k in range(4):
            stt = state if k == kp else "sound"
            a = radians(90 * k)
            tdir, d = (cos(a), sin(a)), (sin(a), -cos(a))

            def fn(u, v, tdir=tdir, d=d, lift=0.0):
                wv = lerp(wb, W, v)
                x = (u - 0.5) * (wv - 0.014)
                out = wv / 2 - 0.004 + lift
                return (cx + tdir[0] * x + d[0] * out, tdir[1] * x + d[1] * out, zb0 + hb * v)

            if stt == "missing":
                # a few shards left in the putty
                vs, fs = [], []
                for (ua, ub, uc, va, vb, vc) in ((0.0, 0.34, 0.0, 0.0, 0.0, 0.3), (1.0, 0.7, 1.0, 1.0, 1.0, 0.72), (0.55, 1.0, 1.0, 0.0, 0.0, 0.17)):
                    i0 = len(vs)
                    vs += [fn(ua, va), fn(ub, vb), fn(uc, vc)]
                    fs += [(i0, i0 + 1, i0 + 2), (i0 + 2, i0 + 1, i0)]
                g = mesh_object("lt_shards", vs, fs)
                assign(g, glass)
                paint(g, grime=0.35)
                continue
            g = surface("lt_pane", fn, lod_pick(6, 6, 1), lod_pick(8, 8, 1))
            assign(g, glass)
            paint(g, grime=pane_mask((cx, 0.0), tdir, zb0, hb, wb, W, soot * (1.8 if stt == "sooted" else 1.0), 0.3 + 0.3 * wr,
                                     0.3 if stt == "sooted" else 0.0, sd + 30 + k + 7 * idx))
            if stt == "cracked":
                rk = rc.fork("crack")
                u0, v0 = rk.u(0.35, 0.65), rk.u(0.3, 0.55)
                vs, fs = [], []
                nrm = Vector((d[0], d[1], -0.2)).normalized()
                a0 = rk.u(0, 360)
                for j in range(5):
                    ang = radians(a0 + j * 72 + rk.jitter(0, 18))
                    pu, pv = u0, v0
                    prev = Vector(fn(pu, pv, lift=0.0018))
                    for stp in range(5):
                        ang += radians(rk.jitter(0, 22))
                        pu += cos(ang) * 0.17
                        pv += sin(ang) * 0.13
                        stop = not (0.0 < pu < 1.0 and 0.0 < pv < 1.0)
                        pu, pv = clamp(pu, 0.0, 1.0), clamp(pv, 0.0, 1.0)
                        cur = Vector(fn(pu, pv, lift=0.0018))
                        sdv = (cur - prev)
                        if sdv.length > 1e-5:
                            wv_ = sdv.cross(nrm).normalized() * 0.0013
                            i0 = len(vs)
                            vs += [tuple(prev - wv_), tuple(prev + wv_), tuple(cur + wv_), tuple(cur - wv_)]
                            fs += [(i0, i0 + 1, i0 + 2, i0 + 3), (i0 + 3, i0 + 2, i0 + 1, i0)]
                        prev = cur
                        if stop:
                            break
                ck = mesh_object("lt_cracks", vs, fs)
                assign(ck, mat("glass_clear", "#8a8a8a", opacity=0.85))
                tiny.append(ck)
        # top frame (cornice) and the white-enamel reflector under the roof
        ri2, ro2 = (W / 2 - 0.014) * s2, (W / 2 + 0.02) * s2
        head.append(lathe("lt_top", lod_pick([(ri2, 0.0), (ro2 - 0.014, 0.0), (ro2 - 0.008, 0.01), (ro2, 0.016), (ro2 + 0.005, 0.028), (ro2 + 0.005, 0.036),
                                              (ri2, 0.036), (ri2, 0.0)], None, [(ri2, 0.014), (ri2, 0.0), (ro2 - 0.014, 0.0), (ro2 + 0.005, 0.036)]),   # far: inner lip up to the reflector
                          segments=4, loc=(cx, 0, zt), rot=(0, 0, 45), smooth_angle=30))
        refl = lathe("lt_reflector", [(0.0, 0.05), (ri2 + 0.004, 0.012)], segments=4, loc=(cx, 0, zt), rot=(0, 0, 45))
        if refl.data.polygons[0].normal.z > 0:
            refl.data.flip_normals()
        assign(refl, mat("emissive", "#ffc27a", glow=1.3) if lit else mat("ceramic_glazed", "#cfc8b8"))
        # roof
        zr0 = zt + 0.036
        re = (W / 2 + 0.05) * s2
        hr = W * 0.5
        hips = True
        if p.lantern == "ogee":
            shp = [(1.0, 0.0), (1.0, 0.035), (0.8, 0.1), (0.64, 0.22), (0.52, 0.37), (0.45, 0.54), (0.4, 0.7), (0.33, 0.84), (0.25, 0.94), (0.2, 1.0)]
            if far:
                shp = [(1.0, 0.0), (1.0, 0.035), (0.62, 0.25), (0.42, 0.63), (0.2, 1.0)]
        elif p.lantern == "pyramid":
            shp = [(1.0, 0.0), (1.0, 0.035), (0.74, 0.31), (0.47, 0.6), (0.2, 0.9)]
        else:
            hips = False
            shp = None
        zsf = zr0 - zl
        if far:                                   # the soffit dips into the open top frame: no slit under the eaves
            zsf -= 0.01
        if hips:
            zr1 = zr0 + hr * shp[-1][1]
            roof = lathe("lt_roof", [(ro2 - 0.012, zsf)] + [(re * a_, zr0 - zl + hr * b_) for a_, b_ in shp] + [(0.0, zr1 - zl)],
                         segments=4, loc=(cx, 0, zl), rot=(0, 0, 45), smooth_angle=35)
            dphi = radians(45.0 + 90.0 * (1 + (sd % 2) * 2))
            dcen = Vector((cos(dphi) * re * 0.5 / s2, sin(dphi) * re * 0.5 / s2, zr0 - zl + hr * 0.36))
            for k in range(0 if far else 4):      # hip rolls: not in the far level
                a = radians(45 + 90 * k)
                pts = [(cx + cos(a) * re * 1.05, sin(a) * re * 1.05, zr0 + 0.018)]
                pts += [(cx + cos(a) * re * a_, sin(a) * re * a_, zr0 + hr * b_ + 0.003) for a_, b_ in shp[1:]]
                head.append(sweep("lt_hip", pts, 0.0065, sides=5, smooth_path=False))
        else:
            rd = W / 2 + 0.03
            head.append(lathe("lt_roof_plate", [(ro2 - 0.012, zsf), (re, zr0 - zl), (re, zr0 - zl + 0.012), (re - 0.022, zr0 - zl + 0.022), (0.0, zr0 - zl + 0.022)],
                              segments=4, loc=(cx, 0, zl), rot=(0, 0, 45), smooth_angle=30))
            rt = 0.2 * re / s2
            dp = [(rd + 0.006, zr0 - zl + 0.02), (rd + 0.006, zr0 - zl + 0.03), (rd, zr0 - zl + 0.036)]
            for i in range(1, 8):
                an = radians(90.0 * i / 7)
                dp.append((rt + (rd - rt) * cos(an), zr0 - zl + 0.036 + hr * 0.8 * sin(an)))
            zr1 = zr0 + 0.036 + hr * 0.8
            roof = lathe("lt_roof", dp + [(0.0, zr1 - zl)], segments=lod_pick(20, 20, 8), loc=(cx, 0, zl), smooth_angle=50)
            dphi = radians(200.0 + 37.0 * (sd % 5))
            dcen = Vector((cos(dphi) * rd * 0.75, sin(dphi) * rd * 0.75, zr0 - zl + 0.036 + hr * 0.5))
        if dent > 0.01:
            densify(roof, 0.05)
            sg = 0.26 * W
            push = Vector((cos(dphi) * 0.55, sin(dphi) * 0.55, 0.83)) * (0.04 * dent)

            def dfn(co):
                f = exp(-((co - dcen).length / sg) ** 2)
                return co - push * f
            transform_verts(roof, dfn)
        head.append(roof)
        # vent chimney, cap and finial
        rv = 0.2 * (W / 2 + 0.05) + 0.006
        tv = (rc.jitter(0, 2.5 * wk), rc.jitter(0, 2.5 * wk), 0)
        if far:                                   # vent, cap and finial as one five-sided turning
            head.append(lathe("lt_vent", [(rv, -0.004), (rv, 0.052), (rv + 0.027, 0.07), (0.021, 0.108), (0.026, 0.146), (0.0, 0.215)],
                              segments=lod_exact(5), loc=(cx, 0, zr1), rot=tv))
        else:
            head.append(lathe("lt_vent", [(rv + 0.012, -0.004), (rv + 0.012, 0.006), (rv, 0.012), (rv, 0.05), (rv - 0.014, 0.052), (rv - 0.014, 0.064),
                                          (rv + 0.024, 0.066), (rv + 0.027, 0.074), (rv + 0.012, 0.09), (0.02, 0.108), (0.0, 0.108)],
                              segments=14, loc=(cx, 0, zr1), rot=tv))
            fin = lathe("lt_finial", [(0.02, 0.104), (0.013, 0.122), (0.024, 0.136), (0.028, 0.15), (0.022, 0.164), (0.009, 0.176), (0.006, 0.2), (0.0, 0.215)],
                        segments=12, loc=(cx, 0, zr1), rot=tv)
            head.append(fin)
        # burner: gas pipe through the floor, brass gallery, incandescent mantle
        zbn = 0.03 + hb * 0.5
        pp = cylinder("lt_pipe", 0.0065, zbn - 0.015, loc=(cx, 0, zl + (zbn + 0.015) / 2), verts=8)
        smooth(pp, 40)
        bits.append(pp)
        bits.append(lathe("lt_burner", [(0.0065, zbn - 0.03), (0.012, zbn - 0.024), (0.012, zbn - 0.014), (0.008, zbn - 0.008), (0.02, zbn + 0.004),
                                        (0.025, zbn + 0.014), (0.025, zbn + 0.02), (0.014, zbn + 0.024), (0.0, zbn + 0.024)], segments=12, loc=(cx, 0, zl)))
        tiny.extend([pp, bits[-1]])
        if far:                                   # pipe and burner as one turning: it keeps the lantern floor in its shadow
            bits.append(lathe("lt_burner", [(0.0075, 0.03), (0.0075, zbn - 0.026), (0.036, zbn + 0.016), (0.0, zbn + 0.024)],
                              segments=lod_exact(4), loc=(cx, 0, zl)))
        zm = zbn + 0.024
        if far:                                   # six sides, drawn round the full mantle so it shows as much light
            mt = lathe("lt_mantle", [(0.0, zm), (0.0225, zm + 0.008), (0.0225, zm + 0.05), (0.0, zm + 0.068)], segments=lod_exact(6), loc=(cx, 0, zl))
        else:
            mt = lathe("lt_mantle", [(0.011, zm), (0.018, zm + 0.01), (0.02, zm + 0.03), (0.017, zm + 0.05), (0.009, zm + 0.062), (0.0, zm + 0.066)],
                       segments=12, loc=(cx, 0, zl))
        assign(mt, mat("emissive", "#ffd9a0", glow=12.0) if lit else mat("ceramic_glazed", "#cfc8b8"))
        lamps.append((cx, 0.0, zl + zm + 0.033))
        socket(f"lantern{idx}_mount", "attach", Lp((cx, 0, zl)), role="lantern_mount")
        if lit:
            light(f"lantern{idx}_lamp", Lp((cx, 0, zl + zm + 0.033)), color="#ffbf6a", intensity=p.intensity, range=11.0, flicker=0.14, shadow_near=sn)
        return zm + 0.033

    # ---- head: capital with gas tap, scrolled cradle (single) or swan-neck arms with their own cradles
    ztop = zp + H
    bar = [(-0.011, -0.004), (0.011, -0.004), (0.011, 0.004), (-0.011, 0.004)]
    crn = (W * 0.56 / 2 + 0.014) * s2 - 0.004      # where a stay meets the lantern's bottom frame

    def cradle(cx, za_, zb_, rst, cr, s):
        hh = zb_ - za_
        pf = [(a * s, b * s) for a, b in bar]
        for k in range(4):
            a = radians(45 + 90 * k)
            ca, sa = cos(a), sin(a)
            ctl = [(rst, za_), (cr * 0.75, za_ + hh * 0.12), (cr * 1.03, za_ + hh * 0.5), (cr, zb_)]
            pts = bezier_path(*[(cx + r_ * ca, r_ * sa, z_) for r_, z_ in ctl], n=lod_pick(8, 8, 3))
            if far:                               # a three-sided bar in three straight runs; no scroll under it
                head.append(sweep("stay", lod_exact(pts), [(0.013 * s, 0.0), (-0.0065 * s, 0.0113 * s), (-0.0065 * s, -0.0113 * s)],
                                  up=(-sa, ca, 0), smooth_path=False, cap=False))
                continue
            head.append(sweep("stay", pts, pf, up=(-sa, ca, 0), smooth_path=False))
            sc = spiral_path(0.006 * s, 0.04 * s, 1.3, n=12, a0_deg=200, plane="XZ", center=(cr * 0.62, 0, za_ - 0.005))
            o = sweep("scroll", sc, pf, scale_fn=lambda t: 0.55 + 0.45 * t, up=(0, 1, 0), smooth_path=False)
            place(o, loc=(cx, 0, 0), rot=(0, 0, 45 + 90 * k))
            head.append(o)

    def tap(zt_):
        if far:
            return
        tbd = cylinder("tap_body", 0.011, 0.1, loc=(0, 0, zt_), rot=(90, 0, 0), verts=10)
        smooth(tbd, 40)
        head.append(tbd)
        bits.append(cylinder("tap_nut", 0.016, 0.014, loc=(0, -0.046, zt_), rot=(90, 0, 0), verts=6))
        head.append(box("tap_lever", (0.15, 0.005, 0.011), loc=(0, -0.056, zt_), rot=(0, 24, 0), bevel=0.0015, segments=1))
        for sx in (-1, 1):
            head.append(torus("tap_eye", 0.011, 0.0032, loc=(sx * 0.083 * cos(radians(24)), -0.056, zt_ - sx * 0.083 * sin(radians(24))),
                              rot=(90, 0, 0), major_seg=12, minor_seg=6))

    state = p.pane
    if state == "auto":
        state = "cracked" if wr > 0.85 else ("sooted" if wr > 0.65 else "sound")
    kp = rc.pick([0, 0, 1, 3])
    lz0 = 0.03 + W * 1.05 * 0.5 + 0.057          # mantle height above a lantern's base
    capb = [(r1 + 0.004, z1), (r1 + 0.02, z1 + 0.012), (r1 + 0.026, z1 + 0.03), (r1 + 0.01, z1 + 0.05)]
    if n == 1:
        head.append(lathe("capital", capb + [(0.032, z1 + 0.09), (0.028, z1 + 0.2), (0.034, z1 + 0.24), (0.04, z1 + 0.3), (0.05, ztop - 0.012),
                                             (0.05, ztop), (0.0, ztop)], segments=24))
        cradle(0.0, z1 + 0.06, ztop + 0.022, 0.03, crn, 1.0)
        tap(z1 + 0.15)
        # shadow_near sits just under the lantern's base frame: the glazing bars throw no cage shadow, but the capital,
        # stays and ladder bar do (the dark disc and thin radiating shadows at the foot of a real gas lamp)
        lantern(0.0, ztop, 0, True, state, kp, p.dent, lz0 + 0.014)
    else:
        R = max(p.spread, W / 2 + 0.16)
        zl = ztop + 0.04
        za = z1 + 0.13
        zcn = ztop + 0.34
        if n == 2:
            std = capb + [(0.036, z1 + 0.08), (0.034, z1 + 0.34), (0.05, z1 + 0.37), (0.056, z1 + 0.41), (0.03, z1 + 0.46), (0.02, z1 + 0.54)]
            head.append(lathe("standard", std, segments=24))
            sf = lathe("spire", [(0.02, z1 + 0.54), (0.034, z1 + 0.58), (0.03, z1 + 0.63), (0.008, z1 + 0.72), (0.0, z1 + 0.745)], segments=16)
            head.append(sf)
        else:
            std = capb + [(0.036, z1 + 0.08), (0.034, zcn - 0.25), (0.044, zcn - 0.235), (0.044, zcn - 0.215), (0.03, zcn - 0.195), (0.026, zcn - 0.06),
                          (0.034, zcn - 0.03), (0.05, zcn - 0.012), (0.05, zcn), (0.0, zcn)]
            head.append(lathe("standard", std, segments=24))
            cradle(0.0, zcn - 0.18, zcn + 0.022, 0.028, crn, 0.9)
        col = lathe("arm_collar", [(0.034, za - 0.062), (0.05, za - 0.052), (0.055, za - 0.036), (0.047, za - 0.028), (0.047, za + 0.028),
                                   (0.055, za + 0.036), (0.05, za + 0.052), (0.034, za + 0.062)], segments=24)
        head.append(col)
        tap(za + 0.125)
        dead = [1, 0][:min(int(p.unlit), n - 1)]
        for i, sx in enumerate((-1, 1)):
            pts = bezier_path((sx * 0.04, 0, za), (sx * R * 0.5, 0, za - 0.27), (sx * R * 1.03, 0, za - 0.22), (sx * R, 0, zl - 0.19), n=20)
            head.append(sweep(f"arm{i}", pts, 0.0225, sides=10, scale_fn=lambda t: 1.0 - 0.29 * t, up=(0, 1, 0)))
            head.append(lathe("arm_stem", [(0.016, zl - 0.2), (0.027, zl - 0.186), (0.03, zl - 0.172), (0.02, zl - 0.158), (0.017, zl - 0.05),
                                           (0.03, zl - 0.03), (0.044, zl - 0.01), (0.044, zl), (0.0, zl)], segments=16, loc=(sx * R, 0, 0)))
            cradle(sx * R, zl - 0.16, zl + 0.022, 0.019, crn, 0.82)
            # scroll infill and a tie bar bracing the arm against the standard
            mid = pts[13]
            head.append(sweep(f"arm_tie{i}", [(sx * 0.036, 0, za + 0.1), (sx * R * 0.42, 0, za + 0.06), (mid[0], 0, mid[2] + 0.014)], bar, up=(0, 1, 0)))
            sc = spiral_path(0.008, 0.062, 1.6, n=lod_pick(22, 22, 8), a0_deg=90, plane="XZ", center=(sx * R * 0.4, 0, za - 0.055), ccw=sx > 0)
            if far:                               # the scroll stays in the outline: eight straight runs of a three-sided bar
                head.append(sweep(f"arm_scroll{i}", lod_exact(sc), [(0.012, 0.0), (-0.006, 0.0104), (-0.006, -0.0104)],
                                  scale_fn=lambda t: 0.6 + 0.4 * t, up=(0, 1, 0), smooth_path=False))     # (its ends stand free)
            else:
                head.append(sweep(f"arm_scroll{i}", sc, bar, scale_fn=lambda t: 0.5 + 0.5 * t, up=(0, 1, 0), smooth_path=False))
            lantern(sx * R, zl, i, i not in dead, state if i == 0 else "sound", kp, p.dent if i == 0 else 0.0, lz0 + 0.014)
        if n == 3:
            lantern(0.0, zcn, 2, True, "sound", kp, 0.0, lz0 + 0.014)

    lod_drop(tiny, above=1)
    for o in body + head:
        assign(o, iron)
    for o in gilded:
        assign(o, gilt)
    for o in bits:
        assign(o, brass)

    # ---- art-directed ageing
    zrust = zp + 0.05 + 0.3 * wr * rc.u(0.6, 1.0)
    n_fine = noise_mask(0.03, seed=sd + 2, lo=0.42, hi=0.6)
    n_chip = noise_mask(0.035, seed=sd + 3, lo=0.56, hi=0.66)
    n_halo = noise_mask(0.035, seed=sd + 3, lo=0.49, hi=0.6)
    n_big = noise_mask(0.22, seed=sd + 4, lo=0.2, hi=0.8)
    n_str = noise_mask(0.011, seed=sd + 5, lo=0.54, hi=0.78)
    n_hand = noise_mask(0.06, seed=sd + 6, lo=0.3, hi=0.7)
    sq = np.array([1.0, 1.0, 0.05])
    ledges = [(bz - 0.03, 0.42), (zp + d1, 0.3), (fz1 + 0.016, 0.3)]     # rust runs hang below these

    def prox(co):                                 # heat-baked, sooty paint close to the burners (matt and dark)
        out = np.zeros(len(co))
        for lx, ly, lz in lamps:
            dd = np.sqrt((co[:, 0] - lx) ** 2 + (co[:, 1] - ly) ** 2 + (co[:, 2] - lz) ** 2)
            out = np.maximum(out, np.clip((1.3 - dd) / 0.8, 0.0, 1.0) * np.where(co[:, 2] > lz + 0.12, 0.3, 0.88))
        return out

    def g_body(co, nn):                           # street dirt splashed up the bottom 0.15 m
        z = co[:, 2]
        return np.clip(_ramp(z, zp + 0.02, zp + 0.19) * (0.45 + 0.55 * n_big(co, nn)) * (0.55 + 0.45 * wr), 0.0, 1.0)

    def w_body(co, nn):                           # chips low down (boots, carts), hands polish the shaft at ~1.3 m
        z = co[:, 2]
        chips = n_chip(co, nn) * (0.2 + 0.8 * _ramp(z, zp + 0.15, zp + 1.0))
        hand = np.clip(1.0 - np.abs(z - (zp + 1.36)) / 0.36, 0.0, 1.0) ** 0.6 * (0.45 + 0.55 * n_hand(co, nn))
        return np.clip(np.maximum(chips * wk, hand * (0.5 + 0.5 * wr)), 0.0, 1.0)

    def s_body(co, nn):                           # rust: scale at the foot, bleeding from chips, runs under ledges and fixings
        z = co[:, 2]
        foot = np.clip((zrust + 0.1 * (n_big(co, nn) - 0.5) - z) / 0.05, 0.0, 1.0) * n_fine(co, nn)
        halo = n_halo(co, nn) * (0.2 + 0.8 * _ramp(z, zp + 0.15, zp + 1.0)) * 0.85
        out = np.maximum(foot, halo)
        st = n_str(co * sq, nn)
        for zl_, ln_ in ledges:
            out = np.maximum(out, _ramp(z, zl_, zl_ - ln_) ** 1.5 * (z < zl_ + 0.004) * st)
        for bx, by in bolt_xy:                    # holding-down bolts weep rust onto the flange
            dd = np.hypot(co[:, 0] - bx, co[:, 1] - by)
            out = np.maximum(out, np.clip(1.0 - dd / 0.045, 0.0, 1.0) * (z < zp + 0.08) * 0.9)
        for hz in hinge_z:                        # and the door hinges run down the door
            m = (np.abs(co[:, 0] + pw / 2 - 0.004) < 0.022) * (co[:, 1] < -apo + 0.02) * _ramp(z, hz - 0.02, hz - 0.16) * (z < hz + 0.03)
            out = np.maximum(out, m * 0.8)
        return np.clip(out * (0.15 + 0.75 * wr), 0.0, 1.0)

    def g_head(co, nn):                           # decades of gas soot: matt and dark towards the vent
        return np.clip(0.08 + 0.2 * n_big(co, nn), 0.0, 1.0)

    paint(body, grime=g_body, wear=w_body, special=s_body)
    paint(head, grime=g_head, wear=0.0, special=0.0)
    paint(gilded, grime=0.15, wear=mask_mul(n_hand, 0.5))
    paint(bits, grime=lambda co, nn: np.clip(0.12 + prox(co) * 0.75, 0.0, 1.0))

    # ---- apply the lean; the base flange is trimmed flush to the paving / plinth
    for o in list(bpy.context.scene.objects):
        if o.type == 'MESH' and o not in stones:
            loc, rot, scl = (M @ world_matrix(o)).decompose()
            o.location, o.rotation_euler, o.scale = loc, rot.to_euler(), scl
    mw = world_matrix(ped)
    mwi = mw.inverted()
    zmin = zp + 0.0005 if zp < 0.01 else zp - 0.004

    def trim(co):
        w = mw @ co
        if w.z < zmin:
            w.z = zmin
        return mwi @ w

    transform_verts(ped, trim)

    socket("banner_l", "attach", Lp((-0.07, 0, zp + H * 0.72)), rot=(0, 0, 90), role="banner")
    socket("basket_r", "attach", Lp((0.07, 0, zp + H * 0.66)), rot=(0, 0, -90), role="planter")
