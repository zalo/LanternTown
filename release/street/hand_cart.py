"""Costermonger's hand barrow: two spoked, iron-tyred wheels on a forged axle with semi-elliptic leaf
springs, a pair of long shafts that run on into turned push handles, cross bearers, a plank deck with
flared, lettered side boards on forged stays, two prop legs with iron shoes, and a load (paper-lined
pine crates of fruit with price tickets, zinc buckets of cut flowers, slumped burlap sacks, or a few
leftovers).  Optional striped canvas canopy: four poles strapped to the end boards, eaves plates, end
trusses, rafters and a ridge pole with the canvas laced to the end rafters and guyed to the poles, and
a small oil lantern (on a forged bracket under the canopy or on an iron crook).
The long side faces -Y (the customer side); the handles point to +X.
"""
from math import sin, cos, pi, radians, degrees, atan2, hypot, sqrt

import bmesh
import numpy as np
from tk import *

META = {"name": "Vendor hand cart", "class": "furniture", "mount": "ground", "tri_budget": 30000,
        "focus": [-0.3, -0.3, 0.98]}
PARAMS = {
    "length": {"default": 1.6, "min": 1.2, "max": 2.2, "desc": "bed length (m); the handles add ~0.75 m"},
    "width": {"default": 0.75, "min": 0.6, "max": 0.95, "desc": "bed width between the wheels (m)"},
    "wheel_diameter": {"default": 0.84, "min": 0.7, "max": 1.0, "desc": "wheel diameter over the iron tyre (m)"},
    "spokes": {"default": 12, "min": 8, "max": 16, "desc": "spokes per wheel"},
    "load": {"default": "fruit", "choices": ["fruit", "flowers", "sacks", "empty"], "desc": "what is on the bed"},
    "canopy": {"default": True, "desc": "striped canvas canopy on four poles"},
    "canopy_height": {"default": 1.22, "min": 1.05, "max": 1.5, "desc": "canopy eaves above the deck (m)"},
    "canopy_color": {"default": "#8e2a22", "desc": "canopy stripe colour"},
    "stripe_color": {"default": "#e6dcc2", "desc": "second canopy stripe colour"},
    "paint": {"default": "#467d62", "desc": "body paint colour (boards, shafts, legs)"},
    "trim": {"default": "#63362f", "desc": "paint colour of wheels, capping rails and canopy frame"},
    "bloom_color": {"default": "#b02f4c", "desc": "main flower colour (load = flowers); cream and a seeded colour go with it"},
    "lettering": {"default": "FRESH FRUIT", "desc": "name painted on the front side board ('' = none)"},
    "stencil": {"default": "KENT", "desc": "grower's mark stencilled on the produce crates ('' = none)"},
    "prices": {"default": "2d,3d,4d", "desc": "comma separated price tickets stuck in the produce crates ('' = none)"},
    "lantern": {"default": True, "desc": "hanging oil lantern"},
    "intensity": {"default": 9.0, "min": 0.0, "max": 40.0, "desc": "lantern intensity (cd)"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "condition: 0 = freshly painted, 1 = battered"},
}
VARIANTS = {
    "flower_seller": {"load": "flowers", "length": 1.5, "paint": "#585070", "trim": "#a8996f", "canopy_color": "#2f5d4c",
                      "stripe_color": "#e2dac2", "lettering": "FLOWERS", "bloom_color": "#b8452a", "intensity": 7.0, "wear": 0.3, "seed": 4},
    "coster_open": {"load": "fruit", "canopy": False, "length": 1.85, "paint": "#2c4f7c", "trim": "#9a7630",
                    "lettering": "CHOICE FRUIT", "stencil": "EVESHAM", "prices": "3d,2d,6d", "spokes": 14, "wheel_diameter": 0.92, "seed": 11},
    "sack_barrow": {"load": "sacks", "canopy": False, "length": 1.95, "width": 0.82, "wheel_diameter": 0.98, "spokes": 14,
                    "paint": "#6b4830", "trim": "#86714a", "lettering": "POTATOES", "stencil": "JERSEY", "wear": 0.95, "seed": 23},
    "empty_small": {"load": "empty", "canopy": False, "length": 1.25, "width": 0.64, "wheel_diameter": 0.72, "spokes": 10,
                    "paint": "#7c3a30", "trim": "#ab9a6c", "lettering": "", "intensity": 7.0, "wear": 0.7, "seed": 31},
}


# ------------------------------------------------------------------ small helpers
def _bx(name, size, loc=(0, 0, 0), rot=None, bev=0.002, grain='X'):
    o = box(name, size, loc=loc, rot=rot, bevel=bev, segments=1)
    uv_grain(o, grain)
    return o


def _mix(*terms):
    """Weighted sum of masks: terms = (weight, fn or float), clipped to 0..1."""
    def fn(co, n):
        acc = np.zeros(len(co))
        for w, t in terms:
            acc = acc + w * (t(co, n) if callable(t) else float(t))
        return np.clip(acc, 0.0, 1.0)
    return fn


def _lite_text(o, sides=True, ang=9.0):
    """Lettering clean-up: drop the faces against the board (and the sides for stencilled / painted
    marks) and dissolve the over-dense font outline."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bm.normal_update()
    lim = -0.5 if sides else 0.5
    bmesh.ops.delete(bm, geom=[fc for fc in bm.faces if fc.normal.z < lim], context='FACES')
    bmesh.ops.dissolve_limit(bm, angle_limit=radians(ang), use_dissolve_boundaries=False, verts=bm.verts[:], edges=bm.edges[:])
    bm.to_mesh(o.data)
    bm.free()
    return o


def _face_to(o, ref):
    """Flip an open surface so that it faces `ref` on average."""
    acc = Vector((0.0, 0.0, 0.0))
    for pl in o.data.polygons:
        acc += pl.normal * pl.area
    if acc.dot(Vector(ref)) < 0:
        o.data.flip_normals()
    return o


def _lift(hexcol, top=1.0):
    """Scale a colour so that its strongest channel reaches `top` (petal tints multiply a mid-value blade albedo)."""
    c = [int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    g = top / max(max(c), 1e-3)
    return "#" + "".join(f"{int(round(255 * clamp(v * g, 0.0, 1.0))):02x}" for v in c)


def _blade_mesh(name, blades, tile=0.25):
    """Petals / leaves, modelled two-sided (the runtime culls back faces).  A blade is (base, left, mid, right, tip)
    -> 4 triangles, or (base, left, right, tip) -> 2; each is mapped onto the whole tile of the 'leaf' texture
    (V from base to tip) so every blade gets its own midrib."""
    K5 = [(0.5, 0.04), (0.08, 0.5), (0.5, 0.48), (0.92, 0.5), (0.5, 0.96)]
    K4 = [(0.5, 0.04), (0.1, 0.55), (0.9, 0.55), (0.5, 0.96)]
    verts, faces, uvs = [], [], []
    for b in blades:
        key = K5 if len(b) == 5 else K4
        tri = [(0, 3, 2), (0, 2, 1), (2, 3, 4), (1, 2, 4)] if len(b) == 5 else [(0, 2, 3), (0, 3, 1)]
        for back in (False, True):
            i = len(verts)
            verts += [tuple(q) for q in b]
            uvs += key
            faces += [((i + t[0], i + t[2], i + t[1]) if back else (i + t[0], i + t[1], i + t[2])) for t in tri]
    o = mesh_object(name, verts, faces)
    uvl = o.data.uv_layers[0] if o.data.uv_layers else o.data.uv_layers.new(name="UVMap")
    for lp in o.data.loops:
        u, v = uvs[lp.vertex_index]
        uvl.data[lp.index].uv = (u * tile, v * tile)
    uv_keep(o)
    smooth(o, 80)
    return o


def _head(kind, Rh):
    """Blades of one flower head (stem axis = +Z, head radius ~Rh)."""
    blades = []

    def ring(n, a0, rt, ht, rs, hs, rm, hm, half, z0=0.0, rb=0.0):
        for i in range(n):
            a = a0 + 2 * pi * i / n

            def P(r, h, da=0.0):
                return (r * Rh * cos(a + da), r * Rh * sin(a + da), h * Rh)
            if rm is None:
                blades.append((P(rb, z0), P(rs, hs, half), P(rs, hs, -half), P(rt, ht)))
            else:
                blades.append((P(rb, z0), P(rs, hs, half), P(rm, hm), P(rs, hs, -half), P(rt, ht)))

    if kind == "rose":                                    # an open outer whorl round a tight cupped heart
        ring(5, 0.0, 1.0, 0.42, 0.8, 0.5, 0.62, 0.16, radians(35))
        ring(3, 0.5, 0.5, 0.98, 0.5, 0.74, 0.5, 0.36, radians(58), 0.04)
    elif kind == "daisy":                                 # flat ray florets round a boss
        ring(8, 0.0, 1.0, 0.16, 0.62, 0.2, None, 0.0, radians(17), 0.1, rb=0.16)
    else:                                                 # tulip: two whorls of upright petals
        ring(3, 0.0, 0.46, 1.75, 0.64, 1.0, 0.68, 0.75, radians(54))
        ring(3, pi / 3, 0.4, 1.66, 0.54, 0.95, 0.56, 0.72, radians(52), 0.05)
    return blades


def _wheel(R, nsp, wood_m, iron, kw, seed):
    """One wheel around local Z (axle), outer face = +Z, centre at the origin; weathering painted in place."""
    ty = 0.012
    fo = R - ty
    fd = min(0.062, R * 0.15)
    fi = fo - fd
    fw = 0.046
    chip = 0.55 + 0.5 * kw
    lo_ = 0.68 - 0.2 * kw
    nz = noise_mask(0.045, seed=seed + 3, lo=lo_, hi=lo_ + 0.2)
    nz2 = noise_mask(0.11, seed=seed + 5, lo=0.3, hi=0.7)

    def rr(co):
        return np.hypot(co[:, 0], co[:, 1])

    wood, metal = [], []
    nf = max(4, int(round(nsp / 2)))
    sa = 360.0 / nf
    g = 0.28
    na = max(5, int(round(sa / 7.5)))
    outer = circle_pts(fo + 0.0005, na, -sa / 2 + g, sa / 2 - g)
    inner = circle_pts(fi, na, -sa / 2 + g, sa / 2 - g)
    seg = extrude_profile("felloe", list(outer) + list(reversed(inner)), fw, bevel=0.004, segments=1)
    uv_grain(seg, 'Y')
    fel = instances_to_mesh("felloes", seg, [trs((0, 0, -fw / 2), (0, 0, k * sa + sa / 2)) for k in range(nf)], keep_base=False)

    def fel_wear(co, n):                                   # paint knocked off the arris by the tyre and at the joints
        r = rr(co)
        rim = np.clip((r - (fo - 0.014)) / 0.012, 0, 1)
        a = np.degrees(np.arctan2(co[:, 1], co[:, 0])) % sa
        joint = np.clip(1.0 - np.minimum(a, sa - a) / 3.0, 0, 1)
        return np.clip((0.04 + 0.5 * rim + 0.3 * joint * (0.4 + 0.6 * rim) + 0.7 * nz(co, n)) * chip, 0, 1)

    paint(fel, wear=fel_wear, grime=lambda co, n: 0.12 + 0.4 * np.clip((rr(co) - fi) / fd, 0, 1) * (0.4 + 0.6 * nz2(co, n)),
          special=_mix((0.5, nz2)))
    wood.append(fel)
    # spokes: square shoulder at the nave, egg section tapering to the felloe
    r0, r1 = 0.058, fi + 0.008
    ls = r1 - r0
    sp = lathe("spoke", [(0.0215, 0.0), (0.022, 0.036), (0.0165, 0.078), (0.0135, ls * 0.64), (0.0135, ls)], segments=6, smooth_angle=60)
    transform_verts(sp, lambda co: Vector((co.x * 1.35, co.y, co.z)))
    uv_grain(sp, 'Z')
    sm = []
    for k in range(nsp):
        a = radians((k + 0.5) * 360.0 / nsp)
        sm.append(trs((r0 * cos(a), r0 * sin(a), 0.0), (0, 90, degrees(a))))
    spk = instances_to_mesh("spokes", sp, sm, keep_base=False)
    smooth(spk, 60)

    def sp_wear(co, n):
        r = rr(co)
        sh = np.clip(1.0 - (r - r0 - 0.03) / 0.06, 0, 1)
        tip = np.clip((r - (r1 - 0.05)) / 0.05, 0, 1)
        return np.clip((0.04 + 0.42 * sh + 0.3 * tip + 0.7 * nz(co, n)) * chip, 0, 1)

    paint(spk, wear=sp_wear, grime=lambda co, n: 0.15 + 0.35 * nz2(co, n), special=_mix((0.5, nz2)))
    wood.append(spk)
    hub = lathe("hub", [(0, -0.085), (0.046, -0.085), (0.05, -0.08), (0.056, -0.05), (0.074, -0.034), (0.079, -0.012),
                        (0.079, 0.012), (0.074, 0.034), (0.058, 0.05), (0.05, 0.085), (0.044, 0.092), (0.0, 0.092)], segments=16)
    paint(hub, wear=_mix((0.1 * chip, 1.0), (0.7 * chip, nz)), grime=0.5)
    wood.append(hub)
    for z_, r_, w_ in ((-0.064, 0.0535, 0.022), (0.066, 0.055, 0.024)):
        metal.append(lathe("nave_band", [(r_, z_ - w_ / 2), (r_ + 0.005, z_ - w_ / 2 + 0.002), (r_ + 0.005, z_ + w_ / 2 - 0.002), (r_, z_ + w_ / 2)], segments=16))
    metal.append(lathe("axle_cap", [(0.03, 0.09), (0.033, 0.097), (0.03, 0.104), (0.019, 0.108), (0.016, 0.124), (0.0, 0.127)], segments=10))
    metal.append(box("linchpin", (0.052, 0.007, 0.007), loc=(0, 0, 0.116), bevel=0.0))
    for o in metal:
        paint(o, special=_mix((0.35, 1.0), (0.5, nz)), grime=0.4, wear=_mix((0.5, nz2)))
    tyre = ring_band("tyre", radius=fo, width=fw + 0.008, thick=ty, segments=36)
    # the tread is rolled bright; the flanks and the joint against the felloe rust
    paint(tyre, wear=lambda co, n: np.clip((rr(co) - (fo + ty * 0.3)) / (ty * 0.5), 0, 1),
          special=lambda co, n: np.clip(((fo + ty * 0.6) - rr(co)) / (ty * 0.5), 0, 1), grime=0.0)
    metal.append(tyre)
    nail = lathe("tyre_nail", [(0.0058, 0.0), (0.004, 0.0022), (0.0, 0.003)], segments=5)
    tn = instances_to_mesh("tyre_nails", nail, [trs(((R - 0.0005) * cos(radians(k * 360.0 / nsp)), (R - 0.0005) * sin(radians(k * 360.0 / nsp)), 0),
                                                    (0, 90, k * 360.0 / nsp)) for k in range(nsp)], keep_base=False)
    paint(tn, wear=1.0)
    metal.append(tn)
    for o in wood:
        assign(o, wood_m)
    for o in metal:
        assign(o, iron)
    return join(wood + metal, "wheel")


def _crate(cl, cw, ch, wood):
    """Slatted produce crate, base centre at the origin, long side along X (the bottom is a separate board)."""
    t = 0.009
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(_bx("crate_post", (0.022, 0.022, ch - 0.004), (sx * (cl / 2 - t - 0.011), sy * (cw / 2 - t - 0.011), ch / 2), grain='Z', bev=0.0))
    gap = 0.016
    sw = (ch - 0.012 - gap) / 2
    for k in range(2):
        z = 0.008 + sw / 2 + k * (sw + gap)
        for sy in (-1, 1):
            parts.append(_bx("crate_slat", (cl, t, sw), (0, sy * (cw / 2 - t / 2), z), grain='X', bev=0.0))
        for sx in (-1, 1):
            parts.append(_bx("crate_end", (t, cw - 2 * t, sw), (sx * (cl / 2 - t / 2), 0, z), grain='Y', bev=0.0))
    for o in parts:
        assign(o, wood)
    return join(parts, "crate_base"), 0.008 + sw + gap + sw / 2


def _liner(cl, cw, ch, rng, paper):
    """Paper lining of a crate: four strips up the inside of the walls, folded over the top slats, ragged hems."""
    t = 0.009
    parts = []
    for k, (length, half, along_x, sgn, nu) in enumerate(((cl - 0.05, cw / 2, True, -1, 6), (cl - 0.05, cw / 2, True, 1, 6),
                                                          (cw - 0.07, cl / 2, False, -1, 4), (cw - 0.07, cl / 2, False, 1, 4))):
        ph = rng.u(0, 6.28)
        yi, yo = half - t - 0.0015, half + 0.003

        def fn(u, v, length=length, sgn=sgn, along_x=along_x, ph=ph, yi=yi, yo=yo):
            a = (u - 0.5) * length
            drop = 0.034 + 0.013 * sin(7 * u + ph) + 0.008 * sin(23 * u + 2 * ph)
            c, z = ((yi, ch - 0.05), (yi + 0.001, ch + 0.0035), (yo, ch + 0.003),
                    (yo + 0.004 + 0.004 * sin(11 * u + ph), ch - drop))[int(round(v * 3))]
            return (a, sgn * c, z) if along_x else (sgn * c, a, z)

        parts.append(_face_to(surface(f"liner{k}", fn, nu=nu, nv=3), (0, 0, 1)))
    for o in parts:
        assign(o, paper)
    return join(parts, "liner_base")


def _fill(M, cl, cw, ch, r, squash, rng, top=0.4, tilt=10.0):
    """Produce (radius r) laid in a crate with world matrix M: (bottom layer matrices, heaped matrices, layer z)."""
    lo, hi = [], []
    z1 = max(ch - 0.45 * r, r + 0.014)
    ins = 0.016
    hx = cl / 2 - ins - r
    hy = cw / 2 - ins - r
    nx = max(1, int(2 * hx / (2.0 * r) + 1.3))
    ny = max(1, int(2 * hy / (1.85 * r) + 1.3))
    px_ = 2 * hx / max(nx - 1, 1) if nx > 1 else 0.0
    py_ = 2 * hy / max(ny - 1, 1) if ny > 1 else 0.0
    for j in range(ny):
        for i in range(nx):
            x = clamp((i - (nx - 1) / 2) * px_ + rng.jitter(0, 0.08 * r), -hx, hx)
            y = clamp((j - (ny - 1) / 2) * py_ + rng.jitter(0, 0.08 * r), -hy, hy)
            s = rng.u(0.86, 1.06)
            lo.append(M @ trs((x, y, z1 - (1 - s) * r * 0.6), (rng.jitter(0, tilt), rng.jitter(0, tilt), rng.u(0, 360)), (s, s, s * squash)))
    for j in range(ny - 1):
        for i in range(nx - 1):
            if rng.chance(top):
                x = (i + 0.5 - (nx - 1) / 2) * px_
                y = (j + 0.5 - (ny - 1) / 2) * py_
                s = rng.u(0.88, 1.04)
                hi.append(M @ trs((x + rng.jitter(0, 0.08 * r), y + rng.jitter(0, 0.08 * r), z1 + 1.3 * r * squash),
                                  (rng.u(-40, 40), rng.u(-40, 40), rng.u(0, 360)), (s, s, s * squash)))
    return lo, hi, z1


def _round(name, prof, r, seg, m):
    """Smooth-shaded produce body from a unit profile [(radius, z)] scaled by r."""
    o = lathe(name, [(a * r, b * r) for a, b in prof], segments=seg, smooth_angle=89)
    assign(o, m)
    return o


def _sack(name, R, H, rng, flat=0.55, slump=0.0, neck=False, flop=0.0):
    """Burlap sack: a stuffed pillow along local Z (0..H), wide along X.  Sewn top with two 'ears', or (neck)
    a gathered, tied mouth with a floppy frill; slump = the fill settles to the bottom, flop = the slack
    top folds over towards +Y.  Returns (object, local centre of the tie)."""
    ph = [rng.u(0, 6.28) for _ in range(5)]

    def centre(v):
        q = smoothstep(0.5, 1.0, v)
        return (0.02 * slump * H * sin(pi * v) + flop * H * 0.45 * q * q, H * v - flop * H * 0.3 * q * q * q)

    def fn(u, v):
        a = 2 * pi * u
        e = abs(2 * v - 1)
        prof = (1 - e ** 3.0) ** 0.5
        w = R * (0.8 + 0.2 * (1 - e ** 4) ** 0.5 + 0.07 * e ** 8)
        t = R * flat * prof * (1 + slump * (0.5 - v)) + 0.003
        if neck:
            g = smoothstep(0.64, 0.84, v)
            fr = smoothstep(0.86, 1.0, v)
            w = lerp(w, R * (0.2 + 0.24 * fr), g)
            t = lerp(t, R * 0.17 * (1 - 0.8 * fr), g)
        ca, sa = cos(a), sin(a)
        x = w * (abs(ca) ** 0.7) * (1 if ca >= 0 else -1)
        lum = 1 + prof * (0.07 * sin(3 * a + ph[0] + 4 * v) + 0.06 * sin(8 * v + ph[1]) * cos(2 * a + ph[2]))
        yc, zc = centre(v)
        z = zc + 0.014 * sin(4 * a + ph[3]) * e ** 3 - slump * 0.05 * H * sin(pi * v) * (1 - abs(ca))
        return (x * lum, t * sa * lum + yc, z)

    yc, zc = centre(0.85)
    return surface(name, fn, nu=14, nv=12, closed_u=True), (0.0, yc, zc)


def _lantern(hx, hy, hz, metal, glass, intensity):
    """Small hurricane lamp hanging from (hx, hy, hz); returns the objects."""
    zb = hz - 0.34
    objs = []
    objs.append(lathe("lant_font", [(0.03, 0), (0.052, 0), (0.058, 0.005), (0.058, 0.028), (0.05, 0.044), (0.03, 0.05), (0.027, 0.058),
                                    (0.038, 0.064), (0.04, 0.078), (0.034, 0.084)], segments=12, loc=(hx, hy, zb)))
    objs.append(lathe("lant_cap", [(0.03, 0.196), (0.041, 0.2), (0.043, 0.208), (0.03, 0.22), (0.022, 0.24), (0.03, 0.246),
                                   (0.034, 0.253), (0.0, 0.262)], segments=12, loc=(hx, hy, zb)))
    for sx in (-1, 1):
        objs.append(sweep("lant_tube", [(hx + sx * 0.055, hy, zb + 0.03), (hx + sx * 0.068, hy, zb + 0.08), (hx + sx * 0.068, hy, zb + 0.16),
                                        (hx + sx * 0.04, hy, zb + 0.212)], 0.004, sides=5, up=(0, 1, 0), smooth_path=False))
    # tin smoke hood / reflector: keeps the light off whatever the lamp hangs from, throws it down on the goods
    objs.append(lathe("lant_hood", [(0.098, 0.236), (0.021, 0.272), (0.021, 0.2695), (0.096, 0.2335), (0.098, 0.236)], segments=14, loc=(hx, hy, zb)))
    objs.append(sweep("lant_bail", [(hx - 0.02, hy, zb + 0.268), (hx - 0.012, hy, zb + 0.31), (hx, hy, zb + 0.34), (hx + 0.012, hy, zb + 0.31),
                                    (hx + 0.02, hy, zb + 0.268)], 0.0024, sides=4, up=(0, 1, 0), smooth_path=False))
    for o in objs:
        assign(o, metal)
    globe = lathe("lant_globe", [(0.033, 0.082), (0.048, 0.1), (0.055, 0.132), (0.05, 0.165), (0.036, 0.19), (0.03, 0.198)],
                  segments=12, loc=(hx, hy, zb))
    assign(globe, glass)
    objs.append(globe)
    # the flame sits high in the chimney; the lamp's own font, tubes, hood and bail (all within 0.2 m) cast no shadow
    light("lantern", (hx, hy, zb + 0.15), color="#ffbf6a", intensity=intensity, range=6.5, flicker=0.22, shadow_near=0.21)
    return objs


# ------------------------------------------------------------------ build
def build(p, rng):
    L, W = p.length, p.width
    R = p.wheel_diameter / 2
    nsp = int(p.spokes)
    k_wear = p.wear
    seed = p.seed
    HL = 0.6                                   # handle length beyond the bed
    cx = -0.36                                 # bed centre: keeps the whole footprint about the origin
    xb0, xb1 = cx - L / 2, cx + L / 2
    ax = cx - L * 0.06                         # axle a little ahead of the bed centre: weight on the legs
    SW, SH = 0.055, 0.075                      # shaft section
    ys = W / 2 - SW / 2
    z_sb = R + 0.24 + 0.6 * max(0.0, R - 0.42)  # shaft underside (big wheels: longer spring hangers, body rides a little higher)
    zsc = z_sb + SH / 2
    z_st = z_sb + SH
    zd = z_st + 0.045 + 0.022                  # deck top
    yw = W / 2 + 0.125                         # wheel plane
    f = 8.0                                    # side board flare (deg)
    sf, cf = sin(radians(f)), cos(radians(f))
    bh = 0.19                                  # side board height
    zb0 = zd - 0.03                            # side board bottom edge
    flowers = p.load == "flowers"

    body = mat("wood_painted", p.paint)
    trim = mat("wood_painted", p.trim)
    pine = mat("wood_oak", "#d9bd8e")                          # raw pale deal: crates, deck, sticks
    iron = mat("iron_wrought")
    cream = mat("paper", "#d8c690")                            # signwriter's cream, crate paper, tickets, cord
    deck_m = body if flowers else pine                         # (material budget: the flower cart has a painted bed)

    kc = 0.55 + 0.5 * k_wear                                   # paint chipping gain
    lo_ = 0.7 - 0.22 * k_wear
    nzs = noise_mask(0.055, seed=seed + 13, lo=lo_, hi=lo_ + 0.2)        # small chips
    nzl = noise_mask(0.22, seed=seed + 14, lo=lo_ + 0.04, hi=lo_ + 0.3)  # worn patches
    nzg = noise_mask(0.3, seed=seed + 15, lo=0.25, hi=0.8)     # dirt

    ironwork = []          # everything forged, painted with rust at the end
    bolts = []             # dome bolt matrices
    nails = []

    # ---------------------------------------------------------------- wheels, axle, springs
    wheel = _wheel(R, nsp, trim, iron, k_wear, seed)
    wheel_b = duplicate(wheel, "wheel_b")
    r_w = rng.fork("wheels")
    place(wheel, (ax, -yw, R), (90 + r_w.jitter(0, 0.8), r_w.u(0, 360), 0))
    place(wheel_b, (ax, yw, R), (-90 + r_w.jitter(0, 0.8), r_w.u(0, 360), 0))
    for w in (wheel, wheel_b):
        paint(w, grime=mask_mul(height_ramp(0.0, R * 0.9), 0.6))

    axle = box("axle", (0.036, 2 * yw - 0.12, 0.036), loc=(ax, 0, R), bevel=0.004, segments=1)
    ironwork.append(axle)
    leaf_prof = [(-0.021, -0.003), (0.021, -0.003), (0.021, 0.003), (-0.021, 0.003)]
    h0 = min(0.36, L * 0.25)
    camber = 0.085
    zc = R + 0.043
    for s in (-1, 1):
        y = s * ys
        for k in range(4):
            hk = h0 * (1.0 - 0.23 * k)
            pts = [(ax + hk * t, y, zc - k * 0.0072 + camber * (hk * t / h0) ** 2) for t in np.linspace(-1, 1, 7 - 2 * (k // 2))]
            ironwork.append(sweep(f"leaf{k}", pts, leaf_prof, up=(0, 0, 1), smooth_path=False))
        ironwork.append(box("spring_seat", (0.075, 0.062, 0.012), loc=(ax, y, R - 0.024), bevel=0.0))
        ironwork.append(box("spring_plate", (0.075, 0.062, 0.008), loc=(ax, y, zc + 0.008), bevel=0.0))
        for dx in (-0.026, 0.026):
            ironwork.append(sweep("u_bolt", [(ax + dx, y - 0.027, zc + 0.022), (ax + dx, y - 0.027, R - 0.03), (ax + dx, y + 0.027, R - 0.03),
                                             (ax + dx, y + 0.027, zc + 0.022)], 0.0045, sides=5, smooth_path=False, up=(1, 0, 0)))
        for e in (-1, 1):                                    # spring eyes + drop hangers from the shaft
            xe, ze_ = ax + e * h0, zc + camber
            ironwork.append(cylinder("spring_eye", 0.011, 0.05, loc=(xe, y, ze_ + 0.006), rot=(90, 0, 0), verts=8))
            for dy in (-0.028, 0.028):
                ironwork.append(box("hanger", (0.03, 0.012, z_sb - ze_ + 0.02), loc=(xe, y + dy, (z_sb + ze_) / 2 + 0.004), bevel=0.003 if dy * s > 0 else 0.0, segments=1))
            ironwork.append(box("hanger_plate", (0.11, SW + 0.004, 0.007), loc=(xe, y, z_sb - 0.003), bevel=0.0))

    # ---------------------------------------------------------------- shafts running into handles
    painted = []           # body-coloured timber
    grips = []
    r_f = rng.fork("frame")
    yh = min(ys, 0.3)
    for s in (-1, 1):
        y0 = s * ys
        sh = _bx(f"shaft{s}", (L + 0.105, SW, SH), (cx - 0.0325, y0, zsc), bev=0.007)
        densify(sh, 0.22)
        painted.append(sh)
        ym = lerp(y0, s * yh, 0.55)
        path = [(xb1 - 0.03, y0, zsc), (xb1 + 0.12, y0, zsc), (xb1 + HL * 0.55, ym, zsc + 0.016), (xb1 + HL, s * yh, zsc + 0.07)]
        hd = sweep(f"handle{s}", path, rounded_rect(SW, SH, 0.012, 2), up=(0, 0, 1), step=0.09,
                   scale_fn=lambda t: 1.0 - 0.38 * smoothstep(0.12, 1.0, t))
        painted.append(hd)
        ddx, ddy, ddz = HL * 0.45, s * yh - ym, 0.054
        yaw = degrees(atan2(ddy, ddx))
        pitch = degrees(atan2(ddz, hypot(ddx, ddy)))
        grip = lathe(f"grip{s}", [(0.022, -0.014), (0.022, 0.0), (0.0245, 0.006), (0.021, 0.016), (0.0185, 0.05), (0.019, 0.1),
                                  (0.0225, 0.118), (0.024, 0.128), (0.019, 0.14), (0.0, 0.143)], segments=8,
                     loc=(xb1 + HL, s * yh, zsc + 0.07), rot=(0, 90 - pitch, yaw))
        grips.append(grip)
        ironwork.append(ring_band("ferrule", radius=0.0262, width=0.03, thick=0.003, loc=(xb1 + HL - 0.008, s * yh, zsc + 0.069),
                                  rot=(0, 90 - pitch, yaw), segments=10))

    # cross bearers on the shafts, deck planks on the bearers
    nb = max(3, int(round(L / 0.42)) + 1)
    bxs = [xb0 + 0.04 + (L - 0.08) * i / (nb - 1) for i in range(nb)]
    for i, x in enumerate(bxs):
        painted.append(_bx(f"bearer{i}", (0.055, W, 0.045), (x, 0, z_st + 0.0225), bev=0.004, grain='Y'))
    deck = []
    nd = max(4, int(round(W / 0.125)))
    pw = W / nd
    r_d = rng.fork("deck")
    for i in range(nd):
        y = -W / 2 + pw * (i + 0.5)
        dl = L - r_d.u(0.0, 0.012)
        pl = _bx(f"deck{i}", (dl, pw - r_d.u(0.003, 0.006), 0.022), (cx + r_d.jitter(0, 0.003), y, zd - 0.011 + r_d.jitter(0, 0.0012)), bev=0.003)
        densify(pl, 0.8)
        deck.append(pl)
        for x in (bxs[0], bxs[-1]):
            for dy in (-pw * 0.25, pw * 0.25):
                nails.append(trs((x + r_d.jitter(0, 0.006), y + dy, zd - 0.0005)))
    for o in deck:
        assign(o, deck_m)
        paint(o, grime=_mix((0.2, 1.0), (0.5, nzg)), wear=_mix((0.3, 1.0), (0.7, nzl), (0.5, nzs)) if flowers else _mix((0.3, nzl)))

    # ---------------------------------------------------------------- flared side boards, capping, end boards
    boards, caps = [], []

    def side_pt(s, h, out=0.0):
        """Point on the side board centre plane at height h along the board, pushed `out` along its outer normal."""
        return (s * (W / 2 + 0.012 + h * sf + out * cf), zb0 + h * cf - out * sf)

    for s in (-1, 1):
        yc, zc_b = side_pt(s, bh / 2)
        sb = _bx(f"side{s}", (L + 0.06, bh, 0.022), (cx, yc, zc_b), rot=(90 - s * f, 0, 0), bev=0.003)
        densify(sb, 0.11)
        boards.append(sb)
        yc, zc_c = side_pt(s, bh + 0.011)
        cp = _bx(f"cap{s}", (L + 0.11, 0.044, 0.024), (cx, yc + s * 0.004, zc_c), rot=(-s * f, 0, 0), bev=0.007)
        densify(cp, 0.16)
        caps.append(cp)

    def endboard(name, h, x):
        o = _bx(name, (h, 1.0, 0.022), bev=0.003, grain='Y')
        densify(o, 0.19)
        wb = W + 0.02
        wt = W + 0.02 + 2 * h * sf / cf
        transform_verts(o, lambda co: Vector((co.x, co.y * lerp(wb, wt, 0.5 - co.x / h), co.z)))
        place(o, (x, 0, zb0 + h / 2), (0, 90, 0))
        return o

    boards.append(endboard("end_front", bh * cf, xb0 - 0.011))
    h_tail = bh * 0.58
    boards.append(endboard("end_tail", h_tail, xb1 + 0.011))
    cp = _bx("cap_front", (0.044, W + 0.02 + 2 * bh * sf + 0.07, 0.024), (xb0 - 0.011, 0, zb0 + bh * cf + 0.012), bev=0.007, grain='Y')
    densify(cp, 0.16)
    caps.append(cp)

    # forged stays from the shaft up the outside of the board; short straps near the ends
    stay_prof = rounded_rect(0.006, 0.03, 0.0015, 1)
    fr_stays = [0.2, 0.8]
    for s in (-1, 1):
        brot = (-s * (90 + f), 0, 0)
        for fr in fr_stays:
            x = xb0 + L * fr
            c_, d_ = side_pt(s, 0.012, 0.0145), side_pt(s, bh * 0.94, 0.0145)
            path = [(x, s * (W / 2 + 0.0035), z_sb + 0.015), (x, s * (W / 2 + 0.0035), zb0 - 0.05), (x, c_[0], c_[1]), (x, d_[0], d_[1])]
            ironwork.append(sweep("stay", path, stay_prof, up=(1, 0, 0), smooth_path=False))
            bolts.append(trs((x, s * (W / 2 + 0.006), z_sb + 0.04), (-s * 90, 0, 0)))
            for hh in (0.25, 0.78):
                q = side_pt(s, bh * hh, 0.017)
                bolts.append(trs((x, q[0], q[1]), brot))
        for x in (xb0 + 0.012, xb1 - 0.012):
            c_, d_ = side_pt(s, 0.008, 0.0145), side_pt(s, bh * 0.96, 0.0145)
            ironwork.append(sweep("strap", [(x, c_[0], c_[1]), (x, d_[0], d_[1])], stay_prof, up=(1, 0, 0), smooth_path=False))
            for hh in (0.2, 0.8):
                q = side_pt(s, bh * hh, 0.017)
                bolts.append(trs((x, q[0], q[1]), brot))

    # lettering on the customer side
    if p.lettering.strip():
        t_lo = max(0.04, 2 * R - zb0 - 0.02)                    # keep the name clear of the wheel top
        t_hi = bh - 0.022
        ts = min(0.1, t_hi - t_lo, (L * 0.46) / max(len(p.lettering), 4) * 1.55)
        q = side_pt(-1, (t_lo + t_hi) / 2, 0.011)
        tx = text("lettering", p.lettering, size=ts, depth=0.003, loc=(xb0 + L * 0.5, q[0], q[1]), rot=(90 + f, 0, 0), font="serif")
        _lite_text(tx, sides=False)
        assign(tx, cream)
        paint(tx, grime=_mix((0.5 * (0.3 + k_wear), noise_mask(0.05, seed=seed + 19, lo=0.4, hi=0.8))))

    # ---------------------------------------------------------------- prop legs
    xl = xb1 - 0.14
    z_leg = z_st - 0.004
    leg_prof = rounded_rect(0.045, 0.05, 0.006, 2)

    def leg_pt(s, t):
        return (xl + 0.065 * t, s * (W / 2 + 0.0235 + 0.03 * t), lerp(z_leg, 0.012, t))

    for s in (-1, 1):
        lg = sweep(f"leg{s}", [leg_pt(s, t) for t in (0, 0.25, 0.5, 0.75, 1)], leg_prof, up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 1.0 - 0.18 * t)
        painted.append(lg)
        ft = leg_pt(s, 1)
        ironwork.append(box("leg_shoe", (0.05, 0.048, 0.045), loc=(ft[0], ft[1], 0.0225), bevel=0.004, segments=1))
        for zz in (z_st - 0.02, z_st - 0.055):
            bolts.append(trs((xl + 0.002, s * (W / 2 + 0.047), zz), (-s * 90, 0, 0), 1.25))
        a_ = leg_pt(s, 0.52)
        ironwork.append(sweep("leg_brace", [(a_[0] - 0.024, a_[1], a_[2] - 0.03), (a_[0] - 0.026, a_[1], a_[2]), (xl - 0.3, s * (W / 2 - 0.022), z_sb - 0.004),
                                            (xl - 0.38, s * (W / 2 - 0.022), z_sb - 0.004)], rounded_rect(0.024, 0.006, 0.0015, 1), up=(0, 0, 1), smooth_path=False))
    m_ = leg_pt(1, 0.62)
    st = cylinder("leg_stretcher", 0.015, 2 * m_[1] + 0.01, loc=(m_[0], 0, m_[2]), rot=(90, 0, 0), verts=8)
    uv_grain(st, 'Z')
    smooth(st, 60)
    painted.append(st)

    # ---------------------------------------------------------------- canopy
    hw = W / 2 - 0.0475                               # pole line
    ze = zd + p.canopy_height
    hang = None                                       # lantern suspension point
    cords = []                                        # cotton cord / lashings (cream)
    frame = []
    if p.canopy:
        canvas = mat("canvas", p.canopy_color, stripe=0.11, tint2=p.stripe_color)
        r_c = rng.fork("canopy")
        px = (xb0 - 0.052, xb1 + 0.052)                # poles stand on the shafts, outside the end boards
        span = px[1] - px[0]
        lash = lathe("lashing", [(0.0265, 0.0), (0.0315, 0.006), (0.0265, 0.012), (0.0315, 0.018), (0.0265, 0.024)], segments=6)
        lash_m = []
        strap_prof = [(-0.0025, -0.014), (0.0025, -0.014), (0.0025, 0.014), (-0.0025, 0.014)]
        for ie, x in enumerate(px):
            e = -1 if ie == 0 else 1                  # outward along X
            xf = (xb0 - 0.022 if ie == 0 else xb1 + 0.022) + e * 0.003
            zs_ = (zb0 + 0.045, zb0 + bh * cf - 0.04) if ie == 0 else (zb0 + h_tail * 0.5,)
            for s in (-1, 1):
                y = s * hw
                lean = r_c.jitter(0, 0.35)
                frame.append(_bx("pole", (0.036, 0.036, ze - z_st), (x, y, (ze + z_st) / 2), rot=(lean, 0, 0), bev=0.006, grain='Z'))
                # iron shoe on the shaft and U-straps bolted to the end board
                ironwork.append(box("pole_shoe", (0.05, 0.058, 0.04), loc=(x, y, z_st + 0.02), bevel=0.003, segments=1))
                bolts.append(trs((x, y + s * 0.029, z_st + 0.02), (-s * 90, 0, 0)))
                for zz in zs_:
                    ironwork.append(sweep("pole_strap", [(xf, y - 0.05, zz), (xf, y - 0.0215, zz), (x + e * 0.0215, y - 0.0215, zz), (x + e * 0.0215, y + 0.0215, zz),
                                                         (xf, y + 0.0215, zz), (xf, y + 0.05, zz)], strap_prof, up=(0, 0, 1), smooth_path=False))
                    for dy in (-0.038, 0.038):
                        bolts.append(trs((xf + e * 0.002, y + dy, zz), (0, e * 90, 0), 0.85))
                ironwork.append(sweep("pole_brace", [(x - e * 0.004, y, ze - 0.22), (x - e * 0.2, y, ze + 0.002)], rounded_rect(0.02, 0.005, 0.001, 1),
                                      up=(0, 1, 0), smooth_path=False))
                lash_m.append(trs((x, y, ze - 0.075)))
        rise = hw * 0.5
        slope = rise / hw
        zr = ze + 0.045 + rise
        th = degrees(atan2(rise, hw))
        for s in (-1, 1):
            frame.append(_bx("plate", (span + 0.16, 0.034, 0.04), (cx, s * hw, ze + 0.02), bev=0.005))
        for ie, x in enumerate(px):
            e = -1 if ie == 0 else 1
            frame.append(_bx("tie", (0.03, 2 * hw + 0.1, 0.036), (x - e * 0.034, 0, ze - 0.02), bev=0.004, grain='Y'))
            frame.append(_bx("king", (0.03, 0.03, rise + 0.03), (x - e * 0.034, 0, ze + rise / 2 + 0.012), bev=0.003, grain='Z'))
        nr = max(3, int(round(span / 0.56)) + 1)
        rxs = [px[0] + span * i / (nr - 1) for i in range(nr)]
        for x in rxs:
            for s in (-1, 1):
                frame.append(_bx("rafter", (0.028, hypot(hw, rise) + 0.05, 0.03), (x, s * hw / 2, ze + 0.045 + rise / 2), rot=(-s * th, 0, 0), bev=0.003, grain='Y'))
        frame.append(_bx("ridge", (span + 0.24, 0.03, 0.036), (cx, 0, zr - 0.006), bev=0.004))
        for e, x in ((-1, px[0] - 0.12), (1, px[1] + 0.12)):
            frame.append(lathe("finial", [(0.012, 0.0), (0.016, 0.006), (0.012, 0.012), (0.022, 0.024), (0.024, 0.036), (0.016, 0.05), (0.0, 0.056)], segments=8,
                               loc=(x, 0, zr - 0.006), rot=(0, e * 90, 0)))

        oe, oy = 0.03, 0.17
        ph = [r_c.u(0, 6.28) for _ in range(4)]
        sag = r_c.u(0.02, 0.028)

        def roof_z(x, y):
            ay = abs(y)
            z = zr + 0.024 - slope * sqrt(y * y + 0.018 ** 2)
            bay = sin(pi * clamp((x - px[0]) / span, 0.0, 1.0) * (nr - 1)) ** 2
            if ay <= hw:
                z -= sag * bay * sin(pi * ay / hw) ** 0.8                               # slack cloth between the rafters
            else:
                d = ay - hw
                z -= 0.22 * d * d / oy + 0.6 * sag * bay * d / oy                       # droop past the plate, tied down at the rafter ends
            return z + 0.005 * sin(17 * x + ph[0]) * sin(11 * y + ph[1]) + 0.003 * sin(41 * x + 23 * y + ph[3])

        xc0, xc1 = px[0] - oe, px[1] + oe
        Lc = xc1 - xc0
        nu_c = 8 * (nr - 1) + 2
        nv_c = 12

        def xu(u):
            i = u * nu_c
            if i <= 1:
                return xc0 + oe * i
            if i >= nu_c - 1:
                return px[1] + oe * (i - (nu_c - 1))
            return px[0] + span * (i - 1) / (nu_c - 2)

        def yv(v):
            j = v * nv_c
            if j <= 2:
                return -(hw + oy) + oy * j / 2
            if j >= nv_c - 2:
                return hw + oy * (j - (nv_c - 2)) / 2
            return -hw + 2 * hw * (j - 2) / (nv_c - 4)

        CT = 0.003
        cv = surface("canvas", lambda u, v: (xu(u), yv(v), roof_z(xu(u), yv(v))), nu=nu_c, nv=nv_c, thickness=CT)
        uv_planar(cv, (1, 0, 0), (0, 1, 0))
        cloth = [cv]
        ns = max(4, int(round(Lc / 0.19)))

        def lobe(t):
            fr = (t % 1.0) * 2 - 1
            return sqrt(max(0.0, 1 - fr * fr))

        for s in (-1, 1):                               # scalloped valance, gathered into soft folds
            def vfn(u, v, s=s):
                x = xc0 + u * Lc
                y = s * (hw + oy)
                lb = lobe(u * ns)
                d = v * (0.08 + 0.065 * lb)
                fold = 0.013 * sin(u * ns * 2 * pi * 1.5 + ph[2] + s) + 0.009 * sin(u * ns * 2 * pi * 0.5 + ph[1])
                return (x, y + s * (0.006 * v + (v ** 0.8) * (0.008 * lb + fold)), roof_z(x, y) - d)
            vl = surface(f"valance{s}", vfn, nu=ns * 4, nv=2, thickness=CT)
            uv_planar(vl, (1, 0, 0), (0, 0, 1))
            cloth.append(vl)
        for e, xe in ((-1, xc0), (1, xc1)):              # turned-down hem at the gables
            hm = surface(f"hem{e}", lambda u, v, e=e, xe=xe: (xe + e * 0.004 * v, yv(u), roof_z(xe, yv(u)) - 0.036 * v), nu=nv_c, nv=1, thickness=CT)
            uv_planar(hm, (1, 0, 0), (0, 1, 0))
            cloth.append(hm)
        for o in cloth:
            assign(o, canvas)
            paint(o, special=mask_mul(facing((0, 0, 1)), 0.25 + 0.5 * k_wear),
                  grime=_mix((0.3 + 0.5 * k_wear, noise_mask(0.3, seed=seed + 23, lo=0.4, hi=0.85)), (0.45, height_ramp(ze - 0.2, ze - 0.02))))
        # cord: the canvas laced round the end rafters, guys from the eave corners to the poles, lashings at the plates
        sth, cth = sin(radians(th)), cos(radians(th))
        for e, x in ((-1, px[0]), (1, px[1])):
            for s in (-1, 1):
                nt = max(3, int(hypot(hw, rise) / 0.085))
                pts = []
                for i in range(nt * 3 + 1):
                    q = 0.06 + 0.9 * i / (nt * 3)
                    a = 2 * pi * i / 3
                    yy = s * hw * q
                    zz = roof_z(x, yy) - 0.021
                    rad = 0.027
                    pts.append((x + e * (0.011 + rad * cos(a)), yy + rad * sin(a) * s * sth, zz + rad * sin(a) * cth))
                cords.append(sweep("lacing", pts, 0.0032, sides=3, smooth_path=False, cap=False))
                yc_, zc_ = s * (hw + oy - 0.01), roof_z(x, s * (hw + oy)) - 0.012
                cords.append(sweep("guy", [(x + e * 0.02, yc_, zc_), (x, s * (hw + 0.022), ze - 0.4), (x + e * 0.006, s * (hw + 0.026), ze - 0.52)], 0.0032, sides=3, smooth_path=False))
                lash_m.append(trs((x, s * hw, ze - 0.415), (0, 0, 0), (0.92, 0.92, 0.6)))
        cords.append(instances_to_mesh("lashings", lash, lash_m, keep_base=False))
        for o in frame:
            assign(o, trim)
            paint(o, wear=_mix((0.05, 1.0), (0.8 * kc, nzs), (0.3 * kc, height_ramp(zd + 0.1, zd + 0.7))), grime=_mix((0.35, nzg)), special=_mix((0.5, nzg)))
        if p.lantern:
            mids = [(rxs[i] + rxs[i + 1]) / 2 for i in range(nr - 1)]
            lx = min(mids, key=lambda v: abs(v - (cx - 0.12 * L)))
            yo = -hw - 0.021
            ironwork.append(sweep("lantern_bracket", [(lx, -hw + 0.012, ze + 0.046), (lx, yo, ze + 0.046), (lx, yo, ze - 0.26), (lx, yo - 0.4, ze - 0.26), (lx, yo - 0.418, ze - 0.246),
                                                      (lx, yo - 0.41, ze - 0.228)], rounded_rect(0.02, 0.007, 0.0015, 1), up=(1, 0, 0), smooth_path=False))
            ironwork.append(sweep("lantern_brace", [(lx, yo - 0.004, ze - 0.08), (lx, yo - 0.07, ze - 0.2), (lx, yo - 0.26, ze - 0.252)], 0.0055, sides=5, up=(1, 0, 0), step=0.05))
            bolts.append(trs((lx, yo - 0.0035, ze + 0.02), (90, 0, 0)))
            hang = (lx, yo - 0.39, ze - 0.264)
    elif p.lantern:
        # forged crook bolted to the shaft, up the outside of the front side board, curling out over the customers
        kx = xb0 + L * 0.74
        rr = 0.12
        Ht = bh + 0.8
        dxk, dyk = -cos(radians(55)), -sin(radians(55))            # horizontal direction of the curl
        q0, q1, q2 = side_pt(-1, 0.012, 0.021), side_pt(-1, bh * 0.6, 0.021), side_pt(-1, bh + 0.02, 0.04)
        path = [(kx, -(W / 2 + 0.008), z_sb + 0.015), (kx, -(W / 2 + 0.008), zb0 - 0.05), (kx, q0[0], q0[1]), (kx, q1[0], q1[1]), (kx, q2[0], q2[1])]
        qa = side_pt(-1, Ht - 0.1, 0.04)
        path.append((kx, qa[0], qa[1]))
        qt = side_pt(-1, Ht, 0.04)

        def crook_pt(d, dz):
            return (kx + dxk * d, qt[0] + dyk * d, qt[1] + dz)

        for a in range(20, 161, 20):
            path.append(crook_pt(rr * (1 - cos(radians(a))), rr * sin(radians(a))))
        for d, dz in ((2 * rr - 0.003, 0.01), (2 * rr + 0.022, -0.012), (2 * rr + 0.05, 0.012)):
            path.append(crook_pt(d, dz))
        ironwork.append(sweep("crook", path, 0.008, sides=7, up=(0, 1, 0), step=0.05, scale_fn=lambda t: 1.0 - 0.45 * smoothstep(0.8, 1.0, t)))
        for hh in (0.25, 0.8):
            qq = side_pt(-1, bh * hh, 0.025)
            ironwork.append(box("crook_staple", (0.05, 0.022, 0.02), loc=(kx, qq[0], qq[1]), rot=(f, 0, 0), bevel=0.003, segments=1))
        bolts.append(trs((kx, -(W / 2 + 0.014), z_sb + 0.04), (90, 0, 0), 1.2))
        hang = crook_pt(2 * rr + 0.022, -0.008)

    if hang is not None:
        glass = mat("glass_frosted", "#ffb962", glow=1.3)
        _lantern(hang[0], hang[1], hang[2], iron, glass, p.intensity)

    # ---------------------------------------------------------------- load
    r_l = rng.fork("load")
    x0, x1 = xb0 + 0.065, xb1 - 0.06
    stencil = p.stencil.strip()

    def crate_set(cl, cw, ch, mats_, upturned=()):
        """Stamp crates (+ stencilled marks, + a bottom board for the upturned ones)."""
        crate, z_mark = _crate(cl, cw, ch, pine)
        o = instances_to_mesh("crates", crate, mats_, keep_base=False)
        paint(o, grime=_mix((0.5, nzg), (0.25 * k_wear, 1.0)))
        if stencil:
            sz = min(0.036, (cl - 0.1) / max(len(stencil), 3) * 1.5)
            tx = text("stencil", stencil, size=sz, depth=0.0008, loc=(0, -cw / 2 - 0.0002, z_mark), font="sans_narrow")
            _lite_text(tx, sides=False, ang=14.0)
            apply_transform(tx)
            assign(tx, body)
            paint(tx, grime=1.0)
            instances_to_mesh("stencils", tx, mats_, keep_base=False, uv_jitter=False)
        if upturned:
            fl = _bx("crate_bottom", (cl - 0.004, cw - 0.004, 0.008), bev=0.0)
            assign(fl, pine)
            instances_to_mesh("crate_bottoms", fl, [M @ trs((0, 0, 0.004)) for M in upturned], keep_base=False)

    if p.load == "fruit":
        ncol = max(2, int((x1 - x0) / 0.46))
        pitch = (x1 - x0) / ncol
        cl = min(0.5, pitch - 0.022)
        cw = min(W / 2 - 0.06, 0.28)
        ch = 0.15
        tilt = 15.0
        yb = W / 4 - 0.004
        rz = 0.10
        xr1 = x0 + pitch * (ncol - 1)                                  # the back row is raised and tilted towards the customer
        rs_ = [_bx("riser_front", (xr1 - x0 - 0.05, 0.024, rz), ((x0 + xr1) / 2, yb - cw / 2 + 0.05, zd + rz / 2), bev=0.003),
               _bx("riser_back", (xr1 - x0 - 0.05, 0.024, rz + (cw - 0.1) * sin(radians(tilt))), ((x0 + xr1) / 2, yb + cw / 2 - 0.05, zd + (rz + (cw - 0.1) * sin(radians(tilt))) / 2), bev=0.003)]
        for o in rs_:
            assign(o, pine)
            paint(o, grime=0.5)
        green = mat("leaf", "#7d9440")
        HALF = [(1.0, 0.0), (0.9, 0.48), (0.5, 0.86), (0.13, 0.78), (0.0, 0.74)]
        FULL = [(0.7, -0.7)] + HALF
        kinds = [
            dict(name="apples", r=0.039, squash=0.9, m=mat("plain", "#8e1d1b", rough=0.55), seg=8, top=0.3, tilt=10, lo=HALF, hi=FULL, stalk=0.74),
            dict(name="oranges", r=0.042, squash=0.97, m=mat("leather", "#d2570a", scale=2.5, rough=1.5), seg=9, top=0.3, tilt=9,
                 lo=[(1.0, 0.0), (0.82, 0.57), (0.42, 0.907), (0.0, 1.0)], hi=[(0.72, -0.69), (1.0, 0.0), (0.82, 0.57), (0.42, 0.907), (0.0, 1.0)], stalk=None),
            dict(name="pears", r=0.039, squash=1.0, m=green, seg=8, top=0.0, tilt=19,
                 lo=[(1.0, 0.0), (0.97, 0.42), (0.72, 0.98), (0.47, 1.4), (0.34, 1.7), (0.0, 1.84)], hi=None, stalk=1.8),
        ]
        prices = [s_.strip() for s_ in p.prices.split(",") if s_.strip()]
        off = r_l.pick([0, 1, 2, 3, 4])
        cm, fm, tick = [], [], {}
        prod = {k["name"]: ([], []) for k in kinds}
        leaf_m, calyx_m = [], []
        spare = None
        for row in range(2):
            for c in range(ncol):
                x = x0 + pitch * (c + 0.5) + r_l.jitter(0, 0.006)
                if row == 1 and c == ncol - 1:                      # the seller's corner: an upturned crate as a counter
                    spare = trs((x, yb - 0.01, zd + ch + 0.001), (180, 0, r_l.jitter(0, 4)))
                    continue
                ctilt = 0.0
                if row == 0:
                    M = trs((x, -W / 4 + 0.004 + r_l.jitter(0, 0.004), zd + 0.001), (0, 0, r_l.jitter(0, 2.5)))
                else:
                    ctilt = tilt + r_l.jitter(0, 1.5)
                    M = trs((x, yb, zd + rz + (cw / 2 - 0.05) * sin(radians(tilt)) + 0.004), (ctilt, 0, r_l.jitter(0, 2.0)))
                cm.append(M)
                ki = (0, 1, 2, 0, 1)[(len(cm) - 1 + off) % 5]
                kd = kinds[ki]
                r = kd["r"]
                lo, hi, z1 = _fill(M, cl, cw, ch, r, kd["squash"], r_l, kd["top"], kd["tilt"])
                prod[kd["name"]][0].extend(lo)
                prod[kd["name"]][1].extend(hi)
                fm.append(M @ trs((0, 0, z1 - 0.3 * r)))
                for Mf in lo + hi:
                    if kd["stalk"] is None:                         # orange: green calyx button anywhere on the upper side
                        calyx_m.append(Mf @ trs((0, 0, 0), (0, r_l.u(0, 62), r_l.u(0, 360))) @ trs((0, 0, r * 0.985)))
                    elif r_l.chance(0.16):                          # the odd leaf left on the stalk
                        leaf_m.append(Mf @ trs((0, 0, kd["stalk"] * r + 0.012), (r_l.jitter(0, 25), -r_l.u(5, 35), r_l.u(0, 360)), r_l.u(0.045, 0.065)))
                if prices:
                    tick.setdefault(prices[ki % len(prices)], []).append(
                        M @ trs((cl / 2 - 0.085 + r_l.jitter(0, 0.02), cw / 2 - 0.05, ch + 0.1), (-ctilt - 7 + r_l.jitter(0, 4), r_l.jitter(0, 5), r_l.jitter(0, 9))))
        crate_set(cl, cw, ch, cm + ([spare] if spare is not None else []), [spare] if spare is not None else [])
        fl = _bx("crate_fill", (cl - 0.024, cw - 0.024, 0.004), bev=0.0)
        assign(fl, pine)
        fo_ = instances_to_mesh("crate_fills", fl, fm, keep_base=False)
        paint(fo_, grime=0.9)
        ln = instances_to_mesh("liners", _liner(cl, cw, ch, r_l, cream), cm, keep_base=False)
        paint(ln, grime=_mix((0.15, 1.0), (0.4, noise_mask(0.08, seed=seed + 27, lo=0.3, hi=0.8))))
        for kd in kinds:
            r = kd["r"]
            for which, prof in ((0, kd["lo"]), (1, kd["hi"])):
                ms = prod[kd["name"]][which]
                if not ms or prof is None:
                    continue
                b = _round("fruit", prof, r, kd["seg"], kd["m"])
                if kd["name"] == "apples":                          # blush on one cheek, shaded stalk cavity
                    paint(b, wear=lambda co, n: np.clip(co[:, 0] / r, 0, 1) * 0.55,
                          grime=lambda co, n, r=r: np.clip(1.0 - np.hypot(co[:, 0], co[:, 1]) / (0.34 * r), 0, 1) * (co[:, 2] > 0.4 * r) * 0.9)
                elif kd["name"] == "pears":
                    uv_transform(b, scale=(0.22, 0.22))
                    paint(b, grime=lambda co, n, r=r: np.clip((co[:, 0] / r) * 0.5 + 0.1, 0, 0.5))
                parts = [b]
                if kd["stalk"] is not None:
                    sk = cylinder("stalk", 0.0021, 0.022, loc=(0.002, 0.0, kd["stalk"] * r + 0.008), rot=(0, 14, 0), verts=3, cap=False)
                    assign(sk, green)
                    paint(sk, grime=0.85)
                    parts.append(sk)
                o = instances_to_mesh(kd["name"], join(parts, "fruit_base"), ms, keep_base=False)
                smooth(o, 89)
        if calyx_m:
            cb = lathe("calyx", [(0.0062, 0.0), (0.0, 0.003)], segments=5)
            assign(cb, green)
            paint(cb, grime=0.5)
            instances_to_mesh("calyces", cb, calyx_m, keep_base=False)
        if leaf_m:
            lf = _blade_mesh("fruit_leaf", [((0, 0, 0), (0.45, 0.19, 0.04), (0.45, 0, -0.01), (0.45, -0.19, 0.04), (1, 0, -0.1))])
            assign(lf, green)
            instances_to_mesh("fruit_leaves", lf, leaf_m, keep_base=False, uv_jitter=False)
        # price tickets on sticks
        if tick:
            card = box("ticket_card", (0.08, 0.003, 0.056), loc=(0, 0, 0.0), bevel=0.0)
            assign(card, cream)
            stick = _bx("ticket_stick", (0.007, 0.004, 0.17), (0, 0.0035, -0.085), bev=0.0, grain='Z')
            assign(stick, pine)
            tb = join([card, stick], "ticket_base")
            instances_to_mesh("tickets", tb, [M for ms in tick.values() for M in ms], keep_base=False)
            for label, ms in tick.items():
                tx = text("price", label, size=0.043, depth=0.0008, loc=(0, -0.0017, 0.0), font="serif")
                _lite_text(tx, sides=False, ang=14.0)
                apply_transform(tx)
                assign(tx, body)
                paint(tx, grime=1.0)
                instances_to_mesh("prices", tx, ms, keep_base=False, uv_jitter=False)
        # on the counter crate: a sheaf of paper bags under an iron weight, a second weight beside it
        if spare is not None:
            sx_, sy_, sz_ = spare.translation.x, spare.translation.y, zd + ch + 0.002
            for i in range(4):
                bg = box("paper_bag", (0.15, 0.23, 0.006), loc=(sx_ - 0.07 + r_l.jitter(0, 0.008), sy_ + r_l.jitter(0, 0.008), sz_ + 0.003 + i * 0.0062),
                         rot=(0, 0, -8 + i * 6 + r_l.jitter(0, 3)), bevel=0.0)
                assign(bg, cream)
                paint(bg, grime=0.1 + 0.12 * i)
            wt = lathe("weight", [(0.03, 0.0), (0.032, 0.004), (0.028, 0.034), (0.013, 0.04), (0.017, 0.05), (0.0, 0.054)], segments=7)
            ironwork.append(instances_to_mesh("weights", wt, [trs((sx_ - 0.06, sy_ + 0.03, sz_ + 0.027)), trs((sx_ + 0.11, sy_ - 0.03, sz_), (0, 0, 0), 0.75)], keep_base=False))
    elif p.load == "flowers":
        ncol = max(2, int((x1 - x0) / 0.3))
        pitch = (x1 - x0) / ncol
        rb = min(0.118, W / 4 - 0.035, pitch / 2 - 0.012)
        k = rb / 0.118
        zinc = mat("zinc")
        green = mat("leaf", "#7f9c4a")
        pal = ["#b03a3a", "#8a5cb4", "#d06a2c", "#c4607f"]
        order = sorted(range(len(pal)), key=lambda i: r_l.u())
        petals = [mat("leaf", _lift(p.bloom_color, 0.92)), mat("leaf", _lift(pal[order[0]], 0.92)), cream]
        # stepped shelf for the back row
        sh_h = 0.16
        shelf = [plank("shelf", length=x1 - x0 + 0.02, width=W / 2 - 0.05, thick=0.022, loc=((x0 + x1) / 2, W / 4 - 0.005, zd + sh_h - 0.011), rng=r_l)]
        for i in range(3):
            shelf.append(_bx("shelf_block", (0.035, W / 2 - 0.09, sh_h - 0.022), (x0 + 0.04 + (x1 - x0 - 0.08) * i / 2, W / 4 - 0.005, zd + (sh_h - 0.022) / 2), grain='Y', bev=0.003))
        densify(shelf[0], 0.2)
        for o in shelf:
            assign(o, trim)
            paint(o, wear=_mix((0.1 * kc, 1.0), (0.9 * kc, nzs), (0.5 * kc, nzl)), grime=_mix((0.4, nzg)))
        bucket = lathe("bucket", [(0.084 * k, 0), (0.088 * k, 0.005), (0.113 * k, 0.236), (0.118 * k, 0.24), (0.118 * k, 0.247), (0.111 * k, 0.249),
                                  (0.107 * k, 0.2), (0, 0.2)], segments=12)
        assign(bucket, zinc)
        stem = cylinder("stem", 0.0027, 1.0, loc=(0, 0, 0), verts=3, cap=False)
        transform_verts(stem, lambda co: Vector((co.x, co.y, co.z + 0.5)))
        assign(stem, green)
        leaf = _blade_mesh("leaf", [((0, 0, 0), (0.45, 0.17, 0.035), (0.45, 0, -0.015), (0.45, -0.17, 0.035), (1, 0, -0.09))])
        assign(leaf, green)
        kinds = ["rose", "daisy", "tulip"]
        Rh = {"rose": 0.052, "daisy": 0.056, "tulip": 0.038}
        koff = r_l.pick([0, 1, 2])
        bm_, stem_m, leaf_m, wraps = [], [], [], []
        heads = {}
        for row in range(2):
            for c in range(ncol):
                x = x0 + pitch * (c + 0.5) + r_l.jitter(0, 0.008)
                y = (-1 if row == 0 else 1) * (W / 4 - 0.006) + r_l.jitter(0, 0.006)
                z = zd + (sh_h if row else 0.0)
                sc = r_l.u(0.92, 1.04)
                bm_.append(trs((x, y, z), (0, 0, r_l.u(0, 360)), sc))
                kind = kinds[(c + row + koff) % 3]
                # the saturated colours in the front row next to the lamp, the pale bunches on the back shelf
                ci = (0 if (c + row) % 2 == 0 else 1) if row == 0 else (2 if (c + order[1]) % 2 == 0 else (c // 2 + order[2]) % 2)
                zr_ = z + 0.247 * k * sc                                   # bucket rim
                tall = kind != "tulip"
                wrapped = (c + 2 * row + koff) % 4 == 1
                nst = r_l.pick([8, 9, 9, 10])
                pha = r_l.u(0, 6.28)
                hmax = r_l.u(0.44, 0.55) if tall else r_l.u(0.36, 0.44)
                for i in range(nst):
                    fr_ = i / (nst - 1)                                    # the middle stems stand tallest, the outer ones splay
                    th_ = radians(3 + 21 * fr_ ** 0.7 + r_l.jitter(0, 3))
                    az = pha + i * 2.39996 + r_l.jitter(0, 0.25)
                    hgt = lerp(hmax, 0.3, fr_ ** 1.3) + r_l.jitter(0, 0.025) + 0.03
                    ln = hgt / cos(th_)
                    d = (sin(th_) * cos(az), sin(th_) * sin(az), cos(th_))
                    s0 = (x + 0.035 * k * fr_ * cos(az), y + 0.035 * k * fr_ * sin(az), zr_ - 0.03)
                    stem_m.append(trs(s0, (0, degrees(th_), degrees(az)), (1, 1, ln)))
                    end = (s0[0] + d[0] * ln, s0[1] + d[1] * ln, s0[2] + d[2] * ln)
                    heads.setdefault((kind, ci), []).append(trs(end, (0, degrees(th_) * 1.5 + r_l.jitter(0, 10), degrees(az)), r_l.u(0.85, 1.2)))
                    for _ in range(r_l.pick([1, 1, 2])):                   # leaves up the stem
                        fr = r_l.u(0.35, 0.78)
                        ll = r_l.u(0.09, 0.15)
                        leaf_m.append(trs((s0[0] + d[0] * ln * fr, s0[1] + d[1] * ln * fr, s0[2] + d[2] * ln * fr),
                                          (r_l.jitter(0, 30), -r_l.u(15, 55), degrees(az) + r_l.jitter(0, 70)), (ll, ll * 1.25, ll)))
                for i in range(7):                                         # a collar of broad leaves over the rim
                    a = pha + i * 2 * pi / 7 + r_l.jitter(0, 0.2)
                    ll = r_l.u(0.17, 0.27)
                    leaf_m.append(trs((x + 0.03 * k * cos(a), y + 0.03 * k * sin(a), zr_ - 0.01),
                                      (r_l.jitter(0, 25), -r_l.u(38, 68), degrees(a)), (ll, ll * 1.5, ll)))
                for i in range(3):                                         # tall strap leaves from the middle
                    lnf = r_l.u(0.3, 0.46)
                    leaf_m.append(trs((x, y, zr_ - 0.02), (r_l.jitter(0, 40), -r_l.u(62, 82), r_l.u(0, 360)), (lnf, lnf * 0.34, lnf)))
                if wrapped:
                    wraps.append((x, y, zr_, r_l.u(0, 6.28), r_l.u(0, 6.28)))
        o = instances_to_mesh("buckets", bucket, bm_, keep_base=False)
        paint(o, grime=height_ramp(zd, zd + 0.5), special=mask_mul(noise_mask(0.06, seed=seed + 31, lo=0.5, hi=0.8), 0.5))
        instances_to_mesh("stems", stem, stem_m, keep_base=False)
        lv = instances_to_mesh("leaves", leaf, leaf_m, keep_base=False, uv_jitter=False)
        paint(lv, grime=_mix((0.5, noise_mask(0.07, seed=seed + 33, lo=0.3, hi=0.8))))
        for (kind, ci), ms in heads.items():
            rh = Rh[kind]
            hb = _blade_mesh("bloom", _head(kind, rh))
            assign(hb, petals[ci])
            paint(hb, grime=lambda co, n, rh=rh: np.clip(0.75 - np.hypot(co[:, 0], co[:, 1]) / (0.9 * rh), 0, 0.7))      # deeper colour in the throat
            if kind == "daisy":
                cal = lathe("boss", [(0.3 * rh, 0.08 * rh), (0.2 * rh, 0.26 * rh), (0.0, 0.32 * rh)], segments=6)
            else:
                cal = lathe("calyx", [(0.004, -0.4 * rh), (0.3 * rh, 0.07 * rh)], segments=4)
            assign(cal, green)
            paint(cal, grime=0.5)
            instances_to_mesh(f"blooms_{kind}{ci}", join([hb, cal], "bloom_base"), ms, keep_base=False, uv_jitter=False)
        for i, (x, y, zr_, a0, ph_) in enumerate(wraps):                    # some bunches still in their paper
            def wfn(u, v, x=x, y=y, zr_=zr_, a0=a0, ph_=ph_):
                a = 2 * pi * u
                hgt = 0.2 + 0.07 * cos(a - a0)
                r = lerp(0.075 * k, 0.14 + 0.035 * cos(a - a0), v ** 0.8) * (1 + 0.05 * sin(5 * a + ph_) * v)
                return (x + r * cos(a), y + r * sin(a), zr_ - 0.025 + hgt * v)
            wr = surface(f"wrap{i}", wfn, nu=12, nv=2, closed_u=True, thickness=0.002)
            assign(wr, cream)
            paint(wr, grime=_mix((0.45, 1.0), (0.35, noise_mask(0.1, seed=seed + 35, lo=0.3, hi=0.8))))
    elif p.load == "sacks":
        Rs = min(0.19, W / 4 - 0.01)
        bur = [mat("burlap", "#9a7f52"), mat("burlap", "#75603f")]
        rope = mat("rope")
        sacks, ties = [], []

        def put(sk_c, M, m):
            sk, c = sk_c
            sk.matrix_world = M
            assign(sk, m)
            sacks.append(sk)
            return M @ trs(c)

        # back row: standing sacks of different fill, leaning on the back board and on each other
        nbk = max(2, int((x1 - x0) / (1.8 * Rs)))
        slack = r_l.pick(list(range(nbk)))                                  # one half-empty sack, its top flopped over
        for i in range(nbk):
            x = x0 + (x1 - x0) * (i + 0.5) / nbk + r_l.jitter(0, 0.03)
            full = r_l.u(0.0, 1.0)
            if i == slack:
                sk = _sack(f"sack_b{i}", Rs * 0.98, r_l.u(0.4, 0.46), r_l, flat=0.8, slump=1.0, neck=True, flop=r_l.u(0.2, 0.3))
                M = trs((x, W / 4 - 0.03, zd + 0.004), (-6 + r_l.jitter(0, 3), r_l.jitter(0, 8), 180 + r_l.jitter(0, 25)))
                ties.append(put(sk, M, bur[(i + 1) % 2]) @ trs((0, 0, 0), (-25, 0, 0)))
                continue
            neck = (i + slack) % 2 == 0
            H = lerp(0.5, 0.74, full) if neck else lerp(0.46, 0.66, full)
            sk = _sack(f"sack_b{i}", Rs * r_l.u(0.9, 1.04), H, r_l, flat=r_l.u(0.56, 0.68), slump=r_l.u(0.35, 0.75), neck=neck)
            M = trs((x, W / 4 - 0.035 + r_l.jitter(0, 0.025), zd + 0.004), (-9 + r_l.jitter(0, 4), r_l.jitter(0, 11), r_l.jitter(0, 20)))
            Mt = put(sk, M, bur[i % 2])
            if neck:
                ties.append(Mt @ trs((0, 0, 0), (r_l.jitter(0, 8), r_l.jitter(0, 8), 0), (1.0, 0.9, 1.0)))
        # front row: a crate of potatoes, an opened sack with its mouth rolled down, lying sacks
        cl, cw, ch = 0.42, min(0.3, W / 2 - 0.05), 0.16
        Mc = trs((x0 + cl / 2 + 0.01, -W / 4 + 0.004, zd + 0.001), (0, 0, r_l.jitter(0, 3)))
        crate_set(cl, cw, ch, [Mc])
        pm, pm_hi, z1 = _fill(Mc, cl, cw, ch, 0.036, 0.78, r_l, 0.5, 14)
        fl = _bx("crate_fill", (cl - 0.024, cw - 0.024, 0.004), bev=0.0)
        assign(fl, pine)
        paint(instances_to_mesh("crate_fills", fl, [Mc @ trs((0, 0, z1 - 0.012))], keep_base=False), grime=0.9)
        xs = x0 + cl + 0.05
        Ro, Ho = min(0.17, W / 4 - 0.03), 0.3
        if x1 - xs > 2 * Ro + 0.5:
            xo, yo_ = xs + Ro + 0.01, -W / 4 + 0.015
            osk = lathe("open_sack", [(Ro * 0.6, 0.0), (Ro * 0.96, 0.03), (Ro * 1.04, Ho * 0.35), (Ro * 0.98, Ho * 0.8), (Ro * 0.9, Ho * 0.93), (Ro * 1.0, Ho * 0.97),
                                      (Ro * 1.02, Ho * 1.04), (Ro * 0.92, Ho * 1.08), (Ro * 0.8, Ho * 1.02), (0.0, Ho * 0.98)], segments=12, smooth_angle=75,
                        loc=(xo, yo_, zd + 0.003), rot=(0, 0, r_l.u(0, 360)))
            apply_transform(osk)
            displace_noise(osk, 0.012, 0.13, seed=seed + 43)
            assign(osk, bur[0])
            sacks.append(osk)
            for i in range(11):
                a = i * 2.4
                rr_ = Ro * 0.72 * sqrt((i + 0.5) / 11)
                pm_hi.append(trs((xo + rr_ * cos(a), yo_ + rr_ * sin(a), zd + Ho * 1.0 + 0.03 * (1 - (i + 0.5) / 11)), (r_l.u(-40, 40), r_l.u(-40, 40), r_l.u(0, 360)), r_l.u(0.85, 1.1)))
            xs = xo + Ro + 0.04
        POT = [(1.0, 0.0), (0.88, 0.5), (0.5, 0.86), (0.0, 1.0)]
        for which, prof, ms in ((0, POT, pm), (1, [(0.7, -0.7)] + POT, pm_hi)):
            if not ms:
                continue
            pot = _round("potato", prof, 0.036, 8, mat("terracotta", "#a08458"))
            transform_verts(pot, lambda co: Vector((co.x * 1.25, co.y * 0.95, co.z)))
            displace_noise(pot, 0.004, 0.03, seed=seed + which)
            o = instances_to_mesh("potatoes", pot, ms, keep_base=False)
            smooth(o, 85)
            paint(o, grime=_mix((0.5, noise_mask(0.05, seed=seed + 45, lo=0.3, hi=0.8))))
        Hl = 0.56
        nly = max(0, int((x1 - xs) / (Hl + 0.03)))
        for i in range(nly):
            x = xs + (x1 - xs - nly * Hl) * 0.5 + i * (Hl + 0.01)
            Rl = Rs * r_l.u(0.84, 0.96)
            Mt = put(_sack(f"sack_l{i}", Rl, Hl * r_l.u(0.92, 1.0), r_l, flat=0.6, neck=(i == nly - 1)),
                     trs((x, -W / 4 + 0.01 + r_l.jitter(0, 0.02), zd + Rl * 0.6 * 0.98), (90, 0, 90 + r_l.jitter(0, 7))), bur[(i + 1) % 2])
            if i == nly - 1:
                ties.append(Mt)
        if nly >= 1 and x1 - xs > Hl * 0.9:
            put(_sack("sack_top", Rs * 0.9, Hl * 0.9, r_l, flat=0.56, slump=0.3),
                trs((xs + (x1 - xs - Hl) * (0.5 if nly == 1 else 0.3), -W / 4 + 0.08, zd + Rs * 0.92 * 0.6 * 2.65), (99, 0, 90 + r_l.jitter(11, 7))), bur[0])
        for sk in sacks:
            paint(sk, grime=_mix((0.3 + 0.5 * k_wear, noise_mask(0.12, seed=seed + 41, lo=0.4, hi=0.8)), (0.5, height_ramp(zd, zd + 0.2))))
        if ties:
            tie = torus("sack_tie", major=Rs * 0.22, minor=0.0065, major_seg=10, minor_seg=4)
            assign(tie, rope)
            instances_to_mesh("sack_ties", tie, ties, keep_base=False)
    else:
        # leftovers: two empty crates, a coil of rope, folded sacks
        cl, cw, ch = 0.42, min(0.3, W / 2 - 0.05), 0.15
        xc = x1 - cl / 2 - 0.03
        up = trs((xc, W / 4 - 0.01, zd + ch + 0.001), (180, 0, r_l.jitter(0, 4)))
        cm = [up]
        cm.append(trs((xc - 0.03, W / 4 - 0.025, zd + ch + 0.01), (0, 0, 10 + r_l.jitter(0, 6))))
        crate_set(cl, cw, ch, cm, cm)
        rope = mat("rope")
        rcx, rcy = x0 + 0.2, -W / 4 + 0.03
        pts = []
        turns = 4.5
        for i in range(int(turns * 12) + 1):
            a = i / 12 * 2 * pi
            rr_ = 0.12 + 0.012 * sin(a * 0.37) + 0.004 * (i / 12)
            pts.append((rcx + rr_ * cos(a), rcy + rr_ * sin(a), zd + 0.012 + 0.0195 * (i / 12) * 0.9))
        pts.append((rcx + 0.2, rcy + 0.1, zd + 0.011))
        rp = sweep("rope_coil", pts, 0.011, sides=6, up=(0, 0, 1))
        assign(rp, rope)
        bur = mat("burlap", "#8f7650")
        xs1 = xc - cl / 2 - 0.05                                      # empty sacks folded flat in a loose pile
        for i in range(3):
            Hs = r_l.u(0.44, 0.5) if i < 2 else 0.3
            sk, _c = _sack(f"folded_sack{i}", 0.17 - 0.012 * i, Hs, r_l, flat=0.1 + 0.05 * i, slump=0.2)
            sk.matrix_world = trs((xs1 - Hs - 0.02 * i + r_l.jitter(0, 0.015), W / 4 - 0.03 + r_l.jitter(0, 0.02), zd + 0.02 + i * 0.03),
                                  (90 + r_l.jitter(0, 2), 0, 90 + r_l.jitter(0, 11)))
            assign(sk, bur)
            paint(sk, grime=_mix((0.2, 1.0), (0.6, nzg)))

    # ---------------------------------------------------------------- fixings + materials + weathering
    if bolts:
        bt = lathe("bolt", [(0.0078, 0.0), (0.0058, 0.0036), (0.0, 0.0052)], segments=5)
        ironwork.append(instances_to_mesh("bolts", bt, bolts, keep_base=False))
    if nails:
        nl = lathe("nail", [(0.0045, 0.0), (0.0, 0.0018)], segments=5)
        ironwork.append(instances_to_mesh("nails", nl, nails, keep_base=False))
    for o in ironwork:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.06, seed=seed + 11, lo=0.38, hi=0.72), 0.3 + 0.6 * k_wear), grime=height_ramp(0.0, 0.3),
              wear=_mix((0.35, nzs)))
    for o in cords:
        assign(o, cream)
        paint(o, grime=0.45)

    x_grip0, x_grip1 = xb1 + HL * 0.4, xb1 + HL * 0.85

    def hand_polish(co, n):                                   # paint rubbed through to the wood towards the grips
        return np.clip((co[:, 0] - x_grip0) / (x_grip1 - x_grip0), 0.0, 1.0)

    def corners(co, n):                                       # dirt gathers where the boards meet the ends and the ironwork
        d = np.minimum(np.abs(co[:, 0] - xb0), np.abs(co[:, 0] - xb1))
        for fr in fr_stays:
            d = np.minimum(d, np.abs(co[:, 0] - (xb0 + L * fr)))
        return np.clip(1.0 - d / 0.07, 0, 1)

    def top_edge(co, n):                                      # upper arris of the side / end boards
        return np.clip((co[:, 2] - (zb0 + bh * cf * 0.6)) / (bh * cf * 0.4), 0, 1)

    for o in painted:
        assign(o, body)
        paint(o, wear=_mix((0.04, 1.0), (0.8 * kc, nzs), (0.5 * kc, nzl)), grime=_mix((0.7, height_ramp(0.0, 0.45)), (0.3, nzg)), special=_mix((0.4, nzg)))
        if o.name.startswith("handle"):
            paint(o, wear=hand_polish)
    for o in grips:                                           # bare ash, dark and polished from hands
        assign(o, body if flowers else pine)
        paint(o, grime=_mix((0.45, 1.0), (0.3, nzg)), wear=1.0 if flowers else 0.5)
    for o in boards:
        assign(o, body)
        paint(o, wear=_mix((0.03, 1.0), (0.42 * kc, top_edge), (0.8 * kc, nzs), (0.45 * kc, nzl)),
              special=_mix((0.25 + 0.5 * k_wear, noise_mask(0.4, seed=seed + 15, lo=0.3, hi=0.8))),
              grime=_mix((0.3, height_ramp(zb0, zb0 + bh * 0.6)), (0.45, corners), (0.2, nzg)))
    for o in caps:
        assign(o, trim)
        paint(o, wear=_mix((0.1 * kc, 1.0), (0.9 * kc, nzs), (0.6 * kc, nzl)), grime=_mix((0.44, 1.0), (0.55, corners), (0.45, nzg)), special=_mix((0.6, nzg)))

    socket("load_top", "attach", (cx, 0, zd), role="prop_top")
