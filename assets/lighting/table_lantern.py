"""Table-top candle lantern for the cafe tables, in three constructions:
'hurricane' - a turned chamberstick (saucer foot, baluster stem, gallery cup with spring tabs) carrying a blown
tulip shade of clear glass, with a finger ring and thumb plate soldered to the saucer rim;
'box' - a sheet-metal four-pane lantern on bun feet: moulded tray, arched window plates riveted to corner beads,
a hinged door with knob and catch, swept pyramid roof, vented chimney cap and a carry ring flopped over the cap;
'jar' - a preserving jar standing in a tin dish, candle bedded in sand, screw band on the neck and a wire bail
with a twisted hanging eye.
All of them hold a real wax candle (melted rim, thin runs of wax pooling at the foot, charred wick) with an
emissive flame, a flame fx socket and one small warm light at the flame.

Glass notes: the walls are thin, nearly clear glass (the `glass` colour shows in the thick parts: lip, foot,
neck); soot is a separate smoked film inside the top of the glass. The tulip shade is shaped from the flame
height so that its far wall never mirrors the candle light towards a viewer standing at the table.
"""
from tk import *
import numpy as np
from math import sin, cos, tan, pi, radians, sqrt, atan2, hypot, exp

# auto_grime / auto_wear: the automatic ground-splash dirt (everything below 0.28 m) and convex-edge wear are
# tuned for street furniture; on a 25 cm object they cover everything, so both are turned down and the
# dirt / polish is painted by hand instead.
META = {"name": "Table candle lantern", "class": "prop_small", "mount": "ground", "focus": 0.48,
        "auto_grime": 0.3, "auto_wear": 0.25}
PARAMS = {
    "style": {"default": "hurricane", "choices": ["hurricane", "box", "jar"], "desc": "construction: chamberstick with glass shade / four-pane box lantern / jar with wire bail"},
    "metal": {"default": "brass", "choices": ["brass", "tin", "copper", "iron"], "desc": "metal of the base and fittings"},
    "height": {"default": 0.27, "min": 0.2, "max": 0.35, "desc": "overall height (m)"},
    "intensity": {"default": 0.5, "min": 0.2, "max": 12.0, "desc": "candle light intensity (cd)"},
    "candle": {"default": 0.7, "min": 0.15, "max": 1.0, "desc": "how much of the candle is left (1 = fresh)"},
    "drips": {"default": 5, "min": 0, "max": 10, "desc": "number of wax runs down the candle"},
    "wax": {"default": "#efe4c8", "desc": "wax colour"},
    "glass": {"default": "#ffffff", "desc": "glass tint (shows in the thick parts of the glass)"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "tarnish / rust / soot amount"},
    "lit": {"default": True, "desc": "candle burning (flame, glow and light)"},
}
VARIANTS = {
    "box_tin": {"style": "box", "metal": "tin", "height": 0.3, "candle": 0.55, "age": 0.6, "seed": 4},
    "jar_green": {"style": "jar", "metal": "tin", "height": 0.23, "glass": "#8fdcae", "candle": 0.5, "drips": 3, "wax": "#e9d9b0", "seed": 9},
    "box_copper_tall": {"style": "box", "metal": "copper", "height": 0.35, "candle": 0.95, "drips": 8, "age": 0.8, "intensity": 0.8, "seed": 15},
    "hurricane_iron_stub": {"style": "hurricane", "metal": "iron", "height": 0.21, "candle": 0.2, "drips": 9, "wax": "#d9b98a", "glass": "#ffe2b0", "age": 0.7, "seed": 23},
}


# ----------------------------------------------------------------------------- helpers
def _metals(name):
    """(body, trim, seam) materials. Texture scales are raised: the library tiles are sized for street furniture.
    seam = soldered joints / rolled edges (on copper these have gone green)."""
    brass = mat("brass", scale=2.0, rough=1.5, glow=0.1)
    if name == "brass":
        return brass, brass, brass
    if name == "copper":                                # planished sheet copper, joints gone green
        cu = mat("metal_plain", "#e58a5c", rough=0.42, glow=0.05)
        return cu, cu, mat("copper_patina", scale=5.0)
    if name == "tin":
        zn = mat("zinc", scale=5.0)
        return zn, brass, zn
    fe = mat("iron_painted", "#0c0c0d", scale=3.0, grime=0.02, special=0.02, wear=0.02)      # japanned black ironwork
    return fe, fe, fe


def _mix(c0, c1, t):
    a, b = hex_rgb(c0), hex_rgb(c1)
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _flat_uv(o):
    """Collapse the UVs of a glass piece: the wavy window-glass pattern is far too coarse for blown glass 8 cm wide."""
    realize(o)
    uv_transform(o, scale=(0.01, 0.01))
    uv_keep(o)
    return o


def _sq_lathe(name, prof, corner=0.004, seg=2, cap_top=False, cap_bottom=False):
    """Square 'turned' section: rings of a rounded square of half-width hw at height z (bottom to top
    along the visible outside), for moulded sheet-metal trays and swept pyramid roofs."""
    verts, faces = [], []
    n = 0
    for hw, z in prof:
        hw = max(hw, 2e-4)
        ring = rounded_rect(2 * hw, 2 * hw, min(corner, hw * 0.6), seg)
        n = len(ring)
        verts += [(x, y, z) for x, y in ring]
    for i in range(len(prof) - 1):
        for j in range(n):
            a = i * n + j
            b = i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap_bottom:
        faces.append(tuple(reversed(range(n))))
    if cap_top:
        faces.append(tuple(range((len(prof) - 1) * n, len(prof) * n)))
    o = mesh_object(name, verts, faces)
    smooth(o, 35)
    uv_box(o)
    return o


def _ring(name, centre, u, v, radius, wire, n=16, sides=6):
    """Closed wire ring in the plane spanned by unit vectors u, v."""
    c = Vector(centre)
    u = Vector(u).normalized()
    v = Vector(v).normalized()
    pts = [tuple(c + radius * (cos(2 * pi * i / n) * u + sin(2 * pi * i / n) * v)) for i in range(n)]
    return sweep(name, pts, wire, closed=True, sides=sides, smooth_path=False)


def _flame_rise(fs):
    """Height of the flame centre (= light position) above the top of the candle."""
    return 0.0035 + 0.013 * (0.72 + 0.2 * fs)


def _candle(p, rng, z0, r, h, zmin, fs=1.0, pool_r=None, char=None):
    """Wax candle standing on z0 (radius r, height h, visible from zmin up) with a melted rim, thin runs of wax
    that pool at the foot, charred wick, flame, light and fx socket. The wax carries a little self-glow: the
    candle is its own lamp shade (and stands inside glass, where the baked AO would turn it grey)."""
    lit = bool(p.lit)
    wax = mat("wax", _mix(p.wax, "#f2cd8e", 0.45), grime=0.0, rough=1.6, glow=0.32 if lit else 0.12)
    warm = mat("wax", _mix(p.wax, "#f9bd70", 0.55), grime=0.0, rough=1.6, glow=0.42) if lit else wax
    hot = mat("wax", _mix(p.wax, "#ffa448", 0.65), grime=0.0, rough=1.6, glow=0.56) if lit else wax
    char = char or mat("rope", "#1b1511")
    zt = z0 + h
    hb = min(0.02, h * 0.45)                           # the top 2 cm glow through
    ph0 = rng.u(0, 2 * pi)
    seg = 14

    def rad(z):                                         # hand-dipped: a hair of taper and waviness
        t = (z - z0) / h
        return r * (1.0 - 0.03 * t + 0.012 * sin(t * 7.0 + ph0))

    nb = max(2, int((h - hb) / 0.02) + 1)
    zs = [z0 + (zt - hb - z0) * k / nb for k in range(nb + 1)]
    body = lathe("candle", [(rad(z), z) for z in zs], segments=seg)
    zw = zt - hb * 0.42
    band = lathe("candle_warm", [(rad(zt - hb), zt - hb), (rad(zw), zw)], segments=seg)
    rt = rad(zt)
    top = lathe("candle_top", [(rad(zw), zw), (rt, zt - 0.0035), (rt * 0.985, zt - 0.0012), (rt * 0.9, zt), (rt * 0.72, zt - 0.0012),
                               (rt * 0.35, zt - 0.0032), (0.0, zt - 0.0038)], segments=seg)
    a0, ph = rng.u(0, 2 * pi), rng.u(0, 2 * pi)

    def gutter(a):                                      # the rim has burnt down on one side
        return 0.0034 * max(0.0, cos(a - a0)) ** 2

    def melt(co):
        if co.z > zt - 0.006:
            rr = hypot(co.x, co.y)
            if rr > r * 0.5:
                a = atan2(co.y, co.x)
                dz = (rr / r) * (0.0011 * sin(2 * a + ph) - gutter(a))
                return Vector((co.x, co.y, co.z + dz))
        return co

    transform_verts(top, melt)
    assign(body, wax)
    assign(band, warm)
    assign(top, hot)
    waxes = [body, band, top]

    # runs of wax: thin half-sunk ridges that start at the rim, wander a little, thin out and end in a
    # slim tear; the long ones reach the foot and feed the puddle
    dr = rng.fork("drips")
    full = zt - zmin
    pooled = 0
    for i in range(int(p.drips)):
        a = a0 + dr.u(-1.9, 1.9)                       # most runs on the guttered side
        L = full * dr.u(0.16, 1.25)
        reach = L >= full - 0.004
        if reach:
            L = full + 0.0005
            pooled += 1
        if L < 0.007:
            continue
        w = dr.u(0.0011, 0.0019) * fs
        drift = dr.u(-0.22, 0.22)
        wob, wf, wp = dr.u(0.02, 0.07), dr.u(5.0, 11.0), dr.u(0, 2 * pi)
        zr = zt - gutter(a)
        path = [(r * 0.8 * cos(a), r * 0.8 * sin(a), zr - 0.0009), (r * 0.965 * cos(a), r * 0.965 * sin(a), zr + 0.0002)]
        n = max(4, min(7, int(L / 0.008)))
        for k in range(n + 1):
            t = k / n
            z = zr - 0.003 - (L - 0.003) * t
            aa = a + drift * t + wob * sin(wf * t + wp) * t
            q = rad(z) + w * 0.22
            path.append((q * cos(aa), q * sin(aa), z))
        if reach:
            fn = lambda t: 1.0 - 0.35 * smoothstep(0.08, 0.5, t) + 0.55 * smoothstep(0.86, 1.0, t)
        else:
            fn = lambda t: (1.0 - 0.42 * smoothstep(0.08, 0.6, t)) * (1.0 + 0.5 * exp(-((t - 0.9) / 0.06) ** 2)) * (1.0 - 0.86 * smoothstep(0.93, 1.0, t))
        d = sweep(f"drip{i}", path, w, sides=4, smooth_path=False, scale_fn=fn)
        assign(d, wax)
        waxes.append(d)
    # low puddle of set wax round the foot, lobed towards the guttered side
    if int(p.drips) >= 2:
        amt = min(1.0, 0.35 + 0.22 * pooled + 0.04 * int(p.drips))
        rp = rad(zmin) + (0.0035 + 0.006 * amt) * fs
        if pool_r is not None:
            rp = min(rp, pool_r)
        rb = rad(zmin)
        pud = lathe("wax_puddle", [(rp, zmin - 0.0002), (rp * 0.985, zmin + 0.0007), (lerp(rb, rp, 0.5), zmin + 0.0013),
                                   (rb * 1.04, zmin + 0.0022), (rb * 0.99, zmin + 0.0048)], segments=seg)
        l1, l2 = dr.u(0, 2 * pi), dr.u(0, 2 * pi)

        def lobes(co):
            rr = hypot(co.x, co.y)
            a = atan2(co.y, co.x)
            k = 0.42 + 0.33 * max(0.0, cos(a - a0)) + 0.16 * sin(3 * a + l1) + 0.09 * sin(5 * a + l2)
            q = rb * 0.99 + (rr - rb * 0.99) * clamp(k, 0.12, 1.0)
            return Vector((co.x * q / rr, co.y * q / rr, co.z))

        transform_verts(pud, lobes)
        assign(pud, wax)
        waxes.append(pud)
    # wick
    lean = rng.u(0, 2 * pi)
    wk = sweep("wick", [(0, 0, zt - 0.0045), (0.0004 * cos(lean), 0.0004 * sin(lean), zt + 0.002),
                        (0.0017 * cos(lean), 0.0017 * sin(lean), zt + 0.0062)], 0.0008, sides=5, smooth_path=False)
    assign(wk, char)
    if lit:
        f = 0.72 + 0.2 * fs
        fl = lathe("flame", [(0.0, 0.0), (0.0028 * f, 0.002 * f), (0.0046 * f, 0.0075 * f), (0.0042 * f, 0.013 * f), (0.0027 * f, 0.02 * f),
                             (0.0011 * f, 0.026 * f), (0.0, 0.03 * f)], segments=8, loc=(0.0006 * cos(lean), 0.0006 * sin(lean), zt + 0.0035))
        assign(fl, mat("emissive", "#ffb45e", glow=8))
        light("candle", (0, 0, zt + _flame_rise(fs)), color="#ffbf6a", intensity=float(p.intensity), range=3.0, flicker=0.5,
              shadow_near=max(0.1, p.height * 0.6))
        fx("flame", "flame", (0.0006 * cos(lean), 0.0006 * sin(lean), zt + 0.004), size=0.024 * f)
    return waxes


def _age_metal(objs, p, s, seams=(), soot_top=None, polish=0.3, verd=1.0, dirt=1.0):
    """Tarnish / verdigris / rust where damp sits (at the foot and in the listed seam heights), hand-polish
    on the rest, soot under the vent. Only smooth height masks: the meshes are too coarse for noise masks."""
    age = p.age

    def seam(co, n):
        m = np.zeros(len(co))
        for z in seams:
            m = np.maximum(m, np.clip(1.0 - np.abs(co[:, 2] - z) / (0.004 * s), 0.0, 1.0))
        return m

    for o in objs:
        paint(o, special=mask_mul(height_ramp(0.0, 0.02 * s), 0.6 * age * verd), grime=mask_mul(height_ramp(0.0, 0.03 * s), (0.15 + 0.3 * age) * dirt),
              wear=polish * (1.0 - 0.5 * age))
        if seams:
            paint(o, special=mask_mul(seam, (0.3 + 0.5 * age) * verd), grime=mask_mul(seam, (0.2 + 0.3 * age) * dirt))
        if soot_top is not None:
            paint(o, grime=mask_mul(height_ramp(soot_top, soot_top - 0.06 * s), 0.3 + 0.55 * age))


# ----------------------------------------------------------------------------- hurricane
def _shade_profile(zb, rb, fit, zf, R, ztop, e=-25.0, neck=0.4, n_lo=9, n_up=14, n_ch=3):
    """Lamp chimney: fitter collar (rb) from zb, S-swell to the belly R a little below the flame (height zf), a
    bulb that closes over the flame to a neck, then a straight, slightly flared chimney up to ztop.
    The far wall of a glass mirrors the flame: a ghost flame hangs in it wherever the wall throws the reflection
    towards the eye. The bulb therefore leans inward by (alpha - e) / 2 at every height (alpha = elevation of the
    wall point seen from the flame), which sends all reflections e degrees below the horizon, and the chimney is
    narrow and high enough to send them steeply upward; the crease at the neck jumps over the angles between.
    Returns (bulb profile, chimney profile)."""
    er = radians(e)
    zm = zf - R * tan(-er)                              # widest point: here the wall is plumb
    rn = neck * R
    fine = [(R, zm)]
    q, z = R, zm
    dz = R / 400
    while q > rn and z < ztop - 0.012:
        q -= tan(max(0.0, (atan2(z - zf, q) - er) / 2)) * dz
        z += dz
        fine.append((q, z))
    prof = [(rb, zb), (rb, zb + fit)]
    z0 = zb + fit
    for i in range(1, n_lo + 1):
        t = i / n_lo
        prof.append((rb + (R - rb) * sin(t * pi / 2) ** 1.6, z0 + (zm - z0) * t))
    for i in range(1, n_up + 1):
        prof.append(fine[(i * (len(fine) - 1)) // n_up])
    qn, zn = prof[-1]
    chim = [(qn + 0.0004 + tan(radians(5.0)) * (ztop - zn) * i / n_ch, zn - 0.0005 + (ztop - zn + 0.0005) * i / n_ch) for i in range(n_ch + 1)]
    return prof, chim


def _hurricane(p, rng, body, trim, seamm, thin, thick, soot):
    H = p.height
    s = H / 0.26
    fs = s ** 0.5

    def S(pts, dz=0.0):
        return [(q * s, z * s + dz) for q, z in pts]

    # the shade is designed round the flame: belly radius from the glass height above the flame; a short candle
    # stands on a longer stem rather than at the bottom of a tall glass
    rc = 0.0118 * s ** 0.75
    hc = (0.045 + 0.06 * p.candle) * s
    ztop = H - 0.0012 * s
    above = ztop - (0.072 * s + hc + _flame_rise(fs))
    R = min(0.049 * s, above / 1.25)                    # belly radius of the glass
    lift = max(0.0, above - 2.0 * R)
    zc = 0.072 * s + lift
    zf = zc + hc + _flame_rise(fs)

    met = []
    # saucer foot (spun), planar UVs so the polish does not pinch to the centre
    sau = lathe("saucer", S([(0.040, 0.0), (0.046, 0.0008), (0.0485, 0.004), (0.058, 0.0088), (0.066, 0.0118), (0.0688, 0.0142),
                             (0.0682, 0.0166), (0.066, 0.0172), (0.0645, 0.016), (0.056, 0.0125), (0.036, 0.0105), (0.029, 0.0118)]),
                segments=28)
    uv_planar(sau, (1, 0, 0), (0, 1, 0))
    met.append(sau)
    # baluster stem and gallery cup, turned in one piece
    stem = [(0.029, 0.0118), (0.024, 0.0165), (0.0165, 0.0215), (0.0135, 0.0265), (0.0175, 0.0305), (0.0182, 0.0328), (0.0175, 0.035),
            (0.0125, 0.0385)]
    stem = S(stem)
    if lift > 0.006 * s:                                # long stem: a ring knop half way up
        m = 0.0385 * s + (0.0075 * s + lift) / 2
        stem += [(0.0108 * s, m - 0.0042 * s), (0.0146 * s, m - 0.0016 * s), (0.0146 * s, m + 0.0016 * s), (0.0108 * s, m + 0.0042 * s)]
    stem += S([(0.0112, 0.046), (0.0155, 0.052), (0.026, 0.0575), (0.0335, 0.0625), (0.0362, 0.066), (0.0362, 0.0735), (0.0372, 0.0745),
               (0.0368, 0.076), (0.0338, 0.076), (0.0338, 0.068), (0.0, 0.068)], lift)
    met.append(lathe("stem", stem, segments=20))
    # candle socket inside the gallery
    met.append(lathe("socket", S([(0.0128, 0.068), (0.0156, 0.0686), (0.0160, 0.079), (0.0174, 0.0802), (0.017, 0.0816), (0.0132, 0.0813),
                                  (0.0132, 0.072)], lift), segments=16))
    # spring tabs of the gallery gripping the foot of the glass
    tab = extrude_profile("tab", [(-0.0038 * s, 0.0), (0.0038 * s, 0.0), (0.0038 * s, 0.0085 * s), (0.0, 0.0135 * s), (-0.0038 * s, 0.0085 * s)],
                          0.0011 * s)
    recalc_normals(tab)
    nt = 10
    rt = 0.0352 * s
    tabs = instances_to_mesh("tabs", tab, [trs((rt * sin(radians(i * 360 / nt)), -rt * cos(radians(i * 360 / nt)), 0.0752 * s + lift),
                                               (90 - 4, 0, i * 360 / nt)) for i in range(nt)], keep_base=False)
    # two thumbscrews through the gallery wall that pinch the glass
    for i in range(2):
        a = radians(rng.jitter(70.0, 10.0) + i * 180)
        met.append(lathe(f"thumbscrew{i}", [(0.0016 * s, 0.0), (0.0016 * s, 0.003 * s), (0.0042 * s, 0.0036 * s), (0.0046 * s, 0.0048 * s),
                                            (0.0042 * s, 0.006 * s), (0.0, 0.0064 * s)], segments=8,
                         loc=(0.036 * s * cos(a), 0.036 * s * sin(a), 0.0705 * s + lift), rot=(0, 90, degrees(a))))
    # finger ring soldered to the saucer rim (upright, clear of the table) with the thumb plate on top of it
    ha = radians(rng.jitter(-38.0, 14.0))
    hx, hy = cos(ha), sin(ha)
    rr = (0.0688 + 0.0082) * s
    ring = _ring("finger_ring", (rr * hx, rr * hy, 0.0224 * s), (hx, hy, 0), (0, 0, 1), 0.0105 * s, 0.0021 * s, n=14, sides=6)
    tp = (0.0688 + 0.0085) * s
    thumb = sphere("thumb_plate", 0.011 * s, loc=(tp * hx, tp * hy, 0.0342 * s), segments=10, rings=5, scale=(1.0, 0.74, 0.16))
    place(thumb, rot=(0, -12, degrees(ha)))
    for o in met:
        assign(o, body)
    for o in (tabs, ring, thumb):
        assign(o, trim)
    iron = p.metal == "iron"
    _age_metal(met + [tabs, thumb, ring], p, s, seams=(0.0108 * s, 0.037 * s, 0.049 * s + lift), polish=0.0 if iron else 0.45,
               verd=0.0 if iron else 0.4, dirt=0.0 if iron else 1.0)
    if not iron:
        paint([ring, thumb], wear=0.6)
        paint(sau, wear=mask_mul(height_ramp(0.0175 * s, 0.012 * s), 0.6))           # rim rubbed bright by handling

    # blown glass shade
    zb = 0.069 * s + lift
    prof, chim = _shade_profile(zb, 0.0322 * s, 0.011 * s, zf, R, ztop)
    for nm, pr in (("shade", prof[1:]), ("shade_chimney", chim)):
        ch = lathe(nm, pr, segments=40, smooth_angle=30.0)
        assign(ch, thin)
        _flat_uv(ch)
    # thick parts: ground foot inside the gallery and the fire-polished lip
    qe, ze = chim[-1]
    lip = lathe("shade_lip", [(qe, ze), (qe + 0.0003 * s, ze + 0.0009 * s), (qe - 0.0006 * s, ze + 0.0012 * s), (qe - 0.0014 * s, ze - 0.0006 * s)],
                segments=40)
    foot = lathe("shade_foot", [prof[0], prof[1], (prof[1][0] - 0.0014 * s, prof[1][1]), (prof[0][0] - 0.0014 * s, prof[0][1])], segments=16)
    for o in (lip, foot):
        assign(o, thick)
        _flat_uv(o)
    # window reflections: two soft upright streaks on the outside of the glass
    for i, (az, wd) in enumerate(((ha + 2.2, 0.0034 * s), (ha + 2.75, 0.0017 * s))):
        rows = prof[3:-1:2]
        vs, fc = [], []
        for j, (q, z) in enumerate(rows):
            w = wd * sin(pi * (j + 0.5) / len(rows)) ** 0.7 / q
            vs += [((q + 0.0004) * cos(az - w / 2), (q + 0.0004) * sin(az - w / 2), z), ((q + 0.0004) * cos(az + w / 2), (q + 0.0004) * sin(az + w / 2), z)]
            if j:
                fc.append((2 * j - 2, 2 * j - 1, 2 * j + 1, 2 * j))
        stv = mesh_object(f"shade_glint{i}", vs, fc)
        smooth(stv, 60)
        assign(stv, thick)
    # soot: smoked film inside the top of the glass, three overlapping coats so it fades out downwards
    for i, d in enumerate((0.024 * s, 0.045 * s, 0.066 * s)):
        inset = (0.0007 + 0.0005 * i) * s               # coats must not be coplanar (they would z-fight into stripes)
        whole = prof[:-1] + chim
        zd = ztop - d
        pts = [(float(np.interp(zd, [z for q, z in whole], [q for q, z in whole])) - inset, zd)]
        pts += [(q - inset, z) for q, z in prof[:-1] if z > zd + 0.003 * s][::-1][::2][::-1] + [(q - inset, z) for q, z in chim if z > zd + 0.003 * s]
        if len(pts) >= 2:
            sh = lathe(f"soot{i}", pts, segments=14, smooth_angle=25.0)
            assign(sh, soot)
            _flat_uv(sh)

    _candle(p, rng.fork("candle"), zc, rc, hc, 0.0816 * s + lift, fs=fs, pool_r=0.0146 * s)

    # a spent match left in the saucer
    ma = ha + radians(rng.jitter(150.0, 30.0))
    mr = 0.047 * s
    mz = 0.0128 * s + 0.0011
    wood = mat("wood_oak", "#d8c49c")
    st = box("match", (0.036, 0.0019, 0.0019), loc=(mr * cos(ma), mr * sin(ma), mz), rot=(0, 0, degrees(ma) + 90 + rng.jitter(0, 12)))
    uv_grain(st, "X")
    assign(st, wood)
    assign_faces(st, mat("rope", "#1b1511"), lambda c, n: c.x > 0.011)


# ----------------------------------------------------------------------------- box
def _box(p, rng, body, trim, seamm, thin, thick, soot):
    H = p.height
    s = H / 0.265
    hw = 0.05 * s
    z0, z1, z2, z3 = 0.006 * s, 0.026 * s, 0.168 * s, 0.212 * s
    Hp = z2 - z1
    t = 0.0016 * s
    fs = s ** 0.5
    met, trims, joints, plates = [], [], [], []
    # A flat pane mirrors the flame: seen through the lantern a ghost flame hangs on the far pane above the real
    # one. The panes are therefore crown glass that curls inward above flame height (5 mm at the top, hidden
    # behind the head of the plate) so that they throw the reflection down onto the tray, and the lantern takes
    # a short stout candle, raised on a pricket stand when it has burnt low, so that the flame always burns
    # 4 cm under the crown of the window arch.
    rc = 0.0118 * s ** 0.75
    hc = (0.022 + 0.046 * p.candle) * s
    zA = z2 - 0.014 * s                                # crown of the window arch
    lift = clamp(zA - 0.04 * s - (z1 + 0.004 * s + hc + _flame_rise(fs)), 0.0, 0.045 * s)
    zL = z1 + 0.004 * s + lift + hc + _flame_rise(fs)

    # bun feet
    foot = lathe("foot", [(0.0, 0.0), (0.006 * s, 0.0), (0.0086 * s, 0.003 * s), (0.008 * s, 0.006 * s), (0.006 * s, 0.0078 * s)], segments=8)
    fo = hw - 0.005 * s
    trims.append(instances_to_mesh("feet", foot, [trs((sx * fo, sy * fo, 0)) for sx in (-1, 1) for sy in (-1, 1)], keep_base=False))
    # moulded tray
    met.append(_sq_lathe("tray", [(hw - 0.008 * s, z0), (hw + 0.004 * s, z0), (hw + 0.0056 * s, z0 + 0.0025 * s), (hw + 0.0056 * s, z0 + 0.005 * s),
                                  (hw + 0.003 * s, z0 + 0.007 * s), (hw + 0.0015 * s, z0 + 0.011 * s), (hw + 0.0015 * s, z0 + 0.015 * s),
                                  (hw + 0.004 * s, z0 + 0.017 * s), (hw + 0.004 * s, z1), (hw - 0.004 * s, z1)],
                         corner=0.004 * s, cap_top=True, cap_bottom=True))
    # window plates with arched openings; the front one is the door, set proud on two hinges
    ow, oh = 2 * hw - 0.026 * s, zA - (z1 + 0.014 * s)
    arch = [(x, y - Hp / 2 + 0.014 * s) for x, y in arch_pts(ow, oh, kind="round", n=10)]
    outer = [(-hw, -Hp / 2), (hw, -Hp / 2), (hw, Hp / 2), (-hw, Hp / 2)]
    rivet = lathe("rivet", [(0.0019 * s, 0.0), (0.0014 * s, 0.0009 * s), (0.0, 0.0013 * s)], segments=6)
    rm = []
    zc = (z1 + z2) / 2
    head = z2 - zA                                     # solid head of the plate above the arch
    ztop_p = zA + 0.004 * s
    # crown-glass panes: plumb up to the flame, above it they curl in behind the head of the plate
    e0 = radians(-12.0)
    bow = [(0.0, z1), (0.0, zL + (hw - t) * tan(e0))]
    nb = 14
    for i in range(nb):
        zq = bow[-1][1] + (ztop_p - bow[1][1]) / nb
        hm = (zq + bow[-1][1]) / 2 - zL
        bow.append((bow[-1][0] + tan((atan2(hm, hw - t - bow[-1][0]) - e0) / 2) * (zq - bow[-1][1]), zq))

    def sheet(name, a, q, zlo):
        rows = [(lerp(bow[0][0], bow[1][0], 0.0), zlo)] + bow[1:] if zlo < bow[1][1] else \
               [(np.interp(zlo, [b[1] for b in bow], [b[0] for b in bow]), zlo)] + [b for b in bow if b[1] > zlo + 1e-4]
        vs, fc = [], []
        for j, (ins, z) in enumerate(rows):
            u = min(hw - 0.006 * s, q - ins - 0.0006)
            for sg in (-1, 1):
                vs.append((sg * u * cos(a) + (q - ins) * sin(a), sg * u * sin(a) - (q - ins) * cos(a), z))
            if j:
                fc.append((2 * j - 2, 2 * j - 1, 2 * j + 1, 2 * j))
        o = mesh_object(name, vs, fc)
        smooth(o, 60)
        uv_box(o)
        return _flat_uv(o)

    for k in range(4):
        a = radians(k * 90)
        out = hw + (0.0018 * s if k == 0 else 0.0)
        cx, cy = (out - t) * sin(a), -(out - t) * cos(a)
        pl = shape_extrude("door" if k == 0 else f"plate{k}", outer, [arch], depth=t, loc=(cx, cy, zc), rot=(90, 0, k * 90))
        met.append(pl)
        plates.append(pl)
        assign(sheet(f"pane{k}", a, out - t - 0.0012, z1 + 0.004 * s), thin)
        # soot on the inside of the pane under the roof: two overlapping smoked coats
        for i, d in enumerate((0.02 * s, 0.038 * s)):
            assign(sheet(f"pane_soot{k}_{i}", a, out - t - 0.0022 - 0.0006 * i, ztop_p - d), soot)
        for ux in (-1, 1):
            for uz in (-1, 1):
                lx, lz = ux * (hw - 0.0065 * s), uz * (Hp / 2 - 0.0065 * s)
                rm.append(trs((lx * cos(a) + out * sin(a), lx * sin(a) - out * cos(a), zc + lz), (90, 0, k * 90)))
        if head > 0.02 * s:                             # a row of rivets holds the stiffening strip behind the head
            for ux in (-0.5, 0.0, 0.5):
                lx = ux * hw
                rm.append(trs((lx * cos(a) + out * sin(a), lx * sin(a) - out * cos(a), z2 - head * 0.42), (90, 0, k * 90)))
    trims.append(instances_to_mesh("rivets", rivet, rm, keep_base=False))
    # corner beads (the sheet is lapped and soldered over a wire)
    for sx in (-1, 1):
        for sy in (-1, 1):
            joints.append(cylinder("bead", 0.0036 * s, Hp + 0.002 * s, loc=(sx * (hw - 0.0006 * s), sy * (hw - 0.0006 * s), zc), verts=8, cap=False))
    # door furniture: hinges on the left, knob and catch on the right
    yd = -(hw + 0.0018 * s)
    hinges = []
    for fz in (0.22, 0.78):
        hinges.append(cylinder("hinge", 0.0027 * s, 0.02 * s, loc=(-hw - 0.0014 * s, yd + 0.0006 * s, z1 + Hp * fz), verts=8))
        hinges.append(box("hinge_leaf", (0.011 * s, 0.001 * s, 0.016 * s), loc=(-hw + 0.0045 * s, yd - 0.0004 * s, z1 + Hp * fz)))
    trims += hinges
    # knob and catch sit just above flame height: the candle light casts no shadows, below the flame their upper
    # faces would be lit through the door and glow like beads
    zkn = zL + 0.008 * s
    knob = lathe("knob", [(0.0016 * s, 0.0), (0.0016 * s, 0.003 * s), (0.0036 * s, 0.0045 * s), (0.004 * s, 0.0065 * s), (0.0026 * s, 0.0085 * s),
                                (0.0, 0.009 * s)], segments=8, loc=(hw - 0.0065 * s, yd, zkn), rot=(90, 0, 0))
    catch = box("catch", (0.0035 * s, 0.011 * s, 0.0045 * s), loc=(hw + 0.0012 * s, -hw + 0.002 * s, zkn + 0.009 * s), bevel=0.0006 * s, segments=1)
    trims += [knob, catch]
    # top frame and swept pyramid roof with a turned-down drip edge
    joints.append(_sq_lathe("top_frame", [(hw + 0.0012 * s, z2 - 0.006 * s), (hw + 0.004 * s, z2 - 0.004 * s), (hw + 0.004 * s, z2 - 0.0005 * s),
                                       (hw + 0.002 * s, z2 + 0.0015 * s)], corner=0.004 * s))
    r3 = 0.0215 * s
    rs = z3 - z2
    met.append(_sq_lathe("roof", [(hw + 0.003 * s, z2 + 0.001 * s), (hw + 0.0115 * s, z2 + 0.0005 * s), (hw + 0.0122 * s, z2 + 0.0022 * s),
                                  (hw + 0.006 * s, z2 + 0.0065 * s), (lerp(hw, r3, 0.34), z2 + rs * 0.3), (lerp(hw, r3, 0.62), z2 + rs * 0.52),
                                  (lerp(hw, r3, 0.84), z2 + rs * 0.76), (r3, z3)], corner=0.005 * s, seg=2))
    # rolled seams down the four hips of the roof
    for sx in (-1, 1):
        for sy in (-1, 1):
            hp = []
            for q, z in ((hw + 0.0118 * s, z2 + 0.0026 * s), (hw + 0.006 * s, z2 + 0.0068 * s), (lerp(hw, r3, 0.34), z2 + rs * 0.3),
                         (lerp(hw, r3, 0.62), z2 + rs * 0.52), (lerp(hw, r3, 0.84), z2 + rs * 0.76), (r3, z3)):
                c = min(0.005 * s, q * 0.6) * 0.29          # the rounded corner sits this far inside the square's corner
                hp.append((sx * (q - c), sy * (q - c), z + 0.0003 * s))
            joints.append(sweep("hip_seam", hp, 0.0016 * s, sides=5, smooth_path=False))
    # vent: collar, three stand-offs, conical cap with finial
    met.append(lathe("vent", [(0.0185 * s, z3 - 0.006 * s), (0.0185 * s, z3 + 0.012 * s), (0.0205 * s, z3 + 0.013 * s), (0.0205 * s, z3 + 0.015 * s),
                              (0.015 * s, z3 + 0.015 * s), (0.015 * s, z3 + 0.006 * s)], segments=16))
    for k in range(3):
        a = radians(30 + k * 120)
        met.append(cylinder("standoff", 0.0015 * s, 0.006 * s, loc=(0.0172 * s * cos(a), 0.0172 * s * sin(a), z3 + 0.0175 * s), verts=6, cap=False))
    met.append(lathe("cap", [(0.012 * s, z3 + 0.0195 * s), (0.028 * s, z3 + 0.0188 * s), (0.0288 * s, z3 + 0.0202 * s), (0.02 * s, z3 + 0.027 * s),
                             (0.008 * s, z3 + 0.032 * s), (0.005 * s, z3 + 0.034 * s), (0.0044 * s, z3 + 0.038 * s), (0.006 * s, z3 + 0.0402 * s),
                             (0.0045 * s, z3 + 0.0428 * s), (0.0, z3 + 0.0436 * s)], segments=16))
    # eye and carry ring, flopped over onto the cap
    da = radians(rng.jitter(-55.0, 25.0))
    d = Vector((cos(da), sin(da), 0.0))
    v = Vector((-d.y, d.x, 0.0))
    E = Vector((0, 0, z3 + 0.047 * s))
    trims.append(_ring("eye", E, v, (0, 0, 1), 0.0045 * s, 0.0013 * s, n=10, sides=5))
    dl = radians(rng.jitter(40.0, 3.0))
    u = Vector((d.x * cos(dl), d.y * cos(dl), -sin(dl)))
    R = 0.015 * s
    ring = _ring("carry_ring", E + R * u, u, v, R, 0.002 * s, n=20, sides=7)
    trims.append(ring)

    for o in met:
        assign(o, body)
    for o in trims:
        assign(o, trim)
    for o in joints:
        assign(o, seamm)
    # the inside of the plates is black with soot (bare metal there mirrors the flame as a white blob)
    black = mat("plain", "#15120f")
    for pl in plates:
        assign_faces(pl, black, lambda c, n: n.z < -0.5)
    for o in hinges:                                    # the side of the hinges the door would shade from the flame
        assign_faces(o, black, lambda c, n: n.x > 0.3)
    assign_faces(knob, black, lambda c, n: n.x < -0.4)
    assign_faces(catch, black, lambda c, n: n.y > 0.3)
    _age_metal(met + trims + joints, p, s, seams=(z0 + 0.008 * s, z1 + 0.001 * s, z2 - 0.003 * s, z2 + 0.006 * s), soot_top=z3 + 0.044 * s,
               polish=0.45, verd=0.5)
    if p.metal == "copper":
        paint(joints, special=0.25 + 0.5 * p.age)      # verdigris has crept along the soldered joints
    paint(ring, wear=0.6)

    # candle socket on the tray floor (on a pricket stem when the candle has burnt low) and the candle
    zk = z1 + lift
    sp = [(0.02 * s, z1), (0.0195 * s, z1 + 0.0015 * s)]
    if lift > 0.006 * s:
        sp += [(0.0105 * s, z1 + 0.003 * s), (0.0062 * s, z1 + 0.0065 * s), (0.0062 * s, zk - 0.002 * s), (0.0105 * s, zk + 0.001 * s)]
    sp += [(0.0152 * s, zk + 0.0025 * s), (0.0156 * s, zk + 0.011 * s), (0.017 * s, zk + 0.0122 * s), (0.0166 * s, zk + 0.0135 * s),
           (0.013 * s, zk + 0.0132 * s), (0.013 * s, zk + 0.004 * s)]
    sk = lathe("socket", sp, segments=14)
    assign(sk, trim)
    _candle(p, rng.fork("candle"), zk + 0.004 * s, rc, hc, zk + 0.0135 * s, fs=fs, pool_r=0.0164 * s)


# ----------------------------------------------------------------------------- jar
def _jar(p, rng, body, trim, seamm, thin, thick, soot):
    H = p.height
    s = H / 0.25
    fs = s ** 0.5
    zj = 0.003 * s

    def S(pts, dz=0.0):
        return [(q * s, z * s + dz) for q, z in pts]

    met = []
    # pie tin the jar stands in (catches wax, keeps the hot glass off the table). Raked sides with a rolled rim:
    # they face away from the flame, so the tin stays dark with only the rim catching the light
    dish = lathe("dish", S([(0.034, 0.0), (0.0475, 0.0), (0.0492, 0.0012), (0.0515, 0.0078), (0.0524, 0.0086), (0.0522, 0.0098),
                            (0.0508, 0.0098), (0.0502, 0.0086), (0.0482, 0.0036), (0.0465, 0.003), (0.0, 0.003)]), segments=28)
    uv_planar(dish, (1, 0, 0), (0, 1, 0))
    met.append(dish)
    # preserving jar. Thick moulded foot and lip (these carry the glass colour), thin clear wall between them;
    # the shoulder is one arc that meets the neck in a crisp moulded step
    zsh = 0.104                                        # shoulder starts here (unscaled)
    rho, th1 = 0.0333, radians(48.0)
    wall = [(0.042, 0.0115), (0.042, 0.06), (0.042, zsh)]
    for i in range(1, 8):
        th = th1 * i / 7
        wall.append((0.042 - rho * (1 - cos(th)), zsh + rho * sin(th)))
    zn = zsh + rho * sin(th1)                           # ~0.1287, foot of the neck (r = 0.031)
    jar = lathe("jar", S(wall, zj), segments=32)
    assign(jar, thin)
    _flat_uv(jar)
    # moulded screw thread: no plumb glass in the neck (a plumb neck would mirror the flame towards the eye)
    neck = [wall[-1]] + [(0.0326 if i % 2 else 0.031, zn + 0.0016 * i) for i in range(1, 9)] + [(0.031, 0.143)]
    wall += neck[1:]
    nk = lathe("jar_neck", S(neck, zj), segments=32, smooth_angle=30.0)
    base = lathe("jar_foot", S([(0.034, 0.0), (0.0385, 0.0012), (0.0412, 0.005), (0.042, 0.0115), (0.0398, 0.0115), (0.036, 0.008), (0.0, 0.007)], zj),
                 segments=32)
    lip = lathe("jar_lip", S([(0.031, 0.143), (0.0312, 0.149), (0.0304, 0.15), (0.0288, 0.1495), (0.0268, 0.136)], zj), segments=32)
    for o in (base, lip, nk):
        assign(o, thick)
        _flat_uv(o)
    # soot under the shoulder
    for i, za in enumerate((0.118, 0.108)):
        pts = [(q - 0.0008 - 0.0005 * i, z) for q, z in wall if za <= z <= zn + 1e-6]
        sh = lathe(f"soot{i}", S(pts, zj), segments=20)
        assign(sh, soot)
        _flat_uv(sh)

    # candle: its foot is bedded in sand, deep enough that the flame burns up in the shoulder
    # (there the glass already leans in over the flame and throws its reflection down, not at the eye)
    rc = 0.0155 * s ** 0.75
    zfl = zj + (zsh + rho * sin(radians(12.0))) * s - _flame_rise(fs)        # top of the candle
    hc = min((0.055 + 0.045 * p.candle) * s, zfl - (zj + 0.013 * s))
    zc = zfl - hc
    zs = zc + 0.0045 * s                               # sand surface
    sand = lathe("sand", [(0.0412 * s, zj + 0.0118 * s), (0.0412 * s, zs - 0.0012 * s), (0.033 * s, zs + 0.0008 * s), (0.02 * s, zs),
                          (0.0, zs + 0.0006 * s)], segments=20)
    displace_noise(sand, strength=0.0012 * s, scale=0.012, seed=p.seed, axis_mask=(0, 0, 1))
    assign(sand, mat("mortar", "#6a573c", scale=3.0))
    # screw band on the neck with two lugs for the bail
    zb = zj + 0.145 * s
    met.append(lathe("band", S([(0.0318, 0.1392), (0.0338, 0.1398), (0.0342, 0.1415), (0.0338, 0.1432), (0.0338, 0.1478), (0.0342, 0.1492),
                                (0.0336, 0.151), (0.031, 0.1518), (0.0296, 0.1512)], zj), segments=28))
    xe = 0.0362 * s
    wire = mat("iron_wrought", scale=2.5, wear=0.35)
    wires = []
    for sx in (-1, 1):
        wires.append(_ring("lug", (sx * (xe - 0.0012 * s), 0, zb), (0, 1, 0), (0, 0, 1), 0.003 * s, 0.001 * s, n=8, sides=5))
    # wire tied round the neck under the band, ends twisted together
    wires.append(torus("neck_wire", 0.0322 * s, 0.0012 * s, loc=(0, 0, zj + 0.137 * s), major_seg=24, minor_seg=5))
    ta = radians(rng.jitter(140.0, 30.0))
    tx, ty = cos(ta), sin(ta)
    wires.append(sweep("twist", [(0.032 * s * tx, 0.032 * s * ty, zj + 0.137 * s), (0.039 * s * tx, 0.039 * s * ty, zj + 0.1365 * s),
                                 (0.044 * s * tx - 0.003 * s * ty, 0.044 * s * ty + 0.003 * s * tx, zj + 0.133 * s)], 0.0017 * s, sides=5,
                       twist_deg=540))
    # bail: arch of wire with a twisted hanging eye at the crown, leaning a little off upright
    tilt = radians(rng.jitter(0.0, 14.0))
    er = 0.0058 * s
    rise = H - zb - 2 * er - 0.0014 * s

    def lean(x, y, z):
        dz = z - zb
        return (x, y * cos(tilt) - dz * sin(tilt), zb + y * sin(tilt) + dz * cos(tilt))

    n = 18
    path = [lean(-xe * 1.02, 0, zb - 0.002 * s)]
    for i in range(n + 1):
        th = pi * i / n
        path.append(lean(-xe * cos(th) * (1.0 + 0.1 * sin(th)), 0, zb + rise * sin(th) ** 0.75))
    path.append(lean(xe * 1.02, 0, zb - 0.002 * s))
    wires.append(sweep("bail", path, 0.0014 * s, sides=6))
    ec = lean(0, 0, zb + rise + er - 0.0006 * s)
    wires.append(_ring("bail_eye", ec, (1, 0, 0), Vector(lean(0, 0, zb + 1)) - Vector((0, 0, zb)), er, 0.0014 * s, n=12, sides=6))

    for o in met:
        assign(o, body)
    for o in wires:
        assign(o, wire)
        paint(o, special=mask_mul(noise_mask(0.02, seed=p.seed + 5, lo=0.4, hi=0.7), 0.4 + 0.6 * p.age))
    _age_metal(met, p, s, seams=(0.004 * s,))
    paint(dish, grime=0.85, special=0.4 + 0.4 * p.age)   # dulled and rusted by spilt wax, damp and candle smoke
    paint(wires[-2], wear=mask_mul(height_ramp(H, H - 0.03 * s), 0.6))

    _candle(p, rng.fork("candle"), zc, rc, hc, zs + 0.0004 * s, fs=fs, pool_r=rc + 0.006 * s, char=wire)


def build(p, rng):
    body, trim, seamm = _metals(p.metal)
    g = hex_rgb(p.glass)
    lo = min(g)
    # thin wall: clear, only reflections (any body here turns into an opaque cream fill under the candle light),
    # a tinted glass keeps a breath of its colour; thick glass: carries the colour; soot: smoked brown-black film
    sat = max(g) - lo
    thin = mat("glass_clear", tuple(c * 0.07 + (c - lo) * 0.8 for c in g), opacity=0.18 * sat, grime=0.0)
    thick = mat("glass_clear", tuple(0.6 * (c - 0.6 * lo) / (1.0 - 0.6 * lo) for c in g), opacity=0.5, grime=0.0, glow=0.4 * sat)
    soot = mat("glass_clear", "#0c0906", opacity=0.04 + 0.08 * p.age, grime=0.0)
    if p.style == "box":
        _box(p, rng, body, trim, seamm, thin, thick, soot)
    elif p.style == "jar":
        _jar(p, rng, body, trim, seamm, thin, thick, soot)
    else:
        _hurricane(p, rng, body, trim, seamm, thin, thick, soot)
