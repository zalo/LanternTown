"""Hanging chain lantern for archways, the canal tunnel and timber galleries. A turned ceiling rose (or a
forged plate) carries a hook; from it a forged chain (long drops add a twisted hanger rod, optionally a smoke
bell) holds the lantern by its top loop. Three bodies: 'cylinder' (turned ogee cap with vent chimney, pierced
vent gallery, glass cylinder inside tie-rod guards, pierced air gallery, pan with scroll stays and drip finial),
'hex' (tented six-sided roof with hip rolls, pierced gallery, arched face plates over tapered glass, corner
drops) and 'globe' (blown globe in a wire cage under a pierced crown). Everything below the hook is one
'swing' part.
"""
from tk import *
from math import sqrt, ceil, floor, asin, acos, atan2, degrees
import numpy as np
import bpy

META = {"name": "Hanging chain lantern", "class": "component", "mount": "hang", "focus": 0.3, "mount_height": 3.6}
PARAMS = {
    "drop": {"default": 0.6, "min": 0.3, "max": 2.4, "desc": "drop from the fixing to the top of the lantern loop (m); long drops add a forged hanger rod above the chain"},
    "style": {"default": "cylinder", "choices": ["cylinder", "hex", "globe"], "desc": "lantern body: glass cylinder in tie-rod guards / tented hexagon / caged globe"},
    "size": {"default": 0.26, "min": 0.18, "max": 0.4, "desc": "lantern body diameter (m); everything else scales with it"},
    "metal": {"default": "iron", "choices": ["iron", "brass", "copper"], "desc": "metalwork: black iron with brass trim / aged brass / copper with verdigris cap"},
    "paint": {"default": "#3b4248", "desc": "paint colour of the sheet-iron cap, galleries and pan (iron only)"},
    "glass_tint": {"default": "#ffb469", "desc": "glass tint"},
    "frosted": {"default": False, "desc": "dense opal glass instead of clear seeded glass"},
    "taper": {"default": 0.0, "min": 0.0, "max": 0.3, "desc": "narrowing of the glazed body towards the bottom (cylinder / hex)"},
    "bars": {"default": 6, "min": 3, "max": 10, "desc": "guard rods (cylinder) / cage ribs (globe)"},
    "fixing": {"default": "rose", "choices": ["rose", "plate"], "desc": "turned ceiling rose (vaults) or forged square plate (timber beams)"},
    "smoke_bell": {"default": False, "desc": "smoke bell hung on the chain above the lantern (needs ~0.4 m of drop)"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "soot, rust / verdigris and dents"},
    "lit": {"default": True, "desc": "burning or out"},
    "intensity": {"default": 35.0, "min": 4, "max": 120, "desc": "light intensity (cd)"},
}
VARIANTS = {
    "hex_brass": {"style": "hex", "metal": "brass", "taper": 0.24, "size": 0.3, "drop": 0.45, "fixing": "plate",
                  "glass_tint": "#ffc47c", "age": 0.35, "intensity": 24, "seed": 4},
    "globe_copper": {"style": "globe", "metal": "copper", "frosted": True, "size": 0.3, "drop": 0.9, "bars": 8,
                     "smoke_bell": True, "glass_tint": "#ffcf90", "age": 0.7, "intensity": 36, "seed": 11},
    "tunnel_long": {"style": "cylinder", "drop": 0.74, "size": 0.21, "bars": 4, "taper": 0.16, "age": 1.0,
                    "glass_tint": "#f2d79a", "paint": "#4a3b2d", "intensity": 24, "seed": 23},
    "hex_red_small": {"style": "hex", "metal": "iron", "size": 0.2, "drop": 0.34, "paint": "#8c2620",
                      "glass_tint": "#ff9a5c", "age": 0.6, "intensity": 20, "seed": 31},
}

PANE_GLOW, PANE_CLEAR = 0.2, 0.06     # clear seeded glass: glow, see-through share at size 0.26 (see the materials section)
OPAL_GLOW, OPAL_CLEAR = 0.2, 0.035   # opal glass
LINER_GLOW = 1.15    # bright inside seen through the pierced galleries
WARM_K, WARM2_K = 0.003, 0.04   # tint scale of the flame-side faces: inside the housing / above the cap
BURN_GLOW = 0.75     # oil font and burner
FAR_REACH = 0.34     # m at intensity 35: beyond this the lamp no longer burns the ceiling fixing out
PAINT_GLOW = 0.17    # hex body: paint colour picked up from the flame by the plates and rail that face away from it
# turned profiles: (radius, dz) bottom-to-top, see _turn()
STEM = [(0.026, -0.088), (0.031, -0.084), (0.031, -0.079), (0.021, -0.075), (0.013, -0.069), (0.012, -0.063), (0.019, -0.058),
        (0.0225, -0.053), (0.019, -0.048), (0.011, -0.045), (0.008, -0.041), (0.0, -0.040)]
CAP = [(0.92, -0.131), (1.07, -0.135), (1.115, -0.136), (1.14, -0.132), (1.14, -0.127), (1.115, -0.123), (1.04, -0.120),
       (0.95, -0.114), (0.87, -0.104), (0.80, -0.092), (0.74, -0.078), (0.66, -0.064), (0.56, -0.054), (0.45, -0.049),
       (0.34, -0.044), (0.31, -0.037), (0.31, -0.022), (0.41, -0.024), (0.46, -0.018), (0.42, -0.011), (0.3, -0.004), (0.2, 0.0)]
# rings grip the glass with a steep bezel so no flat face stares at the flame
RING = [(0.875, -0.040), (0.985, -0.040), (1.02, -0.031), (1.01, -0.019), (0.975, -0.006), (0.95, 0.0), (0.86, 0.0)]
PAN = [(0.1, -0.062), (0.17, -0.060), (0.22, -0.055), (0.3, -0.052), (0.5, -0.044), (0.68, -0.032), (0.80, -0.019),
       (0.865, -0.008), (0.88, -0.001), (0.875, 0.004)]
# pendant finial: nearly every face looks down / outwards, away from the flame above it, so it stays a dark silhouette
DRIP = [(0.0, -0.088), (0.0035, -0.0845), (0.0065, -0.078), (0.0078, -0.072), (0.0066, -0.067), (0.0056, -0.063), (0.0085, -0.054),
        (0.013, -0.042), (0.0172, -0.030), (0.0196, -0.019), (0.0196, -0.013), (0.0235, -0.010), (0.0245, -0.005), (0.0215, 0.002)]
HEX_VENT = [(0.034, -0.046), (0.034, -0.020), (0.05, -0.022), (0.055, -0.019), (0.051, -0.014), (0.034, -0.005), (0.026, 0.0)]
HEX_ROOF = [(0.78, -0.105), (1.20, -0.112), (1.245, -0.1125), (1.25, -0.105), (1.22, -0.100), (1.04, -0.088), (0.8, -0.068),
            (0.58, -0.045), (0.42, -0.022), (0.33, 0.0), (0.22, 0.003)]
HEX_RAIL = [(0.955, -0.036), (1.02, -0.012), (1.04, -0.007), (1.03, -0.003), (0.99, 0.0), (0.80, 0.0)]
HEX_PAN = [(0.2, -0.060), (0.3, -0.052), (0.52, -0.041), (0.74, -0.024), (0.88, -0.009), (0.95, -0.001), (0.95, 0.004), (0.9, 0.006)]
GLOBE_CAP = [(0.056, -0.050), (0.074, -0.054), (0.0785, -0.052), (0.078, -0.048), (0.066, -0.041), (0.052, -0.029),
             (0.042, -0.017), (0.035, -0.008), (0.026, 0.0)]
GLOBE_COLLAR = [(0.05, -0.014), (0.066, -0.014), (0.0705, -0.011), (0.0705, -0.005), (0.066, -0.002), (0.062, 0.0), (0.05, 0.0)]
GLOBE_CUP = [(0.012, -0.070), (0.019, -0.066), (0.022, -0.059), (0.03, -0.053), (0.044, -0.041), (0.054, -0.025), (0.0585, -0.010),
             (0.060, 0.0), (0.0635, 0.004), (0.0635, 0.008), (0.058, 0.010)]
# (the font starts at the pan floor: a foot ring tucked 2-3 mm under the floor fought it once the font glows - CLEARANCE RULE,
#  a glowing surface keeps >= 4 mm to any parallel surface, or the hidden one goes)
FONT = [(0.036, 0.0), (0.041, 0.005), (0.043, 0.014), (0.038, 0.023), (0.024, 0.029), (0.015, 0.032)]
HEX_SILL = [(0.80, -0.040), (0.99, -0.040), (1.07, -0.036), (1.092, -0.030), (1.085, -0.023), (1.045, -0.012), (0.975, 0.0125),
            (0.955, 0.014), (0.90, 0.014)]
ACORN = [(0.0, -0.036), (0.0035, -0.033), (0.0078, -0.026), (0.0088, -0.019), (0.007, -0.014), (0.0098, -0.012), (0.0108, -0.008),
         (0.009, -0.004), (0.0045, -0.002), (0.0045, 0.004)]
FLAME = [(0.0, 0.0), (0.008, 0.004), (0.013, 0.014), (0.012, 0.026), (0.007, 0.040), (0.002, 0.054), (0.0, 0.058)]
BELL = [(0.070, 0.0), (0.0735, 0.002), (0.072, 0.006), (0.062, 0.016), (0.047, 0.030), (0.031, 0.042), (0.017, 0.050),
        (0.011, 0.056), (0.008, 0.062), (0.004, 0.062), (0.004, 0.054), (0.014, 0.047), (0.029, 0.039), (0.044, 0.027),
        (0.058, 0.014), (0.067, 0.004), (0.070, 0.0)]


def _turn(name, pts, z, kr=1.0, kz=1.0, seg=28, **kw):
    return lathe(name, [(r * kr, z + dz * kz) for r, dz in pts], segments=seg, **kw)


def _mix_hex(a, b, t):
    try:
        pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
        pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
        return "#%02x%02x%02x" % tuple(int(round(lerp(x, y, t))) for x, y in zip(pa, pb))
    except Exception:
        return a


def _strip(name, cols, mapf, thick, closed=True, rim=False):
    """Pierced sheet-metal band. cols = [(s, zs, hole_id)]: s = position along the band, zs = even-length
    ascending heights (metal between zs[0]-zs[1], zs[2]-zs[3], ...), hole_id = the opening this column crosses
    (None = solid). mapf(s, z, d) -> point d below the outer face. rim=True: only the outer skin and the walls of
    the piercings are built (for bands whose back is hidden by a liner); otherwise the sheet is solidified."""
    verts, cidx, vsz = [], [], []
    for s_, zs, _h in cols:
        idx = []
        for k, z in enumerate(zs):
            if k and abs(z - zs[k - 1]) < 1e-6:
                idx.append(idx[-1])
            else:
                verts.append(tuple(mapf(s_, z)))
                vsz.append((s_, z))
                idx.append(len(verts) - 1)
        cidx.append(idx)
    faces, inner = [], {}

    def add(f):
        g = []
        for i in f:
            if not g or g[-1] != i:
                g.append(i)
        if len(g) > 1 and g[0] == g[-1]:
            g.pop()
        if len(set(g)) >= 3:
            faces.append(g)

    def inn(i):
        if i not in inner:
            verts.append(tuple(mapf(vsz[i][0], vsz[i][1], thick)))
            inner[i] = len(verts) - 1
        return inner[i]

    n = len(cols)
    for j in range(n if closed else n - 1):
        k2 = (j + 1) % n
        a, b = cidx[j], cidx[k2]
        if cols[j][2] is not None and cols[j][2] == cols[k2][2]:
            for k in range(0, len(a), 2):
                add([a[k], b[k], b[k + 1], a[k + 1]])
            if rim:
                for k in range(1, len(a) - 1, 2):
                    add([a[k], b[k], inn(b[k]), inn(a[k])])
                    add([b[k + 1], a[k + 1], inn(a[k + 1]), inn(b[k + 1])])
        else:
            add([a[0]] + list(b) + list(reversed(a[1:])))
            if rim:                                        # upright jambs where an opening starts / ends square
                if cols[k2][2] is not None and len(b) == 4 and b[1] != b[2]:
                    add([b[2], b[1], inn(b[1]), inn(b[2])])
                if cols[j][2] is not None and len(a) == 4 and a[1] != a[2]:
                    add([a[1], a[2], inn(a[2]), inn(a[1])])
    o = mesh_object(name, verts, faces)
    if not rim:
        solidify(o, thick, offset=-1.0)
    smooth(o, 50)
    return o


def _band_cols(perim, nb, z0, z1, hole, nh=6, ds_max=None):
    """Columns for nb equal bays; hole(i) -> (half_width, fn(dx) -> (lo, hi)[, dx samples]) centred in bay i."""
    cols = []
    bw = perim / nb
    for i in range(nb):
        left = i * bw
        h = hole(i)
        a, fn = h[0], h[1]
        xs = h[2] if len(h) > 2 else [-a * cos(pi * k / nh) for k in range(nh + 1)]
        c = left + bw / 2
        g = bw / 2 - a
        m = max(1, int(ceil(g / ds_max))) if ds_max else 1
        for t in range(m):
            cols.append((left + g * t / m, [z0, z1], None))
        for dx in xs:
            lo, hi = fn(dx)
            cols.append((c + dx, [z0, lo, hi, z1], i))
        for t in range(1, m):
            cols.append((c + a + g * t / m, [z0, z1], None))
    return cols


def _slot(zb, zt, a):            # round-ended slot (a circle when zt - zb == 2a)
    def fn(dx):
        cp = sqrt(max(a * a - dx * dx, 0.0))
        return zb + a - cp, zt - a + cp
    return a, fn


def _quatre(zc, r, c, sub=2):    # quatrefoil: four lobes of radius r centred c from the middle
    a = c + r
    xc = (c + sqrt(max(2 * r * r - c * c, 0.0))) / 2      # cusp between a side lobe and the top lobe
    pc = acos(clamp((xc - c) / r, -1.0, 1.0))
    qc = asin(clamp(xc / r, -1.0, 1.0))
    half = [c + r * cos(pc * k / sub) for k in range(sub)] + [xc] + [r * sin(qc * (sub - k) / sub) for k in range(1, sub)]
    xs = [-x for x in half] + [0.0] + list(reversed(half))

    def fn(dx):
        x = abs(dx)
        e = sqrt(max(r * r - (x - c) ** 2, 0.0))
        if x < r:
            e = max(e, c + sqrt(r * r - x * x))
        return zc - e, zc + e
    return a, fn, xs


def _lancet(zb, zs, a):          # flat sill, pointed (equilateral) head springing at zs
    def fn(dx):
        return zb, zs + sqrt(max(4 * a * a - (abs(dx) + a) ** 2, 0.0))
    return a, fn


def _map_round(perim, rfn):
    def f(s_, z, d=0.0):
        r = rfn(z) - d
        a = 2 * pi * s_ / perim
        return (r * cos(a), r * sin(a), z)
    return f


def _map_poly(perim, n, rfn):
    def f(s_, z, d=0.0):
        u = (s_ / perim * n) % n
        k = int(floor(u + 1e-7))
        t = u - k
        a0, a1 = 2 * pi * k / n, 2 * pi * (k + 1) / n
        r = rfn(z) - d * 1.15
        return (r * lerp(cos(a0), cos(a1), t), r * lerp(sin(a0), sin(a1), t), z)
    return f


def _bar(name, stations, T, sec, cap=True):
    """Solid bar: stations = [(P, out, (ka, kb))]; sec = closed section [(a, b)]: vertex = P + a*ka*T + b*kb*out.
    T is the fixed 'across' direction, out the per-station direction pointing away from the flame."""
    verts, n = [], len(sec)
    for P, o, k in stations:
        for a, b in sec:
            verts.append((P[0] + a * k[0] * T[0] + b * k[1] * o[0], P[1] + a * k[0] * T[1] + b * k[1] * o[1],
                          P[2] + a * k[0] * T[2] + b * k[1] * o[2]))
    faces = []
    m = len(stations)
    for i in range(m - 1):
        for j in range(n):
            j2 = (j + 1) % n
            faces.append((i * n + j, i * n + j2, (i + 1) * n + j2, (i + 1) * n + j))
    if cap:
        faces.append(tuple(range(n - 1, -1, -1)))
        faces.append(tuple(range((m - 1) * n, m * n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _rod_sec(w, d=0.5):          # chamfered flat bar: back at b = 0 (towards the glass), every other face leans outwards
    return [(w / 2, 0.0), (0.21 * w, d * w), (-0.21 * w, d * w), (-w / 2, 0.0)]


def _lin(h, k=1.0):              # '#rrggbb' -> linear rgb * k
    c = [int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
    return tuple(((x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4) * k for x in c)


def _lit_tint(base, flame="#ffb060"):
    """colour of a surface of colour `base` in the flame's light, normalised (the brightness is set by glow)"""
    c = [y * (0.12 + x) for x, y in zip(_lin(flame), _lin(base))]
    m = max(c)
    return tuple(round(v / m, 4) for v in c)


def _split_lit(o, lp, sm=None, thr=-0.3, reach=1e9):
    """The lamp throws no shadows inside its own housing (shadow_near) and sits a hand's breadth from the metalwork,
    so every face that looks at the flame - or nearly does: the normal maps tip grazing faces into the light - burns
    out to white, and the stem, chain and ceiling rose are lit straight through the cap. Those faces are moved to a
    second object, which the caller gives a very dark metal: its whole response to the lamp scales with the tint, so
    it is 'exposed' for the flame. The lamp lies well behind every face that stays (or further off than `reach`).
    -> (kept, lit), either may be None"""
    if sm is None:
        sm = o.get("tsq_smooth", 0.0)
    ev = None
    if len(o.modifiers):
        ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
    else:
        me = o.data
    M = o.matrix_basis.copy()                            # (matrix_world is stale until the depsgraph runs)
    M3 = M.to_3x3()
    V = [tuple(M @ v.co) for v in me.vertices]
    lpv = Vector(lp)
    keep, lit = [], []
    for f in me.polygons:
        d = lpv - (M @ f.center)
        dist = max(d.length, 1e-9)
        n = (M3 @ f.normal).normalized()
        (lit if n.dot(d) / dist > thr and dist < reach else keep).append(list(f.vertices))
    if ev is not None:
        ev.to_mesh_clear()
    if not lit:
        return o, None
    name = o.name
    delete(o)
    out = []
    for suffix, faces in (("", keep), ("_lit", lit)):
        if not faces:
            out.append(None)
            continue
        used = sorted({i for f in faces for i in f})
        rm = {i: k for k, i in enumerate(used)}
        q = mesh_object(name + suffix, [V[i] for i in used], [[rm[i] for i in f] for f in faces])
        if sm:
            smooth(q, sm)
        else:
            flat(q)
        out.append(q)
    return out[0], out[1]


def _prof_z(prof, r):            # height of a (r, z) profile at radius r (linear, first matching span)
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        if min(r0, r1) <= r <= max(r0, r1) and abs(r1 - r0) > 1e-9:
            return lerp(z0, z1, (r - r0) / (r1 - r0))
    return prof[-1][1]


def _band_z(z0, half):           # mask: 1 at height z0 fading to 0 at +-half
    return lambda co, n: np.clip(1.0 - np.abs(co[:, 2] - z0) / half, 0.0, 1.0)


def _near_pts(Q, rad, down=2.2):  # mask: blotch round each point, drawn out downwards (rust bleeding from rivets)
    Q = np.array(Q, dtype=float).reshape(-1, 3)

    def fn(co, n):
        d = co[:, None, :] - Q[None, :, :]
        dz = d[:, :, 2]
        dz = np.where(dz < 0, dz / down, dz)
        dist = np.sqrt(d[:, :, 0] ** 2 + d[:, :, 1] ** 2 + dz ** 2).min(axis=1)
        return np.clip(1.0 - dist / rad, 0.0, 1.0)
    return fn


def _streaks(k, seed):           # mask: vertical streaks round the Z axis
    def fn(co, n):
        a = np.arctan2(co[:, 1], co[:, 0])
        return np.clip(0.5 + 0.5 * np.sin(k * a + 2.4 * np.sin(3.0 * a + seed) + seed * 1.7), 0.0, 1.0)
    return fn


def build(p, rng):
    s = p.size / 0.26
    R = p.size * 0.5
    age = p.age
    sd = p.seed
    # frame iron, rings, painted sheet (cap / pan / plates), pierced galleries, accent, scroll stays, finial,
    # glowing liners behind the piercings, flame-side faces (near / above the cap), burner, glass, flame
    FR, RINGS, CAPS, GALS, AC, STAY, DR, LINER, WARM, WARM2, GLOWP, BURN, GL, FL = ([] for _ in range(14))
    fixed = []                                            # parts that stay on the ceiling
    bell = []
    knops = []                                            # brass knops above the cap
    zt = -p.drop                                          # top of the lantern's loop
    r_j = rng.fork("irregular")

    # ------------------------------------------------------------------ ceiling fixing + hook
    if p.fixing == "rose":
        fixed.append(lathe("rose", [(0.0, -0.037), (0.008, -0.037), (0.0115, -0.034), (0.0115, -0.029), (0.018, -0.026),
                                    (0.024, -0.021), (0.026, -0.016), (0.040, -0.0135), (0.056, -0.0115), (0.064, -0.0095),
                                    (0.0665, -0.006), (0.064, -0.003), (0.06, 0.0)], segments=24))
        sc = bolt("rose_screw", 0.0055, 0.003, kind="dome")
        a0 = r_j.u(0, 120)
        fixed.append(instances_to_mesh("rose_screws", sc, [trs((0.049 * cos(radians(a0 + k * 120)), 0.049 * sin(radians(a0 + k * 120)), -0.0118),
                                                               (180, 0, r_j.u(0, 90))) for k in range(3)], keep_base=False))
    else:
        fixed.append(box("ceil_plate", (0.125, 0.125, 0.007), loc=(0, 0, -0.0035), rot=(0, 0, r_j.jitter(0, 3)), bevel=0.0018))
        fixed.append(lathe("plate_boss", [(0.0, -0.037), (0.008, -0.037), (0.0115, -0.034), (0.0115, -0.028), (0.02, -0.024),
                                          (0.027, -0.017), (0.027, -0.010), (0.022, -0.006)], segments=8, smooth_angle=30))
        nl = bolt("plate_nail", 0.0075, 0.0055, kind="pyramid")
        fixed.append(instances_to_mesh("plate_nails", nl, [trs((sx * 0.047, sy * 0.047, -0.007), (180, 0, r_j.u(0, 90)))
                                                           for sx in (-1, 1) for sy in (-1, 1)], keep_base=False))
    hw = 0.0045
    hook_c = -0.089
    hp = [(0, 0, -0.026), (0, 0, -0.056)]
    for ph in (40, 70, 100, 135, 180, 225, 270):
        hp.append((0.017 * sin(radians(ph)), 0, hook_c + 0.017 * cos(radians(ph))))
    hp += [(-0.0185, 0, -0.079), (-0.024, 0, -0.070)]
    fixed.append(sweep("hook", hp, hw, sides=7, step=0.007, scale_fn=lambda t: 1.0 - 0.5 * t ** 5))
    hook_top = hook_c - 0.017 + hw                       # top of the hook wire in the bowl = pivot of the swing
    sup_top = hook_top

    swing_iron = []                                      # chain, rod, bell stem
    # ------------------------------------------------------------------ lantern loop
    lr, lw = 0.019 * s, 0.0045 * s
    under = zt - 2 * lw                                  # underside of the loop wire at its top
    loop_rot = (90, 0, 0)                                # loop in the XZ plane

    # ------------------------------------------------------------------ smoke bell
    bell_len = 0.012 + 0.135 * s + 0.026
    if p.smoke_bell and (sup_top - under) > bell_len + 0.16:
        e, ew = 0.012, 0.004
        ze = under - ew + e                              # bottom eye (YZ plane) linked through the lantern loop
        swing_iron.append(torus("bell_eye", e, ew, loc=(0, 0, ze), rot=(0, 90, 0), major_seg=14, minor_seg=6))
        zr = ze + e + 0.135 * s
        swing_iron.append(cylinder("bell_stem", 0.0036, zr - ze - 2 * e + 0.008, loc=(0, 0, (zr + ze) / 2), verts=8))
        zb = ze + e + 0.045 * s
        bl = _turn("smoke_bell", BELL, zb, s, s, seg=22)
        bell.append(bl)
        knops.append(_turn("bell_knop", [(0.004, -0.004), (0.011, -0.002), (0.013, 0.004), (0.009, 0.010), (0.004, 0.012)], zb + 0.064 * s, s, s, seg=12))
        ring_c = zr + 0.011
        bell_ring = ("bell_ring", 0.013, 0.0042, ring_c)
        under = ring_c + 0.013 - 2 * 0.0042
    else:
        bell_ring = None

    # ------------------------------------------------------------------ forged hanger rod for long drops
    G = sup_top - under
    if G > 0.6:
        e, ew = 0.0165, 0.005
        z1 = sup_top - (e - ew)                          # top eye (YZ plane) on the hook
        z2 = under + 0.22 + (e - ew)                     # bottom eye (XZ plane): 0.22 m of chain below
        swing_iron.append(torus("rod_eye_t", e, ew, loc=(0, 0, z1), rot=(0, 90, 0), major_seg=16, minor_seg=6))
        swing_iron.append(torus("rod_eye_b", e, ew, loc=(0, 0, z2), rot=(90, 0, 0), major_seg=16, minor_seg=6))
        za, zb = z1 - e + 0.002, z2 + e - 0.002
        Lr = za - zb
        bow = r_j.jitter(0, 0.004)
        h = 0.0054                                       # 11 mm square bar, twisted hot
        rod = sweep("hanger_rod", [(0, 0, za), (bow * 0.6, 0, za - Lr * 0.25), (bow, 0, za - Lr * 0.5), (bow * 0.6, 0, za - Lr * 0.75), (0, 0, zb)],
                    [(-h, -h), (h, -h), (h, h), (-h, h)], step=0.012, twist_deg=360.0 * max(1, round(Lr / 0.12)), up=(1, 0, 0), smooth_path=True)
        flat(rod)
        swing_iron.append(rod)
        sh = [(0.0076, -0.009), (0.0104, -0.005), (0.0104, 0.005), (0.0076, 0.009)]
        swing_iron.append(_turn("rod_shoulder_t", sh, za - 0.006, 1, 1, seg=8))
        swing_iron.append(_turn("rod_shoulder_b", sh, zb + 0.006, 1, 1, seg=8))
        swing_iron.append(_turn("rod_knop", [(0.0076, -0.022), (0.0115, -0.018), (0.0108, -0.012), (0.0165, -0.006), (0.0185, 0.0), (0.0165, 0.006),
                                     (0.0108, 0.012), (0.0115, 0.018), (0.0076, 0.022)], (za + zb) / 2, 1, 1, seg=12, loc=(bow, 0, 0)))
        sup_top = z2 - e + ew
        G = sup_top - under

    # ------------------------------------------------------------------ chain (links sized so it hangs taut hook-to-loop)
    L0 = 0.06
    N = max(2, int(round(G / (0.72 * L0))))
    L = G / (0.72 * N)
    w = 0.07 * L
    zc0, zcN = sup_top - L / 2 + 2 * w, under + L / 2 - 2 * w
    cpath = [(0, 0, zc0), (0, 0, (zc0 + zcN) / 2), (0, 0, zcN)]
    n_links = len(resample_path(cpath, step=L * 0.72, smooth_path=True))
    ch = chain("chain", cpath, link_len=L, wire=w, sides=4)
    co = np.array([v.co[:] for v in ch.data.vertices])
    top = co[co[:, 2] > zc0 + L * 0.1]
    if len(top) and np.ptp(top[:, 0]) > np.ptp(top[:, 1]):   # first link must lie in the YZ plane (hook is in XZ)
        transform_verts(ch, lambda c: Vector((-c.y, c.x, c.z)))
    swing_iron.append(ch)
    last_in_yz = (n_links % 2 == 1)
    end_rot = (90, 0, 0) if last_in_yz else (0, 90, 0)   # what the last link passes through lies in the other plane
    if bell_ring:
        swing_iron.append(torus(bell_ring[0], bell_ring[1], bell_ring[2], loc=(0, 0, bell_ring[3]), rot=end_rot, major_seg=14, minor_seg=6))
    else:
        loop_rot = end_rot
    loop = torus("loop", lr, lw, loc=(0, 0, zt - lw - lr), rot=loop_rot, major_seg=18, minor_seg=7)
    stem = _turn("stem", STEM, zt, s, s, seg=16)
    zc = zt - 0.086 * s                                  # top of the cap / roof / crown

    # ------------------------------------------------------------------ heights, lamp position
    if p.style == "cylinder":
        zg1 = zc - 0.130 * s
        zg0 = zg1 - 0.060 * s
        zr1 = zg0 + 0.003 * s                             # top of the top ring
        zA = zr1 - 0.040 * s                              # glass visible from here ...
        hg = 0.23 * s
        zB = zA - hg                                      # ... down to here
        z_fl = zB + 0.36 * hg
    elif p.style == "hex":
        zrf = zc - 0.032 * s
        zg1 = zrf - 0.098 * s
        zg0 = zrf - 0.150 * s
        zr1 = zg0 + 0.003 * s
        zA = zr1 - 0.036 * s
        hg = 0.24 * s
        zB = zA - hg
        z_fl = zB + 0.36 * hg
    else:
        zg1 = zc - 0.050 * s
        zg0 = zg1 - 0.042 * s
        zcl = zg0 + 0.003 * s
        th_t = asin(0.43)
        th_b = pi - asin(0.40)
        zg = zcl - 0.008 * s - R * cos(th_t)               # globe centre
        z_fl = zg - 0.04 * s
    z_light = z_fl + 0.026 * s                             # light socket in the flame, as far from the glass as it gets
    lp = (0.0, 0.0, z_light)

    def put(lst, o, sm=45, far=None):
        """File a finished mesh under its material; its flame-side faces go to the flame-exposed material
        (far = list for the parts above the cap, which get the less dark one; the ceiling fixing is only
        split when a short drop brings it within reach of the lamp)."""
        if p.lit:
            o, w = _split_lit(o, lp, sm, thr=-0.3 if far is None else -0.15, reach=FAR_REACH * sqrt(p.intensity / 35.0) if far is fixed_w else 1e9)
            if w is not None:
                (WARM if far is None else far).append(w)
        if o is not None and lst is not None:
            lst.append(o)
        return o

    fixed_w = []
    loop = put(FR, loop, None, far=WARM2)
    stem = put(AC, stem, None, far=WARM2)
    for o in knops:
        put(AC, o, None, far=WARM2)
    bell[:] = [q for q in (put(CAPS, o, None, far=WARM2) for o in bell) if q is not None]
    swing_iron[:] = [q for q in (put(None, o, None, far=WARM2) for o in swing_iron) if q is not None]
    fixed[:] = [q for q in (put(None, o, None, far=fixed_w) for o in fixed) if q is not None]

    # ------------------------------------------------------------------ shared small parts
    nut_base = lathe("nut", [(0.0068 * s, 0.0), (0.0072 * s, 0.005 * s), (0.0048 * s, 0.0095 * s), (0.0, 0.0125 * s)], segments=6)
    nut_m = []
    rivet_base = lathe("rivet", [(0.0042 * s, 0.0), (0.0027 * s, 0.0011 * s), (0.0, 0.0019 * s)], segments=6)
    rivet_m = []
    cap_riv = []                                         # rivet positions that bleed rust on the painted sheet

    def dent(o, k=1.0):
        if age > 0.05:
            displace_noise(o, strength=0.0007 * s * age * k, scale=0.045 * s, seed=sd + 7)
        return o

    def burner(z_floor, r_floor, z_edge, seg=14):
        """flame-lit pan floor, brass oil font, burner with wick wheel, flat-wick flame"""
        zn = max(z_fl - 0.02 * s, z_floor + 0.037 * s)
        prof = [(r * s, z_floor + dz * s) for r, dz in FONT]
        prof += [(0.012 * s, zn - 0.004 * s), (0.02 * s, zn), (0.021 * s, zn + 0.004 * s), (0.014 * s, zn + 0.011 * s),
                 (0.0095 * s, zn + 0.013 * s), (0.0085 * s, z_fl + 0.002 * s), (0.0, z_fl + 0.002 * s)]
        BURN.append(lathe("burner", prof, segments=14))
        aw_ = r_j.u(0, 360)
        ca, sa = cos(radians(aw_)), sin(radians(aw_))
        BURN.append(cylinder("wick_spindle", 0.0016 * s, 0.03 * s, loc=(0.02 * s * ca, 0.02 * s * sa, zn + 0.002 * s), rot=(0, 90, aw_), verts=5))
        BURN.append(cylinder("wick_wheel", 0.0075 * s, max(0.0024 * s, 0.0042), loc=(0.036 * s * ca, 0.036 * s * sa, zn + 0.002 * s), rot=(0, 90, aw_), verts=10))
        WARM.append(lathe("floor", [(r_floor, z_edge), (0.026 * s, z_floor + 0.001 * s)], segments=seg))
        if p.lit:
            off = r_j.vec(0.0015)
            fl = lathe("flame", [(r * s, z_fl + dz * s) for r, dz in FLAME], segments=10, loc=(off[0], off[1], 0))
            transform_verts(fl, lambda c: Vector((c.x, c.y * 0.55, c.z)))
            FL.append(fl)

    # ================================================================== CYLINDER
    if p.style == "cylinder":
        Rb = R * (1.0 - p.taper)
        z_eave = zc - 0.133 * s
        cap = put(CAPS, dent(_turn("cap", CAP, zc, R, s, seg=24)))
        a0 = r_j.u(0, 30)
        for k in range(12):                               # the spun cap is riveted to the gallery band
            a = radians(a0 + 30.0 * k)
            q = (R * cos(a), R * sin(a), zc - 0.1172 * s)
            cap_riv.append(q)
            rivet_m.append(trs(q, (0, 63, degrees(a))))
        rg = 0.9 * R
        per = 2 * pi * rg
        zm_ = (zg0 + zg1) / 2 - 0.001 * s
        gal = put(GALS, _strip("gallery_top", _band_cols(per, 18, zg0, zg1, lambda i: _quatre(zm_, 0.0108 * s, 0.0078 * s)),
                               _map_round(per, lambda z: rg), 0.003 * s, rim=True), 50)
        LINER.append(cylinder("vent_glow", rg - 0.0055 * s, 0.056 * s, loc=(0, 0, (zg0 + zg1) / 2), verts=18, cap=False))
        put(RINGS, _turn("ring_top", RING, zr1, R, s, seg=24))

        def rfn(z):
            return lerp(R, Rb, clamp((zA - z) / hg))
        put(RINGS, lathe("ring_bot", [(r * Rb, zB - (0.040 + dz) * s) for r, dz in reversed(RING)], segments=24))
        nr = 9
        zs = [lerp(zB - 0.004 * s, zA + 0.004 * s, i / (nr - 1)) for i in range(nr)]
        GL.append(lathe("glass", [(0.85 * rfn(z), z) for z in zs], segments=24))
        WARM.append(lathe("ceiling", [(0.0, zA + 0.026 * s), (0.875 * R, zA + 0.0045 * s)], segments=14))
        nb = int(p.bars)
        zra, zrb = zr1 + 0.001 * s, zB - 0.041 * s
        zh = max(lerp(zB, zA, 0.5), z_light + 0.012 * s) + abs(r_j.jitter(0, 0.012 * s))
        rr = 0.955
        wr = 0.0064 * s                                   # chamfered flat guard bars, back to the glass
        sec = _rod_sec(wr)
        for k in range(nb):
            a = radians(360.0 * k / nb + (0 if nb % 2 == 0 else 90))
            ca, sa = cos(a), sin(a)
            bend = r_j.jitter(0, 0.0012 * s)
            st = []
            for i in range(5):
                z = lerp(zra, zrb, i / 4.0)
                r = rr * rfn(z) - 0.002 * s + bend * sin(pi * i / 4.0)
                st.append(((r * ca, r * sa, z), (ca, sa, 0.0), (1.0, 1.0)))
            put(FR, _bar(f"rod{k}", st, (-sa, ca, 0.0), sec), 30)
            nut_m.append(trs(st[0][0], (0, 0, degrees(a))))
            nut_m.append(trs(st[-1][0], (180, 0, degrees(a))))
            rh_ = rr * rfn(zh) - 0.002 * s + 0.5 * wr + 0.003 * s
            rivet_m.append(trs((rh_ * ca, rh_ * sa, zh), (0, 90, degrees(a))))
        hi_ = rr * rfn(zh) - 0.002 * s + 0.5 * wr - 0.0006 * s
        ho_, hh_ = hi_ + 0.0032 * s, 0.0065 * s
        hoop = put(FR, lathe("hoop", [(hi_, zh - hh_), (ho_, zh - hh_ * 0.8), (ho_ + 0.0006 * s, zh), (ho_, zh + hh_ * 0.8), (hi_, zh + hh_),
                                      (hi_, zh - hh_)], segments=28, smooth_angle=35), 35)
        zbr = zB - 0.040 * s                              # underside of the bottom ring
        zl1 = zbr + 0.003 * s
        zl0 = zl1 - 0.046 * s
        rl = 0.9 * Rb
        perl = 2 * pi * rl
        al = min(0.0085 * s, perl / 24 * 0.29)
        gal2 = put(GALS, _strip("gallery_bot", _band_cols(perl, 24, zl0, zl1, lambda i: _slot(zl0 + 0.007 * s, zl1 - 0.009 * s, al), nh=4),
                                _map_round(perl, lambda z: rl), 0.003 * s, rim=True), 50)
        LINER.append(cylinder("inlet_glow", rl - 0.0055 * s, 0.042 * s, loc=(0, 0, (zl0 + zl1) / 2), verts=18, cap=False))
        skirt = put(CAPS, lathe("skirt", [(0.872 * Rb, zl0 - 0.0015 * s), (0.905 * Rb, zl0 - 0.0015 * s), (0.914 * Rb, zl0 + 0.003 * s),
                                          (0.904 * Rb, zl0 + 0.008 * s)], segments=24))
        pan = put(CAPS, dent(_turn("pan", PAN, zl0, Rb, s, seg=24)))      # its rim tucks up inside the gallery skirt
        zf = zl0 - 0.058 * s
        drip = put(DR, _turn("drip", DRIP, zf, s, s, seg=14))
        # forged scroll stays: palm riveted to the bottom ring, out round the gallery, riveted under the pan, curled at the finial
        ns = nb if nb <= 5 else (nb // 2 if nb % 2 == 0 else 3)
        wS = 0.0064 * s
        hS = 0.36 * wS
        key = [(0.985 * Rb, zbr + 0.018 * s), (1.04 * Rb, zbr + 0.003 * s), (1.15 * Rb, zbr - 0.016 * s), (1.185 * Rb, zbr - 0.034 * s),
               (1.13 * Rb, zl0 - 0.012 * s), (0.93 * Rb, zl0 - 0.027 * s), (0.62 * Rb, zl0 - 0.0395 * s), (0.36 * Rb + 0.013 * s, zl0 - 0.066 * s)]
        m2 = [(q[0], q[2]) for q in resample_path([(r, 0.0, z) for r, z in key], n=11, smooth_path=True)]
        tr_, tz_ = m2[-1][0] - m2[-2][0], m2[-1][1] - m2[-2][1]
        th0 = atan2(-tr_, tz_)
        rho0, rho1, nsp = 0.019 * s, 0.0055 * s, 14
        C = (m2[-1][0] - rho0 * cos(th0), m2[-1][1] - rho0 * sin(th0))
        path = list(m2)
        for i in range(1, nsp + 1):
            t = i / nsp
            th = th0 + 2 * pi * 1.25 * t
            rho = lerp(rho0, rho1, t ** 0.85)
            path.append((C[0] + rho * cos(th), C[1] + rho * sin(th)))
        cum = [0.0]
        for (r0, z0), (r1, z1) in zip(path, path[1:]):
            cum.append(cum[-1] + sqrt((r1 - r0) ** 2 + (z1 - z0) ** 2))
        prof_st = []
        for i, (r, z) in enumerate(path):
            pa, pb = path[max(i - 1, 0)], path[min(i + 1, len(path) - 1)]
            dr, dz = pb[0] - pa[0], pb[1] - pa[1]
            dl = max(sqrt(dr * dr + dz * dz), 1e-9)
            f = smoothstep(0.82, 1.0, cum[i] / cum[-1])      # fishtail end: thinner in the curl plane, wider across
            g = 1.0 - smoothstep(0.0, 0.035, cum[i] / cum[-1])   # palm flattened on to the ring
            prof_st.append((r, z, -dz / dl, dr / dl, (1.0 + 0.8 * f + 0.9 * g, 1.0 - 0.6 * f - 0.45 * g)))
        ssec = [(wS / 2, 0.0), (0.2 * wS, hS), (-0.2 * wS, hS), (-wS / 2, 0.0), (-0.2 * wS, -hS), (0.2 * wS, -hS)]
        i_pan = min(range(len(m2)), key=lambda i: abs(m2[i][0] - 0.62 * Rb))
        for k in range(ns):
            a = radians(360.0 * k / ns + (0 if nb % 2 == 0 else 90))
            ca, sa = cos(a), sin(a)
            put(STAY, _bar(f"stay{k}", [((r * ca, r * sa, z), (nr_ * ca, nr_ * sa, nz_), kk) for r, z, nr_, nz_, kk in prof_st],
                           (-sa, ca, 0.0), ssec), 30)
            r0_, z0_ = m2[0][0] + 0.002 * s, m2[0][1] - 0.001 * s
            rivet_m.append(trs((r0_ * ca, r0_ * sa, z0_), (0, 68, degrees(a)), 1.25))
            rp_, zp_ = m2[i_pan]
            rivet_m.append(trs((rp_ * ca, rp_ * sa, zp_ - hS), (180, 0, 0), 1.25))
        burner(zB - 0.012 * s, 0.88 * Rb, zB - 0.0045 * s)
        z_bottom = zf - 0.088 * s
        top_o, top_z = cap, zc
        low_parts = [pan, gal2, skirt]
        glass_top = zA

    # ================================================================== HEX
    elif p.style == "hex":
        Rb = R * (1.0 - p.taper)
        put(CAPS, _turn("vent", HEX_VENT, zc, s, s, seg=16))
        z_eave = zrf - 0.108 * s
        roof = put(CAPS, dent(_turn("roof", HEX_ROOF, zrf, R, s, seg=6, smooth_angle=40), 0.6), 40)
        # hip rolls run out over the eave corners, each finished with a turned acorn drop
        hip = [(r * R, zrf + dz * s + 0.0025 * s) for r, dz in reversed(HEX_ROOF[3:10])]
        hip += [(1.262 * R, zrf - 0.1055 * s), (1.268 * R, zrf - 0.115 * s)]
        for k in range(6):
            a = radians(60.0 * k)
            ca, sa = cos(a), sin(a)
            put(FR, sweep(f"hip{k}", [(r * ca, r * sa, z) for r, z in hip], 0.0046 * s, sides=5, step=0.016 * s,
                          scale_fn=lambda t: 1.0 - 0.3 * smoothstep(0.8, 1.0, t)))
            put(AC, _turn(f"acorn{k}", ACORN, zrf - 0.116 * s, s, s, seg=8, loc=(1.262 * R * ca, 1.262 * R * sa, 0)), 50)
        rg = 0.8 * R
        per = 6 * rg
        gal = put(GALS, _strip("gallery_top", _band_cols(per, 18, zg0, zg1, lambda i: _slot(zg0 + 0.006 * s, zrf - 0.115 * s, 0.0085 * s), nh=4),
                               _map_poly(per, 6, lambda z: rg), 0.003 * s, rim=True), 50)
        LINER.append(cylinder("vent_glow", 0.655 * R, 0.05 * s, loc=(0, 0, (zg0 + zg1) / 2), verts=12, cap=False))
        put(CAPS, _turn("rail_top", HEX_RAIL, zr1, R, s, seg=6, smooth_angle=40), 40)

        def rfn(z):
            return lerp(R, Rb, (zA - z) / hg)
        # bottom rail, weathered so steeply that none of its outside looks up at the flame
        glowp = p.lit and p.metal == "iron"             # painted plates and rail pick up the flame light (see materials)
        rail = lathe("rail_bot", [(r * Rb, zB + dz * s) for r, dz in HEX_SILL], segments=6, smooth_angle=40)
        rail = put(None, rail, 40)
        (GLOWP if glowp else CAPS).append(rail)
        # face plates: one arched opening per side, cut from thick sheet (reveals and backs are flame-lit)
        rf = 0.95
        per = 6 * rf * R
        aw = rf * R / 2 - 0.017 * s
        z0p, z1p = zB - 0.010 * s, zA + 0.028 * s
        zsill = zB + 0.016 * s
        zspr = zA - 0.016 * s - 1.732 * aw
        fp = put(None, _strip("faces", _band_cols(per, 6, z0p, z1p, lambda i: _lancet(zsill, zspr, aw), nh=8),
                         _map_poly(per, 6, lambda z: rf * rfn(z)), 0.0045 * s), 50)
        (GLOWP if glowp else CAPS).append(fp)
        rgl = lambda z: rf * rfn(z) - 0.008 * s
        nr = 7
        zs = [lerp(zB - 0.004 * s, zA + 0.004 * s, i / (nr - 1)) for i in range(nr)]
        GL.append(lathe("glass", [(rgl(z), z) for z in zs], segments=6, smooth_angle=40))
        WARM.append(lathe("ceiling", [(0.0, zA + 0.024 * s), (rgl(zA), zA + 0.0045 * s)], segments=6))
        zd = zB - 0.040 * s - 0.022 * s                   # corner posts run on below the bottom rail as drops
        psec = _rod_sec(0.0105 * s, 0.45)
        for k in range(6):
            a = radians(60.0 * k)
            ca, sa = cos(a), sin(a)
            r0, r1 = 0.975 * rfn(zA + 0.004 * s) - 0.003 * s, 0.975 * rfn(zd) - 0.003 * s
            put(FR, _bar(f"post{k}", [((r0 * ca, r0 * sa, zA + 0.004 * s), (ca, sa, 0.0), (1.0, 1.0)),
                                      ((r1 * ca, r1 * sa, zd), (ca, sa, 0.0), (1.0, 1.0))], (-sa, ca, 0.0), psec), 30)
            nut_m.append(trs(((r1 + 0.003 * s) * ca, (r1 + 0.003 * s) * sa, zd + 0.002 * s), (180, 0, degrees(a)), 1.4))
            for zz in (zsill + 0.012 * s, zA - 0.008 * s):   # rivets fixing the plates either side of each post
                for sg in (-1, 1):
                    rv = rf * rfn(zz)
                    a2 = a + radians(60.0) * sg
                    px, py = rv * lerp(ca, cos(a2), 0.13), rv * lerp(sa, sin(a2), 0.13)
                    rivet_m.append(trs((px, py, zz), (0, 90, degrees(a) + 30 * sg)))
                    if zz < zB + 0.05 * s:
                        cap_riv.append((px, py, zz))
        zbr = zB - 0.040 * s
        pan = put(CAPS, dent(_turn("pan", HEX_PAN, zbr, Rb, s, seg=6, smooth_angle=40), 0.6), 40)
        zf = zbr - 0.054 * s
        put(AC, _turn("pan_boss", [(0.016, -0.012), (0.026, -0.009), (0.03, -0.003), (0.03, 0.004), (0.022, 0.008)], zf, s, s, seg=14))
        drip = put(DR, _turn("drip", DRIP, zf - 0.008 * s, s, s, seg=14))
        burner(zB - 0.012 * s, rgl(zB), zB - 0.0045 * s, seg=6)
        z_bottom = zf - 0.096 * s
        top_o, top_z = roof, zrf
        low_parts = [pan]
        glass_top = zA

    # ================================================================== GLOBE
    else:
        z_eave = zc - 0.052 * s
        cap = put(CAPS, dent(_turn("cap", GLOBE_CAP, zc, s, s, seg=24), 0.6))
        rg = 0.058 * s
        per = 2 * pi * rg
        gal = put(GALS, _strip("gallery_top", _band_cols(per, 10, zg0, zg1, lambda i: _slot(zg0 + 0.0065 * s, zg1 - 0.007 * s, 0.0088 * s), nh=4),
                               _map_round(per, lambda z: rg), 0.003 * s, rim=True), 50)
        LINER.append(cylinder("vent_glow", 0.0525 * s, 0.04 * s, loc=(0, 0, (zg0 + zg1) / 2), verts=12, cap=False))
        put(RINGS, _turn("collar", GLOBE_COLLAR, zcl, s, s, seg=24))
        ths = [lerp(th_b, th_t, i / 13) for i in range(14)]
        GL.append(lathe("glass", [(R * sin(t), zg + R * cos(t)) for t in ths], segments=28, smooth_angle=60))
        WARM.append(lathe("ceiling", [(0.0, zg + R * cos(th_t) + 0.012 * s), (0.43 * R, zg + R * cos(th_t) + 0.0005 * s)], segments=12))
        zcup = zg - 0.898 * R
        cup = put(CAPS, dent(_turn("cup", GLOBE_CUP, zcup, s, s, seg=24), 0.6))
        zf = zcup - 0.066 * s
        drip = put(DR, _turn("drip", DRIP, zf, s, s, seg=14))
        nb = int(p.bars)
        rc = R + 0.0008 * s
        wr = 0.0062 * s
        sec = _rod_sec(wr)
        for k in range(nb):
            a = radians(360.0 * k / nb + (0 if nb % 2 == 0 else 90))
            ca, sa = cos(a), sin(a)
            st = [((0.064 * s * ca, 0.064 * s * sa, zcl - 0.008 * s), (ca, sa, 0.0), (1.0, 1.0))]
            for i in range(11):
                t = lerp(radians(33), radians(150), i / 10)
                st.append(((rc * sin(t) * ca, rc * sin(t) * sa, zg + rc * cos(t)), (sin(t) * ca, sin(t) * sa, cos(t)), (1.0, 1.0)))
            st.append(((0.061 * s * ca, 0.061 * s * sa, zcup + 0.004 * s), (ca, sa, 0.0), (1.0, 1.0)))
            put(FR, _bar(f"rib{k}", st, (-sa, ca, 0.0), sec), 30)
            rb_ = rc + 0.5 * wr + 0.0028 * s
            rivet_m.append(trs((rb_ * ca, rb_ * sa, zg), (0, 90, degrees(a))))
        ei_ = rc + 0.5 * wr - 0.0006 * s
        eo_, eh_ = ei_ + 0.0032 * s, 0.0065 * s
        put(FR, lathe("equator", [(ei_, zg - eh_), (eo_, zg - eh_ * 0.8), (eo_ + 0.0006 * s, zg), (eo_, zg + eh_ * 0.8), (ei_, zg + eh_),
                                  (ei_, zg - eh_)], segments=28, smooth_angle=35), 35)
        rp_ = rc + 0.5 * wr + 0.0012 * s
        for lat in (38, -40):
            put(FR, torus("parallel", rp_ * cos(radians(lat)), 0.0027 * s, loc=(0, 0, zg + rp_ * sin(radians(lat))),
                          major_seg=28, minor_seg=6))
        burner(zcup - 0.012 * s, 0.40 * R, zg + R * cos(th_b) + 0.0005 * s)
        z_bottom = zf - 0.088 * s
        top_o, top_z = cap, zc
        low_parts = [cup]
        glass_top = zg + R * 0.9

    if nut_m:
        put(AC, instances_to_mesh("nuts", nut_base, nut_m, keep_base=False), 50)
    else:
        delete(nut_base)
    if rivet_m:
        put(FR, instances_to_mesh("rivets", rivet_base, rivet_m, keep_base=False), 50)
    else:
        delete(rivet_base)

    # ------------------------------------------------------------------ materials
    if p.metal == "iron":
        m_frame = mat("iron_painted", "#565c63", rough=1.15)                           # forged bar and chain: blacklead gone grey
        m_cap = mat("iron_painted", p.paint, rough=1.4)                               # brushed-on graphite paint, gone dull
        m_gal = mat("iron_painted", _mix_hex(p.paint, "#a9aaa4", 0.35), rough=0.85)    # stove-enamelled galleries and rings keep some sheen
        m_acc = mat("brass", rough=0.95)
        m_stay, m_drip, m_chain = m_cap, m_cap, m_frame
        base = p.paint
    elif p.metal == "brass":
        m_frame = mat("brass", rough=1.25)
        m_cap = mat("brass", "#eadcbc", rough=1.3)
        m_gal = m_cap
        m_acc = mat("bronze_aged")
        m_stay, m_drip, m_chain = m_frame, m_acc, m_frame
        base = "#b9924a"
    else:
        m_frame = mat("brass", "#f2c9a0", rough=1.25)
        m_cap = mat("copper_patina")
        m_gal = m_cap
        m_acc = mat("brass", rough=0.95)
        m_stay, m_drip, m_chain = m_frame, m_acc, mat("iron_wrought", rough=1.2)
        base = "#a9764e"
    m_ring = m_gal if p.metal == "iron" else m_frame
    if not p.lit:
        m_glass = mat("glass_frosted", p.glass_tint, glow=0.0) if p.frosted else mat("glass_lantern", p.glass_tint, glow=0.0, opacity=0.35)
        m_liner = m_warm = m_warm2 = m_cap
        m_burn = m_acc
    else:
        # the lamp also lights the inside of the far pane, which shows through the near one: dense, softly glowing
        # glass keeps that to a warm fill behind the flame instead of a white card
        if p.frosted:
            m_glass = mat("glass_lantern", p.glass_tint, glow=OPAL_GLOW, opacity=clamp(1.0 - OPAL_CLEAR * s * s, 0.93, 0.975), rough=2.5)
        else:
            m_glass = mat("glass_lantern", p.glass_tint, glow=PANE_GLOW, opacity=clamp(1.0 - PANE_CLEAR * (0.6 if p.style == "hex" else 1.0) * s * s, 0.9, 0.965), grime=0.0, rough=2.5)
        m_liner = mat("emissive", _mix_hex("#ff9438", p.glass_tint, 0.3), glow=LINER_GLOW)   # the bright inside seen through the piercings
        # metal in the flame's light: a metal's whole response scales with its tint, so a nearly black one in the
        # colour of the lit paint / brass is correctly exposed a hand's breadth from the flame (and reads as soot by day)
        lt = _lit_tint(base)
        m_warm = mat("metal_plain", tuple(round(v * WARM_K, 5) for v in lt), rough=1.3)
        m_warm2 = mat("metal_plain", tuple(round(v * WARM2_K, 5) for v in (1.0, 0.62, 0.33)), rough=1.3)
        m_burn = mat("emissive", "#d9993a", glow=BURN_GLOW)                                  # brass font right under the flame
    if GLOWP:
        gm = mat("emissive", p.paint, glow=PAINT_GLOW)
        for o in GLOWP:
            assign(o, gm)
        m_gal = m_ring = m_cap                            # (keeps the material count at 10)
    for lst, m in ((FR, m_frame), (RINGS, m_ring), (CAPS, m_cap), (GALS, m_gal), (AC, m_acc), (STAY, m_stay), (DR, m_drip),
                   (LINER, m_liner), (WARM, m_warm), (WARM2, m_warm2), (BURN, m_burn), (GL, m_glass)):
        for o in lst:
            assign(o, m)
    for o in FL:
        assign(o, mat("emissive", "#ffcf8a", glow=12))
    for o in fixed_w:
        assign(o, m_warm2)
    for o in swing_iron + [f for f in fixed if f.name.startswith("hook")]:
        assign(o, m_chain)
    for o in fixed:
        if not o.name.startswith("hook"):
            assign(o, m_frame if p.fixing == "plate" or p.metal != "iron" else m_cap)
    if p.fixing == "rose":
        for o in fixed:
            if o.name.startswith("rose_screws"):
                assign(o, m_acc)

    # ------------------------------------------------------------------ art-directed ageing
    A = clamp(0.3 + 0.9 * age)                            # how far the weathering has got
    # rust / verdigris creeping up from the drip edge and down from the rivets, bloom on the weather side
    paint(top_o, special=mask_mul(_band_z(z_eave, 0.04 * s), noise_mask(0.035 * s, seed=sd + 10, lo=0.3, hi=0.62), 0.9 * A),
          wear=mask_mul(_band_z(z_eave, 0.016 * s), noise_mask(0.03 * s, seed=sd + 12, lo=0.45, hi=0.62), A),
          grime=mask_mul(height_ramp(top_z + 0.02 * s, top_z - 0.08 * s), 0.3 + 0.5 * age))
    paint(top_o, special=mask_mul(noise_mask(0.06 * s, seed=sd + 1, lo=0.45, hi=0.72), facing((0, 0, 1)), 0.7 * A))
    if cap_riv:
        for o in CAPS:
            paint(o, special=mask_mul(_near_pts(cap_riv, 0.04 * s), noise_mask(0.02 * s, seed=sd + 16, lo=0.2, hi=0.6), min(1.0, 1.2 * A)))
    paint(gal, grime=mask_mul(height_ramp(zg1, zg1 - 0.07 * s), 0.25 + 0.45 * age),
          special=mask_mul(noise_mask(0.04 * s, seed=sd + 13, lo=0.45, hi=0.72), 0.8 * A))
    for o in low_parts:                                   # condensation and lamp oil collect in the pan: rust from its rim down
        paint(o, special=mask_mul(noise_mask(0.05 * s, seed=sd + 2, lo=0.3, hi=0.66), 0.9 * A), grime=0.15 + 0.3 * age,
              wear=mask_mul(noise_mask(0.025 * s, seed=sd + 14, lo=0.5, hi=0.7), 0.7 * A))
    for o in CAPS:
        if o is not top_o and o not in low_parts:
            paint(o, special=mask_mul(noise_mask(0.045 * s, seed=sd + 15, lo=0.35, hi=0.68), 0.8 * A), grime=0.1 + 0.2 * age)
    for o in FR + STAY + RINGS:
        paint(o, special=mask_mul(noise_mask(0.035, seed=sd + 3, lo=0.4, hi=0.75), height_ramp(z_bottom, z_bottom + 0.5 * s), A),
              wear=mask_mul(noise_mask(0.03, seed=sd + 4, lo=0.4, hi=0.8), 0.5))
    if p.style == "cylinder":                             # hoop rubbed bright where the glass is lifted out past it
        paint(hoop, wear=mask_mul(noise_mask(0.03 * s, seed=sd + 21, lo=0.3, hi=0.6), 0.9))
    for o in (stem, drip):                                # tarnished, handled only when the lamplighter trims the wick
        paint(o, grime=0.35 + 0.3 * age, wear=mask_mul(noise_mask(0.02, seed=sd + 8, lo=0.4, hi=0.8), 0.6))
    if p.metal != "iron":                                 # brass / copper left to tarnish, polished only on the edges
        for o in FR + RINGS + CAPS + GALS + STAY:
            paint(o, grime=mask_mul(noise_mask(0.06 * s, seed=sd + 9, lo=0.15, hi=0.85), 0.3 + 0.35 * age), mode="max")
    for o in swing_iron:
        paint(o, special=mask_mul(noise_mask(0.06, seed=sd + 5, lo=0.4, hi=0.75), 0.15 + 0.7 * age))
    for o in fixed:
        paint(o, grime=0.25 + 0.3 * age, special=mask_mul(noise_mask(0.03, seed=sd + 6, lo=0.4, hi=0.8), 0.8 * A))
    # bare, burnished metal where the loop, the links and the hook bear on each other
    for o in swing_iron + [loop] + [f for f in fixed if f.name.startswith("hook")]:
        paint(o, wear=lambda co, n: np.maximum(np.clip(1.0 - np.abs(co[:, 2] - hook_top) / 0.012, 0.0, 1.0),
                                               np.clip(1.0 - np.abs(co[:, 2] - (zt - lw)) / (0.012 * s), 0.0, 1.0)) * 0.95)
    # heat and soot from the vent on the collar, loop and smoke bell above the cap
    for o in [stem, loop] + bell:
        paint(o, grime=0.5 + 0.4 * age, mode="max")

    # opal glass: soot streaks creeping down from under the cap
    if p.frosted:
        streak = _streaks(7, sd)

        def soot_fn(co, n):
            t = np.clip((co[:, 2] - (glass_top - 0.1 * s)) / (0.1 * s), 0.0, 1.0)
            return t ** 1.3 * (0.35 + 0.65 * streak(co, n)) * (0.3 + 0.7 * age)
        for o in GL:
            paint(o, grime=soot_fn)

    # ------------------------------------------------------------------ light, swing part
    # shadow_near reaches just past the finial: the lantern's own cage, pan and scrolls throw no shadows
    if p.lit:
        light("lamp", lp, color=_mix_hex("#ffbf6a", p.glass_tint, 0.35), intensity=p.intensity,
              range=clamp(5.0 + p.intensity * 0.07, 5.0, 11.0), flicker=0.14, shadow=True, shadow_near=(z_light - z_bottom) + 0.02)
    swing = FR + RINGS + CAPS + GALS + AC + STAY + DR + LINER + WARM + WARM2 + GLOWP + BURN + GL + FL + swing_iron
    fixed += fixed_w
    part(swing, "lantern", pivot=(0, 0, hook_top), motion="swing", amp=2.5, speed=0.28)
