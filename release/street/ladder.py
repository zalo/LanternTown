"""Lamplighter's wooden ladder leaning against a wall (or standing as an A-frame step-ladder).
Two riven, slightly bowed stiles that taper towards the top, spindle-turned round rungs through-tenoned
and wedged into them, iron tie rods with washers and nuts under some rungs, iron-shod feet cut level
with the paving, sacking pads roped round the stile tops, a rope-whipped splint repair, a wiping rag and
the lamplighter's small brass hand lantern on a hook. The A-frame style swaps rungs for housed, nailed
treads, adds a cap board, a bolted rear prop frame with nailed rails and rope spreaders.
"""
from tk import *
import numpy as np
from math import sin, cos, tan, radians, pi, hypot, ceil, floor, exp

META = {"name": "Wooden ladder", "class": "prop_small", "mount": "wall_ground", "tri_budget": 8000,
        "focus": 0.5}
PARAMS = {
    "length": {"default": 3.0, "min": 1.8, "max": 5.0, "desc": "length of the stiles (m)"},
    "rungs": {"default": 10, "min": 3, "max": 18, "desc": "number of rungs / treads (kept to a climbable 0.2-0.36 m spacing)"},
    "lean": {"default": 15.0, "min": 6.0, "max": 30.0, "desc": "lean angle from the wall (deg from vertical)"},
    "style": {"default": "tapering", "choices": ["tapering", "straight", "a_frame"], "desc": "tapering pole ladder / parallel ladder / step-ladder A-frame"},
    "width": {"default": 0.46, "min": 0.34, "max": 0.62, "desc": "outside width at the feet (m)"},
    "wood": {"default": "#c9ad80", "desc": "timber tint of the stiles"},
    "paint": {"default": "#7a2b22", "desc": "colour of the painted stile ends"},
    "paint_ends": {"default": 0.34, "min": 0.0, "max": 3.0, "desc": "length of the painted band at both stile ends (0 = bare, >= length/2 = fully painted)"},
    "tie_rods": {"default": 3, "min": 0, "max": 5, "desc": "iron tie rods under rungs"},
    "lashing": {"default": 1, "min": 0, "max": 2, "desc": "rope-whipped splint repairs on the stiles"},
    "pads": {"default": True, "desc": "sacking pads roped round the stile tops (leaning styles)"},
    "iron_shoes": {"default": True, "desc": "iron-shod feet"},
    "rag": {"default": True, "desc": "wiping rag draped over a rung"},
    "lantern": {"default": True, "desc": "lamplighter's hand lantern on the side hook"},
    "lantern_cd": {"default": 1.2, "min": 0.5, "max": 6.0, "desc": "hand lantern intensity (cd, a candle; turned down automatically close to the wall)"},
    "rope_coil": {"default": True, "desc": "spare lashing rope looped over a hook on the left stile"},
    "wear": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "age: bow, worn rung centres, rust, dirt"},
}
VARIANTS = {
    "straight_short": {"style": "straight", "length": 2.2, "rungs": 7, "lean": 11, "width": 0.4, "wood": "#b99670",
                       "paint_ends": 0.0, "lantern": False, "rope_coil": False, "lashing": 0, "tie_rods": 2, "wear": 0.25, "seed": 4},
    "step_ladder": {"style": "a_frame", "length": 2.1, "rungs": 6, "lean": 17, "width": 0.56, "paint": "#2f5140",
                    "paint_ends": 0.28, "tie_rods": 2, "lashing": 0, "wear": 0.4, "seed": 11},
    "tall_worn": {"length": 4.4, "rungs": 15, "lean": 12, "width": 0.52, "wood": "#c9c2b4", "paint": "#2c4257",
                  "paint_ends": 0.45, "tie_rods": 5, "lashing": 2, "wear": 1.0, "seed": 23},
    "painted_green": {"style": "straight", "length": 3.4, "rungs": 12, "lean": 19, "paint": "#41694d", "paint_ends": 3.0,
                      "pads": False, "rope_coil": False, "lashing": 0, "tie_rods": 4, "wear": 0.65, "seed": 31},
}


class _Frame:
    """Members are modelled upright (x across, front = -y, z along the stile) and tilted about X."""

    def __init__(self, ang, oy):
        self.ang, self.oy = ang, oy
        self.c, self.s = cos(radians(ang)), sin(radians(ang))
        self.up = Vector((0.0, self.s, self.c))          # world up in frame coordinates
        self.down = -self.up

    def w(self, x, y, z):
        return (x, y * self.c - z * self.s + self.oy, y * self.s + z * self.c)

    def cut(self, y, h=0.0):
        """frame z at which a point of depth y reaches world height h (level cuts)."""
        return (h - y * self.s) / self.c

    def seat(self, o):
        apply_transform(o)
        place(o, loc=(0, self.oy, 0), rot=(self.ang, 0, 0))
        return o


def _sec(w, d, cf, cb=None):
    """chamfered rectangle, 8 points CCW seen from +Z; cf = front (-y) chamfer, cb = back chamfer."""
    cb = cf if cb is None else cb
    a, b = w / 2, d / 2
    return [(a - cf, -b), (a, -b + cf), (a, b - cb), (a - cb, b), (-a + cb, b), (-a, b - cb), (-a, -b + cf), (-a + cf, -b)]


def _loft(name, rings, cap0=True, cap1=True):
    n = len(rings[0])
    verts = [tuple(v) for r in rings for v in r]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap0:
        faces.append(tuple(reversed(range(n))))
    if cap1:
        faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _stations(length, extra, step=0.28):
    n = max(6, int(round(length / step)))
    zs = [length * i / n for i in range(n + 1)]
    for e in extra:
        if 0.05 < e < length - 0.05:
            zs = [zs[0]] + [z for z in zs[1:-1] if abs(z - e) > 0.06] + [e, zs[-1]]
    return sorted(zs)


def _stile(name, F, length, cfn, w, dfn, foot_h, extra, top_level=None, cap0=True):
    zs = _stations(length - (0.0 if top_level is not None else 0.008), extra)
    rings = []
    for i, z in enumerate(zs):
        cx, cy = cfn(z)
        pts = _sec(w, dfn(z), min(0.009, w * 0.27), 0.004)
        if i == 0:
            rings.append([(cx + x, cy + y, F.cut(cy + y, foot_h)) for x, y in pts])
        elif i == len(zs) - 1 and top_level is not None:
            rings.append([(cx + x, cy + y, F.cut(cy + y, top_level)) for x, y in pts])
        else:
            rings.append([(cx + x, cy + y, z) for x, y in pts])
    if top_level is None:                                   # eased, hand-rounded top
        cx, cy = cfn(length)
        rings.append([(cx + x, cy + y, length) for x, y in _sec(w - 0.014, dfn(length) - 0.016, 0.006, 0.004)])
    o = _loft(name, rings, cap0=cap0)
    smooth(o, 50)
    uv_grain(o, axis="Z")
    return o


def _sleeve(name, F, cfn, w, dfn, zo, cut_h=None, cap0=False, cap1=False):
    """band / wrap hugging a stile: zo = [(z, offset from the timber), ...]."""
    rings = []
    for i, (z, off) in enumerate(zo):
        cx, cy = cfn(z)
        pts = _sec(w + 2 * off, dfn(z) + 2 * off, 0.009 + off * 0.5, 0.004 + off * 0.5)
        if i == 0 and cut_h is not None:
            rings.append([(cx + x, cy + y, F.cut(cy + y, cut_h)) for x, y in pts])
        else:
            rings.append([(cx + x, cy + y, z) for x, y in pts])
    return _loft(name, rings, cap0=cap0, cap1=cap1)


def _rung(name, pl, pr, stations, sides, up, flat, rot0, sag):
    """spindle rung along X; stations = [(x, radius, tread weight)]; worn flat towards `up`."""
    rings = []
    for x, r, ww in stations:
        t = (x - pl[0]) / (pr[0] - pl[0])
        cy = lerp(pl[1], pr[1], t)
        cz = lerp(pl[2], pr[2], t) - sag * ww
        ring = []
        for i in range(sides):
            a = rot0 + 2 * pi * i / sides
            dy, dz = cos(a), sin(a)
            rr = r - flat * ww * max(0.0, dy * up[1] + dz * up[2]) ** 2
            ring.append((x, cy + rr * dy, cz + rr * dz))
        rings.append(ring)
    o = _loft(name, rings)
    smooth(o, 60)
    uv_grain(o, axis="X")
    return o


def _whip(name, cfn, w, d, z0, turns, rd, down, rng, tail=0.09):
    """rope whipping: a tight helix round a w x d section, finished with a hanging tail."""
    ring = _sec(w + rd, d + rd, 0.011, 0.011)
    n = len(ring)
    k0 = int(rng.u(0, n))
    pts = []
    for i in range(int(round(turns * n)) + 1):
        z = z0 + rd * 1.04 * i / n
        cx, cy = cfn(z)
        x, y = ring[(i + k0) % n]
        pts.append((cx + x, cy + y, z))
    if tail > 0:
        x, y, z = pts[-1]
        cx, cy = cfn(z)
        out = Vector((x - cx, y - cy, 0.0)).normalized()
        e, dn = Vector((x, y, z)), Vector(down)
        pts += [tuple(e + out * 0.007 + dn * 0.012), tuple(e + out * 0.011 + dn * tail * 0.5),
                tuple(e + out * rng.u(0.004, 0.012) + dn * tail)]
    return sweep(name, pts, rd / 2, sides=4, smooth_path=False)


def _rag(name, xc, cy, cz, half, l1, l2, width, rng):
    """cloth draped over a bar of half-depth `half` whose top is at cz (world coordinates): hangs
    l1 in front and l2 behind, on the skew, gathered into folds."""
    r = half + 0.004
    k = 0.7 if half < 0.03 else 1.0                        # round rung vs flat tread shoulders
    zs = cz - r * (1 - 0.7 * k) + 0.004 * (1 - k)
    path = [(cy - r, cz - r), (cy - r, cz - r), (cy - r, cz - r), (cy - r, cz - r), (cy - r * k, zs), (cy, cz + 0.004), (cy + r * k, zs),
            (cy + r, cz - r), (cy + r, cz - r), (cy + r, cz - r), (cy + r, cz - r)]
    hang = [1.0, 0.62, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.62, 1.0]
    n = len(path) - 1
    ph, ph2 = rng.u(0, 6.28), rng.u(0, 6.28)
    skew, sl1, sl2 = rng.jitter(0, 0.025), rng.u(-0.5, 0.5), rng.u(-0.5, 0.5)

    def fn(u, v):
        f = min(v * n, n - 1e-6)
        i = int(f)
        t = f - i
        y, z = lerp(path[i][0], path[i + 1][0], t), lerp(path[i][1], path[i + 1][1], t)
        h = lerp(hang[i], hang[i + 1], t)
        front = v < 0.5
        sg = -1.0 if front else 1.0
        ln = (l1 * (1 + sl1 * (u - 0.5))) if front else (l2 * (1 + sl2 * (u - 0.5)))
        x = xc + (u - 0.5) * width * (1 - 0.16 * h) + skew * h * sg
        fold = sin(u * 13.0 + ph + (0.0 if front else 1.7))
        y += sg * (0.004 + 0.011 * (0.5 + 0.5 * fold) * min(1.0, h * 2.2)) * (1.0 if h > 0 else 0.0)
        x += 0.006 * cos(u * 13.0 + ph) * h
        z -= ln * h + 0.018 * h * h * sin(u * 5.1 + ph2)
        return (x, y, z)

    return surface(name, fn, nu=7, nv=n, thickness=0.0022)


def _hook(name, x0, y0, z0, sx, reach, lean, r=0.0034):
    """forged iron hook on a back plate screwed to the outer face of a stile; returns (objects, hang point)."""
    plate = box(name + "_plate", (0.004, 0.024, 0.06), loc=(x0 + sx * 0.002, y0, z0 + 0.002), rot=(-lean, 0, 0), bevel=0.001, segments=1)
    pts = [(0.0, 0.008), (reach * 0.45, 0.009), (reach - 0.022, 0.0), (reach, -0.012), (reach + 0.014, -0.006), (reach + 0.02, 0.014)]
    hk = sweep(name, [(x0 + sx * a, y0, z0 + b) for a, b in pts], r, sides=5, step=0.013, up=(0, 1, 0),
               scale_fn=lambda t: 1.0 - 0.35 * smoothstep(0.75, 1.0, t))
    scr = cylinder(name + "_screw", radius=0.004, depth=0.002, verts=6, radius2=0.0028)
    sc = instances_to_mesh(name + "_screws", scr, [trs((x0 + sx * 0.004, y0, z0 + 0.002 + dz), (0, 90 * sx, 0)) for dz in (-0.021, 0.022)],
                           keep_base=False)
    return [plate, hk, sc], (x0 + sx * reach, y0, z0 - 0.012 - r)


def _coil(name, P, sx, loops, rng):
    """spare lashing rope made up into a hank: long loops gathered by frapping turns, the eye over a hook."""
    objs = []
    m = 14
    lz0 = rng.u(0.42, 0.5)
    for i in range(loops):
        lz, wy = lz0 * rng.u(0.9, 1.06), rng.u(0.085, 0.125)
        lean_y, ph = rng.jitter(0, 0.03), rng.u(0, 6.28)
        pts = []
        for j in range(m):
            th = 2 * pi * (j + 0.5) / m
            q = (1 - cos(th)) / 2
            hw = (0.011 + (wy / 2 - 0.011) * smoothstep(0.2, 0.55, q)) * min(1.0, 2.2 * max(0.0, 1.0 - q) ** 0.5)
            pts.append((P[0] + sx * ((i - (loops - 1) / 2) * 0.0085 * (1 + 1.5 * q) + 0.005 * sin(th * 2 + ph) * q),
                        P[1] + (hw if th < pi else -hw) + lean_y * q, P[2] + 0.004 - lz * q))
        objs.append(sweep(f"{name}{i}", pts, 0.0046, sides=4, closed=True, smooth_path=False, up=(1, 0, 0)))
    zf = P[2] - lz0 * 0.2
    ax, ay = 0.0046 * (loops + 0.6), 0.02
    objs.append(sweep(name + "_frap", [(P[0] + ax * cos(2 * pi * i / 6), P[1] + ay * sin(2 * pi * i / 6), zf + 0.0095 * i / 6) for i in range(19)],
                      0.0044, sides=4, smooth_path=False))
    return objs


def _lantern(P, brass, glass, flame_m):
    """small cylindrical brass hand lantern hanging by its bail from the point P (world coordinates)."""
    x, y, z = P
    objs = []
    zc = z - 0.0145                                         # carrying ring centre
    ring = torus("lan_ring", major=0.012, minor=0.0022, loc=(x, y, zc), rot=(0, 90, 0), major_seg=10, minor_seg=4)
    ztop = zc - 0.0095                                      # top of the bail, inside the ring
    hb, rb = 0.082, 0.046
    zt = ztop - hb                                          # bail ears = cap rim
    bail = sweep("lan_bail", [(x - rb * cos(pi * i / 8), y, zt + hb * sin(pi * i / 8)) for i in range(9)], 0.0021, sides=4,
                 smooth_path=False, up=(0, 1, 0))
    zb = zt - 0.115                                         # underside of the base
    base = lathe("lan_base", [(0.0, 0.0), (0.046, 0.0), (0.049, 0.004), (0.049, 0.013), (0.041, 0.02), (0.012, 0.021),
                              (0.011, 0.04), (0.006, 0.043), (0.0, 0.043)], segments=12, loc=(x, y, zb))
    cap = lathe("lan_cap", [(0.0, -0.004), (0.046, -0.004), (0.049, 0.0), (0.047, 0.004), (0.03, 0.021), (0.016, 0.029),
                            (0.016, 0.04), (0.023, 0.044), (0.02, 0.051), (0.0, 0.056)], segments=12, loc=(x, y, zt))
    rod = cylinder("lan_rod", radius=0.0019, depth=0.094, verts=4)
    guard = instances_to_mesh("lan_guard", rod, [trs((x + 0.0435 * cos(radians(45 + k * 90)), y + 0.0435 * sin(radians(45 + k * 90)), zb + 0.02 + 0.046))
                                                 for k in range(4)], keep_base=False)
    hoop = torus("lan_hoop", major=0.0435, minor=0.002, loc=(x, y, zb + 0.07), major_seg=12, minor_seg=3)
    objs += [ring, bail, base, cap, guard, hoop]
    for o in objs:
        assign(o, brass)
        paint(o, grime=mask_mul(facing((0, 0, 1)), 0.55), special=mask_mul(noise_mask(0.03, seed=5), 0.35))
    g = cylinder("lan_glass", radius=0.039, depth=0.094, loc=(x, y, zb + 0.02 + 0.047), verts=12, cap=False)
    assign(g, glass)
    fl = lathe("lan_flame", [(0.0, 0.0), (0.0055, 0.005), (0.0065, 0.012), (0.003, 0.024), (0.0, 0.032)], segments=6, loc=(x, y, zb + 0.044))
    assign(fl, flame_m)
    return objs + [g, fl], (x, y, zb + 0.058)


def build(p, rng):
    W = float(p.wear)
    aframe = p.style == "a_frame"
    L = float(p.length)
    lean = clamp(p.lean, 11.0, 24.0) if aframe else float(p.lean)
    cl, sl = cos(radians(lean)), sin(radians(lean))

    wood = mat("wood_oak", p.wood, wear=0.5)
    rwood = mat("wood_oak", "#b58f66")
    endg = mat("wood_endgrain", p.wood)
    iron = mat("iron_wrought")
    rope_m = mat("rope", "#d9c9a6")

    # ---- proportions
    w = 0.032 + 0.0015 * L                                  # stile thickness (x)
    d0 = 0.058 + 0.006 * L                                  # stile depth at the foot
    hb = (p.width - w) / 2                                  # half width between stile centres, foot
    ht = min(hb, max(hb * {"tapering": 0.7, "straight": 1.0, "a_frame": 0.84}[p.style], 0.14))
    shoe_t = 0.003 if p.iron_shoes else 0.0
    pads = bool(p.pads) and not aframe
    pad_t = 0.008 if pads else 0.0
    PL = min(float(p.paint_ends), L / 2)
    full_paint = PL >= L / 2 - 0.02
    paint_m = mat("wood_painted", p.paint, wear=1.6) if PL > 0.03 else None

    def dfn(z):
        return d0 * (1.0 - (0.05 if aframe else 0.15) * clamp(z / L))

    def hf(z):
        return hb + (ht - hb) * clamp(z / L)

    rb_ = rng.fork("bow")
    amp = (0.0012 + 0.004 * W) * (0.4 if aframe else 1.0) * min(1.5, L / 3.0)
    bow = {s: (rb_.jitter(0, amp), amp * rb_.u(0.3, 1.3)) for s in (-1, 1)}

    def cfn(side):
        return lambda z: (side * hf(z) + bow[side][0] * sin(pi * clamp(z / L)), bow[side][1] * sin(pi * clamp(z / L)))

    # ---- frames
    if aframe:
        H = L * cl
        a2 = lean * 0.8
        c2, s2 = cos(radians(a2)), sin(radians(a2))
        wr, dr = 0.03, 0.056
        hp = H - 0.075                                      # pivot bolt height
        zpf, Lrp = hp / cl, hp / c2
        Lr = Lrp + 0.045
        Rf = _Frame(a2, -(0.10 + dr / 2))
        y_p = Rf.w(0, 0, Lrp)[1]
        F = _Frame(-lean, y_p - zpf * sl)
    else:
        H = L * cl
        F = _Frame(-lean, -0.004 - ((dfn(L) / 2 + pad_t + 0.001) * cl + (L + pad_t) * sl))

    body = []          # frame-space objects of the front ladder (seated at the end)

    def paint_faces(o, length):
        if paint_m is None:
            return
        if full_paint:
            assign_faces(o, paint_m, lambda c, n: True)
        else:
            assign_faces(o, paint_m, lambda c, n: c.z < PL - 1e-4 or c.z > length - PL + 1e-4)

    # ---- stiles
    stiles = []
    tl = max(tan(radians(lean)), 0.05)
    hz = min(1.3, H * 0.62) if aframe else clamp(H - 0.42 / tl, 0.62, 1.3)     # lantern bracket height
    zlan = F.cut(0, hz)
    for side in (-1, 1):
        ex = [zlan + d for d in (-0.36, -0.27, -0.19, -0.11, -0.03, 0.07)] if p.lantern and side == 1 else []
        st = _stile(f"stile{side}", F, L, cfn(side), w, dfn, shoe_t, ex + ([PL, L - PL] if paint_m and not full_paint else []),
                    top_level=H if aframe else None, cap0=not p.iron_shoes)
        assign(st, wood)
        paint_faces(st, L)
        assign_faces(st, endg, lambda c, n: abs(n.z) > 0.82)
        stiles.append(st)
        body.append(st)
        if p.iron_shoes:
            sh = _sleeve(f"shoe{side}", F, cfn(side), w, dfn, [(0.0, 0.003), (0.075, 0.003), (0.075, 0.0002)], cut_h=0.0, cap0=True)
            smooth(sh, 40)
            assign(sh, iron)
            sh["_kind"] = "shoe"
            body.append(sh)

    # ---- rungs / treads
    rr = rng.fork("rungs")
    nutw = None
    rung_objs, rung_z = [], []
    nail_m, nut_m = [], []
    world_objs = []
    if not aframe:
        z_first, z_last = 0.27, L - (0.33 if pads else 0.25)
        span = z_last - z_first
        n_lo = int(ceil(span / 0.36)) + 1
        n = int(clamp(int(p.rungs), n_lo, max(n_lo, int(floor(span / 0.2)) + 1)))
        pitch = span / (n - 1)
        fresh = int(rr.u(1, n - 1)) if (W > 0.3 and n > 3) else -1      # one rung replaced not long ago
        for k in range(n):
            zk = z_first + k * pitch + rr.jitter(0, 0.004)
            rung_z.append(zk)
            ends = {}
            for s in (-1, 1):
                cx, cy = cfn(s)(zk)
                ends[s] = (cx, cy + 0.003, zk + rr.jitter(0, 0.0025))
            xl, xr = ends[-1][0] - w / 2 - 0.006, ends[1][0] + w / 2 + 0.006
            xil, xir = ends[-1][0] + w / 2 + 0.004, ends[1][0] - w / 2 - 0.004
            rc = 0.0168 * rr.vary(1.0, 0.05) * (1.06 if k == fresh else 1.0)
            flat = 0.0 if k == fresh else (0.0012 + 0.0035 * W) * rr.u(0.5, 1.0)
            stn = [(xl, 0.0112, 0.0), (xil, 0.0136, 0.0), (lerp(xil, xir, 0.25), rc * 0.95, 0.55), (lerp(xil, xir, 0.5), rc, 1.0),
                   (lerp(xil, xir, 0.75), rc * 0.95, 0.55), (xir, 0.0136, 0.0), (xr, 0.0112, 0.0)]
            pl = (xl, ends[-1][1], ends[-1][2])
            pr = (xr, ends[1][1], ends[1][2])
            o = _rung(f"rung{k}", pl, pr, stn, 10 if n <= 12 else 8, F.up, flat, rr.u(0, 1), 0.0005 + 0.002 * W * rr.u(0, 1))
            assign(o, wood if k == fresh else rwood)        # the replacement is the same pale ash as the stiles
            assign_faces(o, endg, lambda c, nrm: abs(nrm.x) > 0.9)
            o["_fresh"] = 1 if k == fresh else 0
            rung_objs.append(o)
            body.append(o)
        rod_drop = 0.034
    else:
        z_first = 0.27 / cl
        n_lo = int(ceil((L - z_first) / 0.34))
        n = int(clamp(int(p.rungs), n_lo, max(n_lo, int(floor((L - z_first) / 0.21)))))
        pitch = (L - z_first) / n
        for k in range(n):
            zk = z_first + k * pitch
            rung_z.append(zk)
            _, yk, hk = F.w(0, 0, zk)
            hx = hf(zk) + 0.002
            sag = 0.001 + 0.003 * W * rr.u(0.3, 1.0)
            tilt = rr.jitter(0, 0.002)
            pts = [(lerp(-hx, hx, i / 4), yk - 0.012, hk + tilt * (i / 2 - 1) - sag * sin(pi * i / 4)) for i in range(5)]
            o = sweep(f"tread{k}", pts, _sec(0.118, 0.025, 0.007, 0.004), smooth_path=False, up=(0, 0, 1), cap=False)
            smooth(o, 50)
            assign(o, rwood)
            rung_objs.append(o)
            world_objs.append(o)
            for s in (-1, 1):                               # two nails through the stile into each tread end
                for dy in (-0.022, 0.02):
                    xo = cfn(s)(zk)[0] + s * w / 2
                    nail_m.append(trs((xo, yk - 0.004 + dy + rr.jitter(0, 0.004), hk + rr.jitter(0, 0.003)), (0, 90 * s, 0)))
        rod_drop = 0.03

    # ---- iron tie rods under some rungs, washer + square nut on the outer faces
    nt = min(int(p.tie_rods), len(rung_z))
    if nt > 0:
        ks = sorted({int(clamp(round((i + 0.5) * len(rung_z) / nt - 0.5), 0, len(rung_z) - 1)) for i in range(nt)})
        washer = cylinder("washer", radius=0.0135, depth=0.003, loc=(0, 0, 0.0015), verts=8)
        nut = bolt("nut", 0.0085, 0.007, loc=(0, 0, 0.003), kind="square")
        nutw = join([washer, nut], "nutw")
        assign(nutw, iron)
        for k in ks:
            zr = rung_z[k] - rod_drop
            if aframe:
                zr -= 0.012 / cl
            e = {}
            for s in (-1, 1):
                cx, cy = cfn(s)(zr)
                e[s] = (cx + s * w / 2, cy + (0.0 if aframe else 0.008), zr)
                nut_m.append(trs(e[s], (rr.u(0, 90), 90 * s, 0)))
            rings = []
            for s in (-1, 1):
                rings.append([(e[s][0] + s * 0.013, e[s][1] + 0.004 * cos(2 * pi * i / 6), e[s][2] + 0.004 * sin(2 * pi * i / 6)) for i in range(6)])
            rod = _loft(f"tie_rod{k}", rings)
            smooth(rod, 70)
            assign(rod, iron)
            body.append(rod)
        nw = instances_to_mesh("nuts", nutw, nut_m, keep_base=True)
        body.append(nw)
    if nail_m:
        nb = cylinder("nail", radius=0.0045, depth=0.002, loc=(0, 0, 0.001), verts=6, radius2=0.003)
        assign(nb, iron)
        world_objs.append(instances_to_mesh("nails", nb, nail_m, keep_base=False))

    # ---- sacking pads roped round the stile tops (so the ladder does not mark walls and lamp posts)
    rl = rng.fork("lash")
    rope_objs = []
    if pads:
        sack = mat("canvas", "#c9b792")
        for side in (-1, 1):
            z0, z1 = L - 0.23 + rl.jitter(0, 0.015), L
            zw = z0 + 0.06 + rl.jitter(0, 0.01)
            t = pad_t
            zo = [(z0, 0.0005), (z0 + 0.007, t * 0.85)]
            nwr = 5
            hwr = (z1 - 0.012 - z0 - 0.007) / nwr
            for j in range(nwr):                            # overlapping turns of the sacking strip
                a = z0 + 0.007 + j * hwr
                for zz, o_ in ((a + 0.005, t * rl.u(1.0, 1.25)), (a + hwr - 0.002, t * rl.u(0.6, 0.8))):
                    zo.append((zz, t * 0.7 if abs(zz - zw - 0.016) < 0.022 else o_))
            zo += [(z1 + 0.002, t * 0.85), (z1 + t, -0.006)]
            pad = _sleeve(f"pad{side}", F, cfn(side), w, dfn, zo, cap1=True)
            smooth(pad, 55)
            uv_box(pad)
            assign(pad, sack)
            pad["_kind"] = "pad"
            body.append(pad)
            rope_objs.append(_whip(f"pad_rope{side}", cfn(side), w + t * 0.9, dfn(zw) + t * 0.9, zw + 0.004, 3.0, 0.0085, F.down, rl,
                                   tail=rl.u(0.07, 0.13)))

    # ---- rope-whipped splint repair on the back edge of a stile
    used = set()
    for i in range(int(p.lashing)):
        if len(rung_z) < 4:
            break
        for _ in range(8):
            k = int(rl.u(1, len(rung_z) - 2))
            side = rl.pick([-1, 1])
            if (k, side) not in used and (k + 1, side) not in used and (k - 1, side) not in used:
                break
        used.add((k, side))
        za, zb = rung_z[k] - 0.15, rung_z[k + 1] + 0.15
        ts = 0.014
        cf_ = cfn(side)

        def sfn(z, cf_=cf_):
            cx, cy = cf_(z)
            return (cx, cy + dfn(z) / 2 + ts / 2)

        sp = _loft(f"splint{i}", [[(sfn(z)[0] + x, sfn(z)[1] + y, z) for x, y in _sec(w * 0.92, ts, 0.003)] for z in np.linspace(za, zb, 4)])
        smooth(sp, 40)
        uv_grain(sp, axis="Z")
        assign(sp, rwood)
        assign_faces(sp, endg, lambda c, n: abs(n.z) > 0.9)
        paint(sp, wear=0.55, grime=mask_mul(noise_mask(0.1, seed=p.seed + 9), 0.5))
        body.append(sp)

        def wfn(z, cf_=cf_):
            cx, cy = cf_(z)
            return (cx, cy + ts / 2)

        for j, zc in enumerate((za + 0.05, zb - 0.085)):
            rope_objs.append(_whip(f"whip{i}_{j}", wfn, w, dfn(zc) + ts, zc, 3.5, 0.008, F.down, rl, tail=rl.u(0.05, 0.11) if j == 0 else 0.0))
    for o in rope_objs:
        assign(o, rope_m)
        paint(o, grime=mask_mul(noise_mask(0.05, seed=p.seed + 3), 0.35))
        body.append(o)

    # ---- seat the front ladder
    for o in body:
        F.seat(o)

    # ---- A-frame: cap board, bolted rear prop frame, rope spreaders
    if aframe:
        y_top = F.w(0, 0, L)[1]
        xe = ht + w / 2 + 0.035
        capb = sweep("cap_board", [(-xe, y_top + 0.012, H + 0.0145), (0, y_top + 0.012, H + 0.0145), (xe, y_top + 0.012, H + 0.0145)],
                     _sec(0.18, 0.029, 0.008, 0.005), smooth_path=False, up=(0, 0, 1))
        smooth(capb, 50)
        assign(capb, rwood)
        assign_faces(capb, endg, lambda c, n: abs(n.x) > 0.9)
        paint(capb, wear=lambda co, n: np.clip(1.2 - np.abs(co[:, 0]) / (xe * 0.8), 0, 1) * (0.25 + 0.5 * W) * np.clip(n[:, 2], 0, 1),
              grime=mask_mul(noise_mask(0.12, seed=p.seed + 6), 0.45))
        world_objs.append(capb)
        for s in (-1, 1):
            for dy in (-0.03, 0.03):
                nail_ = cylinder("capnail", radius=0.0045, depth=0.002, loc=(s * ht, y_top + 0.012 + dy, H + 0.0295), verts=6, radius2=0.003)
                assign(nail_, iron)
                world_objs.append(nail_)
        socket("top", "attach", (0, y_top + 0.012, H + 0.029), role="prop_top")

        hr_b = hb - (w / 2 + wr / 2) - 0.001
        hr_t = hf(zpf) - (w / 2 + wr / 2) - 0.001

        def rfn(side):
            return lambda z: (side * (hr_b + (hr_t - hr_b) * clamp(z / Lrp, 0, 1.1)), 0.0)

        rear = []
        for side in (-1, 1):
            st = _stile(f"rear_stile{side}", Rf, Lr, rfn(side), wr, lambda z: dr, shoe_t, [PL] if paint_m and not full_paint else [],
                        cap0=not p.iron_shoes)
            assign(st, wood)
            if paint_m is not None:
                assign_faces(st, paint_m, (lambda c, n: True) if full_paint else (lambda c, n: c.z < PL - 1e-4))
            assign_faces(st, endg, lambda c, n: abs(n.z) > 0.82)
            stiles.append(st)
            rear.append(st)
            if p.iron_shoes:
                sh = _sleeve(f"rear_shoe{side}", Rf, rfn(side), wr, lambda z: dr, [(0.0, 0.003), (0.06, 0.003), (0.06, 0.0002)], cut_h=0.0, cap0=True)
                smooth(sh, 40)
                assign(sh, iron)
                sh["_kind"] = "shoe"
                rear.append(sh)
        # nailed rails and a diagonal brace on the wall side of the rear frame
        yr = dr / 2 + 0.008
        rz = [Lr * f for f in (0.11, 0.47, 0.8)]
        rn = []
        for j, z in enumerate(rz):
            hx = rfn(1)(z)[0] + wr / 2 + 0.014
            z += rr.jitter(0, 0.006)
            ra = sweep(f"rail{j}", [(-hx, yr, z + rr.jitter(0, 0.003)), (0, yr, z), (hx, yr, z + rr.jitter(0, 0.003))], _sec(0.016, 0.048, 0.003),
                       smooth_path=False, up=(0, 0, 1))
            assign(ra, rwood)
            assign_faces(ra, endg, lambda c, n: abs(n.x) > 0.9)
            paint(ra, grime=mask_mul(noise_mask(0.15, seed=p.seed + j), 0.5), wear=0.15 * W)
            rear.append(ra)
            for s in (-1, 1):
                rn.append(trs((s * rfn(1)(z)[0], yr + 0.008, z + rr.jitter(0, 0.006)), (-90, 0, 0)))
        xa, xb = -rfn(1)(rz[0])[0], rfn(1)(rz[1])[0]
        br = sweep("brace", [(xa, yr, rz[0] + 0.045), ((xa + xb) / 2, yr, (rz[0] + rz[1]) / 2), (xb, yr, rz[1] - 0.045)], _sec(0.016, 0.04, 0.003),
                   smooth_path=False, up=(0, 1, 0))
        assign(br, rwood)
        paint(br, grime=0.3)
        rear.append(br)
        rn += [trs((xa, yr + 0.008, rz[0] + 0.06), (-90, 0, 0)), trs((xb, yr + 0.008, rz[1] - 0.06), (-90, 0, 0))]
        nb = cylinder("rnail", radius=0.0045, depth=0.002, loc=(0, 0, 0.001), verts=6, radius2=0.003)
        assign(nb, iron)
        rear.append(instances_to_mesh("rail_nails", nb, rn, keep_base=False))
        for o in rear:
            Rf.seat(o)
        # pivot: coach bolt through front and rear stile, dome head outside, washer + nut inside
        pv = []
        if nutw is None:
            washer = cylinder("washer", radius=0.0135, depth=0.003, loc=(0, 0, 0.0015), verts=8)
            nutw = join([washer, bolt("nut", 0.0085, 0.007, loc=(0, 0, 0.003), kind="square")], "nutw")
            assign(nutw, iron)
        for s in (-1, 1):
            xo = hf(zpf) + w / 2
            hd = bolt(f"pivot_head{s}", 0.012, 0.006, loc=(s * xo, y_p, hp), rot=(0, 90 * s, 0), kind="dome")
            assign(hd, iron)
            world_objs.append(hd)
            pv.append(trs((s * (hr_t - wr / 2), y_p, hp), (20, -90 * s, 0)))
        world_objs.append(instances_to_mesh("pivot_nuts", nutw, pv, keep_base=True))
        # rope spreaders through bored holes, stopper knots outside
        hs = H * 0.4
        zf_, zr_ = hs / cl, hs / c2
        for s in (-1, 1):
            xf, xr_ = cfn(s)(zf_)[0], rfn(s)(zr_)[0]
            a0 = F.w(xf, -dfn(zf_) / 2 - 0.006, zf_)
            a1 = F.w(xf, dfn(zf_) / 2, zf_)
            b0 = Rf.w(xr_, -dr / 2, zr_)
            b1 = Rf.w(xr_, dr / 2 + 0.006, zr_)
            pts = [a0] + [tuple(q) for q in catenary(a1, b0, sag=0.03 + 0.02 * rl.u(0, 1), n=8)] + [b1]
            sr = sweep(f"spreader{s}", pts, 0.0045, sides=5, smooth_path=False, up=(1, 0, 0))
            k0 = sphere(f"knot_f{s}", 0.011, loc=a0, segments=6, rings=4)
            k1 = sphere(f"knot_r{s}", 0.011, loc=b1, segments=6, rings=4)
            for o in (sr, k0, k1):
                assign(o, rope_m)
                paint(o, grime=0.35)
                world_objs.append(o)
    if nutw is not None:
        delete([nutw])

    # ---- wiping rag over a rung / tread
    cloth = mat("canvas", "#c9b792")
    if p.rag and rung_z:
        rg = rng.fork("rag")
        heights = [F.w(0, 0, z)[2] for z in rung_z]
        k = min(range(len(rung_z)), key=lambda i: abs(heights[i] - min(1.15, H * 0.5)))
        zk = rung_z[k]
        vp = pitch * cl
        if aframe:
            _, yk, hk = F.w(0, 0, zk)
            xc = -hf(zk) * rg.u(0.2, 0.45)
            rag = _rag("rag", xc, yk - 0.012, hk + 0.0125, 0.059, min(0.14, vp * 0.5), rg.u(0.14, 0.2), 0.19, rg)
        else:
            xc = -hf(zk) * rg.u(0.2, 0.4)
            _, yk, hk = F.w(xc, 0.003, zk)
            rag = _rag("rag", xc, yk, hk + 0.0165, 0.0165, min(0.19, vp * 0.7) * rg.u(0.75, 1.0), rg.u(0.17, 0.26), 0.19, rg)
        assign(rag, cloth)
        paint(rag, grime=mask_mul(noise_mask(0.07, seed=p.seed + 13, lo=0.35, hi=0.75), 0.6))
        world_objs.append(rag)

    # ---- lantern bracket on the right stile and the lamplighter's hand lantern; kept low so the flame is
    #      well clear of the wall (a candle half a metre from pale plaster is plenty)
    zl = zlan
    cxh, cyh = cfn(1)(zl)
    hx0, hy0, hz0 = F.w(cxh + w / 2, cyh, zl)
    hk_objs, P = _hook("hook", hx0, hy0, hz0, 1, 0.24, lean)
    fpos = None
    if p.lantern:
        lan, fpos = _lantern(P, mat("brass", rough=0.85), mat("glass_lantern", glow=1.3), mat("emissive", "#ffb45e", glow=8))
        part(lan, "lantern", pivot=P, motion="swing", amp=2.5, speed=0.35)
        cd = min(float(p.lantern_cd), 9.0 * fpos[1] ** 2)   # wick turned down when it hangs close to the wall
        light("lantern", fpos, color="#ffbf6a", intensity=cd, range=4.5, flicker=0.3, shadow_near=0.13)
    else:
        socket("hook", "attach", P, rot=(0, 0, 90), role="lantern")
    # ---- spare lashing rope looped over a second hook on the left stile
    if p.rope_coil:
        rc_ = rng.fork("coil")
        hc = min(1.95, H * 0.66) if aframe else clamp(H - 0.2 / tl, 0.95, H * 0.8)
        zl = F.cut(0, hc)
        cxh, cyh = cfn(-1)(zl)
        hx1, hy1, hz1 = F.w(cxh - w / 2, cyh, zl)
        o2, P2 = _hook("coil_hook", hx1, hy1, hz1, -1, 0.045, lean)
        hk_objs += o2
        for o in _coil("coil", (P2[0], P2[1], P2[2] + 0.0034), -1, 3, rc_):
            assign(o, rope_m)
            paint(o, grime=mask_mul(noise_mask(0.06, seed=p.seed + 17), 0.5))
            world_objs.append(o)
    for o in hk_objs:
        assign(o, iron)
        world_objs.append(o)

    # ---- weathering
    sd = int(p.seed)
    for st in stiles:
        paint(st, grime=mask_mul(height_ramp(0.0, 0.6), 0.55 + 0.4 * W),
              wear=mask_mul(height_ramp(0.0, 0.36), noise_mask(0.03, seed=sd + 1, lo=0.4, hi=0.58), 0.6 + 0.4 * W))
        if fpos is not None:                                # soot and scorch where the lantern always hangs
            paint(st, grime=lambda co, n: np.clip(1.0 - np.linalg.norm(co - np.array(fpos) - np.array((0, 0, 0.1)), axis=1) / 0.42, 0, 1) ** 0.6)
        # hands slide up and down the stiles between waist and head height
        paint(st, wear=mask_mul(height_ramp(1.35, 0.7), height_ramp(1.45, min(2.2, H)), noise_mask(0.08, seed=sd + 2, lo=0.3, hi=0.7),
                                0.25 + 0.45 * W),
              grime=mask_mul(noise_mask(0.25, seed=sd + 3, lo=0.4, hi=0.75), 0.25 + 0.35 * W),
              special=mask_mul(height_ramp(0.0, 0.25), noise_mask(0.06, seed=sd + 4), 0.5 * W))
    for o in rung_objs:
        if o.get("_fresh"):
            paint(o, wear=0.12, grime=0.1)
            continue
        hw_ = max(hb, 0.15)
        paint(o, wear=lambda co, n, hw_=hw_: np.exp(-(co[:, 0] / (hw_ * 0.55)) ** 2) * (0.3 + 0.65 * W) * (0.55 + 0.45 * np.clip(n[:, 2], 0, 1)),
              grime=lambda co, n, hw_=hw_: np.clip(np.abs(co[:, 0]) / hw_ - 0.45, 0, 1) * (0.5 + 0.5 * W) + np.clip(1.0 - co[:, 2] / 1.2, 0, 1) * 0.35)
    for o in body + world_objs:
        kind = o.get("_kind")
        if kind == "shoe":
            paint(o, special=mask_mul(noise_mask(0.03, seed=sd + 5, lo=0.2, hi=0.6), 0.5 + 0.5 * W), grime=0.5)
        elif kind == "pad":
            paint(o, grime=mask_mul(noise_mask(0.05, seed=sd + 6, lo=0.2, hi=0.7), 0.6 + 0.4 * W))
        elif o.name.startswith(("tie_rod", "nuts", "nails", "hook", "coil_hook", "pivot", "capnail")):
            paint(o, special=mask_mul(noise_mask(0.04, seed=sd + 7, lo=0.25, hi=0.7), 0.3 + 0.6 * W))
