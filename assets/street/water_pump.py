"""Village hand water pump: a cast-iron case pump bolted to an apron of old, uneven flagstones laid nearly flush
with the paving. Octagonal panelled pedestal, fluted column, bolted flange joint, plain working barrel with an
enamel number plate, gadrooned domed cap and finial. The curved lever handle swings on a pin between two cast
cheeks and enters the case through a slot; the spout carries a bucket hook and drips into a hewn stone trough
(lumpy, sagging rim, knocked-off corners, worn overflow notch), a strapped plank trough, or (no trough) a barred
cast drain grate with the pail hung on the spout. The water is modelled: a splash crown and drip ripples under the
spout and a meniscus that follows the hewn wall; the flags carry wet splash zones, puddles and moss in the joints;
rust blisters and runs, flaked paint, scum, litter, lost fixings and a dead-hanging lever grow with `age`.
"""
from math import cos, sin, pi, radians, atan2, hypot, exp

import bpy
import numpy as np
from mathutils import Vector

from tk import *

META = {"name": "Cast-iron hand pump and trough", "class": "prop", "mount": "ground", "focus": 0.5}
PARAMS = {
    "height": {"default": 1.5, "min": 1.15, "max": 1.9, "desc": "overall height to the finial tip (m)"},
    "trough": {"default": "stone", "choices": ["stone", "wood", "none"], "desc": "trough under the spout (none = drain grate)"},
    "paint": {"default": "#357050", "desc": "paint colour of the ironwork"},
    "trough_len": {"default": 0.95, "min": 0.7, "max": 1.4, "desc": "trough length (m)"},
    "handle_len": {"default": 0.85, "min": 0.55, "max": 1.1, "desc": "reach of the lever handle from its pivot (m)"},
    "handle_side": {"default": "right", "choices": ["right", "left", "back"], "desc": "side the handle swings on (seen from the front)"},
    "flutes": {"default": 14, "min": 0, "max": 20, "desc": "flutes on the column (0 = plain)"},
    "bucket": {"default": True, "desc": "zinc pail (stands by the pump, or hangs on the spout hook when there is no trough)"},
    "plate_text": {"default": "1887", "desc": "text on the enamel number plate (empty = no plate)"},
    "brass_trim": {"default": True, "desc": "brass finial and spout nozzle ring"},
    "water_level": {"default": 0.8, "min": 0.25, "max": 0.95, "desc": "how full the trough is (0..1)"},
    "age": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "neglect: rust blisters and runs, moss, scum, leaf litter, lost parts"},
}
VARIANTS = {
    "timber_trough": {"trough": "wood", "paint": "#84231d", "handle_side": "left", "height": 1.38, "trough_len": 1.15,
                      "flutes": 10, "plate_text": "No 4", "seed": 5},
    "drain_only": {"trough": "none", "paint": "#27507f", "height": 1.25, "handle_len": 0.68, "flutes": 0,
                   "plate_text": "AQUA", "seed": 11},
    "tall_civic": {"height": 1.85, "trough_len": 1.35, "paint": "#b3a274", "flutes": 18, "handle_side": "back",
                   "handle_len": 1.0, "age": 0.15, "water_level": 0.92, "plate_text": "1864", "seed": 23},
    "neglected": {"age": 1.0, "paint": "#74858c", "trough_len": 0.75, "water_level": 0.68, "bucket": False,
                  "brass_trim": False, "height": 1.3, "handle_len": 0.72, "handle_side": "left", "flutes": 8, "plate_text": "",
                  "seed": 31},
}


def _put(objs, loc, rz=0.0):
    """Bake the build transform, then move / yaw about the world Z axis and bake again."""
    for o in objs:
        apply_transform(o)
        place(o, loc=loc, rot=(0, 0, rz))
        apply_transform(o)


def _hex(name, r, h):
    """Low-poly hex head sitting on z=0 (the toolkit bolt is ~160 tris; this is ~40)."""
    o = cylinder(name, r, h, loc=(0, 0, h / 2), verts=6, bevel=min(0.0015, h * 0.25), segments=1)
    apply_transform(o)
    return o


def _nail(name, r, h):
    """Low-poly forged nail / rivet head on z=0."""
    o = cylinder(name, r, h, loc=(0, 0, h / 2), verts=6, radius2=r * 0.55)
    apply_transform(o)
    return o


def _rr(w, d, r, nx, ny, cs=3):
    """Rounded rectangle ring (CCW) with subdivided sides so every ring has the same point count."""
    hw, hd = w / 2, d / 2

    def arc(cx, cy, a0):
        return [(cx + r * cos(radians(a0 + 90.0 * k / cs)), cy + r * sin(radians(a0 + 90.0 * k / cs))) for k in range(cs + 1)]

    pts = arc(hw - r, -hd + r, -90)
    pts += [(hw, -hd + r + 2 * (hd - r) * k / ny) for k in range(1, ny)]
    pts += arc(hw - r, hd - r, 0)
    pts += [(hw - r - 2 * (hw - r) * k / nx, hd) for k in range(1, nx)]
    pts += arc(-hw + r, hd - r, 90)
    pts += [(-hw, hd - r - 2 * (hd - r) * k / ny) for k in range(1, ny)]
    pts += arc(-hw + r, -hd + r, 180)
    pts += [(-hw + r + 2 * (hw - r) * k / nx, -hd) for k in range(1, nx)]
    return pts


def _loft(name, rings, cap0=True, cap1=True):
    """Skin a stack of same-count CCW rings [[(x, y, z), ...], ...]; outward normals follow the ring order."""
    m = len(rings[0])
    verts, faces = [], []
    for ring in rings:
        verts += [tuple(q) for q in ring]
    for i in range(len(rings) - 1):
        for j in range(m):
            j1 = (j + 1) % m
            faces.append((i * m + j, i * m + j1, (i + 1) * m + j1, (i + 1) * m + j))
    if cap0:
        c = len(verts)
        verts.append(tuple(float(v) for v in np.mean(np.array(rings[0]), axis=0)))
        faces += [(c, (j + 1) % m, j) for j in range(m)]
    if cap1:
        c = len(verts)
        verts.append(tuple(float(v) for v in np.mean(np.array(rings[-1]), axis=0)))
        b = (len(rings) - 1) * m
        faces += [(c, b + j, b + (j + 1) % m) for j in range(m)]
    return mesh_object(name, verts, faces)


def _slab(name, x0, x1, y0, y1, zt, tx, ty, r, ra=(0.006, 0.015), chips=1, step=0.1, zbot=0.0, flare=0.001, bump=0.001):
    """One hand-laid flag: tilted, slightly dished top, worn-round arris of varying radius, knocked edges."""
    nx, ny = max(2, int(round((x1 - x0) / step))), max(2, int(round((y1 - y0) / step)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ph = (r.u(0, 6.28), r.u(0, 6.28))
    per = [(i, 0) for i in range(nx + 1)] + [(nx, j) for j in range(1, ny + 1)] + [(i, ny) for i in range(nx - 1, -1, -1)] + \
          [(0, j) for j in range(ny - 1, 0, -1)]
    rk = {ij: r.u(*ra) for ij in per}
    for _ in range(chips):
        rk[r.pick(per)] = r.u(0.02, 0.036)

    def out(i, j):
        return (-1.0 if i == 0 else 1.0 if i == nx else 0.0), (-1.0 if j == 0 else 1.0 if j == ny else 0.0)

    def xy(i, j):
        return x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny

    def zat(x, y):
        return zt + tx * (x - cx) + ty * (y - cy) + bump * sin(x * 23.0 + ph[0]) * sin(y * 19.0 + ph[1])

    verts, faces = [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = xy(i, j)
            k = rk.get((i, j), 0.0)
            ox, oy = out(i, j)
            verts.append((x - ox * k, y - oy * k, zat(x, y) - 0.12 * k))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces.append((a, a + 1, a + nx + 2, a + nx + 1))
    n = len(per)
    b0 = len(verts)
    for i, j in per:
        x, y = xy(i, j)
        k = rk[(i, j)]
        ox, oy = out(i, j)
        verts.append((x - ox * k * 0.3, y - oy * k * 0.3, zat(x, y) - 0.45 * k))
    for i, j in per:
        x, y = xy(i, j)
        ox, oy = out(i, j)
        verts.append((x + ox * flare, y + oy * flare, zbot))
    for k in range(n):
        k1 = (k + 1) % n
        t0, t1 = per[k][1] * (nx + 1) + per[k][0], per[k1][1] * (nx + 1) + per[k1][0]
        faces.append((t0, b0 + k, b0 + k1, t1))
        faces.append((b0 + k, b0 + n + k, b0 + n + k1, b0 + k1))
    o = mesh_object(name, verts, faces)
    smooth(o, 75)
    uv_box(o)
    return o


def _film(name, blobs, slabs, margin=0.02, nseg=14):
    """Thin irregular patches (wet stone, puddles, scum) lying on the tilted tops of the slabs they cover: every blob
    (x, y, radius, phase, dz) is clipped to each slab it overlaps so nothing floats over a joint or a lower flag."""
    verts, faces = [], []
    for bx, by, br, ph, dz in blobs:
        loc = [(0.0, 0.0)]
        for frac in (0.55, 1.0):
            for k in range(nseg):
                a = 2 * pi * k / nseg
                rr = br * frac * (1.0 + 0.22 * sin(2 * a + ph) + 0.15 * sin(3 * a + 2.1 * ph) + 0.09 * sin(5 * a + 0.7 * ph))
                loc.append((rr * cos(a), rr * sin(a)))
        tris = []
        for k in range(nseg):
            k1 = (k + 1) % nseg
            tris += [(0, 1 + k, 1 + k1), (1 + k, 1 + nseg + k, 1 + nseg + k1), (1 + k, 1 + nseg + k1, 1 + k1)]
        for x0, x1, y0, y1, zt, tx, ty in slabs:
            if bx + br * 1.5 < x0 or bx - br * 1.5 > x1 or by + br * 1.5 < y0 or by - br * 1.5 > y1:
                continue
            if x1 - x0 < 3 * margin + 0.01 or y1 - y0 < 3 * margin + 0.01:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            pts = []
            for qx, qy in loc:
                x, y = clamp(bx + qx, x0 + margin, x1 - margin), clamp(by + qy, y0 + margin, y1 - margin)
                pts.append((x, y, zt + tx * (x - cx) + ty * (y - cy) + dz))
            remap = {}
            for t in tris:
                a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
                if abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) < 4e-6:
                    continue
                for i in t:
                    if i not in remap:
                        remap[i] = len(verts)
                        verts.append(pts[i])
                faces.append(tuple(remap[i] for i in t))
    if not faces:
        return None
    o = mesh_object(name, verts, faces)
    weld(o, 1e-5)
    uv_box(o)
    return o


def _water(name, w, d, drop, r, amp=0.004, lam=0.045, nseg=20, fine=19, round_r=None, coarse=0.06, shore=None):
    """Still water with real geometry: concentric drip ripples spreading from `drop`, a faint swell further out and
    a meniscus climbing the walls along the `shore` contour (CCW (x, y) points where the water meets the basin) -
    its steep curl and the splash crown are what catch a lantern from any side. Polar mesh round the drip point,
    clamped to the w x d basin (z = 0 level)."""
    hx, hy = w / 2, d / 2
    dx, dy = drop
    rmax = max(hypot(sx * hx - dx, sy * hy - dy) for sx in (-1, 1) for sy in (-1, 1))
    inner = [0.0035, 0.0075, 0.012, 0.017, 0.0225, 0.029]          # fine rings for the splash crown itself
    radii = inner + [inner[-1] + 0.0105 * k for k in range(1, fine - 2)]
    fine = len(radii)
    ks = amp / 0.004
    while radii[-1] < rmax * 1.45:
        radii.append(radii[-1] + coarse)
    ph = r.u(0, 6.28)
    r_fade = radii[fine - 1]

    def fit(x, y):
        x, y = clamp(x, -hx, hx), clamp(y, -hy, hy)
        if round_r:
            q = hypot(x, y)
            if q > round_r:
                x, y = x * round_r / q, y * round_r / q
        return x, y

    def zat(x, y):
        rr = hypot(x - dx, y - dy)
        rip = amp * exp(-rr / 0.12) * cos(2 * pi * rr / lam + ph) * (1.0 - smoothstep(r_fade * 0.8, r_fade, rr))
        r2 = hypot(x - dx - 0.5 * w * 0.37, y - dy * 0.3)                # the last drop, spreading and bouncing off the walls
        rip2 = amp * 0.3 * cos(2 * pi * r2 / (lam * 2.6) + 2.0 * ph) * exp(-r2 / 0.5)
        # the drop that has just landed: rebound bead standing in a small crater with a raised crown
        bead = 0.010 * ks * exp(-(rr / 0.0065) ** 2) - 0.003 * ks * exp(-(rr / 0.014) ** 2) + 0.0035 * ks * exp(-((rr - 0.0215) / 0.0055) ** 2)
        return rip + rip2 + bead

    pts = [(dx, dy, zat(dx, dy))]
    for rr in radii:
        for k in range(nseg):
            a = 2 * pi * k / nseg
            x, y = fit(dx + rr * cos(a), dy + rr * sin(a))
            pts.append((x, y, zat(x, y)))
    tris = [(0, 1 + k, 1 + (k + 1) % nseg) for k in range(nseg)]
    for i in range(len(radii) - 1):
        a0, b0 = 1 + i * nseg, 1 + (i + 1) * nseg
        for k in range(nseg):
            k1 = (k + 1) % nseg
            tris += [(a0 + k, b0 + k, b0 + k1), (a0 + k, b0 + k1, a0 + k1)]
    verts, faces, remap = [], [], {}
    for t in tris:
        a, b, c = pts[t[0]], pts[t[1]], pts[t[2]]
        if abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) < 2e-7:
            continue
        for i in t:
            if i not in remap:
                remap[i] = len(verts)
                verts.append(pts[i])
        faces.append(tuple(remap[i] for i in t))
    if shore:                                   # meniscus: a narrow band curling up the wall all round
        n = len(shore)
        b = len(verts)
        prof = ((0.018, 0.0004), (0.007, 0.002), (0.002, 0.007), (-0.002, 0.014))
        for din, h in prof:
            for j in range(n):
                x, y = shore[j]
                tx, ty = shore[(j + 1) % n][0] - shore[j - 1][0], shore[(j + 1) % n][1] - shore[j - 1][1]
                ll = hypot(tx, ty) or 1.0
                px, py = x - ty / ll * din, y + tx / ll * din
                verts.append((px, py, h + zat(px, py) * clamp(din / 0.018, 0.0, 1.0)))
        for i in range(len(prof) - 1):
            for j in range(n):
                j1 = (j + 1) % n
                faces.append((b + i * n + j, b + (i + 1) * n + j, b + (i + 1) * n + j1, b + i * n + j1))
    o = mesh_object(name, verts, faces)
    weld(o, 1e-5)
    smooth(o, 80)
    uv_planar(o)
    return o


class _Decals:
    """Rust blisters and runs: small irregular shells standing ~1.5 mm proud of the painted casting."""

    def __init__(self):
        self.v, self.f = [], []

    def blob(self, fn, w, h, r, n=9):
        ph = [r.u(0, 6.28) for _ in range(3)]
        b = len(self.v)
        self.v.append(fn(0.0, 0.0))
        for k in range(n):
            a = 2 * pi * k / n
            rr = 1.0 + 0.28 * sin(2 * a + ph[0]) + 0.18 * sin(3 * a + ph[1]) + 0.1 * sin(4 * a + ph[2])
            self.v.append(fn(w * rr * cos(a), h * rr * sin(a)))
        self.f += [(b, b + 1 + k, b + 1 + (k + 1) % n) for k in range(n)]

    def streak(self, fn, w, length, r, n=6):
        ph = r.u(0, 6.28)
        b = len(self.v)
        for i in range(n + 1):
            t = i / n
            hw = w * (0.45 + 0.55 * min(1.0, t * 4.0)) * (1.0 - t) ** 0.55 + 0.0006
            c = w * 0.35 * sin(t * 4.0 + ph) * t
            self.v += [fn(c - hw, -t * length), fn(c + hw, -t * length)]
        for i in range(n):
            a = b + 2 * i
            self.f.append((a, a + 2, a + 3, a + 1))

    def mesh(self, name):
        if not self.f:
            return None
        o = mesh_object(name, self.v, self.f)
        uv_box(o)
        return o


def _bucket(rng, zinc, wood, hung, water_m):
    """Galvanised pail with rolled rim, two swaged beads, riveted ears and a wire bail with a wooden grip."""
    rb, rt, h = 0.10, 0.135, 0.26

    def r_at(z):
        return rb + (rt - rb) * z / h

    prof = [(0.0, 0.012), (rb - 0.008, 0.012), (rb - 0.006, 0.0), (rb, 0.0), (rb + 0.001, 0.008)]
    for zz in (0.07, 0.19):
        prof += [(r_at(zz - 0.008), zz - 0.008), (r_at(zz) + 0.0035, zz), (r_at(zz + 0.008), zz + 0.008)]
    prof += [(rt, h - 0.004), (rt + 0.005, h - 0.002), (rt + 0.0065, h + 0.003), (rt + 0.003, h + 0.007), (rt - 0.002, h + 0.004),
             (rt - 0.002, h - 0.004), (r_at(0.1) - 0.002, 0.1), (r_at(0.02) - 0.002, 0.02), (0.0, 0.02)]
    body = lathe("pail", prof, segments=18)
    displace_noise(body, strength=0.0018, scale=0.09, seed=rng.randrange(1000))
    metal = [body]
    for sx in (-1, 1):
        metal.append(box("pail_ear", (0.007, 0.032, 0.045), loc=(sx * (rt + 0.003), 0, h - 0.028), bevel=0.0015, segments=1))
        rv = _nail("pail_rivet", 0.005, 0.003)
        place(rv, loc=(sx * (rt + 0.0065), 0, h - 0.04), rot=(0, sx * 90, 0))
        metal.append(rv)
    th = 0.0 if hung else radians(96.0)
    hs = 1.25 if hung else 1.0
    Rb = rt + 0.016
    ze = h - 0.02
    pts = [(Rb * cos(a), -Rb * hs * sin(a) * sin(th), ze + Rb * hs * sin(a) * cos(th)) for a in np.linspace(0.0, pi, 13)]
    metal.append(sweep("pail_bail", pts, 0.0038, sides=4))
    grip = cylinder("pail_grip", 0.012, 0.085, loc=(0, -Rb * hs * sin(th), ze + Rb * hs * cos(th)), rot=(0, 90, 0), verts=8, bevel=0.002,
                    segments=1)
    uv_grain(grip, "Z")
    for o in metal:
        assign(o, zinc)
    assign(grip, wood)
    objs = metal + [grip]
    if not hung:
        rw = r_at(0.17)
        w = _water("pail_water", 2 * rw, 2 * rw, (0.025, 0.01), rng, amp=0.001, lam=0.03, nseg=12, fine=9, round_r=rw - 0.003)
        place(w, loc=(0, 0, 0.17))
        assign(w, water_m)
        objs.append(w)
    return objs, Rb * hs + ze


def build(p, rng):
    H = p.height
    zp = 0.028                                  # nominal top of the flags: barely proud of the cobbles
    zj = zp - 0.010                             # sand in the joints
    hb = H - zp                                 # height of the ironwork
    s = clamp((hb / 1.42) ** 0.5, 0.85, 1.15)   # girth scale
    age = p.age
    seed = int(p.seed)
    kind = p.trough
    L = p.trough_len
    broken = age > 0.75                         # bits gone missing on a neglected pump
    txt = str(p.plate_text).strip()

    iron = mat("iron_painted", p.paint, rough=0.8, scale=1.6, wear=0.3)
    rusty = mat("iron_rusty")
    stone_m = mat("stone_rough", "#b3aa99")
    wet_m = mat("stone_rough", "#7a7469", rough=0.3)
    scum = smoothstep(0.45, 1.0, age)
    wtint = tuple(lerp(a, b, scum) for a, b in zip(hex_rgb("#707a78"), hex_rgb("#5d6a2c")))
    water_m = mat("water", wtint, rough=1.8 + 1.5 * scum)
    moss_m = mat("moss")
    brass = mat("brass", rough=1.2) if (p.brass_trim or txt) else None
    zinc = mat("zinc") if p.bucket else None
    wood_m = mat("wood_oak", "#6b5038") if (p.bucket or kind == "wood") else None

    Rc, Rh = 0.075 * s, 0.095 * s
    ped_half = 0.17 * s
    if kind == "stone":
        TW, wall = 0.46 * s, 0.07
    elif kind == "wood":
        TW, wall = 0.42, 0.036
    else:
        TW, wall = 0.0, 0.0
    if kind != "none":
        d_tc = ped_half + 0.015 + TW / 2         # pump axis -> trough centre
        reach = ped_half + 0.015 + wall + 0.075  # pump axis -> spout mouth
        Dy = 0.24 + d_tc + TW / 2 + 0.14
        py = Dy / 2 - 0.24
        ty = py - d_tc
        Wx = max(L + 0.26, 1.05)
    else:
        reach = 0.31 * s
        Dy, Wx = 1.0, 0.95
        py = 0.16
        ty = py - reach
    y_mouth = py - reach
    gy, G = ty - 0.03, 0.138                     # drain grate centre / half size of its pit
    pail_side = 1.0 if p.handle_side == "left" else -1.0
    pail_xy = (pail_side * (ped_half + 0.2), py)

    # ------------------------------------------------------------------ flagstone apron
    ar = rng.fork("apron")
    cells = []                                   # (x0, x1, y0, y1, level)

    def split(x0, x1, y0, y1, depth=0):
        w, h = x1 - x0, y1 - y0
        big, area = max(w, h), w * h
        if big < 0.36 or depth > 4 or area < 0.11 or (area < 0.22 and big < 0.68 and ar.chance(0.5)):
            cells.append((x0, x1, y0, y1, False))
        elif w >= h:
            m = x0 + w * ar.u(0.36, 0.64)
            split(x0, m, y0, y1, depth + 1)
            split(m, x1, y0, y1, depth + 1)
        else:
            m = y0 + h * ar.u(0.36, 0.64)
            split(x0, x1, y0, m, depth + 1)
            split(x0, x1, m, y1, depth + 1)

    if kind != "none":
        ysplit = py - ped_half - 0.05
        xa, xb = -(ped_half + ar.u(0.06, 0.13)), ped_half + ar.u(0.06, 0.13)
        cells.append((xa, xb, ysplit, Dy / 2, True))            # the pump's own flag: level, it carries the base plate
        split(-Wx / 2, xa, ysplit, Dy / 2)
        split(xb, Wx / 2, ysplit, Dy / 2)
        split(-Wx / 2, Wx / 2, -Dy / 2, ysplit)
    else:
        xc = 0.3                                                # drain stones framing the grate pit
        cells.append((-xc, xc, gy + G, Dy / 2, True))
        cells.append((-xc, xc, -Dy / 2, gy - G, False))
        cells.append((-xc, -G, gy - G, gy + G, False))
        cells.append((G, xc, gy - G, gy + G, False))
        split(-Wx / 2, -xc, -Dy / 2, Dy / 2)
        split(xc, Wx / 2, -Dy / 2, Dy / 2)

    flags, slabs = [], []
    for x0, x1, y0, y1, level in cells:
        jw = [ar.u(0.004, 0.0075) for _ in range(4)]
        if x0 <= -Wx / 2 + 1e-6:
            jw[0] = ar.u(0.0, 0.024)
        if x1 >= Wx / 2 - 1e-6:
            jw[1] = ar.u(0.0, 0.024)
        if y0 <= -Dy / 2 + 1e-6:
            jw[2] = ar.u(0.0, 0.024)
        if y1 >= Dy / 2 - 1e-6:
            jw[3] = ar.u(0.0, 0.024)
        x0, x1, y0, y1 = x0 + jw[0], x1 - jw[1], y0 + jw[2], y1 - jw[3]
        if level:
            zt_, tx, ty_ = zp, 0.0, 0.0
        else:
            zt_ = zp + ar.jitter(0.0, 0.005)
            tx, ty_ = ar.jitter(0.0, 0.0035) / ((x1 - x0) / 2), ar.jitter(0.0, 0.0035) / ((y1 - y0) / 2)
        flags.append(_slab("flag", x0, x1, y0, y1, zt_, tx, ty_, ar, chips=ar.pick([0, 1, 1, 2]), zbot=zj - 0.004,
                           step=0.1 * max(1.0, Wx / 1.3)))
        slabs.append((x0, x1, y0, y1, zt_, tx, ty_))
    bed = _slab("bed", -Wx / 2 - 0.004, Wx / 2 + 0.004, -Dy / 2 - 0.004, Dy / 2 + 0.004, zj, 0.0, 0.0, ar, ra=(0.008, 0.02), chips=0,
                step=0.13, zbot=0.0, flare=0.016, bump=0.002)
    if kind == "none":                           # the pit under the grate
        transform_verts(bed, lambda co: Vector((co.x, co.y, min(co.z, 0.002))) if abs(co.x) < 0.27 and abs(co.y - gy) < 0.27 else co)

    def ztop(x, y):
        for x0, x1, y0, y1, zt_, tx, ty_ in slabs:
            if x0 - 0.008 <= x <= x1 + 0.008 and y0 - 0.008 <= y <= y1 + 0.008:
                return zt_ + tx * (x - (x0 + x1) / 2) + ty_ * (y - (y0 + y1) / 2)
        return zp

    # where the water goes: splash under the spout / round the pump foot, the overflow end, the pail's place
    wr = rng.fork("wet")
    if kind != "none":
        wet_blobs = [(L / 2 + 0.06, ty + wr.jitter(0.0, 0.04), 0.2 + 0.05 * age), (wr.jitter(0.03, 0.03), py - ped_half * 0.7, 0.25),
                     (-L * wr.u(0.12, 0.3), ty - TW / 2 - 0.05, 0.12 + 0.04 * age)]
        if p.bucket:
            wet_blobs.append((pail_xy[0] * 0.92, pail_xy[1] - 0.03, 0.15))
        puddles = [(L / 2 + 0.11, ty + wr.jitter(0.0, 0.05), 0.085), (-L * 0.2 + wr.jitter(0.0, 0.05), ty - TW / 2 - 0.07, 0.05)]
    else:
        sd = wr.pick([-1, 1])
        wet_blobs = [(wr.jitter(0.0, 0.03), gy - 0.02, 0.3), (sd * 0.3, gy + 0.14, 0.13)]
        puddles = [(-sd * 0.25, gy - 0.1, 0.075), (sd * 0.05, gy - G - 0.08, 0.05)]

    def wet(co, n):                              # soft damp halo round the wet patches
        m = np.zeros(len(co))
        for bx, by, br in wet_blobs:
            m = np.maximum(m, np.clip(1.0 - (np.hypot(co[:, 0] - bx, co[:, 1] - by) - br * 0.8) / 0.16, 0.0, 1.0))
        return m

    for b in flags:
        assign(b, stone_m)
        tone = ar.u(0.03, 0.3)
        paint(b, grime=lambda co, n, t=tone: np.maximum(t, wet(co, n) * 0.6),
              special=mask_mul(lambda co, n: np.maximum(wet(co, n), 0.35), noise_mask(0.09, seed=seed + 8, lo=0.42, hi=0.7), 0.3 + 0.7 * age),
              wear=mask_mul(facing((0, 0, 1)), noise_mask(0.25, seed=seed + 9), 0.3))
    assign(bed, stone_m)
    paint(bed, grime=0.5, special=mask_mul(noise_mask(0.12, seed=seed + 6, lo=0.3, hi=0.6), 0.35 + 0.65 * age))
    film = _film("wet_film", [(bx, by, br, wr.u(0, 6.28), 0.002 + 0.0004 * i) for i, (bx, by, br) in enumerate(wet_blobs)], slabs)
    if film:
        assign(film, wet_m)
    pud = _film("puddle", [(bx, by, br, wr.u(0, 6.28), 0.0042) for bx, by, br in puddles], slabs, margin=0.014)
    if pud:
        assign(pud, water_m)
        smooth(pud, 80)

    # ------------------------------------------------------------------ pump case (local: axis at origin, z=0 on the apron)
    ped_h = 0.27 * s
    z_fl = 0.50 * hb
    z_cap = hb - 0.21 * s
    head_len = z_cap - z_fl
    zs = z_fl + 0.10 * s                         # spout axis
    z_piv = z_fl + 0.72 * head_len               # handle pivot
    body, dark, gilt, letters = [], [], [], []

    body.append(box("base_plate", (2 * ped_half, 2 * ped_half, 0.024), loc=(0, 0, 0.010), bevel=0.004))
    hd = _hex("hd_bolt", 0.014, 0.012)
    c = ped_half - 0.03
    dark.append(instances_to_mesh("hd_bolts", hd, [trs((sx * c, sy * c, 0.022), (0, 0, rng.u(0, 60))) for sx in (-1, 1) for sy in (-1, 1)],
                                  keep_base=False))
    rb = 0.145 * s
    body.append(lathe("pedestal", [(rb + 0.014, 0.02), (rb + 0.014, 0.05), (rb + 0.004, 0.062), (rb - 0.004, 0.07), (rb - 0.004, ped_h - 0.062),
                                   (rb + 0.006, ped_h - 0.05), (rb + 0.006, ped_h - 0.036), (rb - 0.016, ped_h - 0.022), (Rc + 0.045, ped_h - 0.006),
                                   (Rc + 0.02, ped_h)], segments=8, rot=(0, 0, 22.5), smooth_angle=25))
    apo = (rb - 0.004) * cos(radians(22.5))
    fw = 2 * (rb - 0.004) * sin(radians(22.5))
    ph0, ph1 = 0.088, ped_h - 0.08
    panel = shape_extrude("panel", rounded_rect(fw * 0.74, ph1 - ph0, 0.01, 1),
                          [rounded_rect(fw * 0.74 - 0.03, ph1 - ph0 - 0.03, 0.006, 1)], depth=0.008, bevel=0.0)
    stud = bolt("stud", 0.012, 0.009, kind="pyramid")
    pm, sm = [], []
    lost_stud = rng.randrange(8) if broken else -1
    for k in range(8):
        a = radians(k * 45.0)
        cx, cy = apo * sin(a), -apo * cos(a)
        pm.append(trs((cx, cy, (ph0 + ph1) / 2), (90, 0, k * 45.0)))
        if k != lost_stud:
            sm.append(trs((cx, cy, (ph0 + ph1) / 2), (90, 0, k * 45.0)))
    body.append(instances_to_mesh("panels", panel, pm, keep_base=False))
    body.append(instances_to_mesh("studs", stud, sm, keep_base=False))

    # fluted column (lofted, deep flutes with narrow fillets) between a torus base and an astragal necking
    zt = z_fl - 0.05
    nf = int(p.flutes)
    fz0, fz1 = ped_h + 0.07, zt - 0.06
    d_fl = 0.0085 * s

    def f_env(z):
        if nf <= 0:
            return 0.0
        return min(smoothstep(fz0, fz0 + 0.045, z), 1.0 - smoothstep(fz1 - 0.045, fz1, z))

    body.append(lathe("col_base", [(Rc + 0.034, ped_h - 0.002), (Rc + 0.038, ped_h + 0.012), (Rc + 0.03, ped_h + 0.026), (Rc + 0.012, ped_h + 0.036),
                                   (Rc + 0.01, ped_h + 0.046), (Rc - 0.003, ped_h + 0.058)], segments=24))
    body.append(lathe("col_neck", [(Rc - 0.003, zt - 0.047), (Rc + 0.008, zt - 0.038), (Rc + 0.013, zt - 0.028), (Rc + 0.008, zt - 0.018),
                                   (Rc + 0.003, zt - 0.012), (Rc + 0.003, zt)], segments=24))
    msh = max(28, nf * (4 if nf <= 14 else 3))
    rings = []
    for z in (ped_h + 0.05, fz0, fz0 + 0.02, fz0 + 0.045, (fz0 + fz1) / 2, fz1 - 0.045, fz1 - 0.02, fz1, zt - 0.04):
        e = f_env(z)
        ring = []
        for j in range(msh):
            a = 2 * pi * j / msh
            rr = Rc - d_fl * e * (1.0 - (0.5 + 0.5 * cos(a * nf)) ** 3)
            ring.append((rr * cos(a), rr * sin(a), z))
        rings.append(ring)
    col = _loft("column", rings, cap0=False, cap1=False)
    smooth(col, 62)
    uv_cyl(col, "Z")
    body.append(col)

    # bolted flange joint between column and working barrel
    Rf = Rh + 0.036 * s
    body.append(lathe("flange_lo", [(Rc + 0.003, zt - 0.002), (Rc + 0.012, zt + 0.008), (Rf - 0.012, z_fl - 0.026), (Rf - 0.003, z_fl - 0.022),
                                    (Rf, z_fl - 0.018), (Rf, z_fl - 0.003), (Rf - 0.004, z_fl - 0.001)], segments=24))
    body.append(lathe("flange_up", [(Rf - 0.004, z_fl + 0.001), (Rf, z_fl + 0.003), (Rf, z_fl + 0.018), (Rf - 0.003, z_fl + 0.022),
                                    (Rh + 0.006, z_fl + 0.024), (Rh, z_fl + 0.034)], segments=24))
    fb = _hex("fl_bolt", 0.0105 * s, 0.010)
    rbolt = (Rh + Rf) / 2 + 0.002
    lost_bolt = rng.randrange(6) if broken else -1
    dark.append(instances_to_mesh("fl_bolts", fb, [trs((rbolt * cos(radians(30 + k * 60)), rbolt * sin(radians(30 + k * 60)), z_fl + 0.0225),
                                                       (0, 0, rng.u(0, 60))) for k in range(6) if k != lost_bolt], keep_base=False))

    def bead(z, w=0.009):
        return [(Rh, z - 0.013), (Rh + w * 0.75, z - 0.007), (Rh + w, z), (Rh + w * 0.75, z + 0.007), (Rh, z + 0.013)]

    zb0, zb1 = zs + 0.085 * s, z_cap - 0.045
    # (closed with poles: an open tube gives the lathe no way to tell inside from outside)
    body.append(lathe("barrel", [(0.0, z_fl + 0.03), (Rh, z_fl + 0.03)] + bead(zb0) + bead(zb1) + [(Rh, z_cap - 0.012), (Rh + 0.003, z_cap), (0.0, z_cap)],
                      segments=24))

    # gadrooned dome cap, finial
    Rk = Rh + 0.02 * s
    zn = hb - 0.084 * s
    zd0 = z_cap + 0.034
    dp = [(Rh + 0.002, z_cap - 0.004), (Rk, z_cap + 0.004), (Rk + 0.004, z_cap + 0.014), (Rk, z_cap + 0.024), (Rh + 0.004, z_cap + 0.03),
          (Rh - 0.002, zd0)]
    for i in range(1, 7):
        th = radians(82.0 * i / 6)
        dp.append(((Rh - 0.002) * cos(th) + 0.004 * s * (i / 6), zd0 + (zn - zd0) * sin(th) / sin(radians(82.0))))
    cap = lathe("cap", dp, segments=36, smooth_angle=50)

    def gadroon(co):
        if co.z <= zd0:
            return co
        e = smoothstep(zd0, zd0 + 0.012, co.z) * (1.0 - smoothstep(zn - 0.02, zn, co.z))
        k = 1.0 + 0.085 * e * (abs(cos(atan2(co.y, co.x) * 6.0)) - 0.55)
        return Vector((co.x * k, co.y * k, co.z))

    transform_verts(cap, gadroon)
    body.append(cap)
    if broken:                                   # the finial snapped off long ago: only its seating and a stub are left
        fin = lathe("finial", [(0.017 * s, zn - 0.006), (0.024 * s, zn + 0.004), (0.024 * s, zn + 0.01), (0.014 * s, zn + 0.018),
                               (0.013 * s, zn + 0.026), (0.006 * s, zn + 0.031), (0.0, zn + 0.027)], segments=12)
    else:
        fin = lathe("finial", [(0.017 * s, zn - 0.006), (0.024 * s, zn + 0.004), (0.024 * s, zn + 0.01), (0.014 * s, zn + 0.018), (0.02 * s, zn + 0.028),
                               (0.03 * s, zn + 0.042), (0.031 * s, zn + 0.052), (0.024 * s, zn + 0.066), (0.012 * s, zn + 0.076), (0.007 * s, zn + 0.08),
                               (0.0, hb)], segments=14)
    (gilt if p.brass_trim else body).append(fin)

    # spout: cast boss with two studs, swept nose, nozzle ring, bucket hook
    ell = [(0.062 * s * cos(radians(a)), 0.046 * s * sin(radians(a))) for a in range(0, 360, 20)]
    body.append(extrude_profile("spout_boss", ell, 0.047 * s, loc=(0, -(Rh - 0.035 * s), zs), rot=(90, 0, 0), bevel=0.004))
    sb = _hex("spout_stud", 0.0095 * s, 0.008)
    dark.append(instances_to_mesh("spout_studs", sb, [trs((sx * 0.047 * s, -(Rh + 0.012 * s), zs), (90, 0, 0)) for sx in (-1, 1)], keep_base=False))
    rs = 0.03 * s
    zm = zs - 0.1 * s                              # mouth height
    sp_path = [(0, -(Rh - 0.03), zs), (0, -(Rh + 0.05), zs + 0.004), (0, -(reach - 0.075), zs + 0.004), (0, -(reach - 0.02), zs - 0.02),
               (0, -reach, zs - 0.062), (0, -reach, zm)]
    body.append(sweep("spout", sp_path, rs, sides=12, step=0.022, cap=False, scale_fn=lambda t: 1.0 - 0.14 * t))
    ring = torus("nozzle", major=rs * 0.86, minor=0.0065, loc=(0, -reach, zm + 0.003), major_seg=14, minor_seg=5)
    (gilt if p.brass_trim else body).append(ring)
    bore = cylinder("bore", rs * 0.8, 0.003, loc=(0, -reach, zm + 0.012), verts=12)
    yh = reach - 0.036
    body.append(sweep("hook", [(0, -yh, zs - 0.006), (0, -yh - 0.004, zs + 0.046), (0, -yh + 0.014, zs + 0.066), (0, -yh + 0.036, zs + 0.058)],
                      0.0085, sides=6, step=0.014, scale_fn=lambda t: 1.0 - 0.45 * t))
    hook_y = yh - 0.026                             # where a bail sits (distance in front of the axis)
    z_sp_top = zs + 0.004 + rs * 0.95

    # enamel number plate on a cast tablet
    if txt:
        zpl = (zb0 + zb1) / 2
        tw_, th_ = 0.125 * s, min(0.075 * s, zb1 - zb0 - 0.05)
        body.append(box("tablet", (tw_, 0.034, th_), loc=(0, -(Rh + 0.006 - 0.017), zpl), bevel=0.004))
        # enamelled number plate (reads in any light, unlike a flat brass mirror), fired-on letters, brass rivets
        pw, ph = tw_ - 0.02, th_ - 0.018
        plate = box("plate", (pw, 0.003, ph), loc=(0, -(Rh + 0.0075), zpl), bevel=0.001)
        size = min((ph - 0.012) * 0.66, (pw - 0.03) / (0.7 * max(len(txt), 1)))
        letters.append(text("plate_text", txt, size=size, depth=0.0008, loc=(0, -(Rh + 0.009), zpl - size * 0.4), font="serif"))
        rv = _nail("plate_rivet", 0.004, 0.002)
        gilt.append(instances_to_mesh("plate_rivets", rv, [trs((sx * (tw_ / 2 - 0.016), -(Rh + 0.009), zpl), (90, 0, 0)) for sx in (-1, 1)],
                                      keep_base=False))

    # handle fulcrum: two cheeks cast on the barrel, slot, pin - built towards +X, then turned to the chosen side
    xp = Rh + 0.07 * s
    rc_ = 0.032 * s
    outline = [(Rh - 0.025, z_piv - 0.105 * s), (Rh + 0.012, z_piv - 0.1 * s)]
    outline += [(xp + rc_ * cos(radians(a)), z_piv + rc_ * sin(radians(a))) for a in range(-70, 91, 20)]
    outline += [(Rh - 0.025, z_piv + rc_ + 0.012)]
    fulc, fulc_dark, handle, bare = [], [], [], []
    for sy in (-1, 1):
        y0 = 0.034 * s if sy > 0 else -0.02 * s
        fulc.append(shape_extrude("cheek", outline, depth=0.014 * s, loc=(0, y0, 0), rot=(90, 0, 0), bevel=0.002))
    slot = box("slot", (0.014, 0.04 * s, 0.16 * s), loc=(Rh + 0.002, 0, z_piv - 0.03 * s))
    fulc_dark.append(cylinder("pin", 0.009 * s, 0.076 * s, loc=(xp, 0, z_piv), rot=(90, 0, 0), verts=8))
    pin_head = _hex("pin_head", 0.017 * s, 0.011)
    place(pin_head, loc=(xp, -0.034 * s, z_piv), rot=(90, 0, 0))
    fulc_dark.append(pin_head)
    fulc_dark.append(box("pin_nut", (0.027 * s, 0.012, 0.027 * s), loc=(xp, 0.04 * s, z_piv), rot=(0, 19, 0), bevel=0.002, segments=1))

    HL = p.handle_len
    shape = [(0.12, 0.012), (0.3, 0.0), (0.5, -0.06), (0.68, -0.17), (0.82, -0.31), (0.92, -0.41), (1.0, -0.46), (1.07, -0.468)]
    hp = [(Rh - 0.03, 0, z_piv + 0.012), (xp, 0, z_piv)] + [(xp + a * HL, 0, z_piv + b * HL) for a, b in shape]
    hp = [tuple(q) for q in resample_path(hp, n=36)]
    handle.append(sweep("handle_bar", hp[:-6], rounded_rect(0.04 * s, 0.018 * s, 0.005 * s, 1), up=(0, 1, 0), smooth_path=False,
                        scale_fn=lambda t: 1.0 - 0.3 * t))
    bare.append(sweep("handle_grip", hp[-8:], 0.0135 * s, sides=8, smooth_path=False,
                      scale_fn=lambda t: 1.0 + 0.22 * sin(t * pi)))
    bare.append(sphere("handle_knob", 0.021 * s, loc=hp[-1], segments=8, rings=5))
    handle.append(sphere("handle_collar", 0.018 * s, loc=hp[-8], segments=8, rings=5))
    handle.append(cylinder("handle_eye", 0.029 * s, 0.02 * s, loc=(xp, 0, z_piv), rot=(90, 0, 0), verts=12, bevel=0.003, segments=1))
    grip_l = hp[-5]
    if broken:                                     # the pump rod parted years ago: the lever hangs dead on its pin
        droop = radians(rng.u(11.0, 16.0))
        cd_, sd_ = cos(droop), sin(droop)

        def droop_fn(co):
            x, z = co.x - xp, co.z - z_piv
            return Vector((xp + x * cd_ + z * sd_, co.y, z_piv - x * sd_ + z * cd_))

        for o in handle + bare:
            apply_transform(o)
            transform_verts(o, droop_fn)
        grip_l = tuple(droop_fn(Vector(grip_l)))

    # rust: blisters where the paint fails first (flange rims, foot of the barrel, plinth, holding-down bolts, cheeks)
    # and runs weeping from the flange joint down the flutes
    dr = rng.fork("rust")
    dec, decf = _Decals(), _Decals()
    eps = 0.0015
    ksz = 0.75 + 0.7 * age

    def on_cyl(R, th0, z0):
        return lambda a, b: (R * cos(th0 + a / R), R * sin(th0 + a / R), z0 + b)

    def on_plane(o, u, v):
        return lambda a, b: (o[0] + a * u[0] + b * v[0], o[1] + a * u[1] + b * v[1], o[2] + a * u[2] + b * v[2])

    for _ in range(int(5 + 11 * age)):             # flange rims
        z0 = z_fl + (0.0105 if dr.chance(0.5) else -0.0105)
        dec.blob(on_cyl(Rf + eps, dr.u(0, 2 * pi), z0), dr.u(0.008, 0.02) * ksz, 0.0042, dr, n=7)
    for _ in range(int(3 + 9 * age)):              # foot of the barrel just above the joint (clear of the spout boss)
        dec.blob(on_cyl(Rh + eps, dr.u(-0.6, pi + 0.6), z_fl + 0.036 + dr.u(0.008, 0.05)), min(0.02, dr.u(0.006, 0.014) * ksz),
                 dr.u(0.005, 0.012) * ksz, dr)
    if age > 0.55:                                 # whole flakes of paint gone from the barrel
        for _ in range(int(16 * (age - 0.5))):
            dec.blob(on_cyl(Rh + eps, dr.u(-0.5, pi + 0.5), dr.u(zb0 + 0.045, zb1 - 0.045)), dr.u(0.012, 0.02), dr.u(0.012, 0.026), dr)
    apo2 = (rb + 0.014) * cos(radians(22.5)) + eps
    for _ in range(int(4 + 9 * age)):              # pedestal plinth
        a = radians(dr.randrange(8) * 45.0)
        fnp = on_plane((apo2 * sin(a), -apo2 * cos(a), 0.035 + dr.jitter(0.0, 0.003)), (cos(a), sin(a), 0.0), (0.0, 0.0, 1.0))
        dec.blob(lambda x, y, f=fnp, off=dr.u(-0.028, 0.028): f(x + off, y), dr.u(0.008, 0.016) * ksz, dr.u(0.005, 0.007), dr, n=7)
    for sx in (-1, 1):                             # base plate round the holding-down bolts
        for sy in (-1, 1):
            if dr.chance(0.45 + 0.5 * age):
                dec.blob(on_plane((sx * c, sy * c, 0.022 + eps), (1, 0, 0), (0, 1, 0)), dr.u(0.014, 0.018), dr.u(0.014, 0.018), dr, n=8)
    pitch = 2 * pi * Rc / max(nf, 1)
    k_front = int(round((-nf / 2 - 1) / 2)) if nf > 0 else 0
    used = set()
    for i in range(int(3 + 8 * age)):              # runs below the flange joint, lying in the flutes (the first under the spout)
        if nf > 0:
            k = ((k_front + i) if i < 2 else dr.randrange(nf)) % nf
            if k in used:
                continue
            used.add(k)
            th0 = (2 * k + 1) * pi / nf
            hw = min(0.004 + 0.004 * age, 0.17 * pitch)
        else:
            th0 = (-pi / 2 + dr.jitter(0.0, 0.3)) if i < 2 else dr.u(0, 2 * pi)
            hw = 0.005 + 0.005 * age
        ln = min(dr.u(0.08, 0.26) * (0.7 + 0.6 * age), zt - 0.05 - ped_h - 0.07)
        z_top = zt - 0.045

        def run_fn(a, b, th0=th0, z0=z_top):
            rr = Rc - 0.875 * d_fl * f_env(z0 + b) + eps
            return (rr * cos(th0 + a / Rc), rr * sin(th0 + a / Rc), z0 + b)

        dec.streak(run_fn, hw, ln, dr)
    for sy in (-1, 1):                             # cheeks round the pin
        yo = sy * (0.034 * s + eps)
        u = (-1.0, 0.0, 0.0) if sy > 0 else (1.0, 0.0, 0.0)
        for _ in range(int(2 + 4 * age)):
            decf.blob(on_plane((dr.u(Rh + 0.02, xp - 0.014), yo, z_piv + dr.u(-0.034, 0.008)), u, (0, 0, 1)), dr.u(0.005, 0.009) * ksz,
                      dr.u(0.005, 0.009) * ksz, dr, n=7)
        decf.blob(on_plane((xp, yo, z_piv), u, (0, 0, 1)), 0.019 * s, 0.019 * s, dr, n=9)
    rust_b, rust_f = dec.mesh("rust_body"), decf.mesh("rust_fulc")

    for o in body + fulc + handle:
        assign(o, iron)
    for o in dark + fulc_dark + bare + letters + [slot, bore, rust_b, rust_f]:
        assign(o, rusty)
    for o in gilt:
        assign(o, brass)
    if txt:
        assign(plate, mat("ceramic_glazed", "#e4dcc6"))
        _put([plate], (0, py, zp))
        ex, ez = pw / 2, zp + zpl
        paint(plate, wear=lambda co, n: np.clip((np.abs(co[:, 0]) / ex) ** 6 * (np.abs(co[:, 2] - ez) / (ph / 2)) ** 3, 0.0, 1.0) * (0.4 + 0.6 * age),
              grime=mask_mul(noise_mask(0.04, seed=seed + 5, lo=0.35, hi=0.8), 0.25 + 0.6 * age))
    rz = {"right": 0.0, "left": 180.0, "back": 90.0}[p.handle_side]
    _put(body + dark + gilt + letters + [bore, rust_b], (0, py, zp))
    _put(fulc + fulc_dark + handle + bare + [slot, rust_f], (0, py, zp), rz)
    ca, sa = cos(radians(rz)), sin(radians(rz))
    piv_w = (xp * ca, py + xp * sa, zp + z_piv)
    grip_w = np.array((grip_l[0] * ca, py + grip_l[0] * sa, zp + grip_l[2]))
    part(handle + bare, "handle", pivot=piv_w, motion=None)

    # the paint itself: brown bleeding round the blisters and joints, damp grime only at the very foot so that the
    # ribs, flutes and panels keep catching the lantern all the way down
    def rust_zone(co, n):
        z = co[:, 2] - zp
        base = np.clip(1.0 - z / (0.1 * s), 0.0, 1.0)
        front = (co[:, 1] < py - Rc * 0.4) & (z < zs + 0.02) & (z > ped_h * 0.5)
        streak = np.clip(1.0 - np.abs(co[:, 0]) / 0.045, 0.0, 1.0) * front * 0.7
        joint = np.clip(1.0 - np.abs(z - z_fl) / 0.035, 0.0, 1.0) * 0.6
        return np.maximum(np.maximum(base, streak), joint)

    rust = mask_mul(rust_zone, noise_mask(0.05, seed=seed + 2, lo=0.3, hi=0.65), min(1.0, 0.35 + 0.7 * age))
    flake = mask_mul(noise_mask(0.065, seed=seed + 4, lo=0.5, hi=0.68), smoothstep(0.5, 1.0, age))
    mottle = noise_mask(0.025, seed=seed + 16, lo=0.25, hi=0.75)   # old rust is brown and patchy, not one red coat
    for o in body + fulc:
        paint(o, special=lambda co, n: np.maximum(rust(co, n), flake(co, n) * (0.55 + 0.45 * mottle(co, n))),
              wear=lambda co, n: flake(co, n) * (1.0 - mottle(co, n)) * 0.85,
              grime=lambda co, n: np.clip(1.0 - (co[:, 2] - zp) / 0.07, 0.0, 1.0) * 0.45)
    for o in dark + fulc_dark:
        paint(o, wear=mask_mul(noise_mask(0.012, seed=seed + 12, lo=0.3, hi=0.7), 0.6 - 0.4 * age), special=0.3 + 0.7 * age)
    for o in (rust_b, rust_f):
        paint(o, special=0.5 + 0.5 * age, wear=mask_mul(lambda co, n: 1.0 - mottle(co, n), 0.3 + 0.5 * age))
    for o in letters + [slot, bore]:
        paint(o, grime=1.0, wear=0.8)
    for o in bare:                                 # the grip is polished to bare metal by a century of hands
        paint(o, wear=mask_mul(noise_mask(0.02, seed=seed + 15, lo=0.1, hi=0.6), 0.9 - 0.5 * smoothstep(0.6, 1.0, age)))
    for o in handle:
        paint(o, wear=lambda co, n: np.clip(1.0 - (np.linalg.norm(co - grip_w, axis=1) - 0.06) / 0.12, 0.0, 1.0) * 0.9,
              special=mask_mul(noise_mask(0.08, seed=seed + 3, lo=0.45, hi=0.75), lambda co, n: 0.55 + 0.45 * mottle(co, n), 0.25 + 0.75 * age))
    for o in gilt:
        paint(o, special=0.12 + 0.3 * age, wear=0.4)

    z_mouth = zp + zm
    fx("drip", "drip", (0, y_mouth, z_mouth - 0.01), rate=0.6 + 0.8 * age)
    socket("hook", "attach", (0, py - hook_y, zp + z_sp_top), role="bucket")

    # ------------------------------------------------------------------ trough
    tr_ = rng.fork("trough")
    t_yaw = tr_.jitter(0.0, 1.2)
    moss_sites = []                                # (x, y, z, yaw, sx, sy, sz)
    scum_rect = None

    def near_spout(co, rad):
        return np.clip(1.0 - np.hypot(co[:, 0], co[:, 1] - y_mouth) / rad, 0.0, 1.0)

    if kind == "stone":
        zb = zp + 0.05
        Ht, fl = 0.40 * s, 0.085
        nx, ny = max(6, min(16, int(L / 0.062))), max(5, int(TW / 0.062))
        li, wi = L - 2 * wall, TW - 2 * wall
        zwl = min(fl + p.water_level * (Ht - fl - 0.03), Ht - 0.05)      # it cannot stand above the overflow notch
        pr = [tr_.u(0, 6.28) for _ in range(6)]

        def R(w, d, r, z, vary=0.0):
            ring = []
            for x, y in _rr(w, d, r, nx, ny):
                if vary:                           # the hollow was pecked out by eye: the rim is 60-80 mm and wanders
                    th = atan2(y / d, x / w)
                    dd = vary * (0.6 * sin(2 * th + pr[0]) + 0.4 * sin(3 * th + pr[1]) + 0.25 * sin(5 * th + pr[2]))
                    ex_, ey_ = x / (w * w), y / (d * d)
                    nn = hypot(ex_, ey_) or 1.0
                    x, y = x + dd * ex_ / nn, y + dd * ey_ / nn
                ring.append((x, y, z))
            return ring

        nmid = max(4, int(round((Ht - 0.135) / 0.058)) + 1)
        rings = [R(L - 0.04, TW - 0.04, 0.025, 0.0), R(L - 0.012, TW - 0.012, 0.03, 0.022)]
        rings += [R(L - 0.006 * (1.0 - k / (nmid - 1)), TW - 0.006 * (1.0 - k / (nmid - 1)), 0.03, 0.08 + (Ht - 0.135) * k / (nmid - 1))
                  for k in range(nmid)]
        z3 = min(zwl + 0.07, Ht - 0.032)
        z4 = min(zwl + 0.016, z3 - 0.014)
        rings += [R(L - 0.004, TW - 0.004, 0.03, Ht - 0.016), R(L - 0.026, TW - 0.026, 0.026, Ht),
                  R(li + 0.02, wi + 0.02, 0.034, Ht, 0.011), R(li, wi, 0.03, Ht - 0.014, 0.012), R(li - 0.006, wi - 0.006, 0.034, z3, 0.011),
                  R(li - 0.012, wi - 0.012, 0.036, z4, 0.009), R(li - 0.016, wi - 0.016, 0.038, zwl - 0.035, 0.008)]
        trough = _loft("trough", rings)            # (the hollow is closed just below the water line)

        nbr = 1 + (1 if tr_.chance(0.35 + 0.5 * age) else 0) + (1 if age > 0.7 else 0)
        corners = sorted([(-1, -1), (1, -1), (-1, 1), (1, 1)], key=lambda q: tr_.u())
        cuts = []
        for sx, sy in corners[:nbr]:
            nn = Vector((sx * tr_.u(0.6, 1.0), sy * tr_.u(0.6, 1.0), tr_.u(0.5, 1.1))).normalized()
            cuts.append((Vector((sx * L / 2, sy * TW / 2, Ht)), nn, tr_.u(0.045, 0.072)))
        sag0, tilt = tr_.u(0.007, 0.013), tr_.jitter(0.0, 0.006)

        def hew(co):
            x, y, z = co.x, co.y, co.z
            w = smoothstep(Ht * 0.45, Ht * 0.92, z)
            z -= w * (sag0 * (1.0 - (2 * x / L) ** 2) + 0.004 * sin(x * 7.3 + pr[3]) + 0.003 * sin(y * 13.0 + pr[4]) + tilt * 2 * x / L + 0.004)
            if co.z > Ht - 0.05 and x > L / 2 - wall - 0.035 and abs(y) < 0.09:      # worn overflow lip at the right-hand end
                z -= 0.03 * (1.0 - (abs(y) / 0.09) ** 2) ** 2
            v = Vector((x, y, z))
            for c0, nn, dd in cuts:                # corners knocked off by cart wheels
                h = (v - c0).dot(nn) + dd
                if h > 0.0:
                    v = v - nn * h
            return v

        transform_verts(trough, hew)
        recalc_normals(trough)
        displace_noise(trough, strength=0.013, scale=0.13, seed=seed + 20)
        displace_noise(trough, strength=0.0035, scale=0.045, seed=seed + 21)
        smooth(trough, 55)
        uv_box(trough)
        # where the water meets the hewn wall: read the water line back off the displaced mesh
        bpy.context.view_layer.update()
        tv = trough.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices
        m_ = len(rings[0])
        shore = []
        if len(tv) == len(rings) * m_ + 2:
            for j in range(m_):
                a, b = tv[(len(rings) - 2) * m_ + j].co, tv[(len(rings) - 1) * m_ + j].co
                t = clamp((zwl - a.z) / min(b.z - a.z, -1e-5), 0.0, 1.0)
                shore.append((lerp(a.x, b.x, t), lerp(a.y, b.y, t)))
        else:
            shore = [(sx * (li / 2 - 0.006), sy * (wi / 2 - 0.006)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        stones = [trough]
        _put(stones, (0, ty, zb), t_yaw)
        for sx in (-1, 1):
            stones.append(stone_block("bearer", (0.13, TW - 0.07, 0.066), loc=(sx * (L / 2 - 0.17), ty, zp + 0.023), rng=tr_, chamfer=0.01,
                                      rough=0.005, detail=0.1))
        zw = zb + zwl
        wx, wy = li + 0.034, wi + 0.034
        water = _water("trough_water", wx, wy, (0.0, y_mouth - ty), tr_, shore=shore)
        place(water, loc=(0, ty, zw), rot=(0, 0, t_yaw))
        assign(water, water_m)
        scum_rect = (-wx / 2, wx / 2, ty - wy / 2, ty + wy / 2, zw, 0.0, 0.0)
        hx, hy = li / 2 + 0.03, wi / 2 + 0.03
        ztr = zb + Ht

        def inside(co, n):
            return ((np.abs(co[:, 0]) < hx) & (np.abs(co[:, 1] - ty) < hy) & (co[:, 2] < ztr - 0.022)).astype(float)

        def band(co, n):                            # wet, dark band just above the water line
            wall_d = np.minimum(hx - np.abs(co[:, 0]), hy - np.abs(co[:, 1] - ty))
            under = 0.35 + 0.55 * np.clip(1.0 - (wall_d - 0.03) / 0.09, 0.0, 1.0)      # the floor seen through the water
            return inside(co, n) * np.where(co[:, 2] < zw - 0.02, under, np.clip(1.0 - (co[:, 2] - zw - 0.02) / 0.05, 0.0, 1.0))

        def algae(co, n):                           # green tide line, strongest where the spout splashes
            return inside(co, n) * np.clip(1.0 - np.abs(co[:, 2] - zw - 0.03) / 0.075, 0.0, 1.0) * (0.3 + 0.7 * near_spout(co, 0.42))

        def run(co, n):                             # overflow run-off down the end under the notch
            dn = np.clip((ztr - co[:, 2]) / (Ht + 0.05), 0.0, 1.0)
            wid = 0.045 + 0.07 * dn
            return (co[:, 0] > L / 2 - 0.035) * (1.0 - inside(co, n)) * np.clip(1.0 - np.abs(co[:, 1] - ty) / wid, 0.0, 1.0) ** 0.8 * (1.0 - 0.45 * dn)

        def foot(co, n):
            return np.clip(1.0 - (co[:, 2] - zp) / 0.16, 0.0, 1.0)

        def rim_damp(co, n):
            return (co[:, 2] > ztr - 0.06) * near_spout(co, 0.3)

        top_moss = mask_mul(facing((0, 0, 1), 2.0), lambda co, n: (co[:, 2] > ztr - 0.07).astype(float),
                            noise_mask(0.1, seed=seed + 24, lo=0.35, hi=0.6), smoothstep(0.35, 1.0, age))
        nz = noise_mask(0.07, seed=seed + 22, lo=0.2, hi=0.55)
        for o in stones:
            assign(o, stone_m)
            paint(o, grime=lambda co, n: np.maximum.reduce([band(co, n), run(co, n) * (0.35 + 0.4 * nz(co, n)), foot(co, n) * 0.75, rim_damp(co, n) * 0.55]),
                  special=lambda co, n: np.maximum.reduce([algae(co, n) * nz(co, n), run(co, n) * nz(co, n) * (0.6 + 0.4 * age),
                                                           foot(co, n) * nz(co, n) * (0.35 + 0.65 * age), top_moss(co, n)]))
        # moss and dirt gathered at the foot, round the bearers and (when neglected) along the rim
        for _ in range(int(4 + 8 * age)):
            x = tr_.pick([-1, 1]) * (L / 2 - 0.17) + tr_.jitter(0.0, 0.09)
            y = ty + tr_.pick([-1, 1]) * (TW / 2 - 0.045 + tr_.u(0.0, 0.03))
            moss_sites.append((x, y, ztop(x, y) + 0.003, tr_.u(0, 180), tr_.u(0.022, 0.05), tr_.u(0.016, 0.03), tr_.u(0.01, 0.02)))
        for _ in range(int(22 * smoothstep(0.5, 1.0, age))):
            if tr_.chance(0.6):
                x, y = tr_.u(-L / 2 + 0.05, L / 2 - 0.1), ty + tr_.pick([-1, 1]) * (TW / 2 - wall / 2)
            else:
                x, y = -(L / 2 - wall / 2), ty + tr_.u(-TW / 2 + 0.05, TW / 2 - 0.05)
            moss_sites.append((x, y, ztr - 0.016 - sag0 * (1.0 - (2 * x / L) ** 2), tr_.u(0, 180), tr_.u(0.03, 0.085), tr_.u(0.02, 0.034),
                               tr_.u(0.012, 0.02)))
    elif kind == "wood":
        zb = zp + 0.07                               # top of the bearers
        bt, hbx = 0.04, 0.30
        st = wall
        planks, straps = [], []
        for sx in (-1, 1):
            planks.append(plank("bearer", length=TW + 0.13 + tr_.u(0, 0.03), width=0.085, thick=0.078, loc=(sx * (L / 2 - 0.12), ty, zp + 0.03),
                                rot=(0, 0, tr_.jitter(0, 1.5)), axis="Y", bevel=0.005, rng=tr_))
            planks.append(plank("end_board", length=TW - 2 * st - 0.004, width=0.042, thick=hbx - 0.006, loc=(sx * (L / 2 - 0.055), ty, zb + bt + hbx / 2 - 0.003),
                                axis="Y", bevel=0.003, rng=tr_))
        planks.append(plank("bottom", length=L - 0.02, width=TW, thick=bt, loc=(0, ty, zb + bt / 2), axis="X", bevel=0.004, warp=0.002, rng=tr_))
        for sy in (-1, 1):
            for k in range(2):
                hk = hbx / 2 - 0.003
                planks.append(plank("side_board", length=L + tr_.u(0.0, 0.03), width=st, thick=hk, loc=(tr_.jitter(0, 0.008), ty + sy * (TW / 2 - st / 2),
                                    zb + bt + hbx / 4 + k * hbx / 2 + (0.002 if k else 0.0)), rot=(tr_.jitter(0, 0.6) if k else 0.0, 0, 0),
                                    axis="X", bevel=0.004, warp=0.003, rng=tr_))
        ztr = zb + bt + hbx
        bar = rounded_rect(0.036, 0.005, 0.0012, 1)
        nail = _nail("nail", 0.0075, 0.0045)
        nm = []
        for sx in (-1, 1):
            x = sx * (L / 2 - 0.24)
            y0, y1 = ty - TW / 2 - 0.0035, ty + TW / 2 + 0.0035
            straps.append(sweep("strap", [(x, y0, ztr - 0.012), (x, y0, zb - 0.0035), (x, y1, zb - 0.0035), (x, y1, ztr - 0.012)], bar,
                                up=(0, -1, 0), smooth_path=False))
            for zz in (zb + bt * 0.5, zb + bt + hbx * 0.25, zb + bt + hbx * 0.75):
                nm.append(trs((x + tr_.jitter(0, 0.004), y0 - 0.002, zz), (90, 0, 0)))
                nm.append(trs((x + tr_.jitter(0, 0.004), y1 + 0.002, zz), (-90, 0, 0)))
            # forged nails fixing the sides into the end boards
            for sy in (-1, 1):
                for zz in (zb + bt + 0.05, zb + bt + 0.12, zb + bt + 0.2, zb + bt + 0.26):
                    nm.append(trs((sx * (L / 2 - 0.055) + tr_.jitter(0, 0.006), ty + sy * (TW / 2 + 0.0005), zz), (-90 * sy, 0, 0), 0.8))
        straps.append(instances_to_mesh("nails", nail, nm, keep_base=False))
        zw = zb + bt + p.water_level * (hbx - 0.03)
        wx, wy = L - 0.11, TW - 2 * st + 0.006
        sxh, syh = L / 2 - 0.077, TW / 2 - st - 0.001
        water = _water("trough_water", wx, wy, (0.0, y_mouth - ty), tr_,
                       shore=[(-sxh, -syh), (0, -syh), (sxh, -syh), (sxh, 0), (sxh, syh), (0, syh), (-sxh, syh), (-sxh, 0)])
        place(water, loc=(0, ty, zw))
        assign(water, water_m)
        scum_rect = (-wx / 2, wx / 2, ty - wy / 2, ty + wy / 2, zw, 0.0, 0.0)
        hx, hy = L / 2 - 0.05, TW / 2 - st + 0.012

        def inside(co, n):
            return ((np.abs(co[:, 0]) < hx) & (np.abs(co[:, 1] - ty) < hy) & (co[:, 2] > zb + bt * 0.6)).astype(float)

        def damp(co, n):                            # soaked below the water line, at the foot, and where the spout splashes the rim
            return np.maximum.reduce([inside(co, n) * np.clip(1.0 - (co[:, 2] - zw - 0.02) / 0.07, 0.0, 1.0),
                                      np.clip(1.0 - (co[:, 2] - zp) / 0.28, 0.0, 1.0) * 0.85,
                                      (co[:, 2] > ztr - 0.08) * near_spout(co, 0.3) * 0.8,
                                      (co[:, 0] > L / 2 - 0.07) * np.clip(1.0 - np.abs(co[:, 1] - ty) / 0.12, 0.0, 1.0) * 0.8])

        def green(co, n):
            return np.maximum(inside(co, n) * np.clip(1.0 - np.abs(co[:, 2] - zw - 0.03) / 0.08, 0.0, 1.0) * (0.3 + 0.7 * near_spout(co, 0.42)),
                              np.clip(1.0 - (co[:, 2] - zp) / 0.2, 0.0, 1.0) * (0.35 + 0.65 * age))

        for o in planks:
            assign(o, wood_m)
            densify(o, 0.17)
            paint(o, grime=damp, special=mask_mul(green, noise_mask(0.08, seed=seed + 22, lo=0.2, hi=0.6)),
                  wear=mask_mul(facing((0, 0, 1)), lambda co, n: (co[:, 2] > ztr - 0.03).astype(float), 0.6))
        for o in straps:
            assign(o, rusty)
            paint(o, wear=mask_mul(noise_mask(0.03, seed=seed + 13, lo=0.3, hi=0.7), 0.7 - 0.5 * age), special=0.3 + 0.7 * age)
        for _ in range(int(4 + 8 * age)):
            x = tr_.pick([-1, 1]) * (L / 2 - 0.12) + tr_.pick([-1, 1]) * 0.05
            y = ty + tr_.u(-TW / 2 - 0.05, TW / 2 + 0.05)
            moss_sites.append((x, y, ztop(x, y) + 0.003, tr_.u(0, 180), tr_.u(0.022, 0.05), tr_.u(0.016, 0.03), tr_.u(0.01, 0.02)))
    else:
        # cast drain grate set in a pit between the drain stones: frame, bars with worn tops, a wet sump glinting below
        gw = 0.272
        gi = gw - 0.05
        frame = shape_extrude("grate", rounded_rect(gw, gw, 0.02, 3), [rounded_rect(gi, gi, 0.008, 2)], depth=0.02, loc=(0, gy, zp - 0.015),
                              bevel=0.002)
        nb, bw = 7, 0.014
        gap = (gi - nb * bw) / (nb + 1)
        iron_bits = [frame]
        for k in range(nb):
            x = -gi / 2 + gap + bw / 2 + k * (gap + bw)
            iron_bits.append(box("grate_bar", (bw, gi + 0.006, 0.012), loc=(x, gy, zp - 0.002 + tr_.jitter(0.0, 0.0007)), bevel=0.002, segments=1))
        iron_bits.append(box("grate_tie", (gi + 0.006, 0.014, 0.008), loc=(0, gy, zp - 0.007), bevel=0.0015, segments=1))
        gb = _nail("grate_bolt", 0.008, 0.0045)
        iron_bits.append(instances_to_mesh("grate_bolts", gb, [trs((sx * (gw / 2 - 0.0125), gy + sy * (gw / 2 - 0.0125), zp + 0.005))
                                                              for sx in (-1, 1) for sy in (-1, 1)], keep_base=False))
        sump = box("sump", (gw - 0.006, gw - 0.006, 0.004), loc=(0, gy, 0.003))
        sump_w = _water("sump_water", gi + 0.02, gi + 0.02, (0.0, y_mouth - gy), tr_, amp=0.0012, lam=0.03, nseg=14, fine=8)
        place(sump_w, loc=(0, gy, 0.0085))
        assign(sump_w, water_m)
        for o in iron_bits + [sump]:
            assign(o, rusty)
        for o in iron_bits:
            paint(o, wear=mask_mul(facing((0, 0, 1), 3.0), noise_mask(0.05, seed=seed + 14, lo=0.15, hi=0.6), 0.7),
                  special=0.4 + 0.6 * age)
        paint(sump, grime=1.0, wear=1.0)

    # ------------------------------------------------------------------ moss in the joints
    mr = rng.fork("moss")
    joints = []
    for x0, x1, y0, y1, *_ in slabs:
        if x1 < Wx / 2 - 0.03 and y1 - y0 > 0.1:
            joints.append(((x1 + 0.005, y0 + 0.03), (x1 + 0.005, y1 - 0.03), 90.0))
        if y1 < Dy / 2 - 0.03 and x1 - x0 > 0.1:
            joints.append(((x0 + 0.03, y1 + 0.005), (x1 - 0.03, y1 + 0.005), 0.0))
    for _ in range(int(5 + 20 * age)):
        (ax, ay), (bx, by), yaw = mr.pick(joints)
        t = mr.u(0, 1)
        moss_sites.append((lerp(ax, bx, t), lerp(ay, by, t), zj + 0.001, yaw + mr.jitter(0, 12), mr.u(0.018, 0.05), mr.u(0.007, 0.011),
                           mr.u(0.008, 0.014)))
    for _ in range(int(2 + 6 * age)):               # along the back edge of the apron and behind the pedestal
        if mr.chance(0.6):
            x, y = mr.u(-Wx / 2, Wx / 2), Dy / 2 + mr.u(-0.01, 0.012)
            moss_sites.append((x, y, zj - 0.002, mr.jitter(0, 15), mr.u(0.03, 0.06), mr.u(0.012, 0.02), mr.u(0.009, 0.015)))
        else:
            x, y = mr.jitter(0.0, ped_half * 0.8), py + ped_half + mr.u(-0.004, 0.012)
            moss_sites.append((x, y, zp + 0.002, mr.jitter(0, 15), mr.u(0.025, 0.05), mr.u(0.012, 0.02), mr.u(0.009, 0.016)))
    mb = sphere("moss_base", 1.0, segments=6, rings=3)
    moss = instances_to_mesh("moss", mb, [trs((x, y, z), (mr.jitter(0, 8), mr.jitter(0, 8), yaw), (sx, sy, sz))
                                          for x, y, z, yaw, sx, sy, sz in moss_sites], keep_base=False)
    displace_noise(moss, strength=0.004, scale=0.03, seed=seed + 40)
    smooth(moss, 70)
    assign(moss, moss_m)

    if scum_rect and age > 0.55:                    # stagnant green scum gathering along the walls of the trough
        x0, x1, y0, y1 = scum_rect[:4]
        sb_ = []
        for i in range(int(3 + 10 * (age - 0.5))):
            if mr.chance(0.5):
                sb_.append((mr.u(x0, x1), mr.pick([y0, y1]), mr.u(0.07, 0.13), mr.u(0, 6.28), 0.0012 + 0.0003 * i))
            else:
                sb_.append((mr.pick([x0, x1]), mr.u(y0, y1), mr.u(0.07, 0.13), mr.u(0, 6.28), 0.0012 + 0.0003 * i))
        sc = _film("scum", sb_, [scum_rect], margin=0.0, nseg=12)
        if sc:
            assign(sc, moss_m)

    # ------------------------------------------------------------------ pail
    if p.bucket:
        br = rng.fork("bucket")
        hung = kind == "none"
        objs, bail_top = _bucket(br, zinc, wood_m, hung, water_m)
        if hung:
            _put(objs, (br.jitter(0, 0.004), py - hook_y, zp + z_sp_top + 0.004 - bail_top), br.jitter(0, 4.0))
        else:
            bx, by = pail_xy[0], pail_xy[1] + br.jitter(0.0, 0.03)
            _put(objs, (bx, by, ztop(bx, by) - 0.001), br.u(0, 360))
        for o in objs:
            if o.name.startswith("pail") and not o.name.startswith(("pail_grip", "pail_water")):
                paint(o, grime=lambda co, n: np.clip(1.0 - (co[:, 2] - co[:, 2].min()) / 0.1, 0.0, 1.0) * 0.7,
                      special=mask_mul(noise_mask(0.05, seed=seed + 30, lo=0.55, hi=0.85), 0.08 + 0.5 * age))

    # ------------------------------------------------------------------ last autumn's leaves, never swept
    n_mats = len({m.name for o in bpy.context.scene.objects if o.type == "MESH" for m in o.data.materials if m})
    if age > 0.7 and n_mats < 10:
        lv = [(0.0, -0.5, 0.0), (0.2, -0.22, 0.03), (0.27, 0.05, 0.05), (0.17, 0.3, 0.03), (0.0, 0.5, 0.0), (-0.17, 0.3, 0.03), (-0.27, 0.05, 0.05),
              (-0.2, -0.22, 0.03), (0.0, -0.1, 0.0), (0.0, 0.2, 0.0)]
        lf = [(0, 1, 8), (1, 2, 9, 8), (2, 3, 9), (3, 4, 9), (4, 5, 9), (5, 6, 9), (6, 8, 9), (6, 7, 8), (7, 0, 8)]
        leaf = mesh_object("leaf_base", lv, lf)
        uvl = leaf.data.uv_layers.new(name="UVMap")
        for lp in leaf.data.loops:
            v = lv[lp.vertex_index]
            uvl.data[lp.index].uv = ((v[0] + 0.5) * 0.25, (v[1] + 0.5) * 0.25)
        uv_keep(leaf)
        lm = []
        for _ in range(int(45 * (age - 0.6))):
            if scum_rect and mr.chance(0.35):
                x, y = mr.u(scum_rect[0] + 0.05, scum_rect[1] - 0.05), mr.u(scum_rect[2] + 0.05, scum_rect[3] - 0.05)
                z = scum_rect[4] + 0.004
            else:
                x, y = mr.u(-Wx / 2 + 0.04, Wx / 2 - 0.04), mr.u(-Dy / 2 + 0.04, Dy / 2 - 0.04)
                z = ztop(x, y) + 0.006
            lm.append(trs((x, y, z), (mr.jitter(0, 7), mr.jitter(0, 7), mr.u(0, 360)), mr.u(0.06, 0.095)))
        leaves = instances_to_mesh("leaves", leaf, lm, uv_jitter=False, keep_base=False)
        assign(leaves, mat("leaf", "#9a6a2c"))
