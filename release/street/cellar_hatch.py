"""Pavement cellar hatch (bulkhead): two sloping ledged-and-braced plank leaves lying on a rebated
oak curb frame. The curb sits on coursed, lime-pointed stone cheek walls with raking copings and a
worn front kerb, against the house wall. Forged strap hinges with knuckles and pins, ring pulls, a
hasp + staple with a brass padlock, dressed lead flashing pointed into the wall. One leaf can be
propped open on a stick: worn stone treads go down into the dark towards the head of the cellar
door, lit by the cellarman's candle lantern hung on a forged swan-neck bracket above the hatch.
"""
from math import atan2, cos, degrees, exp, hypot, pi, radians, sin

import numpy as np
from mathutils import Matrix, Vector

from tk import *

META = {"name": "Pavement cellar hatch", "class": "prop", "mount": "wall_ground",
        "focus": [0.36, -0.74, 0.62]}
PARAMS = {
    "width": {"default": 1.5, "min": 1.1, "max": 2.2, "desc": "overall width over the stone cheeks (m)"},
    "depth": {"default": 1.3, "min": 0.9, "max": 1.7, "desc": "projection from the wall (m)"},
    "rise": {"default": 0.86, "min": 0.55, "max": 1.1, "desc": "height of the stone cheeks at the wall (m)"},
    "kerb": {"default": 0.26, "min": 0.14, "max": 0.35, "desc": "height of the front kerb (m)"},
    "open": {"default": 0.65, "min": 0.0, "max": 1.0, "desc": "0 = both leaves shut and padlocked, 1 = one leaf propped fully open"},
    "side": {"default": "left", "choices": ["left", "right"], "desc": "which leaf is propped open"},
    "paint": {"default": "#31503f", "desc": "paint colour of the leaves"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: flaked paint, bare boards, rust tears, opened joints"},
    "lantern": {"default": True, "desc": "lit candle lantern hung on the wall bracket above the hatch (marks the obstruction at night)"},
    "lamp": {"default": 1.8, "min": 0.5, "max": 12.0, "desc": "lantern intensity (cd)"},
}
VARIANTS = {
    "locked_oxblood": {"open": 0.0, "paint": "#6a2620", "wear": 0.2, "seed": 4},
    "wide_worn_blue": {"width": 2.05, "depth": 1.55, "rise": 0.98, "kerb": 0.3, "open": 0.3, "side": "right",
                       "paint": "#3f6184", "wear": 0.9, "seed": 9},
    "narrow_full_open": {"width": 1.15, "depth": 1.0, "rise": 0.7, "kerb": 0.2, "open": 1.0, "paint": "#3b3a36",
                         "wear": 0.6, "seed": 15},
    "low_grey_shut": {"width": 1.75, "depth": 1.2, "rise": 0.58, "kerb": 0.15, "open": 0.0, "paint": "#98a39a",
                      "wear": 0.7, "lantern": False, "seed": 23},
}


def _prism(name, pts, a0, a1, axis="X"):
    """Extrude a 2D polygon along X (pts = (y, z)) or along Y (pts = (x, z))."""
    n = len(pts)

    def V(a, u, v):
        return (a, u, v) if axis == "X" else (u, a, v)

    verts = [V(a0, u, v) for u, v in pts] + [V(a1, u, v) for u, v in pts]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _stone(name, pts, x0, x1, seed, ch=0.009, sub=1, dens=None, rough=0.0035, spall=None, warp=None):
    """A hewn stone with a 4-sided (y, z) profile between x0 and x1: faces out of true, optional worn
    hollow (warp) and a knocked-off corner (spall = (corner, radius, depth)), arrises chamfered."""
    o = _prism(name, pts, x0, x1, "X")
    if dens:
        densify(o, dens)
    elif sub:
        subdivide(o, sub)
    if warp:
        transform_verts(o, warp)
    if spall is not None:
        c, r, amt = spall
        cen = Vector((0.5 * (x0 + x1), sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts)))

        def chip(v):
            d = (v - c).length
            return v if d >= r else v + (cen - v).normalized() * amt * (1 - d / r)

        transform_verts(o, chip)
    if rough:
        displace_noise(o, strength=rough, scale=0.16, seed=seed)
    transform_verts(o, lambda v: Vector((v.x, min(v.y, -0.0004), max(v.z, 0.0))))
    bevel(o, ch, segments=1, angle=35)
    smooth(o, 42)
    return o


def _ribbon(name, A, B, hint):
    """Quad strip between two polylines, wound so that it faces `hint`."""
    n = len(A)
    a0, a1, b0 = Vector(A[0]), Vector(A[1]), Vector(B[0])
    flip = (a1 - a0).cross(b0 - a0).dot(Vector(hint)) < 0
    faces = [(i, n + i, n + i + 1, i + 1) if flip else (i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    o = mesh_object(name, [tuple(q) for q in A] + [tuple(q) for q in B], faces)
    smooth(o, 60)
    return o


def _board(name, xl0, xr0, xl1, xr1, y0, y1, z0, t, ny, nt, closed, cup=0.0, bow=0.0, lift=0.0, b=0.003, e=0.008):
    """One sawn board running along +Y (grain along Y) with eased top arrises, a tight edge loop just
    inside each arris (so paint worn off the edges stays a crisp line) and vertex columns on the face
    for streaky weathering. The edges may taper (shrunken / split boards); `closed` adds the
    underside (only needed where it can be seen)."""
    verts, faces = [], []
    m = 0
    for j in range(ny + 1):
        f = j / ny
        y = lerp(y0, y1, f)
        xl, xr = lerp(xl0, xl1, f), lerp(xr0, xr1, f)
        dz = bow * sin(pi * f) + lift * (1 - f) ** 3
        ring = [(xl, 0.0), (xr, 0.0), (xr, t - b), (xr - b, t), (xr - b - e, t)]
        for k in range(1, nt):
            q = k / nt
            ring.append((lerp(xr - b - e, xl + b + e, q), t + cup * 4 * q * (1 - q)))
        ring += [(xl + b + e, t), (xl + b, t), (xl, t - b)]
        m = len(ring)
        verts += [(x, y, z0 + z + dz) for x, z in ring]
    for j in range(ny):
        for i in range(m):
            if i == 0 and not closed:
                continue
            a, c = j * m + i, j * m + (i + 1) % m
            faces.append((a, a + m, c + m, c))
    faces.append(tuple(range(m)))
    faces.append(tuple(range(ny * m + m - 1, ny * m - 1, -1)))
    o = mesh_object(name, verts, faces)
    uv_grain(o, 'Y')
    smooth(o, 35)
    return o


def _aniso(sx, sy, seed, lo, hi):
    """Value noise stretched along Y (the grain / the fall of the doors): streaks, not clouds."""
    f = noise_mask(1.0, seed=seed, lo=lo, hi=hi)

    def fn(co, n=None):
        c = np.empty_like(co)
        c[:, 0] = co[:, 0] / sx
        c[:, 1] = co[:, 1] / sy
        c[:, 2] = 0.5
        return f(c, n)

    return fn


def _tone(hexcol, desat, dark):
    """Aged paint: pull an sRGB hex colour towards its own grey and darken it."""
    try:
        r, g, b = (int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    except (ValueError, TypeError, IndexError):
        return hexcol
    l = 0.3 * r + 0.59 * g + 0.11 * b
    return "#%02x%02x%02x" % tuple(int(round(255 * clamp((l + (c - l) * (1 - desat)) * dark))) for c in (r, g, b))


def _xf(objs, M):
    """Pre-multiply the transform of every object by M (objects keep their own local axes / UVs)."""
    for o in objs:
        N = M @ o.matrix_basis
        loc, q, _ = N.decompose()
        o.rotation_mode = 'XYZ'
        o.location = loc
        o.rotation_euler = q.to_euler('XYZ')
    return objs


def _T(x, y, z):
    return Matrix.Translation((x, y, z))


def _R(deg, axis):
    return Matrix.Rotation(radians(deg), 4, axis)


def build(p, rng):
    W, D, H1, H0 = p.width, p.depth, p.rise, p.kerb
    sO = 1 if p.side == "right" else -1          # side of the leaf that opens
    sF = -sO
    theta = p.open * 80.0 if p.open > 0.06 else 0.0
    is_open = theta > 0.0
    wr = p.wear
    sd = int(p.seed)

    slope = (H1 - H0) / D
    a = atan2(H1 - H0, D)                         # slope of the doors
    L = hypot(D, H1 - H0)                         # slope length
    ca, sa = cos(a), sin(a)
    SL = _T(0, -D, H0) @ Matrix.Rotation(a, 4, 'X')   # flat "door" frame (x across, y up-slope, z normal) -> world

    def S(y, z):                                  # flat (y, z) -> world (y, z)
        return (-D + y * ca - z * sa, H0 + y * sa + z * ca)

    def zs(y):                                    # height of the stone top at world y
        return H0 + (y + D) * slope

    stone = mat("stone_rough", "#a69c8b", special=1.6)
    mortar = mat("mortar", "#cdc3ac")
    step = mat("stone_smooth", "#b9ab90")         # dressed, foot-polished treads against the rough-hewn walling
    wood_p = mat("wood_painted", _tone(p.paint, 0.2 + 0.15 * wr, 0.84 - 0.12 * wr), rough=1.9)
    oak = mat("wood_oak", "#57442f", scale=2.0)
    iron = mat("iron_wrought")
    brass = mat("brass")
    lead = mat("lead")

    # ------------------------------------------------------------------ stonework: kerb, coursed cheeks, raking copings
    tks, tkf, g = 0.15, 0.17, 0.013               # cheek thickness, kerb depth, mortar joint
    xi = W / 2 - tks                              # clear half width between the cheeks
    yk = -D + tkf                                 # back of the kerb / front of the cheeks
    pr = 0.012                                    # kerb stands a little proud
    xfoot = sO * xi * 0.5                         # where the cellarman's boots land
    rs = rng.fork("stone")
    stones, cores = [], []

    xe = W / 2 + 0.004
    nk = 2 if W < 1.3 else (3 if W < 1.9 else 4)
    kc = [-xe] + [-xe + 2 * xe * k / nk + rs.jitter(0, 0.24 * xe / nk) for k in range(1, nk)] + [xe]
    dip = 0.018

    def kerb_wear(v):                             # generations of boots have rounded the front arris
        if v.z > H0 * 0.8 and v.y < -D + 0.02:
            q = dip * exp(-((v.x - xfoot) / 0.26) ** 2)
            return Vector((v.x, v.y + 0.7 * q, v.z - q))
        return v

    for k in range(nk):
        xa, xb = kc[k] + (g / 2 if k else 0), kc[k + 1] - (g / 2 if k < nk - 1 else 0)
        e = pr + rs.u(0, 0.005)
        sp = None
        if rs.chance(0.6):
            sp = (Vector((xa if rs.chance(0.5) else xb, -D - e, zs(-D) if rs.chance(0.75) else 0.0)), rs.u(0.07, 0.11), rs.u(0.018, 0.032))
        stones.append(_stone(f"kerb{k}", [(-D - e, 0), (yk, 0), (yk, zs(yk) - 0.001), (-D - e, zs(-D - e))], xa, xb, sd + 70 + k,
                             ch=0.011, sub=2, spall=sp, warp=kerb_wear))
    rec = 0.005
    cores.append(_prism("core_kerb", [(-D - pr + rec + 0.004, 0), (yk - 0.003, 0), (yk - 0.003, zs(yk) - rec), (-D - pr + rec + 0.004, zs(-D) - rec)],
                        -xe + rec, xe - rec, "X"))

    tc = min(0.115, zs(yk) * 0.45)                # vertical thickness of the raking coping

    def under(y):                                 # underside of the coping
        return zs(y) - tc

    def y_under(z):
        return (z - H0 + tc) / slope - D

    c1 = under(yk)
    rem = under(0.0) - c1
    nup = max(1, int(round(rem / 0.25)))
    zc = [c1] + [c1 + rem * k / nup + rs.jitter(0, 0.02) for k in range(1, nup)] + [under(0.0)]
    emin = 0.05
    for s in (-1, 1):
        cnt = [0]

        def add(poly, out=0.0, sub=1, spall=None, s=s, cnt=cnt):
            dx = rs.u(0, 0.004) + out
            x0, x1 = (xi, W / 2 + dx) if s > 0 else (-W / 2 - dx, -xi)
            cnt[0] += 1
            stones.append(_stone(f"cheek{s}_{cnt[0]}", poly, x0, x1, sd + 11 * cnt[0] + (40 if s > 0 else 0), sub=sub, spall=spall))

        n0 = 2 if D < 1.2 else 3
        js = [yk] + [yk - yk * k / n0 + rs.jitter(0, 0.06) for k in range(1, n0)] + [-0.001]
        for k in range(n0):
            ya, yb = js[k] + g / 2, js[k + 1] - (g / 2 if k < n0 - 1 else 0)
            add([(ya, 0), (yb, 0), (yb, c1 - g / 2), (ya, c1 - g / 2)])
        for ci in range(nup):
            za, zb = zc[ci], zc[ci + 1]
            y_s = y_under(za + emin)
            y_x = min(y_under(zb), -0.001)
            lw = y_x - y_s
            nw = 2 if lw > 0.62 else 1
            jo = [y_s] + [y_s + lw * k / nw + rs.jitter(0, 0.04) for k in range(1, nw)]
            if y_x < -0.14:
                lr_ = -0.001 - y_x
                nr = max(1, int(round(lr_ / 0.42)))
                jo += [y_x] + [y_x + lr_ * k / nr + rs.jitter(0, 0.05) for k in range(1, nr)]
            jo.append(-0.001)
            for k in range(len(jo) - 1):
                ya, yb = jo[k] + (g / 2 if k else 0), jo[k + 1] - (g / 2 if k < len(jo) - 2 else 0)
                ha, hb = min(zb, under(ya)) - g / 2, min(zb, under(yb)) - g / 2
                add([(ya, za + g / 2), (yb, za + g / 2), (yb, hb), (ya, ha)])
        ncp = 2 if D < 1.2 else 3
        jb = [yk] + [yk - yk * k / ncp + rs.jitter(0, 0.07) for k in range(1, ncp)] + [-0.001]
        off = tc * ca * sa                        # coping joints are cut square to the rake
        for k in range(ncp):
            ya, yb = jb[k] + g / 2, jb[k + 1] - (g / 2 if k < ncp - 1 else 0)
            yat, ybt = ya - (off if k else 0), yb - (off if k < ncp - 1 else 0)
            sp = None
            if rs.chance(0.45):
                sp = (Vector((s * (W / 2 + 0.008), ya, zs(yat))), rs.u(0.06, 0.09), rs.u(0.015, 0.028))
            add([(ya, under(ya) + g / 2), (yb, under(yb) + g / 2), (ybt, zs(ybt)), (yat, zs(yat))], out=0.008, spall=sp)
        x0, x1 = (xi, W / 2) if s > 0 else (-W / 2, -xi)
        cores.append(_prism(f"core_cheek{s}", [(yk - 0.002, 0), (-0.0005, 0), (-0.0005, zs(-0.0005) - rec), (yk - 0.002, zs(yk) - rec)],
                            x0 + rec, x1 - rec, "X"))

    # ------------------------------------------------------------------ inside the well
    going = 0.0
    if is_open:
        # a threshold stone level with the back of the kerb, then a flight of shallow worn treads stepping
        # down towards the cellar door. The fall is kept flatter than a passer-by's line of sight over the
        # kerb, so the nosings read from the street; each tread is dirtier and darker than the last.
        yb0 = -0.13
        nst = int(clamp(round((yb0 - 0.006 - yk) / 0.25), 3, 6))
        going = (yb0 - 0.006 - yk) / nst
        treads = []
        rise_t = clamp(0.2 * going, 0.04, 0.06)
        for k in range(nst):
            ztop = max(0.03, zs(yk) - 0.012 - k * rise_t)
            ya, yb = yk + k * going + 0.004, (yk + (k + 1) * going if k < nst - 1 else -0.068)
            xj = sF * xi * rs.u(0.2, 0.45)        # joint lies under the shut leaf
            dk = k / (nst - 1)

            def hollow(v, ztop=ztop, ya=ya, yb=yb):
                if v.z < ztop - 0.001:
                    return v
                q = min(0.011, 0.3 * ztop) * exp(-((v.x - xfoot) / 0.2) ** 2) * (0.35 + 0.65 * clamp((v.y - ya) / (yb - ya)))
                return Vector((v.x, v.y - (0.6 * q if v.y > yb - 0.001 else 0.0), v.z - q))

            for (x0, x1) in ((-xi + 0.003, xj - g / 2), (xj + g / 2, xi - 0.003)):
                vis = (x0 + x1) * sO > 0
                nose, foot = min(0.045, 0.3 * (yb - ya)), min(0.055, 0.3 * (yb - ya))   # own vertex rows for the nosing and the riser foot
                st = _stone(f"tread{k}", [(ya, 0), (yb, 0), (yb, ztop), (yb - nose, ztop), (ya + foot, ztop), (ya, ztop)] if vis else
                            [(ya, 0), (yb, 0), (yb, ztop), (ya, ztop)], x0, x1, sd + 90 + 3 * k + (1 if vis else 0),
                            ch=min(0.012, 0.3 * ztop), sub=0, dens=0.16 if vis else None, warp=hollow, rough=0.003 if vis else 0.0015)
                assign(st, step)
                # dirt packed against the riser behind each tread, the nosing trodden pale where the boots land
                mott = noise_mask(0.12, seed=sd + 150 + k, lo=0.3, hi=0.7)
                paint(st, grime=lambda co, n, k=k, ya=ya, yb=yb, dk=dk, mott=mott, foot=foot, nose=nose: np.clip(
                          0.04 + 0.6 * dk + (0.14 + 0.2 * mott(co, n)) * np.clip((yb - co[:, 1]) / nose - 0.5, 0, 1)
                          + (0.6 * np.clip(1.4 - (co[:, 1] - ya) / foot, 0, 1) if k else 0.0)
                          + 0.25 * np.clip(np.abs(co[:, 0] - xfoot) / xi - 0.45, 0, 1) + (1 - np.clip(n[:, 2], 0, 1)) * 0.35, 0, 0.97),
                      wear=lambda co, n, yb=yb, dk=dk, nose=nose: np.clip(n[:, 2], 0, 1) * (0.35 + 0.65 * np.exp(-((co[:, 0] - xfoot) / 0.32) ** 2))
                      * (0.12 + 0.88 * np.clip(1.5 - (yb - co[:, 1]) / nose, 0, 1)) * (1.0 - 0.55 * dk))
                treads.append(st)
        # back wall under the house: stone jambs, oak lintel and the head of the cellar door
        zl0 = clamp(H1 - 0.36, 0.2, 0.52)
        zl1 = zl0 + 0.12
        dwh = min(0.4, xi * 0.62)
        back = []
        for s in (-1, 1):
            x0, x1 = (dwh, xi - 0.003) if s > 0 else (-xi + 0.003, -dwh)
            zj = zl0 * rs.u(0.45, 0.6)
            back.append(_stone(f"jamb{s}a", [(yb0, 0), (-0.001, 0), (-0.001, zj - g / 2), (yb0, zj - g / 2)], x0, x1, sd + 120 + s, sub=0))
            back.append(_stone(f"jamb{s}b", [(yb0, zj + g / 2), (-0.001, zj + g / 2), (-0.001, zl0 - 0.003), (yb0, zl0 - 0.003)], x0, x1, sd + 125 + s, sub=0))
            cores.append(box(f"core_jamb{s}", (x1 - x0 - 0.016, 0.11, zl0 - 0.02), loc=((x0 + x1) / 2, -0.062, zl0 / 2)))
        xm = rs.jitter(0, xi * 0.3)
        for k, (x0, x1) in enumerate(((-xi + 0.003, xm - g / 2), (xm + g / 2, xi - 0.003))):
            back.append(_stone(f"overl{k}", [(yb0, zl1 + g / 2), (-0.001, zl1 + g / 2), (-0.001, H1 - 0.006), (yb0, zs(yb0) - 0.006)], x0, x1, sd + 130 + k, sub=0))
        cores.append(box("core_over", (2 * xi - 0.016, 0.11, zs(yb0) - zl1 - 0.02), loc=(0, -0.062, (zl1 + zs(yb0)) / 2 - 0.004)))
        for o in back:
            assign(o, stone)
            paint(o, grime=lambda co, n: np.clip(0.45 + 0.5 * np.clip(1 - co[:, 2] / max(zl0, 0.25), 0, 1), 0, 0.97))
        lintel = box("lintel", (2 * xi - 0.008, 0.14, 0.12), loc=(0, yb0 + 0.05, zl0 + 0.058), bevel=0.007)
        displace_noise(lintel, strength=0.002, scale=0.2, seed=sd + 140)
        uv_grain(lintel, 'X')
        assign(lintel, oak)
        paint(lintel, grime=0.55, wear=mask_mul(noise_mask(0.05, seed=sd + 141, lo=0.45, hi=0.8), 0.5))
        nb = max(3, int(round(2 * dwh / 0.17)))
        bw = 2 * dwh / nb
        cdoor = []
        for k in range(nb):
            bd = box(f"cdoor{k}", (bw - 0.005, 0.022, zl0 - 0.008), loc=(-dwh + bw * (k + 0.5), -0.05 + rs.jitter(0, 0.004), zl0 / 2 + 0.002))
            uv_grain(bd, 'Z')
            cdoor.append(bd)
        cdoor.append(box("cdoor_ledge", (2 * dwh - 0.03, 0.02, 0.09), loc=(0, -0.071, zl0 - 0.09)))
        uv_grain(cdoor[-1], 'X')
        for o in cdoor:
            assign(o, oak)
            paint(o, grime=lambda co, n: np.clip(0.55 + 0.6 * (1 - co[:, 2] / zl0), 0, 1))
    else:
        vfloor = box("void_floor", (2 * xi + 0.004, D - tkf - 0.004, 0.006), loc=(0, (yk - 0.002) / 2, 0.003))
        assign(vfloor, mat("plain", "#000000"))     # only blocks the view of the paving through the board joints

    # mortar hearting, set back behind the stone faces so every joint is a recessed lime joint
    moss = noise_mask(0.1, seed=sd + 3, lo=0.3, hi=0.6)
    for o in cores:
        densify(o, 0.45)
        assign(o, mortar)
        paint(o, grime=lambda co, n: np.clip(0.08 + 0.5 * np.clip(1 - co[:, 2] / 0.3, 0, 1) + (np.abs(co[:, 0]) < xi) * (co[:, 1] > yk) * 0.4, 0, 0.95),
              special=lambda co, n: np.clip(1 - co[:, 2] / 0.35, 0, 1) * moss(co, n) * (np.abs(co[:, 0]) > xi - 0.02))

    def st_grime(co, n):
        z = co[:, 2]
        inside = (np.abs(co[:, 0]) < xi + 0.02) & (co[:, 1] > yk + 0.01)
        return np.clip(np.clip(1 - z / 0.3, 0, 1) * 0.7 + np.clip(1 + co[:, 1] / 0.18, 0, 1) * 0.3
                       + inside * np.clip(0.3 + (co[:, 1] - yk) / D * 0.6 + (0.4 - z) * 0.7, 0, 0.9), 0, 0.95)

    def st_moss(co, n):
        outside = ~((np.abs(co[:, 0]) < xi + 0.02) & (co[:, 1] > yk + 0.01))
        low = np.clip(1 - co[:, 2] / 0.26, 0, 1)
        wj = np.clip(1 + co[:, 1] / 0.25, 0, 1) * 0.85
        return np.clip((low + wj) * (0.3 + 0.7 * moss(co, n)), 0, 1) * outside

    for o in stones:
        assign(o, stone)
        paint(o, grime=rs.u(0.0, 0.24))               # every stone weathers to its own tone
        paint(o, grime=st_grime, special=st_moss,
              wear=lambda co, n: np.clip(n[:, 2], 0, 1) * (co[:, 1] < -D + 0.03) * np.exp(-((co[:, 0] - xfoot) / 0.3) ** 2) * 0.8)

    # dirt, silt and moss packed into the angle with the paving; lime fillet against the house wall
    yf = -D - pr
    xs = W / 2 + 0.004
    per = [(-xs, -0.002, -1, 0), (-xs, yf, -1, 0), (-xs, yf, -0.7, -0.7), (-xs, yf, 0, -1), (xs, yf, 0, -1), (xs, yf, 0.7, -0.7), (xs, yf, 1, 0), (xs, -0.002, 1, 0)]
    A, B = [], []
    rd = rng.fork("dirt")
    for k in range(len(per) - 1):
        x0, y0, nx0, ny0 = per[k]
        x1, y1, nx1, ny1 = per[k + 1]
        n = max(1, int(hypot(x1 - x0, y1 - y0) / 0.075))
        for i in range(n + (1 if k == len(per) - 2 else 0)):
            f = i / n
            x, y, nx, ny = lerp(x0, x1, f), lerp(y0, y1, f), lerp(nx0, nx1, f), lerp(ny0, ny1, f)
            h, w = rd.u(0.018, 0.05), rd.u(0.018, 0.055)
            A.append((x - nx * 0.007, min(y - ny * 0.007, -0.001), h))
            B.append((x + nx * w, min(y + ny * w, -0.001), 0.0015))
    skirt = _ribbon("skirt", A, B, (-1, 0, 1))
    fillets = [skirt]
    for s in (-1, 1):
        A, B = [], []
        n = max(4, int(H1 / 0.085))
        for i in range(n + 1):
            z = lerp(0.0, zs(0.0) - 0.004, i / n)
            w = rd.u(0.016, 0.03)
            xb = W / 2 + (0.0095 if z > under(0.0) - 0.01 else 0.002)
            A.append((s * (xb + w), -0.0008, z))
            B.append((s * (xb - 0.004), -w * 1.25, z))
        fillets.append(_ribbon(f"wallfillet{s}", A, B, (s, -1, 0)))
    for o in fillets:
        assign(o, mortar)
    paint(skirt, grime=0.85, special=lambda co, n: 0.45 + 0.55 * moss(co, n))
    paint(fillets[1:], grime=lambda co, n: np.clip(0.25 + 0.6 * np.clip(1 - co[:, 2] / 0.35, 0, 1), 0, 1),
          special=lambda co, n: np.clip(1 - co[:, 2] / 0.45, 0, 1) * moss(co, n))

    # ------------------------------------------------------------------ oak curb frame (rebated side rails, sill, head)
    fw, fh, t, inset, up = 0.11, 0.06, 0.032, 0.02, 0.046
    xr1 = W / 2 - inset
    xr0 = xr1 - fw
    xo = xr1 - up - 0.003                         # outer edge of a leaf
    xh = xr1 - up - 0.0015                        # hinge axis
    zt = fh + t                                   # top face of the leaves (flat frame)
    zax = zt + 0.0085
    ytop = L - 0.10
    zh = zt + 0.012
    frame = []
    for s in (-1, 1):
        pts = [(xr0, 0), (xr1, 0), (xr1, zt), (xr1 - up, zt), (xr1 - up, fh), (xr0, fh)]
        if s < 0:
            pts = [(-x, z) for x, z in reversed(pts)]
        r = _prism(f"rail{s}", pts, 0.0, ytop + 0.01, "Y")
        bevel(r, 0.004, segments=2)
        uv_grain(r, 'Y')
        frame.append(r)
    sill = box("sill", (2 * xr0 + 0.004, 0.1, fh), loc=(0, 0.055, fh / 2), bevel=0.005)
    uv_grain(sill, 'X')
    frame.append(sill)
    _xf(frame, SL)
    p4 = S(ytop, zh)
    head = _prism("head", [S(ytop, 0), (-0.001, H1), (-0.001, p4[1] + (-0.001 - p4[0]) * sa / ca), p4], -xr1, xr1, "X")
    bevel(head, 0.004, segments=2)
    uv_grain(head, 'X')
    frame.append(head)
    for o in frame:
        assign(o, oak)
        paint(o, grime=mask_mul(noise_mask(0.2, seed=sd + 5, lo=0.25, hi=0.75), 0.6), wear=mask_mul(noise_mask(0.06, seed=sd + 6, lo=0.5, hi=0.9), 0.12 + 0.3 * wr))

    # ------------------------------------------------------------------ lead flashing: upstand pointed into the wall, apron
    # dressed down over the head rail and finished with a welted drip; two sheets lapped
    ywf = (D - 0.004 + zh * sa) / ca               # flat y where the head's top face meets the wall
    zw = S(ywf, zh + 0.0035)[1]
    hu = 0.075
    nodes = [(-0.003, zw + hu), (-0.003, zw + 0.034), (-0.0048, zw + 0.013), S(ywf - 0.017, zh + 0.0042), S(ytop + 0.022, zh + 0.0034),
             S(ytop + 0.004, zh + 0.0046), S(ytop - 0.006, zh + 0.0026), S(ytop - 0.0105, zh - 0.004), S(ytop - 0.0085, zh - 0.0105)]
    fr = rng.fork("lead")
    xle = xr1 + 0.014
    xlap = fr.jitter(0, xle * 0.35)
    sheets = []
    for k, (xa, xb) in enumerate(((-xle, xlap + 0.035), (xlap, xle))):
        ph = [fr.u(0, 6.28) for _ in range(4)]

        def fl(u, v, xa=xa, xb=xb, ph=ph, k=k):
            j = min(int(round(v * (len(nodes) - 1))), len(nodes) - 1)
            y, z = nodes[j]
            x = lerp(xa, xb, u)
            d = 0.0011 * sin(x * 23 + ph[0]) + 0.0007 * sin(x * 61 + ph[1])     # bossing-mallet dents
            if j >= 5:
                z += 0.0018 * sin(x * 13 + ph[2]) + d                              # soft, wandering drip edge
            elif j >= 3:
                z += d
            else:
                y -= abs(d) * 0.5
            if k:
                y -= 0.0027
                z += 0.0027
            return (x, y, z)

        sh = surface(f"flashing{k}", fl, max(4, int((xb - xa) / 0.15)), len(nodes) - 1, thickness=0.0026)
        smooth(sh, 70)
        assign(sh, lead)
        sheets.append(sh)
    lstreak = _aniso(0.02, 9.0, sd + 8, 0.3, 0.7)
    paint(sheets, grime=lambda co, n: np.clip(0.5 + 0.4 * lstreak(co) - 0.25 * (co[:, 2] > zw + 0.03) * (1 - lstreak(co)), 0.15, 0.95))
    A, B = [], []
    n = max(8, int(2 * xle / 0.07))
    pf = [fr.u(0, 6.28) for _ in range(2)]
    for i in range(n + 1):
        x = lerp(-xle - 0.012, xle + 0.012, i / n)
        wv = 0.003 * sin(x * 9 + pf[0]) + 0.0015 * sin(x * 31 + pf[1])
        A.append((x, -0.0006, zw + hu + 0.015 + wv + fr.u(-0.0015, 0.0015)))
        B.append((x, -0.0125 + fr.u(-0.001, 0.001), zw + hu - 0.009 + 0.6 * wv))
    raggle = _ribbon("raggle_fillet", A, B, (0, -1, 0.3))
    assign(raggle, mortar)
    paint(raggle, grime=lambda co, n: 0.3 + 0.45 * noise_mask(0.12, seed=sd + 9, lo=0.3, hi=0.7)(co, n))

    # ------------------------------------------------------------------ leaves
    x_in = 0.003
    wl = xo - x_in
    y0l, y1l = -0.035, ytop - 0.012
    yl = [0.2, (0.2 + ytop - 0.17) / 2, ytop - 0.17]           # ledge positions up the slope
    xa_l, xb_l = 0.03, xr0 - 0.012                             # ledge extent (clear of the side rail)
    nail = cylinder("nail", 0.005, 0.0026, loc=(0, 0, 0.0013), verts=3, radius2=0.0032, cap=False)
    sbolt = bolt("sbolt", 0.0075, 0.005, kind="pyramid")
    rbolt = cylinder("rbolt", 0.006, 0.004, loc=(0, 0, 0.002), verts=6, radius2=0.004)
    yh = 0.56 * ytop                                           # hasp height up the slope
    yrp = 0.34 * ytop                                          # ring pull height up the slope
    gap = 0.0035

    def leaf_matrix(s):
        if s == sO and is_open:
            return SL @ _T(s * xh, 0, zax) @ _R(s * theta, 'Y') @ _T(-s * xh, 0, -zax)
        return SL.copy()

    def hang_angle(M, lr, jit):
        gl = M.to_3x3().transposed() @ Vector((0, 0, -1))
        return degrees(atan2(gl.y, gl.x)) + 90.0 + lr.jitter(0, jit)

    def ring_pull(name, pos, ang):
        plate = lathe(name + "_plate", [(0, 0), (0.03, 0), (0.03, 0.0025), (0.026, 0.004), (0, 0.004)], segments=14)
        eye = sweep(name + "_eye", arc_path(0.011, 0, 180, n=6, plane="YZ", center=(0, 0, 0.004)), 0.0032, sides=6, smooth_path=False)
        ring = torus(name + "_ring", major=0.04, minor=0.0046, loc=(0, -0.036, 0.0069), rot=(3, 0, 0), major_seg=14, minor_seg=5)
        smooth(ring, 60)
        smooth(eye, 60)
        grp = [plate, eye, ring]
        for o in grp:
            assign(o, iron)
        _xf(grp, _T(*pos) @ _R(ang, 'Z'))
        return grp, ring

    def leaf_masks(s, tears, rub):
        """Weathering in the leaf's own flat frame (|x| from the meeting edge, y up the slope). Everything
        is built from per-column noise stretched along the boards, so the paint goes in tongues and
        streaks along the grain / down the fall, never in round clouds."""
        q = sd + (40 if s > 0 else 0)
        col = _aniso(0.014, 0.9, q + 13, 0.3, 0.7)        # one value per vertex column: how far the bare tongue runs up
        strk = _aniso(0.016, 0.7, q + 11, 0.0, 1.0)       # long streaks where the paint has gone thin
        fine = _aniso(0.02, 0.07, q + 12, 0.3, 0.75)      # breaks up the worn edges
        tcol = _aniso(0.011, 1.6, q + 14, 0.3, 0.65)
        thr = 0.8 - 0.42 * wr

        def wear(co, n, kw):
            ax, y = np.abs(co[:, 0]), co[:, 1]
            u = (y - y0l) / (y1l - y0l)
            cl, fi = col(co), fine(co)
            face = np.abs(n[:, 0]) < 0.2
            # boots and barrels: paint gone from the bottom end up to a ragged tide line, highest in the middle of the leaf
            tide = kw * ((0.02 + 0.3 * wr) + (0.07 + 0.36 * wr) * cl) * (1.1 - 0.35 * np.clip(ax / wl, 0, 1))
            v = np.clip((tide + 0.035 - u) / 0.13, 0, 1)
            v = v + 0.95 * clamp((wr - 0.5) / 0.3) * kw * np.clip((strk(co) - thr) / 0.06, 0, 1) * np.clip(1.3 - u, 0, 1)
            v = v + 0.6 * np.clip(1 - (y1l - y) / 0.04, 0, 1)
            v = v + (0.35 + 0.65 * wr) * np.clip(1 - (ax - x_in) / 0.03, 0, 1) * (0.6 + 0.4 * fi)        # hands on the meeting edge
            for (cx_, cy_, r) in rub:                                                    # ring / padlock scuffing bare wood
                v = v + np.clip(1.3 - np.hypot(ax - cx_, y - cy_) / r, 0, 1) * (0.3 + 0.4 * wr)
            arris = np.clip(0.2 + 1.1 * wr, 0, 1) * (0.45 + 0.55 * fi) * np.clip(kw, 0.5, 1.0)   # every plank edge rubbed through
            v = np.where(face, v, np.maximum(v, arris))
            return np.clip(v * np.where(n[:, 2] < -0.5, 0.25, 1.0), 0, 1)

        def grime(co, n):
            ax, y = np.abs(co[:, 0]), co[:, 1]
            u = (y - y0l) / (y1l - y0l)
            cl, tq = col(co), tcol(co)
            side = np.clip((np.abs(n[:, 0]) - 0.2) / 0.5, 0, 1)                           # arrises and the joints between boards
            v = 0.06 + tq * (0.06 + 0.3 * np.clip(1 - u, 0, 1)) + 0.65 * np.clip(1 - (y - y0l) / 0.08, 0, 1)   # run-off, dirt on the bottom rail
            v = v + side * (0.3 + 0.45 * np.clip(1.2 - u, 0, 1))
            v = v + (0.2 + 0.2 * tq) * np.clip((((0.02 + 0.3 * wr) + (0.07 + 0.36 * wr) * cl) - u) / 0.1, 0, 1)   # bare boards weather grey
            for (xa, xb, yt_, ln) in tears:                                              # rust tears below the ironwork
                inx = np.clip((ax - xa) / 0.01 + 1, 0, 1) * np.clip((xb - ax) / 0.01 + 1, 0, 1)
                d = yt_ - y
                v = v + 0.8 * inx * (d > -0.005) * np.clip(1 - d / (ln * (0.4 + 1.2 * tq)), 0, 1) * (0.35 + 0.65 * tq)
            return np.clip(v * (0.8 + 0.4 * wr), 0, 0.95)

        return wear, grime

    def make_leaf(s):
        lr = rng.fork(f"leaf{s}")
        M = leaf_matrix(s)
        opened = s == sO and is_open
        fixed, planks, irons = [], [], []
        npl = max(3, int(round(wl / 0.145)))
        ws = [lr.vary(1.0, 0.14) for _ in range(npl)]
        k = wl / sum(ws)
        ed = [x_in]
        for w in ws:
            ed.append(ed[-1] + w * k)
        cx = [((ed[i] + ed[i + 1]) / 2, ed[i + 1] - ed[i]) for i in range(npl)]
        Ls = {j: wl * lr.u(0.66, 0.76) for j in (0, 2)}
        xrp = min(0.16, wl * 0.35)
        tears = [(xo - Ls[j], xo, yl[j] - 0.015, 0.1 + 0.2 * wr) for j in (0, 2)]
        tears.append((xrp - 0.028, xrp + 0.028, yrp - 0.07, 0.08 + 0.1 * wr))
        tears.append((0.0, 0.085, yh - 0.02, 0.1 + 0.12 * wr) if s == sO else (0.02, 0.06, yh - 0.075, 0.08 + 0.1 * wr))
        rub = [(xrp, yrp - 0.045, 0.052)]
        if s != sO:
            rub.append((0.04, yh - 0.055, 0.042))
        wear_fn, grime_fn = leaf_masks(s, tears, rub)
        # boards running up the slope; with age a joint opens and a board splits from its lower end
        mid = list(range(1, npl - 1)) or [0]
        iopen = lr.pick(mid) if wr > 0.5 else -1
        isplit = lr.pick([i for i in mid if i != iopen] or mid) if wr > 0.62 else -1
        inew = lr.pick([i for i in range(npl) if i not in (iopen, isplit)]) if lr.chance(0.7) else -1   # one board replaced not so long ago
        ny = int(clamp((y1l - y0l) / 0.066, 8, max(12, 84 // npl)))
        for i in range(npl):
            a0 = ed[i] + (gap / 2 if i else 0)
            b0 = ed[i + 1] - (gap / 2 if i < npl - 1 else 0)
            opn = 0.004 + 0.012 * (wr - 0.5) / 0.5 if i == iopen else 0.0
            pieces = [(a0, b0 - opn, a0, b0)]
            if i == isplit:
                m0 = a0 + (b0 - a0) * lr.u(0.38, 0.62)
                m1 = m0 + lr.jitter(0, 0.018)
                pieces = [(a0, m0 - 0.003, a0, m1 - 0.0007), (m0 + 0.003, b0, m1 + 0.0007, b0)]
            dy = lr.u(0.0, 0.008 + 0.014 * wr)
            zj = lr.jitter(0, 0.0012)
            kw = 0.35 if i == inew else lr.u(0.75, 1.2)
            fade = (0.1 if i == inew else lr.u(0.3, 0.75)) * (0.35 + 0.65 * wr)
            cup, bow, lift = lr.jitter(0, 0.0012), lr.jitter(0, 0.0012), wr * lr.u(0, 0.004)
            for q, (pa0, pb0, pa1, pb1) in enumerate(pieces):
                xl0, xr0_ = sorted((s * pa0, s * pb0))
                xl1, xr1_ = sorted((s * pa1, s * pb1))
                wb = pb0 - pa0
                bd = _board(f"plank{s}_{i}_{q}", xl0, xr0_, xl1, xr1_, y0l + dy, y1l - lr.u(0, 0.003), fh + zj, t - zj, ny,
                            1 if wb < 0.1 else (2 if wb < 0.19 else 3), opened, cup=cup, bow=bow, lift=lift)
                assign(bd, wood_p)
                paint(bd, wear=lambda co, n, kw=kw: wear_fn(co, n, kw), grime=grime_fn, special=fade)
                planks.append(bd)
        # ledges and braces on the underside (only where they can ever be seen: under the propped leaf)
        if opened:
            under_ = []
            for j, y in enumerate(yl):
                under_.append(plank(f"ledge{s}_{j}", length=xb_l - xa_l, width=0.095, thick=0.028,
                                    loc=(s * (xa_l + xb_l) / 2, y, fh - 0.0145), bevel=0.005, rng=lr, axis='X'))
            for j in range(2):
                ax, ay = xb_l - 0.05, yl[j] + 0.0475
                bx, by = xa_l + 0.05, yl[j + 1] - 0.0475
                ln = hypot(bx - ax, by - ay)
                under_.append(plank(f"brace{s}_{j}", length=ln + 0.03, width=0.075, thick=0.026,
                                    loc=(s * (ax + bx) / 2, (ay + by) / 2, fh - 0.0135),
                                    rot=(0, 0, degrees(atan2(by - ay, s * (bx - ax)))), bevel=0.004, rng=lr, axis='X'))
            for o in under_:
                assign(o, wood_p)
                paint(o, grime=0.3, wear=0.08 + 0.15 * wr, special=0.1)
            planks += under_
        # clench nails through every ledge
        nm = []
        for j, y in enumerate(yl):
            for i, (c, w) in enumerate(cx):
                for q in (-1, 1):
                    nm.append(trs((s * (c + q * w * 0.24 + lr.jitter(0, 0.004)), y + q * 0.031 * (1 if i % 2 else -1) + lr.jitter(0, 0.004), zt - 0.0004),
                                  (lr.jitter(0, 6), lr.jitter(0, 6), lr.u(0, 90))))
        irons.append(instances_to_mesh(f"nails{s}", nail, nm))
        # strap hinges over the bottom and top ledges
        bm_, fm = [], []
        for j in (0, 2):
            y = yl[j]
            Lj = Ls[j]
            hw0, hw1 = 0.024, 0.0155
            upper = [(0.0, hw0), (Lj - 0.085, hw1), (Lj - 0.068, hw1 + 0.004), (Lj - 0.05, 0.028), (Lj - 0.03, 0.021), (Lj - 0.012, 0.009), (Lj, 0.0)]
            lower = [(d, -c) for d, c in reversed(upper[:-1])]
            out = [(s * (xo - d), y + c) for d, c in upper + lower]
            if s < 0:
                out.reverse()
            strap = shape_extrude(f"strap{s}_{j}", out, depth=0.0055, loc=(0, 0, zt - 0.0005), bevel=0.0012)
            irons.append(strap)
            for (c, w) in cx:
                d = xo - c
                if 0.03 < d < Lj - 0.09:
                    bm_.append(trs((s * c, y + lr.jitter(0, 0.002), zt + 0.0048), (0, 0, lr.u(0, 90))))
            bm_.append(trs((s * (xo - Lj + 0.047), y, zt + 0.0048), (0, 0, lr.u(0, 90))))
            irons.append(cylinder(f"knuckle{s}_{j}", 0.0105, 0.022, loc=(s * xh, y, zax), rot=(90, 0, 0), verts=10))
            # fixed half: plate on the rail rebate, outer knuckles and the pin
            fixed.append(box(f"hplate{s}_{j}", (0.04, 0.07, 0.005), loc=(s * (xh + 0.0215), y, zt + 0.0022)))
            for q in (-1, 1):
                fixed.append(cylinder(f"knuckf{s}_{j}{q}", 0.0105, 0.02, loc=(s * xh, y + q * 0.0225, zax), rot=(90, 0, 0), verts=10))
                fm.append(trs((s * (xh + 0.026), y + q * 0.021, zt + 0.0045), (0, 0, lr.u(0, 90))))
            fixed.append(cylinder(f"pin{s}_{j}", 0.0048, 0.078, loc=(s * xh, y, zax), rot=(90, 0, 0), verts=7))
        irons.append(instances_to_mesh(f"sbolts{s}", sbolt, bm_))
        fixed.append(instances_to_mesh(f"fbolts{s}", rbolt, fm))
        # ring pull, hanging down the slope from its eye
        grp, ring = ring_pull(f"pull{s}", (s * xrp, yrp, zt), hang_angle(M, lr, 9))
        irons += grp
        # hasp (on the leaf that opens) / staple and padlock (on the fixed leaf)
        lock = []
        if s == sO:
            c = -s
            hp = [box("hasp_plate", (0.04, 0.05, 0.004), loc=(-0.026, 0, 0.002), bevel=0.001),
                  cylinder("hasp_knuckle", 0.006, 0.05, loc=(0, 0, 0.006), rot=(90, 0, 0), verts=10)]
            hp.append(instances_to_mesh("hasp_bolts", rbolt, [trs((-0.03, q * 0.014, 0.0035)) for q in (-1, 1)]))
            outer = [(0.0, -0.017), (0.095, -0.017), (0.113, -0.012), (0.126, -0.004), (0.126, 0.004), (0.113, 0.012), (0.095, 0.017), (0.0, 0.017)]
            slot = [(0.1 + px, py) for px, py in rounded_rect(0.012, 0.029, 0.004, 3)]
            hstrap = shape_extrude("hasp_strap", outer, [slot], depth=0.004, loc=(0, 0, 0.004), bevel=0.0008)
            if is_open:                              # door open: the hasp has flopped back over its own hinge
                _xf([hstrap], _T(0, 0, 0.006) @ _R(180 + lr.u(4, 10), 'Y') @ _T(0, 0, -0.006))
            hp.append(hstrap)
            _xf(hp, _T(s * 0.06, yh, zt) @ _R(0 if c > 0 else 180, 'Z'))
            irons += hp
        else:
            sp_ = [box("staple_plate", (0.03, 0.06, 0.003), loc=(0, 0, 0.0015), bevel=0.0008),
                   sweep("staple", [(0, -0.011, 0.001), (0, -0.011, 0.014), (0, -0.008, 0.022), (0, 0, 0.0255), (0, 0.008, 0.022),
                                    (0, 0.011, 0.014), (0, 0.011, 0.001)], 0.0035, sides=6, smooth_path=False)]
            smooth(sp_[1], 60)
            _xf(sp_, _T(s * 0.04, yh, zt))
            irons += sp_
            zc_ = 0.0125
            sh = sweep("shackle", [(-0.014, -0.036, zc_), (-0.014, -0.013, zc_), (-0.0105, -0.004, zc_), (0, 0.0008, zc_), (0.0105, -0.004, zc_),
                                   (0.014, -0.013, zc_), (0.014, -0.036, zc_)], 0.0042, sides=7, step=0.007)
            smooth(sh, 60)
            body = rounded_box("padlock", (0.054, 0.048, 0.02), radius=0.011, loc=(0, -0.056, 0.0108), seg=3, bevel=0.002)
            smooth(body, 40)
            esc = lathe("escutcheon", [(0, 0), (0.0095, 0), (0.0095, 0.0015), (0.007, 0.0026), (0, 0.0026)], segments=12, loc=(0, -0.058, 0.0205))
            drop = box("keydrop", (0.007, 0.017, 0.0016), loc=(0.002, -0.063, 0.0238), rot=(0, 0, 14), bevel=0.0005)
            riv = instances_to_mesh("lock_rivets", cylinder("lr", 0.0026, 0.0016, loc=(0, 0, 0.0008), verts=6, radius2=0.0016),
                                    [trs((qx * 0.018, -0.056 + qy * 0.015, 0.0206)) for qx in (-1, 1) for qy in (-1, 1)], keep_base=False)
            for o in (body, riv):
                assign(o, brass)
            for o in (sh, esc, drop):
                assign(o, iron)
            lock = [sh, body, esc, drop, riv]
            _xf(lock, _T(s * 0.04, yh, zt) @ _R(hang_angle(M, lr, 12), 'Z'))
        for o in irons + fixed:
            if not o.material_slots or o.material_slots[0].material is None:
                assign(o, iron)
        _xf(planks + irons + lock, M)
        _xf(fixed, SL)
        return irons + fixed, lock, ring

    rust = noise_mask(0.035, seed=sd + 14, lo=0.62 - 0.4 * wr, hi=0.95 - 0.35 * wr)
    for s in (-1, 1):
        irons, lock, ring = make_leaf(s)
        for o in irons:
            paint(o, special=rust)
        paint(ring, wear=0.75)                       # hands keep the ring bright
        for o in lock:
            paint(o, special=mask_mul(rust, 0.6), wear=mask_mul(noise_mask(0.01, seed=sd + 16), 0.5))

    # ------------------------------------------------------------------ prop stick under the open leaf
    if is_open:
        Mh = _T(sO * xh, 0, zax) @ _R(sO * theta, 'Y') @ _T(-sO * xh, 0, -zax)
        top = Mh @ Vector((sO * 0.07, 0.13, fh - 0.001))
        foot = Vector((top.x - sO * min(0.07, top.z * 0.2), 0.06, fh - 0.002))
        prop = sweep("prop", [tuple(foot), tuple(top)], rounded_rect(0.038, 0.03, 0.004, 2), smooth_path=False, up=(0, 1, 0))
        assign(prop, oak)
        _xf([prop], SL)
        paint(prop, wear=mask_mul(noise_mask(0.05, seed=sd + 20, lo=0.4, hi=0.8), 0.6), grime=0.2)

    # ------------------------------------------------------------------ forged swan-neck lantern bracket on the wall
    # The bar runs straight from the wall plate at the flame and only curls over the lantern close to it,
    # so the candle throws no bar-shaped shadow up the wall.
    ly = -clamp(0.5 * D, 0.5, 0.66)
    zF = H1 + 0.53                                 # flame height
    zr = zF - 0.17                                 # arm root on the wall plate
    ph0 = atan2(zr - zF, -ly)
    Lr = hypot(-ly, zr - zF)
    arm = [(0, -0.006, zr), (0, ly + (Lr * 0.6) * cos(ph0), zF + (Lr * 0.6) * sin(ph0))]
    na = 9
    for k in range(na + 1):
        f = k / na
        ang = lerp(ph0, radians(104), f)
        r = lerp(0.215, 0.158, f ** 0.8)
        arm.append((0, ly + r * cos(ang), zF + r * sin(ang)))
    arm += [(0, ly - 0.03, zF + 0.128), (0, ly - 0.006, zF + 0.109), (0, ly + 0.014, zF + 0.113), (0, ly + 0.02, zF + 0.126)]
    hook = [box("hook_plate", (0.044, 0.006, 0.17), loc=(0, -0.003, zr - 0.045), bevel=0.0012),
            sweep("hook_arm", arm, [(0.006, 0.006), (-0.006, 0.006), (-0.006, -0.006), (0.006, -0.006)], up=(1, 0, 0), step=0.045,
                  scale_fn=lambda q: 1.0 if q < 0.8 else lerp(1.0, 0.55, (q - 0.8) / 0.2)),
            sweep("hook_stay", bezier_path((0, -0.006, zr - 0.118), (0, -0.075, zr - 0.118), (0, ly + (Lr * 0.6) * cos(ph0), zF + (Lr * 0.6) * sin(ph0) - 0.075),
                                           (0, ly + (Lr * 0.66) * cos(ph0), zF + (Lr * 0.66) * sin(ph0) - 0.004), n=8), 0.005, sides=6, smooth_path=False),
            instances_to_mesh("hook_bolts", rbolt, [trs((0, -0.006, zr - 0.045 + qz * 0.068), (90, 0, 0)) for qz in (-1, 1)])]
    smooth(hook[2], 60)
    for o in hook:
        assign(o, iron)
        paint(o, special=rust)
    if p.lantern:
        lx = 0.0
        lz = zF - 0.066
        base = lathe("lan_base", [(0, 0), (0.046, 0), (0.048, 0.004), (0.048, 0.012), (0.041, 0.018), (0.035, 0.03), (0.0, 0.03)], segments=12, loc=(lx, ly, lz))
        cap = lathe("lan_cap", [(0.0, 0.118), (0.039, 0.118), (0.043, 0.123), (0.032, 0.137), (0.019, 0.147), (0.019, 0.16), (0.026, 0.164),
                                (0.013, 0.175), (0, 0.177)], segments=12, loc=(lx, ly, lz))
        glass = lathe("lan_glass", [(0.033, 0.03), (0.037, 0.072), (0.034, 0.118)], segments=12, loc=(lx, ly, lz))
        stub = lathe("lan_stub", [(0, 0.03), (0.011, 0.03), (0.011, 0.05), (0.0, 0.05)], segments=8, loc=(lx, ly, lz))
        flame = lathe("lan_flame", [(0, 0.05), (0.0055, 0.056), (0.0068, 0.064), (0.003, 0.078), (0, 0.087)], segments=8, loc=(lx, ly, lz))
        rod = box("lan_rod", (0.006, 0.0025, 0.092))
        rods = instances_to_mesh("lan_rods", rod, [trs((lx + 0.0405 * cos(radians(45 + 90 * k)), ly + 0.0405 * sin(radians(45 + 90 * k)), lz + 0.074),
                                                       (0, 0, 135 + 90 * k)) for k in range(4)], keep_base=False)
        band = ring_band("lan_band", radius=0.0385, width=0.006, thick=0.002, loc=(lx, ly, lz + 0.074), segments=12)
        bail = sweep("lan_bail", arc_path(0.046, 12, 168, n=10, plane="XZ", center=(lx, ly, lz + 0.128)), 0.002, sides=5, smooth_path=False)
        for o in (base, cap, stub):
            assign(o, brass)
            paint(o, grime=0.45, special=mask_mul(noise_mask(0.02, seed=sd + 30), 0.4))
        for o in (rods, bail, band):
            assign(o, iron)
        assign(glass, mat("glass_lantern", glow=1.0))
        assign(flame, mat("emissive", "#ffb45e", glow=7))
        part([base, cap, glass, stub, flame, rods, band, bail], "lantern", pivot=(lx, ly, lz + 0.174), motion="swing", amp=2.0, speed=0.3)
        light("lantern", (lx, ly, zF), intensity=p.lamp, range=4.5, flicker=0.35, shadow=True, shadow_near=0.24)

    delete([nail, sbolt, rbolt])
    socket("top", "attach", (0, -D * 0.5, (H0 + H1) / 2 + 0.11), role="prop_top")

