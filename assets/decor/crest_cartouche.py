"""Carved heraldic cartouche for above doors and on the town gate. A shaped two-layer back board carries a
C-scroll of moulded strapwork down each flank whose ends roll forward into volutes that hang past the board as
four ears; a cresting rises behind the scallop shell / coronet. On the board sits a domed shield with a
bolection rim; the charge (crown / key / star / lion mask) is carved in relief out of the field (a height
field with a filleted root), with palm sprays, a drop boss and a folded motto ribbon with raised lettering
below. Made as carved limestone (limewashed field, worn gold leaf), carved and painted wood, or a cast bronze
plaque (enamelled field, verdigris, fire-gilt highlights), held to the wall by visible studs. Chipped
arrises, a nicked volute, a snapped palm tip, flaked paint and lost gilding are seeded and scale with wear.
"""
from math import sin, cos, pi, radians, hypot

import numpy as np

from tk import *

META = {"name": "Heraldic crest cartouche", "class": "prop", "mount": "wall", "mount_height": 2.7, "focus": 0.55, "tri_budget": 18500}
PARAMS = {
    "size": {"default": 0.9, "min": 0.45, "max": 1.8, "desc": "overall height shell-top to ribbon (m)"},
    "charge": {"default": "crown", "choices": ["crown", "key", "star", "lion"], "desc": "charge carved on the shield"},
    "material": {"default": "stone", "choices": ["stone", "wood", "bronze"], "desc": "carved stone / painted wood / cast bronze"},
    "motto": {"default": "LUX IN TENEBRIS", "desc": "motto lettering on the ribbon (empty = plain ribbon; up to ~28 characters keeps it legible and in budget)"},
    "shield": {"default": "heater", "choices": ["heater", "french", "oval"], "desc": "shield outline"},
    "crest": {"default": "shell", "choices": ["shell", "coronet", "none"], "desc": "ornament above the shield (the board cresting follows it)"},
    "gilding": {"default": "accents", "choices": ["none", "accents", "rich"], "desc": "none / charge + letters / also rim, bosses and crest"},
    "painted": {"default": True, "desc": "paint (stone, wood) or enamel (bronze) the shield field"},
    "field": {"default": "#7d2320", "desc": "field colour of the shield"},
    "paint": {"default": "#22382f", "desc": "surround colour (painted wood only)"},
    "stone": {"default": "#c9bb9c", "desc": "stone tint (stone only)"},
    "width": {"default": 1.0, "min": 0.9, "max": 1.25, "desc": "relative width of the scrolled surround"},
    "ears": {"default": 1.0, "min": 0.7, "max": 1.45, "desc": "relative size of the four volute ears"},
    "wear": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "age: grime, moss / verdigris, chips, flaked paint and gilding"},
}
VARIANTS = {
    "gate_bronze_lion": {"size": 1.5, "material": "bronze", "charge": "lion", "shield": "oval", "crest": "coronet",
                         "gilding": "rich", "field": "#1d3557", "motto": "A BETTER TOMORROW", "width": 1.2, "ears": 1.35,
                         "wear": 0.6, "seed": 4},
    "wood_key": {"size": 0.7, "material": "wood", "charge": "key", "shield": "french", "crest": "shell", "gilding": "rich",
                 "field": "#1f3f6b", "paint": "#4d2620", "motto": "CLAVIS CIVITATIS", "width": 0.92, "ears": 0.8,
                 "wear": 0.3, "seed": 11},
    "old_stone_star": {"size": 0.6, "material": "stone", "charge": "star", "shield": "heater", "crest": "none",
                       "gilding": "none", "painted": False, "stone": "#b3aa96", "motto": "AD ASTRA", "width": 1.05, "ears": 1.2,
                       "wear": 1.0, "seed": 23},
    "wood_lion_cream": {"size": 1.1, "material": "wood", "charge": "lion", "shield": "french", "crest": "coronet",
                        "gilding": "accents", "field": "#24452f", "paint": "#cfc3a4", "motto": "LUX IN TENEBRIS",
                        "width": 1.1, "ears": 0.95, "wear": 0.7, "seed": 31},
}


# ------------------------------------------------------------------------------------------- 2D helpers
def _mirror_half(pts):
    """Half outline from bottom centre up the right side to top centre -> full CCW loop."""
    return list(pts) + [(-x, z) for (x, z) in reversed(list(pts)[1:-1])]


def _shield_outline(kind, w, h):
    a, top, bot = w / 2, h / 2, -h / 2
    pts = []
    if kind == "oval":
        m = 28
        for i in range(m + 1):
            t = radians(-90 + 180 * i / m)
            pts.append((a * 0.95 * cos(t), top * sin(t)))
        pts[0], pts[-1] = (0.0, bot), (0.0, top)
    elif kind == "french":
        zb, r = bot + 0.075 * h, 0.36 * a
        pts += [(0.0, bot), (0.07 * a, bot + 0.022 * h), (0.16 * a, bot + 0.058 * h), (0.3 * a, zb)]
        for i in range(8):
            t = radians(-90 + 90 * i / 7)
            pts.append((a - r + r * cos(t), zb + r + r * sin(t)))
        pts += [(a, top - 0.3 * h), (a, top - 0.12 * h), (a, top)]
        m = 12
        for i in range(1, m):
            pts.append((a * (1 - i / m), top - 0.055 * h * sin(pi * i / m)))
        pts.append((0.0, top))
    else:
        zs, m = top - 0.3 * h, 16
        for i in range(m + 1):
            s = 1 - i / m
            pts.append((a * (1 - s ** 1.65), zs - s * (zs - bot)))
        pts += [(a, top - 0.15 * h), (a, top - 0.014 * h), (a - 0.014 * h, top)]
        for i in range(1, 6):
            pts.append((a * (1 - i / 6), top))
        pts.append((0.0, top))
    return _mirror_half(pts)


def _smooth_path(pts, n):
    """Uniform Catmull-Rom through pts, resampled to n points evenly spaced by arc length."""
    P = np.asarray(pts, dtype=np.float64)
    dense = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, len(P) - 1)]
        for j in range(10):
            t = j / 10.0
            dense.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    dense.append(P[-1])
    D = np.asarray(dense)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))])
    tt = np.linspace(0.0, cum[-1], n)
    return list(zip(*[np.interp(tt, cum, D[:, k]) for k in range(D.shape[1])]))


def _seg_dist(X, Z, a, b):
    """Distance from (X, Z) arrays to segment a-b and the clamped parameter along it."""
    dx, dz = b[0] - a[0], b[1] - a[1]
    t = np.clip(((X - a[0]) * dx + (Z - a[1]) * dz) / (dx * dx + dz * dz + 1e-18), 0.0, 1.0)
    return np.hypot(X - a[0] - t * dx, Z - a[1] - t * dz), t


def _poly_sd(X, Z, pts):
    """Signed distance to a closed polygon, positive inside."""
    d = np.full(np.shape(X), 1e9)
    ins = np.zeros(np.shape(X), dtype=bool)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = np.minimum(d, _seg_dist(X, Z, a, b)[0])
        if a[1] != b[1]:
            ins ^= ((a[1] > Z) != (b[1] > Z)) & (X < (b[0] - a[0]) * (Z - a[1]) / (b[1] - a[1]) + a[0])
    return np.where(ins, d, -d)


def _smax(a, b, k):
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return b + (a - b) * h + k * h * (1.0 - h)


def _bump(q, base, h, pw=0.5):
    """Elliptic boss: q = squared elliptic radius (1 at the edge)."""
    return np.where(q < 1.0, base + h * np.clip(1.0 - q, 0.0, 1.0) ** pw, 0.0)


def _blur(H, n=1):
    for _ in range(n):
        P = np.pad(H, 1, mode="edge")
        H = (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] + 4.0 * P[1:-1, 1:-1]) / 8.0
    return H


def _blotch(C, f, s):
    """Cheap warped-sine blotches in -1..1 from (N, 3) positions (feature size ~ 3 / f)."""
    x, z = C[:, 0] * f, C[:, 2] * f
    return np.sin(x + 1.7 * np.sin(z * 0.9 + s)) * np.sin(z * 1.1 + 1.3 * np.sin(x * 0.8 + 2.0 * s))


# ------------------------------------------------------------------------------------------- mesh builders
def _grid_mesh(name, fn, us, vs):
    """Quad sheet fn(u, v) -> (x, y, z); the face normal is du x dv."""
    nu = len(us)
    verts = [fn(u, v) for v in vs for u in us]
    faces = [(j * nu + i, j * nu + i + 1, (j + 1) * nu + i + 1, (j + 1) * nu + i) for j in range(len(vs) - 1) for i in range(nu - 1)]
    return mesh_object(name, verts, faces)


def _relief(name, hfun, bounds, cell, base_fn, sink, tris=4000, blur=2):
    """Carved relief as a height field over the ground surface base_fn(X, Z) (depth in front of the wall).
    The sheet starts `sink` below the ground, so the charge grows out of the field with a filleted root and
    the hidden skirt needs no clipping. Decimated to about `tris` triangles (flat parts collapse first)."""
    x0, x1, z0, z1 = bounds
    nx, nz = max(4, int(round((x1 - x0) / cell))), max(4, int(round((z1 - z0) / cell)))
    X, Z = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(z0, z1, nz + 1))
    H = _blur(hfun(X, Z), blur)
    D = base_fn(X, Z) + H - sink
    up = H > sink * 0.35
    keep = up[:-1, :-1] | up[1:, :-1] | up[:-1, 1:] | up[1:, 1:]
    used = np.zeros(X.shape, dtype=bool)
    used[:-1, :-1] |= keep
    used[1:, :-1] |= keep
    used[:-1, 1:] |= keep
    used[1:, 1:] |= keep
    idx = -np.ones(X.shape, dtype=np.int64)
    idx[used] = np.arange(int(used.sum()))
    verts = np.stack([X[used], -D[used], Z[used]], axis=1)
    jj, ii = np.nonzero(keep)
    faces = np.stack([idx[jj, ii], idx[jj, ii + 1], idx[jj + 1, ii + 1], idx[jj + 1, ii]], axis=1)
    o = mesh_object(name, verts.tolist(), faces.tolist())
    ratio = tris / max(1.0, 2.0 * len(faces))
    if ratio < 0.95:
        md = o.modifiers.new("dec", "DECIMATE")
        md.ratio = ratio
        apply_modifiers(o)
    smooth(o, 60)
    uv_planar(o, u_axis=(1, 0, 0), v_axis=(0, 0, 1))
    return o


def _field_mesh(name, outline, depth_fn, cell, grow, hide=None, margin=0.0):
    """Regular grid clipped to a closed outline: border nodes are snapped onto the outline (pushed `grow`
    outwards, under the rim). The depth is analytic in (x, z) so the swell shades as one continuous surface.
    hide: outline of something that covers the sheet - cells wholly inside it (by margin) are left out."""
    xs = [q[0] for q in outline]
    zs = [q[1] for q in outline]
    x0, x1, z0, z1 = min(xs) - cell, max(xs) + cell, min(zs) - cell, max(zs) + cell
    nx, nz = int(round((x1 - x0) / cell)), int(round((z1 - z0) / cell))
    X, Z = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(z0, z1, nz + 1))
    sd = _poly_sd(X, Z, outline)
    ins = sd > 0.0
    best = np.full(X.shape, 1e9)
    QX, QZ = X.copy(), Z.copy()
    n = len(outline)
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        d, t = _seg_dist(X, Z, a, b)
        hit = d < best
        best = np.where(hit, d, best)
        QX = np.where(hit, a[0] + t * (b[0] - a[0]), QX)
        QZ = np.where(hit, a[1] + t * (b[1] - a[1]), QZ)
    k = grow / np.maximum(best, 1e-6)
    PX = np.where(ins, X, QX + (X - QX) * k)
    PZ = np.where(ins, Z, QZ + (Z - QZ) * k)
    keep = ins[:-1, :-1] | ins[1:, :-1] | ins[:-1, 1:] | ins[1:, 1:]
    if hide is not None:
        hd = _poly_sd(X, Z, hide) > margin
        keep &= ~(hd[:-1, :-1] & hd[1:, :-1] & hd[:-1, 1:] & hd[1:, 1:])
    used = np.zeros(X.shape, dtype=bool)
    used[:-1, :-1] |= keep
    used[1:, :-1] |= keep
    used[:-1, 1:] |= keep
    used[1:, 1:] |= keep
    idx = -np.ones(X.shape, dtype=np.int64)
    idx[used] = np.arange(int(used.sum()))
    verts = np.stack([PX[used], -depth_fn(PX[used], PZ[used]), PZ[used]], axis=1)
    jj, ii = np.nonzero(keep)
    faces = np.stack([idx[jj, ii], idx[jj, ii + 1], idx[jj + 1, ii + 1], idx[jj + 1, ii]], axis=1)
    o = mesh_object(name, verts.tolist(), faces.tolist())
    smooth(o, 80)
    return o


def _inset(loop, dist):
    """Offset every point of a closed CCW loop inwards by dist[i] (m)."""
    n = len(loop)
    out = []
    for i in range(n):
        (xa, za), (xb, zb) = loop[i - 1], loop[(i + 1) % n]
        ln = hypot(xb - xa, zb - za) or 1.0
        out.append((loop[i][0] - (zb - za) / ln * dist[i], loop[i][1] + (xb - xa) / ln * dist[i]))
    return out


def _slab(name, loop, d0, d1, cham, flat_in, chips=(), lip=0.0):
    """Rim of a shaped board layer from depth d0 to d1: wall, chamfered arris and a flat ring (its inner
    edge sunk by lip, to pass under the face sheet). chips: [(index, half_span, depth_m, k0, k1)] knock
    irregular pieces off the arris. Returns (object, loop halfway across the flat ring)."""
    n = len(loop)
    ex_in, ex_back = [0.0] * n, [0.0] * n
    for (c, hs, a, k0, k1) in chips:
        for j in range(-hs, hs + 1):
            f = 1.0 - (abs(j) / (hs + 1.0)) ** 2
            i = (c + j) % n
            ex_in[i] = max(ex_in[i], a * f * (k0 if j % 2 else 1.0))
            ex_back[i] = max(ex_back[i], a * f * (k1 if j % 2 else 0.75))
    r2 = _inset(loop, [cham + e for e in ex_in])
    r3 = _inset(loop, [cham + flat_in + e for e in ex_in])
    verts = [(x, -d0, z) for x, z in loop]
    verts += [(x, -(d1 - cham - ex_back[i]), z) for i, (x, z) in enumerate(loop)]
    verts += [(x, -d1, z) for x, z in r2]
    verts += [(x, -(d1 - lip), z) for x, z in r3]
    faces = []
    for r in range(3):
        a0, b0 = r * n, (r + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a0 + i, a0 + j, b0 + j, b0 + i))
    o = mesh_object(name, verts, faces)
    smooth(o, 50)
    return o, _inset(loop, [cham + 0.5 * flat_in + e for e in ex_in])


def _volute_pts(c, r0, k, turns, a0, ccw, seg=24):
    """Log spiral from the entry point (angle a0, radius r0) into the eye: [(x, z, r / r0, t)]."""
    n = max(10, int(round(turns * seg)))
    d = 1.0 if ccw else -1.0
    out = []
    for i in range(n + 1):
        t = i / n
        ang = radians(a0) + d * turns * 2 * pi * t
        rr = k ** (turns * t)
        out.append((c[0] + r0 * rr * cos(ang), c[1] + r0 * rr * sin(ang), rr, t))
    return out


STRAP_PROF = [(-1.0, 0.36), (-0.82, 0.88), (-0.62, 1.0), (-0.34, 0.78), (0.0, 0.7), (0.34, 0.78), (0.62, 1.0), (0.82, 0.88), (1.0, 0.36)]
STRAP_PROF_S = [(-1.0, 0.36), (-0.74, 0.96), (-0.36, 0.76), (0.0, 0.7), (0.36, 0.76), (0.74, 0.96), (1.0, 0.36)]


def _strap(name, st, y_back, sx=1.0, prof=STRAP_PROF):
    """Moulded strap (raised fillets either side of a hollow, quirked edges) along stations
    [(x, z, width, top_depth, moulding_height)]; the side walls run back to y_back so the scroll is solid
    where it leaves the board."""
    n, m = len(st), len(prof) + 2
    verts = []
    for i, (x, z, w, top, hh) in enumerate(st):
        xa, za = st[max(i - 1, 0)][:2]
        xb, zb = st[min(i + 1, n - 1)][:2]
        tx, tz = (xb - xa) * sx, zb - za
        ln = hypot(tx, tz) or 1.0
        nx_, nz_ = tz / ln, -tx / ln                         # side = tangent x up(-Y)
        verts.append((x * sx - nx_ * w / 2, -y_back, z - nz_ * w / 2))
        for s, f in prof:
            verts.append((x * sx + nx_ * s * w / 2, -(top - hh * (1.0 - f)), z + nz_ * s * w / 2))
        verts.append((x * sx + nx_ * w / 2, -y_back, z + nz_ * w / 2))
    faces = [(i * m + k, i * m + k + 1, (i + 1) * m + k + 1, (i + 1) * m + k) for i in range(n - 1) for k in range(m - 1)]
    o = mesh_object(name, verts, faces)
    smooth(o, 50)
    return o


def _leaf(name, base, length, width, ang, y0, hgt, cut=1.0):
    """Carved palm leaf: lens outline with a raised midrib, lying on the board (cut < 1: tip snapped off)."""
    ca, sa = cos(radians(ang)), sin(radians(ang))
    n = 7
    half = [(min(length * i / n, length * cut), 0.5 * width * sin(pi * (i / n) ** 0.8)) for i in range(n + 1)]
    lens = [(x, -w) for x, w in half] + [(x, w) for x, w in reversed(half[1:-1])]
    outline = [(base[0] + x * ca - w * sa, base[1] + x * sa + w * ca) for x, w in lens]
    cx, cz = base[0] + 0.45 * length * min(cut, 1.0) * ca, base[1] + 0.45 * length * min(cut, 1.0) * sa
    m = len(outline)
    verts = [(cx, -(y0 + hgt), cz)]
    for f in (0.5, 1.0):
        for (x, z) in outline:
            verts.append((cx + (x - cx) * f, -(y0 + hgt * (1 - f ** 1.5)), cz + (z - cz) * f))
    faces = []
    for i in range(m):
        j = (i + 1) % m
        faces.append((0, 1 + i, 1 + j))
        faces.append((1 + i, 1 + m + i, 1 + m + j, 1 + j))
    o = mesh_object(name, verts, faces)
    smooth(o, 60)
    return o


# ------------------------------------------------------------------------------------------- charges (height fields)
def _crown_h(cw, dep):
    """Crown: curved circlet with torus mouldings and bosses, five arched points with midribs (the tall ones
    trefoiled), pearls half sunk on the tips. Heights follow a cylinder so the crown reads round."""
    ups = [((0.0, -0.17), (0.0, 0.38), 0.1, True), ((0.2, -0.17), (0.245, 0.1), 0.085, False),
           ((0.385, -0.17), (0.53, 0.24), 0.085, True)]
    bow = 0.035

    def fn(X, Z):
        u = np.abs(X / cw)
        v = Z / cw + bow * np.clip(1.0 - (u / 0.5) ** 2, 0.0, None)
        cyl = 0.6 + 0.4 * np.sqrt(np.clip(1.0 - (u / 0.64) ** 2, 0.0, 1.0))
        H = np.zeros(u.shape)
        for (b, tp, w0, fleuron) in ups:
            d, t = _seg_dist(u, v, b, tp)
            if fleuron:                                      # stem, waisted neck, pointed leaf with two curled side lobes
                s = np.clip((t - 0.42) / 0.58, 0.0, 1.0)
                w = np.where(t < 0.42, w0 * (1.0 - 0.5 * t / 0.42), 0.62 * w0 * np.sin(pi * s ** 0.7) ** 0.8 + 0.5 * w0 * (1.0 - s)) + 0.004
            else:
                w = w0 * (1.0 - t) ** 0.6 + 0.004
            e = w - d
            plate = 0.42 * np.clip(e / 0.032, 0.0, 1.0) * (1.0 - 0.25 * t)
            rib = 0.22 * np.clip(1.0 - d / (0.32 * w + 0.008), 0.0, 1.0) * (e > 0) * (1.0 - 0.3 * t)
            H = np.maximum(H, plate + rib)
            if fleuron:
                ax, az = tp[0] - b[0], tp[1] - b[1]
                ln = hypot(ax, az)
                px, pz = az / ln, -ax / ln
                n0 = (b[0] + 0.36 * ax, b[1] + 0.36 * az)
                for sg in (-1.0, 1.0):
                    tip = (b[0] + 0.6 * ax + sg * 1.3 * w0 * px, b[1] + 0.6 * az + sg * 1.3 * w0 * pz)
                    dl, tl = _seg_dist(u, v, n0, tip)
                    wl = w0 * (0.3 + 0.22 * np.sin(pi * tl ** 0.8))
                    H = np.maximum(H, (0.36 - 0.06 * tl) * np.clip((wl - dl) / 0.028, 0.0, 1.0) + 0.14 * np.clip(1.0 - dl / (0.4 * wl), 0.0, 1.0))
            else:
                dp = np.hypot(u - tp[0], v - tp[1] - 0.014)
                H = np.maximum(H, _bump((dp / 0.05) ** 2, 0.3, 0.52))
        e = np.minimum(np.minimum(0.44 - u, v + 0.375), -0.165 - v)
        band = 0.56 * np.clip(e / 0.02, 0.0, 1.0)
        for vt in (-0.197, -0.343):
            band = band + 0.27 * np.sqrt(np.clip(1.0 - ((v - vt) / 0.03) ** 2, 0.0, 1.0))
        for uc in (0.105, 0.315):
            band = band + 0.2 * np.clip(1.0 - (np.abs(u - uc) + np.abs(v + 0.27)) / 0.045, 0.0, 1.0)
        H = np.maximum(H, np.where(e > 0, band, 0.0))
        return dep * H * cyl

    jewels = [(uc * cw, (-0.27 - bow * (1 - (uc / 0.5) ** 2)) * cw, 0.038 * cw, 1.25 if uc else 0.9) for uc in (-0.21, 0.0, 0.21)]
    return fn, (-0.6 * cw, 0.6 * cw, -0.44 * cw, 0.47 * cw), jewels


def _key_h(kh, dep, ang):
    """Key laid bendwise: trefoil bow ring, half-round shank with turned collars, warded bit."""
    ca, sa = cos(radians(ang)), sin(radians(ang))
    bit = [(0.02, 0.19), (0.25, 0.19), (0.25, 0.26), (0.16, 0.26), (0.16, 0.30), (0.25, 0.30), (0.25, 0.45), (0.17, 0.45), (0.17, 0.37),
           (0.105, 0.37), (0.105, 0.45), (0.02, 0.45)]

    def fn(X, Z):
        A = (X * ca + Z * sa) / kh
        B = (-X * sa + Z * ca) / kh
        rho = np.hypot(A, B + 0.33)
        th = np.arctan2(B + 0.33, A)
        ro = 0.15 * (1 + 0.2 * np.cos(3 * (th - pi / 2)))
        rm, hw = (ro + 0.066) / 2, (ro - 0.066) / 2
        H = 0.72 * np.sqrt(np.clip(1.0 - ((rho - rm) / hw) ** 2, 0.0, 1.0))
        d, _ = _seg_dist(A, B, (0.0, -0.2), (0.0, 0.5))
        H = np.maximum(H, 0.86 * np.sqrt(np.clip(1.0 - (d / 0.034) ** 2, 0.0, 1.0)))
        for bc, rc in ((-0.172, 0.056), (-0.127, 0.045), (0.13, 0.047), (0.5, 0.043)):
            along = np.clip((0.018 - np.abs(B - bc)) / 0.008, 0.0, 1.0)
            H = np.maximum(H, (0.6 + 0.4 * along) * np.sqrt(np.clip(1.0 - (A / rc) ** 2, 0.0, 1.0)) * (along > 0))
        H = np.maximum(H, 0.56 * np.clip(_poly_sd(A, B, bit) / 0.014, 0.0, 1.0))
        H = np.maximum(H, _bump((A / 0.042) ** 2 + ((B - 0.53) / 0.042) ** 2, 0.0, 0.9))
        return dep * H

    return fn, (-0.47 * kh, 0.47 * kh, -0.57 * kh, 0.57 * kh), []


def _lion_h(R, dep, rng):
    """Lion mask: scowling brow ridge over sunk eyes, broad bridge and nose pad, whisker pads split by the
    philtrum, chin, round ears, set in two layers of mane locks that sweep down the flanks."""
    locks = []

    def ring(n, r0, r1, w0, h0, h1, off):
        for i in range(n):
            a = 360.0 * (i + off) / n + rng.u(-5, 5)
            ar = radians(a)
            ln = r1 * (1.0 + 0.1 * max(0.0, -sin(ar)) + rng.u(-0.07, 0.05))
            curl = -30.0 * cos(ar) + rng.u(-8, 8)            # tips sweep downwards on both flanks
            pts = []
            for j in range(5):
                t = j / 4.0
                rr = r0 + (ln - r0) * t
                aa = radians(a + curl * t * t)
                pts.append((rr * cos(aa), rr * sin(aa)))
            locks.append((pts, w0 * rng.u(0.9, 1.12), h0, h1))

    ring(13, 0.5, 0.95, 0.16, 0.44, 0.2, 0.25)
    ring(10, 0.4, 0.74, 0.15, 0.64, 0.42, 0.0)
    fs = 1.12

    def fn(X, Z):
        U, V = X / R, Z / R
        H = np.zeros(U.shape)
        for pts, w0, h0, h1 in locks:
            for j in range(4):
                d, t = _seg_dist(U, V, pts[j], pts[j + 1])
                tt = (j + t) / 4.0
                w = w0 * (1.0 - tt) ** 0.65 + 0.012
                H = np.maximum(H, (h0 + (h1 - h0) * tt) * np.clip(1.0 - (d / w) ** 1.5, 0.0, 1.0))
        u, v = U / fs, (V - 0.04) / fs
        au = np.abs(u)

        def ell(cu_, cv_, ru_, rv_):
            return ((au - cu_) / ru_) ** 2 + ((v - cv_) / rv_) ** 2

        F = _smax(0.6 * np.clip(1.0 - ell(0, 0.1, 0.5, 0.44), 0.0, 1.0) ** 0.45,
                  0.62 * np.clip(1.0 - ell(0, -0.26, 0.37, 0.41), 0.0, 1.0) ** 0.45, 0.08)
        ear = 0.58 * np.clip(1.0 - ell(0.43, 0.44, 0.14, 0.14), 0.0, 1.0) ** 0.5 - 0.24 * np.exp(-ell(0.41, 0.42, 0.07, 0.07))
        F = np.maximum(F, ear)
        d, t = _seg_dist(au, v, (0.05, 0.135), (0.38, 0.275))
        F = _smax(F, (0.9 - 0.2 * t) * np.clip(1.0 - (d / 0.1) ** 2, 0.0, 1.0) ** 0.6, 0.05)
        F = F - 0.2 * np.exp(-ell(0.215, 0.07, 0.11, 0.06))                          # eye sockets under the brow
        F = np.maximum(F, _bump(ell(0.215, 0.07, 0.085, 0.047), 0.45, 0.15))           # almond eyeballs
        d, t = _seg_dist(au, v, (0.0, 0.17), (0.0, -0.2))
        F = _smax(F, (0.78 + 0.2 * t) * np.clip(1.0 - (d / (0.078 + 0.04 * t)) ** 3, 0.0, 1.0) ** 0.5, 0.04)
        F = np.maximum(F, _bump(ell(0.13, -0.4, 0.185, 0.135), 0.0, 0.88))            # whisker pads
        F = np.maximum(F, _bump(ell(0, -0.6, 0.145, 0.08), 0.0, 0.62))                # chin
        F = np.maximum(F, _bump(ell(0, -0.24, 0.16, 0.092), 0.0, 1.0, 0.4))           # nose pad
        F = F - 0.08 * np.exp(-ell(0.09, -0.268, 0.03, 0.025))                       # nostrils
        F = F - 0.13 * np.exp(-(u / 0.02) ** 2) * np.clip((v + 0.54) / 0.03, 0, 1) * np.clip((-0.31 - v) / 0.03, 0, 1)
        F = F - 0.17 * np.exp(-((v - (-0.528 + 0.1 * au)) / 0.022) ** 2) * np.clip((0.28 - au) / 0.05, 0, 1)
        F = F - 0.05 * np.exp(-(u / 0.028) ** 2) * np.clip((v - 0.2) / 0.05, 0, 1) * np.clip((0.46 - v) / 0.05, 0, 1)
        return dep * np.maximum(H, np.where(F > 0.06, F, 0.0))     # (the smooth max lifts empty ground a little)

    return fn, (-1.12 * R, 1.12 * R, -1.15 * R, 1.1 * R), []


def _star(R, dep):
    """Eight-pointed faceted star (explicit facets stay crisp) with a flared root and a centre boss."""
    n = 16
    ring = []
    for i in range(n):
        t = radians(90 + i * 360.0 / n)
        r = R * (0.25 if i % 2 else (1.0 if (i // 2) % 2 == 0 else 0.62))
        ring.append((r * cos(t), r * sin(t)))
    verts = [(0.0, -dep, 0.0)] + [(x, -dep * 0.2, z) for x, z in ring]
    verts += [(x * 1.03 + 0.012 * R * np.sign(x), -dep * 0.05, z * 1.03 + 0.012 * R * np.sign(z)) for x, z in ring]
    verts += [(x * 1.08 + 0.03 * R * np.sign(x), dep * 0.1, z * 1.08 + 0.03 * R * np.sign(z)) for x, z in ring]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((0, 1 + i, 1 + j))
        faces.append((1 + i, 1 + n + i, 1 + n + j, 1 + j))
        faces.append((1 + n + i, 1 + 2 * n + i, 1 + 2 * n + j, 1 + n + j))
    star = mesh_object("star", verts, faces)
    recalc_normals(star)
    flat(star)
    boss = sphere("star_boss", R * 0.085, loc=(0, -dep * 0.9, 0), segments=10, rings=6, scale=(1, 0.7, 1))
    pellets = [sphere("star_pellet", R * 0.06, loc=(R * 0.74 * cos(radians(22.5 + k * 45)), -dep * 0.02, R * 0.74 * sin(radians(22.5 + k * 45))),
                      segments=8, rings=5) for k in range(8)]
    return join([star, boss] + pellets, "star")


# ------------------------------------------------------------------------------------------- build
def build(p, rng):
    S = p.size / 0.925                                       # design unit: shell top to ribbon bottom = 0.925
    wr, seed, m, gl = p.wear, int(p.seed), p.material, p.gilding
    W, E = p.width, p.ears

    # ---- materials
    gold = mat("brass", "#ffe6a8", rough=1.1)
    if m == "stone":
        body = mat("mortar", p.stone, grime=0.5)                      # fine sandy limestone, no direction
        ground = mat("stone_rough", p.stone, scale=2.5, grime=0.5)    # punched ground of the back plate
        field = mat("plaster_lime", p.field) if p.painted else body
        dark = mat("plaster_lime", "#2a231d")
        charge_plain, ribbon_m, bole = body, body, body
        stud_m = mat("lead")
    elif m == "wood":
        body = mat("wood_painted", p.paint)
        ground = body
        field = mat("wood_painted", p.field) if p.painted else body
        dark = mat("wood_painted", "#1d1915")
        charge_plain = mat("wood_painted", "#d8ccac")
        ribbon_m = field if (gl != "none" and p.painted) else charge_plain
        bole = mat("wood_painted", "#7d3a22")                          # red bole under the leaf
        stud_m = mat("brass", rough=0.9)
    else:
        body = mat("brass", "#c4a57a", rough=1.15)                     # rubbed statuary bronze
        ground = mat("bronze_aged")
        field = mat("ceramic_glazed", p.field) if p.painted else body
        dark = body
        charge_plain, ribbon_m, bole = body, body, body
        stud_m = mat("brass", rough=0.9)
    charge_m = gold if gl != "none" else charge_plain
    letter_m = mat("iron_painted", "#d9a93c") if gl != "none" else dark
    trim_m = gold if gl == "rich" else body

    body_objs, field_objs, gold_objs, trim_objs, rib_objs, reliefs = [], [], [], [], [], []

    # ---- layout of the surround: spine of the C-scroll each side, volute ears hanging off it
    rs = rng.fork("scrolls")
    t_back, t_front = 0.016 * S, 0.022 * S
    yb = t_back + t_front                                    # front face of the board
    K = 0.42                                                 # radius ratio per turn of the volutes
    cwid = 1.04 * 2 * (1 - K) / (1 + K)                      # strap width / radius so successive turns just touch
    hs = 0.024 * S                                           # moulding height of the strap
    side = {}
    for sx in (1, -1):
        ru, rl = 0.06 * E * rs.vary(1.0, 0.09) * S, 0.051 * E * rs.vary(1.0, 0.09) * S
        xu, xl = 0.313 * W * S, 0.283 * W * S
        side[sx] = dict(ru=ru, rl=rl, tu=rs.vary(1.6, 0.07), tl=rs.vary(1.5, 0.07), cu=(xu + ru, 0.255 * S), cl=(xl + rl, -0.14 * S),
                        xu=xu, xl=xl)
    crest_top = {"shell": [(0.25, 0.345), (0.185, 0.362), (0.159, 0.402), (0.122, 0.46), (0.064, 0.497), (0.0, 0.51)],
                 "coronet": [(0.25, 0.345), (0.188, 0.358), (0.16, 0.392), (0.15, 0.45), (0.118, 0.505), (0.06, 0.532), (0.0, 0.54)],
                 "none": [(0.25, 0.345), (0.17, 0.372), (0.085, 0.398), (0.0, 0.405)]}[p.crest]

    def half_outline(q):
        cu, cl, ru, rl = q["cu"], q["cl"], q["ru"], q["rl"]
        xb = cl[0] - 0.8 * rl
        pts = [(0.0, -0.345 * S), (0.1 * S, -0.338 * S), (0.19 * S, -0.322 * S), (xb - 0.04 * S, -0.287 * S), (xb - 0.008 * S, -0.24 * S),
               (xb, cl[1] - 0.9 * rl), (cl[0] - 0.45 * rl, cl[1] - 0.35 * rl), (cl[0] - 0.4 * rl, cl[1] + 0.6 * rl),
               ((q["xu"] + q["xl"]) / 2 + 0.036 * S, (cu[1] + cl[1]) / 2),
               (cu[0] - 0.4 * ru, cu[1] - 0.6 * ru), (cu[0] - 0.45 * ru, cu[1] + 0.35 * ru), (cu[0] - 0.85 * ru, cu[1] + 0.95 * ru)]
        return pts + [(x * S, z * S) for x, z in crest_top]

    hr, hl = half_outline(side[1]), half_outline(side[-1])
    loop = hr + [(-x, z) for x, z in reversed(hl[1:-1])]
    NL = 96
    loop = [(q[0], q[1]) for q in resample_path([(x, z, 0.0) for x, z in loop], n=NL, closed=True, smooth_path=True)]

    zs = 0.04 * S
    sw, sh = (0.39 * S, 0.50 * S) if p.shield == "oval" else (0.40 * S, 0.50 * S)
    so = [(x, z + zs) for x, z in _shield_outline(p.shield, sw, sh)]

    # ---- back board: wide ground plate + shaped front layer with a chamfered, chipped arris
    rc = rng.fork("chips")
    chips = []
    n_chip = 0 if m == "bronze" else int(round((2 + 7 * wr) * (1.0 if m == "stone" else 0.5)))
    for _ in range(n_chip):
        chips.append((int(rc.u(0, NL)), rc.pick((0, 1, 1, 2)), rc.u(0.008, 0.024) * S * (0.5 + 0.7 * wr), rc.u(0.4, 0.9), rc.u(0.3, 1.1)))
    front, mid = _slab("board_front", loop, t_back - 0.003 * S, yb, 0.006 * S, 0.014 * S, chips=chips, lip=0.0012 * S)
    face = _field_mesh("board_face", mid, lambda X, Z: np.full(np.shape(X), yb), 0.04 * S, 0.0, hide=so, margin=0.004 * S)
    back_loop = _inset(loop, [-0.015 * S] * NL)
    chips_b = [(int(rc.u(0, NL)), rc.pick((0, 1)), rc.u(0.004, 0.011) * S * (0.5 + 0.7 * wr), rc.u(0.4, 0.9), rc.u(0.3, 1.0))
               for _ in range(n_chip // 2)]
    back, _ = _slab("board_back", back_loop, 0.0005 * S, t_back, 0.004 * S, 0.024 * S, chips=chips_b)
    body_objs += [front, face]
    assign(back, ground)

    # ---- strapwork: a moulded C-scroll down each flank rolling forward into volute ears, small scrolls off the cresting
    eye_list = []
    straps = []
    x_lobe = {"shell": 0.165, "coronet": 0.16, "none": 0.075}[p.crest]
    z_lobe = {"shell": 0.372, "coronet": 0.37, "none": 0.393}[p.crest]
    for sx in (1, -1):
        q = side[sx]
        cu, cl, ru, rl = q["cu"], q["cl"], q["ru"], q["rl"]
        up = _volute_pts(cu, ru, K, q["tu"], 180, False)
        lo = _volute_pts(cl, rl, K, q["tl"], 180, True)
        st = []
        for (x, z, rr, t) in reversed(up):
            hh = hs * max(0.5, rr ** 0.4)
            st.append((x, z, cwid * ru * rr, yb + hh + 0.03 * S * t ** 1.3, hh))
        k = 0.36 * (cu[1] - cl[1])
        bz = bezier_path((q["xu"], 0, cu[1]), (q["xu"], 0, cu[1] - k), (q["xl"], 0, cl[1] + k), (q["xl"], 0, cl[1]), n=8)
        for i, b in enumerate(bz[1:-1]):
            s = (i + 1) / (len(bz) - 1)
            st.append((b[0], b[2], cwid * lerp(ru, rl, s) * (1.0 - 0.1 * sin(pi * s)), yb + hs, hs))
        for (x, z, rr, t) in lo:
            hh = hs * max(0.5, rr ** 0.4)
            st.append((x, z, cwid * rl * rr, yb + hh + 0.026 * S * t ** 1.3, hh))
        cs = _strap("scroll_c", st, 0.001 * S, sx)
        straps.append(cs)
        for (c, r0, tn, top) in ((cu, ru, q["tu"], st[0][3]), (cl, rl, q["tl"], st[-1][3])):
            eye_list.append((sx * c[0], c[1], top - 0.006 * S, r0 * K ** tn * (1 + cwid / 2) * 1.12))
        # small scroll lying on the shoulder beside the cresting
        rt = rs.vary(0.03, 0.1) * S
        ct = (min(max(x_lobe + 0.085, 0.2) * S, q["xu"] - 0.068 * S), 0.352 * S)
        tv = _volute_pts(ct, rt, K, rs.vary(1.25, 0.08), 90, False, seg=16)
        p0, p3 = (x_lobe * S, z_lobe * S), (ct[0], ct[1] + rt)
        kk = 0.4 * (p3[0] - p0[0])
        bz = bezier_path((p0[0], 0, p0[1]), (p0[0] + kk, 0, p0[1] + 0.1 * kk), (p3[0] - kk, 0, p3[1]), (p3[0], 0, p3[1]), n=6)
        st2 = [(b[0], b[2], cwid * rt * (0.8 + 0.2 * i / 5.0), yb + 0.016 * S, 0.016 * S) for i, b in enumerate(bz[:-1])]
        for (x, z, rr, t) in tv:
            hh = 0.016 * S * max(0.55, rr ** 0.4)
            st2.append((x, z, cwid * rt * rr, yb + hh + 0.014 * S * t ** 1.3, hh))
        straps.append(_strap("scroll_top", st2, 0.001 * S, sx, STRAP_PROF_S))
        eye_list.append((sx * ct[0], ct[1], st2[-1][3] - 0.004 * S, rt * K ** 1.25 * (1 + cwid / 2) * 1.15))
    # one ear has taken a knock
    if wr > 0.3:
        sx = rs.pick((1, -1))
        q = side[sx]
        c, r0 = rs.pick(((q["cu"], q["ru"]), (q["cl"], q["rl"])))
        a = radians(rs.u(-70, 70))
        nk = (sx * (c[0] + 0.85 * r0 * cos(a)), c[1] + 0.85 * r0 * sin(a))
        rn = (0.3 + 0.35 * wr) * r0
        cx_, cz_ = sx * c[0], c[1]

        def nick(co):
            d = hypot(co.x - nk[0], co.z - nk[1])
            if d >= rn or co.y > -0.004 * S:
                return co
            f = (1.0 - d / rn) ** 0.7
            return Vector((co.x + (cx_ - co.x) * 0.3 * f, min(-0.004 * S, co.y + 0.022 * S * f), co.z + (cz_ - co.z) * 0.3 * f))

        for o in straps:
            transform_verts(o, nick)
    body_objs += straps
    eye = lathe("eye", [(0.0, 0.0), (1.0, 0.0), (1.0, 0.22), (0.82, 0.42), (0.6, 0.4), (0.42, 0.62), (0.0, 0.72)], segments=8)
    trim_objs.append(instances_to_mesh("eyes", eye, [trs((x, -d, z), (90, 0, rs.u(0, 360)), r) for (x, z, d, r) in eye_list],
                                       keep_base=False))

    # ---- palm sprays fanning out from the drop boss under the shield (one tip may have snapped off)
    snapped = (rs.pick((1, -1)), rs.pick((0, 1))) if wr > 0.4 else None
    for sx in (1, -1):
        for j, (ang, ln) in enumerate(((10, 0.165), (33, 0.17), (58, 0.13))):
            a2 = ang if sx > 0 else 180 - ang
            cut = rs.u(0.5, 0.7) if snapped == (sx, j) else 1.0
            body_objs.append(_leaf("leaf", (sx * 0.028 * S, -0.262 * S + 0.012 * S * j), rs.vary(ln, 0.08) * S, 0.05 * S, a2 + rs.jitter(0, 4),
                                   yb - 0.002 * S, 0.017 * S, cut))

    # ---- shield: domed field (regular grid, analytic swell) inside a bolection rim
    y_edge, dome = yb + 0.012 * S, 0.026 * S
    fa, fb, fzc = 0.54 * sw, 0.54 * sh, zs + 0.01 * S

    def field_d(X, Z):
        return y_edge + dome * np.clip(1.0 - (X / fa) ** 2, 0.0, 1.0) * np.clip(1.0 - ((Z - fzc) / fb) ** 2, 0.0, 1.0)

    fld = _field_mesh("field", so, field_d, 0.023 * S, 0.006 * S)
    field_objs.append(fld)
    rw, rh = 0.03 * S, 0.03 * S
    rim_prof = [(-0.74 * rw, -0.004 * S), (0.74 * rw, -0.004 * S), (0.72 * rw, 0.03 * rh), (0.56 * rw, 0.12 * rh), (0.5 * rw, 0.3 * rh), (0.34 * rw, 0.6 * rh), (0.2 * rw, 0.95 * rh), (0.0, rh),
                (-0.2 * rw, 0.95 * rh), (-0.34 * rw, 0.6 * rh), (-0.5 * rw, 0.3 * rh), (-0.56 * rw, 0.12 * rh), (-0.72 * rw, 0.03 * rh)]
    rim = sweep("rim", [(x, -yb, z) for x, z in so], rim_prof, closed=True, smooth_path=False, up=(0, -1, 0))
    smooth(rim, 50)
    trim_objs.append(rim)

    # ---- charge: relief carved out of the field
    sink = 0.002 * S
    cc = (0.0, zs + 0.03 * S)
    jewel_m = mat("ceramic_glazed", p.field) if p.painted else charge_m
    rq = rng.fork("charge")

    def carve(name, hf, bounds, centre, dep, base_fn, tris):
        hfun = lambda X, Z: hf(X - centre[0], Z - centre[1])
        o = _relief(name, hfun, (bounds[0] + centre[0], bounds[1] + centre[0], bounds[2] + centre[1], bounds[3] + centre[1]),
                    (bounds[1] - bounds[0]) / 104.0, base_fn, sink, tris=tris)
        hp = lambda co: np.clip((-co[:, 1] - base_fn(co[:, 0], co[:, 2]) + sink) / dep, 0.0, 1.2)
        reliefs.append((o, hp))
        return o, (lambda x, z: float(base_fn(np.array([x]), np.array([z]))[0] + hfun(np.array([[x]]), np.array([[z]]))[0, 0] - sink))

    def set_jewels(jewels, centre, depth_at):
        out = []
        for (jx, jz, jr, asp) in jewels:
            x, z = centre[0] + jx, centre[1] + jz
            out.append(sphere("jewel", jr, loc=(x, -depth_at(x, z) + 0.15 * jr, z), segments=8, rings=5, scale=(asp, 0.75, 1.0 / max(asp, 1.0) ** 0.5)))
        if out:
            assign(out, jewel_m)

    if p.charge == "crown":
        hf, bnd, jw = _crown_h(0.27 * S, 0.027 * S)
        ctr = (cc[0], cc[1] - 0.002 * S)
        ch, dat = carve("charge", hf, bnd, ctr, 0.027 * S, field_d, 3000)
        set_jewels(jw, ctr, dat)
    elif p.charge == "key":
        hf, bnd, _ = _key_h(0.37 * S, 0.026 * S, -32)
        ch, _ = carve("charge", hf, bnd, (cc[0] - 0.012 * S, cc[1] - 0.005 * S), 0.026 * S, field_d, 3000)
    elif p.charge == "lion":
        ctr = (cc[0], cc[1] + (0.03 * S if p.shield == "heater" else 0.005 * S))
        hf, bnd, _ = _lion_h((0.132 if p.shield == "heater" else 0.146) * S, 0.034 * S, rq)
        ch, _ = carve("charge", hf, bnd, ctr, 0.034 * S, field_d, 3350)
    else:
        ch = _star(0.175 * S, 0.042 * S)
        ctr = (cc[0], cc[1] + 0.01 * S)
        transform_verts(ch, lambda co: Vector((co.x + ctr[0], co.y - float(field_d(np.array([co.x + ctr[0]]), np.array([co.z + ctr[1]]))[0]),
                                               co.z + ctr[1])))
    assign(ch, charge_m)

    # ---- crest above the shield, seated on the cresting of the board
    if p.crest == "shell":
        hc, Rs, ribs = (0.0, 0.338 * S), 0.148 * S, 9
        y0s = yb + 0.01 * S

        def shell_fn(u, v):
            uu, vc = min(max(u, 0.0), 1.0), min(v, 1.0)
            ang = radians(-82 + 164 * u)
            rib = 0.5 + 0.5 * cos(uu * ribs * 2 * pi)
            vv = 0.1 + 0.9 * vc
            r = Rs * vv * (1 + 0.055 * rib * vc)
            d = y0s + 0.036 * S * sin(pi * (0.12 + 0.62 * vv)) + 0.011 * S * rib * vv - 0.012 * S * vc ** 4
            if v > 1.0 or u < 0.0 or u > 1.0:              # the edge turns back into the board: solid carving, no gap behind
                r, d = r + 0.004 * S * (v > 1.0), yb - 0.003 * S
            return (hc[0] + r * sin(ang), -d, hc[1] + r * cos(ang))

        us = [-0.025] + [i / (ribs * 4) for i in range(ribs * 4 + 1)] + [1.025]
        shell = _grid_mesh("shell", shell_fn, us, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.08])
        smooth(shell, 55)
        trim_objs.append(shell)
        trim_objs.append(lathe("shell_hinge", [(0.0, 0.0), (0.03 * S, 0.0), (0.034 * S, 0.012 * S), (0.026 * S, 0.026 * S), (0.012 * S, 0.034 * S), (0.0, 0.036 * S)],
                               segments=14, loc=(0, -(yb + 0.018 * S), hc[1] + 0.004 * S), rot=(90, 0, 0)))
        for sx in (-1, 1):                                 # little hinge ears
            trim_objs.append(sphere("shell_ear", 0.022 * S, loc=(sx * 0.052 * S, -(yb + 0.026 * S), hc[1] - 0.004 * S), segments=10, rings=6,
                                    scale=(1.5, 0.8, 0.8)))
    elif p.crest == "coronet":
        # moulded bracket shelf + twisted torse carry the coronet, which is carved on the cresting behind it
        z0, L = 0.316 * S, 0.128 * S
        sec = [(0.0, 0.0), (0.0, 0.007), (0.006, 0.016), (0.012, 0.019), (0.015, 0.03), (0.021, 0.031), (0.022, 0.0)]

        def shelf_fn(u, v):
            zz, dd = sec[int(v)]
            x = -L + 2 * L * min(max(u, 0.0), 1.0)
            if u < 0.0 or u > 1.0:
                return (x + (0.012 * S if u > 1 else -0.012 * S) * (zz / 0.022), -(yb - 0.002 * S), z0 + zz * S)
            x *= 1.0 - 0.1 * (1.0 - zz / 0.022)                # the shelf spreads upwards like a bracket
            return (x, -(yb + dd * S), z0 + zz * S)

        shelf = _grid_mesh("crest_shelf", shelf_fn, [-0.1, 0.0, 0.5, 1.0, 1.1], list(range(len(sec))))
        smooth(shelf, 40)
        body_objs.append(shelf)
        lob = [(0.0125 * S * (1 + 0.28 * cos(4 * a)) * cos(a), 0.0125 * S * (1 + 0.28 * cos(4 * a)) * sin(a))
               for a in [2 * pi * i / 12 for i in range(12)]]
        zt = z0 + 0.022 * S + 0.009 * S
        torse = sweep("torse", [(-0.108 * S + 0.216 * S * i / 12, -(yb + 0.017 * S), zt) for i in range(13)], lob, smooth_path=False,
                      twist_deg=760, scale_fn=lambda t: 0.8 + 0.2 * sin(pi * t) ** 0.5)
        smooth(torse, 50)
        trim_objs.append(torse)
        cwc = 0.2 * S
        hf, bnd, jw = _crown_h(cwc, 0.026 * S)
        ctr = (0.0, zt + 0.006 * S + 0.375 * cwc)
        flat_d = lambda X, Z: np.full(np.shape(X), yb)
        co, dat = carve("coronet", hf, bnd, ctr, 0.026 * S, flat_d, 1400)
        assign(co, charge_m)
        gold_objs.append(co)
        set_jewels(jw, ctr, dat)
    else:
        trim_objs.append(lathe("top_boss", [(0.0, 0.0), (0.034 * S, 0.0), (0.037 * S, 0.006 * S), (0.028 * S, 0.014 * S), (0.02 * S, 0.013 * S),
                                            (0.013 * S, 0.022 * S), (0.0, 0.027 * S)], segments=16, loc=(0, -yb, 0.36 * S), rot=(90, 0, 0)))

    # ---- motto ribbon: bent strip with a Z-fold and a swallow tail at each end
    L, Wr, Rr, zc, dF, thk = 0.36 * S, 0.105 * S, 0.62 * S, -0.372 * S, 0.085 * S, 0.007 * S

    def rib_map(s, hh, d):
        a = s / Rr
        rr = Rr - hh
        return (rr * sin(a), -d, zc + Rr - rr * cos(a))

    rr_ = rng.fork("ribbon")
    broken = rr_.pick((-1.0, 1.0)) if wr > 0.8 else 0.0      # very old: one tail has snapped off short
    jag = [rr_.u(-0.012, 0.012) * S for _ in range(5)]

    def half_path(sg):
        tl, dr, fl = rr_.vary(0.125, 0.15) * S, rr_.vary(0.062, 0.2) * S, rr_.u(0.0, 0.012) * S
        if sg == broken:
            tl, dr, fl = 0.07 * S, 0.052 * S, -0.02 * S
        k = [(L * 0.5, 0, dF), (L - 0.03 * S, 0, dF), (L + 0.018 * S, -0.006 * S, dF - 0.013 * S), (L - 0.012 * S, -0.016 * S, dF - 0.026 * S),
             (L - 0.048 * S, -0.026 * S, dF - 0.039 * S), (L - 0.012 * S, -0.036 * S, dF - 0.052 * S), (L + 0.05 * S, -0.05 * S, 0.02 * S),
             (L + tl * 0.62, -dr * 0.8, 0.024 * S), (L + tl, -dr, 0.046 * S + fl)]
        return [(sg * a, b, c) for a, b, c in k]

    key = list(reversed(half_path(-1.0))) + [(0.0, 0.0, dF)] + half_path(1.0)
    nu = 52
    RP = _smooth_path(key, nu + 1)

    def camber(hv):
        return 0.004 * S * cos(pi * hv / Wr)

    def rib_fn(u, v):
        i = int(round(u * nu))
        s, hh, d = RP[i]
        hv = (v - 0.5) * Wr
        if i in (0, nu):
            sg = -1.0 if i == 0 else 1.0
            if sg == broken:
                s -= sg * jag[int(round(v * 3))]                 # ragged fracture
            else:
                s -= sg * 0.04 * S * (1 - abs(2 * v - 1))        # swallow-tail cut
        return rib_map(s, hh + hv, d + camber(hv))

    ribbon = surface("ribbon", rib_fn, nu=nu, nv=3)
    solidify(ribbon, thk, offset=0.0, even=False)
    rib_objs.append(ribbon)
    txt = None
    if p.motto.strip():
        cap = 0.052 * S
        txt = text("motto", p.motto.strip(), size=cap, depth=0.005 * S, rot=(0, 0, 0), font="serif")
        md = txt.modifiers.new("dis", "DECIMATE")             # letters are flat: dissolve the needless outline points
        md.decimate_type = "DISSOLVE"
        md.angle_limit = radians(10)
        apply_modifiers(txt)
        tv_ = [tuple(v.co) for v in txt.data.vertices]
        tf_ = [tuple(q.vertices) for q in txt.data.polygons if max(tv_[i][2] for i in q.vertices) > 0.001 * S]   # hidden backs go
        delete(txt)
        txt = mesh_object("motto", tv_, tf_)
        xs = [v.co.x for v in txt.data.vertices]
        ys = [v.co.y for v in txt.data.vertices]
        if xs:
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            k = min((2 * L - 0.075 * S) / max(x1 - x0, 1e-4), 0.5 * Wr / max(y1 - y0, 1e-4))
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dT = dF + thk / 2 - 0.0006 * S

            def bend(co):
                hh = (co.y - my) * k
                return Vector(rib_map((co.x - mx) * k, hh, dT + camber(hh) + co.z * min(k, 1.6)))

            transform_verts(txt, bend)
            assign(txt, letter_m)

    # ---- fixings and small bosses
    xin = min(side[1]["xl"], side[-1]["xl"]) - 0.5 * cwid * max(side[1]["rl"], side[-1]["rl"])
    xg = 0.5 * (0.5 * sw + 0.62 * rw + xin) + 0.004 * S
    sk = clamp((xin - 0.5 * sw - 0.62 * rw) / (0.05 * S), 0.6, 1.0)
    stud = bolt("stud", 0.012 * S, 0.008 * S, kind="dome")
    washer = lathe("washer", [(0.0, 0.0), (0.019 * S, 0.0), (0.02 * S, 0.003 * S), (0.016 * S, 0.005 * S), (0.0, 0.005 * S)], segments=8)
    rb = rng.fork("studs")
    spos = [(xg, 0.075 * S + rb.jitter(0, 0.006) * S, sk), (-xg, 0.075 * S + rb.jitter(0, 0.006) * S, sk),
            (0.2 * S, -0.262 * S, 0.9), (-0.2 * S, -0.262 * S, 0.9)]
    mats_ = [trs((x, -yb, z), (90, 0, rb.u(0, 90)), k) for x, z, k in spos]
    studs = instances_to_mesh("studs", stud, [M @ Matrix.Translation((0, 0, 0.004 * S)) for M in mats_], keep_base=False)
    washers = instances_to_mesh("washers", washer, mats_, keep_base=False)
    assign(studs, stud_m)
    assign(washers, stud_m)
    drop = lathe("drop", [(0.0, 0.0), (0.03 * S, 0.0), (0.033 * S, 0.005 * S), (0.025 * S, 0.012 * S), (0.018 * S, 0.011 * S), (0.012 * S, 0.02 * S),
                          (0.0, 0.025 * S)], segments=16, loc=(0, -yb, -0.262 * S), rot=(90, 0, 0))
    trim_objs.append(drop)

    # ---- materials
    assign(body_objs, body)
    assign(field_objs, field)
    assign(trim_objs, trim_m)
    assign(rib_objs, ribbon_m)
    if m == "wood":
        for o in [front, face, back, fld, ribbon] + straps:
            uv_grain(o, axis="Z" if o is not ribbon else "X")
    if m != "bronze":
        for o in (front, face):
            displace_noise(o, strength=0.0012 * S, scale=0.1 * S, seed=seed)
        for o in straps:
            displace_noise(o, strength=0.0009 * S, scale=0.05 * S, seed=seed + 5)

    # ---- gilding: leaf lost in sharp-edged flakes over the high points, showing the bole / stone / bronze
    fl_thr = 1.05 - 0.4 * wr

    def flake(o, hp=None):
        if o is None or not o.data.materials or o.data.materials[0] != gold or bole == gold:
            return
        apply_modifiers(o)
        me = o.data
        n = len(me.polygons)
        C = np.empty(n * 3)
        me.polygons.foreach_get("center", C)
        C = C.reshape(-1, 3)
        v = _blotch(C, 60.0 / S, seed * 0.37) + (0.35 * (hp(C) - 0.5) if hp else 0.0)      # high points go first
        sel = v > fl_thr
        if sel.any():
            me.materials.append(bole)
            me.polygons.foreach_set("material_index", np.where(sel, 1, 0).astype(np.int32))

    hp_of = {id(o): hp for o, hp in reliefs}
    for o in [ch, txt] + gold_objs + trim_objs:
        if o is not None:
            flake(o, hp_of.get(id(o)))

    # ---- weathering: soot and rain streaks under the ledges, moss / verdigris with age, worn high points
    low = height_ramp(-0.45 * S, 0.1 * S)
    blot = noise_mask(0.16 * S, seed=seed + 1, lo=0.35, hi=0.7)
    fine = noise_mask(0.045 * S, seed=seed + 2, lo=0.4, hi=0.75)
    patch = noise_mask(0.05 * S, seed=seed + 3, lo=0.8 - 0.2 * wr, hi=0.85 - 0.2 * wr)
    upf = facing((0, 0, 1), 1.5)
    dnf = facing((0, 0, -1), 1.0)
    frontf = facing((0, -1, 0), 2.0)
    sidef = lambda co, n: np.clip(1.0 - 1.4 * np.abs(n[:, 1]), 0.0, 1.0)     # walls and undercuts, where verdigris sits
    stn = noise_mask(0.022 * S, seed=seed + 4, lo=0.3, hi=0.8)
    ledges = []
    for sx in (1, -1):
        q = side[sx]
        for c, r0, ln in ((q["cu"], q["ru"], 0.2), (q["cl"], q["rl"], 0.13)):
            x0, x1 = sorted((sx * (c[0] - 1.3 * r0), sx * (c[0] + 0.9 * r0)))
            ledges.append((x0, x1, c[1] - 0.6 * r0, ln * S))
    ledges.append((-0.15 * S, 0.15 * S, zs + 0.5 * sh + 0.02 * S, 0.16 * S))
    ledges.append((-0.12 * S, 0.12 * S, zs - 0.42 * sh, 0.1 * S))

    def streak(co, n):
        sq = co.copy()
        sq[:, 2] *= 0.07                                       # stretch the noise vertically into runs
        st = stn(sq, n)
        mk = np.zeros(len(co))
        for (x0, x1, zt, ln) in ledges:
            inx = np.clip(np.minimum(co[:, 0] - x0, x1 - co[:, 0]) / (0.025 * S) + 0.3, 0.0, 1.0)
            dz = (zt - co[:, 2]) / ln
            mk = np.maximum(mk, inx * np.where((dz > -0.08) & (dz < 1.0), (1.0 - np.clip(dz, 0.0, 1.0)) ** 1.4, 0.0))
        return mk * (0.35 + 0.65 * st)

    def relief_masks(o, hp):
        hi = lambda co, n: np.clip((hp(co) - 0.5) * 2.2, 0.0, 1.0)
        lo_ = lambda co, n: np.clip(0.85 - 2.0 * hp(co), 0.0, 1.0)
        return hi, lo_

    for o in body_objs + [back] + trim_objs + rib_objs + gold_objs + [ch, txt]:
        if o is None:
            continue
        apply_modifiers(o)
        is_gold = bool(o.data.materials) and o.data.materials[0] == gold
        hp = hp_of.get(id(o))
        if hp is not None:
            hi, lo_ = relief_masks(o, hp)
        else:
            hi, lo_ = mask_mul(frontf, fine), mask_mul(dnf, 0.8)
        if o is txt:
            paint(o, grime=mask_mul(fine, 0.3 * wr))
        elif is_gold:
            paint(o, grime=mask_mul(lo_, 0.45 + 0.4 * wr), wear=mask_mul(hi, 0.9))
            paint(o, grime=mask_mul(fine, 0.1 + 0.25 * wr), wear=0.55)
        elif m == "stone":
            paint(o, grime=mask_mul(streak, 0.45 + 0.2 * wr), special=mask_mul(upf, fine, 1.1 * wr), wear=mask_mul(fine, 0.3 * wr))
            paint(o, grime=mask_mul(dnf, 0.45))
            paint(o, grime=mask_mul(low, blot, 0.2 + 0.2 * wr))
        elif m == "wood":
            paint(o, grime=mask_mul(streak, 0.45 + 0.35 * wr), wear=mask_mul(patch, 0.5 + 0.5 * wr), special=mask_mul(blot, 0.35 * wr))
            paint(o, grime=mask_mul(low, blot, 0.25 + 0.35 * wr))
        else:
            paint(o, grime=mask_mul(lo_, 0.4), wear=mask_mul(hi if hp is not None else frontf, fine, 0.85),
                  special=mask_mul(streak, 0.35 + 0.3 * wr))
            paint(o, special=mask_mul(sidef, blot, 0.3 + 0.3 * wr), wear=mask_mul(frontf, blot, 0.75))
            paint(o, wear=0.4)
    pn = noise_mask(0.075 * S, seed=seed + 6, lo=0.0, hi=1.0)
    f_thr = 0.52 - 0.17 * wr

    def field_flake(co, n):                                    # paint lets go from the edge inwards, where the rim holds the wet
        edge = np.clip(1.0 - _poly_sd(co[:, 0], co[:, 2], so) / (0.09 * S), 0.0, 1.0)
        return np.clip((pn(co, n) * (0.5 + 0.55 * edge) - f_thr) / 0.22, 0.0, 1.0)

    for o in field_objs:
        k = {"bronze": 0.0, "wood": 0.8}.get(m, 1.0)
        paint(o, grime=mask_mul(streak, 0.45 + 0.35 * wr), wear=mask_mul(field_flake, k), special=mask_mul(blot, 0.3 * wr))
        paint(o, grime=mask_mul(blot, 0.2 + 0.3 * wr))
    paint(studs, special=0.1 + 0.2 * wr, wear=0.5)
    paint(washers, special=0.15 + 0.25 * wr, grime=0.3)
