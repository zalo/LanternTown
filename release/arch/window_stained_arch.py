"""Arched stained-glass window of the clock-tower hall. A dressed-stone surround (jamb courses of uneven
height with long quoins toothed into the wall, voussoirs with V-joints and the odd spalled arris,
cavetto-and-bead reveal with nook shafts) under a roll-and-hollow hood mould that dies onto carved
rosette label stops, standing on a weathered sill with a drip and an apron. Inside it, two-order stone
bar tracery (single cusped light / twin lancets with a foiled oculus / lancets under a wheel rose with
cusped petals round an emblem roundel, trefoil-pierced spandrels) holds leaded glazing: every light is
a came panel with a coloured border, a field of diamond quarries and an emblem medallion (star / crown /
heart) between iron saddle bars. Each quarry sits a hair out of plane and carries its own patina, the
glass darkens towards the cames, the foot and the corners, and the hall behind it rakes the reveals,
mullions and sill with coloured light.
"""
from math import acos, asin, atan2, ceil, cos, hypot, log, pi, radians, sin, sqrt

import bmesh
import numpy as np

from tk import *

META = {"name": "Arched stained-glass window", "class": "component", "mount": "wall",
        "mount_height": 1.3, "focus": 0.62, "tri_budget": 16000}
PARAMS = {
    "width": {"default": 1.3, "min": 0.7, "max": 2.4, "desc": "outer width over sill and hood stops (m)"},
    "height": {"default": 2.8, "min": 1.6, "max": 5.0, "desc": "outer height, sill bed to hood apex (m)"},
    "arch": {"default": "pointed", "choices": ["pointed", "round"], "desc": "arch form"},
    "tracery": {"default": "twin", "choices": ["single", "twin", "rose"],
                "desc": "single cusped light / two lancets + oculus / lancets under a wheel rose"},
    "palette": {"default": "jewel", "choices": ["jewel", "royal", "ember", "garden", "rosy"], "desc": "glass colour scheme"},
    "emblem": {"default": "star", "choices": ["star", "crown", "heart"], "desc": "emblem leaded into the medallions"},
    "point": {"default": 1.0, "min": 0.6, "max": 1.5, "desc": "pointed arch sharpness (arc radius / span; 1 = equilateral)"},
    "relief": {"default": 0.17, "min": 0.12, "max": 0.26, "desc": "projection of the stone surround from the wall (m)"},
    "cusps": {"default": True, "desc": "cusped (foiled) heads in the lights and rose petals"},
    "stone": {"default": "#c4b8a2", "desc": "stone tint"},
    "glow": {"default": 1.3, "min": 0.4, "max": 5.0, "desc": "brightness of the lit glass"},
    "intensity": {"default": 6.0, "min": 0.0, "max": 60.0, "desc": "coloured light thrown onto reveals, sill and street (cd)"},
    "moss": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "moss / lichen on sill, hood and lower jambs"},
}
VARIANTS = {
    "round_heart": {"arch": "round", "tracery": "single", "palette": "ember", "emblem": "heart", "width": 1.0,
                    "height": 2.1, "stone": "#b6a58c", "seed": 4},
    "rose_grand": {"tracery": "rose", "palette": "royal", "emblem": "crown", "width": 2.2, "height": 4.4, "seed": 11},
    "lancet": {"tracery": "single", "width": 0.75, "height": 2.9, "point": 1.4, "palette": "garden", "emblem": "star",
               "stone": "#a9a08f", "moss": 0.9, "seed": 23},
    "round_rose": {"arch": "round", "tracery": "rose", "width": 1.8, "height": 2.7, "palette": "rosy", "emblem": "heart",
                   "relief": 0.2, "seed": 31},
}

# glass tints [field 1, field 2, deep accent A, accent B, emblem]; light colours (reveal rim, arch head, sill pool)
PALETTES = {
    "jewel": (["#f0c76a", "#69b58e", "#2a4fc4", "#c22f3a", "#ffe08a"], ("#d9e486", "#7f9cff", "#f0d870")),
    "royal": (["#6c9ce6", "#9a7ada", "#2a30a8", "#cf4a9a", "#ffe08a"], ("#8ea8ff", "#b68cff", "#98a8ff")),
    "ember": (["#f4b860", "#ea8a3c", "#b0242c", "#6c2c80", "#fff0b4"], ("#ff9a48", "#ff6e58", "#ff8a3c")),
    "garden": (["#cfe09a", "#74c0a4", "#1c7a4c", "#e0a030", "#f8eab8"], ("#a6e8ae", "#7adcc0", "#c8ec94")),
    "rosy": (["#ee9cb6", "#b68ce0", "#a02460", "#4264cc", "#ffe6a0"], ("#ff98c2", "#c698ff", "#ff9cbc")),
}


# ----------------------------------------------------------------------------- 2D helpers
def _ring_poly(c, ra, rb, a0, a1, n):
    return ([(c[0] + ra * cos(a0 + (a1 - a0) * i / n), c[1] + ra * sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)] +
            [(c[0] + rb * cos(a1 + (a0 - a1) * i / n), c[1] + rb * sin(a1 + (a0 - a1) * i / n)) for i in range(n + 1)])


def _area(p):
    return 0.5 * sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))


def _ccw(p):
    return list(p) if _area(p) >= 0 else list(reversed(p))


def _circle(c, r, n, a0=0.0):
    return [(c[0] + r * cos(a0 + 2 * pi * k / n), c[1] + r * sin(a0 + 2 * pi * k / n)) for k in range(n)]


def _clip(subj, clip):
    """Sutherland-Hodgman: subj polygon clipped to a convex CCW polygon."""
    out = list(subj)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        ex, ez = b[0] - a[0], b[1] - a[1]
        if abs(ex) + abs(ez) < 1e-12:
            continue
        inp, out = out, []
        if not inp:
            break
        prev = inp[-1]
        sp = ex * (prev[1] - a[1]) - ez * (prev[0] - a[0])
        for cur in inp:
            sc = ex * (cur[1] - a[1]) - ez * (cur[0] - a[0])
            if (sc >= 0) != (sp >= 0):
                t = sp / (sp - sc)
                out.append((prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t))
            if sc >= 0:
                out.append(cur)
            prev, sp = cur, sc
    return out


def _seg_clip(p0, p1, clip):
    """Parameter interval (t0, t1) of segment p0-p1 inside a convex CCW polygon, or None."""
    t0, t1 = 0.0, 1.0
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        ex, ez = b[0] - a[0], b[1] - a[1]
        if abs(ex) + abs(ez) < 1e-12:
            continue
        s0 = ex * (p0[1] - a[1]) - ez * (p0[0] - a[0])
        ds = ex * dz - ez * dx
        if abs(ds) < 1e-12:
            if s0 < 0:
                return None
            continue
        t = -s0 / ds
        if ds > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1 - 1e-9:
            return None
    return t0, t1


def _hull(pts):
    pts = sorted(set((round(x, 6), round(z, 6)) for x, z in pts))

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 1e-12:
                h.pop()
            h.append(q)
        return h

    lo, up = half(pts), half(reversed(pts))
    return lo[:-1] + up[:-1]


def _dp(pts, tol):
    if len(pts) < 3:
        return list(pts)
    a, b = pts[0], pts[-1]
    dx, dz = b[0] - a[0], b[1] - a[1]
    ln = hypot(dx, dz) or 1e-12
    im, dm = 0, 0.0
    for i in range(1, len(pts) - 1):
        d = abs(dx * (pts[i][1] - a[1]) - dz * (pts[i][0] - a[0])) / ln
        if d > dm:
            im, dm = i, d
    if dm <= tol:
        return [a, b]
    return _dp(pts[:im + 1], tol)[:-1] + _dp(pts[im:], tol)


def _simplify(poly, tol=0.0012):
    k = len(poly) // 2
    return _dp(poly[:k + 1], tol)[:-1] + _dp(poly[k:] + [poly[0]], tol)[:-1]


def _star(P, discs, polys, n=240):
    """Radial outline (CCW) of the union of discs (cx, cz, r) and convex polygons as seen from P."""
    pts = []
    for k in range(n):
        th = 2 * pi * k / n
        ux, uz = cos(th), sin(th)
        best = 0.0
        for (cx, cz, r) in discs:
            fx, fz = P[0] - cx, P[1] - cz
            b = fx * ux + fz * uz
            d = b * b - (fx * fx + fz * fz - r * r)
            if d > 0:
                t = -b + sqrt(d)
                if t > best:
                    best = t
        for poly in polys:
            m = len(poly)
            for i in range(m):
                a, q = poly[i], poly[(i + 1) % m]
                ex, ez = q[0] - a[0], q[1] - a[1]
                den = ux * ez - uz * ex
                if abs(den) < 1e-12:
                    continue
                wx, wz = a[0] - P[0], a[1] - P[1]
                t = (wx * ez - wz * ex) / den
                s = (wx * uz - wz * ux) / den
                if t > best and -1e-9 <= s <= 1 + 1e-9:
                    best = t
        pts.append((P[0] + ux * best, P[1] + uz * best))
    return pts


def _pip(P, poly):
    x, z = P[:, 0], P[:, 1]
    ins = np.zeros(len(P), bool)
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        if abs(z1 - z0) < 1e-12:
            continue
        ins ^= ((z0 > z) != (z1 > z)) & (x < (x1 - x0) * (z - z0) / (z1 - z0) + x0)
    return ins


def _dense(poly, step=0.008):
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(1, int(ceil(hypot(b[0] - a[0], b[1] - a[1]) / step)))
        out += [(a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k) for j in range(k)]
    return out


class Arch:
    """Two-centred arched opening: centre x, half span a, bottom z0, springing zs, arc radius R (R == a: round).
    pt() addresses it by side (+1 right, -1 left), zone ('j' jamb: v = z; 'a' arc: v = 0 spring .. 1 apex) and
    an outward offset r, so concentric mouldings mitre exactly on the apex line."""

    def __init__(self, cx, a, z0, zs, R):
        self.cx, self.a, self.z0, self.zs, self.R = cx, a, z0, zs, max(R, a)

    def th(self, r=0.0):
        return acos(clamp((self.R - self.a) / max(self.R + r, 1e-6), -1.0, 1.0))

    def apex(self, r=0.0):
        return self.zs + sqrt(max((self.R + r) ** 2 - (self.R - self.a) ** 2, 0.0))

    def arc_len(self, r=0.0):
        return (self.R + r) * self.th(r)

    def pt(self, side, zone, v, r=0.0):
        if zone == 'j':
            x, z = self.a + r, v
        else:
            th = self.th(r) * v
            x, z = self.a - self.R + (self.R + r) * cos(th), self.zs + (self.R + r) * sin(th)
        return (self.cx + side * x, z)

    def outline(self, r=0.0, zb=None, n=10):
        zb = self.z0 - r if zb is None else zb
        pts = [(self.cx + self.a + r, zb)]
        pts += [self.pt(1, 'a', i / n, r) for i in range(n + 1)]
        pts += [self.pt(-1, 'a', i / n, r) for i in range(n - 1, -1, -1)]
        pts.append((self.cx - self.a - r, zb))
        return pts


def _emblem(kind):
    """Unit emblem outline (CCW, fits a circle of radius ~1)."""
    if kind == "star":
        return [(cos(pi / 2 + k * pi / 5) * (1.0 if k % 2 == 0 else 0.45), sin(pi / 2 + k * pi / 5) * (1.0 if k % 2 == 0 else 0.45) - 0.06)
                for k in range(10)]
    if kind == "heart":
        pts = []
        for k in range(20):
            t = 2 * pi * k / 20
            pts.append((16 * sin(t) ** 3 / 17.0, (13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t)) / 17.0 + 0.12))
        return _ccw(pts)
    return [(-0.74, -0.66), (0.74, -0.66), (0.8, -0.3), (1.0, 0.5), (0.47, -0.02), (0.0, 0.84), (-0.47, -0.02), (-1.0, 0.5), (-0.8, -0.3)]


def _foil_lancet(A, grow=0.003):
    """Trefoil-headed light outline inside lancet A (two side lobes and a top lobe meeting in cusps)."""
    b, R = A.a, A.R
    phi = 0.30 * A.th(0)

    def geom(r):
        S = (b - R + (R - r) * cos(phi), A.zs + (R - r) * sin(phi))
        T = (0.0, A.zs + sqrt(max((R - r) ** 2 - (R - b) ** 2, 0.0)))
        return S, T

    lo, hi = 0.15 * b, 0.85 * b
    for _ in range(28):
        r = 0.5 * (lo + hi)
        S, T = geom(r)
        if hypot(S[0] - T[0], S[1] - T[1]) > 1.8 * r:
            lo = r
        else:
            hi = r
    S, T = geom(r)
    ztan = A.zs + R * sin(phi)
    big = 1e3
    lower = _clip(A.outline(r=grow, n=12), [(-big, -big), (big, -big), (big, ztan), (-big, ztan)])
    cx = A.cx
    discs = [(cx + S[0], S[1], r), (cx - S[0], S[1], r), (cx, T[1], r)]
    tri = [(cx + S[0], S[1]), (cx, T[1]), (cx - S[0], S[1])]
    return _simplify(_star((cx, ztan - 0.01), discs, [lower, tri], n=300), 0.002)


def _foil_circle(c, ro, n, a0=pi / 2):
    """n-foil outline inside a circle (lobes tangent to the circle, meeting in cusps)."""
    sn = sin(pi / n)
    rl = ro / (1.0 + 0.86 / sn)
    d = ro - rl
    cs = [(c[0] + d * cos(a0 + 2 * pi * k / n), c[1] + d * sin(a0 + 2 * pi * k / n)) for k in range(n)]
    return _simplify(_star(c, [(x, z, rl) for x, z in cs], [cs], n=300), 0.002)


def _petal(ro, rin, bs, npet, cusped):
    """Wheel-rose petal in local (s along the petal axis, t across): a wedge between two spokes of constant
    width bs, closed by the hub ring (radius rin) and by a trefoil-cusped (or plain round) head under the
    outer ring (radius ro)."""
    al = pi / npet
    sa, ca = sin(al), cos(al)
    ph = max(al - asin(min(0.95, bs / (2 * rin))), 0.03)
    inner = [(rin * cos(ph * (1 - 2 * i / 3)), rin * sin(ph * (1 - 2 * i / 3))) for i in range(4)]
    dc = (ro + bs / 2) / (1 + sa)
    rc = dc * sa - bs / 2
    if cusped and rc >= 0.045:
        lo, hi = 0.25 * rc, 0.9 * rc
        for _ in range(26):
            r = 0.5 * (lo + hi)
            s0, s1 = (bs / 2 + r) / sa, ro
            for _ in range(30):
                s = 0.5 * (s0 + s1)
                t = (s * sa - bs / 2 - r) / ca
                if hypot(s, t) < ro - r:
                    s0 = s
                else:
                    s1 = s
            if hypot(s - (ro - r), t) > 1.8 * r:
                lo = r
            else:
                hi = r
        discs = [(s, t, r), (s, -t, r), (ro - r, 0.0, r)]
        Q = (s - r * sa, t + r * ca)
        extra = [[(Q[0], -Q[1]), (s, -t), (ro - r, 0.0), (s, t), Q]]
    else:
        discs = [(dc, 0.0, rc)]
        Q = (dc - rc * sa, rc * ca)
        extra = []
    wedge = [(Q[0], -Q[1]), Q] + inner
    return _simplify(_star((Q[0], 0.0), discs, [wedge] + extra, n=200), clamp(0.006 * ro, 0.002, 0.003))


# ----------------------------------------------------------------------------- mesh helpers
def _cen(pts):
    n = float(len(pts))
    return (sum(q[0] for q in pts) / n, sum(q[1] for q in pts) / n, sum(q[2] for q in pts) / n)


def _newell(pts):
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return nx, ny, nz


def _loft(name, rings, lo=0, hi=None, caps=(False, False)):
    """Skin between successive rings (equal point counts) along section edges lo .. hi-1 (edge j joins point
    j and j+1); the other edges lie against the wall or inside other stone and stay open. Faces are wound
    outwards from the section's centroid; caps close the first / last ring."""
    m, nr = len(rings[0]), len(rings)
    hi = m - 1 if hi is None else hi
    verts = [tuple(q) for r in rings for q in r]
    k0 = (nr - 1) // 2
    quad = [rings[k0][lo], rings[k0 + 1][lo], rings[k0 + 1][lo + 1], rings[k0][lo + 1]]
    c, f, nn = _cen(rings[k0] + rings[k0 + 1]), _cen(quad), _newell(quad)
    flip = sum(nn[i] * (f[i] - c[i]) for i in range(3)) < 0
    faces = []
    for k in range(nr - 1):
        for j in range(lo, hi):
            j2 = (j + 1) % m
            q = (k * m + j, (k + 1) * m + j, (k + 1) * m + j2, k * m + j2)
            faces.append(q[::-1] if flip else q)
    for end, other, on in ((0, 1, caps[0]), (nr - 1, nr - 2, caps[1])):
        if not on:
            continue
        idx = [end * m + j for j in range(m)]
        nn, c0, c1 = _newell(rings[end]), _cen(rings[end]), _cen(rings[other])
        if sum(nn[i] * (c0[i] - c1[i]) for i in range(3)) < 0:
            idx.reverse()
        faces.append(tuple(idx))
    return mesh_object(name, verts, faces)


def _drop_back(o, front_only=False, keep=None):
    """Remove the faces of a stood-up shape_extrude that are never seen: those looking into the wall and,
    with `keep` (a 2D outline), the edge faces outside it that are buried in the surrounding stone."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    dead = [f for f in bm.faces if f.normal.z < (0.5 if front_only else -0.3)]
    if keep is not None:
        gone = set(dead)
        side = [f for f in bm.faces if f not in gone and f.normal.z < 0.9]
        if side:
            cen = np.array([[f.calc_center_median().x, f.calc_center_median().y] for f in side])
            dead += [f for f, ins in zip(side, _pip(cen, keep)) if not ins]
    bmesh.ops.delete(bm, geom=dead, context='FACES')
    bm.to_mesh(o.data)
    bm.free()
    return o


def _mitre(pts, closed, hw):
    """Mitred half-width offsets (dx, dz) for every point of a 2D polyline."""
    n = len(pts)
    out = []
    for i in range(n):
        p = pts[i]
        pa = pts[i - 1] if (closed or i > 0) else None
        pb = pts[(i + 1) % n] if (closed or i < n - 1) else None
        dirs = []
        for q0, q1 in ((pa, p), (p, pb)):
            if q0 is None or q1 is None:
                continue
            dx, dz = q1[0] - q0[0], q1[1] - q0[1]
            ln = hypot(dx, dz) or 1e-9
            dirs.append((-dz / ln, dx / ln))
        mx, mz = sum(d[0] for d in dirs), sum(d[1] for d in dirs)
        ml = hypot(mx, mz)
        if ml < 1e-6:
            mx, mz, k = dirs[0][0], dirs[0][1], 1.0
        else:
            mx, mz = mx / ml, mz / ml
            k = 1.0 / max(mx * dirs[0][0] + mz * dirs[0][1], 0.4)
        out.append((mx * hw * k, mz * hw * k))
    return out


def _came(verts, faces, pts, closed, w, y0, y1):
    """Lead came of width w along a 2D polyline: a ridge from the glass face y1 up to the crown y0."""
    n = len(pts)
    if n < 2:
        return
    b = len(verts)
    for p, (ox, oz) in zip(pts, _mitre(pts, closed, 0.5 * w)):
        verts += [(p[0] + ox, y1, p[1] + oz), (p[0], y0, p[1]), (p[0] - ox, y1, p[1] - oz)]
    for i in range(n if closed else n - 1):
        a, c = b + 3 * i, b + 3 * ((i + 1) % n)
        faces += [(a, a + 1, c + 1, c), (a + 1, a + 2, c + 2, c + 1)]


def _band(verts, faces, pts, closed, w, y):
    """Flat ribbon of width w along a 2D polyline (the dirt halo under a came)."""
    n = len(pts)
    if n < 2:
        return
    b = len(verts)
    for p, (ox, oz) in zip(pts, _mitre(pts, closed, 0.5 * w)):
        verts += [(p[0] + ox, y, p[1] + oz), (p[0] - ox, y, p[1] - oz)]
    for i in range(n if closed else n - 1):
        a, c = b + 2 * i, b + 2 * ((i + 1) % n)
        faces.append((a + 1, c + 1, c, a))


def _rosette(name, c, y, r, n, rot, h):
    """Carved boss facing -Y: n pointed petals with a raised midrib round a domed centre."""
    verts, faces = [], []

    def add(u, v, d):
        verts.append((c[0] + u, y - d, c[1] + v))
        return len(verts) - 1

    for k in range(n):
        th = rot + 2 * pi * k / n
        cx, sx = cos(th), sin(th)
        wd = r * min(0.42, 0.95 * sin(pi / n))

        def P(s, t, d):
            return add(s * cx - t * sx, s * sx + t * cx, d)

        B, Rr, L = P(0.22 * r, 0.0, 0.55 * h), P(0.62 * r, -wd, -0.004), P(0.62 * r, wd, -0.004)
        M, T = P(0.6 * r, 0.0, h), P(r, 0.0, -0.003)
        faces += [(B, Rr, M), (B, M, L), (M, Rr, T), (M, T, L)]
    rb, ns = 0.3 * r, 6
    r0 = [add(rb * cos(2 * pi * i / ns), rb * sin(2 * pi * i / ns), 0.3 * h) for i in range(ns)]
    r1 = [add(0.62 * rb * cos(2 * pi * i / ns), 0.62 * rb * sin(2 * pi * i / ns), 1.15 * h) for i in range(ns)]
    top = add(0.0, 0.0, 1.4 * h)
    for i in range(ns):
        j = (i + 1) % ns
        faces += [(r0[i], r0[j], r1[j], r1[i]), (r1[i], r1[j], top)]
    return mesh_object(name, verts, faces)


def _stain(name, x, y, z1, length, w):
    """Rain-stain ribbon lying on a face that looks -Y: full width at the top, running out to a point. Its
    edge vertices take the tone of the stone below, only the middle line is dirty, so it has no outline."""
    rows = [(0.0, 1.0), (0.5, 0.8), (0.85, 0.45)]
    verts = []
    for s, f in rows:
        z = z1 - s * length
        verts += [(x - 0.5 * w * f, y, z), (x, y, z), (x + 0.5 * w * f, y, z)]
    verts.append((x, y, z1 - length))
    faces = []
    for i in range(len(rows) - 1):
        b, c = 3 * (i + 1), 3 * i
        faces += [(b, b + 1, c + 1, c), (b + 1, b + 2, c + 2, c + 1)]
    t, b = len(verts) - 1, 3 * (len(rows) - 1)
    faces += [(t, b + 1, b), (t, b + 2, b + 1)]
    return mesh_object(name, verts, faces)


def _cushion(name, c, nrm, along, r, h, rt, n=7):
    """Moss cushion: a low lumpy dome on the surface point c (normal nrm), stretched along `along`."""
    nrm = np.array(nrm, float)
    nrm /= np.linalg.norm(nrm)
    u = np.array(along, float)
    u -= nrm * u.dot(nrm)
    u /= (np.linalg.norm(u) or 1.0)
    v = np.cross(nrm, u)
    c = np.array(c, float)
    verts = [tuple(c + nrm * h)]
    ph = rt.uniform(0, 2 * pi)
    for fr, fh in ((0.55, 0.72), (1.0, -0.25)):
        for k in range(n):
            th = ph + 2 * pi * (k + rt.uniform(-0.2, 0.2)) / n
            rad = r * fr * rt.uniform(0.7, 1.2)
            verts.append(tuple(c + u * (1.6 * rad * cos(th)) + v * (rad * sin(th)) + nrm * (h * fh * rt.uniform(0.7, 1.15))))
    faces = []
    for k in range(n):
        j = (k + 1) % n
        faces += [(0, 1 + k, 1 + j), (1 + k, 1 + n + k, 1 + n + j, 1 + j)]
    return mesh_object(name, verts, faces)


# ----------------------------------------------------------------------------- build
def build(p, rng):
    W, H, D = p.width, p.height, p.relief
    fw = clamp(0.07 + 0.07 * W, 0.12, 0.24)            # face width of the stone surround
    hw = clamp(0.4 * fw + 0.01, 0.055, 0.1)            # hood mould width
    xs = 0.02                                          # label stops stand this much proud of the hood
    a = W / 2 - fw - hw - xs                           # clear half span of the opening
    hs = clamp(0.13 + 0.03 * W, 0.15, 0.2)             # sill height
    dout = fw + hw
    zmin = hs + 0.12 + 0.1 * a
    R = a if p.arch == "round" else max(a, 2.0 * a * p.point)

    def rise(rad):
        return sqrt((rad + dout) ** 2 - (rad - a) ** 2)

    while H - rise(R) < zmin and R > a + 1e-5:         # too squat for the asked sharpness: flatten the arch
        R = max(a, R - 0.03 * a)
    zs = max(H - rise(R), zmin)
    A0 = Arch(0.0, a, hs, zs, R)
    apex0 = A0.apex()

    yg = -0.28 * D                                     # glass plane
    y_tf = yg - 0.040                                  # front of the main tracery order
    y_ff = yg - 0.060                                  # front of the fillet order
    d0 = -y_ff + 0.008                                 # depth of the surround's inner edge
    dh = D + 0.07                                      # hood nose projection
    ds = D + 0.075                                     # sill nose projection

    tints, (c_rim, c_head, c_pool) = PALETTES[p.palette]
    st = mat("stone_smooth", p.stone)
    lead = mat("lead")
    iron = mat("iron_wrought")
    gl = [mat("glass_stained", tints[i], glow=0.72 * p.glow * (1.0, 0.96, 0.86, 0.9, 1.05)[i]) for i in range(5)]
    # a smoky film in front of the glass: per-quarry patina, the dim foot and corners, dirt along the cames
    patina = mat("glass_clear", "#1e150c", opacity=0.24)

    groups = {"frame": [], "trac": [], "hood": [], "sill": [], "label": []}

    tone = {}

    def stone(o, g, r=None):
        """Register a stone; every block gets its own tone (a little dirtier or a little more scrubbed)."""
        tone[o.name] = (r.u(0.0, 0.32), r.u(0.0, 0.3)) if r is not None else (0.1, 0.08)
        paint(o, grime=tone[o.name][0], wear=tone[o.name][1], mode="set")
        groups[g].append(o)
        return o

    # ---- sill: stooled top, weathering slope, nose with drip throat, apron; one to three stones
    rs = rng.fork("sill")
    sprof = [(0.0, hs), (-(d0 - 0.02), hs), (-(d0 + 0.004), hs), (-0.5 * (d0 + ds), hs - 0.022), (-(ds - 0.010), hs - 0.040),
             (-ds, hs - 0.049), (-ds, hs - 0.082), (-(ds - 0.008), hs - 0.090), (-(ds - 0.022), hs - 0.090),
             (-(ds - 0.031), hs - 0.079), (-(ds - 0.040), hs - 0.090), (-(D + 0.008), hs - 0.100), (-(D + 0.008), 0.03),
             (-(D - 0.02), 0.0), (0.0, 0.0)]
    scy, scz = sum(q[0] for q in sprof) / len(sprof), sum(q[1] for q in sprof) / len(sprof)

    def sring(x, shrink):
        out = []
        for (y, z) in sprof:
            if shrink and y < -0.02:
                ey, ez = scy - y, scz - z
                el = hypot(ey, ez) or 1.0
                y, z = y + ey / el * shrink, z + ez / el * shrink
            out.append((x, y, z))
        return out

    ns = 1 if W < 1.5 else (2 if W < 2.1 else 3)
    cuts = [-W / 2] + [-W / 2 + W * (k + rs.jitter(0.0, 0.12)) / ns for k in range(1, ns)] + [W / 2]
    for k in range(ns):
        x0, x1 = cuts[k], cuts[k + 1]
        xsn = [x0 + 0.006]
        if k == 0:                                      # close columns at the ends carry the rain streaks
            xsn += [x0 + 0.035, x0 + 0.075, x0 + 0.125]
        lo_, hi_ = xsn[-1], x1 - (0.125 if k == ns - 1 else 0.006)
        n = max(1, int(ceil((hi_ - lo_) / clamp(0.065 * W, 0.085, 0.125))))   # close enough for moss patches
        xsn += [lo_ + (hi_ - lo_) * i / n for i in range(1, n + 1)]
        if k == ns - 1:
            xsn += [x1 - 0.075, x1 - 0.035, x1 - 0.006]
        rings = [sring(x0, 0.005)] + [sring(x, 0.0) for x in xsn] + [sring(x1, 0.005)]
        s = _loft(f"sill{k}", rings, 0, len(sprof) - 1, caps=(k == 0, k == ns - 1))
        displace_noise(s, 0.0015, 0.22, seed=p.seed + 3, along_normal=False)
        stone(s, "sill", rs)

    # ---- surround: jamb courses and voussoirs carrying the reveal moulding (quirk, cavetto, bead)
    ch = 0.008
    c1 = fw * 0.52
    rbd = min(0.016, fw * 0.1)
    ya, yb = -D + 0.008, -d0
    xb = 2 * rbd
    u_, v_ = c1 - 0.004 - xb, yb - ya

    def sect(fo):
        s = [(0.0, 0.0), (fo, 0.0), (fo, -(D - ch)), (fo - ch, -D), (c1, -D), (c1 - 0.004, ya),
             (xb + u_ * cos(pi / 4), ya + v_ * sin(pi / 4))]
        return s + [(rbd + rbd * cos(q), yb - rbd * sin(q)) for q in (0.0, pi / 3, 2 * pi / 3, pi)]

    def ring(side, zone, v, sec, dy=0.0, chip=0.0):
        """Section carried to a station of the arch; dy pulls the worked faces back (V-joints, uneven bedding),
        chip knocks the outer arris off."""
        out = []
        for j, (r, y) in enumerate(sec):
            if y < -0.012:
                y += dy
            if chip and j in (2, 3):
                y += chip
                r -= 0.6 * chip
            x, z = A0.pt(side, zone, v, r)
            out.append((x, min(y, 0.0), z))
        return out

    cj = 0.006
    zj0 = hs - 0.03
    la = A0.arc_len(fw * 0.5)
    key = p.arch == "round" or R < a * 1.05
    tol = 0.0016 * clamp(H / 2.8, 1.0, 1.6)
    chord = clamp(sqrt(8 * (R + fw) * tol), 0.07, 0.2)
    nv = max(3, int(la / 0.21 + 0.5))
    tk0 = 1.0 - 0.55 / nv if key else 1.0
    drop = hw * 0.9
    zt = zs - drop + 0.004                              # top of the label stops
    zl_bot = zt - 0.245
    zl = zl_bot + 0.012                                # a bed joint right under the label stops
    nj = (max(1, int((zl - zj0) / 0.3 + 0.5)) + 1) if zl - zj0 > 0.15 else 1
    yt = clamp(0.3 * D, 0.04, 0.06)
    rj = rng.fork("jambs")

    def tail(side, lo, hi, tl, dep, fine):
        """Long quoin: the return of a jamb stone toothed into the wall, standing a little proud of it."""
        c = 0.005
        x0, x1 = a + fw - 0.004, a + fw + tl
        nx = 3 if fine else 0
        xi = [x1 - c - (x1 - c - x0) * i / (nx + 1) for i in range(1, nx + 1)]
        zz = [lo, lo + c] + ([lo + c + (hi - lo - 2 * c) * i / 4 for i in range(1, 4)] if fine else []) + [hi - c, hi]
        rings = []
        for k, z in enumerate(zz):
            e = c if k in (0, len(zz) - 1) else 0.0
            pts = [(x1, 0.0), (x1, -(dep - c)), (x1 - c, -(dep - e))] + [(x, -(dep - e)) for x in xi] + [(x0, -(dep - e)), (x0, 0.0)]
            rings.append([(side * x, y, z) for x, y in pts])
        return _loft("quoin", rings, 0, len(rings[0]) - 2, caps=(True, True))

    jtop = {}
    for side in (1, -1):
        r_ = rj.fork("R" if side > 0 else "L")
        if nj > 1:
            wj = [r_.u(0.8, 1.2) for _ in range(nj - 1)]                # course heights +-20 %
            zb = [zj0 + (zl - zj0) * float(t) for t in np.cumsum([0.0] + wj) / sum(wj)] + [zs]
        else:
            zb = [zj0, zs]
        wv = [r_.u(0.85, 1.15) for _ in range(nv)]
        tb = [tk0 * float(t) for t in np.cumsum([0.0] + wv) / sum(wv)]
        blocks = [('j', zb[k], zb[k + 1], 1.0) for k in range(nj)] + [('a', tb[k], tb[k + 1], 1.0 / la) for k in range(nv)]
        nb = len(blocks)
        yjs = [r_.jitter(0.0, 0.0035) for _ in range(nb)]          # each stone bedded a few mm in or out
        fos = [fw + r_.jitter(0.0, 0.003) for _ in range(nb)]
        chp = [0.0] + [(r_.u(0.008, 0.02) if r_.chance(0.4) else r_.u(0.0, 0.003)) for _ in range(nb - 1)] + [0.0]
        yJ = [0.0] + [0.5 * (yjs[k - 1] + yjs[k]) for k in range(1, nb)] + [0.5 * (yjs[-1] - 0.014) if key else 0.0]
        fJ = [fw] + [0.5 * (fos[k - 1] + fos[k]) for k in range(1, nb)] + [fw]
        jtop[side] = yJ[-1]
        for k, (zone, v0, v1, sc) in enumerate(blocks):
            c_ = cj * sc
            sk = sect(fos[k])
            rings = [ring(side, zone, v0, sect(fJ[k]), cj + yJ[k], chp[k]),
                     ring(side, zone, v0 + c_, sk, yjs[k], 0.45 * chp[k])]
            if zone == 'j':
                if v0 < hs + 0.5:                       # closer rings near the foot carry the moss line
                    n = max(1, int(ceil((v1 - v0) / 0.14)))
                    rings += [ring(side, zone, v0 + (v1 - v0) * i / n, sk, yjs[k]) for i in range(1, n)]
            else:
                n = max(1, int((v1 - v0) * la / chord + 0.7))
                rings += [ring(side, zone, v0 + (v1 - v0) * i / n, sk, yjs[k]) for i in range(1, n)]
            rings += [ring(side, zone, v1 - c_, sk, yjs[k], 0.45 * chp[k + 1]),
                      ring(side, zone, v1, sect(fJ[k + 1]), cj + yJ[k + 1], chp[k + 1])]
            blk = stone(_loft("jamb" if zone == 'j' else "vouss", rings, 1, 10), "frame", r_)
            if zone == 'j' and nj > 1 and k == nj - 2:
                # rain running off the label stop has stained the face of the stone below it
                wf = fos[k] - ch - c1                   # width of the block's face
                for fq, fl, fw_ in ((0.26, r_.u(0.85, 1.0), 0.46), (0.63, r_.u(0.45, 0.7), 0.34), (0.87, r_.u(0.25, 0.4), 0.22)):
                    w_ = wf * fw_ * r_.u(0.85, 1.1)
                    stn_ = _stain("rain_stain", side * (a + fos[k] - ch - wf * fq), -D + yjs[k] - 0.0012,
                                  v1 - 0.005, (v1 - v0 - 0.02) * fl, w_)
                    g0, w0 = tone[blk.name]
                    xc_, zt_, ln_ = stn_.data.vertices[1].co.x, v1 - 0.005, (v1 - v0) * fl
                    amp = max(r_.u(0.66, 0.78) - g0, 0.3)
                    paint(stn_, wear=w0, grime=lambda co, n, xc_=xc_, w_=w_, amp=amp, g0=g0, zt_=zt_, ln_=ln_: g0 + amp * (
                        np.abs(co[:, 0] - xc_) < 0.1 * w_) * np.clip(1.0 - 0.75 * (zt_ - co[:, 2]) / ln_, 0.0, 1.0), mode="set")
                    groups["frame"].append(stn_)
        # every other course is a long stone; the one right under the label stop always is (it takes the drip)
        vis = [k for k in range(nj) if zb[k + 1] <= zl_bot + 0.03]
        for k in vis:
            lo_, hi_ = max(zb[k], hs - 0.02) + 0.002, zb[k + 1] - 0.002
            if (vis[-1] - k) % 2 == 0 and hi_ - lo_ > 0.12:
                top = k == vis[-1]
                stone(tail(side, lo_, hi_, (hw + xs - 0.006) if top else hw * r_.u(0.6, 0.85),
                           yt + r_.jitter(0.0, 0.004), top), "frame", r_)
    if key:                                             # round arch: a keystone straddles the crown, a little proud
        sk = sect(fw)
        ct_ = cj / la
        tm = 0.5 * (tk0 + 1.0)
        rings = [ring(1, 'a', tk0, sk, cj + jtop[1]), ring(1, 'a', tk0 + ct_, sk, -0.014), ring(1, 'a', tm, sk, -0.014),
                 ring(1, 'a', 1.0, sk, -0.014), ring(-1, 'a', tm, sk, -0.014), ring(-1, 'a', tk0 + ct_, sk, -0.014),
                 ring(-1, 'a', tk0, sk, cj + jtop[-1])]
        stone(_loft("keystone", rings, 1, 10), "frame", rj)

    # the bead of the reveal reads as a nook shaft: moulded capital at the springing, base on the sill
    cap_p = [(rbd * 1.05, -0.036), (rbd * 1.5, -0.03), (rbd * 1.15, -0.02), (rbd * 1.25, -0.004), (rbd * 1.95, 0.022),
             (rbd * 2.05, 0.03), (rbd * 2.05, 0.042), (0.0, 0.046)]
    base_p = [(rbd * 2.1, 0.0), (rbd * 2.1, 0.022), (rbd * 1.6, 0.03), (rbd * 1.8, 0.042), (rbd * 1.6, 0.054), (rbd * 1.1, 0.062)]
    for side in (1, -1):
        stone(lathe("shaft_cap", cap_p, segments=6, loc=(side * (a + rbd), yb, zs - 0.01)), "frame")
        stone(lathe("shaft_base", base_p, segments=6, loc=(side * (a + rbd), yb, hs - 0.014)), "frame")

    # ---- hood mould: weathered top, fillet, roll nose, deep hollow (drip throat) - long stones on label stops
    r0 = fw - 0.003
    e_ = dh - D
    rr_ = min(0.19 * hw, 0.4 * e_)
    crr, cry = fw + 0.42 * hw, -(dh - rr_)
    hsect = [(r0, 0.0), (fw + hw, 0.0), (fw + hw, -0.016), (fw + 0.80 * hw, -(D + 0.30 * e_)),
             (crr + 1.55 * rr_, cry + 0.95 * rr_), (crr + 1.03 * rr_, cry + 0.80 * rr_)]
    hsect += [(crr + rr_ * cos(radians(q)), cry - rr_ * sin(radians(q))) for q in (15, 80, 150, 215)]
    hsect += [(fw + 0.46 * hw, -(D + 0.56 * e_)), (fw + 0.46 * hw, -(D + 0.30 * e_)), (fw + 0.10 * hw, -(D + 0.14 * e_)),
              (r0, -(D + 0.004))]
    mh = len(hsect)
    lh = A0.arc_len(fw + hw * 0.5)
    nh = max(2, int(lh / 0.5 + 0.5))
    cht = cj / lh
    chord_h = clamp(sqrt(8 * (R + dout) * tol), 0.07, 0.2)
    rh = rng.fork("hood")
    for side in (1, -1):
        r_ = rh.fork("R" if side > 0 else "L")
        hb = [0.0] + [(k + r_.jitter(0.0, 0.12)) / nh for k in range(1, nh)] + [1.0]
        yhs = [r_.jitter(0.0, 0.003) for _ in range(nh)]
        yhJ = [0.0] + [0.5 * (yhs[k - 1] + yhs[k]) for k in range(1, nh)] + [0.0]
        jtop[("h", side)] = (hb, yhs, yhJ)
    for side in (1, -1):
        hb, yhs, yhJ = jtop[("h", side)]
        for k in range(nh):
            t0, t1 = hb[k], hb[k + 1]
            n = max(1, int(ceil((t1 - t0) * lh / chord_h)))
            mid = [ring(side, 'a', t0 + (t1 - t0) * i / n, hsect, yhs[k]) for i in range(1, n)]
            if k == 0:                                  # the leg drops straight onto the label stop
                rings = [ring(side, 'j', zs - drop, hsect, yhs[k]), ring(side, 'a', 0.0, hsect, yhs[k])] + mid
            else:
                rings = [ring(side, 'a', t0, hsect, cj + yhJ[k]), ring(side, 'a', t0 + cht, hsect, yhs[k])] + mid
            if k == nh - 1 and key:                     # round hood: one crown stone runs over the apex
                if side == -1:
                    continue
                ob, oy, oJ = jtop[("h", -1)]
                u0 = ob[nh - 1]
                n2 = max(1, int(ceil((1.0 - u0) * lh / chord_h)))
                rings += [ring(1, 'a', 1.0, hsect, yhs[k])]
                rings += [ring(-1, 'a', 1.0 - (1.0 - u0) * i / n2, hsect, yhs[k]) for i in range(1, n2)]
                rings += [ring(-1, 'a', u0 + cht, hsect, yhs[k]), ring(-1, 'a', u0, hsect, cj + oJ[nh - 1])]
            else:
                rings += [ring(side, 'a', t1 - cht, hsect, yhs[k]), ring(side, 'a', t1, hsect, cj + yhJ[k + 1])]
            ho = _loft("hood", rings, 1, mh - 1)
            stone(ho, "hood", rh)
        # label stop: moulded abacus and necking over a corbel bell carved with a rosette
        cw2 = 0.5 * (hw + 2 * xs - 0.004)
        yf = dh + 0.008
        xc = side * (a + fw + hw * 0.5)
        lprof = [(0.0, cw2, yf, 0.006), (-0.024, cw2, yf, 0.006), (-0.031, cw2 - 0.008, yf - 0.008, 0.006),
                 (-0.041, cw2 - 0.010, yf - 0.010, 0.007), (-0.047, cw2 - 0.004, yf - 0.004, 0.008),
                 (-0.055, cw2 - 0.010, yf - 0.012, 0.009), (-0.072, cw2 - 0.006, yf - 0.016, 0.010),
                 (-0.158, cw2 - 0.010, yf - 0.019, 0.010), (-0.205, 0.6 * cw2, 0.55 * yf, 0.010),
                 (-0.245, 0.26 * cw2, 0.16 * yf, 0.005)]
        rings = [[(xc - wx, 0.0, zt + dz), (xc - wx, -(dy - c), zt + dz), (xc - wx + c, -dy, zt + dz),
                  (xc + wx - c, -dy, zt + dz), (xc + wx, -(dy - c), zt + dz), (xc + wx, 0.0, zt + dz)] for (dz, wx, dy, c) in lprof]
        r_ = rh.fork("lab%d" % side)
        stone(_loft("label_stop", rings, 0, 5, caps=(True, True)), "label", r_)
        npl = r_.pick([4, 5, 6])
        stone(_rosette("label_boss", (xc, zt - 0.115), -(yf - 0.0175), min(0.037, cw2 - 0.016), npl,
                       pi / 4 if npl == 4 else pi / 2, 0.012), "label", r_)

    # ---- tracery layout: lights (lancets), oculi / wheel rose and pierced spandrels
    mw = clamp(0.1 * a + 0.02, 0.045, 0.085)            # mullion / bar width
    m = 0.75 * mw                                      # tracery margin against the surround
    z0l = hs + 0.022
    holes = []

    def unit(U, mm, bw_, mode, level, main):
        if mode == "single" or (U.a - mm - bw_ / 2) / 2 < 0.085:
            holes.append({"t": "lancet", "A": Arch(U.cx, U.a - mm, U.z0, U.zs, U.R - mm), "main": main})
            return
        b = (U.a - mm - bw_ / 2) / 2
        xr = bw_ / 2 + b
        rl = b * U.R / U.a
        ri = U.R - mm
        e = xr + rl - b

        def solve(ro):
            zo = U.zs + sqrt(max((ri - ro) ** 2 - (U.R - U.a) ** 2, 0.0))
            return zo, sqrt(max((rl + bw_ + ro) ** 2 - e * e, 0.0)) - (zo - U.zs)

        ro_nat = (ri ** 2 - (rl + bw_) ** 2 - (U.R - U.a) ** 2 + e * e) / (2 * (ri + rl + bw_))
        ro = max(ro_nat, (0.72 if mode == "rose" else 0.42) * (U.a - mm))
        while True:
            zo, dl = solve(ro)
            if ro <= ro_nat + 1e-6:
                dl = 0.0
                break
            if U.zs - dl >= U.z0 + 0.6 * b + 0.12:
                break
            ro = max(ro_nat, ro - 0.01)
        holes.append({"t": "rose" if (mode == "rose" and ro > 0.15) else "oculus", "c": (U.cx, zo), "r": ro, "main": main})
        for sx in (-1, 1):
            L = Arch(U.cx + sx * xr, b, U.z0, U.zs - dl, rl)
            if level > 0 and b > 0.3:
                unit(L, 0.0, bw_ * 0.72, "twin", level - 1, False)
            else:
                holes.append({"t": "lancet", "A": L, "main": False})

    unit(Arch(0.0, a, z0l, zs, R), m, mw, p.tracery, 1, True)

    for h in holes:                                     # wheel rose: spokes, hub ring and petal count
        if h["t"] == "rose":
            ro = h["r"]
            bs = mw * 0.68
            h.update(npet=6 if ro < 0.3 else 8, bs=bs, rhub=max(0.26 * ro, 0.6 * bs + 0.02))
    mains = [h for h in holes if h["t"] in ("rose", "oculus") and h["main"]]

    def ncirc(r):
        return int(clamp(r * 95, 12, 28))

    def outline(h, e=0.0):
        t = h["t"]
        if t == "lancet":
            return h["A"].outline(r=e, zb=h["A"].z0 - min(e, 0.012), n=8)
        if t == "eye" and h["tre"]:
            return _foil_circle(h["c"], h["r"] + e, 3, h["a0"])
        r = h["r"] + e
        return _circle(h["c"], r, ncirc(r))

    # spandrels: the largest circles that fit in the solid stone left between the lights, pierced as trefoils
    lanc = [h for h in holes if h["t"] == "lancet"]
    if len(lanc) > 1:
        bar = mw * 0.78
        outer = A0.outline(r=-m * 0.35, zb=z0l, n=14)
        zlo = min(h["A"].zs for h in lanc) - 0.02
        step = clamp(a / 45.0, 0.006, 0.014)
        gx, gz = np.meshgrid(np.arange(0.0, a, step), np.arange(zlo, apex0, step))
        cand = np.stack([gx.ravel(), gz.ravel()], 1)
        keep = _pip(cand, outer)
        bnd = _dense(outer)
        for h in holes:
            o = outline(h)
            keep &= ~_pip(cand, o)
            bnd += _dense(o)
        cand = cand[keep]
        bnd = np.array(bnd)
        picks = 0
        for _ in range(5):
            if not len(cand) or picks >= 2:
                break
            dmin = np.full(len(cand), 1e9)
            for i0 in range(0, len(bnd), 400):
                bb = bnd[i0:i0 + 400]
                dmin = np.minimum(dmin, np.sqrt(((cand[:, None, :] - bb[None, :, :]) ** 2).sum(-1)).min(1))
            i = int(np.argmax(dmin - 0.02 * cand[:, 0]))      # prefer the centre line on ties
            re = float(dmin[i]) - bar
            if re < 0.03:
                break
            cx_, cz_ = float(cand[i, 0]), float(cand[i, 1])
            if cx_ < re + bar * 0.5:
                if cx_ > step * 1.5:
                    cand = cand[np.hypot(cand[:, 0] - cx_, cand[:, 1] - cz_) > re + bar]
                    continue
                cx_ = 0.0
            mc = mains[0]["c"] if mains else (0.0, apex0)
            for sx in ((1, -1) if cx_ > 0 else (1,)):
                a0 = atan2(cz_ - mc[1], sx * cx_ - mc[0]) if cx_ > 0 else (-pi / 2 if cz_ < mc[1] else pi / 2)
                holes.append({"t": "eye", "c": (sx * cx_, cz_), "r": re, "main": False, "tre": re >= 0.045, "a0": a0})
            picks += 1
            bnd = np.concatenate([bnd, np.array(_circle((cx_, cz_), re, 40))])
            cand = cand[np.hypot(cand[:, 0] - cx_, cand[:, 1] - cz_) > re + 1e-4]

    # ---- stone tracery: main order, a narrower fillet order in front, cusp plates around the glass
    tr_out = A0.outline(r=0.025, zb=hs - 0.012, n=12)
    tr_keep = A0.outline(r=0.01, zb=hs - 0.004, n=12)
    bev = 0.005
    tr = shape_extrude("tracery", tr_out, [outline(h) for h in holes], depth=-y_tf - 0.004, loc=(0, -0.004, 0), rot=(90, 0, 0))
    fe = min(0.016, mw * 0.26)
    fil = shape_extrude("tracery_fillet", tr_out, [outline(h, fe + bev) for h in holes], depth=y_tf - y_ff + 0.002,
                        loc=(0, y_tf + 0.002, 0), rot=(90, 0, 0), bevel=bev, bevel_res=0)
    stone(_drop_back(tr, keep=tr_keep), "trac")
    stone(_drop_back(fil, keep=tr_keep), "trac")
    for h in holes:
        if h["t"] == "lancet" and h["A"].a >= 0.1 and p.cusps:
            A = h["A"]
            cp = shape_extrude("cusps", A.outline(r=0.012, zb=A.z0 - 0.012, n=8), [_foil_lancet(A)], depth=0.036,
                               loc=(0, yg + 0.018, 0), rot=(90, 0, 0), bevel=0.003 if len(lanc) <= 2 else 0.0, bevel_res=0)
            stone(_drop_back(cp, keep=A.outline(r=0.005, zb=A.z0 - 0.005, n=8)), "trac")
        elif h["t"] == "oculus" and h["r"] >= 0.085 and p.cusps:
            nf = 4 if h["r"] < 0.2 else 6
            cp = shape_extrude("cusps", _circle(h["c"], h["r"] + 0.012, ncirc(h["r"])), [_foil_circle(h["c"], h["r"], nf)],
                               depth=0.036, loc=(0, yg + 0.018, 0), rot=(90, 0, 0), bevel=0.003, bevel_res=0)
            stone(_drop_back(cp, keep=_circle(h["c"], h["r"] + 0.005, 24)), "trac")
        elif h["t"] == "rose":
            # the wheel: spokes of constant width radiating from a hub ring to cusped petal heads under the ring
            c, ro, npet, bs, rhub = h["c"], h["r"], h["npet"], h["bs"], h["rhub"]
            wb = 0.005
            pet = _petal(ro + wb, rhub + bs - wb, bs - 2 * wb, npet, p.cusps)
            cut = [_circle(c, rhub + wb, ncirc(rhub) + 2)]
            for k in range(npet):
                th = pi / 2 + 2 * pi * k / npet
                cut.append([(c[0] + s * cos(th) - t * sin(th), c[1] + s * sin(th) + t * cos(th)) for s, t in pet])
            wh = shape_extrude("wheel", _circle(c, ro + 0.014, ncirc(ro)), cut, depth=0.05,
                               loc=(0, yg + 0.018, 0), rot=(90, 0, 0), bevel=wb, bevel_res=0)
            stone(_drop_back(wh, keep=_circle(c, ro + 0.006, 32)), "trac")

    # ---- leaded glazing
    G = {i: [] for i in range(5)}                      # convex glass pieces per tint: (outline, patina weight)
    GO = {i: [] for i in range(5)}                     # glass objects with holes (medallion grounds, emblems)
    C = []                                             # cames: (polyline, closed, width)
    CB = []                                            # short cames between border pieces
    gth = 0.004
    ypl = yg - gth / 2                                 # nominal glass face
    tuck = 0.012
    emb = _emblem(p.emblem)
    rgl = rng.fork("glass")

    def medallion(shape, c, er, ground):
        """Medallion ground `shape` with the emblem leaded into it."""
        if er < 0.035:
            G[ground].append((shape, 0.5))
            return
        em = [(c[0] + x * er, c[1] + z * er) for x, z in emb]
        GO[ground].append(shape_extrude("gl_ground", shape, [em], depth=gth, loc=(0, yg + gth / 2, 0), rot=(90, 0, 0)))
        GO[4].append(shape_extrude("gl_emblem", em, [], depth=gth, loc=(0, yg + gth / 2, 0), rot=(90, 0, 0)))
        C.append((em, True, 0.007))

    def ring_q(c, ra, rb_, a0, a1, nsub):
        out = []
        for k in range(nsub):
            u0, u1 = a0 + (a1 - a0) * k / nsub, a0 + (a1 - a0) * (k + 1) / nsub
            out.append([(c[0] + ra * cos(u0), c[1] + ra * sin(u0)), (c[0] + rb_ * cos(u0), c[1] + rb_ * sin(u0)),
                        (c[0] + rb_ * cos(u1), c[1] + rb_ * sin(u1)), (c[0] + ra * cos(u1), c[1] + ra * sin(u1))])
        return out

    def glaze_lancet(A, big):
        bw = clamp(0.2 * A.a, 0.028, 0.055)
        pit = max(2.6 * bw, 0.105)                      # length of a border piece
        zb1 = A.z0 + bw
        # border: alternating coloured strips between two cames
        seq = [3, 2]
        k = 0
        xin = A.a - bw
        nb = max(1, int(2 * xin / (2.6 * bw) + 0.5))
        for i in range(nb):
            x0, x1 = -xin + 2 * xin * i / nb, -xin + 2 * xin * (i + 1) / nb
            G[seq[k % 2]].append(([(A.cx + x0, A.z0 - tuck), (A.cx + x1, A.z0 - tuck), (A.cx + x1, zb1), (A.cx + x0, zb1)], 0.6))
            k += 1
            if i:
                CB.append(([(A.cx + x0, A.z0 - tuck), (A.cx + x0, zb1)], False, 0.007))
        for side in (1, -1):
            xo, xi = A.cx + side * (A.a + tuck), A.cx + side * xin
            G[4].append(([(xi, A.z0 - tuck), (xo, A.z0 - tuck), (xo, zb1), (xi, zb1)], 0.6))
            CB.append(([(xi, A.z0 - tuck), (xi, zb1)], False, 0.007))
            nj_ = max(1, int((A.zs - zb1) / pit + 0.5))
            k = 0
            for i in range(nj_):
                z0_, z1_ = zb1 + (A.zs - zb1) * i / nj_, zb1 + (A.zs - zb1) * (i + 1) / nj_
                G[seq[k % 2]].append(([(xi, z0_), (xo, z0_), (xo, z1_), (xi, z1_)], 0.6))
                CB.append(([(xi, z0_), (xo, z0_)], False, 0.007))
                k += 1
            na = max(2, int(A.arc_len(0) / pit + 0.5))
            for i in range(na):
                for j in range(2):
                    t0, t1 = (i + j / 2) / na, (i + (j + 1) / 2) / na
                    G[seq[k % 2]].append(([A.pt(side, 'a', t0, -bw), A.pt(side, 'a', t0, tuck), A.pt(side, 'a', t1, tuck),
                                           A.pt(side, 'a', t1, -bw)], 0.6))
                CB.append(([A.pt(side, 'a', i / na, -bw), A.pt(side, 'a', i / na, tuck)], False, 0.007))
                k += 1
        F = A.outline(r=-bw, zb=zb1, n=9)
        C.append((F, True, 0.009))
        # field: diamond quarries; a lozenge medallion leaded into the lattice
        half = A.a - bw
        ncol = max(1, int(half / clamp(0.045 + 0.05 * half, 0.05, 0.085) + 0.5))
        hx = half / ncol
        hz = hx * 1.65
        nl = min(ncol, 5 if big else 3)
        zc = zb1 + (A.zs - zb1) * (0.56 if big else 0.6)
        zc = min(zc, A.apex(-bw) - nl * hz * 1.25)
        has_loz = zc - nl * hz > zb1 + hz
        if not has_loz:
            nl = 1
            zc = zb1 + (A.zs - zb1) * 0.5
        par = (nl - 1) % 2
        ztop = A.apex(-bw)
        jlo, jhi = int((zb1 - zc) / hz) - 2, int((ztop - zc) / hz) + 2
        for j in range(jlo, jhi + 1):
            for i in range(-ncol - 1, ncol + 2):
                if (i + j - par) % 2:
                    continue
                if has_loz and abs(i) + abs(j) <= nl - 1:
                    continue
                x, z = A.cx + i * hx, zc + j * hz
                poly = _clip([(x + hx, z), (x, z + hz), (x - hx, z), (x, z - hz)], F)
                if len(poly) < 3 or abs(_area(poly)) < 2e-5:
                    continue
                uu, vv = (i + j - par) // 2, (i - j - par) // 2
                if uu % 3 == 0 and vv % 3 == 0:
                    ti = 3
                elif uu % 3 == 0 or vv % 3 == 0:
                    ti = 1
                else:
                    ti = 0
                if rgl.chance(0.05):                    # the odd pane replaced by the glazier over the years
                    ti = 1 - ti if ti < 2 else 0
                G[ti].append((poly, 1.0))
        loz = [(A.cx + nl * hx, zc), (A.cx, zc + nl * hz), (A.cx - nl * hx, zc), (A.cx, zc - nl * hz)]
        S = ncol + 3 + abs(jlo) + abs(jhi)
        for cc in range(-S, S + 1):
            if (cc - nl) % 2:
                continue
            for fam in (1, -1):
                q0 = (A.cx - S * hx, zc + fam * (cc + S) * hz)
                q1 = (A.cx + S * hx, zc + fam * (cc - S) * hz)
                tt = _seg_clip(q0, q1, F)
                if not tt:
                    continue
                spans = [tt]
                if has_loz and abs(cc) < nl:
                    ti_ = _seg_clip(q0, q1, loz)
                    if ti_:
                        spans = [(tt[0], min(ti_[0], tt[1])), (max(ti_[1], tt[0]), tt[1])]
                for (s0, s1) in spans:
                    if (s1 - s0) * 2 * S * hx < 0.004:
                        continue
                    C.append(([(q0[0] + (q1[0] - q0[0]) * s0, q0[1] + (q1[1] - q0[1]) * s0),
                               (q0[0] + (q1[0] - q0[0]) * s1, q0[1] + (q1[1] - q0[1]) * s1)], False, 0.007))
        if has_loz:
            rin = nl * hx * nl * hz / hypot(nl * hx, nl * hz)
            medallion(loz, (A.cx, zc), rin * 0.6, 2)
            return zc, nl * hz
        return None

    def glaze_disc(h):
        c, r, t = h["c"], h["r"], h["t"]
        if t == "eye":
            # spandrel light: three (or four) leaded pieces round a small boss
            nq = 3 if h["tre"] else 4
            a0 = h["a0"] + pi / nq
            rb_ = 0.3 * r
            for k in range(nq):
                u0, u1 = a0 + 2 * pi * k / nq, a0 + 2 * pi * (k + 1) / nq
                for q in ring_q(c, rb_, r + tuck, u0, u1, 3 if nq == 3 else 2):
                    G[2 if (nq == 3 or k % 2 == 0) else 3].append((q, 0.8))
                C.append(([(c[0] + rb_ * cos(u0), c[1] + rb_ * sin(u0)), (c[0] + (r + tuck) * cos(u0), c[1] + (r + tuck) * sin(u0))],
                          False, 0.006))
            G[4 if nq == 3 else 0].append((_circle(c, rb_, 3 * nq, a0), 0.0))
            C.append((_circle(c, rb_, 3 * nq, a0), True, 0.006))
            return
        if t == "oculus" and r < 0.085:
            for k in range(4):
                for q in ring_q(c, 0.0, r + tuck, pi / 4 + pi / 2 * k, pi / 4 + pi / 2 * (k + 1), 2):
                    G[3 if k % 2 == 0 else 2].append((q[1:], 0.8))
                C.append(([c, (c[0] + (r + tuck) * cos(pi / 4 + pi / 2 * k), c[1] + (r + tuck) * sin(pi / 4 + pi / 2 * k))], False, 0.006))
            return
        if t == "oculus":
            bw = clamp(0.2 * r, 0.02, 0.05) if h["main"] else 0.0
            ri = r - bw
            if bw > 0:
                nseg = 12 if r < 0.3 else 16
                for k in range(nseg):
                    a0, a1 = 2 * pi * k / nseg, 2 * pi * (k + 1) / nseg
                    for q in ring_q(c, ri, r + tuck, a0, a1, 2):
                        G[3 if k % 2 == 0 else 0].append((q, 0.6))
                    C.append(([(c[0] + ri * cos(a0), c[1] + ri * sin(a0)), (c[0] + (r + tuck) * cos(a0), c[1] + (r + tuck) * sin(a0))], False, 0.007))
                C.append((_circle(c, ri, 24), True, 0.008))
                medallion(_circle(c, ri, 24), c, ri * 0.6, 2)
            else:
                # small oculus: a jewel, a ring of four deep pieces and a ring of eight quarries
                r1, r2 = r * 0.26, r * 0.6
                for k in range(8):
                    a0, a1 = 2 * pi * k / 8, 2 * pi * (k + 1) / 8
                    for q in ring_q(c, r2, r + tuck, a0, a1, 2):
                        G[0 if k % 2 == 0 else 1].append((q, 1.0))
                    C.append(([(c[0] + r2 * cos(a0), c[1] + r2 * sin(a0)), (c[0] + (r + tuck) * cos(a0), c[1] + (r + tuck) * sin(a0))],
                              False, 0.007))
                for k in range(4):
                    a0, a1 = pi / 8 + pi / 2 * k, pi / 8 + pi / 2 * (k + 1)
                    for q in ring_q(c, r1, r2, a0, a1, 4):
                        G[2 if k % 2 == 0 else 3].append((q, 0.7))
                    C.append(([(c[0] + r1 * cos(a0), c[1] + r1 * sin(a0)), (c[0] + r2 * cos(a0), c[1] + r2 * sin(a0))], False, 0.007))
                G[4].append((_circle(c, r1, 16, pi / 8), 0.0))
                C.append((_circle(c, r1, 8, pi / 8), True, 0.007))
                C.append((_circle(c, r2, 16), True, 0.007))
            return
        # wheel rose: emblem roundel in the hub; the petals are leaded as kite quarries on spiralling cames,
        # with a jewel on every petal axis and deep-coloured rings against the hub and the outer ring
        npet, rhub, bs = h["npet"], h["rhub"], h["bs"]
        rr, R1 = rhub + tuck, r + tuck
        medallion(_circle(c, rr, 20), c, rhub * 0.74, 2)
        nth = 2 * npet
        if r * 2 * pi / nth > 0.2:
            nth *= 2
        dl = 2 * pi / nth
        K = max(2, int(log(R1 / rr) / (0.62 * dl) + 0.5))
        q_ = (R1 / rr) ** (1.0 / K)
        rk = [rr * q_ ** k for k in range(K + 1)]
        per = 2 * nth // npet
        dcp = (r + bs / 2) / (1 + sin(pi / npet))
        kf = log(dcp / rr) / log(q_)
        kc = int(clamp(2 * round((kf - 1) / 2) + 1, 1, max(1, K - 1)))

        def node(k, j):
            th = pi / 2 + j * dl / 2
            return (c[0] + rk[k] * cos(th), c[1] + rk[k] * sin(th))

        for k in range(K + 1):
            for j in range(2 * nth):
                if (k + j) % 2 == 0:
                    continue
                poly = [node(k, j - 1)] + ([node(k + 1, j)] if k < K else []) + [node(k, j + 1)] + ([node(k - 1, j)] if k > 0 else [])
                if j % per == 0 and k == kc:
                    G[4].append((poly, 0.0))
                elif k <= 1:
                    G[3 if (j // 2) % 2 == 0 else 2].append((poly, 0.7))
                elif k >= K - 1 and k != kc:
                    G[2 if (j // 2) % 2 == 0 else 3].append((poly, 0.7))
                else:
                    G[0 if ((k + j) // 2) % 2 == 0 else 1].append((poly, 1.0))
        for j0 in range(0, 2 * nth, 2):
            for sg in (1, -1):
                C.append(([node(k, j0 + sg * k) for k in range(K + 1)], False, 0.007))

    bars = []

    def bar(A, z):
        half = A.a
        if z > A.zs:
            if z - A.zs >= A.R:
                return
            half = A.a - A.R + sqrt(A.R ** 2 - (z - A.zs) ** 2)
        if half > 0.06:
            bars.append(cylinder("saddle_bar", 0.0065, 2 * half + 0.03, loc=(A.cx, yg - 0.0135, z), rot=(0, 90, 0),
                                 verts=6, cap=False))

    for h in holes:
        if h["t"] == "lancet":
            A = h["A"]
            med = glaze_lancet(A, h["main"])
            # wrought saddle bars the panels are tied to: one under and one over the medallion, the rest spaced between
            ztop = A.zs - 0.03
            if med:
                zlo, zhi = med[0] - med[1] - 0.02, med[0] + med[1] + 0.02
                bar(A, zlo)
                L = zlo - A.z0
                n = max(0, int(L / 0.4 + 0.5) - 1)
                for i in range(n):
                    bar(A, A.z0 + L * (i + 1) / (n + 1))
                if zhi < A.apex() - 0.12:
                    bar(A, zhi)
                    L = ztop - zhi
                    if L > 0.3:
                        n = max(1, int(L / 0.4 + 0.5))
                        for i in range(1, n + 1):
                            bar(A, zhi + L * i / n)
            else:
                n = max(1, int((ztop - A.z0) / 0.4 + 0.5))
                for i in range(1, n + 1):
                    bar(A, A.z0 + (ztop - A.z0) * i / n)
        else:
            glaze_disc(h)

    # glass: every piece tilted a hair out of plane; the patina film over it thickens away from the heart of
    # the window (the lit hall shows brightest a little above mid-height) and varies from quarry to quarry
    zmid = hs + (apex0 - hs) * 0.58
    ov_v, ov_f = [], []
    glass_objs = []
    for i in range(5):
        rt = np.random.RandomState(p.seed * 7 + i)
        verts, faces = [], []
        for (poly, wgt) in G[i]:
            poly = _ccw(poly)
            n = len(poly)
            cx, cz = sum(q[0] for q in poly) / n, sum(q[1] for q in poly) / n
            ext = max(max(abs(q[0] - cx), abs(q[1] - cz)) for q in poly) + 1e-6
            sl = min(0.03, 0.0014 / ext)
            gx_, gz_ = rt.uniform(-sl, sl), rt.uniform(-sl, sl)
            pts = [(x, ypl + gx_ * (x - cx) + gz_ * (z - cz), z) for x, z in poly]
            b = len(verts)
            verts += pts
            faces.append(tuple(range(b, b + n)))
            dz_ = (cz - zmid) / ((apex0 - zmid) if cz > zmid else (zmid - hs))
            d = min(1.0, hypot(0.9 * cx / a, dz_))
            p2 = 0.45 * d * d * wgt
            p1 = min(1.0 - p2, (0.2 + 0.45 * d) * wgt)
            u = rt.uniform()
            for L in range(2 if u < p2 else (1 if u < p2 + p1 else 0)):
                b = len(ov_v)
                # (CLEARANCE RULE: nothing lies parallel within 4 mm of the glowing glass; the film follows each quarry's tilt)
                ov_v += [(x, y - 0.0045 - 0.001 * L, z) for x, y, z in pts]
                ov_f.append(tuple(range(b, b + n)))
        objs = [_drop_back(o, front_only=True) for o in GO[i]]
        if verts:
            objs.append(mesh_object(f"glass{i}", verts, faces))
        for o in objs:
            assign(o, gl[i])
            # the shader draws its own quarry lattice from the UVs; ours is modelled, so give each tint one flat value
            uv_planar(o, (1, 0, 0), (0, 0, 1), scale=0.0004, offset=(0.324, 0.472))
        glass_objs += objs
    cv, cf = [], []
    for (pts, closed, w) in C:
        _came(cv, cf, pts, closed, w, ypl - 0.0075, ypl - 0.001)
        _band(ov_v, ov_f, pts, closed, w + 0.013, ypl - 0.0058)       # dirt gathered against the leads (>= 4.4 mm off the tilted glass)
    for (pts, closed, w) in CB:
        _came(cv, cf, pts, closed, w, ypl - 0.0075, ypl - 0.001)
    came = mesh_object("came", cv, cf)
    assign(came, lead)
    uv_box(came)
    smooth(came, 140)
    film = mesh_object("patina", ov_v, ov_f)
    assign(film, patina)
    uv_planar(film, (1, 0, 0), (0, 0, 1), scale=0.0004, offset=(0.324, 0.472))
    for b_ in bars:
        assign(b_, iron)
        uv_grain(b_, axis='Z')
        paint(b_, special=mask_mul(noise_mask(0.07, seed=p.seed + 5, lo=0.3, hi=0.7), 0.7))

    # ---- weathering: light soot in the reveals, moss on the weathered tops and the damp foot of the jambs,
    # rain streaks where the sill ends and the label stops shed their water
    # (the stone shader turns mossy between mask values of about 0.45 and 0.6, breaking the edge up itself)
    mk = clamp(0.5 + p.moss, 0.0, 1.08) if p.moss > 0.02 else 0.0
    mmid = 0.68 - 0.3 * p.moss                          # noise level above which moss takes hold
    patch = noise_mask(0.17, seed=p.seed + 2, lo=mmid - 0.1, hi=mmid + 0.1)
    fine = noise_mask(0.045, seed=p.seed + 9, lo=0.2, hi=0.8)
    sootn = noise_mask(0.3, seed=p.seed + 1, lo=0.05, hi=0.6)
    stn = noise_mask(1.0, seed=p.seed + 4, lo=0.4, hi=0.66)

    def streak(co, n):
        q = co.copy()
        q[:, 0] = (co[:, 0] + 0.7 * co[:, 1]) / 0.022
        q[:, 1] = 0.0
        q[:, 2] /= 0.6
        return stn(q, n)

    def moss(top_w, low_w, base_w=0.0):
        def fn(co, n):
            top = np.clip((n[:, 2] - 0.12) / 0.43, 0.0, 1.0) * top_w
            low = np.clip(1.35 - (co[:, 2] - hs) / 0.45, 0.0, 1.0) * low_w
            b = np.maximum(np.maximum(top, low), base_w)
            return np.clip(b * (0.38 + 0.32 * patch(co, n)) * (0.92 + 0.08 * fine(co, n)) * mk, 0.0, 1.0)
        return fn

    def soot(co, n):
        up = np.clip((co[:, 2] - zs) / max(H - zs, 0.1), 0.0, 1.0)
        deep = np.clip((co[:, 1] + D) / max(D, 0.05), 0.0, 1.0)      # deeper into the reveal = dirtier
        return np.clip(0.2 * up + 0.36 * deep * (0.45 + 0.55 * up), 0.0, 1.0) * sootn(co, n)

    def label_run(co, n):
        ax = np.abs(co[:, 0])
        near = np.clip((ax - (a + fw - 0.004)) / 0.003, 0.0, 1.0)      # outer flank of the jamb and the quoins
        s_ = streak(co, n)                              # every line of vertices runs its own length
        fall = np.clip(1.0 - (zl_bot - co[:, 2]) / (0.4 + 0.6 * s_), 0.0, 1.0) * (co[:, 2] < zt - 0.02)
        return near * fall ** 0.8 * (0.45 + 0.45 * s_)

    def sill_run(co, n):
        endn = np.clip(1.0 - (W / 2 - np.abs(co[:, 0])) / 0.17, 0.0, 1.0)
        front = (co[:, 1] < -(D - 0.03)) & (co[:, 2] < hs - 0.045)
        under = (co[:, 2] < hs - 0.085) & (co[:, 2] > hs - 0.095)
        return np.clip(front * streak(co, n) * (0.35 + 0.6 * endn) + under * 0.3, 0.0, 1.0)

    def hood_dirt(co, n):
        return 0.22 * np.clip(-n[:, 2], 0.0, 1.0) + 0.12 * sootn(co, n)

    paint(groups["frame"], grime=lambda co, n: np.maximum(soot(co, n), label_run(co, n)), mode="add")
    paint(groups["trac"], grime=soot, mode="add")
    paint(groups["hood"], grime=hood_dirt, mode="add")
    paint(groups["sill"], grime=sill_run, mode="add")
    paint(groups["label"], grime=lambda co, n: 0.45 * streak(co, n) * np.clip((zt - 0.03 - co[:, 2]) / 0.1, 0.0, 1.0), mode="add")
    if mk > 0:
        paint(groups["frame"], special=moss(0.9, 1.0))
        paint(groups["trac"], special=moss(0.5, 0.8))
        paint(groups["hood"], special=moss(1.0, 0.0))
        paint(groups["sill"], special=moss(1.0, 0.0, 0.62))
        paint(groups["label"], special=moss(1.0, 0.0))

    # moss cushions where the water lingers: the sill's corners and ears, the back of the hood
    tufts = []
    if p.moss > 0.05:
        rt = np.random.RandomState(p.seed * 13 + 5)
        ys0, ys1 = -(d0 + 0.004), -(ds - 0.010)
        nsl = (0.0, -0.040, ys0 - ys1)
        xe = a + fw + 0.5 * (hw + xs)
        spots = [((sx * (a - 0.03), ys0 - 0.012, hs - 0.004), nsl) for sx in (1, -1)]
        spots += [((sx * xe, -0.55 * d0, hs), (0.0, 0.0, 1.0)) for sx in (1, -1)]
        if len(lanc) > 1:
            spots.append(((0.0, ys0 - 0.012, hs - 0.004), nsl))
        for _ in range(3):
            f_ = rt.uniform(0.3, 0.75)
            spots.append(((rt.uniform(-a, a), ys0 + (ys1 - ys0) * f_, hs - 0.042 * f_), nsl))
        for i in rt.permutation(len(spots))[:int(p.moss * 6 + 0.5)]:
            c_, n_ = spots[i]
            sc = rt.uniform(0.75, 1.2) * clamp(0.6 + 0.3 * W, 0.8, 1.3)
            if n_[1] == 0.0:                            # on a sill ear: a cushion along the jamb's flank
                tufts.append(_cushion("moss_tuft", c_, n_, (0.0, 1.0, 0.0), min(0.036 * sc, 0.36 * (hw + xs)), 0.02 * sc, rt))
                continue
            tufts.append(_cushion("moss_tuft", c_, n_, (1.0, 0.0, 0.0), 0.042 * sc, 0.02 * sc, rt))
            if rt.uniform() < 0.6:                      # with a smaller one beside it
                dx = (1 if rt.uniform() < 0.5 else -1) * 0.07 * sc
                if abs(c_[0] + dx) < a - 0.03:
                    tufts.append(_cushion("moss_tuft", (c_[0] + dx, c_[1] - 0.012, c_[2] - 0.005), n_, (1.0, 0.0, 0.0),
                                          0.024 * sc, 0.013 * sc, rt, n=6))
        for _ in range(int(p.moss * 4 + 0.5)):
            side = 1 if rt.uniform() < 0.5 else -1
            t = rt.uniform(0.45, 0.97)
            q0, q1 = np.array(ring(side, 'a', t, hsect)), np.array(ring(side, 'a', t - 0.02, hsect))
            f_ = rt.uniform(0.2, 0.6)
            c_ = q0[2] + (q0[3] - q0[2]) * f_
            n_ = np.cross(q1[2] - q0[2], q0[3] - q0[2]) * side
            if n_[2] < 0:
                n_ = -n_
            sc = rt.uniform(0.75, 1.2) * clamp(0.6 + 0.3 * W, 0.8, 1.3)
            if c_[2] + 0.02 * sc > H - 0.004:           # never above the hood apex (outer height is a contract)
                continue
            tufts.append(_cushion("moss_tuft", c_, n_, q1[2] - q0[2], 0.04 * sc, 0.019 * sc, rt))
        for o in tufts:
            paint(o, grime=0.25, wear=0.0, special=rt.uniform(0.8, 1.0), mode="set")
    allst = [o for g in groups.values() for o in g] + tufts

    for o in allst:
        uv_box(o)
        smooth(o, 32)
    assign(join(allst, "stone"), st)

    # ---- the hall's light: it cannot pass the glass as a beam, so coloured lights stand in the room behind the
    # glass (they rake the reveals, mullion flanks, cusps, hood soffit and sill top, never the glass itself)
    # and one small pool hangs under the sill nose for the colour thrown onto the wall foot
    if p.intensity > 0:
        e = p.intensity / 6.0
        yl = 0.34
        if len(lanc) <= 2:
            cols = [(h["A"].cx, h["A"].a, h["A"].zs) for h in lanc]
        else:
            cols = []
            for sx in (-1, 1):
                grp = [h["A"] for h in lanc if h["A"].cx * sx > 0]
                x0, x1 = min(A.cx - A.a for A in grp), max(A.cx + A.a for A in grp)
                cols.append((0.5 * (x0 + x1), 0.5 * (x1 - x0), max(A.zs for A in grp)))

        def reach(q, cap):
            """Lights throw no shadows: keep the undersides of the label stops (outside, in the rain) out of range."""
            d_ = min(sqrt((q[0] - sx * (a + fw)) ** 2 + (q[1] + 0.5 * D) ** 2 + (q[2] - zl_bot) ** 2) for sx in (1, -1))
            return cap if q[2] > zl_bot else clamp(0.94 * d_, 0.6, cap)

        for i, (cx_, hb_, zs_) in enumerate(cols):
            k = clamp(0.55 + 1.6 * hb_, 0.7, 1.5)
            # (the upper one stands level with the label stops, so that it cannot light them from below)
            for nm, q, col_, pw in (("glow_low", (cx_, yl + 0.2, hs + clamp(0.3 * (zl_bot - hs), 0.3 + 0.3 * hb_, 0.8)), c_pool, 4.2),
                                    ("glow_mid", (cx_, yl, max(hs + 0.62 * (zs_ - hs), zl_bot + 0.03)), c_rim, 4.6)):
                light(f"{nm}{i}", q, color=col_, intensity=pw * e * k, range=reach(q, 3.0 + 2.0 * hb_), flicker=0.05)
        zh_ = mains[0]["c"][1] if mains else zs + 0.45 * (apex0 - zs)
        light("glow_head", (0.0, yl, zh_), color=c_head, intensity=3.0 * e * clamp(0.55 + 1.2 * a, 0.7, 1.5),
              range=3.0 + 2.0 * a, flicker=0.05)
        # (low and close to the wall, so that its reflection in the glass stays hidden behind the sill)
        light("glass_spill", (0.0, -(D + 0.13), hs - 0.2), color=c_pool, intensity=0.42 * e * clamp(W / 1.3, 0.7, 1.5),
              range=1.8, flicker=0.05)
    socket("sill", "attach", (0.0, -(D + 0.03), hs), role="prop_top")
