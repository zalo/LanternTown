"""Wall-mounted neon emblem (fleur-de-lis, star, crescent moon, cup, arrow or key) as on the reference cafe.
Each outline is a real hand-bent glass tube: one continuous round rod per shape with proper bend radii and a
little bench wobble, opened at a small gap where both ends turn back (painted with black block-out) into
porcelain electrode housings, carried on porcelain stand-off posts with copper tie wires.  Backing is a
riveted sheet-iron pan (flat sheet, folded rim, rain hood) with the emblem sign-written on it as a dark
silhouette and a painted line under every tube, an open welded angle-iron frame with cross bars, clamp plates
and cranked wall brackets, or nothing (tubes fixed straight to the wall).  A zinc transformer box with ribbed
HT insulators, GTO leads, series jumpers and a supply conduit feeds it.
"""
import math

import bmesh
import numpy as np
from tk import *

META = {"name": "Neon emblem sign", "class": "component", "mount": "wall", "mount_height": 2.7,
        "focus": [0.0, -0.06, 0.0], "tri_budget": 14000}
PARAMS = {
    "glyph": {"default": "fleur", "choices": ["fleur", "star", "moon", "cup", "arrow", "key"], "desc": "emblem drawn in neon"},
    "size": {"default": 1.0, "min": 0.5, "max": 1.8, "desc": "largest dimension of the emblem (m)"},
    "colour": {"default": "#ffc83a", "desc": "main tube colour"},
    "colour2": {"default": "#ff2f3f", "desc": "accent tube colour"},
    "accent": {"default": True, "desc": "second-colour accent tubes (off = single colour outline only)"},
    "backing": {"default": "panel", "choices": ["panel", "frame", "none"], "desc": "sheet-iron pan / open angle-iron frame / tubes straight on the wall"},
    "panel_shape": {"default": "rect", "choices": ["rect", "oval", "octagon"], "desc": "outline of the panel or frame"},
    "panel_paint": {"default": "#2b3833", "desc": "paint colour of the backing panel (chalked dark colours)"},
    "art_paint": {"default": "#8c7040", "desc": "sign-writer's paint: line under the tubes, pinstripe and (darkened) emblem silhouette"},
    "hood": {"default": True, "desc": "sheet-metal rain hood over a rect / octagon panel"},
    "mirror": {"default": False, "desc": "mirror the emblem left-right (arrow direction, moon facing)"},
    "transformer": {"default": "right", "choices": ["right", "left", "none"], "desc": "transformer box below the sign (none = hidden in the wall)"},
    "glow": {"default": 1.2, "min": 0.4, "max": 4.0, "desc": "emissive strength of the tubes"},
    "intensity": {"default": 1.3, "min": 0.4, "max": 5.0, "desc": "light output of the whole sign (cd at 1 m size)"},
    "flicker": {"default": 0.04, "min": 0.0, "max": 0.6, "desc": "light flicker (0 steady, 0.4 = failing transformer)"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: grime, rust and chipped paint"},
}
VARIANTS = {
    "star_frame": {"glyph": "star", "backing": "frame", "panel_shape": "octagon", "size": 0.8, "colour": "#27c4ff",
                   "colour2": "#ff3a98", "transformer": "left", "seed": 3},
    "moon_oval": {"glyph": "moon", "panel_shape": "oval", "size": 0.9, "colour": "#3f9dff", "colour2": "#ffd24a",
                  "panel_paint": "#262e47", "art_paint": "#8f9bb8", "age": 0.25, "seed": 5},
    "cup_bare": {"glyph": "cup", "backing": "none", "size": 0.7, "colour": "#ff8a2e", "colour2": "#dff2ff",
                 "transformer": "left", "seed": 8},
    "key_octagon": {"glyph": "key", "panel_shape": "octagon", "size": 1.25, "colour": "#12d050", "colour2": "#ffc83a",
                    "panel_paint": "#40221d", "art_paint": "#c9b48a", "age": 0.85, "intensity": 0.7, "seed": 11},
    "arrow_frame": {"glyph": "arrow", "backing": "frame", "size": 1.4, "colour": "#ff3b45", "colour2": "#ffd24a",
                    "mirror": True, "flicker": 0.3, "seed": 14},
}


# --- 2D glyph maths (pure python) -----------------------------------------------------------------
def _d(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(len(a))))


def _lp(a, b, t):
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(len(a)))


def _unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def _cr(pts, step):
    """Catmull-Rom through pts (open), sampled every ~step."""
    if len(pts) < 3:
        return [tuple(v) for v in pts]
    n = len(pts)
    P = [_lp(pts[1], pts[0], 2.0)] + [tuple(v) for v in pts] + [_lp(pts[-2], pts[-1], 2.0)]
    out = [tuple(pts[0])]
    for i in range(n - 1):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        k = max(2, int(math.ceil(_d(p1, p2) / step)))
        for j in range(1, k + 1):
            t = j / k
            out.append(tuple(0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t * t
                                    + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t ** 3) for c in range(len(p1))))
    return out


def _arc(cx, cz, r, a0, a1, n=None):
    n = n or max(4, int(abs(a1 - a0) / 5))
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cz + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def _rot(pts, deg, c=(0.0, 0.0)):
    ca, sa = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(c[0] + (x - c[0]) * ca - (z - c[1]) * sa, c[1] + (x - c[0]) * sa + (z - c[1]) * ca) for x, z in pts]


def _cum(pts):
    c = [0.0]
    for i in range(1, len(pts)):
        c.append(c[-1] + _d(pts[i - 1], pts[i]))
    return c


def _at(pts, cum, s):
    s = min(max(s, 0.0), cum[-1])
    lo, hi = 0, len(cum) - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if cum[m] <= s:
            lo = m
        else:
            hi = m
    seg = cum[hi] - cum[lo]
    return _lp(pts[lo], pts[hi], (s - cum[lo]) / seg if seg > 1e-12 else 0.0)


def _sub(pts, cum, a, b):
    out = [_at(pts, cum, a)]
    for i, c in enumerate(cum):
        if a + 1e-7 < c < b - 1e-7:
            out.append(tuple(pts[i]))
    out.append(_at(pts, cum, b))
    return out


def _assemble(segs, step):
    """segs: [(kind, pts)], kind 'c' = smooth curve, 'l' = polyline with corners, 'a' = dense smooth points.
    -> (pts, cum, corner arc positions, arc position of the electrode gap)."""
    pts, idx, gap_s = [], [], 0.0
    for k, (kind, sp) in enumerate(segs):
        q = _cr(sp, step) if kind == "c" else [tuple(v) for v in sp]
        if pts:
            base = len(pts) - 1
            idx.append(base)
            pts.extend(q[1:])
            if kind == "l":
                idx.extend(range(base + 1, len(pts) - 1))
        else:
            pts.extend(q)
            if kind == "l":
                idx.extend(range(1, len(pts) - 1))
            gap_s = 0.5 * (_d(q[0], q[1]) if kind == "l" else _cum(q)[-1])
    cum = _cum(pts)
    return pts, cum, [cum[i] for i in idx], gap_s


def _open_loop(pts, cum, corners, gap_s, gap):
    """Closed loop -> open tube path with a gap centred on gap_s; corners re-based to the new start."""
    L = cum[-1]
    a = gap_s + gap / 2
    out = _sub(pts, cum, a, L) + _sub(pts, cum, 0.0, gap_s - gap / 2)[1:]
    cs = sorted(((c - a) % L) for c in list(corners) + [0.0])
    return out, [c for c in cs if c < L - gap - 1e-6]


def _fillet(pts, corners, R):
    """Replace every corner (arc positions) of an open polyline by a true circular bend of radius ~R.  Where two
    bends compete for a short leg the sharper one (longer tangent) gets what the other does not need."""
    cum = _cum(pts)
    L = cum[-1]
    cs = sorted(c for c in corners if 1e-5 < c < L - 1e-5)
    want = []
    for c in cs:
        C = _at(pts, cum, c)
        e = min(0.003, c * 0.5, (L - c) * 0.5)
        u = _unit([a - b for a, b in zip(_at(pts, cum, c - e), C)])
        v = _unit([a - b for a, b in zip(_at(pts, cum, c + e), C)])
        phi = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v)))))
        want.append(R / math.tan(phi / 2) if phi < math.radians(166) else 0.0)
    out, s = [], 0.0
    for i, c in enumerate(cs):
        if i:
            d = c - cs[i - 1]
            before = max(0.48 * d, 0.96 * d - want[i - 1])
        else:
            before = c * 0.9
        if i + 1 < len(cs):
            d = cs[i + 1] - c
            after = max(0.48 * d, 0.96 * d - want[i + 1])
        else:
            after = (L - c) * 0.9
        T = min(want[i], before, after)
        if T < 1e-4:
            continue
        C = _at(pts, cum, c)
        A, B = _at(pts, cum, c - T), _at(pts, cum, c + T)
        u = [a - b for a, b in zip(A, C)]
        v = [a - b for a, b in zip(B, C)]
        ta = min(math.sqrt(sum(a * a for a in u)), math.sqrt(sum(a * a for a in v)))
        u, v = _unit(u), _unit(v)
        phi = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v)))))
        if phi > math.radians(172):
            continue
        r = ta * math.tan(phi / 2)
        w = _unit([a + b for a, b in zip(u, v)])
        O = [cc + ww * ta / math.cos(phi / 2) for cc, ww in zip(C, w)]
        e1 = _unit([cc + uu * ta - oo for cc, uu, oo in zip(C, u, O)])
        b2 = [cc + vv * ta - oo for cc, vv, oo in zip(C, v, O)]
        dt = sum(a * b for a, b in zip(b2, e1))
        e2 = _unit([a - b * dt for a, b in zip(b2, e1)])
        sw = math.pi - phi
        n = max(3, int(math.ceil(sw / math.radians(15))))
        out += _sub(pts, cum, s, c - T)
        for j in range(1, n):
            a = sw * j / n
            out.append(tuple(oo + r * (math.cos(a) * x1 + math.sin(a) * x2) for oo, x1, x2 in zip(O, e1, e2)))
        s = c + T
    out += _sub(pts, cum, s, L)
    return out


def _dp(pts, tol):
    """Douglas-Peucker: drop points that deviate less than tol from the simplified polyline."""
    n = len(pts)
    if n < 3:
        return list(pts)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    dim = range(len(pts[0]))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        pa, pb = pts[a], pts[b]
        ab = [pb[k] - pa[k] for k in dim]
        l2 = sum(c * c for c in ab)
        best, bi = -1.0, -1
        for i in range(a + 1, b):
            ap = [pts[i][k] - pa[k] for k in dim]
            if l2 > 1e-18:
                t = max(0.0, min(1.0, sum(x * y for x, y in zip(ap, ab)) / l2))
                ap = [ap[k] - ab[k] * t for k in dim]
            d2 = sum(c * c for c in ap)
            if d2 > best:
                best, bi = d2, i
        if best > tol * tol:
            keep[bi] = True
            stack += [(a, bi), (bi, b)]
    return [q for q, k in zip(pts, keep) if k]


def _area(poly):
    return 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))


def _offset(poly, d):
    """Grow a closed polygon (no repeated last point) outwards by d (mitred, limited)."""
    n = len(poly)
    sg = 1.0 if _area(poly) > 0 else -1.0
    out = []
    for i in range(n):
        p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % n]
        e1 = _unit((p1[0] - p0[0], p1[1] - p0[1]))
        e2 = _unit((p2[0] - p1[0], p2[1] - p1[1]))
        n1, n2 = (e1[1], -e1[0]), (e2[1], -e2[0])
        m = (n1[0] + n2[0], n1[1] + n2[1])
        ml = math.hypot(m[0], m[1])
        if ml < 1e-6:
            m, k = n1, 1.0
        else:
            m = (m[0] / ml, m[1] / ml)
            k = 1.0 / max(m[0] * n1[0] + m[1] * n1[1], 0.4)
        out.append((p1[0] + m[0] * d * k * sg, p1[1] + m[1] * d * k * sg))
    return out


def _chord(poly, v, axis=1):
    """(min, max) along the other axis where the closed polygon crosses coordinate v on `axis`."""
    hits = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[axis] - v) * (b[axis] - v) <= 0 and a[axis] != b[axis]:
            t = (v - a[axis]) / (b[axis] - a[axis])
            hits.append(a[1 - axis] + (b[1 - axis] - a[1 - axis]) * t)
    return (min(hits), max(hits)) if hits else (0.0, 0.0)


def _cross(path, cum, v, axis=1):
    out = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if (a[axis] - v) * (b[axis] - v) < 0:
            out.append(cum[i] + (cum[i + 1] - cum[i]) * (v - a[axis]) / (b[axis] - a[axis]))
    return out


def _mx(pts):
    return [(-x, z) for x, z in pts]


def _mxr(pts):
    return [(-x, z) for x, z in reversed(pts)]


def _star(cx, cz, R, r, rot=-90.0, n=5):
    pts = []
    for k in range(2 * n):
        a = math.radians(rot + k * 180.0 / n)
        rr = r if k % 2 == 0 else R
        pts.append((cx + rr * math.cos(a), cz + rr * math.sin(a)))
    return pts + [pts[0]]


def _glyph(name, accent):
    """Emblem as a list of tube shapes in unit coordinates (x right, z up).  The first edge / segment of a
    closed shape carries the electrode gap.  col 1 = accent colour, opt = dropped when accent is off."""
    out = []

    def shape(segs, col=0, closed=True, fill=None, opt=False, hole=None):
        if opt and not accent:
            return
        out.append({"segs": segs, "col": col if accent else 0, "closed": closed,
                    "fill": closed if fill is None else fill, "hole": hole})

    if name == "fleur":
        cen = [(0.04, -0.035), (0.072, 0.05), (0.12, 0.17), (0.115, 0.28), (0.062, 0.40), (0.0, 0.5)]
        shape([("l", [(-0.04, -0.035), (0.04, -0.035)]), ("c", cen), ("c", _mxr(cen))])
        outer = [(0.085, -0.035), (0.12, 0.06), (0.17, 0.165), (0.245, 0.255), (0.325, 0.25), (0.385, 0.17), (0.39, 0.075),
                 (0.365, 0.01), (0.325, -0.03)]
        inner = [(0.325, -0.03), (0.322, 0.045), (0.30, 0.10), (0.265, 0.12), (0.235, 0.095), (0.21, 0.04), (0.19, -0.035)]
        base = [(0.19, -0.035), (0.085, -0.035)]
        shape([("l", base), ("c", outer), ("c", inner)])
        shape([("l", _mx(base)), ("c", _mx(outer)), ("c", _mx(inner))])
        shape([("l", [(-0.2, -0.135), (0.2, -0.135), (0.2, -0.065), (-0.2, -0.065), (-0.2, -0.135)])], col=1)
        tail = [(0.16, -0.165), (0.19, -0.23), (0.245, -0.30), (0.262, -0.35)]
        under = [(0.262, -0.35), (0.19, -0.315), (0.12, -0.28), (0.078, -0.30), (0.05, -0.38), (0.0, -0.5)]
        shape([("l", [(-0.16, -0.165), (0.16, -0.165)]), ("c", tail), ("c", under), ("c", _mxr(under)), ("c", _mxr(tail))])
        vein = [(0.14, 0.0), (0.168, 0.06), (0.207, 0.135), (0.258, 0.188), (0.315, 0.182), (0.352, 0.12)]
        shape([("l", [(0.0, 0.03), (0.0, 0.37)])], col=1, closed=False, opt=True)
        shape([("c", vein)], col=1, closed=False, opt=True)
        shape([("c", _mx(vein))], col=1, closed=False, opt=True)
    elif name == "star":
        shape([("l", _star(0, 0, 0.5, 0.215))])
        shape([("l", _star(0, 0, 0.26, 0.112))], col=1, fill=False, opt=True)
    elif name == "moon":
        tilt = 28.0
        a_out = math.degrees(math.atan2(0.4115, 0.284))
        a_in = math.degrees(math.atan2(0.4115, 0.084))
        shape([("a", _rot(_arc(0, 0, 0.5, 222, 360 - a_out), tilt)),
               ("a", _rot(_arc(0.2, 0, 0.42, 360 - a_in, a_in), tilt)),
               ("a", _rot(_arc(0, 0, 0.5, a_out, 222), tilt))])
        c = _rot([(0.17, 0.0)], tilt)[0]
        shape([("l", _star(c[0], c[1], 0.18, 0.088))], col=1, fill=False)
    elif name == "cup":
        side = [(0.10, -0.24), (0.20, -0.16), (0.255, -0.02), (0.27, 0.12)]
        shape([("l", [(-0.10, -0.24), (0.10, -0.24)]), ("c", side), ("l", [(0.27, 0.12), (-0.27, 0.12)]), ("c", _mxr(side))])
        shape([("c", [(0.297, 0.055), (0.37, 0.06), (0.415, 0.0), (0.385, -0.08), (0.31, -0.115), (0.262, -0.112)])],
              closed=False)
        shape([("c", [(-0.37, -0.275), (-0.2, -0.31), (0.0, -0.32), (0.2, -0.31), (0.37, -0.275)])], closed=False)
        for x0, top in ((-0.14, 0.44), (0.0, 0.5), (0.14, 0.44)):
            h = top - 0.19
            shape([("c", [(x0, 0.19), (x0 + 0.035, 0.19 + h * 0.25), (x0 - 0.035, 0.19 + h * 0.5), (x0 + 0.035, 0.19 + h * 0.75),
                          (x0, top)])], col=1, closed=False)
    elif name == "arrow":
        shape([("l", [(-0.5, -0.11), (0.08, -0.11), (0.08, -0.27), (0.5, 0.0), (0.08, 0.27), (0.08, 0.11), (-0.5, 0.11),
                      (-0.42, 0.0), (-0.5, -0.11)])])
        shape([("l", [(-0.33, 0.0), (0.12, 0.0)])], col=1, closed=False, opt=True)
        shape([("l", [(0.17, -0.12), (0.36, 0.0), (0.17, 0.12), (0.17, -0.12)])], col=1, fill=False, opt=True)
    else:  # key
        a = math.degrees(math.asin(0.035 / 0.17))
        x0 = -0.33 + 0.17 * math.cos(math.radians(a))
        hole = _arc(-0.33, 0, 0.085, 270, 630, 40)
        shape([("l", [(x0, -0.035), (0.25, -0.035), (0.25, -0.175), (0.325, -0.175), (0.325, -0.095), (0.395, -0.095),
                      (0.395, -0.175), (0.47, -0.175), (0.47, 0.035), (x0, 0.035)]),
               ("a", _arc(-0.33, 0, 0.17, a, 360 - a))], hole=hole[:-1])
        shape([("a", hole)], col=1, fill=False)
    return out


def _layout(shapes, S, mirror):
    """Normalise so the largest dimension is S metres, centred on the origin."""
    xs, zs = [], []
    for sh in shapes:
        for kind, sp in sh["segs"]:
            q = _cr(sp, 0.01) if kind == "c" else sp
            xs += [v[0] for v in q]
            zs += [v[1] for v in q]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    k = S / max(max(xs) - min(xs), max(zs) - min(zs))
    sx = -k if mirror else k

    def f(v):
        return ((v[0] - cx) * sx, (v[1] - cz) * k)

    out = []
    for sh in shapes:
        d = dict(sh)
        d["segs"] = [(kind, [f(v) for v in sp]) for kind, sp in sh["segs"]]
        d["hole"] = [f(v) for v in sh["hole"]] if sh.get("hole") else None
        out.append(d)
    return out, (max(xs) - min(xs)) * k, (max(zs) - min(zs)) * k




def _resample_max(poly, closed, h):
    """Insert points so no segment is longer than h (closed: no repeated last point)."""
    out, n = [], len(poly)
    for i in range(n if closed else n - 1):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(1, int(math.ceil(_d(a, b) / h)))
        out += [_lp(a, b, j / k) for j in range(k)]
    if not closed:
        out.append(tuple(poly[-1]))
    return out


def _wobble(path, rng, amp, S):
    """Hand-bent glass: nothing is dead straight or exactly on the pattern.  -> [(x, z, dy)]"""
    q = _resample_max(path, False, 0.022)
    cum = _cum(q)
    l1, l2, l3 = rng.u(0.24, 0.42) * S ** 0.5, rng.u(0.09, 0.15) * S ** 0.5, rng.u(0.3, 0.6)
    f1, f2, f3 = rng.u(0, 6.283), rng.u(0, 6.283), rng.u(0, 6.283)
    ang = math.radians(rng.u(-0.45, 0.45))
    ox, oz = rng.u(-1, 1) * 0.0015 * S, rng.u(-1, 1) * 0.0015 * S
    cx, cz = sum(v[0] for v in q) / len(q), sum(v[1] for v in q) / len(q)
    ca, sa = math.cos(ang), math.sin(ang)
    out = []
    for i, (x, z) in enumerate(q):
        a, b = q[max(i - 1, 0)], q[min(i + 1, len(q) - 1)]
        t = _unit((b[0] - a[0], b[1] - a[1]))
        s = cum[i]
        w = amp * (0.65 * math.sin(6.283 * s / l1 + f1) + 0.35 * math.sin(6.283 * s / l2 + f2))
        x, z = x - t[1] * w - cx, z + t[0] * w - cz
        out.append((cx + x * ca - z * sa + ox, cz + x * sa + z * ca + oz, 0.5 * amp * math.sin(6.283 * s / l3 + f3)))
    return out


def _tubes(shapes, r_t, rng, S, gap=0.03):
    """Tube centre-lines: open paths with glass-bender's bend radii (2 x tube diameter) and slight wobble."""
    for k, sh in enumerate(shapes):
        pts, cum, corners, gap_s = _assemble(sh["segs"], 0.003)
        if sh["closed"]:
            path, cs = _open_loop(pts, cum, corners, gap_s, min(gap, 1.2 * gap_s))
        else:
            path, cs = pts, corners
        sh["outline"] = pts
        sh["path"] = _wobble(_fillet(path, cs, 4.0 * r_t), rng.fork(f"tube{k}"), 0.002 + 0.002 * S, S)
        sh["pcum"] = _cum(sh["path"])
    return shapes
# --- end glyph maths --------------------------------------------------------------------------------


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _xz(pts, y):
    return [(q[0], y, q[1]) for q in pts]


def _flat_mesh(name, verts2, faces, y):
    """Flat sign-written paint layer / sheet facing -Y from 2D (x, z) verts; faces are wound to face the viewer."""
    fs = []
    for f in faces:
        fs.append(list(f) if _area([verts2[i] for i in f]) > 0 else list(reversed(f)))
    return mesh_object(name, _xz(verts2, y), fs)


def _ribbon(name, pts, w_out, w_in, closed, y):
    """Painted line of constant width around a 2D polyline.  -> (object, outer loop, inner loop)."""
    n = len(pts)
    sg = (1.0 if _area(pts) > 0 else -1.0) if closed else 1.0
    A, B = [], []
    for i in range(n):
        p1 = pts[i]
        p0 = pts[i - 1] if (closed or i > 0) else None
        p2 = pts[(i + 1) % n] if (closed or i < n - 1) else None
        e1 = _unit((p1[0] - p0[0], p1[1] - p0[1])) if p0 else None
        e2 = _unit((p2[0] - p1[0], p2[1] - p1[1])) if p2 else e1
        e1 = e1 or e2
        n1, n2 = (e1[1], -e1[0]), (e2[1], -e2[0])
        m = (n1[0] + n2[0], n1[1] + n2[1])
        ml = math.hypot(m[0], m[1])
        if ml < 1e-6:
            m, k = n1, 1.0
        else:
            m = (m[0] / ml, m[1] / ml)
            k = 1.0 / max(m[0] * n1[0] + m[1] * n1[1], 0.42)
        A.append((p1[0] + m[0] * w_out * k * sg, p1[1] + m[1] * w_out * k * sg))
        B.append((p1[0] - m[0] * w_in * k * sg, p1[1] - m[1] * w_in * k * sg))
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n if closed else n - 1)]
    return _flat_mesh(name, A + B, faces, y), A, B


def _loft(name, loop, prof, smooth_angle=30.0):
    """Folded / rolled section run around a closed loop: prof = [(inset, y), ...] from the outer back edge over the
    front to the inside; every ring is the loop offset inwards by `inset` at depth y (evenly segmented quads)."""
    if _area(loop) < 0:
        loop = list(reversed(loop))
    n = len(loop)
    verts, faces = [], []
    for inset, y in prof:
        verts += _xz(_offset(loop, -inset) if inset else loop, y)
    for k in range(len(prof) - 1):
        a, b = k * n, (k + 1) * n
        faces += [(a + i, a + (i + 1) % n, b + (i + 1) % n, b + i) for i in range(n)]
    o = mesh_object(name, verts, faces)
    smooth(o, smooth_angle)
    return o


def _grid_face(name, outer, holes, y, dx, dz, extra_z=()):
    """Flat plate facing -Y cut into a regular dx x dz lattice so per-vertex AO and weathering masks have even
    resolution (a fan-triangulated n-gon shades as diamonds)."""
    o = shape_extrude(name, outer, holes, depth=0.001, loc=(0, 0, 0))
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z < 0.5], context="FACES")
    try:
        bmesh.ops.dissolve_limit(bm, angle_limit=0.01, verts=bm.verts[:], edges=bm.edges[:])
    except Exception as e:                                    # keep the triangles
        print("DBG dissolve failed", e)
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    cuts = [((k * dx, 0, 0), (1, 0, 0)) for k in range(int(math.floor(min(xs) / dx)) + 1, int(math.ceil(max(xs) / dx)))]
    cuts += [((0, k * dz, 0), (0, 1, 0)) for k in range(int(math.floor(min(ys) / dz)) + 1, int(math.ceil(max(ys) / dz)))]
    cuts += [((0, v, 0), (0, 1, 0)) for v in extra_z if min(ys) < v < max(ys)]
    for co, no in cuts:
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-6, plane_co=co, plane_no=no)
    for v in bm.verts:
        v.co.z = 0.0
    bm.to_mesh(o.data)
    bm.free()
    set_loc_rot(o, (0, y, 0), (90, 0, 0))
    return o


def _front_only(o):
    """Keep only the front (local +Z) face of a thin sign-written paint layer."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bm.normal_update()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z < 0.5], context="FACES")
    bm.to_mesh(o.data)
    bm.free()
    return o


def _seg_dist(x, z, poly):
    """Distance of numpy points (x, z) to a closed 2D polyline."""
    d = np.full(len(x), 1e9)
    n = len(poly)
    for i in range(n):
        ax, az = poly[i]
        bx, bz = poly[(i + 1) % n]
        ex, ez = bx - ax, bz - az
        l2 = ex * ex + ez * ez
        if l2 < 1e-12:
            continue
        t = np.clip(((x - ax) * ex + (z - az) * ez) / l2, 0.0, 1.0)
        d = np.minimum(d, np.hypot(x - ax - t * ex, z - az - t * ez))
    return d


def build(p, rng):
    S = float(p.size)
    age = float(p.age)
    seed = int(p.seed)
    r_t = 0.0045 + 0.002 * S                      # tube radius: 11 - 16 mm glass
    shapes, gw, gh = _layout(_glyph(p.glyph, bool(p.accent)), S, bool(p.mirror))
    _tubes(shapes, r_t, rng.fork("bend"), S)

    g_t = float(p.glow)
    neon = [mat("emissive", p.colour, glow=g_t), mat("emissive", p.colour2, glow=g_t)]
    china = mat("ceramic_glazed", "#e2d9c6")
    black = mat("plain", "#0b0b0c")
    zinc = mat("zinc")
    iron = mat("iron_wrought")
    brass = mat("brass")

    # ---- backing outline -------------------------------------------------------------------------
    backing = p.backing
    m = 0.07 + 0.06 * S
    W, H = gw + 2 * m, gh + 2 * m
    if p.panel_shape == "oval":
        allp = [v for sh in shapes for v in sh["outline"]]
        k = max(math.hypot(x / (gw / 2), z / (gh / 2)) for x, z in allp)
        W, H = k * gw + 1.7 * m, k * gh + 1.7 * m
    cut = 0.16 * min(W, H)
    r_c = 0.045 if backing == "panel" else 0.008  # pan corners are rolled, the angle frame is mitred

    def outline(i=0.0):
        if p.panel_shape == "oval":
            return [((W / 2 - i) * math.cos(2 * math.pi * k / 56), (H / 2 - i) * math.sin(2 * math.pi * k / 56)) for k in range(56)]
        a, b = W / 2 - i, H / 2 - i
        if p.panel_shape == "octagon":
            c = max(cut - i * 0.586, 0.012)
            return [(a, -b + c), (a, b - c), (a - c, b), (-a + c, b), (-a, b - c), (-a, -b + c), (-a + c, -b), (a - c, -b)]
        return rounded_rect(2 * a, 2 * b, max(r_c - i, 0.004), 5 if backing == "panel" else 2)

    def ring_pts(poly, n, y):
        """n evenly spaced points on a closed polygon, lifted to 3D at depth y."""
        loop = list(poly) + [poly[0]]
        cum = _cum(loop)
        return [(q[0], y, q[1]) for q in (_at(loop, cum, cum[-1] * (k + 0.37) / n) for k in range(n))]

    if backing == "panel":
        y_s = -0.023                              # front face of the sheet
    elif backing == "frame":
        y_s = -0.073                              # front of the clamp plates on the cross bars
    else:
        y_s = 0.0                                 # the wall itself
        W, H = gw, gh
    H_s = 0.036 + 0.008 * S                       # stand-off of the tube axis from the backing
    y_t = y_s - H_s
    bottom = -H / 2 if backing != "none" else -gh / 2 - 0.04

    # ---- open frame: where do the cross bars go?  Across the short span, where they catch the most tube ----
    bars, ax = [], 1
    if backing == "frame":
        ax = 0 if W > H * 1.08 else 1             # 0: upright bars at x = v, 1: level bars at z = v
        ext = gw if ax == 0 else gh

        def score(v):
            sc = 0.0
            for sh in shapes:
                n = len([s for s in _cross(sh["path"], sh["pcum"], v, ax) if 0.04 < s < sh["pcum"][-1] - 0.04])
                sc += min(n, 2) + 0.25 * min(max(n - 2, 0), 2)
            return sc

        for tgt in ((-0.2, 0.2) if ext < 0.8 else (-0.3, 0.0, 0.3)):
            bars.append(max(((tgt - 0.1 + 0.2 * k / 16) * ext for k in range(17)), key=score))

    # ---- tubes, electrode housings, stand-off supports ---------------------------------------------
    total_len = sum(sh["pcum"][-1] for sh in shapes)
    sides = 8 if total_len < 4.2 else 6
    back = H_s - 0.012
    house_m, post_m, ring_m, clamp_m, tabs, posts = [], [], [], [], [], []
    srng = rng.fork("supports")

    def support(sh, s):
        path, cum = sh["path"], sh["pcum"]
        x, z, dy = _at(path, cum, s)
        a, b = _at(path, cum, s - 0.004), _at(path, cum, s + 0.004)
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        post_m.append(trs((x, y_s, z), (90, 0, 0)))
        ring_m.append(trs((x, y_t + dy, z), (0, 90 - ang, 0)))
        posts.append((x, z))
        return x, z

    for i, sh in enumerate(shapes):
        path, cum = sh["path"], sh["pcum"]
        L = cum[-1]
        p3 = ([(path[0][0], y_t + back, path[0][1])] + [(x, y_t + dy, z) for x, z, dy in path]
              + [(path[-1][0], y_t + back, path[-1][1])])
        c3 = _cum(p3)
        p3 = _dp(_fillet(p3, [c3[1], c3[-2]], min(0.013, back * 0.6)), 0.0009 * S ** 0.8)
        tube = sweep(f"tube{i}", p3, r_t, sides=sides, smooth_path=False)
        assign(tube, neon[sh["col"]])
        assign_faces(tube, black, lambda c, n: c.y > y_t + 0.0075)      # block-out paint on the returns
        for q in (path[0], path[-1]):
            house_m.append(trs((q[0], y_s, q[1]), (90, 0, 0)))
        if backing == "frame":
            ss = sorted(s for v in bars for s in _cross(path, cum, v, ax) if 0.03 < s < L - 0.03)
            keep = []
            for s in ss:
                if not keep or s - keep[-1] > 0.085:
                    keep.append(s)
            for s in keep:
                x, z = support(sh, s)
                clamp_m.append(trs((x, -0.0715, z)))
            if len(keep) < 2:                     # short tube that misses the bars: welded outrigger tabs
                for f in ((0.3, 0.7) if L > 0.18 else (0.5,)):
                    x, z = support(sh, L * f)
                    q = (x, z)[ax]
                    v = min(bars, key=lambda b: abs(b - q))
                    clamp_m.append(trs((x, -0.0715, z)))
                    if abs(v - q) > 0.012:
                        ln = abs(v - q) + 0.024
                        if ax == 0:
                            tabs.append(box("tab", (ln, 0.005, 0.02), loc=((x + v) / 2, -0.0675, z), bevel=0.001))
                        else:
                            tabs.append(box("tab", (0.02, 0.005, ln), loc=(x, -0.0675, (z + v) / 2), bevel=0.001))
        else:
            n_s = max(2, int(round(L / (0.12 + 0.12 * S))))
            for j in range(n_s):                  # the fitter spaces them by eye
                support(sh, 0.035 + (L - 0.07) * (j + 0.5 + srng.u(-0.2, 0.2)) / n_s)

    hp = [(0.0112, 0.0), (0.0112, 0.003), (0.0086, 0.005), (0.0086, 0.019), (r_t + 0.001, 0.0225)]
    house = lathe("housing", ([(0.0, 0.0)] if backing == "frame" else []) + hp, segments=6)
    assign(instances_to_mesh("housings", house, house_m, keep_base=False), china)
    post = lathe("post", [(0.0095, 0.0), (0.0042, 0.004), (0.004, H_s - r_t - 0.005), (0.0064, H_s - r_t * 0.7)],
                 segments=5)
    assign(instances_to_mesh("posts", post, post_m, keep_base=False), china)
    tie = torus("tie", major=r_t + 0.0009, minor=0.0011, major_seg=4, minor_seg=3)
    assign(instances_to_mesh("ties", tie, ring_m, keep_base=False), brass)

    # ---- panel: sheet-iron pan, folded rim, rivets, sign-written emblem + pinstripe -----------------
    if backing == "panel":
        # flat oil paint gone chalky on sheet iron: the matte slate-board material tinted to the paint colour
        # (iron_painted is sand-cast orange peel under gloss - wrong for a flat sheet)
        lib, mk = "chalkboard", dict(rough=1.2)
        al = hex_rgb(p.art_paint)

        def tone(c, k, desat):
            g = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            return tuple((v + (g - v) * desat) * k for v in c)

        panel_m = mat(lib, p.panel_paint, **mk)
        line_m = mat(lib, tone(al, 0.5, 0.1), **mk)       # the painted line under the tubes, pinstripe
        fill_m = mat(lib, tone(al, 0.1, 0.4), **mk)      # the emblem silhouette, a deep shade of it

        o0 = outline(0.0)
        ob = outline(0.003)
        dxg, dzg = 0.028, 0.1
        zb = -H / 2 + 0.003
        sheet = _grid_face("sheet", ob, (), y_s, dxg, dzg, extra_z=(zb + 0.02, zb + 0.05))
        nb = list(range(len(ob)))
        sheet_back = mesh_object("sheet_back", _xz(ob, y_s + 0.003), [nb[::-1] if _area(ob) > 0 else nb])
        rim = _loft("rim", _resample_max(o0, True, 0.07),
                    [(0.0, -0.017), (0.0, -0.046), (0.001, -0.047), (0.004, -0.047), (0.005, -0.046), (0.005, y_s)])
        for o in (sheet, sheet_back):
            assign(o, panel_m)
        assign(rim, iron)
        uv_box(sheet, rot90=True)

        # where the paint has failed: at fixings (rivets, stand-off posts) and a few knocks; rust runs down from them
        st = rng.fork("stains")

        def snap(v, d):
            return round(v / d) * d

        n_riv = max(8, int(round(_cum(o0 + [o0[0]])[-1] / 0.14)))
        riv_pts = ring_pts(outline(0.018), n_riv, y_s)
        streaks, chips = [], []                   # (x, z, half width, length, strength) / (x, z, radius)
        for q in riv_pts:
            if st.chance(0.2 + 0.6 * age):
                streaks.append((snap(q[0], dxg), q[2], dxg * 1.25, st.u(0.05, 0.1) + st.u(0.1, 0.35) * age * H * 0.5, 0.4 + 0.6 * age))
        for x, z in posts:
            if st.chance(0.25 + 0.7 * age):
                chips.append((x, z, st.u(0.02, 0.026) + 0.006 * age))
            if st.chance(0.1 + 0.4 * age):
                streaks.append((snap(x, dxg), z, dxg * 1.2, st.u(0.04, 0.16), 0.3 + 0.4 * age))
        for _ in range(1 + int(round(4 * age))):
            x, z = snap(st.u(-0.45, 0.45) * gw, 0.035), snap(st.u(-0.45, 0.45) * gh, 0.035)
            r = st.u(0.02, 0.036)
            chips.append((x, z, r))
            streaks.append((snap(x, dxg), z, dxg * 1.25, st.u(0.08, 0.22), 0.5))

        def rust(co, n):
            x, z = co[:, 0], co[:, 2]
            out = np.zeros(len(x))
            for sx_, sz_, w, ln, k in streaks:
                dz = sz_ - z
                fall = np.where(dz >= 0, np.clip(1 - dz / ln, 0, 1) ** 1.5, np.exp(-(dz / (0.5 * w)) ** 2))
                out = np.maximum(out, k * np.clip(1 - np.abs(x - sx_) / w, 0, 1) * fall)
            for cx_, cz_, r in chips:
                out = np.maximum(out, 0.5 * (1 - _ss(0.8 * r, 1.6 * r, np.hypot(x - cx_, z - cz_))))
            return out

        nz_b = noise_mask(0.09, seed=seed + 2, lo=0.25, hi=0.75)

        def rust_sheet(co, n):                    # + water standing in the bottom fold of the pan
            x, z = co[:, 0], co[:, 2]
            band = (1 - _ss(0.0, 0.035 + 0.07 * age, _seg_dist(x, z, ob))) * _ss(0.0, -0.35 * H, z)
            return np.maximum(rust(co, n), band * (0.45 + 0.55 * nz_b(co, n)) * (0.5 + 0.6 * age))

        def chipw(co, n):
            x, z = co[:, 0], co[:, 2]
            out = np.zeros(len(x))
            for cx_, cz_, r in chips:
                out = np.maximum(out, 1 - _ss(0.45 * r, r, np.hypot(x - cx_, z - cz_)))
            return out

        flake = noise_mask(0.13, seed=seed + 3, lo=0.74 - 0.22 * age, hi=0.86 - 0.22 * age)
        flake2 = noise_mask(0.2, seed=seed + 6, lo=0.72 - 0.2 * age, hi=0.86 - 0.2 * age)
        ns = noise_mask(0.045, seed=seed + 1, lo=0.3, hi=0.8)

        def streaky(co, n):                       # soft vertical rain streaks, heavier lower down
            c2 = co.copy()
            c2[:, 2] *= 0.1
            return ns(c2, n) * (0.3 + 0.6 * age) * (0.5 + 0.5 * np.clip((H / 2 - co[:, 2]) / H, 0, 1))

        chalk = mask_mul(noise_mask(0.3, seed=seed + 7, lo=0.2, hi=0.9), 0.25 + 0.4 * age)

        def dirt(co, n):                          # rain streaks + rust runs
            return np.maximum(streaky(co, n), rust_sheet(co, n))

        def pnt(o, fl):                           # one weathering story for sheet, pinstripe, line and fill
            paint(o, grime=lambda co, n: np.maximum(dirt(co, n), fl(co, n)), wear=lambda co, n: 0.3 * chipw(co, n))

        pnt(sheet, lambda co, n: 0.0 * co[:, 0])
        paint(rim, grime=mask_mul(facing((0, 0, 1)), 0.5 + 0.4 * age),
              special=mask_mul(height_ramp(-H / 2 - 0.02, -H / 2 + 0.3 * H), noise_mask(0.08, seed=seed + 4, lo=0.3, hi=0.7), 0.3 + age),
              wear=mask_mul(noise_mask(0.15, seed=seed + 5, lo=0.62, hi=0.78), age))
        w_ps = 0.003 + 0.001 * S
        stripe, _, _ = _ribbon("pinstripe", _resample_max(outline(0.034 + w_ps), True, 0.05), w_ps, w_ps, True, y_s - 0.0016)
        assign(stripe, line_m)
        uv_box(stripe, rot90=True)
        pnt(stripe, mask_mul(flake2, 0.5 + 0.5 * age))

        w_o, w_i = r_t + 0.006, r_t + 0.004
        for i, sh in enumerate(shapes):
            closed = sh["closed"]
            base = _dp(sh["outline"], 0.0008)
            if closed:
                base = base[:-1]
            ln, _, _ = _ribbon(f"line{i}", _resample_max(base, closed, 0.024), w_o, w_i, closed, y_s - 0.0016)
            assign(ln, line_m)
            uv_box(ln, rot90=True)
            pnt(ln, flake)
            if not sh["fill"]:
                continue
            holes = [_offset(sh["hole"], w_o - 0.003)] if sh["hole"] else []
            fill = _grid_face(f"art{i}", _offset(base, -(w_i - 0.003)), holes, y_s - 0.001, 0.035, 0.035)
            assign(fill, fill_m)

            def wear_fill(co, n, oc=base):        # flaking creeps in from the edges of the silhouette
                edge = 1 - _ss(0.008, 0.04, _seg_dist(co[:, 0], co[:, 2], oc))
                return flake(co, n) * edge

            uv_box(fill, rot90=True)
            pnt(fill, wear_fill)

        # rust that has crept out from under the paint: tears below fixings and knocks, the knocks themselves and
        # the wet bottom fold of the pan.  One thin skin of iron a hair proud of the paint: rusty along its middle,
        # black and grimed at its edges so it fades into the dark paint.
        rv, rf, r_sp, r_gr = [], [], [], []

        def rvert(x, z, sp, gr):
            rv.append((x, z))
            r_sp.append(sp)
            r_gr.append(gr)
            return len(rv) - 1

        def rface(*ids):
            q = [rv[i] for i in ids]
            rf.append(list(ids) if _area(q) > 0 else list(ids)[::-1])

        for sx_, sz_, w, ln, k in streaks:
            z_lo = _chord(ob, sx_, axis=0)[0] + 0.012
            ln = min(ln * st.u(0.7, 1.2), sz_ - z_lo)
            if ln < 0.025:
                continue
            hw = st.u(0.006, 0.011) * (0.7 + 0.6 * k)
            drift = st.u(-0.006, 0.006)
            prev = None
            for t, wk in ((-0.04, 0.5), (0.1, 1.0), (0.45, 0.62), (0.75, 0.3)):
                cx_, cz_, h = sx_ + drift * t * t, sz_ - 0.004 - t * ln, hw * wk
                row = [rvert(cx_ - h, cz_, 0.0, 1.0), rvert(cx_, cz_, 0.85 * k * (1.0 - 0.5 * max(t, 0.0)), 0.35), rvert(cx_ + h, cz_, 0.0, 1.0)]
                if prev:
                    rface(prev[0], prev[1], row[1], row[0])
                    rface(prev[1], prev[2], row[2], row[1])
                prev = row
            tip = rvert(sx_ + drift, sz_ - 0.004 - ln, 0.0, 1.0)
            rface(prev[0], prev[1], tip)
            rface(prev[1], prev[2], tip)
        for cx_, cz_, r in chips:
            c = rvert(cx_, cz_, 0.15, 0.2)
            a0, nn = st.u(0, 6.28), 7
            ring = [rvert(cx_ + math.cos(a0 + 6.2832 * j / nn) * rr, cz_ + math.sin(a0 + 6.2832 * j / nn) * rr * 0.85, 0.6, 0.6)
                    for j, rr in ((j, r * st.u(0.35, 0.65)) for j in range(nn))]
            for j in range(nn):
                rface(c, ring[j], ring[(j + 1) % nn])
        lo = _resample_max(outline(0.006), True, 0.03)
        k0 = max(range(len(lo)), key=lambda i: lo[i][1])
        lo = lo[k0:] + lo[:k0]
        sgn = 1.0 if _area(lo) > 0 else -1.0
        prev, hp = None, 0.0
        for i in range(len(lo)):
            a, b = lo[i - 1], lo[(i + 1) % len(lo)]
            d = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            nx, nz_ = -sgn * (b[1] - a[1]) / d, sgn * (b[0] - a[0]) / d
            if nz_ < 0.3:
                prev = None
                continue
            hp = 0.45 * hp + 0.55 * (0.02 + 0.09 * age) * st.u(0.1, 1.0) ** 1.6
            h = (0.01 + hp) * min(1.0, (nz_ - 0.3) / 0.5)
            q = lo[i]
            col = [rvert(q[0], q[1], 0.7, 0.4), rvert(q[0] + nx * h * 0.4, q[1] + nz_ * h * 0.4, 0.5 + 0.5 * age, 0.2),
                   rvert(q[0] + nx * h, q[1] + nz_ * h, 0.0, 1.0)]
            if prev:
                rface(prev[0], col[0], col[1], prev[1])
                rface(prev[1], col[1], col[2], prev[2])
            prev = col
        if rf:
            rust_o = mesh_object("rust", _xz(rv, y_s - 0.0022), rf)
            assign(rust_o, iron)
            uv_box(rust_o)
            RV = np.array(rv)

            def look(vals):
                vals = np.array(vals)

                def fn(co, n):
                    d2 = (co[:, None, 0] - RV[None, :, 0]) ** 2 + (co[:, None, 2] - RV[None, :, 1]) ** 2
                    return vals[np.argmin(d2, axis=1)]
                return fn

            paint(rust_o, special=look(r_sp), grime=look(r_gr), mode="set")

        riv = lathe("rivet", [(0.007, 0.0), (0.0052, 0.003), (0.0, 0.0042)], segments=6)
        rivets = instances_to_mesh("rivets", riv, [trs(q, (90, 0, 0)) for q in riv_pts], keep_base=False)
        assign(rivets, iron)
        paint(rivets, special=0.55 + 0.45 * age, wear=0.2 + 0.3 * age)
        # two stiffening flats behind the sheet and four cranked hanging lugs lag-bolted to the wall
        for sz in (-1, 1):
            x0, x1 = _chord(o0, sz * H * 0.27)
            stf = box("stiffener", (max(x1 - x0 - 0.08, 0.1), 0.012, 0.022), loc=((x0 + x1) / 2, -0.014, sz * H * 0.27))
            assign(stf, iron)
        lug_prof = [(-0.0025, -0.016), (0.0025, -0.016), (0.0025, 0.016), (-0.0025, 0.016)]
        hd = lathe("lag", [(0.0085, 0.0), (0.0085, 0.0045), (0.006, 0.0058), (0.0, 0.0058)], segments=6, smooth_angle=25)
        hm = []
        for sxl in (-1, 1):
            xl = sxl * (W / 2) * 0.62
            z0, z1 = _chord(o0, xl, axis=0)
            for zt, sz in ((z1, 1), (z0, -1)):
                path = [(xl, -0.003, zt + sz * 0.07), (xl, -0.003, zt + sz * 0.014), (xl, -0.0175, zt - sz * 0.004), (xl, -0.0175, zt - sz * 0.06)]
                cm = _cum(path)
                lug = sweep("lug", _fillet(path, [cm[1], cm[2]], 0.006), lug_prof, smooth_path=False, up=(1, 0, 0))
                assign(lug, iron)
                paint(lug, special=0.3 + 0.5 * age)
                hm.append(trs((xl, -0.0055, zt + sz * 0.045), (90, 0, rng.u(0, 60))))
        lags = instances_to_mesh("lags", hd, hm, keep_base=False)
        assign(lags, iron)
        paint(lags, special=0.4 + 0.5 * age)

        if p.hood and p.panel_shape != "oval":
            hw = W + 0.03 - (2 * cut if p.panel_shape == "octagon" else 0.0)
            hood = [box("hood", (hw, 0.085, 0.003), loc=(0, -0.0495, H / 2 + 0.016), rot=(14, 0, 0), bevel=0.0008, segments=1),
                    box("hood_lip", (hw, 0.003, 0.016), loc=(0, -0.0905, H / 2 - 0.002), bevel=0.0008, segments=1)]
            for sxh in (-1, 1):
                hood.append(box("hood_end", (0.003, 0.08, 0.024), loc=(sxh * hw / 2, -0.05, H / 2 + 0.006), rot=(14, 0, 0),
                                bevel=0.0008, segments=1))
            densify(hood[0], 0.09)
            for o in hood:
                assign(o, iron)
                paint(o, grime=mask_mul(facing((0, 0, 1)), 0.6 + 0.3 * age),
                      special=mask_mul(noise_mask(0.1, seed=seed + 15, lo=0.35, hi=0.7), 0.3 + age),
                      wear=mask_mul(noise_mask(0.16, seed=seed + 16, lo=0.62, hi=0.76), age))

    # ---- open frame: mitred 25 x 25 x 3 angle iron, flat cross bars lapped behind it, clamp plates under the
    #      stand-offs, corner gussets and four cranked flat-bar brackets on fixing plates ---------------------
    elif backing == "frame":
        o0 = outline(0.0)
        frame_objs = [_loft("angle", _resample_max(o0, True, 0.09),
                            [(0.0, -0.045), (0.0, -0.0722), (0.0008, -0.073), (0.025, -0.073), (0.025, -0.070), (0.003, -0.070),
                             (0.003, -0.045), (0.0, -0.045)])]
        oi = outline(0.004)
        trim = 0.0 if p.panel_shape == "rect" else 0.012
        for v in bars:
            a0, a1 = _chord(oi, v, axis=ax)
            a0, a1 = a0 + trim, a1 - trim
            if ax == 0:
                frame_objs.append(box("bar", (0.025, 0.005, a1 - a0), loc=(v, -0.0675, (a0 + a1) / 2), bevel=0.0012))
            else:
                frame_objs.append(box("bar", (a1 - a0, 0.005, 0.025), loc=((a0 + a1) / 2, -0.0675, v), bevel=0.0012))
        if p.panel_shape == "rect":
            g = 0.06 + 0.03 * S
            for sxg in (-1, 1):
                for szg in (-1, 1):
                    cx_, cz_ = sxg * (W / 2 - 0.004), szg * (H / 2 - 0.004)
                    tri = [(cx_, cz_), (cx_ - sxg * g, cz_), (cx_, cz_ - szg * g)]
                    if _area(tri) < 0:
                        tri.reverse()
                    frame_objs.append(extrude_profile("gusset", tri, 0.003, loc=(0, -0.067, 0), rot=(90, 0, 0)))
        br_prof = [(-0.003, -0.016), (0.003, -0.016), (0.003, 0.016), (-0.003, 0.016)]
        hd = lathe("lag", [(0.0085, 0.0), (0.0085, 0.0045), (0.006, 0.0058), (0.0, 0.0058)], segments=6, smooth_angle=25)
        hm = []
        for sz in (-1, 1):
            za = sz * H * 0.3
            x0, x1 = _chord(o0, za)
            for xa, sxb in ((x0, -1), (x1, 1)):
                xl = xa + sxb * 0.003
                path = [(xa + sxb * 0.13, -0.007, za), (xa + sxb * 0.05, -0.007, za), (xl, -0.034, za), (xl, -0.069, za)]
                cm = _cum(path)
                frame_objs.append(sweep("bracket", _fillet(path, [cm[1], cm[2]], 0.012), br_prof, smooth_path=False, up=(0, 0, 1)))
                frame_objs.append(box("fix_plate", (0.092, 0.004, 0.06), loc=(xa + sxb * 0.092, -0.002, za), bevel=0.001, segments=1))
                hm += [trs((xa + sxb * dx_, -0.010, za), (90, 0, rng.u(0, 60))) for dx_ in (0.072, 0.112)]
                hm.append(trs((xl + sxb * 0.003, -0.0575, za), (0, sxb * 90, 0)))
        frame_objs.append(instances_to_mesh("lags", hd, hm, keep_base=False))
        frame_objs += tabs
        for o in frame_objs:
            assign(o, iron)
            paint(o, special=mask_mul(noise_mask(0.07, seed=seed + 9, lo=0.45, hi=0.9), 0.15 + 0.6 * age),
                  grime=mask_mul(facing((0, 0, 1)), 0.6))
        cl = [box("clamp", (0.05, 0.003, 0.018) if ax == 0 else (0.018, 0.003, 0.05), bevel=0.0008, segments=1)]
        for sgn in (-1, 1):
            off = (sgn * 0.0195, -0.0015, 0.0) if ax == 0 else (0.0, -0.0015, sgn * 0.0195)
            cl.append(lathe("clamp_screw", [(0.0032, 0.0), (0.0032, 0.0016), (0.0, 0.0022)], segments=6, loc=off, rot=(90, 0, 0)))
        clamps = instances_to_mesh("clamps", join(cl, "clamp"), clamp_m, keep_base=False)
        assign(clamps, zinc)
        paint(clamps, special=0.2 + 0.5 * age)

    # ---- transformer box, HT leads, series jumpers, supply conduit ---------------------------------
    sx = {"right": 1, "left": -1}.get(p.transformer, 0)
    wires = []

    def cable(a, b, c1, c2):
        n = max(6, int(_d(a, b) / 0.035))
        return sweep("cable", bezier_path(a, c1, c2, b, n=n), 0.0032, sides=5, smooth_path=False)

    # series wiring order: greedy chain starting from the transformer side
    bx = sx * max(min(0.24 * W, W / 2 - 0.16), 0.0)
    bz = bottom - 0.07 - 0.06
    rem = list(range(len(shapes)))
    cur = (bx, bottom)
    order = []
    while rem:
        best = None
        for k in rem:                              # open strokes may be wired from either end
            for rev in ((False,) if shapes[k]["closed"] else (False, True)):
                dk = _d(cur, shapes[k]["path"][-1 if rev else 0])
                if best is None or dk < best[0]:
                    best = (dk, k, rev)
        j, rev = best[1], best[2]
        pj = shapes[j]["path"]
        order.append((pj[-1], pj[0]) if rev else (pj[0], pj[-1]))     # (start, end) of each tube in series
        rem.remove(j)
        cur = order[-1][1]
    y_w = y_s - 0.006                              # cables leave the housings just above their bases
    y_c = min(y_s + 0.022, -0.006)                 # and run close behind / on the wall
    if backing != "panel":
        for t_a, t_b in zip(order[:-1], order[1:]):
            a2, b2 = t_a[1], t_b[0]
            dd = _d(a2, b2)
            dr = 0.012 + 0.12 * dd
            if dd > 0.28:                          # long hops go through the wall (bare) / along it behind the frame
                if backing == "none":
                    continue
                wires.append(cable((a2[0], y_w, a2[1]), (b2[0], y_w, b2[1]),
                                   (a2[0], 0.03, a2[1] - dr), (b2[0], 0.03, b2[1] - dr)))
                continue
            wires.append(cable((a2[0], y_w, a2[1]), (b2[0], y_w, b2[1]),
                               (a2[0], y_c, a2[1] - dr), (b2[0], y_c, b2[1] - dr)))

    if sx:
        tx = []
        body = box("tx_body", (0.20, 0.084, 0.12), loc=(bx, -0.045, bz), bevel=0.004)
        lid = box("tx_lid", (0.208, 0.005, 0.128), loc=(bx, -0.0895, bz), bevel=0.0015)
        ears = box("tx_ears", (0.27, 0.004, 0.05), loc=(bx, -0.003, bz), bevel=0.001)
        tx += [body, lid, ears]
        for k in range(4):                        # pressed louvres
            tx.append(box("tx_louvre", (0.085, 0.005, 0.0065), loc=(bx + 0.045, -0.0925, bz - 0.042 + k * 0.013), rot=(-28, 0, 0)))
        px = bx - sx * 0.06                        # supply conduit drops from the box and turns into the wall
        zc = bz - 0.06
        cpath = [(px, -0.04, zc + 0.005), (px, -0.04, zc - 0.115), (px, -0.002, zc - 0.115)]
        tx.append(sweep("conduit", _fillet(cpath, [_cum(cpath)[1]], 0.028), 0.0075, sides=8, smooth_path=False))
        tx.append(cylinder("conduit_flange", 0.018, 0.005, loc=(px, -0.0035, zc - 0.115), rot=(90, 0, 0), verts=12, bevel=0.001))
        tx.append(box("conduit_saddle", (0.034, 0.006, 0.012), loc=(px, -0.047, zc - 0.045), bevel=0.001))
        for o in tx:
            assign(o, zinc)
            paint(o, grime=mask_mul(facing((0, 0, 1)), 0.5 + 0.4 * age),
                  special=mask_mul(noise_mask(0.05, seed=seed + 12, lo=0.4, hi=0.8), height_ramp(bz - 0.2, bz + 0.1), age * 1.3),
                  wear=mask_mul(noise_mask(0.03, seed=seed + 13, lo=0.55, hi=0.8), age * 0.6))
        sc = lathe("tx_screw", [(0.005, 0.0), (0.004, 0.002), (0.0, 0.003)], segments=6)
        fix = [trs((bx + ax * 0.094, -0.092, bz + az * 0.054), (90, 0, 0)) for ax in (-1, 1) for az in (-1, 1)]
        fix += [trs((bx + ax * 0.118, -0.005, bz), (90, 0, 0), 1.5) for ax in (-1, 1)]
        assign(instances_to_mesh("tx_screws", sc, fix, keep_base=False), brass)
        assign(box("tx_plate", (0.07, 0.0016, 0.028), loc=(bx - 0.05, -0.0926, bz + 0.03), bevel=0.0005), brass)
        assign(cylinder("tx_gland", 0.012, 0.014, loc=(px, -0.04, zc - 0.004), verts=6), brass)
        assign(cylinder("tx_switch", 0.012, 0.01, loc=(bx + sx * 0.105, -0.05, bz + 0.018), rot=(0, 90, 0), verts=12, bevel=0.001), brass)
        assign(box("tx_lever", (0.008, 0.006, 0.03), loc=(bx + sx * 0.113, -0.05, bz + 0.028), rot=(18, 0, 0), bevel=0.0015), black)
        ins = lathe("insulator", [(0.014, 0.0), (0.014, 0.006), (0.009, 0.009), (0.016, 0.014), (0.009, 0.019), (0.0145, 0.024),
                                  (0.008, 0.029), (0.006, 0.04), (0.0, 0.04)], segments=10)
        ztop = bz + 0.06
        ix = [bx - 0.055, bx + 0.055]
        assign(instances_to_mesh("insulators", ins, [trs((x, -0.045, ztop)) for x in ix], keep_base=False), china)
        if backing == "panel":                    # leads disappear behind the pan
            for x in ix:
                a3 = (x, -0.045, ztop + 0.038)
                b3 = (x, -0.011, bottom + 0.03)
                wires.append(cable(a3, b3, (x, -0.045, ztop + 0.06), (x, -0.011, bottom - 0.02)))
        else:                                     # one lead climbs to the first electrode, the return runs in the wall
            first, last = order[0][0], order[-1][1]
            xi = min(ix, key=lambda x: abs(x - first[0]))
            xo = ix[0] if xi == ix[1] else ix[1]
            a3 = (xi, -0.045, ztop + 0.038)
            rise = 0.35 * abs(first[1] - a3[2]) + 0.03
            wires.append(cable(a3, (first[0], y_w, first[1]), (xi, -0.045, ztop + 0.038 + rise), (first[0], y_c, first[1] - rise)))
            zg = ztop + 0.1
            wires.append(cable((xo, -0.045, ztop + 0.038), (xo, -0.004, zg), (xo, -0.045, ztop + 0.085), (xo, -0.032, zg)))
            assign(cylinder("grommet", 0.009, 0.006, loc=(xo, -0.003, zg), rot=(90, 0, 0), verts=8), china)
            if backing == "frame":
                wires.append(cable((last[0], y_w, last[1]), (last[0], -0.004, last[1] - 0.03),
                                   (last[0], y_w + 0.03, last[1]), (last[0], -0.03, last[1] - 0.03)))
    for o in wires:
        assign(o, black)

    # ---- light: a few low sockets spread along the tubes of each colour, close in front of the glass ----
    def spread(pts, k):
        xs, zs = [q[0] for q in pts], [q[1] for q in pts]
        a = 0 if (max(xs) - min(xs)) >= (max(zs) - min(zs)) else 1
        srt = sorted(pts, key=lambda q: q[a])
        out = []
        for j in range(k):
            grp = srt[j * len(srt) // k:(j + 1) * len(srt) // k]
            c = [sum(q[i] for q in grp) / len(grp) for i in (0, 1)]
            out.append(min(grp, key=lambda q: (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2))
        return out

    lens, cpts = [0.0, 0.0], [[], []]
    for sh in shapes:
        lens[sh["col"]] += sh["pcum"][-1]
        cpts[sh["col"]] += sh["path"]
    total = float(p.intensity) * S ** 0.7 * {"none": 0.25, "frame": 0.23}.get(backing, 1.0)    # bare wall is close and pale
    share = [1.0, 0.0] if lens[1] < 1e-6 else [max(0.3, min(0.7, lens[0] / (lens[0] + lens[1]))), 0.0]
    share[1] = 1.0 - share[0]
    h_l = 0.1 + 0.04 * S
    for c, (nm, col) in enumerate((("neon_main", p.colour), ("neon_accent", p.colour2))):
        if lens[c] < 1e-6:
            continue
        k = (1 if lens[c] < 1.5 else 2 if lens[c] < 3.5 else 3) if c == 0 else (1 if lens[c] < 2.0 else 2)
        for j, q in enumerate(spread(cpts[c], k)):
            light(f"{nm}{j}", (q[0], y_t - h_l, q[1]), color=col, intensity=total * share[c] / k,
                  range=0.9 + 0.65 * S, flicker=float(p.flicker), shadow_near=0.6 * S + 0.3)
    socket("supply", "attach", (bx, 0.0, bz - 0.2), role="cable")
