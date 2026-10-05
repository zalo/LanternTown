"""Vertical projecting neon sign (the HOTEL blade sign of the reference): a riveted, painted sheet-metal box
standing off the wall on two box-section arms with flanges, braced by turnbuckle stay rods. Each face carries
channel letters stacked top to bottom (sheet returns + enamelled faces) with bent glass neon tubes on
stand-offs whose ends dive into ceramic boots, a raised riveted border (optionally a chaser-bulb border),
lapped sheet seams, and a cut-out emblem can (crown / star) on top with its own neon. Fed by a conduit and a
wall junction box. Lettering is a built-in single-stroke font so tubes and channels follow the same strokes.
"""
from math import acos, tan, ceil, degrees, radians, sin, cos, pi

import numpy as np
from mathutils.geometry import delaunay_2d_cdt
from tk import *

META = {"name": "Vertical neon hotel sign", "class": "component", "mount": "wall", "mount_height": 4.5,
        "focus": 0.66, "tri_budget": 22000}   # budget: letters, tubes, rivets all exist twice (two faces)
PARAMS = {
    "text": {"default": "HOTEL", "desc": "sign text, stacked top to bottom (A-Z, 0-9, space, '-'; max 9 characters)"},
    "letter_height": {"default": 0.34, "min": 0.2, "max": 0.5, "desc": "height of one channel letter (m)"},
    "tube_colour": {"default": "#37e0ff", "desc": "neon gas colour of the letters"},
    "box_colour": {"default": "#1e6178", "desc": "paint colour of the sheet-metal box"},
    "trim_colour": {"default": "#c9992b", "desc": "paint colour of rolled edges, border and letter returns"},
    "emblem": {"default": "crown", "choices": ["crown", "star", "none"], "desc": "neon emblem on top of the box"},
    "emblem_colour": {"default": "#ffd24a", "desc": "neon gas colour of the emblem"},
    "bulbs": {"default": False, "desc": "chaser-bulb border on the raised frame of both faces"},
    "edge_tube": {"default": True, "desc": "hairpin neon stripe running up the outer edge of the box"},
    "shape": {"default": "square", "choices": ["square", "round", "point"], "desc": "shape of the bottom of the box"},
    "depth": {"default": 0.2, "min": 0.14, "max": 0.32, "desc": "thickness of the box (m)"},
    "standoff": {"default": 0.24, "min": 0.12, "max": 0.6, "desc": "gap between wall and box (m)"},
    "stays": {"default": 2, "choices": [2, 4], "desc": "stay rods: upper pair, or upper + lower pair"},
    "dead": {"default": 0, "min": 0, "max": 3, "desc": "number of burnt-out (unlit) letters"},
    "wear": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "rust streaks, grime and chipped paint"},
    "intensity": {"default": 2.6, "min": 0.5, "max": 8.0, "desc": "light per metre of sign and face (cd)"},
}
VARIANTS = {
    "cafe_bulbs": {"text": "CAFE", "letter_height": 0.3, "tube_colour": "#ff3a4a", "box_colour": "#5e1616",
                   "trim_colour": "#e3d6b0", "emblem": "star", "emblem_colour": "#ffe9a8", "bulbs": True,
                   "shape": "round", "seed": 4},
    "bar_point": {"text": "BAR", "letter_height": 0.46, "tube_colour": "#ff4fc4", "box_colour": "#1d1f26",
                  "trim_colour": "#8f979c", "emblem": "none", "shape": "point", "stays": 4, "depth": 0.26,
                  "standoff": 0.4, "edge_tube": False, "seed": 11},
    "rooms_worn": {"text": "ROOMS", "letter_height": 0.3, "tube_colour": "#ff8a2a", "box_colour": "#27402d",
                   "trim_colour": "#b9b08c", "emblem": "crown", "emblem_colour": "#ff5a3a", "dead": 1, "wear": 1.0,
                   "stays": 4, "seed": 23},
    "theatre_long": {"text": "THEATRE", "letter_height": 0.26, "tube_colour": "#ffd24a", "box_colour": "#3a1f4f",
                     "trim_colour": "#d9c48a", "emblem": "star", "emblem_colour": "#ff5fa0", "bulbs": True,
                     "depth": 0.17, "standoff": 0.18, "seed": 31},
}

# ---- single-stroke font: unit box x 0..1, y 0..1; a stroke whose first point == last point is a closed loop
_O = [(0.78, 0), (0.22, 0), (0, 0.12), (0, 0.88), (0.22, 1), (0.78, 1), (1, 0.88), (1, 0.12), (0.78, 0)]
FONT = {
    "A": [[(0, 0), (0.5, 1), (1, 0)], [(0.18, 0.36), (0.82, 0.36)]],
    "B": [[(0, 0.52), (0.8, 0.52), (1, 0.64), (1, 0.88), (0.8, 1), (0, 1), (0, 0), (0.8, 0), (1, 0.12), (1, 0.4), (0.8, 0.52)]],
    "C": [[(1, 0.76), (1, 0.88), (0.78, 1), (0.22, 1), (0, 0.88), (0, 0.12), (0.22, 0), (0.78, 0), (1, 0.12), (1, 0.24)]],
    "D": [[(0.72, 0), (0, 0), (0, 1), (0.72, 1), (1, 0.84), (1, 0.16), (0.72, 0)]],
    "E": [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, 0.52), (0.72, 0.52)]],
    "F": [[(1, 1), (0, 1), (0, 0)], [(0, 0.5), (0.72, 0.5)]],
    "G": [[(1, 0.76), (1, 0.88), (0.78, 1), (0.22, 1), (0, 0.88), (0, 0.12), (0.22, 0), (0.78, 0), (1, 0.12), (1, 0.46), (0.52, 0.46)]],
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.52), (1, 0.52)]],
    "I": [[(0.5, 0), (0.5, 1)], [(0.0, 1), (1.0, 1)], [(0.0, 0), (1.0, 0)]],
    "J": [[(1, 1), (1, 0.14), (0.78, 0), (0.22, 0), (0, 0.14), (0, 0.32)]],
    "K": [[(0, 0), (0, 1)], [(1, 1), (0, 0.45)], [(0.36, 0.648), (1, 0)]],
    "L": [[(0, 1), (0, 0), (1, 0)]],
    "M": [[(0, 0), (0, 1), (0.5, 0.35), (1, 1), (1, 0)]],
    "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "O": [_O],
    "P": [[(0, 0), (0, 1), (0.8, 1), (1, 0.88), (1, 0.6), (0.8, 0.48), (0, 0.48)]],
    "Q": [_O, [(0.58, 0.28), (0.89, 0.06)], [(0.89, 0.06), (1.1, -0.09)]],
    "R": [[(0, 0), (0, 1), (0.8, 1), (1, 0.88), (1, 0.6), (0.8, 0.48), (0, 0.48)], [(0.45, 0.48), (1, 0)]],
    "S": [[(1, 0.78), (1, 0.88), (0.78, 1), (0.22, 1), (0, 0.88), (0, 0.64), (0.22, 0.52), (0.78, 0.52), (1, 0.4), (1, 0.12),
           (0.78, 0), (0.22, 0), (0, 0.12), (0, 0.22)]],
    "T": [[(0, 1), (1, 1)], [(0.5, 1), (0.5, 0)]],
    "U": [[(0, 1), (0, 0.12), (0.22, 0), (0.78, 0), (1, 0.12), (1, 1)]],
    "V": [[(0, 1), (0.5, 0), (1, 1)]],
    "W": [[(0, 1), (0.22, 0), (0.5, 0.65), (0.78, 0), (1, 1)]],
    "X": [[(0, 0), (1, 1)], [(0, 1), (0.5, 0.5)], [(0.5, 0.5), (1, 0)]],
    "Y": [[(0, 1), (0.5, 0.48), (1, 1)], [(0.5, 0.48), (0.5, 0)]],
    "Z": [[(0, 1), (1, 1), (0, 0), (1, 0)]],
    "0": [_O],
    "1": [[(0.15, 0.76), (0.55, 1), (0.55, 0)], [(0.15, 0), (0.95, 0)]],
    "2": [[(0, 0.78), (0, 0.88), (0.22, 1), (0.78, 1), (1, 0.88), (1, 0.62), (0, 0), (1, 0)]],
    "3": [[(0, 0.88), (0.22, 1), (0.78, 1), (1, 0.88), (1, 0.64), (0.78, 0.52), (0.35, 0.52)],
          [(0.78, 0.52), (1, 0.4), (1, 0.12), (0.78, 0), (0.22, 0), (0, 0.12)]],
    "4": [[(0, 1), (0, 0.42), (0.8, 0.42)], [(0.8, 1), (0.8, 0)]],
    "5": [[(1, 1), (0, 1), (0, 0.55), (0.78, 0.55), (1, 0.43), (1, 0.12), (0.78, 0), (0.22, 0), (0, 0.12)]],
    "6": [[(1, 0.88), (0.78, 1), (0.22, 1), (0, 0.88), (0, 0.12), (0.22, 0), (0.78, 0), (1, 0.12), (1, 0.43), (0.78, 0.55), (0, 0.55)]],
    "7": [[(0, 1), (1, 1), (0.35, 0)]],
    "8": [[(0.22, 0.52), (0, 0.64), (0, 0.88), (0.22, 1), (0.78, 1), (1, 0.88), (1, 0.64), (0.78, 0.52), (0.22, 0.52), (0, 0.4),
           (0, 0.12), (0.22, 0), (0.78, 0), (1, 0.12), (1, 0.4), (0.78, 0.52)]],
    "9": [[(0, 0.12), (0.22, 0), (0.78, 0), (1, 0.12), (1, 0.88), (0.78, 1), (0.22, 1), (0, 0.88), (0, 0.57), (0.22, 0.45), (1, 0.45)]],
    "-": [[(0.05, 0.5), (0.95, 0.5)]],
}
_WIDE = {"M": 1.3, "W": 1.35, "I": 0.55, "1": 0.7, "-": 0.8}


# ---------------------------------------------------------------- 2D / 3D polyline helpers
def _clip_dive(pts, dmin):
    """Cut both ends of a tube path (u, v, d) off at depth dmin: the lit tube stops at the lid of its ceramic boot instead
    of running on inside it. CLEARANCE RULE (depth fighting): a glowing surface keeps >= 4 mm to any parallel surface in
    front of it - the hidden tube end lay 2-4 mm inside the boot wall."""
    P = [np.array(q, dtype=float) for q in pts]
    for _ in range(2):
        while len(P) > 2 and P[1][2] <= dmin:
            P.pop(0)
        a, b = P[0], P[1]
        if a[2] < dmin < b[2]:
            P[0] = a + (b - a) * ((dmin - a[2]) / (b[2] - a[2]))
        P.reverse()
    return P


def _fillet(pts, r, closed=False, seg_deg=30.0, min_turn=32.0):
    """Round the corners of a polyline (2D or 3D) with radius r (clamped to the leg lengths)."""
    P = [np.array(q, dtype=float) for q in pts]
    n = len(P)
    out = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            out.append(P[i])
            continue
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        u, v = a - b, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-9 or lv < 1e-9:
            out.append(b)
            continue
        u, v = u / lu, v / lv
        ang = acos(max(-1.0, min(1.0, float(np.dot(u, v)))))      # angle between the legs
        turn = pi - ang
        if turn < radians(min_turn) or ang < radians(4):
            out.append(b)
            continue
        t = min(r / tan(ang / 2), 0.48 * lu, 0.48 * lv)
        rr = t * tan(ang / 2)
        bis = u + v
        bis /= np.linalg.norm(bis)
        cen = b + bis * (rr / sin(ang / 2))
        s = b + u * t - cen
        e = b + v * t - cen
        w = e - s * cos(turn)
        w = w / max(np.linalg.norm(w), 1e-12) * rr
        m = max(1, int(ceil(degrees(turn) / seg_deg)))
        for k in range(m + 1):
            th = turn * k / m
            out.append(cen + s * cos(th) + w * sin(th))
    return out


def _vnormals(P):
    """Per-vertex right-hand (outward for CCW) miter normals of a closed 2D loop."""
    P = np.asarray(P, dtype=float)
    e = np.roll(P, -1, 0) - P
    e /= np.maximum(np.linalg.norm(e, axis=1), 1e-12)[:, None]
    nr = np.stack([e[:, 1], -e[:, 0]], 1)
    nv = nr + np.roll(nr, 1, 0)
    ln = np.maximum(np.linalg.norm(nv, axis=1), 1e-9)
    nv /= ln[:, None]
    cosh = np.clip((nv * nr).sum(1), 0.45, 1.0)
    return nv / cosh[:, None]


def _offset(P, d):
    """Offset a closed loop to its left (inwards for CCW) by d."""
    P = np.asarray(P, dtype=float)
    return P - _vnormals(P) * d


def _area(P):
    P = np.asarray(P)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _inside(pt, P):
    x, y = pt
    c = False
    n = len(P)
    for i in range(n):
        x0, y0 = P[i]
        x1, y1 = P[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            c = not c
    return c


def _dp(P, tol):
    """Douglas-Peucker on an open polyline (np array) -> kept indices."""
    keep = {0, len(P) - 1}
    stack = [(0, len(P) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = P[i0], P[i1]
        ab = b - a
        L = np.linalg.norm(ab)
        q = P[i0 + 1:i1] - a
        if L < 1e-12:
            d = np.linalg.norm(q, axis=1)
        else:
            d = np.abs(q[:, 0] * ab[1] - q[:, 1] * ab[0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i0 + 1
            keep.add(k)
            stack += [(i0, k), (k, i1)]
    return sorted(keep)


def _simplify_closed(P, tol):
    P = np.asarray(P)
    n = len(P)
    i0 = int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))
    P = np.roll(P, -i0, 0)
    i1 = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    a = _dp(P[:i1 + 1], tol)
    Pb = np.vstack([P[i1:], P[:1]])
    b = _dp(Pb, tol)
    idx = a[:-1] + [i1 + k for k in b[:-1]]
    return P[idx]


def _resample_closed(P, spacing, phase=0.5):
    P = np.asarray(P, dtype=float)
    seg = np.roll(P, -1, 0) - P
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    total = cum[-1]
    n = max(4, int(round(total / spacing)))
    out = []
    for k in range(n):
        s = (k + phase) / n * total
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(P) - 1)
        out.append(P[i] + seg[i] * ((s - cum[i]) / max(L[i], 1e-9)))
    return out


def _outline(strokes, hw, ext, cell):
    """Outline(s) of a set of strokes fattened by hw (butt ends extended by ext): signed-distance grid +
    marching squares. strokes: [(Nx2 array, closed)]. Returns [(outer CCW, [holes CW])]."""
    allp = np.vstack([s for s, _ in strokes])
    pad = hw + ext + 3 * cell
    lo = allp.min(0) - pad
    hi = allp.max(0) + pad
    nx = int(ceil((hi[0] - lo[0]) / cell))
    ny = int(ceil((hi[1] - lo[1]) / cell))
    xs = lo[0] + np.arange(nx + 1) * cell
    ys = lo[1] + np.arange(ny + 1) * cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    D = np.full(X.shape, 1e9)
    for pts, closed in strokes:
        n = len(pts)
        ns = n if closed else n - 1
        for k in range(ns):
            a, b = pts[k], pts[(k + 1) % n]
            ab = b - a
            L = float(np.linalg.norm(ab))
            if L < 1e-9:
                continue
            u = ab / L
            px, py = X - a[0], Y - a[1]
            t = px * u[0] + py * u[1]
            perp = np.abs(px * u[1] - py * u[0])
            tc = np.clip(t, 0, L)
            d = np.hypot(px - tc * u[0], py - tc * u[1])
            if not closed and k == ns - 1:
                d = np.where(t > L, np.maximum(perp, t - L - ext + hw), d)
            if not closed and k == 0:
                d = np.where(t < 0, np.maximum(perp, -t - ext + hw), d)
            D = np.minimum(D, d)
    F = D - hw
    ins = F < 0
    case = (ins[:-1, :-1] * 1 + ins[1:, :-1] * 2 + ins[1:, 1:] * 4 + ins[:-1, 1:] * 8)
    table = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)],
             8: [(2, 3)], 9: [(0, 2)], 10: [(0, 1), (2, 3)], 11: [(1, 2)], 12: [(3, 1)], 13: [(0, 1)], 14: [(3, 0)]}
    adj = {}
    for i, j in np.argwhere((case > 0) & (case < 15)):
        i, j = int(i), int(j)
        ek = ((0, i, j), (1, i + 1, j), (0, i, j + 1), (1, i, j))
        for e0, e1 in table[int(case[i, j])]:
            adj.setdefault(ek[e0], []).append(ek[e1])
            adj.setdefault(ek[e1], []).append(ek[e0])

    def pos(key):
        kind, i, j = key
        if kind == 0:
            f0, f1 = F[i, j], F[i + 1, j]
            return (xs[i] + cell * f0 / (f0 - f1), ys[j])
        f0, f1 = F[i, j], F[i, j + 1]
        return (xs[i], ys[j] + cell * f0 / (f0 - f1))

    loops = []
    seen = set()
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nb = adj[cur]
            nxt = nb[0] if nb[0] != prev else nb[-1]
            if nxt == start or nxt in seen:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 6:
            P = _simplify_closed(np.array([pos(k) for k in loop]), cell * 0.22)
            if len(P) >= 3:
                loops.append(P)
    depth = [sum(1 for k, Q in enumerate(loops) if k != i and _inside(P[0], Q)) for i, P in enumerate(loops)]
    res = []
    for i, P in enumerate(loops):
        if depth[i] % 2 == 0:
            outer = P if _area(P) > 0 else P[::-1]
            holes = []
            for k, Q in enumerate(loops):
                if depth[k] == depth[i] + 1 and _inside(Q[0], outer):
                    holes.append(Q if _area(Q) < 0 else Q[::-1])
            res.append((outer, holes))
    return res


# ---------------------------------------------------------------- mesh helpers (u, v, d face coords -> world)
def _prism(name, loops, fm, d0, d1, cap0=False, cap1=True, sides=True):
    """Extrude 2D loops (outer CCW first, holes CW) from d0 to d1 through the mapping fm(u, v, d)."""
    flat = [(float(q[0]), float(q[1])) for lp in loops for q in lp]
    n = len(flat)
    verts = [fm(u, v, d0) for (u, v) in flat] + [fm(u, v, d1) for (u, v) in flat]
    faces = []
    if cap0 or cap1:
        # constrained Delaunay over all loops, then keep triangles inside the outer loop and outside the holes
        edges = []
        o = 0
        for lp in loops:
            m = len(lp)
            edges += [(o + i, o + (i + 1) % m) for i in range(m)]
            o += m
        res = delaunay_2d_cdt([Vector(q) for q in flat], edges, [], 0, 1e-7)
        cv, cf, orig = res[0], res[2], res[3]
        remap = []
        for k, v in enumerate(cv):
            if orig[k]:
                remap.append(orig[k][0])
            else:
                remap.append(None)
        lp_np = [np.asarray(lp, dtype=float) for lp in loops]
        for tri in cf:
            if len(tri) != 3 or any(remap[t] is None for t in tri):
                continue
            a, b, c = (remap[t] for t in tri)
            pa, pb, pc = flat[a], flat[b], flat[c]
            ar = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if abs(ar) < 1e-12:
                continue
            cen = ((pa[0] + pb[0] + pc[0]) / 3, (pa[1] + pb[1] + pc[1]) / 3)
            if not _inside(cen, lp_np[0]) or any(_inside(cen, hl) for hl in lp_np[1:]):
                continue
            if ar < 0:
                b, c = c, b
            if cap1:
                faces.append((n + a, n + b, n + c))
            if cap0:
                faces.append((a, c, b))
    if sides:
        o = 0
        for lp in loops:
            m = len(lp)
            for i in range(m):
                j = (i + 1) % m
                faces.append((o + i, o + j, n + o + j, n + o + i))
            o += m
    return mesh_object(name, verts, faces)


def _tube_strokes(strokes, trim, gap):
    """Tube centre lines from glyph strokes: closed loops are opened (two electrodes side by side),
    open ends are pulled back so tubes never touch at junctions."""
    out = []
    for pts, closed in strokes:
        P = [np.array(q, dtype=float) for q in pts]
        if closed:
            a, b = P[0], P[1]
            mid = (a + b) / 2
            u = (b - a) / np.linalg.norm(b - a)
            P = [mid + u * gap / 2] + P[1:] + [P[0], mid - u * gap / 2]
        else:
            for end in (0, -1):
                a, b = P[end], P[1 if end == 0 else -2]
                L = np.linalg.norm(b - a)
                P[end] = a + (b - a) / L * min(trim, 0.3 * L)
        out.append(P)
    return out


def build(p, rng):
    w = float(p.wear)
    box_m = mat("iron_painted", p.box_colour)
    trim_m = mat("iron_painted", p.trim_colour)
    enamel = mat("ceramic_glazed", "#ece5cf", grime=0.5)
    iron = mat("iron_wrought")
    zinc = mat("zinc")
    neon_m = mat("neon", p.tube_colour, glow=6)
    dead_m = mat("glass_frosted", "#aab4b6", glow=0.0, opacity=0.8)
    bulb_m = mat("emissive", "#ffd9a0", glow=4)

    # ---- text and layout
    chars = [c for c in str(p.text).upper() if c in FONT or c == " "][:9]
    if not any(c != " " for c in chars):
        chars = list("HOTEL")
    h = float(p.letter_height)
    hw = 0.088 * h                       # half width of a channel stroke
    hc = h - 2 * hw                      # centre-line height of a glyph
    wc0 = 0.72 * h - 2 * hw              # centre-line width of a normal glyph
    row_gap = 0.25 * h
    rows = [(c, 0.5 * h if c == " " else h) for c in chars]
    text_h = sum(r[1] for r in rows) + row_gap * (len(rows) - 1)
    max_w = max([wc0 * _WIDE.get(c, 1.0) + 2 * hw for c in chars if c != " "])
    frame_w = 0.05
    pad = 0.045 + 0.08 * h
    W = max_w + 2 * (frame_w + 0.006 + pad)
    H = text_h + 2 * (frame_w + 0.006 + pad)
    T = float(p.depth)
    gap = float(p.standoff)
    yc = -(gap + W / 2)
    y_out = -(gap + W)
    print(f"DBG box W={W:.3f} H={H:.3f} T={T:.3f} chars={''.join(chars)}")

    def fm(s, half=T / 2, v0=0.0):
        return lambda u, v, d: (s * (half + d), yc + s * u, v0 + v)

    # ---- box outline (u across, v up), CCW
    if p.shape == "round":
        base = [(W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2), (-W / 2, -H / 2)]
        outline = _fillet(base[:3] + [base[3]], 0.03, closed=False)[1:3]
        outline = [np.array(base[0])] + _fillet([base[0], base[1], base[2], base[3]], 0.03)[1:-1] + [np.array(base[3])]
        outline += [np.array((W / 2 * cos(radians(a)), -H / 2 + W / 2 * sin(radians(a)))) for a in np.linspace(180, 360, 15)[1:-1]]
        z_bot = -H / 2 - W / 2
    elif p.shape == "point":
        outline = _fillet([(W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2), (-W / 2, -H / 2), (0, -H / 2 - 0.5 * W)], 0.035,
                          closed=True, min_turn=20)
        z_bot = -H / 2 - 0.5 * W
    else:
        outline = _fillet([(W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2), (-W / 2, -H / 2)], 0.03, closed=True)
        z_bot = -H / 2
    outline = np.array(outline)
    if _area(outline) < 0:
        outline = outline[::-1]

    # ---- weathering masks (world space)
    sd = float(p.seed % 97)

    def streak_mask(co, nrm):
        y, z, x = co[:, 1], co[:, 2], co[:, 0]
        s = 0.5 + 0.5 * np.sin(y * 57.0 + 2.5 * np.sin(y * 13.0 + sd) + sd)
        s = s ** 3
        top = np.clip((H / 2 - z) / 0.9, 0.0, 1.0)
        low = np.clip((z_bot + 0.35 - z) / 0.35, 0.0, 1.0)
        side = np.clip(np.abs(x) / (T / 2 + 1e-6), 0, 1)
        return np.clip(w * (0.75 * s * (1.0 - 0.6 * top) * side + 0.5 * low) + 0.08, 0, 1)

    nm = noise_mask(0.11, seed=p.seed + 3, lo=0.42, hi=0.7)
    nm2 = noise_mask(0.05, seed=p.seed + 8, lo=0.5, hi=0.75)

    def rust_mask(co, nrm):
        z = co[:, 2]
        low = np.clip((z_bot + 0.5 - z) / 0.5, 0.0, 1.0)
        return np.clip(w * (nm(co, nrm) * (0.35 + 0.9 * low) + 0.5 * low * low), 0, 1)

    def chip_mask(co, nrm):
        return np.clip(w * nm2(co, nrm) * 0.8, 0, 1)

    def weather(o, rust=1.0):
        paint(o, grime=streak_mask, special=lambda c, n: rust_mask(c, n) * rust, wear=chip_mask)

    # ---- box core: two skins + edge band
    core = _prism("box_core", [outline], lambda u, v, d: (d, yc + u, v), -T / 2, T / 2, cap0=True, cap1=True, sides=True)
    densify(core, 0.11)
    smooth(core, 35)
    assign(core, box_m)
    weather(core)
    # rolled edges: a bead wrapping each face edge
    bead_loop = _offset(outline, 0.004)
    for s in (-1, 1):
        b = sweep(f"rolled_edge{s}", [(s * (T / 2 - 0.004), yc + q[0], q[1]) for q in bead_loop], 0.0125, closed=True,
                  sides=7, smooth_path=False, up=(1, 0, 0))
        assign(b, trim_m)
        weather(b, 0.7)
    # raised riveted border on each face
    f_out = _offset(outline, 0.014)
    f_in = _offset(outline, 0.014 + frame_w)
    f_mid = _offset(outline, 0.014 + frame_w / 2)
    fd = 0.004
    for s in (-1, 1):
        fr = _prism(f"border{s}", [f_out, f_in[::-1]], fm(s), 0.0, fd)
        bevel(fr, 0.0012, segments=1)
        assign(fr, trim_m)
        weather(fr, 0.8)

    rivet = lathe("rivet", [(0.0075, 0.0), (0.0056, 0.0034), (0.0, 0.0048)], segments=5)
    rivets = []
    bulb_pos = []
    spacing = 0.062 if p.bulbs else 0.085
    ring = _resample_closed(f_mid, spacing, phase=0.5)
    if p.bulbs and len(ring) % 2:
        ring = _resample_closed(f_mid, spacing * len(ring) / (len(ring) + 1), phase=0.5)
    for s in (-1, 1):
        for k, q in enumerate(ring):
            if p.bulbs and k % 2 == 0:
                bulb_pos.append((s, q[0], q[1]))
            else:
                rivets.append(trs(fm(s)(q[0], q[1], fd), (0, 90 * s, 0), rng.vary(1.0, 0.08)))

    # ---- lapped sheet seams between letter rows
    tops = []
    z = text_h / 2
    for c, rh in rows:
        tops.append((c, z - rh / 2, rh))
        z -= rh + row_gap
    seam_z = []
    last = H / 2
    for i in range(len(tops) - 1):
        zs = tops[i][1] - tops[i][2] / 2 - row_gap / 2
        if last - zs > 0.78 and zs - (-H / 2) > 0.5:
            seam_z.append(zs)
            last = zs
    iw = W - 2 * (0.014 + frame_w)
    for zs in seam_z:
        for s in (-1, 1):
            st = box("seam", (0.0024, iw + 0.004, min(0.026, row_gap * 0.5)), loc=(s * (T / 2 + 0.0008), yc, zs), bevel=0.0008, segments=1)
            assign(st, box_m)
            weather(st, 1.3)
            nr = max(3, int(iw / 0.075))
            for k in range(nr):
                rivets.append(trs(fm(s)(-iw / 2 + iw * (k + 0.5) / nr, zs, 0.002), (0, 90 * s, 0), 0.8))
        band = box("seam_edge", (T - 0.02, 0.0024, 0.026), loc=(0, y_out - 0.0008, zs), bevel=0.0008, segments=1)
        assign(band, box_m)
        weather(band, 1.3)

    # ---- outer edge band: rivet columns, service hatch, drain / vent slots
    nz = max(2, int(H / 0.17))
    hatch_z = -H / 2 + min(0.42, H * 0.3)
    hatch_h = min(0.26, H * 0.3)
    for k in range(nz):
        zz = -H / 2 + H * (k + 0.5) / nz
        for sx in (-1, 1):
            rivets.append(trs((sx * (T / 2 - 0.03), y_out, zz), (90, 0, 0), 0.9))
            rivets.append(trs((sx * (T / 2 - 0.03), -gap, zz), (-90, 0, 0), 0.9))
    hatch = box("hatch", (T - 0.085, 0.004, hatch_h), loc=(0, y_out - 0.002, hatch_z), bevel=0.0012, segments=1)
    assign(hatch, box_m)
    weather(hatch, 1.2)
    for sz in (-1, 1):
        hg = cylinder("hatch_hinge", 0.006, 0.05, loc=(-(T - 0.085) / 2 - 0.004, y_out - 0.005, hatch_z + sz * hatch_h * 0.3), verts=8)
        assign(hg, iron)
        rivets.append(trs(((T - 0.085) / 2 - 0.014, y_out - 0.004, hatch_z + sz * hatch_h * 0.36), (90, 0, 0), 0.75))
    latch = box("hatch_latch", (0.012, 0.008, 0.045), loc=((T - 0.085) / 2 - 0.016, y_out - 0.007, hatch_z), bevel=0.002, segments=1)
    assign(latch, iron)
    nlv = 4
    for k in range(nlv):
        lz = hatch_z + hatch_h / 2 + 0.05 + k * 0.026
        lv = box("louvre", (T - 0.1, 0.012, 0.004), loc=(0, y_out - 0.004, lz), rot=(-35, 0, 0), bevel=0.001, segments=1)
        assign(lv, box_m)
        weather(lv, 1.2)

    # ---- channel letters + neon on both faces
    r_tube = max(0.006, min(0.0085, 0.023 * h))
    d_wall = 0.028 + 0.02 * h
    d_floor = d_wall + 0.0006          # enamelled lid of the letter can, a hair proud of the returns
    d_tube = 0.017
    r_bend = max(0.016, 0.55 * hw)
    cell = hw / 9.0
    lit = [i for i, (c, _, _) in enumerate(tops) if c != " "]
    rd = rng.fork("dead")
    dead = set()
    pool = list(lit)
    for _ in range(min(int(p.dead), max(0, len(lit) - 1))):
        k = pool.pop(int(rd.u(0, len(pool) - 1e-6)))
        dead.add(k)
    boot = lathe("boot", [(r_tube * 1.9, 0.0), (r_tube * 1.6, 0.011), (0.0, 0.011)], segments=6)
    post = cylinder("post", 0.0028, d_tube, loc=(0, 0, d_tube / 2), verts=5, cap=False)
    boots, posts = [], []          # (location, euler) of ceramic boots / glass stand-offs
    cache = {}
    letters_walls, letters_floor, tubes_lit, tubes_dead = [], [], [], []

    def glyph(c):
        if c in cache:
            return cache[c]
        wc = wc0 * _WIDE.get(c, 1.0)
        raw = []
        for st in FONT[c]:
            closed = len(st) > 2 and st[0] == st[-1]
            pts = st[:-1] if closed else st
            raw.append((np.array([((q[0] - 0.5) * wc, (q[1] - 0.5) * hc) for q in pts]), closed))
        loops = _outline(raw, hw, 0.55 * hw, cell)
        tubes = _tube_strokes(raw, max(0.021, 2.6 * r_tube), 3.9 * r_tube)
        cache[c] = (loops, tubes)
        return cache[c]

    tube_sides = 6 if len(chars) <= 5 else 5        # long texts: slimmer tube section keeps the budget

    def neon_tubes(name, strokes2d, fmap, v0, d_base, rot, material_list, rad, bend):
        """Bent glass tubes on stand-offs above d_base; ends dive back into ceramic boots."""
        for k, P in enumerate(strokes2d):
            pts = [(P[0][0], P[0][1] + v0, d_base + 0.004)] + [(q[0], q[1] + v0, d_base + d_tube) for q in P] + [(P[-1][0], P[-1][1] + v0, d_base + 0.004)]
            pts = _clip_dive(_fillet(pts, bend, closed=False, seg_deg=30.0), d_base + 0.011)      # 0.011 = boot height
            t = sweep(f"{name}_{k}", [fmap(*q) for q in pts], rad, sides=tube_sides, smooth_path=False, cap=False, up=(0.37, 0.6, 0.71))
            material_list.append(t)
            for q in (P[0], P[-1]):
                boots.append((fmap(q[0], q[1] + v0, d_base), rot))
            # glass stand-offs on the long straights
            seg = [float(np.linalg.norm(P[i + 1] - P[i])) for i in range(len(P) - 1)]
            imax = int(np.argmax(seg))
            for i, sl in enumerate(seg):
                if (i == imax and sl > 0.06) or sl > 0.3:
                    npst = max(1, int(round(sl / 0.42)))
                    for j in range(npst):
                        mq = P[i] + (P[i + 1] - P[i]) * ((j + 0.5) / npst)
                        posts.append((fmap(mq[0], mq[1] + v0, d_base), rot))

    for i, (c, zc, rh) in enumerate(tops):
        if c == " ":
            continue
        loops, tubes = glyph(c)
        for s in (-1, 1):
            f = fm(s, T / 2, zc)
            for (outer, holes) in loops:
                letters_floor.append(_prism(f"floor_{i}_{s}", [outer] + holes, f, 0.0, d_floor, sides=False))
                rt = _prism(f"return_{i}_{s}", [outer] + holes, f, 0.0, d_wall, cap1=False)
                smooth(rt, 35)
                letters_walls.append(rt)
            neon_tubes(f"tube_{i}_{s}", tubes, fm(s, T / 2, 0.0), zc, d_floor, (0, 90 * s, 0), tubes_dead if i in dead else tubes_lit,
                       r_tube, r_bend)
    for o in letters_floor:
        assign(o, enamel)
        paint(o, grime=lambda c, n: np.full(len(c), 0.05 + 0.3 * w), wear=chip_mask)
    for o in letters_walls:
        assign(o, trim_m)
        weather(o, 0.9)
    for o in tubes_lit:
        assign(o, neon_m)
    for o in tubes_dead:
        assign(o, dead_m)

    # ---- hairpin neon stripe up the outer edge (both electrodes at the bottom) + zinc rain cap
    ez0 = hatch_z + hatch_h / 2 + 0.05 + nlv * 0.026 + 0.05
    ez1 = H / 2 - 0.09
    if p.edge_tube and ez1 - ez0 > 0.3:
        et = []
        ecol = p.emblem_colour if p.emblem != "none" else p.tube_colour
        neon_tubes("edge_tube", [[np.array(q) for q in ((-0.024, ez0), (-0.024, ez1), (0.024, ez1), (0.024, ez0))]],
                   lambda u, v, d: (u, y_out - d, v), 0.0, 0.0, (90, 0, 0), et, r_tube, 0.022)
        for o in et:
            assign(o, mat("neon", ecol, glow=6))
    cap_t = box("top_cap", (T + 0.014, W - 0.03, 0.009), loc=(0, yc, H / 2 + 0.0035), bevel=0.002, segments=1)
    assign(cap_t, zinc)
    paint(cap_t, grime=0.25 + 0.4 * w, special=0.2 * w)

    # ---- emblem can on top
    em_lights = []
    if p.emblem != "none":
        Te = min(0.11, T * 0.55)
        hwe = 0.036
        eneon = mat("neon", p.emblem_colour, glow=6)
        em_tubes = []
        plinth_h = 0.045
        if p.emblem == "crown":
            cw = max(0.46, W * 1.02)
            unit = [([(0.84, 0.0), (0.16, 0.0), (0.0, 0.6), (0.26, 0.33), (0.5, 0.8), (0.74, 0.33), (1.0, 0.6)], True),
                    ([(0.2, 0.15), (0.8, 0.15)], False)]
            raw = [(np.array([((q[0] - 0.5) * cw, q[1] * cw) for q in pts]), cl) for pts, cl in unit]
            v0 = H / 2 + plinth_h + hwe - 0.004
            tips = [((x - 0.5) * cw, y * cw) for x, y in ((0.0, 0.6), (0.5, 0.8), (1.0, 0.6))]
            e_top = v0 + 0.8 * cw
            pl_w = cw * 0.74
        else:
            R = max(0.25, W * 0.52)
            cen = R * cos(radians(36))
            star = [(R * (1.0 if k % 2 == 0 else 0.42) * sin(radians(180 + k * 36)), cen + R * (1.0 if k % 2 == 0 else 0.42) * cos(radians(180 + k * 36)))
                    for k in range(10)]
            star = star[3:] + star[:3]
            inner = [(q[0] * 0.5, cen + (q[1] - cen) * 0.5) for q in star]
            raw = [(np.array(star), True), (np.array(inner), True)]
            v0 = H / 2 + plinth_h + hwe + 0.05
            tips = []
            e_top = v0 + cen + R
            pl_w = R * 0.9
            postb = box("emblem_post", (Te * 0.7, 0.07, 0.05 + cen * 0.6), loc=(0, yc, H / 2 + plinth_h + (0.05 + cen * 0.6) / 2 - 0.01), bevel=0.003)
            assign(postb, trim_m)
            weather(postb)
            cw = 2 * R
        pl = box("emblem_plinth", (Te + 0.05, pl_w, plinth_h), loc=(0, yc, H / 2 + plinth_h / 2 - 0.002), bevel=0.006)
        assign(pl, trim_m)
        weather(pl)
        eloops = _outline(raw, hwe, 0.0, 0.008)
        for (outer, _holes) in eloops:
            caps = _prism("emblem_skin", [outer], lambda u, v, d: (d, yc + u, v0 + v), -Te / 2, Te / 2, cap0=True, cap1=True, sides=False)
            assign(caps, box_m)
            weather(caps)
            band = _prism("emblem_band", [outer], lambda u, v, d: (d, yc + u, v0 + v), -Te / 2, Te / 2, cap0=False, cap1=False)
            smooth(band, 35)
            assign(band, trim_m)
            weather(band, 0.8)
        estrokes = _tube_strokes(raw, 0.03, 0.034)
        for s in (-1, 1):
            neon_tubes(f"emblem_tube{s}", estrokes, fm(s, Te / 2, 0.0), v0, 0.0, (0, 90 * s, 0), em_tubes, 0.0085, 0.024)
        for o in em_tubes:
            assign(o, eneon)
        for (tu, tv) in tips:
            zz = v0 + tv + hwe
            col = cylinder("jewel_socket", 0.017, 0.026, loc=(0, yc + tu, zz + 0.009), verts=8)
            assign(col, zinc)
            jb = sphere("jewel_bulb", 0.031, loc=(0, yc + tu, zz + 0.045), segments=10, rings=6)
            assign(jb, bulb_m)
        em_lights = [(s * (Te / 2 + 0.45), y_out - 0.1, v0 + (e_top - v0) * 0.5) for s in (-1, 1)]

    # stamp boots / stand-offs now that every tube exists
    bm_ = [trs(loc, rot) for (loc, rot) in boots]
    pm_ = [trs(loc, rot) for (loc, rot) in posts]
    bo = instances_to_mesh("boots", boot, bm_, keep_base=False)
    assign(bo, enamel)
    if pm_:
        po = instances_to_mesh("standoffs", post, pm_, keep_base=False)
        assign(po, enamel)
    else:
        delete([post])

    # ---- chaser bulbs
    if bulb_pos:
        # (the stamped socket sits wholly inside the glowing globe; slimmer, so its wall is 4 mm inside the globe's - CLEARANCE RULE)
        sock = cylinder("socket", 0.0095, 0.012, loc=(0, 0, 0.006), verts=6, cap=False)
        globe = sphere("globe", 0.0165, loc=(0, 0, 0.024), segments=6, rings=3)
        sm = [trs(fm(s)(u, v, fd), (0, 90 * s, 0)) for (s, u, v) in bulb_pos]
        so = instances_to_mesh("bulb_sockets", sock, sm, keep_base=False)
        assign(so, zinc)
        go = instances_to_mesh("bulbs", globe, sm, keep_base=False)
        assign(go, bulb_m)

    # ---- wall brackets: plate + box-section arm + flange + brace
    sep = max(0.07, H / 2 - 0.24)
    hexb = cylinder("hex", 0.0095, 0.008, loc=(0, 0, 0.004), verts=6)
    hexes = []
    arm_s = 0.05
    bh = min(0.24, sep - 0.12)                      # drop of the curved brace along the wall
    braced = bh > 0.08 and gap > 0.15
    plate_h = (0.1 + bh + arm_s / 2) if braced else min(0.2, sep * 1.6)
    rust_i = mask_mul(noise_mask(0.07, seed=p.seed + 5, lo=0.35, hi=0.7), 0.35 + 0.6 * w)
    bar = rounded_rect(0.032, 0.009, 0.002, 1)
    for sz in (1, -1):
        za = sz * sep
        zp_ = za - sz * (plate_h / 2 - 0.06) if braced else za
        wp = box("wall_plate", (0.17, 0.012, plate_h), loc=(0, -0.006, zp_), bevel=0.003)
        arm = box("arm", (arm_s, gap + 0.004, arm_s), loc=(0, -(gap + 0.004) / 2, za), bevel=0.004)
        fl = box("arm_flange", (0.125, 0.01, 0.125), loc=(0, -gap + 0.005, za), bevel=0.003)
        parts_ = [wp, arm, fl]
        for sx in (-1, 1):
            for sv in (-1, 1):
                hexes.append(trs((sx * 0.062, -0.012, zp_ + sv * (plate_h / 2 - 0.028)), (90, 0, rng.u(0, 60))))
                hexes.append(trs((sx * 0.046, -gap + 0.01, za + sv * 0.046), (-90, 0, rng.u(0, 60)), 0.8))
        if braced:
            # forged quarter-round brace with a scroll in the corner, as on old shop-sign brackets
            a_ = gap - 0.03
            ky, kz = -0.012, za - sz * arm_s / 2
            arc = [(0.0, ky - a_ * sin(radians(t)), kz - sz * bh * cos(radians(t))) for t in np.linspace(0, 90, 10)]
            parts_.append(sweep("arm_brace", arc, bar, smooth_path=False, up=(1, 0, 0)))
            rc = 0.3 * min(a_, bh)
            sc = spiral_path(0.007, rc, 1.75, n=30, a0_deg=rng.u(0, 40) + (200 if sz > 0 else 20), plane="YZ",
                             center=(0, ky - 0.42 * a_, kz - sz * 0.42 * bh), ccw=sz > 0)
            parts_.append(sweep("arm_scroll", sc, bar, scale_fn=lambda t: 0.6 + 0.4 * t, up=(1, 0, 0)))
        for o in parts_:
            assign(o, iron)
            paint(o, special=rust_i)

    # ---- stay rods with turnbuckles
    eye_parts = []
    y_l = -(gap + W * 0.8)
    ax = T / 2 + 0.3
    rise = 0.2 + 0.62 * (gap + W * 0.8)
    for sz in ((1, -1) if int(p.stays) == 4 else (1,)):
        for sx in (-1, 1):
            a = np.array((sx * (T / 2 + fd + 0.016), y_l, sz * (H / 2 - 0.014 - frame_w / 2)))
            b = np.array((sx * ax, -0.03, sz * (H / 2 + rise)))
            dv = (b - a) / np.linalg.norm(b - a)
            rod = sweep("stay", [tuple(a), tuple(b)], 0.0048, sides=6, smooth_path=False)
            tb0 = a + (b - a) * 0.3
            tbk = sweep("turnbuckle", [tuple(tb0 - dv * 0.06), tuple(tb0 - dv * 0.045), tuple(tb0 + dv * 0.045), tuple(tb0 + dv * 0.06)],
                        0.0105, sides=6, smooth_path=False, scale_fn=lambda t: 0.62 if (t < 0.05 or t > 0.95) else 1.0)
            flat(tbk)
            for o in (rod, tbk):
                assign(o, iron)
            for (q, axis_rot) in ((a, (0, 90, 0)), (b, (90, 0, 0))):
                eye_parts.append(torus("stay_eye", 0.013, 0.0042, loc=tuple(q), rot=(90, 0, 90 if axis_rot[1] else 0), major_seg=10, minor_seg=5))
            lug = cylinder("lug", 0.009, 0.014, loc=(sx * (T / 2 + fd + 0.005), y_l, a[2]), rot=(0, 90, 0), verts=6)
            ap = box("anchor_plate", (0.085, 0.008, 0.085), loc=(b[0], -0.004, b[2]), bevel=0.002)
            an = cylinder("anchor_stud", 0.008, 0.03, loc=(b[0], -0.017, b[2]), rot=(90, 0, 0), verts=6)
            for o in (lug, ap, an):
                assign(o, iron)
            for cx, cz in ((-1, -1), (1, 1)):
                hexes.append(trs((b[0] + cx * 0.028, -0.008, b[2] + cz * 0.028), (90, 0, rng.u(0, 60)), 0.65))
    for o in eye_parts:
        assign(o, iron)
    hx = instances_to_mesh("bolts", hexb, hexes, keep_base=False)
    assign(hx, iron)
    paint(hx, special=0.3 + 0.5 * w)

    rv = instances_to_mesh("rivets", rivet, rivets, keep_base=False)
    assign(rv, box_m)
    weather(rv, 1.6)

    # ---- cabling: wall junction box, rigid conduit down the wall, flexible feed into the box edge
    zj = sep - 0.03
    xj = 0.2
    jb = box("junction_box", (0.1, 0.055, 0.13), loc=(xj, -0.0275, zj), bevel=0.004)
    lid = box("junction_lid", (0.086, 0.006, 0.116), loc=(xj, -0.057, zj), bevel=0.002)
    Lc = max(0.25, min(0.9, H * 0.4))
    pipe = sweep("conduit", _fillet([(xj, -0.02, zj - 0.06), (xj, -0.02, zj - 0.065 - Lc), (xj, -0.002, zj - 0.065 - Lc - 0.03)], 0.02),
                 0.0105, sides=8, smooth_path=False, up=(1, 0, 0))
    gl0 = cylinder("gland", 0.015, 0.022, loc=(xj, -0.02, zj - 0.072), verts=8)
    gl1 = cylinder("gland", 0.014, 0.024, loc=(xj, -0.066, zj + 0.02), rot=(90, 0, 0), verts=8)
    gl2 = cylinder("gland", 0.014, 0.024, loc=(0.055, -gap + 0.01, zj - 0.115), rot=(90, 0, 0), verts=8)
    for o in (jb, lid, pipe, gl0, gl1, gl2):
        assign(o, zinc)
        paint(o, special=0.15 + 0.5 * w, grime=0.2 + 0.3 * w)
    for k in range(2):
        zc_ = zj - 0.065 - Lc * (0.3 + 0.55 * k)
        clip = box("conduit_clip", (0.06, 0.005, 0.02), loc=(xj, -0.031, zc_), bevel=0.0015, segments=1)
        assign(clip, zinc)
    g = min(gap, 0.5)
    flex = sweep("flex_feed", bezier_path((xj, -0.07, zj + 0.02), (xj, -0.07 - 0.45 * g, zj - 0.02 - 0.3 * g),
                                          (0.055, -gap + 0.55 * g, zj - 0.14 - 0.35 * g), (0.055, -gap + 0.012, zj - 0.115), n=12),
                 0.0075, sides=7, smooth_path=False, up=(1, 0, 0))
    assign(flex, iron)

    # ---- lights: letters in the tube colour, emblem in its own colour, optional warm bulb wash
    n_lit = max(1, len(lit) - len(dead))
    frac = n_lit / max(1, len(lit))
    per_face = float(p.intensity) * max(H, 0.6) * frac
    slots = 6 - len(em_lights) - (2 if p.bulbs else 0)
    n_l = 2 if (H > 1.5 and slots >= 4) else 1
    for s in (-1, 1):
        for k in range(n_l):
            zz = (H * (0.24 if k == 0 else -0.24)) if n_l == 2 else 0.0
            light(f"neon_{'r' if s > 0 else 'l'}{k}", (s * (T / 2 + 0.45), y_out - 0.15, zz), color=p.tube_colour,
                  intensity=per_face / n_l * (0.62 if p.bulbs else 1.0), range=7.0, flicker=0.04, shadow_near=0.6)
    for k, loc in enumerate(em_lights):
        light(f"emblem_{k}", loc, color=p.emblem_colour, intensity=1.6, range=5.0, flicker=0.04, shadow_near=0.5)
    if p.bulbs:
        for s in (-1, 1):
            light(f"bulbs_{'r' if s > 0 else 'l'}", (s * (T / 2 + 0.55), y_out - 0.15, -H * 0.05), color="#ffc98a", intensity=0.4 * max(H, 0.6),
                  range=6.0, flicker=0.02, shadow_near=0.5)
    socket("wall_fix_top", "attach", (0, 0, sep), role="bracket")
    socket("wall_fix_bottom", "attach", (0, 0, -sep), role="bracket")
