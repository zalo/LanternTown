"""Inscribed commemorative wall plaque. A cast-bronze plate (raised, drafted letters rubbed bright on a dark
patinated field, ogee frame) or a stone / marble / slate tablet (V-cut letters filled with gilt, lead or
dark paint, sunk field with a moulded border) held to a dressed-stone backing slab by four rosette-headed
bolts, with a small relief emblem (crown, star, key or laurel wreath). Run-off from the plate stains the
stone below in thin streaks; a brass-shaded gooseneck lamp (or votive candles) lights the inscription.
The plate frame is lofted from one outline through a moulding profile (true mitres on every shape); the
field, the inscription and the slab face are constrained Delaunay triangulations of the real glyph / stain
outlines, so letters are cut into (or cast onto) the surface without booleans.
"""
from tk import *
import math
import bmesh
import bpy
import numpy as np
from mathutils.geometry import delaunay_2d_cdt

META = {"name": "Inscribed wall plaque", "class": "prop_small", "mount": "wall", "mount_height": 1.55,
        "tri_budget": 9000,            # multi-line 3D lettering is the whole point of the object
        "focus": 0.5}
PARAMS = {
    "width": {"default": 0.9, "min": 0.4, "max": 2.2, "desc": "plate width (m)"},
    "height": {"default": 0.6, "min": 0.3, "max": 1.5, "desc": "plate height (m)"},
    "material": {"default": "bronze", "choices": ["bronze", "stone", "marble", "slate"], "desc": "plate material"},
    "shape": {"default": "rect", "choices": ["rect", "arched", "cut_corner"], "desc": "plate outline"},
    "frame": {"default": 1.0, "min": 0.6, "max": 1.6, "desc": "moulded frame width / relief multiplier"},
    "line1": {"default": "Let every lantern", "desc": "inscription line 1 (empty = skipped)"},
    "line2": {"default": "light the way home.", "desc": "inscription line 2"},
    "line3": {"default": "", "desc": "inscription line 3"},
    "line4": {"default": "", "desc": "inscription line 4"},
    "footer": {"default": "", "desc": "small closing line (date, dedication)"},
    "text_size": {"default": 0.078, "min": 0.02, "max": 0.16, "desc": "letter size (m, em); shrinks automatically to fit"},
    "font": {"default": "serif", "choices": ["serif", "serif_regular", "roman", "noto_serif", "script", "sans"], "desc": "letter face"},
    "lettering": {"default": "auto", "choices": ["auto", "raised", "engraved", "inlaid"],
                  "desc": "auto = raised (cast) on bronze, V-cut in stone, flush inlaid when the fill is lead"},
    "letter_fill": {"default": "gilt", "choices": ["gilt", "lead", "dark"], "desc": "infill of cut letters"},
    "emblem": {"default": "crown", "choices": ["crown", "star", "key", "wreath", "none"], "desc": "relief emblem"},
    "emblem_pos": {"default": "below", "choices": ["below", "above"], "desc": "emblem under or over the inscription"},
    "emblem_size": {"default": 0.11, "min": 0.04, "max": 0.4, "desc": "emblem height (m)"},
    "backing": {"default": "slab", "choices": ["slab", "tablet", "none"], "desc": "stone backing: plain slab, slab with cornice and sill, or none"},
    "margin": {"default": 0.06, "min": 0.03, "max": 0.16, "desc": "backing slab margin around the plate (m)"},
    "backing_tint": {"default": "#a79b86", "desc": "backing stone colour"},
    "stain": {"default": 0.65, "min": 0.0, "max": 1.0, "desc": "water / verdigris staining on and below the plate"},
    "lamp": {"default": "gooseneck", "choices": ["gooseneck", "candles", "none"],
             "desc": "brass-shaded gooseneck lamp over the plaque, votive candles on a shelf / the sill below, or unlit"},
    "intensity": {"default": 2.2, "min": 0.5, "max": 12.0, "desc": "lamp intensity (cd); the candle group burns at a fraction of it"},
}
VARIANTS = {
    "marble_memorial": {"material": "marble", "shape": "arched", "width": 0.72, "height": 0.95, "backing": "tablet",
                        "line1": "IN MEMORY OF", "line2": "THE LAMPLIGHTERS", "line3": "WHO KEPT THE", "line4": "OLD QUARTER BRIGHT",
                        "footer": "MDCCCXLVII", "font": "roman", "text_size": 0.05, "letter_fill": "lead", "emblem": "wreath",
                        "emblem_pos": "above", "emblem_size": 0.16, "lamp": "candles", "stain": 0.8, "seed": 4},
    "stone_wide": {"material": "stone", "shape": "cut_corner", "width": 1.7, "height": 0.7, "line1": "RAISED BY THE TOWNSFOLK",
                   "line2": "IN GRATITUDE", "line3": "", "footer": "ANNO MDCCCLXII", "font": "roman", "text_size": 0.085,
                   "emblem": "star", "emblem_pos": "above", "emblem_size": 0.14, "margin": 0.08, "backing_tint": "#8f8676", "seed": 9},
    "slate_small": {"material": "slate", "width": 0.52, "height": 0.44, "line1": "Here lived", "line2": "A. Merle", "line3": "Clockmaker",
                    "font": "script", "text_size": 0.075, "emblem": "key", "emblem_size": 0.05, "margin": 0.04, "backing_tint": "#a39a88",
                    "stain": 0.4, "seed": 14},
    "bronze_works": {"material": "bronze", "shape": "cut_corner", "width": 0.62, "height": 0.46, "line1": "CANAL QUARTER",
                     "line2": "WATER COMPANY", "footer": "ESTD 1887", "font": "sans", "text_size": 0.05, "emblem": "none",
                     "frame": 1.3, "backing": "none", "lamp": "candles", "stain": 1.0, "seed": 23},
}


# ------------------------------------------------------------------ 2D outline helpers (x, z seen from the front, CCW)
def _outline(W, H, shape, rc, rise, chip=None):
    hw, hh = W / 2, H / 2
    if shape == "cut_corner":
        pts = []
        for (cx, cz, t0) in ((-hw, -hh, 90), (hw, -hh, 180), (hw, hh, 270), (-hw, hh, 360)):
            for i in range(7):
                t = math.radians(t0 - 90 * i / 6)
                pts.append((cx + rc * math.cos(t), cz + rc * math.sin(t)))
        return pts
    bl, br = [(-hw, -hh)], [(hw, -hh)]
    if chip:
        k, a, b = chip
        if k == 0:
            bl = [(-hw, -hh + a), (-hw + b, -hh)]
        else:
            br = [(hw - b, -hh), (hw, -hh + a)]
    if shape == "arched":
        R = (hw * hw + rise * rise) / (2 * rise)
        zc = hh - R
        a0 = math.asin(min(hw / R, 1.0))
        top = [(R * math.sin(a0 - 2 * a0 * i / 18), zc + R * math.cos(a0 - 2 * a0 * i / 18)) for i in range(19)]
    else:
        top = [(hw, hh), (-hw, hh)]
    return bl + br + top


def _resample(pts, step):
    out, n = [], len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        out.append(a)
        k = int(math.hypot(b[0] - a[0], b[1] - a[1]) / step)
        for j in range(1, k + 1):
            t = j / (k + 1)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _offset(pts, d):
    """Inward mitred offset of a CCW polygon."""
    if abs(d) < 1e-9:
        return list(pts)
    out, n = [], len(pts)
    for i in range(n):
        x0, z0 = pts[i - 1]
        x1, z1 = pts[i]
        x2, z2 = pts[(i + 1) % n]
        l1 = math.hypot(x1 - x0, z1 - z0) or 1e-9
        l2 = math.hypot(x2 - x1, z2 - z1) or 1e-9
        n1 = (-(z1 - z0) / l1, (x1 - x0) / l1)
        n2 = (-(z2 - z1) / l2, (x2 - x1) / l2)
        k = max(1.0 + n1[0] * n2[0] + n1[1] * n2[1], 0.4)
        out.append((x1 + (n1[0] + n2[0]) * d / k, z1 + (n1[1] + n2[1]) * d / k))
    return out


def _grow(lp, e):
    """Offset a loop whose solid lies on its LEFT (outer contours CCW, holes CW) outwards by e."""
    a = np.asarray(lp, dtype=float)
    d1 = a - np.roll(a, 1, axis=0)
    d2 = np.roll(a, -1, axis=0) - a
    d1 /= np.maximum(np.hypot(d1[:, 0], d1[:, 1]), 1e-12)[:, None]
    d2 /= np.maximum(np.hypot(d2[:, 0], d2[:, 1]), 1e-12)[:, None]
    n1 = np.stack([d1[:, 1], -d1[:, 0]], axis=1)
    n2 = np.stack([d2[:, 1], -d2[:, 0]], axis=1)
    k = np.maximum(1.0 + (n1 * n2).sum(1), 0.5)
    return a + (n1 + n2) * (e / k)[:, None]


def _winding(P, loops):
    """Winding number of each point of P (N, 2) with respect to a set of closed loops."""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    w = np.zeros(len(P), dtype=int)
    x, y = P[:, 0], P[:, 1]
    for lp in loops:
        lp = np.asarray(lp, dtype=float)
        mn, mx = lp.min(0), lp.max(0)
        idx = np.nonzero((x >= mn[0]) & (x <= mx[0]) & (y >= mn[1]) & (y <= mx[1]))[0]
        if not len(idx):
            continue
        px, py = x[idx][:, None], y[idx][:, None]
        x1, y1 = lp[:, 0][None], lp[:, 1][None]
        x2, y2 = np.roll(lp[:, 0], -1)[None], np.roll(lp[:, 1], -1)[None]
        cr = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
        w[idx] += ((y1 <= py) & (y2 > py) & (cr > 0)).sum(1) - ((y1 > py) & (y2 <= py) & (cr < 0)).sum(1)
    return w


def _seg_dist(P, loops):
    """Distance of each point of P (N, 2) to the nearest edge of any loop."""
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    d = np.full(len(P), 1e9)
    for lp in loops:
        a = np.asarray(lp, dtype=float)
        ab = np.roll(a, -1, axis=0) - a
        l2 = np.maximum((ab ** 2).sum(1), 1e-18)
        ap = P[:, None, :] - a[None]
        t = np.clip((ap * ab[None]).sum(2) / l2[None], 0.0, 1.0)
        q = ap - t[..., None] * ab[None]
        d = np.minimum(d, np.sqrt((q ** 2).sum(2)).min(1))
    return d


def _cdt(loops, points=()):
    """Constrained Delaunay triangulation of closed loops + free points -> (verts (N,2), tris (M,3) CCW, centroids)."""
    vs, es = [], []
    for lp in loops:
        b, n = len(vs), len(lp)
        vs += [Vector((float(q[0]), float(q[1]))) for q in lp]
        es += [(b + i, b + (i + 1) % n) for i in range(n)]
    vs += [Vector((float(q[0]), float(q[1]))) for q in points]
    out = delaunay_2d_cdt(vs, es, [], 0, 1e-6)
    V = np.array([(v.x, v.y) for v in out[0]], dtype=float).reshape(-1, 2)
    T = np.array([f for f in out[2] if len(f) == 3], dtype=int).reshape(-1, 3)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    T[area < 0] = T[area < 0][:, ::-1]
    ok = np.abs(area) > 1e-13
    return V, T[ok], ((a + b + c) / 3.0)[ok]


def _grid_pts(x0, x1, z0, z1, g, rng):
    nx, nz = max(1, int(round((x1 - x0) / g))), max(1, int(round((z1 - z0) / g)))
    return [(x0 + (x1 - x0) * (i + (0.5 if j % 2 else 0.0)) / nx + rng.jitter(0, 0.18 * g), z0 + (z1 - z0) * j / nz + rng.jitter(0, 0.18 * g))
            for j in range(nz + 1) for i in range(nx + 1)]


def _mat_slots(o, mats, idx):
    o.data.materials.clear()
    for mm in mats:
        o.data.materials.append(mm)
    o.data.polygons.foreach_set("material_index", [int(i) for i in idx])
    o.data.update()


def _const(arr):
    """paint() callable returning a precomputed per-vertex array (falls back to 0 if the vertex count changed)."""
    arr = np.asarray(arr, dtype=float)
    return lambda co, n: arr if len(co) == len(arr) else np.zeros(len(co))


# ------------------------------------------------------------------ lettering: glyph outlines -> cast or V-cut geometry
def _text_loops(s, size, font, spacing=1.0):
    """Outline loops of a text line in line-local (x, up) coordinates; solid on the left of every loop."""
    o = text("ln", s, size=size, depth=0.002, font=font, spacing=spacing)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bm.normal_update()
    cap = set(f for f in bm.faces if f.normal.z > 0.5)
    nxt = {}
    for f in cap:
        for l in f.loops:
            if sum(1 for g in l.edge.link_faces if g in cap) == 1:
                nxt.setdefault(l.vert.index, []).append(l.link_loop_next.vert.index)
    co = {v.index: (v.co.x, v.co.y) for v in bm.verts}
    bm.free()
    delete(o)
    loops = []
    while nxt:
        start = next(iter(nxt))
        cur, lp = start, []
        for _ in range(100000):
            lp.append(co[cur])
            outs = nxt.get(cur)
            if not outs:
                break
            nx = outs.pop()
            if not outs:
                del nxt[cur]
            cur = nx
            if cur == start:
                break
        if len(lp) >= 3:
            loops.append(np.array(lp, dtype=float))
    return loops


def _dp(a, tol):
    n = len(a)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p, d = a[i], a[j] - a[i]
        L = math.hypot(d[0], d[1])
        seg = a[i + 1:j]
        if L < 1e-12:
            dist = np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            k += i + 1
            keep[k] = True
            stack += [(i, k), (k, j)]
    return a[keep]


def _simplify_loop(lp, tol):
    a = np.asarray(lp, dtype=float)
    if len(a) <= 4:
        return a
    a = np.roll(a, -int(np.argmin(a[:, 0] + 0.37 * a[:, 1])), axis=0)
    i1 = int(np.argmax(np.hypot(a[:, 0] - a[0, 0], a[:, 1] - a[0, 1])))
    if i1 < 1 or i1 >= len(a) - 1:
        return a
    h1, h2 = _dp(a[:i1 + 1], tol), _dp(np.vstack([a[i1:], a[:1]]), tol)
    return np.vstack([h1[:-1], h2[:-1]])


def _letter_tris(loops):
    V, T, cen = _cdt(loops)
    return V, T[_winding(cen, loops) >= 1]


def _prep_loops(loops, tol, minw):
    """Simplify the outlines and embolden the line until its hairlines are at least `minw` wide (cuttable / castable)."""
    out = [_simplify_loop(lp, tol) for lp in loops]
    out = [lp for lp in out if len(lp) >= 3]
    if not out:
        return out
    V, T = _letter_tris(out)
    if len(T):
        e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        ue = ue[cnt == 2]
        if len(ue) > 4:
            hair = float(np.percentile(np.hypot(*(V[ue[:, 0]] - V[ue[:, 1]]).T), 12))
            grow = clamp((minw - hair) / 2, 0.0, 0.0014)
            if grow > 3e-5:
                out = [_grow(lp, grow) for lp in out]
    return out


def _chordal(V, T, isl, y0, vdepth, slope=0.95, sign=1.0):
    """Triangles of a triangulation as 3D geometry at depth y0. Triangles flagged `isl` (inside letters) are split
    along the chordal axis of the stroke and their axis points moved by sign * min(vdepth, slope * half stroke
    width): a V-cut (sign +1, into the plate) or a softly ridged top (sign -1). Returns (verts, faces, fm)."""
    verts = [(float(x), y0, float(z)) for x, z in V]
    faces, fm = [], []
    for t in T[~isl]:
        faces.append((int(t[0]), int(t[1]), int(t[2])))
        fm.append(0)
    LT = T[isl]
    if vdepth < 2e-4:
        for t in LT:
            faces.append((int(t[0]), int(t[1]), int(t[2])))
            fm.append(1)
        return verts, faces, fm
    cnt = {}
    for t in LT:
        for k in range(3):
            e = (min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3]))
            cnt[e] = cnt.get(e, 0) + 1
    mid = {}

    def M(i, j):
        e = (min(i, j), max(i, j))
        if cnt[e] < 2:
            return None
        if e not in mid:
            L = math.hypot(V[i, 0] - V[j, 0], V[i, 1] - V[j, 1])
            mid[e] = len(verts)
            verts.append(((V[i, 0] + V[j, 0]) / 2, y0 + sign * min(vdepth, slope * L / 2), (V[i, 1] + V[j, 1]) / 2))
        return mid[e]

    def add(*tris):
        for q in tris:
            faces.append(q)
            fm.append(1)

    for t in LT:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for _ in range(3):                           # rotate until the edge pattern is canonical
            ms = (M(a, b), M(b, c), M(c, a))
            k = sum(1 for q in ms if q is not None)
            if k in (0, 3) or (k == 2 and ms[0] is None) or (k == 1 and ms[2] is not None):
                break
            a, b, c = b, c, a
        mab, mbc, mca = ms
        if k == 3:
            add((a, mab, mca), (b, mbc, mab), (c, mca, mbc), (mab, mbc, mca))
        elif k == 2:
            add((a, b, mbc), (a, mbc, mca), (mca, mbc, c))
        elif k == 1:
            add((a, b, mca), (b, c, mca))
        else:
            L = min(math.hypot(V[a, 0] - V[b, 0], V[a, 1] - V[b, 1]), math.hypot(V[b, 0] - V[c, 0], V[b, 1] - V[c, 1]))
            g = len(verts)
            verts.append(((V[a, 0] + V[b, 0] + V[c, 0]) / 3, y0 + sign * min(vdepth, slope * L / 3), (V[a, 1] + V[b, 1] + V[c, 1]) / 3))
            add((a, b, g), (b, c, g), (c, a, g))
    return verts, faces, fm


def _field(name, V, T, isl, yf, vdepth):
    """Field face; letter triangles become a clean V-cut (or stay flush when vdepth = 0)."""
    verts, faces, fm = _chordal(V, T, isl, yf, vdepth)
    return mesh_object(name, verts, faces), fm


def _raised(name, groups, yf, depth, ridge):
    """Cast letters: softly ridged, rubbed top `depth` proud of the field, drafted sides flaring out to the root.
    groups = [(loops, draft)]. Returns (object, per-face material index: 0 = rubbed top, 1 = side)."""
    verts, faces, fm = [], [], []
    for loops, draft in groups:
        V, T = _letter_tris(loops)
        b0 = len(verts)
        v2, f2, _ = _chordal(V, T, np.ones(len(T), dtype=bool), yf - depth, ridge, slope=0.4, sign=-1.0)
        verts += v2
        faces += [tuple(i + b0 for i in f) for f in f2]
        fm += [0] * len(f2)
        for lp in loops:
            n, b = len(lp), len(verts)
            root = _grow(lp, draft)
            verts += [(x, yf - depth, z) for x, z in lp] + [(x, yf + 0.0006, z) for x, z in root]
            faces += [(b + i, b + n + i, b + n + (i + 1) % n, b + (i + 1) % n) for i in range(n)]
            fm += [1] * n
    return mesh_object(name, verts, faces), fm


def _loft(name, outline, prof, y0):
    """Rings of `outline` inset by prof[i][0] at depth y0 - prof[i][1] (open: the field closes the front)."""
    n = len(outline)
    verts, faces = [], []
    for d, h in prof:
        verts += [(x, y0 - h, z) for x, z in _offset(outline, d)]
    for i in range(len(prof) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            faces.append((a, b, b + n, a + n))
    return mesh_object(name, verts, faces)


def _prism(name, sec, x0, x1):
    """Straight moulding: (y, z) section extruded from x0 to x1, closed ends."""
    n = len(sec)
    verts = [(x0, y, z) for y, z in sec] + [(x1, y, z) for y, z in sec]
    faces = [(j, (j + 1) % n, (j + 1) % n + n, j + n) for j in range(n)]
    faces += [tuple(range(n)), tuple(reversed(range(n, 2 * n)))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


# ------------------------------------------------------------------ emblems (unit height, centred, base on y = 0, relief to -Y)
def _emblem(kind, rng):
    objs = []
    if kind == "crown":
        tips = [(-0.62, 0.78), (-0.31, 0.9), (0.0, 1.0), (0.31, 0.9), (0.62, 0.78)]
        body = [(-0.5, 0.0), (0.5, 0.0), (0.52, 0.2), (0.62, 0.78), (0.45, 0.44), (0.31, 0.9), (0.155, 0.44), (0.0, 1.0),
                (-0.155, 0.44), (-0.31, 0.9), (-0.45, 0.44), (-0.62, 0.78), (-0.52, 0.2)]
        body = [(x, z - 0.5) for x, z in body]
        objs.append(shape_extrude("crown_body", body, depth=0.05, rot=(90, 0, 0), bevel=0.012))
        objs.append(box("crown_band", (1.08, 0.03, 0.13), loc=(0, -0.058, -0.4), bevel=0.012, segments=1))
        objs.append(box("crown_rim", (1.12, 0.022, 0.035), loc=(0, -0.07, -0.3), bevel=0.008, segments=1))
        for x, z in tips:
            objs.append(sphere("crown_ball", 0.055, loc=(x, -0.035, z - 0.5), segments=6, rings=4))
        for k in range(5):
            objs.append(sphere("crown_gem", 0.036, loc=(-0.4 + 0.2 * k, -0.07, -0.41), segments=6, rings=4,
                               scale=(1.0, 0.6, 1.2 if k % 2 == 0 else 0.9)))
        aspect = 1.36
    elif kind == "star":
        R = 0.43
        verts = [(0.0, -0.13, 0.0)]
        for k in range(10):
            a = math.radians(90 + k * 36)
            r = R if k % 2 == 0 else R * 0.4
            verts.append((r * math.cos(a), -0.012 if k % 2 == 0 else -0.03, r * math.sin(a)))
        for k in range(10):
            a = math.radians(90 + k * 36)
            r = (R if k % 2 == 0 else R * 0.4) + 0.012
            verts.append((r * math.cos(a), 0.0, r * math.sin(a)))
        faces = [(0, 1 + k, 1 + (k + 1) % 10) for k in range(10)]
        faces += [(1 + k, 11 + k, 11 + (k + 1) % 10, 1 + (k + 1) % 10) for k in range(10)]
        st = mesh_object("star", verts, faces)
        flat(st)
        objs.append(st)
        objs.append(torus("star_ring", major=0.465, minor=0.035, loc=(0, -0.022, 0), rot=(90, 0, 0), major_seg=28, minor_seg=6))
        aspect = 1.0
    elif kind == "key":
        bow = shape_extrude("key_bow", circle_pts(0.46, 18, center=(-0.86, 0)), [list(reversed(circle_pts(0.24, 12, center=(-0.86, 0))))],
                            depth=0.1, rot=(90, 0, 0), bevel=0.03)
        objs.append(bow)
        objs.append(cylinder("key_shank", 0.075, 1.72, loc=(0.46, -0.05, 0), rot=(0, 90, 0), verts=8))
        for x in (-0.33, -0.2, 1.2):
            objs.append(cylinder("key_collar", 0.115, 0.07, loc=(x, -0.055, 0), rot=(0, 90, 0), verts=8, bevel=0.015, segments=1))
        bit = [(0.72, -0.06), (0.72, -0.46), (0.86, -0.46), (0.86, -0.3), (0.94, -0.3), (0.94, -0.46), (1.1, -0.46), (1.1, -0.06)]
        objs.append(shape_extrude("key_bit", bit, depth=0.07, rot=(90, 0, 0), bevel=0.015))
        objs.append(sphere("key_tip", 0.09, loc=(1.32, -0.05, 0), segments=8, rings=5))
        aspect = 2.75
    elif kind == "wreath":
        L, w, t = 0.2, 0.05, 0.03
        leaf = mesh_object("leaf", [(0, 0, 0), (w, 0, L * 0.42), (0, 0, L), (-w, 0, L * 0.42), (0, -t, L * 0.4)],
                           [(0, 4, 1), (1, 4, 2), (2, 4, 3), (3, 4, 0)])
        recalc_normals(leaf)
        flat(leaf)
        R, ms, nl = 0.4, [], 9
        for sx in (-1, 1):
            for i in range(nl):
                a = math.radians(-78 + i * 150.0 / (nl - 1))
                for side in (-1, 1):
                    rr = R + side * 0.035
                    dx, dz = -math.sin(a), math.cos(a)
                    splay = math.radians(34 * side + rng.jitter(0, 6))
                    ddx = dx * math.cos(splay) + dz * math.sin(splay)
                    ddz = -dx * math.sin(splay) + dz * math.cos(splay)
                    th = math.degrees(math.atan2(sx * ddx, ddz))
                    sc = (0.78 if i == nl - 1 else 1.0) * rng.vary(1.0, 0.1)
                    ms.append(trs((sx * rr * math.cos(a), -0.004 * (i % 2), rr * math.sin(a)), (0, th, 0), sc))
        objs.append(instances_to_mesh("leaves", leaf, ms, keep_base=False))
        for sx in (-1, 1):
            path = [(sx * R * math.cos(math.radians(-84 + k * 21)), -0.012, R * math.sin(math.radians(-84 + k * 21))) for k in range(8)]
            objs.append(sweep("stem", path, 0.014, sides=4))
        objs.append(sphere("knot", 0.05, loc=(0, -0.03, -R - 0.005), segments=8, rings=5, scale=(1.4, 0.8, 0.9)))
        for sx in (-1, 1):
            objs.append(sweep("ribbon", [(sx * 0.02, -0.03, -R), (sx * 0.1, -0.02, -R - 0.05), (sx * 0.17, -0.012, -R - 0.02)],
                              rounded_rect(0.045, 0.014, 0.004, 1), up=(0, 1, 0)))
        aspect = 1.05
    else:
        return None, 0.0
    return join(objs, "emblem"), aspect


# ------------------------------------------------------------------ backing slab, lamp, candles


def _streak_poly(x, zt, L, w, rng):
    """Thin tapering run-off streak hanging from (x, zt): CCW polygon."""
    k = max(3, min(6, int(L / 0.022)))
    drift = rng.jitter(0, 0.004 + 0.03 * L)
    left, right = [], []
    for i in range(k):
        t = i / k
        cx = x + drift * t * t + (rng.jitter(0, 0.0012) if i else 0.0)
        h = 0.5 * w * (1.0 - t) ** 0.55 * (rng.u(0.8, 1.15) if i else 1.0)
        left.append((cx - h, zt - L * t))
        right.append((cx + h, zt - L * t))
    core = [((left[i][0] + right[i][0]) / 2, left[i][1] - 0.003) for i in range(1, k)]
    return left + [(x + drift, zt - L)] + right[::-1], core


def _slab(name, x0, x1, z0, z1, tb, c, hidden, streaks, rng, seed):
    """Dressed stone slab: front face triangulated around the plate (hidden part left out) with the run-off
    streaks as their own faces, a tooled chamfer with knocked corners / chips, and the returns to the wall.
    Returns (object, per-face material index, per-vertex 'on a streak' weights)."""
    step = clamp((x1 - x0) / 20, 0.042, 0.07)
    O = _resample([(x0, z0), (x1, z0), (x1, z1), (x0, z1)], step)
    n = len(O)
    ext = [0.0] * n
    corners = [i for i, q in enumerate(O) if q[0] in (x0, x1) and q[1] in (z0, z1)]
    hits = [(rng.pick(corners), rng.u(0.016, 0.028), 1)] + [(int(rng.u(0, n)) % n, rng.u(0.007, 0.02), rng.pick([1, 1, 2])) for _ in range(5 + n // 20)]
    for k, dpt, wd in hits:
        for j in range(-wd, wd + 1):
            i = (k + j) % n
            ext[i] = max(ext[i], dpt * (1.0 - abs(j) / (wd + 1.0)) * rng.u(0.7, 1.0))
    F0 = _offset(O, c)
    F = []
    for i in range(n):
        dx, dz = F0[i][0] - O[i][0], F0[i][1] - O[i][1]
        L = math.hypot(dx, dz) or 1e-9
        F.append((F0[i][0] + dx / L * ext[i] * 1.3, F0[i][1] + dz / L * ext[i] * 1.3))
    g = step * 1.15
    spolys = [np.array(q[0], dtype=float) for q in streaks]
    cores = np.array([c_ for q in streaks for c_ in q[1]], dtype=float).reshape(-1, 2)
    guard = [np.array(F)] + ([np.array(hidden)] if hidden else []) + spolys
    P = np.array(_grid_pts(x0, x1, z0, z1, g, rng))
    keep = (_winding(P, [np.array(F)]) >= 1) & (_seg_dist(P, guard) > 0.38 * g)
    if hidden:
        keep &= _winding(P, [np.array(hidden)]) < 1
    if spolys:
        keep &= _winding(P, spolys) < 1
    V, T, cen = _cdt(guard, np.vstack([P[keep], cores]))
    ok = _winding(cen, [np.array(F)]) >= 1
    if hidden:
        ok &= _winding(cen, [np.array(hidden)]) < 1
    T, cen = T[ok], cen[ok]
    iss = (_winding(cen, spolys) >= 1) if spolys else np.zeros(len(T), dtype=bool)
    n1 = noise_mask(0.16, seed=seed, lo=0.0, hi=1.0)
    n2 = noise_mask(0.045, seed=seed + 1, lo=0.0, hi=1.0)

    def yface(q):
        q = np.asarray(q, dtype=float).reshape(-1, 2)
        co = np.stack([q[:, 0], np.zeros(len(q)), q[:, 1]], axis=1)
        return np.minimum(-tb + (n1(co, co) - 0.5) * 0.006 + (n2(co, co) - 0.5) * 0.0024, -0.0006)
    yv = yface(V)
    verts = [(float(V[i, 0]), float(yv[i]), float(V[i, 1])) for i in range(len(V))]
    faces = [(int(t[0]), int(t[1]), int(t[2])) for t in T]
    fm = [1 if s else 0 for s in iss]
    b = len(verts)
    yF = yface(F)
    verts += [(F[i][0], float(yF[i]), F[i][1]) for i in range(n)]
    verts += [(O[i][0] + rng.jitter(0, 0.001), min(-tb + c + ext[i], -0.004), O[i][1] + rng.jitter(0, 0.001)) for i in range(n)]
    Bk = _offset(O, -0.55 * tb)                        # sides splay out to the wall: no razor shadow line beside the slab
    verts += [(Bk[i][0], 0.0, Bk[i][1]) for i in range(n)]
    for i in range(n):
        j = (i + 1) % n
        faces += [(b + n + i, b + n + j, b + j, b + i), (b + 2 * n + i, b + 2 * n + j, b + n + j, b + n + i)]
        fm += [0, 0]
    on = np.zeros(len(verts))
    if spolys:
        on[:len(V)] = np.where(_winding(V, spolys) >= 1, 1.0, np.where(_seg_dist(V, spolys) < 1e-5, 0.3, 0.0))
    o = mesh_object(name, verts, faces)
    smooth(o, 50)
    return o, fm, on


def _lamp(x, zb, B, Lp, tilt, hang, intensity, near, iron, brass, idx):
    """Turned wall rose with two screws, forged gooseneck, lamp-holder collar, spun-brass shade with a rolled
    rim and a bare warm bulb whose tip shows just below the rim. B = bulb position; Lp = where its light is
    emitted from (a little way down the shade axis, so the pool is centred on the plate, not on the wall above)."""
    st, ct = math.sin(math.radians(tilt)), math.cos(math.radians(tilt))

    def along(d):
        return (B[0], B[1] - st * d, B[2] + ct * d)
    hs = 0.0865
    out = [lathe("lamp_rose", [(0.05, 0.0), (0.05, 0.003), (0.047, 0.0055), (0.034, 0.006), (0.03, 0.0095), (0.0165, 0.017),
                               (0.0165, 0.03), (0.0125, 0.034), (0.0, 0.034)], segments=24, loc=(x, 0, zb), rot=(90, 0, 0))]
    for sz in (-1, 1):
        out.append(cylinder("lamp_screw", 0.0052, 0.003, loc=(x + sz * 0.0125, -0.0068, zb + sz * 0.039), rot=(90, 0, 0), verts=8, radius2=0.0036))
    top = along(hang + hs + 0.026)
    pts = bezier_path((x, -0.031, zb), (x, -0.031 - 0.5 * abs(top[1] + 0.031), zb + 0.012), (top[0], top[1] - st * 0.15, top[2] + ct * 0.15), top, n=14)
    arm = sweep("lamp_arm", pts, 0.0078, sides=8, cap=False, smooth_path=False)
    smooth(arm, 60)
    out.append(arm)
    for o in out:
        assign(o, iron)
        paint(o, special=mask_mul(noise_mask(0.05, seed=idx + 3, lo=0.45, hi=0.8), 0.5), wear=0.15)
    shade = lathe("lamp_shade", [(0.0, 0.05), (0.03, 0.046), (0.068, 0.0065), (0.0745, 0.0008), (0.0785, 0.001), (0.0795, 0.0048),
                                 (0.0765, 0.008), (0.062, 0.022), (0.04, 0.046), (0.0225, 0.064), (0.02, 0.082), (0.0, hs)],
                  segments=24, loc=along(hang), rot=(tilt, 0, 0))
    collar = cylinder("lamp_collar", 0.0125, 0.034, loc=along(hang + hs + 0.012), rot=(tilt, 0, 0), verts=10)
    holder = cylinder("lamp_holder", 0.0135, 0.036, loc=along(hang + 0.034), rot=(tilt, 0, 0), verts=10, cap=False)
    for o in (shade, collar, holder):
        assign(o, brass)
    paint(shade, grime=mask_mul(facing((0, 0, 1)), 0.7), special=mask_mul(noise_mask(0.03, seed=idx + 5), 0.2), wear=0.25)
    bulb = sphere("lamp_bulb", 0.0195, loc=B, segments=8, rings=5, scale=(1.0, 1.0, 1.22))
    place(bulb, rot=(tilt, 0, 0))
    assign(bulb, mat("emissive", "#ffd9a0", glow=6))
    light(f"lamp{idx}", Lp, intensity=intensity, range=2.6, flicker=0.05, shadow=True, shadow_near=near)


def _candles(xc, y, z, n, rng):
    """Votive candles in red glass jars standing on a ledge at height z (one may have burnt out).
    Returns the flame positions."""
    glass = mat("glass_lantern", "#e86a3c", glow=0.5, opacity=0.55)
    wax = mat("wax", "#efe2c4")
    flame_m = mat("emissive", "#ffb45e", glow=5)
    xs = [xc] if n == 1 else [xc + (k - (n - 1) / 2) * 0.075 + rng.jitter(0, 0.012) for k in range(n)]
    dead = rng.pick(list(range(n))) if n >= 3 and rng.chance(0.6) else -1
    lit = []
    for k, x in enumerate(xs):
        yk = y + rng.jitter(0, 0.004)
        sc = rng.u(0.85, 1.1)
        jar = lathe("votive", [(0.0, 0.0), (0.019 * sc, 0.0), (0.022 * sc, 0.005), (0.0235 * sc, 0.046 * sc), (0.025 * sc, 0.05 * sc),
                               (0.0225 * sc, 0.05 * sc), (0.021 * sc, 0.007), (0.0, 0.006)], segments=12, loc=(x, yk, z))
        assign(jar, glass)
        hc = rng.u(0.016, 0.034) * sc
        w = cylinder("candle", 0.0165 * sc, hc, loc=(x, yk, z + 0.006 + hc / 2), verts=10)
        assign(w, wax)
        paint(w, grime=facing((0, 0, 1)) if k == dead else 0.0)
        if k == dead:
            continue
        fz = z + 0.006 + hc + 0.011
        fl = sphere("flame", 0.0045, loc=(x, yk, fz), segments=6, rings=4, scale=(1, 1, 2.3))
        assign(fl, flame_m)
        fx(f"flame{k}", "flame", (x, yk, fz), size=0.022)
        lit.append((x, yk, fz))
    return lit


def build(p, rng):
    W, H = p.width, p.height
    hw, hh = W / 2, H / 2
    kind = p.material
    lo = min(W, H)
    arched = p.shape == "arched"
    fw = clamp(0.06 * lo + 0.012, 0.026, 0.075) * p.frame
    rr = clamp(0.03 * lo + 0.006, 0.013, 0.03)
    rc = 0.15 * lo if p.shape == "cut_corner" else 0.0
    rise = min(0.2 * W, 0.28 * H) if arched else 0.0
    backing = p.backing
    m = p.margin
    tb = 0.035 if backing != "none" else 0.0
    mode = p.lettering
    if mode == "auto":
        mode = "raised" if kind == "bronze" else "inlaid" if p.letter_fill == "lead" else "engraved"
    raised = mode == "raised"
    stain = p.stain
    seed = int(p.seed)
    bronze_plate = kind == "bronze"

    # ---- materials
    brass = mat("brass", rough=0.85)
    bronze = mat("bronze_aged", scale=6.0)
    if bronze_plate:
        plate_m = bronze
    elif kind == "stone":
        plate_m = mat("stone_smooth", "#bdb096")
    elif kind == "marble":
        plate_m = mat("stone_smooth", "#ece7dd", rough=0.6)
    else:
        plate_m = mat("slate", "#3e4149", rough=1.25, scale=0.35)
    fill_m = {"gilt": lambda: mat("metal_plain", "#d9ad52", rough=1.45), "lead": lambda: mat("lead"),
              "dark": lambda: mat("stone_smooth", "#2a2521")}[p.letter_fill]()
    top_m = mat("metal_plain", "#c29a56", rough=0.9)       # rubbed-bright faces of the cast letters
    # applied castings on stone: gilt-brass next to gilt letters, otherwise dark bronze with rubbed high points
    emb_m = bronze if bronze_plate or p.letter_fill != "gilt" else brass

    # ---- plate frame: one outline lofted through the moulding profile
    chip = None
    r_chip = rng.fork("chip")
    if not bronze_plate and p.shape != "cut_corner" and r_chip.chance(0.6):
        chip = (r_chip.pick([0, 1]), r_chip.u(0.012, 0.022), r_chip.u(0.01, 0.02))
    outline = _resample(_outline(W, H, p.shape, rc, rise, chip), clamp(0.2 * lo, 0.1, 0.15))
    fs = p.frame
    ring_wear, ring_spec, ring_grime = {}, {}, {}
    if bronze_plate:
        T, fr = 0.008, 0.011 * fs
        prof = [(0, 0), (0, T - 0.002), (0.002, T), (0.15 * fw, T), (0.2 * fw, T + 0.5 * fr), (0.3 * fw, T + 0.9 * fr), (0.42 * fw, T + fr),
                (0.54 * fw, T + 0.88 * fr), (0.62 * fw, T + 0.5 * fr), (0.69 * fw, T + 0.36 * fr), (0.75 * fw, T + 0.5 * fr),
                (0.83 * fw, T + 0.58 * fr), (0.91 * fw, T + 0.42 * fr), (fw, T)]
        ring_wear = {2: 0.45, 5: 0.7, 6: 1.0, 7: 0.7, 11: 0.5}          # outer bead and the small inner bead rubbed bright
        ring_spec = {3: 0.55, 8: 0.45, 9: 1.0, 10: 0.45, 12: 0.5, 13: 0.95}   # verdigris sits in the cove and at the frame root
        ring_grime = {3: 0.6, 8: 0.4, 9: 0.7, 13: 0.7, 0: 0.5, 1: 0.5}
    elif kind == "slate":
        T = 0.018
        prof = [(0, 0), (0, T - 0.005), (0.005, T), (0.52 * fw, T), (0.57 * fw, T - 0.003), (0.66 * fw, T - 0.003), (0.71 * fw, T), (fw, T)]
    else:
        T, fr = (0.026, 0.008 * fs) if kind == "stone" else (0.02, 0.007 * fs)
        prof = [(0, 0), (0, T + fr - 0.007), (0.007, T + fr), (0.34 * fw, T + fr), (0.39 * fw, T + fr - 0.004), (0.47 * fw, T + fr - 0.004),
                (0.52 * fw, T + fr), (0.7 * fw, T + fr), (0.78 * fw, T + fr * 0.8), (0.88 * fw, T + fr * 0.25), (fw, T)]
    plate = _loft("plate", outline, prof, -tb)
    assign(plate, plate_m)
    smooth(plate, 32)
    yf = -(tb + T)                                   # field plane
    n_out = len(outline)

    def ring(d):
        a = np.zeros(len(prof) * n_out)
        for i, v in d.items():
            a[i * n_out:(i + 1) * n_out] = v
        return a
    zrel = np.tile(np.array([q[1] for q in outline]), len(prof))
    lowness = np.clip(0.5 - zrel / H, 0.0, 1.0)      # 1 at the bottom rail, 0 at the top
    if bronze_plate:
        nz_a, nz_b = noise_mask(0.07, seed=seed + 2, lo=0.25, hi=0.75), noise_mask(0.1, seed=seed + 6, lo=0.2, hi=0.8)
        k_sp = (0.3 + 0.7 * lowness ** 1.3) * (0.35 + 0.65 * stain)
        paint(plate, special=lambda co, n: np.clip(ring(ring_spec) * k_sp * (0.45 + 0.55 * nz_a(co, n)), 0, 1) if len(co) == len(k_sp) else 0 * co[:, 0],
              wear=lambda co, n: ring(ring_wear) * (0.55 + 0.45 * nz_b(co, n)) if len(co) == len(k_sp) else 0 * co[:, 0],
              grime=_const(ring(ring_grime)))
    else:
        nz_a = noise_mask(0.09, seed=seed + 2, lo=0.3, hi=0.7)
        paint(plate, grime=_const(lowness ** 3 * 0.55 * stain),
              special=lambda co, n: lowness ** 4 * nz_a(co, n) * 0.6 * stain if len(co) == len(lowness) else 0 * co[:, 0])

    # ---- rosette fixing positions
    e = fw + rr + 0.012 + rc * 0.62
    ex, ez = hw - e, hh - e
    zt = ez
    if arched:
        R = (hw * hw + rise * rise) / (2 * rise)
        zt = (hh - R) + math.sqrt(max(R * R - ex * ex, 0.0)) - e
    fix = [(-ex, -ez), (ex, -ez), (ex, zt), (-ex, zt)]

    # ---- inscription layout (lines, footer, emblem stacked and scaled to fit)
    size = p.text_size
    items = []                                       # [kind, loops / obj, height, width, x-centre, em size]
    for i, s in enumerate([p.line1, p.line2, p.line3, p.line4]):
        s = str(s).strip()
        if not s:
            continue
        lps = _text_loops(s, size, p.font)
        if not lps:
            continue
        allp = np.vstack(lps)
        items.append(["text", lps, size * 1.3, allp[:, 0].max() - allp[:, 0].min(), (allp[:, 0].max() + allp[:, 0].min()) / 2, size])
    emb, aspect = _emblem(p.emblem, rng.fork("emblem"))
    if emb is not None:
        it = ["emblem", emb, p.emblem_size * 1.3 + (0.012 if items else 0.0), p.emblem_size * aspect, 0.0, 0.0]
        if p.emblem_pos == "above":
            items.insert(0, it)
        else:
            items.append(it)
    if str(p.footer).strip():
        lps = _text_loops(str(p.footer).strip(), size * 0.55, p.font, spacing=1.15)
        if lps:
            allp = np.vstack(lps)
            items.append(["text", lps, size * 0.95, allp[:, 0].max() - allp[:, 0].min(), (allp[:, 0].max() + allp[:, 0].min()) / 2, size * 0.55])
    pad = 0.022 + 0.02 * lo
    z_lo, z_hi = -hh + fw + pad, hh - fw - pad - rise * 0.5
    Wt = max(W - 2 * (e + rr + 0.012), 0.1)
    raw, relief = [], []                             # raw: (loops in plate coords, em size)
    if items:
        Ht = sum(it[2] for it in items)
        s = min(1.0, Wt / max(it[3] for it in items), (z_hi - z_lo) / Ht)
        zc = (z_hi + z_lo) / 2 + Ht * s / 2
        for it in items:
            zi = zc - it[2] * s / 2
            zc -= it[2] * s
            if it[0] == "text":
                raw.append(([(lp - np.array([it[4], 0.0])) * s + np.array([0.0, zi]) for lp in it[1]], it[5] * s))
            else:
                place(it[1], loc=(0, yf + 0.0005, zi), scale=p.emblem_size * s)
                relief.append(it[1])
    minw = 0.0026 if raised else 0.003
    lines = []
    for k_tol in (1.0, 1.6, 2.4, 3.4):               # coarsen the outlines until the inscription fits the budget
        lines = [(q, em) for q, em in ((_prep_loops(lps, max(0.00016, 0.0055 * em) * k_tol, minw), em) for lps, em in raw) if q]
        if sum(len(lp) for lps, _ in lines for lp in lps) <= (560 if raised else 760):
            break
    em_main = max([em for _, em in lines], default=size)
    cut_loops = [lp for lps, _ in lines for lp in lps] if not raised else []
    z_text = min([float(lp[:, 1].min()) for lps, _ in lines for lp in lps], default=0.0)

    # ---- field: triangulated inside the frame, with the cut letters as part of the same surface
    inner = np.array(_offset(outline, fw))
    r_f = rng.fork("field")
    g = clamp(0.085 * lo, 0.036, 0.062)
    P = np.array(_grid_pts(-hw, hw, -hh, hh, g, r_f))
    halo = np.array([(fx_ + rad * math.cos(a0 + 2 * math.pi * k / cnt), fz_ + rad * math.sin(a0 + 2 * math.pi * k / cnt))
                     for fx_, fz_ in fix for rad, cnt, a0 in ((rr * 1.22, 10, 0.0), (rr * 2.1, 8, 0.4)) for k in range(cnt)])
    guard = [inner] + cut_loops
    keep = (_winding(P, [inner]) >= 1) & (_seg_dist(P, guard) > 0.36 * g)
    keep_h = (_winding(halo, [inner]) >= 1) & (_seg_dist(halo, guard) > 0.005)
    if cut_loops:
        keep &= _winding(P, cut_loops) < 1
        keep_h &= _winding(halo, cut_loops) < 1
    V, Tr, cen = _cdt(guard, np.vstack([P[keep], halo[keep_h]]))
    ok = _winding(cen, [inner]) >= 1
    Tr, cen = Tr[ok], cen[ok]
    isl = (_winding(cen, cut_loops) >= 1) if cut_loops else np.zeros(len(Tr), dtype=bool)
    vdepth = clamp(0.05 * em_main, 0.0016, 0.004) if mode == "engraved" else 0.0
    field, fm = _field("field", V, Tr, isl, yf, vdepth)
    _mat_slots(field, [plate_m, fill_m] if cut_loops else [plate_m], fm if cut_loops else [0] * len(fm))
    smooth(field, 33)
    if kind == "stone":                              # lay the stone's tooling drag horizontal instead of diagonal
        for o in (field, plate):
            uv_planar(o, u_axis=(0.848, 0.0, -0.53), v_axis=(0.53, 0.0, 0.848))
    fixa = np.array(fix)
    in_lo = float(inner[:, 1].min())

    def d_fix(co):
        return np.min(np.hypot(co[:, 0][:, None] - fixa[:, 0][None], co[:, 2][:, None] - fixa[:, 1][None]), axis=1)

    def d_edge(co):
        return _seg_dist(np.stack([co[:, 0], co[:, 2]], axis=1), [inner])
    if bronze_plate:
        nz_c, nz_d = noise_mask(0.06, seed=seed + 7, lo=0.3, hi=0.7), noise_mask(0.14, seed=seed + 8, lo=0.25, hi=0.75)

        def f_spec(co, n):
            halo_m = np.clip(1.0 - (d_fix(co) - rr) / (1.5 * rr), 0.0, 1.0) ** 1.5 * (0.35 + 0.65 * nz_d(co, n))
            edge = np.clip(1.0 - d_edge(co) / 0.012, 0.0, 1.0) * (0.35 + 0.65 * np.clip(0.5 - co[:, 2] / H, 0, 1))
            low = np.clip(1.0 - (co[:, 2] - in_lo) / (0.11 * H), 0.0, 1.0) ** 1.5
            return np.clip((0.9 * halo_m + 0.8 * edge + 0.8 * low * nz_d(co, n)) * (0.35 + 0.65 * nz_c(co, n)) * (0.3 + 0.7 * stain), 0.0, 1.0)
        paint(field, special=f_spec, grime=lambda co, n: 0.5 + 0.14 * nz_c(co, n))
    else:
        nz_c = noise_mask(0.08, seed=seed + 7, lo=0.3, hi=0.7)

        def f_grime(co, n):                          # damp only along the bottom of the field and in the rosette halos: the text band stays clean
            low = np.clip(1.0 - (co[:, 2] - in_lo) / max(min(0.05, (z_text - in_lo) * 0.6), 0.008), 0.0, 1.0)
            halo_m = np.clip(1.0 - (d_fix(co) - rr) / (1.2 * rr), 0.0, 1.0)
            return np.clip((0.55 * low * (0.5 + 0.5 * nz_c(co, n)) + 0.5 * halo_m) * stain, 0.0, 1.0)
        paint(field, grime=f_grime)

    # ---- cast letters (bronze): same metal as the plate, tops rubbed bright, patina left in the roots
    if raised and lines:
        depth = clamp(0.062 * em_main, 0.0032, 0.0055)
        lett, lfm = _raised("letters", [(lps, min(0.3 * depth, 0.022 * em)) for lps, em in lines], yf, depth, 0.0009)
        _mat_slots(lett, [top_m, plate_m], lfm)
        smooth(lett, 40)
        nz_l = noise_mask(0.035, seed=seed + 3, lo=0.2, hi=0.8)
        ytop = yf - depth
        top = lambda co: co[:, 1] < ytop + 0.0005
        paint(lett, grime=lambda co, n: np.where(top(co), 0.5 * (1.0 - nz_l(co, n)), 1.0),
              special=lambda co, n: np.where(top(co), 0.0, 0.5 * stain * nz_l(co, n)))
    for o in relief:
        assign(o, emb_m)
        mn, mx = world_bbox([o])
        y_hi, dy = mn.y, max(yf - mn.y, 1e-4)
        nz_e = noise_mask(0.03, seed=seed + 9, lo=0.2, hi=0.8)
        paint(o, wear=lambda co, n: np.clip(((yf - co[:, 1]) / dy - 0.5) / 0.4, 0.0, 1.0) * (0.75 + 0.25 * nz_e(co, n)),
              grime=lambda co, n: np.clip(1.0 - (yf - co[:, 1]) / dy / 0.4, 0.0, 1.0),
              special=lambda co, n: np.clip(1.0 - (yf - co[:, 1]) / dy / 0.3, 0.0, 1.0) * 0.5 * stain * nz_e(co, n))

    # ---- four rosette-headed fixing bolts (one sometimes replaced later by a plain dome-headed bolt on a washer)
    ros = lathe("rosette", [(rr, 0.0), (rr, 0.0025), (rr * 0.86, 0.0065), (rr * 0.56, 0.0055), (rr * 0.46, 0.0085), (rr * 0.3, 0.012),
                            (0.0, 0.0135)], segments=12)

    def petal(co):
        r = math.hypot(co.x, co.y)
        if r < rr * 0.5:
            return co
        k = 1.0 + 0.13 * math.cos(6 * math.atan2(co.y, co.x)) * smoothstep(rr * 0.5, rr * 0.8, r)
        return Vector((co.x * k, co.y * k, co.z - 0.0012 * (k - 1.0) / 0.13 * (1 if co.z > 0.001 else 0)))
    transform_verts(ros, petal)
    r_fix = rng.fork("fix")
    odd = r_fix.pick([0, 1, 2, 3]) if r_fix.chance(0.35) else -1
    rosettes = instances_to_mesh("rosettes", ros, [trs((x, yf, z), (90, 0, r_fix.u(0, 60))) for k, (x, z) in enumerate(fix) if k != odd], keep_base=False)
    assign(rosettes, brass)
    paint(rosettes, special=mask_mul(noise_mask(0.02, seed=seed + 4), 0.6 * stain), wear=0.4)
    iron = mat("iron_wrought", scale=2.0)
    if odd >= 0:
        ox, oz = fix[odd]
        washer = lathe("washer", [(rr * 0.78, 0.0), (rr * 0.78, 0.0022), (rr * 0.7, 0.0034), (rr * 0.36, 0.0034)], segments=16, loc=(ox, yf, oz), rot=(90, 0, 0))
        dome = bolt("plain_bolt", rr * 0.42, rr * 0.3, loc=(ox, yf - 0.003, oz), rot=(90, 0, 0), kind="dome")
        for o in (washer, dome):
            assign(o, brass)
            paint(o, special=0.25 * stain, grime=0.3)

    # ---- stone backing slab with the run-off streaks below the plate
    top = hh
    bot = -hh
    if backing != "none":
        stone_m = mat("stone_rough", p.backing_tint)
        base = hex_rgb(p.backing_tint)
        run = hex_rgb("#4fb59c") if bronze_plate else hex_rgb("#3d3a30")
        streak_m = mat("stone_rough", tuple(lerp(base[k], run[k], 0.8 if bronze_plate else 0.6) for k in range(3)))
        bx0, bx1 = -hw - m, hw + m
        bz0, bz1 = -hh - m * 2.5, hh + m
        top, bot = bz1, bz0
        cs = clamp(0.2 * m, 0.008, 0.016)
        hidden = _offset(_outline(W, H, p.shape, rc, rise), 0.03)
        r_s = rng.fork("streaks")
        streaks = []
        if stain > 0.05:
            Lmax = 2.5 * m - cs - 0.02
            xr = hw - rc - 0.012
            xs_s = [-ex, ex, -xr + r_s.u(0, 0.01), xr - r_s.u(0, 0.01)] + [r_s.u(-xr, xr) for _ in range(2 + int(W / 0.5))]
            xs_s = sorted(xs_s)
            xs_s = [x for i, x in enumerate(xs_s) if i == 0 or x - xs_s[i - 1] > 0.028]
            for x in xs_s:
                Ls = Lmax * clamp(stain * r_s.u(0.55, 1.5), 0.25, 1.0)
                streaks.append(_streak_poly(x, -hh + 0.012, Ls + 0.012, r_s.u(0.011, 0.02), r_s))
        slab, sfm, on = _slab("backing", bx0, bx1, bz0, bz1, tb, cs, hidden, streaks, r_s, seed + 40)
        _mat_slots(slab, [stone_m, streak_m], sfm)
        nz_s = noise_mask(0.05, seed=seed + 11, lo=0.25, hi=0.75)

        def s_grime(co, n):                          # dark core on the streaks, a faint damp apron under the plate, dirt on the foot
            x, z = co[:, 0], co[:, 2]
            core = on * 0.85 if len(co) == len(on) else 0.0
            apron = np.clip((hw + 0.02 - np.abs(x)) / 0.04, 0.0, 1.0) * np.clip((-hh + 0.02 - z) / 0.03, 0.0, 1.0) * \
                np.clip(1.0 - (-hh - z) / (2.5 * m), 0.0, 1.0) * nz_s(co * np.array([1.0, 1.0, 0.12]), n) * 0.4
            foot = np.clip(1.0 - (z - bz0) / 0.05, 0.0, 1.0) * 0.3
            return np.clip((core + apron) * stain + foot, 0.0, 1.0)
        paint(slab, grime=s_grime, special=mask_mul(facing((0, 0, 1), 2.0), noise_mask(0.08, seed=seed + 12), 0.15))
        if backing == "tablet":
            dress = mat("stone_smooth", p.backing_tint)
            ov = 0.03
            y1 = -tb
            cor = _prism("cornice", [(0.0, 0.0), (y1 - 0.006, 0.0), (y1 - 0.009, 0.008), (y1 - 0.014, 0.012), (y1 - 0.024, 0.016),
                                     (y1 - 0.034, 0.026), (y1 - 0.042, 0.031), (y1 - 0.042, 0.046), (y1 - 0.036, 0.052), (0.0, 0.064)],
                         bx0 - ov, bx1 + ov)
            place(cor, loc=(0, 0, bz1))
            sill = _prism("sill", [(0.0, 0.0), (0.0, -0.05), (y1 - 0.006, -0.05), (y1 - 0.014, -0.042), (y1 - 0.03, -0.034), (y1 - 0.048, -0.03),
                                   (y1 - 0.052, -0.022), (y1 - 0.052, -0.008), (y1 - 0.046, -0.002), (y1 - 0.004, 0.006), (0.0, 0.006)],
                          bx0 - ov, bx1 + ov)
            place(sill, loc=(0, 0, bz0))
            for o in (cor, sill):
                bevel(o, 0.0025, segments=1)
                assign(o, dress)
                smooth(o, 30)
            paint(cor, special=mask_mul(facing((0, 0, 1), 1.5), noise_mask(0.07, seed=seed + 13, lo=0.3, hi=0.6)), grime=mask_mul(facing((0, 0, -1)), 0.35))
            paint(sill, grime=mask_mul(noise_mask(0.06, seed=seed + 14), 0.7 * stain),
                  special=mask_mul(facing((0, 0, 1), 1.5), noise_mask(0.05, seed=seed + 15, lo=0.3, hi=0.6), stain))
            r_c = rng.fork("corbel")
            for sx in (-1, 1):
                cb = stone_block("corbel", (0.07, tb + 0.03, 0.085), rng=r_c, chamfer=0.012, rough=0.002, detail=0.05)
                transform_verts(cb, lambda co: Vector((co.x, co.y * (1.0 - 0.45 * clamp(0.5 - co.z / 0.085)), co.z)))
                transform_verts(cb, lambda co: Vector((co.x + sx * (hw + m - 0.07), min(co.y - (tb + 0.03) / 2, -0.0004), co.z + bz0 - 0.05 - 0.0425)))
                assign(cb, dress)
                paint(cb, grime=0.4)
            top, bot = bz1 + 0.064, bz0 - 0.135
            socket("sill", "attach", (0, y1 - 0.025, bz0 + 0.006), role="prop_top")

    # ---- gooseneck lamp(s): the bulb hangs in front of the plate's top edge, so the pool is centred on the plate
    if p.lamp == "gooseneck":
        h_l = clamp(0.15 + 0.27 * H, 0.24, 0.42)
        lamp_iron = mat("iron_painted", "#1f2124")
        xs = [0.0] if W < 1.25 else [-W * 0.26, W * 0.26]
        tilt = 34.0
        zb, zl = hh + 0.01, 0.11                     # bulb height; the light sits 11 cm down the shade axis, at the mouth
        d_l = h_l - zl * math.sin(math.radians(tilt))
        # the light must not throw the arm's shadow up the wall: nothing nearer than the wall rose casts from it
        near = math.hypot(yf - d_l, top + 0.12 - zb) + 0.1
        power = p.intensity * (d_l / 0.28) ** 2 * {"bronze": 0.85, "slate": 0.55}.get(kind, 0.7) * (1.0 if len(xs) == 1 else 0.52)
        for i, x in enumerate(xs):
            _lamp(x, top + 0.12, (x, yf - h_l, zb), (x, yf - d_l, zb - zl * math.cos(math.radians(tilt))), tilt, 0.012, power,
                  near, lamp_iron, brass, i)

    # ---- votive candles: on the tablet sill, or on a small forged shelf under the plaque
    if p.lamp == "candles":
        r_cd = rng.fork("candles")
        n = 3 if W >= 0.55 else 1
        if backing == "tablet":
            lit = _candles(r_cd.jitter(0, 0.04), -tb - 0.03, -hh - m * 2.5 + 0.006, n, r_cd)
        else:
            d = tb + 0.105
            sw = 0.12 + 0.075 * (n - 1)
            zs = bot - 0.04
            shelf = [box("shelf", (sw, d, 0.005), loc=(0, -d / 2, zs - 0.0025), bevel=0.001, segments=1),
                     box("shelf_lip", (sw, 0.004, 0.014), loc=(0, -d + 0.002, zs + 0.004), bevel=0.001, segments=1),
                     box("shelf_back", (sw + 0.03, 0.004, 0.05), loc=(0, -0.002, zs - 0.02), bevel=0.001, segments=1)]
            for sx in (-1, 1):
                x = sx * (sw / 2 - 0.012)
                shelf.append(sweep("shelf_stay", bezier_path((x, -0.005, zs - 0.085), (x, -0.012, zs - 0.03), (x, -d * 0.55, zs - 0.035), (x, -d + 0.012, zs - 0.005), n=8),
                                   rounded_rect(0.012, 0.004, 0.001, 1), up=(1, 0, 0), smooth_path=False))
                shelf.append(bolt("shelf_nail", 0.004, 0.003, loc=(sx * (sw / 2 + 0.006), -0.004, zs - 0.02), rot=(90, 0, 0), kind="pyramid"))
            for o in shelf:
                assign(o, iron)
                paint(o, special=mask_mul(noise_mask(0.03, seed=seed + 20), 0.6), grime=0.3)
            lit = _candles(0.0, -d + 0.034, zs, n, r_cd)
        if lit:
            # the group's light sits well clear of the surface so the glow washes the lower lines instead of burning a spot
            h_c = clamp(0.4 * H, 0.18, 0.3)
            cx = sum(q[0] for q in lit) / len(lit)
            fz = sum(q[2] for q in lit) / len(lit)
            light("candles", (cx, yf - h_c, fz + 0.5 * h_c), color="#ffc98a", intensity=p.intensity * 0.085 * (h_c / 0.18) ** 1.5, range=2.8, flicker=0.5,
                  shadow_near=h_c * 0.8)
