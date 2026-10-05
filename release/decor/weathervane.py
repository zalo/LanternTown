"""Roof-top weather vane. A forged, black-painted iron spindle rises out of a beaten lead skirt
(wired bottom edge, welted seam, dressed square over the roof apex) through two spun copper balls
with a soldered equator and a twisted square section to a drilled hub that carries the four
cardinal rods: cut sheet letters brazed to riveted sleeves, bud tips and scrolled stays. On the
spindle top turns the vane: a sleeve with an arrow bar (barbed head, slotted fletching) carrying
a silhouette cut from 3-5 mm hand-beaten copper sheet - rooster, cat, ship, shooting star or an
ornate scrolled arrow - gilded (leaf rubbed through to the copper at edges, rivets and drip
edges) or left as bare weathering copper with verdigris running down from the top edges.
Origin = spindle foot (centre of the lead skirt, z = 0).
"""
from math import atan2, cos, degrees, pi, radians, sin
from tk import *

META = {"name": "Weather vane", "class": "prop_small", "mount": "ground", "focus": 0.74}
PARAMS = {
    "motif": {"default": "rooster", "choices": ["rooster", "arrow", "ship", "star", "cat"],
              "desc": "cut sheet silhouette riding on the arrow"},
    "height": {"default": 1.5, "min": 0.9, "max": 2.4, "desc": "overall height, spindle foot to top of the motif (m)"},
    "gilt": {"default": True, "desc": "gilded sheet work, letters and upper ball (False = bare weathering copper)"},
    "verdigris": {"default": 0.25, "min": 0.0, "max": 1.0, "desc": "amount of green patina on the copper / failed gilding"},
    "wind": {"default": 8.0, "min": -180.0, "max": 180.0, "desc": "heading the vane hunts about (deg about Z, 0 = pointing +X)"},
    "north": {"default": 12.0, "min": -180.0, "max": 180.0, "desc": "bearing of the N arm from +Y (deg): the cross is set to true north, not to the roof"},
    "letters": {"default": "NESW", "desc": "cardinal letters in the order north, east, south, west"},
    "arm_reach": {"default": 0.3, "min": 0.2, "max": 0.42, "desc": "length of a cardinal arm at 1.5 m height (m)"},
    "rust": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "rust on the ironwork"},
}
VARIANTS = {
    "cat_verdigris": {"motif": "cat", "height": 1.2, "gilt": False, "verdigris": 0.9, "wind": -172.0, "north": 16.0, "rust": 0.7, "seed": 4},
    "ship_tall": {"motif": "ship", "height": 2.1, "gilt": True, "verdigris": 0.08, "wind": 2.0, "north": 9.0, "arm_reach": 0.34, "rust": 0.15, "seed": 11},
    "star_copper": {"motif": "star", "height": 1.3, "gilt": False, "verdigris": 0.3, "wind": 176.0, "north": 14.0, "arm_reach": 0.26, "seed": 23},
    "arrow_small": {"motif": "arrow", "height": 1.0, "gilt": True, "verdigris": 0.55, "wind": 12.0, "north": 18.0, "arm_reach": 0.36, "rust": 0.55, "seed": 31},
}

S = "s"   # marks a sharp outline point


# ----------------------------------------------------------------------------- 2D outline helpers
def _outline(pts, s=1.0, ox=0.0, oy=0.0, iters=2, tol=7.0):
    """Corner-cutting smoothing of a closed polygon; points tagged S stay sharp. Nearly collinear
    points are dropped again, so curves are dense only where they bend."""
    P = [((q[0], q[1]), len(q) > 2) for q in pts]
    for _ in range(iters):
        n = len(P)
        Q = []
        for i in range(n):
            c, sh = P[i]
            if sh:
                Q.append((c, True))
                continue
            a = P[i - 1][0]
            b = P[(i + 1) % n][0]
            Q.append(((0.25 * a[0] + 0.75 * c[0], 0.25 * a[1] + 0.75 * c[1]), False))
            Q.append(((0.75 * c[0] + 0.25 * b[0], 0.75 * c[1] + 0.25 * b[1]), False))
        P = Q
    out = []
    n = len(P)
    for i in range(n):
        c, sh = P[i]
        if not sh and out:
            a = out[-1]
            b = P[(i + 1) % n][0]
            v1 = (c[0] - a[0], c[1] - a[1])
            v2 = (b[0] - c[0], b[1] - c[1])
            if abs(degrees(atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1]))) < tol:
                continue
        out.append(c)
    area = sum(out[i - 1][0] * out[i][1] - out[i][0] * out[i - 1][1] for i in range(len(out)))
    if area < 0:
        out.reverse()
    return [(ox + x * s, oy + y * s) for x, y in out]


SHEETS = {}                                        # sheet name -> cut loops (outline + holes) in the XZ plane


def _loop_fields(loops):
    """fn(x, z) -> (distance to the cut edge, metal above the point up to the next edge (0 = none),
    metal below it, inside flag) for points in the plane of a sheet cut along `loops`."""
    import numpy as np
    A = np.array([q for lp in loops for q in lp], dtype=float)
    B = np.array([q for lp in loops for q in (lp[1:] + lp[:1])], dtype=float)
    ax, az, bx, bz = A[None, :, 0], A[None, :, 1], B[None, :, 0], B[None, :, 1]
    dx, dz = bx - ax, bz - az
    l2 = np.maximum(dx * dx + dz * dz, 1e-12)
    ok = np.abs(dx) > 1e-9
    xlo, xhi = np.minimum(ax, bx), np.maximum(ax, bx)

    def fn(x, z, eps=0.0013):
        x, z = np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(z, dtype=float).reshape(-1, 1)
        t = np.clip(((x - ax) * dx + (z - az) * dz) / l2, 0.0, 1.0)
        d_edge = np.sqrt((x - ax - t * dx) ** 2 + (z - az - t * dz) ** 2).min(axis=1)
        zc = az + (x - ax) / np.where(ok, dx, 1.0) * dz
        hit = ok & (x >= xlo) & (x < xhi)
        up, dn = hit & (zc > z + eps), hit & (zc < z - eps)
        odd_up, odd_dn = up.sum(1) % 2 == 1, dn.sum(1) % 2 == 1
        d_top = np.where(odd_up, np.where(up, zc - z, 1e6).min(axis=1), 0.0)
        d_bot = np.where(odd_dn, np.where(dn, z - zc, 1e6).min(axis=1), 0.0)
        return d_edge, d_top, d_bot, odd_up
    return fn


def _sheet(name, outline, thick, holes=(), k=1.0, dens=0.015, beat=0.0018):
    """Sheet cut-out standing in the XZ plane, centred on y = 0, with real thickness. Unlike a plain
    extrusion the two faces carry a scatter of interior vertices (constrained Delaunay), so the
    weathering masks have something to live on, and the sheet is planished: both faces move together
    and it stops reflecting like a mirror-flat plate."""
    import bmesh
    import numpy as np
    from mathutils import Vector, noise
    from mathutils.geometry import delaunay_2d_cdt
    loops = [list(outline)] + [list(h) for h in holes]
    fld = _loop_fields(loops)
    pts, edges = [], []
    for li, lp in enumerate(loops):
        b0 = len(pts)
        pts += [Vector((x, z)) for x, z in lp]
        ccw = sum(lp[i - 1][0] * lp[i][1] - lp[i][0] * lp[i - 1][1] for i in range(len(lp))) > 0
        for i in range(len(lp)):                    # metal on the left of every edge
            e = (b0 + i, b0 + (i + 1) % len(lp))
            edges.append(e if ccw == (li == 0) else e[::-1])
    nb = len(pts)
    d = dens * k
    P = np.array(outline, dtype=float)
    x0, z0, x1, z1 = P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
    # interior vertices stand in columns (close in x, sparse in z) so that rain streaks, which vary
    # quickly across the sheet and slowly down it, survive the per-vertex masks
    gx, gz = [], []
    j = 0
    xx = x0 + d * 0.45
    while xx < x1:
        zz = z0 + d * (0.35 + 1.1 * ((j * 0.618) % 1.0))
        while zz < z1:
            gx.append(xx)
            gz.append(zz)
            zz += d * 1.5
        xx += d * 0.62
        j += 1
    if gx:
        de, _, _, ins = fld(gx, gz, eps=0.0)
        pts += [Vector((gx[i], gz[i])) for i in range(len(gx)) if ins[i] and de[i] > 0.42 * d]
    vs, _, fs, _, _, _ = delaunay_2d_cdt(pts, edges, [], 0, 1e-7)
    if len(vs) != len(pts):
        raise RuntimeError(f"sheet {name}: outline self-intersects")
    cen = np.array([[sum(vs[i].x for i in f) / len(f), sum(vs[i].y for i in f) / len(f)] for f in fs])
    keep = fld(cen[:, 0], cen[:, 1], eps=0.0)[3]
    f = 1.0 / (0.062 * k)
    ph = (sum(ord(c) for c in name) % 17) * 1.7
    bm = bmesh.new()
    fr, bk = [], []
    for q in vs:
        dy = beat * k * noise.noise(Vector((q.x * f + ph, 1.7, q.y * f)))
        fr.append(bm.verts.new((q.x, dy - thick / 2, q.y)))
        bk.append(bm.verts.new((q.x, dy + thick / 2, q.y)))
    for fi, face in enumerate(fs):
        if keep[fi]:
            ar = sum(vs[face[i - 1]].x * vs[face[i]].y - vs[face[i]].x * vs[face[i - 1]].y for i in range(len(face)))
            face = list(face) if ar > 0 else list(face)[::-1]
            bm.faces.new([fr[i] for i in face])
            bm.faces.new([bk[i] for i in reversed(face)])
    # the cut edge gets its own vertices: it weathers differently from the faces (leaf gone, metal burnished)
    ef = [bm.verts.new(fr[i].co) for i in range(nb)]
    eb = [bm.verts.new(bk[i].co) for i in range(nb)]
    for i, j2 in edges:
        bm.faces.new([ef[j2], ef[i], eb[i], eb[j2]])
    o = from_bmesh(name, bm)
    smooth(o, 38)
    SHEETS[name] = loops
    return o


def _sheet_masks(loops, rivets, k, v, gilt, rng, seed, tone=1.0):
    """Weathering of a cut sheet from its own outline: (grime, wear, special) mask callables.
    Rain runs down the faces from every top edge, slot and rivet (streaks), hangs along the bottom
    edges (drip line) and creeps under rivet heads; the leaf rubs through at the cut edges."""
    import numpy as np
    fld = _loop_fields(loops)
    A = np.array([q for lp in loops for q in lp], dtype=float)
    x0, x1 = float(A[:, 0].min()) - 0.01, float(A[:, 0].max()) + 0.01
    n = max(4, int((x1 - x0) / (0.017 * k)) + 2)
    grid = np.linspace(x0, x1, n)
    vals = np.array([rng.u() ** 2.0 for _ in range(n)])                 # how wet each run is
    runs = np.array([(0.05 + 0.25 * rng.u() ** 1.5) * k for _ in range(n)])  # how far down it reaches
    R = np.array(rivets, dtype=float).reshape(-1, 2) if len(rivets) else None
    n1 = noise_mask(0.075 * k, seed=seed + 2, lo=0.25, hi=0.8)
    n2 = noise_mask(0.022 * k, seed=seed + 3, lo=0.2, hi=0.9)
    cache = {}

    def fields(co, nrm):
        key = (len(co), float(co[0, 0]), float(co[-1, 2]))
        if key in cache:
            return cache[key]
        d_edge, d_top, d_bot, _ = fld(co[:, 0], co[:, 2])
        edge = np.exp(-d_edge / (0.004 * k))
        drip = np.exp(-d_bot / (0.02 * k))
        streak = np.interp(co[:, 0], grid, vals) * np.exp(-d_top / np.interp(co[:, 0], grid, runs))
        if tone < 1.0:
            streak = streak * 0.3                                    # overlays sit in the lee of the sheet behind
        if R is not None:
            dr = np.sqrt((co[:, 0:1] - R[None, :, 0]) ** 2 + (co[:, 2:3] - R[None, :, 1]) ** 2).min(axis=1)
            riv = np.exp(-dr / (0.012 * k))
        else:
            riv = np.zeros(len(co))
        a, b = n1(co, nrm), n2(co, nrm)
        side = np.abs(nrm[:, 1]) < 0.6                               # vertices of the cut edge itself
        rub = np.clip((b - 0.45) / 0.3, 0.0, 1.0)
        if gilt:
            patch = np.clip((a - (1.0 - 0.7 * v)) / 0.25, 0.0, 1.0)
            lost = np.clip(0.85 * edge * rub + 0.7 * riv + 0.75 * drip * b + 0.9 * patch + 0.8 * streak * (0.25 + v), 0.0, 1.0)
            sp = np.clip((0.3 + 1.0 * v) * (0.7 * drip * (0.35 + 0.65 * b) + 2.0 * streak + 0.45 * riv) * (0.45 + 0.55 * lost), 0.0, 0.46 + 0.4 * v)
            g = np.where(side, 0.2, np.clip(tone * (1.0 - lost) * (0.76 + 0.24 * b) * (1.0 - 0.9 * np.clip((sp - 0.42) / 0.2, 0.0, 1.0)), 0.0, 1.0))
            w = np.where(side, 0.6, 0.15 * edge * rub + 0.1 * lost * b)
        else:
            sp = np.clip((0.2 + 0.8 * v) * (0.7 * drip * (0.35 + 0.65 * b) + 2.2 * streak * (0.6 + 0.4 * a) + 0.45 * riv), 0.0, 0.42 + 0.4 * v)
            g = np.where(side, 0.3, np.clip(tone * (0.64 + 0.32 * b) * (1.0 - 0.9 * np.clip((sp - 0.42) / 0.2, 0.0, 1.0)), 0.0, 1.0))   # dirt would blacken the patina
            w = np.where(side, 0.5, 0.1 * edge * rub)
        cache[key] = (g, w, sp)
        return cache[key]

    return (lambda co, nrm: fields(co, nrm)[0]), (lambda co, nrm: fields(co, nrm)[1]), (lambda co, nrm: fields(co, nrm)[2])


def _lean(o, angle=11.0):
    """Font outlines come far richer than a 9 cm sawn letter needs: weld, dissolve the near-straight
    runs of the outline and re-triangulate."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
    bmesh.ops.dissolve_limit(bm, angle_limit=radians(angle), verts=bm.verts[:], edges=bm.edges[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(o.data)
    bm.free()
    return o


def _rivet(r, h):
    """Cheap snap-head rivet (the toolkit dome bolt is 84 tris - too rich for dozens of these)."""
    return lathe("rivet", [(r, 0.0), (r * 0.8, h * 0.62), (0.0, h)], segments=6)


def _circle(cx, cy, r, n=8):
    return [(cx + r * cos(2 * pi * i / n), cy + r * sin(2 * pi * i / n)) for i in range(n)]


ROOSTER = [
    (0.40, 0.0, S), (0.495, 0.0, S), (0.495, 0.02, S), (0.455, 0.035), (0.468, 0.20), (0.495, 0.235, S),
    (0.535, 0.20), (0.535, 0.035), (0.52, 0.02, S), (0.52, 0.0, S), (0.635, 0.0, S), (0.635, 0.02, S), (0.578, 0.04),
    (0.578, 0.21), (0.62, 0.27), (0.72, 0.325), (0.80, 0.43), (0.835, 0.54), (0.82, 0.64), (0.80, 0.71), (0.805, 0.765),
    (0.835, 0.74), (0.862, 0.695), (0.882, 0.74), (0.868, 0.80), (0.89, 0.818), (0.975, 0.83, S), (0.89, 0.868),
    (0.872, 0.89), (0.878, 0.93), (0.862, 0.975, S), (0.835, 0.925, S), (0.812, 0.995, S), (0.79, 0.93, S),
    (0.755, 0.985, S), (0.745, 0.92, S), (0.70, 0.95, S), (0.712, 0.885), (0.70, 0.83), (0.675, 0.74), (0.635, 0.66),
    (0.585, 0.615, S), (0.52, 0.585), (0.44, 0.585), (0.37, 0.62), (0.31, 0.70), (0.265, 0.80), (0.215, 0.89),
    (0.15, 0.95), (0.07, 0.945), (0.015, 0.875), (0.0, 0.76, S), (0.045, 0.835), (0.09, 0.868), (0.14, 0.86, S),
    (0.075, 0.80), (0.033, 0.72), (0.028, 0.615, S), (0.075, 0.70), (0.12, 0.742), (0.175, 0.75, S),
    (0.11, 0.67), (0.083, 0.58), (0.093, 0.485, S), (0.13, 0.57), (0.17, 0.612), (0.222, 0.63, S),
    (0.172, 0.54), (0.165, 0.46), (0.185, 0.395, S), (0.212, 0.47), (0.25, 0.52), (0.30, 0.50), (0.335, 0.42),
    (0.37, 0.33), (0.415, 0.275), (0.432, 0.20), (0.42, 0.035), (0.40, 0.02, S),
]
ROOSTER_WING = [
    (0.68, 0.49), (0.655, 0.555), (0.58, 0.572), (0.48, 0.55), (0.385, 0.505), (0.30, 0.445, S), (0.385, 0.435),
    (0.365, 0.385, S), (0.445, 0.39), (0.435, 0.335, S), (0.52, 0.355), (0.60, 0.375), (0.66, 0.425),
]
CAT = [
    (0.22, 0.0, S), (0.335, 0.0, S), (0.34, 0.028), (0.30, 0.05), (0.305, 0.20), (0.34, 0.30), (0.42, 0.335),
    (0.52, 0.33), (0.60, 0.30), (0.625, 0.20), (0.63, 0.04), (0.625, 0.0, S), (0.745, 0.0, S), (0.75, 0.028),
    (0.705, 0.05), (0.70, 0.22), (0.715, 0.33), (0.74, 0.40), (0.778, 0.435), (0.84, 0.432), (0.89, 0.468, S),
    (0.868, 0.535), (0.862, 0.56), (0.868, 0.648, S), (0.815, 0.59, S), (0.788, 0.66, S), (0.758, 0.575),
    (0.72, 0.555), (0.62, 0.60), (0.50, 0.655), (0.38, 0.64), (0.295, 0.585), (0.262, 0.57, S), (0.228, 0.68),
    (0.208, 0.80), (0.222, 0.895), (0.262, 0.945), (0.31, 0.935), (0.335, 0.965), (0.275, 1.0), (0.20, 0.985),
    (0.155, 0.92), (0.145, 0.80), (0.162, 0.68), (0.185, 0.58), (0.185, 0.50), (0.172, 0.40), (0.185, 0.28),
    (0.215, 0.18), (0.225, 0.05), (0.21, 0.028),
]
HULL = [
    (0.20, 0.0, S), (0.72, 0.0), (0.83, 0.055), (0.90, 0.15), (0.935, 0.225, S), (1.07, 0.295, S), (1.065, 0.315, S),
    (0.91, 0.255, S), (0.905, 0.28, S), (0.74, 0.26, S), (0.735, 0.205, S), (0.55, 0.188), (0.385, 0.20, S),
    (0.38, 0.268, S), (0.20, 0.288, S), (0.195, 0.335, S), (0.085, 0.352, S), (0.10, 0.22), (0.135, 0.10),
]
ARROW_HEAD = [
    (1.0, 0.0, S), (0.55, 0.17), (0.2, 0.33), (-0.02, 0.43, S), (0.14, 0.2), (0.2, 0.078, S), (-0.22, 0.06, S),
    (-0.22, -0.06, S), (0.2, -0.078, S), (0.14, -0.2), (-0.02, -0.43, S), (0.2, -0.33), (0.55, -0.17),
]
ARROW_TAIL = [
    (0.0, 0.07, S), (-0.10, 0.20), (-0.22, 0.45, S), (-1.0, 0.45, S), (-0.92, 0.22), (-0.80, 0.0, S), (-0.92, -0.22),
    (-1.0, -0.45, S), (-0.22, -0.45, S), (-0.10, -0.20), (0.0, -0.07, S),
]


def _slit(pts, w, s=1.0, ox=0.0, oy=0.0, n=3):
    """Pierced slot along a short polyline: a lens-shaped hole, widest in the middle."""
    path = resample_path([(x, y, 0.0) for x, y in pts], n=n * 2 + 1)
    up, lo = [], []
    m = len(path)
    for i, q in enumerate(path):
        a = path[max(i - 1, 0)]
        b = path[min(i + 1, m - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        ln = max((tx * tx + ty * ty) ** 0.5, 1e-9)
        hw = w * 0.5 * sin(pi * i / (m - 1)) ** 0.6
        if 0 < i < m - 1:
            up.append((q[0] - ty / ln * hw, q[1] + tx / ln * hw))
            lo.append((q[0] + ty / ln * hw, q[1] - tx / ln * hw))
        else:
            up.append((q[0], q[1]))
    return [(ox + x * s, oy + y * s) for x, y in up + lo[::-1]]


ROOSTER_SLITS = [
    ((0.165, 0.835), (0.215, 0.775), (0.255, 0.70)), ((0.20, 0.73), (0.245, 0.675), (0.28, 0.61)),
    ((0.245, 0.615), (0.28, 0.575), (0.31, 0.535)), ((0.725, 0.77), (0.71, 0.70), (0.685, 0.635)),
    ((0.765, 0.755), (0.757, 0.69), (0.737, 0.63)),
]


def _bracket_scroll(name, u0, z0, u1, z1, curl, w, t, n_bez=8, n_curl=11):
    """Forged stay in the XZ plane: leaves a vertical member at (u0, z0), swings out to touch a
    horizontal member at (u1, z1) and finishes in a volute curling back from it.
    z1 > z0 -> stay under the horizontal, z1 < z0 -> stay standing on it."""
    sg = 1.0 if z1 > z0 else -1.0
    du = 1.0 if u1 > u0 else -1.0
    pts = bezier_path((u0, 0, z0), (u0, 0, z0 + (z1 - z0) * 0.7), (u0 + (u1 - u0) * 0.45, 0, z1), (u1, 0, z1), n=n_bez)
    cz = z1 - sg * curl
    for i in range(1, n_curl + 1):
        f = i / n_curl
        a = radians(90.0 - 430.0 * f)
        r = curl * (1.0 - 0.72 * f)
        pts.append((u1 + du * r * cos(a), 0, cz + sg * r * sin(a)))
    prof = [(-t / 2, -w / 2), (t / 2, -w / 2), (t / 2, w / 2), (-t / 2, w / 2)]
    return sweep(name, pts, prof, up=(0, 1, 0), smooth_path=False, scale_fn=lambda q: 1.0 - 0.45 * smoothstep(0.55, 1.0, q))


# ----------------------------------------------------------------------------- motifs (XZ plane)
def _motif(p, rng, k, zb, rb, sheet_m, iron_m):
    """Return (objects, height above the bar axis, rivet positions). Motif faces +X, the
    pivot is x = 0, sheet feet are let 2 mm into the bar top."""
    zf = zb + rb - 0.002 * k
    t3, t5 = max(0.0028, 0.003 * k), max(0.0045, 0.005 * k)
    objs = []
    rv = []                                        # rivet matrices
    rv2 = []                                       # rivet positions in the sheet plane (for the weathering)
    m = p.motif

    def rivet(x, z, th):
        rv2.append((x, z))
        rv.append(trs((x, -th / 2, z), (90, 0, 0)))
        rv.append(trs((x, th / 2, z), (-90, 0, 0)))

    if m == "rooster":
        sc = 0.40 * k
        ox = -0.517 * sc
        holes = [_circle(ox + 0.826 * sc, zf + 0.868 * sc, 0.012 * sc, 6)] + [_slit(q, 0.016, sc, ox, zf) for q in ROOSTER_SLITS]
        body = _sheet("rooster", _outline(ROOSTER, sc, ox, zf, tol=9.0), t3, holes, k=k)
        wing = _sheet("rooster_wing", _outline(ROOSTER_WING, sc, ox, zf, tol=9.0), t3 + 0.005 * k, k=k, dens=0.024)
        objs += [body, wing]
        for (x, y) in ((0.635, 0.50), (0.57, 0.525), (0.50, 0.505)):
            rivet(ox + x * sc, zf + y * sc, t3 + 0.005 * k)
        top = 1.0 * sc
    elif m == "cat":
        sc = 0.40 * k
        ox = -0.485 * sc
        eye = [(ox + (0.835 + dx) * sc, zf + (0.512 + dy) * sc) for dx, dy in ((-0.016, 0.0), (0.0, 0.009), (0.016, 0.004), (0.0, -0.008))]
        objs.append(_sheet("cat", _outline(CAT, sc, ox, zf), t3, [eye], k=k))
        # whiskers: three fine wires brazed to the muzzle each side of the sheet
        for j, a in enumerate((-18, 4, 24)):
            x0, z0 = ox + 0.862 * sc, zf + 0.468 * sc
            objs.append(sweep(f"whisker{j}", [(x0 - 0.02 * sc, 0, z0), (x0 + 0.09 * sc * cos(radians(a)), 0, z0 + 0.09 * sc * sin(radians(a)))],
                              0.0012 * k, sides=4, smooth_path=False))
        top = 1.0 * sc
    elif m == "ship":
        sc = 0.46 * k
        ox = -0.47 * sc
        ports = [[(ox + (x + dx) * sc, zf + (0.10 + dy) * sc) for dx, dy in ((0, 0), (0.036, 0), (0.036, 0.036), (0, 0.036))]
                 for x in (0.29, 0.41, 0.53, 0.65)]
        ports.append(_circle(ox + 0.135 * sc, zf + 0.285 * sc, 0.014 * sc))
        objs.append(_sheet("hull", _outline(HULL, sc, ox, zf), t5, ports, k=k, dens=0.024))
        masts = ((0.52, 0.19, 0.97), (0.765, 0.26, 0.78), (0.26, 0.28, 0.68))
        for j, (x, y0, y1) in enumerate(masts):
            mo = cylinder(f"mast{j}", 0.0042 * k, (y1 - y0) * sc, loc=(ox + x * sc, 0, zf + (y0 + y1) / 2 * sc), verts=6, radius2=0.003 * k)
            assign(mo, iron_m)
            objs.append(mo)
        sails = ((0.40, 0.62, 0.275, 0.50, 0.05), (0.42, 0.60, 0.54, 0.74, 0.045), (0.45, 0.58, 0.775, 0.895, 0.035),
                 (0.67, 0.83, 0.33, 0.50, 0.045), (0.69, 0.82, 0.535, 0.685, 0.04), (0.17, 0.32, 0.37, 0.57, 0.035))
        for j, (xl, xr, yb, yt, bu) in enumerate(sails):
            sp = [(xl, yb, S), ((xl + xr) / 2 + bu * 0.5, yb + 0.018), (xr + bu * 0.45, yb, S), (xr + bu, (yb + yt) / 2),
                  (xr + bu * 0.25, yt, S), (xl + 0.012, yt, S), (xl + bu * 0.7, (yb + yt) / 2)]
            objs.append(_sheet(f"sail{j}", _outline(sp, sc, ox, zf), t3, k=k, dens=0.028))
            rivet(ox + masts[0 if j < 3 else (1 if j < 5 else 2)][0] * sc, zf + (yb + yt) / 2 * sc, 0.0084 * k)
        for j, (x, y, ln) in enumerate(((0.52, 0.95, 0.15), (0.765, 0.765, 0.10), (0.26, 0.665, 0.09))):
            fp = [(x, y + 0.02, S), (x + ln * 0.4, y + 0.026), (x + ln * 0.75, y + 0.006), (x + ln, y + 0.012, S),
                  (x + ln * 0.75, y - 0.012), (x + ln * 0.4, y + 0.004), (x, y - 0.022, S)]
            objs.append(_sheet(f"pennant{j}", _outline(fp, sc, ox, zf), t3, k=k, dens=0.03))
        for j, (a, b) in enumerate((((1.06, 0.305), (0.52, 0.93)), ((1.0, 0.285), (0.765, 0.75)), ((0.10, 0.345), (0.52, 0.93)),
                                    ((0.26, 0.66), (0.52, 0.80)))):
            st = sweep(f"stay{j}", [(ox + a[0] * sc, 0.003 * k, zf + a[1] * sc), (ox + b[0] * sc, 0.003 * k, zf + b[1] * sc)],
                       0.0014 * k, sides=4, smooth_path=False)
            assign(st, iron_m)
            objs.append(st)
        top = 0.98 * sc
    elif m == "star":
        sc = 0.50 * k
        cx, cy = 0.11 * k, 0.243 * sc
        star, inner = [], []
        for i in range(10):
            a = radians(90 + i * 36)
            r = 0.30 if i % 2 == 0 else 0.125
            star.append((cx + r * sc * cos(a), zf + cy + r * sc * sin(a)))
            ri = 0.105 if i % 2 == 0 else 0.046
            inner.append((cx + ri * sc * cos(a), zf + cy + ri * sc * sin(a)))
        objs.append(_sheet("star", star, t5, [inner], k=k))
        for j, (ye, amp, ln, w0) in enumerate(((0.47, 0.035, 0.74, 0.085), (0.275, 0.028, 0.86, 0.075), (0.085, 0.03, 0.72, 0.07))):
            up, lo = [], []
            n = 9
            for i in range(n + 1):
                f = i / n
                x = cx - (0.03 + ln * f) * sc
                y = cy + (ye - 0.243) * sc * f + amp * sc * sin(f * 2.3 * pi + j * 1.3) * f
                w = w0 * sc * (1.0 - f) ** 0.75 * 0.5
                up.append((x, zf + y + w))
                lo.append((x, zf + y - w))
            ol = up + lo[::-1][1:]
            ar = sum(ol[i - 1][0] * ol[i][1] - ol[i][0] * ol[i - 1][1] for i in range(len(ol)))
            objs.append(_sheet(f"comet_tail{j}", ol if ar > 0 else ol[::-1], t3, k=k, dens=0.022))
            rivet(cx - 0.075 * sc, zf + cy + (ye - 0.243) * sc * 0.06, t5)
        rivet(cx, zf + cy + 0.2 * sc, t5)
        top = cy + 0.30 * sc
    else:                                           # ornate arrow: scrollwork about the sleeve and a spear finial
        w, t = 0.013 * k, 0.0045 * k
        r_sl = 0.0145 * k
        for j, (sx, sz, reach, rise) in enumerate(((1, 1, 0.17, 0.17), (-1, 1, 0.21, 0.19), (1, -1, 0.12, 0.085), (-1, -1, 0.15, 0.1))):
            s = _bracket_scroll(f"vane_scroll{j}", sx * (r_sl + t * 0.4), zb + sz * rise * k, sx * reach * k, zb + sz * (rb + t * 0.5),
                                0.034 * k * (reach / 0.17), w, t)
            assign(s, iron_m)
            objs.append(s)
        top = 0.30 * k
    for o in objs:
        if not o.data.materials:
            assign(o, sheet_m)
    if rv:
        b = _rivet(0.0038 * k, 0.0022 * k)
        r = instances_to_mesh("motif_rivets", b, rv, keep_base=False)
        assign(r, sheet_m)
        objs.append(r)
    return objs, top, rv2


# ----------------------------------------------------------------------------- build
def build(p, rng):
    import numpy as np
    SHEETS.clear()
    H = p.height
    k = (H / 1.5) ** 0.8
    v = p.verdigris
    sd = p.seed
    rust = p.rust
    # Finishes. The night square lights metals almost only by the lantern key, and a mirror-flat fully
    # metallic sheet then shows nothing but black sky - so:
    #  * ironwork is weathered black paint gone graphite grey (dielectric: the shaded side keeps its form),
    #    chipped at the arrises and bleeding rust at joints, tips and scroll ends;
    #  * gilding / bare copper are the library brass (tinted gold / copper). Its `grime` layer is the dull,
    #    matt bloom of weathered leaf / oxide (lighter and partly non-metallic), bare of it the brown
    #    tarnished metal shows with bright streaks, `wear` is burnished metal (hot glints on the arrises)
    #    and `special` verdigris. The masks are painted from each sheet's own outline (_sheet_masks).
    iron = mat("iron_painted", "#7a8083")
    copper_ball = mat("copper_patina")
    sheet = mat("brass", "#ffe090", rough=1.35) if p.gilt else mat("brass", "#ffc09a", rough=1.6)
    lead = mat("lead", grime=1.7)
    ornate = p.motif == "arrow"

    rb = 0.0065 * k                                  # arrow bar radius
    tops = {"rooster": 0.40, "cat": 0.40, "ship": 0.451, "star": 0.2715, "arrow": 0.30}
    zb = H - tops[p.motif] * k - rb                  # height of the arrow bar axis
    zs0 = zb - (0.085 if not ornate else 0.125) * k  # underside of the turning sleeve
    z_sock = 0.23 * k
    span = zs0 - z_sock
    R1, R2 = 0.058 * k, 0.037 * k
    zb1 = z_sock + span * 0.2
    zb2 = z_sock + span * 0.45
    zc = z_sock + span * 0.72

    def rs(z):                                       # spindle radius: forged taper
        return lerp(0.0125 * k, 0.0085 * k, clamp((z - z_sock) / max(span, 1e-3)))

    body = []

    # ---- lead skirt: sheet lead beaten down over the roof apex (square at the foot where it laps the
    # hips, round under the storm collar), bottom edge turned round a wire, welted and soldered seam
    r0 = 0.122 * k
    sk = [(r0 - 0.005 * k, 0.0), (r0 + 0.003 * k, 0.0012 * k), (r0 + 0.0065 * k, 0.006 * k), (r0 + 0.004 * k, 0.0115 * k), (r0 - 0.003 * k, 0.0135 * k),
          (r0 - 0.011 * k, 0.019 * k), (0.092 * k, 0.05 * k), (0.063 * k, 0.096 * k), (0.041 * k, 0.14 * k), (0.031 * k, 0.163 * k), (0.0255 * k, 0.178 * k)]
    skirt = lathe("lead_skirt", sk, segments=16, smooth_angle=24)

    def squared(x, y, z):                            # plan goes from rounded square (foot) to round (top)
        a = atan2(y, x) - radians(45.0)
        w = clamp(1.0 - z / (0.15 * k)) ** 1.3
        return 1.0 + 0.9 * w * ((abs(cos(a)) ** 4 + abs(sin(a)) ** 4) ** -0.25 - 1.0)

    def sq_fn(c):
        f = squared(c.x, c.y, c.z)
        c.x *= f
        c.y *= f
        return c
    transform_verts(skirt, sq_fn)
    displace_noise(skirt, strength=0.0042 * k, scale=0.045 * k, seed=sd, axis_mask=(1, 1, 0.25))
    transform_verts(skirt, lambda c: c if c.z > 0.0005 else type(c)((c.x, c.y, 0.0)))
    sa = radians(rng.fork("skirt").jitter(-88.0, 8.0))          # the welt runs down a flat side
    wp = []
    for (r, z) in sk[4:]:
        f = squared(r * cos(sa), r * sin(sa), z)
        wp.append(((r * f + 0.0012 * k) * cos(sa), (r * f + 0.0012 * k) * sin(sa), z))
    welt = sweep("lead_welt", wp, [(-0.0075 * k, -0.003 * k), (0.0075 * k, -0.003 * k), (0.005 * k, 0.003 * k), (-0.005 * k, 0.003 * k)],
                 up=(cos(sa), sin(sa), 0.3), smooth_path=False)
    dots = []
    dot = _rivet(0.0058 * k, 0.003 * k)
    for j in range(4):                               # lead-capped fixings through the lap at each hip
        a = radians(45.0 + 90.0 * j + rng.fork("dots").jitter(0.0, 6.0))
        r = 0.106 * k
        f = squared(r * cos(a), r * sin(a), 0.035 * k)
        dots.append(trs((r * f * cos(a), r * f * sin(a), 0.035 * k), (0, 62, degrees(a))))
    leadwork = [skirt, welt, instances_to_mesh("lead_dots", dot, dots, keep_base=False)]
    lr = rng.fork("lead")
    lg = np.array([lr.u() ** 1.6 for _ in range(34)] * 2)

    def lead_grime(co, nrm):                         # sooty run-off streaks below the collar, dirt in the lap
        a = (np.arctan2(co[:, 1], co[:, 0]) + np.pi) / (2 * np.pi) * 34.0
        st = np.interp(a, np.arange(68.0), lg)
        hz = np.clip(co[:, 2] / (0.18 * k), 0.0, 1.0)
        return np.clip(0.3 + 0.7 * st * (0.4 + 0.6 * hz) + 0.35 * np.exp(-co[:, 2] / (0.02 * k)), 0.0, 1.0)
    for o in leadwork:
        assign(o, lead)
        paint(o, grime=lead_grime, mode="set")

    # ---- spindle: one turned / forged profile with collars, hub and bearing shoulder
    prof = [(0.02 * k, 0.158 * k), (0.035 * k, 0.163 * k), (0.038 * k, 0.176 * k), (0.03 * k, 0.186 * k), (0.022 * k, 0.21 * k),
            (0.026 * k, 0.218 * k), (0.021 * k, 0.228 * k), (rs(z_sock), z_sock + 0.012 * k)]

    def bead(z, ra, h):
        r = rs(z)
        prof.extend([(r, z - h / 2), (r + ra * 0.8, z - h * 0.25), (r + ra, z), (r + ra * 0.8, z + h * 0.25), (r, z + h / 2)])

    def cup(z, ra, h, up):                           # flared seat that receives a ball
        r = rs(z)
        if up:
            prof.extend([(r, z - h), (r + ra * 0.5, z - h * 0.6), (r + ra, z)])
        else:
            prof.extend([(r + ra, z), (r + ra * 0.5, z + h * 0.6), (r, z + h)])

    cup(zb1 - R1 * 0.95, 0.012 * k, 0.03 * k, True)
    cup(zb1 + R1 * 0.95, 0.012 * k, 0.03 * k, False)
    cup(zb2 - R2 * 0.95, 0.009 * k, 0.022 * k, True)
    cup(zb2 + R2 * 0.95, 0.009 * k, 0.022 * k, False)
    bead(zc - 0.052 * k, 0.005 * k, 0.014 * k)
    r = rs(zc)
    prof.extend([(r + 0.004 * k, zc - 0.036 * k), (r + 0.0135 * k, zc - 0.027 * k), (r + 0.0135 * k, zc + 0.027 * k), (r + 0.004 * k, zc + 0.036 * k)])
    bead(zc + 0.052 * k, 0.005 * k, 0.014 * k)
    bead(zs0 - 0.012 * k, 0.0085 * k, 0.02 * k)
    prof.append((rs(zs0), zs0 + 0.03 * k))
    spindle = lathe("spindle", prof, segments=8, smooth_angle=50)
    body.append(spindle)
    joints = [0.17 * k, 0.215 * k, zb1 - R1, zb1 + R1, zb2 - R2, zb2 + R2, zc - 0.052 * k, zc, zc + 0.052 * k, zs0]

    # twisted square section between the balls (forged, a little uneven)
    ta, tb = zb1 + R1 + 0.035 * k, zb2 - R2 - 0.027 * k
    if tb - ta > 0.05 * k:
        a = 0.0128 * k
        turns = max(1, round((tb - ta) / (0.085 * k)))
        n = int(turns * 6) + 1
        tw = sweep("twist", [(0, 0, lerp(ta, tb, i / n)) for i in range(n + 1)], [(-a, -a), (a, -a), (a, a), (-a, a)],
                   up=(1, 0, 0), smooth_path=False, twist_deg=360.0 * turns)
        smooth(tw, 20)
        body.append(tw)

    # ---- spun copper balls: two halves, each one continuous surface with a thin solder bead at the equator
    balls = []
    for nm, R, z in (("ball_big", R1, zb1), ("ball_small", R2, zb2)):
        bp = []
        for i in range(9):
            a = radians(-78 + 156 * i / 8)
            if i == 4:
                bp += [(R * 1.002, z - R * 0.03), (R * 1.026, z - R * 0.012), (R * 1.026, z + R * 0.012), (R * 1.002, z + R * 0.03)]
            else:
                bp.append((R * cos(a), z + R * sin(a)))
        b = lathe(nm, bp, segments=12, smooth_angle=75)
        displace_noise(b, strength=0.0012 * k, scale=0.035 * k, seed=sd + len(balls) * 3)
        balls.append(b)

    # ---- cardinal arms: rods through the drilled hub, bud tips, scrolled stays; each cut letter stands
    # in a saw cut on a short sleeve that is slid over its rod and riveted through
    La = p.arm_reach * k
    ra = 0.0068 * k
    fr = rng.fork("arms")
    letters_txt = (str(p.letters) + "NESW")[:4] if len(str(p.letters)) < 4 else str(p.letters)[:4]
    north = p.north + fr.jitter(0.0, 2.0)            # the cross is never set perfectly true
    lh = 0.088 * k                                   # letter height
    lth = max(0.003, 0.0034 * k)
    letter_parts, letter_rv = [], []
    lrv = lathe("letter_rivet", [(0.0038 * k, 0.0), (0.0, 0.0026 * k)], segments=6)
    bud_base = lathe("bud", [(ra * 0.9, 0.0), (ra * 1.9, 0.011 * k), (ra * 1.7, 0.021 * k), (ra * 0.5, 0.036 * k), (0.0, 0.046 * k)], segments=6)
    buds = []
    for i, (ch, ang) in enumerate(zip(letters_txt, (90, 0, 270, 180))):
        za = zc + (0.009 * k if i % 2 == 0 else -0.009 * k)      # N-S rod passes over the E-W rod
        droop = fr.jitter(0.6, 1.6)
        ar = radians(ang)
        d = (cos(ar), sin(ar))
        cz, sz = cos(radians(droop)), sin(radians(droop))
        end = (d[0] * La * cz, d[1] * La * cz, za - La * sz)
        body.append(sweep(f"arm{i}", [(-d[0] * 0.01 * k, -d[1] * 0.01 * k, za), end], ra, sides=8, smooth_path=False))
        buds.append(trs(end, (0, 90 + droop, ang)))
        # letter + sleeve are built on a rod along +-X (letter readable from -Y), then turned onto the arm:
        # E and W read from the south, N and S from the west
        sx = 1.0 if ang in (0, 270) else -1.0
        tilt = fr.jitter(0.0, 2.0)
        probe = text("probe", ch, size=0.1, depth=lth, font="serif", bevel=0.0)
        mn, mx = world_bbox([probe])
        delete(probe)
        lt = text(f"letter{i}", ch, size=0.1 * lh / max(mx.z - mn.z, 1e-4), depth=lth, rot=(90, tilt, 0), font="serif", bevel=0.0)
        realize(lt)
        apply_transform(lt)
        _lean(lt)
        mn, mx = world_bbox([lt])
        lw = mx.x - mn.x
        f = 0.69
        zl = za - La * f * sz
        lt.location = (sx * La * f - (mn.x + mx.x) / 2, -(mn.y + mx.y) / 2, zl + ra * 0.55 - mn.z)
        apply_transform(lt)
        sl_len = max(0.03 * k, lw * 0.86)
        slv = lathe(f"letter_sleeve{i}", [(ra + 0.0002 * k, -sl_len / 2), (ra + 0.0026 * k, -sl_len / 2 + 0.0015 * k), (ra + 0.0026 * k, sl_len / 2 - 0.0015 * k),
                                          (ra + 0.0002 * k, sl_len / 2)], segments=8, loc=(sx * La * f, 0, zl), rot=(0, 90, 0), smooth_angle=40)
        realize(slv)
        apply_transform(slv)
        rm = []
        for u in (-0.3, 0.3):
            for sy in (-1, 1):
                rm.append(trs((sx * La * f + u * sl_len, sy * (ra + 0.0022 * k), zl), (90 * sy * -1, 0, 0)))
        rvs = instances_to_mesh(f"letter_rivets{i}", lrv, rm)
        for o in (lt, slv, rvs):
            place(o, rot=(0, 0, ang if sx > 0 else ang - 180))
            apply_transform(o)
        letter_parts += [lt, slv]
        letter_rv.append(rvs)
        # scrolled stay from the spindle up under the arm
        st = _bracket_scroll(f"stay{i}", rs(zc) + 0.002 * k, zc - min(0.2 * k, (zc - zb2 - R2) * 0.8), La * 0.4, za - La * 0.4 * sz - ra - 0.002 * k,
                             0.024 * k * fr.vary(1.0, 0.08), 0.012 * k, 0.0042 * k, n_bez=7, n_curl=9)
        place(st, rot=(0, 0, ang))
        body.append(st)
    delete(lrv)
    body.append(instances_to_mesh("buds", bud_base, buds, keep_base=False))
    for o in body + letter_parts + letter_rv + balls:
        realize(o)
        apply_transform(o)
        place(o, rot=(0, 0, north))
        apply_transform(o)

    # ---- the turning part: sleeve, arrow bar, head, fletching, motif
    spin = []
    r_sl = 0.0145 * k
    if ornate:
        sl = [(0.011 * k, zs0), (0.0175 * k, zs0 + 0.003 * k), (0.019 * k, zs0 + 0.011 * k), (0.017 * k, zs0 + 0.02 * k), (r_sl, zs0 + 0.026 * k),
              (r_sl, zb + 0.2 * k), (0.019 * k, zb + 0.206 * k), (0.019 * k, zb + 0.214 * k), (0.012 * k, zb + 0.222 * k)]
        spin.append(lathe("sleeve", sl, segments=10))
        fz = zb + 0.222 * k
        fin = lathe("finial", [(0.011 * k, fz - 0.004 * k), (0.02 * k, fz + 0.006 * k), (0.026 * k, fz + 0.022 * k), (0.02 * k, fz + 0.038 * k), (0.009 * k, fz + 0.046 * k),
                               (0.013 * k, fz + 0.052 * k), (0.008 * k, fz + 0.058 * k), (0.004 * k, fz + 0.066 * k), (0.0, fz + 0.0785 * k)], segments=12)
        assign(fin, sheet)
        spin.append(fin)
        Lh, Lt, hl, tl = 0.40 * k, 0.38 * k, 0.21 * k, 0.27 * k
    else:
        sl = [(0.011 * k, zs0), (0.0175 * k, zs0 + 0.003 * k), (0.019 * k, zs0 + 0.011 * k), (0.017 * k, zs0 + 0.02 * k), (r_sl, zs0 + 0.026 * k),
              (r_sl, zb - 0.012 * k), (0.0175 * k, zb - 0.008 * k), (0.0175 * k, zb + 0.004 * k), (0.012 * k, zb + 0.009 * k), (0.0, zb + 0.0105 * k)]
        spin.append(lathe("sleeve", sl, segments=10))
        Lh, Lt, hl, tl = 0.36 * k, 0.35 * k, 0.14 * k, 0.185 * k
    sag = rng.fork("bar").jitter(0.002, 0.002) * k
    spin.append(sweep("arrow_bar", [(-Lt, 0, zb - sag), (-Lt * 0.4, 0, zb - sag * 0.2), (0, 0, zb), (Lh * 0.5, 0, zb), (Lh, 0, zb - sag * 0.5)], rb, sides=8, smooth_path=False))
    t3 = max(0.0028, 0.003 * k)
    head = _sheet("arrow_head", _outline(ARROW_HEAD, hl, Lh - 0.02 * k, zb), t3, k=k)
    slots = []
    for sgn in (1, -1):
        for x0 in (-0.30, -0.49, -0.68):
            q = [(x0, 0.10), (x0 + 0.075, 0.10), (x0 - 0.055, 0.375), (x0 - 0.13, 0.375)]
            slots.append([(-Lt + 0.03 * k + x * tl, zb + sgn * y * tl) for x, y in q])
    tail = _sheet("arrow_tail", _outline(ARROW_TAIL, tl, -Lt + 0.03 * k, zb), t3, slots, k=k)
    assign(head, sheet)
    assign(tail, sheet)
    spin += [head, tail]
    # the bar is split and the sheets riveted through it
    rvb = _rivet(0.004 * k, 0.0024 * k)
    rm = []
    bar_rv = [Lh + 0.005 * k, Lh + 0.04 * k, -Lt + 0.012 * k, -Lt - 0.03 * k, -Lt - 0.075 * k]
    for x in bar_rv:
        rm += [trs((x, -rb * 0.92, zb), (90, 0, 0)), trs((x, rb * 0.92, zb), (-90, 0, 0))]
    spin.append(instances_to_mesh("bar_rivets", rvb, rm, keep_base=False))
    mobjs, top, mrv = _motif(p, rng.fork("motif"), k, zb, rb, sheet, iron)
    spin += mobjs
    mrv = list(mrv) + [(x, zb) for x in bar_rv]
    for o in spin:
        if not o.data.materials:
            assign(o, iron)
        realize(o)
        apply_transform(o)

    # ---- weathering (the turning part is painted before it is swung into the wind, in its own XZ plane)
    def near(zs, w):
        zs = np.array(zs)
        return lambda co, nrm: np.exp(-np.abs(co[:, 2:3] - zs[None, :]) / w).max(axis=1)

    under = facing((0, 0, -1))
    brk = noise_mask(0.028 * k, seed=sd + 5, lo=0.2, hi=0.75)
    blot = noise_mask(0.09 * k, seed=sd + 6, lo=0.75 - 0.6 * rust, hi=1.15 - 0.6 * rust)
    jn = near(joints, 0.016 * k)

    def iron_rust(co, nrm):                          # rust blooms from the joints, rod tips and scroll ends
        rr = np.sqrt(co[:, 0] ** 2 + co[:, 1] ** 2)
        tip = np.clip((rr - 0.78 * La) / (0.2 * La), 0.0, 1.0) * (np.abs(co[:, 2] - zc) < 0.06 * k)
        scroll = np.exp(-np.abs(rr - 0.44 * La) / (0.03 * k)) * ((co[:, 2] < zc - 0.012 * k) & (co[:, 2] > zc - 0.07 * k))
        src = np.clip(jn(co, nrm) * (rr < 0.05 * k) + tip + 0.9 * scroll, 0.0, 1.0)
        return np.clip((0.4 + 0.6 * rust) * src * (0.15 + 0.85 * brk(co, nrm)) + 0.7 * blot(co, nrm) * brk(co, nrm), 0.0, 0.8)

    def spin_rust(co, nrm):                          # bar ends, rivets and the sleeve foot
        ends = np.clip((np.abs(co[:, 0]) - 0.27 * k) / (0.08 * k), 0.0, 1.0)
        foot = np.exp(-np.abs(co[:, 2] - zs0) / (0.02 * k))
        return np.clip((0.4 + 0.6 * rust) * np.maximum(ends, foot) * (0.15 + 0.85 * brk(co, nrm)) + 0.7 * blot(co, nrm) * brk(co, nrm), 0.0, 0.8)

    chips = mask_mul(noise_mask(0.012 * k, seed=sd + 7, lo=0.6, hi=0.85), 0.5)
    for o in body:
        assign(o, iron)
        paint(o, special=iron_rust, wear=chips, grime=mask_mul(under, 0.45))
    for o in spin:
        if o.data.materials[0] == iron:
            paint(o, special=spin_rust, wear=chips, grime=mask_mul(under, 0.45))

    # the big ball was never gilded: brown oxide with verdigris creeping from the seam and the rain side
    assign(balls[0], copper_ball)
    lo, hi = 0.85 - 0.75 * v, 1.1 - 0.65 * v
    paint(balls[0], special=mask_mul(noise_mask(0.06 * k, seed=sd + 9, lo=lo, hi=hi), 0.45 * min(1.0, 0.15 + v)),
          wear=mask_mul(noise_mask(0.05 * k, seed=sd + 2, lo=0.5, hi=0.9), 0.12), grime=mask_mul(under, 0.3), mode="set")

    # turned / small sheet-finish parts (upper ball, letters, sleeves, rivets, finial): leaf or oxide bloom
    # thinned by a slow noise, verdigris hanging on the undersides and under the solder bead
    bloom = noise_mask(0.05 * k, seed=sd + 12, lo=0.1, hi=0.9)
    fine = noise_mask(0.016 * k, seed=sd + 13, lo=0.2, hi=0.9)
    base_g = 1.0 if p.gilt else 0.75

    def small_masks(z_lo, z_hi, rub=0.0):
        hgt = max(z_hi - z_lo, 1e-4)

        def s(co, nrm):
            low = np.clip(1.0 - (co[:, 2] - z_lo) / (0.45 * hgt), 0.0, 1.0)
            dn = np.clip(-nrm[:, 2], 0.0, 1.0)
            return np.clip((0.2 + 1.0 * v) * (0.7 * low + 0.5 * dn) * (0.35 + 0.65 * fine(co, nrm)), 0.0, 0.42 + 0.4 * v)

        def g(co, nrm):                              # no dirt under the patina: it would blacken it
            return np.clip((base_g * (0.85 + 0.15 * bloom(co, nrm)) - rub * fine(co, nrm)) * (1.0 - 0.9 * np.clip((s(co, nrm) - 0.3) / 0.25, 0.0, 1.0)), 0.0, 1.0)
        return g, s

    # upper ball: one even, matt skin of leaf / oxide over both halves (a bright metal ball mirrors the
    # horizon as a hard two-tone line), verdigris only under the belly
    g, s = small_masks(zb2 - R2, zb2 + R2)
    assign(balls[1], sheet)
    paint(balls[1], grime=lambda co, nrm: ((0.9 if p.gilt else 0.8) + 0.1 * bloom(co, nrm)) * (1.0 - 0.9 * np.clip((0.7 * s(co, nrm) - 0.3) / 0.25, 0.0, 1.0)),
          special=mask_mul(s, 0.7), wear=0.0, mode="set")
    for o in letter_parts + letter_rv:
        mn, mx = world_bbox([o])
        g, s = small_masks(mn.z, mx.z, rub=0.12)
        assign(o, sheet)
        paint(o, grime=g, special=s, wear=0.1, mode="set")
    for o in spin:
        if o.data.materials[0] != sheet:
            continue
        if o.name in SHEETS:
            tone = 0.78 if o.name.startswith(("rooster_wing", "comet_tail", "sail", "pennant")) else 1.0   # overlaid sheets: older, thinner leaf
            g, w, s = _sheet_masks(SHEETS[o.name], mrv, k, v, p.gilt, rng.fork("streak_" + o.name), sd + len(o.name), tone)
            paint(o, grime=g, wear=w, special=s, mode="set")
        else:
            mn, mx = world_bbox([o])
            g, s = small_masks(mn.z, mx.z, rub=0.2)
            paint(o, grime=g, special=s, wear=0.15, mode="set")

    for o in spin:
        place(o, rot=(0, 0, p.wind))
        apply_transform(o)

    # A vane holds its heading and hunts a few degrees about it; it does not whirl. ('spin' turns the part
    # continuously from wall-clock time, so the rest heading - and every review render - came out random;
    # 'swing' about the engine's default axis, which lower-case "z" selects, is the yaw about the spindle.)
    part(spin, "vane", pivot=(0, 0, zb), motion="swing", axis="z", amp=5.0, speed=0.07)
    socket("pivot", "attach", (0, 0, zb), role="prop_top")
