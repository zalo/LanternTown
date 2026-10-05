"""Roof dormer built for a main roof of `pitch` degrees. Origin = centre of the front base on the roof
surface (the roof plane is z = y * tan(pitch)); cheeks and roof run back (+Y) until they die into the
slope. Pegged oak front (sill beam, corner posts, tie beam) around a painted casement with glazing
bars, curtains and a lit room behind; gable / hip / shed roof in individually laid clay tiles or slates
(cut to open lead valleys / hips) or batten-rolled lead / copper sheet, or a swept 'eyebrow' whose
sheet-metal curve runs out in ogee tails onto the main roof; half-round ridge tiles or wood-cored lead
rolls, bargeboards with finial, fascias; cheeks tile-hung, lead-sheeted or plastered in a timber frame;
thin dressed leadwork: bossed apron with clips, stepped side flashings, valley / top flashings.
`roof_patch` > 0 adds a piece of the surrounding main roof for presentation only (leave 0 on a building).
"""
from math import ceil, acos, atan, atan2, hypot, tan, sin, cos, radians, degrees, pi, log

import numpy as np
from mathutils import Matrix
from tk import *

META = {"name": "Roof dormer", "class": "component", "mount": "ground", "allow_below_ground": True,
        "focus": 0.45}
PARAMS = {
    "width": {"default": 1.2, "min": 0.7, "max": 2.6, "desc": "width over the cheeks (m); eyebrow tails sweep out beyond it"},
    "height": {"default": 1.9, "min": 0.9, "max": 2.6, "desc": "height of the front, base to ridge / top of roof (m)"},
    "pitch": {"default": 45.0, "min": 25.0, "max": 65.0, "desc": "pitch of the main roof the dormer sits on (deg)"},
    "roof": {"default": "gable", "choices": ["gable", "hip", "shed", "eyebrow"], "desc": "dormer roof form"},
    "roof_pitch": {"default": 50.0, "min": 30.0, "max": 65.0, "desc": "pitch of the dormer's own gable / hip roof (deg); shed uses a fraction"},
    "cover": {"default": "tile", "choices": ["tile", "slate", "lead", "copper"], "desc": "roof covering (eyebrow is always sheet metal)"},
    "cladding": {"default": "tile", "choices": ["tile", "lead", "plaster"], "desc": "cheek cladding"},
    "lit": {"default": True, "desc": "room behind the window is lit"},
    "glow": {"default": 1.5, "min": 0.3, "max": 6.0, "desc": "brightness of the lit room seen through the glass"},
    "intensity": {"default": 4.0, "min": 0.0, "max": 60.0, "desc": "window spill light (cd)"},
    "curtain": {"default": 0.42, "min": 0.0, "max": 1.0, "desc": "how far the curtains are drawn"},
    "ajar": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "how far one casement leaf stands open"},
    "barge": {"default": "plain", "choices": ["plain", "scalloped"], "desc": "gable bargeboard style"},
    "gable_infill": {"default": "plaster", "choices": ["plaster", "boards"], "desc": "infill of the gable / under-window panels"},
    "flower_box": {"default": True, "desc": "planted window box on iron brackets"},
    "flowers": {"default": "#c9304a", "desc": "flower colour"},
    "timber": {"default": "#5d4533", "desc": "stain of the oak frame"},
    "paint": {"default": "#2f4a3d", "desc": "paint of the casement, bargeboards and fascias"},
    "plaster": {"default": "#e3d3aa", "desc": "limewash colour of plaster panels"},
    "tile_color": {"default": "#c98f78", "desc": "tint of the clay tiles"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: moss, slipped tiles, flaking paint, patina"},
    "roof_patch": {"default": 0.0, "min": 0.0, "max": 1.2, "desc": "presentation only: margin (m) of surrounding main roof to add; 0 on a building"},
}
VARIANTS = {
    "hip_slate": {"roof": "hip", "cover": "slate", "cladding": "lead", "width": 1.05, "height": 1.6, "pitch": 50,
                  "roof_pitch": 45, "paint": "#b3a888", "flower_box": False, "ajar": 0.0, "curtain": 0.7, "intensity": 1.4, "roof_patch": 0.45,
                  "seed": 4},
    "shed_wide": {"roof": "shed", "width": 2.2, "height": 1.25, "pitch": 42, "cladding": "plaster", "paint": "#5f2620",
                  "flowers": "#e8c23a", "ajar": 0.6, "age": 0.7, "curtain": 0.3, "intensity": 2.5, "seed": 11},
    "eyebrow_copper": {"roof": "eyebrow", "cover": "copper", "width": 1.7, "height": 1.05, "pitch": 40,
                       "glow": 1.3, "intensity": 2.0, "curtain": 0.25, "paint": "#34465a", "flower_box": False,
                       "gable_infill": "boards", "ajar": 0.0, "roof_patch": 0.4, "age": 0.8, "seed": 23},
    "tall_scalloped": {"width": 0.95, "height": 2.2, "roof_pitch": 60, "pitch": 52, "barge": "scalloped",
                       "gable_infill": "boards", "cladding": "plaster", "paint": "#5f2620", "timber": "#3d2e24",
                       "flowers": "#f1ece0", "age": 0.9, "curtain": 0.6, "intensity": 2.5, "roof_patch": 0.5, "seed": 31},
}


# ------------------------------------------------------------------------------------------ helpers
def _dedupe(pts):
    out = []
    for q in pts:
        q = Vector(q)
        if not out or (q - out[-1]).length > 1e-5:
            out.append(q)
    if len(out) > 1 and (out[0] - out[-1]).length < 1e-5:
        out.pop()
    return out


def prism(name, pts, off, bev=0.0):
    """Solid from a planar 3D polygon swept by the vector `off`."""
    pts = _dedupe(pts)
    n = len(pts)
    off = Vector(off)
    vs = pts + [q + off for q in pts]
    fs = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    if bev > 0:
        bevel(o, bev, segments=1)
    return o


def mesh_uv(name, V, F, UV):
    """Mesh with explicit per-vertex UVs (metres)."""
    o = mesh_object(name, V, F)
    me = o.data
    uvl = me.uv_layers[0] if me.uv_layers else me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            uvl.data[li].uv = UV[me.loops[li].vertex_index]
    uv_keep(o)
    return o


def clip_x(poly, x0, x1):
    """Clip a 2D polygon to the strip x0 <= x <= x1 (Sutherland-Hodgman)."""
    def clip(pts, xc, ge):
        out = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ia = (a[0] >= xc) if ge else (a[0] <= xc)
            ib = (b[0] >= xc) if ge else (b[0] <= xc)
            if ia:
                out.append(a)
            if ia != ib:
                f = (xc - a[0]) / (b[0] - a[0])
                out.append((xc, a[1] + f * (b[1] - a[1])))
        return out
    q = clip(list(poly), x0, True)
    return clip(q, x1, False) if len(q) >= 3 else []


def ribbon_solid(name, ra, rb, off):
    """Closed solid band between two matching 3D point rows, swept by `off` (curved fascias)."""
    off = Vector(off)
    vs = []
    for a, b in zip(ra, rb):
        a, b = Vector(a), Vector(b)
        vs += [a, b, a + off, b + off]
    fs = []
    n = len(ra)
    for i in range(n - 1):
        k, m = 4 * i, 4 * (i + 1)
        fs += [(k, m, m + 1, k + 1), (k + 2, k + 3, m + 3, m + 2), (k, k + 2, m + 2, m), (k + 1, m + 1, m + 3, k + 3)]
    e = 4 * (n - 1)
    fs += [(0, 1, 3, 2), (e, e + 2, e + 3, e + 1)]
    o = mesh_object(name, vs, fs)
    recalc_normals(o)
    return o


def sheet_mesh(name, rows, up, th=0.0025):
    """Thin dressed sheet metal: rows of 3D points (equal length) give the top surface, a skirt of
    thickness `th` round the perimeter gives the visible edge. `up` = rough outward direction."""
    P = np.array([[tuple(Vector(q)) for q in r] for r in rows], dtype=float)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    N = np.cross(du, dv)
    if float((N.reshape(-1, 3) @ np.array(tuple(up), dtype=float)).sum()) < 0.0:
        P = P[:, ::-1].copy()
        du = np.gradient(P, axis=1)
        dv = np.gradient(P, axis=0)
        N = np.cross(du, dv)
    N /= (np.linalg.norm(N, axis=2, keepdims=True) + 1e-12)
    nr, nc = P.shape[:2]
    V = [Vector(P[i, j]) for i in range(nr) for j in range(nc)]
    F = [(i * nc + j, i * nc + j + 1, (i + 1) * nc + j + 1, (i + 1) * nc + j) for i in range(nr - 1) for j in range(nc - 1)]
    loop = [(0, j) for j in range(nc)] + [(i, nc - 1) for i in range(1, nr)] + [(nr - 1, j) for j in range(nc - 2, -1, -1)] + \
           [(i, 0) for i in range(nr - 2, 0, -1)]
    b0 = len(V)
    for (i, j) in loop:
        V.append(Vector(P[i, j] - N[i, j] * th))
    m = len(loop)
    for k in range(m):
        a = loop[k][0] * nc + loop[k][1]
        b = loop[(k + 1) % m][0] * nc + loop[(k + 1) % m][1]
        F.append((a, b0 + k, b0 + (k + 1) % m, b))
    return mesh_object(name, V, F)


def tile_field(name, O, A, U, s0, s1, spans, tw, gauge, thick, rng, base=0.0, slip=0.0, gap=0.0015, chip=0.0, max_tiles=0):
    """Individually laid tiles / slates on the plane O + A*r + U*s (A x U = outward normal), one mesh.

    spans(s) -> [(lo, hi), ...] run-intervals covered at slope position s; tiles at the ends of every
    interval are cut to the (possibly raking) boundary, so valleys, hips and verges are followed exactly.
    Each tile is tilted (tail one thickness proud, head on the batten), has an open joint to its
    neighbours, its own UV offset and a random tone (returned per vertex for the weathering masks).
    max_tiles > 0 caps the tile count: very large fields get a proportionally larger tile format (budget guard)."""
    N = A.cross(U)
    V, F, UV, R = [], [], [], []
    if max_tiles > 0 and s1 > s0:
        ds = (s1 - s0) / 24
        area = sum(sum(max(h_ - l_, 0.0) for l_, h_ in spans(s0 + (i + 0.5) * ds)) for i in range(24)) * ds
        est = area / (tw * gauge)
        if est > max_tiles:
            k_ = (est / max_tiles) ** 0.5
            tw, gauge = tw * k_, gauge * k_
    nc = max(1, int(round((s1 - s0) / gauge)))
    g = (s1 - s0) / nc

    def P(r, s, h):
        return O + A * r + U * s + N * h

    for k in range(nc):
        sa, sb = s0 + k * g, s0 + (k + 1) * g
        La, Lb = spans(sa + 1e-4), spans(sb - 1e-4)
        if len(La) != len(Lb):
            La = Lb = spans(0.5 * (sa + sb))
        off = (0.5 * tw if k % 2 else 0.0) + rng.u(-0.012, 0.012)
        for (la, ha), (lb, hb) in zip(La, Lb):
            if max(ha - la, hb - lb) < 0.015:
                continue
            lo_in, hi_in = max(la, lb), min(ha, hb)
            edges = [(la, lb)]
            if hi_in > lo_in:
                y = off + ceil((lo_in + 0.3 * tw - off) / tw) * tw
                while y < hi_in - 0.3 * tw:
                    edges.append((y, y))
                    y += tw
            edges.append((ha, hb))
            ne = len(edges) - 1
            for ie in range(ne):
                (a0, b0), (a1, b1) = edges[ie], edges[ie + 1]
                if a1 - a0 < 0.004 and b1 - b0 < 0.004:
                    continue
                first, last = ie == 0, ie == ne - 1
                if not first and min(a1 - a0, b1 - b0) > 0.03:
                    a0, b0 = a0 + gap, b0 + gap
                if not last and min(a1 - a0, b1 - b0) > 0.03:
                    a1, b1 = a1 - gap, b1 - gap
                j = rng.u(-0.004, 0.004)
                dh = rng.u(0.0, 0.003)
                slipped = slip > 0 and rng.chance(0.05 * slip)
                if slipped:
                    j -= rng.u(0.012, 0.03)
                    dh += 0.003
                tone = rng.u(0.0, 0.2) if rng.u() < 0.7 else rng.u(0.25, 0.48)
                h0 = base + thick + dh
                h1 = base + 0.001
                sl = sa + j
                ru, rv = rng.u(0, 1), rng.u(0, 1)
                hs = lambda s_: h0 + (h1 - h0) * (s_ - sl) / (sb - sl)
                if a1 - a0 < 0.004:        # triangle with its point down
                    tp, tail = [(a0, sl), (b1, sb), (b0, sb)], False
                elif b1 - b0 < 0.004:      # triangle with its point up
                    tp, tail = [(a0, sl), (a1, sl), (b0, sb)], True
                else:
                    tp, tail = [(a0, sl), (a1, sl), (b1, sb), (b0, sb)], True
                    if chip > 0 and a1 - a0 > 0.12 and rng.chance(chip):       # knocked-off tail corner
                        c = rng.u(0.02, 0.045)
                        f = c / (sb - sl)
                        if rng.chance(0.5):
                            tp = [(a0 + c, sl), (a1, sl), (b1, sb), (b0, sb), (a0 + (b0 - a0) * f, sl + c)]
                        else:
                            tp = [(a0, sl), (a1 - c, sl), (a1 + (b1 - a1) * f, sl + c), (b1, sb), (b0, sb)]
                i0 = len(V)
                nt = len(tp)
                for (r_, s_) in tp:
                    V.append(P(r_, s_, hs(s_)))
                    UV.append((r_ + ru, s_ + rv))
                F.append(tuple(range(i0, i0 + nt)))
                if tail:
                    V += [P(tp[0][0], sl, h0 - thick), P(tp[1][0], sl, h0 - thick)]
                    UV += [(tp[0][0] + ru, sl + rv - thick), (tp[1][0] + ru, sl + rv - thick)]
                    kb = i0 + nt
                    F.append((kb, kb + 1, i0 + 1, i0))
                    if first or slipped:
                        F.append((i0, i0 + nt - 1, kb))
                    if last or slipped:
                        F.append((i0 + 1, kb + 1, i0 + 2))
                    nt += 2
                R += [tone] * nt
    if not F:
        return None
    o = mesh_uv(name, V, F, UV)
    flat(o)
    return o, np.array(R, dtype=float)


def sheet_rolls(name, O, A, U, s0, s1, spans, spacing, r, lift):
    """Batten rolls of a lead / copper sheet roof running up the slope, clipped to the covered region."""
    N = A.cross(U)
    ss = [s0 + (s1 - s0) * i / 30 for i in range(31)]
    sp = [spans(s) for s in ss]
    lo = min(a for L in sp for a, b in L)
    hi = max(b for L in sp for a, b in L)
    n = max(2, int(round((hi - lo) / spacing)))
    out = []
    for k in range(n + 1):
        rk = lo + (hi - lo) * k / n
        rk = min(max(rk, lo + r), hi - r)
        ok = [s for s, L in zip(ss, sp) if any(a - 1e-6 <= rk <= b + 1e-6 for a, b in L)]
        if len(ok) < 2 or ok[-1] - ok[0] < 0.1:
            continue
        out.append(sweep(name, [O + A * rk + U * ok[0] + N * lift, O + A * rk + U * ok[-1] + N * lift], r, sides=6,
                         smooth_path=False))
    return out


def ridge_run(name, p0, p1, r, rng, seg=0.3):
    """Half-round ridge / hip tiles laid with lapped, slightly tapering joints from p0 to p1."""
    p0, p1 = Vector(p0), Vector(p1)
    L = (p1 - p0).length
    d = (p1 - p0) / L
    n = max(1, int(round(L / max(seg, L / 9.0))))
    outer = circle_pts(r, 7, -18.0, 198.0)
    inner = circle_pts(r - 0.015, 7, -18.0, 198.0)
    prof = list(outer) + list(inner)[::-1]
    out = []
    for i in range(n):
        a = p0 + d * (i * L / n - (0.025 if i else 0.0))
        b = p0 + d * ((i + 1) * L / n)
        jz = Vector((0, 0, rng.u(-0.003, 0.003)))
        out.append(sweep(name, [a + jz, b + jz], prof, up=(0, 0, 1), smooth_path=False,
                         scale_fn=lambda u: 1.05 - 0.09 * u))
    return out


def lead_ridge(name, p0, p1, r, wing, ang, lap=0.75):
    """Wood-cored lead roll with wings dressed down both roof faces and welted laps (slate ridges / hips)."""
    p0, p1 = Vector(p0), Vector(p1)
    L = (p1 - p0).length
    d = (p1 - p0) / L
    ta = tan(ang)
    c = r * 0.75
    arc = [(r * cos(a), c + r * sin(a)) for a in np.radians(np.linspace(-35.0, 215.0, 8))]
    prof = [(wing, -wing * ta), (r * 1.05, -r * 1.05 * ta + 0.002)] + arc + \
           [(-r * 1.05, -r * 1.05 * ta + 0.002), (-wing, -wing * ta), (-wing, -wing * ta - 0.003), (0.0, -0.004), (wing, -wing * ta - 0.003)]
    out = [sweep(name, [p0, p1], prof, up=(0, 0, 1), smooth_path=False)]
    big = [(a * 1.1, b * 1.1 + 0.003) for a, b in prof]
    n = max(1, int(round(L / lap)))
    for i in range(1, n + 1):
        q = p0 + d * (L * (i - 0.5) / n)
        out.append(sweep(name + "_welt", [q - d * 0.02, q + d * 0.02], big, up=(0, 0, 1), smooth_path=False))
    return out


def edge_dirt(pts_xz, k=0.04, amp=0.5):
    """Mask: dirt wash hugging the outline of a panel (polygon in x, z)."""
    Pq = np.array(pts_xz, dtype=float)

    def f(co, n):
        q = co[:, [0, 2]]
        d = np.full(len(q), 1e9)
        m = len(Pq)
        for i in range(m):
            a, b = Pq[i], Pq[(i + 1) % m]
            ab = b - a
            tt = np.clip(((q - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
            d = np.minimum(d, np.linalg.norm(q - (a + tt[:, None] * ab), axis=1))
        return amp * np.exp(-d / k)
    return f


# -------------------------------------------------------------------------------------------- build
def build(p, rng):
    th = radians(p.pitch)
    t, ct, st = tan(th), cos(th), sin(th)
    w = float(p.width)
    H = float(p.height)
    hw2 = w / 2
    kind = p.roof
    cover = p.cover
    if kind == "eyebrow" and cover in ("tile", "slate"):
        cover = "lead"
    sheet = cover in ("lead", "copper")
    age = float(p.age)
    seed = int(p.seed)

    m_timber = mat("wood_oak", p.timber)
    m_paint = mat("wood_painted", p.paint, scale=5.0, rough=1.6, wear=0.6)
    m_lead = mat("lead", "#5c646d", scale=5.0, rough=1.6, wear=0.3)
    m_plaster = mat("plaster_lime", p.plaster, scale=2.5)
    slate = cover == "slate"
    if slate:
        m_tile = mat("slate", wear=0.12, rough=0.85)
        TW, GA, TT = 0.26, 0.17, 0.007
    else:
        m_tile = mat("terracotta", p.tile_color)
        TW, GA, TT = 0.165, 0.105, 0.014
    m_sheet = mat("copper_patina") if cover == "copper" else m_lead
    if p.lit:
        m_room = mat("window_lit", glow=float(p.glow), curtain=0.0, rough=2.0)
        m_curt = mat("window_lit", glow=0.55 * float(p.glow), curtain=1.0, rough=2.0)
    else:
        m_room = mat("window_dark")
        m_curt = None

    T, PT, LD, PL, TL, SH, RT, WIN = [], [], [], [], [], [], [], []   # timber, painted, lead, plaster, tiles, sheet, ridge tiles, window
    SHP = []                                                         # (objects, O, A, U, S) of flat sheet-metal planes

    def tb(name, size, loc, grain="X", bev=0.005, lst=None):
        o = box(name, size, loc, bevel=bev, segments=1)
        uv_grain(o, axis=grain)
        (T if lst is None else lst).append(o)
        return o

    def PM(x, y, lift=0.0):                      # point on the main roof plane, lifted along its normal
        return Vector((x, y - st * lift, y * t + ct * lift))

    DK = 0.028          # sarking board deck
    post = 0.095        # corner post section
    sbh = 0.09          # sill beam height
    hbh = 0.10          # tie / head beam height
    xe = hw2 + 0.09     # roof edge incl. eaves overhang
    VG = 0.0 if sheet else 0.045                 # tiles are cut back from the open lead valley
    r_tiles = rng.fork("tiles")
    nm = Vector((0, -st, ct))                    # main roof normal
    dsl = Vector((0, ct, st))                    # up the main roof

    def cover_field(name, O, A, U, s0, s1, dspan, tspan, grain):
        """deck boards + covering for one roof plane; dspan = deck outline spans, tspan = covering spans"""
        N = A.cross(U)
        (a0, b0), (a1, b1) = dspan(0.0)[0], dspan(s1)[0]
        pts = [O + A * a0, O + A * b0, O + A * b1 + U * s1, O + A * a1 + U * s1]
        dk = prism(name + "_deck", pts, -N * DK)
        uv_grain(dk, axis=grain)
        T.append(dk)
        if sheet:
            sh = prism(name + "_sheet", [q + N * 0.001 for q in pts], N * 0.006)
            if cover == "copper":
                densify(sh, max(0.1, max((q - pts[0]).length for q in pts) / 13.0))
            rolls = sheet_rolls(name + "_roll", O, A, U, 0.0, s1, dspan, 0.42, 0.018, 0.011)[:14]
            SH.append(sh)
            SH.extend(rolls)
            SHP.append(([sh] + rolls, O, A, U, s1))
        else:
            tf = tile_field(name + "_tiles", O, A, U, s0, s1, tspan, TW, GA, TT, r_tiles, slip=age,
                            chip=0.16 if slate else 0.0, gap=0.002 if slate else 0.0015, max_tiles=420 if kind == "shed" else 210)
            if tf:
                TL.append(tf)

    cheeks = kind != "eyebrow"
    head_y = 0.045
    # =============================================================== roof forms
    if kind in ("gable", "hip"):
        trp = tan(radians(p.roof_pitch))
        top = 0.05
        hw = H - top - hw2 * trp
        if hw < 0.52 * H:
            hw = 0.52 * H
            trp = (H - top - hw) / hw2
        rp = atan(trp)
        cr, sr = cos(rp), sin(rp)
        za = hw + hw2 * trp                       # deck apex
        zedge = hw - 0.09 * trp                   # deck at the eaves edge
        S = xe / cr
        yr = za / t
        yfe = -0.16 if kind == "gable" else -0.13
        runf = min(xe, 0.6 * (yr - yfe))
        yh = yfe + runf
        yedge = zedge / t
        zc0, tc = hw - DK / cr, 0.0
        head_len = w - 0.004

        def side_span(s, sx, ext=0.0, vg=0.0):
            yb = (zedge + s * sr) / t - vg
            yl = (yfe + runf * clamp(s / S)) if kind == "hip" else yfe - ext
            yb = max(yb, yl)
            return (yl, yb) if sx > 0 else (-yb, -yl)

        for sx in (1, -1):
            O = Vector((sx * xe, 0, zedge))
            A = Vector((0, sx, 0))
            U = Vector((-sx * cr, 0, sr))
            cover_field(f"slope{sx}", O, A, U, -0.03, S - 0.02, lambda s, sx=sx: [side_span(s, sx)],
                        lambda s, sx=sx: [side_span(s, sx, 0.035, VG)], "Y")
            # eaves fascia cut to the main roof
            zt, zb = zedge - 0.012, zedge - 0.115
            f = prism("fascia_side", [(sx * (xe - 0.02), yfe, zt), (sx * (xe - 0.02), zt / t, zt), (sx * (xe - 0.02), zb / t, zb),
                                      (sx * (xe - 0.02), yfe, zb)], (sx * 0.02, 0, 0), bev=0.003)
            uv_grain(f, axis="Y")
            PT.append(f)
        if kind == "hip":
            rpf = atan2(za - zedge, runf)
            Sf = hypot(za - zedge, runf)
            O = Vector((0, yfe, zedge))
            A = Vector((1, 0, 0))
            U = Vector((0, cos(rpf), sin(rpf)))
            fs = lambda s: [(-xe * (1 - clamp(s / Sf)), xe * (1 - clamp(s / Sf)))]
            cover_field("hipface", O, A, U, -0.03, Sf - 0.02, fs, fs, "X")
            tb("fascia_front", (2 * xe, 0.02, 0.103), (0, yfe + 0.01, zedge - 0.0635), "X", 0.003, PT)
        # ridge + hips
        y0r = yfe - 0.035 if kind == "gable" else yh - 0.04
        y1r = (za - 0.035) / t
        rr = rng.fork("ridge")
        if sheet:
            SH.append(sweep("ridge_roll", [(0, y0r + 0.03, za + 0.012), (0, yr - 0.02, za + 0.012)], 0.025, sides=8, smooth_path=False))
            if kind == "hip":
                for sx in (1, -1):
                    SH.append(sweep("hip_roll", [(sx * xe, yfe, zedge + 0.014), (0, yh, za + 0.014)], 0.022, sides=8, smooth_path=False))
        elif slate:                                # wood-cored lead rolls, wings dressed over the slates
            LD.extend(lead_ridge("ridge_lead", (0, y0r, za + 0.017), (0, y1r + 0.05, za + 0.017), 0.025, 0.09, rp))
            if kind == "hip":
                for sx in (1, -1):
                    LD.extend(lead_ridge("hip_lead", (sx * (xe + 0.004), yfe - 0.004, zedge + 0.017), (0, yh + 0.01, za + 0.02),
                                         0.022, 0.08, 0.55 * rp))
        else:
            RT.extend(ridge_run("ridge_tile", (0, y0r, za - 0.03), (0, y1r, za - 0.03), 0.088, rr))
            if kind == "hip":
                for sx in (1, -1):
                    RT.extend(ridge_run("hip_tile", (sx * (xe + 0.01), yfe - 0.01, zedge - 0.01), (0, yh, za - 0.03), 0.08, rr))
        J = [(-xe, yedge), (0.0, yr), (xe, yedge)]
        ytop = yr

        def zdorm(x, y):
            if abs(x) > xe + 0.04 or y < yfe - 0.06:
                return -1e9
            z = za - abs(x) * trp
            return min(z, zedge + (y - yfe) * (za - zedge) / runf) if kind == "hip" else z

        def excl(y):
            if y < 0 or y > yr:
                return 0.0
            if y < yedge:
                return hw2 - 0.012
            return xe * (1 - (y - yedge) / (yr - yedge)) + 0.03

    elif kind == "shed":
        spd = clamp(min(p.roof_pitch * 0.42, p.pitch - 14.0), 6.0, 32.0)
        sp = radians(spd)
        tsp, cs, ss_ = tan(sp), cos(sp), sin(sp)
        hw = H - 0.045
        yfe = -0.14
        zfe = hw + yfe * tsp
        yj = hw / (t - tsp)
        Ssh = (yj - yfe) / cs
        O = Vector((-xe, yfe, zfe))
        A = Vector((1, 0, 0))
        U = Vector((0, cs, ss_))
        cover_field("shed", O, A, U, -0.03, Ssh, lambda s: [(0.0, 2 * xe)], lambda s: [(-0.035, 2 * xe + 0.035)], "X")
        tb("fascia_front", (2 * xe + 0.044, 0.022, 0.11), (0, yfe - 0.011, zfe - 0.067), "X", 0.003, PT)
        for sx in (1, -1):
            topl = lambda y: hw + 0.004 + y * tsp
            y2 = (hw + 0.004) / (t - tsp)
            y3 = (hw + 0.004 - 0.125) / (t - tsp)
            ya = yfe - 0.022
            bg = prism("barge_side", [(sx * xe, ya, topl(ya)), (sx * xe, y2, y2 * t), (sx * xe, y3, y3 * t), (sx * xe, ya, topl(ya) - 0.125)],
                       (sx * 0.022, 0, 0), bev=0.003)
            uv_grain(bg, axis=(0, cs, ss_))
            PT.append(bg)
        J = [(-xe, yj), (xe, yj)]
        ytop = yj
        zc0, tc = hw - DK / cs, tsp
        head_len = w + 0.05
        head_y = 0.066

        def zdorm(x, y):
            if abs(x) > xe + 0.05 or y < yfe - 0.06:
                return -1e9
            return hw + y * tsp

        def excl(y):
            if y < 0 or y > yj + 0.04:
                return 0.0
            return hw2 - 0.012 if y < yj - 0.12 else xe + 0.035

    else:   # eyebrow: one swept sheet-metal curve, a flat-topped bell whose ogee tails die into the main roof
        tailw = clamp(0.5 * w, 0.35, 0.95)
        Xt = hw2 + tailw                          # half width over the tails
        zend = 0.03
        Htop = H - 0.03
        kexp = log(acos(2 * 0.72 - 1) / pi) / log(hw2 / Xt)      # bell is at 72 % of its height over the posts

        def Zx(x):
            u = min(abs(x) / Xt, 1.0)
            return zend + (Htop - zend) * 0.5 * (1.0 + cos(pi * u ** kexp))

        def Xof(z):
            b = clamp((z - zend) / (Htop - zend))
            return Xt * (acos(2 * b - 1) / pi) ** (1.0 / kexp)

        hw = min(Zx(hw2) - 0.07, Zx(hw2 + 0.035) - 0.03)     # head beam ends stay under a steep, narrow bell
        yfe = -0.055

        def yfx(x):                               # front edge: small verge overhang fading out along the tails
            return yfe * float(smoothstep(0.0, 0.3, Zx(x) - zend))

        def eb(u, v):
            x = (2 * u - 1) * Xt
            z = Zx(x)
            y0 = yfx(x)
            return (x, y0 + v * (z / t - y0), z + 0.006)

        nv = int(clamp(round((Htop / t - yfe) / 0.065), 8, 26))
        nu = 28                                   # UVs run with the fall (V = arc length up from the tail, U = depth)
        xs_r = [(2 * i / nu - 1) * Xt for i in range(nu + 1)]
        arc = [0.0]
        for i in range(1, nu // 2 + 1):
            arc.append(arc[-1] + hypot(xs_r[i] - xs_r[i - 1], Zx(xs_r[i]) - Zx(xs_r[i - 1])))
        Vr, UVr, Fr = [], [], []
        for i in range(nu + 1):
            for j in range(nv + 1):
                pt = eb(i / nu, j / nv)
                Vr.append(pt)
                UVr.append((pt[1] + (0.37 if i > nu // 2 else 0.0), arc[min(i, nu - i)]))
        for i in range(nu):
            for j in range(nv):
                a = i * (nv + 1) + j
                Fr.append((a, a + nv + 1, a + nv + 2, a + 1))
        ro = mesh_uv("eyebrow_roof", Vr, Fr, UVr)
        smooth(ro, 60)
        SH.append(ro)
        xs_e = np.linspace(-Xt + 0.012, Xt - 0.012, 25)
        bead = sweep("eb_bead", [(float(x), yfx(x) - 0.012, Zx(x) + 0.012) for x in xs_e], 0.016, sides=5)
        smooth(bead, 60)
        SH.append(bead)
        roll_ys = []
        y = 0.3
        eb_sp = max(0.42, (Htop / t) / 7.0)
        while y < Htop / t - 0.2:                 # batten rolls over the curve, in the direction of fall
            xl = Xof(max(y * t, zend)) - 0.03
            if xl > 0.15:
                pts = [(float(x), y, Zx(x) + 0.016) for x in np.linspace(-xl, xl, 11)]
                rl = sweep("eb_roll", pts, 0.018, sides=5, smooth_path=False)
                smooth(rl, 60)
                SH.append(rl)
                roll_ys.append(y)
            y += eb_sp
        # curved fascia following the whole eyebrow, soffit board behind it
        xs_f = [float(x) for x in np.linspace(-Xt + 0.008, Xt - 0.008, 29)]
        ra = [(x, yfx(x) - 0.022, Zx(x) + 0.004) for x in xs_f]
        rb = [(x, yfx(x) - 0.022, min(max(Zx(x) - 0.105, yfx(x) * t + 0.03), Zx(x))) for x in xs_f]
        fa = ribbon_solid("arch_fascia", ra, rb, (0, 0.022, 0))
        uv_grain(fa, axis="X")
        PT.append(fa)
        xs_s = [float(x) for x in np.linspace(-Xt + 0.02, Xt - 0.02, 15)]
        vs = []
        for x in xs_s:
            vs += [(x, yfx(x), Zx(x) - 0.03), (x, 0.016, Zx(x) - 0.03)]
        so = mesh_object("eb_soffit", vs, [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(xs_s) - 1)])
        uv_grain(so, axis="X")
        T.append(so)
        J = [(float(x), Zx(x) / t) for x in np.linspace(-Xt, Xt, 19)]
        ytop = Htop / t
        zc0, tc = hw - 0.03, 0.0
        head_len = w + 0.05

        def zdorm(x, y):
            return Zx(x) if abs(x) <= Xt and y > yfe - 0.05 else -1e9

        def excl(y):
            if y < 0 or y > ytop:
                return 0.0
            return Xof(max(y * t, zend)) + 0.03

    # =============================================================== cheeks
    ye = zc0 / (t - tc)                           # where the cheek top dies into the slope
    ztop = lambda y: zc0 + y * tc
    inset = {"plaster": 0.016, "lead": 0.004, "tile": 0.026}[p.cladding]
    r_ch = rng.fork("cheek")
    for sx in ((1, -1) if cheeks else ()):
        xo, xi = sx * (hw2 - inset), sx * (hw2 - 0.075)
        poly = [(0.012, 0.012 * t - 0.04), (ye, ye * t - 0.04), (ye, ye * t), (0.012, ztop(0.012))]
        ck = prism("cheek", [(xi, y, z) for y, z in poly], (xo - xi, 0, 0))
        if p.cladding == "plaster":
            densify(ck, max(0.2, ye / 7.0))
            PL.append(ck)
            # top plate, raking sole and a stud framing the plaster
            yb = (zc0 - 0.085) / (t - tc)
            if yb > post + 0.05:
                tp = prism("cheek_plate", [(xi, post - 0.01, ztop(post)), (xi, ye, ye * t), (xi, yb, yb * t), (xi, post - 0.01, ztop(post) - 0.085)],
                           (sx * hw2 - xi, 0, 0), bev=0.004)
                uv_grain(tp, axis=(0, 1, tc))
                T.append(tp)
            bh = 0.115 / ct
            ys = (zc0 - 0.085 - bh) / (t - tc)
            if ys > post + 0.08:
                so = prism("cheek_sole", [(xi, post - 0.01, post * t - 0.03), (xi, ys, ys * t - 0.03), (xi, ys, ys * t + bh),
                                          (xi, post - 0.01, post * t + bh)], (sx * hw2 - xi, 0, 0), bev=0.004)
                uv_grain(so, axis=(0, ct, st))
                T.append(so)
                if ys - post > 0.7:
                    ym = post + (ys - post) * 0.5 + r_ch.u(-0.04, 0.04)
                    z0s, z1s = ym * t + bh * 0.5, ztop(ym) - 0.04
                    tb("cheek_stud", (0.059, 0.08, z1s - z0s), (sx * (hw2 - 0.0295), ym, (z0s + z1s) / 2), "Z")
        elif p.cladding == "lead":
            densify(ck, max(0.12, ye / 9.0))
            LD.append(ck)
            n = max(1, int(round(ye / 0.42)))
            for k in range(1, n + 1):
                yk = post + (ye - post) * (k - 0.5) / n
                z0s, z1s = yk * t + 0.03, ztop(yk)
                if z1s - z0s > 0.1:
                    rl = sweep("cheek_roll", [(xo + sx * 0.006, yk, z0s), (xo + sx * 0.006, yk, z1s)], 0.016, sides=6, smooth_path=False)
                    LD.append(rl)
        else:
            T.append(ck)
            uv_grain(ck, axis="Y")

            def cspan(s, sx=sx):
                yl = post + 0.004
                if tc > 0 and s > zc0:
                    yl = max(yl, (s - zc0) / tc)
                yh_ = max((s - 0.02) / t, yl)
                return [(yl, yh_)] if sx > 0 else [(-yh_, -yl)]

            ztile = ye * t if tc > 0 else zc0
            tf = tile_field("cheek_tiles", Vector((xo, 0, 0)), Vector((0, sx, 0)), Vector((0, 0, 1)), post * t * 0.6, ztile - 0.004,
                            cspan, TW, GA, TT, r_ch, base=0.005, slip=age * 0.5, chip=0.12 if slate else 0.0, max_tiles=70)
            if tf:
                TL.append(tf)

    # =============================================================== leadwork (2.5 mm sheet, dressed over the roof)
    r_ld = rng.fork("lead")
    LF = 0.026                                    # dressed over the tiles of the main roof
    if cheeks:
        # stepped side flashings: lapped pieces up the cheek, turned up against it with a level top edge
        upn = 0.075 if p.cladding != "tile" else 0.035
        yc2 = max(0.12, (zc0 - 0.012 - upn / ct) / (t - tc))
        nstep = max(2, int(round((yc2 + 0.02) / max(0.2, (yc2 + 0.02) / 12.0))))
        dy = (yc2 + 0.02) / nstep
        for sx in (1, -1):
            xu = sx * (hw2 + 0.004) if p.cladding == "plaster" else sx * (hw2 - inset + 0.003)
            for k in range(nstep):
                ya = -0.02 + k * dy - (0.035 if k else 0.0)
                yb = -0.02 + (k + 1) * dy
                xw = sx * (hw2 + 0.115 + r_ld.u(-0.006, 0.006))
                lf = LF + (0.004 if k % 2 else 0.0)
                rows = [[PM(xw, ya, lf + 0.004), PM(xw, yb, lf - 0.004)],
                        [Vector((xu, ya, ya * t + (lf + 0.004) / ct)), Vector((xu, yb, yb * t + lf / ct))]]
                ztp = min(yb * t + upn, ztop(yb) - 0.006)
                if ztp - (ya * t + lf / ct) > 0.02:
                    rows.append([Vector((xu, ya, ztp)), Vector((xu, yb, ztp))])
                LD.append(sheet_mesh("flash_step", rows, nm))
    # valley / top flashing round the junction with the main roof
    if kind in ("gable", "hip"):
        rows = [[], [], []]
        for (x, y) in J:
            sg = (x > 1e-6) - (x < -1e-6)
            k = abs(x) / xe
            rows[0].append(Vector((x, y - VG - 0.02, za - abs(x) * trp + 0.0008)))
            rows[1].append(PM(x, y, 0.003))
            rows[2].append(PM(x + sg * 0.085, y + 0.06 + 0.03 * (1 - k), 0.006))
        o = sheet_mesh("flash_valley", rows if not sheet else rows[1:], nm)
        (LD if not sheet or cover == "lead" else SH).append(o)
        for sx in (1, -1):                         # tail of the valley down to the top of the side flashing
            rows = [[PM(sx * (hw2 - 0.01), yc2 - 0.03, 0.005), PM(sx * xe, yedge, 0.005)],
                    [PM(sx * (hw2 + 0.115), yc2 - 0.03, 0.007), PM(sx * (xe + 0.085), yedge + 0.06, 0.007)]]
            LD.append(sheet_mesh("flash_tail", rows, nm))
    elif kind == "shed":
        xs_j = [float(x) for x in np.linspace(-(xe + 0.03), xe + 0.03, max(4, int(round(2 * xe / 0.165)) + 1))]
        rows = [[], [], []]
        for i, x in enumerate(xs_j):
            wv = 0.007 * (1 if i % 2 else -1) + r_ld.u(-0.004, 0.004)
            yi = yj - 0.11 + wv
            rows[0].append(Vector((x, yi, hw + yi * tsp + (0.007 if sheet else TT + 0.006))))
            rows[1].append(PM(x, yj - 0.005, 0.02))
            rows[2].append(PM(x, yj + 0.09, 0.006))
        LD.append(sheet_mesh("flash_top", rows, nm))
    else:
        rows = [[], []]
        for i, (x, y) in enumerate(J):
            x0_, y0_ = J[max(i - 1, 0)]
            x1_, y1_ = J[min(i + 1, len(J) - 1)]
            L_ = hypot(x1_ - x0_, y1_ - y0_)
            nx, ny = -(y1_ - y0_) / L_, (x1_ - x0_) / L_
            rows[0].append(PM(x - nx * 0.012, y - ny * 0.012, 0.011))
            rows[1].append(PM(x + nx * 0.09, y + ny * 0.09, 0.006))
        SH.append(sheet_mesh("flash_junction", rows, nm))
    # apron dressed down over the roof in front of the sill, bossed to the tile profile, held by clips
    ax_in = hw2 + 0.03 if cheeks else Xof(0.16) + 0.02
    ax_out = hw2 + 0.135 if cheeks else Xt + 0.06
    twm = 0.19
    ncol = max(6, min(int(round(2 * ax_in / 0.062)), 36) + 1)
    xs_a = [float(x) for x in np.linspace(-ax_in, ax_in, ncol)]
    ph = r_ld.u(0, 6.28)
    yf0 = -0.0325
    rows = [[], [], [], []]
    low_edge = []
    for x in xs_a:
        wv = sin(2 * pi * x / twm + ph)
        d_e = 0.15 + 0.008 * wv + r_ld.u(-0.002, 0.002)
        rows[0].append(Vector((x, yf0, 0.068 + r_ld.u(-0.002, 0.002))))
        rows[1].append(Vector((x, yf0 - 0.002, yf0 * t + LF / ct)))
        rows[2].append(PM(x, yf0 - 0.075 * ct, LF + 0.004 * wv))
        rows[3].append(PM(x, yf0 - d_e * ct, LF - 0.009 + 0.004 * wv))
        low_edge.append((x, yf0 - d_e * ct))
    LD.append(sheet_mesh("flash_apron", rows, nm))
    for sx in (1, -1):                             # ears of the apron running under the side flashings
        xa_, xb_ = sx * ax_in, sx * ax_out
        rows = [[PM(xa_, 0.06, LF - 0.004), PM(xb_, 0.06, LF - 0.004)], [PM(xa_, yf0, LF - 0.002), PM(xb_, yf0, LF - 0.002)],
                [PM(xa_, yf0 - 0.15 * ct, LF - 0.009), PM(xb_, yf0 - 0.145 * ct, LF - 0.009)]]
        LD.append(sheet_mesh("flash_ear", rows, nm))
    nclip = max(2, int(round(2 * ax_in / 0.42)))
    for i in range(nclip):
        x, yl = low_edge[int((i + 0.5) / nclip * (len(low_edge) - 1))]
        c = PM(x, yl + 0.014 * ct, LF + 0.001)
        LD.append(box("lead_clip", (0.03, 0.05, 0.005), tuple(c), rot=(degrees(th), 0, 0), bevel=0.0))

    # sarking boards closing the dormer where it meets the roof, framed by trimmers (hidden once on a building)
    if cheeks:
        a_ = hw2 - 0.02
        upts = [(-a_, 0.02), (a_, 0.02), (a_, ye - 0.05)] + [(x, y - 0.07) for (x, y) in reversed(J) if abs(x) < a_ - 0.01] + [(-a_, ye - 0.05)]
    else:
        a_ = Xt - 0.05
        upts = [(-a_, 0.02), (a_, 0.02)] + [(x, y - 0.05) for (x, y) in reversed(J) if abs(x) < a_ - 0.01 and y - 0.05 > 0.05]
    un = prism("sarking", [PM(x, y, -0.022) for x, y in upts], nm * 0.012)
    uv_grain(un, axis="X")
    T.append(un)
    cx_ = sum(q[0] for q in upts) / len(upts)
    cy_ = sum(q[1] for q in upts) / len(upts)
    for i in range(len(upts) if len(upts) <= 7 else 0):   # trimmer frame round the opening in the roof
        (x0_, y0_), (x1_, y1_) = upts[i], upts[(i + 1) % len(upts)]
        ex, ey = x1_ - x0_, (y1_ - y0_) / ct
        L_ = hypot(ex, ey)
        if L_ < 0.05:
            continue
        nx, ny = -ey / L_, ex / L_                # in-plane normal (slope metric)
        if nx * (cx_ - x0_) + ny * (cy_ - y0_) / ct < 0:
            nx, ny = -nx, -ny
        wq = 0.075
        fr = prism("trimmer", [PM(x0_, y0_, -0.03), PM(x1_, y1_, -0.03), PM(x1_ + nx * wq, y1_ + ny * wq * ct, -0.03),
                               PM(x0_ + nx * wq, y0_ + ny * wq * ct, -0.03)], -nm * 0.045)
        uv_grain(fr, axis=(ex, ey * ct, ey * st))
        T.append(fr)

    # =============================================================== timber front
    zh0 = hw - hbh
    sill_len = w + 0.06 if cheeks else 2 * Xof(0.16)
    tb("sill_beam", (sill_len, 0.135, sbh), (0, 0.0375, sbh / 2), "X", 0.007)
    tb("head_beam", (head_len, 0.125, hbh), (0, head_y, hw - hbh / 2), "X", 0.007)
    for sx in (1, -1):
        tb("post", (post, post, zh0 - sbh + 0.01), (sx * (hw2 - post / 2), post / 2, (sbh + zh0) / 2), "Z", 0.006)
    # oak pegs at the joints
    pegs = []
    for sx in (1, -1):
        for zz in (sbh * 0.5, hw - hbh * 0.5):
            for dx in (-0.02, 0.022):
                pegs.append(trs((sx * (hw2 - post / 2) + dx, -0.03 if zz < 0.1 else head_y - 0.0625, zz + dx * 0.4), (90, 0, 0)))
    pg = cylinder("peg", 0.008, 0.008, verts=5)
    o = instances_to_mesh("pegs", pg, pegs, keep_base=False)
    T.append(o)

    cw = w - 2 * post
    z0, z1 = sbh, zh0
    avail = z1 - z0
    n_leaf = max(1, int(round(cw / 0.5)))
    if n_leaf > 2 and n_leaf % 2:
        n_leaf += 1
    groups = 1 if n_leaf <= 2 else n_leaf // 2
    per = n_leaf // groups
    mull = 0.07
    fw = 0.048
    gw = (cw - (groups - 1) * mull) / groups
    lw = (gw - 2 * fw) / per
    wh = min(avail, max(0.6, 1.9 * lw + 0.12))
    if avail - wh < 0.14:
        wh = avail
    zw0 = z1 - wh
    r_w = rng.fork("window")
    boards_infill = p.gable_infill == "boards"

    def infill(name, pts_xz, y0, y1):
        """lime-plaster panel or painted vertical boards filling a region of the front (pts in x, z)"""
        if not boards_infill:
            o = prism(name, [(x, y0 + 0.012, z) for x, z in pts_xz], (0, y1 - y0 - 0.012, 0))
            ex = [max(c[k] for c in pts_xz) - min(c[k] for c in pts_xz) for k in (0, 1)]
            densify(o, max(0.08, (ex[0] * ex[1]) ** 0.5 / 7.0))
            paint(o, grime=edge_dirt(pts_xz, 0.035, 0.3 + 0.3 * age))
            PL.append(o)
            return
        xs = [q[0] for q in pts_xz]
        xa, xb = min(xs), max(xs)
        nbd = max(1, int(round((xb - xa) / max(0.115, (xb - xa) / 16.0))))
        bw_ = (xb - xa) / nbd
        for i in range(nbd):                       # every board is cut to the outline of the panel
            q = clip_x(pts_xz, xa + i * bw_ + 0.0015, xa + (i + 1) * bw_ - 0.0015)
            if len(q) < 3 or max(c[1] for c in q) - min(c[1] for c in q) < 0.02:
                continue
            dy = r_w.u(0.0, 0.004)
            o = prism(name, [(x, y0 + dy, z) for x, z in q], (0, 0.02, 0))
            uv_grain(o, axis="Z")
            PT.append(o)

    if zw0 - z0 > 0.02:                           # panel under the window
        rh = min(zw0 - z0, 0.08)
        tb("mid_rail", (cw + 0.01, 0.1, rh), (0, 0.045, zw0 - rh / 2), "X", 0.005)
        if zw0 - rh - z0 > 0.03:
            infill("panel", [(-cw / 2, z0), (cw / 2, z0), (cw / 2, zw0 - rh), (-cw / 2, zw0 - rh)], 0.012, 0.08)

    RS = 0.35                                     # room UV scale: the whole window looks into one room
    rc = (0.0, zw0 + wh / 2)

    def room_quad(name, x0_, x1_, za_, zb_, y, mt, both=False):
        """glazing showing the room: room-scale UVs shared by the whole window, so all leaves look into one interior"""
        V = [(x0_, y, za_), (x1_, y, za_), (x1_, y, zb_), (x0_, y, zb_)]
        F = [(0, 1, 2, 3)] + ([(3, 2, 1, 0)] if both else [])
        if p.lit:
            o = mesh_uv(name, V, F, [(0.5 + RS * (q[0] - rc[0]), 0.48 + RS * (q[2] - rc[1])) for q in V])
        else:
            o = mesh_object(name, V, F)
        assign(o, mt)
        return o

    ajar_leaf = r_w.pick(list(range(n_leaf))) if p.ajar > 0.01 else -1
    li = 0
    for gi in range(groups):
        gx0 = -cw / 2 + gi * (gw + mull)
        gx1 = gx0 + gw
        gxc = (gx0 + gx1) / 2
        if gi > 0:
            tb("mullion", (mull, post, wh), (gx0 - mull / 2, post / 2 - 0.003, zw0 + wh / 2), "Z", 0.005)
        for xx in (gx0 + fw / 2, gx1 - fw / 2):
            tb("win_jamb", (fw, 0.08, wh), (xx, 0.035, zw0 + wh / 2), "Z", 0.003, PT)
        tb("win_head", (gw - 2 * fw + 0.002, 0.08, fw), (gxc, 0.035, z1 - fw / 2), "X", 0.003, PT)
        tb("win_sill", (gw + 0.03, 0.118, 0.045), (gxc, 0.016, zw0 + 0.0225), "X", 0.005, PT)
        lz0, lz1 = zw0 + 0.048, z1 - fw - 0.003
        ox0, ox1 = gx0 + fw, gx1 - fw
        WIN.append(room_quad("win_room", ox0, ox1, lz0 - 0.004, lz1 + 0.004, 0.068, m_room))
        # curtains hanging in the room: gathered to tie-backs at the sides, short valance across the head
        if m_curt is not None and p.curtain > 0.03:
            cv, cf, cuv = [], [], []
            hgt = lz1 - lz0
            wc = clamp(float(p.curtain)) * (ox1 - ox0) / 2
            open_ = 1.0 - clamp(float(p.curtain))
            prof = [(0.0, 1.0), (0.22, 0.95), (0.45, 0.7), (0.6, 0.5), (0.72, 0.56), (1.0, 0.68)]
            for sd, xo_ in ((1, ox0), (-1, ox1)):
                u0 = r_w.u(0.1, 0.5)
                i0 = len(cv)
                for (f, k) in prof:
                    k = 1.0 - (1.0 - k) * (0.35 + 0.65 * open_)
                    z = lz1 - f * hgt
                    for c in (0.0, 0.5, 1.0):
                        cv.append((xo_ + sd * wc * k * c, 0.056 - 0.006 * sin(pi * c), z))
                        cuv.append((u0 + c * wc * 1.3, 0.3 + 0.03 * f))
                for r_ in range(len(prof) - 1):
                    for c in range(2):
                        a = i0 + r_ * 3 + c
                        cf.append((a, a + 3, a + 4, a + 1) if sd > 0 else (a, a + 1, a + 4, a + 3))
            i0 = len(cv)
            nvl = max(2, int(round((ox1 - ox0) / 0.09)))
            for i in range(nvl + 1):
                x = ox0 + (ox1 - ox0) * i / nvl
                # (the valance hangs 4.5 mm in front of the nearest curtain fold: glowing sheets keep >= 4 mm apart)
                cv += [(x, 0.0455, lz1 + 0.004), (x, 0.0455, lz1 - 0.065 - (0.02 if i % 2 else 0.0))]
                cuv += [(0.2 + 1.3 * (x - ox0), 0.34), (0.2 + 1.3 * (x - ox0), 0.3)]
            for i in range(nvl):
                a = i0 + 2 * i
                cf.append((a, a + 1, a + 3, a + 2))
            cu = mesh_uv("curtains", cv, cf, cuv)
            assign(cu, m_curt)
            WIN.append(cu)
        for i in range(per):
            xa = gx0 + fw + i * lw + 0.003
            xb = xa + lw - 0.006
            hinge_left = (i == 0)
            stl, thk = 0.04, 0.032
            lh = lz1 - lz0
            parts = []

            def lb(name, size, loc, grain, m=m_paint, bev=0.0025):
                o = box(name, size, loc, bevel=bev, segments=1)
                uv_grain(o, axis=grain)
                assign(o, m)
                parts.append(o)
                return o

            yc = 0.002 + thk / 2
            lb("stile", (stl, thk, lh), (xa + stl / 2, yc, lz0 + lh / 2), "Z")
            lb("stile", (stl, thk, lh), (xb - stl / 2, yc, lz0 + lh / 2), "Z")
            lb("rail", (xb - xa - 2 * stl, thk, stl), ((xa + xb) / 2, yc, lz1 - stl / 2), "X", bev=0.0)
            lb("rail", (xb - xa - 2 * stl, thk, stl * 1.35), ((xa + xb) / 2, yc, lz0 + stl * 0.675), "X", bev=0.0)
            ix0, ix1 = xa + stl, xb - stl
            iz0, iz1 = lz0 + stl * 1.35, lz1 - stl
            lb("gbar", (0.016, 0.022, iz1 - iz0), ((ix0 + ix1) / 2, yc, (iz0 + iz1) / 2), "Z", bev=0.0)
            nbar = max(1, int(round((iz1 - iz0) / 0.23)) - 1)
            for k in range(nbar):
                zz = iz0 + (iz1 - iz0) * (k + 1) / (nbar + 1)
                lb("gbar", (ix1 - ix0, 0.022, 0.016), ((ix0 + ix1) / 2, yc, zz), "X", bev=0.0)
            if li == ajar_leaf and not p.lit:      # dark room: the open leaf carries its own black pane; lit: the room shows through
                parts.append(room_quad("glass", ix0 - 0.004, ix1 + 0.004, iz0 - 0.004, iz1 + 0.004, yc + 0.004, m_room, both=True))
            hx = xa if hinge_left else xb
            sgn = 1 if hinge_left else -1
            for zz in (lz0 + 0.13, lz1 - 0.13):   # painted-over butt hinges
                o = cylinder("hinge_pin", 0.008, 0.06, loc=(hx - sgn * 0.004, -0.004, zz), verts=6)
                assign(o, m_paint)
                parts.append(o)
                o = box("hinge_strap", (0.07, 0.005, 0.022), (hx + sgn * 0.035, -0.0005, zz))
                assign(o, m_paint)
                parts.append(o)
            fx_ = xb - stl / 2 if hinge_left else xa + stl / 2      # fastener on the free stile
            if per == 1 or i == 0:
                o = box("latch", (0.014, 0.012, 0.085), (fx_, -0.004, lz0 + lh * 0.45), bevel=0.002, segments=1)
                assign(o, m_paint)
                parts.append(o)
                o = sphere("latch_knob", 0.011, loc=(fx_, -0.017, lz0 + lh * 0.45 - 0.03), segments=6, rings=4)
                assign(o, m_paint)
                parts.append(o)
            leaf = join(parts, "leaf")
            if li == ajar_leaf:
                ang = radians(p.ajar * 38.0) * (-1 if hinge_left else 1)
                ca, sa = cos(ang), sin(ang)
                piv = Vector((hx, 0.0, 0.0))

                def rotz(co, piv=piv, ca=ca, sa=sa):
                    d = co - piv
                    return Vector((piv.x + d.x * ca - d.y * sa, piv.y + d.x * sa + d.y * ca, co.z))

                transform_verts(leaf, rotz)
            WIN.append(leaf)
            li += 1

    # =============================================================== roof-form specific front
    if kind == "gable":
        rd = 0.09 / cr
        zt0, zt1 = za - DK / cr, hw - DK / cr
        for sx in (1, -1):
            rf = prism("gable_rafter", [(0, -0.006, zt0), (sx * hw2, -0.006, zt1), (sx * hw2, -0.006, zt1 - rd), (0, -0.006, zt0 - rd)],
                       (0, 0.096, 0), bev=0.005)
            uv_grain(rf, axis=(sx * cr, 0, -sr))
            T.append(rf)
        zin = zt0 - rd                               # underside of the rafters at the apex
        if zin - hw > 0.2:
            tb("king_post", (0.085, 0.092, zin - hw + 0.03), (0, 0.04, (zin + hw) / 2 + 0.015), "Z", 0.005)
            for sx in (1, -1):
                xo_ = sx * (hw2 - 0.02)
                infill("gable_fill", [(sx * 0.04, hw), (xo_, hw), (xo_, zt1 - rd + 0.02 * trp + 0.01), (sx * 0.04, zin - 0.04 * trp + 0.01)], 0.012, 0.08)
        else:
            infill("gable_fill", [(-(hw2 - 0.02), hw), (hw2 - 0.02, hw), (0, zin + 0.01)], 0.012, 0.08)
        # plate and ridge ends carrying the verge
        for sx in (1, -1):
            tb("plate_end", (0.085, -yfe + 0.02, 0.085), (sx * (hw2 - 0.0425), yfe / 2 + 0.01, hw - 0.05 - DK / cr), "Y", 0.005)
        tb("ridge_end", (0.07, -yfe + 0.02, 0.1), (0, yfe / 2 + 0.01, za - DK / cr - 0.07 - 0.035 * trp), "Y", 0.005)
        # bargeboards
        bw_ = 0.135
        dv = bw_ / cr
        xb_ = xe + 0.03
        Lb = xb_ / cr
        for sx in (1, -1):
            e = Vector((sx * cr, 0, -sr))
            dp = Vector((-sx * sr, 0, -cr))
            y_b = yfe - 0.024
            topA = Vector((0, y_b, za + 0.004))
            botA = Vector((0, y_b, za + 0.004 - dv))
            pts = [topA, topA + e * Lb]
            if p.barge == "scalloped":
                n = max(2, int(round(Lb / 0.13)))
                for i in range(n, 0, -1):
                    for j in range(5):
                        u = Lb * (i - j / 5.0) / n
                        pts.append(botA + e * u + dp * (0.032 * sin(pi * j / 5.0)))
            else:
                pts += [botA + e * Lb + Vector((0, 0, 0.045)), botA + e * (Lb - 0.05)]      # nipped tail
            pts.append(botA)
            bg = prism("bargeboard", pts, (0, 0.024, 0), bev=0.003 if p.barge == "plain" else 0.0)
            uv_grain(bg, axis=e)
            PT.append(bg)
        fin = lathe("finial", [(0.0, -0.27), (0.016, -0.255), (0.027, -0.23), (0.017, -0.2), (0.022, -0.185), (0.022, 0.09), (0.034, 0.105),
                               (0.034, 0.12), (0.02, 0.14), (0.03, 0.18), (0.018, 0.225), (0.007, 0.3), (0.0, 0.31)], segments=8,
                    loc=(0, yfe - 0.014, za), smooth_angle=35)
        PT.append(fin)
    elif kind == "eyebrow":
        a1 = hw2
        pts = [(-a1, hw), (a1, hw)] + [(float(x), Zx(x) - 0.036) for x in np.linspace(a1, -a1, 13)]
        infill("tympanum", pts, 0.012, 0.08)
        xsb = Xof(0.16) - 0.012
        if xsb - hw2 > 0.06:                         # spandrels under the tails, standing on the long sill beam
            for sx in (1, -1):
                pts = [(sx * hw2, sbh), (sx * xsb, sbh)] + [(sx * float(x), Zx(x) - 0.036) for x in np.linspace(xsb, hw2, 8)]
                infill("spandrel", pts, 0.012, 0.08)

    if kind != "eyebrow":
        if kind == "gable":
            zb_k, W_k = hw - 0.05 - DK / cr - 0.0425, -yfe - 0.03
        elif kind == "hip":
            zb_k, W_k = zedge - DK / cr - 0.012, -yfe - 0.03
        else:
            zb_k, W_k = zfe - DK - 0.012, -yfe - 0.02
        Hb_k = min(0.2, max(0.12, (zb_k - zw0) * 0.3))
        xs_k = [-(hw2 - post / 2), hw2 - post / 2] + [-cw / 2 + gi * (gw + mull) - mull / 2 for gi in range(1, groups)]
        for xk in xs_k:
            pts = [(0.0, zb_k), (-W_k, zb_k), (-W_k, zb_k - 0.028)]
            for a in np.linspace(pi / 2, 0, 6)[1:]:
                pts.append((-W_k + (W_k - 0.022) * cos(a), zb_k - Hb_k + (Hb_k - 0.028) * sin(a)))
            pts.append((0.0, zb_k - Hb_k))
            br = prism("knee_bracket", [(xk - 0.026, y, z) for y, z in pts], (0.052, 0, 0), bev=0.004)
            uv_grain(br, axis=(0, 1, -1))
            T.append(br)

    # =============================================================== window box
    if p.flower_box:
        r_f = rng.fork("flowers")
        bl = min(cw + 0.08, w - 0.06)
        bd, bh, by0, zb = 0.16, 0.125, -0.046, 0.014
        BX = []
        tb("box_front", (bl, 0.018, bh), (0, by0 - bd + 0.009, zb + bh / 2), "X", 0.003, BX)
        tb("box_back", (bl, 0.018, bh), (0, by0 - 0.009, zb + bh / 2), "X", 0.0, BX)
        for sx in (1, -1):
            tb("box_end", (0.018, bd - 0.036, bh), (sx * (bl / 2 - 0.009), by0 - bd / 2, zb + bh / 2), "Y", 0.003, BX)
        tb("box_bottom", (bl - 0.036, bd - 0.036, 0.016), (0, by0 - bd / 2, zb + 0.012), "X", 0.0, BX)
        tb("box_trim", (bl + 0.02, 0.012, 0.03), (0, by0 - bd - 0.004, zb + bh - 0.02), "X", 0.003, BX)
        for sx in (1, -1):                         # painted iron brackets and stays
            xb2 = sx * bl * 0.33
            BX.append(box("box_bracket", (0.024, bd + 0.02, 0.008), (xb2, by0 - bd / 2 + 0.01, zb - 0.004)))
            BX.append(sweep("box_stay", [(xb2, by0 - bd + 0.01, zb - 0.006), (xb2, by0 - bd * 0.45, zb - 0.05), (xb2, by0 + 0.014, zb - 0.012 - 0.09)],
                            0.006, sides=4))
        PT.extend(BX)
        m_leafg = mat("leaf", "#3f6a2e")
        m_flower = mat("leaf", p.flowers)
        # one folded leaf blade (2 triangles) mapped onto the whole leaf tile
        lv = [(0.0, 0.0, 0.0), (0.5, -0.42, 0.16), (1.0, 0.0, 0.1), (0.5, 0.42, 0.16)]
        leaf = mesh_uv("leaf", lv, [(0, 1, 2), (0, 2, 3)], [(c[0] * 0.25, (c[1] * 0.6 + 0.5) * 0.25) for c in lv])
        assign(leaf, m_leafg)
        nblob = max(3, int(round(bl / 0.17)))
        blob = sphere("blob", 1.0, segments=6, rings=3)
        cores, leaves, fm = [], [], []
        zax = Vector((0, 0, 1))

        def leaf_m(pos, d, s_):
            q = zax.rotation_difference(d.normalized())
            return Matrix.Translation(pos) @ q.to_matrix().to_4x4() @ Matrix.Rotation(r_f.u(0, 2 * pi), 4, "Z") @ Matrix.Diagonal((s_, s_, s_, 1.0))

        for i in range(nblob):
            x = -bl / 2 + bl * (i + 0.5) / nblob + r_f.u(-0.02, 0.02)
            rr_ = r_f.u(0.065, 0.095) * (1.25 if i % 3 == 1 else 1.0)
            c = Vector((x, by0 - bd / 2 + r_f.u(-0.03, 0.0), zb + bh + rr_ * r_f.u(0.2, 0.45)))
            ax = Vector((rr_ * 1.3, rr_ * 0.95, rr_ * r_f.u(0.85, 1.15)))
            cores.append(trs(c, (r_f.u(-15, 15), r_f.u(-15, 15), r_f.u(0, 180)), tuple(ax * 0.85)))
            for _ in range(14):
                a, e_ = r_f.u(0, 2 * pi), r_f.u(0.0, 1.45)
                d = Vector((cos(a) * cos(e_), -abs(sin(a) * cos(e_)) if r_f.chance(0.8) else sin(a) * cos(e_), sin(e_)))
                pos = c + Vector((d.x * ax.x, d.y * ax.y, d.z * ax.z)) * r_f.u(0.9, 1.08)
                leaves.append(leaf_m(pos, d + Vector((0, -0.25, 0.35)), r_f.u(0.036, 0.056)))
            for _ in range(3):                      # flower heads standing just clear of the leaves
                a, e_ = r_f.u(pi, 2 * pi), r_f.u(0.5, 1.4)
                d = Vector((cos(a) * cos(e_), sin(a) * cos(e_), sin(e_)))
                pos = c + Vector((d.x * ax.x, d.y * ax.y, d.z * ax.z)) * 1.22
                s_ = r_f.u(0.02, 0.03)
                fm.append(trs(pos, (r_f.u(-30, 30), r_f.u(-30, 30), 0), (s_, s_, s_ * 0.7)))
            if r_f.chance(0.65):                    # trailing growth over the front edge
                x2 = x + r_f.u(-0.05, 0.05)
                ln = r_f.u(0.08, 0.2)
                for k in range(int(ln / 0.04)):
                    pos = Vector((x2 + r_f.u(-0.02, 0.02), by0 - bd - 0.012 - r_f.u(0, 0.012), zb + bh - 0.01 - k * 0.04))
                    leaves.append(leaf_m(pos, Vector((r_f.u(-0.4, 0.4), -1, r_f.u(-0.1, 0.5))), r_f.u(0.034, 0.048)))
        fo = instances_to_mesh("foliage_core", blob, cores, keep_base=False)
        smooth(fo, 80)
        assign(fo, m_leafg)
        paint(fo, grime=0.75)
        lf = instances_to_mesh("foliage_leaves", leaf, leaves, uv_jitter=False, keep_base=False)
        smooth(lf, 80)
        fl_ = sphere("flower", 1.0, segments=4, rings=2)
        fo2 = instances_to_mesh("flowers", fl_, fm, keep_base=False)
        assign(fo2, m_flower)
    else:
        socket("planter", "attach", (0, -0.045, zw0 + 0.02), role="planter")

    # =============================================================== presentation roof patch
    if p.roof_patch > 0.01:
        m_ = float(p.roof_patch)
        Xp = (xe if cheeks else Xt) + m_
        ylo, yhi = -0.7 * m_, ytop + m_
        sl = prism("patch_deck", [PM(-Xp, ylo, -0.05), PM(Xp, ylo, -0.05), PM(Xp, yhi, -0.05), PM(-Xp, yhi, -0.05)], nm * 0.048)
        uv_grain(sl, axis="X")
        T.append(sl)

        def pspan(s):
            X = excl(s * ct)
            return [(-Xp, -X), (X, Xp)] if X > 0.0 else [(-Xp, Xp)]

        psc = 1.0 if slate else 1.15
        tf = tile_field("patch_tiles", Vector((0, 0, 0)), Vector((1, 0, 0)), dsl, ylo / ct, yhi / ct, pspan,
                        TW * psc, GA * psc, TT, rng.fork("patch"), base=0.008, slip=age, chip=0.16 if slate else 0.0,
                        gap=0.002 if slate else 0.0015, max_tiles=280)
        if tf:
            TL.append(tf)

    # =============================================================== materials + weathering
    def above_roof(d0, d1):                       # 1 close to the main roof plane, fading out by d1 (m above it)
        return lambda co, n: np.clip(1.0 - (co[:, 2] - co[:, 1] * t - d0) / max(d1 - d0, 1e-3), 0.0, 1.0)

    def streak(co3, sd, lo=0.35, hi=0.7):          # noise stretched along the fall: run-off streaks
        return noise_mask(1.0, seed=sd, lo=lo, hi=hi)(co3, None)

    for o in T:
        assign(o, m_timber)
        paint(o, grime=mask_mul(above_roof(0.0, 0.3), 0.6))
    fl_n = noise_mask(0.06, seed=seed + 2, lo=0.4, hi=0.8)

    def flake(co, n):                             # paint flakes on weathered upper faces, lower edges and near the roof
        e = 0.12 + 0.6 * np.clip(-n[:, 2], 0, 1) + 0.45 * np.clip(n[:, 2], 0, 1) + 0.3 * above_roof(0.0, 0.25)(co, n)
        return np.clip(fl_n(co, n) * e * (0.15 + 0.85 * age), 0.0, 1.0)

    for o in PT:
        assign(o, m_paint)
        paint(o, wear=flake, special=mask_mul(facing((0, -0.5, 1)), 0.3 * age), grime=mask_mul(above_roof(0.0, 0.2), 0.45))
    for o in LD:
        assign(o, m_lead)
        paint(o, grime=0.12, wear=0.28, mode="set")       # dull the sheet: lead is not tintable, a little wear keeps it blue-grey
    for o in PL:
        assign(o, m_plaster)
        paint(o, special=mask_mul(above_roof(0.0, 0.35), 0.25 + 0.4 * age), grime=mask_mul(above_roof(0.02, 0.3), 0.55))
    for o in SH:
        assign(o, m_sheet)
    if cover == "copper":                          # verdigris follows the water: streaks down the fall, pooled at the eaves
        g_age = 0.55 + 0.45 * age
        if kind == "eyebrow":
            def eb_pat(co, n):
                u = np.clip(np.abs(co[:, 0]) / Xt, 0.0, 1.0)
                pool = np.clip((u - 0.3) / 0.3, 0.0, 1.0)                       # water gathers in the tails: solid verdigris
                s_ = streak(np.stack([co[:, 0] * 0.25 + 3.0 * np.sign(co[:, 0]), co[:, 1] * 6.5, 0 * u], 1), seed + 7, 0.44, 0.56)
                run = s_ * (0.3 + 0.7 * np.clip(u / 0.3, 0.0, 1.0))             # run-off streaks starting below the crown
                return np.clip(np.maximum(pool, run) * (0.45 + 0.55 * g_age) + 0.04, 0.0, 1.0)
            for o in SH:
                paint(o, special=eb_pat, grime=lambda co, n: 0.35 * np.clip(1.0 - np.abs(co[:, 0]) / (0.3 * Xt), 0.0, 1.0), mode="set")
        else:
            for objs, O_, A_, U_, S_ in SHP:
                def pl_pat(co, n, O_=np.array(O_), A_=np.array(A_), U_=np.array(U_), S_=S_):
                    d = co - O_
                    s = d @ U_
                    low = np.clip(1.0 - s / S_, 0.0, 1.0)
                    s_ = streak(np.stack([(d @ A_) * 15.0, s * 0.5, 0 * s], 1), seed + 7)
                    return np.clip((0.85 * low ** 1.6 + (0.25 + 0.75 * low) * 0.65 * s_) * g_age, 0.0, 1.0)
                for o in objs:
                    paint(o, special=pl_pat, mode="set")
    elif sheet:
        for o in SH:
            paint(o, grime=0.06)
    soot = noise_mask(0.7, seed=seed + 9, lo=0.3, hi=0.8)
    for o in RT:
        assign(o, m_tile)
        paint(o, grime=0.3 + 0.25 * age, special=0.15 * age)
    for o, tone in TL:                             # per-tile tone: one material, every tile / slate its own shade
        assign(o, m_tile)

        def tgrime(co, n, tone=tone):
            b = tone if len(tone) == len(co) else 0.3
            if slate:
                return np.clip(b * 0.75 + 0.3 * above_roof(0.02, 0.3)(co, n), 0.0, 1.0)
            return np.clip(b * (0.55 + 0.35 * age) + 0.12 * soot(co, n) + 0.3 * above_roof(0.02, 0.3)(co, n), 0.0, 1.0)

        def tmoss(co, n, tone=tone):
            b = tone if len(tone) == len(co) else 0.3
            m = np.maximum(above_roof(0.02, 0.4)(co, n) * 0.8, np.clip((b - 0.55) * 3.0, 0.0, 1.0) * 0.7)
            return np.clip(m * (0.1 + 0.5 * age) * (0.35 if slate else 1.0), 0.0, 1.0)

        paint(o, grime=tgrime, special=tmoss)

    # sanity check: nothing of the carcass may stand proud of the roof covering
    worst = {}
    skip = ("barge", "finial", "fascia", "arch_fascia", "deck", "sarking", "trimmer", "patch", "box_", "eb_soffit")
    for o in T + PT + PL + WIN:
        if o.name.startswith(skip) or "_deck" in o.name:
            continue
        M = world_matrix(o)
        for v in o.data.vertices:
            c = M @ v.co
            zd = zdorm(c.x, c.y)
            if zd < -1e8 and c.y < 0.0:               # in front of the dormer, outside its roof (open casement, sill)
                continue
            d = c.z - max(zd, c.y * t) - 0.004
            if d > 0 and d > worst.get(o.name, 0.0):
                worst[o.name] = d
    if worst:
        print("DBG PIERCE", {k: round(v, 3) for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:6]})

    # =============================================================== light
    if p.lit and p.intensity > 0.1:
        light("window", (0, -0.6, zw0 + wh * 0.55), color="#ffc37a", intensity=float(p.intensity), range=5.0, flicker=0.05,
              shadow_near=0.45)
    socket("roof_point", "attach", (0, 0, 0), role="roof")
