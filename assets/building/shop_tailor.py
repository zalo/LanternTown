"""The 'Dress Suit' tailor's shop: a rendered town house with a stone plinth, rusticated quoins and a tall
ground-floor shopfront (round-arched display window with a fan-light, a jester mannequin, a tailor's dummy and
bolts of cloth on a lit stage, glazed shop door, ornate fascia sign).  The upper facade and the shaped gable
are faced with individually modelled glazed lozenge tiles in 2-4 pastel tints (harlequin bond) around little
lantern-bay windows with copper hoods and a carved jester shop-figure on a bracket between two wall lanterns.
The scrolled / bell / crow-stepped parapet gable (moulded coping, volutes, urn and ball finials, lit oculus)
fronts a clay-tiled roof with ridge tiles, copper gutters, downpipes and a chimney; the side and back walls
have shuttered windows (some lit), tie anchors, a plank back door and plain mouldings.
"""
from tk import *
import numpy as np
from math import sin, cos, tan, pi, radians, hypot, atan2

META = {"name": "Tailor's shop with harlequin facade", "class": "building", "mount": "ground",
        "focus": [0.5, 0.0, 3.2]}
PARAMS = {
    "width": {"default": 7.0, "min": 6.0, "max": 9.0, "desc": "frontage width (m), walls span x = +-width/2"},
    "depth": {"default": 7.5, "min": 6.0, "max": 10.0, "desc": "body depth (m), front wall at y = 0"},
    "storeys": {"default": 2, "min": 2, "max": 3, "desc": "storeys below the eaves (ground-floor shop + 1-2 upper)"},
    "pitch": {"default": 46.0, "min": 40.0, "max": 52.0, "desc": "roof pitch (deg), ridge runs front to back"},
    "palette": {"default": "pastel", "choices": ["pastel", "carnival", "rose", "dusk"],
                "desc": "diamond tile palette (also sets wall, shopfront and joinery colours)"},
    "tints": {"default": 4, "min": 2, "max": 4, "desc": "number of tile tints used from the palette"},
    "diamond": {"default": 0.52, "min": 0.36, "max": 0.85, "desc": "lozenge tile width (m)"},
    "diamond_aspect": {"default": 1.55, "min": 1.0, "max": 2.1, "desc": "lozenge height / width"},
    "gable": {"default": "scroll", "choices": ["scroll", "bell", "stepped"], "desc": "shaped gable style"},
    "gable_tiles": {"default": True, "desc": "continue the harlequin tiling up into the gable"},
    "sign_text": {"default": "Dress Suit", "desc": "fascia sign lettering"},
    "door_side": {"default": "left", "choices": ["left", "right"], "desc": "which side the shop door is on"},
    "figure": {"default": True, "desc": "carved jester shop-figure on a bracket in the middle of the first floor"},
    "wear": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "age: chipped / missing tiles, grime, moss"},
    "lit": {"default": 0.75, "min": 0.0, "max": 1.0, "desc": "fraction of upper windows that are lit"},
    "lantern_cd": {"default": 3.5, "min": 0.0, "max": 45.0, "desc": "intensity of each wall lantern (cd)"},
    "display_cd": {"default": 28.0, "min": 0.0, "max": 50.0, "desc": "display window lamp intensity (cd)"},
}
VARIANTS = {
    "carnival_stepped_tall": {"width": 6.2, "depth": 8.0, "storeys": 3, "palette": "carnival", "gable": "stepped",
                              "sign_text": "Harlequin", "door_side": "right", "diamond": 0.46, "pitch": 50.0, "seed": 7},
    "rose_bell_wide": {"width": 9.0, "depth": 7.0, "palette": "rose", "gable": "bell", "tints": 3, "diamond": 0.62,
                       "diamond_aspect": 1.3, "sign_text": "Modes & Robes", "pitch": 42.0, "seed": 13},
    "dusk_worn": {"width": 8.0, "depth": 9.0, "storeys": 3, "palette": "dusk", "gable": "scroll", "tints": 2,
                  "diamond": 0.8, "diamond_aspect": 1.9, "gable_tiles": False, "wear": 0.95, "lit": 0.45,
                  "sign_text": "Tailor", "seed": 21},
}

PAL = {
    "pastel": dict(tints=["#93b4d8", "#eeb0a0", "#f1dc98", "#9fd0b8"], wall="#cdd8e0", ground="#c98356",
                   trim="#efe9da", joinery="#27456b", sign="#13171e"),
    "carnival": dict(tints=["#d5634a", "#39898a", "#e6bd58", "#efe2c4"], wall="#e3d8be", ground="#6d3232",
                     trim="#f1ead8", joinery="#1d4a3b", sign="#191311"),
    "rose": dict(tints=["#e09cb0", "#c0a8dc", "#f4ead7", "#aed4c0"], wall="#ead9d6", ground="#b47c6e",
                 trim="#f3ece2", joinery="#5a2338", sign="#22141a"),
    "dusk": dict(tints=["#7891b6", "#c79a9a", "#d8cea6", "#8fb0a2"], wall="#b9bfc4", ground="#a48f6a",
                 trim="#e4dccb", joinery="#3a2a22", sign="#15171b"),
}
JESTER = ("#b3272c", "#dba836", "#1e5a45")      # red, gold, green

H0, HS, ZP = 4.2, 2.9, 0.45        # ground storey, upper storey, plinth height
TW, TS = 0.35, 0.30                # gable-wall / side-wall thickness
OV = 0.30                          # eaves overhang
KN = 0.36                          # kneeler projection beyond the side walls


# ------------------------------------------------------------------------------------------ helpers
def _mould(name, pts, axis, prof, closed=False, miter_max=2.2, sm=40.0):
    """Run a moulding section along a planar polyline with mitred corners.
    prof = [(a, b)]: a along `axis`, b along the in-plane normal cross(tangent, axis)."""
    ax = np.array(axis, float)
    P = [np.array(pts[0], float)]
    for q in pts[1:]:
        q = np.array(q, float)
        if np.linalg.norm(q - P[-1]) > 1e-5:
            P.append(q)
    n, m = len(P), len(prof)

    def segn(i, j):
        t = P[j] - P[i]
        t = t / max(np.linalg.norm(t), 1e-9)
        return np.cross(t, ax)

    verts = []
    for i in range(n):
        if closed:
            n1, n2 = segn((i - 1) % n, i), segn(i, (i + 1) % n)
        else:
            n1 = segn(i - 1, i) if i > 0 else segn(0, 1)
            n2 = segn(i, i + 1) if i < n - 1 else segn(n - 2, n - 1)
        mv = (n1 + n2) / max(1.0 + float(np.dot(n1, n2)), 2.0 / (miter_max ** 2))
        for a, b in prof:
            verts.append(tuple(P[i] + a * ax + b * mv))
    faces = []
    for i in range(n if closed else n - 1):
        i1 = (i + 1) % n
        for k in range(m):
            k1 = (k + 1) % m
            faces.append((i * m + k, i * m + k1, i1 * m + k1, i1 * m + k))
    if not closed:
        faces.append(tuple(range(m)))
        faces.append(tuple((n - 1) * m + k for k in reversed(range(m))))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    smooth(o, sm)
    return o


def _cornice(h, proj):
    """Classical cornice section as (z, out) pairs: fillet, cyma, corona, weathered top."""
    q = [(0.0, 0.0), (0.0, 0.2), (0.18, 0.22), (0.32, 0.4), (0.5, 0.72), (0.58, 0.95), (0.84, 1.0), (0.88, 0.95)]
    return [(0.0, -0.03)] + [(a * h, b * proj) for a, b in q] + [(h, 0.0), (h, -0.03)]


def _clip(poly, px, pz, nx, nz):
    """Keep the part of a convex polygon where (p - P0).n >= 0."""
    out = []
    k = len(poly)
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        da = (a[0] - px) * nx + (a[1] - pz) * nz
        db = (b[0] - px) * nx + (b[1] - pz) * nz
        if da >= 0:
            out.append(a)
        if (da > 0 > db) or (da < 0 < db):
            t = da / (da - db)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    res = []
    for q in out:
        if not res or hypot(q[0] - res[-1][0], q[1] - res[-1][1]) > 1e-5:
            res.append(q)
    if len(res) > 1 and hypot(res[0][0] - res[-1][0], res[0][1] - res[-1][1]) < 1e-5:
        res.pop()
    return res if len(res) >= 3 else []


def _area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def _sub_rect(poly, r):
    """Convex polygon minus an axis-aligned rect -> list of convex pieces."""
    x0, z0, x1, z1 = r
    out = [_clip(poly, x0, 0, -1, 0), _clip(poly, x1, 0, 1, 0)]
    mid = _clip(poly, x0, 0, 1, 0)
    mid = _clip(mid, x1, 0, -1, 0) if mid else []
    if mid:
        out += [_clip(mid, 0, z0, 0, -1), _clip(mid, 0, z1, 0, 1)]
    return [q for q in out if q]


def _inset(poly, g):
    res = list(poly)
    k = len(poly)
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        dx, dz = b[0] - a[0], b[1] - a[1]
        ln = hypot(dx, dz)
        if ln < 1e-6:
            continue
        nx, nz = -dz / ln, dx / ln
        res = _clip(res, a[0] + nx * g, a[1] + nz * g, nx, nz)
        if not res:
            return []
    return res


def _tile_mesh(name, polys, thick, bev, rng):
    """Individually modelled chamfered tiles standing proud of the wall plane y = 0 (front face -Y)."""
    verts, faces = [], []
    for poly in polys:
        k = len(poly)
        cx = sum(q[0] for q in poly) / k
        cz = sum(q[1] for q in poly) / k
        rin = 1e9
        for i in range(k):
            a, b = poly[i], poly[(i + 1) % k]
            dx, dz = b[0] - a[0], b[1] - a[1]
            rin = min(rin, abs((cx - a[0]) * dz - (cz - a[1]) * dx) / max(hypot(dx, dz), 1e-6))
        f = max(0.35, 1.0 - bev / max(rin, 1e-3))
        th = thick * rng.u(0.8, 1.2)
        b0 = len(verts)
        verts += [(x, 0.004, z) for x, z in poly]
        verts += [(cx + (x - cx) * f, -th, cz + (z - cz) * f) for x, z in poly]
        for i in range(k):
            j = (i + 1) % k
            faces.append((b0 + i, b0 + j, b0 + k + j, b0 + k + i))
        faces.append(tuple(b0 + k + i for i in range(k)))
    o = mesh_object(name, verts, faces)
    flat(o)
    return o


def _gable_half(style, hw, Hg, n_steps):
    """Right half of the shaped gable above the eaves: [(x, h)] from the kneeler top to the apex (0, Hg)."""
    if style == "stepped":
        sh = Hg / n_steps
        xs = [hw - k * (hw - 0.5) / (n_steps - 1) for k in range(n_steps)]
        pts = []
        for k in range(n_steps):
            pts.append((xs[k], (k + 1) * sh))
            if k < n_steps - 1:
                pts.append((xs[k + 1], (k + 1) * sh))
        pts.append((0.0, Hg))
        return pts
    if style == "bell":
        kn = 0.45
        pts = [(hw, kn), (hw - 0.22, kn)]
        p0, p1, p2, p3 = (hw - 0.22, kn), (0.52 * hw, kn + 0.1), (0.42 * hw, 0.45 * Hg), (0.37 * hw, 0.72 * Hg)
        for i in range(1, 15):
            t = i / 14
            w = ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3)
            pts.append((w[0] * p0[0] + w[1] * p1[0] + w[2] * p2[0] + w[3] * p3[0],
                        w[0] * p0[1] + w[1] * p1[1] + w[2] * p2[1] + w[3] * p3[1]))
        x3, z3 = p3
        pts += [(x3 + 0.14, z3), (x3 + 0.14, z3 + 0.14), (x3, z3 + 0.14)]
        for i in range(1, 11):
            a = radians(90 * i / 10)
            pts.append((x3 * cos(a), z3 + 0.14 + (Hg - z3 - 0.14) * sin(a)))
        return pts
    # scroll: cove, shoulder block, convex quadrant, round head
    kn = 0.5
    pts = [(hw, kn), (hw - 0.2, kn)]
    xs, x1, z1 = hw - 0.2, 0.72 * hw, 0.5 * Hg
    for i in range(1, 11):
        t = radians(90 * i / 10)
        pts.append((xs - (xs - x1) * sin(t), kn + (z1 - kn) * (1 - cos(t))))
    pts += [(x1 + 0.12, z1), (x1 + 0.12, z1 + 0.16), (x1 - 0.04, z1 + 0.16)]
    xa, za = x1 - 0.04, z1 + 0.16
    x2 = max(0.23 * hw, 0.75)
    z2 = Hg - 0.12 - x2
    for i in range(1, 11):
        t = radians(90 * i / 10)
        pts.append((x2 + 0.12 + (xa - x2 - 0.12) * cos(t), za + (z2 - za) * sin(t)))
    pts += [(x2 + 0.12, z2 + 0.12), (x2, z2 + 0.12)]
    for i in range(1, 9):
        a = radians(90 * i / 8)
        pts.append((x2 * cos(a), z2 + 0.12 + x2 * sin(a)))
    return pts


def _grain(objs, axis):
    for o in objs:
        uv_grain(o, axis=axis)
    return objs


# --------------------------------------------------------------------------------- repeated pieces
def _plain_window(w, h, m_join, m_pane, m_stone, m_iron, m_curtain=None, shutters=False):
    """Recessed two-light window for a w x h opening; origin = sill centre on the outer wall plane (-Y out)."""
    objs = []
    # CLEARANCE RULE: the pane is ONE sheet (the old front face of a 12 mm box whose back lay 1 mm off the reveal's
    # back wall), 13 mm clear of it and 5 mm behind the curtains
    pane = lod_faces(box("pane", (w + 0.04, 0.012, h + 0.04), loc=(0, 0.15, h / 2)), keep=(0, -1, 0), above=-1)
    assign(pane, m_pane)
    extra = []
    if m_curtain is not None:
        yc = 0.139
        cv, cf = [], []
        for sx in (-1, 1):
            q = [(sx * w / 2, 0.06), (sx * (w / 2 - 0.13 * w), 0.06), (sx * (w / 2 - 0.1 * w), h * 0.42), (sx * (w / 2 - 0.34 * w), h), (sx * w / 2, h)]
            b0 = len(cv)
            cv += [(x, yc, z) for x, z in q]
            cf.append(tuple(range(b0, b0 + 5)) if sx > 0 else tuple(reversed(range(b0, b0 + 5))))
        cur = mesh_object("curtains", cv, cf)
        recalc_normals(cur)
        uv_box(cur)
        assign(cur, m_curtain)
        extra.append(cur)
    if shutters:
        sw = w / 2 - 0.01
        for sx in (-1, 1):
            xc = sx * (w / 2 + 0.035 + sw / 2)
            for k in range(3):
                bd = box("shutter_board", (sw / 3 - 0.006, 0.024, h - 0.04), loc=(xc + (k - 1) * sw / 3, -0.02, h / 2 + 0.01))
                uv_grain(bd, axis="Z")
                extra.append(bd)
            for fz in (0.17, 0.83):
                bt = box("shutter_batten", (sw - 0.03, 0.02, 0.085), loc=(xc, -0.042, h * fz))
                uv_grain(bt, axis="X")
                extra.append(bt)
            for o in extra[-5:]:
                assign(o, m_join)
                paint(o, wear=mask_mul(noise_mask(0.1, seed=7 + int(sx)), 0.6), special=0.25)
            hk = box("shutter_hinge", (0.03, 0.012, 0.16), loc=(sx * (w / 2 + 0.02), -0.036, h * 0.83))
            hk2 = box("shutter_hinge", (0.03, 0.012, 0.16), loc=(sx * (w / 2 + 0.02), -0.036, h * 0.17))
            assign(hk, m_iron)
            assign(hk2, m_iron)
            extra += [hk, hk2]
    fr = 0.055
    st = [box("stile", (fr, 0.07, h), loc=(sx * (w / 2 - fr / 2), 0.095, h / 2)) for sx in (-1, 1)]
    st.append(box("mullion", (0.04, 0.055, h - 0.1), loc=(0, 0.1, h / 2)))
    rl = [box("head", (w - 0.1, 0.07, fr), loc=(0, 0.095, h - fr / 2)),
          box("rail", (w - 0.1, 0.07, 0.075), loc=(0, 0.095, 0.0375)),
          box("transom", (w - 0.1, 0.05, 0.035), loc=(0, 0.1, h * 0.63))]
    _grain(st, "Z")
    _grain(rl, "X")
    for o in st + rl:
        assign(o, m_join)
        paint(o, wear=mask_mul(noise_mask(0.12, seed=3), 0.5))
    sill = box("sill", (w + 0.2, 0.24, 0.075), loc=(0, 0.035, -0.02), bevel=0.012, segments=1)
    lintel = box("lintel", (w + 0.26, 0.06, 0.18), loc=(0, -0.012, h + 0.09), bevel=0.012, segments=1)
    key = extrude_profile("key", [(-0.06, 0), (0.06, 0), (0.085, 0.22), (-0.085, 0.22)], 0.075,
                          loc=(0, 0.018, h - 0.005), rot=(90, 0, 0))
    for o in (sill, lintel, key):
        assign(o, m_stone)
        paint(o, grime=0.25)
    return join([pane, sill, lintel, key] + st + rl + extra, "plain_window")


def _corbel_standin(w, d, m_trim, ms, far):
    """The corbel under a lantern-bay window as the reduced levels get it (lod_standin): the same square turning,
    closed top and bottom, so the reducer has nothing to tear (it left 6-10 loose triangles of each at lod 2 and
    open seams at lod 1).  lod 1: every course of the outline, the fillet as a plain box, the drop as a small ball.
    lod 2: the two cushion courses in three bands, ending in a point where the drop hangs."""
    if far:
        prof = [(0.0, -0.085), (0.07, 0.0), (0.265, 0.17), (0.48, 0.33), (0.5, 0.42), (0.0, 0.42)]
    else:
        prof = [(0.0, 0.0), (0.075, 0.0), (0.075, 0.03), (0.17, 0.055), (0.25, 0.125), (0.255, 0.16), (0.275, 0.165), (0.275, 0.19),
                (0.37, 0.215), (0.45, 0.285), (0.455, 0.32), (0.475, 0.325), (0.475, 0.35), (0.5, 0.355), (0.5, 0.42), (0.0, 0.42)]
    corb = lathe("corbel", prof, segments=4, rot=(0, 0, 45), smooth_angle=30)
    place(corb, loc=(0, 0, -0.48), scale=((w + 0.06) / 0.7071, (d + 0.03) / 0.35355, 1.0))
    parts = [corb]
    if not far:
        parts.append(box("corbel_fillet", (w + 0.14, 2 * d + 0.12, 0.035), loc=(0, 0, -0.075), segments=1))
        parts.append(smooth(sphere("drop", 0.055, loc=(0, -0.03, -0.5), segments=6, rings=4, scale=(1, 1, 1.25)), 60))
    for o in parts:
        assign(o, m_trim)
    return [instances_to_mesh("corbels", join(parts, "corbel_standin"), ms, keep_base=False)]


def _lantern_window(w, h, d, m_join, m_pane, m_trim, m_copper, m_curtain=None, ms=()):
    """Little projecting lantern-bay window with a corbel and an ogee copper hood; origin = sill centre.
    ms: where the windows will be stamped (the reduced levels get their corbels as clean stand-ins there)."""
    objs = []
    back = box("backplate", (w + 0.26, 0.05, h + 0.2), loc=(0, -0.005, h / 2), bevel=0.01, segments=1)
    # corbelled out in two cushion courses (a plain inverted pyramid, and then an ogee that stayed close to one, read
    # as a paper wedge from the street: the outline has to step): collar, lower ovolo, fillet, upper ovolo, fillet,
    # abacus - each ovolo swells out fast and ends upright. Square in plan like the hood, half of it in the wall
    corb = lathe("corbel", [(0.0, 0.0), (0.06, 0.0), (0.075, 0.012), (0.075, 0.03),
                            (0.10, 0.04), (0.17, 0.055), (0.225, 0.085), (0.25, 0.125), (0.255, 0.16),
                            (0.275, 0.165), (0.275, 0.19),
                            (0.30, 0.2), (0.37, 0.215), (0.425, 0.245), (0.45, 0.285), (0.455, 0.32),
                            (0.475, 0.325), (0.475, 0.35), (0.5, 0.355), (0.5, 0.42)], segments=4, rot=(0, 0, 45), smooth_angle=30)
    place(corb, loc=(0, 0, -0.48), scale=((w + 0.06) / 0.7071, (d + 0.03) / 0.35355, 1.0))
    corb2 = box("corbel_fillet", (w + 0.14, 2 * d + 0.12, 0.035), loc=(0, 0, -0.075), bevel=0.008, segments=1)
    drop = sphere("drop", 0.055, loc=(0, -0.03, -0.5), segments=10, rings=6, scale=(1, 1, 1.25))
    for o in (back, corb, corb2, drop):
        assign(o, m_trim)
    smooth(drop, 60)
    if len(ms):
        lod_standin([corb, corb2, drop], mid=lambda: _corbel_standin(w, d, m_trim, ms, False), far=lambda: _corbel_standin(w, d, m_trim, ms, True))
    objs += [back, corb, corb2, drop]
    ps = 0.055
    sill = box("sill", (w + 0.1, d + 0.05, 0.05), loc=(0, -(d + 0.05) / 2, -0.025), bevel=0.006, segments=1)
    eave = box("eave", (w + 0.14, d + 0.07, 0.045), loc=(0, -(d + 0.07) / 2, h + 0.022), bevel=0.006, segments=1)
    wood = [sill, eave]
    posts = [box("post", (ps, ps, h), loc=(sx * (w / 2 - ps / 2), -(d - ps / 2), h / 2)) for sx in (-1, 1)]
    posts.append(box("mullion", (0.04, 0.045, h), loc=(0, -(d - 0.025), h / 2)))
    rails = [box("top", (w, ps, 0.07), loc=(0, -(d - ps / 2), h - 0.035)),
             box("bot", (w, ps, 0.07), loc=(0, -(d - ps / 2), 0.035))]
    for fz in (0.36, 0.68):
        rails.append(box("bar", (w - 0.08, 0.03, 0.026), loc=(0, -(d - 0.02), h * fz)))
    sides = []
    for sx in (-1, 1):
        sides.append(box("side_top", (ps, d, 0.07), loc=(sx * (w / 2 - ps / 2), -d / 2, h - 0.035)))
        sides.append(box("side_bot", (ps, d, 0.07), loc=(sx * (w / 2 - ps / 2), -d / 2, 0.035)))
        for fz in (0.36, 0.68):
            sides.append(box("side_bar", (0.03, d - 0.05, 0.026), loc=(sx * (w / 2 - 0.02), -d / 2, h * fz)))
    _grain(posts, "Z")
    _grain(rails + [sill, eave], "X")
    _grain(sides, "Y")
    for o in wood + posts + rails + sides:
        assign(o, m_join)
        paint(o, wear=mask_mul(noise_mask(0.1, seed=5), 0.45))
    objs += wood + posts + rails + sides
    # glazing: separate quads per pane so every pane differs a little
    pw = (w - 2 * ps - 0.04) / 2
    zr = [0.07, h * 0.36, h * 0.68, h - 0.07]
    for ix in (-1, 1):
        for k in range(3):
            # CLEARANCE RULE: one sheet per pane (the end caps of the old 8 mm boxes lay in the posts' cheek planes)
            g = lod_faces(box("pane", (pw, 0.008, zr[k + 1] - zr[k] - 0.012), loc=(ix * (pw / 2 + 0.02), -(d - 0.052), (zr[k] + zr[k + 1]) / 2)),
                          keep=(0, -1, 0), above=-1)
            assign(g, m_pane)
            objs.append(g)
    for sx in (-1, 1):
        g = box("side_pane", (0.008, d - 0.03, h - 0.12), loc=(sx * (w / 2 - 0.03), -d / 2 - 0.01, h / 2))
        assign(g, m_pane)
        objs.append(g)
    if m_curtain is not None:
        yv = -(d - 0.042)
        sc = [(-w / 2 + 0.05, h - 0.07)]
        nsc = 5
        for k in range(nsc):
            x0 = -w / 2 + 0.05 + k * (w - 0.1) / nsc
            for j in range(1, 5):
                t = j / 4
                sc.append((x0 + t * (w - 0.1) / nsc, h - 0.07))
        vv = [(x, yv, z) for x, z in sc]
        low = []
        for k in range(nsc):
            x0 = -w / 2 + 0.05 + k * (w - 0.1) / nsc
            for j in range(0, 5):
                t = j / 4
                low.append((x0 + t * (w - 0.1) / nsc, yv, h - 0.25 - 0.09 * sin(pi * t)))
        top = [(q[0], yv, h - 0.07) for q in low]
        fcs = [(i, i + 1, len(low) + i + 1, len(low) + i) for i in range(len(low) - 1)]
        val = mesh_object("valance", low + top, fcs)
        dv = []
        for sx in (-1, 1):
            dv.append(box("drape", (0.1, 0.004, h - 0.3), loc=(sx * (w / 2 - 0.105), yv, (h - 0.2) / 2 + 0.03)))
        for o in [val] + dv:
            recalc_normals(o)
            uv_box(o)
            assign(o, m_curtain)
            objs.append(o)
    hood_h = 0.72
    hood = lathe("hood", [(0.5, 0.0), (0.52, 0.015), (0.5, 0.04), (0.4, 0.13), (0.26, 0.32), (0.13, 0.52), (0.055, 0.66),
                          (0.03, 0.75), (0.0, 0.78)], segments=4, rot=(0, 0, 45), smooth_angle=30)
    place(hood, loc=(0, 0, h + 0.044), scale=((w + 0.24) / 0.7071, (d + 0.12) / 0.35355, hood_h / 0.78))
    ball = sphere("hood_ball", 0.04, loc=(0, -0.012, h + 0.044 + hood_h + 0.015), segments=10, rings=6)
    for o in (hood, ball):
        assign(o, m_copper)
        paint(o, special=lambda co, n, f=noise_mask(0.25, seed=9, lo=0.2, hi=0.8): np.clip(0.45 + 0.55 * f(co, n), 0, 1))
    objs += [hood, ball]
    return join(objs, "lantern_window")


def _wall_lantern(m_iron, m_glass, m_emis, m_brass):
    """Small wrought-iron scroll bracket carrying a tapered glazed lantern; origin = fixing point, -Y out.
    Returns (mesh, flame position)."""
    objs = []
    bar = rounded_rect(0.024, 0.009, 0.002, 2)
    objs.append(box("plate", (0.08, 0.012, 0.42), loc=(0, -0.006, -0.02), bevel=0.003))
    for z in (0.15, -0.19):
        objs.append(sphere("stud", 0.012, loc=(0, -0.012, z), segments=8, rings=5))
    R = 0.57
    arm = bezier_path((0, -0.012, 0.1), (0, -0.26, 0.14), (0, -0.47, 0.2), (0, -R, 0.12), n=14)
    objs.append(sweep("arm", arm, bar, up=(1, 0, 0), smooth_path=False))
    stay = bezier_path((0, -0.012, -0.17), (0, -0.16, -0.16), (0, -0.33, -0.02), (0, -0.4, 0.165), n=12)
    objs.append(sweep("stay", stay, bar, up=(1, 0, 0), smooth_path=False))
    sc = spiral_path(0.012, 0.085, 1.6, n=26, a0_deg=20, plane="YZ", center=(0, -0.15, 0.0))
    objs.append(sweep("scroll", sc, bar, up=(1, 0, 0), scale_fn=lambda t: 0.55 + 0.45 * t, smooth_path=False))
    # lantern hanging from the arm tip
    zt = 0.12
    objs.append(torus("ring", 0.02, 0.004, loc=(0, -R, zt - 0.025), rot=(0, 90, 0), major_seg=12, minor_seg=6))
    ztop = zt - 0.06
    lh, r0, r1 = 0.3, 0.075, 0.115         # body: narrow at the bottom
    roof = lathe("lroof", [(r1 + 0.035, 0.0), (r1 + 0.03, 0.012), (0.07, 0.07), (0.03, 0.1), (0.022, 0.13), (0.03, 0.145), (0.0, 0.16)],
                 segments=4, loc=(0, -R, ztop - 0.12), rot=(0, 0, 45), smooth_angle=30)
    objs.append(roof)
    zb = ztop - 0.12 - lh
    objs.append(lathe("lbase", [(0.0, -0.05), (0.02, -0.045), (0.03, -0.02), (r0 + 0.012, -0.012), (r0 + 0.012, 0.0), (0.0, 0.0)],
                      segments=4, loc=(0, -R, zb), rot=(0, 0, 45), smooth_angle=30))
    for k in range(4):
        a = radians(45 + 90 * k)
        objs.append(sweep(f"rib{k}", [(r0 * cos(a), -R + r0 * sin(a), zb), (r1 * cos(a), -R + r1 * sin(a), zb + lh)], 0.006, sides=5,
                          smooth_path=False))
    for o in objs:
        assign(o, m_iron)
        paint(o, special=mask_mul(noise_mask(0.08, seed=4, lo=0.4, hi=0.8), 0.5))
    glass = cylinder("lglass", radius=r0 - 0.004, radius2=r1 - 0.004, depth=lh, verts=4, loc=(0, -R, zb + lh / 2), rot=(0, 0, 45), cap=False)
    assign(glass, m_glass)
    burner = cylinder("burner", 0.012, 0.07, loc=(0, -R, zb + 0.035), verts=8)
    assign(burner, m_brass)
    fpos = (0.0, -R, zb + 0.135)
    flame = sphere("flame", 0.03, loc=fpos, segments=10, rings=8, scale=(0.8, 0.8, 1.6))
    assign(flame, m_emis)
    return join(objs + [glass, burner, flame], "wall_lantern"), fpos


def _jester(m_fab, m_oak, m_skin, m_brass):
    """Jester shop mannequin on a turned base (origin at the floor): mi-parti doublet and hose, dagged
    collar and skirt, three-horned cap with bells, curly-toed shoes."""
    red, gold, green = m_fab
    objs = []
    base = lathe("mq_base", [(0.0, 0.0), (0.25, 0.0), (0.26, 0.02), (0.24, 0.045), (0.2, 0.055), (0.18, 0.08), (0.17, 0.1), (0.0, 0.1)], segments=24)
    assign(base, m_oak)
    objs.append(base)
    bells = []
    # legs + shoes
    for sx, mleg in ((1, gold), (-1, red)):
        fy = -0.05 if sx > 0 else 0.03
        leg = sweep("leg", [(sx * 0.085, 0.0, 0.98), (sx * 0.095, fy * 0.6, 0.6), (sx * 0.1, fy, 0.2)], 0.082, sides=8, step=0.09,
                    scale_fn=lambda t: 1.0 - 0.3 * t - 0.25 * t * t + 0.12 * sin(pi * min(1.0, t * 1.6)) * (t > 0.45))
        assign(leg, mleg)
        shoe = sweep("shoe", [(sx * 0.1, fy + 0.08, 0.145), (sx * 0.1, fy, 0.135), (sx * 0.105, fy - 0.11, 0.128),
                              (sx * 0.11, fy - 0.2, 0.15), (sx * 0.11, fy - 0.25, 0.21), (sx * 0.11, fy - 0.22, 0.265)],
                     0.05, sides=7, step=0.05, scale_fn=lambda t: 1.0 - 0.82 * t ** 1.5)
        assign(shoe, green)
        objs += [leg, shoe]
        bells.append((sx * 0.11, fy - 0.22, 0.275))
    # torso (mi-parti) and dagged skirt
    torso = lathe("torso", [(0.0, 0.88), (0.13, 0.88), (0.165, 0.94), (0.17, 1.02), (0.148, 1.12), (0.16, 1.22), (0.19, 1.34),
                            (0.2, 1.42), (0.175, 1.48), (0.08, 1.52), (0.058, 1.58), (0.0, 1.58)], segments=18)
    place(torso, scale=(1.0, 0.66, 1.0))
    assign(torso, red)
    assign_faces(torso, gold, lambda c, n: c.x < 0)
    objs.append(torso)
    npt = 10

    def tri(u, k):
        f = (u * k) % 1.0
        return 1.0 - abs(2 * f - 1.0)

    skirt = surface("skirt", lambda u, v: ((0.152 + 0.13 * v) * cos(2 * pi * u), (0.152 + 0.13 * v) * 0.72 * sin(2 * pi * u),
                                           1.13 - 0.2 * v - 0.11 * v * tri(u, npt)), nu=npt * 4, nv=3, closed_u=True, thickness=0.006)
    assign(skirt, gold)
    assign_faces(skirt, red, lambda c, n: int(((atan2(c.y / 0.72, c.x) / (2 * pi)) % 1.0) * npt + 0.5) % 2 == 0)
    belt = torus("belt", 0.152, 0.016, loc=(0, 0, 1.125), major_seg=20, minor_seg=6)
    place(belt, scale=(1.0, 0.7, 1.0))
    assign(belt, green)
    objs += [skirt, belt]
    # arms: one hand on the hip, one raised with a marotte
    for sx, marm, path in ((1, gold, [(0.185, 0.0, 1.45), (0.33, -0.03, 1.24), (0.22, -0.1, 1.1)]),
                           (-1, red, [(-0.185, 0.0, 1.45), (-0.37, -0.06, 1.36), (-0.45, -0.14, 1.6)])):
        arm = sweep("arm", path, 0.058, sides=8, step=0.07, scale_fn=lambda t: 1.0 - 0.32 * t)
        assign(arm, marm)
        hand = sphere("hand", 0.043, loc=path[-1], segments=10, rings=7, scale=(1, 0.8, 1.2))
        assign(hand, m_skin)
        cuff = cone("cuff", r1=0.068, r2=0.04, depth=0.07, verts=10,
                    loc=tuple(path[-1][i] + (path[1][i] - path[-1][i]) * 0.2 for i in range(3)),
                    rot=(35 if sx < 0 else -70, 0, 0))
        assign(cuff, green)
        objs += [arm, hand, cuff]
    stick = cylinder("marotte", 0.011, 0.62, loc=(-0.45, -0.15, 1.72), rot=(8, 6, 0), verts=8)
    assign(stick, m_oak)
    mhead = sphere("marotte_head", 0.05, loc=(-0.44, -0.19, 2.04), segments=10, rings=8)
    assign(mhead, m_skin)
    mcap = cone("marotte_cap", r1=0.055, r2=0.0, depth=0.12, verts=10, loc=(-0.44, -0.19, 2.12))
    assign(mcap, red)
    objs += [stick, mhead, mcap]
    bells.append((-0.44, -0.19, 2.185))
    # collar with points and bells
    ncp = 8
    collar = surface("collar", lambda u, v: ((0.065 + v * (0.15 + 0.09 * tri(u, ncp))) * cos(2 * pi * u),
                                             (0.065 + v * (0.15 + 0.09 * tri(u, ncp))) * 0.78 * sin(2 * pi * u) - 0.01,
                                             1.555 - 0.1 * v - 0.05 * v * v * tri(u, ncp)), nu=ncp * 4, nv=3, closed_u=True, thickness=0.006)
    assign(collar, green)
    objs.append(collar)
    for k in range(ncp):
        a = 2 * pi * (k + 0.5) / ncp
        bells.append((0.305 * cos(a), 0.305 * 0.78 * sin(a) - 0.01, 1.4))
    # head, nose, cap with three horns
    head = sphere("head", 0.105, loc=(0, -0.005, 1.7), segments=16, rings=12, scale=(0.9, 1.0, 1.16))
    nose = sphere("nose", 0.024, loc=(0, -0.108, 1.69), segments=8, rings=6, scale=(0.8, 1.2, 1.0))
    for o in (head, nose):
        assign(o, m_skin)
    cap = lathe("cap", [(0.101, 1.735), (0.108, 1.75), (0.106, 1.78), (0.085, 1.82), (0.045, 1.842), (0.0, 1.848)], segments=16, loc=(0, -0.005, 0))
    assign(cap, red)
    assign_faces(cap, gold, lambda c, n: c.x < 0)
    band = torus("cap_band", 0.103, 0.014, loc=(0, -0.005, 1.74), major_seg=18, minor_seg=6)
    assign(band, green)
    objs += [head, nose, cap, band]
    horns = [(red, [(0.05, 0.0, 1.78), (0.17, 0.0, 1.93), (0.31, -0.02, 1.96), (0.41, -0.04, 1.87), (0.43, -0.05, 1.76)]),
             (gold, [(-0.05, 0.0, 1.78), (-0.16, 0.0, 1.94), (-0.29, -0.02, 2.0), (-0.4, -0.04, 1.93), (-0.44, -0.05, 1.83)]),
             (green, [(0.0, 0.03, 1.8), (0.0, 0.06, 1.98), (0.0, -0.02, 2.1), (0.0, -0.13, 2.08), (0.0, -0.18, 1.99)])]
    for mh, path in horns:
        h = sweep("horn", path, 0.08, sides=8, step=0.05, scale_fn=lambda t: 1.0 - 0.86 * t)
        assign(h, mh)
        objs.append(h)
        bells.append((path[-1][0], path[-1][1], path[-1][2] - 0.02))
    bl = sphere("bell", 0.022, segments=7, rings=5)
    assign(bl, m_brass)
    objs.append(instances_to_mesh("bells", bl, [trs(b) for b in bells], keep_base=False))
    return join(objs, "jester")


def _dress_form(m_cloth, m_oak, m_brass, m_tape):
    """Tailor's dummy on a turned stand with a tape measure round its neck (origin at the floor)."""
    objs = []
    stand = lathe("df_stand", [(0.0, 0.0), (0.2, 0.0), (0.21, 0.015), (0.18, 0.035), (0.06, 0.06), (0.03, 0.1), (0.022, 0.2), (0.03, 0.42),
                               (0.02, 0.46), (0.026, 0.5), (0.018, 0.56), (0.018, 0.95), (0.0, 0.95)], segments=16)
    assign(stand, m_oak)
    body = lathe("df_body", [(0.0, 0.9), (0.11, 0.9), (0.165, 0.94), (0.17, 1.08), (0.132, 1.24), (0.165, 1.4), (0.185, 1.5), (0.16, 1.58),
                             (0.07, 1.615), (0.052, 1.66), (0.0, 1.66)], segments=18)
    place(body, scale=(1.0, 0.68, 1.0))
    assign(body, m_cloth)
    capn = lathe("df_cap", [(0.0, 1.655), (0.055, 1.655), (0.058, 1.67), (0.03, 1.685), (0.018, 1.7), (0.026, 1.72), (0.0, 1.735)], segments=14)
    assign(capn, m_brass)
    tape = sweep("tape", [(-0.075, -0.1, 1.2), (-0.07, -0.085, 1.45), (-0.05, -0.02, 1.625), (0.0, 0.03, 1.64), (0.05, -0.02, 1.625),
                          (0.075, -0.09, 1.42), (0.085, -0.105, 1.08)], rounded_rect(0.02, 0.003, 0.001, 1), up=(0, -1, 0), step=0.06)
    assign(tape, m_tape)
    objs += [stand, body, capn, tape]
    return join(objs, "dress_form")


# ------------------------------------------------------------------------------------------ build
def build(p, rng):
    W, D = float(p.width), float(p.depth)
    NU = int(p.storeys) - 1
    pal = PAL[p.palette]
    ZE = H0 + NU * HS
    tp = tan(radians(p.pitch))
    Hr = W / 2 * tp
    hw = W / 2 + KN
    s = -1.0 if p.door_side == "left" else 1.0
    wear = float(p.wear)
    seed = int(p.seed)

    m_wall = mat("plaster_lime", pal["wall"], grime=0.6)
    m_ground = mat("plaster_lime", pal["ground"])
    m_trim = mat("plaster_lime", pal["trim"], grime=0.55)
    m_stone = mat("stone_smooth", "#d6cebc", grime=0.7)
    m_plinth = mat("ashlar_limestone")
    m_join = mat("wood_painted", pal["joinery"])
    # (window_lit renders unlit in the engine, so occupied rooms are warm frosted glass behind modelled bars and curtains)
    m_lit = mat("glass_frosted", "#ffb768", glow=1.5)
    m_dark = mat("window_dark")
    m_glass = mat("glass_clear", opacity=0.12)
    m_roof = mat("roof_tile_clay")
    m_terra = mat("terracotta")
    m_copper = mat("copper_patina")
    m_oak = mat("wood_oak", "#6b4a2e")
    m_skin = mat("wood_oak", "#e2c9a4")
    m_iron = mat("iron_wrought")
    m_brass = mat("brass")
    m_emis = mat("emissive", "#ffd9a0", glow=9)
    m_lglass = mat("glass_lantern")
    m_fab = [mat("canvas", c) for c in JESTER]
    nt = int(p.tints)
    m_tiles = [mat("ceramic_glazed", c, rough=1.25) for c in pal["tints"][:nt]]

    def zroof(x):
        return ZE + 0.1 + Hr * (1.0 - abs(x) / (W / 2))

    # ================================================================== front ground floor
    DWd = 1.3
    xd = s * (W / 2 - 1.45)
    za = xd - s * (DWd / 2 + 0.5)
    zb = -s * (W / 2 - 0.78)
    xw = (za + zb) / 2
    ww = min(abs(zb - za) - 0.44, 3.7)
    zs = 0.8
    jamb = clamp(3.14 - zs - ww / 2, 0.45, 1.2)
    wh = jamb + ww / 2
    crown = zs + wh
    arch = arch_pts(ww, wh, kind="round", n=28)
    hole = [(xw + x, zs + y) for x, y in arch]
    fg = shape_extrude("wall_front_ground", [(-W / 2, 0), (W / 2, 0), (W / 2, H0), (-W / 2, H0)], [hole], depth=TW,
                       loc=(0, TW, 0), rot=(90, 0, 0))
    apply_transform(fg)
    densify(fg, 0.5)
    assign(fg, m_ground)
    paint(fg, grime=lambda co, n: np.clip(1.15 - co[:, 2] / 1.1, 0, 1) * 0.75,
          special=mask_mul(height_ramp(0.3, 1.6), noise_mask(0.7, seed=seed + 1), 0.5 * wear))

    # archivolt, keystone, imposts, sill
    apath = [(xw + x, 0.0, zs + y) for x, y in arch]
    arv = _mould("archivolt", apath, (0, -1, 0), [(-0.02, 0.0), (0.045, 0.0), (0.06, 0.018), (0.06, 0.06), (0.04, 0.08), (0.04, 0.145),
                                                    (0.07, 0.165), (0.078, 0.21), (-0.02, 0.21)], sm=35)
    assign(arv, m_trim)
    key = extrude_profile("keystone", [(-0.1, 0), (0.1, 0), (0.15, 0.36), (-0.15, 0.36)], 0.15, loc=(xw, 0.02, crown - 0.05),
                          rot=(90, 0, 0), bevel=0.01)
    stones = [key]
    for sx in (-1, 1):
        stones.append(box("impost", (0.3, 0.13, 0.13), loc=(xw + sx * (ww / 2 + 0.105), -0.045, zs + jamb), bevel=0.012))
    wsill = box("display_sill", (ww + 0.56, 0.3, 0.1), loc=(xw, -0.05, zs - 0.05), bevel=0.014)
    stones.append(wsill)
    for o in stones:
        assign(o, m_stone)
        paint(o, grime=0.2, wear=mask_mul(facing((0, 0, 1)), 0.4))
    # stall riser panel under the sill
    riser = shape_extrude("riser_panel", rounded_rect(ww + 0.1, zs - ZP - 0.16, 0.02, 2),
                          [rounded_rect(ww - 0.06, zs - ZP - 0.32, 0.02, 2)], depth=0.03, loc=(xw, 0.0, (zs + ZP) / 2 - 0.03),
                          rot=(90, 0, 0), bevel=0.004)
    assign(riser, m_trim)

    # display window joinery, set back in the reveal
    yj = 0.2
    jo = [_mould("dw_frame", [(q[0], yj, q[2]) for q in apath], (0, -1, 0), [(-0.045, 0.01), (0.045, 0.01), (0.045, -0.06), (0.03, -0.075),
                                                                              (-0.045, -0.075)], sm=35)]
    jo.append(box("dw_bottom", (ww, 0.09, 0.08), loc=(xw, yj, zs + 0.04), bevel=0.005))
    ztr = zs + jamb
    jo.append(box("dw_transom", (ww - 0.08, 0.08, 0.085), loc=(xw, yj, ztr), bevel=0.005))
    for fx in (-1 / 6, 1 / 6):
        jo.append(box("dw_mullion", (0.06, 0.065, jamb - 0.08), loc=(xw + fx * ww, yj, zs + 0.04 + jamb / 2), bevel=0.004))
    R = ww / 2
    for ang in (36, 72, 108, 144):
        a = radians(ang)
        r0, r1 = 0.3 * R, R - 0.06
        jo.append(box("dw_fanbar", (r1 - r0, 0.055, 0.05), loc=(xw + cos(a) * (r0 + r1) / 2, yj, ztr + sin(a) * (r0 + r1) / 2),
                      rot=(0, -ang, 0), bevel=0.003))
    hub = [(xw + 0.3 * R * cos(radians(k * 15)), yj, ztr + 0.3 * R * sin(radians(k * 15))) for k in range(13)]
    jo.append(_mould("dw_hub", hub, (0, -1, 0), [(-0.028, -0.022), (0.028, -0.022), (0.028, 0.022), (-0.028, 0.022)], sm=35))
    for o in jo:
        assign(o, m_join)
        paint(o, wear=mask_mul(noise_mask(0.15, seed=seed + 2), 0.35 + 0.4 * wear), grime=height_ramp(zs, zs + 0.5))
    glass = extrude_profile("dw_glass", arch, 0.006, loc=(xw, yj + 0.003, zs), rot=(90, 0, 0))
    assign(glass, m_glass)

    # display niche: stage, panelled sides, pleated curtain, pendant lamp
    nx0, nx1 = xw - ww / 2 - 0.25, xw + ww / 2 + 0.25
    ny0, ny1 = TW, 1.5
    nz0, nz1 = zs - 0.04, crown + 0.14
    nv = [(nx0, ny0, nz0), (nx1, ny0, nz0), (nx1, ny1, nz0), (nx0, ny1, nz0), (nx0, ny0, nz1), (nx1, ny0, nz1), (nx1, ny1, nz1), (nx0, ny1, nz1)]
    niche = mesh_object("niche", nv, [(0, 4, 7, 3), (1, 2, 6, 5), (4, 5, 6, 7), (3, 7, 6, 2)])
    assign(niche, m_join)
    nb = int((nx1 - nx0) / 0.2)
    stage = [plank("stage", length=ny1 - ny0, width=(nx1 - nx0) / nb - 0.004, thick=0.04, axis="Y",
                   loc=(nx0 + (k + 0.5) * (nx1 - nx0) / nb, (ny0 + ny1) / 2, zs - 0.02), bevel=0.0) for k in range(nb)]
    stg = join(stage, "stage")
    assign(stg, m_oak)
    paint(stg, wear=mask_mul(noise_mask(0.3, seed=seed + 3), 0.5))
    nfold = int((nx1 - nx0) / 0.16)
    cur = surface("curtain", lambda u, v: (nx0 + u * (nx1 - nx0), ny1 - 0.07 - 0.035 * sin(u * 2 * pi * nfold) * (0.6 + 0.4 * v),
                                           zs + v * (nz1 - zs)), nu=nfold * 6, nv=3)
    assign(cur, m_fab[2])
    lx, ly, lz = xw, 0.72, crown - 0.42
    rod = cylinder("lamp_rod", 0.007, nz1 - lz - 0.08, loc=(lx, ly, (nz1 + lz + 0.08) / 2), verts=6)
    shade = lathe("lamp_shade", [(0.018, 0.11), (0.03, 0.1), (0.06, 0.075), (0.19, 0.0), (0.185, -0.006), (0.055, 0.066), (0.0, 0.07)],
                  segments=20, loc=(lx, ly, lz))
    for o in (rod, shade):
        assign(o, m_brass)
    bulb = sphere("lamp_bulb", 0.04, loc=(lx, ly, lz + 0.02), segments=12, rings=8)
    assign(bulb, m_emis)
    if p.display_cd > 0:
        light("display", (lx, ly - 0.1, lz - 0.05), color="#ffc98a", intensity=p.display_cd, range=8.0, flicker=0.03, shadow_near=0.25)

    # mannequin, tailor's dummy and bolts of cloth
    jr = rng.fork("display")
    jes = _jester(m_fab, m_oak, m_skin, m_brass)
    place(jes, loc=(xw - s * ww * 0.16, 0.92, zs), rot=(0, 0, s * jr.u(8, 20)))
    if ww > 2.6:
        df = _dress_form(m_fab[0], m_oak, m_brass, m_fab[1])
        place(df, loc=(xw + s * ww * 0.24, 1.0, zs), rot=(0, 0, -s * jr.u(10, 30)), scale=0.92)
    bx = xw - s * (ww / 2 + 0.02)
    for k in range(3):
        bolt_c = cylinder(f"cloth_bolt{k}", jr.u(0.05, 0.075), jr.u(0.8, 1.15), verts=12, bevel=0.006)
        L = bolt_c.dimensions.z if hasattr(bolt_c, "dimensions") else 1.0
        place(bolt_c, loc=(bx + s * (0.02 + 0.1 * k), 0.75 + 0.2 * k, zs + 0.5 + 0.02 * k), rot=(jr.u(-6, 6), s * (-13 - 4 * k), 0))
        uv_cyl(bolt_c, axis="Z")
        assign(bolt_c, m_fab[k % 3])

    # shop door, fascia sign
    door = use("door_shop", width=DWd, height=3.0, paint=pal["joinery"], trim=pal["trim"], hand="left" if s < 0 else "right",
               wear=0.3 + 0.5 * wear, intensity=4.5, seed=seed + 11)
    door.place((xd, 0, 0), sockets={"light"})
    zsign = 3.4
    sgn = use("fascia_sign", text=str(p.sign_text), width=min(0.7 * W, 6.0), height=0.7, board=pal["sign"], trim=pal["joinery"],
              lamps=2, intensity=3.5, age=0.25 + 0.5 * wear, seed=seed + 12)
    sgn.place((0, 0, zsign), sockets={"light"})

    # ================================================================== front upper wall + shaped gable
    style = p.gable
    Hg = Hr + {"scroll": 1.3, "bell": 1.15, "stepped": 0.95}[style]
    half = _gable_half(style, hw, Hg, 5 if W < 7.6 else 6)
    half = [(max(x, (W / 2) * (1 - (h - 0.1) / Hr) + 0.2) if 0 < x else x, h) for x, h in half]
    gpts = [(x, ZE + h) for x, h in half]
    outline = [(W / 2, H0), (W / 2, ZE - 0.6), (hw, ZE - 0.28)] + gpts + [(-x, z) for x, z in reversed(gpts[:-1])] + \
              [(-hw, ZE - 0.28), (-W / 2, ZE - 0.6), (-W / 2, H0)]
    fu = shape_extrude("wall_front_upper", outline, depth=TW, loc=(0, TW, 0), rot=(90, 0, 0))
    apply_transform(fu)
    densify(fu, 0.6)
    assign(fu, m_wall)
    paint(fu, grime=mask_mul(noise_mask(1.2, seed=seed + 5), 0.08 + 0.3 * wear))

    cpath = [(x, TW / 2, z) for x, z in gpts] + [(-x, TW / 2, z) for x, z in reversed(gpts[:-1])]
    t2 = TW / 2
    cop = _mould("gable_coping", cpath, (0, -1, 0),
                 [(-t2 - 0.05, 0.0), (-t2 - 0.05, 0.05), (-t2 - 0.015, 0.075), (-0.1, 0.115), (0.0, 0.13), (0.1, 0.115), (t2 + 0.015, 0.075),
                  (t2 + 0.06, 0.05), (t2 + 0.06, -0.02), (t2 + 0.04, -0.04), (t2 + 0.04, -0.15), (t2 + 0.022, -0.17), (t2 - 0.01, -0.17),
                  (t2 - 0.01, -0.03), (-t2 + 0.01, -0.03)], sm=35)
    assign(cop, m_stone)
    paint(cop, grime=mask_mul(noise_mask(0.5, seed=seed + 6), 0.2 + 0.25 * wear), special=mask_mul(facing((0, 0, 1)), noise_mask(0.4, seed=seed + 7), 0.15 + 0.55 * wear))
    # finials
    urn = lathe("finial_urn", [(0.0, 0.0), (0.14, 0.0), (0.15, 0.03), (0.09, 0.06), (0.07, 0.1), (0.12, 0.16), (0.165, 0.25), (0.155, 0.33),
                               (0.09, 0.39), (0.06, 0.41), (0.078, 0.44), (0.04, 0.48), (0.018, 0.57), (0.0, 0.6)], segments=20,
                loc=(0, TW / 2, ZE + Hg + 0.12))
    fin = [urn]
    ballp = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.04), (0.06, 0.07), (0.052, 0.1), (0.11, 0.17), (0.125, 0.25), (0.09, 0.33), (0.0, 0.36)]
    spots = [(half[0][0] - 0.13, half[0][1])]
    if style == "scroll":
        spots.append((0.72 * hw + 0.02, 0.5 * Hg + 0.16))
    elif style == "bell":
        spots.append((0.37 * hw + 0.04, 0.72 * Hg + 0.14))
    else:
        spots += [(half[2 * k][0] - 0.13, half[2 * k][1]) for k in range(1, len(half) // 2 - 1)]
    for bxp, bzp in spots:
        for sx in (-1, 1):
            fin.append(lathe("finial_ball", ballp, segments=14, loc=(sx * bxp, TW / 2, ZE + bzp + 0.12)))
    for o in fin:
        assign(o, m_stone)
        paint(o, grime=0.3, special=mask_mul(noise_mask(0.2, seed=seed + 8), 0.5 * wear))

    # oculus (lit attic workroom)
    zoc = ZE + (0.4 if style != "stepped" else 0.36) * Hg
    roc = 0.42
    kos = [(0.0, zoc, roc + 0.2)]
    if style == "scroll":
        vr = 0.36
        vx, vz = 0.72 * hw - 0.15 - vr, ZE + 0.5 * Hg - 0.2
        vol = []
        for sx in (-1, 1):
            sp = spiral_path(0.05, vr, 1.75, n=44, a0_deg=90, plane="XZ", center=(sx * vx, -0.012, vz), ccw=sx > 0)
            sp = list(sp) + [(sx * (vx + vr), -0.012, vz + 0.12), (sx * (vx + vr), -0.012, vz + 0.26)]
            vol.append(sweep("volute", sp, [(-0.04, -0.02), (0.04, -0.02), (0.04, 0.012), (0.016, 0.036), (-0.016, 0.036), (-0.04, 0.012)],
                             up=(0, -1, 0), smooth_path=False, scale_fn=lambda t: 0.45 + 0.55 * min(1.0, t * 1.4)))
            vol.append(sphere("volute_eye", 0.075, loc=(sx * vx, -0.01, vz), segments=12, rings=8, scale=(1, 0.7, 1)))
            kos.append((sx * vx, vz, vr + 0.09))
        for o in vol:
            assign(o, m_stone)
            paint(o, grime=0.2)

    def keep_out(x, z):
        return any(hypot(x - cx_, z - cz_) < r_ for cx_, cz_, r_ in kos)
    ring = lathe("oculus_ring", [(roc, -0.02), (roc, 0.05), (roc + 0.03, 0.078), (roc + 0.07, 0.078), (roc + 0.1, 0.05), (roc + 0.14, 0.062),
                                 (roc + 0.165, 0.04), (roc + 0.165, -0.02)], segments=32, loc=(0, 0, zoc), rot=(90, 0, 0))
    assign(ring, m_trim)
    # CLEARANCE RULE: the oculus glass is ONE disc 14 mm off the wall (the old 12 mm drum put its back 2 mm from the
    # plaster); the hub ring stands 4 mm in front of it instead of dipping through it
    oglass = lod_faces(cylinder("oculus_glass", roc + 0.01, 0.012, loc=(0, -0.008, zoc), rot=(90, 0, 0), verts=32), keep=(0, -1, 0), above=-1)
    assign(oglass, m_lit)
    ob = [box("oculus_bar", (2 * roc, 0.035, 0.035), loc=(0, -0.025, zoc), bevel=0.003),
          box("oculus_bar", (0.035, 0.035, 2 * roc), loc=(0, -0.025, zoc), bevel=0.003),
          torus("oculus_hub", 0.14, 0.016, loc=(0, -0.034, zoc), rot=(90, 0, 0), major_seg=20, minor_seg=6)]
    for o in ob:
        assign(o, m_join)

    # ---- lantern-bay windows on the upper storeys
    lw, lh, ld = 0.92, 1.42, 0.27
    nwx = 2 if W < 7.6 else 3
    wx2 = [-W * 0.27, W * 0.27]
    wx3 = [-W * 0.31, 0.0, W * 0.31]
    lr = rng.fork("lit")
    slots = [(x, H0 + k * HS + 0.98) for k in range(NU) for x in (wx2 if (k == 0 and p.figure) or nwx == 2 else wx3)]
    lits = [lr.chance(p.lit) for _ in slots]
    if p.lit > 0 and not any(lits):
        lits[0] = True
    excl = []
    for lit_flag in (True, False):
        ms = [trs((x, 0, z)) for (x, z), l in zip(slots, lits) if l == lit_flag]
        if ms:
            basew = _lantern_window(lw, lh, ld, m_join, m_lit if lit_flag else m_dark, m_trim, m_copper, m_fab[0] if lit_flag else None, ms=ms)
            instances_to_mesh("lantern_windows_" + ("lit" if lit_flag else "dark"), basew, ms, keep_base=False)
    for x, z in slots:
        excl.append((x - lw / 2 - 0.15, z - 0.12, x + lw / 2 + 0.15, z + lh + 0.12))

    # ---- harlequin tiles
    fx0, fx1 = -W / 2 + 0.46, W / 2 - 0.46
    fz0, fz1 = H0 + 0.3, ZE - 0.16
    ncol = max(2, round((fx1 - fx0) / p.diamond))
    dw = (fx1 - fx0) / ncol
    mrow = max(2, round((fz1 - fz0) / (dw * p.diamond_aspect / 2)))
    dh = 2 * (fz1 - fz0) / mrow
    tr = rng.fork("tiles")
    polys = [[] for _ in range(nt)]
    gap = 0.011
    gout = np.array(outline, float)
    ga, gb = gout, np.roll(gout, -1, axis=0)

    def edge_dist(x, z):
        ab = gb - ga
        ap = np.array((x, z)) - ga
        t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return float(np.sqrt((((ga + ab * t[:, None]) - (x, z)) ** 2).sum(1)).min())

    def inside(x, z):
        c = ((ga[:, 1] > z) != (gb[:, 1] > z))
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = ga[:, 0] + (z - ga[:, 1]) * (gb[:, 0] - ga[:, 0]) / (gb[:, 1] - ga[:, 1])
        return int(np.sum(c & (x < xi))) % 2 == 1

    def tint_of(j, i):
        u = 2 * i + (j % 2)
        d1, d2 = (u + j) // 2, (u - j) // 2
        if nt == 2:
            return j % 2
        if nt == 3:
            return (2 * d1 + d2) % 3
        return (d1 % 2) + 2 * (d2 % 2)

    jmax = mrow
    if p.gable_tiles:
        jmax = int((ZE + Hg - fz0) / (dh / 2)) + 1
    miss = 0.03 * wear
    for j in range(0, jmax + 1):
        cz = fz0 + j * dh / 2
        in_field = j <= mrow
        for i in range(-3, ncol + 4):
            cx = fx0 + (i + 0.5 * (j % 2)) * dw
            dia = [(cx + dw / 2, cz), (cx, cz + dh / 2), (cx - dw / 2, cz), (cx, cz - dh / 2)]
            pieces = []
            if in_field:
                q = dia
                for (px, pz, nx, nz) in ((fx0, 0, 1, 0), (fx1, 0, -1, 0), (0, fz0, 0, 1), (0, fz1, 0, -1)):
                    q = _clip(q, px, pz, nx, nz) if q else []
                pieces = [q] if q else []
                for r in excl:
                    pieces = [pp for q in pieces for pp in _sub_rect(q, r)]
            if p.gable_tiles and cz + dh / 2 > ZE + 0.17:
                q = _clip(dia, 0, ZE + 0.17, 0, 1)
                if q and all(inside(x, z) and edge_dist(x, z) > 0.36 for x, z in q) and all(not keep_out(x, z) for x, z in q + [(cx, cz)]) and \
                        True:
                    pieces.append(q)
            for q in pieces:
                q = _inset(q, gap)
                if not q or _area(q) < 0.006:
                    continue
                if tr.chance(miss):
                    continue
                polys[tint_of(j, i)].append(q)
    ntile = 0
    for k in range(nt):
        if not polys[k]:
            continue
        ntile += len(polys[k])
        to = _tile_mesh(f"tiles_{k}", polys[k], 0.017, 0.007, tr)
        assign(to, m_tiles[k])
        paint(to, wear=mask_mul(noise_mask(0.35, seed=seed + 20 + k, lo=0.5, hi=0.8), 0.25 + 0.75 * wear),
              grime=mask_mul(noise_mask(0.9, seed=seed + 30, lo=0.3, hi=0.8), 0.15 + 0.45 * wear))

    # ================================================================== side and back walls
    def wall_xy(code, u):
        return {"R": (W / 2, u), "B": (-u, D), "L": (-W / 2, -u)}[code]

    RZ = {"R": 90, "B": 180, "L": 270}
    LOC = {"R": (W / 2 - TS, 0, 0), "B": (0, D - TS, 0), "L": (-W / 2 + TS, 0, 0)}
    URNG = {"R": (TW, D), "B": (-(W / 2 - TS), W / 2 - TS), "L": (-D, -TW)}
    pw_, ph_ = 0.85, 1.3
    wr = rng.fork("windows")
    nside = 2 if D < 8.5 else 3
    ys = [1.7 + k * (D - 1.7 - 2.1) / (nside - 1) for k in range(nside)]
    xb = W * 0.22
    xbd = -xb                                   # back door x
    holes = {c: {"g": [], "u": []} for c in "RBL"}
    wins = []                                   # (code, u, z, force_lit)

    for c in "RL":
        sgnu = 1 if c == "R" else -1
        for k in range(NU):
            for i, y in enumerate(ys):
                wins.append((c, sgnu * y, H0 + k * HS + 0.92, "u", k == 0 and i == 0))
        wins.append((c, sgnu * ys[-1 if c == "R" else 0] * (0.8 if c == "R" else 1.0), 1.55, "g", False))
    for k in range(NU):
        for sx in (-1, 1):
            wins.append(("B", -sx * xb, H0 + k * HS + 0.92, "u", k == 0 and sx > 0))
    wins.append(("B", -xb, 1.25, "g", True))
    if Hr > 2.3:
        wins.append(("B", 0.0, ZE + 0.3, "u", False))
    for c, u, z, lvl, force in wins:
        holes[c][lvl].append([(u - pw_ / 2, z), (u + pw_ / 2, z), (u + pw_ / 2, z + ph_), (u - pw_ / 2, z + ph_)])
    bdw, bdh, bdz = 0.95, 2.05, 0.18
    ubd = -xbd
    holes["B"]["g"].append([(ubd - bdw / 2, bdz), (ubd + bdw / 2, bdz), (ubd + bdw / 2, bdz + bdh), (ubd - bdw / 2, bdz + bdh)])

    for c in "RBL":
        u0, u1 = URNG[c]
        for lvl, z0, z1, m in (("g", 0.0, H0, m_ground), ("u", H0, ZE + 0.1, m_wall)):
            if c == "B" and lvl == "u":
                ol = [(u0, H0), (u1, H0), (u1, ZE - 0.6), (hw, ZE - 0.28), (hw, ZE + 0.28), (hw - 0.22, ZE + 0.28), (0.4, ZE + Hr + 0.6),
                      (-0.4, ZE + Hr + 0.6), (-hw + 0.22, ZE + 0.28), (-hw, ZE + 0.28), (-hw, ZE - 0.28), (u0, ZE - 0.6)]
            else:
                ol = [(u0, z0), (u1, z0), (u1, z1), (u0, z1)]
            wl = shape_extrude(f"wall_{c}_{lvl}", ol, holes[c][lvl], depth=TW if c == "B" else TS,
                               loc=(0, D - TW, 0) if c == "B" else LOC[c], rot=(90, 0, RZ[c]))
            apply_transform(wl)
            densify(wl, 0.55)
            assign(wl, m)
            if lvl == "g":
                paint(wl, grime=lambda co, n: np.clip(1.15 - co[:, 2] / 1.1, 0, 1) * 0.8,
                      special=mask_mul(height_ramp(0.3, 2.0), noise_mask(0.8, seed=seed + 40), 0.15 + 0.5 * wear))
            else:
                paint(wl, grime=mask_mul(noise_mask(1.3, seed=seed + 41), 0.12 + 0.3 * wear),
                      special=mask_mul(height_ramp(ZE + 0.1, ZE - 1.5), noise_mask(0.7, seed=seed + 42), 0.08 + 0.4 * wear))
    wms = {}
    for c, u, z, lvl, force in wins:
        x, y = wall_xy(c, u)
        shut = lvl == "u" and z < ZE - 1.0
        wms.setdefault((bool(force or wr.chance(0.4 * p.lit + 0.1)), shut), []).append(trs((x, y, z), (0, 0, RZ[c])))
    for (flag, shut), ms in wms.items():
        basew = _plain_window(pw_, ph_, m_join, m_lit if flag else m_dark, m_stone, m_iron, m_fab[0] if flag else None, shut)
        instances_to_mesh("windows_" + ("lit" if flag else "dark") + ("_sh" if shut else ""), basew, ms, keep_base=False)

    # back door: ledged plank door in the opening, strap hinges, ring pull, stone lintel and step
    bd = []
    npl = 5
    for k in range(npl):
        pk = plank("bd_plank", length=bdh, width=bdw / npl - 0.004, thick=0.04, axis="Z", loc=((k - (npl - 1) / 2) * bdw / npl, 0.14, bdh / 2),
                   bevel=0.004, warp=0.003, rng=wr)
        assign(pk, m_join)
        paint(pk, wear=mask_mul(noise_mask(0.2, seed=seed + 50 + k), 0.4 + 0.5 * wear), grime=height_ramp(0.0, 0.6))
        bd.append(pk)
    for z in (0.35, 1.65):
        st = box("bd_strap", (bdw * 0.8, 0.008, 0.05), loc=(-bdw * 0.08, 0.116, z), bevel=0.002)
        assign(st, m_iron)
        bd.append(st)
    rp = torus("bd_ring", 0.045, 0.007, loc=(bdw * 0.33, 0.112, 1.02), rot=(90, 0, 0), major_seg=14, minor_seg=6)
    assign(rp, m_iron)
    bl = box("bd_lintel", (bdw + 0.36, 0.07, 0.22), loc=(0, -0.015, bdh + 0.11), bevel=0.014)
    assign(bl, m_stone)
    bstep = stone_block("bd_step", (bdw + 0.5, 0.36, bdz + 0.02), loc=(0, -0.08, -(bdz + 0.02) / 2 + 0.01), rng=wr, detail=0.2)
    assign(bstep, m_stone)
    paint(bstep, grime=0.4, wear=mask_mul(facing((0, 0, 1)), 0.6), special=mask_mul(noise_mask(0.2, seed=seed + 51), 0.6))
    bdo = join(bd + [rp, bl, bstep], "back_door")
    place(bdo, loc=(xbd, D, bdz), rot=(0, 0, 180))

    # ================================================================== plinth, quoins, pilasters, mouldings
    PR = 0.05
    pl = []
    dj = DWd / 2 + 0.02
    for a, b in ((-W / 2 - PR, xd - dj), (xd + dj, W / 2 + PR)):
        if b - a > 0.05:
            pl.append(box("plinth_f", (b - a, 0.2, ZP), loc=((a + b) / 2, 0.1 - PR, ZP / 2), bevel=0.018))
    for sx in (-1, 1):
        pl.append(box("plinth_s", (0.2, D + 2 * PR, ZP), loc=(sx * (W / 2 + PR - 0.1), D / 2, ZP / 2), bevel=0.018))
    for a, b in ((-W / 2 - PR, xbd - bdw / 2 - 0.02), (xbd + bdw / 2 + 0.02, W / 2 + PR)):
        pl.append(box("plinth_b", (b - a, 0.2, ZP), loc=((a + b) / 2, D + PR - 0.1, ZP / 2), bevel=0.018))
    for o in pl:
        densify(o, 0.4)
        assign(o, m_plinth)
        paint(o, grime=height_ramp(0.0, 0.5), special=mask_mul(noise_mask(0.35, seed=seed + 60), facing((0, 1, 0)), 0.8))

    qr = rng.fork("quoins")
    qb = stone_block("quoin", (0.52, 0.3, 0.31), chamfer=0.016, rough=0.004, rng=qr, detail=0.3)
    assign(qb, m_stone)
    nq = int((H0 - 0.1 - ZP) / 0.33)
    qh = (H0 - 0.1 - ZP) / nq
    qm = []
    pr = 0.035
    for sx in (-1, 1):
        for sy, cy in ((-1, 0.0), (1, D)):
            for k in range(nq):
                lx_, ly_ = (0.52, 0.3) if k % 2 == 0 else (0.3, 0.52)
                sc_ = qr.u(0.96, 1.03)
                qm.append(trs((sx * (W / 2 + pr - lx_ * sc_ / 2), cy + sy * (pr - ly_ * sc_ / 2), ZP + (k + 0.5) * qh),
                              (0, 0, 0 if k % 2 == 0 else 90), (sc_, sc_, qh / 0.31 * 0.965)))
    qo = instances_to_mesh("quoins", qb, qm, keep_base=False)
    paint(qo, grime=height_ramp(0.2, 1.4), wear=mask_mul(noise_mask(0.3, seed=seed + 61), 0.3))

    pil = []
    for sx in (-1, 1):
        for sy, cy in ((-1, 0.0), (1, D)):
            z0, z1 = H0 + 0.12, ZE - 0.58
            pil.append(box("pilaster", (0.44, 0.44, z1 - z0), loc=(sx * (W / 2 + 0.03 - 0.22), cy + sy * (0.03 - 0.22), (z0 + z1) / 2), bevel=0.012))
            pil.append(box("pilaster_cap", (0.5, 0.5, 0.09), loc=(sx * (W / 2 + 0.03 - 0.22), cy + sy * (0.03 - 0.22), z1 - 0.2), bevel=0.012))
    zc = H0 - 0.05
    pil.append(_mould("string_course", [(-W / 2, 0, zc), (W / 2, 0, zc), (W / 2, D, zc), (-W / 2, D, zc)], (0, 0, 1), _cornice(0.2, 0.13), closed=True))
    ze0 = ZE - 0.22
    pil.append(_mould("eaves_cornice_r", [(W / 2, TW, ze0), (W / 2, D - TW, ze0)], (0, 0, 1), _cornice(0.3, 0.2)))
    pil.append(_mould("eaves_cornice_l", [(-W / 2, D - TW, ze0), (-W / 2, TW, ze0)], (0, 0, 1), _cornice(0.3, 0.2)))
    pil.append(_mould("gable_band_f", [(-hw, 0, ZE - 0.12), (hw, 0, ZE - 0.12)], (0, 0, 1), _cornice(0.24, 0.12)))
    pil.append(_mould("gable_band_b", [(hw, D, ZE - 0.12), (-hw, D, ZE - 0.12)], (0, 0, 1), _cornice(0.24, 0.12)))
    for o in pil:
        assign(o, m_trim)
        paint(o, grime=mask_mul(noise_mask(0.6, seed=seed + 62), 0.08 + 0.3 * wear), special=mask_mul(facing((0, 0, 1)), 0.12 + 0.4 * wear))
    # back gable coping
    bpath = [(hw, D - TW / 2, ZE + 0.28), (hw - 0.22, D - TW / 2, ZE + 0.28), (0.4, D - TW / 2, ZE + Hr + 0.6), (-0.4, D - TW / 2, ZE + Hr + 0.6),
             (-hw + 0.22, D - TW / 2, ZE + 0.28), (-hw, D - TW / 2, ZE + 0.28)]
    bc = _mould("back_coping", bpath, (0, -1, 0), [(-t2 - 0.05, 0.0), (-t2 - 0.05, 0.05), (-0.1, 0.1), (0.0, 0.11), (0.1, 0.1), (t2 + 0.05, 0.05),
                                                    (t2 + 0.05, 0.0), (t2 - 0.01, -0.03), (-t2 + 0.01, -0.03)])
    assign(bc, m_stone)
    paint(bc, grime=0.2, special=mask_mul(facing((0, 0, 1)), noise_mask(0.4, seed=seed + 63), 0.2 + 0.5 * wear))

    # ================================================================== roof
    y0, y1 = TW - 0.1, D - TW + 0.1
    nu = max(8, int((y1 - y0) / 0.45))
    rr = rng.fork("roof")
    sag = rr.u(0.02, 0.05) + 0.05 * wear
    ph = rr.u(0, 6.28)

    def roof_fn(sx):
        def fn(u, v):
            uu = u if sx > 0 else 1.0 - u
            x = sx * (W / 2 + OV) * (1.0 - v)
            z = zroof(x) + 0.055 * max(0.0, 1.0 - v / 0.16) ** 2
            z -= sag * sin(pi * uu) * sin(pi * v) + 0.012 * sin(uu * 9.0 + ph + v * 3.0) * sin(pi * v)
            return (x, y0 + uu * (y1 - y0), z)
        return fn

    for sx in (-1, 1):
        ro = surface(f"roof_{'r' if sx > 0 else 'l'}", roof_fn(sx), nu=nu, nv=10, thickness=0.07)
        assign(ro, m_roof)
        paint(ro, special=lambda co, n, f=noise_mask(0.9, seed=seed + 70): np.clip(
            f(co, n) * (0.25 + 0.6 * wear) + np.clip(1.0 - (co[:, 2] - ZE + 0.2) / 1.3, 0, 1) * 0.45 * (0.4 + wear), 0, 1),
              grime=mask_mul(noise_mask(1.6, seed=seed + 71), 0.4))
    zr = ZE + 0.1 + Hr
    rt = cylinder("ridge_tile", radius=0.115, radius2=0.1, depth=0.42, verts=10, rot=(90, 0, 0))
    apply_transform(rt)
    assign(rt, m_terra)
    nrt = int((y1 - y0 - 0.5) / 0.36)
    rms = []
    for k in range(nrt):
        t = (k + 0.5) / nrt
        rms.append(trs((rr.u(-0.006, 0.006), y0 + 0.25 + t * (y1 - y0 - 0.5), zr - 0.03 - sag * 0.0 + rr.u(-0.006, 0.006)),
                       (rr.u(-1.5, 1.5), 0, rr.u(-2, 2)), rr.u(0.97, 1.03)))
    rto = instances_to_mesh("ridge_tiles", rt, rms, keep_base=False)
    paint(rto, special=mask_mul(noise_mask(0.3, seed=seed + 72), 0.5 + 0.4 * wear), grime=0.3)

    # gutters, outlets, downpipes
    zg = zroof(W / 2 + OV) + 0.03
    xg = W / 2 + OV + 0.045
    gprof = [(0.065 * sin(radians(a)), 0.065 * cos(radians(a))) for a in range(180, 361, 30)] + \
            [(0.057 * sin(radians(a)), 0.057 * cos(radians(a))) for a in range(360, 179, -30)]
    cu = []
    cu.append(_mould("gutter_r", [(xg, TW + 0.03, zg), (xg, D - TW - 0.03, zg)], (0, 0, 1), gprof, sm=60))
    cu.append(_mould("gutter_l", [(-xg, D - TW - 0.03, zg), (-xg, TW + 0.03, zg)], (0, 0, 1), gprof, sm=60))
    ydp = D - 0.85
    zhop = min(ZE - 0.62, 11.9)
    for sx in (-1, 1):
        cu.append(sweep(f"outlet{sx}", [(sx * xg, ydp, zg - 0.05), (sx * xg, ydp, zg - 0.14), (sx * (W / 2 + 0.115), ydp, zhop + 0.2),
                                        (sx * (W / 2 + 0.115), ydp, zhop + 0.02)], 0.042, sides=10, step=0.06))
        nbk = max(3, int((D - 2 * TW) / 1.0))
        for k in range(nbk):
            yy = TW + 0.25 + k * (D - 2 * TW - 0.5) / (nbk - 1)
            bk = sweep("gutter_bracket", [(sx * (xg - 0.3), yy, zg + 0.06), (sx * (xg - 0.075), yy, zg - 0.005), (sx * xg, yy, zg - 0.074),
                                          (sx * (xg + 0.074), yy, zg - 0.005)], rounded_rect(0.025, 0.005, 0.001, 1), up=(0, 1, 0), step=0.08)
            cu.append(bk)
    for o in cu:
        assign(o, m_copper)
        paint(o, special=mask_mul(noise_mask(0.4, seed=seed + 73, lo=0.2, hi=0.8), 0.6 + 0.4 * wear))
    xp = W / 2 + 0.115
    dpo = []
    for sx in (-1, 1):
        dpo.append(lathe("hopper", [(0.045, -0.2), (0.052, -0.14), (0.1, -0.03), (0.115, -0.02), (0.115, 0.1), (0.13, 0.11), (0.13, 0.135),
                                    (0.1, 0.135), (0.1, 0.0), (0.0, 0.0)], segments=4, loc=(sx * xp, ydp, zhop), rot=(0, 0, 45), smooth_angle=30))
        dpo.append(cylinder("downpipe", 0.042, zhop - 0.18 - 0.42, loc=(sx * xp, ydp, (zhop - 0.18 + 0.42) / 2), verts=10))
        dpo.append(sweep("shoe", [(sx * xp, ydp, 0.46), (sx * xp, ydp, 0.3), (sx * (xp + 0.07), ydp, 0.17), (sx * (xp + 0.2), ydp, 0.1)], 0.042, sides=10,
                         step=0.05))
        nck = max(2, int(zhop / 1.6))
        for k in range(nck):
            zk = 0.75 + k * (zhop - 1.3) / max(nck - 1, 1)
            dpo.append(cylinder("pipe_collar", 0.052, 0.06, loc=(sx * xp, ydp, zk), verts=10))
            dpo.append(box("pipe_ears", (0.04, 0.19, 0.035), loc=(sx * (xp - 0.06), ydp, zk)))
    dpj = join(dpo, "downpipes")
    assign(dpj, m_copper)
    paint(dpj, special=lambda co, n, f=noise_mask(0.5, seed=seed + 74, lo=0.25, hi=0.75): np.clip(f(co, n) * (0.5 + 0.5 * wear), 0, 1),
          grime=height_ramp(0.0, 0.7))
    # wrought-iron tie anchors at the floor levels of the side walls
    anc = sweep("anchor", [(0.07, 0, 0.2), (0.0, 0, 0.23), (-0.075, 0, 0.16), (-0.06, 0, 0.07), (0.06, 0, -0.07), (0.075, 0, -0.16), (0.0, 0, -0.23),
                           (-0.07, 0, -0.2)], rounded_rect(0.03, 0.012, 0.003, 1), up=(0, -1, 0), step=0.05)
    anb = sphere("anchor_bolt", 0.022, segments=8, rings=5, scale=(1, 0.6, 1))
    anj = join([anc, anb], "anchor")
    assign(anj, m_iron)
    ams = []
    for k in range(NU):
        for i in range(len(ys) - 1):
            ym = (ys[i] + ys[i + 1]) / 2
            ams.append(trs((W / 2 + 0.008, ym, H0 + k * HS + 0.25), (0, 0, 90)))
            ams.append(trs((-W / 2 - 0.008, ym, H0 + k * HS + 0.25), (0, 0, 270)))
        ams.append(trs((0.0, D + 0.008, H0 + k * HS + 0.25), (0, 0, 180)))
    ao = instances_to_mesh("wall_anchors", anj, ams, keep_base=False)
    paint(ao, special=0.5)
    if NU > 1:
        zb2 = H0 + HS - 0.1
        fb = _mould("floor_band", [(W / 2, 0.05, zb2), (W / 2, D, zb2), (-W / 2, D, zb2), (-W / 2, 0.05, zb2)], (0, 0, 1),
                    [(0.0, -0.02), (0.0, 0.03), (0.02, 0.05), (0.09, 0.06), (0.12, 0.04), (0.12, -0.02)])
        assign(fb, m_trim)
        paint(fb, grime=mask_mul(noise_mask(0.6, seed=seed + 62), 0.15 + 0.3 * wear))

    # chimney on the slope away from the door
    xc = -s * W * 0.23
    yc = D * 0.6
    hc = clamp(0.6 + Hr * abs(xc) / (W / 2) + 0.95, 1.2, 4.5)
    ch = use("chimney", width=0.8, depth=0.56, height=hc, pitch=float(p.pitch), material="brick_red", pots=2, sink=0.6, soot=0.4 + 0.4 * wear,
             seed=seed + 15)
    ch.place((xc, yc, zroof(xc) - 0.6), rot=(0, 0, 90 if xc > 0 else 270), sockets={"fx"})

    # ================================================================== wall lanterns + sockets
    lan, fpos = _wall_lantern(m_iron, m_lglass, m_emis, m_brass)
    lxo = clamp(0.14 * W, 0.86, 1.2) if p.figure else W / 2 - 0.19
    lz_ = H0 + (1.55 if p.figure else 0.62)
    lms = []
    for sx in (-1, 1):
        lxp = sx * lxo
        lms.append(trs((lxp, -0.02, lz_)))
        if p.lantern_cd > 0:
            light(f"lantern_{'r' if sx > 0 else 'l'}", (lxp + fpos[0], -0.02 + fpos[1], lz_ + fpos[2]), intensity=p.lantern_cd, range=9.0,
                  flicker=0.14, shadow_near=0.2)
    instances_to_mesh("wall_lanterns", lan, lms, keep_base=False)
    if p.figure:
        zsh = H0 + 0.5
        shelf = lathe("figure_bracket", [(0.0, -0.42), (0.05, -0.41), (0.075, -0.36), (0.06, -0.33), (0.1, -0.3), (0.2, -0.22), (0.27, -0.11),
                                         (0.36, -0.075), (0.37, -0.05), (0.41, -0.04), (0.42, 0.0), (0.0, 0.0)], segments=24, loc=(0, -0.1, zsh))
        place(shelf, scale=(1.0, 0.86, 1.0))
        assign(shelf, m_stone)
        paint(shelf, grime=mask_mul(height_ramp(zsh - 0.42, zsh), 0.5), wear=mask_mul(facing((0, 0, 1)), 0.4))
        fig = _jester(m_fab, m_oak, m_skin, m_brass)
        place(fig, loc=(0, -0.19, zsh), rot=(0, 0, rng.fork("figure").u(-6, 6)), scale=0.94)
    socket("hanging_sign", "attach", (-s * (W / 2 - 0.2), -0.03, 3.0), role="sign")
    socket("planter", "attach", (xw, -0.2, zs), role="planter")
