"""Half-timbered town house, the generic house of the square. A masonry ground floor (rubble stone, brick or
ashlar with dressed plinth blocks, quoins, stone sills and lintels) carries one or two jettied oak-framed upper
storeys: girding beam, projecting joist ends, bressummer and curved knee brackets that land on stone corbels
bedded in the piers between the openings, then a real frame of corner posts, bay posts, rails, studs and braces
(five patterns) pegged at the joints. The lime-plaster infill is triangulated panel by panel between the timbers
(constrained Delaunay), so it sits flat and trowelled behind the frame with thin run-off streaks under rails.
The framed gables (tie beam, principal rafters, queen posts round the attic window, collar, king post and
struts) follow the sagging roof, whose covering is laid tile by tile on the roof module's slopes. Large lit
dormers near the street, a tiled pentice hood on arch braces over the door, an optional oriel, chimney stacks,
the door / shopfront, wall lantern and window boxes are catalogue components re-skinned to one shared palette.
Age leans the upper storeys and sags the floors. All four sides are closed and detailed.

Levels: everything measured on the roof (ray casts, ridge height, the laid tiles, the triangle-cap decisions) is
taken once on the full model (lod_memo), so dormers, lights and tiles sit where they sit at lod 0. lod 1 keeps
every timber, peg, came and tile: timbers lose arris and back, pegs are their caps, tiles their top faces, the
plaster keeps one ring round each timber. lod 2 is flat plaster cut round the windows only, the frame as cards
in front of it, a coarser came lattice, no laid tiles (the roof module's own sheet carries their weathering).
"""
from tk import *
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import delaunay_2d_cdt

META = {"name": "Half-timbered town house", "class": "building", "mount": "ground", "focus": [-0.6, 0.0, 2.9], "lod": True}
PARAMS = {
    "width": {"default": 7.0, "min": 5.0, "max": 11.0, "desc": "front width of the ground floor (m), x in [-w/2, w/2]"},
    "depth": {"default": 8.0, "min": 6.0, "max": 11.0, "desc": "depth of the ground floor (m), y in [0, depth]"},
    "storeys": {"default": 2, "min": 2, "max": 3, "desc": "storeys below the eaves (ground floor + 1 or 2 timber storeys)"},
    "jetty": {"default": 0.35, "min": 0.0, "max": 0.45, "desc": "front jetty per storey (m); clamped so jetties + eaves stay within 0.7 m (3 storeys: the second jetty is the deeper one)"},
    "roof": {"default": "gable_front", "choices": ["gable_front", "eaves_front"], "desc": "roof orientation: gable or eaves to the street"},
    "roof_pitch": {"default": 52.0, "min": 42.0, "max": 58.0, "desc": "roof pitch (deg); eased a little on very deep spans"},
    "covering": {"default": "roof_tile_clay", "choices": ["roof_tile_clay", "roof_slate"], "desc": "roof covering (laid tile by tile)"},
    "pattern": {"default": "curved_braces", "choices": ["close_studding", "square_panels", "herringbone", "st_andrew", "curved_braces"],
                "desc": "timber-frame pattern of the upper storeys and gables"},
    "plaster": {"default": "#e6d8b6", "desc": "plaster / limewash tint of the infill panels"},
    "timber": {"default": "#4d3a2b", "desc": "oak frame stain (the colour the frame shows in neutral light)"},
    "accent": {"default": "#35553f", "desc": "paint of door, shutters, shopfront and window boxes"},
    "ground": {"default": "stone", "choices": ["stone", "brick", "ashlar"], "desc": "ground-floor masonry"},
    "glazing": {"default": "diamond", "choices": ["diamond", "square"], "desc": "leaded light pattern"},
    "lit_ratio": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "share of windows with the lights on"},
    "shop": {"default": False, "desc": "ground-floor shopfront instead of plain door and windows"},
    "sign_text": {"default": "PROVISIONS", "desc": "shop fascia / hanging sign lettering (shop only)"},
    "goods": {"default": "bottles", "choices": ["luggage", "hats", "bottles", "books", "cakes"], "desc": "shop display (shop only)"},
    "emblem": {"default": "bottle", "choices": ["cup", "boot", "key", "suitcase", "scissors", "bottle", "book", "star"], "desc": "hanging sign emblem (shop only)"},
    "door_side": {"default": "left", "choices": ["left", "right"], "desc": "which end of the front the door is at"},
    "door_hood": {"default": True, "desc": "tiled pentice hood on arch braces over the front door (house door only)"},
    "oriel": {"default": False, "desc": "oriel bay window on the first-floor front (the first jetty is reduced so it stays within 0.7 m)"},
    "shutters": {"default": True, "desc": "plank shutters on the ground-floor windows (where there is room)"},
    "dormers": {"default": 3, "min": 0, "max": 4, "desc": "number of roof dormers (large, nearest the street first; the middle one of an eaves-front roof grows into a cross gable)"},
    "chimneys": {"default": 1, "min": 0, "max": 2, "desc": "number of chimney stacks"},
    "window_boxes": {"default": 2, "min": 0, "max": 4, "desc": "flower boxes under first-floor front windows"},
    "lantern_cd": {"default": 16.0, "min": 0.0, "max": 60.0, "desc": "wall lantern by the door (cd); 0 = no lantern"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "irregularity, lean / sag of the upper storeys and ridge, weathering"},
}
VARIANTS = {
    "eaves_inn": {"width": 10.0, "depth": 7.0, "roof": "eaves_front", "pattern": "st_andrew", "plaster": "#dcc48f", "dormers": 3, "chimneys": 2,
                  "accent": "#6e2420", "window_boxes": 2, "door_side": "right", "age": 0.85, "seed": 13},
    "tall_shop": {"width": 6.0, "depth": 9.0, "storeys": 3, "shop": True, "pattern": "close_studding", "ground": "brick", "plaster": "#ead9c3",
                  "accent": "#64201d", "sign_text": "LUGGAGE", "goods": "luggage", "emblem": "suitcase", "jetty": 0.3, "window_boxes": 0,
                  "oriel": True, "dormers": 2, "age": 0.4, "seed": 7},
    "narrow_herringbone": {"width": 5.0, "depth": 7.0, "storeys": 3, "pattern": "herringbone", "plaster": "#d9b9a2", "timber": "#3a2a20",
                           "accent": "#27405e", "glazing": "square", "dormers": 2, "window_boxes": 1, "age": 0.7, "seed": 21},
    "square_slate": {"width": 8.5, "depth": 6.5, "roof": "eaves_front", "pattern": "square_panels", "ground": "ashlar", "covering": "roof_slate",
                     "plaster": "#cfd3c2", "timber": "#2f2823", "accent": "#1f2f3a", "jetty": 0.15, "lit_ratio": 0.3, "dormers": 2,
                     "lantern_cd": 10.0, "window_boxes": 0, "shutters": False, "age": 0.15, "seed": 34},
}

TD = 0.13          # frame timber depth
PR = 0.024         # timbers stand this proud of the plaster
SILL = 0.95        # window sill above the floor of a timber storey


def _ccw(pts):
    a = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        a += x0 * y1 - x1 * y0
    return pts if a > 0 else pts[::-1]


def _rect(a, b, c, d):
    return [(a, c), (b, c), (b, d), (a, d)]


def _lin(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def _tris(o):
    o.data.calc_loop_triangles()
    return len(o.data.loop_triangles)


def _pip(P, poly):
    """Even-odd point-in-polygon for an (N,2) array."""
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = poly[i][0], poly[i][1], poly[j][0], poly[j][1]
        dy = yj - yi
        if abs(dy) > 1e-12:
            c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / dy + xi)
            inside ^= c
        j = i
    return inside


def _offset(poly, d):
    """Offset a CCW polygon outwards by d (mitred, clamped)."""
    P = np.asarray(poly, dtype=float)
    e1 = P - np.roll(P, 1, 0)
    e2 = np.roll(P, -1, 0) - P

    def nrm(e):
        n = np.stack([e[:, 1], -e[:, 0]], 1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    n1, n2 = nrm(e1), nrm(e2)
    b = n1 + n2
    bl = np.linalg.norm(b, axis=1, keepdims=True)
    b = np.where(bl > 1e-6, b / np.maximum(bl, 1e-9), n1)
    k = d / np.maximum((b * n1).sum(1, keepdims=True), 0.4)
    return P + b * k


def _resample(poly, step):
    P = np.asarray(poly, dtype=float)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
        for j in range(n):
            out.append(a + (b - a) * (j / n))
    return np.array(out)


def wall_panel(name, L, z0, z1, openings, reveal, max_edge=0.35):
    """Wall skin in the local XZ plane (outside = -Y) with rectangular openings and their reveals."""
    xs = sorted(set([0.0, round(L, 4)] + [round(v, 4) for o in openings for v in (o[0], o[1])]))
    zs = sorted(set([round(z0, 4), round(z1, 4)] + [round(v, 4) for o in openings for v in (o[2], o[3])]))
    verts, faces, idx = [], [], {}

    def vid(x, y, z):
        k = (round(x, 4), round(y, 4), round(z, 4))
        if k not in idx:
            idx[k] = len(verts)
            verts.append(k)
        return idx[k]

    def inside(x, z):
        return any(o[0] < x < o[1] and o[2] < z < o[3] for o in openings)

    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            if inside((xs[i] + xs[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2):
                continue
            faces.append((vid(xs[i], 0, zs[j]), vid(xs[i + 1], 0, zs[j]), vid(xs[i + 1], 0, zs[j + 1]), vid(xs[i], 0, zs[j + 1])))
    r = reveal
    for (a, b, c, d) in openings:
        faces.append((vid(a, 0, c), vid(b, 0, c), vid(b, r, c), vid(a, r, c)))
        faces.append((vid(a, 0, d), vid(a, r, d), vid(b, r, d), vid(b, 0, d)))
        faces.append((vid(a, 0, c), vid(a, r, c), vid(a, r, d), vid(a, 0, d)))
        faces.append((vid(b, 0, c), vid(b, 0, d), vid(b, r, d), vid(b, r, c)))
    o = mesh_object(name, verts, faces)
    densify(o, max_edge)
    return o


def cames(name, rects, y, kind, pitch=0.13, cw=0.009):
    """Lead cames as a lattice of thin strips just in front of the glass (local XZ plane, facing -Y)."""
    verts, faces = [], []

    def strip(p, q):
        dx, dz = q[0] - p[0], q[1] - p[1]
        l = math.hypot(dx, dz)
        if l < 0.03:
            return
        nx, nz = -dz / l * cw / 2, dx / l * cw / 2
        i = len(verts)
        verts.extend([(p[0] - nx, y, p[1] - nz), (q[0] - nx, y, q[1] - nz), (q[0] + nx, y, q[1] + nz), (p[0] + nx, y, p[1] + nz)])
        faces.append((i, i + 1, i + 2, i + 3))

    for (a, b, c, d) in rects:
        if kind == "diamond":
            s = pitch * 1.4142
            k = a - d + s * 0.5
            while k < b - c:
                x0, x1 = max(a, c + k), min(b, d + k)
                if x1 - x0 > 0.01:
                    strip((x0, x0 - k), (x1, x1 - k))
                k += s
            k = a + c + s * 0.5
            while k < b + d:
                x0, x1 = max(a, k - d), min(b, k - c)
                if x1 - x0 > 0.01:
                    strip((x0, k - x0), (x1, k - x1))
                k += s
        else:
            nx_ = max(1, int(round((b - a) / pitch)))
            nz_ = max(1, int(round((d - c) / (pitch * 1.4))))
            for i in range(1, nx_):
                strip((a + (b - a) * i / nx_, c), (a + (b - a) * i / nx_, d))
            for j in range(1, nz_):
                strip((a, c + (d - c) * j / nz_), (b, c + (d - c) * j / nz_))
    if not faces:
        return None
    return mesh_object(name, verts, faces)


def build(p, rng):
    W, D, S = float(p.width), float(p.depth), int(p.storeys)
    shop = bool(p.shop)
    eaves_front = p.roof == "eaves_front"
    age = float(p.age)
    irr = 0.4 + 1.2 * age
    H0 = 3.4 if shop else 3.0            # ground-floor height
    HS = 2.75                            # timber storey height
    ZE = H0 + (S - 1) * HS               # eaves level
    OV = 0.2                             # roof overhang asked from the roof module
    EXTRA = 0.36 if eaves_front else 0.3  # what eaves + gutter / verge + bargeboard add beyond the wall face
    J = max(0.0, min(float(p.jetty) * (S - 1), 0.7 - EXTRA))   # total jetty of the top storey
    if J < 0.06:
        J = 0.0
    JY = [0.0, J] if S == 2 else [0.0, 0.4 * J, J]             # front offset of each storey (3 storeys: deeper second jetty)
    oriel = bool(p.oriel)
    ORIEL_OUT = 0.5                      # what the oriel bay projects from the frame
    if oriel and JY[1] + PR + ORIEL_OUT > 0.69:
        JY[1] = max(0.0, 0.69 - PR - ORIEL_OUT)
        if S == 2:
            J = JY[1]
    seed = int(p.seed)
    clay = p.covering == "roof_tile_clay"
    Lv = lod()                           # 0 full model, 1, 2 (see "Levels" above)
    OUTF, BACKF = (0, -1, 0), (0, 1, 0)  # in a wall face's local frame -Y looks out
    bv = (lambda b: b) if Lv == 0 else (lambda b: 0.0)      # arrises are modelled in the full model only

    # ---------------------------------------------------------------- palette (one shared set of 10 materials)
    tl = _lin(p.timber)
    lum = 0.2126 * tl[0] + 0.7152 * tl[1] + 0.0722 * tl[2] + 1e-5
    lvl = clamp(0.135 * (lum / 0.0475) ** 0.5, 0.08, 0.45)
    # wood_oak is already a mid brown and the tint multiplies it: aim the *result* at the stain colour
    TIMBER = mat("wood_oak", tuple(clamp(lvl * (c / lum) ** 0.88, 0.02, 1.0) for c in tl))
    PLASTER = mat("plaster_lime", p.plaster)
    MASON = mat({"stone": "rubble_fieldstone", "brick": "brick_red", "ashlar": "ashlar_limestone"}[p.ground])
    STONE = mat("stone_smooth", "#b3a68e")
    PAINT_A = mat("wood_painted", p.accent)
    IRON = mat("iron_wrought")
    LIT = mat("window_lit", "#ffc37a", glow=3.0, curtain=0.55)   # NB: without an explicit tint window_lit renders black
    DARK = mat("window_dark")
    LEAF = mat("leaf", "#7fa85c")
    ROOFM = mat("terracotta", "#d9a085", rough=0.8) if clay else mat("slate", "#c3cad2", rough=0.7)
    PAINT_F = TIMBER                     # window frames are oak like the frame they sit in
    LEAD = IRON                          # cames, flashings: one dark metal
    # catalogue components are re-skinned onto this palette: one draw call per material for the whole house
    RULES = {"wood_oak": TIMBER, "wood_endgrain": TIMBER, "wood_planks": TIMBER, "wood_weathered": TIMBER, "wood_painted": PAINT_A,
             "stone_smooth": STONE, "stone_rough": STONE, "mortar": STONE, "plaster_lime": PLASTER, "plaster_rough": PLASTER,
             "iron_painted": IRON, "iron_rusty": IRON, "brass": IRON, "zinc": IRON, "copper_patina": IRON, "metal_plain": IRON, "lead": IRON,
             "window_lit": LIT, "window_dark": DARK, "emissive": LIT, "glass_lantern": LIT, "glass_frosted": LIT, "glass_clear": DARK,
             "brick_red": MASON, "brick_ochre": MASON, "leaf": LEAF, "hedge": LEAF, "moss": LEAF, "soil": LEAF,
             "ceramic_glazed": PLASTER, "leather": TIMBER, "iron_wrought": IRON, "paper": PLASTER,
             "roof_tile_clay": ROOFM, "roof_slate": ROOFM, "wood_shingle": ROOFM, "terracotta": ROOFM, "slate": ROOFM, "_bloom": ROOFM}
    keep = {m.name for m in (TIMBER, PLASTER, MASON, STONE, PAINT_A, IRON, LIT, DARK, LEAF, ROOFM)}
    TRI_CAP = 157500                     # optional dressing (dormers, flower boxes) is dropped before the class budget is exceeded

    def scene_tris():
        return sum(_tris(ob) for ob in bpy.context.scene.objects if ob.type == "MESH")

    def skin(o, tag, **over):
        """Re-skin a placed catalogue component onto the shared palette (keeps the draw calls of the house low)."""
        rules = dict(RULES)
        rules.update(over)
        ms = o.data.materials
        for i, m in enumerate(ms):
            if m is None or m.name in keep:
                continue
            base = m.name.split("~")[0].split(".")[0]
            r = rules.get(base)
            t = m.get("tsq_tint")
            lum_ = (0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2]) if t else 0.0
            if base == "plain":            # blooms: clay red or limewash cream, whichever is nearer
                r = PLASTER if lum_ > 0.4 else rules["_bloom"]
            elif base == "wood_painted" and lum_ > 0.25 and "wood_painted" not in over:
                r = PLASTER                # cream lettering / sign boards stay light against the accent paint
            if r is not None:
                ms[i] = r
        return o

    def tone(o, r, wk=0.0, g=0.4):
        """Every timber its own value: some darker with old tar and dirt, some bleached paler."""
        paint(o, grime=r.u(0.0, g), wear=min(1.0, r.u(0.0, 0.1) + wk * r.u(0.4, 1.0)))
        return o

    # ---------------------------------------------------------------- roof module first
    # NOTE: roof_gable (ridge='y') rotates every object in the scene while it builds, so nothing of ours may exist yet.
    PDp = D + J
    yc = (D - J) / 2
    span = PDp if eaves_front else W
    pitch = clamp(float(p.roof_pitch) - max(0.0, span - 8.0) * 2.2, 42.0, 58.0)
    rr = rng.fork("roof")
    rkw = dict(pitch=pitch, type="gable", overhang=OV, covering=p.covering, sag=clamp(0.07 + 0.24 * age, 0.0, 0.35), gable="bargeboard",
               gable_wall="none", attic_window="none", rooflight="none", eaves="rafters", ridge_tile="half_round",
               finial="ball" if not eaves_front else "none", gutter="iron", wall_color=p.plaster, timber_color=p.timber, trim_color=p.timber,
               moss=0.15 + 0.5 * age, seed=seed + 11)
    if PDp <= 10.0:
        roof = use("roof_gable", width=W, depth=PDp, ridge="x" if eaves_front else "y", **rkw)
        Mroof = trs((0, yc, ZE), (0, 0, 0), 1.0)
    else:
        dd = min(W, 10.0)
        roof = use("roof_gable", width=PDp, depth=dd, ridge="y" if eaves_front else "x", **rkw)
        Mroof = trs((0, yc, ZE), (0, 0, 90), (1.0, W / dd, 1.0))
    ro = roof.place_matrix(Mroof, sockets=False)
    bpy.context.view_layer.update()

    def world_mesh(o, slots=None):
        me = o.data
        mw = np.array(o.matrix_world, dtype=np.float64)
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
        polys = [tuple(pl.vertices) for pl in me.polygons if slots is None or pl.material_index in slots]
        return co, polys

    cov_slots = {i for i, m in enumerate(ro.data.materials) if m is not None and m.name.split("~")[0].split(".")[0] == p.covering}
    rco, rpolys = world_mesh(ro, cov_slots)
    cov = rco[np.unique(np.concatenate([np.array(q) for q in rpolys]))]
    bvh = BVHTree.FromPolygons([Vector(v) for v in rco], rpolys)
    ridge_z = lod_memo("ridge_z", lambda: float(cov[:, 2].max()))
    skin(ro, "roof", wood_painted=TIMBER)
    DOWN = Vector((0, 0, -1))

    def roof_cast(x, y):
        loc, nor, _i, _d = bvh.ray_cast(Vector((x, y, ZE + 40.0)), DOWN)
        if loc is None:
            return None
        if nor.z < 0:
            nor = -nor
        return tuple(loc), tuple(nor)

    def roof_hit(x, y):
        """Point and normal of the roof covering above (x, y) - of the FULL model's roof at every level."""
        hit = lod_memo(("roof", round(x, 4), round(y, 4)), lambda: roof_cast(x, y))
        if hit is None:
            return None, None
        return Vector(hit[0]), Vector(hit[1])

    # ---------------------------------------------------------------- wall faces
    class Face:
        """One wall face. Local u runs left -> right as seen from outside, local -Y is outwards."""

        def __init__(s, name, ox, oy, th, L):
            s.name, s.ox, s.oy, s.th, s.L = name, ox, oy, th, L
            s.c, s.s = round(math.cos(math.radians(th)), 6), round(math.sin(math.radians(th)), 6)
            s.g = {}

        def add(s, grp, o):
            s.g.setdefault(grp, []).append(o)
            return o

        def w(s, u, y, z):
            return (s.ox + u * s.c - y * s.s, s.oy + u * s.s + y * s.c, z)

        def finish(s):
            out = {}
            for k, objs in s.g.items():
                o = join(objs, f"{s.name}_{k}")
                place(o, loc=(s.ox, s.oy, 0), rot=(0, 0, s.th))
                out[k] = o
            return out

    def faces_of(tag, yf):
        return {"front": Face(tag + "_front", -W / 2, yf, 0, W), "right": Face(tag + "_right", W / 2, yf, 90, D - yf),
                "back": Face(tag + "_back", W / 2, D, 180, W), "left": Face(tag + "_left", -W / 2, D, 270, D - yf)}

    lit_spots = []        # (face key, storey, world point just inside the room, above the window head)

    # ---------------------------------------------------------------- leaded window (inline, between studs / in masonry)
    def add_window(F, fkey, storey, uc, za, w, h, lit, kind, r, shutters=False):
        ua, ub, zb = uc - w / 2, uc + w / 2, za + h
        y0 = 0.1 if kind == "stone" else 0.035
        fw, fd = 0.055, 0.06
        n = 3 if w > 1.2 else 2 if w > 0.62 else 1

        def fr(sz, loc, ax):
            o = box("wf", sz, loc=loc, bevel=0.004, segments=1)
            uv_grain(o, axis=ax)
            assign(o, PAINT_F)
            lod_faces(o, drop=BACKF)
            lod_faces(o, keep=OUTF, above=1)
            F.add("win", o)

        fr((fw, fd, h), (ua + fw / 2, y0 + fd / 2, za + h / 2), "Z")
        fr((fw, fd, h), (ub - fw / 2, y0 + fd / 2, za + h / 2), "Z")
        fr((w - 2 * fw, fd, fw), (uc, y0 + fd / 2, zb - fw / 2), "X")
        fr((w - 2 * fw, fd, fw + 0.01), (uc, y0 + fd / 2, za + fw / 2 + 0.005), "X")
        for i in range(1, n):
            fr((0.045, fd * 0.9, h - 2 * fw), (ua + w * i / n, y0 + fd * 0.45 + 0.006, za + h / 2), "Z")
        if h > 1.42:
            fr((w - 2 * fw, fd * 0.8, 0.045), (uc, y0 + fd * 0.4 + 0.012, za + h * 0.7), "X")
        yg = y0 + 0.04
        rects = [(ua + w * i / n, ua + w * (i + 1) / n, za, zb) for i in range(n)]
        # one glass sheet per window, one UV cell across it (each window its own cell: its own room, curtains, warmth)
        g = mesh_object("glass", [(ua, yg, za), (ub, yg, za), (ub, yg, zb), (ua, yg, zb)], [(0, 1, 2, 3)])
        uvl = g.data.uv_layers[0] if g.data.uv_layers else g.data.uv_layers.new(name="UVMap")
        ou, ov = float(int(r.u(0, 37))), float(int(r.u(0, 23)))
        guv = [(ou, ov), (ou + 1.0, ov), (ou + 1.0, ov + 1.0), (ou, ov + 1.0)]
        for li, lp in enumerate(g.data.loops):
            uvl.data[li].uv = guv[lp.vertex_index]
        uv_keep(g)
        assign(g, LIT if lit else DARK)
        if lit:
            paint(g, grime=r.pick([0.0, 0.12, 0.3, 0.5]))
        F.add("glass", g)
        if Lv == 0:
            c = cames("cames", rects, yg - 0.004, p.glazing)
        elif Lv == 1:                          # one lattice across the window: the mullions hide where the lights' lattices met
            c = cames("cames", [(ua, ub, za, zb)], yg - 0.004, p.glazing)
        else:                                  # far: half the cames, twice as wide (the pane keeps its tone and its leaded read)
            c = cames("cames", [(ua, ub, za, zb)], yg - 0.004, p.glazing, pitch=0.26, cw=0.02)
        if Lv >= 2 and kind != "stone":        # the frame is cards and so are the timbers round it: close the reveal
            ya_ = -PR
            rv = mesh_object("reveal", [(ua, ya_, za), (ub, ya_, za), (ub, ya_, zb), (ua, ya_, zb), (ua, yg, za), (ub, yg, za), (ub, yg, zb), (ua, yg, zb)],
                             [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            assign(rv, PAINT_F)
            uv_box(rv)
            F.add("win", rv)
        if c:
            assign(c, LEAD)
            uv_box(c)
            F.add("win", c)
        if kind == "stone":
            s = stone_block("sill", (w + 0.26, 0.27, 0.11), loc=(uc, 0.135 - 0.07, za - 0.055), rng=r, chamfer=0.014, detail=0.3)
            l = stone_block("lintel", (w + 0.42, 0.24, 0.25), loc=(uc, 0.12 - 0.035, zb + 0.125), rng=r, chamfer=0.014, detail=0.3)
            for o in (s, l):
                assign(o, STONE)
                F.add("stone", lod_faces(o, drop=BACKF))
        else:
            s = box("sill", (w + 0.16, 0.06, 0.05), loc=(uc, -PR - 0.03 + 0.004, za - 0.03), bevel=bv(0.008), segments=1)
            uv_grain(s, axis="X")
            assign(s, TIMBER)
            tone(s, r)
            lod_faces(s, drop=BACKF)
            lod_faces(s, keep=[OUTF, (0, 0, 1)], above=1)
            F.add("timber", s)
        if shutters:
            lw = w / 2 - 0.012
            for sd in (-1, 1):
                x0 = ua - 0.03 - lw if sd < 0 else ub + 0.03
                nb = 3
                bw = lw / nb
                for i in range(nb):
                    o = box("shb", (bw - 0.005, 0.026, h - r.u(0.0, 0.012)), loc=(x0 + bw * (i + 0.5), -0.03, za + h / 2), bevel=0.004, segments=1,
                            rot=(0, r.u(-0.25, 0.25) * irr, 0))
                    uv_grain(o, axis="Z")
                    assign(o, PAINT_A)
                    lod_faces(o, drop=BACKF)             # folded back on the wall
                    lod_faces(o, keep=OUTF, above=1)
                    F.add("paint", o)
                for fz in (0.17, 0.83):
                    if Lv < 2:
                        o = box("shl", (lw - 0.04, 0.02, 0.085), loc=(x0 + lw / 2, -0.053, za + h * fz), bevel=0.004, segments=1)
                        uv_grain(o, axis="X")
                        assign(o, PAINT_A)
                        F.add("paint", lod_faces(o, drop=BACKF))
                    hx = x0 + lw if sd < 0 else x0
                    st = box("strap", (lw * 0.62, 0.006, 0.032), loc=(hx - sd * (lw * 0.31 - 0.015), -0.066, za + h * fz), bevel=0.0015, segments=1)
                    assign(st, IRON)
                    F.add("iron", lod_faces(st, keep=OUTF))
                    if Lv < 2:
                        pin = cylinder("pin", 0.011, 0.07, loc=(hx + sd * 0.012, -0.05, za + h * fz), verts=lod_pick(8, 4))
                        assign(pin, IRON)
                        F.add("iron", pin)
        if lit:
            lit_spots.append((fkey, storey, F.w(uc + r.u(-0.2, 0.2) * w, 0.42, za + h * 0.8)))

    # ---------------------------------------------------------------- timber framing primitives (shared by storeys and gables)
    class Framer:
        """Frame members in a face's local (u, z) plane. Records every member's outline so the plaster can be cut round it."""

        def __init__(s, F, r, wk=0.0, lod=False):
            s.F, s.r, s.wk, s.lod = F, r, wk, lod
            s.pegs, s.polys = [], []

        def timber(s, u0, u1, a, b, jit=True, grain=None, skew=True, proud=0.0):
            r = s.r
            w_, h_ = u1 - u0, b - a
            if w_ < 0.02 or h_ < 0.02:
                return
            pr = PR + proud + (r.u(-0.005, 0.005) if jit else 0.0)
            vert = h_ > w_
            rot = (0, r.u(-0.35, 0.35) * irr, 0) if (jit and skew and max(w_, h_) < 1.6) else None
            if s.lod and min(w_, h_) < 0.14:          # side / back walls: small members without the 7 mm arris
                o = box("tm", (w_, TD, h_), loc=((u0 + u1) / 2, TD / 2 - pr, (a + b) / 2), rot=rot)
            else:
                o = box("tm", (w_, TD, h_), loc=((u0 + u1) / 2, TD / 2 - pr, (a + b) / 2), bevel=bv(0.007), segments=1, rot=rot)
            uv_grain(o, axis=grain or ("Z" if vert else "X"))
            assign(o, TIMBER)
            tone(o, r, s.wk)
            # the back is in the wall, and so are the two ends of a member framed between others
            lod_faces(o, drop=[BACKF] + (([(0, 0, 1), (0, 0, -1)] if vert else [(1, 0, 0), (-1, 0, 0)]) if jit else []))
            lod_faces(o, keep=OUTF, above=1)         # far: a card in front of the plaster
            s.F.add("timber", o)
            s.polys.append(_rect(u0, u1, a, b))

        def tpoly(s, pts, gdir, proud=0.0):
            r = s.r
            pts = _ccw([(round(x, 4), round(y, 4)) for x, y in pts])
            q = [pts[0]]
            for pt in pts[1:]:
                if abs(pt[0] - q[-1][0]) + abs(pt[1] - q[-1][1]) > 1e-3:
                    q.append(pt)
            if abs(q[0][0] - q[-1][0]) + abs(q[0][1] - q[-1][1]) < 1e-3:
                q.pop()
            if len(q) < 3:
                return
            pr = PR + proud - 0.006 + r.u(-0.003, 0.003)
            o = extrude_profile("br", q, TD, loc=(0, TD - pr, 0), rot=(90, 0, 0), bevel=bv(0.006), segments=1)
            uv_grain(o, axis=(gdir[0], gdir[1], 0.0))
            assign(o, TIMBER)
            tone(o, r, s.wk)
            lod_faces(o, drop=BACKF)
            lod_faces(o, keep=OUTF, above=1)
            s.F.add("timber", o)
            s.polys.append(q)

        def stud(s, uc, a, b, sw=0.115):
            sw = s.r.vary(sw, 0.12)
            s.timber(uc - sw / 2, uc + sw / 2, a, b)
            s.pegs.append((uc, a - 0.05))
            s.pegs.append((uc, b + 0.05))

        def studs(s, u0, u1, a, b, spacing):
            n = int(round((u1 - u0) / spacing)) - 1
            for i in range(1, n + 1):
                s.stud(u0 + (u1 - u0) * i / (n + 1) + s.r.u(-0.02, 0.02) * irr, a, b)
            return n

        def diag(s, u0, u1, a, b, up, tw=0.105):
            du, dz = u1 - u0, b - a
            if du < 0.12 or dz < 0.12:
                return
            c = min(du * 0.6, tw * math.hypot(du, dz) / dz)
            if up:
                pts = [(u0, a), (u0 + c, a), (u1, b), (u1 - c, b)]
                g = (du, dz)
            else:
                pts = [(u1 - c, a), (u1, a), (u0 + c, b), (u0, b)]
                g = (-du, dz)
            s.tpoly(pts, g)

        def brace(s, uc, zc, su, sz, a, b, tw=0.12):
            if a < 0.18 or b < 0.25:
                return
            n = (9, 5, 3)[Lv]
            outer, inner = [], []
            for i in range(n + 1):
                t = (math.pi / 2) * i / n
                for lst, k in ((outer, tw / 2), (inner, -tw / 2)):
                    du = a - (a + k) * math.sin(t)
                    dz = b - (b + k) * math.cos(t)
                    lst.append((uc + su * max(du, -0.01), zc + sz * max(dz, -0.01)))
            s.tpoly(outer + inner[::-1], (su * -a, sz * b))
            s.pegs.append((uc - su * 0.07, zc + sz * (b - 0.06)))
            s.pegs.append((uc + su * (a - 0.06), zc - sz * 0.07))

        @staticmethod
        def subcells(u0, u1, target, sw=0.095):
            n = max(1, int(round((u1 - u0) / target)))
            cw_ = (u1 - u0 - (n - 1) * sw) / n
            cells = []
            for i in range(n):
                a_ = u0 + i * (cw_ + sw)
                cells.append((a_, a_ + cw_))
            return cells

        def pegs_out(s, L, ok=None):
            """Riven oak pegs, a little proud, some missing."""
            r = s.r
            pv = [(0.0125 * math.cos(0.5 + i * 1.5708), 0.0125 * math.sin(0.5 + i * 1.5708), z) for z in (-0.011, 0.011) for i in range(4)]
            if Lv == 0:
                peg = mesh_object("peg", pv, [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)] + [(4, 5, 6, 7)])
            else:                                    # the cap alone (lod 2: none, but every peg is still drawn from r)
                peg = mesh_object("peg", pv[4:], [(0, 1, 2, 3)])
            ms = [trs((u + r.u(-0.008, 0.008), -PR - 0.004 + r.u(-0.003, 0.004), z), (90, 0, r.u(0, 60))) for (u, z) in s.pegs
                  if 0.05 < u < L - 0.05 and r.chance(0.55 if s.lod else 0.92) and (ok is None or ok(u, z))]
            if ms and Lv < 2:
                po = instances_to_mesh("pegs", peg, ms, keep_base=False)
                assign(po, TIMBER)
                paint(po, grime=0.35)
                s.F.add("pegs", po)
            else:
                delete([peg])

    # ---------------------------------------------------------------- lime plaster infill, cut round the timbers
    def infill(F, name, outline, holes, r, step=0.25, gs=0.3, streaks=(), damp=None, rings=(0.028, 0.095), far=None):
        """Plaster panels = constrained Delaunay triangulation of the wall outline minus every timber / opening outline.
        No plaster vertex lies under a timber (so no baked occlusion bleeds into the panels); a ring of vertices just
        outside each timber keeps the contact shadow to a thin line. The surface only undulates by a few mm.
        lod 1: one ring, straight edges left whole, a quarter of the free grid. lod 2: cut round `far` (the openings)
        only - the timbers are cards in front of it."""
        if Lv == 1:
            step, rings = 2.5, (0.06,)
        elif Lv >= 2:
            step, rings, streaks = 2.5, (), ()
            holes = far if far is not None else holes
        out = np.array(_ccw([tuple(q) for q in outline]), dtype=float)
        H = [np.array(_ccw([tuple(q) for q in h]), dtype=float) for h in holes if len(h) >= 3]
        V, E = [], []

        def loop(P):
            i0, n = len(V), len(P)
            V.extend(P.tolist())
            E.extend((i0 + i, i0 + (i + 1) % n) for i in range(n))

        loop(_resample(out, step))
        Hs = [_offset(h, -0.008) for h in H]            # the plaster tucks 8 mm under each timber
        for h, hs_ in zip(H, Hs):
            loop(_resample(hs_, step))
            # two constrained rings round every timber: the baked contact shadow cannot spread past them into the panel
            hr = h[::2] if len(h) > 9 else h           # curved braces: rings on every other arc point
            for rd_ in rings:
                loop(_resample(_offset(hr, rd_), step * 1.5))
        free = []
        (umin, zmin), (umax, zmax) = out.min(0), out.max(0)
        nu, nz = max(1, int((umax - umin) / gs)), max(1, int((zmax - zmin) / gs))
        gu = umin + (np.arange(nu) + 0.5) * (umax - umin) / nu
        gz = zmin + (np.arange(nz) + 0.5) * (zmax - zmin) / nz
        G_ = np.stack(np.meshgrid(gu, gz), -1).reshape(-1, 2)
        nr = np.random.default_rng(int(r.u(0, 1e6)))
        Gj = G_ + nr.uniform(-0.3, 0.3, G_.shape) * gs
        if Lv:                                   # the same grid thinned: nr stays in step, level vertices are full-model vertices
            ii, jj = np.meshgrid(np.arange(nu), np.arange(nz))
            Gj = Gj[(((ii % 2 == 0) & (jj % 2 == 0)) if Lv == 1 else ((ii % 3 == 1) & (jj % 3 == 1))).ravel()]
        free.append(Gj)
        for (us_, zt_, ln, amp, hw) in streaks:
            n = max(3, int(ln / 0.12) + 1) if Lv == 0 else 2        # the streak fades linearly: its two ends carry it
            zz = zt_ - 0.015 - ln * np.arange(n) / (n - 1)
            for du in (-hw, 0.0, hw):
                free.append(np.stack([np.full(n, us_ + du), zz], 1))
        P = np.concatenate(free)
        ok = _pip(P, _offset(out, -0.025))
        for h in H:
            ok &= ~_pip(P, _offset(h, (rings[-1] if rings else 0.0) + 0.03))
        P = P[ok]
        if len(P):
            _, ui = np.unique(np.round(P / 0.02).astype(np.int64), axis=0, return_index=True)
            P = P[np.sort(ui)]
            V.extend(P.tolist())
        res = delaunay_2d_cdt([Vector(v) for v in V], E, [], 0, 1e-4)
        vs = np.array([(v[0], v[1]) for v in res[0]], dtype=float)
        tris = np.array([t for t in res[2] if len(t) == 3], dtype=np.int64)
        if len(tris) == 0:
            return None
        cen = vs[tris].mean(1)
        ok = _pip(cen, out)
        for h in Hs:
            ok &= ~_pip(cen, h)
        tris = tris[ok]
        a, b, c = vs[tris[:, 0]], vs[tris[:, 1]], vs[tris[:, 2]]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        good = np.abs(area) > 1e-8
        tris, area = tris[good], area[good]
        if len(tris) == 0:
            return None
        neg = area < 0
        tris[neg] = tris[neg][:, ::-1]
        used = np.unique(tris)
        remap = -np.ones(len(vs), dtype=np.int64)
        remap[used] = np.arange(len(used))
        vv = vs[used]
        ph = nr.uniform(0, 6.28, 4)
        yy = 0.0022 * np.sin(vv[:, 0] * 2.3 + ph[0]) * np.sin(vv[:, 1] * 1.9 + ph[1]) + 0.0013 * np.sin(vv[:, 0] * 5.1 + vv[:, 1] * 3.7 + ph[2])
        o = mesh_object(name, np.stack([vv[:, 0], yy, vv[:, 1]], 1).tolist(), remap[tris].tolist())
        assign(o, PLASTER)
        uv_box(o)
        smooth(o, 60)
        sl = list(streaks)

        def gfn(co, n_):
            u, z = co[:, 0], co[:, 2]
            g = np.zeros(len(co))
            for (us_, zt_, ln, amp, hw) in sl:
                g = np.maximum(g, amp * np.clip(1.0 - np.abs(u - us_) / hw, 0.0, 1.0) * np.clip((z - (zt_ - ln)) / ln, 0.0, 1.0) * (z <= zt_ + 0.02))
            return g

        def sfn(co, n_):
            if damp is None:
                return np.zeros(len(co))
            zb_, hgt, amt = damp
            t = np.clip(1.0 - (co[:, 2] - zb_) / hgt, 0.0, 1.0)
            return amt * t * t * (0.55 + 0.45 * np.sin(co[:, 0] * 1.7 + ph[3]))

        paint(o, grime=gfn, special=sfn)
        F.add("plaster", o)
        return o

    # ---------------------------------------------------------------- timber frame of one storey face
    def frame_face(F, L, z0, z1, bays, wins, r, sill=True, pat=None, coarse=False):
        fr = Framer(F, r, lod=coarse)
        timber, stud, studs, diag, brace, subcells, pegs = fr.timber, fr.stud, fr.studs, fr.diag, fr.brace, fr.subcells, fr.pegs
        zb = z0 + (0.2 if sill else 0.26)
        zt = z1 - 0.18

        def fill(kind, u0, u1, a, b, post):
            wd, ht = u1 - u0, b - a
            if wd < 0.14 or ht < 0.14:
                return
            if pat == "plain":
                studs(u0, u1, a, b, 0.85)
                if kind == "side" and wd > 0.45 and ht > 0.9:
                    diag(u0, u1, a + ht * 0.3, b, post < 0)
            elif pat == "close_studding":
                studs(u0, u1, a, b, 0.58 if coarse else 0.4)
            elif pat == "square_panels":
                if kind == "lower":
                    studs(u0, u1, a, b, 0.75)
                else:
                    zm = a + ht * 0.5
                    if wd > 0.3:
                        timber(u0, u1, zm - 0.055, zm + 0.055, skew=False)
                    studs(u0, u1, a, zm - 0.055, 0.8)
                    studs(u0, u1, zm + 0.055, b, 0.8)
            elif pat == "herringbone":
                rows = [(a, b)]
                if kind != "lower" and ht > 1.1 and not coarse:
                    zm = a + ht * 0.5
                    timber(u0, u1, zm - 0.055, zm + 0.055, skew=False)
                    rows = [(a, zm - 0.055), (zm + 0.055, b)]
                for ri, (ra, rb) in enumerate(rows):
                    cells = subcells(u0, u1, min(0.8, (rb - ra) * 0.95) if coarse else 0.5)
                    for i, (ca, cb) in enumerate(cells):
                        if i > 0:
                            timber(ca - 0.095, ca, ra, rb)
                        up = (i + ri) % 2 == 0
                        if post > 0:
                            up = not up
                        diag(ca, cb, ra, rb, up)
            elif pat == "st_andrew":
                if kind == "lower":
                    cells = subcells(u0, u1, max(0.6, ht * (1.7 if coarse else 1.15)))
                    for i, (ca, cb) in enumerate(cells):
                        if i > 0:
                            timber(ca - 0.095, ca, a, b)
                        diag(ca, cb, a, b, True, tw=0.09)
                        diag(ca, cb, a, b, False, tw=0.09)
                        pegs.append(((ca + cb) / 2, (a + b) / 2))
                elif kind == "upper":
                    studs(u0, u1, a, b, 0.62)
                else:
                    diag(u0, u1, a + ht * 0.45, b, post < 0)
                    timber(u0, u1, a + ht * 0.45 - 0.1, a + ht * 0.45, skew=False)
            else:   # curved_braces
                if kind == "lower":
                    aa = min(ht * 1.25, wd * 0.42)
                    brace(u0, a, 1, 1, aa, ht - 0.01)
                    brace(u1, a, -1, 1, aa, ht - 0.01)
                    if wd > 1.3:
                        stud((u0 + u1) / 2, a, b)
                elif kind == "upper":
                    aa = min(0.8, wd * 0.4)
                    bb = min(ht - 0.05, max(1.0, aa * 1.5))
                    brace(u0, b, 1, -1, aa, bb)
                    brace(u1, b, -1, -1, aa, bb)
                    if wd > 1.2:
                        stud((u0 + u1) / 2, a, b)
                else:
                    aa = min(wd - 0.02, 0.55)
                    bb = min(ht - 0.05, max(0.8, aa * 2.2))
                    if post < 0:
                        brace(u0, b, 1, -1, aa, bb)
                    else:
                        brace(u1, b, -1, -1, aa, bb)

        if sill:
            timber(0, L, z0, zb, jit=False)
        timber(0, L, zt, z1, jit=False)
        for i in range(len(bays) - 1):
            timber(bays[i][1], bays[i + 1][0], zb, zt, skew=False)
            uc = (bays[i][1] + bays[i + 1][0]) / 2
            pegs.extend([(uc - 0.03, zb - 0.07), (uc + 0.03, zb - 0.12), (uc - 0.03, zt + 0.06), (uc + 0.03, zt + 0.11)])
        zs, rh = z0 + SILL, 0.13
        for (ua, ub), win in zip(bays, wins):
            timber(ua, ub, zs - rh, zs, skew=False)
            pegs.extend([(ua - 0.07, zs - rh / 2), (ub + 0.07, zs - rh / 2)])
            fill("lower", ua, ub, zb, zs - rh, 0)
            if win:
                wa, wb, _, wz1 = win
                sw = 0.11
                timber(wa - sw, wa, zs, zt, skew=False)
                timber(wb, wb + sw, zs, zt, skew=False)
                timber(wa, wb, wz1, wz1 + 0.12, skew=False)
                pegs.extend([(wa - sw / 2, zs - 0.05), (wb + sw / 2, zs - 0.05), (wa - sw / 2, zt + 0.05), (wb + sw / 2, zt + 0.05),
                             (wa - sw / 2, wz1 + 0.06), (wb + sw / 2, wz1 + 0.06)])
                if zt - (wz1 + 0.12) > 0.12 and wb - wa > 0.8:
                    stud((wa + wb) / 2, wz1 + 0.12, zt, sw=0.09)
                fill("side", ua, wa - sw, zs, zt, -1)
                fill("side", wb + sw, ub, zs, zt, 1)
            else:
                fill("upper", ua, ub, zs, zt, 0)
        fr.pegs_out(L)
        return fr

    # ---------------------------------------------------------------- framed gable under the roof
    def gable_face(F, fk, r, pat, lit, show):
        L = F.L
        z0 = ZE
        N = max(8, int(round(L / 0.3)) // 2 * 2)
        us = [L * i / N for i in range(N + 1)]
        zr = []
        for u in us:
            wx, wy, _ = F.w(min(max(u, 0.04), L - 0.04), 0.07, 0.0)
            loc, _n = roof_hit(wx, wy)
            zr.append(max(z0 + 0.06, loc.z - 0.035) if loc is not None else None)
        for i in range(len(zr)):
            if zr[i] is None:
                near = [zr[j] for j in sorted(range(len(zr)), key=lambda j: abs(j - i)) if zr[j] is not None]
                zr[i] = near[0] if near else z0 + 0.2
        rd, tb, sw, ch = 0.2, 0.2, 0.12, 0.16
        ua_ = np.array(us)
        za_ = np.array(zr)

        def top_at(u, m=0.0):                   # underside of the principal rafters
            return float(np.interp(u, ua_, za_)) - rd + m

        fr = Framer(F, r, wk=(0.06 + 0.2 * age) if show else 0.05)
        h = N // 2
        uc, zap = L / 2, zr[h]
        zl0 = z0 + tb
        fr.timber(0, L, z0, zl0, jit=False, proud=0.014)
        tanp = math.tan(math.radians(pitch))
        def thin(q):                            # the rafter line: every sample / every other one / foot, middle, apex
            if Lv == 0:
                return q
            if Lv == 1:
                return q[::2] if len(q) % 2 else q[::2] + [q[-1]]
            return [q[0], q[len(q) // 2], q[-1]]

        left = thin([(us[i], zr[i]) for i in range(h + 1)])
        right = thin([(us[i], zr[i]) for i in range(h, N + 1)])
        fr.tpoly(left + [(u, z - rd) for (u, z) in reversed(left)], (1.0, tanp), proud=0.012)
        fr.tpoly(right + [(u, z - rd) for (u, z) in reversed(right)], (1.0, -tanp), proud=0.012)

        def raked(u, hw, a):                    # a stud cut to the rake of the rafters (or stopped under the collar)
            tl_, tr_ = min(cap, top_at(u - hw, 0.03)), min(cap, top_at(u + hw, 0.03))
            if min(tl_, tr_) - a < 0.22:
                return None
            fr.tpoly([(u - hw, a), (u + hw, a), (u + hw, tr_), (u - hw, tl_)], (0.0, 1.0))
            fr.pegs.append((u, a - 0.05))
            return min(tl_, tr_)

        def span_at(z):
            fine = np.linspace(0, L, 200)
            okk = np.interp(fine, ua_, za_) - rd > z
            return (float(fine[okk].min()), float(fine[okk].max())) if okk.any() else (uc, uc)

        # attic window between queen posts, collar above it
        ww = clamp(L * 0.125, 0.6, 0.95)
        zsill = zl0 + 0.6
        wh = min(1.05, top_at(uc - ww / 2 - sw) - zsill - 0.26)
        has_win = wh >= 0.55
        ops = []
        streaks = []
        if has_win:
            wa, wb, wz1 = uc - ww / 2, uc + ww / 2, zsill + wh
            cap = wz1
            cl, cr = span_at(wz1 + ch * 0.6)
            fr.timber(cl - 0.05, cr + 0.05, wz1, wz1 + ch, skew=False)
            fr.timber(wa - sw, wa, zl0, wz1, skew=False)
            fr.timber(wb, wb + sw, zl0, wz1, skew=False)
            fr.timber(wa, wb, zsill - 0.12, zsill, skew=False)
            fr.pegs.extend([(wa - sw / 2, zl0 - 0.06), (wb + sw / 2, zl0 - 0.06), (wa - sw / 2, wz1 + 0.07), (wb + sw / 2, wz1 + 0.07),
                            (wa - sw / 2, zsill - 0.06), (wb + sw / 2, zsill - 0.06), (cl + 0.1, wz1 + ch / 2), (cr - 0.1, wz1 + ch / 2)])
            ops.append((wa, wb, zsill, wz1))
            add_window(F, fk, S, uc, zsill, ww, wh, lit, "timber", r)
            zk = wz1 + ch
            e_l, e_r = wa - sw, wb + sw
            # under the window
            if pat == "st_andrew":
                fr.diag(wa, wb, zl0, zsill - 0.12, True, tw=0.085)
                fr.diag(wa, wb, zl0, zsill - 0.12, False, tw=0.085)
            elif pat == "herringbone":
                fr.diag(wa, uc - 0.045, zl0, zsill - 0.12, False)
                fr.timber(uc - 0.045, uc + 0.045, zl0, zsill - 0.12)
                fr.diag(uc + 0.045, wb, zl0, zsill - 0.12, True)
            elif pat == "close_studding":
                fr.studs(wa, wb, zl0, zsill - 0.12, 0.34)
            elif ww > 0.7:
                fr.stud(uc, zl0, zsill - 0.12, sw=0.1)
            for uu in (wa + 0.03, wb - 0.03):
                streaks.append((uu, zsill - 0.12, r.u(0.2, 0.4), r.u(0.2, 0.4), 0.03))
        else:
            cap = 1e9
            zk = zl0
            e_l = e_r = uc
            raked(uc, 0.07, zl0)
        # king post and struts above the collar
        if has_win and zap - rd - zk > 0.3:
            fr.timber(uc - 0.07, uc + 0.07, zk, zap - rd + 0.05, skew=False)
            fr.pegs.append((uc, zk - 0.06))
            if zap - rd - zk > 0.95 and show:
                cl2, cr2 = span_at(zk + 0.05)
                for sd in (-1, 1):
                    um = uc + sd * (cr2 - cl2) * 0.27
                    bz = top_at(um, 0.03)
                    if sd < 0:
                        fr.diag(um, uc - 0.07, zk, bz, False)
                    else:
                        fr.diag(uc + 0.07, um, zk, bz, True)
        # studs / braces either side, cut to the rake
        if not show:
            sp, first = 0.95, 0.95
        elif pat == "close_studding":
            sp, first = 0.42, 0.42
        elif pat == "curved_braces":
            sp, first = 0.8, clamp(L * 0.13, 0.75, 1.05)
        elif pat == "herringbone":
            sp, first = 0.56, 0.56
        else:
            sp, first = 0.72, 0.72
        zrail = zsill if has_win else zl0 + 0.72
        for sd in (-1, 1):
            e0 = e_l if sd < 0 else e_r
            prev_u, prev_t = e0, (cap if has_win else top_at(e0))
            u = e0 + sd * first
            i = 0
            if show and pat == "curved_braces" and has_win:
                bb = min(wz1 - zl0 - 0.12, 1.25)
                if top_at(e0 + sd * (first - 0.1)) - zl0 > 0.25:
                    fr.brace(e0, zl0, sd, 1, first - 0.1, bb)
            while 0.2 < u < L - 0.2:
                if show and pat == "square_panels" and has_win and top_at(u + sd * 0.06) > zrail + 0.3:
                    fr.timber(u - 0.05, u + 0.05, zl0, zrail - 0.12)
                    t = raked(u, 0.05, zrail)
                else:
                    t = raked(u, 0.052, zl0)
                if t is None:
                    break
                if show and pat == "herringbone" and min(t, prev_t) - zl0 > 0.4:
                    c0, c1 = (prev_u, u) if sd > 0 else (u, prev_u)
                    fr.diag(c0 + 0.055, c1 - 0.055, zl0, min(t, prev_t) - 0.05, (i % 2 == 0) == (sd > 0))
                if show and pat == "st_andrew" and i == 0 and has_win and min(t, prev_t) > zrail:
                    c0, c1 = (prev_u, u) if sd > 0 else (u, prev_u)
                    fr.diag(c0 + 0.055, c1 - 0.055, zl0, zrail - 0.12, True, tw=0.085)
                    fr.diag(c0 + 0.055, c1 - 0.055, zl0, zrail - 0.12, False, tw=0.085)
                prev_u, prev_t = u, t
                u += sd * sp * r.u(0.94, 1.06)
                i += 1
            if show and pat in ("square_panels", "st_andrew") and has_win:
                # sill-level rail from the queen post out to the rafter
                rl, rr_ = span_at(zrail + 0.02)
                if sd < 0 and e0 - rl > 0.3:
                    fr.timber(rl - 0.04, e0, zrail - 0.12, zrail, skew=False)
                elif sd > 0 and rr_ - e0 > 0.3:
                    fr.timber(e0, rr_ + 0.04, zrail - 0.12, zrail, skew=False)
        outline = [(0.0, z0), (L, z0)] + (left + right[1:])[::-1]
        fr.pegs_out(L, ok=lambda u, z: z < top_at(u, rd - 0.05))
        for _ in range(max(1, int(L / 2.2))):
            uu = r.u(0.8, L - 0.8)
            streaks.append((uu, top_at(uu), r.u(0.3, 0.7), r.u(0.12, 0.28) * (0.6 + 0.8 * age), r.u(0.03, 0.05)))
        infill(F, "gplaster", outline, fr.polys + [_rect(a, b, c, d) for (a, b, c, d) in ops], r, step=0.45 if show else 0.6,
               gs=0.42 if show else 0.55, streaks=streaks, rings=(0.028, 0.095) if (show and not eaves_front) else (0.06,),
               far=[_rect(a, b, c, d) for (a, b, c, d) in ops])

    def plan_bays(L, r, target=2.2, cp=0.21, pw=0.16):
        n = max(1, int(round((L - 2 * cp) / target)))
        clear = (L - 2 * cp - (n - 1) * pw) / n
        offs = [0.0] + [r.u(-0.1, 0.1) * irr for _ in range(n - 1)] + [0.0]
        return [(cp + i * (clear + pw) + offs[i], cp + i * (clear + pw) + clear + offs[i + 1]) for i in range(n)]

    def lit_flags(n, r, force=True):
        """n booleans with about lit_ratio of them on (stochastic rounding, shuffled)."""
        if n <= 0:
            return []
        k = int(p.lit_ratio * n + r.u(0.0, 0.999))
        if force and p.lit_ratio > 0.05:
            k = max(k, 1)
        if p.lit_ratio < 0.95 and n > 1:
            k = min(k, n - 1)
        order = sorted(range(n), key=lambda i: r.u(0, 1))
        on = set(order[:k])
        return [i in on for i in range(n)]

    # ---------------------------------------------------------------- roof tiles (one base tile, stamped)
    TILE = dict(tw=0.235, g=0.16, lv=0.215, hb=0.032) if clay else dict(tw=0.3, g=0.21, lv=0.275, hb=0.017)
    hb_, hh_, zl_ = TILE["hb"], 0.004, 0.003
    if Lv == 0:
        tbase = mesh_object("tile", [(-.5, 0, hb_), (.5, 0, hb_), (.5, 1, hh_), (-.5, 1, hh_), (-.5, 0, zl_), (.5, 0, zl_), (.5, 0, hb_), (-.5, 0, hb_)],
                            [(0, 1, 2, 3), (4, 5, 6, 7)])
    else:                                # a tile is its top face (no butt)
        tbase = mesh_object("tile", [(-.5, 0, hb_), (.5, 0, hb_), (.5, 1, hh_), (-.5, 1, hh_)], [(0, 1, 2, 3)])
    tuv = tbase.data.uv_layers[0] if tbase.data.uv_layers else tbase.data.uv_layers.new(name="UVMap")
    for li, uv_ in enumerate([(0, 0), (TILE["tw"], 0), (TILE["tw"], TILE["lv"]), (0, TILE["lv"]), (0, 0), (TILE["tw"], 0), (TILE["tw"], 0.03), (0, 0.03)][:len(tbase.data.loops)]):
        tuv.data[li].uv = uv_
    uv_keep(tbase)
    assign(tbase, ROOFM)
    tbase["tsq_smooth"] = 20.0

    def tile_masks(o, n, wear, grime, special):
        nv = len(o.data.vertices) // max(n, 1)
        for nm, vals in (("m_wear", wear), ("m_grime", grime), ("m_special", special)):
            at = o.data.attributes.get(nm) or o.data.attributes.new(nm, 'FLOAT', 'POINT')
            at.data.foreach_set("value", np.repeat(np.clip(vals, 0.0, 1.0).astype(np.float32), nv))

    def hood(F, uc, hw, ya, ztop, proj, drop, zfoot, curved, r):
        """Small tiled pentice over a door (face-local coords, -Y outwards): ledger at (ya, ztop), sloping out and down
        to a plate carried by two arch braces that stand on stone corbels bedded in the wall."""
        yb, zbt = ya - proj, ztop - drop
        th = math.atan2(drop, proj)
        c, s = math.cos(th), math.sin(th)
        Lr = math.hypot(proj, drop)
        ym, zm = (ya + yb) / 2, (ztop + zbt) / 2
        tim = []
        for du in ((-hw + 0.07, 0.0, hw - 0.07) if Lv < 2 else ()):
            tim.append(box("hraft", (0.065, Lr, 0.075), loc=(uc + du, ym + s * 0.06, zm - c * 0.06), rot=(math.degrees(th), 0, 0), bevel=bv(0.006), segments=1))
            uv_grain(tim[-1], axis="Y")
        bd = box("hboards", (2 * hw + 0.05, Lr + 0.03, 0.02), loc=(uc, ym + s * 0.01, zm - c * 0.01), rot=(math.degrees(th), 0, 0), bevel=0.004, segments=1)
        uv_grain(bd, axis="X")
        tim.append(bd)
        for sd in (-1, 1):
            vb = box("hverge", (0.026, Lr + 0.06, 0.12), loc=(uc + sd * (hw + 0.04), ym + s * 0.035, zm - c * 0.035), rot=(math.degrees(th), 0, 0), bevel=0.005, segments=1)
            uv_grain(vb, axis="Y")
            tim.append(vb)
        zp = zbt - 0.14
        pl = box("hplate", (2 * hw + 0.02, 0.09, 0.1), loc=(uc, yb + 0.075, zp), bevel=bv(0.008), segments=1)
        uv_grain(pl, axis="X")
        tim.append(pl)
        A, Hh = -(yb + 0.075) + 0.035, (zp - 0.05) - zfoot
        for sd in (-1, 1):
            ub_ = uc + sd * (hw - 0.075)
            if curved:
                n = (10, 5, 3)[Lv]
                lo = [(A - (A - 0.1) * math.cos(math.pi / 2 * i / n), (Hh - 0.1) * math.sin(math.pi / 2 * i / n)) for i in range(n + 1)]
                up = [(A - A * math.cos(math.pi / 2 * i / n), Hh * math.sin(math.pi / 2 * i / n)) for i in range(n, -1, -1)]
                pts = lo + up
            else:
                pts = [(0.0, 0.0), (0.1, 0.0), (A, Hh - 0.11), (A, Hh), (A - 0.11, Hh), (0.0, 0.13)]
            bk = extrude_profile("hbrace", _ccw(pts), 0.09, loc=(ub_ + 0.045, 0, zfoot), rot=(90, 0, -90), bevel=bv(0.008), segments=1)
            uv_grain(bk, axis=(A, Hh, 0.0))
            tim.append(bk)
            cb = extrude_profile("hcorbel", [(-0.08, -0.24), (0.0, -0.24), (0.05, -0.19), (0.1, -0.1), (0.125, -0.03), (0.125, 0.0), (-0.08, 0.0)], 0.17,
                                 loc=(ub_ + 0.085, 0, zfoot), rot=(90, 0, -90), bevel=0.012, segments=2)
            assign(cb, STONE)
            uv_box(cb)
            F.add("corbel", cb)
            for pz in (0.1, Hh - 0.05):
                F.add("iron", assign(bolt("hbolt", 0.011, 0.008, loc=(ub_ + sd * 0.047, -0.05 if pz < 0.2 else -(A - 0.05), zfoot + pz), rot=(0, sd * 90, 0), kind="dome"), IRON))
        for o in tim:
            assign(o, TIMBER)
            tone(o, r, 0.1, g=0.3)
            F.add("timber", o)
        fl = box("hflash", (2 * hw + 0.1, 0.014, 0.11), loc=(uc, ya - 0.009, ztop + 0.035), bevel=0.003, segments=1)
        assign(fl, LEAD)
        F.add("iron", fl)
        # tiles
        tw, g_, lv = TILE["tw"], TILE["g"], TILE["lv"]
        ncol = max(2, int(round((2 * hw + 0.1) / tw)))
        twa = (2 * hw + 0.1) / ncol
        ms, j = [], 0
        while True:
            sj = -0.035 + j * g_
            ly = min(1.0, (Lr - sj) / lv)
            if ly < 0.3:
                break
            segs = [(uc - hw - 0.05 + twa * (i + 0.5), 1.0) for i in range(ncol)] if j % 2 == 0 else \
                [(uc - hw - 0.05 + twa * 0.25, 0.5)] + [(uc - hw - 0.05 + twa * i, 1.0) for i in range(1, ncol)] + [(uc + hw + 0.05 - twa * 0.25, 0.5)]
            for (uu, sx) in segs:
                lift = r.u(0.001, 0.005)
                M = np.identity(4)
                M[:3, 0] = (twa * sx - 0.004, 0, 0)
                M[:3, 1] = (0, c * lv * ly, s * lv * ly)
                M[:3, 2] = (0, -s, c)
                M[:3, 3] = (uu, yb + c * sj - s * lift, zbt + s * sj + c * lift)
                ms.append(M)
            j += 1
        if Lv >= 2:                             # far: one sloping sheet where the tiles lay (it takes their weathering)
            r.u(0, 1e6), r.u(0, 1e6)
            P = lambda sj, lf=0.02: (yb + c * sj - s * lf, zbt + s * sj + c * lf)
            (y0_, z0_), (y1_, z1_) = P(-0.035), P(Lr)
            to = mesh_object("htiles", [(uc - hw - 0.05, y0_, z0_), (uc + hw + 0.05, y0_, z0_), (uc + hw + 0.05, y1_, z1_), (uc - hw - 0.05, y1_, z1_)], [(0, 1, 2, 3)])
            assign(to, ROOFM)
            uv_box(to)
            F.add("tiles", to)
            return
        to = instances_to_mesh("htiles", tbase, ms, uv_jitter=True, seed=int(r.u(0, 1e6)))
        nrr = np.random.default_rng(int(r.u(0, 1e6)))
        tile_masks(to, len(ms), nrr.uniform(0, 0.5, len(ms)), nrr.uniform(0, 0.3, len(ms)) ** 2 * 3, nrr.uniform(0, 1, len(ms)) ** 4 * (0.3 + 0.6 * age))
        F.add("tiles", to)

    # ================================================================ ground floor
    G = faces_of("g", 0.0)
    rg = rng.fork("ground")
    mir = (lambda u: u) if p.door_side == "left" else (lambda u: W - u)
    ud = 1.38                                   # door centre from its corner
    g_open = {k: [] for k in G}
    g_skip = {k: [] for k in G}                 # plinth gaps (u0, u1)
    forb = []                                   # world-x intervals of the ground-floor front where no jetty bracket may land

    def g_window(fk, uc, w, h, lit, r, shut=False, za=1.02):
        g_open[fk].append((uc - w / 2, uc + w / 2, za, za + h))
        add_window(G[fk], fk, 0, uc, za, w, h, lit, "stone", r, shutters=shut)

    def front_window(u, w, lit, shut):
        g_window("front", u, w, 1.3, lit, rg, shut)
        ext = (w + 0.14) if shut else (w / 2 + 0.21 + 0.25)
        forb.append((u - W / 2 - ext, u - W / 2 + ext))

    door_w = 1.3
    lantern_u = None
    sign_u = None
    if not shop:
        door = use("door_plank", width=door_w, height=2.42, head=rg.pick(["round", "round", "flat"]), frame="stone", frame_tone="#b3a68e",
                   finish="painted", tone=p.accent, studs=4, lit=False, grille=True, hinge_side="left" if p.door_side == "left" else "right",
                   wear=0.3 + 0.5 * age, seed=seed + 2)
        xd = mir(ud) - W / 2
        skin(door.place((xd, 0, 0), sockets=False), "door")
        g_skip["front"].append((mir(ud) - door_w / 2 - 0.02, mir(ud) + door_w / 2 + 0.02))
        dhw = clamp(door.size.x / 2, door_w / 2, door_w / 2 + 0.3)     # half width of door + stone surround
        forb.append((xd - dhw - 0.25, xd + dhw + 0.25))
        if p.door_hood:
            hhw = dhw + 0.2
            hood(G["front"], mir(ud), hhw, -(JY[1] + PR + 0.03), H0 + 0.25, 0.6, 0.3, 1.74, True, rg.fork("hood"))
            forb.append((xd - hhw - 0.135, xd + hhw + 0.135))
            socket("sign", "attach", (xd, -(JY[1] + PR + 0.66), H0 - 0.2), role="sign")
        else:
            socket("sign", "attach", (xd, -0.03, 2.72), role="sign")
        lantern_u = ud + door_w / 2 + 0.7
        ws0, ws1 = lantern_u + 0.48, W - 0.72
        seg = ws1 - ws0
        if seg >= 1.05:
            n = max(1, int(seg / 2.4))
            slot = seg / n
            w = min(1.2, slot - 0.4)
            shut = bool(p.shutters) and slot >= 2 * w + 0.2
            fl = lit_flags(n, rg)
            for i in range(n):
                front_window(mir(ws0 + slot * (i + 0.5)), w, fl[i], shut)
    else:
        dw = 1.25
        ud = 1.2
        sdoor = use("door_shop", width=dw, height=2.95, paint=p.accent, sign_text="OPEN", lit=True, wear=0.3 + 0.4 * age,
                    hand="right" if p.door_side == "left" else "left", seed=seed + 2)
        xd = mir(ud) - W / 2
        skin(sdoor.place((xd, 0, 0), sockets=False), "door")
        g_skip["front"].append((mir(ud) - dw / 2 - 0.02, mir(ud) + dw / 2 + 0.02))
        s0, s1 = ud + dw / 2 + 0.25, W - 0.55
        sw = clamp(s1 - s0, 1.6, 3.6)
        su = s0 + sw / 2
        if s1 - (s0 + sw) >= 1.75:              # wide fronts: a plain window fills what the shopfront leaves
            rest0 = s0 + sw + 0.35
            front_window(mir((rest0 + W - 0.6) / 2), min(1.2, W - 0.6 - rest0 - 0.3), rg.chance(p.lit_ratio), False)
        sh = use("shop_window", width=sw, height=3.0, bays=max(2, int(round(sw / 0.85))), paint=p.accent, goods=p.goods, sign_text=p.sign_text,
                 depth=0.42, wear=0.3 + 0.4 * age, seed=seed + 3)
        xs_ = mir(su) - W / 2
        skin(sh.place((xs_, 0, 0), sockets=False), "shop")
        g_skip["front"].append((mir(su) - sw / 2 - 0.02, mir(su) + sw / 2 + 0.02))
        forb.append((xd - dw / 2 - 0.22, xd + dw / 2 + 0.22))
        forb.append((xs_ - sw / 2 - 0.12, xs_ + sw / 2 + 0.12))
        light("shop_spill", (xs_, -0.95, 2.35), intensity=14.0, range=7.5, flicker=0.05)
        light("door_spill", (xd, 0.45, 2.45), intensity=6.0, range=4.5, flicker=0.05)
        sign_u = ud - dw / 2 - 0.3 if ud - dw / 2 > 0.5 else None
        socket("awning", "attach", (xs_, -0.5, 2.45), role="awning")
        if sign_u is not None:
            hs = use("hanging_sign", emblem=p.emblem, text=p.sign_text, board_color=p.accent, accent_color="#d8cfb4", reach=0.7, lamp="none",
                     age=age, seed=seed + 4)
            skin(hs.place((mir(sign_u) - W / 2, 0, 2.85), sockets=False), "sign")
            forb.append((mir(sign_u) - W / 2 - 0.25, mir(sign_u) - W / 2 + 0.25))

    # wall lantern by the door: clear of the door surround, the hood braces and the window, its own fitting casts no shadow
    if lantern_u is not None and p.lantern_cd > 0.5:
        lan = use("wall_lantern", reach=0.54, style=rg.pick(["scroll", "swan"]), carry="stand", finish="forged", gilt=False, drop=0, gas_pipe=False,
                  intensity=float(p.lantern_cd), rust=0.2 + 0.4 * age, seed=seed + 5)
        lx, lz = mir(lantern_u) - W / 2, 1.96
        skin(lan.place((lx, 0, lz), sockets=False), "lantern")
        for s in lan.sockets:
            if s[1] == "light":
                t = s[2].translation
                light("lantern", (lx + t.x, t.y - 0.1, lz + t.z + 0.02), intensity=float(p.lantern_cd), range=9.0, flicker=0.14, shadow_near=1.1)
        forb.append((lx - 0.5, lx + 0.5))

    # side and back ground-floor windows, back door
    for fk in ("right", "left"):
        r = rg.fork(fk)
        L = G[fk].L
        n = max(1, int(round(L / 3.6)))
        fl = lit_flags(n, r)
        for i in range(n):
            if n > 1 and r.chance(0.2):
                continue
            g_window(fk, 0.9 + (L - 1.8) * (i + 0.5) / n + r.u(-0.2, 0.2), r.pick([0.8, 0.95]), 1.15, fl[i], r, za=1.15)
    r = rg.fork("back")
    bd_u = W * 0.28 if r.chance(0.5) else W * 0.72
    bd_w, bd_h, bd_s = 0.95, 2.02, 0.15
    g_open["back"].append((bd_u - bd_w / 2, bd_u + bd_w / 2, 0.0, bd_s + bd_h))
    g_skip["back"].append((bd_u - bd_w / 2 - 0.2, bd_u + bd_w / 2 + 0.2))
    FB = G["back"]
    step = stone_block("bstep", (bd_w + 0.36, 0.5, bd_s), loc=(bd_u, -0.07, bd_s / 2 + 0.014), rng=r, chamfer=0.015, detail=0.3)
    lint = stone_block("blintel", (bd_w + 0.44, 0.24, 0.26), loc=(bd_u, 0.12 - 0.035, bd_s + bd_h + 0.13), rng=r, chamfer=0.014, detail=0.3)
    for o in (step, lint):
        assign(o, STONE)
        FB.add("stone", o)
    for sx in (-1, 1):
        o = box("bjamb", (0.08, 0.1, bd_h), loc=(bd_u + sx * (bd_w / 2 - 0.04), 0.09, bd_s + bd_h / 2), bevel=0.006, segments=1)
        uv_grain(o, axis="Z")
        assign(o, TIMBER)
        FB.add("timber", o)
    o = box("bhead", (bd_w - 0.16, 0.1, 0.08), loc=(bd_u, 0.09, bd_s + bd_h - 0.04), bevel=0.006, segments=1)
    uv_grain(o, axis="X")
    assign(o, TIMBER)
    FB.add("timber", o)
    nb = 5
    bw = (bd_w - 0.16) / nb
    for i in range(nb):
        yb = 0.125 + r.u(-0.002, 0.002)
        # lod 2 keeps the outer faces only: the boards edge to edge and in one plane there, or the open joints
        # between the faces would show the inside of the house
        o = box("bboard", (bw if Lv == 2 else bw - 0.004, 0.035, bd_h - 0.085),
                loc=(bd_u - (bd_w - 0.16) / 2 + bw * (i + 0.5), 0.125 if Lv == 2 else yb, bd_s + (bd_h - 0.08) / 2), bevel=0.005, segments=1)
        uv_grain(o, axis="Z")
        assign(o, PAINT_A)
        FB.add("paint", lod_faces(o, keep=OUTF, above=1))
    for fz in (0.2, 0.8):
        st = box("bstrap", (bd_w * 0.6, 0.006, 0.04), loc=(bd_u - 0.06, 0.104, bd_s + bd_h * fz), bevel=0.0015, segments=1)
        assign(st, IRON)
        FB.add("iron", st)
    if Lv < 2:
        ring = torus("bring", 0.045, 0.007, loc=(bd_u + bd_w * 0.28, 0.098, bd_s + 1.02), rot=(90, 0, 0), major_seg=14, minor_seg=6)
        assign(ring, IRON)
        FB.add("iron", ring)
    # lean-to hood over the back door on straight struts
    hood(FB, bd_u, bd_w / 2 + 0.3, -0.005, min(H0 - 0.1, bd_s + bd_h + 0.72), 0.62, 0.3, bd_s + bd_h - 0.62, False, r.fork("hood"))
    nwb = max(1, int((W - 2.6) / 2.6))
    fl = lit_flags(nwb, r)
    span0, span1 = (bd_u + bd_w / 2 + 0.85, W - 0.8) if bd_u < W / 2 else (0.8, bd_u - bd_w / 2 - 0.85)
    for i in range(nwb):
        slot = (span1 - span0) / nwb
        g_window("back", span0 + slot * (i + 0.5), 0.95, 1.2, fl[i], r, shut=bool(p.shutters) and slot >= 2.15 and i == 0, za=1.1)

    # masonry skins, plinth blocks (each stone weathered on its own), quoins
    for fk, F in G.items():
        r = rg.fork("m" + fk)
        wl = wall_panel("wall", F.L, 0.0, H0, g_open[fk], 0.2, 0.42)
        assign(wl, MASON)
        uv_box(wl)
        displace_noise(wl, strength=0.007 if p.ground == "stone" else 0.003, scale=0.45, seed=seed + len(fk))
        F.add("mason", wl)
        u = -0.05
        while u < F.L + 0.04:
            ln = min(r.u(0.55, 1.0), F.L + 0.05 - u)
            if F.L + 0.05 - (u + ln) < 0.3:
                ln = F.L + 0.05 - u
            a, b = u, u + ln
            for (s0_, s1_) in g_skip[fk]:
                if a < s1_ and b > s0_:
                    if a < s0_:
                        b = s0_
                    else:
                        a = b = max(b, s1_) if a >= s0_ and b <= s1_ else a
                        if a < s1_:
                            a = s1_
            if b - a > 0.12:
                blk = stone_block("plinth", (b - a - 0.008, 0.13, r.jitter(0.43, 0.012)), loc=((a + b) / 2, 0.065 - 0.05 - r.u(0, 0.008), 0.232), rng=r,
                                  chamfer=0.016, detail=0.34)
                assign(blk, STONE)
                uv_box(blk, offset=(r.u(0.0, 9.0), r.u(0.0, 9.0)))
                paint(blk, grime=r.u(0.05, 0.5), special=(r.u(0.25, 0.8) if r.chance(0.3 + 0.3 * age) else r.u(0.0, 0.12)), wear=r.u(0.0, 0.3))
                lod_faces(blk, drop=[BACKF, (0, 0, -1)])
                F.add("plinth", lod_faces(blk, keep=[OUTF, (0, 0, 1)], above=1))
            u += ln
    rq = rg.fork("quoins")
    q_top = H0 - (0.36 if JY[1] > 0 else 0.02)
    qo = []
    for (cx, cy, ox, oy) in ((-W / 2, 0, -1, -1), (W / 2, 0, 1, -1), (W / 2, D, 1, 1), (-W / 2, D, -1, 1)):
        z = 0.44
        i = 0
        while z < q_top - 0.12:
            h = min(rq.u(0.27, 0.33), q_top - z)
            lx_, ly_ = (0.44, 0.25) if i % 2 == 0 else (0.25, 0.44)
            lx_, ly_ = rq.vary(lx_, 0.06), rq.vary(ly_, 0.06)
            e = 0.022
            b = stone_block("quoin", (lx_, ly_, h - 0.008), loc=(cx + ox * (e - lx_ / 2), cy + oy * (e - ly_ / 2), z + h / 2), rng=rq, chamfer=0.014, detail=0.4)
            assign(b, STONE)
            uv_box(b, offset=(rq.u(0.0, 9.0), rq.u(0.0, 9.0)))
            paint(b, grime=rq.u(0.05, 0.35) + 0.25 * clamp(1.0 - z / 1.2, 0.0, 1.0), special=rq.u(0.0, 0.5) * (0.4 + 0.6 * age) * clamp(1.3 - z / 1.5, 0.0, 1.0))
            qo.append(lod_faces(b, keep=[(ox, 0, 0), (0, oy, 0)]))      # the two faces that look out of the corner (and their chamfers)
            z += h
            i += 1
    join(qo, "quoins")

    # ================================================================ timber storeys
    UP = []                                  # (k, faces, z0, z1, yf)
    front_posts = {}
    front_wins = {}
    box_spots = []
    oriel_spot = None
    for k in range(1, S):
        z0 = H0 + (k - 1) * HS
        z1 = z0 + HS
        yf = -JY[k]
        Fs = faces_of(f"s{k}", yf)
        UP.append((k, Fs, z0, z1, yf))
        for fk, F in Fs.items():
            r = rng.fork(f"s{k}{fk}")
            L = F.L
            front = fk == "front"
            bays = plan_bays(L, r, target=2.15 if front else 2.5)
            nb_ = len(bays)
            if front or fk == "back":
                want = [True] * nb_
                if fk == "back" and nb_ >= 3:
                    want[int(r.u(0, nb_ - 0.001))] = False
            else:
                want = [r.chance(0.45) for _ in range(nb_)]
            if not any(want):
                want[int(r.u(0, nb_ - 0.001))] = True
            ob = -1
            if front and k == 1 and oriel:
                ob = min(range(nb_), key=lambda i: abs((bays[i][0] + bays[i][1]) / 2 - L / 2) - (0.01 if (i % 2 == 0) == (p.door_side == "left") else 0.0))
                if bays[ob][1] - bays[ob][0] < 1.25:
                    ob = -1
            fl = lit_flags(sum(want), r)
            wins, ops, li = [], [], 0
            for bi, ((ua, ub), wnt) in enumerate(zip(bays, want)):
                clear = ub - ua
                w = min(1.32 if front else 1.0, clear - 0.62)
                if bi == ob:
                    oriel_spot = ((ua + ub) / 2, min(1.7, clear - 0.3), z0)
                    wins.append(None)
                    li += 1
                    continue
                if not wnt or w < 0.42:
                    wins.append(None)
                    li += 1 if wnt else 0
                    continue
                h = 1.22 if front else 1.1
                uc = (ua + ub) / 2
                wn = (uc - w / 2, uc + w / 2, z0 + SILL, z0 + SILL + h)
                wins.append(wn)
                ops.append(wn)
                add_window(F, fk, k, uc, z0 + SILL, w, h, fl[li], "timber", r)
                li += 1
                if front and k == 1:
                    box_spots.append((uc, w))
            fr = frame_face(F, L, z0, z1, bays, wins, r, sill=not front, pat=p.pattern, coarse=not front)
            holes = fr.polys + [_rect(a, b, c, d) for (a, b, c, d) in ops] + [_rect(-0.05, 0.205, z0 - 0.01, z1 + 0.01), _rect(L - 0.205, L + 0.05, z0 - 0.01, z1 + 0.01)]
            if front:
                holes.append(_rect(-0.05, L + 0.05, z0 - 0.02, z0 + 0.245))
            streaks = []
            for wn in ops:
                for uu in (wn[0] + r.u(0.0, 0.07), wn[1] - r.u(0.0, 0.07)):
                    streaks.append((uu, z0 + SILL - 0.13, r.u(0.22, 0.55), r.u(0.2, 0.38) * (0.6 + 0.8 * age), 0.03))
            for _ in range(int(L / 1.7)):
                streaks.append((r.u(0.4, L - 0.4), z1 - 0.18, r.u(0.3, 0.8), r.u(0.1, 0.26) * (0.6 + 0.8 * age), r.u(0.03, 0.05)))
            infill(F, "plaster", _rect(0.0, L, z0, z1), holes, r, step=0.45 if front else 0.6, gs=0.42 if front else 0.55, streaks=streaks,
                   damp=(z0 + 0.25, 0.55, 0.12 + 0.4 * age), rings=(0.028, 0.095) if front else (0.06,),
                   far=[_rect(a, b, c, d) for (a, b, c, d) in ops])
            if front:
                front_posts[k] = [(bays[i][1] + bays[i + 1][0]) / 2 for i in range(nb_ - 1)]
                front_wins[k] = [(wn[0] - W / 2 - 0.12, wn[1] - W / 2 + 0.12) for wn in ops]
                if ob >= 0:
                    front_wins[k].append((oriel_spot[0] - W / 2 - oriel_spot[1] / 2 - 0.3, oriel_spot[0] - W / 2 + oriel_spot[1] / 2 + 0.3))
        # corner posts (square, a touch prouder than the rails that run into them)
        cps, e = 0.2 + PR + 0.012, PR + 0.012
        rc = rng.fork(f"cp{k}")
        for (cx, cy, ox, oy) in ((-W / 2, yf, -1, -1), (W / 2, yf, 1, -1), (W / 2, D, 1, 1), (-W / 2, D, -1, 1)):
            o = box("cpost", (cps, cps, HS), loc=(cx + ox * (e - cps / 2), cy + oy * (e - cps / 2), z0 + HS / 2), bevel=bv(0.012), segments=1)
            uv_grain(o, axis="Z")
            assign(o, TIMBER)
            tone(o, rc, 0.05)
            lod_faces(o, drop=[(0, 0, 1), (0, 0, -1)])
            lod_faces(o, keep=[(ox, 0, 0), (0, oy, 0)], above=1)
            Fs["front"].add("posts", o)       # world-space objects: front face has theta 0 but an origin shift
            place(o, loc=(o.location.x + W / 2, o.location.y - yf, o.location.z))

    # ---- framed gables under the roof (street / side gables get the full pattern)
    Ftop = UP[-1][1]
    if eaves_front:
        gable_face(Ftop["right"], "right", rng.fork("gable_r"), p.pattern, rr.chance(p.lit_ratio), True)
        gable_face(Ftop["left"], "left", rng.fork("gable_l"), p.pattern, rr.chance(p.lit_ratio * 0.6), True)
    else:
        gable_face(Ftop["front"], "front", rng.fork("gable_f"), p.pattern, rr.chance(max(p.lit_ratio, 0.0)), True)
        gable_face(Ftop["back"], "back", rng.fork("gable_b"), "plain", rr.chance(p.lit_ratio * 0.6), False)

    # ---- jetties (front): girding beam, joist ends, soffit boards, bressummer, knee brackets on bedded corbels
    def snap_x(x, fb, hw=0.075, lim=1.25):
        iv = sorted((a - hw, b + hw) for (a, b) in fb)
        mg = []
        for a, b in iv:
            if mg and a <= mg[-1][1] + 0.03:
                mg[-1][1] = max(mg[-1][1], b)
            else:
                mg.append([a, b])
        hit = [m for m in mg if m[0] < x < m[1]]
        if not hit:
            return x
        best = None
        for c in (hit[0][0] - 0.01, hit[0][1] + 0.01):
            if abs(c) <= W / 2 - 0.12 and abs(c - x) <= lim and (best is None or abs(c - x) < abs(best - x)):
                best = c
        return best

    jr = rng.fork("jetty")
    jet = []
    cor = []
    for (k, Fs, z0, z1, yf) in UP:
        y_low = -JY[k - 1]
        JE = JY[k] - JY[k - 1]
        e = PR + 0.022
        br = box("bressummer", (W + 2 * e, 0.2, 0.26), loc=(0, yf - e + 0.1, z0 + 0.13), bevel=0.014, segments=2)
        densify(br, 0.5)
        uv_grain(br, axis="X")
        tone(br, jr, 0.05, g=0.25)
        jet.append(br)
        if JE > 0.04:
            n = max(4, int(round(W / 0.46)))
            for i in range(n):
                x = -W / 2 + 0.085 + (W - 0.17) * i / (n - 1)
                jw = jr.vary(0.13, 0.08)
                yj = yf - 0.012 - jr.u(0, 0.012)
                jo = box("joist", (jw * lod_pick(1.0, 1.0, 1.5), JE + 0.22, 0.16), loc=(x + jr.u(-0.015, 0.015), yj + (JE + 0.22) / 2, z0 - 0.08), bevel=bv(0.012), segments=1)
                uv_grain(jo, axis="Y")
                tone(jo, jr, 0.0, g=0.3)
                paint(jo, grime=lambda co, n_, yj=yj: np.where(co[:, 1] < yj + 0.02, 0.8, 0.0))      # dark end grain
                lod_faces(jo, drop=[BACKF, (0, 0, 1)])                   # in the wall / under the soffit boards
                lod_faces(jo, keep=OUTF, above=1)                        # far: the end grain of every second joist, half as wide again
                if i % 2 and i != n - 1:
                    lod_drop([jo], above=1)
                jet.append(jo)
            sof = box("soffit", (W, JE, 0.03), loc=(0, yf + JE / 2, z0 - 0.015))
            uv_grain(sof, axis="X")
            paint(sof, grime=0.35)
            jet.append(lod_faces(sof, keep=(0, 0, -1)))
            if k == 1:
                gb = box("girding", (W + 2 * PR, 0.11, 0.2), loc=(0, -PR + 0.055, z0 - 0.26), bevel=bv(0.01), segments=1)
                uv_grain(gb, axis="X")
                tone(gb, jr, 0.0, g=0.25)
                jet.append(lod_faces(gb, drop=BACKF))
            if JE >= 0.14:
                # brackets only where there is solid wall below: corners (quoins) first, then under the bay posts, snapped
                # into the piers between openings (>= 0.25 m clear of lintel ends) or dropped
                fb = forb if k == 1 else front_wins.get(k - 1, [])
                xs = []
                for x in [-W / 2 + 0.13, W / 2 - 0.13] + [u - W / 2 for u in front_posts.get(k, [])]:
                    x2 = snap_x(x, fb)
                    if x2 is not None and all(abs(x2 - q) > 0.5 for q in xs):
                        xs.append(x2)
                hb = 0.58 if k == 1 else 0.46
                ja = JE - 0.015
                zb_ = z0 - 0.16 - hb
                for x in xs:
                    pts = [(0.0, 0.0), (0.05, 0.0)]
                    na = (8, 4, 2)[Lv]
                    for i in range(1, na + 1):
                        t = (math.pi / 2) * i / na
                        pts.append((ja - (ja - 0.05) * math.cos(t), (hb - 0.07) * math.sin(t)))
                    pts += [(ja, hb), (0.0, hb)]
                    th = 0.11
                    bk = extrude_profile("bracket", pts, th, loc=(x + th / 2, y_low, zb_), rot=(90, 0, -90), bevel=bv(0.008), segments=1)
                    uv_grain(bk, axis=(ja, hb, 0.0))
                    tone(bk, jr, 0.0, g=0.3)
                    jet.append(bk)
                    if k == 1:
                        cb = extrude_profile("corbel", [(-0.08, -0.25), (0.0, -0.25), (0.05, -0.2), (0.1, -0.1), (0.125, -0.03), (0.125, 0.0), (-0.08, 0.0)], 0.2,
                                             loc=(x + 0.1, y_low, zb_), rot=(90, 0, -90), bevel=0.012, segments=2)
                        assign(cb, STONE)
                        uv_box(cb)
                        paint(cb, grime=jr.u(0.1, 0.35))
                        cor.append(cb)
    for o in jet:
        assign(o, TIMBER)
    join(jet, "jetty")
    if cor:
        join(cor, "corbels")

    # ================================================================ chimneys, dormers
    rs = {"chimney": [], "downpipe": []}
    for s in roof.sockets:
        for key in rs:
            if s[0].startswith(key):
                M = Mroof @ s[2]
                out = -Vector((M[0][1], M[1][1], 0.0))
                if out.length > 1e-6:
                    out.normalize()
                rs[key].append((M.translation.copy(), out, math.degrees(math.atan2(M[1][0], M[0][0]))))
    cover_objs = []
    chim_pos = []
    nc = int(p.chimneys)
    if nc > 0 and rs["chimney"]:
        ch = sorted(rs["chimney"], key=lambda c: (-c[0].y, c[0].x * (1 if p.door_side == "left" else -1)))
        chosen = [ch[0]]
        if nc > 1 and len(ch) > 1:
            # eaves-front roofs keep both stacks on the back slope so the street slope is free for dormers
            pool = [c for c in ch[1:] if c[0].y > yc] if eaves_front else []
            chosen.append(max(pool or ch[1:], key=lambda c: (c[0] - ch[0][0]).length))
        for i, (pos, out, ang) in enumerate(chosen):
            sink = 0.6
            cm = use("chimney", width=rr.pick([0.8, 0.95]), depth=0.56, height=rr.u(2.5, 2.9), material="brick_red", pots=rr.pick([1, 2, 2]) if i == 0 else 1,
                     dentils=(i == 0), strap=(i == 0), pitch=pitch, sink=sink, soot=0.4 + 0.5 * age, moss=0.2 + 0.5 * age, lean=rr.u(-1.5, 1.5) * irr, smoke=True, seed=seed + 30 + i)
            cover_objs.append(skin(cm.place((pos.x, pos.y, pos.z - sink), rot=(0, 0, ang), sockets=False), "chimney"))
            chim_pos.append((pos.x, pos.y))
            if i == 0:
                for s in cm.sockets:
                    if s[1] == "fx":
                        t = s[2].translation
                        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                        fx("smoke", "smoke", (pos.x + t.x * ca - t.y * sa, pos.y + t.x * sa + t.y * ca, pos.z - sink + t.z), size=0.5)

    nd = int(p.dormers)
    dormer_light = None
    nbx_want = min(int(p.window_boxes), len(box_spots))
    ia, ir = (0, 1) if not eaves_front else (1, 0)           # slope axis, ridge axis
    rmin, rmax, c_s, E_s = lod_memo("covering", lambda: (float(cov[:, ir].min()), float(cov[:, ir].max()),
                                                         float(cov[:, ia].min() + cov[:, ia].max()) / 2, float(cov[:, ia].max() - cov[:, ia].min()) / 2))
    cosp = math.cos(math.radians(pitch))
    reserve = int(8.2 * (E_s / cosp / TILE["g"]) * ((rmax - rmin) / TILE["tw"])) + 5200 + (8800 if oriel_spot is not None else 0)
    if nd > 0:
        inset = clamp(0.135 * span, 0.68, 1.05)
        spots = []
        if eaves_front:
            nf = min(nd, max(1, int((W - 1.2) / 2.7)))
            for i in range(nf):
                spots.append((-W / 2 + W * (i + 0.5) / nf + rr.u(-0.12, 0.12), -J + inset, nf % 2 == 1 and i == nf // 2 and W >= 7.5))
            spots.sort(key=lambda sp_: abs(sp_[0]))
            if nd > nf:
                spots.append((W * rr.pick([-0.2, 0.2]), D - inset, False))
        else:
            y1 = -J + clamp(0.2 * PDp, 1.45, 2.0)
            fs = 1 if p.door_side == "left" else -1
            order = [(fs, y1), (-fs, y1 + rr.u(-0.15, 0.3)), (fs, y1 + 3.1), (-fs, y1 + 3.3)]
            for (sd, y) in order[:nd]:
                if y <= D - 1.5:
                    spots.append((sd * (W / 2 - inset), y, False))

        def clash(x, y):
            for (cx, cy) in chim_pos:
                same = ((y - yc) * (cy - yc) > 0) if eaves_front else (x * cx > 0)
                d = (x - cx) if eaves_front else (y - cy)
                if same and abs(d) < 1.5:
                    return d
            return None

        for i, (x, y, big) in enumerate(spots):
            # the triangle cap is a decision of the full model: the levels repeat it instead of counting themselves
            if lod_memo(("room_dormer", i), lambda: scene_tris() + (7600 if big else 6300) + reserve + 5300 * (min(nbx_want, 1) if i < 2 else nbx_want) > TRI_CAP):
                break
            d = clash(x, y)
            if d is not None:
                sh_ = (1.55 - abs(d)) * (1 if d >= 0 else -1)
                if eaves_front:
                    x += sh_
                else:
                    y += sh_
                if clash(x, y) is not None or abs(x) > W / 2 - 0.5 or not (-J + 1.0 < y < D - 1.0):
                    continue
            loc, nor = roof_hit(x, y)
            if loc is None:
                continue
            hmax = (ridge_z - loc.z - 0.3) / 1.25
            if hmax < 0.95:
                continue
            dh = min(1.9, hmax)
            dw = clamp(dh * 0.74, 0.8, 1.4)
            if big:
                dw = clamp(dh * 1.15, 1.4, 2.15)
                dh = min(2.3, max(dh, (ridge_z - loc.z - 0.3) - dw * 0.42))
            lit_d = (i < 2 or rr.chance(p.lit_ratio)) and p.lit_ratio > 0.05
            dm = use("dormer", width=dw, height=dh, pitch=pitch, roof="gable", cover="tile" if clay else "slate",
                     cladding="plaster", lit=lit_d, intensity=0.0, flower_box=False,
                     timber=p.timber, paint=p.timber, plaster=p.plaster, age=age, seed=seed + 20 + i)
            out = Vector((nor.x, nor.y, 0.0))
            out.normalize()
            ang = math.degrees(math.atan2(out.y, out.x)) + 90.0
            cover_objs.append(skin(dm.place((loc.x, loc.y, loc.z), rot=(0, 0, ang), sockets=False), "dormer", wood_painted=TIMBER))
            if lit_d and dormer_light is None:
                # window spill onto the tiles below the dormer: well clear of the dormer face, half a metre off the slope
                l2, n2 = roof_hit(loc.x + out.x * 0.8, loc.y + out.y * 0.8)
                dormer_light = tuple(l2 + n2 * 0.8) if l2 is not None else (loc.x + out.x * 0.5, loc.y + out.y * 0.5, loc.z + 0.42)

    # ================================================================ roof covering laid tile by tile on the module's slopes
    bpy.context.view_layer.update()
    cv, cp_, off = [], [], 0
    for ob_ in cover_objs:
        co_, po_ = world_mesh(ob_)
        cv.append(co_)
        cp_.extend(tuple(i + off for i in q) for q in po_)
        off += len(co_)
    cbvh = BVHTree.FromPolygons([Vector(v) for v in np.concatenate(cv)], cp_) if cv else None
    UPV = Vector((0, 0, 1))
    tw, g_, lv = TILE["tw"], TILE["g"], TILE["lv"]
    gx = g_ * cosp
    ncol = max(1, int(round((rmax - rmin) / tw)))
    twa = (rmax - rmin) / ncol
    tr_ = rng.fork("tiles")
    nrt = np.random.default_rng(seed * 7 + 3)
    def lay_tiles():
        ms, tw_, tg_, tsp_ = [], [], [], []
        ph1, ph2, ph3 = tr_.u(0, 6.28), tr_.u(0, 6.28), tr_.u(0, 6.28)
        for sd in (-1, 1):
            j = 0
            while True:
                hd = E_s - 0.012 - j * gx          # horizontal distance to the ridge line
                if hd < 0.06:
                    break
                ly = min(1.0, hd / (lv * cosp))
                if ly < 0.3:
                    break
                if j % 2 == 0:
                    segs = [(rmin + twa * (i + 0.5), 1.0) for i in range(ncol)]
                else:
                    segs = [(rmin + twa * 0.25, 0.5)] + [(rmin + twa * i, 1.0) for i in range(1, ncol)] + [(rmax - twa * 0.25, 0.5)]
                for (rc_, sx) in segs:
                    q = [0.0, 0.0]
                    q[ia], q[ir] = c_s + sd * hd, rc_
                    loc, nor = roof_hit(q[0], q[1])
                    if loc is None:
                        continue
                    yv = UPV - nor * nor.z
                    if yv.length < 1e-4:
                        continue
                    yv.normalize()
                    if cbvh is not None:
                        c_ = loc + yv * (lv * 0.5) + nor * 0.07
                        if cbvh.ray_cast(c_, UPV, 5.0)[0] is not None or cbvh.ray_cast(c_ + yv * 0.02, -UPV, 0.2)[0] is not None:
                            continue
                    xv = yv.cross(nor)
                    yaw = math.radians(tr_.u(-1.3, 1.3) * irr)
                    cy_, sy_ = math.cos(yaw), math.sin(yaw)
                    xv, yv = xv * cy_ + yv * sy_, yv * cy_ - xv * sy_
                    o_ = loc + nor * tr_.u(0.0, 0.005)
                    if tr_.chance(0.012 + 0.03 * age):
                        o_ = o_ - yv * tr_.u(0.015, 0.05) + nor * 0.004      # a slipped tile
                    M = np.identity(4)
                    M[:3, 0] = xv * (twa * sx - 0.004)
                    M[:3, 1] = yv * (lv * ly)
                    M[:3, 2] = nor
                    M[:3, 3] = o_
                    ms.append(M)
                    a_ = 0.5 + 0.5 * math.sin(rc_ * 0.83 + ph1) * math.sin(hd * 1.21 + ph2)
                    eav = clamp(1.0 - (E_s - hd) / 1.6, 0.0, 1.0)
                    tw_.append(0.1 + 0.3 * a_)
                    tg_.append(0.12 * (1 - a_) + 0.25 * eav * eav)
                    tsp_.append((0.25 + 0.6 * age) * (0.25 + 0.75 * eav) * (0.5 + 0.5 * math.sin(rc_ * 0.5 + hd * 0.7 + ph3)) * (1.0 if sd < 0 else 0.55))
                j += 1
        return ms, tw_, tg_, tsp_

    # laid once, on the full model: every level stamps the same tiles in the same places (what differs is the tile)
    ms, tw_, tg_, tsp_ = lod_memo("roof_tiles", lay_tiles)
    if ms and Lv == 1:
        # mid: two neighbours of a course share one top face (courses, slipped tiles and the gaps round dormers stay;
        # the weathering of both comes across from the full model)
        ms1, wide, i = [], [], 0
        while i < len(ms):
            A = np.asarray(ms[i], dtype=float)
            B = np.asarray(ms[i + 1], dtype=float) if i + 1 < len(ms) else None
            if B is not None:
                d = B[:3, 3] - A[:3, 3]
                dl, wa, wb, la = float(np.linalg.norm(d)), float(np.linalg.norm(A[:3, 0])), float(np.linalg.norm(B[:3, 0])), float(np.linalg.norm(A[:3, 1]))
                if not (abs(wa - wb) < 0.01 and 0.8 * wa < dl < 1.25 * wa and float(A[:3, 2] @ B[:3, 2]) > 0.985 and abs(float(d @ A[:3, 1])) < 0.07 * la):
                    B = None
            if B is None:
                ms1.append(A)
                wide.append(1.0)
                i += 1
                continue
            xd = d / dl * (1.0 if float(d @ A[:3, 0]) > 0 else -1.0)
            M = np.identity(4)
            M[:3, 0] = xd * (dl + (wa + wb) / 2)
            M[:3, 1] = (A[:3, 1] + B[:3, 1]) / 2
            M[:3, 2] = (A[:3, 2] + B[:3, 2]) / float(np.linalg.norm(A[:3, 2] + B[:3, 2]))
            M[:3, 3] = (A[:3, 3] + B[:3, 3]) / 2
            ms1.append(M)
            wide.append(2.0)
            i += 2
        tiles = instances_to_mesh("roof_tiles", tbase, ms1, uv_jitter=True, seed=seed + 77)
        uvd = tiles.data.uv_layers[0].data
        uv = np.empty(len(uvd) * 2, dtype=np.float32)
        uvd.foreach_get("uv", uv)
        uv = uv.reshape(len(ms1), -1, 2)
        u0 = uv[:, :, 0].min(1, keepdims=True)
        uv[:, :, 0] = u0 + (uv[:, :, 0] - u0) * np.array(wide, dtype=np.float32)[:, None]      # the material keeps its scale
        uvd.foreach_set("uv", uv.ravel())
    elif ms and Lv < 2:
        tiles = instances_to_mesh("roof_tiles", tbase, ms, uv_jitter=True, seed=seed + 77)
        k_ = len(ms)
        tile_masks(tiles, k_, np.array(tw_) + nrt.uniform(0, 0.45, k_) ** 1.5, np.array(tg_) + nrt.uniform(0, 0.42, k_) ** 3,
                   np.array(tsp_) * nrt.uniform(0, 1, k_) ** 2.5)
    paint(ro, grime=lambda co, n_: np.where(n_[:, 2] > 0.3, 0.45, 0.0))

    # ================================================================ downpipes (follow the jetties with swan-necks)
    def downpipe(F, u, d_out, o_of, r, tag):
        rp, so = 0.04, 0.078
        pts = [(d_out, ZE - 0.02), (d_out, ZE - 0.14)]
        runs = []

        def bend(p0, p1, n=7):
            dz = p0[1] - p1[1]
            for i in range(1, n + 1):
                t = i / n
                c0, c1, c2, c3 = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
                pts.append((c0 * p0[0] + c1 * p0[0] + c2 * p1[0] + c3 * p1[0],
                            c0 * p0[1] + c1 * (p0[1] - 0.5 * dz) + c2 * (p1[1] + 0.5 * dz) + c3 * p1[1]))

        o = o_of(S - 1) + so
        bend(pts[-1], (o, ZE - 0.58))
        zh = ZE - 0.82                          # hopper base
        pts.append((o, zh + 0.1))
        ztop = zh
        for k in range(S - 1, 0, -1):
            zf = H0 + (k - 1) * HS
            o_low = o_of(k - 1) + so
            if o - o_low > 0.05:
                zb_ = zf - 0.2
                pts.append((o, zb_))
                runs.append((o, ztop, zb_))
                z2 = zb_ - max(0.38, 1.5 * (o - o_low))
                bend((o, zb_), (o_low, z2))
                o, ztop = o_low, z2
        pts.append((o, 0.82))
        runs.append((o, ztop, 0.82))
        bend((o, 0.82), (o + 0.07, 0.56), 5)
        pts.append((o + 0.07, 0.2))
        runs.append((o + 0.07, 0.56, 0.2))
        pts.append((o + 0.19, 0.085))
        path = [(u, -a, z) for (a, z) in pts]
        pipe = sweep("pipe", path, rp, sides=lod_pick(8, 6, 3), smooth_path=False)
        F.add("iron", pipe)
        o_top = o_of(S - 1) + so
        if Lv < 2:
            hop = lathe("hopper", [(0.05, 0.0), (0.075, 0.025), (0.105, 0.16), (0.125, 0.18), (0.125, 0.235), (0.11, 0.24), (0.1, 0.2), (0.0, 0.2)], segments=4,
                        loc=(u, -o_top, zh), rot=(0, 0, 45), smooth_angle=20)
            F.add("iron", hop)
        for (oo, za_, zb_) in (runs if Lv < 2 else ()):         # far: no brackets
            ln = za_ - zb_
            n = max(1, int(round(ln / 1.5)))
            for i in range(n):
                z = zb_ + ln * (i + 0.5) / n
                F.add("iron", ring_band("collar", radius=rp + 0.004, width=0.05, thick=0.008, loc=(u, -oo, z), segments=lod_pick(10, 6)))
                F.add("iron", box("ear", (0.2, 0.012, 0.045), loc=(u, -(oo - so) - 0.006, z), bevel=0.002, segments=1))
                F.add("iron", box("stub", (0.035, so - rp, 0.035), loc=(u, -(oo - so) - (so - rp) / 2, z)))
                for sx in (-1, 1):
                    F.add("iron", bolt("ebolt", 0.008, 0.006, loc=(u + sx * 0.075, -(oo - so) - 0.012, z), rot=(90, 0, 0), kind="dome"))
        for ob in F.g["iron"]:
            if ob.name.startswith(("pipe", "hopper", "collar", "ear", "stub", "ebolt")) and not ob.data.materials:
                assign(ob, IRON)
        gl = stone_block("gully", (0.34, 0.3, 0.075), loc=(u, -(runs[-1][0] + 0.17), 0.052), rng=r, chamfer=0.015, detail=0.3)
        assign(gl, STONE)
        F.add("stone", gl)

    for i, (pos, out, ang) in enumerate(rs["downpipe"][:2]):
        r = rng.fork(f"pipe{i}")
        if pos.y < -J - 0.05:
            fk, u, d_out = "front", pos.x + W / 2, -pos.y
            o_of = lambda k: (JY[k] + PR) if k > 0 else 0.0
        elif pos.y > D + 0.05:
            fk, u, d_out = "back", W / 2 - pos.x, pos.y - D
            o_of = lambda k: PR
        elif pos.x > 0:
            fk, u, d_out = "right", pos.y, pos.x - W / 2
            o_of = lambda k: PR
        else:
            fk, u, d_out = "left", D - pos.y, -pos.x - W / 2
            o_of = lambda k: PR
        u = clamp(u, 0.45, G[fk].L - 0.45)
        downpipe(G[fk], u, d_out, o_of, r, f"pipe{i}")

    # ================================================================ oriel, window boxes
    if oriel_spot is not None:
        ou_, ow_, oz0 = oriel_spot
        orl = use("window_oriel", width=clamp(ow_, 1.2, 1.7), height=1.1, depth=0.32, plan="canted", roof="lead", roof_rise=0.35, apron=0.15, lit=True, intensity=0.0,
                  stained=False, finish="oak", wear=0.3 + 0.4 * age, seed=seed + 50)
        skin(orl.place((ou_ - W / 2, -JY[1] - PR, oz0 + 0.85), sockets=False), "oriel", wood_painted=TIMBER, glass_clear=LIT)
        lit_spots.append(("front", 1, (ou_ - W / 2, -JY[1] + 0.3, oz0 + SILL + 1.0)))

    nbx = min(int(p.window_boxes), len(box_spots))
    if nbx > 0:
        order = sorted(range(len(box_spots)), key=lambda i: abs(i - (len(box_spots) - 1) / 2.0 + 0.01))
        chosen = sorted(order[:nbx]) if nbx < len(box_spots) else list(range(len(box_spots)))
        for i in chosen:
            if lod_memo(("room_box", i), lambda: scene_tris() + 5200 > TRI_CAP):
                break
            uc, w = box_spots[i]
            wb = use("window_box", width=min(1.08, w + 0.06), style="plank", finish="painted", paint=p.accent, flower_a="#c22a34", flower_b="#f0e6d8",
                     fullness=0.5, trailing=0.3, blooms=0.75, seed=seed + 40 + i)
            skin(wb.place((uc - W / 2, -JY[1] - PR - 0.065, H0 + SILL - 0.06), sockets=False), "wbox")

    # ================================================================ finish faces, weathering
    delete([tbase])
    for fk, F in G.items():
        out = F.finish()
        if "mason" in out:
            paint(out["mason"], grime=mask_mul(height_ramp(0.0, 1.1), 0.5 + 0.4 * age),
                  special=mask_mul(height_ramp(0.0, 0.9), noise_mask(0.5, seed=seed + 3, lo=0.35, hi=0.7), 0.4 + 0.5 * age))
        if "stone" in out:
            paint(out["stone"], grime=mask_mul(height_ramp(0.0, 1.0), 0.5),
                  special=mask_mul(height_ramp(0.0, 0.7), noise_mask(0.35, seed=seed + 12), 0.5 + 0.4 * age))
        if "iron" in out:
            paint(out["iron"], special=mask_mul(noise_mask(0.25, seed=seed + 14, lo=0.4, hi=0.75), 0.3 + 0.5 * age), grime=height_ramp(0.0, 0.8))
        if "paint" in out:
            paint(out["paint"], wear=mask_mul(noise_mask(0.2, seed=seed + 15, lo=0.55, hi=0.8), 0.25 + 0.4 * age), special=0.2 + 0.3 * age)
    for (k, Fs, z0, z1, yf) in UP:
        for fk, F in Fs.items():
            F.finish()

    # ================================================================ lights (<= 6) and sockets
    used = 2 if shop else (1 if lantern_u is not None and p.lantern_cd > 0.5 else 0)
    order = [("front", 0, 9.0, 5.5), ("front", 1, 8.0, 5.5), ("dormer", None, 3.0, 4.5), ("back", None, 8.0, 5.5), ("right", None, 7.0, 5.0),
             ("left", None, 7.0, 5.0), ("front", 2, 7.0, 5.0)]
    wi = 0
    for (fk, st, cd, rg_) in order:
        if used >= 6:
            break
        if fk == "dormer":
            if dormer_light is not None:
                light("dormer", dormer_light, color="#ffc37a", intensity=cd, range=rg_, flicker=0.04)
                used += 1
            continue
        c = [s for s in lit_spots if s[0] == fk and (st is None or s[1] == st)]
        if not c:
            continue
        c.sort(key=lambda s: (s[1], abs(s[2][0]) + abs(s[2][1] - D / 2)))
        light(f"window_{fk}{wi}", c[0][2], color=["#ffc37a", "#ffb060", "#ffd09a"][wi % 3], intensity=cd, range=rg_, flicker=0.04)
        used += 1
        wi += 1
    socket("lantern_corner", "attach", ((W / 2 - 0.3) * (1 if p.door_side == "left" else -1), -0.02, 2.3), role="lantern")
    socket("bench", "attach", (mir(W * 0.62) - W / 2, -0.45, 0.0), role="seat")

    # ================================================================ age: the upper storeys lean, the floors sag, one side has settled
    wr = rng.fork("warp")
    zw = H0 - 0.35
    kx = (1 if wr.chance(0.5) else -1) * (0.004 + 0.03 * age) * wr.u(0.65, 1.0)
    sagz = 0.055 * age * wr.u(0.7, 1.0)
    rake = 0.04 * age * wr.u(-1.0, 1.0)

    def warp(co):
        t = np.clip(co[:, 2] - zw, 0.0, None)
        rmp = np.clip(t / 0.6, 0.0, 1.0)
        rmp = rmp * rmp * (3 - 2 * rmp)
        xn = np.clip(co[:, 0] / (W / 2), -1.15, 1.15)
        out = co.copy()
        out[:, 0] += kx * t
        out[:, 2] -= rmp * (sagz * np.clip(1.0 - xn * xn, 0.0, 1.0) + rake * 0.5 * (1.0 + xn) * np.sign(rake) * np.sign(rake))
        return out

    bpy.context.view_layer.update()
    for o in list(bpy.context.scene.objects):
        if o.parent is not None:
            continue
        if o.type == 'MESH':
            if len(o.data.vertices) == 0:
                continue
            realize(o)
            apply_transform(o)
            me = o.data
            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3).astype(np.float64)
            if co[:, 2].max() <= zw:
                continue
            me.vertices.foreach_set("co", warp(co).astype(np.float32).ravel())
            me.update()
        elif o.type == 'EMPTY':
            l = np.array([list(o.location)], dtype=np.float64)
            if l[0, 2] > zw:
                o.location = tuple(float(v) for v in warp(l)[0])
