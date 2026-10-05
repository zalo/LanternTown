"""Grand hotel on the square: a symmetrical 3-4 storey block. Rusticated dressed-stone ground storey built
from individual chamfered blocks with voussoired round arches; stucco upper storeys (a welded regular grid,
rain-streaked under sills, hoods and the cornice) with stone quoins, pilasters, string course and modillion
cornice; sash / French windows in pedimented stone surrounds opening into real recessed rooms (glowing walls,
drawn or tied-back curtains, lamps, furniture silhouettes); wrought-iron balconies on stone consoles; panelled
double door with fanlight under a glazed iron canopy with marquee bulbs and a pendant lantern; a tall neon
blade sign with a bulb-outlined crown on scrolled brackets at one corner; a mansard (or steep hipped) roof
laid slate by slate (or tile by tile) in courses with lead hip rolls, aprons and deck, stone dormers, iron
cresting and chimney stacks. All four sides are closed: architraved windows, a service door and downpipes.
"""
from tk import *
import bpy
import bmesh
import numpy as np
from math import sin, cos, tan, asin, atan, atan2, radians, degrees, pi, sqrt, hypot, ceil
from mathutils import Matrix, Vector

META = {"name": "Hotel", "class": "building", "mount": "ground", "focus": [0.0, -0.4, 3.5]}
PARAMS = {
    "width": {"default": 16.0, "min": 12.0, "max": 20.0, "desc": "front width (m); walls span x in [-width/2, +width/2]"},
    "depth": {"default": 10.0, "min": 8.0, "max": 12.0, "desc": "body depth (m); body from y = 0 to y = +depth"},
    "storeys": {"default": 3, "min": 3, "max": 4, "desc": "storeys below the roof (grand ground storey + 2 or 3)"},
    "tint": {"default": "#d3a45c", "desc": "facade stucco colour"},
    "stone": {"default": "#cfc9ba", "desc": "dressed stone colour (ground storey, surrounds, cornices)"},
    "joinery": {"default": "#e8e1cd", "desc": "paint colour of sashes and doors"},
    "iron": {"default": "#1b2521", "desc": "paint colour of balconies, canopy, sign case, pipes"},
    "roof": {"default": "mansard", "choices": ["mansard", "steep"], "desc": "mansard with flat top or steep hipped roof"},
    "roof_cover": {"default": "slate", "choices": ["slate", "tile"], "desc": "roof covering: large blue slates or small red clay tiles"},
    "balconies": {"default": "all", "choices": ["all", "centre", "none"], "desc": "first-floor iron balconies"},
    "canopy_reach": {"default": 1.5, "min": 1.0, "max": 1.52, "desc": "projection of the glazed entrance canopy (m)"},
    "canopy_text": {"default": "GRAND", "desc": "neon lettering standing on the canopy (empty = none)"},
    "sign_text": {"default": "HOTEL", "desc": "letters of the vertical neon sign, stacked top to bottom"},
    "sign_side": {"default": "left", "choices": ["left", "right", "none"], "desc": "corner carrying the vertical neon sign"},
    "neon": {"default": "#2fd0ff", "desc": "neon tube colour"},
    "room_light": {"default": "#ffbe6e", "desc": "colour of the lit room walls"},
    "curtains": {"default": "#ffa868", "desc": "colour of the back-lit curtains in lit rooms"},
    "lit_ratio": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "share of upper windows with the lights on"},
    "chimneys": {"default": 4, "min": 2, "max": 6, "desc": "number of chimney stacks"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: rain streaks, damp, lichen, rust"},
    "sign_color": {"default": "#173a9c", "desc": "back-lit enamel face of the blade sign (gold letters and crown stand on it)"},
    "landing_door": {"default": True, "desc": "first-floor door in the back (canal) wall for an outside stair landing"},
    "landing_door_x": {"default": 3.5, "min": -9.0, "max": 9.0,
                       "desc": "wanted x of that door (m, hotel frame); it takes the nearest pier between two back windows"},
    "landing_door_z": {"default": 3.0, "min": 2.6, "max": 3.4, "desc": "sill of that door = level of the stair landing (m)"},
    "landing_lantern_cd": {"default": 5.0, "min": 0.0, "max": 20.0, "desc": "candela of the lantern over the landing door (0 = glowing fitting, no light socket)"},
}
VARIANTS = {
    "grand_four": {"width": 20.0, "depth": 12.0, "storeys": 4, "tint": "#bd8c7c", "stone": "#d2ccc0", "sign_side": "right",
                   "canopy_text": "SPLENDIDE", "neon": "#ff3f7d", "chimneys": 6, "lit_ratio": 0.7, "seed": 11},
    "narrow_steep": {"width": 12.0, "depth": 8.0, "storeys": 3, "roof": "steep", "roof_cover": "tile", "tint": "#d9b468",
                     "joinery": "#2d4a3b", "balconies": "centre", "canopy_text": "", "sign_text": "INN", "neon": "#ffc93a",
                     "curtains": "#ff8f5a", "chimneys": 2, "lit_ratio": 0.5, "seed": 23},
    "blue_tall": {"width": 14.0, "depth": 9.0, "storeys": 4, "tint": "#8499a6", "stone": "#d0cabc", "joinery": "#f1ede2",
                  "iron": "#20232a", "canopy_text": "ROYAL", "neon": "#5dff8c", "lit_ratio": 0.4, "age": 0.75, "seed": 37},
    "faded_plain": {"width": 18.0, "depth": 11.0, "storeys": 3, "tint": "#bf8667", "stone": "#bcb09a", "balconies": "none",
                    "roof": "steep", "sign_side": "right", "canopy_text": "PENSION", "neon": "#ff7a2e", "lit_ratio": 0.3,
                    "age": 1.0, "chimneys": 3, "seed": 52},
}

DEBUG = False


ROOM_GLOW, CURT_GLOW, BULB_GLOW, NEON_GLOW = 0.9, 4.0, 6.0, 1.3      # emissive strengths
SIGN_GLOW = 0.9                                                      # back-lit face of the blade sign
CD_CANOPY, CD_LANTERN, CD_UPPER = 19.0, 6.0, 10.0                    # candela of the fixtures


# --------------------------------------------------------------------------------------- helpers
def _slabs(name, items, ch=0.015):
    """One mesh of chamfered facing stones. items: (quad [(x, z)] x4 CCW seen from the front, y_front, depth[, sides])."""
    V, F = [], []
    for it in items:
        quad, yf, dep = it[:3]
        sides = it[3] if len(it) > 3 else True
        c = min(ch, dep * 0.6)
        b = len(V)
        for (x, z) in quad:
            V.append((x, yf + c, z))
        for i in range(4):
            x, z = quad[i]
            px, pz = quad[i - 1]
            nx, nz = quad[(i + 1) % 4]
            l0 = hypot(px - x, pz - z) or 1.0
            l1 = hypot(nx - x, nz - z) or 1.0
            k0, k1 = min(c, l0 * 0.3) / l0, min(c, l1 * 0.3) / l1
            V.append((x + (px - x) * k0 + (nx - x) * k1, yf, z + (pz - z) * k0 + (nz - z) * k1))
        F.append((b + 4, b + 5, b + 6, b + 7))
        for i in range(4):
            j = (i + 1) % 4
            F.append((b + i, b + j, b + 4 + j, b + 4 + i))
        if sides:
            b2 = len(V)
            for (x, z) in quad:
                V.append((x, yf + dep, z))
            for i in range(4):
                j = (i + 1) % 4
                F.append((b2 + i, b2 + j, b + j, b + i))
    return mesh_object(name, V, F)


_BF = {"f": (0, 1, 5, 4), "k": (3, 2, 6, 7), "l": (0, 4, 6, 2), "r": (1, 3, 7, 5), "t": (4, 5, 7, 6), "b": (0, 2, 3, 1)}


def _boxes(name, items):
    """One mesh of axis-aligned bars with only the faces that can be seen. items: (cx, cy, cz, sx, sy, sz, faces)
    with faces a string of f(ront, -y) k(back) l(eft) r(ight) t(op) b(ottom)."""
    V, F = [], []
    for (cx, cy, cz, sx, sy, sz, faces) in items:
        b = len(V)
        for dz in (-0.5, 0.5):
            for dy in (-0.5, 0.5):
                for dx in (-0.5, 0.5):
                    V.append((cx + dx * sx, cy + dy * sy, cz + dz * sz))
        for ch in faces:
            F.append(tuple(b + i for i in _BF[ch]))
    return mesh_object(name, V, F)


def _quads(name, quads, uvs=None):
    """Mesh of loose quads (each 4 world-space points); optional per-corner UVs."""
    V, F, UV = [], [], []
    for k, q in enumerate(quads):
        b = len(V)
        V.extend(tuple(v) for v in q)
        F.append((b, b + 1, b + 2, b + 3))
        if uvs is not None:
            UV.extend(uvs[k])
    if not V:
        return None
    o = mesh_object(name, V, F)
    if uvs is not None:
        _set_uv(o, UV)
    return o


def _axis(lo, hi, must, extra, step):
    pts = sorted(set([round(lo, 4), round(hi, 4)] + [round(m, 4) for m in must if lo + 1e-4 < m < hi - 1e-4]))
    for e in extra:
        if lo + 0.03 < e < hi - 0.03 and all(abs(e - q) > 0.03 for q in pts):
            pts.append(e)
    pts.sort()
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(ceil((b - a) / step - 1e-6)))
        out += [a + (b - a) * k / n for k in range(n)]
    out.append(pts[-1])
    return np.array(out)


def _wall_grid(name, xa, xb, za, zb, holes, step, T, xextra=(), zextra=(), caps=False):
    """Wall with rectangular openings as one welded, regular quad grid (so vertex masks and the slight
    out-of-true interpolate cleanly) plus reveals of depth T. holes: (x0, z0, x1, z1)."""
    xc = _axis(xa, xb, [h[0] for h in holes] + [h[2] for h in holes], xextra, step)
    zc = _axis(za, zb, [h[1] for h in holes] + [h[3] for h in holes], zextra, step)
    nx, nz = len(xc), len(zc)
    solid = np.ones((nx - 1, nz - 1), dtype=bool)
    for (x0, z0, x1, z1) in holes:
        i0, i1 = np.searchsorted(xc, x0 - 1e-4), np.searchsorted(xc, x1 - 1e-4)
        j0, j1 = np.searchsorted(zc, z0 - 1e-4), np.searchsorted(zc, z1 - 1e-4)
        solid[i0:i1, j0:j1] = False
    V, F, vid, bid = [], [], {}, {}

    def fv(i, j):
        if (i, j) not in vid:
            vid[(i, j)] = len(V)
            V.append((xc[i], 0.0, zc[j]))
        return vid[(i, j)]

    def bv(i, j):
        if (i, j) not in bid:
            bid[(i, j)] = len(V)
            V.append((xc[i], T, zc[j]))
        return bid[(i, j)]
    for i in range(nx - 1):
        for j in range(nz - 1):
            if solid[i, j]:
                F.append((fv(i, j), fv(i + 1, j), fv(i + 1, j + 1), fv(i, j + 1)))
                continue
            if i > 0 and solid[i - 1, j]:
                F.append((fv(i, j), bv(i, j), bv(i, j + 1), fv(i, j + 1)))
            if i < nx - 2 and solid[i + 1, j]:
                F.append((fv(i + 1, j), fv(i + 1, j + 1), bv(i + 1, j + 1), bv(i + 1, j)))
            if j > 0 and solid[i, j - 1]:
                F.append((fv(i, j), fv(i + 1, j), bv(i + 1, j), bv(i, j)))
            if j < nz - 2 and solid[i, j + 1]:
                F.append((fv(i, j + 1), bv(i, j + 1), bv(i + 1, j + 1), fv(i + 1, j + 1)))
    if caps:
        b = len(V)
        V += [(xa, 0, za), (xa, 0, zb), (xa, T, zb), (xa, T, za), (xb, 0, za), (xb, T, za), (xb, T, zb), (xb, 0, zb)]
        F += [(b, b + 1, b + 2, b + 3), (b + 4, b + 5, b + 6, b + 7)]
    return mesh_object(name, V, F)


def _set_uv(o, uv):
    me = o.data
    lay = me.uv_layers.active or me.uv_layers.new(name="UVMap")
    idx = np.zeros(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", idx)
    lay.data.foreach_set("uv", np.asarray(uv, dtype=np.float32)[idx].ravel())
    uv_keep(o)


def _ring(name, x0, x1, y0, y1, prof, seg=1.0):
    """Mitred moulding / roof running round a rectangle. prof = [(outward offset, z)] bottom to top.
    UVs: U along the walls (m), V up the profile (m, eaves at 0)."""
    def corners(o):
        return [(x0 - o, y0 - o), (x1 + o, y0 - o), (x1 + o, y1 + o), (x0 - o, y1 + o)]
    lens = [x1 - x0, y1 - y0, x1 - x0, y1 - y0]
    cols, u = [], 0.0
    for s in range(4):
        n = max(1, int(ceil(lens[s] / seg)))
        for i in range(n):
            cols.append((s, i / n, u + lens[s] * i / n))
        u += lens[s]
    cols.append((3, 1.0, u))
    V, UV, vv, prev = [], [], 0.0, None
    for (o, z) in prof:
        if prev is not None:
            vv += hypot(o - prev[0], z - prev[1])
        prev = (o, z)
        c = corners(o)
        for (s, t, uu) in cols:
            a, b = c[s], c[(s + 1) % 4]
            V.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, z))
            UV.append((uu, vv))
    n = len(cols)
    F = [(k * n + j, k * n + j + 1, (k + 1) * n + j + 1, (k + 1) * n + j) for k in range(len(prof) - 1) for j in range(n - 1)]
    o = mesh_object(name, V, F)
    _set_uv(o, UV)
    return o


def _xf(objs, M):
    for o in objs:
        o.matrix_basis = M @ o.matrix_basis


def _strip(o, pred):
    bm = bmesh.new()
    bm.from_mesh(o.data)
    kill = [f for f in bm.faces if pred(f.calc_center_median(), f.normal)]
    bmesh.ops.delete(bm, geom=kill, context='FACES')
    bm.to_mesh(o.data)
    bm.free()


def _gap(o, x0, x1, ymin):
    """Cut the stretch x0..x1 out of the run of a moulding that lies beyond y = ymin (UVs are kept)."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    for xc in (x0, x1):
        fs = [f for f in bm.faces if f.calc_center_median().y > ymin]
        geom = fs + list({e for f in fs for e in f.edges}) + list({v for f in fs for v in f.verts})
        bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(xc, 0, 0), plane_no=(1, 0, 0))
    kill = [f for f in bm.faces if f.calc_center_median().y > ymin and x0 < f.calc_center_median().x < x1]
    bmesh.ops.delete(bm, geom=kill, context='FACES')
    bm.to_mesh(o.data)
    bm.free()


def _fit_text(name, s, cap, cx, cz, y, back=False, font="sans", depth=0.03, maxw=None, wide=1.0):
    """Lettering scaled to an exact cap height, centred on (cx, cz), standing on the plane y. wide > 1 = extended
    letters (stretched sideways up to maxw, the cap height is kept)."""
    t = text(name, s, size=cap * 1.5, depth=depth, rot=(90, 0, 180 if back else 0), font=font)
    apply_transform(t)
    me = t.data
    n = len(me.vertices)
    if n == 0:
        return t
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mn, mx = co.min(0), co.max(0)
    k = cap / max(mx[2] - mn[2], 1e-4)
    kx = k * wide
    if maxw and (mx[0] - mn[0]) * kx > maxw:
        kx = maxw / (mx[0] - mn[0])
        if wide <= 1.0:
            k = kx
    c = (mn + mx) / 2
    co = (co - c) * np.array([kx, 1.0, k], dtype=np.float32)
    co[:, 0] += cx
    co[:, 2] += cz
    co[:, 1] += (y + depth / 2) if back else (y - depth / 2)
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()
    return t


def _T(x, y, z):
    return Matrix.Translation((x, y, z))


def _RZ(a):
    return Matrix.Rotation(radians(a), 4, 'Z')


def _resample(pts, step):
    """Evenly spaced points along a closed polyline."""
    P = [np.array(q, dtype=float) for q in pts] + [np.array(pts[0], dtype=float)]
    seg = [float(np.linalg.norm(b - a)) for a, b in zip(P[:-1], P[1:])]
    tot = sum(seg)
    n = max(3, int(round(tot / step)))
    out, k, acc = [], 0, 0.0
    for i in range(n):
        d = tot * i / n
        while k < len(seg) - 1 and acc + seg[k] < d:
            acc += seg[k]
            k += 1
        t = (d - acc) / max(seg[k], 1e-9)
        out.append(P[k] + (P[k + 1] - P[k]) * t)
    return out


# ----------------------------------------------------------------------------------------- build
def build(p, rng):
    W, D, S = float(p.width), float(p.depth), int(p.storeys)
    age = float(p.age)
    sd = int(p.seed)
    tile = p.roof_cover == "tile"
    T = 0.30                       # wall thickness shown in the reveals
    PL, CH, NC = 0.85, 0.45, 7     # plinth height, course height, courses of the rusticated storey
    ZS = PL + NC * CH              # underside of the string course (4.0)
    HG, HU = ZS + 0.42, 3.2        # first-floor level, upper storey height
    ZW = HG + (S - 1) * HU         # wall head
    FL = 0.45                      # ground-floor level (three risers)
    large = W * S > 56
    I4 = Matrix.Identity(4)

    M_plaster = mat("plaster_rough", p.tint, grime=0.8)
    M_stone = mat("stone_smooth", p.stone)             # also every lead-dressed part (painted fully "grimed" = lead grey)
    M_roof = mat("terracotta", "#c56a45", rough=1.1, wear=0.4) if tile else mat("slate", rough=1.5, wear=0.2)
    M_iron = mat("iron_painted", p.iron)
    M_wood = mat("wood_painted", p.joinery)
    def _lin(h):
        v = [int(h.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in v]
    rl = _lin(p.room_light)
    M_room = mat("emissive", (rl[0], rl[1] * 0.78, rl[2] * 0.5), glow=ROOM_GLOW)      # lit walls / ceilings of the rooms
    M_curt = mat("window_lit", p.curtains, glow=CURT_GLOW, curtain=1.0)   # back-lit curtains, lantern panes
    M_dark = mat("window_dark")
    M_bulb = mat("emissive", "#ffc152", glow=BULB_GLOW)
    M_neon = mat("neon", p.neon, glow=NEON_GLOW, sign=1)
    M_sign = mat("emissive", p.sign_color, glow=SIGN_GLOW, sign=1)       # back-lit enamel face of the blade sign
    # sign=1: the blade sign's own light (tubes, face, gold letters, crown, chaser bulbs) - the web bake and the page draw a
    # sign brighter than a lamp or a lit room of the same material (tools/webbake_bl.py: LOOK["neon"])
    M_gold = mat("emissive", (rl[0], rl[1] * 0.78, rl[2] * 0.5), glow=ROOM_GLOW, sign=1)
    M_sbulb = mat("emissive", "#ffc152", glow=BULB_GLOW, sign=1)
    G = {k: [] for k in ("stone", "plaster", "roof", "iron", "wood", "lead", "int", "floor", "other")}
    MM = {"stone": M_stone, "plaster": M_plaster, "roof": M_roof, "iron": M_iron, "wood": M_wood, "lead": M_stone,
          "int": M_plaster, "floor": M_wood}

    def add(kind, o, m=None):
        if o is None:
            return None
        assign(o, m or MM[kind])
        G[kind].append(o)
        return o

    # ---- roof section (same on all four sides)
    OE, ZE = 0.14, ZW + 0.07
    if p.roof == "mansard":
        ph1 = radians(72.0)
        OC, ZC = OE - 2.55 / tan(ph1), ZE + 2.55
        OT = -(D / 2 - 1.2)
        ZT = ZC + 0.04 + (OC - OT) * tan(radians(22.0))
        SEGS = [(OE, ZE, OC, ZC), (OC, ZC + 0.04, OT, ZT)]
    else:
        run = OE + D / 2 - 0.9
        ph1 = min(radians(56.0), atan(5.6 / run))
        OT, ZT = -(D / 2 - 0.9), ZE + run * tan(ph1)
        OC, ZC = OT, ZT
        SEGS = [(OE, ZE, OT, ZT)]

    def slope_y(z):                # local y of the lower roof slope at height z
        return -OE + (z - ZE) / tan(ph1)

    # ---- bay rhythm
    nbf = int((W - 0.9) / 2.1)
    if nbf % 2 == 0:
        nbf -= 1
    nbf = max(3, nbf)
    pitch_f = (W - 0.9) / nbf
    gw = clamp(pitch_f * 0.55, 1.1, 1.35)                  # arched ground window
    dw = clamp(2 * (pitch_f - 0.5) - gw, 1.7, 2.1)         # entrance
    ow = clamp(pitch_f * 0.48, 0.95, 1.15)                 # upper windows

    # ---- shared accumulators (world space)
    inst = {"bal": [], "ringx": [], "post": [], "fin": [], "bulb": [], "sbulb": [], "mod": [], "spear": [], "lamp": []}
    Q = {"room": [], "int": [], "floor": [], "curt": [], "curt_uv": []}
    kits, panes_dark = {}, []
    wall_paint, slate_paint = [], []

    def wq(M, pts):
        return [tuple(M @ Vector(q)) for q in pts]

    def curtain(M, x0, x1, z0, z1, y, w, h, ku, kv):
        if x1 - x0 < 0.02 or z1 - z0 < 0.02:
            return
        Q["curt"].append(wq(M, [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]))
        u0, u1, v0, v1 = ku + 0.02 + 0.96 * (x0 + w / 2) / w, ku + 0.02 + 0.96 * (x1 + w / 2) / w, kv + 0.02 + 0.96 * z0 / h, kv + 0.02 + 0.96 * z1 / h
        Q["curt_uv"].append([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])

    def room(M, w, h, r, kind=None, y0=T, depth=0.6, mx=0.2, top=0.3, bot=0.3, small=False):
        """A real recessed room behind a lit window: glowing back wall and ceiling, dark side walls and floor,
        curtains in the plane of the inner reveal, a lamp and a furniture silhouette. Local frame of the window."""
        kind = kind or r.pick(["drawn", "drawn", "drawn", "half", "half", "half", "open", "open", "dim"])
        xa, xb, za, zb, y1 = -w / 2 - mx, w / 2 + mx, -bot, h + top, y0 + depth
        dimk = kind == "dim"
        Q["int" if dimk else "room"] += [wq(M, [(xa, y1, za), (xb, y1, za), (xb, y1, zb), (xa, y1, zb)]),
                                         wq(M, [(xa, y0, zb), (xa, y1, zb), (xb, y1, zb), (xb, y0, zb)])]
        Q["int"] += [wq(M, [(xa, y0, za), (xa, y1, za), (xa, y1, zb), (xa, y0, zb)]),
                     wq(M, [(xb, y0, za), (xb, y0, zb), (xb, y1, zb), (xb, y1, za)])]
        Q["floor"].append(wq(M, [(xa, y0, za), (xb, y0, za), (xb, y1, za), (xa, y1, za)]))
        yc = y0 - 0.035
        ku, kv = int(r.u(0, 9)), int(r.u(0, 9))
        if kind == "drawn":
            gap = r.pick([0.0, 0.0, 0.05, 0.14])
            curtain(M, -w / 2, -gap / 2, 0, h, yc, w, h, ku, kv)
            curtain(M, gap / 2, w / 2, 0, h, yc, w, h, ku, kv)
        elif kind in ("half", "open", "salon"):
            fa = r.u(0.2, 0.34) if kind == "half" else r.u(0.11, 0.17)
            fb_ = fa if (kind != "half" or r.chance(0.6)) else r.u(0.1, 0.45)
            curtain(M, -w / 2, -w / 2 + fa * w, 0, h, yc, w, h, ku, kv)
            curtain(M, w / 2 - fb_ * w, w / 2, 0, h, yc, w, h, ku, kv)
            if kind == "salon":                       # lambrequin in the arch and a half-height cafe curtain
                curtain(M, -w / 2, w / 2, h - w * 0.36, h, yc - 0.012, w, h, ku + 1, kv)
                curtain(M, -w / 2 + fa * w * 0.6, w / 2 - fb_ * w * 0.6, 0, r.u(0.5, 0.68), yc + 0.012, w, h, ku + 2, kv)
            elif r.chance(0.4) and not small:
                curtain(M, -w / 2, w / 2, h - r.u(0.16, 0.26), h, yc - 0.012, w, h, ku + 1, kv)
        if kind == "drawn":
            return
        # lamp: pendant below the ceiling or a table lamp near the sill
        sc = 0.75 if small else (1.35 if kind == "salon" else 1.0)
        if kind == "salon":
            lp = (r.jitter(0, w * 0.1), y0 + depth * 0.6, h - w * 0.36 - 0.3)
            for dx_ in (-0.2, 0.2):
                inst["lamp"].append(M @ _T(lp[0] + dx_, lp[1], lp[2] + 0.07) @ Matrix.Diagonal((0.7, 0.7, 0.8, 1)))
        elif r.chance(0.55):
            lp = (r.jitter(0, w * 0.12), y0 + depth * 0.55, min(h - 0.12, zb - 0.42) if not small else h - 0.14)
        else:
            lp = (r.pick([-1, 1]) * w * r.u(0.12, 0.3), y0 + depth * 0.6, max(0.16, min(0.5, h * 0.25)))
        inst["lamp"].append(M @ _T(*lp) @ Matrix.Diagonal((sc, sc, sc * 1.15, 1)))
        if dimk:                                      # only a pool of lamplight on the wall behind
            hx = 0.34 * sc
            Q["curt"].append(wq(M, [(lp[0] - hx, y1 - 0.02, lp[2] - hx * 1.3), (lp[0] + hx, y1 - 0.02, lp[2] - hx * 1.3),
                                    (lp[0] + hx, y1 - 0.02, lp[2] + hx * 0.7), (lp[0] - hx, y1 - 0.02, lp[2] + hx * 0.7)]))
            Q["curt_uv"].append([(ku + 0.35, kv + 0.3), (ku + 0.65, kv + 0.3), (ku + 0.65, kv + 0.7), (ku + 0.35, kv + 0.7)])
            return
        # furniture silhouettes against the glowing wall
        yb_ = y1 - 0.025
        if not small:                                 # picture rail under the ceiling
            Q["int"].append(wq(M, [(xa, yb_, zb - 0.34), (xb, yb_, zb - 0.34), (xb, yb_, zb - 0.29), (xa, yb_, zb - 0.29)]))
        fk = "dado" if kind == "salon" else r.pick(["wardrobe", "picture", "picture", "bed", "door", "door"])
        if fk == "wardrobe" and not small:
            sx = r.pick([-1, 1])
            x0_, x1_ = sorted((sx * (w / 2 + mx - 0.04), sx * (w / 2 + mx - 0.04 - r.u(0.45, 0.7))))
            Q["int"].append(wq(M, [(x0_, yb_, za), (x1_, yb_, za), (x1_, yb_, za + (zb - za) * r.u(0.62, 0.78)), (x0_, yb_, za + (zb - za) * r.u(0.62, 0.78))]))
        elif fk == "picture":
            cx_, cz_ = r.jitter(0, w * 0.25), h * r.u(0.55, 0.75)
            pw, phh = r.u(0.36, 0.6), r.u(0.3, 0.5)
            ft = 0.04
            for (ax, az, bx_, bz) in ((-pw / 2, -phh / 2, pw / 2, -phh / 2 + ft), (-pw / 2, phh / 2 - ft, pw / 2, phh / 2),
                                      (-pw / 2, -phh / 2, -pw / 2 + ft, phh / 2), (pw / 2 - ft, -phh / 2, pw / 2, phh / 2)):
                Q["int"].append(wq(M, [(cx_ + ax, yb_, cz_ + az), (cx_ + bx_, yb_, cz_ + az), (cx_ + bx_, yb_, cz_ + bz), (cx_ + ax, yb_, cz_ + bz)]))
        elif fk == "door" and not small:
            cx_, dw_, dh_, ft = r.pick([-1, 1]) * w * r.u(0.05, 0.3), 0.8, 2.05, 0.06
            for (ax, az, bx_, bz) in ((-dw_ / 2, 0, -dw_ / 2 + ft, dh_), (dw_ / 2 - ft, 0, dw_ / 2, dh_), (-dw_ / 2, dh_ - ft, dw_ / 2, dh_),
                                      (-0.03, 0, 0.03, dh_) if r.chance(0.5) else (-dw_ / 2, 0.95, dw_ / 2, 1.0)):
                Q["int"].append(wq(M, [(cx_ + ax, yb_, za + az), (cx_ + bx_, yb_, za + az), (cx_ + bx_, yb_, za + bz), (cx_ + ax, yb_, za + bz)]))
        elif fk in ("bed", "dado"):
            zt_ = (za + 0.5) if fk == "dado" else max(0.25, min(0.7, h * 0.3))
            if zt_ > za + 0.1:
                Q["int"].append(wq(M, [(xa, yb_, za), (xb, yb_, za), (xb, yb_, zt_), (xa, yb_, zt_)]))

    def glaze(M, w, h, lit, r, **kw):
        if lit:
            room(M, w, h, r, **kw)
        else:
            panes_dark.append((M, w - 0.03, h - 0.03, 0.2, h / 2, 0.0))

    def put(key, maker, M):
        if key not in kits:
            kits[key] = {"trim": maker(), "mats": []}
        kits[key]["mats"].append(M)

    # ---- window kits (built once, stamped with instances_to_mesh; only faces that can be seen)
    def sash(w, h, y0=0.13, cols=3, rows=2, french=False):
        fw, fd = 0.05, 0.1
        yc = y0 + fd / 2
        it = [(-w / 2 + fw / 2, yc, h / 2, fw, fd, h, "fr"), (w / 2 - fw / 2, yc, h / 2, fw, fd, h, "fl"),
              (0, yc, h - fw / 2, w - 2 * fw, fd, fw, "fb"), (0, yc - 0.015, 0.0225, w - 2 * fw, fd + 0.03, 0.045, "ft")]
        iw = w - 2 * fw
        if french:
            lw, sw, z0, z1 = iw / 2, 0.06, 0.045, h - fw
            yy = y0 + 0.045
            for sx in (-1, 1):
                cx = sx * lw / 2
                it.append((cx - (lw / 2 - sw / 2), yy, (z0 + z1) / 2, sw, 0.045, z1 - z0, "fr"))
                it.append((cx + (lw / 2 - sw / 2), yy, (z0 + z1) / 2, sw, 0.045, z1 - z0, "fl"))
                it.append((cx, yy, z1 - sw / 2, lw - 2 * sw, 0.045, sw, "fb"))
                it.append((cx, yy, z0 + 0.1, lw - 2 * sw, 0.045, 0.2, "ft"))
                for k in range(1, 4):
                    it.append((cx, yy, z0 + 0.2 + (z1 - sw - z0 - 0.2) * k / 4, lw - 2 * sw, 0.028, 0.022, "ftb"))
        else:
            zm, sw = h * 0.5, 0.045
            for (za, zb, yy) in ((0.045, zm + 0.02, y0 + 0.065), (zm - 0.02, h - fw, y0 + 0.03)):
                it.append((-(iw / 2 - sw / 2), yy, (za + zb) / 2, sw, 0.035, zb - za, "fr"))
                it.append((iw / 2 - sw / 2, yy, (za + zb) / 2, sw, 0.035, zb - za, "fl"))
                it.append((0, yy, za + sw / 2, iw - 2 * sw, 0.035, sw, "ftb"))
                it.append((0, yy, zb - sw / 2, iw - 2 * sw, 0.035, sw, "ftb"))
                for k in range(1, cols):
                    it.append((-iw / 2 + iw * k / cols, yy, (za + zb) / 2, 0.02, 0.025, zb - za - 2 * sw, "flr"))
                for k in range(1, rows):
                    it.append((0, yy, za + (zb - za) * k / rows, iw - 2 * sw, 0.025, 0.02, "ftb"))
        return it

    def mk_window(w, h, hood="none", french=False, plain=False, cols=3, sill=True, arch=True, lintel=False):
        def maker():
            st, bx, it = [], [], []
            aw = 0.16 if plain else 0.15
            yf = -0.05 if plain else -0.04
            dp = -yf + 0.03
            ear = 0.0 if plain else 0.05
            zt = h
            if arch:
                it += [([(w / 2, 0), (w / 2 + aw, 0), (w / 2 + aw, h), (w / 2, h)], yf, dp),
                       ([(-w / 2 - aw, 0), (-w / 2, 0), (-w / 2, h), (-w / 2 - aw, h)], yf, dp),
                       ([(-w / 2 - aw - ear, h), (w / 2 + aw + ear, h), (w / 2 + aw + ear, h + aw), (-w / 2 - aw - ear, h + aw)], yf, dp)]
                zt = h + aw
            if hood == "key":
                it.append(([(-0.075, h - 0.02), (0.075, h - 0.02), (0.11, zt + 0.04), (-0.11, zt + 0.04)], yf - 0.03, dp + 0.03))
            if hood in ("cornice", "tri", "seg") or lintel:
                cw_ = w + 2 * aw
                zf_ = zt
                if not lintel:
                    it.append(([(-cw_ / 2, zt), (cw_ / 2, zt), (cw_ / 2, zt + 0.13), (-cw_ / 2, zt + 0.13)], -0.02, 0.05))
                    zf_ = zt + 0.13
                    for sx in (-1, 1):
                        bx.append((sx * (w / 2 + aw - 0.045), -0.05, zt, 0.09, 0.1, 0.26, "flrb"))
                bx.append((0, -0.045, zf_ + 0.025, cw_ + 0.14, 0.13, 0.05, "flrb"))
                bx.append((0, -0.08, zf_ + 0.07, cw_ + 0.26, 0.2, 0.045, "flrtb"))
                zp, hw = zf_ + 0.0925, cw_ / 2 + 0.13
                if hood == "tri":
                    ph = hw * 0.36
                    st.append(extrude_profile("ped", [(-hw + 0.04, 0), (hw - 0.04, 0), (0, ph - 0.015)], 0.09, loc=(0, 0.02, zp), rot=(90, 0, 0)))
                    ang, Lr = atan2(ph, hw), hypot(hw, ph) + 0.03
                    for sx in (-1, 1):
                        st.append(box("rake", (Lr, 0.2, 0.05), loc=(sx * hw / 2, -0.08, zp + ph / 2 + 0.012), rot=(0, sx * degrees(ang), 0)))
                elif hood == "seg":
                    ph = hw * 0.3
                    Rr = (hw * hw + ph * ph) / (2 * ph)
                    a = asin(hw / Rr)
                    arc = [(Rr * sin(-a + 2 * a * k / 6), ph - Rr + Rr * cos(-a + 2 * a * k / 6)) for k in range(7)]
                    st.append(extrude_profile("pedS", [(x_ * 0.97, z_ - 0.01) for x_, z_ in reversed(arc)], 0.09, loc=(0, 0.02, zp), rot=(90, 0, 0)))
                    st.append(sweep("rakeS", [(x_, -0.08, zp + z_ + 0.015) for x_, z_ in arc],
                                    [(-0.1, -0.025), (0.1, -0.025), (0.1, 0.025), (-0.1, 0.025)], smooth_path=False))
            if french:
                bx.append((0, 0.1, -0.035, w, 0.3, 0.07, "ft"))
            elif sill:
                bx.append((0, -0.04, -0.04, w + 2 * aw + 0.1, 0.24, 0.08, "flrtb"))
                for sx in (-1, 1):
                    bx.append((sx * (w / 2 + aw / 2), -0.03, -0.15, 0.1, 0.09, 0.14, "flrb"))
            if it:
                st.append(_slabs("arch", it, 0.012))
            if bx:
                st.append(_boxes("trim", bx))
            for o in st:
                assign(o, M_stone)
            wd = _boxes("sash", sash(w, h, cols=cols, french=french, rows=2 if h < 2.0 else 3))
            assign(wd, M_wood)
            return join(st + [wd], "kit")
        return maker

    def mk_arch_win():
        w, zs, R, y0 = gw, 3.1 - 1.3, gw / 2, 0.14
        yc = y0 + 0.05
        path = [(w / 2 - 0.03, yc, 0.0), (w / 2 - 0.03, yc, zs)]
        path += [((R - 0.03) * cos(pi * k / 10), yc, zs + (R - 0.03) * sin(pi * k / 10)) for k in range(1, 10)]
        path += [(-w / 2 + 0.03, yc, zs), (-w / 2 + 0.03, yc, 0.0)]
        wd = [sweep("af", path, [(-0.03, -0.05), (0.03, -0.05), (0.03, 0.05), (-0.03, 0.05)], smooth_path=False, up=(0, 1, 0))]
        it = [(0, yc - 0.01, 0.025, w, 0.13, 0.05, "ft"), (0, yc, zs, w - 0.06, 0.08, 0.06, "ftb"), (0, yc, zs / 2, 0.05, 0.07, zs, "flr")]
        for sx in (-1, 1):
            cx = sx * (w / 4 - 0.01)
            it.append((cx - (w / 4 - 0.055), yc + 0.01, zs / 2, 0.04, 0.04, zs - 0.08, "fr"))
            it.append((cx + (w / 4 - 0.055), yc + 0.01, zs / 2, 0.04, 0.04, zs - 0.08, "fl"))
            for zz in (0.07, zs - 0.05):
                it.append((cx, yc + 0.01, zz, w / 2 - 0.1, 0.04, 0.045, "ftb"))
            for k in (1, 2):
                it.append((cx, yc + 0.01, zs * k / 3, w / 2 - 0.1, 0.028, 0.02, "ftb"))
        wd.append(_boxes("ab", it))
        for a in (45, 90, 135):
            ar = radians(a)
            wd.append(box("sp", (0.024, 0.03, R - 0.05), loc=((R - 0.05) / 2 * cos(ar), yc + 0.01, zs + 0.02 + (R - 0.05) / 2 * sin(ar)), rot=(0, 90 - a, 0)))
        wd.append(sweep("ah", [(0.52 * R * cos(pi * k / 8), yc + 0.01, zs + 0.02 + 0.52 * R * sin(pi * k / 8)) for k in range(9)],
                        [(-0.012, -0.015), (0.012, -0.015), (0.012, 0.015), (-0.012, 0.015)], smooth_path=False, up=(0, 1, 0)))
        for o in wd:
            assign(o, M_wood)
        return join(wd, "kit_arch")

    def course(items, z0, z1, a_, b_, excl, Lb, off, r, yf=0.0, dep=0.05, sides=False):
        cur, free = a_, []
        for (a, b) in sorted(excl):
            if a > cur + 0.04:
                free.append((cur, min(a, b_)))
            cur = max(cur, b)
        if cur < b_ - 0.04:
            free.append((cur, b_))
        for (a, b) in free:
            cuts = [a]
            x = off + ceil((a - off) / Lb) * Lb
            while x < b - 0.2:
                if x > a + 0.2:
                    cuts.append(x + r.jitter(0, 0.025))
                x += Lb
            cuts.append(b)
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                items.append(([(c0, z0), (c1, z0), (c1, z1), (c0, z1)], yf + r.u(0.001, 0.007), dep, sides))

    def balcony(objs, Mf, xc, z, bw, proj):
        s = add("stone", box("bal_slab", (bw, proj + 0.03, 0.11), loc=(xc, -proj / 2 + 0.015, z - 0.035), bevel=0.02, segments=1))
        objs.append(s)
        ncon = 2 if bw < 3.0 else 4
        cp = min(proj, 0.62)
        prof = [(0, 0), (-cp * 0.92, 0), (-cp * 0.92, -0.06), (-cp * 0.78, -0.1), (-cp * 0.6, -0.12), (-cp * 0.4, -0.2),
                (-0.16, -0.3), (-0.1, -0.4), (0, -0.44)]
        for k in range(ncon):
            cx = xc + (k / (ncon - 1) - 0.5) * (bw - 0.4)
            c = add("stone", extrude_profile("bal_console", prof, 0.14, loc=(cx - 0.07, 0, z - 0.085), rot=(90, 0, 90)))
            objs.append(c)
        xl, xr, yo = xc - bw / 2 + 0.06, xc + bw / 2 - 0.06, -proj + 0.06
        runs = [((xl, -0.02), (xl, yo)), ((xl, yo), (xr, yo)), ((xr, yo), (xr, -0.02))]
        posts = [(xl, -0.045), (xr, -0.045), (xl, yo), (xr, yo)]
        nmid = int((xr - xl) / 1.45)
        for k in range(1, nmid + 1):
            posts.append((xl + (xr - xl) * k / (nmid + 1), yo))
        for (px, py) in posts:
            inst["post"].append(Mf @ _T(px, py, z + 0.02))
            inst["fin"].append(Mf @ _T(px, py, z + 1.075))
        rails = []
        for (a, b) in runs:
            L = hypot(b[0] - a[0], b[1] - a[1])
            alongx = abs(b[0] - a[0]) > abs(b[1] - a[1])
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            for (zz, sw, sh) in ((z + 1.0, 0.05, 0.028), (z + 0.86, 0.022, 0.016), (z + 0.1, 0.03, 0.02)):
                rails.append((mx, my, zz) + ((L + 0.02, sw, sh) if alongx else (sw, L + 0.02, sh)) + ("fktb" if alongx else "lrtb",))
            n = max(2, int(round(L / 0.125)))
            for k in range(1, n):
                t = k / n
                px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                inst["bal"].append(Mf @ _T(px, py, z + 0.1) @ _RZ(0 if alongx else 90))
                if k % 2 == 1:
                    inst["ringx"].append(Mf @ _T(px + (0.0625 * (b[0] - a[0]) / L), py + (0.0625 * (b[1] - a[1]) / L), z + 0.93)
                                         @ _RZ(0 if alongx else 90))
        objs.append(add("iron", _boxes("bal_rails", rails)))

    def dormer(objs, Mf, key, xc, zb, w, h, big_, lit, kind, r, excl):
        yf = slope_y(zb)
        owd, ohd, sill = (w - 0.5, h - 0.62, 0.34) if big_ else (w - 0.44, h - 0.5, 0.3)
        hw = w / 2 + 0.1
        ph = hw * (0.5 if kind == "tri" else 0.36)
        zp = zb + h + 0.07
        yb = slope_y(zp + ph) + 0.12
        excl.append((xc - w / 2 + 0.04, xc + w / 2 - 0.04, zb, zb + h - 0.02, zp))
        # lead cheeks and top (no front: the window opens into the dormer room)
        objs.append(add("lead", _boxes("dorm_cheeks", [(xc, (yf + 0.225 + yb) / 2, zb + (h - 0.3) / 2, w - 0.05, yb - yf - 0.225, h + 0.3, "lrt")])))
        ys, dp = yf - 0.02, 0.25
        it = [([(xc - w / 2, zb - 0.12), (xc - owd / 2, zb - 0.12), (xc - owd / 2, zb + h), (xc - w / 2, zb + h)], ys, dp),
              ([(xc + owd / 2, zb - 0.12), (xc + w / 2, zb - 0.12), (xc + w / 2, zb + h), (xc + owd / 2, zb + h)], ys, dp),
              ([(xc - owd / 2, zb - 0.12), (xc + owd / 2, zb - 0.12), (xc + owd / 2, zb + sill), (xc - owd / 2, zb + sill)], ys, dp),
              ([(xc - owd / 2, zb + sill + ohd), (xc + owd / 2, zb + sill + ohd), (xc + owd / 2, zb + h), (xc - owd / 2, zb + h)], ys, dp)]
        for sx in (-1, 1):       # pilaster strips
            it.append(([(xc + sx * (w / 2 - 0.09) - 0.065, zb + 0.02), (xc + sx * (w / 2 - 0.09) + 0.065, zb + 0.02),
                        (xc + sx * (w / 2 - 0.09) + 0.065, zb + h - 0.02), (xc + sx * (w / 2 - 0.09) - 0.065, zb + h - 0.02)], ys - 0.03, 0.05))
        objs.append(add("stone", _slabs("dorm_front", it, 0.012)))
        objs.append(add("stone", _boxes("dorm_trim", [(xc, yf + 0.1, zb + h + 0.035, w + 0.2, 0.36, 0.07, "flrtb"),
                                                      (xc, yf - 0.04, zb + sill - 0.03, owd + 0.16, 0.12, 0.06, "flrtb")])))
        # lead apron dressed down over the slates below the sill
        objs.append(add("lead", _quads("dorm_apron", [[(xc - w / 2 - 0.1, slope_y(zb - 0.3) - 0.03, zb - 0.3), (xc + w / 2 + 0.1, slope_y(zb - 0.3) - 0.03, zb - 0.3),
                                                       (xc + w / 2 + 0.1, yf - 0.035, zb - 0.02), (xc - w / 2 - 0.1, yf - 0.035, zb - 0.02)]])))
        if kind == "tri":
            pts = [(-hw, 0), (hw, 0), (0, ph)]
        else:
            Rr = (hw * hw + ph * ph) / (2 * ph)
            a = asin(hw / Rr)
            pts = [(Rr * sin(a - 2 * a * k / 6), ph - Rr + Rr * cos(a - 2 * a * k / 6)) for k in range(7)]
        objs.append(add("stone", extrude_profile("dorm_ped", pts, 0.34, loc=(xc, yf + 0.26, zp), rot=(90, 0, 0))))
        pts2 = [(x_ * 0.95, z_ * 0.95) for x_, z_ in pts]
        objs.append(add("lead", extrude_profile("dorm_roof", pts2, yb - yf - 0.2, loc=(xc, yb, zp), rot=(90, 0, 0))))
        if big_:
            objs.append(add("stone", sphere("dorm_ball", 0.11, loc=(xc, yf + 0.09, zp + ph + 0.1), segments=8, rings=5)))
            objs.append(add("stone", box("dorm_ballbase", (0.16, 0.16, 0.08), loc=(xc, yf + 0.09, zp + ph - 0.0))))
        Mw = Mf @ _T(xc, yf, zb + sill)
        put(key, mk_window(owd, ohd, plain=True, arch=False, cols=2 if owd < 0.95 else 3, sill=False), Mw)
        glaze(Mw, owd, ohd, lit, r, y0=0.24, depth=0.42, mx=0.03, top=0.03, bot=0.0, small=True)

    def drainpipe(objs, x, r):
        zt = ZW - 0.95
        y = -0.115
        objs.append(add("iron", cylinder("dp_pipe", 0.048, zt - 0.32, loc=(x, y, (zt + 0.32) / 2), verts=6)))
        hop = lathe("dp_hopper", [(0.05, 0.0), (0.06, 0.02), (0.075, 0.1), (0.13, 0.26), (0.14, 0.3), (0.15, 0.31), (0.15, 0.36), (0.125, 0.36), (0.12, 0.3)],
                    segments=4, loc=(x, y - 0.02, zt - 0.02), rot=(0, 0, 45), smooth_angle=20)
        objs.append(add("iron", hop))
        nz = max(2, int((zt - 0.6) / 1.85))
        bx = []
        for k in range(nz + 1):
            zz = 0.5 + (zt - 0.75) * k / nz
            objs.append(add("iron", cylinder("dp_collar", 0.06, 0.07, loc=(x, y, zz), verts=6)))
            bx.append((x, y + 0.065, zz, 0.2, 0.012, 0.04, "flrtb"))
            bx.append((x, y + 0.06, zz, 0.03, 0.1, 0.03, "lrtb"))
        objs.append(add("iron", _boxes("dp_ears", bx)))
        objs.append(add("iron", sweep("dp_shoe", [(x, y, 0.36), (x, y, 0.2), (x, y - 0.05, 0.11), (x, y - 0.16, 0.07)], 0.048, sides=6)))
        objs.append(add("stone", box("dp_gully", (0.4, 0.34, 0.07), loc=(x, y - 0.12, 0.035), bevel=0.012, segments=1)))

    # ---- one roof side: lead-grey underlay with dormer openings, then the covering laid slate by slate
    SG, SWD, LIFT = (0.19, 0.215, 0.03) if tile else (0.26, 0.34, 0.03)
    roof_area = sum(hypot(a - c, d - b) * (2 * (W + D) + 4 * (a + c)) for (a, b, c, d) in SEGS[:1])
    kscale = max(1.0, sqrt(roof_area / (SG * SWD) / (5600.0 if tile else 4600.0)))       # keep the count bounded on the largest roofs
    SG, SWD = SG * kscale, SWD * kscale

    def roof_side(side, L, Mf, excl):
        nr = np.random.RandomState(sd * 31 + len(side) * 7 + ord(side[0]))
        quads = []
        for (o0, z0, o1, z1) in SEGS:
            cuts = sorted(set([z0, z1] + [z for e in excl for z in (e[2], e[3]) if z0 + 0.01 < z < z1 - 0.01]))
            for za, zb in zip(cuts[:-1], cuts[1:]):
                oa, ob = o0 + (o1 - o0) * (za - z0) / (z1 - z0), o0 + (o1 - o0) * (zb - z0) / (z1 - z0)
                ha, hb, zm = L / 2 + oa, L / 2 + ob, (za + zb) / 2
                ex = sorted((e[0], e[1]) for e in excl if e[2] - 0.001 <= zm <= e[3] + 0.001)
                lefts = [(-ha, -hb)] + [(b, b) for a, b in ex]
                rights = [(a, a) for a, b in ex] + [(ha, hb)]
                for l_, r_ in zip(lefts, rights):
                    quads.append([(l_[0], -oa, za), (r_[0], -oa, za), (r_[1], -ob, zb), (l_[1], -ob, zb)])
        base = add("lead", _quads("roof_underlay_" + side, quads))
        base.matrix_basis = Mf
        V, F, UV, gr, we = [], [], [], [], []
        pm, ps = 0.003 + 0.012 * age, 0.008 + 0.03 * age
        for sgi, (o0, z0, o1, z1) in enumerate(SEGS):
            Ls = hypot(o1 - o0, z1 - z0)
            n = max(1, int(round(Ls / SG)))
            g = Ls / n
            co, si = (o0 - o1) / Ls, (z1 - z0) / Ls

            def P(x, s, hh):
                return (x, -(o0 - co * s) - hh * si, z0 + si * s + hh * co)
            if sgi == 1:               # the flat upper slope of a mansard is sheeted in lead over wood-cored rolls, with one drip
                rq = []
                h0 = L / 2 + o0
                nrl = max(2, int(round(2 * h0 / 0.62)))
                sd_ = Ls * 0.52
                for k in range(1, nrl):
                    x = -h0 + 2 * h0 * k / nrl
                    s1 = min(Ls, (h0 - abs(x) - 0.06) / co)
                    if s1 < 0.25:
                        continue
                    for (sa, sb, e) in ((0.02, min(s1, sd_), 0.0), (sd_, s1, 0.012)):
                        if sb - sa < 0.08:
                            continue
                        a_, b_, hh = x - 0.026, x + 0.026, 0.05 + e
                        rq += [[P(a_, sa, 0), P(a_, sb, 0), P(a_ + 0.008, sb, hh), P(a_ + 0.008, sa, hh)],
                               [P(b_, sb, 0), P(b_, sa, 0), P(b_ - 0.008, sa, hh), P(b_ - 0.008, sb, hh)],
                               [P(a_ + 0.008, sa, hh), P(a_ + 0.008, sb, hh), P(b_ - 0.008, sb, hh), P(b_ - 0.008, sa, hh)],
                               [P(a_, sa, 0), P(a_ + 0.008, sa, hh), P(b_ - 0.008, sa, hh), P(b_, sa, 0)]]
                hd = h0 - co * sd_ - 0.03       # the drip: upper sheets lap 12 mm proud over the lower ones
                rq += [[P(-hd, sd_, 0.0), P(hd, sd_, 0.0), P(hd, sd_, 0.014), P(-hd, sd_, 0.014)],
                       [P(-hd, sd_, 0.014), P(hd, sd_, 0.014), P(L / 2 + o1 - 0.03, Ls, 0.012), P(-(L / 2 + o1 - 0.03), Ls, 0.012)]]
                rl_ = add("lead", _quads("roof_lead_rolls_" + side, rq))
                rl_.matrix_basis = Mf
                continue
            for c in range(n):
                sa, sb = c * g - (0.04 if c == 0 else 0.0), min(Ls, (c + 1) * g + g * 0.12)
                h0, h1 = L / 2 + o0 - co * sa, L / 2 + o0 - co * sb
                zc_ = z0 + si * (sa + sb) / 2
                ex = [(e[0], e[1]) for e in excl if e[2] - 0.06 < zc_ < e[4]]
                x = -h0 - nr.uniform(0, SWD) - (SWD / 2 if c % 2 else 0.0)
                while x < h0:
                    wv = SWD * (1.0 if tile else nr.uniform(0.86, 1.14))
                    xa, xb = x + 0.003, x + wv - 0.003
                    x += wv
                    xm = (xa + xb) / 2
                    xa0, xb0 = max(xa, -h0), min(xb, h0)
                    if xb0 - xa0 < 0.03 or any(a < xm < b for a, b in ex) or nr.uniform() < pm:
                        continue
                    xa1, xb1 = max(xa, -h1), min(xb, h1)
                    if xb1 < xa1:
                        xa1 = xb1 = min(max(xm, -h1), h1)
                    ds = -nr.uniform(0.03, 0.08) if (nr.uniform() < ps and c > 0) else 0.0
                    lf = LIFT * nr.uniform(0.75, 1.3)
                    tw = LIFT * nr.uniform(-0.3, 0.3)
                    b = len(V)
                    V += [P(xa1, sb + ds, 0.003), P(xb1, sb + ds, 0.003), P(xa0, sa + ds, lf + tw), P(xb0, sa + ds, lf - tw),
                          P(xa0, sa + ds, 0.0), P(xb0, sa + ds, 0.0)]
                    F += [(b + 2, b + 3, b + 1, b), (b + 4, b + 5, b + 3, b + 2)]
                    ou, ov = nr.uniform(0, 4), nr.uniform(0, 4)
                    UV += [(ou + xa1 - xa, ov + sb - sa), (ou + xb1 - xa, ov + sb - sa), (ou + xa0 - xa, ov), (ou + xb0 - xa, ov),
                           (ou + xa0 - xa, ov - 0.03), (ou + xb0 - xa, ov - 0.03)]
                    gv = nr.uniform(0, 1) ** 2.0 * 0.2 if tile else nr.uniform(0, 1) ** 1.6 * 0.75
                    wv_ = nr.uniform(0.15, 0.45) if nr.uniform() < (0.12 if tile else 0.07) else 0.0
                    gr += [gv] * 4 + [min(1.0, gv + (0.25 if tile else 0.35))] * 2
                    we += [wv_] * 6
        o = mesh_object("roof_cover_" + side, V, F)
        _set_uv(o, UV)
        o.matrix_basis = Mf
        add("roof", o)
        slate_paint.append((o, np.array(gr, dtype=np.float32), np.array(we, dtype=np.float32)))

    # ------------------------------------------------------------------------------- one facade
    def facade(side, L, Mf, r):
        front, back = side == "front", side == "back"
        fb = front or back
        objs = []
        ins = 0.0 if fb else T - 0.01
        xa, xb = -L / 2 + ins, L / 2 - ins
        if fb:
            nb, pitch = nbf, pitch_f
        else:
            nb = max(2, int(round((L - 1.2) / 2.3)))
            pitch = (L - 1.2) / nb
        xs = [(i - (nb - 1) / 2) * pitch for i in range(nb)]
        mid = nb // 2 if fb else -99
        litk = 1.0 if front else 0.75
        any_lit = p.lit_ratio > 0.01
        # landing door: on the pier (between two window bays) nearest the asked position, so it never meets a window
        ld = None
        piers = [(a + b) / 2 for a, b in zip(xs[:-1], xs[1:]) if abs(a + b) / 2 < L / 2 - 1.5]
        if back and p.landing_door and piers:
            ld = (min(piers, key=lambda q: abs(q + float(p.landing_door_x))), clamp(float(p.landing_door_z), 2.6, 3.4))

        # ground storey openings: [kind, x, w, z bottom, z spring / head, arch radius, lit]
        gops = []
        for i, x in enumerate(xs):
            if front and i == mid:
                gops.append(["door", x, dw, 0.02, 2.65, dw / 2, False])
            elif front:
                gops.append(["arch", x, gw, 1.3, 3.1, gw / 2, any_lit and r.chance(max(p.lit_ratio, 0.88))])
            elif back and i == mid:
                gops.append(["bdoor", x, 1.15, 0.02, 2.65, 0.0, False])
            else:
                gops.append(["rect", x, 1.0, 1.3, 3.1, 0.0, r.chance(p.lit_ratio * litk)])
        if not front and any_lit and not any(g_[6] for g_ in gops):
            r.pick([g_ for g_ in gops if g_[0] == "rect"])[6] = True
        holes = []
        for kind, x, w, zb, zs, R, lit in gops:
            if R > 0:
                holes.append([(x + px, zb + pz) for (px, pz) in arch_pts(w, zs - zb + R, spring=zs - zb, kind="round", n=18)])
            else:
                holes.append([(x - w / 2, zb), (x + w / 2, zb), (x + w / 2, zs), (x - w / 2, zs)])
        outline = [(xa, 0), (xb, 0), (xb, ZS + 0.02), (xa, ZS + 0.02)]
        if ld:                    # the door runs up through the string course: a notch in the head of the ground wall
            outline[3:3] = [(ld[0] + LDH, ZS + 0.02), (ld[0] + LDH, ld[1]), (ld[0] - LDH, ld[1]), (ld[0] - LDH, ZS + 0.02)]
        gwall = shape_extrude("ground_wall", outline, holes, depth=T - 0.04, loc=(0, T, 0), rot=(90, 0, 0))
        _strip(gwall, lambda c, n: n.z < -0.5)
        objs.append(add("stone", gwall))

        # facing: plinth blocks, rusticated courses in running bond, voussoirs, keystones, sills
        pl, items, proud, gbx = [], [], [], []
        ea = 0.0 if fb else 0.05
        course(pl, 0.0, PL, -L / 2 - ea, L / 2 + ea, [(x - w / 2, x + w / 2) for kind, x, w, zb, zs, R, lit in gops if zb < PL - 0.01],
               1.4, r.u(0, 1.4), r, yf=-0.055, dep=0.105, sides=True)
        Lb = 0.92 if front else 1.25
        ld_z0 = ZS
        for c in range(NC):
            z0, z1 = PL + c * CH, PL + (c + 1) * CH
            ex = []
            for kind, x, w, zb, zs, R, lit in gops:
                if z1 <= zb + 0.01:
                    continue
                if z0 < zs - 0.01:
                    ex.append((x - w / 2, x + w / 2))
                elif R > 0 and z0 - zs < R - 0.04:
                    hw = sqrt(R * R - (z0 - zs) ** 2) + 0.03
                    ex.append((x - hw, x + hw))
            if ld and z1 > ld[1] + 0.01:
                ex.append((ld[0] - LDO, ld[0] + LDO))
                ld_z0 = min(ld_z0, z0)
            course(items, z0, z1, -L / 2, L / 2, ex, Lb, (c % 2) * Lb / 2 + r.jitter(0, 0.04), r)
        for kind, x, w, zb, zs, R, lit in gops:
            if R > 0:
                n = 13 if kind == "door" else 9
                vd = 0.26
                for k in range(n):
                    if k == n // 2:
                        continue
                    a0, a1 = pi * k / n, pi * (k + 1) / n
                    q = [(x + R * cos(a0), zs + R * sin(a0)), (x + (R + vd) * cos(a0), zs + (R + vd) * sin(a0)),
                         (x + (R + vd) * cos(a1), zs + (R + vd) * sin(a1)), (x + R * cos(a1), zs + R * sin(a1))]
                    proud.append((q, -0.02 - r.u(0, 0.007), 0.075))
                b0 = R * sin(pi / (2 * n)) * 1.2
                b1 = b0 * 1.45 + 0.03
                proud.append(([(x - b0, zs + R - 0.04), (x + b0, zs + R - 0.04), (x + b1, zs + R + vd + 0.05), (x - b1, zs + R + vd + 0.05)], -0.06, 0.115))
                for sx in (-1, 1):        # impost blocks at the springing
                    xi = x + sx * (R + 0.13)
                    proud.append(([(xi - 0.17, zs - 0.11), (xi + 0.17, zs - 0.11), (xi + 0.17, zs), (xi - 0.17, zs)], -0.035, 0.09))
            else:
                proud.append(([(x - 0.085, zs), (x + 0.085, zs), (x + 0.125, zs + CH), (x - 0.125, zs + CH)], -0.035, 0.09))
            if kind in ("arch", "rect"):
                gbx.append((x, -0.045, zb - 0.04, w + 0.34, 0.24, 0.1, "flrtb"))
        objs.append(add("stone", _slabs("plinth", pl, 0.03)))
        objs.append(add("stone", _slabs("rustication", items, 0.02)))
        objs.append(add("stone", _slabs("voussoirs", proud, 0.016)))
        if gbx:
            objs.append(add("stone", _boxes("ground_sills", gbx)))

        # ground windows
        for kind, x, w, zb, zs, R, lit in gops:
            Mw = Mf @ _T(x, 0, zb)
            if kind == "arch":
                put("g_arch", mk_arch_win, Mw)
                if lit:
                    room(Mw, w, zs - zb + R, r, kind="salon" if r.chance(0.8) else "drawn", depth=0.8, mx=0.22, top=0.12, bot=0.25)
                else:
                    panes_dark.append((Mw, w + 0.1, zs - zb + R + 0.1, 0.234, (zs - zb + R) / 2, 0.0))    # 6 mm in front of the frame's back face (was in its plane)
            elif kind == "rect":
                put("g_rect", mk_window(w, zs - zb, plain=True, arch=False, sill=False), Mw)
                glaze(Mw, w, zs - zb, lit, r, bot=0.4, top=0.3)

        # upper storeys: stucco wall on a regular welded grid with real openings
        holes, wins, xextra, zextra, streaks = [], [], [], [ZS + 0.44, ZW - 0.84], []
        for lv in range(1, S):
            F = HG + (lv - 1) * HU
            top = lv == S - 1
            for i, x in enumerate(xs):
                lit = r.chance(p.lit_ratio * litk)
                hood, french = None, False
                if front:
                    hasb = lv == 1 and (p.balconies == "all" or (p.balconies == "centre" and abs(i - mid) <= 1))
                    if hasb or lv == 1:
                        hood = "seg" if i == mid else ("tri" if abs(i - mid) % 2 == 1 else "cornice")
                    if hasb:
                        french = True
                        w, h, zb, key, mk = ow, 2.5, F + 0.1, "fr_" + hood, mk_window(ow, 2.5, hood=hood, french=True)
                    elif lv == 1:
                        w, h, zb, key, mk = ow, 1.85, F + 0.85, "f1_" + hood, mk_window(ow, 1.85, hood=hood)
                    elif top:
                        w, h, zb, key, mk = ow, 1.45, F + 0.85, "top", mk_window(ow, 1.45, hood="key")
                    else:
                        hood = "cornice"
                        w, h, zb, key, mk = ow, 1.85, F + 0.85, "mid", mk_window(ow, 1.85, hood="cornice")
                else:
                    h = 1.45 if top else 1.75
                    w, zb = 1.0, F + 0.85
                    key, mk = "plain%d%d" % (int(h * 100), lv == 1), mk_window(1.0, h, plain=True, lintel=lv == 1)
                    hood = "lintel" if lv == 1 else None
                holes.append((x - w / 2, zb, x + w / 2, zb + h))
                wins.append({"key": key, "mk": mk, "x": x, "zb": zb, "w": w, "h": h, "lit": lit, "lv": lv, "i": i, "french": french})
                aw = 0.15 if front else 0.16
                zh = zb + h + aw + (0.2 if hood in ("seg", "tri", "cornice") else 0.07)
                for sx in (-1, 1):
                    xs_ = x + sx * (w / 2 + aw + 0.06)
                    xextra += [xs_, xs_ + sx * 0.09]
                    if not french:
                        streaks.append((xs_, zb - 0.10, r.u(0.5, 1.4), r.u(0.45, 0.95)))
                    if hood:
                        streaks.append((xs_, zh, r.u(0.4, 1.0), r.u(0.35, 0.75)))
                if i == 0:
                    zextra += [zb - 0.10] + ([zh, zh + 0.04] if hood else [])
        if ld:
            holes.append((ld[0] - LDH, ZS + 0.3, ld[0] + LDH, ld[1] + LDT + 0.03))
        if not front and any_lit:          # at least one lit window per pair of storeys on every elevation
            for grp in ((1, 2), (3, 4)):
                cand = [wn for wn in wins if wn["lv"] in grp]
                if cand and not any(wn["lit"] for wn in cand):
                    r.pick(cand)["lit"] = True
        for wn in wins:
            Mw = Mf @ _T(wn["x"], 0, wn["zb"])
            put(wn["key"], wn["mk"], Mw)
            if wn["french"]:
                glaze(Mw, wn["w"], wn["h"], wn["lit"], r, bot=0.0, top=0.25)
            else:
                glaze(Mw, wn["w"], wn["h"], wn["lit"], r, bot=0.45, top=0.3)
        uwall = _wall_grid("upper_wall", xa, xb, ZS + 0.3, ZW - 0.7, holes, 0.55 if large else 0.42, T, xextra, zextra, caps=fb)
        objs.append(add("other", uwall, M_plaster))
        wall_paint.append((uwall, Mf, streaks, L))

        # quoins at both corners
        qi = []
        qh = ZW - 0.8 - (HG + 0.01)
        nq = max(2, int(round(qh / 0.42)))
        h1 = qh / nq
        e = 0.0 if fb else 0.03
        for k in range(nq):
            ql = 0.62 if (k % 2 == 0) == fb else 0.38
            z0 = HG + 0.01 + k * h1
            qi.append(([(-L / 2 - e, z0), (-L / 2 + ql, z0), (-L / 2 + ql, z0 + h1), (-L / 2 - e, z0 + h1)], -0.03 - r.u(0, 0.004), 0.05))
            qi.append(([(L / 2 - ql, z0), (L / 2 + e, z0), (L / 2 + e, z0 + h1), (L / 2 - ql, z0 + h1)], -0.03 - r.u(0, 0.004), 0.05))
        if front:                 # giant pilasters framing the centre bay
            for sx in (-1, 1):
                xp = sx * pitch / 2
                z0, z1 = HG + 0.01, ZW - 0.8
                qi.append(([(xp - 0.17, z0 + 0.3), (xp + 0.17, z0 + 0.3), (xp + 0.17, z1 - 0.26), (xp - 0.17, z1 - 0.26)], -0.06, 0.08))
                qi.append(([(xp - 0.22, z0), (xp + 0.22, z0), (xp + 0.22, z0 + 0.3), (xp - 0.22, z0 + 0.3)], -0.09, 0.11))
                qi.append(([(xp - 0.2, z1 - 0.26), (xp + 0.2, z1 - 0.26), (xp + 0.2, z1 - 0.14), (xp - 0.2, z1 - 0.14)], -0.085, 0.105))
                qi.append(([(xp - 0.25, z1 - 0.14), (xp + 0.25, z1 - 0.14), (xp + 0.25, z1), (xp - 0.25, z1)], -0.11, 0.13))
        objs.append(add("stone", _slabs("quoins", qi, 0.018)))

        # balconies
        if front and p.balconies != "none":
            balcony(objs, Mf, 0.0, HG, 2 * pitch + ow + 0.75, 0.8)
            if p.balconies == "all":
                for i, x in enumerate(xs):
                    if abs(i - mid) > 1:
                        balcony(objs, Mf, x, HG, ow + 0.72, 0.56)

        # dormers on the roof slope
        excl = []
        dl = [any_lit and r.chance(p.lit_ratio * litk) for _ in xs]
        zb = ZE + 0.3
        placed = []
        for i, x in enumerate(xs):
            lim = L / 2 + OE - (zb + 2.4 - ZE) / tan(ph1) - 0.85
            if abs(x) > lim:
                continue
            if front:
                if i == mid:
                    dormer(objs, Mf, "dorm_big", x, zb, 1.75, 2.15, True, any_lit, "seg", r, excl)
                else:
                    dormer(objs, Mf, "dorm", x, zb, 1.3, 1.78, False, dl[i], "tri" if abs(i - mid) % 2 == 1 else "seg", r, excl)
            elif (i + (1 if back else 0)) % 2 == 0 or nb <= 3:
                placed.append(i)
        if placed and any_lit and S == 3 and not any(dl[i] for i in placed) and r.chance(0.6):
            dl[r.pick(placed)] = True
        if placed and any_lit and S == 4 and not any(dl[i] for i in placed) and not any(wn["lit"] for wn in wins if wn["lv"] == 3):
            dl[r.pick(placed)] = True
        for i in placed:
            if p.roof == "steep" and not fb:
                yw = D / 2 + (xs[i] if side == "right" else -xs[i])
                if any((sx > 0) == (side == "right") and abs(yw - cy) < cdp / 2 + 0.95 for (sx, cy, cdp) in chim):
                    continue
            dormer(objs, Mf, "dorm", xs[i], zb, 1.3, 1.78, False, dl[i], "tri", r, excl)
        if p.roof == "steep" and fb:              # second row of small roof lights on the tall slope
            zb2 = min(ZE + 3.3, ZT - 1.9)
            if zb2 >= ZE + 2.6:
                for i, x in enumerate(xs):
                    lim = L / 2 + OE - (zb2 + 1.6 - ZE) / tan(ph1) - 0.9
                    if abs(i - mid) % 2 == (1 if front else 0) and abs(x) < lim:
                        dormer(objs, Mf, "dorm_s", x, zb2, 0.95, 1.2, False, any_lit and r.chance(p.lit_ratio * 0.6), "tri", r, excl)
        roof_side(side, L, Mf, excl)

        # rainwater pipes
        if side == "right":
            drainpipe(objs, L / 2 - 0.85, r)
        elif side == "left":
            drainpipe(objs, -L / 2 + 0.85, r)
        elif back:
            drainpipe(objs, L / 2 - 0.8, r)
            drainpipe(objs, -L / 2 + 0.8, r)
            # service door with stone steps and a transom light
            x = xs[mid]
            dwb = 1.15
            objs.append(add("stone", box("b_step1", (dwb + 0.5, 0.62, 0.16), loc=(x, -0.2, 0.08), bevel=0.014, segments=1)))
            objs.append(add("stone", box("b_step2", (dwb + 0.2, 0.5, 0.16), loc=(x, 0.0, 0.24), bevel=0.014, segments=1)))
            bx = [(x - dwb / 2 + 0.035, 0.2, 0.32 + 1.165, 0.07, 0.1, 2.33, "fr"), (x + dwb / 2 - 0.035, 0.2, 0.32 + 1.165, 0.07, 0.1, 2.33, "fl"),
                  (x, 0.2, 2.615, dwb, 0.1, 0.07, "fb"), (x, 0.2, 2.25, dwb - 0.14, 0.1, 0.07, "ftb"), (x, 0.21, 1.27, dwb - 0.14, 0.05, 1.9, "f")]
            for sx in (-1, 1):
                for (zc_, hh) in ((0.75, 0.62), (1.7, 0.9)):
                    bx.append((x + sx * (dwb / 4 - 0.035), 0.185, zc_, dwb / 2 - 0.2, 0.03, hh, "flrtb"))
            objs.append(add("wood", _boxes("b_door", bx)))
            objs.append(add("iron", cylinder("b_knob", 0.03, 0.05, loc=(x + dwb / 2 - 0.2, 0.165, 1.25), rot=(90, 0, 0), verts=8)))
            panes_dark.append((Mf, dwb - 0.1, 0.36, 0.215, 2.44, x))
        if ld:
            # stone doorcase standing on a threshold block, proud of the string course (which dies against its jambs);
            # frame, panelled leaf with a lit six-pane light, knob and kick plate
            xd, zd = ld
            zt = zd + LDT
            if zd - ld_z0 > 0.03:
                objs.append(add("stone", box("ld_threshold", (2 * LDO + 0.08, 0.32, zd - ld_z0), loc=(xd, 0.08, (zd + ld_z0) / 2), bevel=0.014, segments=1)))
            for sx in (-1, 1):
                objs.append(add("stone", box("ld_jamb", (LDO - LDW, 0.42, LDT), loc=(xd + sx * (LDO + LDW) / 2, 0.03, zd + LDT / 2), bevel=0.012, segments=1)))
                objs.append(add("stone", box("ld_base", (LDO - LDW + 0.03, 0.45, 0.2), loc=(xd + sx * (LDO + LDW) / 2, 0.03, zd + 0.1), bevel=0.012, segments=1)))
            objs.append(add("stone", box("ld_head", (2 * LDO, 0.42, 0.1), loc=(xd, 0.03, zt + 0.05), bevel=0.012, segments=1)))
            objs.append(add("stone", box("ld_cap", (2 * LDO + 0.05, 0.47, 0.035), loc=(xd, 0.03, zt + 0.0825), bevel=0.008, segments=1)))
            y_, fw_ = 0.2, 0.055
            lw_ = 2 * LDW - 2 * fw_
            z0l, z1l = zd + 0.012, zt - fw_
            gz0, gz1 = z0l + 1.02, z1l - 0.12
            bx = [(xd - LDW + fw_ / 2, y_, zd + LDT / 2, fw_, 0.1, LDT, "fr"), (xd + LDW - fw_ / 2, y_, zd + LDT / 2, fw_, 0.1, LDT, "fl"),
                  (xd, y_, zt - fw_ / 2, lw_, 0.1, fw_, "fb"), (xd, y_ + 0.035, z0l + 0.56, lw_ - 0.2, 0.02, 0.7, "f"),
                  (xd, y_ + 0.015, z0l + 0.56, lw_ - 0.36, 0.03, 0.5, "flrtb"), (xd, y_ + 0.02, (gz0 + gz1) / 2, 0.024, 0.03, gz1 - gz0, "flr")]
            for sx in (-1, 1):
                bx.append((xd + sx * (lw_ / 2 - 0.055), y_ + 0.02, (z0l + z1l) / 2, 0.11, 0.05, z1l - z0l, "flr"))
            for (zc_, hh) in ((z0l + 0.11, 0.22), (z0l + 0.965, 0.11), (z1l - 0.06, 0.12)):
                bx.append((xd, y_ + 0.02, zc_, lw_ - 0.22, 0.05, hh, "ftb"))
            for k in (1, 2):
                bx.append((xd, y_ + 0.02, gz0 + (gz1 - gz0) * k / 3, lw_ - 0.22, 0.03, 0.022, "ftb"))
            objs.append(add("wood", _boxes("ld_door", bx)))
            objs.append(add("iron", _boxes("ld_kick", [(xd, y_ - 0.008, z0l + 0.11, lw_ - 0.06, 0.004, 0.16, "flrtb"),
                                                         (xd + lw_ / 2 - 0.055, y_ - 0.012, z0l + 1.0, 0.05, 0.006, 0.2, "flrtb")])))
            objs.append(add("iron", sphere("ld_knob", 0.032, loc=(xd + lw_ / 2 - 0.055, y_ - 0.05, z0l + 1.0), segments=8, rings=5)))
            objs.append(add("iron", cylinder("ld_knob_neck", 0.012, 0.05, loc=(xd + lw_ / 2 - 0.055, y_ - 0.025, z0l + 1.0), rot=(90, 0, 0), verts=6)))
            gx = lw_ / 2 - 0.11
            Q["curt"].append(wq(Mf, [(xd - gx, y_ + 0.03, gz0), (xd + gx, y_ + 0.03, gz0), (xd + gx, y_ + 0.03, gz1), (xd - gx, y_ + 0.03, gz1)]))
            Q["curt_uv"].append([(4.2, 2.1), (4.8, 2.1), (4.8, 2.9), (4.2, 2.9)])
            LD.append((-xd, zd, zt))
        _xf(objs, Mf)

    # chimney stacks are sited first so the side dormers of a steep roof can keep clear of them
    rc = rng.fork("chimneys")
    nch = int(p.chimneys)
    chim = []
    for si, sx in enumerate((-1, 1)):
        m = [(nch + 1) // 2, nch // 2][si]
        for k in range(m):
            chim.append((sx, D * (k + 1) / (m + 1) + rc.jitter(0, 0.25), rc.pick([1.1, 1.3, 1.5])))

    # ---- the four facades
    LDW, LDH, LDO, LDT = 0.42, 0.47, 0.52, 1.95      # landing door: half clear width, half wall opening, half doorcase, clear height
    LD = []                                          # [(x in the hotel frame, sill z, head z)] of the door actually built
    FM = {"front": I4, "right": _T(W / 2, D / 2, 0) @ _RZ(90), "back": _T(0, D, 0) @ _RZ(180), "left": _T(-W / 2, D / 2, 0) @ _RZ(270)}
    for side, L in (("front", W), ("right", D), ("back", W), ("left", D)):
        facade(side, L, FM[side], rng.fork("facade_" + side))

    # ---- stamp windows, glazing and rooms
    for key, k in kits.items():
        G["other"].append(instances_to_mesh("win_" + key, k["trim"], k["mats"], keep_base=False))
    rp = rng.fork("panes")
    Vp, Fp, UVp = [], [], []
    for (M, w, h, y, zc, xc) in panes_dark:
        b = len(Vp)
        ku, kv = int(rp.u(0, 9)), int(rp.u(0, 9))
        for (sx, sz) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            Vp.append(tuple(M @ Vector((xc + sx * w / 2, y, zc + sz * h / 2))))
            UVp.append((ku + (sx + 1) / 2, kv + (sz + 1) / 2))
        Fp.append((b, b + 1, b + 2, b + 3))
    if Vp:
        o = mesh_object("glass_dark", Vp, Fp)
        _set_uv(o, UVp)
        add("other", o, M_dark)

    # ---- entrance: steps, door, lobby, canopy, lanterns (front, world space)
    r = rng.fork("entrance")
    R = dw / 2
    zsd = 2.65
    sw_ = dw + 1.7
    for k, (zz, yy) in enumerate(((0.15, -1.22), (0.30, -0.89), (0.45, -0.56))):
        cuts = [-sw_ / 2, -sw_ / 6 + r.jitter(0, 0.12), sw_ / 6 + r.jitter(0, 0.12), sw_ / 2]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            st = box("step", (c1 - c0 - 0.004, 0.3 - yy, 0.15 + (0.02 if k else 0.0)), loc=((c0 + c1) / 2, (0.3 + yy) / 2, zz - 0.075 - (0.01 if k else 0.0)),
                     bevel=0.016, segments=1)
            add("stone", st)
            paint(st, wear=lambda co, n: np.exp(-(co[:, 0] / 0.75) ** 2) * (n[:, 2] > 0.5) * 0.75,
                  grime=lambda co, n: np.clip(np.abs(co[:, 0]) / (sw_ / 2) - 0.45, 0, 1) * 0.8 * (0.4 + age))
    for sx in (-1, 1):           # cheek blocks with covered urns
        cxk = sx * (sw_ / 2 + 0.2)
        add("stone", box("step_cheek", (0.42, 1.38, 0.5), loc=(cxk, -0.63, 0.25), bevel=0.016, segments=1))
        add("stone", box("step_cheek_cap", (0.52, 1.46, 0.1), loc=(cxk, -0.64, 0.55), bevel=0.025, segments=1))
        add("stone", box("urn_plinth", (0.36, 0.36, 0.1), loc=(cxk, -1.05, 0.65), bevel=0.012, segments=1))
        add("stone", lathe("urn", [(0.0, 0.0), (0.15, 0.0), (0.15, 0.04), (0.1, 0.06), (0.06, 0.12), (0.075, 0.15), (0.06, 0.17), (0.16, 0.26),
                                   (0.215, 0.36), (0.225, 0.44), (0.21, 0.5), (0.15, 0.535), (0.17, 0.55), (0.2, 0.565), (0.2, 0.585),
                                   (0.14, 0.62), (0.07, 0.66), (0.035, 0.68), (0.05, 0.71), (0.045, 0.74), (0.0, 0.765)], segments=12,
                           loc=(cxk, -1.05, 0.7)))
    # door frame, transom, fanlight
    yd = 0.2
    path = [(R - 0.04, yd, FL), (R - 0.04, yd, zsd)] + [((R - 0.04) * cos(pi * k / 12), yd, zsd + (R - 0.04) * sin(pi * k / 12)) for k in range(1, 12)]
    path += [(-R + 0.04, yd, zsd), (-R + 0.04, yd, FL)]
    add("wood", sweep("door_frame", path, [(-0.04, -0.06), (0.04, -0.06), (0.04, 0.06), (-0.04, 0.06)], smooth_path=False, up=(0, 1, 0)))
    add("wood", box("door_transom", (dw - 0.1, 0.14, 0.1), loc=(0, yd - 0.01, zsd), bevel=0.006, segments=1))
    for a in (30, 60, 90, 120, 150):
        ar = radians(a)
        add("wood", box("fan_spoke", (0.03, 0.035, R - 0.3), loc=((0.26 + (R - 0.3) / 2) * cos(ar), yd, zsd + 0.04 + (0.26 + (R - 0.34) / 2) * sin(ar)), rot=(0, 90 - a, 0)))
    add("wood", sweep("fan_hub", [(0.27 * cos(pi * k / 8), yd, zsd + 0.04 + 0.27 * sin(pi * k / 8)) for k in range(9)],
                      [(-0.016, -0.02), (0.016, -0.02), (0.016, 0.02), (-0.016, 0.02)], smooth_path=False, up=(0, 1, 0)))
    # leaves: panelled below, glazed above behind an iron grille
    lw = (dw - 0.16) / 2
    z0, z1 = FL + 0.01, zsd - 0.05
    wbx, ibx = [], []
    for sx in (-1, 1):
        cx = sx * (lw / 2 + 0.002)
        for ex in (-1, 1):
            wbx.append((cx + ex * (lw / 2 - 0.055), yd, (z0 + z1) / 2, 0.11, 0.055, z1 - z0, "flr"))
        for (zc_, hh) in ((z0 + 0.11, 0.22), (z0 + 0.95, 0.12), (z1 - 0.06, 0.12)):
            wbx.append((cx, yd, zc_, lw - 0.22, 0.055, hh, "ftb"))
        wbx.append((cx, yd + 0.005, z0 + 0.555, lw - 0.22, 0.025, 0.67, "f"))
        add("wood", box("leaf_fielded", (lw - 0.34, 0.03, 0.5), loc=(cx, yd - 0.012, z0 + 0.555), bevel=0.012, segments=1))
        gz0, gz1 = z0 + 1.01, z1 - 0.12
        for k in range(1, 4):
            ibx.append((cx - (lw - 0.22) / 2 + (lw - 0.22) * k / 4, yd - 0.02, (gz0 + gz1) / 2, 0.012, 0.012, gz1 - gz0, "flr"))
        for k in range(4):
            add("iron", torus("grille_ring", (lw - 0.22) / 8 - 0.008, 0.006, loc=(cx + (k - 1.5) * (lw - 0.22) / 4, yd - 0.02, gz1 - (lw - 0.22) / 8),
                              rot=(90, 0, 0), major_seg=10, minor_seg=3))
        ibx.append((cx, yd - 0.02, gz1 - (lw - 0.22) / 4 - 0.004, lw - 0.22, 0.012, 0.014, "ftb"))
        ibx.append((cx, yd - 0.02, gz0 + 0.3, lw - 0.22, 0.012, 0.014, "ftb"))
        hx = sx * 0.1
        add("iron", cylinder("pull", 0.014, 0.5, loc=(hx, yd - 0.085, z0 + 1.2), verts=6))
        for dz in (-0.2, 0.2):
            ibx.append((hx, yd - 0.057, z0 + 1.2 + dz, 0.02, 0.06, 0.02, "lrtb"))
        ibx.append((cx, yd - 0.03, z0 + 0.1, lw - 0.06, 0.004, 0.16, "flrtb"))
    wbx.append((0, yd - 0.035, (z0 + z1) / 2, 0.045, 0.02, z1 - z0, "flr"))
    add("wood", _boxes("door_leaves", wbx))
    add("iron", _boxes("door_iron", ibx))
    socket("door", "attach", (0, 0, FL), role="door")
    # the lobby behind: glowing walls, an inner glazed screen, a reception desk and a chandelier
    xa_, xb_, y0_, y1_, zc_l = -dw / 2 - 0.3, dw / 2 + 0.3, T, T + 1.3, min(zsd + R + 0.22, ZS - 0.06)
    Q["room"] += [[(xa_, y1_, FL), (xb_, y1_, FL), (xb_, y1_, zc_l), (xa_, y1_, zc_l)], [(xa_, y0_, zc_l), (xa_, y1_, zc_l), (xb_, y1_, zc_l), (xb_, y0_, zc_l)]]
    Q["int"] += [[(xa_, y0_, FL), (xa_, y1_, FL), (xa_, y1_, zc_l), (xa_, y0_, zc_l)], [(xb_, y0_, FL), (xb_, y0_, zc_l), (xb_, y1_, zc_l), (xb_, y1_, FL)]]
    Q["floor"].append([(xa_, y0_, FL), (xb_, y0_, FL), (xb_, y1_, FL), (xa_, y1_, FL)])
    Q["curt"].append([(-0.5, y1_ - 0.02, FL), (0.5, y1_ - 0.02, FL), (0.5, y1_ - 0.02, FL + 2.15), (-0.5, y1_ - 0.02, FL + 2.15)])
    Q["curt_uv"].append([(3.02, 5.02), (3.98, 5.02), (3.98, 5.98), (3.02, 5.98)])
    for (qa, qb, qz) in ((-dw / 2 - 0.3, -0.62, 1.0), (0.62, dw / 2 + 0.3, 1.0), (-0.62, -0.5, 2.3), (0.5, 0.62, 2.3), (-0.62, 0.62, 2.3)):
        zlo = FL + 2.15 if qb - qa > 1.0 else FL
        Q["int"].append([(qa, y1_ - 0.03, zlo), (qb, y1_ - 0.03, zlo), (qb, y1_ - 0.03, FL + qz), (qa, y1_ - 0.03, FL + qz)])
    for k in range(6):
        inst["bulb"].append(_T(0.2 * cos(pi * k / 3), T + 0.55 + 0.2 * sin(pi * k / 3), zsd + 0.42) @ Matrix.Diagonal((0.9, 0.9, 0.9, 1)))
    ch_ = [cylinder("chandelier_rod", 0.01, zc_l - zsd - 0.5, loc=(0, T + 0.55, (zc_l + zsd + 0.5) / 2), verts=5),
           torus("chandelier_ring", 0.2, 0.012, loc=(0, T + 0.55, zsd + 0.38), major_seg=12, minor_seg=3)]
    for o in ch_:
        add("iron", o)

    # glazed iron canopy with marquee bulbs
    cp = float(p.canopy_reach)
    cw = clamp(dw + 1.9, 3.6, 4.4)
    zr = ZS - 0.06
    zf = zr - 0.13 * cp
    th = atan2(zr - zf, cp)

    def sloped(n, sx_, sz_, x, dz, kind="iron", m=None, y0=0.0, y1=None):
        y1 = cp if y1 is None else y1
        ym = (y0 + y1) / 2
        return add(kind, box(n, (sx_, (y1 - y0) / cos(th), sz_), loc=(x, -ym, zr - (zr - zf) * ym / cp + dz), rot=(degrees(th), 0, 0)), m)
    add("iron", box("can_wallplate", (cw, 0.05, 0.16), loc=(0, -0.02, zr - 0.05)))
    for sx in (-1, 1):
        sloped("can_side", 0.05, 0.13, sx * (cw / 2 - 0.025), -0.05)
    add("iron", box("can_fascia", (cw + 0.04, 0.05, 0.24), loc=(0, -cp, zf - 0.09), bevel=0.006, segments=1))
    add("iron", box("can_fascia_top", (cw + 0.1, 0.09, 0.025), loc=(0, -cp, zf + 0.04)))
    add("iron", box("can_fascia_bot", (cw + 0.08, 0.075, 0.02), loc=(0, -cp, zf - 0.215)))
    nraf = max(4, int(round(cw / 0.52)))
    for k in range(1, nraf):
        sloped("can_rafter", 0.028, 0.055, -cw / 2 + cw * k / nraf, -0.02)
    for t in (0.36, 0.7):
        add("iron", box("can_purlin", (cw - 0.06, 0.03, 0.035), loc=(0, -cp * t, zr - (zr - zf) * t - 0.045)))
    sloped("can_glass", cw - 0.08, 0.008, 0.0, -0.012, "other", M_dark, 0.02, cp - 0.02)
    xbr = dw / 2 + 0.34
    for sx in (-1, 1):
        xb_ = sx * xbr
        sloped("can_cantilever", 0.045, 0.085, xb_, -0.07)
        add("iron", box("can_wallbar", (0.045, 0.035, 1.35), loc=(xb_, -0.018, zr - 0.72)))
        pts = bezier_path((xb_, -0.03, zr - 1.36), (xb_, -cp * 0.42, zr - 1.3), (xb_, -cp * 0.86, zr - 0.75), (xb_, -cp + 0.14, zf - 0.1), n=10)
        add("iron", sweep("can_brace", pts, [(-0.022, -0.007), (0.022, -0.007), (0.022, 0.007), (-0.022, 0.007)], up=(1, 0, 0), smooth_path=False))
        sc = spiral_path(0.035, 0.2, 1.6, n=20, a0_deg=-30, plane="YZ", center=(xb_, -0.36, zr - 0.52))
        add("iron", sweep("can_scroll", sc, [(-0.02, -0.006), (0.02, -0.006), (0.02, 0.006), (-0.02, 0.006)], up=(1, 0, 0), smooth_path=False,
                          scale_fn=lambda t: 0.6 + 0.4 * t))
        sc2 = spiral_path(0.025, 0.11, 1.4, n=14, a0_deg=150, plane="YZ", center=(xb_, -cp * 0.66, zr - 0.44))
        add("iron", sweep("can_scroll2", sc2, [(-0.02, -0.006), (0.02, -0.006), (0.02, 0.006), (-0.02, 0.006)], up=(1, 0, 0), smooth_path=False,
                          scale_fn=lambda t: 0.6 + 0.4 * t))
    nbu = int(round(cw / 0.2))
    for k in range(nbu + 1):
        inst["bulb"].append(_T(-cw / 2 + 0.03 + (cw - 0.06) * k / nbu, -cp - 0.05, zf - 0.09))
    nsb = int(round((cp - 0.25) / 0.22))
    for sx in (-1, 1):
        for k in range(nsb):
            yy = 0.22 + (cp - 0.4) * k / max(nsb - 1, 1)
            inst["bulb"].append(_T(sx * (cw / 2 + 0.012), -yy, zr - (zr - zf) * yy / cp - 0.07))
    if p.canopy_text.strip():
        t = _fit_text("canopy_letters", p.canopy_text.strip(), 0.3, 0.0, zf + 0.26, -cp + 0.02, font="sans", depth=0.05, maxw=cw - 0.5)
        add("other", t, M_neon)
        bb = world_bbox([t])
        tw = bb[1].x - bb[0].x
        lb = [(0, -cp + 0.035, zz, tw + 0.24, 0.02, 0.02, "fktb") for zz in (zf + 0.085, zf + 0.435)]
        nst = max(2, int(tw / 0.45))
        for k in range(nst + 1):
            lb.append((-tw / 2 - 0.1 + (tw + 0.2) * k / nst, -cp + 0.04, zf + 0.26, 0.016, 0.016, 0.42, "fklr"))
        add("iron", _boxes("can_letter_frame", lb))
    # hexagonal pendant lantern under the canopy: the fixture of the entrance light
    yp = -cp * 0.58
    zpn = zr - (zr - zf) * 0.58 - 0.07
    pl_ = [cylinder("pend_rod", 0.009, 0.2, loc=(0, yp, zpn - 0.1), verts=5),
           cone("pend_cap", 0.17, 0.035, 0.1, loc=(0, yp, zpn - 0.25), verts=6),
           cylinder("pend_rim", 0.15, 0.02, loc=(0, yp, zpn - 0.31), verts=6),
           cylinder("pend_base", 0.1, 0.02, loc=(0, yp, zpn - 0.66), verts=6),
           sphere("pend_drop", 0.028, loc=(0, yp, zpn - 0.69), segments=6, rings=4)]
    for k in range(6):
        a = pi / 3 * k
        pl_.append(sweep("pend_bar", [(0.14 * cos(a), yp + 0.14 * sin(a), zpn - 0.31), (0.095 * cos(a), yp + 0.095 * sin(a), zpn - 0.66)], 0.007, sides=4))
    for o in pl_:
        add("iron", o)
    # CLEARANCE RULE: the glowing reflector has no top (it lay in the plane of the underside of the rim, hidden by both)
    add("other", lod_faces(cone("pend_reflector", 0.03, 0.135, 0.05, loc=(0, yp, zpn - 0.345), verts=6), drop=(0, 0, 1), above=-1, cos=0.9), M_room)
    add("iron", cylinder("pend_burner", 0.022, 0.1, loc=(0, yp, zpn - 0.6), verts=6))
    add("other", sphere("pend_mantle", 0.05, loc=(0, yp, zpn - 0.47), segments=6, rings=4, scale=(1, 1, 1.7)), M_bulb)
    light("canopy", (0, yp, zpn - 0.47), color="#ffab55", intensity=CD_CANOPY, range=10, flicker=0.03, shadow_near=0.4)

    # bracket lanterns: a pair on the piers of the ground storey and a pair on the piers between the first-floor windows
    def lantern(M, name, reach, cd, rng_):
        o = [box("lan_plate", (0.1, 0.016, 0.46), loc=(0, -0.008, 0))]
        pts = bezier_path((0, -0.015, -0.16), (0, -reach * 0.6, -0.2), (0, -reach - 0.04, 0.1), (0, -reach, 0.34), n=8)
        o.append(sweep("lan_arm", pts, 0.013, sides=5, smooth_path=False))
        o.append(sweep("lan_scroll", spiral_path(0.02, 0.1, 1.5, n=14, a0_deg=200, plane="YZ", center=(0, -reach * 0.42, 0.06)),
                       [(-0.015, -0.005), (0.015, -0.005), (0.015, 0.005), (-0.015, 0.005)], up=(1, 0, 0), smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t))
        o.append(sweep("lan_stay", [(0, -0.015, 0.2), (0, -reach * 0.65, 0.18), (0, -reach + 0.02, 0.3)], 0.008, sides=4, smooth_path=False))
        yl, zl = -reach, 0.27
        o.append(cylinder("lan_hook", 0.008, 0.08, loc=(0, yl, zl + 0.03), verts=5))
        o.append(cone("lan_roof", 0.16, 0.03, 0.1, loc=(0, yl, zl - 0.05), rot=(0, 0, 45), verts=4))
        o.append(box("lan_top", (0.21, 0.21, 0.016), loc=(0, yl, zl - 0.105)))
        o.append(box("lan_bottom", (0.15, 0.15, 0.02), loc=(0, yl, zl - 0.44)))
        o.append(sphere("lan_drop", 0.022, loc=(0, yl, zl - 0.465), segments=6, rings=4))
        for sx in (-1, 1):
            for sy in (-1, 1):
                o.append(sweep("lan_bar", [(sx * 0.098, yl + sy * 0.098, zl - 0.105), (sx * 0.07, yl + sy * 0.07, zl - 0.44)], 0.007, sides=4))
        for q in o:
            add("iron", q)
        o.append(cylinder("lan_burner", 0.018, 0.1, loc=(0, yl, zl - 0.385), verts=6))
        add("iron", o[-1])
        gl = add("other", cone("lan_reflector", 0.025, 0.125, 0.045, loc=(0, yl, zl - 0.14), rot=(0, 0, 45), verts=4), M_room)
        mn = add("other", sphere("lan_mantle", 0.04, loc=(0, yl, zl - 0.27), segments=6, rings=4, scale=(1, 1, 1.7)), M_bulb)
        _xf(o + [gl, mn], M)
        if cd > 0:
            light(name, tuple(M @ Vector((0, yl, zl - 0.28))), color="#ffab55", intensity=cd, range=rng_, flicker=0.12, shadow_near=0.3)
    xl_ = 1.5 * pitch_f if nbf >= 5 else pitch_f / 2
    lantern(_T(-xl_, 0, 2.85), "lantern_l", 0.5, CD_LANTERN, 9)
    lantern(_T(xl_, 0, 2.85), "lantern_r", 0.5, CD_LANTERN, 9)
    xu_ = (2.5 if nbf >= 7 else 1.5) * pitch_f
    lantern(_T(-xu_, 0, HG + 2.2), "lantern_upper_l", 0.68, CD_UPPER, 11)
    lantern(_T(xu_, 0, HG + 2.2), "lantern_upper_r", 0.68, CD_UPPER, 11)
    for (xw, zd, zt) in LD:       # a lantern over the landing door: the only light on the canal side
        lantern(_T(xw, D, zt + 0.72) @ _RZ(180), "landing_lantern", 0.38, float(p.landing_lantern_cd), 6)
        socket("landing_door", "attach", (xw, D, zd), rot=(0, 0, 180), role="door", width=2 * LDW, height=LDT)
    # canal-side sign position: the clear band between the first-floor lintels and the sills above, centred on a pier
    xsb = pitch_f / 2 * (-1 if (LD and LD[0][0] > 0) else 1)
    socket("sign_back", "attach", (xsb, D, HG + 3.34), rot=(0, 0, 180), role="sign", side="back", clear_w=W - 2.0 - 2 * abs(xsb), clear_h=0.95)

    # ---- mouldings round the whole block
    sc_ = add("stone", _ring("string_course", -W / 2, W / 2, 0, D, [(0.0, ZS - 0.02), (0.06, ZS - 0.02), (0.06, ZS + 0.2), (0.08, ZS + 0.22), (0.1, ZS + 0.28),
                                                               (0.17, ZS + 0.31), (0.17, ZS + 0.37), (0.04, ZS + 0.42), (0.0, ZS + 0.425)], seg=1.4))
    for (xw, zd, zt) in LD:
        _gap(sc_, xw - LDO + 0.02, xw + LDO - 0.02, D + 0.001)
    zt_ = HG + (S - 2) * HU + 0.77
    add("stone", _ring("sill_band", -W / 2, W / 2, 0, D, [(0.0, zt_ - 0.07), (0.035, zt_ - 0.07), (0.05, zt_ - 0.05), (0.05, zt_ + 0.02), (0.0, zt_ + 0.05)], seg=2.5))
    add("stone", _ring("cornice", -W / 2, W / 2, 0, D, [
        (0.0, ZW - 0.82), (0.05, ZW - 0.82), (0.05, ZW - 0.75), (0.03, ZW - 0.74), (0.03, ZW - 0.45), (0.06, ZW - 0.43), (0.09, ZW - 0.39), (0.09, ZW - 0.37),
        (0.12, ZW - 0.36), (0.4, ZW - 0.355), (0.4, ZW - 0.26), (0.42, ZW - 0.26), (0.42, ZW - 0.15), (0.45, ZW - 0.13), (0.49, ZW - 0.07), (0.5, ZW - 0.02),
        (0.5, ZW + 0.0), (0.46, ZW + 0.045), (0.2, ZW + 0.075), (0.1, ZW + 0.075)], seg=1.4))
    for (L_, M) in ((W, FM["front"]), (D, FM["right"]), (W, FM["back"]), (D, FM["left"])):
        n = int(round((L_ + 0.5) / 0.52))
        for k in range(n + 1):
            inst["mod"].append(M @ _T(-L_ / 2 - 0.25 + (L_ + 0.5) * k / n, -0.25, ZW - 0.31))

    # ---- roof dressings: lead flat with rolls, lead hip rolls and mansard apron, iron gutter and cresting
    dx0, dx1, dy0, dy1 = -W / 2 - OT, W / 2 + OT, -OT, D + OT
    add("lead", mesh_object("roof_deck", [(dx0, dy0, ZT), (dx1, dy0, ZT), (dx1, dy1, ZT), (dx0, dy1, ZT)], [(0, 1, 2, 3)]))
    nroll = max(2, int(round((dx1 - dx0) / 0.62)))
    add("lead", _boxes("deck_rolls", [(dx0 + (dx1 - dx0) * k / nroll, (dy0 + dy1) / 2, ZT + 0.02, 0.05, dy1 - dy0 - 0.06, 0.045, "lrt") for k in range(1, nroll)]))
    add("lead", _ring("deck_curb", dx0, dx1, dy0, dy1, [(0.09, ZT - 0.1), (0.07, ZT + 0.0), (0.06, ZT + 0.04), (0.02, ZT + 0.07), (-0.06, ZT + 0.05)], seg=4.0))
    HIP = [(OE, ZE)]
    if p.roof == "mansard":
        c72, s72 = cos(ph1), sin(ph1)
        apron = [(OC + 0.2 * c72 + 0.03 * s72, ZC - 0.2 * s72 + 0.03 * c72), (OC + 0.02 * c72 + 0.04 * s72, ZC - 0.02 * s72 + 0.04 * c72)]
        apron += [(OC + 0.01 + 0.065 * cos(radians(a)), ZC + 0.045 + 0.065 * sin(radians(a))) for a in (-25, 20, 65, 110, 150)]
        apron += [(OC - 0.3, ZC + 0.04 + 0.3 * tan(radians(22.0)) + 0.03)]
        add("lead", _ring("mansard_roll", -W / 2, W / 2, 0, D, apron, seg=4.0))
        HIP += [(OC, ZC), (OC, ZC + 0.06)]
    HIP.append((OT, ZT + 0.02))
    gut = add("iron", _ring("eaves_gutter", -W / 2, W / 2, 0, D, [(0.1, ZW + 0.07), (0.44, ZW + 0.05), (0.47, ZW + 0.12), (0.44, ZW + 0.125), (0.42, ZW + 0.075),
                                                           (0.2, ZW + 0.09), (0.12, ZW + 0.13)], seg=4.0))
    for sx in (-1, 1):
        for sy in (-1, 1):
            hp = [(sx * (W / 2 + o), (D / 2) + sy * (D / 2 + o), z + 0.035) for (o, z) in HIP]
            add("lead", sweep("hip_roll", hp, 0.06, sides=6, smooth_path=False))
            for (o, z) in HIP[:1]:        # rolled lead foot at the eaves
                add("lead", sphere("hip_foot", 0.085, loc=(sx * (W / 2 + o), (D / 2) + sy * (D / 2 + o), z + 0.05), segments=6, rings=4))
    crest = []
    for (xa_, ya_, xb__, yb_, rz) in ((dx0, dy0, dx1, dy0, 0), (dx1, dy0, dx1, dy1, 90), (dx1, dy1, dx0, dy1, 180), (dx0, dy1, dx0, dy0, 270)):
        L_ = hypot(xb__ - xa_, yb_ - ya_)
        n = max(2, int(round(L_ / 0.3)))
        for k in range(n + 1):
            t = k / n
            inst["spear"].append(_T(xa_ + (xb__ - xa_) * t, ya_ + (yb_ - ya_) * t, ZT + 0.05) @ _RZ(rz) @ Matrix.Diagonal((1, 1, 1.0 if k % 2 == 0 else 0.72, 1)))
        for zz in (ZT + 0.1, ZT + 0.22):
            crest.append(((xa_ + xb__) / 2, (ya_ + yb_) / 2, zz) + ((L_, 0.014, 0.018) if rz in (0, 180) else (0.014, L_, 0.018)) + ("fktb" if rz in (0, 180) else "lrtb",))
    add("iron", _boxes("crest_rails", crest))

    # ---- chimney stacks
    PROF = [(OE, ZE), (OC, ZC), (OT, ZT)] if p.roof == "mansard" else [(OE, ZE), (OT, ZT)]

    def roof_z(inset):             # height of the roof surface at a horizontal distance inside the wall plane
        o = -inset
        for (o0, z0), (o1, z1) in zip(PROF[:-1], PROF[1:]):
            if o1 <= o <= o0 and o0 > o1:
                return z0 + (z1 - z0) * (o0 - o) / (o0 - o1)
        return ZT if o < PROF[-1][0] else ZE

    stacks = []
    for (sx, cy, cdp) in chim:
        cx = sx * (W / 2 - 1.55)
        cwd = 0.74
        ins_lo = max(0.0, min(W / 2 - abs(cx) - cwd / 2, cy - cdp / 2, D - cy - cdp / 2))
        ins_hi = min(W / 2 - abs(cx) + cwd / 2, D / 2)
        z_lo, z_hi = roof_z(ins_lo), roof_z(ins_hi)
        zb = z_lo - 0.4
        zt = (ZT + rc.u(1.0, 1.5)) if p.roof == "mansard" else min(ZT + 1.2, z_hi + rc.u(2.3, 2.9))
        stack = _ring("chimney_stack", cx - cwd / 2, cx + cwd / 2, cy - cdp / 2, cy + cdp / 2, [(0.0, zb + (zt - zb) * j / 4) for j in range(5)], seg=0.6)
        assign(stack, M_plaster)
        stacks.append((stack, zb, zt))
        add("stone", _ring("chimney_cap", cx - cwd / 2, cx + cwd / 2, cy - cdp / 2, cy + cdp / 2,
                           [(0.0, zt - 0.34), (0.03, zt - 0.32), (0.03, zt - 0.25), (0.0, zt - 0.23), (0.0, zt - 0.14), (0.05, zt - 0.1), (0.09, zt - 0.08),
                            (0.09, zt + 0.0), (0.03, zt + 0.04), (-0.12, zt + 0.06)], seg=3.0))
        add("stone", box("chimney_top", (cwd - 0.2, cdp - 0.2, 0.06), loc=(cx, cy, zt + 0.045)))
        add("lead", _ring("chimney_flashing", cx - cwd / 2, cx + cwd / 2, cy - cdp / 2, cy + cdp / 2,
                          [(0.17, z_lo - 0.1), (0.016, z_hi + 0.24), (0.0, z_hi + 0.24)], seg=3.0))
        npot = 2 + int(cdp > 1.2) + int(cdp > 1.4)
        for j in range(npot):
            py = cy + (j - (npot - 1) / 2) * (cdp - 0.36) / max(npot - 1, 1)
            hpot = rc.u(0.42, 0.75)
            rp_ = rc.u(0.095, 0.12)
            pot = lathe("chimney_pot", [(rp_ + 0.03, 0.0), (rp_ + 0.035, 0.04), (rp_ + 0.005, 0.07), (rp_ - 0.01, hpot * 0.8), (rp_ + 0.012, hpot * 0.86),
                                        (rp_ + 0.018, hpot * 0.93), (rp_ + 0.004, hpot), (rp_ - 0.02, hpot), (rp_ - 0.02, hpot - 0.1)],
                        segments=8, loc=(cx + rc.jitter(0, 0.02), py, zt + 0.07))
            add("stone", pot)
            paint(pot, grime=lambda co, n, zz=zt + 0.07, hh=hpot: np.clip((co[:, 2] - zz) / hh, 0, 1) * 0.9)

    # ---- landmark blade sign off one front corner: a deep-blue back-lit enamel face with gold channel letters, a chaser
    # ---- border of bulbs and a neon tube on both faces, under a big glowing crown outlined in bulbs; scrolled brackets
    if p.sign_side != "none" and p.sign_text.strip():
        sgn = -1 if p.sign_side == "left" else 1
        letters = [c for c in p.sign_text.strip().upper() if c != " "][:9]
        nl = len(letters)
        PH = radians(35.0)                           # the blade crosses the corner at 35 deg to the front wall, facing the square
        SA, SC, bt = 0.25, 0.3, 0.2                  # its inner end: 0.25 m in from the corner, 0.3 m proud of the wall; case thickness
        Ms = _T(sgn * (W / 2 - SA), -SC, 0.0) @ _RZ(180 + 35 if sgn < 0 else 360 - 35)      # local +x runs out along the blade
        xe = (0.695 + SA - sin(PH) * (bt / 2 + 0.065)) / cos(PH)      # outer edge: the sign keeps inside 0.7 m beyond the side wall
        qa = Ms.inverted() @ Vector((sgn * (W / 2 + 0.05), -0.05, 0.0))        # the arris of the quoins in this frame
        wy = 1.0 if qa.y > 0 else -1.0
        x0 = 0.0
        bw_ = xe - x0
        hc = 0.92                                    # height of the crown
        ztop = ZW - 0.98 - hc
        bh = min(clamp(ztop - (HG + 0.2), 3.0, 5.0), nl * 0.95 + 0.4)
        zb = ztop - bh
        lh = (bh - 0.36) / nl
        cap = min(lh * 0.76, 0.66)
        xc0 = x0 + bw_ / 2
        rw = 0.075                                   # width of the bulb rim
        so = [add("iron", box("sign_case", (bw_, bt, bh), loc=(xc0, 0, zb + bh / 2), bevel=0.012, segments=1))]
        rim, faces, gold = [], [], []
        for fy in (-1, 1):
            yy = fy * (bt / 2 + 0.008)
            ff = "flrtb" if fy < 0 else "klrtb"
            rim += [(xc0, yy, zb + rw / 2, bw_, 0.022, rw, ff), (xc0, yy, zb + bh - rw / 2, bw_, 0.022, rw, ff),
                    (x0 + rw / 2, yy, zb + bh / 2, rw, 0.022, bh - 2 * rw, ff), (x0 + bw_ - rw / 2, yy, zb + bh / 2, rw, 0.022, bh - 2 * rw, ff)]
            faces.append((xc0, fy * (bt / 2 + 0.002), zb + bh / 2, bw_ - 2 * rw, 0.004, bh - 2 * rw, "f" if fy < 0 else "k"))
            rr_ = [(xc0 + px, fy * (bt / 2 + 0.017), zb + bh / 2 + pz) for (px, pz) in rounded_rect(bw_ - 2 * rw - 0.09, bh - 2 * rw - 0.09, 0.06, 3)]
            so.append(add("other", sweep("sign_border_tube", rr_, 0.012, sides=5, closed=True, smooth_path=False), M_neon))
            for k, ch in enumerate(letters):
                zc = zb + bh - 0.18 - lh * (k + 0.5)
                t = _fit_text("sign_letter", ch, cap, xc0, zc, fy * (bt / 2), back=fy > 0, font="sans", depth=0.045, maxw=bw_ - 2 * rw - 0.2, wide=1.3)
                # CLEARANCE RULE: the glowing letters have no backs (they lay on the face of the case, under the enamel face)
                lod_faces(t, drop=(0, -fy, 0), above=-1, cos=0.9)
                gold.append(add("other", t, M_gold))
            hx_, hz_ = (bw_ - rw) / 2, (bh - rw) / 2
            for q in _resample([(-hx_, 0.0, -hz_), (hx_, 0.0, -hz_), (hx_, 0.0, hz_), (-hx_, 0.0, hz_)], 0.2):
                inst["sbulb"].append(Ms @ _T(xc0 + q[0], fy * (bt / 2 + 0.034), zb + bh / 2 + q[2]) @ Matrix.Diagonal((0.85, 0.85, 0.85, 1)))
        so.append(add("iron", _boxes("sign_rims", rim)))
        so.append(add("other", _boxes("sign_faces", faces), M_sign))
        # ironwork: flat bars under and over the case, V struts back to a strap round the quoins at four levels, and a
        # pendant of scrollwork under the case
        flat = [(-0.02, -0.007), (0.02, -0.007), (0.02, 0.007), (-0.02, 0.007)]
        so.append(add("iron", _boxes("sign_bracket_bars", [(xe / 2, 0, zb - 0.03, xe + 0.04, 0.045, 0.05, "fklrtb"),
                                                             (xe / 2, 0, zb + bh + 0.03, xe + 0.04, 0.045, 0.05, "fklrtb")])))
        zst = [zb + 0.16, zb + bh * 0.36, zb + bh * 0.66, zb + bh - 0.16]
        for zz in zst:
            so.append(add("iron", sweep("sign_strut", [(min(qa.x + 0.5, xe - 0.2), wy * bt / 2, zz), (qa.x, qa.y, zz), (max(qa.x - 0.02, 0.05), wy * bt / 2, zz)],
                                        0.015, sides=4, smooth_path=False)))
        so.append(add("iron", sweep("sign_brace", bezier_path((0.05, 0, zb - 0.055), (0.12, 0, zb - 0.4), (xe - 0.12, 0, zb - 0.4), (xe - 0.05, 0, zb - 0.055), n=12),
                                    flat, up=(0, 1, 0), smooth_path=False)))
        so.append(add("iron", sweep("sign_scroll", spiral_path(0.02, 0.085, 1.7, n=20, a0_deg=20, plane="XZ", center=(xc0 - 0.2, 0, zb - 0.15)),
                                    flat, up=(0, 1, 0), smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t)))
        so.append(add("iron", sweep("sign_scroll2", spiral_path(0.02, 0.085, 1.7, n=20, a0_deg=160, plane="XZ", center=(xc0 + 0.2, 0, zb - 0.15)),
                                    flat, up=(0, 1, 0), smooth_path=False, scale_fn=lambda t: 0.6 + 0.4 * t)))
        so.append(add("iron", sphere("sign_drop", 0.035, loc=(xc0, 0, zb - 0.35), segments=6, rings=4)))
        so.append(add("iron", cylinder("sign_pendant_rod", 0.008, 0.27, loc=(xe - 0.05, 0, zb - 0.19), verts=5)))
        so.append(add("iron", cone("sign_pendant_cap", 0.012, 0.045, 0.04, loc=(xe - 0.05, 0, zb - 0.315), verts=6)))
        so.append(add("other", sphere("sign_pendant", 0.055, loc=(xe - 0.05, 0, zb - 0.385), segments=8, rings=5), M_neon))    # glass globe: the light source
        # crown: a glowing gold silhouette on an iron band, outlined in bulbs on both faces, pearls on the points
        hwc = bw_ / 2
        cpts = [(-0.80, 0), (0.80, 0), (0.86, 0.22), (1.0, 0.80), (0.72, 0.48), (0.5, 0.92), (0.25, 0.52), (0, 1.0), (-0.25, 0.52), (-0.5, 0.92),
                (-0.72, 0.48), (-1.0, 0.80), (-0.86, 0.22)]
        # CLEARANCE RULE: the foot of the glowing crown stands 7 mm inside its iron band (it lay in the plane of the band's
        # underside), and the bands run 1 cm past its flanks (they ended on them)
        out = [(hwc * a, max(hc * b, 0.007)) for a, b in cpts]
        zc0 = zb + bh + 0.075
        gold.append(add("other", shape_extrude("sign_crown", out, depth=0.06, loc=(xc0, 0.03, zc0), rot=(90, 0, 0)), M_gold))
        so.append(add("iron", _boxes("sign_crown_band", [(xc0, 0, zc0 + 0.05, hwc * 1.70, 0.1, 0.1, "fklrtb"),
                                                         (xc0, 0, zc0 + 0.2, hwc * 1.78, 0.085, 0.035, "fklrtb")])))
        ins_ = [((a * 0.93) * hwc, (0.5 + (b - 0.5) * 0.9) * hc) for a, b in cpts[2:]]
        bl = []
        for fy in (-1, 1):
            for q in _resample([(a, 0.0, b) for a, b in ins_] + [(-hwc * 0.74, 0.0, hc * 0.3), (hwc * 0.74, 0.0, hc * 0.3)], 0.105):
                bl.append((xc0 + q[0], fy * 0.045, zc0 + q[2], 0.9))
            for k in range(5):
                bl.append((xc0 + (k - 2) * hwc * 0.33, fy * 0.062, zc0 + 0.05, 1.05))
        for (a, b) in cpts[3:12:2]:
            bl.append((xc0 + a * hwc * 0.97, 0.0, zc0 + b * hc + 0.01, 1.7))
        for b in bl:
            inst["sbulb"].append(Ms @ _T(*b[:3]) @ Matrix.Diagonal((b[3], b[3], b[3], 1)))
        so += gold
        _xf(so, Ms)
        # guy rods from the head of the case back to both walls
        tip = Ms @ Vector((xe - 0.03, 0, zb + bh + 0.03))
        gy = []
        for zz in zst:               # the strap round the corner that takes the struts
            gy += [(sgn * (W / 2 - 0.07), -0.058, zz, 0.24, 0.016, 0.12, "fklrtb"), (sgn * (W / 2 + 0.058), 0.06, zz, 0.016, 0.22, 0.12, "fklrtb")]
        for (gx, gy_) in ((sgn * (W / 2 - 1.5), 0.0), (sgn * W / 2, 1.5)):
            add("iron", sweep("sign_guy", [tuple(tip), (gx, gy_, tip.z + 0.72)], 0.007, sides=4))
            gy.append((gx, gy_ - 0.01 if gy_ == 0.0 else gy_, tip.z + 0.72, 0.08 if gy_ == 0.0 else 0.03, 0.03 if gy_ == 0.0 else 0.08, 0.08, "fklrtb"))
        add("iron", _boxes("sign_guy_eyes", gy))
        lc = Ms @ Vector((xe - 0.05, 0, zb - 0.385))  # at the glass globe under the tip (inside the case it burns the letter edges)
        light("neon_sign", tuple(lc), color=p.neon, intensity=clamp(3.5 * bh, 10, 16), range=10, flicker=0.04, shadow_near=0.6)
        socket("sign", "attach", (sgn * W / 2, 0.0, zb), role="sign")

    # ---- a light-tight core behind the rooms, so no sun or lamp light crosses the hollow block through the glazing
    zk = ZE + (0.25 + OE) * tan(ph1)
    if p.roof == "mansard":
        cprof = [(-1.0, ZS + 0.1), (-1.0, ZC - 0.2), (OT - 0.45, ZT - 0.3)]
    else:
        cprof = [(-1.0, ZS + 0.1), (-1.0, zk), (-(D / 2 - 0.25), ZE + (D / 2 - 1.0 + OE) * tan(ph1))]
    add("int", _ring("core_upper", -W / 2, W / 2, 0, D, cprof, seg=99.0))
    oc_, zc_ = -cprof[-1][0], cprof[-1][1]
    add("int", _quads("core_caps", [[(-W / 2 + oc_, oc_, zc_), (W / 2 - oc_, oc_, zc_), (W / 2 - oc_, D - oc_, zc_), (-W / 2 + oc_, D - oc_, zc_)],
                                    [(-W / 2 + 1.0, 1.0, ZS + 0.1), (-W / 2 + 1.0, D - 1.0, ZS + 0.1), (W / 2 - 1.0, D - 1.0, ZS + 0.1), (W / 2 - 1.0, 1.0, ZS + 0.1)]]))
    add("int", _boxes("core_ground", [(0, (1.75 + D - 1.0) / 2, (0.1 + ZS + 0.1) / 2, W - 2.0, D - 2.75, ZS, "fklrt")]))
    bq = []                                     # corner baffles: wall corner back to the core, ground and upper storeys
    for sx in (-1, 1):
        for (yo, yg, yu) in ((0.06, 1.75, 1.0), (D - 0.06, D - 1.0, D - 1.0)):
            xo, xi = sx * (W / 2 - 0.06), sx * (W / 2 - 1.0)
            bq.append([(xo, yo, 0.1), (xi, yg, 0.1), (xi, yg, ZS + 0.1), (xo, yo, ZS + 0.1)])
            bq.append([(xo, yo, ZS + 0.1), (xi, yu, ZS + 0.1), (xi, yu, cprof[1][1]), (xo, yo, cprof[1][1])])
    add("int", _quads("core_baffles", bq))

    # ---- instanced small parts
    def stamp(name, base, key, kind, m=None):
        if inst[key]:
            apply_transform(base)
            return add(kind, instances_to_mesh(name, base, inst[key], keep_base=False), m)
        delete([base])
        return None
    stamp("balusters", _boxes("baluster", [(0, 0, 0.38, 0.014, 0.014, 0.76, "fklr")]), "bal", "iron")
    stamp("rail_rings", torus("rail_ring", 0.054, 0.006, rot=(90, 0, 0), major_seg=8, minor_seg=3), "ringx", "iron")
    stamp("posts", _boxes("post", [(0, 0, 0.51, 0.034, 0.034, 1.02, "fklr")]), "post", "iron")
    stamp("finials", lathe("finial", [(0.0, -0.04), (0.02, -0.04), (0.012, -0.012), (0.034, 0.02), (0.0, 0.058)], segments=6), "fin", "iron")
    stamp("bulbs", sphere("bulb", 0.034, segments=6, rings=3), "bulb", "other", M_bulb)
    stamp("sign_bulbs", sphere("sign_bulb", 0.034, segments=6, rings=3), "sbulb", "other", M_sbulb)
    stamp("room_lamps", sphere("room_lamp", 0.07, segments=6, rings=3), "lamp", "other", M_bulb)
    stamp("modillions", _boxes("modillion", [(0, 0, 0, 0.13, 0.3, 0.1, "flrb")]), "mod", "stone")
    sp_ = [(-0.012, 0), (0.012, 0), (0.012, 0.2), (0.05, 0.25), (0.02, 0.28), (0.0, 0.4), (-0.02, 0.28), (-0.05, 0.25), (-0.012, 0.2)]
    spear = mesh_object("spear", [(x_, -0.005, z_) for x_, z_ in sp_] + [(x_, 0.005, z_) for x_, z_ in sp_], [tuple(range(9)), tuple(range(17, 8, -1))])
    stamp("cresting", spear, "spear", "iron")
    add("other", _quads("room_walls", Q["room"]), M_room)
    add("other", _quads("room_curtains", Q["curt"], Q["curt_uv"]), M_curt)
    add("int", _quads("room_shadows", Q["int"]))
    add("floor", _quads("room_floors", Q["floor"]))
    bpy.context.view_layer.update()

    # ---- art-directed weathering
    blot = noise_mask(1.6, seed=sd + 5, lo=0.4, hi=0.75)
    soft = noise_mask(3.5, seed=sd + 7, lo=0.3, hi=0.9)
    ka = 0.3 + 0.7 * age
    for (o, Mf, streaks, L) in wall_paint:      # stucco: rain streaks under sill ends, hood ends and the cornice; damp over the string course
        Mi = np.array(Mf.inverted())
        ph_ = L * 1.37 + sd

        def wall_mask(co, n, Mi=Mi, streaks=streaks, ph_=ph_):
            lc = co @ Mi[:3, :3].T + Mi[:3, 3]
            x, z = lc[:, 0], lc[:, 2]
            g = np.zeros(len(x))
            for (xs_, zt_, ln, st) in streaks:
                g = np.maximum(g, np.clip(1 - np.abs(x - xs_) / 0.07, 0, 1) * np.clip(1 - (zt_ - z) / ln, 0, 1) * (z <= zt_ + 0.012) * st)
            hsh = np.modf(np.abs(np.sin(np.round(x / 0.07) * 12.9898 + ph_) * 43758.5453))[0]
            dl = 0.18 + 1.25 * hsh ** 3
            g = np.maximum(g, np.clip(1 - (ZW - 0.84 - z) / dl, 0, 1) * (0.3 + 0.45 * hsh) * (z <= ZW - 0.83))
            g = np.maximum(g, np.clip(1 - (z - ZS - 0.43) / 0.55, 0, 1) * 0.3)
            return np.clip(g * ka + soft(co, n) * 0.07, 0, 1)
        paint(o, grime=wall_mask, wear=mask_mul(noise_mask(2.8, seed=sd + 9, lo=0.7, hi=0.95), 0.2 * age))
    for (o, zb, zt) in stacks:
        paint(o, grime=lambda co, n, zb=zb, zt=zt: np.clip(0.15 + 0.75 * np.clip((co[:, 2] - zt + 1.6) / 1.6, 0, 1) * ka + soft(co, n) * 0.15, 0, 1))
    for o in G["stone"]:
        paint(o, grime=lambda co, n: np.clip(np.clip(1 - co[:, 2] / 1.3, 0, 1) * 0.6 + blot(co, n) * 0.14, 0, 1) * ka,
              special=lambda co, n: np.clip(1 - co[:, 2] / 0.8, 0, 1) * blot(co * 2.5, n) * (co[:, 1] > 0.3) * 0.8 * age
              + (co[:, 2] > ZW - 0.2) * blot(co * 2.0, n) * np.clip(n[:, 2], 0, 1) * 0.6 * age)
    for o in G["lead"]:
        paint(o, grime=lambda co, n: 0.82 + 0.18 * soft(co * 3.0, n), mode="set")
    for o in G["int"] + G["floor"]:
        paint(o, grime=1.0, mode="set")
    lich = noise_mask(1.3, seed=sd + 13, lo=0.55, hi=0.85)
    for (o, gr, we) in slate_paint:
        paint(o, grime=lambda co, n, gr=gr: gr, wear=lambda co, n, we=we: we, special=mask_mul(lich, (0.02 + 0.14 * age) if tile else (0.04 + 0.3 * age)), mode="set")
    for o in G["wood"]:
        paint(o, grime=lambda co, n: np.clip(np.clip(1.15 - (co[:, 2] - FL) / 1.0, 0, 1) * (co[:, 2] < 3.0) * 0.75 * (0.45 + age), 0, 1),
              wear=mask_mul(noise_mask(0.25, seed=sd + 21, lo=0.6, hi=0.85), 0.15 + 0.5 * age))
    paint(gut, grime=0.75)
    for o in G["iron"]:
        paint(o, special=mask_mul(noise_mask(0.35, seed=sd + 17, lo=0.5, hi=0.8), 0.2 + 0.6 * age))
    if DEBUG:
        tot = 0
        for k, lst in G.items():
            n = 0
            for o in lst:
                if o.type == 'MESH':
                    o.data.calc_loop_triangles()
                    n += len(o.data.loop_triangles)
            tot += n
            print("DBG tris", k, n)
        for o in G["other"] + G["roof"]:
            o.data.calc_loop_triangles()
            print("DBG   ", o.name, len(o.data.loop_triangles))
        print("DBG total", tot)
