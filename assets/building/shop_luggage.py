"""The 'Luggage' shop of the square: a narrow 2-3 storey house whose ground floor is a painted timber
shopfront (display window + glazed shop door under one long gilt-lettered fascia sign), with trunks,
suitcases and hat boxes stacked on the pavement, a wall lantern and a lamp-lit hanging suitcase sign.
Upper floors are lime-plastered or brick masonry with stone quoins, string course and sash windows, or a
jettied oak frame with casements.  The roof is built here (not a catalogue roof): tile courses, bell-cast
eaves, bargeboards or hips, lead-cheeked dormers that share the main roof's covering, a brick chimney and
cast-iron rainwater goods.  Walls are one dense welded skin per storey band (gables included), so the
weathering - splash-back, algae, sill streaks, hopper stains, patched plaster - is painted at fine grain.
All four sides are finished (plinth, windows, back door, downpipes).
The window display (leather cases on stepped shelves before a panelled back), the striped awning, the hanging
suitcase sign and the pavement luggage are modelled here; the display panes are left unglazed (any glass sheet
veils what stands behind it in the renderer), only the transom lights are glazed.  Every material slot is
re-pointed at one canonical material per library (13-16 draw calls), and a last pass keeps the heaviest
parameter sets inside the triangle budget by re-laying the tiles / wall skins coarser - nothing is left out.
"""
import json
from math import atan, cos, sin, tan, radians, degrees, pi, hypot, sqrt, ceil

import bmesh
import bpy
import numpy as np

from mathutils import Euler, Matrix, Vector

from tk import *

META = {"name": "Luggage shop", "class": "building", "mount": "ground", "focus": [0.5, -0.6, 1.75]}
PARAMS = {
    "width": {"default": 6.0, "min": 5.0, "max": 8.0, "desc": "frontage width, ground-floor walls span x in +-width/2 (m)"},
    "depth": {"default": 7.0, "min": 6.0, "max": 10.0, "desc": "plan depth, body from y = 0 to y = depth (m)"},
    "storeys": {"default": 2, "min": 2, "max": 3, "desc": "storeys below the eaves (shop + 1 or 2 upper floors)"},
    "style": {"default": "plaster", "choices": ["plaster", "brick", "timber"],
              "desc": "upper floors: limewashed masonry / facing brick (both with quoins and sashes) / jettied oak frame with casements"},
    "roof": {"default": "eaves_front", "choices": ["eaves_front", "gable_front"], "desc": "ridge parallel to the street (dormers to the front) or gable to the street"},
    "roof_form": {"default": "gable", "choices": ["gable", "half_hip", "hip"], "desc": "roof form (the ridge direction always follows `roof`)"},
    "pitch": {"default": 50.0, "min": 42.0, "max": 58.0, "desc": "roof pitch (deg); eased automatically so the roof never outgrows the walls on deep plans"},
    "covering": {"default": "roof_tile_clay", "choices": ["roof_tile_clay", "roof_slate"], "desc": "roof covering (dormers always match)"},
    "dormers": {"default": 2, "min": 1, "max": 3, "desc": "dormers (street slope for eaves_front, both side slopes for gable_front); the first is always lit"},
    "storey_height": {"default": 2.9, "min": 2.7, "max": 3.2, "desc": "upper storey height (m)"},
    "jetty": {"default": 0.34, "min": 0.0, "max": 0.4, "desc": "projection of the timber-framed upper floors over the shopfront (m, timber style)"},
    "pattern": {"default": "close_studding", "choices": ["close_studding", "square_panels", "herringbone", "st_andrew", "curved_braces"],
                "desc": "timber-frame panel pattern of the front (timber style)"},
    "ground": {"default": "plaster", "choices": ["plaster", "brick", "stone"], "desc": "ground-floor wall finish"},
    "paint": {"default": "#7a1d1a", "desc": "paint colour of the shopfront joinery, bargeboards, back door and the painted trunk"},
    "plaster": {"default": "#e3d2a4", "desc": "limewash colour of the walls (plaster and timber styles)"},
    "sign_text": {"default": "Luggage", "desc": "fascia sign lettering"},
    "sign_caps": {"default": True, "desc": "set the fascia lettering in capitals"},
    "sign_spaced": {"default": True, "desc": "letter-space a short one-word name across the board"},
    "sign_font": {"default": "serif", "choices": ["serif", "roman", "sans", "script", "noto_serif"], "desc": "fascia letter face"},
    "goods": {"default": "luggage", "choices": ["luggage", "hats", "bottles", "books", "cakes"], "desc": "goods displayed in the shop window"},
    "trunks": {"default": 7, "min": 0, "max": 8, "desc": "pieces of luggage stacked on the pavement"},
    "door_side": {"default": "left", "choices": ["left", "right", "centre"], "desc": "position of the shop door in the shopfront"},
    "hanging_sign": {"default": True, "desc": "trade sign on scrolled iron brackets: a 0.8 m leather suitcase with gilt fittings under two gooseneck lamps (hung end-on under a jetty)"},
    "awning": {"default": False, "desc": "striped canvas awning over the display window"},
    "window_boxes": {"default": True, "desc": "flower boxes under the first-floor front windows"},
    "lit": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "share of upper windows with a lit room behind"},
    "shop_cd": {"default": 26.0, "min": 5.0, "max": 60.0, "desc": "display lamps under the fascia, lighting the goods and the pavement (cd)"},
    "lantern_cd": {"default": 4.0, "min": 1.0, "max": 15.0, "desc": "wall lantern intensity (cd)"},
    "number": {"default": "7", "desc": "house number on the door fanlight"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: grime, damp, chipped paint, moss, ridge sag"},
}
VARIANTS = {
    "timber_gable": {"style": "timber", "roof": "gable_front", "paint": "#1f4a3a", "plaster": "#ece2c8", "width": 5.4, "depth": 7.5,
                     "storeys": 3, "storey_height": 2.75, "pattern": "close_studding", "ground": "brick", "door_side": "right",
                     "hanging_sign": False, "awning": True, "window_boxes": False, "trunks": 5, "dormers": 2, "sign_text": "Trunks",
                     "sign_font": "roman", "seed": 7},
    "wide_three_storey": {"width": 7.8, "depth": 8.0, "storeys": 3, "door_side": "centre", "paint": "#1d3050", "plaster": "#d9c7b0",
                          "ground": "stone", "dormers": 3, "roof_form": "half_hip", "covering": "roof_slate", "trunks": 7, "window_boxes": False,
                          "hanging_sign": False, "sign_text": "Luggage & Trunks", "pitch": 46.0, "seed": 12},
    "timber_eaves": {"style": "timber", "roof": "eaves_front", "pattern": "curved_braces", "paint": "#5a3a12", "plaster": "#e9d9a6",
                     "width": 6.6, "depth": 6.4, "storeys": 2, "storey_height": 3.0, "jetty": 0.4, "dormers": 2, "trunks": 6,
                     "sign_text": "Portmanteaux", "lit": 0.8, "seed": 21},
    "narrow_brick": {"style": "brick", "width": 5.0, "depth": 9.0, "storeys": 3, "ground": "brick", "paint": "#22313a",
                     "roof": "gable_front", "door_side": "right", "dormers": 2, "roof_form": "gable", "pitch": 52.0, "awning": True,
                     "window_boxes": False, "trunks": 4, "lit": 0.45, "age": 0.8, "sign_text": "Luggage", "seed": 33},
    "hipped_slate": {"roof_form": "hip", "covering": "roof_slate", "width": 7.0, "depth": 6.2, "paint": "#2f4a2c", "plaster": "#d8d2c2",
                     "door_side": "right", "dormers": 2, "sign_text": "Valises", "trunks": 8, "lit": 0.7, "age": 0.3, "seed": 5},
}

TIMBER = "#5a4130"
JOINERY = "#d6cdb6"
STONE = "#c6b99c"
BOARD = "#101114"        # sentinel tint: parts asked for in this colour become the shop paint, darkened by grime
HEEL = 0.17              # tile surface above the wall plate, at the wall face
THK = 0.13               # roof build-up (tiles, battens, rafters): the soffit lies this far below the tiles
# ------------------------------------------------------------------------------------------ helpers
def _tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


def _bx(*a, **k):
    """Box with a single-segment bevel (small parts: half the triangles of the default 2-segment bevel)."""
    k.setdefault("segments", 1)
    return box(*a, **k)


def _m(o, m):
    assign(o, m)
    return o


def _xform(o, rot=(0, 0, 0), loc=(0, 0, 0)):
    M = Matrix.Translation(Vector(loc)) @ Euler(tuple(radians(a) for a in rot)).to_matrix().to_4x4()
    transform_verts(o, lambda co: M @ co)
    return o


def _inset(o, t=0.012, amin=0.004):
    """Border loop just inside every large flat face: keeps the exporter's per-vertex edge wear and cavity grime
    to the rim of a coarse panel instead of smearing them over the whole face."""
    realize(o)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    fs = [f for f in bm.faces if f.calc_area() > amin]
    if fs:
        bmesh.ops.inset_individual(bm, faces=fs, thickness=t, depth=0.0, use_even_offset=True)
    bm.to_mesh(o.data)
    bm.free()
    return o


def _bb(o):
    cs = [v.co for v in o.data.vertices]
    return (Vector((min(c.x for c in cs), min(c.y for c in cs), min(c.z for c in cs))),
            Vector((max(c.x for c in cs), max(c.y for c in cs), max(c.z for c in cs))))


def _prism_x(name, yz, x0, x1, bev=0.0):
    """Polygon in the YZ plane extruded along X."""
    n = len(yz)
    verts = [(x0, y, z) for y, z in yz] + [(x1, y, z) for y, z in yz]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for i in range(n):
        k = (i + 1) % n
        faces.append((i, k, n + k, n + i))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    if bev > 0:
        bevel(o, bev, 1)
    return o


def _beam(name, a, b, wdt, thk, normal, proud=0.025, bev=0.006):
    """Timber between two points lying on a wall face; `normal` = outward wall normal."""
    a, b, nrm = Vector(a), Vector(b), Vector(normal).normalized()
    d = b - a
    L = d.length
    xd = d / L
    yd = nrm.cross(xd)
    o = _bx(name, (L, wdt, thk), bevel=bev)
    c = (a + b) / 2 + nrm * (proud - thk / 2)
    M = Matrix((xd, yd, nrm)).transposed().to_4x4()
    M.translation = c
    transform_verts(o, lambda co: M @ co)
    uv_grain(o, axis=tuple(xd))
    return o


def _sock(comp, name, loc=(0, 0, 0), yaw=0.0):
    for (n, _kind, M, props) in comp.sockets:
        if n == name:
            v = M.translation
            c, s = cos(radians(yaw)), sin(radians(yaw))
            return (loc[0] + c * v.x - s * v.y, loc[1] + s * v.x + c * v.y, loc[2] + v.z), props
    return None, {}



def _lin(hx):
    """'#rrggbb' -> linear rgb (the space library tints are stored in)."""
    out = []
    for k in (1, 3, 5):
        c = int(hx[k:k + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def _lum(t):
    return 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2]


def _near(a, b, tol=0.045):
    return max(abs(max(x, 0.0) ** 0.5 - max(y, 0.0) ** 0.5) for x, y in zip(a, b)) < tol


def _sheet(name, groups, ref=(0, 0, 1)):
    """One mesh from several point lattices.  groups = [(rows, uv_rows | None, ref | None)]; every lattice is wound
    so that its faces look along `ref`.  UVs (metres) are kept when given."""
    verts, faces, uvs = [], [], []
    has_uv = False
    for g in groups:
        rows, uvr = g[0], g[1]
        rf = Vector(g[2]) if len(g) > 2 and g[2] is not None else Vector(ref)
        nr, nc = len(rows), len(rows[0])
        base = len(verts)
        for a in range(nr):
            for b in range(nc):
                verts.append(tuple(rows[a][b]))
                if uvr is not None:
                    uvs.append(tuple(uvr[a][b]))
                    has_uv = True
                else:
                    uvs.append((0.0, 0.0))
        fs = []
        acc = 0.0
        for a in range(nr - 1):
            for b in range(nc - 1):
                i = base + a * nc + b
                f = (i, i + 1, i + nc + 1, i + nc)
                fs.append(f)
                p0, p1, p2, p3 = (Vector(verts[k]) for k in f)
                acc += (p2 - p0).cross(p3 - p1).dot(rf)
        if acc < 0:
            fs = [f[::-1] for f in fs]
        faces += fs
    o = mesh_object(name, verts, faces)
    if has_uv:
        me = o.data
        lay = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
        idx = np.zeros(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", idx)
        lay.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32)[idx].ravel())
        uv_keep(o)
    return o


def _loft(name, pts, prof, side, up=(0, 0, 1), closed=False):
    """Closed profile polygon prof = [(a, b)] carried along pts: vertex = P + side_i * a + up * b (mitre by passing
    one side vector per point)."""
    up = Vector(up)
    n, m = len(prof), len(pts)
    sides = side if isinstance(side, list) else [side] * m
    verts = []
    for P, S in zip(pts, sides):
        P, S = Vector(P), Vector(S)
        for a, b in prof:
            verts.append(tuple(P + S * a + up * b))
    faces = []
    for i in range(m if closed else m - 1):
        i2 = (i + 1) % m
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((i * n + k, i * n + k2, i2 * n + k2, i2 * n + k))
    if not closed:
        faces.append(tuple(range(n))[::-1])
        faces.append(tuple(range((m - 1) * n, m * n)))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _span(a, b, step):
    a, b = Vector(a), Vector(b)
    n = max(1, int(ceil((b - a).length / step)))
    return [a.lerp(b, k / n) for k in range(n + 1)]


def _wall(name, loop, z0, top, closed=True, cell=0.15, row=0.28, fine=1.4):
    """Dense single-skin wall over the plan polyline `loop` (outward = right of the direction of travel) from z0 up
    to top(x, y): eaves walls and gables are ONE welded skin, so there is no seam at the wall plate and the vertex
    masks have ~15 cm resolution (rows of half height in the bottom `fine` metres, where the splash-back,
    algae and flaking render are painted)."""
    pts = []
    m = len(loop)
    for i in range(m if closed else m - 1):
        a, b = Vector(loop[i]), Vector(loop[(i + 1) % m])
        n = max(1, int(round((b - a).length / cell)))
        for k in range(n):
            pts.append(a.lerp(b, k / n))
    if not closed:
        pts.append(Vector(loop[-1]))
    tops = [float(top(q.x, q.y)) for q in pts]
    z1 = min(tops)
    tops = [z1 if t - z1 < 0.02 else t for t in tops]
    zf = z0 + fine if z0 < 0.5 and z1 - z0 > fine + row else z0
    nf = int(round((zf - z0) / (row * 0.5)))
    lv = [z0 + (zf - z0) * k / nf for k in range(nf)]
    nu = max(1, int(round((z1 - zf) / row)))
    lv += [zf + (z1 - zf) * k / nu for k in range(nu + 1)]
    nr1 = len(lv) - 1
    nr2 = int(ceil((max(tops) - z1) / row)) if max(tops) - z1 > 0.02 else 0
    R_ = nr1 + 1 + nr2
    verts = []
    for q, zt in zip(pts, tops):
        for k in range(nr1 + 1):
            verts.append((q.x, q.y, lv[k]))
        for k in range(1, nr2 + 1):
            verts.append((q.x, q.y, z1 + (zt - z1) * k / nr2))
    faces = []
    nc = len(pts)
    for c in range(nc if closed else nc - 1):
        c2 = (c + 1) % nc
        flat = tops[c] - z1 < 1e-4 and tops[c2] - z1 < 1e-4
        for r in range(R_ - 1):
            if r >= nr1 and flat:
                continue
            faces.append((c * R_ + r, c2 * R_ + r, c2 * R_ + r + 1, c * R_ + r + 1))
    o = mesh_object(name, verts, faces)
    weld(o, 0.0005)
    return o


def _frame(a, b, up=(0, 0, 1), lift=0.0, sx=1.0):
    """Instance matrix: local X from a to b (scaled sx), local Z as close to `up` as possible, origin at the middle."""
    a, b = Vector(a), Vector(b)
    x = (b - a).normalized()
    y = Vector(up).cross(x).normalized()
    z = x.cross(y)
    M = Matrix((x * sx, y, z)).transposed().to_4x4()
    M.translation = (a + b) / 2 + z * lift
    return M


class _Roof:
    """Roof geometry in its own frame: u along the ridge, v across (slope sg = -1 on the v < 0 side), d = plan
    distance up-slope from the eaves edge.  eaves_front: u = x, the sg = -1 slope faces the street.  gable_front:
    the frame is turned 90 deg, u = y, the u = -Lu/2 gable faces the street.  Hips are built in the same frame, so
    the ridge direction never depends on the roof form."""

    def __init__(s, Lu, Lv, zE, pitch, form, ovh, vo, sag, ridge_x, cy):
        s.Lu, s.Lv, s.zE, s.form, s.ovh, s.sg, s.ridge_x, s.cy = Lu, Lv, zE, form, ovh, sag, ridge_x, cy
        s.pitch = pitch
        s.tanp, s.cosp = tan(radians(pitch)), cos(radians(pitch))
        s.dmax = Lv / 2 + ovh
        s.ze = zE + HEEL - ovh * s.tanp
        if form == "gable":
            s.Uv, s.dh, s.eh = Lu / 2 + vo, s.dmax, 0.0
        elif form == "hip":
            k = ovh / s.dmax
            s.Uv = Lu / 2 + ovh
            s.dh = 0.0
            if s.Uv - 1.0 < s.dmax:               # short ridge: steeper hip ends, overhang reduced to keep one eaves level
                s.Uv = (Lu / 2 - k) / (1 - k)
            s.eh = min(s.dmax, s.Uv - 1.0)
        else:
            s.Uv, s.dh = Lu / 2 + vo, 0.6 * s.dmax
            s.eh = min(s.dmax - s.dh, s.Uv - 1.0)
        s.hw = s.dmax - s.dh                      # half width of a hip end at its eaves
        s.q = s.eh / s.hw if form != "gable" else 0.0
        s.ur = s.Uv - s.eh                        # half length of the ridge
        s.rise = s.dmax * s.tanp

    def zs(s, u, d):
        d = np.maximum(d, 0.0)
        bell = 0.09 * np.clip(1.0 - d / 0.9, 0.0, 1.0) ** 2
        sag = s.sg * (1.0 - np.clip(np.abs(u) / s.Uv, 0.0, 1.0) ** 2) * (d / s.dmax) ** 1.5
        return s.ze + d * s.tanp + bell - sag

    def ul(s, d):
        return s.Uv - s.q * max(0.0, d - s.dh)

    def top_uv(s, u, v):
        z = s.zs(u, s.dmax - np.abs(v))
        if s.form != "gable":
            e = np.clip((s.Uv - np.abs(u)) / s.eh, 0.0, 1.0)
            z = np.minimum(z, s.zs(u, s.dh + s.hw * e))
        return z

    def top(s, x, y):
        """Tile surface height over the world plan point (numpy-friendly)."""
        if s.ridge_x:
            return s.top_uv(x, y - s.cy)
        return s.top_uv(y - s.cy, -x)

    def W(s, u, v, z):
        return Vector((u, s.cy + v, z)) if s.ridge_x else Vector((-v, s.cy + u, z))

    def Wv(s, du, dv, dz=0.0):
        return Vector((du, dv, dz)) if s.ridge_x else Vector((-dv, du, dz))

    def P(s, sg, u, d, dz=0.0):
        return s.W(u, sg * (s.dmax - d), float(s.zs(u, d)) + dz)

    def E(s, su, e, t, dz=0.0):
        u = su * (s.Uv - s.eh * e)
        return s.W(u, t * s.hw * (1.0 - e), float(s.zs(u, s.dh + s.hw * e)) + dz)

    def normal(s, sg):
        return s.Wv(0.0, sg * s.tanp, 1.0).normalized()

    def end_normal(s, su):
        te = float(s.zs(s.ur, s.dmax) - s.zs(s.Uv, s.dh)) / s.eh
        return s.Wv(su * te, 0.0, 1.0).normalized()

    def yaw(s, sg):
        return (0.0 if sg < 0 else 180.0) + (0.0 if s.ridge_x else 90.0)

    def tile_groups(s, gauge, lift, r, cap=8000):
        """Courses of tiles / slates as stepped strips (riser + lap), UVs in metres with V = 0 at the eaves."""
        groups = []
        nk = max(4, int(round(s.dmax / (gauge * s.cosp))))
        nc = max(6, int(round(2 * s.Uv / 0.45)))
        nke = nce = 0
        if s.form != "gable":
            sl = hypot(s.eh, float(s.zs(s.ur, s.dmax) - s.zs(s.Uv, s.dh)))
            nke = max(3, int(round(sl / gauge)))
            nce = max(4, int(round(2 * s.hw / 0.45)))
        tot = 8 * (nk * nc + nke * nce)            # triangles of the whole covering
        if tot > cap:                              # big roofs: wider columns, same courses
            nc = max(3, int(nc * cap / tot))
            nce = max(3, int(nce * cap / tot))
        dzl = lift / s.cosp
        for sg in (-1, 1):
            uo = r.u(0.0, 1.2)
            for k in range(nk):
                d0, d1 = s.dmax * k / nk, s.dmax * (k + 1) / nk
                u0, u1 = s.ul(d0), s.ul(d1)
                R0, R1, R2, U0, U2 = [], [], [], [], []
                for i in range(nc + 1):
                    t = 2.0 * i / nc - 1.0
                    lz = dzl * (1.7 if k == 0 else 1.0) + r.u(0.0, 0.45 * dzl)
                    R0.append(s.P(sg, t * u0, d0))
                    R1.append(s.P(sg, t * u0, d0, lz))
                    R2.append(s.P(sg, t * u1, d1))
                    U0.append((t * u0 + uo, k * gauge))
                    U2.append((t * u1 + uo, (k + 1) * gauge))
                groups.append(([R0, R1, R2], [U0, U0, U2], s.normal(sg)))
        if s.form != "gable":
            nk, nc = nke, nce
            dzl = lift * sl / s.eh
            for su in (-1, 1):
                uo = r.u(0.0, 1.2)
                for k in range(nk):
                    e0, e1 = k / nk, (k + 1) / nk
                    R0, R1, R2, U0, U2 = [], [], [], [], []
                    for i in range(nc + 1):
                        t = 2.0 * i / nc - 1.0
                        lz = dzl * (1.7 if k == 0 else 1.0) + r.u(0.0, 0.45 * dzl)
                        R0.append(s.E(su, e0, t))
                        R1.append(s.E(su, e0, t, lz))
                        R2.append(s.E(su, e1, t))
                        U0.append((t * s.hw * (1 - e0) + uo, k * gauge))
                        U2.append((t * s.hw * (1 - e1) + uo, (k + 1) * gauge))
                    groups.append(([R0, R1, R2], [U0, U0, U2], s.end_normal(su)))
        return groups

    def soffit_groups(s):
        """Boarded underside where it can be seen (eaves, verges, hip ends) and the eaves fascia boards."""
        g = []
        nc = 8
        dn = (0, 0, -1)
        for sg in (-1, 1):
            rows = [[s.P(sg, (2.0 * i / nc - 1.0) * s.ul(d), d, -THK) for i in range(nc + 1)] for d in (0.0, s.ovh * 0.5, s.ovh + 0.07)]
            g.append((rows, None, dn))
            rows = [[s.P(sg, (2.0 * i / nc - 1.0) * s.Uv, 0.0, dz) for i in range(nc + 1)] for dz in (-THK - 0.035, 0.012)]
            g.append((rows, None, s.Wv(0, sg)))
            if s.form != "hip":
                dend = s.dh if s.form == "half_hip" else s.dmax
                ds = [d for d in (0.0, 0.3, 0.6, 0.9) if d < dend - 0.1] + [dend]
                for su in (-1, 1):
                    rows = [[s.P(sg, su * (s.Lu / 2 - 0.06), d, -THK), s.P(sg, su * s.Uv, d, -THK)] for d in ds]
                    g.append((rows, None, dn))
        if s.form != "gable":
            ew = min(1.0, (s.Uv - s.Lu / 2 + 0.07) / s.eh)
            ts = (-1.0, -0.5, 0.0, 0.5, 1.0)
            for su in (-1, 1):
                g.append(([[s.E(su, e, t, -THK) for t in ts] for e in (0.0, ew * 0.5, ew)], None, dn))
                g.append(([[s.E(su, 0.0, t, dz) for t in ts] for dz in (-THK - 0.035, 0.012)], None, s.Wv(su, 0)))
        return g

    def crest_lines(s):
        """Polylines to be capped with ridge / hip tiles."""
        n = max(2, int(round(2 * s.ur / 1.0)))
        lines = [[s.P(-1, -s.ur + 2 * s.ur * i / n, s.dmax) for i in range(n + 1)]]
        if s.form != "gable":
            for su in (-1, 1):
                for sg in (-1, 1):
                    lines.append([s.P(sg, su * s.ul(s.dh + s.hw * f), s.dh + s.hw * f) for f in (0.0, 0.34, 0.67, 1.0)])
        return lines
# ------------------------------------------------------------------------------------------ luggage
def _trunk(name, r, L, Dp, H, body, wood, metal, strap, label, domed):
    """Steamer trunk: canvas / leather covered box, hardwood slats, metal edge bindings, corner bumpers,
    lock and drawbolts, leather end handles, paper labels.  Sits on z = 0, centred."""
    o = []
    hb = H * (0.6 if domed else 0.7)
    z1 = hb + 0.006
    e = 0.0045
    o.append(_m(_inset(_bx(name + "_body", (L, Dp, hb), loc=(0, 0, hb / 2), bevel=0.008), 0.014), body))
    o.append(_m(_bx(name + "_gap", (L - 0.014, Dp - 0.014, 0.02), loc=(0, 0, hb)), metal))
    if domed:
        side = (H - z1) * 0.3
        rise = H - z1 - side
        n = 10
        arc = [(-Dp / 2 * cos(pi * k / n), z1 + side + rise * sin(pi * k / n)) for k in range(n + 1)]
        prof = [(-Dp / 2, z1)] + arc + [(Dp / 2, z1)]
        lid = _prism_x(name + "_lid", prof, -L / 2, L / 2, 0.006)
        smooth(lid, 50)
    else:
        prof = None
        lid = _inset(_bx(name + "_lid", (L, Dp, H - z1), loc=(0, 0, (H + z1) / 2), bevel=0.008), 0.014)
    o.append(_m(lid, body))
    hoops = (-L / 2 + 0.03, -L * 0.2, L * 0.2, L / 2 - 0.03)
    for i, hx in enumerate(hoops):
        m_ = metal if i in (0, 3) else wood
        w = 0.05 if i in (0, 3) else 0.042
        for sy in (-1, 1):
            s = _bx(f"{name}_slat", (w, 2 * e, hb - 0.004), loc=(hx, sy * (Dp / 2 + e - 0.001), hb / 2 + 0.002))
            uv_grain(s, "Z")
            o.append(_m(s, m_))
        if domed:
            k = 1 + 2 * e / Dp
            kz = 1 + e / (H - z1)
            path = [(hx, y * k, z1 + (z - z1) * kz) for (y, z) in prof]
            path[0] = (hx, path[0][1], z1 + 0.003)
            path[-1] = (hx, path[-1][1], z1 + 0.003)
            s = sweep(f"{name}_hoop", path, rounded_rect(2 * e, w, 0.0015, 1), up=(1, 0, 0), smooth_path=False)
            o.append(_m(s, m_))
        else:
            for sy in (-1, 1):
                s = _bx(f"{name}_slat", (w, 2 * e, H - z1 - 0.002), loc=(hx, sy * (Dp / 2 + e - 0.001), (H + z1) / 2))
                uv_grain(s, "Z")
                o.append(_m(s, m_))
            s = _bx(f"{name}_slat", (w, Dp + 4 * e - 0.002, 2 * e), loc=(hx, 0, H + e - 0.001))
            uv_grain(s, "Y")
            o.append(_m(s, m_))
    # long slats on the lid / front between the hoops (flat trunks)
    if not domed:
        for sy in (-0.25, 0.25):
            s = _bx(f"{name}_slat", (L - 0.02, 0.04, 2 * e), loc=(0, sy * Dp, H + e * 0.6))
            uv_grain(s, "X")
            o.append(_m(s, wood))
    # corner bumpers
    cz = [0.031, (z1 + 0.034) if domed else (H - 0.028)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for z in cz:
                o.append(_m(_bx(f"{name}_corner", (0.062, 0.062, 0.062), loc=(sx * (L / 2 - 0.028), sy * (Dp / 2 - 0.028), z), bevel=0.007), metal))
    # lock plate, hasp, drawbolts
    yf = -Dp / 2
    o.append(_m(_bx(name + "_lockplate", (0.08, 0.006, 0.1), loc=(0, yf - 0.003, hb - 0.02), bevel=0.002), metal))
    o.append(_m(_bx(name + "_hasp", (0.036, 0.008, 0.07), loc=(0, yf - 0.009, hb + 0.012), bevel=0.003), metal))
    for sx in (-1, 1):
        o.append(_m(_bx(name + "_bolt", (0.032, 0.012, 0.075), loc=(sx * L * 0.36, yf - 0.006, hb + 0.004), bevel=0.003), metal))
        # leather end handle on two metal loops
        zc = hb * 0.62
        xe = sx * L / 2
        path = [(xe, -0.085, zc), (xe + sx * 0.03, -0.06, zc - 0.012), (xe + sx * 0.04, 0.0, zc - 0.022),
                (xe + sx * 0.03, 0.06, zc - 0.012), (xe, 0.085, zc)]
        o.append(_m(sweep(name + "_handle", path, rounded_rect(0.009, 0.03, 0.003, 1), up=(0, 0, 1), smooth_path=False), strap))
        for sy in (-1, 1):
            o.append(_m(_bx(name + "_hmount", (0.012, 0.05, 0.045), loc=(xe + sx * 0.004, sy * 0.085, zc), bevel=0.003), metal))
    if not domed:
        # two leather belts buckled round the trunk
        bh = H + 2 * e + 0.012
        for sx in (-1, 1):
            xs = sx * L * 0.33
            loop = [(xs, y, bh / 2 + 0.001 + z) for (y, z) in rounded_rect(Dp + 4 * e + 0.012, bh, 0.012, 1)]
            o.append(_m(sweep(name + "_belt", loop, [(-0.0025, -0.02), (0.0025, -0.02), (0.0025, 0.02), (-0.0025, 0.02)], closed=True,
                              up=(1, 0, 0), smooth_path=False), strap))
            o.append(_m(_bx(name + "_buckle", (0.054, 0.008, 0.046), loc=(xs, yf - 2 * e - 0.011, hb * 0.5), bevel=0.002), metal))
    for k in range(1 + int(r.chance(0.6))):
        lx = r.u(-L * 0.36, L * 0.36) * 0.8
        lz = r.u(0.1, hb - 0.1)
        o.append(_m(_bx(name + "_label", (r.u(0.09, 0.13), 0.0016, r.u(0.06, 0.085)), loc=(lx, yf - 0.0012, lz), rot=(0, r.jitter(0, 14), 0)), label))
    return join(o, name)


def _suitcase(name, r, L, Hh, T, hide, trim, metal):
    """Leather suitcase lying flat (L x Hh footprint, T thick), handle on the -Y edge."""
    o = []
    o.append(_m(_inset(rounded_box(name + "_shell", (L, Hh, T), radius=0.035, loc=(0, 0, T / 2), seg=4, bevel=0.012), 0.014), hide))
    o.append(_m(rounded_box(name + "_seam", (L + 0.006, Hh + 0.006, 0.014), radius=0.038, loc=(0, 0, T * 0.66), seg=4), trim))
    for sx in (-1, 1):
        for sy in (-1, 1):
            o.append(_m(rounded_box(name + "_cap", (0.08, 0.08, T + 0.007), radius=0.037, loc=(sx * (L / 2 - 0.037), sy * (Hh / 2 - 0.037), T / 2),
                                    seg=3), trim))
    yf = -Hh / 2
    zc = T * 0.5
    path = [(-0.085, yf, zc), (-0.07, yf - 0.03, zc), (-0.04, yf - 0.042, zc), (0.04, yf - 0.042, zc), (0.07, yf - 0.03, zc), (0.085, yf, zc)]
    o.append(_m(sweep(name + "_handle", path, rounded_rect(0.013, 0.026, 0.004, 1), up=(0, 0, 1), smooth_path=False), trim))
    for sx in (-1, 1):
        o.append(_m(_bx(name + "_hmount", (0.034, 0.012, 0.034), loc=(sx * 0.085, yf - 0.004, zc), bevel=0.003), metal))
        o.append(_m(_bx(name + "_latch", (0.045, 0.012, 0.034), loc=(sx * L * 0.3, yf - 0.004, T * 0.66), bevel=0.003), metal))
        # strap with buckle round the case
        xs = sx * L * 0.2
        loop = [(xs, y, T / 2 + z) for (y, z) in rounded_rect(Hh + 0.009, T + 0.009, 0.016, 2)]
        o.append(_m(sweep(name + "_strap", loop, [(-0.002, -0.017), (0.002, -0.017), (0.002, 0.017), (-0.002, 0.017)], closed=True, up=(1, 0, 0), smooth_path=False), trim))
        o.append(_m(_bx(name + "_buckle", (0.046, 0.034, 0.007), loc=(xs, -Hh * 0.22, T + 0.008), bevel=0.002), metal))
    return join(o, name)


def _hatbox(name, r, R, H, hide, trim, metal):
    o = []
    hl = H * 0.3
    prof = [(0, 0), (R - 0.004, 0), (R, 0.004), (R, H - hl), (R + 0.005, H - hl), (R + 0.005, H - 0.008), (R - 0.002, H), (0, H + 0.006)]
    o.append(_m(lathe(name + "_body", prof, segments=20), hide))
    o.append(_m(ring_band(name + "_band", radius=R + 0.006, width=0.02, thick=0.003, loc=(0, 0, H - hl + 0.012), segments=20), trim))
    # strap over the lid
    e = R + 0.008
    for sx in (-1, 1):
        o.append(_m(_bx(name + "_strap", (0.004, 0.032, H - 0.03), loc=(sx * e, 0, H / 2 + 0.015), bevel=0.001), trim))
    path = [(-e, 0, H - 0.002), (-R * 0.6, 0, H + 0.006), (0, 0, H + 0.01), (R * 0.6, 0, H + 0.006), (e, 0, H - 0.002)]
    o.append(_m(sweep(name + "_strap_top", path, rounded_rect(0.032, 0.004, 0.001, 1), up=(0, 0, 1)), trim))
    path = [(0, -0.06, H + 0.01), (0, -0.04, H + 0.04), (0, 0.04, H + 0.04), (0, 0.06, H + 0.01)]
    o.append(_m(sweep(name + "_handle", path, 0.007, sides=6, smooth_path=False), trim))
    o.append(_m(_bx(name + "_clasp", (0.008, 0.04, 0.04), loc=(-e - 0.003, 0, H - hl), bevel=0.002), metal))
    return join(o, name)


def _bag(name, r, L, Dp, H, hide, trim, metal):
    """Gladstone bag: soft leather body gathered into a hinged metal frame, two handles."""
    o = []
    n = 6
    left = [(-Dp / 2 * (1 - 0.86 * (k / n) ** 2.2) - 0.012 * sin(pi * k / n), H * 0.92 * k / n) for k in range(n + 1)]
    prof = [(-Dp / 2 + 0.02, 0)] + left[1:] + [(-y, z) for (y, z) in reversed(left[1:])] + [(Dp / 2 - 0.02, 0)]
    body = _prism_x(name + "_body", prof, -L / 2, L / 2, 0.018)
    smooth(body, 60)
    o.append(_m(body, hide))
    zt = H * 0.92
    for sy in (-1, 1):
        o.append(_m(_bx(name + "_frame", (L + 0.012, 0.016, 0.022), loc=(0, sy * 0.022, zt + 0.004), bevel=0.004), metal))
        path = [(-0.07, sy * 0.03, zt), (-0.06, sy * 0.05, zt + 0.06), (0.0, sy * 0.055, zt + 0.075), (0.06, sy * 0.05, zt + 0.06), (0.07, sy * 0.03, zt)]
        o.append(_m(sweep(name + "_handle", path, 0.009, sides=6, smooth_path=False), trim))
    o.append(_m(_bx(name + "_clasp", (0.05, 0.065, 0.03), loc=(0, 0, zt + 0.012), bevel=0.004), metal))
    for sx in (-1, 1):
        o.append(_m(_bx(name + "_end", (0.006, Dp * 0.5, H * 0.5), loc=(sx * (L / 2 + 0.002), 0, H * 0.3), bevel=0.002), trim))
    return join(o, name)

# ------------------------------------------------------------------------------------------ inline joinery
def _plain_window(name, w, h, lit, frame_m, sur_m, pane_m, bar_m, stone_sur, r, drip_m=None):
    """Plain two-light casement in a stone (or oak) surround.  Local: wall plane y = 0, opening bottom centre
    at the origin, projects to -Y."""
    o = []
    jw = 0.13 if stone_sur else 0.11
    pj = 0.07 if stone_sur else 0.06
    for sx in (-1, 1):
        b = _bx(name + "_jamb", (jw, pj, h), loc=(sx * (w / 2 + jw / 2), -pj / 2, h / 2), bevel=0.008)
        uv_grain(b, "Z")
        o.append(_m(b, sur_m))
    lh = 0.2 if stone_sur else 0.14
    b = _bx(name + "_lintel", (w + 2 * jw + (0.1 if stone_sur else 0.16), pj + 0.015, lh), loc=(0, -(pj + 0.015) / 2, h + lh / 2), bevel=0.01)
    uv_grain(b, "X")
    o.append(_m(b, sur_m))
    if stone_sur:
        o.append(_m(_prism_x(name + "_key", [(0, h - 0.02), (-pj - 0.035, h - 0.02), (-pj - 0.045, h + lh + 0.03), (0, h + lh + 0.03)], -0.075, 0.075, 0.008), sur_m))
    sp = pj + 0.07
    sill = _prism_x(name + "_sill", [(0, -0.11), (-sp + 0.01, -0.11), (-sp, -0.03), (-sp, -0.012), (0, 0.012)], -(w / 2 + jw + 0.05), w / 2 + jw + 0.05, 0.007)
    uv_grain(sill, "X")
    o.append(_m(sill, sur_m))
    # timber frame + two casements
    ft, fd = 0.045, 0.05
    for sx in (-1, 1):
        b = _bx(name + "_frame", (ft, fd, h), loc=(sx * (w / 2 - ft / 2), -fd / 2, h / 2))
        o.append(_m(b, frame_m))
    for z in (ft / 2, h - ft / 2):
        o.append(_m(_bx(name + "_frame", (w - 2 * ft, fd, ft), loc=(0, -fd / 2, z)), frame_m))
    o.append(_m(_bx(name + "_mullion", (0.055, fd + 0.012, h - 2 * ft), loc=(0, -(fd + 0.012) / 2, h / 2)), frame_m))
    lw = (w - 2 * ft - 0.055) / 2
    nb = 2 if h < 1.3 else 3
    for sx in (-1, 1):
        cx = sx * (0.0275 + lw / 2)
        for sz in (ft + 0.018, h - ft - 0.018):
            o.append(_m(_bx(name + "_rail", (lw, 0.034, 0.036), loc=(cx, -0.03, sz)), frame_m))
        for ex in (-1, 1):
            o.append(_m(_bx(name + "_stile", (0.034, 0.034, h - 2 * ft), loc=(cx + ex * (lw / 2 - 0.017), -0.03, h / 2)), frame_m))
        for k in range(1, nb + 1):
            o.append(_m(_bx(name + "_bar", (lw - 0.06, 0.0225, 0.018), loc=(cx, -0.02475, ft + (h - 2 * ft) * k / (nb + 1))), bar_m))
        o.append(_m(_bx(name + "_bar", (0.018, 0.0225, h - 2 * ft - 0.06), loc=(cx, -0.02475, h / 2)), bar_m))
    # CLEARANCE RULE: a glowing pane is ONE sheet (all materials are double sided: the two skins of a thin box of lit
    # glass fight each other) with nothing parallel within 4 mm of it. The sheet stands at y = -0.018; rails, stiles
    # and bars pass through it and have their backs 4.5-5 mm behind it, the wall relief ends 13 mm behind it.
    o.append(_m(lod_faces(_bx(name + "_pane", (w - 2 * ft, 0.008, h - 2 * ft), loc=(0, -0.014, h / 2)), keep=(0, -1, 0), above=-1), pane_m))
    o.append(_m(_bx(name + "_stay", (0.012, 0.014, 0.12), loc=(0.0, -fd - 0.02, h * 0.5), bevel=0.003), mat("iron_wrought")))
    return join(o, name)


def _back_door(name, w, h, r, oak, frame_m, iron, stone, lead_m):
    """Ledged plank back door in a plain frame on a stone step.  Local: wall plane y = 0, threshold centre origin.
    (One densified leaf with applied joints, densified posts: the dirt and chipping are height ramps on the
    vertices, and full-height boards with vertices only at their ends would be dirtied evenly from foot to head.)"""
    o = []
    st = 0.14
    o.append(_m(stone_block(name + "_step", (w + 0.5, 0.42, st), loc=(0, -0.19, st / 2), chamfer=0.02, rough=0.004, rng=r, detail=0.7), stone))
    ft = 0.13
    for sx in (-1, 1):
        b = _bx(name + "_post", (ft, 0.11, h), loc=(sx * (w / 2 + ft / 2), -0.045, st + h / 2))
        densify(b, 0.3)                           # (height-ramped dirt needs vertices up the post)
        uv_grain(b, "Z")
        o.append(_m(b, frame_m))
    b = _bx(name + "_head", (w + 2 * ft + 0.18, 0.12, 0.17), loc=(0, -0.05, st + h + 0.085))
    uv_grain(b, "X")
    o.append(_m(b, frame_m))
    n = 6
    pw_ = w / n
    leaf = _bx(name + "_leaf", (w, 0.03, h - 0.012), loc=(0, -0.032, st + h / 2))
    densify(leaf, 0.24)
    uv_grain(leaf, "Z")
    o.append(_m(leaf, oak))
    for i in range(1, n):                         # the joints between the boards
        o.append(_m(_bx(name + "_joint", (0.007, 0.004, h - 0.03 - r.u(0, 0.03)), loc=(-w / 2 + pw_ * i, -0.0475, st + h / 2)), iron))
    for z in (0.32, h - 0.34):
        s = _bx(name + "_strap", (w * 0.8, 0.008, 0.05), loc=(-w * 0.08, -0.05, st + z), bevel=0.002)
        o.append(_m(s, iron))
        o.append(_m(cylinder(name + "_pin", 0.018, 0.09, loc=(-w / 2 + 0.004, -0.052, st + z), verts=8), iron))
        for k in range(4):
            o.append(_m(bolt(name + "_nail", 0.007, loc=(-w * 0.4 + k * w * 0.2, -0.054, st + z), rot=(90, 0, 0)), iron))
    zt = st + h + 0.34
    o.append(_m(_bx(name + "_hood", (w + 0.74, 0.64, 0.04), loc=(0, -0.3, zt), rot=(16, 0, 0), bevel=0.006), lead_m))
    for sx in (-1, 1):
        x = sx * (w / 2 + 0.2)
        b = _prism_x(name + "_hood_bracket", [(0, zt - 0.52), (0, zt + 0.06), (-0.52, zt - 0.088), (-0.52, zt - 0.16)], x - 0.035, x + 0.035, 0.006)
        uv_grain(b, "Y")
        o.append(_m(b, frame_m))
    o.append(_m(ring_band(name + "_ring", radius=0.05, width=0.012, thick=0.01, loc=(w / 2 - 0.13, -0.062, st + 1.02), rot=(90, 0, 0), segments=14), iron))
    o.append(_m(_bx(name + "_plate", (0.05, 0.006, 0.15), loc=(w / 2 - 0.13, -0.048, st + 1.05), bevel=0.002), iron))
    return join(o, name)


def _downpipe(name, top, loc, yaw, hz, iron, r):
    """Cast-iron rainwater pipe: swan-neck from the gutter outlet `top`, hopper head with rim at z = hz, socketed
    pipe lengths with eared collars, shoe at the foot.  `loc` = foot on the wall plane, yaw = wall yaw."""
    o = []
    yp = -0.085
    rr = 0.042
    # hopper head (tapered box with moulded rim)
    prof = [(0.045, -0.26), (0.05, -0.2), (0.105, -0.07), (0.105, -0.03), (0.125, -0.022), (0.125, 0.0), (0.095, 0.0), (0.095, -0.02)]
    hop = lathe(name + "_hopper", [(a * 1.25, b) for a, b in prof], segments=4)
    _xform(hop, (0, 0, 45))
    transform_verts(hop, lambda co: Vector((co.x * 1.15, co.y * 0.82 - 0.115, co.z + hz)))
    o.append(hop)
    o.append(_bx(name + "_hopper_back", (0.2, 0.012, 0.3), loc=(0, -0.006, hz - 0.12), bevel=0.003))
    o.append(cylinder(name + "_pipe", rr, hz - 0.5, loc=(0, yp - 0.03, (hz - 0.2) / 2 + 0.17), verts=10))
    z = 0.42
    while z < hz - 0.5:
        o.append(cylinder(name + "_collar", rr + 0.013, 0.085, loc=(0, yp - 0.03, z), verts=10))
        o.append(_bx(name + "_ears", (0.21, 0.012, 0.05), loc=(0, -0.05, z), bevel=0.003))
        o.append(_bx(name + "_stand", (0.05, 0.075, 0.035), loc=(0, -0.04, z), bevel=0.003))
        for sx in (-1, 1):
            o.append(bolt(name + "_nail", 0.008, loc=(sx * 0.085, -0.058, z), rot=(90, 0, 0)))
        z += 1.83
    path = [(0, yp - 0.03, 0.34), (0, yp - 0.035, 0.22), (0, yp - 0.085, 0.12), (0, yp - 0.17, 0.085)]
    o.append(sweep(name + "_shoe", path, rr + 0.004, sides=10, smooth_path=False))
    for b in o:
        _xform(b, (0, 0, yaw), loc)
    c, s_ = cos(radians(yaw)), sin(radians(yaw))
    ip = Vector((loc[0] - s_ * (-0.115), loc[1] + c * (-0.115), hz + 0.02))
    p0 = Vector(top)
    path = bezier_path(tuple(p0), tuple(p0 - Vector((0, 0, 0.3))), tuple(ip + Vector((0, 0, 0.32))), tuple(ip - Vector((0, 0, 0.06))), n=12)
    o.append(sweep(name + "_swan_neck", path, rr + 0.002, sides=10, smooth_path=False))
    o.append(cylinder(name + "_outlet", rr + 0.016, 0.06, loc=tuple(p0 - Vector((0, 0, 0.04))), verts=10))
    for b in o:
        assign(b, iron)
    j = join(o, name)
    paint(j, special=mask_mul(noise_mask(0.25, seed=r.pick(range(999)), lo=0.45, hi=0.8), 0.7), grime=mask_mul(height_ramp(0.0, 0.5), 0.7))
    return j


def _blind_window(name, w, h, sur_m, infill_m, stone_sur):
    """A window that was bricked up long ago: the old surround with a brick infill.  Local frame as _plain_window."""
    o = []
    jw = 0.13 if stone_sur else 0.11
    pj = 0.07 if stone_sur else 0.06
    for sx in (-1, 1):
        b = _bx(name + "_jamb", (jw, pj, h), loc=(sx * (w / 2 + jw / 2), -pj / 2, h / 2), bevel=0.008)
        uv_grain(b, "Z")
        o.append(_m(b, sur_m))
    lh = 0.2 if stone_sur else 0.14
    b = _bx(name + "_lintel", (w + 2 * jw + (0.1 if stone_sur else 0.16), pj + 0.015, lh), loc=(0, -(pj + 0.015) / 2, h + lh / 2), bevel=0.01)
    uv_grain(b, "X")
    o.append(_m(b, sur_m))
    sp = pj + 0.07
    sill = _prism_x(name + "_sill", [(0, -0.11), (-sp + 0.01, -0.11), (-sp, -0.03), (-sp, -0.012), (0, 0.012)], -(w / 2 + jw + 0.05), w / 2 + jw + 0.05, 0.007)
    uv_grain(sill, "X")
    o.append(_m(sill, sur_m))
    inf = _bx(name + "_infill", (w, 0.05, h), loc=(0, -0.005, h / 2))
    densify(inf, 0.22)
    o.append(_m(inf, infill_m))
    return join(o, name)



def _prism_y(name, xz, y0, y1, bev=0.0):
    """Polygon in the XZ plane extruded along Y."""
    n = len(xz)
    verts = [(x, y0, z) for x, z in xz] + [(x, y1, z) for x, z in xz]
    faces = [tuple(range(n)), tuple(range(n, 2 * n))[::-1]]
    for i in range(n):
        k = (i + 1) % n
        faces.append((i, k, n + k, n + i))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    if bev > 0:
        bevel(o, bev, 1)
    return o


def _mask_slot(ob, slot, **vals):
    """Raise vertex masks on the faces of one material slot (colour variation inside a shared material)."""
    def fn(val):
        def f(co, n):
            me = ob.data
            pm = np.zeros(len(me.polygons), dtype=np.int32)
            lt = np.zeros(len(me.polygons), dtype=np.int32)
            vi = np.zeros(len(me.loops), dtype=np.int32)
            me.polygons.foreach_get("material_index", pm)
            me.polygons.foreach_get("loop_total", lt)
            me.loops.foreach_get("vertex_index", vi)
            flag = np.zeros(len(co), dtype=np.float32)
            if len(flag) == len(me.vertices):
                flag[vi[np.repeat(pm, lt) == slot]] = val
            return flag
        return f
    paint(ob, **{k: fn(v) for k, v in vals.items()})


# ------------------------------------------------------------------------------------------ roof furniture
def _dormer(name, tanp, sc, pane_m, C, frame_m, gab_m, gauge, lift):
    """Gabled dormer standing on a roof slope of tan(pitch) = tanp.  Local: front to -Y, the origin on the slope at
    the foot of the face, +Y runs up-slope (slope surface z = y * tanp).  Tile courses are returned separately
    so that they take the SAME covering and moss mask as the main roof.  Lead cheeks, soakers, valleys, apron."""
    dw, hc = 1.15 * sc, 1.34 * sc
    dp = radians(47.0)
    tdp = tan(dp)
    ovs, ovf = 0.13, 0.2
    xe = dw / 2 + ovs
    hr = hc + dw / 2 * tdp
    zev = hc - ovs * tdp
    lead, cream = C["lead"], C["cream"]
    o = []
    pt = 0.09
    yf = -0.012
    for sx in (-1, 1):
        b = _bx(name + "_post", (pt, pt, hc), loc=(sx * (dw / 2 - pt / 2), yf + pt / 2, hc / 2), bevel=0.006)
        uv_grain(b, "Z")
        o.append(_m(b, frame_m))
    zs_ = 0.2 * sc + 0.04
    b = _bx(name + "_sill", (dw + 0.08, pt + 0.06, 0.065), loc=(0, yf + pt / 2 - 0.03, zs_ - 0.0325), bevel=0.006)
    uv_grain(b, "X")
    o.append(_m(b, frame_m))
    b = _bx(name + "_head", (dw - 2 * pt + 0.01, pt, 0.1), loc=(0, yf + pt / 2, hc - 0.05), bevel=0.004)
    uv_grain(b, "X")
    o.append(_m(b, frame_m))
    # two-light casement
    wo, z0, z1 = dw - 2 * pt, zs_, hc - 0.1
    ho = z1 - z0
    yc = yf + 0.04
    for sx in (-1, 1):
        o.append(_m(_bx(name + "_stile", (0.04, 0.035, ho), loc=(sx * (wo / 2 - 0.02), yc, (z0 + z1) / 2)), cream))
        o.append(_m(_bx(name + "_bar", (wo / 2 - 0.06, 0.0265, 0.018), loc=(sx * wo / 4, yc - 0.00075, z0 + ho * 0.5)), cream))
        o.append(_m(_bx(name + "_bar", (0.018, 0.0265, ho - 0.07), loc=(sx * wo / 4, yc - 0.00075, (z0 + z1) / 2)), cream))
    for z in (z0 + 0.02, z1 - 0.02):
        o.append(_m(_bx(name + "_rail", (wo - 0.08, 0.035, 0.04), loc=(0, yc, z)), cream))
    o.append(_m(_bx(name + "_mullion", (0.05, 0.05, ho), loc=(0, yc - 0.005, (z0 + z1) / 2)), cream))
    # (CLEARANCE RULE, see _plain_window: one sheet at yc + 0.008, the bars run through it to 4.5 mm behind)
    o.append(_m(lod_faces(_bx(name + "_pane", (wo - 0.06, 0.008, ho - 0.06), loc=(0, yc + 0.012, (z0 + z1) / 2), bevel=0.0), keep=(0, -1, 0), above=-1),
                pane_m))
    # gablet, bargeboards, pendant
    o.append(_m(_prism_y(name + "_gablet", [(-dw / 2, hc - 0.005), (dw / 2, hc - 0.005), (0, hr - 0.02)], yf + 0.02, yf + 0.06), gab_m))
    o.append(_m(_bx(name + "_gablet_post", (0.07, 0.03, hr - hc - 0.12), loc=(0, yf + 0.012, (hr + hc) / 2 - 0.06), bevel=0.004), frame_m))
    bp = [(0.0, -0.15), (0.03, -0.15), (0.03, 0.03), (0.0, 0.03)]
    for sx in (-1, 1):
        o.append(_m(_loft(name + "_barge", [(sx * xe, -ovf, zev), (0, -ovf, hr)], bp, (0, 1, 0)), frame_m))
        # boarded underside of the little roof and its eaves board
        yb = zev / tanp + 0.07
        sf = _sheet(name + "_soffit", [([[(0, -ovf + 0.03, hr - 0.04), (0, hr / tanp + 0.07, hr - 0.04)],
                                         [(sx * xe, -ovf + 0.03, zev - 0.04), (sx * xe, yb, zev - 0.04)]], None, (0, 0, -1))])
        o.append(_m(sf, cream))
        o.append(_m(_bx(name + "_eaves", (0.025, yb + ovf, 0.075), loc=(sx * (xe - 0.0125), (yb - ovf) / 2, zev - 0.018), bevel=0.0), frame_m))
    o.append(_m(_bx(name + "_pendant", (0.055, 0.055, 0.3), loc=(0, -ovf - 0.004, hr - 0.11), bevel=0.01), frame_m))
    # leadwork
    y0c = yf + pt
    for sx in (-1, 1):
        x1, x0 = sx * dw / 2, sx * (dw / 2 - 0.035)
        o.append(_m(_prism_x(name + "_cheek", [(y0c, 0.0), (y0c, hc), (hc / tanp + 0.03, hc)], min(x0, x1), max(x0, x1)), lead))
        ym = y0c + (hc / tanp - y0c) * 0.42
        o.append(_m(_bx(name + "_cheek_roll", (0.02, 0.03, hc - ym * tanp + 0.03), loc=(sx * (dw / 2 + 0.006), ym, (hc + ym * tanp) / 2), bevel=0.006), lead))
        ys = hc / tanp + 0.06
        xa, xb = sx * (dw / 2 - 0.01), sx * (dw / 2 + 0.12)
        o.append(_m(_prism_x(name + "_soaker", [(-0.05, -0.05 * tanp + 0.03), (ys, ys * tanp + 0.03), (ys, ys * tanp + 0.046), (-0.05, -0.05 * tanp + 0.046)],
                             min(xa, xb), max(xa, xb)), lead))
        nrm = Vector((0, -tanp, 1.0)).normalized()
        A = Vector((sx * xe, zev / tanp, zev)) + nrm * 0.032
        B = Vector((0, hr / tanp, hr)) + nrm * 0.032
        o.append(_m(_loft(name + "_valley", [A, B + (B - A) * 0.04], [(-0.09, 0.0), (0.09, 0.0), (0.09, 0.012), (-0.09, 0.012)], (1, 0, 0), up=tuple(nrm)), lead))
    o.append(_m(_prism_x(name + "_apron", [(yf + 0.004, zs_ - 0.07), (yf - 0.012, zs_ - 0.07), (yf - 0.012, 0.05), (-0.22, -0.22 * tanp + 0.052),
                                           (-0.22, -0.22 * tanp + 0.036), (yf + 0.004, 0.02)], -dw / 2 - 0.11, dw / 2 + 0.11), lead))
    o.append(_m(_bx(name + "_upstand", (dw - 2 * pt, 0.03, zs_ + 0.02), loc=(0, yf + 0.03, zs_ / 2 - 0.04), bevel=0.0), lead))
    lr = hr / tanp + ovf + 0.06
    o.append(_m(cylinder(name + "_ridge_roll", 0.036, lr, loc=(0, lr / 2 - ovf, hr + 0.012), rot=(90, 0, 0), verts=8), lead))
    body = join(o, name)
    # tile courses
    sl = xe / cos(dp)
    nk = max(4, int(round(sl / gauge)))
    dzl = lift / cos(dp)
    groups = []
    for sx in (-1, 1):
        for k in range(nk):
            lz = dzl * (1.7 if k == 0 else 1.0)

            def row(d, dz):
                z = zev + d * tdp
                return [(sx * (xe - d), -ovf, z + dz), (sx * (xe - d), z / tanp + 0.08, z + dz)]
            R0, R1, R2 = row(xe * k / nk, 0.0), row(xe * k / nk, lz), row(xe * (k + 1) / nk, 0.0)
            U0 = [(q[1] + 0.3 * sx, k * gauge) for q in R0]
            U2 = [(q[1] + 0.3 * sx, (k + 1) * gauge) for q in R2]
            groups.append(([R0, R1, R2], [U0, U0, U2], (sx * sin(dp), 0.0, cos(dp))))
    tiles = _sheet(name + "_tiles", groups)
    assign(tiles, C["roof"])
    return body, tiles, hr


def _rooflight(name, w, h, iron, pane, lead):
    """Cast-iron rooflight lying in the XY plane (z = tile surface, +Y up-slope)."""
    o = [_m(_bx(name + "_flashing", (w + 0.18, h + 0.18, 0.03), loc=(0, 0, 0.03), bevel=0.004), lead)]
    for sx in (-1, 1):
        o.append(_m(_bx(name + "_frame", (0.05, h, 0.07), loc=(sx * (w / 2 - 0.025), 0, 0.075), bevel=0.004), iron))
        o.append(_m(_bx(name + "_frame", (w - 0.1, 0.05, 0.07), loc=(0, sx * (h / 2 - 0.025), 0.075), bevel=0.004), iron))
    o.append(_m(_bx(name + "_bar", (0.025, h - 0.1, 0.06), loc=(0, 0, 0.082)), iron))
    o.append(_m(_bx(name + "_pane", (w - 0.09, h - 0.09, 0.01), loc=(0, 0, 0.09), bevel=0.0), pane))     # edges 5 mm inside the frame, not in its faces
    return join(o, name)


# ------------------------------------------------------------------------------------------ street furniture
def _lantern(name, iron, glass, emis, gilt):
    """Wrought-iron wall lantern: back plate, arm with scrolled brace, a tapered four-pane lantern standing on the
    arm end.  Local: wall plane y = 0, origin at the arm root.  Returns (object, flame position)."""
    o = []
    Rr = 0.44
    o.append(_m(_bx(name + "_plate", (0.085, 0.014, 0.5), loc=(0, -0.007, -0.06), bevel=0.004), iron))
    for z in (0.14, -0.26):
        o.append(_m(bolt(name + "_bolt", 0.011, 0.008, loc=(0, -0.014, z), rot=(90, 0, 0)), iron))
    o.append(_m(_bx(name + "_arm", (0.026, Rr + 0.03, 0.026), loc=(0, -(Rr + 0.03) / 2, 0.0), bevel=0.004), iron))
    path = bezier_path((0, -0.012, -0.24), (0, -0.16, -0.24), (0, -Rr + 0.12, -0.14), (0, -Rr + 0.04, -0.014), n=8)
    o.append(_m(sweep(name + "_brace", path, 0.011, sides=6), iron))
    o.append(_m(ring_band(name + "_scroll", radius=0.06, width=0.014, thick=0.012, loc=(0, -0.1, -0.085), rot=(0, 90, 0), segments=12), iron))
    z0 = 0.02
    b0, b1, hh = 0.07, 0.105, 0.3
    o.append(_m(_bx(name + "_base", (2 * b0 + 0.03, 2 * b0 + 0.03, 0.022), loc=(0, -Rr, z0 + 0.004), bevel=0.004), iron))
    quads = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        a = (sx * b0, -Rr + sy * b0, z0 + 0.015)
        b = (sx * b1, -Rr + sy * b1, z0 + hh)
        o.append(_m(sweep(name + "_bar", [a, b], 0.007, sides=4, smooth_path=False), iron))
    cs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for i in range(4):
        (ax, ay), (bx_, by) = cs[i], cs[(i + 1) % 4]
        rows = [[(ax * b0, -Rr + ay * b0, z0 + 0.02), (bx_ * b0, -Rr + by * b0, z0 + 0.02)],
                [(ax * b1, -Rr + ay * b1, z0 + hh), (bx_ * b1, -Rr + by * b1, z0 + hh)]]
        quads.append((rows, None, ((ax + bx_) / 2, (ay + by) / 2, 0.0)))
        o.append(_m(_bx(name + "_rim", (2 * b1 + 0.016 if ay == by else 0.016, 0.016 if ay == by else 2 * b1 + 0.016, 0.016),
                        loc=((ax + bx_) / 2 * b1, -Rr + (ay + by) / 2 * b1, z0 + hh), bevel=0.0), iron))
    o.append(_m(_sheet(name + "_glass", quads), glass))
    rf = lathe(name + "_roof", [(0.185, 0.0), (0.12, 0.028), (0.06, 0.07), (0.03, 0.125), (0.016, 0.17), (0.0, 0.175)], segments=4)
    _xform(rf, (0, 0, 45), (0, -Rr, z0 + hh + 0.006))
    o.append(_m(rf, iron))
    o.append(_m(sphere(name + "_finial", 0.02, loc=(0, -Rr, z0 + hh + 0.195), segments=8, rings=5), gilt))
    o.append(_m(cylinder(name + "_burner", 0.022, 0.07, loc=(0, -Rr, z0 + 0.05), verts=8), gilt))
    fl = (0.0, -Rr, z0 + 0.15)
    o.append(_m(sphere(name + "_flame", 0.03, loc=fl, segments=8, rings=6, scale=(1, 1, 1.7)), emis))
    return join(o, name), fl


def _hanging_sign(name, r, hide_m, paint_m, gilt, iron, emis, under):
    """Trade sign: a 0.8 m leather suitcase (straps, corner caps and handle in the shop colour, gilt fittings).  On a plain
    facade it hangs by two chains from a scrolled iron rail, broadside to the street, under two gooseneck lamps;
    below a jetty (`under`) it hangs end-on from a plain bar and takes its light from the shopfront.
    Local: wall plane y = 0, origin on the wall at rail height.  Returns (object, lamp bulb positions)."""
    o = []
    k = 1.27
    case = _suitcase(name + "_case", r, 0.62, 0.4, 0.17, hide_m, paint_m, gilt)
    transform_verts(case, lambda co: co * k)
    L, H, T = 0.62 * k, 0.4 * k, 0.17 * k
    zt = -0.15                                     # top of the case below the rail
    lamps = []
    if under:
        yc = -0.2 - L / 2
        _xform(case, (-90, 0, 90), (T / 2, yc, zt - H / 2))
        o.append(_m(_bx(name + "_plate", (0.08, 0.014, 0.3), loc=(0, -0.007, 0.03), bevel=0.004), iron))
        o.append(_m(_bx(name + "_bar", (0.03, -yc + L / 2 + 0.04, 0.03), loc=(0, (yc - L / 2 - 0.04) / 2, 0.0), bevel=0.004), iron))
        o.append(_m(sphere(name + "_knob", 0.03, loc=(0, yc - L / 2 - 0.05, 0.0), segments=8, rings=5), gilt))
        hang = [(0.0, yc - L * 0.3), (0.0, yc + L * 0.3)]
    else:
        yc = -0.3
        _xform(case, (-90, 0, 180), (0.0, yc - T / 2, zt - H / 2))
        for sx in (-1, 1):
            x = sx * 0.3
            o.append(_m(_bx(name + "_plate", (0.06, 0.014, 0.4), loc=(x, -0.007, 0.07), bevel=0.004), iron))
            o.append(_m(_bx(name + "_arm", (0.026, -yc + 0.05, 0.026), loc=(x, (yc - 0.05) / 2, 0.0), bevel=0.004), iron))
            path = bezier_path((x, -0.012, 0.24), (x, -0.12, 0.24), (x, yc + 0.02, 0.14), (x, yc - 0.02, 0.02), n=8)
            o.append(_m(sweep(name + "_stay", path, 0.009, sides=5), iron))
            o.append(_m(ring_band(name + "_scroll", radius=0.05, width=0.014, thick=0.01, loc=(x, -0.085, 0.07), rot=(0, 90, 0), segments=12), iron))
            o.append(_m(sphere(name + "_knob", 0.028, loc=(sx * (L / 2 + 0.1), yc, 0.0), segments=8, rings=5), gilt))
            # gooseneck lamp reaching out over the case
            path = bezier_path((sx * 0.2, -0.012, 0.34), (sx * 0.2, -0.3, 0.62), (sx * 0.22, -0.74, 0.5), (sx * 0.22, -0.66, 0.22), n=10)
            o.append(_m(sweep(name + "_gooseneck", path, 0.008, sides=5), iron))
            o.append(_m(_bx(name + "_rose", (0.07, 0.012, 0.07), loc=(sx * 0.2, -0.006, 0.34), bevel=0.003), iron))
            sh = cone(name + "_shade", 0.08, 0.02, 0.075, verts=10)
            _xform(sh, (28, 0, 0), (sx * 0.22, -0.65, 0.2))
            o.append(_m(sh, iron))
            bp = (sx * 0.22, -0.635, 0.165)
            o.append(_m(sphere(name + "_bulb", 0.026, loc=bp, segments=8, rings=5), emis))
            lamps.append(bp)
        o.append(_m(_bx(name + "_rail", (L + 0.2, 0.026, 0.026), loc=(0, yc, 0.0), bevel=0.004), iron))
        hang = [(-L * 0.3, yc), (L * 0.3, yc)]
    for (hx, hy) in hang:
        o.append(_m(_bx(name + "_link", (0.01, 0.01, -zt - 0.02), loc=(hx, hy, zt / 2 + 0.005), bevel=0.0), iron))
        o.append(_m(ring_band(name + "_eye", radius=0.022, width=0.008, thick=0.006, loc=(hx, hy, zt + 0.012), rot=(0, 90, 0) if not under else (90, 0, 0),
                              segments=8), iron))
    o.append(case)
    return join(o, name), lamps


def _window_box(name, w, paint_m, cream, iron, hedge, r, back=None):
    """Painted flower box on iron brackets.  Local: wall side at +Y, box bottom centre at the origin.
    back: the planting is pressed flat at this y (it would grow through the lit glazing of the window behind)."""
    o = []
    o.append(_m(_bx(name + "_box", (w, 0.2, 0.17), loc=(0, 0, 0.085), bevel=0.008), paint_m))
    o.append(_m(_bx(name + "_rim", (w + 0.03, 0.23, 0.028), loc=(0, 0, 0.165), bevel=0.006), paint_m))
    for sx in (-1, 1):
        x = sx * (w / 2 - 0.12)
        o.append(_m(_prism_x(name + "_bracket", [(0.12, 0.0), (0.12, -0.2), (0.1, -0.2), (-0.09, -0.02), (-0.09, 0.0)], x - 0.008, x + 0.008), iron))
    nf = max(3, int(round(w / 0.17)))
    for i in range(nf):
        x = -w / 2 + 0.09 + (w - 0.18) * i / (nf - 1)
        rr = r.u(0.085, 0.12)
        lf = sphere(name + "_leaf", rr, loc=(x + r.jitter(0, 0.02), r.jitter(-0.01, 0.03), 0.2 + r.u(0.0, 0.05)), segments=7, rings=5,
                    scale=(1.15, 0.95, r.u(0.75, 1.0)))
        if back is not None:
            apply_transform(lf)
            transform_verts(lf, lambda co: Vector((co.x, min(co.y, back), co.z)))
        o.append(_m(lf, hedge))
        if r.chance(0.55):
            o.append(_m(sphere(name + "_trail", 0.06, loc=(x + r.jitter(0, 0.04), -0.11, 0.1 + r.u(-0.04, 0.03)), segments=6, rings=4, scale=(1.0, 0.5, 1.6)), hedge))
        for k in range(2):
            o.append(_m(sphere(name + "_flower", r.u(0.022, 0.032), loc=(x + r.jitter(0, 0.06), r.u(-0.1, 0.02), 0.24 + r.u(0.02, 0.1)), segments=6, rings=4),
                        cream if (i + k) % 3 == 0 else paint_m))
    return join(o, name)


def _luggage(p, r, cols, yfront, C, n):
    """Stack luggage in front of the stall riser.  cols = [x of column A, x of column B].  Everything is made from
    the building's own materials (shop paint, cream, one leather, one yellow metal): the darker hides are painted
    in by the grime mask when the materials are consolidated."""
    if n <= 0:
        return []
    metal = C["gilt"]
    tan_ = C["leather"]
    dark = mat("leather", "#2a1c14")
    ox = mat("leather", "#5c2a22")
    red, cream = C["paint"], C["cream"]
    xa, xb = cols
    made = []

    def wear_it(o):
        paint(o, wear=mask_mul(noise_mask(0.07, seed=int(p.seed) + len(made), lo=0.5, hi=0.8), 0.7),
              grime=mask_mul(height_ramp(0.0, 0.25), 0.6))
        made.append(o)

    ya = yfront - 0.07
    a0 = (1.0, 0.58, 0.6)
    b0 = (0.8, 0.5, 0.5)
    a1 = (0.7, 0.42, 0.44)
    yaw_a0, yaw_b0 = r.jitter(0, 3), r.jitter(0, 5)
    order = ["a0", "b0", "a1", "lean", "b1", "hat", "bag", "stand"][:n]
    if "a0" in order:
        o = _trunk("trunk_a0", r, *a0, red, tan_, metal, dark, cream, False)
        wear_it(_xform(o, (0, 0, yaw_a0), (xa, ya - a0[1] / 2, 0)))
    if "b0" in order:
        o = _trunk("trunk_b0", r, *b0, ox, tan_, metal, tan_, cream, False)
        wear_it(_xform(o, (0, 0, yaw_b0), (xb, ya - b0[1] / 2 - 0.03, 0)))
    if "a1" in order:
        o = _trunk("trunk_a1", r, *a1, tan_, dark, metal, dark, cream, True)
        wear_it(_xform(o, (0, 0, yaw_a0 + r.jitter(0, 9)), (xa + r.jitter(0, 0.05), ya - a0[1] / 2 + 0.03, a0[2] + 0.009)))
    if "lean" in order:
        L, Hh, T = 0.66, 0.42, 0.19
        o = _suitcase("case_lean", r, L, Hh, T, dark, tan_, metal)
        _xform(o, (-90 - 11, 0, 0))
        mn, mx = _bb(o)
        yb = ya - a0[1] - 0.012
        wear_it(_xform(o, (0, 0, 0), (xa + 0.1 - (mn.x + mx.x) / 2, yb - mx.y, -mn.z)))
    if "b1" in order:
        o = _suitcase("case_b1", r, 0.7, 0.44, 0.2, tan_, dark, metal)
        wear_it(_xform(o, (0, 0, yaw_b0 + r.jitter(180, 7)), (xb + r.jitter(0, 0.03), ya - b0[1] / 2 - 0.03, b0[2] + 0.009)))
    if "hat" in order:
        o = _hatbox("hatbox", r, 0.19, 0.24, cream, tan_, metal)
        z = b0[2] + 0.009 + (0.2 + 0.008 if "b1" in order else 0.0)
        wear_it(_xform(o, (0, 0, r.u(0, 360)), (xb - 0.08, ya - b0[1] / 2 - 0.02, z)))
    if "bag" in order:
        o = _bag("gladstone", r, 0.5, 0.24, 0.32, ox, dark, metal)
        wear_it(_xform(o, (0, 0, r.jitter(18, 8)), (xb + 0.12, ya - b0[1] - 0.22, 0)))
    if "stand" in order:
        L, Hh, T = 0.56, 0.36, 0.17
        o = _suitcase("case_stand", r, L, Hh, T, tan_, dark, metal)
        _xform(o, (-90 - 4, 0, 0))
        mn, mx = _bb(o)
        _xform(o, (0, 0, r.jitter(-12, 5)), (0, 0, 0))
        mn, mx = _bb(o)
        wear_it(_xform(o, (0, 0, 0), (xa + 0.62 - (mn.x + mx.x) / 2, ya - 0.2 - mx.y, -mn.z)))
    return made


# ------------------------------------------------------------------------------------------ draw-call hygiene
def _awning(name, r, w, pr, pitch, vh, stripe_m, cream_m, iron):
    """Shop blind: striped canvas (one strip of cloth per stripe, sagging between the roller and the front lath),
    scalloped valance, roller box on the wall and two iron stay arms.  Local: roller at the origin on the wall
    plane, the cloth runs out to -Y.  (The catalogue awning costs 10k triangles and three canvas materials.)"""
    drop = pr * tan(radians(pitch))
    n = max(7, int(round(w / 0.17)) | 1)           # odd count: both edges the same colour
    sw_ = w / n
    ns, m_ = 4, 5
    mats = (stripe_m, cream_m)
    grp = {0: [], 1: []}

    def cz(t, x):
        return 0.02 - drop * t - 0.04 * sin(pi * t) * (1.0 - 0.6 * abs(2 * x / w) ** 3)

    for i in range(n):
        xa, xb = -w / 2 + i * sw_, -w / 2 + (i + 1) * sw_
        rows, uvr = [], []
        for k in range(ns + 1):
            t = k / ns
            y = -0.05 - (pr - 0.05) * t
            rows.append([(xa, y, cz(t, xa)), (xb, y, cz(t, xb))])
            uvr.append([(xa, y), (xb, y)])
        g = grp[i % 2]
        g.append((rows, uvr, (0, 0, 1)))
        g.append(([[(x, y, z - 0.005) for (x, y, z) in row] for row in rows], uvr, (0, 0, -1)))
        za, zb = cz(1.0, xa), cz(1.0, xb)
        up = [(lerp(xa, xb, q / m_), -pr, lerp(za, zb, q / m_)) for q in range(m_ + 1)]
        lo = [(x, y - 0.006, -drop - vh + 0.06 * (1.0 - sin(pi * q / m_))) for q, (x, y, z) in enumerate(up)]
        uvv = [[(x, z) for (x, y, z) in up], [(x, z) for (x, y, z) in lo]]
        g.append(([up, lo], uvv, (0, -1, 0)))
        g.append(([[(x, y + 0.005, z) for (x, y, z) in up], [(x, y + 0.005, z) for (x, y, z) in lo]], uvv, (0, 1, 0)))
    o = [_m(_sheet(f"{name}_cloth{k}", grp[k]), mats[k]) for k in (0, 1)]
    paint(o, special=mask_mul(noise_mask(0.5, seed=r.pick(range(999)), lo=0.3, hi=0.8), 0.35),
          grime=mask_mul(noise_mask(0.3, seed=r.pick(range(999)), lo=0.4, hi=0.85), 0.3))
    o.append(_m(_bx(name + "_roller_box", (w + 0.12, 0.1, 0.11), loc=(0, -0.05, 0.035), bevel=0.008), stripe_m))
    o.append(_m(cylinder(name + "_front_lath", 0.014, w + 0.03, loc=(0, -pr, -drop + 0.008), rot=(0, 90, 0), verts=8), iron))
    for sx in (-1, 1):
        x = sx * (w / 2 + 0.012)
        o.append(_m(sweep(name + "_arm", [(x, -0.02, -0.72), (x, -pr * 0.55, -0.42 - drop * 0.5), (x, -pr, -drop + 0.008)], 0.011, sides=6,
                          smooth_path=False), iron))
        o.append(_m(_bx(name + "_arm_plate", (0.04, 0.012, 0.14), loc=(x, -0.006, -0.72), bevel=0.003), iron))
        o.append(_m(sphere(name + "_lath_knob", 0.022, loc=(x + sx * 0.012, -pr, -drop + 0.008), segments=8, rings=5), iron))
    return join(o, name)


def _strip_goods(ob, libs=("leather",)):
    """Delete the faces of a placed catalogue component that use the given material libraries (the shop window's
    own coarse stock, which is replaced by the display dressed here; its single glass sheet)."""
    idx = {i for i, sl in enumerate(ob.material_slots) if sl.material and sl.material.get("tsq_lib") in libs}
    if not idx:
        return
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in idx], context="FACES")
    bm.to_mesh(ob.data)
    bm.free()


def _dcase(name, L, H, T, hide, trim, metal, slat=None, handle=True):
    """Display case standing on its long edge, face to the street: strapped, capped corners, lock, top handle (or
    hardwood slats = a small trunk).  Local: bottom centre at the origin, back at y = 0, front at y = -T."""
    o = []
    sh = rounded_box(name + "_shell", (L, H, T), radius=min(0.035, H * 0.12), seg=3)
    _xform(sh, (-90, 0, 0), (0, -T / 2, H / 2))
    _inset(sh, 0.014)
    o.append(_m(sh, hide))
    yf = -T
    for sx in (-1, 1):
        xs = sx * L * 0.25
        o.append(_m(_bx(name + "_strap", (0.032, 0.006, H + 0.004), loc=(xs, yf - 0.002, H / 2)), trim))
        o.append(_m(_bx(name + "_strap", (0.032, T + 0.006, 0.006), loc=(xs, -T / 2, H + 0.002)), trim))
        o.append(_m(_bx(name + "_buckle", (0.046, 0.008, 0.036), loc=(xs, yf - 0.006, H * 0.63)), metal))
        for z in (0.03, H - 0.03):
            o.append(_m(_bx(name + "_cap", (0.058, 0.007, 0.058), loc=(sx * (L / 2 - 0.03), yf - 0.002, z)), metal if slat else trim))
    o.append(_m(_bx(name + "_lock", (0.055, 0.009, 0.04), loc=(0, yf - 0.003, H - 0.045)), metal))
    if slat:
        for z in (H * 0.27, H * 0.73):
            s = _bx(name + "_slat", (L - 0.01, 0.007, 0.036), loc=(0, yf - 0.001, z))
            uv_grain(s, "X")
            o.append(_m(s, slat))
    if handle:
        o.append(_m(_bx(name + "_handle", (0.13, 0.024, 0.014), loc=(0, -T / 2, H + 0.036)), trim))
        for sx in (-1, 1):
            o.append(_m(_bx(name + "_hpost", (0.014, 0.024, 0.034), loc=(sx * 0.058, -T / 2, H + 0.015)), metal))
    return join(o, name)


def _display(name, r, x0, x1, z0, yb, C):
    """The shop-window display between x0 and x1 (floor z0, backdrop at yb, 0.24 m deep): cases on cloth-covered
    risers, a hat box on a pedestal, a small slatted trunk with a Gladstone bag, price tickets."""
    tan = C["leather"]
    dark = mat("leather", "#2a1c14")
    ox = mat("leather", "#5c2a22")
    cream, paint_m, gilt = C["cream"], C["paint"], C["gilt"]
    o = []
    n = max(2, int(round((x1 - x0) / 0.8)))
    sw_ = (x1 - x0) / n
    t0 = r.pick([0, 1, 2])

    def put(ob, x, y, z):
        _xform(ob, (0, 0, r.jitter(0, 2.5)), (x, y, z))
        o.append(ob)

    # half-height panelled window back in the shop's paint, darkened: the cases stand out against it under the
    # display lamps and the lit shop shows above its brass rail
    hb_ = 0.64
    xa, xb = x0 - 0.1, x1 + 0.1
    # (10 mm board: its back stands 6 mm clear of the deepest folds of the lit curtain the shop window hangs behind it)
    bk = [_inset(_bx(name + "_back", (xb - xa, 0.010, hb_), loc=((xa + xb) / 2, yb + 0.009, z0 + hb_ / 2)), 0.02)]
    for z, h_ in ((z0 + 0.04, 0.08), (z0 + hb_ - 0.035, 0.07)):
        bk.append(_bx(name + "_back_rail", (xb - xa, 0.008, h_), loc=((xa + xb) / 2, yb, z)))
    bk.append(_bx(name + "_back_cap", (xb - xa, 0.027, 0.022), loc=((xa + xb) / 2, yb + 0.0005, z0 + hb_ + 0.011)))
    ns = max(2, int(round((xb - xa) / 0.46)))
    for k in range(ns + 1):
        xs = xa + 0.03 + (xb - xa - 0.06) * k / ns
        bk.append(_bx(name + "_back_stile", (0.06, 0.008, hb_ - 0.15), loc=(xs, yb, z0 + hb_ / 2 + 0.005)))
    for b in bk:
        assign(b, paint_m)
    bk = join(bk, name + "_back")
    paint(bk, grime=0.42)
    o.append(bk)
    zr = z0 + hb_ + 0.075
    rod = cylinder(name + "_rail", 0.006, xb - xa - 0.04, loc=((xa + xb) / 2, yb - 0.004, zr), rot=(0, 90, 0), verts=6)
    o.append(_m(rod, gilt))
    for k in range(ns + 1):
        xs = xa + 0.03 + (xb - xa - 0.06) * k / ns
        o.append(_m(cylinder(name + "_rail_post", 0.005, 0.07, loc=(xs, yb - 0.004, zr - 0.03), verts=6), gilt))
        o.append(_m(sphere(name + "_rail_knob", 0.011, loc=(xs, yb - 0.004, zr + 0.008), segments=6, rings=4), gilt))

    for i in range(n):
        xc = x0 + sw_ * (i + 0.5) + r.jitter(0, 0.02)
        kind = (t0 + i) % 3
        if kind == 0:
            wr = min(sw_ - 0.1, 0.64)
            o.append(_m(_inset(_bx(name + "_riser", (wr, 0.16, 0.14), loc=(xc, yb - 0.085, z0 + 0.07)), 0.012), paint_m))
            put(_dcase(name + "_case", min(0.58, wr - 0.04), 0.42, 0.12, tan, dark, gilt), xc, yb - 0.012, z0 + 0.14)
            put(_dcase(name + "_case", 0.33, 0.24, 0.085, ox, tan, gilt), xc + sw_ * 0.14, yb - 0.15, z0)
            tx = xc - sw_ * 0.24
        elif kind == 1:
            xp = xc - sw_ * 0.24
            o.append(_m(_inset(_bx(name + "_pedestal", (0.2, 0.2, 0.3), loc=(xp, yb - 0.115, z0 + 0.15)), 0.012), paint_m))
            hb = _hatbox(name + "_hatbox", r, 0.098, 0.15, cream, paint_m, gilt)
            put(hb, xp, yb - 0.115, z0 + 0.3)
            put(_dcase(name + "_case", min(0.46, sw_ * 0.56), 0.33, 0.11, ox, dark, gilt), xc + sw_ * 0.17, yb - 0.02, z0)
            tx = xc + sw_ * 0.17
        else:
            lt = min(0.62, sw_ - 0.12)
            put(_dcase(name + "_trunk", lt, 0.3, 0.21, paint_m, dark, gilt, slat=tan, handle=False), xc, yb - 0.015, z0)
            put(_bag(name + "_bag", r, 0.34, 0.15, 0.19, tan, dark, gilt), xc + r.jitter(0, 0.05), yb - 0.12, z0 + 0.3)
            tx = xc + lt * 0.3
        tk_ = _bx(name + "_ticket", (0.075, 0.003, 0.05))
        _xform(tk_, (-14, 0, r.jitter(0, 8)), (tx, yb - 0.238, z0 + 0.03))
        o.append(_m(tk_, cream))
    return join(o, name)


def _remap(C, timber, keep=()):
    """The catalogue components each bring their own tints of brass, iron, oak, stone...: a building made of a
    dozen of them costs ~40 draw calls.  Every material slot in the scene is re-pointed at ONE canonical material
    per library; the colour differences that matter (dark fascia board, dark hides, rusty iron) are kept as
    vertex masks inside the shared material."""
    paint_t = [float(c) for c in C["paint"]["tsq_tint"]][:3]
    cream_t = [float(c) for c in C["cream"]["tsq_tint"]][:3]
    board_t = _lin(BOARD)
    canon = set(m.name for m in C.values() if m is not None) | set(keep)

    def dist(a, b):
        return max(abs(max(x, 0.0) ** 0.5 - max(y, 0.0) ** 0.5) for x, y in zip(a, b))

    def target(m):
        lib = m.get("tsq_lib")
        if lib is None:
            return None, None
        t = m.get("tsq_tint")
        t = None if t is None else [float(c) for c in t][:3]
        prm = json.loads(m["tsq_params"]) if m.get("tsq_params") else {}
        lum = _lum(t) if t is not None else None
        woody = C["oak"] if timber else C["paint"]
        if lib.startswith("plaster"):
            return C["plaster"], None
        if lib == "ashlar_limestone":
            return (C["ashlar"] or C["stone"]), None
        if lib.startswith("stone") or lib in ("mortar", "cobble", "concrete"):
            return C["stone"], None
        if lib.startswith("brick"):
            return C["brick"], None
        if lib == "terracotta":
            return C["brick"], {"grime": 0.2}
        if lib.startswith("roof"):
            return C["roof"], None
        if lib == "wood_painted":
            if t is None:
                return C["paint"], None
            if _near(t, board_t, 0.03):
                return C["paint"], {"grime": 0.5}
            return (C["paint"] if dist(t, paint_t) <= dist(t, cream_t) else C["cream"]), None
        if lib.startswith("wood"):
            return woody, None
        if lib in ("canvas", "paper", "rope", "ceramic_glazed", "plain", "fabric", "wax"):
            if t is None:
                return C["cream"], None
            return (C["paint"] if dist(t, paint_t) <= dist(t, cream_t) else C["cream"]), None
        if lib == "iron_rusty":
            return C["iron"], {"special": 0.5}
        if lib.startswith("iron") or lib in ("steel", "copper"):
            return C["iron"], None
        if lib in ("lead", "zinc"):
            return C["lead"], None
        if lib in ("brass", "metal_plain", "bronze", "gold"):
            return C["gilt"], None
        if lib == "leather":
            if lum is not None and lum < 0.03:
                return C["leather"], {"grime": 0.5}
            if lum is not None and lum < 0.07:
                return C["leather"], {"grime": 0.34}
            return C["leather"], None
        if lib == "glass_frosted":
            return (C["lit"] if float(prm.get("glow", 0.0)) > 0.05 else C["glass"]), None
        if lib.startswith("glass"):
            return C["glass"], None
        if lib == "window_lit":
            return C["lit"], None
        if lib == "window_dark":
            return C["dark"], None
        if lib == "emissive":
            return C["emis"], None
        if lib in ("hedge", "leaf", "moss", "soil", "ivy", "grass"):
            return C["hedge"], None
        return None, None

    cache = {}
    for ob in list(bpy.data.objects):
        if ob.type != "MESH":
            continue
        for i, sl in enumerate(ob.material_slots):
            m = sl.material
            if m is None or m.name in canon:
                continue
            if m.name not in cache:
                cache[m.name] = target(m)
            tgt, msk = cache[m.name]
            if tgt is None:
                continue
            if msk:
                _mask_slot(ob, i, **msk)
            sl.material = tgt


# ------------------------------------------------------------------------------------------ building
def build(p, rng):
    W, D, NS = float(p.width), float(p.depth), int(p.storeys)
    seed = int(p.seed)
    age = float(p.age)
    timber = p.style == "timber"
    brickst = p.style == "brick"
    j = clamp(float(p.jetty), 0.0, 0.4) if timber else 0.0
    SF_H = 2.72                                   # shopfront height (door + window units)
    hs = 0.56 if timber else 0.64                 # fascia sign height
    G = SF_H + hs + (0.26 if j > 0.01 else 0.10)  # first-floor level
    S = float(p.storey_height)
    zE = G + (NS - 1) * S                         # eaves (wall plate) level
    ridge_x = p.roof == "eaves_front"
    slate = p.covering == "roof_slate"

    # one material per library: everything in the scene ends up in these (see _remap)
    C = {"plaster": mat("plaster_lime", p.plaster), "brick": mat("brick_red"), "stone": mat("stone_smooth", STONE),
         "paint": mat("wood_painted", p.paint), "cream": mat("wood_painted", JOINERY), "roof": mat(p.covering),
         "iron": mat("iron_wrought"), "gilt": mat("metal_plain", "#d2a84c", rough=0.5), "leather": mat("leather", "#7a4a2a"),
         "lit": mat("window_lit", "#ffc37a", glow=2.2), "dark": mat("window_dark"), "emis": mat("emissive", "#ffd9a0", glow=8.0),
         "lead": mat("lead"), "glass": mat("glass_clear", opacity=0.07, rough=0.03), "hedge": mat("hedge"),
         "oak": mat("wood_oak", TIMBER) if timber else None,
         "ashlar": mat("ashlar_limestone") if p.ground == "stone" else None}
    plaster, stone, joinery, cream, lead, iron = C["plaster"], C["stone"], C["paint"], C["cream"], C["lead"], C["iron"]
    oak = C["oak"]
    up_m = C["brick"] if brickst else plaster
    gmat = {"plaster": plaster, "brick": C["brick"], "stone": C["ashlar"]}[p.ground]
    up_plaster = not brickst
    g_plaster = p.ground == "plaster"

    def wloc(side, u, z, yf=0.0):
        if side == "front":
            return (u, yf, z), 0.0
        if side == "back":
            return (u, D, z), 180.0
        if side == "right":
            return (W / 2, u, z), 90.0
        return (-W / 2, u, z), -90.0

    def wdir(yaw):
        c, s = cos(radians(yaw)), sin(radians(yaw))
        return Vector((c, s, 0.0)), Vector((s, -c, 0.0))      # tangent, outward normal

    # ---------------------------------------------------------------- roof geometry (meshes are made further down)
    hip = p.roof_form == "hip"
    ovh = min(0.45, 0.565 - j) if (ridge_x or hip) else 0.45
    vo = 0.3 if ridge_x else min(0.3, 0.66 - j)
    Dr = D + j
    cy = (D - j) / 2
    Lu, Lv = (W, Dr) if ridge_x else (Dr, W)
    dmx = Lv / 2 + ovh
    pit = min(float(p.pitch), degrees(atan(0.74 * zE / dmx)))     # the roof may not outgrow the walls...
    if Lv > 1.3 * Lu:
        pit = min(pit, 45.0)                                      # ...least of all across a deep plan
    pit = max(pit, 38.0)
    R = _Roof(Lu, Lv, zE, pit, p.roof_form, ovh, vo, 0.03 + 0.07 * age, ridge_x, cy)
    tanp = R.tanp

    def under(x, y):
        return R.top(x, y) - THK + 0.03

    # ---------------------------------------------------------------- wall skins
    PL_H = 0.5
    t_bay = 0.16 if timber else 0.0
    y0u = -j + t_bay
    gl = [(-W / 2, 0.0), (W / 2, 0.0), (W / 2, D), (-W / 2, D)]
    # grain of the skin: ~15 cm, eased on big plans so the walls never take more than ~11k triangles
    wsc = max(1.0, sqrt(2 * (W + D) * (zE + 0.12 * R.rise) / (5000 * 0.15 * 0.28)))

    def skin(sc):
        wk = {"cell": 0.15 * sc, "row": 0.28 * sc}
        ws = []
        if not timber and ((g_plaster and up_plaster) or (p.ground == "brick" and brickst)):
            ws.append((_wall("walls", gl, 0.0, under, **wk), up_m))
        else:
            ws.append((_wall("wall_ground", gl, 0.0, lambda x, y: G + 0.02, **wk), gmat))
            ws.append((_wall("wall_upper", [(-W / 2, y0u), (W / 2, y0u), (W / 2, D), (-W / 2, D)], G, under, **wk), up_m))
            if timber and not ridge_x and not hip:
                ws.append((_wall("wall_gable_front", [(-W / 2, -j + 0.02), (W / 2, -j + 0.02)], zE - 0.03, under, closed=False, **wk), up_m))
        for k, (o, m_) in enumerate(ws):
            displace_noise(o, 0.005, 1.3, seed=seed + k)
            assign(o, m_)
        tuck(ws, sc)
        return ws

    # CLEARANCE RULE: the sash windows and the shop windows stand on the uncut wall with their lit room card / lit
    # curtain 6 mm in front of the wall plane, and the wall relief (+-5 mm) would come to within 1-3 mm of it: behind
    # each of them (the opening and one grid cell round it) the relief is kept at or behind the plane, so the glowing
    # sheet has its full 6 mm.
    lit_cards = []                                # (x, z, w, h) of those openings on the front wall

    def tuck(ws, sc, cards=None):
        for (x, z, w, h) in (lit_cards if cards is None else cards):
            mx, mz = 0.15 * sc + 0.03, 0.28 * sc + 0.03
            for o, _ in ws:
                transform_verts(o, lambda co: Vector((co.x, 0.0, co.z)) if (-0.02 < co.y < 0.0 and abs(co.x - x) < w / 2 + mx
                                                                            and z - mz < co.z < z + h + mz) else co)

    walls = skin(wsc)
    if timber and j > 0.01:
        sf = _sheet("jetty_soffit", [([[(-W / 2, -j, G - 0.004), (W / 2, -j, G - 0.004)], [(-W / 2, 0.03, G - 0.004), (W / 2, 0.03, G - 0.004)]], None)], ref=(0, 0, -1))
        assign(sf, plaster)
        paint(sf, grime=0.45)
    sills = []                                    # (centre, tangent, normal, width, streak length): rain marks on the walls
    hoppers = []                                  # (point at the hopper, tangent, normal)

    # ---------------------------------------------------------------- stone plinth, quoins, string course
    rs = rng.fork("stone")
    bases = [box(f"plinth_base{i}", (0.6, 0.16, PL_H), bevel=0.016, segments=1) for i in range(3)]
    pm = [[], [], []]

    def plinth_run(ax, ay, bx, by, yaw):
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length < 0.12:
            return
        n = max(1, round(length / 0.62))
        ws = [rs.u(0.75, 1.25) for _ in range(n)]
        tot = sum(ws)
        t = 0.0
        for w in ws:
            l = w / tot * length
            c = (t + l / 2) / length
            t += l
            pm[rs.pick([0, 1, 2])].append(trs((lerp(ax, bx, c), lerp(ay, by, c), PL_H / 2 * rs.u(0.985, 1.0)), (0, 0, yaw),
                                              ((l - 0.007) / 0.6, 1.0, rs.u(0.985, 1.0))))

    # shopfront layout
    pier = 0.55
    avail = W - 2 * pier
    layout = p.door_side
    if layout == "centre" and avail - 1.3 < 3.3:
        layout = "left"
    if layout == "centre":
        door_w = 1.3
        win_w = min(4.6, (avail - door_w) / 2)
        sfw = door_w + 2 * win_w
        door_x = 0.0
        win_xs = [sfw / 2 - win_w / 2, -(sfw / 2 - win_w / 2)]
        lan_sx = -1
    else:
        door_w = clamp(avail - 4.6, 1.25, 1.9)
        win_w = min(4.6, avail - door_w)
        sfw = door_w + win_w
        sg = -1 if layout == "left" else 1
        door_x = sg * (sfw / 2 - door_w / 2)
        win_xs = [-sg * (sfw / 2 - win_w / 2)]
        lan_sx = sg
    pw = (W - sfw) / 2                            # masonry pier each side of the shopfront

    plinth_run(-W / 2 + 0.03, -0.05, -W / 2 + 0.03, D + 0.05, 90)
    plinth_run(W / 2 - 0.03, -0.05, W / 2 - 0.03, D + 0.05, 90)
    plinth_run(-W / 2, D - 0.03, W / 2, D - 0.03, 0)
    plinth_run(-W / 2, 0.03, -sfw / 2 + 0.02, 0.03, 0)
    plinth_run(sfw / 2 - 0.02, 0.03, W / 2, 0.03, 0)
    back_w = lambda co, n: 0.35 + 0.65 * np.clip(n[:, 1], 0.0, 1.0)      # the damp side: away from the street
    for i, b in enumerate(bases):
        assign(b, stone)
        if pm[i]:
            o = instances_to_mesh(f"plinth{i}", b, pm[i], keep_base=False)
            paint(o, grime=mask_mul(height_ramp(0.0, 0.42), noise_mask(0.5, seed=seed + 7, lo=0.1, hi=0.8), 0.9),
                  special=mask_mul(noise_mask(0.45, seed=seed + 8, lo=0.3, hi=0.65), back_w, 0.5 + 0.5 * age))
        else:
            delete(b)

    q_top = zE - 0.17 if not timber else G - (0.2 if j > 0.01 else 0.0)
    nq = max(1, round((q_top - PL_H) / 0.31))
    hq = (q_top - PL_H) / nq
    qb = [box("quoin_a", (0.47, 0.29, 0.3), bevel=0.014, segments=1), box("quoin_b", (0.29, 0.47, 0.3), bevel=0.014, segments=1)]
    qm = [[], []]
    for sx in (-1, 1):
        for cy_, sy in ((0.0, -1), (D, 1)):
            for k in range(nq):
                a = (k + (sx > 0) + (sy > 0)) % 2
                ax, ay = (0.47, 0.29) if a == 0 else (0.29, 0.47)
                qm[a].append(trs((sx * (W / 2 + 0.03 - ax / 2), cy_ + sy * (0.03 - ay / 2), PL_H + hq * (k + 0.5)), (0, 0, 0),
                                 (rs.u(0.96, 1.03), rs.u(0.96, 1.03), (hq - 0.008) / 0.3)))
    for i, b in enumerate(qb):
        assign(b, stone)
        o = instances_to_mesh(f"quoins{i}", b, qm[i], keep_base=False)
        paint(o, grime=mask_mul(noise_mask(0.35, seed=seed + 9, lo=0.25, hi=0.8), 0.45),
              special=mask_mul(noise_mask(0.6, seed=seed + 10, lo=0.4, hi=0.75), back_w, height_ramp(zE, 0.0), 0.8))

    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    if not timber:
        sc_p = [(-0.02, 0.0), (0.05, 0.0), (0.05, 0.1), (0.085, 0.125), (0.085, 0.185), (-0.02, 0.2)]
        o = _loft("string_course", [(sx * W / 2, D / 2 + sy * D / 2, G + 0.01) for sx, sy in corners], sc_p,
                  [Vector((sx, sy, 0)) for sx, sy in corners], closed=True)
        densify(o, 0.4)
        assign(o, stone)
        paint(o, grime=mask_mul(facing((0, 0, -1)), 0.6), special=mask_mul(facing((0, 0, 1)), noise_mask(0.4, seed=seed + 12), 0.8))

    # ---------------------------------------------------------------- shopfront: window(s), door, fascia
    def comp_place(c, loc, yaw=0.0):
        return c.place(loc, rot=(0, 0, yaw), sockets=False)

    sw = use("shop_window", width=win_w, height=SF_H, bays=max(2, round(win_w / 0.88)), paint=p.paint, goods=p.goods, lit=True,
             depth=0.42, riser=0.62, transom=0.34, fascia=0.28, sign_text="", wear=age, seed=seed + 1)
    rdp = rng.fork("display")
    for i, x in enumerate(win_xs):
        so_ = comp_place(sw, (x, 0, 0))
        lit_cards.append((x, 0.0, win_w, SF_H))   # (the lit curtain at the back of the window hangs 6 mm off the wall plane)
        tuck(walls, wsc, lit_cards[-1:])
        # the catalogue pane is one sheet of glass in front of everything: the renderer lays a milky veil over
        # whatever stands behind it (the stock reads as grey card).  The display panes are left open and only the
        # top lights are glazed.
        _strip_goods(so_, ("glass_clear",))
        _m(_bx(f"shop_toplight_glass{i}", (win_w - 0.42, 0.004, 0.3), loc=(x, -0.31, 2.27 - 0.15)), C["glass"])
        if p.goods == "luggage":                  # the window is dressed here: strapped cases on risers, hat box, trunk
            _strip_goods(so_)
            _display(f"window_display{i}", rdp, x - win_w / 2 + 0.3, x + win_w / 2 - 0.3, 0.69, -0.045, C)
    door = use("door_shop", width=door_w, height=SF_H, paint=p.paint, trim=p.paint, number=p.number, lit=True, reveal=0.36,
               hand="left" if door_x > 0.01 else "right", wear=age, seed=seed + 2)
    comp_place(door, (door_x, 0, 0))

    yb = 0.27                                     # sign box depth: brings the fascia forward over the shopfront cornice
    fw = min(6.0, sfw)
    sign = str(p.sign_text)
    if p.sign_spaced and len(sign) <= 9 and " " not in sign:
        sign = " ".join(sign)                     # sign-writer's letter-spacing fills the long board
    fs = use("fascia_sign", text=sign, width=fw, height=hs, caps=bool(p.sign_caps), font=p.sign_font, board=BOARD,
             trim=p.paint, frame="moulded", consoles=True, lamps=2, lamp_reach=0.34, age=age * 0.6, seed=seed + 3)
    comp_place(fs, (0, -yb, SF_H))
    sb = box("sign_box", (sfw - 0.02, yb + 0.02, hs), loc=(0, -yb / 2 + 0.01, SF_H + hs / 2), bevel=0.004)
    uv_grain(sb, "X")
    assign(sb, joinery)
    cap = box("sign_capping", (sfw + 0.08, yb + 0.1, 0.012), loc=(0, -(yb + 0.1) / 2 + 0.01, SF_H + hs + 0.022), rot=(7, 0, 0), bevel=0.003)
    assign(cap, lead)
    paint(cap, grime=0.3)

    # ---------------------------------------------------------------- jetty carpentry
    if j > 0.01:
        jo = []
        nj = max(5, round(W / 0.52))
        for i in range(nj):
            x = -W / 2 + 0.08 + (W - 0.16) * i / (nj - 1)
            b = box("joist", (0.13, j + 0.2, 0.18), loc=(x, (0.16 - j - 0.04) / 2, G - 0.09), bevel=0.012)
            uv_grain(b, "Y")
            jo.append(b)
        zt = G - 0.18
        for sx in (-1, 1):
            x = sx * (W / 2 - min(pw, 0.5) / 2)
            P0, Cc, P1 = (-j, zt - 0.11), (-j * 0.2, zt - 0.2), (0.0, zt - 0.7)
            curve = [((1 - t) ** 2 * P0[0] + 2 * (1 - t) * t * Cc[0] + t * t * P1[0], (1 - t) ** 2 * P0[1] + 2 * (1 - t) * t * Cc[1] + t * t * P1[1])
                     for t in [k / 8 for k in range(9)]]
            br = _prism_x("jetty_bracket", [(0.05, zt - 0.7), (0.05, zt), (-j, zt)] + curve, x - 0.07, x + 0.07, 0.008)
            uv_grain(br, "Z")
            jo.append(br)
        for b in jo:
            assign(b, oak)
        o = join(jo, "jetty")
        paint(o, grime=0.25)

    # ---------------------------------------------------------------- upper floors: front
    rl = rng.fork("lit")
    rw = rng.fork("win")
    sur_m = oak if timber else stone
    n_plain = [0]

    def plain(side, u, z, w, h, lit, yf=0.0):
        o = _plain_window("win", w, h, lit, cream, sur_m, C["lit"] if lit else C["dark"], cream, not timber, rw)
        loc, yaw = wloc(side, u, z, yf)
        _xform(o, (0, 0, yaw), loc)
        n_plain[0] += 1
        paint(o, grime=mask_mul(noise_mask(0.22, seed=seed + 100 + n_plain[0], lo=0.2, hi=0.85), 0.2 + 0.25 * age),
              special=mask_mul(facing((0, 0, 1), 2.0), noise_mask(0.3, seed=seed + 13, lo=0.2, hi=0.7), 0.4 + 0.5 * age))
        tg, nr = wdir(yaw)
        sills.append((Vector(loc), tg, nr, w + (0.36 if not timber else 0.32), rw.u(0.7, 1.7)))

    if timber:
        ww, wh, sill = 1.0, 1.25, 0.92
        cas = dict(width=ww, height=wh, lights=2, head="flat", surround="timber", glazing="diamond", shutters=False,
                   frame_paint=JOINERY, timber=TIMBER, age=age)
        f_lit = use("window_casement", lit=True, seed=seed + 20, **cas)
        f_dark = use("window_casement", lit=False, seed=seed + 21, **cas)
        bay = dict(width=W / 2, height=S, pattern=p.pattern, plaster=p.plaster, timber=TIMBER, jetty=False, opening_w=ww + 0.02,
                   opening_h=wh + 0.02, opening_sill=sill, age=age)
        bay_l = use("timber_frame_bay", posts="both", seed=seed + 30, **bay)
        bay_r = use("timber_frame_bay", posts="right", seed=seed + 31, **bay)
        front_xs = [-W / 4, W / 4]
    else:
        ww, wh, sill = 0.95, 1.6, 0.82
        sash = dict(width=ww, height=wh, panes="6/6", surround="stone", guard=False, paint=JOINERY, surround_colour=STONE, wear=age, spill=0.0)
        f_lit = use("window_sash", lit=True, head="keystone", seed=seed + 20, **sash)
        f_dark = use("window_sash", lit=False, head="keystone", seed=seed + 21, **sash)
        front_xs = [-W * 0.235, W * 0.235] if W < 6.6 else [-W * 0.31, 0.0, W * 0.31]
    rb = rng.fork("boxes")
    for k in range(NS - 1):
        z0 = G + k * S
        if timber:
            comp_place(bay_l, (-W / 4, -j, z0))
            comp_place(bay_r, (W / 4, -j, z0))
        lits = [rl.chance(p.lit) for _ in front_xs]
        if k == 0 and not any(lits) and p.lit > 0.05:
            lits[rl.pick(range(len(lits)))] = True
        if k == 0 and all(lits) and p.lit < 0.95:
            lits[rl.pick(range(len(lits)))] = False
        attic = NS == 3 and k == 1                 # the top floor of a three-storey house gets small plain casements
        for x, lt in zip(front_xs, lits):
            if attic:
                if timber:
                    plain("front", x, z0 + sill, ww - 0.2, wh - 0.13, lt, yf=-j - 0.01)
                    sills.pop()                    # (the catalogue bay is not one of the painted wall skins)
                else:
                    plain("front", x, z0 + 0.95, 0.9, 1.15, lt)
            else:
                comp_place(f_lit if lt else f_dark, (x, -j, z0 + sill))
                if not timber:                 # (lit or not: _remap gives every sash room card the lit material)
                    lit_cards.append((x, z0 + sill, ww, wh))
                    tuck(walls, wsc, lit_cards[-1:])
                if not timber:
                    sills.append((Vector((x, 0.0, z0 + sill)), Vector((1, 0, 0)), Vector((0, -1, 0)), ww + 0.42, rw.u(0.8, 1.6)))
                if p.window_boxes and k == 0:
                    sp, _ = _sock(f_lit, "planter", (x, -j, z0 + sill))
                    sp = sp or (x, -j - 0.22, z0 + sill)
                    # (CLEARANCE RULE: the leaded lights of the casement stand 30 mm off the bay; the leaves stop 6.5 mm short)
                    o = _window_box("window_box", ww * 0.95, joinery, cream, iron, C["hedge"], rb, back=(-j - 0.0365 - sp[1]) if timber else None)
                    _xform(o, (0, 0, 0), (sp[0], sp[1], sp[2] - 0.13))
                    paint(o, grime=mask_mul(height_ramp(sp[2] + 0.02, sp[2] - 0.15), 0.4), wear=mask_mul(noise_mask(0.1, seed=seed + 41, lo=0.55, hi=0.8), 0.5 * age))

    # ---------------------------------------------------------------- sides and back: plain windows, back door, framing
    n_side = 2 if D < 8.5 else 3
    side_us = [D * (i + 0.5) / n_side for i in range(n_side)]
    free_u = D / 2 if n_side == 2 else D / 3
    back_xs = [-W * 0.23, W * 0.23]
    g_sill = 1.1
    pw_, ph_ = 0.9, 1.35
    for side in ("left", "right"):
        plain(side, D * 0.62, g_sill, pw_, 1.25, rl.chance(p.lit * 0.6))
        for k in range(NS - 1):
            for u in side_us:
                plain(side, u, G + k * S + sill, pw_, ph_, rl.chance(p.lit * 0.8))
    bd = _back_door("back_door", 0.95, 2.02, rw, joinery, oak if timber else joinery, iron, stone, lead)
    _xform(bd, (0, 0, 180), (W * 0.22, D, 0))
    paint(bd, grime=mask_mul(height_ramp(0.0, 0.7), 0.55), wear=mask_mul(noise_mask(0.07, seed=seed + 51, lo=0.55, hi=0.8), height_ramp(0.1, 0.9), 0.15 + 0.4 * age))
    plain("back", -W * 0.2, g_sill, pw_, 1.25, True)
    bw = _blind_window("blocked_window", pw_, 1.25, sur_m, C["brick"], not timber)       # bricked up long ago
    loc, yaw = wloc("left", D * 0.3, g_sill)
    _xform(bw, (0, 0, yaw), loc)
    paint(bw, grime=mask_mul(noise_mask(0.25, seed=seed + 53, lo=0.2, hi=0.8), 0.35 + 0.2 * age),
          special=mask_mul(noise_mask(0.3, seed=seed + 54, lo=0.3, hi=0.7), height_ramp(g_sill, g_sill + 0.9), 0.5 + 0.4 * age))
    tg, nr = wdir(yaw)
    sills.append((Vector(loc), tg, nr, pw_ + 0.36, 1.0))
    for k in range(NS - 1):
        for x in back_xs:
            plain("back", x, G + k * S + sill, pw_, ph_, rl.chance(p.lit))

    if not timber:
        an = []
        for side, nrm in (("left", Vector((-1, 0, 0))), ("right", Vector((1, 0, 0)))):
            for k in range(NS - 1):
                za = G + k * S - (0.2 if k == 0 else 0.12)
                for u in (D * 0.25, D * 0.75):
                    c = Vector(wloc(side, u, za)[0])
                    for sg_ in (-1, 1):
                        d_ = Vector((0, 0.2, 0.2 * sg_))
                        an.append(_beam("wall_anchor", c - d_, c + d_, 0.045, 0.014, nrm, proud=0.016, bev=0.0))
                    an.append(bolt("anchor_nut", 0.022, 0.016, loc=tuple(c + nrm * 0.016), rot=(0, 90 * nrm.x, 0)))
                    sills.append((c + Vector((0, 0, 0.12)), Vector((0, 1, 0)), nrm, 0.02, rw.u(0.5, 1.0)))
        for b in an:
            assign(b, iron)
        o = join(an, "wall_anchors")
        paint(o, special=0.5 + 0.4 * age)

    tm = []                                       # oak framing that is built here (sides, back, gables)
    rt = rng.fork("frame")
    if timber:
        fwalls = [("left", Vector((-1, 0, 0)), y0u + 0.02, D, side_us), ("right", Vector((1, 0, 0)), y0u + 0.02, D, side_us),
                  ("back", Vector((0, 1, 0)), -W / 2, W / 2, back_xs)]
        for side, nrm, u0, u1, wins in fwalls:
            def P(u, z):
                return Vector(wloc(side, u, z)[0])
            for k in range(NS - 1):
                zf = G + k * S
                tm.append(_beam("girt", P(u0, zf + 0.1), P(u1, zf + 0.1), 0.2, 0.14, nrm))
                zt = zf + S - (0.18 if k == NS - 2 else 0.0)
                marks = sorted([u0 + 0.1] + list(wins) + [u1 - 0.1])
                for a, b in zip(marks[:-1], marks[1:]):
                    um = (a + b) / 2 + rt.jitter(0, 0.03)
                    tm.append(_beam("stud", P(um, zf + 0.2), P(um + rt.jitter(0, 0.015), zt), rt.u(0.15, 0.18), 0.12, nrm))
                tm.append(_beam("sill_rail", P(u0, zf + sill - 0.07), P(u1, zf + sill - 0.07), 0.13, 0.11, nrm, proud=0.02))
                for uc, sg_ in ((u0, 1), (u1, -1)):
                    tm.append(_beam("brace", P(uc + sg_ * 0.95, zf + 0.2), P(uc + sg_ * 0.16, zf + sill - 0.1), 0.14, 0.11, nrm, proud=0.022))
            tm.append(_beam("plate", P(u0, zE - 0.09), P(u1, zE - 0.09), 0.18, 0.14, nrm))
        for sx in (-1, 1):
            b = box("corner_post", (0.21, 0.21, zE - G), loc=(sx * (W / 2 - 0.105 + 0.03), D - 0.105 + 0.03, (G + zE) / 2), bevel=0.01)
            uv_grain(b, "Z")
            tm.append(b)

    # ---------------------------------------------------------------- gables: attic windows, timber gable frames
    if not hip:
        for su in (-1, 1):
            street = (not ridge_x) and su < 0
            off = 0.02 if (street and timber) else 0.0
            nrm = R.Wv(su, 0.0)

            def Pg(v, z):
                return R.W(su * Lu / 2, v, z) - nrm * off

            def zt(v):
                return float(R.top_uv(su * (Lu / 2 - off), v)) - THK - 0.02
            side = ("left" if su < 0 else "right") if ridge_x else ("front" if su < 0 else "back")
            uc = cy if ridge_x else 0.0
            zw = zE + 0.52
            win = None
            for (w_, h_) in ((0.75, 0.95), (0.6, 0.72)):
                if zt(w_ / 2 + 0.2) - (zw + h_ + 0.3) > 0.12:
                    win = (w_, h_)
                    break
            if win:
                plain(side, uc, zw, win[0], win[1], True if street else rl.chance(p.lit), yf=(-j + off) if street else 0.0)
            if timber:
                tm.append(_beam("gable_tie", Pg(-Lv / 2 + 0.02, zE + 0.13), Pg(Lv / 2 - 0.02, zE + 0.13), 0.22, 0.16, nrm, proud=0.04))
                vk = R.hw if p.roof_form == "half_hip" else 0.0
                vs_ = [-Lv / 2 + 0.14, -vk, vk, Lv / 2 - 0.14] if vk > 0.05 else [-Lv / 2 + 0.14, 0.0, Lv / 2 - 0.14]
                for a, b in zip(vs_[:-1], vs_[1:]):
                    tm.append(_beam("gable_rafter", Pg(a, zt(a) - 0.1), Pg(b, zt(b) - 0.1), 0.17, 0.13, nrm, proud=0.03))
                apex = zt(0.0)
                if apex - zE > 2.6:
                    zc = zE + 0.13 + 0.56 * (apex - zE - 0.13)
                    hv = max(0.4, (apex - 0.2 - zc) / tanp + vk)
                    tm.append(_beam("gable_collar", Pg(-hv, zc), Pg(hv, zc), 0.15, 0.12, nrm, proud=0.028))
                nst = max(2, int(round((Lv - 0.9) / 0.52)))
                for i in range(nst + 1):
                    v = -Lv / 2 + 0.45 + (Lv - 0.9) * i / nst
                    top = zt(v) - 0.2
                    bot = zE + 0.22
                    if win and abs(v) < win[0] / 2 + 0.16:
                        bot = zw + win[1] + 0.16
                    if top - bot > 0.28:
                        tm.append(_beam("gable_stud", Pg(v, bot), Pg(v + rt.jitter(0, 0.012), top), rt.u(0.11, 0.14), 0.11, nrm, proud=0.024))
    if tm:
        for b in tm:
            assign(b, oak)
        o = join(tm, "oak_frame")
        displace_noise(o, 0.002, 0.5, seed=seed + 60)
        paint(o, grime=mask_mul(noise_mask(0.4, seed=seed + 61), 0.5), wear=mask_mul(noise_mask(0.2, seed=seed + 62, lo=0.55, hi=0.8), 0.5))

    # ---------------------------------------------------------------- roof: tiles, soffits, verges, crest, gutters
    rr_ = rng.fork("roof")
    gauge, lift = (0.15, 0.009) if slate else (0.1, 0.014)
    moss_n = noise_mask(0.55, seed=seed + 70, lo=0.38, hi=0.78)
    soot_n = noise_mask(1.3, seed=seed + 71, lo=0.25, hi=0.8)

    def moss(co, n):
        h = np.clip((co[:, 2] - R.ze) / R.rise, 0.0, 1.0)
        shade = 0.5 + 0.5 * np.clip(n[:, 1], -1.0, 1.0)
        return np.clip(moss_n(co, n) * (0.2 + 0.55 * (1.0 - h) + 0.45 * shade) * (0.25 + 0.95 * age), 0.0, 1.0)

    def soot(co, n):
        h = np.clip((co[:, 2] - R.ze) / R.rise, 0.0, 1.0)
        return np.clip(0.3 * soot_n(co, n) + 0.22 * (1.0 - h) ** 3 + 0.1 * age, 0.0, 0.7)

    def lay_tiles(cap):
        o = _sheet("roof_tiles", R.tile_groups(gauge, lift, rng.fork("roof_tiles"), cap))
        assign(o, C["roof"])
        paint(o, special=moss, grime=soot)
        return o

    tiles = lay_tiles(8000)
    soff_m = oak if timber else cream
    sof = _sheet("roof_soffit", R.soffit_groups())
    assign(sof, soff_m)
    paint(sof, grime=0.2 + 0.25 * age, wear=mask_mul(noise_mask(0.15, seed=seed + 72, lo=0.55, hi=0.8), 0.6 * age))
    trim = []
    if not hip:
        dend = R.dmax if p.roof_form == "gable" else R.dh
        ds = [d for d in (0.0, 0.3, 0.6, 0.9) if d < dend - 0.15] + [dend]
        for su in (-1, 1):
            for sg in (-1, 1):
                b = _loft("bargeboard", [R.P(sg, su * R.Uv, d) for d in ds], [(-0.04, -0.26), (0.0, -0.26), (0.0, 0.04), (-0.04, 0.04)], R.Wv(su, 0.0))
                trim.append(_m(b, joinery))
            if p.roof_form == "gable":
                a = R.P(-1, su * R.Uv, R.dmax)
                b = _bx("verge_pendant", (0.07, 0.07, 0.55), loc=tuple(a + R.Wv(su * -0.02, 0.0, -0.12)), bevel=0.012)
                trim.append(_m(b, joinery))
    o = join(trim, "roof_trim") if trim else None
    if o is not None:
        paint(o, grime=0.15 + 0.2 * age, wear=mask_mul(noise_mask(0.12, seed=seed + 73, lo=0.5, hi=0.78), 0.3 + 0.5 * age), special=0.4 * age)
    crest_m = lead if slate else C["roof"]
    base = cylinder("ridge_tile_base", 0.062 if slate else 0.095, 0.36, verts=10)
    _xform(base, (0, 90, 0))
    assign(base, crest_m)
    fm = []
    for line in R.crest_lines():
        for a, b in zip(line[:-1], line[1:]):
            L_ = (b - a).length
            n = max(1, int(round(L_ / 0.34)))
            for k in range(n):
                fm.append(_frame(a.lerp(b, k / n), a.lerp(b, (k + 1) / n), lift=-0.03 + rr_.u(0.0, 0.012), sx=(L_ / n) / 0.36 * 1.05))
    o = instances_to_mesh("ridge_tiles", base, fm, keep_base=False)
    paint(o, special=mask_mul(noise_mask(0.4, seed=seed + 74, lo=0.4, hi=0.8), 0.3 + 0.6 * age), grime=0.3)
    fin = []
    for su in (-1, 1):
        a = R.P(-1, su * (R.ur - 0.1), R.dmax)
        f = lathe("finial", [(0.0, 0.0), (0.06, 0.0), (0.07, 0.05), (0.04, 0.1), (0.034, 0.15), (0.08, 0.22), (0.085, 0.27), (0.05, 0.33), (0.0, 0.35)],
                  segments=10, loc=(a.x, a.y, a.z + 0.03))
        fin.append(_m(f, crest_m))
    paint(join(fin, "finials"), grime=0.35)

    # gutters (half-round cast iron on the eaves; all round a hipped roof)
    gp = []
    ro, ri = 0.065, 0.052
    for k in range(7):
        gp.append((ro * cos(pi + pi * k / 6), ro * sin(pi + pi * k / 6)))
    for k in range(7):
        gp.append((ri * cos(2 * pi - pi * k / 6), ri * sin(2 * pi - pi * k / 6)))
    GD, GO = -0.045, 0.085                        # gutter centre: below the tile edge, out from the eaves board
    gut = []
    if hip:
        pts = [R.P(sg, su * R.Uv, 0.0, GD) + R.Wv(su, sg) * GO for su, sg in corners]
        gut.append(_loft("gutter", pts, gp, [R.Wv(su, sg) for su, sg in corners], closed=True))
    else:
        for sg in (-1, 1):
            out = R.Wv(0.0, sg)
            ug = R.Uv - 0.03
            gut.append(_loft("gutter", [R.P(sg, -ug, 0.0, GD) + out * GO, R.P(sg, ug, 0.0, GD) + out * GO], gp, out))
    for sg in (-1, 1):
        out = R.Wv(0.0, sg)
        nb = max(2, int(round(2 * R.Uv / 0.95)))
        for i in range(nb):
            u = (-R.Uv + 0.25) + (2 * R.Uv - 0.5) * i / (nb - 1)
            c = R.P(sg, u, 0.0, GD - 0.035) + out * 0.045
            gut.append(_beam("gutter_bracket", c - out * 0.045, c + out * 0.11, 0.025, 0.012, (0, 0, -1), proud=0.012, bev=0.0))
    hz = zE - 0.55
    pipes = []                                    # (gutter outlet, wall side, position along the wall)
    if ridge_x:
        eu = -lan_sx
        if j > 0.01:
            pipes.append((R.P(-1, eu * (W / 2 + 0.115), 0.0, GD - ro) + R.Wv(0, -1) * GO, "right" if eu > 0 else "left", 0.32))
        else:
            pipes.append((R.P(-1, eu * (W / 2 - 0.24), 0.0, GD - ro) + R.Wv(0, -1) * GO, "front", eu * (W / 2 - 0.24)))
        pipes.append((R.P(1, -eu * (W / 2 - 0.3), 0.0, GD - ro) + R.Wv(0, 1) * GO, "back", -eu * (W / 2 - 0.3)))
    else:
        for sg, yw in ((-1, D - 0.45), (1, 0.4 if j < 0.01 else 0.32)):
            pipes.append((R.P(sg, yw - cy, 0.0, GD - ro) + R.Wv(0, sg) * GO, "right" if sg < 0 else "left", yw))
    rp = rng.fork("rain")
    for top, side, u in pipes:
        loc, yaw = wloc(side, u, 0.0)
        gut.append(_downpipe("downpipe", tuple(top), loc, yaw, hz, iron, rp))
        tg, nr = wdir(yaw)
        hoppers.append((Vector((loc[0], loc[1], hz - 0.25)), tg, nr))
    for b in gut:
        assign(b, iron)
    o = join(gut, "rainwater")
    paint(o, special=mask_mul(noise_mask(0.3, seed=seed + 75, lo=0.45, hi=0.8), 0.35 + 0.5 * age))

    # eaves: moulded stone cornice (masonry) or rafter feet (oak frame)
    if not timber:
        cp = [(-0.02, 0.0), (0.035, 0.0), (0.05, 0.06), (0.095, 0.1), (0.13, 0.15), (0.13, 0.215), (-0.02, 0.215)]
        zc = zE - 0.17
        cor = []
        if hip:
            cor.append(_loft("cornice", [R.W(su * Lu / 2, sg * Lv / 2, zc) for su, sg in corners], cp, [R.Wv(su, sg) for su, sg in corners], closed=True))
        else:
            for sg in (-1, 1):
                cor.append(_loft("cornice", [R.W(-Lu / 2 - 0.07, sg * Lv / 2, zc), R.W(Lu / 2 + 0.07, sg * Lv / 2, zc)], cp, R.Wv(0.0, sg)))
        for b in cor:
            densify(b, 0.4)
            assign(b, stone)
        o = join(cor, "eaves_cornice")
        paint(o, grime=mask_mul(noise_mask(0.5, seed=seed + 76, lo=0.2, hi=0.8), 0.5), special=mask_mul(noise_mask(0.5, seed=seed + 77, lo=0.4, hi=0.75), back_w, 0.7))
    else:
        base = _bx("rafter_base", (1.0, 0.075, 0.1), bevel=0.0)
        assign(base, oak)
        fm = []
        for sg in (-1, 1):
            n = max(4, int(round(Lu / 0.42)))
            for i in range(n):
                u = -Lu / 2 + 0.12 + (Lu - 0.24) * i / (n - 1)
                a, b = R.P(sg, u, 0.03, -THK - 0.05), R.P(sg, u, ovh + 0.1, -THK - 0.05)
                fm.append(_frame(a, b, sx=(b - a).length))
        o = instances_to_mesh("rafter_feet", base, fm, keep_base=False)
        paint(o, grime=0.35)

    # chimney (brick, on the back of the roof near the ridge), roof ladder beside it
    rc = rng.fork("chimney")
    sgc = 1 if ridge_x else rc.pick([-1, 1])
    dc = R.dmax - 0.8
    uc = min(Lu / 2 - 1.25, R.ul(dc) - 0.85)
    ec = (rc.pick([-1, 1]) if ridge_x else 1)
    uc = max(uc, 0.0) * ec
    cpos = R.P(sgc, uc, dc)
    sink = 0.75
    ch_h = clamp(sink + (float(R.zs(uc, R.dmax)) - cpos.z) + 0.85, 1.6, 4.4)
    ch = use("chimney", width=0.9, depth=0.58, height=ch_h, material="brick_red", pots=2, pitch=min(60.0, pit), sink=sink,
             roof_stub="none", soot=0.4 + 0.5 * age, moss=0.2 + 0.5 * age, seed=seed + 6)
    ch.place((cpos.x, cpos.y, cpos.z - sink), rot=(0, 0, R.yaw(sgc)), sockets={"fx"})
    ul_ = uc - ec * 1.0 if abs(uc) > 0.9 else uc + ec * 1.0
    if abs(ul_) + 0.3 < R.ul(R.dmax - 0.3):
        nl = R.normal(sgc)
        lad = []
        d1 = R.dmax - 0.12
        for e in (-1, 1):
            lad.append(_beam("ladder_rail", R.P(sgc, ul_ + e * 0.17, 0.5), R.P(sgc, ul_ + e * 0.17, d1), 0.035, 0.03, nl, proud=0.075, bev=0.0))
        nrg = max(3, int(round((d1 - 0.5) / R.cosp / 0.32)))
        for i in range(nrg):
            d = 0.6 + (d1 - 0.7) * i / (nrg - 1)
            lad.append(_beam("ladder_rung", R.P(sgc, ul_ - 0.19, d), R.P(sgc, ul_ + 0.19, d), 0.024, 0.02, nl, proud=0.07, bev=0.0))
        for b in lad:
            assign(b, iron)
        paint(join(lad, "roof_ladder"), special=0.3 + 0.4 * age)

    # ---------------------------------------------------------------- dormers (always at least one, the first one lit) and rooflights
    nd = int(p.dormers)
    d0 = max(ovh + 0.5, 0.95)
    sc0 = clamp((R.dmax - 0.45 - d0) * tanp / 1.96, 0.6, 1.0)
    spots = []                                    # (slope, u, scale)
    if ridge_x:
        done = False
        for n in range(nd, 0, -1):
            for sc in (sc0, max(0.6, 0.88 * sc0), max(0.6, 0.76 * sc0)):
                dwd = 1.15 * sc
                dtop = d0 + 1.96 * sc / tanp
                lim = min(R.ul(dtop) - 0.25, Lu / 2 + 0.1) - dwd / 2 - 0.42
                gap = dwd + 0.42
                if n == 1:
                    xs = [0.0]
                elif len(front_xs) == n and abs(front_xs[0]) <= lim and (front_xs[1] - front_xs[0]) >= gap:
                    xs = list(front_xs)             # dormers stand over the window axes
                else:
                    a = min(lim, Lu * (0.235 if n == 2 else 0.31))
                    xs = [-a, a] if n == 2 else [-a, 0.0, a]
                    if a < gap * (0.5 if n == 2 else 1.0):
                        continue
                spots = [(-1, x, sc) for x in xs]
                done = True
                break
            if done:
                break
        if nd > 1:                                 # the lit one is never the middle of three: it sits over the door side
            spots.sort(key=lambda s: (s[1] * lan_sx < 0, abs(s[1])))
    else:
        dtop = d0 + 1.96 * sc0 / tanp
        lim_c = min(uc - 1.75, R.ul(dtop) - 1.1)
        lim_o = min(Lu / 2 - 1.4, R.ul(dtop) - 1.1)
        uf = max(-Lu / 2 + 1.7, -R.ul(dtop) + 1.1)
        cnt = {1: 0, -1: 0}
        for k in range(nd):
            for sg in ((1, -1) if k % 2 == 0 else (-1, 1)):
                u = uf + 2.5 * cnt[sg] + (0.55 if sg < 0 else 0.0)
                if u <= (lim_c if sg == sgc else lim_o) or not spots:
                    spots.append((sg, min(u, lim_o), sc0))
                    cnt[sg] += 1
                    break
    dorm_light = None
    rd = rng.fork("dormer")
    dtop_max = d0
    for i, (sg, u, sc) in enumerate(spots):
        lt = i == 0 or rl.chance(0.25 + 0.5 * p.lit)
        body, dt, hr_ = _dormer("dormer", tanp, sc, C["lit"] if lt else C["dark"], C, oak if timber else joinery,
                                plaster if up_plaster else cream, gauge, lift)
        loc = R.P(sg, u, d0)
        yaw = R.yaw(sg)
        for ob in (body, dt):
            _xform(ob, (0, 0, yaw), loc)
        paint(dt, special=moss, grime=soot)
        paint(body, grime=mask_mul(noise_mask(0.3, seed=seed + 80 + i, lo=0.2, hi=0.8), 0.25 + 0.3 * age),
              wear=mask_mul(noise_mask(0.1, seed=seed + 85, lo=0.55, hi=0.8), 0.5 * age))
        dtop_max = max(dtop_max, d0 + hr_ / tanp)
        if i == 0:
            tg, nr = wdir(yaw)
            dorm_light = tuple(loc + nr * 0.7 + Vector((0, 0, 0.62 * sc)))
    pr_ = degrees(atan(tanp))
    lights_roof = []

    def rooflight(sg, u, d, w=0.62, h=0.9):
        o = _rooflight("rooflight", w, h, iron, C["dark"], lead)
        _xform(o, (pr_, 0, R.yaw(sg)), R.P(sg, u, d, 0.0))
        paint(o, grime=0.3)

    if ridge_x:
        if R.dmax - dtop_max > 1.9 and spots:      # a long slope above the dormers: a rooflight for the loft
            us = sorted(s[1] for s in spots)
            u = (us[0] + us[1]) / 2 if len(us) > 1 else min(1.5, R.ul(dtop_max + 0.9) - 0.7)
            rooflight(-1, u, min(dtop_max + 0.95, R.dmax - 0.9))
        ub = -ec * min(1.35, R.ul(R.dmax * 0.55) - 0.8)
        if abs(ub - ul_) > 0.75 and R.dmax > 3.0:
            rooflight(1, ub, R.dmax * 0.52)
    else:
        for sg in (-1, 1):
            taken = [s[1] for s in spots if s[0] == sg]
            u = Lu / 2 - 2.1 if sg != sgc else 0.2
            if all(abs(u - t) > 1.5 for t in taken) and abs(u - (uc if sg == sgc else 99)) > 1.4 and abs(u - (ul_ if sg == sgc else 99)) > 0.7 \
                    and abs(u) < R.ul(R.dmax * 0.6) - 0.6:
                rooflight(sg, u, R.dmax * 0.55)

    # ---------------------------------------------------------------- lantern, hanging sign, awning
    lan_x = lan_sx * (W / 2 - pw / 2)
    lo_, fl = _lantern("wall_lantern", iron, C["glass"], C["emis"], C["gilt"])
    lan_loc = Vector((lan_x, 0.0, 2.34))
    _xform(lo_, (0, 0, 0), lan_loc)
    paint(lo_, special=mask_mul(noise_mask(0.15, seed=seed + 90, lo=0.45, hi=0.8), 0.3 + 0.5 * age))
    lan_light = tuple(lan_loc + Vector(fl))
    sign_light = None
    if p.hanging_sign:
        und = j > 0.01
        so, lamps = _hanging_sign("hanging_sign", rng.fork("sign"), C["leather"], joinery, C["gilt"], iron, C["emis"], und)
        if und:
            hloc = Vector((-lan_sx * (W / 2 - pw / 2), 0.0, G - 0.39))
        else:
            hx = 0.0 if len(front_xs) == 2 else (front_xs[1] + front_xs[2]) / 2
            hloc = Vector((hx, 0.0, G + 1.62))
        _xform(so, (0, 0, 0), hloc)
        paint(so, wear=mask_mul(noise_mask(0.08, seed=seed + 91, lo=0.6, hi=0.85), 0.3 * age), special=mask_mul(noise_mask(0.2, seed=seed + 93), 0.4))
        if lamps:
            sign_light = tuple(hloc + Vector((0.0, lamps[0][1] - 0.1, lamps[0][2] - 0.1)))
        socket("hanging_sign", "attach", tuple(hloc), role="sign")
    aw_obj = None
    if p.awning:
        ap, _ = _sock(sw, "awning", (win_xs[0], 0, 0))
        aw_obj = _awning("awning", rng.fork("awning"), clamp(win_w - 0.25, 1.6, 5.0), 1.05, 13.0, 0.21, C["paint"], cream, iron)
        _xform(aw_obj, (0, 0, 0), ap)

    # ---------------------------------------------------------------- luggage on the pavement
    rlg = rng.fork("luggage")
    wx = win_xs[0]
    if win_w >= 2.6:
        sgn = 1 if wx >= door_x else -1
        cols = [wx + sgn * (win_w / 2 - 0.62), wx - sgn * (win_w / 2 - 0.58)]
    else:
        cols = [wx, win_xs[-1] if len(win_xs) > 1 else wx]
    n_trunks = int(p.trunks)
    if abs(cols[0] - cols[1]) < 0.95:
        n_trunks = min(n_trunks, 1) if n_trunks < 3 else n_trunks
    _luggage(p, rlg, cols, sw.min.y if sw.min.y > -0.6 else -0.46, C, n_trunks)

    # ---------------------------------------------------------------- lights (<= 6, in order of importance) and sockets
    bulbs = []
    L = []
    for i, x in enumerate(win_xs):
        s0, _ = _sock(sw, "sign0", (x, 0, 0))
        s1, _ = _sock(sw, "sign1", (x, 0, 0))
        s0 = s0 or (x - 0.6, -0.81, 2.54)
        s1 = s1 or (x + 0.6, -0.81, 2.54)
        for s_ in (s0, s1):
            bulbs.append(sphere("display_bulb", 0.03, loc=(s_[0], s_[1], s_[2] - 0.035), segments=8, rings=5))
        if p.awning and i == 0:                    # under the canopy: a bare bulb on the awning frame
            lp_ = ((s0[0] + s1[0]) / 2, -0.86, 1.9)
            bulbs.append(sphere("awning_bulb", 0.035, loc=lp_, segments=8, rings=5))
            bulbs.append(_bx("awning_flex", (0.008, 0.008, 0.2), loc=(lp_[0], lp_[1], 2.02)))
        else:
            lp_ = ((s0[0] + s1[0]) / 2, min(s0[1], s1[1]) - 0.5, s0[2] - 0.32)
        L.append((f"display_lamps{i}", lp_, "#ffc98a",
                  float(p.shop_cd) * (1.0 if len(win_xs) == 1 else 0.75) * (0.7 if p.awning else 1.0), 8.0, 0.04, {"shadow_near": 0.15}))
    for b in bulbs:
        assign(b, C["emis"])
    join(bulbs, "display_bulbs")
    sp, pr = _sock(door, "door_glass", (door_x, 0, 0))
    L.append(("door_glass", sp, "#ffc37a", 9.0, 5.5, 0.04, {}))
    L.append(("wall_lantern", lan_light, "#ffb25c", float(p.lantern_cd), 6.0, 0.14, {"shadow_near": 0.3}))
    if dorm_light:
        L.append(("dormer", dorm_light, "#ffc37a", 4.5, 5.5, 0.05, {"shadow_near": 0.6}))
    s0, _ = _sock(fs, "lamp0", (0, -yb, SF_H))
    s1, _ = _sock(fs, "lamp1", (0, -yb, SF_H))
    if s0 and s1:
        fx_ = (s0[0] + s1[0]) / 2 + (door_x * 0.35 if len(win_xs) == 1 else 0.0)
        pr_, pg_, pb_ = _lin(p.paint)[:3]         # green and blue boards answer a warm lamp a third as well as a red one
        fcd = 11.5 * clamp(sqrt(0.2 / max(pr_ + 0.62 * pg_ + 0.3 * pb_, 0.02)), 1.0, 1.6)
        L.append(("fascia_lamps", (fx_, min(s0[1], s1[1]) - 0.55, s0[2] - 0.3), "#ffcf96", fcd, 5.5, 0.03, {"shadow_near": 0.1}))
    if sign_light:
        L.append(("sign_lamps", sign_light, "#ffcf96", 3.0, 3.2, 0.03, {"shadow_near": 0.05}))
    for (nm, loc, col, cd, rg, fk, kw) in L[:6]:
        light(nm, tuple(loc), color=col, intensity=cd, range=rg, flicker=fk, **kw)

    socket("bunting_l", "attach", (-W / 2 + 0.25, -j, zE - 0.5), role="banner")
    socket("bunting_r", "attach", (W / 2 - 0.25, -j, zE - 0.5), role="banner")
    if not p.awning:
        ap, _ = _sock(sw, "awning", (win_xs[0], 0, 0))
        socket("awning", "attach", ap, role="awning", width=win_w - 0.25)

    # ---------------------------------------------------------------- exposed brick where the render has blown off
    rpz = rng.fork("patch")
    bare = None                                   # brickwork showing on the right wall: (u, z, rx, rz)
    if g_plaster:
        bare = (D * 0.3 + rpz.jitter(0, 0.2), PL_H + 0.2 + rpz.jitter(0, 0.05), rpz.u(0.6, 0.85), rpz.u(0.62, 0.8))
    elif up_plaster and not timber:
        bare = (free_u + rpz.jitter(0, 0.1), G + S * 0.42, rpz.u(0.34, 0.46), rpz.u(0.3, 0.42))
    if bare:
        bu, bz, brx, brz = bare
        n = 26
        rim = []
        zlo = PL_H - 0.04 if g_plaster else -1e9   # on the ground floor the render has gone from the plinth upwards
        ff = 1.0
        for k in range(n):
            th = 2 * pi * k / n
            ff = clamp(ff + rpz.u(-0.16, 0.16), 0.7, 1.14)
            rim.append((W / 2 + 0.013, bu + brx * ff * cos(th) * (1.0 + 0.25 * (sin(th) < 0)),
                        max(zlo, bz + brz * ff * sin(th) - 0.12 * brz * abs(cos(th)))))
        o = mesh_object("bare_brick", [(W / 2 + 0.016, bu, bz)] + rim, [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)])
        assign(o, C["brick"])
        paint(o, grime=mask_mul(noise_mask(0.2, seed=seed + 95), 0.25))

    _remap(C, timber)

    # ---------------------------------------------------------------- weathering of the wall skins (world-space, ~15 cm grain)
    n1 = noise_mask(0.55, seed=seed + 201, lo=0.2, hi=0.8)
    n2 = noise_mask(2.2, seed=seed + 202, lo=0.3, hi=0.7)
    n3 = noise_mask(0.4, seed=seed + 203, lo=0.25, hi=0.8)
    n4 = noise_mask(0.7, seed=seed + 204, lo=0.25, hi=0.75)
    n5 = noise_mask(0.3, seed=seed + 205, lo=0.55, hi=0.8)
    n6 = noise_mask(0.35, seed=seed + 206, lo=0.2, hi=0.8)
    n7 = noise_mask(1.1, seed=seed + 207, lo=0.5, hi=0.78)

    def rain(co):
        streak = np.zeros(len(co))
        wash = np.zeros(len(co))
        for (c, t, nr, w, ln) in sills:
            rel = co - np.array(c)
            on = (np.abs(rel @ np.array(nr)) < 0.2) & (rel[:, 2] < 0.0)
            dz = -rel[:, 2] - 0.1
            e = np.abs(rel @ np.array(t)) - w / 2
            streak = np.maximum(streak, on * np.exp(-(e / 0.1) ** 2) * np.clip(1.0 - dz / ln, 0.0, 1.0))
            wash = np.maximum(wash, on * (e < 0.0) * np.clip(1.0 - dz / 0.5, 0.0, 1.0))
        hop = np.zeros(len(co))
        for (c, t, nr) in hoppers:
            rel = co - np.array(c)
            on = (np.abs(rel @ np.array(nr)) < 0.2) & (rel[:, 2] < 0.1)
            hop = np.maximum(hop, on * np.exp(-((rel @ np.array(t)) / 0.17) ** 2) * (0.45 + 0.55 * np.clip(1.0 + rel[:, 2] / 3.0, 0.0, 1.0)))
        return streak, wash, hop

    def halo(co, n):                              # the grey render coat showing round the bare brick
        if not bare:
            return np.zeros(len(co))
        u, z, rx_, rz_ = bare
        m = np.sqrt(((co[:, 1] - u) / rx_) ** 2 + ((co[:, 2] - z) / rz_) ** 2)
        return (n[:, 0] > 0.5) * np.clip(1.0 - (m - 0.85) / 0.55 + 0.3 * (n6(co, n) - 0.5), 0.0, 1.0)

    def w_grime(co, n):
        z = co[:, 2]
        streak, wash, hop = rain(co)
        soot_ = np.clip(1.0 - (R.top(co[:, 0], co[:, 1]) - THK - z) / 0.8, 0.0, 1.0)
        damp = n7(co, n) * np.clip(1.0 - z / 3.2, 0.0, 1.0)
        g = 0.75 * np.clip(1.0 - z / (0.3 + 0.55 * n1(co, n)), 0.0, 1.0) + 0.16 * n2(co, n) + (0.18 + 0.3 * age) * soot_ * n3(co, n) \
            + (0.42 + 0.3 * age) * streak + 0.14 * wash + 0.6 * hop + (0.12 + 0.2 * age) * damp
        return np.clip(g, 0.0, 0.92)

    def w_special(co, n):
        z = co[:, 2]
        streak, wash, hop = rain(co)
        wt = 0.22 + 0.4 * np.abs(n[:, 0]) + 0.78 * np.clip(n[:, 1], 0.0, 1.0)
        a = np.clip(1.15 - z / (0.6 + 1.3 * n4(co, n)), 0.0, 1.0) * wt * (0.55 + 0.45 * age) + 0.75 * hop * np.clip(1.0 - z / (hz + 0.2), 0.0, 1.0) ** 0.5 \
            + 0.25 * streak * age
        return np.clip(a, 0.0, 1.0)

    def w_wear(co, n):
        z = co[:, 2]
        chips = n5(co, n) * np.clip(1.0 - z / 2.4, 0.0, 1.0) * (0.25 + 0.6 * age)
        return np.clip(chips + 0.75 * halo(co, n), 0.0, 1.0)

    # ---------------------------------------------------------------- triangle budget, then the wall paint
    # The widest plans with two shop windows, an awning and a full pavement of trunks would pass the class budget.
    # Nothing is left out: the roof covering is re-laid with wider tile columns (same courses, same silhouette)
    # and, if that is not enough, the wall skins are re-made at a coarser grain.
    def scene_tris():
        return sum(_tris(ob) for ob in bpy.data.objects if ob.type == "MESH" and not ob.name.startswith("cmp_"))

    def drop(ob):
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)

    BUDGET = 156000
    total = scene_tris()
    if total > BUDGET:
        cap = max(2000, _tris(tiles) - (total - BUDGET))
        drop(tiles)
        lay_tiles(cap)
        total = scene_tris()
    if total > BUDGET:
        wt = sum(_tris(o) for o, _ in walls)
        f = sqrt(wt / max(3000.0, wt - (total - BUDGET) * 1.15))
        for o, _ in walls:
            drop(o)
        walls = skin(wsc * f)
    for (o, m_) in walls:
        paint(o, grime=w_grime, special=w_special, wear=w_wear)
