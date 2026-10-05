"""Classical stone column / square pier / pilaster. Built the way a mason builds it: a rough-dressed
square plinth, an Attic base (torus - scotia - torus - apophyge) turned from one stone, a shaft of
separate drums (each its own block: own bedding, own tone) on thin recessed lime-mortar beds, diminution
and entasis, optional fluting with rounded flute ends, and a capital in one of three orders: Tuscan
(astragal, necking, annulets, echinus, square abacus), cushion (Romanesque cube capital with lunette
faces under a hollow-chamfered impost slab on a mortar bed) or foliate (bell with two rows of solid
acanthus leaves - midrib, lobes, rolled-over tips -, thick corner volutes on caulicoli fused to the bell,
concave abacus with rosettes). mode = column (round), pier (square) or pilaster (the pier cut in half by
the wall plane y = 0). Damage is real geometry: spalls knocked out of the flute arrises and the drum-joint
edges, conchoidal chips in the slabs. Weathering: graded splash zone, thin rain runs under the astragal,
moss low on the damp side; now and then a rusty tethering ring with its rust run.
Contract: centred on the origin, base at z = 0, top of the capital at z = height.
"""
from math import sin, cos, pi, radians, sqrt, acos, atan2, hypot

import bmesh
import numpy as np
from mathutils import Vector

from tk import *

META = {"name": "Stone column / pilaster", "class": "component", "mount": "ground", "focus": 0.86}
PARAMS = {
    "height": {"default": 3.2, "min": 1.2, "max": 7.0, "desc": "total height, paving to top of the capital (m)"},
    "diameter": {"default": 0.4, "min": 0.18, "max": 1.0, "desc": "lower shaft diameter / pier width (m)"},
    "order": {"default": "foliate", "choices": ["tuscan", "foliate", "cushion"], "desc": "capital order"},
    "fluted": {"default": True, "desc": "fluted shaft (False = plain)"},
    "flutes": {"default": 20, "min": 12, "max": 28, "desc": "flutes around a round shaft (a pier face gets a quarter of them)"},
    "mode": {"default": "column", "choices": ["column", "pier", "pilaster"],
             "desc": "round column, square pier, or pilaster (half-depth, back on the wall plane y = 0, projecting to -Y)"},
    "mount": {"default": "ground", "choices": ["ground", "wall_ground"], "desc": "use wall_ground with mode = pilaster"},
    "taper": {"default": 0.85, "min": 0.7, "max": 1.0, "desc": "top / bottom shaft diameter (diminution)"},
    "entasis": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "swelling of the shaft outline (0 = straight taper)"},
    "drums": {"default": 4, "min": 1, "max": 8, "desc": "number of shaft drums (mortar joints between them)"},
    "stone": {"default": "#b9ad98", "desc": "stone colour"},
    "damage": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "spalled arrises / joint edges, chipped slabs"},
    "grime": {"default": 0.55, "min": 0.0, "max": 1.0, "desc": "splash-back dirt and rain runs"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "moss / lichen on the damp side and the base"},
    "ring": {"default": "auto", "choices": ["auto", "yes", "no"],
             "desc": "rusty iron tethering ring leaded into the shaft (auto = about one column in four, by seed)"},
}
VARIANTS = {
    "foliate_tall": {"order": "foliate", "height": 4.4, "diameter": 0.44, "flutes": 20, "stone": "#cfc4ae", "drums": 4,
                     "damage": 0.25, "moss": 0.15, "seed": 4, "ring": "no"},
    "cushion_stout": {"order": "cushion", "height": 2.3, "diameter": 0.52, "fluted": False, "taper": 0.94, "entasis": 0.2,
                      "drums": 3, "stone": "#a8906f", "damage": 0.8, "moss": 0.7, "grime": 0.8, "seed": 11, "ring": "no"},
    "pier_tuscan": {"mode": "pier", "order": "tuscan", "height": 3.0, "diameter": 0.5, "stone": "#9c9a94", "drums": 5,
                    "damage": 0.55, "seed": 23, "ring": "yes"},
    "pilaster_foliate": {"mode": "pilaster", "mount": "wall_ground", "order": "foliate", "height": 3.4, "diameter": 0.46,
                         "flutes": 28, "stone": "#c2a988", "ring": "no", "moss": 0.1, "seed": 31},
    "plain_tuscan_small": {"order": "tuscan", "fluted": False, "height": 2.4, "diameter": 0.3, "drums": 2,
                           "stone": "#d6cdbb", "damage": 0.15, "grime": 0.3, "moss": 0.0, "seed": 40, "ring": "no"},
}

FLUTE_PTS = ((-1.0, 0.0), (-0.45, 0.88), (0.45, 0.88), (1.0, 0.0))
FLUTE_HW = 0.41          # flute half-width as a fraction of the pitch (the rest is the fillet)


# ------------------------------------------------------------------ small helpers
def _tone(hexcol, k=1.0, sat=1.0):
    """Value (k) and saturation shift of a hex colour: every block of stone has its own tone."""
    h = hexcol.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    g = sum(c) / 3.0
    c = [min(1.0, max(0.0, (g + (x - g) * sat) * k)) for x in c]
    return "#%02x%02x%02x" % tuple(int(round(x * 255)) for x in c)


def _mix(h1, h2, t):
    a, b = h1.lstrip("#"), h2.lstrip("#")
    return "#%02x%02x%02x" % tuple(int(round(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t)) for i in (0, 2, 4))


def _circ(r, n, ox=0.0, oy=0.0):
    return [(ox + r * cos(2 * pi * i / n), oy + r * sin(2 * pi * i / n)) for i in range(n)]


def _sq(r, c, ox=0.0, oy=0.0):
    """Square of half-width r with chamfered corners (c), CCW, 8 points."""
    return [(ox + x, oy + y) for x, y in ((r, -r + c), (r, r - c), (r - c, r), (-r + c, r), (-r, r - c), (-r, -r + c),
                                          (-r + c, -r), (r - c, -r))]


def _loft(name, rings, cap_top=False, cap_bot=False):
    """rings: [(pts2d CCW, z), ...] bottom to top, same point count -> quad skin (outward normals)."""
    n = len(rings[0][0])
    verts, faces = [], []
    for pts, z in rings:
        verts += [(x, y, z) for x, y in pts]
    for j in range(len(rings) - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    if cap_top:
        faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
    if cap_bot:
        faces.append(tuple(reversed(range(n))))
    return mesh_object(name, verts, faces)


def _arc(P, cx, cz, rad, a0, a1, n):
    for i in range(n + 1):
        a = radians(a0 + (a1 - a0) * i / n)
        P.append((cx + rad * cos(a), cz + rad * sin(a)))


def _turned(name, prof, sqm, c, segs=24):
    """A moulding profile [(r, z)] turned round the axis (column) or run round a square (pier)."""
    if not sqm:
        return lathe(name, prof, segments=segs, smooth_angle=42)
    o = _loft(name, [(_sq(r, min(c, r * 0.3)), z) for r, z in prof])
    smooth(o, 38)
    uv_box(o)
    return o


def _bed(o, r, rot):
    """Give a block its own bedding: resolve its UVs, then turn / shift them (the figure of the stone lies level)."""
    realize(o)
    uv_transform(o, offset=(r.u(0.0, 3.0), r.u(0.0, 3.0)), rot_deg=rot + r.u(-5.0, 5.0))
    uv_keep(o)
    return o


def _merge_strip(A, B, faces):
    """Skin between two neighbouring vertex columns A, B = [(z, ring_id | None, vertex index)] (ascending z, A then
    B counter-clockwise). Columns may carry different numbers of vertices (local spalls, streak ends): shared rings
    pair into quads, everything else fans into triangles - no T-junctions."""
    a = b = 0
    la, lb = len(A) - 1, len(B) - 1
    while a < la or b < lb:
        na = A[a + 1] if a < la else None
        nb = B[b + 1] if b < lb else None
        if na is not None and nb is not None and ((na[1] is not None and na[1] == nb[1]) or abs(na[0] - nb[0]) < 1e-6):
            faces.append((A[a][2], B[b][2], nb[2], na[2]))
            a += 1
            b += 1
        elif nb is None or (na is not None and na[0] <= nb[0]):
            faces.append((A[a][2], B[b][2], na[2]))
            a += 1
        else:
            faces.append((A[a][2], B[b][2], nb[2]))
            b += 1


def _grid_box(name, size, bottom=True):
    """Box as a welded grid, denser towards the arrises so spalls have vertices to cut into."""
    F = (-1, -0.8, -0.45, 0, 0.45, 0.8, 1)
    xs = [f * size[0] / 2 for f in F]
    ys = [f * size[1] / 2 for f in F]
    zs = [f * size[2] / 2 for f in (-1, 0, 1)]
    vid, verts, faces = {}, [], []

    def V(i, j, k):
        if (i, j, k) not in vid:
            vid[(i, j, k)] = len(verts)
            verts.append((xs[i], ys[j], zs[k]))
        return vid[(i, j, k)]

    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    for i in range(nx):
        for j in range(ny):
            faces.append((V(i, j, nz), V(i + 1, j, nz), V(i + 1, j + 1, nz), V(i, j + 1, nz)))
            if bottom:
                faces.append((V(i, j, 0), V(i, j + 1, 0), V(i + 1, j + 1, 0), V(i + 1, j, 0)))
    for i in range(nx):
        for k in range(nz):
            faces.append((V(i, 0, k), V(i + 1, 0, k), V(i + 1, 0, k + 1), V(i, 0, k + 1)))
            faces.append((V(i, ny, k), V(i, ny, k + 1), V(i + 1, ny, k + 1), V(i + 1, ny, k)))
    for j in range(ny):
        for k in range(nz):
            faces.append((V(0, j, k), V(0, j, k + 1), V(0, j + 1, k + 1), V(0, j + 1, k)))
            faces.append((V(nx, j, k), V(nx, j + 1, k), V(nx, j + 1, k + 1), V(nx, j, k + 1)))
    return mesh_object(name, verts, faces)


def _block(name, size, loc, r, damage, chamfer, bottom=True):
    """Rough-dressed block (plinth) with knocked corners and scalloped edge chips."""
    o = _grid_box(name, size, bottom)
    bevel(o, width=min(chamfer, min(size) * 0.2), segments=1)
    realize(o)
    displace_noise(o, strength=0.0022, scale=max(size) * 0.5, seed=int(r.u(0, 999)))
    h = (size[0] / 2, size[1] / 2, size[2] / 2)
    ref = min(h[0], h[1])
    dm = min(damage, 0.6)
    corners, edges = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                if r.chance(0.2 + 0.75 * damage):
                    d = Vector((sx * r.u(0.6, 1.2), sy * r.u(0.6, 1.2), sz * r.u(0.5, 1.3))).normalized()
                    s = min(r.u(0.25, 1.0) * (0.15 + dm) * 0.2 * ref, 1.3 * h[2])
                    corners.append((Vector((sx * h[0], sy * h[1], sz * h[2])), d, s))
    for _ in range(int(damage * 7 + r.u(0, 1.2))):
        ax = r.pick([0, 0, 1, 1, 2])
        oth = [i for i in range(3) if i != ax]
        sg = [r.pick([-1, 1]), r.pick([-1, 1])]
        w = [r.u(0.5, 1.2), r.u(0.5, 1.2)]
        nrm = hypot(w[0], w[1])
        edges.append((ax, oth, sg, [w[0] / nrm, w[1] / nrm], r.u(-0.8, 0.8) * h[ax], r.u(0.1, 0.3) * ref + 0.01,
                      r.u(0.25, 1.0) * (0.15 + dm) * 0.07 * ref))

    def knock(co):
        for c0, d, s in corners:
            k = (co - c0).dot(d) + s
            if k > 0:
                co = co - d * k
        for ax, oth, sg, w, q0, L, s in edges:
            q = (co[ax] - q0) / L
            if abs(q) < 1.0:
                k = ((co[oth[0]] - sg[0] * h[oth[0]]) * sg[0] * w[0] + (co[oth[1]] - sg[1] * h[oth[1]]) * sg[1] * w[1]
                     + s * (1.0 - q * q))
                if k > 0:
                    co = co.copy()
                    co[oth[0]] -= sg[0] * w[0] * k
                    co[oth[1]] -= sg[1] * w[1] * k
        return co

    transform_verts(o, knock)
    smooth(o, 35)
    uv_box(o)
    place(o, loc)
    return o


def _slab(name, rings, r, damage, zlow=None):
    """Crisp dressed slab (abacus / impost): a closed loft of square rings that stays planar; damage is only
    localised conchoidal chips scooped out of its arrises and corners (boolean cuts)."""
    o = _loft(name, rings, cap_top=True, cap_bot=True)
    smooth(o, 35)
    uv_box(o)
    z0, z1 = rings[0][1], rings[-1][1]
    zlow = z0 if zlow is None else zlow
    w = max(max(abs(x), abs(y)) for x, y in rings[-2][0])
    ref = min(w, (z1 - zlow) * 3.0)
    for k in range(int(damage * 6.5 + r.u(0.2, 1.4))):
        ang = r.pick([0, 1, 2, 3]) * pi / 2
        nx, ny = cos(ang), sin(ang)
        tx, ty = -ny, nx
        kind = r.weighted([("top", 3), ("low", 2), ("corner", 2)])
        rad = r.u(0.07, 0.15) * ref * (0.7 + 0.5 * damage)
        bite = rad * r.u(0.3, 0.55)
        if kind == "corner":
            up = r.chance(0.55)
            p0 = Vector((w * (nx + tx), w * (ny + ty), z1 if up else zlow))
            d = Vector((nx + tx, ny + ty, 1.1 if up else -1.1)).normalized()
            rad *= 1.5
            bite *= 1.5
            sc = (1.0, 1.0, r.u(0.8, 1.2))
        else:
            a = r.u(-0.85, 0.85) * w
            up = kind == "top"
            p0 = Vector((nx * w + tx * a, ny * w + ty * a, z1 if up else zlow))
            d = Vector((nx, ny, 1.0 if up else -1.0)).normalized()
            el = r.u(1.1, 2.0)
            sc = (el if abs(tx) > 0.5 else 1.0, el if abs(ty) > 0.5 else 1.0, 1.0)
        c = p0 + d * (rad - bite)
        cut = sphere("chip_cut", rad, loc=tuple(c), segments=7, rings=4, scale=sc)
        displace_noise(cut, strength=rad * 0.14, scale=rad * 0.9, seed=int(r.u(0, 9999)))
        boolean(o, cut)
    return o


def _knock(o, planes):
    """Planar spalls: everything beyond a plane (point c0, outward unit normal d, depth s) is cut back to it."""
    def fn(co):
        for c0, d, s in planes:
            k = (co - c0).dot(d) + s
            if k > 0:
                co = co - d * k
        return co
    if planes:
        transform_verts(o, fn)


def _acanthus(name, W, Ht, thick, curv, lobes, s_start=0.0, lean=4.0, curl=165.0, rin=None, na=6):
    """One carved acanthus leaf as a SOLID: standing on z = 0, outward = -Y, x across. A raised midrib, `lobes`
    lobes per side on the upright part - every lobe a hollowed spoon with its tip flicked forward and up, the sinuses
    between them cut back deep towards the rib - and a tip that rolls forward and over in a smooth arc of `na`
    segments. `thick` is the body thickness; the upright part grows out of the bell (its rim runs back into the
    stone), the roll is closed all round. Ht = overall height to the top of the roll, curv = 1 / bell radius."""
    ln, phi = radians(lean), radians(curl)
    rin = rin or 0.13 * Ht
    Ls = max(0.25 * Ht, (Ht - thick - rin * (1.0 - sin(ln))) / cos(ln))
    arc = rin * phi
    Lt = Ls + arc
    s0 = s_start * Ls
    pitch = (Ls - s0) / lobes

    def env(f):
        return 1.0 - 0.2 * f * f

    rows = []                                                     # (s, half width factor, edge lift, mid lift, edge shift)
    for m in range(lobes):                                        # sinus, swelling flank, hooked tip
        b0 = s0 + pitch * m
        rows.append((b0, env(b0 / Ls) * (0.64 if m else 0.86), -0.15, -0.2, -0.04 * pitch))
        rows.append((b0 + 0.4 * pitch, env((b0 + 0.4 * pitch) / Ls) * 0.95, 0.35, 0.02, 0.0))
        rows.append((b0 + 0.72 * pitch, env((b0 + 0.72 * pitch) / Ls), 0.6, 0.1, 0.2 * pitch))
    rows.append((Ls, env(1.0) * 0.62, -0.1, -0.15, -0.04 * pitch))
    cw = [(0.8, 0.4), (0.74, 0.3), (0.64, 0.22), (0.52, 0.15), (0.4, 0.08), (0.24, 0.0)]
    for k in range(1, na + 1):
        wf, el = cw[min(len(cw) - 1, int((k - 1) * len(cw) / na))]
        rows.append((Ls + arc * k / na, wf * env(1.0), el, 0.0, 0.0))
    nr = len(rows)
    jb = 3 * lobes - 1                                            # first row that has a back face
    Cx, Cz = Ls * sin(ln) + rin * cos(ln), Ls * cos(ln) - rin * sin(ln)
    verts, faces = [], []
    F, B = [], []
    for j, (s, wf, el, md, sh) in enumerate(rows):
        v = s / Lt
        if s <= Ls:
            th, po, pz = ln, s * sin(ln), s * cos(ln)
            wrap = 1.0
        else:
            th = ln + (s - Ls) / rin
            po, pz = Cx - rin * cos(th), Cz + rin * sin(th)
            wrap = 1.0 - 0.8 * (s - Ls) / arc
        no, nz = cos(th), -sin(th)                                # front normal in the (outward, z) plane
        to, tz = sin(th), cos(th)                                 # spine tangent
        hw = 0.5 * W * wf
        tk = thick * (1.0 - 0.25 * v)
        rib = 0.5 * thick * (1.0 - 0.5 * v)

        def pt(x, off, shift=0.0):
            o = po + no * off + to * shift - curv * x * x * 0.5 * wrap
            return (x, -o, pz + nz * off + tz * shift)

        fr = []
        for x, off, shf in ((-hw, el * thick, sh), (-0.5 * hw, md * thick, 0.4 * sh), (0.0, rib, 0.0),
                            (0.5 * hw, md * thick, 0.4 * sh), (hw, el * thick, sh)):
            fr.append(len(verts))
            verts.append(pt(x, off, shf))
        F.append(fr)
        eb = -1.35 * thick if j < jb else el * thick - 0.62 * tk  # rim: into the bell below, free on the roll
        bk = [len(verts), None, len(verts) + 1]
        verts.append(pt(-hw, eb, sh))
        verts.append(pt(hw, eb, sh))
        if j >= jb:
            bk[1] = len(verts)
            verts.append(pt(0.0, -tk))
        B.append(bk)
    for j in range(nr - 1):
        for i in range(4):
            faces.append((F[j][i], F[j][i + 1], F[j + 1][i + 1], F[j + 1][i]))
        faces.append((B[j][0], F[j][0], F[j + 1][0], B[j + 1][0]))            # left rim
        faces.append((F[j][4], B[j][2], B[j + 1][2], F[j + 1][4]))            # right rim
        if j >= jb:
            for i in (0, 1):
                faces.append((B[j][i], B[j + 1][i], B[j + 1][i + 1], B[j][i + 1]))
    f, b = F[-1], B[-1]                                           # tip end
    faces += [(f[0], f[1], b[0]), (f[1], f[2], b[1], b[0]), (f[2], f[3], b[2], b[1]), (f[3], f[4], b[2])]
    lf = mesh_object(name, verts, faces)
    smooth(lf, 62)
    uv_box(lf)
    return lf


def _clip_back(o):
    """Pilaster: cut away everything behind the wall plane y = 0 and close the cut."""
    realize(o)
    apply_transform(o)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, 0), plane_no=(0, 1, 0),
                                 clear_outer=True, dist=1e-5)
    cut = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    if cut:
        try:
            bmesh.ops.holes_fill(bm, edges=cut, sides=0)
        except Exception as ex:                                  # open back is hidden by the wall anyway
            print("DBG clip fill failed", o.name, ex)
    nv = len(bm.verts)
    bm.to_mesh(o.data)
    bm.free()
    o.data.update()
    return nv > 0


# ------------------------------------------------------------------ build
def build(p, rng):
    H, D = float(p.height), float(p.diameter)
    mode, order = p.mode, p.order
    sqm = mode != "column"
    pil = mode == "pilaster"
    dmg, G, M = float(p.damage), float(p.grime), float(p.moss)
    BED = 25.0                                                    # uv turn that lays the tooling / bedding of stone_smooth level

    # every block is its own stone: three tones of the same quarry
    tones = [mat("stone_smooth", _tone(p.stone, k, s)) for k, s in ((1.0, 1.0), (0.962, 1.07), (1.032, 0.92))]
    stone_pl = mat("stone_rough", _tone(p.stone, 0.95, 0.95))
    mortar = mat("mortar", _tone(p.stone, 0.84, 0.6))
    mossm = mat("moss")
    rt_ = rng.fork("tone")
    ti = [int(rt_.u(0, 2.999))]
    for _ in range(12):
        ti.append(rt_.pick([i for i in range(3) if i != ti[-1]]))
    t_base, t_cap, t_drums = tones[ti[0]], tones[ti[1]], [tones[i] for i in ti[2:]]
    if t_drums[0] is t_base:
        t_drums[0] = tones[ti[1]]
    rb_ = rng.fork("bedding")

    rb = D / 2
    rt = rb * (p.taper if not sqm else 1.0 - (1.0 - p.taper) * 0.3)
    ub = min(D, 0.30 * H)                                         # base module
    kc = {"tuscan": 0.62, "cushion": 0.95, "foliate": 1.18}[order]
    uc = min(D, 0.27 * H / kc)                                    # capital module
    hc = kc * uc
    cch = 0.035 * D                                               # pier corner chamfer
    objs = []
    cap_objs, base_objs = [], []

    # ---------------- plinth + Attic base
    hp = 0.16 * ub
    wp = rb + 0.18 * ub
    plinth = _block("plinth", (2 * wp, 2 * wp, hp), (0, 0, hp / 2), rng.fork("plinth"), dmg, 0.008 + 0.01 * ub,
                    bottom=False)
    assign(plinth, stone_pl)
    objs.append(plinth)

    u = ub
    P = []
    z = hp - 0.001
    t1 = 0.058 * u
    _arc(P, rb + 0.10 * u, z + t1, t1, -90, 90, 3)                # lower torus
    z += 2 * t1
    P += [(rb + 0.085 * u, z), (rb + 0.085 * u, z + 0.012 * u)]   # fillet
    z += 0.012 * u
    for i in range(1, 3):                                         # scotia
        t = i / 3
        P.append((rb + (0.085 - 0.02 * t) * u - 0.03 * u * sin(pi * t), z + 0.06 * u * t))
    z += 0.06 * u
    P += [(rb + 0.065 * u, z), (rb + 0.065 * u, z + 0.012 * u)]   # fillet
    z += 0.012 * u
    t2 = 0.04 * u
    _arc(P, rb + 0.055 * u, z + t2, t2, -90, 90, 3)               # upper torus
    z += 2 * t2
    P += [(rb + 0.04 * u, z), (rb + 0.04 * u, z + 0.014 * u)]     # fillet
    z += 0.014 * u
    for i in range(1, 3):                                         # apophyge sweeping into the shaft
        th = i / 2 * pi / 2
        P.append((rb + 0.04 * u * (1 - sin(th)), z + 0.045 * u * (1 - cos(th))))
    z += 0.045 * u
    P.append((rb - 0.006, z))
    zs0 = z
    base = _turned("base", P, sqm, cch)
    base_objs.append(base)

    # ---------------- shaft: drums, entasis, flutes
    zs1 = H - hc
    Ls = zs1 - zs0
    e_pow = 1.0 + 1.3 * p.entasis

    def r_at(zz):
        t = clamp((zz - zs0) / Ls)
        return rb - (rb - rt) * t ** e_pow

    fluted = bool(p.fluted)
    nf = int(p.flutes)
    nfl = max(3, int(round(nf / 4.0)))                            # flutes per pier face
    margin = 0.15                                                 # plain margin at pier corners (fraction of r)
    if sqm:
        fl_w = 2 * FLUTE_HW * (2 - 2 * margin) * rb / nfl
    else:
        fl_w = 2 * FLUTE_HW * 2 * pi * rb / nf
    le = fl_w * 0.5                                               # quarter-round flute ends
    zf0, zf1 = zs0 + 0.05 * ub, zs1 - 0.03 * uc
    if zf1 - zf0 < 4 * le:
        fluted = False
    nplain = 56                                                   # columns of a plain round shaft
    mplain = 6                                                    # interior columns of a plain pier face

    def s_at(zz):
        if not fluted or zz <= zf0 or zz >= zf1:
            return 0.0
        t = min((zz - zf0) / le, (zf1 - zz) / le, 1.0)
        return sqrt(max(0.0, 1.0 - (1.0 - t) ** 2))

    nd = max(1, min(int(p.drums), int(Ls / 0.3)))
    rj = rng.fork("drums")
    jl, jh = (zf0 + le + 0.03, zf1 - le - 0.03) if fluted else (zs0 + 0.05, zs1 - 0.05)
    joints = []
    for k in range(1, nd):
        zj = zs0 + Ls * (k / nd + rj.u(-0.05, 0.05) / nd * 2)
        if jl < zj < jh:
            joints.append(zj)
    nj = len(joints)
    offs = [(0.0, 0.0)] + [(rj.u(-1, 1) * 0.005 * D, rj.u(-1, 1) * 0.005 * D) for _ in joints]
    gjs = [(0.0019 + 0.0016 * D) * rj.u(0.7, 1.35) for _ in joints]   # mortar bed half height
    gd = 0.0013 + 0.0012 * D                                      # how far the mortar sits back
    jwob = [(rj.u(0.0004, 0.0012), rj.u(0, 6.28), int(rj.u(1, 3.99)), rj.u(0, 6.28), int(rj.u(2, 5.99))) for _ in joints]

    sect_cache = {}

    def section(zz, jm=-1):
        key = (round(zz, 6), jm)
        if key in sect_cache:
            return sect_cache[key]
        k = sum(1 for zj in joints if zz > zj + 1e-6)
        ox, oy = offs[k]
        dr = 0.0
        if jm >= 0:
            ox, oy = (offs[jm][0] + offs[jm + 1][0]) / 2, (offs[jm][1] + offs[jm + 1][1]) / 2
            dr = -gd
        r = r_at(zz) + dr
        s = s_at(zz)
        sw = max(0.2, s)
        pts = []
        if not sqm:
            if not fluted:
                pts = _circ(r, nplain, ox, oy)
            else:
                per = 2 * pi / nf
                dep = 0.3 * fl_w * (r / rb) * s
                for i in range(nf):
                    ac = (i + 0.5) * per
                    for ph, dd in FLUTE_PTS:
                        a = ac + ph * FLUTE_HW * sw * per
                        rr = r - dep * dd
                        pts.append((ox + rr * cos(a), oy + rr * sin(a)))
        else:
            pitch = (2 - 2 * margin) * r / nfl
            dep = 0.3 * fl_w * s
            for q in range(4):
                ang = q * pi / 2
                n, t = (cos(ang), sin(ang)), (-sin(ang), cos(ang))
                pts.append((ox + n[0] * r + t[0] * (-r + cch), oy + n[1] * r + t[1] * (-r + cch)))
                if fluted:
                    for i in range(nfl):
                        ac = -r + margin * r + (i + 0.5) * pitch
                        for ph, dd in FLUTE_PTS:
                            a = ac + ph * FLUTE_HW * sw * pitch
                            b = r - dep * dd
                            pts.append((ox + n[0] * b + t[0] * a, oy + n[1] * b + t[1] * a))
                else:
                    for i in range(mplain):
                        a = (-r + cch) + 2 * (r - cch) * (i + 1) / (mplain + 1)
                        pts.append((ox + n[0] * r + t[0] * a, oy + n[1] * r + t[1] * a))
                pts.append((ox + n[0] * r + t[0] * (r - cch), oy + n[1] * r + t[1] * (r - cch)))
        sect_cache[key] = pts
        return pts

    # global rings: (z, joint index or -1, part -1 / 0 / +1)
    z_splash = zs0 + 0.16 + 0.12 * G                              # top of the splash-back zone on the shaft
    keys = [(zs0 - 0.008, -1, 0), (zs1 + 0.012, -1, 0)]
    if fluted:
        for f in (0.0, 0.4, 1.0):
            keys += [(zf0 + le * f, -1, 0), (zf1 - le * f, -1, 0)]
    lo_free = (zf0 + le) if fluted else zs0
    if lo_free + 0.06 < z_splash < (joints[0] - 0.05 if joints else zs1 - 0.2):
        keys.append((z_splash, -1, 0))
    for j, zj in enumerate(joints):
        keys += [(zj - gjs[j], j, -1), (zj + gjs[j], j, 1)]
    keys.sort()
    gap = Ls / (2.0 if fluted else 7.0)
    ringsG = []
    for a, b in zip(keys[:-1], keys[1:]):
        ringsG.append(a)
        if a[1] >= 0 and b[1] == a[1]:
            continue
        m = int((b[0] - a[0]) / gap)
        for i in range(1, m + 1):
            ringsG.append((a[0] + (b[0] - a[0]) * i / (m + 1), -1, 0))
    ringsG.append(keys[-1])

    zmid = 0.5 * (zs0 + zs1)
    ref = section(zmid)
    ncol = len(ref)
    cang = [atan2(y, x) for x, y in ref]

    # entries per column: [z, z for the section, joint-mid index, ring id, slide-to column, slide dist, inward push]
    entries = [[] for _ in range(ncol)]
    ring_of = {}
    for g, (zr_, j, part) in enumerate(ringsG):
        if j >= 0:
            ring_of[(j, part)] = g
        for i in range(ncol):
            zz = zr_
            if j >= 0:
                a_, p1, m1, p2, m2 = jwob[j]
                th = 2 * pi * i / ncol
                zz = joints[j] + a_ * sin(m1 * th + p1) + part * gjs[j] * (1.0 + 0.45 * sin(m2 * th + p2))
            entries[i].append([zz, zr_, j if (j >= 0 and part == 0) else -1, g, None, 0.0, 0.0])
    ringz = [[e[0] for e in entries[i]] for i in range(ncol)]
    occ = [[] for _ in range(ncol)]

    def free(i, za, zb, rings=True):
        if rings:
            for zz in ringz[i]:
                if za - 0.006 < zz < zb + 0.006:
                    return False
        for a, b in occ[i]:
            if za < b + 0.008 and zb > a - 0.008:
                return False
        return True

    def add(i, zz, nb=None, ds=0.0, dr=0.0):
        entries[i].append([zz, zz, -1, None, nb, ds, dr])

    # tethering ring: which columns of the shaft it sits on (kept free of damage)
    ring_on = p.ring == "yes" or (p.ring == "auto" and rng.fork("ring_on").chance(0.25))
    ring_on = ring_on and H >= 1.7
    cR = min(range(ncol), key=lambda i: abs((cang[i] + pi / 2 + pi) % (2 * pi) - pi))
    if ring_on:
        for k in range(-3, 4):
            occ[(cR + k) % ncol].append((zs0, zs1))

    # --- spalls knocked out of the arrises (flute fillets, pier corners): real geometry
    arrises = []
    if not sqm and fluted:
        arrises = [(4 * i + 2, 4 * i + 3, (4 * i + 4) % ncol, (4 * i + 5) % ncol) for i in range(nf)]
    elif sqm:
        m = ncol // 4
        for q in range(4):
            e = q * m + m - 1
            for _ in range(3 if fluted else 2):                   # corners take most of the knocks
                arrises.append((e - 1, e, (e + 1) % ncol, (e + 2) % ncol))
            if fluted:
                for k in range(nfl - 1):
                    b0 = q * m + 1 + 4 * k
                    arrises.append((b0 + 2, b0 + 3, b0 + 4, b0 + 5))
    rch = rng.fork("chips")
    # very big, many-fluted, many-drum shafts: thin the knocks out so the whole asset stays inside the budget
    room = 11000 - (6300 if order == "foliate" else 3600) - 2 * ncol * len(ringsG)
    est = dmg * (min(len(arrises), 24) * 25 + nj * 140) + 1.0
    kd = clamp(room / est, 0.2, 1.0)
    chips = []                                                    # (column B, column C, z, radius) for the fresh-stone mask
    if arrises and dmg > 0.02:
        zlo, zhi = (zf0 + le + 0.03, zf1 - le - 0.03) if fluted else (zs0 + 0.06, zs1 - 0.06)
        want = int(dmg * min(len(arrises), 24) * 1.25 * kd + 0.5)
        made = tries = 0
        while made < want and tries < want * 8:
            tries += 1
            A_, B_, C_, D_ = rch.pick(arrises)
            L = rch.u(0.035, 0.14) * (0.7 + 0.75 * D)
            zc = zlo + (zhi - zlo) * rch.u(0.0, 1.0) ** 1.5
            za, zb = zc - L / 2, zc + L / 2
            if za < zlo or zb > zhi or not all(free(i, za, zb) for i in (A_, B_, C_, D_)):
                continue
            big = rch.u(0.005, 0.015) * (0.7 + 0.8 * dmg)
            dsB, dsC = big * rch.u(0.45, 1.0), big * rch.u(0.45, 1.0)
            push = rch.u(0.001, 0.004)
            ts = [-0.5, -0.5 + rch.u(0.08, 0.2), rch.u(-0.15, 0.15), 0.5 - rch.u(0.08, 0.25), 0.5]
            fs = [0.0, rch.u(0.7, 1.0), 1.0, rch.u(0.6, 1.0), 0.0]
            for t, f in zip(ts, fs):
                add(B_, zc + t * L, A_, dsB * f, push * f)
                add(C_, zc + t * L, D_, dsC * f, push * f)
            for i in (A_, B_, C_, D_):
                occ[i].append((za, zb))
            chips.append((B_, C_, zc, L * 0.4))
            made += 1

    # --- spalled edges at the drum joints (above or below the bed), scoops across a few columns
    for j in range(nj):
        for _ in range(int((0.7 + 3.4 * dmg + rch.u(0, 1)) * kd) if dmg > 0.02 else 0):
            w = int(rch.u(3, 9.99)) if (fluted or sqm) else int(rch.u(2, 5.99))
            c0 = int(rch.u(0, ncol))
            sg = 1 if rch.chance(0.5) else -1
            g = ring_of[(j, sg)]
            depth = rch.u(0.004, 0.011) * (0.7 + 0.8 * dmg)
            hh = rch.u(0.014, 0.045) * (0.7 + 0.75 * D)
            cols_ = [(c0 + k) % ncol for k in range(w)]
            es = [next(e for e in entries[i] if e[3] == g) for i in cols_]
            if not all(free(i, min(e[0], e[0] + sg * hh) - 0.004, max(e[0], e[0] + sg * hh) + 0.004, rings=False)
                       for i, e in zip(cols_, es)):
                continue
            for k, (i, e) in enumerate(zip(cols_, es)):
                lat = sin(pi * (k + 0.5) / w) ** 0.8
                e[6] = depth * lat
                add(i, e[0] + sg * hh * lat * 0.62, dr=depth * lat * 0.85)
                add(i, e[0] + sg * (hh * lat + 0.003), dr=0.0)
                occ[i].append((min(e[0], e[0] + sg * hh) - 0.004, max(e[0], e[0] + sg * hh) + 0.004))
            for i in ((c0 - 1) % ncol, (c0 + w) % ncol):          # sound stone either side keeps the scar local
                e = next(e for e in entries[i] if e[3] == g)
                if free(i, min(e[0], e[0] + sg * hh) - 0.004, max(e[0], e[0] + sg * hh) + 0.004, rings=False):
                    add(i, e[0] + sg * hh * 0.5)
                    occ[i].append((min(e[0], e[0] + sg * hh) - 0.004, max(e[0], e[0] + sg * hh) + 0.004))
            mc = cols_[w // 2]
            chips.append((mc, mc, es[w // 2][0] + sg * hh * 0.4, max(hh * 0.5, 0.012)))

    # --- bruises on a plain round shaft (no arrises to chip)
    if not sqm and not fluted and dmg > 0.02:
        for _ in range(int(dmg * 9 + 0.5)):
            w = int(rch.u(2, 4.99))
            c0 = int(rch.u(0, ncol))
            zc = zs0 + 0.1 + (Ls - 0.3) * rch.u(0, 1) ** 1.6
            hh = rch.u(0.02, 0.06)
            depth = rch.u(0.003, 0.007) * (0.7 + 0.8 * dmg)
            cols_ = [(c0 + k) % ncol for k in range(w)]
            flank = [(c0 - 1) % ncol, (c0 + w) % ncol]
            if not all(free(i, zc - hh, zc + hh) for i in cols_ + flank):
                continue
            for i in flank:                                       # sound stone either side keeps the scar local
                add(i, zc - hh * 0.5)
                add(i, zc + hh * 0.5)
                occ[i].append((zc - hh, zc + hh))
            for k, i in enumerate(cols_):
                lat = sin(pi * (k + 0.5) / w) ** 0.8
                h2 = hh * (0.4 + 0.6 * lat)
                add(i, zc - h2)
                add(i, zc - h2 * 0.6, dr=depth * lat * 0.8)
                add(i, zc + h2 * 0.35, dr=depth * lat)
                add(i, zc + h2)
                occ[i].append((zc - hh, zc + hh))
            mc = cols_[w // 2]
            chips.append((mc, mc, zc, hh * 0.6))

    # --- rain runs under the astragal: thin streaks down single flutes / arrises, each ending on its own vertex
    rs = rng.fork("streaks")
    groups = []
    if not sqm and fluted:
        for i in range(nf):
            groups.append((4 * i + 1, 4 * i + 2))
            groups.append((4 * i + 3, (4 * i + 4) % ncol))
    elif sqm and fluted:
        m = ncol // 4
        for q in range(4):
            for k in range(nfl):
                groups.append((q * m + 1 + 4 * k + 1, q * m + 1 + 4 * k + 2))
            for k in range(nfl - 1):
                groups.append((q * m + 1 + 4 * k + 3, q * m + 1 + 4 * k + 4))
    else:
        i = 0
        while i < ncol:
            if sqm and (i % (ncol // 4) in (0, ncol // 4 - 1)):
                i += 1
                continue
            if sqm and rs.chance(0.3) and (i + 1) % ncol != 0 and (i + 1) % (ncol // 4) != ncol // 4 - 1:
                groups.append((i, i + 1))
                i += 2
            else:
                groups.append((i,))
                i += 1
    streaks = []                                                  # (theta, inner half width, outer half width, length, strength)

    def adiff(a, b):
        return abs((a - b + pi) % (2 * pi) - pi)

    for mem in groups:
        if not rs.chance((0.12 + 0.28 * G) if fluted else (0.1 + 0.2 * G)):
            continue
        L = rs.u(0.07, 0.3) * (0.6 + 1.3 * G)
        if rs.chance(0.28):
            L *= rs.u(1.8, 3.4)
        if not fluted:
            L *= 1.5
        L = min(L, Ls - 0.35)
        if L < 0.05:
            continue
        S = rs.u(0.3, 0.8) * (1.0 if fluted else 0.62)
        vx = sum(cos(cang[i]) for i in mem)
        vy = sum(sin(cang[i]) for i in mem)
        th = atan2(vy, vx)
        h0 = max(adiff(cang[i], th) for i in mem)
        hw = min(adiff(cang[(mem[0] - 1) % ncol], th), adiff(cang[(mem[-1] + 1) % ncol], th)) * 0.97
        if hw <= h0 + 1e-4:
            continue
        ze = zs1 - L
        ok = [i for i in mem if free(i, ze - 0.003, ze + 0.003)]
        for i in ok:
            add(i, ze)
            occ[i].append((ze - 0.002, ze + 0.002))
        if ok:
            streaks.append((th, h0, hw, L, S))

    # --- skin
    verts, vcol, vjoint = [], [], []
    cols = []
    for i in range(ncol):
        col = []
        for e in sorted(entries[i], key=lambda e: e[0]):
            pts = section(e[1], e[2])
            x, y = pts[i]
            if e[4] is not None and e[5] > 0:
                x2, y2 = pts[e[4]]
                f = min(0.62, e[5] / max(hypot(x2 - x, y2 - y), 1e-6))
                x, y = x + (x2 - x) * f, y + (y2 - y) * f
            if e[6]:
                k = 1.0 - e[6] / max(hypot(x, y), 1e-6)
                x, y = x * k, y * k
            col.append((e[0], e[3], len(verts)))
            verts.append((x, y, e[0]))
            vcol.append(i)
            g = e[3]
            vjoint.append(ringsG[g][1] if g is not None else -1)
        cols.append(col)
    faces = []
    for i in range(ncol):
        _merge_strip(cols[i], cols[(i + 1) % ncol], faces)
    shaft = mesh_object("shaft", verts, faces)
    me = shaft.data
    S_ = [0.0]
    for i in range(ncol):
        a, b = ref[i], ref[(i + 1) % ncol]
        S_.append(S_[-1] + hypot(b[0] - a[0], b[1] - a[1]))
    drum_mats = t_drums[:nj + 1]
    slots = []
    for m_ in drum_mats + [mortar]:
        if m_ not in slots:
            slots.append(m_)
            me.materials.append(m_)
    xf = []
    for k in range(nj + 1):
        a = radians(BED + rb_.u(-5, 5) + (180.0 if rb_.chance(0.5) else 0.0))
        xf.append((cos(a), sin(a), rb_.u(0, 3), rb_.u(0, 3)))
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        vs = list(poly.vertices)
        cs = [vcol[v] for v in vs]
        wrapc = (ncol - 1 in cs) and (0 in cs)
        zc = sum(verts[v][2] for v in vs) / len(vs)
        js = set(vjoint[v] for v in vs)
        k = sum(1 for zj in joints if zc > zj)
        if len(js) == 1 and -1 not in js:
            poly.material_index = slots.index(mortar)
        else:
            poly.material_index = slots.index(drum_mats[k])
        ca, sa, ou, ov = xf[k]
        for li, v in zip(poly.loop_indices, vs):
            uu = S_[ncol] if (wrapc and vcol[v] == 0) else S_[vcol[v]]
            vv = verts[v][2]
            uvl.data[li].uv = (uu * ca - vv * sa + ou, uu * sa + vv * ca + ov)
    uv_keep(shaft)
    smooth(shaft, 40)
    objs.append(shaft)

    def col_point(i, zz):
        C = cols[i]
        for k in range(len(C) - 1):
            if C[k][0] <= zz <= C[k + 1][0]:
                a, b = verts[C[k][2]], verts[C[k + 1][2]]
                t = (zz - C[k][0]) / max(C[k + 1][0] - C[k][0], 1e-9)
                return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, zz)
        return verts[C[-1][2]]

    chip_pts = []
    for b_, c_, zc, rad in chips:
        a, b = col_point(b_, zc), col_point(c_, zc)
        chip_pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, zc, rad))

    # ---------------- capital
    u = uc
    zc = zs1
    ka = 1.5 if order == "cushion" else 1.0                       # astragal weight
    A = [(rt - 0.008, zc), (rt + 0.010 * u, zc), (rt + 0.012 * u, zc + 0.010 * u)]       # astragal
    _arc(A, rt + 0.018 * u * ka, zc + 0.036 * u, 0.026 * u * ka, -80, 80, 3)
    A += [(rt + 0.012 * u, zc + 0.062 * u), (rt, zc + 0.076 * u)]
    rc = rng.fork("capital")
    cap_bed = []                                                  # parts that get the capital block's bedding

    if order == "tuscan":
        Pc = list(A)
        Pc += [(rt, zc + 0.22 * u), (rt + 0.013 * u, zc + 0.223 * u), (rt + 0.013 * u, zc + 0.24 * u),
               (rt + 0.026 * u, zc + 0.243 * u), (rt + 0.026 * u, zc + 0.26 * u), (rt + 0.039 * u, zc + 0.263 * u),
               (rt + 0.039 * u, zc + 0.28 * u)]
        for i in range(1, 6):                                     # echinus (ovolo)
            th = i / 5 * pi / 2
            Pc.append((rt + (0.039 + 0.15 * sin(th)) * u, zc + (0.432 - 0.15 * cos(th)) * u))
        Pc += [(rt + 0.15 * u, zc + 0.434 * u), (rt * 0.8, zc + 0.434 * u)]
        capm = _turned("capital", Pc, sqm, cch, segs=28)
        cap_objs.append(capm)
        cap_bed.append(capm)
        wa = rt + 0.215 * u
        z_a, cf = zc + 0.43 * u, 0.006 + 0.008 * u
        ab = _slab("abacus", [(_sq(wa - cf, cch * 0.6), z_a), (_sq(wa, cch * 0.6), z_a + cf), (_sq(wa, cch * 0.6), H - cf),
                              (_sq(wa - cf, cch * 0.6), H)], rc, dmg, zlow=z_a + cf)
        cap_objs.append(ab)
        cap_bed.append(ab)
        # rosettes carved on the necking
        ros = lathe("rosette", [(0.0, 0.0), (0.042 * u, 0.0), (0.046 * u, 0.008 * u), (0.03 * u, 0.016 * u), (0.02 * u, 0.013 * u),
                                (0.014 * u, 0.024 * u), (0.0, 0.027 * u)], segments=8)
        rm_ = []
        zro = zc + 0.148 * u
        if not sqm:
            for k in range(8):
                ar = radians(k * 45.0)
                rm_.append(trs(((rt - 0.002) * sin(ar), -(rt - 0.002) * cos(ar), zro), (90, 0, k * 45.0)))
        else:
            for q in range(4):
                ar = radians(q * 90.0)
                n, t = (sin(ar), -cos(ar)), (cos(ar), sin(ar))
                for off in (-0.5, 0.5):
                    rm_.append(trs((n[0] * (rt - 0.002) + t[0] * off * rt, n[1] * (rt - 0.002) + t[1] * off * rt, zro),
                                   (90, 0, q * 90.0)))
        cap_objs.append(instances_to_mesh("rosettes", ros, rm_, keep_base=False))

    elif order == "cushion":
        ast = _turned("astragal", A, sqm, cch)
        cap_objs.append(ast)
        cap_bed.append(ast)
        wa = min(rt + 0.215 * u, 0.675 * D)                       # impost slab half width: capped at 1.35 x diameter
        w = wa - 0.05 * u                                         # the cube itself
        z0 = zc + 0.074 * u
        hcv, hcu = 0.46 * u, 0.62 * u
        rings = []
        if not sqm:
            rn = rt * 1.01
            Rmax = w * 1.37
            for i in range(11):
                th = min(i, 9) / 9 * pi / 2
                R = rn + (Rmax - rn) * sin(th)
                zz = z0 + hcv * (1 - cos(th)) if i < 10 else z0 + hcu
                thc = acos(min(1.0, w / R)) if R > w else 0.0
                pts = []
                for q in range(4):
                    fa = q * pi / 2
                    for a in (-thc, 0.0, thc):
                        rr = w / cos(a)
                        pts.append((rr * cos(fa + a), rr * sin(fa + a)))
                    for j in range(1, 6):
                        a = thc + (pi / 2 - 2 * thc) * j / 6
                        pts.append((R * cos(fa + a), R * sin(fa + a)))
                rings.append((pts, zz))
            cush = _loft("cushion", rings, cap_top=True)
            smooth(cush, 20)
            uv_box(cush)
            # raised lunette band carved on each face, following the shield outline
            lun = []
            for i in range(11):
                th = min(i, 9) / 9 * pi / 2
                R = rn + (Rmax - rn) * sin(th)
                if R > w * 1.0005:
                    lun.append((sqrt(R * R - w * w), z0 + hcv * (1 - cos(th)) if i < 10 else z0 + hcu - 0.03 * u))
            ztop = z0 + hcu
            k_in = 0.8
            half = [(y * k_in, ztop - (ztop - zz) * k_in - 0.035 * u) for y, zz in lun]
            th0 = float(np.arcsin(min(1.0, (w - rn) / (Rmax - rn))))
            half = [(0.0, ztop - (ztop - (z0 + hcv * (1 - cos(th0)))) * k_in - 0.035 * u)] + half
            path2 = [(-y, zz) for y, zz in reversed(half[1:])] + [(y, zz) for y, zz in half]
            path2 = resample_path([(a, 0, zz) for a, zz in path2], n=22)
            bandp = [(-0.016 * u, -0.01 * u), (0.016 * u, -0.01 * u), (0.016 * u, 0.01 * u), (-0.016 * u, 0.01 * u)]
            for q in range(4):
                ar = radians(q * 90.0)
                n, t = (sin(ar), -cos(ar)), (cos(ar), sin(ar))
                pts = [(n[0] * (w + 0.004 * u) + t[0] * q3[0], n[1] * (w + 0.004 * u) + t[1] * q3[0], q3[2]) for q3 in path2]
                cap_objs.append(sweep(f"lunette{q}", pts, bandp, up=n + (0,), smooth_path=False))
        else:
            for i in range(11):
                th = min(i, 9) / 9 * pi / 2
                r = rt * 1.01 + (w - rt * 1.01) * sin(th)
                zz = z0 + hcv * (1 - cos(th)) if i < 10 else z0 + hcu
                rings.append((_sq(r, cch * 1.2), zz))
            cush = _loft("cushion", rings, cap_top=True)
            smooth(cush, 30)
            uv_box(cush)
        cap_objs.append(cush)
        cap_bed.append(cush)
        # mortar bed, then the impost slab: hollow chamfer below a crisp fascia
        z1 = z0 + hcu
        mb = 0.0035 + 0.003 * u
        bedm = _loft("impost_bed", [(_sq(w - 0.004 - 0.004 * u, cch), z1 - 0.002), (_sq(w - 0.004 - 0.004 * u, cch), z1 + mb + 0.002)])
        uv_box(bedm)
        assign(bedm, mortar)
        objs.append(bedm)
        z1 += mb
        Pa = [(w - 0.004 * u, z1), (w + 0.002 * u, z1 + 0.014 * u)]
        for i in range(1, 6):                                     # hollow chamfer under the slab
            th = i / 5 * pi / 2
            Pa.append((w + 0.002 * u + (wa - w - 0.002 * u) * (1 - cos(th)), z1 + 0.014 * u + 0.08 * u * sin(th)))
        zf_ = z1 + 0.094 * u
        cf = 0.005 + 0.006 * u
        Pa += [(wa, zf_ + 0.004 * u), (wa, H - cf), (wa - cf, H)]
        ab = _slab("impost", [(_sq(r, cch), zz) for r, zz in Pa], rc, dmg, zlow=zf_ + 0.004 * u)
        t_imp = tones[(ti[1] + 1) % 3]                            # the impost is another stone than the cube
        assign(ab, t_imp)
        _bed(ab, rb_, BED)
        objs.append(ab)

    else:  # foliate
        zb0 = zc + 0.074 * u
        zb1 = zc + 0.96 * u
        flare = 0.15 * u

        def rbell(zz):
            t = clamp((zz - zb0) / (zb1 - zb0))
            return rt + 0.004 * u + flare * t ** 2.4

        def rdiag(zz):
            return rbell(zz) if not sqm else (rbell(zz) - cch * 0.5) * sqrt(2.0)

        Pb = list(A[:-1])
        for i in range(0, 5):
            zz = zb0 + (zb1 - zb0) * i / 4
            Pb.append((rbell(zz), zz))
        Pb += [(rbell(zb1) + 0.012 * u, zb1 + 0.004 * u), (rbell(zb1) + 0.012 * u, zb1 + 0.014 * u), (rt * 0.8, zb1 + 0.014 * u)]
        if sqm:
            bell = _turned("bell", Pb, sqm, cch)
        else:                                                     # the astragal shows, the bell hides behind the leaves
            ast = lathe("astragal", list(A), segments=24, smooth_angle=42)
            cap_objs.append(ast)
            cap_bed.append(ast)
            bell = lathe("bell", Pb[len(A) - 1:], segments=16, smooth_angle=42)
        cap_objs.append(bell)
        cap_bed.append(bell)

        # abacus: concave sides, cut horns, cavetto + fillet
        wa = rt + 0.32 * u
        cw = 0.085 * u
        dep = 0.07 * u

        def ab_outline(k):
            pts = []
            for q in range(4):
                fa = q * pi / 2
                n, t = (cos(fa), sin(fa)), (-sin(fa), cos(fa))
                half = wa - cw
                for i in range(5):
                    a = -half + 2 * half * i / 4
                    b = wa - dep * (1 - (a / half) ** 2)
                    pts.append(((n[0] * b + t[0] * a) * k, (n[1] * b + t[1] * a) * k))
            return pts

        za = zb1 + 0.012 * u
        ha = H - za
        abac = _loft("abacus", [(ab_outline(0.9), za), (ab_outline(0.912), za + 0.22 * ha), (ab_outline(0.958), za + 0.5 * ha),
                                (ab_outline(0.963), za + 0.58 * ha), (ab_outline(1.0), za + 0.64 * ha),
                                (ab_outline(1.0), H - 0.005), (ab_outline(0.988), H)], cap_top=True)
        planes = []                                               # knocked horns
        for q in range(4):
            if rc.chance(0.15 + 0.9 * dmg):
                a = radians(45.0 + 90.0 * q)
                dv = Vector((cos(a) * rc.u(0.7, 1.3), sin(a) * rc.u(0.7, 1.3), rc.u(0.5, 1.3))).normalized()
                c0 = Vector(((wa - cw * 0.5) * sqrt(2.0) * cos(a), (wa - cw * 0.5) * sqrt(2.0) * sin(a), H))
                planes.append((c0, dv, rc.u(0.012, 0.03) * (0.5 + dmg) * (u / 0.4)))
        _knock(abac, planes)
        smooth(abac, 30)
        uv_box(abac)
        cap_objs.append(abac)
        cap_bed.append(abac)

        # acanthus: a lower row of 8 and a taller upper row of 8 standing behind / between them
        zl = zb0 - 0.004 * u
        thick = 0.05 * u
        ml, mh = [], []
        if not sqm:
            r0 = rbell(zl)
            pitch = 2 * pi * (r0 + thick) / 8
            lo = _acanthus("leaf_lo", pitch * 1.06, 0.41 * u, thick, 1.0 / (r0 + thick), 3, lean=4.0)
            hi = _acanthus("leaf_hi", pitch * 1.04, 0.70 * u, thick, 1.0 / (r0 + thick), 2, s_start=0.46, lean=6.5, rin=0.075 * u)
            for k in range(8):
                a = k * 45.0 + 22.5
                ar = radians(a)
                rr = r0 + 0.95 * thick
                ml.append(trs((rr * sin(ar), -rr * cos(ar), zl), (rc.u(-1.5, 1.5), 0, a + 5.0 + rc.u(-2, 2)),
                              (1.0, 1.0, rc.u(0.95, 1.05))))
                a = k * 45.0
                ar = radians(a)
                rr = r0 + 0.38 * thick
                sz = rc.u(1.09, 1.15) if k % 2 == 0 else rc.u(0.97, 1.02)      # face-centre leaves reach for the rosette
                mh.append(trs((rr * sin(ar), -rr * cos(ar), zl), (rc.u(-1.5, 1.5), 0, a + rc.u(-2, 2)), (1.0, 1.0, sz)))
        else:
            rs_ = rbell(zl)
            lo = _acanthus("leaf_lo", rs_ * 1.0, 0.41 * u, thick, 0.0, 3, lean=4.0)
            hi = _acanthus("leaf_hi", rs_ * 0.96, 0.70 * u, thick, 0.0, 2, s_start=0.46, lean=6.5, rin=0.075 * u)
            for q in range(4):
                a = q * 90.0
                ar = radians(a)
                n, t = (sin(ar), -cos(ar)), (cos(ar), sin(ar))
                for off in (-0.5, 0.5):
                    rr = rs_ + 0.95 * thick
                    ml.append(trs((n[0] * rr + t[0] * off * rs_, n[1] * rr + t[1] * off * rs_, zl),
                                  (rc.u(-1.5, 1.5), 0, a + 4.0 + rc.u(-2, 2)), (1.0, 1.0, rc.u(0.95, 1.05))))
                rr = rs_ + 0.38 * thick
                mh.append(trs((n[0] * rr, n[1] * rr, zl), (0, 0, a), (1.0, 1.0, rc.u(1.09, 1.15))))
                a2 = radians(a + 45.0)
                rd = (rs_ - cch * 0.5) * sqrt(2.0) + 0.2 * thick
                mh.append(trs((rd * sin(a2), -rd * cos(a2), zl), (0, 0, a + 45.0), (0.8, 1.0, rc.u(0.97, 1.02))))
        for lf in (lo, hi):
            assign(lf, t_cap)
            _bed(lf, rb_, BED)
        cap_objs.append(instances_to_mesh("leaves_lo", lo, ml, keep_base=False))
        cap_objs.append(instances_to_mesh("leaves_hi", hi, mh, keep_base=False))

        # corner volutes: a thick scroll under every abacus horn, carried on a caulicolus fused to the bell
        tb, bw = 0.075 * u, 0.105 * u                             # band thickness in the scroll plane / width across
        r1 = 0.14 * u
        Rc1, Rc0, turns = r1 - tb * 0.5, 0.035 * u, 1.3
        dcorner = (wa - cw * 0.5) * sqrt(2.0) * 0.9
        xc, zv = dcorner - 0.72 * r1, za + 0.004 - r1
        top = zv + Rc1
        zq0, zq1 = zb0 + 0.50 * u, min(zb0 + 0.68 * u, zv - 0.2 * r1)
        a1 = rdiag(zq1) + 0.2 * tb
        p2 = [(rdiag(zq0) - 0.15 * tb, zq0), (rdiag(0.5 * (zq0 + zq1)) + 0.05 * tb, 0.5 * (zq0 + zq1))]
        p2 += [(q[0], q[2]) for q in bezier_path((a1, 0, zq1), (a1 + 0.08 * (xc - a1), 0, zq1 + 0.62 * (top - zq1)),
                                                 (xc - 1.1 * Rc1, 0, top), (xc, 0, top), n=5)]
        nn = int(turns * 10)
        for i in range(1, nn + 1):
            t = i / nn
            a = pi / 2 - turns * 2 * pi * t
            rr = Rc1 + (Rc0 - Rc1) * t
            p2.append((xc + rr * cos(a), zv + rr * sin(a)))
        band = [(-tb / 2, -bw / 2), (tb / 2, -bw / 2), (tb / 2, bw / 2), (-tb / 2, bw / 2)]
        for k in range(4):
            yr = radians(45.0 + 90.0 * k)
            d = (cos(yr), sin(yr))
            vol = sweep(f"volute{k}", [(d[0] * a, d[1] * a, zz) for a, zz in p2], band, up=(-d[1], d[0], 0),
                        smooth_path=False, scale_fn=lambda t: 1.0 - 0.3 * t)
            smooth(vol, 42)
            cap_objs.append(vol)
            eye = cylinder(f"volute_eye{k}", 0.05 * u, bw * 0.98, loc=(xc * d[0], xc * d[1], zv), rot=(90, 0, 45.0 + 90.0 * k),
                           verts=6, bevel=0.0)
            smooth(eye, 50)
            cap_objs.append(eye)

        # rosette on the middle of every abacus face
        Rr, hr = 0.47 * ha, 0.24 * ha
        ro = lathe("rosette", [(Rr, 0.0), (Rr * 0.9, hr * 0.7), (Rr * 0.3, hr * 0.35), (Rr * 0.22, hr), (0.0, hr * 1.05)],
                   segments=8)

        def petal(co):                                            # four petals round a button
            rr = hypot(co.x, co.y)
            if rr < Rr * 0.26:
                return co
            f = 0.5 - 0.5 * cos(4.0 * atan2(co.y, co.x))
            k = 0.52 + 0.48 * f
            return Vector((co.x * k, co.y * k, co.z * (0.4 + 0.6 * f)))

        transform_verts(ro, petal)
        smooth(ro, 50)
        fm = []
        for q in range(4):
            ar = radians(q * 90.0)
            rr = (wa - dep) * 0.955
            fm.append(trs((rr * sin(ar), -rr * cos(ar), za + 0.54 * ha), (90, 0, q * 90.0 + 0.0), 1.0))
        cap_objs.append(instances_to_mesh("rosettes", ro, fm, keep_base=False))

    for o in cap_objs:
        assign(o, t_cap)
    for o in base_objs:
        assign(o, t_base)
    for o in cap_bed + base_objs:
        _bed(o, rb_, BED)
    objs += base_objs + cap_objs

    # ---------------- rusty iron tethering ring, leaded into the shaft, and its rust run down the stone
    zr = None
    if ring_on:
        iron = mat("iron_rusty")
        zr = min(1.12, zs0 + 0.5 * Ls)
        pr = col_point(cR, zr)
        ry = hypot(pr[0], pr[1])
        rr = 0.05 + 0.02 * D
        wire = 0.007 + 0.004 * D
        ri = rng.fork("ring")
        staple = torus("ring_staple", major=wire * 2.1, minor=wire * 0.8, loc=(0, -ry + wire * 0.3, zr), rot=(0, 90, 0),
                       major_seg=8, minor_seg=4)
        ring = torus("ring", major=rr, minor=wire, loc=(0, -ry - wire * 1.9, zr - rr + wire * 1.2),
                     rot=(90 + ri.u(4, 9), ri.u(-6, 6), 0), major_seg=14, minor_seg=5)
        plug = cylinder("ring_plate", wire * 3.2, 0.007, loc=(0, -ry + 0.0005, zr), rot=(90, 0, 0), verts=8)
        displace_noise(plug, strength=0.0015, scale=0.02, seed=p.seed)
        for o in (staple, ring, plug):
            assign(o, iron)
            smooth(o, 50)
            paint(o, special=noise_mask(0.03, seed=p.seed + 5, lo=0.2, hi=0.7), wear=0.15)
        objs += [staple, ring, plug]
        # rust run: a tapering tongue of stained stone lying on the shaft surface under the staple
        kk = [-2, -1, 0, 1, 2] if (fluted or sqm) else [-1, 0, 1]
        L0 = ri.u(0.3, 0.55)
        lens = [0.0] + [L0 * (1.0 if k == len(kk) // 2 else ri.u(0.3, 0.7)) for k in range(1, len(kk) - 1)] + [0.0]
        ztop = zr - wire * 1.5
        sv, sf, scols = [], [], []
        for k, Lk in zip(kk, lens):
            i = (cR + k) % ncol
            line = []
            if Lk > 0:
                ze = max(zs0 + 0.02, ztop - Lk)
                line.append((ze, None, col_point(i, ze)))
                for zz, g, vi in cols[i]:
                    if ze + 0.002 < zz < ztop - 0.002:
                        line.append((zz, g, verts[vi]))
            line.append((ztop, "top", col_point(i, ztop)))
            col = []
            for zz, g, pt in line:
                rr_ = hypot(pt[0], pt[1])
                kq = 1.0 + 0.0012 / rr_
                col.append((zz, g, len(sv)))
                sv.append((pt[0] * kq, pt[1] * kq, pt[2]))
            scols.append(col)
        for a, b in zip(scols[:-1], scols[1:]):
            _merge_strip(a, b, sf)
        stain = mesh_object("rust_run", sv, sf)
        smooth(stain, 60)
        uv_box(stain)
        assign(stain, mat("stone_smooth", _mix(_tone(p.stone, 0.8), "#7a4424", 0.6)))
        paint(stain, grime=lambda co, n: np.clip(0.55 - (ztop - co[:, 2]) / L0 * 0.6, 0.0, 1.0))
        objs.append(stain)

    # ---------------- moss cushions where the plinth meets the paving
    rm = rng.fork("moss")
    for i in range(int(M * 7 + 0.5)):
        a = rm.u(180, 360) if pil else rm.u(20, 160) if rm.chance(0.7) else rm.u(0, 360)
        ca, sa = cos(radians(a)), sin(radians(a))
        k = (wp + 0.004) / max(abs(ca), abs(sa))
        rad = rm.u(0.02, 0.045) * (0.6 + D)
        ms = sphere(f"moss{i}", rad, loc=(ca * k, sa * k, rad * 0.42), segments=7, rings=4,
                    scale=(rm.u(1.0, 1.9), rm.u(0.8, 1.2), 0.75))
        displace_noise(ms, strength=rad * 0.2, scale=rad * 0.9, seed=p.seed + i)
        place(ms, rot=(0, 0, a + 90))
        transform_verts(ms, lambda co: Vector((co.x, co.y, max(co.z, -rad * 0.4))))
        assign(ms, mossm)
        objs.append(ms)

    # ---------------- pilaster: slice by the wall plane
    if pil:
        keep = []
        for o in objs:
            if _clip_back(o):
                keep.append(o)
            else:
                delete([o])
        objs = keep

    # ---------------- weathering story
    seed = int(p.seed)
    nmf = noise_mask(0.05, seed=seed + 3, lo=0.2, hi=0.8)
    nm = noise_mask(0.07, seed=seed + 4, lo=0.35, hi=0.7)
    nm2 = noise_mask(0.22, seed=seed + 9, lo=0.3, hi=0.75)
    gk = 0.5 + 0.8 * G

    v_foot = 0.45 + 0.15 * G                                       # grime where the shaft stands on the base

    def splash(z):
        """Splash-back dirt, piecewise LINEAR in z (extra vertices of spalls sit on the ramp and make no blotches):
        dirty plinth, fading up the base, a ragged edge a hand or two up the shaft."""
        return np.interp(z, [0.0, zs0, z_splash, z_splash + 0.25], [0.76, v_foot + 0.04, 0.27, 0.0])

    def stone_grime(co, n):
        z = co[:, 2]
        under = np.clip(-n[:, 2], 0, 1) * 0.45 * gk                                     # dirt under the mouldings
        return np.clip(np.maximum(splash(z) * (0.8 + 0.2 * nmf(co, n)), under), 0, 0.8)

    def shaft_grime(co, n):
        z = co[:, 2]
        ang = np.arctan2(co[:, 1], co[:, 0])
        # the dirt climbs unevenly, flute by flute: a factor per vertex column (constant along z)
        g = splash(z) * (1.0 + 0.16 * np.sin(17.0 * ang + seed) + 0.1 * np.sin(29.0 * ang + 2.0 * seed)
                         + 0.07 * np.sin(41.0 * ang + 3.0 * seed))
        run = np.zeros(len(z))
        for th, h0, hw, L, S in streaks:                                                # thin rain runs
            d = np.abs((ang - th + pi) % (2 * pi) - pi)
            lat = np.clip((hw - d) / (hw - h0), 0, 1)
            run = np.maximum(run, S * lat * np.clip(1.0 - (zs1 - z) / L, 0, 1))
        return np.clip(np.maximum(g, run * (z < zs1 + 0.02) * gk * 0.85), 0, 0.8)

    def stone_wear(co, n):
        z = co[:, 2]
        foot = np.clip(1 - np.abs(z - hp * 1.2) / (hp * 1.5), 0, 1) * np.clip(n[:, 2], 0, 1) * 0.6   # boots on the plinth
        return np.clip(foot * nm(co, n) + dmg * 0.4 * nm(co * 1.7, n) * nm2(co, n), 0, 1)

    def shaft_wear(co, n):
        w = stone_wear(co, n)
        for cx, cy, cz, rad in chip_pts:                                                # fresh stone in the spalls
            d2 = (co[:, 0] - cx) ** 2 + (co[:, 1] - cy) ** 2 + (co[:, 2] - cz) ** 2
            w = np.maximum(w, (d2 < rad * rad) * 0.85)
        return w

    def stone_moss(co, n):
        z = co[:, 2]
        north = np.clip(n[:, 1], 0, 1) if not pil else np.clip(np.abs(n[:, 0]), 0, 1) * 0.7
        low = np.clip(1 - z / (zs0 + 0.04), 0, 1) ** 0.5                                # the base and the plinth only
        up = np.clip(n[:, 2], 0, 1)
        m = np.maximum(north * low, up * low) * nm2(co, n)
        return np.clip(m * M * 1.25, 0, 0.75)

    for o in objs:
        if o.name.startswith(("ring", "moss", "rust")):
            continue
        if o is shaft:
            paint(o, grime=shaft_grime, wear=shaft_wear)
        else:
            paint(o, grime=stone_grime, wear=stone_wear, special=stone_moss)

    socket("top", "attach", (0, 0, H), role="prop_top")
    ysurf = -r_at(H * 0.7)
    socket("lantern", "attach", (0, ysurf if not pil else -rb, H * 0.7), role="lantern")
