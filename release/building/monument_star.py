"""Star shrine monument: a dressed-limestone shrine built against a retaining wall. A coursed plinth with an
inscription tablet (three jointed slabs, incised bronze-filled gilt lettering, moulded frame) carries two battered pylons of coursed
blocks faced with fluted pilasters (attic bases, Doric caps), an entablature with dentils and a curved
pediment. A ring of voussoirs between the pylons frames a dark fluted shell niche; in front of it stands a
free-standing faceted star lantern of leaded amber glass (lit from inside, tied back by a bronze stay) on a
glazed bronze foot and a stone pedestal, with a gilt crown on its top point. Either side a scupper in the wing wall spills a tearing sheet of
water into a raised stone tank which overflows through weir notches into a ground basin; broad shallow steps
with raked cheek walls, bronze handrails and glazed newel lanterns lead up to the inscription.
"""
from tk import *
from math import sin, cos, pi, radians, sqrt, asin, atan2, degrees, hypot, ceil, exp
import numpy as np
import bmesh

META = {"name": "Star shrine monument", "class": "landmark", "mount": "wall_ground", "focus": 0.56}

PALETTES = {  # a = golden star facets, b = deep star facets / roundels, f = field glass, l = light colour
    "amber": {"a": "#ffb42a", "b": "#f07a0c", "f": "#2446b8", "l": "#ffc070"},
    "azure": {"a": "#6cc0ff", "b": "#1c74ee", "f": "#3a22a0", "l": "#8cc6ff"},
    "rose": {"a": "#ff8fb2", "b": "#ee2c6a", "f": "#4a2090", "l": "#ff9ab8"},
    "emerald": {"a": "#a4ec62", "b": "#1cb45a", "f": "#0e5a68", "l": "#9af0a8"},
    "sunset": {"a": "#ffa236", "b": "#ee381c", "f": "#6a2070", "l": "#ff9450"},
}
# texels of the stained-glass sheet with a clean light / mid / deep tone (one pane = one piece of glass)
PICKS = {
    "light": [(0.35, 0.77), (0.45, 0.80), (0.75, 0.59), (0.85, 0.62)],
    "mid": [(0.05, 0.68), (0.15, 0.56), (0.31, 0.47), (0.35, 0.47)],
    "deep": [(0.35, 0.62), (0.45, 0.65), (0.65, 0.86)],
}

STAR_GLOW = 1.7     # emissive gain of the star glass (the light it throws is the `glow` param)

PARAMS = {
    "width": {"default": 8.0, "min": 6.0, "max": 10.0, "desc": "overall width along the wall (m)"},
    "height": {"default": 7.4, "min": 5.6, "max": 9.6, "desc": "overall height to the finial (m)"},
    "star_points": {"default": 5, "min": 4, "max": 8, "desc": "number of points of the stained-glass star"},
    "palette": {"default": "amber", "choices": ["amber", "azure", "rose", "emerald", "sunset"], "desc": "stained-glass palette"},
    "inscription": {"default": "Let every lantern|light the way home.", "desc": "incised inscription, | = line break"},
    "water": {"default": True, "desc": "cascades running and basins full (False = drained)"},
    "tiers": {"default": 2, "choices": [1, 2], "desc": "cascade tiers per side (2 = raised tank + ground basin)"},
    "steps": {"default": 4, "min": 3, "max": 6, "desc": "number of risers up to the inscription landing"},
    "pediment": {"default": "segmental", "choices": ["segmental", "round"], "desc": "curved pediment shape"},
    "handrail": {"default": True, "desc": "bronze handrails on the stair cheeks"},
    "stone": {"default": "#b9ad98", "desc": "tint of the dressed stone"},
    "moss": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "amount of moss, algae and damp staining"},
    "date": {"default": "ANNO MDCCCXLVII", "desc": "small date line at the foot of the inscription stone"},
    "glow": {"default": 55.0, "min": 20.0, "max": 200.0, "desc": "intensity of the star light (cd)"},
    "gilding": {"default": 0.85, "min": 0.0, "max": 1.0, "desc": "gold leaf surviving in the incised letters (0 = bare dark bronze)"},
    "tablet": {"default": "#22252b", "desc": "tint of the honed inscription slab: dark slate / black marble, so gilt letters read from across "
                                             "the court (bare bronze letters, gilding 0, keep the pale dressed stone instead)"},
}
VARIANTS = {
    "grand_azure": {"width": 10.0, "height": 9.4, "star_points": 8, "palette": "azure", "pediment": "round", "steps": 5,
                    "inscription": "LUX IN TENEBRIS|LUCET", "stone": "#a9a8a2", "gilding": 1.0, "seed": 4},
    "compact_rose": {"width": 6.0, "height": 5.9, "star_points": 6, "palette": "rose", "tiers": 1, "steps": 3,
                     "inscription": "For those|who wander", "stone": "#c4b094", "moss": 0.25, "seed": 11},
    "dry_emerald": {"width": 7.2, "height": 7.0, "star_points": 7, "palette": "emerald", "water": False, "moss": 0.95,
                    "handrail": False, "stone": "#9d9a8a", "inscription": "The spring|sleeps", "gilding": 0.0, "seed": 23},
    "sunset_four": {"width": 9.0, "height": 6.6, "star_points": 4, "palette": "sunset", "pediment": "round", "steps": 6,
                    "stone": "#bfa98c", "glow": 48.0, "inscription": "Until the|evening star", "seed": 37},
}


# ----------------------------------------------------------------------------------------------- helpers
def _d(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _set_uv(o, fmap):
    """Write explicit meter UVs: fmap = {frozenset(face vertex ids): {vertex id: (u, v)}}."""
    me = o.data
    lay = me.uv_layers[0] if len(me.uv_layers) else me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        d = fmap.get(frozenset(poly.vertices))
        if d is None:
            continue
        for li in poly.loop_indices:
            lay.data[li].uv = d[me.loops[li].vertex_index]
    uv_keep(o)


def _uv_per_vertex(o, uvs):
    me = o.data
    lay = me.uv_layers[0] if len(me.uv_layers) else me.uv_layers.new(name="UVMap")
    for lp in me.loops:
        lay.data[lp.index].uv = uvs[lp.vertex_index]
    uv_keep(o)


def _vmask(o, name, vals):
    """Write a weathering mask (grime / wear / special) straight from a per-vertex list."""
    me = o.data
    if len(vals) != len(me.vertices):
        return
    at = me.attributes.get("m_" + name) or me.attributes.new("m_" + name, 'FLOAT', 'POINT')
    at.data.foreach_set("value", np.clip(np.asarray(vals, dtype=np.float32), 0.0, 1.0))


def _h1(a, b, c=0.0):
    """Stable 0..1 hash of a position: every stone gets its own tone without touching the layout streams."""
    v = sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453
    return v - int(v) if v >= 0 else v - int(v) + 1.0


def _stone_tone(a, b, c=0.0):
    """(grime, wear) of one stone: most blocks sit near the mean, a few are darker beds, a few paler and
    harder (quarried from another bed, or a later repair)."""
    h, k = _h1(a, b, c), _h1(b + 3.1, a - 1.7, c + 0.4)
    g = 0.04 + 0.22 * h ** 1.5 + (0.13 if k > 0.86 else 0.0)
    w = 0.45 + 0.35 * _h1(c + 5.0, a, b) if k < 0.22 else 0.12 * k
    return g, w


def _loft(name, secs, closed=False, caps=True):
    """Skin equal-length closed cross-sections into one solid; U runs along the sections, V around them."""
    m, n = len(secs[0]), len(secs)
    verts = [tuple(v) for s in secs for v in s]
    vv = [0.0]
    for j in range(m):
        vv.append(vv[-1] + _d(secs[0][j], secs[0][(j + 1) % m]))
    cen = [[sum(q[i] for q in s) / m for i in range(3)] for s in secs]
    nu = n if closed else n - 1
    uu = [0.0]
    for i in range(nu):
        uu.append(uu[-1] + _d(cen[i], cen[(i + 1) % n]))
    faces, fmap = [], {}
    for i in range(nu):
        a, b = i * m, ((i + 1) % n) * m
        for j in range(m):
            k = (j + 1) % m
            f = (a + j, a + k, b + k, b + j)
            faces.append(f)
            fmap[frozenset(f)] = {a + j: (uu[i], vv[j]), a + k: (uu[i], vv[j + 1]), b + k: (uu[i + 1], vv[j + 1]), b + j: (uu[i + 1], vv[j])}
    if caps and not closed:
        for f in (tuple(range(m - 1, -1, -1)), tuple(range((n - 1) * m, n * m))):
            faces.append(f)
            fmap[frozenset(f)] = {v: (verts[v][0] + verts[v][1], verts[v][2]) for v in f}
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    _set_uv(o, fmap)
    return o


def prism_x(name, yz, x0, x1):
    return _loft(name, [[(x0, y, z) for y, z in yz], [(x1, y, z) for y, z in yz]])


def prism_y(name, xz, y0, y1):
    return _loft(name, [[(x, y0, z) for x, z in xz], [(x, y1, z) for x, z in xz]])


def run_mould(name, path, prof, closed=False):
    """Mitred moulding along a plan polyline [(x, y)]; prof = closed section [(d, z)], d = offset to the
    right of the direction of travel (walk the path counter-clockwise seen from above = outwards)."""
    n = len(path)

    def nrm(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        l = hypot(dx, dy) or 1.0
        return (dy / l, -dx / l)

    secs = []
    for i in range(n):
        p1 = path[i]
        if closed:
            n0, n1 = nrm(path[i - 1], p1), nrm(p1, path[(i + 1) % n])
        else:
            n0 = nrm(path[i - 1], p1) if i > 0 else nrm(p1, path[i + 1])
            n1 = nrm(p1, path[i + 1]) if i < n - 1 else n0
        k = max(1.0 + n0[0] * n1[0] + n0[1] * n1[1], 0.3)
        mx, my = (n0[0] + n1[0]) / k, (n0[1] + n1[1]) / k
        secs.append([(p1[0] + mx * d, p1[1] + my * d, z) for d, z in prof])
    return _loft(name, secs, closed=closed)


def arc_mould(name, cx, cz, a0, a1, prof, n=48):
    """Moulding following a circular arc in the facade (XZ) plane; prof = closed section [(radius, y)].
    UVs follow the arc, so tooling marks run tangentially."""
    full = abs((a1 - a0) - 360.0) < 1e-6
    secs = []
    for i in range(n if full else n + 1):
        a = radians(a0 + (a1 - a0) * i / n)
        secs.append([(cx + r * cos(a), y, cz + r * sin(a)) for r, y in prof])
    o = _loft(name, secs, closed=full)
    smooth(o, 32)
    return o


def frame_mould(name, x0, x1, z0, z1, prof):
    """Mitred rectangular frame in the facade plane around [x0, x1] x [z0, z1]; prof = closed section
    [(d, y)] with d = offset outwards from the opening."""
    secs = []
    for (cx, cz, sx, sz) in ((x0, z0, -1, -1), (x1, z0, 1, -1), (x1, z1, 1, 1), (x0, z1, -1, 1)):
        secs.append([(cx + sx * d, y, cz + sz * d) for d, y in prof])
    return _loft(name, secs, closed=True)


def offset_poly(pts, d):
    """Mitred offset of a CCW polygon (d > 0 grows, d < 0 shrinks)."""
    n = len(pts)
    out = []
    for i in range(n):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]

        def nrm(a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            l = hypot(dx, dy) or 1.0
            return (dy / l, -dx / l)

        n0, n1 = nrm(p0, p1), nrm(p1, p2)
        k = max(1.0 + n0[0] * n1[0] + n0[1] * n1[1], 0.22)
        out.append((p1[0] + (n0[0] + n1[0]) / k * d, p1[1] + (n0[1] + n1[1]) / k * d))
    return out


def tri_mesh(name, tris, uvs, zc=0.0, back=0.005):
    """Glass panes from triangles [(p, q, r)] of (x, y, z): faces towards -Y plus a reversed copy behind.
    uvs[i] = one (u, v) for the whole pane (a single clean piece of glass) or None = quarry lattice."""
    verts, faces, vuv = [], [], []
    for (a, b, c), uv in zip(tris, uvs):
        if (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0]) < 0:
            b, c = c, b
        i = len(verts)
        for q in (a, b, c):
            verts.append(q)
            vuv.append(uv if uv else (q[0], q[2] - zc))
        for q in (a, b, c):
            verts.append((q[0], q[1] + back, q[2]))
            vuv.append(uv if uv else (q[0], q[2] - zc))
        faces += [(i, i + 1, i + 2), (i + 5, i + 4, i + 3)]
    o = mesh_object(name, verts, faces)
    _uv_per_vertex(o, vuv)
    return o


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return hypot(px - ax - t * dx, py - ay - t * dy)


def split_run(a, b, target, r, jit=0.2):
    """Split [a, b] into pieces of roughly `target` length with random variation -> [(start, end)]."""
    L = b - a
    n = max(1, int(round(L / target)))
    ws = [r.u(1.0 - jit, 1.0 + jit) for _ in range(n)]
    s = sum(ws)
    out, pos = [], a
    for w in ws:
        l = L * w / s
        out.append((pos, pos + l))
        pos += l
    return out


_HIDE = {"b": (0, 0, -1), "t": (0, 0, 1), "k": (0, 1, 0), "f": (0, -1, 0), "l": (-1, 0, 0), "r": (1, 0, 0)}


def _chip_corners(o, size, ch, key, hide="", prob=0.24):
    """Knock a few exposed corners off a dressed block: the little triangle the chamfers leave at a corner is
    dragged out along the three arrises into a spall of a different length on each (old frost and cart damage).
    Corners bedded against a hidden face are left alone. Hash-driven, so no random stream is touched."""
    if ch < 0.008:
        return
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    kmax = 0.3 * min(size) / ch
    if kmax < 1.3:
        return
    lim = ch * 1.6
    ci = 0
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                ci += 1
                if any(Vector(_HIDE[h]).dot(Vector((sx, sy, sz))) > 0.5 for h in hide):
                    continue
                if _h1(key[0] + ci * 1.37, key[1] - ci * 0.61, key[2] + ci * 2.11) > prob:
                    continue
                kk = [min(kmax, 1.5 + 3.2 * _h1(key[1] + ci + a * 3.3, key[0] - a, key[2])) for a in range(3)]
                cx, cy, cz = sx * hx, sy * hy, sz * hz
                for v in o.data.vertices:
                    dx, dy, dz = v.co.x - cx, v.co.y - cy, v.co.z - cz
                    if abs(dx) < lim and abs(dy) < lim and abs(dz) < lim:
                        v.co.x, v.co.y, v.co.z = cx + dx * kk[0], cy + dy * kk[1], cz + dz * kk[2]


def blk(name, size, loc, r, ch=0.014, hide="", dens=0.0, rough=0.003, chip=0.24):
    """One dressed stone: chamfered arrises, faces a little out of true, faces that bed against other
    stones removed (hide: b bottom, t top, k back +Y, f front, l -X, r +X), tooling direction per stone."""
    o = box(name, size, bevel=min(ch, 0.2 * min(size)), segments=1)
    if hide:
        me = o.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        hv = [Vector(_HIDE[h]) for h in hide]
        kill = [f for f in bm.faces if any(f.normal.dot(d) > 0.98 for d in hv)]
        bmesh.ops.delete(bm, geom=kill, context="FACES")
        bm.to_mesh(me)
        bm.free()
    if chip > 0:
        _chip_corners(o, size, min(ch, 0.2 * min(size)), loc, hide, chip)
    if dens > 0:
        densify(o, dens)
    if rough > 0:
        displace_noise(o, strength=rough, scale=max(size) * 0.7, seed=int(r.u(0, 9999)))
    set_loc_rot(o, loc)
    smooth(o, 40)
    uv_box(o, rot90=r.chance(0.5))
    g_, w_ = _stone_tone(loc[0], loc[2], loc[1])
    _vmask(o, "grime", [g_] * len(o.data.vertices))
    _vmask(o, "wear", [w_] * len(o.data.vertices))
    return o


def ashlar(name, org, udir, L, z0, z1, r, course=0.3, blen=0.62, openings=(), sub=(0.12, 0.16), inset=0.012, proud=0.012, chip=0.16):
    """Coursed ashlar facing built stone by stone: every block is a slightly proud face with chamfered
    arrises meeting its neighbours in a V joint (running bond, random lengths and course heights).
    org = (x, y) of the start of the joint plane, udir = unit direction along the wall; the face looks to
    the right of the direction of travel. openings = [(u0, u1, z0, z1)] are left out (courses line up with
    them). sub = max cell size of the block faces so weathering masks have something to bite on."""
    ox, oy = org
    ux, uy = udir
    nx, ny = uy, -ux
    cuts = [z0, z1]
    for op in openings:
        for z in (op[2], op[3]):
            if all(abs(z - c) > 0.1 for c in cuts) and z0 < z < z1:
                cuts.append(z)
    cuts.sort()
    zs = [cuts[0]]
    for a, b in zip(cuts[:-1], cuts[1:]):
        for (c0, c1) in split_run(a, b, course, r, 0.12):
            zs.append(c1)
    verts, vuz, faces, fmap = [], [], [], {}
    vgr, vwr = [], []
    tone = [0.0, 0.0]

    def add(u, z, d):
        verts.append((ox + ux * u + nx * d, oy + uy * u + ny * d, z))
        vuz.append((u, z))
        vgr.append(tone[0] if d > 0 else 0.92)     # the joint plane: damp, dirt-filled pointing
        vwr.append(tone[1] if d > 0 else 0.0)
        return len(verts) - 1

    for ci in range(len(zs) - 1):
        za, zb = zs[ci], zs[ci + 1]
        zm = (za + zb) / 2
        iv = [(0.0, L)]
        for (u0, u1, zo0, zo1) in openings:
            if zo0 - 0.05 < zm < zo1 + 0.05:
                nv = []
                for (a, b) in iv:
                    if u1 <= a or u0 >= b:
                        nv.append((a, b))
                    else:
                        if u0 - a > 0.04:
                            nv.append((a, u0))
                        if b - u1 > 0.04:
                            nv.append((u1, b))
                iv = nv
        for (a, b) in iv:
            if b - a < blen * 1.3:
                bs = [a, b]
            else:
                runs = split_run(a, b, blen, r, 0.2)
                bs = [a] + [q[1] for q in runs[:-1]] + [b]
                if ci % 2:
                    bs = [a] + [(bs[i] + bs[i + 1]) / 2 + r.u(-0.04, 0.04) for i in range(len(bs) - 1)] + [b]
            for (ba, bb) in zip(bs[:-1], bs[1:]):
                pr = proud + r.u(0.0, 0.004)
                tone[0], tone[1] = _stone_tone(ox + ux * (ba + bb) / 2, za, oy + uy * (ba + bb) / 2)
                ins = min(inset, 0.3 * (bb - ba), 0.3 * (zb - za))
                nu = max(1, int(ceil((bb - ba - 2 * ins) / sub[0])))
                nz = max(1, int(ceil((zb - za - 2 * ins) / sub[1])))
                us = [ba + ins + (bb - ba - 2 * ins) * i / nu for i in range(nu + 1)]
                zz = [za + ins + (zb - za - 2 * ins) * j / nz for j in range(nz + 1)]

                def fv(i, j):                  # face vertex; a corner now and then is spalled back into the joint
                    u_, z_, d_ = us[i], zz[j], pr
                    if i in (0, nu) and j in (0, nz):
                        wx_, wy_ = ox + ux * u_, oy + uy * u_
                        if _h1(wx_ * 3.1 + z_ * 7.7, z_ * 3.3 - wy_, wy_ + wx_) < chip:
                            k_ = 0.35 + 0.65 * _h1(u_, z_, 1.3)
                            cu_ = min(0.05 * k_, 0.45 * (us[1] - us[0]))
                            cz_ = min(0.045 * _h1(z_, u_, 2.9) + 0.012, 0.45 * (zz[1] - zz[0]))
                            u_ += cu_ if i == 0 else -cu_
                            z_ += cz_ if j == 0 else -cz_
                            d_ = pr * 0.15
                    return add(u_, z_, d_)

                g = [[fv(i, j) for i in range(nu + 1)] for j in range(nz + 1)]
                uo = [ba] + us[1:-1] + [bb]
                zo = [za] + zz[1:-1] + [zb]
                ob = [add(uo[i], za, 0.0) for i in range(nu + 1)]
                ot = [add(uo[i], zb, 0.0) for i in range(nu + 1)]
                ol = [ob[0]] + [add(ba, zo[j], 0.0) for j in range(1, nz)] + [ot[0]]
                orr = [ob[nu]] + [add(bb, zo[j], 0.0) for j in range(1, nz)] + [ot[nu]]
                fs = []
                for j in range(nz):
                    for i in range(nu):
                        fs.append((g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i]))
                for i in range(nu):
                    fs.append((ob[i], ob[i + 1], g[0][i + 1], g[0][i]))
                    fs.append((g[nz][i], g[nz][i + 1], ot[i + 1], ot[i]))
                for j in range(nz):
                    fs.append((g[j][nu], orr[j], orr[j + 1], g[j + 1][nu]))
                    fs.append((ol[j], g[j][0], g[j + 1][0], ol[j + 1]))
                du_, dv_, swp = r.u(0, 5), r.u(0, 5), r.chance(0.5)
                for f in fs:
                    faces.append(f)
                    fmap[frozenset(f)] = {v: ((vuz[v][1] + dv_, vuz[v][0] + du_) if swp else (vuz[v][0] + du_, vuz[v][1] + dv_)) for v in f}
    o = mesh_object(name, verts, faces)
    _set_uv(o, fmap)
    smooth(o, 30)
    _vmask(o, "grime", vgr)
    _vmask(o, "wear", vwr)
    return o


def plain_wall(name, org, udir, L, z0, z1, du=0.3, dz=0.2):
    """Plain vertical stone face (looks to the right of the direction of travel)."""
    ox, oy = org
    ux, uy = udir
    nu, nz = max(1, int(ceil(L / du))), max(1, int(ceil((z1 - z0) / dz)))
    verts = [(ox + ux * L * i / nu, oy + uy * L * i / nu, z0 + (z1 - z0) * j / nz) for j in range(nz + 1) for i in range(nu + 1)]
    faces = [(j * (nu + 1) + i, j * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i + 1, (j + 1) * (nu + 1) + i) for j in range(nz) for i in range(nu)]
    o = mesh_object(name, verts, faces)
    uv_box(o)
    return o


def cavity(name, x0, x1, yf, yb, z0, z1):
    """Inside of a rectangular wall opening (jambs, soffit, back), open to the front and below."""
    v = [(x0, yf, z0), (x1, yf, z0), (x1, yb, z0), (x0, yb, z0), (x0, yf, z1), (x1, yf, z1), (x1, yb, z1), (x0, yb, z1)]
    f = [(3, 2, 6, 7), (0, 3, 7, 4), (2, 1, 5, 6), (4, 7, 6, 5)]
    o = mesh_object(name, v, f)
    uv_box(o)
    return o


# ------------------------------------------------------------------------------------------------- build
def build(p, rng):
    W, H = float(p.width), float(p.height)
    npt, nst, tiers = int(p.star_points), int(p.steps), int(p.tiers)
    pal = PALETTES.get(p.palette, PALETTES["amber"])
    seed = int(p.seed)
    mossv = float(p.moss)
    wet = bool(p.water)

    ST = mat("stone_smooth", p.stone)
    # the inscription slab: gilt on warm limestone under warm lanterns has no contrast, so the letters are cut in a dark
    # honed slab (as on memorials); dark bronze letters without leaf stay on the pale stone
    TABM = mat("stone_smooth", p.tablet, rough=0.38) if float(p.gilding) > 0.02 else ST
    IRON = mat("bronze_aged")                 # all the metalwork is dark bronze: rails, lantern cages, letter infill
    gv = float(p.gilding)
    GA = mat("glass_stained", pal["a"], glow=STAR_GLOW, grime=0.0)          # sunlit facets of the star lantern
    GB = mat("glass_stained", pal["b"], glow=0.8 * STAR_GLOW, grime=0.0)    # facets in shade, foot lantern
    GOLD = mat("metal_plain", "#d4a84b", rough=0.42, glow=round(0.8 * gv, 3))   # leaf gilding: crown, letters (a breath of
    #                                                                           glow stands in for the leaf catching the lanterns)
    LANT = mat("glass_lantern", "#ffa94e", glow=1.15, opacity=0.27)
    FLAME = mat("emissive", "#ffc27a", glow=7.0)
    WAT = mat("water", "#182c2c", ripple=0.5, depth=0.3, grime=0.0)
    WFL = mat("water_fall", "#eef7ff", glow=0.14, opacity=0.7, grime=0.0)
    MOSS = mat("moss")
    stones, irons = [], []

    # ---- governing dimensions
    T = 0.5                                   # wing wall thickness (back face on y = 0)
    sw = clamp(min(0.53 * W, 0.60 * H), 3.0, 5.0)   # shrine width at the pylon tops
    cw = sw + 0.36                            # central plinth width
    wh = clamp(0.33 * H, 1.9, 3.1)            # wing wall height
    zp = wh + 0.22                            # plinth top
    rise = 0.15
    zl = nst * rise                           # landing level
    ypf = -1.15                               # plinth front face
    yl = ypf - 0.55                           # landing nosing
    going = min(0.36, (3.2 + yl) / max(nst - 1, 1))
    ysf = yl - going * (nst - 1)              # bottom nosing
    ck = 0.34                                 # cheek wall thickness
    swi = cw - 2 * ck                         # clear stair width
    wt, cop = 0.22, 0.12                      # basin wall thickness, coping thickness
    ub = clamp(0.44 * wh, 0.85, 1.3)          # raised tank height
    lb = 0.45                                 # ground basin height
    wl_u, wl_l = ub - 0.024, lb - 0.05        # water levels: both tanks stand brim-full
    sill = ub - 0.05                          # weir sill of the raised tank
    y_uf = -1.65                              # raised tank front face
    y_lf = -2.75 if tiers == 2 else -2.05     # ground basin front face
    z_sc = wh - 0.52                          # scupper lip level
    span0, span1 = cw / 2 + 0.1, W / 2 - 0.62
    xs = (span0 + span1) / 2                  # scupper centre (|x|)
    ws = clamp((span1 - span0) * 0.72, 0.28, 1.3)   # scupper width
    xp = W / 2 - 0.32                         # end pier centre
    pier_h = wh + 0.3

    zb = zp + 0.18                            # pylon base
    fin = 0.3
    if p.pediment == "round":
        prise = min(0.34 * sw, 0.23 * (H - zb))
    else:
        prise = min(0.15 * sw, 0.16 * (H - zb))
    z_et = H - fin - prise - 0.14             # entablature top
    z_eb = z_et - 0.42                        # entablature bottom = pylon top
    hp = z_eb - zb
    R = min(0.81 * (sw - 0.24 * sw) / 2, (hp - 0.22) / 2)   # outer radius of the round surround
    xi = R / 0.81                             # pylon inner face: the star fills ~0.8 of the niche
    pwt = sw / 2 - xi                         # pylon width at the top
    Ri = R - 0.15                             # glass opening radius
    zc = z_eb - 0.1 - R                       # star centre height
    batter = 0.085 * hp
    dp, ycp = 0.94, -0.51                     # pylon depth, centre y
    yfp = ycp - dp / 2                        # pylon front face
    yg = -0.51                                # glass plane
    ye = -1.06                                # entablature face

    r_f = rng.fork("falls")
    impacts_all = []                          # (|x|, y, water level) of every fall, for the spatter staining
    r_m = rng.fork("moss")

    # ------------------------------------------------------------------------------------ moss cushions
    mv, mf, muv = [], [], []

    def moss_patch(pos, nrm, rad, along=None, aspect=1.0):
        """Low ragged cushion of moss hugging a stone face (rim sunk into the joint so it has no hard edge)."""
        n = Vector(nrm).normalized()
        t1 = Vector(along) if along is not None else n.cross(Vector((0, 0, 1)))
        if t1.length < 1e-4:
            t1 = Vector((1, 0, 0))
        t1 = (t1 - n * t1.dot(n)).normalized()
        t2 = n.cross(t1)
        c = Vector(pos)
        hgt = min(0.026, rad * 0.32) * r_m.u(0.6, 1.2)
        b = len(mv)
        ns = 8
        a0 = r_m.u(0, 6.28)
        uo, vo = r_m.u(0, 3), r_m.u(0, 3)
        loc = [(0.0, 0.0, hgt)]
        for ring, (rf, hf) in enumerate(((0.5, 0.8), (1.0, 0.0))):
            for k in range(ns):
                a = a0 + 2 * pi * k / ns + (0.25 if ring else 0.0)
                rr = rad * rf * r_m.u(0.65, 1.3)
                loc.append((rr * cos(a) * aspect, rr * sin(a), hgt * hf * r_m.u(0.7, 1.1) - (0.004 if ring else 0.0)))
        for (a_, b_, h_) in loc:
            q = c + t1 * a_ + t2 * b_ + n * h_
            mv.append((q.x, max(q.y, -3.49), max(q.z, 0.003)))
            muv.append((a_ + uo, b_ + vo))
        for k in range(ns):
            k1 = (k + 1) % ns
            mf.append((b, b + 1 + k, b + 1 + k1))
            mf.append((b + 1 + k, b + 1 + ns + k, b + 1 + ns + k1, b + 1 + k1))

    def moss_line(p0, p1, nrm, density, rad=(0.035, 0.08), aspect=(1.4, 2.6), spread=0.0, along=None):
        """Scatter cushions along a joint / waterline from p0 to p1 (density = cushions per metre at moss 1)."""
        a, b = Vector(p0), Vector(p1)
        L = (b - a).length
        if L < 1e-4:
            return
        d = (b - a) / L
        n = Vector(nrm).normalized()
        side = n.cross(d)
        cnt = L * density * mossv
        cnt = int(cnt) + (1 if r_m.u() < cnt - int(cnt) else 0)
        for _ in range(cnt):
            q = a + d * (L * r_m.u(0.03, 0.97)) + side * r_m.u(-spread, spread)
            moss_patch(q, n, r_m.u(*rad), along=along if along is not None else d, aspect=r_m.u(*aspect))

    # ------------------------------------------------------------------------------------------- water
    # water_fall UV convention (see the engine shader): U = metres across the fall, V = along the flow, up from
    # the foot and paced by the speed of the water (the pattern scrolls down V at 2.9 units/s, so V advances by
    # ds * 2.9 / max(speed, 1.2)). The weathering masks carry two extra hints for the shader: wear = break-up
    # (0 glassy lip .. 0.45 first rents .. 0.75 ropes .. 1 threads and drops), special = foam (1 = white water).
    def water_mesh(name, V, F, UV, WR, SP):
        o = mesh_object(name, V, F)
        _uv_per_vertex(o, UV)
        smooth(o, 60)
        assign(o, WFL)
        _vmask(o, "wear", WR)
        _vmask(o, "special", SP)
        return o

    def cascade(name, x, w, y_in, y_edge, z_top, z_bot, v0, r):
        """Water running over a lip and falling in a parabola as ONE thin continuous sheet: a glassy film on
        the lip, a clear curtain that necks in under surface tension and ripples as it accelerates, and only
        low down a few V-shaped rents opening towards the foot. The material tears the foot into ropes and
        drops (wear) - no drops are modelled, so nothing hangs still in the air."""
        tf = sqrt(max(z_top - z_bot, 0.05) / 4.9)
        hk = clamp(sqrt(max(z_top - z_bot, 0.0) / 0.8), 0.5, 1.0)       # a short fall never gets to break up
        lf = y_in - y_edge
        kv = 2.9 / 1.2                         # V per metre of slow water (see the shader: V is time-like)
        V, F, UV, WR, SP = [], [], [], [], []
        uo = r.u(0, 7)

        def pt(s, xo, dy=0.0):
            t = s * tf
            return (x + xo, y_edge - v0 * t + dy, z_top - 4.9 * t * t)

        def alen(s):                           # V along the flow: path length paced by the speed of the water
            t, acc, n = s * tf, lf * kv, 10
            for i in range(n):
                ta, tb = t * i / n, t * (i + 1) / n
                acc += hypot(v0 * (tb - ta), 4.9 * (tb * tb - ta * ta)) * 2.9 / max(hypot(v0, 4.9 * (ta + tb)), 1.2)
            return acc

        ltot = alen(1.05)
        nt = max(1, int(round(w / 0.17)))
        # rents: (centre -0.5..0.5, height where it opens, half gap at the foot); the two outer edges never close
        cuts = [(-0.5, 0.0, 0.0)] + [((i + 1 + r.u(-0.28, 0.28)) / nt - 0.5, r.u(0.4, 0.8), r.u(0.01, 0.032)) for i in range(nt - 1)] + [(0.5, 0.0, 0.0)]
        ph1, ph2, ph3, ph4 = r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28)
        srow = (0.0, 0.04, 0.09, 0.15, 0.22, 0.3, 0.38, 0.46, 0.54, 0.62, 0.7, 0.77, 0.84, 0.9, 0.95, 1.0)

        def xoff(c, s):                        # the sheet necks in and sways a little as it falls
            return c * w * (1.0 - 0.13 * s ** 1.4) + 0.006 * s * sin(c * 9.0 + s * 5.0 + ph1)

        def bulge(c, s):                       # ripples running down the curtain (they carry the highlights)
            return (0.004 * sin(c * w * 38.0 + ph2) + 0.003 * sin(s * 21.0 + c * 6.0 + ph3)) * smoothstep(0.03, 0.4, s)

        for bi in range(len(cuts) - 1):
            cl, sl, gl = cuts[bi]
            cr, sr, gr = cuts[bi + 1]
            ncol = max(2, int(round((cr - cl) * w / 0.035)))
            s_end = r.u(0.99, 1.05)
            b = len(V)
            for (yy, zz_, ul) in ((y_in, z_top - 0.004, 0.0), (y_edge + 0.03, z_top, lf - 0.03)):   # film on the lip
                for j in range(ncol + 1):
                    c = cl + (cr - cl) * j / ncol
                    V.append((x + c * w, yy, zz_))
                    UV.append((c * w + uo, ltot - ul * kv))
                    WR.append(0.02)
                    SP.append(0.0)
            for s0_ in srow:
                s = s0_ * s_end
                va = ltot - alen(s)
                ol, orr = smoothstep(sl, 1.0, s) if gl else 0.0, smoothstep(sr, 1.0, s) if gr else 0.0
                xl, xr = xoff(cl, s) + gl * ol ** 0.8, xoff(cr, s) - gr * orr ** 0.8
                for j in range(ncol + 1):
                    f = j / ncol
                    c = cl + (cr - cl) * f
                    xx = xl + (xr - xl) * f
                    V.append(pt(s, xx, bulge(c, s)))
                    UV.append((xx + uo, va))
                    # edge of a rent / of the sheet: thicker, more broken water
                    ed = max((ol if gl else 0.5 * s) * clamp(1.0 - f * ncol / 1.5, 0.0, 1.0), (orr if gr else 0.5 * s) * clamp(1.0 - (1.0 - f) * ncol / 1.5, 0.0, 1.0))
                    sc = clamp(s, 0.0, 1.0)
                    cw = 0.09 * sin(c * w * 23.0 + ph4) + 0.05 * sin(c * w * 61.0 + ph2)
                    WR.append(clamp(0.02 + hk * (0.7 * sc ** 1.6 + cw * sc) + (0.08 + 0.3 * sc) * ed * (0.3 + 0.7 * sc) + 0.16 * smoothstep(0.86, 1.0, sc), 0.0, 1.0))
                    SP.append(clamp(0.6 * smoothstep(0.87, 1.0, sc) + 0.12 * ed * smoothstep(0.5, 1.0, sc), 0.0, 1.0))
            nrw = len(srow) + 2
            for k in range(nrw - 1):
                for j in range(ncol):
                    a = b + k * (ncol + 1) + j
                    F.append((a, a + ncol + 1, a + ncol + 2, a + 1))
        water_mesh(name, V, F, UV, WR, SP)
        return y_edge - v0 * tf

    def splash(name, x, y, w, z, r, bounds, amp=0.012):
        """Where a fall hits the pool: a low boil of froth along the line of impact, one ragged band of spray
        thrown up round it and a few ribbons of drops flung clear (the material tears the band and the ribbons
        into drops travelling along them: V runs DOWN every throw). The flat foam and the rings on the surface
        are not modelled: they are painted on the water (water_surface) and drawn by the `water` material."""
        bx0, bx1, by0, by1 = bounds
        V, F, UV, WR, SP = [], [], [], [], []
        uo, vo = r.u(0, 5), r.u(0, 3)
        k_ = amp / 0.012

        def add(px, py, pz, u, v, sp=1.0, wr=0.55):
            px, py = clamp(px, bx0 + 0.012, bx1 - 0.012), clamp(py, by0 + 0.012, by1 - 0.012)
            V.append((px, py, pz))
            UV.append((u + uo, v + vo))
            WR.append(wr)
            SP.append(sp)
            return len(V) - 1

        ea, eb = w / 2 + 0.02, 0.055
        per = 4 * ea + 2 * eb
        nseg = max(14, int(per / 0.06))
        # boil: a low mound of froth, whitest on its crown, torn to nothing at its edge
        c0 = add(x, y, z + 0.02 * k_, 0.0, 0.0, 0.6, 0.7)
        rings = []
        for (rf, dz, sp_, wr_) in ((0.45, 0.017, 0.6, 0.76), (0.8, 0.01, 0.5, 0.84), (1.0, -0.004, 0.2, 0.96)):
            rings.append([add(x + ea * rf * r.u(0.94, 1.06) * cos(2 * pi * k / nseg), y + eb * rf * r.u(0.92, 1.08) * sin(2 * pi * k / nseg),
                              z + (dz * k_ * r.u(0.85, 1.1) if dz > 0 else dz), ea * rf * cos(2 * pi * k / nseg), -0.45 * rf, sp_, wr_) for k in range(nseg)])
        for k in range(nseg):
            k1 = (k + 1) % nseg
            F.append((c0, rings[0][k], rings[0][k1]))
            for a_, b_ in zip(rings[:-1], rings[1:]):
                F.append((a_[k], b_[k], b_[k1], a_[k1]))
        # crown: one ragged band of spray leaning outwards all round the landing
        h = r.u(0.06, 0.1) * k_
        p1, p2, p3 = r.u(0, 6.28), r.u(0, 6.28), r.u(0, 6.28)
        rows = ((0.0, 0.0, 0.93, 0.4), (0.1, 0.4, 0.96, 0.2), (0.24, 0.78, 0.98, 0.08), (0.4, 1.0, 1.0, 0.0))
        b = len(V)
        for i in range(nseg + 1):
            th = 2 * pi * i / nseg
            kh = 0.72 + 0.3 * sin(3 * th + p1) * sin(2 * th + p2) + 0.16 * sin(7 * th + p3)
            for (fr, fz, wr_, sp_) in rows:
                off = 0.5 * eb + 0.09 * fr * kh * k_
                add(x + (ea * 0.94 + off) * cos(th), y + (eb * 0.5 + off) * sin(th), z - 0.005 + h * fz * kh,
                    2.5 * per * i / nseg, 1.7 - 9.0 * h * kh * (fz + 0.5 * fr), sp_, wr_)
        nr = len(rows)
        for i in range(nseg):
            for q in range(nr - 1):
                i0 = b + i * nr + q
                F.append((i0, i0 + nr, i0 + nr + 1, i0 + 1))
        # drops flung clear: thin ribbons along short ballistic arcs
        K = 6
        for _ in range(int(per / 0.3) + 2):
            a = r.u(0, 2 * pi)
            ca, sa = cos(a), sin(a)
            px0, py0 = x + ea * 0.9 * ca, y + eb * 0.9 * sa
            da = atan2(sa * 2.2, ca) + r.u(-0.4, 0.4)             # mostly thrown across the line of impact
            ca, sa = cos(da), sin(da)
            sp0, el = r.u(0.8, 1.4) * sqrt(k_), r.u(0.9, 1.25)
            vr, vz = sp0 * cos(el), sp0 * sin(el)
            tfl = 2 * vz / 9.8 * r.u(0.7, 0.96)
            wd, tilt = r.u(0.008, 0.014) * k_, r.u(-0.7, 0.7)
            u_, vv = r.u(0, 3), r.u(0, 3)
            b = len(V)
            prev = None
            for q in range(K + 1):
                t = tfl * q / K
                pr, pz = vr * t, vz * t - 4.9 * t * t
                if prev is not None:
                    vv -= hypot(pr - prev[0], pz - prev[1]) * 2.9 / max(hypot(vr, vz - 9.8 * t), 0.8)
                prev = (pr, pz)
                hw = 0.5 * wd * (0.55 + 0.9 * q / K)
                for e in (-1.0, 1.0):
                    add(px0 + pr * ca - sa * hw * e, py0 + pr * sa + ca * hw * e, z + 0.006 + pz + tilt * hw * e,
                        u_ + hw * e, vv, 0.15 * (1.0 - q / K) ** 2, 0.93 + 0.07 * q / K)
            for q in range(K):
                i0 = b + 2 * q
                F.append((i0, i0 + 1, i0 + 3, i0 + 2))
        water_mesh(name, V, F, UV, WR, SP)

    def water_surface(name, cx, cy, sx_, sy_, z, impacts):
        """Flat pool surface. Its ripples, the rings running out from the falls and the foam are drawn by the
        `water` material from two painted masks: wear = agitation (1 on the line of impact, falling linearly to
        0 over 1.2 m - the rings run down its gradient), special = foam (white water round the impact, lacy
        a hand further out)."""
        g = grid(name, (sx_, sy_), cuts=(max(6, int(sx_ / 0.04)), max(6, int(sy_ / 0.04))), loc=(cx, cy, z))
        co = np.array([v.co[:] for v in g.data.vertices], dtype=np.float64)
        wx, wy = co[:, 0] + cx, co[:, 1] + cy
        ag, fo = np.zeros(len(co)), np.zeros(len(co))
        for (ix, iy, hw, amp) in impacts:
            d = np.hypot(np.maximum(np.abs(wx - ix) - hw, 0.0), wy - iy)
            k = clamp(amp / 0.012, 0.5, 1.0)
            rf = 0.1 * k
            ag = np.maximum(ag, np.clip(k - d / 1.2, 0.0, 1.0))
            fo = np.maximum(fo, np.clip(1.0 - (d - 0.55 * rf) / (1.25 * rf + 0.03), 0.0, 1.0) ** 1.4)
        smooth(g, 70)
        assign(g, WAT)
        _vmask(g, "wear", ag)
        _vmask(g, "special", fo)
        return g

    # ----------------------------------------------------------------------------------------- lantern
    def lantern(name, loc, s, cd):
        """Square tapering lantern: turned stem, iron cage with glazing bars and seeded glass, hipped
        roof with a vent cap, a burner and a visible flame inside."""
        x, y, z = loc
        ps = [box(name + "_foot", (s * 0.95, s * 0.95, 0.02), loc=(x, y, z + 0.01), bevel=0.004, segments=1),
              lathe(name + "_stem", [(0.34 * s, 0.0), (0.34 * s, 0.012), (0.16 * s, 0.03), (0.13 * s, 0.07), (0.24 * s, 0.085), (0.24 * s, 0.1), (0, 0.1)],
                    segments=8, loc=(x, y, z + 0.02))]
        z0 = z + 0.12
        h = 1.25 * s
        wb, wt_ = 0.36 * s, 0.5 * s
        ps.append(box(name + "_tray", (2 * wb + 0.03, 2 * wb + 0.03, 0.02), loc=(x, y, z0), bevel=0.003, segments=1))
        ps.append(box(name + "_head", (2 * wt_ + 0.03, 2 * wt_ + 0.03, 0.022), loc=(x, y, z0 + h), bevel=0.003, segments=1))
        sq = [(-0.008, -0.008), (0.008, -0.008), (0.008, 0.008), (-0.008, 0.008)]
        for ax in (-1, 1):
            for ay in (-1, 1):
                ps.append(sweep(name + "_post", [(x + ax * wb, y + ay * wb, z0), (x + ax * wt_, y + ay * wt_, z0 + h)], sq, smooth_path=False))
        hk = 0.62
        wk = wb + (wt_ - wb) * hk
        for (ax, ay) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tx, ty = -ay, ax
            ps.append(sweep(name + "_bar", [(x + ax * (wb + 0.002), y + ay * (wb + 0.002), z0), (x + ax * (wt_ + 0.002), y + ay * (wt_ + 0.002), z0 + h)],
                            0.0045, sides=4, smooth_path=False))
            ps.append(sweep(name + "_bar", [(x + ax * (wk + 0.002) - tx * wk, y + ay * (wk + 0.002) - ty * wk, z0 + h * hk),
                                            (x + ax * (wk + 0.002) + tx * wk, y + ay * (wk + 0.002) + ty * wk, z0 + h * hk)], 0.0045, sides=4, smooth_path=False))
        r0 = (wt_ + 0.05) * 1.414
        zr = z0 + h + 0.011
        ps.append(lathe(name + "_roof", [(0, 0), (r0, 0), (r0, 0.012), (r0 * 0.72, s * 0.1), (r0 * 0.44, s * 0.24), (r0 * 0.27, s * 0.36), (0.0, s * 0.36)],
                        segments=4, loc=(x, y, zr), rot=(0, 0, 45), smooth_angle=20))
        ps.append(lathe(name + "_vent", [(0.11 * s, 0), (0.11 * s, 0.12 * s), (0.18 * s, 0.13 * s), (0.18 * s, 0.16 * s), (0.05 * s, 0.2 * s), (0.035 * s, 0.25 * s),
                                         (0.075 * s, 0.3 * s), (0.055 * s, 0.36 * s), (0, 0.38 * s)], segments=8, loc=(x, y, zr + s * 0.33)))
        ps.append(lathe(name + "_burner", [(0, 0), (0.1 * s, 0), (0.1 * s, 0.035), (0.045 * s, 0.05), (0.045 * s, 0.09), (0, 0.09)], segments=8, loc=(x, y, z0 + 0.01)))
        for o in ps:
            assign(o, IRON)
        irons.extend(ps)
        gv, gf = [], []
        for (ax, ay) in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            tx, ty = -ay, ax
            b = len(gv)
            a_, c_ = wb - 0.003, wt_ - 0.003
            gv += [(x + ax * a_ - tx * a_, y + ay * a_ - ty * a_, z0 + 0.01), (x + ax * a_ + tx * a_, y + ay * a_ + ty * a_, z0 + 0.01),
                   (x + ax * c_ + tx * c_, y + ay * c_ + ty * c_, z0 + h - 0.01), (x + ax * c_ - tx * c_, y + ay * c_ - ty * c_, z0 + h - 0.01)]
            gf.append((b, b + 1, b + 2, b + 3))
        gl = mesh_object(name + "_glass", gv, gf)
        uv_box(gl)
        assign(gl, LANT)
        fl = lathe(name + "_flame", [(0, 0), (0.075 * s, 0.05 * s), (0.105 * s, 0.14 * s), (0.08 * s, 0.26 * s), (0.032 * s, 0.38 * s), (0, 0.45 * s)],
                   segments=8, loc=(x, y, z0 + 0.1))
        assign(fl, FLAME)
        light(name, (x, y, z0 + 0.1 + 0.2 * s), intensity=cd, range=7.0 if cd < 18 else 8.5, flicker=0.15, shadow_near=s * 1.7)

    # ================================================================= wings: wall, scupper, pier, basins
    r_w = rng.fork("wings")
    xw0, xw1 = cw / 2, W / 2 - 0.3
    notch_abs = []
    nw_ = 0.34
    if tiers == 2:
        in0, in1 = cw / 2 + wt + 0.28, (W / 2 - 0.06) - wt - 0.28
        nn = 1 if in1 - in0 < 0.6 else (2 if in1 - in0 < 1.7 else 3)
        notch_abs = [(in0 + in1) / 2] if nn == 1 else [in0 + (in1 - in0) * k / (nn - 1) for k in range(nn)]
    for sx in (-1, 1):
        sd = "l" if sx < 0 else "r"
        xa, xb = (cw / 2, W / 2 - 0.06) if sx > 0 else (-(W / 2 - 0.06), -cw / 2)
        z0w = (wl_u - 0.32) if tiers == 2 else 0.03
        org_x = xw0 if sx > 0 else -xw1
        u_sc = (sx * xs - ws / 2) - org_x
        stones.append(ashlar("wing_wall", (org_x, -T), (1, 0), xw1 - xw0, z0w, wh - 0.14, r_w, course=0.3, blen=0.66,
                             openings=[(u_sc, u_sc + ws, z_sc, z_sc + 0.2)], sub=(0.06, 0.12)))
        stones.append(cavity("scupper_cavity", sx * xs - ws / 2, sx * xs + ws / 2, -T, -T + 0.32, z_sc - 0.01, z_sc + 0.2))
        for (a, b) in split_run(xw0 - 0.05, xw1, 0.8, r_w):
            stones.append(blk("wing_coping", (b - a - 0.006, T + 0.07, 0.14), (sx * (a + b) / 2, -(T + 0.07) / 2, wh - 0.07 + r_w.u(0, 0.004)), r_w,
                              ch=0.02, hide="b", dens=0.25))
        x0s, x1s = sorted((sx * (cw / 2 + 0.02), sx * (xp - 0.27)))
        stones.append(prism_x("string_course", [(-T + 0.02, wh - 0.3), (-T - 0.035, wh - 0.3), (-T - 0.05, wh - 0.27), (-T - 0.05, wh - 0.22), (-T - 0.02, wh - 0.19),
                                                (-T - 0.02, wh - 0.14), (-T + 0.02, wh - 0.14)], x0s, x1s))
        # scupper: thin spout slab with an undercut drip nose, side kerbs, corbels and a drip label
        lipw = ws + 0.12
        lip = prism_x("scupper_lip", [(-T + 0.26, z_sc), (-T - 0.285, z_sc), (-T - 0.3, z_sc - 0.012), (-T - 0.3, z_sc - 0.032), (-T - 0.275, z_sc - 0.048),
                                      (-T - 0.05, z_sc - 0.09), (-T + 0.26, z_sc - 0.09)], sx * xs - lipw / 2, sx * xs + lipw / 2)
        stones.append(lip)
        for e in (-1, 1):
            stones.append(box("scupper_kerb", (0.04, 0.3, 0.035), loc=(sx * xs + e * (ws / 2 + 0.025), -T - 0.145, z_sc + 0.016), bevel=0.008, segments=1))
            xc = sx * xs + e * (ws / 2 - 0.05)
            c = prism_x("corbel", [(-T + 0.02, z_sc - 0.06), (-T - 0.2, z_sc - 0.06), (-T - 0.2, z_sc - 0.1), (-T - 0.06, z_sc - 0.3), (-T + 0.02, z_sc - 0.3)],
                        xc - 0.05, xc + 0.05)
            bevel(c, 0.008)
            stones.append(c)
            xr = sx * xs + e * (ws / 2 + 0.3)
            if span0 + 0.08 < abs(xr) < span1 - 0.02:
                stones.append(lathe("wing_rosette", [(0, 0), (0.1, 0), (0.108, 0.012), (0.085, 0.03), (0.06, 0.022), (0.035, 0.045), (0, 0.055)], segments=12,
                                    loc=(xr, -T, z_sc + 0.1), rot=(90, 0, 0)))
        stones.append(blk("scupper_label", (ws + 0.3, 0.11, 0.07), (sx * xs, -T - 0.045, z_sc + 0.245), r_w, ch=0.02, hide="k"))
        # end pier of coursed blocks with a moulded cap and a lantern
        z = 0.0
        hs = [r_w.u(0.85, 1.15) for _ in range(max(3, int(round(pier_h / 0.44))))]
        for hh in hs:
            hh = hh * pier_h / sum(hs)
            stones.append(blk("pier", (0.58 + r_w.u(-0.006, 0.006), 0.58, hh - 0.005), (sx * xp, -0.33, z + hh / 2), r_w, ch=0.016, hide="btk", dens=0.2))
            z += hh
        stones.append(lathe("pier_cap", [(0, 0), (0.43, 0), (0.43, 0.04), (0.405, 0.058), (0.405, 0.085), (0.445, 0.098), (0.445, 0.128), (0.3, 0.172),
                                         (0.25, 0.172), (0.25, 0.2), (0.0, 0.2)], segments=4, loc=(sx * xp, -0.33, pier_h), rot=(0, 0, 45), smooth_angle=20))
        lantern("pier_lamp_%s" % sd, (sx * xp, -0.35, pier_h + 0.2), 0.3, 18.0)

        # ---- basins
        def basin(tag, y_back, y_front, ztop, wlevel, floor_z, notches=(), z_out0=0.0):
            zt_ = ztop - cop
            stones.append(ashlar(tag + "_front", (xa, y_front), (1, 0), xb - xa, z_out0, zt_, r_w, course=0.32, blen=0.7, sub=(0.06, 0.12)))
            stones.append(ashlar(tag + "_end", (xb, y_front), (0, 1), y_back - y_front, 0.0, zt_, r_w, course=0.32, blen=0.7, sub=(0.2, 0.2)))
            stones.append(ashlar(tag + "_end", (xa, y_back), (0, -1), y_back - y_front, 0.0, zt_, r_w, course=0.32, blen=0.7, sub=(0.2, 0.2)))
            Li, Di = xb - xa - 2 * wt, y_back - y_front - wt
            for (og, ud, L_) in (((xb - wt, y_front + wt), (-1, 0), Li), ((xa + wt, y_front + wt), (0, 1), Di), ((xb - wt, y_back), (0, -1), Di)):
                stones.append(plain_wall(tag + "_inner", og, ud, L_, floor_z, zt_ + 0.01, du=0.25, dz=0.1))
            cx, cy = (xa + xb) / 2, (y_front + wt + y_back) / 2
            fl = grid(tag + "_floor", (Li + 0.02, Di + 0.02), cuts=(5, 4), loc=(cx, cy, floor_z))
            stones.append(fl)
            lz = wlevel - 0.04                 # footing course just under the water line
            if lz - floor_z > 0.03:
                for (bx_, by_, sx_, sy_) in ((cx, y_front + wt + 0.055, Li, 0.11), (cx, y_back - 0.055, Li, 0.11), (xa + wt + 0.055, cy, 0.11, Di),
                                             (xb - wt - 0.055, cy, 0.11, Di)):
                    fo_ = box(tag + "_footing", (sx_, sy_, lz - floor_z), loc=(bx_, by_, (lz + floor_z) / 2), bevel=0.012, segments=1)
                    stones.append(fo_)
            edges = [xa - 0.03] + [v for (nx, nw) in sorted(notches) for v in (nx - nw / 2, nx + nw / 2)] + [xb + 0.03]
            for i in range(0, len(edges), 2):
                for (a, b) in split_run(edges[i], edges[i + 1], 0.8, r_w):
                    stones.append(blk(tag + "_coping", (b - a - 0.006, wt + 0.08, cop), ((a + b) / 2, y_front + wt / 2, ztop - cop / 2 + r_w.u(0, 0.004)), r_w,
                                      ch=0.022, hide="b", dens=0.22))
            for xx in (xa + wt / 2, xb - wt / 2):
                for (a, b) in split_run(y_front + wt + 0.045, y_back, 0.8, r_w):
                    stones.append(blk(tag + "_coping", (wt + 0.08, b - a - 0.006, cop), (xx, (a + b) / 2, ztop - cop / 2 + r_w.u(0, 0.004)), r_w,
                                      ch=0.022, hide="b", dens=0.22))
            for (nx, nw) in notches:
                hw_ = sill - (ztop - cop)
                stones.append(blk(tag + "_weir", (nw - 0.012, wt + 0.22, hw_), (nx, y_front + wt / 2 - 0.11, ztop - cop + hw_ / 2), r_w, ch=0.01))
            # moss: coping joints and inner arris, waterline, foot of the wall on the paving
            moss_line((xa + wt, y_front + wt + 0.03, ztop), (xb - wt, y_front + wt + 0.03, ztop), (0, 0, 1), 5.0, spread=0.04)
            moss_line((xa, y_front + 0.05, ztop), (xb, y_front + 0.05, ztop), (0, 0, 1), 2.0, rad=(0.03, 0.07), spread=0.04)
            for xx in (xa + wt + 0.03, xb - wt - 0.03):
                moss_line((xx, y_front + wt, ztop), (xx, y_back, ztop), (0, 0, 1), 4.0, spread=0.04)
            zw = wlevel + 0.035
            moss_line((xa + wt, y_front + wt, zw), (xb - wt, y_front + wt, zw), (0, 1, 0), 6.0, spread=0.03)
            moss_line((xa + wt, y_front + wt, zw), (xa + wt, y_back, zw), (1, 0, 0), 6.0, spread=0.03)
            moss_line((xb - wt, y_front + wt, zw), (xb - wt, y_back, zw), (-1, 0, 0), 6.0, spread=0.03)
            moss_line((xa + wt, y_back - 0.012, zw), (xb - wt, y_back - 0.012, zw), (0, -1, 0), 6.0, spread=0.03)
            if z_out0 < 0.01:
                moss_line((xa, y_front - 0.012, 0.012), (xb, y_front - 0.012, 0.012), (0, -0.75, 0.66), 3.5, rad=(0.03, 0.08))
                xo = xb if sx > 0 else xa
                moss_line((xo + sx * 0.012, y_front, 0.012), (xo + sx * 0.012, y_back, 0.012), (sx * 0.75, 0, 0.66), 3.5, rad=(0.03, 0.08))
            if not wet:                        # drained: moss and litter creep over the floor
                for _ in range(int(Li * Di * 9 * mossv)):
                    moss_patch((cx + r_m.u(-0.46, 0.46) * Li, cy + r_m.u(-0.46, 0.46) * Di, floor_z + 0.002), (0, 0, 1), r_m.u(0.05, 0.14), aspect=r_m.u(1.0, 2.0))
            return (cx, cy, Li, Di, (xa + wt, xb - wt, y_front + wt, y_back))

        notches = [(sx * v, nw_) for v in notch_abs]
        fw_ = ws - 0.05
        if tiers == 2:
            bu = basin("tank_u", -T, y_uf, ub, wl_u, wl_u - 0.3, notches, z_out0=lb - 0.2)
            bl = basin("tank_l", y_uf, y_lf, lb, wl_l, 0.05)
            if wet:
                yi = cascade("fall_top", sx * xs, fw_, -T + 0.22, -T - 0.3, z_sc + 0.016, wl_u, 0.6, r_f)
                splash("splash_top", sx * xs, yi, fw_ * 0.86, wl_u, r_f, bu[4])
                impacts_all.append((xs, yi, wl_u))
                imp = []
                for (nx, nw) in notches:
                    yj = cascade("fall_low", nx, nw - 0.06, y_uf + wt + 0.04, y_uf - 0.22, sill + 0.014, wl_l, 0.45, r_f)
                    splash("splash_low", nx, yj, (nw - 0.06) * 0.86, wl_l, r_f, bl[4], 0.01)
                    imp.append((nx, yj, (nw - 0.06) * 0.4, 0.01))
                    impacts_all.append((abs(nx), yj, wl_l))
                water_surface("water_u", bu[0], bu[1], bu[2] + 0.02, bu[3] + 0.02, wl_u, [(sx * xs, yi, fw_ * 0.4, 0.012)])
                water_surface("water_l", bl[0], bl[1], bl[2] + 0.02, bl[3] + 0.02, wl_l, imp)
            for (nx, nw) in notches:           # algae runs under the weirs
                for e in (-1, 1):
                    moss_line((nx + e * nw * 0.45, y_uf - 0.012, ub - cop - 0.03), (nx + e * nw * 0.45, y_uf - 0.012, lb + 0.05), (0, -1, 0), 5.0,
                              rad=(0.022, 0.045), aspect=(2.5, 4.5), spread=0.04)
        else:
            bl = basin("tank_l", -T, y_lf, lb, wl_l, 0.05)
            if wet:
                yi = cascade("fall_top", sx * xs, fw_, -T + 0.22, -T - 0.3, z_sc + 0.016, wl_l, 0.6, r_f)
                splash("splash_top", sx * xs, yi, fw_ * 0.86, wl_l, r_f, bl[4])
                impacts_all.append((xs, yi, wl_l))
                water_surface("water_l", bl[0], bl[1], bl[2] + 0.02, bl[3] + 0.02, wl_l, [(sx * xs, yi, fw_ * 0.4, 0.012)])
        # algae runs down the wall either side of the scupper, moss on the wall head
        zlow = (wl_u if tiers == 2 else wl_l) + 0.08
        for e in (-1, 1):
            moss_line((sx * xs + e * (ws / 2 + 0.05), -T - 0.014, z_sc - 0.1), (sx * xs + e * (ws / 2 + 0.05), -T - 0.014, zlow), (0, -1, 0), 4.5,
                      rad=(0.025, 0.05), aspect=(2.5, 4.5), spread=0.06)
        moss_line((sx * xw0, -0.1, wh + 0.002), (sx * xw1, -0.1, wh + 0.002), (0, 0, 1), 2.5, spread=0.07)

    # ================================================================================== central plinth
    r_c = rng.fork("plinth")
    zf0, zf1 = zl + 0.34, zp - 0.2
    hq = max(0.45, (zf1 - zf0) - 0.32)         # inscription field
    wq = min(0.68 * cw, swi - 0.4)
    zq = (zf0 + zf1) / 2
    ow_ = wq / 2 + 0.075                       # half width of the opening left for the inscription stone
    stones.append(ashlar("plinth_front", (-cw / 2, ypf), (1, 0), cw, zl - 0.02, zp - 0.2, r_c, course=0.34, blen=0.8,
                         openings=[(cw / 2 - ow_, cw / 2 + ow_, zq - hq / 2 - 0.075, zq + hq / 2 + 0.075)], sub=(0.06, 0.14)))
    stones.append(ashlar("plinth_side", (-cw / 2, 0.0), (0, -1), -ypf, 0.3, zp - 0.2, r_c, course=0.34, blen=0.7, sub=(0.2, 0.2)))
    stones.append(ashlar("plinth_side", (cw / 2, ypf), (0, 1), -ypf, 0.3, zp - 0.2, r_c, course=0.34, blen=0.7, sub=(0.2, 0.2)))
    stones.append(blk("plinth_top", (cw, -ypf, 0.03), (0, ypf / 2, zp - 0.016), r_c, ch=0.004, hide="bk", rough=0.0))
    def subdiv(path, step=0.45):
        out = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            n = max(1, int(ceil(hypot(b[0] - a[0], b[1] - a[1]) / step)))
            out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
        return out

    upath = subdiv([(-cw / 2, -0.001), (-cw / 2, -0.1), (-cw / 2, ypf), (cw / 2, ypf), (cw / 2, -0.1), (cw / 2, -0.001)])
    z0 = zl - 0.01
    pb = run_mould("plinth_base", subdiv([(-swi / 2, ypf), (swi / 2, ypf)]), [(-0.02, z0), (0.07, z0), (0.07, z0 + 0.2), (0.055, z0 + 0.23), (0.04, z0 + 0.25), (0.04, z0 + 0.27),
                                          (0.02, z0 + 0.3), (0.0, z0 + 0.32), (-0.02, z0 + 0.32)])
    z1 = zp - 0.2
    pc = run_mould("plinth_cornice", upath, [(-0.02, z1), (0.0, z1), (0.015, z1 + 0.03), (0.015, z1 + 0.05), (0.05, z1 + 0.1), (0.085, z1 + 0.125),
                                             (0.085, z1 + 0.16), (0.11, z1 + 0.2), (-0.02, z1 + 0.2)])
    for o in (pb, pc):
        bevel(o, 0.005)
        stones.append(o)

    # inscription stone: one big honed slab let into the plinth, bolection frame, incised lead-filled letters
    yfc = ypf + 0.004                          # face of the slab
    tab = box("inscription_stone", (wq, 0.05, hq), loc=(0, yfc + 0.025, zq))
    densify(tab, 0.13)
    assign(tab, TABM)
    # bronze fixing bosses at the corners of the field
    bosses = [lathe("tablet_boss", [(0, 0), (0.036, 0), (0.038, 0.006), (0.03, 0.012), (0.022, 0.011), (0.014, 0.02), (0, 0.023)], segments=10,
                    loc=(ex * (wq / 2 - 0.075), yfc, zq + ez * (hq / 2 - 0.075)), rot=(90, 0, 0)) for ex in (-1, 1) for ez in (-1, 1)]
    stones.append(frame_mould("inscription_frame", -wq / 2, wq / 2, zq - hq / 2, zq + hq / 2,
                              [(0.0, yfc + 0.01), (0.0, yfc - 0.014), (0.012, yfc - 0.032), (0.03, yfc - 0.042), (0.05, yfc - 0.042), (0.064, yfc - 0.03),
                               (0.072, yfc - 0.012), (0.085, yfc - 0.012), (0.085, yfc + 0.01)]))
    lines = [s.strip() for s in str(p.inscription).split("|") if s.strip()][:4] or [" "]
    if hq - 0.14 < 0.62 and len(lines) > 1:    # a low plinth takes the inscription as one long line
        lines = [" ".join(lines)]

    def text_w(o):
        xsv = [v.co.x for v in o.data.vertices] or [0.0]
        return max((max(xsv) - min(xsv)) * abs(o.scale[0]), 1e-3)

    def text_h(o):
        ysv = [v.co.y for v in o.data.vertices] or [0.0]
        return max(ysv) - min(ysv), (max(ysv) + min(ysv)) / 2

    probe = [text("probe", ln, size=0.2, depth=0.01, font="roman", spacing=1.06) for ln in lines]
    wmax = max(text_w(o) for o in probe)
    delete(probe)
    pr = text("probe", "H", size=0.2, depth=0.01, font="roman")
    cap02, mid02 = text_h(pr)                  # cap height and its centre above the text origin at size 0.2
    delete(pr)
    cap02 = max(cap02, 0.02)
    wi, hi = wq - 0.34, hq - 0.14
    dtx = str(p.date).strip()
    es = clamp(hi / 1.0, 0.5, 1.0)             # emblem scale
    zbot = zq - hi / 2
    zcur = zbot + 0.015
    ds = zd = 0.0
    if dtx and hi > 0.8:                       # date line at the foot
        pr = text("probe", dtx, size=0.2, depth=0.01, font="roman", spacing=1.15)
        dcap = min(0.085 * es, cap02 * 0.62 * wi / text_w(pr))
        delete(pr)
        ds = 0.2 * dcap / cap02
        zd = zcur + dcap / 2
        zcur += dcap + 0.06 * es
    emblem = hi > 0.6                          # crown between two rules
    zem = zcur + 0.09 * es
    if emblem:
        zcur = zem + 0.09 * es + 0.07 * es
    z_lo, z_hi = zcur, zq + hi / 2 - 0.03
    nl = len(lines)
    caps = min((z_hi - z_lo) / (1.3 + 1.9 * (nl - 1)), cap02 * 0.92 * wi / wmax, 0.24)
    ts = 0.2 * caps / cap02
    pitch = 1.9 * caps
    ztop = (z_lo + z_hi) / 2 + caps * (1.0 + 1.9 * (nl - 1)) / 2 + 0.1 * caps
    cut_d = 0.017
    cutters = []
    for i, ln in enumerate(lines):
        cutters.append(text("inscription", ln, size=ts, depth=0.03, loc=(0, yfc + cut_d, ztop - caps / 2 - i * pitch - mid02 * ts / 0.2),
                            font="roman", spacing=1.06))
    if ds:
        cutters.append(text("date", dtx, size=ds, depth=0.03, loc=(0, yfc + cut_d, zd - mid02 * ds / 0.2), font="roman", spacing=1.15))
    for c in cutters:
        weld(c, 0.0012)
        apply_modifiers(c)
    crown = [(es * a, es * b) for (a, b) in ((-0.1, -0.06), (0.1, -0.06), (0.125, 0.06), (0.055, -0.005), (0.0, 0.09), (-0.055, -0.005), (-0.125, 0.06))]
    if emblem:
        cutters.append(shape_extrude("crown", crown, depth=0.03, loc=(0, yfc + cut_d, zem), rot=(90, 0, 0)))
        for e in (-1, 1):
            cutters.append(box("rule", (wi * 0.24, 0.03, 0.01), loc=(e * (0.19 * es + wi * 0.12), yfc + cut_d - 0.015, zem)))
            cutters.append(extrude_profile("rule_dot", [(0.022, 0), (0, 0.022), (-0.022, 0), (0, -0.022)], 0.03,
                                           loc=(e * (0.19 * es + wi * 0.24 + 0.05), yfc + cut_d, zem), rot=(90, 0, 0)))
    # the tablet is laid in long honed courses: fine V joints run between the lines of lettering
    zj = [ztop - caps / 2 - i * pitch - pitch / 2 for i in range(nl - 1)]
    if emblem:
        zj.append((ztop - caps * 1.3 - (nl - 1) * pitch + zem + 0.09 * es) / 2)
    zj = sorted(z for z in zj if zq - hq / 2 + 0.08 < z < zq + hq / 2 - 0.08)
    for z in zj:
        try:
            boolean(tab, box("tablet_joint", (wq + 0.1, 0.01, 0.01), loc=(0, yfc, z), rot=(45, 0, 0)))
        except Exception:
            pass
    raised = []
    for c in cutters:
        try:
            boolean(tab, c)
        except Exception:
            raised.append(c)
    for c in raised:                           # fallback: thin bronze letters proud of the stone
        set_loc_rot(c, (c.location.x, yfc + 0.027, c.location.z))
        assign(c, IRON)
    # the whole cut is gilded - floor and walls - so some face of every letter catches a lantern from any side
    assign_faces(tab, GOLD if gv > 0.02 else IRON,
                 lambda c, n: (n.y < -0.9 and c.y > -0.022) or (abs(n.y) < 0.35 and -0.0245 < c.y < -0.004 and abs(c.x) < wq / 2 - 0.004 and abs(c.z) < hq / 2 - 0.004))
    smooth(tab, 30)
    stones.append(tab)

    # ================================================================== steps, cheeks, newels, handrails
    r_s = rng.fork("steps")
    for i in range(nst - 1):
        yn = ysf + i * going
        for (a, b) in split_run(-swi / 2, swi / 2, 1.05 + 0.2 * (i % 2), r_s):
            stones.append(blk("step", (b - a - 0.006, going + 0.05, rise), ((a + b) / 2, yn + (going + 0.05) / 2, (i + 0.5) * rise + r_s.u(0, 0.004)), r_s,
                              ch=0.016, hide="bk", dens=0.3))
        for sx in (-1, 1):                     # moss creeping out of the corner against the cheek
            if r_m.u() < 0.35 + mossv:
                moss_line((sx * (swi / 2 - 0.02), yn + 0.05, (i + 1) * rise + 0.002), (sx * (swi / 2 - 0.02), yn + going, (i + 1) * rise + 0.002),
                          (-sx * 0.5, 0, 0.86), 3.0, rad=(0.03, 0.06))
    for (a, b) in split_run(-swi / 2, swi / 2, 0.95, r_s):
        stones.append(blk("landing", (b - a - 0.006, (ypf + 0.04) - yl, rise), ((a + b) / 2, (yl + ypf + 0.04) / 2, (nst - 0.5) * rise + r_s.u(0, 0.004)), r_s,
                          ch=0.016, hide="bk", dens=0.3))
    slope = rise / going
    zt = zl + 0.4
    yfoot = ysf + 0.1
    zfoot = zt - slope * (yl - yfoot)
    for sx in (-1, 1):
        xc = sx * (cw / 2 - ck / 2)
        # cheek wall and its coping are built of stones of believable length (same outline as one big block)
        def c_top(y):
            return zt if y >= yl else zt - slope * (yl - y)

        def c_up(y):                           # top of the coping
            return zt + 0.07 if y >= yl - 0.012 else zt + 0.07 - (zt - zfoot) * (yl - 0.012 - y) / (yl - 0.012 - yfoot)

        def stone_piece(nm, poly, x0_, x1_, bv):
            o_ = prism_x(nm, poly, x0_, x1_)
            bevel(o_, bv)
            cy_ = sum(q[0] for q in poly) / len(poly)
            cz_ = sum(q[1] for q in poly) / len(poly)
            g_, w_ = _stone_tone(xc, cz_, cy_)
            _vmask(o_, "grime", [g_] * len(o_.data.vertices))
            _vmask(o_, "wear", [w_] * len(o_.data.vertices))
            stones.append(o_)

        jg = 0.003
        ycs = [yfoot] + [b_ for (a_, b_) in split_run(yfoot, ypf + 0.02, 0.6, r_s, 0.15)]
        zbed = 0.48 * zt + r_s.u(-0.03, 0.03)
        for a_, b_ in zip(ycs[:-1], ycs[1:]):
            ya_, yb2 = a_ + (jg if a_ > yfoot else 0.0), b_ - (jg if b_ < ypf + 0.019 else 0.0)
            knee = [(yl, zt)] if ya_ + 0.02 < yl < yb2 - 0.02 else []
            if min(c_top(ya_), c_top(yb2)) > zbed + 0.12:      # tall enough for two courses
                stone_piece("cheek", [(yb2, 0.0), (yb2, zbed - jg), (ya_, zbed - jg), (ya_, 0.0)], xc - ck / 2, xc + ck / 2, 0.008)
                stone_piece("cheek", [(yb2, zbed + jg), (yb2, c_top(yb2))] + knee + [(ya_, c_top(ya_)), (ya_, zbed + jg)], xc - ck / 2, xc + ck / 2, 0.008)
            else:
                stone_piece("cheek", [(yb2, 0.0), (yb2, c_top(yb2))] + knee + [(ya_, c_top(ya_)), (ya_, 0.0)], xc - ck / 2, xc + ck / 2, 0.008)
        ykn = yl - 0.22                        # the kneeler stone carries the bend
        stone_piece("cheek_coping", [(ypf + 0.02, zt), (ypf + 0.02, zt + 0.07), (yl - 0.012, zt + 0.07), (ykn + jg, c_up(ykn + jg)), (ykn + jg, c_top(ykn + jg)), (yl, zt)],
                    xc - ck / 2 - 0.03, xc + ck / 2 + 0.03, 0.012)
        for (a_, b_) in split_run(yfoot, ykn, 0.62, r_s, 0.15):
            ya_, yb2 = a_ + (jg if a_ > yfoot else 0.0), b_ - jg
            stone_piece("cheek_coping", [(yb2, c_top(yb2)), (yb2, c_up(yb2)), (ya_, c_up(ya_)), (ya_, c_top(ya_))], xc - ck / 2 - 0.03, xc + ck / 2 + 0.03, 0.012)
        stones.append(blk("newel", (0.44, 0.44, 0.74), (xc, ysf, 0.37), r_s, ch=0.02, hide="bt", dens=0.2))
        stones.append(lathe("newel_cap", [(0, 0), (0.34, 0), (0.34, 0.03), (0.32, 0.045), (0.355, 0.06), (0.355, 0.085), (0.25, 0.115), (0.2, 0.115), (0.2, 0.135), (0, 0.135)],
                            segments=4, loc=(xc, ysf, 0.74), rot=(0, 0, 45), smooth_angle=20))
        lantern("step_lamp_%s" % ("l" if sx < 0 else "r"), (xc, ysf, 0.875), 0.24, 14.0)
        moss_line((xc - 0.2, ysf - 0.232, 0.012), (xc + 0.2, ysf - 0.232, 0.012), (0, -0.75, 0.66), 3.0, rad=(0.025, 0.06))
        if p.handrail:
            hr = 0.52
            ya, yb_ = ypf - 0.06, ysf + 0.3
            zb_ = zt + 0.07 - slope * (yl - yb_)
            path = [(xc, ya, zt + 0.07 + hr), (xc, yl, zt + 0.07 + hr), (xc, yb_, zb_ + hr)]
            irons.append(sweep("rail", path, 0.017, sides=8, smooth_path=False))
            irons.append(sweep("rail_low", [(a, b, c - 0.26) for a, b, c in path], 0.009, sides=6, smooth_path=False))
            pys = [ya, yl] + [yl + (yb_ - yl) * k / 2 for k in (1, 2)]
            for py in pys:
                zc_ = zt + 0.07 - slope * max(yl - py, 0.0)
                irons.append(box("rail_post", (0.022, 0.022, hr + 0.03), loc=(xc, py, zc_ + hr / 2 - 0.005), bevel=0.003, segments=1))
                irons.append(cylinder("rail_shoe", 0.026, 0.014, loc=(xc, py, zc_ + 0.006), verts=8))
            irons.append(sphere("rail_knob", 0.032, loc=(xc, yb_, zb_ + hr + 0.012), segments=10, rings=6))
            irons.append(sphere("rail_knob", 0.026, loc=(xc, ya, zt + 0.07 + hr + 0.008), segments=10, rings=6))

    # ============================================================================================ shrine
    r_p = rng.fork("pylons")
    xbase = sw / 2 + batter + 0.07
    for (a, b) in split_run(-xbase, xbase, 1.1, r_p):
        stones.append(blk("shrine_base", (b - a - 0.006, 1.12, 0.18), ((a + b) / 2, -0.58, zp + 0.09 + r_p.u(0, 0.003)), r_p, ch=0.02, hide="bk", dens=0.3))
    bslope = batter / hp
    for sx in (-1, 1):
        hs = [r_p.u(0.84, 1.16) for _ in range(max(3, int(round(hp / 0.43))))]
        z = zb
        for hh in hs:
            hh = hh * hp / sum(hs)
            zm = z + hh / 2
            wdt = sw / 2 + batter * (1 - (zm - zb) / hp) - xi
            b = blk("pylon", (wdt - 0.004, dp + r_p.u(-0.008, 0.008), hh - 0.005), (sx * (xi + wdt / 2), ycp, zm), r_p, ch=0.016, hide="btk", dens=0.11)
            transform_verts(b, lambda co, sx=sx: Vector((co.x - sx * bslope * co.z, co.y, co.z)) if co.x * sx > 0 else co)
            stones.append(b)
            z += hh
        # fluted pilaster with attic base and Doric cap on the pylon front
        xpc = sx * (xi + pwt / 2)
        fw, pd = clamp(0.76 * pwt, 0.28, 0.62), 0.07
        nfl = 3 if fw < 0.36 else (4 if fw < 0.5 else 5)
        rf = fw / (2 * nfl + 0.7 * (nfl + 1))
        g = 0.7 * rf
        capH, baseH = 0.27, 0.24
        pts = [(-fw / 2, 0.0), (-fw / 2, -pd)]
        for k in range(nfl):
            cxk = -fw / 2 + g + rf + k * (2 * rf + g)
            pts += [(cxk + rf * cos(radians(a)), -pd + 0.85 * rf * sin(radians(a))) for a in (180, 150, 120, 90, 60, 30, 0)]
        pts += [(fw / 2, -pd), (fw / 2, 0.0)]
        k0 = z_eb - capH
        nsec = max(2, int(ceil((k0 - zb - baseH) / 0.3)))
        sh = _loft("pilaster", [[(xpc + a_, yfp + b_, zb + baseH + (k0 - zb - baseH) * q / nsec) for (a_, b_) in pts] for q in range(nsec + 1)], caps=False)
        smooth(sh, 35)
        stones.append(sh)
        pp = [(xpc - fw / 2, yfp + 0.01), (xpc - fw / 2, yfp - pd), (xpc + fw / 2, yfp - pd), (xpc + fw / 2, yfp + 0.01)]
        base = [(-0.012, zb), (0.065, zb), (0.065, zb + 0.075)]
        base += [(0.037 + 0.03 * cos(radians(a)), zb + 0.105 + 0.03 * sin(radians(a))) for a in (-60, -30, 0, 30, 60, 90)]
        base += [(0.026, zb + 0.14), (0.02, zb + 0.155), (0.022, zb + 0.172)]
        base += [(0.02 + 0.018 * cos(radians(a)), zb + 0.19 + 0.018 * sin(radians(a))) for a in (-60, -20, 20, 60, 90)]
        base += [(0.01, zb + 0.212), (0.01, zb + 0.225), (0.0, zb + baseH), (-0.012, zb + baseH)]
        o = run_mould("pilaster_base", pp, base)
        smooth(o, 35)
        stones.append(o)
        cap = [(-0.012, k0), (0.0, k0)]
        cap += [(0.014 * cos(radians(a)), k0 + 0.016 + 0.014 * sin(radians(a))) for a in (-60, -20, 20, 60)]
        cap += [(0.0, k0 + 0.032), (0.0, k0 + 0.085), (0.008, k0 + 0.09), (0.008, k0 + 0.1), (0.016, k0 + 0.105), (0.016, k0 + 0.116)]
        cap += [(0.02 + 0.045 * sin(radians(a)), k0 + 0.185 - 0.065 * cos(radians(a))) for a in (0, 25, 50, 70, 90)]
        cap += [(0.078, k0 + 0.19), (0.078, k0 + 0.245), (0.086, k0 + 0.25), (0.086, k0 + capH), (-0.012, k0 + capH)]
        o = run_mould("pilaster_cap", pp, cap)
        smooth(o, 35)
        stones.append(o)

    # entablature: architrave blocks + mitred cornice with dentils
    xe = sw / 2 + 0.03
    for (a, b) in ((-xe, -xi + 0.1), (-xi + 0.1, xi - 0.1), (xi - 0.1, xe)):
        stones.append(blk("architrave", (b - a - 0.006, -ye - 0.02, 0.24), ((a + b) / 2, (ye - 0.02) / 2, z_eb + 0.12), r_p, ch=0.014, hide="tk", dens=0.12))
    c0 = z_eb + 0.24
    epath = subdiv([(-xe, -0.02), (-xe, -0.12), (-xe, ye), (xe, ye), (xe, -0.12), (xe, -0.02)], 0.6)
    stones.append(run_mould("cornice", epath, [(-0.2, c0), (0.0, c0), (0.02, c0 + 0.03), (0.02, c0 + 0.05), (0.07, c0 + 0.09), (0.10, c0 + 0.11),
                                               (0.10, c0 + 0.15), (0.13, c0 + 0.18), (-0.2, c0 + 0.18)]))
    nd = max(6, int(round(2 * xe / 0.16)))
    dent = box("dentil", (0.07, 0.05, 0.05), bevel=0.004, segments=1)
    stones.append(instances_to_mesh("dentils", dent, [trs((-xe + 0.06 + (2 * xe - 0.12) * k / (nd - 1), ye - 0.03, c0 + 0.06)) for k in range(nd)], keep_base=False))

    # curved pediment: tympanum, raking cornice, acroteria and finial
    cch = 2 * xe + 0.02
    Rp = (cch * cch / 4 + prise * prise) / (2 * prise)
    th = degrees(asin(min(1.0, (cch / 2) / Rp)))
    czp = z_et + prise - Rp
    na = 22
    arc = [(Rp * cos(radians(90 - th + 2 * th * k / na)), czp + Rp * sin(radians(90 - th + 2 * th * k / na))) for k in range(na + 1)]
    yt_ = ye + 0.1                             # tympanum face
    Rt, zt0 = Rp - 0.013, c0 + 0.181           # face mesh kept just clear of the cornices (no buried vertices)
    xm_ = sqrt(max(Rt * Rt - (zt0 + 0.012 - czp) ** 2, 0.01))
    ncol = max(8, int(ceil(2 * xm_ / 0.14)))
    nrow = 5
    tv = [(-xm_ + 2 * xm_ * i / ncol, yt_, zt0 + (czp + sqrt(Rt * Rt - (-xm_ + 2 * xm_ * i / ncol) ** 2) - zt0) * j / nrow)
          for i in range(ncol + 1) for j in range(nrow + 1)]
    tfc = [(i * (nrow + 1) + j, (i + 1) * (nrow + 1) + j, (i + 1) * (nrow + 1) + j + 1, i * (nrow + 1) + j + 1) for i in range(ncol) for j in range(nrow)]
    tym = mesh_object("tympanum", tv, tfc)
    uv_box(tym)
    stones.append(tym)
    stones.append(arc_mould("raking_cornice", 0.0, czp, 90 - th, 90 + th,
                            [(Rp - 0.03, yt_ + 0.02), (Rp, yt_ - 0.03), (Rp + 0.035, yt_ - 0.03), (Rp + 0.05, yt_ - 0.07), (Rp + 0.09, yt_ - 0.1), (Rp + 0.12, yt_ - 0.13),
                             (Rp + 0.14, yt_ - 0.13), (Rp + 0.14, -0.5), (Rp + 0.14, -0.14), (Rp + 0.14, -0.02), (Rp - 0.03, -0.02)], n=30))
    for sx in (-1, 1):
        stones.append(blk("acroterion", (0.3, 0.34, 0.36), (sx * (xe - 0.12), -0.54, z_et + 0.18), r_p, ch=0.016, hide="b"))
        stones.append(lathe("acroterion_cap", [(0, 0), (0.26, 0), (0.26, 0.035), (0.23, 0.055), (0.0, 0.2)], segments=4,
                            loc=(sx * (xe - 0.12), -0.54, z_et + 0.36), rot=(0, 0, 45), smooth_angle=20))
    ztop = H - fin
    stones.append(lathe("finial", [(0, 0), (0.15, 0), (0.15, 0.04), (0.08, 0.07), (0.055, 0.1), (0.1, 0.15), (0.125, 0.2), (0.105, 0.255), (0.05, 0.29), (0, 0.3)],
                        segments=16, loc=(0, -0.53, ztop - 0.012)))
    # carved star and paterae in the tympanum
    sr = min(0.32, prise * 0.36)
    zs = z_et + prise * 0.42
    sv, sf = [(0.0, yt_ - 0.05, zs)], []
    for k in range(2 * npt):
        a = pi / 2 + pi * k / npt
        rr = sr if k % 2 == 0 else sr * 0.45
        sv.append((rr * cos(a), yt_, zs + rr * sin(a)))
    for k in range(2 * npt):
        sf.append((0, 1 + k, 1 + (k + 1) % (2 * npt)))
    tstar = mesh_object("tympanum_star", sv, sf)
    recalc_normals(tstar)
    uv_box(tstar)
    stones.append(tstar)
    stones.append(lathe("tympanum_boss", [(sr * 1.2, 0), (sr * 1.28, 0.012), (sr * 1.36, 0.012), (sr * 1.42, 0.0)], segments=28, loc=(0, yt_, zs), rot=(90, 0, 0)))
    for sx in (-1, 1):
        xpt = sx * min(cch * 0.3, sr * 1.42 + 0.3 + 0.12 * cch)
        zarc = czp + sqrt(max(Rp * Rp - xpt * xpt, 0.0))
        if zarc - z_et > 0.34:
            stones.append(lathe("patera", [(0, 0), (0.1, 0), (0.108, 0.012), (0.085, 0.03), (0.06, 0.022), (0.035, 0.045), (0, 0.055)], segments=12,
                                loc=(xpt, yt_, z_et + (zarc - z_et) * 0.42), rot=(90, 0, 0)))

    # back panel with the round opening, ring of voussoirs, keystone, sill and apron
    ypan = -0.64
    zlo_, zhi_ = zb - zc, z_eb - zc
    angs = sorted([2 * pi * k / 72 for k in range(72)] + [atan2(zz_, xx_) % (2 * pi) for xx_ in (-xi, xi) for zz_ in (zlo_, zhi_)])
    npr = 5
    pv, pf = [], []
    for a in angs:
        ca, sa = cos(a), sin(a)
        t = xi / max(abs(ca), 1e-6)
        if sa > 1e-6:
            t = min(t, zhi_ / sa)
        elif sa < -1e-6:
            t = min(t, zlo_ / sa)
        r0_ = R + 0.003
        for j in range(npr + 1):
            rr = r0_ + max(t - r0_, 0.004) * j / npr
            pv.append((rr * ca, ypan, zc + rr * sa))
    na_ = len(angs)
    for i in range(na_):
        i1 = (i + 1) % na_
        for j in range(npr):
            pf.append((i * (npr + 1) + j, i * (npr + 1) + j + 1, i1 * (npr + 1) + j + 1, i1 * (npr + 1) + j))
    pan = mesh_object("back_panel", pv, pf)
    uv_box(pan)
    stones.append(pan)
    ring_prof = [(Ri, -0.42), (Ri, -0.585), (Ri + 0.022, -0.62), (Ri + 0.045, -0.62), (Ri + 0.058, -0.665), (Ri + 0.085, -0.7), (Ri + 0.115, -0.7),
                 (Ri + 0.13, -0.675), (R, -0.665), (R, -0.42)]
    nvs = 12 if R < 1.1 else 16
    gap = degrees(0.004 / R)
    for k in range(nvs):
        a0 = 90 + 180.0 / nvs + k * 360.0 / nvs
        stones.append(arc_mould("voussoir", 0.0, zc, a0 + gap, a0 + 360.0 / nvs - gap, ring_prof, n=5))
    ks = blk("keystone", (0.24, 0.135, 0.3), (0, -0.7125, zc + R - 0.03), r_p, ch=0.014, hide="k")
    transform_verts(ks, lambda co: Vector((co.x * (1.0 + 1.3 * co.z), co.y, co.z)))
    stones.append(ks)
    ha = zc - R - zb
    ow = 2 * xi
    stones.append(blk("sill", (ow - 0.01, 0.245, 0.08), (0, -0.7675, zc - R - 0.035), r_p, ch=0.016, hide="k", dens=0.12))
    for e in (-1, 1):
        cb = prism_x("sill_corbel", [(-0.645, zc - R - 0.075), (-0.87, zc - R - 0.075), (-0.87, zc - R - 0.11), (-0.7, zc - R - 0.24), (-0.645, zc - R - 0.24)],
                     e * ow * 0.3 - 0.06, e * ow * 0.3 + 0.06)
        bevel(cb, 0.008)
        stones.append(cb)
    if ha > 0.52:
        fwd, fht = ow * 0.4, ha - 0.36
        zfr = zb + (ha - 0.08) / 2
        stones.append(frame_mould("apron_frame", -fwd, fwd, zfr - fht / 2, zfr + fht / 2,
                                  [(0.0, -0.63), (0.0, -0.652), (0.015, -0.672), (0.04, -0.672), (0.055, -0.652), (0.055, -0.63)]))
    ros = []
    for ex in (-1, 1):
        for zz in (z_eb - 0.17, zb + 0.17):
            xx = ex * (xi - 0.17)
            if hypot(xx, zz - zc) > R + 0.12 and abs(zz - (zc - R)) > 0.2:
                ros.append(trs((xx, -0.64, zz), (90, 0, 0)))
    if ros:
        rb = lathe("rosette", [(0, 0), (0.085, 0), (0.092, 0.01), (0.07, 0.026), (0.05, 0.02), (0.03, 0.04), (0, 0.05)], segments=12)
        stones.append(instances_to_mesh("rosettes", rb, ros, keep_base=False))

    # ============================================ shell niche and the free-standing star lantern with its crown
    r_g = rng.fork("glass")
    dark = []
    # niche: plain reveal behind the voussoir ring, the back dished and cut with a sunburst of flutes that
    # catch the light of the star standing in front of it
    y_nb, nsg, nrr, nflu = -0.14, 96, 6, 16
    nv_ = [(0.0, y_nb + 0.075, zc)]
    for j in range(1, nrr + 1):
        q = j / nrr
        for i in range(nsg):
            a = 2 * pi * i / nsg
            fl_ = abs(((a * nflu / (2 * pi)) % 1.0) - 0.5) * 2.0          # 0 in the arris .. 1 in the hollow of a flute
            nv_.append((Ri * q * cos(a), y_nb + 0.075 * (1 - q * q) + 0.05 * q * (fl_ ** 0.7 - 0.5) * (1.0 if j < nrr else 0.0), zc + Ri * q * sin(a)))
    nf_ = [(0, 1 + i, 1 + (i + 1) % nsg) for i in range(nsg)]
    for j in range(nrr - 1):
        for i in range(nsg):
            a_, b_ = 1 + j * nsg + i, 1 + j * nsg + (i + 1) % nsg
            nf_.append((a_, a_ + nsg, b_ + nsg, b_))
    b0_ = len(nv_)
    nv_ += [(Ri * cos(2 * pi * i / nsg), -0.42, zc + Ri * sin(2 * pi * i / nsg)) for i in range(nsg)]
    o0_ = 1 + (nrr - 1) * nsg
    for i in range(nsg):
        i1 = (i + 1) % nsg
        nf_.append((b0_ + i, b0_ + i1, o0_ + i1, o0_ + i))
    nch = mesh_object("niche", nv_, nf_)
    uv_box(nch)
    smooth(nch, 50)
    stones.append(nch)
    bw_ = mesh_object("back_wall", [(-xi, -0.03, zb), (-xi, -0.03, z_eb), (xi, -0.03, z_eb), (xi, -0.03, zb)], [(0, 1, 2, 3)])
    uv_box(bw_)
    stones.append(bw_)

    # the star: a faceted double-skinned lantern of leaded glass (ridge to every point, valley to every re-entrant,
    # each facet glazed in three bands), on a glazed bronze foot standing on a stone pedestal, a gilt crown on top
    eb_ = 1.0 if npt % 2 == 0 else cos(pi / npt)           # how far the lowest point(s) hang below the centre
    z_s = zc - R + 0.005                                   # top of the sill
    zpd = z_s + 0.075                                      # top of the stone pedestal
    ro = min(0.9 * R, (z_eb - 0.04 - zpd) / (1.27 + eb_ + 0.3))
    zcs = z_eb - 0.04 - 1.27 * ro                          # centre: the crown just clears the architrave
    ri = ro * {4: 0.38, 5: 0.42, 6: 0.52, 7: 0.47, 8: 0.45}.get(npt, 0.45)
    tk = 0.14 * ro
    ys = -0.9                                              # plane of the star, proud of the niche
    half = pi / npt
    A = [pi / 2 + 2 * pi * k / npt for k in range(npt)]
    tips = [(ro * cos(a), ro * sin(a)) for a in A]
    inn = [(ri * cos(a + half), ri * sin(a + half)) for a in A]
    zft = zcs - eb_ * ro                                   # the lowest point(s) rest on the foot

    def S3(q, relief=0.0):
        return (q[0], ys - relief, zcs + q[1])

    def L3(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)

    def fw3(a, d=0.004):
        return (a[0], a[1] - d, a[2])

    def tone(t):
        return r_g.pick(PICKS[t])

    ta, ua, tb, ub_, cames, gilt = [], [], [], [], [], []
    bands = (0.0, 0.36, 0.69, 1.0)
    for (front, rel) in ((True, tk), (False, -0.6 * tk)):
        O3 = S3((0, 0), rel)
        for k in range(npt):
            Tp = S3(tips[k])
            for (I, lit) in ((S3(inn[k]), True), (S3(inn[k - 1]), False)):
                tl, ul = (ta, ua) if lit else (tb, ub_)
                if not front:                  # back skin: one pane per facet
                    tl.append((O3, Tp, I))
                    ul.append(tone("mid"))
                    continue
                for b_ in range(3):
                    a0, a1 = L3(O3, Tp, bands[b_]), L3(O3, Tp, bands[b_ + 1])
                    b0, b1 = L3(O3, I, bands[b_]), L3(O3, I, bands[b_ + 1])
                    uv = tone((("light", "mid", "mid") if lit else ("mid", "deep", "deep"))[b_])
                    if b_ == 0:
                        tl.append((O3, a1, b1))
                        ul.append(uv)
                    else:
                        tl += [(a0, a1, b1), (a0, b1, b0)]
                        ul += [uv, uv]
                        cames.append(sweep("came", [fw3(a0), fw3(b0)], 0.008, sides=4))
                cames.append(sweep("came", [fw3(Tp, 0.0), fw3(I, 0.0)], 0.017, sides=6))            # rim of the star
            cames.append(sweep("came", [fw3(O3), fw3(Tp)], 0.012, sides=5) if front else sweep("came", [O3, Tp], 0.008, sides=4))
            if front:
                cames.append(sweep("came", [fw3(O3), fw3(S3(inn[k]))], 0.01, sides=5))
                cames.append(sphere("came_boss", 0.026, loc=S3(tips[k]), segments=8, rings=4))
    gilt.append(sphere("star_boss", 0.045, loc=S3((0, 0), tk + 0.012), segments=10, rings=6))
    cames.append(sweep("star_stay", [(0, ys + 0.6 * tk - 0.01, zcs), (0, y_nb + 0.05, zcs)], 0.022, sides=6))   # tie rod back into the niche
    cames.append(lathe("stay_rose", [(0, 0), (0.07, 0), (0.075, 0.008), (0.05, 0.02), (0.03, 0.02), (0.0, 0.03)], segments=10,
                       loc=(0, y_nb + 0.072, zcs), rot=(90, 0, 0)))

    # glazed foot: a tapering bronze-framed lantern box the lowest points stand on
    hwb, hwt = 0.74 * ro, 0.62 * ro
    hdb, hdt = 0.13, 0.1
    z0f, z1f = zpd + 0.03, zft - 0.025
    ncf = max(3, int(round(2 * hwb / 0.34)))

    def FQ(u, v, side):                        # u -1..1 across, v 0..1 up; side: -1 front, +1 back
        hw_ = hwb + (hwt - hwb) * v
        return (u * hw_, ys + side * (hdb + (hdt - hdb) * v), z0f + (z1f - z0f) * v)

    for side in (-1, 1):
        for i in range(ncf):
            for j in range(2):
                u0, u1, v0_, v1_ = -1 + 2 * i / ncf, -1 + 2 * (i + 1) / ncf, j / 2, (j + 1) / 2
                q0, q1, q2, q3 = (FQ(u0, v0_, side * 0.93), FQ(u1, v0_, side * 0.93), FQ(u1, v1_, side * 0.93), FQ(u0, v1_, side * 0.93))
                uv = tone("mid" if (i + j) % 2 else "light")
                tl, ul = (ta, ua) if (i + j) % 2 == 0 else (tb, ub_)
                tl += [(q0, q1, q2), (q0, q2, q3)]
                ul += [uv, uv]
        for i in range(ncf + 1):
            u = -1 + 2 * i / ncf
            cames.append(sweep("foot_bar", [FQ(u, 0, side), FQ(u, 1, side)], 0.011 if 0 < i < ncf else 0.016, sides=4, smooth_path=False))
        cames.append(sweep("foot_bar", [FQ(-1, 0.5, side), FQ(1, 0.5, side)], 0.009, sides=4, smooth_path=False))
    for e in (-1, 1):                          # glazed ends
        q0, q1, q2, q3 = FQ(e * 0.985, 0, -1), FQ(e * 0.985, 0, 1), FQ(e * 0.985, 1, 1), FQ(e * 0.985, 1, -1)
        end = mesh_object("foot_end", [q0, q1, q2, q3], [(0, 1, 2, 3)])
        _uv_per_vertex(end, [tone("deep")] * 4)
        assign(end, GB)
    for (zz_, hw_, hd_, th_) in ((zpd + 0.015, hwb + 0.035, hdb + 0.035, 0.03), (z1f + 0.0125, hwt + 0.03, hdt + 0.03, 0.025)):
        cames.append(box("foot_plate", (2 * hw_, 2 * hd_, th_), loc=(0, ys, zz_), bevel=0.005, segments=1))
    for (nm, tl, m, uvl) in (("glass_star_a", ta, GA, ua), ("glass_star_b", tb, GB, ub_)):
        if tl:
            assign(tri_mesh(nm, tl, uvl, zcs), m)
    # stone pedestal under the foot, rising through the sill
    wpd = 2 * hwb + 0.2
    stones.append(blk("star_pedestal", (wpd, 0.46, zpd - 0.035 - zb), (0, -0.85, (zb + zpd - 0.035) / 2), r_p, ch=0.016, hide="bk", dens=0.14))
    stones.append(blk("star_pedestal_cap", (wpd + 0.07, 0.5, 0.036), (0, -0.865, zpd - 0.018), r_p, ch=0.012, hide="k", dens=0.25, chip=0.0))

    # crown: moulded band and five leaning leaves with ball tips, open inside (a closed shell with real thickness)
    def crown3d(cx, cy, z0c, rb, hc):
        npk, seg = 5, 6
        n_ = npk * seg
        V_, F_ = [], []
        rows = []
        for (shell, dr) in ((0, 0.0), (1, -0.011)):
            for lev in range(4):
                ids = []
                for i in range(n_):
                    a = -pi / 2 + 2 * pi * i / n_
                    pk = 1.0 - min(i % seg, seg - i % seg) / (seg / 2)        # 1 at the tip of a leaf, 0 in the dip between two
                    zf = (0.0, 0.16, 0.3, 0.3 + 0.7 * pk ** 1.7)[lev]
                    rr = rb * (1.0 + 0.06 * (lev == 1) + 0.85 * max(zf - 0.16, 0.0) ** 1.25) + dr
                    V_.append((cx + rr * cos(a), cy + rr * sin(a), z0c + hc * zf))
                    ids.append(len(V_) - 1)
                rows.append(ids)
        for lev in range(3):
            for i in range(n_):
                i1 = (i + 1) % n_
                o_, in_ = rows[lev], rows[4 + lev]
                o1, in1 = rows[lev + 1], rows[4 + lev + 1]
                F_.append((o_[i], o_[i1], o1[i1], o1[i]))
                F_.append((in_[i1], in_[i], in1[i], in1[i1]))
        for i in range(n_):
            i1 = (i + 1) % n_
            F_.append((rows[3][i], rows[3][i1], rows[7][i1], rows[7][i]))
            F_.append((rows[0][i1], rows[0][i], rows[4][i], rows[4][i1]))
        cr = mesh_object("crown", V_, F_)
        recalc_normals(cr)
        uv_box(cr)
        smooth(cr, 40)
        out = [cr]
        out.append(lathe("crown_band", [(rb * 0.98, -0.004), (rb * 1.13, 0.0), (rb * 1.16, 0.16 * hc * 0.3), (rb * 1.1, 0.16 * hc * 0.55), (rb * 1.16, 0.16 * hc * 0.8),
                                        (rb * 1.13, 0.16 * hc + 0.003), (rb * 1.0, 0.16 * hc + 0.006)], segments=20, loc=(cx, cy, z0c)))
        for k in range(npk):
            a = -pi / 2 + 2 * pi * k / npk
            rr = rb * (1.0 + 0.85 * 0.84 ** 1.25)
            out.append(sphere("crown_ball", 0.13 * rb, loc=(cx + rr * cos(a), cy + rr * sin(a), z0c + hc + 0.06 * rb), segments=8, rings=5))
        return out

    gilt += crown3d(0.0, ys, zcs + 0.94 * ro, 0.15 * ro, 0.305 * ro)
    for o in gilt:
        assign(o, GOLD)
    cm = join(cames, "cames")
    assign(cm, IRON)
    # the star is the brightest thing in the court: its light sits just in front of the glass (so the facade, the
    # niche behind it and the steps are lit from it and the pylons throw real shadows) with a weaker spill light
    # well in front that carries the colour out over the paving
    light("star", (0, ys - 0.62, zcs), color=pal["l"], intensity=0.55 * float(p.glow), range=14.0, flicker=0.04, shadow_near=ro + 0.45)
    light("star_spill", (0, ys - 1.5, zcs - 0.25), color=pal["l"], intensity=0.5 * float(p.glow), range=13.0, flicker=0.04, shadow_near=0.3)

    # ------------------------------------------------------------------------------ lichen on the heads
    for _ in range(int(2 * th / 57.3 * Rp * 4.0 * mossv)):
        a = radians(90 + r_m.u(-0.95, 0.95) * th)
        moss_patch(((Rp + 0.141) * cos(a), r_m.u(yt_ - 0.08, -0.1), czp + (Rp + 0.141) * sin(a)), (cos(a), 0, sin(a)), r_m.u(0.04, 0.11), along=(0, 1, 0), aspect=r_m.u(1.0, 2.2))
    moss_line((-cw / 2 + 0.05, ypf - 0.03, zp + 0.001), (cw / 2 - 0.05, ypf - 0.03, zp + 0.001), (0, 0, 1), 1.6, rad=(0.03, 0.08), spread=0.05)
    for sx in (-1, 1):
        moss_line((sx * (cw / 2 + 0.03), ypf, zp + 0.001), (sx * (cw / 2 + 0.03), -0.1, zp + 0.001), (0, 0, 1), 2.5, rad=(0.03, 0.08), spread=0.04)
        moss_line((sx * xbase, -1.1, zp + 0.012), (sx * xbase, -0.1, zp + 0.012), (sx * 0.7, 0, 0.7), 2.5, rad=(0.03, 0.07))
    moss_line((-xi + 0.1, -0.8, zc - R + 0.006), (xi - 0.1, -0.8, zc - R + 0.006), (0, 0, 1), 1.2, rad=(0.025, 0.06), spread=0.05)
    if mv:
        ms = mesh_object("moss", mv, mf)
        _uv_per_vertex(ms, muv)
        smooth(ms, 80)
        assign(ms, MOSS)

    # ====================================================================== materials and weathering
    r_k = rng.fork("streak")
    ph = [r_k.u(0, 6.28) for _ in range(5)]

    def S(h):                                  # sparse vertical streaks as a function of the horizontal position
        v = 0.55 * np.sin(h * 17.0 + ph[0]) + 0.3 * np.sin(h * 29.0 + ph[1]) + 0.15 * np.sin(h * 7.0 + ph[2])
        return np.clip(v * 1.5 + 0.2, 0.0, 1.0)

    def Lv(h):                                 # every streak runs a different length
        return 0.6 + 0.4 * np.sin(h * 9.0 + ph[3])

    # (x0, x1, z of the drip edge, length of the runs, strength): rain washes dirt down from every projection
    drips = [(-sw / 2 - batter - 0.1, sw / 2 + batter + 0.1, z_eb, 1.5, 0.3),
             (-cw / 2 - 0.05, cw / 2 + 0.05, zp - 0.2, 0.9, 0.26),
             (-W / 2, -cw / 2 - 0.02, wh - 0.3, 0.6, 0.24), (cw / 2 + 0.02, W / 2, wh - 0.3, 0.6, 0.24),
             (-xi, xi, zc - R - 0.07, 0.35, 0.26),
             (-W / 2, -cw / 2, lb - cop, 0.3, 0.2), (cw / 2, W / 2, lb - cop, 0.3, 0.2)]
    if tiers == 2:
        drips += [(-W / 2, -cw / 2, ub - cop, 0.45, 0.22), (cw / 2, W / 2, ub - cop, 0.45, 0.22)]

    nz_l = noise_mask(1.3, seed=seed + 5, lo=0.3, hi=0.75)     # big blotches: beds of different stone, old damp
    nz_s = noise_mask(0.2, seed=seed + 6, lo=0.3, hi=0.7)      # break-up of the wet and mossy zones
    k_m = clamp(0.45 + 1.2 * mossv, 0.0, 1.5)

    def wet_zones(co, n):
        """0..1 where running or splashing water keeps the stone wet: behind and beside the scupper sheets,
        under the weirs, round the water line of the tanks and on the copings the falls spatter."""
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        ax = np.abs(x)
        h = x + 0.7 * y
        zlo = wl_u if tiers == 2 else wl_l
        # the wall behind the scupper sheet and the runs either side of it, fading out sideways
        m = np.clip(1.0 - (np.abs(ax - xs) - ws / 2) / 0.3, 0.0, 1.0) * (z < z_sc + 0.03) * (z > zlo - 0.05) * (y > -T - 0.06) * (ax > cw / 2)
        wz = m * (0.55 + 0.45 * S(h * 1.9))
        if tiers == 2:
            mw = np.zeros(len(x))
            for v in notch_abs:
                mw = np.maximum(mw, np.clip(1.0 - (np.abs(ax - v) - nw_ / 2) / 0.14, 0.0, 1.0))
            wz = np.maximum(wz, mw * (np.abs(y - y_uf) < 0.3) * (y < y_uf + 0.02) * (z < ub - cop + 0.08) * (0.6 + 0.4 * S(h * 2.3)))
            wl = np.where(y > y_uf + 0.03, wl_u, wl_l)
        else:
            wl = np.full(len(x), wl_l)
        inside = (ax > cw / 2 + wt - 0.05) * (ax < W / 2 - 0.06 - wt + 0.05) * (y > y_lf + wt - 0.05) * (y < -T + 0.03)
        if wet:
            wz = np.maximum(wz, inside * np.clip(1.0 - (z - wl - 0.02) / 0.16, 0.0, 1.0))
            for (ix, iy, iz) in impacts_all:   # spatter round every impact
                d = np.hypot(np.clip(ax - ix, None, 0.0) * 0.6 + np.clip(ax - ix, 0.0, None), y - iy)
                wz = np.maximum(wz, np.clip(1.0 - d / 0.8, 0.0, 1.0) * (z > iz - 0.06) * (z < iz + 0.3) * (ax > cw / 2 + 0.02))
        else:
            wz = wz * 0.45                     # drained: only old tide marks
        return np.clip(wz, 0.0, 1.0)

    def stone_grime(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        ax = np.abs(x)
        h = x + 0.7 * y
        vert = np.clip(1.0 - np.abs(n[:, 2]) * 1.5, 0.0, 1.0)
        g = np.zeros(len(x))
        for (xa_, xb_, zt_, L_, amp) in drips:
            inx = np.clip((x - xa_) / 0.08, 0, 1) * np.clip((xb_ - x) / 0.08, 0, 1)
            d = (zt_ - z) / (L_ * Lv(h))
            g = np.maximum(g, 2.4 * amp * inx * np.clip(1.0 - d, 0.0, 1.0) * (d > -0.03) * (0.25 + 0.75 * S(h)))
        # tympanum under the raking cornice
        zarc = czp + np.sqrt(np.clip(Rp * Rp - x * x, 0.0, None))
        dt = (zarc - 0.02 - z) / 0.45
        g = np.maximum(g, 0.3 * (z > c0 + 0.17) * (ax < cch / 2) * np.clip(1.0 - dt, 0, 1) * (dt > -0.05) * (0.4 + 0.6 * S(h)))
        g = g * vert
        # blotchy old staining of the faces, heavier low down and on the wings than on the lit shrine front
        nl_, ns_ = nz_l(co, n), nz_s(co, n)
        g = np.maximum(g, (0.19 + 0.1 * (ax > cw / 2)) * nl_ * vert)
        # rising damp and splash-back along the foot of every wall
        g = np.maximum(g, 0.72 * np.clip(1.0 - z / 0.55, 0.0, 1.0) ** 1.3 * (0.45 + 0.55 * ns_))
        # soffits
        g = np.maximum(g, 0.22 * (n[:, 2] < -0.4))
        g = np.clip(g, 0.0, 0.72)
        # the shell niche behind the star: never washed by rain, a century of soot and dust, darkest in the flutes
        rn_ = np.hypot(x, z - zc)
        fl_n = np.abs(((np.arctan2(z - zc, x) * 16 / (2 * pi)) % 1.0) - 0.5) * 2.0
        g = np.maximum(g, (rn_ < Ri + 0.012) * (y > -0.43) * (0.64 + 0.2 * fl_n * np.clip(rn_ / Ri, 0.0, 1.0)))
        g = np.maximum(g, 0.6 * (y > -0.06) * (y < -0.01) * (z > zb) * (ax < xi + 0.01))
        # wet stone is dark stone
        wz = wet_zones(co, n)
        return np.clip(np.maximum(g, 0.9 * wz * (0.55 + 0.45 * ns_)), 0.0, 0.9)

    def stone_moss(co, n):
        """Algae and moss film on the stone itself (the cushions are separate): wet zones, the foot of the
        walls, and lichen on the weathered ledges."""
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        ax = np.abs(x)
        nl_, ns_ = nz_l(co, n), nz_s(co, n)
        wz = wet_zones(co, n)
        sp = wz * (0.35 + 0.65 * ns_) * 0.95
        sp = np.maximum(sp, 0.7 * np.clip(1.0 - z / 0.32, 0.0, 1.0) * ns_ * (ax > cw / 2 - 0.2))
        sp = np.maximum(sp, 0.45 * np.clip(1.0 - z / 0.2, 0.0, 1.0) * ns_ * nl_)          # green foot of the steps and newels
        sp = np.maximum(sp, 0.62 * (n[:, 2] > 0.6) * nl_ * ns_ * (z > 0.3) * (y > -1.5))
        return np.clip(sp * k_m, 0.0, 1.0)

    for o in stones:
        if o.name.startswith("inscription_stone"):
            continue
        assign(o, ST)
        paint(o, grime=stone_grime, special=stone_moss)
    # inscription tablet: honed (paler, smoother) slabs, each its own tone, dirt in the joints, a few runs from
    # the frame; the letter floors are bronze with what is left of the gilding
    nz_g = noise_mask(0.07, seed=seed + 9, lo=0.25, hi=0.75)

    def tab_wear(co, n):
        floor_ = co[:, 1] > yfc + cut_d - 0.002
        return np.where(floor_, 0.2 + 0.5 * gv + 0.12 * (nz_g(co, n) - 0.5), 0.12)

    def tab_grime(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        course_ = np.zeros(len(x), dtype=int)
        for zz in zj:
            course_ += (z > zz).astype(int)
        g = np.array([0.06, 0.0, 0.08, 0.03, 0.06])[np.clip(course_, 0, 4)] + 0.12 * S(x) * np.clip(1.0 - (zq + hq / 2 - z) / (0.45 * Lv(x)), 0.0, 1.0)
        g = np.where((y > yfc + 0.002) & (y < yfc + cut_d - 0.002), 0.7, g)       # the V joints
        return np.where(y > yfc + cut_d - 0.002, 0.0, g)

    paint(tab, grime=tab_grime, wear=tab_wear)
    nz_b = noise_mask(0.16, seed=seed + 2, lo=0.35, hi=0.7)
    for o in irons + [cm]:
        assign(o, IRON)
        paint(o, special=mask_mul(nz_b, 0.5))
    for o in bosses + raised:
        assign(o, IRON)
        paint(o, wear=0.35 + 0.5 * gv)

    socket("offering", "attach", (0.0, yl + 0.32, zl), role="prop_top")
    socket("wreath", "attach", (0.0, ypf - 0.06, zf0 + 0.02), role="planter")
    for sx in (-1, 1):
        socket("bench_%s" % ("l" if sx < 0 else "r"), "attach", (sx * (cw / 2 + (W - cw) / 4), y_lf - 0.45, 0.0), role="seat")
