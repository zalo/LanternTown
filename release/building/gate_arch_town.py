"""Town gate: a massive dressed-stone gateway. Two battlemented pier-towers, each a stepped buttress (deep base
stage and two weathered set-offs, arrow loops, corbelled parapet), flank a recessed curtain wall pierced by a round
arch of full-depth voussoirs on long-and-short jamb stones with imposts. Every facing stone is an individual
chamfered block laid in courses in recessed lime mortar over a rubble core; a few are chipped, proud, tilted or
replaced. Behind the front arch ring the passage widens into a rebated, oak-beamed chamber where the heavy studded
door leaves fold back flat against the walls, so the walkable passage stays clear. Over the arch hangs a shaped
sign board with gilded moulding, raised gilt lettering and a crown, lit by a cage lamp on an iron arm; scroll-bracket
lanterns flank the arch on both faces, crown banners hang from iron rods on the piers and a notice board with a
printed placard is fixed to one pier. On top a corbel table carries either a jettied timber-framed gatehouse room
with lit leaded windows under a tiled roof, or a crenellated wall-walk.
"""
from tk import *
import numpy as np
from math import sin, cos, radians, degrees, pi, atan2, sqrt, ceil, hypot
from mathutils import Vector, Matrix

META = {"name": "Town gate arch", "class": "landmark", "mount": "ground", "focus": [0.0, -1.9, 4.7]}
PARAMS = {
    "width": {"default": 10.0, "min": 8.0, "max": 13.0, "desc": "overall width of the gate body along X (m)"},
    "depth": {"default": 4.0, "min": 2.5, "max": 5.0, "desc": "overall depth along the passage axis Y (m)"},
    "passage_width": {"default": 3.8, "min": 3.0, "max": 4.5, "desc": "clear walkable width of the passage (m)"},
    "passage_height": {"default": 4.6, "min": 4.0, "max": 5.4, "desc": "clear height at the crown of the arch (m)"},
    "height": {"default": 8.0, "min": 7.0, "max": 10.0, "desc": "height of the stone gate to the top of the battlements (m)"},
    "sign_text": {"default": "LANTERN SQUARE", "desc": "raised gilded lettering on the sign board over the arch"},
    "poster_text": {"default": "A BETTER TOMORROW", "desc": "printed text of the big placard on the notice board"},
    "banners": {"default": True, "desc": "crown banners hanging on the piers"},
    "gatehouse": {"default": True, "desc": "timber-framed gatehouse room above (otherwise a crenellated wall-walk)"},
    "doors": {"default": True, "desc": "heavy timber door leaves folded open inside the passage"},
    "posters": {"default": True, "desc": "notice board with placard and bills on the left pier"},
    "back_lanterns": {"default": True, "desc": "second pair of wall lanterns on the town (+Y) face"},
    "roof": {"default": "half_hip", "choices": ["half_hip", "gable", "hip"], "desc": "gatehouse roof form"},
    "covering": {"default": "roof_tile_clay", "choices": ["roof_tile_clay", "roof_slate", "wood_shingle"], "desc": "gatehouse roof covering"},
    "stone": {"default": "#b3a48e", "desc": "tint of the stone (clean dressed stone; the field blocks weather darker)"},
    "banner_color": {"default": "#26397c", "desc": "livery colour: banner cloth, sign board and placard (crowns and lettering are gold)"},
    "plaster": {"default": "#d6c7a4", "desc": "limewash colour of the gatehouse panels"},
    "timber": {"default": "#4f3b2b", "desc": "stain of the oak carpentry and doors"},
    "lantern_intensity": {"default": 11.0, "min": 2.0, "max": 30.0, "desc": "light intensity of each wall lantern (cd)"},
    "moss": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "moss and damp on the stonework and roof"},
}
VARIANTS = {
    "battlement": {"gatehouse": False, "width": 9.0, "depth": 3.4, "height": 7.6, "passage_width": 3.4, "passage_height": 4.3,
                   "stone": "#a3a09b", "sign_text": "NORTH GATE", "banner_color": "#6a1a1a", "moss": 0.7, "seed": 7},
    "grand_slate": {"width": 13.0, "depth": 5.0, "height": 9.4, "passage_width": 4.5, "passage_height": 5.2, "roof": "hip",
                    "covering": "roof_slate", "stone": "#c2b599", "plaster": "#cfc9bb", "timber": "#3a2c22",
                    "sign_text": "LANTERN TOWN", "poster_text": "KEEP THE LAMPS LIT", "moss": 0.2, "seed": 12},
    "narrow_gable": {"width": 8.0, "depth": 2.8, "height": 7.0, "passage_width": 3.0, "passage_height": 4.0, "roof": "gable",
                     "covering": "wood_shingle", "stone": "#ac9476", "banners": False, "plaster": "#c9b089",
                     "banner_color": "#1f4a36", "sign_text": "OLD TOWN", "moss": 0.85, "seed": 23},
    "plain_wallwalk": {"gatehouse": False, "banners": False, "posters": False, "width": 11.0, "depth": 4.4, "height": 8.6,
                       "passage_height": 5.0, "stone": "#b8a98c", "sign_text": "MARKET GATE", "banner_color": "#4a1c1c",
                       "back_lanterns": False, "moss": 0.5, "seed": 31},
}

G = 0.012                                  # nominal masonry joint
MB = 0.008                                 # recess of the mortar bed behind the nominal wall face
BL = (0.36, 0.5, 0.66, 0.84)               # facing block base lengths
BH = (0.30, 0.35, 0.40)                    # course heights
PBL = (0.7, 0.95, 1.25)                    # plinth block base lengths
GOLD = "#e0b24c"
STRIPE = 12.0                              # one canvas material carries navy (U ~ UN) and gold (U ~ UG) in two wide stripes
UN, UG = 0.5 * STRIPE, 0.0
TONES = ((0.03, 0.0), (0.09, 0.0), (0.16, 0.0), (0.25, 0.0), (0.05, 0.25), (0.12, 0.4), (0.02, 0.5))   # per-stone (grime, wear)
TONE_W = (0.2, 0.24, 0.18, 0.08, 0.14, 0.1, 0.06)
CROWN = [(-0.42, 0.0), (0.42, 0.0), (0.44, 0.12), (0.5, 0.52), (0.25, 0.28), (0.0, 0.62), (-0.25, 0.28), (-0.5, 0.52), (-0.44, 0.12)]


def XZ(u, v, d):
    return (u, d, v)


def prism(name, poly, d0, d1, fn, bev=0.0, seg=1):
    """Extrude a 2D polygon [(u, v)] from d0 to d1 through a mapping fn(u, v, d) -> (x, y, z)."""
    n = len(poly)
    verts = [fn(u, v, d0) for u, v in poly] + [fn(u, v, d1) for u, v in poly]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    if bev > 0:
        bevel(o, bev, seg)
    return o


def sheet(name, verts, faces, outward):
    """Open sheet mesh whose faces are wound to face `outward`."""
    V = np.asarray(verts, dtype=np.float64)
    out = np.asarray(outward, dtype=np.float64)
    ff = []
    for f in faces:
        nrm = np.cross(V[f[1]] - V[f[0]], V[f[-1]] - V[f[0]])
        ff.append(tuple(f) if nrm @ out >= 0 else tuple(reversed(f)))
    return mesh_object(name, [tuple(v) for v in V], ff)


def ring_moulding(name, x0, x1, y0, y1, prof, k=(1, 1, 1, 1)):
    """Mitred moulding around a rectangle: prof = closed loop of (outward offset, z); k scales the projection on the
    (x0, x1, y0, y1) sides."""
    verts, faces = [], []
    n = len(prof)
    for (o, z) in prof:
        a, b, c, d = (o * k[0] if o > 0 else o), (o * k[1] if o > 0 else o), (o * k[2] if o > 0 else o), (o * k[3] if o > 0 else o)
        verts += [(x0 - a, y0 - c, z), (x1 + b, y0 - c, z), (x1 + b, y1 + d, z), (x0 - a, y1 + d, z)]
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(4):
            j2 = (j + 1) % 4
            faces.append((i * 4 + j, i * 4 + j2, i2 * 4 + j2, i2 * 4 + j))
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    flat(o)
    return o


def split(span, r, lo, hi):
    if span < hi:
        if span < lo * 1.6:
            return [span]
        d = r.u(-0.06, 0.06)
        return [span * 0.5 + d, span * 0.5 - d]
    ls, tot = [], 0.0
    while tot < span:
        l = r.u(lo, hi)
        ls.append(l)
        tot += l
    if tot - span > ls[-1] * 0.5 and len(ls) > 1:
        tot -= ls.pop()
    k = span / tot
    return [l * k for l in ls]


def make_courses(z0, z1, r):
    if z1 - z0 < 0.05:
        return []
    hts, tot = [], 0.0
    while tot < (z1 - z0) - 0.17:
        h = r.pick(BH)
        hts.append(h)
        tot += h
    if not hts:
        hts, tot = [z1 - z0], z1 - z0
    k = (z1 - z0) / tot
    out, z = [], z0
    for h in hts:
        out.append((z, z + h * k))
        z += h * k
    return out


def nearest(seq, v):
    return min(range(len(seq)), key=lambda i: abs(seq[i] - v))


def n1(t, cell, seed):
    """1D value noise 0..1 (vectorised)."""
    tab = np.random.default_rng(int(seed)).random(2048)
    q = np.asarray(t, dtype=np.float64) / cell + 512.0
    i = np.floor(q).astype(np.int64)
    f = q - i
    f = f * f * (3 - 2 * f)
    return tab[i % 2048] * (1 - f) + tab[(i + 1) % 2048] * f


def shell(name, L, dep, h, cols, rows, b, r, chip=False):
    """One facing stone: dressed front face (subdivided so weathering masks resolve), chamfered arrises, returns into
    the wall, no hidden back face. Local -Y is the outward face."""
    xs = [-L / 2 + b + (L - 2 * b) * i / cols for i in range(cols + 1)]
    zs_ = [-h / 2 + b + (h - 2 * b) * j / rows for j in range(rows + 1)]
    yf, y1, yb = -dep / 2, -dep / 2 + b, dep / 2
    V, I = [], {}

    def add(key, co):
        I[key] = len(V)
        V.append(list(co))

    for j in range(rows + 1):
        for i in range(cols + 1):
            add(("f", i, j), (xs[i], yf + r.u(-0.003, 0.003), zs_[j]))
    for i in range(cols + 1):
        add(("t", i), (xs[i], y1, h / 2))
        add(("b", i), (xs[i], y1, -h / 2))
    for j in range(rows + 1):
        add(("l", j), (-L / 2, y1, zs_[j]))
        add(("r", j), (L / 2, y1, zs_[j]))
    for sx in (0, 1):
        for sz in (0, 1):
            add(("k", sx, sz), ((sx - 0.5) * L, yb, (sz - 0.5) * h))
    if chip:                                    # a spalled corner
        ci, cj = (cols if r.chance(0.5) else 0), (rows if r.chance(0.65) else 0)
        kx, kz, ky = r.u(0.035, 0.08), r.u(0.03, 0.065), r.u(0.012, 0.028)
        sx, sz = (-1 if ci else 1), (-1 if cj else 1)
        v = V[I[("f", ci, cj)]]
        v[0] += sx * kx
        v[2] += sz * kz
        v[1] += ky * 0.7
        v = V[I[("t" if cj else "b", ci)]]
        v[0] += sx * kx
        v[1] += ky
        v = V[I[("r" if ci else "l", cj)]]
        v[2] += sz * kz
        v[1] += ky
    Fq = []
    f_ = lambda i, j: I[("f", i, j)]
    t_, b_, l_, r_ = (lambda i: I[("t", i)]), (lambda i: I[("b", i)]), (lambda j: I[("l", j)]), (lambda j: I[("r", j)])
    k_ = lambda sx, sz: I[("k", sx, sz)]
    for j in range(rows):
        for i in range(cols):
            Fq.append((f_(i, j), f_(i + 1, j), f_(i + 1, j + 1), f_(i, j + 1)))
    for i in range(cols):
        Fq.append((f_(i, rows), f_(i + 1, rows), t_(i + 1), t_(i)))
        Fq.append((f_(i + 1, 0), f_(i, 0), b_(i), b_(i + 1)))
    for j in range(rows):
        Fq.append((f_(cols, j), r_(j), r_(j + 1), f_(cols, j + 1)))
        Fq.append((f_(0, j + 1), l_(j + 1), l_(j), f_(0, j)))
    Fq += [(f_(cols, rows), r_(rows), t_(cols)), (f_(0, rows), t_(0), l_(rows)), (f_(cols, 0), b_(cols), r_(0)), (f_(0, 0), l_(0), b_(0))]
    Fq.append(tuple([t_(i) for i in range(cols + 1)] + [k_(1, 1), k_(0, 1)]))
    Fq.append(tuple([b_(i) for i in range(cols, -1, -1)] + [k_(0, 0), k_(1, 0)]))
    Fq.append(tuple([r_(0), k_(1, 0), k_(1, 1)] + [r_(j) for j in range(rows, 0, -1)]))
    Fq.append(tuple([l_(j) for j in range(rows + 1)] + [k_(0, 1), k_(0, 0)]))
    Fq += [(t_(cols), r_(rows), k_(1, 1)), (l_(rows), t_(0), k_(0, 1)), (r_(0), b_(cols), k_(1, 0)), (b_(0), l_(0), k_(0, 0))]
    o = mesh_object(name, [tuple(v) for v in V], Fq)
    flat(o)
    return o


def knock(o, r, n=1, size=0.05):
    """Knock off n corners of a dressed stone (applied to the realised mesh, local space)."""
    realize(o)
    me = o.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    c = (co.min(0) + co.max(0)) / 2
    ext = np.maximum(co.max(0) - co.min(0), 1e-4)
    for _ in range(n):
        d = np.array([r.pick((-1, 1)), r.pick((-1, 1)), r.pick((-1, 1, 1))], dtype=np.float64)
        corner = co[np.argmax(((co - c) / ext) @ d)].copy()
        dist = np.linalg.norm(co - corner, axis=1)
        k = np.clip(1 - dist / size, 0, 1)
        dv = c - corner
        dv /= max(np.linalg.norm(dv), 1e-6)
        co += k[:, None] * dv[None, :] * size * 0.6
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()
    return o


def bar(name, a, b, w, sides=4):
    a, b = Vector(a), Vector(b)
    d = b - a
    e = Vector((0, 0, 1)).rotation_difference(d.normalized()).to_euler()
    return cylinder(name, w * (0.7071 if sides == 4 else 0.5), d.length, loc=tuple((a + b) / 2),
                    rot=(degrees(e.x), degrees(e.y), degrees(e.z)), verts=sides)


def tim(name, length, width, thick, loc, axis='X', rot=None, bev=0.01):
    """Cheap squared timber with grain along `axis`."""
    size = {'X': (length, width, thick), 'Y': (width, length, thick), 'Z': (width, thick, length)}[axis]
    o = box(name, size, loc=loc, rot=rot, bevel=min(bev, width * 0.3, thick * 0.3), segments=1)
    uv_grain(o, axis)
    return o


def build(p, rng):
    W, D = p.width, p.depth
    pw = min(p.passage_width, W - 3.6)
    R = pw / 2
    zc = p.passage_height
    zs = zc - R                                  # spring line
    tr = 0.5                                     # arch ring thickness
    H = max(p.height, zc + 2.9)
    zdeck, zt = H - 1.3, H - 0.55                # wall-walk level, top of breast wall / pier parapets
    PL = 0.07                                    # plinth projection
    tyo = D / 2 - PL
    bp = min(clamp(D * 0.2, 0.5, 0.85), max(0.3, tyo - 0.8))   # buttress projection in front of the curtain wall
    o1, o2 = 0.30 * bp, 0.27 * bp                # the two weathered offsets
    ty1, ty2 = tyo, tyo - o1
    ty3 = ty2 - o2
    cy = tyo - bp                                # curtain wall face
    rd = min(0.45, cy * 0.4)                     # depth of the arch rings
    bw = clamp((W - pw) / 2 - tr - 0.75, 0.8, 1.6)
    xo, xt = W / 2 - PL, W / 2 - bw
    zp = 0.52
    zcor = zdeck - 0.24
    zA = clamp(0.36 * zcor, 2.1, 2.6)            # first offset
    zB = zA + 0.36 * (zcor - zA)                 # second offset
    ri1, ri2 = clamp(o1 * 1.3, 0.2, 0.34), clamp(o2 * 1.3, 0.2, 0.34)
    Ld = R + 0.08                                # door leaf width
    chamber = 2 * cy - 2 * rd
    nfold = max(1, int(ceil(Ld / (chamber - 0.1)))) if p.doors else 1
    lp = Ld / nfold
    rc = 0.045 + nfold * 0.105                   # door rebate each side
    xw = R + rc
    zceil = max(zc + 0.2, 4.7)
    td = 0.3                                     # facing block depth
    gd = max(3.2, 2 * cy + 0.9)                  # gatehouse depth (jettied over both faces)
    ov = 0.14                                    # parapet overhang carried on the corbel table
    py, pth = cy + ov, 0.36                      # wall-walk breast wall: outer face, thickness
    typ = ty3 + ov
    sd = p.seed * 13 + 5

    m_field = mat("stone_rough", p.stone)
    m_trim = m_field                             # one stone: dressed work is told apart by its tooling, tone and cleanliness
    m_moss = mat("moss") if p.moss > 0.02 else None
    m_oak = mat("wood_oak", p.timber)
    m_iron = mat("iron_wrought")
    m_gilt = mat("metal_plain", "#f0c566", rough=0.34)
    m_can = mat("canvas", p.banner_color, stripe=STRIPE, tint2=GOLD)
    m_gl = mat("glass_lantern", glow=1.8)
    m_pl = mat("plaster_lime", p.plaster)
    m_lit = mat("window_lit", "#ffc37a", glow=7.0) if p.gatehouse else None
    m_fl = m_lit if p.gatehouse else mat("emissive", "#ffcf8a", glow=12)

    def livery(o, U, xc):
        """The canvas stripes run across U: park the object's UVs (local-space planar) in the middle of a stripe."""
        assign(o, m_can)
        uv_planar(o, (1, 0, 0), (0, 0, 1), offset=(U - (xc - o.location.x), 0.0))
        return o

    def navy(o, xc=0.0):
        return livery(o, UN, xc)

    def gold(o, xc=0.0):
        return livery(o, UG, xc)

    moss_sites = []

    def moss_at(pos, nrm, sx_, sy_, sz_, tangent=None):
        if m_moss is not None:
            moss_sites.append((Vector(pos), Vector(nrm).normalized(), sx_, sy_, sz_, tangent))

    # ------------------------------------------------------------------ arrow loops (set out first: courses align to them)
    def loop_span(z0, z1, hmax):
        zl0 = z0 + 0.12
        hl = min(hmax, z1 - zl0 - 0.32)
        return (zl0, zl0 + hl) if hl >= 0.55 else None

    loop2 = loop_span(zA + ri1, zB, 1.0)
    loop3 = loop_span(zB + ri2 + 0.25, zcor - 0.35, 1.25) if not p.banners else None
    lw2 = min(0.23, (xo - xt) / 2 - 0.2)
    xpc = (xt + xo) / 2

    def loop_courses(z0, z1, lpn, r):
        if lpn is None:
            return make_courses(z0, z1, r)
        return make_courses(z0, lpn[0], r) + make_courses(lpn[0], lpn[1], r) + make_courses(lpn[1], z1, r)

    def clipl(lpn):
        L = xo - xt

        def f(za, zb):
            if lpn is not None and za >= lpn[0] - 1e-3 and zb <= lpn[1] + 1e-3:
                return [(0.0, L / 2 - lw2), (L / 2 + lw2, L)]
            return [(0.0, L)]
        return f

    # ------------------------------------------------------------------ facing blocks
    r_bl = rng.fork("blocks")
    r_sh = rng.fork("shells")
    bases = {}
    for i, L0 in enumerate(BL):
        for j, h0 in enumerate(BH):
            for cv in range(3):
                bases[("f", i, j, cv)] = shell("blk", L0, 0.3, h0, max(3, int(round(L0 / 0.115))), 1, 0.012, r_sh, chip=cv > 0)
    for i, L0 in enumerate(PBL):
        for cv in range(2):
            bases[("p", i, 0, cv)] = shell("pblk", L0, 0.3, zp, max(4, int(round(L0 / 0.16))), 2, 0.026, r_sh, chip=cv > 0)
    groups = {}

    def lay_face(a, b, crs, tdep, r, clip=None, trim=(None, None), trim_d=(0.3, 0.3), kind="f", lens=(0.42, 0.86)):
        ax, ay = a
        L = hypot(b[0] - ax, b[1] - ay)
        if L < 0.05:
            return
        ux, uy = (b[0] - ax) / L, (b[1] - ay) / L
        nx, ny = uy, -ux
        ang = degrees(atan2(uy, ux))
        bl = BL if kind == "f" else PBL
        for ci, (za, zb) in enumerate(crs):
            u0, u1 = 0.0, L
            if trim[0] is not None and (ci + trim[0]) % 2 == 0:
                u0 += trim_d[0] + G
            if trim[1] is not None and (ci + trim[1]) % 2 == 1:
                u1 -= trim_d[1] + G
            for (ia, ib) in (clip(za, zb) if clip else [(0.0, L)]):
                ia, ib = max(ia, u0), min(ib, u1)
                if ib - ia < 0.1:
                    continue
                u = ia
                for l in split(ib - ia, r, lens[0], lens[1]):
                    proud = r.u(-0.004, 0.008) + (0.011 if r.chance(0.07) else 0.0)
                    off = proud - tdep / 2
                    g = r.u(0.008, 0.016)
                    iL = nearest(bl, l - g)
                    ih = nearest(BH, zb - za - g) if kind == "f" else 0
                    h0 = BH[ih] if kind == "f" else zp
                    M = trs((ax + ux * (u + l / 2) + nx * off, ay + uy * (u + l / 2) + ny * off, (za + zb) / 2),
                            (r.u(-0.5, 0.5), r.u(-0.25, 0.25), ang + r.u(-0.45, 0.45)),
                            ((l - g) / bl[iL], tdep / 0.3, (zb - za - g) / h0))
                    cv = (r.pick((1, 2)) if kind == "f" else 1) if r.chance(0.16) else 0
                    rep = kind == "f" and r.chance(0.04)
                    tone = r.weighted(list(zip(range(len(TONES)), TONE_W)))
                    groups.setdefault((kind, iL, ih, cv, tone, rep), []).append(M)
                    u += l

    def lay_poly(poly, crs, r, clips=None, tds=None, kind="f", lens=(0.42, 0.86), dep=td):
        n = len(poly)
        clips, tds = clips or {}, tds or {}
        conv = []
        for j in range(n):
            p0, p1, p2 = poly[j - 1], poly[j], poly[(j + 1) % n]
            conv.append((p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]) > 0)
        for i in range(n):
            j2 = (i + 1) % n
            lay_face(poly[i], poly[j2], crs, tds.get(i, dep), r, clip=clips.get(i),
                     trim=(i if conv[i] else None, j2 if conv[j2] else None),
                     trim_d=(tds.get((i - 1) % n, dep), tds.get(j2, dep)), kind=kind, lens=lens)

    def footprint(xi, xout, ty, cyy):
        return [(xi, -ty), (xout, -ty), (xout, ty), (xi, ty), (xi, cyy), (-xi, cyy), (-xi, ty), (-xout, ty), (-xout, -ty),
                (-xi, -ty), (-xi, -cyy), (xi, -cyy)]

    def clipf(xi):
        rr = R + tr - 0.06

        def f(za, zb):
            dz = zb - zs
            if dz >= rr:
                return [(0.0, 2 * xi)]
            e = rr if dz <= 0 else sqrt(rr * rr - dz * dz)
            return [(0.0, xi - e), (xi + e, 2 * xi)]
        return f

    td_wall = min(td, rd - 0.08)
    wall_tds = {4: td_wall, 10: td_wall}

    def stage_clips(xi, lpn):
        c = {4: clipf(xi), 10: clipf(xi)}
        for k in (0, 2, 6, 8):
            c[k] = clipl(lpn)
        return c

    crs1 = make_courses(zp, zA, r_bl)
    crs2 = loop_courses(zA, zB, loop2, r_bl)
    crs3 = loop_courses(zB, zcor, loop3, r_bl)
    crs4 = make_courses(zdeck, zt, r_bl)
    lay_poly(footprint(xt - PL, W / 2, D / 2, cy + PL), [(0.0, zp)], r_bl, clips={4: clipf(xt - PL), 10: clipf(xt - PL)},
             tds=wall_tds, kind="p", lens=(0.65, 1.3))
    lay_poly(footprint(xt, xo, ty1, cy), crs1, r_bl, clips=stage_clips(xt, None), tds=wall_tds)
    lay_poly(footprint(xt, xo, ty2, cy), crs2, r_bl, clips=stage_clips(xt, loop2), tds=wall_tds)
    lay_poly(footprint(xt, xo, ty3, cy), crs3, r_bl, clips=stage_clips(xt, loop3), tds=wall_tds)
    lay_poly([(xt, -typ), (xo, -typ), (xo, typ), (xt, typ)], crs4, r_bl)
    lay_poly([(-xo, -typ), (-xt, -typ), (-xt, typ), (-xo, typ)], crs4, r_bl)
    # chamber side walls (behind the folded doors)
    crs_ch = make_courses(0.0, zceil + 0.25, r_bl)
    lay_face((xw, cy - rd), (xw, -(cy - rd)), crs_ch, 0.2, r_bl)
    lay_face((-xw, -(cy - rd)), (-xw, cy - rd), crs_ch, 0.2, r_bl)
    if not p.gatehouse:                       # breast walls of the wall-walk, corbelled out: outer and inner wythe
        lay_face((-xt, -py), (xt, -py), crs4, 0.17, r_bl)
        lay_face((xt, py), (-xt, py), crs4, 0.17, r_bl)
        lay_face((xt, -(py - pth)), (-xt, -(py - pth)), crs4, 0.17, r_bl)
        lay_face((-xt, py - pth), (xt, py - pth), crs4, 0.17, r_bl)

    # ------------------------------------------------------------------ mortar beds / rubble core just behind the facing
    backs = []

    def cbox(x0, x1, y0, y1, z0, z1, m=None, name="core"):
        b = box(name, (x1 - x0, y1 - y0, z1 - z0), loc=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
        if m is None:
            backs.append(b)
        else:
            assign(b, m)
        return b

    trim = []
    r_tr = rng.fork("trim")

    def dress(o, r=r_tr, chips=0.3, size=0.05, tone=True):
        """Finish a dressed stone: knocked corners and its own tone (grime darker / wear lighter)."""
        if chips > 0 and r.chance(chips):
            knock(o, r, 1 if r.chance(0.7) else 2, size)
        if tone:
            paint(o, grime=r.u(0.0, 0.13), wear=r.u(0.0, 0.3), mode='set')
        trim.append(o)
        return o

    merl = []

    def merlon(x, y, sx_, sy_, r, ridge):
        hb = (H - zt) - 0.12 + r.u(-0.02, 0.012)
        merl.append((x, y, sx_, sy_, hb, ridge))
        poly = [(x - sx_ / 2, y - sy_ / 2), (x + sx_ / 2, y - sy_ / 2), (x + sx_ / 2, y + sy_ / 2), (x - sx_ / 2, y + sy_ / 2)]
        lay_poly(poly, [(zt, zt + hb)], r_bl, lens=(0.5, 0.9), dep=min(0.17, sx_ * 0.4, sy_ * 0.4))
        cbox(x - sx_ / 2 + MB, x + sx_ / 2 - MB, y - sy_ / 2 + MB, y + sy_ / 2 - MB, zt - 0.02, zt + hb - 0.012)
        ca, cb = (sy_, sx_) if ridge == 'x' else (sx_, sy_)          # across / along the ridge
        ca, cb = ca + 0.075, cb + 0.075
        e = r.u(-0.03, 0.03)
        prof = [(-ca / 2, 0.0), (ca / 2, 0.0), (ca / 2, 0.04 + r.u(0, 0.014)), (e + 0.02, 0.105 + r.u(-0.012, 0.012)),
                (e - 0.02, 0.105 + r.u(-0.012, 0.012)), (-ca / 2, 0.04 + r.u(0, 0.014))]
        fn = (lambda u, v, d: (d, u, v)) if ridge == 'x' else (lambda u, v, d: (u, d, v))
        c = prism("merlon_coping", prof, -cb / 2, cb / 2, fn, bev=0.012)
        set_loc_rot(c, (x + r.u(-0.008, 0.008), y + r.u(-0.008, 0.008), zt + hb - 0.004), (r.u(-1.2, 1.2), r.u(-1.2, 1.2), r.u(-1.5, 1.5)))
        dress(c, r, chips=0.75, size=0.07)

    for sx in (-1, 1):
        cols = [(sx * (xt + 0.27), 0.5), (sx * (xo - 0.27), 0.5)] if bw >= 1.45 else [(sx * (xt + xo) / 2, xo - xt - 0.04)]
        ny_ = max(2, int(round(2 * typ / 1.12)) + 1)
        for (mx, msx) in cols:
            for k in range(ny_):
                merlon(mx, -typ + 0.28 + (2 * typ - 0.56) * k / (ny_ - 1), msx, 0.52, r_tr, 'y')
    if not p.gatehouse:
        nm = max(3, int(round(2 * xt / 1.2)))
        pitch = 2 * xt / nm
        for sgn in (-1, 1):
            for k in range(nm):
                merlon(-xt + (k + 0.5) * pitch, sgn * (py - pth / 2), 0.56 * pitch, pth, r_tr, 'x')

    field, repl = [], []
    for (kind, iL, ih, cv, tone, rep), mats in groups.items():
        base = bases[(kind, iL, ih, cv)]
        gt, wt_ = TONES[tone] if not rep else (0.0, 0.0)     # replaced stones are still clean
        paint(base, grime=gt, wear=wt_, special=0.0, mode='set')
        o = instances_to_mesh("facing", base, mats, seed=p.seed * 31 + iL * 5 + ih * 3 + tone * 7 + cv)
        assign(o, m_trim if rep else m_field)
        (repl if rep else field).append(o)
    delete(list(bases.values()))
    field = join(field, "facing")
    repl = join(repl, "replaced_stones") if repl else None

    piers_box = {}
    for sx in (-1, 1):
        x0, x1 = sorted((sx * (xt + MB), sx * (xo - MB)))
        piers_box[(sx, 1)] = cbox(x0, x1, -ty1 + MB, ty1 - MB, 0.0, zA)
        piers_box[(sx, 2)] = cbox(x0, x1, -ty2 + MB, ty2 - MB, zA, zB)
        piers_box[(sx, 3)] = cbox(x0, x1, -ty3 + MB, ty3 - MB, zB, zdeck)
        cbox(x0, x1, -typ + MB, typ - MB, zdeck, zt - 0.01)
        t_ = cbox(x0 + 0.03, x1 - 0.03, -typ + 0.06, typ - 0.06, zt - 0.03, zt + 0.025, m_trim, "pier_top")
        dress(t_, chips=0)
        x0, x1 = sorted((sx * (xw + 0.03), sx * (xt + 0.05)))
        cbox(x0, x1, -cy + rd, cy - rd, 0.0, zdeck - 0.02)
    cbox(-(xw + 0.03), xw + 0.03, -cy + rd, cy - rd, zceil + 0.27, zdeck - 0.02)
    hr = R + 0.22
    slab = [(-xt - 0.05, 0.0), (-hr, 0.0), (-hr, zs)]
    slab += [(hr * cos(pi - pi * k / 20), zs + hr * sin(pi - pi * k / 20)) for k in range(1, 20)]
    slab += [(hr, zs), (hr, 0.0), (xt + 0.05, 0.0), (xt + 0.05, zdeck - 0.02), (-xt - 0.05, zdeck - 0.02)]
    backs.append(prism("core_front", slab, -cy + MB, -cy + rd, XZ))
    backs.append(prism("core_back", slab, cy - rd, cy - MB, XZ))
    if not p.gatehouse:
        for sgn in (-1, 1):
            y0, y1 = sorted((sgn * (py - MB), sgn * (py - pth + MB)))
            cbox(-xt, xt, y0, y1, zdeck - 0.02, zt - 0.012)
        dress(cbox(-xt, xt, -(py - pth + 0.05), py - pth + 0.05, zdeck - 0.04, zdeck + 0.03, m_trim, "walk"), chips=0)

    # arrow loops: dressed splayed jambs, weathered sill, lintel, dark embrasure cut into the core
    pockets = []

    def arrow_loop(xc, sgn, tyf, lpn, stage_box):
        z0, z1 = lpn
        s2, hs, hl_ = 0.04, 0.15, 0.17
        za, zb = z0 + hs, z1 - hl_
        plan = [(s2 + 0.05, -0.008), (lw2 - 0.006, -0.008), (lw2 - 0.006, 0.26), (s2, 0.26), (s2, 0.055)]
        nst = 2 if zb - za > 0.62 else 1
        for side in (-1, 1):
            zz = za
            for k in range(nst):
                hh = (zb - za) / nst + (r_tr.u(-0.05, 0.05) if (nst == 2 and k == 0) else 0.0)
                hh = hh if k == 0 else zb - zz
                j = prism("loop_jamb", plan, zz + 0.004, zz + hh - 0.004,
                          lambda u, v, d, side=side: (xc + side * u, sgn * (tyf - v), d), bev=0.008)
                dress(j, chips=0.5, size=0.04)
                zz += hh
        sill = prism("loop_sill", [(-0.035, z0 + 0.004), (0.26, z0 + 0.004), (0.26, za - 0.004), (0.06, za - 0.004), (-0.035, za - 0.07)],
                     -lw2 + 0.004, lw2 - 0.004, lambda u, v, d: (xc + d, sgn * (tyf - u), v), bev=0.008)
        dress(sill, chips=0.4)
        lin = stone_block("loop_lintel", (2 * lw2 - 0.008, 0.27, hl_ - 0.008), loc=(xc, sgn * (tyf - 0.125), z1 - hl_ / 2), rng=r_tr,
                          chamfer=0.012, rough=0.003, detail=0.3)
        dress(lin, chips=0.4)
        cut = box("loop_cut", (2 * s2 + 0.02, 0.6, zb - za), loc=(xc, sgn * (tyf - 0.1), (za + zb) / 2))
        boolean(stage_box, cut)
        pockets.append((xc, sgn * tyf, (za + zb) / 2))

    for sx in (-1, 1):
        for sgn in (-1, 1):
            if loop2:
                arrow_loop(sx * xpc, sgn, ty2, loop2, piers_box[(sx, 2)])
            if loop3:
                arrow_loop(sx * xpc, sgn, ty3, loop3, piers_box[(sx, 3)])

    # ------------------------------------------------------------------ arch rings: voussoirs, jambs, imposts
    def arch_ring(sgn, r):
        y_out, y_in = sgn * (cy + 0.045), sgn * (cy - rd - 0.02)
        ya, yb = min(y_out, y_in), max(y_out, y_in)
        nv = int(round(pi * (R + tr / 2) / 0.42))
        nv += 1 - nv % 2
        for k in range(nv):
            key = (k == nv // 2)
            a0, a1 = pi - pi * k / nv, pi - pi * (k + 1) / nv
            am, hw = (a0 + a1) / 2, (a0 - a1) / 2 - 0.005 / R
            r1 = R + tr + (0.16 if key else (0.07 if k % 2 else 0.0) + r.u(-0.02, 0.02))
            rm = (R + r1) / 2
            # each voussoir is built in its own frame (local Z = radial) so the tooling of the stone follows the radius
            poly = [(R * sin(-hw + 2 * hw * t / 3), R * cos(-hw + 2 * hw * t / 3) - rm) for t in range(4)]
            poly += [(r1 * sin(hw - 2 * hw * t / 3), r1 * cos(hw - 2 * hw * t / 3) - rm) for t in range(4)]
            pr = 0.04 if key else r.u(-0.006, 0.008)
            v = prism("voussoir", poly, ya - (pr if sgn < 0 else 0), yb + (pr if sgn > 0 else 0), XZ, bev=0.014)
            set_loc_rot(v, (rm * cos(am), 0.0, zs + rm * sin(am)), (0.0, 90.0 - degrees(am) + r.u(-0.25, 0.25), 0.0))
            uv_box(v, rot90=r.chance(0.3))
            smooth(v, 25)
            dress(v, r, chips=0.0 if key else 0.45, size=0.06)
        zj = zs - 0.22
        nj = max(3, int(round((zj - zp) / 0.46)))
        hj = (zj - zp) / nj
        for sx in (-1, 1):
            for k in range(-1, nj + 1):
                if k == -1:
                    z0, z1, wj, pr, ch = 0.0, zp, tr + 0.24, PL, 0.03          # plinth stone
                elif k == nj:
                    z0, z1, wj, pr, ch = zj, zs, tr + 0.12, 0.05, 0.028         # impost
                else:
                    z0, z1, wj, pr, ch = zp + k * hj, zp + (k + 1) * hj, tr + (0.2 if k % 2 == 0 else 0.0) + r.u(-0.02, 0.02), r.u(0.0, 0.01), 0.014
                sy = (yb - ya) + pr
                yc = (ya + yb) / 2 + sgn * pr / 2
                j = stone_block("jamb", (wj, sy, z1 - z0 - 0.01), loc=(sx * (R + 0.006 + wj / 2), yc, (z0 + z1) / 2),
                                rot=(0, r.u(-0.3, 0.3), 0), rng=r, chamfer=ch, rough=0.003, detail=0.3)
                uv_box(j, rot90=r.chance(0.5))
                dress(j, r, chips=0.5 if k not in (-1, nj) else 0.25, size=0.06)

    arch_ring(-1, rng.fork("ring_f"))
    arch_ring(1, rng.fork("ring_b"))

    # ------------------------------------------------------------------ corbel tables, cornices, set-offs
    corn = [(-0.05, 0.0), (0.14, 0.0), (0.14, 0.06), (0.125, 0.078), (0.125, 0.115), (0.17, 0.17), (0.19, 0.182), (0.19, 0.232),
            (0.165, 0.262), (-0.05, 0.262)]
    zc0 = zcor - 0.02
    for sgn in (-1, 1):
        x = -xt
        for l in split(2 * xt, r_tr, 0.8, 1.4):
            c = prism("cornice", [(o, zc0 + z) for o, z in corn], x + 0.004, x + l - 0.004,
                      lambda u, v, d, s=sgn: (d, s * (cy + u), v))
            flat(c)
            dress(c, chips=0.35, size=0.05)
            x += l
    hc = 0.2
    cb = prism("corbel", [(-0.04, 0.0), (0.0, 0.0), (0.05, 0.022), (0.095, 0.075), (0.122, 0.13), (0.132, hc), (-0.04, hc)], -0.075, 0.075,
               lambda u, v, d: (d, -u, v), bev=0.007)
    cmats = []
    ncb = max(4, int(round((2 * xt - 0.5) / 0.46)))
    for sgn in (-1, 1):
        for k in range(ncb + 1):
            cmats.append(trs((-xt + 0.25 + (2 * xt - 0.5) * k / ncb, sgn * cy, zc0 - hc), (0, 0, 0 if sgn < 0 else 180),
                             (r_tr.u(0.92, 1.06), 1.0, 1.0)))
    npc = max(2, int(round((xo - xt - 0.3) / 0.42)) + 1)
    for sx in (-1, 1):
        x0, x1 = sorted((sx * xt, sx * xo))
        kk = (0.4, 0.36) if sx > 0 else (0.36, 0.4)
        c = ring_moulding("pier_cornice", x0, x1, -ty3, ty3, [(o, zc0 + z) for o, z in corn], k=(kk[0], kk[1], 1, 1))
        dress(c, chips=0)
        for sgn in (-1, 1):
            for k in range(npc):
                cmats.append(trs((x0 + 0.15 + (x1 - x0 - 0.3) * k / (npc - 1), sgn * ty3, zc0 - hc), (0, 0, 0 if sgn < 0 else 180),
                                 (r_tr.u(0.92, 1.06), 1.0, 1.0)))
            for (zo, so, ri, tyf) in ((zA, o1, ri1, ty2), (zB, o2, ri2, ty3)):      # weathered set-off stones
                xa = x0
                for l in split(x1 - x0, r_tr, 0.6, 1.0):
                    e = r_tr.u(-0.012, 0.012)
                    wz = [(-0.03, 0.0), (so + 0.035, 0.0), (so + 0.035, 0.045 + e), (0.0, ri + e), (-0.03, ri + e)]
                    s_ = prism("setoff", [(o, zo - 0.03 + z) for o, z in wz], xa + 0.004, xa + l - 0.004,
                               lambda u, v, d, s=sgn, tyf=tyf: (d, s * (tyf + u), v), bev=0.009)
                    dress(s_, chips=0.6, size=0.06)
                    xa += l
    paint(cb, grime=0.06, wear=0.1, mode='set')
    co_ = instances_to_mesh("corbels", cb, cmats, keep_base=False)
    assign(co_, m_trim)
    trim.append(co_)

    # carved cartouche over the arch on the town (+Y) face: tablet, shield with a lantern device, crown
    z0c = zc + tr + 0.22
    chh = min(1.25, zc0 - hc - 0.08 - z0c)
    if chh > 0.45:
        def bmap(u, v, d):
            return (-u, cy + d, z0c + v)
        tw = chh * 0.95
        dress(prism("tablet", [(-tw / 2, 0.0), (tw / 2, 0.0), (tw / 2, chh * 0.86), (tw * 0.36, chh), (-tw * 0.36, chh), (-tw / 2, chh * 0.86)],
                    -0.1, 0.05, bmap, bev=0.014), chips=1.0)
        shw, shh = chh * 0.56, chh * 0.6
        shield = [(-shw / 2, shh), (shw / 2, shh), (shw / 2, shh * 0.45), (shw * 0.36, shh * 0.2), (0.0, 0.0), (-shw * 0.36, shh * 0.2), (-shw / 2, shh * 0.45)]
        dress(prism("shield", [(x, chh * 0.07 + z) for x, z in shield], 0.04, 0.1, bmap, bev=0.016), chips=0)
        lk = shh * 0.62
        lan = [(-0.16, 0.0), (0.16, 0.0), (0.2, 0.06), (0.26, 0.5), (0.32, 0.56), (0.07, 0.8), (0.07, 0.9), (-0.07, 0.9), (-0.07, 0.8), (-0.32, 0.56), (-0.26, 0.5), (-0.2, 0.06)]
        dress(prism("device", [(x * lk, chh * 0.07 + shh * 0.3 + z * lk * 0.72) for x, z in lan], 0.09, 0.125, bmap, bev=0.006), chips=0, tone=False)
        ck_ = chh * 0.36
        dress(prism("stone_crown", [(x * ck_, chh * 0.71 + z * ck_ * 0.62) for x, z in CROWN], 0.04, 0.085, bmap, bev=0.008), chips=0, tone=False)
    # guard stones at the foot of the jambs (outside the clear passage width) keep cart wheels off the arch
    for sx in (-1, 1):
        for sgn in (-1, 1):
            gx_, gy_ = sx * (R + 0.25), sgn * (cy + 0.13)
            gs = lathe("guard_stone", [(0.0, 0.0), (0.19, 0.0), (0.185, 0.2), (0.14, 0.5), (0.11, 0.6), (0.06, 0.66), (0.0, 0.67)], segments=12,
                       loc=(gx_, gy_, 0.0), rot=(r_tr.u(-3, 3), r_tr.u(-3, 3), r_tr.u(0, 90)))
            displace_noise(gs, strength=0.008, scale=0.25, seed=p.seed + 50 + sx + 2 * sgn)
            uv_box(gs)
            dress(gs, chips=0)
    for o in trim:
        assign(o, m_trim)
    trim = join(trim, "dressed_stone")
    for o in backs:
        assign(o, m_trim)
    backs = join(backs, "mortar_beds")

    # ------------------------------------------------------------------ passage ceiling: oak beams and boards
    wood = []
    nb = max(3, int(round(chamber / 0.55)))
    nb += 1 - nb % 2
    r_w = rng.fork("wood")
    for k in range(nb):
        y = -(chamber / 2 - 0.2) + (chamber - 0.4) * k / (nb - 1)
        wood.append(tim("ceil_beam", 2 * xw + 0.3, 0.2, 0.22, (0, y, zceil + 0.11 + r_w.u(-0.006, 0.006)), 'X', rot=(r_w.u(-1, 1), 0, r_w.u(-0.3, 0.3)), bev=0.014))
    bd = cbox(-xw - 0.05, xw + 0.05, -chamber / 2 - 0.02, chamber / 2 + 0.02, zceil + 0.22, zceil + 0.28, m_oak, "ceil_boards")
    uv_grain(bd, 'Y')

    # ------------------------------------------------------------------ doors folded open in the rebate
    iron = []
    if p.doors:
        Rr = R + 0.08
        tk_ = 0.07
        yh = -cy + rd + 0.04
        stud = bolt("stud", 0.017, 0.013, kind="pyramid")
        studs = []
        r_d = rng.fork("doors")

        def htop(s):
            return zs + sqrt(max(Rr * Rr - (Rr - s) ** 2, 0.0)) - 0.02

        for side in (-1, 1):
            for k in range(nfold):
                t0 = 0.035 + k * 0.105
                s0 = k * lp

                def ymap(s, k=k, s0=s0):
                    return yh + (s - s0) if k % 2 == 0 else yh + lp - (s - s0)

                def dmap(u, v, d, side=side, ymap=ymap):
                    return (side * (xw - d), ymap(u), v)

                npl = max(2, int(round(lp / 0.24)))
                edges = [s0 + lp * i / npl for i in range(npl + 1)]
                for i in range(npl):
                    sa, sb = edges[i] + 0.002, edges[i + 1] - 0.002
                    top = [(sb - (sb - sa) * t / 3, htop(sb - (sb - sa) * t / 3)) for t in range(4)]
                    dz = r_d.u(0.0, 0.012)
                    pl = prism("door_plank", [(sa, 0.045 + dz), (sb, 0.045 + dz)] + top, t0 + r_d.u(0, 0.004), t0 + tk_ + r_d.u(-0.003, 0.003), dmap, bev=0.005)
                    uv_grain(pl, 'Z')
                    wood.append(pl)
                if k != nfold - 1:
                    continue
                tf = t0 + tk_
                zstr = [0.42, zs * 0.5 + 0.1, zs - 0.28] if zs > 2.1 else [0.42, zs - 0.28]
                Ls = lp * 0.8
                for z_ in zstr:
                    w = 0.036
                    st = [(s0, z_ - w), (s0 + Ls * 0.84, z_ - w * 0.75), (s0 + Ls * 0.9, z_ - w * 1.7), (s0 + Ls, z_),
                          (s0 + Ls * 0.9, z_ + w * 1.7), (s0 + Ls * 0.84, z_ + w * 0.75), (s0, z_ + w)]
                    iron.append(prism("strap", st, tf, tf + 0.008, dmap, bev=0.002))
                    for i in range(npl):
                        sm = (edges[i] + edges[i + 1]) / 2
                        if sm < s0 + Ls * 0.86:
                            studs.append(trs((side * (xw - tf - 0.008), ymap(sm), z_), (0, -90 * side, 0)))
                    iron.append(cylinder("knuckle", 0.026, 0.15, loc=(side * (xw - tf - 0.004), ymap(s0) - (0.03 if k % 2 == 0 else -0.03), z_), verts=8))
                    if nfold == 1:
                        iron.append(box("pintle", (rc - 0.02, 0.05, 0.05), loc=(side * (xw - (rc - 0.02) / 2 + 0.01), yh - 0.035, z_ - 0.1)))
                for z_ in [zz for zz in (1.0, 1.55, zs + 0.25) if all(abs(zz - q) > 0.2 for q in zstr)]:
                    for i in range(npl):
                        sm = (edges[i] + edges[i + 1]) / 2
                        if z_ < htop(sm) - 0.15:
                            studs.append(trs((side * (xw - tf), ymap(sm), z_), (0, -90 * side, 0), 0.8))
                # ring pull on a rose near the meeting stile
                sp_ = s0 + lp - 0.2
                iron.append(cylinder("pull_rose", 0.05, 0.012, loc=(side * (xw - tf - 0.006), ymap(sp_), 1.25), rot=(0, 90, 0), verts=10))
                iron.append(torus("pull_ring", 0.075, 0.01, loc=(side * (xw - tf - 0.022), ymap(sp_), 1.18), rot=(0, 90, 0), major_seg=14, minor_seg=5))
        iron.append(instances_to_mesh("door_studs", stud, studs, keep_base=False))

    # ------------------------------------------------------------------ sign board over the arch
    zb = zc + 0.3
    avail = (zc0 - hc - 0.06) - zb
    sh, rise = min(1.0, 0.7 * avail), min(0.42, 0.28 * avail)
    sw = min(2 * R + 0.7, 2 * xt - 1.5)
    ys = -cy - 0.1

    def outline(ins, n=30):
        hw, cr = sw / 2 - ins, 0.07
        pts = [(-hw + cr - cr * sin(radians(a)), zb + ins + cr - cr * cos(radians(a))) for a in (90, 60, 30, 0)]
        pts += [(hw - cr + cr * sin(radians(a)), zb + ins + cr - cr * cos(radians(a))) for a in (0, 30, 60, 90)]
        for i in range(n + 1):
            x = hw - 2 * hw * i / n
            pts.append((x, zb + sh - ins + rise * (1 - smoothstep(0.17 * sw, 0.41 * sw, abs(x)))))
        return pts

    board = prism("sign_board", outline(0.0), ys - 0.05, ys, XZ, bev=0.006)
    navy(board)
    paint(board, special=0.15, grime=lambda co, n: np.clip((co[:, 2] - (zb + sh * 0.75)) / 0.6, 0, 1) * 0.35)
    gilt = []
    fr = sweep("sign_frame", [(x, ys - 0.05, z) for x, z in outline(0.05)], rounded_rect(0.075, 0.05, 0.016, 2), closed=True,
               smooth_path=False, up=(0, -1, 0))
    gilt.append(fr)
    body = str(p.sign_text).strip()
    if body:
        tsz = sh * 0.46
        zt_ = zb + sh * 0.45
        t = text("sign_text", body, size=tsz, depth=0.03, loc=(0, ys - 0.05, zt_), font="serif", bevel=0.002)
        mn, mx = world_bbox([t])
        wt = mx.x - mn.x
        if wt > sw - 0.55:
            delete([t])
            t = text("sign_text", body, size=tsz * (sw - 0.55) / wt, depth=0.03, loc=(0, ys - 0.05, zt_), font="serif", bevel=0.002)
        gilt.append(t)
    cwid = min(0.62, sw * 0.16)
    zcr = zb + sh * 0.84
    gilt.append(prism("sign_crown", [(x * cwid, zcr + z * cwid) for x, z in CROWN], ys - 0.075, ys - 0.05, XZ, bev=0.005))
    for (x, z) in ((0.5, 0.52), (0.0, 0.62), (-0.5, 0.52)):
        gilt.append(sphere("crown_ball", 0.07 * cwid, loc=(x * cwid, ys - 0.064, zcr + (z + 0.05) * cwid), segments=10, rings=6))
    gilt.append(box("crown_band", (0.9 * cwid, 0.036, 0.09 * cwid), loc=(0, ys - 0.062, zcr + 0.045 * cwid), bevel=0.004, segments=1))
    dome = bolt("sign_bolt", 0.02, 0.012, kind="dome")
    bm_ = [trs((sx * (sw / 2 - 0.05), ys - 0.07, zb + f * sh), (90, 0, 0)) for sx in (-1, 1) for f in (0.12, 0.5, 0.88)]
    iron.append(instances_to_mesh("sign_bolts", dome, bm_, keep_base=False))
    for sx in (-1, 1):                         # iron hold-fasts gripping the shoulders
        for f in (0.75, 0.2):
            iron.append(box("holdfast", (0.05, 0.14, 0.16), loc=(sx * (sw / 2 + 0.012), ys + 0.02, zb + sh * f), bevel=0.004, segments=1))

    # ------------------------------------------------------------------ lanterns (modelled inline: frame, glass, burner, flame)
    glassy, flames, soot, rusts = [], [], [], []

    def lantern_head(x, y, z0, lw):
        """Tapered square lantern standing on z0. Returns (flame position, top of the finial)."""
        wb, hb = lw * 0.68, lw * 1.25
        z1 = z0 + 0.016 + hb
        iron.append(box("lh_base", (wb + 0.035, wb + 0.035, 0.016), loc=(x, y, z0 + 0.008), bevel=0.004, segments=1))
        for sx in (-1, 1):
            for sy in (-1, 1):
                iron.append(bar("lh_bar", (x + sx * wb / 2, y + sy * wb / 2, z0 + 0.012), (x + sx * lw / 2, y + sy * lw / 2, z1 + 0.004), 0.013))
        iron.append(box("lh_top", (lw + 0.035, lw + 0.035, 0.016), loc=(x, y, z1 + 0.008), bevel=0.004, segments=1))
        hr_ = lw * 0.4
        iron.append(cone("lh_roof", (lw + 0.09) * 0.7071, 0.05, hr_, loc=(x, y, z1 + 0.016 + hr_ / 2), rot=(0, 0, 45), verts=4))
        iron.append(cylinder("lh_vent", 0.04, 0.035, loc=(x, y, z1 + 0.016 + hr_ + 0.012), verts=8))
        iron.append(cone("lh_cap", 0.062, 0.012, 0.04, loc=(x, y, z1 + 0.016 + hr_ + 0.048), verts=8))
        iron.append(sphere("lh_finial", 0.02, loc=(x, y, z1 + hr_ + 0.098), segments=8, rings=5))
        gi = 0.005
        cs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        gv = [(x + sx * (wb / 2 - gi), y + sy * (wb / 2 - gi), z0 + 0.018) for sx, sy in cs]
        gv += [(x + sx * (lw / 2 - gi), y + sy * (lw / 2 - gi), z1 - 0.002) for sx, sy in cs]
        g = mesh_object("lh_glass", gv, [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)])
        flat(g)
        glassy.append(g)
        zf_ = z0 + 0.016 + hb * 0.42
        iron.append(cylinder("lh_burner", 0.016, hb * 0.26, loc=(x, y, z0 + 0.016 + hb * 0.13), verts=8, radius2=0.01))
        flames.append(sphere("lh_flame", lw * 0.095, loc=(x, y, zf_), segments=10, rings=6, scale=(1, 1, 1.8)))
        return (x, y, zf_), z1 + hr_ + 0.118

    def head_height(lw):
        return 0.016 + lw * 1.25 + lw * 0.4 + 0.118

    xl = (R + tr + 0.07 + xt) / 2
    zl = zs + 0.3 * R
    li = 0

    def wall_lantern(x, sgn):
        nonlocal li
        reach, lw = 0.52, 0.3
        yo = lambda d: sgn * (cy + d)               # d = distance out from the curtain wall
        iron.append(box("wl_plate", (0.08, 0.016, 0.5), loc=(x, yo(0.016), zl - 0.12), bevel=0.004, segments=1))
        for dz in (0.08, -0.32):
            iron.append(box("wl_fixing", (0.03, 0.02, 0.03), loc=(x, yo(0.03), zl + dz), rot=(0, 45, 0)))
        iron.append(box("wl_arm", (0.026, reach + 0.05, 0.026), loc=(x, yo(0.016 + (reach + 0.05) / 2), zl), bevel=0.004, segments=1))
        brace = bezier_path((x, yo(0.024), zl - 0.34), (x, yo(0.2), zl - 0.36), (x, yo(reach * 0.92), zl - 0.3), (x, yo(reach * 0.84), zl - 0.012), n=9)
        iron.append(sweep("wl_brace", brace, 0.009, sides=6))
        vol = spiral_path(0.085, 0.02, turns=1.3, n=16, a0_deg=90, plane='YZ', center=(x, yo(reach * 0.36), zl - 0.1), ccw=sgn < 0)
        iron.append(sweep("wl_scroll", vol, 0.0075, sides=5))
        yt = yo(0.016 + reach)
        iron.append(cylinder("wl_collar", 0.028, 0.05, loc=(x, yt, zl + 0.022), verts=8))
        iron.append(cone("wl_drop", 0.02, 0.004, 0.06, loc=(x, yt, zl - 0.042), rot=(180, 0, 0), verts=8))
        fpos, _ = lantern_head(x, yt, zl + 0.045, lw)
        light("lantern_%d" % li, fpos, intensity=p.lantern_intensity, range=8.0, flicker=0.14, shadow_near=0.34)
        soot.append((x, sgn * cy, zl + 0.5))
        rusts.append((x, sgn * cy, zl - 0.37, 0.9))
        li += 1

    for sx in (-1, 1):
        wall_lantern(sx * xl, -1)
        if p.back_lanterns:
            wall_lantern(sx * xl, 1)

    # passage lantern hung from the centre beam on a rod and hook
    lwp = 0.21
    zlt = zceil - 0.1
    fpos, ztop = lantern_head(0, 0, zlt - head_height(lwp), lwp)
    iron.append(cylinder("pl_rod", 0.007, zceil - ztop + 0.03, loc=(0, 0, (zceil + ztop) / 2), verts=6))
    iron.append(torus("pl_ring", 0.022, 0.005, loc=(0, 0, ztop + 0.012), rot=(90, 0, 0), major_seg=10, minor_seg=5))
    iron.append(box("pl_plate", (0.09, 0.09, 0.012), loc=(0, 0, zceil - 0.006)))
    light("passage", fpos, intensity=34, range=9.0, shadow_near=0.3)

    # sign lamp: a cage lantern hung well out in front of the board on a stayed wrought-iron arm
    lws = 0.2
    yfix = -(gd / 2) if p.gatehouse else -py                 # jetty sill beam / breast wall
    ytip = min(ys - 0.9, yfix - 0.35)
    zm = zdeck + (0.31 if p.gatehouse else 0.2)
    iron.append(box("arm", (0.03, yfix - ytip + 0.05, 0.03), loc=(0, (yfix + ytip) / 2 - 0.025, zm), bevel=0.004, segments=1))
    iron.append(bar("arm_stay", (0, yfix - 0.008, zm + 0.5), (0, ytip + 0.06, zm + 0.012), 0.016))
    iron.append(box("arm_plate", (0.08, 0.016, 0.18), loc=(0, yfix - 0.008, zm), bevel=0.004, segments=1))
    iron.append(box("arm_plate", (0.07, 0.016, 0.11), loc=(0, yfix - 0.008, zm + 0.5), bevel=0.004, segments=1))
    iron.append(sweep("arm_scroll", spiral_path(0.09, 0.022, turns=1.25, n=14, a0_deg=90, plane='YZ', center=(0, yfix - 0.13, zm + 0.105), ccw=True), 0.0075, sides=5))
    iron.append(sweep("arm_curl", spiral_path(0.05, 0.015, turns=1.1, n=10, a0_deg=-90, plane='YZ', center=(0, ytip - 0.02, zm + 0.065), ccw=False), 0.007, sides=5))
    zst = zm - 0.015 - 0.11
    fpos, ztop = lantern_head(0, ytip, zst - head_height(lws), lws)
    iron.append(cylinder("sl_rod", 0.006, zm - ztop, loc=(0, ytip, (zm + ztop) / 2), verts=6))
    iron.append(torus("sl_ring", 0.02, 0.005, loc=(0, ytip, ztop + 0.01), rot=(0, 90, 0), major_seg=10, minor_seg=5))
    light("sign_lamp", (0, ytip + 0.02, fpos[2]), color="#ffc880", intensity=20, range=6.0, flicker=0.1, shadow_near=0.3)
    for o in glassy:
        assign(o, m_gl)
    for o in flames:
        assign(o, m_fl)

    # ------------------------------------------------------------------ banners on the piers
    if p.banners:
        zrod = zc0 - hc - 0.2
        bwid = min(0.95, bw - 0.38)
        Lb = clamp(zrod - (zB + ri2) - 0.12, 0.9, 2.6)
        notch = min(0.24, Lb * 0.16)
        r_b = rng.fork("banners")
        bi = 0
        for sx in (-1, 1):
            for sgn in (-1, 1):
                xc = sx * xpc
                yr = sgn * (ty3 + 0.15)
                ph = r_b.u(0, 6.28)

                def yfold(u, v, ph=ph, yr=yr):
                    return yr + v * 0.03 * sin(u * 3 * pi + ph) + 0.014 * v * sin(v * 6 + ph)

                def vlen(u):
                    return Lb - notch * (1 - abs(2 * u - 1))

                def cl(u, v, off=0.0, xc=xc, sgn=sgn, yfold=yfold):
                    wv = bwid * (1 - 0.05 * v)
                    return (xc + sgn * (u - 0.5) * wv, yfold(u, v) + sgn * off, zrod - 0.02 - v * vlen(u))

                c = surface("banner", cl, 12, 16, thickness=0.004)
                navy(c, xc)
                paint(c, special=mask_mul(noise_mask(0.5, seed=p.seed + bi, lo=0.3, hi=0.8), 0.4),
                      grime=lambda co, n, z0=zrod - Lb: np.clip(1 - (co[:, 2] - z0) / 0.5, 0, 1) * 0.45)
                parts = [c]
                # appliqued gold crown, ~62% of the banner width: head band, five-pointed circlet, pearls
                cw_ = bwid * 0.62
                ch_ = cw_ * 0.8
                vc = 0.42

                def cpt(a, b, d=0.0035, xc=xc, sgn=sgn, yfold=yfold):
                    u = 0.5 + sgn * a / bwid
                    zc_ = zrod - 0.02 - vc * Lb + ch_ * 0.5
                    z = zc_ - ch_ + b
                    v = clamp((zrod - 0.02 - z) / vlen(u), 0.0, 1.0)
                    return (xc + a * (1 - 0.05 * v), yfold(u, v) + sgn * d, z)

                nodes = [(-0.5, 0.86), (-0.375, 0.5), (-0.25, 0.8), (-0.125, 0.5), (0.0, 1.0), (0.125, 0.5), (0.25, 0.8), (0.375, 0.5), (0.5, 0.86)]

                def top(t):
                    for (ta, ha), (tb, hb_) in zip(nodes[:-1], nodes[1:]):
                        if ta <= t <= tb:
                            return ha + (hb_ - ha) * (t - ta) / (tb - ta)
                    return 0.5

                ncol = 24
                V, F = [], []
                rows_ = ((0.0, 0.15), (0.2, None))
                for i in range(ncol + 1):
                    t_ = -0.5 + i / ncol
                    spread = 1.0 + 0.12 * 0                         # straight sided circlet
                    V += [cpt(t_ * cw_ * 0.92, 0.0), cpt(t_ * cw_ * 0.92, 0.15 * ch_), cpt(t_ * cw_ * spread, 0.2 * ch_),
                          cpt(t_ * cw_ * (1.0 + 0.1 * (top(t_) - 0.2)), (0.2 + (top(t_) - 0.2) * 0.5) * ch_),
                          cpt(t_ * cw_ * (1.0 + 0.2 * (top(t_) - 0.2)), top(t_) * ch_)]
                for i in range(ncol):
                    a_, b_ = i * 5, (i + 1) * 5
                    F += [(a_, b_, b_ + 1, a_ + 1), (a_ + 2, b_ + 2, b_ + 3, a_ + 3), (a_ + 3, b_ + 3, b_ + 4, a_ + 4)]
                for (t_, h_) in nodes[::2]:                        # pearls on the points
                    cx_, cz_ = t_ * cw_ * (1.0 + 0.2 * (h_ - 0.2)), (h_ + 0.07) * ch_
                    k0 = len(V)
                    V.append(cpt(cx_, cz_, 0.004))
                    for q in range(8):
                        V.append(cpt(cx_ + 0.055 * cw_ * cos(q * pi / 4), cz_ + 0.055 * cw_ * sin(q * pi / 4), 0.004))
                    F += [(k0, k0 + 1 + q, k0 + 1 + (q + 1) % 8) for q in range(8)]
                parts.append(gold(sheet("banner_crown", V, F, (0, sgn, 0)), xc))
                for (v0, v1) in ((0.035, 0.065), (0.875, 0.9)):
                    s_ = surface("banner_band", lambda u, v, v0=v0, v1=v1, cl=cl: cl(u, v0 + (v1 - v0) * v, 0.0036), 12, 1)
                    s2_ = surface("banner_band", lambda u, v, v0=v0, v1=v1, cl=cl: cl(1 - u, v0 + (v1 - v0) * v, -0.0036), 12, 1)
                    parts += [gold(s_, xc), gold(s2_, xc)]
                sl = cylinder("banner_sleeve", 0.021, bwid, loc=(xc, yr, zrod), rot=(0, 90, 0), verts=8)
                parts.append(navy(sl, xc))
                part(parts, "banner_%d" % bi, pivot=(xc, yr, zrod), motion="flutter", amp=0.02, speed=0.5)
                bi += 1
                iron.append(cylinder("banner_rod", 0.013, bwid + 0.26, loc=(xc, yr, zrod), rot=(0, 90, 0), verts=8))
                for ex in (-1, 1):
                    iron.append(sphere("rod_finial", 0.03, loc=(xc + ex * (bwid / 2 + 0.14), yr, zrod), segments=8, rings=5))
                    iron.append(cylinder("rod_bracket", 0.011, 0.18, loc=(xc + ex * (bwid / 2 + 0.07), sgn * (ty3 + 0.07), zrod), rot=(90, 0, 0), verts=6))
                    iron.append(box("rod_plate", (0.07, 0.03, 0.11), loc=(xc + ex * (bwid / 2 + 0.07), sgn * (ty3 + 0.008), zrod)))
                    rusts.append((xc + ex * (bwid / 2 + 0.07), sgn * ty3, zrod - 0.05, 0.7))

    # ------------------------------------------------------------------ notice board with placard and bills (left pier, approach side)
    if p.posters:
        r_n = rng.fork("posters")
        nbw = min(1.2, bw - PL - 0.24)
        nz0 = 0.78
        nbh = min(1.36, zA - nz0 - 0.16)
        xc = -xpc
        yf = -ty1 - 0.012
        npk = 5
        for i in range(npk):
            wood.append(tim("nb_plank", nbh + r_n.u(-0.02, 0.02), nbw / npk - 0.004, 0.028, (xc + (i - (npk - 1) / 2) * nbw / npk, yf - 0.036, nz0 + nbh / 2),
                            'Z', bev=0.004))
        for z_ in (nz0 + 0.07, nz0 + nbh - 0.07):
            wood.append(tim("nb_rail", nbw + 0.08, 0.03, 0.085, (xc, yf - 0.064, z_), 'X', bev=0.005))
        wood.append(tim("nb_hood", nbw + 0.16, 0.17, 0.025, (xc, yf - 0.09, nz0 + nbh + 0.035), 'X', rot=(-22, 0, 0), bev=0.004))
        nails = [trs((xc + sx * (nbw / 2 - 0.02), yf - 0.079, z_), (90, 0, 0)) for sx in (-1, 1) for z_ in (nz0 + 0.07, nz0 + nbh - 0.07)]
        yp = yf - 0.0515

        def paper(name, cx, cz, w, h, tilt, yoff, curl, nu=5, nv=6):
            ca, sa = cos(radians(tilt)), sin(radians(tilt))

            def fn(u, v):
                lx, lz = (0.5 - u) * w, (0.5 - v) * h
                c_ = curl * (v ** 3) * (abs(2 * u - 1) ** 2) + 0.0012 * sin(u * 9 + v * 7)
                return (cx + lx * ca - lz * sa, yp - yoff - c_, cz + lx * sa + lz * ca)
            o = surface(name, fn, nu, nv, thickness=0.0014)
            for (u, v) in ((0.05, 0.04), (0.95, 0.04), (0.05, 0.96), (0.95, 0.96)):
                q = fn(u, v)
                nails.append(trs((q[0], q[1] - 0.001, q[2]), (90, 0, 0), 0.6))
            return o

        # big printed placard in the town livery: gold lettering and crowns on the livery colour
        inner = nbh - 0.3
        pwd, phh = nbw * 0.9, inner * (0.98 if nbw < 0.75 else 0.74)
        pcx, pcz = xc, nz0 + 0.15 + inner - phh / 2
        pl_ = navy(paper("placard", pcx, pcz, pwd, phh, 0.0, 0.003, 0.012), pcx)
        paint(pl_, grime=lambda co, n: np.clip(1 - (co[:, 2] - (pcz - phh / 2)) / 0.3, 0, 1) * 0.4, special=0.2)
        yt_ = yp - 0.0052
        words = str(p.poster_text).split()
        maxc = 8 if pwd > 0.7 else 7
        lines, cur = [], ""
        for w_ in words:
            if cur and len(cur) + 1 + len(w_) > maxc:
                lines.append(cur)
                cur = w_
            else:
                cur = (cur + " " + w_).strip()
        if cur:
            lines.append(cur)
        lines = lines[:4]
        ck = min(pwd * 0.26, phh * 0.24)
        if lines:
            band = phh * 0.6
            lh = min(0.27, band / len(lines))
            ztop_ = pcz + phh * 0.13 + lh * len(lines) / 2
            for i, ln in enumerate(lines):
                z_ = ztop_ - (i + 0.5) * lh
                t = text("placard_text", ln, size=lh * 0.8, depth=0.0015, loc=(pcx, yt_, z_), font="sans")
                mn, mx = world_bbox([t])
                if mx.x - mn.x > pwd * 0.88:
                    delete([t])
                    t = text("placard_text", ln, size=lh * 0.8 * pwd * 0.88 / (mx.x - mn.x), depth=0.0015, loc=(pcx, yt_, z_), font="sans")
                gold(t, pcx)
            zk = pcz - phh / 2 + phh * 0.06
            gold(prism("placard_crown", [(pcx + x * ck, zk + z * ck) for x, z in CROWN], yt_ - 0.0015, yt_, XZ), pcx)
            gold(box("placard_rule", (pwd * 0.8, 0.0015, 0.012), loc=(pcx, yt_ - 0.0007, zk + ck * 0.62 + 0.04)), pcx)
            gold(box("placard_rule", (pwd * 0.8, 0.0015, 0.012), loc=(pcx, yt_ - 0.0007, pcz + phh / 2 - 0.045)), pcx)
        if nbw >= 0.75:                                           # two small hand bills pasted below
            for (fx_, wb_, tl) in ((-0.24, 0.4, 3.0), (0.25, 0.36, -4.0)):
                hb_ = inner - phh - 0.04
                if hb_ < 0.12:
                    break
                bcx, bcz = xc + fx_ * nbw, nz0 + 0.16 + hb_ / 2
                b_ = paper("bill", bcx, bcz, nbw * wb_, hb_, tl, 0.0, 0.02, 4, 4)
                assign(b_, m_pl)
                paint(b_, grime=0.3 + r_n.u(0, 0.15), special=mask_mul(noise_mask(0.15, seed=p.seed + int(fx_ * 10), lo=0.4, hi=0.7), 0.7))
                for k in range(4):                                 # lines of print
                    ln_ = box("bill_print", (nbw * wb_ * r_n.u(0.5, 0.8), 0.001, 0.012 if k else 0.022),
                              loc=(bcx, yp - 0.0022, bcz + hb_ * (0.3 - 0.19 * k)), rot=(0, -tl, 0))
                    iron.append(ln_)
        nl = bolt("nb_nail", 0.008, 0.004, kind="pyramid")
        iron.append(instances_to_mesh("nb_nails", nl, nails, keep_base=False))

    # ------------------------------------------------------------------ gatehouse room
    plaster_o = None
    if p.gatehouse:
        r_g = rng.fork("gatehouse")
        gw = 2 * xt - 0.16
        gx, gy = gw / 2, gd / 2
        zf = zdeck + 0.2
        wh = 2.3
        nj_ = max(4, int(round(gw / 0.5)))
        for k in range(nj_ + 1):
            x = -gx + 0.07 + (gw - 0.14) * k / nj_
            wood.append(tim("joist", gd - 0.02 + r_g.u(-0.02, 0.03), 0.14, 0.2, (x, 0, zdeck + 0.1), 'Y', rot=(0, r_g.u(-1.5, 1.5), 0), bev=0.014))
        fb = cbox(-gx + 0.02, gx - 0.02, -gy + 0.04, gy - 0.04, zdeck + 0.165, zdeck + 0.2, m_oak, "floor_boards")
        uv_grain(fb, 'X')
        pb = cbox(-gx + 0.045, gx - 0.045, -gy + 0.045, gy - 0.045, zf + 0.05, zf + wh, m_pl, "plaster")
        densify(pb, 0.2)
        displace_noise(pb, strength=0.004, scale=0.6, seed=p.seed + 3)
        plaster_o = pb
        ww, whh = 1.05, 1.2
        zsill = zf + 0.82
        xs = [-gw * 0.25, gw * 0.25] if gw < 7.6 else [-gw * 0.31, 0.0, gw * 0.31]
        zrail = zsill - 0.07
        tt = 0.14
        for sgn in (-1, 1):
            yc = sgn * (gy - tt / 2)
            wood.append(plank("sill_beam", gw, 0.2, 0.22, loc=(0, sgn * (gy - 0.09), zf + 0.11), bevel=0.014, axis='X', rng=r_g, warp=0.006))
            wood.append(plank("top_plate", gw, tt + 0.02, 0.18, loc=(0, yc, zf + wh - 0.09), bevel=0.01, axis='X', rng=r_g, warp=0.005))
            wood.append(tim("mid_rail", gw - 0.3, tt, 0.13, (0, yc, zrail), 'X', bev=0.008))
            nst = max(4, int(round(gw / 0.6)))
            for k in range(1, nst):
                x = -gx + gw * k / nst + r_g.u(-0.03, 0.03)
                inwin = any(abs(x - xw_) < ww / 2 + 0.16 for xw_ in xs)
                z0, z1 = zf + 0.22, (zrail - 0.065 if inwin else zf + wh - 0.18)
                if inwin and min(abs(x - xw_) for xw_ in xs) > ww / 2 - 0.1:
                    continue
                if abs(x) > gx - 1.05:
                    continue
                wood.append(tim("stud", z1 - z0, r_g.u(0.12, 0.15), tt, (x, yc, (z0 + z1) / 2), 'Z', rot=(0, r_g.u(-0.6, 0.6), 0), bev=0.008))
            for sx in (-1, 1):
                wood.append(tim("corner_post", wh - 0.22, 0.19, 0.19, (sx * (gx - 0.095), sgn * (gy - 0.095), zf + 0.22 + (wh - 0.22) / 2), 'Z', bev=0.012))
                run, rs = 0.8, zrail - 0.065 - (zf + 0.22)                # knee brace from the sill up to the corner post
                bl_ = hypot(run, rs) + 0.1
                wood.append(tim("brace", bl_, tt - 0.01, 0.13, (sx * (gx - 0.19 - run / 2), yc, zf + 0.22 + rs / 2), 'X',
                                rot=(0, -sx * degrees(atan2(rs, run)), 0), bev=0.008))
                xq = sx * (gx - 0.19 - run)
                if not any(abs(xq - xw_) < ww / 2 + 0.16 for xw_ in xs):
                    wood.append(tim("stud", wh - 0.4, 0.14, tt, (xq, yc, zf + 0.22 + (wh - 0.4) / 2), 'Z', bev=0.008))
        for sx in (-1, 1):
            xc_ = sx * (gx - tt / 2)
            wood.append(tim("end_sill", gd - 0.38, 0.2, 0.22, (sx * (gx - 0.09), 0, zf + 0.11), 'Y', bev=0.014))
            wood.append(tim("end_plate", gd - 0.38, tt + 0.02, 0.18, (xc_, 0, zf + wh - 0.09), 'Y', bev=0.01))
            wood.append(tim("end_rail", gd - 0.38, tt, 0.13, (xc_, 0, zf + wh * 0.5), 'Y', bev=0.008))
            for k in (1, 2, 3):
                wood.append(tim("end_stud", wh - 0.4, tt, 0.14, (xc_, -gy + gd * k / 4, zf + 0.22 + (wh - 0.4) / 2), 'Z', rot=(0, 0, 90), bev=0.008))

        # leaded casements: oak frame and mullion, lit glazing behind diamond cames
        def window(xc_, sgn):
            fw, fd = 0.075, 0.115
            yo = lambda d: sgn * (gy + d)
            for sx in (-1, 1):
                wood.append(tim("w_jamb", whh, fw, fd, (xc_ + sx * (ww / 2 - fw / 2), yo(0.014 - fd / 2), zsill + whh / 2), 'Z', bev=0.006))
            wood.append(tim("w_head", ww + 0.12, fd + 0.02, fw + 0.02, (xc_, yo(0.018 - fd / 2), zsill + whh - fw / 2 + 0.01), 'X', bev=0.008))
            wood.append(tim("w_sill", ww + 0.18, fd + 0.08, 0.07, (xc_, yo(0.05 - fd / 2), zsill + 0.012), 'X', rot=(sgn * 7, 0, 0), bev=0.008))
            wood.append(tim("w_mullion", whh - fw, 0.065, fd - 0.02, (xc_, yo(0.008 - fd / 2), zsill + whh / 2), 'Z', bev=0.006))
            wood.append(tim("w_transom", ww - fw, fd - 0.04, 0.05, (xc_, yo(-0.005 - fd / 2 + 0.02), zsill + whh * 0.68), 'X', bev=0.005))
            yg = yo(-0.034)
            za, zb_ = zsill + 0.045, zsill + whh - fw + 0.005
            V, F, GV, GF = [], [], [], []
            s = 0.105 * 1.4142
            wc, rdg = 0.0045, 0.005
            for sx in (-1, 1):
                xa, xb = sorted((xc_ + sx * 0.03, xc_ + sx * (ww / 2 - fw + 0.005)))
                k0 = len(GV)
                GV += [(xa, yg, za), (xb, yg, za), (xb, yg, zb_), (xa, yg, zb_)]
                GF.append((k0, k0 + 1, k0 + 2, k0 + 3))
                for d in (1, -1):
                    c0, c1 = (xa - zb_, xb - za) if d > 0 else (xa + za, xb + zb_)
                    c = c0 + s * 0.5
                    while c < c1:
                        if d > 0:
                            x0, x1 = max(xa, za + c), min(xb, zb_ + c)
                            A, B = (x0, x0 - c), (x1, x1 - c)
                        else:
                            x0, x1 = max(xa, c - zb_), min(xb, c - za)
                            A, B = (x0, c - x0), (x1, c - x1)
                        c += s
                        if x1 - x0 < 0.01:
                            continue
                        px, pz = wc * 0.7071, -d * wc * 0.7071
                        k0 = len(V)
                        for (qx, qz) in (A, B):
                            V += [(qx - px, yg + sgn * 0.001, qz - pz), (qx, yg + sgn * rdg, qz), (qx + px, yg + sgn * 0.001, qz + pz)]
                        F += [(k0, k0 + 3, k0 + 4, k0 + 1), (k0 + 1, k0 + 4, k0 + 5, k0 + 2)]
            cames = sheet("w_cames", V, F, (0, sgn, 0))
            flat(cames)
            iron.append(cames)
            gl = sheet("w_glass", GV, GF, (0, sgn, 0))
            assign(gl, m_lit)
            uv_box(gl)

        for xw_ in xs:
            for sgn in (-1, 1):
                window(xw_, sgn)
        roof = use("roof_gable", width=gw, depth=gd, pitch=50.0, type=p.roof, ridge="x", overhang=0.38, covering=p.covering,
                   gable_wall="timber", attic_window="none", rooflight="none", eaves="rafters", gutter="none", sag=0.07,
                   wall_color=p.plaster, timber_color=p.timber, trim_color=p.timber, moss=p.moss, seed=p.seed + 2)
        ro = roof.place((0, 0, zf + wh), sockets=False)
        if ro.data.users > 1:
            ro.data = ro.data.copy()
        cover = None
        for sl in ro.material_slots:
            if sl.material is not None and sl.material.get("tsq_lib", sl.material.name.split("~")[0]) == p.covering and cover is None:
                cover = sl.material
        remap = {"wood_oak": m_oak, "wood_endgrain": m_oak, "wood_painted": m_oak, "wood_weathered": m_oak, "plaster_lime": m_pl, "zinc": m_iron,
                 "lead": m_iron, "iron_wrought": m_iron, "iron_rusty": m_iron, "iron_painted": m_iron, "mortar": m_pl, "terracotta": cover,
                 "roof_tile_clay": cover, "roof_slate": cover, "wood_shingle": cover}
        for i, sl in enumerate(ro.material_slots):
            if sl.material is None or sl.material == cover:
                continue
            lib = sl.material.get("tsq_lib", sl.material.name.split("~")[0])
            ro.data.materials[i] = remap.get(lib) or cover or m_oak      # ridge / hip pieces share the covering material
        if m_moss is not None:                         # moss cushions in the tile laps, thicker towards the eaves and on the north slope
            mw = np.array(world_matrix(ro), dtype=np.float64)
            me = ro.data
            ci = [i for i, sl in enumerate(ro.material_slots) if sl.material == cover]
            cand, wts = [], []
            for pg in me.polygons:
                if pg.material_index not in ci:
                    continue
                nw = mw[:3, :3] @ np.array(pg.normal)
                if nw[2] < 0.35 or pg.area < 1e-4:
                    continue
                cw = mw[:3, :3] @ np.array(pg.center) + mw[:3, 3]
                cand.append((pg.index, nw))
                wts.append(pg.area * (1.7 if cw[1] > 0 else 1.0) * clamp(1.5 - (cw[2] - (zf + wh)) / 1.6, 0.35, 1.5))
            if cand:
                g_ = np.random.default_rng(sd + 77)
                wts = np.array(wts) / sum(wts)
                for k in g_.choice(len(cand), size=int(gw * gd * 2.6 * p.moss + 2), p=wts):
                    pg = me.polygons[cand[k][0]]
                    vs = [mw[:3, :3] @ np.array(me.vertices[v].co) + mw[:3, 3] for v in pg.vertices]
                    q = int(g_.integers(1, len(vs) - 1))
                    a_, b_ = g_.random(), g_.random()
                    if a_ + b_ > 1:
                        a_, b_ = 1 - a_, 1 - b_
                    pt = vs[0] + a_ * (vs[q] - vs[0]) + b_ * (vs[q + 1] - vs[0])
                    nw = cand[k][1]
                    moss_at(tuple(pt + nw * 0.012), tuple(nw), g_.uniform(0.08, 0.26), g_.uniform(0.04, 0.09), g_.uniform(0.02, 0.04))
        socket("chimney", "attach", (gx * 0.5, 0, zf + wh), role="chimney")

        # limewash: drips under the sills, dirt above the sill beam and along the rails, soot under the eaves
        def pl_grime(co, n):
            x, z = co[:, 0], co[:, 2]
            t = np.where(np.abs(n[:, 1]) >= np.abs(n[:, 0]), x, co[:, 1] + 31.7)
            st = np.clip((n1(t, 0.2, sd + 20) - 0.45) / 0.3, 0, 1)
            g = np.clip(1 - (z - zf - 0.2) / 0.3, 0, 1) * (0.25 + 0.3 * st) + np.clip((z - (zf + wh - 0.5)) / 0.3, 0, 1) * 0.22
            g = np.maximum(g, np.clip(1 - np.abs(z - zrail - 0.12) / 0.14, 0, 1) * 0.2 * st)
            for xw_ in xs:
                under = (np.abs(x - xw_) < ww / 2 + 0.12) * (z < zsill) * (np.abs(n[:, 1]) > 0.5)
                g = np.maximum(g, under * np.clip(1 - (zsill - z) / 0.7, 0, 1) * (0.15 + 0.5 * st))
            return np.clip(g, 0, 1)

        paint(pb, grime=pl_grime)

    # ------------------------------------------------------------------ materials and weathering
    for o in wood:
        assign(o, m_oak)
    wood = join(wood, "carpentry")
    paint(wood, grime=lambda co, n: np.clip(1 - co[:, 2] / 0.7, 0, 1) * 0.6)
    for o in iron:
        assign(o, m_iron)
    iron = join(iron, "ironwork")
    paint(iron, special=mask_mul(noise_mask(0.12, seed=p.seed + 6, lo=0.4, hi=0.75), 0.5))
    for o in gilt:
        assign(o, m_gilt)
    gilt = join(gilt, "gilding")
    paint(gilt, grime=mask_mul(noise_mask(0.1, seed=p.seed + 13, lo=0.6, hi=0.85), 0.15))

    # Stone weathering. (In the engine all three masks darken: grime towards grey, wear towards brown - so the
    # story is painted with those two and the green is real, modelled moss.)
    def tcoord(co, n):
        return np.where(np.abs(n[:, 1]) >= np.abs(n[:, 0]), co[:, 0], co[:, 1] + 31.7)

    def stone_grime(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        ax, ay = np.abs(x), np.abs(y)
        t = tcoord(co, n)
        st = np.clip((n1(t, 0.27, sd) - 0.55) / 0.15, 0, 1)            # where the water runs: vertical runs a hand wide
        st = np.maximum(st, 0.6 * np.clip((n1(t, 0.13, sd + 5) - 0.6) / 0.14, 0, 1))
        ln = 0.35 + 2.2 * n1(t, 0.6, sd + 1) ** 2                     # how far each one runs

        def under(zl, where, k=0.85, band=0.13):
            d = zl - z
            return np.maximum(np.clip(1 - d / ln, 0, 1) * st * k, np.clip(1 - d / 0.2, 0, 1) * band) * (d > -0.02) * where

        outer = (ay > cy - 0.06) | (ax > xt - 0.02)
        pierf = (ax > xt - 0.02) & (ay > cy + 0.05)
        g = under(zc0 - hc, outer)                                     # run-off under the corbel table
        g = np.maximum(g, under(zA - 0.02, pierf))                     # ... under the two set-offs
        g = np.maximum(g, under(zB - 0.02, pierf))
        g = np.maximum(g, under(zb + 0.02, (ax < sw / 2) & (y < -cy + 0.2), 0.6, 0.1))     # ... under the sign board
        g = np.maximum(g, under(zdeck - 0.02, outer & (z > zcor), 0.3, 0.12) * 0.0)
        for (lx, ly, lz) in pockets:                                   # ... and under the arrow-loop sills
            g = np.maximum(g, np.exp(-((x - lx) / 0.08) ** 2) * np.clip(1 - (lz - 0.3 - z) / 1.1, 0, 1) * (z < lz - 0.3) * (np.abs(y - ly) < 0.5) * 0.45)
        dz = 0.3 + 0.22 * n1(t, 0.5, sd + 2)                           # damp, dirty band at the foot of the plinth
        g = np.maximum(g, np.clip(1 - (z - dz) / 0.18, 0, 1) * 0.5)
        g = np.maximum(g, np.clip(1 - z / 1.4, 0, 1) * 0.3 * st)       # splash-back
        for (sx_, sy_, sz_) in soot:                                   # soot only above the lanterns
            dzz = z - sz_
            g = np.maximum(g, np.exp(-((x - sx_) / 0.26) ** 2) * np.clip(dzz / 0.25 + 0.2, 0, 1) * np.clip(1 - dzz / 1.6, 0, 1)
                           * (np.abs(y - sy_) < 0.5) * (dzz > -0.2) * 0.6)
        crown = np.exp(-(x / (0.5 * R)) ** 2) * np.clip((z - (zc - 0.3)) / 0.3, 0, 1) * np.clip(1 - (z - zc) / 1.1, 0, 1) * (ay > cy - 0.1) * 0.5
        g = np.maximum(g, crown * (ax < xt))                           # torch soot over the crown of the arch
        inside = (ay < cy - 0.05) & (ax < xw + 0.3)
        g = np.maximum(g, inside * (0.12 + 0.4 * np.clip((z - zs) / R, 0, 1)))
        spring = np.exp(-((ax - R - 0.25) / 0.4) ** 2) * np.exp(-((z - zs) / 0.3) ** 2) * 0.38      # damp at the springing
        feet = (np.abs(ax - R - 0.2) < 0.55) * np.clip(1 - (z - 0.45) / 0.6, 0, 1) * 0.42            # and at the jamb feet
        g = np.maximum(g, np.maximum(spring, feet) * (ax < xt))
        return np.clip(g, 0, 1)

    def stone_rust(co, n):                                             # brown runs below the iron fixings
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        w = np.zeros(len(x))
        for (rx, ry, rz, rl) in rusts:
            w = np.maximum(w, np.exp(-((x - rx) / 0.045) ** 2) * np.clip(1 - (rz - z) / rl, 0, 1) * (z < rz + 0.02) * (np.abs(y - ry) < 0.4) * 0.8)
        return w

    for o in (field, repl, trim):
        if o is not None:
            paint(o, grime=stone_grime, wear=stone_rust, mode='add')

    def bed_grime(co, n):                                              # pale lime mortar, damp at the foot
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        g = 0.03 + np.clip(1 - z / 0.9, 0, 1) * 0.45
        for (lx, ly, lz) in pockets:                                   # the embrasures are dark
            deep = (np.abs(x - lx) < 0.1) & (np.abs(y) < abs(ly) - 0.1) & (y * ly > 0) & (np.abs(z - lz) < 1.0)
            g = np.where(deep, 1.0, g)
        return g

    paint(backs, grime=bed_grime, wear=0.0, special=0.0, mode='set')

    # ------------------------------------------------------------------ moss: cushions on the weatherings, in the joints of the plinth
    if m_moss is not None:
        r_m = rng.fork("moss")
        mk_ = p.moss
        for sx in (-1, 1):
            x0, x1 = sorted((sx * xt, sx * xo))
            for sgn in (-1, 1):
                north = 1.35 if sgn > 0 else 0.9
                for (zo, so, ri, tyf) in ((zA, o1, ri1, ty2), (zB, o2, ri2, ty3)):          # on the slopes of the set-offs
                    sl_ = hypot(so + 0.035, ri - 0.045)
                    nrm = (0.0, sgn * (ri - 0.045) / sl_, (so + 0.035) / sl_)
                    for _ in range(int((x1 - x0) * 9 * mk_ * north + r_m.u(0, 1))):
                        s_ = r_m.pick((0.1, 0.5, 0.9)) + r_m.u(-0.09, 0.09)
                        ln_ = r_m.u(0.09, 0.3)
                        moss_at((r_m.u(x0 + ln_, x1 - ln_), sgn * (tyf + (so + 0.035) * (1 - s_)), zo - 0.03 + 0.045 + (ri - 0.045) * s_ + 0.004),
                                nrm, ln_, r_m.u(0.045, 0.1), r_m.u(0.018, 0.04))
                for _ in range(int((x1 - x0) * 4 * mk_ * north + r_m.u(0, 1))):             # on the ledge of the pier cornice
                    ln_ = r_m.u(0.04, 0.14)
                    moss_at((r_m.u(x0 + ln_, x1 - ln_), sgn * (ty3 + 0.166), zc0 + 0.262), (0, sgn * 0.3, 1), ln_ * 1.5, r_m.u(0.02, 0.034), r_m.u(0.014, 0.03))
        if not p.gatehouse:
            for sgn in (-1, 1):
                for _ in range(int(2 * xt * 4 * mk_ + r_m.u(0, 1))):
                    ln_ = r_m.u(0.04, 0.14)
                    moss_at((r_m.u(-xt + 0.2, xt - 0.2), sgn * (cy + 0.166), zc0 + 0.262), (0, sgn * 0.3, 1), ln_ * 1.5, r_m.u(0.02, 0.034), r_m.u(0.014, 0.03))
        for (poly, ledge) in ((footprint(xt - PL, W / 2, D / 2, cy + PL), True), (footprint(xt, xo, ty1, cy), False)):
            zmax = 0.5 + 1.3 * mk_
            for i in range(12):
                a, b = poly[i], poly[(i + 1) % 12]
                L = hypot(b[0] - a[0], b[1] - a[1])
                if L < 0.3:
                    continue
                ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
                nx, ny = uy, -ux
                north = 1.35 if ny > 0.5 else 0.9
                rows_ = [(zp, 6.0)] if ledge else [(za, 2.4 * (1 - (za - zp) / (zmax - zp))) for (za, _zb) in crs1 if zp + 0.05 < za < zmax]
                for (zr_, dens) in rows_:
                    for _ in range(int(L * dens * mk_ * north + r_m.u(0, 1))):
                        ln_ = r_m.u(0.08, 0.3)
                        u = r_m.u(ln_, L - ln_)
                        ins = -0.046 if ledge else 0.004
                        x, y = a[0] + ux * u + nx * ins, a[1] + uy * u + ny * ins
                        if i in (4, 10) and abs(x) < R + tr + 0.35:
                            continue
                        if ledge:                                      # top of the plinth course
                            moss_at((x, y, zp - 0.008), (nx * 0.35, ny * 0.35, 0.94), ln_, r_m.u(0.03, 0.05), r_m.u(0.016, 0.034), tangent=(ux, uy, 0))
                        else:                                          # in the bed joints low on the wall
                            moss_at((x, y, zr_ + r_m.u(-0.01, 0.02)), (nx, ny, 0.0), ln_, r_m.u(0.022, 0.05), r_m.u(0.01, 0.018), tangent=(ux, uy, 0))
        for (x, y, sx_, sy_, hb, ridge) in merl:                       # on the merlon copings
            for _ in range(int(3.0 * mk_ + r_m.u(0, 1))):
                ca = (sy_ if ridge == 'x' else sx_) + 0.075
                cb_ = (sx_ if ridge == 'x' else sy_) + 0.075
                a_, b_ = r_m.u(-0.42, 0.42) * ca, r_m.u(-0.36, 0.36) * cb_
                h_ = 0.04 + 0.062 * (1 - abs(a_) / (ca / 2))
                tl_ = (1 if a_ > 0 else -1) * 0.062 / (ca / 2)
                if ridge == 'x':
                    moss_at((x + b_, y + a_, zt + hb + h_), (0, tl_, 1), r_m.u(0.07, 0.15), r_m.u(0.04, 0.075), r_m.u(0.016, 0.034), tangent=(1, 0, 0))
                else:
                    moss_at((x + a_, y + b_, zt + hb + h_), (tl_, 0, 1), r_m.u(0.07, 0.15), r_m.u(0.04, 0.075), r_m.u(0.016, 0.034), tangent=(0, 1, 0))
        r_ms = rng.fork("moss_shape")
        tufts = []
        for v in range(3):
            nn = 7
            V = [(cos(2 * pi * i / nn) * r_ms.u(0.8, 1.15), sin(2 * pi * i / nn) * r_ms.u(0.8, 1.15), -0.15) for i in range(nn)]
            V += [(cos(2 * pi * (i + 0.5) / nn) * r_ms.u(0.45, 0.7), sin(2 * pi * (i + 0.5) / nn) * r_ms.u(0.45, 0.7), r_ms.u(0.55, 0.8)) for i in range(nn)]
            V.append((r_ms.u(-0.15, 0.15), r_ms.u(-0.15, 0.15), 1.0))
            F = [(i, (i + 1) % nn, nn + i) for i in range(nn)] + [((i + 1) % nn, nn + (i + 1) % nn, nn + i) for i in range(nn)]
            F += [(nn + i, nn + (i + 1) % nn, 2 * nn) for i in range(nn)]
            base = mesh_object("moss_tuft", V, F)
            smooth(base, 80)
            mats_ = []
            for (pos, nrm, sx_, sy_, sz_, tg) in moss_sites[v::3]:
                t_ = Vector(tg) if tg is not None else Vector((0, 0, 1)).cross(nrm)
                t_ = t_ - nrm * t_.dot(nrm)
                if t_.length < 0.05:
                    t_ = Vector((1, 0, 0)) - nrm * nrm.x
                t_.normalize()
                b_ = nrm.cross(t_)
                mats_.append(Matrix(((t_.x * sx_, b_.x * sy_, nrm.x * sz_, pos.x), (t_.y * sx_, b_.y * sy_, nrm.y * sz_, pos.y),
                                     (t_.z * sx_, b_.z * sy_, nrm.z * sz_, pos.z), (0, 0, 0, 1))))
            if mats_:
                tufts.append(instances_to_mesh("moss", base, mats_, keep_base=False, seed=p.seed + v))
            else:
                delete([base])
        for o in tufts:
            assign(o, m_moss)
        if tufts:
            join(tufts, "moss")

    # ------------------------------------------------------------------ sockets for the town generator
    socket("wall_west", "attach", (-W / 2, 0, 0), rot=(0, 0, -90), role="wall")
    socket("wall_east", "attach", (W / 2, 0, 0), rot=(0, 0, 90), role="wall")
    socket("pier_sign_r", "attach", (xpc, -ty1, 1.6), role="sign")
