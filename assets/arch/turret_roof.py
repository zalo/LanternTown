"""Spire roof for towers and turrets (the red clock-tower spire of the square): a steep pyramidal /
polygonal / conical roof with swept (bell-cast) eaves, sitting on a deep moulded eaves cornice. The
covering is laid the way it is built: individual clay beaver-tail tiles (real thickness, cambered),
slates or split shingles in half-bond courses over a boarded deck, with a double eaves undercourse,
the field cut to the hip line and covered by half-round hip tiles bedded in mortar (or lead hip rolls),
or copper sheet with standing seams and cross welts. The apex is dressed with a lead / copper cap
carrying a turned iron stem with gilded ball, spike and a weathervane socket. Optional lucarnes: tiny
lead-roofed gabled dormers with a lit pointed-arched light, lead apron, stepped soakers and valleys.
Origin: centre of the eaves ring, z = 0 at eaves level (the bed of the cornice = top of the tower wall).
"""
from tk import *
import math
from math import ceil, sqrt, hypot, atan2, degrees, radians, sin, cos, tan, pi
import numpy as np

META = {"name": "Conical turret roof / spire", "class": "component", "mount": "ground",
        # individually modelled tiles (as the material library asks for on hero roofs) need more than 12k
        "tri_budget": 24000, "focus": 0.25}
PARAMS = {
    "diameter": {"default": 3.4, "min": 1.4, "max": 6.0, "desc": "overall width over the eaves drip edge; across flats for polygonal roofs (m)"},
    "height": {"default": 6.8, "min": 1.5, "max": 9.0, "desc": "eaves level (origin) to the apex of the covering; the finial stands above (m)"},
    "sides": {"default": "4", "choices": ["4", "6", "8", "round", 4, 6, 8], "desc": "plan: 4 / 6 / 8-sided pyramid or round cone (a flat faces -Y)"},
    "bellcast": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "how strongly the eaves sweep out (0 = straight cone)"},
    "covering": {"default": "tile", "choices": ["tile", "slate", "shingle", "copper"], "desc": "clay beaver-tail tiles / slates / split shingles / copper sheet with standing seams"},
    "color": {"default": "auto", "desc": "covering tint ('auto' = natural: crimson-brown clay, blue slate, silvered wood)"},
    "tile_width": {"default": 0.19, "min": 0.12, "max": 0.4, "desc": "width of one tile / slate (m); enlarged automatically on very big roofs"},
    "lucarnes": {"default": 4, "min": 0, "max": 8, "desc": "number of tiny dormers, spread evenly (first one on the front)"},
    "lucarne_size": {"default": 1.0, "min": 0.6, "max": 1.6, "desc": "lucarne scale (1 = 0.56 m wide)"},
    "lucarne_level": {"default": 0.2, "min": 0.08, "max": 0.55, "desc": "height of the lucarne sills as a fraction of the roof height"},
    "lit": {"default": True, "desc": "lucarne windows lit from inside"},
    "cornice": {"default": "stone", "choices": ["stone", "timber"], "desc": "deep moulded stone cornice on modillions / oak wall plate with curved rafter brackets"},
    "overhang": {"default": 0.26, "min": 0.12, "max": 0.5, "desc": "eaves projection beyond the wall face (m)"},
    "trim": {"default": "#d6c9ad", "desc": "paint colour of the lucarne joinery and eaves fascia"},
    "finial_height": {"default": 1.05, "min": 0.4, "max": 2.2, "desc": "finial height above the apex (m)"},
    "finial_metal": {"default": "gilt", "choices": ["gilt", "copper", "iron"], "desc": "finish of the finial ball and weathervane socket"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: run-off staining, moss, slipped and uneven tiles"},
}
VARIANTS = {
    "slate_octagon": {"sides": "8", "covering": "slate", "diameter": 3.0, "height": 7.4, "bellcast": 0.85, "lucarnes": 4,
                      "lucarne_level": 0.22, "finial_height": 1.3, "trim": "#2f3d35", "seed": 7},
    "copper_cone": {"sides": "round", "covering": "copper", "diameter": 2.6, "height": 3.2, "bellcast": 1.0, "lucarnes": 3, "lucarne_size": 0.85,
                    "finial_metal": "copper", "finial_height": 0.8, "age": 0.8, "seed": 12},
    "shingle_hex": {"sides": "6", "covering": "shingle", "diameter": 4.4, "height": 3.0, "bellcast": 0.35, "lucarnes": 3,
                    "lucarne_size": 0.85, "lucarne_level": 0.16, "cornice": "timber", "overhang": 0.4, "trim": "#7a2f26",
                    "finial_metal": "iron", "finial_height": 0.7, "age": 0.9, "seed": 21},
    "witch_hat": {"sides": "round", "covering": "tile", "diameter": 2.3, "height": 4.6, "bellcast": 0.9, "lucarnes": 2,
                  "lucarne_size": 0.8, "color": "#a35a48", "lit": True, "finial_height": 0.9, "age": 0.7, "seed": 33},
}

ZV = np.array([0.0, 0.0, 1.0])


def _shade(hexc, f, tw=(1.0, 1.0, 1.0)):
    h = hexc.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    c = [max(0.0, min(1.0, v * f * k)) for v, k in zip(c, tw)]
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


def _frame(xa, ya, za, t):
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = xa, ya, za, t
    return m


def _dirs(a):
    """outward horizontal normal and horizontal tangent of the roof at plan angle a (rad; 0 = front, -Y)."""
    return np.array([sin(a), -cos(a), 0.0]), np.array([cos(a), sin(a), 0.0])


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _get_co(o):
    me = o.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def _set_co(o, co):
    o.data.vertices.foreach_set("co", co.astype(np.float32).ravel())
    o.data.update()


class _Sheet:
    """Collects loose quads (sheet lead pieces); each quad is wound to face `hint`."""

    def __init__(self):
        self.v, self.f = [], []

    def quad(self, a, b, c, d, hint):
        pts = [np.array(q, dtype=float) for q in (a, b, c, d)]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0]) + np.cross(pts[2] - pts[0], pts[3] - pts[0])
        if float(np.dot(n, np.array(hint, dtype=float))) < 0:
            pts = pts[::-1]
        i = len(self.v)
        self.v += [tuple(q) for q in pts]
        self.f.append((i, i + 1, i + 2, i + 3))

    def obj(self, name):
        return mesh_object(name, self.v, self.f)


def _prism_x(name, dz_pts, width):
    """Bracket: a (d = outwards, z) outline extruded across X; outwards is -Y."""
    n = len(dz_pts)
    verts = [(-width / 2, -d, z) for d, z in dz_pts] + [(width / 2, -d, z) for d, z in dz_pts]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _tile_mesh(kind, tw, L, g, t, lod=0):
    """One tile lying on the roof plane: x across, y up the slope (tail at y = 0), z = roof normal.
    The tile is a wedge resting on the course below; only faces that can be seen are built: the top
    (cambered on clay tiles) and the splayed tail edge with the real thickness t."""
    zh = 0.25 * t
    zt = zh + t * L / g

    def ztop(y):
        return zt - (zt - zh) * y / L

    if kind == "tile":                      # beaver-tail: segmental rounded tail
        ya = 0.36 * tw
        sp = 0.0035
        if lod == 0:
            angs = (180, 225, 270, 315, 360)
            cam = 0.0045
        else:
            angs = (180, 240, 300, 360)
            cam = 0.0
        arc = [((tw / 2) * cos(radians(a)), ya + ya * sin(radians(a))) for a in angs]

        def cz(x):
            return cam * (1.0 - (2.0 * x / tw) ** 2)

        n = len(arc)
        top = [(x, y, ztop(y) + cz(x)) for x, y in arc]
        bot = [(x + 0.3 * sp * cos(radians(a)), y + sp * sin(radians(a)), ztop(y) + cz(x) - t) for (x, y), a in zip(arc, angs)]
        if lod == 0:
            verts = top + [(tw / 2, L, zh), (0.0, L, zh + cam), (-tw / 2, L, zh)] + bot
            faces = [(0, 1, 2, 6, 7), (2, 3, 4, 5, 6)] + [(8 + i, 9 + i, i + 1, i) for i in range(4)]
        else:
            verts = top + [(tw / 2, L, zh), (-tw / 2, L, zh)] + bot
            faces = [(0, 1, 2, 3, 4, 5)] + [(6 + i, 7 + i, i + 1, i) for i in range(3)]
    else:                                   # slate / shingle: square tail
        top = [(-tw / 2, 0, zt), (tw / 2, 0, zt), (tw / 2, L, zh), (-tw / 2, L, zh)]
        bot = [(-tw / 2, -0.002, zt - t), (tw / 2, -0.002, zt - t)]
        verts = top + bot
        faces = [(0, 1, 2, 3), (4, 5, 1, 0)]
        if kind == "shingle":               # thick butts: show the side edges too
            verts += [(-tw / 2, g, ztop(g) - t), (tw / 2, g, ztop(g) - t), (-tw / 2, g, ztop(g)), (tw / 2, g, ztop(g))]
            faces += [(4, 0, 8, 6), (5, 7, 9, 1)]
    o = mesh_object("tile_base", verts, faces)
    smooth(o, 50)
    return o


def _under_tile_mesh(tw, L, g, t):
    """Square-tailed eaves undercourse tile lying directly under the first course (top + tail edge)."""
    zh = 0.25 * t
    zt = zh + t * L / g
    Lu = 0.95 * g
    z0 = zt - t
    z1 = z0 - (zt - zh) * Lu / L
    verts = [(-tw / 2, 0, z0), (tw / 2, 0, z0), (tw / 2, Lu, z1), (-tw / 2, Lu, z1), (-tw / 2, -0.002, z0 - t), (tw / 2, -0.002, z0 - t)]
    return mesh_object("under_tile_base", verts, [(0, 1, 2, 3), (4, 5, 1, 0)])


def _hip_tile_mesh(w, Lh, th, Sz, lap, nseg=4):
    """Half-round hip tile: x across the hip, y up the hip (tail at y = 0), z = unit height of the
    arched section (the instance matrix scales z to the real rise Sz over the arris). The tile tapers by
    its own thickness per gauge so each one nests over the one below; the tail shows the clay thickness."""
    angs = [180.0 - 180.0 * i / nseg for i in range(nseg + 1)]
    st = th * lap
    tail = [(w * cos(radians(a)), 0.0, sin(radians(a))) for a in angs]
    head = [((w - st) * cos(radians(a)), Lh, (1.0 - st / Sz) * sin(radians(a))) for a in angs]
    inner = [((w - th) * cos(radians(a)), 0.003, (1.0 - th / Sz) * sin(radians(a))) for a in angs]
    verts = tail + head + inner
    m = nseg + 1
    faces = [(i, i + 1, m + i + 1, m + i) for i in range(nseg)]
    faces += [(2 * m + i, 2 * m + i + 1, i + 1, i) for i in range(nseg)]
    o = mesh_object("hip_tile_base", verts, faces)
    smooth(o, 50)
    return o


def build(p, rng):
    sides = str(p.sides)
    rnd = sides == "round"
    N = 0 if rnd else int(sides)
    D, H = float(p.diameter), float(p.height)
    Re = D / 2 - (0.02 if rnd else 0.045)                  # tile tails / hip tiles stand proud of the surface
    age = float(p.age)
    cov = p.covering
    seed = int(p.seed)
    ov = min(p.overhang, Re * 0.32)
    Rw = Re - ov                                           # wall face (apothem)
    kc = 1.0 if rnd else 1.0 / cos(pi / N)                 # apothem -> corner radius
    tanN = 0.0 if rnd else tan(pi / N)
    lrot = 0.0 if rnd else (-90.0 - 180.0 / N) % (360.0 / N)
    lseg = 48 if rnd else N
    lsm = 50.0 if rnd else 24.0
    is_cu = cov == "copper"

    def ring(name, prof, seg=None):
        return lathe(name, [(max(r, 0.0) * kc, z) for r, z in prof], segments=seg or lseg, rot=(0, 0, lrot), smooth_angle=lsm)

    # ------------------------------------------------------------------ profile of the roof surface
    if p.cornice == "stone":
        zc = clamp(0.125 * D, 0.26, 0.5)                   # deep moulded cornice: its height = top of the stone
    else:
        zc = clamp(0.085 * D, 0.2, 0.36) - 0.03
    zc = min(zc, 0.3 * H)
    ze = zc + 0.03                                         # tile drip edge above the cornice bed
    b = float(p.bellcast)
    Rm = Re * (1.0 - 0.36 * b)
    zk = ze + (H - ze) * (0.2 + 0.14 * b)

    def rz(z):
        v = Rm * (H - z) / (H - ze)
        if z < zk:
            q_ = (zk - z) / (zk - ze)
            v += (Re - Rm) * q_ * q_
        return max(v, 0.0)

    zs = np.linspace(ze, H, 801)
    rs = np.array([rz(z) for z in zs])
    S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(rs), np.diff(zs)))])
    Stot = float(S[-1])

    def at(s):
        s = clamp(s, 0.0, Stot)
        return float(np.interp(s, S, rs)), float(np.interp(s, S, zs))

    def s_of_r(r):
        return float(np.interp(r, rs[::-1], S[::-1]))

    def s_of_z(z):
        return float(np.interp(z, zs, S))

    def slope(s, ds):
        s = clamp(s, 0.0, Stot)
        r0, z0 = at(s)
        r1, z1 = at(min(s + ds, Stot))
        if s + ds > Stot:
            r0, z0 = at(Stot - ds)
        d = hypot(r1 - r0, z1 - z0) or 1.0
        return (r1 - r0) / d, (z1 - z0) / d

    # ------------------------------------------------------------------ materials
    lead = mat("copper_patina", rough=1.2, scale=2.2) if is_cu else mat("lead")
    paintm = mat("wood_painted", p.trim)
    if cov == "tile":
        base_col = "#93463f" if p.color == "auto" else p.color
        tints = [base_col, _shade(base_col, 0.74, (1.0, 0.94, 0.97)), _shade(base_col, 1.1, (1.0, 1.05, 0.98))]
        cmats = [mat("terracotta", c) for c in tints]
    elif cov == "slate":
        base_col = "#9ca3b3" if p.color == "auto" else p.color
        tints = [base_col, _shade(base_col, 0.8, (1.0, 1.0, 1.05)), _shade(base_col, 0.95, (1.08, 0.98, 1.06))]
        cmats = [mat("slate", c) for c in tints]
    elif cov == "shingle":
        base_col = "#cfc6b8" if p.color == "auto" else p.color
        tints = [base_col, _shade(base_col, 0.74, (1.0, 0.97, 0.92)), _shade(base_col, 1.1, (1.0, 1.0, 1.04))]
        cmats = [mat("wood_weathered", c) for c in tints]
    else:
        cmats = [lead]

    north = lambda co, nrm: np.clip(co[:, 1] / (np.hypot(co[:, 0], co[:, 1]) + 1e-6), 0.0, 1.0)
    streak_n = noise_mask(0.16, seed=seed + 3, lo=0.36, hi=0.7)
    streaks = lambda co, nrm: streak_n(co * np.array([1.0, 1.0, 0.1]), nrm)

    # ------------------------------------------------------------------ eaves cornice
    h = zc
    if p.cornice == "stone":
        stone = mat("stone_smooth", "#b9ad98")
        q = max(ov - 0.065, 0.05)                          # projection of the corona beyond the bed mould
        r1 = Rw + 0.03
        prof = [(Rw - 0.07, 0.0), (Rw + 0.014, 0.0), (Rw + 0.014, 0.045 * h), (Rw + 0.03, 0.06 * h), (Rw + 0.042, 0.085 * h),
                (Rw + 0.03, 0.11 * h), (Rw + 0.016, 0.12 * h), (Rw + 0.016, 0.30 * h),                       # necking, astragal, frieze
                (r1, 0.315 * h), (r1 + 0.03 * q, 0.34 * h), (r1 + 0.10 * q, 0.38 * h), (r1 + 0.17 * q, 0.44 * h),
                (r1 + 0.18 * q, 0.455 * h), (r1 + 0.18 * q, 0.635 * h),                                      # bed mould, modillion band
                (r1 + 0.80 * q, 0.635 * h), (r1 + 0.80 * q, 0.775 * h), (r1 + 0.83 * q, 0.79 * h),           # soffit, corona
                (r1 + 0.85 * q, 0.84 * h), (r1 + 0.93 * q, 0.915 * h), (r1 + 0.985 * q, 0.95 * h),           # cyma
                (r1 + q, 0.96 * h), (r1 + q, h), (r1 + q - 0.12, h)]
        corn = ring("cornice", prof)
        assign(corn, stone)
        dm, hm = 0.58 * q, 0.175 * h
        z0 = 0.458 * h
        brk = _prism_x("modillion", [(0.0, z0), (dm * 0.5, z0), (dm * 0.66, z0 + 0.12 * hm), (dm * 0.84, z0 + 0.42 * hm),
                                     (dm, z0 + 0.6 * hm), (dm, z0 + hm), (0.0, z0 + hm)], clamp(0.04 * D, 0.07, 0.13))
        assign(brk, stone)
        brk_r, brk_sp = r1 + 0.18 * q - 0.004, clamp(0.085 * D, 0.2, 0.34)
    else:
        oak = mat("wood_oak", "#6b4a2e")
        Rc = Re - 0.05
        q = max(Rc - Rw - 0.03, 0.04)
        prof = [(Rw - 0.07, 0.0), (Rw + 0.03, 0.0), (Rw + 0.045, 0.03 * h), (Rw + 0.045, 0.4 * h), (Rw + 0.028, 0.45 * h),
                (Rw + 0.028, 0.62 * h), (Rw + 0.03 + q, 0.8 * h), (Rw + 0.03 + q, h), (Rw + 0.03 + q - 0.1, h)]
        corn = ring("cornice", prof)
        assign(corn, oak)
        qq = q - 0.012
        brk = _prism_x("rafter_foot", [(0.0, 0.05 * h), (0.04, 0.05 * h), (0.05, 0.22 * h), (0.07 + 0.2 * qq, 0.47 * h),
                                       (0.03 + 0.62 * qq, 0.63 * h), (0.03 + qq, 0.68 * h), (0.03 + qq, 0.81 * h), (0.0, 0.64 * h)],
                       clamp(0.03 * D, 0.06, 0.1))
        uv_grain(brk, axis="Y")
        assign(brk, oak)
        brk_r, brk_sp = Rw + 0.03, clamp(0.11 * D, 0.28, 0.46)
    # modillions / rafter feet round the eaves
    mm = []
    if rnd:
        n = max(8, int(round(2 * pi * brk_r / brk_sp)))
        for j in range(n):
            a = 2 * pi * j / n
            nd, td = _dirs(a)
            mm.append(trs(tuple(nd * brk_r), (0, 0, degrees(a))))
    else:
        for k in range(N):
            a = 2 * pi * k / N
            nd, td = _dirs(a)
            half = brk_r * tanN - 0.5 * brk_sp * 0.45
            n = max(2, int(round(2 * half / brk_sp)) + 1)
            for j in range(n):
                x = -half + 2 * half * j / (n - 1)
                mm.append(trs(tuple(nd * brk_r + td * x), (0, 0, degrees(a))))
    brackets = instances_to_mesh("brackets", brk, mm, keep_base=False, seed=seed)
    for o in (corn, brackets):
        paint(o, grime=lambda co, nrm: np.clip((0.08 + 0.34 * streaks(co, nrm)) * (0.5 + age), 0, 1) * np.clip(1.15 - co[:, 2] / max(h, 1e-3) * 0.5, 0, 1),
              special=lambda co, nrm: (0.12 + 0.88 * north(co, nrm)) * noise_mask(0.3, seed=seed + 5)(co, nrm) * age * 0.9)

    # ------------------------------------------------------------------ covering dimensions
    tw = float(p.tile_width) * (0.74 if cov == "shingle" else 1.0)
    gk = {"tile": 0.84, "slate": 0.8, "shingle": 1.15, "copper": 1.0}[cov]
    tthick = {"tile": 0.015, "slate": 0.008, "shingle": 0.015, "copper": 0.0}[cov]
    tri_hi = {"tile": 14, "slate": 4, "shingle": 8, "copper": 0}[cov]
    tri_lo = {"tile": 10, "slate": 4, "shingle": 8, "copper": 0}[cov]
    tri_max = {"tile": 13600 - (3000 if rnd else 0), "slate": 9000, "shingle": 10500, "copper": 0}[cov]

    # tilting fillet / painted fascia closing the eaves under the double undercourse
    de, dze = slope(0.0, 0.08)
    if is_cu:
        fprof = [(Re - 0.12, zc), (Re - 0.012, zc), (Re - 0.012, ze + 0.012), (Re - 0.012 + de * 0.1, ze + 0.012 + dze * 0.1 - 0.012)]
    else:
        zf = ze + 1.15 * tthick * de + 0.01 * dze - 0.002
        fprof = [(Re - 0.12, zc), (Re - 0.022, zc), (Re - 0.022, zf), (Re - 0.022 + de * 0.12, zf + dze * 0.12)]
    fas = ring("eaves_fascia", fprof)
    assign(fas, paintm if not is_cu else lead)
    if is_cu:
        paint(fas, special=0.8, grime=0.2)
    else:
        paint(fas, grime=lambda co, nrm: 0.35 + 0.5 * streaks(co, nrm), wear=0.25 * age)

    # ------------------------------------------------------------------ lucarne layout
    nl = int(p.lucarnes)
    ls = p.lucarne_size * clamp(D / 3.4, 0.75, 1.4)
    zb = ze + (H - ze) * p.lucarne_level
    luc_angles = []
    if nl > 0:
        r_need = rz(min(zb + 0.52 * ls, H))
        avail = (r_need * tanN - 0.1) if not rnd else r_need / 0.95
        if avail < 0.28 * ls:
            ls *= max(avail, 0.0) / (0.28 * ls)
        if ls >= 0.5 and zb + 0.9 * ls < H * 0.8:
            if rnd:
                luc_angles = [2 * pi * i / nl for i in range(nl)]
            else:
                luc_angles = [2 * pi * k / N for k in sorted({int(round(i * N / nl)) % N for i in range(nl)})]
    lw, lh, rise = 0.56 * ls, 0.52 * ls, 0.37 * ls
    r_b = rz(zb)
    r_f = (sqrt(max(r_b * r_b - lw * lw / 4, 0.01)) if rnd else r_b) + 0.02

    def in_lucarne(a, x, zc_):
        """is the exposed part of a tile (centre at face-x / plan angle a, height zc_) inside a lucarne?"""
        for la in luc_angles:
            if rnd:
                da = (a - la + pi) % (2 * pi) - pi
                xx = abs(da) * rz(zc_)
            else:
                if abs(a - la) > 1e-6:
                    continue
                xx = abs(x)
            zl = zc_ - zb
            if xx < lw / 2 - 0.03 and 0.03 < zl < lh + rise * (1.0 - xx / (lw / 2)) - 0.02:
                return True
        return False

    def below_sills(co):
        """run-off zone under every lucarne sill (0..1)."""
        out = np.zeros(len(co))
        for la in luc_angles:
            nd, td = _dirs(la)
            xl = co @ td
            dp = co @ nd
            dzl = zb - co[:, 2]
            wm = np.clip((lw * 0.5 + 0.1 - np.abs(xl)) / 0.14, 0.0, 1.0)
            fall = np.clip(1.0 - dzl / (0.9 * ls + 0.35), 0.0, 1.0) * (dzl > -0.04)
            out = np.maximum(out, wm * fall * (dp > 0.2 * r_b))
        return out

    # ------------------------------------------------------------------ courses
    def face_row(i, hw, tw_, r_=None):
        """tiles of one course on a flat face: constant width in half bond, cut later at the hips."""
        out = []
        if cov == "shingle" and r_ is not None:
            x = -hw - tw_ * r_.u(0.0, 0.9)
            while x < hw:
                w = tw_ * r_.u(0.62, 1.4)
                out.append((x + w / 2, w))
                x += w
        else:
            x = -(int(ceil(hw / tw_)) + 1) * tw_ + (0.5 * tw_ if i % 2 else 0.0)
            while x < hw:
                out.append((x + tw_ / 2, tw_))
                x += tw_
        return [(xc, w) for xc, w in out if min(xc + w / 2, hw) - max(xc - w / 2, -hw) > 0.03]

    def plan(tw_):
        g_ = tw_ * gk
        # the covering stops where a face gets too narrow to lay between the hips; the lead cap takes over
        r_cap = max(0.1, tw_ * 0.55) if rnd else max(0.12, 0.36 * tw_ / tanN, (0.13 if cov == "tile" else 0.07) / tanN)
        r_cap = min(r_cap, Re * 0.3, 0.32)
        s_top = s_of_r(r_cap)
        nc_ = max(2, int(ceil(s_top / g_)))
        s_hi_ = 0.36 * Stot
        if luc_angles:
            s_hi_ = max(s_hi_, s_of_z(min(zb + lh + rise, H)) + 2 * g_)
        rows_, tris = [], 0
        for i in range(nc_):
            r, z = at(i * g_)
            if rnd:
                n = max(5, int(round(2 * pi * r / min(tw_, sqrt(8 * r * 0.012) + 1e-3))))
            else:
                n = len(face_row(i, r * tanN, tw_)) * N
            lod = 0 if i * g_ <= s_hi_ else 1
            tris += n * (tri_hi if lod == 0 else tri_lo)
            rows_.append((i, r, z, n, lod))
        return rows_, g_, nc_, tris

    rows, g, nc, ftris = plan(tw)
    # hip tiles and extra lucarnes are paid for by coarser field tiles on very big roofs
    hs_ = clamp(D / 3.4, 0.85, 1.3)
    hip_seg = 4 if N == 4 else 3                           # flatter hips need fewer facets
    if cov == "tile" and not rnd:
        tri_max -= max(0.0, N * Stot / (0.125 * hs_) * 4 * hip_seg - 3600)
    tri_max = max(tri_max - max(0, len(luc_angles) - 4) * 700, 0.55 * tri_max, 1.0)
    if not is_cu:
        while ftris > tri_max:
            tw *= 1.04
            rows, g, nc, ftris = plan(tw)
    s_cb = min(nc * g - 0.3 * g, Stot - 0.12)              # bottom edge of the apex cap
    if is_cu:
        s_cb = s_of_r(max(0.16, Re * 0.1))
    z_cap = at(s_cb)[1]

    # ------------------------------------------------------------------ weathering stories of the covering
    moss_n = noise_mask(0.5, seed=seed + 11, lo=0.3, hi=0.7)
    fine_n = noise_mask(0.12, seed=seed + 13, lo=0.35, hi=0.75)

    def moss(co, nrm):
        low = np.clip(1.0 - (co[:, 2] - ze) / (0.6 * (H - ze)), 0.0, 1.0)
        bell = np.clip(1.0 - (co[:, 2] - ze) / max(zk - ze, 0.2), 0.0, 1.0)
        nf = north(co, nrm)
        v = (0.08 + 0.92 * nf) * (0.25 + 0.75 * low) * moss_n(co, nrm) * (0.35 + 0.65 * fine_n(co, nrm)) * 1.25
        patch = np.clip((0.6 * fine_n(co, nrm) + 0.6 * moss_n(co, nrm) - 0.42) * 2.2, 0.0, 1.0)
        v = v + 0.85 * bell ** 0.8 * patch * (0.5 + 0.5 * nf)          # damp swept eaves courses, all round
        return np.clip(v * age * 1.15, 0, 1)

    def soot(co, nrm):
        st = streaks(co, nrm)
        eav = np.clip(1.0 - (co[:, 2] - ze) / 0.45, 0.0, 1.0)
        capr = np.clip(1.0 - (z_cap - co[:, 2]) / (0.26 * (H - ze)), 0.0, 1.0) * (co[:, 2] < z_cap + 0.1)
        sill = below_sills(co)
        v = 0.05 + 0.2 * st + 0.28 * eav + 0.6 * capr * (0.35 + 0.65 * st) + 0.8 * sill * (0.4 + 0.6 * st)
        return np.clip(v * (0.45 + 0.8 * age), 0, 1)

    def clip_cover(o, nv, nds, tds, Ps, Ys):
        """cut the stamped tiles to the hip planes and stop them against the lucarne cheeks / sills."""
        if nv == 0:
            return
        co = _get_co(o)
        ND, TD = np.repeat(nds, nv, axis=0), np.repeat(tds, nv, axis=0)
        if not rnd:
            x = (co * TD).sum(1)
            lim = np.maximum((co * ND).sum(1) * tanN - 0.003, 0.0)
            co += TD * (np.clip(x, -lim, lim) - x)[:, None]
        if luc_angles:
            PV, YV = np.repeat(Ps, nv, axis=0), np.repeat(Ys, nv, axis=0)
            for la in luc_angles:
                nd, td = _dirs(la)
                xl, dp, zl = co @ td, co @ nd, co[:, 2] - zb
                ins = (np.abs(xl) < lw / 2 - 0.006) & (zl > 0.012) & (zl < lh + rise * (1.0 - np.abs(xl) / (lw / 2))) & (dp > 0.3 * r_b)
                xc = PV @ td
                side = np.abs(xc) >= lw / 2 - 0.03
                m1 = ins & side
                co[m1] += td[None, :] * (np.sign(xc[m1]) * (lw / 2 - 0.006) - xl[m1])[:, None]
                m2 = ins & ~side & ((PV[:, 2] - zb) < 0.03)
                co[m2] -= YV[m2] * ((zl[m2] - 0.012) / np.maximum(YV[m2, 2], 0.2))[:, None]
        _set_co(o, co)

    # ------------------------------------------------------------------ deck (or the copper skin itself)
    n_seam = 0
    bay = clamp(0.16 * D, 0.38, 0.6)
    if is_cu and rnd:
        n_seam = max(8, int(round(2 * pi * Re / bay / 2)) * 2)
    off = 0.0 if is_cu else -0.012
    nd0, nd1 = (15, 17) if is_cu else (8, 7)              # the copper skin carries the patina mask: denser
    ss = sorted(set(list(np.linspace(0.0, s_of_z(zk), nd0)) + list(np.linspace(s_of_z(zk), Stot, nd1))))
    dprof = []
    for s in ss[:-1]:
        r, z = at(s)
        dprof.append((r + off, z))
    dprof.append((0.0, H + off))
    deck = ring("deck", dprof, seg=(n_seam * 4 if (is_cu and rnd) else None))
    assign(deck, cmats[1] if len(cmats) > 1 else cmats[0])

    # ------------------------------------------------------------------ covering
    hip_extra = 0.0
    if not is_cu:
        L = 1.6 * g
        bases = [_tile_mesh(cov, tw, L, g, tthick, 0), _tile_mesh(cov, tw, L, g, tthick, 1) if cov == "tile" else None]
        ubase = _under_tile_mesh(tw, L, g, tthick)
        if cov == "shingle":
            uv_grain(bases[0], axis="Y")
            uv_grain(ubase, axis="Y")
        ragged = {"tile": 0.6, "slate": 0.45, "shingle": 1.6}[cov]
        tr = rng.fork("tiles")
        groups = {}

        def add(key, nd, td, x, w, r, z, dr, dz, under=0):
            X = td
            Y = nd * dr + ZV * dz
            Zn = nd * dz - ZV * dr
            if under:
                psi, slip, lift = radians(tr.jitter(0.0, 0.3)), -0.005 * under, 1.0
                org = nd * r + td * x + ZV * z - Y * slip - Zn * (tthick * (under - 1))
            else:
                psi = radians(tr.jitter(0.0, (0.35 + 1.1 * age) * ragged))
                slip = tr.jitter(0.0, 0.003 * ragged)
                lift = tr.u(0.97, 1.05 + 0.08 * age * ragged)
                if tr.chance(0.005 + 0.014 * age):               # a handful of slipped / re-fixed ones
                    psi += radians(tr.jitter(0.0, 4.0))
                    slip += tr.u(0.012, 0.03)
                if cov == "shingle":
                    slip += tr.u(-0.012, 0.012)
                org = nd * r + td * x + ZV * z - Y * slip
            X2 = X * cos(psi) + Y * sin(psi)
            Y2 = -X * sin(psi) + Y * cos(psi)
            sx = max(w - tr.u(0.002, 0.005), 0.02) / tw
            gr = (tr.u(0, 1) ** 2.4) * 0.55 * (0.5 + age)       # per-tile value range: a few dark, a few pale
            wr = (tr.u(0, 1) ** 3.0) * 0.3
            groups.setdefault(key, []).append((_frame(X2 * sx, Y2, Zn * lift, org), nd, td, org, Y, gr, wr))

        pick = lambda: tr.weighted([(0, 0.5), (1, 0.28), (2, 0.22)])
        for (i, r, z, n, lod) in rows:
            dr, dz = slope(i * g, g)
            zmid = at(i * g + 0.5 * g)[1]
            lod = lod if bases[1] is not None else 0
            if rnd:
                wn = 2 * pi * r / n
                for j in range(n):
                    a = 2 * pi * (j + (0.5 if i % 2 else 0.0)) / n
                    if in_lucarne(a, 0.0, zmid):
                        continue
                    nd, td = _dirs(a)
                    add((pick(), lod), nd, td, 0.0, wn, r, z, dr, dz)
                    if i == 0:                                   # double eaves undercourse, half bond
                        a2 = 2 * pi * (j + 0.5) / n
                        nd2, td2 = _dirs(a2)
                        add((1, 2), nd2, td2, 0.0, wn, r, z, dr, dz, under=1)
                        add((1, 2), nd, td, 0.0, wn, r, z, dr, dz, under=2)
            else:
                hw = r * tanN
                for k in range(N):
                    a = 2 * pi * k / N
                    nd, td = _dirs(a)
                    for (xc, w) in face_row(i, hw, tw, tr):
                        if in_lucarne(a, xc, zmid):
                            continue
                        add((pick(), lod), nd, td, xc, w, r, z, dr, dz)
                    if i == 0:
                        for (xc, w) in face_row(1, hw, tw):
                            add((1, 2), nd, td, xc, w, r, z, dr, dz, under=1)
                        for (xc, w) in face_row(0, hw, tw):
                            add((1, 2), nd, td, xc, w, r, z, dr, dz, under=2)
        ntile = 0
        for (gi, lod), recs in sorted(groups.items()):
            base = ubase if lod == 2 else bases[lod]
            assign(base, cmats[gi])
            o = instances_to_mesh(f"covering_{gi}_{lod}", base, [rc[0] for rc in recs], keep_base=True, seed=seed + gi + 10 * lod)
            nv = len(o.data.vertices) // len(recs)
            clip_cover(o, nv, np.array([rc[1] for rc in recs]), np.array([rc[2] for rc in recs]),
                       np.array([rc[3] for rc in recs]), np.array([rc[4] for rc in recs]))
            gv = np.repeat(np.array([rc[5] for rc in recs]), nv)
            wv = np.repeat(np.array([rc[6] for rc in recs]), nv)
            paint(o, grime=lambda co, nrm, gv=gv: np.maximum(soot(co, nrm), gv) if len(gv) == len(co) else soot(co, nrm),
                  wear=lambda co, nrm, wv=wv: wv if len(wv) == len(co) else np.zeros(len(co)),
                  special=moss)
            ntile += len(recs)
        delete([o_ for o_ in bases if o_ is not None] + [ubase])
        paint(deck, grime=0.9)
        print(f"DBG tiles={ntile} tile_w={tw:.3f} gauge={g:.3f} courses={nc} field_tris~{ftris}")
    else:
        # copper sheet: the deck lathe is the skin; standing seams up the slope + cross welts
        if not rnd:
            densify(deck, 0.16)
        cu_streak = noise_mask(0.1, seed=seed + 17, lo=0.3, hi=0.75)
        nw = max(1, int(s_cb / 1.15))
        welt_s = [s_cb * wi / (nw + 1) + rng.jitter(0.0, 0.04) for wi in range(1, nw + 1)]
        welt_z = [at(s)[1] for s in welt_s]

        def seam_d(co):
            """metres (horizontal) to the nearest standing seam."""
            a = np.arctan2(co[:, 0], -co[:, 1])
            if rnd:
                st = 2 * pi / n_seam
                return np.abs((a + st / 2) % st - st / 2) * np.hypot(co[:, 0], co[:, 1])
            st = 2 * pi / N
            ak = np.round(a / st) * st
            ax = np.abs(co[:, 0] * np.cos(ak) + co[:, 1] * np.sin(ak))
            return np.abs((ax / bay) % 1.0 - 0.5) * bay

        def verdigris(co, nrm):
            """green where the water runs and stands: along the seams, under the welts and sills, at the eaves."""
            z = co[:, 2]
            st = cu_streak(co * np.array([1.0, 1.0, 0.07]), nrm)
            near = np.exp(-(seam_d(co) / 0.06) ** 2)
            low = np.clip(1.0 - (z - ze) / (0.3 * (H - ze)), 0.0, 1.0)
            wl = np.zeros(len(co))
            for zw in welt_z:
                wl = np.maximum(wl, np.clip(1.0 - (zw - z) / 0.3, 0.0, 1.0) * (z < zw + 0.01))
            v = 0.9 * low ** 1.4 * (0.62 + 0.38 * st) + near * (0.3 + 0.5 * st) + 0.5 * wl * (0.3 + 0.7 * st) + 0.85 * below_sills(co) * (0.45 + 0.55 * st) + 0.16 * st
            return np.clip(v * (0.45 + 0.7 * age), 0, 1)

        def bronze(co, nrm):
            """sheltered sheet stays brown-bronze."""
            return np.clip(0.75 - 1.5 * verdigris(co, nrm), 0.0, 0.75)

        paint(deck, special=verdigris, wear=bronze, grime=lambda co, nrm: 0.3 * np.clip(1.0 - 2.0 * verdigris(co, nrm), 0, 1))
        # double-lock standing seam, 28 mm high with the fold to one side
        seam_prof = [(-0.006, -0.004), (0.006, -0.004), (0.006, 0.018), (0.011, 0.02), (0.011, 0.03), (-0.006, 0.03)]
        seams = []

        def seam(a, x, s0, s1, nm):
            nd, td = _dirs(a)
            pts = []
            for s in np.linspace(s0, s1, max(4, int((s1 - s0) / 0.3) + 2)):
                r, z = at(s)
                pts.append(tuple(nd * r + td * x + ZV * z))
            dr, dz = slope(s0, 0.1)
            seams.append(sweep(nm, pts, seam_prof, smooth_path=False, up=tuple(nd * dz - ZV * dr)))

        if rnd:
            s_half = s_of_r(Re * 0.42)
            for j in range(n_seam):
                seam(2 * pi * j / n_seam, 0.0, 0.0, s_cb + 0.05 if j % 2 == 0 else min(s_half, s_cb), f"seam{j}")
        else:
            for k in range(N):
                a = 2 * pi * k / N
                hw0 = Re * tanN
                nb = max(1, int(round(hw0 / bay - 0.5)))
                for j in range(-nb, nb + 1):
                    if j == 0:
                        continue
                    x = (j + (0.5 if j < 0 else -0.5)) * bay
                    s1 = min(s_of_r((abs(x) + 0.02) / tanN), s_cb)
                    if s1 > 0.25:
                        seam(a, x, 0.0, s1, f"seam{k}_{j}")
                # hip rolls
                ah = a + pi / N
                nd, td = _dirs(ah)
                pts = []
                for s in np.linspace(0.0, s_cb + 0.05, max(5, int(s_cb / 0.3) + 2)):
                    r, z = at(s)
                    pts.append(tuple(nd * r * kc + ZV * z))
                seams.append(sweep(f"hip{k}", pts, [(-0.011, -0.006), (0.011, -0.006), (0.011, 0.036), (-0.011, 0.036)],
                                   smooth_path=False, up=tuple(nd)))
        # cross welts: flat-lock seams round the roof, one per sheet length
        for wi, s in enumerate(welt_s):
            dr, dz = slope(s, 0.03)
            r, z = at(s)
            seams.append(ring(f"welt{wi}", [(r + dr * 0.024 + 0.001, z + dz * 0.024), (r + dz * 0.009 + dr * 0.018, z - dr * 0.009 + dz * 0.018),
                                            (r + dz * 0.009 - dr * 0.02, z - dr * 0.009 - dz * 0.02), (r - dr * 0.024 + 0.001, z - dz * 0.024)][::-1],
                              seg=(n_seam * 2 if rnd else None)))
        for o in seams:
            assign(o, lead)
            paint(o, special=lambda co, nrm: 0.55 * verdigris(co, nrm), wear=lambda co, nrm: np.clip(0.8 - 0.9 * verdigris(co, nrm), 0, 0.8), grime=0.15)

    # ------------------------------------------------------------------ hips
    if not rnd and not is_cu:
        hr = rng.fork("hips")
        s_end = s_cb - 0.02
        hwid = 0.10 * hs_                                   # half width of a hip tile (0.20 m tiles)
        hsp = 0.125 * hs_                                   # exposed gauge
        Lh = hsp * 1.5
        th = 0.015
        hbase = None
        hm_, hgr, rolls, cores = [], [], [], []
        for k in range(N):
            a = 2 * pi * k / N
            ah = a + pi / N
            nf, _ = _dirs(a)
            dh, _ = _dirs(ah)

            def station(s):
                s = clamp(s, 0.0, Stot)
                r, z = at(s)
                r2, z2 = at(min(s + 0.2, Stot))
                if s + 0.2 > Stot:
                    ra, za = at(Stot - 0.2)
                    tv = _unit(dh * (r2 - ra) * kc + ZV * (z2 - za))
                else:
                    tv = _unit(dh * (r2 - r) * kc + ZV * (z2 - z))
                nh = _unit(dh * tv[2] - ZV * float(np.dot(tv, dh)))
                xh = np.cross(tv, nh)
                dr, dz = slope(s, 0.2)
                m = nf * dz - ZV * dr
                T = abs(float(np.dot(m, xh))) / max(float(np.dot(m, nh)), 1e-3)
                return dh * r * kc + ZV * z, tv, nh, xh, T

            if cov == "tile":
                # half-round hip tiles following the bell-cast, each nested on the one below
                Tm = station(0.6 * s_end)[4]
                Szr = Tm * hwid + 0.032
                if hbase is None:
                    hbase = _hip_tile_mesh(hwid, Lh, th, Szr, 0.75 * Lh / hsp, hip_seg)
                nst = max(2, int((s_end + 0.03 - Lh) / hsp) + 1)
                for j in range(nst):
                    s = j * hsp
                    P0, tv, nh, xh, T = station(s + 0.3 * hsp)
                    P0 = station(s)[0]
                    Sz = T * hwid + 0.032
                    zE = -T * hwid + 0.028 * sqrt(1.0 + T * T) + 0.006
                    psi = radians(clamp(hr.jitter(0.0, 0.5 + 1.6 * age), -3.0, 3.0))
                    X2 = xh * cos(psi) + tv * sin(psi)
                    Y2 = -xh * sin(psi) + tv * cos(psi)
                    hm_.append(_frame(X2 * hr.u(0.985, 1.025), Y2, nh * Sz * hr.u(0.985, 1.025), P0 + nh * (zE + hr.u(0.0, 0.003))))
                    hgr.append((hr.u(0, 1) ** 2.0) * 0.5 * (0.5 + age))
                # continuous mortar bedding: fills under the tiles and shows as a fillet along both wing edges
                e_ = 0.028 * sqrt(1.0 + Tm * Tm)
                zEm = -Tm * hwid + e_ + 0.006
                Szm = (Szr - th * Lh / hsp) * 0.95
                wm = hwid + 0.013
                cprof_ = [(wm, -Tm * wm + 0.45 * e_), (hwid + 0.005, zEm + 0.014), (0.62 * hwid, zEm + 0.72 * Szm), (0.0, zEm + Szm),
                          (-0.62 * hwid, zEm + 0.72 * Szm), (-(hwid + 0.005), zEm + 0.014), (-wm, -Tm * wm + 0.45 * e_), (0.0, -0.03)]
                sk = s_of_z(zk)
                sl_ = sorted(set([float(v) for v in np.linspace(0.012, min(sk, s_end), 7)] + [float(v) for v in np.linspace(min(sk, s_end), s_end + 0.04, 7)]))
                pts = [tuple(station(s)[0]) for s in sl_]
                _, _, nq0, _, _ = station(0.0)
                cores.append(sweep(f"hip_bed{k}", pts, cprof_, smooth_path=False, up=tuple(nq0)))
            else:
                P0, tv, nh, xh, T = station(Stot * 0.55)
                roll_off = 0.008 + 1.85 * tthick * (1.0 + 0.1 * age * ragged)     # clear of the lifted tails
                pts = []
                for s in np.linspace(0.0, s_end + 0.06, max(6, int(s_end / 0.34) + 2)):
                    Pq, _, nq, _, _ = station(s)
                    pts.append(tuple(Pq + nq * roll_off))
                wwl, rr, tl = 0.088, 0.027, 0.006
                c0 = tl + 0.1 * rr
                prof = [(wwl, -wwl * T), (wwl, -wwl * T + tl), (rr * 1.25, -rr * 1.25 * T + tl)]
                prof += [(rr * cos(radians(t_)), c0 + rr * sin(radians(t_))) for t_ in (15, 50, 90, 130, 165)]
                prof += [(-rr * 1.25, -rr * 1.25 * T + tl), (-wwl, -wwl * T + tl), (-wwl, -wwl * T), (0.0, -0.012)]
                _, _, n0_, _, _ = station(0.0)
                rolls.append(sweep(f"hip_roll{k}", pts, prof, smooth_path=False, up=tuple(n0_), scale_fn=lambda t: 1.0 - 0.42 * t))
        if hm_:
            hip_extra = 0.03
            assign(hbase, cmats[0])
            hips = instances_to_mesh("hip_tiles", hbase, hm_, keep_base=False, seed=seed + 9)
            nvh = len(hips.data.vertices) // len(hm_)
            hgv = np.repeat(np.array(hgr), nvh)
            paint(hips, grime=lambda co, nrm: np.maximum(soot(co, nrm), hgv) if len(hgv) == len(co) else soot(co, nrm), special=moss)
            mort = stone if p.cornice == "stone" else mat("mortar", "#b9ad98")
            for o in cores:
                assign(o, mort)
                paint(o, grime=lambda co, nrm: np.clip(0.5 + 0.4 * streaks(co, nrm), 0, 1) * (0.6 + 0.6 * age), special=lambda co, nrm: 0.8 * moss(co, nrm))
        for o in rolls:
            assign(o, lead)
            paint(o, grime=lambda co, nrm: 0.3 + 0.4 * streaks(co, nrm))

    # ------------------------------------------------------------------ apex cap + finial
    r0, z0 = at(s_cb)
    co_ = 0.012 if is_cu else 0.04 + hip_extra
    cprof = [(r0 + co_ - 0.02, z0 - 0.03), (r0 + co_ + 0.012, z0 - 0.024), (r0 + co_ + 0.02, z0 - 0.008), (r0 + co_ + 0.012, z0 + 0.008),
             (r0 + co_ + 0.002, z0 + 0.03)]
    for s in np.linspace(s_cb, Stot, 5)[1:]:
        r, z = at(s)
        cprof.append((r + co_ * (1.0 - 0.45 * (s - s_cb) / max(Stot - s_cb, 1e-3)), z + 0.012))
    cprof.append((0.03 / kc, H + 0.05))
    cap = ring("apex_cap", cprof)
    capm = lead if is_cu else mat("lead", rough=0.7)
    assign(cap, capm)
    if is_cu:
        paint(cap, special=0.2, wear=0.55)
    else:
        paint(cap, grime=lambda co, nrm: 0.12 + 0.3 * streaks(co, nrm), wear=0.45)
    fh = float(p.finial_height)
    fr = (0.5 + 0.45 * fh) * clamp(D / 3.4, 0.8, 1.25)
    capr = (rz(H - 0.1) + co_) * kc
    collar = lathe("finial_collar", [(max(0.085 * fr, capr + 0.01), H - 0.1), (max(0.09 * fr, capr + 0.014), H - 0.05), (0.07 * fr, H),
                                     (0.076 * fr, H + 0.02), (0.05 * fr, H + 0.045), (0.034 * fr, H + 0.09)], segments=16)
    assign(collar, capm)
    if is_cu:
        paint(collar, special=0.15, wear=0.5)
    else:
        paint(collar, wear=0.4)
    zb0 = H + 0.06
    sp = fh - 0.06
    stem = lathe("finial_stem", [(0.03 * fr, zb0), (0.024 * fr, zb0 + 0.10 * sp), (0.027 * fr, zb0 + 0.13 * sp), (0.05 * fr, zb0 + 0.15 * sp),
                                 (0.05 * fr, zb0 + 0.165 * sp), (0.027 * fr, zb0 + 0.185 * sp), (0.022 * fr, zb0 + 0.22 * sp),
                                 (0.022 * fr, zb0 + 0.49 * sp), (0.04 * fr, zb0 + 0.51 * sp), (0.04 * fr, zb0 + 0.525 * sp),
                                 (0.02 * fr, zb0 + 0.545 * sp), (0.018 * fr, zb0 + 0.63 * sp), (0.03 * fr, zb0 + 0.65 * sp),
                                 (0.016 * fr, zb0 + 0.68 * sp), (0.0125 * fr, zb0 + 0.84 * sp), (0.0, zb0 + 0.85 * sp)], segments=12)
    assign(stem, mat("iron_wrought"))
    paint(stem, special=lambda co, nrm: 0.5 * fine_n(co, nrm) * (0.4 + age))
    if p.finial_metal == "gilt":
        fm = mat("metal_plain", "#d4a84b", rough=0.55)
    elif p.finial_metal == "copper":
        fm = mat("copper_patina", rough=1.2, scale=2.2) if is_cu else mat("copper_patina")
    else:
        fm = mat("iron_wrought")
    Rb = min(0.14 * fr, 0.16 * sp)
    fin = []
    for nm, zc_, rb_, sg in (("finial_ball", zb0 + 0.345 * sp, Rb, 20), ("finial_knob", zb0 + 0.59 * sp, min(0.052 * fr, 0.05 * sp), 12)):
        bl = sphere(nm, rb_, loc=(0, 0, zc_), segments=sg, rings=sg // 2, scale=(1.0, 1.0, 0.92))
        smooth(bl, 60)
        fin.append(bl)
    # weathervane socket: a ferrule with a moulded collar, open at the top, with a pinch bolt
    sk0 = zb0 + 0.79 * sp
    sl0 = 0.21 * sp
    sock = lathe("vane_socket", [(0.014 * fr, sk0), (0.03 * fr, sk0 + 0.08 * sl0), (0.043 * fr, sk0 + 0.16 * sl0), (0.043 * fr, sk0 + 0.24 * sl0),
                                 (0.028 * fr, sk0 + 0.32 * sl0), (0.028 * fr, sk0 + 0.9 * sl0), (0.033 * fr, sk0 + 0.92 * sl0), (0.033 * fr, sk0 + sl0),
                                 (0.018 * fr, sk0 + sl0), (0.018 * fr, sk0 + 0.7 * sl0), (0.0, sk0 + 0.7 * sl0)], segments=14)
    fin.append(sock)
    fin.append(cylinder("vane_pin", 0.007 * fr, 0.09 * fr, loc=(0.005 * fr, 0, sk0 + 0.62 * sl0), rot=(0, 90, 0), verts=6))
    for bl in fin:
        assign(bl, fm)
        if p.finial_metal == "gilt":
            paint(bl, wear=lambda co, nrm: 0.45 + 0.4 * np.clip(nrm[:, 2] + 0.3, 0, 1), grime=lambda co, nrm: np.clip(-nrm[:, 2], 0, 1) * 0.5, special=0.1 * age)
        elif p.finial_metal == "copper":
            paint(bl, special=lambda co, nrm: 0.25 + 0.5 * np.clip(nrm[:, 2], 0, 1), wear=lambda co, nrm: 0.6 * np.clip(-nrm[:, 2] + 0.2, 0, 1))
        else:
            paint(bl, special=0.2 + 0.3 * age)
    socket("weathervane", "attach", (0, 0, H + fh), role="weathervane")

    # ------------------------------------------------------------------ lucarnes
    if luc_angles:
        parts = []
        ow, sill = lw - 0.15 * ls, 0.07 * ls
        oh = lh + rise * 0.5 - sill
        spring = lh - 0.13 * ls - sill
        arch = [(x, y + sill) for x, y in arch_pts(ow, oh, spring=spring, kind="pointed", n=8)]
        fr_d = 0.05
        frame = shape_extrude("luc_frame", [(-lw / 2, 0), (lw / 2, 0), (lw / 2, lh), (0, lh + rise), (-lw / 2, lh)], [arch],
                              depth=fr_d, rot=(90, 0, 0), bevel=0.004)
        assign(frame, paintm)
        parts.append(frame)
        glass = extrude_profile("luc_glass", [(x * 1.03, y) for x, y in arch], 0.004, loc=(0, -0.012, 0), rot=(90, 0, 0), cap=True)
        # the lit pane is ONE sheet (its front, 16 mm out): a thin glowing box is two sheets that fight each other.
        # CLEARANCE RULE: nothing lies parallel within 4 mm of it - the bars run through it, 4.5 mm behind
        lod_faces(glass, keep=(0, -1, 0), above=-1)
        assign(glass, mat("window_lit", "#ffc37a", glow=1.4, curtain=0.7, rough=3.0) if p.lit else mat("window_dark"))
        bw = 0.02 * ls
        for nm, size, loc in (("luc_mullion", (bw, 0.0255, oh - 0.005), (0, -0.02425, sill + oh / 2)),
                              ("luc_transom", (ow, 0.0255, bw), (0, -0.02425, sill + spring)),
                              ("luc_transom2", (ow, 0.0255, bw * 0.8), (0, -0.02425, sill + spring * 0.5))):
            o = box(nm, size, loc=loc, bevel=0.002, segments=1)
            assign(o, paintm)
            parts.append(o)
        # weathered timber sill, just proud of the frame; the lead apron is tucked under it
        sl_ = box("luc_sill", (lw + 0.07, 0.085, 0.042), loc=(0, -0.058, 0.02), rot=(8, 0, 0), bevel=0.005, segments=1)
        assign(sl_, paintm)
        parts.append(sl_)
        for o in parts:
            paint(o, wear=0.3 * age + 0.1, grime=lambda co, nrm: np.clip(0.15 + 0.45 * (co[:, 2] < sill).astype(float), 0, 1), special=0.3 * age)
        parts.append(glass)
        # lead-dressed body (cheeks), gabled roof with a roll and knob
        d_ridge = r_b - rz(min(zb + lh + rise, H)) + 0.04
        body = extrude_profile("luc_body", [(-lw / 2 + 0.012, 0), (lw / 2 - 0.012, 0), (lw / 2 - 0.012, lh), (0, lh + rise - 0.012), (-lw / 2 + 0.012, lh)],
                               d_ridge, loc=(0, d_ridge, 0), rot=(90, 0, 0))
        lparts = [body]
        gp = atan2(rise, lw / 2)
        ovl = 0.07 * ls
        sl = hypot(lw / 2, rise) + ovl
        ylen = d_ridge + 0.1 * ls
        ymid = d_ridge - ylen / 2
        for sx in (-1, 1):
            cx = sx * (cos(gp) * sl / 2 + sin(gp) * 0.016)
            cz = lh + rise - sin(gp) * sl / 2 + cos(gp) * 0.016
            lparts.append(box(f"luc_roof{sx}", (sl, ylen, 0.032), loc=(cx, ymid, cz), rot=(0, sx * degrees(gp), 0), bevel=0.004, segments=1))
        lparts.append(cylinder("luc_roll", 0.024 * ls, ylen + 0.01, loc=(0, ymid - 0.005, lh + rise + 0.03), rot=(90, 0, 0), verts=8))
        zr = lh + rise + 0.03
        lparts.append(lathe("luc_knob", [(0.02 * ls, zr), (0.03 * ls, zr + 0.03 * ls), (0.016 * ls, zr + 0.055 * ls), (0.036 * ls, zr + 0.09 * ls),
                                         (0.03 * ls, zr + 0.125 * ls), (0.01 * ls, zr + 0.16 * ls), (0.0, zr + 0.24 * ls)],
                            segments=8, loc=(0, d_ridge - ylen + 0.035, 0)))

        # --- sheet lead where the lucarne meets the roof: apron, stepped soakers up the cheeks, valleys
        fo = 0.008 if is_cu else 0.012 + 1.85 * tthick     # clear of the tile tails

        def lsurf(x, s, o_=0.0):
            """point on (o_ above) the main roof surface in lucarne space: x across, s up the slope."""
            r, z = at(s)
            dr, dz = slope(s, 0.05)
            R = r + o_ * dz
            y = r_f - (sqrt(max(R * R - x * x, 1e-6)) if rnd else R)
            return (x, y, z - o_ * dr - zb)

        out_h = (0.0, -1.0, 0.25)
        fl = _Sheet()
        lrng = rng.fork("flashing")
        s_sill, s_eave = s_of_z(zb), s_of_z(min(zb + lh, H))
        # apron: tucked under the sill, dressed about 0.15 m down over the course below
        xe = lw / 2 + 0.085
        xin = lw / 2 + 0.03
        xs_ = [-xe, -xin - 0.008, -xin, -0.22 * lw, 0.0, 0.22 * lw, xin, xin + 0.008, xe]
        lap = [0.16 + lrng.jitter(0.0, 0.009) for _ in xs_]
        rows_a = []
        r_top = []
        for x in xs_:
            q_ = lsurf(x, s_sill + 0.014, fo)
            r_top.append((x, min(q_[1], -0.056) if abs(x) <= xin + 0.001 else q_[1], q_[2]))
        rows_a.append(r_top)
        rows_a.append([lsurf(x, s_sill - 0.05, fo) for x in xs_])
        rows_a.append([lsurf(x, s_sill - 0.11, fo) for x in xs_])
        rows_a.append([lsurf(x, s_sill - lp, fo * 0.92) for x, lp in zip(xs_, lap)])
        rows_a.append([lsurf(x, s_sill - lp - 0.003, fo - 0.022) for x, lp in zip(xs_, lap)])
        for i in range(len(rows_a) - 1):
            hint = out_h if i < 3 else (0.0, -0.4, -1.0)
            for j in range(len(xs_) - 1):
                fl.quad(rows_a[i][j], rows_a[i][j + 1], rows_a[i + 1][j + 1], rows_a[i + 1][j], hint)
        for sx, j in ((-1, 0), (1, len(xs_) - 1)):
            for i in range(3):
                dlist = (s_sill + 0.014, s_sill - 0.05, s_sill - 0.11, s_sill - lap[j])
                a0, a1 = rows_a[i][j], rows_a[i + 1][j]
                b0 = lsurf(sx * (xe + 0.004), dlist[i], fo - 0.022)
                b1 = lsurf(sx * (xe + 0.004), dlist[i + 1], fo - 0.022)
                fl.quad(a0, a1, b1, b0, (sx, 0, 0))
        # stepped soakers: one lead piece per course, each lapping the one below, turned up the cheek
        xi = lw / 2 - 0.012
        nS = max(2, int(round((s_eave - s_sill) / 0.14)))
        stp = (s_eave - s_sill + 0.02) / nS
        for sx in (-1, 1):
            for j in range(nS):
                s0 = s_sill - 0.02 + j * stp - 0.012
                s1 = s0 + stp + 0.03
                o0, o1 = fo + 0.009, fo - 0.001
                xo = lw / 2 + 0.085 + lrng.jitter(0.0, 0.008)
                A, B, C, Dd = lsurf(sx * xi, s0, o0), lsurf(sx * xo, s0, o0), lsurf(sx * xo, s1, o1), lsurf(sx * xi, s1, o1)
                fl.quad(A, B, C, Dd, out_h)
                fl.quad(B, C, lsurf(sx * (xo + 0.003), s1, o1 - 0.022), lsurf(sx * (xo + 0.003), s0, o0 - 0.022), (sx, 0, 0))
                fl.quad(A, B, lsurf(sx * xo, s0 - 0.003, o0 - 0.022), lsurf(sx * xi, s0 - 0.003, o0 - 0.022), (0.0, -0.4, -1.0))
                # cover flashing up the cheek, stepped
                xu = sx * (xi + 0.005)
                yf = max(Dd[1] - 0.075, -0.049)
                fl.quad((xu, A[1], A[2]), (xu, Dd[1], Dd[2]), (xu, yf, Dd[2]), (xu, min(yf, A[1]), A[2]), (sx, 0, 0))
        # valleys where the lucarne roof runs into the main roof, with a saddle piece at the ridge
        xe2 = lw / 2 + ovl * cos(gp)
        vw = 0.085
        for sx in (-1, 1):
            prev = None
            for j in range(4):
                x = xe2 * j / 3.0
                sj = s_of_z(min(zb + lh + rise + 0.045 - x * tan(gp), H))
                inn = lsurf(sx * x, sj - 0.01, fo - 0.012)
                outp = lsurf(sx * (x + vw * sin(gp)), sj + vw * cos(gp), fo + 0.003)
                if prev is not None:
                    fl.quad(prev[0], inn, outp, prev[1], out_h)
                prev = (inn, outp)
        s_r = s_of_z(min(zb + lh + rise + 0.045, H))
        fl.quad(lsurf(0.0, s_r - 0.01, fo - 0.012), lsurf(vw * sin(gp), s_r + vw * cos(gp), fo + 0.003),
                lsurf(0.0, s_r + vw / max(cos(gp), 0.3), fo + 0.003), lsurf(-vw * sin(gp), s_r + vw * cos(gp), fo + 0.003), out_h)
        flash = fl.obj("luc_flashing")
        lparts.append(flash)
        for o in lparts:
            assign(o, lead)
            if is_cu:
                paint(o, special=0.25 + 0.3 * age, wear=0.3)
            else:
                paint(o, grime=0.45, wear=0.2)
        luc = join(parts + lparts, "lucarne_base")
        lmats = []
        for i, la in enumerate(luc_angles):
            nd, td = _dirs(la)
            lmats.append(trs(tuple(nd * r_f + ZV * zb), (0, 0, degrees(la))))
            if p.lit and i < 4:
                # window spill onto the roof: in front of and just below the sill, so that seen from the square its
                # mirror image never lands in the glass and the joinery is not burnt out
                light(f"lucarne_{i}", tuple(nd * (r_f + 0.36 * ls) + ZV * (zb - 0.1 * ls)), color="#ffbf6a", intensity=(0.42 if is_cu else 0.6) * ls,
                      range=4.0, flicker=0.1, shadow_near=0.3)
        instances_to_mesh("lucarnes", luc, lmats, uv_jitter=False, keep_base=False)
    print(f"DBG eaves_r={Re:.2f} wall_r={Rw:.2f} cornice_h={zc:.2f} roof_apex={H:.2f} top={H + fh:.2f} lucarnes={len(luc_angles)}")
