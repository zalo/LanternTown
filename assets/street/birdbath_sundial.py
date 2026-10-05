"""Garden-square pedestal carved from limestone: jointed base step(s), a plinth block and a turned
baluster (vase with gadrooned belly and a collar of eight water-leaves / fluted column / barley twist /
panelled square pier) carrying either a stone table on a turned bed mould with a cast-bronze horizontal sundial
(dull verdigris ground, chapter ring with raised Roman numerals, hour lines laid out for the latitude, pierced
gnomon screwed down through a foot flange, motto) or a shallow gadrooned birdbath bowl with still dark water
(tide line, floating leaves) and a little patinated bronze sparrow on the rim.
Old and settled: slight lean, damp base, algae run-off under the drip line, crusty lichen rosettes and moss
cushions (both grown onto the finished stone and snapped to its surface) on the ledges and the north side.
"""
from tk import *
from math import sin, cos, tan, pi, radians, degrees, atan2, sqrt, hypot
import numpy as np
import bmesh
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

META = {"name": "Sundial / birdbath pedestal", "class": "prop", "mount": "ground", "focus": 0.86}
PARAMS = {
    "top": {"default": "sundial", "choices": ["sundial", "birdbath"], "desc": "what the pedestal carries"},
    "height": {"default": 1.05, "min": 0.75, "max": 1.4, "desc": "height of the table top / bowl rim (m)"},
    "baluster": {"default": "vase", "choices": ["vase", "fluted", "twist", "square"], "desc": "shaft style"},
    "plan": {"default": "octagon", "choices": ["square", "octagon", "round"], "desc": "plan of steps, plinth and sundial table"},
    "steps": {"default": 1, "min": 0, "max": 2, "desc": "number of base steps under the plinth"},
    "base_width": {"default": 0.46, "min": 0.36, "max": 0.6, "desc": "width of the plinth block (m)"},
    "dial": {"default": 0.34, "min": 0.26, "max": 0.44, "desc": "diameter of the bronze dial plate (m)"},
    "latitude": {"default": 50.0, "min": 35.0, "max": 60.0, "desc": "latitude the dial is laid out for (gnomon angle, deg)"},
    "motto": {"default": "TEMPUS FUGIT", "desc": "motto cast into the south arc of the dial"},
    "bowl": {"default": 0.64, "min": 0.45, "max": 0.85, "desc": "birdbath bowl diameter (m)"},
    "bowl_style": {"default": "gadroon", "choices": ["gadroon", "plain"], "desc": "underside of the bowl"},
    "water": {"default": 0.8, "min": 0.0, "max": 1.0, "desc": "water level in the bowl (0 = dry)"},
    "bird": {"default": True, "desc": "bronze sparrow perched on the bowl rim"},
    "inscription": {"default": "MDCCCXC", "desc": "lead-filled inscription on the plinth front ('' = none; flat-fronted plans only)"},
    "stone": {"default": "#b9ad98", "desc": "stone colour"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "amount of lichen / moss"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "grime, drip streaks and wear"},
    "lean": {"default": 0.5, "min": 0.0, "max": 2.5, "desc": "how far the pedestal has settled out of plumb (deg)"},
}
VARIANTS = {
    "birdbath_fluted": {"top": "birdbath", "baluster": "fluted", "plan": "round", "height": 0.84, "bowl": 0.68, "moss": 0.5,
                        "stone": "#c2b49a", "seed": 4},
    "twist_tall": {"baluster": "twist", "plan": "square", "steps": 2, "height": 1.28, "dial": 0.3, "base_width": 0.42,
                   "stone": "#a3a39c", "moss": 0.12, "motto": "CARPE DIEM", "inscription": "ANNO 1790", "seed": 9},
    "mossy_pier_bath": {"top": "birdbath", "baluster": "square", "plan": "square", "steps": 0, "height": 0.78, "bowl": 0.56,
                        "bowl_style": "plain", "water": 0.35, "moss": 0.95, "age": 0.9, "lean": 1.8, "stone": "#9f9a88",
                        "inscription": "W. H.", "bird": False, "seed": 15},
    "round_fluted_dial": {"baluster": "fluted", "plan": "round", "steps": 2, "height": 1.12, "dial": 0.42, "base_width": 0.5,
                          "stone": "#cbbfa6", "moss": 0.25, "age": 0.3, "motto": "HORAS NON NUMERO NISI SERENAS", "seed": 23},
}


# ------------------------------------------------------------------ profile helpers (r, z)
def _bez2(p0, p1, p2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        s = 1.0 - t
        out.append((s ** 3 * p0[0] + 3 * s * s * t * p1[0] + 3 * s * t * t * p2[0] + t ** 3 * p3[0],
                    s ** 3 * p0[1] + 3 * s * s * t * p1[1] + 3 * s * t * t * p2[1] + t ** 3 * p3[1]))
    return out


def _bead(r, z0, h, out, n=5):
    """Half-round torus projecting `out` from radius r between z0 and z0 + h."""
    return [(r + out * sin(pi * i / n), z0 + h * 0.5 * (1 - cos(pi * i / n))) for i in range(n + 1)]


def _quart(p0, p1, kind, n=5):
    """Quarter round from p0 to p1. 'a' leaves p0 horizontally, arrives vertically; 'b' the reverse."""
    out = []
    for i in range(1, n + 1):
        a = pi / 2 * i / n
        if kind == "a":
            out.append((p0[0] + (p1[0] - p0[0]) * sin(a), p0[1] + (p1[1] - p0[1]) * (1 - cos(a))))
        else:
            out.append((p0[0] + (p1[0] - p0[0]) * (1 - cos(a)), p0[1] + (p1[1] - p0[1]) * sin(a)))
    return out


def _clean(pts):
    out = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - out[-1][0]) > 1e-5 or abs(q[1] - out[-1][1]) > 1e-5:
            out.append((max(q[0], 0.0), q[1]))
    return out


def _attic(z, rb, s):
    """Attic base: torus, scotia, torus. Returns (profile, z_top, r_top)."""
    pts = [(rb - 0.02 * s, z)]
    pts += _bead(rb - 0.02 * s, z, 0.042 * s, 0.02 * s, 4)
    pts += [(rb - 0.026 * s, z + 0.047 * s)]
    pts += _quart((rb - 0.026 * s, z + 0.047 * s), (rb - 0.05 * s, z + 0.078 * s), "a", 3)
    pts += [(rb - 0.05 * s, z + 0.083 * s)]
    pts += _bead(rb - 0.052 * s, z + 0.083 * s, 0.028 * s, 0.014 * s, 3)
    pts += [(rb - 0.06 * s, z + 0.116 * s)]
    return pts, z + 0.116 * s, rb - 0.06 * s


def _edge_dist(n, ap, ang):
    """Distance from the axis to the edge of an n-gon (flats facing the axes) in direction ang (rad)."""
    if n > 8:
        return ap
    sec = 2 * pi / n
    loc = ((ang + sec / 2) % sec) - sec / 2
    return ap / cos(loc)


def _ngon(name, n, prof, bev=0.004):
    """Lathe a profile given in apothem distances as an n-gon prism with flats facing the axes."""
    if n > 8:
        return lathe(name, _clean(prof), segments=n)
    k = 1.0 / cos(pi / n)
    o = lathe(name, _clean([(r * k, z) for r, z in prof]), segments=n, rot=(0, 0, 180.0 / n), smooth_angle=25)
    if bev > 0:
        bevel(o, bev, 1)
    return o


def _ctext(name, body, size, depth, font="roman", lim=14.0, walls=True):
    """Lettering lying flat in XY (reads along +X, extruded +Z), centred on the origin. Returns (obj, w, h)."""
    o = text(name, body, size=size, depth=depth, rot=(0, 0, 0), font=font)
    lo, hi = world_bbox([o])
    c = (lo + hi) * 0.5
    transform_verts(o, lambda co: Vector((co.x - c.x, co.y - c.y, co.z - lo.z)))
    bm = bmesh.new()                                 # drop the hidden back cap, merge needless curve points
    bm.from_mesh(o.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z < (-0.5 if walls else 0.5)], context="FACES")
    bmesh.ops.dissolve_limit(bm, angle_limit=radians(lim), verts=bm.verts[:], edges=bm.edges[:], delimit={"NORMAL"})
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(o.data)
    bm.free()
    return o, hi.x - lo.x, hi.y - lo.y


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"}


SEG = 20                     # segments of plain turned parts (the carved parts get their own, denser grids)
UST = 0.16                   # dressed stone UVs are squeezed along U, so the stone's (diagonal) veining lies down as
UROT = 0.0                   # near-level bedding / turning rings instead of streaking across the smooth faces


def _seg(r):
    """Radial segments of a plain turned moulding of radius r."""
    return 16 if r < 0.08 else 20 if r < 0.16 else 24


def _bezpt(P, t):
    s = 1.0 - t
    return (s ** 3 * P[0][0] + 3 * s * s * t * P[1][0] + 3 * s * t * t * P[2][0] + t ** 3 * P[3][0],
            s ** 3 * P[0][1] + 3 * s * s * t * P[1][1] + 3 * s * t * t * P[2][1] + t ** 3 * P[3][1])


def _insert_z(prof, z):
    """Insert a point at height z into a profile (so a vertex ring sits exactly there)."""
    out = [prof[0]]
    for (r0, z0), (r1, z1) in zip(prof[:-1], prof[1:]):
        if (z0 - z) * (z1 - z) < 0 and abs(z1 - z0) > 1e-6:
            out.append((r0 + (r1 - r0) * (z - z0) / (z1 - z0), z))
        out.append((r1, z1))
    return out


def _rev_grid(name, angs, rows, fn=None, smooth_angle=45.0):
    """Quad shell around Z: one vertex column per angle in `angs` (rad, increasing over one turn), one ring
    per (r, z) in `rows` (bottom to top). fn(i, a, j, r, z) -> (r, z, a) carves each vertex."""
    nc, nr = len(angs), len(rows)
    verts = []
    for j, (r, z) in enumerate(rows):
        for i, a in enumerate(angs):
            rr, zz, aa = fn(i, a, j, r, z) if fn else (r, z, a)
            verts.append((rr * cos(aa), rr * sin(aa), zz))
    faces = []
    for j in range(nr - 1):
        for i in range(nc):
            i2 = (i + 1) % nc
            faces.append((j * nc + i, j * nc + i2, (j + 1) * nc + i2, (j + 1) * nc + i))
    o = mesh_object(name, verts, faces)
    uv_cyl(o)
    smooth(o, smooth_angle)
    return o


_LOBES = {}                                       # name -> (lobes, start angle) of the gadrooned parts, for the dirt in their creases
G_US = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)          # columns across one gadroon: crease, flank, shoulder, crown, ...
G_HS = (0.0, 0.6, 0.87, 1.0, 0.87, 0.6)           # half-round section


def _gadroons(name, rows, nl, depth, env, wid, a0=-pi / 2):
    """Convex half-round lobes with a sharp crease between them, carved into the surface `rows`.
    env / wid: per-row relief (0..1) and lobe width (0..1, narrows into rounded tongue ends)."""
    pitch = 2 * pi / nl
    angs = [a0 + (i - 0.5 + u) * pitch for i in range(nl) for u in G_US]

    def carve(i, a, j, r, z):
        k = i % 6
        ac = a0 + (i // 6) * pitch
        if k:
            a = ac + (G_US[k] - 0.5) * pitch * wid[j]
        return r + depth * env[j] * (G_HS[k] - 0.62), z, a

    _LOBES[name] = (nl, a0)
    return _rev_grid(name, angs, rows, carve, smooth_angle=42)


# ------------------------------------------------------------------ base courses
def _course(name, n, ap, z0, z1, split, r, inner=0.0):
    """One stone course of the base: jointed blocks (square plan) or a single dressed drum / octagon.
    Returns (objects, open joints as [(p0, p1), ...] on the top surface)."""
    objs, joints = [], []
    h = z1 - z0
    if n == 4:
        j = 0.003                                      # half joint width
        jx = r.jitter(0.0, ap * 0.22) if split >= 2 else None
        jy = r.jitter(0.0, ap * 0.22) if split >= 4 else None
        xs = [(-ap, ap)] if jx is None else [(-ap, jx - j), (jx + j, ap)]
        ys = [(-ap, ap)] if jy is None else [(-ap, jy - j), (jy + j, ap)]
        if split == 2 and r.chance(0.5):
            xs, ys = ys, xs
        if len(xs) > 1:
            jc = (xs[0][1] + xs[1][0]) / 2
            joints += [((jc, -ap, z1), (jc, -inner, z1)), ((jc, inner, z1), (jc, ap, z1))]
        if len(ys) > 1:
            jc = (ys[0][1] + ys[1][0]) / 2
            joints += [((-ap, jc, z1), (-inner, jc, z1)), ((inner, jc, z1), (ap, jc, z1))]
        for (x0, x1) in xs:
            for (y0, y1) in ys:
                dz = r.u(-0.003, 0.0) if split > 1 else 0.0
                b = stone_block(name, (x1 - x0, y1 - y0, h + dz), loc=((x0 + x1) / 2, (y0 + y1) / 2, z0 + (h + dz) / 2),
                                chamfer=0.011, rough=0.003, rng=r, detail=0.13)
                objs.append(b)
    else:
        ch = 0.014
        prof = [(ap - 0.004, z0), (ap, z0 + 0.008), (ap, z0 + h * 0.5), (ap, z1 - ch), (ap - ch * 0.4, z1 - ch * 0.25), (ap - ch, z1),
                ((ap + inner) * 0.5, z1), (max(inner - 0.03, 0.0), z1)]
        o = _ngon(name, n, prof, bev=0.005)
        objs.append(o)
    return objs, joints


# ------------------------------------------------------------------ shafts
def _capital(z2, zc, r1, rb, s):
    pr = [(r1, z2)] + _bead(r1, z2 + 0.004 * s, 0.02 * s, 0.011 * s, 3) + [(r1, z2 + 0.03 * s), (r1, z2 + 0.046 * s)]
    pr += _quart((r1, z2 + 0.046 * s), (rb * 0.76, z2 + 0.082 * s), "a", 4)
    pr += [(rb * 0.78, z2 + 0.088 * s), (rb * 0.78, zc)]
    return lathe("capital", _clean(pr), segments=SEG), rb * 0.78


def _leaf_bell(z0, z1, r0, r1, T, s, nlv=8):
    """Bell (cavetto) with `nlv` carved water-leaves standing T proud of it: each leaf has side walls down to
    the bell (a V cut where two leaves touch near the foot), a raised centre rib, and a tip that curls ~20 mm
    outward, hangs over its own shadow and rolls back up into the bell under the abacus."""
    # per row: v on the bell, v of the leaf surface, relief, outward curl (m), width (fraction of the pitch)
    rows = ((0.00, 0.00, 0.40, 0.000, 0.95), (0.07, 0.07, 0.92, 0.000, 0.95), (0.20, 0.20, 1.00, 0.000, 0.94),
            (0.38, 0.38, 1.00, 0.000, 0.90), (0.54, 0.54, 1.00, 0.000, 0.85), (0.66, 0.66, 1.00, 0.002, 0.78),
            (0.75, 0.74, 1.00, 0.008, 0.70), (0.81, 0.775, 1.00, 0.016, 0.62), (0.86, 0.80, 1.00, 0.021, 0.54),
            (0.91, 0.86, 0.90, 0.016, 0.46), (0.955, 0.93, 0.50, 0.006, 0.38), (1.00, 1.00, 0.00, 0.000, 0.30))
    us = (0.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0)        # valley, foot, edge, blade, rib, blade, edge, foot (in half widths)
    hs = (0.0, 0.0, 0.70, 0.84, 1.0, 0.84, 0.70, 0.0)       # section of the leaf
    cs = (0.0, 0.0, 0.45, 0.84, 1.0, 0.84, 0.45, 0.0)       # the tip is rounded in plan
    h = z1 - z0
    pitch = 2 * pi / nlv
    a0 = -pi / 2
    nk = len(us)
    angs = [a0 + i * pitch for i in range(nlv) for _ in range(nk)]

    def bell(v):
        return r0 + (r1 - r0) * v ** 2.0

    def carve(i, a, j, r, z):
        k = i % nk
        vb, vl, g, c, w = rows[j]
        ac = a0 + (i // nk) * pitch
        if k == 0:                                          # bell showing between the leaves
            return bell(vb), z0 + h * vb, ac - 0.5 * pitch
        if k in (1, nk - 1):                                # foot of the side wall, on the bell
            return bell(vb), z0 + h * vb, ac + us[k] * min(0.5 * w + 0.022, 0.494) * pitch
        rib = 0.003 * s * min(g, 1.0) if k == 4 else 0.0
        return (bell(vl) + T * g * hs[k] + rib + c * s * cs[k], z0 + h * vl, ac + us[k] * (0.5 * w - 0.03) * pitch)

    return _rev_grid("leaf_bell", angs, [(0.0, 0.0)] * len(rows), carve, smooth_angle=64)


def _shaft_vase(zb, zc, rb, s, rng):
    pts, z1, r1 = _attic(zb, rb, s)
    Ls = zc - z1
    rv = rb * 0.88
    base = lathe("vase_base", _clean(pts + [(r1 - 0.02 * s, z1 + 0.001)]), segments=_seg(rb))
    B = ((r1, z1 - 0.001), (rv * 1.18, z1 + 0.02 * Ls), (rv * 1.18, z1 + 0.40 * Ls), (rb * 0.40, z1 + 0.56 * Ls))
    # gadrooned lower belly, lobes stopping in rounded tongues under a carved band
    tb = 0.575
    ts = (0.0, 0.03, 0.08, 0.17, 0.3, 0.42, 0.51, 0.55, tb)
    env = (0.0, 0.55, 1.0, 1.0, 1.0, 1.0, 0.9, 0.5, 0.0)
    wid = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.92, 0.7, 0.4)
    nl = int(clamp(round(2 * pi * rv * 1.1 / 0.075), 12, 14))
    gad = _gadroons("vase_gadroons", [_bezpt(B, t) for t in ts], nl, 0.0115 * s, env, wid)
    rbd, zbd = _bezpt(B, tb)
    p0, p1 = _bezpt(B, tb - 0.03), _bezpt(B, tb + 0.03)
    slope = (p1[0] - p0[0]) / max(p1[1] - p0[1], 1e-4)
    bh = 0.011 * s
    band = lathe("belly_band", [(rbd - slope * 1.7 * bh - 0.004, zbd - 1.7 * bh), (rbd - slope * bh + 0.004 * s, zbd - bh), (rbd + 0.009 * s, zbd - bh * 0.35),
                                (rbd + 0.009 * s, zbd + bh * 0.35), (rbd + slope * bh + 0.004 * s, zbd + bh), (rbd + slope * 1.7 * bh - 0.004, zbd + 1.7 * bh)],
                 segments=_seg(rbd))
    # shoulder, neck and astragal
    up = [_bezpt(B, t) for t in (tb, 0.68, 0.8, 0.9, 1.0)]
    zr = z1 + 0.70 * Ls
    rn = rb * 0.38
    up += _bez2((rb * 0.40, z1 + 0.56 * Ls), (rb * 0.345, z1 + 0.61 * Ls), (rb * 0.345, z1 + 0.66 * Ls), (rn, zr), 3)[1:]
    up += [(rn, zr + 0.002 * s)] + _bead(rn, zr + 0.004 * s, 0.024 * s, 0.014 * s, 4) + [(rn, zr + 0.031 * s), (rn - 0.01, zr + 0.036 * s)]
    upper = lathe("vase_neck", _clean(up), segments=SEG)
    # bell with water-leaves under a plain abacus (all turned on the same axis)
    lz0, lz1 = zr + 0.03 * s, zc - 0.022 * s
    objs = [base, gad, band, upper]
    r_bell = rb * 0.56
    r_ab = rb * 0.75
    if lz1 - lz0 > 0.06:
        objs.append(_leaf_bell(lz0, lz1 + 0.002, rn, r_bell, 0.0135 * s, s))
    else:                                                    # very short pedestal: plain cavetto bell
        objs.append(lathe("bell", _clean([(rn, lz0)] + _quart((rn, lz0), (r_bell, lz1 + 0.002), "b", 5)), segments=SEG))
    c = 0.003 * s
    ab = [(r_bell - 0.012, lz1 - 0.001), (r_ab - c, lz1), (r_ab, lz1 + c), (r_ab, zc - c), (r_ab - c, zc), (r_ab - 0.03, zc)]
    objs.append(lathe("abacus", _clean(ab), segments=_seg(r_ab)))
    return objs, r_ab, r_ab


def _shaft_fluted(zb, zc, rb, s, rng):
    pts, z1, r1 = _attic(zb, rb, s)
    base = lathe("col_base", _clean(pts + [(r1 - 0.02 * s, z1)]), segments=_seg(rb))
    z2 = zc - 0.105 * s
    r0, r2 = rb * 0.60, rb * 0.50
    L = z2 - z1
    nf = int(clamp(round(2 * pi * r0 / 0.034 / 2) * 2, 16, 20))
    pitch = 2 * pi / nf
    fw = 0.385                                                # flute half width / pitch (the rest is the fillet)
    D = min(0.0115 * s, r2 * pitch * fw * 0.92)              # flute depth
    Wl = r0 * pitch * fw * 1.15                              # length of the rounded stops
    f0, f1 = z1 + max(0.03 * L, 0.014), z2 - max(0.025 * L, 0.012)
    us = (-1.0, -0.62, 0.0, 0.62, 1.0)
    qs = (0.0, 0.25, 0.6, 1.0)
    zs = [z1 - 0.002] + [f0 + q * Wl for q in qs] + [f0 + Wl + (f1 - f0 - 2 * Wl) * k / 3.0 for k in (1, 2)]
    zs += [f1 - q * Wl for q in reversed(qs)] + [z2 + 0.002]
    rows = [(r0 + (r2 - r0) * clamp((z - z1) / L) ** 1.5 + (0.004 * s if z < z1 else 0.0), z) for z in zs]
    angs = [-pi / 2 + (i + u * fw) * pitch for i in range(nf) for u in us]

    def carve(i, a, j, r, z):
        q = min((z - f0) / Wl, (f1 - z) / Wl)
        if q < -1e-6:
            return r, z, a
        wq = 1.0 if q >= 1.0 else max(sqrt(max(1.0 - (1.0 - q) ** 2, 0.0)), 0.1)
        u = us[i % 5]
        return r - D * wq * sqrt(max(1.0 - u * u, 0.0)), z, -pi / 2 + (i // 5 + u * fw * wq) * pitch

    sh = _rev_grid("col_shaft", angs, rows, carve, smooth_angle=50)
    cap, rt = _capital(z2, zc, r2, rb, s)
    return [base, sh, cap], rt, rt


def _shaft_twist(zb, zc, rb, s, rng):
    pts, z1, r1 = _attic(zb, rb, s)
    base = lathe("col_base", _clean(pts + [(r1 - 0.02 * s, z1)]), segments=_seg(rb))
    z2 = zc - 0.105 * s
    r0, r2 = rb * 0.55, rb * 0.47
    nz = 26
    sp = [(r0 + 0.006 * s, z1 - 0.002), (r0 + 0.006 * s, z1 + 0.012 * s)]
    sp += [(r0 + (r2 - r0) * (i / nz), z1 + 0.018 * s + (z2 - z1 - 0.018 * s) * (i / nz)) for i in range(nz + 1)]
    sh = lathe("twist_shaft", _clean(sp), segments=24, smooth_angle=70)
    L = z2 - z1
    turns = max(1.0, L / 0.34)
    hand = 1.0 if rng.chance(0.5) else -1.0
    f0, f1 = z1 + 0.03 * s, z2 - 0.004
    fd = min(0.05, L * 0.12)

    def twist(co):
        z = co.z
        if z < f0 or z > f1:
            return co
        e = min(smoothstep(f0, f0 + fd, z), 1.0 - smoothstep(f1 - fd, f1, z))
        a = atan2(co.y, co.x)
        ph = 2.0 * (a - hand * 2 * pi * turns * (z - z1) / L)
        k = 1.0 + e * (0.2 * cos(ph) - 0.03 * cos(2 * ph) - 0.05)
        return Vector((co.x * k, co.y * k, co.z))

    transform_verts(sh, twist)
    cap, rt = _capital(z2, zc, r2, rb, s)
    return [base, sh, cap], rt, rt


def _shaft_square(zb, zc, rb, s, rng):
    ab, a_s, ac = rb * 0.98, rb * 0.62, rb * 0.74
    zs0 = zb + 0.1 * s
    zs1 = zc - 0.09 * s
    pr = [(ab, zb), (ab, zb + 0.04 * s), (ab - 0.008 * s, zb + 0.048 * s)]
    pr += _quart((ab - 0.008 * s, zb + 0.048 * s), (a_s + 0.01 * s, zb + 0.09 * s), "a", 5)
    pr += [(a_s + 0.01 * s, zs0 - 0.004 * s), (a_s, zs0)] + [(a_s, zs0 + (zs1 - zs0) * k / 4.0) for k in (1, 2, 3, 4)]
    pr += [(a_s + 0.008 * s, zs1 + 0.006 * s), (a_s + 0.008 * s, zs1 + 0.02 * s)]
    pr += _quart((a_s + 0.008 * s, zs1 + 0.02 * s), (ac - 0.006 * s, zs1 + 0.056 * s), "b", 5)
    pr += [(ac - 0.006 * s, zs1 + 0.062 * s), (ac, zs1 + 0.066 * s), (ac, zc - 0.006 * s), (ac - 0.005 * s, zc)]
    pier = _ngon("pier", 4, pr, bev=0.006)
    objs = [pier]
    # raised fielded panel frame and a carved boss on each face
    ph = zs1 - zs0 - 0.07 * s
    pw = 2 * a_s - 0.065 * s
    if ph > 0.1:
        fr = shape_extrude("panel", rounded_rect(pw, ph, 0.012, 2), [rounded_rect(pw - 0.034 * s, ph - 0.034 * s, 0.006, 2)],
                           depth=0.009, bevel=0.0025)
        recalc_normals(fr)
        boss = lathe("boss", [(0.036 * s, 0.0), (0.038 * s, 0.005), (0.028 * s, 0.011), (0.02 * s, 0.01), (0.014 * s, 0.017), (0.0, 0.021)], segments=4,
                     rot=(0, 0, 0), smooth_angle=20)
        fm = []
        for k in range(4):
            a = radians(k * 90.0)
            fm.append(trs((a_s * sin(a), -a_s * cos(a), (zs0 + zs1) / 2), (90, 0, k * 90.0)))
        objs.append(instances_to_mesh("panels", fr, fm, keep_base=False))
        objs.append(instances_to_mesh("bosses", boss, fm, keep_base=False))
    return objs, ac * 0.98, ac * sqrt(2.0)


# ------------------------------------------------------------------ tops
def _bar():
    """Unit raised stroke (open underneath, slightly battered sides like a cast letter): x, y in -0.5..0.5, z 0..1."""
    v = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0), (-0.34, -0.485, 1), (0.34, -0.485, 1), (0.34, 0.485, 1), (-0.34, 0.485, 1)]
    o = mesh_object("bar", v, [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    flat(o)
    return o


def _roman_strokes(txt, h, sw):
    """Strokes of a Roman numeral, cap height h: [(x, y, rot_deg, length, width), ...] centred on the origin, and its width."""
    wv, gap = 0.46 * h, 0.17 * h
    out, x = [], 0.0
    for ch in txt:
        if ch == "I":
            out.append((x + sw / 2, 0.0, 0.0, h, sw))
            x += sw + gap
        elif ch == "V":
            a, ln = degrees(atan2(wv / 2, h)), hypot(wv / 2, h)
            out.append((x + sw / 2 + wv * 0.25, 0.0, a, ln, sw * 1.15))
            out.append((x + sw / 2 + wv * 0.75, 0.0, -a, ln, sw * 0.8))
            x += wv + sw + gap
        else:                                           # X
            a, ln = degrees(atan2(wv, h)), hypot(wv, h)
            out.append((x + sw / 2 + wv / 2, 0.0, a, ln, sw * 1.15))
            out.append((x + sw / 2 + wv / 2, 0.0, -a, ln, sw * 0.8))
            x += wv + sw + gap
    W = x - gap
    out = [(sx - W / 2, sy, a, ln, w) for sx, sy, a, ln, w in out]
    for sy in (-1, 1):                                  # serif bars across the head and the foot
        out.append((0.0, sy * h / 2, 90.0, W + 0.16 * h, sw * 0.75))
    return out, W


def _sundial(p, rng, zt, R, bz):
    """Cast-bronze horizontal dial on the table top at z = zt. North (the gnomon's high end) is +Y.
    One dark bronze casting: the sunk ground carries the verdigris, the raised work is rubbed back to brown-gold
    (painted later through the weathering masks). Returns (objects, y of the gnomon toe, gnomon length)."""
    objs = []
    phi = radians(p.latitude)
    t = 0.008
    face = zt + t
    sg = 28
    plate = lathe("dial_plate", [(R - 0.0015, zt), (R, zt + 0.002)] + [(R * k, face) for k in (1.0, 0.9, 0.78, 0.66, 0.52, 0.36, 0.18, 0.0)],
                  segments=sg, smooth_angle=35)
    objs.append(plate)

    def band(r_out, r_in, h, z0=None):
        e = min(0.0009, (r_out - r_in) * 0.3)
        zb_ = face - 0.0004 if z0 is None else z0
        return lathe("ring", [(r_out, zb_), (r_out - e, face + h), (r_in + e, face + h), (r_in, face - 0.0004)], segments=sg, smooth_angle=35)

    objs.append(band(R + 0.0012, 0.925 * R, 0.0034, zt + 0.0035))        # moulded rim
    objs.append(band(0.645 * R, 0.612 * R, 0.0024))                      # chapter ring

    yc = 0.13 * R
    marks = []
    lw = 0.0024

    def hit(th, rho):
        """Point where the hour line at bearing th (from north, towards east) crosses radius rho."""
        sdist = yc * cos(th) + sqrt(max(yc * yc * cos(th) ** 2 - yc * yc + rho * rho, 0.0))
        return sdist, (sdist * sin(th), -yc + sdist * cos(th))

    nh = clamp(0.113 * R, 0.014, 0.024)             # numeral cap height (19 mm on the default dial)
    sw = nh * 0.16
    rel = 0.0024                                    # relief of numerals and motto
    for h in range(-7, 8):
        ha = radians(15.0 * h)
        th = atan2(sin(phi) * sin(ha), cos(ha))
        s1, _ = hit(th, 0.59 * R)
        s0 = 0.2 * R
        sm = (s0 + s1) / 2
        marks.append(trs((sm * sin(th), -yc + sm * cos(th), face - 0.0003), (0, 0, -degrees(th)), (lw, s1 - s0, 0.0017)))
        if h < 7:                                   # half-hour tick
            hb = radians(15.0 * (h + 0.5))
            tb = atan2(sin(phi) * sin(hb), cos(hb))
            a0, _ = hit(tb, 0.5 * R)
            a1, _ = hit(tb, 0.59 * R)
            am = (a0 + a1) / 2
            marks.append(trs((am * sin(tb), -yc + am * cos(tb), face - 0.0003), (0, 0, -degrees(tb)), (lw * 0.8, a1 - a0, 0.0014)))
        # numeral built from raised strokes, heads outward
        _, (nx, ny) = hit(th, 0.785 * R)
        hour = (12 + h - 1) % 12 + 1
        strokes, _w = _roman_strokes(ROMAN[hour], nh, sw)
        Mn = Matrix(trs((nx, ny, face - 0.0003), (0, 0, -degrees(atan2(nx, ny)))))
        for sx, sy, a, ln, w in strokes:
            marks.append(Mn @ Matrix(trs((sx, sy, 0.0), (0, 0, a), (w, ln, rel + 0.0003))))
    objs.append(instances_to_mesh("hour_marks", _bar(), marks, keep_base=False))

    # motto on the south arc, letter by letter, tops towards the centre
    motto = str(p.motto).strip().upper()
    if motto:
        rho = 0.785 * R
        avail = radians(112.0) * rho
        msz = nh * 0.9
        walls = False                               # the motto is cast in low relief (small letters: their faces carry them)
        for attempt in range(2):
            glyphs, total = [], 0.0
            for ch in motto:
                if ch == " ":
                    glyphs.append((None, msz * 0.5))
                    total += msz * 0.5
                else:
                    o, w, hh = _ctext("motto", ch, msz, rel if walls else 0.0009, lim=40.0, walls=walls)
                    glyphs.append((o, w))
                    total += w + msz * 0.2
            if total <= avail or attempt == 1:
                break
            delete([g for g, _ in glyphs if g is not None])
            msz *= max(avail / total, 0.45)
        arc = -total / 2
        for o, w in glyphs:
            if o is None:
                arc += w
                continue
            psi = (arc + w / 2) / rho
            place(o, loc=(rho * sin(psi), -rho * cos(psi), face - 0.0003), rot=(0, 0, degrees(psi)))
            objs.append(o)
            arc += w + msz * 0.2

    # compass star under the gnomon toe
    star = []
    for k in range(16):
        a = radians(k * 22.5)
        rr = (0.17 * R if k % 4 == 0 else 0.115 * R) if k % 2 == 0 else 0.048 * R
        star.append((rr * sin(-a), rr * cos(-a)))
    st = shape_extrude("star", star, depth=0.0022, loc=(0, -yc, face - 0.0004), bevel=0.0004)
    recalc_normals(st)                              # (shape_extrude can come out inside-out)
    objs.append(st)

    # gnomon: pierced bronze plate, style edge at the latitude angle, on a screwed foot flange
    G = 0.73 * R
    hg = G * tan(phi)
    tip = (0.92 * G, 0.92 * hg)
    out = [(0.0, 0.0), (G, 0.0), (G, 0.01)]
    out += _bez2((G, 0.01), (0.42 * G, 0.10 * hg), (0.97 * G, 0.5 * hg), (tip[0] + 0.004, tip[1] - 0.006), 9)[1:]
    out += [tip]
    hole = circle_pts(0.06 * hg, 10, center=(0.5 * G, 0.2 * hg))
    gt = 0.006
    gn = shape_extrude("gnomon", out, [hole], depth=gt, loc=(-gt / 2, -yc, face + 0.003), rot=(90, 0, 90), bevel=0.001)
    recalc_normals(gn)
    objs.append(gn)
    objs.append(box("gnomon_foot", (0.022, G + 0.016, 0.0045), loc=(0, -yc + G / 2, face + 0.0018), bevel=0.001, segments=1))
    scr = lathe("screw", [(0.003, 0.0), (0.0026, 0.0012), (0.0, 0.0019)], segments=6)
    sm = [trs((sx * 0.0078, -yc + yy, face + 0.004)) for sx in (-1, 1) for yy in (0.014, G - 0.012)]
    sm += [trs((0.962 * R * sin(radians(a)), 0.962 * R * cos(radians(a)), face + 0.0033), scale=1.2) for a in (45, 135, 225, 315)]
    objs.append(instances_to_mesh("screws", scr, sm, keep_base=False))
    le = hypot(tip[0], tip[1])
    objs.append(box("gnomon_style", (gt + 0.0014, le - 0.004, 0.0024), loc=(0, -yc + tip[0] / 2, face + 0.003 + tip[1] / 2),
                    rot=(degrees(atan2(tip[1], tip[0])), 0, 0), bevel=0.0006, segments=1))
    for o in objs:
        assign(o, bz)
    return objs, -yc, G


def _bird(bz, s=1.0):
    """Small cast-bronze sparrow, feet at the origin, facing +X: lofted body and head in one skin, stout conical
    beak, folded wings standing proud of the flanks with their tips crossing over the rump, wedge tail, legs and toes."""
    # stations along the spine: x, z of the centre, half width, half height
    st = ((-0.043, 0.0300, 0.004, 0.003), (-0.033, 0.0310, 0.011, 0.009), (-0.018, 0.0330, 0.018, 0.015), (-0.002, 0.0360, 0.0215, 0.019),
          (0.012, 0.0410, 0.0205, 0.0195), (0.022, 0.0490, 0.0155, 0.016), (0.029, 0.0570, 0.0135, 0.0135), (0.037, 0.0600, 0.0115, 0.0115),
          (0.044, 0.0585, 0.006, 0.0065))
    nr = 8
    V, F = [], []
    for (x, zc_, hw, hh) in st:
        for k in range(nr):
            a = 2 * pi * (k + 0.5) / nr
            sz = sin(a)
            V.append((x * s, hw * cos(a) * s, (zc_ + hh * (sz if sz > 0 else sz * 0.85)) * s))
    for i in range(len(st) - 1):
        for k in range(nr):
            k2 = (k + 1) % nr
            F.append((i * nr + k, i * nr + k2, (i + 1) * nr + k2, (i + 1) * nr + k))
    V.append(((st[0][0] - 0.003) * s, 0.0, st[0][1] * s))
    V.append(((st[-1][0] + 0.003) * s, 0.0, st[-1][1] * s))
    for k in range(nr):
        k2 = (k + 1) % nr
        F.append((len(V) - 2, k2, k))
        F.append((len(V) - 1, (len(st) - 1) * nr + k, (len(st) - 1) * nr + k2))
    body = mesh_object("bird_body", V, F)
    recalc_normals(body)
    parts = [body]
    parts.append(cone("bk", 0.0052 * s, 0.0005 * s, 0.013 * s, loc=(0.0515 * s, 0, 0.0575 * s), rot=(0, 97, 0), verts=6))
    tail = box("tl", (0.05 * s, 0.019 * s, 0.0042 * s), bevel=0.0012 * s, segments=1)

    def wedge(co):
        t = clamp(0.5 - co.x / (0.05 * s))            # 0 at the root, 1 at the tip
        return Vector((co.x, co.y * (0.55 + 0.45 * t), co.z * (1.2 - 0.5 * t)))

    transform_verts(tail, wedge)
    place(tail, loc=(-0.062 * s, 0, 0.0265 * s), rot=(0, -11, 0))
    parts.append(tail)
    for sy in (-1, 1):
        w = sphere("wg", 1.0, segments=8, rings=5)

        def wing(co, sy=sy):
            k = 1.0 if co.x > 0 else max(1.0 + co.x, 0.0) ** 0.55         # pointed primaries
            return Vector((co.x * 0.034 * s, co.y * 0.0062 * s * (0.4 + 0.6 * k), co.z * 0.0145 * s * (0.12 + 0.88 * k) - (1.0 - k) * 0.004 * s))

        transform_verts(w, wing)
        place(w, loc=(-0.012 * s, sy * 0.0175 * s, 0.0395 * s), rot=(sy * 14, -17, sy * 13))
        parts.append(w)
        fx_, fy = 0.004 * s, sy * 0.008 * s
        parts.append(sweep("lg", [(fx_ - 0.006 * s, fy, 0.024 * s), (fx_ - 0.003 * s, fy, 0.01 * s), (fx_, fy, 0.0012 * s)], 0.0016 * s, sides=5, smooth_path=False))
        for ta, tl in ((0, 0.013), (32, 0.011), (-32, 0.011), (180, 0.008)):
            a = radians(ta)
            parts.append(box("toe", (tl * s, 0.0024 * s, 0.0024 * s), loc=(fx_ + cos(a) * tl * s / 2, fy + sin(a) * tl * s / 2, 0.0012 * s),
                             rot=(0, 0, ta), bevel=0.0007 * s, segments=1))
    for o in parts:
        assign(o, bz)
        smooth(o, 50)
    return join(parts, "bird")


def _birdbath(p, rng, zc, H, R, rn, stone):
    """Bowl from z = zc (top of the capital) to the rim at z = H. Returns (stone objs, inner profile, depth)."""
    hb = H - zc
    U = ((rn + 0.012, zc + 0.02), (R * 0.64, zc + 0.016), (R * 0.975, zc + hb * 0.42), (R - 0.004, H - 0.05))
    foot = [(rn, zc), (rn, zc + 0.012), (rn + 0.012, zc + 0.02)]
    rim = [(R - 0.004, H - 0.05), (R - 0.004, H - 0.044), (R + 0.008, H - 0.04)] + _bead(R + 0.008, H - 0.037, 0.031, 0.011, 4)
    rim += [(R + 0.003, H - 0.002), (R - 0.005, H)]
    ri = R - 0.032
    rim += [(ri, H)]
    objs = []
    if p.bowl_style == "gadroon":
        ts = (0.0, 0.04, 0.11, 0.3, 0.55, 0.78, 0.9, 0.96, 1.0)
        env = (0.0, 0.55, 1.0, 1.0, 1.0, 1.0, 0.85, 0.45, 0.0)
        wid = (0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 0.92, 0.7, 0.4)
        nl = int(clamp(round(2 * pi * R * 0.85 / 0.09 / 2) * 2, 12, 20))
        objs.append(_gadroons("bowl_gadroons", [_bezpt(U, t) for t in ts], nl, 0.017, env, wid))
        objs.append(lathe("bowl_foot", _clean(foot + [(rn + 0.004, zc + 0.022)]), segments=_seg(rn)))
        objs.append(lathe("bowl_rim", _clean([(R - 0.02, H - 0.052)] + rim), segments=32))
    else:
        objs.append(lathe("bowl", _clean(foot + [_bezpt(U, k / 7.0) for k in range(1, 8)] + rim[1:]), segments=32))
    db = min(hb - 0.05, 0.09)
    inner = [(ri, H), (ri - 0.005, H - 0.004)]
    inner += _bez2((ri - 0.005, H - 0.004), (ri - 0.014, H - db * 0.78), (R * 0.45, H - db), (0.0, H - db), 8)[1:]
    return objs, inner, db


def _radius_at(inner, z):
    """Radius of the basin's inner surface at height z."""
    for (r0, z0), (r1, z1) in zip(inner[:-1], inner[1:]):
        if (z0 - z) * (z1 - z) <= 0 and abs(z1 - z0) > 1e-7:
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return inner[0][0]


# ------------------------------------------------------------------ growth: lichen rosettes and moss cushions
class _Surf:
    """World-space faces of a set of objects: BVH for conforming / occlusion and arrays for area sampling."""

    def __init__(self, objs):
        self.V, self.F = [], []
        C, N, A = [], [], []
        for o in objs:
            realize(o)
            M = world_matrix(o)
            M3 = M.to_3x3()
            b = len(self.V)
            self.V.extend(M @ v.co for v in o.data.vertices)
            for pl in o.data.polygons:
                self.F.append([b + i for i in pl.vertices])
                C.append(tuple(M @ pl.center))
                N.append(tuple((M3 @ pl.normal).normalized()))
                A.append(pl.area)
        self.C, self.N, self.A = np.array(C), np.array(N), np.array(A)
        self.bvh = BVHTree.FromPolygons([tuple(v) for v in self.V], self.F)

    def point_in(self, fi, r):
        """Random point inside face fi."""
        f = self.F[fi]
        c = Vector(self.C[fi])
        k = int(r.u(0, len(f) - 1e-6))
        a, b = self.V[f[k]], self.V[f[(k + 1) % len(f)]]
        t = r.u(0, 1)
        return c + ((a * (1 - t) + b * t) - c) * sqrt(r.u(0, 1))


def _patch(bvh, P, N, rad, hgt, sink, r, out, nseg=8, rings=2, along=None, elong=1.0):
    """One irregular, domed patch hugging the surface under it (every vertex is snapped to the stone)."""
    N = Vector(N).normalized()
    T = (along - N * along.dot(N)) if along is not None else N.cross(Vector((0, 0, 1)))
    if T.length < 0.15:
        T = N.cross(Vector((1, 0, 0)))
    T.normalize()
    B = N.cross(T)
    V, F = out
    b = len(V)
    a0 = r.u(0, 2 * pi)

    def put(x, y, h):
        Q = P + T * (x * elong) + B * y
        hit = bvh.find_nearest(Q, rad * 3 + 0.06)
        loc, nrm = (hit[0], hit[1]) if hit[0] is not None else (Q, N)
        d = (nrm + N).normalized() if nrm.dot(N) > 0.2 else N
        V.append(tuple(loc + d * h))

    put(0.0, 0.0, hgt)
    rr = [rad * r.u(0.6, 1.22) for _ in range(nseg)]
    if rings == 2:
        for k in range(nseg):
            a = a0 + 2 * pi * k / nseg
            f = 0.5 * (rr[k] + rr[k - 1]) * 0.56
            put(cos(a) * f, sin(a) * f, hgt * r.u(0.6, 0.92))
        for k in range(nseg):
            a = a0 + 2 * pi * (k + 0.5) / nseg
            put(cos(a) * rr[k], sin(a) * rr[k], -sink)
        for k in range(nseg):
            k2 = (k + 1) % nseg
            F.append((b, b + 1 + k, b + 1 + k2))
            F.append((b + 1 + k, b + 1 + nseg + k, b + 1 + k2))
            F.append((b + 1 + k2, b + 1 + nseg + k, b + 1 + nseg + k2))
    else:
        for k in range(nseg):
            a = a0 + 2 * pi * k / nseg
            put(cos(a) * rr[k], sin(a) * rr[k], -sink)
        for k in range(nseg):
            F.append((b, b + 1 + k, b + 1 + (k + 1) % nseg))


def _free(occ, P, N):
    """True when the surface point P is exposed (not buried under / inside another part)."""
    o = P + N * 0.004
    hit = occ.ray_cast(o, N, 0.05)
    if hit[0] is not None:
        return False
    hit = occ.ray_cast(o, N)
    return hit[0] is None or hit[1].dot(N) < 0.1


# ------------------------------------------------------------------ build
def build(p, rng):
    H = p.height
    m = p.moss
    age = p.age
    sd = p.seed
    stone = mat("stone_smooth", p.stone)
    stone_r = mat("stone_rough", p.stone)
    bz = mat("bronze_aged")                                        # one dark statuary bronze for dial, gnomon and bird
    n = {"square": 4, "octagon": 8, "round": 28}[p.plan]
    bw = p.base_width
    rb = bw * 0.36
    nst = int(p.steps)
    sh = 0.085
    upper, lower = [], []              # upper leans with the pedestal, lower (steps) stays put
    stones = []                        # everything carved from stone
    joints = []
    ang = p.lean * rng.u(0.6, 1.0)     # settlement out of plumb (deg) about a random horizontal axis
    ax = radians(rng.u(0, 360))

    # ---- base steps (each projects 0.11 m beyond the course above)
    r_st = rng.fork("steps")
    z = 0.0
    for i in range(nst):
        ap = bw / 2 + 0.11 * (nst - i)
        cs, js = _course(f"step{i}", n, ap, z, z + sh, 4 if i == 0 else 2, r_st, inner=ap - 0.11)
        for o in cs:
            assign(o, stone_r)
        lower += cs
        joints += js
        z += sh
    z_step = z

    # ---- plinth block
    hp = 0.13
    r_pl = rng.fork("plinth")
    if n == 4:
        pl = stone_block("plinth", (bw, bw, hp), loc=(0, 0, z + hp / 2), chamfer=0.014, rough=0.0015, rng=r_pl, detail=0.09)
    else:
        pl = _ngon("plinth", n, [(bw / 2 - 0.003, z), (bw / 2, z + 0.006), (bw / 2, z + hp * 0.45), (bw / 2, z + hp - 0.03),
                                 (bw / 2 - 0.008, z + hp - 0.018), (bw / 2 - 0.012, z + hp - 0.016), (bw / 2 - 0.012, z + hp - 0.008),
                                 (bw / 2 - 0.02, z + hp), (bw * 0.3, z + hp)], bev=0.005)
    assign(pl, stone)
    upper.append(pl)
    zb = z + hp

    # ---- top geometry heights
    sundial = p.top == "sundial"
    if sundial:
        R = p.dial / 2
        zc = H - 0.1
    else:
        R = p.bowl / 2
        zc = H - clamp(0.42 * R + 0.02, 0.11, 0.2)
    L = zc - zb
    s = min(rb / 0.16, L / 0.6)

    # ---- shaft
    fn = {"vase": _shaft_vase, "fluted": _shaft_fluted, "twist": _shaft_twist, "square": _shaft_square}[p.baluster]
    sobjs, r_top, r_corner = fn(zb, zc, rb, s, rng.fork("shaft"))
    for o in sobjs:
        assign(o, stone)
    upper += sobjs

    # ---- top
    extra = []                         # non-stone parts (own weathering)
    level = []                         # parts that stay level when the pedestal leans (water finds its own level)
    blockers = []                      # non-stone parts that cover stone (nothing grows under them)
    basin = None
    zw = None
    rw = 0.0
    gy = G = 0.0
    if sundial:
        # turned bed mould (fillet, ovolo, fillet, cavetto, fillet - one lathe, concentric with the shaft) under a plain table slab
        at = R + 0.05
        sq = n == 4 and p.baluster == "square"
        r0 = min(r_top - 0.006, at - 0.075)
        rm = at - 0.024
        w = rm - r0
        zs_ = zc + 0.046                                   # underside of the slab
        ro = r0 + 0.42 * w
        bed = [(r0 - 0.012, zc - 0.001), (r0, zc), (r0, zc + 0.006)]
        bed += _quart((r0, zc + 0.006), (ro, zc + 0.022), "a", 3)
        bed += [(ro, zc + 0.026), (ro + 0.002, zc + 0.026)]
        bed += _quart((ro + 0.002, zc + 0.026), (rm, zc + 0.042), "b", 3)
        bed += [(rm, zs_ + 0.002)]
        if sq:
            bedo = _ngon("bed_mould", 4, bed, bev=0.003)
        else:
            bedo = lathe("bed_mould", _clean(bed), segments=(n if n > 8 else _seg(rm)))
        assign(bedo, stone)
        upper.append(bedo)
        slab = [((rm - 0.008) * cos(pi / n) - 0.004, zs_), (at - 0.005, zs_), (at, zs_ + 0.005), (at, (zs_ + H) / 2), (at, H - 0.016)]
        slab += _quart((at, H - 0.016), (at - 0.012, H), "b", 3) + [(at - 0.03, H), (R * 0.6, H)]
        table = _ngon("table", n, slab, bev=0.004)
        if n <= 8 and age > 0.25:
            realize(table)
            r_c = rng.fork("chip")
            for _ in range(1 if age < 0.7 else 2):
                ac = radians(180.0 / n + 360.0 / n * int(r_c.u(0, n)))
                cz = H if r_c.chance(0.7) else zs_
                cc = Vector((at / cos(pi / n) * cos(ac), at / cos(pi / n) * sin(ac), cz))
                cr = r_c.u(0.03, 0.05)
                dn = -1.0 if cz == H else 1.0

                def chip(co, cc=cc, cr=cr, dn=dn, ac=ac):
                    d = (co - cc).length
                    if d > cr:
                        return co
                    w_ = (1.0 - d / cr) ** 1.5
                    return Vector((co.x - cos(ac) * 0.016 * w_, co.y - sin(ac) * 0.016 * w_, co.z + dn * 0.009 * w_))

                transform_verts(table, chip)
        assign(table, stone)
        upper.append(table)
        dial, gy, G = _sundial(p, rng.fork("dial"), H - 0.0005, R, bz)
        assign(dial[0], mat("mortar", "#5d7365", scale=2.0, grime=1.4))       # the sunk ground: a dull, powdery grey-green verdigris crust
        upper += dial
        extra += dial
        blockers.append(dial[0])
        z_drip = zs_
        top_r = at
    else:
        bobjs, inner, db = _birdbath(p, rng.fork("bowl"), zc, H, R, r_corner + 0.004, stone)
        r_w = rng.fork("water")
        wet = p.water > 0.04
        zw = H - 0.016 - (1.0 - p.water) * (db - 0.03) if wet else H - db + 0.001
        if wet:                                        # vertex rings around the water line and the wet tide mark above it
            for dzw in (-0.012, -0.002, 0.004, 0.012, 0.019):
                inner = _insert_z(inner, zw + dzw)
        basin = lathe("basin", _clean(inner), segments=32)
        bobjs.append(basin)
        for o in bobjs:
            assign(o, stone)
        upper += bobjs
        z_drip = H - 0.03
        top_r = R
        if wet:
            # still, dark water: a level disc (wide enough to meet the wall of the leaning bowl on its low side) with a meniscus
            rw = _radius_at(inner, zw)
            ro = _radius_at(inner, min(zw + tan(radians(ang)) * rw * 1.1, H - 0.004))
            ri = rw - 0.03
            wat = lathe("water", [(ri, zw), (rw * 0.62, zw), (rw * 0.3, zw), (0.0, zw)], segments=32)
            assign(wat, mat("metal_plain", "#aab6b2", rough=0.2))             # still water: a dark mirror of the sky ...
            edge = lathe("water_edge", [(ro + 0.004, zw + 0.0018), (ro - 0.002, zw + 0.0006), (ro - 0.012, zw), (rw - 0.012, zw), (ri, zw)], segments=32)
            assign(edge, mat("glass_clear", "#3a4a44", opacity=0.6, rough=0.2))       # ... clear at the shallow edge, where the bowl shows through
            level.append(edge)
            level.append(wat)
            blockers.append(wat)
        # fallen leaves floating on the water / lying in the dry bowl
        lf = surface("leaf", lambda u, v: ((u - 0.5) * 0.034 * (sin(pi * min(v * 1.25, 1.0)) ** 0.7) * (1.0 - 0.5 * v),
                                           (v - 0.5) * 0.06, 0.007 * (u - 0.5) ** 2 * 4 + 0.005 * (v - 0.5) ** 2 * 4),
                     3, 5, uv_len=(0.25, 0.25))
        realize(lf)
        rl = _radius_at(inner, zw) - 0.05
        lm = []
        for k in range(int(2 + age * 4)):
            a = r_w.u(0, 2 * pi)
            rr = rl * sqrt(r_w.u(0.05, 1.0))
            zz = zw - 0.0006 if wet else max(zw, H - db + (rr / max(rl, 0.01)) ** 2 * 0.01) + 0.002
            lm.append(trs((rr * cos(a), rr * sin(a), zz), (r_w.u(-9, 9), r_w.u(-9, 9), r_w.u(0, 360)), r_w.u(0.7, 1.2)))
        lv = instances_to_mesh("leaves", lf, lm, keep_base=False, uv_jitter=False)
        assign(lv, mat("leaf", r_w.pick(["#7d5a26", "#6b4424", "#77692c"]), rough=1.3))
        (level if wet else upper).append(lv)
        extra.append(lv)
        if p.bird:
            bd = _bird(bz, 1.05)
            ab = r_w.u(15, 60) * (1 if r_w.chance(0.5) else -1) - 90.0          # somewhere on the front half of the rim
            rr = R - 0.016
            place(bd, loc=(rr * cos(radians(ab)), rr * sin(radians(ab)), H - 0.0005), rot=(0, 0, ab + 180 + r_w.u(-50, 50)))
            upper.append(bd)
            extra.append(bd)

    # ---- lead-filled inscription cut into the plinth front
    itxt = str(p.inscription).strip()
    ins = None
    if itxt and n <= 8:
        fw = bw if n == 4 else bw * tan(pi / 8)
        to, tw, th = _ctext("inscription", itxt, 0.046 if len(itxt) <= 6 else 0.04, 0.0036, font="serif", lim=34.0, walls=False)
        avail = fw - (0.05 if n == 4 else 0.026)
        kx = clamp(avail / max(tw, 1e-3), 0.62, 1.0)                # condense first, only then shrink
        k = min(1.0, avail / max(tw * kx, 1e-3))
        zpq = z_step + hp * 0.5 - (0.008 if n != 4 else 0.0)
        place(to, loc=(0, -bw / 2 + 0.0019, zpq), rot=(90, 0, 0), scale=(k * kx, k, 1.0))
        assign(to, mat("lead"))
        upper.append(to)
        extra.append(to)
        ins = [Vector((0, -bw / 2, zpq)), tw * k * kx / 2 + 0.028, th * k / 2 + 0.022]

    # ---- settle: everything above the steps leans a touch
    if ang > 0.02:
        M = Matrix.Translation((0, 0, z_step)) @ Matrix.Rotation(radians(ang), 4, Vector((cos(ax), sin(ax), 0.0))) @ Matrix.Translation((0, 0, -z_step))
        dz = sin(radians(ang)) * bw * 0.5 * (1.0 if nst == 0 else 0.35)       # keep the low edge from sinking far into the step / ground
        M = Matrix.Translation((0, 0, dz)) @ M
    else:
        M = Matrix.Identity(4)
    for o in upper:
        realize(o)
        apply_transform(o)
        transform_verts(o, lambda co: M @ co)
    wc = Vector((0, 0, 0))
    if level:
        c0 = Vector((0, 0, zw))
        dv = (M @ c0) - c0
        for o in level:
            realize(o)
            apply_transform(o)
            transform_verts(o, lambda co: co + dv)
        zw = zw + dv.z
        wc = c0 + dv
    top_c = M @ Vector((0, 0, H))
    if ins is not None:
        ins[0] = M @ ins[0]

    stones = [o for o in upper + lower if o not in extra]
    for o in stones:                                   # limestone bedding / turning rings run horizontally
        if o.active_material is not None and o.active_material.name.startswith(stone.name) and o not in lower:
            realize(o)
            uv_transform(o, scale=(UST, 1.0), rot_deg=UROT)
            uv_keep(o)

    def in_panel(P, k=1.0):
        """Inside the cleaned field around the inscription?"""
        if ins is None:
            return False
        c, hw_, hh_ = ins
        return abs(P.x - c.x) < hw_ * k and abs(P.z - c.z) < hh_ * k and abs(P.y - c.y) < 0.03

    # ---- lichen rosettes and moss cushions grown onto the finished stone
    r_g = rng.fork("growth")
    clumps = []                                        # (x, y, z, radius) of the moss, for the algae halo
    if m > 0.02:
        surf = _Surf(stones)
        occ = _Surf(stones + blockers).bvh if blockers else surf.bvh
        rs = np.random.RandomState(sd * 7 + 13)
        Cn, Nn = surf.C, surf.N
        up = np.clip(Nn[:, 2], 0, 1)
        north = np.clip(Nn[:, 1], 0, 1)
        side = 1.0 - np.abs(Nn[:, 2])
        okf = (Nn[:, 2] > -0.25) & (Cn[:, 2] > 0.012)
        flat_ = Nn[:, 2] > 0.6
        colony = noise_mask(0.16, seed=sd + 31, lo=0.3, hi=0.7)(Cn, Nn)

        # lichen: thin crusty pale grey and ochre-yellow rosettes on upward and north faces
        wl = surf.A * okf * (0.06 + 1.0 * up + 0.85 * north * side + 0.12 * side) * (0.15 + colony)
        n_l = int(6 + 46 * m)
        lich = (([], []), ([], []))
        if wl.sum() > 0:
            cand = rs.choice(len(wl), size=n_l * 3, p=wl / wl.sum())
            done = 0
            for fi in cand:
                if done >= n_l:
                    break
                N0 = Vector(surf.N[fi])
                P0 = surf.point_in(int(fi), r_g)
                if in_panel(P0, 1.15) or not _free(occ, P0, N0):
                    continue
                grp = 1 if r_g.chance(0.42) else 0
                rad = r_g.u(0.012, 0.03) * (0.85 + 0.3 * m)
                _patch(surf.bvh, P0, N0, rad, 0.001, 0.0006, r_g, lich[grp], nseg=8, rings=1)
                for _ in range(int(r_g.u(0, 3.2))):                          # rosettes merge into blotches
                    off = Vector(r_g.vec(1.0))
                    off = (off - N0 * off.dot(N0))
                    if off.length < 1e-3:
                        continue
                    Q = P0 + off.normalized() * rad * r_g.u(0.9, 1.5)
                    hit = surf.bvh.find_nearest(Q, 0.08)
                    if hit[0] is None or not _free(occ, hit[0], hit[1]):
                        continue
                    _patch(surf.bvh, hit[0], hit[1], rad * r_g.u(0.45, 0.8), 0.0009, 0.0006, r_g, lich[grp], nseg=6, rings=1)
                done += 1
        for grp, tint in ((0, "#a9a995"), (1, "#b3923a")):
            if lich[grp][0]:
                lo_ = mesh_object("lichen%d" % grp, lich[grp][0], lich[grp][1])
                assign(lo_, mat("mortar", tint, wear=0.0, grime=0.2))
                smooth(lo_, 70)
                extra.append(lo_)
                lower.append(lo_)

        # moss: domed cushions tucked into joints and inside corners, spreading over the ledges (never the faces) as `moss` grows
        mv, mf = [], []
        n_m = int(round(4 + 95 * m ** 1.6))
        zrel = np.clip(Cn[:, 2] / max(H, 0.1), 0, 1)
        northp = 0.55 + 0.45 * np.clip(Cn[:, 1] / max(bw * 0.4, 0.05), -1, 1)
        wm = surf.A * okf * flat_ * up ** 3 * (1.0 - 0.6 * zrel * (1.0 - m)) * northp * (0.25 + colony)
        if sundial:                                    # the table top stays clean around the dial (lichen only)
            wm = wm * (Cn[:, 2] < top_c.z - 0.04)
        ledge_p = 0.1 + 0.9 * m ** 1.5                 # chance to accept a spot that is not in a joint / corner
        hd = [Vector((cos(k * pi / 4), sin(k * pi / 4), 0.0)) for k in range(8)]

        def cushion(P0, N0, rad, along=None, elong=1.0):
            if in_panel(P0, 1.25):
                return False
            for (cx, cy, cz, cr) in clumps:
                if (P0.x - cx) ** 2 + (P0.y - cy) ** 2 + (P0.z - cz) ** 2 < (0.75 * (rad + cr)) ** 2:
                    return False
            _patch(surf.bvh, P0, N0, rad, rad * r_g.u(0.38, 0.6), 0.003, r_g, (mv, mf), nseg=6, rings=2, along=along, elong=elong)
            clumps.append((P0.x, P0.y, P0.z, rad * max(elong, 1.0)))
            return True

        def seated(P0, rad):
            """Is the ledge under the cushion wide enough to carry it (not an arris, a bevel or a small ornament)?"""
            k = 0
            for d in hd:
                hit = occ.ray_cast(P0 + d * (rad * 0.8) + Vector((0, 0, 0.012)), Vector((0, 0, -1)), 0.03)
                if hit[0] is not None and hit[1].z > 0.4:
                    k += 1
            return k >= 4

        # open joints between the blocks of the steps
        for (q0, q1) in joints:
            q0, q1 = Vector(q0), Vector(q1)
            ln = (q1 - q0).length
            t = r_g.u(0.0, 0.25)
            while t < 1.0 and len(clumps) < n_m:
                if r_g.chance(0.35 + 0.6 * m):
                    rad = r_g.u(0.012, 0.022) * (0.8 + 0.5 * m)
                    cushion(q0.lerp(q1, t) + Vector((0, 0, -0.002)), Vector((0, 0, 1)), rad, along=q1 - q0, elong=r_g.u(1.3, 2.2))
                t += r_g.u(0.04, 0.16) / max(ln, 0.05)
        if wm.sum() > 0:
            cand = rs.choice(len(wm), size=n_m * 14, p=wm / wm.sum())
            for fi in cand:
                if len(clumps) >= n_m:
                    break
                N0 = Vector(surf.N[fi])
                P0 = surf.point_in(int(fi), r_g)
                if not _free(occ, P0, N0):
                    continue
                o = P0 + N0 * 0.009
                wall = None
                for d in hd:                               # is there a wall right next to it (joint / inside corner)?
                    hit = occ.ray_cast(o, d, 0.045)
                    if hit[0] is not None and (wall is None or hit[3] < wall[1]):
                        wall = (d, hit[3])
                if wall is None and not r_g.chance(ledge_p):
                    continue
                rad = r_g.u(0.013, 0.03) * (0.75 + 0.55 * m) * (0.8 if P0.z > 0.6 * H else 1.0)
                if wall is not None:                       # push it into the corner and stretch it along the joint
                    P0 = P0 + wall[0] * max(wall[1] - rad * 0.45, 0.0)
                if not seated(P0, rad):
                    continue
                if wall is not None:
                    al = Vector((-wall[0].y, wall[0].x, 0.0))
                    ok = cushion(P0, N0, rad, along=al, elong=r_g.u(1.2, 1.9))
                else:
                    ok = cushion(P0, N0, rad)
                if ok:
                    for _ in range(int(r_g.u(0, 2.2 + 2.0 * m))):       # satellites
                        off = Vector(r_g.vec(1.0))
                        off = off - N0 * off.dot(N0)
                        if off.length < 1e-3:
                            continue
                        r2 = rad * r_g.u(0.45, 0.8)
                        hit = surf.bvh.find_nearest(P0 + off.normalized() * (rad + r2) * r_g.u(0.75, 1.0), 0.08)
                        if hit[0] is not None and hit[1].z > 0.6 and _free(occ, hit[0], hit[1]) and seated(hit[0], r2):
                            cushion(hit[0], hit[1], r2)
        if mv:
            mo = mesh_object("moss_cushions", mv, mf)
            assign(mo, mat("mortar", "#6a7a30", wear=0.0))         # (the library moss goes black in the shade)
            smooth(mo, 80)
            nzm = noise_mask(0.05, seed=sd + 41, lo=0.2, hi=0.8)
            paint(mo, grime=lambda co, nr: 0.04 + 0.14 * nzm(co, nr))         # dull olive, darker in patches
            extra.append(mo)
            lower.append(mo)

    # ---- weathering
    CL = np.array(clumps) if clumps else None
    nz_big = noise_mask(0.2, seed=sd + 11, lo=0.35, hi=0.75)
    nz_fine = noise_mask(0.04, seed=sd + 12, lo=0.3, hi=0.7)
    nz_dirt = noise_mask(0.1, seed=sd + 13, lo=0.25, hi=0.8)
    nz_stain = noise_mask(0.28, seed=sd + 14, lo=0.38, hi=0.72)
    nz_edge = noise_mask(0.07, seed=sd + 15, lo=0.2, hi=0.8)
    grow = max(m, 0.35 * age)
    ph1, ph2 = rng.u(0, 6.28), rng.u(0, 6.28)

    def streaks(co):
        """Soft vertical run-off streaks: a few irregular stripes around the axis (wide enough for the turned parts' columns)."""
        a = np.arctan2(co[:, 1] - top_c.y, co[:, 0] - top_c.x)
        st = 0.5 + 0.5 * np.sin(a * 5.0 + 1.9 * np.sin(a * 2.0 + ph1) + ph2)
        return np.clip((st - 0.42) * 1.9, 0, 1) ** 1.4

    def clear(co):
        """Field around the inscription that has been kept scrubbed (so the lead reads on old, dark stone)."""
        if ins is None:
            return 1.0
        c, hw_, hh_ = ins
        d = np.maximum.reduce([np.abs(co[:, 0] - c.x) / hw_, np.abs(co[:, 2] - c.z) / hh_, np.abs(co[:, 1] - c.y) / 0.03])
        return 1.0 - 0.8 * np.clip((1.3 - d) * 3.5, 0, 1)

    def walls(nr):
        return np.clip(1.0 - np.abs(nr[:, 2]) * 1.3, 0, 1)        # run-off shows on walls: not on ledges, not on undersides

    def algae_mask(co, nr):
        """Grey-green algae: a soft band under the drip line and run-off stains down the shaft, a little at the damp foot and around the moss."""
        below = z_drip - co[:, 2]
        under = np.clip(below / 0.015, 0, 1)
        a1 = under * np.clip(1.0 - below / 0.06, 0, 1) * (0.3 + 0.3 * nz_edge(co, nr))
        a2 = under * streaks(co) * np.clip(1.0 - below / (0.22 + 0.4 * grow), 0, 1) * (0.2 + 0.5 * nz_fine(co, nr)) * (0.35 + 0.65 * nz_big(co, nr))
        v = np.maximum(a1, a2) * walls(nr) * (0.55 + 0.6 * grow)
        a5 = np.clip(1.0 - co[:, 2] / 0.12, 0, 1) * nz_big(co, nr) * 0.3 * (0.4 + 0.6 * grow) * (nr[:, 2] > -0.2)
        v = np.maximum(v, a5)
        if CL is not None:
            d = np.sqrt(((co[:, None, :] - CL[None, :, :3]) ** 2).sum(axis=2)) - CL[None, :, 3]
            v = np.maximum(v, np.clip(1.0 - d / 0.018, 0, 1).max(axis=1) * 0.26)
        return np.clip(v, 0, 0.45) * clear(co)

    def grime_mask(co, nr):
        zz = co[:, 2]
        damp = np.clip(1.0 - (zz - 0.05 * nz_edge(co, nr)) / 0.13, 0, 1) * (0.25 + 0.17 * age)              # damp band at the ground
        foot = np.clip(1.0 - (zz - z_step) / 0.2, 0, 1) * (0.2 + 0.3 * age) * (zz > z_step - 0.01)
        below = z_drip - zz
        dr = streaks(co) * np.clip(below / 0.015, 0, 1) * np.clip(1.0 - below / 0.7, 0, 1) * walls(nr) * (0.35 + 0.35 * age)
        ledge = np.clip(nr[:, 2], 0, 1) * np.clip((top_c.z - 0.03 - zz) / 0.02, 0, 1) * (0.2 + 0.3 * age)
        patch = nz_stain(co, nr) * (0.15 + 0.3 * age)
        g = np.maximum.reduce([foot, dr, ledge, patch]) * (0.5 + 0.5 * nz_dirt(co, nr))
        return np.clip(np.maximum(g, damp), 0, 0.45) * clear(co)            # capped: old stone still reads as stone

    def wear_mask(co, nr):
        rr = np.hypot(co[:, 0] - top_c.x, co[:, 1] - top_c.y)
        w_ = np.clip(1.0 - (top_c.z - co[:, 2]) / 0.03, 0, 1) * np.clip(rr / max(top_r, 0.01) - 0.75, 0, 1) * 4.0
        return np.clip(w_ * (0.25 + 0.5 * age), 0, 1)

    for o in stones:
        paint(o, grime=grime_mask, wear=wear_mask, special=algae_mask)
        lob = _LOBES.get(o.name.split(".")[0])
        if lob is not None:                            # dirt and a little algae settle in the creases between the gadroons

            def crease(co, nr, lob=lob):
                pitch = 2 * pi / lob[0]
                a = np.arctan2(co[:, 1] - top_c.y, co[:, 0] - top_c.x) - lob[1] + 0.5 * pitch
                d = np.abs((a + 0.5 * pitch) % pitch - 0.5 * pitch) / pitch
                return np.clip(1.0 - d / 0.07, 0, 1)

            paint(o, grime=lambda co, nr: crease(co, nr) * (0.3 + 0.25 * age) * (0.6 + 0.4 * nz_dirt(co, nr)),
                  special=lambda co, nr: crease(co, nr) * 0.3 * grow * nz_fine(co, nr))
    if basin is not None and p.water > 0.04:
        def wet_stone(co, nr):
            """Darker wet tide line above the water; under it the floor shows pale at the shallow edge and goes dark in the middle."""
            dzw = co[:, 2] - zw
            band = np.clip((0.017 - dzw) / 0.005, 0, 1) * np.clip((dzw + 0.003) / 0.003, 0, 1)
            rr = np.hypot(co[:, 0] - wc.x, co[:, 1] - wc.y) / max(rw, 0.01)
            deep = np.clip(-dzw / 0.004, 0, 1) * (0.3 + 0.6 * np.clip(1.1 - rr, 0, 1) ** 0.5)
            return np.maximum.reduce([band * 0.78, deep, 0.1 + 0.2 * nz_dirt(co, nr)])

        def slime(co, nr):
            dzw = co[:, 2] - zw
            return np.clip((0.012 - dzw) / 0.01, 0, 1) * (0.12 + 0.2 * nz_fine(co, nr))

        paint(basin, grime=wet_stone, special=slime, wear=0.0, mode="set")
    vg_n = noise_mask(0.035, seed=sd + 21, lo=0.25, hi=0.75)
    vg_b = noise_mask(0.09, seed=sd + 22, lo=0.3, hi=0.7)

    def dxy(co):
        return co[:, 0] - top_c.x, co[:, 1] - top_c.y

    def d_foot(co):
        """Distance from the gnomon's foot flange (plan)."""
        x, y = dxy(co)
        return np.hypot(x, y - np.clip(y, gy, gy + G))

    def patina(co, nr):
        """Dirt on the verdigris ground: mottled brown, darkest around the gnomon foot, washed out in streaks towards the edge."""
        x, y = dxy(co)
        rr = np.hypot(x, y) / max(R, 0.01)
        streak = 0.5 + 0.5 * np.sin(np.arctan2(y, x) * 7.0 + 5.0 * vg_b(co, nr))
        v = 0.2 + 0.2 * age + 1.0 * vg_n(co, nr) * (0.25 + 0.75 * vg_b(co, nr)) - 0.3 * rr ** 2 * streak
        return np.clip(np.maximum(v, 0.85 * np.clip(1.0 - d_foot(co) / 0.03, 0, 1)), 0.05, 0.85)

    def bloom(co, nr):
        x, y = dxy(co)
        rr = np.hypot(x, y) / max(R, 0.01)
        return np.clip(0.4 * rr ** 2 * (0.5 + 0.5 * np.sin(np.arctan2(y, x) * 7.0 + 5.0 * vg_b(co, nr))), 0, 0.4)

    def rubbed(lo, hi):
        return lambda co, nr: np.clip(lo + (hi - lo) * vg_n(co, nr), 0, 1)

    for o in extra:
        nm_ = o.name.split(".")[0]
        if nm_ == "inscription":
            paint(o, grime=0.85)
        elif nm_ == "dial_plate":
            paint(o, grime=patina, wear=bloom, special=0.0, mode="set")
        elif nm_ == "gnomon":
            paint(o, special=rubbed(0.1, 0.5), grime=0.2, wear=0.1, mode="set")
        elif nm_ == "ring":
            paint(o, wear=rubbed(0.15, 0.5), special=rubbed(0.0, 0.3), grime=0.15, mode="set")
        elif nm_ == "bird":
            paint(o, wear=lambda co, nr: np.clip(0.02 + 0.07 * np.clip(nr[:, 2], 0, 1) * vg_n(co, nr), 0, 0.1),
                  special=lambda co, nr: np.clip((0.25 + 0.45 * vg_n(co, nr)) * (1.0 - 0.6 * np.clip(nr[:, 2], 0, 1)), 0, 0.6), grime=0.3, mode="set")
        elif nm_ in ("hour_marks", "motto", "star", "screws", "gnomon_foot", "gnomon_style"):
            paint(o, wear=rubbed(0.75, 1.0), special=0.0, grime=0.0, mode="set")      # raised work rubbed back to brown-gold

    socket("top", "attach", (top_c.x, top_c.y, top_c.z), role="prop_top")
