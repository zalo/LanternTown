"""Covered market passage: a house bridging an alley. The ground storey is dressed stone with a walkable
arched carriage passage along Y (voussoir portals with keystone, hood mould, imposts, jamb stones and guard
stones at both ends). Inside, the barrel roof is real structure: a timber wagon roof (moulded arch-braced oak
ribs on stone springers, butt purlins, ridge piece with turned pendants, wall plates, boards laid lengthwise)
or a stone net vault (moulded transverse, diagonal and ridge ribs cut into separate stones, carved bosses, webs
laid in visible courses of brick or limewashed stone), sooted above every lantern - or a flat beamed ceiling.
The passage walls are divided into bays between the ribs: glazed shop windows (lit or dark), shuttered booths,
stall hatches, stone seat niches, plank doorways and plain bays with tie rings and hook rails, over a stone
plinth course. Every stall is the lit hatch of a shop with its own hanging oil lamp, a counter (cloth over it by
trade), shelves, a hook rail and a striped canvas awning on forged arms, stocked by the rule of its trade - baker
(scored loaves, rolls in a basket, ring loaves on the rod), greengrocer (crates, egg basket, pumpkins, gourds,
strings of onions and garlic, herbs), cheesemonger (stacked wheels, a cut wheel, wedges on a board, sausages and
hams), potter (jugs, jars, bowls, plates, mugs on hooks), draper (bolts, folded lengths, skeins, yarn) - all
smooth lathed / swept goods modelled once and stamped, the trades taking turns along both walls. Lanterns hang
on chains down the centre line and pool the floor, low bracket lanterns stand just inside both mouths, a painted
market sign and a framed gilt hall sign hang on chains across the walkway (readable from both ends), a forged
bracket carries a little sign at each mouth, banners hang between. The passage is paved by the asset (setts or
cobbles with a centre gutter, damp enough to mirror the lanterns); goods stay on the counters and against the
walls, the clear width is never narrowed. ground = rubble is coursed rubble laid stone by stone on the passage
walls and both arch fronts. Above: one to three storeys (half-timbered, plastered, brick or rubble stone) with an
oriel over each arch, casement windows, a tiled roof with dormers and a chimney. Composed from the catalogue
windows / oriel / roof / dormer / chimney / door / lantern head; all masonry, carpentry, goods and the ironwork
are modelled here.
"""
from tk import *
import bpy
import bmesh
import numpy as np
from math import sin, cos, pi, atan2, radians, degrees, hypot, ceil
from mathutils import Matrix, Vector

META = {"name": "Covered alley passage", "class": "building", "mount": "ground", "focus": [0.2, -3.2, 2.3]}
GLOW = 1.6            # emissive strength of the lantern glass / gilt lettering: bright, just under heavy bloom
PARAMS = {
    "width": {"default": 8.0, "min": 6.0, "max": 10.0, "desc": "building width along X (m)"},
    "depth": {"default": 8.0, "min": 6.0, "max": 11.0, "desc": "building depth along Y = length of the passage (m)"},
    "passage_width": {"default": 2.8, "min": 2.4, "max": 3.5, "desc": "clear walkable width of the passage (m)"},
    "storeys": {"default": 3, "min": 2, "max": 4, "desc": "storeys including the stone passage storey"},
    "lit_ratio": {"default": 0.65, "min": 0.0, "max": 1.0, "desc": "share of windows with a lit room behind"},
    "ground_height": {"default": 4.3, "min": 4.1, "max": 5.0, "desc": "height of the stone passage storey (m)"},
    "storey_height": {"default": 2.9, "min": 2.7, "max": 3.3, "desc": "height of each upper storey (m)"},
    "arch": {"default": "round", "choices": ["round", "segmental", "pointed"], "desc": "shape of the passage arch"},
    "ceiling": {"default": "vault", "choices": ["vault", "beams"], "desc": "barrel roof following the arch (see vault) / flat beamed timber ceiling"},
    "vault": {"default": "timber", "choices": ["timber", "stone_ribbed"],
              "desc": "structure of the barrel roof (ceiling = vault): timber wagon roof (moulded arch-braced oak ribs on stone springers, butt purlins, ridge piece with carved pendant bosses, wall plates, boards laid lengthwise) / stone net vault (moulded transverse ribs of separate stones, diagonal and ridge ribs with carved bosses, webs of coursed limewashed stones or brick - see soffit)"},
    "ground": {"default": "ashlar", "choices": ["ashlar", "rubble"], "desc": "masonry of the passage storey"},
    "walls": {"default": "timber", "choices": ["timber", "plaster", "brick", "stone"], "desc": "construction of the upper storeys"},
    "framing": {"default": "braced", "choices": ["braced", "close_studding", "square_panels"], "desc": "half-timber pattern (walls = timber)"},
    "wall_color": {"default": "#dccfae", "desc": "limewash colour of plaster"},
    "timber_color": {"default": "#5b4332", "desc": "stain of the oak carpentry"},
    "stone_color": {"default": "#b9ab92", "desc": "tint of the dressed stone"},
    "paint": {"default": "#3d5a4b", "desc": "paint of shutters, oriel and dormer joinery"},
    "shutters": {"default": True, "desc": "plank shutters on the flanking windows (where they fit)"},
    "glazing": {"default": "square", "choices": ["square", "diamond", "bars"], "desc": "window glazing pattern"},
    "oriel": {"default": "canted", "choices": ["canted", "bowed", "box"], "desc": "plan of the oriel over the arch"},
    "roof": {"default": "roof_tile_clay", "choices": ["roof_tile_clay", "roof_slate", "wood_shingle"], "desc": "roof covering"},
    "roof_type": {"default": "gable", "choices": ["gable", "half_hip", "hip"], "desc": "roof form"},
    "ridge": {"default": "x", "choices": ["x", "y"], "desc": "ridge direction (x = eaves over the arches, y = gable over the arches)"},
    "pitch": {"default": 48.0, "min": 38.0, "max": 58.0, "desc": "roof pitch (deg)"},
    "dormers": {"default": 2, "min": 0, "max": 3, "desc": "dormers on the main slope (one fewer on the other)"},
    "chimneys": {"default": 1, "min": 0, "max": 2, "desc": "chimney stacks"},
    "door": {"default": True, "desc": "plank door in the front pier (if the pier is wide enough)"},
    "door_number": {"default": "4", "desc": "house number on the door"},
    "date": {"default": "1687", "desc": "date carved on the keystones ('' = none)"},
    "lantern_cd": {"default": 15.0, "min": 2.0, "max": 60.0,
                   "desc": "intensity of the hanging passage lantern (cd); the bracket lanterns, stall lamp and facade lanterns follow it (0.45 / 0.45 / 0.6 x)"},
    "lanterns": {"default": 3, "min": 0, "max": 5, "desc": "lanterns hanging under the passage ceiling down the centre line (spread evenly over the bays; they pool the floor)"},
    "max_lights": {"default": 14, "min": 4, "max": 14, "desc": "upper limit of the light sockets the asset emits (hanging lanterns, stall lamps, facade lanterns, oriel, bracket lanterns - in that order of priority); lower it for weak devices"},
    "lantern_drop": {"default": 0.2, "min": 0.05, "max": 0.9,
                     "desc": "chain length of the hanging lanterns (m); shortened automatically so that no lantern hangs into the view of the far opening from eye height at either mouth"},
    "lantern_size": {"default": 0.22, "min": 0.22, "max": 0.4, "desc": "width of the hanging lantern heads (m)"},
    "wall_lamps": {"default": 2, "min": 0, "max": 2, "desc": "low bracket lanterns on the passage walls just inside the mouths (they light the ribs, vault and bays)"},
    "bays": {"default": "mixed", "choices": ["mixed", "shops", "niches", "plain"],
             "desc": "what the wall bays between the ribs hold: seed-driven mix of shop windows, shuttered booths, stall niches, seat niches, doorways and plain bays / shop fronts and doors / niches and benches / bare walls with tie rings"},
    "bay_lit": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "share of the glazed shop windows in the passage with a lit room behind"},
    "soffit": {"default": "ochre_brick", "choices": ["ochre_brick", "limewash", "red_brick"], "desc": "finish of the vault webs (ceiling = vault, vault = stone_ribbed)"},
    "passage_sign": {"default": "MARKET", "desc": "lettering of the little sign boards hung from the forged brackets inside both mouths ('' = empty brackets, hang your own on the sign_passage sockets)"},
    "floor": {"default": "setts", "choices": ["setts", "cobble", "none"],
              "desc": "paving of the passage, laid by the asset 2 cm above the ground with a centre gutter and wet patches that mirror the lanterns: basalt setts / granite cobbles / none (the terrain shows through)"},
    "awnings": {"default": True, "desc": "striped canvas awnings on forged arms over the stalls and lit shop windows of the inner bays (steep pentice blinds: they stay within 0.15 m of the clear passage width)"},
    "stalls_lit": {"default": True, "desc": "the stall niches are the open hatches of lit shops: warm room behind the counter, stocked shelves, a hanging lamp in every stall"},
    "stock": {"default": 1.0, "min": 0.0, "max": 1.0, "desc": "how full the stall counters and shelves are (loaves, cheeses, eggs, produce crates, onion strings, cloth bolts, crockery)"},
    "trades": {"default": "mixed", "choices": ["mixed", "baker", "grocer", "cheese", "potter", "draper"],
               "desc": "what the stalls sell: mixed = every stall another trade in turn (baker, greengrocer, cheesemonger, potter, draper; the seed picks the first) / one trade for all"},
    "hall_sign": {"default": "MARKET HALL", "desc": "lettering of the second hanging sign board (oak frame, gilt letters and crown, both faces) hung on chains across the walkway deep inside the passage ('' = none)"},
    "market_sign": {"default": "TOWN MARKET", "desc": "lettering of the big lit sign board with crown hung under the ceiling inside the front mouth, one word per line ('' = none)"},
    "banners": {"default": 3, "min": 0, "max": 4, "desc": "crown banners hung from forged rods under the ceiling between the ribs (all above 2.8 m)"},
    "banner_color": {"default": "#22407e", "desc": "cloth colour of the banners and the enamel of the market sign"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: grime, damp, moss, sagging roof"},
}
VARIANTS = {
    "brick_beamed": {"walls": "brick", "ceiling": "beams", "arch": "segmental", "storeys": 4, "width": 7.0, "depth": 9.0,
                     "passage_width": 3.2, "roof": "roof_slate", "roof_type": "half_hip", "paint": "#6a2a22", "glazing": "bars",
                     "oriel": "canted", "dormers": 0, "chimneys": 2, "lit_ratio": 0.5, "date": "1792", "door_number": "11",
                     "bays": "shops", "bay_lit": 0.7, "lanterns": 2, "passage_sign": "ARCADE", "seed": 4},
    "plaster_low": {"walls": "plaster", "arch": "segmental", "storeys": 2, "width": 9.6, "depth": 6.5, "passage_width": 2.6,
                    "wall_color": "#d9b77c", "paint": "#2f4a6b", "roof_type": "half_hip", "dormers": 3, "oriel": "bowed", "vault": "stone_ribbed",
                    "glazing": "diamond", "lit_ratio": 0.8, "pitch": 52.0, "date": "1711", "soffit": "limewash", "bays": "niches",
                    "passage_sign": "", "seed": 9},
    "stone_gatehouse": {"walls": "stone", "ground": "rubble", "arch": "pointed", "ridge": "y", "width": 6.4, "depth": 10.5,
                        "passage_width": 3.3, "ground_height": 4.7, "roof": "roof_slate", "oriel": "box", "shutters": False,
                        "dormers": 2, "lit_ratio": 0.4, "paint": "#3a2c22", "age": 0.85, "date": "1540", "door": False,
                        "soffit": "limewash", "bay_lit": 0.7, "passage_sign": "QUAY", "seed": 15},
    "close_studded": {"framing": "close_studding", "ceiling": "beams", "width": 9.0, "depth": 7.0, "passage_width": 3.0,
                      "wall_color": "#e6dcc6", "timber_color": "#3d2c20", "paint": "#7c2f24", "roof": "wood_shingle",
                      "pitch": 54.0, "dormers": 1, "storeys": 3, "lit_ratio": 0.9, "date": "1602", "seed": 23},
    "market_lane": {"depth": 11.0, "width": 7.6, "passage_width": 3.4, "arch": "segmental", "soffit": "red_brick", "bays": "shops", "vault": "stone_ribbed",
                    "bay_lit": 0.8, "lanterns": 3, "lantern_size": 0.26, "passage_sign": "BAZAAR", "walls": "plaster",
                    "wall_color": "#c9b48e", "paint": "#7a5a22", "roof_type": "hip", "dormers": 1, "storeys": 3, "lit_ratio": 0.6,
                    "date": "1820", "door_number": "7", "seed": 31},
}


def _tris(o):
    return sum(len(f.vertices) - 2 for f in o.data.polygons)


class _Face:
    """One outer wall plane: u runs left -> right as seen from outside, the normal points outwards."""

    def __init__(self, cx, cy, th, L):
        self.cx, self.cy, self.th, self.L = cx, cy, th, L
        self.ux, self.uy = int(round(cos(radians(th)))), int(round(sin(radians(th))))
        self.nx, self.ny = self.uy, -self.ux

    def pt(self, u, z, out=0.0):
        return (self.cx + u * self.ux + out * self.nx, self.cy + u * self.uy + out * self.ny, z)


def _sheet(name, P):
    """Quad sheet from a point grid P[i][j]; the normal is d/di x d/dj."""
    nu, nv = len(P) - 1, len(P[0]) - 1
    return surface(name, lambda u, v: P[int(round(u * nu))][int(round(v * nv))], nu, nv)


def _front_only(o, n, lim=-0.5):
    """Letters lying on a board: keep only the faces not looking back into the board (n = outward direction, world);
    lim = 0.5 keeps the front faces alone (thin gilt lettering: the 4 mm edges are not worth two thirds of the triangles)."""
    bpy.context.view_layer.update()
    R3 = o.matrix_world.to_3x3()
    nv = Vector(n)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    dead = [fc for fc in bm.faces if (R3 @ fc.normal).dot(nv) < lim]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bm.to_mesh(o.data)
    bm.free()
    return o


def _arch_poly(kind, half, zs, rise, n=24):
    """Arch intrados from the left springing to the right one: points (x, z) and outward normals."""
    pts, nr = [], []
    if kind == "pointed":
        c = (half * half - rise * rise) / (2 * half)
        R = half - c
        at = atan2(rise, -c)
        h = n // 2
        for i in range(h + 1):
            a = pi - at * i / h
            pts.append((-c + R * cos(a), zs + R * sin(a)))
            nr.append((cos(a), sin(a)))
        nr[-1] = (0.0, 1.0)
        for i in range(1, h + 1):
            a = at * (1 - i / h)
            pts.append((c + R * cos(a), zs + R * sin(a)))
            nr.append((cos(a), sin(a)))
    else:
        R = (half * half + rise * rise) / (2 * rise)
        cz = zs + rise - R
        aL, aR = atan2(R - rise, -half), atan2(R - rise, half)
        for i in range(n + 1):
            a = aL + (aR - aL) * i / n
            pts.append((R * cos(a), cz + R * sin(a)))
            nr.append((cos(a), sin(a)))
    return pts, nr


def _arc_param(pts, nrm):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    tot = cum[-1]

    def at(s):
        s = min(max(s, 0.0), tot)
        i = 1
        while i < len(cum) - 1 and cum[i] < s:
            i += 1
        t = (s - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
        x = lerp(pts[i - 1][0], pts[i][0], t)
        z = lerp(pts[i - 1][1], pts[i][1], t)
        nx = lerp(nrm[i - 1][0], nrm[i][0], t)
        nz = lerp(nrm[i - 1][1], nrm[i][1], t)
        l = hypot(nx, nz) or 1.0
        return x, z, nx / l, nz / l

    return at, tot


def _add_stone(verts, faces, at, sa, sb, rin, rout, ya, yb, m=3):
    """One voussoir: a wedge between arc lengths sa..sb, radii rin..rout (along the normal), y ya..yb."""
    base = len(verts)
    k = m + 1
    for y in (ya, yb):
        for r in (rin, rout):
            for i in range(k):
                x, z, nx, nz = at(sa + (sb - sa) * i / m)
                verts.append((x + nx * r, y, z + nz * r))
    fi = lambda i: base + i
    fo = lambda i: base + k + i
    bi = lambda i: base + 2 * k + i
    bo = lambda i: base + 3 * k + i
    for i in range(m):
        faces.append((fi(i), fi(i + 1), bi(i + 1), bi(i)))
        faces.append((fo(i), bo(i), bo(i + 1), fo(i + 1)))
        faces.append((fi(i), fo(i), fo(i + 1), fi(i + 1)))
        faces.append((bi(i), bi(i + 1), bo(i + 1), bo(i)))
    faces.append((fi(0), bi(0), bo(0), fo(0)))
    faces.append((fi(m), fo(m), bo(m), bi(m)))


def _prism(name, prof, y0, y1, bev=0.008):
    """Convex profile [(x, z)] extruded along Y."""
    n = len(prof)
    verts = [(x, y0, z) for x, z in prof] + [(x, y1, z) for x, z in prof]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    if bev:
        bevel(o, bev, segments=1, angle=30)
    return o


def _cbox(name, size, loc, rot=None, c=0.008):
    """Box (x, y, z) whose -Y face is the front: chamfered front edges, no back face (it sits against a wall).
    18 triangles instead of the 44 of a fully bevelled box."""
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    c = min(c, sx * 0.45, sz * 0.45, sy)
    q = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    verts = [(a * (sx - c), -sy, b * (sz - c)) for a, b in q] + [(a * sx, -sy + c, b * sz) for a, b in q] + [(a * sx, sy, b * sz) for a, b in q]
    faces = [(0, 1, 2, 3)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append((4 + i, 4 + j, j, i))
        faces.append((8 + i, 8 + j, 4 + j, 4 + i))
    o = mesh_object(name, verts, faces)
    flat(o)
    return set_loc_rot(o, loc, rot)


def _parea(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def _inset_poly(poly, c):
    """Convex counter-clockwise polygon [(u, z)] with every edge moved inwards by c."""
    n, out = len(poly), []
    for i in range(n):
        p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % n]
        e1, e2 = (p1[0] - p0[0], p1[1] - p0[1]), (p2[0] - p1[0], p2[1] - p1[1])
        l1, l2 = hypot(*e1) or 1e-9, hypot(*e2) or 1e-9
        n1, n2 = (-e1[1] / l1, e1[0] / l1), (-e2[1] / l2, e2[0] / l2)
        k = c / max(1.0 + n1[0] * n2[0] + n1[1] * n2[1], 0.35)
        out.append((p1[0] + (n1[0] + n2[0]) * k, p1[1] + (n1[1] + n2[1]) * k))
    return out


def _clip_right(poly, ax, az, bx, bz):
    """The part of a convex polygon [(u, z)] on the right-hand side of the directed line A -> B."""
    out, n = [], len(poly)
    side = lambda q: (bx - ax) * (q[1] - az) - (bz - az) * (q[0] - ax)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        sa, sb = side(a), side(b)
        if sa <= 0:
            out.append(a)
        if (sa < 0 < sb) or (sb < 0 < sa):
            t = sa / (sa - sb)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def _stone_field(f, region, blocks, r, hc, ln, proud, joint=0.012, cham=(0.007, 0.012), clip=None, out0=0.0, cap=400, void=0.0):
    """Coursed rubble laid stone by stone on the wall plane f: courses of random height hc = (min, max), stones of
    random length ln, each a flat-faced block with chamfered arrises standing `proud` of the (mortar) wall behind,
    12 mm joints. blocks = rectangles (u0, u1, z0, z1) kept free (openings and their dressings, plinth, corbels):
    the stones are cut to butt against them. clip(poly) -> [polys] trims against curved outlines (the arch).
    Returns two tone sets of (verts, faces, grime, wear) - grime / wear are per-vertex so that every stone
    has its own shade and the joints sit in shadow."""
    u0, u1, z0, z1 = region
    blocks = [b for b in blocks if b[1] > u0 and b[0] < u1 and b[3] > z0 and b[2] < z1]
    free = (u1 - u0) * (z1 - z0) - sum((min(b[1], u1) - max(b[0], u0)) * (min(b[3], z1) - max(b[2], z0)) for b in blocks)
    n_est = max(free - void, 0.1) / ((hc[0] + hc[1]) * (ln[0] + ln[1]) * 0.25)
    k = max(1.0, (n_est / cap) ** 0.5)                # too many stones for the triangle budget: lay them larger
    hc, ln = (hc[0] * k, hc[1] * k), (ln[0] * k, ln[1] * k)
    hm = 0.5 * (hc[0] + hc[1])
    sets = [([], [], [], []), ([], [], [], [])]
    z = z0
    while z < z1 - 1e-4:
        h = r.u(*hc)
        if z1 - (z + h) < hc[0] * 0.7:
            h = z1 - z
        za, zb = z, z + h
        z += h
        u = u0 - r.u(0.0, ln[0])
        while u < u1 - 1e-4:
            l = r.u(*ln) * (0.7 + 0.3 * h / hm)
            if u1 - (u + l) < ln[0] * 0.6:
                l = u1 - u
            rects = [(max(u, u0) + joint / 2, min(u + l, u1) - joint / 2, za + joint / 2, zb - joint / 2)]
            u += l
            for (b0, b1, c0, c1) in blocks:
                nxt = []
                for (a0, a1, d0, d1) in rects:
                    if a1 <= b0 or a0 >= b1 or d1 <= c0 or d0 >= c1:
                        nxt.append((a0, a1, d0, d1))
                        continue
                    if b0 - a0 > 0.03:
                        nxt.append((a0, b0, d0, d1))
                    if a1 - b1 > 0.03:
                        nxt.append((b1, a1, d0, d1))
                    m0, m1 = max(a0, b0), min(a1, b1)
                    if m1 - m0 > 0.03:
                        if c0 - d0 > 0.03:
                            nxt.append((m0, m1, d0, c0))
                        if d1 - c1 > 0.03:
                            nxt.append((m0, m1, c1, d1))
                rects = nxt
            for (a0, a1, d0, d1) in rects:
                if a1 - a0 < 0.025 or d1 - d0 < 0.025:
                    continue
                polys = [[(a0, d0), (a1, d0), (a1, d1), (a0, d1)]]
                if clip is not None:
                    polys = clip(polys[0])
                for poly in polys:
                    if len(poly) < 3:
                        continue
                    ar = _parea(poly)
                    if ar < 0.0012:
                        continue
                    wmin = min(max(q[0] for q in poly) - min(q[0] for q in poly), max(q[1] for q in poly) - min(q[1] for q in poly))
                    top = _inset_poly(poly, min(r.u(*cham), 0.28 * wmin))
                    if _parea(top) < 0.3 * ar:
                        cu, cz = sum(q[0] for q in poly) / len(poly), sum(q[1] for q in poly) / len(poly)
                        top = [(cu + (q[0] - cu) * 0.72, cz + (q[1] - cz) * 0.72) for q in poly]
                    V, Fc, Gr, Wr = sets[0 if r.chance(0.6) else 1]
                    n, b = len(poly), len(V)
                    pr = r.u(*proud)
                    g = 0.5 * r.u(0.0, 1.0) ** 2
                    wr = r.u(0.1, 0.45) if r.chance(0.3) else 0.0
                    V += [f.pt(q[0], q[1], out0) for q in poly]
                    V += [f.pt(q[0] + r.u(-0.003, 0.003), q[1] + r.u(-0.003, 0.003), out0 + pr + r.u(-0.002, 0.002)) for q in top]
                    Fc += [(b + i, b + (i + 1) % n, b + n + (i + 1) % n, b + n + i) for i in range(n)]
                    Fc.append(tuple(range(b + n, b + 2 * n)))
                    Gr += [min(1.0, g + 0.4)] * n + [g] * n
                    Wr += [0.0] * n + [wr] * n
    return sets


def build(p, rng):
    W, D, PW = p.width, p.depth, p.passage_width
    G, SH = p.ground_height, p.storey_height
    n_up = int(p.storeys) - 1
    top = G + n_up * SH
    half = PW / 2
    zc = G - 0.62                                   # crown of the arch (clear height)
    rise = {"round": half, "segmental": 0.3 * PW, "pointed": 0.62 * PW}[p.arch]
    zs = zc - rise                                  # springing
    REC = 0.12                                      # the passage walls stand this far behind the clear line
    jx = half + REC
    RING = 0.38
    age = p.age
    sd = int(p.seed)

    # NOTE: the roof component must be built before anything else exists in the scene - roof_gable turns
    # every parentless mesh in the scene by 90 deg when its ridge runs along Y.
    tw = p.walls == "timber"
    roofc = use("roof_gable", width=D, depth=W, pitch=p.pitch, type=p.roof_type, ridge="y" if p.ridge == "x" else "x", overhang=0.45,
                covering=p.roof, sag=0.02 + 0.1 * age, gable="bargeboard" if p.walls in ("timber", "plaster") else "parapet",
                gable_wall={"timber": "timber", "plaster": "plaster", "brick": "brick", "stone": "plaster"}[p.walls],
                attic_window="lit" if rng.fork("attic").chance(p.lit_ratio) else "dark", rooflight="none", eaves="rafters" if tw else "cornice",
                ridge_tile="half_round", finial="ball", gutter="zinc", wall_color=p.wall_color if p.walls != "stone" else "#b3a68f",
                timber_color=p.timber_color, trim_color=p.timber_color, moss=0.2 + 0.5 * age, seed=sd + 60)

    # rubble: the library tile is 2 m with stones of 0.3-0.4 m - halved, the stones read 0.15-0.2 m like the other walls.
    # NB the shader divides the metre UVs by `scale`, so 0.5 = pattern half as large (the tk.mat docstring says the opposite)
    # ground = rubble: coursed rubble. Where one walks past it (both passage walls, the two arch fronts) every stone
    # is modelled - flat split faces 9-20 cm high standing proud of a recessed mortar bed; the flanks and the upper
    # storeys (seen from further away) take the fine coursed masonry texture at 0.7 scale (10 cm courses).
    # NB the shader divides the metre UVs by `scale`, so 0.7 = pattern smaller (the tk.mat docstring says the opposite)
    rub = p.ground == "rubble"
    # ONE material per family for the whole house (every material variant is a draw call): the composed windows, oriel,
    # dormers, roof and chimney are re-mapped onto these at the end of build() (see unify()).
    st_wall = mat("ashlar_limestone", scale=0.7) if (rub or p.walls == "stone") else mat("ashlar_limestone")
    coursed = st_wall
    st_back = st_wall                                         # the wall behind the modelled stones
    st = mat("stone_smooth", p.stone_color, scale=0.6)        # every modelled stone: dressings, rubble, ribs, jambs
    rb_m = (st, st) if rub else None                          # the two tones of the rubble come from the grime mask
    WO = 0.017 if rub else 0.0                                # fittings stand off the wall by the relief of the stones
    st_rough = st_q = st_d = st
    brick = mat("brick_red")
    plaster = mat("plaster_lime", p.wall_color if p.walls in ("timber", "plaster") else "#d8cfba")
    vault_m = {"ochre_brick": mat("brick_ochre"), "red_brick": brick, "limewash": plaster}[p.soffit]
    shutter_m = mat("wood_painted", p.paint)
    dark_m = mat("window_dark")
    pot_m = mat("terracotta")
    oak = mat("wood_oak", p.timber_color)
    iron = mat("iron_wrought")
    lit_m = mat("window_lit", "#ffc27a", glow=2.6, curtain=0.18)
    glow_m = mat("emissive", "#ffb45e", glow=GLOW)            # lantern panes, flames, gilt lettering
    ban_m = mat("canvas", p.banner_color, double=1)           # banners, sign board, cloth
    up_mat = {"timber": plaster, "plaster": plaster, "brick": brick, "stone": coursed}[p.walls]

    def lay_stones(name, f, region, blocks, r, **kw):
        """Model the coursed rubble of one wall area; returns the number of stones."""
        objs, n = [], 0
        for si, ((V, Fc, Gr, Wr), m) in enumerate(zip(_stone_field(f, region, blocks, r, **kw), rb_m)):
            if not Fc:
                continue
            o = mesh_object(name, V, Fc)
            flat(o)
            assign(o, m)
            uv_box(o)
            ga, wa = np.clip(np.array(Gr, dtype=np.float32) * 0.6 + 0.2 * si, 0.0, 1.0), np.array(Wr, dtype=np.float32)
            if len(o.data.vertices) == len(ga):
                paint(o, grime=lambda co, nn, a=ga: a, wear=lambda co, nn, a=wa: a)
            objs.append(o)
            n += len(Fc) // 5
        if objs:
            stone_objs.append(join(objs, name))
        return n

    F = {"f": _Face(0, -D / 2, 0, W), "r": _Face(W / 2, 0, 90, D), "b": _Face(0, D / 2, 180, W), "l": _Face(-W / 2, 0, 270, D)}
    stone_objs, wall_objs, up_objs, timber_objs, iron_objs = [], [], [], [], []

    def fbox(name, f, u, z, su, sz, thick=0.1, proud=0.03, roll=0.0, bev=0.008, seg=1):
        return _cbox(name, (su, thick, sz), f.pt(u, z, proud - thick / 2), (0, roll, f.th), c=bev)

    def wall(name, f, z0, z1, m, cell=0.6):
        nu = max(1, int(f.L / cell))
        nv = max(1, int((z1 - z0) / cell))
        P = [[f.pt(lerp(-f.L / 2, f.L / 2, i / nu), lerp(z0, z1, j / nv)) for j in range(nv + 1)] for i in range(nu + 1)]
        o = _sheet(name, P)
        assign(o, m)
        uv_box(o)
        flat(o)
        return o

    def band(name, z, prof, m, keys="frbl"):
        """Moulded course round the building; prof = [(projection, dz)] bottom -> top; mitred at the corners."""
        out = []
        for key in keys:
            f = F[key]
            nu = max(2, int(f.L / 0.7))
            P = [[f.pt((-f.L / 2 - o) + (f.L + 2 * o) * i / nu, z + dz, o) for (o, dz) in prof] for i in range(nu + 1)]
            b = _sheet(name, P)
            assign(b, m)
            smooth(b, 30)
            out.append(b)
        return out

    # ------------------------------------------------------------------ arch geometry
    NJ, NA = 4, 24
    apts, anr = _arch_poly(p.arch, half, zs, rise, NA)
    at0, S0 = _arc_param(apts, anr)

    def outline(rr, xclamp=None):
        L = [(-(half + rr), zs * i / NJ) for i in range(NJ)]
        L += [(x + nx * rr, z + nz * rr) for (x, z), (nx, nz) in zip(apts, anr)]
        L += [((half + rr), zs * (NJ - 1 - i) / NJ) for i in range(NJ)]
        if xclamp is not None:
            L = [(max(-xclamp, min(xclamp, x)), z) for x, z in L]
        return L

    def rectmap(x0, x1, zt):
        a, b = zt - zs, x1 - x0
        k1 = max(1, int(round(NA * a / (2 * a + b))))
        O = [(x0, zs * i / NJ) for i in range(NJ)]
        for k in range(NA + 1):
            if k <= k1:
                O.append((x0, zs + a * k / k1))
            elif k < NA - k1:
                O.append((x0 + b * (k - k1) / (NA - 2 * k1), zt))
            else:
                O.append((x1, zt - a * (k - (NA - k1)) / k1))
        O += [(x1, zs * (NJ - 1 - i) / NJ) for i in range(NJ)]
        return O

    def strip(name, inner, outer, y, nrad=5, flip=False):
        idx = list(range(len(inner)))
        if flip:
            idx.reverse()
        P = [[(lerp(inner[i][0], outer[i][0], j / nrad), y, lerp(inner[i][1], outer[i][1], j / nrad)) for j in range(nrad + 1)] for i in idx]
        o = _sheet(name, P)
        uv_box(o)
        flat(o)
        return o

    def ring(name, r, y0, y1, r_in, r_out, stone_len, key=False, jit=0.015, arc=None, keyout=0.16, bev=0.014, gap=0.006):
        at, S = arc or (at0, S0)
        n = max(3, int(round(S / stone_len)))
        if n % 2 == 0:
            n += 1
        cuts = [S * i / n for i in range(n + 1)]
        for i in range(1, n):
            if not (key and i in (n // 2, n // 2 + 1)):
                cuts[i] += r.u(-0.02, 0.02)
        verts, faces = [], []
        for i in range(n):
            k = key and i == n // 2
            _add_stone(verts, faces, at, cuts[i] + gap, cuts[i + 1] - gap, r_in - (0.03 if k else 0.0),
                       r_out + (keyout if k else r.u(-jit, jit)), y0 - (0.05 if k else r.u(-0.012, 0.012)), y1)
        o = mesh_object(name, verts, faces)
        recalc_normals(o)
        bevel(o, bev, segments=1, angle=30)
        smooth(o, 30)
        return o

    # ------------------------------------------------------------------ portal (front-local, rotated for the back)
    def portal(r, back):
        objs = []
        yF = -D / 2
        s = strip("ground_face", outline(0.30), rectmap(-W / 2, W / 2, G), yF)
        assign(s, st_back)
        wall_objs.append(s)
        if back:
            place(s, rot=(0, 0, 180))
        zj = zs - 0.17
        nrow = max(3, int(round(zj / 0.38)))
        hj = zj / nrow
        for sx in (-1, 1):
            for i in range(nrow):
                wj = (0.66 if i % 2 == 0 else 0.45) * r.vary(1.0, 0.05)
                pr = 0.09 if i == 0 else 0.05 + r.u(-0.006, 0.006)
                objs.append(box("jamb", (wj, 0.46 + pr, hj - 0.003), loc=(sx * (half + wj / 2), yF + 0.46 - (0.46 + pr) / 2, hj * (i + 0.5)),
                                bevel=0.012, segments=1))
            objs.append(box("impost", (0.74, 0.56, 0.17), loc=(sx * (half + 0.37), yF + 0.46 - 0.28, zs - 0.085), bevel=0.022, segments=2))
            objs.append(box("impost_fillet", (0.70, 0.52, 0.05), loc=(sx * (half + 0.35), yF + 0.46 - 0.26, zs - 0.195), bevel=0.012, segments=1))
            # label stop under the hood mould (the hood of a segmental arch ends higher and further in)
            nx0, nz0 = abs(anr[0][0]), anr[0][1]
            objs.append(box("label_stop", (0.15, 0.1, 0.13), loc=(sx * (half + nx0 * (RING + 0.05)), yF - 0.045, zs + nz0 * (RING + 0.05) + 0.065),
                            bevel=0.02, segments=1))
            if nz0 > 0.05:
                # skewback: the springer stone a segmental arch bears on (also closes the wedge between impost and ring)
                xo = half + max(nx0 * RING + 0.1, 0.36)
                prof = [(half - 0.004, zs), (xo, zs), (xo, zs + nz0 * RING + 0.02), (half + nx0 * RING, zs + nz0 * RING + 0.02)]
                objs.append(_prism("skewback", [(sx * x, z) for x, z in (prof if sx > 0 else prof[::-1])], yF - 0.04, yF + 0.46, bev=0.012))
        objs.append(ring("voussoirs", r, yF - 0.05, yF + 0.46, 0.0, RING, 0.34, key=True))
        hp = [(x + nx * (RING + 0.055), yF, z + nz * (RING + 0.055)) for (x, z), (nx, nz) in zip(apts, anr)]
        hood = sweep("hood_mould", hp, [(-0.05, 0.0), (0.05, 0.0), (0.05, 0.035), (0.028, 0.062), (-0.01, 0.07), (-0.05, 0.045)],
                     smooth_path=False, up=(0, -1, 0))
        objs.append(hood)
        if p.date:
            objs.append(text("date", str(p.date), size=0.088, depth=0.012, loc=(0, yF - 0.10, zc + 0.19), font="roman"))
        # guard stones protecting the jambs from cart wheels (outside the clear width)
        for sx in (-1, 1):
            g = lathe("guard_stone", [(0.0, 0.0), (0.17, 0.0), (0.175, 0.06), (0.15, 0.26), (0.105, 0.44), (0.06, 0.54), (0.0, 0.57)],
                      segments=12, loc=(sx * (half + 0.2), yF - 0.07, 0.0))
            displace_noise(g, 0.008, 0.15, seed=sd + (3 if back else 1))
            assign(g, st_rough)
            objs.append(g)
        for o in objs:
            if not o.name.startswith("guard_stone"):
                assign(o, st_q if o.name.startswith("jamb") else st)
        j = join(objs, "portal_b" if back else "portal_f")
        if back:
            place(j, rot=(0, 0, 180))
        stone_objs.append(j)

    portal(rng.fork("portal_f"), False)
    portal(rng.fork("portal_b"), True)

    # ------------------------------------------------------------------ ground storey shell, plinth, quoins, string course
    for key in "lr":
        wall_objs.append(wall("ground_side", F[key], 0.0, G, st_wall))
    rp = rng.fork("plinth")
    pl = []
    for key in "frbl":
        f = F[key]
        spans = [(-f.L / 2, -(half + 0.66)), (half + 0.66, f.L / 2)] if key in "fb" else [(-f.L / 2, f.L / 2)]
        for (a, b) in spans:
            if b - a < 0.25:
                continue
            u = a
            while u < b - 1e-3:
                ln = min(rp.u(0.7, 1.25), b - u)
                if b - (u + ln) < 0.35:
                    ln = b - u
                hh = 0.5 + rp.u(-0.012, 0.012)
                pl.append(fbox("plinth", f, u + ln / 2, hh / 2, ln - 0.004, hh, thick=0.2, proud=0.07 + rp.u(-0.006, 0.006), bev=0.018))
                u += ln
    rq = rng.fork("quoins")

    def quoins(z0, z1, out):
        n = max(1, int(round((z1 - z0) / 0.36)))
        h = (z1 - z0) / n
        for ci, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            for i in range(n):
                a, b = (0.62, 0.36) if (i + ci) % 2 == 0 else (0.36, 0.62)
                a, b = rq.vary(a, 0.06), rq.vary(b, 0.06)
                pr = 0.03 + rq.u(-0.005, 0.005)
                out.append(box("quoin", (a, b, h - 0.004), loc=(sx * (W / 2 + pr - a / 2), sy * (D / 2 + pr - b / 2), z0 + h * (i + 0.5)),
                               bevel=0.012, segments=1))

    quoins(0.5, G - 0.13, pl)
    if p.walls in ("plaster", "brick", "stone"):
        quoins(G + 0.1, top - 0.05, pl)
    for o in pl:
        assign(o, st_q)
    stone_objs.append(join(pl, "dressings"))
    sc = band("string_course", G, [(0.0, -0.13), (0.045, -0.13), (0.06, -0.11), (0.06, -0.05), (0.10, -0.01), (0.115, 0.03),
                                   (0.115, 0.06), (0.03, 0.095), (0.0, 0.1)], st)
    stone_objs += sc
    if p.walls != "timber":
        for k in range(2, n_up + 1):
            stone_objs += band("floor_band", G + (k - 1) * SH, [(0.0, -0.07), (0.035, -0.07), (0.05, -0.05), (0.05, 0.03), (0.0, 0.06)], st)

    # ------------------------------------------------------------------ passage interior
    ya, yb = -D / 2 + 0.3, D / 2 - 0.3
    ny = max(4, int((yb - ya) / 0.5))
    rr_ = rng.fork("ribs")
    rb = rng.fork("bays")
    pas_stone = []
    shutter_objs0 = []
    zwall = zs if p.ceiling == "vault" else G - 0.2
    nb = max(2, int(round(D / 1.9)))                  # wall bays (= the vault bays between the ribs)
    bw = D / nb
    bay_c = [-D / 2 + bw * (k + 0.5) for k in range(nb)]
    IW = {-1: _Face(-jx, 0, 90, D), 1: _Face(jx, 0, 270, D)}      # the two wall planes, normal into the passage
    vpts = [(x + nx * REC, z + nz * REC) for (x, z), (nx, nz) in zip(apts, anr)]
    # a segmental arch does not spring tangentially: bring the vault's two edges down onto the wall heads (no open slot)
    vpts[0], vpts[-1] = (-jx, zs), (jx, zs)

    def wall_x(z):
        """Distance of the wall / vault surface from the passage axis at height z."""
        if p.ceiling != "vault" or z <= zs:
            return jx
        for (x0, z0), (x1, z1) in zip(vpts[NA // 2:], vpts[NA // 2 + 1:]):
            if (z0 - z) * (z1 - z) <= 0 and abs(z1 - z0) > 1e-6:
                return lerp(x0, x1, (z - z0) / (z1 - z0))
        return jx

    def fquads(name, f, quads):
        """Flat faces from (u, z, out) corners in face space, counter-clockwise as seen from the visible side."""
        verts, faces = [], []
        for q in quads:
            b = len(verts)
            verts += [f.pt(*c) for c in q]
            faces.append(tuple(range(b, b + len(q))))
        o = mesh_object(name, verts, faces)
        flat(o)
        return o

    def grid_q(u0, u1, z0, z1, out, cell=0.45):
        nu_, nz_ = max(1, int(ceil((u1 - u0) / cell))), max(1, int(ceil((z1 - z0) / cell)))
        return [[(lerp(u0, u1, i / nu_), lerp(z0, z1, j / nz_), out), (lerp(u0, u1, (i + 1) / nu_), lerp(z0, z1, j / nz_), out),
                 (lerp(u0, u1, (i + 1) / nu_), lerp(z0, z1, (j + 1) / nz_), out), (lerp(u0, u1, i / nu_), lerp(z0, z1, (j + 1) / nz_), out)]
                for i in range(nu_) for j in range(nz_)]

    def cuts(a, b, req, step=0.5):
        pts = sorted([a, b] + [min(max(v, a), b) for v in req])
        P = [pts[0]]
        for v in pts[1:]:
            if v - P[-1] > 1e-4:
                P.append(v)
        out = [P[0]]
        for v0, v1 in zip(P, P[1:]):
            n = max(1, int(ceil((v1 - v0) / step - 1e-6)))
            out += [lerp(v0, v1, i / n) for i in range(1, n + 1)]
        return out

    def holed_wall(name, f, ua, ub, ztop, holes):
        """Wall sheet with rectangular openings (u0, u1, z0, z1); the grid follows the opening edges."""
        us = cuts(ua, ub, [h[0] for h in holes] + [h[1] for h in holes])
        zz = cuts(0.0, ztop, [h[2] for h in holes] + [h[3] for h in holes])
        vid, verts, faces = {}, [], []

        def v(i, j):
            if (i, j) not in vid:
                vid[(i, j)] = len(verts)
                verts.append(f.pt(us[i], zz[j]))
            return vid[(i, j)]

        for i in range(len(us) - 1):
            um = (us[i] + us[i + 1]) / 2
            for j in range(len(zz) - 1):
                zm = (zz[j] + zz[j + 1]) / 2
                if any(h[0] < um < h[1] and h[2] < zm < h[3] for h in holes):
                    continue
                faces.append((v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)))
        o = mesh_object(name, verts, faces)
        assign(o, st_back)
        uv_box(o)
        flat(o)
        return o

    # ---- the wall bays: shop windows, shuttered booths, stall niches, seat niches, doorways, plain bays
    zh = min(zwall - 0.26, 2.1)                       # head of the wall openings
    zd = min(zwall - 0.3, 2.02)                       # head of a doorway
    # a market passage: stalls (niche) on both sides wherever the wall allows, shop windows, booths, seats and doors between
    KINDS = {"mixed": [("niche", 5.0), ("shop", 2.2), ("shut", 0.7), ("seat", 0.6), ("door", 0.8), ("plain", 0.3)],
             "shops": [("shop", 3.0), ("niche", 2.0), ("shut", 1.0), ("door", 0.8)],
             "niches": [("niche", 3.0), ("seat", 1.2), ("plain", 0.5)],
             "plain": [("plain", 1.0)]}[p.bays]
    spans, bay_kind, n_door = {}, {}, 0
    for sx in (-1, 1):
        prev = None
        for k in range(nb):
            lo = max(-D / 2 + bw * k + 0.3, -D / 2 + 0.66)
            hi = min(-D / 2 + bw * (k + 1) - 0.3, D / 2 - 0.66)
            spans[(sx, k)] = (lo, hi)
            opts = [c for c in KINDS if c[0] == "niche" or c[0] not in (prev, bay_kind.get((-sx, k)))] or KINDS
            kind = rb.weighted(opts) if hi - lo >= 0.85 else "plain"
            if kind == "door" and (zd < 1.8 or n_door >= 2 or hi - lo < 1.0):
                kind = "shut"
            n_door += kind == "door"
            bay_kind[(sx, k)] = prev = kind
    # the inner bays (clear of the bracket lanterns and sign brackets at the mouths) carry the trade: two market stalls
    # on either side (one where the bays hold shop fronts), and a mixed passage always has a shop window as well
    wide = lambda key: spans[key][1] - spans[key][0] >= 0.85
    inner = [(sx, k) for sx in (-1, 1) for k in range(1, nb - 1) if wide((sx, k))] or [key for key in sorted(bay_kind) if wide(key)]
    if p.bays != "plain":
        for sx in (-1, 1):
            inn = [key for key in inner if key[0] == sx]
            need = min(len(inn), 1 if p.bays == "shops" else 2)
            while sum(bay_kind[key] == "niche" for key in inn) < need:
                cand = sorted((key for key in inn if bay_kind[key] != "niche"),
                              key=lambda key: (("shut", "plain", "door", "seat", "shop").index(bay_kind[key]), abs(key[1] - (nb - 1) / 2), key))
                bay_kind[cand[0]] = "niche"
        if p.bays == "mixed" and "shop" not in bay_kind.values():
            cand = [key for key in sorted(bay_kind) if wide(key) and bay_kind[key] in ("shut", "plain", "seat")] or \
                   [key for key in sorted(bay_kind) if wide(key) and key not in inner and bay_kind[key] == "niche"]
            if cand:
                bay_kind[cand[0]] = "shop"
    stall_keys = [key for key in sorted(bay_kind) if bay_kind[key] == "niche"]
    shop_keys = [key for key in sorted(bay_kind) if bay_kind[key] == "shop"]
    shop_lit = {key: rb.chance(p.bay_lit) for key in shop_keys}
    # lit windows: at least two (one) of the inner shops unless bay_lit is low, never all of several unless it is high
    pref = [key for key in shop_keys if key in inner] + [key for key in shop_keys if key not in inner]
    for key in pref[:(2 if p.bay_lit >= 0.3 else 1) if p.bay_lit > 0.15 else 0]:
        shop_lit[key] = True
    if len(shop_keys) > 2 and p.bay_lit < 0.85 and all(shop_lit.values()):
        shop_lit[pref[-1]] = False
    bay_st, bay_q, wares, bay_plaster, bay_tris0 = [], [], [], [], 0
    seat_backs, in_blocks = [], {-1: [], 1: []}       # coursed rubble: backs of the seat niches / areas kept free of stones
    holes, pl_gap = {-1: [], 1: []}, {-1: [], 1: []}
    lan_iron, lan_glass, brass_l, flame_m = iron, glow_m, glow_m, glow_m
    rc = rb.fork("canvas")
    awn_1 = mat("canvas", rc.pick([("#8a3b2e", "#3f5c4a"), ("#3d5570", "#8a3b2e"), ("#3f5c4a", "#a8843a"), ("#a8843a", "#3d5570")])[0],
                stripe=0.15, tint2="#d8cdb0", double=1)
    awn_m = [awn_1, awn_1]
    prod_m, herb_m, apple_m, wick_m = plaster, pot_m, pot_m, oak
    AWP = REC + 0.14                                  # reach of awnings, trade signs: at most 0.15 m into the clear width
    STALL_D = clamp(W / 2 - jx - 0.5, 0.3, 0.5)       # depth of the stall hatches
    CB = AWP - 0.03                                   # reach of the drop-leaf stall counters
    n_stall, t0_ = [0], int(rb.u(0, 4.999))
    bay_lamps, lamp_objs = [], []                     # (x, y, z of the flame, hangs under an awning, lights a stall)
    wo_at = lambda z: (jx - wall_x(z) + 0.004) if z > zwall else WO      # wall / vault surface in wall-face coordinates
    no_lamp = {(-1, 0), (1, nb - 1)}                  # these two bays hold the bracket lanterns

    def stall_lamp(loc, hang=0.04, under=False, s=1.0, stall=False):
        """Little hanging oil lamp (wire, vent cap, glowing glass chimney, font with burner); loc = top of the wire."""
        x, y, z = loc
        o = box("sl_wire", (0.004, 0.004, hang), loc=(x, y, z - hang / 2), bevel=0.0, segments=1)
        assign(o, iron)
        iron_objs.append(o)
        c = (x, y, z - hang)
        for nm_, prof, m_ in (("sl_cap", [(0.041, -0.034), (0.037, -0.024), (0.015, -0.01), (0.007, -0.004), (0.0, 0.0)], lan_iron),
                              ("sl_glass", [(0.019, -0.126), (0.028, -0.116), (0.034, -0.101), (0.037, -0.084), (0.036, -0.066), (0.031, -0.046), (0.026, -0.03)], lan_glass),
                              ("sl_font", [(0.0, -0.168), (0.016, -0.166), (0.027, -0.158), (0.032, -0.146), (0.03, -0.134), (0.022, -0.127), (0.0, -0.124)], lan_iron)):
            o = lathe(nm_, [(r_ * s, z_ * s) for r_, z_ in prof], segments=12, loc=c, smooth_angle=70.0)
            assign(o, m_)
            lamp_objs.append(o)
        bay_lamps.append((x, y, z - hang - 0.09 * s, under, stall))

    def bracket_lamp(f, u, z, out):
        """Stall lamp on a short forged arm with a stay."""
        for o in (fbox("lamp_plate", f, u, z - 0.035, 0.03, 0.13, thick=0.006, proud=out + 0.006, bev=0.002),
                  box("lamp_arm", (0.011, 0.16, 0.011), loc=f.pt(u, z, out + 0.08), rot=(0, 0, f.th), bevel=0.0, segments=1),
                  box("lamp_stay", (0.009, 0.135, 0.009), loc=f.pt(u, z - 0.045, out + 0.052), rot=(-41.0, 0, f.th), bevel=0.0, segments=1)):
            assign(o, iron)
            iron_objs.append(o)
        stall_lamp(f.pt(u, z - 0.005, out + 0.14))

    def awning(f, uc, aw, zr, zf, proj, m):
        """Striped canvas awning: laced to a wall rail at zr, stretched over a front rod at zf on two forged arms,
        scalloped valance. Returns the (u, out) of the front rod."""
        o_r = wo_at(zr) + 0.012
        ns = max(4, int(round(aw / 0.2)))
        P = [[f.pt(uc + (0.5 - i / ns) * aw, lerp(zr, zf, j / 3) - (0.02 + 0.016 * (1 - (2 * i / ns - 1) ** 2)) * sin(pi * j / 3),
                   lerp(o_r, proj, j / 3)) for j in range(4)] for i in range(ns + 1)]
        cl = _sheet("awning", P)
        vv, vf = [], []
        for i in range(ns):
            ua_, ub_ = uc + (i / ns - 0.5) * aw, uc + ((i + 1) / ns - 0.5) * aw
            b = len(vv)
            vv += [f.pt(ua_, zf - 0.075, proj + 0.004), f.pt((ua_ + ub_) / 2, zf - 0.115, proj + 0.006), f.pt(ub_, zf - 0.075, proj + 0.004),
                   f.pt(ub_, zf + 0.002, proj), f.pt(ua_, zf + 0.002, proj)]
            vf.append(tuple(range(b, b + 5)))
        va = mesh_object("awning_valance", vv, vf)
        for o in (cl, va):
            assign(o, m)
            uv_planar(o, (f.ux, f.uy, 0), (0, 0, 1))
            smooth(o, 40)
            wares.append(o)
        for o in ([box("awn_arm", (0.014, proj - WO, 0.014), loc=f.pt(uc + sg * (aw / 2 - 0.01), zf - 0.012, (proj + WO) / 2), rot=(0, 0, f.th),
                       bevel=0.0, segments=1) for sg in (-1, 1)] +
                  [box("awn_stay", (0.01, hypot(proj * 0.62, 0.2), 0.01), loc=f.pt(uc + sg * (aw / 2 - 0.01), zf - 0.112, WO + proj * 0.31),
                       rot=(-degrees(atan2(0.2, proj * 0.62)), 0, f.th), bevel=0.0, segments=1) for sg in (-1, 1)] +
                  [box("awn_rod", (aw + 0.05, 0.016, 0.016), loc=f.pt(uc, zf - 0.01, proj), rot=(0, 0, f.th), bevel=0.0, segments=1),
                   box("awn_rail", (aw, 0.012, 0.02), loc=f.pt(uc, zr, o_r - 0.004), rot=(0, 0, f.th), bevel=0.0, segments=1)]):
            assign(o, iron)
            iron_objs.append(o)

    def ware(o, m_, sm=None):
        assign(o, m_)
        if sm:
            smooth(o, sm)
        wares.append(o)
        return o

    # ---- the goods of the stalls.  Every kind is modelled once - smooth lathes and sweeps of sensible segment counts, the
    # colours painted into the masks of the house's own three materials (oak: wear = pale crust, straw, wicker, cheese
    # rind; terracotta: grime = dark glaze / smoked, special = green; limewash plaster: eggs, garlic, cream cheese, linen)
    # - and stamped where the rule of the stall's trade puts it (instances_to_mesh): a full stall is about 2 500 triangles.
    G_INST = {}
    QN = clamp(5.5 / max(1, len(stall_keys)), 0.62, 1.0)     # a long lane of stalls: each a little less crammed
    cloth = [ban_m, awn_1, plaster, pot_m]

    def gm(f, u, z, out, rz=0.0, s=1.0, rx=0.0, ry=0.0):
        """Matrix of one ware on the wall face f: local +X runs along the wall, -Y points into the passage."""
        sc = (s, s, s) if isinstance(s, (int, float)) else s
        return (Matrix.Translation(f.pt(u, z, out)) @ Matrix.Rotation(radians(f.th + rz), 4, "Z") @ Matrix.Rotation(radians(rx), 4, "X")
                @ Matrix.Rotation(radians(ry), 4, "Y") @ Matrix.Diagonal((sc[0], sc[1], sc[2], 1.0)))

    def stamp(kind, M):
        G_INST.setdefault(kind, []).append(M)

    def bake(o):
        bpy.context.view_layer.update()
        o.data.transform(o.matrix_world)
        o.matrix_world = Matrix.Identity(4)
        return o

    def tone(o, m_, g=0.0, w=0.0, s=0.0):
        bake(o)
        assign(o, m_)
        paint(o, grime=g, wear=w, special=s, mode="set")
        return o

    def rev(name, prof, n, a0=0.0, a1=360.0, mod=None):
        """Own lathe for what tk.lathe does not do: a radius modulated round the axis (pumpkin ribs) and part turns
        with their cut faces (a cheese with a wedge out).  Returns (shell, cut faces or None)."""
        full = a1 - a0 >= 359.9
        V, Fc, idx = [], [], {}

        def vid(i, j):
            key = (i, 0) if prof[i][0] < 1e-6 else (i, j % n if full else j)
            if key not in idx:
                a = radians(a0 + (a1 - a0) * key[1] / n)
                r = prof[i][0] * (mod(key[1], i) if mod else 1.0)
                idx[key] = len(V)
                V.append((r * cos(a), r * sin(a), prof[i][1]))
            return idx[key]

        for j in range(n):
            for i in range(len(prof) - 1):
                q = []
                for v in (vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)):
                    if v not in q:
                        q.append(v)
                if len(q) >= 3:
                    Fc.append(tuple(q))
        o = mesh_object(name, V, Fc)
        smooth(o, 75)
        uv_box(o)
        cp = None
        if not full:
            CV, CF = [], []
            pts = ([(0.0, prof[0][1])] if prof[0][0] > 1e-6 else []) + list(prof) + ([(0.0, prof[-1][1])] if prof[-1][0] > 1e-6 else [])
            for a, flip in ((a0, False), (a1, True)):
                b = len(CV)
                CV += [(r * cos(radians(a)), r * sin(radians(a)), z) for r, z in pts]
                ids = list(range(b, b + len(pts)))
                CF.append(tuple(ids[::-1] if flip else ids))
            cp = mesh_object(name + "_cut", CV, CF)
            flat(cp)
            uv_box(cp)
        return o, cp

    def hz_of(prof):
        top = sorted(prof[max(range(len(prof)), key=lambda i: prof[i][0]):])

        def hz(x, y):
            r = hypot(x, y)
            for (r0, z0), (r1, z1) in zip(top, top[1:]):
                if r0 <= r <= r1:
                    return lerp(z0, z1, (r - r0) / max(r1 - r0, 1e-9))
            return top[0][1] if r < top[0][0] else top[-1][1]

        return hz

    def scores(hz, lines, wd=0.012, lift=0.0025, n=4):
        """The cuts of a loaf: pale pointed strips lying on the crust."""
        V, Fc = [], []
        for (xa, ya_, xb_, yb_) in lines:
            l = hypot(xb_ - xa, yb_ - ya_)
            px, py = -(yb_ - ya_) / l * wd / 2, (xb_ - xa) / l * wd / 2
            b = len(V)
            for i in range(n + 1):
                t = i / n
                ww = 0.3 + 0.7 * sin(pi * t)
                for sg in (-1, 1):
                    xx, yy = lerp(xa, xb_, t) + sg * px * ww, lerp(ya_, yb_, t) + sg * py * ww
                    V.append((xx, yy, hz(xx, yy) + lift))
            Fc += [(b + 2 * i, b + 2 * i + 2, b + 2 * i + 3, b + 2 * i + 1) for i in range(n)]
        o = mesh_object("score", V, Fc)
        smooth(o, 60)
        uv_planar(o, (1, 0, 0), (0, 1, 0))
        return tone(o, plaster, w=0.5)

    def crust(dark, h):
        return dict(g=lambda co, n: 0.28 * dark * np.clip(co[:, 2] / h, 0.0, 1.0) ** 2,
                    w=lambda co, n: 0.95 - (0.2 + 0.35 * dark) * np.clip(co[:, 2] / h, 0.0, 1.0))

    P_LOAF = [(0.072, 0.0), (0.1, 0.018), (0.106, 0.043), (0.09, 0.07), (0.05, 0.087), (0.0, 0.092)]
    P_ROLL = [(0.03, 0.0), (0.046, 0.018), (0.035, 0.042), (0.0, 0.052)]
    P_EGG = [(0.0, -0.027), (0.0165, -0.018), (0.021, -0.002), (0.0135, 0.018), (0.0, 0.027)]
    P_WHEEL = [(0.108, 0.0), (0.128, 0.012), (0.132, 0.043), (0.128, 0.074), (0.108, 0.086), (0.0, 0.089)]
    P_BULB = [(0.0, 0.0), (0.02, 0.006), (0.032, 0.021), (0.033, 0.037), (0.021, 0.055), (0.006, 0.073)]
    P_JUG = [(0.048, 0.0), (0.076, 0.045), (0.08, 0.095), (0.05, 0.16), (0.044, 0.19), (0.054, 0.214), (0.046, 0.212), (0.0, 0.188)]
    LA = lambda name, prof, seg, **kw: lathe(name, prof, segments=seg, smooth_angle=80.0, **kw)

    def mk_loaf(cuts, dark):
        return join([tone(LA("loaf", P_LOAF, 14), oak, **crust(dark, 0.092)), scores(hz_of(P_LOAF), cuts)], "loaf")

    def mk_long():
        R = 0.043
        o = LA("long_loaf", [(0.0, -0.21), (0.024, -0.2), (0.04, -0.15), (R, 0.0), (0.04, 0.15), (0.024, 0.2), (0.0, 0.21)], 10, loc=(0, 0, R), rot=(0, 90, 0))
        sc_ = scores(lambda x, y: R + max(R * R - y * y, 0.0) ** 0.5, [(c - 0.02, -0.02, c + 0.02, 0.02) for c in (-0.12, -0.06, 0.0, 0.06, 0.12)], wd=0.01, n=2)
        return join([tone(o, oak, **crust(0.6, 2 * R)), sc_], "long_loaf")

    def mk_roll():
        return join([tone(LA("roll", P_ROLL, 10), oak, **crust(0.5, 0.052)), scores(hz_of(P_ROLL), [(-0.03, 0.0, 0.03, 0.0)], wd=0.008, lift=0.002, n=2)], "roll")

    def mk_ring():
        r_ = tone(torus("ring_loaf", 0.07, 0.023, loc=(0, 0, -0.2), rot=(90, 0, 0), major_seg=16, minor_seg=7), oak, g=0.08, w=0.72)
        s_ = tone(box("loaf_string", (0.004, 0.004, 0.135), loc=(0, 0, -0.0675), bevel=0.0, segments=1), oak, w=1.0)
        return join([r_, s_], "ring_loaf")

    def mk_cut(a1, n):
        o, cp = rev("cheese", P_WHEEL, n, 0.0, a1)
        return join([tone(o, oak, g=0.05, w=0.9), tone(cp, plaster, w=0.35)], "cheese_cut")

    def mk_braid(n, m_, sc, seg):
        """Onions / garlic plaited into a string: real bulbs, their necks tucked into the straw plait."""
        ps = [tone(sweep("plait", [(0, 0, 0.0), (0.004, 0, -0.06 - 0.3 * n * 0.047 * sc), (-0.003, 0, -0.07 - n * 0.047 * sc)], 0.007, sides=5, smooth_path=False),
                   oak, w=1.0)]
        for i in range(n):
            b = LA("bulb", P_BULB, seg)
            b.matrix_world = (Matrix.Translation((0, 0, -0.07 - i * 0.047 * sc)) @ Matrix.Rotation(i * 2.4, 4, "Z") @ Matrix.Rotation(radians(-42), 4, "Y")
                              @ Matrix.Translation((0, 0, -0.073 * sc)) @ Matrix.Diagonal((sc, sc, sc * (1.0 if sc > 0.8 else 0.85), 1.0)))
            v_ = (i * 37 % 10) / 10.0
            ps.append(tone(b, m_, g=0.05 + (0.1 * v_ if sc < 0.8 else 0.0), w=(0.3 + 0.5 * v_) if sc > 0.8 else 0.5))
        return join(ps, "braid")

    def mk_herbs():
        o = LA("herbs", [(0.0, -0.27), (0.022, -0.263), (0.04, -0.243), (0.05, -0.217), (0.046, -0.2), (0.056, -0.181), (0.05, -0.16), (0.053, -0.14),
                         (0.042, -0.121), (0.04, -0.1), (0.026, -0.08), (0.014, -0.06), (0.011, -0.03), (0.015, 0.0)], 12)
        displace_noise(o, 0.006, 0.03, seed=sd + 5)
        return tone(o, pot_m, s=lambda co, n: np.where(co[:, 2] < -0.075, 0.95, 0.4), w=lambda co, n: np.where(co[:, 2] < -0.075, 0.0, 0.5))

    def mk_sausage():
        return tone(sweep("sausage", [(-0.028, 0, 0.0), (-0.037, 0, -0.07), (-0.035, 0, -0.15), (-0.019, 0, -0.2), (0.0, 0, -0.216), (0.019, 0, -0.2),
                                      (0.035, 0, -0.15), (0.037, 0, -0.07), (0.028, 0, 0.0)], 0.017, sides=8, smooth_path=False), pot_m, g=0.6)

    def mk_ham():
        o = LA("ham", [(0.0, -0.33), (0.045, -0.315), (0.085, -0.25), (0.09, -0.19), (0.06, -0.1), (0.028, -0.045), (0.016, -0.02), (0.013, 0.0)], 12)
        return tone(o, pot_m, g=lambda co, n: 0.5 * np.clip((-co[:, 2] - 0.04) / 0.05, 0.0, 1.0), w=lambda co, n: 0.8 * np.clip((co[:, 2] + 0.05) / 0.03, 0.0, 1.0))

    def mk_pumpkin():
        P = [(0.05, 0.0), (0.12, 0.022), (0.147, 0.072), (0.13, 0.125), (0.07, 0.152), (0.022, 0.147), (0.0, 0.138)]
        o, _ = rev("pumpkin", P, 16, mod=lambda j, i: 1.0 + (0.05 if j % 2 == 0 else -0.05) * (0.5 + 0.5 * sin(pi * i / 6)))
        st_ = LA("stem", [(0.017, 0.133), (0.011, 0.16), (0.014, 0.178), (0.0, 0.181)], 6)
        return join([tone(o, pot_m, g=0.05, w=0.15), tone(st_, pot_m, g=0.3, s=0.7)], "pumpkin")

    def mk_noisy(name, prof, seg, st, m_, **kw):
        o = LA(name, prof, seg)
        displace_noise(o, st, 0.05, seed=sd + 6)
        return tone(o, m_, **kw)

    def mk_crate():
        cw, cd, ch, t = 0.36, 0.25, 0.11, 0.012
        ps = [box("crate", (cw, t, 0.045), loc=(0, sg * (cd / 2 - t / 2), zz), bevel=0.0, segments=1) for sg in (-1, 1) for zz in (0.0275, 0.0875)]
        ps += [box("crate", (t, cd - 2 * t, ch), loc=(sg * (cw / 2 - t / 2), 0, ch / 2), bevel=0.0, segments=1) for sg in (-1, 1)]
        ps.append(box("crate", (cw - 2 * t, cd - 2 * t, t), loc=(0, 0, t / 2 + 0.004), bevel=0.0, segments=1))
        for i, o in enumerate(ps):
            uv_grain(o, "X" if i < 4 else "Y")
            tone(o, oak, g=0.1 + 0.08 * (i % 3), w=0.6)
        return join(ps, "crate")

    def mk_basket():
        o = LA("basket", [(0.09, 0.0), (0.11, 0.045), (0.116, 0.05), (0.126, 0.098), (0.136, 0.104), (0.13, 0.114), (0.118, 0.1), (0.11, 0.066)], 12)
        tone(o, oak, g=lambda co, n: 0.3 * np.clip(1.0 - co[:, 2] / 0.08, 0.0, 1.0), w=0.5)
        fl = LA("basket_straw", [(0.113, 0.068), (0.06, 0.082), (0.0, 0.086)], 8)
        displace_noise(fl, 0.006, 0.03, seed=sd + 8)
        tone(fl, oak, g=0.1, w=1.0)
        hd = sweep("basket_handle", [(-0.131 * cos(radians(a_)) * (1.0 if a_ > 0 else 1.0), 0, 0.108 + 0.182 * sin(radians(a_)) ** 0.8) for a_ in range(0, 181, 15)],
                   0.008, sides=5, smooth_path=False)
        return join([o, fl, tone(hd, oak, g=0.15, w=0.45)], "basket")

    def mk_sack():
        o = LA("sack", [(0.1, 0.0), (0.14, 0.04), (0.15, 0.14), (0.13, 0.24), (0.095, 0.29), (0.105, 0.31), (0.09, 0.318), (0.075, 0.295), (0.0, 0.288)], 12)
        displace_noise(o, 0.012, 0.12, seed=sd + 9)
        return tone(o, plaster, g=lambda co, n: np.where((co[:, 2] > 0.27) & (np.hypot(co[:, 0], co[:, 1]) < 0.08), 0.0, 0.42),
                    w=lambda co, n: np.where((co[:, 2] > 0.27) & (np.hypot(co[:, 0], co[:, 1]) < 0.08), 1.0, 0.0))

    def mk_jug(m_, glaze):
        o = LA("jug", P_JUG, 18)

        def spout(co):
            if co.z > 0.186 and co.x < 0.0:
                co.x -= 0.02 * (max(0.0, -co.x) / max(hypot(co.x, co.y), 1e-6)) ** 6 * min(1.0, (co.z - 0.186) / 0.02)
            return co

        transform_verts(o, spout)
        h = sweep("jug_handle", [(0.046, 0, 0.198), (0.082, 0, 0.2), (0.104, 0, 0.165), (0.1, 0, 0.115), (0.079, 0, 0.085)], 0.009, sides=5, smooth_path=False)
        gl = (lambda co, n: glaze * np.clip((co[:, 2] - 0.09) / 0.02, 0.0, 1.0)) if glaze else (0.12 if m_ is plaster else 0.0)
        return join([tone(o, m_, g=gl), tone(h, m_, g=gl)], "jug")

    def mk_mug(m_, glaze):
        o = LA("mug", [(0.036, 0.0), (0.042, 0.01), (0.04, 0.085), (0.044, 0.092), (0.037, 0.09), (0.0, 0.03)], 14)
        h = sweep("mug_handle", [(0.04, 0, 0.076), (0.066, 0, 0.07), (0.07, 0, 0.04), (0.042, 0, 0.022)], 0.006, sides=4, smooth_path=False)
        gl = (lambda co, n: glaze * np.clip((co[:, 2] - 0.04) / 0.015, 0.0, 1.0)) if glaze else 0.1
        return join([tone(o, m_, g=gl), tone(h, m_, g=gl)], "mug")

    def mk_dish(name, prof, seg, m_):
        return tone(LA(name, prof, seg), m_, g=(lambda co, n: 0.55 * (n[:, 2] > 0.25)) if m_ is pot_m else 0.08)

    def mk_scales():
        ps = [box("scales", (0.13, 0.07, 0.014), loc=(0, 0, 0.007), bevel=0.003, segments=1),
              box("scales", (0.012, 0.012, 0.25), loc=(0, 0, 0.139), bevel=0.0, segments=1),
              box("scales", (0.3, 0.007, 0.012), loc=(0, 0, 0.262), bevel=0.0, segments=1)]
        for sg in (-1, 1):
            ps.append(LA("scale_pan", [(0.0, 0.085), (0.05, 0.09), (0.062, 0.104), (0.058, 0.105), (0.047, 0.094), (0.0, 0.09)], 8, loc=(sg * 0.14, 0, 0)))
            ps += [box("scale_wire", (0.0035, 0.0035, 0.162), loc=(sg * 0.14, s2 * 0.026, 0.181), rot=(s2 * 18.2, 0, 0), bevel=0.0, segments=1) for s2 in (-1, 1)]
        return join([tone(o, iron, s=0.04) for o in ps], "scales")

    def mk_cloth(what, m_):
        kw = dict(g=0.25) if m_ is pot_m else dict(w=0.2) if m_ is plaster else {}
        if what == "bolt":
            o = cylinder("cloth_bolt", 0.05, 0.36, loc=(0, 0, 0.05), rot=(0, 90, 0), verts=14, bevel=0.012, segments=1)
            smooth(o, 60)
        elif what == "fold":
            o = box("cloth_fold", (0.27, 0.21, 0.034), loc=(0, 0, 0.017), bevel=0.012, segments=1)
            smooth(o, 50)
        else:                                             # a length hung over the rail
            o = _sheet("cloth_length", [[((0.5 - i / 4) * 0.2 * (1 - 0.12 * j / 3), 0.014 * sin(i * 1.9) * (0.3 + 0.7 * j / 3), -0.46 * j / 3) for j in range(4)]
                                        for i in range(5)])
            smooth(o, 60)
        return tone(o, m_, **kw)

    MAKE = {
        "boule": lambda: mk_loaf([(-0.066, 0.0, 0.066, 0.0), (0.0, -0.066, 0.0, 0.066)], 0.35),
        "boule_d": lambda: mk_loaf([(-0.058, -0.032, 0.058, -0.032), (-0.066, 0.0, 0.066, 0.0), (-0.058, 0.032, 0.058, 0.032)], 0.9),
        "batard": lambda: mk_loaf([(c - 0.012, -0.055, c + 0.012, 0.055) for c in (-0.045, 0.0, 0.045)], 0.6),
        "long": mk_long, "roll": mk_roll, "ring": mk_ring,
        "egg_w": lambda: tone(LA("egg", P_EGG, 8), plaster, w=0.9),
        "egg_b": lambda: tone(LA("egg", P_EGG, 8), pot_m, w=1.0),
        "wheel": lambda: tone(lathe("cheese", P_WHEEL, segments=18, smooth_angle=60.0), oak, g=0.05, w=0.9),
        "wheel_p": lambda: tone(lathe("cheese", P_WHEEL, segments=16, smooth_angle=60.0), plaster, g=0.1, w=0.3),
        "wheel_r": lambda: tone(lathe("cheese", P_WHEEL, segments=14, smooth_angle=60.0), pot_m, g=0.12),
        "wheel_cut": lambda: mk_cut(300.0, 14), "wedge": lambda: mk_cut(52.0, 3),
        "onion": lambda: mk_braid(6, pot_m, 1.0, 10), "garlic": lambda: mk_braid(7, plaster, 0.68, 9),
        "herbs": mk_herbs, "sausage": mk_sausage, "ham": mk_ham, "pumpkin": mk_pumpkin,
        "gourd": lambda: tone(LA("gourd", [(0.0, 0.0), (0.045, 0.008), (0.075, 0.05), (0.07, 0.1), (0.04, 0.145), (0.03, 0.18), (0.036, 0.215), (0.022, 0.245),
                                           (0.0, 0.25)], 12), pot_m, w=0.3, s=0.9),
        "cabbage": lambda: mk_noisy("cabbage", [(0.035, 0.0), (0.085, 0.03), (0.095, 0.08), (0.075, 0.13), (0.035, 0.155), (0.0, 0.16)], 12, 0.007, plaster, g=0.12, s=0.62),
        "apple_r": lambda: tone(LA("apple", [(0.02, 0.0), (0.036, 0.025), (0.031, 0.05), (0.011, 0.061), (0.0, 0.055)], 9), pot_m, g=0.1),
        "apple_g": lambda: tone(LA("apple", [(0.02, 0.0), (0.036, 0.025), (0.031, 0.05), (0.011, 0.061), (0.0, 0.055)], 9), pot_m, w=0.5, s=0.5),
        "heap_r": lambda: mk_noisy("heap", [(0.17, 0.0), (0.14, 0.035), (0.075, 0.055), (0.0, 0.06)], 8, 0.012, pot_m, g=0.45),
        "heap_g": lambda: mk_noisy("heap", [(0.17, 0.0), (0.14, 0.035), (0.075, 0.055), (0.0, 0.06)], 8, 0.012, pot_m, g=0.35, w=0.5, s=0.5),
        "crate": mk_crate, "basket": mk_basket, "sack": mk_sack, "scales": mk_scales,
        "board": lambda: tone(box("board", (0.36, 0.22, 0.018), loc=(0, 0, 0.009), bevel=0.004, segments=1), oak, g=0.1, w=0.6),
        "jug": lambda: mk_jug(pot_m, 0.0), "jug_g": lambda: mk_jug(pot_m, 0.6), "jug_c": lambda: mk_jug(plaster, 0.0),
        "mug": lambda: mk_mug(pot_m, 0.6), "mug_c": lambda: mk_mug(plaster, 0.0),
        "bowl": lambda: mk_dish("bowl", [(0.045, 0.0), (0.1, 0.045), (0.108, 0.05), (0.1, 0.047), (0.045, 0.012), (0.0, 0.01)], 20, pot_m),
        "bowl_rim": lambda: mk_dish("bowl", [(0.088, 0.035), (0.1, 0.045), (0.108, 0.05), (0.1, 0.047)], 20, pot_m),
        "plate": lambda: mk_dish("plate", [(0.05, 0.0), (0.105, 0.014), (0.112, 0.019), (0.1, 0.02), (0.055, 0.009), (0.0, 0.009)], 20, pot_m),
        "plate_c": lambda: mk_dish("plate", [(0.05, 0.0), (0.105, 0.014), (0.112, 0.019), (0.1, 0.02), (0.055, 0.009), (0.0, 0.009)], 20, plaster),
        "jar": lambda: tone(LA("jar", [(0.06, 0.0), (0.098, 0.06), (0.098, 0.14), (0.066, 0.2), (0.073, 0.212), (0.03, 0.232), (0.014, 0.236), (0.017, 0.254),
                                       (0.0, 0.258)], 16), pot_m, g=lambda co, n: 0.6 * np.clip((co[:, 2] - 0.12) / 0.02, 0.0, 1.0)),
        "flowerpot": lambda: tone(LA("flowerpot", [(0.034, 0.0), (0.05, 0.08), (0.058, 0.085), (0.058, 0.1), (0.047, 0.1), (0.04, 0.04), (0.0, 0.04)], 12), pot_m, w=0.15),
        "bottle": lambda: tone(LA("bottle", [(0.035, 0.0), (0.05, 0.03), (0.052, 0.12), (0.02, 0.18), (0.016, 0.24), (0.022, 0.25), (0.0, 0.25)], 12), pot_m, g=0.8),
        "crock": lambda: tone(LA("crock", [(0.05, 0.0), (0.062, 0.02), (0.062, 0.1), (0.067, 0.108), (0.056, 0.108), (0.0, 0.1)], 12), plaster, g=0.12),
        "skein": lambda: tone(LA("skein", [(0.0, -0.2), (0.022, -0.18), (0.028, -0.11), (0.012, -0.085), (0.024, -0.06), (0.012, -0.02), (0.004, 0.0)], 7), pot_m, g=0.2),
        "skein_c": lambda: tone(LA("skein", [(0.0, -0.2), (0.022, -0.18), (0.028, -0.11), (0.012, -0.085), (0.024, -0.06), (0.012, -0.02), (0.004, 0.0)], 7), plaster, w=0.3),
        "ball": lambda: tone(LA("yarn", [(0.0, 0.0), (0.03, 0.012), (0.038, 0.038), (0.026, 0.066), (0.0, 0.076)], 9), pot_m, g=0.15),
        "ball_c": lambda: tone(LA("yarn", [(0.0, 0.0), (0.03, 0.012), (0.038, 0.038), (0.026, 0.066), (0.0, 0.076)], 9), plaster, g=0.1),
        "ball_g": lambda: tone(LA("yarn", [(0.0, 0.0), (0.03, 0.012), (0.038, 0.038), (0.026, 0.066), (0.0, 0.076)], 9), pot_m, w=0.3, s=0.75),
    }
    for i_, m_ in enumerate(cloth):
        for what in ("bolt", "fold", "len"):
            MAKE["%s%d" % (what, i_)] = lambda what=what, m_=m_: mk_cloth(what, m_)

    TRADES = ["baker", "grocer", "cheese", "potter", "draper"]

    def stock_stall(f, uc, w, zt, z1, d, sh_z, trade, ki, rod, lamp_u=None):
        """Stock one stall by the rule of its trade: three groups of wares on the counter (two on a narrow one), a back row
        under the shelves, the two shelves, things strung from the hook rail in the hatch and from the awning rod."""
        full = (0.25 + 0.75 * p.stock) * QN
        have = (lambda q=1.0: rb.chance(min(1.0, full * q))) if p.stock > 0.01 else (lambda q=1.0: False)
        jr = lambda a: rb.u(-a, a)
        S = lambda kind, u, z, out, **kw: stamp(kind, gm(f, u, z, out, **kw))
        ns = int(clamp((w - 0.04) // 0.4, 1, 3))
        sl = [uc + (i - (ns - 1) / 2) * (w - 0.14) / ns for i in range(ns)]
        order = list(rb.pick([(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]))
        cl_m = {"baker": plaster, "cheese": ban_m, "grocer": pot_m if ki % 2 else None}.get(trade)
        zc = zt + 0.002
        if cl_m is not None and p.stock > 0.01:
            drape(f, uc, zt, w - 0.1, cl_m, drop=0.24, back=-0.3)
            zc = zt + 0.008
        bk = -d + 0.3                                          # the back row stands in front of the shelves

        def eggs(u, o, s_, kinds):
            S("basket", u, zc, o, rz=jr(40), s=s_)
            for i in range(7):
                a_, r_ = i * 1.047 + 0.3, (0.062 if i < 6 else 0.0) * s_
                S(rb.pick(kinds), u + r_ * cos(a_), zc + (0.1 + (0.02 if i == 6 else jr(0.004))) * s_, o + r_ * sin(a_), rz=jr(180), rx=90 + jr(25))

        def crate_of(u, o, kind, heap):
            Mc = gm(f, u, zc + 0.031, o, rz=jr(4), rx=14.0)
            stamp("crate", Mc)
            stamp(heap, Mc @ Matrix.Translation((0, 0, 0.03)) @ Matrix.Diagonal((0.95, 0.64, 1.0, 1.0)))
            for i in range(4):
                for j in range(2):
                    stamp(kind, Mc @ Matrix.Translation(((i - 1.5) * 0.079 + jr(0.006), (j - 0.5) * 0.1 + jr(0.006), 0.058 + jr(0.006)))
                          @ Matrix.Rotation(jr(3.1), 4, "Z") @ Matrix.Rotation(jr(0.5), 4, "X") @ Matrix.Diagonal((rb.u(0.9, 1.08),) * 3 + (1.0,)))
            for i in range(2):
                stamp(kind, Mc @ Matrix.Translation(((i - 0.5) * 0.085 + jr(0.01), jr(0.012), 0.1)) @ Matrix.Rotation(jr(3.1), 4, "Z") @ Matrix.Rotation(jr(0.6), 4, "X"))

        def g_baker(i, u):
            if i == 0:                                         # loaves heaped on the cloth
                S("boule", u - 0.09, zc, 0.095 + jr(0.012), rz=jr(180), s=rb.u(0.92, 1.04))
                S("boule_d", u + 0.11, zc, 0.07 + jr(0.012), rz=jr(180), s=rb.u(0.88, 1.0))
                S("batard", u, zc, -0.115, rz=jr(20), s=(1.45, 0.72, 0.85))
                S("batard", u + jr(0.03), zc + 0.064, 0.0, rz=90 + jr(25), s=(1.3, 0.66, 0.8))
            elif i == 1:                                       # a basket of rolls
                s_ = 1.2
                S("basket", u, zc, 0.05, rz=jr(40), s=s_)
                for q in range(7):
                    a_, r_ = q * 1.047 + 0.3, (0.066 if q < 6 else 0.0) * s_
                    S("roll", u + r_ * cos(a_), zc + (0.082 + (0.03 if q == 6 else jr(0.006))) * s_, 0.05 + r_ * sin(a_), rz=jr(180), rx=jr(22), ry=jr(22),
                      s=rb.u(0.92, 1.1))
            else:                                              # long loaves stacked on a board
                S("board", u, zc, 0.06, rz=jr(5))
                for dv, dz in ((-0.048, 0.0), (0.048, 0.0), (0.0, 0.078)):
                    S("long", u + jr(0.012), zc + 0.018 + dz, 0.06 + dv, rz=jr(5), s=(rb.u(0.82, 0.9), 1.0, 1.0))

        def g_grocer(i, u):
            if i == 0:
                k_ = rb.pick(["r", "g"])
                crate_of(u, 0.055, "apple_" + k_, "heap_" + k_)
            elif i == 1:
                S("pumpkin", u - 0.07, zc, 0.06, rz=jr(180), s=rb.u(0.85, 1.0))
                S("gourd", u + 0.13, zc + 0.066, 0.12, rz=60 + jr(30), rx=90, s=0.88)
                S("cabbage", u + 0.1, zc, -0.09, rz=jr(180), s=rb.u(0.9, 1.05))
                S("cabbage", u - 0.1, zc, -0.16, rz=jr(180), s=rb.u(0.85, 1.0))
            else:
                eggs(u, 0.06, 1.15, ["egg_w", "egg_b", "egg_w"])

        def g_cheese(i, u):
            if i == 0:                                         # wheels stacked, the top one cut
                s0 = rb.u(0.95, 1.05)
                S("wheel", u, zc, 0.04, rz=jr(180), s=s0)
                S("wheel_p", u + jr(0.015), zc + 0.089 * s0, 0.04 + jr(0.015), rz=jr(180), s=0.82)
                S("wheel_cut", u + jr(0.01), zc + 0.089 * (s0 + 0.82), 0.045, rz=-60 + jr(25), s=0.68)
            elif i == 1:                                       # cutting board: wedges, a waxed truckle
                S("board", u, zc, 0.06, rz=jr(6))
                S("wedge", u - 0.14, zc + 0.018, 0.02, rz=20 + jr(15), s=0.95)
                S("wedge", u + 0.02, zc + 0.018, 0.14, rz=215 + jr(15), s=0.8)
                S("wheel_r", u + 0.1, zc + 0.018, 0.02, rz=jr(180), s=0.5)
                S("wheel_cut", u - 0.02, zc, -0.17, rz=-60 + jr(20), s=0.8)
            else:                                              # the scales, little cheeses
                S("scales", u - 0.02, zc, -0.03, rz=jr(12))
                for q, (du, do, dz) in enumerate(((-0.09, 0.13, 0.0), (0.03, 0.14, 0.0), (0.15, 0.12, 0.0), (-0.03, 0.135, 0.04))):
                    S(("wheel_p", "wheel_r", "wheel")[q % 3], u + du, zc + dz, do, rz=jr(180), s=0.44)

        def g_potter(i, u):
            if i == 0:
                S("jug_g", u - 0.1, zc, 0.09, rz=jr(180), s=rb.u(0.95, 1.08))
                S("jug", u + 0.1, zc, 0.06, rz=jr(180), s=0.86)
                S("jug_c", u + 0.01, zc, -0.13, rz=jr(180), s=(0.92, 0.92, 1.32))
            elif i == 1:
                S("bowl", u - 0.08, zc, 0.08, rz=jr(180))
                for q in range(1, 4):
                    S("bowl_rim", u - 0.08 + jr(0.004), zc + 0.021 * q, 0.08 + jr(0.004), rz=jr(180))
                S("plate", u + 0.14, zc, 0.1, rz=jr(180))
                S("plate_c", u + 0.14, zc + 0.012, 0.1, rz=jr(180), s=0.96)
                S("mug", u + 0.1, zc, -0.1, rz=jr(180))
                S("mug_c", u - 0.06, zc, -0.14, rz=jr(180))
            else:
                S("jar", u + 0.05, zc, -0.03, rz=jr(180), s=1.05)
                for q in range(3):                             # flower pots, nested
                    S("flowerpot", u - 0.12, zc + 0.036 * q * 1.1, 0.1, rz=jr(180), s=1.1)
                S("crock", u + 0.14, zc, 0.13, rz=jr(180))

        def g_draper(i, u):
            if i == 0:                                         # bolts of cloth piled with their ends to the customer
                for q, (du_, dz_) in enumerate(((-1, 0), (0, 0), (1, 0), (-0.5, 0.87), (0.5, 0.87), (0, 1.74))):
                    S("bolt%d" % ((q + ki) % 4), u + du_ * 0.1, zc + dz_ * 0.1, -0.03 + jr(0.02), rz=90 + jr(4), s=(rb.u(0.9, 1.05), 1.0, 1.0))
            elif i == 1:                                       # folded lengths
                for q in range(5):
                    S("fold%d" % ((q + ki + 1) % 4), u + jr(0.01), zc + 0.034 * q, 0.06 + jr(0.01), rz=jr(7), s=(1.0 - 0.03 * q, 1.0 - 0.03 * q, 1.0))
            else:                                              # a length thrown over the counter, yarn in a basket
                drape(f, u, zt, 0.3, cloth[ki % 2])
                S("basket", u + 0.02, zc, -0.12, rz=jr(40), s=1.1)
                for q in range(5):
                    a_, r_ = q * 1.57 + 0.4, (0.06 if q < 4 else 0.0) * 1.1
                    S(("ball", "ball_c", "ball_g")[q % 3], u + 0.02 + r_ * cos(a_), zc + (0.07 + (0.045 if q == 4 else 0.0)) * 1.1, -0.12 + r_ * sin(a_), rz=jr(180),
                      rx=jr(40), s=rb.u(0.9, 1.1))

        def back(i, u):
            if trade == "baker":
                if i == 0:
                    S("sack", u, zc, bk + 0.02, rz=jr(180), s=0.8)
                else:
                    S("boule_d" if i % 2 else "boule", u, zc, bk, rz=jr(180), s=0.95)
            elif trade == "grocer":
                S(("pumpkin", "cabbage", "gourd")[i % 3], u, zc, bk, rz=jr(180), s=(0.8, 1.0, 1.05)[i % 3])
            elif trade == "cheese":
                if i == 1:
                    S("crock", u - 0.07, zc, bk, rz=jr(180), s=1.2)
                    S("bottle", u + 0.07, zc, bk, rz=jr(180))
                else:
                    S("wheel", u, zc + 0.132 * 0.9, bk + 0.03, rz=jr(10), rx=78.0, s=0.9)      # a wheel stood on edge
            elif trade == "potter":
                S(("jar", "jug_c", "jug_g")[i % 3], u, zc, bk, rz=jr(180), s=(1.2, (1.0, 1.0, 1.4), 1.15)[i % 3])
            else:
                for q in range(2):                             # bolts stood on end
                    S("bolt%d" % ((q + i + ki) % 4), u + (q - 0.5) * 0.13, zc + 0.176, bk - 0.02, rz=jr(20), ry=-90 + jr(5), s=(0.98, 1.0, 1.0))

        grp = {"baker": g_baker, "grocer": g_grocer, "cheese": g_cheese, "potter": g_potter, "draper": g_draper}[trade]
        for i, u in enumerate(sl):
            if have():
                grp(order[i], u)
            if have(0.85):
                back(order[i], u + jr(0.03))

        def on_shelf(j, i, u, z, o):
            if trade == "baker":
                if (i + j) % 2:
                    S("batard", u, z, o, rz=jr(10), s=(1.15, 0.6, 0.78))
                elif j == 0:
                    S("boule" if i % 4 else "boule_d", u, z, o, rz=jr(180), s=0.76)
                else:
                    S("long", u, z, o, rz=jr(6), s=(0.6, 0.9, 0.9))
            elif trade == "grocer":
                if j == 0:
                    S(("cabbage", "gourd", "pumpkin", "cabbage")[i % 4], u, z, o, rz=jr(180), s=(0.85, 0.8, 0.52, 0.8)[i % 4])
                else:
                    S(("bottle", "crock", "jar", "bottle", "crock")[i % 5], u, z, o, rz=jr(180), s=(1.0, 1.0, 0.6, 0.9, 1.1)[i % 5])
            elif trade == "cheese":
                if j == 0:
                    kd = ("wheel", "wheel_r", "wheel_p")[i % 3]
                    S(kd, u, z, o, rz=jr(180), s=0.56)
                    if rb.chance(0.6):
                        S(("wheel_p", "wheel", "wheel_r")[i % 3], u + jr(0.01), z + 0.089 * 0.56, o, rz=jr(180), s=0.48)
                else:
                    S(("crock", "bottle", "crock", "jar", "bottle")[i % 5], u, z, o, rz=jr(180), s=(1.1, 1.0, 0.9, 0.6, 0.9)[i % 5])
            elif trade == "potter":
                if j == 0:
                    S(("jug", "bowl", "jar", "mug", "jug_c", "flowerpot")[(i + ki) % 6], u, z, o, rz=jr(180), s=(0.8, 0.75, 0.62, 1.0, 0.78, 1.0)[(i + ki) % 6])
                elif i % 2 == 0:                               # plates stood on edge against the wall
                    S("plate" if i % 4 else "plate_c", u, z + 0.112 * sin(radians(76)), o - 0.02, rz=jr(8), rx=76.0)
                else:
                    S(("bottle", "mug_c", "jug_g")[i % 3], u, z, o, rz=jr(180), s=(1.0, 1.0, 0.72)[i % 3])
            else:
                if j == 0 and i % 2:
                    for q in range(3):
                        S("fold%d" % ((q + i + ki) % 4), u, z + 0.034 * q, o, rz=jr(6), s=(0.56, 0.62, 1.0))
                else:
                    S("bolt%d" % ((i + j + ki) % 4), u, z + 0.127, o, rz=jr(25), ry=-90 + jr(5), s=(0.7, 0.8, 0.8))

        nsh = max(2, int(round(w / 0.3)))
        for j in range(2):
            for i in range(nsh):
                if have():
                    on_shelf(j, i, uc + ((i + 0.5) / nsh - 0.5) * (w - 0.14) + jr(0.015), sh_z[j] + 0.016, -d + 0.095 + jr(0.012))

        hung = {"baker": ["ring"], "grocer": ["onion", "herbs", "garlic", "herbs"], "cheese": ["sausage", "ham", "sausage"],
                "potter": ["mug", "mug_c"], "draper": ["len", "skein", "len", "skein_c"]}[trade]

        def hang_one(kind, u, z, o):
            if kind == "len":
                S("len%d" % int(rb.u(0, 3.99)), u, z, o, rz=jr(8), s=(rb.u(0.8, 1.1), 1.0, rb.u(0.75, 1.0)))
            elif kind in ("mug", "mug_c"):                     # mugs hung by their handles
                S(kind, u, z - 0.085, o, rz=jr(25), ry=-90.0 + jr(8))
            elif kind == "ham":
                S(kind, u, z, o, rz=jr(30), s=(1.0, 0.62, 1.0))
            elif kind == "ring":
                S(kind, u, z, o, rz=jr(22), s=rb.u(0.88, 1.08))
            else:
                S(kind, u, z, o, rz=jr(180), s=rb.u(0.88, 1.06))

        nh = max(2, int(round((w - 0.2) / 0.24)))
        for i in range(nh):                                    # strung from the hook rail inside the hatch
            uh = uc + ((i + 0.5) / nh - 0.5) * (w - 0.2) + jr(0.02)
            if have() and (lamp_u is None or abs(uh - lamp_u) > 0.17):
                hang_one(hung[(i + ki) % len(hung)], uh, z1 - 0.082, -0.07)
        if rod is not None:                                    # ... and from the front rod of the awning
            for i, t_ in enumerate((-0.37, -0.15, 0.15, 0.37)):
                if have(0.75) and trade != "potter":
                    hang_one([h for h in hung if h != "ham"][(i + ki) % len([h for h in hung if h != "ham"])], uc + t_ * rod[1] + jr(0.02), rod[0], AWP)

    def drape(f, u, z, wd, m_, drop=0.34, back=None):
        """Length of cloth laid over the counter (from `back`, default 0.2 m behind its edge) and hanging down its front."""
        nsg = max(3, int(round(wd / 0.09)))
        prof = ((z + 0.005, CB - 0.2 if back is None else back), (z + 0.007, CB + 0.004), (z - 0.03, CB + 0.016), (z - drop * 0.55, CB + 0.012),
                (z - drop, CB + 0.016))
        P = [[f.pt(u + (0.5 - i / nsg) * wd, zz + 0.004 * sin(i * 2.4 + j), oo + (0.011 * sin(i * 2.1 + 0.8 * j) if j > 1 else 0.0))
              for j, (zz, oo) in enumerate(prof)] for i in range(nsg + 1)]
        o = _sheet("drape", P)
        uv_planar(o, (f.ux, f.uy, 0), (0, 0, 1))
        ware(o, m_, 40)

    def trade_sign(f, u, z, m, emblem):
        """Small painted board hanging from a forged arm, square to the wall (it faces whoever walks the passage)."""
        o_w, reach = wo_at(z), AWP
        for o in (fbox("ts_plate", f, u, z - 0.05, 0.034, 0.2, thick=0.006, proud=o_w + 0.006, bev=0.002),
                  box("ts_arm", (0.013, reach, 0.013), loc=f.pt(u, z, o_w + reach / 2), rot=(0, 0, f.th), bevel=0.0, segments=1),
                  box("ts_stay", (0.01, 0.215, 0.01), loc=f.pt(u, z - 0.07, o_w + 0.082), rot=(-40.5, 0, f.th), bevel=0.0, segments=1),
                  box("ts_link", (0.004, 0.004, 0.05), loc=f.pt(u, z - 0.03, o_w + 0.07), bevel=0.0, segments=1),
                  box("ts_link", (0.004, 0.004, 0.05), loc=f.pt(u, z - 0.03, o_w + 0.22), bevel=0.0, segments=1)):
            assign(o, iron)
            iron_objs.append(o)
        zb_ = z - 0.055 - 0.105
        painted(box("ts_board", (0.02, 0.2, 0.21), loc=f.pt(u, zb_, o_w + 0.145), rot=(0, 0, f.th), bevel=0.003, segments=1), "Z", m=m)
        for sg in (-1, 1):
            c_ = f.pt(u + sg * 0.012, zb_, o_w + 0.145)
            if emblem == 0:
                e = [cylinder("ts_emblem", 0.05, 0.006, loc=c_, rot=(0, 90, f.th), verts=10)]
            elif emblem == 1:
                e = [box("ts_emblem", (0.006, 0.08, 0.08), loc=c_, rot=(45, 0, f.th), bevel=0.0, segments=1)]
            else:
                e = [box("ts_emblem", (0.006, 0.13, 0.028), loc=f.pt(u + sg * 0.012, zb_ + dz, o_w + 0.145), rot=(0, 0, f.th), bevel=0.0, segments=1)
                     for dz in (-0.05, 0.0, 0.05)]
            # CLEARANCE RULE: no glowing face buried in the board (the emblems' backs lay 1 mm under its paint)
            n_ = (Vector(c_) - Vector(f.pt(u, zb_, o_w + 0.145))).normalized()
            for o in e:
                _front_only(o, n_)
                assign(o, brass_l)
                iron_objs.append(o)

    def hook(f, u, z, out):
        for (su_, sz_, th_, dz) in ((0.012, 0.012, 0.05, 0.0), (0.012, 0.034, 0.012, 0.012)):
            h_ = fbox("hook", f, u, z + dz, su_, sz_, thick=th_, proud=out + 0.05, bev=0.002)
            assign(h_, iron)
            iron_objs.append(h_)

    def tie_ring(f, u, z, out):
        pl_ = fbox("ring_plate", f, u, z, 0.06, 0.06, thick=0.008, proud=out + 0.008, bev=0.002)
        bo_ = bolt("ring_staple", 0.012, 0.014, loc=f.pt(u, z, out + 0.008), rot=(90, 0, f.th), kind="dome")
        rg_ = torus("tie_ring", 0.042, 0.0065, loc=f.pt(u, z - 0.04, out + 0.02), rot=(90 + 8, 0, f.th), major_seg=12, minor_seg=5)
        for o in (pl_, bo_, rg_):
            assign(o, iron)
            iron_objs.append(o)

    def pot(f, u, z, out, kind=0, s=1.0):
        stamp(("flowerpot", "jug", "crock")[kind], gm(f, u, z, out, rz=rb.u(-180, 180), s=s))

    def dress(f, u0, u1, z0, z1, d, nblk, lint=0.2):
        """Dressed surround of a wall opening: long-and-short jamb blocks, lintel, stone reveals."""
        hb = (z1 - z0) / nblk
        for sg, ue in ((-1, u0), (1, u1)):
            for i in range(nblk):
                wj = (0.17 if i % 2 == 0 else 0.11) * rb.vary(1.0, 0.07)
                bay_st.append(fbox("bay_jamb", f, ue + sg * wj / 2, z0 + hb * (i + 0.5), wj, hb - 0.004, thick=0.08,
                                   proud=0.022 + rb.u(-0.004, 0.004), bev=0.01))
        bay_st.append(fbox("bay_lintel", f, (u0 + u1) / 2, z1 + lint / 2, u1 - u0 + 0.3, lint, thick=0.1, proud=0.034, bev=0.014))
        rv = fquads("bay_reveal", f, [[(u0, z0, 0), (u0, z0, -d), (u0, z1, -d), (u0, z1, 0)],
                                      [(u1, z0, -d), (u1, z0, 0), (u1, z1, 0), (u1, z1, -d)],
                                      [(u0, z1, -d), (u1, z1, -d), (u1, z1, 0), (u0, z1, 0)]])
        uv_box(rv)
        bay_st.append(rv)

    shop_paints = [shutter_m]
    cur_paint = [shutter_m]
    booth_m = oak
    blind_m = plaster

    def painted(o, grain="Z", m=None):
        uv_grain(o, grain)
        assign(o, m or cur_paint[0])
        shutter_objs0.append(o)

    def oaken(o, grain="Z"):
        uv_grain(o, grain)
        assign(o, oak)
        timber_objs.append(o)

    for (sx, k), kind in sorted(bay_kind.items()):
        f = IW[sx]
        lo, hi = spans[(sx, k)]
        uc, fw = -sx * (lo + hi) / 2, hi - lo
        is_inner = 0 < k < nb - 1
        awn = bool(p.awnings) and is_inner and fw >= 0.95
        lamp_ok = (sx, k) not in no_lamp
        if kind in ("shop", "shut"):
            cur_paint[0] = shop_paints[(k + (0 if sx < 0 else 1)) % len(shop_paints)]
            w = min(1.3, fw - 0.16)
            u0, u1, z0, z1 = uc - w / 2, uc + w / 2, 0.84, zh
            d = 0.15 if kind == "shop" else 0.12
            hgt = z1 - z0
            holes[sx].append((u0, u1, z0, z1))
            in_blocks[sx].append((u0 - 0.1, u1 + 0.1, z0 - 0.08, z1 + 0.18))
            dress(f, u0, u1, z0, z1, d, 3)
            bay_st.append(fbox("bay_sill", f, uc, z0 - 0.045, w + 0.22, 0.09, thick=d + 0.055, proud=0.055, bev=0.012))
            if kind == "shop":
                pf = -0.085                               # front of the joinery
                nm = 2 if w > 1.05 else 1
                zt = z0 + 0.07 + (hgt - 0.13) * 0.66
                painted(fbox("shop_frame", f, u0 + 0.0275, (z0 + z1) / 2, 0.055, hgt, thick=0.05, proud=pf, bev=0.005))
                painted(fbox("shop_frame", f, u1 - 0.0275, (z0 + z1) / 2, 0.055, hgt, thick=0.05, proud=pf, bev=0.005))
                painted(fbox("shop_frame", f, uc, z1 - 0.03, w - 0.11, 0.06, thick=0.05, proud=pf, bev=0.005), "X")
                painted(fbox("shop_frame", f, uc, z0 + 0.035, w - 0.11, 0.07, thick=0.05, proud=pf + 0.008, bev=0.005), "X")
                painted(fbox("shop_transom", f, uc, zt, w - 0.11, 0.045, thick=0.05, proud=pf + 0.004, bev=0.005), "X")
                for i in range(1, nm + 1):
                    um = u0 + w * i / (nm + 1)
                    painted(fbox("shop_mullion", f, um, (z0 + z1) / 2, 0.045, hgt - 0.12, thick=0.045, proud=pf - 0.002, bev=0.005))
                for i in range(nm + 1):                   # one glazing bar in each lower light, two in each top light
                    ua_, ub_ = u0 + w * i / (nm + 1), u0 + w * (i + 1) / (nm + 1)
                    painted(fbox("shop_bar", f, (ua_ + ub_) / 2, z0 + 0.07 + (zt - z0 - 0.07) * 0.5, ub_ - ua_ - 0.04, 0.02, thick=0.025,
                                 proud=pf - 0.012, bev=0.003), "X")
                    for t_ in (1 / 3, 2 / 3):
                        painted(fbox("shop_bar", f, lerp(ua_, ub_, t_), (zt + z1) / 2, 0.02, z1 - zt - 0.05, thick=0.025, proud=pf - 0.012,
                                     bev=0.003))
                pane = fquads("shop_pane", f, [[(u0 + 0.03, z0 + 0.03, pf - 0.038), (u1 - 0.03, z0 + 0.03, pf - 0.038),
                                                (u1 - 0.03, z1 - 0.03, pf - 0.038), (u0 + 0.03, z1 - 0.03, pf - 0.038)]])
                lit = shop_lit[(sx, k)]
                assign(pane, lit_m if lit else dark_m)
                uv_planar(pane, (f.ux, f.uy, 0), (0, 0, 1), offset=(rb.u(0.0, 2.9), 0.45))
                paint(pane, grime=0.25 + 0.25 * age, special=rb.u(0.03, 0.97))
                if not lit:
                    # shut for the night: a linen blind drawn most of the way down behind the lower lights
                    zb_ = lerp(z0 + 0.09, zt - 0.03, rb.u(0.0, 0.4))
                    bl = fquads("shop_blind", f, [[(u0 + 0.05, zb_, pf - 0.034), (u1 - 0.05, zb_, pf - 0.034),
                                                   (u1 - 0.05, zt - 0.02, pf - 0.034), (u0 + 0.05, zt - 0.02, pf - 0.034)]])
                    assign(bl, blind_m)
                    uv_planar(bl, (f.ux, f.uy, 0), (0, 0, 1))
                    wares.append(bl)
                    painted(fbox("shop_blind_lath", f, uc, zb_ + 0.012, w - 0.12, 0.024, thick=0.008, proud=pf - 0.03, bev=0.0), "X", m=booth_m)
                npot, ub0 = rb.pick([0, 2, 3]), uc + rb.u(-0.25, 0.25) * (w - 0.5)
                for i in range(npot):                     # pots and jugs on the sill
                    pot(f, ub0 + (i - (npot - 1) / 2) * 0.15, z0, -0.012 + rb.u(-0.008, 0.008), kind=rb.pick([0, 0, 1, 2]), s=rb.u(0.85, 1.08))
                if awn and lit:
                    aw, zf_ = min(w + 0.3, fw + 0.12), max(z1 + 0.07, 2.0)
                    awning(f, uc, aw, zf_ + 0.42, zf_, AWP, awn_1)
                    stall_lamp(f.pt(uc + rb.pick([-1, 1]) * (aw / 2 - 0.17), zf_ - 0.018, AWP - 0.07), under=True)
                else:
                    if lamp_ok:
                        bracket_lamp(f, uc + rb.pick([-1, 1]) * w * 0.33, z1 + 0.1, 0.034)
                    if is_inner and not lit:
                        trade_sign(f, uc, max(z1 + 0.38, 2.3), cur_paint[0], (k + (sx > 0)) % 3)
            else:
                # closed booth: plank shutter in two leaves, ledges, strap hinges, hasp and padlock
                pf = -d + 0.04
                nbd = max(4, int(round(w / 0.15)))
                bwd = w / nbd
                for i in range(nbd):
                    painted(fbox("booth_board", f, u0 + bwd * (i + 0.5), (z0 + z1) / 2, bwd - 0.006, hgt - 0.012, thick=0.03,
                                 proud=pf + rb.u(-0.004, 0.003), bev=0.004), m=booth_m if (k + sx) % 3 else None)
                for zz in (z0 + 0.17, z1 - 0.17):
                    for sg in (-1, 1):
                        painted(fbox("booth_ledge", f, uc + sg * w * 0.25, zz, w / 2 - 0.05, 0.075, thick=0.022, proud=pf + 0.02, bev=0.004), "X")
                        hg = fbox("booth_hinge", f, uc + sg * (w / 2 - w * 0.17), zz, w * 0.34, 0.03, thick=0.006, proud=pf + 0.026, bev=0.0015)
                        assign(hg, iron)
                        iron_objs.append(hg)
                for (su_, sz_, th_, dz, du) in ((0.16, 0.03, 0.006, 0.0, 0.0), (0.045, 0.06, 0.02, -0.045, 0.03)):
                    hp = fbox("booth_hasp", f, uc + du, (z0 + z1) / 2 + dz, su_, sz_, thick=th_, proud=pf + 0.006 + th_, bev=0.002)
                    assign(hp, iron)
                    iron_objs.append(hp)
                bk = fquads("booth_back", f, [[(u0, z0, -d), (u1, z0, -d), (u1, z1, -d), (u0, z1, -d)]])
                assign(bk, dark_m)
                if lamp_ok:
                    bracket_lamp(f, uc + rb.pick([-1, 1]) * w * 0.33, z1 + 0.1, 0.034)
                if is_inner:
                    trade_sign(f, uc, max(z1 + 0.38, 2.3), cur_paint[0], (k + (sx > 0)) % 3)
        elif kind == "niche":
            # market stall: an open hatch with a drop-leaf counter on struts, stocked shelves in front of the lit room behind
            w = min(1.5, fw - 0.16)
            u0, u1, z0, z1, d = uc - w / 2, uc + w / 2, 0.88, zh, STALL_D
            holes[sx].append((u0, u1, z0, z1))
            in_blocks[sx].append((u0 - 0.1, u1 + 0.1, z0 - 0.1, z1 + 0.18))
            dress(f, u0, u1, z0, z1, d, 3)
            bay_st.append(fbox("bay_counter", f, uc, z0 - 0.055, w + 0.24, 0.11, thick=d + 0.06, proud=0.06, bev=0.014))
            # limewashed inside: the stall's lamp (a real light, see below) and the lanterns show the stock on the shelves.
            # NB emissive / window_lit surfaces go dark at the raking angle the passage is seen from, so no glowing back
            bay_plaster.append(fquads("niche_back", f, grid_q(u0, u1, z0, z1, -d, cell=0.3)))
            zt = z0 + 0.04                                 # top of the counter board
            oaken(fbox("stall_board", f, uc, z0 + 0.02, w - 0.02, 0.04, thick=CB + d - 0.012, proud=CB, bev=0.006), "X")
            for sg in (-1, 1):
                dz_, do_ = 0.4, CB - 0.07
                oaken(box("stall_strut", (0.04, hypot(dz_, do_), 0.04), loc=f.pt(uc + sg * (w / 2 - 0.16), z0 - 0.005 - dz_ / 2, 0.04 + do_ / 2),
                          rot=(-degrees(atan2(dz_, do_)), 0, f.th), bevel=0.004, segments=1), "Y")
            sh_z = (z0 + 0.5, z0 + 0.86)
            for zsh in sh_z:
                oaken(fbox("niche_shelf", f, uc, zsh, w - 0.012, 0.03, thick=0.17, proud=-d + 0.17, bev=0.004), "X")
                for sg in (-1, 1):
                    br = fbox("shelf_bracket", f, uc + sg * w * 0.3, zsh - 0.06, 0.016, 0.09, thick=0.13, proud=-d + 0.13, bev=0.003)
                    assign(br, iron)
                    iron_objs.append(br)
            rail = box("hook_rail", (w + 0.02, 0.012, 0.012), loc=f.pt(uc, z1 - 0.075, -0.07), rot=(0, 0, f.th), bevel=0.0, segments=1)
            assign(rail, iron)
            iron_objs.append(rail)
            ki = n_stall[0]
            n_stall[0] += 1
            trade = p.trades if p.trades != "mixed" else TRADES[(t0_ + ki) % len(TRADES)]
            sgl = rb.pick([-1, 1])
            rod, lamp_u = None, None
            if awn:
                aw, zf_ = min(w + 0.3, fw + 0.12), max(z1 + 0.07, 2.0)
                awning(f, uc, aw, zf_ + 0.42, zf_, AWP, awn_1)
                rod = (zf_ - 0.018, aw)
            elif is_inner:
                trade_sign(f, uc, max(z1 + 0.38, 2.3), shutter_m, TRADES.index(trade) % 3)
            if p.stalls_lit:                               # the stall's own lamp hangs on the hook rail inside the hatch, among the goods
                lamp_u = uc + sgl * w * rb.u(0.16, 0.24)
                stall_lamp(f.pt(lamp_u, z1 - 0.08, -0.07), hang=0.045, s=1.5, stall=True)
            stock_stall(f, uc, w, zt, z1, d, sh_z, trade, ki, rod, lamp_u)
        elif kind == "seat":
            w = min(1.5, fw - 0.16)
            u0, u1, z1, d, zt = uc - w / 2, uc + w / 2, zh, 0.36, 0.47
            holes[sx].append((u0, u1, 0.0, z1))
            in_blocks[sx].append((u0 - 0.1, u1 + 0.1, -1.0, z1 + 0.18))
            pl_gap[sx].append((u0 - 0.2, u1 + 0.2))
            dress(f, u0, u1, 0.0, z1, d, 5)
            bay_st.append(fbox("bench_seat", f, uc, zt - 0.05, w - 0.004, 0.1, thick=d + 0.04, proud=0.04, bev=0.016))
            nrs = 3
            for i in range(nrs):
                bay_q.append(fbox("bench_riser", f, u0 + w * (i + 0.5) / nrs, (zt - 0.1) / 2, w / nrs - 0.005, zt - 0.1, thick=0.12,
                                  proud=-0.02 + rb.u(-0.006, 0.006), bev=0.012))
            bk = fquads("seat_back", f, grid_q(u0, u1, zt, z1, -d))
            assign(bk, st_back)
            uv_box(bk)
            wall_objs.append(bk)
            seat_backs.append((f, u0, u1, zt, z1, -d))
            tie_ring(f, uc + rb.u(-0.2, 0.2) * w, zt + 0.72, -d + WO)
            if lamp_ok:
                stall_lamp(f.pt(uc + rb.u(-0.25, 0.25) * w, z1, -d * 0.45), hang=0.08)
        elif kind == "door":
            w = 0.84
            u0, u1, z1, d, stp_ = uc - w / 2, uc + w / 2, zd, 0.2, 0.12
            holes[sx].append((u0, u1, 0.0, z1))
            in_blocks[sx].append((u0 - 0.1, u1 + 0.1, -1.0, z1 + 0.2))
            pl_gap[sx].append((u0 - 0.2, u1 + 0.2))
            dress(f, u0, u1, 0.0, z1, d, 5, lint=0.22)
            bay_st.append(fbox("bay_step", f, uc, stp_ / 2, w - 0.004, stp_, thick=d + 0.012, proud=0.012, bev=0.012))
            pf = -d + 0.055
            for i in range(5):
                oaken(fbox("bay_door_board", f, u0 + w * (i + 0.5) / 5, (stp_ + z1) / 2, w / 5 - 0.005, z1 - stp_ - 0.012, thick=0.035,
                           proud=pf + rb.u(-0.004, 0.003), bev=0.004))
            for zz in (stp_ + 0.32, z1 - 0.34):
                sp_ = fbox("bay_door_strap", f, uc - w * 0.1, zz, w * 0.76, 0.045, thick=0.008, proud=pf + 0.008, bev=0.002)
                assign(sp_, iron)
                iron_objs.append(sp_)
            tie_ring(f, uc + w * 0.3, stp_ + 0.98, pf)
            bk = fquads("bay_door_back", f, [[(u0, 0.0, -d), (u1, 0.0, -d), (u1, z1, -d), (u0, z1, -d)]])
            assign(bk, dark_m)
            if lamp_ok:
                bracket_lamp(f, uc, z1 + 0.12, 0.034)
        else:
            tie_ring(f, uc + rb.u(-0.2, 0.2) * fw, 1.08 + rb.u(-0.05, 0.05), WO)
            if fw >= 0.95:
                zr_ = min(zh - 0.06, 1.95)
                rl_ = min(1.0, fw - 0.2)
                rail = fbox("hook_rail", f, uc, zr_, rl_, 0.026, thick=0.01 + WO, proud=0.012 + WO, bev=0.002)
                assign(rail, iron)
                iron_objs.append(rail)
                for i in range(4):
                    hook(f, uc + (i - 1.5) * rl_ * 0.27, zr_, 0.012 + WO)
            if lamp_ok and is_inner:
                bracket_lamp(f, uc, min(zh + 0.14, 2.14), WO)
            socket("bay_wall_%s%d" % ("l" if sx < 0 else "r", k), "attach", f.pt(uc, 1.45, WO), rot=(0, 0, f.th), role="poster")
    # plinth course along both walls, broken at the doorways and seat niches
    rpi = rng.fork("in_plinth")
    for sx in (-1, 1):
        f = IW[sx]
        edges = [-D / 2 + 0.48] + [e for g_ in sorted(pl_gap[sx]) for e in g_] + [D / 2 - 0.48]
        for a, b in zip(edges[0::2], edges[1::2]):
            in_blocks[sx].append((a - 0.021, b + 0.021, -1.0, 0.385))
            u = a
            while u < b - 0.12:
                ln = min(rpi.u(0.7, 1.2), b - u)
                if b - (u + ln) < 0.35:
                    ln = b - u
                hh = 0.4 + rpi.u(-0.01, 0.01)
                bay_q.append(fbox("in_plinth", f, u + ln / 2, hh / 2, ln - 0.004, hh, thick=0.12, proud=0.045 + rpi.u(-0.005, 0.005), bev=0.016))
                u += ln
        wall_objs.append(holed_wall("passage_wall_l" if sx < 0 else "passage_wall_r", f, ya, yb, zwall, holes[sx]))
    if shutter_objs0:
        shutter_objs0[:] = [join(shutter_objs0, "bay_joinery")]
    for o in bay_st:
        if not o.data.materials:
            assign(o, st_d)
    for o in bay_q:
        assign(o, st_q)
    stone_objs.append(join(bay_st + bay_q, "bay_stone"))
    if bay_plaster:
        npl = join(bay_plaster, "niche_backs")
        assign(npl, plaster)
        uv_box(npl)
        paint(npl, grime=mask_mul(noise_mask(0.7, seed=sd + 31, lo=0.3, hi=0.85), 0.25 + 0.35 * age))
    goods = []
    for kind in sorted(G_INST):
        base = MAKE[kind]()
        if base.get("tsq_smooth") == "keep":
            base["_tsq_smooth_angle"] = 75.0
        goods.append(instances_to_mesh("goods_" + kind, base, G_INST[kind], seed=sd + 70 + len(goods), keep_base=False))
    if goods:
        join(goods, "goods")
    if lamp_objs:
        join(lamp_objs, "stall_lamps")
    if wares:
        wj_ = join(wares, "wares")
        paint(wj_, grime=mask_mul(noise_mask(0.3, seed=sd + 32, lo=0.3, hi=0.8), 0.3 + 0.3 * age))

    def rib(name, r, y0, y1, depth=0.17, cham=0.055, stone_len=0.42, gap=0.005, m=2, prof=None, yfn=None, pieces=None, grain=False):
        """Rib along the arch built of separate pieces (stones / lengths of timber). prof = section [(y, q)], q measured from
        the clear arch line outwards, first and last point inside the vault (default: a chamfered rib whose chamfers face the
        mouths and catch the lantern light); yfn(s) shifts the section along the passage (diagonal ribs)."""
        n = pieces or max(3, int(round(S0 / stone_len)))
        if n % 2 == 0 and not pieces:
            n += 1
        cs = [S0 * i / n + (r.u(-0.02, 0.02) if 0 < i < n else 0.0) for i in range(n + 1)]
        prof = prof or [(y0, depth), (y0, cham * 0.8), (y0 + cham, 0.0), (y1 - cham, 0.0), (y1, cham * 0.8), (y1, depth)]
        npf = len(prof)
        pl = [0.0]
        for (pa, qa), (pb, qb) in zip(prof, prof[1:]):
            pl.append(pl[-1] + hypot(pb - pa, qb - qa))
        verts, faces, uvs = [], [], []
        for i in range(n):
            base = len(verts)
            for k_ in range(m + 1):
                s = cs[i] + gap + (cs[i + 1] - cs[i] - 2 * gap) * k_ / m
                x, z, nx, nz = at0(s)
                dy = yfn(s) if yfn else 0.0
                verts += [(x + nx * q, y + dy, z + nz * q) for (y, q) in prof]
                uvs += [(s + 0.37 * i, t) for t in pl]
            for k_ in range(m):
                for j in range(npf - 1):                   # the back (inside the vault) stays open
                    a, b = base + k_ * npf + j, base + k_ * npf + j + 1
                    faces.append((a, b, b + npf, a + npf))
            faces.append(tuple(base + j for j in reversed(range(npf))))
            faces.append(tuple(base + m * npf + j for j in range(npf)))
        o = mesh_object(name, verts, faces)
        me = o.data
        if grain and len(me.vertices) == len(uvs):         # grain along the curve
            lv = np.zeros(len(me.loops), dtype=np.int32)
            me.loops.foreach_get("vertex_index", lv)
            me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.asarray(uvs, dtype=np.float32)[lv].ravel())
            o["tsq_uv"] = "keep"
        smooth(o, 30)
        return o

    hang = []                                         # (y, z of the hook)
    corbel_blocks = []
    n_lan = int(p.lanterns)

    def pick_spots(spots, zhook):
        """The hanging lanterns spread evenly down the passage, each on the nearest free spot (bay crown / beam gap)."""
        for i in range(n_lan):
            wy = -D / 2 + D * (i + 0.5) / n_lan
            free = [b for b in spots if all(abs(b - h[0]) > 1e-3 for h in hang)]
            if free:
                hang.append((min(free, key=lambda b: (round(abs(b - wy), 3), abs(b), b)), zhook))

    vault, vault_tim = None, []
    timber_v = p.ceiling == "vault" and p.vault == "timber"
    if p.ceiling == "vault":
        rv_ = rng.fork("vault_web")

        def web(name, m_, ln, jit, shrink, tone, sub=0.55):
            """The vault between the ribs laid in strips along the passage, one to each arch segment, every strip cut into
            lengths (boards / coursed stones) that sit a few millimetres proud of or behind their neighbours, each of its
            own tone.  A dark backing closes the joints."""
            V, Fc, Gr, Wr = [], [], [], []
            for a in range(NA):
                (xa, za), (xb, zb_) = vpts[a], vpts[a + 1]
                nx_, nz_ = anr[a][0] + anr[a + 1][0], anr[a][1] + anr[a + 1][1]
                l_ = hypot(nx_, nz_) or 1.0
                nx_, nz_ = nx_ / l_, nz_ / l_
                tx, tz = xb - xa, zb_ - za
                tl = hypot(tx, tz) or 1.0
                tx, tz = tx / tl * shrink, tz / tl * shrink
                y = ya
                y1 = ya + rv_.u(0.3, 1.0) * ln[0]
                while y < yb - 1e-4:
                    if yb - y1 < 0.5 * ln[0]:
                        y1 = yb
                    dq = rv_.u(-jit, jit)
                    g_, w_ = tone(rv_)
                    ns = max(1, int(ceil((y1 - y) / sub)))
                    b = len(V)
                    for j in range(ns + 1):
                        yy = lerp(y + shrink, y1 - shrink, j / ns)
                        V += [(xa + tx + nx_ * dq, yy, za + tz + nz_ * dq), (xb - tx + nx_ * dq, yy, zb_ - tz + nz_ * dq)]
                        Gr += [g_, g_]
                        Wr += [w_, w_]
                    Fc += [(b + 2 * j, b + 2 * j + 2, b + 2 * j + 3, b + 2 * j + 1) for j in range(ns)]
                    y, y1 = y1, y1 + rv_.u(*ln)
            o = mesh_object(name, V, Fc)
            flat(o)
            assign(o, m_)
            if len(o.data.vertices) == len(Gr):
                ga, wa = np.array(Gr, dtype=np.float32), np.array(Wr, dtype=np.float32)
                paint(o, grime=lambda co, nn, a_=ga: a_, wear=lambda co, nn, a_=wa: a_)
            bk = surface("vault_backing", lambda u, v: (vpts[int(round(v * NA))][0] + anr[int(round(v * NA))][0] * (jit + 0.006), lerp(ya, yb, u),
                                                        vpts[int(round(v * NA))][1] + anr[int(round(v * NA))][1] * (jit + 0.006)), 2, NA)
            assign(bk, m_)
            paint(bk, grime=1.0)
            return o, bk

        yks = [-D / 2 + D * k / nb for k in range(1, nb)]
        for yk in yks:                                    # stone springers of the ribs
            for sx in (-1, 1):
                prof = [(sx * (jx + 0.02), zs), (sx * (jx - 0.118), zs), (sx * (jx - 0.118), zs - 0.07), (sx * (jx - 0.085), zs - 0.13),
                        (sx * (jx - 0.065), zs - 0.22), (sx * (jx + 0.02), zs - 0.38)]
                pas_stone.append(_prism("corbel", prof, yk - 0.19, yk + 0.19, bev=0.012))
            corbel_blocks.append((yk - 0.19, yk + 0.19, zs - 0.34, zs + 0.2))
        if timber_v:
            # wagon roof: moulded arch ribs in three lengths on the stone springers, a wall rib against either portal, butt
            # purlins and a ridge piece tenoned between them, wall plates on the wall heads, boards laid lengthwise above
            vault, vbk = web("vault_boards", oak, (1.5, 2.6), 0.004, 0.0025,
                             lambda r: (r.u(0.0, 0.5), r.pick([0.0, 0.0, r.u(0.1, 0.45)])))
            uv_grain(vault, "Y")
            uv_grain(vbk, "Y")
            RW, WR = 0.105, 0.09
            for yk in yks:
                y0, y1 = yk - RW, yk + RW
                vault_tim.append(rib("vault_rib", rr_, y0, y1, pieces=3, m=NA // 3, gap=0.0025, grain=True,
                                     prof=[(y0, 0.135), (y0, 0.058), (y0 + 0.03, 0.042), (y0 + 0.052, 0.0), (y1 - 0.052, 0.0), (y1 - 0.03, 0.042),
                                           (y1, 0.058), (y1, 0.135)]))
                ps_ = clamp((zc - 2.84) / 0.112, 0.0, 1.0)
                if ps_ > 0.4:
                    vault_tim.append(lathe("vault_pendant", [(r_ * ps_, z_ * ps_) for r_, z_ in ((0.0, -0.112), (0.022, -0.103), (0.035, -0.083),
                                                             (0.023, -0.061), (0.031, -0.049), (0.063, -0.03), (0.076, -0.006), (0.076, 0.02))],
                                           segments=10, loc=(0, yk, zc), smooth_angle=50.0))
            for y0, y1, sg in ((ya, ya + WR, 1), (yb - WR, yb, -1)):
                pr_ = [(0.0, 0.135), (0.0, 0.0), (0.04, 0.0), (0.062, 0.04), (WR, 0.055), (WR, 0.135)]
                vault_tim.append(rib("vault_wall_rib", rr_, y0, y1, pieces=3, m=NA // 3, gap=0.0025, grain=True,
                                     prof=[(y0 + t_, q_) for t_, q_ in pr_] if sg > 0 else [(y1 - t_, q_) for t_, q_ in reversed(pr_)]))
            segs = list(zip([ya + WR] + [yk + RW for yk in yks], [yk - RW for yk in yks] + [yb - WR]))
            for fr, pw_, pd_ in ((0.2, 0.1, 0.085), (0.36, 0.1, 0.085), (0.5, 0.135, 0.11), (0.64, 0.1, 0.085), (0.8, 0.1, 0.085)):
                x, z, nx, nz = at0(S0 * fr)
                qc = REC + 0.012 - pd_ / 2
                for y0, y1 in segs:
                    o = box("vault_purlin", (pw_ * rr_.vary(1, 0.05), y1 - y0 + 0.012, pd_), loc=(x + nx * qc, (y0 + y1) / 2, z + nz * qc),
                            rot=(0, degrees(atan2(nx, nz)), 0), bevel=0.012, segments=1)
                    uv_grain(o, "Y")
                    vault_tim.append(o)
            for sx in (-1, 1):
                prof = [(sx * (jx + 0.01), zs - 0.012), (sx * (jx - 0.034), zs - 0.012), (sx * (jx - 0.042), zs - 0.002), (sx * (jx - 0.042), zs + 0.07),
                        (sx * (jx - 0.02), zs + 0.1), (sx * (jx + 0.01), zs + 0.1)]
                for y0, y1 in segs:
                    o = _prism("vault_plate", prof, y0, y1, bev=0.0)
                    uv_grain(o, "Y")
                    vault_tim.append(o)
            for o in vault_tim:
                assign(o, oak)
            vault_tim = [join(vault_tim, "vault_carpentry")]
            z_hk = zc + REC + 0.012 - 0.11                # under the ridge piece
        else:
            # net vault: moulded transverse ribs of separate stones, two diagonal ribs to every bay meeting in a carved boss
            # at the crown, a ridge rib from boss to boss; the webs laid in courses (limewashed stones / brick)
            brick_w = p.soffit != "limewash"
            vault, vbk = web("vault_web", vault_m, (0.5, 1.1) if brick_w else (0.3, 0.62), 0.002 if brick_w else 0.004, 0.0,
                             (lambda r: (r.u(0.0, 0.3), r.u(0.0, 0.25))) if brick_w else (lambda r: (r.u(0.0, 0.38), r.pick([0.0, r.u(0.0, 0.7)]))))
            uv_box(vault)
            uv_box(vbk)
            RW = 0.16
            for yk in yks:
                y0, y1 = yk - RW, yk + RW
                pas_stone.append(rib("rib", rr_, y0, y1, prof=[(y0, 0.17), (y0, 0.078), (y0 + 0.03, 0.064), (y0 + 0.052, 0.03), (y0 + 0.088, 0.0),
                                                               (y1 - 0.088, 0.0), (y1 - 0.052, 0.03), (y1 - 0.03, 0.064), (y1, 0.078), (y1, 0.17)]))
            edges = list(zip([ya] + [yk + RW for yk in yks], [yk - RW for yk in yks] + [yb]))
            dprof = [(-0.058, 0.15), (-0.058, 0.08), (-0.03, 0.05), (0.03, 0.05), (0.058, 0.08), (0.058, 0.15)]
            for bc, (y0, y1) in zip(bay_c, edges):
                span = 2 * min(bc - y0, y1 - bc) - 0.12
                for sg in (-1, 1):
                    pas_stone.append(rib("diagonal_rib", rr_, 0, 0, prof=dprof, stone_len=0.5, m=2, gap=0.004,
                                         yfn=lambda s, bc=bc, sg=sg, span=span: bc + sg * (s / S0 - 0.5) * span))
                pas_stone.append(lathe("vault_boss", [(0.0, 0.012), (0.05, 0.02), (0.088, 0.036), (0.102, 0.056), (0.088, 0.076), (0.112, 0.092),
                                                      (0.112, 0.135)], segments=12, loc=(0, bc, zc), smooth_angle=50.0))
                for ys0, ys1 in ((y0, bc - 0.1), (bc + 0.1, y1)):
                    npc = max(1, int(round((ys1 - ys0) / 0.45)))
                    for i in range(npc):
                        pas_stone.append(_prism("ridge_rib", [(-0.055, zc + 0.14), (-0.055, zc + 0.08), (-0.028, zc + 0.052), (0.028, zc + 0.052),
                                                              (0.055, zc + 0.08), (0.055, zc + 0.14)],
                                                lerp(ys0, ys1, i / npc) + 0.003, lerp(ys0, ys1, (i + 1) / npc) - 0.003, bev=0.0))
            z_hk = zc + 0.014                             # under the bosses
        pick_spots(bay_c, z_hk)
        for b in bay_c:                                   # spare hooks in the crown of the other bays
            if all(abs(b - h[0]) > 1e-3 for h in hang):
                for o in (torus("crown_hook", 0.022, 0.006, loc=(0, b, z_hk - 0.02), rot=(90, 0, 0), major_seg=10, minor_seg=5),
                          cylinder("crown_hook_plate", 0.04, 0.008, loc=(0, b, z_hk - 0.002), verts=10)):
                    assign(o, iron)
                    iron_objs.append(o)
    else:
        zceil = G - 0.2
        y0i, y1i = -D / 2 + 0.45, D / 2 - 0.45
        nx_ = max(2, int(2 * jx / 0.5))
        boards = _sheet("ceiling_boards", [[(lerp(-jx, jx, j / nx_), lerp(ya, yb, i / ny), zceil) for j in range(nx_ + 1)] for i in range(ny + 1)])
        assign(boards, oak)
        uv_planar(boards, u_axis=(1, 0, 0), v_axis=(0, 1, 0))
        flat(boards)
        timber_objs.append(boards)
        nj_ = max(4, int(round(2 * jx / 0.42)))
        for k in range(nj_ + 1):
            xk = -jx + 0.06 + (2 * jx - 0.12) * k / nj_
            o = box("joist", (0.085 * rr_.vary(1, 0.08), y1i - y0i + 0.1, 0.13), loc=(xk + rr_.u(-0.01, 0.01), 0, zceil - 0.065), bevel=0.006, segments=1)
            uv_grain(o, "Y")
            timber_objs.append(o)
        nbm = max(2, int(round((y1i - y0i) / 1.35)))
        zb_ = zceil - 0.13
        beams_y = [y0i + (y1i - y0i) * (k + 0.5) / nbm for k in range(nbm)]
        for yk in beams_y:
            bh = 0.27 * rr_.vary(1, 0.06)
            o = box("beam", (2 * jx + 0.1, 0.23 * rr_.vary(1, 0.07), bh), loc=(0, yk, zb_ - bh / 2), rot=(rr_.u(-1.5, 1.5), 0, rr_.u(-0.6, 0.6)),
                    bevel=0.02, segments=2)
            densify(o, 0.5)
            displace_noise(o, 0.006, 0.5, seed=sd + int(yk * 10) % 50)
            uv_grain(o, "X")
            timber_objs.append(o)
            for sx in (-1, 1):
                zt_ = zb_ - 0.27
                prof = [(sx * (jx + 0.02), zt_), (sx * (jx - 0.118), zt_), (sx * (jx - 0.118), zt_ - 0.08), (sx * (jx - 0.08), zt_ - 0.15),
                        (sx * (jx - 0.06), zt_ - 0.24), (sx * (jx + 0.02), zt_ - 0.4)]
                pas_stone.append(_prism("corbel", prof, yk - 0.17, yk + 0.17, bev=0.012))
            corbel_blocks.append((yk - 0.17, yk + 0.17, zb_ - 0.27 - 0.36, zceil + 0.1))
        # inner faces of the two end walls above the portal arches
        for sy, flip in ((-1, True), (1, False)):
            e = strip("end_wall", outline(0.06, xclamp=jx), rectmap(-jx, jx, zceil), sy * (D / 2 - 0.45), flip=flip)
            assign(e, st_wall)
            wall_objs.append(e)
        gaps = [0.5 * (beams_y[k] + beams_y[k + 1]) for k in range(nbm - 1)] or [0.0]
        pick_spots(gaps, zceil - 0.13)
    for o in pas_stone:
        assign(o, st_d)
    if pas_stone:
        stone_objs.append(join(pas_stone, "passage_stone"))

    # ------------------------------------------------------------------ coursed rubble of the passage walls
    if rub:
        rst = rng.fork("rubble_in")
        n_st = 0
        for sx in (-1, 1):
            n_st += lay_stones("rubble_passage", IW[sx], (-(D / 2 - 0.46), D / 2 - 0.46, 0.0, zwall), in_blocks[sx] + corbel_blocks, rst,
                               hc=(0.085, 0.15), ln=(0.16, 0.4), proud=(0.006, 0.014), cap=300)
        for (f, u0, u1, z0, z1, o_) in seat_backs:
            n_st += lay_stones("rubble_seat", f, (u0 + 0.004, u1 - 0.004, z0 + 0.004, z1 - 0.004), [], rst, hc=(0.1, 0.16), ln=(0.2, 0.42),
                               proud=(0.006, 0.014), out0=o_, cap=45)

    # ------------------------------------------------------------------ paving of the passage
    if p.floor != "none":
        # laid 2 cm proud of the ground with a fall to a centre gutter, dipping under the ground line at both mouths;
        # the wet mask (special) leaves a damp film down the middle and at the mouths that mirrors the lanterns
        fm = mat("sett_basalt", scale=1.8) if p.floor == "setts" else mat("cobble_granite")
        xs = [-jx, -jx * 0.55, -0.19, 0.0, 0.19, jx * 0.55, jx]
        ys = cuts(-D / 2, D / 2, [-D / 2 + 0.6, D / 2 - 0.6], step=0.5)

        def fz(x, y):
            t = clamp((D / 2 - abs(y)) / 0.6)
            return lerp(-0.004, 0.02 - 0.009 * clamp(1.2 - abs(x) / (jx * 0.55)), t * t * (3 - 2 * t))

        nmw = noise_mask(1.3, seed=sd + 41, lo=0.25, hi=0.8)

        def wetf(co, n):
            a = 0.42 + 0.5 * np.clip(1.0 - np.abs(co[:, 0]) / jx, 0.0, 1.0) + 0.35 * np.clip((np.abs(co[:, 1]) - (D / 2 - 1.6)) / 1.6, 0.0, 1.0)
            return np.clip(a * (0.62 + 0.38 * nmw(co, n)), 0.0, 1.0)

        for nm_, cols, ua, va in (("passage_floor", (0, 1, 4, 5), (1, 0, 0), (0, 1, 0)), ("passage_gutter", (2, 3), (0, 1, 0), (1, 0, 0))):
            fv, ff = [], []
            for c in cols:
                b = len(fv)
                for y in ys:
                    fv += [(xs[c], y, fz(xs[c], y)), (xs[c + 1], y, fz(xs[c + 1], y))]
                ff += [(b + 2 * i, b + 2 * i + 1, b + 2 * i + 3, b + 2 * i + 2) for i in range(len(ys) - 1)]
            fl = mesh_object(nm_, fv, ff)
            assign(fl, fm)
            uv_planar(fl, ua, va)
            smooth(fl, 30)
            paint(fl, special=wetf, grime=lambda co, n: 0.45 * np.clip((np.abs(co[:, 0]) - (jx - 0.45)) / 0.45, 0.0, 1.0))

    # ------------------------------------------------------------------ lanterns
    lq = float(p.lantern_size) / 0.22
    s_len, s_off = 0.43 * lq, -0.27 * lq

    def soffit_z(x):
        """Height of the vault (or of the joists) over x."""
        if p.ceiling != "vault":
            return G - 0.33
        x = min(abs(x), jx)
        for (x0, z0), (x1, z1) in zip(vpts[NA // 2:], vpts[NA // 2 + 1:]):
            if x0 - 1e-6 <= x <= x1 + 1e-6 and x1 - x0 > 1e-6:
                return lerp(z0, z1, (x - x0) / (x1 - x0))
        return zs

    def box_lantern(name, q=1.0):
        """Tapered four-pane lantern hanging from the origin: ring, vent cap, pyramid roof, glazed body, tray, drip.
        The horn panes are the light source: they glow."""
        I, zt_, zb2, ht, hb = [], -0.15 * q, -0.37 * q, 0.1 * q, 0.06 * q
        I.append(torus("bl_ring", 0.016 * q, 0.004 * q, loc=(0, 0, -0.016 * q), rot=(90, 0, 0), major_seg=10, minor_seg=5))
        I.append(cylinder("bl_stem", 0.006 * q, 0.03 * q, loc=(0, 0, -0.045 * q), verts=6))
        I.append(lathe("bl_vent", [(0.0, -0.055 * q), (0.03 * q, -0.06 * q), (0.034 * q, -0.074 * q), (0.02 * q, -0.08 * q)], segments=8))
        I.append(lathe("bl_roof", [(0.026 * q, -0.078 * q), (0.168 * q, -0.14 * q), (0.168 * q, -0.151 * q), (0.0, -0.151 * q)], segments=4,
                       rot=(0, 0, 45)))
        I.append(box("bl_tray", (0.135 * q, 0.135 * q, 0.012 * q), loc=(0, 0, zb2 - 0.006 * q), bevel=0.002 * q, segments=1))
        I.append(lathe("bl_drip", [(0.0, -0.43 * q), (0.01 * q, -0.42 * q), (0.018 * q, -0.402 * q), (0.008 * q, -0.386 * q), (0.008 * q, -0.38 * q)],
                       segments=6))
        bar = [(-0.006 * q, -0.006 * q), (0.006 * q, -0.006 * q), (0.006 * q, 0.006 * q), (-0.006 * q, 0.006 * q)]
        gv, gf = [], []
        for c in range(4):
            ca, sa = [(1, 0), (0, 1), (-1, 0), (0, -1)][c]
            rot_ = lambda x, y, z: (x * ca - y * sa, x * sa + y * ca, z)
            I.append(sweep("bl_bar", [rot_(ht, ht, zt_), rot_(hb, hb, zb2)], bar, smooth_path=False, up=rot_(1, -1, 0)))
            I.append(box("bl_rim", (0.012 * q, 2 * ht + 0.012 * q, 0.012 * q), loc=rot_(ht, 0, zt_ - 0.004 * q), rot=(0, 0, 90 * c), bevel=0.0, segments=1))
            I.append(sweep("bl_mid", [rot_(ht * 0.995, 0, zt_), rot_(hb * 0.995, 0, zb2)], [(a_ * 0.6, b_ * 0.6) for a_, b_ in bar], smooth_path=False,
                           up=rot_(0, 1, 0)))
            b0 = len(gv)
            gv += [rot_(hb - 0.002 * q, -hb, zb2), rot_(hb - 0.002 * q, hb, zb2), rot_(ht - 0.002 * q, ht, zt_), rot_(ht - 0.002 * q, -ht, zt_)]
            gf.append((b0, b0 + 1, b0 + 2, b0 + 3))
        for o in I:
            assign(o, iron)
        gl = mesh_object("bl_glass", gv, gf)
        flat(gl)
        assign(gl, glow_m)
        return join(I + [gl], name)

    lights = []                                       # (priority, name, loc, kwargs)
    soot_at = []                                      # (x, y, radius) of the soot patches on the vault
    hang.sort(key=lambda h: (round(abs(h[0]), 3), h[0]))
    EYE = 1.65
    for i, (hy, hz) in enumerate(hang):
        # a lantern may not hang into the line from an eye 3 m outside one mouth to the crown of the other mouth
        z_need = EYE + (zc - EYE) * (0.5 * D + abs(hy) + 3.0) / (D + 3.0) + 0.08
        drop = clamp(min(p.lantern_drop, hz - s_len - z_need), 0.05, 0.9)
        eye = torus("hook_eye", 0.022, 0.006, loc=(0, hy, hz - 0.02), rot=(90, 0, 0), major_seg=12, minor_seg=6)
        stem = cylinder("hook_stem", 0.008, 0.07, loc=(0, hy, hz + 0.0), verts=8)
        plate = cylinder("hook_plate", 0.04, 0.008, loc=(0, hy, hz - 0.002), verts=12)
        for o in (eye, stem, plate):
            assign(o, iron)
            iron_objs.append(o)
        if drop >= 0.1:
            ch = chain("lantern_chain", [(0, hy, hz - 0.04), (0, hy, hz - drop)], link_len=0.06, wire=0.006, sides=4)
        else:
            ch = torus("lantern_link", 0.5 * (drop - 0.012), 0.005, loc=(0, hy, hz - 0.02 - 0.5 * (drop - 0.012)), rot=(0, 90, 0), major_seg=10, minor_seg=5)
        assign(ch, iron)
        lo = place(box_lantern(f"passage_lantern{i}", lq), loc=(0, hy, hz - drop), rot=(0, 0, 45 if i % 2 else 0))
        part([lo, ch], f"passage_lantern{i}", pivot=(0, hy, hz - 0.03), motion="swing", amp=2.0, speed=0.35)
        lights.append((0.0 if i == 0 else 0.5 + 0.01 * i, f"passage{i}", (0, hy, hz - drop + s_off),
                       dict(intensity=p.lantern_cd, range=10.0, flicker=0.14, shadow=(i == 0), shadow_near=0.3)))
        soot_at.append((0.0, hy, 1.25))

    def wall_lantern(f, u, z, reach=0.5, small=False, k=1.0, q=1.0):
        """Forged gallows bracket with scroll stay; the lantern hangs from its hook."""
        o = []
        o.append(box("wl_plate", (0.05, 0.008, 0.46 * k), loc=(0, -0.004, -0.15 * k), bevel=0.002, segments=1))
        o.append(box("wl_arm", (0.022, reach + 0.07, 0.022), loc=(0, -(reach + 0.07) / 2, 0.0), bevel=0.003, segments=1))
        bar = [(-0.01, -0.004), (0.01, -0.004), (0.01, 0.004), (-0.01, 0.004)]
        stay = bezier_path((0, -0.01, -0.34 * k), (0, -reach * 0.25, -0.33 * k), (0, -reach * 0.8, -0.2 * k), (0, -reach * 0.86, -0.012), n=8)
        o.append(sweep("wl_stay", stay, bar, up=(1, 0, 0), smooth_path=not small))
        sp = spiral_path(0.008, 0.05, 1.5, n=14, a0_deg=90, plane="YZ", center=(0, -(reach + 0.07), 0.05))
        o.append(sweep("wl_scroll", sp, bar, up=(1, 0, 0), scale_fn=lambda t: 0.6 + 0.4 * t, smooth_path=False))
        if not small:
            sp2 = spiral_path(0.006, 0.04 * k, 1.4, n=12, a0_deg=270, plane="YZ", center=(0, -reach * 0.42, -0.2 * k))
            o.append(sweep("wl_scroll2", sp2, bar, up=(1, 0, 0), scale_fn=lambda t: 0.6 + 0.4 * t, smooth_path=False))
        o.append(torus("wl_eye", 0.018, 0.005, loc=(0, -reach, -0.028), rot=(0, 90, 0), major_seg=10, minor_seg=6))
        for dz in (0.04 * k, -0.34 * k):
            o.append(bolt("wl_bolt", 0.009, 0.007, loc=(0, -0.008, dz), rot=(90, 0, 0), kind="square"))
        for x in o:
            assign(x, iron)
        j = join(o, "wall_bracket")
        place(j, loc=f.pt(u, z), rot=(0, 0, f.th))
        iron_objs.append(j)
        place(box_lantern("wall_lamp", q), loc=f.pt(u, z - 0.045, reach), rot=(0, 0, f.th))
        return f.pt(u, z - 0.045 - 0.27 * q, reach)

    # low bracket lanterns just inside the mouths, on short arms: lantern and bracket stay within 0.15 m of the clear
    # passage width (the bay walls stand REC behind the jambs); unshadowed fill for the rib flanks, the vault and the bays
    z_arm = clamp(zs + 0.45, 2.55, 2.8)
    r_arm = max(0.14, AWP - 0.122 * lq)
    wl_locs = []
    for i, (sx, k) in enumerate([(-1, 0), (1, nb - 1)][:int(p.wall_lamps)]):
        yc = sum(spans[(sx, k)]) / 2
        fL = _Face(sx * (wall_x(z_arm - 0.1) - 0.004 - (WO if z_arm <= zwall else 0.0)), 0, 90 if sx < 0 else 270, D)
        loc = wall_lantern(fL, -sx * yc, z_arm, reach=r_arm, small=True, k=0.55, q=lq)
        wl_locs.append(loc)
        soot_at.append((sx * (jx - r_arm), yc, 0.7))

    # every stall's lamp is a real light (it hangs on the hook rail inside the hatch: lit stock, a pool on the counter and
    # the floor in front); they are ordered so that a low max_lights keeps the lit stalls spread along both walls.  The
    # first keeps the light name "bay_lamp", the others are "stall_lamp1" ...; the bracket lanterns are "passage_wall0/1".
    cand = [b for b in bay_lamps if b[4]]
    anchors = [(0.0, h[0]) for h in hang[:1]] + [(-0.5 * (jx + W / 2), -D / 2 - 0.6), (0.5 * (jx + W / 2), D / 2 + 0.6)]
    st_l = []
    while cand:
        b = max(cand, key=lambda b: (round(min(hypot(b[0] - ax, b[1] - ay) for ax, ay in anchors), 2), -b[1], b[0]))
        cand.remove(b)
        st_l.append(b)
        anchors.append((b[0], b[1]))
    for i, b in enumerate(st_l):
        lights.append((1.0 + 0.05 * i if i < 6 else 3.5 + 0.05 * i, "bay_lamp" if i == 0 else "stall_lamp%d" % i, b[:3],
                       dict(color="#ffb25c", intensity=0.36 * p.lantern_cd, range=4.5, flicker=0.16, shadow_near=0.3)))
    for i, loc in enumerate(wl_locs):
        lights.append((4.0 + 0.1 * i, "passage_wall%d" % i, loc, dict(color="#ffb25c", intensity=0.45 * p.lantern_cd, range=7.0, flicker=0.14)))
    for i, b in enumerate([b for b in bay_lamps if not b[4]][:4]):      # the little lamps of the shop fronts, doors and seats
        lights.append((5.0 + 0.1 * i, "shop_lamp%d" % i, b[:3], dict(color="#ffb25c", intensity=0.12 * p.lantern_cd, range=3.5, flicker=0.16)))

    # forged sign brackets inside both mouths (opposite the bracket lanterns) with a little hanging board: lifted clear
    # of 2.8 m where the vault allows it, otherwise a short bracket with a narrow board close to the wall
    bwid, bh = 0.44, 0.23
    z_sgn, reach, cb = 2.8 + 0.08 + bh, 0.62, 0.36
    wx_ = lambda z: wall_x(z) - 0.004 - (WO if z <= zwall else 0.0)
    if soffit_z(wx_(z_sgn - 0.2) - reach - 0.07) < z_sgn + 0.12 or z_sgn > G - 0.75:
        z_sgn, reach, cb, bwid = clamp(zs + 0.55, 2.66, 2.95), AWP - 0.012, 0.135, 0.22
    for tag, (sx, k) in (("f", (1, 0)), ("b", (-1, nb - 1))):
        yc = sum(spans[(sx, k)]) / 2
        fS = _Face(sx * wx_(z_sgn - 0.2), 0, 90 if sx < 0 else 270, D)
        ey = min(0.15, bwid / 2 - 0.035)
        o = [box("sb_plate", (0.045, 0.008, 0.3), loc=(0, -0.004, -0.08), bevel=0.002, segments=1),
             box("sb_arm", (0.02, reach + 0.25, 0.02), loc=(0, -(reach + 0.05) / 2 + 0.1, 0.0), bevel=0.003, segments=1)]
        bar = [(-0.009, -0.004), (0.009, -0.004), (0.009, 0.004), (-0.009, 0.004)]
        o.append(sweep("sb_stay", bezier_path((0, -0.01, -0.21), (0, -reach * 0.2, -0.2), (0, -reach * 0.3, -0.05), (0, -reach * 0.42, -0.012), n=7),
                       bar, up=(1, 0, 0), smooth_path=False))
        o.append(sweep("sb_scroll", spiral_path(0.007, 0.045, 1.5, n=12, a0_deg=90, plane="YZ", center=(0, -(reach + 0.05), 0.045)), bar,
                       up=(1, 0, 0), scale_fn=lambda t: 0.6 + 0.4 * t, smooth_path=False))
        for yy in (cb - ey, cb + ey):
            o.append(torus("sb_eye", 0.014, 0.004, loc=(0, -yy, -0.022), rot=(0, 90, 0), major_seg=8, minor_seg=5))
        if p.passage_sign:
            for yy in (cb - ey, cb + ey):
                for (zz, rz) in ((-0.046, 0), (-0.068, 90)):
                    o.append(torus("sign_link", 0.014, 0.0035, loc=(0, -yy, zz), rot=(0, 90, rz), major_seg=8, minor_seg=4))
        for x in o:
            assign(x, iron)
        j = join(o, "sign_bracket")
        place(j, loc=fS.pt(-sx * yc, z_sgn), rot=(0, 0, fS.th))
        iron_objs.append(j)
        socket("sign_passage_" + tag, "attach", fS.pt(-sx * yc, z_sgn - 0.036, cb), rot=(0, 0, fS.th), role="market_sign")
        if p.passage_sign:
            body = str(p.passage_sign)
            zb0 = -0.075 - bh
            sb = [box("sign_board", (0.026, bwid, bh), loc=(0, -cb, zb0 + bh / 2), bevel=0.004, segments=1)]
            assign(sb[0], shutter_m)
            for zz in (zb0 + 0.012, zb0 + bh - 0.012):
                for xx in (-0.015, 0.015):
                    e = box("sign_fillet", (0.006, bwid - 0.03, 0.01), loc=(xx, -cb, zz), bevel=0.0, segments=1)
                    assign(e, plaster)
                    sb.append(e)
            tsz = min(0.12, (bwid - 0.06) / (0.56 * max(len(body), 3)))
            for sg in (-1, 1):
                t_ = text("sign_text", body, size=tsz, depth=0.004, loc=(sg * 0.013, -cb, zb0 + bh / 2), rot=(90, 0, 90 * sg), font="sans_narrow")
                assign(t_, plaster)
                sb.append(_front_only(t_, (sg, 0, 0), lim=0.5))
            sj = join(sb, "passage_sign_" + tag)
            place(sj, loc=fS.pt(-sx * yc, z_sgn), rot=(0, 0, fS.th))
            paint(sj, wear=mask_mul(noise_mask(0.2, seed=sd + 18, lo=0.5, hi=0.85), 0.3 + 0.4 * age), grime=0.2 + 0.2 * age)

    # ------------------------------------------------------------------ the market sign and the banners under the vault
    def crown(name, c, s, ax, th=0.008):
        """Flat gilt crown (band, five points) centred on c in the plane square to the passage axis; ax = +-1 faces -+Y."""
        cx, cy, cz = c
        o = [box(name, (0.2 * s, th, 0.03 * s), loc=(cx, cy, cz - 0.045 * s), bevel=0.0, segments=1)]
        for dx, h_ in ((-0.08, 0.075), (-0.04, 0.06), (0.0, 0.095), (0.04, 0.06), (0.08, 0.075)):
            pts = [(cx + (dx - 0.022) * s, cz - 0.03 * s), (cx + (dx + 0.022) * s, cz - 0.03 * s), (cx + dx * 1.18 * s, cz - 0.03 * s + h_ * s)]
            o.append(_prism(name, pts, cy - th / 2, cy + th / 2, 0.0))
        return o

    def hung_sign(words, y_s, key, framed):
        """Double-sided lettered board with a crown gable hung on two chains across the walkway (above 2.82 m): enamelled
        board in an iron band, or (framed) an oak-framed board with turned drops.  False if it does not fit."""
        if framed and sum(len(w_) for w_ in words) + len(words) - 1 <= 14:
            words = [" ".join(words)]                    # the hall board carries its name in one line
        Wb = clamp(2 * half * (0.5 if framed else 0.42), 0.8, 1.45 if framed else 1.3)
        tsz = [min(0.17, (Wb - 0.16) / (0.68 * max(len(w_), 3))) for w_ in words]
        Hb = 0.07 + sum(t_ * 1.36 for t_ in tsz)
        Ht = Hb + 0.15                                   # with the crown gable
        ex = Wb / 2 - 0.12                               # chain eyes
        z_hook = soffit_z(ex) - (0.17 if p.ceiling == "vault" else 0.27)
        fit = clamp((z_hook - 0.1 - 2.82 - (0.05 if framed else 0.0)) / Ht, 0.0, 1.0)
        if fit < 0.6:
            return False
        if fit < 1.0:
            Wb, Hb, Ht, ex, tsz = Wb * fit, Hb * fit, Ht * fit, ex * fit, [t_ * fit for t_ in tsz]
        zb = max(2.82 + (0.05 if framed else 0.0), min(3.0, z_hook - 0.3 - Ht))
        g_ = Ht - Hb
        pts = [(-Wb / 2, zb), (Wb / 2, zb), (Wb / 2, zb + Hb), (0.2, zb + Hb), (0.16, zb + Hb + g_ * 0.3), (0.09, zb + Hb + g_ * 0.82),
               (0.0, zb + Ht), (-0.09, zb + Hb + g_ * 0.82), (-0.16, zb + Hb + g_ * 0.3), (-0.2, zb + Hb)]
        brd = join([box(key + "_board", (Wb, 0.032, Hb), loc=(0, y_s, zb + Hb / 2), bevel=0.004, segments=1),
                    _prism(key + "_gable", pts[3:] + [(-0.2, zb + Hb - 0.01), (0.2, zb + Hb - 0.01)], y_s - 0.016, y_s + 0.016, 0.004)], key + "_board")
        assign(brd, ban_m)
        flat(brd)
        uv_planar(brd, (1, 0, 0), (0, 0, 1))
        ms_iron, ms_gilt, ms_oak = [], [], []
        for sg in (-1, 1):
            yf = y_s + sg * 0.0165
            zl_ = zb + Hb - 0.035
            for t_, w_ in zip(tsz, words):
                zl_ -= t_ * 0.5
                tx = text(key + "_text", w_, size=t_, depth=0.006, loc=(0, yf, zl_), rot=(90, 0, 0 if sg < 0 else 180), font="serif", spacing=1.06)
                ms_gilt.append(_front_only(tx, (0, sg, 0), lim=0.5))
                zl_ -= t_ * 0.86
            # CLEARANCE RULE: no glowing face buried in the board (the backs of the crown and the fillets lay 0.5 / 2 mm
            # under its face), and the fillets stand 4.5 mm proud of it (they stood 3)
            ms_gilt += [_front_only(o, (0, sg, 0)) for o in
                        crown(key + "_crown", (0, yf + sg * 0.003, zb + Hb + g_ * 0.42), min(1.0, g_ / 0.15) * 0.95, sg)]
            for zz in (zb + 0.022, zb + Hb - 0.012):
                ms_gilt.append(_front_only(box(key + "_fillet", (Wb - 0.07, 0.008, 0.008), loc=(0, yf, zz), bevel=0.0, segments=1), (0, sg, 0)))
        if framed:                                       # oak frame: bottom rail, stiles, top rails either side of the gable, turned drops
            fw_ = 0.046
            for nm_, sz_, lc_, gr_ in (("rail", (Wb + 2 * fw_, 0.052, fw_), (0.0, zb - fw_ / 2), "X"),
                                       ("stile", (fw_, 0.052, Hb + fw_ * 0.5), (-Wb / 2 - fw_ / 2, zb + Hb / 2 + fw_ * 0.25), "Z"),
                                       ("stile", (fw_, 0.052, Hb + fw_ * 0.5), (Wb / 2 + fw_ / 2, zb + Hb / 2 + fw_ * 0.25), "Z"),
                                       ("rail", (Wb / 2 - 0.2 + fw_, 0.052, fw_ * 0.8), (-(Wb / 2 + 0.2 + fw_) / 2, zb + Hb + fw_ * 0.4), "X"),
                                       ("rail", (Wb / 2 - 0.2 + fw_, 0.052, fw_ * 0.8), ((Wb / 2 + 0.2 + fw_) / 2, zb + Hb + fw_ * 0.4), "X")):
                o = box(key + "_" + nm_, sz_, loc=(lc_[0], y_s, lc_[1]), bevel=0.006, segments=1)
                uv_grain(o, gr_)
                ms_oak.append(o)
            for sg in (-1, 1):                           # turned finials on the stiles
                ms_oak.append(lathe(key + "_finial", [(0.02, 0.0), (0.02, 0.012), (0.011, 0.026), (0.019, 0.042), (0.012, 0.058), (0.0, 0.066)], segments=8,
                                    loc=(sg * (Wb / 2 + fw_ / 2), y_s, zb + Hb + fw_ * 0.5), smooth_angle=50.0))
            ze = zb + Hb + fw_ * 0.8
        else:                                            # iron band round the board
            for (sx_, sz_, lx, lz) in ((Wb + 0.02, 0.012, 0.0, zb - 0.006), (0.012, Hb + 0.012, -Wb / 2 - 0.006, zb + Hb / 2 - 0.006),
                                       (0.012, Hb + 0.012, Wb / 2 + 0.006, zb + Hb / 2 - 0.006)):
                ms_iron.append(box(key + "_band", (sx_, 0.04, sz_), loc=(lx, y_s, lz), bevel=0.002, segments=1))
            ze = zb + Hb
        for sg in (-1, 1):                               # eyes, chains and the hooks in the rib / beam
            ms_iron.append(torus(key + "_eye", 0.02, 0.005, loc=(sg * ex, y_s, ze + 0.016), rot=(90, 0, 0), major_seg=10, minor_seg=5))
            ms_iron.append(chain(key + "_chain", [(sg * ex, y_s, ze + 0.03), (sg * ex, y_s, z_hook - 0.03)], link_len=0.085, wire=0.007, sides=4))
            hk = [torus(key + "_hook", 0.022, 0.006, loc=(sg * ex, y_s, z_hook - 0.02), rot=(90, 0, 0), major_seg=10, minor_seg=5),
                  cylinder(key + "_hook_stem", 0.008, 0.2, loc=(sg * ex, y_s, z_hook + 0.09), verts=6)]
            for o in hk:
                assign(o, iron)
                iron_objs.append(o)
        for o in ms_iron:
            assign(o, iron)
        for o in ms_gilt:
            assign(o, glow_m)
        for o in ms_oak:
            assign(o, oak)
        mj = join([brd] + ms_iron + ms_gilt + ms_oak, key)
        paint(mj, grime=0.12 + 0.15 * age, wear=0.15 * age)
        part([mj], key, pivot=(0, y_s, z_hook - 0.02), motion="swing", amp=0.7, speed=0.25, axis="x")
        socket(key, "attach", (0, y_s, zb + Hb / 2), rot=(0, 0, 0), role="market_sign", width=round(Wb, 3), height=round(Hb, 3))
        return True

    # the big enamelled board under the first rib / beam inside the front mouth, and an oak-framed one deep inside the
    # passage under the last: whoever walks through reads one of them ahead in either direction
    msign = [w_ for w_ in str(p.market_sign).upper().split() if w_][:3]
    hsign = [w_ for w_ in str(p.hall_sign).upper().split() if w_][:3]
    y_sgn = y_hall = None
    if msign:
        y_sgn = (-D / 2 + bw) if p.ceiling == "vault" else min(beams_y, key=lambda b: abs(b + D / 2 - 1.8))
        if not hung_sign(msign, y_sgn, "market_sign", False):
            y_sgn = None
    if hsign:
        y_hall = (D / 2 - bw) if p.ceiling == "vault" else min(beams_y, key=lambda b: abs(b - D / 2 + 1.8))
        if (y_sgn is not None and abs(y_hall - y_sgn) < 1.2) or not hung_sign(hsign, y_hall, "hall_sign", True):
            y_hall = None

    nban = int(p.banners)
    if nban > 0:
        rbn = rng.fork("banners")
        xb, wb = half * 0.48, clamp(half * 0.34, 0.36, 0.55)
        spots_y = [y for y in bay_c[1:] if all(ys_ is None or abs(y - ys_) > 0.5 for ys_ in (y_sgn, y_hall))] if nb > 2 else list(bay_c)
        step_ = max(1, len(spots_y) // nban)
        ban_cloth, ban_crown, ban_iron = [], [], []
        for i, yb_ in enumerate(spots_y[::step_][:nban]):
            sxb = 1 if i % 2 == 0 else -1
            zr_ = soffit_z(xb + wb / 2 + 0.03) - 0.11
            Lb = min(zr_ - 2.82, wb * 1.9, 1.15)
            if Lb < 0.42:
                continue
            ph = rbn.u(0, 6.0)
            nu_, nv_ = 6, 7
            P = []
            for iu in range(nu_ + 1):
                tu = iu / nu_
                tail = Lb - 0.16 * (1 - abs(2 * tu - 1))     # swallow-tail hem
                P.append([(sxb * xb + (0.5 - tu) * wb * (1 - 0.03 * (jv / nv_)),
                           yb_ + 0.022 * (jv / nv_) ** 0.7 * sin(tu * 7.0 + ph + jv * 0.5) + 0.012 * (jv / nv_),
                           zr_ - 0.012 - tail * jv / nv_) for jv in range(nv_ + 1)])
            cl = _sheet("banner", P)
            uv_planar(cl, (1, 0, 0), (0, 0, 1))
            smooth(cl, 50)
            ban_cloth.append(cl)
            for sg in (-1, 1):
                ban_crown += crown("banner_crown", (sxb * xb, yb_ + sg * 0.012 + 0.006, zr_ - Lb * 0.36), wb / 0.34, sg, th=0.004)
                ban_crown.append(box("banner_stripe", (wb * 0.9, 0.004, 0.022), loc=(sxb * xb, yb_ + sg * 0.008 + 0.002, zr_ - 0.07), bevel=0.0, segments=1))
            # forged rod with knob ends through the sleeve, hung on two chains from hooks in the vault
            ban_iron.append(cylinder("banner_rod", 0.009, wb + 0.1, loc=(sxb * xb, yb_, zr_), rot=(0, 90, 0), verts=6))
            for sg in (-1, 1):
                xe = sxb * xb + sg * (wb / 2 + 0.05)
                ban_iron.append(sphere("banner_knob", 0.017, loc=(xe, yb_, zr_), segments=6, rings=4))
                xh = sxb * xb + sg * (wb / 2 - 0.03)
                zh_ = soffit_z(xh)
                # hanger rods with an eye at the foot, let into the vault
                ban_iron.append(cylinder("banner_hanger", 0.005, zh_ + 0.06 - zr_ - 0.02, loc=(xh, yb_, (zh_ + 0.06 + zr_ + 0.02) / 2), verts=5, cap=False))
                ban_iron.append(torus("banner_eye", 0.014, 0.004, loc=(xh, yb_, zr_ + 0.004), rot=(0, 90, 0), major_seg=8, minor_seg=4))
        for o in ban_cloth:
            assign(o, ban_m)
        for o in ban_crown:
            assign(o, plaster)
        for o in ban_iron:
            assign(o, iron)
            iron_objs.append(o)
        if ban_cloth:
            bj = join(ban_cloth + ban_crown, "banners")
            paint(bj, grime=mask_mul(height_ramp(3.6, 2.7), 0.2 + 0.3 * age))

    pier_a, pier_b = half + 0.72, W / 2 - 0.66
    avail = pier_b - pier_a
    xP = (pier_a + pier_b) / 2 if avail > 0.35 else W / 2 - 0.5
    zl = min(G - 0.85, 3.55)
    lan_kw = dict(color="#ffb25c", intensity=0.5 * p.lantern_cd, range=9.0, flicker=0.14, shadow_near=0.4)
    # one lantern beside each portal (the other pier carries the sign socket)
    fF, fB = (_Face(0, -D / 2 - 0.02, 0, W), _Face(0, D / 2 + 0.02, 180, W)) if rub else (F["f"], F["b"])     # on the faces of the rubble
    # (the light sockets sit a hand's breadth outside the glass: a softer wash over pier and arch, no white core on the wall)
    for pr_, nm_, f_ in ((1.5, "front_lantern0", fF), (1.6, "back_lantern", fB)):
        lx_, ly_, lz_ = wall_lantern(f_, -xP, zl, reach=0.486, q=1.45 * lq)
        lights.append((pr_, nm_, (lx_ + 0.22 * f_.nx, ly_ + 0.22 * f_.ny, lz_), lan_kw))

    # ------------------------------------------------------------------ ground-storey door and small windows
    tri_log = {}

    def put(comp, f, u, z, tag, out=0.0):
        o = comp.place(f.pt(u, z, out), rot=(0, 0, f.th), sockets=False)
        tri_log[tag] = tri_log.get(tag, 0) + _tris(o)
        return o

    rw = rng.fork("windows")
    gw = None
    if avail >= 0.75:
        gww = clamp(avail - 0.5, 0.5, 0.8)
        gw = {lit: use("window_casement", width=gww, height=1.0, lights=1, surround="stone", glazing="bars", panes_x=2, panes_y=3,
                       shutters=False, lit=lit, stone=p.stone_color, age=age, curtain=0.5, seed=sd + 30)
              for lit in (True, False)}
    door_ok = bool(p.door) and avail >= 1.12
    if door_ok:
        f = F["f"]
        rdo = rng.fork("door")
        dw = clamp(avail - 0.36, 0.8, 1.0)
        stp = 0.15
        dzs = stp + 1.62
        dpts, dnr = _arch_poly("round", dw / 2, dzs, dw / 2, 12)
        darc = _arc_param(dpts, dnr)
        so, wo, io = [], [], []
        so.append(ring("door_arch", rdo, -0.13, 0.02, -0.02, 0.2, 0.2, key=True, jit=0.01, arc=darc, keyout=0.06, bev=0.008, gap=0.003))
        nrow = 5
        hj = (dzs - stp) / nrow
        for sx in (-1, 1):
            for i in range(nrow):
                wj = (0.3 if i % 2 == 0 else 0.2) * rdo.vary(1.0, 0.06)
                so.append(box("door_jamb", (wj + 0.02, 0.15 + rdo.u(-0.006, 0.006), hj - 0.003), loc=(sx * (dw / 2 - 0.02 + (wj + 0.02) / 2), -0.055, stp + hj * (i + 0.5)),
                              bevel=0.009, segments=1))
        st_ = box("door_step", (dw + 0.62, 0.46, stp), loc=(0, -0.21, stp / 2), bevel=0.014, segments=2)
        densify(st_, 0.2)
        def worn(co):
            if co.z > stp * 0.9:
                co.z -= 0.012 * max(0.0, 1 - abs(co.x) / 0.3)
            return co

        transform_verts(st_, worn)
        so.append(st_)
        nbd = 5
        xa, xb = -dw / 2 - 0.015, dw / 2 + 0.015
        zarc = lambda x: dzs + (max((dw / 2 + 0.02) ** 2 - x * x, 0.0)) ** 0.5
        for i in range(nbd):
            x0, x1 = lerp(xa, xb, i / nbd) + 0.003, lerp(xa, xb, (i + 1) / nbd) - 0.003
            yb_ = -0.045 - rdo.u(0.0, 0.006)
            b_ = _prism("door_board", [(x0, stp + 0.005), (x1, stp + 0.005), (x1, zarc(x1)), ((x0 + x1) / 2, zarc((x0 + x1) / 2)), (x0, zarc(x0))], yb_, -0.005, bev=0.004)
            uv_grain(b_, "Z")
            wo.append(b_)
        for zz in (stp + 0.42, stp + 1.5):
            io.append(box("door_strap", (dw * 0.78, 0.008, 0.05), loc=(-dw * 0.09, -0.054, zz), bevel=0.002, segments=1))
            io.append(cylinder("door_pintle", 0.014, 0.085, loc=(-dw / 2 + 0.03, -0.056, zz), verts=8))
            for kx in range(4):
                io.append(bolt("door_nail", 0.011, 0.006, loc=(-dw * 0.4 + kx * dw * 0.21, -0.058, zz), rot=(90, 0, 0), kind="dome"))
        io.append(cylinder("door_rose", 0.035, 0.008, loc=(dw * 0.3, -0.052, stp + 1.02), rot=(90, 0, 0), verts=12))
        io.append(torus("door_ring", 0.05, 0.007, loc=(dw * 0.3, -0.066, stp + 0.975), rot=(96, 0, 0), major_seg=14, minor_seg=6))
        io.append(box("door_keyplate", (0.035, 0.005, 0.09), loc=(dw * 0.3, -0.052, stp + 0.8), bevel=0.002, segments=1))
        for o in so:
            assign(o, st)
        for o in wo:
            assign(o, oak)
        for o in io:
            assign(o, iron)
        if p.door_number:
            nm_ = text("door_number", str(p.door_number), size=0.085, depth=0.006, loc=(0, -0.058, stp + 1.72), font="serif")
            assign(nm_, iron)
            io.append(nm_)
        js, jw, ji = join(so, "door_stone"), join(wo, "door_leaf"), join(io, "door_iron")
        for j in (js, jw, ji):
            place(j, loc=f.pt(-xP, 0.0), rot=(0, 0, f.th))
        stone_objs.append(js)
        timber_objs.append(jw)
        iron_objs.append(ji)
        socket("door", "attach", f.pt(-xP, stp, 0.5), role="door")
    ext_blocks = {"f": [], "b": []}
    if door_ok:
        ext_blocks["f"].append((-xP - dw / 2 - 0.19, -xP + dw / 2 + 0.19, 0.0, dzs))
        for i in range(5):                                 # the arched head, stepped
            a0_, a1_, rm_ = pi / 2 * i / 5, pi / 2 * (i + 1) / 5, dw / 2 + 0.11
            ext_blocks["f"].append((-xP - rm_ * cos((a0_ + a1_) / 2), -xP + rm_ * cos((a0_ + a1_) / 2), dzs + rm_ * sin(a0_) - 0.001, dzs + rm_ * sin(a1_)))
    if gw:
        spots = [("f", xP), ("b", -xP), ("b", xP)] + ([] if door_ok else [("f", -xP)])
        for key, u in spots:
            put(gw[rw.chance(p.lit_ratio * 0.7)], F[key], u, 1.5, "ground_window")
            ext_blocks[key].append((u + gw[True].min.x + 0.03, u + gw[True].max.x - 0.03, 1.5 + gw[True].min.z + 0.02, 1.5 + gw[True].max.z - 0.02))

    # ------------------------------------------------------------------ coursed rubble of the two arch fronts
    if rub:
        rse = rng.fork("rubble_out")
        aoff = [(x + nx * 0.36, z + nz * 0.36) for (x, z), (nx, nz) in zip(apts, anr)][NA // 2:]      # right half: apex -> springing
        z_apex = aoff[0][1]

        def arch_x(z):
            if z >= z_apex:
                return None
            if z <= zs:
                return half + 0.4
            for (x0, z0_), (x1, z1_) in zip(aoff, aoff[1:]):
                if (z0_ - z) * (z1_ - z) <= 0 and abs(z1_ - z0_) > 1e-6:
                    return lerp(x0, x1, (z - z0_) / (z1_ - z0_))
            return half + 0.36

        def clip_arch(poly):
            za_, zb_ = min(q[1] for q in poly), max(q[1] for q in poly)
            xa_ = arch_x(za_)
            if xa_ is None or min(q[0] for q in poly) >= xa_ or max(q[0] for q in poly) <= -xa_:
                return [poly]
            xb_ = arch_x(zb_)
            bx_, bz_ = (xb_, zb_) if xb_ is not None else (0.0, z_apex)
            if bz_ - za_ < 1e-4:
                return [poly]
            right = _clip_right(_clip_right(poly, 0.0, 0.0, 0.0, 1.0), xa_, za_, bx_, bz_)
            left = _clip_right(_clip_right(poly, 0.0, 1.0, 0.0, 0.0), -bx_, bz_, -xa_, za_)
            return [q for q in (right, left) if len(q) >= 3]

        void = 2 * (half + 0.4) * zs + 1.4 * (half + 0.36) * (z_apex - zs)
        n_st = 0
        for key in "fb":
            n_st += lay_stones("rubble_front", F[key], (-(W / 2 - 0.335), W / 2 - 0.335, 0.487, G - 0.127), ext_blocks[key], rse,
                               hc=(0.11, 0.2), ln=(0.2, 0.52), proud=(0.008, 0.019), clip=clip_arch, cap=300, void=void)

    # ------------------------------------------------------------------ upper storeys
    ow = clamp(PW * 0.72, 1.7, 2.4)
    a0, a1 = ow / 2 + 0.3, W / 2 - 0.35
    av = a1 - a0
    shut = bool(p.shutters)
    if av >= 3.3:
        flank_us, wf, shut = [a0 + av * 0.25, a0 + av * 0.75], 0.9, False
    else:
        flank_us = [(a0 + a1) / 2]
        wf = av / 2.45
        if not shut or wf < 0.75:
            shut, wf = False, clamp(av - 0.55, 0.6, 1.1)
    sur = {"timber": "timber", "plaster": "timber", "brick": "stone", "stone": "stone"}[p.walls]
    head = "arched" if p.walls == "brick" else "flat"
    w_h = 1.35
    w_sill = 0.95
    px = {"square": 3, "diamond": 4, "bars": 2}[p.glazing]
    py = {"square": 5, "diamond": 6, "bars": 3}[p.glazing]
    cache = {}

    def casement(kind, lit):
        k = (kind, lit)
        if k not in cache:
            if kind == "flank":
                # plain barred lights (a regular 2 x 3 grid) with a drawn curtain: cheap, and calm from across the square
                cache[k] = use("window_casement", width=wf, height=w_h, lights=2, surround=sur, head=head, glazing="bars", panes_x=2,
                               panes_y=3, shutters=False, timber=p.timber_color, open=0.0, curtain=0.45,
                               stone=p.stone_color, lit=lit, age=age, seed=sd + 40)
            else:
                cache[k] = use("window_casement", width=1.3, height=w_h, lights=2, surround=sur, head=head, glazing=p.glazing, panes_x=px,
                               panes_y=py, shutters=False, transom=True, timber=p.timber_color, stone=p.stone_color, lit=lit, age=age,
                               open=0.0, curtain=0.35, seed=sd + 41)
        return cache[k]

    last = n_up == 1
    oriel = use("window_oriel", width=ow, height=1.1 if last else 1.45, depth=0.5, plan=p.oriel,
                roof="tile" if p.roof == "roof_tile_clay" else "lead", roof_rise=0.36 if last else 0.6, apron=0.0 if last else 0.22,
                finish="oak" if p.walls == "timber" else "painted", paint=p.paint, trim=p.wall_color, lit=True, wear=age * 0.7,
                open_casement=0.0, stained=False, quarry=0.15, seed=sd + 50)
    zo = G + 0.1 - oriel.min.z
    if last and zo + oriel.max.z > top - 0.02:
        oriel = use("window_oriel", width=ow, height=clamp(1.1 - (zo + oriel.max.z - top + 0.02), 1.0, 1.1), depth=0.45, plan=p.oriel,
                    roof="tile" if p.roof == "roof_tile_clay" else "lead", roof_rise=0.35, apron=0.0,
                    finish="oak" if p.walls == "timber" else "painted", paint=p.paint, trim=p.wall_color, lit=True, wear=age * 0.7,
                    open_casement=0.0, stained=False, quarry=0.15, seed=sd + 50)
        zo = G + 0.06 - oriel.min.z
    osock = [s for s in oriel.sockets if s[1] == "light"]

    plan = {}
    for key in "fb":
        for k in range(1, n_up + 1):
            L = [(0.0, "oriel", True)] if k == 1 else [(0.0, "centre", rw.chance(p.lit_ratio))]
            for xf in flank_us:
                for sg in (-1, 1):
                    L.append((sg * xf, "flank", rw.chance(p.lit_ratio)))
            plan[(key, k)] = L
    for ki, key in enumerate("lr"):                    # the flanks mostly face neighbours: one window per storey
        for k in range(1, n_up + 1):
            u = D * 0.22 * (1 if (k + ki) % 2 else -1) + rw.u(-0.15, 0.15)
            keep = k < 3 if n_up < 3 else (k == 1 + ki)        # tall houses: fewer flank windows (triangle budget)
            plan[(key, k)] = [(u, "flank", rw.chance(p.lit_ratio * 0.6))] if keep else []

    shutter_objs = shutter_objs0
    rs = rng.fork("shutters")

    def shutters_at(f, u, zs_, w, h):
        lw_ = w / 2 + 0.035
        for sg in (-1, 1):
            o = []
            uc = u + sg * (w / 2 + 0.085 + lw_ / 2)
            pr = 0.045 + rs.u(0.0, 0.012)
            nb_ = 3 if lw_ < 0.5 else 4
            for i in range(nb_):
                bw = lw_ / nb_
                b_ = fbox("shutter_board", f, uc - lw_ / 2 + bw * (i + 0.5), zs_ + h / 2 + rs.u(-0.006, 0.006), bw - 0.006, h + 0.03, thick=0.024, proud=pr, bev=0.004)
                uv_grain(b_, "Z")
                o.append(b_)
            for zz in (zs_ + 0.2, zs_ + h - 0.2):
                l_ = fbox("shutter_ledge", f, uc, zz, lw_ - 0.05, 0.085, thick=0.022, proud=pr + 0.02, bev=0.004)
                uv_grain(l_, "X")
                o.append(l_)
                hg = fbox("shutter_hinge", f, uc - sg * (lw_ * 0.22), zz, lw_ * 0.6, 0.03, thick=0.006, proud=pr + 0.025, bev=0.0015)
                assign(hg, iron)
                iron_objs.append(hg)
            for b_ in o:
                assign(b_, shutter_m)
            j = join(o, "shutter")
            shutter_objs.append(j)

    # ------------------------------------------------------------------ triangle budget: thin out the repeated parts
    ctris = lambda c: _tris(c.obj) if getattr(c, "obj", None) is not None else 0
    cnt_ = {"dorm_f": int(p.dormers), "dorm_b": max(0, int(p.dormers) - 1), "chim": int(p.chimneys)}
    t_fl, t_ce = ctris(casement("flank", True)), ctris(casement("centre", True)) if n_up > 1 else 0
    base_tris = sum(_tris(o) for o in bpy.context.scene.objects if o.type == "MESH")
    perim = 2 * (W + D) * n_up

    def estimate():
        nfl = sum(1 for v in plan.values() for w in v if w[1] == "flank")
        nsh = sum(1 for (k_, v) in plan.items() for w in v if w[1] == "flank" and k_[0] in "fb") if shut else 0
        nce = sum(1 for v in plan.values() for w in v if w[1] == "centre")
        walls_ = perim * ((130 if p.framing != "close_studding" else 190) + SH * 22) if p.walls == "timber" else n_up * 1800 + 1500
        return (base_tris + ctris(roofc) + 2 * ctris(oriel) + nfl * t_fl + nce * t_ce + nsh * 720 + walls_
                + (cnt_["dorm_f"] + cnt_["dorm_b"]) * 5200 + cnt_["chim"] * 6800 - 2500)

    def thin_out():
        for k in range(n_up, 0, -1):                              # windows in the flanks, from the top
            for key in "lr":
                if plan[(key, k)]:
                    plan[(key, k)] = []
                    return True
        if cnt_["dorm_b"] > 0:
            cnt_["dorm_b"] -= 1
            return True
        for k in range(n_up, 1, -1):                              # paired windows -> one per side
            for key in "fb":
                fl = [w for w in plan[(key, k)] if w[1] == "flank"]
                if len(fl) > 2:
                    um = sum(abs(w[0]) for w in fl) / len(fl)
                    plan[(key, k)] = [w for w in plan[(key, k)] if w[1] != "flank"] + [(-um, "flank", fl[0][2]), (um, "flank", fl[1][2])]
                    return True
        if cnt_["dorm_f"] > 1:
            cnt_["dorm_f"] -= 1
            return True
        if cnt_["chim"] > 1:
            cnt_["chim"] -= 1
            return True
        for k in range(n_up, 1, -1):
            for key in "fb":
                if any(w[1] == "centre" for w in plan[(key, k)]):
                    plan[(key, k)] = [w for w in plan[(key, k)] if w[1] != "centre"]
                    return True
        for key in "fb":
            fl = [w for w in plan[(key, 1)] if w[1] == "flank"]
            if len(fl) > 2:
                um = sum(abs(w[0]) for w in fl) / len(fl)
                plan[(key, 1)] = [w for w in plan[(key, 1)] if w[1] != "flank"] + [(-um, "flank", fl[0][2]), (um, "flank", fl[1][2])]
                return True
        if cnt_["dorm_f"] > 0:
            cnt_["dorm_f"] -= 1
            return True
        return False

    budget = float(META.get("tri_budget", 160000)) - 3000
    while estimate() > budget and thin_out():
        pass

    rt = rng.fork("timber")
    wf_eff = wf * (2.3 if shut else 1.0) + 0.3

    def timber_storey(f, z0, z1, wins):
        L = f.L
        tb, ts = 0.2, 0.125
        zb, zt = z0 + 0.24, z1 - 0.18
        zr = z0 + 0.74
        ops = []
        for (u, kind, lit) in wins:
            ops.append((u, {"flank": wf_eff, "centre": 1.9, "oriel": ow + 0.1}[kind], kind))

        def T(u, z, su, sz, proud=0.03, roll=0.0, grain="Z", thick=0.1, bev=0.007):
            o = fbox("timber", f, u, z, su, sz, thick=thick, proud=proud, roll=roll, bev=bev)
            uv_grain(o, grain)
            timber_objs.append(o)

        def in_open(u, margin=0.0):
            return any(abs(u - uc) < w / 2 + margin for (uc, w, kd) in ops)

        T(0, z0 + 0.12, L + 0.12, 0.24, proud=0.065, grain="X", thick=0.17, bev=0.012)          # bressummer
        T(0, z1 - 0.09, L + 0.07, 0.18, proud=0.04, grain="X", thick=0.12)                       # wall plate
        for sg in (-1, 1):
            T(sg * (L / 2 - tb / 2 + 0.035), (zb + zt) / 2, tb, zt - zb, proud=0.045, thick=0.12)   # corner posts
        T(0, zr, L - 0.2, 0.14, proud=0.032, grain="X")                                          # mid rail
        sp = {"braced": 0.62, "close_studding": 0.36, "square_panels": 1.05}[p.framing]
        if p.framing == "square_panels":
            T(0, (zr + zt) / 2, L - 0.2, 0.13, proud=0.031, grain="X")
        # braces in free end bays
        braced = {}
        for sg in (-1, 1):
            free = not in_open(sg * (L / 2 - 0.75), 0.6)
            braced[sg] = free and p.framing != "close_studding"
            if braced[sg]:
                run = 1.0 if p.framing == "braced" else 0.55
                za, zb2 = zr + 0.07, (zt if p.framing == "braced" else zr + 0.07 + 0.6)
                ua, ub = sg * (L / 2 - tb - run), sg * (L / 2 - tb + 0.02)
                if p.framing == "braced":
                    ua, ub = ub, ua                                 # foot at the corner post, head inwards
                ln = hypot(ub - ua, zb2 - za)
                T((ua + ub) / 2, (za + zb2) / 2, 0.15, ln, proud=0.036, roll=degrees(atan2(ub - ua, zb2 - za)))
        n = max(2, int(round((L - 2 * tb) / sp)))
        for i in range(1, n):
            u = -L / 2 + tb + (L - 2 * tb) * i / n + rt.u(-0.03, 0.03)
            w_ = ts * rt.vary(1.0, 0.12)
            rl = rt.u(-0.5, 0.5) * (0.5 + age)
            op = [o for o in ops if abs(u - o[0]) < o[1] / 2 + 0.05]
            in_brace = any(braced[sg] and sg * u > L / 2 - tb - 1.12 for sg in (-1, 1)) and p.framing == "braced"
            if not op or op[0][2] != "oriel":
                if op and p.framing == "braced" and op[0][2] != "oriel":
                    pass                                             # saltire goes here instead
                else:
                    T(u, (zb + zr - 0.07) / 2, w_, zr - 0.07 - zb, roll=rl)
            if op:
                ztop = z0 + w_sill + w_h + 0.2
                if op[0][2] != "oriel" and zt - ztop > 0.16:
                    T(u, (ztop + zt) / 2, w_, zt - ztop, roll=rl)
            elif not in_brace:
                T(u, (zr + 0.07 + zt) / 2, w_, zt - zr - 0.07, roll=rl)
        if p.framing == "braced":
            for (uc, w, kd) in ops:                                  # St Andrew's cross under each window
                if kd == "oriel":
                    continue
                cw = min(w, 1.3) - 0.1
                hh = zr - 0.07 - zb
                for sg in (-1, 1):
                    T(uc, zb + hh / 2, 0.11, hypot(cw, hh), proud=0.028 + 0.004 * sg, roll=sg * degrees(atan2(cw, hh)))
                for sg in (-1, 1):
                    T(uc + sg * (cw / 2 + 0.07), zb + hh / 2, ts, hh)

    for key in "frbl":
        f = F[key]
        for k in range(1, n_up + 1):
            z0 = G + (k - 1) * SH
            up_objs.append(wall("upper_wall", f, z0, z0 + SH, up_mat, cell=0.3 if p.walls == "timber" else 0.6))
            wins = plan[(key, k)]
            if p.walls == "timber":
                timber_storey(f, z0, z0 + SH, wins)
            for (u, kind, lit) in wins:
                if kind == "oriel":
                    put(oriel, f, 0.0, zo, "oriel")
                    for s in osock:
                        t = s[2].translation
                        pr = dict(s[3]) if len(s) > 3 and isinstance(s[3], dict) else {}
                        pr["intensity"] = min(pr.get("intensity", 14.0), 12.0)
                        lights.append((2 if key == "f" else 6, f"oriel_{key}", f.pt(t.x, zo + t.z, -t.y), pr))
                else:
                    put(casement(kind, lit), f, u, z0 + w_sill, "windows")
                    if kind == "flank" and shut and key in "fb":
                        shutters_at(f, u, z0 + w_sill, wf, w_h)
    if p.walls == "timber":
        pegs = []
        peg = cylinder("peg", 0.011, 0.02, verts=6)
        assign(peg, oak)
        for key in "frbl":
            f = F[key]
            for k in range(1, n_up + 1):
                z0 = G + (k - 1) * SH
                for sg in (-1, 1):
                    for (zz, du) in ((z0 + 0.3, 0.05), (z0 + 0.3, -0.04), (z0 + 0.74, 0.04), (z0 + 0.74, -0.05), (z0 + SH - 0.25, 0.05), (z0 + SH - 0.25, -0.04)):
                        pegs.append(Matrix.Translation(f.pt(sg * (f.L / 2 - 0.07) + du, zz, 0.05)) @ Matrix.Rotation(radians(f.th), 4, "Z")
                                    @ Matrix.Rotation(radians(90), 4, "X"))
        timber_objs.append(instances_to_mesh("pegs", peg, pegs, keep_base=False))

    # ------------------------------------------------------------------ roof, dormers, chimneys
    Mr = Matrix.Translation((0, 0, top)) @ Matrix.Rotation(radians(-90), 4, "Z")
    ro = roofc.place((0, 0, top), rot=(0, 0, -90), sockets=False)
    tri_log["roof"] = _tris(ro)
    dsock, csock, psock = [], [], []
    for s in roofc.sockets:
        role = s[3].get("role") if len(s) > 3 and isinstance(s[3], dict) else None
        if role == "dormer":
            dsock.append((Mr @ s[2], s[3]))
        elif role == "chimney":
            csock.append((Mr @ s[2], s[3]))
        elif role == "downpipe":
            psock.append(Mr @ s[2])
    rd = rng.fork("roof")
    ddir = [M.to_3x3() @ Vector((0, -1, 0)) for (M, pr) in dsock]
    has_y, has_x = any(abs(d.y) > 0.7 for d in ddir), any(abs(d.x) > 0.7 for d in ddir)
    ridge_w = "x" if (has_y and (p.ridge == "x" or not has_x)) else "y"      # hipped roofs run the long way
    main_dir = Vector((0, -1, 0)) if ridge_w == "x" else Vector((1, 0, 0))
    dorm = {}
    for sgn, cnt in ((1, cnt_["dorm_f"]), (-1, cnt_["dorm_b"])):
        dv = main_dir * sgn
        S_ = [(M, pr) for (M, pr) in dsock if (M.to_3x3() @ Vector((0, -1, 0))).dot(dv) > 0.7]
        if not S_ or cnt <= 0:
            continue
        ax = 0 if ridge_w == "x" else 1
        amax = max(abs(M.translation[ax]) for M, pr in S_)
        span = (W if ridge_w == "x" else D) / 2 - 1.25
        if p.roof_type != "gable":
            span = min(span, amax + 0.2)
        xf0 = min(flank_us[0] if ridge_w == "x" else D * 0.24, span)
        if cnt == 3 and xf0 < 1.5:
            cnt = 2
        pos = {1: [0.0], 2: [-xf0, xf0], 3: [-max(xf0, 1.7), 0.0, max(xf0, 1.7)]}[cnt]
        pos = [clamp(a, -span, span) for a in pos]
        M0, pr0 = S_[0]
        for a in pos:
            lit = rw.chance(p.lit_ratio)
            if lit not in dorm:
                dorm[lit] = use("dormer", width=1.15, height=1.7, pitch=float(pr0.get("pitch", p.pitch)), roof="gable" if p.roof_type == "gable" else "hip",
                                cover={"roof_tile_clay": "tile", "roof_slate": "slate", "wood_shingle": "lead"}[p.roof],
                                cladding="plaster" if p.walls in ("timber", "plaster") else "lead", lit=lit, intensity=0.0, flower_box=False,
                                timber=p.timber_color, paint=p.paint, plaster=p.wall_color, age=age, ajar=0.2, seed=sd + 70)
            M = M0.copy()
            t = M0.translation.copy()
            t[ax] = a
            M.translation = t
            o = dorm[lit].place_matrix(M, sockets=False)
            tri_log["dormers"] = tri_log.get("dormers", 0) + _tris(o)
    if csock and cnt_["chim"] > 0:
        csock.sort(key=lambda c: (round(c[0].translation.x, 2), round(c[0].translation.y, 2)))
        picks = [csock[rd.pick(range(len(csock)))]]
        if cnt_["chim"] > 1 and len(csock) > 1:
            far = max(csock, key=lambda c: (c[0].translation - picks[0][0].translation).length)
            picks.append(far)
        for i, (M, pr) in enumerate(picks):
            sink = 0.6
            ch_ = use("chimney", width=0.9 if i == 0 else 0.7, depth=0.56, height=2.3 if i == 0 else 1.9,
                      material="stone" if p.walls == "stone" else "brick_red", pots=2 if i == 0 else 1, pitch=float(pr.get("pitch", p.pitch)),
                      sink=sink, roof_stub="none", soot=0.4 + 0.5 * age, moss=0.2 + 0.5 * age, lean=rd.u(-1.5, 1.5), seed=sd + 80 + i)
            o = ch_.place_matrix(M @ Matrix.Translation((0, 0, -sink)), sockets={"fx"})
            tri_log["chimney"] = tri_log.get("chimney", 0) + _tris(o)

    # ------------------------------------------------------------------ rainwater pipes, wall anchors, barred slits
    zinc = iron
    pipes = []
    for M in psock:
        pos = M.translation
        hit = None
        for key, f in F.items():
            dx, dy = pos.x - f.cx, pos.y - f.cy
            out, u = dx * f.nx + dy * f.ny, dx * f.ux + dy * f.uy
            if 0.15 < out < 1.3 and abs(u) < f.L / 2 + 0.3:
                hit = (key, f, u, out)
        if hit is None:
            continue
        key, f, u, out = hit
        sg = 1 if u >= 0 else -1
        u0 = sg * (f.L / 2 - 0.21)
        wext = (wf + 0.12) if (key in "fb" and shut) else (wf / 2 + 0.16)
        if any(kind != "oriel" and uw * sg > 0 and abs(uw) + wext > f.L / 2 - 0.27 for k in range(1, n_up + 1) for (uw, kind, lit) in plan[(key, k)]):
            continue
        zg, so_ = pos.z, 0.17
        path = [f.pt(u0, zg - 0.03, out), f.pt(u0, zg - 0.2, out), f.pt(u0, zg - 0.2 - (out - so_) * 1.15, so_), f.pt(u0, 0.5, so_),
                f.pt(u0, 0.3, so_ + 0.04), f.pt(u0 - sg * 0.0, 0.16, so_ + 0.2)]
        prof = [(0.04 * cos(2 * pi * i / 8), 0.04 * sin(2 * pi * i / 8)) for i in range(8)]
        pp = sweep("downpipe", path, prof, smooth_path=False, up=(f.ux, f.uy, 0))
        assign(pp, zinc)
        pipes.append(pp)
        zb_ = zg - 0.2 - (out - so_) * 1.15 - 0.35
        nb_ = max(2, int((zb_ - 0.9) / 2.3) + 1)
        for i in range(nb_):
            zz = lerp(zb_, 0.9, i / (nb_ - 1))
            hb = fbox("pipe_bat", f, u0, zz, 0.115, 0.03, thick=0.225, proud=0.22, bev=0.004)
            assign(hb, iron)
            iron_objs.append(hb)
            sk = fbox("pipe_socket", f, u0, zz + 0.06, 0.1, 0.07, thick=0.1, proud=0.22, bev=0.01)
            assign(sk, zinc)
            pipes.append(sk)
    for o in pipes:
        paint(o, grime=mask_mul(noise_mask(0.5, seed=sd + 21, lo=0.3, hi=0.8), 0.3 + 0.4 * age),
              special=mask_mul(height_ramp(0.0, 0.9), noise_mask(0.3, seed=sd + 22, lo=0.35, hi=0.8), 0.3 + 0.4 * age))

    def anchor(f, u, z, proud=0.016):
        for sg in (-1, 1):
            a_ = fbox("wall_anchor", f, u, z, 0.042, 0.52, thick=0.012, proud=proud + 0.004 * (sg + 1), roll=sg * 36.0, bev=0.003)
            assign(a_, iron)
            iron_objs.append(a_)
        b_ = bolt("anchor_bolt", 0.016, 0.012, loc=f.pt(u, z, proud + 0.008), rot=(90, 0, f.th), kind="square")
        assign(b_, iron)
        iron_objs.append(b_)

    ra = rng.fork("anchors")
    for key in "lr":
        f = F[key]
        for sg in (-1, 1):
            anchor(f, sg * f.L * 0.36 + ra.u(-0.1, 0.1), G - 0.62)
            if p.walls != "timber":
                for k in range(2, n_up + 1):
                    anchor(f, sg * f.L * 0.4 + ra.u(-0.08, 0.08), G + (k - 1) * SH - 0.32)
        # barred slit lighting the cellar stair / store behind the passage wall
        u = ra.u(-0.12, 0.12) * f.L + (0.18 * f.L if key == "l" else -0.18 * f.L)
        w_, h_, z0_ = 0.4, 0.8, 1.55
        sl = [fbox("slit_sill", f, u, z0_ - 0.07, w_ + 0.36, 0.14, thick=0.18, proud=0.08, bev=0.012),
              fbox("slit_lintel", f, u, z0_ + h_ + 0.09, w_ + 0.44, 0.18, thick=0.15, proud=0.05, bev=0.012)]
        for sg in (-1, 1):
            sl.append(fbox("slit_jamb", f, u + sg * (w_ / 2 + 0.085), z0_ + h_ * 0.3, 0.17, h_ * 0.6, thick=0.14, proud=0.045, bev=0.012))
            sl.append(fbox("slit_jamb", f, u + sg * (w_ / 2 + 0.065), z0_ + h_ * 0.8, 0.13, h_ * 0.4, thick=0.14, proud=0.04, bev=0.012))
        for o in sl:
            assign(o, st)
        stone_objs.append(join(sl, "slit_stone"))
        dk = fbox("slit_dark", f, u, z0_ + h_ / 2, w_ + 0.02, h_ + 0.02, thick=0.02, proud=0.008, bev=0.001)
        assign(dk, dark_m)
        for du in (-w_ / 6, w_ / 6):
            b_ = fbox("slit_bar", f, u + du, z0_ + h_ / 2, 0.02, h_ + 0.04, thick=0.02, proud=0.04, bev=0.004)
            assign(b_, iron)
            iron_objs.append(b_)
        b_ = fbox("slit_bar", f, u, z0_ + h_ * 0.55, w_ + 0.06, 0.035, thick=0.012, proud=0.046, bev=0.003)
        assign(b_, iron)
        iron_objs.append(b_)

    # ------------------------------------------------------------------ materials of the modelled parts and weathering
    for o in timber_objs:
        if not o.data.materials:
            assign(o, oak)
    for o in iron_objs:
        # ironwork under the vault stays dry: far less rust than the brackets and anchors out in the weather
        cy = sum((o.matrix_world @ Vector(c)).y for c in o.bound_box) / 8.0
        paint(o, special=mask_mul(noise_mask(0.22, seed=sd + 23, lo=0.5, hi=0.9), 0.22 + 0.3 * age) if abs(cy) > D / 2 else (0.03 + 0.06 * age))
    tim = join([o for o in timber_objs], "carpentry") if timber_objs else None
    north = facing((0, 1, 0))
    damp = mask_mul(height_ramp(0.0, 1.2), noise_mask(0.8, seed=sd + 5, lo=0.35, hi=0.8), 0.25 + 0.55 * age)
    splash = mask_mul(height_ramp(0.0, 0.7), 0.38)
    for o in wall_objs + stone_objs:
        paint(o, grime=splash, special=damp)
        paint(o, grime=mask_mul(noise_mask(2.2, seed=sd + 7, lo=0.4, hi=0.9), 0.1 + 0.2 * age),
              special=mask_mul(north, height_ramp(0.0, 2.6), 0.45 * age))
    if vault is not None:
        def soot(co, n):
            m = np.zeros(len(co))
            for (sx_, sy_, sr_) in soot_at:
                m = np.maximum(m, np.clip(1.0 - np.hypot(co[:, 0] - sx_, co[:, 1] - sy_) / sr_, 0.0, 1.0))
            return m * (0.85 if timber_v or p.soffit != "limewash" else 0.6)

        if timber_v:                                  # dry oak under a roof: smoke-browned, darkest over the lanterns
            sm_ = mask_mul(noise_mask(1.1, seed=sd + 9, lo=0.3, hi=0.8), 0.2 + 0.25 * age)
            for o in [vault] + vault_tim:
                paint(o, grime=lambda co, n: np.maximum(soot(co, n), sm_(co, n)))
            paint(vault_tim[0], wear=mask_mul(noise_mask(0.25, seed=sd + 15, lo=0.6, hi=0.85), 0.2 + 0.2 * age))
        else:
            paint(vault, grime=soot, special=mask_mul(noise_mask(0.9, seed=sd + 9, lo=0.5, hi=0.85), (0.1 if p.soffit == "limewash" else 0.2) + 0.25 * age))
    for o in shutter_objs:
        paint(o, wear=mask_mul(noise_mask(0.2, seed=sd + 16, lo=0.45, hi=0.8), 0.35 + 0.5 * age),
              grime=mask_mul(noise_mask(0.5, seed=sd + 17, lo=0.3, hi=0.8), 0.3 + 0.3 * age))
    for o in up_objs:
        paint(o, grime=mask_mul(noise_mask(2.0, seed=sd + 11, lo=0.45, hi=0.9), 0.06 + 0.14 * age),
              special=mask_mul(height_ramp(G, G + 0.9), noise_mask(0.6, seed=sd + 12, lo=0.4, hi=0.8), 0.25 * age),
              wear=mask_mul(noise_mask(0.5, seed=sd + 13, lo=0.66, hi=0.85), 0.3 * age))
    if tim is not None:
        paint(tim, grime=mask_mul(noise_mask(0.8, seed=sd + 14, lo=0.3, hi=0.8), 0.25 + 0.3 * age),
              wear=mask_mul(noise_mask(0.25, seed=sd + 15, lo=0.55, hi=0.8), 0.3 + 0.3 * age))

    # ------------------------------------------------------------------ one material per family (= one draw call each)
    canon = {"ashlar_limestone": st_wall, "rubble_fieldstone": st_wall, "stone_smooth": st, "stone_rough": st, "mortar": st, "slate": st,
             "plaster_lime": plaster, "plaster_rough": plaster, "paper": plaster,
             "brick_red": vault_m if p.soffit == "ochre_brick" else brick, "brick_ochre": vault_m if p.soffit == "ochre_brick" else brick,
             "wood_oak": oak, "wood_weathered": oak, "wood_endgrain": oak, "wood_planks": oak, "wicker": oak,
             "wood_painted": shutter_m, "iron_wrought": iron, "iron_painted": iron, "iron_rusty": iron, "zinc": iron, "lead": iron,
             "brass": iron, "bronze_aged": iron, "metal_plain": iron,
             "window_lit": lit_m, "glass_stained": lit_m, "glass_lantern": lit_m, "window_dark": dark_m, "glass_clear": dark_m,
             "terracotta": pot_m, "ceramic_glazed": pot_m, "emissive": glow_m, "neon": glow_m}
    keep = {m_.name for m_ in (awn_1, ban_m)}
    soft = {"zinc", "lead", "brass", "bronze_aged", "metal_plain", "iron_painted"}      # their 'special' mask is not rust
    seen = set()
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or o.data.name in seen:
            continue
        seen.add(o.data.name)
        me = o.data
        calm = []
        for i, m_ in enumerate(me.materials):
            if m_ is None or m_.name in keep:
                continue
            lib = m_.name.split("~")[0]
            tgt = canon.setdefault(lib, m_)
            if tgt.name != m_.name:
                me.materials[i] = tgt
                if lib in soft or lib == "iron_wrought":
                    calm.append((i, 0.0 if lib in soft else 0.45))
        att = me.attributes.get("m_special")
        if calm and att is not None and len(me.polygons):
            npoly = len(me.polygons)
            pm, lt = np.zeros(npoly, dtype=np.int32), np.zeros(npoly, dtype=np.int32)
            me.polygons.foreach_get("material_index", pm)
            me.polygons.foreach_get("loop_total", lt)
            lv = np.zeros(len(me.loops), dtype=np.int32)
            me.loops.foreach_get("vertex_index", lv)
            lm = np.repeat(pm, lt)
            val = np.zeros(len(me.vertices), dtype=np.float32)
            att.data.foreach_get("value", val)
            for i, k_ in calm:
                val[lv[lm == i]] *= k_
            att.data.foreach_set("value", val)

    # ------------------------------------------------------------------ lights (the max_lights first by priority) and sockets
    lights.sort(key=lambda c: c[0])
    for pr_, nm, loc, kw in lights[:int(p.max_lights)]:
        light(nm, loc, **kw)
    socket("sign_front", "attach", F["f"].pt(xP, min(G - 0.5, 3.9)), rot=(0, 0, 0), role="sign")
    socket("sign_back", "attach", F["b"].pt(xP, min(G - 0.5, 3.9)), rot=(0, 0, 180), role="sign")
    socket("banner_front", "attach", F["f"].pt(0.0, top - 0.6), rot=(0, 0, 0), role="banner")
