"""Corner shop ("Items / Accessories") for the lantern-lit square.

Construction, ground up: a masonry ground storey (brick / ashlar / roughcast) on a dressed-stone plinth with
alternating quoins carries two joiner-made shopfronts (front and +X street face, fascia lettering is a param)
and a chamfered corner with the glazed shop door.  A moulded oak bressummer runs round the building; above it
the upper storeys are oak box-framing (sill beam, posts, rails, studs, braces / St Andrew's crosses, pegged)
over lime-plaster infill, with leaded casements and plank shutters.  Over the corner door a turret is
corbelled out on a moulded stone corbel: timber ring beams, a ribbon of leaded lights, a lantern storey of
small lit panes and a steep conical / octagonal / bell roof covered in individually laid slates, shingles or
tiles (or seamed copper) with a lead cap, gilt ball and a turning weather vane.  A neon trade emblem on an iron
board is fixed to the turret above the door, a forged hanging sign projects beside the front shopfront.  The
main roof, dormers, chimney, shopfronts, door, street-side casements and sign are composed from catalogue
assets; the roof is trimmed where it dies into the turret.  Back and side walls are closed, framed and windowed,
with a plank back door, lantern and downpipes.  Six lights: two display spills, two fascia lamps, door, neon.
"""
import json
import math
from math import cos, sin, pi, radians, degrees, atan2, hypot

import bmesh
import bpy
from mathutils import Matrix, Vector

from tk import *

META = {"name": "Corner shop (Items / Accessories)", "class": "building", "mount": "ground",
        "tri_budget": 200000,          # composed hero building: two full shopfronts, door, roof, dormers, chimney
        "focus": [3.7, -0.6, 3.3]}
PARAMS = {
    "width": {"default": 9.0, "min": 7.0, "max": 11.0, "desc": "front width along X (m); ground walls span +-width/2"},
    "depth": {"default": 9.0, "min": 7.0, "max": 11.0, "desc": "depth along +Y (m)"},
    "storeys": {"default": 2, "min": 2, "max": 3, "desc": "storeys below the eaves (ground shop storey + 1 or 2 framed storeys)"},
    "sign_front": {"default": "Items", "desc": "fascia lettering of the front shopfront"},
    "sign_side": {"default": "Accessories", "desc": "fascia lettering of the +X shopfront"},
    "sign_hang": {"default": "ITEMS", "desc": "lettering on the hanging sign ('' = emblem only)"},
    "hang_emblem": {"default": "bottle", "choices": ["bottle", "key", "star", "suitcase", "book", "boot", "cup", "scissors"],
                    "desc": "trade emblem on the hanging sign"},
    "emblem": {"default": "potion", "choices": ["potion", "star", "gem", "crown"], "desc": "neon emblem over the corner door"},
    "emblem_size": {"default": 0.8, "min": 0.6, "max": 0.95, "desc": "diameter of the neon emblem board (m)"},
    "neon": {"default": "#37e0ff", "desc": "neon tube colour (outline)"},
    "neon2": {"default": "#ffd24a", "desc": "neon tube colour (accents)"},
    "paint": {"default": "#7a1d1a", "desc": "paint of shopfronts, shop door and shutters"},
    "trim": {"default": "#cfc6ac", "desc": "paint of window frames, doorcase and turret eaves"},
    "plaster": {"default": "#e4d6b4", "desc": "limewash colour of the infill panels"},
    "timber": {"default": "#5b4433", "desc": "stain of the oak framing"},
    "wall": {"default": "brick_red", "choices": ["brick_red", "brick_ochre", "ashlar", "plaster"], "desc": "ground-storey masonry"},
    "framing": {"default": "cross", "choices": ["cross", "close", "square", "brace"], "desc": "pattern of the timber framing"},
    "shutters": {"default": True, "desc": "plank shutters on the street-side upper windows (where they fit)"},
    "turret": {"default": "cone", "choices": ["cone", "octagon", "bell"], "desc": "corner turret: round with conical roof, octagonal spire, round with bell roof"},
    "turret_cover": {"default": "slate", "choices": ["slate", "shingle", "tile", "copper"], "desc": "turret roof covering"},
    "turret_size": {"default": 1.35, "min": 1.2, "max": 1.5, "desc": "turret radius (m)"},
    "turret_rise": {"default": 1.9, "min": 1.0, "max": 2.6, "desc": "height of the turret wall above the main eaves (m); never less than needed to clear the main roof"},
    "turret_spire": {"default": 3.0, "min": 1.8, "max": 3.6, "desc": "turret roof height as a multiple of its radius"},
    "roof_cover": {"default": "roof_tile_clay", "choices": ["roof_tile_clay", "roof_slate"], "desc": "main roof covering"},
    "roof_pitch": {"default": 46.0, "min": 40.0, "max": 55.0, "desc": "main roof pitch (deg)"},
    "dormers": {"default": 2, "min": 0, "max": 3, "desc": "dormers on the front slope"},
    "chimneys": {"default": 1, "min": 0, "max": 2, "desc": "chimney stacks on the back slope"},
    "goods_front": {"default": "bottles", "choices": ["luggage", "hats", "bottles", "books", "cakes"], "desc": "display in the front shopfront"},
    "goods_side": {"default": "luggage", "choices": ["luggage", "hats", "bottles", "books", "cakes"], "desc": "display in the side shopfront"},
    "lit": {"default": 0.65, "min": 0.0, "max": 1.0, "desc": "fraction of upper / rear windows that are lit"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: grime, damp, moss, chipped paint"},
}
VARIANTS = {
    "tall_octagon": {"width": 8.0, "depth": 8.0, "storeys": 3, "turret": "octagon", "turret_cover": "shingle", "turret_rise": 1.9,
                     "paint": "#1f4a3a", "framing": "close", "wall": "ashlar", "emblem": "star", "neon": "#ff3b6b", "neon2": "#ffe9a0",
                     "plaster": "#e9e1cc", "timber": "#3d2f26", "dormers": 1, "hang_emblem": "key", "seed": 7},
    "wide_bell_copper": {"width": 11.0, "depth": 8.0, "turret": "bell", "turret_cover": "copper", "turret_size": 1.5, "turret_spire": 2.4,
                         "roof_cover": "roof_slate", "paint": "#1c2f4a", "plaster": "#d5d9d2", "timber": "#4a4039", "framing": "square",
                         "emblem": "gem", "neon": "#ffd24a", "neon2": "#ff6ad5", "sign_front": "Accessories", "sign_side": "Items",
                         "goods_front": "hats", "goods_side": "luggage", "dormers": 3, "chimneys": 2, "hang_emblem": "star", "seed": 12},
    "narrow_ochre": {"width": 7.0, "depth": 10.0, "wall": "brick_ochre", "plaster": "#d9b56e", "paint": "#1f5560", "trim": "#e4d9bd",
                     "framing": "brace", "turret_cover": "tile", "turret_size": 1.25, "turret_rise": 1.0, "turret_spire": 3.3,
                     "emblem": "crown", "neon": "#7dff8a", "neon2": "#ffd24a", "sign_front": "Potions", "sign_side": "Charms",
                     "goods_front": "bottles", "goods_side": "books", "dormers": 1, "roof_pitch": 52.0, "hang_emblem": "book", "seed": 21},
    "old_dark_three": {"width": 10.0, "depth": 9.0, "storeys": 3, "wall": "plaster", "plaster": "#b9b3a2", "timber": "#2e2622",
                       "paint": "#3b2a4a", "framing": "brace", "turret_cover": "shingle", "turret_rise": 1.1, "roof_cover": "roof_slate",
                       "age": 0.95, "lit": 0.35, "shutters": False, "emblem": "potion", "neon": "#b07cff", "neon2": "#7dffd8",
                       "sign_front": "Curios", "sign_side": "Relics", "goods_front": "books", "goods_side": "luggage", "seed": 33},
}

S22 = cos(radians(22.5))


class Frame:
    """A wall in plan: origin at its left end (seen from outside), s runs to the right, `out` along the normal."""

    def __init__(self, origin, n):
        l = hypot(n[0], n[1])
        self.o = (float(origin[0]), float(origin[1]))
        self.n = (n[0] / l, n[1] / l)
        self.d = (-self.n[1], self.n[0])
        self.theta = degrees(atan2(self.n[0], -self.n[1]))

    def pt(self, s=0.0, out=0.0, z=0.0):
        return (self.o[0] + self.d[0] * s + self.n[0] * out, self.o[1] + self.d[1] * s + self.n[1] * out, z)


class Timbers:
    """Collects the framing of one wall: bevelled boxes with grain along their length, plus oak pegs."""

    def __init__(self, rng, irr):
        self.objs, self.pegs, self.rng, self.irr = [], [], rng, irr

    def tb(self, fr, s, z, length, face, phi=0.0, rec=0.0, thick=0.07):
        r = self.rng
        face = face * r.vary(1.0, 0.07 * self.irr)
        phi = phi + r.jitter(0.0, 0.45 * self.irr)
        proud = 0.042 - rec
        o = box("tb", (length, thick, face), loc=fr.pt(s, proud - thick / 2, z), rot=(0.0, -phi, fr.theta), bevel=0.009, segments=1)
        uv_grain(o, "X")
        self.objs.append(o)
        return o

    def peg(self, fr, s, z):
        self.pegs.append(trs(fr.pt(s, 0.04, z), (90.0, 0.0, fr.theta)))

    def finish(self, name, oak):
        out = []
        if self.objs:
            j = join(self.objs, name)
            assign(j, oak)
            out.append(j)
        if self.pegs:
            pg = cylinder("peg_base", 0.013, 0.02, verts=6)
            k = instances_to_mesh(name + "_pegs", pg, self.pegs, keep_base=False)
            assign(k, oak)
            out.append(k)
        self.objs, self.pegs = [], []
        return out


def frame_storey(T, fr, sa, sb, zb, Hs, wins, pattern, post_a=True, post_b=True):
    """Box frame of one storey between s = sa..sb: sill beam, plate, posts, sill rail and pattern infill.
    wins = [(centre_s, half_width_kept_clear)]."""
    beam, pw, sf = 0.2, 0.2, 0.12
    L = sb - sa
    T.tb(fr, (sa + sb) / 2, zb + beam / 2, L, beam)
    T.tb(fr, (sa + sb) / 2, zb + Hs - beam / 2, L, beam)
    z0, z1 = zb + beam, zb + Hs - beam
    cs = sorted(w[0] for w in wins)
    if len(cs) > 1:
        inner = [(cs[i] + cs[i + 1]) / 2 for i in range(len(cs) - 1)]
    elif cs:
        inner = []
    else:
        n = max(1, round(L / 2.4))
        inner = [sa + L * i / n for i in range(1, n)]
    E = [sa + pw / 2] + inner + [sb - pw / 2]
    for i, s in enumerate(E):
        if (i == 0 and not post_a) or (i == len(E) - 1 and not post_b):
            continue
        T.tb(fr, s, (z0 + z1) / 2, z1 - z0, pw * (1.1 if i in (0, len(E) - 1) else 1.0), phi=90.0)
        for dz in (zb + beam / 2, zb + Hs - beam / 2):
            T.peg(fr, s - 0.045, dz)
            T.peg(fr, s + 0.045, dz)
    zr = zb + 0.87

    def blocked(s, half=0.08):
        return any(abs(s - w[0]) < w[1] + half for w in wins)

    def studs(a, b, za, zc, spacing, check):
        n = max(1, round((b - a) / spacing))
        for k in range(1, n):
            s = a + (b - a) * k / n
            if check and blocked(s):
                continue
            T.tb(fr, s, (za + zc) / 2, zc - za, sf, phi=90.0, rec=0.006)

    nb = len(E) - 1
    for bi in range(nb):
        a, b = E[bi] + pw / 2, E[bi + 1] - pw / 2
        if b - a < 0.15:
            continue
        T.tb(fr, (a + b) / 2, zr, b - a, 0.14)
        T.peg(fr, a - 0.08, zr)
        T.peg(fr, b + 0.08, zr)
        zl0, zl1 = z0, zr - 0.07
        zu0, zu1 = zr + 0.07, z1
        hl, hu = zl1 - zl0, zu1 - zu0
        if pattern == "cross":
            n = max(1, round((b - a) / 0.78))
            cw = (b - a) / n
            for k in range(1, n):
                T.tb(fr, a + cw * k, (zl0 + zl1) / 2, hl, sf, phi=90.0, rec=0.006)
            for k in range(n):
                dx = cw - sf
                ang = degrees(atan2(hl, dx))
                ln = hypot(dx, hl) * 0.99
                T.tb(fr, a + cw * (k + 0.5), (zl0 + zl1) / 2, ln, 0.095, phi=ang, rec=0.014)
                T.tb(fr, a + cw * (k + 0.5), (zl0 + zl1) / 2, ln, 0.095, phi=-ang, rec=0.02)
            studs(a, b, zu0, zu1, 0.62, True)
        elif pattern == "close":
            studs(a, b, zl0, zl1, 0.36, False)
            studs(a, b, zu0, zu1, 0.36, True)
        elif pattern == "square":
            studs(a, b, zl0, zl1, 0.8, False)
            studs(a, b, zu0, zu1, 0.8, True)
            zm = zu0 + hu * 0.5
            cuts = [a]
            for w in sorted(wins):
                lo, hi = w[0] - w[1] - 0.02, w[0] + w[1] + 0.02
                if hi > a and lo < b:
                    cuts += [max(a, lo), min(b, hi)]
            cuts.append(b)
            for k in range(0, len(cuts) - 1, 2):
                if cuts[k + 1] - cuts[k] > 0.25:
                    T.tb(fr, (cuts[k] + cuts[k + 1]) / 2, zm, cuts[k + 1] - cuts[k], 0.11, rec=0.012)
        else:  # brace
            studs(a, b, zl0, zl1, 0.55, False)
            studs(a, b, zu0, zu1, 0.55, True)
            for side in ((-1,) if bi == 0 else ()) + ((1,) if bi == nb - 1 else ()):
                edge = a if side < 0 else b
                free = b - a
                for w in wins:
                    d = (w[0] - w[1] - edge) if side < 0 else (edge - (w[0] + w[1]))
                    if d > 0:
                        free = min(free, d)
                    elif abs(w[0] - edge) < w[1]:
                        free = 0.0
                if free >= 0.6:
                    bw = min(free - 0.04, 1.25)
                    bh = hu - 0.05
                    ang = degrees(atan2(bh, bw))
                    T.tb(fr, edge - side * bw / 2, zu0 + bh / 2, hypot(bw, bh) * 0.985, 0.15, phi=ang if side < 0 else -ang, rec=-0.004,
                         thick=0.075)


def make_window(name, w, h, glass, M, surround="oak", bars=(2, 3), rng=None):
    """Light casement drawn in the wall frame (x along the wall, -y out, z = 0 at the sill line). Returns one mesh."""
    parts = []

    def bx(nm, size, loc, m, bev=0.006, grain=None):
        o = box(nm, size, loc=loc, bevel=bev, segments=1)
        assign(o, m)
        if grain:
            uv_grain(o, grain)
        parts.append(o)
        return o

    # CLEARANCE RULE: the glowing glass is one sheet (the front of the old slab, 5 mm in front of the backs of the
    # frame and the bars): a slab had its ends, top and bottom lying in the jambs, the lintel and the sill
    g = lod_faces(box("glass", (w, 0.012, h), loc=(0, -0.014, h / 2)), keep=(0, -1, 0), above=-1)
    assign(g, glass)
    parts.append(g)
    fw = 0.055
    for sx in (-1, 1):
        bx("fr_j", (fw, 0.05, h), (sx * (w / 2 - fw / 2), -0.04, h / 2), M["trim"], grain="Z")
    bx("fr_t", (w, 0.05, fw), (0, -0.04, h - fw / 2), M["trim"], grain="X")
    bx("fr_b", (w, 0.055, fw * 1.2), (0, -0.042, fw * 0.6), M["trim"], grain="X")
    nx, ny = bars
    for i in range(1, nx):
        bx("mull", (0.042, 0.05, h), (-w / 2 + w * i / nx, -0.04, h / 2), M["trim"], grain="Z")
    for j in range(1, ny):
        bx("bar", (w, 0.03, 0.024), (0, -0.03, h * j / ny), M["trim"], bev=0.003, grain="X")
    if surround == "oak":
        bx("lintel", (w + 0.38, 0.12, 0.17), (0, -0.055, h + 0.085), M["oak"], bev=0.012, grain="X")
        for sx in (-1, 1):
            bx("jamb", (0.11, 0.1, h), (sx * (w / 2 + 0.055), -0.045, h / 2), M["oak"], bev=0.01, grain="Z")
        bx("sill", (w + 0.32, 0.18, 0.075), (0, -0.08, -0.037), M["oak"], bev=0.012, grain="X")
    else:
        lt = stone_block("lintel", (w + 0.44, 0.15, 0.26), loc=(0, -0.04, h + 0.13), rng=rng, chamfer=0.014, detail=0.3)
        sl = stone_block("sill", (w + 0.3, 0.22, 0.11), loc=(0, -0.08, -0.055), rng=rng, chamfer=0.014, detail=0.3)
        for o in (lt, sl):
            assign(o, M["stone"])
            parts.append(o)
    return join(parts, name)


def make_shutters(name, hx, h, M, rng, iron):
    """Pair of ledged plank shutters folded back against the wall either side of a window of half-width hx."""
    parts = []
    lw = 0.46
    for sx in (-1, 1):
        cx = sx * (hx + lw / 2 + 0.02)
        lean = rng.jitter(0.0, 0.6)
        for k in range(3):
            bw = lw / 3
            o = box("sh_board", (bw - 0.005, 0.022, h - rng.u(0.0, 0.015)), loc=(cx + (k - 1) * bw, -0.062, h / 2), rot=(0, lean, 0), bevel=0.004,
                    segments=1)
            assign(o, M["paint"])
            uv_grain(o, "Z")
            parts.append(o)
        for zf in (0.16, 0.84):
            o = box("sh_ledge", (lw - 0.04, 0.018, 0.085), loc=(cx, -0.08, h * zf), rot=(0, lean, 0), bevel=0.004, segments=1)
            assign(o, M["paint"])
            uv_grain(o, "X")
            parts.append(o)
            hg = box("sh_hinge", (0.26, 0.008, 0.035), loc=(cx - sx * (lw / 2 - 0.1), -0.092, h * zf), bevel=0.002, segments=1)
            assign(hg, iron)
            parts.append(hg)
    return join(parts, name)


def make_back_door(name, M, rng, iron):
    parts = []
    dw, dh = 1.0, 2.05
    for sx in (-1, 1):
        o = stone_block("jamb", (0.2, 0.2, dh), loc=(sx * (dw / 2 + 0.1), -0.05, dh / 2), rng=rng, chamfer=0.016, detail=0.4)
        assign(o, M["stone"])
        parts.append(o)
    o = stone_block("lintel", (dw + 0.56, 0.22, 0.28), loc=(0, -0.055, dh + 0.14), rng=rng, chamfer=0.018, detail=0.35)
    assign(o, M["stone"])
    parts.append(o)
    o = stone_block("step", (dw + 0.5, 0.46, 0.15), loc=(0, -0.2, 0.075), rng=rng, chamfer=0.02, detail=0.3)
    assign(o, M["stone"])
    parts.append(o)
    nb = 5
    for k in range(nb):
        bw = dw / nb
        o = box("board", (bw - 0.005, 0.04, dh - 0.16), loc=(-dw / 2 + bw * (k + 0.5), -0.03, 0.15 + (dh - 0.16) / 2), bevel=0.005, segments=1)
        assign(o, M["paint"])
        uv_grain(o, "Z")
        parts.append(o)
    for zf in (0.5, 1.72):
        o = box("strap", (0.62, 0.01, 0.05), loc=(-dw / 2 + 0.33, -0.055, zf), bevel=0.003, segments=1)
        assign(o, iron)
        parts.append(o)
    o = torus("ring", 0.05, 0.008, loc=(dw / 2 - 0.16, -0.062, 1.08), rot=(90, 0, 0), major_seg=14, minor_seg=6)
    assign(o, iron)
    parts.append(o)
    return join(parts, name)


def make_lantern(name, iron, flame, glass):
    """Small forged wall lantern; origin at the wall fixing, lantern axis 0.3 m out."""
    parts = []

    def add(o, m):
        assign(o, m)
        parts.append(o)

    add(box("plate", (0.07, 0.012, 0.34), loc=(0, -0.006, 0.02), bevel=0.003, segments=1), iron)
    add(sweep("arm", [(0, -0.01, 0.12), (0, -0.14, 0.2), (0, -0.27, 0.17), (0, -0.3, 0.1)], 0.011, sides=6), iron)
    add(sweep("stay", [(0, -0.01, -0.1), (0, -0.1, -0.02), (0, -0.17, 0.12), (0, -0.2, 0.185)], 0.008, sides=5), iron)
    add(cone("cap", 0.105, 0.02, 0.09, loc=(0, -0.3, 0.055), verts=4, rot=(0, 0, 45)), iron)
    add(sphere("knob", 0.018, loc=(0, -0.3, 0.115), segments=8, rings=5), iron)
    for sx in (-1, 1):
        for sy in (-1, 1):
            add(box("rod", (0.012, 0.012, 0.22), loc=(sx * 0.066, -0.3 + sy * 0.066, -0.1)), iron)
    add(box("base", (0.15, 0.15, 0.014), loc=(0, -0.3, -0.215), bevel=0.003, segments=1), iron)
    # CLEARANCE RULE: four panes inside the rods, no glowing lid 3 mm over the base plate
    add(lod_faces(box("glass", (0.118, 0.118, 0.21), loc=(0, -0.3, -0.1)), drop=[(0, 0, 1), (0, 0, -1)], above=-1), glass)
    add(sphere("flame", 0.026, loc=(0, -0.3, -0.13), segments=8, rings=6, scale=(1, 1, 1.5)), flame)
    return join(parts, name)


def stamp(tmpl, fr, s, z, out=0.0):
    o = duplicate(tmpl)
    place(o, loc=fr.pt(s, out, z), rot=(0, 0, fr.theta))
    return o


def clip_quadrant(o, x0, y0):
    """Trim a (transform-applied) mesh: remove everything with x > x0 and y < y0 (where the roof dies into the turret)."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    for co, no in (((x0, 0, 0), (1, 0, 0)), ((0, y0, 0), (0, 1, 0))):
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=co, plane_no=no)
    kill = []
    for f in bm.faces:
        c = f.calc_center_median()
        if c.x > x0 and c.y < y0:
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context='FACES')
    bm.to_mesh(o.data)
    bm.free()


def consolidate(seed_mats):
    """Composed assets bring dozens of near-identical material variants; fold them onto one palette."""
    strict = {"emissive", "neon", "glass_stained", "window_lit", "glass_lantern", "glass_frosted", "canvas", "glass_clear", "window_dark"}
    painted = {"wood_painted", "iron_painted", "ceramic_glazed", "plain", "metal_plain", "leaf", "paper"}
    canon = {}

    def srgb(m):
        t = m.get("tsq_tint")
        t = (1.0, 1.0, 1.0) if t is None else tuple(t)
        return tuple(max(c, 0.0) ** (1 / 2.2) for c in t)

    def canon_of(m):
        lib = m.get("tsq_lib")
        if lib is None:
            return m
        t = srgb(m)
        lst = canon.setdefault(lib, [])
        if lib == "window_lit":                 # lit rooms: fold onto a few glow / curtain steps, whatever the tint
            try:
                pr = json.loads(m.get("tsq_params", "") or "{}")
            except Exception:
                pr = {}
            g, cu = float(pr.get("glow", 3.0)), float(pr.get("curtain", 0.3))
            for (g2, cu2, m2) in lst:
                if abs(g - g2) <= 0.6 and abs(cu - cu2) <= 0.3:
                    return m2
            lst.append((g, cu, m))
            return m
        pk = m.get("tsq_params", "") if lib in strict else ""
        tol = 0.03 if lib in strict else (0.1 if lib in painted else 0.22)
        for (t2, pk2, m2) in lst:
            if pk2 == pk and sum((a - b) ** 2 for a, b in zip(t, t2)) ** 0.5 < tol:
                return m2
        lst.append((t, pk, m))
        return m

    for m in seed_mats:
        canon_of(m)
    seen = set()
    for o in bpy.context.scene.objects:
        if o.type != 'MESH' or o.data.name in seen:
            continue
        seen.add(o.data.name)
        for i, m in enumerate(o.data.materials):
            if m is not None:
                m2 = canon_of(m)
                if m2 is not m:
                    o.data.materials[i] = m2


def emblem_paths(kind):
    """Neon tube centre lines in the sign plane (x, z), designed in a 0.6 m circle. -> (main, accent) lists of (pts, closed)."""
    main, acc = [], []
    if kind == "potion":
        cz, r, nw = -0.075, 0.2, 0.06
        a0 = degrees(math.acos(nw / r))
        body = [(-0.09, 0.3), (-nw, 0.27), (-nw, cz + r * sin(radians(180 - a0)) + 0.015)]
        body += [(r * cos(radians(a)), cz + r * sin(radians(a))) for a in [180 - a0 + (360 - (180 - 2 * a0)) * k / 26 for k in range(27)]]
        body += [(nw, cz + r * sin(radians(a0)) + 0.015), (nw, 0.27), (0.09, 0.3)]
        main.append((body, True))
        acc.append(([(-0.15 + 0.3 * k / 12, -0.07 + 0.022 * sin(k * pi / 3)) for k in range(13)], False))
        acc.append((circle_pts(0.026, 8, center=(-0.05, 0.01)), True))
        acc.append((circle_pts(0.017, 7, center=(0.045, 0.075)), True))
        acc.append(([(0.2, 0.2), (0.2, 0.3)], False))
        acc.append(([(0.15, 0.25), (0.25, 0.25)], False))
    elif kind == "star":
        st = [((0.3 if k % 2 == 0 else 0.122) * sin(radians(36 * k)), (0.3 if k % 2 == 0 else 0.122) * cos(radians(36 * k)) - 0.01) for k in range(10)]
        main.append((st, True))
        acc.append(([(x * 0.5, (z + 0.01) * 0.5 - 0.01) for x, z in st], True))
    elif kind == "gem":
        main.append(([(-0.27, 0.09), (-0.15, 0.23), (0.15, 0.23), (0.27, 0.09), (0.0, -0.28)], True))
        acc.append(([(-0.2, 0.085), (0.2, 0.085)], False))
        acc.append(([(-0.09, 0.05), (0.0, -0.19), (0.09, 0.05)], False))
        acc.append(([(-0.1, 0.12), (-0.07, 0.2)], False))
        acc.append(([(0.1, 0.12), (0.07, 0.2)], False))
    else:  # crown
        main.append(([(-0.25, -0.16), (-0.3, 0.12), (-0.15, 0.0), (0.0, 0.22), (0.15, 0.0), (0.3, 0.12), (0.25, -0.16)], True))
        acc.append(([(-0.2, -0.085), (0.2, -0.085)], False))
        for c in ((-0.3, 0.165), (0.0, 0.265), (0.3, 0.165)):
            acc.append((circle_pts(0.028, 8, center=c), True))
    return main, acc


def build(p, rng):
    W, D = float(p.width), float(p.depth)
    hw = W / 2
    n_up = int(p.storeys) - 1
    G, Hs = 3.7, 2.85
    Ze = G + n_up * Hs
    c = 1.45                               # chamfer cut along each wall
    R = float(p.turret_size)
    ov = 0.42                              # turret overhang past the wall planes
    a = R - ov
    Cx, Cy = hw - a, a                     # turret axis
    d_roof = min(D, 10.0)                  # the catalogue roof spans up to 10 m; deeper plots stretch it
    sy = D / d_roof
    # the turret wall must rise past the point where the main roof dies into it
    Zt = Ze + max(float(p.turret_rise), a * math.tan(radians(p.roof_pitch)) / sy + 0.32)
    octa = p.turret == "octagon"
    age = float(p.age)
    seed = int(p.seed)

    oak_c = p.timber
    M = {
        "oak": mat("wood_oak", oak_c), "plaster": mat("plaster_lime", p.plaster), "stone": mat("stone_smooth", "#b9ad98"),
        "paint": mat("wood_painted", p.paint), "trim": mat("wood_painted", p.trim),
        "lit": mat("window_lit", "#ffc37a", glow=3.0, curtain=0.35), "lit2": mat("window_lit", "#ffb765", glow=2.2, curtain=0.7),
        "dark": mat("window_dark"),
    }
    iron = mat("iron_wrought")
    lead = mat("lead")
    brass = mat("brass")
    pipe_m = mat("iron_painted", "#20282a")
    wall_m = {"brick_red": mat("brick_red"), "brick_ochre": mat("brick_ochre"), "ashlar": mat("ashlar_limestone"),
              "plaster": mat("plaster_rough", p.plaster)}[p.wall]
    seeds = [M["oak"], M["plaster"], M["stone"], M["paint"], M["trim"], M["lit"], M["lit2"], M["dark"], iron, lead, brass, pipe_m, wall_m]

    fr_f = Frame((-hw, 0), (0, -1))        # front, length W - c
    fr_c = Frame((hw - c, 0), (1, -1))     # chamfer, length c * sqrt(2)
    fr_r = Frame((hw, c), (1, 0))          # +X street face, length D - c
    fr_b = Frame((hw, D), (0, 1))          # back, length W
    fr_l = Frame((-hw, D), (-1, 0))        # left, length D
    Lf, Lr = W - c, D - c
    chord = math.sqrt(max(R * R - a * a, 0.01))
    vis_f = Lf - (chord - (c - a))         # front wall is visible up to here, then it runs into the turret
    vis_r = chord - (c - a)                # side wall becomes visible from here

    # ------------------------------------------------------------------ walls
    foot = [(-hw, 0), (hw - c, 0), (hw, c), (hw, D), (-hw, D)]
    gw = extrude_profile("ground_wall", foot, G)
    densify(gw, 0.7)
    assign(gw, wall_m)
    paint(gw, grime=mask_mul(height_ramp(0.0, 1.5), noise_mask(0.7, seed=seed, lo=0.15, hi=0.8), 0.5 + 0.5 * age),
          special=mask_mul(height_ramp(0.0, 1.1), facing((0, 1, 0)), noise_mask(0.4, seed=seed + 1), age))
    uw = extrude_profile("upper_wall", foot, Ze - G, loc=(0, 0, G))
    densify(uw, 0.6)
    assign(uw, M["plaster"])
    paint(uw, grime=mask_mul(height_ramp(Ze + 0.1, Ze - 1.6), noise_mask(0.3, seed=seed + 2, lo=0.3, hi=0.75), 0.4 + 0.6 * age),
          special=mask_mul(facing((0, 1, 0)), noise_mask(0.8, seed=seed + 3, lo=0.4, hi=0.8), 0.7 * age),
          wear=mask_mul(noise_mask(0.5, seed=seed + 4, lo=0.62, hi=0.8), 0.5 * age))

    # moulded oak bressummer round the building under the framed storeys
    bs = [(-0.07, -0.13), (0.07, -0.13), (0.07, -0.09), (0.095, -0.06), (0.095, -0.03), (0.125, 0.0), (0.125, 0.1),
          (-0.125, 0.1), (-0.125, 0.0), (-0.095, -0.03), (-0.095, -0.06), (-0.07, -0.09)]
    br = sweep("bressummer", [(x, y, G - 0.11) for x, y in foot], bs, closed=True, smooth_path=False)
    assign(br, M["oak"])
    paint(br, grime=mask_mul(noise_mask(0.5, seed=seed + 5), 0.5))

    # ------------------------------------------------------------------ components shared by the walls
    r_w = rng.fork("windows")
    cas = {}

    def casement(lit):
        if lit not in cas:
            cas[lit] = use("window_casement", width=1.0, height=1.35, lights=2, glazing="diamond", shutters=False, lit=lit,
                           frame_paint=p.trim, timber=oak_c, age=age, curtain=0.4, seed=seed + (5 if lit else 6))
        return cas[lit]

    lw_w, lw_h = 0.9, 1.25
    lw_hx = lw_w / 2 + 0.11
    lite = {True: make_window("lw_lit", lw_w, lw_h, M["lit2"], M, "oak", (2, 3)),
            False: make_window("lw_dark", lw_w, lw_h, M["dark"], M, "oak", (2, 3))}
    gwin = {True: make_window("gw_lit", 1.0, 1.55, M["lit"], M, "stone", (2, 4), rng.fork("gwl")),
            False: make_window("gw_dark", 1.0, 1.55, M["dark"], M, "stone", (2, 4), rng.fork("gwd"))}
    shut = {}

    def shutters_for(hx, h):
        k = (round(hx, 3), round(h, 3))
        if k not in shut:
            shut[k] = make_shutters("shutters", hx, h, M, rng.fork("shut%d" % len(shut)), iron)
        return shut[k]

    templates = list(lite.values()) + list(gwin.values())
    T = Timbers(rng.fork("timber"), 0.5 + age * 0.6)
    timber_objs = []

    def upper_wall(fr, sa, sb, va, vb, n_win, hero, post_a, post_b, tag, pattern):
        for i in range(n_up):
            zb = G + i * Hs
            zs = zb + 0.95
            wins = []
            pitch = (vb - va) / n_win
            for k in range(n_win):
                sc = va + pitch * (k + 0.5) + r_w.jitter(0.0, 0.04)
                lit = r_w.chance(p.lit if not hero else min(1.0, p.lit + 0.2))
                near = (n_win - 1 - k) if hero == "end" else k      # 0 = next to the corner turret
                if hero and i == 0 and near < 2:
                    comp = casement(lit)
                    comp.place(fr.pt(sc, 0.02, zs), rot=(0, 0, fr.theta), sockets=False)
                    hx, hh = comp.max.x, 1.35
                else:
                    stamp(lite[lit], fr, sc, zs, out=0.02)
                    hx, hh = lw_hx, lw_h
                block = hx
                if p.shutters and hero and pitch >= 2 * (hx + 0.5) + 0.25:
                    stamp(shutters_for(hx, hh), fr, sc, zs, out=0.0)
                    block = hx + 0.5
                wins.append((sc, block))
            frame_storey(T, fr, sa, sb, zb, Hs, wins, pattern, post_a, post_b)
        timber_objs.extend(T.finish("frame_" + tag, M["oak"]))

    nf = max(1, int(vis_f / 2.9 + 0.35))
    nr = max(1, int((Lr - vis_r) / 2.9 + 0.35))
    upper_wall(fr_f, 0.0, vis_f + 0.3, 0.0, vis_f - 0.1, nf, "end", True, False, "front", p.framing)
    upper_wall(fr_r, vis_r - 0.3, Lr, vis_r + 0.1, Lr, nr, "start", False, True, "side", p.framing)
    upper_wall(fr_b, 0.0, W, 0.2, W - 0.2, max(2, round(W / 3.6)), "", True, True, "back", p.framing)
    upper_wall(fr_l, 0.0, D, 0.2, D - 0.2, max(2, round(D / 3.6)), "", True, True, "left", p.framing)
    for o in timber_objs:
        paint(o, grime=mask_mul(noise_mask(0.6, seed=seed + 6, lo=0.3, hi=0.8), 0.3 + 0.5 * age),
              wear=mask_mul(noise_mask(0.25, seed=seed + 7, lo=0.55, hi=0.85), 0.6 * age))

    # ------------------------------------------------------------------ ground storey: shopfronts, corner door, rear
    stones = []
    r_s = rng.fork("stone")

    def plinth_run(fr, s0, s1):
        if s1 - s0 < 0.25:
            return
        n = max(1, round((s1 - s0) / 0.85))
        cuts = [s0] + sorted(s0 + (s1 - s0) * (k + r_s.jitter(0.0, 0.12)) / n for k in range(1, n)) + [s1]
        for k in range(n):
            ln = cuts[k + 1] - cuts[k]
            b = stone_block("plinth", (ln - 0.008, 0.15, 0.5), loc=fr.pt((cuts[k] + cuts[k + 1]) / 2, 0.015, 0.25), rot=(0, 0, fr.theta),
                            rng=r_s, chamfer=0.016, detail=0.25)
            stones.append(b)

    def quoins(cx, cy, ox, oy):
        z, k = 0.5, 0
        while z + 0.3 <= G - 0.2:
            lx, ly = (0.52, 0.3) if k % 2 == 0 else (0.3, 0.52)
            lx, ly = lx + r_s.jitter(0.0, 0.03), ly + r_s.jitter(0.0, 0.03)
            b = stone_block("quoin", (lx, ly, 0.3), loc=(cx - ox * (lx / 2 - 0.025), cy - oy * (ly / 2 - 0.025), z + 0.15), rng=r_s,
                            chamfer=0.014, detail=0.25)
            stones.append(b)
            z += 0.305
            k += 1

    quoins(-hw, 0.0, -1, -1)
    quoins(-hw, D, -1, 1)
    quoins(hw, D, 1, 1)

    shop_h = 3.3

    def shopfront(fr, L, corner_at_end, text, goods, sd, tag):
        """Shopfront next to the corner, lite windows / service door in the rest of the wall."""
        ws = clamp(L - 0.3 - 1.4, 2.6, 4.6)
        comp = use("shop_window", width=ws, height=shop_h, bays=max(2, round(ws / 0.9)), paint=p.paint, goods=goods, lit=True,
                   fascia=0.58, sign_text=str(text).upper(), wear=0.25 + 0.5 * age, seed=sd, **({} if corner_at_end else {"depth": 0.4}))
        # the +X face may only overhang the plot by 0.7 m: shallower shopfront there
        ysc = 1.0 if corner_at_end else min(1.0, 0.69 / max(-comp.min.y, 0.01))
        t_c = 0.3 + ws / 2                     # distance of the shop centre from the corner end
        sc = L - t_c if corner_at_end else t_c
        comp.place(fr.pt(sc, 0.0, 0.0), rot=(0, 0, fr.theta), scale=(1.0, ysc, 1.0), sockets=False)
        rest0, rest1 = 0.3 + ws + 0.25, L - 0.55
        rest = rest1 - rest0
        els = []
        if rest >= 3.6:
            els = [rest0 + rest * 0.27, rest0 + rest * 0.76]
        elif rest >= 1.25:
            els = [rest0 + rest * 0.5]
        for t in els:
            s = L - t if corner_at_end else t
            stamp(gwin[r_w.chance(0.6)], fr, s, 1.05, out=0.0)
        lo, hi = (0.52, L - (0.3 + ws) - 0.02) if corner_at_end else (0.3 + ws + 0.02, L - 0.52)
        plinth_run(fr, lo, hi)
        return comp, sc, ysc

    shop_f, s_shop_f, _ = shopfront(fr_f, Lf, True, p.sign_front, p.goods_front, seed + 11, "front")
    shop_r, s_shop_r, ysc_r = shopfront(fr_r, Lr, False, p.sign_side, p.goods_side, seed + 12, "side")

    door = use("door_shop", width=1.3, height=2.95, paint=p.paint, trim=p.trim, lit=True, wear=0.3 + 0.4 * age, seed=seed + 13)
    Lc = c * math.sqrt(2.0)
    door.place(fr_c.pt(Lc / 2, 0.0, 0.0), rot=(0, 0, fr_c.theta), sockets=False)

    # back wall: plank door with lantern, windows; left wall: windows
    bdoor = make_back_door("back_door", M, rng.fork("bdoor"), iron)
    s_bd = W * 0.3
    stamp(bdoor, fr_b, s_bd, 0.0)
    templates.append(bdoor)
    lan = make_lantern("lantern", iron, mat("emissive", "#ffb45e", glow=9.0), mat("glass_lantern", glow=1.6))
    stamp(lan, fr_b, s_bd + 0.95, 2.35)
    templates.append(lan)
    nbw = max(1, round((W - s_bd - 1.6) / 3.0))
    for k in range(nbw):
        s = s_bd + 1.7 + (W - s_bd - 2.4) * (k + 0.5) / nbw
        stamp(gwin[k == 0 or r_w.chance(0.4)], fr_b, s, 1.05)
    plinth_run(fr_b, 0.52, s_bd - 0.72)
    plinth_run(fr_b, s_bd + 0.72, W - 0.52)
    nlw = max(2, round(D / 3.6))
    for k in range(nlw):
        stamp(gwin[r_w.chance(0.45)], fr_l, D * (k + 0.5) / nlw, 1.05)
    plinth_run(fr_l, 0.52, D - 0.52)

    if stones:
        st = join(stones, "stonework")
        assign(st, M["stone"])
        paint(st, grime=mask_mul(height_ramp(0.0, 0.7), 0.5 + 0.4 * age),
              special=mask_mul(height_ramp(0.0, 0.8), noise_mask(0.3, seed=seed + 8, lo=0.35, hi=0.7), facing((0, 1, 0)), age))

    # downpipes: front gutter drops round the corner on the left wall, back gutter on the back wall
    def downpipe(fr, s, ztop):
        parts = []
        parts.append(cylinder("pipe", 0.045, ztop - 0.5, loc=(0, -0.13, 0.28 + (ztop - 0.5) / 2), verts=10))
        parts.append(cone("hopper", 0.06, 0.15, 0.26, loc=(0, -0.13, ztop - 0.13), verts=4, rot=(0, 0, 45)))
        parts.append(box("hopper_rim", (0.24, 0.24, 0.04), loc=(0, -0.13, ztop + 0.01), bevel=0.006, segments=1))
        parts.append(cylinder("shoe", 0.045, 0.3, loc=(0, -0.21, 0.2), rot=(42, 0, 0), verts=10))
        z = 0.5
        while z < ztop - 0.5:
            parts.append(cylinder("socket", 0.058, 0.07, loc=(0, -0.13, z), verts=10))
            parts.append(box("ear", (0.2, 0.012, 0.04), loc=(0, -0.085, z), bevel=0.003, segments=1))
            parts.append(box("holdfast", (0.03, 0.09, 0.03), loc=(0, -0.045, z)))
            z += 1.75
        j = join(parts, "downpipe")
        assign(j, pipe_m)
        place(j, loc=fr.pt(s, 0.0, 0.0), rot=(0, 0, fr.theta))
        paint(j, special=mask_mul(noise_mask(0.25, seed=seed + 9, lo=0.45, hi=0.75), 0.4 + 0.6 * age), grime=height_ramp(0.0, 0.6))

    downpipe(fr_l, D - 0.75, Ze - 0.12)
    downpipe(fr_b, W - 0.8, Ze - 0.12)

    # ------------------------------------------------------------------ hanging sign on the front corner post
    hs = use("hanging_sign", emblem=p.hang_emblem, shape="shield", text=str(p.sign_hang), board_color=p.paint, accent_color="#e6d8b4",
             gilded=True, reach=0.7, lamp="none", age=age, seed=seed + 14)
    # CLEARANCE RULE: the back plate can stand across a lit casement light (glass at out 0.050, panes tilted up to
    # 3 mm): at out 0.050 its face is 5 mm or more in front of every pane (it was 0.045: 0.4 to 6 mm)
    hs.place(fr_f.pt(max(0.6, Lf - 0.3 - shop_f.size.x - 0.55), 0.050, G + 0.75), rot=(0, 0, fr_f.theta), sockets=False)

    # ------------------------------------------------------------------ main roof (composed), trimmed at the turret
    slate = p.roof_cover == "roof_slate"
    roof = use("roof_gable", width=W, depth=d_roof, pitch=p.roof_pitch, type="gable", ridge="x", overhang=0.25, covering=p.roof_cover,
               gable="bargeboard", gable_wall="timber", attic_window="lit", rooflight="none", eaves="rafters", gutter="zinc",
               wall_color=p.plaster, timber_color=oak_c, trim_color=p.trim, moss=0.25 + 0.6 * age, seed=seed + 15)
    ro = roof.place((0.0, D / 2, Ze), scale=(1.0, sy, 1.0), sockets=False)
    ro.data = ro.data.copy()
    apply_transform(ro)
    clip_quadrant(ro, Cx, Cy)

    def roof_socks(prefix):
        out = []
        for (nm, kind, sm, props) in roof.sockets:
            if nm.startswith(prefix):
                t = sm.translation
                out.append((t.x, D / 2 + t.y * sy, Ze + t.z))
        return sorted(out)

    nd = int(p.dormers)
    if nd > 0:
        spots = [q for q in roof_socks("dormer_f") if q[0] + 0.85 < Cx - R - 0.25]
        spots = spots[:nd]
        if spots:
            dm = use("dormer", width=1.3, height=1.9, pitch=degrees(atan2(math.tan(radians(p.roof_pitch)), sy)), roof="gable",
                     cover="slate" if slate else "tile", cladding="lead" if slate else "tile", lit=True, flower_box=False,
                     timber=oak_c, paint=p.trim, plaster=p.plaster, age=age, seed=seed + 16)
            for q in spots:
                dm.place(q, sockets=False)
    nc = int(p.chimneys)
    if nc > 0:
        spots = roof_socks("chimney_b")
        spots = [spots[0]] + ([q for q in spots[1:] if q[0] < Cx - R - 1.0][-1:] if nc > 1 else [])
        ch = use("chimney", pitch=p.roof_pitch, soot=0.4 + 0.5 * age, moss=0.6 * age, seed=seed + 17)
        for q in spots:
            ch.place((q[0], q[1], q[2] - 0.5), rot=(0, 0, 180), sockets={"fx"})

    # ------------------------------------------------------------------ corner turret
    seg_n = 8 if octa else 32
    lrot = 22.5 if octa else 0.0
    sm_a = 25.0 if octa else 50.0
    tloc = (Cx, Cy, 0.0)
    r_t = rng.fork("turret")

    def rad_at(phi):
        """Wall radius at world angle phi (deg): constant for the round turret, the flat faces for the octagon."""
        if not octa:
            return R
        d = ((phi + 22.5) % 45.0) - 22.5
        return R * S22 / cos(radians(d))

    def ring(name, prof, m):
        o = lathe(name, prof + [prof[0]], segments=seg_n, loc=tloc, rot=(0, 0, lrot), smooth_angle=sm_a)
        assign(o, m)
        return o

    # corbel: stone mouldings stepping out from the door cornice to the turret base
    rc0 = 0.54 + (2 * a - c) / math.sqrt(2.0) + 0.02
    zc0 = G - 0.75
    cp = [(0.0, zc0), (rc0, zc0), (rc0 + 0.035, zc0 + 0.03), (rc0 + 0.035, zc0 + 0.075), (rc0 + 0.01, zc0 + 0.1)]
    r1c, z1c = R - 0.1, G - 0.26
    for k in range(1, 7):
        q = k / 6
        cp.append((rc0 + 0.01 + (r1c - rc0 - 0.01) * (1 - cos(q * pi / 2)), zc0 + 0.1 + (z1c - zc0 - 0.1) * sin(q * pi / 2)))
    cp += [(R - 0.1, G - 0.22), (R - 0.03, G - 0.2), (R + 0.03, G - 0.15), (R + 0.03, G - 0.1), (R + 0.075, G - 0.08), (R + 0.075, G), (0.0, G)]
    corbel = lathe("turret_corbel", cp, segments=seg_n, loc=tloc, rot=(0, 0, lrot), smooth_angle=sm_a)
    assign(corbel, M["stone"])
    paint(corbel, grime=mask_mul(noise_mask(0.3, seed=seed + 20, lo=0.3, hi=0.8), 0.4 + 0.5 * age))

    # shaft
    nz = max(2, int((Zt - G) / 0.6))
    shaft = lathe("turret_shaft", [(R, G + (Zt - G) * k / nz) for k in range(nz + 1)], segments=seg_n, loc=tloc, rot=(0, 0, lrot),
                  smooth_angle=sm_a)
    assign(shaft, M["plaster"])
    paint(shaft, grime=mask_mul(height_ramp(Zt, Zt - 1.6), noise_mask(0.3, seed=seed + 21, lo=0.3, hi=0.75), 0.4 + 0.6 * age))

    t_oak, t_trim, t_glass = [], [], []
    t_oak.append(ring("t_base", [(R - 0.03, G), (R + 0.065, G), (R + 0.065, G + 0.17), (R + 0.045, G + 0.2), (R - 0.03, G + 0.2)], M["oak"]))
    t_oak.append(ring("t_plate", [(R - 0.03, Zt - 0.17), (R + 0.055, Zt - 0.17), (R + 0.055, Zt - 0.01), (R - 0.03, Zt - 0.01)], M["oak"]))
    for i in range(n_up):
        zf = G + i * Hs
        zs, zh = zf + 0.95, zf + 2.35
        if i > 0:
            t_oak.append(ring("t_floor", [(R - 0.03, zf - 0.02), (R + 0.06, zf - 0.02), (R + 0.06, zf + 0.18), (R - 0.03, zf + 0.18)], M["oak"]))
        t_oak.append(ring("t_sill", [(R - 0.03, zs - 0.085), (R + 0.05, zs - 0.085), (R + 0.095, zs - 0.03), (R + 0.095, zs), (R + 0.045, zs + 0.03),
                                     (R - 0.03, zs + 0.03)], M["oak"]))
        t_oak.append(ring("t_head", [(R - 0.03, zh), (R + 0.055, zh), (R + 0.055, zh + 0.13), (R - 0.03, zh + 0.13)], M["oak"]))
    t_oak.append(ring("t_lantern_sill", [(R - 0.03, Ze + 0.05), (R + 0.06, Ze + 0.05), (R + 0.06, Ze + 0.19), (R - 0.03, Ze + 0.19)], M["oak"]))

    def visible(phi):
        d = ((phi + 45.0 + 180.0) % 360.0) - 180.0      # angle from the outward diagonal
        return abs(d) < 97.0

    # studs every 22.5 deg (corner posts + face mullions on the octagon)
    for k in range(16):
        phi = -45.0 + 22.5 * k
        corner = octa and (k % 2 == 1)
        rr = rad_at(phi)
        n = (cos(radians(phi)), sin(radians(phi)))
        z_lo = G + 0.2 if visible(phi) else Ze - 0.2
        if corner:
            o = box("t_post", (0.16, 0.16, Zt - 0.17 - z_lo), loc=(Cx + (rr - 0.03) * n[0], Cy + (rr - 0.03) * n[1], (z_lo + Zt - 0.17) / 2),
                    rot=(0, 0, phi), bevel=0.01, segments=1)
        else:
            fr = Frame((Cx + rr * n[0], Cy + rr * n[1]), n)
            o = box("t_stud", (0.125 * r_t.vary(1.0, 0.05), 0.07, Zt - 0.17 - z_lo), loc=fr.pt(0.0, 0.013, (z_lo + Zt - 0.17) / 2),
                    rot=(0, 0, fr.theta), bevel=0.009, segments=1)
        uv_grain(o, "Z")
        assign(o, M["oak"])
        t_oak.append(o)

    def cell(j):
        """Glazing cell j between studs: -> (frame at the cell centre, clear chord width)."""
        phi = -45.0 + 11.25 + 22.5 * j
        if octa:
            face = round((phi - 11.25 * (1 if (j % 2 == 0) else -1)) / 45.0) * 45.0
            fa = radians(face)
            apo = R * S22
            fwid = 2 * R * sin(radians(22.5))
            n = (cos(fa), sin(fa))
            f0 = Frame((Cx + apo * n[0], Cy + apo * n[1]), n)
            side = 1.0 if ((phi - face + 540.0) % 360.0 - 180.0) > 0 else -1.0
            return Frame(f0.pt(side * fwid / 4), n), fwid / 2
        n = (cos(radians(phi)), sin(radians(phi)))
        return Frame((Cx + R * n[0], Cy + R * n[1]), n), 2 * R * sin(radians(11.25))

    # ribbon of leaded lights across the street side of the turret on every framed storey
    for i in range(n_up):
        zf = G + i * Hs
        z0, z1 = zf + 0.98, zf + 2.35
        hh = z1 - z0
        lit_storey = p.lit > 0.02
        for j in range(-3, 3):
            fr, cw = cell(j)
            # CLEARANCE RULE: every light is one glowing sheet 16 mm off the shaft (the front of the old slab; the
            # slabs of two lights met end to end in one plane), the backs of the bars stand 4 mm behind it
            g = lod_faces(box("t_glass", (cw, 0.012, hh), loc=fr.pt(0.0, 0.01, (z0 + z1) / 2), rot=(0, 0, fr.theta)),
                          keep=(fr.n[0], fr.n[1], 0.0), above=-1)
            assign(g, (M["lit"] if (j + i) % 2 == 0 else M["lit2"]) if lit_storey else M["dark"])
            t_glass.append(g)
            o = box("t_gbar_v", (0.03, 0.033, hh), loc=fr.pt(0.0, 0.0285, (z0 + z1) / 2), rot=(0, 0, fr.theta), bevel=0.004, segments=1)
            t_trim.append(o)
            for q in (0.3, 0.74):
                o = box("t_gbar_h", (cw, 0.03, 0.03 if q > 0.5 else 0.024), loc=fr.pt(0.0, 0.027, z0 + hh * q), rot=(0, 0, fr.theta), bevel=0.004,
                        segments=1)
                t_trim.append(o)
    # lantern storey above the eaves: small lit panes all the way round
    zl0, zl1 = Ze + 0.19, Zt - 0.17
    if zl1 - zl0 >= 0.4:
        ph = min(0.62, zl1 - zl0 - 0.14)
        zc = (zl0 + zl1) / 2
        for j in range(16):
            fr, cw = cell(j)
            g = lod_faces(box("t_lant_glass", (cw, 0.012, ph), loc=fr.pt(0.0, 0.01, zc), rot=(0, 0, fr.theta)),
                          keep=(fr.n[0], fr.n[1], 0.0), above=-1)
            assign(g, M["lit"] if (j % 2 == 0) else M["lit2"])
            t_glass.append(g)
            for dz in (-ph / 2 - 0.02, ph / 2 + 0.02):
                o = box("t_lant_rail", (cw, 0.05, 0.05), loc=fr.pt(0.0, 0.03, zc + dz), rot=(0, 0, fr.theta), bevel=0.006, segments=1)
                uv_grain(o, "X")
                assign(o, M["oak"])
                t_oak.append(o)
            o = box("t_lant_bar", (0.022, 0.026, ph), loc=fr.pt(0.0, 0.025, zc), rot=(0, 0, fr.theta))
            t_trim.append(o)
    for o in t_trim:
        assign(o, M["trim"])
    for o in t_oak:
        paint(o, grime=mask_mul(noise_mask(0.5, seed=seed + 22, lo=0.3, hi=0.8), 0.3 + 0.5 * age))

    # eaves: moulded cornice + soffit ring and little shaped rafter feet
    rb = R + 0.26
    ring("t_eaves", [(R - 0.03, Zt - 0.03), (R + 0.06, Zt - 0.03), (R + 0.08, Zt + 0.0), (R + 0.13, Zt + 0.03), (rb - 0.05, Zt + 0.015),
                     (rb - 0.05, Zt + 0.07), (R - 0.03, Zt + 0.09)], M["trim"])
    nfeet = 16 if octa else 24
    ft = box("t_foot_base", (0.07, 0.17, 0.07), bevel=0.012, segments=1)
    fm = []
    for k in range(nfeet):
        phi = 360.0 * (k + 0.5) / nfeet
        rr = rad_at(phi) + 0.12
        fm.append(trs((Cx + rr * cos(radians(phi)), Cy + rr * sin(radians(phi)), Zt - 0.015), (0, 0, phi + 90.0)))
    feet = instances_to_mesh("t_feet", ft, fm, keep_base=False)
    assign(feet, M["oak"])

    # roof profile from the eaves to the apex
    Hc = float(p.turret_spire) * R
    z0r = Zt + 0.07
    prof = []
    if p.turret == "bell":
        P = [(rb, z0r), (rb * 1.1, z0r + 0.44 * Hc), (0.11, z0r + 0.4 * Hc), (0.05, z0r + Hc)]
        for k in range(49):
            t = k / 48
            b0, b1, b2, b3 = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
            prof.append((b0 * P[0][0] + b1 * P[1][0] + b2 * P[2][0] + b3 * P[3][0], b0 * P[0][1] + b1 * P[1][1] + b2 * P[2][1] + b3 * P[3][1]))
    else:
        rk, zk = R + 0.04, z0r + 0.34
        for k in range(5):
            t = k / 4
            prof.append((rb + (rk - rb) * t, z0r + (zk - z0r) * t ** 1.3))
        for k in range(1, 31):
            t = k / 30
            prof.append((rk + (0.05 - rk) * t, zk + (z0r + Hc - zk) * t))
    cum = [0.0]
    for k in range(1, len(prof)):
        cum.append(cum[-1] + hypot(prof[k][0] - prof[k - 1][0], prof[k][1] - prof[k - 1][1]))

    def at(s):
        s = clamp(s, 0.0, cum[-1] - 1e-6)
        k = 0
        while cum[k + 1] < s:
            k += 1
        t = (s - cum[k]) / max(cum[k + 1] - cum[k], 1e-9)
        dr, dz = prof[k + 1][0] - prof[k][0], prof[k + 1][1] - prof[k][1]
        l = hypot(dr, dz)
        return prof[k][0] + dr * t, prof[k][1] + dz * t, dr / l, dz / l

    cover = p.turret_cover
    ztop = prof[-1][1]
    if cover == "copper":
        cu = mat("copper_patina")
        sk = lathe("t_roof", [(0.0, z0r - 0.005), (rb - 0.02, z0r - 0.005)] + prof + [(0.0, ztop)], segments=seg_n, loc=tloc, rot=(0, 0, lrot),
                   smooth_angle=sm_a)
        assign(sk, cu)
        ribs = []
        nrib = 8 if octa else 14
        for k in range(nrib):
            phi = radians(22.5 + 45.0 * k) if octa else 2 * pi * k / nrib
            path = []
            for q in range(15):
                rr, zz, _, _ = at(cum[-1] * (0.02 + 0.9 * q / 14))
                path.append(((rr + 0.008) * cos(phi) + Cx, (rr + 0.008) * sin(phi) + Cy, zz + 0.004))
            ribs.append(sweep("t_rib", path, 0.016 if octa else 0.012, sides=5))
        rj = join(ribs, "t_ribs")
        assign(rj, cu)
        for o in (sk, rj):
            paint(o, special=mask_mul(noise_mask(0.5, seed=seed + 23, lo=0.25, hi=0.7), 0.5 + 0.5 * age), grime=height_ramp(z0r, z0r + 0.5))
    else:
        tile_m, gauge, tw, th = {"slate": (mat("slate", "#8b93a0"), 0.2, 0.2, 0.01),
                                 "shingle": (mat("wood_weathered", "#b5a48e"), 0.19, 0.165, 0.014),
                                 "tile": (mat("terracotta", "#c98f78"), 0.16, 0.17, 0.014)}[cover]
        under = lathe("t_underlay", [(0.0, z0r - 0.005), (rb - 0.03, z0r - 0.005)] + [(max(r - 0.012, 0.01), z) for r, z in prof] + [(0.0, ztop)],
                      segments=seg_n, loc=tloc, rot=(0, 0, lrot), smooth_angle=sm_a)
        assign(under, lead)
        Lt = gauge * 1.6
        base = box("t_tile_base", (tw, Lt, th), loc=(0, Lt / 2, th / 2))
        apply_transform(base)
        transform_verts(base, lambda co: Vector((co.x, co.y, co.z + 0.011 * (1.0 - co.y / Lt))))
        uv_planar(base, (1, 0, 0), (0, 1, 0))
        mats = []

        def tile_m4(px, py, pz, fa, ur, uz, sx, sl):
            cs, sn = cos(fa), sin(fa)
            return Matrix(((-sn * sx, ur * cs * sl, uz * cs, px), (cs * sx, ur * sn * sl, uz * sn, py), (0.0, uz * sl, -ur, pz), (0, 0, 0, 1)))

        i = 0
        while True:
            s = i * gauge + 0.005
            if s > cum[-1] - 0.12:
                break
            rr, zz, ur, uz = at(s)
            if rr < 0.15:
                break
            if octa:
                fwid = 2 * rr * sin(radians(22.5))
                m = max(1, round(fwid / tw)) + (i % 2)
                for f in range(8):
                    fa = radians(45.0 * f)
                    apo = rr * S22
                    for j in range(m):
                        off = -fwid / 2 + fwid * (j + 0.5) / m
                        mats.append(tile_m4(Cx + apo * cos(fa) - sin(fa) * off, Cy + apo * sin(fa) + cos(fa) * off,
                                            zz - r_t.u(0.0, 0.012), fa, ur, uz, fwid / m / tw * 0.96, r_t.vary(1.0, 0.06)))
            else:
                n = max(6, round(2 * pi * rr / tw))
                for j in range(n):
                    fa = 2 * pi * (j + 0.5 * (i % 2) + r_t.jitter(0.0, 0.06)) / n
                    mats.append(tile_m4(Cx + rr * cos(fa), Cy + rr * sin(fa), zz - r_t.u(0.0, 0.012), fa, ur, uz,
                                        2 * pi * rr / n / tw * 0.95, r_t.vary(1.0, 0.06)))
            i += 1
        tiles = instances_to_mesh("t_tiles", base, mats, keep_base=False)
        assign(tiles, tile_m)
        paint(tiles, special=mask_mul(noise_mask(0.45, seed=seed + 23, lo=0.4, hi=0.75), 0.3 + 0.6 * age),
              grime=mask_mul(noise_mask(0.2, seed=seed + 24, lo=0.3, hi=0.8), 0.5))
        if octa:
            hips = []
            for k in range(8):
                phi = radians(22.5 + 45.0 * k)
                path = []
                for q in range(9):
                    rr, zz, _, _ = at(cum[-1] * (0.0 + 0.93 * q / 8))
                    path.append(((rr + 0.02) * cos(phi) + Cx, (rr + 0.02) * sin(phi) + Cy, zz + 0.012))
                hips.append(sweep("t_hip", path, 0.034, sides=6, smooth_path=False))
            hj = join(hips, "t_hips")
            assign(hj, lead)

    # lead cap, gilt ball, spike and a turning vane
    zc_i = next(k for k in range(len(prof)) if prof[k][0] <= 0.2)
    capp = [(prof[zc_i][0] + 0.03, prof[zc_i][1] - 0.03)] + [(r + 0.022, z) for r, z in prof[zc_i:]]
    capp += [(0.085, ztop + 0.01), (0.1, ztop + 0.035), (0.1, ztop + 0.06), (0.05, ztop + 0.09), (0.035, ztop + 0.2), (0.0, ztop + 0.2)]
    cap = lathe("t_cap", [(0.0, capp[0][1])] + capp, segments=16 if not octa else 8, loc=tloc, rot=(0, 0, lrot), smooth_angle=sm_a)
    assign(cap, lead)
    ball = sphere("t_ball", 0.095, loc=(Cx, Cy, ztop + 0.27), segments=16, rings=10)
    assign(ball, brass)
    spike = cylinder("t_spike", 0.012, 0.95, loc=(Cx, Cy, ztop + 0.36 + 0.475), verts=8, radius2=0.006)
    assign(spike, iron)
    card = []
    for ang in (0, 90):
        card.append(cylinder("t_card", 0.007, 0.5, loc=(Cx, Cy, ztop + 0.62), rot=(90, 0, ang), verts=6))
    cj = join(card, "t_cardinal")
    assign(cj, iron)
    zv = ztop + 1.0
    v1 = shape_extrude("vane_arrow", [(-0.34, -0.012), (0.2, -0.012), (0.2, -0.06), (0.4, 0.0), (0.2, 0.06), (0.2, 0.012), (-0.34, 0.012)],
                       depth=0.005, loc=(Cx, Cy + 0.0025, zv), rot=(90, 0, 0))
    v2 = shape_extrude("vane_tail", [(-0.4, -0.11), (-0.14, -0.075), (-0.14, 0.075), (-0.4, 0.11), (-0.31, 0.0)],
                       depth=0.005, loc=(Cx, Cy + 0.0025, zv), rot=(90, 0, 0))
    v3 = cylinder("vane_hub", 0.02, 0.2, loc=(Cx, Cy, zv), verts=8)
    for o in (v1, v2, v3):
        assign(o, brass)
    part([v1, v2, v3], "vane", pivot=(Cx, Cy, zv), motion="spin", axis="Z")

    # ------------------------------------------------------------------ neon emblem on the turret above the door
    k_e = float(p.emblem_size) / 0.8
    eparts = []
    rbd = 0.4 * k_e
    bd = shape_extrude("neon_board", circle_pts(rbd, 28), depth=0.02, rot=(90, 0, 0), bevel=0.003)
    assign(bd, mat("iron_painted", "#15181c"))
    eparts.append(bd)
    rim = torus("neon_rim", rbd, 0.016, loc=(0, -0.012, 0), rot=(90, 0, 0), major_seg=32, minor_seg=6)
    assign(rim, brass)
    eparts.append(rim)
    for ang in (45, 135, 225, 315):
        so = cylinder("neon_standoff", 0.012, 0.22, loc=(rbd * 0.8 * cos(radians(ang)), 0.1, rbd * 0.8 * sin(radians(ang))), rot=(90, 0, 0), verts=6)
        assign(so, iron)
        eparts.append(so)
    main, acc = emblem_paths(p.emblem)
    sc_e = k_e * 1.05
    clips = []
    for grp, col in ((main, p.neon), (acc, p.neon2)):
        nm = mat("neon", col, glow=4.5)
        for pts, closed in grp:
            path = [(x * sc_e, -0.05, z * sc_e) for x, z in pts]
            tb = sweep("neon_tube", path, 0.0095, closed=closed, sides=6, smooth_path=False)
            assign(tb, nm)
            eparts.append(tb)
            step = max(1, len(path) // 3)
            for q in path[::step]:
                clips.append(trs((q[0], -0.035, q[2]), (90, 0, 0)))
    cl = instances_to_mesh("neon_clips", cylinder("clip_base", 0.006, 0.03, verts=5), clips, keep_base=False)
    assign(cl, brass)
    eparts.append(cl)
    emb = join(eparts, "neon_emblem")
    dgn = (cos(radians(-45.0)), sin(radians(-45.0)))
    re = rad_at(-45.0) + 0.17
    ez = G + 0.07 + rbd + 0.02
    place(emb, loc=(Cx + re * dgn[0], Cy + re * dgn[1], ez), rot=(0, 0, 45.0))
    light("neon", (Cx + (re + 0.16) * dgn[0], Cy + (re + 0.16) * dgn[1], ez), color=p.neon, intensity=7.0, range=5.0, flicker=0.03,
          shadow_near=0.12)

    # ------------------------------------------------------------------ lights (emissive sources are the lit glazing)
    # Lights sit on their visible sources: the lit display glazing (pavement spill), the swan-neck fascia lamps (one small
    # light per fascia for the lettering), the door's glazing and the neon tubes.  Turret and back lantern only glow.
    def sock(comp, names, fr, s, ysc=1.0, back=0.0):
        ps = [sm.translation for (nm, kind, sm, props) in comp.sockets if nm in names]
        if not ps:
            return None
        q = sum(ps, Vector((0, 0, 0))) / len(ps)
        return fr.pt(s + q.x, (-q.y - back) * ysc, q.z)

    # The shop window throws its spill from 1.5 m out in front of its glass. The +X face stands on the plot line (its
    # shopfront may overhang by 0.7 m, see shopfront): what lies beyond is a lane or a neighbour, and a light out there
    # is a ball of glow hanging in mid-alley with nothing to shine it - on that face it is taken back onto the pane.
    for comp, fr, sc_, ysc, tag, cd, back in ((shop_f, fr_f, s_shop_f, 1.0, "front", 34.0, 0.0), (shop_r, fr_r, s_shop_r, ysc_r, "side", 30.0, 1.38)):
        q = sock(comp, ("spill0",), fr, sc_, ysc, back) or fr.pt(sc_, 0.05, 1.6)
        light("shop_" + tag, q, color="#ffc37a", intensity=cd, range=9.0, flicker=0.05, shadow_near=0.8)
        q = sock(comp, ("sign0", "sign1"), fr, sc_, ysc)
        q = fr.pt(sc_, 0.9 * ysc, shop_h - 0.15) if q is None else fr.pt(sc_, fr.n[0] * (q[0] - fr.o[0]) + fr.n[1] * (q[1] - fr.o[1]), q[2])
        light("fascia_" + tag, q, color="#ffc37a", intensity=4.0, range=3.5, flicker=0.03, shadow_near=0.15)
    light("door", fr_c.pt(Lc / 2, 0.6, 2.25), color="#ffc37a", intensity=12.0, range=6.0, flicker=0.04, shadow_near=0.4)

    socket("awning_front", "attach", fr_f.pt(s_shop_f, 0.5, shop_h - 0.55), rot=(0, 0, fr_f.theta), role="awning")
    socket("lantern_left", "attach", fr_f.pt(0.9, 0.045, G + 0.6), rot=(0, 0, fr_f.theta), role="lantern")
    socket("door", "attach", fr_c.pt(Lc / 2, 0.0, 0.0), rot=(0, 0, fr_c.theta), role="door")

    delete(templates + list(shut.values()))
    consolidate(seeds)

