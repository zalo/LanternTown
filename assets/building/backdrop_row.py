"""Distant roofscape row: a cheap background terrace of simplified old-town houses along X, meant to be
seen over the rooftops of the square from 30-80 m. Narrow eaves-on, gable-on, cross-gabled and crow-stepped
houses with varied eaves heights, storey heights, roof pitches and front lines; limewashed, half-timbered
(some jettied) or brick with pale stone dressings; sagging ridges with ridge caps, fascias and barge boards,
jerkinhead ends, stout chimney stacks with pots, gabled and shed dormers. Windows are framed units: casing
or stone surround, glazing bars, sill, lintel, some with shutters. Rather more than half are dark; the lit
ones are lamp-lit rooms that differ in room and curtains (per-window seed) or, about a third of them, deep
amber fire- / candle-lit rooms at a third of the glow, and every house keeps one bright room on the street side.
One house may carry a landmark on its own bay: a clock tower (broach spire or crenellated top) or an octagonal
stair turret with a conical roof. Its shaft - ashlar, brick or render under stone dressings (landmark_wall) - is
built in stages of 3.8 m at most: battered plinth, a weathered string course with an undercut bed at every
stage and at the offset on the eaves line, pale corner lesenes, a lancet / stair slit in a stone surround in
every stage of every face that stands free (lit by the same draw as the house windows), a belfry stage with a
two-light louvred, softly lit opening on all four faces (turret: a ring of small lights under the eaves), a
back-lit dial with rim and hands, a corbel course under the cornice, dirt washed down under every course and a
darker foot. A few street lamps stand in front of the row.

Runs do not always butt against a neighbour, so ends and backs are finished too. A run-end wall stands 14 cm
inside the extent and carries, flush with it, corner piers, a brick chimney breast with shoulders that rises
into a stack (end_stacks), windows on every floor and one in the gable, the verge / barge board and the
house's courses or timbers. Backs have windows in pale casings (some lit), and on the near layer a lean-to
outshot with its own door or window (outshots) or a back door. Every end wall, most backs and the landmark keep
at least one lit window. Roof slopes are laid in two courses with a sprocketed (bell-cast) eaves skirt, each
band and hip in its own tone, clay ridge tiles on every covering, rooflights and back dormers.

layer="far" builds the second ring further out with the same vocabulary: taller houses under bigger roofs
with eaves, ridge tiles, stacks and pots, regular grids of recessed framed windows (most dark, the lit ones
mullioned), stout half-timbering, the odd stepped gable or church tower. Its colours are hazed: drained to a
grey-blue of their OWN value (limewash stays pale, brick mid, oak and slate dark - never one flat navy), and
its unlit panes are dark grey-blue glass in a pale casing rather than black holes.

Night: max_lights real lights per run (default two: street lamps first, then the clock, then one dormer).
Everything else that glows is a surface: the two glazings, of which the dial is a flat patch of the dim one
and the lantern heads a flat patch of the lit one. Nothing is faked onto walls or paving - a lamp beyond
max_lights just keeps its glowing head.

Materials (10 at most): two limewashes, brick, ashlar, clay tile, slate, stained oak, dark glass, lit and dim
glazing.

Level of detail: the row is seen from 60-150 m, i.e. mostly as LOD1 / LOD2, so it is built in two tiers. The
DISTANCE tier is everything that makes it read as real houses from far and above - plumb hulls, roof slopes
with fascia, soffit, barge boards and ridge tiles, dormers, chimney stacks with caps and pots, the windows as
openings cut into the wall with a set-back casing sheet, the pane and (lit windows) one stout mullion and
transom as a regular mask over the glow, sills, lintels, main timbers, everything on the run ends and backs, the
whole landmark (plinth, courses, lesenes, lancets, belfry, corbels, dial; nothing under 0.14 m). Each of
its faces is emitted as a triangle fan whose diagonals carry a pin - a coplanar twin of the neighbouring fan
triangle (lod_pins; see _MB._emit) - which the town's LOD decimator leaves intact, so LOD1 / LOD2 show exactly
this tier; a pin has no extent of its own, so nothing can show on or through a surface. The CLOSE tier
(reveals, sash frames, fine glazing bars, shutters, brackets, lamp bases) is plain unpinned quads that the LOD
steps dissolve. tier="distance" builds the distance tier alone (a preview of the LODs); lod_guard=True drops the
sub-10 cm detail. About 3-4.5k triangles for 20 m, never more than 8.2k; roughly 100-135 per metre are pinned.

The run spans exactly x in [-length/2, +length/2] with flush end walls so runs butt together.
"""
from tk import *
import math
import numpy as np
from mathutils import Vector

# materials: 2 limewashes, brick, ashlar, tile, slate, oak, dark / lit / dim glazing (10, the toolkit's limit)
# lod: the far level is modelled (docs/LOD.md, META["lod"] = "far"): the distance tier as it stands, without its pins
# and without the close tier - what the decimated copy showed, at about half its triangles (a pin is there to stop a
# decimator; a modelled level is never decimated). The row is a distance model to begin with, so that level keeps
# a larger share of lod 0 than the usual tenth (lod_share).
META = {"name": "Distant roofscape row", "class": "backdrop", "mount": "ground", "focus": 0.42, "auto_grime": 0.2,
        "lod": "far", "lod_share": {2: 0.7}}
PARAMS = {
    "length": {"default": 20.0, "min": 12.0, "max": 60.0, "desc": "run length along X (m); spans exactly +-length/2"},
    "count": {"default": 4, "min": 3, "max": 7, "desc": "houses per 20 m of run (clamped so every house stays 4.6-9 m wide)"},
    "layer": {"default": "near", "choices": ["near", "far"],
              "desc": "near = detailed row; far = second ring: taller houses, distance tier only, hazed, fewer lit windows"},
    "height_min": {"default": 6.0, "min": 4.5, "max": 10.0, "desc": "lowest eaves height (m); the far layer scales it by 1.6"},
    "height_max": {"default": 9.8, "min": 5.5, "max": 15.0, "desc": "highest eaves height (m); the far layer scales it by 1.6"},
    "depth": {"default": 8.0, "min": 6.0, "max": 12.0, "desc": "typical house depth along Y (m)"},
    "stagger": {"default": 1.1, "min": 0.0, "max": 2.5, "desc": "front-line stagger between neighbours (m)"},
    "roof_pitch": {"default": 49.0, "min": 38.0, "max": 58.0, "desc": "typical roof pitch (deg); every house varies it"},
    "hips": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "chance of a jerkinhead (half-hipped) end on exposed gables"},
    "jetty": {"default": 0.26, "min": 0.0, "max": 0.4, "desc": "overhang of jettied upper floors on half-timbered houses (m)"},
    "lit_ratio": {"default": 0.42, "min": 0.0, "max": 1.0,
                  "desc": "fraction of windows that are lit (every house keeps one bright street-side room unless it is 0)"},
    "curtains": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "how many lit windows have their curtains drawn"},
    "lit_color": {"default": "#ffcc62", "desc": "interior light colour of the lit windows (and of the street lantern heads)"},
    "dim_color": {"default": "#dcc04c", "desc": "light colour of the fire- / candle-lit rooms, the belfry and the clock dial (about a third of the glow of a lamp-lit room)"},
    "landmark": {"default": "spire", "choices": ["spire", "tower", "turret", "none"], "desc": "tall accent on its own bay of one house"},
    "landmark_height": {"default": 4.0, "min": 3.0, "max": 10.0, "desc": "landmark wall height above the highest ridge (m)"},
    "landmark_wall": {"default": "auto", "choices": ["auto", "ashlar", "brick", "render"],
                      "desc": "landmark shaft walling under its stone dressings (plinth, string courses, corner lesenes, surrounds); auto = by seed"},
    "clock": {"default": True, "desc": "softly lit clock dial on the tower / spire landmark"},
    "end_stacks": {"default": True, "desc": "exposed run-end walls carry a brick chimney breast rising into a stack flush with the end"},
    "outshots": {"default": 0.5, "min": 0.0, "max": 1.0,
                 "desc": "near layer: chance per house of a lean-to outshot against the back wall (houses without one often get a back door)"},
    "walls": {"default": "mixed", "choices": ["mixed", "plaster", "brick"], "desc": "wall materials"},
    "roofs": {"default": "mixed", "choices": ["mixed", "tile", "slate"], "desc": "roof coverings"},
    "wall_a": {"default": "#d8c6a2", "desc": "first limewash colour"},
    "wall_b": {"default": "#c79f78", "desc": "second limewash colour"},
    "trim": {"default": "#3b2b20", "desc": "timber / frame stain colour"},
    "glow": {"default": 6.4, "min": 0.5, "max": 110.0, "desc": "emissive strength of the lit windows (each room is turned down from it by its own amount)"},
    "lamps": {"default": 1, "min": 0, "max": 4,
              "desc": "street lamps in front of the row; max_lights of them cast light, any further lamp only has a glowing head"},
    "max_lights": {"default": 2, "min": 0, "max": 4, "desc": "real light sockets per run: street lamps first, then the clock, then one dormer"},
    "spill": {"default": 80.0, "min": 0.0, "max": 90.0, "desc": "street lamp intensity (cd); clock and dormer spill are fractions of it"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "weathering: moss, soot, ridge sag (stacks lean only with stack_lean)"},
    "haze": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "far layer only: how far the wall tints sink into blue-grey night haze"},
    "lod_pins": {"default": True, "desc": "pin the distance tier (hulls, roofs with eaves / verges / ridge tiles, dormers, stacks and pots, "
                                          "window casings, lit panes and their mullions) so LOD1 / LOD2 keep it exactly"},
    "slate_share": {"default": 0.75, "min": 0.0, "max": 1.0, "desc": "roofs=mixed: share of houses under navy slate (the rest clay tile)"},
    "moonlit": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "how readily roofs and walls take the moon / sky: smoother, cleaner roof coverings, lighter haze"},
    "stack_lean": {"default": 0.0, "min": 0.0, "max": 1.0, "desc": "lets old chimney stacks lean a little (0 = every stack plumb)"},
    "tier": {"default": "full", "choices": ["full", "distance"],
             "desc": "distance = build only the pinned distance tier (exactly what LOD1 / LOD2 keep): preview, or an explicit far LOD"},
    "lod_guard": {"default": False, "desc": "build the lean tier only (no casing returns, brackets, shutters, fine bars): nothing thin for an LOD step to dissolve"},
    "seed": {"default": 3, "min": 0, "max": 99999, "desc": "layout seed: widths, heights, depths, house types, lit windows"},
}
VARIANTS = {
    "long_tower": {"length": 34.0, "count": 5, "landmark": "tower", "landmark_height": 4.0, "landmark_wall": "ashlar", "lit_ratio": 0.45,
                   "height_min": 6.5, "height_max": 11.0, "wall_a": "#cfc3ac", "wall_b": "#b8926e", "lamps": 3, "max_lights": 3,
                   "seed": 11},
    "short_brick": {"length": 14.0, "count": 4, "landmark": "none", "walls": "brick", "roofs": "slate",
                    "height_min": 8.0, "height_max": 12.5, "roof_pitch": 54.0, "lit_ratio": 0.4, "hips": 0.2, "lamps": 1, "seed": 4},
    "low_cottages": {"length": 30.0, "count": 4, "landmark": "none", "walls": "plaster", "roofs": "tile",
                     "height_min": 4.5, "height_max": 6.4, "roof_pitch": 43.0, "lit_ratio": 0.5, "curtains": 0.7, "stagger": 2.0,
                     "wall_a": "#e2d6bd", "wall_b": "#b9a58a", "age": 0.85, "hips": 0.9, "jetty": 0.34, "lamps": 2, "seed": 23},
    "slate_spire": {"length": 22.0, "count": 4, "landmark": "spire", "landmark_height": 6.0, "landmark_wall": "render", "roofs": "slate",
                    "height_min": 8.0, "height_max": 11.5, "roof_pitch": 56.0, "lit_ratio": 0.36, "lit_color": "#ffd77a",
                    "wall_a": "#bfc0b6", "wall_b": "#a9a08f", "trim": "#2a2622", "age": 0.25, "hips": 0.3, "seed": 31},
    "corner_turret": {"length": 26.0, "count": 4, "landmark": "turret", "landmark_height": 5.0, "lit_ratio": 0.42,
                      "height_min": 6.8, "height_max": 10.4, "roof_pitch": 52.0, "stagger": 1.5, "wall_a": "#dccfb4",
                      "wall_b": "#c2a27c", "curtains": 0.3, "hips": 0.35, "seed": 47},
    "far_ring": {"length": 34.0, "layer": "far", "landmark": "tower", "clock": False, "lamps": 0, "height_min": 5.6,
                 "height_max": 10.2, "lit_ratio": 0.5, "glow": 9.0, "haze": 0.75, "lod_guard": True, "seed": 21},
}

UP = Vector((0.0, 0.0, 1.0))
SINK = 0.3             # how far dormers are sunk below the nominal roof plane (covers the ridge sag)
KEYS = ("plA", "plB", "brick", "ash", "tile", "slate", "lit", "dim", "dark", "trim")
BT, BU, BD = 0.07, 0.03, 0.2       # barge board: thickness, stand above the tiles, depth below the roof plane
FD = 0.19                          # fascia depth
EI, RI = 0.14, 0.03    # run ends: the end wall stands EI inside the extent (room for its chimney breast, sills and
#                        verge, all of which stop at +-length/2), the roof verge RI inside it (behind the flush stack)
FLAT_UV = (0.3, 0.5)   # a patch of the lit glazing's room that is evenly bright from every angle (see flat_uv())
FLAT_SEED = 0.35
HAZE = "#5d6269"       # night haze: a desaturated grey-blue the far ring's values are pulled towards (never navy)
HAZE_MOON = "#777a80"


def _ss(a, b, x):
    if a == b:
        return 1.0 if x >= b else 0.0
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def _mix(ca, cb, t):
    a = [int(ca.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4)]
    b = [int(cb.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(a, b))


def _hazed(c, grey, t_des, t_val):
    """Colour seen through night haze: its hue drains towards a cool grey of its OWN value (t_des), then the value
    moves a little towards the haze grey (t_val). Pale limewash, brick and stained timber so stay apart in value
    instead of all sinking into one saturated blue."""
    rgb = [int(c.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4)]
    hz = [int(grey.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4)]
    y = 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    cool = (0.96 * y, 0.99 * y, 1.06 * y)
    out = []
    for x, g, h in zip(rgb, cool, hz):
        v = x + (g - x) * t_des
        out.append(min(255, max(0, int(round(v + (h - v) * t_val)))))
    return "#" + "".join(f"{v:02x}" for v in out)


class _MB:
    """Per-material face soup with explicit metre UVs and per-vertex weathering masks (plain quads / n-gons).

    Wall soups (wall=True) are held back until every window unit is known: flush() then opens the wall behind
    each unit, frames the opening with a narrow ring of wall and dices what is left on a coarse world grid. The
    per-vertex masks (the painted weathering and the ambient occlusion baked at export) so sit on vertices that
    see what the wall sees - no vertex hides behind a casing and smears its shadow over the storey."""

    def __init__(self):
        self.v, self.f, self.uv, self.m = [], [], [], []
        self.nt = 0
        self.nk = 0           # triangles of the pinned distance tier (what an LOD keeps)
        self.sub = 0.0
        self.wall = False
        self.pend = []
        self.far = False      # lod 2: the distance tier without its pins, the close tier counted but not built

    def face(self, pts, hint, mask=(0.15, 0.1, 0.0), uvs=None, xf=None, keep=False, uvx=None):
        P = [Vector(q) for q in pts]
        if xf is not None:
            P = [xf(q) for q in P]
        k = len(P)
        n = Vector((0.0, 0.0, 0.0))
        for i in range(k):
            a, b = P[i], P[(i + 1) % k]
            n += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y)))
        if n.length < 1e-8:
            return
        n.normalize()
        ml = list(mask) if isinstance(mask, list) else None
        if n.dot(Vector(hint)) < 0.0:
            P.reverse()
            n = -n
            if uvs is not None:
                uvs = list(reversed(uvs))
            if ml is not None:
                ml.reverse()
        if uvs is None:
            if abs(n.z) < 0.97:
                va = UP - n * n.z
                va.normalize()
                ua = va.cross(n)
            else:
                ua = Vector((1.0, 0.0, 0.0))
                va = Vector((0.0, 1.0 if n.z > 0 else -1.0, 0.0))
            uvs = [(q.dot(ua), q.dot(va)) for q in P]
            if uvx is not None:
                # per-panel placement of the plaster pattern: mirrored, turned a little, scaled and shifted
                c_, s_, ou, ov, fl = uvx
                uvs = [(fl * u * c_ - v * s_ + ou, fl * u * s_ + v * c_ + ov) for u, v in uvs]
        poly = [(q, uv, ml[i] if ml is not None else None) for i, (q, uv) in enumerate(zip(P, uvs))]
        if self.wall:
            self.pend.append((poly, n, mask, keep))
        else:
            self._emit(poly, mask, keep)

    def _emit(self, poly, mask, keep=False):
        k = len(poly)
        ar = Vector((0.0, 0.0, 0.0))
        for i in range(1, k - 1):
            ar += (poly[i][0] - poly[0][0]).cross(poly[i + 1][0] - poly[0][0])
        if k < 3 or ar.length < 2e-5:
            return
        if self.far and not keep:        # (counted: the row must take the same economy decisions as lod 0)
            self.nt += k - 2
            return
        if keep == "far":                # close tier that lod 2 keeps too: a plain face at either level
            keep = False
        i0 = len(self.v)
        for q, uv, m in poly:
            self.v.append((q.x, q.y, q.z))
            self.m.append(m if m is not None else (mask(q) if callable(mask) else mask))
        uvs = [uv for q, uv, m in poly]
        if not keep:
            self.f.append(tuple(range(i0, i0 + k)))
            self.uv.append(uvs)
            self.nt += k - 2
            return
        # distance tier: the face is emitted as a triangle fan, and every fan diagonal carries one PIN - a twin
        # of the fan triangle beside it, in the same plane with the same UVs and masks; only its apex is a vertex
        # of its own, 2 mm inside the corner it doubles (so no weld-by-distance can fold the pair into one). The
        # diagonal so becomes a three-face edge, and an LOD collapse never touches an edge that meets such a
        # vertex: the face keeps its exact outline at every LOD. A pin has no extent of its own - nothing stands
        # behind, beside or through a surface, and no vertex sits out in the middle of a face, where the AO baked
        # at export would differ from the corners' and draw the pin as a paler triangle.
        P = [q for q, uv, m in poly]
        ms = self.m[i0:i0 + k]
        for j in range(1, k - 1):
            self.f.append((i0, i0 + j, i0 + j + 1))
            self.uv.append([uvs[0], uvs[j], uvs[j + 1]])
        self.nt += k - 2
        self.nk += k - 2
        if self.far:                     # no decimator comes after a modelled level: the fan alone (pins counted)
            self.nt += 2 if k == 3 else k - 3
            self.nk += 2 if k == 3 else k - 3
            return

        def pin(ia, ib, ic, inset=0.002):
            wc = 1.0 - min(0.4, inset / max(1e-4, (P[ic] - (P[ia] + P[ib]) * 0.5).length))
            wa = 0.5 * (1.0 - wc)
            q = P[ia] * wa + P[ib] * wa + P[ic] * wc
            self.v.append((q.x, q.y, q.z))
            self.m.append(tuple(ms[ic]))
            self.f.append((i0 + ia, i0 + ib, len(self.v) - 1))
            self.uv.append([uvs[ia], uvs[ib], tuple(uvs[ia][c] * wa + uvs[ib][c] * wa + uvs[ic][c] * wc for c in range(2))])
            self.nt += 1
            self.nk += 1
        if k == 3:
            pin(0, 1, 2)
            pin(0, 1, 2, 0.004)
        else:
            for j in range(2, k - 1):
                pin(0, j, j + 1)

    def _count(self, poly, keep):
        """what _emit would add to nt / nk for this polygon, without building it"""
        k = len(poly)
        ar = Vector((0.0, 0.0, 0.0))
        for i in range(1, k - 1):
            ar += (poly[i][0] - poly[0][0]).cross(poly[i + 1][0] - poly[0][0])
        if k < 3 or ar.length < 2e-5:
            return
        self.nt += k - 2
        if keep and keep != "far":
            e = 2 if k == 3 else k - 3
            self.nt += e
            self.nk += k - 2 + e

    def flush(self, holes, whole=()):
        """whole: indices of the openings this level does not cut - at lod 2 a window unit stands proud on a wall left
        whole (nothing to look past at a raking angle, and the wall is not cut up round every unit). They are COUNTED
        as cut all the same: the row must take the economy decisions of lod 0."""
        S = self.sub
        for poly, n, mask, keep in self.pend:
            def cut(skip):
                polys, hit = [poly], False
                if abs(n.z) < 0.5:
                    for i, ho in enumerate(holes):
                        if n.dot(ho[1]) > 0.999 and abs((poly[0][0] - ho[0]).dot(ho[1])) < 0.004:
                            if i in skip:
                                hit = True
                                continue
                            polys = [c for pl in polys for c in _punch(pl, n, ho)]
                if S > 0.0:
                    if abs(n.z) < 0.97:
                        h = Vector((n.y, -n.x, 0.0))
                        h.normalize()
                        fns = (lambda q: q.z, lambda q: q.dot(h))
                    else:
                        fns = (lambda q: q.x, lambda q: q.y)
                    # (strips - courses, quoins, piers, returns - are not diced: they have no field to shade)
                    if min(max(fn(q[0]) for q in poly) - min(fn(q[0]) for q in poly) for fn in fns) >= 0.6:
                        for fn in fns:
                            polys = [c for pl in polys for c in _dice(pl, fn, S)]
                return polys, hit
            polys, hit = cut(whole)
            if hit:
                nt, nk = self.nt, self.nk
                for pl in polys:
                    self._emit(pl, mask, keep)
                self.nt, self.nk = nt, nk
                for pl in cut(())[0]:
                    self._count(pl, keep)
            else:
                for pl in polys:
                    self._emit(pl, mask, keep)
        self.pend = []


def _cut(poly, fn, c):
    """Split a convex polygon [(point, uv, mask)] by the line fn = c. Returns (part with fn <= c, part with fn >= c)."""
    low, high = [], []
    m = len(poly)
    fs = [fn(q[0]) - c for q in poly]
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        fa, fb = fs[i], fs[(i + 1) % m]
        if abs(fa) < 1e-7:
            fa = 0.0
        if abs(fb) < 1e-7:
            fb = 0.0
        if fa <= 0.0:
            low.append(a)
        if fa >= 0.0:
            high.append(a)
        if fa * fb < 0.0:
            t = fa / (fa - fb)
            x = (a[0].lerp(b[0], t), (a[1][0] + (b[1][0] - a[1][0]) * t, a[1][1] + (b[1][1] - a[1][1]) * t),
                 None if a[2] is None else tuple(u + (v - u) * t for u, v in zip(a[2], b[2])))
            low.append(x)
            high.append(x)
    return (low if len(low) >= 3 else None), (high if len(high) >= 3 else None)


def _dice(poly, fn, S):
    """Split a convex polygon along the grid lines fn = k * S (no slivers under S / 4)."""
    cs = [fn(q[0]) for q in poly]
    lo, hi = min(cs), max(cs)
    if hi - lo <= 1.3 * S:
        return [poly]
    out = []
    k = math.floor((lo + 0.25 * S) / S) + 1
    rest = poly
    while rest is not None and k * S < hi - 0.25 * S:
        low, rest = _cut(rest, fn, k * S)
        if low is not None:
            out.append(low)
        k += 1
    if rest is not None:
        out.append(rest)
    return out


def _punch(poly, n, ho):
    """Open a wall polygon for one window unit. ho = (origin, normal, right, inner (u0, u1, v0, v1), outer rect):
    the wall inside the inner rectangle goes, between inner and outer rectangle it becomes a mitred ring."""
    o, hn, rt, (u0, u1, v0, v1), (U0, U1, V0, V1) = ho

    def fu(q):
        return (q - o).dot(rt)

    def fv(q):
        return q.z - o.z
    us = [fu(q[0]) for q in poly]
    vs = [fv(q[0]) for q in poly]
    if max(us) <= U0 + 1e-4 or min(us) >= U1 - 1e-4 or max(vs) <= V0 + 1e-4 or min(vs) >= V1 - 1e-4:
        return [poly]
    if any(q[2] is not None for q in poly):
        return [poly]
    out = []
    a, rest = _cut(poly, fu, U0)
    if a:
        out.append(a)
    if rest:
        rest, a = _cut(rest, fu, U1)
        if a:
            out.append(a)
    if rest:
        a, rest = _cut(rest, fv, V0)
        if a:
            out.append(a)
    if rest:
        rest, a = _cut(rest, fv, V1)
        if a:
            out.append(a)
    if not rest:
        return out
    if abs(U0 - u0) + abs(U1 - u1) + abs(V0 - v0) + abs(V1 - v1) < 1e-6:
        return out                      # plain opening, no ring
    # affine uv of the wall plane from three corners of the source polygon
    p0 = poly[0]
    best = None
    for i in range(1, len(poly) - 1):
        e1 = (us[i] - us[0], vs[i] - vs[0])
        e2 = (us[i + 1] - us[0], vs[i + 1] - vs[0])
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if best is None or abs(det) > abs(best[0]):
            best = (det, e1, e2, poly[i][1], poly[i + 1][1])
    det, e1, e2, uva, uvb = best
    if abs(det) < 1e-9:
        return out + [rest]
    dpl = (p0[0] - o).dot(hn)

    def vert(u, v):
        du, dv = u - us[0], v - vs[0]
        s_ = (du * e2[1] - dv * e2[0]) / det
        t_ = (dv * e1[0] - du * e1[1]) / det
        uv = tuple(p0[1][k] + s_ * (uva[k] - p0[1][k]) + t_ * (uvb[k] - p0[1][k]) for k in (0, 1))
        return (o + rt * u + UP * v + hn * dpl, uv, None)
    area = sum(fu(rest[i][0]) * fv(rest[(i + 1) % len(rest)][0]) - fu(rest[(i + 1) % len(rest)][0]) * fv(rest[i][0])
               for i in range(len(rest)))
    sg = 1.0 if area > 0 else -1.0
    ring = ([(U0, V0), (U1, V0), (u1, v0), (u0, v0)], [(U1, V0), (U1, V1), (u1, v1), (u1, v0)],
            [(U1, V1), (U0, V1), (u0, v1), (u1, v1)], [(U0, V1), (U0, V0), (u0, v0), (u0, v1)])
    for tr in ring:
        t = [vert(u, v) for u, v in tr]
        if sg < 0:
            t.reverse()
        m = len(rest)
        for i in range(m):
            if t is None:
                break
            A, Bq = rest[i][0], rest[(i + 1) % m][0]
            au, av, bu, bv = fu(A), fv(A), fu(Bq), fv(Bq)
            if abs(bu - au) + abs(bv - av) < 1e-6:
                continue
            t = _cut(t, lambda q, au=au, av=av, bu=bu, bv=bv: sg * ((bu - au) * (fv(q) - av) - (bv - av) * (fu(q) - au)), 0.0)[1]
        if t:
            out.append(t)
    return out


class _Fr:
    """Wall-plane frame: a runs along the wall (to the right seen from outside), b is up, d out of the wall."""

    def __init__(self, o, nrm):
        self.o = Vector(o)
        self.n = Vector(nrm)
        self.rt = UP.cross(self.n)

    def P(self, a, b, d=0.0):
        return self.o + self.rt * a + UP * b + self.n * d


def _shuffled(r, seq):
    a = list(seq)
    for i in range(len(a) - 1, 0, -1):
        j = min(i, int(r.u(0, i + 1)))
        a[i], a[j] = a[j], a[i]
    return a


def build(p, rng, _det=0):
    L = float(p.length)
    far = p.layer == "far"
    lean = bool(p.lod_guard)
    hsc = 1.6 if far else 1.0
    if far:
        n = max(2, int(round(L / 8.5)))
    else:
        n = int(round(p.count * max(1.0, L / 20.0)))
        n = max(int(math.ceil(L / 9.0)), min(n, int(L // 4.6)))
    n = max(n, 1)
    hmin, hmax = sorted((float(p.height_min) * hsc, float(p.height_max) * hsc))
    hmax = max(hmax, hmin + 0.2)
    age = float(p.age)
    sin, cos, tan, pi, sqrt, radians = math.sin, math.cos, math.tan, math.pi, math.sqrt, math.radians
    # DET = level of economy; the build restarts itself one level leaner while the row is over its triangle cap:
    # 1 drops casing returns / hexagonal pots, 2 brackets, shutters and extra glazing bars, 3 thins the windows.
    # lod_guard starts at level 2.
    DET = max(_det, 2 if lean else 0)
    TRI_CAP = min(8200.0, 4000.0 + 120.0 * L)

    PIN = bool(p.lod_pins)
    B = {k: _MB() for k in KEYS}
    for mb_ in B.values():
        mb_.far = lod() == 2
    for k_ in ("plA", "plB", "brick", "ash"):
        B[k_].wall = True
        B[k_].sub = (4.2 if far else 3.0) + 0.6 * min(DET, 2)
    holes = []        # wall openings behind the window units
    whole = set()     # ... those of them that lod 2 does not cut (see _MB.flush)
    XF = [None]
    UVX = [None]      # placement of the plaster pattern on the wall panel being emitted (see _MB.face)
    wins = []         # deferred window units
    dorm_lights = []  # candidate dormer spill lights (x, loc, window dict)
    lcalls = []

    def K(key, pts, hint, mask=(0.15, 0.1, 0.0), uvs=None):
        """Distance tier: everything that has to read from 40-150 m - hull, roof and its edges, stacks, pots,
        casings, panes, mullions. Pinned (see _MB._emit), so it is also exactly what LOD1 / LOD2 show."""
        B[key].face(pts, hint, mask, uvs, XF[0], PIN, UVX[0] if key in ("plA", "plB") else None)

    def D(key, pts, hint, mask=(0.15, 0.1, 0.0), uvs=None):
        """Close-range detail (casing returns, reveals, brackets, fine bars, lamp posts): LOD0 of the near layer only."""
        if not far and p.tier != "distance":
            B[key].face(pts, hint, mask, uvs, XF[0], False)

    def DK(key, pts, hint, mask=(0.15, 0.1, 0.0), uvs=None):
        """Close-range detail that the modelled lod 2 keeps as well (a plain face, never pinned): what still reads
        from 45 m on - the studs, rails and braces of a timber front, open shutters, the finial that is the top of the row's bounds -
        and what stands in the beam of one of the row's own lights (a town light may be given a shadow map, and the
        far level must not let it out past a shadow lod 0 casts)."""
        if not far and p.tier != "distance":
            B[key].face(pts, hint, mask, uvs, XF[0], "far")

    def flat_uv(k):
        """UVs that pin a face of k corners to one small patch of the lit glazing: no room, no curtains - the face
        glows evenly like back-lit parchment or a lantern pane (the patch is small, not a point: tangents stay defined)."""
        return [(FLAT_UV[0] + 0.006 * cos(2 * pi * i / k), FLAT_UV[1] + 0.006 * sin(2 * pi * i / k)) for i in range(k)]
    flat_mask = (0.0, 0.0, FLAT_SEED)

    # ---------------------------------------------------------------- smooth noise for the masks
    rn = rng.fork("noise")
    ph = [rn.u(0.0, 6.283) for _ in range(6)]

    def nz(q, s=1.0):
        x, y, z = q.x / s, q.y / s, q.z / s
        v = (sin(x * 1.7 + ph[0] + 1.3 * sin(y * 0.9 + ph[1])) + sin(y * 2.1 + ph[2] + 1.1 * sin(z * 1.3 + ph[3]))
             + sin(z * 1.9 + ph[4] + 1.2 * sin(x * 1.1 + ph[5])))
        return min(1.0, max(0.0, 0.5 + v / 3.6))

    trim_mask = (0.08, 0.0, 0.0)        # stained timber stays even: no blotches on beams
    join_mask = (0.0, 0.0, 0.0)         # window joinery: one even dark stain (only the sill nosing is rubbed pale)
    ruv = rng.fork("plaster")

    def new_uvx():
        """Every wall face / infill panel lays the limewash pattern its own way: turned, mirrored, scaled, shifted."""
        an = radians(ruv.u(-16.0, 16.0)) + (pi if ruv.chance(0.5) else 0.0)
        sc = ruv.u(0.8, 1.2)
        UVX[0] = (cos(an) * sc, sin(an) * sc, ruv.u(0.0, 2.0), ruv.u(0.0, 2.0), -1.0 if ruv.chance(0.5) else 1.0)

    soffit_mask = (0.55, 0.05, 0.0)
    fmask = (0.2, 0.12, 0.1 * age)      # stone surrounds, sills and lintels
    TD = 0.045 if far else 0.022         # timber / dressing proud of the wall plane
    TW = 0.13 if far else 0.1            # half the depth of a rail, half the width of a post + 1 cm

    # ---------------------------------------------------------------- primitive emitters
    def boxf(E, key, lo, hi, faces, mask):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        if "x" in faces:
            E(key, [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0), mask)
        if "X" in faces:
            E(key, [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0), mask)
        if "y" in faces:
            E(key, [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0), mask)
        if "Y" in faces:
            E(key, [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0), mask)
        if "z" in faces:
            E(key, [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1), mask)
        if "Z" in faces:
            E(key, [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1), mask)

    def pbox(E, key, W, a0, a1, b0, b1, d0, d1, faces, mask):
        """Box in a wall-plane frame: a along the wall (right), b up, d out of the wall."""
        P = W.P
        if "f" in faces:
            E(key, [P(a0, b0, d1), P(a1, b0, d1), P(a1, b1, d1), P(a0, b1, d1)], W.n, mask)
        if "t" in faces:
            E(key, [P(a0, b1, d0), P(a1, b1, d0), P(a1, b1, d1), P(a0, b1, d1)], UP, mask)
        if "b" in faces:
            E(key, [P(a0, b0, d0), P(a1, b0, d0), P(a1, b0, d1), P(a0, b0, d1)], -UP, mask)
        if "l" in faces:
            E(key, [P(a0, b0, d0), P(a0, b1, d0), P(a0, b1, d1), P(a0, b0, d1)], -W.rt, mask)
        if "r" in faces:
            E(key, [P(a1, b0, d0), P(a1, b1, d0), P(a1, b1, d1), P(a1, b0, d1)], W.rt, mask)

    def rect(E, key, W, a0, a1, b0, b1, d, mask):
        E(key, [W.P(a0, b0, d), W.P(a1, b0, d), W.P(a1, b1, d), W.P(a0, b1, d)], W.n, mask)

    def beam(key, W, pa, pb, wd, d, mask):
        """Timber member lying on the wall plane between two (a, b) points (any direction)."""
        (a0, b0), (a1, b1) = pa, pb
        ln = math.hypot(a1 - a0, b1 - b0)
        if ln < 1e-3:
            return
        nx, ny = -(b1 - b0) / ln * wd / 2, (a1 - a0) / ln * wd / 2
        c = [(a0 + nx, b0 + ny), (a1 + nx, b1 + ny), (a1 - nx, b1 - ny), (a0 - nx, b0 - ny)]
        DK(key, [W.P(a, b, d) for a, b in c], W.n, mask)

    def prism(E, key, cx, cy, z0, z1, r0, r1, nside, mask, a0=0.0, cap=True):
        st = 2 * pi / nside
        lo = [Vector((cx + r0 * cos(a0 + k * st), cy + r0 * sin(a0 + k * st), z0)) for k in range(nside)]
        hi = [Vector((cx + r1 * cos(a0 + k * st), cy + r1 * sin(a0 + k * st), z1)) for k in range(nside)]
        for k in range(nside):
            j = (k + 1) % nside
            am = a0 + (k + 0.5) * st
            if r1 < 1e-4:
                E(key, [lo[k], lo[j], Vector((cx, cy, z1))], (cos(am), sin(am), 0.3), mask)
            else:
                E(key, [lo[k], lo[j], hi[j], hi[k]], (cos(am), sin(am), 0.0), mask)
        if cap and r1 >= 1e-4:
            E(key, hi, (0, 0, 1), mask)

    rv_ = rng.fork("roofvar")

    def roof(rk, axis, a0, a1, c0, c1, He, tanp, oe0, oe1, ov0, ov1, sag, nseg, rmask, hip=(False, False), hf=0.0, r=None,
             barge=(True, True), vk="trim"):
        """Gabled roof, ridge along `axis` from a0..a1 (+ verge overhangs ov), spanning c0..c1 (+ eaves overhangs
        oe), meeting the wall heads at He. hip = jerkinhead at end 0 / 1 over the top `hf` of the roof height.
        The ridge sags by `sag` at mid-span and wobbles a little. The whole roof is seated a few cm above the wall
        heads - by as much as the sag pulls the slopes down over the wall lines - so no wall head ever shows
        through a slope. Every slope is laid in bands from eaves to ridge - a sprocketed (bell-cast) eaves skirt at
        a flatter pitch, the lower courses, the upper courses - and each band of each bay carries its own tone
        (damp mossy eaves courses, re-laid and weathered patches), so a slope never reads as one flat plane.
        Slopes, fascias, soffits, barge boards (key `vk`) and the ridge cap are distance tier."""
        vmask = trim_mask if vk == "trim" else (0.2, 0.1, 0.1 * age)

        def tone(dg, dw, ds):
            def f(q):
                g, w_, s_ = rmask(q)
                return (min(1.0, max(0.0, g + dg)), min(1.0, max(0.0, w_ + dw)), min(1.0, max(0.0, s_ + ds)))
            return f
        cm, half = 0.5 * (c0 + c1), 0.5 * (c1 - c0)
        sg_ = sag + 0.05 * age
        He = He + 0.012 + sg_ * max(max(oe0, oe1) / (max(oe0, oe1) + half), max(ov0, ov1) * nseg / max(1.0, a1 - a0 + ov0 + ov1))
        Hr = He + half * tanp
        if axis == "x":
            def W(a, c, z):
                return Vector((a, c, z))
        else:
            def W(a, c, z):
                return Vector((c, a, z))
        A = (a0 - ov0, a1 + ov1)
        zmid = Hr - hf * (Hr - He)
        e = hf * (Hr - He) / (tanp * 1.15)
        ins = (e if hip[0] else 0.0, e if hip[1] else 0.0)
        chh = (Hr - zmid) / tanp
        wob = [0.0] + [(r.u(-1.0, 1.0) if r else 0.0) * 0.04 * age for _ in range(nseg - 1)] + [0.0]

        def rdg(i):
            t = i / nseg
            lo, hi = A[0] + ins[0], A[1] - ins[1]
            return (lo + (hi - lo) * t, cm, Hr - sag * sin(pi * t) + wob[i])

        for side, oe in ((-1, oe0), (1, oe1)):
            ce = (c0 - oe) if side < 0 else (c1 + oe)
            cw = c0 if side < 0 else c1
            ze = He - oe * tanp
            run = half + oe
            upn = W(0.0, side * tanp, 1.0)
            # sprocketed eaves: the bottom courses kick out at a flatter pitch (only where there is an eaves overhang)
            sk = min(0.3 * run, oe + 0.42) if oe > 0.02 else 0.0
            zel = ze + 0.3 * sk * tanp
            ck, zk = ce - side * sk, ze + sk * tanp
            sm = 0.56
            cmid = ce - side * sm * run

            def ax(i):
                return A[0] + (A[1] - A[0]) * i / nseg

            def zm(i):
                return ze + sm * (rdg(i)[2] - ze)
            for i in range(nseg):
                if sk > 0.0:
                    K(rk, [W(ax(i), ce, zel), W(ax(i + 1), ce, zel), W(ax(i + 1), ck, zk), W(ax(i), ck, zk)], upn,
                      tone(0.14, 0.0, 0.35 * age))
                c_lo, z_lo = (ck, zk) if sk > 0.0 else (ce, ze)
                K(rk, [W(ax(i), c_lo, z_lo), W(ax(i + 1), c_lo, z_lo), W(ax(i + 1), cmid, zm(i + 1)), W(ax(i), cmid, zm(i))], upn,
                  tone(rv_.u(-0.1, 0.16), rv_.u(0.0, 0.2), 0.0))
                poly = [(ax(i), cmid, zm(i)), (ax(i + 1), cmid, zm(i + 1))]
                if i == nseg - 1 and hip[1]:
                    poly.append((A[1], cm + side * chh, zmid))
                poly += [rdg(i + 1), rdg(i)]
                if i == 0 and hip[0]:
                    poly.append((A[0], cm + side * chh, zmid))
                K(rk, [W(*q) for q in poly], upn, tone(rv_.u(-0.12, 0.1), rv_.u(0.05, 0.35), 0.0))
            if oe > 0.02:
                # fascia and a closed eaves soffit back to the wall (the roof so has a real, thick edge at any distance)
                K("trim", [W(A[0], ce, zel - FD), W(A[1], ce, zel - FD), W(A[1], ce, zel), W(A[0], ce, zel)], W(0, side, 0), trim_mask)
                K("trim", [W(A[0], ce, zel - FD), W(A[1], ce, zel - FD), W(A[1], cw, zel - FD), W(A[0], cw, zel - FD)],
                  (0, 0, -1), soffit_mask)
            dnn = W(0, -side * tanp, -1.0)
            for k, (Ak, sgn) in enumerate(((A[0], -1), (A[1], 1))):
                if not barge[k]:
                    continue
                # barge board: a real plank (BT thick, BU + BD deep) sitting on the tile edge, with the verge
                # overhang closed by a raking soffit between the board and the gable wall
                zt = zmid if hip[k] else Hr
                ct = cm + side * (Hr - zt) / tanp
                Ai = Ak - sgn * BT
                Aw = Ak - sgn * max((ov0, ov1)[k], 0.07)
                K(vk, [W(Ak, ce, ze - BD), W(Ak, ct, zt - BD), W(Ak, ct, zt + BU), W(Ak, ce, ze + BU)], W(sgn, 0, 0), vmask)
                DK(vk, [W(Ak, ce, ze - BD), W(Ak, ct, zt - BD), W(Ai, ct, zt - BD), W(Ai, ce, ze - BD)], dnn, soffit_mask)   # (seen from the street: lod 2 too)
                D(vk, [W(Ak, ce, ze + BU), W(Ak, ct, zt + BU), W(Ai, ct, zt + BU), W(Ai, ce, ze + BU)], upn, vmask)
                D(vk, [W(Ai, ce, ze - BD), W(Ai, ct, zt - BD), W(Ai, ct, zt + BU), W(Ai, ce, ze + BU)], W(-sgn, 0, 0), vmask)
                if abs(Aw - Ai) > 0.02:
                    K("trim", [W(Ai, ce, ze - 0.07), W(Ai, ct, zt - 0.07), W(Aw, ct, zt - 0.07), W(Aw, ce, ze - 0.07)], dnn, soffit_mask)
                if oe > 0.02:
                    # boxed eaves return: closes the end of the soffit box below the board, and the end of the
                    # sprocketed skirt above it
                    Am = Ak - sgn * 0.5 * BT
                    K(vk, [W(Am, ce, zel - FD), W(Am, cw, zel - FD), W(Am, cw, He), W(Am, ce, zel)], W(sgn, 0, 0), vmask)
                    if sk > abs(cw - ce) + 0.05:
                        K(vk, [W(Am, cw, He), W(Am, ck, zk), W(Am, ce, zel)], W(sgn, 0, 0), vmask)
                else:
                    D(vk, [W(Ak, ce, ze - BD), W(Ai, ce, ze - BD), W(Ai, ce, ze + BU), W(Ak, ce, ze + BU)], W(0, side, 0), vmask)
        for k, sgn in ((0, -1), (1, 1)):
            if not hip[k]:
                continue
            Ak = A[k]
            ar = Ak - sgn * ins[k]
            K(rk, [W(Ak, cm - chh, zmid), W(Ak, cm + chh, zmid), W(ar, cm, Hr)], W(sgn * 1.0, 0, 0.8),
              tone(rv_.u(-0.05, 0.2), rv_.u(0.0, 0.2), 0.15 * age))
            if barge[k]:
                Ai = Ak - sgn * BT
                Aw = Ak - sgn * max((ov0, ov1)[k], 0.07)
                c_a, c_b = cm - chh, cm + chh
                K(vk, [W(Ak, c_a, zmid - BD), W(Ak, c_b, zmid - BD), W(Ak, c_b, zmid + BU), W(Ak, c_a, zmid + BU)], W(sgn, 0, 0), vmask)
                DK(vk, [W(Ak, c_a, zmid - BD), W(Ak, c_b, zmid - BD), W(Ai, c_b, zmid - BD), W(Ai, c_a, zmid - BD)], (0, 0, -1), soffit_mask)
                D(vk, [W(Ak, c_a, zmid + BU), W(Ak, c_b, zmid + BU), W(Ai, c_b, zmid + BU), W(Ai, c_a, zmid + BU)], (0, 0, 1), vmask)
                if abs(Aw - Ai) > 0.02:
                    D(vk, [W(Ai, c_a, zmid - BD), W(Ai, c_b, zmid - BD), W(Ai, c_b, zmid), W(Ai, c_a, zmid)], W(-sgn, 0, 0), vmask)
                    K("trim", [W(Ai, c_a, zmid - 0.07), W(Ai, c_b, zmid - 0.07), W(Aw, c_b, zmid - 0.07), W(Aw, c_a, zmid - 0.07)],
                      (0, 0, -1), soffit_mask)
        # ridge cap (chevron of clay ridge tiles following the sag; clay also on slate - a warm line along every
        # ridge). It stands 13 cm proud: a ridge line that holds at distance; its two ends are closed
        rw_, rh_ = 0.25, 0.13
        for i in range(nseg):
            (aa, _c, za), (ab, _c2, zb) = rdg(i), rdg(i + 1)
            for side in (-1, 1):
                K(RIDGE[0] or rk, [W(aa, cm + side * rw_, za - rw_ * tanp + 0.05), W(ab, cm + side * rw_, zb - rw_ * tanp + 0.05),
                           W(ab, cm, zb + rh_), W(aa, cm, za + rh_)], W(0, side * tanp, 1.0), (0.3, 0.2, 0.3 * age))
        for i, sgn in ((0, -1), (nseg, 1)):
            aa, _c, za = rdg(i)
            DK(RIDGE[0] or rk, [W(aa, cm - rw_, za - rw_ * tanp + 0.05), W(aa, cm + rw_, za - rw_ * tanp + 0.05), W(aa, cm, za + rh_)],
              W(sgn, 0, 0), (0.3, 0.2, 0.0))

    RIDGE = [None]    # clay ridge tiles on every roof, slate ones included

    def chimney(cx, cy, zb, zt, w, d, npots, r, cmask, ck="brick", fl=0):
        """Stout brick stack (pinned shaft) with an oversailing cap and a row of tapered pots (detail).
        fl = -1 / +1: the stack stands flush on a run end (its -X / +X face in the end plane): plumb, the cap does
        not oversail on that side."""
        lean_ = float(p.stack_lean)
        ax, ay = radians(r.u(-1.3, 1.3) * (0.3 + age)) * lean_, radians(r.u(-1.3, 1.3) * (0.3 + age)) * lean_
        org = Vector((cx, cy, zb))

        def xf(q):
            v = q - org
            return org + Vector((v.x + v.z * ay, v.y + v.z * ax, v.z))
        XF[0] = xf if (lean_ > 1e-3 and not fl) else None          # plumb unless stack_lean asks otherwise
        hw, hd = w / 2, d / 2
        oc = 0.09
        boxf(K, ck, (cx - hw, cy - hd, zb), (cx + hw, cy + hd, zt - 0.24), "xXyY", cmask)
        # oversailing cap: two courses stepping out, flaunched top (all in the distance tier)
        c0x, c1x = cx - hw - (0.0 if fl < 0 else oc), cx + hw + (0.0 if fl > 0 else oc)
        boxf(K, ck, (c0x, cy - hd - oc, zt - 0.24), (c1x, cy + hd + oc, zt - 0.06), "xXyYZ", cmask)
        boxf(DK, ck, (c0x, cy - hd - oc, zt - 0.24), (c1x, cy + hd + oc, zt - 0.06), "z", cmask)     # its underside: what the street sees of it
        along_x = w >= d
        span = (w if along_x else d) - 0.3
        if DET >= 2:
            npots = min(npots, 3 if DET == 2 else 2)
        for k in range(npots):
            t = 0.0 if npots == 1 else (k / (npots - 1) - 0.5)
            px, py = (cx + t * span, cy) if along_x else (cx, cy + t * span)
            hp = r.u(0.38, 0.52) if not r.chance(0.3) else r.u(0.62, 0.8)
            rp = r.u(0.125, 0.15)
            # square tapered pot with a dark flue (pots: terracotta red or buff, sooted towards the rim)
            prism(K, "brick", px, py, zt - 0.06, zt + hp, rp, rp * 0.76, 4, (0.35, 0.1, 0.0), a0=pi / 4, cap=False)
            D("dark", [(px + rp * 0.7 * cos(pi / 4 + j * pi / 2), py + rp * 0.7 * sin(pi / 4 + j * pi / 2), zt + hp - 0.03) for j in range(4)],
              (0, 0, 1), (0.9, 0.0, 0.0))
            if DET == 0:
                prism(D, "brick", px, py, zt + hp - 0.07, zt + hp - 0.02, rp * 0.9, rp * 0.9, 4, (0.5, 0.1, 0.0), a0=pi / 4, cap=False)
        XF[0] = None

    # ---------------------------------------------------------------- layout
    rl = rng.fork("layout")
    wmin = min(5.2 if far else 3.7, L / n * 0.85)
    wts = [rl.u(0.6, 1.4) for _ in range(n)]
    sw = sum(wts)
    widths = [wmin + (L - n * wmin) * w / sw for w in wts]
    xs = [-L / 2]
    for w in widths:
        xs.append(xs[-1] + w)
    xs[-1] = L / 2
    hs = sorted(hmin + (hmax - hmin) * (i + rl.u(0.15, 0.85)) / n for i in range(n))
    lo, hi = _shuffled(rl, hs[:n // 2]), _shuffled(rl, hs[n // 2:])
    first_hi = rl.chance(0.5)
    heights = []
    for i in range(n):
        src = (hi, lo) if (i % 2 == 0) == first_hi else (lo, hi)
        heights.append(src[0].pop() if src[0] else src[1].pop())
    sgn0 = 1 if rl.chance(0.5) else -1

    houses = []
    prev = None
    for i in range(n):
        r = rng.fork(f"house{i}")
        w = xs[i + 1] - xs[i]
        opts = [("side", 5.0)]
        if w <= 8.6 * (1.3 if far else 1.0):
            opts.append(("front", 4.5))
        if w >= 5.7:
            opts.append(("cross", 3.0))
        typ = r.weighted(opts)
        if typ == prev and r.chance(0.6):
            typ = r.weighted(opts)
        prev = typ
        if p.walls == "plaster":
            wk = r.pick(["plA", "plB"])
        elif p.walls == "brick":
            wk = "brick"
        else:
            wk = r.weighted([("plA", 4), ("plB", 3), ("brick", 3)])
        ssh = float(p.slate_share)
        rk = p.roofs if p.roofs != "mixed" else r.weighted([("tile", max(1e-3, 1.0 - ssh)), ("slate", max(1e-3, ssh))])
        d = float(p.depth) * r.u(0.86, 1.12) * (1.25 if far else 1.0)
        off = float(p.stagger) * sgn0 * (1 if i % 2 == 0 else -1) * r.u(0.45, 1.0)
        tanp = tan(radians(min(62.0, max(35.0, p.roof_pitch + r.u(-7.0, 7.0)))))
        houses.append(dict(i=i, r=r, x0=xs[i], x1=xs[i + 1], w=w, typ=typ, wk=wk, rk=rk, yf=-d / 2 + off, yb=d / 2 + off,
                           He=heights[i], tanp=tanp))
    if p.roofs == "mixed":
        # a fixed share of the run is under slate (stratified, not per-house chance: a short run never comes
        # out all clay or all slate unless slate_share says so)
        ns_ = int(round(float(p.slate_share) * n))
        for k, i_ in enumerate(_shuffled(rng.fork("roofs"), list(range(n)))):
            houses[i_]["rk"] = "slate" if k < ns_ else "tile"
    RIDGE[0] = "tile"
    # the landmark needs an eaves-on host, preferably not at a run end
    host = None
    style = p.landmark
    if style != "none":
        inner = houses[1:-1] if n >= 3 else houses
        cands = [h for h in inner if h["typ"] == "side" and h["w"] >= 5.0]
        host = rl.pick(cands) if cands else max(inner, key=lambda h: h["w"])
        host["typ"] = "side"
    if not any(h["typ"] in ("side", "cross") for h in houses):
        max(houses, key=lambda h: h["w"])["typ"] = "side"
    for H in houses:
        H["Hr0"] = H["He"] + (H["w"] / 2 if H["typ"] == "front" else (H["yb"] - H["yf"]) / 2) * H["tanp"]

    def reg(H, W, a, b, w, h, kind=None, back=False, top=False, fz=0.0, sill=True, **kw):
        """Register a window / door unit on frame W (emitted once the lit / dark split is known)."""
        brick = H["wk"] == "brick"
        proud = back or kw.get("rec", 1.0) <= 0.0
        # units laid on the wall (backs, end walls) always get the pale casing: a dark frame round dark glass would
        # be a black patch pasted on the wall
        fk = kw.get("fk", H["pk"] if proud else H["fk"])
        wd = dict(W=W, a=a, b=b, w=w, h=h, kind=kind, back=back, top=top, fz=fz, sill=sill and not back, hi=H["i"],
                  bars=H["bars"], brick=brick, fk=fk, sk=fk, eo=0.1 if brick else 0.03, shut=False, tiny=far,
                  wkey=H["wk"], rec=0.0 if back else (0.11 if brick else 0.085), rm=0.2, arch=False, grp=None)
        wd.update(kw)
        if wd["back"]:
            wd["rec"] = 0.0
        if "ec" not in wd:
            # width of casing seen round the pane: pale casings are stout enough to hold at distance
            wd["ec"] = 0.075 if wd["fk"] == "trim" else (0.14 if far else 0.12)
        wins.append(wd)
        return wd

    # ---------------------------------------------------------------- houses
    ridge_max = 0.0
    for H in houses:
        r = H["r"]
        i, x0, x1, yf, yb, He, tanp, wk, rk, w = (H[k] for k in ("i", "x0", "x1", "yf", "yb", "He", "tanp", "wk", "rk", "w"))
        typ = H["typ"]
        xm = 0.5 * (x0 + x1)
        ns = max(1, int(He / (3.2 if far else 2.9) + 0.5))
        fh = He / ns
        rs_ = r.fork("style")
        timber = (not far) and wk != "brick" and rs_.chance(0.6)
        stepped = typ == "front" and wk == "brick" and rs_.chance(0.75)
        H["bars"] = rs_.weighted([((1, 1), 5), ((0, 1), 2), ((1, 2), 3)] + ([((2, 2), 2)] if DET == 0 else []))
        H["ww"], H["wh"] = rs_.u(0.8, 1.05), rs_.u(1.2, 1.5)
        if far:
            H["bars"] = (1, 1)
            # the far ring keeps its share of half-timbered fronts: stout posts and rails that hold through the haze
            timber = wk != "brick" and r.fork("fartimber").chance(0.45)
        # window casings: pale (stone surround on limewash and brick, limewashed on the darker render) so an unlit
        # window is dark glass in a light frame; only half-timbered fronts keep stained oak frames between their posts
        H["pk"] = "ash" if wk == "plA" else "plA"
        H["fk"] = "trim" if (timber and not far) else H["pk"]
        vk_ = "plA" if wk == "brick" else "trim"        # verges: pale painted boards on brick, stained oak on limewash
        shut = (not far) and (not timber) and wk != "brick" and DET <= 2 and rs_.chance(0.5)
        shed = rs_.chance(0.3)
        jet = 0.0
        if timber and not far and ns >= 2 and typ != "cross" and H is not host and p.jetty > 0.02 and rs_.chance(0.8):
            jet = float(p.jetty) * rs_.u(0.85, 1.1)
        yfu = yf - jet
        if typ == "front":
            if w / 2 * tanp > 6.2 * hsc:
                tanp = 6.2 * hsc / (w / 2)
            half = w / 2
        else:
            half = (yb - yfu) / 2
        ym = 0.5 * (yfu + yb)
        Hr = He + half * tanp
        ridge_max = max(ridge_max, Hr)
        H.update(tanp=tanp, Hr=Hr, ns=ns, fh=fh, yfu=yfu, ymin=yfu)

        # ---- jerkinhead ends
        rh = r.fork("hip")
        hip = [False, False]
        if typ == "front" and not stepped:
            hip[0] = rh.chance(float(p.hips))
        elif typ == "side":
            for e_, nb in ((0, i - 1), (1, i + 1)):
                exposed = not (0 <= nb < n) or houses[nb]["Hr0"] < He + 0.45 * (Hr - He)
                if exposed and rh.chance(float(p.hips)):
                    hip[e_] = True
        hf = rh.u(0.3, 0.42) if any(hip) else 0.0
        zmid = Hr - hf * (Hr - He)
        e_hip = hf * (Hr - He) / (tanp * 1.15)

        if typ == "front":
            def top(x, y, Hr=Hr, xm=xm, tanp=tanp):
                return Hr - abs(x - xm) * tanp
        else:
            def top(x, y, Hr=Hr, ym=ym, tanp=tanp):
                return Hr - abs(y - ym) * tanp
        H.update(top=top)

        # ---- landmark bay (keeps windows / dormers / chimneys clear)
        tw = None
        if host is H:
            left = r.chance(0.5)
            if n > 1 and i in (0, n - 1):
                left = i != 0                   # on an end house the landmark takes the inner side: nothing of it
                #                                 (plinth, cornice, roof flare) can cross the run end
            if style == "turret":
                wt = 2.0 * min(1.5, max(1.2, 0.24 * w))
                tcx = (x0 + 0.1 + wt / 2) if left else (x1 - 0.1 - wt / 2)
                tw = (tcx, yfu - 0.08 * wt, wt)
            else:
                wt = min(3.4, max(2.6, 0.4 * w)) * (1.3 if far else 1.0)
                tcx = (x0 + 0.45 + wt / 2) if left else (x1 - 0.45 - wt / 2)
                tw = (tcx, yfu - 0.34 * wt + wt / 2, wt)
            H["tower"] = tw
            H["ymin"] = tw[1] - wt / 2 - 0.25

        # ---- cross-gable wing
        wing = None
        if typ == "cross":
            ww_ = min(5.2, max(2.7, w * r.u(0.42, 0.52)))
            while He + ww_ / 2 * tanp > Hr - 0.35 and ww_ > 2.4:
                ww_ -= 0.2
            xc = x0 + w * (r.u(0.3, 0.38) if r.chance(0.5) else r.u(0.62, 0.7))
            xa, xb = max(x0 + 0.35, xc - ww_ / 2), min(x1 - 0.35, xc + ww_ / 2)
            yw = yf - r.u(0.7, 1.1)
            wing = (xa, xb, yw, He + (xb - xa) / 2 * tanp)
            H["ymin"] = yw

        def clear(x, m):
            if tw and abs(x - tw[0]) < tw[2] / 2 + m:
                return False
            if wing and wing[0] - m - 0.3 < x < wing[1] + m + 0.3:
                return False
            return True

        # ---- chimney positions (chosen first: the roof soot mask needs them)
        rc = r.fork("chim")
        chims = []
        nst = 1 + (1 if (w > 4.8 and rc.chance(0.8)) else 0) + (1 if (w > 8.0 and rc.chance(0.5)) else 0)
        ends = _shuffled(rc, [-1, 1])
        for k in range(nst):
            e_ = ends[k % 2]
            big = (rc.u(0.72, 0.95) if k == 0 else rc.u(0.64, 0.8)) * (1.25 if far else 1.0)
            small = rc.u(0.54, 0.66) * (1.25 if far else 1.0)
            if typ == "front":
                cw_, cd_ = small, big
                cx = xm + e_ * rc.u(0.3, 0.55) * w / 2
                cy = ym + (rc.u(-0.12, 0.3) if k < 2 else rc.u(-0.36, -0.24)) * (yb - yfu)
                cand = [cx, x0 + x1 - cx]
            else:
                cw_, cd_ = big, small
                inset = 0.45 + (e_hip + 0.25 if hip[0 if e_ < 0 else 1] else 0.0)
                cx = x0 + inset + cw_ / 2 if e_ < 0 else x1 - inset - cw_ / 2
                cy = ym + (rc.u(-0.08, 0.08) if rc.chance(0.65) else rc.u(0.8, 1.5))
                cand = ([cx, x0 + x1 - cx] if k < 2 else []) + [x0 + w * f_ for f_ in (rc.u(0.42, 0.58), 0.32, 0.68)]
            cm_ = cw_ / 2 + (1.2 if tw else 0.8)
            cand = [x for x in cand if clear(x, cm_) and all(math.hypot(x - c[0], cy - c[1]) > 1.5 for c in chims)]
            if not cand and wing is not None and typ != "front":
                # beside a cross wing the stack stands on the back slope instead
                cy = ym + rc.u(0.9, 1.4)
                cand = [x for x in (cx, x0 + x1 - cx) if all(math.hypot(x - c[0], cy - c[1]) > 1.5 for c in chims)]
            if not cand:
                continue
            cx = cand[0]
            zt = Hr + (rc.u(1.2, 1.9) if k == 0 else rc.u(0.9, 1.4)) * (1.3 if far else 1.0)
            npots = max(2, min(4, int(round(max(cw_, cd_) / 0.26))))
            chims.append((cx, cy, cw_, cd_, zt, top(cx, cy) - 0.9, npots, 0))

        # ---- run ends: a brick chimney breast on the exposed end wall, narrowing at a shoulder into a stack that
        # stands flush with the end (it replaces any ordinary stack near that end)
        ren = r.fork("endwall")
        rb_ = r.fork("back")
        fsc = 1.25 if far else 1.0
        ebr = {}
        for e_, (sx, xe, nb) in enumerate(((-1, x0, i - 1), (1, x1, i + 1))):
            if (0 <= nb < n) or not bool(p.end_stacks) or (tw and abs(xe - tw[0]) < tw[2] / 2 + 1.5):
                continue
            wx = ren.u(0.56, 0.64) * fsc
            wu = ren.u(0.9, 1.15) * fsc
            wl = wu + ren.u(0.45, 0.7)
            cx = xe - sx * wx / 2
            if typ == "front":
                cy = ym + ren.u(-0.22, 0.22) * (yb - yfu)
                zt = min(Hr + 0.5, He + 4.4 * fsc)
            else:
                sb = 1 if (typ == "cross" or ren.chance(0.6)) else -1
                cy = ym + sb * ren.u(1.05, 1.6)
                zt = Hr + ren.u(0.9, 1.4) * (1.3 if far else 1.0)
            zsh = max(1.8, min(He - ren.u(0.3, 0.9), top(cx, cy) - 1.6))
            chims[:] = [c for c in chims if abs(c[0] - xe) > 2.4]
            chims.append((cx, cy, wx, wu, zt, zsh + 0.6, max(2, min(4, int(round(wu / 0.28)))), sx))
            ebr[e_] = (cy * sx, wl, wu, zsh)

        H["chims"] = chims

        # ---- weathering masks: clean mid-wall, damp foot, dirt under eaves / verges
        soot = r.u(0.35, 0.6)
        brickw = wk == "brick"

        def wmask(q, He=He, Hr=Hr, typ=typ, xm=xm, ym=ym, tanp=tanp, soot=soot, brickw=brickw):
            # the field of a wall stays clean; dirt is painted only where it belongs - the bottom metre and a narrow
            # band under eaves / verges (brick keeps a faint even soot film)
            g = (0.08 + 0.08 * nz(q, 2.6)) if brickw else 0.0
            g = max(g, 0.26 * _ss(1.0, 0.0, q.z) * (0.7 + 0.3 * nz(q, 0.9)))
            if q.z <= He + 0.05:
                g = max(g, soot * 0.22 * _ss(He - 0.5, He - 0.03, q.z))
            else:
                rl_ = Hr - (abs(q.x - xm) if typ == "front" else abs(q.y - ym)) * tanp
                g = max(g, soot * 0.16 * _ss(0.4, 0.0, rl_ - q.z))
            sp = age * (0.8 if brickw else 0.45) * _ss(1.0, 0.0, q.z) * (0.5 + 0.5 * nz(q, 1.7)) * (1.0 if q.y > 0 else 0.6)
            return (min(1.0, g), (0.25 if brickw else 0.1) * nz(q, 0.9) * _ss(2.0, 0.0, q.z), sp)

        def dmask(q):
            return (0.06 + 0.1 * nz(q, 1.3), 0.1, 0.0)

        def rmask(q, He=He, Hr=Hr, ymid=0.5 * (yf + yb), chims=chims):
            k = min(1.0, max(0.0, (q.z - He) / max(Hr - He, 0.1)))
            g = 0.1 + 0.3 * nz(q, 2.6) * (0.4 + 0.6 * k)
            for c in chims:
                g = max(g, 0.75 * _ss(2.4, 0.4, math.hypot(q.x - c[0], q.y - c[1])))
            m = age * min(1.0, 1.7 * nz(q, 1.9)) * (1.0 - 0.6 * k) * (1.0 if q.y > ymid else 0.55)
            return (min(1.0, g), 0.2 * nz(q, 0.7), m)

        # ---- wall frames
        Wlo = _Fr((0, yf, 0), (0, -1, 0))
        Whi = _Fr((0, yfu, 0), (0, -1, 0))
        Wbk = _Fr((0, yb, 0), (0, 1, 0))

        def Wfront(f):
            return Wlo if f == 0 else Whi

        def bands(W, a0, a1, z0, z1, foot=True, eaves=True, cols=None, rails=()):
            """One wall plane as two or three stacked quads, so the damp foot and the soot under the eaves stay
            in narrow bands instead of grading over the whole height. cols / rails = post and rail lines of a
            timber frame: the wall is then laid panel by panel (seams under the timbers), each infill panel with
            its own placement of the plaster pattern."""
            cuts = [z0]
            if foot and z1 - z0 > 2.2:
                cuts.append(z0 + 0.85)
            for zr_ in rails:
                if cuts[-1] + 0.3 < zr_ < z1 - 0.3:
                    cuts.append(zr_)
            if eaves and z1 - cuts[-1] > 1.3:
                cuts.append(z1 - 0.5)
            cuts.append(z1)
            xs_ = [a0] + [c for c in (cols or ()) if a0 + 0.3 < c < a1 - 0.3] + [a1]
            for ca, cb in zip(xs_, xs_[1:]):
                new_uvx()
                for ba, bb in zip(cuts, cuts[1:]):
                    if any(abs(ba - zr_) < 1e-6 for zr_ in rails):
                        new_uvx()
                    rect(K, wk, W, ca, cb, ba, bb, 0.0, wmask)

        def strip(Wof, a_lo, a_hi, back, door, skip=None, bdoor=None):
            """One wall strip with its window / door units, storey by storey. skip = a-range of the ground floor
            hidden by an outshot; bdoor = where along a back wall its door stands (0..1)."""
            span = a_hi - a_lo
            nc, sp, ww = 0, span, 0.0
            if span >= 1.5:
                nc = max(1, int(round(span / ((2.7 if far else 2.3) if DET < 3 else 3.2))))
                sp = span / nc
                ww = min(H["ww"], sp - 0.8)
                while ww < 0.66 and nc > 1:
                    nc -= 1
                    sp = span / nc
                    ww = min(H["ww"], sp - 0.8)
                if ww < 0.66:
                    nc, sp = 0, span
            street = Wof is Wfront
            framed = timber and street and wing is None and tw is None and nc > 0
            cols = [a_lo + k * sp for k in range(1, nc)] if framed else None
            if back:
                bands(Wof(0), a_lo, a_hi, 0.0, He, foot=False, eaves=False)
            elif far:
                bands(Wof(0), a_lo, a_hi, 0.0, He)
            elif street and ns >= 2 and (jet > 0 or framed):
                bands(Wlo, a_lo, a_hi, 0.0, fh, eaves=False)
                bands(Whi, a_lo, a_hi, fh, He, foot=False, cols=cols, rails=[f * fh for f in range(2, ns)] if framed else ())
            else:
                bands(Wof(0), a_lo, a_hi, 0.0, He, cols=cols)
            if nc == 0:
                return 0, span
            wh = min(H["wh"], fh - 1.35)
            dcol = min(nc - 1, int(r.u(0, nc))) if (door and not far) else -1
            if bdoor is not None:
                dcol = min(nc - 1, int(bdoor * nc))
            sh_ = shut and not back and sp >= 2.0 * ww + 0.3
            for f in range(ns):
                W = Wof(f)
                for k in range(nc):
                    ac = a_lo + (k + 0.5) * sp
                    gone = back and (rb_.chance(0.5) if far else r.chance(0.35 if DET < 2 else 0.7))
                    if back and f == 0 and skip and skip[0] - ww / 2 - 0.25 < ac < skip[1] + ww / 2 + 0.25:
                        continue
                    if f == 0 and k == dcol:
                        dh = min(2.15, fh - 0.2)
                        dw = min(1.05, ww + 0.15)
                        reg(H, W, ac, dh / 2, dw, dh, kind="door", back=back)
                        if not back:
                            H["door"] = (ac, dw / 2)
                    elif not gone:
                        reg(H, W, ac, f * fh + 0.95 + wh / 2, ww, wh, back=back, top=(f == ns - 1), fz=f * fh, shut=sh_,
                            grp=("back", i) if back else None)
            return nc, sp

        def spans(a0, a1):
            """Runs of the street front that take a course / rail / plinth: never a stub beside the landmark or the wing."""
            cut = None
            if tw is not None:
                cut = (tw[0] - tw[2] / 2 - 0.13, tw[0] + tw[2] / 2 + 0.13)
            elif wing is not None:
                cut = (wing[0], wing[1])
            if cut is None:
                return [(a0, a1)]
            return [(u, v) for u, v in ((a0, cut[0]), (cut[1], a1)) if v - u > 0.9]

        # ---- front walls
        fr = (0, w)
        if wing is not None:
            xa, xb, yw, Hrw = wing
            Wwg = _Fr((0, yw, 0), (0, -1, 0))
            strip(Wfront, x0, xa, False, False)
            strip(Wfront, xb, x1, False, False)
            strip(lambda f: Wwg, xa, xb, False, True)
            new_uvx()
            for sx, xe in ((-1, xa), (1, xb)):
                K(wk, [(xe, yw, 0), (xe, yf, 0), (xe, yf, He), (xe, yw, He)], (sx, 0, 0), wmask)

            def wing_mask(q, Hrw=Hrw, xc_=0.5 * (xa + xb)):
                g, w_, s_ = wmask(q)
                if q.z > He + 0.05:
                    g = max(0.04, soot * 0.38 * _ss(0.6, 0.0, Hrw - abs(q.x - xc_) * tanp - q.z))
                return (g, w_, s_)
            K(wk, [Wwg.P(xa, He), Wwg.P(xb, He), Wwg.P(0.5 * (xa + xb), Hrw)], Wwg.n, wing_mask)
            if Hrw - He > 2.3:
                reg(H, Wwg, 0.5 * (xa + xb), He + 0.95, 0.7, 1.0, top=True, fz=He, bars=(1, 0))
            roof(rk, "y", yw, yf + (Hrw - He) / tanp + 0.3, xa, xb, He, tanp, 0.28, 0.28, 0.38, 0.0, 0.0, 1, rmask,
                 barge=(True, False), vk=vk_)
        elif tw is not None:
            hb = tw[2] / 2 + 0.13
            strip(Wfront, x0, tw[0] - hb, False, False)
            strip(Wfront, tw[0] + hb, x1, False, False)
            if style == "turret":
                # the round turret leaves a little wall either side of its chord (a square tower hides it all)
                bands(Whi, max(x0, tw[0] - hb), min(x1, tw[0] + hb), 0.0, He, foot=False, eaves=False)
        else:
            fr = strip(Wfront, x0, x1, False, True)

        # ---- back wall + party / end walls
        # the back is lived in too: windows on every floor, and on the near layer here and there a lean-to
        # outshot (scullery / wash-house under a catslide of the house's own covering) or a back door
        lt = None
        bdoor = None
        if not far:
            if w >= 4.4 and rb_.chance(float(p.outshots)):
                lw_ = min(w - 1.3, rb_.u(2.2, 3.3))
                xl0 = x0 + 0.55 + rb_.u(0.0, 1.0) * (w - 1.1 - lw_)
                ld, le = rb_.u(1.25, 1.65), rb_.u(2.05, 2.3)
                lp = tan(radians(rb_.u(24.0, 32.0)))
                lt = (xl0, xl0 + lw_)
                Wlt = _Fr((0, yb + ld, 0), (0, 1, 0))
                new_uvx()
                rect(K, wk, Wlt, -lt[1], -lt[0], 0.0, le, 0.0, wmask)
                for sx, xe in ((-1, lt[0]), (1, lt[1])):
                    K(wk, [(xe, yb, 0.0), (xe, yb + ld, 0.0), (xe, yb + ld, le), (xe, yb, le + ld * lp)], (sx, 0, 0), wmask)
                oe_ = 0.2
                ye, zE = yb + ld + oe_, le - oe_ * lp + 0.04
                K(rk, [(lt[0] - 0.12, ye, zE), (lt[1] + 0.12, ye, zE), (lt[1] + 0.12, yb, le + ld * lp + 0.04),
                       (lt[0] - 0.12, yb, le + ld * lp + 0.04)], (0, lp, 1.0),
                  lambda q, zE=zE: (0.3 + 0.3 * nz(q, 1.4), 0.1, age * (0.3 + 0.7 * nz(q, 1.2))))
                K("trim", [(lt[0] - 0.12, ye, zE - 0.15), (lt[1] + 0.12, ye, zE - 0.15), (lt[1] + 0.12, ye, zE), (lt[0] - 0.12, ye, zE)],
                  (0, 1, 0), trim_mask)
                K("trim", [(lt[0] - 0.12, ye, zE - 0.15), (lt[1] + 0.12, ye, zE - 0.15), (lt[1] + 0.12, yb + ld, zE - 0.15),
                           (lt[0] - 0.12, yb + ld, zE - 0.15)], (0, 0, -1), soffit_mask)
                la = -(lt[0] + lw_ * rb_.u(0.3, 0.7))
                if rb_.chance(0.6):
                    reg(H, Wlt, la, 0.97, 0.84, 1.94, kind="door", back=True)
                else:
                    reg(H, Wlt, la, 1.4, 0.62, 0.74, back=True, fz=0.0, bars=(1, 0), grp=("back", i))
            elif rb_.chance(0.55):
                bdoor = rb_.u(0.0, 1.0)
        strip(lambda f: Wbk, -x1, -x0, True, False, skip=(-lt[1], -lt[0]) if lt else None, bdoor=bdoor)
        for e_, (sx, xe, nb) in enumerate(((-1, x0, i - 1), (1, x1, i + 1))):
            run_end = not (0 <= nb < n)
            # a run end is an exposed wall: it stands EI inside the extent, so its corner piers, chimney breast,
            # sills and courses have real projection and still stop exactly at +-length/2
            Ws = _Fr((xe - (sx * EI if run_end else 0.0), 0, 0), (sx, 0, 0))

            def ar(ya, yb_, sx=sx):
                return (ya, yb_) if sx > 0 else (-yb_, -ya)
            new_uvx()
            if jet > 0:
                al, ah = ar(yf, yb)
                rect(K, wk, Ws, al, ah, 0.0, fh, 0.0, wmask)
                al, ah = ar(yfu, yb)
                rect(K, wk, Ws, al, ah, fh, He, 0.0, wmask)
            else:
                al, ah = ar(yf, yb)
                rect(K, wk, Ws, al, ah, 0.0, He, 0.0, wmask)
            br = ebr.get(e_) if run_end else None
            if typ != "front":
                al, ah = ar(yfu, yb)
                zt_ = zmid if hip[e_] else Hr
                if hip[e_]:
                    dd = (zt_ - He) / tanp
                    K(wk, [Ws.P(al, He), Ws.P(ah, He), Ws.P(ah - dd, zt_), Ws.P(al + dd, zt_)], Ws.n, wmask)
                else:
                    K(wk, [Ws.P(al, He), Ws.P(ah, He), Ws.P(0.5 * (al + ah), Hr)], Ws.n, wmask)
                exposed = run_end or houses[nb]["Hr0"] < He + 0.3 * (Hr - He)
                if exposed and zt_ - He > 2.5 and not (tw and abs(xe - tw[0]) < tw[2]):
                    ga = 0.5 * (al + ah)
                    if br:
                        # the attic window stands beside the breast, as far over as the gable allows
                        room = (zt_ - (He + 1.55)) / tanp - 0.62 if not hip[e_] else 0.5
                        ga += (-1.0 if br[0] > ga else 1.0) * max(0.0, min(0.6, room))
                    if not br or abs(ga - br[0]) > br[2] / 2 + 0.35 + 0.16:
                        reg(H, Ws, ga, He + 1.0, 0.7, 1.0, top=True, fz=He, bars=(1, 0), grp=("end", i, e_) if run_end else None)
            if not run_end:
                continue
            # ---- exposed run end: corner piers, chimney breast, windows on every floor, courses / rails
            rwe = r.fork(f"endwin{e_}")
            ck_ = "trim" if timber else H["pk"]
            cmk = trim_mask if timber else wmask
            pwd = 0.2 if timber else 0.34
            for ya, yb_, za, zb in ([(yf, yb, 0.0, fh), (yfu, yb, fh, He)] if jet > 0 else [(yf, yb, 0.0, He)]):
                al, ah = ar(ya, yb_)
                pbox(K, ck_, Ws, al, al + pwd, za, zb, 0.0, EI, "fr", cmk)
                pbox(K, ck_, Ws, ah - pwd, ah, za, zb, 0.0, EI, "fl", cmk)
            al, ah = ar(yfu, yb)
            if br:
                a_c, wl, wu, zsh = br

                def bm(q):
                    return (0.22 + 0.2 * nz(q, 0.7) + 0.3 * _ss(1.2, 0.0, q.z), 0.2, 0.3 * age * nz(q, 0.9))
                pbox(K, "brick", Ws, a_c - wl / 2, a_c + wl / 2, 0.0, zsh, 0.0, EI, "flr", bm)
                K("brick", [Ws.P(a_c - wl / 2, zsh, EI), Ws.P(a_c + wl / 2, zsh, EI), Ws.P(a_c + wu / 2, zsh + 0.6, EI),
                            Ws.P(a_c - wu / 2, zsh + 0.6, EI)], Ws.n, bm)
                for s_ in (-1, 1):
                    K("brick", [Ws.P(a_c + s_ * wl / 2, zsh, 0.0), Ws.P(a_c + s_ * wl / 2, zsh, EI), Ws.P(a_c + s_ * wu / 2, zsh + 0.6, EI),
                                Ws.P(a_c + s_ * wu / 2, zsh + 0.6, 0.0)], Ws.rt * s_ + UP, bm)
            wwe, whe = min(H["ww"], 0.9), min(H["wh"], fh - 1.35)
            ecols = [al + f_ * (ah - al) for f_ in (0.2, 0.5, 0.8)]
            if br:
                ecols = [c for c in ecols if abs(c - br[0]) > br[1] / 2 + wwe / 2 + 0.42]
            for f in range(ns):
                pick = [c for c in ecols if rwe.chance(0.72)]
                if not pick and ecols:
                    pick = [rwe.pick(ecols)]
                for c in pick:
                    reg(H, Ws, c, f * fh + 0.95 + whe / 2, wwe, whe, top=(f == ns - 1), fz=f * fh, rec=0.0, grp=("end", i, e_))
            if wk == "brick":
                for f in range(1, ns):
                    rect(K, "plA", Ws, al, ah, f * fh - 0.08, f * fh + 0.08, 0.03, wmask)
                if typ != "front":
                    rect(K, "plA", Ws, al, ah, He - 0.1, He + 0.08, 0.03, wmask)
            elif timber:
                for f in range(1, ns):
                    rect(K, "trim", Ws, al, ah, f * fh - 0.1, f * fh + 0.1, TD, trim_mask)
                if typ != "front":
                    rect(K, "trim", Ws, al, ah, He - 0.1, He + 0.1, TD, trim_mask)              # tie beam
                    zc_ = He + 0.58 * (Hr - He)
                    hwc = (Hr - zc_) / tanp - 0.14
                    if not hip[e_] and zc_ > He + 1.75 and hwc > 0.5:
                        rect(K, "trim", Ws, 0.5 * (al + ah) - hwc, 0.5 * (al + ah) + hwc, zc_ - 0.1, zc_ + 0.1, TD, trim_mask)   # collar
            elif ns >= 2:
                rect(K, H["pk"], Ws, al, ah, fh - 0.08, fh + 0.08, 0.03, wmask)
            if typ == "front":
                # the eaves of a gable-fronted end house: a board and soffit closing the roof edge over the inset wall
                vm_ = trim_mask if vk_ == "trim" else (0.2, 0.1, 0.1 * age)
                rect(K, vk_, Ws, al, ah, He - 0.2, He + 0.03, EI - 0.02, vm_)
                K("trim", [Ws.P(al, He - 0.2, 0.0), Ws.P(ah, He - 0.2, 0.0), Ws.P(ah, He - 0.2, EI - 0.02), Ws.P(al, He - 0.2, EI - 0.02)],
                  (0, 0, -1), soffit_mask)

        # ---- roof
        def verge(nb):
            if not (0 <= nb < n):
                return 0.0                      # clean, flush run ends
            return 0.26 if (houses[nb]["He"] < He - 0.25 and not far) else 0.07
        sag = (0.05 + 0.2 * age) * r.u(0.75, 1.25) * min(1.0, 0.5 + (x1 - x0) / 12.0)
        nseg = 3 if (DET == 0 and w > 7.0) else 2
        a_top = 0.0
        ztop = Hr
        gwin = []
        e0, e1 = (EI if i == 0 else 0.0), (EI if i == n - 1 else 0.0)

        def sag_at(x, f_):
            """How far the laid roof lies below the nominal slope at x, f_ of the way up (the ridge sag, bay by bay)."""
            t = min(1.0, max(0.0, (x - x0) / max(w, 0.1))) * nseg
            k_ = min(nseg - 1, int(t))
            return sag * f_ * (sin(pi * k_ / nseg) * (k_ + 1 - t) + sin(pi * (k_ + 1) / nseg) * (t - k_))

        def rooflight(pc, ac, uh, rr):
            """Cast-iron rooflight lying on a slope: a stout frame and its pane (dark, or the glow of an attic room).
            pc = middle of its lower edge on the slope, ac = unit vector along the eaves, uh = horizontal unit vector up the slope."""
            cp = 1.0 / sqrt(1.0 + tanp * tanp)
            us, nn = (uh + UP * tanp) * cp, (UP - uh * tanp) * cp
            lw_, ll_, m_ = (0.8, 1.1, 0.12) if not far else (1.05, 1.4, 0.15)

            def q(a, b, h):
                return pc + ac * a + us * b + nn * h
            K("trim", [q(-lw_ / 2, 0.0, 0.07), q(lw_ / 2, 0.0, 0.07), q(lw_ / 2, ll_, 0.07), q(-lw_ / 2, ll_, 0.07)], nn, trim_mask)
            lit = float(p.lit_ratio) > 0.01 and rr.chance(min(0.6, 0.9 * float(p.lit_ratio)))
            K("dim" if lit else "dark", [q(-lw_ / 2 + m_, m_, 0.085), q(lw_ / 2 - m_, m_, 0.085), q(lw_ / 2 - m_, ll_ - m_, 0.085),
                                         q(-lw_ / 2 + m_, ll_ - m_, 0.085)], nn, flat_mask if lit else (0.35, 0.0, 0.0),
              uvs=flat_uv(4) if lit else None)

        rr_ = r.fork("rooflights")
        if typ in ("side", "cross"):
            roof(rk, "x", x0 + e0, x1 - e1, yfu, yb, He, tanp, 0.42, 0.38, (EI - RI) if e0 else verge(i - 1),
                 (EI - RI) if e1 else verge(i + 1), sag, nseg, rmask, hip=tuple(hip), hf=hf, r=r, vk=vk_)
        else:
            # gable wall(s) with attic windows
            ztop = zmid if hip[0] else Hr
            if Hr - He > 2.3:
                ng = 2 if (w > 6.2 and Hr - He > 3.2) else 1
                for k in range(ng):
                    gx = xm + (0.0 if ng == 1 else (k - 0.5) * 1.7)
                    bb_ = He + 1.0
                    if (Hr - (bb_ + 0.55)) / tanp > abs(gx - xm) + 0.36 + 0.3 and bb_ + 0.55 < ztop - 0.25:
                        reg(H, Whi, gx, bb_, 0.72, 1.1, top=True, fz=He, bars=(1, 0))
                        gwin.append(gx)
                        a_top = bb_ + 0.55
            new_uvx()
            if stepped:
                nstep = max(3, int(round(w / 2 / 0.85)))
                N = 2 * nstep + 1
                cw = w / N
                tops = [He + min((s + 1) * cw, w / 2) * tanp + 0.28 for s in range(nstep + 1)]
                Wst = _Fr((0, yf + 0.32, 0), (0, 1, 0))
                for s in range(nstep + 1):
                    zl = He if s == 0 else tops[s - 1]
                    xa_, xb_ = x0 + s * cw, x1 - s * cw
                    rect(K, wk, Whi, xa_, xb_, zl, tops[s], 0.0, wmask)
                    rect(K, wk, Wst, -xb_, -xa_, zl, tops[s], 0.0, wmask)
                    # pale stone coping on every step, and the step's riser
                    for sx, xo in ((-1, xa_), (1, xb_)):
                        xi = xo - sx * cw if s < nstep else xb_
                        cl, ch = max(-L / 2, min(xo, xi) - 0.03), min(L / 2, max(xo, xi) + 0.03)
                        boxf(K, "plA", (cl, yf - 0.04, tops[s]), (ch, yf + 0.36, tops[s] + 0.07), "Z", (0.3, 0.2, 0.2 * age))
                        boxf(D, "plA", (cl, yf - 0.04, tops[s]), (ch, yf + 0.36, tops[s] + 0.07), "y" + ("x" if sx < 0 else "X"),
                             (0.3, 0.2, 0.2 * age))
                        K(wk, [(xo, yf, zl), (xo, yf + 0.32, zl), (xo, yf + 0.32, tops[s]), (xo, yf, tops[s])], (sx, 0, 0), wmask)
                        if s == nstep:
                            break
            else:
                if hip[0]:
                    dd = (ztop - He) / tanp
                    K(wk, [Whi.P(x0, He), Whi.P(x1, He), Whi.P(x1 - dd, ztop), Whi.P(x0 + dd, ztop)], Whi.n, wmask)
                else:
                    K(wk, [Whi.P(x0, He), Whi.P(x1, He), Whi.P(xm, Hr)], Whi.n, wmask)
            K(wk, [Wbk.P(-x1, He), Wbk.P(-x0, He), Wbk.P(-xm, Hr)], Wbk.n, wmask)
            if Hr - He > 2.6:
                reg(H, Wbk, -xm, He + 0.9, 0.7, 1.0, back=True, top=True, fz=He, bars=(1, 0), grp=("back", i))
            roof(rk, "y", yfu + (0.3 if stepped else 0.0), yb, x0, x1, He, tanp, 0.0, 0.0, 0.0 if stepped else 0.42, 0.3,
                 sag * 0.7, 2, rmask, hip=(hip[0], False), hf=hf, r=r, barge=(not stepped, True), vk=vk_)
            # rooflights on the slopes that stand clear of the neighbours (a gable-fronted house shows its long
            # slopes to the side): one or two per exposed slope, between the stacks
            for sx, nb in ((-1, i - 1), (1, i + 1)):
                zn = houses[nb]["Hr0"] if 0 <= nb < n else 0.0
                f_ = max(0.22, (zn + 0.35 - He) / max(Hr - He, 0.1))
                if f_ > 0.5 or Hr - He < 2.4:
                    continue
                for yc in (ym + rr_.u(-0.3, -0.12) * (yb - yfu), ym + rr_.u(0.1, 0.3) * (yb - yfu)):
                    f2 = min(0.5, f_ + rr_.u(0.0, 0.14))
                    xc = xm + sx * (1.0 - f2) * w / 2
                    if rr_.chance(0.6) and all(math.hypot(xc - c[0], yc - c[1]) > 0.5 * max(c[2], c[3]) + 1.0 for c in chims):
                        rooflight(Vector((xc, yc, He + f2 * (Hr - He) + 0.012 - sag * 0.7 * f2 * sin(pi * (yc - yfu) / (yb - yfu)))),
                                  Vector((0.0, 1.0, 0.0)), Vector((-sx, 0.0, 0.0)), rr_)

        # ---- dressing: timber frame on limewashed houses, pale stone plinth / courses on brick
        if wk != "brick":
            for f in range(1, ns):
                if jet > 0 and f == 1:
                    continue
                if timber or f == 1:
                    for sa, sb in spans(x0 + 0.02, x1 - 0.02):
                        rect(K, "trim", Wfront(f), sa, sb, f * fh - TW, f * fh + TW, TD, trim_mask)
            nc, sp = fr
            if jet > 0:
                # bressummer carrying the jettied storeys on knee brackets; its underside is the dark jetty soffit
                pbox(K, "trim", Wlo, x0, x1, fh - 0.24, fh + 0.07, 0.0, jet + 0.045, "f", trim_mask)
                pbox(DK, "trim", Wlo, x0, x1, fh - 0.24, fh + 0.07, 0.0, jet + 0.045, "lr", trim_mask)
                pbox(K, "trim", Wlo, x0, x1, fh - 0.24, fh + 0.07, 0.0, jet + 0.045, "b", soffit_mask)
                zk = fh - 0.24
                if DET <= 1:
                    for k in range(max(1, nc) + 1):
                        bx = min(x1 - 0.1, max(x0 + 0.1, x0 + k * (w / max(1, nc))))
                        for sx in (-1, 1):
                            D("trim", [(bx + sx * 0.06, yf, zk), (bx + sx * 0.06, yf - jet, zk), (bx + sx * 0.06, yf, zk - 0.5)],
                              (sx, 0, 0), trim_mask)
                        D("trim", [(bx - 0.06, yf - jet, zk), (bx + 0.06, yf - jet, zk), (bx + 0.06, yf, zk - 0.5),
                                   (bx - 0.06, yf, zk - 0.5)], (0, -1, -0.6), trim_mask)
            if timber and wing is None and tw is None and nc:
                zb_, zt_ = (fh + 0.1 if ns >= 2 else 0.0), He - 0.03
                for k in range(nc + 1):
                    sx = min(x1 - TW - 0.01, max(x0 + TW + 0.01, x0 + k * sp))
                    rect(K if (DET < 3 or k in (0, nc)) else DK, "trim", Whi, sx - TW + 0.01, sx + TW - 0.01, zb_, zt_, TD, trim_mask)
                rect(K, "trim", Whi, x0 + 0.02, x1 - 0.02, He - 0.2, He - 0.02, TD, trim_mask)
                zs_ = (ns - 1) * fh + 0.95 - 0.12                     # sill rail between the posts
                rect(DK, "trim", Whi, x0 + 0.02, x1 - 0.02, zs_ - 0.12, zs_, TD, trim_mask)
                bl = min(0.75, sp * 0.3)
                zf_ = (ns - 1) * fh + 0.1
                beam("trim", Whi, (x0 + 0.2, zs_ - 0.12), (x0 + 0.2 + bl, zf_), 0.13, TD, trim_mask)
                beam("trim", Whi, (x1 - 0.2, zs_ - 0.12), (x1 - 0.2 - bl, zf_), 0.13, TD, trim_mask)
            if timber and typ == "front":
                rect(K, "trim", Whi, x0 + 0.02, x1 - 0.02, He - 0.1, He + 0.1, TD, trim_mask)
                zlim = ztop - 0.04
                # collar beam above the attic window head with a king post up to the apex, studs below
                zc_ = max(a_top + 0.3, He + 0.55 * (Hr - He))
                hwc = (Hr - zc_) / tanp - 0.12
                has_collar = hwc > 0.45 and zc_ + 0.1 < zlim
                if has_collar:
                    rect(K, "trim", Whi, xm - hwc, xm + hwc, zc_ - 0.09, zc_ + 0.09, TD, trim_mask)
                    if min(zlim, Hr - 0.25) - zc_ > 0.5:
                        rect(DK, "trim", Whi, xm - 0.08, xm + 0.08, zc_ + 0.09, min(zlim, Hr - 0.25), TD, trim_mask)
                k = 0
                while k * 1.05 < w / 2 - 0.5:
                    for sx in ((-1, 1) if k else (1,)):
                        gx = xm + sx * k * 1.05
                        zt_ = min(zlim, He + (w / 2 - k * 1.05) * tanp - 0.12)
                        if has_collar and abs(gx - xm) < hwc:
                            zt_ = min(zt_, zc_ - 0.09)
                        if any(abs(gx - g_) < 0.36 + 0.2 for g_ in gwin):
                            continue
                        if zt_ > He + 0.3:
                            rect(DK, "trim", Whi, gx - 0.08, gx + 0.08, He + 0.1, zt_, TD, trim_mask)
                    k += 1
        else:
            # brick houses: pale stone plinth (broken at the door), string courses at the floor lines, eaves band
            dr = H.get("door")
            new_uvx()
            for sa, sb in spans(x0 + 0.02, x1 - 0.02):
                segs = [(sa, sb)] if (dr is None or wing is not None) else \
                    [(sa, min(sb, dr[0] - dr[1] - 0.14)), (max(sa, dr[0] + dr[1] + 0.14), sb)]
                for ua_, ub_ in segs:
                    if ub_ - ua_ > 0.2:
                        pbox(D, "plA", Wlo, ua_, ub_, 0.0, 0.42, 0.0, 0.05, "ft", wmask)
                for f in range(1, ns):
                    rect(K, "plA", Wlo, sa, sb, f * fh - 0.08, f * fh + 0.08, 0.03, wmask)
                if typ != "front":
                    rect(K, "plA", Wlo, sa, sb, He - 0.2, He - 0.02, 0.03, wmask)

        # ---- dormers: gabled or shed, dying into the main slope
        def dormer(xd, wd, s, q0, hd, pd):
            c0 = yfu if s < 0 else yb
            cf = c0 - s * q0
            zb = top(xd, cf)
            ze = zb + hd
            zr = ze + (0.0 if shed else wd / 2 * pd)
            if zr + 0.12 > Hr - 0.4:
                return None
            for e_, xe_ in ((0, x0), (1, x1)):
                if hip[e_] and zr > zmid - 0.15 and abs(xd - xe_) < e_hip + wd / 2 + 0.35:
                    return None
            osd, of = 0.11, 0.16

            def c_at(z):
                return ym + s * (Hr - SINK - z) / tanp
            # keep clear of every stack whose footprint the dormer (front face to where its roof dies) would meet
            ytop = cf - s * (hd + SINK + 0.1) / (tanp - 0.27) if shed else c_at(zr)
            for c in chims:
                if abs(xd - c[0]) < (c[2] + wd) / 2 + 0.3 and s * (c[1] - ytop) + c[3] / 2 + 0.2 > 0.0 \
                        and s * (c[1] - cf) - c[3] / 2 - 0.2 < 0.0:
                    return None
            Wd = _Fr((0, cf, 0), (0, s, 0))
            ac = -s * xd
            cfo = cf + s * of
            if shed:
                # flat-ish lean-to roof rising gently back into the main slope
                t15 = 0.27
                dy = (hd + SINK + 0.1) / (tanp - t15)
                K(dk, [Wd.P(ac - wd / 2, zb - SINK), Wd.P(ac + wd / 2, zb - SINK), Wd.P(ac + wd / 2, ze), Wd.P(ac - wd / 2, ze)], Wd.n, dmask)
                yk = cf - s * dy
                zk_ = ze + dy * t15
                for sx in (-1, 1):
                    xe = xd + sx * wd / 2
                    K(dk, [(xe, cf, zb - SINK), (xe, cf, ze), (xe, yk, zk_)], (sx, 0, 0), dmask)
                xo0, xo1 = xd - wd / 2 - osd, xd + wd / 2 + osd
                K(rk, [(xo0, cfo, ze - of * t15 + 0.03), (xo1, cfo, ze - of * t15 + 0.03), (xo1, yk, zk_ + 0.03), (xo0, yk, zk_ + 0.03)],
                  (0, s * t15, 1.0), rmask)
                if True:
                    K("trim", [(xo0, cfo, ze - of * t15 - 0.09), (xo1, cfo, ze - of * t15 - 0.09), (xo1, cfo, ze - of * t15 + 0.03),
                               (xo0, cfo, ze - of * t15 + 0.03)], (0, s, 0), trim_mask)
                    DK("trim", [(xo0, cfo, ze - of * t15 - 0.02), (xo1, cfo, ze - of * t15 - 0.02), (xo1, yk, zk_ - 0.02), (xo0, yk, zk_ - 0.02)],
                      (0, -s * t15, -1.0), soffit_mask)                 # underside of the overhang
            else:
                K(dk, [Wd.P(ac - wd / 2, zb - SINK), Wd.P(ac + wd / 2, zb - SINK), Wd.P(ac + wd / 2, ze), Wd.P(ac, zr),
                       Wd.P(ac - wd / 2, ze)], Wd.n, dmask)
                for sx in (-1, 1):
                    xe = xd + sx * wd / 2
                    K(dk, [(xe, cf, zb - SINK), (xe, cf, ze), (xe, c_at(ze), ze)], (sx, 0, 0), dmask)   # triangular cheek
                    xo = xd + sx * (wd / 2 + osd)
                    zE = ze - osd * pd
                    K(rk, [(xo, cfo, zE), (xd, cfo, zr), (xd, c_at(zr), zr), (xo, c_at(zE), zE)], (sx * pd, 0, 1.0), rmask)
                    if True:
                        K("trim", [(xo, cfo, zE - 0.1), (xd, cfo, zr - 0.1), (xd, cfo, zr), (xo, cfo, zE)], (0, s, 0), trim_mask)
                        DK("trim", [(xo, cfo, zE - 0.03), (xd, cfo, zr - 0.03), (xd, c_at(zr), zr - 0.03), (xo, c_at(zE), zE - 0.03)],
                          (-sx * pd, 0, -1.0), soffit_mask)             # underside of the overhang
            wd_ = reg(H, Wd, ac, zb + 0.2 + (hd - 0.32) / 2, wd - 0.42, hd - 0.32, back=s > 0, top=True, fz=zb - 0.8,
                      sill=False, bars=(1, 0), fk=dfk, sk=dfk, ec=0.1 if far else 0.075, eo=0.03, brick=False, dormer=True, wkey=dk,
                      rec=0.045, rm=0.1, grp=("back", i) if s > 0 else None)
            if s < 0:
                dorm_lights.append((xd, (xd, cf - 0.5, wd_["b"]), wd_))
            return zr

        dk = wk if wk != "brick" else "plA"
        dfk = "trim" if not far else ("ash" if dk == "plA" else "plA")      # dormer casings: oak near, pale through the haze
        dxs = []
        if typ in ("side", "cross") and Hr - He > 2.5:
            rd_ = r.fork("dormers")
            wd = rd_.u(1.15, 1.45) * (1.25 if far else 1.0)
            pd = tan(radians(rd_.u(44, 52)))
            q0 = rd_.u(0.4, 0.7)
            hd = rd_.u(1.1, 1.25) * (1.2 if far else 1.0)
            # the street slope is furnished along every free run of the eaves (clear of hips, landmark and wing)
            runs = [(x0 + 0.3 + (e_hip if hip[0] else 0.0), x1 - 0.3 - (e_hip if hip[1] else 0.0))]
            blocks = [(c[0] - c[2] / 2 - 0.25, c[0] + c[2] / 2 + 0.25) for c in chims if c[1] < ym - 0.3]
            if tw:
                blocks.append((tw[0] - tw[2] / 2 - 0.45, tw[0] + tw[2] / 2 + 0.45))
            if wing:
                blocks.append((wing[0] - 0.6, wing[1] + 0.6))
            for b0_, b1_ in blocks:
                runs = [s_ for u, v in runs for s_ in ((u, min(v, b0_)), (max(u, b1_), v))]
            for u, v in runs:
                if v - u < wd + 0.5:
                    continue
                kd = max(1, int((v - u + 0.8) / (wd + 1.15)))
                for j in range(kd):
                    xd = u + (j + 0.5) * (v - u) / kd + rd_.u(-0.12, 0.12)
                    xd = min(v - wd / 2 - 0.2, max(u + wd / 2 + 0.2, xd))
                    if rd_.chance(0.65 if far else 0.92) and dormer(xd, wd, -1, q0, hd, pd) is not None:
                        dxs.append((-1, xd, wd))
            if rd_.chance(0.9) and (not far or r.fork("fardormer").chance(0.6)):
                # one dormer on the back slope, at the first of a few places clear of the landmark and the stacks
                f0 = rd_.u(0.3, 0.7)
                for fx in (f0, 1.0 - f0, 0.5, 0.26, 0.74):
                    xd = x0 + w * fx
                    if ((not tw) or abs(xd - tw[0]) > tw[2] / 2 + wd) and dormer(xd, wd, 1, q0, hd, pd) is not None:
                        dxs.append((1, xd, wd))
                        break
        if typ in ("side", "cross") and Hr - He > 2.1:
            # rooflights: on a street slope that has no dormer, and beside / instead of the dormer on the back slope
            for s in (1, -1):
                have = [d_ for d_ in dxs if d_[0] == s]
                if s < 0 and have:
                    continue
                for fx in (rr_.u(0.2, 0.32), rr_.u(0.66, 0.8)):
                    xc = x0 + w * fx
                    f_ = rr_.u(0.3, 0.45)
                    yc = (yfu if s < 0 else yb) - s * f_ * half
                    ok = rr_.chance(0.5 if have else 0.8)
                    ok = ok and all(abs(xc - d_[1]) > d_[2] / 2 + 1.0 for d_ in have)
                    ok = ok and all(math.hypot(xc - c[0], yc - c[1]) > 0.5 * max(c[2], c[3]) + 1.0 for c in chims)
                    ok = ok and not (tw and abs(xc - tw[0]) < tw[2] / 2 + 1.0)
                    ok = ok and not (s < 0 and wing and wing[0] - 1.0 < xc < wing[1] + 1.0)
                    ok = ok and not any(hip[e_] and abs(xc - xe_) < e_hip + 1.0 for e_, xe_ in ((0, x0), (1, x1)))
                    if ok:
                        rooflight(Vector((xc, yc, He + f_ * (Hr - He) + 0.012 - sag_at(xc, f_))), Vector((1.0, 0.0, 0.0)),
                                  Vector((0.0, -s, 0.0)), rr_)

        # ---- chimneys
        for cx, cy, cw_, cd_, zt, zb_, npots, fl_ in chims:
            def cm_(q, zt=zt):
                return (0.3 + 0.2 * nz(q, 0.7) + 0.5 * _ss(zt - 1.2, zt, q.z), 0.2, 0.3 * age * nz(q, 0.9))
            chimney(cx, cy, zb_, zt, cw_, cd_, npots, rc, cm_, fl=fl_)

    # ---------------------------------------------------------------- landmark
    rt_ = rng.fork("landmark")
    hints = [(0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]
    lm = (0.5, 0.3, 0.0)
    clock_light = None

    def finial(tcx, tcy, fz, vane=True):
        """Ball, rod and a weather vane / pennant (lead-grey); all detail."""
        rb = 0.17
        cb = Vector((tcx, tcy, fz + 0.45))
        prism(DK, "slate", tcx, tcy, fz + 0.45, fz + 0.45 + rb, rb, 0.0, 4, lm, a0=pi / 4)
        prism(DK, "slate", tcx, tcy, fz + 0.45, fz + 0.45 - rb, rb, 0.0, 4, lm, a0=pi / 4)
        boxf(DK, "slate", (tcx - 0.03, tcy - 0.03, fz), (tcx + 0.03, tcy + 0.03, fz + 1.75), "xXyY", lm)
        if not vane or far:
            return
        va = radians(rt_.u(-50, 50))
        dv = Vector((cos(va), sin(va), 0))
        nv_ = Vector((-sin(va), cos(va), 0))
        base = cb + UP * 1.0
        vq = [base + dv * 0.05 - UP * 0.16, base + dv * 0.75 - UP * 0.04, base + dv * 0.75 + UP * 0.04, base + dv * 0.05 + UP * 0.16]
        D("slate", [q + nv_ * 0.008 for q in vq], nv_, lm)
        D("slate", [q - nv_ * 0.008 for q in vq], -nv_, lm)

    # Landmark shafts are built in stages, never more than 3.8 m of plain wall: a battered stone plinth, a
    # projecting string course at every stage (weathered top, undercut bed), pale corner lesenes, a lancet or
    # stair slit in every stage of every face that stands free, a belfry stage / eaves ring and a corbel course
    # under the cap. All of it is distance tier, all members 0.14 m or stouter.
    lw_ = p.landmark_wall
    if lw_ == "auto":
        lw_ = rng.fork("lmwall").weighted([("ashlar", 3), ("brick", 4), ("render", 3)])
    hwk_ = host["wk"] if host is not None else "plA"
    tk_ = {"ashlar": "ash", "brick": "brick", "render": "plA" if hwk_ == "plB" else "plB"}[lw_]    # shaft walling
    dk_ = "ash"                                                                                     # stone dressings
    sfk = dk_ if tk_ != dk_ else "plA"       # window surrounds: never the stuff of the wall they sit in
    LM = ("lm", 0)
    fs = 1.3 if far else 1.0                 # the far ring's members are stouter still
    PJ, UPJ, DNJ = 0.16 * fs, 0.22 * fs, 0.14 * fs
    cm_top = (0.05, 0.55, 0.3 * age)         # weathered top of a course: rain-washed, pale
    cm_und = (0.9, 0.0, 0.0)                 # its undercut bed: always in its own shadow

    def stages(za, zb, hmax_=3.8):
        m = max(1, int(math.ceil((zb - za) / hmax_ - 1e-6)))
        return [za + (zb - za) * k / m for k in range(m + 1)]

    def lancet(W, a, zc, w, h, **kw):
        """Pointed light laid on the wall in a stone surround (lit by the same draw as every other window)."""
        reg(host, W, a, zc, w, h, fz=zc - h / 2 - 0.9, bars=(0, 0), fk=sfk, sk=sfk, brick=False, eo=0.08,
            ec=0.13 if w > 0.4 else 0.11, sill=False, wkey=tk_, rec=0.0, arch=True, grp=LM, **kw)

    def lancets(W, a, zz, w, zlo=-1e9, zhi=1e9, first=0.0):
        """One light in every stage of zz, in the free field between its string courses."""
        for j, (za, zb) in enumerate(zip(zz[:-1], zz[1:])):
            lo, hi = max(za + UPJ + 0.3 + (first if j == 0 else 0.0), zlo), min(zb - DNJ - 0.3, zhi)
            h = min(1.5, hi - lo - 0.26)
            if h >= 0.8:
                lancet(W, a, 0.5 * (lo + hi), w, h)

    if host is not None and style == "turret":
        tcx, tcy, wt = host["tower"]
        R = wt / 2
        Ht = host["Hr"] + float(p.landmark_height) * (0.55 if not far else 1.2)
        st8 = pi / 4
        ri = R * cos(pi / 8)
        fw_ = 2 * R * sin(pi / 8)
        zpl = 1.1                               # top of the battered plinth
        zcb = Ht - 0.5                          # foot of the corbel course under the eaves
        ZT = stages(zpl, zcb)
        levels = ZT[1:-1] + [zcb]
        zrf = host["top"](tcx, tcy + 0.7 * ri)  # host roof against the faces that stand inside the house
        STREET = (3, 4, 5, 6, 7)                # faces 4, 5, 6 look at the street, 3 and 7 along it

        def tmask(q):
            g = 0.05 + 0.1 * nz(q, 2.0) + 0.3 * _ss(Ht - 1.2, Ht, q.z)
            g = max(g, 0.6 * _ss(1.6, 0.0, q.z))
            for zc_ in levels:
                if q.z < zc_ + 0.01:
                    g = max(g, 0.5 * _ss(1.6, 0.0, zc_ - q.z) * (0.6 + 0.4 * nz(q, 0.6)))    # dirt washed down under every course
            return (min(1.0, g), 0.2 * nz(q, 0.8), age * 0.7 * nz(q, 1.6) * _ss(Ht, 0.0, q.z))

        def pmask(q):
            return (min(1.0, 0.4 + 0.45 * _ss(zpl, 0.0, q.z) + 0.15 * nz(q, 0.7)), 0.25, age * 0.9 * nz(q, 1.1))

        def sband(dg):
            return lambda q: (min(1.0, max(0.0, 0.25 + 0.4 * nz(q, 1.5) + dg)), 0.2, age * 0.8 * nz(q, 1.2) * (1.0 if q.y > tcy else 0.5))

        def VX(rad, k, z):
            return Vector((tcx + rad * cos(pi / 8 + k * st8), tcy + rad * sin(pi / 8 + k * st8), z))

        def fn(k):
            return Vector((cos((k + 1) * st8), sin((k + 1) * st8), 0.0))

        def band(key, k, r0, z0, r1, z1, mask, tilt=0.0):
            K(key, [VX(r0, k, z0), VX(r0, k + 1, z0), VX(r1, k + 1, z1), VX(r1, k, z1)], fn(k) + UP * tilt, mask)

        def Wk(k):
            return _Fr(Vector((tcx, tcy, 0.0)) + fn(k) * ri, fn(k))

        new_uvx()
        if not far:
            band(tk_, 5, R, 0.0, R, zpl + 0.03, tmask)         # the door's face has no plinth
        for za, zb in zip(ZT[:-1], ZT[1:]):
            for k in (range(8) if zb > zrf else STREET):
                band(tk_, k, R, za + 0.03, R, zb, tmask)
        pj = 0.22 * fs
        for k in STREET:
            if k == 5 and not far:
                for v_, w_ in ((5, 6), (6, 5)):
                    K(dk_, [VX(R, v_, 0.0), VX(R + pj, v_, 0.0), VX(R, v_, zpl)], VX(R, w_, 0.0) - VX(R, v_, 0.0), pmask)
                continue
            band(dk_, k, R + pj, 0.0, R, zpl, pmask, 0.2)
        for z in ZT[1:-1]:
            for k in (range(8) if z - DNJ > zrf else STREET):
                band(dk_, k, R + PJ, z, R, z + UPJ, cm_top, 1.0)
                band(dk_, k, R, z - DNJ, R + PJ, z, cm_und, -1.0)
        for k in range(8):
            band(dk_, k, R, zcb, R + 0.34, Ht - 0.1, (0.75, 0.0, 0.0), -1.0)       # corbel course carrying the eaves
        # stair slits winding up the street faces, a ring of small lights under the eaves all the way round
        sw0 = 0.34 * fs
        for j, (za, zb) in enumerate(zip(ZT[:-1], ZT[1:])):
            for k, f_ in ((5, 0.7), (4 if j % 2 == 0 else 6, 0.3)):
                lo, hi = za + UPJ + 0.25, zb - DNJ - 0.25
                if j == len(ZT) - 2:
                    hi = Ht - 1.95
                if k == 5 and j == 0 and not far:
                    lo = max(lo, 2.45)
                h = min(1.2, hi - lo - 0.24)
                if h >= 0.75:
                    lancet(Wk(k), 0.0, lo + 0.12 + h / 2 + f_ * (hi - lo - 0.24 - h), sw0, h)
        for k in range(8):
            reg(host, Wk(k), 0.0, Ht - 1.25, min(0.62, fw_ - 0.42), 0.62, fz=Ht - 2.6, top=True, bars=(1, 0), fk=sfk, sk=sfk,
                brick=False, eo=0.06, ec=0.1, sill=False, small=True, wkey=tk_, rec=0.0, grp=LM)
        if not far:
            reg(host, Wk(5), 0.0, 1.0, 0.78, 2.0, kind="door", fk="ash", sk="ash", brick=False, eo=0.08, wkey=tk_, rec=0.1, rm=0.08)
        # conical roof in two courses of its own tone, with a flared (bell-cast) foot
        hc = 2.5 * R
        zm_, rm_ = Ht + 0.12 + 0.42 * (hc - 0.12), (R + 0.16) * 0.58
        tl_ = rt_.u(-0.1, 0.1)
        for k in range(8):
            band("slate", k, R + 0.36, Ht - 0.1, R + 0.16, Ht + 0.12, sband(0.3), 0.5)
            band("slate", k, R + 0.16, Ht + 0.12, rm_, zm_, sband(tl_), 0.5)
            K("slate", [VX(rm_, k, zm_), VX(rm_, k + 1, zm_), Vector((tcx, tcy, Ht + hc))], fn(k) + UP * 0.3, sband(-tl_ - 0.08))
        finial(tcx, tcy, Ht + hc - 0.3)
    elif host is not None:
        tcx, tcy, wt = host["tower"]
        hh = wt / 2
        hb = hh + 0.13
        Ht = ridge_max + float(p.landmark_height) * (1.0 if style == "spire" else 0.8) * (1.5 if far else 1.0)
        zs1 = host["He"] + 0.3                       # top of the base stage = host eaves line
        zbel = max(Ht - 3.15, zs1 + 0.6)             # belfry floor
        zpl = 1.25                                   # top of the battered plinth
        zck_ = Ht - 0.3                              # foot of the corbel course
        host_top = host["top"]
        ZB = stages(zpl, zs1)                        # base stage, standing in front of the house
        ZS = stages(zs1, zbel)                       # shaft
        levels = ZB[1:] + ZS[1:] + [zck_]
        zrf = host_top(tcx, tcy + hh)                # host roof against the back of the shaft
        av = min(hb, host["yf"] + 0.12 - tcy)        # the base's flanks show from its front corners back to the house front

        def tmask(q):
            g = 0.04 + 0.1 * nz(q, 2.0) + 0.25 * _ss(Ht - 2.5, Ht + 0.5, q.z) * nz(q, 1.1)
            g = max(g, 0.6 * _ss(1.6, 0.0, q.z))
            for zc_ in levels:
                if q.z < zc_ + 0.01:
                    g = max(g, 0.5 * _ss(1.8, 0.0, zc_ - q.z) * (0.6 + 0.4 * nz(q, 0.6)))    # dirt washed down under every course
            return (min(1.0, g), 0.2 * nz(q, 0.8), age * 0.7 * nz(q, 1.6) * (1.0 if q.y > tcy else 0.45) * _ss(Ht, 0.0, q.z))

        def pmask(q):
            return (min(1.0, 0.4 + 0.45 * _ss(zpl, 0.0, q.z) + 0.15 * nz(q, 0.7)), 0.25, age * 0.9 * nz(q, 1.1))

        def qmask(q):
            return (0.03 + 0.3 * _ss(2.5, 0.0, q.z), 0.45, 0.15 * age * nz(q, 1.3))

        def sband(dg):
            return lambda q: (min(1.0, max(0.0, 0.25 + 0.4 * nz(q, 1.5) + dg)), 0.2, age * 0.8 * nz(q, 1.2) * (1.0 if q.y > tcy else 0.5))

        smask = sband(0.0)

        def tfr(k, half):
            nrm = Vector(hints[k])
            return _Fr(Vector((tcx, tcy, 0)) + nrm * half, nrm)

        def span(k, half):
            """The run of face k (frame coordinate a) that stands free of the house front."""
            if k == 1:
                return -half, min(half, av)
            if k == 3:
                return -min(half, av), half
            return -half, half

        def course(k, half, z, a0, a1, pj_=PJ):
            """String course on one face: weathered top and undercut bed, mitred at the free corners."""
            W = tfr(k, half)
            m0 = pj_ if a0 <= -half + 1e-6 else 0.0
            m1 = pj_ if a1 >= half - 1e-6 else 0.0
            K(dk_, [W.P(a0 - m0, z, pj_), W.P(a1 + m1, z, pj_), W.P(a1, z + UPJ, 0.0), W.P(a0, z + UPJ, 0.0)], W.n + UP, cm_top)
            K(dk_, [W.P(a0, z - DNJ, 0.0), W.P(a1, z - DNJ, 0.0), W.P(a1 + m1, z, pj_), W.P(a0 - m0, z, pj_)], W.n - UP, cm_und)

        # ---- walls: base stage carried down to the ground in front of the facade, shaft and belfry stage above
        new_uvx()
        for k in (0, 1, 3):
            W = tfr(k, hb)
            a0, a1 = span(k, hb)
            for za, zb in zip(ZB[:-1], ZB[1:]):
                rect(K, tk_, W, a0, a1, za + 0.03, zb, 0.0, tmask)
        if not far:
            rect(K, tk_, tfr(0, hb), -hb, hb, 0.0, zpl + 0.03, 0.0, tmask)        # behind the door, between the plinth's jambs
        for k in range(4):
            W = tfr(k, hh)
            for za, zb in list(zip(ZS[:-1], ZS[1:])) + [(zbel, zck_)]:
                if k == 2 and zb < zrf:
                    continue
                rect(K, tk_, W, -hh, hh, za + 0.03, zb, 0.0, tmask)
        # ---- battered plinth (split round the door)
        pj = 0.24 * fs
        dw_ = 0.72
        for k in (0, 1, 3):
            W = tfr(k, hb)
            a0, a1 = span(k, hb)
            f0 = pj if a0 <= -hb + 1e-6 else 0.0
            f1 = pj if a1 >= hb - 1e-6 else 0.0
            if k == 0 and not far:
                K(dk_, [W.P(a0 - f0, 0.0, pj), W.P(-dw_, 0.0, pj), W.P(-dw_, zpl, 0.0), W.P(a0, zpl, 0.0)], W.n + UP * 0.2, pmask)
                K(dk_, [W.P(dw_, 0.0, pj), W.P(a1 + f1, 0.0, pj), W.P(a1, zpl, 0.0), W.P(dw_, zpl, 0.0)], W.n + UP * 0.2, pmask)
                for s in (-1, 1):
                    K(dk_, [W.P(s * dw_, 0.0, 0.0), W.P(s * dw_, 0.0, pj), W.P(s * dw_, zpl, 0.0)], W.rt * -s, pmask)
            else:
                K(dk_, [W.P(a0 - f0, 0.0, pj), W.P(a1 + f1, 0.0, pj), W.P(a1, zpl, 0.0), W.P(a0, zpl, 0.0)], W.n + UP * 0.2, pmask)
        # ---- string courses at every stage, the weathered offset at the eaves line, the belfry string
        for k in (0, 1, 3):
            a0, a1 = span(k, hb)
            for z in ZB[1:-1]:
                course(k, hb, z, a0, a1)
            W = tfr(k, hb)
            m0 = 1.0 if a0 <= -hb + 1e-6 else 0.0
            m1 = 1.0 if a1 >= hb - 1e-6 else 0.0
            o_ = 0.12 * fs
            K(dk_, [W.P(a0 - m0 * o_, zs1, o_), W.P(a1 + m1 * o_, zs1, o_), W.P(a1 - m1 * 0.13, zs1 + 0.3, -0.13),
                    W.P(a0 + m0 * 0.13, zs1 + 0.3, -0.13)], W.n + UP, cm_top)
            K(dk_, [W.P(a0, zs1 - DNJ, 0.0), W.P(a1, zs1 - DNJ, 0.0), W.P(a1 + m1 * o_, zs1, o_), W.P(a0 - m0 * o_, zs1, o_)],
              W.n - UP, cm_und)
        for k in range(4):
            for z in ZS[1:]:
                if k == 2 and z - DNJ < zrf:
                    continue
                course(k, hh, z, -hh, hh)
            # corbel course under the cornice / parapet
            W = tfr(k, hh)
            K(dk_, [W.P(-hh, zck_, 0.0), W.P(hh, zck_, 0.0), W.P(hh + 0.2, Ht + 0.1, 0.2), W.P(-hh - 0.2, Ht + 0.1, 0.2)], W.n - UP,
              (0.75, 0.0, 0.0))
        # ---- corner lesenes: pale dressed stone up every free corner (cleaner stone of the same bed on an ashlar shaft)
        qw, qp = 0.36 * fs, 0.05

        def lesene(k, half, za, zb, s):
            e_ = half + qp
            if zb - za > 0.6:
                rect(K, dk_, tfr(k, half), -e_ if s < 0 else e_ - qw, -e_ + qw if s < 0 else e_, za, zb, qp, qmask)
        for k, s in ((0, -1), (0, 1), (1, -1), (3, 1)):
            lesene(k, hb, zpl, zs1 - DNJ, s)
        for k in range(4):
            for s in (-1, 1):
                rear = k == 2 or (k == 1 and s > 0) or (k == 3 and s < 0)
                lesene(k, hh, max(zs1 + 0.3, zrf) if rear else zs1 + 0.3, zck_, s)
        # ---- door, and a lancet in every stage of every face that stands free
        rcl = min(0.36 * wt, 0.82)
        zck = zbel - 0.5 - rcl
        has_clock = bool(p.clock) and zck - rcl > zs1 + 0.5
        zcl = zck - rcl - 0.45 if has_clock else 1e9
        lw0 = 0.42 * fs
        if not far:
            reg(host, tfr(0, hb), 0.0, 1.15, 1.1, 2.3, kind="door", fk="ash", sk="ash", brick=False, eo=0.1, wkey=tk_, rec=0.12)
        lancets(tfr(0, hb), 0.0, ZB, lw0, zlo=0.0 if far else 2.75)
        lancets(tfr(0, hh), 0.0, ZS, lw0, zhi=zcl, first=0.1)
        lancets(tfr(2, hh), 0.0, ZS, lw0, zlo=host["Hr"] + 0.3, zhi=zcl, first=0.1)
        for k in (1, 3):
            a0, a1 = span(k, hb)
            f0, f1 = (a0 + qw, a1 - 0.12) if k == 1 else (a0 + 0.12, a1 - qw)
            if f1 - f0 >= 0.3 * fs + 0.3:
                lancets(tfr(k, hb), 0.5 * (f0 + f1), ZB, 0.3 * fs)
            lancets(tfr(k, hh), 0.0, ZS, lw0, zlo=host_top(tcx, tcy + lw0) + 0.35, first=0.1)
        # ---- belfry openings on all four faces: pointed, two lights under one head, softly lit from inside and
        # closed by stout timber louvres
        bw = wt * 0.32
        z0b, ztb = zbel + UPJ + 0.3, zck_ - 0.3
        zsp = ztb - bw * 0.75
        for k in range(4):
            W = tfr(k, hh)
            pts = [(-bw / 2, z0b), (bw / 2, z0b), (bw / 2, zsp), (0.0, ztb), (-bw / 2, zsp)]
            sd = rt_.u(0.05, 0.95)
            sw_ = 0.17
            K(sfk, [W.P(-bw / 2 - sw_, z0b - 0.16, 0.04), W.P(bw / 2 + sw_, z0b - 0.16, 0.04), W.P(bw / 2 + sw_, zsp + 0.07, 0.04),
                    W.P(0.0, ztb + 0.24, 0.04), W.P(-bw / 2 - sw_, zsp + 0.07, 0.04)], W.n, fmask)
            K("dim", [W.P(a, b, 0.07) for a, b in pts], W.n, (0.3, 0.0, sd), uvs=[(a, b - zbel) for a, b in pts])

            def hwid(z):
                return bw / 2 if z <= zsp else bw / 2 * max(0.0, (ztb - z) / (ztb - zsp))
            zz = z0b + 0.08
            lst = 0.44 if (far or DET > 0) else 0.4      # louvre blades: the regular dark mask over the belfry glow
            lbl = 0.6 * lst
            while zz + lbl < ztb - 0.1:
                a0_, a1_ = hwid(zz) + 0.01, hwid(zz + lbl) + 0.01
                K("trim", [W.P(-a0_, zz, 0.1), W.P(a0_, zz, 0.1), W.P(a1_, zz + lbl, 0.1), W.P(-a1_, zz + lbl, 0.1)], W.n,
                  (0.5, 0.2, 0.0))
                zz += lst
            rect(K, sfk, W, -0.075, 0.075, z0b, zsp + 0.55 * (ztb - zsp), 0.13, fmask)       # stone mullion
            pbox(D, sfk, W, -bw / 2 - sw_ - 0.04, bw / 2 + sw_ + 0.04, z0b - 0.16, z0b - 0.06, 0.0, 0.16, "ft", fmask)
        # lead flashing where the shaft comes through the host roof
        if not far:
            y_a, y_b = host["yfu"] - 0.25, tcy + hh + 0.03
            for sx in (-1, 1):
                xe = tcx + sx * (hh + 0.03)
                D("slate", [(xe, y_a, host_top(tcx, y_a) - 0.2), (xe, y_b, host_top(tcx, y_b) - 0.2),
                            (xe, y_b, host_top(tcx, y_b) + 0.28), (xe, y_a, host_top(tcx, y_a) + 0.28)], (sx, 0, 0), lm)
        # clock: softly back-lit round dial with a dark timber rim, hour batons and two hands
        if has_clock:
            for s in (-1, 1):
                # the rear dial is left out where the host roof or one of its stacks would stand in front of it
                if s > 0 and (zck - rcl - 0.2 < host_top(tcx, tcy + hh) or any(abs(c[0] - tcx) < hh + 2.6 for c in host["chims"])):
                    continue
                nrm = Vector((0, s, 0))
                rt = UP.cross(nrm)
                o = Vector((tcx, tcy + s * hh, zck))

                def P(a, rad, dd):
                    return o + rt * (rad * cos(a)) + UP * (rad * sin(a)) + nrm * dd
                N = 24
                NK = 12
                # dark timber rim as one disc behind the dial (distance tier); the modelled rim ring is detail
                K("trim", [P(2 * pi * k / NK, rcl + 0.15, 0.06) for k in range(NK)], nrm, (0.3, 0.2, 0.0))
                K("dim", [P(2 * pi * k / NK, rcl, 0.085) for k in range(NK)], nrm, flat_mask, uvs=flat_uv(NK))
                for k in range(N):
                    a, b = 2 * pi * k / N, 2 * pi * (k + 1) / N
                    if DET == 0:
                        D("trim", [P(a, rcl - 0.03, 0.11), P(b, rcl - 0.03, 0.11), P(b, rcl + 0.14, 0.11), P(a, rcl + 0.14, 0.11)], nrm,
                          (0.3, 0.2, 0.0))
                    if DET == 0 and s < 0:
                        D("trim", [P(a, rcl + 0.14, 0.0), P(b, rcl + 0.14, 0.0), P(b, rcl + 0.14, 0.11), P(a, rcl + 0.14, 0.11)],
                          rt * cos(0.5 * (a + b)) + UP * sin(0.5 * (a + b)), (0.3, 0.2, 0.0))
                for k in range(12):
                    am = 2 * pi * k / 12
                    ca_, sa_ = cos(am), sin(am)
                    tw_ = 0.05 if k % 3 == 0 else 0.03
                    D("trim", [o + rt * (ra * ca_ - t_ * sa_) + UP * (ra * sa_ + t_ * ca_) + nrm * 0.11
                               for ra, t_ in ((rcl * 0.66, -tw_), (rcl * 0.9, -tw_), (rcl * 0.9, tw_), (rcl * 0.66, tw_))],
                      nrm, (0.3, 0.2, 0.0))
                for ang, ln, wd_ in ((rt_.u(0, 6.28), rcl * 0.8, 0.06), (rt_.u(0, 6.28), rcl * 0.55, 0.08)):
                    ca, sa = cos(ang), sin(ang)

                    def Q(al, ac):
                        return o + rt * (al * ca - ac * sa) + UP * (al * sa + ac * ca) + nrm * 0.13
                    K("trim", [Q(-0.1, -wd_), Q(ln, -wd_ * 0.5), Q(ln, wd_ * 0.5), Q(-0.1, wd_)], nrm, (0.3, 0.2, 0.0))
            clock_light = (tcx, tcy - hh - 1.2, zck - 0.2)

        zc = Ht + 0.32
        if style == "spire":
            boxf(K, dk_, (tcx - hh - 0.2, tcy - hh - 0.2, Ht + 0.1), (tcx + hh + 0.2, tcy + hh + 0.2, zc), "xXyY", cm_top)
            boxf(D, dk_, (tcx - hh - 0.2, tcy - hh - 0.2, Ht + 0.1), (tcx + hh + 0.2, tcy + hh + 0.2, zc), "z", tmask)
            hsp = wt * rt_.u(2.0, 2.4) * (1.25 if far else 1.0)
            c8 = cos(pi / 8)
            za = zc + hsp
            z1_ = zc + 0.3 * wt
            r1_ = 0.62 * hh / c8
            rings = [((hh + 0.14) / c8, zc), (r1_, z1_), (0.56 * r1_, z1_ + 0.44 * (za - z1_))]
            V = [[Vector((tcx + R * cos(pi / 8 + k * pi / 4), tcy + R * sin(pi / 8 + k * pi / 4), z)) for k in range(8)] for R, z in rings]
            apex = Vector((tcx, tcy, za))
            tl_ = rt_.u(-0.1, 0.1)
            for k in range(8):
                j = (k + 1) % 8
                am = (k + 1) * pi / 4
                hint = (cos(am), sin(am), 0.6)
                K("slate", [V[0][k], V[0][j], V[1][j], V[1][k]], hint, sband(0.3))
                K("slate", [V[1][k], V[1][j], V[2][j], V[2][k]], hint, sband(tl_))
                K("slate", [V[2][k], V[2][j], apex], hint, sband(-tl_ - 0.08))
            # corner pinnacles hiding the broach corners
            pc = hh + 0.2 - 0.27
            for sx in (-1, 1):
                for sy in (-1, 1):
                    px, py = tcx + sx * pc, tcy + sy * pc
                    boxf(D, dk_, (px - 0.23, py - 0.23, zc), (px + 0.23, py + 0.23, zc + 0.55), "xXyY", tmask)
                    prism(K, "slate", px, py, zc, zc + 1.7, 0.3, 0.0, 4, smask, a0=pi / 4)     # (stands on the cornice at every LOD)
            fz = za - 0.25
        else:
            # crenellated parapet (it carries the cornice) with a low slated cap
            po = hh + 0.2
            boxf(K, dk_, (tcx - po, tcy - po, Ht + 0.1), (tcx + po, tcy + po, zc + 0.5), "xXyY", tmask)
            ms = 0.52
            pos = [-(po - ms / 2), 0.0, po - ms / 2]
            for a in range(3):
                for b in range(3):
                    if a != 1 or b != 1:
                        mx, my = tcx + pos[a], tcy + pos[b]
                        boxf(K, dk_, (mx - ms / 2, my - ms / 2, zc + 0.5), (mx + ms / 2, my + ms / 2, zc + 1.08), "xXyYZ", tmask)
            za = zc + 0.5 + 0.55 * wt
            prism(K, "slate", tcx, tcy, zc + 0.5, za, (po - 0.12) * sqrt(2.0), 0.0, 4, smask, a0=pi / 4)
            fz = za - 0.2
        finial(tcx, tcy, fz)

    # ---------------------------------------------------------------- street lamps in front of the row
    rs = rng.fork("lamps")
    nl = 0 if far else int(round(p.lamps))
    lamps = []
    for k in range(nl):
        lx = -L / 2 + (k + 0.5) * L / nl + rs.u(-0.9, 0.9)
        lx = min(L / 2 - 0.6, max(-L / 2 + 0.6, lx))
        ly = min(h["ymin"] for h in houses if h["x0"] - 1.2 < lx < h["x1"] + 1.2) - rs.u(1.5, 1.9)
        hp = rs.u(3.7, 4.0)
        pm = (0.3, 0.3, 0.0)
        boxf(D, "trim", (lx - 0.15, ly - 0.15, 0.0), (lx + 0.15, ly + 0.15, 0.5), "xXyYZ", pm)
        prism(K, "trim", lx, ly, 0.5, hp - 0.55, 0.075, 0.05, 4, pm, a0=pi / 4, cap=False)
        boxf(D, "trim", (lx - 0.3, ly - 0.02, hp - 1.0), (lx + 0.3, ly + 0.02, hp - 0.96), "yYZz", pm)       # ladder rest
        prism(DK, "trim", lx, ly, hp - 0.55, hp - 0.45, 0.07, 0.14, 4, pm, a0=pi / 4, cap=False)
        # glowing amber lantern: four tapered panes, closed below so it also glows seen from the street
        lq = [[(lx + r_ * cos(pi / 4 + j * pi / 2), ly + r_ * sin(pi / 4 + j * pi / 2), z_) for j in range(4)]
              for r_, z_ in ((0.14, hp - 0.45), (0.23, hp - 0.02))]
        for j in range(4):
            j1 = (j + 1) % 4
            K("lit", [lq[0][j], lq[0][j1], lq[1][j1], lq[1][j]], (cos((j + 1) * pi / 2), sin((j + 1) * pi / 2), 0.0), flat_mask, uvs=flat_uv(4))
        K("lit", lq[0], (0, 0, -1), flat_mask, uvs=flat_uv(4))
        prism(DK, "trim", lx, ly, hp - 0.02, hp + 0.2, 0.31, 0.0, 4, pm, a0=pi / 4)
        K("trim", [(lx + 0.31 * cos(pi / 4 + j * pi / 2), ly + 0.31 * sin(pi / 4 + j * pi / 2), hp - 0.02) for j in range(4)],
          (0, 0, -1), pm)
        lamps.append((lx, ly, hp - 0.23))

    # ---------------------------------------------------------------- windows: which are lit, and how
    rw = rng.fork("windows")
    free = [wd for wd in wins if wd["kind"] is None]
    # some households are mostly out or asleep; rooms under the roofs (seen over the foreground roofscape) light first
    mood = {H["i"]: rw.weighted([(0.55, 3), (1.0, 4), (2.2, 3)]) for H in houses}
    score = [rw.u(0.05, 1.0) * mood[wd["hi"]] * (0.6 if wd["top"] else 1.0) * (1.25 if wd["back"] else 1.0) for wd in free]
    order = sorted(range(len(free)), key=lambda k: score[k])
    nlit = int(round(float(p.lit_ratio) * (0.6 if far else 1.0) * len(free)))
    chosen = []
    if nlit > 0:
        # no house front is dead: every house keeps one clearly lit room on the street side, upstairs if it can
        for H in houses:
            c_ = [k for k in order if free[k]["hi"] == H["i"] and not free[k]["back"] and not free[k].get("dormer")
                  and not free[k].get("small")]
            c_ = [k for k in c_ if free[k]["fz"] > 0.5 and free[k]["w"] > 0.5] or c_
            if c_:
                chosen.append(c_[0])
                free[c_[0]]["sure"] = True
        # nor is an exposed side: every run-end wall and the landmark keep a lit window, and so do most backs
        rg = rng.fork("litgroups")
        first = {}
        for k in order:
            g_ = free[k].get("grp")
            if g_ is not None and g_ not in first:
                first[g_] = k
        for g_, k in first.items():
            if (g_[0] != "back" or rg.chance(0.75)) and k not in chosen:
                chosen.append(k)
    for k in order:
        if len(chosen) >= nlit:
            break
        if k not in chosen:
            chosen.append(k)
    chosen = set(chosen)
    for k, wd in enumerate(free):
        wd["kind"] = "lit" if k in chosen else "dark"
    lit_all = [wd for wd in wins if wd["kind"] == "lit"]
    for wd in wins:
        if wd["kind"] == "lit":
            # two glazings: the lamp-lit room, and a deep amber one at a third of its glow for rooms lit by the
            # fire or one candle. Every pane has its own room, lamp position and curtains (special = per-window
            # seed) and its own film of dirt on the glass; the guaranteed street-side room is always fully lit
            if rw.chance(0.36) and not wd.get("sure"):
                wd["kind"] = "dim"
            wd["pm"] = (rw.u(0.0, 0.3), rw.u(0.03, 0.97), 0.0)
        elif wd["kind"] == "dark":
            wd["pm"] = (rw.u(0.0, 0.5), 0.0, 0.0)

    # ---------------------------------------------------------------- lights: max_lights real ones per run
    # (street lamps first, spread along the run, then the clock, then a dormer); any further lamp keeps its
    # glowing lantern head but casts no light - nothing is faked onto the walls or the paving
    nlights = 0
    MAXL = int(round(p.max_lights))
    if p.spill > 0.5 and not far:
        sp_ = float(p.spill)
        used_x = []
        while nlights < MAXL and len(used_x) < len(lamps):
            k = max((k for k in range(len(lamps)) if lamps[k][0] not in used_x),
                    key=lambda k: min([abs(lamps[k][0] - x) for x in used_x] or [-abs(lamps[k][0])]))
            lcalls.append(((f"lamp{k}", lamps[k]), dict(color="#ffb45e", intensity=sp_, range=13.0, flicker=0.1, shadow_near=0.55)))
            used_x.append(lamps[k][0])
            nlights += 1
        if clock_light is not None and nlights < MAXL:
            lcalls.append((("clock", clock_light), dict(color="#ffc67a", intensity=0.1 * sp_, range=7.0, flicker=0.03)))
            used_x.append(clock_light[0])
            nlights += 1
        cands = [d for d in dorm_lights if d[2]["kind"] in ("lit", "dim")]
        for d in sorted(cands, key=lambda d: -min([abs(d[0] - x) for x in used_x] or [0.0])):
            if nlights >= MAXL:
                break
            if all(abs(d[0] - x) > 3.5 for x in used_x):
                d[2]["kind"] = "lit"
                lcalls.append(((f"dormer{nlights}", d[1]), dict(color="#ffb45e", intensity=0.13 * sp_, range=6.0, flicker=0.06)))
                used_x.append(d[0])
                nlights += 1

    # ---------------------------------------------------------------- window units
    # A unit is a real opening: the wall is cut out behind it (see _MB.flush) and the casing sits back in the
    # hole as ONE sheet that is wider than the hole - so the reveal is there from every angle without side faces -
    # with the pane on it and a stout mullion / transom over the pane (the regular dark mask in the glow).
    # Sheet, pane, the bars of lit windows, sill and lintel are distance tier; reveals, sash frames, fine bars
    # and shutters are close-range detail. Joinery is one even dark stain (no painted weathering: blotches on a
    # 4 cm glazing bar read as noise); only the nosing of a timber sill is rubbed pale.
    edir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for wd in wins:
        W, a, b = wd["W"], wd["a"], wd["b"]
        hw, hh2 = wd["w"] / 2, wd["h"] / 2
        kind = wd["kind"]
        door = kind == "door"
        back = wd["back"]
        rec = float(wd["rec"])
        proud = rec <= 0.0
        # lod 2: a recessed window stands proud on a wall left whole - past the casing sheet of an opening one looks into
        # the house at a raking angle (lod 0 has reveals there), and a wall cut up round every unit is the dearest thing
        # of the far level. It stands clear in front of timbers, courses and lintels (2 cm and more to each).
        onwall = lod() == 2 and not proud and not door
        df_ = max(TD, 0.03) + 0.02
        ds, dp, db = (0.02, 0.045, 0.075) if proud else ((df_, df_ + 0.025, df_ + 0.055) if onwall else (-rec, -rec + 0.014, -rec + 0.036))
        fk, eo = wd["fk"], wd["eo"]
        bk = "trim"                                      # glazing bars are dark joinery, whatever the casing
        # casing seen around the pane (a door laid on a back wall gets a pale frame round its leaf)
        ec = (0.11 if proud else 0.0) if door else max(eo, wd["ec"])
        bot, topz = b - hh2, b + hh2
        lit_ = kind in ("lit", "dim")
        cmask = join_mask if fk == "trim" else fmask
        bmask = join_mask if bk == "trim" else fmask
        cor = [(a - hw, bot), (a + hw, bot), (a + hw, topz), (a - hw, topz)]
        inner = (a - hw - ec, a + hw + ec, bot - (0.0 if door else ec), topz + ec)
        rm = 0.0 if (proud or onwall) else float(wd["rm"])
        leaf = door and not proud
        K("trim" if leaf else fk, [W.P(inner[0] - rm, inner[2] - (0.0 if door else rm), ds), W.P(inner[1] + rm, inner[2] - (0.0 if door else rm), ds),
                                   W.P(inner[1] + rm, inner[3] + rm, ds), W.P(inner[0] - rm, inner[3] + rm, ds)], W.n,
          (0.12, 0.0, 0.0) if leaf else cmask)
        if door and proud:
            K("trim", [W.P(x_, y_, dp) for x_, y_ in cor], W.n, (0.12, 0.0, 0.0))
        if not proud:
            holes.append((W.P(0.0, 0.0, 0.0), W.n, W.rt, inner, inner))
            if onwall:
                whole.add(len(holes) - 1)
            if DET <= 1 and not far:
                # the reveal proper, in the wall's own material
                rv = [(inner[0], inner[2]), (inner[1], inner[2]), (inner[1], inner[3]), (inner[0], inner[3])]
                for k in range(4):
                    if door and k == 0:
                        continue
                    j = (k + 1) % 4
                    (DK if door else D)(wd["wkey"], [W.P(*rv[k], 0.0), W.P(*rv[j], 0.0), W.P(*rv[j], ds), W.P(*rv[k], ds)],   # (a door stays an opening at lod 2)
                      -(W.rt * edir[k][0] + UP * edir[k][1]), (0.5, 0.1, 0.0))
        if door:
            pbox(D, "ash" if fk == "ash" else "brick", W, a - hw - eo - 0.1, a + hw + eo + 0.1, 0.0, 0.13, 0.0, 0.34, "ftlr",
                 (0.5, 0.4, 0.2 * age))
            if DET <= 1:
                rect(D, fk, W, a - hw, a + hw, topz - 0.42, topz - 0.37, db, cmask)       # transom rail
                rect(D, "trim", W, a - 0.02, a + 0.02, bot, topz - 0.42, db, (0.7, 0.0, 0.0))     # meeting stile shadow
            continue
        g, sd, _w = wd.get("pm", (0.0, 0.0, 0.0))
        v0 = bot - wd["fz"]
        if wd["arch"]:
            # pointed head (lancets, stair slits): the pane is a pentagon on the rectangular stone surround
            zsp = topz - 1.1 * hw
            K(kind, [W.P(a - hw, bot, dp), W.P(a + hw, bot, dp), W.P(a + hw, zsp, dp), W.P(a, topz, dp), W.P(a - hw, zsp, dp)], W.n,
              (g, 0.0, sd), uvs=[(-hw, v0), (hw, v0), (hw, v0 + zsp - bot), (0.0, v0 + 2 * hh2), (-hw, v0 + zsp - bot)])
        else:
            # (lod 2 has no sash frame: a lit pane is drawn as much smaller as the frame covers of it at lod 0, the
            #  casing sheet shows in its place - the row glows no brighter from far than from near)
            fi = 0.05 if (lod() == 2 and lit_ and DET == 0 and not back and not far and p.tier != "distance") else 0.0
            K(kind, [W.P(x_ + sx_ * fi, y_ + sy_ * fi, dp) for (x_, y_), (sx_, sy_) in zip(cor, ((1, 1), (-1, 1), (-1, -1), (1, -1)))],
              W.n, (g, 0.0, sd),
              uvs=[(-hw + fi, v0 + fi), (hw - fi, v0 + fi), (hw - fi, v0 + 2 * hh2 - fi), (-hw + fi, v0 + 2 * hh2 - fi)])
        # glazing bars: one stout mullion / transom per window holds at distance (a far window stays a window of
        # several lights, never one glowing quad); further bars and the sash frame are fine detail
        bv, bh = wd["bars"]
        if DET >= 2 or back or far:
            bv, bh = min(bv, 1), min(bh, 1)
        EB = K if lit_ else (D if DET <= 1 else None)
        bk_ = (0.042 if bv <= 1 else 0.03) * (1.45 if far else 1.0)
        if EB is not None:
            for k in range(bv):
                xa_ = a - hw + (k + 1) * (2 * hw) / (bv + 1)
                rect(EB if (k == bv // 2 or bv == 1) else D, bk, W, xa_ - bk_, xa_ + bk_, bot, topz, db, bmask)
            for k in range(bh):
                zb_ = bot + (k + 1) * (2 * hh2) / (bh + 1) + (0.14 * hh2 if bh == 1 else 0.0)
                if k == bh - 1:
                    # (a lit window on a back wall holds one bar at distance: its mullion, or else its transom)
                    rect(EB if not (back and bv) else D, bk, W, a - hw, a + hw, zb_ - bk_, zb_ + bk_, db + 0.004, bmask)
                else:
                    rect(D, bk, W, a - hw, a + hw, zb_ - 0.024, zb_ + 0.024, db, bmask)
        if DET == 0 and not back:
            fw = 0.05
            inn = [(a - hw + fw, bot + fw), (a + hw - fw, bot + fw), (a + hw - fw, topz - fw), (a - hw + fw, topz - fw)]
            for k in range(4):
                j = (k + 1) % 4
                D(bk, [W.P(*cor[k], db - 0.006), W.P(*cor[j], db - 0.006), W.P(*inn[j], db - 0.006), W.P(*inn[k], db - 0.006)], W.n, bmask)
        if wd["sill"]:
            # weathered sill as one sloping stone / board: reads from the street and from the roofs
            sa, sb_ = a - hw - ec - 0.07, a + hw + ec + 0.07
            smk = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.9, 0.0), (0.0, 0.9, 0.0)] if wd["sk"] == "trim" else \
                [(0.3, 0.0, 0.0), (0.3, 0.0, 0.0), (0.05, 0.3, 0.0), (0.05, 0.3, 0.0)]
            K(wd["sk"], [W.P(sa, bot - ec, 0.0), W.P(sb_, bot - ec, 0.0), W.P(sb_, bot - ec - 0.09, 0.13), W.P(sa, bot - ec - 0.09, 0.13)],
              -W.n * 0.75 + UP, smk)
            if DET <= 1:
                D(wd["sk"], [W.P(sa, bot - ec - 0.15, 0.0), W.P(sb_, bot - ec - 0.15, 0.0), W.P(sb_, bot - ec - 0.09, 0.13),
                             W.P(sa, bot - ec - 0.09, 0.13)], W.n - UP, (0.6, 0.1, 0.0))
            if wd["brick"] and not proud:
                rect(K, "plA", W, a - hw - eo - 0.06, a + hw + eo + 0.06, topz + eo, topz + eo + 0.15, 0.03, fmask)   # stone lintel
        if wd["shut"]:
            sw_ = hw * 0.96
            for sx in (-1, 1):
                xa_ = a + sx * (hw + ec + 0.03)
                xb_ = xa_ + sx * sw_
                rect(DK, "trim", W, min(xa_, xb_), max(xa_, xb_), bot - 0.02, topz + 0.02, 0.04, trim_mask)

    # ---------------------------------------------------------------- budget
    for mb in B.values():
        mb.flush(holes, whole)
    total = sum(mb.nt for mb in B.values())
    if total > TRI_CAP and DET < 3:
        return build(p, rng, DET + 1)

    # ---------------------------------------------------------------- emit one mesh per material
    hz = float(p.haze) if far else 0.0

    ml = float(p.moonlit)
    hazec = _mix(HAZE, HAZE_MOON, ml)

    def tint(c):
        # the far ring's colours drain to grey-blue but keep their own value: limewash stays pale, brick mid,
        # timber and slate dark - walls, frames and roofs so stay apart through the haze
        return _hazed(c, hazec, 0.3 + 0.6 * hz, 0.08 + 0.26 * hz) if far else c
    gl = float(p.glow)
    # roof coverings: a little smoother and cleaner with `moonlit`, so the slopes pick up the sky instead of
    # sinking into brown-black; the library roofs are not tintable, the far layer's haze is left to the fog
    rkw = dict(rough=round(1.0 - 0.3 * ml, 3), grime=round(1.0 - 0.55 * ml, 3)) if ml > 0.01 else {}
    mats = {
        "plA": lambda: mat("plaster_lime", tint(p.wall_a)), "plB": lambda: mat("plaster_lime", tint(p.wall_b)),
        # the far ring is seen through 100 m+ of night air: its brick, stone and roof coverings are tintable
        # stand-ins sunk into the same haze as its limewash (the coursed library materials cannot be tinted and
        # would stand out of the haze brown-black or sand-bright); bond and coursing are below a pixel there
        "brick": (lambda: mat("plaster_rough", tint("#8f5f4c"))) if far else (lambda: mat("brick_red")),
        "ash": (lambda: mat("stone_smooth", tint("#b3a892"))) if far else (lambda: mat("ashlar_limestone")),
        "tile": (lambda: mat("plaster_rough", tint("#6a4c44"), rough=0.8)) if far else (lambda: mat("roof_tile_clay", **rkw)),
        "slate": (lambda: mat("plaster_rough", tint("#3e434e"), rough=0.7)) if far else (lambda: mat("roof_slate", **rkw)),
        "lit": lambda: mat("window_lit", p.lit_color, glow=gl * (1.0 if far else 1.5), curtain=0.3 + 0.4 * float(p.curtains)),
        # the deep amber glazing: fire- and candle-lit rooms, the belfry, and (as a flat patch) the clock dial
        "dim": lambda: mat("window_lit", p.dim_color, glow=gl * (0.42 if far else 0.48), curtain=0.6),
        # joinery and timbers: an even stain - the automatic edge wear / cavity grime would mottle 4 cm members
        "dark": (lambda: mat("ceramic_glazed", tint("#2b323c"), rough=0.6)) if far else (lambda: mat("window_dark")),
        "trim": lambda: mat("wood_oak", tint(p.trim), grime=0.6, wear=0.22),
    }
    for args, kw in lcalls:
        light(*args, **kw)
    xmin, xmax = 1e9, -1e9
    for key, mb in B.items():
        if not mb.f:
            continue
        xmin = min(xmin, min(v[0] for v in mb.v))
        xmax = max(xmax, max(v[0] for v in mb.v))
        ob = mesh_object("row_" + key, mb.v, mb.f)
        uvl = ob.data.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", [c for fu in mb.uv for q in fu for c in q])
        assign(ob, mats[key]())
        uv_keep(ob)
        flat(ob)
        if len(ob.data.vertices) == len(mb.m):
            arr = np.array(mb.m, dtype=np.float32)
            paint(ob, grime=lambda co, nn, a=arr: a[:, 0], wear=lambda co, nn, a=arr: a[:, 1], special=lambda co, nn, a=arr: a[:, 2])
        else:
            print(f"DBG mask mismatch on {key}: {len(ob.data.vertices)} verts vs {len(mb.m)} masks")
    print(f"DBG layer={p.layer} houses={n} tris={total} pinned={sum(mb.nk for mb in B.values())} cap={TRI_CAP:.0f} det={DET} mats={sum(1 for mb in B.values() if mb.f)} windows={len(wins)} "
          f"lit={len(lit_all)} lamps={len(lamps)} lights={nlights} x=[{xmin:.3f},{xmax:.3f}] L/2={L / 2:.3f}")
