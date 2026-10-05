"""Far landmark for the skyline: a castle / great church on a wooded crag, built to stand 150-300 m behind
the roofscape rows and to read as a pale, softly glowing silhouette against the navy night sky. A crenellated
curtain wall on a battered plinth, with round and square mural towers of different girth, height and roof and a
twin-drum gatehouse, rings the summit of a rock crag; inside stand a square keep with corner bartizans, a helm
roof with lit lucarnes and a softly lit clock, a great hall with a row of tall lancets, a chapel and a cluster
of slim spired towers (belfry stages, lucarnes, finials, corbelled flying turrets, a flying bridge). style
switches the summit to a cruciform cathedral (twin west towers, crossing spire, rose window in the transept
gable, buttressed aisles, apse), a squat citadel (double ring, great drum keep with a beacon) or just a chapel
on a hogback ridge whose crest sinks to both ends, the hamlet in two rows along it (a third silhouette layer).

The crag is an irregular rock: a broken rim, a cliff band of varying height under the walls, ribs and gullies,
ledges, leaning stacks, two shoulders with terraces below the gate, steep flanks and an apron at the foot. A
lane winds round the hill from the foot to the gate on a retaining wall. Every cottage of the hamlet (about
8 x 6 m, chimney stack, lit dormer, often a lean-to) stands on its own level yard with a low retaining wall:
a row on the terraces under the walls, the rest on the apron; a spot is searched until the number asked for
stands, none on top of another. The woodland stands in two to five woods on the flanks: rounded canopy crowns
7-10 m across with one or two smaller ones leaning on each, rank below rank from under the cliffs to the skirt,
with a ragged tree-line and two-tiered firs on the benches; the crowns are welded and shaded smooth.

Night: no light sockets. Everything is self-lit, in baked moonlight tones. The masonry is a textured library
stone that glows through its own texture (coursed limestone ashlar, dressed or rough stone, lime render -
masonry), so the coursing reads at any distance. A wall face takes one of three tones from the moon (moon_az):
moon side, flank (0.35 of it), shade (0.165); corbel rings, string courses, batters and soffits take one even
step below the wall they belong to. Wall heads, wall walks, the inner faces of merlons and every ledge of the
rock carry no baked light: they are seen from above only, where the far view never looks. Staining on the
lowest courses, streaks under parapets and a tonal drift from tower to tower sit in the vertex grime. Slate is
a muted blue-grey well below the wall tone, rock and woodland darker still; cliff faces that are square to the
moon and turned away from the near view (near_az) carry a strong tone that is laid on heavier down the flank
(valley_fog), because the ground fog of the valley takes it away again. Windows, lucarnes, the dial and the
rose are emissive (deep warm orange; cottage lights the hottest; the dial a warm cream at 0.55 of the window
glow, with dark hands; the citadel carries a small brazier flame). floodlit scales all baked tones together,
glow the windows and the dial.

Fog: the night scene has a dense ground fog. Everything below about the rim of a 40 m crag sinks into one dark
navy tone whatever it is painted with, so the hills are tall and compact (37-40 m, about 85 m across) to lift
walls and hamlet terraces clear of it, and the ridge variant is a short hogback with a 30 m crest and very strong
lights (floodlit 25, glow 150) rather than a long low ridge; hill 14-20 with hill_width 160-220 builds that one
for a place with thin fog. floodlit 1.7 is set for the far view; seen from a few dozen metres without the haze
the same stone is pale and bright - about 1.0 suits a near placement.

Distance-proof construction: there are no overlays and no coplanar layers. Every window, the dial and the rose
are cells of the wall itself: a window column is a stack of cells, the piers between columns are single polygons
that carry the column's vertices on one edge, a blind wall is one quad - no T-junctions, nothing to z-fight or
crack at 300 m. Heads, corbels and parapets grow out of the wall faces (overlap, never a shared split edge).
Every face is a triangle fan with hidden 12 mm fin triangles behind it (lod_pins), which the town's LOD
decimator leaves intact - LOD1 / LOD2 are the full model, so the whole thing stays under 15k triangles (the
summit thins out its window storeys first if hill, hamlet and woodland would not fit in full).
Origin at the centre of the hill on the ground, main front (gate, clock, rose) facing -Y."""
from tk import *
from math import sin, cos, pi, hypot, atan2, radians, sqrt, floor
import numpy as np
from mathutils import Vector

META = {"name": "Far hilltop castle", "class": "backdrop", "mount": "ground", "focus": 0.66, "tri_budget": 15000,
        "max_materials": 10, "auto_grime": 0.0}
PARAMS = {
    "style": {"default": "castle", "choices": ["castle", "cathedral", "citadel", "ridge"],
              "desc": "castle = keep, hall and many spired towers; cathedral = great church; citadel = drum keep in two rings; "
                      "ridge = chapel and hamlet on a long hill (a third silhouette layer)"},
    "size": {"default": 0.62, "min": 0.45, "max": 1.6, "desc": "plan scale of the summit buildings (and of the plateau they stand on)"},
    "height": {"default": 20.0, "min": 14.0, "max": 40.0, "desc": "wall height of the keep / nave towers (m); spires rise above it"},
    "spires": {"default": 9, "min": 0, "max": 18, "desc": "castle: slim spired towers clustered inside the walls"},
    "soar": {"default": 1.0, "min": 0.7, "max": 2.0, "desc": "castle: how far the spired towers out-top the keep (height factor)"},
    "slender": {"default": 1.0, "min": 0.6, "max": 1.5, "desc": "castle: girth factor of the spired towers (below 1 = slimmer)"},
    "bridges": {"default": 1, "min": 0, "max": 6, "desc": "castle: flying bridges between neighbouring spired towers"},
    "turrets": {"default": 3, "min": 0, "max": 14, "desc": "castle: corbelled flying turrets hung on the tallest towers"},
    "hill": {"default": 40.0, "min": 0.0, "max": 60.0, "desc": "height of the crag (m); 0 = no hill, stands on the ground; 14-20 with style ridge is a low hogback (needs thin ground fog to be seen)"},
    "hill_width": {"default": 92.0, "min": 60.0, "max": 320.0, "desc": "width of the hill along X (m); the foot never comes closer to the walls than the flanks need"},
    "crag": {"default": 1.0, "min": 0.5, "max": 2.0, "desc": "ruggedness of the hill: taller cliff band under the walls, more hollow flanks"},
    "hamlet": {"default": 12, "min": 0, "max": 30, "desc": "cottages with tiny lights, each on its own level yard with a low retaining wall: a row on the terraces below the gate, the rest on the apron at the foot; ridge: in two rows along the crest, then at the foot; a crag under 100 m across has room for 20-25"},
    "trees": {"default": 32, "min": 0, "max": 48, "desc": "canopy clumps of the woodland (a tall round crown 7-10 m across with one or two lower ones leaning on it, some with a fir), massed rank below rank in two to five woods on the flanks, every fifth in the tree-line on the benches; 0 = bare crag"},
    "feature": {"default": "auto", "choices": ["auto", "clock", "rose", "none"],
                "desc": "softly lit clock dial or rose window on the main front (auto: castle clock, cathedral rose)"},
    "lit_ratio": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "fraction of windows that are lit (the rest are dark openings)"},
    "lit_color": {"default": "#ffc46a", "desc": "colour of the lit windows"},
    "glow": {"default": 12.0, "min": 0.5, "max": 200.0, "desc": "emissive strength of the lit windows; the clock dial / rose glows at 0.55 of it"},
    "floodlit": {"default": 1.7, "min": 0.05, "max": 30.0, "desc": "self-glow of stone, slate, rock and woodland (the baked moonlight), set for the far view at 150-300 m; about 1.0 for a near placement, 20-30 for a low ridge deep in the ground fog"},
    "wall": {"default": "#dfe4f2", "desc": "stone colour in the moonlight (tints dressed / rough / rendered masonry; coursed ashlar keeps its own limestone colour)"},
    "masonry": {"default": "auto", "choices": ["auto", "ashlar", "dressed", "rough", "render"],
                "desc": "walling: coursed limestone ashlar (natural colour), dressed or rough-hewn stone or lime render in the wall colour; "
                        "auto = ashlar for a pale neutral wall colour, dressed stone for a coloured one"},
    "roof": {"default": "#6d86c8", "desc": "roof colour in the moonlight (slate: used at half its saturation, below the wall luminance)"},
    "haze": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "how far all tones sink into the blue night haze"},
    "moon_az": {"default": 78.0, "min": -180.0, "max": 180.0, "desc": "direction the moon stands in, degrees from the front (-Y) towards +X"},
    "valley_fog": {"default": 1.0, "min": 0.0, "max": 4.0, "desc": "moonlight on the cliffs that holds against the ground fog of the valley: laid on stronger towards the foot, on every rock face the near view (near_az) cannot see; 0 = none, the crag sinks into the fog"},
    "near_az": {"default": -25.0, "min": -180.0, "max": 180.0, "desc": "bearing of the nearest viewpoint, degrees from the front (-Y) towards +X: rock faces of the upper crag turned to it keep a faint tone, so they do not burn out when seen without the fog"},
    "lod_pins": {"default": True, "desc": "pin every face so LOD1 / LOD2 keep the model exactly"},
    "seed": {"default": 5, "min": 0, "max": 99999, "desc": "layout seed"},
}
VARIANTS = {
    "cathedral": {"style": "cathedral", "size": 0.62, "height": 16.0, "hill": 38.0, "hill_width": 84.0, "hamlet": 14, "trees": 24,
                  "wall": "#f4e6cf", "masonry": "dressed", "roof": "#58a79b", "glow": 18.0, "floodlit": 2.2, "seed": 9},
    "citadel": {"style": "citadel", "size": 0.56, "hill": 40.0, "hill_width": 84.0, "height": 17.0, "wall": "#e8c79a", "masonry": "rough",
                "roof": "#59638c", "glow": 18.0, "floodlit": 2.2, "lit_ratio": 0.3, "hamlet": 8, "trees": 28, "crag": 1.4, "seed": 14},
    "fairy_spires": {"spires": 15, "height": 18.0, "soar": 1.3, "slender": 0.78, "bridges": 4, "turrets": 8, "hill": 37.0,
                     "hill_width": 64.0, "crag": 1.8, "size": 0.5, "wall": "#f7e4f1", "masonry": "render", "roof": "#9a80e2",
                     "glow": 18.0, "floodlit": 2.6, "lit_ratio": 0.5, "hamlet": 6, "trees": 16, "seed": 27},
    "ridge_hamlet": {"style": "ridge", "size": 0.56, "height": 14.0, "hill": 30.0, "hill_width": 88.0, "hamlet": 12, "trees": 30,
                     "lit_ratio": 0.6, "wall": "#e6dcc6", "masonry": "render", "glow": 150.0, "floodlit": 25.0, "valley_fog": 0.2, "crag": 0.7, "seed": 33},
}

TRI_CAP = 14950
# glow per unit of floodlit: the four kinds of masonry (calibrated to the same wall tone), slate, moonlit cliff
# faces away from the near view (per unit of valley_fog), all other rock, foliage
K_ASH, K_DRS, K_RGH, K_RND, K_SLT, K_RK, K_RS, K_WOD = 3.3, 3.2, 4.15, 3.9, 0.75, 40.0, 0.5, 0.5
FOG_FULL, FOG_HALF = 12.0, 2.5       # that cliff tone is full from 12 m below the rim, halved every 2.5 m above
W_FLANK, W_SHADE = 0.35, 0.165       # wall tone of the flank and the shade side as a fraction of the moon side
UP = Vector((0.0, 0.0, 1.0))
KEYS = ("wl", "w0", "w1", "roof", "rk", "rs", "wood", "lit", "dim", "cool")
# stone tone ladder (material, extra grime): moon side, flank, shade, and two deeper steps for bands and soffits
LAD = (("wl", 0.0), ("w0", 0.0), ("w1", 0.0), ("w1", 0.35), ("w1", 0.6))
HZC = "#727a8e"
NS = 36                             # columns of the hill
SL = (0.0, 0.07, 0.17, 0.31, 0.47, 0.64, 0.8, 1.0)     # its rows: rim, cliff foot ... foot of the hill


def _mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(a[i] + (b[i] - a[i]) * t)):02x}" for i in range(3))


class _MB:
    """Face soup of one material with a grime value per vertex. Every face is a triangle fan; a pinned face also
    carries a hidden fin triangle 12 mm behind every diagonal (for a triangle: on one edge) - a 3-face edge the
    decimator never collapses."""

    def __init__(self, pin, weld=False):
        self.v, self.f, self.uv, self.g, self.nt, self.pin = [], [], [], [], 0, pin
        self.p = []                             # the part (tower, block ...) every vertex belongs to; 0 = hill and loose faces
        self.weld = {} if weld else None        # welded corners: a rounded form that can be shaded smooth (canopies)

    def face(self, P, n, G, part=0):
        k = len(P)
        t1 = n.cross(UP)
        t1 = Vector((1.0, 0.0, 0.0)) if t1.length < 1e-4 else t1.normalized()
        t2 = n.cross(t1)
        uvs = [(q.dot(t1), q.dot(t2)) for q in P]
        if self.weld is None:
            ix = list(range(len(self.v), len(self.v) + k))
            self.v += [(q.x, q.y, q.z) for q in P]
            self.g += G
            self.p += [part] * k
        else:
            ix = []
            for q, g_ in zip(P, G):
                kq = (round(q.x, 3), round(q.y, 3), round(q.z, 3))
                if kq not in self.weld:
                    self.weld[kq] = len(self.v)
                    self.v.append((q.x, q.y, q.z))
                    self.g.append(g_)
                    self.p.append(part)
                ix.append(self.weld[kq])
            if len(set(ix)) < k:
                return
        for j in range(1, k - 1):
            self.f.append((ix[0], ix[j], ix[j + 1]))
            self.uv.append([uvs[0], uvs[j], uvs[j + 1]])
        self.nt += k - 2
        if not self.pin:
            return
        off = min(0.012, 0.2 * min((P[(i + 1) % k] - P[i]).length for i in range(k)))

        def fin(ia, ib, apex):
            self.v.append((apex.x, apex.y, apex.z))
            self.g.append(0.5 * (G[ia] + G[ib]))
            self.p.append(part)
            self.f.append((ix[ia], ix[ib], len(self.v) - 1))
            self.uv.append([uvs[ia], uvs[ib], uvs[ia]])
            self.nt += 1
        if k == 3:
            e = (P[1] - P[0]).normalized()
            mid = (P[0] + P[1]) * 0.5 - n * off
            fin(0, 1, mid)
            fin(0, 1, mid + n.cross(e) * off)
        else:
            for j in range(2, k - 1):
                fin(0, j, (P[0] + P[j]) * 0.5 - n * off)


def _gen(p, rng, DET):
    """The whole model as face soups per tone. DET = 0 is the full detail; 1-3 thin out the summit buildings
    (window storeys, arrow loops, sides of the round towers) until hill, hamlet and woodland fit the triangle
    budget in full; only at 4 the slopes are cut short."""
    S, HT, Hh = float(p.size), float(p.height), float(p.hill)
    SOAR, SLN, CRG = float(p.soar), float(p.slender), float(p.crag)
    RS = min(1.0, 0.48 + 0.62 * S)        # girth of the mural towers follows the plan scale a little
    B = {k: _MB(bool(p.lod_pins), weld=(k == "wood")) for k in KEYS}
    az, el = radians(float(p.moon_az)), radians(38.0)
    Lm = Vector((sin(az) * cos(el), -cos(az) * cos(el), sin(el)))
    VF = float(p.valley_fog)
    na = radians(float(p.near_az))
    NEAR = Vector((sin(na) * 138.0, -cos(na) * 138.0, Hh + 75.0))      # the near viewpoint: 150 m off, looking down
    XF, TB, TW = [None], [0.0], [0]
    ZN, TG, GB, SZ = [1], [0.0], [Hh], [None]     # tone zone, tonal drift (grime), local ground, wall-head line
    CT = [0]                                      # tone step of humble buildings (cottages sit a step below the castle)
    LR = float(p.lit_ratio)
    rw = rng.fork("windows")
    feat = p.feature if p.feature != "auto" else {"castle": "clock", "cathedral": "rose"}.get(p.style, "none")

    MK = [0, []]
    CNT = [0, 0]
    PS, PN, REC = [0], [0], []      # stack of the part being built, parts so far, the masonry / roof faces with their part

    def pb(new=False):
        """Begin a part: a form that stands on its own (tower, block, stack ...) and may be shifted as a whole by
        settle(). Inside another part it stays with that one unless new."""
        if new or PS[-1] == 0:
            PN[0] += 1
            PS.append(PN[0])
        else:
            PS.append(PS[-1])

    def pe():
        PS.pop()

    def tris():
        return sum(mb.nt for mb in B.values())

    def mark(name):
        t = tris()
        MK[1].append(f"{name}={t - MK[0]}")
        MK[0] = t

    def V3(x, y, z):
        t = XF[0]
        if t:
            x, y = t[0] + x * t[2] - y * t[3], t[1] + x * t[3] + y * t[2]
        return Vector((x, y, z))

    def hsh(q, k=0.0):
        v = sin(q.x * 12.9898 + q.y * 78.233 + q.z * 37.719 + k) * 43758.5453
        return v - floor(v)

    def deep(z):
        """0 down to eight metres below the rim of the crag .. 1 from twenty metres below it: the woodland is a
        little paler down the flanks, where the night haze of the valley takes most of its tone away again."""
        return max(0.0, min(1.0, (Hh - 8.0 - z) / 12.0))

    def lift(e):
        """Grime that leaves the fraction e (0.16 .. 1) of a stone's glow: the grime of the library stone is a
        noise threshold, so the steps between mottle the rock."""
        e = max(0.16, min(1.0, e))
        if e > 0.9:
            return 0.3 + (1.0 - e)
        if e > 0.24:
            return 0.4 + 0.2 * (0.9 - e) / 0.66
        return 0.6 + 0.2 * (0.24 - e) / 0.08

    def F(key, pts, g=0.0, gv=None):
        """One face, counter-clockwise seen from outside. "W" / "R" / "K" / "T" pick the stone / slate / rock /
        canopy tone from the moon; "W+" is a step darker (corbel bands, string courses, under eaves), "Wi" the
        unlit inside of a parapet. Every vertex gets a grime value: tonal drift of the tower, damp towards the ground,
        streaks under wall heads, a little noise - the masonry is stained, not paper."""
        k = len(pts)
        n = Vector((0.0, 0.0, 0.0))
        for i in range(k):
            a, b = pts[i], pts[(i + 1) % k]
            n += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y)))
        if n.length < 1e-6:
            return
        n.normalize()
        if key[0] in "WR" or key in ("lit", "dim", "cool", "dk", "dw"):
            REC.append((PS[-1], [q.copy() for q in pts], n.copy()))
        if key[0] == "W":
            band = key == "W+"
            if band:        # bands, corbels, soffits: the tone of the wall they belong to (its azimuth), one step down
                hl = hypot(n.x, n.y)
                t = ((n.x * Lm.x + n.y * Lm.y) / hl if hl > 0.35 else (0.3 if n.z > 0.0 else -1.0)) + TB[0]
            else:
                # wall heads, walks and the inner faces of merlons carry no baked light
                t = (n.dot(Lm) if (n.z < 0.6 and key != "Wi") else -1.0) + TB[0]
            i = (0 if t > 0.5 else (1 if t > 0.02 else 2)) + CT[0]
            if band:
                i += 1
            key, gx = LAD[max(0, min(4, i))]
            g += gx + TG[0] + 0.07 * ZN[0]
            G = []
            for q in pts:
                if band:        # one even step for the whole ring
                    G.append(min(1.0, g + 0.1))
                    continue
                v = g + 0.08 * hsh(q) + 0.3 * max(0.0, min(1.0, 1.0 - (q.z - GB[0]) / 5.0)) * (0.5 + hsh(q, 1.7))
                if SZ[0] is not None and abs(q.z - SZ[0]) < 0.02:
                    v += 0.04 + 0.26 * hsh(q, 3.1) ** 2
                G.append(min(1.0, v))
        elif key == "R":
            g += 0.0 if n.dot(Lm) > 0.3 else 0.45
            key, G = "roof", [min(1.0, g + 0.12 * hsh(q)) for q in pts]
        elif key == "K":                # rock: moonlit faces, flanks and shade, so ribs, ledges and gullies read
            hm = (n.x * Lm.x + n.y * Lm.y) / cos(el)
            c = sum(pts, Vector((0.0, 0.0, 0.0))) / k
            dv = NEAR - c
            if VF > 0.0 and hm > 0.3 and n.z < 0.75 and c.z < Hh - 4.0 and n.dot(dv) < -0.2 * dv.length:
                # moonlit cliff faces the near view cannot see: a tone that holds against the ground fog - it
                # doubles every FOG_HALF metres down from the rim, as the fog does, and is full FOG_FULL below it
                key, G = "rk", [lift(2.0 ** ((Hh - FOG_FULL - q.z) / FOG_HALF)) + 0.04 * hsh(q) for q in pts]
            else:
                t = n.dot(Lm) if n.z < 0.75 else -1.0      # ledges and the rim are seen from above only
                key, G = "rs", [min(1.0, g + (0.1 if t > 0.22 else (0.48 if t > -0.2 else 0.72)) + 0.1 * hsh(q)) for q in pts]
        elif key == "T" and gv is not None:     # canopy: the tone follows the rounded form, not the facet
            key, G = "wood", [max(0.0, v - 0.18 * deep(q.z)) for v, q in zip(gv, pts)]
        elif key == "T":                # woodland floor and grass
            t = n.dot(Lm)
            g += 0.5 if t > 0.45 else (0.6 if t > 0.0 else 0.75)
            key, G = "wood", [min(1.0, g - 0.2 * deep(q.z) + 0.1 * hsh(q)) for q in pts]
        elif key in ("dk", "dw"):       # dark openings, door, hands of the clock
            key, G = "rs", [1.0] * k
        elif key == "P":                # trodden ground of the wards
            key, G = "rs", [0.7 + 0.25 * hsh(q) for q in pts]
        elif key == "L":                # lane and yards
            key, G = "w1", [0.45 + 0.3 * hsh(q) for q in pts]
        else:
            G = [0.0] * k
        B[key].face(pts, n, G, PS[-1])

    def settle():
        """CLEARANCE RULE: no two faces of different forms may lie in one plane. The forms of the summit are grown
        into each other wherever the seed puts them, and now and then a tower face comes to lie within a
        millimetre of a wall it stands in (both glow, every material is double sided: the two fight at any
        distance). Wherever two parts have parallel faces closer than GAP that overlap, the later part is shifted
        as a whole until its face stands SEP behind the other one. Crossing faces are left alone."""
        from mathutils.bvhtree import BVHTree
        GAP, SEP = 0.006, 0.03
        for _ in range(4):
            V, Fx = [], []
            for pid, pts, n in REC:
                Fx.append(list(range(len(V), len(V) + len(pts))))
                V += pts
            if not Fx:
                return
            bvh = BVHTree.FromPolygons(V, Fx, all_triangles=False, epsilon=0.0)
            mv = {}
            for i, (pid, pts, n) in enumerate(REC):
                c = sum(pts, Vector((0.0, 0.0, 0.0))) / len(pts)
                for q in [c] + [c * (1.0 - f) + v * f for v in pts for f in (0.35, 0.7)]:
                    t0 = -GAP
                    for _k in range(8):
                        hit = bvh.ray_cast(q + n * t0, n, GAP - t0)
                        if hit[0] is None:
                            break
                        j, t = hit[2], t0 + hit[3]
                        t0 = t + 1e-5
                        pj, nj = REC[j][0], REC[j][2]
                        if j == i or pj == pid or abs(nj.dot(n)) < 0.9994:
                            continue
                        if pj > pid:        # j's face stands t in front of i's: its part goes SEP behind i's face
                            mv.setdefault(pj, n * (-SEP - t))
                        else:               # i's face stands -t along n off j's: seen along j's normal ...
                            ti = -t if nj.dot(n) > 0.0 else t
                            mv.setdefault(pid, nj * (-SEP - ti))
            if not mv:
                return
            for pid, pts, n in REC:
                if pid in mv:
                    for q in pts:
                        q += mv[pid]
            for mb in B.values():
                for iv, pv in enumerate(mb.p):
                    if pv in mv:
                        x, y, z = mb.v[iv]
                        d = mv[pv]
                        mb.v[iv] = (x + d.x, y + d.y, z + d.z)

    def wk(boost=1.0, bright=False):
        if rw.u(0.0, 1.0) < min(0.95, LR * boost):
            return "lit" if (bright or rw.u(0.0, 1.0) < 0.55) else "dim"
        return "dw"

    # ------------------------------------------------------------------ walls, windows, dial
    def disc_cell(P, u0, u1, za, zb, spec):
        """A square wall cell holding a round dial / rose: wall corners, dark rim, glowing disc."""
        uc, zc = (u0 + u1) / 2, (za + zb) / 2
        R, SEG = 0.92 * min((u1 - u0) / 2, (zb - za) / 2), 12

        def pt(k, f=1.0):
            an = k * 2 * pi / SEG
            return P(uc + f * R * cos(an), zc + f * R * sin(an))
        for c, k0 in ((P(u0, za), 9), (P(u1, za), 12), (P(u1, zb), 3), (P(u0, zb), 6)):
            F("W", [c] + [pt(k0 - q) for q in range(4)])
        F("W", [P(u0, za), P(u1, za), pt(9)])
        F("W", [P(u1, zb), P(u0, zb), pt(3)])
        F("W", [P(u1, za), P(u1, zb), pt(0)])
        F("W", [P(u0, zb), P(u0, za), pt(6)])
        fi = 0.9
        for k in range(SEG):        # the moulded stone surround
            F("W+", [pt(k, fi), pt(k, 1.0), pt(k + 1, 1.0), pt(k + 1, fi)], 0.2)
        if spec == "rose":
            fc = 0.42
            for k in range(SEG):
                F("lit" if k % 2 == 0 else "dim", [pt(k, fc), pt(k, fi), pt(k + 1, fi), pt(k + 1, fc)])
            F("cool", [pt(k, fc) for k in range(SEG)])
        else:       # clock: dark iron chapter ring, glowing dial, bold hands and quarter batons proud of it
            fr = 0.79
            for k in range(SEG):
                F("dk", [pt(k, fr), pt(k, fi), pt(k + 1, fi), pt(k + 1, fr)])
            F("cool", [pt(k, fr) for k in range(SEG)])
            o = P(uc, zc)
            ex = (P(uc + 1.0, zc) - o).normalized()
            nn = ex.cross(UP)
            marks = [(radians(305.0), -0.12, 0.47, 0.085, 0.5), (radians(48.0), -0.12, 0.7, 0.06, 0.5)]
            marks += [(k * pi / 2, 0.56, 0.74, 0.05, 1.0) for k in range(4)]
            for ang, l0, l1, hw_, tp in marks:
                d, q = ex * sin(ang) + UP * cos(ang), ex * cos(ang) - UP * sin(ang)
                b0, tip = o + nn * 0.25 + d * (l0 * R), o + nn * 0.25 + d * (l1 * R)
                F("dk", [b0 - q * (hw_ * R), b0 + q * (hw_ * R), tip + q * (hw_ * R * tp), tip - q * (hw_ * R * tp)])

    def wall(a, b, zb, zt, cols=(), wkey="W", streak=True):
        """wall_ with the rain streaks of a wall head (streak=False for a storey that ends in a string course)."""
        SZ[0] = zt if streak else None
        r = wall_(a, b, zb, zt, cols, wkey)
        SZ[0] = None
        return r

    def wall_(a, b, zb, zt, cols=(), wkey="W"):
        """Vertical wall from a to b (xy), outside on the right of a->b. cols = window columns (u0, u1, cells) in
        metres along the wall, cells = [(za, zb, key)] bottom-up; key = tone key, ("arch", key) pointed head or
        ("disc", spec). Every column is a stack of cells; the piers between columns are single polygons that
        carry the column's vertices on one edge - no T-junctions, and a blind wall is one quad.
        Returns the bottom and top vertex chains."""
        ax, ay = a
        bx, by = b
        Lw = hypot(bx - ax, by - ay)

        def P(u, z):
            f = u / Lw
            return V3(ax + (bx - ax) * f, ay + (by - ay) * f, z)
        cols = sorted((c for c in cols if c[2]), key=lambda c: c[0])
        if not cols:
            F(wkey, [P(0.0, zb), P(Lw, zb), P(Lw, zt), P(0.0, zt)])
            return [P(0.0, zb), P(Lw, zb)], [P(0.0, zt), P(Lw, zt)]
        bounds = [0.0] + [(cols[i][1] + cols[i + 1][0]) / 2 for i in range(len(cols) - 1)] + [Lw]
        us = [0.0]
        for ci, (u0, u1, cells) in enumerate(cols):
            zl, z = [zb], zb
            for za, zc, key in sorted(cells, key=lambda c: c[0]):
                za, zc = max(za, z), min(zc, zt)
                if zc - za < 0.05:
                    continue
                if za - z > 0.02:
                    F(wkey, [P(u0, z), P(u1, z), P(u1, za), P(u0, za)])
                    zl.append(za)
                else:
                    za = z
                if isinstance(key, tuple) and key[0] == "arch":
                    p00, p10, p11, p01 = P(u0, za), P(u1, za), P(u1, zc), P(u0, zc)
                    apex = P((u0 + u1) / 2, zc - 0.08 * (zc - za))
                    F(key[1], [p00, p10, apex])
                    F(wkey, [p00, apex, p01])
                    F(wkey, [p10, p11, apex])
                    F(wkey, [p01, apex, p11])
                elif isinstance(key, tuple):
                    disc_cell(P, u0, u1, za, zc, key[1])
                else:
                    F(key, [P(u0, za), P(u1, za), P(u1, zc), P(u0, zc)])
                zl.append(zc)
                z = zc
            if zt - z > 0.02:
                F(wkey, [P(u0, z), P(u1, z), P(u1, zt), P(u0, zt)])
                zl.append(zt)
            else:
                zl[-1] = zt
            ul, ur = bounds[ci], bounds[ci + 1]
            if u0 - ul > 0.02:
                F(wkey, [P(ul, zb)] + [P(u0, zz) for zz in zl] + [P(ul, zt)])
                us.append(u0)
            if ur - u1 > 0.02:
                F(wkey, [P(ur, zt)] + [P(u1, zz) for zz in reversed(zl)] + [P(ur, zb)])
                us.append(u1)
            us.append(ur)
        us = sorted(set(round(u, 4) for u in us))
        return [P(u, zb) for u in us], [P(u, zt) for u in us]

    def stitch(Bc, Tc, key="W"):
        """Band between two vertex chains (both left to right seen from outside); Tc may be a single apex."""
        if len(Bc) == 2 and len(Tc) == 2:
            F(key, [Bc[0], Bc[1], Tc[1], Tc[0]])
            return
        if len(Tc) == 1:        # a gable: one polygon fanned from the apex
            F(key, [Tc[0]] + list(Bc))
            return
        if len(Tc) == 2:        # straight chain to a straight edge: one planar polygon fanned from the far edge
            F(key, [Tc[1], Tc[0]] + list(Bc))
            return
        if len(Bc) == 2:
            F(key, [Bc[0], Bc[1]] + list(Tc)[::-1])
            return

        def par(C):
            d = [0.0]
            for i in range(1, len(C)):
                d.append(d[-1] + (C[i] - C[i - 1]).length)
            return [x / max(d[-1], 1e-9) for x in d]
        pb, pt_ = par(Bc), par(Tc)
        i = j = 0
        while i < len(Bc) - 1 or j < len(Tc) - 1:
            if i < len(Bc) - 1 and (j == len(Tc) - 1 or pb[i + 1] <= pt_[j + 1] + 1e-9):
                F(key, [Bc[i], Bc[i + 1], Tc[j]])
                i += 1
            else:
                F(key, [Bc[i], Tc[j + 1], Tc[j]])
                j += 1

    def wall2(a, b, zb, zm, zt, cols_lo, cols_hi):
        """Two wall storeys with their own window columns, joined by a string-course band. Returns the top chain."""
        _, t1 = wall(a, b, zb, zm, cols_lo, streak=False)
        b2, t2 = wall(a, b, zm + 0.45, zt, cols_hi)
        stitch(t1, b2, "W+")
        return t2

    def wcols(Lw, n, ww, lev, boost=1.0, keep=1.0, margin=0.0, bright=False):
        """n evenly spaced window columns; lev = [(z, body height, pointed head height[, forced tone])]; every
        window is lit, dim or dark on its own, keep < 1 leaves some out."""
        if n <= 0 or not lev:
            return []
        pitch = (Lw - 2 * margin) / n
        ww = min(ww, 0.62 * pitch)
        out = []
        for k in range(n):
            uc, cells = margin + (k + 0.5) * pitch, []
            for lv in lev:
                z, hb, hh = lv[0], lv[1], lv[2]
                if rw.u(0.0, 1.0) > keep:
                    continue
                kind = lv[3] if len(lv) > 3 else wk(boost, bright)
                cells.append((z, z + hb, kind))
                if hh > 0:
                    cells.append((z + hb, z + hb + hh, ("arch", kind)))
            out.append((uc - ww / 2, uc + ww / 2, cells))
        return out

    def crenel_us(Lw, mw=2.0, gap=1.5):
        """Column lines of a battlement: merlon, gap, merlon ... merlon."""
        ng = max(1, int(round((Lw - mw) / (mw + gap))))
        f = Lw / ((ng + 1) * mw + ng * gap)
        us, u = [0.0], 0.0
        for k in range(2 * ng + 1):
            u += (mw if k % 2 == 0 else gap) * f
            us.append(u)
        us[-1] = Lw
        return us

    def battlement(a, b, z0, zp, zm, mw=2.0, gap=1.5, key="W"):
        """Wall / parapet z0..zp on the line a->b crowned by merlons to zm (merlons are two-sided, their inner face in shade).
        CLEARANCE RULE: no two sheets in one place (every material is double sided, coincident sheets fight each
        other at any distance): the shaded inner face of a merlon stands MI behind the outer one, and stops MI
        short of the ends of the wall so that the inner faces of two sides do not cross at a corner."""
        MI = 0.08
        ax, ay = a
        bx, by = b
        Lw = hypot(bx - ax, by - ay)
        if Lw < 2 * mw + gap:       # a short face (round towers) carries one merlon, the crenels sit on the angles
            us, m0 = [0.0, 0.2 * Lw, 0.8 * Lw, Lw], 1
        else:
            us, m0 = crenel_us(Lw, mw, gap), 0

        def P(u, z):
            f = u / Lw
            return V3(ax + (bx - ax) * f, ay + (by - ay) * f, z)

        def Pi(u, z):
            f = u / Lw
            return V3(ax + (bx - ax) * f - (by - ay) / Lw * MI, ay + (by - ay) * f + (bx - ax) / Lw * MI, z)
        F(key, [P(0.0, z0), P(Lw, z0)] + [P(u, zp) for u in reversed(us)])
        for ci in range(m0, len(us) - 1, 2):
            u0, u1 = us[ci], us[ci + 1]
            F("W", [P(u0, zp), P(u1, zp), P(u1, zm), P(u0, zm)])
            v0, v1 = max(u0, MI), min(u1, Lw - MI)
            F("Wi", [Pi(v1, zp), Pi(v0, zp), Pi(v0, zm), Pi(v1, zm)])

    def box(x0, x1, y0, y1, z0, z1, key="W", top=True):
        pb()
        F(key, [V3(x0, y0, z0), V3(x1, y0, z0), V3(x1, y0, z1), V3(x0, y0, z1)])
        F(key, [V3(x1, y0, z0), V3(x1, y1, z0), V3(x1, y1, z1), V3(x1, y0, z1)])
        F(key, [V3(x1, y1, z0), V3(x0, y1, z0), V3(x0, y1, z1), V3(x1, y1, z1)])
        F(key, [V3(x0, y1, z0), V3(x0, y0, z0), V3(x0, y0, z1), V3(x0, y1, z1)])
        if top:
            F(key, [V3(x0, y0, z1), V3(x1, y0, z1), V3(x1, y1, z1), V3(x0, y1, z1)])
        pe()

    def buttress(x, y, d, hw_, zb, zt, key="W"):
        """Buttress against a wall that faces -Y at (x, y): d deep, its weathered head sloping back to the wall."""
        F(key, [V3(x - hw_, y - d, zb), V3(x + hw_, y - d, zb), V3(x + hw_, y - d, zt), V3(x - hw_, y - d, zt)])
        F(key, [V3(x + hw_, y - d, zb), V3(x + hw_, y, zb), V3(x + hw_, y, zt + 1.3), V3(x + hw_, y - d, zt)])
        F(key, [V3(x - hw_, y, zb), V3(x - hw_, y - d, zb), V3(x - hw_, y - d, zt), V3(x - hw_, y, zt + 1.3)])
        F("W", [V3(x - hw_, y - d, zt), V3(x + hw_, y - d, zt), V3(x + hw_, y, zt + 1.3), V3(x - hw_, y, zt + 1.3)])

    def Fc(key, pts, c, g=0.0):
        """Face wound so that it looks away from the inside point c."""
        m = Vector((0.0, 0.0, 0.0))
        for q in pts:
            m += q
        nn = (pts[1] - pts[0]).cross(pts[2] - pts[0])
        F(key, pts if nn.dot(m / len(pts) - c) >= 0.0 else pts[::-1], g)

    def dormer(cx, cy, ox, oy, rho, zL, w, h, g, key, u0=0.0):
        """Lucarne on a roof slope: glowing front, little gable and saddle roof running back into the slope.
        (cx, cy) = point on the roof axis, (ox, oy) = outward direction, rho(z) = distance of the slope from it."""
        tx, ty = -oy, ox

        def P(u, r_, z):
            return V3(cx + ox * r_ + tx * (u0 + u), cy + oy * r_ + ty * (u0 + u), z)
        rf, rb, rr_ = rho(zL) + 0.2, rho(zL + h) - 0.1, rho(zL + h + g) - 0.1
        hw_ = w / 2
        F(key, [P(-hw_, rf, zL), P(hw_, rf, zL), P(hw_, rf, zL + h), P(-hw_, rf, zL + h)])
        F("W", [P(-hw_ - 0.15, rf, zL + h), P(hw_ + 0.15, rf, zL + h), P(0.0, rf, zL + h + g)])
        F("W+", [P(hw_, rf, zL), P(hw_, rb, zL + h), P(hw_, rf, zL + h)])
        F("W+", [P(-hw_, rf, zL), P(-hw_, rf, zL + h), P(-hw_, rb, zL + h)])
        F("R", [P(hw_ + 0.15, rf, zL + h), P(hw_ + 0.15, rb, zL + h), P(0.0, rr_, zL + h + g), P(0.0, rf, zL + h + g)])
        F("R", [P(-hw_ - 0.15, rf, zL + h), P(0.0, rf, zL + h + g), P(0.0, rr_, zL + h + g), P(-hw_ - 0.15, rb, zL + h)])

    def finial(x, y, z, s=1.0):
        """Knop and needle on a roof apex."""
        rg = [V3(x + 0.42 * s * cos(2 * pi * i / 3 + 0.5), y + 0.42 * s * sin(2 * pi * i / 3 + 0.5), z - 0.5 * s) for i in range(3)]
        hi = V3(x, y, z + 3.4 * s)
        for i in range(3):
            F("R", [rg[i], rg[(i + 1) % 3], hi])

    # ------------------------------------------------------------------ towers and blocks
    def tower(cx, cy, r, zb, zf, zt, **kw):
        """Polygonal tower with its own slight tonal drift; see _tower."""
        TW[0] += 1
        old = TB[0], TG[0]
        rt_ = rng.fork(f"tone{TW[0]}")
        TB[0], TG[0] = rt_.u(-0.14, 0.14), rt_.u(0.0, 0.3)
        pb(True)
        z = _tower(cx, cy, r, zb, zf, zt, **kw)
        pe()
        TB[0], TG[0] = old
        return z

    def _tower(cx, cy, r, zb, zf, zt, n=6, top="cone", sh=None, maxlev=3, corbel=False, lev=None, wf=0.62, boost=1.0, pw=0.55,
               keep=0.7, rot=0.0, belfry=0.0, luc=0, fin=False, course=None, pl=0.0):
        """Shaft zb..zt with window slits from zf up (belfry = height of an open bell stage under the head), a
        thicker basement storey under a weathered string course (course), arrow loops in blind faces (pl = how many of them), a
        flared head, then a bellcast spire ("cone", with luc lucarnes and a finial), a crenellated wall-walk
        ("batt"), both ("batt+cone") or a saddle roof (square towers). Returns the top z."""
        rr = rng.fork(f"tower{TW[0]}")
        if DET >= 1:
            maxlev, pl = max(min(1, maxlev), maxlev - DET), pl * max(0.0, 1.0 - 0.4 * DET)
        if DET >= 2 and n > 6:
            n = max(6 if DET >= 3 else 8, n - 2 * (DET - 1))
        a0 = -pi / 2 - pi / n + rot

        def xy(rad):
            return [(cx + rad * cos(a0 + 2 * pi * i / n), cy + rad * sin(a0 + 2 * pi * i / n)) for i in range(n)]

        def ring(rad, z):
            return [V3(x, y, z) for x, y in xy(rad)]
        ztw = zt - (belfry + 2.6 if belfry else 1.2)
        if lev is None:
            lev, z = [], zf + rr.u(2.5, 5.0)
            while z + 2.8 < ztw and len(lev) < maxlev:
                lev.append((z, 2.8, 0.0))
                z += rr.u(5.5, 8.5)
        c0, cb = xy(r), xy(r + 0.22)
        Ls = hypot(c0[1][0] - c0[0][0], c0[1][1] - c0[0][1])
        ww = min(1.7, wf * Ls)
        bots, zs = [], None
        if course is None:
            course = not corbel and r >= 1.9 and n >= 6
        if course:
            zs = min([zf + 0.3 * (zt - zf)] + [lv[0] - 1.0 for lv in lev])
            if zs - zf < 1.6 or zt - zs < 4.5:
                zs = None
        rl = rng.fork(f"loops{TW[0]}")
        zl0, zl1 = (zf + 2.5 if zs is None else zs + 1.2), ztw - 3.6
        for i in range(n):
            cols = wcols(Ls, 1, ww, lev, boost, keep) if (lev and rr.u(0.0, 1.0) < pw) else []
            if belfry:
                cell = (zt - belfry - 1.1, zt - 1.1, wk(boost * 1.8))
                if cols:
                    cols[0][2].append(cell)
                else:
                    cols = [(Ls / 2 - ww / 2, Ls / 2 + ww / 2, [cell])]
            if not cols and zl1 > zl0 and rl.u(0.0, 1.0) < pl:       # an arrow loop
                zl, lw_ = rl.u(zl0, zl1), min(0.5, 0.3 * Ls)
                cols = [(Ls / 2 - lw_ / 2, Ls / 2 + lw_ / 2, [(zl, zl + 2.3, "dw")])]
            a_, b_ = c0[i], c0[(i + 1) % n]
            if zs is None:
                bots.append(wall(a_, b_, zb, zt, cols)[0])
            else:
                bc, t1 = wall(cb[i], cb[(i + 1) % n], zb, zs, (), streak=False)
                stitch(t1, wall(a_, b_, zs + 0.4, zt, cols)[0], "W+")
                bots.append(bc)
        if corbel:
            ap = V3(cx, cy, zb - 1.7 * r)
            for bc in bots:
                for q in range(len(bc) - 1):
                    F("W+", [bc[q + 1], bc[q], ap])
        W0 = ring(r - 0.03, zt - 0.7)
        if top.startswith("batt"):
            rc = r * 1.14 + 0.3
            c1, R1 = xy(rc), ring(rc, zt + 1.0)
            for i in range(n):
                j = (i + 1) % n
                F("W+", [W0[i], W0[j], R1[j], R1[i]])
                battlement(c1[i], c1[j], zt + 0.8, zt + 2.0, zt + 3.3, 1.6, 1.2)
            F("W+", ring(rc, zt + 1.3))
            zr, rr_ = zt + 1.3, rc * 0.8
            if top == "batt":
                return zt + 3.3
        else:
            zr, rr_ = zt + 0.5, r * 1.1 + 0.5
            R1 = ring(rr_, zr)
            for i in range(n):
                j = (i + 1) % n
                F("W+", [W0[i], W0[j], R1[j], R1[i]])
        if top == "saddle" and n == 4:
            hs = 1.5 * r if sh is None else sh
            m12, m30 = (R1[1] + R1[2]) * 0.5 + UP * hs, (R1[3] + R1[0]) * 0.5 + UP * hs
            F("R", [R1[0], R1[1], m12, m30])
            F("R", [R1[2], R1[3], m30, m12])
            F("W", [R1[1], R1[2], m12])
            F("W", [R1[3], R1[0], m30])
            return zr + hs
        sh = r * 4.0 if sh is None else sh
        h1 = 0.26 * sh
        A, M, ap = ring(rr_, zr), ring(rr_ * 0.56, zr + h1), V3(cx, cy, zr + sh)
        for i in range(n):
            j = (i + 1) % n
            if r < 1.3:         # turrets and pinnacles: a plain cone
                F("R", [A[i], A[j], ap])
                continue
            F("R", [A[i], A[j], M[j], M[i]])
            F("R", [M[i], M[j], ap])
        if luc and h1 > 2.4:
            ca = cos(pi / n)

            def rho(z):
                return ca * rr_ * (1.0 - 0.44 * min(1.0, (z - zr) / h1))
            lw_ = min(1.6, 1.5 * rr_ * 0.6 * sin(pi / n) + 0.3)
            lh = min(2.0, 0.42 * h1)
            for k in range(luc):
                an = a0 + 2 * pi * ((k * n) // luc + 0.5) / n
                dormer(cx, cy, cos(an), sin(an), rho, zr + 0.1 * h1, lw_, lh, 0.7 * lw_, wk(3.0))
        if fin:
            finial(cx, cy, zr + sh, min(1.2, 0.5 + 0.03 * sh))
        return zr + sh

    def block(*a, **kw):
        """Rectangular building, a part of its own unless it belongs to one already; see _block."""
        pb()
        z = _block(*a, **kw)
        pe()
        return z

    def _block(x0, x1, y0, y1, zb, zt, cols=None, roof=None, rh=0.0, ov=0.5, top=None, skip="", luc=None, fin=False, base=None,
               ld=(1.7, 1.9, 1.2), lb=3.0, lbright=False):
        """Rectangular building; cols = {"f" / "r" / "b" / "l": window columns}. Roof: gable along X ("gx") or Y
        ("gy"), pyramid ("pyr"); top="batt" adds a corbelled crenellated parapet (combinable with "pyr").
        luc = number of lucarnes on the front slope ("gx") or the sides that get one ("pyr", e.g. "fr"), ld their
        width, height and gable; base = z of the weathered head of a plinth that stands 0.25 m proud (blind)."""
        cs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        cols = cols or {}
        ch = {}
        for si, side in enumerate("frbl"):
            a, b = cs[si], cs[(si + 1) % 4]
            if side in skip:
                ch[side] = [V3(*a, zt), V3(*b, zt)]
                continue
            if base is not None and base - zb > 1.0 and zt - base > 3.0:
                tx_, ty_ = (b[0] - a[0]) / hypot(b[0] - a[0], b[1] - a[1]), (b[1] - a[1]) / hypot(b[0] - a[0], b[1] - a[1])
                o_ = 0.25
                # CLEARANCE RULE: the plinth runs o_ past a corner, but not past a side that is left out (there a twin
                # block carries on in the same plane, and the two plinths would lie in each other as two sheets)
                ea = 0.0 if "frbl"[si - 1] in skip else o_
                eb = 0.0 if "frbl"[(si + 1) % 4] in skip else o_
                t1 = wall((a[0] + o_ * ty_ - ea * tx_, a[1] - o_ * tx_ - ea * ty_), (b[0] + o_ * ty_ + eb * tx_, b[1] - o_ * tx_ + eb * ty_),
                          zb, base - 0.4, (), streak=False)[1]
                b2, ch[side] = wall(a, b, base, zt, cols.get(side, ()))
                stitch(t1, b2, "W+")
            else:
                ch[side] = wall(a, b, zb, zt, cols.get(side, ()))[1]
        zc = zt
        if top == "batt":
            o, e = 0.5, 0.03
            ci_ = [(x0 + e, y0 + e), (x1 - e, y0 + e), (x1 - e, y1 - e), (x0 + e, y1 - e)]
            co = [(x0 - o, y0 - o), (x1 + o, y0 - o), (x1 + o, y1 + o), (x0 - o, y1 + o)]
            for si in range(4):
                sj = (si + 1) % 4
                F("W+", [V3(*ci_[si], zt - 0.8), V3(*ci_[sj], zt - 0.8), V3(*co[sj], zt + 0.9), V3(*co[si], zt + 0.9)])
                battlement(co[si], co[sj], zt + 0.7, zt + 1.9, zt + 3.3, 2.2, 1.6)
            F("W+", [V3(x, y, zt + 1.2) for x, y in co])
            zc = zt + 1.2
            x0, x1, y0, y1, ov = x0 + 0.5, x1 - 0.5, y0 + 0.5, y1 - 0.5, 0.0
        zr = zc + rh
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        if roof == "gx":
            dz = ov * rh / ((y1 - y0) / 2)
            F("R", [V3(x0 - ov, y0 - ov, zc - dz), V3(x1 + ov, y0 - ov, zc - dz), V3(x1 + ov, ym, zr), V3(x0 - ov, ym, zr)])
            F("R", [V3(x1 + ov, y1 + ov, zc - dz), V3(x0 - ov, y1 + ov, zc - dz), V3(x0 - ov, ym, zr), V3(x1 + ov, ym, zr)])
            if "r" not in skip:
                stitch(ch["r"], [V3(x1, ym, zr)])
            if "l" not in skip:
                stitch(ch["l"], [V3(x0, ym, zr)])
            for k in range(int(luc or 0)):
                def rho(z):
                    return (ym - y0 + ov) * (zr - z) / (rh + dz)
                dormer(x0 + (k + 0.5) * (x1 - x0) / luc, ym, 0.0, -1.0, rho, zc + 0.16 * rh, ld[0], ld[1], ld[2], wk(lb, lbright))
        elif roof == "gy":
            dz = ov * rh / ((x1 - x0) / 2)
            F("R", [V3(x0 - ov, y1 + ov, zc - dz), V3(x0 - ov, y0 - ov, zc - dz), V3(xm, y0 - ov, zr), V3(xm, y1 + ov, zr)])
            F("R", [V3(x1 + ov, y0 - ov, zc - dz), V3(x1 + ov, y1 + ov, zc - dz), V3(xm, y1 + ov, zr), V3(xm, y0 - ov, zr)])
            if "f" not in skip:
                stitch(ch["f"], [V3(xm, y0, zr)])
            if "b" not in skip:
                stitch(ch["b"], [V3(xm, y1, zr)])
        elif roof == "pyr":
            cr = [V3(x0 - ov, y0 - ov, zc), V3(x1 + ov, y0 - ov, zc), V3(x1 + ov, y1 + ov, zc), V3(x0 - ov, y1 + ov, zc)]
            ap = V3(xm, ym, zr)
            for i in range(4):
                F("R", [cr[i], cr[(i + 1) % 4], ap])
            for side in (luc or ""):
                ox, oy = {"f": (0.0, -1.0), "r": (1.0, 0.0), "b": (0.0, 1.0), "l": (-1.0, 0.0)}[side]
                half = ((y1 - y0) if side in "fb" else (x1 - x0)) / 2 + ov

                def rho(z, half=half):
                    return half * (zr - z) / rh
                dormer(xm, ym, ox, oy, rho, zc + 0.14 * rh, 1.8, 2.1, 1.3, wk(3.0))
            if fin:
                finial(xm, ym, zr, 1.2)
        return zr

    def spire_on(x0, x1, y0, y1, zt, sh, pin=4.5):
        """Octagonal spire with lucarnes, a finial and four corner pinnacles on a square tower head."""
        cx, cy, hw_ = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) / 2
        tower(cx, cy, hw_ * 0.84, zt - 0.6, zt, zt + 0.9, n=8, sh=sh, maxlev=0, luc=4, fin=True)
        for sx in (-1, 1):
            for sy in (-1, 1):
                tower(cx + sx * (hw_ - 0.5), cy + sy * (hw_ - 0.5), 0.7, zt - 0.4, zt, zt + 1.8, n=4, sh=pin, maxlev=0)

    def bridge(ax, ay, bx, by, z, wd=1.8):
        """Flying bridge: a walled deck on a segmental arch between two towers."""
        dx, dy = bx - ax, by - ay
        L = hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        rise, K = min(3.2, 0.24 * L), 4

        def P(f, sd, zz):
            return V3(ax + dx * f + nx * sd * wd / 2, ay + dy * f + ny * sd * wd / 2, zz)

        def zu(f):
            return z - 0.7 - rise * (2 * f - 1) ** 2
        zt_ = z + 1.1
        for q in range(K):
            f0, f1 = q / K, (q + 1) / K
            F("W", [P(f1, 1, zu(f1)), P(f0, 1, zu(f0)), P(f0, 1, zt_), P(f1, 1, zt_)])
            F("W", [P(f0, -1, zu(f0)), P(f1, -1, zu(f1)), P(f1, -1, zt_), P(f0, -1, zt_)])
            F("W+", [P(f0, -1, zu(f0)), P(f0, 1, zu(f0)), P(f1, 1, zu(f1)), P(f1, -1, zu(f1))])
        F("W+", [P(0, -1, zt_), P(1, -1, zt_), P(1, 1, zt_), P(0, 1, zt_)])

    # ------------------------------------------------------------------ the crag
    foot = {"castle": (30.0, 19.0), "cathedral": (33.0, 16.0), "citadel": (35.0, 22.5), "ridge": (41.0, 8.0)}[p.style]
    ap_, bp_ = foot[0] * S + 4.5, foot[1] * S + 4.5
    RIDGE = p.style == "ridge"
    if RIDGE:       # the crest of the hogback runs along most of the hill
        ap_ = max(ap_, 0.34 * float(p.hill_width))
    thg, SC = -pi / 2, SL[1]
    z0 = Hh
    NT = int(p.trees) if Hh > 4.0 else 0

    def frun(th):
        """Least run of the flank below the plateau edge; a ridge is a hogback: long gentle flanks, shorter ends."""
        return (0.6 * Hh + 3.0) * (0.6 + 0.4 * sin(th) ** 2) if RIDGE else 0.3 * Hh + 5.0

    def crest(x):
        """Height factor along the hill: the crest of a ridge sinks from the chapel towards both ends."""
        return 1.0 - 0.25 * min(1.0, abs(x) / ap_) ** 2 if RIDGE else 1.0

    rx = max(float(p.hill_width) / 2, ap_ + frun(0.0))
    ry = max(rx * (0.5 if RIDGE else 0.92), bp_ + frun(pi / 2))
    # the foot lies off-centre: long flanks to the front and left, cliffs behind
    hcx, hcy = (0.0, -0.05 * ry) if RIDGE else (0.03 * rx, -0.16 * ry)
    rh_ = rng.fork("hill")
    ph = [rh_.u(0.0, 6.283) for _ in range(14)]

    def rp(th):
        """Edge of the summit plateau, just outside the walls (never inside them)."""
        return (1.0 + 0.03 * (1.0 + sin(3 * th + ph[10])) + 0.02 * (1.0 + sin(7 * th + ph[11]))) / hypot(cos(th) / ap_, sin(th) / bp_)

    def ro(th):
        """Foot of the hill: an off-centre ellipse with a few bays."""
        c, s_ = cos(th), sin(th)
        qa = (c / rx) ** 2 + (s_ / ry) ** 2
        qb = c * hcx / rx ** 2 + s_ * hcy / ry ** 2
        qc = (hcx / rx) ** 2 + (hcy / ry) ** 2 - 1.0
        bay = 1.0 - 0.035 * (1.0 - sin(2 * th + ph[0])) - 0.03 * (1.0 - sin(3 * th + ph[1])) - 0.015 * (1.0 - sin(5 * th + ph[2]))
        return max(rp(th) + frun(th) - 2.0, bay * (qb + sqrt(qb * qb - qa * qc)) / qa)

    def lobes(th):
        """The two shoulders of the crag, front left and front right of the gate (0..1 each)."""
        return min(1.0, 1.1 * max(0.0, cos(th - thg + 0.95)) ** 2), min(1.0, 1.15 * max(0.0, cos(th - thg - 0.8)) ** 2)

    def shoulder(th):
        """0..1: where a broad bench lies below the cliff - the terraces of the upper hamlet on the two front
        shoulders; on a ridge the two long ends of the crest."""
        if RIDGE:
            return min(1.0, 1.25 * cos(th) ** 2)
        return min(1.0, sum(lobes(th)))

    def bench(th):
        """Width of the bench as a fraction of the slope."""
        return (0.4 if RIDGE else 0.55) * shoulder(th)

    def prof(th, s):
        """Height fraction of the hill at angle th, s = 0 at the plateau edge .. 1 at the foot: a cliff band of
        changing height under the walls, a bench where there is a shoulder, then a hollow flank and a skirt."""
        if s <= 0.0:
            return 1.0
        if s >= 1.0:
            return 0.0
        if RIDGE:       # hogback: a round crest, convex shoulders, the flanks running out gently
            return max(0.0, 0.5 * (1.0 + cos(pi * s ** 0.9)) + 0.02 * sin(pi * s) ** 2 * sin(7 * th + 9 * s + ph[7]))
        sd = shoulder(th)
        cf = min(0.62, max(0.1, CRG * (0.32 + 0.15 * sin(2 * th + ph[3]) + 0.1 * sin(5 * th + ph[4]))))
        sl_, sr_ = lobes(th)        # the right terrace lies just under the walls, the left one a storey lower
        cf = cf * (1.0 - sd) + sd * (0.21 * sl_ + 0.14 * sr_) / max(1e-6, sl_ + sr_)
        if s < SC:
            return 1.0 - cf * s / SC
        u = (s - SC) / (1.0 - SC)
        ex = max(1.1, (1.5 + 0.25 * sin(th + ph[5]) + 0.2 * sin(3 * th + ph[6])) * CRG ** 0.6)
        wb = bench(th)
        ap = min(0.9 * (1.0 - wb), (5.0 if RIDGE else 9.5) / max(1.0, (1.0 - SC) * (ro(th) - rp(th))))     # the apron: 9 m wide
        drop = 0.06 * min(1.0, wb / 0.1)
        ha = 0.085 * (0.75 + 0.25 * sin(2 * th + ph[8]))       # the apron at the foot: a gentle skirt of scree and meadow
        if u < wb:
            b = 1.0 - drop * u / wb
        elif u < 1.0 - ap:
            b = ha + (1.0 - drop - ha) * (1.0 - (u - wb) / (1.0 - ap - wb)) ** ex
        else:
            b = ha * (1.0 - u) / ap
        b += 0.03 * sin(pi * u) ** 2 * sin(7 * th + 9 * u + ph[7])
        return max(0.0, (1.0 - cf) * b)

    # the woods: a few sectors of the flanks, each widening towards the skirt, with a ragged upper edge
    rz = rng.fork("woods")
    nw = 0 if NT <= 0 else max(2, min(5, int(round(NT / 8.0))))
    WOFF = (0.0, 2.6, -2.4, 0.9, -0.9) if RIDGE else (0.8, -0.95, 1.95, -2.2, 3.2)
    WZ = [(thg + WOFF[j] + rz.u(-0.12, 0.12), rz.u(0.4, 0.5) * (1.5 if RIDGE else 1.0), rz.u(0.1, 0.2), rz.u(0.0, 6.283))
          for j in range(nw)]

    def wood_at(th, s):
        for tc, hw_, st, phz in WZ:
            if abs((th - tc + pi) % (2 * pi) - pi) < hw_ * (0.6 + 0.7 * s) and s > st + 0.05 * sin(9.0 * th + phz):
                return True
        return False

    HTR, WARD = [[] for _ in range(NS)], []     # triangles of the built hill surface, per column, and of the summit

    def _zt(x, y, T):
        for a, b, c in T:
            d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
            if abs(d) < 1e-9:
                continue
            u = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / d
            v = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / d
            if u >= -1e-6 and v >= -1e-6 and u + v <= 1.0 + 1e-6:
                return u * a.z + v * b.z + (1.0 - u - v) * c.z
        return None

    def zg(x, y):
        """Height of the hill as built (its facets), so that everything planted on it sits on the surface."""
        if Hh <= 0.5:
            return 0.0
        i0 = int(floor((atan2(y, x) % (2 * pi)) / (2 * pi / NS)))
        for di in (0, -1, 1, -2, 2):
            z = _zt(x, y, HTR[(i0 + di) % NS])
            if z is not None:
                return z
        z = _zt(x, y, WARD)
        if z is not None:
            return z
        return 0.0 if hypot(x, y) > rp(atan2(y, x)) else Hh * crest(x)

    RP = []
    if Hh > 0.5:
        # the surface: a ring under the wards, the rim (dropped and broken, no flat board round the walls), the
        # cliff foot and six rows down the flank. Every other column stands proud as a rib, the ones between
        # are gullies (amplitude from the seed); ledges raise the rows under the cliff into a second step.
        ribk = (0.4 if RIDGE else 1.0) * CRG ** 0.7 * min(1.0, Hh / 30.0)
        RIBA = (0.6, 1.7, 2.4, 2.4, 2.0, 1.4, 0.8, 0.0)
        G = []
        for i in range(NS):
            th0 = 2 * pi * i / NS
            amp = (0.5 if i % 2 else -0.5) + rh_.u(-0.8, 0.8)
            lam = 0.0 if RIDGE else (1.0 - shoulder(th0)) * max(0.0, min(1.0, 0.5 + 0.8 * sin(3 * th0 + ph[12]))) * min(1.0, CRG)
            ri = 1.0 / hypot(cos(th0) / max(2.0, ap_ - 6.2), sin(th0) / max(2.0, bp_ - 6.2))
            row = [Vector((ri * cos(th0), ri * sin(th0), Hh * crest(ri * cos(th0))))]
            zc = Hh
            for j, s in enumerate(SL):
                th = th0 + (rh_.u(-0.05, 0.05) if j > 1 else 0.0)
                s2 = s + (rh_.u(-0.03, 0.03) if 1 < j < len(SL) - 1 else 0.0)
                rho = rp(th) + s2 * (ro(th) - rp(th)) + (max(0.0, amp) if j == 0 else amp) * RIBA[j] * ribk
                x, y = rho * cos(th), rho * sin(th)
                z = Hh * crest(x) * prof(th, s2)
                if j == 0:
                    z -= min(rh_.u(0.4, 2.2), 0.45 * Hh * (1.0 - prof(th, SC)))
                elif j == 1:
                    zc = z
                elif j in (2, 3):
                    z += lam * (0.6, 0.45)[j - 2] * (zc - z)
                row.append(Vector((x, y, max(0.0, z))))
            G.append(row)
        for i in range(NS // 2):            # the wards in strips across the hill, so they can follow a sinking crest
            q = [G[i][0], G[i + 1][0], G[NS - i - 1][0], G[(NS - i) % NS][0]]
            q = q[1:] if i == 0 else q[:2] + q[3:] if i == NS // 2 - 1 else q
            WARD += [(q[0], q[m], q[m + 1]) for m in range(1, len(q) - 1)]
            F("T" if RIDGE else "P", q, 0.1)
        for i in range(NS):
            i2 = (i + 1) % NS
            for r in range(len(SL)):
                a, b, c, d = G[i][r], G[i][r + 1], G[i2][r + 1], G[i2][r]
                HTR[i] += [(a, b, c), (a, c, d)]
                run = hypot((b.x + c.x - a.x - d.x) / 2, (b.y + c.y - a.y - d.y) / 2)
                steep = ((a.z + d.z) - (b.z + c.z)) / 2 / max(run, 0.1)
                thc, sc_ = 2 * pi * (i + 0.5) / NS, (0.0 if r == 0 else (SL[r - 1] + SL[r]) / 2)
                if r == 0:
                    key, g = ("T", 0.1) if RIDGE else ("K", 0.0)
                elif (r == 1 and not RIDGE) or steep > 1.15:
                    key, g = "K", 0.0
                elif wood_at(thc, sc_):
                    key, g = "T", 0.12           # the woodland floor
                elif steep < 0.55:
                    key, g = "T", 0.0            # grass on the benches and the skirt
                else:
                    key, g = "K", 0.0            # bare rock and scree
                F(key, [a, b, c, d], g)

        # the lane: from the foot behind the hill round its left flank up to the gate, climbing steadily, laid
        # on the built surface with a retaining wall on its outer side
        RDA, RDN = 4.7, 30

        def road_pt(t):
            th = (thg - 2.0 + 2.0 * t) if RIDGE else thg - RDA * (1.0 - t)
            lo, hi = rp(th) + 1.0, ro(th) + 2.0
            zt_ = min(Hh * t ** 0.75, zg(lo * cos(th), lo * sin(th)))
            for _ in range(14):
                mid = (lo + hi) / 2
                if zg(mid * cos(th), mid * sin(th)) > zt_:
                    lo = mid
                else:
                    hi = mid
            return lo, th

        if Hh > 4.0:
            rr_ = [road_pt(i / RDN) for i in range(RDN + 1)]
            for i, (rho, th) in enumerate(rr_):
                if 0 < i < RDN:
                    rho = 0.25 * rr_[i - 1][0] + 0.5 * rho + 0.25 * rr_[i + 1][0]
                RP.append((rho * cos(th), rho * sin(th), zg(rho * cos(th), rho * sin(th)) + 0.3, th))
        for i in range(len(RP) - 1):
            ed = []
            for x, y, z, th in (RP[i], RP[i + 1]):
                c, s = cos(th), sin(th)
                xo, yo = x + 2.9 * c, y + 2.9 * s
                ed.append((V3(x - 1.7 * c, y - 1.7 * s, z), V3(x + 1.9 * c, y + 1.9 * s, z),
                           V3(xo, yo, max(0.0, min(z - 0.8, zg(xo, yo) - 0.9)))))
            (Ia, Ea, Da), (Ib, Eb, Db) = ed
            F("L", [Ea, Eb, Ib, Ia])
            F("K", [Da, Db, Eb, Ea])
        # rock outcrops: leaning stacks and buttresses on the upper slopes
        rk = rng.fork("rocks")
        for k in range(int((4 + 3 * CRG) * (0.5 if RIDGE else 1.0))):
            th = thg + rk.u(0.45, 2 * pi - 0.45)
            s = rk.u(0.05, 0.24)
            rho = rp(th) + s * (ro(th) - rp(th))
            x, y = rho * cos(th), rho * sin(th)
            zc = zg(x, y)
            r, n = rk.u(1.9, 3.4), 4 if rk.chance(0.5) else 5
            h = min(rk.u(7.0, 13.0) * min(1.5, CRG) * (0.45 if RIDGE else 1.0), z0 + 2.0 - zc)
            if h < 3.0:
                continue
            a0, lo, hi = rk.u(0.0, 6.283), [], []
            xt, yt = x - 0.3 * r * cos(th), y - 0.3 * r * sin(th)
            for i in range(n):
                an = a0 + 2 * pi * i / n
                rr_ = r * rk.u(0.75, 1.25)
                xv, yv = x + rr_ * cos(an), y + rr_ * sin(an)
                lo.append(V3(xv, yv, max(0.0, zg(xv, yv) - 1.5)))
                rr_ *= rk.u(0.2, 0.42)
                hi.append(V3(xt + rr_ * cos(an), yt + rr_ * sin(an), zc + h * rk.u(0.65, 1.0)))
            for i in range(n):
                j = (i + 1) % n
                F("K", [lo[i], lo[j], hi[j], hi[i]])
            F("K", hi)
    mark("hill")

    def slopes():
        """The hamlet and the woodland, planted on the built surface in the numbers asked for."""
        if Hh <= 4.0:
            return
        rm = rng.fork("hamlet")
        nh_ = int(p.hamlet)
        placed = []
        CT[0] = 1
        # the upper hamlet: a row of cottages on the terrace under the walls, as many as it has room for
        UA0, UA1 = -1.05, 1.15
        n_up = 0 if RIDGE else min((nh_ + 1) // 2, int((UA1 - UA0) * (rp(thg) + 8.0) / 10.6))
        for k in range(nh_):
            if DET >= 4 and tris() > TRI_CAP - 150 - min(1700, 62 * NT):
                break
            # a cottage wants a spot it can stand on: try a number of places on the benches and round the flanks
            # (ridge: along the crest and on the front flank) and take the most level one that is clear of its
            # neighbours and of the lane. Its yard is cut into the slope, so the wall that holds it stays low.
            best = None
            upper = k < n_up
            for attempt in range(216):
                if attempt >= 60 and best is not None:
                    if best[0] < (3.5 if upper else 4.5):
                        break
                    if upper:
                        best, upper = None, False   # no sound spot left on the terraces: down to the foot
                w, d, h, rh2 = rm.u(7.2, 8.2 if upper else 9.0), rm.u(5.6, 6.4), rm.u(3.0, 3.7), rm.u(2.9, 3.4)
                if RIDGE and attempt < 120 and rm.chance(0.62):     # two rows along the crest, clear of the chapel
                    cx_ = rm.u(-0.88, 0.88) * (ap_ - 4.0)
                    off = d / 2 + rm.u(1.4, 2.4) + (3.2 * S + 1.5 if -12.5 * S - w / 2 < cx_ < 8.0 * S + w / 2 else 0.0)
                    cy_, th = (off if rm.chance(0.45) else -off), thg
                else:
                    if RIDGE and attempt < 120:
                        tha = thg + rm.u(-1.15, 1.15)
                        rho = rp(tha) + rm.u(0.2, 0.72) * (ro(tha) - rp(tha))
                    elif upper and attempt < 60:    # the upper hamlet: a row on each terrace, backs to the cliff
                        tha = thg + UA0 + (UA1 - UA0) * (k + 0.5) / n_up + rm.u(-0.01, 0.01) * min(8, 1 + attempt // 4)
                        if bench(tha) < 0.2:
                            continue
                        rho = rp(tha) + SC * (ro(tha) - rp(tha)) + d / 2 + rm.u(0.8, 5.0)
                    elif attempt < 120:             # the lower hamlet on the apron at the foot, mostly in front
                        tha = thg + rm.u(-1.0, 1.0) * (2.4 if attempt < (60 if k >= n_up else 100) else pi)
                        rho = ro(tha) - d / 2 - rm.u(1.0, 1.8)
                    else:                           # a crowded hill: search the whole foot, from the front round to the back
                        tha = thg + (1 if attempt % 2 else -1) * ((attempt - 118) // 2) * (2 * pi / 96)
                        rho = ro(tha) - d / 2 - 1.4
                    if not rp(tha) + d / 2 + 2.5 <= rho <= ro(tha) - d / 2 - 1.0:
                        continue
                    cx_, cy_ = rho * cos(tha), rho * sin(tha)
                    gx, gy = zg(cx_ + 2.0, cy_) - zg(cx_ - 2.0, cy_), zg(cx_, cy_ + 2.0) - zg(cx_, cy_ - 2.0)
                    th = atan2(-gy, -gx) if hypot(gx, gy) > 0.3 else tha      # the front looks down the slope
                ox, oy = cos(th), sin(th)
                tx, ty = -oy, ox
                tight = 1.2 if (upper or attempt >= 40) else 0.0     # a row on the terrace, a crowded hill: the yards may touch
                if any(abs((q[0] - cx_) * tx + (q[1] - cy_) * ty) < 10.8 - tight and
                       abs((q[0] - cx_) * ox + (q[1] - cy_) * oy) < 9.6 - tight for q in placed):
                    continue                        # no two cottages on one spot, none stacked on another
                if any(-d / 2 - 2.3 < (q[0] - cx_) * ox + (q[1] - cy_) * oy < d / 2 + 3.6 and
                       abs((q[0] - cx_) * tx + (q[1] - cy_) * ty) < w / 2 + 2.6 for q in RP):
                    continue                        # clear of the lane
                zc = zg(cx_, cy_)
                fr = d / 2 + 1.5
                zs = [zg(cx_ + ox * a_ + tx * b_, cy_ + oy * a_ + ty * b_)
                      for a_, b_ in ((fr, -w / 2 - 0.9), (fr, w / 2 + 0.9), (fr, 0.0), (-d / 2, 0.0), (0.0, -w / 2), (0.0, w / 2))]
                zf = min(zc, min(zs[:3]) + 2.0)
                score = max(0.0, zs[3] - zf - 2.6) + 0.5 * (zc - zf) + 0.7 * abs(zs[4] - zs[5])
                if best is None or score < best[0]:
                    best = (score, cx_, cy_, th, w, d, h, rh2, zf)
                if score < 1.2 and (attempt >= 12 or score < 0.3):
                    break
            if best is None or best[0] > 6.0:       # rather one cottage short than one buried in a cliff
                continue
            score, cx_, cy_, th, w, d, h, rh2, zf = best
            ox, oy = cos(th), sin(th)
            tx, ty = -oy, ox
            CNT[0] += 1
            sd = 1 if rm.chance(0.5) else -1
            lean = rm.u(2.2, 2.9) if (rm.chance(0.5) and all(hypot(cx_ - q[0], cy_ - q[1]) > 14.0 for q in placed)) else 0.0
            placed.append((cx_, cy_))
            XF[0] = (cx_, cy_, cos(th + pi / 2), sin(th + pi / 2))
            pb()
            # the bench it stands on: a level yard, held by a low retaining wall wherever the ground falls away
            xa, xb = -w / 2 - 0.9 - (lean if sd > 0 else 0.0), w / 2 + 0.9 + (lean if sd < 0 else 0.0)
            cs = [(xa, -d / 2 - 1.5), (xb, -d / 2 - 1.5), (xb, d / 2 + 0.4), (xa, d / 2 + 0.4)]
            gl_ = [zg(cx_ + x_ * tx - y_ * ox, cy_ + x_ * ty - y_ * oy) for x_, y_ in cs]
            # CLEARANCE RULE: the yard may cut through sloping ground, but it must not lie in a patch of all but level
            # ground (both glow in the floodlight: two sheets a few millimetres apart fight over metres). There it
            # stands 6 cm proud of the patch or, where that would lift the cottage, 6 cm under it.
            fz_ = []
            for qa in range(5):
                for qb in range(5):
                    x_, y_ = xa + (xb - xa) * qa / 4, -d / 2 - 1.5 + (d + 1.9) * qb / 4
                    xw, yw = cx_ + x_ * tx - y_ * ox, cy_ + x_ * ty - y_ * oy
                    if hypot(zg(xw + 0.1, yw) - zg(xw - 0.1, yw), zg(xw, yw + 0.1) - zg(xw, yw - 0.1)) < 0.014:   # under 4 degrees
                        fz_.append(zg(xw, yw))
            fz_.sort()
            while fz_:
                q = 1
                while q < len(fz_) and fz_[q] - fz_[q - 1] < 0.3:
                    q += 1
                dn, up = fz_[0] - 0.06, fz_[q - 1] + 0.06
                if dn < zf < up:
                    zf = up if (up - zf < 0.35 or up - zf < zf - dn) else dn
                fz_ = fz_[q:]
            GB[0] = zf - 1.5
            F("L", [V3(x_, y_, zf) for x_, y_ in cs])
            for q in range(4):
                q2 = (q + 1) % 4
                if gl_[q] > zf + 0.2 and gl_[q2] > zf + 0.2:
                    continue
                F("W", [V3(*cs[q], max(0.0, min(zf - 0.3, gl_[q] - 0.9))), V3(*cs[q2], max(0.0, min(zf - 0.3, gl_[q2] - 0.9))),
                        V3(*cs[q2], zf), V3(*cs[q], zf)])
            GB[0] = zf
            lv = [(zf + 1.0, 1.25, 0.0)]
            block(-w / 2, w / 2, -d / 2, d / 2, max(0.0, zf - 0.3), zf + h, roof="gx", rh=rh2, ov=0.45,
                  cols={"f": wcols(w, 2, 1.1, lv, 2.2, 0.8, 0.7, bright=True),
                        "r" if sd > 0 else "l": wcols(d, 1, 1.1, lv, 2.0, 1.0, bright=True)},
                  luc=1 if rm.chance(0.5) else 0, ld=(1.2, 1.05, 0.7), lb=2.2, lbright=True)
            # a chimney stack through the ridge by one gable, often a lean-to against the other
            xc = sd * (w / 2 - rm.u(0.8, 1.5))
            box(xc - 0.42, xc + 0.42, -0.4, 0.4, zf + h + 0.55 * rh2, zf + h + rh2 + rm.u(0.8, 1.2), key="W+")
            if lean:
                xi, xo = -sd * w / 2, -sd * (w / 2 + lean)
                y0_, zl_, zh_, zb_ = -d / 2 + rm.u(0.5, 1.2), zf + 1.9, zf + min(h - 0.2, 2.9), max(0.0, zf - 0.3)
                ci_ = V3((xi + xo) / 2, 0.0, zf)
                Fc("W", [V3(xi, y0_, zb_), V3(xo, y0_, zb_), V3(xo, y0_, zl_), V3(xi, y0_, zh_)], ci_)
                Fc("W", [V3(xo, y0_, zb_), V3(xo, d / 2, zb_), V3(xo, d / 2, zl_), V3(xo, y0_, zl_)], ci_)
                xe = xo - sd * 0.3
                Fc("R", [V3(xi, y0_ - 0.3, zh_ + 0.12), V3(xe, y0_ - 0.3, zl_ - 0.1), V3(xe, d / 2, zl_ - 0.1), V3(xi, d / 2, zh_ + 0.12)],
                   ci_)
            pe()
            XF[0] = None
            GB[0] = Hh
        CT[0] = 0
        mark("hamlet")
        # woodland: every clump is a tall rounded crown with one or two lower ones leaning on it along the
        # contour, some with a fir. The clumps of a wood stand rank below rank from under the cliffs to the
        # skirt, crowns overlapping; every fifth clump joins the ragged tree-line on the benches. The tone of
        # a crown follows its rounded form: pale on top and towards the moon, dark on the far side.
        rt = rng.fork("trees")

        def clear(x, y, r, f=1.0):
            return all(hypot(x - q[0], y - q[1]) > (r + 5.4) * f for q in placed) and \
                all(hypot(x - q[0], y - q[1]) > 0.5 * r + 1.9 for q in RP)

        def dome(x, y, r, h, n, low=False):
            a0, zc = rt.u(0.0, 6.283), zg(x, y)
            rings = []
            for fr, fz in (((1.0, 0.0), (0.6, 0.86)) if low else ((0.62, None), (1.0, 0.45), (0.58, 0.93))):
                ring = []
                for i in range(n):
                    an = a0 + 2 * pi * i / n
                    rr_ = r * fr * rt.u(0.9, 1.1)
                    xv, yv = x + rr_ * cos(an), y + rr_ * sin(an)
                    dz = zg(xv, yv) - zc
                    lean = max(-1.6, min(1.6, 0.45 * dz))       # the crown is a round mass that only leans a little
                    if fz is None:      # its underside: tucked in, sunk into the ground uphill, clear of it downhill
                        # CLEARANCE RULE: it does not climb a steep slope with the ground (there it came to lie in
                        # the side of the crown, two sheets of one glowing canopy in one place)
                        ring.append(V3(xv, yv, max(0.0, zc + max(min(dz - 0.5, 0.2 * h), lean + 0.06 * h))))
                    elif fz == 0.0:     # a low crown: its skirt follows the ground
                        ring.append(V3(xv, yv, max(0.0, zc + dz - 0.4)))
                    else:
                        ring.append(V3(xv, yv, zc + lean + h * fz * rt.u(0.93, 1.07)))
                rings.append(ring)
            c = Vector((x, y, zc + 0.2 * h))

            def gr(q):
                return max(0.0, min(0.95, 0.52 - 0.3 * (q - c).normalized().dot(Lm) + 0.08 * hsh(q)))
            for lo, hi in zip(rings[:-1], rings[1:]):
                for i in range(n):
                    j = (i + 1) % n
                    P = [lo[i], lo[j], hi[j], hi[i]]
                    F("T", P, gv=[gr(q) for q in P])
            F("T", rings[-1], gv=[gr(q) for q in rings[-1]])
            if not low and any(q.z > zg(q.x, q.y) + 0.3 for q in rings[0]):      # the shadowed underside where it overhangs
                F("T", rings[0][::-1], gv=[0.95] * n)

        def fir(x, y, r, h):
            zb_, a0 = zg(x, y), rt.u(0.0, 6.283)

            def ring(rad, z, ph_=0.0):
                return [V3(x + rad * cos(a0 + ph_ + pi * i / 2), y + rad * sin(a0 + ph_ + pi * i / 2), z) for i in range(4)]

            def gr(q, z):
                d = Vector((q.x - x, q.y - y, 0.0))
                d = (d.normalized() if d.length > 1e-6 else Vector((0.0, 0.0, 0.0))) * 0.8 + UP * 0.6
                return max(0.25, min(0.98, 0.6 - 0.3 * d.dot(Lm) + 0.08 * hsh(q)))
            A, M_, C = ring(r, zb_ + 1.2), ring(0.34 * r, zb_ + 0.52 * h), ring(0.7 * r, zb_ + 0.44 * h, pi / 4)
            for q in A:         # the lowest boughs sweep down the slope
                q.z = max(0.0, min(q.z, max(zg(q.x, q.y) + 0.6, zb_ - 2.0)))
            ap = V3(x, y, zb_ + h)
            for i in range(4):
                j = (i + 1) % 4
                P = [A[i], A[j], M_[j], M_[i]]
                F("T", P, gv=[gr(q, 0) for q in P])
                P = [C[i], C[j], ap]
                F("T", P, gv=[gr(q, 0) for q in P])

        wc = []
        nr = max(1, -(-NT // max(1, nw)))       # ranks of a wood
        for k in range(NT):
            if DET >= 4 and tris() > TRI_CAP - 110:
                break
            tc, hw_, st, phz = WZ[k % nw]
            cand = None
            for attempt in range(40):
                if k % 5 == 4 and attempt < 10:      # the tree-line: on the benches under the cliffs
                    th = thg + 2 * pi * ((k * 0.61803 + 0.071 * attempt) % 1.0) + rt.u(-0.08, 0.08)
                    if bench(th) <= 0.1:
                        continue
                    s = SC + bench(th) * rt.u(0.2, 0.95) * (1.0 - SC)
                else:                                # the woods: rank below rank
                    s = st + (0.97 - st) * min(1.0, (k // nw + rt.u(0.1, 0.9)) / nr)
                    th = tc + rt.u(-1.0, 1.0) * hw_ * (0.6 + 0.7 * s) * (0.9 + 0.02 * attempt)
                r = rt.u(3.6, 5.2) * (0.85 if s < 0.3 else 1.0)
                rho = min(rp(th) + s * (ro(th) - rp(th)), ro(th) - 0.7 * r)
                x, y = rho * cos(th), rho * sin(th)
                if not clear(x, y, r, max(0.75, 1.0 - 0.01 * attempt)):
                    continue
                cand = (x, y, r, th)                 # good enough if nothing better turns up
                ex_, ey_ = 0.8 * r * cos(th), 0.8 * r * sin(th)
                if attempt < 30 and zg(x - ex_, y - ey_) - zg(x + ex_, y + ey_) > 2.9 * r:      # not on the rock faces
                    continue
                fs = 0.62 * max(0.45, 1.0 - 0.02 * attempt)
                if all(hypot(x - q[0], y - q[1]) > fs * (r + q[2]) for q in wc):
                    break
            for attempt in range(60 if cand is None else 0):     # no room in its wood: anywhere on the flanks
                th, r = rt.u(0.0, 6.283), rt.u(3.6, 4.6)
                rho = min(rp(th) + rt.u(0.15, 0.95) * (ro(th) - rp(th)), ro(th) - 0.7 * r)
                if clear(rho * cos(th), rho * sin(th), r, 0.65):
                    cand = (rho * cos(th), rho * sin(th), r, th)
                    break
            if cand is None:
                continue
            x, y, r, th = cand
            wc.append((x, y, r))
            h = rt.u(7.5, 10.5)
            CNT[1] += 1
            dome(x, y, r, h, 6)
            sg = 1 if rt.chance(0.5) else -1
            for q in range(2 if rt.chance(0.45) else 1):      # the lower ones stand along the contour
                r2 = r * rt.u(0.62, 0.86)
                dd, da = 0.7 * (r + r2), th + sg * (pi / 2 + rt.u(-0.4, 0.4))
                x2, y2 = x + dd * cos(da), y + dd * sin(da)
                sg = -sg
                if clear(x2, y2, r2, 0.8) and rp(atan2(y2, x2)) + 1.0 < hypot(x2, y2) < ro(atan2(y2, x2)) - 0.5 * r2:
                    wc.append((x2, y2, r2))
                    dome(x2, y2, r2, h * rt.u(0.55, 0.8), 5, low=True)
            if rt.chance(0.35):
                xf, yf = x + rt.u(-0.8, 0.8) * r, y + rt.u(-0.8, 0.8) * r
                if clear(xf, yf, 0.5, 0.8):
                    fir(xf, yf, rt.u(1.9, 2.6), rt.u(11.0, 15.0))
        mark("woods")

    zb0 = max(0.0, z0 - 12.0)       # foundations sunk into the crag
    zb1 = max(0.0, z0 - 2.0)

    def curtain(a_, b_, nv, hw, tower_fn, gate="cone"):
        """Crenellated ring wall on a battered plinth, on an ellipse, with mural towers at its corners and a
        gatehouse at the front."""
        rc = rng.fork("curtain")
        cv = []
        for i in range(nv):
            th = -pi / 2 + 2 * pi * i / nv + (0.0 if i == 0 else rc.u(-0.08, 0.08))
            f = 1.0 if i == 0 else rc.u(0.9, 1.04)
            cv.append((a_ * f * cos(th), b_ * f * sin(th)))
        zt = z0 + hw

        def sc_(q, f):
            return (q[0] * (1 + f / hypot(*q)), q[1] * (1 + f / hypot(*q)))
        ZN[0] = 0
        rl = rng.fork("loops")
        zc_ = zt - 1.9
        for i in range(nv):
            a, b = cv[i], cv[(i + 1) % nv]
            ai, bi = sc_(a, -1.9), sc_(b, -1.9)
            ao, bo = sc_(a, 1.5), sc_(b, 1.5)
            ap, bp = sc_(a, 0.6), sc_(b, 0.6)
            Lw = hypot(b[0] - a[0], b[1] - a[1])
            loops = []
            for f in ((0.5,) if Lw < 14.0 else (0.36, 0.64)):
                if Lw > 9.0 and rl.chance(0.8) and DET < 2:
                    u, zl = f * Lw + rl.u(-0.6, 0.6), z0 + rl.u(3.6, 4.4)
                    loops.append((u - 0.26, u + 0.26, [(zl, zl + 2.3, "dw")]))
            tc = wall(a, b, zb0, zc_, loops, streak=False)[1]
            # the wall-walk rides out on a corbel table, its merlons stand clear of the wall face
            stitch(tc, [V3(*ap, zc_ + 0.8), V3(*bp, zc_ + 0.8)], "W+")
            battlement(ap, bp, zc_ + 0.8, zt, zt + 1.9, 2.5, 1.8)
            F("W", [V3(*ao, zb0), V3(*bo, zb0), V3(*b, z0 + 3.0), V3(*a, z0 + 3.0)])
            wall(bi, ai, zb1, zt - 0.3)
            F("W+", [V3(*ap, zt - 0.3), V3(*bp, zt - 0.3), V3(*bi, zt - 0.3), V3(*ai, zt - 0.3)])
        for i in range(1, nv):
            c = cv[i]
            tower_fn(i, c, rc, zt, atan2(c[1] / (b_ * b_), c[0] / (a_ * a_)) + pi / 2)
        if gate:
            gy = cv[0][1]
            for sx in (-1, 1):
                tower(sx * 4.7, gy - 0.4, 3.0, zb0, z0, zt + 3.0, n=10, top=gate, sh=8.0, maxlev=2, boost=1.5, pw=0.45, fin=True,
                      pl=0.5)
            zg0, zsp = z0 + 0.1, z0 + 5.9
            door = [(4.7 - 1.7, 4.7 + 1.7, [(zg0, zg0 + 3.1, "dk"), (zg0 + 3.1, zg0 + 4.8, ("arch", "dk"))])]
            lanc = [(4.7 + sg * 1.15 - 0.5, 4.7 + sg * 1.15 + 0.5, [(zsp + 1.5, zsp + 3.3, "lit"), (zsp + 3.3, zsp + 4.2, ("arch", "lit"))])
                    for sg in (-1, 1)]
            pb()
            wall2((-4.7, gy - 2.4), (4.7, gy - 2.4), zb0, zsp, zt + 2.6, door, lanc)
            block(-4.7, 4.7, gy - 2.4, gy + 2.6, zb0, zt + 2.6, top="batt", skip="f")
            pe()
        ZN[0] = 1
        return cv

    # ------------------------------------------------------------------ the summit
    if p.style == "castle":
        a_, b_ = 30.0 * S, 19.0 * S

        def mural(i, c, rc, zt, rot):
            if i % 2 == 0 and rc.u(0.0, 1.0) < 0.4:
                return
            kind = rc.weighted([("cone", 4), ("batt", 3), ("sq_batt", 2), ("sq_roof", 2), ("big", 1.5)])
            if kind == "cone":
                r = rc.u(2.5, 3.3) * RS
                tower(c[0], c[1], r, zb0, z0, zt + rc.u(3.0, 6.5), n=8, sh=r * rc.u(3.0, 4.4), maxlev=2, pl=0.4)
            elif kind == "batt":
                tower(c[0], c[1], rc.u(3.3, 4.3) * RS, zb0, z0, zt + rc.u(0.3, 2.5), n=8, top="batt", maxlev=1, pl=0.5)
            elif kind == "sq_batt":
                tower(c[0], c[1], rc.u(3.7, 4.5) * RS, zb0, z0, zt + rc.u(2.0, 5.5), n=4, rot=rot, top="batt", maxlev=2, pw=0.7,
                      pl=0.6)
            elif kind == "sq_roof":
                r = rc.u(3.3, 4.1) * RS
                tower(c[0], c[1], r, zb0, z0, zt + rc.u(2.5, 6.0), n=4, rot=rot, top=rc.weighted([("saddle", 1), ("cone", 1)]),
                      sh=r * rc.u(1.5, 2.3), maxlev=2, pw=0.7, pl=0.6)
            else:
                r = rc.u(3.8, 4.6) * RS
                tower(c[0], c[1], r, zb0, z0, zt + rc.u(6.0, 10.0), n=10, top="batt+cone", sh=r * 3.2, maxlev=3, fin=True, pl=0.5)
        curtain(a_, b_, max(8, min(14, int(round(15.0 * S)))), 9.5, mural)
        mark("curtain")
        # keep: square great tower with the clock, corbelled battlement, helm roof with lucarnes, corner bartizans
        kx0, kx1, ky0, ky1 = -12.5 * S, 0.5 * S, -2.5 * S, 9.5 * S
        kzt = z0 + 1.15 * HT      # the clock stage rides above the gatehouse
        Lk, Dk = kx1 - kx0, ky1 - ky0
        cw = min(5.8, Lk - 2.4, 0.3 * HT) if feat != "none" else 0.0
        ztop = kzt - (cw + 2.6 if cw > 2.5 else 1.2)
        klev, z = [], z0 + 4.0
        while z + 3.3 + 1.0 < ztop:
            klev.append((z, 2.3, 1.0))
            z += 5.6
        fcols = wcols(Lk, 2, 1.7, klev, 1.3, 1.0)
        fcols = [(1.2, 2.9, fcols[0][2]), (Lk - 2.9, Lk - 1.2, fcols[1][2])] if (fcols and Lk / 2 - cw / 2 > 3.4) else []
        if cw > 2.5:
            fcols.append((Lk / 2 - cw / 2, Lk / 2 + cw / 2, [(kzt - cw - 1.7, kzt - 1.7, ("disc", feat))]))
        slev = klev + ([(klev[-1][0] + 5.6, 2.3, 1.0)] if (klev and klev[-1][0] + 5.6 + 4.3 < kzt) else [])
        block(kx0, kx1, ky0, ky1, zb1, kzt, top="batt", roof="pyr", rh=0.85 * Lk, luc="fr", fin=True, base=z0 + 2.6,
              cols={"f": fcols, "r": wcols(Dk, 2, 1.7, slev, 1.0, 0.85, 1.0), "l": wcols(Dk, 2, 1.7, slev, 1.0, 0.85, 1.0),
                    "b": wcols(Lk, 2, 1.7, slev, 1.0, 0.7, 1.0)})
        for cx_ in (kx0, kx1):
            for cy_ in (ky0, ky1):
                tower(cx_, cy_, 1.7, kzt - 6.0, kzt - 6.0, kzt + 4.2, n=6, sh=8.0, maxlev=0, corbel=True)
        mark("keep")
        # great hall with its row of tall lancets, lucarnes and chimneys
        hx0, hx1, hy0, hy1 = 4.0 * S, 25.0 * S, -3.0 * S, 6.5 * S
        nh = max(3, int((hx1 - hx0) / 4.4))
        hlev = [(z0 + 3.2, 4.4, 1.5)]
        hrh = 0.85 * (hy1 - hy0)
        block(hx0, hx1, hy0, hy1, zb1, z0 + 12.0, roof="gx", rh=hrh, luc=3, base=z0 + 2.0,
              cols={"f": wcols(hx1 - hx0, nh, 1.8, hlev, 2.0), "b": wcols(hx1 - hx0, nh, 1.8, [(z0 + 3.2, 5.6, 0.0)], 1.6),
                    "r": wcols(hy1 - hy0, 1, 2.0, [(z0 + 4.0, 5.0, 1.8)], 2.0)})
        for fx in (0.2, 0.78):
            xc = hx0 + fx * (hx1 - hx0)
            box(xc - 0.8, xc + 0.8, (hy0 + hy1) / 2 + 0.6, (hy0 + hy1) / 2 + 2.0, z0 + 14.0, z0 + 12.0 + hrh + 2.2)
        for k in range(nh + 1):     # buttresses between the bays
            buttress(min(max(hx0 + k * (hx1 - hx0) / nh, hx0 + 0.6), hx1 - 0.6), hy0, 1.15, 0.5, zb1, z0 + 9.2)
        # chapel with a fleche
        cx0, cx1, cy0, cy1 = -24.5 * S, -14.5 * S, -7.5 * S, -1.5 * S
        clev = [(z0 + 2.6, 3.0, 1.2)]
        block(cx0, cx1, cy0, cy1, zb1, z0 + 8.5, roof="gx", rh=1.05 * (cy1 - cy0), base=z0 + 1.6,
              cols={"f": wcols(cx1 - cx0, 3, 1.6, clev, 1.8), "b": wcols(cx1 - cx0, 3, 1.6, [(z0 + 2.6, 4.0, 0.0)], 1.5),
                    "l": wcols(cy1 - cy0, 1, 1.7, clev, 2.0)})
        tower((cx0 + cx1) / 2, (cy0 + cy1) / 2, 1.0, z0 + 10.0, z0 + 10.0, z0 + 8.5 + 1.05 * (cy1 - cy0) + 2.0, n=6, sh=9.0,
              maxlev=0, fin=True)
        mark("hall+chapel")
        # the cluster of slim spired towers, tallest at the heart; none may stand in front of the clock
        rs = rng.fork("spires")
        N = int(p.spires)
        TL = []
        for k in range(N):
            rho = sqrt((k + 0.5) / N)
            for attempt in range(8):
                phi = k * 2.39996 + rs.u(-0.3, 0.3) + attempt * 0.7
                x, y = 0.72 * a_ * rho * cos(phi), 0.62 * b_ * rho * sin(phi) + 2.0 * S
                if not (kx0 - 3.5 < x < kx1 + 3.5 and y < ky0 + 1.0) and not (hx0 + 1.0 < x < hx1 - 1.0 and y < hy0 + 1.0):
                    break
            n_ = 10 if k == 0 else rs.weighted([(6, 5), (8, 2), (4, 2)])
            r = rs.u(2.1, 2.8) * SLN * (1.3 if k == 0 else 1.0) * (1.2 if n_ == 4 else 1.0)
            ztk = z0 + HT * (1.45 - 0.75 * rho) * SOAR + rs.u(-2.0, 3.0)
            top = rs.weighted([("cone", 5), ("batt+cone", 2)] + ([("saddle", 3)] if n_ == 4 else []))
            shk = r * rs.u(1.7, 2.5) if top == "saddle" else r * (0.8 if k == 0 else 1.0) * rs.u(4.5, 6.2) * sqrt(SOAR / SLN)
            tower(x, y, r, zb1, z0, ztk, n=n_, top=top, sh=shk, maxlev=3, belfry=3.2 if k < 3 else 0.0,
                  luc=(4 if k == 0 else 2) if k < 4 else 0, fin=k < 6, rot=rs.u(0.0, 0.8) if n_ == 4 else 0.0)
            TL.append((x, y, r, ztk))
        # flying bridges between neighbouring towers (never through the keep or the hall roof)
        nb, used = int(p.bridges), set()
        pairs = sorted((hypot(TL[i][0] - TL[j][0], TL[i][1] - TL[j][1]), i, j) for i in range(len(TL)) for j in range(i))
        for d_, i, j in pairs:
            if nb <= 0:
                break
            if d_ < TL[i][2] + TL[j][2] + 3.0 or d_ > 24.0 or i in used or j in used:
                continue
            zd = min(TL[i][3], TL[j][3]) - rs.u(3.0, 6.5)
            hit = False
            for q in range(1, 6):
                xq, yq = TL[i][0] + (TL[j][0] - TL[i][0]) * q / 6, TL[i][1] + (TL[j][1] - TL[i][1]) * q / 6
                if kx0 - 2.0 < xq < kx1 + 2.0 and ky0 - 2.0 < yq < ky1 + 2.0 and zd < kzt + 0.85 * Lk + 4.0:
                    hit = True
                if hx0 - 1.0 < xq < hx1 + 1.0 and hy0 - 1.0 < yq < hy1 + 1.0 and zd < z0 + 12.0 + hrh + 5.0:
                    hit = True
            if hit or zd < z0 + 0.5 * HT:
                continue
            bridge(TL[i][0], TL[i][1], TL[j][0], TL[j][1], zd)
            used.add(i)
            used.add(j)
            nb -= 1
        # corbelled flying turrets on the tallest towers
        for j in range(int(p.turrets)):
            if not TL:
                break
            x, y, r, ztk = TL[j % min(len(TL), 5)]
            an = -pi / 2 + rs.u(-2.0, 2.0)
            zq = ztk - rs.u(4.5, 8.0) - 4.0 * (j // 5)
            if zq < z0 + 8.0:
                continue
            tower(x + (r + 0.3) * cos(an), y + (r + 0.3) * sin(an), 1.05, zq, zq, zq + 4.4, n=6, sh=5.4, corbel=True,
                  lev=[(zq + 1.2, 2.0, 0.0)], pw=0.34, keep=1.0, boost=2.2, wf=0.7)
        mark("spires")

    elif p.style == "cathedral":
        zc = z0 + 0.75 * HT
        nx0, nx1, hy = -20.0 * S, 20.0 * S, 5.5 * S
        rhn = 1.55 * hy
        nb = max(4, int((nx1 - nx0) / 5.2))
        Ln = nx1 - nx0
        pb()        # nave, aisles, transept, crossing and west towers share walls: one part
        # nave with clerestory lancets
        nlev = [(zc - 5.6, 2.9, 1.2)]
        block(nx0, nx1, -hy, hy, zb1, zc, roof="gx", rh=rhn, cols={"f": wcols(Ln, nb, 1.8, nlev, 1.6), "b": wcols(Ln, nb, 1.8, nlev, 1.6)})
        # aisles under lean-to roofs, buttresses between the bays
        za_, wa = z0 + 0.42 * (zc - z0), 4.5 * S
        La = Ln - 2.0 * S
        alev = [(z0 + 2.2, max(1.6, min(3.0, za_ - z0 - 3.8)), 1.1)]
        for sg in (-1, 1):
            yo, yi = sg * (hy + wa), sg * hy
            block(nx0, nx0 + La, min(yo, yi), max(yo, yi), zb1, za_, cols={"f" if sg < 0 else "b": wcols(La, nb, 1.7, alev, 1.3)},
                  skip="b" if sg < 0 else "f")
            xa0, xa1, zl, zh = nx0 - 0.3, nx0 + La + 0.3, za_ - 0.25, za_ + 3.2
            if sg < 0:
                F("R", [V3(xa0, yo - 0.4, zl), V3(xa1, yo - 0.4, zl), V3(xa1, yi, zh), V3(xa0, yi, zh)])
            else:
                F("R", [V3(xa1, yo + 0.4, zl), V3(xa0, yo + 0.4, zl), V3(xa0, yi, zh), V3(xa1, yi, zh)])
            lo, hi_ = min(yo, yi), max(yo, yi)
            F("W", [V3(nx0 + La, lo, za_), V3(nx0 + La, hi_, za_), V3(nx0 + La, yi, zh)])
            F("W", [V3(nx0, hi_, za_), V3(nx0, lo, za_), V3(nx0, yi, zh)])
            pitch = La / nb
            for k in range(1, nb):
                xb = nx0 + k * pitch
                # CLEARANCE RULE: the back of a buttress stands inside the aisle, not in the plane of its wall
                ya, yb = (yo - 1.3, yo + 0.3) if sg < 0 else (yo - 0.3, yo + 1.3)
                box(xb - 0.5, xb + 0.5, ya, yb, zb1, za_ + 1.4, key="W+")
        # transept with the rose over three lancets
        tx0, tx1, ty = 3.0 * S, 13.0 * S, hy + wa + 3.0 * S
        Lt = tx1 - tx0
        rwd = min(7.2, Lt - 2.6)
        zro = zc - rwd - 0.6
        hbl = min(5.0, zro - (z0 + 2.5) - 1.4 - 1.4)
        lw = min(1.5, rwd / 4.6)
        if hbl > 1.5:
            zmid = z0 + 2.5 + hbl + 1.4 + 0.5
            kinds = [wk(2.6) for _ in range(3)]
            lc = [(Lt / 2 + (q - 1) * 1.55 * lw - lw / 2, Lt / 2 + (q - 1) * 1.55 * lw + lw / 2,
                   [(z0 + 2.5, z0 + 2.5 + hbl, kinds[q]), (z0 + 2.5 + hbl, z0 + 2.5 + hbl + 1.4, ("arch", kinds[q]))]) for q in range(3)]
            _, tca = wall((tx0, -ty), (tx1, -ty), zb1, zmid, lc)
            zup = zmid + 0.45
        else:
            tca, zup = None, zb1
        rc_ = [(Lt / 2 - rwd / 2, Lt / 2 + rwd / 2, [(zro, zro + rwd, ("disc", feat))])] if feat != "none" else []
        bcb, tcb = wall((tx0, -ty), (tx1, -ty), zup, zc, rc_)
        if tca is not None:
            stitch(tca, bcb, "W+")
        block(tx0, tx1, -ty, ty, zb1, zc, roof="gy", rh=1.55 * Lt / 2, skip="f",
              cols={"b": wcols(Lt, 2, 1.7, [(z0 + 2.5, 4.0, 1.4)], 1.5)})
        xm_ = (tx0 + tx1) / 2
        stitch(tcb, [V3(xm_, -ty, zc + 1.55 * Lt / 2)])
        # crossing tower and spire
        zx = zc + rhn + 0.42 * HT
        xlev = [(zx - 5.6, 3.0, 1.2)]
        block(tx0, tx1, -hy, hy, zc + 0.5, zx, cols={"f": wcols(Lt, 2, 1.6, xlev, 1.6), "b": wcols(Lt, 2, 1.6, xlev, 1.2),
                                                    "l": wcols(2 * hy, 2, 1.6, xlev, 1.2), "r": wcols(2 * hy, 2, 1.6, xlev, 1.2)})
        spire_on(tx0, tx1, -hy, hy, zx, 1.25 * HT, pin=5.5)
        # twin west towers with belfries and spires
        zw = z0 + 1.45 * HT
        tw = hy + 0.6 * wa
        wx0, wx1 = nx0 - 1.5 * tw + 2.0, nx0 + 2.0
        wl = [(z0 + 4.0, 3.0, 1.2), (z0 + 0.62 * (zw - z0) - 2.0, 3.0, 1.2), (zw - 8.2, 4.8, 1.6)]
        for sg in (-1, 1):
            y0_, y1_ = (-tw, 0.0) if sg < 0 else (0.0, tw)
            block(wx0, wx1, y0_, y1_, zb1, zw, skip="b" if sg < 0 else "f", base=z0 + 2.6,
                  cols={s_: wcols((wx1 - wx0) if s_ in "fb" else tw, 2, 1.6, wl, 1.2, 0.9, 0.6) for s_ in "frbl"})
            spire_on(wx0, wx1, y0_, y1_, zw, 0.95 * HT)
        pe()
        # apse
        tower(nx1, 0.0, hy * 0.98, zb1, z0, zc - 1.0, n=12, sh=1.5 * hy, lev=[(z0 + 4.5, max(2.0, zc - z0 - 10.0), 1.5)], wf=0.4,
              boost=1.6, pw=0.9, keep=1.0, fin=True)
        mark("cathedral")

    elif p.style == "citadel":
        a_, b_ = 35.0 * S, 22.5 * S

        def mural(i, c, rc, zt, rot):
            if i % 2 == 0 and rc.u(0.0, 1.0) < 0.3:
                return
            if rc.u(0.0, 1.0) < 0.3:
                tower(c[0], c[1], rc.u(4.0, 4.8) * RS, zb0, z0, zt + rc.u(1.5, 4.5), n=4, rot=rot, top="batt", maxlev=2, pw=0.7,
                      pl=0.6)
            else:
                tower(c[0], c[1], rc.u(3.3, 4.4) * RS, zb0, z0, zt + rc.u(0.5, 4.5), n=10, top="batt", maxlev=2, pl=0.5)
        curtain(a_, b_, max(9, min(14, int(round(17.5 * S)))), 9.5, mural, gate="batt")
        mark("curtain")
        # inner ward: a crenellated podium
        zp = z0 + 7.0
        pc = (-2.0 * S, 2.0 * S)
        pod = [(pc[0] + 20.0 * S * cos(2 * pi * i / 10 + 0.2), pc[1] + 12.5 * S * sin(2 * pi * i / 10 + 0.2)) for i in range(10)]
        for i in range(10):
            battlement(pod[i], pod[(i + 1) % 10], zb1, zp, zp + 1.7, 2.4, 1.7)
        F("W+", [V3(x, y, zp - 0.3) for x, y in pod])
        # great drum keep with a look-out turret and a beacon
        dx, dy, dr = pc[0] - 2.0 * S, pc[1] + 1.0 * S, 8.5 * S
        zk = zp + HT
        tower(dx, dy, dr, zp - 1.0, zp, zk, n=16, top="batt", maxlev=3, wf=0.5, pw=0.7, keep=0.8, boost=1.3, pl=0.6)
        tower(dx + 0.45 * dr, dy + 0.3 * dr, 3.2, zk + 0.5, zk + 1.4, zk + 10.0, n=8, top="batt+cone", sh=8.5, maxlev=1, pw=0.6, keep=1.0,
              fin=True)
        # the beacon: an iron brazier on a stone stand on the keep roof, its flame just clear of the merlons
        bx_, by_ = dx - 0.42 * dr, dy - 0.3 * dr
        box(bx_ - 0.5, bx_ + 0.5, by_ - 0.5, by_ + 0.5, zk + 1.3, zk + 3.45, key="W+", top=False)
        box(bx_ - 0.7, bx_ + 0.7, by_ - 0.7, by_ + 0.7, zk + 3.45, zk + 3.9, key="dk")
        fb = [V3(bx_ + 0.52 * cos(0.4 + pi * i / 2), by_ + 0.52 * sin(0.4 + pi * i / 2), zk + 3.9) for i in range(4)]
        for i in range(4):
            F("lit", [fb[i], fb[(i + 1) % 4], V3(bx_ + 0.12, by_ - 0.05, zk + 5.3)])
        # square clock / gate tower, a second tower and a barracks range on the podium
        sx0, sx1, sy0, sy1 = pc[0] + 8.0 * S, pc[0] + 15.0 * S, pc[1] - 8.0 * S, pc[1] - 1.5 * S
        zs_, Ls_ = zp + 0.72 * HT, sx1 - sx0
        dw = min(4.6, Ls_ - 2.0)
        scol = [(Ls_ / 2 - 0.85, Ls_ / 2 + 0.85, [(zp + 3.5, zp + 5.7, "lit"), (zp + 5.7, zp + 6.6, ("arch", "lit"))])]
        if feat != "none" and zs_ - dw - 1.0 > zp + 7.4:
            pb()
            wall2((sx0, sy0), (sx1, sy0), zp - 1.0, zp + 7.4, zs_, scol,
                  [(Ls_ / 2 - dw / 2, Ls_ / 2 + dw / 2, [(zs_ - dw - 1.0, zs_ - 1.0, ("disc", feat))])])
            block(sx0, sx1, sy0, sy1, zp - 1.0, zs_, top="batt", roof="pyr", rh=5.0, skip="f", fin=True)
            pe()
        else:
            block(sx0, sx1, sy0, sy1, zp - 1.0, zs_, top="batt", roof="pyr", rh=5.0, cols={"f": scol}, fin=True)
        blev = [(zp + 3.5, 2.4, 0.9)]
        block(pc[0] - 17.0 * S, pc[0] - 11.0 * S, pc[1] + 2.0 * S, pc[1] + 8.0 * S, zp - 1.0, zp + 0.6 * HT, top="batt",
              cols={s_: wcols(6.0 * S, 1, 1.7, blev, 1.5) for s_ in "frl"})
        block(pc[0] - 12.0 * S, pc[0] + 6.0 * S, pc[1] - 9.5 * S, pc[1] - 4.5 * S, zp - 1.0, zp + 6.0, roof="gx", rh=4.2,
              cols={"f": wcols(18.0 * S, 5, 1.7, [(zp + 2.0, 2.4, 0.0)], 1.8)})
        mark("citadel")

    else:   # ridge: a chapel with a west tower on the crest
        block(-6.0 * S, 7.0 * S, -3.5 * S, 3.5 * S, zb1, z0 + 6.5, roof="gx", rh=6.0 * S,
              cols={"f": wcols(13.0 * S, 3, 1.6, [(z0 + 2.0, 2.6, 1.0)], 1.6), "b": wcols(13.0 * S, 3, 1.6, [(z0 + 2.0, 2.6, 1.0)], 1.6),
                    "r": wcols(7.0 * S, 1, 1.7, [(z0 + 2.4, 3.0, 1.2)], 2.0)})
        tower(-8.5 * S, 0.0, 2.9 * S, zb1, z0, z0 + 0.6 * HT, n=4, sh=0.45 * HT, maxlev=2, pw=0.8, boost=1.6, belfry=2.6, fin=True)
        mark("chapel")

    slopes()
    settle()
    return B, tris(), MK[1], CNT


def build(p, rng):
    for det in range(5):        # the summit gives way until hill, hamlet and woodland fit in full
        B, nt, marks, cnt = _gen(p, rng, det)
        if nt <= TRI_CAP:
            break
    hz, fl, gl = float(p.haze), float(p.floodlit), float(p.glow)

    def rgb(c):
        return [int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]

    def hx(c, extra=0.0):
        return _mix(c, HZC, min(1.0, 0.7 * hz + extra))

    def muted(c, f):
        """Colour at a fraction f of its saturation."""
        r, g, b = rgb(c)
        y = 0.3 * r + 0.59 * g + 0.11 * b
        return "#" + "".join(f"{int(round(255 * (y + (v - y) * f))):02x}" for v in (r, g, b))

    def level(c, lum):
        """Colour scaled to the (linear) luminance lum: every roof colour gives the same depth of slate."""
        lin = [v ** 2.2 for v in rgb(c)]
        f = lum / max(1e-4, 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])
        return "#" + "".join(f"{int(round(255 * min(1.0, v * f) ** (1 / 2.2))):02x}" for v in lin)

    wr = rgb(p.wall)
    msn = p.masonry if p.masonry != "auto" else ("ashlar" if max(wr) - min(wr) < 0.1 else "dressed")
    lib, kf, tsc = {"ashlar": ("ashlar_limestone", K_ASH, 0.5), "dressed": ("stone_smooth", K_DRS, 0.3),
                    "rough": ("stone_rough", K_RGH, 0.3), "render": ("plaster_lime", K_RND, 0.5)}[msn]
    tl = None if msn == "ashlar" else hx(p.wall)
    ts = None if msn == "ashlar" else hx(_mix(p.wall, "#7f97e0", 0.4))
    rock = hx("#8f97ad", -0.1)
    mats = {
        "wl": lambda: mat(lib, tl, glow=kf * fl, scale=tsc),
        "w0": lambda: mat(lib, tl, glow=kf * W_FLANK * fl, scale=tsc),
        "w1": lambda: mat(lib, ts, glow=kf * W_SHADE * fl, scale=tsc),
        "roof": lambda: mat("slate", level(hx(muted(p.roof, 0.6)), 0.085), glow=K_SLT * fl, scale=0.5),
        "rk": lambda: mat("stone_rough", hx("#aab0c2", -0.1), glow=max(K_RS, K_RK * float(p.valley_fog)) * fl, scale=0.25),
        "rs": lambda: mat("stone_rough", rock, glow=K_RS * fl, scale=0.25),
        "wood": lambda: mat("hedge", "#3f7a5c", glow=K_WOD * fl, scale=0.4),
        "lit": lambda: mat("emissive", _mix(p.lit_color, "#ff8418", 0.5), glow=gl),
        "dim": lambda: mat("emissive", _mix(p.lit_color, "#f06a18", 0.7), glow=0.5 * gl),
        "cool": lambda: mat("emissive", "#ffc27a", glow=min(40.0, 0.55 * gl)),      # dial / rose: scaled with the windows
    }
    for key, mb in B.items():
        if not mb.f:
            continue
        ob = mesh_object("far_" + key, mb.v, mb.f)
        uvl = ob.data.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", [c for fu in mb.uv for q in fu for c in q])
        assign(ob, mats[key]())
        uv_keep(ob)
        if key == "wood":
            smooth(ob, 75.0)
        else:
            flat(ob)
        if key in ("wl", "w0", "w1", "roof", "rk", "rs", "wood") and len(ob.data.vertices) == len(mb.g):
            paint(ob, grime=lambda co, no, g=mb.g: np.asarray(g, dtype=np.float32), mode="set")
    print(f"DBG {p.style} det={det} tris={nt} " + " ".join(marks) + f" cottages={cnt[0]}/{int(p.hamlet)} clumps={cnt[1]}/{int(p.trees)}")
