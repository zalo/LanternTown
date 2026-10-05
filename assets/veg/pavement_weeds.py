"""Pavement weeds: a scatter patch of the small plants that colonise an old paved square - tufts of
meadow grass (every blade a tapered, keeled, arching ribbon with a paler underside, some gone to straw,
a few seed stalks), dandelion rosettes (runcinate back-swept teeth, sunk midrib, browning old leaves;
scapes carrying dense all-yellow heads of strap florets on a green involucre, closed buds, or airy seed
clocks: a dark receptacle, hair-thin achene stalks and little pappus parasols, partly blown away),
broad-leaf plantain with seed spikes, and clumpy cushion moss (clustered lobes on a ragged feathered
skirt, a few sporophyte hairs).  Every clump is ROOTED IN A PAVING JOINT and spills over the stones:
`paving='cobble'` uses the measured joint skeleton of the square's 'cobble_granite' ground material
(1.6 m tile, ground UV = world XY; pass the asset's world position / yaw as tile_x, tile_y, tile_rot),
`paving='courses'` generates regular coursed setts from joint_pitch / joint_len / joint_angle.
Habits: a loose round patch, a drift along a wall foot (wall at the +Y edge) and low growth that
follows the joint lines.  Four materials (leaf, dry paper fibre, moss, gravel); straw / dead brown /
petal yellow / white pappus and light leaf undersides are per-vertex tone, not extra draw calls.
"""
from tk import *
import numpy as np
import zlib
import base64
from math import sin, cos, pi, radians, sqrt, hypot, atan2, floor, ceil

META = {"name": "Weeds and moss tufts", "class": "vegetation", "mount": "ground", "focus": 0.3,
        # every tone mask is painted per vertex here; the automatic convex-edge wear / ground-splash grime
        # would chalk and blacken paper-thin foliage that is all edge and all within splash height
        "auto_wear": 0.0, "auto_grime": 0.0,
        # lod 2 is modelled (docs/LOD.md): beyond 44 m a pixel is 4 cm, so a tuft is a few broad spikes, a rosette a
        # few kites, a flower head or seed clock one small gem, a moss cushion two or three low pyramids
        "lod": "far"}
PARAMS = {
    "spread": {"default": 1.0, "min": 0.5, "max": 1.5, "desc": "patch width along X (m)"},
    "count": {"default": 16, "min": 3, "max": 36, "desc": "number of plant clumps (tufts + rosettes + moss groups)"},
    "grass": {"default": 1.0, "min": 0.0, "max": 1.0, "desc": "kinds mix: share of grass tufts"},
    "dandelion": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "kinds mix: share of dandelion rosettes"},
    "plantain": {"default": 0.25, "min": 0.0, "max": 1.0, "desc": "kinds mix: share of broad-leaf plantain rosettes"},
    "moss": {"default": 0.7, "min": 0.0, "max": 1.0, "desc": "kinds mix: share of moss cushions"},
    "dryness": {"default": 0.15, "min": 0.0, "max": 1.0, "desc": "0 = lush and damp, 1 = parched straw, shrunken moss"},
    "height": {"default": 0.16, "min": 0.05, "max": 0.32, "desc": "length of the longest grass blades (m)"},
    "aspect": {"default": 1.0, "min": 0.2, "max": 1.0, "desc": "patch depth (Y) as a fraction of spread"},
    "habit": {"default": "patch", "choices": ["patch", "wall_base", "joints"],
              "desc": "patch = loose round scatter; wall_base = drift against a wall at the +Y edge; joints = low growth following the joint lines"},
    "flowers": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "amount of flower stalks / seed heads"},
    "litter": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "grit and blown dead leaves caught in the patch"},
    "paving": {"default": "cobble", "choices": ["cobble", "courses"],
               "desc": "joint layout the plants root in: cobble = measured joints of the cobble_granite ground tile; courses = regular coursed setts / flags"},
    "tile_x": {"default": 0.0, "min": -4000.0, "max": 4000.0, "desc": "world X of the asset origin (m): phase of the paving pattern"},
    "tile_y": {"default": 0.0, "min": -4000.0, "max": 4000.0, "desc": "world Y of the asset origin (m): phase of the paving pattern"},
    "tile_rot": {"default": 0.0, "min": -360.0, "max": 360.0, "desc": "world yaw of the asset (deg) so the cobble joint map is looked up correctly"},
    "joint_pitch": {"default": 0.11, "min": 0.05, "max": 0.8, "desc": "paving='courses': course height = spacing of the bed joints (m)"},
    "joint_len": {"default": 0.17, "min": 0.06, "max": 1.6, "desc": "paving='courses': stone length along a course (m), running bond"},
    "joint_angle": {"default": 0.0, "min": -180.0, "max": 180.0, "desc": "paving='courses': direction of the courses relative to asset X (deg)"},
}
VARIANTS = {
    "wall_drift_lush": {"habit": "wall_base", "spread": 1.5, "aspect": 0.32, "count": 30, "height": 0.27, "dryness": 0.0,
                        "grass": 1.0, "dandelion": 0.35, "plantain": 0.3, "moss": 0.6, "flowers": 0.8, "seed": 4},
    "dandelion_bloom": {"spread": 0.8, "count": 11, "grass": 0.25, "dandelion": 1.0, "plantain": 0.1, "moss": 0.3,
                        "flowers": 1.0, "height": 0.12, "dryness": 0.05, "litter": 0.2, "seed": 11},
    "moss_joints": {"habit": "joints", "spread": 1.2, "aspect": 0.8, "count": 30, "grass": 0.35, "dandelion": 0.08,
                    "plantain": 0.1, "moss": 1.0, "height": 0.08, "dryness": 0.0, "flowers": 0.3, "litter": 0.6, "seed": 23},
    "parched_tufts": {"spread": 0.65, "count": 9, "grass": 1.0, "dandelion": 0.25, "plantain": 0.0, "moss": 0.15,
                      "dryness": 0.92, "height": 0.2, "flowers": 0.9, "litter": 0.9, "seed": 37},
}

LEAF_T = 0.25          # tile size of the 'leaf' material (m): one blade / leaf maps onto the whole tile
TILE, CELL = 1.6, 0.008
# Joint skeleton of one 1.6 m tile of the 'cobble_granite' ground material (world-anchored, tile centred on
# the world origin), measured from rectified review renders: 200 x 200 bitmap of 8 mm cells, zlib + base64.
COBBLE_JOINTS = (
    "eNql2D+P3MYVAPA3HGLJgjgSUJEtDiKB5AMwSZELfAhpqFDpjxAiLpwui1QK4BwnUHGlU7oIoK+QMkCMaM4xpHT3BYyIsgM4RYCjosKMzZuX996Q++92ZSfG"
    "QcIuf8eZ4cybN49XgIGialQPhQL6nEEbRo4v0r8GEOs12BDiCWr5mcF4KI9B9p2B+kwgvAs0bBOvoaDLDWTaKPxLIqA2UEB2v43ARigA0OIL66Ht1Qh1CHYk"
    "OAELq6onqNDKrSG0l3iDXyNQY/cjfNInMjcECsozGUkG98asyxI/PAa65KGgJ8riHSgnWDLYNdDXCTIGfkqac82XtiARoIkKNtDwp3AXgP+vPRTQ6DHZvmMD"
    "v73E4RBUVxCm/74Laaeoj9QtzwgsnOrOww8Hw50rjdd6SNvotTtvEQfEG/BAk+vSgaa6iP72A0iBFod+AoIFhNMaxTL6NQQClh98HxJ6aJ7TO8ATdxTsMeCf"
    "LJnB7jUVz1AfBGXUcAhqZaIInXYbKAVshSMsldPPjR8VPWJLoPBj+s2Y5vsRDkBz98UYfX2F+BSlbYYarl+kNZxHwwkFTOfnyIMpZJEomCgitsF6aEIflAyJ"
    "QO2hCKmpQhkG3hrNFnDUTSC74QD4HVYKlHdhHTqHoFyDdu1HOXYzFGs4UebzV3AGHQNFDwxhOkKn/9PzE5sk/xPYTEU3iB/gPzQ+cScgoPsUXyPSnbw2/Nja"
    "Q0Af78sWmYGe2U4Q7wN1ugO03N8Ey2MQ34UFj6qO1RZo6ZyiNB9pB1k0a8ArSJ9SSugjSgyVwc8YhnPKSq3u38ZuAT6DlhXF1b30C+xDcBSG1ThBEeKgkX6t"
    "5lVouBeGhBf0zGXBBMUMIUMODGYLEg+0PMH8zUO8A7GHZB/sBPE+1G+A001TyQQyc/mXvnNlTuUBp8Rb5z2tKYMe3rWnGxh+TQHeMVxEpn09Fok0ZdSTxwDV"
    "RxpStLxVI8TuDBrncl66Irq97Ll33mdZdUPb8/FLAU0xGc5An1KwlA+A+yp2QFLAN0KyA3YD8Q4YD/oYBAfBeFhy7t0AZalb7AI5XrQpafgMCTd0GsHZgg8D"
    "d9G+oAnJ/O6tsFcQhTDqz941HJnVwNGrPqdfoNjF20tH3xmylCI4xnNggHIBwQzwS3g/lsDm4yXYghhyPAL0dXkMsjcC1Se7UPqPIHPioRQo7gLPzPIYZDPQ"
    "adjw5cXUVDb3Eaz4RAngZA00qphmxiXpHwNIzVv2oh2hhxVDPlBTur8XDeUr2/71kwj/oP8OFzTnNDhFx8aQdw+AN12pHaWEKxk1bxraz78A2TccOnJqJwK1"
    "khwSCxiBhQcwiz0IZuDYWE5g1mD9Hb6uMbABOU5jKUaOQE3NbSDbviM7DMUEyzVYgWiYoHIp5fncn8wUN+Piea8vscNO9+jyVyiFXUxTq8fKjaoCPqZ4VR73"
    "q/epBuQui6lSqynzUpmbXz/DT2AbCumcpkTL2s5QbsBHwwzFDEoy/y5oAbgLwS6Ux6CQxEeV2wEIQAop3iVryPmSllPdVyhcePFzrPr2ctRjO9DRqtGl1k98"
    "Vtn0w1G7yEbRn9HBg6/OW8vQdnXso0GNvJHrQNtV/pzSAufnpU89oQCl3+6RFCSxX6kNhFOlQqCOAqxBQ73YhcCDkpveAFTeUIk0w0KaCmgWLMFb44m2enxG"
    "8Bv78CsGaoegiZRrXfT0Fq9sNaRZw9mv5mOYixYaygIuoMgU4nX3/Zf4cczjlujjGSk4GFbv/SznHekhlEE1UgSpf8lWpVbWMFVHoYfg20M856oJFhsodiD8"
    "VpAdhnIHpj7iqRwp9kd1EC4oeGYotR6VjXoGfPysWUO2GCM8/zS/pomlsP2p4wPgezaDR0sKOF1SPUwvTRoi/Ta6H430xnbd+ZfRdVHZwI9vf08nRwB74HN0"
    "lhyE5TGgcYV7kEwQr2HpIZ5h3VS8B3EgqfMI1P832Lzbh4FBY38eS16ljRnz0G1E542pLujcpoowHeStNMUv0ZxDsUJLxwW9umGfRi7ilFAA/lNyieQRKEoY"
    "9BAZznkm3obMv3JmM9gZwAPnMQGzB4Wv9I6DmqEBTliLGRou75QvbhjCGfhinaIzUmmvIUEq9+DBkOgRcFRdS9PRIiXRFH7+gVOOvj4czHuYX9FLvqUIpOKu"
    "Hh44PmKb1PgXOl9kynPLFinmN73DYL4bNIfBSuW4Az4szRbUagKaQvkLRqIcjrpT6Ki+gDMubKsWxwpfO2UeugoLqnmH9maEuKJwemh9QWo4gBYQXRkpJ2LZ"
    "4NJvLAOXsyb23/wyTqD3wG7dUW5DfaypWu2A2gI+2WbItu+QgzjZg9KDkapoG7L/AZIdqCozw1NKVqe+j0cVXnWrkGM1Ux828LvhJ3LsRpf8cp+jvPtT0IFJ"
    "X3YFvENxOP3Fi0qUX1n4L3HQCH0="
)


def _mix_hex(a, b, t):
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(ca[i] + (cb[i] - ca[i]) * t)) for i in range(3))


def _hash(i, j):
    v = sin(i * 12.9898 + j * 78.233) * 43758.5453
    return v - floor(v)


def _cobble_tile():
    """Joint points of the cobble tile: x, y (tile coords, m), tangent angle, junction flag."""
    m = int(round(TILE / CELL))
    bits = np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(COBBLE_JOINTS)), np.uint8))[:m * m]
    bm = bits.reshape(m, m).astype(np.float64)
    sxx, sxy, syy = np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, m))
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx or dy:
                nb = np.roll(bm, (-dy, -dx), (0, 1))
                sxx += nb * dx * dx
                sxy += nb * dx * dy
                syy += nb * dy * dy
    iy, ix = np.nonzero(bm > 0.5)
    a, b, c = sxx[iy, ix], sxy[iy, ix], syy[iy, ix]
    ang = 0.5 * np.arctan2(2 * b, a - c)
    rt = np.sqrt(((a - c) * 0.5) ** 2 + b * b)
    lmax, lmin = (a + c) * 0.5 + rt, (a + c) * 0.5 - rt
    junc = lmin / np.maximum(lmax, 1e-6) > 0.2
    return (ix + 0.5) * CELL - TILE / 2, (iy + 0.5) * CELL - TILE / 2, ang, junc


def _joint_points(p, R):
    """Joint points around the asset in ASSET-LOCAL coordinates: x, y, tangent angle, junction flag."""
    cx, cy = float(p.tile_x), float(p.tile_y)
    X, Y, A, J = [], [], [], []
    if p.paving == "cobble":
        rot = radians(float(p.tile_rot))
        tx, ty, ta, tj = _cobble_tile()
        for k in range(int(floor((cx - R + TILE / 2) / TILE)), int(floor((cx + R + TILE / 2) / TILE)) + 1):
            for l in range(int(floor((cy - R + TILE / 2) / TILE)), int(floor((cy + R + TILE / 2) / TILE)) + 1):
                wx, wy = tx + TILE * k - cx, ty + TILE * l - cy
                m = wx * wx + wy * wy < R * R
                X.append(wx[m]); Y.append(wy[m]); A.append(ta[m]); J.append(tj[m])
    else:
        rot = -radians(float(p.joint_angle))
        pitch, ln = float(p.joint_pitch), float(p.joint_len)
        for j in range(int(floor((cy - R) / pitch)), int(ceil((cy + R) / pitch)) + 1):
            xs = np.arange(cx - R, cx + R, CELL)
            X.append(xs - cx); Y.append(j * pitch + 0.003 * np.sin(xs * 7.0 + j * 1.7) - cy)
            A.append(np.zeros(len(xs))); J.append(np.zeros(len(xs), bool))
            for i in range(int(floor((cx - R) / ln)) - 1, int(ceil((cx + R) / ln)) + 1):
                xj = (i + 0.5 * (j % 2) + 0.3 * (_hash(i, j) - 0.5)) * ln
                ys = np.arange(j * pitch + CELL * 0.5, (j + 1) * pitch, CELL)
                X.append(np.full(len(ys), xj - cx)); Y.append(ys - cy)
                A.append(np.full(len(ys), pi / 2))
                J.append((ys - j * pitch < 0.013) | ((j + 1) * pitch - ys < 0.013))
    X, Y, A, J = np.concatenate(X), np.concatenate(Y), np.concatenate(A), np.concatenate(J)
    c, s = cos(-rot), sin(-rot)
    lx, ly = X * c - Y * s, X * s + Y * c
    m = lx * lx + ly * ly < R * R
    return lx[m], ly[m], (A - rot)[m], J[m]


class Buf:
    """Accumulates verts / UVs / per-vertex wear + grime tone / faces of thousands of small pieces into one mesh."""

    def __init__(self):
        self.V, self.UV, self.W, self.G, self.F, self.n, self.tris = [], [], [], [], [], 0, 0

    def add(self, v, uv, faces, wear=0.0, grime=0.0):
        v = np.asarray(v, float).reshape(-1, 3)
        k, o = len(v), self.n
        self.V.append(v)
        self.UV.append(np.broadcast_to(np.asarray(uv, float), (k, 2)))
        self.W.append(np.broadcast_to(np.asarray(wear, float), (k,)))
        self.G.append(np.broadcast_to(np.asarray(grime, float), (k,)))
        for f in faces:
            self.F.append(tuple(int(i) + o for i in f))
            self.tris += len(f) - 2
        self.n += k

    def obj(self, name, material, sm=40.0, ymax=None):
        if not self.F:
            return None
        V = np.concatenate(self.V)
        UV = np.concatenate(self.UV)
        V[:, 2] = np.maximum(V[:, 2], 0.0)
        if ymax is not None:
            V[:, 1] = np.minimum(V[:, 1], ymax)
        o = mesh_object(name, V, self.F)
        me = o.data
        idx = np.zeros(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", idx)
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", UV[idx].astype(np.float32).ravel())
        uv_keep(o)
        assign(o, material)
        smooth(o, sm)
        w = np.clip(np.concatenate(self.W), 0.0, 1.0)
        g = np.clip(np.concatenate(self.G), 0.0, 1.0)
        if len(me.vertices) == len(w):
            paint(o, wear=lambda co, n: w, grime=lambda co, n: g, mode="set")
        else:
            print("WARN pavement_weeds: vertex count changed, tone masks skipped:", name, len(me.vertices), len(w))
        return o


def _norm(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


# lod 2: build() sets "on", and every primitive below then emits its far stand-in. The random draws are the same
# at every level; what is left out is chosen by counting (_far_keep), never by the stream.
_FAR = {"on": False, "n": 0, "out": []}


def _far_keep(every):
    _FAR["n"] += 1
    return _FAR["n"] % every == 0


def _far_out(lo, hi, emit):
    """lod 2: note a part that is left out (its extent and how to build its stand-in); _far_outline puts back the
    few that carry the outline of the patch."""
    _FAR["out"].append((np.asarray(lo, float), np.asarray(hi, float), emit))


def _far_outline(bufs, tol=0.03):
    """lod 2: the footprint and height of the patch must stay. Where everything that was kept ends more than tol
    short of a part that was left out, that part comes back as its plain stand-in (at most five of them)."""
    out = _FAR["out"]
    V = [np.concatenate(b.V) for b in bufs.values() if b.F]
    if not out:
        return
    lo = np.min([v.min(0) for v in V], 0) if V else np.full(3, 1e9)
    hi = np.max([v.max(0) for v in V], 0) if V else np.full(3, -1e9)
    for ax, up in ((0, False), (0, True), (1, False), (1, True), (2, True)):
        k = max(range(len(out)), key=lambda i: out[i][1][ax] if up else -out[i][0][ax])
        if (out[k][1][ax] > hi[ax] + tol) if up else (out[k][0][ax] < lo[ax] - tol):
            out[k][2]()
            lo, hi = np.minimum(lo, out[k][0]), np.maximum(hi, out[k][1])


def _ribbon_far(buf, P, WL, WR, S, N, fold, floor, ur, wear, wear_e, grime_e, grime_m, under, gain):
    """lod 2 stand-in of a ribbon: a grass blade is one ridged spike from its foot to its tip (3 triangles), a
    broad leaf a kite through its widest ring with its own pale underside (4 triangles). gain widens it to stand
    for the neighbours that were left out."""
    n = len(P)

    def col(a):
        return np.broadcast_to(np.asarray(a, float), (n,))

    WLc, WRc, w, ge, gm = col(WL) * gain, col(WR) * gain, col(wear), col(grime_e), col(grime_m)
    we = w if wear_e is None else col(wear_e)
    u0, u1 = ur[0] * LEAF_T, ur[1] * LEAF_T
    um = 0.5 * (u0 + u1)
    v0, v1 = 0.03 * LEAF_T, 0.97 * LEAF_T
    if under is None:
        hl, hr = 1.5 * float(WLc.mean()), 1.5 * float(WRc.mean())     # a triangle with the area of the blade
        V = np.stack([P[0] - S[0] * hl, P[0] + N[0] * float(col(fold)[0]) * gain, P[0] + S[0] * hr, P[-1]])
        V[:, 2] = np.maximum(V[:, 2], floor)
        # (the tone of the fold all over: the pale margins of one broad spike would outshine the tuft it stands for,
        # whose blades shade each other)
        g0 = max(float(ge[0]), float(gm[0])) + 0.12
        buf.add(V, [[v0, u0], [v0, um], [v0, u1], [v1, um]], [(0, 1, 3), (1, 2, 3), (2, 0, 3)],
                wear=[w[0], w[0], w[0], w[-1]], grime=[g0, g0, g0, max(float(ge[-1]), float(gm[-1]))])
        return
    j = 1 + int(np.argmax((WLc + WRc)[1:-1])) if n > 2 else 0
    wu = col(under)
    vj = (0.03 + 0.94 * j / max(n - 1, 1)) * LEAF_T
    K = np.stack([P[0], P[j] - S[j] * WLc[j], P[j] + S[j] * WRc[j], P[-1]])
    K[:, 2] = np.maximum(K[:, 2], floor)
    uv = [[v0, um], [vj, u0], [vj, u1], [v1, um]]
    buf.add(np.vstack([K, K]), uv + uv, [(0, 2, 3), (0, 3, 1), (4, 7, 6), (4, 5, 7)],
            wear=[w[0], we[j], we[j], we[-1], wu[0], wu[j], wu[j], wu[-1]],
            grime=[gm[0], ge[j], ge[j], ge[-1]] + [ge[0] * 0.5, ge[j] * 0.5, ge[j] * 0.5, ge[-1] * 0.5])


def _gem(buf, c, axv, A, Bv, rad, up, dn, wear=0.0, grime=0.0):
    """lod 2: a flower head / seed clock as one closed gem (4-sided double pyramid, 8 triangles)."""
    c = np.asarray(c, float)
    V = np.stack([c + A * rad, c + Bv * rad, c - A * rad, c - Bv * rad, c + axv * up, c - axv * dn])
    t = 0.5 * LEAF_T
    buf.add(V, [[t, t]] * 6, [(i, (i + 1) % 4, 4) for i in range(4)] + [((i + 1) % 4, i, 5) for i in range(4)], wear=wear, grime=grime)


def _path(base, ts, elev, phi, L):
    """Integrate a path whose direction has elevation elev(t) (rad above horizontal) and azimuth phi(t)."""
    dirs = np.stack([np.cos(elev) * np.cos(phi), np.cos(elev) * np.sin(phi), np.sin(elev)], 1)
    seg = 0.5 * (dirs[1:] + dirs[:-1]) * (ts[1:] - ts[:-1])[:, None] * L
    return np.asarray(base, float) + np.vstack([np.zeros((1, 3)), np.cumsum(seg, axis=0)])


def _frames(P, s_ref, twist=None):
    T = _norm(np.gradient(P, axis=0))
    s_ref = np.asarray(s_ref, float)
    S = _norm(s_ref[None, :] - (T @ s_ref)[:, None] * T)
    N = np.cross(S, T)
    if twist is not None:
        c, s = np.cos(twist)[:, None], np.sin(twist)[:, None]
        S = S * c + N * s
        N = np.cross(S, T)
    return T, S, N


def ribbon(buf, P, WL, WR, S, N, fold, ts, lift_l=0.0, lift_r=0.0, floor=0.0008, ur=(0.12, 0.88),
           T=None, back_l=0.0, back_r=0.0, wear=0.0, wear_e=None, grime_e=0.0, grime_m=0.0, under=None, far=1.0):
    """Leaf / blade along P ending in a point: an upper skin of two rows of quads L-P-R and a flat underside
    of triangles between the margins.  The AO baked at export looks along the vertex normals, so the layout
    decides how thin foliage is lit from behind:
      under=None  (grass): midrib RAISED by `fold` (ridged blade) and the underside reuses the margin verts ->
                  their normals point sideways out of the blade, neither side of it bakes black;
      under=w     (broad leaves, midrib sunk by lift_*): the underside has its own verts with wear tone w
                  (pale: it only ever sees the ground), the cupped upper skin keeps clean up-facing normals.
    Tone per ring: wear / grime_m on the midrib, wear_e / grime_e on the margins.
    far: lod 2 only - width gain of the stand-in (_ribbon_far); 0 leaves the ribbon out."""
    if _FAR["on"]:
        if far:
            _ribbon_far(buf, P, WL, WR, S, N, fold, floor, ur, wear, wear_e, grime_e, grime_m, under, far)
        else:
            _far_out(np.min(P, 0), np.max(P, 0), lambda: _ribbon_far(buf, P, WL, WR, S, N, fold, floor, ur, wear, wear_e, grime_e, grime_m, under, 1.0))
        return
    n = len(P)

    def col(a):
        return np.broadcast_to(np.asarray(a, float), (n,))[:, None]

    Lp = P - S * col(WL) + N * col(lift_l)
    Rp = P + S * col(WR) + N * col(lift_r)
    if T is not None:
        Lp = Lp - T * col(back_l)
        Rp = Rp - T * col(back_r)
    Pm = P + N * col(fold)
    u0, u1 = ur[0] * LEAF_T, ur[1] * LEAF_T          # across the blade (tile V), midrib at the centre
    um = 0.5 * (u0 + u1)
    w, ge, gm = col(wear), col(grime_e), col(grime_m)
    we = w if wear_e is None else col(wear_e)
    m = n - 1
    vv = (0.03 + 0.94 * np.asarray(ts, float)) * LEAF_T
    k = 3 if under is None else 5
    if under is None:
        rings = np.stack([Lp, Pm, Rp], 1)
        us = [u0, um, u1]
        wr, gr = np.concatenate([we, w, we], 1), np.concatenate([ge, gm, ge], 1)
        bl, br = 0, 2
    else:
        wu = col(under)
        rings = np.stack([Lp, Pm, Rp, Lp, Rp], 1)
        us = [u0, um, u1, u0, u1]
        wr, gr = np.concatenate([we, w, we, wu, wu], 1), np.concatenate([ge, gm, ge, ge * 0.5, ge * 0.5], 1)
        bl, br = 3, 4
    V = np.vstack([rings[:m].reshape(-1, 3), P[-1:], P[-1:]])
    V[:, 2] = np.maximum(V[:, 2], floor)
    uv = np.vstack([np.stack([np.repeat(vv[:m], k), np.tile(np.array(us), m)], 1), [[vv[-1], um]], [[vv[-1], um]]])
    faces = []
    for i in range(m - 1):
        a0, a1 = k * i, k * i + k
        faces += [(a0, a0 + 1, a1 + 1, a1), (a0 + 1, a0 + 2, a1 + 2, a1 + 1), (a0 + br, a0 + bl, a1 + bl), (a0 + br, a1 + bl, a1 + br)]
    a0, apex = k * (m - 1), k * m
    faces += [(a0, a0 + 1, apex), (a0 + 1, a0 + 2, apex), (a0 + br, a0 + bl, apex + (0 if under is None else 1))]
    buf.add(V, uv, faces, wear=np.concatenate([wr[:m].ravel(), we[-1], wr[-1, -1:]]),
            grime=np.concatenate([gr[:m].ravel(), ge[-1], ge[-1]]))


def strap(buf, base, T0, N0, L, w, n=5, curl=0.0, thick=None, shape=0.7, wear=0.0, grime=0.0):
    """Blown dead leaf: lanceolate cupped strap starting at base along T0, curling towards N0."""
    T0 = _norm(np.asarray(T0, float))
    N0 = np.asarray(N0, float)
    N0 = _norm(N0 - T0 * float(N0 @ T0))
    ts = np.linspace(0.0, 1.0, n)
    ang = curl * ts
    dirs = np.cos(ang)[:, None] * T0[None, :] + np.sin(ang)[:, None] * N0[None, :]
    seg = 0.5 * (dirs[1:] + dirs[:-1]) * (L / (n - 1))
    P = np.asarray(base, float) + np.vstack([np.zeros((1, 3)), np.cumsum(seg, axis=0)])
    T = _norm(np.gradient(P, axis=0))
    S = np.tile(np.cross(T0, N0), (n, 1))
    N = np.cross(S, T)
    W = w * np.sin(pi * (0.14 + 0.86 * ts)) ** shape + w * 0.03
    ribbon(buf, P, W, W, S, N, 0.0, ts, lift_l=W * 0.22, lift_r=W * 0.22, wear=wear, grime_e=grime,
           grime_m=np.minimum(1.0, np.asarray(grime) + 0.15), under=np.minimum(1.0, np.asarray(wear) + 0.25), far=0.0)


def tube(buf, P, rad, sides=4, ur=(0.3, 0.7), wear=0.0, grime=0.0, point=False, far=2.0):
    """Round stalk / bud / spike along P with per-ring radius (point=True: the last ring is a single apex vertex).
    far: lod 2 only - radius gain of the stand-in (a 3-sided spike from the foot to the tip; point=True: a spindle
    through the widest ring), 0 leaves the tube out, "keep" builds it as given."""
    P = np.asarray(P, float)
    n = len(P)
    rad = np.broadcast_to(np.asarray(rad, float), (n,))
    T = _norm(np.gradient(P, axis=0))
    ref = np.where(np.abs(T[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    A = _norm(ref - (T * ref).sum(1, keepdims=True) * T)
    Bv = np.cross(T, A)
    if _FAR["on"] and far != "keep":
        def spike(gain):
            k = int(np.argmax(rad)) if point else 0
            a = np.arange(3) * (2 * pi / 3)
            V = [P[k] + (np.cos(a)[:, None] * A[k] + np.sin(a)[:, None] * Bv[k]) * rad[k] * gain, P[-1:]]
            faces = [(0, 1, 3), (1, 2, 3), (2, 0, 3)]
            if k > 0:
                V.append(P[:1])
                faces += [(1, 0, 4), (2, 1, 4), (0, 2, 4)]
            wv, gv = np.broadcast_to(np.asarray(wear, float), (n,)), np.broadcast_to(np.asarray(grime, float), (n,))
            ix = [k, k, k, n - 1] + ([0] if k > 0 else [])
            buf.add(np.vstack(V), [[(0.05 + 0.9 * i / max(n - 1, 1)) * LEAF_T, 0.5 * LEAF_T] for i in ix], faces, wear=wv[ix], grime=gv[ix])

        if far:
            spike(far)
        else:
            _far_out(P.min(0), P.max(0), lambda: spike(2.0))
        return T
    a = np.arange(sides) * (2 * pi / sides)
    ring = (np.cos(a)[None, :, None] * A[:, None, :] + np.sin(a)[None, :, None] * Bv[:, None, :]) * rad[:, None, None]
    m = n - 1 if point else n
    V = (P[:m, None, :] + ring[:m]).reshape(-1, 3)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    vv = (0.05 + 0.9 * d / max(d[-1], 1e-6)) * LEAF_T
    uu = (ur[0] + (ur[1] - ur[0]) * np.arange(sides) / sides) * LEAF_T
    uv = np.stack([np.repeat(vv[:m], sides), np.tile(uu, m)], 1)
    w = np.repeat(np.broadcast_to(np.asarray(wear, float), (n,))[:m], sides)
    g = np.repeat(np.broadcast_to(np.asarray(grime, float), (n,))[:m], sides)
    faces = []
    for i in range(m - 1):
        for k in range(sides):
            k1 = (k + 1) % sides
            faces.append((i * sides + k, i * sides + k1, (i + 1) * sides + k1, (i + 1) * sides + k))
    if point:
        V = np.vstack([V, P[-1:]])
        uv = np.vstack([uv, [[vv[-1], 0.5 * LEAF_T]]])
        w = np.concatenate([w, np.broadcast_to(np.asarray(wear, float), (n,))[-1:]])
        g = np.concatenate([g, np.broadcast_to(np.asarray(grime, float), (n,))[-1:]])
        a0 = (m - 1) * sides
        for k in range(sides):
            faces.append((a0 + k, a0 + (k + 1) % sides, m * sides))
    buf.add(V, uv, faces, wear=w, grime=g)
    return T


def tris2(buf, A, Bp, C, wear=0.0, grime=0.0, uv=None):
    """Two-sided triangles (hairs, spikelets, bracts): corners A, B, C are (m, 3); tone per corner (m, 3) or scalar.
    Not in the far level (lod 2)."""
    if _FAR["on"]:
        return
    A, Bp, C = (np.asarray(v, float).reshape(-1, 3) for v in (A, Bp, C))
    m = len(A)
    V = np.stack([A, Bp, C], 1).reshape(-1, 3)
    w = np.broadcast_to(np.asarray(wear, float), (m, 3)).reshape(-1)
    g = np.broadcast_to(np.asarray(grime, float), (m, 3)).reshape(-1)
    if uv is None:
        uv = np.array([[0.2, 0.42], [0.2, 0.58], [0.8, 0.5]]) * LEAF_T
    uvs = np.tile(np.asarray(uv, float), (m, 1))
    faces = [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(m)] + [(3 * m + 3 * i + 2, 3 * m + 3 * i + 1, 3 * m + 3 * i) for i in range(m)]
    buf.add(np.vstack([V, V]), np.vstack([uvs, uvs]), faces, wear=np.concatenate([w, w]), grime=np.concatenate([g, g]))


def quads(buf, A, Bp, C, Dp, wear=0.0, grime=0.0, two_sided=True):
    """Flat quads A-B-C-D (m, 3 each; strap florets), optionally with a separate back skin; tone per corner (m, 4).
    Not in the far level (lod 2): flower_head puts one gem there."""
    if _FAR["on"]:
        return
    A, Bp, C, Dp = (np.asarray(v, float).reshape(-1, 3) for v in (A, Bp, C, Dp))
    m = len(A)
    V = np.stack([A, Bp, C, Dp], 1).reshape(-1, 3)
    w = np.broadcast_to(np.asarray(wear, float), (m, 4)).reshape(-1)
    g = np.broadcast_to(np.asarray(grime, float), (m, 4)).reshape(-1)
    uvs = np.tile(np.array([[0.1, 0.4], [0.1, 0.6], [0.9, 0.6], [0.9, 0.4]]) * LEAF_T, (m, 1))
    faces = [(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3) for i in range(m)]
    if two_sided:
        faces += [(4 * m + 4 * i + 3, 4 * m + 4 * i + 2, 4 * m + 4 * i + 1, 4 * m + 4 * i) for i in range(m)]
        V, uvs, w, g = np.vstack([V, V]), np.vstack([uvs, uvs]), np.concatenate([w, w]), np.concatenate([g, g])
    buf.add(V, uvs, faces, wear=w, grime=g)


def lobe(buf, r, cx, cy, rx, ry, rot, h, nu=6, mid=False, wear0=0.0, crown=0.3, z0=0.0006, foot=0.12, far=1.0):
    """One tufted moss lobe / pebble: an irregular low-poly dome (ragged base ring, 1-2 rings, off-centre apex).
    far: lod 2 only - size gain (or (along, across)) of the stand-in, a 4-sided pyramid; 0 leaves the lobe out."""
    th = np.arange(nu) * (2 * pi / nu) + r.u(0, 6.3)
    jit = np.array([r.u(0.76, 1.24) for _ in range(nu)])
    levels = [(1.0, 0.0), (0.82, 0.48), (0.46, 0.87)] if mid else [(1.0, 0.0), (0.62, 0.72)]
    cr, sr = cos(rot), sin(rot)
    ox, oy = r.u(-0.2, 0.2) * rx, r.u(-0.2, 0.2) * ry
    X, Y, Z, W, G = [], [], [], [], []
    for li, (rho, zf) in enumerate(levels):
        lx = rho * rx * jit * np.cos(th) + ox * (1.0 - rho)
        ly = rho * ry * jit * np.sin(th) + oy * (1.0 - rho)
        lump = np.array([r.u(-0.24, 0.24) for _ in range(nu)]) if li else np.zeros(nu)
        if li == 0:
            l0 = (lx, ly)
        X.append(cx + lx * cr - ly * sr)
        Y.append(cy + lx * sr + ly * cr)
        Z.append(z0 + h * zf * (1.0 + lump))
        W.append(np.full(nu, wear0 + crown * zf))
        G.append(np.full(nu, foot * (1.0 - zf) ** 2))
    if _FAR["on"]:
        uo, vo = r.u(0, 1), r.u(0, 1)

        def pyramid(ga, gb):
            ix = (np.arange(4) * nu) // 4
            fx, fy = np.append(l0[0][ix] * ga * 1.12, ox), np.append(l0[1][ix] * gb * 1.12, oy)
            X, Y = cx + fx * cr - fy * sr, cy + fx * sr + fy * cr
            buf.add(np.stack([X, Y, [z0] * 4 + [z0 + h]], 1), np.stack([X + uo, Y + vo], 1), [(i, (i + 1) % 4, 4) for i in range(4)],
                    wear=[wear0] * 4 + [wear0 + crown], grime=[foot] * 4 + [0.0])

        if far:
            pyramid(*(far if isinstance(far, tuple) else (far, far)))
        else:
            q = max(rx, ry)
            _far_out((cx - q, cy - q, z0), (cx + q, cy + q, z0 + h), lambda: pyramid(1.0, 1.0))
        return
    X.append([cx + ox * cr - oy * sr]); Y.append([cy + ox * sr + oy * cr]); Z.append([z0 + h]); W.append([wear0 + crown]); G.append([0.0])
    X, Y, Z, W, G = np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(W), np.concatenate(G)
    faces = []
    nl = len(levels)
    for li in range(nl - 1):
        a0, a1 = li * nu, (li + 1) * nu
        for i in range(nu):
            i1 = (i + 1) % nu
            faces.append((a0 + i, a0 + i1, a1 + i1, a1 + i))
    a0 = (nl - 1) * nu
    for i in range(nu):
        faces.append((a0 + i, a0 + (i + 1) % nu, nl * nu))
    uo, vo = r.u(0, 1), r.u(0, 1)
    buf.add(np.stack([X, Y, Z], 1), np.stack([X + uo, Y + vo], 1), faces, wear=W, grime=G)


def fan(buf, r, cx, cy, ra, rb, rot, hc, nu=9, ragged=(0.6, 1.3), rings=1, wear_c=0.0, wear_r=0.0, zr=0.0006, far=1.0):
    """Thin ragged smear feathering out to nothing at an irregular rim (moss skirt, soil drift).
    far: lod 2 only - size gain of the stand-in (4 triangles); 0 leaves the smear out."""
    th = (np.arange(nu) + np.array([r.u(-0.3, 0.3) for _ in range(nu)])) * (2 * pi / nu) + r.u(0, 6.3)
    rad = np.array([r.u(*ragged) * (1.0 if i % 2 else 0.72) for i in range(nu)])
    cr, sr = cos(rot), sin(rot)
    lx, ly = rad * ra * np.cos(th), rad * rb * np.sin(th)
    if _FAR["on"]:
        uo, vo = r.u(0, 1), r.u(0, 1)

        def smear(gain):
            ix = (np.arange(4) * nu) // 4
            fx, fy = np.append(0.0, lx[ix] * gain), np.append(0.0, ly[ix] * gain)
            X, Y = cx + fx * cr - fy * sr, cy + fx * sr + fy * cr
            buf.add(np.stack([X, Y, [hc] + [zr] * 4], 1), np.stack([X + uo, Y + vo], 1), [(0, 1 + i, 1 + (i + 1) % 4) for i in range(4)],
                    wear=[wear_c] + [wear_r] * 4)

        if far:
            smear(far)
        else:
            q = max(ra, rb)
            _far_out((cx - q, cy - q, zr), (cx + q, cy + q, hc), lambda: smear(1.0))
        return
    X, Y, Z, W = [[cx]], [[cy]], [[hc]], [[wear_c]]
    fr = [0.55, 1.0] if rings == 2 else [1.0]
    for f in fr:
        X.append(cx + (lx * cr - ly * sr) * f); Y.append(cy + (lx * sr + ly * cr) * f)
        Z.append(np.full(nu, zr if f == 1.0 else zr + hc * 0.55)); W.append(np.full(nu, wear_r if f == 1.0 else 0.5 * (wear_c + wear_r)))
    X, Y, Z, W = np.concatenate(X), np.concatenate(Y), np.concatenate(Z), np.concatenate(W)
    faces = [(0, 1 + i, 1 + (i + 1) % nu) for i in range(nu)]
    if rings == 2:
        for i in range(nu):
            i1 = (i + 1) % nu
            faces.append((1 + i, 1 + nu + i, 1 + nu + i1, 1 + i1))
    uo, vo = r.u(0, 1), r.u(0, 1)
    buf.add(np.stack([X, Y, Z], 1), np.stack([X + uo, Y + vo], 1), faces, wear=W)


T3, T4, T5 = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.36, 0.7, 1.0]), np.array([0.0, 0.27, 0.52, 0.77, 1.0])


def build(p, rng):
    d = float(p.dryness)
    S = float(p.spread)
    D = S * float(p.aspect)
    ax_, ay_ = S / 2, D / 2
    habit = p.habit
    wall = habit == "wall_base"
    H = float(p.height)
    _FAR.update(on=lod() == 2, n=0, out=[])
    far = _FAR["on"]

    # ---------------------------------------------------------------- materials (4 draw calls)
    # leaf: one mid dandelion green (olive when dry); deeper / paler / underside tones are vertex grime + wear.
    # dry : paper fibre tinted petal yellow; wear bleaches it to straw and on to white pappus, grime browns it.
    M = {
        "leaf": mat("leaf", _mix_hex("#8c9c58", "#a89c5c", d * 0.8), rough=1.3),
        "dry": mat("paper", "#ffc412"),
        "moss": mat("moss", "#ece6a6"),
        "soil": mat("gravel", "#e2ceb0"),
    }
    B = {k: Buf() for k in M}
    LF, DR, MO, SO = B["leaf"], B["dry"], B["moss"], B["soil"]

    def tot():
        return sum(b.tris for b in B.values())

    # ---------------------------------------------------------------- paving joints
    JX, JY, JA, JJ = _joint_points(p, hypot(ax_, ay_) + 0.15)
    back = ay_ - 0.035
    depth = max(D - 0.07, 0.05)
    ywall = ay_ - 0.012                       # the joint between paving and wall
    if wall:
        keep = JY < ywall - 0.02              # no paving joints under / behind the wall
        xs = np.arange(-ax_, ax_, CELL)
        JX, JY = np.concatenate([JX[keep], xs]), np.concatenate([JY[keep], np.full(len(xs), ywall)])
        JA, JJ = np.concatenate([JA[keep], np.zeros(len(xs))]), np.concatenate([JJ[keep], np.zeros(len(xs), bool)])

    def snap(x, y, junc=0.0):
        """Nearest joint point (junctions of three joints attract rosettes): x, y, joint direction."""
        if not len(JX):
            return x, y, 0.0
        d2 = (JX - x) ** 2 + (JY - y) ** 2
        if junc > 0.0:
            d2 = np.where(JJ, d2 * (1.0 - junc), d2)
        i = int(np.argmin(d2))
        return float(JX[i]), float(JY[i]), float(JA[i])

    def faz(a, y):
        """Growth azimuth: plants at the wall foot lean along / away from the wall, never into it."""
        if not wall or y < ywall - 0.05:
            return a
        sy = sin(a)
        if sy > 0.12:
            sy = 0.12 - (sy - 0.12) * 0.7
        return atan2(sy, cos(a))

    def nrng(r):
        return np.random.default_rng(int(r.u(0, 2 ** 31 - 1)))

    # ---------------------------------------------------------------- plant builders
    def blade(r, base, phi, th0, bend, L, w0, zf, dry, deep, short=False):
        n = 3 if (short or L < 0.055) else (4 if L < 0.125 else 5)
        ts = (T3, T4, T5)[n - 3]
        elev = pi / 2 - (th0 + bend * ts ** 1.5)
        ph = phi + r.u(-0.5, 0.5) * ts
        P = _path(base, ts, elev, ph, L)
        P[:, 2] = np.maximum(P[:, 2], zf)
        T, S_, N_ = _frames(P, (sin(phi), -cos(phi), 0.0), twist=r.u(-1.3, 1.3) * ts)
        W = w0 * np.minimum(1.0, 0.6 + ts / 0.2 * 0.4) * (1.0 - ts ** 2.2) ** 0.85 + 0.0002
        fg = (4.0 if _far_keep(5) else 0.0) if far else 1.0       # lod 2: one blade in five, four times as broad
        if dry:
            tone, gr = r.u(0.4, 0.62), r.u(0.0, 0.22)
            ribbon(DR, P, W, W, S_, N_, W * 0.45, ts, floor=zf * 0.5, wear=tone, wear_e=tone + 0.08, grime_e=gr, grime_m=gr, far=fg)
        else:
            g = np.clip(0.2 - ts / 0.3 * 0.2, 0.0, 1.0) + (0.14 if deep else 0.04)        # damp shaded base
            w = 0.04 + 0.12 * ts + 0.3 * d * ts ** 2                                     # tips bleach as it dries
            # folded blade: deeper green along the fold, paler margins / underside (stands in for translucency)
            ribbon(LF, P, W, W, S_, N_, W * 0.5, ts, floor=zf * 0.5, wear=w, wear_e=w + 0.2, grime_e=g * 0.6, grime_m=g, far=fg)

    def grass_tuft(r, cx, cy, ang, sz, q, sprig=False):
        nb = max(3, int((r.u(3, 6) if sprig else lerp(15, 34, sz)) * q))
        ra = 0.006 if sprig else 0.012 + 0.02 * sz          # the crown is squeezed into the joint
        rb = 0.004 if sprig else 0.007 + 0.004 * sz
        hh = H * (0.45 + 0.55 * sz) * (0.55 if sprig else 1.0)
        ca, sa = cos(ang), sin(ang)
        for _ in range(nb):
            a = r.u(0, 2 * pi)
            out = sqrt(r.u())
            lx, ly = ra * out * cos(a), rb * out * sin(a)
            bx, by = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
            dry = r.chance(0.05 + 0.88 * d)
            L = hh * (0.38 + 0.62 * r.u() ** 0.6) * (1.0 - 0.3 * out)
            th0 = radians(lerp(2, 32, out) + r.u(-6, 10))
            bend = radians(r.u(12, 80)) * (1.0 + 0.5 * d)
            if dry:
                th0 += radians(r.u(4, 38))
                bend *= 1.3
                L *= 0.88
            w0 = r.u(0.0025, 0.0044) * (0.75 + 0.25 * sz) * (0.85 if dry else 1.0)
            blade(r, (bx, by, 0.0), faz(ang + a + r.u(-0.6, 0.6), cy), th0, bend, max(L, 0.02), w0, r.u(0.0012, 0.005), dry,
                  r.chance(0.35), short=sprig)
        if sprig or sz < 0.45 or p.flowers <= 0.01:
            return
        # seed stalks with a loose panicle of spikelets
        for _ in range(int(r.u(0.0, 3.2) * sz * (0.4 + p.flowers) * (1.0 + 0.6 * d))):
            a = faz(r.u(0, 2 * pi), cy)
            L = hh * r.u(1.25, 1.7)
            ts = np.array([0.0, 0.3, 0.6, 0.82, 1.0])
            th0, bend = radians(r.u(2, 14)), radians(r.u(4, 34))
            P = _path((cx + 0.005 * cos(a), cy + 0.005 * sin(a), 0.0), ts, pi / 2 - (th0 + bend * ts ** 1.6), a + 0.0 * ts, L)
            T = tube(DR, P, 0.0013 - 0.0007 * ts, sides=3, wear=0.5, grime=0.1, far=3.0)
            nr = nrng(r)
            m = int(r.u(8, 13))
            tt = 0.64 + 0.36 * (np.arange(m) + nr.uniform(0, 0.8, m)) / m
            pts = np.stack([np.interp(tt, ts, P[:, k]) for k in range(3)], 1)
            tg = _norm(np.stack([np.interp(tt, ts, T[:, k]) for k in range(3)], 1))
            side0 = np.array([sin(a), -cos(a), 0.0])
            rot = np.arange(m) * 2.4 + nr.uniform(-0.4, 0.4, m)
            side = _norm(np.cross(tg, side0) * np.sin(rot)[:, None] + side0[None, :] * np.cos(rot)[:, None])
            beta = np.radians(nr.uniform(16, 40, m))[:, None]
            dv = tg * np.cos(beta) + side * np.sin(beta)
            wv = _norm(np.cross(dv, tg))
            ln = (nr.uniform(0.008, 0.014, m) * (1.25 - 0.5 * (tt - 0.64) / 0.36))[:, None]
            tris2(DR, pts - wv * 0.0008, pts + wv * 0.0008, pts + dv * ln, wear=0.42, grime=[0.3, 0.3, 0.1])

    def dandelion_leaf(r, cx, cy, a, L, e0, bend, wmax, zf, age, fg=1.0):
        """Runcinate leaf: back-swept teeth growing towards an arrow-shaped terminal lobe, the two margins out
        of step.  age 0 = fresh, 1 = old (browning margins, yellowing tip), 2 = dead (dry material)."""
        nt = int(r.u(3, 5.99))
        t0, t1 = 0.18, 0.72
        h = (t1 - t0) / (2 * nt)
        ts, wl, wr, bl, br = [0.0, 0.09], [0.085, 0.08], [0.085, 0.08], [0.0, 0.0], [0.0, 0.0]
        for j in range(2 * nt):
            g = 0.36 + 0.64 * ((j // 2 + 1) / nt) ** 0.8
            pk, vl = g * r.u(0.72, 1.08), r.u(0.16, 0.27)
            ts.append(t0 + j * h)
            if j % 2 == 0:
                wl.append(pk); bl.append(0.6 * h); wr.append(vl); br.append(0.0)
            else:
                wl.append(vl); bl.append(0.0); wr.append(pk); br.append(0.6 * h)
        tl = r.u(0.88, 1.12)
        ts += [t1 + 0.015, t1 + 0.075, 0.93, 1.0]
        wl += [0.24, tl, tl * 0.5, 0.0]
        wr += [0.24, tl * r.u(0.88, 1.1), tl * 0.5, 0.0]
        bl += [0.0, 0.35 * h, 0.0, 0.0]
        br += [0.0, 0.35 * h, 0.0, 0.0]
        ts = np.array(ts)
        WL, WR = np.array(wl) * wmax, np.array(wr) * wmax
        P = _path((cx + 0.005 * cos(a), cy + 0.005 * sin(a), 0.004), ts, e0 - bend * ts ** 1.3, a + r.u(-0.35, 0.35) * ts, L)
        P[:, 2] = np.maximum(P[:, 2], zf)
        T, S_, N_ = _frames(P, (sin(a), -cos(a), 0.0), twist=r.u(-0.25, 0.25) * ts)
        wav = 0.0015 * np.sin(ts * r.u(18, 30) + r.u(0, 6))
        cup = r.u(0.1, 0.22)
        if age == 2:
            buf, wear, wk = DR, 0.5 + 0.06 * ts, 0.05
            ge, gm = np.clip(0.45 + 0.2 * ts + r.u(-0.1, 0.1), 0, 1), 0.6 + 0.1 * ts
        else:
            buf, wk = LF, 0.04
            wear = 0.0 + 0.03 * ts + (0.09 * ts ** 3 if age else 0.0)
            ge = 0.2 + 0.25 * d + (0.3 + 0.4 * ts ** 2 if age else 0.0) + np.zeros(len(ts))
            gm = 0.45 + np.zeros(len(ts))
            if r.chance(0.55):                 # a bruise / snail-grazed patch
                k = int(r.u(3, len(ts) - 2))
                ge[k] += 0.45
                gm[k] += 0.25
        ribbon(buf, P, WL, WR, S_, N_, 0.0, ts, lift_l=np.minimum(WL, 0.5 * wmax) * cup + wav,
               lift_r=np.minimum(WR, 0.5 * wmax) * cup - wav, floor=zf * 0.6, T=T, back_l=np.array(bl) * L,
               back_r=np.array(br) * L, wear=wear * (1.0 if age == 2 else 0.5), wear_e=wear + wk, grime_e=ge, grime_m=gm, under=wear + 0.18,
               far=fg)

    def flower_head(r, tip, axv, kind, q):
        axv = _norm(np.asarray(axv, float))
        ref = np.array([0.0, 0.0, 1.0]) if abs(axv[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        A = _norm(ref - axv * float(ref @ axv))
        Bv = np.cross(axv, A)
        nr = nrng(r)

        def radial(a):
            return A * np.cos(a)[..., None] + Bv * np.sin(a)[..., None]

        if kind == "bud":
            hs = np.array([0.0, 0.004, 0.009, 0.0145])[:, None]
            tube(LF, tip + axv[None, :] * hs, np.array([0.002, 0.0042, 0.004, 0.0]), sides=5, grime=np.array([0.1, 0.12, 0.25, 0.45]), point=True,
                 far=0.0)
            return
        a = np.arange(6) * (pi / 3) + nr.uniform(-0.2, 0.2, 6)
        rad = radial(a)
        tang = np.cross(axv[None, :], rad)
        if kind == "flower":
            if far:                            # the whole head: one yellow gem
                _gem(DR, tip + axv * 0.0075, axv, A, Bv, 0.0165, 0.007, 0.006, wear=0.03, grime=0.02)
                return
            # green involucre cup with reflexed outer bracts
            tube(LF, tip + axv[None, :] * np.array([0.0, 0.003, 0.0075])[:, None], np.array([0.0019, 0.0044, 0.005]), sides=5,
                 grime=0.06, wear=0.1)
            b0 = tip + axv * 0.002 + rad * 0.0032
            tris2(LF, b0 - tang * 0.0011, b0 + tang * 0.0011,
                  tip - axv * nr.uniform(0.002, 0.005, 6)[:, None] + rad * nr.uniform(0.0062, 0.008, 6)[:, None], grime=0.15)
            # dense all-yellow head: a low central dome of young florets (a shade deeper at the very centre)
            # and three overlapping tiers of short strap florets
            c = tip + axv * 0.0075
            tube(DR, c + axv[None, :] * np.array([0.0, 0.0032, 0.0056, 0.0068])[:, None], np.array([0.0042, 0.0038, 0.0022, 0.0]), sides=7,
                 grime=np.array([0.0, 0.0, 0.05, 0.12]), point=True)
            # (floret bases sit just outside the boss: a base buried inside it would bake black AO into the floret)
            for (cnt, ln, ang, hw, hb, rb, g0) in ((16, 0.0145, 98, 0.0017, 0.0002, 0.0047, 0.0),
                                                   (15, 0.013, 76, 0.0016, 0.001, 0.0045, 0.0),
                                                   (13, 0.0105, 60, 0.0015, 0.0024, 0.0041, 0.0),
                                                   (9, 0.0062, 78, 0.0012, 0.0058, 0.0024, 0.03)):
                a = nr.uniform(0, 6.3) + np.arange(cnt) * (2 * pi / cnt) + nr.uniform(-0.14, 0.14, cnt)
                rad = radial(a)
                tang = np.cross(axv[None, :], rad)
                an = np.radians(ang + nr.uniform(-8, 8, cnt))[:, None]
                f = axv[None, :] * np.cos(an) + rad * np.sin(an)
                b0 = c + axv * hb + rad * rb
                l = (ln * nr.uniform(0.82, 1.1, cnt))[:, None]
                quads(DR, b0 - tang * 0.0007, b0 + tang * 0.0007, b0 + f * l + tang * hw, b0 + f * l - tang * hw,
                      wear=[0.0, 0.0, 0.06, 0.06], grime=[g0 + 0.03, g0 + 0.03, g0, g0])
            return
        # seed clock: small dark receptacle, drooping bracts, hair-thin achene stalks each carrying a little
        # parasol of pappus hairs; a cap of seeds has usually blown away already
        b0 = tip + axv * 0.003 + rad * 0.002
        tris2(LF, b0 - tang * 0.001, b0 + tang * 0.001, tip - axv * nr.uniform(0.005, 0.008, 6)[:, None] + rad * nr.uniform(0.003, 0.005, 6)[:, None],
              grime=0.25, wear=0.1)
        R = r.u(0.0195, 0.022)
        c0 = tip + axv * 0.004
        tube(DR, c0 + axv[None, :] * np.array([-0.002, 0.0, 0.0022, 0.0036])[:, None], np.array([0.0012, 0.0031, 0.0026, 0.0]), sides=5,
             grime=0.95, point=True, far=0.0)
        ns = max(68, int(92 * q))
        i = np.arange(ns)
        zc = 1.0 - (i + 0.5) / ns * 1.72               # skip the cone around the scape
        rr = np.sqrt(np.maximum(0.0, 1.0 - zc * zc))
        dv = _norm(axv[None, :] * zc[:, None] + radial(i * 2.39996 + nr.uniform(-0.2, 0.2, ns)) * rr[:, None] + nr.normal(0, 0.05, (ns, 3)))
        keep = nr.uniform(size=ns) > 0.05 + 0.2 * d
        if r.chance(0.6 + 0.4 * d):
            bd = _norm(nr.normal(size=3) + axv * 0.5)
            keep &= (dv @ bd) < r.u(0.2, 0.8) - 0.5 * d
        dv = dv[keep]
        m = len(dv)
        if m == 0:
            return
        if far:                                # the airy ball of pappus that is left: one pale gem
            rg = R * 0.72 * sqrt(m / ns)
            _gem(DR, c0, axv, A, Bv, rg, rg, rg, wear=1.0)
            return
        ref2 = np.where(np.abs(dv[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
        e1 = _norm(ref2 - dv * (ref2 * dv).sum(1, keepdims=True))
        e2 = np.cross(dv, e1)
        roll = nr.uniform(0, 2 * pi, m)
        w1 = e1 * np.cos(roll)[:, None] + e2 * np.sin(roll)[:, None]
        root = c0 + dv * (R * nr.uniform(0.93, 1.05, m) - 0.0042)[:, None]
        b = c0 + dv * 0.0027
        tris2(DR, b - w1 * 0.0006, b + w1 * 0.0006, root, wear=[0.8, 0.8, 1.0], grime=[0.3, 0.3, 0.0])
        ks = 6
        az = roll[:, None] + np.arange(ks)[None, :] * (2 * pi / ks) + nr.uniform(-0.25, 0.25, (m, ks))
        beta = np.radians(nr.uniform(40, 64, (m, ks)))[..., None]
        radv = e1[:, None, :] * np.cos(az)[..., None] + e2[:, None, :] * np.sin(az)[..., None]
        sv = dv[:, None, :] * np.cos(beta) + radv * np.sin(beta)
        wv = -e1[:, None, :] * np.sin(az)[..., None] + e2[:, None, :] * np.cos(az)[..., None]
        ln = nr.uniform(0.0065, 0.0085, (m, ks))[..., None]
        r3 = np.repeat(root[:, None, :], ks, 1)
        tris2(DR, (r3 + sv * ln - wv * 0.0009).reshape(-1, 3), (r3 + sv * ln + wv * 0.0009).reshape(-1, 3), r3.reshape(-1, 3), wear=1.0)

    def flower_stalk(r, cx, cy, kind, q):
        a = faz(r.u(0, 2 * pi), cy)
        L = (0.035 + r.u(0.45, 1.25) * H) * {"bud": 0.4, "clock": 1.15}.get(kind, 1.0)
        ts = np.linspace(0, 1, 5)
        th0, bend = radians(r.u(4, 24)), radians(r.u(-22, 14))
        P = _path((cx + 0.004 * cos(a), cy + 0.004 * sin(a), 0.002), ts, pi / 2 - (th0 + bend * ts ** 1.4), a + r.u(-0.4, 0.4) * ts, L)
        T = tube(LF, P, 0.0021 - 0.0005 * ts, sides=4, wear=0.16 + 0.1 * ts, grime=0.25 * (1 - ts))
        flower_head(r, P[-1], T[-1], kind, q)

    def dandelion(r, cx, cy, sz, q, idx=9):
        nl = max(5, int(r.u(7, 11.5) * (0.7 + 0.3 * sz) * q))
        Lmax = lerp(0.075, 0.15, sz) * (0.7 + 0.3 * min(1.6, H / 0.16)) * (1.0 - 0.2 * d)
        a0 = r.u(0, 2 * pi)
        for i in range(nl):
            rank = i / nl
            a = faz(a0 + i * 2.39996 + r.u(-0.25, 0.25), cy)
            L = Lmax * (1.0 - 0.5 * rank) * r.u(0.82, 1.1)
            e0 = radians(lerp(9, 52, rank ** 1.3) + r.u(-4, 8)) * (1.0 - 0.45 * d)
            bend = radians(lerp(26, 58, rank) + r.u(-8, 12))
            age = 0
            if rank < 0.5:
                age = 2 if r.chance(0.03 + 0.4 * d) else (1 if r.chance(0.45 + 0.4 * d) else 0)
            dandelion_leaf(r, cx, cy, a, L * (0.8 if age == 2 else 1.0), e0, bend, L * r.u(0.15, 0.2) * (0.6 if age == 2 else 1.0),
                           0.002 + 0.006 * rank + r.u(0, 0.002), age, fg=1.5 if i % 2 == 0 else 0.0)     # lod 2: every other leaf
        nf = int(round(p.flowers * r.u(0.8, 4.4) * (0.5 + 0.5 * sz)))
        kinds = ["clock" if r.chance(0.22 + 0.55 * d) else ("bud" if r.chance(0.2) else "flower") for _ in range(nf)]
        if p.flowers > 0.25 and idx < 2:       # every flowering patch shows at least one bloom and one clock
            kinds = [("flower", "clock")[idx]] + kinds[1:]
        for kind in kinds:
            flower_stalk(r, cx, cy, kind, q)

    def plantain(r, cx, cy, sz, q):
        nl = max(4, int(r.u(5, 8.5) * q))
        Lmax = lerp(0.06, 0.125, sz) * (1.0 - 0.2 * d)
        a0 = r.u(0, 2 * pi)
        ts = np.array([0.0, 0.16, 0.3, 0.44, 0.58, 0.72, 0.87, 1.0])
        tb = np.clip((ts - 0.22) / 0.78, 0.0, 1.0)
        shape0 = 0.09 + 0.91 * np.sin(pi * tb ** 0.8) ** 0.75
        for i in range(nl):
            rank = i / nl
            a = faz(a0 + i * 2.39996 + r.u(-0.3, 0.3), cy)
            L = Lmax * (1.0 - 0.4 * rank) * r.u(0.85, 1.1)
            e0 = radians(lerp(6, 40, rank) + r.u(-3, 8)) * (1.0 - 0.4 * d)
            bend = radians(lerp(14, 40, rank) + r.u(-6, 10))
            zf = 0.002 + 0.006 * rank + r.u(0, 0.002)
            P = _path((cx + 0.004 * cos(a), cy + 0.004 * sin(a), 0.004), ts, e0 - bend * ts ** 1.4, a + r.u(-0.3, 0.3) * ts, L)
            P[:, 2] = np.maximum(P[:, 2], zf)
            T, S_, N_ = _frames(P, (sin(a), -cos(a), 0.0), twist=r.u(-0.3, 0.3) * ts)
            WL = shape0 * L * r.u(0.23, 0.3)
            WR = WL.copy()
            if r.chance(0.4):                  # a torn / eaten margin
                k = int(r.u(3, 6.99))
                if r.chance(0.5):
                    WL[k] *= r.u(0.3, 0.6)
                else:
                    WR[k] *= r.u(0.3, 0.6)
            wav = 0.0018 * np.sin(ts * r.u(9, 16) + r.u(0, 6))
            dead = rank < 0.4 and r.chance(0.04 + 0.35 * d)
            old = rank < 0.5 and r.chance(0.4 + 0.4 * d)
            if dead:
                WL, WR = WL * 0.5, WR * 0.5
                wav = wav * 2.0
                buf, wear, wk, ge, gm = DR, 0.5 + 0.06 * ts, 0.05, np.clip(0.45 + 0.2 * ts, 0, 1), 0.6 + 0.1 * ts
            else:
                buf, wk = LF, 0.04
                wear = 0.0 + 0.03 * ts + (0.09 * ts ** 3 if old else 0.0)
                ge = 0.26 + 0.2 * d + (0.3 + 0.35 * ts ** 2 if old else 0.0) + np.zeros(len(ts))
                gm = 0.5 + np.zeros(len(ts))
                if r.chance(0.5):
                    k = int(r.u(3, 6.99))
                    ge[k] += 0.4
            ribbon(buf, P, WL, WR, S_, N_, 0.0, ts, lift_l=WL * 0.16 + wav, lift_r=WR * 0.16 - wav,
                   floor=zf * 0.6, wear=wear * (1.0 if dead else 0.5), wear_e=wear + wk, grime_e=ge, grime_m=gm, under=wear + 0.18,
                   far=1.5 if i % 2 == 0 else 0.0)
        for _ in range(int(round(p.flowers * r.u(0.5, 3.4) * sz))):
            a = faz(r.u(0, 2 * pi), cy)
            L = r.u(0.07, 0.15) * (0.6 + 0.4 * sz)
            ts2 = np.array([0.0, 0.3, 0.52, 0.6, 0.85, 1.0])
            th0, bend = radians(r.u(8, 30)), radians(r.u(-28, -4))
            P = _path((cx, cy, 0.002), ts2, pi / 2 - (th0 + bend * ts2), a + 0 * ts2, L)
            rad = np.array([0.0012, 0.0011, 0.0011, 0.0027, 0.003, 0.0])
            if d > 0.3 or r.chance(0.5):
                tube(DR, P, rad, sides=4, wear=0.2, grime=np.array([0.5, 0.5, 0.6, 0.85, 0.8, 0.7]), point=True, far=1.6)
            else:
                tube(LF, P, rad, sides=4, wear=np.array([0.15, 0.15, 0.15, 0.3, 0.3, 0.3]), grime=np.array([0.2, 0.1, 0.1, 0.25, 0.3, 0.3]), point=True,
                     far=1.6)

    kd = 1.0 - 0.45 * d                       # moss shrinks as it dries
    mw0 = 0.3 * d                             # ... and bleaches

    def cushion(r, cx, cy, ang, ra, rb, nl, spor=False):
        """Clumpy moss cushion: clustered lobes of different sizes on a thin ragged skirt that feathers out to nothing."""
        # (lod 2: no skirt and no outline tufts; the big lobe and every other of the rest, a third larger)
        fan(MO, r, cx, cy, ra * 1.12, rb * 1.2, ang, min(0.003, rb * 0.2), nu=11, ragged=(0.55, 1.45), wear_c=mw0 + 0.04, wear_r=mw0 + 0.2, far=0.0)
        tops = []
        ca, sa = cos(ang), sin(ang)
        for i in range(nl):
            if i == 0:
                lx, ly, s = r.u(-0.15, 0.15) * ra, r.u(-0.15, 0.15) * rb, 1.0
            else:
                a, rr = r.u(0, 2 * pi), sqrt(r.u(0.12, 1.0))
                lx, ly, s = rr * ra * 0.68 * cos(a), rr * rb * 0.68 * sin(a), lerp(0.8, 0.36, rr) * r.u(0.8, 1.15)
            lr = min(ra, rb) * 0.78 * s
            hh = lr * r.u(0.55, 0.85) * kd
            x, y = cx + lx * ca - ly * sa, cy + lx * sa + ly * ca
            lobe(MO, r, x, y, lr * r.u(1.0, 1.4), lr, ang + r.u(-0.6, 0.6), hh, nu=6, mid=(i == 0 and lr > 0.012), wear0=mw0, crown=r.u(0.14, 0.3),
                 far=1.35 if i % 2 == 0 else 0.0)
            tops.append((x, y, hh))
        for i in range(nl + 2):                # small tufts breaking up the outline and trailing off along the joint
            a = r.u(0, 2 * pi)
            lx, ly = ra * r.u(0.85, 1.35) * cos(a), rb * r.u(0.85, 1.25) * sin(a)
            sr_ = min(ra, rb) * r.u(0.2, 0.4)
            lobe(MO, r, cx + lx * ca - ly * sa, cy + lx * sa + ly * ca, sr_ * r.u(1.0, 1.5), sr_, ang + r.u(-0.8, 0.8), sr_ * r.u(0.5, 0.85) * kd,
                 nu=5, wear0=mw0 + r.u(0.0, 0.1), crown=r.u(0.1, 0.25), far=0.0)
        if spor and tops and r.chance(0.55 * (1.0 - d)):
            nr = nrng(r)
            n = int(r.u(3, 8))
            x, y, hh = tops[0]
            base = np.stack([x + nr.uniform(-0.5, 0.5, n) * min(ra, rb), y + nr.uniform(-0.5, 0.5, n) * min(ra, rb), np.full(n, hh * 0.5)], 1)
            lean = nr.normal(0, 0.22, (n, 2))
            top = base + np.stack([lean[:, 0], lean[:, 1], np.ones(n)], 1) * nr.uniform(0.012, 0.022, n)[:, None]
            az = nr.uniform(0, pi, n)
            wv = np.stack([np.cos(az), np.sin(az), np.zeros(n)], 1) * 0.00055
            tris2(DR, top - wv, top + wv, base, wear=[0.3, 0.3, 0.1], grime=[0.45, 0.45, 0.8])

    def moss_group(r, cx, cy, ang, sz, q):
        ra = r.u(0.03, 0.05) * (0.6 + 0.6 * sz) * kd
        cushion(r, cx, cy, ang, ra, ra * r.u(0.45, 0.7), max(3, int(r.u(5, 8.5) * q)), spor=True)
        for _ in range(int(r.u(1, 2.99))):                    # satellites further along the joint
            t = r.u(0.05, 0.11) * (0.6 + sz) * r.pick([-1, 1])
            x, y, a2 = snap(cx + t * cos(ang) + r.u(-0.01, 0.01), cy + t * sin(ang) + r.u(-0.01, 0.01))
            if inside(x, y, 0.0):
                ra2 = ra * r.u(0.4, 0.65)
                cushion(r, x, y, a2, ra2, ra2 * r.u(0.5, 0.75), 3)
        for _ in range(int(r.u(2, 5))):                       # crumbs
            t = r.u(0.03, 0.14) * (0.6 + sz) * r.pick([-1, 1])
            x, y, a2 = snap(cx + t * cos(ang) + r.u(-0.02, 0.02), cy + t * sin(ang) + r.u(-0.02, 0.02))
            if inside(x, y, 0.0):
                crumb(r, x, y, a2)

    def crumb(r, x, y, ang):
        rx = r.u(0.007, 0.015)
        lobe(MO, r, x + r.u(-0.004, 0.004), y + r.u(-0.004, 0.004), rx * r.u(1.1, 1.7), rx * r.u(0.55, 0.8), ang + r.u(-0.3, 0.3),
             rx * r.u(0.4, 0.6) * kd, nu=5, wear0=mw0 + 0.05, crown=0.3, far=0.0)

    def soil_drift(r, x, y, ang, la, lb, rings=1, fg=0.0):
        """Wind-blown dirt caught in the joint: a thin, pale, ragged smear along the joint line.
        (lod 2: the dirt under a plant is left out; fg is the gain of a smear that stays)"""
        fan(SO, r, x + r.u(-0.004, 0.004), y + r.u(-0.004, 0.004), la, lb * 1.3, ang + r.u(-0.2, 0.2), r.u(0.0018, 0.0032), nu=8 if rings == 1 else 9,
            ragged=(0.5, 1.4), rings=rings, wear_c=r.u(0.22, 0.34), wear_r=r.u(0.34, 0.5), far=fg)

    # ---------------------------------------------------------------- layout
    def gauss(r):
        return (r.u() + r.u() + r.u() - 1.5) * 2.0

    rl = rng.fork("layout")
    centres = [(0.0, 0.0)] + [(rl.u(-0.45, 0.45) * ax_, rl.u(-0.45, 0.45) * ay_) for _ in range(1 + int(S * 2.2))]
    xcl = [rl.u(-0.8, 0.8) * ax_ for _ in range(2 + int(S * 2))]

    def inside(x, y, pad):
        if wall:
            if abs(x) > ax_ - pad:
                return False
            return back - depth * sqrt(max(0.0, 1.0 - (x / ax_) ** 2)) - 1e-6 <= y <= ywall + 1e-6
        return (x / max(ax_ - pad, 0.03)) ** 2 + (y / max(ay_ - pad, 0.03)) ** 2 <= 1.0

    def sample(r, junc=0.0):
        """A rooting point: where the habit wants a plant, snapped into the nearest paving joint."""
        if wall:
            x = r.pick(xcl) + gauss(r) * S * 0.12 if r.chance(0.6) else r.u(-1, 1) * ax_
            y = ywall if r.chance(0.5) else back - abs(gauss(r)) * depth * 0.42
            sz = clamp(1.0 - (back - y) / depth * 0.9 + r.u(-0.2, 0.2), 0.3, 1.0)
        elif habit == "joints" and len(JX):
            for _ in range(8):
                i = min(int(r.u(0, len(JX))), len(JX) - 1)
                x, y = float(JX[i]), float(JY[i])
                if r.u() < 1.2 - 0.8 * hypot(x / ax_, y / ay_):
                    break
            sz = r.u(0.3, 0.65)
        else:
            c = r.pick(centres)
            x, y = c[0] + gauss(r) * S * 0.2, c[1] + gauss(r) * D * 0.2
            sz = clamp(1.0 - hypot(x / ax_, y / ay_) * 0.65 + r.u(-0.25, 0.3), 0.3, 1.0)
        x, y, ang = snap(x, y, junc)
        return x, y, sz, ang

    placed = []

    def place_one(r, rad, pad, junc):
        best = None
        for it in range(50):
            x, y, sz, ang = sample(r, junc)
            if not inside(x, y, pad):
                continue
            best = (x, y, sz, ang)
            rr = rad * (0.5 + 0.5 * sz)
            if all(hypot(x - px, y - py) > (rr + pr) * (0.85 if it < 30 else 0.5) for (px, py, pr) in placed):
                placed.append((x, y, rr))
                return best
        if best is not None:
            placed.append((best[0], best[1], rad * 0.5))
        return best

    # how many of each kind (largest remainder so small counts still honour the mix)
    wts = [("dandelion", p.dandelion), ("plantain", p.plantain), ("grass", p.grass), ("moss", p.moss)]
    tot_w = sum(w for _, w in wts)
    if tot_w <= 1e-6:
        wts, tot_w = [("grass", 1.0)], 1.0
    cnt = int(p.count)
    quota = [(k, cnt * w / tot_w) for k, w in wts]
    nk = {k: int(v) for k, v in quota}
    rest = cnt - sum(nk.values())
    for k, v in sorted(quota, key=lambda kv: -(kv[1] - int(kv[1])))[:rest]:
        nk[k] += 1
    # triangle budget: instanced ground clutter, ~11k for the default patch, never more than ~19k
    cshare = min(1.0, 0.22 + 0.55 * d)
    est = (nk.get("grass", 0) * (540 + 60) + nk.get("plantain", 0) * 420 + nk.get("moss", 0) * 470
           + nk.get("dandelion", 0) * (850 + p.flowers * 2.1 * (cshare * 1250 + (1 - cshare) * 300 + 32)))
    budget = lerp(10500.0, 17500.0, clamp((cnt - 16) / 14.0, 0.0, 1.0)) - 900.0
    q = clamp(budget / max(est, 1.0), 0.4, 1.0)

    radius = {"dandelion": 0.085, "plantain": 0.07, "grass": 0.055, "moss": 0.06}
    if habit == "joints":
        radius = {k: v * 0.55 for k, v in radius.items()}
    juncw = {"dandelion": 0.6, "plantain": 0.6, "grass": 0.35, "moss": 0.0}
    pad = min(0.07, 0.35 * ay_)
    rs = rng.fork("soil")
    for kind, _ in wts:
        r = rng.fork(kind)
        for i in range(nk.get(kind, 0)):
            got = place_one(rl, radius[kind], pad, juncw[kind])
            if got is None:
                continue
            x, y, sz, ang = got
            if kind == "grass":
                grass_tuft(r, x, y, ang, sz, q)
                soil_drift(rs, x, y, ang, 0.03 + 0.03 * sz, 0.011 + 0.006 * sz)
            elif kind == "dandelion":
                dandelion(r, x, y, sz, q, i)
                soil_drift(rs, x, y, ang, 0.035, 0.012)
            elif kind == "plantain":
                plantain(r, x, y, sz, q)
                soil_drift(rs, x, y, ang, 0.03, 0.011)
            else:
                moss_group(r, x, y, ang, sz * (0.5 if habit == "joints" else 1.0), q)

    # extras that knit the clumps into one patch: single sprigs of grass, moss crumbs, dirt in the joints
    r = rng.fork("sprigs")
    for _ in range(int(round(nk.get("grass", 0) * 1.6 * q + (2 if p.grass > 0 else 0)))):
        x, y, sz, ang = sample(r)
        if inside(x, y, 0.02):
            grass_tuft(r, x, y, ang, r.u(0.3, 0.7), 1.0, sprig=True)
    r = rng.fork("crumbs")
    for _ in range(int(round(p.moss * (1 - 0.6 * d) * (5 + cnt * 0.6)))):
        x, y, sz, ang = sample(r)
        if inside(x, y, 0.01):
            crumb(r, x, y, ang)
    r = rng.fork("dirt")
    for k in range(int(round(3 + cnt * 0.45))):
        x, y, sz, ang = sample(r)
        x, y, ang = snap(x + r.u(-0.05, 0.05), y + r.u(-0.05, 0.05))
        if inside(x, y, 0.0):
            soil_drift(r, x, y, ang, r.u(0.025, 0.05), r.u(0.008, 0.013), fg=1.6 if k % 2 == 0 else 0.0)
    if wall:
        # wind-blown dirt and moss gathered in the angle between paving and wall
        r = rng.fork("wallfoot")
        n = max(4, int(S / 0.07))
        for i in range(n + 1):
            xx = -ax_ * 0.94 + S * 0.94 * i / n + r.u(-0.015, 0.015)
            if r.chance(0.8):
                soil_drift(r, xx, ywall - 0.004, 0.0, r.u(0.035, 0.06), r.u(0.01, 0.018), rings=2, fg=1.6 if i % 2 == 0 else 0.0)
            if p.moss > 0.02 and r.chance(0.2 + 0.5 * p.moss * (1 - 0.6 * d)):
                rx = r.u(0.012, 0.022)
                lobe(MO, r, xx + r.u(-0.02, 0.02), ywall - r.u(0.0, 0.006), rx * r.u(1.2, 1.9), rx * 0.6, r.u(-0.2, 0.2), rx * r.u(0.4, 0.65) * kd,
                     nu=6, wear0=mw0, crown=0.32, far=1.6 if i % 2 == 0 else 0.0)
    if habit == "joints" and len(JX) and p.moss > 0.02:
        # moss creeping along the real joints (denser towards the middle of the patch)
        r = rng.fork("jointmoss")
        step = 0.026
        fill = clamp(0.12 + nk.get("moss", 0) / 22.0, 0.0, 0.9) * (1.0 - 0.5 * d)
        nmax = int(min(215 * q, fill * len(JX) * CELL * 0.8 / step))
        cov = np.zeros(len(JX), bool)
        done, tries = 0, 0
        while done < nmax and tries < 600:
            tries += 1
            i = min(int(r.u(0, len(JX))), len(JX) - 1)
            x, y = float(JX[i]), float(JY[i])
            if cov[i] or not inside(x, y, 0.01) or r.u() > 1.25 - 0.8 * hypot(x / ax_, y / ay_):
                continue
            ang = float(JA[i]) + (pi if r.chance(0.5) else 0.0)
            for _ in range(int(r.u(4, 17))):
                big = r.chance(0.3)
                lobe(MO, r, x + r.u(-0.003, 0.003), y + r.u(-0.003, 0.003), step * r.u(0.6, 0.9), r.u(0.007, 0.013) * kd, float(JA[i]) + r.u(-0.25, 0.25),
                     r.u(0.006, 0.012) * kd * (1.0 if big else 0.7), nu=7 if big else 5, mid=big, wear0=mw0, crown=r.u(0.14, 0.3),
                     far=(2.6, 1.3) if done % 3 == 0 else 0.0)     # lod 2: every third lobe, drawn out along the joint
                cov |= (JX - x) ** 2 + (JY - y) ** 2 < (0.8 * step) ** 2
                done += 1
                d2 = (JX - (x + cos(ang) * step)) ** 2 + (JY - (y + sin(ang) * step)) ** 2
                d2[cov] = 1e9
                j = int(np.argmin(d2))
                if d2[j] > (0.55 * step) ** 2 or not inside(float(JX[j]), float(JY[j]), 0.0) or done >= nmax:
                    break
                ang = atan2(float(JY[j]) - y, float(JX[j]) - x)
                x, y, i = float(JX[j]), float(JY[j]), j

    # grit caught in the joints and blown dead leaves lying across the stones
    r = rng.fork("litter")
    area = S * D
    for _ in range(int(round(p.litter * (8 + 30 * area)))):
        x, y, sz, ang = sample(r)
        if r.chance(0.3):
            x, y = x + r.u(-0.05, 0.05), y + r.u(-0.05, 0.05)
        else:
            x, y = x + r.u(-0.006, 0.006), y + r.u(-0.006, 0.006)
        if inside(x, y, 0.0):
            rx = r.u(0.004, 0.009)
            lobe(SO, r, x, y, rx, rx * r.u(0.6, 1.0), r.u(0, pi), rx * r.u(0.45, 0.8), nu=5, wear0=r.u(0.15, 0.45), crown=0.2, foot=0.3, far=0.0)
    for _ in range(int(round(p.litter * (2 + 7 * area) * (0.6 + 0.8 * d)))):
        x, y, sz, ang = sample(r)
        x, y = x + r.u(-0.06, 0.06), y + r.u(-0.06, 0.06)
        if not inside(x, y, 0.0):
            continue
        a = r.u(0, 2 * pi)
        tilt = radians(r.u(0, 5))
        T0 = np.array([cos(a) * cos(tilt), sin(a) * cos(tilt), sin(tilt)])
        strap(DR, (x, y, r.u(0.003, 0.006)), T0, (0, 0, 1), r.u(0.028, 0.05), r.u(0.006, 0.011), n=5, curl=r.u(0.05, 0.35),
              thick=0.0011, shape=0.75, wear=r.u(0.48, 0.6), grime=r.u(0.6, 0.85))

    # ---------------------------------------------------------------- meshes
    if far:
        _far_outline(B)
    for key, buf in B.items():
        buf.obj("weeds_" + key, M[key], sm=75.0 if key in ("moss", "soil") else 180.0, ymax=(ay_ - 0.002) if wall else None)
    socket("centre", "attach", (0, 0, 0), role="ground_clutter")
