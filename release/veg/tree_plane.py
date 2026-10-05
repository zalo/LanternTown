"""Mature London-plane street tree. A leaning, slightly sinuous bole rises from a buttress of 4-6 unequal
lobes with real valleys between them; each lobe runs out as a surface root - a half-buried ridge with gravel
lying against its flanks that dives under a low mound of heaved gravel and, if strong, lifts the sett ring of
the pit. The bole is one quad grid; 2-5 scaffold limbs leave it at staggered heights and are welded into it
(the grid cells under each limb base are cut out and a collar strip joins the hole's own vertices to the limb,
with a raised branch-bark ridge in the crotch), then a leader carries on. Every limb is a tapered, oval swept
tube with two elbows, grown towards one of the foliage masses that tile an irregular crown envelope; the
envelope is cut by two or three clefts (`open`) so the crown is a few unequal lobes with sky between them.
Each scaffold forks once or twice below the leaves and carries pruning stubs and burrs; a graft that would
pass within a limb's diameter of another limb is not made. The lower crown is lifted so the scaffolds show.
Bark is built the way a plane wears it: a furrowed skin (`bark`, relief and all, so the trunk shades under a
lantern) shed in irregular plates (Voronoi cells in warped space), a minority fresh cream, the oldest browner
and grimier - painted as soft wear / grime masks the shader breaks up with its own noise, never black and
never a hard-edged second material; the plates fade out half a metre below the fork and the limbs wear one
even skin with a faint mottle and a graded stain in each crotch. On top lies a separate shell of old furrowed
bark that covers the foot and breaks up into ragged tongues, islands and small lifting flakes. Old pruning
wounds are sunk slits with unequal callus lips over a brown heart; cut limbs are flush callus rings or low
burrs, never stubs.
The foliage is cut-out leaf clusters (material parameter `leafy`): thousands of small cards, each one cell of
the `hedge` leaf mosaic with the leaves cut out of it and thinned towards its rim, lying over two nested lumpy
hulls of every foliage mass and in pairs along twiglets on every shoot; inside each mass a few hundred big
sprays (boughs) close the view through it. Cards carry their depth in the crown (darker inside) and one of
two colours per material, so no crown is one flat tone. The full model is also budgeted for a collapsed
copy (the fallback if a modelled level is ever refused): its middle level keeps every card and a proper trunk,
the far level the boughs and the trunk.
A pollard cuts the limbs back to tapered knuckles, one limb to each, that throw their own tuft of straight
shoots clothed the same way.
Levels: the tree is grown once and the same at every level - every limb, fork, weld and random draw - and only
what is meshed changes. lod 1 keeps the bole as it is (grid, welds, wounds, plates and old bark: it is at eye
level) without the padding the collapsed copy needs, gives the limbs fewer rings and sides, turns the shoots into
ribbons, and carries the same leaf area on fewer cards: the outer cards of the crown as they are (its outline
and its fringe against the sky are those of the full model), the ones inside it four to one, each of those a
spray twice the size. lod 2 is a plain bole (old bark below, underbark above) with
the limbs run into it, no shoots, and over the same hulls one big spray for every twenty small cards - a card as
large as a bough, in the canopy's colours, its leaves at their true size - around two boughs in five.
"""
import math

import bpy
import bmesh
import numpy as np

from tk import *

# (tri_budget: a canopy of small leaf clusters that holds up at 5 m, and whose middle level of detail still
# has every one of them, costs more than the class default; 2x the old 55k is the agreed ceiling)
META = {"name": "Street plane tree", "class": "vegetation", "mount": "ground", "focus": 0.35,
        "auto_wear": 0.0, "auto_grime": 0.18, "tri_budget": 110000, "lod": True}
PARAMS = {
    "height": {"default": 9.2, "min": 7.0, "max": 11.0, "desc": "overall height (m)"},
    "crown_width": {"default": 7.0, "min": 4.2, "max": 9.0, "desc": "approximate crown spread (m)"},
    "branchiness": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "0 = few big limbs and leaf masses, 1 = many limbs, forks and twigs"},
    "leaf_density": {"default": 1.0, "min": 0.3, "max": 1.5, "desc": "amount of foliage (1 = full summer crown)"},
    "season": {"default": 0.0, "min": 0.0, "max": 1.0, "desc": "0 = summer green ... 1 = late autumn (gold / russet, thinner crown, leaf litter)"},
    "lean": {"default": 3.0, "min": 0.0, "max": 14.0, "desc": "lean of the trunk (degrees, from the ground up)"},
    "lean_dir": {"default": 20.0, "min": 0.0, "max": 360.0, "desc": "compass direction of the lean (degrees, 0 = +X)"},
    "bole": {"default": 0.33, "min": 0.26, "max": 0.44, "desc": "clear trunk below the first limbs, fraction of the height"},
    "girth": {"default": 1.15, "min": 0.7, "max": 1.4, "desc": "trunk thickness multiplier"},
    "open": {"default": 0.25, "min": 0.0, "max": 1.0, "desc": "sky gaps: 0 = closed crown, 1 = a few separate lobes of foliage around the leader"},
    "leaf_size": {"default": 0.145, "min": 0.11, "max": 0.18, "desc": "leaf scale (m): a leaf-cluster card is up to 1.5x this across (less on a small tree)"},
    "mottle": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "how far the old bark has shed: 0 = rough bark up to the fork, 1 = only the foot keeps it; also the amount of cream / olive patches"},
    "moss": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "moss at the damp base and north side of the trunk"},
    "scars": {"default": 2, "min": 0, "max": 4, "desc": "old pruning wounds (callus roll around a dark heart) on the bole"},
    "litter": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "fallen leaves on the ground (scaled up by season)"},
    "surround": {"default": "setts", "choices": ["setts", "soil", "none"], "desc": "tree pit: packed gravel ringed with granite setts, bare soil, or nothing (for a tree grate)"},
    "leaf_tint": {"default": "#4a6e2c", "desc": "summer leaf colour"},
    "form": {"default": "natural", "choices": ["natural", "pollard"],
             "desc": "natural crown, or pollarded: limbs cut back to knuckles that throw a brush of straight shoots"},
}
VARIANTS = {
    "autumn": {"season": 1.0, "height": 9.6, "crown_width": 7.0, "litter": 1.0, "lean": 5.0, "lean_dir": 200.0, "seed": 4},
    "young_slender": {"height": 8.2, "crown_width": 4.3, "branchiness": 0.55, "girth": 0.74, "bole": 0.3, "scars": 0, "open": 0.75,
                      "leaf_density": 0.9, "mottle": 0.9, "moss": 0.1, "lean": 4.0, "surround": "soil", "seed": 11},
    "veteran_broad": {"height": 10.8, "crown_width": 8.8, "branchiness": 1.0, "girth": 1.35, "bole": 0.28, "scars": 4,
                      "moss": 0.8, "lean": 2.0, "season": 0.25, "mottle": 0.45, "leaf_size": 0.155, "open": 0.4, "seed": 23},
    "leaning_thin": {"height": 8.4, "crown_width": 6.0, "lean": 13.0, "lean_dir": 205.0, "leaf_density": 0.55,
                     "season": 0.6, "branchiness": 0.75, "litter": 0.7, "girth": 0.9, "seed": 37},
    "pollard": {"form": "pollard", "height": 7.4, "crown_width": 5.6, "girth": 1.3, "bole": 0.36, "branchiness": 0.7,
                "scars": 1, "lean": 1.0, "mottle": 0.8, "seed": 5},
}

UP = np.array([0.0, 0.0, 1.0])
TRI_CAP = 97000
LEAF_TILE = 2.3                                   # scale of the hedge tile on the foliage: its 2.8 cm leaves become 6.5 cm
# The foliage is thousands of small leaf-cluster cards. Each is mapped onto exactly one cell of the leaf mosaic,
# and the engine (material parameter `leafy` = the size of that cell) cuts the leaves of the cell out of it, more
# of them away towards its rim: a card is a ragged spray of leaves, never a polygon.
CARD_K = 1.5                                      # a card is at most this many leaf_size across
BIG_K = 5.0                                       # ... and a bough (a big inner spray) this many cards
# cards lie on two nested hulls of every mass: (share of the hull radius, cover, depth = shade)
CARD_LAYERS = [(1.0, 1.5, 0.0), (0.84, 0.8, 0.35)]
TWIG_SHARE = 0.12                                 # twiglets per card
# what the town's reduced levels keep (tk.final: collapse ratios), and the wood that must be left at each
LOD_K = (0.4, 0.1)
WOOD_LOD = (4500, 1300)
TOP_K = 0.6
# The modelled levels (lod 0, 1, 2). Small cards, from the outside of the crown inwards: (the cards up to this
# share of them by depth, share of those kept, size, as a big spray) - the same leaf area on fewer cards; a card
# that is only larger shows its cluster and its leaves larger (so: inside the crown; and the outer share is
# less where the level's budget is short), a big spray is one cell of the mosaic at the size of a bough (its
# own material: the leaves keep their size).
# Boughs: (share kept, size). Limbs: share of the sides around, tolerance of the path (x 4 mm / 18 mm).
CARD_LOD = ([(1.0, 1.0, 1.0, 0)], [(0.4, 1.0, 1.0, 0), (1.0, 0.25, 2.0, 0)], [(1.0, 0.05, BIG_K, 1)])
BOUGH_LOD = ((1.0, 1.0), (1.0, 1.0), (0.4, 1.5))
TUBE_LOD = ((1.0, 0.0), (0.65, 3.0), (0.42, 2.5))

# Leaf blades in polar form around the petiole end (apex along +X, unit length). Three levels of detail of the
# same broad palmate outline: the full five-lobed blade (only on the lowest sprays, over people's heads, and
# for the litter), a three-lobed blade with two sinuses (everywhere else), and a plain broad blade that is
# only used if the triangle budget runs out.
def _tmpl(ang, rad, tri):
    a, r = np.radians(ang), np.array(rad)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1), np.array(tri)


LEAVES = [
    _tmpl([0, -60, 0, 60], [0, 0.58, 1.0, 0.58], [[0, 1, 2], [0, 2, 3]]),
    _tmpl([0, -52, -22, 0, 22, 52], [0, 0.84, 0.56, 1.0, 0.56, 0.84], [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]),
    _tmpl([0, -104, -47, -23, 0, 23, 47, 104], [0, 0.52, 0.86, 0.64, 1.0, 0.64, 0.86, 0.52],
          [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 6], [0, 6, 7]]),
]
LEAF_T = [len(t[1]) for t in LEAVES]


def _n(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        l = np.linalg.norm(v)
        return v / l if l > 1e-9 else v
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def _sstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _bez(p0, p1, p2, p3, n):
    t = (np.linspace(0.0, 1.0, n) if np.isscalar(n) else np.asarray(n, dtype=np.float64))[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _cum(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def _frames(P, flat=None):
    """Tangents + parallel-transported normals / binormals along a polyline. flat = per-point weight
    (0 = ring plane horizontal, 1 = perpendicular to the path) so a leaning trunk still meets the ground flat."""
    n = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T = _n(T)
    if flat is not None:
        T = _n(UP[None, :] * (1 - flat)[:, None] + T * flat[:, None])
    N = np.zeros_like(P)
    ref = np.array([1.0, 0, 0]) if abs(T[0][0]) < 0.9 else np.array([0, 1.0, 0])
    N[0] = _n(ref - T[0] * np.dot(ref, T[0]))
    for i in range(1, n):
        N[i] = _n(N[i - 1] - T[i] * np.dot(N[i - 1], T[i]))
    return T, N, np.cross(T, N)


def _zip(a0, a1):
    """Stitch two rings with different vertex counts (angles ascending around the same axis): a list of
    triangles (ring, i, ring, i, ring, i), plus the unwrapped angles of both rings and ring 1's start."""
    n0, n1 = len(a0), len(a1)
    A = a0[0] + np.concatenate([(a0 - a0[0]) % (2 * pi), [2 * pi]])
    j0 = int(np.argmin(np.abs((a1 - a0[0] + pi) % (2 * pi) - pi)))
    b0 = a0[0] + ((a1[j0] - a0[0] + pi) % (2 * pi) - pi)
    Bv = b0 + np.concatenate([(a1[(j0 + np.arange(n1)) % n1] - a1[j0]) % (2 * pi), [2 * pi]])
    i = j = 0
    out = []
    while i < n0 or j < n1:
        if j >= n1 or (i < n0 and A[i + 1] <= Bv[j + 1]):
            out.append((0, i, 0, i + 1, 1, j))
            i += 1
        else:
            out.append((0, i, 1, j + 1, 1, j))
            j += 1
    return out, A, Bv, j0


class _Buf:
    """Accumulates quads / triangles with per-corner UVs (metres), a material slot per face and three
    per-vertex attributes (plate strength, fixed tone or -1, locked)."""

    def __init__(self):
        self.v, self.a, self.q, self.quv, self.qm, self.t, self.tuv, self.tm = [], [], [], [], [], [], [], []
        self.nv = 0

    @property
    def tris(self):
        return sum(len(q) for q in self.q) * 2 + sum(len(t) for t in self.t)

    def faces(self, quads=None, quv=None, tris=None, tuv=None, qslot=0, tslot=0, base=0):
        """Faces over vertices that are already in the buffer (absolute indices + base)."""
        if quads is not None and len(quads):
            self.q.append(np.asarray(quads, dtype=np.int64).reshape(-1, 4) + base)
            self.quv.append(np.asarray(quv, dtype=np.float64).reshape(-1, 2))
            self.qm.append(np.broadcast_to(np.asarray(qslot, np.int32), (len(self.q[-1]),)).copy())
        if tris is not None and len(tris):
            self.t.append(np.asarray(tris, dtype=np.int64).reshape(-1, 3) + base)
            self.tuv.append(np.asarray(tuv, dtype=np.float64).reshape(-1, 2))
            self.tm.append(np.broadcast_to(np.asarray(tslot, np.int32), (len(self.t[-1]),)).copy())

    def add(self, verts, quads=None, quv=None, tris=None, tuv=None, qslot=0, tslot=0, va=None):
        base = self.nv
        self.v.append(np.asarray(verts, dtype=np.float64).reshape(-1, 3))
        n = len(self.v[-1])
        self.a.append(np.broadcast_to(np.asarray([0.0, -1.0, 0.0] if va is None else va, dtype=np.float64), (n, 3)).copy())
        self.nv += n
        self.faces(quads, quv, tris, tuv, qslot, tslot, base)
        return base

    def tube(self, P, R, sides, cap=True, rfn=None, frames=None, first=None, pf=None, exact=False):
        """Tapered tube along P. U = true girth (bark ridges converge as the limb tapers), V = length.
        sides may change from ring to ring (never growing): the rings are stitched, so a limb is round where
        it is thick and cheap where it is thin. rfn(i, A) -> radii of ring i at angles A (lobed / flared
        sections). first = (vertex indices, angles) reuses an existing ring as ring 0 (a welded base).
        Levels: the same tube through the rings its bends and its taper need, with fewer sides (see TUBE_LOD);
        exact = build it as given at every level."""
        P = np.asarray(P, dtype=np.float64)
        n = len(P)
        R = np.broadcast_to(np.asarray(R, dtype=np.float64), (n,)).copy()
        T, N, B = frames if frames is not None else _frames(P)
        sides = np.minimum.accumulate(np.broadcast_to(np.asarray(sides, dtype=np.int64), (n,)))
        pf = np.zeros(n) if pf is None else np.broadcast_to(np.asarray(pf, dtype=np.float64), (n,))
        cum = _cum(P)
        Lv = lod()
        fat = None
        if Lv and not exact:
            if n > 2:
                pts = [tuple(P[i]) + (2.0 * R[i],) for i in range(n)]
                at_ = {q: i for i, q in enumerate(pts)}
                kp = np.array(sorted({at_[q] for q in lod_path(pts, k=TUBE_LOD[Lv][1])}))
                if len(kp) < n:
                    rfn0 = rfn
                    P, R, T, N, B, pf, cum, sides = P[kp], R[kp], T[kp], N[kp], B[kp], pf[kp], cum[kp], sides[kp]
                    rfn = None if rfn0 is None else (lambda i, A: rfn0(int(kp[i]), A))
                    n = len(P)
            lo = 5 - Lv
            sides = np.where(sides > lo, np.maximum(np.ceil(sides * TUBE_LOD[Lv][0] - 0.3).astype(np.int64), lo), sides)
            fat = (pi / sides) / np.sin(pi / sides)       # ... and a ring of few sides is as wide, on average, as the round one
        base = self.nv
        vs, at, rings = [], [], []
        k = 0
        for i in range(n):
            if i == 0 and first is not None:
                rings.append((np.asarray(first[0], dtype=np.int64), np.asarray(first[1], dtype=np.float64), 2 * pi * R[0]))
                continue
            m = int(sides[i])
            A = np.arange(m) * 2 * pi / m
            rr = np.full(m, R[i]) if rfn is None else rfn(i, A)
            if fat is not None:
                rr = rr * fat[i]
            vs.append(P[i][None, :] + rr[:, None] * (np.cos(A)[:, None] * N[i][None, :] + np.sin(A)[:, None] * B[i][None, :]))
            at.append(np.tile([pf[i], -1.0, 0.0], (m, 1)))
            rings.append((base + k + np.arange(m), A, 2 * pi * float(rr.mean())))
            k += m
        if cap:
            vs.append((P[-1] + T[-1] * R[-1] * 0.8)[None, :])
            at.append(np.array([[pf[-1], -1.0, 0.0]]))
        self.add(np.vstack(vs), va=np.vstack(at))
        Q, QUV, TR, TUV = [], [], [], []
        for i in range(n - 1):
            (i0, a0, g0), (i1, a1, g1) = rings[i], rings[i + 1]
            if len(i0) == len(i1) and not (i == 0 and first is not None):
                m = len(i0)
                kk = np.arange(m)
                k1 = (kk + 1) % m
                Q.append(np.stack([i0[kk], i0[k1], i1[k1], i1[kk]], axis=1))
                f0, f1 = kk / m - 0.5, (kk + 1) / m - 0.5
                c0, c1 = np.full(m, cum[i]), np.full(m, cum[i + 1])
                QUV.append(np.stack([np.stack([f0 * g0, c0], -1), np.stack([f1 * g0, c0], -1),
                                     np.stack([f1 * g1, c1], -1), np.stack([f0 * g1, c1], -1)], axis=1))
            else:
                zz, Aa, Bb, j0 = _zip(a0, a1)
                n0, n1 = len(i0), len(i1)
                for tri in zz:
                    idx, uv = [], []
                    for c in range(3):
                        rg_, l = tri[2 * c], tri[2 * c + 1]
                        if rg_ == 0:
                            idx.append(i0[l % n0])
                            uv.append(((Aa[l] / (2 * pi) - 0.5) * g0, cum[i]))
                        else:
                            idx.append(i1[(j0 + l) % n1])
                            uv.append(((Bb[l] / (2 * pi) - 0.5) * g1, cum[i + 1]))
                    TR.append(idx)
                    TUV.append(uv)
        if cap:
            i0, a0, g0 = rings[-1]
            m = len(i0)
            kk = np.arange(m)
            tipi = base + k
            for q in kk:
                TR.append([i0[q], i0[(q + 1) % m], tipi])
                TUV.append([((q / m - 0.5) * g0, cum[-1]), (((q + 1) / m - 0.5) * g0, cum[-1]), (((q + 0.5) / m - 0.5) * g0, cum[-1] + R[-1])])
        self.faces(np.vstack(Q) if Q else None, np.vstack(QUV) if QUV else None,
                   np.array(TR) if TR else None, np.array(TUV) if TUV else None)
        return T

    def mesh(self, name, uvk=(1.0, 1.0)):
        verts = np.vstack(self.v).astype(np.float32)
        quads = np.vstack(self.q).astype(np.int32) if self.q else np.zeros((0, 4), np.int32)
        tris = np.vstack(self.t).astype(np.int32) if self.t else np.zeros((0, 3), np.int32)
        uv = np.vstack(self.quv + self.tuv) * np.asarray(uvk)[None, :]
        return _to_mesh(name, verts, quads, tris, uv, np.concatenate(self.qm + self.tm))


def _to_mesh(name, verts, quads, tris, uv, slots):
    nq, nt = len(quads), len(tris)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.asarray(verts, dtype=np.float32).ravel())
    me.loops.add(nq * 4 + nt * 3)
    me.loops.foreach_set("vertex_index", np.concatenate([quads.ravel(), tris.ravel()]).astype(np.int32))
    me.polygons.add(nq + nt)
    me.polygons.foreach_set("loop_start", np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nt) * 3]).astype(np.int32))
    me.polygons.foreach_set("material_index", np.asarray(slots).astype(np.int32))
    me.update(calc_edges=True)
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", np.asarray(uv, dtype=np.float32).ravel())
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    uv_keep(o)
    return o


def _warp(p, sd):
    """Slow wobble (in plate cells) that keeps the plate outlines from being straight-edged polygons."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack([0.28 * np.sin(y * 5.3 + z * 3.1 + sd) + 0.18 * np.sin(z * 8.7 - x * 6.3 + 2.0 * sd),
                     0.28 * np.sin(x * 5.9 - z * 2.7 + 3.0 * sd) + 0.18 * np.sin(z * 7.9 + y * 6.7 + 4.0 * sd),
                     0.28 * np.sin(x * 4.7 + y * 5.1 + 5.0 * sd) + 0.18 * np.sin(z * 6.1 + x * 3.3 + 6.0 * sd)], axis=1)


def _plate_cells(q, sd):
    """Nearest jittered seed (3D Voronoi) for points q in cell units -> plate id, seed position."""
    c0 = np.floor(q).astype(np.int64)
    best = np.full(len(q), 1e9)
    pid = np.zeros(len(q), dtype=np.int64)
    sp = np.zeros((len(q), 3))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = c0 + np.array([dx, dy, dz])
                x = c[:, 0] * 127.1 + c[:, 1] * 311.7 + c[:, 2] * 74.7 + sd * 13.37
                jit = np.stack([np.sin(x + j * 91.3) * 43758.5453 for j in range(3)], axis=1)
                s = c + 0.12 + 0.76 * (jit - np.floor(jit))
                d = np.sum((q - s) ** 2, axis=1)
                m = d < best
                best[m] = d[m]
                pid[m] = ((c[m, 0] + 4096) << 26) | ((c[m, 1] + 4096) << 13) | (c[m, 2] + 4096)
                sp[m] = s[m]
    return pid, sp


def _skin(buf, name, scale, sd, uvk):
    """Build the underbark mesh. Where the mesh is fine and nearly flat (attribute 0 > 0) every face belongs to a bark plate
    (a 3D Voronoi cell, taller than wide): the vertices are split along the plate outlines so each plate can
    carry its own tone with a crisp edge, and the outline vertices are slid onto the true border between the
    two plates so the edge is an irregular plate outline, not a staircase of grid cells."""
    V = np.vstack(buf.v)
    A = np.vstack(buf.a)
    Q = np.vstack(buf.q) if buf.q else np.zeros((0, 4), np.int64)
    T = np.vstack(buf.t) if buf.t else np.zeros((0, 3), np.int64)
    nq, nt = len(Q), len(T)
    fc = np.vstack([V[Q].mean(axis=1), V[T].mean(axis=1)])
    is_pl = np.concatenate([A[Q, 0].max(axis=1), A[T, 0].max(axis=1)]) > 0.02
    pid_f = np.full(nq + nt, -1, dtype=np.int64)
    sp_f = np.zeros((nq + nt, 3))
    if is_pl.any():
        pid_f[is_pl], sp_f[is_pl] = _plate_cells(fc[is_pl] / scale + _warp(fc[is_pl], sd), sd)
    cv = np.concatenate([Q.ravel(), T.ravel()])
    cf = np.concatenate([np.repeat(np.arange(nq), 4), nq + np.repeat(np.arange(nt), 3)])
    cp = np.where(A[cv, 0] > 0.02, pid_f[cf], -1)
    order = np.lexsort((cp, cv))
    sv, spid = cv[order], cp[order]
    new = np.ones(len(sv), dtype=bool)
    new[1:] = (sv[1:] != sv[:-1]) | (spid[1:] != spid[:-1])
    nid = np.empty(len(cv), dtype=np.int64)
    nid[order] = np.cumsum(new) - 1
    first = order[new]
    v_old, v_pid, v_seed = cv[first], cp[first], sp_f[cf[first]]
    # slide the outline vertices onto the border between their two plates
    cnt = np.bincount(v_old, minlength=len(V))
    start = np.searchsorted(v_old, np.arange(len(V)))
    two = np.nonzero((cnt == 2) & (A[:, 2] < 0.5))[0]
    if len(two):
        two = two[v_pid[start[two]] >= 0]
    if len(two):
        fnq = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
        fnt = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        vn = np.zeros_like(V)
        emin = np.full(len(V), 1e9)
        for i in range(4):
            np.add.at(vn, Q[:, i], fnq)
            l = np.linalg.norm(V[Q[:, i]] - V[Q[:, (i + 1) % 4]], axis=1)
            np.minimum.at(emin, Q[:, i], l)
            np.minimum.at(emin, Q[:, (i + 1) % 4], l)
        for i in range(3):
            np.add.at(vn, T[:, i], fnt)
            l = np.linalg.norm(V[T[:, i]] - V[T[:, (i + 1) % 3]], axis=1)
            np.minimum.at(emin, T[:, i], l)
            np.minimum.at(emin, T[:, (i + 1) % 3], l)
        pv = V[two]
        wv = _warp(pv, sd)
        a_p, b_p = (v_seed[start[two]] - wv) * scale, (v_seed[start[two] + 1] - wv) * scale
        gval = np.sum(((pv - a_p) / scale) ** 2, axis=1) - np.sum(((pv - b_p) / scale) ** 2, axis=1)
        grad = 2.0 * (b_p - a_p) / scale ** 2
        Nn = _n(vn[two])
        td = grad - Nn * np.sum(grad * Nn, axis=1, keepdims=True)
        den = np.sum(td * grad, axis=1)
        ok = den > 0.15 * np.sum(grad * grad, axis=1)
        disp = td * (-gval / np.where(ok, den, 1.0))[:, None]
        dl = np.linalg.norm(disp, axis=1)
        disp = disp * np.minimum(1.0, 0.42 * emin[two] / np.maximum(dl, 1e-9))[:, None]
        V[two[ok]] += disp[ok]
    # (soft plates: the vertex is not split, it just takes the tone of the plate it lies in)
    soft = A[v_old, 0] < -0.02
    if soft.any():
        v_pid = v_pid.copy()
        v_pid[soft] = _plate_cells(V[v_old[soft]] / scale + _warp(V[v_old[soft]], sd), sd)[0]
    uv = np.vstack(buf.quv + buf.tuv) * np.asarray(uvk)[None, :]
    o = _to_mesh(name, V[v_old], nid[:nq * 4].reshape(-1, 4), nid[nq * 4:].reshape(-1, 3), uv, np.concatenate(buf.qm + buf.tm))
    return o, {"pid": v_pid, "pf": np.abs(A[v_old, 0]), "hard": A[v_old, 0] > 0.02, "fg": A[v_old, 1]}


class _Trunk:
    """The bole as a queryable surface: point at fractional ring index s and frame angle a (~ world azimuth)."""

    def __init__(self, P, R, rfn, nb):
        self.P, self.R, self.rfn, self.nb = P, R, rfn, nb
        self.T, self.N, self.B = _frames(P, flat=_sstep(0.05, 0.9, P[:, 2]))
        self.cum = _cum(P)

    def s_of_z(self, z):
        return np.interp(z, self.P[:self.nb, 2], np.arange(self.nb))

    def _at(self, s):
        s = np.asarray(s, dtype=np.float64)
        i = np.clip(np.floor(s).astype(int), 0, len(self.P) - 2)
        f = (s - i)[:, None]
        c = self.P[i] * (1 - f) + self.P[i + 1] * f
        Nn = _n(self.N[i] * (1 - f) + self.N[i + 1] * f)
        Bn = _n(self.B[i] * (1 - f) + self.B[i + 1] * f)
        R = self.R[i] * (1 - f[:, 0]) + self.R[i + 1] * f[:, 0]
        return i, f, c, Nn, Bn, R

    def pts(self, s, a, extra=0.0):
        i, f, c, Nn, Bn, R = self._at(s)
        rr = self.rfn(c[:, 2], R, a)
        d = np.cos(a)[:, None] * Nn + np.sin(a)[:, None] * Bn
        return c + (rr + extra)[:, None] * d, d, rr

    def inv(self, p):
        """World points -> (ring index s, frame angle a, distance from the axis, bole radius there)."""
        s = self.s_of_z(p[:, 2])
        for _ in range(3):
            i, f, c, Nn, Bn, R = self._at(s)
            Tn = np.cross(Nn, Bn)
            seg = np.linalg.norm(self.P[i + 1] - self.P[i], axis=1)
            s = np.clip(s + np.sum((p - c) * Tn, axis=1) / seg, 0.0, self.nb - 1.0)
        i, f, c, Nn, Bn, R = self._at(s)
        d = p - c
        x, y = np.sum(d * Nn, axis=1), np.sum(d * Bn, axis=1)
        a = np.arctan2(y, x)
        return s, a, np.hypot(x, y), self.rfn(c[:, 2], R, a)


def _refine(obj, more):
    """Splits faces of obj in four, the lowest first, until it has about `more` triangles more (see the budget
    of the foliage: the reduced levels of the town keep a share of ALL triangles, and take it from the wood)."""
    if more < 200:
        return
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    edges, n = set(), 0
    for f in sorted(bm.faces, key=lambda f: f.calc_center_median().z):
        if n >= more:
            break
        n += 3 * (len(f.verts) - 2)
        edges.update(f.edges)
    bmesh.ops.subdivide_edges(bm, edges=list(edges), cuts=1, use_grid_fill=True)
    bm.to_mesh(me)
    bm.free()
    me.update()


def _blades(buf, lvl, pos, xd, nn, size, fold, curl):
    """Stamp one blade of template `lvl` per row into buf: pos = petiole end, xd = towards the apex, nn = upper
    face. Each blade is mapped once onto the whole leaf tile (petiole at the tile's base, midrib on its axis)."""
    m = len(pos)
    if m == 0:
        return
    XY, TRI = LEAVES[lvl]
    nn = _n(nn)
    xd = _n(xd - nn * np.sum(xd * nn, axis=1, keepdims=True))
    yd = np.cross(nn, xd)
    lx, ly = XY[:, 0], XY[:, 1]
    # gently folded along the midrib, the apex curling down: one soft surface, shaded smooth
    lz = -fold[:, None] * np.abs(ly)[None, :] - curl[:, None] * (lx ** 2)[None, :]
    V = pos[:, None, :] + size[:, None, None] * (lx[None, :, None] * xd[:, None, :] + ly[None, :, None] * yd[:, None, :]
                                                  + lz[:, :, None] * nn[:, None, :])
    nv = len(XY)
    tris = TRI[None, :, :] + nv * np.arange(m)[:, None, None]
    uv1 = np.stack([0.006 + (lx + 0.13) / 1.13 * 0.238, 0.125 + ly * 0.185], axis=1)
    uv = np.broadcast_to(uv1[TRI][None, :, :, :], (m, len(TRI), 3, 2))
    buf.add(V.reshape(-1, 3), tris=tris.reshape(-1, 3), tuv=uv.reshape(-1, 2))


def _mix_hex(a, b, t):
    ca, cb = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(ca, cb))


def build(p, rng):
    g = np.random.default_rng(rng.fork("np").randrange(1 << 30))
    Lv = lod()
    H, W, bra = float(p.height), float(p.crown_width), float(p.branchiness)
    season = float(p.season)
    pollard = p.form == "pollard"
    zf = H * p.bole                               # fork height
    r1 = 0.031 * H * p.girth                      # trunk radius at breast height
    lean = radians(p.lean)
    ldir = np.array([cos(radians(p.lean_dir)), sin(radians(p.lean_dir)), 0.0])
    rt = rng.fork("trunk")
    ph = [rt.u(0, 2 * pi) for _ in range(6)]
    sd = int(p.seed) % 997
    mot, mos = float(p.mottle), float(p.moss)

    def axis(z):
        """Trunk centre line: leaves the ground already leaning, sweeps back a little higher up, wanders slowly."""
        z = max(float(z), 0.0)
        off = tan(lean) * z * (1.0 - 0.2 * z / H)
        s = float(_sstep(0.0, 1.8, z))
        wx = s * (0.06 * sin(z * 0.9 + ph[0]) + 0.028 * sin(z * 2.2 + ph[1]))
        wy = s * (0.06 * sin(z * 0.8 + ph[2]) + 0.028 * sin(z * 2.0 + ph[3]))
        return np.array([ldir[0] * off + wx, ldir[1] * off + wy, z])

    z_shr = [zf * 1.14]                           # the bole has given its wood to the scaffolds by this height

    def r_axis(z):
        return r1 * (1.0 - 0.13 * np.clip(z / zf, 0, 1)) * (1.0 - 0.42 * _sstep(zf * 0.8, z_shr[0], z))

    # ------------------------------------------------------------------ crown envelope and leaf masses
    rc_ = rng.fork("crown")
    zc = zf + (H - zf) * 0.45
    C = axis(zc) + np.array([rc_.u(-1, 1), rc_.u(-1, 1), 0.0]) * 0.05 * W
    ea, c_up, c_dn = W / 2 * 0.97, H - zc + 0.065 * H, zc - zf * 0.86
    pl = [rc_.u(0, 2 * pi) for _ in range(3)]

    def env(u):
        c = c_up if u[2] >= 0 else c_dn
        r = 1.0 / sqrt((u[0] ** 2 + u[1] ** 2) / ea ** 2 + u[2] ** 2 / c ** 2)
        az = atan2(u[1], u[0])
        return r * (1.0 + 0.15 * sin(3 * az + pl[0]) * (1 - abs(u[2])) + 0.1 * sin(2 * az + 4 * u[2] + pl[1]) + 0.07 * sin(5 * az + pl[2]))

    area = 4 * pi * ((ea * ea * (c_up + c_dn) / 2) ** (1 / 3)) ** 2 * 0.8
    opn = float(p.open)
    k_lift = clamp((W - 3.5) / 3.0, 0.2, 1.0)     # how far the crown has been lifted: least on a slender young tree
    k0 = int(round((14 + 17 * bra) * clamp(sqrt(area / 100.0), 0.75, 1.3) * (1.0 + 0.5 * opn)))
    dirs = []
    a0 = rc_.u(0, 2 * pi)
    for i in range(0 if pollard else k0):
        uz = 1 - 2 * (i + 0.5) / k0 + rc_.u(-0.05, 0.05)
        if uz < -0.42 - 0.3 * (1.0 - k_lift):     # (a crown that has not been lifted keeps its low masses)
            continue
        az = a0 + i * 2.39996 + rc_.u(-0.22, 0.22)
        s = sqrt(max(0.0, 1 - uz * uz))
        dirs.append(np.array([s * cos(az), s * sin(az), uz]))
    rc0 = clamp(0.65 * sqrt(area * 0.71 / max(len(dirs), 1)), 0.6, 1.6) * (1.0 - 0.1 * opn)
    clumps = []                                   # dict(E, rc, w)
    z_clear = 2.2                                 # crown lifted clear of heads and hats
    F0 = axis(zf)
    # the crown has been lifted, as street planes are: over the fork the canopy starts two metres up, so the
    # scaffold limbs and their forks stand bare against the lit underside of the leaves; it hangs lowest at
    # the drip line
    # (a slender young crown has not been lifted as far)
    bot_in, bot_out = zf + clamp(0.32 * (H - zf), 1.4, 2.2) * k_lift, max(z_clear, zf + clamp(0.17 * (H - zf), 0.75, 1.15) * k_lift)
    # sky gaps: two or three clefts run in from the outside of the crown towards the leader, each as wide and
    # as high as chance and `open` make it, so the crown is a few unequal lobes rather than one ball
    clefts = [(a0 + (j + rc_.u(-0.25, 0.25)) * 2 * pi / 3, (0.06 + 0.6 * opn) * rc_.u(0.6, 1.35), rc_.u(0.35, 0.8))
              for j in range(3 if opn > 0.05 else 0)]
    for i, u in enumerate(dirs):
        az = atan2(u[1], u[0])
        if any(abs((az - ca + pi) % (2 * pi) - pi) < cw_ * (1.0 - u[2] / ct) for (ca, cw_, ct) in clefts if u[2] < ct):
            continue
        rcl = rc0 * rc_.u(0.86, 1.18)
        e = env(u)
        E = C + u * max(e * rc_.u(0.86, 1.04) - 0.9 * rcl, 0.3 * e)
        k_out = float(_sstep(0.16 * W, 0.4 * W, hypot(E[0] - F0[0], E[1] - F0[1])))
        E[2] = max(E[2], bot_in + (bot_out - bot_in) * k_out + 0.72 * rcl + rc_.u(-0.1, 0.3))
        clumps.append({"E": E, "rc": rcl, "w": 1.0})
    if clumps:                                    # stretch the upper crown so the top leaves sit at `height`
        top = max(c["E"][2] + TOP_K * c["rc"] for c in clumps)
        sc = clamp((H - zc) / max(top - zc, 0.5), 0.8, 1.35)
        for c in clumps:
            if c["E"][2] > zc:
                c["E"][2] = zc + (c["E"][2] - zc) * sc
            # the low masses carry more leaf: they are the ceiling that catches the lantern light
            c["w"] = 1.0 + 0.3 * (1.0 - float(_sstep(zf + 1.0, zc + 1.0, c["E"][2])))
    for i in range(int(round(len(dirs) * 0.14 * (1.0 - opn)))):   # inner masses that close the upper middle of the crown
        uz = rc_.u(0.3, 0.9)
        az = rc_.u(0, 2 * pi)
        s = sqrt(1 - uz * uz)
        u = np.array([s * cos(az), s * sin(az), uz])
        E = C + u * env(u) * rc_.u(0.32, 0.52)
        E[2] = max(E[2], bot_in + 0.9 * rc0)
        clumps.append({"E": E, "rc": rc0 * 0.9, "w": 0.6})
    if pollard:
        # the "leaf masses" are the pollard heads: knuckles at staggered heights around a higher central one,
        # far enough apart that each carries its own tuft of regrowth with sky between it and the next
        n_h = 3 + int(round(3 * bra))
        for j in range(n_h):
            Lw = rc_.u(1.15, 1.6)                 # length of its shoots: two or three summers' regrowth
            if j == n_h - 1:
                E = axis(H - Lw * 0.95)
            else:
                az = a0 + j * 2 * pi / (n_h - 1) + rc_.u(-0.2, 0.2)
                zh = lerp(zf + 1.1, H - 2.1, ((j * 0.62) % 1.0) * 0.8 + rc_.u(0.0, 0.2))
                E = axis(zh) + np.array([cos(az), sin(az), 0.0]) * W * rc_.u(0.25, 0.31)
            clumps.append({"E": E, "rc": 1.2, "w": 1.0, "Lw": Lw})

    # ------------------------------------------------------------------ limb architecture: who carries which mass
    rb = rng.fork("limbs")
    under = _Buf()                                # smooth mottled underbark: bole above the foot, limbs, twigs
    rough = _Buf()                                # thick old bark: buttress, roots, the shell of plates
    twigs = _Buf()                                # the fine shoots that carry the leaves
    n_s = min(len(clumps) - 1, 5) if pollard else 2 + int(round(bra * 3))
    s_az = [rb.u(0, 2 * pi)]
    for j in range(1, n_s):
        s_az.append(s_az[0] + j * 2.39996 + rb.u(-0.3, 0.3))
    groups = {j: [] for j in range(n_s + 1)}      # 0 = the leader
    for ci, c in enumerate(clumps):
        v = c["E"] - F0
        if pollard:                               # every head has its own limb; the last one tops the leader
            groups[(ci + 1) % len(clumps)].append(ci)
            continue
        if v[2] / max(np.linalg.norm(v), 1e-6) > 0.9 and hypot(v[0], v[1]) < 0.22 * W:
            groups[0].append(ci)
            continue
        az = atan2(v[1], v[0])
        dj = [abs((az - s + pi) % (2 * pi) - pi) for s in s_az]
        groups[1 + int(np.argmin(dj))].append(ci)
    if not groups[0]:
        top = max(range(len(clumps)), key=lambda i: clumps[i]["E"][2])
        for gl in groups.values():
            if top in gl:
                gl.remove(top)
        groups[0].append(top)
    scaf = []
    Et0 = max(clumps[i]["E"][2] for i in groups[0])
    for j in range(1, n_s + 1):
        cl = groups[j]
        if not cl:
            continue
        zj = zf * (0.80 + 0.3 * (j - 0.5) / n_s)
        q0 = axis(zj)
        far = max(cl, key=lambda i: np.linalg.norm(clumps[i]["E"] - q0))
        E = clumps[far]["E"]
        outj = _n(np.array([E[0] - q0[0], E[1] - q0[1], 0.0]))
        ang = radians(rb.u(27, 48))
        rj = float(r_axis(zj))
        r0 = rj * rb.u(0.5, 0.63) * (1.1 if pollard else clamp(0.75 + 0.08 * len(cl), 0.8, 1.15))
        az = atan2(outj[1], outj[0])
        # Scaffolds are staggered up the bole: each base needs its own stretch of bark to grow a collar in, so
        # one that would crowd a lower neighbour on the same side leaves the bole higher up (or is a lesser limb).
        fit = False
        for shrink in (1.0, 0.86, 0.74):
            rs_ = r0 * shrink
            hh = 1.25 * rs_ / sin(ang) + 0.14
            z_hi = min(zf * 1.5, Et0 - 1.3 - (rj * 1.15 + 1.6 * rs_) / tan(ang) - rs_)
            for dz in np.arange(0.0, max(z_hi - zj, 0.0) + 1e-6, 0.07):
                zc_ = zj + dz + rj / tan(ang)
                hw = math.asin(min(1.25 * rs_ / (float(r_axis(zc_ + 0.5 * hh)) + 0.35 * rs_), 0.97)) + 0.35
                if all(abs((s_["az"] - az + pi) % (2 * pi) - pi) > s_["hw"] + hw or abs(s_["zc"] - zc_) > s_["hh"] + hh for s_ in scaf):
                    fit = True
                    break
            if fit:
                break
        if fit:
            zj, r0 = zj + float(dz), r0 * shrink
            q0 = axis(zj)
        z_e = zj + rj / tan(ang)                  # where the limb's underside clears the bole
        scaf.append({"j": j, "zj": zj, "p0": q0, "far": far, "out": outj, "ang": ang, "r0": r0, "hw": hw, "hh": hh, "zc": z_e,
                     "az": az, "zb": z_e - 0.5 * r0, "amp": 0.35 * r0, "hb": 3.2 * r0 + 0.25, "ha": 1.3 * r0})
    if scaf:
        z_shr[0] = max(z_shr[0], max(s_["zj"] for s_ in scaf) + 0.12 * zf)

    # ------------------------------------------------------------------ trunk + leader
    lodk = clamp(27.0 / max(len(clumps), 1), 0.72, 1.0)      # a many-limbed crown spends less on each limb

    def nsides(r):
        r = r * lodk ** 2
        return 14 if r > 0.115 else 12 if r > 0.085 else 9 if r > 0.055 else 7 if r > 0.032 else 5

    top_i = max(groups[0], key=lambda i: clumps[i]["E"][2])
    Et = clumps[top_i]["E"]
    # The bole is one fine grid of vertices up to above the highest scaffold: the limbs and the wounds are
    # welded into it, the shell of old bark is cut out of it, and its faces are small enough to carry the
    # plates of the mottle.
    z_g = zf * 1.06
    for sc in scaf:
        z_g = max(z_g, sc["zj"] + (float(r_axis(sc["zj"])) * 1.15 + 1.6 * sc["r0"]) / tan(sc["ang"]) + sc["r0"] + 0.25)
    z_g = min(z_g, zf * 1.8, Et[2] - 1.0)
    dzr = 0.055 * clamp(r1 / 0.3, 1.0, 1.3)       # rows of the grid: finer on a slender bole
    zs = [0.0, 0.035] + [float(z) for z in np.arange(0.08, z_g + dzr, dzr)]
    n_bole = len(zs)
    Pt = np.array([axis(z) for z in zs])
    t_dir = _n(axis(zs[-1] + 0.2) - axis(zs[-1]))
    Ll = float(np.linalg.norm(Et - Pt[-1]))
    tl = np.unique(np.round(np.concatenate([np.minimum(np.array([0.055, 0.11, 0.165, 0.22, 0.28, 0.35, 0.44]) / Ll, 0.3),
                                            np.linspace(min(0.6 / Ll, 0.45), 1.0, 10)]), 4))
    Pl = _bez(Pt[-1], Pt[-1] + t_dir * Ll * 0.4, Et - _n(UP * 0.8 + _n(Et - Pt[-1]) * 0.4) * Ll * 0.3, Et, tl)
    Pl = Pl + (np.array([cos(ph[4]), sin(ph[4]), 0.0])[None, :] * (0.06 * Ll * np.sin(pi * tl) * np.sin(2 * pi * tl + ph[5]))[:, None])
    P_tr = np.vstack([Pt, Pl])
    R_tr = r_axis(P_tr[:, 2])
    r_top = max(0.06, 0.42 * R_tr[n_bole - 1]) if pollard else 0.034
    R_tr[n_bole:] = R_tr[n_bole - 1] + (r_top - R_tr[n_bole - 1]) * tl ** 0.9

    def knuckle(P, R):
        """Pollard head: the cut-back limb end swells into a knobbly fist."""
        cum = _cum(P)
        rk = R[-1]
        # (a modest swelling, domed over: the head is the base of its shoots, never a bare club)
        R = R * (1.0 + 0.28 * np.exp(-((cum[-1] - cum) / (2.2 * rk)) ** 2))
        Tn = _n(P[-1] - P[-2])
        P = np.vstack([P, P[-1] + Tn * R[-1] * 0.5, P[-1] + Tn * R[-1] * 0.9])
        return P, np.concatenate([R, [R[-1] * 0.86, R[-1] * 0.5]])

    if pollard:
        P_tr, R_tr = knuckle(P_tr, R_tr)
    # root buttress: 4-6 distinct lobes, each one continued below as a surface root
    n_root = rt.pick([4, 5, 5, 6])
    roots = []
    for j in range(n_root):
        roots.append({"a": ph[0] + j * 2 * pi / n_root + rt.u(-0.3, 0.3), "s": rt.u(0.65, 1.15),
                      "h": clamp(r1 * rt.u(1.2, 2.0), 0.35, 0.6), "pw": rt.u(6.0, 10.0)})
    h_fl = 0.9 * r1 + 0.05

    def trunk_r(z, R, a):
        """Bole radius at height z, frame angle a: flared, lobed foot; flutes; a shoulder under every scaffold."""
        z = np.maximum(z, 0.0)
        # (little even flare: the foot is its lobes, knee-high, with the bole's own girth showing in the valleys)
        rr = R * (1.0 + 0.12 * np.exp(-z / h_fl) + 0.08 * np.exp(-z / 1.0))
        for rtj in roots:
            c = np.maximum(0.0, np.cos(a - rtj["a"]))
            rr = rr + r1 * 1.4 * rtj["s"] * np.exp(-(z / rtj["h"]) ** 1.2) * c ** rtj["pw"] + r1 * 0.07 * rtj["s"] * np.exp(-z / 2.2) * c ** 2
        for sc in scaf:
            dz = z - sc["zb"]
            fall = np.where(dz < 0, np.exp(-(dz / sc["hb"]) ** 2), np.exp(-(dz / sc["ha"]) ** 2))
            rr = rr + sc["amp"] * fall * np.maximum(0.0, np.cos(a - sc["az"])) ** 4
        # (flutes run straight up the bole: a phase that turns with height reads as a twisted rope)
        return rr * (1.0 + 0.03 * np.sin(2 * a + z * 0.22 + ph[1]) + 0.022 * np.sin(3 * a - z * 0.3 + ph[2]))

    trunk = _Trunk(P_tr, R_tr, trunk_r, n_bole)
    KS = n_bole                                   # grid rows = the rings of the bole
    MS = int(clamp(round(2 * pi * r1 * 1.12 / 0.05), 34, 50))       # columns around it
    da = 2 * pi / MS
    Sg, Ag = np.meshgrid(np.arange(KS, dtype=np.float64), np.arange(MS) * da, indexing="ij")
    S2, A2 = Sg.copy(), Ag.copy()                 # where each grid vertex sits on the bole (ring index, angle)
    lock = np.zeros((KS, MS), dtype=bool)         # vertices a weld depends on
    lock[-1, :] = True
    hole = np.zeros((KS - 1, MS), dtype=bool)     # cells cut away for a limb base or a wound
    ridge = np.zeros((KS, MS))                    # branch-bark ridges: radial swelling
    keep_off = np.zeros((KS, MS))                 # the old bark has long gone here
    pfg = np.ones((KS, MS))                       # strength of the plate mottle
    g_blk = len(under.v)
    gb = under.add(np.zeros((KS * MS, 3)))        # placed at the end, when everything is welded in

    def grid_pts():
        return trunk.pts(S2.ravel(), A2.ravel())[0].reshape(KS, MS, 3)

    def cut_hole(f3, pick=None):
        """Cut the grid cells with f3 < 1 out of the bole and return the loop of grid vertices around the hole
        (counter-clockwise seen from outside), moved onto the outline f3 = 1 so that whatever is built into
        the hole joins the bole along a smooth curve with shared vertices."""
        Pg = grid_pts()
        cc = (Pg[:-1] + Pg[1:] + np.roll(Pg, -1, axis=1)[:-1] + np.roll(Pg, -1, axis=1)[1:]) / 4
        ins = f3(cc.reshape(-1, 3)).reshape(KS - 1, MS) < 1.0
        if pick is not None and ins.any():        # only the patch of cells under the limb's base, not a stray one
            sc_ = np.where(ins, pick(cc.reshape(-1, 3)).reshape(KS - 1, MS), 1e9)
            k0_, m0_ = np.unravel_index(int(np.argmin(sc_)), sc_.shape)
            one = np.zeros_like(ins)
            one[k0_, m0_] = True
            stack = [(int(k0_), int(m0_))]
            while stack:
                a_, b_ = stack.pop()
                for a2, b2 in ((a_ + 1, b_), (a_ - 1, b_), (a_, (b_ + 1) % MS), (a_, (b_ - 1) % MS)):
                    if 0 <= a2 < KS - 1 and ins[a2, b2] and not one[a2, b2]:
                        one[a2, b2] = True
                        stack.append((a2, b2))
            ins = one
        kk_, mm_ = np.nonzero(ins)
        if len(kk_) < 2 or kk_.min() < 2 or kk_.max() > KS - 4:
            return None
        near = np.zeros((KS, MS), dtype=bool)
        for dk_ in (0, 1):
            for dm_ in (0, 1):
                near[kk_ + dk_, (mm_ + dm_) % MS] = True
        for _ in range(1):
            n2 = near | np.roll(near, 1, axis=1) | np.roll(near, -1, axis=1)
            n2[1:] |= near[:-1]
            n2[:-1] |= near[1:]
            near = n2
        if (near & lock).any():
            return None
        nxt = {}
        for k_, m_ in zip(kk_.tolist(), mm_.tolist()):
            m1_ = (m_ + 1) % MS
            for (va_, vb_, k2, m2) in (((k_, m_), (k_, m1_), k_ - 1, m_), ((k_, m1_), (k_ + 1, m1_), k_, m1_),
                                       ((k_ + 1, m1_), (k_ + 1, m_), k_ + 1, m_), ((k_ + 1, m_), (k_, m_), k_, (m_ - 1) % MS)):
                if not ins[k2, m2]:
                    if va_ in nxt:
                        return None
                    nxt[va_] = vb_
        v0 = next(iter(nxt))
        loop = [v0]
        while nxt[loop[-1]] != v0:
            loop.append(nxt[loop[-1]])
            if len(loop) > len(nxt):
                return None
        if len(loop) != len(nxt):
            return None
        lk, lm = np.array([v[0] for v in loop]), np.array([v[1] for v in loop])
        s0, a0_ = S2[lk, lm].copy(), A2[lk, lm].copy()
        s, a = s0.copy(), a0_.copy()
        fv = lambda s_, a_: f3(trunk.pts(s_, a_)[0])
        for _ in range(4):
            f0 = fv(s, a)
            fs = (fv(s + 0.3, a) - fv(s - 0.3, a)) / 0.6
            fa = (fv(s, a + 0.3 * da) - fv(s, a - 0.3 * da)) / 0.6
            den = fs * fs + fa * fa + 1e-9
            s = np.clip(s - (f0 - 1.0) * fs / den, s0 - 0.6, s0 + 0.6)
            a = np.clip(a - (f0 - 1.0) * fa / den * da, a0_ - 0.6 * da, a0_ + 0.6 * da)
        S2[lk, lm], A2[lk, lm] = s, a
        lock[lk, lm] = True
        hole[kk_, mm_] = True
        return lk, lm

    # the leader carries on from the top row of the grid
    sl = slice(n_bole - 1, None)
    pf_l = 1.0 - _sstep(0.1, 0.33, trunk.cum[sl] - trunk.cum[n_bole - 1])
    if Lv < 2:                                    # (lod 2: bole and leader are one plain tube, made further down)
        under.tube(P_tr[sl], R_tr[sl], [max(nsides(r), 20 if q > 0.02 else 0) for r, q in zip(R_tr[sl], pf_l)],
                   frames=(trunk.T[sl], trunk.N[sl], trunk.B[sl]),
                   rfn=lambda i, A: trunk_r(np.full(len(A), P_tr[n_bole - 1 + i, 2]), R_tr[n_bole - 1 + i], A),
                   first=(gb + (KS - 1) * MS + np.arange(MS), np.arange(MS) * da))
    branches = []                                 # dict(P, R, T, cum, grp, i0)
    crotches = []                                 # (point, radius): darkened, rougher bark

    def reg(P, R, T, grp, i0=2):
        br = {"P": P, "R": R, "T": T, "cum": _cum(P), "grp": grp, "i0": min(i0, len(P) - 1)}
        branches.append(br)
        return br

    reg(P_tr, R_tr, trunk.T, 0, i0=n_bole)

    def clear(Pq, rq, par):
        """True if a limb along Pq (radii rq) keeps its own diameter of air between itself and every other limb,
        its parent included once it has left it."""
        for br in branches:
            A = br["P"][:-1]
            ab = br["P"][1:] - A
            tq = np.clip(np.sum((Pq[:, None, :] - A[None, :, :]) * ab[None, :, :], axis=2) / np.maximum(np.sum(ab * ab, axis=1), 1e-9)[None, :], 0, 1)
            gap = np.linalg.norm(Pq[:, None, :] - (A[None, :, :] + ab[None, :, :] * tq[:, :, None]), axis=2) - br["R"][:-1][None, :] - 3.0 * rq[:, None]
            if gap[len(Pq) * (3 if br is par else 1) // 10:].min() < 0.0:
                return False
        return True

    def limb(p0, d0, p3, d1, r0, r_end, grp, par_T, par_r, wig=0.04, flare=0.8, weld=False, par=None):
        """Swept limb from p0 (on the parent's axis, heading d0) to p3 (arriving along d1). It wanders and has
        two elbows where an old leader was lost; None if it would crowd another limb (par = the limb it forks
        from). A scaffold is welded into the bole: the cells of the bole's
        grid under its base are cut away and a collar of faces runs from the edge of that hole (the bole's own
        vertices) out to the first free ring of the limb, so bark, shading and mottle run from one into the
        other. A lesser limb swells into a collar where it leaves its parent."""
        L = float(np.linalg.norm(p3 - p0))
        ax = _n(p3 - p0)
        side = np.cross(ax, UP)
        side = _n(side) if np.linalg.norm(side) > 0.05 else np.array([1.0, 0, 0])
        upv = np.cross(side, ax)
        # (wood grows in straight runs between the forks where a leader was lost: the wander is a fraction of the
        # limb's own thickness, the elbows are shallow - a limb that snakes reads as a rubber hose)
        w1, w2, w3, w4 = rb.u(0.45, 0.8), rb.u(0, 1), rb.u(0.4, 0.75), rb.u(0, 1)
        wig = min(wig * 0.3, 0.35 * r0 / max(L, 1e-3))
        t0 = rb.u(0.3, 0.55)
        kd = _n(side * rb.u(-1, 1) + upv * rb.u(0.15, 1.0))
        ka = rb.u(0.012, 0.03) if pollard else rb.u(0.025, 0.05)
        t1 = t0 + rb.u(0.2, 0.35)
        kd1 = _n(side * rb.u(-1, 1) - upv * rb.u(0.0, 0.7))
        ka1 = 0.5 * ka * rb.u(0.5, 1.0)

        def curve(t):
            e = np.sin(pi * t) ** 0.8
            return (_bez(p0, p0 + d0 * L * 0.4, p3 - d1 * L * 0.3, p3, t)
                    + side[None, :] * (wig * L * e * np.sin(2 * pi * (w1 * t + w2)))[:, None]
                    + upv[None, :] * (wig * 0.7 * L * e * np.sin(2 * pi * (w3 * t + w4)))[:, None]
                    + kd[None, :] * (ka * L * np.minimum(t / t0, (1 - t) / (1 - t0)))[:, None]
                    + kd1[None, :] * (ka1 * L * np.minimum(t / t1, (1 - t) / (1 - t1)))[:, None])

        tf = np.linspace(0.0, 1.0, 161)
        Pf = curve(tf)
        if par is not None and not clear(Pf[::8], r0 + (r_end - r0) * tf[::8] ** 0.8, par):
            return None
        cumf = _cum(Pf)
        Lc = float(cumf[-1])
        n = int(clamp(round(L * lodk / clamp(1.7 * r0, 0.12, 0.45)) + 2, 6, 34))
        tr_ = np.linspace(0, 1, n)
        tr_ = 1 - (1 - tr_) ** 1.6 if pollard else tr_ ** 1.4       # rings closest where the limb is thick and seen
        s_reg = np.interp(tr_, tf, cumf)
        q1, q2 = rb.u(0, 6.3), rb.u(0, 6.3)

        def lob(A, s):
            return 1.0 + 0.09 * np.sin(2 * A + s * 0.3 + q1) + 0.04 * np.sin(3 * A - s * 0.5 + q2)

        def rings(s_list):
            tt_ = np.interp(s_list, cumf, tf)
            return curve(tt_), r0 + (r_end - r0) * tt_ ** 0.8

        if weld:
            # the first ring that stands clear of the bole all round
            Tf, Nf, Bf = _frames(Pf)
            A8 = np.arange(8) * pi / 4
            s_c = None
            for i in range(3, 110):
                rr_ = r0 + (r_end - r0) * tf[i] ** 0.8
                ring = Pf[i][None, :] + rr_ * (np.cos(A8)[:, None] * Nf[i][None, :] + np.sin(A8)[:, None] * Bf[i][None, :])
                s_t, a_t, rho, rs = trunk.inv(ring)
                if s_t.max() > KS - 5:
                    break
                if float((rho - rs).min()) > 0.25 * rr_ + 0.02:
                    s_c = float(cumf[i])
                    break
            if s_c is not None and Lc > s_c + 1.3:
                nd = int(clamp((Lc - s_c) * 0.45 / 0.09, 4, 10))
                span = 0.09 * (nd - 1)
                s_list = np.concatenate([s_c + 0.09 * np.arange(nd), s_reg[s_reg > s_c + span + 0.09]])
                P, R = rings(s_list)
                fine = s_list - s_c < 0.95 * span                  # round where it meets the bole
                if pollard:
                    P, R = knuckle(P, R)
                    fine = np.concatenate([fine, [False, False]])
                cum = _cum(P)
                T, N, B = _frames(P)
                Pc, Tc, Nc, Bc, R0 = P[0], T[0], N[0], B[0], float(R[0])

                def polar(X):
                    d = X - Pc[None, :]
                    return d @ Tc, np.hypot(d @ Nc, d @ Bc), np.arctan2(d @ Bc, d @ Nc)

                def f3(X, kf=1.25):
                    axl, rp, phi = polar(X)
                    return np.where(axl > -s_c, rp / (R0 * lob(phi, s_c) * kf), 9.0)

                pick = lambda X: -polar(X)[0]
                res = cut_hole(f3, pick)
                if res is None:               # crowded fork: a tighter collar may still fit between its neighbours
                    f3 = lambda X, f_=f3: f_(X, 1.08)
                    res = cut_hole(f3, pick)
                if res is not None:
                    lk, lm = res
                    axl, rp, phi = polar(trunk.pts(S2[lk, lm], A2[lk, lm])[0])
                    phi_u = phi[0] + np.concatenate([[0.0], np.cumsum((np.diff(phi) + pi) % (2 * pi) - pi)])
                    phi_u = np.maximum.accumulate(phi_u) + 1e-4 * np.arange(len(phi_u))
                    nl = len(lk)
                    rr0 = R0 * lob(phi_u, s_c)
                    M = int(clamp(math.ceil(max(float(-axl.min()), 0.0) / 0.085), 2, 7))
                    rows = [gb + lk * MS + lm]
                    dr_ = np.cos(phi_u)[:, None] * Nc[None, :] + np.sin(phi_u)[:, None] * Bc[None, :]
                    for q in range(1, M + 1):
                        tq = q / M
                        X = Pc[None, :] + Tc[None, :] * (axl * (1 - tq))[:, None] + dr_ * (rr0 * (1.0 + (rp / rr0 - 1.0) * (1 - tq) ** 2))[:, None]
                        rows.append(under.add(X) + np.arange(nl))
                    jj = np.arange(nl)
                    j1 = (jj + 1) % nl
                    u0 = (phi_u / (2 * pi) - 0.5) * 2 * pi * R0
                    u1 = (np.concatenate([phi_u[1:], [phi_u[0] + 2 * pi]]) / (2 * pi) - 0.5) * 2 * pi * R0
                    for q in range(M if Lv < 2 else 0):
                        va_, vb_ = axl * (1 - q / M), axl * (1 - (q + 1) / M)
                        under.faces(quads=np.stack([rows[q][jj], rows[q][j1], rows[q + 1][j1], rows[q + 1][jj]], axis=1),
                                    quv=np.stack([np.stack([u0, va_], -1), np.stack([u1, va_[j1]], -1),
                                                  np.stack([u1, vb_[j1]], -1), np.stack([u0, vb_], -1)], axis=1))
                    if Lv < 2:
                        under.tube(P, R, [max(nsides(r), 14 if q else 0) for r, q in zip(R, fine)],
                                   rfn=lambda i, A: R[i] * lob(A, s_c + cum[i]), frames=(T, N, B), first=(rows[-1], phi_u))
                    else:                     # far level: no collar - the limb runs on into the bole from its axis,
                        # as stout there as its collar (or the crotch opens to the sky)
                        under.tube(np.vstack([p0[None, :], P]), np.concatenate([[1.35 * R0], R]), [nsides(R0)] + [nsides(r) for r in R])
                    # branch-bark ridge: the bole's bark is pushed up into a welt over the crotch
                    Pg = grid_pts().reshape(-1, 3)
                    axg, rpg, phg = polar(Pg)
                    fg_ = f3(Pg)
                    e_up = par_T - Tc * float(Tc @ par_T)
                    phi_up = atan2(float(e_up @ Bc), float(e_up @ Nc))
                    bump = (0.012 + 0.06 * R0) * _sstep(1.0, 1.15, fg_) * (1 - _sstep(1.25, 1.7, fg_)) * _sstep(0.0, 0.8, np.cos(phg - phi_up))
                    ridge[:] = np.maximum(ridge, bump.reshape(KS, MS))
                    keep_off[:] = np.maximum(keep_off, (1.0 - _sstep(1.5, 2.3, fg_)).reshape(KS, MS))
                    # the plate mottle dies away over half a metre towards the fork: limbs wear one even skin
                    pfg[:] = np.minimum(pfg, _sstep(1.3, 1.3 + 0.5 / (1.25 * R0), fg_).reshape(KS, MS))
                    crotches.append((Pc + _n(e_up) * R0, r0))
                    return reg(P, R, T, grp, i0=int(np.searchsorted(cum, 0.5)))
            if weld == 1:
                return None

        sn = sqrt(max(1.0 - float(np.dot(d0, par_T)) ** 2, 0.0))
        s_e = min(par_r / max(sn, 0.42), 0.4 * Lc)                # where the limb clears its parent
        s_list = np.concatenate([[0.45 * s_e, s_e], s_reg[s_reg > s_e + 0.05]])
        P, R = rings(s_list)
        if pollard:
            P, R = knuckle(P, R)
        cum = _cum(P)
        T, N, B = _frames(P)
        Fl = flare * np.exp(-np.maximum(cum + s_list[0] - s_e, 0.0) / (1.5 * r0 + 0.02))
        e_up = par_T[None, :] - T * (T @ par_T)[:, None]          # towards the crotch
        a_up = np.arctan2(np.sum(e_up * B, axis=1), np.sum(e_up * N, axis=1))

        def rfn_u(i, A):
            d_ = A - a_up[i]
            return R[i] * (1.0 + Fl[i] * (0.18 + 0.62 * (0.5 - 0.5 * np.cos(d_)) + 0.25 * np.cos(d_) ** 2)) * lob(A, cum[i])

        under.tube(P, R, [nsides(r) for r in R], rfn=rfn_u if r0 > 0.035 else None, frames=(T, N, B))
        crotches.append((P[1] + _n(e_up[1]) * r0, r0))
        return reg(P, R, T, grp)

    crown_core = C - UP * 0.25 * c_dn

    def arrive(q, E):
        o = _n(E - crown_core)
        if pollard:
            return _n(UP * 0.8 + _n(E - q) * 0.45)
        return _n(_n(E - q) * 0.6 + o * 0.45 + UP * 0.3 * o[2])

    def attach(ci, grp, low=None):
        """Graft the limb for mass ci onto a limb of its group: at `low` = (limb, ring) if given, else on the
        nearest ones, leaving them well back so the fork is acute. A graft that would crowd another limb is
        not made; None if there is no clear way to the mass."""
        E = clumps[ci]["E"]
        cands = [] if low is None else [low]
        near = []
        for br in branches:
            if br["grp"] == grp:
                d = np.linalg.norm(br["P"][br["i0"]:] - E, axis=1)
                near.append((float(d.min()), br, int(np.argmin(d)) + br["i0"]))
        for dmin, br, i in sorted(near, key=lambda c: c[0])[:3]:
            for back in (0.8, 1.2, 0.45):
                j = i
                while j > br["i0"] and br["cum"][i] - br["cum"][j] < back * dmin:
                    j -= 1
                if all(b_ is not br or j_ != j for b_, j_ in cands):
                    cands.append((br, j))
        for br, j in cands:
            q = br["P"][j]
            rp = br["R"][j]
            d0 = _n(br["T"][j] * 0.62 + _n(E - q) * 0.5)
            r0 = clamp(rp * (rb.u(0.64, 0.8) if low is not None else rb.u(0.56, 0.74)), min(0.03, rp * 0.9), 0.24)
            nb = limb(q, d0, E, arrive(q, E), r0, max(0.017, r0 * 0.36), grp, br["T"][j], rp, flare=0.5, par=br)
            if nb is not None:
                clumps[ci]["br"] = nb
                return nb
        return None

    # leader's own mass, then the scaffolds, then every other mass grafts onto the nearest limb of its group
    clumps[top_i]["br"] = branches[0]
    hang_pts = []
    for sc in scaf:
        E = clumps[sc["far"]]["E"]
        d0 = _n(UP * cos(sc["ang"]) + sc["out"] * sin(sc["ang"]))
        br, zj0 = None, sc["zj"]
        for dz, mode in ((0.0, 1), (0.14, 1), (0.28, 1), (-0.14, 1), (0.42, 1), (0.56, 1), (-0.28, 1), (0.7, 1), (0.0, 2)):     # no room for its collar here: try a little higher
            zj = zj0 + dz
            sc.update(zj=zj, p0=axis(zj), zb=zj + sc["zb"] - sc["zj"])
            par_T = _n(axis(zj + 0.1) - axis(zj - 0.1))
            br = limb(sc["p0"], d0, E, arrive(sc["p0"], E), sc["r0"], max(0.02, sc["r0"] * (0.5 if pollard else 0.27)), sc["j"],
                      par_T, float(r_axis(zj)) * 1.06, wig=0.05, flare=0.85, weld=mode)
            if br is not None:
                break
        clumps[sc["far"]]["br"] = br
        k = min(int(np.searchsorted(br["cum"], 0.42 * br["cum"][-1])), len(br["P"]) - 1)
        hang_pts.append(br["P"][k] - UP * br["R"][k])
    for j in range(0, n_s + 1):
        rest = [i for i in groups[j] if "br" not in clumps[i]]
        rest.sort(key=lambda i: -np.linalg.norm(clumps[i]["E"] - F0))
        sb = next((b_ for b_ in branches[1:] if b_["grp"] == j), None)
        # every scaffold forks low, below the leaves, towards the mass that lies furthest off its own line -
        # a big one twice
        for k in range(0 if sb is None else 1 + (len(rest) >= 5)):
            i = int(np.searchsorted(sb["cum"], sb["cum"][-1] * (rb.u(0.24, 0.36) + 0.2 * k)))
            i = min(max(i, sb["i0"]), len(sb["P"]) - 2)
            v = [clumps[c_]["E"] - sb["P"][i] for c_ in rest]
            off = [float(np.linalg.norm(np.cross(v_, sb["T"][i]))) * (float(v_ @ sb["T"][i]) > 0.3 * np.linalg.norm(v_)) for v_ in v]
            if off and max(off) > 0.0:
                ci = rest[int(np.argmax(off))]
                if attach(ci, j, low=(sb, i)) is not None:
                    rest.remove(ci)
        for ci in rest:
            if "br" not in clumps[ci] and attach(ci, j) is None:
                attach(ci, 0)                     # no clear way from its own scaffold: off the leader instead

    # what a street tree's limbs carry below the leaves: stubs of pruned branches, callused over, and burrs
    rk_ = rng.fork("knobs")
    for br in branches[1:]:
        r_b = float(br["R"][0])
        cum, P = br["cum"], br["P"]
        if r_b < 0.05 or len(P) < 5:
            continue
        for k in range(rk_.pick([0, 1, 1, 2]) if r_b > 0.085 else rk_.pick([0, 0, 1])):
            i = int(np.clip(np.searchsorted(cum, cum[-1] * rk_.u(0.1, 0.55)), 1, len(P) - 2))
            T, R = br["T"][i], float(br["R"][i])
            sd_ = np.cross(T, UP)
            sd_ = _n(sd_) if np.linalg.norm(sd_) > 0.05 else np.array([1.0, 0, 0])
            a = rk_.u(-0.5, 0.5) + rk_.pick([0.0, pi, pi / 2 + rk_.u(-0.4, 0.4) * 2])
            dv = sd_ * cos(a) - np.cross(sd_, T) * sin(a)     # sideways or underneath, where the low wood was taken off
            if rk_.chance(0.4):                   # a flush pruning cut: a callus ring barely proud of the limb
                rs, ln = R * rk_.u(0.3, 0.42), R * rk_.u(0.06, 0.12)     # (never a stub: a stub reads as a club)
                dv = _n(dv + T * rk_.u(0.1, 0.3))
                Ps = P[i] + dv[None, :] * np.array([0.5 * R, R + 0.4 * ln, R + ln, R + ln + 0.003])[:, None]
                if Lv < 2:
                    under.tube(Ps, rs * np.array([1.5, 1.2, 0.85, 0.4]), 8)
            else:                                 # a burr: a low boss of dormant buds
                rs = R * rk_.u(0.3, 0.45)
                Ps = P[i] + dv[None, :] * np.array([0.6 * R, R + 0.06 * rs, R + 0.2 * rs])[:, None]
                if Lv < 2:
                    under.tube(Ps, rs * np.array([1.3, 0.9, 0.42]), 8)

    # ------------------------------------------------------------------ twig sprays
    rw = rng.fork("twigs")
    def stem(P, R, sides):
        """A shoot: a tube closed at both ends (a closed tube collapses away at the reduced levels; an open one cannot).
        Levels: a stout shoot (a pollard's leaders) stays a tube, open at its buried base; a thin one is a two-sided
        ribbon at lod 1, turned another way from one shoot to the next, and is left out at lod 2."""
        if Lv:
            if R[0] > 0.022:
                twigs.tube(P, R, sides, cap=True)
            elif Lv == 1:
                i3 = [0, len(P) // 2, len(P) - 1]
                Tt = _n(P[-1] - P[0])
                e = np.cross(Tt, UP)
                e = _n(e) if np.linalg.norm(e) > 0.05 else np.array([1.0, 0, 0])
                if len(stems) % 2:
                    e = np.cross(Tt, e)
                cu = _cum(P[i3])
                twigs.add(np.vstack([P[i] + sg * 1.3 * R[i] * e for i in i3 for sg in (-1.0, 1.0)]), quads=[[0, 1, 3, 2], [2, 3, 5, 4]],
                          quv=[[0, cu[0]], [0.03, cu[0]], [0.03, cu[1]], [0, cu[1]], [0, cu[1]], [0.03, cu[1]], [0.03, cu[2]], [0, cu[2]]])
            return
        b0 = twigs.nv
        twigs.tube(P, R, sides, cap=True)
        twigs.faces(tris=[[b0, b0 + k + 1, b0 + k] for k in range(1, sides - 1)], tuv=np.zeros((3 * (sides - 2), 2)))

    stems = []                                    # (P, clump index, length, is_whip)
    n_tw = 4 + int(round(3 * bra))
    n_wh = 7 + int(round(5 * bra))
    bare = clamp(season * 0.9 + (1.0 - p.leaf_density) * 0.8, 0.0, 1.0)     # how visible the fine twigs are
    for ci, c in enumerate(clumps if pollard else []):
        # pollard regrowth: a brush of straight, vigorous shoots off every knuckle, curving up to the light
        br = c.get("br")
        if br is None:
            continue
        E = c["E"]
        rk = float(br["R"][-3])
        for k in range(n_wh):
            th, az = radians(rw.u(4, 62)), rw.u(0, 2 * pi)
            d = np.array([sin(th) * cos(az), sin(th) * sin(az), cos(th)])
            Lw = c["Lw"] * rw.u(0.68, 1.0)
            q0 = E + d * rk * 0.55 + _n(np.array([d[0], d[1], 0.0])) * rk * 0.4
            q3 = q0 + d * Lw
            P = _bez(q0, q0 + d * Lw * 0.33, q3 - _n(d * 0.45 + UP) * Lw * 0.3, q3, 6)
            # (the first few are stout: the head divides into its leaders, it does not sprout wires)
            r0 = rk * rw.u(0.3, 0.44) * (1.0 - 0.1 * k) if k < 5 else rw.u(0.016, 0.028)
            stem(P, r0 + (0.0045 - r0) * np.linspace(0, 1, 6) ** 0.8, 6 if r0 > 0.035 else 4 if r0 > 0.022 else 3)
            stems.append((P, ci, Lw, 1))
            for m in range(rw.pick([1, 2, 2, 3])):
                i = rw.pick([1, 2, 3])
                q = P[i] + (P[i + 1] - P[i]) * rw.u(0, 1)
                Tt = _n(P[i + 1] - P[i])
                ds = _n(Tt * 0.7 + _n(np.array([rw.u(-1, 1), rw.u(-1, 1), rw.u(-0.2, 0.5)])) * 0.75)
                Ls = Lw * rw.u(0.14, 0.26)
                Ps = np.array([q, q + ds * Ls * 0.5, q + _n(ds + UP * 0.25) * Ls])
                stem(Ps, np.array([0.008, 0.0055, 0.0035]), 3)
                stems.append((Ps, ci, Ls, 1))
    for ci, c in enumerate([] if pollard else clumps):
        br = c.get("br")
        if br is None:
            continue
        E, rcl = c["E"], c["rc"]
        o = _n(E - crown_core)
        cum = br["cum"]
        span = min(cum[-1] * 0.6, 1.25 * rcl)
        nt = max(3, int(round(n_tw * (0.75 if c["w"] < 1 else 1.0))))
        for k in range(nt):
            s = cum[-1] - span * (1.0 - (k + 1) / nt)
            i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(cum) - 2))
            f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-6)
            q0 = br["P"][i] + (br["P"][i + 1] - br["P"][i]) * f
            Tp = br["T"][i]
            rp = br["R"][i] + (br["R"][i + 1] - br["R"][i]) * f
            v = _n(np.array([rw.u(-1, 1), rw.u(-1, 1), rw.u(-0.7, 0.7)]))
            d = o * 0.55 + v * 0.95 + Tp * (0.9 if k == nt - 1 else 0.3)
            d[2] = max(d[2] * (0.95 if d[2] > 0 else 0.7), -0.3)
            d = _n(d)
            L = rcl * rw.u(0.7, 1.1)
            droop = rw.u(0.12, 0.34)
            q3 = q0 + d * L - UP * L * droop
            P = _bez(q0, q0 + _n(Tp * 0.5 + d) * L * 0.35, q3 - _n(d - UP * droop * 2.4) * L * 0.3, q3, 5)
            r0 = min(0.021, rp * 0.8)
            # the mesh is coarser than the path the leaves follow: these twigs are buried in foliage
            Pm = P if bare > 0.4 else P[[0, 2, 4]]
            stem(Pm, r0 + (0.0045 - r0) * np.linspace(0, 1, len(Pm)) ** 0.8, 3)
            stems.append((P, ci, L, 0))
            Tt = _n(P[3] - P[1])                  # side shoots
            lat = np.cross(Tt, UP)
            lat = _n(lat) if np.linalg.norm(lat) > 0.05 else np.array([1.0, 0, 0])
            for sgn in (-1, 1):
                if not rw.chance(0.85):
                    continue
                ts = rw.u(0.3, 0.72)
                q = P[1] + (P[3] - P[1]) * (ts - 0.25) / 0.5 if 0.25 <= ts <= 0.75 else P[2]
                ds = _n(Tt * 0.55 + lat * sgn * rw.u(0.6, 1.0) + UP * rw.u(-0.25, 0.3))
                Ls = L * rw.u(0.32, 0.55)
                Ps = np.array([q, q + ds * Ls * 0.5 - UP * Ls * 0.03, q + ds * Ls - UP * Ls * rw.u(0.1, 0.3)])
                stem(Ps, np.array([0.009, 0.006, 0.0035]), 3)
                stems.append((Ps, ci, Ls, 0))

    # ------------------------------------------------------------------ surface roots (old bark)
    rp_ = clamp(r1 * 4.6, 1.0, 1.8)               # tree pit radius
    root_runs = []

    def root_top(tt, h0):
        """Top of a surface root above the paving: high where it leaves the buttress, sinking into the ground
        well before its end - a root runs on under the gravel, it does not stop on top of it."""
        return (0.03 + 0.2 * h0) * np.exp(-3.2 * tt) + 0.085 * (1 - tt) ** 0.6 - 0.02

    for rtj in roots:
        a = rtj["a"]
        rho0 = float(trunk_r(np.array([0.12]), R_tr[0], np.array([a]))[0]) - 0.07     # it leaves the toe of its buttress lobe
        # it runs on as a ridge for 0.5-0.8 m (a weak one less) before it dives; the strong ones go under the ring
        reach_v = rho0 + (rt.u(0.5, 0.8) if rtj["s"] > 0.85 else rt.u(0.35, 0.55)) / 0.78
        reach = max(min(reach_v, rp_ - 0.08 if p.surround == "setts" else 0.8 * rp_), rho0 + 0.3)
        tt = np.linspace(0, 1, 12)
        rho = rho0 + (reach - rho0) * tt
        aw = a + 0.3 * rt.u(-1, 1) * tt ** 1.5 + 0.09 * np.sin(tt * 7 + rt.u(0, 6))
        h0 = 0.8 * rtj["h"] * rtj["s"]
        top = root_top(tt, h0) + 0.015 * np.sin(tt * 9 + rt.u(0, 6)) * tt * (1 - tt) * 4
        # only the back of the root shows: its section is widest at ground level and flat below, and it keeps
        # most of its width to where it dives - a ridge that sinks under the gravel, not a claw lying on it
        wu = np.maximum(top + 0.01, 0.008)
        wh = r1 * 0.33 * rtj["s"] * (1 - 0.35 * tt) + 0.03
        if p.surround == "none":                  # under a tree grate there is no open ground to heave
            continue
        Pr = np.stack([rho * np.cos(aw), rho * np.sin(aw), np.full(len(tt), -0.01)], axis=1)
        fr_ = _frames(Pr)
        a_v = np.arctan2(fr_[2][:, 2], fr_[1][:, 2])
        wu_, wh_ = wu, wh
        if Lv:                                    # levels: the ridge through fewer rings, with fewer sides
            kr = [[0, 2, 4, 6, 8, 11], [0, 4, 8, 11]][Lv - 1]
            Pr, fr_, a_v, wu_, wh_ = Pr[kr], tuple(f_[kr] for f_ in fr_), a_v[kr], wu[kr], wh[kr]
        rough.tube(Pr, wh_, (8, 6, 4)[Lv], cap=False, frames=fr_, exact=True,
                   rfn=lambda i, A, wu=wu_, wh=wh_, a_v=a_v: 1.0 / np.sqrt((np.cos(A - a_v[i]) / np.where(np.cos(A - a_v[i]) > 0, wu[i], 0.02)) ** 2
                                                                         + (np.sin(A - a_v[i]) / wh[i]) ** 2))
        root_runs.append({"a0": a, "a1": float(aw[-1]), "reach": reach, "reach_v": reach_v, "rho0": rho0, "h0": h0,
                          "wh0": float(wh[0]), "s": rtj["s"]})

    # ------------------------------------------------------------------ pruning wounds
    # Each is where a low limb was taken off decades ago: the callus has rolled in from the sides over the cut
    # and all but closed it - a spindle-shaped seam with unequal lips around a sliver of brown heartwood, barely
    # proud of the bole. Each is welded into the bole's grid; where on the bole they sit is left to chance.
    scars = []                                    # dict(pos, c, asp)
    rs_ = rng.fork("scars")
    for k in range(int(p.scars)):
        for _try in range(30):
            z = rs_.u(0.95, max(1.3, zf * 0.84))
            az = rs_.u(0, 2 * pi)
            pt, _d, rl = trunk.pts(trunk.s_of_z(np.array([z])), np.array([az]))
            pt, rl = pt[0], float(rl[0])
            cr = min(clamp(r1 * 0.27, 0.055, 0.095) * [1.1, 0.75, 1.25, 0.9][k % 4] * rs_.u(0.9, 1.1), 0.3 * rl)   # radius of the cut limb
            asp, clos = rs_.u(1.35, 2.1), rs_.u(0.4, 0.92)
            if not all(np.linalg.norm(pt - s["pos"]) > 0.7 + cr + s["c"] for s in scars):
                continue

            def f3(X, z=z, az=az, cr=cr, asp=asp, rl=rl):
                a_ = trunk.inv(X)[1]
                return np.hypot(((a_ - az + pi) % (2 * pi) - pi) * rl, (X[:, 2] - z) / asp) / (1.75 * cr)

            res = cut_hole(f3)
            if res is None:
                continue
            lk, lm = res
            Xl = trunk.pts(S2[lk, lm], A2[lk, lm])[0]
            du_l = ((A2[lk, lm] - az + pi) % (2 * pi) - pi) * rl
            dz_l = Xl[:, 2] - z
            phi = np.arctan2(dz_l / asp, du_l)
            nl = len(lk)
            wob = 1.0 + 0.1 * np.sin(2 * phi + rs_.u(0, 6)) + 0.07 * np.sin(3 * phi + rs_.u(0, 6))
            th = np.clip(1.0 + 0.45 * np.cos(phi - rs_.u(0, 6.3)) + 0.2 * np.cos(2 * phi + rs_.u(0, 6.3)), 0.4, 1.7)   # one lip is fatter
            shf = rs_.u(-0.3, 0.3) * cr           # the seam is off-centre
            rows, uvr = [gb + lk * MS + lm], [np.stack([du_l, z + dz_l], axis=1)]
            # (radius in cut radii, height in cut radii, how far the callus has closed in, painted tone)
            for (rho_k, h_k, sq, g_k) in ((1.42, 0.03, 0.0, 0.2), (1.16, 0.10, 0.15, 0.16), (0.96, 0.13, 0.45, 0.12),
                                          (0.8, 0.07, 0.8, 0.2), (0.68, -0.035, 1.0, 0.26)):
                du = rho_k * cr * np.cos(phi) * wob * (1.0 - clos * sq) + shf * sq * clos
                dzz = rho_k * cr * asp * np.sin(phi) * wob
                X = trunk.pts(trunk.s_of_z(z + dzz), az + du / rl, h_k * cr * (th if h_k > 0 else 1.0))[0]
                rows.append(under.add(X, va=[0.0, g_k, 0.0]) + np.arange(nl))
                uvr.append(np.stack([du, z + dzz], axis=1))
            jj = np.arange(nl)
            j1 = (jj + 1) % nl
            for q in range(len(rows) - 1 if Lv < 2 else 0):
                under.faces(quads=np.stack([rows[q][jj], rows[q][j1], rows[q + 1][j1], rows[q + 1][jj]], axis=1),
                            quv=np.stack([uvr[q][jj], uvr[q][j1], uvr[q + 1][j1], uvr[q + 1][jj]], axis=1))
            # the heart: old brown wood, deep in the seam
            Xc = trunk.pts(trunk.s_of_z(np.array([z])), np.array([az + shf * clos / rl]), -0.075 * cr)[0]
            uvh = np.vstack([np.stack([du, dzz], axis=1), [[shf * clos, 0.0]]]) * 3.0
            tri = np.stack([jj, j1, np.full(nl, nl)], axis=1)
            if Lv < 2:
                rough.add(np.vstack([X, Xc]), tris=tri, tuv=uvh[tri.reshape(-1)], va=[0.0, 0.32, 0.0])
            fg_ = f3(grid_pts().reshape(-1, 3)).reshape(KS, MS)
            keep_off[:] = np.maximum(keep_off, 1.0 - _sstep(1.25, 1.8, fg_))     # the plates have long gone around a wound
            pfg[:] = np.minimum(pfg, _sstep(1.05, 1.5, fg_))
            scars.append({"pos": pt, "c": cr, "asp": asp})
            break

    # ------------------------------------------------------------------ the shell of old bark
    n_pl = noise_mask(0.26, seed=sd + 3, lo=0.0, hi=1.0)
    n_fine = noise_mask(0.11, seed=sd + 29, lo=0.0, hi=1.0)
    n_big = noise_mask(0.7, seed=sd + 41, lo=0.0, hi=1.0)
    n_fl = noise_mask(0.085, seed=sd + 61, lo=0.0, hi=1.0)
    tall = np.array([1.0, 1.0, 0.45])             # plates are taller than wide
    z_mid = lerp(zf * 1.12, 0.9, mot ** 0.6)      # the old bark has broken up half way by this height:
    z_a, z_b = z_mid - 1.3, z_mid + 1.5           # solid at the foot, tongues and islands over a metre or more
    pos0 = grid_pts().reshape(-1, 3)
    rank = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1.0)          # noise -> even 0..1, whatever its spread
    zsh = pos0[:, 2] - 1.0 * (rank(n_big(pos0, None)) - 0.5)             # long tongues up one side, bays in another
    cover = 1.0 - _sstep(z_a, z_b, zsh)                                  # = the share of the girth still under old bark
    Ff = (0.06 + 0.88 * cover) - rank(0.45 * n_pl(pos0 * tall, None) + 0.3 * n_fine(pos0 * tall, None) + 0.25 * n_fl(pos0 * tall, None))
    Ff = np.where(zsh < max(z_a + 0.1, 0.55), 1.0, np.where(cover < 0.03, -1.0, Ff))
    Ff = (Ff - 3.0 * keep_off.ravel()).reshape(KS, MS)
    Ff[-3:, :] = -1.0
    cell = (Ff[:-1] + Ff[1:] + np.roll(Ff, -1, axis=1)[:-1] + np.roll(Ff, -1, axis=1)[1:]) > 0.0        # (KS-1, MS)

    def _prune(mask, nmin):
        """Clear the connected patches of `mask` (wrapping around the trunk) smaller than nmin cells."""
        seen = np.zeros_like(mask)
        for k0_, m0_ in zip(*np.nonzero(mask)):
            if seen[k0_, m0_]:
                continue
            comp, stack = [], [(k0_, m0_)]
            seen[k0_, m0_] = True
            while stack:
                a_, b_ = stack.pop()
                comp.append((a_, b_))
                for a2, b2 in ((a_ + 1, b_), (a_ - 1, b_), (a_, (b_ + 1) % MS), (a_, (b_ - 1) % MS)):
                    if 0 <= a2 < mask.shape[0] and mask[a2, b2] and not seen[a2, b2]:
                        seen[a2, b2] = True
                        stack.append((a2, b2))
            if len(comp) < nmin:
                for a_, b_ in comp:
                    mask[a_, b_] = False
        return mask

    cell = ~_prune(~cell | hole, 5) & ~hole       # no pin-holes in the old bark ...
    cell = _prune(cell, 3)                        # ... and no crumbs of it left on the smooth: flakes, a hand long
    cz = np.zeros((KS + 1, MS), dtype=bool)
    cz[1:KS] = cell                               # padded: cz[k+1, m] = cell (k, m)
    adj = [cz[:-1], cz[1:], np.roll(cz, 1, axis=1)[:-1], np.roll(cz, 1, axis=1)[1:]]       # the 4 cells around vertex (k, m)
    n_in = sum(a.astype(int) for a in adj)
    n_cor = np.full((KS, MS), 4)
    n_cor[0] = 2
    edge_v = (n_in > 0) & (n_in < n_cor)
    # pull the outline vertices onto the contour of the field: ragged but smooth plate edges, not grid steps
    gk = np.gradient(Ff, axis=0)
    gm = (np.roll(Ff, -1, axis=1) - np.roll(Ff, 1, axis=1)) / 2
    den = gk * gk + gm * gm + 1e-4
    mv = edge_v & ~lock
    dk = np.clip(-Ff * gk / den, -0.46, 0.46) * mv
    dm = np.clip(-Ff * gm / den, -0.46, 0.46) * mv
    dk[0] = 0.0
    S2 = np.clip(S2 + dk, 0.0, KS - 1.0)
    A2 = A2 + dm * da
    P_in = trunk.pts(S2.ravel(), A2.ravel(), ridge.ravel())[0]          # the underbark surface: the plates stand on it
    # a plate is thick in the middle; higher up its rim has dried, curled and lifted clear of the bole
    ev = edge_v.ravel()
    thick = np.where(ev, 0.006 + 0.017 * n_fl(P_in * 1.3, None) * _sstep(0.45, 1.3, P_in[:, 2]), 0.012 + 0.009 * n_fine(P_in * 1.7, None))
    # (the rim stops a little short of the outline of the patch of bole that the plate has kept dark and damp)
    gl = np.sqrt(gk * gk + gm * gm) + 1e-6
    rim_in = 0.24 * edge_v
    rim_in[0] = 0.0
    P_out = trunk.pts(np.clip(S2 + rim_in * gk / gl, 0.0, KS - 1.0).ravel(), (A2 + rim_in * gm / gl * da).ravel(), ridge.ravel() + thick)[0]
    # ... and it is undercut: the edge of the bark drops back to the bole half a cell in under the plate, so a
    # dark gap shows under every lifted rim
    uc = 0.7 * edge_v * _sstep(0.3, 0.9, P_in[:, 2].reshape(KS, MS))
    P_uc = trunk.pts(np.clip(S2 + uc * gk / gl, 0.0, KS - 1.0).ravel(), (A2 + uc * gm / gl * da).ravel(), ridge.ravel() + 0.002)[0]
    row_g = np.array([2 * pi * np.mean(trunk.pts(np.full(12, float(k)), np.arange(12) * 2 * pi / 12)[2]) for k in range(KS)])
    v_len = np.interp(S2, np.arange(len(trunk.cum)), trunk.cum)
    ufr = A2 / (2 * pi)                           # fraction of the girth, unwrapped per corner below
    vid = lambda k_, m_: k_ * MS + m_

    def uv_of(k_, m_, wrap):
        return np.stack([(ufr[k_, m_] + wrap - 0.5) * row_g[k_], v_len[k_, m_]], axis=-1)

    def cells(mask):
        k_, m_ = np.nonzero(mask)
        m1_ = (m_ + 1) % MS
        w_ = (m_ == MS - 1).astype(np.float64)
        return (k_, m_, np.stack([vid(k_, m_), vid(k_, m1_), vid(k_ + 1, m1_), vid(k_ + 1, m_)], axis=1),
                np.stack([uv_of(k_, m_, 0), uv_of(k_, m1_, w_), uv_of(k_ + 1, m1_, w_), uv_of(k_ + 1, m_, 0)], axis=1))

    kk, mm, quads, quv = cells(cell)
    e2 = edge_v[:-1] | edge_v[1:] | np.roll(edge_v, -1, axis=1)[:-1] | np.roll(edge_v, -1, axis=1)[1:]
    _ke, _me, q_e, uv_e = cells((~cell & ~hole) | (cell & e2))          # bare bole, and the bole under every plate rim
    if Lv < 2:
        under.faces(quads=q_e, quv=uv_e.reshape(-1, 2), base=gb)
    # skirts: the thickness of the bark, from the rim back to the bole
    sk_q, sk_uv = [], []
    NVs = KS * MS
    for (dk_, dm_, e0, e1) in ((-1, 0, 0, 1), (0, 1, 1, 2), (1, 0, 2, 3), (0, -1, 3, 0)):
        k2, m2 = kk + dk_, (mm + dm_) % MS
        ok = (k2 >= 0) & (k2 < KS - 1)
        nb_in = np.zeros(len(kk), dtype=bool)
        nb_in[ok] = cell[k2[ok], m2[ok]]
        open_ = ~nb_in & ~((dk_ == -1) & (kk == 0))
        va, vb = quads[open_, e0], quads[open_, e1]
        # (the rim is doubled: the top of a plate keeps its own outward normals and stays clear of the dark gap)
        sk_q.append(np.stack([va + 2 * NVs, va + NVs, vb + NVs, vb + 2 * NVs], axis=1))
        ua, ub = quv[open_, e0], quv[open_, e1]
        sk_uv.append(np.stack([ua, ua + [0.0, 0.03], ub + [0.0, 0.03], ub], axis=1))
    sk_q, sk_uv = np.vstack(sk_q), np.vstack(sk_uv)
    allv = np.vstack([P_out, P_uc, P_out])
    used, inv = np.unique(np.concatenate([quads.ravel(), sk_q.ravel()]), return_inverse=True)
    if Lv < 2:
        rough.add(allv[used], inv.reshape(-1, 4), np.vstack([quv.reshape(-1, 2), sk_uv.reshape(-1, 2)]))
    else:
        # far level: bole and leader are one plain tube off the same surface (no welds, wounds or plates), ringed
        # where its foot, its shoulders and its bends need it; as far up as the old bark holds it wears old bark
        ptsb = [tuple(P_tr[i]) + (2.0 * R_tr[i],) for i in range(len(P_tr))]
        atb = {q_: i for i, q_ in enumerate(ptsb)}
        zk = [0.0, 0.14, 0.34, 0.62, z_mid] + [sc[q_] for sc in scaf for q_ in ("zb", "zc")]
        kb = sorted({atb[q_] for q_ in lod_path(ptsb, k=1.5)} | {n_bole - 1}
                    | {int(np.argmin(np.abs(P_tr[:n_bole, 2] - z_))) for z_ in zk})
        sd_b = [16 if P_tr[i, 2] < 0.7 else 10 if i < n_bole else 6 if R_tr[i] > 0.1 else 5 if R_tr[i] > 0.05 else 4 for i in kb]
        ks = max(1, min(range(len(kb)), key=lambda j: abs(P_tr[kb[j], 2] - z_mid) + (1e3 if kb[j] >= n_bole - 1 else 0.0)))

        def bole(buf, kq, sq, cap):
            kq = np.array(kq)
            buf.tube(P_tr[kq], R_tr[kq], sq, cap=cap, exact=True, frames=(trunk.T[kq], trunk.N[kq], trunk.B[kq]),
                     rfn=lambda i, A: trunk_r(np.full(len(A), P_tr[kq[i], 2]), R_tr[kq[i]], A))

        bole(rough, kb[:ks + 1], sd_b[:ks + 1], False)
        bole(under, kb[ks:], sd_b[ks:], True)
    under.v[g_blk] = P_in
    # (split plates only where the bole is nearly flat: split vertices on the welts and saddles of a fork would
    # each get their own normal and shade as facets)
    pfg = pfg * (1.0 - _sstep(z_g - 0.6, z_g - 0.1, P_in[:, 2].reshape(KS, MS)))
    pfg = np.where(lock | (ridge > 0.003), -pfg, pfg)
    under.a[g_blk] = np.stack([pfg.ravel(), np.full(KS * MS, -1.0), lock.ravel().astype(np.float64)], axis=1)

    # ------------------------------------------------------------------ wood meshes, bark materials and masks
    # (the old bark at the foot is the browner tint of the same rough, mottled material - coarse relief from its
    # normal map and from the real step at every plate rim, no inked furrows, no stripes)
    bark_old = mat("bark", bump=1.7)
    # Plane underbark: a thin, hard, finely fissured grey-olive skin. It is shed in plates: each face of the
    # bole belongs to one plate, and each plate has its own age - cream where it has just come away (few),
    # grey-olive, olive-green, or old olive-brown skin about to go. The oldest plates are a second, browner
    # tint of the same material (the grime mask could only take them towards black, and black plates read
    # as holes under a lantern); the engine's weathering masks carry the rest: wear = pale cream, grime = a
    # light dirtying only, special = moss at the foot.
    bark_new = mat("bark", bump=1.3, scale=0.8)
    bark_twig = mat("bark", bump=1.0, scale=0.5, double=1)
    tree, sk_ = _skin(under, "underbark", np.array([0.14, 0.14, 0.23]), float(sd), (1.6, 0.75))
    smooth(tree, 180)
    assign(tree, bark_new)
    foot = rough.mesh("old_bark")
    smooth(foot, 50)
    assign(foot, bark_old)
    n_moss = noise_mask(0.33, seed=sd + 17, lo=0.3, hi=0.7)
    n_tone = noise_mask(0.2, seed=sd + 7, lo=0.0, hi=1.0)
    cr_p = np.array([c[0] for c in crotches]) if crotches else np.zeros((0, 3))
    cr_r = np.array([c[1] for c in crotches]) if crotches else np.zeros(0)

    def crotch_k(co):
        k = np.zeros(len(co))
        for q, r in zip(cr_p, cr_r):
            if r < 0.04:
                continue
            d = co - q[None, :]
            # dark in the crotch and in the runnel that drains down from it
            dd = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + (d[:, 2] * np.where(d[:, 2] < 0, 0.55, 1.0)) ** 2)
            k = np.maximum(k, np.exp(-(dd / (1.6 * r + 0.05)) ** 2))
        return k

    pid, pfv, fgv = sk_["pid"], sk_["pf"], sk_["fg"]
    hs = lambda q: np.modf(np.abs(np.sin((pid % 100003) * 12.9898 + q * 78.233 + sd) * 43758.5453))[0]
    h1, h2 = hs(1.0), hs(2.0)
    me = tree.data
    z_v = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", z_v)
    p_c = (0.12 + 0.16 * mot) * _sstep(0.5, 1.6, z_v[2::3])
    p_d = 0.3 - 0.08 * mot
    cream = h1 < p_c
    old = ~cream & (h1 < p_c + p_d) & sk_["hard"] & (pfv > 0.7) & (pid >= 0)
    olive = ~cream & ~old & (h1 < p_c + p_d + 0.24)
    tone_c = {}

    def tones(co, n):
        key = len(co)
        if key in tone_c:
            return tone_c[key]
        z = co[:, 2]
        under_ = np.clip(-n[:, 2], 0, 1) ** 1.5 * _sstep(zf * 0.8, zf * 1.1, z)
        # the limbs: one skin, whole stretches of it a little older and darker than others
        G = 0.13 + 0.12 * (n_big(co, None) - 0.5) + 0.04 * (n_tone(co * tall, None) - 0.5)
        # ... and, as on any plane, soft pale stretches where a sheet of it has come away, far up the limbs
        Wr = 0.62 * (0.35 + 0.65 * mot) * _sstep(0.6, 0.85, rank(n_big(co + np.array([3.1, 1.7, 0.4]), None))) * _sstep(zf * 0.7, zf, z)
        if key == len(pid):
            pl_g = np.where(cream, 0.03, np.where(old, 0.22 + 0.04 * h2, np.where(olive, 0.16 + 0.05 * h2, 0.07 + 0.08 * h2)))
            k = pfv * (pid >= 0)
            G = G * (1 - k) + pl_g * k
            # (cream by half: the engine breaks a half-worn patch up along the relief of the skin, so a fresh
            # plate has a ragged, flaked outline and never the outline of its polygons)
            Wr = np.where(cream, 0.38 + 0.18 * h2, 0.0) * k + Wr * (1 - k)
        # a graded stain in every crotch and down the runnel below it - a stain, never a black cuff
        G = np.clip(G + np.clip(0.21 - G, 0, 1) * crotch_k(co) + 0.03 * under_, 0, 0.26)
        Sp = n_moss(co, n) * mos * (1.0 - _sstep(0.15, 0.9 + 1.4 * mos, z)) * (0.4 + 0.6 * np.clip(n[:, 1], 0, 1))
        if key == len(pid):                       # callus: its own tones, ring by ring
            fx = fgv >= 0
            G, Wr, Sp = np.where(fx, fgv, G), np.where(fx, 0.0, Wr), np.where(fx, 0.0, Sp)
        tone_c[key] = (G, Wr, Sp)
        return tone_c[key]

    paint(tree, grime=lambda co, n: tones(co, n)[0], wear=lambda co, n: tones(co, n)[1], special=lambda co, n: tones(co, n)[2])
    fgr = np.vstack(rough.a)[:, 1]

    def m_special(co, n):    # moss creeping up the damp side of the foot (+Y = north); a dust of algae in every furrow
        return np.maximum(n_moss(co, n) * mos * (1.0 - _sstep(0.15, 0.9 + 1.4 * mos, co[:, 2])) * (0.4 + 0.6 * np.clip(n[:, 1], 0, 1)), 0.16)

    def m_grime(co, n):      # rain-splashed foot of the trunk; the heart of a wound
        gq = np.clip(1.0 - co[:, 2] / 0.45, 0, 1) * 0.3 + 0.12 * n_moss(co * 1.7, n)
        return np.where(fgr >= 0, fgr, gq) if len(co) == len(fgr) else gq

    paint(foot, special=m_special, grime=m_grime, wear=lambda co, n: 0.12 + 0.2 * n_tone(co, n))

    # ------------------------------------------------------------------ tree pit
    rg = rng.fork("pit")
    extra_tris = 0

    def soil_h(rho, az):
        h = 0.014 + 0.075 * np.exp(-np.maximum(rho - r1 * 1.3, 0.0) / 0.3)
        for rr_ in root_runs:                     # the ground is heaved up along every root
            tu = (rho - rr_["rho0"]) / (rr_["reach"] - rr_["rho0"])
            tt = np.clip(tu, 0, 1)
            a_c = rr_["a0"] + (rr_["a1"] - rr_["a0"]) * tt ** 1.5
            d = np.abs((az - a_c + pi) % (2 * pi) - pi) * rho
            # gravel lies against its flanks, and closes over its back where it dives: the ridge runs on as a
            # low mound of heaved gravel past the last of the bare wood
            mound = 0.4 * np.maximum(root_top(tt, rr_["h0"]), 0.0) + 0.035 * _sstep(0.6, 0.85, tu) * (1.0 - _sstep(0.95, 1.35, tu))
            h = h + mound * np.exp(-(d / (1.6 * rr_["wh0"] * (1 - 0.3 * tt) + 0.03)) ** 2)
        return h

    if p.surround != "none":
        nr, ns = ((12, 56), (7, 28), (4, 16))[Lv]
        rr = np.linspace(r1 * 1.05, rp_ - (0.05 if p.surround == "setts" else 0.0), nr)
        if Lv and p.surround == "setts":
            rr[-1] /= cos(pi / ns)                # levels: the coarser rim lies around the full model's, never inside it
        aa = np.arange(ns) * 2 * pi / ns
        soft = 0.0 if p.surround == "setts" else 1.0
        edge = rp_ * (1.0 + soft * (0.09 * np.sin(3 * aa + ph[3]) + 0.06 * np.sin(5 * aa + ph[4]) + 0.03 * np.sin(11 * aa + ph[5])))
        rho = rr[:, None] * (edge / rp_)[None, :]
        zz = soil_h(rho, aa[None, :] + 0 * rho) + 0.013 * np.sin(rho * 9 + aa[None, :] * 3 + ph[0]) * np.sin(aa[None, :] * 5 + ph[1])
        zz[-1, :] = 0.004 if p.surround == "soil" else 0.02
        Vs = np.stack([rho * np.cos(aa)[None, :], rho * np.sin(aa)[None, :], zz], axis=-1)
        i = np.arange(nr - 1)[:, None]
        k = np.arange(ns)[None, :]
        k1 = (k + 1) % ns
        qs = np.stack([i * ns + k, (i + 1) * ns + k, (i + 1) * ns + k1, i * ns + k1], axis=-1).reshape(-1, 4)
        sb = _Buf()
        sb.add(Vs.reshape(-1, 3), qs, Vs.reshape(-1, 3)[qs.reshape(-1)][:, :2])
        soil = sb.mesh("soil")
        smooth(soil, 180)
        assign(soil, mat("gravel"))
        if p.surround == "soil":
            # bare trodden earth: a pale dry crust with darker damp hollows, darkest against the bole - never a
            # flat black disc under the lanterns
            n_damp = noise_mask(0.3, seed=sd + 53, lo=0.35, hi=0.75)
            paint(soil, grime=lambda co, n: 0.2 + 0.4 * n_damp(co, n) + 0.25 * (1 - _sstep(r1 * 1.2, r1 * 2.2, np.hypot(co[:, 0], co[:, 1]))))
        extra_tris += sb.tris
    if p.surround == "setts":
        sett = box("sett", size=(0.2, 0.125, 0.09), bevel=lod_pick(0.014, 0.0), segments=1)
        lod_faces(sett, drop=(0, 0, -1))          # levels: plain blocks without their beds (the ends stay: the joints are open)
        n_st = int(round(2 * pi * (rp_ + 0.06) / 0.215))
        ms = []
        for k in range(n_st):
            a = (k + rg.u(-0.12, 0.12)) * 2 * pi / n_st
            lift = 0.0
            for rr_ in root_runs:                 # roots heave the ring where they reach it
                d = abs((a - rr_["a1"] + pi) % (2 * pi) - pi) * rp_
                lift = max(lift, 0.12 * rr_["s"] * math.exp(-(d / 0.26) ** 2) * clamp((rr_["reach_v"] / rp_ - 0.85) / 0.2, 0.15, 1.0))
            r = rp_ + 0.065 + rg.u(-0.008, 0.012) + lift * 0.3
            ms.append(trs((r * cos(a), r * sin(a), 0.002 + rg.u(-0.003, 0.006) + lift),
                          (rg.u(-2.5, 2.5) - lift * 230, rg.u(-2.5, 2.5) + lift * rg.u(-90, 90), degrees(a) + 90 + rg.u(-3.5, 3.5)),
                          (rg.u(0.9, 1.05), rg.u(0.85, 1.1), 1.0)))
        setts = instances_to_mesh("setts", sett, ms, keep_base=False, seed=sd)
        assign(setts, mat("stone_rough", "#a09a8e"))
        paint(setts, grime=lambda co, n: 0.3 * n_moss(co, n), special=lambda co, n: 0.5 * mos * n_moss(co * 1.3, n))
        extra_tris += len(setts.data.polygons) * 2

    # ------------------------------------------------------------------ foliage
    ta_k, tb_k = clamp((season - 0.3) / 0.7 * 1.15, 0.0, 1.0), clamp(season * 1.2, 0.0, 1.0)
    # Deep and light foliage, each of two colours that the engine mixes card by card (mask `special`) and
    # darkens with the depth of the card inside the crown (mask `wear`): summer greens from deep to
    # yellow-green; in autumn dull gold, russet, crisp brown and the last olive green - never one flat colour.
    t_a = _mix_hex(_mix_hex(p.leaf_tint, "#35502a", 0.45), "#6a5229", tb_k * 0.9)
    t_a2 = _mix_hex(p.leaf_tint, "#8f7d3a", ta_k)
    t_b = _mix_hex(_mix_hex(p.leaf_tint, "#7d8f3a", 0.3), "#96843c", ta_k * 0.92)
    t_b2 = _mix_hex(_mix_hex(p.leaf_tint, "#8a9a44", 0.55), "#7a5a2e", tb_k * 0.8)
    fmat_lit = mat("hedge", _mix_hex(t_b, "#6a4a2a", 0.5), double=1, bump=0.8, scale=LEAF_TILE)      # fallen leaves
    dens = p.leaf_density * (1.0 - 0.3 * season)
    dens_k = clamp(dens, 0.3, 1.25)
    live = [ci for ci, c in enumerate(clumps) if "br" in c]
    sq_ = lambda zq: np.array([1.0, 1.0, zq])
    for ci in live:
        c = clumps[ci]
        c["o"] = UP if pollard else _n(c["E"] - crown_core)
        # (a pollard tuft is an upright brush that starts below its head: the head is never a bare club)
        c["sh"] = ((c["E"] + UP * 0.44 * c["Lw"], 0.56 * c["Lw"], 1.2) if pollard
                   else (c["E"] - c["o"] * 0.1 * c["rc"], 0.84 * c["rc"], g.uniform(0.62, 0.86)))
        c["lump"] = (_n(g.normal(size=(3, 3))), g.uniform(0, 6.3, 3))
        c["bias"] = clamp(0.5 + 0.17 * float(g.normal()), 0.2, 0.8)
    cen_c = np.zeros((len(clumps), 3))
    a_c, z_c, b_c, o_c = np.ones(len(clumps)), np.ones(len(clumps)), np.zeros(len(clumps)), np.zeros((len(clumps), 3))
    for ci in live:
        cen_c[ci], a_c[ci], z_c[ci] = clumps[ci]["sh"]
        b_c[ci], o_c[ci] = clumps[ci]["bias"], clumps[ci]["o"]

    def hull_r(ci, d):
        """Radius of the lumpy hull of mass ci in the directions d (before the vertical squash)."""
        lv, lq = clumps[ci]["lump"]
        r = a_c[ci] * (1.0 + 0.24 * np.sin(2.6 * (d @ lv[0]) + lq[0]) + 0.15 * np.sin(4.7 * (d @ lv[1]) + lq[1])
                       + 0.1 * np.sin(8.3 * (d @ lv[2]) + lq[2]))
        return r if pollard else r * np.where(d[:, 2] < 0, 1.0 - 0.38 * d[:, 2] ** 2, 1.0)      # a dome: flat underneath

    def tone_of(X, ci):
        """Light / deep tone of the foliage at X in mass ci: a smooth field, lighter on the upper, outer side."""
        rel = (X - cen_c[ci]) / a_c[ci][:, None]
        return (b_c[ci] + 0.2 * rel[:, 2] + 0.12 * np.sum(rel * o_c[ci], axis=1) + 0.22 * (X[:, 2] - zc) / max(c_up, 1e-3)
                + g.normal(0, 0.06 + 0.16 * season, len(X)))

    n_lit = int(round(p.litter * (22 + 170 * season) * (W / 6.4) ** 2 * 0.8))
    ls0 = float(p.leaf_size)
    A_c = {ci: 4 * pi * a_c[ci] ** 2 * (0.45 + 0.55 * z_c[ci]) * 0.8 for ci in live}
    A_h = sum(A_c[ci] * clumps[ci]["w"] for ci in live)

    # ---- the triangle budget, and what the reduced levels keep of it
    # The town's reduced levels collapse the cheapest edges of the whole tree first, and the smooth closed wood is
    # cheaper than any loose card: a tree that has more cards than a level may keep loses its trunk and limbs to
    # the last stick before one card goes. So the tree is budgeted so that no card ever has to go: the cards
    # (and what else cannot be collapsed) are no more than the middle level may keep, less a proper trunk; the
    # far level drops every island below a size that the small cards and the twiglets stay under, and keeps
    # the big inner boughs and a trunk. Where that takes more wood than there is, the wood is split finer
    # (_refine, at the end).
    lit_tris = n_lit * LEAF_T[2]
    st_res = 2 * len(stems) + 150                 # what is left of the shoots when they are collapsed
    # (the levels are budgeted from the tallies of the full model: the same cards from the same random draws)
    w_under, w_rough, w_extra, w_twigs = lod_memo("wood_tris", lambda: (under.tris, rough.tris, extra_tris, twigs.tris))
    cheap0 = w_under + w_rough + w_extra + w_twigs
    e_card = LOD_K[0] * TRI_CAP - WOOD_LOD[0] - lit_tris - st_res
    cheap = min(TRI_CAP - lit_tris - e_card, cheap0 + 3 * w_under)
    e_card = min(e_card, (LOD_K[0] * (cheap + lit_tris) - WOOD_LOD[0] - lit_tris - st_res) / (1 - LOD_K[0]))
    n_big = int(clamp((LOD_K[1] * (cheap - 0.7 * w_extra) - st_res - WOOD_LOD[1]) / (2 * (1 - LOD_K[1])), 150, 0.1 * e_card)
                * clamp(dens_k, 0.5, 1.0))
    n_small = int((e_card - 2 * n_big) / (2 + TWIG_SHARE) * clamp(dens_k, 0.45, 1.0))
    Vw = np.vstack(under.v + rough.v + twigs.v)
    Vh = np.vstack([cen_c[ci][None, :] + s_ * a_c[ci] * sq_(z_c[ci])[None, :] * np.eye(3) for ci in live for s_ in (-1.0, 1.0)])
    isl_of = lambda V: 0.14 * clamp(float(np.linalg.norm(V.max(0) - V.min(0))) / 6.0, 0.6, 2.5)
    isl = lod_memo("isl_hulls", lambda: isl_of(np.vstack([Vw, Vh])))             # islands below this size go at the far level (tk.final)
    cs0 = math.floor(min(CARD_K * ls0, 0.55 * isl) * 50) / 50
    cb0 = BIG_K * cs0
    fmats = [mat("hedge", t, double=1, bump=0.8, scale=LEAF_TILE, leafy=cs0, tint2=t2) for t, t2 in ((t_a, t_a2), (t_b, t_b2))]
    fmat_big = mat("hedge", t_a, double=1, bump=0.8, scale=LEAF_TILE, leafy=cb0, tint2=t_b)

    # ---- 1. leaf-cluster cards on twiglets
    # Every card is one cell of the cut-out leaf mosaic: a ragged spray of a few leaves with sky between them
    # (see CARD_K). They lie over two nested hulls of every mass - the outer ones held out on twiglets - and in
    # pairs along twiglets that leave every shoot, left and right, like the leaves of a branchlet.
    QX, QN, QH, QS, QD, QC, TW_a, TW_b = [], [], [], [], [], [], [], []
    est = sum(cv * (0.8 if pollard else 1.0) * clumps[ci]["w"] * A_c[ci] * fl * fl / cs0 ** 2 * 0.62 for ci in live for (fl, cv, shd) in CARD_LAYERS)
    est += sum(2 * L * clumps[ci]["w"] / ((0.3 if pollard else 0.5) * CARD_K * ls0) for (P, ci, L, whip) in stems)
    mul = clamp(1.12 * n_small / max(est, 1.0), 0.2, 10.0)

    def push(X, N, Hd, S, Dp, ci):
        for arr, v in zip((QX, QN, QH, QS, QD, QC), (X, N, Hd, S, Dp, np.broadcast_to(ci, (len(X),)))):
            arr.append(np.array(v))

    for ci in live:
        c = clumps[ci]
        for (fl, cv, shd) in CARD_LAYERS:
            m = int(round(mul * cv * (0.8 if pollard else 1.0) * c["w"] * A_c[ci] * fl * fl / cs0 ** 2))
            d = _n(g.normal(size=(m, 3)))
            # (most on the sides that are seen: facing out of the crown, and the ceiling over the lanterns)
            d = d[g.random(m) < ((0.45 + 0.55 * np.clip(0.55 + 0.5 * (d @ c["o"]) - 0.3 * d[:, 2], 0, 1)) if fl == 1.0 else 0.8)]
            r = hull_r(ci, d) * fl * g.uniform(0.86, 1.08, len(d))
            X = cen_c[ci][None, :] + d * r[:, None] * sq_(z_c[ci])[None, :]
            keep = np.ones(len(d), dtype=bool)
            for cj in live:                       # none buried in the next mass
                if cj != ci:
                    keep &= np.linalg.norm((X - cen_c[cj][None, :]) / sq_(z_c[cj])[None, :], axis=1) > 0.82 * fl * a_c[cj]
            k = int(keep.sum())
            X, hn = X[keep], _n(d[keep] / sq_(z_c[ci])[None, :])
            # leaves are held flat to the light: a card lies somewhere between its hull and the horizontal
            u = g.uniform(0.15, 1.0, k)[:, None]
            N = _n(hn * u + UP[None, :] * (1 - u) * np.where(g.random(k) < 0.8, 1.0, -1.0)[:, None] + g.normal(0, 0.28, (k, 3)))
            S = cs0 * g.uniform(0.85, 1.15, k)
            push(X, N, _n(g.normal(size=(k, 3))) - UP[None, :] * 0.45 + hn * 0.2, S, np.clip(shd + g.uniform(0, 0.2, k), 0, 1), ci)
            if fl == 1.0:                         # a twiglet comes up out of the mass to carry it
                tw = g.random(k) < 0.5
                kt = int(tw.sum())
                TW_a.append(X[tw] - hn[tw] * (S[tw] * g.uniform(1.0, 1.9, kt))[:, None] + g.normal(0, 0.05, (kt, 3)))
                TW_b.append(X[tw])

    def along(P, m, whip):
        """m points along the twig P, crowding towards its tip: (t, point, tangent)."""
        cum = _cum(P)
        seg = np.diff(cum)
        t = np.sort(1.03 - (0.78 if whip else 0.66) * g.random(m) ** 1.2)
        s = np.clip(t, 0, 1) * cum[-1]
        idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(P) - 2)
        f = (s - cum[idx]) / np.maximum(seg[idx], 1e-6)
        return t, P[idx] + (P[idx + 1] - P[idx]) * f[:, None], _n(P[idx + 1] - P[idx])

    for (P, ci, L, whip) in stems:
        m = int(round(mul * L * clumps[ci]["w"] / ((0.3 if pollard else 0.5) * CARD_K * ls0)))
        if m <= 0:
            continue
        t, base, Tt = along(P, m, whip)
        side = _n(np.cross(Tt, UP[None, :]) + 1e-6)
        vert = np.cross(side, Tt)
        al = g.normal(0, 0.6, m) + pi * ((np.arange(m) + g.integers(0, 2)) % 2)      # left, right, left up the shoot
        ax = _n(Tt * (0.35 + 0.5 * np.clip(t, 0, 1))[:, None] + (side * np.cos(al)[:, None] + vert * np.sin(al)[:, None]) * 0.8
                - UP[None, :] * 0.2 + g.normal(0, 0.12, (m, 3)))
        ln = cs0 * g.uniform(1.1, 2.3, m)
        tip = base + ax * ln[:, None] - UP[None, :] * (0.1 * ln)[:, None]
        dep_ = np.linalg.norm((base - cen_c[ci][None, :]) / sq_(z_c[ci])[None, :], axis=1) / a_c[ci]
        dp = 0.85 * (1.0 - _sstep(0.45, 0.95, dep_))
        for f in (0.5, 1.0):
            push(base + (tip - base) * f + g.normal(0, 0.02, (m, 3)), _n(UP[None, :] - ax * ax[:, 2:3] + g.normal(0, 0.4, (m, 3))),
                 ax + g.normal(0, 0.2, (m, 3)), cs0 * g.uniform(0.85, 1.15, m), dp, ci)
        TW_a.append(base)
        TW_b.append(tip)
    QX, QN, QH, QS, QD, QC = np.vstack(QX), np.vstack(QN), np.vstack(QH), np.concatenate(QS), np.concatenate(QD), np.concatenate(QC)
    if len(QX) > n_small:
        sel = g.choice(len(QX), n_small, replace=False)
        QX, QN, QH, QS, QD, QC = QX[sel], QN[sel], QH[sel], QS[sel], QD[sel], QC[sel]
    # ---- 2. boughs: big sprays inside every mass, which give the crown its body and its darker inside, and are
    # what is left of it at the far level
    wsum = sum(clumps[ci]["w"] * A_c[ci] for ci in live)
    BG = []
    for ci in live:
        m = int(round(n_big * clumps[ci]["w"] * A_c[ci] / wsum))
        if m <= 0:
            continue
        d = _n(g.normal(size=(m, 3)))
        fl = 0.25 + 0.6 * g.random(m) ** 0.7
        X = cen_c[ci][None, :] + d * (hull_r(ci, d) * fl)[:, None] * sq_(z_c[ci])[None, :]
        hn = _n(d / sq_(z_c[ci])[None, :])
        N = _n(UP[None, :] * np.where(g.random(m) < 0.85, 1.0, -1.0)[:, None] + hn * g.uniform(0.0, 0.9, m)[:, None] + g.normal(0, 0.3, (m, 3)))
        S = np.minimum(cb0 * g.uniform(0.8, 1.25, m), 1.2 * a_c[ci])
        BG.append((X, N, _n(g.normal(size=(m, 3))) + hn * 0.3, S, np.clip(1.25 - 1.1 * fl + g.uniform(0, 0.15, m), 0.3, 1.0), np.full(m, ci)))

    def cards(X, N, Hd, S, cell, square=False):
        """Quads (n, 4, 3) and their UVs (n, 4, 2): one cell of the mosaic each, turned any of four ways.
        square: sides along the world axes as nearly as the card's plane allows (the smallest bounding box)."""
        n = len(X)
        if square:
            Hd = np.where((g.random(n) < 0.5)[:, None], np.array([1.0, 0.0, 0.0])[None, :], np.array([0.0, 1.0, 0.0])[None, :]) \
                * np.where(g.random(n) < 0.5, 1.0, -1.0)[:, None]
            flat_ = np.abs(np.sum(Hd * N, axis=1)) > 0.9
            Hd[flat_] = Hd[flat_][:, [1, 0, 2]]
        e1 = Hd - N * np.sum(Hd * N, axis=1, keepdims=True)
        bad = np.linalg.norm(e1, axis=1) < 0.1
        e1[bad] = np.cross(N[bad], UP[None, :]) + 1e-3
        e1 = _n(e1)
        e2 = np.cross(N, e1)
        cx, cy = np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0])
        hx, hy = 0.5 * S[:, None] * cx[None, :], 0.5 * (S * g.uniform(0.85, 1.0, n))[:, None] * cy[None, :]
        # (never flat: the far end droops and every corner is a little out of true, so no card is seen exactly edge-on)
        bz = S[:, None] * (np.where(cx > 0, -1.0, 0.0)[None, :] * g.uniform(0.06, 0.22, (n, 1)) + g.normal(0, 0.05, (n, 4)))
        V = X[:, None, :] + e1[:, None, :] * hx[:, :, None] + e2[:, None, :] * hy[:, :, None] + N[:, None, :] * bz[:, :, None]
        cuv = np.array([[0.004, 0.004], [0.996, 0.004], [0.996, 0.996], [0.004, 0.996]])
        rot = g.integers(0, 4, n)
        return V, (g.integers(0, 24, (n, 2))[:, None, :] + cuv[(np.arange(4)[None, :] + rot[:, None]) % 4]) * cell

    def leafage(name, m_, V, UV, dep, hue):
        k = len(V)
        if not k:
            return
        pb_ = _Buf()
        pb_.add(V.reshape(-1, 3), quads=np.arange(4 * k).reshape(-1, 4), quv=UV.reshape(-1, 2))
        o = pb_.mesh(name)
        smooth(o, 180)
        assign(o, m_)
        dv_, hv_ = np.repeat(dep, 4), np.repeat(hue, 4)
        paint(o, grime=0.04, wear=lambda co, n: dv_ if len(co) == len(dv_) else 0.3, special=lambda co, n: hv_ if len(co) == len(hv_) else 0.5)

    n_q = len(QX)
    QV, QUV = cards(QX, QN, QH, QS, cs0, square=True)
    BV = np.zeros((0, 4, 3))
    if BG:
        BX, BN, BH, BS, BD, BC = [np.concatenate([b[i] for b in BG]) for i in range(6)]
        BV, BUV = cards(BX, BN, BH, BS, cb0)
    # the true size of the tree is known now: no card or twiglet may be an island that outlasts the far level
    isl = 0.95 * lod_memo("isl_cards", lambda: isl_of(np.vstack([Vw, QV.reshape(-1, 3), BV.reshape(-1, 3)])))
    q_dg = np.linalg.norm(QV.max(axis=1) - QV.min(axis=1), axis=1)
    QV = QX[:, None, :] + (QV - QX[:, None, :]) * np.minimum(1.0, isl / q_dg)[:, None, None]
    q_tone = tone_of(QX, QC)
    # which of its two colours a card wears: whole boughs lean one way, and every card differs from the next;
    # in autumn a share of the clusters is crisp brown
    q_hue = np.clip(0.5 + 1.1 * (q_tone - 0.5) + g.normal(0, 0.2, n_q), 0, 1)
    QD = np.clip(QD + season * 0.35 * (g.random(n_q) < 0.25), 0, 1)
    # Levels: every card is drawn at every level and only a share of them is meshed (CARD_LOD), in an even
    # shuffle that is the same at every level. A card that stands for several is that much larger and is drawn in
    # towards its mass, so the crown keeps its outline. (Never several cells of the mosaic on one card: their
    # clusters stand in rows, and a crown of them reads as perforated plates.)
    q_u = (np.arange(n_q) * 0.6180339887) % 1.0
    q_in = np.empty(n_q)
    q_in[np.argsort(QD, kind="stable")] = (np.arange(n_q) + 0.5) / max(n_q, 1)       # 0 the outermost card .. 1
    bands = CARD_LOD[Lv]
    if Lv == 1 and lod_budget():
        # the cards that stay as they are go as deep into the crown as the level's budget lets them, after its wood
        rest = under.tris + rough.tris + extra_tris + twigs.tris + 2 * len(BV) + 0.5 * TWIG_SHARE * n_small + lit_tris
        f_ = ((0.94 * lod_budget() - rest) / max(2 * n_q, 1) - bands[1][1]) / (1 - bands[1][1])
        bands = [(clamp(f_, 0.0, bands[0][0]),) + bands[0][1:], bands[1]]

    def grown(V, UV, X, ci, k, big):
        if k == 1:
            return V, UV
        Xk = X - _n(X - cen_c[ci]) * (0.35 * (k - 1) * cs0)
        return Xk[:, None, :] + (V - X[:, None, :]) * k, UV * k if big else UV

    for i in range(2):
        sel = (q_tone >= 0.5) == bool(i)
        if Lv == 0:
            leafage(f"canopy{i}", fmats[i], QV[sel], QUV[sel], QD[sel], q_hue[sel])
            continue
        lo, part = 0.0, {}
        for upto, share, k, big in bands:
            s_ = sel & (q_in >= lo) & (q_in < upto) & (q_u < share)
            lo = upto
            part.setdefault(big, []).append(grown(QV[s_], QUV[s_], QX[s_], QC[s_], k, big) + (QD[s_], q_hue[s_]))
        for big, pp in part.items():
            m_ = (mat("hedge", (t_a, t_b)[i], double=1, bump=0.8, scale=LEAF_TILE, leafy=cb0, tint2=(t_a2, t_b2)[i])
                  if big else fmats[i])
            leafage(f"canopy{i}", m_, *[np.concatenate([q_[j] for q_ in pp]) for j in range(4)])
    n_b = len(BV)
    if n_b:
        b_hue = np.clip(tone_of(BX, BC) + g.normal(0, 0.15, n_b), 0, 1)
        if Lv:
            bs = (np.arange(n_b) * 0.6180339887) % 1.0 < BOUGH_LOD[Lv][0]
            BV, BUV, BD, b_hue = (BX[:, None, :] + (BV - BX[:, None, :]) * BOUGH_LOD[Lv][1])[bs], BUV[bs], BD[bs], b_hue[bs]
        leafage("boughs", fmat_big, BV, BUV, BD, b_hue)
    # twiglets: a sliver each (no more can be afforded: see the budget), the low ones first - they are seen from
    # close by, against the lanterns
    Ta, Tb = np.vstack(TW_a), np.vstack(TW_b)
    n_twl = int(min(len(Ta), TWIG_SHARE * n_small))
    sel = np.argsort(Ta[:, 2] + g.normal(0, 0.8, len(Ta)))[:n_twl]
    Ta, Tb = Ta[sel], Tb[sel]
    if n_twl:
        Tl = np.linalg.norm(Tb - Ta, axis=1)
        Tn = (Tb - Ta) / np.maximum(Tl, 1e-6)[:, None]
        Ta = Tb - Tn * np.minimum(Tl, 0.9 * isl)[:, None]
        e1 = _n(np.cross(Tn, _n(g.normal(size=(n_twl, 3)))) + 1e-6)
        V = np.stack([Ta - e1 * 0.006, Ta + e1 * 0.006, Tb], axis=1)
        if Lv:                                    # levels: every other one (the low ones come first); none far off
            V = V[::2] if Lv == 1 else V[:0]
        if len(V):
            tuv = np.tile(np.array([[0.0, 0.0], [0.012, 0.0], [0.006, 0.3]]), (len(V), 1))
            twigs.add(V.reshape(-1, 3), tris=np.arange(3 * len(V)).reshape(-1, 3), tuv=tuv)
    if twigs.tris:
        tw = twigs.mesh("twigs")
        smooth(tw, 180)
        assign(tw, bark_twig)
        paint(tw, grime=0.24)                     # young shoots are olive-brown, not yet flaking

    e_all = 2 * (n_q + n_b) + n_twl + lit_tris + st_res
    # fallen leaves: flat on the paving and the soil, edges curled up as they dry
    if n_lit > 0:
        rl = np.sqrt(g.random(n_lit)) * (W * 0.46 - r1 * 1.6) + r1 * 1.6
        al = g.uniform(0, 2 * pi, n_lit)
        cx, cy = C[0] * 0.5, C[1] * 0.5
        gp = np.stack([cx + rl * np.cos(al), cy + rl * np.sin(al), np.zeros(n_lit)], axis=1)
        rho_g = np.hypot(gp[:, 0], gp[:, 1])
        on_soil = (rho_g < rp_ - 0.08) & (p.surround != "none")
        on_sett = (np.abs(rho_g - rp_ - 0.06) < 0.14) & (p.surround == "setts")
        keep = ~(on_soil & (rho_g < r1 * 2.7))    # none inside the buttress
        gp[:, 2] = np.where(on_soil, soil_h(rho_g, np.arctan2(gp[:, 1], gp[:, 0])) + 0.024, np.where(on_sett, 0.12, 0.006)) + g.uniform(0, 0.004, n_lit)
        ya = g.uniform(0, 2 * pi, n_lit)
        lit_x = np.stack([np.cos(ya), np.sin(ya), np.zeros(n_lit)], axis=1)
        lit_n = np.stack([g.normal(0, 0.05, n_lit), g.normal(0, 0.05, n_lit), np.ones(n_lit)], axis=1)
        lit_s = ls0 * 1.1 * g.uniform(0.8, 1.25, n_lit)
        lit_t = np.clip(0.62 + 0.3 * season + g.normal(0, 0.25, n_lit), 0, 0.999)
    if n_lit > 0:
        lb = _Buf()
        k = int(keep.sum())
        fold_, curl_ = -g.uniform(0.02, 0.22, k), -g.uniform(0.0, 0.2, k)
        st_ = (1, 1, 3)[Lv]                       # levels: plainer blades; far off a third of them, larger
        kl = np.nonzero(keep)[0][::st_]
        _blades(lb, (2, 1, 0)[Lv], gp[kl], lit_x[kl], lit_n[kl], lit_s[kl] * (1.0, 1.0, 1.7)[Lv], fold_[::st_], curl_[::st_])
        if lb.nv:
            o = lb.mesh("litter")
            smooth(o, 180)
            assign(o, fmat_lit)
            paint(o, grime=0.45 + 0.25 * season)  # (fallen leaves are dull and brown)
    # the wood, split finer: as much of it as lets the middle level keep every card and a proper trunk
    now = cheap0 + (twigs.tris - (cheap0 - under.tris - rough.tris - extra_tris)) + 2 * (n_q + n_b) + lit_tris
    more = int(min((e_all + WOOD_LOD[0]) / LOD_K[0] - now, TRI_CAP - now, 3 * under.tris))
    if Lv == 0:                                   # (padding for the collapsed copy only: a modelled level has no use for it)
        _refine(tree, more)

    socket("grate", "attach", (0, 0, 0), role="tree_grate")
    for k, hp in enumerate(hang_pts[:3]):
        socket(f"hang{k}", "attach", tuple(float(x) for x in hp), role="lantern")
