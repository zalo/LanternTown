"""tk.reduce - cheaper levels (lod 1 / 2) made FROM the full model, for assets that do not model their own.

Where the decimated copy (tk.final._make_lod) collapses everything by one ratio, the reducer works on the
un-merged scene and takes the triangles from where nobody sees them first:

  1. faces no viewer can see (buried in other parts, undersides near the ground, backs against the wall) go
  2. coplanar faces merge (limited dissolve)
  3. thin long pieces (bars, rods, rails, gently bent tubes) become 4-sided (lod 1) / 3-sided (lod 2) prisms
  4. loose pieces smaller than the level's size limit are dropped - never the ones that carry the outline of the
     asset, the last piece of an animated part, or anything next to something that glows
  5. scattered plant cards are thinned (half as hard at lod 1) and the survivors enlarged
  6. fields of small pieces (stones, tiles, cobbles) are welded into one skin, as far as the outline stands it
     and the field stays covered (_open: a weld that lays the darker core behind an ashlar skin bare is undone)
  7. what is still over the budget comes out of the remaining dense meshes by collapse - or, where a field would
     not weld (roof tiles laid over each other) and the collapse would tear it, the field becomes flat patches
     in the planes of its pieces (one rectangle per row of them) and the collapse only gets what is left

The collapse works loose piece by loose piece: a piece only takes a step that leaves it whole (_collapse_part), so
a part may end over its ratio - what cannot be had without tearing is left (see level_candidates: "reduced15",
"reduced20").

Everything that glows is kept as it is (only where it is most of the model it is collapsed along with the rest;
glowing panes tiled in one plane become one pane per block of the size limit, _panes). No step may lay another
surface onto a glowing face: nothing is welded or snapped onto one, a piece takes no collapse step that brings it
within Z_NEAR of one, and _zfix() removes what a glowing card hides and pushes off what still lies on it.
The room card of a window (materials of tk.lodimage.Z_CARD: window_lit whatever its glow, window_dark) is such a
card lit or not: the dark card of an unlit room is kept as it is and kept clear of the wall the same way.

Lettering (tk.geo text(): face attribute "tsq_text") is an identity class: never welded, patched or collapsed. It
loses backs and bedded faces, at lod 2 its sides, its flat fronts are dissolved; a group under tk.lodimage.TEXT_PX
at the level's distance becomes its flat front or goes - whole, never in part.

After every step the bounds and the parts are checked and the step is taken back where it broke them; hidden-face
removal and welding are also checked on a quick image (tk.lodimage.Probe).

    snap = snapshot()                           # right before tk.final.finalize() of lod 0
    ...finalize, reference...
    restore(snap); log = reduce_scene(1, ref, meta, params); tk.final_levels.finalize_level(1, ref, ...)
    cands, log = level_candidates(1, snap, ref, meta, params, stem)     # the three lines above in one, for pick()

    obj = reduce_child(obj, level, meta)        # tk.asm.use(): a composed asset without levels of its own

Nothing here draws from the toolkit's random streams or counters.
"""
import bpy
import bmesh
import json
import math
import os
import time
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .core import world_matrix, apply_modifiers, _link, _STATE
from .surf import realize
from . import lodimage as _im
from . import holes as _ho
from .levels import BUDGET, TOL, _dp_keep, _ntris, _tag_ident

GLOW = _im.GLOW
PLANT = _im.PLANT
NO_OCCLUDE = GLOW + PLANT + ("glass_clear", "water_fall", "water")     # never hide anything behind these

ND = 64                         # view directions of the visibility test
NEED = (0, 2, 5)                # a face stays when it shows from at least this many of them
CAP = 5
EPS = 0.0015
EYE_MIN = 0.6                   # ground-mounted assets: nobody looks up from below this height (m)
ANGLE = (0.0, 3.0, 7.0)         # limited dissolve (degrees)
ANGLE_W = (0.0, 8.0, 14.0)      # ... of a welded field
ANGLE_P = (0.0, 20.0, 45.0)     # ... of plant cards
ANGLE_T = 15.0                  # ... of the outline of a letter that is down to its flat front
ATOM = 28                       # a loose piece of up to this many triangles is not worth collapsing
FIELD_N, FIELD_T, FIELD_REL = 12, 200, 0.3
RMIN = (1.0, 0.30, 0.10)        # collapse ratio below which loose small pieces are dropped first
RCOMF = (1.0, 0.22, 0.06)       # ... below which fields are welded first
RFLOOR = (1.0, 0.12, 0.03)      # the dense meshes are never collapsed further than this in the normal pass
DROPCAP = (0.0, 2.5, 2.5)       # small pieces dropped under pressure: up to this x the size limit
WELD = (0.35, 0.6, 1.0, 1.6)    # weld distances tried, x the size limit
WELD_LESS = (0.2, 0.1)          # ... and when the picture does not stand the first
WELD_SINGLE = 12                # the fields tried one by one when not all stand a distance together
RTHIN = (0.0, 0.02, 0.06)       # 'thin': half thickness up to this (m) - about 3 px at the level's distance
PRISM_N = (0, 4, 3)
PATCH_ANG = 25.0                # a patch: flat pieces of a field that look the same way within this (degrees) ...
PATCH_FLAT = 0.6                # ... flat: |sum of area x normal| / area of the piece is at least this ...
PATCH_N = 6                     # ... and at least this many of them
SIL = _im.SIL
LUMTOL = (0.0, 0.025, 0.04)     # a step may shift the unlit brightness of the picture by this much
SEEN = float(os.environ.get("TSQ_REDUCE_SEEN", 0.06))   # tk.final_levels.finalize_level(seen=): see _transfer_masks there
TONE_CUT = 0.12                 # materials further apart than this (gamma-space luminance) are cut apart for a collapse

Z_NEAR = _im.Z_KEEP - _im.Z_EDGE    # a card (a glowing face, the room card of a window lit or dark: tk.lodimage.z_glows)
                                #     and a parallel face of another material closer than this fight for the depth
                                #     buffer (tk.lodimage.zfight is the gate) ...
Z_GAP = 0.005                   # ... and this is left between them where the reducer moves one of the two
Z_ROUNDS = 4                    # a stack of cards is pushed apart one pair after the other
Z_SHIFT = (0.0, 0.02, 0.06)     # no corner is pushed further than this at lod 1 / 2 (about a pixel there)

K_GLOW, K_PRISM, K_PLANT, K_FIELD, K_ATOM, K_DENSE, K_TEXT = 1, 2, 3, 4, 5, 6, 7
NOTEXT = bool(os.environ.get("TSQ_REDUCE_NOTEXT"))    # lettering is treated as any other geometry (to test the gate)
TEXT_SHARE = 0.15               # a loose piece with less lettering than this is cut in two: the lettering and the rest
NOTHIN = bool(os.environ.get("TSQ_REDUCE_NOTHIN"))    # fields of pieces under the size limit are not thinned (to compare)
TRACE = bool(os.environ.get("TSQ_REDUCE_TRACE"))      # log the probe's reading after every step
HOLES = os.environ.get("TSQ_HOLES", "1") != "0"       # "no holes" (docs/LOD.md): what the street sees stays, _close() at the end


def _lib(m):
    return str(m.get("tsq_lib", "")) if m is not None else ""


def _tone(m, lib):
    """albedo luminance (linear) of a material"""
    t = m.get("tsq_tint") if m is not None else None
    try:
        base = _im.base_colour(lib, list(t)[:3] if t is not None else None)[0]
    except Exception:
        base = np.array([0.5, 0.5, 0.5])
    if lib.startswith(GLOW):
        base = np.minimum(np.asarray(base) * 2.2, 1.0)
    return float(np.dot(np.asarray(base, dtype=np.float64)[:3], (0.2126, 0.7152, 0.0722)))


def _zglows(m, lib):
    """is the material a card of the depth-fight measure (tk.lodimage.z_glows: it glows, or is the room card of a
    window, lit or dark)"""
    p = m.get("tsq_params") if m is not None else None
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            p = None
    return _im.z_glows(lib, p if isinstance(p, dict) else None)


def _two_sided(m, lib):
    if lib.startswith(_im.TWO_SIDED) or lib.startswith(PLANT):
        return True
    p = m.get("tsq_params") if m is not None else None
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            p = None
    return bool(p and p.get("double"))


# ----------------------------------------------------------------------------
# Mesh arrays
# ----------------------------------------------------------------------------

class _M:
    """numpy view of one mesh object (its transform is already baked into the mesh)"""

    def __init__(self, o):
        me = o.data
        self.o, self.me = o, me
        nv, npo, nl = len(me.vertices), len(me.polygons), len(me.loops)
        co = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        self.co = co.reshape(-1, 3).astype(np.float64)
        self.lt = np.empty(npo, dtype=np.int32)
        me.polygons.foreach_get("loop_total", self.lt)
        self.ls = np.empty(npo, dtype=np.int32)
        me.polygons.foreach_get("loop_start", self.ls)
        mi = np.empty(npo, dtype=np.int32)
        me.polygons.foreach_get("material_index", mi)
        self.lv = np.empty(nl, dtype=np.int32)
        me.loops.foreach_get("vertex_index", self.lv)
        mats = list(me.materials) or [None]
        libs = [_lib(m) for m in mats]
        self.libs = libs
        self.names = [m.name if m is not None else "" for m in mats]
        self.tone = np.array([_tone(m, l) for m, l in zip(mats, libs)])
        self.glow = np.array([l.startswith(GLOW) for l in libs], dtype=bool)
        self.plant = np.array([l.startswith(PLANT) for l in libs], dtype=bool)
        self.two = np.array([_two_sided(m, l) for m, l in zip(mats, libs)], dtype=bool)
        self.noocc = np.array([l.startswith(NO_OCCLUDE) for l in libs], dtype=bool)
        self.zglow = np.array([_zglows(m, l) for m, l in zip(mats, libs)], dtype=bool)
        self.see = np.array([l in _im.Z_SEE for l in libs], dtype=bool)
        self.clear = np.array([l.startswith(_im.SEE) for l in libs], dtype=bool)        # clear glass: seen through
        self.cast = np.array([not (l.startswith(_im.NOSHADOW) or l.startswith(GLOW)) for l in libs], dtype=bool)
        self.card = np.array([l in _im.Z_CARD for l in libs], dtype=bool)      # the room card of a window, lit or dark
        self.keep = self.glow | self.card | self.plant | np.array([l.startswith(("glass", "water")) for l in libs], dtype=bool)
        self.mi = np.clip(mi, 0, len(libs) - 1)
        self.ntris = int((self.lt - 2).sum()) if npo else 0
        self._t = self._a = self._n = self._e = self._rk = self._tx = None

    def tris(self):
        if self._t is None:
            me = self.me
            me.calc_loop_triangles()
            nt = len(me.loop_triangles)
            t = np.empty(nt * 3, dtype=np.int32)
            me.loop_triangles.foreach_get("vertices", t)
            tp = np.empty(nt, dtype=np.int32)
            me.loop_triangles.foreach_get("polygon_index", tp)
            self._t = (t.reshape(-1, 3), tp)
        return self._t

    def area(self):
        if self._a is None:
            a = np.empty(len(self.lt), dtype=np.float32)
            self.me.polygons.foreach_get("area", a)
            self._a = a.astype(np.float64)
        return self._a

    def normal(self):
        if self._n is None:
            n = np.empty(len(self.lt) * 3, dtype=np.float32)
            self.me.polygons.foreach_get("normal", n)
            self._n = n.reshape(-1, 3).astype(np.float64)
        return self._n

    def edges(self):
        if self._e is None:
            e = np.empty(len(self.me.edges) * 2, dtype=np.int32)
            self.me.edges.foreach_get("vertices", e)
            self._e = e.reshape(-1, 2)
        return self._e

    def rk(self):
        if self._rk is None:
            a = self.me.attributes.get("rk")
            v = np.zeros(len(self.lt), dtype=np.int32)
            if a is not None and a.domain == 'FACE' and len(a.data) == len(v):
                a.data.foreach_get("value", v)
            self._rk = v
        return self._rk


    def tx(self):
        """lettering tag of every polygon (tk.geo._tag_text): 4 x text number + 1 front | 2 side | 3 back; 0 = none"""
        if self._tx is None:
            v = np.zeros(len(self.lt), dtype=np.int32)
            a = None if NOTEXT else self.me.attributes.get("tsq_text")
            if a is not None and a.domain == 'FACE' and len(a.data) == len(v):
                a.data.foreach_get("value", v)
            self._tx = v
        return self._tx


_components = _im._components


def _islands(m):
    """Loose pieces of a mesh -> dict of per-island arrays (+ pisl: island of every polygon, visl: of every vertex)"""
    lab = _components(len(m.co), m.edges())
    uq, pisl = np.unique(lab[m.lv[m.ls]], return_inverse=True)
    k = len(uq)
    pos = np.minimum(np.searchsorted(uq, lab), k - 1)
    visl = np.where(uq[pos] == lab, pos, -1)
    t = (m.lt - 2).astype(np.float64)
    vv = np.flatnonzero(visl >= 0)
    vo = vv[np.argsort(visl[vv], kind="stable")]
    vs = np.searchsorted(visl[vo], np.arange(k))
    mn = np.minimum.reduceat(m.co[vo], vs, axis=0)
    mx = np.maximum.reduceat(m.co[vo], vs, axis=0)
    ar = m.area()
    ns = len(m.libs)
    dom = np.bincount(pisl * ns + m.mi, weights=ar, minlength=k * ns).reshape(k, ns).argmax(axis=1)
    return {"n": k, "pisl": pisl, "visl": visl, "vo": vo, "vs": vs, "nv": np.diff(np.r_[vs, len(vo)]),
            "tone": np.bincount(pisl, weights=ar * m.tone[m.mi], minlength=k) / np.maximum(np.bincount(pisl, weights=ar, minlength=k), 1e-12),
            "mat": np.array(m.names, dtype=object)[dom],
            "tris": np.bincount(pisl, weights=t, minlength=k).astype(np.int64),
            "area": np.bincount(pisl, weights=m.area(), minlength=k),
            "glow": np.bincount(pisl, weights=t * m.glow[m.mi], minlength=k),
            "card": np.bincount(pisl, weights=t * m.card[m.mi], minlength=k),
            "plant": np.bincount(pisl, weights=t * m.plant[m.mi], minlength=k),
            "prism": np.bincount(pisl, weights=(m.rk() == 1), minlength=k) > 0,
            "text": np.bincount(pisl, weights=t * (m.tx() > 0), minlength=k),
            "tg": _island_text(m.tx(), pisl, k),
            "mn": mn, "mx": mx, "d": np.linalg.norm(mx - mn, axis=1)}


def _island_text(tx, pisl, k):
    """the text number of every island (0: no lettering in it)"""
    out = np.zeros(k, dtype=np.int64)
    w = tx > 0
    if w.any():
        out[pisl[w]] = tx[w] // 4
    return out


# ----------------------------------------------------------------------------
# Visibility
# ----------------------------------------------------------------------------

def _fib(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


_BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7], [0.45, 0.45, 0.1],
                  [0.1, 0.45, 0.45], [0.45, 0.1, 0.45], [0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]])


def visibility(objs, mount="ground", allow_below=False):
    """From how many of ND directions each polygon can be seen (capped at CAP) -> [int8 array per object].
    A polygon is tested at 1-10 points per triangle (by area), on its front (both sides for two-sided materials);
    a direction counts when a ray from the point towards the viewer hits nothing that hides (glass, glow, plants
    and water hide nothing). Viewers stand on the ground (ground-mounted assets: no view from below EYE_MIN) and in
    front of the wall (wall-mounted: no view from behind y = 0). Glow / glass / water / plant faces report CAP."""
    Ms = [_M(o) for o in objs]
    Vs, Ts, off = [], [], 0
    PP, PN, PG, PT = [], [], [], []
    gp = 0
    out = []
    for m in Ms:
        npo = len(m.lt)
        out.append(np.full(npo, CAP, dtype=np.int8))
        if npo == 0:
            Vs.append(m.co)
            off += len(m.co)
            continue
        T, tp = m.tris()
        a, b, c = m.co[T[:, 0]], m.co[T[:, 1]], m.co[T[:, 2]]
        ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        slot = m.mi[tp]
        Vs.append(m.co)
        Ts.append(T[~m.noocc[slot]] + off)
        k = 1 + np.minimum((ar / 0.02).astype(np.int64), 9)
        k[m.keep[slot] | (ar <= 1e-12)] = 0
        ti = np.repeat(np.arange(len(T)), k)
        j = np.arange(len(ti)) - np.repeat(np.cumsum(k) - k, k)
        W = _BARY[j]
        PP.append(a[ti] * W[:, 0:1] + b[ti] * W[:, 1:2] + c[ti] * W[:, 2:3])
        PN.append(m.normal()[tp[ti]])
        PG.append(tp[ti].astype(np.int64) + gp)
        PT.append(m.two[slot][ti])
        gp += npo
        off += len(m.co)
    if not PP or not sum(len(p) for p in PP):
        return out
    P, N, G, TW = np.vstack(PP), np.vstack(PN), np.concatenate(PG), np.concatenate(PT)
    V = np.vstack(Vs)
    Tocc = np.vstack(Ts) if Ts else np.zeros((0, 3), dtype=np.int64)
    if len(Tocc) == 0:
        return out
    bvh = BVHTree.FromPolygons(V.tolist(), Tocc.tolist(), all_triangles=True, epsilon=0.0)
    rc = bvh.ray_cast
    D = _fib(ND)
    wall = mount in ("wall", "wall_ground")
    ground = mount in ("ground", "wall_ground") and not allow_below
    far = float(np.linalg.norm(V.max(0) - V.min(0))) * 2.0 + 10.0
    ef = np.zeros(gp, dtype=np.int16)
    tested = np.zeros(gp, dtype=bool)
    tested[G] = True
    # polygons in chunks (points are grouped by polygon; G is non-decreasing)
    pid = np.flatnonzero(tested)
    CH = 60000
    for c0 in range(0, len(pid), CH):
        pc = pid[c0:c0 + CH]
        lo, hi = np.searchsorted(G, pc[0]), np.searchsorted(G, pc[-1], side="right")
        Pc, Nc, Gc, Tc = P[lo:hi], N[lo:hi], G[lo:hi], TW[lo:hi]
        row = np.searchsorted(pc, Gc)                       # point -> row of its polygon in the chunk
        first = np.searchsorted(Gc, pc)                     # row -> its first point
        Cn = (Nc[first] @ D.T).astype(np.float32)           # (polygons, ND)
        Cs = np.where(Tc[first][:, None], np.abs(Cn), Cn)
        if wall:
            Cs[:, D[:, 1] > 0.0] = -1.0
        order = np.argsort(-Cs, axis=1).astype(np.int16)
        Csort = np.take_along_axis(Cs, order.astype(np.int64), axis=1)
        alive = np.ones(len(pc), dtype=bool)
        cnt = np.zeros(len(pc), dtype=np.int16)
        for r in range(ND):
            alive &= Csort[:, r] > 0.06
            if not alive.any():
                break
            pt = np.flatnonzero(alive[row])
            rw = row[pt]
            dsel = order[rw, r]
            d = D[dsel]
            sg = np.where(Cn[rw, dsel] >= 0.0, 1.0, -1.0)
            o = Pc[pt] + Nc[pt] * (EPS * sg)[:, None]
            dist = np.full(len(pt), far)
            go = np.ones(len(pt), dtype=bool)
            if ground:
                dn = d[:, 2] < -1e-6
                go &= ~(dn & (o[:, 2] <= EYE_MIN))
                dist = np.where(dn, (o[:, 2] - EYE_MIN) / np.maximum(-d[:, 2], 1e-6), dist)
            gi = np.flatnonzero(go)
            if len(gi):
                ol, dl, sl = o[gi].tolist(), d[gi].tolist(), dist[gi].tolist()
                esc = np.fromiter((rc(a_, b_, c_)[0] is None for a_, b_, c_ in zip(ol, dl, sl)), dtype=bool, count=len(gi))
                hit = np.zeros(len(pc), dtype=bool)
                hit[rw[gi[esc]]] = True
                cnt += hit
                alive &= cnt < CAP
        ef[pc] = cnt
    gp = 0
    for m, v in zip(Ms, out):
        npo = len(m.lt)
        t = tested[gp:gp + npo]
        v[t] = np.minimum(ef[gp:gp + npo][t], CAP)
        gp += npo
    return out


LIT_RAYS = 300000               # rays of the caster test per light at most (the nearest faces first)
LIT_CORNER = 0.9                # _light_casters: the corner rays of a triangle aim this far from its centre to the corner


def _light_casters(S, marks, lights, allow_below=False):
    """Of the faces nobody sees (marks: [bool per polygon] per object) those that keep a light of the asset in.
    The engine lets a face cast a shadow when it looks away from the light (or is two-sided): the back of a lamp
    shade, the underside of a clock head are what stops the lamp inside from lighting the wall behind and the paving
    below - and no viewer ever sees them. A marked face stays when, on the ray from the light to it, no face that
    stays casts before it, and behind it the light would fall on a face that stays and looks at the light, or leave
    the asset: onto the ground (or the wall of a wall-mounted asset) where a viewer can see that, or into the open
    within the light's range.
    Tested on the ray to the centre of each triangle; for a light flagged to cast shadows (socket prop "shadow") on
    four rays per triangle (its centre and a point LIT_CORNER of the way to each corner).
    lights: tk.lodimage.lights_of(). -> [bool per polygon] per object"""
    out = [np.zeros(len(pm), dtype=bool) for pm in marks]
    if not lights or not any(pm.any() for pm in marks):
        return out
    Vs, Ts, TG, TW, TM = [], [], [], [], []
    off = gp = 0
    for o, pm in zip(S.objs, marks):
        m = S.an(o)[0]
        npo = len(m.lt)
        if npo:
            T, tp = m.tris()
            slot = m.mi[tp]
            c = m.cast[slot]
            Vs.append(m.co)
            Ts.append(T[c] + off)
            TG.append(tp[c].astype(np.int64) + gp)
            TW.append(m.two[slot][c])
            TM.append(pm[tp[c]])
            off += len(m.co)
        gp += npo
    if not Vs:
        return out
    V, T = np.vstack(Vs), np.vstack(Ts)
    G, W, Mk = np.concatenate(TG), np.concatenate(TW), np.concatenate(TM)
    if not len(T) or not Mk.any():
        return out
    A = V[T[:, 0]]
    N = np.cross(V[T[:, 1]] - A, V[T[:, 2]] - A)
    ctr = (A + V[T[:, 1]] + V[T[:, 2]]) / 3.0
    rc = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True, epsilon=0.0).ray_cast
    Nl, Wl, Ml, Gl = N.tolist(), W.tolist(), Mk.tolist(), G.tolist()
    keep = bytearray(gp)
    wall = S.mount in ("wall", "wall_ground")
    ground = S.mount in ("ground", "wall_ground") and not allow_below
    fy = -1.0 if (S.b0[0][1] + S.b0[1][1]) < 0 else 1.0     # the side of the wall (y = 0) the asset stands on
    UP = [(0.0, 0.0, 1.0), (0.7071, 0.0, 0.7071), (-0.7071, 0.0, 0.7071), (0.0, 0.7071, 0.7071), (0.0, -0.7071, 0.7071)]
    FR = [(0.0, fy, 0.0), (0.7071, 0.7071 * fy, 0.0), (-0.7071, 0.7071 * fy, 0.0), (0.0, 0.7071 * fy, 0.7071),
          (0.0, 0.7071 * fy, -0.7071)]
    far = 2.0 * S.diag + 10.0

    def free(p, dirs):
        """can a viewer see the point p of the ground / the wall, past the faces that stay"""
        for d in dirs:
            o_, left, ok = p, far, True
            for _ in range(32):
                h = rc(o_, d, left)
                if h[0] is None:
                    break
                if not Ml[h[2]]:
                    ok = False
                    break
                s = h[3] + 1e-4
                left -= s
                o_ = (o_[0] + d[0] * s, o_[1] + d[1] * s, o_[2] + d[2] * s)
            if ok:
                return True
        return False

    for _name, pos, rng_, *_sh in lights:
        fine = bool(_sh and _sh[0])
        p = np.array(pos)
        dv = ctr - p
        dist = np.linalg.norm(dv, axis=1)
        away = W | (np.einsum("ij,ij->i", N, dv) > 0.0)
        idx = np.flatnonzero(Mk & away & (dist < rng_) & (dist > 1e-6))
        if len(idx) > LIT_RAYS:
            idx = idx[np.argsort(dist[idx], kind="stable")[:LIT_RAYS]]
        if not len(idx):
            continue
        # the centre of each triangle first, then (lights that are sure of a shadow map) a point towards each of
        # its corners: a face whose middle lies in the shadow of a part that stays (the shell of a clock head under
        # its saddle) still lets a thin beam out past the rim of that part - a lit outline on the paving. For every
        # light of a big building that costs thousands of unseen triangles and the levels their budget (measured:
        # clock_tower_hall + 3343 at lod 1), so the others keep the centre ray only.
        cs = ctr[idx]
        pts = np.vstack([cs] + ([cs + LIT_CORNER * (V[T[idx, k]] - cs) for k in range(3)] if fine else []))
        dv = pts - p
        dist = np.linalg.norm(dv, axis=1)
        ok_ = dist > 1e-6
        idx = np.tile(idx, 4 if fine else 1)[ok_]
        dist = dist[ok_]
        Dl = (dv[ok_] / dist[:, None]).tolist()
        for t, d, dt in zip(idx.tolist(), Dl, dist.tolist()):
            g = Gl[t]
            if keep[g]:
                continue
            o_, left, gone, past, res = pos, rng_, 0.0, False, None
            for _ in range(64):
                h = rc(o_, d, left)
                if h[0] is None:
                    res = "out"
                    break
                ti = h[2]
                gone += h[3]
                if Gl[ti] == g:
                    past = True
                else:
                    if not past and gone > dt + 1e-3:
                        past = True                     # (the ray slipped past its own face)
                    if not Ml[ti]:
                        n = Nl[ti]
                        back = Wl[ti] or (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) > 0.0
                        if not past:
                            if back:
                                res = "shadow"          # a face that stays casts before it
                                break
                        else:
                            res = "lit" if (Wl[ti] or not back) else "shadow"
                            break
                s = h[3] + 1e-4
                left -= s
                gone += 1e-4
                if left <= 0.0:
                    res = "out"
                    break
                o_ = (o_[0] + d[0] * s, o_[1] + d[1] * s, o_[2] + d[2] * s)
            if res == "lit":
                keep[g] = 1
            elif res == "out" and past:
                ok = True
                if ground and d[2] < -1e-6:
                    tg = -pos[2] / d[2]
                    if 0.0 < tg < rng_:
                        ok = free((pos[0] + d[0] * tg, pos[1] + d[1] * tg, 1e-3), UP)
                    else:
                        ok = False                      # (it reaches no ground within its range)
                elif wall and d[1] * fy < -1e-6:
                    tg = -pos[1] / d[1]
                    if 0.0 < tg < rng_:
                        ok = free((pos[0] + d[0] * tg, 1e-3 * fy, pos[2] + d[2] * tg), FR)
                if ok:
                    keep[g] = 1
    kp = np.frombuffer(bytes(keep), dtype=np.uint8).astype(bool)
    gp = 0
    for i, pm in enumerate(marks):
        out[i] = kp[gp:gp + len(pm)] & pm
        gp += len(pm)
    return out


# ----------------------------------------------------------------------------
# State of a reduction
# ----------------------------------------------------------------------------

def _obj_bounds(objs):
    """bounds of what the polygons of the objects use (loose vertices do not count: nothing of them is exported)"""
    mn, mx = np.full(3, 1e30), np.full(3, -1e30)
    for o in objs:
        me = o.data
        n = len(me.vertices)
        if n and len(me.polygons):
            co = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            lv = np.empty(len(me.loops), dtype=np.int32)
            me.loops.foreach_get("vertex_index", lv)
            used = np.zeros(n, dtype=bool)
            used[lv] = True
            if not used.all():
                co = co[used]
            mn, mx = np.minimum(mn, co.min(0)), np.maximum(mx, co.max(0))
    return mn, mx


class _S:
    def __init__(self, objs, level, budget, mount, bounds, diag, log):
        self.objs = list(objs)
        self.level, self.budget, self.mount, self.log = level, budget, mount, log
        self.b0 = bounds if bounds is not None else _obj_bounds(self.objs)
        self.diag = float(diag if diag is not None else np.linalg.norm(self.b0[1] - self.b0[0]))
        k = float(np.clip(self.diag / 6.0, 0.6, 2.5))
        self.ms = (0.0, 0.035 * k, 0.14 * k)[level]
        self.tol = 0.8 * max(0.02, 0.01 * self.diag)            # under the 'bounds differ' warning of finalize_level
        self.tolc = 0.5 * max(0.08, 0.04 * self.diag)           # collapse: half of what rejects a level
        self._an = {}
        self.zs = None                                          # _zstart(): set when something glows
        self.cover = None                                       # _weld(): points on the flat pieces of the fields
        self.cache = None
        self.lights = None                                      # tk.lodimage.lights_of(): the asset's own lights

    def part(self, o):
        return o.get("tsq_part", "body")

    def tris(self):
        return int(sum(_ntris(o.data) for o in self.objs))

    def dirty(self, o):
        self._an.pop(o.name, None)

    def an(self, o):
        r = self._an.get(o.name)
        if r is None:
            m = _M(o)
            r = (m, _islands(m)) if len(m.lt) else (m, None)
            self._an[o.name] = r
        return r

    def soup(self):
        """-> vertices, triangles, and per triangle: glows, two-sided, albedo luminance, clear glass"""
        Vs, Ts, Gs, Ws, Ls, Cs, off = [], [], [], [], [], [], 0
        for o in self.objs:
            m = self.an(o)[0]
            if not len(m.lt):
                continue
            T, tp = m.tris()
            Vs.append(m.co)
            Ts.append(T + off)
            Gs.append(m.glow[m.mi[tp]])
            Ws.append(m.two[m.mi[tp]])
            Ls.append(m.tone[m.mi[tp]])
            Cs.append(m.clear[m.mi[tp]])
            off += len(m.co)
        if not Vs:
            return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool),
                    np.zeros(0), np.zeros(0, dtype=bool))
        return (np.vstack(Vs), np.vstack(Ts), np.concatenate(Gs), np.concatenate(Ws), np.concatenate(Ls),
                np.concatenate(Cs))

    # ---- taking a step back ----
    def save(self, objs=None):
        return {o.name: (o, o.data.copy()) for o in (self.objs if objs is None else objs)}

    def put_back(self, saved, names=None):
        for nm, (o, me) in list(saved.items()):
            if names is not None and nm not in names:
                continue
            old = o.data
            o.data = me
            if old.users == 0:
                bpy.data.meshes.remove(old)
            self.dirty(o)
            del saved[nm]

    def forget(self, saved):
        for nm, (o, me) in saved.items():
            if me.users == 0:
                bpy.data.meshes.remove(me)
        saved.clear()

    def backup(self):
        """the whole state, for a step that replaces objects (see swap) -> unlinked copies"""
        out = []
        for o in self.objs:
            c = o.copy()
            c.data = o.data.copy()
            out.append(c)
        return out

    def swap(self, objs, linked=False):
        """make `objs` (a backup(), or an object list taken out of here before) the state -> the objects replaced,
        still in the scene"""
        old = self.objs
        self.objs = list(objs)
        if not linked:
            for c in self.objs:
                _link(c)
        self._an = {}
        return old

    @staticmethod
    def discard(objs):
        for o in objs:
            me = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)

    def sides(self, tol):
        """the sides of the lod-0 box the model no longer reaches / sticks out of -> [(axis, +1 max | -1 min)]"""
        mn, mx = _obj_bounds(self.objs)
        bad = []
        for a in range(3):
            if abs(mn[a] - self.b0[0][a]) > tol:
                bad.append((a, -1))
            if abs(mx[a] - self.b0[1][a]) > tol:
                bad.append((a, 1))
        return bad

    def empty_parts(self):
        t = {}
        for o in self.objs:
            t[self.part(o)] = t.get(self.part(o), 0) + len(o.data.polygons)
        return [p for p, n in t.items() if n == 0]

    def guard(self, saved, step, tol=None):
        """Bounds and parts after a step; whatever broke them is put back. -> True when the step stands as made."""
        tol = self.tol if tol is None else tol
        ok = True
        ep = self.empty_parts()
        if ep:
            ok = False
            self.put_back(saved, {nm for nm, (o, _) in saved.items() if self.part(o) in ep})
        bad = self.sides(tol)
        if bad:
            ok = False
            names = set()
            for nm, (o, me) in saved.items():
                n = len(me.vertices)
                if not n:
                    continue
                co = np.empty(n * 3, dtype=np.float32)
                me.vertices.foreach_get("co", co)
                co = co.reshape(-1, 3)
                for a, s in bad:
                    if (s < 0 and co[:, a].min() <= self.b0[0][a] + tol) or (s > 0 and co[:, a].max() >= self.b0[1][a] - tol):
                        names.add(nm)
            self.put_back(saved, names)
            if self.sides(tol):
                self.put_back(saved)
        if not ok:
            self.log.setdefault("rollback", []).append(step)
        self.forget(saved)
        return ok


def _prep(objs):
    out = []
    for o in objs:
        if o is None or o.type != 'MESH':
            continue
        if o.modifiers:
            apply_modifiers(o)
        if len(o.data.polygons) == 0:
            continue
        if o.data.users > 1:
            o.data = o.data.copy()
        mw = world_matrix(o)
        if mw != Matrix.Identity(4):
            o.data.transform(mw)
            if mw.determinant() < 0:
                o.data.flip_normals()
            o.parent = None
            o.matrix_parent_inverse = Matrix.Identity(4)
            o.matrix_basis = Matrix.Identity(4)
        out.append(o)
    return out


def _delete_polys(o, mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    me = o.data
    if len(idx) == len(me.polygons):
        me.clear_geometry()
        return len(idx)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    f = bm.faces
    bmesh.ops.delete(bm, geom=[f[i] for i in idx.tolist()], context='FACES')
    bm.to_mesh(me)
    bm.free()
    me.update()
    return len(idx)


def _join(objs, name):
    """Merge mesh objects (transforms already baked) into one new object; the inputs are removed."""
    bm = bmesh.new()
    mats = []
    for o in objs:
        me = o.data
        remap = []
        for m in (list(me.materials) or [None]):
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        n0 = len(bm.faces)
        bm.from_mesh(me)
        if any(i != r for i, r in enumerate(remap)):
            bm.faces.ensure_lookup_table()
            for f in bm.faces[n0:]:
                f.material_index = remap[min(f.material_index, len(remap) - 1)]
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    j = bpy.data.objects.new(name, me)
    _link(j)
    for k in ("tsq_part", "tsq_pivot", "tsq_part_props"):
        if k in objs[0]:
            j[k] = objs[0][k]
    j["tsq_uv"] = "keep"
    j["tsq_smooth"] = "keep"
    for o in objs:
        old = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if old.users == 0:
            bpy.data.meshes.remove(old)
    return j


# ----------------------------------------------------------------------------
# The table of loose pieces
# ----------------------------------------------------------------------------

def _table(S):
    obj, iid, cols = [], [], {k: [] for k in ("tris", "area", "glow", "card", "plant", "prism", "mn", "mx", "d", "nv", "tone", "mat",
                                              "text", "tg")}
    span = []
    n = 0
    for oi, o in enumerate(S.objs):
        m, isl = S.an(o)
        if isl is None:
            continue
        k = isl["n"]
        obj.append(np.full(k, oi, dtype=np.int64))
        iid.append(np.arange(k, dtype=np.int64))
        for c in cols:
            cols[c].append(isl[c])
        span.append((oi, n, n + k))
        n += k
    if not n:
        return None
    T = {c: (np.vstack(v) if c in ("mn", "mx") else np.concatenate(v)) for c, v in cols.items()}
    T["obj"], T["iid"], T["span"] = np.concatenate(obj), np.concatenate(iid), span
    tris, d = T["tris"], T["d"]
    glow = T["glow"] > 0
    scatter = (T["plant"] > 0.5 * tris) & (tris <= ATOM)
    kind = np.full(n, K_DENSE, dtype=np.int8)
    kind[tris <= ATOM] = K_ATOM
    for oi, a, b in span:
        if S.objs[oi].get("tsq_nofield"):
            continue
        od = float(np.linalg.norm(T["mx"][a:b].max(0) - T["mn"][a:b].min(0)))
        f = ((tris[a:b] <= FIELD_T) & (d[a:b] <= FIELD_REL * od) & ~glow[a:b] & ~(T["card"][a:b] > 0) & ~T["prism"][a:b]
             & ~scatter[a:b])
        if int(f.sum()) >= FIELD_N:
            kind[a:b][f] = K_FIELD
    kind[scatter] = K_PLANT
    kind[T["prism"]] = K_PRISM
    kind[glow | (T["card"] > 0)] = K_GLOW   # ... and the dark card of an unlit room: a window is a window at any distance
    kind[T["text"] > 0] = K_TEXT            # lettering: no step but _text_step / _text_drop touches it
    T["kind"] = kind
    # next to something that glows (a lantern's cage, a window's frame): never dropped or welded
    near = np.zeros(n, dtype=bool)
    gi = np.flatnonzero(glow)
    if len(gi) and S.ms > 0:
        gmn, gmx = T["mn"][gi] - S.ms, T["mx"][gi] + S.ms
        step = max(1, 2000000 // max(len(gi), 1))
        for a in range(0, n, step):
            mn, mx = T["mn"][a:a + step], T["mx"][a:a + step]
            near[a:a + step] = ((mn[:, None, :] <= gmx[None]) & (mx[:, None, :] >= gmn[None])).all(axis=2).any(axis=1)
    T["near"] = near
    T["part"] = np.array([S.part(S.objs[oi]) for oi in T["obj"]], dtype=object)
    return T


def _apply_drop(S, T, drop):
    """delete the islands marked in drop (table rows) -> triangles removed"""
    gone = 0
    for oi, a, b in T["span"]:
        dm = drop[a:b]
        if not dm.any():
            continue
        o = S.objs[oi]
        isl = S.an(o)[1]
        _delete_polys(o, dm[isl["pisl"]])
        gone += int(T["tris"][a:b][dm].sum())
        S.dirty(o)
    return gone


def _protect_outline(S, T, drop, tol):
    """un-mark islands so that the survivors still reach every side of the lod-0 box and no part is emptied"""
    live = ~drop
    for a in range(3):
        for s, col, ref in ((-1, T["mn"][:, a], S.b0[0][a]), (1, T["mx"][:, a], S.b0[1][a])):
            have = col[live].min() if (s < 0 and live.any()) else (col[live].max() if live.any() else None)
            if have is None or abs(have - ref) > tol:
                back = drop & (np.abs(col - ref) <= tol)
                drop &= ~back
                live = ~drop
    for p in set(T["part"].tolist()):
        rows = np.flatnonzero(T["part"] == p)
        if drop[rows].all():
            drop[rows[np.argmax(T["tris"][rows])]] = False
    return drop


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------

def _dissolve(o, level, f=1.0):
    me = o.data
    n0 = _ntris(me)
    if n0 < 4:
        return 0
    mats = list(me.materials) or [None]
    plant = {i for i, m in enumerate(mats) if _lib(m).startswith(PLANT)}
    bm = bmesh.new()
    bm.from_mesh(me)
    lay = None if NOTEXT else bm.faces.layers.int.get("tsq_text")
    if lay is not None and any(f[lay] for f in bm.faces):
        for e in bm.edges:                  # lettering never melts into what it sits on, nor a letter's front into its side
            lf = e.link_faces
            if len(lf) == 2 and lf[0][lay] != lf[1][lay]:
                e.seam = True
    if plant:
        pf = [f for f in bm.faces if f.material_index in plant]
        rest = [f for f in bm.faces if f.material_index not in plant]
        sets = ((rest, f * ANGLE[level], {'MATERIAL', 'SHARP', 'UV', 'SEAM'}), (pf, f * ANGLE_P[level], {'MATERIAL', 'UV'}))
    else:
        sets = ((None, f * ANGLE[level], {'MATERIAL', 'SHARP', 'UV', 'SEAM'}),)
    for faces, ang, delim in sets:
        if faces is None:
            vs, es = bm.verts[:], bm.edges[:]
        else:
            faces = [f for f in faces if f.is_valid]
            if not faces:
                continue
            vs = list({v for f in faces for v in f.verts})
            es = list({e for f in faces for e in f.edges})
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(ang), use_dissolve_boundaries=False, verts=vs, edges=es, delimit=delim)
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big, quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(me)
    bm.free()
    me.update()
    return n0 - _ntris(me)


def _min_rect(x, y):
    """smallest bounding rectangle of 2D points -> (angle, half extents h1 >= h2, centre in the rotated frame)"""
    best = None
    for i in range(12):
        a = math.pi / 2 * i / 12
        c, s = math.cos(a), math.sin(a)
        u, v = x * c + y * s, -x * s + y * c
        hu, hv = 0.5 * (u.max() - u.min()), 0.5 * (v.max() - v.min())
        if best is None or hu * hv < best[0]:
            best = (hu * hv, a, hu, hv, 0.5 * (u.max() + u.min()), 0.5 * (v.max() + v.min()))
    _, a, hu, hv, cu, cv = best
    return a, hu, hv, cu, cv


def _fit_prism(P, area, level):
    """A thin long piece as a centre line + section -> dict(path (k,3), m1 (3,) first section axis, h1, h2,
    perim: its girth) or None when the piece is not one."""
    rlim = RTHIN[level]
    c = P.mean(0)
    X = P - c
    try:
        _w, v = np.linalg.eigh(X.T @ X)
    except Exception:
        return None
    a = v[:, 2]
    t = X @ a
    t0, t1 = float(t.min()), float(t.max())
    L = t1 - t0
    if L < 1e-4:
        return None
    Q = X - np.outer(t, a)
    r = np.linalg.norm(Q, axis=1)
    rmax = float(r.max())
    straight = rmax <= rlim * 1.42 and L >= 6.0 * rmax and np.median(r) >= 0.6 * rmax and np.percentile(r, 25) >= 0.45 * rmax
    if straight:
        b1 = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(a, b1)
        ang, h1, h2, cu, cv = _min_rect(Q @ b1, Q @ b2)
        m1 = b1 * math.cos(ang) + b2 * math.sin(ang)
        m2 = np.cross(a, m1)
        if h2 > h1:
            m1, m2, h1, h2, cu, cv = m2, -m1, h2, h1, cv, -cu
        if h1 > rlim or h1 < 1e-5:
            return None
        cc = c + m1 * cu + m2 * cv
        path = np.array([cc + a * t0, cc + a * t1])
        return {"path": path, "m1": m1, "h1": h1, "h2": max(h2, 0.15 * h1), "perim": area / L, "axis": a, "t0": t0, "c": c}
    # a bent piece: centre line through slabs along the main axis
    req = area / (2.0 * math.pi * L)
    if req > rlim or L < 8.0 * req:
        return None
    K = int(np.clip(L / (3.0 * max(req, 1e-4)), 3, 24))
    k = np.minimum(((t - t0) / L * K).astype(np.int64), K - 1)
    cnt = np.bincount(k, minlength=K)
    if cnt.min() < 3:
        return None
    C = np.stack([np.bincount(k, weights=P[:, i], minlength=K) / cnt for i in range(3)], axis=1)
    tan = np.gradient(C, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    if float((tan[1:] * tan[:-1]).sum(1).min()) < 0.5:
        return None
    Xk = P - C[k]
    Qk = Xk - (Xk * tan[k]).sum(1)[:, None] * tan[k]
    rr = np.linalg.norm(Qk, axis=1)
    rmax = float(rr.max())
    if not (rmax <= rlim and np.median(rr) >= 0.55 * rmax and np.percentile(rr, 25) >= 0.4 * rmax):
        return None
    e0 = C[0] - tan[0] * max(float(((C[0] - P[k == 0]) @ tan[0]).max()), 0.0)
    e1 = C[-1] + tan[-1] * max(float(((P[k == K - 1] - C[-1]) @ tan[-1]).max()), 0.0)
    path = np.vstack([e0, C, e1])
    path = path[_dp_keep(path, TOL[level])]
    plen = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if plen < 6.0 * rmax:
        return None
    h = float(np.median(rr))
    b1 = np.cross(tan[0], [0.0, 0.0, 1.0] if abs(tan[0][2]) < 0.9 else [1.0, 0.0, 0.0])
    return {"path": path, "m1": b1 / np.linalg.norm(b1), "h1": h, "h2": h, "perim": area / plen, "axis": a, "t0": t0, "c": c}


def _prism_mesh(fit, level):
    """-> rings (k, n, 3), s (k,) length along the path, q (n + 1,) girth positions, xy (n, 2) section"""
    h1, h2 = fit["h1"], fit["h2"]
    n = PRISM_N[level]
    if n == 3 and h1 > 2.5 * h2:
        n = 4                   # a flat bar stays a flat bar
    if n == 4:
        s = float(np.clip(fit["perim"] / max(4.0 * (h1 + h2), 1e-9), 0.75, 1.0))
        xy = np.array([[h1, h2], [-h1, h2], [-h1, -h2], [h1, -h2]]) * s
    else:
        s = float(np.clip(fit["perim"] / max(5.196 * 0.5 * (h1 + h2), 1e-9), 1.0, 1.3))
        xy = np.array([[math.cos(a) * h1, math.sin(a) * h2] for a in (math.pi / 2, math.pi * 7 / 6, math.pi * 11 / 6)]) * s
    path = fit["path"]
    seg = np.diff(path, axis=0)
    sl = np.linalg.norm(seg, axis=1)
    dirs = seg / np.maximum(sl, 1e-12)[:, None]
    k = len(path)
    rings = np.zeros((k, n, 3))
    m1 = fit["m1"]
    for i in range(k):
        if i == 0:
            tan, sc = dirs[0], 1.0
        elif i == k - 1:
            tan, sc = dirs[-1], 1.0
        else:
            tan = dirs[i - 1] + dirs[i]
            tan /= max(np.linalg.norm(tan), 1e-12)
            sc = 1.0 / max(float(tan @ dirs[i]), 0.7)
        m1 = m1 - tan * float(m1 @ tan)
        m1 = m1 / max(np.linalg.norm(m1), 1e-12)
        m2 = np.cross(tan, m1)
        rings[i] = path[i] + sc * (xy[:, 0:1] * m1[None] + xy[:, 1:2] * m2[None])
    q = np.r_[0.0, np.cumsum(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1))]
    return rings, np.r_[0.0, np.cumsum(sl)], q, xy


def _prisms(S, T, log):
    """thin long pieces -> prisms; -> triangles saved. A piece stays as it is when its prism would lie on a card
    (within Z_NEAR of it, tk.lodimage.z_pairs) over more of it than the piece does: the bars of a leaded light
    have no back, their prisms would get one a hair off the room card."""
    level = S.level
    n = PRISM_N[level]
    new_min = 2 * n + 2 * (n - 2)
    tris, d = T["tris"], T["d"]
    cand = (np.isin(T["kind"], (K_ATOM, K_DENSE, K_FIELD)) & (tris >= int(math.ceil(new_min / 0.6))) & (tris <= 4000)
            & (T["nv"] <= 3000) & (d >= max(S.ms, 6.0 * 0.002)) & (T["area"] <= 2.0 * math.pi * d * 1.4 * RTHIN[level]))
    saved_t = made = held = 0
    b0, b1 = S.b0[0] - 0.5 * S.tol, S.b0[1] + 0.5 * S.tol
    for oi, a, b in T["span"]:
        rows = np.flatnonzero(cand[a:b])
        if not len(rows):
            continue
        o = S.objs[oi]
        m, isl = S.an(o)
        me = o.data
        po = np.argsort(isl["pisl"], kind="stable")
        ps = np.searchsorted(isl["pisl"][po], np.arange(isl["n"] + 1))
        uv = None
        if me.uv_layers:
            uv = np.empty(len(m.lv) * 2, dtype=np.float32)
            me.uv_layers[0].data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2).astype(np.float64)
        sm = np.empty(len(m.lt), dtype=bool)
        me.polygons.foreach_get("use_smooth", sm)
        area = m.area()
        repl, kill, sv = [], [], []
        for i in rows.tolist():
            vids = isl["vo"][isl["vs"][i]:isl["vs"][i] + isl["nv"][i]]
            P = m.co[vids]
            fit = _fit_prism(P, float(isl["area"][i]), level)
            if fit is None:
                continue
            rings, sarc, q, xy = _prism_mesh(fit, level)
            nn = rings.shape[1]
            nt = (len(rings) - 1) * 2 * nn + 2 * (nn - 2)
            if nt > 0.6 * isl["tris"][i]:
                continue
            if (rings.min(axis=(0, 1)) < b0).any() or (rings.max(axis=(0, 1)) > b1).any():
                continue
            polys = po[ps[i]:ps[i + 1]]
            pa = area[polys]
            mat_i = int(np.bincount(m.mi[polys], weights=pa).argmax())
            smooth = bool(pa[sm[polys]].sum() > 0.5 * pa.sum())
            g, uv0 = np.array([1.0, 0.0]), np.zeros(2)
            if uv is not None:
                li = np.concatenate([np.arange(s_, s_ + n_) for s_, n_ in zip(m.ls[polys].tolist(), m.lt[polys].tolist())])
                tl = (m.co[m.lv[li]] - fit["c"]) @ fit["axis"] - fit["t0"]
                tm = tl.mean()
                var = float(((tl - tm) ** 2).sum())
                um = uv[li].mean(0)
                if var > 1e-12:
                    gg = ((uv[li] - um) * (tl - tm)[:, None]).sum(0) / var
                    if 0.3 <= float(np.linalg.norm(gg)) <= 3.5:
                        g = gg
                uv0 = um - g * tm
            repl.append((rings, sarc, q, xy, mat_i, smooth, g, uv0))
            kill.append(polys)
            sv.append(int(isl["tris"][i]) - nt)
        if repl and getattr(S, "zs", None) is not None:
            zc = _zcards(S, o, 0.05, own=True)
            if zc is not None:
                k_ = len(repl)
                names = np.array(m.names, dtype=object)
                T0, tp0 = m.tris()
                pid = np.full(len(m.lt), -1, dtype=np.int64)
                for q_, polys in enumerate(kill):
                    pid[polys] = q_
                sel = pid[tp0] >= 0
                z_old = _zpiece(zc, m.co, T0[sel], names[m.mi[tp0[sel]]], pid[tp0[sel]], k_, tris=True)
                V1, T1, r1, n1, off = [], [], [], [], 0
                for q_, rp in enumerate(repl):
                    rings = rp[0]
                    k, nn = rings.shape[0], rings.shape[1]
                    ix = off + np.arange(k * nn).reshape(k, nn)
                    V1.append(rings.reshape(-1, 3))
                    tq = []
                    for j_ in range(nn):
                        j2 = (j_ + 1) % nn
                        tq.append(np.stack([ix[:-1, j_], ix[:-1, j2], ix[1:, j2]], axis=1))
                        tq.append(np.stack([ix[:-1, j_], ix[1:, j2], ix[1:, j_]], axis=1))
                    for ring in (ix[0], ix[-1]):
                        tq.append(np.array([(ring[0], ring[c_], ring[c_ + 1]) for c_ in range(1, nn - 1)], dtype=np.int64))
                    tq = np.concatenate(tq)
                    T1.append(tq)
                    r1.append(np.full(len(tq), q_, dtype=np.int64))
                    n1.append(np.full(len(tq), names[rp[4]], dtype=object))
                    off += k * nn
                z_new = _zpiece(zc, np.concatenate(V1), np.concatenate(T1), np.concatenate(n1), np.concatenate(r1), k_, tris=True)
                okz = z_new <= z_old * 1.02 + 2e-5
                if not okz.all():
                    held += int((~okz).sum())
                    repl = [x for x, y in zip(repl, okz.tolist()) if y]
                    kill = [x for x, y in zip(kill, okz.tolist()) if y]
                    sv = [x for x, y in zip(sv, okz.tolist()) if y]
        saved_t += sum(sv)
        made += len(repl)
        if not repl:
            continue
        bm = bmesh.new()
        bm.from_mesh(me)
        rk = bm.faces.layers.int.get("rk") or bm.faces.layers.int.new("rk")
        uvl = bm.loops.layers.uv.active or bm.loops.layers.uv.new("UVMap")
        bm.faces.ensure_lookup_table()
        f = bm.faces
        bmesh.ops.delete(bm, geom=[f[j] for j in np.concatenate(kill).tolist()], context='FACES')
        for rings, sarc, q, xy, mat_i, smooth, g, uv0 in repl:
            gp = np.array([-g[1], g[0]])
            k, nn = rings.shape[0], rings.shape[1]
            vs = [[bm.verts.new(p) for p in ring.tolist()] for ring in rings]

            def face(vl, uvs, sm_):
                try:
                    fc = bm.faces.new(vl)
                except ValueError:
                    return None
                fc.material_index = mat_i
                fc.smooth = sm_
                fc[rk] = 1
                for lp, w in zip(fc.loops, uvs):
                    lp[uvl].uv = w
                return fc

            for i in range(k - 1):
                for j in range(nn):
                    j2 = (j + 1) % nn
                    face([vs[i][j], vs[i][j2], vs[i + 1][j2], vs[i + 1][j]],
                         [uv0 + g * sarc[i] + gp * q[j], uv0 + g * sarc[i] + gp * q[j + 1],
                          uv0 + g * sarc[i + 1] + gp * q[j + 1], uv0 + g * sarc[i + 1] + gp * q[j]], smooth)
            for ring, sa, rev in ((vs[0], sarc[0], True), (vs[-1], sarc[-1], False)):
                idx = list(range(nn))[::-1] if rev else list(range(nn))
                fc = face([ring[j] for j in idx], [uv0 + g * sa + g * xy[j][0] + gp * xy[j][1] for j in idx], False)
                if fc is not None:
                    for e in fc.edges:
                        e.smooth = False
        bm.to_mesh(me)
        bm.free()
        me.update()
        S.dirty(o)
    log["prisms"] = made
    log["prism_tris_saved"] = saved_t
    if held:
        log["prisms_held"] = held           # pieces left as they are: their prism would lie on a card
    return saved_t


def _thin_plants(S, T, q, log, rows=None, key="plant"):
    """keep the share q of the plant cards (evenly spread), enlarge the survivors. rows: these rows of the table
    instead (a field of pieces under the size limit, see step 9 of reduce_objects), logged as `key`_..."""
    rows = np.flatnonzero(T["kind"] == K_PLANT) if rows is None else np.asarray(rows, dtype=np.int64)
    n = len(rows)
    if n < 4 or q >= 0.999:
        return 0
    keep = (((np.arange(n) + 1) * 0.6180339887498949) % 1.0) < q
    if keep.all():
        return 0
    sc = min(math.sqrt(1.0 / max(q, 1e-3)), 2.6)
    drop = np.zeros(len(T["kind"]), dtype=bool)
    drop[rows[~keep]] = True
    drop = _protect_outline(S, T, drop, S.tol)      # the cards that reach the sides of the model stay
    if not drop.any():
        return 0
    grow = np.zeros(len(T["kind"]), dtype=bool)
    grow[rows] = ~drop[rows]
    lo, hi = S.b0
    for oi, a, b in T["span"]:
        gm = grow[a:b]
        if not gm.any():
            continue
        o = S.objs[oi]
        m, isl = S.an(o)
        mn, mx = isl["mn"], isl["mx"]
        cen = 0.5 * (mn + mx)
        half = 0.5 * (mx - mn) * sc
        # stay inside the box of the full model
        c2 = np.minimum(np.maximum(cen, lo + half), hi - half)
        c2 = np.where(lo + half > hi - half, 0.5 * (lo + hi), c2)
        vi = isl["visl"]
        ok = (vi >= 0) & gm[np.maximum(vi, 0)]
        co = m.co.copy()
        co[ok] = c2[vi[ok]] + (co[ok] - cen[vi[ok]]) * sc
        o.data.vertices.foreach_set("co", co.astype(np.float32).ravel())
        o.data.update()
        S.dirty(o)
    # the table still matches the polygon order (only coordinates moved): analyses were dropped, so redo the drop on ids
    gone = 0
    for oi, a, b in T["span"]:
        dm = drop[a:b]
        if not dm.any():
            continue
        o = S.objs[oi]
        isl = S.an(o)[1]
        _delete_polys(o, dm[isl["pisl"]])
        gone += int(T["tris"][a:b][dm].sum())
        S.dirty(o)
    log[key + "_keep"] = round(float(q), 3)
    log[key + "_scale"] = round(sc, 2)
    log[key + "_tris_dropped"] = log.get(key + "_tris_dropped", 0) + gone
    return gone


def _reanchor(o, old):
    """UVs of the welded field faces (rk == 2) from the nearest face of the mesh before the weld: the same texture
    density and offset as the piece that was there, instead of whatever the merge left."""
    P0, UV0, N0, bvh = old
    me = o.data
    if not me.uv_layers or not len(P0):
        return
    m = _M(o)
    sel = np.flatnonzero(m.rk() == 2)
    if not len(sel):
        return
    cen = np.empty(len(m.lt) * 3, dtype=np.float32)
    me.polygons.foreach_get("center", cen)
    cen = cen.reshape(-1, 3)[sel]
    nrm = m.normal()[sel]
    tri = np.array([bvh.find_nearest(c)[2] or 0 for c in cen.tolist()], dtype=np.int64)
    uv = np.empty(len(m.lv) * 2, dtype=np.float32)
    me.uv_layers[0].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    cnt = m.lt[sel]
    li = np.repeat(m.ls[sel], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    tl = np.repeat(tri, cnt)
    p = m.co[m.lv[li]]
    p0 = P0[tl, 0]
    e1, e2 = P0[tl, 1] - p0, P0[tl, 2] - p0
    g11, g12, g22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    det = g11 * g22 - g12 * g12
    r1, r2 = ((p - p0) * e1).sum(1), ((p - p0) * e2).sum(1)
    safe = np.where(np.abs(det) < 1e-16, 1.0, det)
    al, be = (g22 * r1 - g12 * r2) / safe, (g11 * r2 - g12 * r1) / safe
    new = UV0[tl, 0] + al[:, None] * (UV0[tl, 1] - UV0[tl, 0]) + be[:, None] * (UV0[tl, 2] - UV0[tl, 0])
    nl = np.repeat(nrm, cnt, axis=0)
    bad = (np.abs(det) < 1e-16) | ((nl * N0[tl]).sum(1) < 0.5)
    if bad.any():
        # box projection in metres, as tk.surf does it, where the old face looks another way
        ax = np.argmax(np.abs(nl), axis=1)
        sg = np.sign(nl[np.arange(len(nl)), ax])
        sg[sg == 0] = 1
        u = np.where(ax == 0, p[:, 1] * sg, np.where(ax == 1, -p[:, 0] * sg, p[:, 0]))
        v = np.where(ax == 2, p[:, 1] * sg, p[:, 2])
        new[bad] = np.stack([u, v], axis=1)[bad]
    uv[li] = new
    me.uv_layers[0].data.foreach_set("uv", uv.astype(np.float32).ravel())


COVER_IN = (0.004, 0.02, 0.05)  # A flat piece of a field (an ashlar block, a slate) has come open where its face is
                                #   gone and what shows in its place is not the same to the eye (_open_mask): the
                                #   core of tower_keep_ornate's walls lies 9 mm behind the joints of its blocks, the
                                #   same stone but darker - too near for any picture of the probe to tell, and
                                #   between the shards of a welded skin plain to see. Depth of a piece: its own and
                                #   this (m; at least / at most the other two)
COVER_OUT = 0.5                 # x the size limit: how far from where it was a face of the field still covers
COVER_TONE = 0.25               # (see _open_mask)
COVER_MAX = 0.03                # share of the fields a weld may open ...
COVER_MAX1 = 0.06               # ... and of any one field (object;
COVER_FEW = 40                  #     one with fewer points than this counts as if it had this many)
COVER_N = 9000                  # points measured


def _cover_points(m, isl, sel, ms):
    """Points on the flat loose pieces `sel` (per piece) of an object, on the faces that look the way the piece does,
    ISL_GRID x the size limit apart (wider when that is too many)
    -> None or {"P": points, "N": the direction of their piece, "tin": its depth (COVER_IN), "mat": the name of
    their material, "g": their painted grime}"""
    T, tp = m.tris()
    if not len(T) or not sel.any():
        return None
    k = isl["n"]
    pi = isl["pisl"][tp]
    A = m.co[T[:, 0]]
    e1, e2 = m.co[T[:, 1]] - A, m.co[T[:, 2]] - A
    cr = np.cross(e1, e2)
    ar = 0.5 * np.linalg.norm(cr, axis=1)
    N = np.stack([np.bincount(pi, weights=cr[:, i], minlength=k) for i in range(3)], axis=1) * 0.5
    ln = np.linalg.norm(N, axis=1)
    flat = sel & (ln >= PATCH_FLAT * np.maximum(np.bincount(pi, weights=ar, minlength=k), 1e-12))
    N /= np.maximum(ln, 1e-12)[:, None]
    use = flat[pi] & (np.einsum("ij,ij->i", cr, N[pi]) > 0.7 * 2.0 * ar) & (ar > 0)
    if not use.any():
        return None
    # the depth of a piece along the way it looks
    h = np.einsum("ij,ij->i", m.co[T].reshape(-1, 3), np.repeat(N[pi], 3, axis=0))
    pj = np.repeat(pi, 3)
    lo, hi = np.full(k, np.inf), np.full(k, -np.inf)
    np.minimum.at(lo, pj, h)
    np.maximum.at(hi, pj, h)
    depth = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    cell = max((ISL_GRID * ms) ** 2, float(ar[use].sum()) / COVER_N, 1e-12)
    rng = np.random.default_rng(23)
    n = np.where(use, np.floor(ar / cell + rng.random(len(ar))), 0).astype(np.int64)
    if not n.sum():
        return None
    ti = np.repeat(np.arange(len(T)), n)
    u, v = rng.random(len(ti)), rng.random(len(ti))
    fl = u + v > 1.0
    u[fl], v[fl] = 1.0 - u[fl], 1.0 - v[fl]
    g = _grime(m)[T[ti]].mean(axis=1)
    return {"P": A[ti] + e1[ti] * u[:, None] + e2[ti] * v[:, None], "N": N[pi[ti]],
            "tin": np.clip(depth[pi[ti]] + COVER_IN[0], COVER_IN[1], COVER_IN[2]),
            "mat": np.array([m.names[i] for i in m.mi[tp[ti]].tolist()], dtype=object), "g": g}


def _grime(m):
    """the painted grime of the vertices of a mesh (0 where none is painted)"""
    a = m.me.attributes.get("m_grime")
    g = np.zeros(len(m.co), dtype=np.float32)
    if a is not None and a.domain == 'POINT' and len(a.data) == len(g):
        a.data.foreach_get("value", g)
    return g.astype(np.float64)


def _set_rf(o, mask):
    """mark faces as faces of a field (see _open)"""
    me = o.data
    a = me.attributes.get("rf")
    if a is None:
        a = me.attributes.new("rf", 'INT', 'FACE')
    v = np.zeros(len(me.polygons), dtype=np.int32)
    a.data.foreach_get("value", v)
    v[mask] = 1
    a.data.foreach_set("value", v)


def _open(S, per=False):
    """share of the cover points (S.cover, see _weld) that have come open since they were taken
    (per=True: -> that share, and the largest share among the fields - objects - they were taken from)"""
    c = S.cover
    if c is None:
        return (0.0, 0.0) if per else 0.0
    new = _open_mask(S, c) & ~c["base"]
    if not per:
        return float(new.mean())
    return float(new.mean()), float((np.bincount(c["oid"], weights=new) / np.maximum(np.bincount(c["oid"]), COVER_FEW)).max())


def _open_mask(S, cover):
    """The cover points that lie open. A point is covered when the first face met along the way its piece looked
    (from COVER_OUT x the size limit before it to as much behind; faces that look away are culled) is a face of a
    field ("rf"), or lies no deeper than a pixel at the level's distance behind it and is of its material and as
    grimy (within COVER_TONE) - a roof may lose its tiles to the deck of the same clay under them, a wall not its
    ashlar to the dark core."""
    P, N, tin = cover["P"], cover["N"], cover["tin"]
    ids = {}
    Vs, Ts, Ws, Rs, Ms, Gs, off = [], [], [], [], [], [], 0
    for o in S.objs:
        m = S.an(o)[0]
        if not len(m.lt):
            continue
        T, tp = m.tris()
        rf = np.zeros(len(m.lt), dtype=np.int32)
        a = o.data.attributes.get("rf")
        if a is not None and a.domain == 'FACE' and len(a.data) == len(rf):
            a.data.foreach_get("value", rf)
        Vs.append(m.co)
        Ts.append(T + off)
        Ws.append(m.two[m.mi[tp]])
        Rs.append(rf[tp] > 0)
        Ms.append(np.array([ids.setdefault(nm, len(ids)) for nm in m.names], dtype=np.int64)[m.mi[tp]])
        Gs.append(_grime(m)[T].mean(axis=1))
        off += len(m.co)
    out = np.ones(len(P), dtype=bool)
    if not Vs:
        return out
    V, T, two, rf, mat, gr = (np.vstack(Vs), np.vstack(Ts), np.concatenate(Ws), np.concatenate(Rs), np.concatenate(Ms),
                              np.concatenate(Gs))
    A = V[T[:, 0]]
    fn = np.cross(V[T[:, 1]] - A, V[T[:, 2]] - A)
    bvh = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True, epsilon=0.0)
    tout = np.maximum(COVER_OUT * S.ms, tin)
    px = np.maximum(tin, _im.DIST[S.level] * _im.PX_PER_M)
    pm = np.array([ids.get(nm, -1) for nm in cover["mat"].tolist()], dtype=np.int64)
    g0 = cover["g"]
    O = P + N * tout[:, None]
    cast = bvh.ray_cast
    for i, (o_, d_, r_) in enumerate(zip(O.tolist(), (-N).tolist(), (2.0 * tout).tolist())):
        o_ = Vector(o_)
        d_ = Vector(d_)
        gone = 0.0
        for _ in range(8):
            h = cast(o_, d_, r_)
            if h[0] is None:
                break
            f = h[2]
            gone += h[3]
            if two[f] or float(fn[f] @ N[i]) > 0.0:
                out[i] = not (rf[f] or (gone - tout[i] <= px[i] and mat[f] == pm[i] and abs(gr[f] - g0[i]) <= COVER_TONE))
                break
            r_ -= h[3] + 1e-5
            gone += 1e-5
            if r_ <= 0.0:
                break
            o_ = o_ + d_ * (h[3] + 1e-5)
    return out


def _weld_obj(o, dist, level):
    me = o.data
    bm = bmesh.new()
    bm.from_mesh(me)
    rk = bm.faces.layers.int.get("rk")
    ff = [f for f in bm.faces if f[rk] == 2]
    if not ff:
        bm.free()
        return
    bmesh.ops.remove_doubles(bm, verts=list({v for f in ff for v in f.verts}), dist=dist)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=bm.edges[:])
    bm.verts.index_update()
    seen, dup = set(), []
    for f in bm.faces:
        key = tuple(sorted(v.index for v in f.verts))
        if key in seen:
            dup.append(f)
        else:
            seen.add(key)
    if dup:
        bmesh.ops.delete(bm, geom=dup, context='FACES')
    ff = [f for f in bm.faces if f[rk] == 2]
    if ff:
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(ANGLE_W[level]), use_dissolve_boundaries=False,
                                 verts=list({v for f in ff for v in f.verts}), edges=list({e for f in ff for e in f.edges}),
                                 delimit={'MATERIAL'})
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big, quad_method='BEAUTY', ngon_method='BEAUTY')
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges[:])
    le = [e for e in bm.edges if not e.link_faces]
    if le:
        bmesh.ops.delete(bm, geom=le, context='EDGES')
    lv = [v for v in bm.verts if not v.link_faces]
    if lv:
        bmesh.ops.delete(bm, geom=lv, context='VERTS')
    bm.to_mesh(me)
    bm.free()
    me.update()


def _set_rk(o, mask, value):
    me = o.data
    a = me.attributes.get("rk")
    if a is None:
        a = me.attributes.new("rk", 'INT', 'FACE')
    v = np.zeros(len(me.polygons), dtype=np.int32)
    a.data.foreach_get("value", v)
    v[mask] = value
    a.data.foreach_set("value", v)


def _weld(S, T, probe, base, enough, log):
    """Fields of small pieces welded into one skin, in growing distances, as long as the picture stands it and the
    flat pieces of the fields stay covered (S.cover: the points _open() measures from here on - a weld wider than
    the pieces eats them, and what lay behind the skin shows through in shards).
    enough(): True once the budget is within reach. -> True when a stage had to be taken back (or none stood)."""
    level = S.level
    fobj, cps = [], []
    for oi, a, b in T["span"]:
        fm = (T["kind"][a:b] == K_FIELD) & ~T["near"][a:b]
        if not fm.any():
            continue
        o = S.objs[oi]
        m, isl = S.an(o)
        pm = fm[isl["pisl"]]
        _set_rk(o, pm, 2)
        _set_rf(o, pm)
        Tt, tp = m.tris()
        ft = Tt[pm[tp]]
        P0 = m.co[ft]
        UV0 = np.zeros((len(ft), 3, 2))
        if o.data.uv_layers:
            uv = np.empty(len(m.lv) * 2, dtype=np.float32)
            o.data.uv_layers[0].data.foreach_get("uv", uv)
            lt_ = np.empty(len(Tt) * 3, dtype=np.int32)
            o.data.loop_triangles.foreach_get("loops", lt_)
            UV0 = uv.reshape(-1, 2).astype(np.float64)[lt_.reshape(-1, 3)[pm[tp]]]
        N0 = np.cross(P0[:, 1] - P0[:, 0], P0[:, 2] - P0[:, 0])
        N0 /= np.maximum(np.linalg.norm(N0, axis=1, keepdims=True), 1e-20)
        bvh = BVHTree.FromPolygons(P0.reshape(-1, 3).tolist(), np.arange(len(P0) * 3).reshape(-1, 3).tolist(), all_triangles=True, epsilon=0.0)
        fobj.append((o, (P0, UV0, N0, bvh)))
        cp = _cover_points(m, isl, fm, S.ms)
        if cp is not None:
            cp["oid"] = np.full(len(cp["P"]), len(cps), dtype=np.int64)         # the field they are of
            cps.append(cp)
        S.dirty(o)
    if not fobj:
        return False
    if cps:
        S.cover = {k: np.concatenate([c[k] for c in cps]) for k in cps[0]}
        S.cover["base"] = _open_mask(S, S.cover)                # open before anything is welded
        if S.cache is not None:
            S.cache.setdefault("cover", {})[level] = S.cover
    objs = [o for o, _ in fobj]
    good = S.save(objs)                 # the last state that stood
    stood, failed = 0.0, False
    t_before = S.tris()
    done = {}                           # object -> the distance it stood (x the size limit)

    def stands():
        m = probe.measure(*S.soup())
        ok = (m["sil"] - base["sil"] <= 0.25 * SIL[level] and m["hole"] - base["hole"] <= 0.01
              and abs(m["lum"] - base.get("lum", 1.0)) <= LUMTOL[level]
              and (m["glow"] is None or base["glow"] is None or m["glow"] >= base["glow"] - 0.03)
              and not S.sides(S.tol) and not S.empty_parts())
        op, op1 = _open(S, per=True) if ok else (0.0, 0.0)
        ok = ok and op <= COVER_MAX and op1 <= COVER_MAX1
        if not ok:
            log.setdefault("weld_stop", []).append(
                [wf, len(act), round(m["sil"] - base["sil"], 4), round(m["hole"] - base["hole"], 4),
                 round(m["lum"] - base.get("lum", 1.0), 4), m["glow"], base["glow"], round(op, 4), round(op1, 4)])
        return ok

    def weld(pairs):
        for o, old in pairs:
            _weld_obj(o, S.ms * wf, level)
            _reanchor(o, old)
            S.dirty(o)

    def back(pairs):
        for o, _ in pairs:
            old_me = o.data
            o.data = good[o.name][1].copy()
            if old_me.users == 0:
                bpy.data.meshes.remove(old_me)
            S.dirty(o)

    def keep(pairs):
        nonlocal good
        S.forget(good)
        good = S.save(objs)
        for o, _ in pairs:
            done[o.name] = wf

    act = list(fobj)
    ladder = list(WELD)
    while ladder and act:
        wf = ladder.pop(0)
        weld(act)
        if stands():
            keep(act)
            stood = max(stood, wf)
            if enough():
                break
            continue
        failed = True
        back(act)
        if len(act) > 1:
            # not all of them at once: one field after the other (the largest first), the ones the picture stands
            act.sort(key=lambda x: -len(x[0].data.polygons))
            ok_ = []
            for pr in act[:WELD_SINGLE]:
                weld([pr])
                if stands():
                    keep([pr])
                    ok_.append(pr)
                else:
                    back([pr])
            act = ok_
            if act:
                stood = max(stood, wf)
                if enough():
                    break
                continue
        if not done and wf > WELD_LESS[-1]:
            # not even the first distance: the shorter ones
            act = list(fobj)
            ladder = [x for x in WELD_LESS if x < wf]
            continue
        break
    S.forget(good)
    for o in objs:
        _set_rk(o, S.an(o)[0].rk() == 2, 0)
        S.dirty(o)
    for o in objs:
        if o.name not in done:
            o["tsq_nofield"] = 1                    # these go on to the collapse like any dense mesh
    if done:
        log["weld_objs"] = [len(done), len(objs)]
    log["weld_m"] = round(S.ms * stood, 4)
    log["weld_failed"] = bool(failed)
    log["weld_tris"] = t_before - S.tris()
    return failed


def _patches(S, T, log):
    """Fields of flat pieces that lie in a plane (roof tiles, slates, paving) -> flat patches in that plane: the
    pieces are grouped by where they look and the plane they lie in (a cone falls into sectors), and every row of
    a group becomes one rectangle over the pieces it holds - so openings in the field stay open and its outline
    stays the outline of its pieces. -> triangles saved"""
    cosa = math.cos(math.radians(PATCH_ANG))
    dtol = 0.5 * S.ms
    saved_t = made = 0
    for oi, a, b in T["span"]:
        fm = T["kind"][a:b] == K_FIELD
        if fm.any():
            _set_rf(S.objs[oi], fm[S.an(S.objs[oi])[1]["pisl"]])        # this plan starts from before the weld
        if int(fm.sum()) < PATCH_N:
            continue
        o = S.objs[oi]
        m, isl = S.an(o)
        me = o.data
        k, pisl = isl["n"], isl["pisl"]
        ar = m.area()
        an = m.normal() * ar[:, None]
        N = np.stack([np.bincount(pisl, weights=an[:, i], minlength=k) for i in range(3)], axis=1)
        ln = np.linalg.norm(N, axis=1)
        N /= np.maximum(ln, 1e-12)[:, None]
        cen = 0.5 * (isl["mn"] + isl["mx"])
        todo = fm & (ln >= PATCH_FLAT * np.maximum(isl["area"], 1e-12))
        seed = np.where(todo, isl["area"], -1.0)
        groups = []
        while True:
            s = int(np.argmax(seed))
            if seed[s] < 0:
                break
            mem = todo & (N @ N[s] >= cosa) & (np.abs((cen - cen[s]) @ N[s]) <= dtol)
            if int(mem.sum()) >= PATCH_N:
                groups.append(np.flatnonzero(mem))
                todo[mem] = False
                seed[mem] = -1.0
            else:
                seed[s] = -1.0                  # founds no group of its own, may still join one
        if not groups:
            continue
        vo, vs, nv = isl["vo"], isl["vs"], isl["nv"]
        rects, kill = [], np.zeros(k, dtype=bool)
        for g in groups:
            n = (N[g] * isl["area"][g][:, None]).sum(0)
            n /= max(float(np.linalg.norm(n)), 1e-12)
            if abs(n[2]) < 0.95:
                u = np.cross([0.0, 0.0, 1.0], n)
            else:
                ang = _min_rect(cen[g][:, 0], cen[g][:, 1])[0]
                u = np.array([math.cos(ang), math.sin(ang), 0.0])
                u -= n * float(u @ n)
            u /= max(float(np.linalg.norm(u)), 1e-12)
            v = np.cross(n, u)
            cnt = nv[g]
            st = np.r_[0, np.cumsum(cnt)[:-1]]
            P = m.co[vo[np.repeat(vs[g] - st, cnt) + np.arange(int(cnt.sum()))]]
            pu, pv = P @ u, P @ v
            u0, u1 = np.minimum.reduceat(pu, st), np.maximum.reduceat(pu, st)
            v0, v1 = np.minimum.reduceat(pv, st), np.maximum.reduceat(pv, st)
            ph = P @ n
            en = st + cnt
            # about the upper face of every piece; a rectangle lies where the pieces of its rows do (a battered
            # wall and the plumb one above it fall into one group, 0.5 x the size limit apart)
            hp = np.array([np.percentile(ph[a_:b_], 60) for a_, b_ in zip(st.tolist(), en.tolist())])
            wu, hv = float(np.median(u1 - u0)), float(np.median(v1 - v0))
            if wu < 1e-5 or hv < 1e-5:
                continue
            vc = 0.5 * (v0 + v1)
            row = np.floor((vc - vc.min()) / hv).astype(np.int64)
            tol = min(0.6 * wu, 0.5 * S.ms)
            done, open_ = [], []                # rectangles [ua, ub, va, vb, first ua, first ub, last row, rows]
            for r in np.unique(row).tolist():
                ii = np.flatnonzero(row == r)
                ii = ii[np.argsort(u0[ii], kind="stable")]
                runs = []
                for i in ii.tolist():
                    if runs and u0[i] <= runs[-1][1] + 0.5 * wu:
                        c = runs[-1]
                        c[1], c[2], c[3] = max(c[1], u1[i]), min(c[2], v0[i]), max(c[3], v1[i])
                        c[4].append(i)
                    else:
                        runs.append([u0[i], u1[i], v0[i], v1[i], [i]])
                nxt = []
                for c in runs:
                    at = (float(np.mean(vc[c[4]])), float(np.median(hp[c[4]])))     # where the run lies: v, height
                    for q in open_:
                        if q[6] == r - 1 and abs(c[0] - q[4]) <= tol and abs(c[1] - q[5]) <= tol and c[2] <= q[3] + 0.5 * hv:
                            q[0], q[1], q[2], q[3], q[6] = min(q[0], c[0]), max(q[1], c[1]), min(q[2], c[2]), max(q[3], c[3]), r
                            q[7].append(at)
                            break
                    else:
                        nxt.append(c[:4] + [c[0], c[1], r, [at]])
                done += [q for q in open_ if q[6] < r]
                open_ = [q for q in open_ if q[6] == r] + nxt
            done += open_
            gt = int(isl["tris"][g].sum())
            if 2 * len(done) > 0.6 * gt:
                continue
            ing = np.zeros(k, dtype=bool)
            ing[g] = True
            pm = ing[pisl]
            mat_i = int(np.bincount(m.mi[pm], weights=ar[pm]).argmax())
            for q in done:
                cs = [(q[0], q[2]), (q[1], q[2]), (q[1], q[3]), (q[0], q[3])]
                vr, hr = np.array(q[7]).T
                if len(vr) > 1 and float(np.ptp(vr)) > 1e-6:
                    k1, k0 = np.polyfit(vr, hr, 1)              # rows one above the other: the slope they lie on
                else:
                    k1, k0 = 0.0, float(hr.mean())
                rects.append(([n * (k0 + k1 * y) + u * x + v * y for x, y in cs], cs, mat_i))
            kill |= ing
            saved_t += gt - 2 * len(done)
            made += len(done)
        if not rects:
            continue
        bm = bmesh.new()
        bm.from_mesh(me)
        rk = bm.faces.layers.int.get("rk") or bm.faces.layers.int.new("rk")
        rf = bm.faces.layers.int.get("rf") or bm.faces.layers.int.new("rf")
        uvl = bm.loops.layers.uv.active or bm.loops.layers.uv.new("UVMap")
        bm.faces.ensure_lookup_table()
        f = bm.faces
        bmesh.ops.delete(bm, geom=[f[j] for j in np.flatnonzero(kill[pisl]).tolist()], context='FACES')
        for corners, uvs, mat_i in rects:
            fc = bm.faces.new([bm.verts.new(p) for p in corners])
            fc.material_index = mat_i
            fc.smooth = False
            fc[rk] = 1                          # as the prisms: not collapsed afterwards
            fc[rf] = 1
            for lp, w in zip(fc.loops, uvs):
                lp[uvl].uv = w                  # metres in the plane, as tk.surf lays its textures
        bm.to_mesh(me)
        bm.free()
        me.update()
        S.dirty(o)
    log["patches"] = made
    log["patch_tris_saved"] = saved_t
    return saved_t


def _tidy(me):
    """what tk.final._clean would drop anyway (faces without area, loose edges and vertices) - so that the bounds
    measured here are the bounds the level ends up with"""
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges[:])
    le = [e for e in bm.edges if not e.link_faces]
    if le:
        bmesh.ops.delete(bm, geom=le, context='EDGES')
    lv = [v for v in bm.verts if not v.link_faces]
    if lv:
        bmesh.ops.delete(bm, geom=lv, context='VERTS')
    bm.to_mesh(me)
    bm.free()
    me.update()


def _split_colours(me):
    """Cut the mesh open where two materials of clearly different colour meet: the collapse keeps open borders,
    but knows nothing of materials - a striped cloth would come out of it with its stripes smeared."""
    mats = list(me.materials)
    if len(mats) < 2:
        return
    g = [_tone(m, _lib(m)) ** (1.0 / 2.2) for m in mats]
    if max(g) - min(g) < TONE_CUT:
        return
    bm = bmesh.new()
    bm.from_mesh(me)
    n = len(g) - 1
    es = []
    for e in bm.edges:
        lf = e.link_faces
        if len(lf) >= 2:
            t = [g[min(f.material_index, n)] for f in lf]
            if max(t) - min(t) >= TONE_CUT:
                es.append(e)
    if es:
        bmesh.ops.split_edges(bm, edges=es)
        bm.to_mesh(me)
        me.update()
    bm.free()


def _decimated(o, ratio, pin=None):
    """collapse-decimated copy of the mesh of o -> new mesh. pin: name of a vertex group of o whose vertices stay
    where they are (Blender never collapses an edge at a vertex of weight 0; the ratio is then not always reached)"""
    m = o.modifiers.new("RED", 'DECIMATE')
    m.decimate_type = 'COLLAPSE'
    m.ratio = float(ratio)
    m.use_collapse_triangulate = True
    if pin:
        m.vertex_group = pin
        m.invert_vertex_group = True
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    o.modifiers.clear()
    me.validate()
    _tidy(me)
    return me


def _decimated_but(o, src, pin, ratio):
    """as _decimated(src), but the polygons in the mask `pin` stay as they are -> new mesh"""
    a = src.copy()
    o.data = a
    _delete_polys(o, pin)
    me = _decimated(o, ratio)
    b = src.copy()
    o.data = b
    _delete_polys(o, ~pin)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.from_mesh(b)
    bm.to_mesh(me)
    bm.free()
    me.update()
    o.data = src
    bpy.data.meshes.remove(a)
    bpy.data.meshes.remove(b)
    return me


ISL_FLOOR = 16                  # the collapse leaves a loose piece of up to this many triangles alone ...
ISL_AREA = 0.5                  # ... and one that would come out of it with less than this share of its surface,
ISL_MOVE = 0.35                 #     or with a side of its box moved by more than this x the size limit (or a quarter
ISL_POW = 0.5                   #     of its own size). Pieces are brought this far (0: not, 1: fully) to one size
                                #     before the collapse, so that it does not eat the small ones first.
ISL_LADDER = (0.35, 0.6, 1.0, 1.7, 2.9, 5.0)    # the ratios tried, x the one the part needs
ISL_GENTLE = (0.25, 0.5, 0.75)   # ... and these, for the pieces that stand none of those
ISL_RELAXED = (1.0, 0.7, 0.45, 0.25, 0.12)     # ... of the last resort: the next harder one when pieces had to be put back
ISL_PROBE = 30000               # vertices of the part measured against every collapsed copy (at least 6 per piece)
ISL_GRID = 0.4                  # ... and points on its faces, this x the size limit apart


def _isl_dev(pts, pid, k, me):
    """how far the collapsed copy `me` lies from the points `pts` of the pieces `pid` (_isl_sample) -> per piece the
    largest distance (m)"""
    from mathutils.bvhtree import BVHTree
    nv = len(me.vertices)
    out = np.zeros(k)
    if nv == 0 or len(me.polygons) == 0:
        return out + 1e9
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    lt = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    if (lt == 3).all():
        polys = lv.reshape(-1, 3).tolist()
    else:
        st = np.cumsum(lt) - lt
        polys = [lv[a:a + n].tolist() for a, n in zip(st.tolist(), lt.tolist())]
    tree = BVHTree.FromPolygons(co.reshape(-1, 3).tolist(), polys, all_triangles=bool((lt == 3).all()))
    d = np.zeros(len(pts))
    near = tree.find_nearest
    for i, p_ in enumerate(pts.tolist()):
        d[i] = near(p_)[3]
    np.maximum.at(out, pid, d)
    return out


def _isl_sample(m, isl, ms):
    """The points a collapsed copy is measured at: up to ISL_PROBE vertices (at least 6 of every piece), and as
    many points on the faces, ISL_GRID x the size limit apart (wider when that is too many) - a flat wall has its
    vertices on its rim only, and a hole torn into the middle of it is far from none of them.
    -> points (n, 3) in the scaled-back space of the part, their pieces"""
    vi, k = isl["visl"], isl["n"]
    ok = np.flatnonzero(vi >= 0)
    if len(ok) > ISL_PROBE:
        o = ok[np.argsort(vi[ok], kind="stable")]
        g = vi[o]
        st = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        cnt = np.diff(np.r_[st, len(g)])
        rank = np.arange(len(g)) - np.repeat(st, cnt)
        step = np.maximum(1, np.repeat(np.minimum(cnt // 6, max(1, int(round(len(ok) / ISL_PROBE)))), cnt))
        ok = o[rank % step == 0]
    P, G = [m.co[ok]], [vi[ok]]
    T, tp = m.tris()
    if len(T):
        A = m.co[T[:, 0]]
        e1, e2 = m.co[T[:, 1]] - A, m.co[T[:, 2]] - A
        ar = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        cell = max((ISL_GRID * ms) ** 2, float(ar.sum()) / ISL_PROBE, 1e-12)
        rng = np.random.default_rng(11)
        n = np.floor(ar / cell + rng.random(len(ar))).astype(np.int64)
        if n.sum():
            ti = np.repeat(np.arange(len(T)), n)
            u, v = rng.random(len(ti)), rng.random(len(ti))
            fl = u + v > 1.0
            u[fl], v[fl] = 1.0 - u[fl], 1.0 - v[fl]
            P.append(A[ti] + e1[ti] * u[:, None] + e2[ti] * v[:, None])
            G.append(isl["pisl"][tp[ti]])
    return np.concatenate(P), np.concatenate(G)


def _isl_read(me, k, ctr, sc):
    """A collapsed copy (face attribute "ri": the loose piece every face came from, vertices scaled about the
    centres of the pieces): puts the vertices back to size -> per face piece, per piece triangles, area, box"""
    npo = len(me.polygons)
    ri = np.empty(npo, dtype=np.int32)
    me.attributes["ri"].data.foreach_get("value", ri)
    lt = np.empty(npo, dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    nv = len(me.vertices)
    v2 = np.full(nv, -1, dtype=np.int64)
    v2[lv] = np.repeat(ri, lt)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    ok = v2 >= 0
    co[ok] = ctr[v2[ok]] + (co[ok] - ctr[v2[ok]]) / sc[v2[ok]][:, None]
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()
    ar = np.empty(npo, dtype=np.float32)
    me.polygons.foreach_get("area", ar)
    t2 = np.bincount(ri, weights=lt - 2, minlength=k)
    a2 = np.bincount(ri, weights=ar, minlength=k)
    order = np.argsort(v2[ok], kind="stable")
    vs_ = v2[ok][order]
    st = np.flatnonzero(np.r_[True, vs_[1:] != vs_[:-1]]) if len(vs_) else np.zeros(0, dtype=np.int64)
    mn2 = np.full((k, 3), np.inf)
    mx2 = np.full((k, 3), -np.inf)
    if len(st):
        cs = co[ok][order]
        mn2[vs_[st]] = np.minimum.reduceat(cs, st, axis=0)
        mx2[vs_[st]] = np.maximum.reduceat(cs, st, axis=0)
    return ri, t2, a2, mn2, mx2


def _zcards(S, j, reach, own=False):
    """The cards of the scene outside the object j (own: and its own; glowing faces, room cards of windows) that
    come within `reach` of j's box -> None or (corners (3n, 3), triangles (n, 3), material name per triangle)"""
    b = _obj_bounds([j])
    lo, hi = b[0] - reach, b[1] + reach
    P, M = [], []
    for o in S.objs:
        if (o is j and not own) or not len(o.data.polygons):
            continue
        m = S.an(o)[0]
        T, tp = m.tris()
        gl = m.zglow[m.mi[tp]]
        if not gl.any():
            continue
        X = m.co[T[gl]]                                                 # (n, 3, 3)
        ok = (X.max(axis=1) >= lo).all(axis=1) & (X.min(axis=1) <= hi).all(axis=1)
        if not ok.any():
            continue
        P.append(X[ok])
        M.append(np.array(m.names, dtype=object)[m.mi[tp][gl][ok]])
    if not P:
        return None
    P = np.concatenate(P)
    return P.reshape(-1, 3), np.arange(3 * len(P), dtype=np.int64).reshape(-1, 3), np.concatenate(M)


def _zpiece(zc, co, polys, pname, ri, k, tris=False):
    """Card area (cards of _zcards) over which a parallel face of another material of this mesh lies within Z_NEAR
    (tk.lodimage.z_pairs: measured, not sampled), per loose piece ri of that face -> (k,) m2"""
    out = np.zeros(k)
    if not len(polys):
        return out
    Vc, Tc, Mc = zc
    if tris:
        T2 = np.asarray(polys, dtype=np.int64).reshape(-1, 3)
        f2 = np.arange(len(T2))
    else:
        T2, f2 = [], []
        for f, pl in enumerate(polys):
            for q in range(1, len(pl) - 1):
                T2.append((pl[0], pl[q], pl[q + 1]))
                f2.append(f)
        T2, f2 = np.asarray(T2, dtype=np.int64).reshape(-1, 3), np.asarray(f2, dtype=np.int64)
    if not len(T2):
        return out
    ids = {}
    mc = np.array([ids.setdefault(x, len(ids)) for x in Mc.tolist()], dtype=np.int64)
    m2 = np.array([ids.setdefault(x, len(ids)) for x in np.asarray(pname, dtype=object).tolist()], dtype=np.int64)[f2]
    r = _im.z_pairs(Vc, Tc, mc, np.ones(len(Tc), dtype=bool), V2=co, T2=T2, mat2=m2, reach=Z_NEAR)
    if r is None or not len(r[0]):
        return out
    np.add.at(out, np.asarray(ri)[f2[r[1]]], r[4])
    return out


def _mesh_faces(me):
    """-> vertices, polygons as lists, all triangles?, normals, material name of every polygon"""
    nv, npo = len(me.vertices), len(me.polygons)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    lt = np.empty(npo, dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    tris = bool((lt == 3).all())
    if tris:
        polys = lv.reshape(-1, 3).tolist()
    else:
        st = np.cumsum(lt) - lt
        polys = [lv[a:a + n].tolist() for a, n in zip(st.tolist(), lt.tolist())]
    nr = np.empty(npo * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", nr)
    mi = np.empty(npo, dtype=np.int32)
    me.polygons.foreach_get("material_index", mi)
    names = np.array([mt.name if mt is not None else "" for mt in me.materials] or [""], dtype=object)
    return co.reshape(-1, 3).astype(np.float64), polys, tris, nr.reshape(-1, 3).astype(np.float64), names[np.clip(mi, 0, len(names) - 1)]


def _collapse_part(S, j, ratio, strict, log, key, pn):
    """Collapse the joined mesh of one part to `ratio` of its triangles, loose piece by loose piece: Blender's
    decimate never crosses from one piece to the next, but it measures its error in absolute terms and so empties
    the small pieces before it touches the large ones - that is what tore crowns and left spires floating. Here the
    pieces are scaled towards one size first, the smallest are left out, and the collapse is made at a ladder of
    ratios; every piece then takes the hardest step at which it is still whole (it is still there, the sides of
    its box are where they were, its surface is not halved), and none at all when there is no such step. The part
    may so end over the ratio: what cannot be had without tearing is left.
    strict=False (last resort, the old way): one ratio for all, harder (ISL_RELAXED) when pieces had to be put
    back and the part so stays over what was asked for; a piece is put back only when it lies on a side of the
    lod-0 box that nothing would hold any more.
    Either way a piece does not take a step that lays it onto a card of the scene (a glowing face, the room card of
    a window: within Z_NEAR of it over more of the card than it lies there now, measured by tk.lodimage.z_pairs): a
    mullion flattened into its pane, a bar that comes to cross it at a slant, fights with it for the depth buffer.
    -> the hardest ratio a piece was given, or None when nothing was done"""
    n0 = _ntris(j.data)
    if n0 <= 8:
        return None
    off0 = S.sides(S.tolc)                              # (a side an earlier step gave up is not this one's to answer for)
    _split_colours(j.data)
    src = j.data
    m = _M(j)
    isl = _islands(m)
    k, pisl, vi = isl["n"], isl["pisl"], isl["visl"]
    t_i = isl["tris"].astype(np.int64)
    d_i = np.maximum(isl["d"], 1e-6)
    ctr = 0.5 * (isl["mn"] + isl["mx"])
    sc = (float(d_i.max()) / d_i) ** ISL_POW if strict else np.ones(k)
    tau = np.minimum(ISL_MOVE * S.ms, 0.25 * d_i)
    pin = (t_i <= ISL_FLOOR) if strict else np.zeros(k, dtype=bool)
    nfree = int(t_i[~pin].sum())
    want = ratio * n0
    if nfree <= 8:
        return None
    need = (want - (n0 - nfree)) / nfree
    top = 0.9 if strict else 0.97
    rs = sorted({float(np.clip(need * f, 0.02, top)) for f in (ISL_LADDER if strict else ISL_RELAXED)}
                | ({r for r in ISL_GENTLE if r > need} if strict else set()), reverse=not strict)
    if min(rs) >= top and need >= top:
        return None
    # the sides of the lod-0 box this part reaches, and the pieces that reach each
    on_side = []
    for a_ in range(3):
        for lo_ in (True, False):
            on = (isl["mn"][:, a_] <= S.b0[0][a_] + S.tolc) if lo_ else (isl["mx"][:, a_] >= S.b0[1][a_] - S.tolc)
            if on.any():
                on_side.append((a_, lo_, on))
    okv = vi >= 0
    co2 = m.co.copy()
    co2[okv] = ctr[vi[okv]] + (m.co[okv] - ctr[vi[okv]]) * sc[vi[okv]][:, None]
    work = src.copy()
    j.data = work
    work.attributes.new("ri", 'INT', 'FACE').data.foreach_set("value", pisl.astype(np.int32))
    work.vertices.foreach_set("co", co2.astype(np.float32).ravel())
    if pin.any():
        _delete_polys(j, pin[pisl])
    # the vertices on the seam to faces that are not collapsed (lettering, glow, boxes) stay where they are
    vgn = None
    a = work.attributes.get("rp")
    if a is not None and len(work.vertices):
        rp = np.empty(len(work.vertices), dtype=np.int32)
        a.data.foreach_get("value", rp)
        if rp.any():
            vg = j.vertex_groups.new(name="tsq_pin")
            vg.add(np.flatnonzero(rp).tolist(), 1.0, 'REPLACE')
            vgn = vg.name
    pts, pid = _isl_sample(m, isl, S.ms) if strict else (None, None)
    # the cards around the part, and how much of them each piece lies on as it is
    zr = z0 = None
    if getattr(S, "zs", None) is not None:
        zr = _zcards(S, j, 0.05)
        if zr is not None:
            c0, p0, t0_, n0_, m0_ = _mesh_faces(src)
            z0 = _zpiece(zr, c0, p0, m0_, pisl, k, tris=t0_)
    zbad = 0
    made = []                                           # per step: mesh, per face piece, per piece triangles, whole
    for r in rs:
        me = _decimated(j, r, vgn)
        if len(me.polygons) == 0:
            bpy.data.meshes.remove(me)
            continue
        ri, t2, a2, mn2, mx2 = _isl_read(me, k, ctr, sc)
        with np.errstate(invalid="ignore"):
            mv = np.maximum(np.abs(mn2 - isl["mn"]), np.abs(mx2 - isl["mx"])).max(axis=1)
        mv[~np.isfinite(mv)] = 1e9
        gone = t2 == 0
        zok = None
        if zr is not None:
            c1, p1, t1_, n1_, m1_ = _mesh_faces(me)
            zok = _zpiece(zr, c1, p1, m1_, ri, k, tris=t1_) <= z0 * 1.02 + 2e-5
        if strict:
            whole = ~(gone | (mv > tau) | (a2 < ISL_AREA * isl["area"]))
            whole &= _isl_dev(pts, pid, k, me) <= tau
        else:
            # a side of the box stays where it is: when no piece that takes this step (or is kept back for a glowing
            # face) still reaches it, the pieces of that side stay as they are. A field of a thousand tufts flat on
            # the ground gives up any of them; a spire keeps its tip.
            whole = np.ones(k, dtype=bool)
            reaches = []
            for a_, lo_, on in on_side:
                with np.errstate(invalid="ignore"):
                    reach = (mn2[:, a_] <= S.b0[0][a_] + S.tolc) if lo_ else (mx2[:, a_] >= S.b0[1][a_] - S.tolc)
                reach = np.where(gone, False, reach) if zok is None else np.where(zok, reach & ~gone, True)
                reaches.append(reach)
                if not (on & reach).any():
                    whole &= ~on
        if zok is not None:
            zbad = max(zbad, int((whole & ~zok & ~pin).sum()))
            whole &= zok
        made.append((r, me, ri, t2, whole & ~pin) + (() if strict else (reaches,)))
        if not strict and float(np.where(whole, t2, t_i).sum()) <= want:
            break                                       # this ratio reaches what was asked for: no harder one is made
    if not strict:
        made.reverse()                                  # (the hardest step first, as the ladder below reads them)
    j.vertex_groups.clear()
    j.data = src
    bpy.data.meshes.remove(work)
    S.dirty(j)
    if zbad:
        log.setdefault("z_held", {})[f"{key}:{pn}"] = zbad      # pieces a step would have laid onto a glowing face

    def drop():
        for x in made:
            bpy.data.meshes.remove(x[1])

    if not made:
        return None
    nm = len(made)
    W = np.stack([x[4] for x in made])                  # step x piece
    Tm = np.stack([x[3] for x in made])
    best = None
    for js in range(nm - 1, -1, -1):                    # the gentlest start of the ladder that reaches the ratio
        sel = np.full(k, -1, dtype=np.int64)
        for q in range(nm - 1, js - 1, -1):
            sel[W[q]] = q
        tot = float(np.where(sel >= 0, Tm[np.maximum(sel, 0), np.arange(k)], t_i).sum())
        best = (sel, tot)
        if tot <= want:
            break
    sel, tot = best
    if js < nm - 1 and tot < want:
        # the step that reaches the ratio usually goes past it: the largest pieces go back to their next gentler
        # step as long as the part stays under what was asked for
        for i in np.argsort(-d_i, kind="stable").tolist():
            if sel[i] != js:
                continue
            up = np.flatnonzero(W[js + 1:, i])
            if not len(up):
                continue
            q2 = js + 1 + int(up[0])
            add = float(Tm[q2, i] - Tm[js, i])
            if add > 0 and tot + add <= want:
                sel[i] = q2
                tot += add
    if not strict:
        # the pieces now take different steps: a side none of them holds as chosen gets its pieces back
        for si, (a_, lo_, on) in enumerate(on_side):
            idx = np.flatnonzero(on)
            if not any(sel[i] < 0 or made[sel[i]][5][si][i] for i in idx.tolist()):
                sel[idx] = -1
        tot = float(np.where(sel >= 0, Tm[np.maximum(sel, 0), np.arange(k)], t_i).sum())
    if not (sel >= 0).any() or tot >= (0.98 if strict else 0.995) * n0:
        drop()
        return None

    def build():
        """the part with every piece at its step `sel` (-1: as it was) -> mesh, the ratios used"""
        out = bpy.data.meshes.new(src.name + "_c")
        for mt in src.materials:
            out.materials.append(mt)
        bm = bmesh.new()
        back = sel < 0
        if back.any():
            b = src.copy()
            j.data = b
            _delete_polys(j, ~back[pisl])
            bm.from_mesh(b)
            j.data = src
            bpy.data.meshes.remove(b)
        used = []
        for q, (r, me, ri, t2) in enumerate(x[:4] for x in made):
            pm = sel[ri] == q
            if not pm.any():
                continue
            used.append(r)
            b = me.copy()
            j.data = b
            _delete_polys(j, ~pm)
            a = b.attributes.get("ri")
            if a is not None:
                b.attributes.remove(a)
            bm.from_mesh(b)
            j.data = src
            bpy.data.meshes.remove(b)
        lay = bm.verts.layers.deform.active
        if lay is not None:
            bm.verts.layers.deform.remove(lay)
        lay = bm.verts.layers.int.get("rp")
        if lay is not None:
            bm.verts.layers.int.remove(lay)
        bm.to_mesh(out)
        bm.free()
        out.update()
        return out, used

    out, used = build()
    j.data = out
    S.dirty(j)
    hc = getattr(S, "hc", None)
    if strict and hc is not None:
        # no holes: a piece is not whole either when a viewer on the street now looks through it (a stoop that lost
        # its top, a sill its front). The rays that show nothing where lod 0 is closed and met this piece before
        # the collapse send it one step back, and to what it was when that does not close them
        ps = box = None
        held = 0
        for last in (False, True):
            if ps is not None:
                hm = _holes_read(S, hc, detail=True, box=box)
            else:
                j.data = src
                s_ = _ho.obj_soup([j])
                j.data = out
                if not len(s_["V"]):
                    break
                pad = 2.0 * S.ms + 0.05
                box = (s_["V"].min(0) - pad, s_["V"].max(0) + pad)
                hm = _holes_read(S, hc, detail=True, box=box)
                if hm["count"]:
                    ps = _ho._Soup(s_["V"], s_["T"], s_["two"], s_["thru"])
            if not hm["count"]:
                break
            D_ = hm["pts"] - hm["eyes"]
            t_ = np.linalg.norm(D_, axis=1)
            D_ = D_ / np.maximum(t_, 1e-12)[:, None]
            _t, tri_, _b = ps.cast(hm["eyes"], D_, t_ + 2.0)
            bad = np.unique(pisl[s_["poly"][tri_[tri_ >= 0]]])
            if TRACE:
                print(f"TSQ_TRACE guard {key}:{pn} {'last' if last else 'first'}: {hm['count']} hole samples, {int((tri_ >= 0).sum())} met the part before, "
                      f"pieces {bad.tolist()[:8]} at steps {sel[bad].tolist()[:8]} of {nm}, tris {t_i[bad].tolist()[:8]}; {(hm.get('where') or [{}])[0]}")
            bad = bad[sel[bad] >= 0]
            if not len(bad):
                break
            for i in bad.tolist():
                up = np.flatnonzero(W[sel[i] + 1:, i]) if not last else ()
                sel[i] = sel[i] + 1 + int(up[0]) if len(up) else -1
            held += len(bad)
            j.data = src                    # (a mesh removed while an object wears it takes the object along)
            bpy.data.meshes.remove(out)
            out, used = build()
            j.data = out
            S.dirty(j)
        if held:
            log.setdefault("holes_held", {})[f"{key}:{pn}"] = held
            tot = float(np.where(sel >= 0, Tm[np.maximum(sel, 0), np.arange(k)], t_i).sum())
            if not (sel >= 0).any():
                j.data = src
                bpy.data.meshes.remove(out)
                S.dirty(j)
                drop()
                return None
    drop()
    off = [x for x in S.sides(S.tolc) if x not in off0]
    if off or len(out.polygons) == 0:
        j.data = src
        bpy.data.meshes.remove(out)
        S.dirty(j)
        log.setdefault("rollback", []).append(f"{key}:{pn}@{min(used):.3f}" + (f" off bounds {off}" if off else ""))
        return None
    if tot > 1.05 * want:
        log.setdefault("islands", {})[f"{key}:{pn}"] = [int(k), int(pin.sum()), int(((sel < 0) & ~pin).sum()), int(_ntris(out)),
                                                       int(round(want))]
    if src.users == 0:
        bpy.data.meshes.remove(src)
    return min(used)


def _collapse(S, pick, ratio, log, key, strict=True):
    """Collapse the polygons pick(object) -> bool mask selects, joined per part, to `ratio` of their triangles
    (see _collapse_part: loose piece by loose piece; the parts may end over the ratio)."""
    groups = {}
    for o in list(S.objs):
        if not len(o.data.polygons):
            continue
        pm = pick(o)
        n = int(pm.sum())
        if n == 0:
            continue
        if n == len(pm):
            piece = o
            S.objs.remove(o)
        else:
            m = S.an(o)[0]
            pl = np.repeat(pm, m.lt)
            nv = len(m.co)
            seam = (np.bincount(m.lv[pl], minlength=nv) > 0) & (np.bincount(m.lv[~pl], minlength=nv) > 0)
            piece = o.copy()
            piece.data = o.data.copy()
            _link(piece)
            if seam.any():
                a = piece.data.attributes.get("rp") or piece.data.attributes.new("rp", 'INT', 'POINT')
                a.data.foreach_set("value", seam.astype(np.int32))
            _delete_polys(piece, ~pm)
            _delete_polys(o, pm)
        S.dirty(o)
        groups.setdefault(S.part(o), []).append(piece)
    used = []
    joined = [(pn, _join(pieces, f"red_{pn}")) for pn, pieces in groups.items()]
    S.objs.extend(j for _, j in joined)     # all of them first: a part still out of the scene hides nothing (holes)
    for pn, j in joined:
        r = _collapse_part(S, j, ratio, strict, log, key, pn)
        if r is not None:
            used.append(round(r, 4))
    if used:
        log[key] = min(used)


# ----------------------------------------------------------------------------
# Glowing faces and what lies on them. A lit window is a stack of cards millimetres apart (room card off the wall,
# veils, blind, glass); closer than Z_NEAR they fight for the depth buffer at walking distance. No step may bring a
# glowing face nearer to a face of another material than lod 0 has them (tk.lodimage.zfight measures it on the level
# file, gates() rejects the level). _zfix() runs after every step that moves or removes faces:
#   * what an opaque glowing card hides goes: faces of another material lying under it, looking at it, wholly
#     covered (the wall patch behind a room card) - and clear glass lying on a glowing face
#   * where a glowing face and another still lie within Z_NEAR, the corners of the other face are pushed along the
#     glowing face's normal until Z_GAP lies between them, to the side most of it is on (a mullion the collapse
#     flattened into the pane comes to lie in front of it). The glowing face never moves, nothing is welded or
#     snapped onto it, and no corner goes further than Z_SHIFT.
# What cannot be solved that way (a face that runs through the glowing one at a slant, corners both share) stays
# and is logged - the gate then decides.
# ----------------------------------------------------------------------------

def _zstart(S):
    """-> {"z0": glowing area with a partner under Z_NEAR as the model stands, "lim": what a level may have} or None
    when nothing glows"""
    z = _zscan(S)
    if z is None:
        return None
    w = z["w"]
    ga = 0.0
    for o in S.objs:
        m = S.an(o)[0]
        if len(m.lt):
            ga += float(m.area()[m.zglow[m.mi]].sum())
    z0 = float(w[z["d"] < Z_NEAR].sum())
    return {"z0": round(z0, 4), "lim": z0 + _im.z_tol(ga)}


def _zscan(S, far=_im.Z_FAR):
    """tk.lodimage.z_rays on the scene, and z_pairs for what lies within Z_NEAR -> None when nothing glows, else
    {"g", "p": triangle of the glowing face / of the face met, "d", "w", "s": distance, area, side per ray (per
    pair of triangles within Z_NEAR), "obj", "poly": object and polygon of every triangle,
    "n": triangle normals, "V", "T", "bvh", "voff": {object: where its vertices start in V}}"""
    names = {}
    Vs, Ts, Ms, Gs, OI, PI, off = [], [], [], [], [], [], 0
    any_ = False
    voff = {}
    for oi, o in enumerate(S.objs):
        m = S.an(o)[0]
        if not len(m.lt):
            continue
        T, tp = m.tris()
        voff[oi] = off
        ids = np.array([names.setdefault(nm, len(names)) for nm in m.names], dtype=np.int64)
        Vs.append(m.co)
        Ts.append(T + off)
        Ms.append(ids[m.mi[tp]])
        g = m.zglow[m.mi[tp]]
        any_ = any_ or bool(g.any())
        Gs.append(g)
        OI.append(np.full(len(T), oi, dtype=np.int64))
        PI.append(tp.astype(np.int64))
        off += len(m.co)
    if not any_:
        return None
    V, T = np.vstack(Vs), np.vstack(Ts)
    bvh = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True, epsilon=0.0)
    Ms, Gs = np.concatenate(Ms), np.concatenate(Gs)
    g, p, d, w, s = _im.z_rays(V, T, Ms, Gs, far, bvh=bvh)
    # within Z_NEAR the rays are replaced by what they sample (z_pairs): one row per card triangle and face over it,
    # its area in place of the ray's - a thin strip between two rays is seen too
    keep = d >= Z_NEAR
    xi, xj, xd, x3, x4, xs = _im.z_pairs(V, T, Ms, Gs, reach=Z_NEAR, bvh2=bvh)
    g, p, s = np.concatenate([g[keep], xi]), np.concatenate([p[keep], xj]), np.concatenate([s[keep], xs])
    d, w = np.concatenate([d[keep], np.minimum(xd, 0.999 * Z_NEAR)]), np.concatenate([w[keep], x4])
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    return {"g": g, "p": p, "d": d, "w": w, "s": s, "obj": np.concatenate(OI), "poly": np.concatenate(PI), "n": n,
            "V": V, "T": T, "bvh": bvh, "voff": voff}


def _zhidden(S, z):
    """delete what an opaque glowing card hides, and clear glass lying on a glowing face -> triangles gone"""
    g, p, d, s = z["g"], z["p"], z["d"], z["s"]
    if not len(g):
        return 0
    obj, poly, n = z["obj"], z["poly"], z["n"]
    # per ray: is the glowing face opaque; the partner: glows, see-through, two-sided, lettering, put back by mend
    fl = {}
    for oi in np.unique(np.r_[obj[g], obj[p]]).tolist():
        m = S.an(S.objs[oi])[0]
        fl[oi] = (m.zglow[m.mi], m.see[m.mi], m.two[m.mi], m.tx() > 0, m.rk() == 1)

    def col(tri, k):
        out = np.zeros(len(tri), dtype=bool)
        for oi in np.unique(obj[tri]).tolist():
            r = obj[tri] == oi
            out[r] = fl[oi][k][poly[tri[r]]]
        return out

    pg, ps, pt_, px, pk = col(p, 0), col(p, 1), col(p, 2), col(p, 3), col(p, 4)
    gs = col(g, 1)
    facing = (n[p] * n[g]).sum(axis=1) > 0.9
    under = ~gs & (s < 0) & facing & ~pg & ~pt_ & ~px & ~pk             # under an opaque card, looking at it
    sheet = ps & ~pg & (d < Z_NEAR) & ~px & ~pk                         # clear glass on a glowing face
    cand = np.flatnonzero(under | sheet)
    if not len(cand):
        return 0
    rc = z["bvh"].ray_cast
    V, T = z["V"], z["T"]
    key = obj[p[cand]] * (len(poly) + 1) + poly[p[cand]]
    uq, first = np.unique(key, return_index=True)
    # triangles of every polygon in question
    tkey = obj * (len(poly) + 1) + poly
    order = np.argsort(tkey, kind="stable")
    lo, hi = np.searchsorted(tkey[order], uq), np.searchsorted(tkey[order], uq, side="right")
    kill = {}
    for q, c0, a_, b_ in zip(uq.tolist(), first.tolist(), lo.tolist(), hi.tolist()):
        r = cand[c0]
        u = n[g[r]] * (-float(s[r]))                                    # from the partner towards the glowing face
        opaque = bool(under[r])
        reach = (_im.Z_FAR if opaque else 2.0 * Z_NEAR) + 1e-4
        ok = True
        for t in order[a_:b_].tolist():
            A, B, C = V[T[t, 0]], V[T[t, 1]], V[T[t, 2]]
            ctr = (A + B + C) / 3.0
            for P_ in (ctr, ctr * 0.4 + A * 0.6, ctr * 0.4 + B * 0.6, ctr * 0.4 + C * 0.6,
                       ctr * 0.03 + A * 0.97, ctr * 0.03 + B * 0.97, ctr * 0.03 + C * 0.97):
                h = rc((P_ + u * 2e-5).tolist(), u.tolist(), reach)
                if h[0] is None:
                    ok = False
                    break
                oj = int(obj[h[2]])
                if oj not in fl:
                    mj = S.an(S.objs[oj])[0]
                    fl[oj] = (mj.zglow[mj.mi], mj.see[mj.mi], mj.two[mj.mi], mj.tx() > 0, mj.rk() == 1)
                pj = int(poly[h[2]])
                if not fl[oj][0][pj] or (opaque and fl[oj][1][pj]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            kill.setdefault(int(obj[p[r]]), []).append(int(poly[p[r]]))
    gone = 0
    for oi, pp in kill.items():
        o = S.objs[oi]
        m = S.an(o)[0]
        pm = np.zeros(len(m.lt), dtype=bool)
        pm[pp] = True
        gone += int((m.lt - 2)[pm].sum())
        _delete_polys(o, pm)
        S.dirty(o)
    return gone


def _zpush(S, z, reach, stuck=None, why=None):
    """Push away what still lies within Z_NEAR of a glowing face. The glowing face stays where it is; the corners of
    the face met move along its normal until Z_GAP lies between the two, to the side most of that face is on (a
    mullion the collapse laid into the pane comes to lie in front of it, a wall that bulged into a room card goes
    back behind it). A corner further than `reach` on the other side stays - that face really runs through the
    glowing one - and so do corners a glowing face of the same piece uses (unless the face met glows itself).
    Lettering moves as a whole group, never corner by corner.
    stuck: list that gets a line per face nothing could be done for, why: {reason: [faces, m2]}.
    -> corners moved, faces left as they are"""
    near = np.flatnonzero(z["d"] < Z_NEAR)
    if not len(near):
        return 0, 0
    stuck = [] if stuck is None else stuck
    why = {} if why is None else why
    g, p, w, obj, poly, T, voff = z["g"], z["p"], z["w"], z["obj"], z["poly"], z["T"], z["voff"]
    n = z["n"][g]
    dom = np.argmax(np.abs(n), axis=1)
    sg = np.where(n[np.arange(len(n)), dom] < 0, -1.0, 1.0)             # one axis for both faces of a sheet
    side = z["s"] * sg                                                  # where the partner is along that axis
    tab = {}

    def table(oi):
        if oi not in tab:
            m = S.an(S.objs[oi])[0]
            gl = m.zglow[m.mi]
            gv = np.zeros(len(m.co), dtype=bool)
            gv[m.lv[np.repeat(gl, m.lt)]] = True
            tab[oi] = {"m": m, "co": m.co.copy(), "gl": gl, "gv": gv, "tx": m.tx(), "n": 0}
        return tab[oi]

    def note(rows, what):
        e = why.setdefault(what, [0, 0.0])
        e[0] += 1
        e[1] = round(e[1] + float(w[rows].sum()), 4)
        if len(stuck) < 6:
            r = rows[0]
            tg, tp = table(int(obj[g[r]])), table(int(obj[p[r]]))
            c = z["V"][T[g[r]]].mean(axis=0)
            stuck.append(f"{what}: {tg['m'].names[tg['m'].mi[poly[g[r]]]]} / {tp['m'].names[tp['m'].mi[poly[p[r]]]]} at "
                         f"{[round(float(v), 2) for v in c]}, {float(w[rows].sum()):.4f} m2")

    key = obj[p[near]] * (len(poly) + 1) + poly[p[near]]
    order = np.argsort(key, kind="stable")
    cut = np.flatnonzero(np.diff(key[order])) + 1
    moved = left = 0
    for grp in np.split(near[order], cut):
        oi, pi = int(obj[p[grp[0]]]), int(poly[p[grp[0]]])
        tp = table(oi)
        m = tp["m"]
        vs = m.lv[m.ls[pi]:m.ls[pi] + m.lt[pi]]
        glows = bool(tp["gl"][pi])
        text = int(tp["tx"][pi]) // 4 if tp["tx"][pi] > 0 else 0
        front = 1.0 if float((w[grp] * side[grp]).sum()) >= 0 else -1.0
        did = False
        deep = locked = False
        for r in grp[np.unique(g[grp], return_index=True)[1]].tolist():
            tg = table(int(obj[g[r]]))
            c = tg["co"][T[g[r]] - voff[int(obj[g[r]])]].mean(axis=0)
            u = n[r] * (front * sg[r])                                  # away from the glowing face, the partner's side
            sx = (tp["co"][vs] - c) @ u
            want = sx < Z_GAP - 1e-6
            if not want.any():
                continue
            if (sx[want] <= Z_GAP - reach).any():
                deep = True
            want &= sx > Z_GAP - reach
            if text:
                if want.any():
                    mv = np.unique(m.lv[np.repeat(tp["tx"] // 4 == text, m.lt)])
                    tp["co"][mv] += u * float((Z_GAP - sx[want]).max())
                    tp["n"] += len(mv)
                    did = True
                continue
            if not glows:
                lk = tp["gv"][vs]
                locked = locked or bool((want & lk).any())
                want &= ~lk
            if want.any():
                tp["co"][vs[want]] += u[None, :] * (Z_GAP - sx[want])[:, None]
                tp["n"] += int(want.sum())
                did = True
        if not did and (deep or locked):                # else: a push of this call has cleared it already
            left += 1
            note(grp, "it runs through the glowing face" if deep else "it shares corners with the glowing face")
    for oi, t in tab.items():
        if not t["n"]:
            continue
        me = S.objs[oi].data
        me.vertices.foreach_set("co", t["co"].astype(np.float32).ravel())
        me.update()
        S.dirty(S.objs[oi])
        moved += t["n"]
    return moved, left


def _zfix(S, zs, log, step):
    """see above -> the glowing area that still has a partner under Z_NEAR (0.0 when nothing glows)"""
    if zs is None:
        return 0.0
    gone = moved = left = undone = 0
    before = after = None
    stuck, why = [], {}
    saved = None                                        # before the last push: near area, corners moved, coordinates
    for rnd in range(Z_ROUNDS + 1):
        z = _zscan(S)
        after = 0.0 if z is None else float(z["w"][z["d"] < Z_NEAR].sum())
        if before is None:
            before = after
        if saved is not None and after > saved[0] + 1e-5:
            # corners other faces share were pushed, and those faces came through a glowing one: as it was
            for o, me, co in saved[2]:
                if o.data is me and len(me.vertices) * 3 == len(co):
                    me.vertices.foreach_set("co", co)
                    me.update()
                    S.dirty(o)
            moved -= saved[1]
            undone += saved[1]
            after = saved[0]
            break
        if z is None or rnd == Z_ROUNDS:
            break
        if not moved:
            n = _zhidden(S, z)
            if n:
                gone += n
                continue
        if after <= 1e-4:
            break
        keep = []
        for o in S.objs:
            co = np.empty(len(o.data.vertices) * 3, dtype=np.float32)
            o.data.vertices.foreach_get("co", co)
            keep.append((o, o.data, co))
        stuck, why = [], {}
        m_, left = _zpush(S, z, Z_SHIFT[S.level], stuck, why)
        if not m_:
            break
        saved = (after, m_, keep)
        moved += m_
    if gone or moved or undone or after > 1e-4:
        log.setdefault("z", []).append([step, round(before or 0.0, 4), round(after, 4), gone, moved, left])
        if undone:
            log["z_undone"] = log.get("z_undone", 0) + undone
        if after > 1e-4 and stuck:
            log["z_stuck"] = stuck
            log["z_why"] = why
    return after


def _panes(S, cell):
    """Glowing panes laid as tiles in one plane - the quarrels of a stained-glass window: thousands of loose quads
    the collapse can take nothing from, and more triangles than a whole lod 2 has. Where rectangular glowing quads
    of one material library tile a plane, every `cell` x `cell` block of them (whole tiles: the block is as wide
    as the tiles that fit into `cell`) becomes one quad, if its tiles fill their common rectangle. The quad gets the
    tint of one of its tiles: the one that has had the least of its share so far, so that the tints keep their part
    of the window. Nothing moves out of the plane or beyond the outline. -> triangles saved"""
    saved = 0
    for o in S.objs:
        m, isl = S.an(o)
        if isl is None or len(m.lt) < 8:
            continue
        pisl = isl["pisl"]
        npi = np.bincount(pisl, minlength=isl["n"])
        cand = np.flatnonzero((m.lt == 4) & m.glow[m.mi] & (npi[pisl] == 1) & (m.tx() == 0) & (m.rk() == 0))
        if len(cand) < 8:
            continue
        C = m.co[m.lv[m.ls[cand][:, None] + np.arange(4)[None]]]         # (n, 4, 3)
        e1, e2 = C[:, 1] - C[:, 0], C[:, 3] - C[:, 0]
        l1, l2 = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
        okq = (l1 > 1e-4) & (l2 > 1e-4)
        okq &= np.linalg.norm(C[:, 0] - C[:, 1] + C[:, 2] - C[:, 3], axis=1) < 1e-4
        with np.errstate(invalid="ignore", divide="ignore"):
            okq &= np.abs((e1 * e2).sum(axis=1)) < 0.01 * l1 * l2
        cand, C, e1, e2 = cand[okq], C[okq], e1[okq], e2[okq]
        if len(cand) < 8:
            continue
        nr = np.cross(e1, e2)
        nr /= np.linalg.norm(nr, axis=1)[:, None]
        libs = np.array(m.libs, dtype=object)[m.mi[cand]]
        key = {}
        for q, (lb, a, d) in enumerate(zip(libs.tolist(), np.round(nr * 500).astype(np.int64).tolist(),
                                           np.round((nr * C[:, 0]).sum(axis=1) * 5000).astype(np.int64).tolist())):
            key.setdefault((lb, tuple(a), d), []).append(q)
        plan = []                                       # (member polygons, corners of the new quad, example polygon)
        for rows in key.values():
            if len(rows) < 8:
                continue
            rows = np.array(rows)
            n = nr[rows].mean(axis=0)
            n /= np.linalg.norm(n)
            u = e1[rows[0]] / np.linalg.norm(e1[rows[0]])
            v = np.cross(n, u)
            al = np.abs(e1[rows] @ u) / np.linalg.norm(e1[rows], axis=1)
            rows = rows[(al > 0.9995) | (al < 0.03)]
            if len(rows) < 8:
                continue
            org = C[rows[0], 0]
            s_ = (C[rows] - org) @ u
            t_ = (C[rows] - org) @ v
            lo = np.stack([s_.min(axis=1), t_.min(axis=1)], axis=1)
            hi = np.stack([s_.max(axis=1), t_.max(axis=1)], axis=1)
            pitch = np.median(hi - lo, axis=0)
            nc = np.floor(cell / np.maximum(pitch, 1e-6) + 1e-6)
            if nc.min() < 1 or nc.max() < 2 or nc.prod() < 2:
                continue
            g0 = lo.min(axis=0)
            gi = np.floor((0.5 * (lo + hi) - g0) / (nc * pitch) + 1e-6).astype(np.int64)
            ck = gi[:, 1] * (int(gi[:, 0].max()) + 2) + gi[:, 0]
            order = np.argsort(ck, kind="stable")
            cut = np.flatnonzero(np.diff(ck[order])) + 1
            area = (hi - lo).prod(axis=1)
            mi = m.mi[cand[rows]]
            owed = np.zeros(len(m.libs))
            given = np.zeros(len(m.libs))
            for grp in np.split(order, cut):
                np.add.at(owed, mi[grp], area[grp])
                if len(grp) < 2:
                    given[mi[grp[0]]] += area[grp[0]]
                    continue
                a, b = lo[grp].min(axis=0), hi[grp].max(axis=0)
                full = float((b - a).prod())
                if not (0.98 * full <= float(area[grp].sum()) <= 1.02 * full):
                    np.add.at(given, mi[grp], area[grp])
                    continue
                ms = np.unique(mi[grp])
                pick = int(ms[np.argmax((owed - given)[ms])])
                given[pick] += full
                ex = grp[mi[grp] == pick]
                ex = int(ex[np.argmax(area[ex])])
                h = float(((C[rows[grp], 0] - org) @ n).mean())
                quad = [org + u * x + v * y + n * h for x, y in ((a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1]))]
                plan.append((cand[rows[grp]], quad, int(cand[rows[ex]]), org, u, v))
        if not plan:
            continue
        me = o.data
        bm = bmesh.new()
        bm.from_mesh(me)
        faces = list(bm.faces)
        uvs = bm.loops.layers.uv.values()
        dead = []
        for mem, quad, exi, org, u, v in plan:
            ex = faces[exi]
            vs = [bm.verts.new(c.tolist(), ex.verts[0]) for c in quad]
            f = bm.faces.new(vs, ex)
            exl = list(ex.loops)
            for l in f.loops:
                l.copy_from(exl[0])
            if uvs:
                P = np.array([[(np.array(l.vert.co) - org) @ u, (np.array(l.vert.co) - org) @ v, 1.0] for l in exl])
                Q = np.array([[(c - org) @ u, (c - org) @ v, 1.0] for c in quad])
                for lay in uvs:
                    X = np.linalg.lstsq(P, np.array([list(l[lay].uv) for l in exl]), rcond=None)[0]
                    for l, w in zip(f.loops, (Q @ X).tolist()):
                        l[lay].uv = w
            dead += [faces[int(i)] for i in mem]
            saved += 2 * len(mem) - 2
        bmesh.ops.delete(bm, geom=dead, context='FACES')
        bm.to_mesh(me)
        bm.free()
        me.update()
        S.dirty(o)
    return saved


def _finish(S):
    out = []
    for o in S.objs:
        me = o.data
        if len(me.polygons) == 0:
            bpy.data.objects.remove(o, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
            continue
        a = me.attributes.get("rk")
        if a is not None:
            me.attributes.remove(a)
        if "tsq_nofield" in o:
            del o["tsq_nofield"]
        out.append(o)
    S.objs = out
    return out


# ----------------------------------------------------------------------------
# Lettering (the faces tk.geo.text() tagged): an identity class of its own. It is never welded, patched or
# collapsed; what may happen to it keeps the letters' shapes:
#   * faces nobody sees go (backs bedded against a board) - never part of the letter shapes of a line that shows
#   * at lod 2, and wherever the letters are under 3 px high: the sides go, the flat fronts stay
#   * engraved lettering under 3 px: the cut is closed (floor lifted to the rim) and becomes plain surface
#   * raised lettering under 3 px goes whole, smallest first, when the level cannot fit otherwise
# tk.lodimage.text_measure() checks the outcome letter by letter (tk.final_levels.finalize_level).
# ----------------------------------------------------------------------------

def _free(S, o):
    """the polygons of o outside lettering islands"""
    m, isl = S.an(o)
    if isl is None:
        return np.zeros(0, dtype=bool)
    return ~(isl["text"] > 0)[isl["pisl"]]


def _text_groups(S, level):
    """-> {text number: {"h": letter height, "shape": 1 raised | 3 engraved, "small": under 3 px at this level}};
    tags without a flat face to judge them by are cleared, and islands that are mostly something else are cut so
    that the lettering (and the ring of faces round it) is an island of its own"""
    if NOTEXT:
        return {}
    from .final_levels import text_tris
    T = text_tris(S.objs)
    if not len(T["g"]):
        return {}
    info = {G["id"]: {"h": G["h"], "shape": G["shape"], "small": bool(_im.text_small(level, G["h"])), "tris": G["tris"]}
            for G in _im.text_prepare(T)}
    for o in S.objs:
        m, isl = S.an(o)
        if isl is None or not m.tx().any():
            continue
        tx = m.tx()
        lost = (tx > 0) & ~np.isin(tx // 4, list(info))
        if lost.any():
            v = tx.copy()
            v[lost] = 0
            o.data.attributes["tsq_text"].data.foreach_set("value", v)
            S.dirty(o)
            m, isl = S.an(o)
            tx = m.tx()
        mixed = (isl["text"] > 0) & (isl["text"] < TEXT_SHARE * isl["tris"])
        if not mixed.any():
            continue
        lp = np.repeat(np.arange(len(m.lt)), m.lt)
        vm = np.zeros(len(m.co), dtype=bool)
        vm[m.lv[(tx > 0)[lp]]] = True
        ring = (np.bincount(lp, weights=vm[m.lv], minlength=len(m.lt)) > 0) & mixed[isl["pisl"]]
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.faces.ensure_lookup_table()
        rs = ring.tolist()
        es = [e for e in bm.edges if len(e.link_faces) >= 2 and len({rs[f.index] for f in e.link_faces}) == 2]
        if es:
            bmesh.ops.split_edges(bm, edges=es)
            bm.to_mesh(o.data)
            o.data.update()
            S.dirty(o)
        bm.free()
    return info


def _text_hidden(S, marks, info):
    """The hidden-face marks, corrected for lettering: the faces that carry the letter shapes stay - unless the
    lettering is under 3 px here and none of them shows. Engraved lettering that small is closed later
    (_text_step) and stays whole until then."""
    if not info:
        return marks
    stat = {}
    for o, pm in zip(S.objs, marks):
        tx = S.an(o)[0].tx() if len(pm) else np.zeros(0, dtype=np.int32)
        if not tx.any():
            continue
        g, k = tx // 4, tx % 4
        for gid in np.unique(g[tx > 0]).tolist():
            sh = (g == gid) & (k == info[gid]["shape"])
            st = stat.setdefault(gid, [0, 0])
            st[0] += int(sh.sum())
            st[1] += int((sh & pm).sum())
    for o, pm in zip(S.objs, marks):
        tx = S.an(o)[0].tx() if len(pm) else np.zeros(0, dtype=np.int32)
        if not tx.any():
            continue
        g, k = tx // 4, tx % 4
        for gid in np.unique(g[tx > 0]).tolist():
            ti = info[gid]
            if ti["small"] and ti["shape"] == 3:
                pm[g == gid] = False
            elif not (ti["small"] and stat[gid][1] == stat[gid][0]):
                pm[(g == gid) & (k == ti["shape"])] = False
    return marks


def _text_fill(o, gids, keep=False):
    """Engraved lettering nobody can read at this level: the floor of every letter is lifted to the rim of its cut,
    the walls go, and the faces become plain surface.
    keep: lettering that still reads, made an inlay - the same, but the floors stay lettering (they keep the tag and
    so the shape of every letter), and only where they are of another material than the face they are cut into
    (a letter of the same material shows by the shade of its cut alone and stays cut). -> text numbers closed"""
    me = o.data
    bm = bmesh.new()
    bm.from_mesh(me)
    lay = bm.faces.layers.int.get("tsq_text")
    done = []
    for gid in gids:
        walls = [f for f in bm.faces if f[lay] == 4 * gid + 2]
        floors = [f for f in bm.faces if f[lay] == 4 * gid + 3]
        if not walls or not floors:
            continue
        fv = {v for f in floors for v in f.verts}
        move = {}
        for f in walls:
            for e in f.edges:
                va, vb = e.verts
                if (va in fv) != (vb in fv):
                    lo, hi = (va, vb) if va in fv else (vb, va)
                    d = (hi.co - lo.co).length
                    if lo not in move or d < move[lo][1]:
                        move[lo] = (hi.co.copy(), d)
        rim = [v for v in fv if any(len([f for f in e.link_faces if f[lay] == 4 * gid + 3]) < 2 for e in v.link_edges)]
        if not move or sum(1 for v in rim if v in move) < 0.9 * len(rim):
            continue                                    # not a plain cut: left as it is
        if keep:
            wv = {v for f in walls for v in f.verts} - fv
            face_m = {f.material_index for v in wv for f in v.link_faces if f[lay] // 4 != gid}
            if not face_m or face_m & {f.material_index for f in floors}:
                continue
        mean = sum(((c - v.co) for v, (c, _) in move.items()), Vector((0.0, 0.0, 0.0))) / len(move)
        touched = set()
        for v in fv:
            v.co = move[v][0] if v in move else v.co + mean
            touched.add(v)
        for f in walls:
            touched.update(f.verts)
        if not keep:
            for f in floors:
                f[lay] = 0
        bmesh.ops.delete(bm, geom=walls, context='FACES')
        vs = [v for v in touched if v.is_valid]
        if vs:
            bmesh.ops.remove_doubles(bm, verts=vs, dist=2e-5)
        done.append(gid)
    if done:
        bm.to_mesh(me)
        me.update()
    bm.free()
    return done


def _text_outline(o, gids, ring=False):
    """limited planar dissolve of letters that are down to their flat faces: the outline loses the vertices where
    it turns by less than ANGLE_T"""
    me = o.data
    bm = bmesh.new()
    bm.from_mesh(me)
    lay = bm.faces.layers.int.get("tsq_text")
    want = set(gids)
    faces = [f for f in bm.faces if f[lay] > 0 and f[lay] // 4 in want]
    if faces:
        own = set(faces)
        if ring:
            # an inlay: the flat faces round the letters lose the same vertices
            for f in faces:
                for v in f.verts:
                    for x in v.link_faces:
                        if x not in own and x[lay] == 0 and x.normal.dot(f.normal) > 0.9995:
                            own.add(x)
            for e in bm.edges:
                lf = e.link_faces
                if len(lf) == 2 and (lf[0] in own) != (lf[1] in own):
                    e.seam = True
        else:
            for e in bm.edges:
                lf = e.link_faces
                if len(lf) == 2 and lf[0][lay] != lf[1][lay]:
                    e.seam = True
        vs = list({v for f in own for v in f.verts if all(x in own for x in v.link_faces)})
        es = list({e for f in own for e in f.edges if all(x in own for x in e.link_faces)})
        bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(ANGLE_T), use_dissolve_boundaries=False, verts=vs, edges=es,
                                 delimit={'MATERIAL', 'SEAM'})
        big = [f for f in bm.faces if len(f.verts) > 4]
        if big:
            bmesh.ops.triangulate(bm, faces=big, quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.to_mesh(me)
        me.update()
    bm.free()


def _text_step(S, level, info, log):
    """the shape-preserving cheapening of lettering (see the head of this section) -> triangles removed"""
    if not info:
        return 0
    t0 = S.tris()
    fronts, filled, inlaid = set(), set(), set()
    for o in list(S.objs):
        m, isl = S.an(o)
        if isl is None or not m.tx().any():
            continue
        tx = m.tx()
        g, k = tx // 4, tx % 4
        here = np.unique(g[tx > 0]).tolist()
        fill = [gid for gid in here if info[gid]["shape"] == 3 and info[gid]["small"]]
        if fill:
            got = _text_fill(o, fill)
            left = [gid for gid in fill if gid not in got]
            filled.update(got)
            if got or left:
                S.dirty(o)
                if left:                                # could not be closed: plain geometry from here on
                    m = S.an(o)[0]
                    v = m.tx().copy()
                    v[np.isin(v // 4, left)] = 0
                    o.data.attributes["tsq_text"].data.foreach_set("value", v)
                    S.dirty(o)
                    log.setdefault("text_plain", []).extend(left)
                m, isl = S.an(o)
                if isl is None:
                    continue
                tx = m.tx()
                g, k = tx // 4, tx % 4
        lay_ = [gid for gid in here if info[gid]["shape"] == 3 and not info[gid]["small"] and level >= 2]
        if lay_:
            got = _text_fill(o, lay_, keep=True)
            if got:
                inlaid.update(got)
                S.dirty(o)
                _text_outline(o, got, ring=True)
                S.dirty(o)
                m, isl = S.an(o)
                if isl is None:
                    continue
                tx = m.tx()
                g, k = tx // 4, tx % 4
        flat = [gid for gid in here if info[gid]["shape"] == 1 and (info[gid]["small"] or level >= 2)]
        if not flat:
            continue
        sel = np.isin(g, flat) & (tx > 0)
        kill = sel & (k == 2)
        # letters that carry untagged faces of their own (a chamfer, sides extruded by the asset): those go too, when
        # the island is the letter and nothing more
        lp = np.repeat(np.arange(len(m.lt)), m.lt)
        for i in np.unique(isl["pisl"][sel]).tolist():
            pi = isl["pisl"] == i
            un = pi & (tx == 0)
            if not un.any() or isl["text"][i] < TEXT_SHARE * isl["tris"][i]:
                continue
            co = m.co[m.lv[(pi & (tx > 0))[lp]]]
            if isl["d"][i] <= 1.3 * float(np.linalg.norm(co.max(0) - co.min(0))) + 1e-6:
                kill |= un
        if kill.any():
            fronts.update(flat)
            _delete_polys(o, kill)
            S.dirty(o)
        _text_outline(o, flat)
        S.dirty(o)
    if fronts:
        log["text_fronts"] = sorted(fronts)
    if filled:
        log["text_filled"] = sorted(filled)
    if inlaid:
        log["text_inlaid"] = sorted(inlaid)
    return t0 - S.tris()


def _text_drop(S, T, info, over, probe, base, level, log):
    """Raised lettering under 3 px goes whole, smallest letters first, until `over` triangles are gone - a line of it
    is never thinned out. Stops short when that changes the brightness of the picture or the outline.
    -> triangles removed"""
    small = sorted((ti["h"], gid) for gid, ti in info.items() if ti["small"] and ti["shape"] == 1)
    rows = {gid: np.flatnonzero((T["kind"] == K_TEXT) & (T["tg"] == gid)) for _, gid in small}
    order = [gid for _, gid in small if len(rows[gid])]
    take, n = [], 0
    for gid in order:
        if n >= over:
            break
        take.append(gid)
        n += int(T["tris"][rows[gid]].sum())
    while take:
        drop = np.zeros(len(T["kind"]), dtype=bool)
        for gid in take:
            drop[rows[gid]] = True
        kept = _protect_outline(S, T, drop.copy(), S.tol)
        part = [gid for gid in take if not kept[rows[gid]].all()]
        if part:                                        # a line that would only go in part stays whole
            take = [gid for gid in take if gid not in part]
            continue
        saved = S.save([S.objs[oi] for oi, a, b in T["span"] if drop[a:b].any()])
        gone = _apply_drop(S, T, drop)
        m_ = probe.measure(*S.soup())
        if abs(m_["lum"] - base["lum"]) <= LUMTOL[level] and not S.sides(S.tol) and not S.empty_parts():
            S.forget(saved)
            base["lum"] = m_["lum"]
            log["text_dropped"] = sorted(take)
            return gone
        S.put_back(saved)
        log.setdefault("rollback", []).append(f"text: brightness {m_['lum']:.3f} without {len(take)} lines of small lettering")
        take = take[:len(take) // 2]
    return 0


# ----------------------------------------------------------------------------
# The reduction
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# What touched in lod 0 keeps touching
# ----------------------------------------------------------------------------

MEND_ISL = 96                   # a loose piece of lod 0 of up to this many triangles is put back whole
MEND_MAX = 60                   # loose pieces looked at per round


class _Touch:
    """The touch test of tk.lodimage (voxels / loose: the gate of the chooser) run on the reducer's own scene, and
    its repair. A step may take away what held two things together - the hanger of a lantern nobody sees from the
    test directions, the stem of a table under the size limit, the foot of a spire the collapse let go of. mend()
    finds the pieces that so came loose, the shortest way from each back to the rest through the cells lod 0 fills
    and the scene no longer does, and puts the faces of lod 0 on that way back (the whole loose piece of lod 0 when
    it is small). What is put back is marked like a prism: no later step touches it."""

    def __init__(self, S):
        self.org = S.b0[0]
        pr = self.prims(S)
        self.c = _im.vox_cell(S.level, S.diag, pr)
        self.K0, self.L0 = _im.voxels(pr, self.c, self.org)
        self.src = []
        for o in S.objs:
            c = o.copy()
            c.data = o.data.copy()
            c.use_fake_user = True
            self.src.append(c)
        self.used = [np.zeros(len(c.data.polygons), dtype=bool) for c in self.src]
        self._m = {}

    def free(self):
        for c in self.src:
            try:
                me = c.data
                bpy.data.objects.remove(c, do_unlink=True)
                if me.users == 0:
                    bpy.data.meshes.remove(me)
            except ReferenceError:
                pass
        self.src = []

    @staticmethod
    def prims(S):
        out = []
        for o in S.objs:
            m = S.an(o)[0]
            if not len(m.lt):
                continue
            T, tp = m.tris()
            pl = m.plant[m.mi[tp]]
            out.append({"P": m.co, "I": T[~pl], "lib": ""})
            if pl.any():
                out.append({"P": m.co, "I": T[pl], "lib": PLANT[0]})
        return out

    def read(self, S):
        """-> cells of the scene, their groups, the loose pieces (cell indices), largest first"""
        K1, L1 = _im.voxels(self.prims(S), self.c, self.org)
        return K1, L1, _im.loose_pieces(self.K0, self.L0, K1, L1)

    def count(self, S):
        return len(self.read(S)[2])

    def _put(self, S, cells):
        n = 0
        for i, c in enumerate(self.src):
            if i not in self._m:
                m = _M(c)
                self._m[i] = (m, _islands(m) if len(m.lt) else None)
            m, isl = self._m[i]
            if isl is None:
                continue
            T, tp = m.tris()
            hit = _im.tris_in_cells(m.co, T, self.c, self.org, cells)
            if not hit.any():
                continue
            pm = np.zeros(len(m.lt), dtype=bool)
            pm[tp[hit]] = True
            pm &= ~m.plant[m.mi]
            whole = np.zeros(isl["n"], dtype=bool)
            whole[isl["pisl"][pm]] = True
            whole &= isl["tris"] <= MEND_ISL
            pm |= whole[isl["pisl"]]
            pm &= ~self.used[i]
            if not pm.any():
                continue
            b = c.copy()
            b.data = c.data.copy()
            b.use_fake_user = False
            _link(b)
            _delete_polys(b, ~pm)
            _set_rk(b, np.ones(len(b.data.polygons), dtype=bool), 1)
            S.objs.append(b)
            self.used[i] |= pm
            n += int((m.lt - 2)[pm].sum())
        return n

    def mend(self, S):
        """-> triangles put back, loose pieces left"""
        added = 0
        left = 0
        for rnd in range(4):
            K1, L1, pcs = self.read(S)
            left = len(pcs)
            if not pcs:
                break
            M = self.K0[~_im._in(K1, self.K0)]
            B = [b for b in (_im.bridge(M, K1, g) for g in pcs[:MEND_MAX]) if b is not None]
            if not B:
                break
            B = np.unique(np.concatenate(B))
            for _ in range(min(rnd, 2)):                # what was put back did not reach: wider
                B = np.unique(np.concatenate([B] + [B + o for o in _im._VOFF]))
                B = B[_im._in(M, B)]
            n = self._put(S, B)
            if not n:
                break
            added += n
        else:
            left = self.count(S)
        return added, left


# ----------------------------------------------------------------------------
# No holes (docs/LOD.md): nobody on the street looks through the level where lod 0 is closed
# ----------------------------------------------------------------------------

def _holes_start(S, cache, touch, fixed=None):
    """Lod 0 as the viewers on the street see it (tk.holes: single-sided, from eye height, close), kept between
    the chains of a level. fixed: the identity parts - not reduced, but they hide what is behind them.
    -> {"H": [tk.holes.Holes per viewer set], "own" / "poly": object and polygon of each lod 0 triangle,
        "src": the copies of the lod 0 objects (those of _Touch), "seen": per object, the polygons a viewer sees,
        "used": the polygons _close() put back, "added": their triangles}"""
    fixed = [o for o in (fixed or []) if o.type == 'MESH' and len(o.data.polygons)]
    hc = ((cache or {}).get("holes") or {}).get(S.level)
    sig = [len(o.data.polygons) for o in S.objs]
    if hc is None or hc["sig"] != sig or hc["src"] is not touch.src:
        s, Hs = _ho.of_objs(S.objs + fixed, S.mount, bounds=S.b0)
        hc = {"H": Hs, "own": s["own"], "poly": s["poly"], "src": touch.src, "sig": sig}
        rows = _ho.seen_rows(Hs)
        hc["seen"] = []
        for i, n in enumerate(sig):
            pm = np.zeros(n, dtype=bool)
            pm[s["poly"][rows[s["own"][rows] == i]]] = True
            hc["seen"].append(pm)
        if cache is not None:
            cache.setdefault("holes", {})[S.level] = hc
    hc["fixed"] = fixed
    hc["used"] = [np.zeros(n, dtype=bool) for n in sig]
    hc["added"] = 0
    _grave(hc, None)
    return hc


BEHIND = 0.3                    # a face taken in step 1 closes a hole when it lies this close (m) behind what lod 0
                                # shows there (or 2.5 x the size limit)


def _grave(hc, objs):
    """the faces step 1 took as unseen, kept (unlinked copies) for _behind(); None: free them"""
    g = hc.pop("grave", None) if hc is not None else None
    for o in (g or {}).get("objs", []):
        try:
            me = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
        except ReferenceError:
            pass
    if hc is None or not objs:
        return
    s = _ho.obj_soup(objs)
    hc["grave"] = {"objs": objs, "own": s["own"], "poly": s["poly"], "soup": _ho._Soup(s["V"], s["T"], s["two"], s["thru"]),
                   "used": [np.zeros(len(o.data.polygons), dtype=bool) for o in objs]}


def _behind(S, hc, E, P, mark):
    """Hole samples (eyes E, the points P lod 0 shows them): where a face that step 1 took as unseen lies right
    behind the point, facing the viewer - the wall behind a moulding, the slab behind the panels of a door - it is
    put back: what hid it is gone or thinner now, so it is seen, and it is the cheapest thing that closes the view.
    mark: the faces are marked like a prism (the end of the chain); else they go on through the steps like the rest.
    -> (per sample: closed, triangles put back)"""
    g = hc.get("grave") if hc is not None else None
    done = np.zeros(len(P), dtype=bool)
    if not g or not len(P):
        return done, 0
    D = np.asarray(P, dtype=np.float64) - E
    t0 = np.linalg.norm(D, axis=1)
    D = D / np.maximum(t0, 1e-12)[:, None]
    t, tri, _b = g["soup"].cast(P - D * 2e-3, D, max(BEHIND, 2.5 * S.ms))
    done = tri >= 0
    n = 0
    own = g["own"][tri[done]]
    pol = g["poly"][tri[done]]
    for i in np.unique(own).tolist():
        c = g["objs"][i]
        pm = np.zeros(len(c.data.polygons), dtype=bool)
        pm[pol[own == i]] = True
        pm &= ~g["used"][i]                 # (once: a step taken back and tried again asks for the same faces)
        if not pm.any():
            continue
        g["used"][i] |= pm
        b = c.copy()
        b.data = c.data.copy()
        b.use_fake_user = False
        _link(b)
        _delete_polys(b, ~pm)
        if mark:
            _set_rk(b, np.ones(len(b.data.polygons), dtype=bool), 1)
            b["tsq_hole"] = 1
        n += _ntris(b.data)
        S.objs.append(b)
    return done, n


def _holes_read(S, hc, detail=False, box=None):
    return _ho.measure_objs(hc["H"][0], S.objs + hc["fixed"], S.level, detail=detail, box=box)


def _close(S, hc, log, step="end"):
    """The steps before this one take faces nobody was thought to see (step 1: seen from fewer than NEED of the test
    directions), swap a moulding for a slimmer prism, drop the small pieces, weld and collapse - and each can leave
    a slit a viewer at the foot of the wall looks through: single-sided, the wall behind the moulding went in step
    1, and behind that is the sky. Here the level is looked at from the street (tk.holes) and, wherever a ray that
    shows a front face in lod 0 now shows none, the faces lod 0 shows there are put back as they were, marked like
    a prism so that nothing later touches them. -> triangles put back"""
    if hc is None or not HOLES:
        return 0
    n0 = len(hc["src"])

    def put(rows):
        n = 0
        own = hc["own"][rows]
        for i in np.unique(own).tolist():
            if i >= n0:
                continue                    # an identity part: it stands as it is
            c = hc["src"][i]
            pm = np.zeros(len(c.data.polygons), dtype=bool)
            pm[hc["poly"][rows[own == i]]] = True
            pm &= ~hc["used"][i]
            if not pm.any():
                continue
            b = c.copy()
            b.data = c.data.copy()
            b.use_fake_user = False
            _link(b)
            _delete_polys(b, ~pm)
            _set_rk(b, np.ones(len(b.data.polygons), dtype=bool), 1)
            b["tsq_hole"] = 1
            n += _ntris(b.data)
            S.objs.append(b)
            hc["used"][i] |= pm
        return n

    nb = [0]

    def behind(E, P):
        done, n = _behind(S, hc, E, P, True)
        nb[0] += n
        return done, n

    at = len(S.objs)
    first, added = _ho.close(hc["H"], S.level, lambda: S.objs + hc["fixed"], put, behind)
    if added:
        new = {i: o for i, o in enumerate(S.objs[at:]) if o.get("tsq_hole")}

        def swap(i, me):
            o = new[i]
            o.data = me
            _set_rk(o, np.ones(len(me.polygons), dtype=bool), 1)
            S.dirty(o)

        was = [(i, o.data) for i, o in new.items()]
        saved = _ho.thin(hc["H"], S.level, lambda: S.objs + hc["fixed"], was, swap)
        for i, me in was:
            if new[i].data is not me and me.users == 0:
                bpy.data.meshes.remove(me)
        if saved:
            log["holes_thinned"] = log.get("holes_thinned", 0) + saved
            added -= saved
    hc["added"] += added
    if first is not None and (first["count"] or added):
        log.setdefault("holes", []).append({"step": step, "before": first["holes"], "back": first["back"], "none": first["none"],
                                            "tris": added, "behind": nb[0], "where": first.get("where", [])[:3]})
        if TRACE:
            print(f"TSQ_TRACE lod {S.level} _close({step}): " + json.dumps(log["holes"][-1]))
    log["closed"] = hc["added"]
    return added


def _force(S, log, limit, relaxed=False):
    """Still over: collapse everything that neither glows nor is lettering towards `limit` triangles, loose piece by
    loose piece - which stops where the pieces would tear. relaxed: when that leaves the level over the limit, the
    old way after it, with one ratio for all and the small pieces given up (log["forced_relaxed"]; the image gates
    then judge the tears)."""

    def pick(o):
        m = S.an(o)[0]
        return ~(m.glow | m.card)[m.mi] & (m.rk() != 1) & _free(S, o)

    for strict in ((True, False, False) if relaxed else (True,)):
        T = S.tris()
        if T <= limit:
            return
        keep = 0
        for o in S.objs:
            m = S.an(o)[0]
            if len(m.lt):
                keep += int((m.lt - 2)[~pick(o)].sum())
        r = float(np.clip((0.96 * limit - keep) / max(T - keep, 1), 0.02, 1.0))
        log.setdefault("pre_force_tris", T)
        _collapse(S, pick, r, log, "forced_ratio" if strict else "relaxed_ratio", strict=strict)
        if not strict:
            log["forced_relaxed"] = True


def reduce_objects(objs, level, budget, mount="ground", allow_below=False, allow=1.0, cache=None, bounds=None, diag=None,
                   patches=True, lights=None, fixed=None):
    """Reduce mesh objects (realized: modifiers, UVs and smoothing resolved) in place to about `budget` triangles.
    mount: 'ground' | 'wall' | 'wall_ground' | 'hang' - where the viewers of the asset can stand.
    allow: the level may end at up to allow x budget before the last-resort collapse of everything is used.
    cache: dict kept between the levels of one asset (the visibility is computed once for both).
    patches: False leaves the flat-patch plan of step 8 out.
    lights: tk.lodimage.lights_of(sockets) - the asset's own lights: the unseen faces that cast their shadows stay
            (_light_casters; the later steps do not know them - a caster dropped or moved there shows as "leak").
    bounds: (min, max) of the full model when the objects are not all of it.
    fixed: objects of the model that are not reduced (identity parts): they hide what is behind them (_close).
    -> log dict; log["objs"] are the objects that now make up the level (some are new)."""
    t0 = time.process_time()
    log = {"level": level, "budget": int(budget)}
    objs = _prep(objs)
    S = _S(objs, level, int(budget), mount, bounds, diag, log)
    tin = S.tris()
    log["tris_in"] = tin
    target = int(0.97 * budget)
    if tin <= budget or level <= 0:
        log.update(tris_out=tin, steps=[], objs=S.objs, seconds=round(time.process_time() - t0, 2))
        return log
    steps = []
    tinfo = _text_groups(S, level)
    if tinfo:
        log["text"] = {"groups": len(tinfo), "small": sum(1 for t in tinfo.values() if t["small"]),
                       "tris_in": int(sum(t["tris"] for t in tinfo.values()))}
    S.lights = list(lights) if lights else None
    V0, T0, G0, W0, L0, C0 = S.soup()
    probe = _im.Probe(V0, T0, G0, W0, mount, level, maxpx=200, nmax=160000, tone=L0, see=C0)
    touch = (cache or {}).get("touch", {}).get(level)
    if touch is None:
        touch = _Touch(S)
        if cache is not None:
            cache.setdefault("touch", {})[level] = touch
    else:
        touch.used = [np.zeros(len(u), dtype=bool) for u in touch.used]
    hc = _holes_start(S, cache, touch, fixed) if (HOLES or TRACE) else None
    S.hc = hc if HOLES else None
    if hc is not None:
        log["holes_s"] = round(time.process_time() - t0, 2)
    base = {"sil": 0.0, "hole": 0.0, "lost": 0.0, "glow": 1.0, "lum": 1.0}
    if cache is not None and "zs" in cache:
        zs = cache["zs"]
    else:
        zs = _zstart(S)
        if cache is not None:
            cache["zs"] = zs
    S.zs = zs
    S.cache = cache
    if zs is not None:
        log["z0"] = zs["z0"]

    def zfix(step, table=True):
        """glowing faces and what came to lie on them (see _zfix) -> the area still within Z_NEAR"""
        nonlocal T
        n = len(log.get("z", []))
        left = _zfix(S, zs, log, step)
        if table and T is not None and len(log.get("z", [])) > n and (log["z"][-1][3] or log["z"][-1][4]):
            T = _table(S)
        return left

    def trace(step):
        if TRACE:
            sp = S.soup()
            m = probe.measure(*sp)
            log.setdefault("trace", []).append(f"{step}: {S.tris()} tris sil {m['sil']:.4f} lost {m['lost']:.4f} hole {m['hole']:.4f} lum {m['lum']:.3f}"
                                              + (f" glow {m['glow']:.3f}" if m.get("glow") is not None else "")
                                              + (f" off bounds {S.sides(S.tol)}" if S.sides(S.tol) else ""))
            if zs is not None:
                z = _zscan(S)
                if z is not None:
                    log["trace"][-1] += f" z {float(z['w'][z['d'] < Z_NEAR].sum()):.4f}"
            if hc is not None:
                h = _holes_read(S, hc, detail=True)
                w = (h.get("where") or [{}])[0]
                log["trace"][-1] += (f" | street holes {100 * h['holes']:.3f} % (back {100 * h['back']:.3f} none {100 * h['none']:.3f}, "
                                     f"worst view {100 * h['worst']:.2f} %)" + (f" at {w.get('at')} {w.get('mat')}" if w else ""))
            print(f"TSQ_TRACE lod {level} " + log["trace"][-1])

    T = None

    def mend(step):
        """what the step tore loose is tied back on (see _Touch)"""
        nonlocal T
        n, left = touch.mend(S)
        if n or left:
            log.setdefault("mended", []).append([step, n, left])
        if n and T is not None:
            T = _table(S)

    # 1. faces nobody sees
    th = time.process_time()
    vis = (cache or {}).get("vis")
    if vis is None or len(vis) != len(S.objs) or any(len(v) != len(o.data.polygons) for v, o in zip(vis, S.objs)):
        vis = visibility(S.objs, mount, allow_below)
        if cache is not None:
            cache["vis"] = vis
            cache["vis_s"] = round(time.process_time() - th, 2)
    # ... and what the probe's own views show of them stays: a face whose test points are all buried may still
    # stick out somewhere
    tw = probe.wins(V0, T0, W0, see=C0)
    seen, off = [], 0
    for o in S.objs:
        m = S.an(o)[0]
        if len(m.lt):
            tp = m.tris()[1]
            seen.append(np.bincount(tp, weights=tw[off:off + len(tp)], minlength=len(m.lt)) >= 2)
            off += len(tp)
        else:
            seen.append(np.zeros(0, dtype=bool))
    if hc is not None and HOLES:
        # ... and what a viewer on the street sees (tk.holes): the test directions are far fewer than his rays
        seen = [a_ | b_ if len(a_) == len(b_) else a_ for a_, b_ in zip(seen, hc["seen"])]
    del V0, T0, G0, W0, L0, C0, tw
    for need in (NEED[level], 1):
        raw = [(v < need) & ~sn for v, sn in zip(vis, seen)]
        if S.lights:
            # ... and what casts the shadows of the asset's own lights
            kl = ((cache or {}).get("lit") or {}).get((level, need))
            if kl is None or len(kl) != len(raw) or any(len(a_) != len(b_) for a_, b_ in zip(kl, raw)):
                tl_ = time.process_time()
                kl = _light_casters(S, raw, S.lights, allow_below)
                if cache is not None:
                    cache.setdefault("lit", {})[(level, need)] = kl
                    cache["lit_s"] = round(cache.get("lit_s", 0.0) + time.process_time() - tl_, 2)
            nk = int(sum(int(k_.sum()) for k_ in kl))
            if nk:
                log["lit_kept"] = nk
            raw = [a_ & ~k_ for a_, k_ in zip(raw, kl)]
        marks = _text_hidden(S, raw, tinfo)
        saved = S.save([o for o, pm in zip(S.objs, marks) if pm.any()])
        gone = 0
        grave = []
        for o, pm in zip(list(S.objs), marks):
            if pm.any():
                m = S.an(o)[0]
                gone += int((m.lt - 2)[pm].sum())
                if hc is not None and HOLES:
                    g_ = o.copy()               # what goes is kept aside: a later step may uncover it (_behind)
                    g_.data = o.data.copy()
                    g_.use_fake_user = True
                    _delete_polys(g_, ~pm)
                    grave.append(g_)
                _delete_polys(o, pm)
                S.dirty(o)
        if hc is not None:
            _grave(hc, grave)
        if not gone:
            S.forget(saved)
            break
        m_ = probe.measure(*S.soup())
        if m_["lost"] > 0.003 or m_["hole"] > 0.004 or (m_["glow"] is not None and m_["glow"] < 0.97):
            _grave(hc, None)
            S.put_back(saved)
            log.setdefault("rollback", []).append(f"hidden@{need}: lost {m_['lost']:.4f} hole {m_['hole']:.4f}")
            continue
        if S.guard(saved, f"hidden@{need}"):
            log["hidden_need"] = need
        log["hidden_tris"] = tin - S.tris()
        base = m_
        steps.append("hidden")
        break
    mend("hidden")
    zfix("hidden")
    trace("hidden")
    log["hidden_s"] = round(time.process_time() - th, 2)

    # 1b. lettering: sides off where only the fronts can be read, engravings too small to read closed
    if tinfo:
        n = _text_step(S, level, tinfo, log)
        if n:
            log["text"]["cheapened"] = n
            steps.append("text")
            base = dict(base, **{k: v for k, v in probe.measure(*S.soup()).items() if k in ("lum",)})
        trace("text")

    # 2. coplanar faces
    n = 0
    for f in (1.0, 0.5):
        saved = S.save()
        t_ = S.tris()
        for o in S.objs:
            if len(o.data.polygons):
                _dissolve(o, level, f)
                S.dirty(o)
        m_ = probe.measure(*S.soup())
        if m_["sil"] - base["sil"] > 0.25 * SIL[level] or m_["hole"] - base["hole"] > 0.01 or m_["lost"] - base["lost"] > 0.003:
            S.put_back(saved)
            log.setdefault("rollback", []).append(f"dissolve x{f}: sil {m_['sil']:.4f} hole {m_['hole']:.4f}")
            continue
        S.guard(saved, "dissolve")
        n = t_ - S.tris()
        base = dict(m_, glow=base["glow"] if m_["glow"] is None else m_["glow"])
        break
    log["tris_dissolved"] = n
    if n:
        steps.append("dissolve")
    trace("dissolve")

    def fits():
        return S.tris() <= target

    T = _table(S) if not fits() else None
    # 3. thin long pieces -> prisms
    if T is not None:
        saved = S.save()
        if _prisms(S, T, log):
            steps.append("prisms")
            S.guard(saved, "prisms")
            T = _table(S)
        else:
            S.forget(saved)
    trace("prisms")

    def drop_step(drop, step):
        """Drop the marked rows of the table. When that shifts the brightness of the picture (dark lettering off a
        light board, light studs off a dark door), the pieces of the material that does it stay. -> triangles gone"""
        nonlocal T, base
        mean = float((T["tone"] * T["area"]).sum() / max(float(T["area"].sum()), 1e-12))
        tries, htries = 0, 0
        while tries < 4:
            tries += 1
            drop = _protect_outline(S, T, drop, S.tol)
            if not drop.any():
                return 0
            saved = S.save([S.objs[oi] for oi, a, b in T["span"] if drop[a:b].any()])
            n = _apply_drop(S, T, drop)
            if hc is not None and HOLES and htries < 3:
                # no holes: a piece that was all that stood between the street and the sky stays - unless a face
                # step 1 took lies right behind it (the slab behind a door's panels): that one comes back instead
                hm = _holes_read(S, hc, detail=True)
                if hm["count"]:
                    left = ~_behind(S, hc, hm["eyes"], hm["pts"], False)[0]
                    rows = np.flatnonzero(drop)
                    stay = np.zeros(len(rows), dtype=bool)
                    Pl = hm["pts"][left]
                    if len(Pl):
                        for a_ in range(0, len(rows), 256):
                            r_ = rows[a_:a_ + 256]
                            stay[a_:a_ + 256] = ((Pl[None, :, :] >= T["mn"][r_][:, None, :] - 0.02)
                                                 & (Pl[None, :, :] <= T["mx"][r_][:, None, :] + 0.02)).all(2).any(1)
                    if stay.any():
                        S.put_back(saved)
                        drop = drop.copy()
                        drop[rows[stay]] = False
                        log.setdefault("rollback", []).append(f"{step}: {int(stay.sum())} pieces stay, the street would look through")
                        htries += 1
                        tries -= 1
                        continue
            m_ = probe.measure(*S.soup())
            if abs(m_["lum"] - base["lum"]) <= LUMTOL[level]:
                S.guard(saved, step)
                base = dict(base, lum=m_["lum"])
                T = _table(S)
                return n
            S.put_back(saved)
            w = {}
            for nm, x in zip(T["mat"][drop].tolist(), (T["area"][drop] * np.abs(T["tone"][drop] - mean)).tolist()):
                w[nm] = w.get(nm, 0.0) + x
            worst = max(w, key=w.get)
            log.setdefault("rollback", []).append(f"{step}: brightness {m_['lum']:.3f}, pieces of {worst} stay")
            drop = drop & (T["mat"] != worst)
        return 0

    # 4. loose pieces under the size limit
    if T is not None and not fits():
        pd = {}
        for p in set(T["part"].tolist()):
            r = T["part"] == p
            pd[p] = float(np.linalg.norm(T["mx"][r].max(0) - T["mn"][r].min(0)))
        lim = np.minimum(S.ms, 0.25 * np.array([pd[p] for p in T["part"]]))
        drop = np.isin(T["kind"], (K_ATOM, K_DENSE)) & ~T["near"] & (T["d"] < lim)
        n = drop_step(drop, "small")
        if n:
            log["small_dropped"] = n
            steps.append("small")

    # 4b. lettering under 3 px, when the rest cannot give enough: whole lines, smallest first
    if T is not None and tinfo and not fits():
        tr_, kd_ = T["tris"], T["kind"]
        D_ = int(tr_[kd_ == K_DENSE].sum())
        over = int(tr_.sum()) - D_ + RMIN[level] * D_ - target
        if over > 0:
            n = _text_drop(S, T, tinfo, over, probe, base, level, log)
            if n:
                log["text"]["dropped"] = n
                steps.append("textdrop")
                T = _table(S)

    if "small" in steps or "textdrop" in steps or "prisms" in steps:
        mend("small")
        zfix("small")
    trace("small")

    # 5. plant cards
    if T is not None and not fits():
        pt = int(T["tris"][T["kind"] == K_PLANT].sum())
        tot = int(T["tris"].sum())
        if pt > 0.15 * tot:
            rest = int(T["tris"][np.isin(T["kind"], (K_DENSE, K_ATOM, K_FIELD))].sum())     # what a collapse can take from
            fixed = tot - pt - rest                                                         # glow, prisms
            q = float(np.clip((target - fixed) / max(pt + rest, 1), 1.0 / 12.0, 1.0))
            if level == 1:
                # half the thinning, the rest comes out of the dense meshes - as long as that leaves them RMIN of
                # themselves (a planter that is mostly leaves would otherwise lose its tub)
                q = max(q, min(1.0 - (1.0 - q) / 2.0, (target - fixed - RMIN[level] * rest) / max(pt, 1)))
            saved = S.save()
            if _thin_plants(S, T, q, log):
                S.guard(saved, "plants")
                steps.append("plants")
                T = _table(S)
            else:
                S.forget(saved)

    def split():
        tr, kd = T["tris"], T["kind"]
        D = int(tr[kd == K_DENSE].sum())
        return int(tr.sum()) - D, D

    trace("plants")

    # 6. small loose pieces, when the dense meshes alone cannot give enough
    if T is not None and not fits():
        A, D = split()
        over = A + RMIN[level] * D - target
        if over > 0:
            c = np.flatnonzero((T["kind"] == K_ATOM) & ~T["near"] & (T["d"] < S.ms * DROPCAP[level]))
            c = c[np.argsort(T["d"][c], kind="stable")]
            cs = np.cumsum(T["tris"][c])
            k_ = int(np.searchsorted(cs, over) + 1)
            n = 0
            pre_a = probe.measure(*S.soup()) if len(c) else None     # (`base` is the state before the plants were thinned)
            for _ in range(4 if len(c) else 0):
                # ... the smallest first, and fewer of them when the outline shows it (finials, rays, posts: small
                # and loose, and all of the silhouette)
                drop = np.zeros(len(T["kind"]), dtype=bool)
                drop[c[:k_]] = True
                cut = float(T["d"][drop].max()) if drop.any() else 0.0
                bak, T_b, base_b = S.backup(), T, dict(base)
                n = drop_step(drop, "atoms")
                if not n:
                    S.discard(bak)
                    break
                m_ = probe.measure(*S.soup())
                if m_["sil"] - pre_a["sil"] <= 0.25 * SIL[level] and m_["lost"] - pre_a["lost"] <= 0.003:
                    S.discard(bak)
                    break
                S.discard(S.swap(bak))
                T, base, n = T_b, base_b, 0
                log.setdefault("rollback", []).append(f"atoms<{cut:.3f}: sil {m_['sil']:.4f} lost {m_['lost']:.4f}")
                k_ //= 2
                if k_ < 1:
                    break
            if n:
                log["atoms_dropped"] = n
                log["atoms_cut_m"] = round(cut, 4)
                steps.append("atoms")

    if "atoms" in steps or "plants" in steps:
        mend("atoms")
    trace("atoms")

    def enough():
        # what is left for the collapse: a real dense mesh stands a lot of it, the welded skin little
        t = _table(S)
        tr, kd = t["tris"], t["kind"]
        skin = np.zeros(len(tr))
        for oi, a, b in t["span"]:
            m, isl = S.an(S.objs[oi])
            skin[a:b] = np.bincount(isl["pisl"], weights=(m.lt - 2) * (m.rk() == 2), minlength=b - a)
        dn = kd == K_DENSE
        w = float(skin[dn].sum())
        d_ = float(tr[dn].sum()) - w
        return float(tr.sum()) - d_ - w + RCOMF[level] * d_ + 0.6 * w <= target

    # 7. fields of small pieces -> one skin. When a stage of it is taken back, the state before the weld is kept:
    #    the collapse that follows tears the loose pieces apart, and flat patches (see 8) may be the better picture
    wstart = wtable = None
    if TRACE and T is not None:
        log["kinds"] = {("", "glow", "prism", "plant", "field", "atom", "dense", "text")[k]:
                        [int((T["kind"] == k).sum()), int(T["tris"][T["kind"] == k].sum()), int((T["near"] & (T["kind"] == k)).sum()),
                         round(float(np.median(T["d"][T["kind"] == k])), 3) if (T["kind"] == k).any() else 0] for k in range(1, 8)}
        log["kinds"]["target_ms"] = [target, round(S.ms, 3), round(S.diag, 2)]
    if T is not None and not fits():
        A, D = split()
        if A + RCOMF[level] * D > target and (T["kind"] == K_FIELD).any():
            wstart, wtable = S.backup(), T
            if not _weld(S, T, probe, base, enough, log):
                S.discard(wstart)
                wstart = wtable = None
            steps.append("weld")
            T = _table(S)

    if steps and steps[-1] == "weld":
        mend("weld")
        zfix("weld")
    trace("weld")

    # 7b. glow is left alone - unless it is so much of the model (a wall of lit windows, a globe lamp, neon tubes)
    #     that the rest would have to shrink below RFLOOR of itself: then it is collapsed by the ratio the whole
    #     level needs, gentler when the glowing picture suffers, not at all when it still does
    if T is not None and not fits():
        pre = probe.measure(*S.soup())
        bak = S.backup()
        n = _panes(S, S.ms)
        if n >= 24:
            m_ = probe.measure(*S.soup())
            if ((m_["glow"] is None or pre["glow"] is None or m_["glow"] >= pre["glow"] - 0.04)
                    and m_["sil"] - pre["sil"] <= 0.1 * SIL[level] and m_["lost"] - pre["lost"] <= 0.002
                    and abs(m_["lum"] - pre["lum"]) <= LUMTOL[level]):
                S.discard(bak)
                log["panes_tris"] = n
                steps.append("panes")
            else:
                S.discard(S.swap(bak))
                log.setdefault("rollback", []).append(f"panes: glow {m_['glow']} sil {m_['sil']:.4f} lum {m_['lum']:.3f}")
        else:
            S.discard(S.swap(bak)) if n else S.discard(bak)
        T = _table(S)
    if T is not None and not fits():
        gt = coll = 0
        for oi, a, b in T["span"]:
            m, isl = S.an(S.objs[oi])
            g = m.glow[m.mi] & (T["kind"][a:b] != K_TEXT)[isl["pisl"]]
            c = np.isin(T["kind"][a:b], (K_DENSE, K_ATOM, K_FIELD))[isl["pisl"]] & ~g
            gt += int((m.lt - 2)[g].sum())
            coll += int((m.lt - 2)[c].sum())
        tot = int(T["tris"].sum())
        kept = tot - coll
        if gt > 0.25 * target and gt > 24 and (target - kept) < RFLOOR[level] * coll:
            pre = probe.measure(*S.soup())
            want = float(np.clip((target - (kept - gt)) / max(gt + coll, 1), 0.02, 1.0))
            for gr in (want, min(1.0, 1.8 * want), min(1.0, 3.2 * want)):
                if gr >= 0.9:
                    break
                bak = S.backup()
                lg = {}
                _collapse(S, lambda o: S.an(o)[0].glow[S.an(o)[0].mi] & _free(S, o), gr, lg, "ratio")
                zfix("glow", table=False)
                m_ = probe.measure(*S.soup())
                if (lg.get("ratio") and (m_["glow"] is None or pre["glow"] is None or m_["glow"] >= pre["glow"] - 0.08)
                        and m_["sil"] - pre["sil"] <= 0.25 * SIL[level] and m_["lost"] - pre["lost"] <= 0.005
                        and abs(m_["lum"] - pre["lum"]) <= LUMTOL[level]):
                    S.discard(bak)
                    log["glow_ratio"] = lg["ratio"]
                    log["glow_tris"] = [gt, tot - S.tris()]
                    steps.append("glow")
                    break
                S.discard(S.swap(bak))
                log.setdefault("rollback", []).append(f"glow@{gr:.3f}: glow {m_['glow']} sil {m_['sil']:.4f} lum {m_['lum']:.3f}")
            T = _table(S)
    trace("glow")

    # 8. the collapse: of the dense meshes alone, or of everything that does not glow (which takes the triangles
    #    where they matter least, but also from the small loose pieces). The one that fits the numbers goes first;
    #    when the picture suffers, the other is tried and the better of the two stays. Where a field would not
    #    weld (roof tiles laid over each other), a third plan starts from the state before the weld: the field as
    #    flat patches in the planes of its pieces, then the collapse of what is left.
    if T is not None and not fits():
        A, D = split()
        r_d = (target - A) / D if D > 0 else 0.0

        def run(plan):
            masks, kept = {}, 0
            for oi, a, b in T["span"]:
                o = S.objs[oi]
                m, isl = S.an(o)
                if plan == "dense":
                    pm = (T["kind"][a:b] == K_DENSE)[isl["pisl"]]
                else:
                    pm = np.isin(T["kind"][a:b], (K_DENSE, K_ATOM, K_FIELD))[isl["pisl"]] & ~m.glow[m.mi]
                masks[o.name] = pm
                kept += int((m.lt - 2)[~pm].sum())
            tot = int(T["tris"].sum())
            if tot <= kept:
                return False
            r = float(np.clip((target - kept) / (tot - kept), RFLOOR[level] if plan == "dense" else 0.02, 1.0))
            if r >= 0.98:
                return False
            lg = {}
            _collapse(S, lambda o: masks.get(o.name, np.zeros(len(o.data.polygons), dtype=bool)), r, lg, "ratio")
            if "rollback" in lg:
                log.setdefault("rollback", []).extend(f"{plan}:{x}" for x in lg["rollback"])
            return lg.get("ratio", 1.0)

        def run_patch():
            nonlocal T
            lg = {}
            if not _patches(S, T, lg) or S.sides(S.tol) or S.empty_parts():
                return False
            plog.update(lg)
            if TRACE:
                trace("patch")
            gr = log.get("glow_ratio")
            if gr:
                # the state before the weld is the state before the collapse of the glow (7b) too: the same one here,
                # or this plan carries the bulbs of a hotel front in full and its walls pay for them
                pre_, bak_, lg_ = probe.measure(*S.soup()), S.backup(), {}
                _collapse(S, lambda o: S.an(o)[0].glow[S.an(o)[0].mi] & _free(S, o), gr, lg_, "ratio")
                zfix("glow", table=False)
                m_ = probe.measure(*S.soup())
                if (lg_.get("ratio") and (m_["glow"] is None or pre_["glow"] is None or m_["glow"] >= pre_["glow"] - 0.08)
                        and m_["sil"] - pre_["sil"] <= 0.25 * SIL[level] and m_["lost"] - pre_["lost"] <= 0.005
                        and abs(m_["lum"] - pre_["lum"]) <= LUMTOL[level]):
                    S.discard(bak_)
                    plog["glow"] = lg_["ratio"]
                else:
                    S.discard(S.swap(bak_))
            T = _table(S)
            if T is None or fits():
                return 1.0
            A_, D_ = split()
            p2 = "dense" if D_ > 0 and (target - A_) / D_ >= RMIN[level] else "all"
            r = run(p2)
            plog["then"] = [p2, r]
            return 1.0

        pre = probe.measure(*S.soup())
        plans = (["dense"] if D > 0 and r_d >= RFLOOR[level] else []) + ["all"]
        if not (D > 0 and r_d >= RMIN[level]):
            plans.reverse()
        start = S.backup() if len(plans) > 1 else None
        if wstart is not None and patches:
            plans.append("patch")
        T_pre, plog = T, {}
        best = held = None
        for i, plan in enumerate(plans):
            if i:
                # the best so far stays in the scene, the plan starts from its own copy of the state
                if plan == "patch":
                    held, wstart, T = S.swap(wstart), None, wtable
                else:
                    held, start, T = S.swap(start), None, T_pre
            r = run_patch() if plan == "patch" else run(plan)
            zl = zfix(plan, table=False)
            m_ = probe.measure(*S.soup())
            fit = S.tris() <= allow * budget
            op = _open(S)
            g_ = (m_["sil"] + 0.5 * m_["hole"] + m_["lost"] + (0.0 if fit else 1.0)
                  + (1.0 if zs is not None and zl > zs["lim"] else 0.0) + op)
            sc = g_ + abs(m_["lum"] - pre["lum"])
            # the patches against the torn field: the two shift the brightness in opposite directions (and a torn
            # field next to torn dark joinery not at all), so these are compared pixel by pixel
            st = g_ + m_["tone"] - pre["tone"]
            if plan == "patch":
                ok = r and (m_["glow"] is None or pre["glow"] is None or m_["glow"] >= pre["glow"] - 0.03)
                wins = bool(ok) and st < best[5]
            else:
                wins = best is None or sc < best[2]
            log.setdefault("plans", []).append(f"{plan}: {S.tris()} tris, sil {m_['sil']:.4f} hole {m_['hole']:.4f} lost {m_['lost']:.4f}"
                                               f" lum {m_['lum']:.3f} open {op:.3f} -> {sc:.4f}, by pixel {st:.4f}"
                                               + (f" ({plog.get('patches')} patches)" if plan == "patch" else ""))
            if wins:
                if held is not None:
                    S.discard(held)
                best = (plan, r, sc, m_, fit, st)
            else:
                S.discard(S.swap(held, linked=True))
            held = None
            m_ = best[3]
            if (best[4] and m_["sil"] - pre["sil"] <= 0.25 * SIL[level] and m_["hole"] - pre["hole"] <= 0.01
                    and m_["lost"] - pre["lost"] <= 0.005 and abs(m_["lum"] - pre["lum"]) <= LUMTOL[level]):
                break
        if start is not None:
            S.discard(start)
        plan, r = best[0], best[1]
        if plan == "patch":
            log["patches"], log["patch_tris_saved"] = plog.get("patches"), plog.get("patch_tris_saved")
            if plog.get("glow"):
                log["patch_glow_ratio"] = plog["glow"]
            steps.append("patch")
            plan, r = plog.get("then", ("all", False))
        if r:
            log["collapse"] = plan
            log["collapse_ratio"] = r
            steps.append("collapse" if plan == "dense" else "collapse_all")
    if wstart is not None:
        S.discard(wstart)
        wstart = None
    if steps and steps[-1] in ("glow", "patch", "collapse", "collapse_all"):
        mend("collapse")
        zfix("collapse")
    trace("collapse")

    # 9. still over. A field of loose pieces under the size limit that would neither weld nor lie in patches (moss in
    #    the joints of a pavement, blossom on a vine) is no surface to collapse: piece by piece there is nothing to
    #    take, and a collapse of the lot empties some and leaves splinters of the others. It is thinned like plant
    #    cards - fewer, evenly spread, larger. Then the plants in full, then everything that does not glow.
    if S.tris() > allow * budget and not NOTHIN:
        T = _table(S)
        if T is not None:
            rows = np.flatnonzero((T["kind"] == K_FIELD) & (T["d"] < S.ms) & (T["tris"] <= ISL_FLOOR) & ~T["near"])
            ft = int(T["tris"][rows].sum())
            tot = int(T["tris"].sum())
            if len(rows) >= FIELD_N and ft > 0.15 * tot:
                pre = probe.measure(*S.soup())
                q = float(np.clip((target - (tot - ft)) / ft, 0.25 if level == 1 else 1.0 / 12.0, 1.0))
                bak = S.backup()
                if _thin_plants(S, T, q, log, rows=rows, key="field"):
                    m_ = probe.measure(*S.soup())
                    if (m_["sil"] - pre["sil"] <= SIL[level] and m_["lost"] - pre["lost"] <= 0.02
                            and abs(m_["lum"] - pre["lum"]) <= 2.0 * LUMTOL[level] and not S.sides(S.tol)):
                        S.discard(bak)
                        steps.append("fields")
                        mend("fields")
                    else:
                        S.discard(S.swap(bak))
                        log.setdefault("rollback", []).append(f"fields@{q:.2f}: sil {m_['sil']:.4f} lost {m_['lost']:.4f} lum {m_['lum']:.3f}")
                        for k_ in ("field_keep", "field_scale", "field_tris_dropped"):
                            log.pop(k_, None)
                else:
                    S.discard(bak)
        trace("fields")
    if S.tris() > allow * budget:
        T = _table(S)
        if T is not None and int(T["tris"][T["kind"] == K_PLANT].sum()) > 0.15 * int(T["tris"].sum()):
            saved = S.save()
            if _thin_plants(S, T, float(np.clip(target / max(int(T["tris"].sum()), 1), 1.0 / 12.0, 1.0)), log):
                S.guard(saved, "plants2")
                steps.append("plants2")
            else:
                S.forget(saved)
    if S.tris() > allow * budget:
        _force(S, log, allow * budget)
        steps.append("force")
        mend("force")
        zfix("force")
        trace("force")
    if zs is not None:
        log["z_left"] = round(zfix("end"), 4)
    th = time.process_time()
    _close(S, hc, log)
    trace("close")
    log["holes_s"] = round(log.get("holes_s", 0.0) + time.process_time() - th, 2)
    if cache is None:
        touch.free()
        _grave(hc, None)
    if S.cover is not None:
        log["open"] = [round(x, 4) for x in _open(S, per=True)]
    _finish(S)
    log.update(tris_out=S.tris(), steps=steps, objs=S.objs, seconds=round(time.process_time() - t0, 2))
    return log


# ----------------------------------------------------------------------------
# Identity parts (tk.levels.lod_standin / lod_keep)
# ----------------------------------------------------------------------------

def ident_info():
    """what snapshot() wrote about the identity groups of the build: {"modes": {group: [lod 1, lod 2]} ("keep" |
    "swap"), "text": {level: [text() numbers inside groups swapped at that level]}} or {}"""
    try:
        return json.loads(bpy.context.scene.get("tsq_ident", "{}"))
    except Exception:
        return {}


def _face_ints(me, name):
    a = me.attributes.get(name)
    n = len(me.polygons)
    if a is None or a.domain != 'FACE' or len(a.data) != n or not n:
        return None
    v = np.empty(n, dtype=np.int32)
    a.data.foreach_get("value", v)
    return v


def _del_faces(me, mask):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f, m in zip(bm.faces, mask) if m], context='FACES')
    bm.to_mesh(me)
    bm.free()
    me.update()


def _drop(o):
    me = o.data
    bpy.data.objects.remove(o, do_unlink=True)
    if me.users == 0:
        bpy.data.meshes.remove(me)


def ident_split(level):
    """The identity parts of the restored scene for `level`: stand-ins of other levels go, the tagged originals of a
    group that has a stand-in at this level go, the stand-ins of this level and the tagged originals that are kept
    are split off into objects of their own carrying the property tsq_keep. reduce_scene / force_scene leave those
    alone. -> the protected objects"""
    scene = bpy.context.scene
    modes = ident_info().get("modes") or {}
    out = []
    for o in list(scene.objects):
        if o.type == 'MESH' and "tsq_sid" in o and "tsq_keep" not in o:
            if int(o["tsq_slevel"]) != level or len(o.data.polygons) == 0:
                _drop(o)
            else:
                o["tsq_keep"] = 1
                out.append(o)
    swap = [int(g) for g, m in modes.items() if m[level - 1] == "swap"]
    for o in list(scene.objects):
        if o.type != 'MESH' or "tsq_keep" in o:
            continue
        v = _face_ints(o.data, "tsq_id")
        if v is None or not v.any():
            continue
        kill = np.isin(v, swap)
        keep = (v > 0) & ~kill
        if keep.all():
            o["tsq_keep"] = 1
            out.append(o)
            continue
        if keep.any():
            c = o.copy()
            c.data = o.data.copy()
            _link(c)
            _del_faces(c.data, ~keep)
            c["tsq_keep"] = 1
            out.append(c)
        if (v > 0).all():
            _drop(o)
        else:
            _del_faces(o.data, v > 0)
    return out


def _kept():
    """(objects reduce_scene / force_scene work on, triangles of the protected identity parts)"""
    objs, n = [], 0
    for o in bpy.context.scene.objects:
        if o.type == 'MESH' and len(o.data.polygons) > 0:
            if "tsq_keep" in o:
                n += _ntris(o.data)
            else:
                objs.append(o)
    return objs, n


def reduce_scene(level, ref, meta, params, allow=1.0, patches=True):
    """Reduce the un-merged scene of the full model (restore(snapshot())) to level 1 / 2, ready for
    tk.final_levels.finalize_level(). ref: tk.final_levels.reference() of lod 0. -> log dict."""
    from .final_levels import budget
    scene = bpy.context.scene
    for o in list(scene.objects):
        if o.type in ('CURVE', 'FONT', 'SURFACE'):
            apply_modifiers(o)
    ident_split(level)
    objs, nk = _kept()                  # identity parts stay out of every step; their triangles come off the budget
    fixed = [o for o in scene.objects if o.type == 'MESH' and "tsq_keep" in o]
    mount = params.get("mount") if params.get("mount") in ("hang", "wall", "ground", "wall_ground") else meta.get("mount", "ground")
    b = ref["info"]["bounds"]
    log = reduce_objects(objs, level, max(budget(level, ref["tris"]) - nk, 1), mount=mount, allow_below=bool(meta.get("allow_below_ground")),
                         allow=allow, cache=ref.setdefault("_red", {}), bounds=(np.array(b["min"]), np.array(b["max"])),
                         diag=ref["diag"], patches=patches, lights=_im.lights_of(ref["info"].get("sockets")), fixed=fixed)
    log.pop("objs", None)
    if nk:
        log["ident_tris"] = nk
        log["tris_out"] = log["tris_out"] + nk
    return log


def force_scene(level, ref, limit=None):
    """The scene of reduce_scene(..., allow > 1) that ended over its budget, brought under `limit` triangles (default:
    the budget). -> log dict"""
    from .final_levels import budget
    objs, nk = _kept()
    b = ref["info"]["bounds"]
    bud = budget(level, ref["tris"])
    log = {"level": level, "budget": bud, "limit": int(limit or bud)}
    hc = (ref.get("_red", {}).get("holes") or {}).get(level) if HOLES else None
    if hc is not None:
        hc["fixed"] = [o for o in bpy.context.scene.objects if o.type == 'MESH' and "tsq_keep" in o and len(o.data.polygons)]
    # what _close() put back so far stays and is not counted against the limit (docs/LOD.md "No holes")
    limit = max(int(limit or bud) - nk, 1) + (hc["added"] if hc is not None else 0)
    S = _S(objs, level, bud, "ground", (np.array(b["min"]), np.array(b["max"])), ref["diag"], log)
    for o in objs:
        if "tsq_hole" in o and len(o.data.polygons):
            _set_rk(o, np.ones(len(o.data.polygons), dtype=bool), 1)       # _finish() took the mark off
    zs = S.zs = ref.get("_red", {}).get("zs")
    S.hc = hc
    S.lights = _im.lights_of(ref["info"].get("sockets")) or None
    _force(S, log, int(limit or bud), relaxed=True)
    touch = ref.get("_red", {}).get("touch", {}).get(level)
    if touch is not None:
        touch.used = [np.zeros(len(u), dtype=bool) for u in touch.used]
        n, left = touch.mend(S)
        if n or left:
            log["mended"] = [n, left]
        if n and S.tris() > int(limit or bud):
            _force(S, log, int(limit or bud), relaxed=True)     # what was put back stays (it is marked), the rest gives a little more
            log["mended"].append(touch.mend(S)[1])
    if zs is not None:
        log["z_left"] = round(_zfix(S, zs, log, "force"), 4)
    _close(S, hc, log, "force")
    S.cover = ref.get("_red", {}).get("cover", {}).get(level)
    if S.cover is not None:
        log["open"] = [round(x, 4) for x in _open(S, per=True)]
    _finish(S)
    log["tris_out"] = S.tris() + nk
    return log


CAND = ("reduced", "reduced15", "reduced20", "reduced_np", "reduced15_np", "reduced20_np")


def cand_file(stem, name, level):
    """the file of a reduced candidate of level_candidates()"""
    return f"{stem}.{name.replace('reduced', 'red').replace('_', '')}.lod{level}.glb"


def allowance(level, meta):
    """how far over its budget a reduced level of this asset may ship (x budget): tk.lodimage.ALLOW2 at lod 2,
    ALLOW2B for buildings and landmarks"""
    if level != 2:
        return 1.0
    return _im.ALLOW2B if meta.get("class") in _im.BIG else _im.ALLOW2


def level_candidates(level, snap, ref, meta, params, stem, times=None):
    """The reduced candidates of a level for tk.final_levels.pick(): puts the snapshot() back into the scene, reduces
    it and exports it through tk.final_levels.finalize_level (cand_file(stem, name, level)).
      "reduced"     the level inside its budget
      "reduced15"   a lod 2 that ended between its budget and tk.lodimage.ALLOW2 x the budget; "reduced" is then the
                    same scene forced under the budget
      "reduced20"   a lod 2 of a building / landmark that ended between ALLOW2 and ALLOW2B x the budget (stopping
                    short of the budget is better than tearing); "reduced15" and "reduced" are then that scene forced
                    further, one after the other
      "..._np"      the same chain without the flat-patch plan, made when the first chain took it: pick() then ships
                    the patches only where they give the lower image difference
    times: optional dict, CPU seconds are added to its "reduce_s" / "finlevel_s".
    -> ([candidate, ...], log of the first chain; log["np"]: of the second)"""
    from . import final_levels as _fl
    T = times if times is not None else {}
    bud = _fl.budget(level, ref["tris"])
    allow = allowance(level, meta)
    out = []

    def fin():
        c0 = time.process_time()
        r = _fl.finalize_level(level, ref, stem + ".redtmp.glb", params, meta, None, None, 0.0, True, seen=SEEN, fit=True,
                               text=True)
        T["finlevel_s"] = T.get("finlevel_s", 0.0) + time.process_time() - c0
        return r

    lims = [bud * f for f in (_im.ALLOW2B, _im.ALLOW2, 1.0) if f <= allow]
    top = len(lims) - 1                                 # the highest tier that may ship

    def chain(patches, sfx):
        clear_meshes()
        restore(snap)
        ref.get("_red", {}).get("cover", {}).pop(level, None)
        c0 = time.process_time()
        log = reduce_scene(level, ref, meta, params, allow=allow, patches=patches)
        T["reduce_s"] = T.get("reduce_s", 0.0) + time.process_time() - c0
        n = log["tris_out"] - log.get("closed", 0)      # what _close() put back is not held against the budget
        last = 4
        while True:
            tier = 0 if n <= bud else 1 if n <= _im.ALLOW2 * bud else 2 if n <= _im.ALLOW2B * bud else 3
            if tier >= last:
                break                                   # the forced collapse got no further
            if tier <= top:
                held = hold() if tier else None
                try:
                    r = fin()
                    name = ("reduced", "reduced15", "reduced20")[tier] + sfx
                    f = cand_file(stem, name, level)
                    os.replace(f"{stem}.redtmp.lod{level}.glb", f)
                    c = _fl.candidate(name, "reduced", r, f, allow=allow, text=r.get("text"),
                                      ident=bool(ident_info().get("modes")))
                    c["relaxed"] = bool(log.get("force") and log["force"][-1].get("forced_relaxed"))
                    c["open"] = ((log["force"][-1] if log.get("force") else log).get("open") or [None])[0]
                    c["closed"] = int((log["force"][-1] if log.get("force") else log).get("closed") or 0)
                    out.append(c)
                    if not tier:
                        break
                    clear_meshes()
                    restore(held)
                finally:
                    if held is not None:
                        release(held)
            c0 = time.process_time()
            fl = force_scene(level, ref, max(x for x in lims if x < n))
            T["reduce_s"] = T.get("reduce_s", 0.0) + time.process_time() - c0
            log.setdefault("force", []).append(fl)
            n, last = fl["tris_out"] - fl.get("closed", 0), tier
        return log

    try:
        log = chain(True, "")
        if "patch" in (log.get("steps") or []):
            log["np"] = chain(False, "_np")
    finally:
        t = ref.get("_red", {}).get("touch", {}).pop(level, None)
        if t is not None:
            t.free()
        _grave(ref.get("_red", {}).get("holes", {}).pop(level, None), None)
    return out, log


CHILD_FLOOR = (0, 60, 24)


def reduce_child(obj, level, meta=None):
    """tk.asm.use(): the joined mesh of a composed asset that has no levels of its own, reduced for `level`
    (40 % / 10 % of its triangles). Replaces obj; -> the new object."""
    name = obj.name
    keys = {k: obj[k] for k in obj.keys()}
    tris = _ntris(obj.data)
    bud = int(max(BUDGET[level] * tris, min(tris, CHILD_FLOOR[level])))
    mount = "wall" if (meta or {}).get("mount") == "wall" else "hang"
    log = reduce_objects([obj], level, bud, mount=mount)
    objs = log["objs"]
    if not objs:
        raise RuntimeError("nothing left of the composed asset")
    if len(objs) == 1:
        j = objs[0]
    else:
        j = _join(objs, name)
    j.name = name
    for k, v in keys.items():
        try:
            j[k] = v
        except Exception:
            pass
    j["tsq_uv"] = "keep"
    j["tsq_smooth"] = "keep"
    return j


# ----------------------------------------------------------------------------
# The un-merged full model, kept for the reducer
# ----------------------------------------------------------------------------

def snapshot():
    """Call right before tk.final.finalize() of lod 0. Resolves modifiers, UVs and smoothing of the scene in the
    order finalize() itself does (so it draws exactly the texture offsets it would have drawn) and returns
    unlinked copies of the un-merged mesh objects with their world transform. clear_scene() destroys them."""
    scene = bpy.context.scene
    for o in list(scene.objects):
        if o.type in ('CURVE', 'FONT', 'SURFACE'):
            apply_modifiers(o)
    groups = {}
    for o in scene.objects:
        if o.type == 'MESH' and len(o.data.polygons) > 0:
            groups.setdefault(o.get("tsq_part", "body"), []).append(o)
    snap = []
    for objs in groups.values():
        realize(objs)
        for o in objs:
            if len(o.data.polygons) == 0:
                continue
            _tag_ident(o)
            c = o.copy()
            c.data = o.data.copy()
            mw = world_matrix(o)
            c.parent = None
            c.matrix_parent_inverse = Matrix.Identity(4)
            c.matrix_basis = mw
            c.use_fake_user = True
            snap.append(c)
    # identity parts (tk.levels.lod_standin): the stand-ins join the snapshot, the groups go on the scene
    modes = {str(g): list(m) for g, m in (_STATE.get("ident") or {}).items()}
    if modes:
        txt = {}
        for lv in (1, 2):
            swap = [int(g) for g, m in modes.items() if m[lv - 1] == "swap"]
            ids = set()
            for c in snap:
                v, t = _face_ints(c.data, "tsq_id"), _face_ints(c.data, "tsq_text")
                if v is not None and t is not None:
                    ids.update(int(x) for x in np.unique(t[np.isin(v, swap) & (t > 0)] // 4))
            txt[str(lv)] = sorted(ids)
        scene["tsq_ident"] = json.dumps({"modes": modes, "text": txt})
        snap.extend(o for o in bpy.data.objects if o.type == 'MESH' and "tsq_sid" in o and o.users_collection == ())
    return snap


def hold():
    """unlinked copies of the mesh objects of the scene as they are now (see restore)"""
    out = []
    for o in bpy.context.scene.objects:
        if o.type == 'MESH' and len(o.data.polygons) > 0:
            c = o.copy()
            c.data = o.data.copy()
            mw = world_matrix(o)
            c.parent = None
            c.matrix_parent_inverse = Matrix.Identity(4)
            c.matrix_basis = mw
            c.use_fake_user = True
            out.append(c)
    return out


def clear_meshes():
    """remove every mesh object from the scene (sockets stay)"""
    for o in list(bpy.context.scene.objects):
        if o.type == 'MESH':
            me = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)


def restore(snap):
    """Put fresh copies of a snapshot() / hold() into the scene (after clear_meshes()) -> the new objects"""
    out = []
    for s in snap:
        c = s.copy()
        c.data = s.data.copy()
        c.use_fake_user = False
        _link(c)
        out.append(c)
    return out


def release(snap):
    """free a snapshot() / hold()"""
    for s in snap:
        try:
            me = s.data
            bpy.data.objects.remove(s, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
        except ReferenceError:
            pass
