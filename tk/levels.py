"""tk.levels - levels of detail that are MODELLED, not decimated (author guide: docs/LOD.md).

A level-aware asset (META["lod"] = True) is built three times with the same params and seed:
lod() == 0 is the full model, 1 the mid level (<= 40 % of its triangles), 2 the far level (<= 10 %).
At levels 1 and 2 the toolkit primitives already cut their own tessellation (round sections by a
sagitta tolerance, bevels to one chamfer or none, simplified profiles and paths, chains to a strip,
bolts to nothing); the tools here are for everything only the author can decide.
At level 0 every function in this module is a no-op, so the full model never changes.
"""
import bpy
import hashlib
import math
import numpy as np

from .core import _LOD, lod, _auto_id, _STATE

# materials that carry the night read: geometry using them is never dropped by lod_upto / lod_drop
GLOW = ("emissive", "neon", "window_lit", "glass_stained", "glass_frosted", "glass_lantern")
# triangle budget of each level as a fraction of level 0, and the absolute floor small assets may keep
BUDGET = (1.0, 0.40, 0.10)
BUDGET_FLOOR = (0, 400, 150)
# sagitta (m) a round section may lose to faceting / deviation a simplified outline may have
TOL = (0.0, 0.004, 0.018)
# bevels narrower than this (m) are dropped, wider ones become a single chamfer
BEVEL_MIN = (0.0, 0.006, 0.03)


# ----------------------------------------------------------------------------
# Author tools
# ----------------------------------------------------------------------------

class _XI(int):
    """an int the toolkit must not rescale"""


class _XF(float):
    """a float the toolkit must not rescale"""


def lod_exact(v):
    """Mark a count / bevel width as final: the toolkit will not scale it at any level.
        cylinder("nut", 0.02, 0.01, verts=lod_exact(6))
    (lod_pick already returns exact values.)"""
    if isinstance(v, (tuple, list)):
        return tuple(lod_exact(c) for c in v)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    return _XI(v) if isinstance(v, int) else _XF(v)


def lod_pick(full, mid, far=None):
    """A value per level: `full` at lod 0, `mid` at lod 1, `far` at lod 2 (default: `mid`).
        lathe("urn", prof, segments=lod_pick(48, 20, 10))
        style = lod_pick("carved", "chamfered", "plain")
    Numbers picked for lod 1 / 2 are used as given (no automatic scaling on top)."""
    L = lod()
    if L == 0:
        return full
    return lod_exact(mid if (L == 1 or far is None) else far)


def lod_thin(seq, mid=0.5, far=0.25):
    """Fewer of a repeated thing: the whole sequence at lod 0, an evenly spread `mid` fraction at
    lod 1 and `far` at lod 2 (first and last always stay; a fraction of 0 gives nothing).
        for x in lod_thin(baluster_xs, 0.5, 0.25): baluster(x, width=bw * lod_pick(1, 1.6, 2.4))
    Make the survivors fatter so the run keeps its visual weight. Never draw from the shared `rng`
    inside such a loop - draw all values before it, or use rng.fork(f"bal{i}")."""
    L = lod()
    if L == 0:
        return seq
    items = list(seq)
    n = len(items)
    f = mid if L == 1 else far
    if f <= 0:
        return []
    k = max(2 if n >= 2 else 1, int(math.ceil(n * f - 1e-9)))
    if k >= n:
        return items
    idx = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)}) if k > 1 else [n // 2]
    return [items[i] for i in idx]


def lod_segs(n, radius):
    """The toolkit's own rule for round sections, for rings you build by hand: `n` at lod 0, at
    lod 1 / 2 the fewest segments that keep the faceting of a circle of `radius` (m) under
    4 mm / 18 mm (never more than n, never under 4 / 3).
        seg = lod_segs(32, r)
        ring = [(r * cos(2 * pi * i / seg), r * sin(2 * pi * i / seg)) for i in range(seg)]"""
    return _segs(n, radius)


def lod_path(pts, closed=False, k=1.0):
    """A hand-built profile, outline or path with only the points this level needs: the list itself at
    lod 0; at lod 1 / 2 the points that keep the line within 4 mm / 18 mm (times k) of the full one. The two
    ends (of a ring: at least three points) always stay. For rings and lofts you mesh yourself - lathe() and
    sweep() already do this to what they are given.
        prof = lod_path(moulding_profile)                      # [(r, z), ...] or [(x, y, z), ...]
        rings = [[plan(f, d, z) for d, z in prof] for f in stations]"""
    L = _LOD["level"]
    if L == 0 or len(pts) <= (3 if closed else 2):
        return pts
    idx = _dp_keep([tuple(float(c) for c in p) for p in pts], TOL[L] * k, closed)
    return [pts[i] for i in idx]


def lod_budget():
    """Triangles this level of the asset may have (40 % / 10 % of what its lod 0 came to with these params),
    for generators that fill a triangle target adaptively. None at lod 0 and inside a use()-composed asset.
        target = lod_budget() or TRI_TARGET"""
    if lod() == 0 or _LOD["scope"]:
        return None
    return _LOD.get("budget")


def lod_memo(key, fn):
    """A measurement every level must agree on: fn() is evaluated on the full model (lod 0) and that same value
    comes back at lod 1 / 2 without calling fn. For whatever is read back from geometry the level changes and then
    places or decides other things - a ray cast on a roof, a measured bound, a triangle count that decides
    whether optional dressing is added.
        z = lod_memo(("roof_z", round(x, 4), round(y, 4)), lambda: ray_z(roof, x, y))
        room = lod_memo("room_for_dormers", lambda: scene_tris() < CAP)
    key: hashable and unique within the asset (give every call its own: include what the value depends on).
    The value must be plain data (numbers, strings, tuples / lists, arrays, Vectors), not Blender objects."""
    import copy
    k = "memo|" + "/".join(_LOD["scope"]) + "|" + repr(key)
    if _LOD["rec"] is not None:
        v = fn()
        try:
            _LOD["rec"][k] = copy.deepcopy(v)
        except Exception:
            _LOD["rec"][k] = v
        return v
    if _LOD["play"] is not None:
        if k in _LOD["play"]:
            v = _LOD["play"][k]
            try:
                return copy.deepcopy(v)
            except Exception:
                return v
        note = "lod_memo(%r) was not measured at lod 0: measured on this level instead" % (key,)
        if note not in _LOD["notes"] and len(_LOD["notes"]) < 40:
            _LOD["notes"].append(note)
    return fn()


class lod_upto:
    """Detail that exists only up to a level: the block runs at EVERY level (so random draws,
    sockets, lights and parts made inside are identical at every level), then the mesh objects
    created inside are emptied when lod() is above `level`.
        with lod_upto(0):            # full model only
            rivets = instances_to_mesh("rivets", bolt("rv", 0.006), mats)
        with lod_upto(1):            # gone at lod 2
            glazing_bars(...)
    Geometry with a glowing material (emissive, neon, window_lit, glass_*) is never emptied.
    Use a plain `if lod() <= n:` instead only when the block draws no random numbers, makes no
    sockets / parts and is expensive to build."""

    def __init__(self, level=0):
        self.level = level
        self.before = None

    def __enter__(self):
        if lod() > self.level:
            self.before = {_uid(o) for o in bpy.data.objects}
        return self

    def __exit__(self, et, ev, tb):
        if et is None and self.before is not None:
            for o in [o for o in bpy.data.objects if _uid(o) not in self.before]:
                _empty(o)
        self.before = None
        return False


def lod_drop(objs, above=0):
    """Function form of lod_upto: empty these objects (one or a list, nested lists allowed) when
    lod() is above `above`. Returns objs. Emptied objects stay valid (join / instances_to_mesh /
    part() keep working); glowing geometry is never emptied."""
    if lod() > above:
        for o in _flat(objs):
            _empty(o)
    return objs


def lod_faces(objs, keep=None, drop=None, above=0, cos=0.5):
    """Delete the faces nobody sees at the cheaper levels (backs, undersides, faces buried in a wall):
    when lod() is above `above`, in these mesh objects (one or a list, nested lists allowed)
        keep=(0, -1, 0)                only faces that look along the direction stay (a box becomes a card)
        drop=[(0, 0, -1), (0, 1, 0)]   faces that look along a direction go (underside and back of a box)
    Directions are asset-space vectors (one or a list) at the moment of the call - place / rotate the
    object first; a face looks along d when the cosine between its normal and d is above `cos`.
    Modifiers are applied first. Returns objs.
        lod_faces(box("bar", (0.5, 0.02, 0.016), loc=...), keep=(0, -1, 0))    # glazing bar: its front
        lod_faces(plinth, drop=(0, 0, -1), above=1)                             # no underside at lod 2"""
    if lod() <= above or (keep is None and drop is None):
        return objs
    import bmesh
    from mathutils import Vector
    from .core import world_matrix, apply_modifiers

    def dirs(d):
        if d is None:
            return []
        if isinstance(d, Vector) or not isinstance(d[0], (list, tuple, Vector)):
            d = [d]
        return [Vector(v).normalized() for v in d]

    kd, dd = dirs(keep), dirs(drop)
    for o in _flat(objs):
        try:
            if o is None or o.type != 'MESH':
                continue
        except ReferenceError:
            continue
        if o.modifiers:
            apply_modifiers(o)
        if len(o.data.polygons) == 0:
            continue
        if o.data.users > 1:
            o.data = o.data.copy()
        nm = world_matrix(o).to_3x3().inverted_safe().transposed()
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.normal_update()
        gone = []
        for f in bm.faces:
            n = (nm @ f.normal).normalized()
            if (kd and not any(n.dot(d) > cos for d in kd)) or any(n.dot(d) > cos for d in dd):
                gone.append(f)
        if gone:
            bmesh.ops.delete(bm, geom=gone, context='FACES')
            bm.to_mesh(o.data)
        bm.free()
    return objs


class lod_as:
    """Model a block at another level than the build's: `with lod_as(0):` keeps something exactly
    as in the full model (dial numerals, lettering, the one detail that carries the asset's
    identity); `with lod_as(lod() + 1):` builds hidden / back / inside parts one level cheaper.
    Has no effect in the lod 0 build."""

    def __init__(self, level):
        self.level = max(0, min(2, int(level)))
        self.prev = 0

    def __enter__(self):
        self.prev = _LOD["level"]
        if _LOD["build"] > 0:
            _LOD["level"] = self.level
        return self

    def __exit__(self, et, ev, tb):
        _LOD["level"] = self.prev
        return False


# ----------------------------------------------------------------------------
# Internals used by the toolkit primitives
# ----------------------------------------------------------------------------

IDENT_IDS = 50_000_000      # build-counter range of the stand-ins (far above anything a build() reaches)


def lod_standin(objs, mid=None, far=None):
    """An IDENTITY PART of an asset that has no META["lod"] (clock hands and hour marks, a shop's emblem, the
    pointer of a weathervane ...) and, natively, its cheap stand-ins for the reduced levels. The full model is not
    touched: build() still runs once, `objs` are built as always and only get a tag.

      objs      the mesh objects of the part as modelled for lod 0 (one or a list, nested lists allowed). Call
                this BEFORE they are joined / instanced: the tag travels through join() and instances_to_mesh().
      mid, far  what stands in their place at lod 1 / lod 2:
                  None                  nothing: the originals stay as modelled and are protected (see below)
                  a callable -> objs    builds the stand-in. Called right here, at lod-0 time, with the build
                                        counter set aside, so no name / UV offset of the full model shifts. It
                                        must build from NUMBERS: draw nothing from rng (nor from a fork), use the
                                        materials the originals use (no new mat() variants), and return finished
                                        mesh objects in WORLD position - originals that are instanced need their
                                        stand-in instanced by the same matrices. The objects are taken out of
                                        the scene and kept for the reducer.
                  objects               the same, already built (their toolkit calls then do move the counter)
                  () / a callable -> () the part is dropped whole at that level

    At lod 1 / lod 2 tk.reduce removes the tagged originals of a group that has a stand-in for the level and puts
    the stand-in there; stand-ins and kept originals are PROTECTED like lettering: never welded, patched,
    collapsed, dropped or replaced by a prism. Their triangles count toward the level's budget (the rest of the
    asset gets what is left), so keep them lean. The clearance rule to cards holds for them as for everything: a
    hand must not lie within 4 mm of the dial card (the depth-fight gate of the level measures it). Lettering
    (text()) inside a group that is swapped at a level is not held to the lettering gate at that level.
    In every exported GLB (lod 0, the decimated copies, both levels) the vertices of tagged originals and of
    stand-ins carry the float vertex attribute _TSQ_KEEP = 1.0 (see docs/LOD.md), so a later simplification
    (the web bake's far pass) can lock them.

    No-op in a level-aware asset (META["lod"]: model the levels with lod() there) and above lod 0. Inside a
    composed child (tk.asm.use) no stand-in is made: the originals are kept and protected. Returns objs.

        hands = [hour, minute]
        lod_standin(hands, mid=lambda: flat_hands(1.0), far=lambda: flat_hands(2.3))
        lod_standin(minute_ticks, mid=(), far=())"""
    lst = [o for o in _flat(objs) if o is not None]
    if not lst or _LOD["rec"] is not None or _LOD["play"] is not None or _LOD["level"] != 0:
        return objs
    _STATE["ident_n"] = _STATE.get("ident_n", 0) + 1
    g = _STATE["ident_n"]
    for o in lst:
        o["tsq_id"] = g
    modes = []
    for level, what in ((1, mid), (2, far)):
        if what is None or _STATE.get("child"):
            modes.append("keep")
        else:
            _standin(g, level, what, lst[0])
            modes.append("swap")
    _STATE.setdefault("ident", {})[g] = modes
    return objs


def lod_keep(objs):
    """lod_standin(objs) without stand-ins: these objects are an identity part of the asset - the reducer keeps
    them as modelled at lod 1 and lod 2 (never welded, patched, collapsed, dropped or replaced by a prism; their
    triangles count toward the budget) and their vertices carry _TSQ_KEEP in the exported GLBs. Returns objs."""
    return lod_standin(objs)


def _standin(g, level, what, like):
    from mathutils import Matrix
    from .core import world_matrix
    from .surf import realize
    c0 = _STATE["counter"]
    _STATE["counter"] = IDENT_IDS + 100_000 * (2 * g + level)
    try:
        made = what() if callable(what) else what
        objs = [o for o in _flat(made) if o is not None and o.type == 'MESH']
        realize(objs)
    finally:
        _STATE["counter"] = c0
    k = 0
    for o in objs:
        if len(o.data.polygons) == 0:
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        mw = world_matrix(o).copy()
        o.parent = None
        o.matrix_parent_inverse = Matrix.Identity(4)
        o.matrix_basis = mw
        for c in list(o.users_collection):
            c.objects.unlink(o)
        o.use_fake_user = True
        o.name = f"~standin{g}.{level}.{k}"
        o.data.name = o.name
        if "tsq_id" in o:
            del o["tsq_id"]
        o["tsq_sid"] = g
        o["tsq_slevel"] = level
        for key in ("tsq_part", "tsq_pivot", "tsq_part_props"):
            if key in like:
                o[key] = like[key]
        _face_tag(o.data, g)
        k += 1


def _face_tag(me, g):
    n = len(me.polygons)
    if not n:
        return
    a = me.attributes.get("tsq_id")
    if a is not None and (a.domain != 'FACE' or a.data_type != 'INT'):
        me.attributes.remove(a)
        a = None
    if a is None:
        a = me.attributes.new("tsq_id", 'INT', 'FACE')
    a.data.foreach_set("value", np.full(n, int(g), dtype=np.int32))


def _tag_ident(o):
    """lod_standin / lod_keep: the group number of a tagged object (custom property tsq_id) written as the face
    attribute tsq_id of its realized mesh - the form join() and instances_to_mesh() carry on. Not exported."""
    try:
        if o.type != 'MESH' or "tsq_id" not in o:
            return
    except ReferenceError:
        return
    _face_tag(o.data, int(o["tsq_id"]))


def _uid(o):
    try:
        return o.session_uid
    except AttributeError:  # pragma: no cover
        return o.as_pointer()


def _flat(objs):
    if objs is None:
        return
    if isinstance(objs, (list, tuple, set)):
        for o in objs:
            yield from _flat(o)
    else:
        yield objs


def _glows(o):
    mats = getattr(o.data, "materials", None) or []
    return any(m is not None and str(m.get("tsq_lib", "")).startswith(GLOW) for m in mats)


def _empty(o):
    """Remove an object's geometry but keep the object (later join / instancing / part() still work)."""
    try:
        kind = o.type
    except (ReferenceError, AttributeError):
        return
    if kind in ('CURVE', 'FONT', 'SURFACE'):
        if not _glows(o):
            bpy.data.objects.remove(o, do_unlink=True)
        return
    if kind != 'MESH' or _glows(o) or (len(o.data.vertices) == 0 and len(o.modifiers) == 0):
        return
    # a new empty mesh (never clear the old one: placed components share their mesh data). The UV / smoothing
    # declarations and the presence of a UV layer stay as they are, so realize() draws this object's texture
    # offset at exactly the moment it does in the full model and nothing after it shifts.
    old = o.data
    me = bpy.data.meshes.new(old.name)
    if old.uv_layers:
        me.uv_layers.new(name="UVMap")
    for m in old.materials:
        me.materials.append(m)
    o.modifiers.clear()
    o.data = me
    o["_lod_empty"] = 1
    if old.users == 0:
        bpy.data.meshes.remove(old)


def _segs(n, radius):
    L = _LOD["level"]
    if L == 0 or isinstance(n, _XI):
        return n
    n = int(n)
    tol = TOL[L]
    r = abs(float(radius))
    if n <= 8 and r >= 2 * tol:
        return n            # few segments on something that size is a deliberate polygon (octagonal post, hex cap)
    lo = (0, 4, 3)[L]
    if r <= tol:
        N = lo
    else:
        N = max(lo, int(math.ceil(math.pi / math.acos(1.0 - tol / r) - 1e-9)))
    if N > 4 and N % 2:
        N += 1
    if n % 4 == 0 and N > 6 and N % 4:
        N += 4 - N % 4
    return max(3, min(n, N))


def _round_smooth(angle, n0, n, sweep=360.0):
    """A round section that was smooth-shaded with n0 segments stays smooth with the n it has at this level
    (its facets must not turn into hard edges: a hexagonal 'crystal' where the full model is turned)."""
    if n >= n0 or n < 3 or sweep / max(n0, 1) > angle:
        return angle
    return max(angle, sweep / n + 2.0)


def _half(n, lo=1):
    """Grid resolution of surface() / grid(): half at lod 1, a quarter at lod 2 (halving keeps features that
    repeat every 2 / 4 cells - scallops, folds, pleats - in phase)."""
    L = _LOD["level"]
    if L == 0 or isinstance(n, _XI):
        return n
    n = int(n)
    return max(min(lo, n), min(n, -(-n // (2 if L == 1 else 4))))


def _ring_keep(verts, closed, twist, uniform=False):
    """sweep(): indices of the cross-sections to keep at this level. verts (n, m, 3) are the rings of the full
    model; a ring is dropped when the surface between its neighbours stays within tolerance without it."""
    L = _LOD["level"]
    n, m = verts.shape[0], verts.shape[1]
    keep = _dp_keep(verts.reshape(n, -1) / math.sqrt(m), TOL[L] * 0.5, closed)
    if uniform and not closed and len(keep) == 2 and n > 2:
        # a straight path the author subdivided on purpose (step=...): it is about to be bent or displaced
        keep = sorted(set(range(0, n, 2 if L == 1 else 4)) | {n - 1})
    if not closed and len(twist) == n and abs(float(twist[-1] - twist[0])) > 1e-6:
        # twisted bars: never span more than a quarter turn between two rings (the faces would pinch)
        lim = math.radians(50.0 if L == 1 else 95.0)
        out = set(keep)
        for a, b in zip(keep[:-1], keep[1:]):
            k = int(math.ceil(abs(float(twist[b] - twist[a])) / lim - 1e-9))
            for j in range(1, k):
                out.add(a + int(round((b - a) * j / k)))
        keep = sorted(out)
    return keep


def _bevel(width, segments):
    """-> (width, segments) for this level; segments 0 = no bevel"""
    L = _LOD["level"]
    if L == 0 or isinstance(width, _XF) or isinstance(segments, _XI):
        return width, segments
    if width < BEVEL_MIN[L]:
        return 0.0, 0
    return width, 1


def _dp_keep(P, tol, closed=False):
    """Douglas-Peucker: indices of the points of polyline P (n, d) to keep so the simplified line
    stays within tol of the original. closed=True treats P as a ring (at least 3 points stay)."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    if n <= (3 if closed else 2) or tol <= 0:
        return list(range(n))
    keep = np.zeros(n, dtype=bool)

    def far(idx, i, j):
        A, B = P[idx[i]], P[idx[j]]
        Q = P[idx[i + 1:j]]
        d = B - A
        l2 = float(d @ d)
        if l2 < 1e-18:
            dist = np.linalg.norm(Q - A, axis=1)
        else:
            t = np.clip((Q - A) @ d / l2, 0.0, 1.0)
            dist = np.linalg.norm(Q - (A + t[:, None] * d), axis=1)
        k = int(np.argmax(dist))
        return i + 1 + k, float(dist[k])

    def run(idx):
        stack = [(0, len(idx) - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            m, dm = far(idx, i, j)
            if dm > tol:
                keep[idx[m]] = True
                stack.append((i, m))
                stack.append((m, j))

    if closed:
        f = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
        if f == 0:
            return [0, n // 3, 2 * n // 3]
        keep[0] = keep[f] = True
        a = np.arange(0, f + 1)
        b = np.concatenate([np.arange(f, n), [0]])
        run(a)
        run(b)
        if keep.sum() < 3:
            cand = [far(ix, 0, len(ix) - 1) for ix in (a, b) if len(ix) > 2]
            if cand:
                ix = a if (len(a) > 2 and (len(b) <= 2 or far(a, 0, len(a) - 1)[1] >= far(b, 0, len(b) - 1)[1])) else b
                keep[ix[far(ix, 0, len(ix) - 1)[0]]] = True
    else:
        keep[0] = keep[-1] = True
        run(np.arange(n))
    return np.nonzero(keep)[0].tolist()


def _simplify2d(pts, closed=True, k=0.5):
    """Simplify a 2D outline for this level (tolerance = k * TOL). Returns the list unchanged at lod 0."""
    L = _LOD["level"]
    if L == 0 or len(pts) <= (4 if closed else 2):
        return pts
    idx = _dp_keep([(float(p[0]), float(p[1])) for p in pts], TOL[L] * k, closed)
    return [pts[i] for i in idx]


# ----------------------------------------------------------------------------
# Build-runner hooks (tools/bl_build.py)
# ----------------------------------------------------------------------------

def _begin(level, record=None, auto=False, where=False, budget=None):
    """Start a level-aware build() run: level 0 records the toolkit's stream ids, levels 1 / 2 replay them.
    auto: composed assets without META["lod"] are modelled at the level too (toolkit scaling only) instead of
    being built in full and reduced. where: join() notes what went into each joined object (tools/tsq lod)."""
    _LOD.update(level=level, build=level, occ={}, scope=[], sigs={}, notes=[], auto=bool(auto),
                where={} if where else None, budget=budget,
                rec={} if level == 0 else None, play=None if level == 0 else dict((record or {}).get("ids") or {}))


def _sig(rng):
    return hashlib.sha1(repr(rng.getstate()).encode()).hexdigest()[:12]


def _end(rng=None):
    """Finish the run -> {"ids", "sigs", "notes"}; the toolkit is back in its plain (level 0, untracked) state."""
    if rng is not None:
        _LOD["sigs"][""] = _sig(rng)
    out = {"ids": _LOD["rec"], "sigs": dict(_LOD["sigs"]), "notes": list(_LOD["notes"])}
    _LOD.update(level=0, build=0, occ={}, scope=[], sigs={}, notes=[], rec=None, play=None, auto=False, where=None, budget=None)
    return out


def _base(name):
    import re
    return re.sub(r"[_.]?\d+$", "", re.sub(r"\.\d{3,}$", "", name)) or name


def _ntris(me):
    n = len(me.polygons)
    if not n:
        return 0
    lt = np.empty(n, dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    return int((lt - 2).sum())


def _where_src(objs):
    """join(): {source name: triangles} of the objects about to be merged (already-joined ones expanded)"""
    w = _LOD.get("where")
    if w is None:
        return None
    d = {}
    for o in objs:
        t = _ntris(o.data)
        sub = w.pop(o.name, None)
        tot = sum(sub.values()) if sub else 0
        if tot:
            for k, v in sub.items():
                d[k] = d.get(k, 0) + v * t / tot
        else:
            k = _base(o.name)
            d[k] = d.get(k, 0) + t
    return d


def _tracked():
    return _LOD["rec"] is not None or _LOD["play"] is not None


def _scope_push(asset_id, phash):
    """use(): streams of a composed asset are keyed under its own scope (id + params + n-th such build)"""
    k = "/".join(_LOD["scope"]) + "|use|" + asset_id + ":" + phash
    n = _LOD["occ"].get(k, 0)
    _LOD["occ"][k] = n + 1
    _LOD["scope"].append(f"{asset_id}:{phash}#{n}")


def _scope_pop(rng=None):
    if rng is not None:
        _LOD["sigs"]["/".join(_LOD["scope"])] = _sig(rng)
    _LOD["scope"].pop()


def _child_plain(mod):
    """use(): True when this composed asset does not model its own levels and the build is at level 1 / 2 -
    it is then built in full and reduced (islands + collapse) like tk.final does for whole assets."""
    m = (getattr(mod, "META", None) or {}).get("lod")      # "far": only lod 2 is modelled (tools/bl_build.py)
    return (_LOD["level"] > 0 and _LOD["play"] is not None and not _LOD.get("auto")
            and (not m or (m == "far" and _LOD["level"] < 2)))


def _reduce(obj, level, meta=None):
    """Reduced copy of a composed, joined mesh for `level` (replaces obj). tk.reduce.reduce_child takes the
    triangles from where nobody sees them first (hidden faces, thin bars as prisms, small loose pieces, fields of
    stones or tiles welded) and collapses only what is left over; the plain collapse of tk.final._make_lod is the
    fallback when that fails. meta: META of the composed asset (its mount says which sides can be seen)."""
    name = obj.name
    before = set(bpy.data.objects)
    bak = obj.copy()                # not linked to the scene: the model as it came, for the fallback
    bak.data = obj.data.copy()
    try:
        from . import reduce as _red
        o = _red.reduce_child(obj, level, meta)
        bpy.data.meshes.remove(bak.data)        # (removes bak with it)
        return o
    except Exception as e:
        _LOD["notes"].append(f"use(): {_base(name)} was collapsed instead of reduced at lod {level} ({type(e).__name__}: {e})")
        for x in list(bpy.data.objects):
            if x not in before and x is not bak:
                me = x.data
                bpy.data.objects.remove(x, do_unlink=True)
                if me is not None and me.users == 0:
                    bpy.data.meshes.remove(me)
        try:
            me = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
        except ReferenceError:
            pass
        from .core import _link
        _link(bak)
        bak.name = name
        obj = bak
    from .final import _make_lod
    bb = np.array([c[:] for c in obj.bound_box])
    k = float(np.clip(np.linalg.norm(bb.max(0) - bb.min(0)) / 6.0, 0.6, 2.5))
    ms, ratio = ((0.035 * k, 0.4), (0.14 * k, 0.1))[level - 1]
    o = _make_lod(obj, ms, ratio)
    name, me = obj.name, obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if me.users == 0:
        bpy.data.meshes.remove(me)
    o.name = name
    return o


def _bbox(comp):
    """use(): a Component reports the bounds of the FULL model at every level (generators size openings and
    place neighbours from comp.size / comp.min / comp.max - a cheaper level must not move them)."""
    key = "bbox|" + "/".join(_LOD["scope"])
    if _LOD["rec"] is not None:
        _LOD["rec"][key] = (tuple(comp.min), tuple(comp.max))
    elif _LOD["play"] is not None and key in _LOD["play"]:
        from mathutils import Vector
        mn, mx = _LOD["play"][key]
        comp.min, comp.max = Vector(mn), Vector(mx)
        comp.size = comp.max - comp.min


def _rng_drift(rec0, rec):
    """Scopes whose generator rng ended in another state than at lod 0 (something drew a different
    number of random values at this level, so placements may have shifted)."""
    a, b = rec0.get("sigs", {}), rec.get("sigs", {})
    return sorted(k or "<this asset>" for k in a if k in b and a[k] != b[k])
