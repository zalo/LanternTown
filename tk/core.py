"""tk.core - params, rng, scene/object basics (Blender 4.1, background mode, no bpy.ops in hot paths)."""
import bpy
import bmesh
import hashlib
import json
import math
import random
from math import radians, pi, sin, cos
from mathutils import Vector, Matrix, Euler

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


class P(dict):
    """Resolved parameters with attribute access: p.width == p['width']."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def resolve_params(schema, overrides=None):
    """Merge overrides into a PARAMS schema, coercing types and clamping ranges.

    schema entries: {"default": v, "min": a, "max": b, "choices": [...], "desc": "..."}
    A "seed" int param is always present (added if the schema omits it).
    """
    schema = dict(schema)
    schema.setdefault("seed", {"default": 1, "min": 0, "max": 2 ** 31 - 1, "desc": "random seed"})
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(schema)
    if unknown:
        raise ValueError(f"unknown params {sorted(unknown)}; valid: {sorted(schema)}")
    out = P()
    for k, spec in schema.items():
        d = spec["default"]
        v = overrides.get(k, d)
        if "choices" in spec and v not in spec["choices"]:
            raise ValueError(f"param {k}={v!r} not in {spec['choices']}")
        if isinstance(d, bool):
            v = bool(v)
        elif isinstance(d, int):
            v = int(round(v))
        elif isinstance(d, float):
            v = float(v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "min" in spec:
                v = max(spec["min"], v)
            if "max" in spec:
                v = min(spec["max"], v)
        out[k] = v
    return out


def params_hash(asset_id, params):
    s = json.dumps([asset_id, params], sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------
# RNG
# ----------------------------------------------------------------------------


class Rng(random.Random):
    """Seeded RNG. Use rng.fork('name') to get an independent, stable stream for a sub-part so
    that changing one part's random draws never reshuffles another part."""

    def __init__(self, seed=1):
        self._seed_key = str(seed)
        super().__init__(int(hashlib.sha1(self._seed_key.encode()).hexdigest()[:12], 16))

    def fork(self, name):
        return Rng(f"{self._seed_key}/{name}")

    def u(self, a=0.0, b=1.0):
        return self.uniform(a, b)

    def jitter(self, v, amount):
        """v +- amount (absolute)."""
        return v + self.uniform(-amount, amount)

    def vary(self, v, frac):
        """v * (1 +- frac)."""
        return v * (1.0 + self.uniform(-frac, frac))

    def chance(self, p):
        return self.random() < p

    def pick(self, seq):
        return seq[int(self.random() * len(seq)) % len(seq)]

    def weighted(self, pairs):
        """pairs: [(item, weight), ...]"""
        tot = sum(w for _, w in pairs)
        r = self.random() * tot
        for it, w in pairs:
            r -= w
            if r <= 0:
                return it
        return pairs[-1][0]

    def vec(self, amount):
        return Vector((self.uniform(-amount, amount), self.uniform(-amount, amount), self.uniform(-amount, amount)))


# global counter so helper-generated names / uv offsets are deterministic per build
# ident / ident_n / ident_text: tk.levels.lod_standin (identity groups of this build); child: depth of tk.asm.use()
_STATE = {"counter": 0, "seed": 1, "ident": {}, "ident_n": 0, "child": 0}


def _reset_state(seed=1):
    _STATE["counter"] = 0
    _STATE["seed"] = seed
    _STATE["ident"] = {}
    _STATE["ident_n"] = 0


def _next_id():
    _STATE["counter"] += 1
    return _STATE["counter"]


def _auto_rng(tag="", name=None):
    return Rng(f"{_STATE['seed']}/auto/{_auto_id(tag, name)}/{tag}")


# ----------------------------------------------------------------------------
# Level of detail (state only; the author tools live in tk/lod.py, the guide is docs/LOD.md)
# ----------------------------------------------------------------------------
# level : the level the toolkit models at right now (0 = the full model; lod_as() can change it for a block)
# build : the level this build() run is for
# rec   : lod-0 run of a level-aware build: {key: counter id} of every toolkit-side random stream / generated name
# play  : lod >= 1 run: the ids recorded at lod 0, replayed by key, so a part that is skipped or simplified at a
#         cheaper level cannot shift the UV offsets, plank warps, stone seeds or socket names of anything else
_LOD = {"level": 0, "build": 0, "rec": None, "play": None, "occ": {}, "scope": [], "sigs": {}, "notes": []}


def lod():
    """Level of detail being modelled: 0 = the full model (what `tools/tsq build` makes), 1 = mid
    (at most 40 % of the triangles of level 0), 2 = far (at most 10 %). build() runs once per level
    with the same params and seed; the toolkit already cuts its own tessellation at 1 and 2, the
    generator decides what else to skip or replace (`if lod() >= 1:`, lod_upto, lod_pick, lod_thin).
    See docs/LOD.md."""
    return _LOD["level"]


def _auto_key(tag, name):
    base = str(name or "")
    i = base.rfind(".")
    if i > 0 and len(base) - i >= 4 and base[i + 1:].isdigit():   # Blender's .001 de-duplication suffix
        base = base[:i]
    k = "/".join(_LOD["scope"]) + "|" + tag + "|" + base
    n = _LOD["occ"].get(k, 0)
    _LOD["occ"][k] = n + 1
    return f"{k}|{n}"


def _auto_id(tag="", name=None):
    """Next id of the build counter. Outside level-aware builds this is exactly _next_id()."""
    if _LOD["rec"] is None and _LOD["play"] is None:
        return _next_id()
    key = _auto_key(tag, name)
    if _LOD["play"] is not None:
        n = _LOD["play"].get(key)
        return n if n is not None else "L" + hashlib.sha1(key.encode()).hexdigest()[:10]
    n = _next_id()
    _LOD["rec"][key] = n
    return n


# ----------------------------------------------------------------------------
# Scene / object basics
# ----------------------------------------------------------------------------


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _link(obj, collection=None):
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def root(name):
    """Create the root Empty for a model."""
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = 'PLAIN_AXES'
    e.empty_display_size = 0.1
    return _link(e)


def empty(name, loc=(0, 0, 0), rot=None):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.05
    _link(e)
    set_loc_rot(e, loc, rot)
    return e


def parent(children, parent_obj, keep_transform=True):
    if isinstance(children, bpy.types.Object):
        children = [children]
    bpy.context.view_layer.update()
    for c in children:
        if c is None or c is parent_obj:
            continue
        mw = c.matrix_world.copy()
        c.parent = parent_obj
        if keep_transform:
            c.matrix_parent_inverse = parent_obj.matrix_world.inverted()
            c.matrix_world = mw
    return children


def select_only(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def mesh_object(name, verts, faces, edges=()):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], list(edges), [tuple(f) for f in faces])
    me.update(calc_edges=True)
    me.validate()
    return _link(bpy.data.objects.new(name, me))


def from_bmesh(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return _link(bpy.data.objects.new(name, me))


def set_loc_rot(obj, loc=None, rot=None, scale=None):
    """rot is Euler degrees (x, y, z)."""
    if loc is not None:
        obj.location = Vector(loc)
    if rot is not None:
        obj.rotation_euler = tuple(radians(a) for a in rot)
    if scale is not None:
        obj.scale = (scale, scale, scale) if isinstance(scale, (int, float)) else scale
    return obj


def place(obj, loc=None, rot=None, scale=None):
    """Alias of set_loc_rot that also works on lists of objects (applied to each)."""
    if isinstance(obj, (list, tuple)):
        for o in obj:
            set_loc_rot(o, loc, rot, scale)
        return obj
    return set_loc_rot(obj, loc, rot, scale)


def world_matrix(obj):
    """World matrix computed from the parent chain without needing a depsgraph update."""
    m = obj.matrix_basis.copy()
    if obj.parent is not None:
        m = world_matrix(obj.parent) @ obj.matrix_parent_inverse @ m
    return m


def apply_transform(obj, location=True, rotation=True, scale=True):
    """Bake the object's local transform into its mesh data (no bpy.ops)."""
    if obj.type != 'MESH':
        return obj
    loc, rot, sca = obj.matrix_basis.decompose()
    m = Matrix.Identity(4)
    if scale:
        m = Matrix.Diagonal((sca.x, sca.y, sca.z, 1.0)) @ m
    if rotation:
        m = rot.to_matrix().to_4x4() @ m
    if location:
        m = Matrix.Translation(loc) @ m
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    obj.data.transform(m)
    if m.determinant() < 0:
        obj.data.flip_normals()
    if location:
        obj.location = (0, 0, 0)
    if rotation:
        obj.rotation_euler = (0, 0, 0)
    if scale:
        obj.scale = (1, 1, 1)
    obj.data.update()
    return obj


def apply_modifiers(obj):
    """Evaluate the modifier stack into the mesh (no bpy.ops). Curves/text become meshes."""
    if obj.type not in ('MESH', 'CURVE', 'FONT', 'SURFACE'):
        return obj
    if obj.type == 'MESH' and len(obj.modifiers) == 0:
        return obj
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    if obj.type == 'MESH':
        obj.modifiers.clear()
        old = obj.data
        obj.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        return obj
    # curve / text -> new mesh object replacing the old one
    new = bpy.data.objects.new(obj.name, me)
    new.matrix_world = obj.matrix_world.copy()
    for k in obj.keys():
        new[k] = obj[k]
    _link(new)
    new.parent = obj.parent
    new.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
    new.matrix_basis = obj.matrix_basis.copy()
    name = obj.name
    bpy.data.objects.remove(obj, do_unlink=True)
    new.name = name
    return new


def delete(objs):
    if isinstance(objs, bpy.types.Object):
        objs = [objs]
    for o in objs:
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)


def duplicate(obj, name=None, linked=False):
    d = obj.copy()
    if obj.data is not None and not linked:
        d.data = obj.data.copy()
    d.name = name or obj.name + "_dup"
    return _link(d)


def mirror(obj, axis='X', name=None):
    """Duplicate obj mirrored across the plane through the world origin perpendicular to `axis`."""
    d = duplicate(obj, name)
    apply_modifiers(d)
    bpy.context.view_layer.update()
    mw = world_matrix(obj)
    s = Matrix.Diagonal((-1 if axis == 'X' else 1, -1 if axis == 'Y' else 1, -1 if axis == 'Z' else 1, 1))
    d.parent = None
    d.matrix_world = Matrix.Identity(4)
    d.data.transform(s @ mw)
    d.data.flip_normals()
    d.data.update()
    return d


def descendants(obj):
    out = []
    for c in obj.children:
        out.append(c)
        out.extend(descendants(c))
    return out


def world_bbox(objs):
    """(min Vector, max Vector) over mesh objects (list or a root)."""
    if isinstance(objs, bpy.types.Object):
        objs = [objs] + descendants(objs)
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    found = False
    for o in objs:
        if o.type != 'MESH' or len(o.data.vertices) == 0:
            continue
        mw = world_matrix(o)
        for c in o.bound_box:
            w = mw @ Vector(c)
            mn = Vector((min(mn.x, w.x), min(mn.y, w.y), min(mn.z, w.z)))
            mx = Vector((max(mx.x, w.x), max(mx.y, w.y), max(mx.z, w.z)))
            found = True
    if not found:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    return mn, mx


def hex_rgb(h):
    """'#RRGGBB' (sRGB) -> linear RGB tuple. Tuples/lists pass through unchanged (assumed linear)."""
    if not isinstance(h, str):
        return tuple(h)[:3]
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)

    def lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return tuple(lin(int(h[i:i + 2], 16)) for i in (0, 2, 4))


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smoothstep(e0, e1, x):
    if e0 == e1:
        return 0.0 if x < e0 else 1.0
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)
