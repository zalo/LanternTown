"""tk.surf - library materials, world-scale UVs (meters), shading flags and vertex masks.

UV convention: UVs are stored in METERS. The runtime divides by each library material's tile size,
so every surface gets a physically consistent texel density.  For directional materials (wood,
brushed metal, fabric warp, rope) the grain runs along +U.

UV / smoothing are *declared* per object and resolved in realize() after modifiers are applied, so
bevels and booleans never leave you with stale UVs.
"""
import bpy
import bmesh
import json
import hashlib
import os
import numpy as np
from math import radians, pi
from mathutils import Vector

from .core import hex_rgb, _auto_rng, world_matrix, _link

# ----------------------------------------------------------------------------
# Library materials
# ----------------------------------------------------------------------------

_CATALOG = None


def material_catalog():
    """{name: {...}} loaded from docs/materials.json (the list of valid library material names)."""
    global _CATALOG
    if _CATALOG is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "materials.json")
        try:
            with open(path) as f:
                data = json.load(f)
            _CATALOG = {m["name"]: m for m in data["materials"]}
        except Exception as e:  # pragma: no cover
            print("WARN: cannot load material catalog:", e)
            _CATALOG = {}
    return _CATALOG


def mat(lib, tint=None, **params):
    """Reference a library material by name -> bpy material.

    lib    : library material name (see docs/materials.json / `tools/tsq mats`).
    tint   : '#RRGGBB' (sRGB) or linear (r,g,b). Multiplies the albedo of tintable materials
             (paint, plaster, fabric, glass colour, emissive colour...). None = material default.
    params : optional runtime shader knobs, all floats:
               rough   roughness multiplier (default 1)
               glow    emissive strength for emissive/glass/neon materials (nits-ish, 1..50)
               grime   gain on the grime vertex mask (default 1)
               wear    gain on the edge-wear vertex mask (default 1)
               special gain on the special mask (moss / rust / fade, material dependent)
               scale   texture scale multiplier (default 1; 2 = pattern twice as small)
               opacity for glass-like materials (0..1)
    The same (lib, tint, params) always returns the same material, so draw calls are shared.
    """
    cat = material_catalog()
    if cat and lib not in cat:
        raise ValueError(f"unknown library material {lib!r}. Valid: {sorted(cat)}")
    t = None if tint is None else [round(float(c), 4) for c in hex_rgb(tint)]
    params = {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in sorted(params.items())}
    if t is None and not params:
        name = lib
    else:
        h = hashlib.sha1(json.dumps([lib, t, params], sort_keys=True).encode()).hexdigest()[:8]
        name = f"{lib}~{h}"
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m["tsq_lib"] = lib
    if t is not None:
        m["tsq_tint"] = t
    if params:
        m["tsq_params"] = json.dumps(params)
    # viewport-ish preview colour only (the real look comes from the runtime library material)
    prev = cat.get(lib, {}).get("preview", [0.6, 0.6, 0.6]) if cat else [0.6, 0.6, 0.6]
    col = [prev[i] * (t[i] if t else 1.0) for i in range(3)]
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (col[0], col[1], col[2], 1.0)
    return m


def _as_mat(m):
    return mat(m) if isinstance(m, str) else m


def assign(obj, material, slot=None):
    """Assign a material (bpy material or library name) to the whole object (or a slot index)."""
    if isinstance(obj, (list, tuple)):
        for o in obj:
            assign(o, material, slot)
        return obj
    material = _as_mat(material)
    if obj.type != 'MESH' and not hasattr(obj.data, "materials"):
        return obj
    mats = obj.data.materials
    if slot is None:
        if len(mats) == 0:
            mats.append(material)
        else:
            mats[0] = material
            if obj.type == 'MESH' and len(mats) > 1:
                # collapse to a single material
                n = len(obj.data.polygons)
                obj.data.polygons.foreach_set("material_index", np.zeros(n, dtype=np.int32))
                while len(mats) > 1:
                    mats.pop(index=len(mats) - 1)
    else:
        while len(mats) <= slot:
            mats.append(material)
        mats[slot] = material
    return obj


def assign_faces(obj, material, predicate):
    """Assign material to faces where predicate(center: Vector, normal: Vector) is True (local space).
    Adds a material slot as needed. Call AFTER the base material is assigned.  Works on the
    current base mesh (apply modifiers first if you need the bevelled result)."""
    material = _as_mat(material)
    me = obj.data
    if len(me.materials) == 0:
        me.materials.append(mat("plain"))
    idx = None
    for i, m in enumerate(me.materials):
        if m == material:
            idx = i
    if idx is None:
        me.materials.append(material)
        idx = len(me.materials) - 1
    n = 0
    for poly in me.polygons:
        if predicate(poly.center, poly.normal):
            poly.material_index = idx
            n += 1
    return n


# ----------------------------------------------------------------------------
# Smoothing declarations
# ----------------------------------------------------------------------------

def smooth(obj, angle=40.0):
    """Smooth shading; edges sharper than `angle` degrees stay sharp (resolved after modifiers)."""
    if isinstance(obj, (list, tuple)):
        for o in obj:
            smooth(o, angle)
        return obj
    obj["tsq_smooth"] = float(angle)
    return obj


def flat(obj):
    obj["tsq_smooth"] = 0.0
    return obj


def _apply_smooth(obj):
    me = obj.data
    spec = obj.get("tsq_smooth", 40.0)
    if spec == "keep" or len(me.polygons) == 0:
        return
    ang = float(spec)
    n = len(me.polygons)
    if ang <= 0.0:
        me.polygons.foreach_set("use_smooth", np.zeros(n, dtype=bool))
        obj["tsq_smooth"] = "keep"
        return
    me.polygons.foreach_set("use_smooth", np.ones(n, dtype=bool))
    if ang < 179.0:
        bm = bmesh.new()
        bm.from_mesh(me)
        lim = radians(ang)
        for e in bm.edges:
            if len(e.link_faces) == 2:
                e.smooth = e.calc_face_angle(0.0) <= lim
            else:
                e.smooth = True
        bm.to_mesh(me)
        bm.free()
    obj["tsq_smooth"] = "keep"


# ----------------------------------------------------------------------------
# UV declarations (resolved in realize())
# ----------------------------------------------------------------------------

def _decl(obj, spec):
    if isinstance(obj, (list, tuple)):
        for o in obj:
            _decl(o, dict(spec))
        return obj
    if spec.get("offset") == "auto":
        r = _auto_rng("uv", "decl:" + obj.name)
        spec["offset"] = [round(r.uniform(0, 7.0), 3), round(r.uniform(0, 7.0), 3)]
    elif spec.get("offset") is None:
        spec["offset"] = [0.0, 0.0]
    obj["tsq_uv"] = json.dumps(spec)
    return obj


def uv_box(obj, scale=1.0, offset="auto", rot90=False):
    """Box (tri-planar) projection in the object's local axes, in meters. V is 'up' (local +Z) on
    vertical faces so masonry courses, plaster streaks etc. stay upright.  rot90 swaps U/V.
    offset='auto' gives each object a random texture offset so repeated parts don't look cloned;
    pass (u, v) in meters or None for an exact placement."""
    return _decl(obj, {"mode": "box", "scale": scale, "offset": offset, "rot90": rot90})


def uv_grain(obj, axis='X', scale=1.0, offset="auto"):
    """Directional projection: U runs along local `axis` ('X'|'Y'|'Z' or a 3-vector) on every face
    (use for planks, beams, bars: wood grain and brushed streaks follow U). Faces perpendicular to
    the axis (end grain) get a planar projection."""
    if isinstance(axis, str):
        axis = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[axis.upper()]
    return _decl(obj, {"mode": "grain", "axis": list(map(float, axis)), "scale": scale, "offset": offset})


def uv_cyl(obj, axis='Z', scale=1.0, offset="auto", radius=None):
    """Cylindrical projection around local `axis`: U = arc length (angle * radius), V = along axis.
    radius=None uses the object's mean radius. Caps get a planar projection."""
    return _decl(obj, {"mode": "cyl", "axis": axis.upper(), "scale": scale, "offset": offset, "radius": radius})


def uv_planar(obj, u_axis=(1, 0, 0), v_axis=(0, 1, 0), scale=1.0, offset=None):
    """Planar projection with explicit local-space U and V directions (meters)."""
    return _decl(obj, {"mode": "planar", "u": list(map(float, u_axis)), "v": list(map(float, v_axis)),
                       "scale": scale, "offset": offset})


def uv_keep(obj):
    """Keep the UVs the object already has (lathe / sweep / surface generate proper UVs)."""
    obj["tsq_uv"] = "keep"
    return obj


def uv_transform(obj, scale=(1.0, 1.0), offset=(0.0, 0.0), rot_deg=0.0):
    """Post-transform existing UVs (applies immediately; object must already have UVs)."""
    me = obj.data
    if not me.uv_layers:
        return obj
    n = len(me.loops)
    uv = np.empty(n * 2, dtype=np.float32)
    me.uv_layers[0].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    if rot_deg:
        a = radians(rot_deg)
        c, s = np.cos(a), np.sin(a)
        uv = np.stack([uv[:, 0] * c - uv[:, 1] * s, uv[:, 0] * s + uv[:, 1] * c], axis=1)
    uv = uv * np.asarray(scale, dtype=np.float32) + np.asarray(offset, dtype=np.float32)
    me.uv_layers[0].data.foreach_set("uv", uv.astype(np.float32).ravel())
    return obj


def _loop_arrays(obj):
    me = obj.data
    nv, nl, npoly = len(me.vertices), len(me.loops), len(me.polygons)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) * np.asarray(obj.scale, dtype=np.float32)
    lv = np.empty(nl, dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    pn = np.empty(npoly * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", pn)
    pn = pn.reshape(-1, 3)
    lt = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    ls = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    # per-loop polygon index (Blender 4.x stores polygon loops contiguously, in polygon order)
    lp = np.repeat(np.arange(npoly, dtype=np.int32), lt)
    sc = np.asarray(obj.scale, dtype=np.float32)
    n = pn[lp] / np.where(np.abs(sc) < 1e-9, 1.0, sc)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln < 1e-12, 1.0, ln)
    return me, co[lv], n, lp, ls, lt


def _apply_uv(obj):
    me = obj.data
    spec = obj.get("tsq_uv", None)
    if spec == "keep" and me.uv_layers:
        return
    if spec is None or spec == "keep":
        r = _auto_rng("uv", obj.name)
        spec = {"mode": "box", "scale": 1.0, "offset": [r.uniform(0, 7), r.uniform(0, 7)], "rot90": False}
    else:
        spec = json.loads(spec)
    if len(me.polygons) == 0:
        return
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    me, p, n, lp, ls, lt = _loop_arrays(obj)
    mode = spec["mode"]
    if mode == "box":
        ax = np.argmax(np.abs(n), axis=1)
        sgn = np.sign(n[np.arange(len(n)), ax])
        sgn[sgn == 0] = 1
        u = np.where(ax == 0, p[:, 1] * sgn, np.where(ax == 1, -p[:, 0] * sgn, p[:, 0]))
        v = np.where(ax == 2, p[:, 1] * sgn, p[:, 2])
        if spec.get("rot90"):
            u, v = v, -u
    elif mode == "grain":
        g = np.asarray(spec["axis"], dtype=np.float32)
        g = g / np.linalg.norm(g)
        d = np.abs(n @ g)
        t = np.cross(n, g)
        tl = np.linalg.norm(t, axis=1, keepdims=True)
        t = t / np.where(tl < 1e-6, 1.0, tl)
        u = p @ g
        v = np.einsum("ij,ij->i", p, t)
        # end-grain faces: planar projection on the two axes perpendicular to g
        end = d > 0.92
        if end.any():
            a = np.array([0, 0, 1.0], dtype=np.float32) if abs(g[2]) < 0.9 else np.array([1.0, 0, 0], dtype=np.float32)
            b1 = np.cross(g, a)
            b1 /= np.linalg.norm(b1)
            b2 = np.cross(g, b1)
            u = np.where(end, p @ b1, u)
            v = np.where(end, p @ b2, v)
    elif mode == "cyl":
        ai = {"X": 0, "Y": 1, "Z": 2}[spec["axis"]]
        o1, o2 = [(1, 2), (2, 0), (0, 1)][ai]
        ang = np.arctan2(p[:, o2], p[:, o1])
        rad = np.sqrt(p[:, o1] ** 2 + p[:, o2] ** 2)
        r_ref = spec.get("radius") or float(np.mean(rad)) or 1.0
        # unwrap the seam per polygon
        pmax = np.full(len(ls), -1e9, dtype=np.float32)
        np.maximum.at(pmax, lp, ang)
        ang = np.where(pmax[lp] - ang > pi, ang + 2 * pi, ang)
        u = ang * r_ref
        v = p[:, ai]
        cap = np.abs(n[:, ai]) > 0.92
        u = np.where(cap, p[:, o1], u)
        v = np.where(cap, p[:, o2], v)
    elif mode == "planar":
        u = p @ np.asarray(spec["u"], dtype=np.float32)
        v = p @ np.asarray(spec["v"], dtype=np.float32)
    else:
        raise ValueError(mode)
    s = float(spec.get("scale", 1.0))
    off = spec.get("offset") or [0.0, 0.0]
    uv = np.stack([u * s + off[0], v * s + off[1]], axis=1).astype(np.float32)
    me.uv_layers[0].data.foreach_set("uv", uv.ravel())
    obj["tsq_uv"] = "keep"


# ----------------------------------------------------------------------------
# realize: modifiers -> smoothing -> UVs
# ----------------------------------------------------------------------------

def realize(objs):
    """Apply modifiers, resolve smoothing and UV declarations. Accepts one object or a list.
    Returns the object(s). Safe to call repeatedly. Bakes object *scale* into the mesh."""
    single = isinstance(objs, bpy.types.Object)
    lst = [objs] if single else [o for o in objs if o is not None]
    meshes = [o for o in lst if o.type == 'MESH']
    with_mods = [o for o in meshes if len(o.modifiers) > 0]
    if with_mods:
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        new = []
        for o in with_mods:
            ev = o.evaluated_get(dg)
            new.append(bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg))
        for o, me in zip(with_mods, new):
            old = o.data
            o.modifiers.clear()
            o.data = me
            if old.users == 0:
                bpy.data.meshes.remove(old)
    for o in meshes:
        if o.data.users > 1:
            o.data = o.data.copy()
        _apply_uv(o)
        _apply_smooth(o)
        if "_end_mat" in o:
            ai = int(o["_end_axis"])
            assign_faces(o, mat(o["_end_mat"]) if o["_end_mat"] not in bpy.data.materials else bpy.data.materials[o["_end_mat"]],
                         lambda c, n, ai=ai: abs(n[ai]) > 0.9)
            del o["_end_mat"]
    return objs


# ----------------------------------------------------------------------------
# Vertex masks (exported as COLOR_0 = [ao, grime, wear, special])
# ----------------------------------------------------------------------------

MASKS = ("grime", "wear", "special")


def paint(obj, grime=None, wear=None, special=None, mode="max"):
    """Paint per-vertex weathering masks (0..1). Each value is a float (uniform) or a vectorised
    callable fn(co, normal) -> array, where co / normal are (N,3) numpy arrays in WORLD space.

      grime   : dirt, soot, damp staining (darkens + roughens)        e.g. near the ground, under sills
      wear    : rubbed / chipped areas revealing lighter, smoother substrate   e.g. handles, tread centres
      special : material dependent - moss on stone/roof, rust on iron, verdigris on copper/brass,
                sun-fade on paint/fabric, wetness on water-adjacent stone

    mode: 'max' (default), 'add' or 'set'. Auto masks (AO, cavity grime, convex-edge wear, ground
    splash) are added on top at export, so only paint the *art-directed* story here.
    Modifiers are applied first so masks land on the final vertices."""
    if isinstance(obj, (list, tuple)):
        for o in obj:
            paint(o, grime, wear, special, mode)
        return obj
    if obj.type != 'MESH':
        return obj
    realize(obj)
    me = obj.data
    nv = len(me.vertices)
    if nv == 0:
        return obj
    need_geo = any(callable(v) for v in (grime, wear, special))
    co = nrm = None
    if need_geo:
        mw = np.array(world_matrix(obj), dtype=np.float64)
        c = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("co", c)
        c = c.reshape(-1, 3).astype(np.float64)
        co = c @ mw[:3, :3].T + mw[:3, 3]
        nn = np.empty(nv * 3, dtype=np.float32)
        me.vertices.foreach_get("normal", nn)
        nn = nn.reshape(-1, 3).astype(np.float64)
        nit = np.linalg.inv(mw[:3, :3]).T
        nrm = nn @ nit.T
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    for name, val in zip(MASKS, (grime, wear, special)):
        if val is None:
            continue
        attr = me.attributes.get("m_" + name)
        if attr is None:
            attr = me.attributes.new("m_" + name, 'FLOAT', 'POINT')
        cur = np.empty(nv, dtype=np.float32)
        attr.data.foreach_get("value", cur)
        new = np.asarray(val(co, nrm), dtype=np.float32) if callable(val) else np.full(nv, float(val), dtype=np.float32)
        if new.shape != (nv,):
            new = np.broadcast_to(new, (nv,)).astype(np.float32)
        if mode == "max":
            cur = np.maximum(cur, new)
        elif mode == "add":
            cur = cur + new
        else:
            cur = new
        attr.data.foreach_set("value", np.clip(cur, 0.0, 1.0).astype(np.float32))
    return obj


def height_ramp(z0, z1):
    """Mask helper: 1 at world height z0 fading to 0 at z1 (e.g. paint(o, grime=height_ramp(0, 0.5)))."""
    def fn(co, n):
        t = np.clip((co[:, 2] - z0) / (z1 - z0 if z1 != z0 else 1e-6), 0.0, 1.0)
        return 1.0 - t * t * (3 - 2 * t)
    return fn


def facing(direction=(0, 0, 1), power=1.0):
    """Mask helper: how much the surface faces `direction` (e.g. moss / dust on up-facing faces)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)

    def fn(co, n):
        return np.clip(n @ d, 0.0, 1.0) ** power
    return fn


def noise_mask(scale=1.0, seed=0, lo=0.35, hi=0.65):
    """Mask helper: smooth 3D value noise (world space, feature size ~`scale` m) remapped lo..hi -> 0..1.
    Multiply with other helpers through mask_mul()."""
    def fn(co, n):
        return _value_noise3(co / max(scale, 1e-6), seed, lo, hi)
    return fn


def mask_mul(*fns):
    """Combine mask helpers by multiplication (floats allowed)."""
    def fn(co, n):
        out = np.ones(len(co))
        for f in fns:
            out = out * (f(co, n) if callable(f) else float(f))
        return out
    return fn


def _hash3(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0xFFFFFF)


def _value_noise3(p, seed=0, lo=0.0, hi=1.0, octaves=3):
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros(len(p))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        q = p * (2 ** o)
        i = np.floor(q).astype(np.int64)
        f = q - i
        f = f * f * (3 - 2 * f)
        acc = np.zeros(len(p))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    acc += w * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed + o * 17)
        out += acc * amp
        tot += amp
        amp *= 0.5
    out /= tot
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
