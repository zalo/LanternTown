"""tk.final - merge, bake vertex masks (AO / cavity grime / edge wear), validate, export GLB + sidecar."""
import bpy
import bmesh
import json
import math
import os
import time
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from .core import world_matrix, _link, apply_modifiers
from .surf import realize, material_catalog, mat, _value_noise3
from .geo import join

# default triangle budgets per asset class (soft: over budget = warning, 2x over = error)
BUDGETS = {"prop_small": 6000, "prop": 16000, "furniture": 22000, "street_furniture": 26000, "component": 12000,
           "structure": 70000, "building": 160000, "landmark": 260000, "ground": 80000, "vegetation": 60000,
           "backdrop": 30000}


def _np_co(me):
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def _np_vn(me):
    n = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("normal", n)
    return n.reshape(-1, 3).astype(np.float64)


def _clean(obj, warnings):
    """Drop degenerate faces, loose verts/edges."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    n0 = len(bm.faces)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges[:])
    loose_e = [e for e in bm.edges if not e.link_faces]
    if loose_e:
        bmesh.ops.delete(bm, geom=loose_e, context='EDGES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    # ngons with many verts triangulate badly at export; do it here with a robust method
    big = [f for f in bm.faces if len(f.verts) > 4]
    if big:
        bmesh.ops.triangulate(bm, faces=big, quad_method='BEAUTY', ngon_method='BEAUTY')
    if len(bm.faces) != n0 and n0 - len(bm.faces) > 0:
        pass
    bm.to_mesh(me)
    bm.free()
    me.update()


def _tri_soup(objs):
    """World-space verts + triangles of all objects (for the AO BVH)."""
    verts, tris, off = [], [], 0
    for o in objs:
        me = o.data
        mw = np.array(world_matrix(o), dtype=np.float64)
        co = _np_co(me) @ mw[:3, :3].T + mw[:3, 3]
        me.calc_loop_triangles()
        t = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", t)
        verts.append(co)
        tris.append(t.reshape(-1, 3) + off)
        off += len(co)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    return np.vstack(verts), np.vstack(tris)


_HEMI = None


def _hemi_dirs(n):
    """Cosine-weighted hemisphere directions (z up), fixed low-discrepancy set."""
    global _HEMI
    if _HEMI is not None and len(_HEMI) == n:
        return _HEMI
    d = []
    g = (1 + 5 ** 0.5) / 2
    for i in range(n):
        u = (i + 0.5) / n
        phi = 2 * math.pi * ((i / g) % 1.0)
        r = math.sqrt(u)
        d.append((r * math.cos(phi), r * math.sin(phi), math.sqrt(max(0.0, 1 - u))))
    _HEMI = np.array(d)
    return _HEMI


def _bake_ao(obj, bvh, maxd, samples=16):
    me = obj.data
    mw = np.array(world_matrix(obj), dtype=np.float64)
    co = _np_co(me) @ mw[:3, :3].T + mw[:3, 3]
    nit = np.linalg.inv(mw[:3, :3]).T
    n = _np_vn(me) @ nit.T
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    nv = len(co)
    if nv > 400000:
        samples = 6
    elif nv > 180000:
        samples = 10
    hemi = _hemi_dirs(samples)
    # tangent frames
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    t = np.cross(a, n)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    b = np.cross(n, t)
    # per-vertex rotation of the sample set to decorrelate
    ang = (np.abs(co[:, 0] * 12.9898 + co[:, 1] * 78.233 + co[:, 2] * 37.719) * 43758.5453) % (2 * math.pi)
    ca, sa = np.cos(ang), np.sin(ang)
    t2 = t * ca[:, None] + b * sa[:, None]
    b2 = -t * sa[:, None] + b * ca[:, None]
    org = co + n * (0.0015 + maxd * 0.004)
    occ = np.zeros(nv)
    ray = bvh.ray_cast
    inv = 1.0 / maxd
    for k in range(samples):
        hx, hy, hz = hemi[k]
        d = t2 * hx + b2 * hy + n * hz
        dl = d.tolist()
        ol = org.tolist()
        acc = np.zeros(nv)
        for i in range(nv):
            hit = ray(ol[i], dl[i], maxd)
            if hit[0] is not None:
                acc[i] = 1.0 - hit[3] * inv
        occ += acc
    ao = 1.0 - occ / samples
    return np.clip(ao, 0.0, 1.0), co, n


def _curvature(me, co, n):
    """Signed mean curvature proxy per vertex (>0 convex), in 1/m."""
    ne = len(me.edges)
    if ne == 0:
        return np.zeros(len(co))
    ev = np.empty(ne * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    d = co[ev[:, 1]] - co[ev[:, 0]]
    dn = n[ev[:, 1]] - n[ev[:, 0]]
    l2 = np.maximum((d * d).sum(1), 1e-10)
    k = (dn * d).sum(1) / l2
    acc = np.zeros(len(co))
    cnt = np.zeros(len(co))
    for j in (0, 1):
        np.add.at(acc, ev[:, j], k)
        np.add.at(cnt, ev[:, j], 1.0)
    return acc / np.maximum(cnt, 1.0)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _bake_masks(obj, bvh, maxd, meta, seed):
    me = obj.data
    nv = len(me.vertices)
    if nv == 0:
        return
    ao, co, n = _bake_ao(obj, bvh, maxd)
    k = _curvature(me, co, n)
    convex = _smoothstep(8.0, 140.0, k)
    concave = _smoothstep(6.0, 90.0, -k)
    nz1 = _value_noise3(co / 0.11, seed + 5, 0.25, 0.75)
    nz2 = _value_noise3(co / 0.37, seed + 9, 0.2, 0.8)
    aw = float(meta.get("auto_wear", 1.0))
    ag = float(meta.get("auto_grime", 1.0))
    wear_auto = convex * (0.25 + 0.75 * nz1) * 0.8 * aw
    cav = np.clip(1.0 - ao, 0.0, 1.0) ** 1.4
    grime_auto = (cav * 0.75 + concave * 0.5) * (0.55 + 0.45 * nz2)
    if meta.get("mount", "ground") == "ground":
        splash = np.clip(1.0 - co[:, 2] / 0.28, 0.0, 1.0) ** 1.6
        grime_auto = np.maximum(grime_auto, splash * (0.3 + 0.5 * nz2) * 0.7)
    # rain streak / dust: up-facing ledges gather a little dirt
    grime_auto = np.clip(grime_auto * ag, 0.0, 1.0)

    def get(name):
        a = me.attributes.get("m_" + name)
        v = np.zeros(nv, dtype=np.float32)
        if a is not None and a.domain == 'POINT' and len(a.data) == nv:
            a.data.foreach_get("value", v)
        return v.astype(np.float64)

    grime = np.clip(np.maximum(get("grime"), grime_auto), 0, 1)
    wear = np.clip(np.maximum(get("wear"), wear_auto), 0, 1)
    special = np.clip(get("special"), 0, 1)
    for nm in ("m_grime", "m_wear", "m_special"):
        a = me.attributes.get(nm)
        if a is not None:
            me.attributes.remove(a)
    for ca in list(me.color_attributes):
        me.color_attributes.remove(ca)
    # exported as the custom glTF attribute _MASK (VEC4: ao, grime, wear, special)
    col = me.attributes.new("_mask", 'FLOAT_COLOR', 'POINT')
    rgba = np.stack([ao, grime, wear, special], axis=1).astype(np.float32)
    col.data.foreach_set("color", rgba.ravel())
    return float(ao.mean())


def _weighted_normals(obj):
    try:
        m = obj.modifiers.new("WN", 'WEIGHTED_NORMAL')
        m.keep_sharp = True
        m.weight = 50
        m.mode = 'FACE_AREA'
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        old = obj.data
        obj.modifiers.clear()
        obj.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
    except Exception as e:  # pragma: no cover
        print("WARN weighted normals failed:", e)
        obj.modifiers.clear()


def _mat_stats(obj):
    me = obj.data
    out = {}
    npoly = len(me.polygons)
    if npoly == 0:
        return out
    mi = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("material_index", mi)
    lt = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    for i, m in enumerate(me.materials):
        t = int((lt[mi == i] - 2).sum())
        if t > 0 and m is not None:
            out[m.name] = out.get(m.name, 0) + t
    return out


# materials whose small islands must survive LOD reduction (they carry the night read)
_GLOW = ("emissive", "neon", "window_lit", "glass_stained", "glass_frosted", "glass_lantern")


def _islands(bm):
    """Connected components -> list of lists of BMVert."""
    seen = set()
    out = []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack = [v]
        seen.add(v.index)
        comp = []
        while stack:
            a = stack.pop()
            comp.append(a)
            for e in a.link_edges:
                b = e.other_vert(a)
                if b.index not in seen:
                    seen.add(b.index)
                    stack.append(b)
        out.append(comp)
    return out


def _make_lod(obj, min_size, ratio):
    """Reduced copy of a merged object: drop loose islands smaller than min_size (m), then collapse-decimate.
    Islands that glow (lamps, lit windows, neon) are always kept."""
    o = obj.copy()
    o.data = obj.data.copy()
    _link(o)
    me = o.data
    glow = {i for i, m in enumerate(me.materials) if m is not None and str(m.get("tsq_lib", "")).startswith(_GLOW)}
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.index_update()
    kill = []
    for comp in _islands(bm):
        if len(comp) > 20000:
            continue
        xs = [v.co.x for v in comp]
        ys = [v.co.y for v in comp]
        zs = [v.co.z for v in comp]
        d = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2)
        if d >= min_size:
            continue
        if glow and any(f.material_index in glow for v in comp[:8] for f in v.link_faces):
            continue
        kill.extend(comp)
    if kill and len(kill) < len(bm.verts):
        bmesh.ops.delete(bm, geom=kill, context='VERTS')
    bm.to_mesh(me)
    bm.free()
    me.update()
    if len(me.polygons) > 60:
        m = o.modifiers.new("LOD", 'DECIMATE')
        m.decimate_type = 'COLLAPSE'
        m.ratio = ratio
        m.use_collapse_triangulate = True
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me2 = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        o.modifiers.clear()
        o.data = me2
        bpy.data.meshes.remove(me)
    o.data.validate()
    return o


def _gltf_kw(glb_path):
    return dict(filepath=glb_path, export_format='GLB', use_selection=False, export_apply=False, export_yup=True,
                export_extras=True, export_materials='EXPORT', export_normals=True, export_tangents=False,
                export_texcoords=True, export_cameras=False, export_lights=False, export_animations=False,
                export_skins=False, export_morph=False, export_colors=False, export_attributes=True)


KEEP_ATTR = "_TSQ_KEEP"
KEEP_UV = "tsq_keep"                 # the UV layer of mark_keep(): glTF TEXCOORD_1, x = 1.0 on identity vertices


def mark_keep(objs, text_ids=None):
    """Mark the identity vertices of the meshes that have any, right before they are exported, so that a later
    simplification (the web bake) can lock them. Identity vertices: every vertex of a face tagged by
    tk.levels.lod_standin / lod_keep (face attribute tsq_id: kept originals and stand-ins) and of lettering
    (tsq_text) - all of it when text_ids is None, else only the text() numbers in text_ids.

    The mark is a second UV layer KEEP_UV, u = 1.0 on identity vertices and 0.0 elsewhere: glTF TEXCOORD_1.x
    (three.js: geometry.attributes.uv1, used by no material here; Blender's importer: the second UV map).
    Why not a custom attribute next to _MASK: Blender 4.1's glTF importer lists the custom attributes of a mesh
    in a set and their data in a list (io_scene_gltf2/blender/imp/gltf2_blender_mesh.py), so with two of them of
    different width the import raises ValueError whenever the set happens to iterate the other way round - always
    with PYTHONHASHSEED=0, every second process otherwise (measured) - and tools/checks/zscan.py and every other
    tool that imports a built GLB would stop working.
    Env TSQ_KEEP=attr writes the float point attribute _TSQ_KEEP instead (glTF custom attribute, for a reader that
    takes accessors straight from the file and never goes through that importer); TSQ_KEEP=0 (or TSQ_NOKEEP=1)
    writes nothing. -> number of marked vertices"""
    mode = "0" if os.environ.get("TSQ_NOKEEP") == "1" else os.environ.get("TSQ_KEEP", "uv")
    total = 0
    for o in objs:
        me = o.data
        old = me.attributes.get(KEEP_ATTR)
        if old is not None:
            me.attributes.remove(old)
        old = me.uv_layers.get(KEEP_UV)
        if old is not None:
            me.uv_layers.remove(old)
        n = len(me.polygons)
        if not n or mode == "0" or (mode != "attr" and not len(me.uv_layers)):
            continue
        m = np.zeros(n, dtype=bool)
        for an in ("tsq_id", "tsq_text"):
            a = me.attributes.get(an)
            if a is None or a.domain != 'FACE' or len(a.data) != n:
                continue
            v = np.empty(n, dtype=np.int32)
            a.data.foreach_get("value", v)
            if an == "tsq_text" and text_ids is not None:
                m |= (v > 0) & np.isin(v // 4, list(text_ids))
            else:
                m |= v > 0
        if not m.any():
            continue
        lt = np.empty(n, dtype=np.int32)
        me.polygons.foreach_get("loop_total", lt)
        ls = np.empty(n, dtype=np.int32)
        me.polygons.foreach_get("loop_start", ls)
        lv = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", lv)
        lm = np.zeros(len(me.loops), dtype=bool)
        for f in np.nonzero(m)[0]:
            lm[ls[f]:ls[f] + lt[f]] = True
        vm = np.zeros(len(me.vertices), dtype=np.float32)
        vm[lv[lm]] = 1.0
        if mode == "attr":
            me.attributes.new(KEEP_ATTR, 'FLOAT', 'POINT').data.foreach_set("value", vm)
        else:
            act = me.uv_layers.active_index
            ren = next((i for i, l in enumerate(me.uv_layers) if l.active_render), 0)
            uv = np.zeros((len(me.loops), 2), dtype=np.float32)
            uv[:, 0] = vm[lv]
            me.uv_layers.new(name=KEEP_UV, do_init=False).data.foreach_set("uv", uv.ravel())
            me.uv_layers.active_index = act          # the texture layer stays the first one written (TEXCOORD_0)
            me.uv_layers[ren].active_render = True
        total += int(vm.sum())
    return total


def _close_holes(Hs, s0, merged, made, level):
    """No holes (docs/LOD.md): where a viewer on the street looks through the decimated copy - an island under the
    size limit went that covered a wall face nobody draws from behind, the collapse pulled an edge off its
    neighbour - the faces lod 0 shows there go back in as they were. -> triangles added"""
    from . import holes as _ho
    used = [np.zeros(len(o.data.polygons), dtype=bool) for o in merged]

    def put(rows):
        n = 0
        own = s0["own"][rows]
        for i in np.unique(own).tolist():
            if i >= len(made):
                continue
            pm = np.zeros(len(merged[i].data.polygons), dtype=bool)
            pm[s0["poly"][rows[own == i]]] = True
            pm &= ~used[i]
            if not pm.any():
                continue
            used[i] |= pm
            if i not in base:
                base[i] = made[i].data.copy()
            pme = merged[i].data.copy()
            bm = bmesh.new()
            bm.from_mesh(pme)
            bm.faces.ensure_lookup_table()
            bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, pm) if not k], context='FACES')
            n += sum(len(f.verts) - 2 for f in bm.faces)
            bm.to_mesh(pme)
            bm.free()
            bm = bmesh.new()
            bm.from_mesh(made[i].data)
            bm.from_mesh(pme)               # appended; the layers match (the copy came from the same mesh)
            bm.to_mesh(made[i].data)
            bm.free()
            made[i].data.update()
            if i in patch:
                bm = bmesh.new()
                bm.from_mesh(patch[i])
                bm.from_mesh(pme)
                bm.to_mesh(patch[i])
                bm.free()
                bpy.data.meshes.remove(pme)
            else:
                patch[i] = pme
        return n

    def swap(i, me):
        bm = bmesh.new()
        bm.from_mesh(base[i])
        bm.from_mesh(me)
        bm.to_mesh(made[i].data)
        bm.free()
        made[i].data.update()

    base, patch = {}, {}
    bpy.context.view_layer.update()
    first, added = _ho.close(Hs, level, lambda: made, put)
    if added:
        added -= _ho.thin(Hs, level, lambda: made, list(patch.items()), swap)
    for me in patch.values():
        bpy.data.meshes.remove(me)
    for me in base.values():
        bpy.data.meshes.remove(me)
    return first, added


def _decimated_lods(merged, tris, diag, glb_path, kw, warnings, suffix="lod", measure=None, mount=None):
    """Reduced copies of the merged level-0 objects -> <variant>.<suffix>1.glb / .<suffix>2.glb.
    Returns [{"tris", "glb_bytes"}, ...]. This is the fallback for assets that do not model their own levels.
    measure(level, objects) -> anything: called on the copies of a level before they are exported; its result is
    stored as "text" (the build gates the lettering of the levels with it, see tk.lodimage.text_measure).
    mount: the asset's mount - given, the copies are closed where the street sees through them (_close_holes;
    "closed": the triangles that took, they are not held against the budget)."""
    lod_stats = []
    hs = None
    if mount is not None and os.environ.get("TSQ_HOLES", "1") != "0":
        try:
            from . import holes as _ho
            bpy.context.view_layer.update()
            hs = _ho.of_objs(merged, mount)
        except Exception as e:  # pragma: no cover
            warnings.append(f"lods: holes not looked for ({type(e).__name__}: {e})")
    # (min island size m, collapse ratio); thresholds grow with the asset so buildings shed more
    k = float(np.clip(diag / 6.0, 0.6, 2.5))
    for li, (ms, ratio) in enumerate(((0.035 * k, 0.4), (0.14 * k, 0.1)), start=1):
        if tris < 1500 and li == 1:
            ratio = 0.7
        made = []
        for o in merged:
            try:
                made.append(_make_lod(o, ms, ratio))
            except Exception as e:  # pragma: no cover
                warnings.append(f"lod{li} failed for {o.name}: {e}")
        bpy.ops.object.select_all(action='DESELECT')
        lt = 0
        for o, src in zip(made, merged):
            o.location = src.location
            for key in ("tsq_part", "tsq_part_props"):
                if key in src:
                    o[key] = src[key]
            o.name = src.name + f"_lod{li}"
            o.select_set(True)
        closed = 0
        if hs is not None and len(made) == len(merged):
            try:
                closed = _close_holes(hs[1], hs[0], merged, made, li)[1]
            except Exception as e:  # pragma: no cover
                warnings.append(f"lod{li}: holes not closed ({type(e).__name__}: {e})")
        for o in made:
            lt += sum(_mat_stats(o).values())
        tx = None
        if measure is not None:
            try:
                tx = measure(li, made)
            except Exception as e:  # pragma: no cover
                warnings.append(f"lod{li}: lettering not measured ({e})")
        lp = os.path.splitext(glb_path)[0] + f".{suffix}{li}.glb"
        mark_keep(made)
        bpy.ops.export_scene.gltf(**dict(kw, filepath=lp, use_selection=True))
        lod_stats.append({"tris": int(lt), "glb_bytes": os.path.getsize(lp), "closed": int(closed)})
        if measure is not None:
            lod_stats[-1]["text"] = tx
        for o in made:
            me_ = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me_.users == 0:
                bpy.data.meshes.remove(me_)
    return lod_stats


def finalize(asset_id, variant, params, meta, glb_path, build_seconds=0.0, lods=False, _keep=None):
    """Merge everything in the scene, bake masks, validate, export. Returns the sidecar dict.
    lods=True also writes <variant>.lod1.glb / .lod2.glb (reduced copies for the town runtime).
    _keep: optional dict that receives the merged objects and export settings (used by tk.final_levels)."""
    t0 = time.time()
    warnings, errors = [], []
    scene = bpy.context.scene
    # a `mount` PARAM of 'hang' / 'wall' / 'ground' overrides META['mount'] for that variant
    if params.get("mount") in ("hang", "wall", "ground", "wall_ground"):
        meta = dict(meta, mount=params["mount"])
    # stray curves / text -> meshes
    for o in list(scene.objects):
        if o.type in ('CURVE', 'FONT', 'SURFACE'):
            apply_modifiers(o)
    meshes = [o for o in scene.objects if o.type == 'MESH' and len(o.data.polygons) > 0]
    for o in list(scene.objects):
        if o.type == 'MESH' and len(o.data.polygons) == 0:
            bpy.data.objects.remove(o, do_unlink=True)
    if not meshes:
        raise RuntimeError("build() produced no mesh geometry")
    n_src = len(meshes)

    # materials present?
    plain = None
    for o in meshes:
        if len(o.data.materials) == 0 or any(m is None for m in o.data.materials):
            if plain is None:
                plain = mat("plain")
            warnings.append(f"object {o.name!r} has no material (rendered as flat grey 'plain')")
            if len(o.data.materials) == 0:
                o.data.materials.append(plain)
            else:
                for i, m in enumerate(o.data.materials):
                    if m is None:
                        o.data.materials[i] = plain

    # group into parts
    groups, part_info = {}, {}
    for o in meshes:
        pn = o.get("tsq_part", "body")
        groups.setdefault(pn, []).append(o)
        if pn != "body" and pn not in part_info:
            part_info[pn] = {"pivot": list(o.get("tsq_pivot", (0, 0, 0))),
                             "props": json.loads(o.get("tsq_part_props", "{}"))}
    merged = []
    for pn, objs in groups.items():
        j = join(objs, name=("body" if pn == "body" else f"part_{pn}"))
        _clean(j, warnings)
        if len(j.data.polygons) == 0:
            bpy.data.objects.remove(j, do_unlink=True)
            continue
        j["tsq_part"] = pn
        merged.append(j)

    # bounds
    allco = np.vstack([_np_co(o.data) for o in merged])
    mn, mx = allco.min(0), allco.max(0)
    size = mx - mn
    diag = float(np.linalg.norm(size))

    # occluders for AO: the asset itself + ground / wall plane
    V, T = _tri_soup(merged)
    mount = meta.get("mount", "ground")
    extra_v, extra_t = [], []
    R = max(diag * 2.0, 4.0)
    cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
    if mount in ("ground", "wall_ground"):
        base = len(V) + sum(len(e) for e in extra_v)
        extra_v.append(np.array([[cx - R, cy - R, -0.001], [cx + R, cy - R, -0.001], [cx + R, cy + R, -0.001], [cx - R, cy + R, -0.001]]))
        extra_t.append(np.array([[0, 1, 2], [0, 2, 3]]) + base)
    if mount in ("wall", "wall_ground"):
        base = len(V) + sum(len(e) for e in extra_v)
        cz = (mn[2] + mx[2]) / 2
        extra_v.append(np.array([[cx - R, 0.001, cz - R], [cx + R, 0.001, cz - R], [cx + R, 0.001, cz + R], [cx - R, 0.001, cz + R]]))
        extra_t.append(np.array([[0, 1, 2], [0, 2, 3]]) + base)
    if extra_v:
        V = np.vstack([V] + extra_v)
        T = np.vstack([T] + extra_t)
    bvh = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True, epsilon=0.0)
    maxd = float(np.clip(diag * 0.35, 0.25, 1.6))
    seed = int(params.get("seed", 1)) % 1000
    ao_mean = 1.0
    for o in merged:
        ao_mean = min(ao_mean, _bake_masks(o, bvh, maxd, meta, seed) or 1.0)

    # stats / validation
    tris = 0
    mats = {}
    for o in merged:
        for k, v in _mat_stats(o).items():
            mats[k] = mats.get(k, 0) + v
    tris = sum(mats.values())
    verts = sum(len(o.data.vertices) for o in merged)
    cls = meta.get("class", "prop")
    budget = int(meta.get("tri_budget", BUDGETS.get(cls, 30000)))
    if tris > budget * 2:
        errors.append(f"triangle count {tris} is more than 2x the budget ({budget}) for class {cls!r}")
    elif tris > budget:
        warnings.append(f"triangle count {tris} exceeds budget {budget} for class {cls!r}")
    if tris < 300:
        warnings.append(f"only {tris} triangles - far too simple to read as a detailed asset")
    cat = material_catalog()
    libs = {}
    for o in merged:
        for m in o.data.materials:
            if m is None:
                continue
            lib = m.get("tsq_lib")
            if lib is None:
                errors.append(f"material {m.name!r} was not created with tk.mat() (no library reference)")
            else:
                libs[m.name] = lib
                if cat and lib not in cat:
                    errors.append(f"unknown library material {lib!r}")
    if len(mats) > int(meta.get("max_materials", 10)):
        warnings.append(f"{len(mats)} material variants = {len(mats)} draw calls; keep it under {meta.get('max_materials', 10)}")
    if mount in ("ground", "wall_ground"):
        if mn[2] < -0.06 and not meta.get("allow_below_ground"):
            warnings.append(f"geometry extends {-mn[2]:.3f} m below ground (z=0)")
        if mn[2] > 0.03:
            warnings.append(f"asset floats {mn[2]:.3f} m above ground (lowest point should touch z=0)")
    if mount == "wall" and mx[1] > 0.05:
        warnings.append(f"wall-mounted asset extends {mx[1]:.3f} m behind the wall plane (y=0); it should project towards -Y")
    # building contract: front wall at y=0, body x in [-width/2, width/2], y in [0, depth] (small overhangs allowed)
    if meta.get("class") == "building" and not meta.get("free_footprint"):
        w, d = params.get("width"), params.get("depth")
        if isinstance(w, (int, float)) and (mn[0] < -w / 2 - 0.85 or mx[0] > w / 2 + 0.85):
            warnings.append(f"footprint contract: x spans {mn[0]:.2f}..{mx[0]:.2f} m but width={w:g} allows only +-{w / 2 + 0.7:.2f} m "
                            "(neighbours stand wall to wall - stray or mis-scaled geometry?)")
        gd = params.get("gallery_depth") if isinstance(params.get("gallery_depth"), (int, float)) else 0.0
        if "passage_width" in params:       # passages are centred on the origin, facades on both the -Y and +Y faces
            y0, y1 = (-d / 2 - 1.75, d / 2 + 1.75) if isinstance(d, (int, float)) else (-1e9, 1e9)
        else:
            y0, y1 = -1.75 - gd, (d + 0.85 if isinstance(d, (int, float)) else 1e9)
        if mn[1] < y0 or mx[1] > y1:
            warnings.append(f"footprint contract: y spans {mn[1]:.2f}..{mx[1]:.2f} m, allowed {y0:.2f}..{y1:.2f} m for depth={d}")
    exp = meta.get("size")  # optional expected [w, d, h] ranges: [[min,max],[min,max],[min,max]]
    if exp:
        for ax, (lo, hi), s in zip("XYZ", exp, size):
            if s < lo * 0.8 or s > hi * 1.25:
                warnings.append(f"size along {ax} is {s:.2f} m, expected {lo}-{hi} m")

    # weighted normals + part pivots
    for o in merged:
        _weighted_normals(o)
        pn = o["tsq_part"]
        if pn != "body":
            pv = Vector(part_info[pn]["pivot"])
            o.data.transform(Matrix.Translation(-pv))
            o.location = pv
            o["tsq_part_props"] = json.dumps(part_info[pn]["props"])

    # sockets
    sockets = []
    for o in scene.objects:
        if o.type == 'EMPTY' and "tsq_socket" in o:
            mw = world_matrix(o)
            loc, rot, _ = mw.decompose()
            e = rot.to_euler()
            sockets.append({"name": o["tsq_name"], "kind": o["tsq_socket"],
                            "pos": [round(c, 4) for c in loc],
                            "rot": [round(math.degrees(a), 2) for a in e],
                            "props": json.loads(o.get("tsq_props", "{}"))})
            o.parent = None
            o.matrix_basis = mw
            o.name = f"sock:{o['tsq_socket']}:{o['tsq_name']}"
        elif o.type == 'EMPTY':
            pass
    for o in list(scene.objects):
        if o.type == 'EMPTY' and "tsq_socket" not in o:
            # helper empties (roots): re-parent children to world and drop
            for c in list(o.children):
                mw = world_matrix(c)
                c.parent = None
                c.matrix_basis = mw
            bpy.data.objects.remove(o, do_unlink=True)
    lights = [s for s in sockets if s["kind"] == "light"]
    if len(lights) > int(meta.get("max_lights", 6)):
        warnings.append(f"{len(lights)} light sockets; keep to {meta.get('max_lights', 6)} or fewer per asset")

    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    kw = _gltf_kw(glb_path)
    mark_keep(merged)
    bpy.ops.export_scene.gltf(**kw)

    lod_stats = []
    if lods:
        lod_stats = _decimated_lods(merged, tris, diag, glb_path, kw, warnings, mount=mount)
    if _keep is not None:
        _keep.update(merged=merged, kw=kw, tris=tris, diag=diag, meta=meta, maxd=maxd, seed=seed)

    info = {
        "id": asset_id, "variant": variant, "params": dict(params),
        "meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, list, dict))},
        "stats": {"tris": int(tris), "verts": int(verts), "materials": mats, "libs": libs, "source_objects": n_src,
                  "ao_mean": round(ao_mean, 3), "glb_bytes": os.path.getsize(glb_path),
                  "build_s": round(build_seconds, 2), "finalize_s": round(time.time() - t0, 2), "lods": lod_stats},
        "bounds": {"min": [round(float(c), 4) for c in mn], "max": [round(float(c), 4) for c in mx],
                   "size": [round(float(c), 4) for c in size]},
        "parts": part_info, "sockets": sockets, "warnings": warnings, "errors": errors,
    }
    with open(os.path.splitext(glb_path)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return info
