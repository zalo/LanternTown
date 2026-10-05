"""tk.final_levels - export and checks of the MODELLED cheaper levels (lod 1 / 2) of a level-aware asset.

tools/bl_build.py builds level 0 exactly as always (tk.final.finalize), captures it with reference(),
then runs build() again at level 1 and 2 and hands each scene to finalize_level():

  * merge per part like level 0, weighted normals, same part pivots
  * the vertex masks (AO / grime / wear / special) are TRANSFERRED from level 0 instead of baked again,
    so a cheaper level keeps the weathering of the full model (no popping when the level switches, and
    edge wear does not vanish with the bevels that produced it)
  * the level is rejected (the caller then ships the decimated fallback) when it is over budget, moves or
    loses a socket / light / part, loses something that glows, changes the bounds, or shifted the random stream

The same finalize_level() exports the REDUCED levels tk.reduce makes from the full model (docs/LOD.md, "Reduced
levels"). What ships at a level is then one of up to three candidates - modelled ("authored"), reduced, decimated -
chosen by pick() from their pictures (tk.lodimage).
"""
import bpy
import json
import math
import os
import time
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from .core import world_matrix, apply_modifiers, _LOD
from .surf import mat
from .geo import join
from . import levels as _lv
from .final import _np_co, _clean, _mat_stats, _weighted_normals, _bake_masks, _tri_soup


def _mesh_arrays(objs):
    """World-space arrays over several mesh objects -> V (n,3), T (m,3) global vertex ids, TM (m,) material id,
    names [material name], offsets [(first vertex, vertex count)] per object."""
    V, T, TM, offs, names, off = [], [], [], [], [], 0
    idx = {}
    for o in objs:
        me = o.data
        mw = np.array(world_matrix(o), dtype=np.float64)
        co = _np_co(me) @ mw[:3, :3].T + mw[:3, 3]
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        t = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", t)
        pi = np.empty(nt, dtype=np.int32)
        me.loop_triangles.foreach_get("polygon_index", pi)
        mi = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get("material_index", mi)
        local = []
        for m in (list(me.materials) or [None]):
            nm = m.name if m is not None else ""
            if nm not in idx:
                idx[nm] = len(names)
                names.append(nm)
            local.append(idx[nm])
        local = np.array(local, dtype=np.int32)
        V.append(co)
        T.append(t.reshape(-1, 3) + off)
        TM.append(local[np.minimum(mi[pi], len(local) - 1)] if nt else np.zeros(0, dtype=np.int32))
        offs.append((off, len(co)))
        off += len(co)
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int32), names, offs
    return np.vstack(V), np.vstack(T), np.concatenate(TM), names, offs


def _tri_area(V, T):
    if len(T) == 0:
        return np.zeros(0)
    return 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)


def _bary(V, T, tri, P):
    """Barycentric weights (k,3) of points P (k,3) in triangles tri (k,) - clamped, normalised."""
    a, b, c = V[T[tri, 0]], V[T[tri, 1]], V[T[tri, 2]]
    v0, v1, v2 = b - a, c - a, P - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1.0, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    W = np.clip(np.stack([1.0 - v - w, v, w], axis=1), 0.0, 1.0)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-12)


def text_tris(objs):
    """The lettering of mesh objects - the faces tk.geo.text() tagged (face attribute tsq_text) - as world-space
    triangles -> {"P": (n,3,3), "g": (n,) number of the text() call, "k": (n,) 1 front | 2 side | 3 back}
    (the input of tk.lodimage.text_prepare / text_measure)."""
    Ps, Gs = [], []
    for o in objs:
        me = o.data
        a = me.attributes.get("tsq_text")
        if a is None or a.domain != 'FACE' or len(a.data) != len(me.polygons) or not len(me.polygons):
            continue
        v = np.empty(len(me.polygons), dtype=np.int32)
        a.data.foreach_get("value", v)
        if not v.any():
            continue
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        t = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", t)
        pi = np.empty(nt, dtype=np.int32)
        me.loop_triangles.foreach_get("polygon_index", pi)
        w = v[pi] > 0
        if not w.any():
            continue
        mw = np.array(world_matrix(o), dtype=np.float64)
        co = _np_co(me) @ mw[:3, :3].T + mw[:3, 3]
        Ps.append(co[t.reshape(-1, 3)[w]])
        Gs.append(v[pi][w])
    if not Ps:
        return {"P": np.zeros((0, 3, 3)), "g": np.zeros(0, dtype=np.int64), "k": np.zeros(0, dtype=np.int64)}
    G = np.concatenate(Gs).astype(np.int64)
    return {"P": np.vstack(Ps), "g": G // 4, "k": G % 4}


def reference(info, keep):
    """Capture level 0 right after tk.final.finalize(): geometry + masks (numpy, survives clear_scene).
    "text": the lettering of lod 0 (tk.lodimage.text_prepare), what the levels' lettering is checked against."""
    from . import lodimage as _im
    merged = keep["merged"]
    V, T, TM, names, _ = _mesh_arrays(merged)
    M = []
    for o in merged:
        me = o.data
        a = me.attributes.get("_mask")
        m = np.zeros(len(me.vertices) * 4, dtype=np.float32)
        if a is not None and len(a.data) == len(me.vertices):
            a.data.foreach_get("color", m)
        else:
            m[0::4] = 1.0
        M.append(m.reshape(-1, 4).astype(np.float64))
    M = np.vstack(M) if M else np.zeros((0, 4))
    area = _tri_area(V, T)
    libs = {}
    for o in merged:
        for m in o.data.materials:
            if m is not None:
                libs[m.name] = str(m.get("tsq_lib", ""))
    # one sample per (vertex, material) with the surface area it stands for
    nm = max(len(names), 1)
    key = T.astype(np.int64).ravel() * nm + np.repeat(TM.astype(np.int64), 3)
    uk, inv = np.unique(key, return_inverse=True)
    sa = np.bincount(inv, weights=np.repeat(area / 3.0, 3), minlength=len(uk))
    mat_area = {names[i]: float(area[TM == i].sum()) for i in range(len(names))}
    return {"info": info, "V": V, "T": T, "TM": TM, "M": M, "names": names, "libs": libs, "mat_area": mat_area,
            "sv": (uk // nm).astype(np.int64), "sm": (uk % nm).astype(np.int64), "sa": sa,
            "tris": int(info["stats"]["tris"]), "diag": float(keep["diag"]), "maxd": keep["maxd"], "seed": keep["seed"],
            "text": _im.text_prepare(text_tris(merged)), "_bvh": {}, "_vl": None}


def _ref_bvh(ref, name=None):
    """BVH of the level-0 surface: all of it (name None) or only the triangles of one material (or of a tuple of them)."""
    if name not in ref["_bvh"]:
        if ref["_vl"] is None:
            ref["_vl"] = ref["V"].tolist()
        if name is None:
            ti = np.arange(len(ref["T"]))
        elif isinstance(name, tuple):
            ti = np.nonzero(np.isin(ref["TM"], [ref["names"].index(n) for n in name]))[0]
        else:
            ti = np.nonzero(ref["TM"] == ref["names"].index(name))[0]
        ref["_bvh"][name] = (BVHTree.FromPolygons(ref["_vl"], ref["T"][ti].tolist(), all_triangles=True, epsilon=0.0), ti)
    return ref["_bvh"][name]


_R2 = (0.7548776662466927, 0.5698402909980532)
_BOTH_SIDES = ("glass_clear", "glass_frosted", "glass_lantern", "water", "water_fall", "moss", "hedge", "leaf")


def _surface_samples(ref, tris, a0, cap=24):
    """Points spread over level-0 triangles (ids `tris`), about one per a0 of area and at most `cap` per triangle
    -> positions, masks (interpolated), the area each stands for, face normals."""
    V, T = ref["V"], ref["T"][tris]
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross(B - A, C - A)
    ar = 0.5 * np.linalg.norm(cr, axis=1)
    ok = ar > 1e-12
    T, A, B, C, cr, ar = T[ok], A[ok], B[ok], C[ok], cr[ok], ar[ok]
    fn = cr / (2.0 * ar[:, None])
    k = np.clip(np.ceil(ar / a0), 1, cap).astype(np.int64)
    tri = np.repeat(np.arange(len(T)), k)
    j = np.arange(len(tri)) - np.repeat(np.cumsum(k) - k, k)
    u = (0.5 + (j + 1) * _R2[0]) % 1.0
    v = (0.5 + (j + 1) * _R2[1]) % 1.0
    one = np.repeat(k, k) == 1
    u, v = np.where(one, 1.0 / 3.0, u), np.where(one, 1.0 / 3.0, v)
    fl = u + v > 1.0
    u, v = np.where(fl, 1.0 - u, u), np.where(fl, 1.0 - v, v)
    w = np.stack([1.0 - u - v, u, v], axis=1)
    P = A[tri] * w[:, 0:1] + B[tri] * w[:, 1:2] + C[tri] * w[:, 2:3]
    M = ref["M"]
    m = M[T[tri, 0]] * w[:, 0:1] + M[T[tri, 1]] * w[:, 1:2] + M[T[tri, 2]] * w[:, 2:3]
    return P, m, (ar / k)[tri], fn[tri]


def _transfer_masks(ref, objs, seen=0.3, fit=False):
    """Masks of the level objects from level 0. Every level-0 vertex is projected onto the nearest level surface of
    the SAME material and its mask spread over that triangle's corners (weighted by the area it stands for and by
    how exposed it is: its baked occlusion) - a proper average of what one SEES, so a flat stand-in for fine or
    layered geometry gets the mean tone of the surface that showed, not of what lay buried under it.
    Level vertices nothing projects onto take the value of the nearest level-0 surface point.
    A level material the full model does not have (a stand-in in another finish: other colour or parameters) takes
    its masks from every level-0 finish of the same library material; only if there is none, from all of it.
    Returns (coverage 0..1 of the level-0 surface that has level surface close by, p95 distance (m) of the level
    vertices from the level-0 surface).
    seen: the baked occlusion from which a level-0 sample counts in full. 0.3 suits a modelled stand-in (what it
    replaces is mostly buried); a level reduced from the full model's own faces (tk.reduce) passes a low value,
    as there every shaded face that is still there showed before, and the level would come out lighter.
    fit: for such a level, whose surface IS the full model's with fewer triangles: the samples are points spread
    over the level-0 surface (not only its vertices), count only on level faces that look the same way (unless the
    material shows both sides) and close by. That keeps the mean tone of every surface whatever way its triangles
    now run (occlusion is per vertex: a board of five corners is lighter or darker with each triangulation), and
    the underside of a thin sheet does not darken its top."""
    V, T, TM, names, offs = _mesh_arrays(objs)
    if fit:
        cr = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]) if len(T) else np.zeros((0, 3))
        fnl = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-20)
        two = {}
        for o in objs:
            for m_ in o.data.materials:
                if m_ is not None:
                    par = m_.get("tsq_params")
                    try:
                        par = json.loads(par) if isinstance(par, str) else None
                    except Exception:
                        par = None
                    lib = str(m_.get("tsq_lib", ""))
                    two[m_.name] = bool(par and par.get("double")) or lib.startswith(_BOTH_SIDES)
        a0 = max(float(ref["sa"].sum()) / 250000.0, 1e-8)
    lib_l = {m.name: str(m.get("tsq_lib", "")) for o in objs for m in o.data.materials if m is not None}

    def source(nm):
        if nm in ref["names"]:
            return (nm,)
        k = lib_l.get(nm, "")
        return tuple(n for n in ref["names"] if k and ref["libs"].get(n, "") == k)

    nv = len(V)
    acc = np.zeros((nv, 4))
    wsum = np.zeros(nv)
    diag = ref["diag"]
    dmax = max(0.06, 0.03 * diag)
    dcov = max(0.03, 0.012 * diag)
    vl = V.tolist()
    cov_a, tot_a = 0.0, float(ref["sa"].sum())
    for mi_, nm in enumerate(names):
        src = source(nm)
        if not src:
            continue
        ti = np.nonzero(TM == mi_)[0]
        if len(ti) == 0:
            continue
        ids = [ref["names"].index(n) for n in src]
        if fit:
            st = np.nonzero(np.isin(ref["TM"], ids))[0]
            if len(st) == 0:
                continue
            P, sm_, sa_, sn_ = _surface_samples(ref, st, a0)
        else:
            sel = np.nonzero(np.isin(ref["sm"], ids))[0]
            if len(sel) == 0:
                continue
            P = ref["V"][ref["sv"][sel]]
        bvh = BVHTree.FromPolygons(vl, T[ti].tolist(), all_triangles=True, epsilon=0.0)
        near = bvh.find_nearest
        hit_i, hit_p, hit_t, hit_d, hit_f = [], [], [], [], []
        one = fit and not two.get(nm, False)
        if one:
            fl_, sl_, around = fnl[ti].tolist(), sn_.tolist(), bvh.find_nearest_range
        for k, p in enumerate(P.tolist()):
            loc, _n, fi, d = near(p, dcov if fit else dmax)
            if loc is not None:
                ok = True
                if one:
                    n0, f = sl_[k], fl_[fi]
                    if n0[0] * f[0] + n0[1] * f[1] + n0[2] * f[2] <= 0.2:
                        # the nearest face looks the other way (the underside of a thin sheet whose two skins the
                        # collapse has moved through each other): the nearest one that looks the same way
                        ok = False
                        for loc2, _n2, fi2, d2 in around(p, dcov):
                            f = fl_[fi2]
                            if n0[0] * f[0] + n0[1] * f[1] + n0[2] * f[2] > 0.2 and (not ok or d2 < d):
                                ok, loc, fi, d = True, loc2, fi2, d2
                hit_i.append(k)
                hit_p.append(loc[:])
                hit_t.append(fi)
                hit_d.append(d)
                hit_f.append(ok)
        if not hit_i:
            continue
        hit_i = np.array(hit_i)
        tri = ti[np.array(hit_t)]
        hit_p, hit_d = np.array(hit_p), np.array(hit_d)
        if fit:
            a = sa_[hit_i]
            m = sm_[hit_i]
            cov_a += float(a.sum())
            if one:
                face = np.array(hit_f, dtype=bool)
                hit_i, tri, hit_p, hit_d, a, m = hit_i[face], tri[face], hit_p[face], hit_d[face], a[face], m[face]
                if not len(hit_i):
                    continue
            W = _bary(V, T, tri, hit_p)
        else:
            W = _bary(V, T, tri, hit_p)
            a = ref["sa"][sel][hit_i]
            m = ref["M"][ref["sv"][sel][hit_i]]
            cov_a += float(a[hit_d <= dcov].sum())
        # buried surface (occlusion ~0) hardly counts, a shaded joint or recess that still shows counts in full
        vis = a * np.clip(m[:, 0] / seen, 0.05, 1.0)
        for c in range(3):
            w = W[:, c] * vis
            np.add.at(wsum, T[tri, c], w)
            np.add.at(acc, T[tri, c], m * w[:, None])
    # vertices without (enough) projected samples: point sample of the level-0 surface
    varea = np.zeros(nv)
    ta = _tri_area(V, T)
    for c in range(3):
        np.add.at(varea, T[:, c], ta / 3.0)
    vmat = np.full(nv, -1, dtype=np.int64)
    for c in range(3):
        vmat[T[:, c]] = TM
    need = np.nonzero(wsum <= np.maximum(varea * 0.02, 1e-12))[0]
    out = np.where(wsum[:, None] > 1e-12, acc / np.maximum(wsum[:, None], 1e-12), 0.0)
    if len(need):
        if fit:
            vnl = np.zeros((nv, 3))                     # which way the level looks at its vertices
            for c in range(3):
                np.add.at(vnl, T[:, c], cr)
            vnl /= np.maximum(np.linalg.norm(vnl, axis=1, keepdims=True), 1e-20)
            if ref.get("_fn") is None:
                c0 = np.cross(ref["V"][ref["T"][:, 1]] - ref["V"][ref["T"][:, 0]], ref["V"][ref["T"][:, 2]] - ref["V"][ref["T"][:, 0]])
                ref["_fn"] = c0 / np.maximum(np.linalg.norm(c0, axis=1, keepdims=True), 1e-20)
        for mi_ in np.unique(vmat[need]):
            nm = names[mi_] if mi_ >= 0 else None
            src = source(nm) if nm is not None else ()
            bvh, ti = _ref_bvh(ref, (src[0] if len(src) == 1 else src) if src else None)
            vs = need[vmat[need] == mi_]
            one = fit and nm is not None and not two.get(nm, False) and len(ti) > 0
            loc, tri = [], []
            for iv, p in zip(vs.tolist(), V[vs].tolist()):
                l, _n, fi, _d = bvh.find_nearest(p)
                if one and l is not None:
                    n0 = vnl[iv]
                    if float(ref["_fn"][ti[fi]] @ n0) <= 0.2:
                        # ... and of the level-0 faces that look the same way, when the nearest does not
                        best = None
                        for l2, _n2, fi2, d2 in bvh.find_nearest_range(p, dcov):
                            if (best is None or d2 < best) and float(ref["_fn"][ti[fi2]] @ n0) > 0.2:
                                best, l, fi = d2, l2, fi2
                loc.append(l[:] if l is not None else p)
                tri.append(fi if fi is not None else 0)
            if len(ti) == 0:
                out[vs] = (1.0, 0.0, 0.0, 0.0)
                continue
            tri = ti[np.array(tri)]
            W = _bary(ref["V"], ref["T"], tri, np.array(loc))
            out[vs] = sum(ref["M"][ref["T"][tri, c]] * W[:, c:c + 1] for c in range(3))
    out = np.clip(out, 0.0, 1.0)
    for o, (off, n) in zip(objs, offs):
        me = o.data
        for nm in ("m_grime", "m_wear", "m_special"):
            a = me.attributes.get(nm)
            if a is not None:
                me.attributes.remove(a)
        for ca in list(me.color_attributes):
            me.color_attributes.remove(ca)
        col = me.attributes.new("_mask", 'FLOAT_COLOR', 'POINT')
        col.data.foreach_set("color", out[off:off + n].astype(np.float32).ravel())
    # how far the level strays from the full model
    dev = 0.0
    if nv:
        bvh, _ = _ref_bvh(ref, None)
        step = max(1, nv // 20000)
        d = [bvh.find_nearest(p)[3] or 0.0 for p in V[::step].tolist()]
        dev = float(np.percentile(d, 95)) if d else 0.0
    return (cov_a / tot_a if tot_a > 0 else 1.0), dev


def _sockets(scene):
    out = []
    for o in scene.objects:
        if o.type == 'EMPTY' and "tsq_socket" in o:
            loc, rot, _ = world_matrix(o).decompose()
            e = rot.to_euler()
            out.append({"name": o["tsq_name"], "kind": o["tsq_socket"], "pos": [round(c, 4) for c in loc],
                        "rot": [round(math.degrees(a), 2) for a in e], "props": json.loads(o.get("tsq_props", "{}"))})
    return out


def _socket_diff(a, b):
    """level-0 sockets a vs level sockets b -> list of human-readable differences (empty = identical)"""
    out = []
    A = {(s["kind"], s["name"]): s for s in a}
    B = {(s["kind"], s["name"]): s for s in b}
    miss = sorted(k for k in A if k not in B)
    extra = sorted(k for k in B if k not in A)
    if miss:
        out.append("missing " + ", ".join(f"{k[0]}:{k[1]}" for k in miss[:4]) + (" ..." if len(miss) > 4 else ""))
    if extra:
        out.append("extra " + ", ".join(f"{k[0]}:{k[1]}" for k in extra[:4]) + (" ..." if len(extra) > 4 else ""))
    for k in A:
        if k not in B:
            continue
        s, t = A[k], B[k]
        dp = max(abs(x - y) for x, y in zip(s["pos"], t["pos"]))
        dr = max(abs(((x - y + 180.0) % 360.0) - 180.0) for x, y in zip(s["rot"], t["rot"]))
        if dp > 3e-4 or dr > 0.06:
            out.append(f"{k[0]}:{k[1]} moved by {dp * 1000:.1f} mm / {dr:.2f} deg")
        elif s["props"] != t["props"]:
            out.append(f"{k[0]}:{k[1]} has other props")
        if len(out) > 5:
            break
    return out


def breakdown():
    """{source object name: triangles} of the scene before the merge (modifiers evaluated; Blender's .001 suffix
    and trailing counters stripped, so repeated things add up). For `tools/tsq lod`: where the triangles are."""
    import re
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for o in bpy.context.scene.objects:
        if o.type not in ('MESH', 'CURVE', 'FONT', 'SURFACE'):
            continue
        try:
            if o.type == 'MESH' and not o.modifiers:
                me, eo = o.data, None
            else:
                eo = o.evaluated_get(dg)
                me = eo.to_mesh()
            n = len(me.polygons)
            lt = np.empty(n, dtype=np.int32)
            me.polygons.foreach_get("loop_total", lt)
            t = int((lt - 2).sum()) if n else 0
            if eo is not None:
                eo.to_mesh_clear()
        except Exception:
            continue
        nm = re.sub(r"[_.]?\d+$", "", re.sub(r"\.\d{3,}$", "", o.name)) or o.name
        pn = o.get("tsq_part")
        if pn:
            nm = f"{nm} [{pn}]"
        src = (_LOD.get("where") or {}).get(o.name)       # what the generator joined into it
        tot = sum(src.values()) if src else 0
        if tot and len(src) > 1:
            for k, v in src.items():
                kk = f"{nm} > {k}"
                out[kk] = out.get(kk, 0) + t * v / tot
        else:
            out[nm] = out.get(nm, 0) + t
    return {k: int(round(v)) for k, v in out.items()}


def budget(level, tris0, meta=None):
    """Triangle budget of a level for an asset of tris0 triangles at level 0. An asset whose lod 0 is itself a
    distance model - a backdrop that is only ever seen from far off, so its far level IS the model - declares the
    share its modelled level may keep: META["lod_share"] = {2: 0.6} (docs/LOD.md); every other asset has the
    shares of tk.levels.BUDGET."""
    share = _lv.BUDGET[level]
    own = (meta or {}).get("lod_share")
    if own:
        share = float(own.get(level, own.get(str(level), share)))
    return int(max(share * tris0, min(tris0, _lv.BUDGET_FLOOR[level])))


AUTHORED_YIELDS = False         # True: a modelled level that keeps its contract but misses an image gate gives way to a
                                # reduced / decimated copy that passes them all with a lower image difference (pick())
REJECTED_SHIPS = "decimated"    # what replaces a modelled level that BREAKS its contract: "decimated" (docs/LOD.md rule;
                                # tools/lod_tests/contract.py pins it) or "best" (reduced / decimated by their pictures)


def hard_reasons(why):
    """the reasons of a finalize_level() result other than the triangle budget (pick() judges the budget itself:
    a reduced lod 2 may use up to tk.lodimage.ALLOW2 x the budget)"""
    return [w for w in why if " budget of " not in w]


def candidate(name, kind, res=None, file=None, **kw):
    """One entry of pick()'s candidate list. kind: "authored" | "reduced" | "decimated"; res: the finalize_level()
    result of the level (the decimated copy has none: pass tris=, budget=, glb_bytes=); file: its GLB. The caller
    adds "m" = tk.lodimage.Metric.score(level, file)."""
    c = {"name": name, "kind": kind, "file": file, "hard": [], "m": None, "tris": 0, "budget": 0, "glb_bytes": 0}
    if res is not None:
        c.update(tris=res["tris"], budget=res["budget"], glb_bytes=res["glb_bytes"],
                 hard=hard_reasons(res.get("why") or []), res=res)
    c.update(kw)
    return c


def pick(level, cands):
    """What ships at a level, from the candidate list [candidate(), ...] with their image metrics in "m".

    Without a modelled level this is tk.lodimage.choose(): hard gates, image gates, lowest image difference; the
    decimated copy only wins when it is better by more than DEC_MARGIN; a reduced lod 2 may use up to ALLOW2 x
    the budget. A modelled ("authored") level is the author's decision and is not put up for that vote:
      * one that keeps its contract ships. When it misses an image gate (silhouette, lit brightness, glow) that
        another candidate passes with a lower image difference, that candidate is named in "better" (the build
        prints it as a warning) - and ships instead when AUTHORED_YIELDS is on.
      * one that breaks its contract (budget, sockets, parts, glow, bounds, random stream, build failed) is replaced
        by the decimated copy (REJECTED_SHIPS), as before.
    -> the dict of choose() plus "kept": the image gates a modelled level misses although it ships, "better": the
       name of the candidate that passes them (or None)."""
    from . import lodimage as _im
    out = _im.choose(level, cands)
    out.update(kept=None, better=None)
    a = next((c for c in cands if c["kind"] == "authored"), None)
    if a is None:
        return out
    by = {c["name"]: c for c in cands}
    dec = next((c for c in cands if c["kind"] == "decimated"), None)
    if a.get("hard") or a["tris"] > a["budget"] or not a.get("m"):
        if REJECTED_SHIPS == "decimated" and dec is not None and out["winner"] != dec["name"]:
            if out["winner"] in by:
                out["why"][out["winner"]] = ["the modelled level broke its contract: the decimated copy ships"]
            m = dec.get("m")
            out.update(winner=dec["name"], kind="decimated", allowance=False, over=0, img=m["img"] if m else None,
                       clean=bool(m) and not _im.gates(level, m, None))
        return out
    am = a["m"]
    missed = _im.gates(level, am, None)
    w = by[out["winner"]]
    wm = w.get("m")
    if w is not a and missed and wm is not None and wm["img"] + _im.DEC_MARGIN < am["img"] and not _im.gates(level, wm, None):
        out["better"] = w["name"]
        if AUTHORED_YIELDS:
            out["why"][a["name"]] = missed + [f"the {w['kind']} copy passes, image difference {wm['img']:.4f} against {am['img']:.4f}"]
            return out
    if w is not a:
        out["why"][w["name"]] = ["the modelled level keeps its contract and ships"
                                 + ("" if wm is None else f" (image difference {am['img']:.4f}, this one {wm['img']:.4f})")]
        out["why"][a["name"]] = []
        out.update(winner=a["name"], kind="authored", allowance=False, over=0, img=am["img"])
    out.update(clean=not missed, kept=missed)
    return out


def finalize_level(level, ref, glb_path, params, meta, rng=None, rec0=None, build_seconds=0.0, keep_rejected=False,
                   seen=0.3, fit=False, text=False):
    """Merge / mask / check / export the scene of a level >= 1 build -> result dict
    {"level", "ok", "why": [reasons it was rejected], "warn": [...], "tris", "glb_bytes", "ratio", "budget",
     "cover", "dev", "seconds"}. The file <variant>.lod<level>.glb is written when the level is accepted
    (or keep_rejected). The scene is a modelled level (build() at tk.lod() == level) or the reduced full model
    (tk.reduce.reduce_scene: seen / fit are the mask transfer of those, see _transfer_masks; text=True adds "text":
    tk.lodimage.text_measure() of the level's lettering against lod 0's - only a level made from the lod-0 scene
    carries the same text numbers)."""
    t0 = time.time()
    scene = bpy.context.scene
    why, warn = [], []
    info0 = ref["info"]
    if params.get("mount") in ("hang", "wall", "ground", "wall_ground"):
        meta = dict(meta, mount=params["mount"])
    for o in list(scene.objects):
        if o.type in ('CURVE', 'FONT', 'SURFACE'):
            apply_modifiers(o)
    # objects emptied by lod_upto / lod_drop stay in the list: realize() then draws their texture offset exactly
    # where the full model does. Everything else without faces is dropped, as at level 0.
    meshes = [o for o in scene.objects if o.type == 'MESH' and (len(o.data.polygons) > 0 or "_lod_empty" in o)]
    for o in list(scene.objects):
        if o.type == 'MESH' and len(o.data.polygons) == 0 and "_lod_empty" not in o:
            bpy.data.objects.remove(o, do_unlink=True)
    if not any(len(o.data.polygons) for o in meshes):
        raise RuntimeError(f"build() produced no mesh geometry at lod {level}")
    plain = None
    for o in meshes:
        if len(o.data.polygons) and (len(o.data.materials) == 0 or any(m is None for m in o.data.materials)):
            if plain is None:
                plain = mat("plain")
            warn.append(f"object {o.name!r} has no material at lod {level}")
            if len(o.data.materials) == 0:
                o.data.materials.append(plain)
            else:
                for i, m in enumerate(o.data.materials):
                    if m is None:
                        o.data.materials[i] = plain
    groups, part_info = {}, {}
    for o in meshes:
        pn = o.get("tsq_part", "body")
        groups.setdefault(pn, []).append(o)
        if pn != "body" and pn not in part_info:
            part_info[pn] = {"pivot": list(o.get("tsq_pivot", (0, 0, 0))), "props": json.loads(o.get("tsq_part_props", "{}"))}
    merged = []
    for pn, objs in groups.items():
        j = join(objs, name=("body" if pn == "body" else f"part_{pn}"))
        _clean(j, warn)
        if len(j.data.polygons) == 0:
            bpy.data.objects.remove(j, do_unlink=True)
            part_info.pop(pn, None)
            continue
        j["tsq_part"] = pn
        merged.append(j)
    tx = None
    if text and ref.get("text"):
        from . import lodimage as _im
        from .reduce import ident_info
        gone = set(ident_info().get("text", {}).get(str(level), ()))    # lettering inside a part swapped for its stand-in
        tx = _im.text_measure(level, [G for G in ref["text"] if G["id"] not in gone], text_tris(merged))

    allco = np.vstack([_np_co(o.data) for o in merged])
    mn, mx = allco.min(0), allco.max(0)
    try:
        cover, dev = _transfer_masks(ref, merged, seen, fit)
    except Exception as e:  # pragma: no cover - never lose a level over its masks
        warn.append(f"mask transfer failed at lod {level} ({e}); masks baked from the level's own geometry")
        cover, dev = 0.0, 0.0
        V, T = _tri_soup(merged)
        bvh = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True, epsilon=0.0)
        for o in merged:
            _bake_masks(o, bvh, ref["maxd"], meta, ref["seed"])

    mats = {}
    for o in merged:
        for k, v in _mat_stats(o).items():
            mats[k] = mats.get(k, 0) + v
    tris = int(sum(mats.values()))
    tris0 = ref["tris"]
    bud = budget(level, tris0, meta)

    # ---- the contract ----
    if tris > bud:
        why.append(f"{tris} tris is over the lod {level} budget of {bud} ({100.0 * tris / max(tris0, 1):.0f} % of lod 0, "
                   f"allowed {100.0 * bud / max(tris0, 1):.0f} %)")
    if rng is not None and rec0 is not None:
        sigs = dict(_LOD["sigs"])
        sigs[""] = _lv._sig(rng)
        drift = _lv._rng_drift(rec0, {"sigs": sigs})
        if drift:
            why.append("the random stream shifted (a different number of rng draws than at lod 0) in: " + ", ".join(drift[:4])
                       + " - draw inside lod_upto(), or from rng.fork('name') for level-only parts")
    warn.extend(n for n in _LOD.get("notes", []) if n not in warn)
    sd = _socket_diff(info0.get("sockets", []), _sockets(scene))
    if sd:
        why.append("sockets / lights differ from lod 0: " + "; ".join(sd[:4]))
    p0 = info0.get("parts", {})
    if set(p0) != set(part_info):
        why.append(f"parts differ from lod 0: have {sorted(part_info)}, lod 0 has {sorted(p0)}")
    else:
        for pn, pi_ in part_info.items():
            if max(abs(a - b) for a, b in zip(pi_["pivot"], p0[pn]["pivot"])) > 3e-4 or pi_["props"] != p0[pn]["props"]:
                why.append(f"part {pn!r} has another pivot / motion than at lod 0")
    # everything that glows must survive
    area = {}
    V, T, TM, names, _ = _mesh_arrays(merged)
    ta = _tri_area(V, T)
    for i, nm in enumerate(names):
        area[nm] = float(ta[TM == i].sum())
    # ... measured as coverage: every glowing face of lod 0 needs glowing surface close by, so a lit pane modelled as
    # a box may become one card and a flame a coarser flame, but no pane may go dark. lod 1: surface of the same
    # material (tints and pattern colours stay); lod 2: of the same kind of material (the tints of a leaded window
    # may merge into one)
    dg = max(0.04, 0.012 * ref["diag"])

    def _lib(nm):
        if nm in ref["libs"]:
            return ref["libs"][nm]
        m = bpy.data.materials.get(nm)
        return str(m.get("tsq_lib", "")) if m is not None else ""

    glow0 = [nm for nm, lib in ref["libs"].items() if lib.startswith(_lv.GLOW) and ref["mat_area"].get(nm, 0.0) > 1e-7]
    if level >= 2:
        fam = {}
        for nm in glow0:
            fam.setdefault(ref["libs"][nm], []).append(nm)
        groups = [(lib, n0, [n for n in names if _lib(n) == lib]) for lib, n0 in fam.items()]
    else:
        groups = [(nm, [nm], [nm] if nm in names else []) for nm in glow0]
    vl = None
    for label, n0, n1 in groups:
        a_ref = sum(ref["mat_area"][n] for n in n0)
        a_lv = sum(area.get(n, 0.0) for n in n1)
        kept = 0.0
        if a_lv > 0.0:
            if vl is None:
                vl = V.tolist()
            bvh = BVHTree.FromPolygons(vl, T[np.isin(TM, [names.index(n) for n in n1])].tolist(), all_triangles=True, epsilon=0.0)
            t0_ = ref["T"][np.isin(ref["TM"], [ref["names"].index(n) for n in n0])]
            a0 = _tri_area(ref["V"], t0_)
            cen = ref["V"][t0_].mean(axis=1)
            hit = np.array([bvh.find_nearest(c, dg)[0] is not None for c in cen.tolist()], dtype=bool)
            kept = float(a0[hit].sum() / max(a0.sum(), 1e-12))
        if kept < 0.8 or a_lv < 0.3 * a_ref:
            why.append(f"glowing material {label!r}: only {100 * kept:.0f} % of its lod 0 surface still glows "
                       f"({100 * a_lv / a_ref:.0f} % of the area; lamps, lit windows, neon and "
                       "stained glass must survive every level)")
    new = sorted(n for n in names if n and n not in ref["names"])
    if new:
        warn.append(f"lod {level} uses materials lod 0 does not have ({', '.join(new[:3])}): the colour will pop when the level switches")
    b0 = info0["bounds"]
    db = max(float(np.abs(mn - np.array(b0["min"])).max()), float(np.abs(mx - np.array(b0["max"])).max()))
    diag = ref["diag"]
    if db > max(0.08, 0.04 * diag):
        why.append(f"bounds differ from lod 0 by {db:.3f} m (footprint and silhouette must stay)")
    elif db > max(0.02, 0.01 * diag):
        warn.append(f"lod {level} bounds differ from lod 0 by {db:.3f} m")

    for o in merged:
        _weighted_normals(o)
        pn = o["tsq_part"]
        if pn != "body":
            pv = Vector(part_info[pn]["pivot"])
            o.data.transform(Matrix.Translation(-pv))
            o.location = pv
            o["tsq_part_props"] = json.dumps(part_info[pn]["props"])
    nbytes = 0
    if not why or keep_rejected:
        from .final import _gltf_kw
        bpy.ops.object.select_all(action='DESELECT')
        for o in merged:
            o.name = o.name + f"_lod{level}"
            o.select_set(True)
        lp = os.path.splitext(glb_path)[0] + f".lod{level}.glb"
        from .final import mark_keep
        from . import lodimage as _imk
        mark_keep(merged, None if not ref.get("text") else
                  {G["id"] for G in ref["text"] if not _imk.text_small(level, G["h"])})
        bpy.ops.export_scene.gltf(**dict(_gltf_kw(lp), use_selection=True))
        nbytes = os.path.getsize(lp)
    return {"level": level, "ok": not why, "why": why, "warn": warn, "tris": tris, "glb_bytes": nbytes,
            "ratio": round(tris / max(tris0, 1), 4), "budget": bud, "cover": round(cover, 3), "dev": round(dev, 4),
            "materials": mats, "text": tx, "seconds": round(build_seconds + time.time() - t0, 2)}
