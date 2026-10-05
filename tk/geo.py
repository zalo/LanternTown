"""tk.geo - parametric geometry builders (meters, Z up, front = -Y).

Everything returns mesh objects linked to the scene. Builders that know their parametrisation
(lathe, sweep, surface) write proper meter-scale UVs; the rest get box UVs unless you declare
otherwise with uv_grain / uv_cyl / uv_planar.
"""
import bpy
import bmesh
import math
import subprocess
import numpy as np
from math import radians, pi, sin, cos
from mathutils import Vector, Matrix, noise as mnoise

from .core import (_link, from_bmesh, mesh_object, set_loc_rot, world_matrix, apply_transform, _auto_rng, Rng, _LOD)
from .surf import realize, uv_keep, uv_grain, smooth
from . import levels as _lod

# Levels of detail (docs/LOD.md): every `if _LOD["level"]:` branch below runs only while a cheaper level (1 = mid,
# 2 = far) of a level-aware asset is being modelled. At level 0 - every ordinary build - none of them is taken.

# ----------------------------------------------------------------------------
# Modifiers (applied automatically in realize()/join()/export)
# ----------------------------------------------------------------------------


def bevel(obj, width=0.003, segments=2, angle=30.0, clamp=True, profile=0.5):
    """Bevel modifier on edges sharper than `angle`. Real objects have no razor edges: bevel
    anything that catches light (0.5-3 mm metal, 2-8 mm wood, 5-20 mm stone)."""
    m = obj.modifiers.new("Bevel", 'BEVEL')
    m.width = width
    m.segments = segments
    m.limit_method = 'ANGLE'
    m.angle_limit = radians(angle)
    m.use_clamp_overlap = clamp
    m.profile = profile
    m.miter_outer = 'MITER_ARC'
    if _LOD["level"]:
        w, s = _lod._bevel(width, segments)
        if s == 0:
            m.show_viewport = m.show_render = False   # too fine to see at this level: kept on the stack, not evaluated
        else:
            m.segments = s
            m.miter_outer = 'MITER_SHARP'
    return m


def subsurf(obj, levels=1, simple=False):
    m = obj.modifiers.new("Subsurf", 'SUBSURF')
    if _LOD["level"] and not isinstance(levels, _lod._XI):
        levels = max(1, levels - _LOD["level"])
    m.levels = levels
    m.render_levels = levels
    if simple:
        m.subdivision_type = 'SIMPLE'
    return m


def solidify(obj, thickness=0.002, offset=-1.0, even=True):
    m = obj.modifiers.new("Solidify", 'SOLIDIFY')
    m.thickness = thickness
    m.offset = offset
    m.use_even_offset = even
    m.use_quality_normals = True
    return m


def array(obj, count, offset=(1, 0, 0), relative=True, merge=False):
    m = obj.modifiers.new("Array", 'ARRAY')
    m.count = count
    m.use_relative_offset = relative
    m.use_constant_offset = not relative
    if relative:
        m.relative_offset_displace = offset
    else:
        m.constant_offset_displace = offset
    m.use_merge_vertices = merge
    return m


def weld(obj, dist=1e-5):
    m = obj.modifiers.new("Weld", 'WELD')
    m.merge_threshold = dist
    return m


def simple_deform(obj, method='BEND', angle_deg=45.0, axis='Z', factor=None):
    """Bend / twist / taper / stretch the object (subdivide first so there are verts to bend)."""
    m = obj.modifiers.new("Deform", 'SIMPLE_DEFORM')
    m.deform_method = method
    m.deform_axis = axis
    if method in ('BEND', 'TWIST'):
        m.angle = radians(angle_deg)
    else:
        m.factor = factor if factor is not None else 0.5
    return m


def boolean(target, cutter, op='DIFFERENCE', delete_cutter=True, solver='EXACT'):
    """Boolean (applied immediately). Slow on dense meshes: cut before bevelling/subdividing."""
    m = target.modifiers.new("Bool", 'BOOLEAN')
    m.operation = op
    m.object = cutter
    m.solver = solver
    uv = target.get("tsq_uv", None)
    sm = target.get("tsq_smooth", None)
    # apply only the modifier stack (keep declarations pending so UVs follow the cut surface)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(target.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = target.data
    target.modifiers.clear()
    target.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    if uv is not None:
        target["tsq_uv"] = uv
    if sm is not None:
        target["tsq_smooth"] = sm
    if delete_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


# ----------------------------------------------------------------------------
# Mesh edits
# ----------------------------------------------------------------------------

def subdivide(obj, cuts=1):
    """Uniformly subdivide all edges `cuts` times (immediate)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def densify(obj, max_edge=0.25):
    """Split edges longer than max_edge (meters) so vertex masks / displacement have resolution.
    Use on big flat walls, floors and long beams (immediate)."""
    sc = obj.scale
    if _LOD["level"] and not isinstance(max_edge, _lod._XF):
        max_edge = max_edge * (1.0, 1.6, 3.2)[_LOD["level"]]
    for _ in range(8):
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        long_edges = []
        for e in bm.edges:
            d = e.verts[0].co - e.verts[1].co
            if Vector((d.x * sc.x, d.y * sc.y, d.z * sc.z)).length > max_edge:
                long_edges.append(e)
        if not long_edges:
            bm.free()
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
        bm.to_mesh(obj.data)
        bm.free()
    return obj


def displace_noise(obj, strength=0.005, scale=0.2, seed=0, along_normal=True, axis_mask=(1, 1, 1)):
    """Organic irregularity: move vertices by smooth 3D noise (feature size ~`scale` m, amplitude
    `strength` m). Welded vertices stay welded. Hand-made things (stone, timber, plaster, forged
    iron) are never perfectly straight - use small amounts liberally."""
    me = obj.data
    if obj.modifiers:
        realize(obj)
        me = obj.data
    off = Vector((seed * 13.37, seed * 7.77, seed * 3.33))
    inv = 1.0 / max(scale, 1e-6)
    for v in me.vertices:
        p = Vector((v.co.x * obj.scale.x, v.co.y * obj.scale.y, v.co.z * obj.scale.z)) * inv + off
        if along_normal:
            d = mnoise.noise(p) * strength
            v.co += v.normal * d
        else:
            n = mnoise.noise_vector(p) * strength
            v.co += Vector((n.x * axis_mask[0], n.y * axis_mask[1], n.z * axis_mask[2]))
    me.update()
    return obj


def transform_verts(obj, fn):
    """Apply fn(co: Vector) -> Vector to every vertex (local space, immediate). For custom warps,
    sags, tapers."""
    for v in obj.data.vertices:
        v.co = fn(v.co.copy())
    obj.data.update()
    return obj


def recalc_normals(obj, inside=False):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    if inside:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bm.to_mesh(obj.data)
    bm.free()
    return obj


# ----------------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------------

def _bm_bevel(bm, width, segments, angle=30.0):
    if width <= 0:
        return
    if _LOD["level"]:
        width, segments = _lod._bevel(width, segments)
        if segments == 0:
            return
    lim = radians(angle)
    edges = [e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > lim]
    if edges:
        bmesh.ops.bevel(bm, geom=edges, offset=width, segments=segments, profile=0.5, affect='EDGES',
                        clamp_overlap=True)


def box(name, size=(1, 1, 1), loc=(0, 0, 0), rot=None, bevel=0.0, segments=2):
    """Box of full size (x, y, z) centred on loc. bevel = edge chamfer radius (m), baked in."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    _bm_bevel(bm, min(bevel, min(size) * 0.45), segments)
    o = from_bmesh(name, bm)
    set_loc_rot(o, loc, rot)
    return o


def cylinder(name, radius=0.5, depth=1.0, loc=(0, 0, 0), rot=None, verts=32, bevel=0.0, segments=2,
             cap=True, radius2=None):
    """Cylinder / cone frustum along local Z centred on loc. radius2 = top radius (default same)."""
    bm = bmesh.new()
    r2 = radius if radius2 is None else radius2
    sm = 50
    if _LOD["level"]:
        v0 = verts
        verts = _lod._segs(verts, max(abs(radius), abs(r2)))
        sm = _lod._round_smooth(sm, v0, verts)
    bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=verts, radius1=radius, radius2=r2, depth=depth)
    _bm_bevel(bm, bevel, segments, angle=50.0)
    o = from_bmesh(name, bm)
    set_loc_rot(o, loc, rot)
    o["tsq_uv"] = '{"mode": "cyl", "axis": "Z", "scale": 1.0, "offset": [0.0, 0.0], "radius": %f}' % max(radius, r2)
    smooth(o, sm)
    return o


def cone(name, r1=0.5, r2=0.0, depth=1.0, loc=(0, 0, 0), rot=None, verts=32):
    return cylinder(name, r1, depth, loc, rot, verts, radius2=r2)


def sphere(name, radius=0.5, loc=(0, 0, 0), segments=32, rings=16, scale=None):
    if _LOD["level"]:
        s3 = (scale, scale, scale) if isinstance(scale, (int, float)) else (scale or (1, 1, 1))
        seg0 = segments
        segments = _lod._segs(segments, radius * max(abs(c) for c in s3))
        if not isinstance(rings, _lod._XI) and segments != seg0:
            rings = max(3, min(rings, int(round(rings * segments / max(seg0, 1)))))
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    o = from_bmesh(name, bm)
    set_loc_rot(o, loc, None, scale)
    o["tsq_uv"] = '{"mode": "cyl", "axis": "Z", "scale": 1.0, "offset": [0.0, 0.0], "radius": %f}' % radius
    smooth(o, 180)
    return o


def torus(name, major=0.5, minor=0.05, loc=(0, 0, 0), rot=None, major_seg=48, minor_seg=12):
    """Torus around local Z. UVs: U along the ring (meters), V around the tube."""
    if _LOD["level"]:
        major_seg = _lod._segs(major_seg, major + minor)
        minor_seg = _lod._segs(minor_seg, minor)
    pts = [(major * cos(2 * pi * i / major_seg), major * sin(2 * pi * i / major_seg), 0.0) for i in range(major_seg)]
    with _lod.lod_as(0):    # the counts are final: sweep must not thin the ring again
        o = sweep(name, pts, minor, closed=True, sides=minor_seg, smooth_path=False)
    set_loc_rot(o, loc, rot)
    smooth(o, 180)
    return o


def lathe(name, profile, segments=48, loc=(0, 0, 0), rot=None, angle=360.0, smooth_angle=50.0):
    """Revolve a 2D profile [(radius, z), ...] around the Z axis, with meter-scale UVs
    (U around, V along the profile).

    Order the points bottom-to-top along the visible outer surface; points with radius 0 become
    poles. For hollow vessels continue the profile back down the inside to give wall thickness:
        [(0,0),(0.10,0),(0.11,0.01),(0.11,0.15),(0.105,0.15),(0.105,0.005),(0,0.005)]
    Build rich turned profiles (beads, coves, fillets) - they are nearly free and read as craft.
    """
    prof = [(float(r), float(z)) for r, z in profile]
    n = len(prof)
    full = abs(angle - 360.0) < 1e-6
    rmax = max(r for r, _ in prof) or 1.0
    vlen = [0.0]
    for k in range(1, n):
        vlen.append(vlen[-1] + math.hypot(prof[k][0] - prof[k - 1][0], prof[k][1] - prof[k - 1][1]))
    if _LOD["level"]:
        # fewer segments around (same faceting error for every radius) and a profile without the points that
        # move the outline by less than can be seen at this level; V coordinates stay those of the full profile
        frac = max(abs(angle) / 360.0, 1e-3)
        s_full = _lod._segs(type(segments)(max(1, round(segments / frac))) if not full else segments, rmax)
        if not isinstance(segments, _lod._XI):
            seg0 = segments
            segments = max(1, min(segments, int(math.ceil(s_full * frac - 1e-9))))
            smooth_angle = _lod._round_smooth(smooth_angle, seg0, segments, abs(angle))
        keep = _lod._dp_keep(prof, _lod.TOL[_LOD["level"]] * 0.4)
        keep = sorted(set(keep) | {k for k, (r, _) in enumerate(prof) if r <= 1e-9})
        if max(prof[k][0] for k in keep) <= 1e-9 < rmax:
            # a small turned thing (a candle flame, a stud) must not fall onto its own axis: its widest point stays
            keep = sorted(set(keep) | {max(range(n), key=lambda k: prof[k][0])})
        if len(keep) < n:
            prof = [prof[k] for k in keep]
            vlen = [vlen[k] for k in keep]
            n = len(prof)
    steps = segments if full else segments + 1
    verts, ring = [], []
    for s in range(steps):
        a = radians(angle) * s / segments
        row = []
        for (r, z) in prof:
            if r <= 1e-9:
                row.append(None)
            else:
                verts.append((r * cos(a), r * sin(a), z))
                row.append(len(verts) - 1)
        ring.append(row)
    poles = {}
    for k, (r, z) in enumerate(prof):
        if r <= 1e-9:
            verts.append((0.0, 0.0, z))
            poles[k] = len(verts) - 1
    faces, fuv = [], []
    circ = radians(angle) * rmax
    for s in range(segments):
        s2 = (s + 1) % steps if full else s + 1
        u0, u1 = circ * s / segments, circ * (s + 1) / segments
        for k in range(n - 1):
            a = poles[k] if k in poles else ring[s][k]
            b = poles[k] if k in poles else ring[s2][k]
            c = poles[k + 1] if (k + 1) in poles else ring[s2][k + 1]
            d = poles[k + 1] if (k + 1) in poles else ring[s][k + 1]
            quad = [(a, (u0, vlen[k])), (b, (u1, vlen[k])), (c, (u1, vlen[k + 1])), (d, (u0, vlen[k + 1]))]
            f, uvs = [], []
            for vi, uv in quad:
                if vi not in f:
                    f.append(vi)
                    uvs.append(uv)
            if len(f) >= 3:
                faces.append(tuple(f))
                fuv.append(uvs)
    o = _mesh_with_uv(name, verts, faces, fuv)
    # a profile drawn bottom-to-top on the outside yields outward normals; make sure
    recalc = bmesh.new()
    recalc.from_mesh(o.data)
    bmesh.ops.recalc_face_normals(recalc, faces=recalc.faces[:])
    recalc.to_mesh(o.data)
    recalc.free()
    set_loc_rot(o, loc, rot)
    smooth(o, smooth_angle)
    return o


def _mesh_with_uv(name, verts, faces, face_uvs):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update(calc_edges=True)
    uvl = me.uv_layers.new(name="UVMap")
    flat = np.array([c for uvs in face_uvs for uv in uvs for c in uv], dtype=np.float32)
    if len(flat) == len(me.loops) * 2:
        uvl.data.foreach_set("uv", flat)
    o = _link(bpy.data.objects.new(name, me))
    o["tsq_uv"] = "keep"
    return o


def _mesh_from_arrays(name, verts, quads, uvs=None, tris=None):
    """verts (N,3) float; quads (M,4) int; uvs per-loop (M*4,2) in loop order; optional extra tris."""
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts, dtype=np.float32)
    quads = np.asarray(quads, dtype=np.int32).reshape(-1, 4)
    nq = len(quads)
    nt = 0 if tris is None else len(tris)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    nloops = nq * 4 + nt * 3
    me.loops.add(nloops)
    lv = quads.ravel()
    if nt:
        lv = np.concatenate([lv, np.asarray(tris, dtype=np.int32).ravel()])
    me.loops.foreach_set("vertex_index", lv)
    me.polygons.add(nq + nt)
    starts = np.concatenate([np.arange(nq, dtype=np.int32) * 4, nq * 4 + np.arange(nt, dtype=np.int32) * 3])
    me.polygons.foreach_set("loop_start", starts)
    me.update(calc_edges=True)
    me.validate()
    if uvs is not None and len(me.loops) == nloops:
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.asarray(uvs, dtype=np.float32).ravel())
    o = _link(bpy.data.objects.new(name, me))
    if uvs is not None:
        o["tsq_uv"] = "keep"
    return o


def rounded_rect(w, d, r, seg=6):
    """2D rounded rectangle outline [(x, y), ...], CCW, centred."""
    r = max(1e-5, min(r, w / 2 - 1e-6, d / 2 - 1e-6))
    pts = []
    for cx, cy, a0 in [(w / 2 - r, d / 2 - r, 0), (-w / 2 + r, d / 2 - r, 90),
                       (-w / 2 + r, -d / 2 + r, 180), (w / 2 - r, -d / 2 + r, 270)]:
        for i in range(seg + 1):
            a = radians(a0 + 90 * i / seg)
            pts.append((cx + r * cos(a), cy + r * sin(a)))
    return pts


def circle_pts(r, n=24, a0=0.0, a1=360.0, center=(0, 0)):
    """2D arc/circle points. Full circle omits the duplicate end point."""
    full = abs((a1 - a0) - 360.0) < 1e-6
    m = n if full else n + 1
    return [(center[0] + r * cos(radians(a0 + (a1 - a0) * i / n)), center[1] + r * sin(radians(a0 + (a1 - a0) * i / n)))
            for i in range(m)]


def arch_pts(width, height, spring=None, kind="round", n=16):
    """2D outline of an arched opening (x across, y up), CCW starting bottom-right:
    kind: 'round' (semicircle), 'segmental' (shallow arc), 'pointed' (gothic), 'flat'.
    width = clear width, height = total height, spring = height where the arch starts
    (default: derived). Use for doors, windows, passages, niches."""
    w = width / 2
    if kind == "flat":
        return [(w, 0), (w, height), (-w, height), (-w, 0)]
    if kind == "round":
        sp = height - w if spring is None else spring
        rise = height - sp
        pts = [(w, 0)]
        for i in range(n + 1):
            a = pi * i / n
            pts.append((w * cos(a), sp + rise * sin(a)))
        pts.append((-w, 0))
        return pts
    if kind == "segmental":
        sp = height - w * 0.35 if spring is None else spring
        rise = height - sp
        R = (w * w + rise * rise) / (2 * rise)
        a_half = math.asin(min(1.0, w / R))
        pts = [(w, 0)]
        for i in range(n + 1):
            a = -a_half + 2 * a_half * i / n
            pts.append((-R * sin(a), sp + rise - R + R * cos(a)))
        pts.append((-w, 0))
        return pts
    if kind == "pointed":
        sp = height - w * 1.3 if spring is None else spring
        rise = height - sp
        # two arcs centred on the opposite springing points, scaled vertically to hit the apex
        pts = [(w, 0)]
        R = 2 * w
        a_top = math.acos(0.5)
        yscale = rise / (R * sin(a_top))
        for i in range(n // 2 + 1):
            a = a_top * i / (n // 2)
            pts.append((-w + R * cos(a), sp + R * sin(a) * yscale))
        for i in range(n // 2 - 1, -1, -1):
            a = a_top * i / (n // 2)
            pts.append((w - R * cos(a), sp + R * sin(a) * yscale))
        pts.append((-w, 0))
        return pts
    raise ValueError(kind)


def extrude_profile(name, pts2d, depth, loc=(0, 0, 0), rot=None, cap=True, bevel=0.0, segments=2):
    """Extrude a closed, simple 2D polygon [(x, y), ...] (CCW) along +Z by depth (base at z=0).
    For outlines with holes use shape_extrude()."""
    if _LOD["level"]:
        pts2d = _lod._simplify2d(pts2d, True)
    bm = bmesh.new()
    bottom = [bm.verts.new((x, y, 0.0)) for x, y in pts2d]
    top = [bm.verts.new((x, y, depth)) for x, y in pts2d]
    n = len(pts2d)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bottom[i], bottom[j], top[j], top[i]))
    if cap:
        bm.faces.new(list(reversed(bottom)))
        bm.faces.new(top)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    _bm_bevel(bm, bevel, segments, angle=35.0)
    o = from_bmesh(name, bm)
    set_loc_rot(o, loc, rot)
    return o


def shape_extrude(name, outer, holes=(), depth=0.01, loc=(0, 0, 0), rot=None, bevel=0.0, bevel_res=1):
    """Extrude a 2D shape WITH HOLES: outer / each hole is a list of (x, y). The shape lies in the
    local XY plane and is extruded from z=0 to z=depth. `bevel` rounds the rim (the outline grows
    by `bevel`). Great for cut-out sign boards, pierced ironwork, tracery, plates with slots.
    Tip: rotate with rot=(90,0,0) to stand the shape up facing -Y."""
    if _LOD["level"]:
        outer = _lod._simplify2d(outer, True)
        holes = [_lod._simplify2d(h, True) for h in holes]
        bevel, bs = _lod._bevel(bevel, bevel_res + 1)
        if bs == 0:
            bevel = 0.0
        elif bs == 1:
            bevel_res = 0
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '2D'
    cu.fill_mode = 'BOTH'
    cu.extrude = max(depth / 2 - bevel, 0.0) if bevel > 0 else depth / 2
    cu.bevel_depth = bevel
    cu.bevel_resolution = bevel_res
    for loop in [outer] + list(holes):
        sp = cu.splines.new('POLY')
        sp.points.add(len(loop) - 1)
        for pt, p in zip(sp.points, loop):
            pt.co = (p[0], p[1], 0.0, 1.0)
        sp.use_cyclic_u = True
    tmp = _link(bpy.data.objects.new(name + "_cu", cu))
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg), depsgraph=dg)
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.curves.remove(cu)
    me.transform(Matrix.Translation((0, 0, depth / 2)))
    o = _link(bpy.data.objects.new(name, me))
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    bm.to_mesh(me)
    bm.free()
    set_loc_rot(o, loc, rot)
    return o


def rounded_box(name, size=(1, 1, 1), radius=0.02, loc=(0, 0, 0), rot=None, seg=5, bevel=0.0):
    """Box with rounded vertical edges (radius) and optional small bevel on top/bottom rims."""
    bseg = 2
    if _LOD["level"]:
        if not isinstance(seg, _lod._XI):
            seg = max(1, int(round(_lod._segs(4 * seg, radius) / 4.0)))
        bevel, bseg = _lod._bevel(bevel, bseg)
        with _lod.lod_as(0):
            o = extrude_profile(name, rounded_rect(size[0], size[1], radius, seg), size[2], bevel=0.0)
    else:
        o = extrude_profile(name, rounded_rect(size[0], size[1], radius, seg), size[2], bevel=0.0)
    for v in o.data.vertices:
        v.co.z -= size[2] / 2
    if bevel > 0 and bseg > 0:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        edges = [e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > radians(60)]
        bmesh.ops.bevel(bm, geom=edges, offset=bevel, segments=bseg, profile=0.5, affect='EDGES', clamp_overlap=True)
        bm.to_mesh(o.data)
        bm.free()
    set_loc_rot(o, loc, rot)
    return o


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

def resample_path(points, step=None, n=None, closed=False, smooth_path=True):
    """Resample a polyline to roughly uniform spacing (Catmull-Rom smoothed when smooth_path)."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 2:
        return P
    if smooth_path and len(P) > 2:
        if closed:
            ext = np.vstack([P[-1], P, P[0], P[1]])
        else:
            ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        segs = len(ext) - 3
        sub = 12
        t = np.linspace(0, 1, sub, endpoint=False)[:, None]
        out = []
        for i in range(segs):
            p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        dense = np.vstack(out)
        if not closed:
            dense = np.vstack([dense, P[-1]])
    else:
        dense = P
    loop = np.vstack([dense, dense[0]]) if closed else dense
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return dense[:1]
    if n is None:
        if step is None:
            if not smooth_path:
                return dense
            # sparse control polylines are subdivided (x6, up to 32 samples); dense paths keep their own count
            step = total / max(8, len(P), min(len(P) * 6, 32))
        n = max(2, int(round(total / step)) + (0 if closed else 1))
    ts = np.linspace(0, total, n, endpoint=not closed)
    out = np.stack([np.interp(ts, cum, loop[:, k]) for k in range(3)], axis=1)
    return out


def path_length(points, closed=False):
    P = np.asarray(points, dtype=np.float64)
    if closed:
        P = np.vstack([P, P[0]])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def spiral_path(r0, r1, turns=1.5, n=48, a0_deg=0.0, plane="XZ", center=(0, 0, 0), ccw=True, z_rise=0.0):
    """Scroll / volute path from radius r0 to r1 over `turns`. plane: 'XY','XZ','YZ'. The classic
    wrought-iron scroll: sweep a flat bar profile along it."""
    pts = []
    for i in range(n + 1):
        t = i / n
        a = radians(a0_deg) + (1 if ccw else -1) * 2 * pi * turns * t
        r = r0 + (r1 - r0) * t
        u, v, w = r * cos(a), r * sin(a), z_rise * t
        if plane == "XY":
            p = (u, v, w)
        elif plane == "XZ":
            p = (u, w, v)
        else:
            p = (w, u, v)
        pts.append((p[0] + center[0], p[1] + center[1], p[2] + center[2]))
    return pts


def arc_path(radius, a0_deg, a1_deg, n=16, plane="XZ", center=(0, 0, 0)):
    pts = []
    for i in range(n + 1):
        a = radians(a0_deg + (a1_deg - a0_deg) * i / n)
        u, v = radius * cos(a), radius * sin(a)
        p = (u, v, 0) if plane == "XY" else ((u, 0, v) if plane == "XZ" else (0, u, v))
        pts.append((p[0] + center[0], p[1] + center[1], p[2] + center[2]))
    return pts


def catenary(p0, p1, sag=0.2, n=24):
    """Hanging rope/chain/wire between two points with the given sag (m) at mid-span."""
    p0, p1 = Vector(p0), Vector(p1)
    pts = []
    for i in range(n + 1):
        t = i / n
        p = p0.lerp(p1, t)
        p.z -= sag * (1 - (2 * t - 1) ** 2)
        pts.append(tuple(p))
    return pts


def bezier_path(p0, p1, p2, p3, n=16):
    P = [Vector(p) for p in (p0, p1, p2, p3)]
    out = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3]
        out.append(tuple(a))
    return out


def _frames(P, closed, up):
    n = len(P)
    T = np.zeros_like(P)
    if closed:
        T = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    else:
        T[1:-1] = P[2:] - P[:-2]
        T[0] = P[1] - P[0]
        T[-1] = P[-1] - P[-2]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    upv = np.asarray(up, dtype=np.float64)
    upv = upv / np.linalg.norm(upv)
    t0 = T[0]
    if abs(np.dot(t0, upv)) > 0.98:
        upv = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
    N = np.zeros_like(P)
    B = np.zeros_like(P)
    b = np.cross(t0, upv)
    b /= np.linalg.norm(b)
    nrm = np.cross(b, t0)
    N[0], B[0] = nrm, b
    for i in range(1, n):
        v = np.cross(T[i - 1], T[i])
        s = np.linalg.norm(v)
        c = np.clip(np.dot(T[i - 1], T[i]), -1, 1)
        if s < 1e-9:
            N[i] = N[i - 1]
        else:
            ax = v / s
            ang = math.atan2(s, c)
            nn = N[i - 1]
            N[i] = nn * cos(ang) + np.cross(ax, nn) * sin(ang) + ax * np.dot(ax, nn) * (1 - cos(ang))
        N[i] -= T[i] * np.dot(N[i], T[i])
        N[i] /= max(np.linalg.norm(N[i]), 1e-12)
        B[i] = np.cross(T[i], N[i])
    return T, N, B


def sweep(name, path, profile=0.01, closed=False, sides=10, smooth_path=True, step=None, cap=True,
          up=(0, 0, 1), scale_fn=None, twist_deg=0.0):
    """Sweep a profile along a 3D path -> tube / bar / moulding with meter UVs (U along the path).

    path     : [(x,y,z), ...]
    profile  : float = round rod radius (with `sides`), or [(a, b), ...] closed CCW cross-section
               in the path's (side, up) frame, e.g. rounded_rect(0.03, 0.008, 0.002, 2) for flat bar,
               or any moulding / gutter / handrail section.
    scale_fn : optional fn(t in 0..1) -> scale of the cross-section (tapers: scroll tips, branches).
    twist_deg: total twist along the path (twisted square bar balusters!).
    up       : reference 'up' for the profile's b axis at the path start.
    """
    P = resample_path(path, step=step, closed=closed, smooth_path=smooth_path)
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    if n < 2:
        raise ValueError("sweep path needs at least 2 points")
    T, N, B = _frames(P, closed, up)
    L = _LOD["level"]
    if isinstance(profile, (int, float)):
        r = float(profile)
        if L:
            sides = _lod._segs(sides, r)
        prof = np.array([(r * cos(2 * pi * k / sides), r * sin(2 * pi * k / sides)) for k in range(sides)])
        round_prof = True
    else:
        if L:
            profile = _lod._simplify2d(list(profile), True)
        prof = np.asarray(profile, dtype=np.float64)
        round_prof = False
    m = len(prof)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[0]]) if closed else P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    per = np.linalg.norm(np.diff(np.vstack([prof, prof[0]]), axis=0), axis=1)
    pcum = np.concatenate([[0], np.cumsum(per)])
    verts = np.zeros((n, m, 3))
    for i in range(n):
        t = cum[i] / total if total > 0 else 0.0
        s = scale_fn(t) if scale_fn else 1.0
        a = radians(twist_deg) * t
        ca, sa = cos(a), sin(a)
        side = B[i] * ca + N[i] * sa
        upv = -B[i] * sa + N[i] * ca
        verts[i] = P[i] + s * (prof[:, 0:1] * side + prof[:, 1:2] * upv)
    if L and n > (3 if closed else 2):
        # keep only the cross-sections the shape needs at this level: the rings that stay are exactly rings of the
        # full model (same frames, taper, twist and U coordinate), so joints with other parts still meet
        keep = _lod._ring_keep(verts, closed, radians(twist_deg) * (np.asarray(cum[:n]) / total if total > 0 else np.zeros(n)),
                               uniform=step is not None)
        if len(keep) < n:
            verts = verts[keep]
            cum = np.concatenate([cum[keep], [total]]) if closed else cum[keep]
            n = len(keep)
    rings = n if closed else n - 1
    i0 = np.arange(rings)[:, None]
    i1 = (i0 + 1) % n
    k0 = np.arange(m)[None, :]
    k1 = (k0 + 1) % m
    quads = np.stack([i0 * m + k0, i0 * m + k1, i1 * m + k1, i1 * m + k0], axis=2).reshape(-1, 4)
    u0 = cum[:-1][:rings] if not closed else cum[:rings]
    u1 = cum[1:rings + 1]
    U0 = np.repeat(u0[:, None], m, axis=1)
    U1 = np.repeat(u1[:, None], m, axis=1)
    V0 = np.repeat(pcum[:-1][None, :], rings, axis=0)
    V1 = np.repeat(pcum[1:][None, :], rings, axis=0)
    uvs = np.stack([np.stack([U0, V0], -1), np.stack([U0, V1], -1), np.stack([U1, V1], -1), np.stack([U1, V0], -1)],
                   axis=2).reshape(-1, 2)
    o = _mesh_from_arrays(name, verts.reshape(-1, 3), quads, uvs)
    if cap and not closed:
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bm.verts.ensure_lookup_table()
        uvl = bm.loops.layers.uv.verify()
        for ring_i, flip in ((0, False), (n - 1, True)):
            vs = [bm.verts[ring_i * m + k] for k in range(m)]
            if flip:
                vs = list(reversed(vs))
            try:
                f = bm.faces.new(vs)
                for l in f.loops:
                    k = (l.vert.index - ring_i * m)
                    l[uvl].uv = (float(prof[k][0]), float(prof[k][1]))
            except ValueError:
                pass
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(o.data)
        bm.free()
    else:
        recalc_normals(o)
    smooth(o, 180 if round_prof else 40)
    return o


def chain(name, path, link_len=0.04, wire=0.004, link_w=None, sides=6):
    """Chain of oval links along a path (alternating 90 deg). Returns one mesh."""
    P = resample_path(path, step=link_len * 0.72, smooth_path=True)
    link_w = link_w or link_len * 0.62
    L = _LOD["level"]
    if L == 2 or (L == 1 and link_len < 0.02):
        # too far to see links: one thin strand along the same path
        o = sweep(name, [tuple(p) for p in P], max(wire * 1.5, link_w * 0.3), sides=_lod._XI(3), smooth_path=False)
        smooth(o, 180)
        return o
    half_l, half_w = link_len / 2 - wire, link_w / 2 - wire
    base_pts = []
    nseg = 16
    if L:
        nseg, sides = 6, _lod._XI(3)
    for i in range(nseg):
        a = 2 * pi * i / nseg
        # stadium-ish oval
        x = half_l * cos(a) * (1.0 if abs(cos(a)) > 0.3 else 1.0)
        y = half_w * sin(a)
        base_pts.append((x, y, 0.0))
    with _lod.lod_as(0):
        base = sweep(name + "_link", base_pts, wire, closed=True, sides=sides, smooth_path=False)
    mats = []
    T, N, B = _frames(np.asarray(P, dtype=np.float64), False, (0, 0, 1))
    for i in range(len(P)):
        t, nn, b = T[i], N[i], B[i]
        if i % 2:
            nn, b = b, -nn
        M = Matrix(((t[0], b[0], nn[0], P[i][0]), (t[1], b[1], nn[1], P[i][1]), (t[2], b[2], nn[2], P[i][2]), (0, 0, 0, 1)))
        mats.append(M)
    o = instances_to_mesh(name, base, mats, uv_jitter=False)
    bpy.data.objects.remove(base, do_unlink=True)
    smooth(o, 180)
    return o


# ----------------------------------------------------------------------------
# Parametric surfaces
# ----------------------------------------------------------------------------

def surface(name, fn, nu=16, nv=16, closed_u=False, closed_v=False, thickness=0.0, uv_len=None):
    """Mesh from a parametric function fn(u, v) -> (x, y, z), u, v in [0, 1]. UVs are arc-length
    based (meters) unless uv_len=(Lu, Lv) is given. thickness > 0 adds a solidify modifier.
    Use for cloth (awnings, flags, sacks, tablecloths), terrain, curved roofs, leaves."""
    if _LOD["level"]:
        nu = _lod._half(nu, 3 if closed_u else 1)
        nv = _lod._half(nv, 3 if closed_v else 1)
    cu = nu + (0 if closed_u else 1)
    cv = nv + (0 if closed_v else 1)
    pts = np.zeros((cu, cv, 3))
    for i in range(cu):
        for j in range(cv):
            pts[i, j] = fn(i / nu, j / nv)
    if uv_len is None:
        du = np.linalg.norm(np.diff(pts, axis=0), axis=2).sum(axis=0).mean() if cu > 1 else 1.0
        dv = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(axis=1).mean() if cv > 1 else 1.0
        uv_len = (du * (nu / max(cu - 1, 1)), dv * (nv / max(cv - 1, 1)))
    i0 = np.arange(nu)[:, None]
    j0 = np.arange(nv)[None, :]
    i1 = (i0 + 1) % cu
    j1 = (j0 + 1) % cv
    quads = np.stack([i0 * cv + j0, i1 * cv + j0, i1 * cv + j1, i0 * cv + j1], axis=2).reshape(-1, 4)
    U0, U1 = i0 / nu * uv_len[0], (i0 + 1) / nu * uv_len[0]
    V0, V1 = j0 / nv * uv_len[1], (j0 + 1) / nv * uv_len[1]
    U0, U1 = np.broadcast_to(U0, (nu, nv)), np.broadcast_to(U1, (nu, nv))
    V0, V1 = np.broadcast_to(V0, (nu, nv)), np.broadcast_to(V1, (nu, nv))
    uvs = np.stack([np.stack([U0, V0], -1), np.stack([U1, V0], -1), np.stack([U1, V1], -1), np.stack([U0, V1], -1)],
                   axis=2).reshape(-1, 2)
    o = _mesh_from_arrays(name, pts.reshape(-1, 3), quads, uvs)
    if thickness > 0:
        solidify(o, thickness, offset=0.0)
    smooth(o, 60)
    return o


def grid(name, size=(1, 1), cuts=(8, 8), loc=(0, 0, 0), rot=None):
    """Flat XY grid (z=0), centred, with meter UVs. Displace / deform it into whatever you need."""
    sx, sy = size
    o = surface(name, lambda u, v: ((u - 0.5) * sx, (v - 0.5) * sy, 0.0), cuts[0], cuts[1], uv_len=(sx, sy))
    set_loc_rot(o, loc, rot)
    return o


# ----------------------------------------------------------------------------
# Instancing into one mesh (fast)
# ----------------------------------------------------------------------------

def instances_to_mesh(name, base, matrices, uv_jitter=True, seed=None, keep_base=True):
    """Stamp `base` (a mesh object, its own transform ignored) at every 4x4 matrix into ONE mesh.
    Fast (numpy) - thousands of roof tiles, leaves, cobbles, bricks, rivets, shingles, petals.
    uv_jitter gives each copy a random UV offset so copies don't share the same texture patch.
    Materials, UVs and masks of the base are preserved. Give each copy individuality through its
    matrix (random rotation / non-uniform scale / tilt)."""
    sm_spec = base.get("tsq_smooth", 40.0)
    if sm_spec != "keep":
        base["_tsq_smooth_angle"] = float(sm_spec)
    realize(base)
    _lod._tag_ident(base)
    me0 = base.data
    nv, nl, npoly = len(me0.vertices), len(me0.loops), len(me0.polygons)
    k = len(matrices)
    if k == 0 or nv == 0:
        return mesh_object(name, [], [])
    co = np.empty(nv * 3, dtype=np.float32)
    me0.vertices.foreach_get("co", co)
    co = np.concatenate([co.reshape(-1, 3), np.ones((nv, 1), dtype=np.float32)], axis=1)
    M = np.array([np.array(m) for m in matrices], dtype=np.float32)  # (k,4,4)
    allco = np.einsum("kij,nj->kni", M, co)[:, :, :3].reshape(-1, 3)
    lv = np.empty(nl, dtype=np.int32)
    me0.loops.foreach_get("vertex_index", lv)
    ls = np.empty(npoly, dtype=np.int32)
    me0.polygons.foreach_get("loop_start", ls)
    lt = np.empty(npoly, dtype=np.int32)
    me0.polygons.foreach_get("loop_total", lt)
    mi = np.empty(npoly, dtype=np.int32)
    me0.polygons.foreach_get("material_index", mi)
    offs_v = (np.arange(k, dtype=np.int32) * nv)[:, None]
    offs_l = (np.arange(k, dtype=np.int32) * nl)[:, None]
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv * k)
    me.vertices.foreach_set("co", allco.astype(np.float32).ravel())
    me.loops.add(nl * k)
    me.loops.foreach_set("vertex_index", (lv[None, :] + offs_v).ravel())
    me.polygons.add(npoly * k)
    me.polygons.foreach_set("loop_start", (ls[None, :] + offs_l).ravel())
    me.polygons.foreach_set("material_index", np.tile(mi, k))
    for m in me0.materials:
        me.materials.append(m)
    me.update(calc_edges=True)
    # flip winding for mirrored instances
    dets = np.linalg.det(M[:, :3, :3])
    if me0.uv_layers:
        uv = np.empty(nl * 2, dtype=np.float32)
        me0.uv_layers[0].data.foreach_get("uv", uv)
        uv = np.tile(uv.reshape(1, nl, 2), (k, 1, 1))
        if uv_jitter:
            r = np.random.default_rng(seed if seed is not None else _auto_rng("inst", name).randrange(1 << 30))
            uv = uv + r.uniform(0, 5.0, size=(k, 1, 2)).astype(np.float32)
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", uv.ravel())
    for an in ("m_grime", "m_wear", "m_special"):
        a0 = me0.attributes.get(an)
        if a0 is not None:
            vals = np.empty(nv, dtype=np.float32)
            a0.data.foreach_get("value", vals)
            me.attributes.new(an, 'FLOAT', 'POINT').data.foreach_set("value", np.tile(vals, k))
    for an in ("tsq_text", "tsq_id"):           # lettering stays lettering (see _tag_text), identity parts too
        a0 = me0.attributes.get(an)
        if a0 is not None and a0.domain == 'FACE' and len(a0.data) == npoly:
            vals = np.empty(npoly, dtype=np.int32)
            a0.data.foreach_get("value", vals)
            me.attributes.new(an, 'INT', 'FACE').data.foreach_set("value", np.tile(vals, k))
    sm = np.empty(npoly, dtype=bool)
    me0.polygons.foreach_get("use_smooth", sm)
    me.polygons.foreach_set("use_smooth", np.tile(sm, k))
    # sharp edges: copy via attribute if present
    se = me0.attributes.get("sharp_edge")
    if se is not None and len(me.edges) == len(me0.edges) * k:
        # edge order is not guaranteed identical; recompute by angle instead
        pass
    o = _link(bpy.data.objects.new(name, me))
    if (dets < 0).any():
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bad = [bm.faces[i * npoly + j] for i in np.nonzero(dets < 0)[0] for j in range(npoly)]
        bmesh.ops.reverse_faces(bm, faces=bad)
        bm.to_mesh(me)
        bm.free()
    o["tsq_uv"] = "keep" if me0.uv_layers else '{"mode": "box", "scale": 1.0, "offset": [0, 0], "rot90": false}'
    o["tsq_smooth"] = float(base.get("_tsq_smooth_angle", 40.0))
    if not keep_base:
        bpy.data.objects.remove(base, do_unlink=True)
    return o


def trs(loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
    """4x4 matrix from location, Euler degrees and scale (float or 3-tuple) - for instances_to_mesh."""
    from mathutils import Euler
    s = (scale, scale, scale) if isinstance(scale, (int, float)) else scale
    return (Matrix.Translation(Vector(loc)) @ Euler(tuple(radians(a) for a in rot)).to_matrix().to_4x4()
            @ Matrix.Diagonal((s[0], s[1], s[2], 1.0)))


# ----------------------------------------------------------------------------
# Joining
# ----------------------------------------------------------------------------

def join(objs, name=None):
    """Merge objects into one mesh (world space, identity transform). Modifiers are applied and
    UV / smoothing declarations resolved per part first, so each part keeps its own mapping.
    Returns the new object; the inputs are removed."""
    objs = [o for o in objs if o is not None and o.type == 'MESH']
    if not objs:
        return None
    realize(objs)
    for o in objs:
        _lod._tag_ident(o)
    bpy.context.view_layer.update()
    src = _lod._where_src(objs) if _LOD.get("where") is not None else None
    bm = bmesh.new()
    mats = []
    for o in objs:
        me = o.data
        tmp = me.copy()
        mw = world_matrix(o)
        tmp.transform(mw)
        if mw.determinant() < 0:
            tmp.flip_normals()
        remap = []
        for m in (list(me.materials) or [None]):
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        nfaces0 = len(bm.faces)
        bm.from_mesh(tmp)
        bm.faces.ensure_lookup_table()
        for f in bm.faces[nfaces0:]:
            f.material_index = remap[min(f.material_index, len(remap) - 1)]
        bpy.data.meshes.remove(tmp)
    me = bpy.data.meshes.new(name or objs[0].name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    nm = name or objs[0].name
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)
    j = _link(bpy.data.objects.new(nm, me))
    j["tsq_uv"] = "keep"
    j["tsq_smooth"] = "keep"
    if src is not None:
        _LOD["where"][j.name] = src
    return j


# ----------------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------------

_FONT_QUERIES = {
    "serif": "C059:bold", "serif_regular": "C059", "roman": "Nimbus Roman:bold", "sans": "Nimbus Sans:bold",
    "sans_narrow": "Nimbus Sans Narrow:bold", "script": "Z003", "noto_serif": "Noto Serif:bold",
    "liberation_serif": "Liberation Serif:bold", "mono": "Liberation Mono:bold",
}
_FONT_CACHE = {}
_TEXT_N = [0]           # text() calls of this process (never reset: the ids only have to differ within one build)


def _tag_text(me):
    """Mark the faces of a lettering mesh (still in the text's own frame: +Z looks at the reader) with the face
    attribute tsq_text = 4 x (number of the text() call) + 1 front face | 2 side or bevel | 3 back face. tk.reduce
    treats tagged faces as lettering (never welded, patched or collapsed) and tk.lodimage gates the levels on them.
    The attribute is not exported (tk.final exports only attributes whose name starts with '_')."""
    n = len(me.polygons)
    if not n:
        return
    _TEXT_N[0] += 1
    nz = np.empty(n * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", nz)
    nz = nz.reshape(-1, 3)[:, 2]
    v = np.full(n, 2, dtype=np.int32)
    v[nz > 0.9] = 1
    v[nz < -0.9] = 3
    me.attributes.new("tsq_text", 'INT', 'FACE').data.foreach_set("value", v + 4 * _TEXT_N[0])


def _font(key):
    """Vector font for a key in _FONT_QUERIES (or any fontconfig pattern). Cached by file path."""
    path = _FONT_CACHE.get(key)
    if path is None:
        try:
            q = _FONT_QUERIES.get(key, key)
            path = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            path = ""
        _FONT_CACHE[key] = path
    if not path:
        return None
    for f in bpy.data.fonts:
        if f.filepath == path:
            return f
    try:
        return bpy.data.fonts.load(path)
    except Exception:
        return None


def text(name, body, size=0.1, depth=0.004, loc=(0, 0, 0), rot=(90, 0, 0), align='CENTER', font="serif",
         bevel=0.0, spacing=1.0):
    """Raised 3D lettering as a mesh. Default rotation stands it up facing -Y (readable from the
    front); the text sits on the plane and extrudes `depth` towards the viewer.
    font: 'serif' | 'serif_regular' | 'roman' | 'sans' | 'sans_narrow' | 'script' | 'noto_serif'.
    size = cap height-ish (m). Returns a mesh object (box UVs)."""
    cu = bpy.data.curves.new(name, 'FONT')
    cu.body = body
    cu.size = size
    cu.extrude = depth / 2
    cu.bevel_depth = bevel
    cu.bevel_resolution = 1
    cu.align_x = align
    cu.align_y = 'CENTER'
    cu.space_character = spacing
    cu.resolution_u = 4
    zoff = depth / 2
    L = _LOD["level"]
    if L:
        # lettering always survives; it only gets coarser curves, and thin raised text becomes its flat front face
        cu.resolution_u = 2 if L == 1 else 1
        bw, bs = _lod._bevel(bevel, 2)
        cu.bevel_depth = bw if bs else 0.0
        cu.bevel_resolution = 0
        if depth < (0.008 if L == 1 else 0.06) and not isinstance(depth, _lod._XF):
            cu.extrude = 0.0
            cu.bevel_depth = 0.0
            cu.fill_mode = 'FRONT'
            zoff = depth
    f = _font(font)
    if f is not None:
        cu.font = f
    tmp = _link(bpy.data.objects.new(name + "_txt", cu))
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg), depsgraph=dg)
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.curves.remove(cu)
    me.transform(Matrix.Translation((0, 0, zoff)))
    _tag_text(me)
    o = _link(bpy.data.objects.new(name, me))
    set_loc_rot(o, loc, rot)
    o["tsq_smooth"] = 30.0
    return o


# ----------------------------------------------------------------------------
# Craft parts
# ----------------------------------------------------------------------------

def plank(name, length=1.0, width=0.12, thick=0.025, loc=(0, 0, 0), rot=None, bevel=0.003, warp=0.0,
          rng=None, axis='X', end_mat=None):
    """A sawn timber board/beam running along `axis` with wood-grain UVs along its length, slightly
    eased edges, optional warp (m of bow/twist) and per-plank random texture offset.
    end_mat: optional material for the two end-grain faces (e.g. 'wood_endgrain')."""
    rng = rng or _auto_rng("plank", name)
    size = {'X': (length, width, thick), 'Y': (width, length, thick), 'Z': (width, thick, length)}[axis]
    o = box(name, size, bevel=min(bevel, thick * 0.3, width * 0.3), segments=2)
    if warp > 0:
        ai = 'XYZ'.index(axis)
        cuts = max(2, int(length / 0.25))
        if _LOD["level"]:
            cuts = max(1, cuts // 2) if _LOD["level"] == 1 else 1
        bm = bmesh.new()
        bm.from_mesh(o.data)
        edges = [e for e in bm.edges if abs((e.verts[0].co - e.verts[1].co)[ai]) > length * 0.5]
        bmesh.ops.subdivide_edges(bm, edges=edges, cuts=cuts)
        bm.to_mesh(o.data)
        bm.free()
        bow = rng.uniform(-warp, warp)
        tw = rng.uniform(-warp, warp) / max(width, 1e-3)
        oi = [i for i in range(3) if i != ai]
        for v in o.data.vertices:
            t = v.co[ai] / length  # -0.5..0.5
            a, b = v.co[oi[0]], v.co[oi[1]]
            ang = tw * t
            v.co[oi[0]] = a * cos(ang) - b * sin(ang)
            v.co[oi[1]] = a * sin(ang) + b * cos(ang) + bow * (1 - (2 * t) ** 2)
    uv_grain(o, axis)
    o["tsq_smooth"] = 35.0
    set_loc_rot(o, loc, rot)
    if end_mat is not None:
        o["_end_mat"] = end_mat if isinstance(end_mat, str) else end_mat.name
        o["_end_axis"] = 'XYZ'.index(axis)
    return o


def stone_block(name, size=(0.4, 0.25, 0.2), loc=(0, 0, 0), rot=None, chamfer=0.012, rough=0.004, rng=None,
                detail=0.08):
    """A dressed stone block: chamfered, subdivided and slightly irregular (each face a little out
    of true, arrises knocked). size is the nominal full size. Use for quoins, copings, steps,
    kerbs, sills, plinths - individually modelled stones read far better than a textured box."""
    rng = rng or _auto_rng("stone", name)
    o = box(name, size, bevel=min(chamfer, min(size) * 0.2), segments=2)
    if rough > 0:
        if _LOD["level"] < 2:   # far level: only the corners move (same noise, so the block keeps its lean)
            densify(o, max(detail, max(size) / 12))
        displace_noise(o, strength=rough, scale=max(size) * 0.6, seed=rng.randrange(1000))
        displace_noise(o, strength=rough * 0.5, scale=max(size) * 0.15, seed=rng.randrange(1000))
    o["tsq_smooth"] = 45.0
    set_loc_rot(o, loc, rot)
    return o


def bolt(name, radius=0.008, height=0.006, loc=(0, 0, 0), rot=None, kind="hex"):
    """Fastener head sitting on z=0 pointing +Z. kind: 'hex' | 'dome' (rivet / clout nail) |
    'square' | 'pyramid' (forged nail/stud). Stamp many with instances_to_mesh."""
    L = _LOD["level"]
    if L and not isinstance(radius, _lod._XF):
        if radius < (0.012 if L == 1 else 0.04):
            o = mesh_object(name, [], [])       # a fastener this small is below a pixel at this level
            set_loc_rot(o, loc, rot)
            return o
        n = 6 if kind in ("hex", "dome") else 4
        o = cylinder(name, radius, height, verts=_lod._XI(n), radius2=radius * (0.25 if kind == "pyramid" else 0.8),
                     loc=(0, 0, height / 2))
        apply_transform(o)
        o["tsq_smooth"] = 30.0
        set_loc_rot(o, loc, rot)
        return o
    if kind == "dome":
        prof = [(0, height)]
        for i in range(1, 5):
            a = (pi / 2) * (1 - i / 4)
            prof.append((radius * cos(a), height * sin(a)))
        prof = list(reversed(prof))
        o = lathe(name, prof, segments=12)
    elif kind == "pyramid":
        o = cylinder(name, radius, height, verts=4, radius2=radius * 0.25, loc=(0, 0, height / 2))
        apply_transform(o)
        o["tsq_smooth"] = 20.0
    else:
        n = 6 if kind == "hex" else 4
        o = cylinder(name, radius, height, verts=n, bevel=min(radius, height) * 0.15, loc=(0, 0, height / 2))
        apply_transform(o)
        o["tsq_smooth"] = 30.0
    set_loc_rot(o, loc, rot)
    return o


def ring_band(name, radius=0.3, width=0.04, thick=0.003, loc=(0, 0, 0), rot=None, segments=48, bulge=0.0):
    """Flat hoop / band around Z (barrel hoops, straps, collars). width = height of the band."""
    r0, r1 = radius, radius + thick
    prof = [(r0, -width / 2), (r1, -width / 2), (r1 + bulge, 0.0), (r1, width / 2), (r0, width / 2)]
    o = lathe(name, prof, segments=segments, smooth_angle=35)
    set_loc_rot(o, loc, rot)
    return o
