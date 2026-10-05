"""Route B: can Blender 4.1's (legacy) EEVEE bake an irradiance volume headless, and can Python read the result?

    blender --background --factory-startup --python tools/probetest/bl_eevee.py -- scratch=<dir>

Builds a small closed test (floor, red wall, one point light), adds an irradiance volume (4 x 4 x 4), runs
bpy.ops.scene.light_cache_bake() in background mode and then looks for the baked coefficients everywhere Python could
reach them. Prints one line "EEVEE {...}".
"""
import bpy, sys, os, json, time, traceback

A = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--") + 1:])
SCRATCH = A["scratch"]
R = {"blender": bpy.app.version_string}
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.temporary_directory = os.path.join(SCRATCH, "tmp")
sc = bpy.context.scene
R["engines"] = [e.identifier for e in type(sc.render).bl_rna.properties["engine"].enum_items]
try:
    sc.render.engine = "BLENDER_EEVEE"
except TypeError as e:
    R["engine_error"] = str(e)
R["engine"] = sc.render.engine

bpy.ops.mesh.primitive_plane_add(size=10)
bpy.ops.mesh.primitive_cube_add(size=1, location=(2, 0, 1.5))
bpy.context.object.scale = (0.2, 6, 3)
m = bpy.data.materials.new("red")
m.diffuse_color = (0.6, 0.045, 0.035, 1)
m.use_nodes = True
next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"].default_value = (0.6, 0.045, 0.035, 1)
bpy.context.object.data.materials.append(m)
ld = bpy.data.lights.new("L", "POINT")
ld.energy = 1000
lo = bpy.data.objects.new("L", ld)
lo.location = (0, 0, 3)
sc.collection.objects.link(lo)

R["probe_types"] = [e.identifier for e in bpy.types.LightProbe.bl_rna.properties["type"].enum_items]
vol_type = "VOLUME" if "VOLUME" in R["probe_types"] else "GRID"
lp = bpy.data.lightprobes.new("vol", vol_type)
for k in ("grid_resolution_x", "grid_resolution_y", "grid_resolution_z"):
    setattr(lp, k, 4)
po = bpy.data.objects.new("vol", lp)
po.location = (0, 0, 1.5)
po.scale = (3, 3, 1.5)
sc.collection.objects.link(po)
bpy.context.view_layer.update()

R["eevee_gi_props"] = [p.identifier for p in sc.eevee.bl_rna.properties if "gi_" in p.identifier]
R["cache_info_before"] = sc.eevee.gi_cache_info
t0 = time.time()
try:
    R["bake_result"] = list(bpy.ops.scene.light_cache_bake())
except Exception as e:
    R["bake_error"] = str(e)[:400]
    traceback.print_exc()
R["bake_seconds"] = round(time.time() - t0, 2)
R["cache_info_after"] = sc.eevee.gi_cache_info

# where could Python read the coefficients?
R["rna_types_with_cache_in_name"] = sorted(n for n in dir(bpy.types) if "cache" in n.lower() and "light" in n.lower())
R["scene_eevee_attrs_cache"] = sorted(a for a in dir(sc.eevee) if "cache" in a.lower() or "irradiance" in a.lower())
R["lightprobe_attrs"] = sorted(a for a in dir(lp) if not a.startswith("_") and a not in dir(bpy.types.ID))
R["probe_object_attrs_lightprobe"] = sorted(a for a in dir(po) if "probe" in a.lower() or "cache" in a.lower())
R["images_after_bake"] = [(i.name, tuple(i.size)) for i in bpy.data.images]
# is the result at least stored in the .blend (C struct LightCache, not exposed to RNA)?
p = os.path.join(SCRATCH, "eevee_cache_test.blend")
try:
    bpy.ops.wm.save_as_mainfile(filepath=p, compress=False, check_existing=False)
    blob = open(p, "rb").read()
    R["blend_bytes"] = len(blob)
    R["blend_has_LightCache_struct_name"] = b"LightCache" in blob
    R["blend_has_LightGridCache"] = b"LightGridCache" in blob
    os.remove(p)
    if os.path.exists(p + "1"):
        os.remove(p + "1")
except Exception as e:
    R["save_error"] = str(e)[:300]
print("EEVEE " + json.dumps(R))
