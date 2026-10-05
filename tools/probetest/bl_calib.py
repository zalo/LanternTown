"""Calibration facts the probe pipeline relies on (Blender 4.1 / Cycles, headless).

    blender --background --factory-startup --python tools/probetest/bl_calib.py -- <scratch_dir> <out_json>

1. is OpenImageIO importable (multilayer EXR -> numpy)?
2. point light: Blender "watts" -> radiant intensity (what three.js calls candela on PointLight.intensity)
3. world colour -> camera radiance
4. equirectangular camera: which pixel looks where (three marker spheres on +X / +Y / +Z)
"""
import bpy, sys, os, json, math
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
SCRATCH, OUT = argv[0], argv[1]
res = {}

try:
    import OpenImageIO as oiio
    res["oiio"] = oiio.__version__
except Exception as e:
    res["oiio"] = None
    res["oiio_error"] = str(e)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.exr_codec = "NONE"
    sc.world = bpy.data.worlds.new("w")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    return sc


def render(sc, name):
    p = os.path.join(SCRATCH, name + ".exr")
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(p)
    w, h = img.size
    a = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)      # row 0 = bottom


def mat_diffuse(name, col):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs[0].default_value = (*col, 1)
    nt.links.new(d.outputs[0], o.inputs[0])
    return m


def mat_emit(name, col, s):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeEmission")
    d.inputs[0].default_value = (*col, 1)
    d.inputs[1].default_value = s
    nt.links.new(d.outputs[0], o.inputs[0])
    return m


# ---- 2. watts -> intensity
sc = reset()
bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
bpy.context.object.data.materials.append(mat_diffuse("white", (1, 1, 1)))
ld = bpy.data.lights.new("p", "POINT")
ld.energy = 1000.0
ld.color = (1, 1, 1)
ld.shadow_soft_size = 0.0
lo = bpy.data.objects.new("p", ld)
lo.location = (0, 0, 2.0)
sc.collection.objects.link(lo)
cd = bpy.data.cameras.new("c")
cd.type = "ORTHO"
cd.ortho_scale = 0.05
co = bpy.data.objects.new("c", cd)
co.location = (0, 0, 1.0)
sc.collection.objects.link(co)
sc.camera = co
sc.render.resolution_x = sc.render.resolution_y = 8
sc.render.resolution_percentage = 100
sc.cycles.samples = 64
a = render(sc, "calib_watt")
L = float(a[..., 0].mean())
E = math.pi * L
I = E * 2.0 ** 2
res["point_1000W_at_2m"] = {"radiance_white_lambert": L, "irradiance": E, "intensity_per_watt": I / 1000.0,
                            "1/(4pi^2)": 1 / (4 * math.pi ** 2), "1/(4pi)": 1 / (4 * math.pi)}
# same light with a radius: is it seen by camera rays, and does visible_camera hide it?
ld.shadow_soft_size = 0.1
cd.type = "PERSP"
co.location = (0, -3, 2.0)
co.rotation_euler = (math.pi / 2, 0, 0)
a = render(sc, "calib_lightvis")
res["light_seen_by_camera_default"] = float(a[..., :3].max())
lo.visible_camera = False
a = render(sc, "calib_lightvis2")
res["light_seen_by_camera_visible_camera_off"] = float(a[..., :3].max())

# ---- 3 + 4. world radiance and equirect layout
sc = reset()
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.1, 0.2, 0.3, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
for nm, pos, col in (("x", (5, 0, 0), (1, 0, 0)), ("y", (0, 5, 0), (0, 1, 0)), ("z", (0, 0, 5), (0, 0, 1))):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.6, location=pos)
    bpy.context.object.data.materials.append(mat_emit(nm, col, 50.0))
cd = bpy.data.cameras.new("c")
cd.type = "PANO"
cd.panorama_type = "EQUIRECTANGULAR"      # 4.1: on the camera datablock
co = bpy.data.objects.new("c", cd)
co.rotation_euler = (math.pi / 2, 0, -math.pi / 2)      # looks along +X, up +Z
sc.collection.objects.link(co)
sc.camera = co
sc.render.resolution_x, sc.render.resolution_y = 64, 32
sc.render.resolution_percentage = 100
sc.cycles.samples = 64
a = render(sc, "calib_equi")
h, w = a.shape[:2]
res["world_radiance_seen"] = [float(v) for v in np.median(a[..., :3].reshape(-1, 3), axis=0)]
out = {}
for ci, nm in enumerate("xyz"):
    j, i = np.unravel_index(np.argmax(a[..., ci] - a[..., (ci + 1) % 3]), (h, w))
    u, v = (i + 0.5) / w, (j + 0.5) / h          # v from the bottom
    lon, lat = (u - 0.5) * 2 * math.pi, (v - 0.5) * math.pi
    # the mapping the baker assumes for this camera orientation
    d = (math.cos(lat) * math.cos(lon), -math.cos(lat) * math.sin(lon), math.sin(lat))
    out[nm] = {"u": u, "v": v, "assumed_dir_blender": [round(x, 3) for x in d]}
res["equirect_markers"] = out

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("CALIB " + json.dumps(res))
