"""Shared by the probe-test Blender scripts: builds the courtyard of out/probetest/scene.json for Cycles.

Every surface becomes a flat Diffuse BSDF (albedo from scene.json), lit windows an Emission shader, lantern glass and
mantles are deleted. Each material also writes the AOV "bf" = 1 where a ray met the BACK of a face (used to find
probes that sit inside walls). Lights are point lights in watts, invisible to camera rays (a probe / the camera only
sees what surfaces send back = what a three.js cube capture sees), and do not shadow on their own lamp post.
"""
import bpy, bmesh, os, re, json, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def t2b(p):
    """three (X, Y, Z) -> blender (x, y, z)"""
    return (p[0], -p[2], p[1])


def b2t(p):
    return (p[0], p[2], -p[1])


def load_scene_json():
    with open(os.path.join(ROOT, "out", "probetest", "scene.json")) as f:
        return json.load(f)


def setup_cycles(scratch, samples=64, denoise=False, seed=0, clamp_indirect=10.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.temporary_directory = os.path.join(scratch, "tmp")
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    c = sc.cycles
    c.device = "GPU"
    c.samples = samples
    c.use_adaptive_sampling = False
    c.use_denoising = denoise
    if denoise:
        c.denoiser = "OPTIX"
    c.seed = seed
    c.max_bounces = 8
    c.diffuse_bounces = 8
    c.glossy_bounces = 2
    c.transmission_bounces = 2
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = clamp_indirect
    c.use_light_tree = True
    sc.render.use_persistent_data = True      # keep the BVH / scene on the GPU between probe frames
    sc.render.use_motion_blur = False
    sc.render.resolution_percentage = 100
    sc.render.threads_mode = "AUTO"
    return sc


_mats = {}


def material(name, spec):
    if name in _mats:
        return _mats[name]
    m = bpy.data.materials.new("pt_" + name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if spec["kind"] == "emit":
        sh = nt.nodes.new("ShaderNodeEmission")
        e = spec["emission"]
        s = max(e) or 1.0
        sh.inputs["Color"].default_value = (e[0] / s, e[1] / s, e[2] / s, 1)
        sh.inputs["Strength"].default_value = s
    else:
        sh = nt.nodes.new("ShaderNodeBsdfDiffuse")
        a = spec["albedo"]
        sh.inputs["Color"].default_value = (a[0], a[1], a[2], 1)
    nt.links.new(sh.outputs[0], out.inputs["Surface"])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    aov = nt.nodes.new("ShaderNodeOutputAOV")
    if hasattr(aov, "aov_name"):
        aov.aov_name = "bf"
    else:
        aov.name = "bf"          # Blender 4.1: the node's name IS the AOV name
    nt.links.new(geo.outputs["Backfacing"], aov.inputs["Value"])
    _mats[name] = m
    return m


def add_box(name, mn, mx, mat):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        for i in range(3):
            v.co[i] = mn[i] + (v.co[i] + 0.5) * (mx[i] - mn[i])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def build(sc, S):
    """-> dict(lights=[objects], inst=[[mesh objects] per instance], tris=int)"""
    w = bpy.data.worlds.new("sky")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*S["sky"], 1)
    bg.inputs[1].default_value = 1.0
    sc.world = w
    aov = sc.view_layers[0].aovs.add()
    aov.name = "bf"
    aov.type = "VALUE"
    sc.view_layers[0].use_pass_z = True

    inst_objs = []
    tris = 0
    for it in S["instances"]:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(ROOT, it["glb"]))
        new = [o for o in bpy.data.objects if o not in before]
        root = bpy.data.objects.new("inst_" + it["id"], None)
        sc.collection.objects.link(root)
        root.location = it["pos"]
        root.rotation_euler = (0, 0, it["rz"])
        meshes = []
        for o in new:
            if o.parent is None:
                o.parent = root
            if o.type != "MESH":
                continue
            me = o.data
            names = [re.sub(r"\.\d{3}$", "", s.name) if s else None for s in me.materials]
            specs = [S["materials"].get(n) for n in names]
            hide = {i for i, sp in enumerate(specs) if sp is None or sp["kind"] == "hide"}
            if hide:
                bm = bmesh.new()
                bm.from_mesh(me)
                bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.material_index in hide], context="FACES")
                bm.to_mesh(me)
                bm.free()
            for i, sp in enumerate(specs):
                if i not in hide:
                    me.materials[i] = material(names[i], sp)
            me.calc_loop_triangles()
            tris += len(me.loop_triangles)
            meshes.append(o)
        inst_objs.append(meshes)
    for b in S["boxes"]:
        add_box(b["name"], b["min"], b["max"], material("box_" + b["name"], {"kind": "diffuse", "albedo": b["albedo"]}))
        tris += 12

    lights = []
    for i, L in enumerate(S["lights"]):
        ld = bpy.data.lights.new("lamp%d" % i, "POINT")
        ld.energy = L["watts"]
        ld.color = L["color"]
        ld.shadow_soft_size = 0.0
        lo = bpy.data.objects.new("lamp%d" % i, ld)
        lo.location = L["pos"]
        lo.visible_camera = False
        sc.collection.objects.link(lo)
        # the lamp does not shadow on its own post / lantern cage (the town engine does the same with a shadow near plane)
        if L.get("own") is not None:
            col = bpy.data.collections.new("own%d" % i)
            for o in inst_objs[L["own"]]:
                col.objects.link(o)
            lo.light_linking.blocker_collection = col
            for co in col.collection_objects:
                co.light_linking.link_state = "EXCLUDE"
        lights.append(lo)
    bpy.context.view_layer.update()
    return {"lights": lights, "inst": inst_objs, "tris": tris}


def look_at(ob, eye, target):
    import mathutils
    ob.location = eye
    d = mathutils.Vector(target) - mathutils.Vector(eye)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def exr_channels(path):
    """multilayer EXR -> (dict channel name -> 2D float32 array, rows top to bottom)"""
    import OpenImageIO as oiio
    inp = oiio.ImageInput.open(path)
    spec = inp.spec()
    px = inp.read_image("float")
    inp.close()
    px = np.asarray(px).reshape(spec.height, spec.width, spec.nchannels)
    return {n: px[..., i] for i, n in enumerate(spec.channelnames)}


def pick(ch, *keys):
    for n in ch:
        if all(k in n for k in keys):
            return ch[n]
    raise KeyError(str(keys) + " not in " + str(list(ch)))


# ---- L2 spherical harmonics, three.js convention (order and signs of LightProbeGrid / SphericalHarmonics3)
def sh_basis_three(d):
    """d: (..., 3) unit directions in THREE space -> (..., 9)"""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([
        0.282095 * np.ones_like(x),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z,
        0.546274 * (x * x - y * y)], axis=-1)


def sh_irradiance_three(sh, n):
    """three's getShIrradianceAt: sh (9, 3) radiance coefficients, n unit normal in THREE space -> irradiance rgb"""
    x, y, z = n
    return (sh[0] * 0.886227 + sh[1] * 2 * 0.511664 * y + sh[2] * 2 * 0.511664 * z + sh[3] * 2 * 0.511664 * x
            + sh[4] * 2 * 0.429043 * x * y + sh[5] * 2 * 0.429043 * y * z + sh[6] * (0.743125 * z * z - 0.247708)
            + sh[7] * 2 * 0.429043 * x * z + sh[8] * 0.429043 * (x * x - y * y))


def equirect_dirs(w, h):
    """pixel centres of an equirect render from a camera with rotation_euler (pi/2, 0, -pi/2) (looks along +X, up +Z;
    verified by bl_calib.py), rows TOP to bottom -> (dirs in Blender space (h, w, 3), dirs in three space, solid angle (h, w))
    This camera orientation is also exactly three's equirect layout (u = atan2(Z, X) / 2pi + 0.5, v up = +Y)."""
    u = (np.arange(w) + 0.5) / w
    v = 1.0 - (np.arange(h) + 0.5) / h
    lon = (u - 0.5) * 2 * np.pi
    lat = (v - 0.5) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    db = np.stack([np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    dt = np.stack([db[..., 0], db[..., 2], -db[..., 1]], axis=-1)
    dw = (2 * np.pi / w) * (np.pi / h) * np.cos(lat)
    return db, dt, dw
