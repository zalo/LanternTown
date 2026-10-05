"""Cycles ground truth for the probe test.

    blender --background --factory-startup --python tools/probetest/bl_ref.py -- scratch=<dir> mode=spots|views [spp=..]

mode=spots  for every spot of scene.json and the normals +Y, -Y (three; = +Z, -Z Blender) and "towards the wall":
            path-trace a WHITE diffuse sphere (albedo 1, radius 0.1 m, seen by camera rays only so it neither shadows
            nor bounces) and read the radiance at the point whose normal is n. Irradiance E = pi * radiance.
            Twice: with the lamps excluded from the sphere by light linking ("indirect": sky + every bounce + lit
            windows = what the probes hold) and with them ("total").
            Also: the same irradiance from the baked SH file(s) evaluated with three's formula at the spot
            (trilinear between probes, like the shader, but WITHOUT its half-cell normal offset), and the cosine
            integral of a 256 x 128 capture at the spot (= what a perfect, non-truncated probe at the spot would give).
            -> out/probetest/reference.json
mode=views  the two fixed cameras with the grey test figures at the stations, 1280x720, linear * exposure -> sRGB
            (the transfer the three page uses) -> out/renders/_probetest/cycles_cam<k>.png
"""
import bpy, sys, os, json, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as C

A = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--") + 1:])
SCRATCH = A["scratch"]
MODE = A.get("mode", "spots")
S = C.load_scene_json()
OUT = os.path.join(C.ROOT, "out", "probetest")


def render_exr(sc, name):
    p = os.path.join(SCRATCH, "exr", name + ".exr")
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    ch = C.exr_channels(p)
    os.remove(p)
    return ch


def white(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs[0].default_value = (1, 1, 1, 1)
    nt.links.new(d.outputs[0], o.inputs[0])
    return m


def load_probes(stem):
    meta = json.load(open(os.path.join(OUT, stem + ".json")))
    sh = np.fromfile(os.path.join(OUT, stem + ".bin"), dtype="<f4").reshape(-1, 9, 3)
    return meta, sh


def sh_at(meta, sh, p_three):
    """trilinear SH at a three-space position (clamped to the box)"""
    g = meta["grid"]
    res = g["res"]
    f = [min(max((p_three[a] - (g["center"][a] - g["size"][a] / 2)) / g["size"][a], 0.0), 1.0) * (res[a] - 1) for a in range(3)]
    i0 = [min(int(math.floor(f[a])), res[a] - 2) for a in range(3)]
    w = [f[a] - i0[a] for a in range(3)]
    out = np.zeros((9, 3))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                k = (i0[0] + dx) + (i0[1] + dy) * res[0] + (i0[2] + dz) * res[0] * res[1]
                out += sh[k] * ((w[0] if dx else 1 - w[0]) * (w[1] if dy else 1 - w[1]) * (w[2] if dz else 1 - w[2]))
    return out


if MODE == "spots":
    SPP = int(A.get("spp", 4096))
    sc = C.setup_cycles(SCRATCH, samples=SPP, denoise=False, seed=7, clamp_indirect=0.0)
    info = C.build(sc, S)
    sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.exr_codec = "NONE"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, segments=96, ring_count=48)
    sph = bpy.context.object
    bpy.ops.object.shade_smooth()
    sph.data.materials.append(white("ref_white"))
    sph.visible_diffuse = sph.visible_glossy = sph.visible_transmission = sph.visible_shadow = False
    # light linking: a receiver collection holding the sphere with state EXCLUDE = "light everything but the sphere"
    excl = bpy.data.collections.new("ref_excl")
    excl.objects.link(sph)
    excl.collection_objects[0].light_linking.link_state = "EXCLUDE"
    cd = bpy.data.cameras.new("ref")
    cd.type = "ORTHO"
    cd.ortho_scale = 0.03
    cd.clip_start = 0.85
    cd.clip_end = 5.0
    cam = bpy.data.objects.new("ref", cd)
    sc.collection.objects.link(cam)
    pd = bpy.data.cameras.new("pano")
    pd.type = "PANO"
    pd.panorama_type = "EQUIRECTANGULAR"
    pd.clip_start = 0.01
    pano = bpy.data.objects.new("pano", pd)
    pano.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
    sc.collection.objects.link(pano)
    probe_sets = {}
    for stem in A.get("probes", "probes_blender").split(","):
        if os.path.exists(os.path.join(OUT, stem + ".json")):
            probe_sets[stem] = load_probes(stem)
    PW, PH = 256, 128
    db, dt, dw = C.equirect_dirs(PW, PH)
    res = {"units": "irradiance, linear Rec.709, same units as three's irradiance (radiance of a white Lambert surface x pi)",
           "sphere_spp": SPP, "spots": []}
    t0 = time.time()
    for sp in S["spots"]:
        pb = sp["pos"]
        normals = {"+Y": (0, 0, 1), "-Y": (0, 0, -1), "wall": tuple(sp["wall"])}
        row = {"name": sp["name"], "pos_blender": pb, "pos_three": list(C.b2t(pb)), "wall_dir_three": list(C.b2t(sp["wall"])), "E": {}}
        # sphere readings
        sph.hide_render = False
        sph.location = pb
        for key, lights_on in (("cycles_sphere_indirect", False), ("cycles_sphere_total", True)):
            for lo in info["lights"]:
                lo.light_linking.receiver_collection = None if lights_on else excl
            row["E"][key] = {}
            sc.camera = cam
            sc.render.resolution_x = sc.render.resolution_y = 8
            sc.cycles.samples = SPP
            for nk, n in normals.items():
                C.look_at(cam, [pb[a] + n[a] * 1.0 for a in range(3)], pb)
                ch = render_exr(sc, "ref")
                L = [float(C.pick(ch, "Combined." + c).mean()) for c in "RGB"]
                row["E"][key][nk] = [round(math.pi * v, 5) for v in L]
        for lo in info["lights"]:
            lo.light_linking.receiver_collection = None
        # a perfect probe at the spot: cosine integral of a larger capture
        sph.hide_render = True
        sc.camera = pano
        pano.location = pb
        sc.render.resolution_x, sc.render.resolution_y = PW, PH
        sc.cycles.samples = 256
        ch = render_exr(sc, "ref")
        rgb = np.stack([C.pick(ch, "Combined." + c) for c in "RGB"], axis=-1)
        row["E"]["cycles_capture_cosine"] = {}
        row["E"]["cycles_capture_sh2"] = {}
        sh_here = ((C.sh_basis_three(dt) * dw[..., None]).reshape(-1, 9)).T @ rgb.reshape(-1, 3)
        for nk, n in normals.items():
            cosw = np.maximum(0.0, (db * np.array(n)).sum(-1)) * dw
            row["E"]["cycles_capture_cosine"][nk] = [round(float(v), 5) for v in (rgb * cosw[..., None]).sum((0, 1))]
            row["E"]["cycles_capture_sh2"][nk] = [round(float(v), 5) for v in C.sh_irradiance_three(sh_here, C.b2t(n))]
        for stem, (meta, sh) in probe_sets.items():
            c = sh_at(meta, sh, C.b2t(pb))
            row["E"][stem + "_trilinear"] = {nk: [round(float(v), 5) for v in C.sh_irradiance_three(c, C.b2t(n))] for nk, n in normals.items()}
        res["spots"].append(row)
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "reference.json"), "w") as f:
        json.dump(res, f, indent=1)
    for row in res["spots"]:
        print("REF", row["name"])
        for k, v in row["E"].items():
            print("REF    %-28s" % k, v)

elif MODE == "refl":
    # reflection probe: the same panoramic camera, bigger and denoised, written as a half-float EXR that three's
    # EXRLoader reads; the camera rotation makes it three's equirect layout as is (no flip, no re-orientation)
    sc = C.setup_cycles(SCRATCH, samples=int(A.get("spp", 256)), denoise=True, seed=5)
    info = C.build(sc, S)
    pd = bpy.data.cameras.new("pano")
    pd.type = "PANO"
    pd.panorama_type = "EQUIRECTANGULAR"
    pd.clip_start = 0.01
    pano = bpy.data.objects.new("pano", pd)
    pano.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
    pano.location = S["refl"]["pos"]
    sc.collection.objects.link(pano)
    sc.camera = pano
    sc.render.resolution_x, sc.render.resolution_y = int(A.get("w", 512)), int(A.get("w", 512)) // 2
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "16"
    sc.render.image_settings.exr_codec = "ZIP"
    sc.render.filepath = os.path.join(OUT, "refl_probe.exr")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print("REFL", sc.render.filepath, os.path.getsize(sc.render.filepath), "bytes", round(time.time() - t0, 2), "s")

else:
    SPP = int(A.get("spp", 256))
    EXPOSURE = float(A.get("exposure", 1.5))
    sc = C.setup_cycles(SCRATCH, samples=SPP, denoise=True, seed=3)
    info = C.build(sc, S)
    F = S["figure"]
    grey = C.material("figure", {"kind": "diffuse", "albedo": F["albedo"]})
    if A.get("figures", "1") == "1":
        for x, y in S["station_pos"]:
            r, hgt = F["capsule_radius"], F["capsule_height"]
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=hgt - 2 * r, location=(x, y, hgt / 2), vertices=48)
            bpy.context.object.data.materials.append(grey)
            bpy.ops.object.shade_smooth()
            for z in (r, hgt - r):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, z), segments=48, ring_count=24)
                bpy.context.object.data.materials.append(grey)
                bpy.ops.object.shade_smooth()
            bpy.ops.mesh.primitive_uv_sphere_add(radius=F["sphere_radius"], location=(x, y, F["sphere_z"]), segments=64, ring_count=32)
            bpy.context.object.data.materials.append(grey)
            bpy.ops.object.shade_smooth()
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = math.log2(EXPOSURE)
    sc.view_settings.gamma = 1.0
    cd = bpy.data.cameras.new("view")
    cd.sensor_fit = "VERTICAL"
    cd.clip_start = 0.05
    cam = bpy.data.objects.new("view", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    os.makedirs(os.path.join(C.ROOT, "out", "renders", "_probetest"), exist_ok=True)
    for k, v in S["cameras"].items():
        if k not in A.get("cams", "1,2").split(","):
            continue
        cd.angle_y = math.radians(v["fov"])
        C.look_at(cam, v["eye"], v["target"])
        t0 = time.time()
        sc.render.filepath = os.path.join(C.ROOT, "out", "renders", "_probetest", "cycles_cam%s.png" % k)
        bpy.ops.render.render(write_still=True)
        print("VIEW cam", k, round(time.time() - t0, 1), "s")
