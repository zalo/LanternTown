#!/usr/bin/env python3
"""Writes out/probetest/scene.json: the one description of the test courtyard that BOTH the Blender baker
(tools/probetest/bl_bake.py) and the three.js page (web/probetest/) read.

    python3 tools/probetest/make_scene.py

Coordinates: instances, boxes, lights, cameras and the path are in the project's asset space (Blender, Z up, metres;
house fronts face -Y). The probe grid is given the way three's LightProbeGrid wants it (three space, Y up).
    three (X, Y, Z) = blender (x, z, -y)        blender (x, y, z) = three (X, -Z, Y)
A rotation `rz` about Blender's Z is the same angle about three's Y.
All colours are linear Rec.709 (no sRGB anywhere); lights are in Blender watts (three PointLight.power = watts,
i.e. PointLight.intensity = watts / 4 pi - measured by bl_calib.py).
"""
import json, os, struct

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lin(hexcol):
    out = []
    for i in (1, 3, 5):
        c = int(hexcol[i:i + 2], 16) / 255.0
        out.append(round(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, 4))
    return out


def glb_json(path):
    with open(path, "rb") as f:
        f.read(12)
        n, _ = struct.unpack("<II", f.read(8))
        return json.loads(f.read(n))


LIB = {m["name"]: m for m in json.load(open(os.path.join(ROOT, "docs", "materials.json")))["materials"]}

LAMP = lin("#ffbf6a")
LAMP_W = 1000.0          # Blender watts = three PointLight.power; intensity 79.6 cd
SKY = [0.010, 0.016, 0.032]      # radiance of the night sky (world colour x strength 1 = scene.background)
WINDOW_GLOW = 1.5        # radiance of a lit window = its tint x this
RED = [0.60, 0.045, 0.035]       # the painted wall

# asset instances: id, variant file stem, position (Blender), rz
INST = [
    ("house_timber", "_8e0a65828f.lod1", [-5.2, 5.0, 0.0], 0.0),
    ("townhouse_narrow", "_561ac32088.lod1", [0.65, 5.0, 0.0], 0.0),
    ("house_stone", "_7b9b165c74.lod1", [6.9, 5.0, 0.0], 0.0),
    ("wall_stone_run", "_087a5f5245", [-8.6, -1.5, 0.0], 1.5707963),
    ("lamp_post_gas", "_02f6488685", [3.6, -1.5, 0.0], 0.0),
    ("lamp_post_gas", "_02f6488685", [-4.0, 1.2, 0.0], 0.0),
    ("awning", "default", [-5.2, 4.15, 3.0], 0.0),
    ("paving_patch", "herringbone", [0.0, -1.5, 0.0], 0.0),
    ("barrel", "_019d20689b", [5.3, 1.6, 0.0], 0.4),
    ("bench_park", "_075ba34598", [-1.0, 3.2, 0.0], 0.0),
]
# plain boxes (min, max in Blender space) for what the asset set has no flat piece for
BOXES = [
    {"name": "floor", "min": [-12.0, -10.0, -0.3], "max": [13.0, 14.0, 0.0], "albedo": LIB["cobble_granite"]["preview"]},
    {"name": "red_wall", "min": [6.0, -5.5, 0.0], "max": [6.4, 2.5, 3.2], "albedo": RED},
]

scene = {"note": __doc__.split("\n\n")[2], "sky": SKY, "instances": [], "boxes": BOXES, "lights": [], "materials": {}}
tris = 0
for aid, var, pos, rz in INST:
    rel = f"out/glb/{aid}/{var}.glb"
    side = json.load(open(os.path.join(ROOT, "out", "glb", aid, var.split(".")[0] + ".json")))
    lod = var.split(".")[1] if "." in var else None
    tris += side["stats"]["lods"][int(lod[3:]) - 1]["tris"] if lod else side["stats"]["tris"]
    scene["instances"].append({"id": aid, "glb": rel, "pos": pos, "rz": rz})
    for m in glb_json(os.path.join(ROOT, rel))["materials"]:
        ex = m.get("extras", {})
        lib = ex.get("tsq_lib", m["name"].split("~")[0])
        tint = ex.get("tsq_tint")
        e = {"lib": lib}
        if lib.startswith("glass") or lib == "emissive":
            e["kind"] = "hide"       # lantern glass and mantles: not part of the lighting test (the lamp is the point light)
        elif lib == "window_lit":
            e["kind"] = "emit"
            e["emission"] = [round(c * WINDOW_GLOW, 4) for c in (tint or [1, 0.6, 0.25])]
        else:
            e["kind"] = "diffuse"
            # tinted library materials carry the designer's colour, the rest use the library preview colour
            col = tint if (tint and LIB.get(lib, {}).get("tintable")) else LIB.get(lib, {}).get("preview", [0.5, 0.5, 0.5])
            if lib == "window_dark":
                col = [0.02, 0.02, 0.025]
            e["albedo"] = [round(min(0.85, c), 4) for c in col]
        scene["materials"][m["name"]] = e
    if aid == "lamp_post_gas":
        s = next(s for s in side["sockets"] if s["kind"] == "light")
        scene["lights"].append({"pos": [round(pos[0] + s["pos"][0], 4), round(pos[1] + s["pos"][1], 4), round(pos[2] + s["pos"][2], 4)],
                                "color": LAMP, "watts": LAMP_W, "own": len(scene["instances"]) - 1})
scene["tris_assets"] = tris

# probe grid, three space: X 17.5 m (8 probes, 2.5 m), Y 4.8 m (4 probes, 1.6 m), Z 12.6 m (8 probes, 1.8 m)
scene["grid"] = {"center": [0.75, 2.7, -0.3], "size": [17.5, 4.8, 12.6], "res": [8, 4, 8]}

# path of the dynamic objects (Blender space, on the ground): dark open corner -> under lamp A -> along the red wall
# -> round its end -> behind it
scene["path"] = [[-7.0, -5.5], [-2.0, -4.0], [2.5, -2.5], [5.0, -2.0], [5.3, 0.5], [5.2, 3.2], [7.0, 3.6], [7.6, 1.0], [7.6, -3.5]]
# where the test objects stand for the stills (parameter 0..1 along the path)
scene["stations"] = [0.04, 0.30, 0.47, 0.60, 0.80, 0.95]


def along(path, t):
    seg = [((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2) ** 0.5 for i in range(len(path) - 1)]
    d = t * sum(seg)
    for i, s in enumerate(seg):
        if d <= s or i == len(seg) - 1:
            k = min(1.0, d / s)
            return [round(path[i][0] + k * (path[i + 1][0] - path[i][0]), 4), round(path[i][1] + k * (path[i + 1][1] - path[i][1]), 4)]
        d -= s


scene["station_pos"] = [along(scene["path"], t) for t in scene["stations"]]
# the test objects: a capsule the size of a person standing on the ground, a sphere floating above its head
scene["figure"] = {"capsule_radius": 0.3, "capsule_height": 1.8, "sphere_radius": 0.35, "sphere_z": 2.4, "albedo": [0.8, 0.8, 0.8]}

# fixed cameras (Blender space): eye, target, vertical fov (deg)
scene["cameras"] = {
    "1": {"eye": [-6.5, -13.5, 6.5], "target": [1.5, -0.5, 1.2], "fov": 55},
    "2": {"eye": [9.6, -7.2, 2.3], "target": [5.6, 1.2, 1.1], "fov": 55},
    "3": {"eye": [1.4, -3.4, 1.9], "target": [4.1, 0.6, 1.5], "fov": 40},      # close on the chrome balls of ?refl=1
}
# spots for the numeric comparison (Blender space): in the open, by the red wall (lit side), behind it, by the
# awning, in the dark corner; "wall" is the horizontal direction used for the third reading
scene["spots"] = [
    {"name": "open_centre", "pos": [0.0, -2.0, 1.2], "wall": [1, 0, 0]},
    {"name": "red_wall_lit_side", "pos": [5.2, -1.0, 1.2], "wall": [1, 0, 0]},
    {"name": "behind_red_wall", "pos": [7.2, -1.0, 1.2], "wall": [-1, 0, 0]},
    {"name": "under_awning", "pos": [-5.2, 3.2, 1.2], "wall": [0, 1, 0]},
    {"name": "dark_corner", "pos": [-7.0, -5.5, 1.2], "wall": [1, 0, 0]},
    {"name": "house_front_high", "pos": [1.0, 3.0, 3.5], "wall": [0, 1, 0]},
]
# ... and spots that sit exactly ON a probe (no interpolation between probes: what is left is the bake itself and the
# L2 truncation). probe = (ix, iy, iz) of the grid
G = scene["grid"]
for name, (ix, iy, iz), wall in [
    ("probe_open", (3, 1, 4), [1, 0, 0]),
    ("probe_red_wall_lit_side", (5, 1, 4), [1, 0, 0]),
    ("probe_behind_red_wall", (6, 1, 4), [-1, 0, 0]),
    ("probe_by_awning", (1, 1, 2), [0, 1, 0]),
    ("probe_dark_corner_low", (0, 0, 7), [1, 0, 0]),
]:
    pt = [G["center"][a] - G["size"][a] / 2 + (ix, iy, iz)[a] * G["size"][a] / (G["res"][a] - 1) for a in range(3)]
    scene["spots"].append({"name": name, "pos": [round(pt[0], 4), round(-pt[2], 4), round(pt[1], 4)], "wall": wall,
                           "probe": [ix, iy, iz], "probe_index": ix + iy * G["res"][0] + iz * G["res"][0] * G["res"][1]})

# where the one reflection probe of the side test is captured (Blender space) - and where the page hangs a chrome ball
scene["refl"] = {"pos": [4.6, 0.6, 1.5], "radius": 0.45}

out = os.path.join(ROOT, "out", "probetest", "scene.json")
with open(out, "w") as f:
    json.dump(scene, f, indent=1)
kinds = {}
for e in scene["materials"].values():
    kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
print("wrote", out, "| instances", len(scene["instances"]), "| asset tris", tris, "| materials", len(scene["materials"]), kinds,
      "| lights", scene["lights"])
