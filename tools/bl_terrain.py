"""Blender-side terrain build for the town (ground patches, skirts, canal water).

    blender --background --factory-startup --python tools/bl_terrain.py -- --layout out/town/layout.json

Writes out/glb/_town/terrain.glb (+ .json sidecar, LODs are not needed: it is always shown) and
out/glb/_town/water.glb. Every patch comes from the layout's `terrain` list - nothing is placed by hand.
"""
import sys
import os
import json
import math
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from tk import *  # noqa: E402,F401,F403
from tk import core, final  # noqa: E402


def rect_patch(t, seed):
    x0, y0, x1, y1 = t["rect"]
    z = t["z"]
    w, d = x1 - x0, y1 - y0
    step = float(t.get("step", 0.6 if max(w, d) < 60 else 3.0))
    nu, nv = max(2, int(math.ceil(w / step))), max(2, int(math.ceil(d / step)))
    o = surface("g_" + t["name"], lambda u, v: (x0 + u * w, y0 + v * d, z), nu, nv, uv_len=(w, d))
    uv_keep(o)
    # texture origin in world metres so neighbouring patches of one material line up
    uv_transform(o, offset=(x0, y0))
    if t.get("water"):
        # what floats in it parts it: each floater's outline (tools/town.py floaters: a hull at its waterline) is cut
        # out, so the water ends in the planking and the boat is not full of the canal
        for i, c in enumerate(q for q in CUTS if q["water"] == t["name"]):
            ol = c["outline"]
            n = len(ol)
            a = sum(ol[k][0] * ol[(k + 1) % n][1] - ol[(k + 1) % n][0] * ol[k][1] for k in range(n))
            if a < 0:
                ol = ol[::-1]
            vs = [(x, y, z - 0.4) for x, y in ol] + [(x, y, z + 0.4) for x, y in ol]
            fs = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
            boolean(o, mesh_object("cut_%d" % i, vs, fs))
        assign(o, mat("water", t.get("tint", "#20302a"), flow=0.35, opacity=0.9, probe=1))
        return o
    assign(o, mat(t["mat"]))
    amp = float(t.get("bump", 0.008))
    if amp > 0 and step < 2:
        displace_noise(o, strength=amp, scale=2.6, seed=seed, axis_mask=(0, 0, 1))
    wet = float(t.get("wet", 0.0))
    paint(o,
          grime=mask_mul(noise_mask(3.2, seed + 3, 0.35, 0.8), 0.55),
          special=mask_mul(noise_mask(4.5, seed + 11, 0.42, 0.72), wet) if wet > 0 else None,
          wear=mask_mul(noise_mask(5.0, seed + 17, 0.45, 0.8), 0.5))
    return o


def skirt_patch(t, seed):
    (ax, ay), (bx, by) = t["a"], t["b"]
    z0, z1 = t["z0"], t["z1"]
    L = math.hypot(bx - ax, by - ay)
    nu, nv = max(1, int(math.ceil(L / 1.0))), max(1, int(math.ceil((z1 - z0) / 0.6)))
    o = surface("s_" + t["name"], lambda u, v: (ax + (bx - ax) * u, ay + (by - ay) * u, z0 + (z1 - z0) * v), nu, nv, uv_len=(L, z1 - z0))
    uv_keep(o)
    want = np.array([(by - ay) / L, -(bx - ax) / L, 0.0])       # visible from the right of a->b
    n = np.array(o.data.polygons[0].normal)
    if float(n @ want) < 0:
        o.data.flip_normals()
    assign(o, mat(t.get("mat", "ashlar_limestone")))
    paint(o, grime=mask_mul(height_ramp(z0, z0 + 1.2), 0.7), special=mask_mul(noise_mask(1.5, seed, 0.4, 0.7), 0.5))
    return o


CUTS = []              # floaters' outlines (--cuts file): [{"water": patch name, "outline": [[x, y], ...]}]
TOWN_DIR = "_town"      # out/glb/<TOWN_DIR>/: "--town town2" (tools/town.py with TSQ_TOWN=town2) writes a second town's ground to _town2


def export(name, items, seed):
    core.clear_scene()
    core._reset_state(seed)
    n = 0
    for i, t in enumerate(items):
        (skirt_patch if t["kind"] == "skirt" else rect_patch)(t, seed + i * 7)
        n += 1
    if not n:
        return None
    meta = {"class": "ground", "mount": "none", "auto_wear": 0.0, "auto_grime": 0.6, "tri_budget": 400000, "max_materials": 16,
            "name": "town " + name}
    glb = os.path.join(ROOT, "out", "glb", TOWN_DIR, name + ".glb")
    return final.finalize(TOWN_DIR, name, {"seed": seed}, meta, glb, 0.0, lods=False)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    layout = json.load(open(argv[argv.index("--layout") + 1]))
    if "--town" in argv:
        global TOWN_DIR
        TOWN_DIR = "_" + argv[argv.index("--town") + 1]
    if "--cuts" in argv:
        CUTS.extend(json.load(open(argv[argv.index("--cuts") + 1])))
    seed = int(layout.get("seed", 1))
    t0 = time.time()
    land = [t for t in layout["terrain"] if not t.get("water")]
    water = [t for t in layout["terrain"] if t.get("water")]
    out = {}
    for name, items in (("terrain", land), ("water", water)):
        info = export(name, items, seed)
        if info:
            out[name] = {"tris": info["stats"]["tris"], "warnings": info["warnings"], "errors": info["errors"]}
    print("TSQ_RESULT " + json.dumps({"ok": True, "parts": out, "seconds": round(time.time() - t0, 1)}))


main()
