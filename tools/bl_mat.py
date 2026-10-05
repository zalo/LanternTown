"""Blender-side material bake runner (numpy only; Blender is just the Python host).

    blender --background --factory-startup --python tools/bl_mat.py -- --mat <name>[,<name>...] [--res 512]

Runs materials/<name>.py build(t) and writes out/mat/<name>/{albedo,normal,orm}.{png,webp},
preview.png, sheet.png, mat.json. Prints `TSQ_RESULT {json}` per material.
"""
import sys
import os
import json
import time
import traceback
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tk import tex  # noqa: E402


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {"mat": None, "res": None, "out": os.path.join(ROOT, "out", "mat")}
    for i in range(0, len(argv), 2):
        a[argv[i].lstrip("-")] = argv[i + 1]
    with open(os.path.join(ROOT, "docs", "materials.json")) as f:
        cat = {m["name"]: m for m in json.load(f)["materials"]}
    for name in a["mat"].split(","):
        t0 = time.time()
        try:
            path = os.path.join(ROOT, "materials", name + ".py")
            spec = importlib.util.spec_from_file_location("tsq_mat_" + name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            meta = dict(cat.get(name, {}))
            meta.update(getattr(mod, "META", {}))
            res = int(a["res"]) if a["res"] else meta.get("res", 1024)
            t = tex.Canvas(res, meta.get("tile", 1.0), seed=meta.get("seed", 1))
            maps = mod.build(t)
            info = tex.save_material(os.path.join(a["out"], name), name, t, maps, meta)
            out = {"name": name, "ok": True, "res": info["res"], "tile": info["tile"], "height_range": info["height_range"],
                   "mean_albedo": info["preview"], "mean_rough": info["mean_rough"], "mean_metal": info["mean_metal"],
                   "seconds": round(time.time() - t0, 1)}
        except Exception:
            out = {"name": name, "ok": False, "error": traceback.format_exc()}
        print("TSQ_RESULT " + json.dumps(out))
        sys.stdout.flush()


main()
