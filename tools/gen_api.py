#!/usr/bin/env python3
"""Generate docs/API.md (modelling toolkit) and docs/API_TEX.md (texture toolkit) from docstrings."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sig(fn):
    return f"{fn.name}({ast.unparse(fn.args)})"


def doc_module(path, public=None, skip=()):
    src = open(path).read()
    tree = ast.parse(src)
    out = []
    md = ast.get_docstring(tree)
    if md:
        out.append(md + "\n")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.name not in skip:
            if public is not None and node.name not in public:
                continue
            out.append(f"### `{sig(node)}`\n")
            d = ast.get_docstring(node)
            out.append((d or "") + "\n")
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_") and node.name not in skip:
            if public is not None and node.name not in public:
                continue
            out.append(f"### class `{node.name}`\n")
            d = ast.get_docstring(node)
            out.append((d or "") + "\n")
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and (not sub.name.startswith("_")):
                    a = ast.unparse(sub.args)
                    a = a[5:].lstrip(", ") if a.startswith("self") else a
                    out.append(f"- **`.{sub.name}({a})`** " + (ast.get_docstring(sub) or "").replace("\n", " ") + "\n")
    return "\n".join(out)


LEVELS_NOTE = """The contract of a level (budget, random stream, sockets / lights, parts, glow, bounds) is checked by
`tk/final_levels.py` (`finalize_level`). Its tests: `blender --background --factory-startup --python
tools/lod_tests/contract.py` - run them after any change to tk/levels.py, tk/final_levels.py, tk/final.py or
tools/bl_build.py. `tools/lod_tests/glbdiff.py a.glb b.glb` (Blender's python) compares two GLBs: order-independent
geometry hash and, per material, the triangles without a partner at 1 mm (proof that lod 0 did not change).
"""
REDUCE = {"reduce_objects", "reduce_scene", "force_scene", "level_candidates", "reduce_child", "visibility",
          "snapshot", "hold", "clear_meshes", "restore", "release"}
LODIMAGE = {"Metric", "gates", "choose", "Probe", "load_glb"}
FINAL_LEVELS = {"reference", "budget", "candidate", "pick", "finalize_level", "breakdown"}


def main():
    tk = os.path.join(ROOT, "tk")
    init = open(os.path.join(tk, "__init__.py")).read()
    tree = ast.parse(init)
    public = set()
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.level == 1:
            public |= {a.name for a in n.names}
    parts = ["# tk - modelling toolkit API (generated; `from tk import *`)\n",
             "Units: meters. Z up. Front of an asset faces -Y. Rotations are Euler degrees.\n"]
    for mod, title in (("core", "Parameters, RNG, objects"), ("geo", "Geometry builders"),
                       ("surf", "Materials, UVs, shading, weathering masks"), ("asm", "Sockets, lights, parts, composition")):
        parts.append(f"\n## {title}  (`tk/{mod}.py`)\n")
        parts.append(doc_module(os.path.join(tk, mod + ".py"), public))
    # levels of detail: the author tools (exported by tk/__init__.py), then the build side (not part of `from tk import *`)
    parts.append("\n## Levels of detail - modelled LOD1 / LOD2 (author guide: docs/LOD.md)  (`tk/levels.py`)\n")
    parts.append(doc_module(os.path.join(tk, "levels.py"), public).replace("\n\n### ", "\n" + LEVELS_NOTE + "\n### ", 1))
    parts.append("\n## Reduced levels - lod 1 / 2 made from the full model (build side, `from tk import reduce`)  (`tk/reduce.py`)\n")
    parts.append(doc_module(os.path.join(tk, "reduce.py"), REDUCE))
    parts.append("\n## Level image metric and the choice between candidates (build side, numpy only)  (`tk/lodimage.py`)\n")
    parts.append(doc_module(os.path.join(tk, "lodimage.py"), LODIMAGE))
    parts.append("\n## Level contract and what ships (build side)  (`tk/final_levels.py`)\n")
    parts.append(doc_module(os.path.join(tk, "final_levels.py"), FINAL_LEVELS))
    open(os.path.join(ROOT, "docs", "API.md"), "w").write("\n".join(parts))
    t = ["# tk.tex - texture synthesis API (generated; `from tk.tex import *`)\n", doc_module(os.path.join(tk, "tex.py"), skip=("write_png", "save_material", "render_preview"))]
    open(os.path.join(ROOT, "docs", "API_TEX.md"), "w").write("\n".join(t))
    print("wrote docs/API.md, docs/API_TEX.md")


main()
