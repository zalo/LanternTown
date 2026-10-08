#!/usr/bin/env python3
"""The project's source as one zip (stdlib only) - what the gallery's "Source code" button downloads.

    python3 tools/srczip.py [--refs] [-o FILE]     write the zip to FILE (default: print what would go in)
    GET /api/source.zip[?refs=1]                   tools/serve.py: built on request from the files as they are now
    python3 tools/srczip.py --skill [-o FILE]      the Claude Code skill `parametric-environment` as one zip: skill/
    GET /api/skill.zip                             (SKILL.md, references, templates) with this source as its project/

In: the asset generators, materials, toolkit, tools (build, bake, pack, checks), town layouts, web pages and engines,
docs, judge reviews, prompt.md and the package files. Out: everything generated (out/), node_modules, the three.js
checkout (upstream/), scratch and logs - `SOURCE.md` at the top of the zip says how to get those back.
`refs`: also the reference images (assets/, docs/ref/).
"""
import io
import os
import sys
import time
import zipfile
import hashlib
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "LanternTown"
DIRS = ["assets", "materials", "tk", "tools", "town", "town2", "web", "docs", "skill"]
SKILL = "parametric-environment"       # the skill's folder name in its zip (and under ~/.claude/skills)
SKILL_DOCS = ("AUTHORING.md", "QUALITY.md", "TOWN_QUALITY.md", "MATERIALS.md", "LOD.md", "API.md", "API_TEX.md")
TOP = ["README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "prompt.md", "package.json", "package-lock.json", ".gitignore", ".github/workflows/pages.yml"]
# source kept beside generated data: (folder, its sub-folders too?)
EXTRA = [("out/web/_nextglb/msft_lod_kit", False)]
TEXT = {".py", ".js", ".mjs", ".html", ".css", ".md", ".json", ".sh", ".txt", ".glsl", ".wgsl", ".toml", ".cfg", ".patch", ".yml", ""}
IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
REF_DIRS = ("assets", "docs/ref")     # where `refs` takes images from
MAX_FILE = 2 << 20                    # a "source" file larger than this is data that strayed in: left out, and listed

_lock = threading.Lock()
_cache = {}                           # refs -> (signature, bytes, file name)


def files(refs=False):
    """-> ([(absolute path, name in the zip, stat)], [names left out for their size])"""
    got, big = [], []

    def take(path, rel):
        ext = os.path.splitext(rel)[1].lower()
        image = refs and ext in IMAGES and rel.startswith(tuple(d + "/" for d in REF_DIRS))
        if ext not in TEXT and not image:
            return
        if os.path.islink(path) or not os.path.isfile(path):
            return
        st = os.stat(path)
        if st.st_size > MAX_FILE and not image:
            big.append(rel)
            return
        got.append((path, rel, st))

    for name in TOP:
        take(os.path.join(ROOT, name), name)
    for top, deep in [(d, True) for d in DIRS] + EXTRA:
        for d, dirs, names in os.walk(os.path.join(ROOT, *top.split("/"))):
            dirs[:] = sorted(x for x in dirs if deep and not x.startswith((".", "__pycache__")) and x != "node_modules")
            for n in sorted(names):
                if n.startswith("."):
                    continue
                path = os.path.join(d, n)
                take(path, os.path.relpath(path, ROOT).replace(os.sep, "/"))
    return got, big


def readme(got, big, refs, stamp):
    count = {}
    for _, rel, st in got:
        top = rel.split("/")[0] if "/" in rel else "(top)"
        c = count.setdefault(top, [0, 0])
        c[0] += 1
        c[1] += st.st_size
    rows = "\n".join(f"| `{k}` | {v[0]} | {v[1] / 1e6:.2f} MB |" for k, v in sorted(count.items()))
    left = "".join(f"\n- `{r}` (over {MAX_FILE >> 20} MB)" for r in big)
    return f"""# Lantern Square - source snapshot

Taken {stamp} from the working tree (newest file in it). Start with `prompt.md`: the brief, the requirements that
were added along the way, the architecture and the traps. Then `docs/AUTHORING.md`, `docs/API.md`, `docs/QUALITY.md`,
`docs/TOWN_QUALITY.md`, `docs/LOD.md`, `docs/MATERIALS.md`.

| folder | files | size |
|---|---|---|
{rows}

## Where things are

- `assets/<category>/<id>.py` - the parametric asset generators (toolkit: `tk/`); `materials/` - procedural materials
- `tools/tsq` - the command authors use (`build`, `render`, `lod`, `mat`, `api`, ...); `tools/bl_build.py` runs in Blender
- `town/layout.py`, `town2/` - the town generators; `tools/town.py gen | build | render`
- `tools/webbake.py`, `webbake_bl.py`, `webbake_pack.mjs`, `lsvb_*` - the web bake (lighting in Blender, packing in Node)
- `tools/aomap.py` - the ground's contact-occlusion map and, beside it, the lanterns' shadow map of the ground; `tools/pvs.py` - baked visibility sets and the hole census
- `tools/glbopt.mjs`, `bake2glb.mjs`, `ktx_layers.py` - model packing, baked format to GLB, KTX2 sets
- `web/town/`, `web/engine/` - the walk page and its two runtimes (`lite.js` baked, `town.js` streaming)
- `web/gallery/` - the gallery; `tools/serve.py` - the server; `tools/manifest.py` - what the gallery shows
- `out/web/_nextglb/msft_lod_kit/` - the MSFT_lod test kit's sources (its models are generated by `generate.mjs`)

## Not in here

- `out/` - everything generated: models, textures, renders, the baked towns. Rebuild with the tools above.
- `node_modules/` - `npm install` (three, meshoptimizer).
- `upstream/three.js` - a checkout of mrdoob/three.js (dev) used for the GLTFLoader work.
- {"The reference images are in `assets/` and `docs/ref/`." if refs else "The reference images (`assets/`, `docs/ref/`): this is the code-only download; the gallery offers one with them."}
- Scratch, logs and release bundles.{left}

## Needs

Blender 4.1 on the command line, Python 3 (numpy for the materials), Node 22, ImageMagick, a Chrome with WebGPU for
the review renders. `python3 tools/serve.py 8765`, then open `/gallery/`.
"""


def build(refs=False):
    """-> (zip bytes, file name). Built again only when a file in it changed."""
    refs = bool(refs)
    with _lock:
        got, big = files(refs)
        sig = hashlib.sha1("\n".join(f"{rel} {st.st_mtime_ns} {st.st_size}" for _, rel, st in got).encode()).hexdigest()
        hit = _cache.get(refs)
        if hit and hit[0] == sig:
            return hit[1], hit[2]
        newest = max(st.st_mtime for _, _, st in got)
        t = time.localtime(newest)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            info = zipfile.ZipInfo(f"{NAME}/SOURCE.md", t[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, readme(got, big, refs, time.strftime("%Y-%m-%d %H:%M", t)))
            for path, rel, st in got:
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue            # gone since the listing
                info = zipfile.ZipInfo(f"{NAME}/{rel}", time.localtime(st.st_mtime)[:6])
                info.compress_type = zipfile.ZIP_STORED if os.path.splitext(rel)[1].lower() in IMAGES else zipfile.ZIP_DEFLATED
                info.external_attr = (st.st_mode & 0o777) << 16     # tools/tsq and the .sh files stay executable
                z.writestr(info, data)
        name = f"{NAME}-source{'-refs' if refs else ''}-{time.strftime('%Y%m%d-%H%M', t)}.zip"
        _cache[refs] = (sig, buf.getvalue(), name)
        return _cache[refs][1], name


def build_skill():
    """-> (zip bytes, file name): the skill as skill/install.py assembles it - SKILL.md, references/ (with the docs
    the references point at), templates/, and this source (less skill/ itself) as project/. Unzip into
    ~/.claude/skills/ and Claude Code lists it as /parametric-environment."""
    with _lock:
        got, big = files(False)
        sig = hashlib.sha1(("skill\n" + "\n".join(f"{rel} {st.st_mtime_ns} {st.st_size}" for _, rel, st in got)).encode()).hexdigest()
        hit = _cache.get("skill")
        if hit and hit[0] == sig:
            return hit[1], hit[2]
        newest = max(st.st_mtime for _, _, st in got)
        t = time.localtime(newest)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path, rel, st in got:
                if rel.startswith("skill/"):
                    name = SKILL + "/" + rel[len("skill/"):]         # SKILL.md, references/, templates/, install.py
                else:
                    name = f"{SKILL}/project/{rel}"
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                info = zipfile.ZipInfo(name, time.localtime(st.st_mtime)[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (st.st_mode & 0o777) << 16
                z.writestr(info, data)
                if rel.startswith("docs/") and rel[len("docs/"):] in SKILL_DOCS:
                    info = zipfile.ZipInfo(f"{SKILL}/references/docs/{rel[len('docs/'):]}", time.localtime(st.st_mtime)[:6])
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    z.writestr(info, data)
        name = f"{SKILL}-skill-{time.strftime('%Y%m%d-%H%M', t)}.zip"
        _cache["skill"] = (sig, buf.getvalue(), name)
        return _cache["skill"][1], name


def main():
    args = sys.argv[1:]
    refs = "--refs" in args
    if "-o" in args:
        body, name = build_skill() if "--skill" in args else build(refs)
        dst = args[args.index("-o") + 1]
        with open(dst, "wb") as f:
            f.write(body)
        print(f"{dst}: {len(body) / 1e6:.2f} MB ({name})")
        return
    got, big = files(refs)
    print(f"{len(got)} files, {sum(st.st_size for _, _, st in got) / 1e6:.2f} MB before compression")
    for rel in big:
        print(f"left out (size): {rel}")


if __name__ == "__main__":
    main()
