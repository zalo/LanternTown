#!/usr/bin/env python3
"""The project as a static site (stdlib only): the folder a host with no tools/serve.py behind it can serve -
GitHub Pages, Cloudflare Pages, any file server. Same URL layout as the live server, so the pages run unchanged.

    python3 tools/site.py [--out out/site] [--town2] [--no-glb] [--copy] [--dry]
    python3 tools/site.py --data-out out/site-data [--town2] [--no-glb]         the generated part alone, in parts
    python3 tools/site.py --upload data [--town2] [--no-glb]                    ..and the parts that changed to a release
    python3 tools/site.py --out _site --data site-data/                         a checkout's pages + those parts

In:   the pages (web/, without the development copies and the pages that need the server), three.js out of
      node_modules/, docs/ (without the reference images), the baked town (out/web/_next, phone set included),
      out/town/town.json, the gallery's pictures, the one-file .glb towns and LOD demo files (--no-glb: not those),
      the MSFT_lod test kit; --town2: the second town too.
Made: api/manifest - the gallery's data, frozen (tools/manifest.py), with a `static` part that tells the gallery
      which of its links have nothing behind them here; api/source.zip (tools/srczip.py); .nojekyll (GitHub Pages
      would otherwise skip every folder that starts with "_").
Out:  the full-detail models (out/glb: the Studio preset and the 3D asset viewer), the first baked runtime, the
      benchmark's files, the reference images, telemetry (web/engine/telemetry.js posts to api/telemetry, which a
      static host refuses: the reports are dropped).

Files are hard links to the originals where the disk allows it (--copy: real copies), so the folder costs no space.
The summary at the end holds the result against the hosts' limits. --dry: the summary only.

The repository holds the source only; everything generated stays in out/. --data-out writes the site's generated
part (what the site has below out/, and api/manifest) as parts to keep outside the repository, e.g. as the assets of
one GitHub release: uncompressed tars - their content is compressed already - named by the hash of what is in them
(`<group>-<hash>.tar`), and site-data.json, the list of them. A file over PART_MB is a part of its own; the others
are gathered by group (the bake, the one-file .glb towns, the pictures, ..) into PART_N tars each, by a hash of the
path, so that a change touches the parts it is in and no other. --upload TAG writes the parts and brings the release
up to date through `gh`: parts it has stay, parts it lacks go up, parts the list no longer names are deleted, the
list replaced - a new bake uploads the bake, a new picture its part, nothing else. --data builds the site in a clean
checkout from the parts (a folder, or the list): pages, three.js and documents come from the checkout, nothing is
read from out/ but the MSFT_lod kit's sources. .github/workflows/pages.yml does that for GitHub Pages.
"""
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

WEB_SKIP_DIRS = {"next", "perf", "ktx", "probetest"}                 # development copies and experiments
WEB_SKIP_FILES = {"render.html", "render.js", "viewer.html", "viewer.js",        # need the render queue / out/glb
                  "bakeglb/index.html", "bakeglb/bench.js"}                      # the benchmark: out/web/_nextglb/bench
GLBS = ["town_best.glb", "town_near.glb", "town_mid.glb", "town_lo.glb", "town_lod.glb",
        "portable_lod_spawn.glb", "exact_lod_spawn.glb"]
# gallery links ( and whole cards, by their `probe` ) with nothing behind them on a static site
DROP = [r"^town/\?quality=studio", r"refs=1", r"^ktx/", r"^probetest/", r"legacy=1", r"^viewer\.html", r"^bakeglb/$"]
LIMITS = [("GitHub Pages", 1000e6, 100e6, None), ("Cloudflare Pages (free)", None, 25 * 1024 * 1024, 20000)]
PART_MB, PART_N = 24, 8   # a file over this is a part of its own; the rest of a group goes into this many parts


def walk(top, rel="", skip_dirs=(), skip=()):
    """-> [(absolute path, path below `top`)] of the plain files below `top`, without dot entries"""
    got = []
    for d, dirs, names in os.walk(os.path.join(top, rel) if rel else top):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x != "__pycache__" and x != "node_modules"
                         and os.path.relpath(os.path.join(d, x), top).replace(os.sep, "/") not in skip_dirs)
        for n in sorted(names):
            p = os.path.join(d, n)
            r = os.path.relpath(p, top).replace(os.sep, "/")
            if n.startswith(".") or os.path.islink(p) or not os.path.isfile(p) or r in skip:
                continue
            got.append((p, r))
    return got


def plan(town2=False, glb=True, data=True):
    """-> ({path in the site: absolute source}, the frozen manifest, pictures it names that are not on disk)
    data=False: the checkout's part only ( no manifest: None )"""
    site = {}

    def add(src, dst):
        if os.path.isfile(src):
            site[dst] = src
            return True
        return False

    for p, r in walk(os.path.join(ROOT, "web"), skip_dirs=WEB_SKIP_DIRS, skip=WEB_SKIP_FILES):
        if not r.endswith(".py"):
            site[r] = p
    for sub in ("build", "examples/jsm"):
        for p, r in walk(os.path.join(ROOT, "node_modules", "three", sub)):
            site[f"node_modules/three/{sub}/{r}"] = p
    for n in ("LICENSE", "package.json"):
        add(os.path.join(ROOT, "node_modules", "three", n), f"node_modules/three/{n}")
    for n in ("build/index.module.js", "LICENSE", "package.json"):     # glbtown/?walk=bvh imports it (the one file, on demand)
        add(os.path.join(ROOT, "node_modules", "three-mesh-bvh", n), f"node_modules/three-mesh-bvh/{n}")
    for p, r in walk(os.path.join(ROOT, "docs"), skip_dirs={"ref"}):
        if os.path.splitext(r)[1] in (".md", ".json", ".patch"):
            site[f"docs/{r}"] = p
    for n in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "prompt.md"):
        add(os.path.join(ROOT, n), n)

    for p, r in walk(os.path.join(ROOT, "out", "web", "_nextglb", "msft_lod_kit")):
        site[f"out/web/_nextglb/msft_lod_kit/{r}"] = p          # its sources are in the repository, its models are not
    if not data:
        return site, None, []

    import manifest
    towns = ["town"] + (["town2"] if town2 else [])
    for t in towns:
        add(os.path.join(ROOT, "out", t, "town.json"), f"out/{t}/town.json")
    bakes = ["_next"] + (["_bake_town2"] if town2 else [])
    for b in bakes:
        for p, r in walk(os.path.join(ROOT, "out", "web", b)):
            site[f"out/web/{b}/{r}"] = p
    if glb:
        for n in GLBS:
            add(os.path.join(ROOT, "out", "web", "_nextglb", n), f"out/web/_nextglb/{n}")

    # the gallery's data: every picture it names comes along, the models it names do not
    m = manifest.build()
    drop = list(DROP) + ([] if town2 else [r"town=town2"]) + ([] if glb else [r"^out/web/_nextglb/town_", r"^glbtown/", r"_spawn\.glb", r"^gltflod/", r"^bakeglb/view"])
    missing = []

    def freeze(o):
        if isinstance(o, dict):
            return {k: freeze(v) for k, v in o.items() if k != "glb"}
        if isinstance(o, list):
            return [freeze(v) for v in o]
        if isinstance(o, str) and re.match(r"^out/[\w./\-]+\.(png|webp|jpg|jpeg)(\?.*)?$", o):
            rel = o.split("?")[0]
            if not add(os.path.join(ROOT, *rel.split("/")), rel):
                missing.append(rel)
        return o

    if not town2:
        m["town2"] = None
    m = freeze(m)
    m["queue"] = {"alive": False}
    # (the time of the newest file, not of this run: the frozen data is the same bytes for the same files, and its
    # part is uploaded again only when they change)
    m["static"] = {"time": max(os.path.getmtime(p) for d, p in site.items() if d.startswith("out/")), "drop": drop}
    m["time"] = m["static"]["time"]
    return site, m, missing


def group_of(dst):
    """the part group of a file of the site's generated data: what changes together"""
    if dst.startswith("out/web/_next/"):
        return "bake"
    if dst.startswith("out/web/_nextglb/"):
        return "glb"
    if dst.startswith("out/web/_"):
        return "bake-" + dst.split("/")[2].strip("_")                     # a second town, the materials, the town pictures
    if dst.startswith("out/web/"):
        return "pictures"
    return "meta"


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_parts(site, manifest_bytes):
    """-> {part name: [(path in the site, source path or bytes)]} - the names carry a hash of the parts' content"""
    files = [(dst, src) for dst, src in sorted(site.items()) if dst.startswith("out/")]
    files.append(("api/manifest", manifest_bytes))
    groups = {}
    for dst, src in files:
        g = group_of(dst)
        if isinstance(src, str) and os.path.getsize(src) > PART_MB * 1e6:
            key = (g, dst)                                                 # its own part
        else:
            key = (g, int(hashlib.sha1(dst.encode()).hexdigest()[:8], 16) % PART_N)
        groups.setdefault(key, []).append((dst, src))
    parts = {}
    for (g, k), members in sorted(groups.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        h = hashlib.sha1()
        for dst, src in members:
            h.update(dst.encode() + b"\0" + (file_hash(src) if isinstance(src, str) else hashlib.sha1(src).hexdigest()).encode() + b"\0")
        stem = re.sub(r"[^\w.\-]+", "_", os.path.basename(k)) if isinstance(k, str) else str(k)
        parts[f"{g}-{stem}-{h.hexdigest()[:12]}.tar"] = members
    return parts


def write_data(folder, site, manifest_bytes):
    """the generated part of the site as parts in `folder`: the tars named by the list, site-data.json, and no others.
    Parts already there with the right name are left as they are. -> the list"""
    os.makedirs(folder, exist_ok=True)
    parts = plan_parts(site, manifest_bytes)
    written = kept = size = 0
    for name, members in parts.items():
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            kept += 1
        else:
            with tarfile.open(path + ".part", "w") as tar:
                for dst, src in members:
                    if isinstance(src, str):
                        tar.add(src, arcname=dst, recursive=False)
                    else:
                        info = tarfile.TarInfo(dst)
                        info.size = len(src)
                        info.mtime = int(time.time())
                        tar.addfile(info, io.BytesIO(src))
            os.replace(path + ".part", path)
            written += 1
        size += os.path.getsize(path)
    for n in os.listdir(folder):
        if n not in parts and n != "site-data.json":
            os.remove(os.path.join(folder, n))                             # a part of an earlier run
    index = {"parts": sorted(parts), "files": sum(len(m) for m in parts.values()), "bytes": size, "time": time.time()}
    with open(os.path.join(folder, "site-data.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"site: {folder}: {len(parts)} parts, {written} written, {kept} as they were, {index['files']} files, {size / 1e6:.1f} MB")
    return index


def upload_data(tag, folder, index):
    """bring the release `tag` up to date with the parts in `folder` (gh: the repository of the checkout)"""
    def gh(*args, **kw):
        r = subprocess.run(["gh", *args], capture_output=True, text=True, **kw)
        if r.returncode:
            sys.exit(f"site: gh {' '.join(args[:3])}: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout
    try:
        have = {a["name"]: a["size"] for a in json.loads(gh("release", "view", tag, "--json", "assets"))["assets"]}
    except SystemExit:
        gh("release", "create", tag, "--title", f"Site data ({tag})", "--notes",
           "The generated data of the GitHub Pages site, in parts: tools/site.py --upload. Not for reading by hand.")
        have = {}
    want = set(index["parts"])
    up = [n for n in sorted(want) if n not in have or have[n] != os.path.getsize(os.path.join(folder, n))]
    gone = [n for n in sorted(have) if n not in want and n != "site-data.json"]
    for n in gone:
        gh("release", "delete-asset", tag, n, "--yes")
    for i, n in enumerate(up):
        print(f"site: upload {i + 1}/{len(up)} {n} ({os.path.getsize(os.path.join(folder, n)) / 1e6:.1f} MB)", flush=True)
        gh("release", "upload", tag, os.path.join(folder, n), "--clobber")
    gh("release", "upload", tag, os.path.join(folder, "site-data.json"), "--clobber")
    mb = sum(os.path.getsize(os.path.join(folder, n)) for n in up) / 1e6
    print(f"site: release {tag}: {len(up)} parts uploaded ({mb:.1f} MB), {len(want) - len(up)} already there, {len(gone)} deleted")


def read_data(path, out):
    """unpack the parts of a --data-out folder (or its site-data.json) below `out` -> {path in the site: bytes};
    plain files with plain relative names only"""
    folder = os.path.dirname(os.path.abspath(path)) if path.endswith(".json") else os.path.abspath(path)
    with open(os.path.join(folder, "site-data.json")) as fh:
        index = json.load(fh)
    got = {}
    for part in index["parts"]:
        with tarfile.open(os.path.join(folder, part), "r") as tar:
            for m in tar:
                name = m.name
                if not m.isfile() or name.startswith("/") or ".." in name.split("/") or not re.match(r"^[\w./\-]+$", name):
                    sys.exit(f"site: {part}: unexpected entry {name!r}")
                f = os.path.join(out, *name.split("/"))
                os.makedirs(os.path.dirname(f), exist_ok=True)
                with tar.extractfile(m) as src, open(f, "wb") as fh:
                    shutil.copyfileobj(src, fh, 1 << 20)
                got[name] = m.size
    if "api/manifest" not in got:
        sys.exit(f"site: {path} has no api/manifest - not written by --data-out")
    return got


def main():
    args = sys.argv[1:]
    opt = lambda k: args[args.index(k) + 1] if k in args else None
    out = os.path.abspath(opt("--out")) if opt("--out") else os.path.join(ROOT, "out", "site")
    data = opt("--data")
    site, m, missing = plan(town2="--town2" in args, glb="--no-glb" not in args, data=not data)
    made = {".nojekyll": b""}
    if m is not None:
        made["api/manifest"] = json.dumps(m, separators=(",", ":")).encode()
    if opt("--data-out") or opt("--upload"):
        folder = os.path.abspath(opt("--data-out") or os.path.join(ROOT, "out", "site-data"))
        index = write_data(folder, site, made["api/manifest"])
        if opt("--upload"):
            upload_data(opt("--upload"), folder, index)
        return
    unpacked = {}
    if "--dry" not in args:
        import srczip
        made["api/source.zip"] = srczip.build(False)[0]
        if os.path.isdir(out):
            if not os.path.isfile(os.path.join(out, ".nojekyll")):
                sys.exit(f"site: {out} exists and is not a site folder made by this tool - not touching it")
            shutil.rmtree(out)
        if data:
            unpacked = read_data(data, out)
        linked = copied = 0
        for dst, src in sorted(site.items()):
            f = os.path.join(out, *dst.split("/"))
            os.makedirs(os.path.dirname(f), exist_ok=True)
            try:
                if "--copy" in args:
                    raise OSError
                os.link(src, f)
                linked += 1
            except OSError:
                shutil.copyfile(src, f)
                copied += 1
        for dst, body in made.items():
            f = os.path.join(out, *dst.split("/"))
            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, "wb") as fh:
                fh.write(body)
        print(f"site: {out}: {linked} linked, {copied} copied, {len(made)} made" + (f", {len(unpacked)} from {data}" if data else ""))

    sizes = dict(unpacked)
    sizes.update({d: os.path.getsize(s) for d, s in site.items()})
    sizes.update({d: len(b) for d, b in made.items()})
    total = sum(sizes.values())
    part = {}
    for d, s in sizes.items():
        k = "/".join(d.split("/")[:3]) if d.startswith("out/web/") else d.split("/")[0] if "/" in d else "(top)"
        if d.startswith("out/web/") and not d.startswith("out/web/_"):
            k = "out/web/<asset pictures>"
        c = part.setdefault(k, [0, 0])
        c[0] += 1
        c[1] += s
    for k, (n, s) in sorted(part.items(), key=lambda x: -x[1][1]):
        print(f"  {k:32s} {n:6d} files {s / 1e6:8.1f} MB")
    big = max(sizes.items(), key=lambda x: x[1])
    print(f"site: {len(sizes)} files, {total / 1e6:.1f} MB, largest {big[0]} {big[1] / 1e6:.1f} MB")
    for name, cap, per_file, count in LIMITS:
        over = sorted(d for d, s in sizes.items() if s > per_file)
        notes = []
        if cap:
            notes.append(f"{total / cap * 100:.0f}% of its {cap / 1e6:.0f} MB")
        if count:
            notes.append(f"{len(sizes)} of {count} files")
        notes.append(f"{len(over)} files over its {per_file / 1e6:.0f} MB a file" + (": " + ", ".join(os.path.basename(x) for x in over[:8]) if over else ""))
        print(f"  {name}: " + "; ".join(notes))
    if missing:
        print(f"  {len(missing)} pictures named by the manifest are not on disk, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
