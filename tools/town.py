#!/usr/bin/env python3
"""Town pipeline (host python, stdlib only).

    tools/town.py gen [--seed N] [--set k=v,...]   layout generator        -> out/town/layout.json
    tools/town.py build [--procs 8] [--force]      Blender CLI builds of every distinct (asset, variant, params)
                                                   used by the layout (+LODs) -> out/glb/<asset>/_<hash>.glb
    tools/town.py terrain                          ground / skirts / water  -> out/glb/_town/*.glb
    tools/town.py resolve                          anchors, stacking, sockets, colliders, lights -> out/town/town.json
    tools/town.py all [--seed N]                   gen + build + terrain + resolve
    tools/town.py release [--hold a,b] [--only a,b]   copy finished asset sources to release/ (what the town is built from)
    tools/town.py render [--shots a,b] [--quality studio] [--size 1600x900]
                                                   review shots through the GPU browser -> out/renders/_town/*.png
    tools/town.py status
"""
import concurrent.futures
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TSQ_TOWN=town2 selects a second, separate town assembly: town2/layout.py -> out/town2/, terrain out/glb/_town2/,
# renders out/renders/_town2/ (page /town/?town=town2). Unset = the first town, exactly as before.
TOWN_ID = os.environ.get("TSQ_TOWN") or "town"
sys.path.insert(0, os.path.join(ROOT, TOWN_ID))
BLENDER = os.environ.get("TSQ_BLENDER") or os.environ.get("BLENDER") or shutil.which("blender") or os.path.expanduser("~/.local/bin/blender")
OUT = os.path.join(ROOT, "out", TOWN_ID)
GLB = os.path.join(ROOT, "out", "glb")
LAYOUT = os.path.join(OUT, "layout.json")
TOWN = os.path.join(OUT, "town.json")
BUILDLOG = os.path.join(OUT, "build.json")


def opts(argv):
    pos, kw = [], {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            k = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                kw[k] = argv[i + 1]
                i += 1
            else:
                kw[k] = True
        else:
            pos.append(argv[i])
        i += 1
    return pos, kw


def wjson(path, obj, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def item_key(it):
    return json.dumps([it["asset"], it.get("variant"), it.get("params") or {}], sort_keys=True)


def item_name(it):
    return "_" + hashlib.sha1(item_key(it).encode()).hexdigest()[:10]


# ------------------------------------------------------------------------------------------------ gen

def cmd_gen(argv):
    import layout
    pos, kw = opts(argv)
    ov = {}
    for pair in (kw.get("set") or "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            ov[k] = float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
    t = layout.generate(int(kw.get("seed", 1)), **ov)
    wjson(LAYOUT, t, indent=1)
    used = sorted({i["asset"] for i in t["items"]})
    print(f"layout seed {t['seed']}: {len(t['items'])} items of {len(used)} assets, {len(t['regions'])} regions, "
          f"{len(t['terrain'])} terrain patches, {len(t['blockouts'])} building blockouts, {len(t['missing'])} assets not authored yet")
    pl = t.get("placement") or {}
    if pl:       # layout.generate has already asserted that no prop stands on a flight of stairs / walking route
        print(f"placement: {pl['props']} ground props clear of {len(pl['stairs'])} stairs + {len(pl['routes'])} routes; "
              f"{len(pl['nudged'])} moved, {len(pl['dropped'])} dropped, {len(pl['sight'])} in a review frame, {pl['unbuilt']} socket look-ups unbuilt")
        for d in pl["dropped"]:
            print("  dropped:", d)
    return 0


# ------------------------------------------------------------------------------------------------ build

_CAT = None


def catalog():
    global _CAT
    if _CAT is None:
        try:
            c = json.load(open(os.path.join(ROOT, "docs", "catalog.json")))
            _CAT = {a["id"]: a for a in (c["assets"] if isinstance(c, dict) else c)}
        except Exception:
            _CAT = {}
    return _CAT


def asset_src(aid):
    for base in filter(None, [os.environ.get("TSQ_ASSET_DIR"), "assets"]):   # released copy first (see cmd_release)
        hits = glob.glob(os.path.join(ROOT, base, "*", aid + ".py"))
        if hits:
            return hits[0]
    return None


RELEASE = "release"    # ROOT/release/<category>/<id>.py: the asset sources the town is built from


def cmd_release(argv):
    """Copy finished asset sources into release/. The town (layout, builds, use() composition) reads an asset from
    there when a copy exists, so authors can keep revising assets/ without half-done states reaching the town.
    --hold a,b   leave these ids as released before (their authors are mid-revision)
    --only a,b   release just these ids"""
    import filecmp
    import shutil
    pos, kw = opts(argv)
    hold = set(filter(None, (kw.get("hold") or "").split(",")))
    only = set(filter(None, (kw.get("only") or "").split(",")))
    new, changed, held = [], [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "assets", "*", "*.py"))):
        aid = os.path.splitext(os.path.basename(p))[0]
        if aid.startswith("_") or (only and aid not in only):
            continue
        dst = os.path.join(ROOT, RELEASE, os.path.basename(os.path.dirname(p)), aid + ".py")
        have = os.path.exists(dst)
        if aid in hold:
            if have and not filecmp.cmp(p, dst, shallow=False):
                held.append(aid)
            continue
        if have and filecmp.cmp(p, dst, shallow=False):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
        os.utime(dst)       # dated now, not as the working file: `build` rebuilds what is older than its source
        (changed if have else new).append(aid)
    print(f"release: {len(new)} new, {len(changed)} updated, {len(held)} held back" + (f" ({', '.join(held)})" if held else ""))
    if changed:
        print("  updated: " + ", ".join(changed))
    never = sorted(a for a in hold if not glob.glob(os.path.join(ROOT, RELEASE, "*", a + ".py")))
    if never:
        print("  held but never released (the town still reads the working file): " + ", ".join(never))
    return 0


def unreleased():
    """ids whose working source (assets/) differs from the released copy the town reads"""
    import filecmp
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "assets", "*", "*.py"))):
        aid = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(ROOT, RELEASE, os.path.basename(os.path.dirname(p)), aid + ".py")
        if not aid.startswith("_") and os.path.exists(dst) and not filecmp.cmp(p, dst, shallow=False):
            out.append(aid)
    return out


_FITTING = re.compile(r"lantern|lamp(?!s)|globe|fire|festoon|flame|candle|bulb|torch|mantle")
_HELPER = re.compile(r"spill|pool|fill|bounce|wash|glow")


def light_is_fitting(name):
    """True when a light socket sits at a visible burner (a lantern, lamp, globe, open fire, festoon bulb) and so may show
    glare at its centre; False for the helper lights of windows, shop fronts, dials, signs and for one light standing
    in for a row of fittings (`..._lamps`), which have no source to look at."""
    n = name.lower()
    return bool(_FITTING.search(n)) and not _HELPER.search(n)


def src_mtime(aid, seen=None):
    """newest mtime of the asset source and of everything it composes with use()"""
    seen = seen if seen is not None else set()
    if aid in seen:
        return 0
    seen.add(aid)
    p = asset_src(aid)
    if not p:
        return 0
    m = os.path.getmtime(p)
    deps = set(catalog().get(aid, {}).get("deps") or [])
    try:
        import re
        deps |= set(re.findall(r"""use\(\s*["']([a-z0-9_]+)["']""", open(p).read()))
    except OSError:
        pass
    for d in deps:
        m = max(m, src_mtime(d, seen))
    return m


def blender(script, args, timeout=3600):
    # PYTHONHASHSEED only reaches Blender's python with --python-use-system-env (same as tools/tsq run_blender)
    cmd = [BLENDER, "--background", "--factory-startup", "--python-use-system-env", "--python",
           os.path.join(ROOT, "tools", script), "--"] + args
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED="0"))
    res, tail = [], []
    for line in p.stdout.splitlines():
        if line.startswith("TSQ_RESULT "):
            res.append(json.loads(line[11:]))
        elif line.strip() and not line.startswith(("Blender ", "Read prefs", "INFO", "Info", "DEBUG", "WARNING")):
            tail.append(line)
    if p.returncode != 0 or not res:
        tail += p.stderr.splitlines()[-15:]
    return res, tail[-25:]


def cmd_build(argv):
    pos, kw = opts(argv)
    t = json.load(open(LAYOUT))
    force = bool(kw.get("force"))
    only = set(kw["only"].split(",")) if kw.get("only") else None
    per = {}
    for it in t["items"]:
        per.setdefault(it["asset"], {})[item_name(it)] = it
    work = {}
    for aid, names in per.items():
        if only and aid not in only:
            continue
        sm = src_mtime(aid)
        jobs = []
        for name, it in names.items():
            side = os.path.join(GLB, aid, name + ".json")
            lod2 = os.path.join(GLB, aid, name + ".lod2.glb")
            if not force and os.path.exists(side) and os.path.exists(lod2) and os.path.getmtime(side) > sm:
                continue
            jobs.append({"name": name, "variant": it.get("variant"), "params": it.get("params") or {}})
        if jobs:
            work[aid] = jobs
    total = sum(len(j) for j in work.values())
    print(f"{sum(len(n) for n in per.values())} distinct builds of {len(per)} assets; {total} to (re)build in {len(work)} Blender runs", flush=True)
    stale = unreleased()
    if stale:
        print(f"NOTE: {len(stale)} working sources differ from their released copies, which is what gets built "
              f"(`town.py release` first?): {', '.join(stale[:12])}{' ...' if len(stale) > 12 else ''}", flush=True)
    try:
        log = json.load(open(BUILDLOG))
    except Exception:
        log = {}
    if not work:
        return 0
    os.makedirs(os.path.join(OUT, "jobs"), exist_ok=True)
    t0 = time.time()

    def run(aid):
        jf = os.path.join(OUT, "jobs", aid + ".json")
        wjson(jf, work[aid])
        try:
            res, tail = blender("bl_build.py", ["--asset", aid, "--jobs", jf, "--lods", "1"])
        except subprocess.TimeoutExpired:
            res, tail = [], ["timeout"]
        return aid, res, tail

    bad = 0
    order = sorted(work, key=lambda a: -len(work[a]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(kw.get("procs", 8))) as ex:
        for aid, res, tail in ex.map(run, order):
            got = {r["variant"]: r for r in res}
            for j in work[aid]:
                r = got.get(j["name"])
                if r is None or r.get("error"):
                    bad += 1
                    err = (r or {}).get("error") or "\n".join(tail)
                    log[f"{aid}/{j['name']}"] = {"ok": False, "error": err[-1500:], "job": j}
                    print(f"FAIL {aid}/{j['name']} {j['variant']} {j['params']}\n     " + err.strip().splitlines()[-1][:300] if err.strip() else "")
                    # a failed build must not leave a stale sidecar that looks valid
                else:
                    log[f"{aid}/{j['name']}"] = {"ok": True, "tris": r["tris"], "notes": r.get("notes", []), "errors": r.get("errors", []),
                                                "warnings": r.get("warnings", []), "seconds": r.get("seconds"),
                                                # which copy ships at lod 1 / 2: authored | reduced | decimated
                                                "lods": [d.get("mode", "decimated") for d in r.get("lods") or []],
                                                "lod_s": r.get("lod_s")}
                    for n in r.get("notes", []):
                        print(f"note {aid}/{j['name']}: {n}")
    wjson(BUILDLOG, log, indent=1)
    modes = {}
    for v in log.values():
        for n, m in enumerate(v.get("lods") or []):
            modes.setdefault(n + 1, {}).setdefault(m, 0)
            modes[n + 1][m] += 1
    for n in sorted(modes):
        print(f"lod {n} ships: " + ", ".join(f"{k} {c}" for k, c in sorted(modes[n].items())))
    print(f"built {total - bad}/{total} in {time.time() - t0:.0f} s" + (f"  ({bad} FAILED - see out/town/build.json)" if bad else ""))
    return 0


def floaters(t):
    """What floats in the town's water: the outline (world x, y) every placed asset with a `float` socket parts the
    water with - a boat's hull at its waterline. bl_terrain.py cuts them out of the water it lays."""
    cuts = []
    waters = [w for w in t["terrain"] if w.get("water") and w.get("kind", "rect") == "rect"]
    for it in t["items"]:
        if "attach" in it or "on" in it:
            continue
        p = os.path.join(GLB, it["asset"], item_name(it) + ".json")
        if not os.path.exists(p):
            continue
        x, y, z = it["pos"]
        s, rot = float(it.get("scale", 1.0)), float(it["rot"])
        for q in json.load(open(p)).get("sockets", []):
            ol = q["kind"] == "float" and q["props"].get("outline")
            if not ol:
                continue
            for w in waters:
                x0, y0, x1, y1 = w["rect"]
                if x0 < x < x1 and y0 < y < y1 and abs(z + q["pos"][2] * s - w["z"]) < 0.25:
                    pts = [rot2((q["pos"][0] + ol[i]) * s, (q["pos"][1] + ol[i + 1]) * s, rot) for i in range(0, len(ol) - 1, 2)]
                    cuts.append({"water": w["name"], "asset": it["asset"], "outline": [[round(x + a, 4), round(y + b, 4)] for a, b in pts]})
    return cuts


def cmd_terrain(argv):
    cuts = floaters(json.load(open(LAYOUT)))
    wjson(os.path.join(OUT, "floaters.json"), cuts)
    print(f"terrain: {len(cuts)} floaters part the water")
    res, tail = blender("bl_terrain.py", ["--layout", LAYOUT, "--cuts", os.path.join(OUT, "floaters.json")] + (["--town", TOWN_ID] if TOWN_ID != "town" else []))
    for r in res:
        print("terrain:", json.dumps(r))
    if not res:
        print("\n".join(tail))
        return 1
    return 0


# ------------------------------------------------------------------------------------------------ resolve

def rot2(x, y, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (x * c - y * s, x * s + y * c)


FLAT = 0.2     # things lower than this never block the player


def cmd_resolve(argv):
    t = json.load(open(LAYOUT))
    protos, items, colliders, lights, notes = {}, [], [], [], []
    regions = [dict(r) for r in t["regions"]]
    sides = {}
    placed = {}       # layout index -> resolved item (for on / attach)
    src = []          # resolved item -> layout index

    def sidecar(it):
        k = it["asset"] + "/" + item_name(it)
        if k not in sides:
            p = os.path.join(GLB, it["asset"], item_name(it) + ".json")
            try:
                sides[k] = json.load(open(p)) if os.path.exists(os.path.splitext(p)[0] + ".glb") else None
            except Exception:
                sides[k] = None
        return k, sides[k]

    for idx, it in enumerate(t["items"]):
        k, sc = sidecar(it)
        if sc is None:
            notes.append(f"{it['asset']} ({it.get('tag')}) not built - skipped")
            continue
        b0, b1 = sc["bounds"]["min"], sc["bounds"]["max"]
        s = float(it.get("scale", 1.0))
        rot = float(it["rot"])
        x, y, z = it["pos"]
        if "attach" in it:
            par = placed.get(it["attach"]["to"])
            if par is None:
                notes.append(f"{it['asset']} ({it.get('tag')}): parent missing - skipped")
                continue
            role = it["attach"].get("role")
            sock = [q for q in par["_side"]["sockets"] if q["kind"] == "attach" and (q["props"].get("role") == role or q["name"] == role)]
            if not sock:
                notes.append(f"{it['asset']} ({it.get('tag')}): parent {par['asset']} has no '{role}' socket - skipped")
                continue
            q = sock[idx % len(sock)]
            ox, oy = rot2(q["pos"][0] * par["s"], q["pos"][1] * par["s"], par["rot"])
            x, y, z = par["pos"][0] + ox, par["pos"][1] + oy, par["pos"][2] + q["pos"][2] * par["s"]
            rot = par["rot"] + (q.get("rot") or [0, 0, 0])[2]
        elif "on" in it:
            par = placed.get(it["on"])
            if par is None:
                notes.append(f"{it['asset']} ({it.get('tag')}): support missing - skipped")
                continue
            top = [q for q in par["_side"]["sockets"] if q["kind"] == "attach" and q["props"].get("role") == "prop_top"]
            if top:
                ox, oy = rot2(top[0]["pos"][0] * par["s"], top[0]["pos"][1] * par["s"], par["rot"])
                x, y, z = par["pos"][0] + ox, par["pos"][1] + oy, par["pos"][2] + top[0]["pos"][2] * par["s"]
            else:
                x, y, z = par["pos"][0], par["pos"][1], par["pos"][2] + par["_side"]["bounds"]["max"][2] * par["s"]
        ox = -(b0[0] + b1[0]) / 2 * s if it.get("cx") else 0.0
        oy = 0.0
        if it.get("anchor") == "miny":
            oy = -b0[1] * s
        elif it.get("anchor") == "maxy":
            oy = -b1[1] * s
        if ox or oy:
            dx, dy = rot2(ox, oy, rot)
            x, y = x + dx, y + dy
        if k not in protos:
            st = sc["stats"]
            protos[k] = {"asset": it["asset"], "cls": sc["meta"].get("class", "prop"), "mount": sc["meta"].get("mount", "ground"),
                         "min": b0, "max": b1, "tris": [st["tris"]] + [l["tris"] for l in st.get("lods", [])],
                         "bytes": [st["glb_bytes"]] + [l["glb_bytes"] for l in st.get("lods", [])],
                         "lights": sum(1 for q in sc["sockets"] if q["kind"] == "light"),
                         "fx": sum(1 for q in sc["sockets"] if q["kind"] == "fx"), "parts": len(sc.get("parts") or {}),
                         "v": int(os.path.getmtime(os.path.join(GLB, it["asset"], item_name(it) + ".glb")))}
        r = {"p": k, "pos": [round(x, 3), round(y, 3), round(z, 3)], "rot": round(rot % 360, 3), "s": s, "tag": it.get("tag", "")}
        items.append(r)
        src.append(idx)
        placed[idx] = dict(r, asset=it["asset"], _side=sc)

        # ---- lights (world space)
        for q in sc["sockets"]:
            if q["kind"] != "light":
                continue
            lx, ly = rot2(q["pos"][0] * s, q["pos"][1] * s, rot)
            pr = q["props"]
            fit = light_is_fitting(q["name"])
            lights.append({"pos": [round(x + lx, 3), round(y + ly, 3), round(z + q["pos"][2] * s, 3)], "color": pr.get("color", "#ffbf6a"),
                           "intensity": pr.get("intensity", 40.0), "range": pr.get("range", 9.0), "flicker": pr.get("flicker", 0.1),
                           "near": pr.get("shadow_near", 0.3), "item": len(items) - 1, "cls": protos[k]["cls"],
                           "src": int(fit), "name": q["name"]})
            # the glow sprite: only a fitting has one (a helper light has nothing to look at). `halo` = the output it
            # stands for, `halo_at` = where its burners are seen when the light sits off them or stands for several,
            # `halo_r` = the radius of an opaque glowing body around a burner (tk.asm light())
            lights[-1]["halo"] = round(float(pr.get("halo", pr.get("intensity", 40.0))), 3) if fit else 0
            if fit and pr.get("halo_at"):
                at = pr["halo_at"] if isinstance(pr["halo_at"][0], list) else [pr["halo_at"]]
                lights[-1]["halo_at"] = [[round(x + hx, 3), round(y + hy, 3), round(z + a[2] * s, 3)]
                                         for a in at for hx, hy in [rot2(a[0] * s, a[1] * s, rot)]]
            if fit and pr.get("halo_r"):
                lights[-1]["halo_r"] = round(float(pr["halo_r"]) * s, 3)

        # ---- walkable ramps
        P = it.get("params") or {}
        if it.get("nav") == "stairs":
            fx_, fy_ = rot2(0, b0[1] * s, rot)
            dx, dy = rot2(0, 1, rot)
            rise = float(P.get("rise", b1[2])) * s
            ln = (b1[1] - b0[1]) * s
            hw = (b1[0] - b0[0]) / 2 * s
            cx_, cy_ = rot2((b0[0] + b1[0]) / 2 * s, (b0[1] + b1[1]) / 2 * s, rot)
            ex, ey = abs(rot2(hw, ln / 2, rot)[0]), abs(rot2(hw, ln / 2, rot)[1])
            ex, ey = max(ex, abs(rot2(hw, -ln / 2, rot)[0])), max(ey, abs(rot2(hw, -ln / 2, rot)[1]))
            regions.append({"name": "stairs_" + it.get("tag", str(idx)), "rect": [round(x + cx_ - ex, 3), round(y + cy_ - ey, 3), round(x + cx_ + ex, 3), round(y + cy_ + ey, 3)],
                            "z": round(z, 3), "ramp": {"o": [round(x + fx_, 3), round(y + fy_, 3)], "d": [round(dx, 4), round(dy, 4)], "len": round(ln, 3),
                                                       "z0": round(z, 3), "z1": round(z + rise, 3)}})

        # ---- colliders
        mode = it.get("collide", "auto")
        zr = [round(z + b0[2] * s, 3), round(z + b1[2] * s, 3)]

        def box(lx0, ly0, lx1, ly1, zr=zr):
            cxl, cyl = (lx0 + lx1) / 2 * s, (ly0 + ly1) / 2 * s
            wx, wy = rot2(cxl, cyl, rot)
            colliders.append({"t": "b", "c": [round(x + wx, 3), round(y + wy, 3)], "h": [round(abs(lx1 - lx0) / 2 * s, 3), round(abs(ly1 - ly0) / 2 * s, 3)],
                              "r": round(rot % 360, 3), "z": zr, "i": len(items) - 1})

        def circ(rad):
            cxl, cyl = rot2((b0[0] + b1[0]) / 2 * s, (b0[1] + b1[1]) / 2 * s, rot)
            colliders.append({"t": "c", "c": [round(x + cxl, 3), round(y + cyl, 3)], "rad": round(rad, 3), "z": zr, "i": len(items) - 1})

        sx_, sy_, sz_ = (b1[0] - b0[0]) * s, (b1[1] - b0[1]) * s, (b1[2] - b0[2]) * s
        if mode == "none":
            pass
        elif mode == "building":
            w, d = float(P.get("width", sx_)), float(P.get("depth", sy_))
            box(-w / 2, 0, w / 2, d, [zr[0], zr[1]])
        elif mode == "passage":
            w, d, pw = float(P.get("width", sx_)), float(P.get("depth", sy_)), float(P.get("passage_width", 3.0))
            box(-w / 2, -d / 2, -pw / 2, d / 2)
            box(pw / 2, -d / 2, w / 2, d / 2)
        elif mode.startswith("circle"):
            circ(float(mode.split(":")[1]) if ":" in mode else min(sx_, sy_) / 2 * 0.9)
        else:
            if sz_ < FLAT or b0[2] * s > 1.9 or (protos[k]["mount"] in ("wall", "hang") and b0[2] * s > 0.6):
                pass
            elif max(sx_, sy_) < 1.3 and min(sx_, sy_) > 0.55 * max(sx_, sy_):
                circ(max(sx_, sy_) / 2 * 0.85)
            else:
                box(b0[0], b0[1], b1[0], b1[1])
        if it.get("nav") == "stairs":
            # cheeks: the flanks of a flight cannot be walked through
            box(b0[0] - 0.05, b0[1], b0[0] + 0.12, b1[1])
            box(b1[0] - 0.12, b0[1], b1[0] + 0.05, b1[1])

    # blockouts for buildings that are not authored yet keep the space readable (and collide)
    blockouts = []
    for b in t.get("blockouts", []):
        w, d = b["width"], b["depth"]
        x, y, z = b["pos"]
        parts = [(-w / 2, 0, w / 2, d)] if not b.get("centered") else [(-w / 2, -d / 2, -b["passage_width"] / 2, d / 2), (b["passage_width"] / 2, -d / 2, w / 2, d / 2)]
        for (x0, y0, x1, y1) in parts:
            wx, wy = rot2((x0 + x1) / 2, (y0 + y1) / 2, b["rot"])
            blockouts.append({"asset": b["asset"], "c": [round(x + wx, 3), round(y + wy, 3), z], "h": [abs(x1 - x0) / 2, abs(y1 - y0) / 2], "r": b["rot"], "height": b["height"]})
            colliders.append({"t": "b", "c": [round(x + wx, 3), round(y + wy, 3)], "h": [abs(x1 - x0) / 2, abs(y1 - y0) / 2], "r": b["rot"], "z": [z, z + b["height"]], "i": -1})

    terrain = []
    for n in ("terrain", "water"):
        p = os.path.join(GLB, "_" + TOWN_ID, n + ".glb")
        if os.path.exists(p):
            terrain.append({"file": "_" + TOWN_ID + "/" + n, "v": int(os.path.getmtime(p))})
    tris = sum(protos[i["p"]]["tris"][0] for i in items)
    out = {"seed": t["seed"], "frame": t["frame"], "spawn": t["spawn"], "shots": t["shots"], "regions": regions, "protos": protos, "items": items,
           "colliders": colliders, "lights": lights, "terrain": terrain, "blockouts": blockouts, "missing": sorted(t["missing"]),
           "notes": notes, "stats": {"items": len(items), "protos": len(protos), "assets": len({p["asset"] for p in protos.values()}),
                                     "lights": len(lights), "tris_lod0": tris, "colliders": len(colliders), "built": int(time.time())}}
    wjson(TOWN, out)
    print(f"town.json: {len(items)} items / {len(protos)} distinct builds / {out['stats']['assets']} assets, {len(lights)} lights, "
          f"{len(colliders)} colliders, {tris / 1e6:.2f} M tris at LOD0, {len(blockouts)} blockouts, {len(notes)} notes")
    for n in sorted(set(notes))[:12]:
        print("  note:", n)
    import layout
    bad = layout.check_resolved(t, out, src)     # the placement assertion again, against the real stairs and colliders
    for b in bad[:40]:
        print("FAIL placement:", b)
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------ render

DRIVER = r'''
import json, time
url = %(url)r
def open_tab():
    tid = None
    for t in cdp("Target.getTargets").get("targetInfos", []):
        if t.get("type") == "page" and "/town/" in t.get("url", "") and "capture=1" in t.get("url", ""):
            tid = t["targetId"]
    if tid is None:
        tid = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
    return tid, cdp("Target.attachToTarget", targetId=tid, flatten=True)["sessionId"]
tid, sid = open_tab()
def ev(expr):
    try:
        r = cdp("Runtime.evaluate", session_id=sid, expression=expr, returnByValue=True)
    except Exception as e:      # the page is busy (streaming / compiling): poll again
        return {"busy": str(e)[:200]}
    if r.get("exceptionDetails"):
        return {"exception": str(r["exceptionDetails"])[:600]}
    return (r.get("result") or {}).get("value")
cdp("Page.enable", session_id=sid)
# a freshly launched Chrome can answer requestAdapter() with null for its first seconds, and the town page then falls
# back to WebGL2 without a word: wait on a plain http page until the hardware adapter is there
cdp("Page.navigate", session_id=sid, url=url.split("/town/")[0] + "/gallery/")
gpu = None
for _ in range(40):
    time.sleep(0.5)
    try:
        r = cdp("Runtime.evaluate", session_id=sid, awaitPromise=True, returnByValue=True,
                expression="(async () => (await navigator.gpu?.requestAdapter())?.info.vendor || null)()")
        gpu = (r.get("result") or {}).get("value")
    except Exception:
        gpu = None
    if gpu:
        break
print("TOWN_GPU " + str(gpu))
cdp("Page.navigate", session_id=sid, url=url)
t0 = time.time()
res = None
lost = 0
while time.time() - t0 < %(timeout)d:
    time.sleep(1.0)
    v = ev("JSON.stringify(window.__capture || null)")
    # the browser went away under us (the render daemon recycles it when its GPU process leaks too much): start over
    if isinstance(v, dict) and "busy" in v and any(k in v["busy"] for k in ("close frame", "No session", "not found", "Connection", "closed")):
        lost += 1
        if lost > 3:            # this CDP connection is dead for good: let the caller start a fresh one
            print("TOWN_RETRY")
            raise SystemExit(0)
        continue
    lost = 0
    try:
        st = json.loads(v) if isinstance(v, str) else None
    except Exception:
        st = None
    if st and (st.get("done") or st.get("failed")):
        res = st
        break
if res is None:
    res = {"done": False, "error": "timeout", "state": ev("JSON.stringify(window.__capture || null)"), "log": ev("JSON.stringify((window.__log || []).slice(-12))")}
for _ in range(3):
    try:
        cdp("Target.closeTarget", targetId=tid)
        break
    except Exception:
        time.sleep(2)
print("TOWN_RESULT " + json.dumps(res))
'''


def cmd_render(argv):
    # quality: studio = the streaming engine, the picture the town is judged in (high / medium / low / phone are the
    # page's baked tiers; studio-medium the streaming engine's medium)
    pos, kw = opts(argv)
    token = open(os.path.join(ROOT, "out", ".upload_token")).read().strip()
    w, h = (kw.get("size") or "1600x900").split("x")
    q = f"capture=1&token={token}&quality={kw.get('quality', 'studio')}&w={w}&h={h}&t={int(time.time())}"
    if kw.get("shots"):
        q += "&shots=" + kw["shots"]
    if kw.get("out"):
        q += "&out=" + kw["out"]
    if kw.get("webgl"):
        q += "&webgl=1"
    if kw.get("look"):
        q += "&" + kw["look"].strip("&")
    if TOWN_ID != "town":
        q += "&town=" + TOWN_ID
    url = f"http://127.0.0.1:{os.environ.get('TSQ_PORT', '8765')}/town/?{q}"
    # town captures get their own GPU Chrome (CDP :9224): in the asset daemon's Chrome they queue behind every asset
    # job in the same GPU process and are killed by its VRAM recycling. Stopped afterwards so the VRAM is returned.
    port = os.environ.get("TSQ_TOWN_CHROME_PORT", "9224")
    cenv = dict(os.environ, TSQ_CHROME_PORT=port, TSQ_CHROME_PROFILE=os.environ.get("TSQ_TOWN_CHROME_PROFILE") or
                os.path.join(ROOT, ".scratch", "chrome", "tsq-chrome-" + port))
    chrome = os.path.join(ROOT, "tools", "gpu_chrome.sh")
    try:
        for attempt in range(3):
            subprocess.run([chrome], env=cenv, check=True, capture_output=True)
            env = dict(os.environ, BU_NAME=f"tsqtown{port}{'r%d' % attempt if attempt else ''}", BU_CDP_URL=f"http://localhost:{port}")
            p = subprocess.run(["browser-use"], input=DRIVER % {"url": url, "timeout": int(kw.get("timeout", 420))}, capture_output=True, text=True,
                               env=env, timeout=int(kw.get("timeout", 420)) + 60)
            if "TOWN_RETRY" not in p.stdout:
                break
            print("browser went away during the capture, retrying")
            subprocess.run([chrome, "stop"], env=cenv, capture_output=True)
            time.sleep(5)
    finally:
        if not kw.get("keep"):
            subprocess.run([chrome, "stop"], env=cenv, capture_output=True)
    for line in p.stdout.splitlines():
        if line.startswith("TOWN_RESULT "):
            r = json.loads(line[12:])
            if not r.get("done"):
                print("render FAILED:", json.dumps(r)[:3000])
                return 1
            print(f"rendered {len(r.get('shots', []))} shots in {r.get('seconds')} s  (load {r.get('load_s')} s, {r.get('gpu')}, "
                  f"{r.get('draws')} draw calls, {r.get('tris')} tris, {r.get('fps')} fps at the spawn; "
                  f"{r.get('cpu_ms')} ms CPU/frame, {r.get('objects')} render objects, {json.dumps(r.get('spawn'))})")
            if r.get("probe"):
                print("  probe:", json.dumps(r["probe"]))
            for s in r.get("shots", []):
                print("  out/renders/" + s)
            for wmsg in r.get("warnings", [])[:10]:
                print("  warn:", wmsg)
            return 0
    print("render FAILED (no result)\n" + p.stdout[-1500:] + p.stderr[-1500:])
    return 1


def cmd_status(argv):
    for p in (LAYOUT, TOWN):
        if os.path.exists(p):
            t = json.load(open(p))
            if p == TOWN:
                print("town.json:", json.dumps(t["stats"]), "missing:", len(t["missing"]))
            else:
                print("layout.json:", len(t["items"]), "items; missing:", ", ".join(sorted(t["missing"])))
    return 0


def cmd_all(argv):
    for fn in (cmd_gen, cmd_build):
        r = fn(argv)
        if r:
            return r
    # rules that read the sockets / real bounds of built assets need those builds: lay out again until it is stable
    for _ in range(3):
        before = open(LAYOUT).read()
        r = cmd_gen(argv)
        if r:
            return r
        if open(LAYOUT).read() == before:
            break
        r = cmd_build(argv)
        if r:
            return r
    for fn in (cmd_terrain, cmd_resolve):
        r = fn(argv)
        if r:
            return r
    return 0


def main():
    cmds = {"gen": cmd_gen, "build": cmd_build, "terrain": cmd_terrain, "resolve": cmd_resolve, "all": cmd_all, "render": cmd_render,
            "status": cmd_status, "release": cmd_release}
    if os.path.isdir(os.path.join(ROOT, RELEASE)):
        os.environ.setdefault("TSQ_ASSET_DIR", RELEASE)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return 1
    return cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
