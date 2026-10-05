#!/usr/bin/env python3
"""Project manifest for the live gallery: scans catalogue, sources, GLB stats, renders and reviews.

    python3 tools/manifest.py          -> writes out/manifest.json and prints a summary
    served live as GET /api/manifest by tools/serve.py (rebuilt at most every few seconds)

Also keeps web-sized copies of the review renders in out/web/<id>/*.webp (made with ImageMagick).
"""
import glob
import json
import os
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
PASS = 75
_lock = threading.Lock()
_cache = {"t": 0.0, "data": None}
_conv_lock = threading.Lock()
_conv_busy = set()


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0


def _load(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def _convert(src, dst, width):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp.webp"
        subprocess.run(["magick", src, "-resize", f"{width}x>", "-quality", "80", tmp], check=True, timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp, dst)
    except Exception:
        pass
    finally:
        with _conv_lock:
            _conv_busy.discard(dst)


def _web(src_rel, dst_rel, width):
    """Return the url (relative to the server root) of a web-sized copy of out/<src_rel>, scheduling
    a conversion when it is missing or stale; falls back to the original meanwhile."""
    src = os.path.join(OUT, src_rel)
    ms = _mtime(src)
    if not ms:
        return None
    dst = os.path.join(OUT, dst_rel)
    if _mtime(dst) >= ms:
        return f"out/{dst_rel}?v={int(ms)}"
    with _conv_lock:
        if dst not in _conv_busy and len(_conv_busy) < 6 and time.time() - ms > 1.0:
            _conv_busy.add(dst)
            threading.Thread(target=_convert, args=(src, dst, width), daemon=True).start()
    return f"out/{src_rel}?v={int(ms)}"


def _reviews(kind, name):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "reviews", kind, name, "r*.json")),
                    key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0)):
        r = _load(p)
        if isinstance(r, dict) and "score" in r:
            r["time"] = _mtime(p)
            out.append(r)
    return out


def build():
    cat = (_load(os.path.join(ROOT, "docs", "catalog.json")) or {}).get("assets", [])
    known = {c["id"] for c in cat}
    for p in glob.glob(os.path.join(ROOT, "assets", "*", "*.py")):
        aid = os.path.basename(p)[:-3]
        c = os.path.basename(os.path.dirname(p))
        if aid not in known and not c.startswith("_") and not aid.startswith("_"):
            cat.append({"id": aid, "category": c, "name": aid.replace("_", " "), "class": "prop", "tier": 1, "brief": "", "deps": []})
    assets = []
    for c in cat:
        aid = c["id"]
        src = os.path.join(ROOT, "assets", c["category"], aid + ".py")
        a = {"id": aid, "name": c["name"], "category": c["category"], "class": c.get("class"), "tier": c.get("tier", 1),
             "brief": c.get("brief", ""), "deps": c.get("deps", []), "source": _mtime(src) > 0, "source_time": _mtime(src)}
        variants = []
        for jp in sorted(glob.glob(os.path.join(OUT, "glb", aid, "*.json"))):
            v = os.path.basename(jp)[:-5]
            if v.startswith("_"):
                continue
            j = _load(jp) or {}
            st = j.get("stats", {})
            variants.append({"name": v, "tris": st.get("tris"), "size": (j.get("bounds") or {}).get("size"),
                             "params": j.get("params"), "glb": f"out/glb/{aid}/{v}.glb?v={int(_mtime(jp))}",
                             "lights": sum(1 for s in j.get("sockets", []) if s.get("kind") == "light"),
                             "warnings": j.get("warnings", [])})
        variants.sort(key=lambda v: (v["name"] != "default", v["name"]))
        a["variants"] = variants
        r = {}
        for k, w in (("hero", 1600), ("sheet", 1600), ("variants", 1600)):
            u = _web(f"renders/{aid}/{k}.png", f"web/{aid}/{k}.webp", w)
            if u:
                r[k] = u
        t = _web(f"renders/{aid}/hero.png", f"web/{aid}/thumb.webp", 560)
        if t:
            r["thumb"] = t
        a["renders"] = r
        a["render_time"] = _mtime(os.path.join(OUT, "renders", aid, "hero.png"))
        rv = _reviews("asset", aid)
        a["reviews"] = rv
        a["score"] = rv[-1]["score"] if rv else None
        a["round"] = len(rv)
        if rv and rv[-1]["score"] >= PASS:
            a["status"] = "passed"
        elif rv:
            a["status"] = "revising"
        elif r.get("hero"):
            a["status"] = "rendered"
        elif a["source"]:
            a["status"] = "authoring"
        else:
            a["status"] = "planned"
        assets.append(a)

    mats = []
    for m in (_load(os.path.join(ROOT, "docs", "materials.json")) or {}).get("materials", []):
        n = m["name"]
        e = {"name": n, "kind": m.get("kind"), "family": m.get("family"), "desc": m.get("desc", ""), "tile": m.get("tile")}
        if m.get("kind") == "baked":
            e["preview"] = _web(f"mat/{n}/preview.png", f"web/_mat/{n}_preview.webp", 640)
            e["far"] = _web(f"mat/{n}/far.png", f"web/_mat/{n}_far.webp", 1280)
            e["sheet"] = _web(f"mat/{n}/sheet.png", f"web/_mat/{n}_sheet.webp", 1600)
        e["swatch"] = _web(f"renders/_mat/{n}.png", f"web/_mat/{n}_swatch.webp", 1280)
        rv = _reviews("mat", n)
        e["reviews"] = rv
        e["score"] = rv[-1]["score"] if rv else None
        e["status"] = ("passed" if rv and rv[-1]["score"] >= PASS else "revising" if rv else
                       "baked" if e.get("preview") else "special" if m.get("kind") == "special" else "planned")
        mats.append(e)

    def _town(name):
        """shots + reviews of one composed town ("town" = the square, "town2" = the second assembly)"""
        d = "_" + name
        shots = []
        for p in sorted(glob.glob(os.path.join(OUT, "renders", d, "*.png"))):
            b = os.path.basename(p)[:-4]
            shots.append({"name": b, "url": _web(f"renders/{d}/{b}.png", f"web/{d}/{b}.webp", 1920),
                          "thumb": _web(f"renders/{d}/{b}.png", f"web/{d}/{b}_t.webp", 720), "time": _mtime(p)})
        trv = _reviews(name, "")
        return {"shots": shots, "reviews": trv, "score": trv[-1]["score"] if trv else None,
                "playable": os.path.exists(os.path.join(ROOT, "web", "town", "index.html")),
                "layout": _mtime(os.path.join(OUT, name, "town.json")) > 0}

    town = _town("town")
    town2 = _town("town2")

    baked = [m for m in mats if m["kind"] == "baked"]
    real = [a for a in assets]
    stats = {
        "assets_total": len(real),
        "assets_authored": sum(1 for a in real if a["source"]),
        "assets_rendered": sum(1 for a in real if a["renders"].get("hero")),
        "assets_reviewed": sum(1 for a in real if a["reviews"]),
        "assets_passed": sum(1 for a in real if a["status"] == "passed"),
        "variants": sum(len(a["variants"]) for a in real),
        "tris": sum((v["tris"] or 0) for a in real for v in a["variants"]),
        "mats_total": len(baked), "mats_passed": sum(1 for m in baked if m["status"] == "passed"),
        "mats_special": len(mats) - len(baked),
        "town_score": town["score"],
        "avg_score": (round(sum(a["score"] for a in real if a["score"] is not None) / max(1, sum(1 for a in real if a["score"] is not None)), 1)),
    }
    hb = _load(os.path.join(OUT, "queue", "daemon.json")) or {}
    queue = {"status": hb.get("status"), "jobs": hb.get("jobs"), "failed": hb.get("failed"), "gpu": hb.get("gpu"),
             "alive": time.time() - (hb.get("time") or 0) < 60,
             "pending": len([p for p in glob.glob(os.path.join(OUT, "queue", "*.json")) if not p.endswith("daemon.json")])}
    return {"time": time.time(), "pass": PASS, "stats": stats, "queue": queue, "assets": assets, "materials": mats, "town": town, "town2": town2,
            "refs": os.path.isfile(os.path.join(ROOT, "docs", "ref", "board.png"))}


def get(max_age=4.0):
    with _lock:
        if _cache["data"] is None or time.time() - _cache["t"] > max_age:
            _cache["data"] = json.dumps(build()).encode()
            _cache["t"] = time.time()
        return _cache["data"]


if __name__ == "__main__":
    d = build()
    time.sleep(0.1)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(d, f)
    print(json.dumps(d["stats"], indent=1), json.dumps(d["queue"]))
