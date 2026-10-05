#!/usr/bin/env python3
"""Holes of the cheaper levels (tk/holes.py; docs/LOD.md "No holes") on built files, read-only.

    python3 tools/checks/holes.py town --out .scratch/holes/before.json            # every variant the town uses (out/town/town.json protos)
    python3 tools/checks/holes.py town --dir .scratch/holes/glb --out after.json    # ... the same variants built somewhere else (<dir>/<id>/<variant>...)
    python3 tools/checks/holes.py townhouse_narrow/_388fad1151 barrel/default       # single variants
    python3 tools/checks/holes.py report before.json [after.json]                   # the tables again from result files
    ... --phase 0.12      # the viewers stand elsewhere on their rings than those the build repairs for (0.37, 0.87)

Per variant and level 1 / 2: the share of the street views' samples where LOD 0 shows a front face and the level
shows nothing (holes = back + none), the worst single view, and where they are (LOD 0 material, position).
Starts at most --jobs Blender processes (default 2); inside Blender the same file is the worker.
"""
import sys
import os
import shutil
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.environ.get("TSQ_BLENDER") or os.environ.get("BLENDER") or shutil.which("blender") or os.path.expanduser("~/.local/bin/blender")

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


def worker(listing, out):
    sys.path.insert(0, ROOT)
    from tk import lodimage, holes
    items = json.load(open(listing))
    with open(out, "w") as f:
        for it in items:
            t0 = time.time()
            rec = {"key": it["key"], "mount": it["mount"], "lv": []}
            try:
                p0 = lodimage.load_glb(it["glb0"], cache=False)
                H = holes.of_prims(p0, it["mount"], **({"phase": it["phase"]} if it.get("phase") is not None else {}))
                rec["tris0"], rec["n"], rec["rays"], rec["views"] = lodimage.tris_of(p0), H.n, H.rays, len(H.views)
                for lv in it["levels"]:
                    if not os.path.exists(lv["file"]):
                        rec["lv"].append({"level": lv["level"], "mode": lv.get("mode"), "missing": True})
                        continue
                    p = lodimage.load_glb(lv["file"], cache=False)
                    m = holes.measure_prims(H, p, lv["level"])
                    r = holes.brief(m)
                    r.update(level=lv["level"], mode=lv.get("mode"), tris=lodimage.tris_of(p), gate=holes.ok(lv["level"], m))
                    rec["lv"].append(r)
            except Exception as e:      # noqa: BLE001
                import traceback
                rec["error"] = f"{e}\n{traceback.format_exc()}"
            rec["s"] = round(time.time() - t0, 2)
            f.write(json.dumps(rec) + "\n")
            f.flush()
    print("HOLES done", len(items))


def items_for(keys, base):
    town = json.load(open(os.path.join(ROOT, "out", "town", "town.json")))["protos"]
    out = []
    for k in keys:
        a, v = k.split("/")
        side = os.path.join(base, a, v + ".json")
        if not os.path.exists(side):
            side = os.path.join(ROOT, "out", "glb", a, v + ".json")
        j = json.load(open(side)) if os.path.exists(side) else {}
        mount = (town.get(k) or {}).get("mount") or (j.get("meta") or {}).get("mount") or "ground"
        lods = (j.get("stats") or {}).get("lods") or [{}, {}]
        out.append({"key": k, "mount": mount, "glb0": os.path.join(base, a, v + ".glb"),
                    "levels": [{"level": n, "file": os.path.join(base, a, f"{v}.lod{n}.glb"),
                                "mode": (lods[n - 1] if len(lods) >= n else {}).get("mode") or "decimated"} for n in (1, 2)]})
    return out


def report(recs, old=None, top=20):
    rows = [(r["key"], dict(lv, n=lv.get("n", r.get("n", 0)))) for r in recs for lv in r.get("lv", []) if not lv.get("missing")]
    err = [r["key"] for r in recs if r.get("error")]
    print(f"{len(recs)} variants, {len(rows)} levels measured" + (f", {len(err)} failed: {err[:6]}" if err else ""))
    print("  mode        level   levels   with holes   over the gate   mean share   worst share")
    for mode in ("reduced", "decimated", "authored"):
        for n in (1, 2):
            s = [lv for _, lv in rows if lv["mode"] == mode and lv["level"] == n and lv.get("n", 0) >= 400]
            if s:
                print(f"  {mode:<10}  lod {n}   {len(s):>6}   {sum(1 for x in s if x['count'] > 0):>10}   {sum(1 for x in s if x['gate']):>13}   "
                      f"{100 * sum(x['holes'] for x in s) / len(s):>8.3f} %   {100 * max(x['holes'] for x in s):>9.3f} %")
    thin = sum(1 for _, x in rows if x.get("n", 0) < 400)
    if thin:
        print(f"  not judged (fewer than 400 samples: next to no closed surface): {thin} levels")
    rows = [(k, x) for k, x in rows if x.get("n", 0) >= 400]
    print(f"  all: {len(rows)} levels, {sum(1 for _, x in rows if x['count'] > 0)} with any hole sample, "
          f"{sum(1 for _, x in rows if x['gate'])} over the gate; variants with a level over the gate: "
          f"{len({k for k, x in rows if x['gate']})} of {len(recs)}")
    print(f"  worst {top}:")
    prev = {(r["key"], lv["level"]): lv for r in (old or []) for lv in r.get("lv", []) if not lv.get("missing")}
    for k, x in sorted(rows, key=lambda kv: -kv[1]["holes"])[:top]:
        w = (x.get("where") or [{}])[0]
        was = prev.get((k, x["level"]))
        print(f"    {k:<44} lod {x['level']} {x['mode']:<9} {100 * x['holes']:.3f} % (back {100 * x['back']:.3f}, none {100 * x['none']:.3f}; "
              f"worst view {100 * x['worst']:.2f} %) {x['tris']} tris" + (f"  was {100 * was['holes']:.3f} %" if was else "")
              + (f"  at {w.get('at')} {w.get('mat')}" if w else ""))


def main(argv):
    pos = [a for a in argv if not a.startswith("--")]
    kw = {}
    i = 0
    args = []
    while i < len(argv):
        if argv[i].startswith("--"):
            kw[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    pos = args
    if not pos:
        print(__doc__)
        return 1
    if pos[0] == "report":
        recs = [json.loads(l) for l in open(pos[1])]
        old = [json.loads(l) for l in open(pos[2])] if len(pos) > 2 else None
        report(recs, old, int(kw.get("top", 20)))
        return 0
    base = os.path.abspath(kw.get("dir", os.path.join(ROOT, "out", "glb")))
    if pos[0] == "town":
        keys = list(json.load(open(os.path.join(ROOT, "out", "town", "town.json")))["protos"].keys())
        if kw.get("only"):
            want = set(kw["only"].split(","))
            keys = [k for k in keys if k.split("/")[0] in want or k in want]
    else:
        keys = pos
    items = [it for it in items_for(keys, base) if os.path.exists(it["glb0"])]
    if kw.get("phase"):
        for it in items:
            it["phase"] = float(kw["phase"])        # viewers elsewhere on their rings than the build's (0.37, 0.87)
    if len(items) < len(keys):
        print(f"{len(keys) - len(items)} variants have no LOD 0 under {base} (skipped)")
    out = os.path.abspath(kw.get("out", os.path.join(ROOT, ".scratch", "holes", "holes.json")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jobs = max(1, min(int(kw.get("jobs", 2)), len(items) or 1))
    import subprocess
    items.sort(key=lambda it: -os.path.getsize(it["glb0"]))
    procs = []
    for j in range(jobs):
        part = items[j::jobs]
        lf, of = f"{out}.{j}.list", f"{out}.{j}.part"
        json.dump(part, open(lf, "w"))
        procs.append((subprocess.Popen([BLENDER, "--background", "--factory-startup", "--python", os.path.abspath(__file__), "--",
                                        "--worker", lf, of], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), lf, of))
    recs = []
    for p, lf, of in procs:
        o, _ = p.communicate()
        if "HOLES done" not in o:
            print("worker failed:\n" + "\n".join(o.splitlines()[-25:]))
        if os.path.exists(of):
            recs += [json.loads(l) for l in open(of)]
            os.remove(of)
        os.remove(lf)
    order = {k: i for i, k in enumerate(keys)}
    recs.sort(key=lambda r: order.get(r["key"], 0))
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    report(recs, top=int(kw.get("top", 20)))
    print("written:", out)
    return 0


if IN_BLENDER:
    a = sys.argv[sys.argv.index("--") + 1:]
    worker(a[1], a[2])
else:
    if __name__ == "__main__":
        sys.exit(main(sys.argv[1:]))
