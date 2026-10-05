#!/usr/bin/env python3
"""Measure the candidates of the cheaper levels (lod 1 / 2) of assets: reduced (tk.reduce), decimated
(tk.final._decimated_lods), and modelled where the asset has META["lod"]; score them on images (tk.lodimage)
and say which one ships (tk.final_levels.pick(), which is tk.lodimage.choose() plus the priority of a modelled
level - exactly what tools/bl_build.py --lods 1 does).

    python3 tools/lod_reduce_eval.py all                      # every asset, default variant
    python3 tools/lod_reduce_eval.py barrel wall_stone_run:default,brick_garden hotel:all
    python3 tools/lod_reduce_eval.py all --jobs 3 --out out/lod_eval --tag run1

Writes, per asset and variant, under out/lod_eval/<id>/ (never into out/glb or out/lod):
    <variant>.glb / .json            lod 0 and its sidecar (with "lod": the verdict per level, "lod_eval": everything)
    <variant>.lod1.glb / .lod2.glb   what ships
    <variant>.dec1.glb / .dec2.glb   the decimated copies
    <variant>.red.lod1.glb / .red.lod2.glb [/ .red15.lod2.glb / .red20.lod2.glb / .red..np.lod2.glb]
                                     the reduced levels (tk.reduce.cand_file; red15 / red20: over the lod-2 budget,
                                     inside the 1.5 x / 2.0 x allowance; .red.lod2 is then the same level forced
                                     into the budget; np: the chain without the flat-patch plan)
    <variant>.auth.lod1.glb / .auth.lod2.glb                     the modelled levels (assets with META["lod"])
and out/lod_eval/summary[.<tag>].json with one record per asset/variant (merged into an existing file).

It starts one Blender per asset (at most --jobs at once and never more than MAX_BLENDER Blender processes on the machine),
with PYTHONHASHSEED=0. Inside Blender the same file is the worker:
    blender --background --factory-startup --python-use-system-env --python tools/lod_reduce_eval.py -- \
        --worker <id> --variants a,b --out <dir>
"""
import sys
import os
import shutil
import json
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER = os.environ.get("TSQ_BLENDER") or os.environ.get("BLENDER") or shutil.which("blender") or os.path.expanduser("~/.local/bin/blender")
MAX_BLENDER = 4

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False


# ----------------------------------------------------------------------------
# Worker (inside Blender)
# ----------------------------------------------------------------------------

def eval_one(mod, asset_id, vname, out_dir):
    import shutil
    import numpy as np  # noqa: F401
    from tk import core, asm, final, levels, final_levels, reduce, lodimage
    T = {}
    t0 = time.time()
    core.clear_scene()
    params = asm.variant_params(mod, vname, None)
    core._reset_state(params["seed"])
    rng = core.Rng(f"{asset_id}/{params['seed']}")
    meta = dict(getattr(mod, "META", {}))
    authored = bool(meta.get("lod"))
    glb = os.path.join(out_dir, asset_id, vname + ".glb")
    stem = os.path.splitext(glb)[0]
    keep = {}
    rec0 = None
    if authored:
        levels._begin(0)
    try:
        mod.build(core.P(params), rng)
        bt = time.time() - t0
        c0 = time.process_time()
        snap = reduce.snapshot()
        T["snapshot_s"] = round(time.process_time() - c0, 2)
        info = final.finalize(asset_id, vname, params, meta, glb, bt, lods=False, _keep=keep)
    finally:
        if authored:
            rec0 = levels._end(rng)
    T["build_s"] = round(time.time() - t0, 2)
    c0 = time.process_time()
    ref = final_levels.reference(info, keep)
    dec = final._decimated_lods(keep["merged"], keep["tris"], keep["diag"], glb, keep["kw"], info["warnings"], suffix="dec",
                                measure=(lambda li, made: lodimage.text_measure(li, ref["text"], final_levels.text_tris(made)))
                                if ref["text"] else None)
    T["dec_s"] = round(time.process_time() - c0, 2)
    mount = params.get("mount") if params.get("mount") in ("hang", "wall", "ground", "wall_ground") else meta.get("mount", "ground")
    cands = {1: [], 2: []}
    logs = {}
    T["reduce_s"] = 0.0
    T["finlevel_s"] = 0.0
    for level in (1, 2):
        bud = final_levels.budget(level, ref["tris"])
        cands[level].append(final_levels.candidate("decimated", "decimated", None, f"{stem}.dec{level}.glb", budget=bud,
                                                   tris=dec[level - 1]["tris"], glb_bytes=dec[level - 1]["glb_bytes"],
                                                   text=dec[level - 1].get("text")))
        for name in reduce.CAND:
            if os.path.exists(reduce.cand_file(stem, name, level)):
                os.remove(reduce.cand_file(stem, name, level))
        try:
            cs, logs[level] = reduce.level_candidates(level, snap, ref, meta, params, stem, T)
            cands[level] += cs
        except Exception as e:
            logs.setdefault(level, {})["error"] = f"{type(e).__name__}: {e}"
            logs[level]["trace"] = traceback.format_exc()
    T["reduce_s"] = round(T["reduce_s"], 2)
    T["finlevel_s"] = round(T["finlevel_s"], 2)
    T["vis_s"] = (ref.get("_red") or {}).get("vis_s")
    reduce.release(snap)

    if authored:
        c0 = time.time()
        for level in (1, 2):
            bud = final_levels.budget(level, ref["tris"])
            saved = asm._MODULES
            asm._MODULES = {}
            rng2 = None
            try:
                m2 = asm.load_asset(asset_id)
                core.clear_scene()
                core._reset_state(params["seed"])
                rng2 = core.Rng(f"{asset_id}/{params['seed']}")
                levels._begin(level, rec0, False, where=False, budget=bud)
                m2.build(core.P(params), rng2)
                r = final_levels.finalize_level(level, ref, stem + ".auth.glb", params, meta, rng2, rec0, 0.0, True)
                cands[level].append(final_levels.candidate("authored", "authored", r, f"{stem}.auth.lod{level}.glb"))
            except Exception as e:
                logs.setdefault(f"auth{level}", {})["error"] = f"{type(e).__name__}: {e}"
            finally:
                levels._end(rng2)
                asm._MODULES = saved
        T["auth_s"] = round(time.time() - c0, 2)

    # ---- images and the choice ----
    M = lodimage.Metric(glb, mount, sockets=info.get("sockets"))
    picks = {}
    lods, lod = [], []
    for level in (1, 2):
        for c in cands[level]:
            c["m"] = M.score(level, c["file"]) if os.path.exists(c["file"]) else None
            if c["m"] is not None and c.get("text") is not None:
                c["m"]["text"] = c["text"]
            if c["m"] is not None and c.get("open") is not None:
                c["m"]["open"] = c["open"]
        pk = final_levels.pick(level, cands[level])       # = lodimage.choose() unless there is a modelled level
        if authored:
            pk["choose"] = lodimage.choose(level, cands[level])
        picks[level] = pk
        win = next(c for c in cands[level] if c["name"] == pk["winner"])
        shutil.copyfile(win["file"], f"{stem}.lod{level}.glb")
        lods.append({"tris": win["tris"], "glb_bytes": win["glb_bytes"], "mode": win["kind"]})
        others = "; ".join(f"{k}: {' | '.join(v)}" for k, v in pk["why"].items() if v)
        lod.append({"level": level, "ok": win["kind"] != "decimated", "tris": win["tris"], "budget": win["budget"],
                    "ratio": round(win["tris"] / max(ref["tris"], 1), 4), "mode": win["kind"], "warn": [],
                    "why": ([f"{win['name']} ships"] if win["kind"] != "decimated" else [f"decimated ships - {others}"])})
    T["metric_s"] = round(M.seconds, 2)
    info["stats"]["lods"] = lods
    info["stats"]["dec"] = dec
    info["lod"] = lod
    rec = {"id": asset_id, "variant": vname, "authored": authored, "mount": mount, "tris0": ref["tris"],
           "diag": round(ref["diag"], 3), "times": T,
           "levels": {str(n): {"budget": final_levels.budget(n, ref["tris"]), "pick": picks[n], "log": logs.get(n),
                               "cand": {c["name"]: {"tris": c["tris"], "hard": c["hard"], "m": c["m"],
                                                    "file": os.path.basename(c["file"]),
                                                    "relaxed": bool(c.get("relaxed")),
                                                    "ident": bool(c.get("ident")),
                                                    "abs": lodimage.gates(n, c["m"], None) if c["m"] else None,
                                                    "why": (c.get("res") or {}).get("why"),
                                                    "warn": (c.get("res") or {}).get("warn"),
                                                    "cover": (c.get("res") or {}).get("cover")} for c in cands[n]}}
                      for n in (1, 2)},
           "errors": {str(k): v for k, v in logs.items() if isinstance(v, dict) and v.get("error")}}
    info["lod_eval"] = rec
    with open(stem + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return rec


def remeasure_one(mod, asset_id, vname, out_dir):
    """Nothing is built: the candidates already in out_dir are scored again with the metric as it is now and the
    choice is made again. -> the record, with the choice it had in "pick0" (no file is written)."""
    from tk import final_levels, reduce, lodimage
    stem = os.path.join(out_dir, asset_id, vname)
    with open(stem + ".json") as f:
        info = json.load(f)
    rec = info["lod_eval"]
    meta = dict(getattr(mod, "META", {}))
    M = lodimage.Metric(stem + ".glb", rec["mount"], sockets=info.get("sockets"))
    for n in (1, 2):
        L = rec["levels"][str(n)]
        L.pop("log", None)
        cands = []
        for name, c in L["cand"].items():
            f = os.path.join(out_dir, asset_id, c["file"])
            old = c.get("m") or {}
            m = M.score(n, f) if os.path.exists(f) else None
            if m is not None:
                for k in ("text", "open"):
                    if old.get(k) is not None:
                        m[k] = old[k]
            c["m"] = m
            c["abs"] = lodimage.gates(n, m, None) if m else None
            kind = "decimated" if name == "decimated" else "authored" if name == "authored" else "reduced"
            cands.append({"name": name, "kind": kind, "tris": c["tris"], "budget": L["budget"], "hard": c.get("hard") or [],
                          "m": m, "file": f, "allow": reduce.allowance(n, meta) if kind == "reduced" else None,
                          "ident": bool(c.get("ident"))})
        L["pick0"] = L["pick"]
        L["pick"] = final_levels.pick(n, cands)
    rec["times"] = {"metric_s": round(M.seconds, 2)}
    return rec


def worker(argv):
    sys.path.insert(0, ROOT)
    from tk import asm
    a = {"worker": None, "variants": "default", "out": os.path.join(ROOT, "out", "lod_eval")}
    i = 0
    while i < len(argv):
        a[argv[i].lstrip("-")] = argv[i + 1]
        i += 2
    out = os.path.abspath(a["out"])
    for bad in (os.path.join(ROOT, "out", "glb"), os.path.join(ROOT, "out", "lod")):
        if out == bad or out.startswith(bad + os.sep):
            raise SystemExit("lod_reduce_eval never writes into out/glb or out/lod")
    aid = a["worker"]
    try:
        mod = asm.load_asset(aid)
    except Exception:
        print("EVAL_RESULT " + json.dumps({"id": aid, "variant": "*", "error": traceback.format_exc()}))
        return
    names = a["variants"].split(",")
    if names == ["all"]:
        names = list(getattr(mod, "VARIANTS", {}) or {})
        if "default" not in names:
            names = ["default"] + names
    for v in names:
        t0 = time.time()
        try:
            rec = remeasure_one(mod, aid, v, out) if a.get("remeasure") else eval_one(mod, aid, v, out)
            rec["seconds"] = round(time.time() - t0, 1)
        except Exception:
            rec = {"id": aid, "variant": v, "error": traceback.format_exc()}
        print("EVAL_RESULT " + json.dumps(rec))
        sys.stdout.flush()


# ----------------------------------------------------------------------------
# Parent (plain python)
# ----------------------------------------------------------------------------

def _nblender():
    n = 0
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                with open(f"/proc/{p}/cmdline", "rb") as f:
                    a = f.read().split(b"\0")
            except Exception:
                continue
            if a and a[0].endswith(b"blender") and b"--background" in a:
                n += 1
    return n


def _all_ids():
    ids = []
    for cat in sorted(os.listdir(os.path.join(ROOT, "assets"))):
        d = os.path.join(ROOT, "assets", cat)
        if os.path.isdir(d) and not cat.startswith("_"):
            ids += [f[:-3] for f in sorted(os.listdir(d)) if f.endswith(".py") and not f.startswith("_")]
    return ids


def summarise(recs):
    """counts per level over the records -> dict"""
    out = {}
    for n in ("1", "2"):
        kinds, tris, tris0, imgs, allow = {}, 0, 0, [], 0
        for r in recs:
            if r.get("error"):
                continue
            L = r["levels"][n]
            pk = L["pick"]
            kinds[pk["kind"]] = kinds.get(pk["kind"], 0) + 1
            win = L["cand"][pk["winner"]]
            tris += win["tris"]
            tris0 += r["tris0"]
            allow += bool(pk["allowance"])
            if pk["img"] is not None:
                imgs.append(pk["img"])
        imgs.sort()
        out[n] = {"ships": kinds, "tris": tris, "tris_lod0": tris0, "allowance": allow,
                  "img_median": imgs[len(imgs) // 2] if imgs else None, "img_worst": imgs[-1] if imgs else None}
    return out


def parent(argv):
    import subprocess
    opt = {"jobs": "3", "out": os.path.join(ROOT, "out", "lod_eval"), "tag": ""}
    specs = []
    i = 0
    while i < len(argv):
        if argv[i].startswith("--"):
            opt[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            specs.append(argv[i])
            i += 1
    todo = []
    for s in specs:
        aid, _, var = s.partition(":")
        for a in (_all_ids() if aid == "all" else [aid]):
            todo.append((a, var or "default"))
    out = os.path.abspath(opt["out"])
    os.makedirs(os.path.join(out, "_log"), exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED="0")
    jobs = int(opt["jobs"])
    run, recs = [], []
    t0 = time.time()
    while todo or run:
        while todo and len(run) < jobs and _nblender() < MAX_BLENDER:
            aid, var = todo.pop(0)
            lp = os.path.join(out, "_log", aid + ".log")
            log = open(lp, "w")
            cmd = [BLENDER, "--background", "--factory-startup", "--python-use-system-env", "--python", os.path.abspath(__file__),
                   "--", "--worker", aid, "--variants", var, "--out", out] + (["--remeasure", "1"] if opt.get("remeasure") else [])
            run.append((aid, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env), log, lp))
            time.sleep(0.3)
        time.sleep(0.4)
        for r in list(run):
            if r[1].poll() is None:
                continue
            r[2].close()
            run.remove(r)
            got = 0
            with open(r[3], errors="replace") as f:
                for line in f:
                    if line.startswith("EVAL_RESULT "):
                        recs.append(json.loads(line[12:]))
                        got += 1
            if not got:
                recs.append({"id": r[0], "variant": "*", "error": f"no result (see {r[3]})"})
            last = recs[-1]
            if last.get("error"):
                print(f"{r[0]:26s} ERROR {last['error'].strip().splitlines()[-1][:120]}")
            else:
                for rec in recs[-got:]:
                    if rec.get("error"):
                        print(f"{rec['id']}:{rec['variant']} ERROR {rec['error'].strip().splitlines()[-1][:120]}")
                        continue
                    s = []
                    for n in ("1", "2"):
                        L = rec["levels"][n]
                        pk = L["pick"]
                        s.append(f"lod{n} {pk['kind'][:3]}{'+' if pk['allowance'] else ' '} {L['cand'][pk['winner']]['tris']:6d}"
                                 f" img {pk['img'] if pk['img'] is not None else -1:.4f}")
                    print(f"{rec['id'] + ':' + rec['variant']:40s} {rec['tris0']:7d}  " + "  ".join(s) + f"  {rec.get('seconds', 0):5.1f}s")
            sys.stdout.flush()
    sp = os.path.join(out, f"summary{'.' + opt['tag'] if opt['tag'] else ''}.json")
    allr = {}
    if os.path.exists(sp):
        try:
            with open(sp) as f:
                allr = json.load(f).get("records", {})
        except Exception:
            allr = {}
    for r in recs:
        allr[f"{r['id']}/{r['variant']}"] = r
    with open(sp, "w") as f:
        json.dump({"totals": summarise(list(allr.values())), "records": allr}, f, indent=1)
    print(f"{len(recs)} records in {time.time() - t0:.0f} s -> {sp}")


if IN_BLENDER:
    worker(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
elif __name__ == "__main__":
    parent(sys.argv[1:])
