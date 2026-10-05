"""Blender-side build runner.

    blender --background --factory-startup --python tools/bl_build.py -- \
        --asset <id> [--variants all|a,b,c] [--params '{"k": v}'] [--name custom] [--out out/glb]
        [--lods 0|1|cmp] [--auto 1]

Builds each requested variant of assets/*/<id>.py and writes out/glb/<id>/<variant>.glb + .json.
Prints one line `TSQ_RESULT {json}` per variant.

--lods 1   also writes <variant>.lod1.glb / .lod2.glb. Per level there are up to three candidates and the one that
           looks most like lod 0 at the distance where the level takes over ships (tk.final_levels.pick, on the
           pictures of tk.lodimage); the sidecar says which in stats.lods[].mode:
             authored   a level-aware asset (META["lod"] = True, docs/LOD.md) is built once more per level with
                        tk.lod() = 1 / 2. A level that keeps its contract ships (with a warning when its picture
                        misses a gate that another candidate passes); one that breaks it (budget, sockets, parts,
                        glow, bounds, random stream) is replaced by the decimated copy, with a warning.
                        META["lod"] = "far": only lod 2 is modelled (build() once more at tk.lod() = 2); lod 0 and
                        lod 1 are made exactly as for an asset without META["lod"] (same files, byte for byte).
             reduced    tk.reduce: the full model with what nobody sees taken out first (every asset). A lod 2 may
                        end over its budget (up to 1.5 x, buildings and landmarks 2.0 x: lod_pick[].over) when that
                        is what keeps it whole; lettering and glowing faces are kept as they are (docs/LOD.md).
             decimated  tk.final._decimated_lods: the plain collapse (every asset; the fallback)
           stats.lods[] also carries what the gates measured on the level that ships: "text" (lettering against
           lod 0) and "zfight" (m2 of glowing faces with another surface closer than 3 mm, against lod 0) - absent
           when the asset has no tagged lettering / nothing that glows.
--lods cmp comparison build for `tools/tsq lod`: the modelled levels (kept even when rejected; without META["lod"]
           the reduced ones, kept even when the decimated copy ships) as .lod1 / .lod2, the decimated ones next to
           them as .dec1 / .dec2, the reduced ones as .red.lod1 / .red.lod2.
--auto 1   treat the asset as level-aware even without META["lod"] (shows what the toolkit's own scaling gives).
"""
import sys
import os
import json
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import tk  # noqa: E402
from tk import core, asm, final  # noqa: E402
from tk import levels, final_levels  # noqa: E402
from tk import reduce, lodimage  # noqa: E402


def parse(argv):
    a = {"asset": None, "variants": "all", "params": None, "name": None, "out": os.path.join(ROOT, "out", "glb"),
         "jobs": None, "lods": "0", "auto": "0"}
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        a[k] = argv[i + 1]
        i += 2
    return a


def build_one(mod, asset_id, vname, overrides, out_dir, base=None, lods=False):
    t0 = time.time()
    core.clear_scene()
    params = asm.variant_params(mod, base if base is not None else (vname if overrides is None else None), overrides)
    core._reset_state(params["seed"])
    rng = core.Rng(f"{asset_id}/{params['seed']}")
    mod.build(core.P(params), rng)
    bt = time.time() - t0
    glb = os.path.join(out_dir, asset_id, vname + ".glb")
    info = final.finalize(asset_id, vname, params, dict(getattr(mod, "META", {})), glb, bt, lods=lods)
    return info


def _level_run(asset_id, level, rec0, ref, params, meta, glb, keep_rejected, auto=False):
    """One more build() of the asset at tk.lod() == level, from fresh module instances (asset files may cache
    things per build at module level) -> result dict of tk.final_levels.finalize_level."""
    t0 = time.time()
    saved = asm._MODULES
    asm._MODULES = {}
    rng = None
    try:
        mod = asm.load_asset(asset_id)
        core.clear_scene()
        core._reset_state(params["seed"])
        rng = core.Rng(f"{asset_id}/{params['seed']}")
        levels._begin(level, rec0, auto, where=keep_rejected, budget=final_levels.budget(level, ref["tris"], meta))
        mod.build(core.P(params), rng)
        bt = time.time() - t0
        where = final_levels.breakdown() if keep_rejected else None
        res = final_levels.finalize_level(level, ref, glb, params, meta, rng, rec0, bt, keep_rejected)
        if where is not None:
            res["where"] = where
    except Exception as e:
        res = {"level": level, "ok": False, "why": [f"build failed at lod {level}: {type(e).__name__}: {e}"], "warn": [],
               "tris": 0, "glb_bytes": 0, "ratio": 0.0, "budget": final_levels.budget(level, ref["tris"], meta),
               "trace": traceback.format_exc(), "seconds": round(time.time() - t0, 2)}
        stale = f"{os.path.splitext(glb)[0]}.lod{level}.glb"
        if keep_rejected and os.path.exists(stale):      # (tools/tsq lod: no sheet of an older build's level)
            os.remove(stale)
    finally:
        levels._end(rng)
        asm._MODULES = saved
    return res


def _record_run(asset_id, params, where=False):
    """META["lod"] = "far": lod 0 and lod 1 were made by the plain build (exactly as without META["lod"]); this is
    one more build() at level 0 with the toolkit's stream ids recorded, for the lod 2 run to replay. Its meshes are
    thrown away. -> (record, {source object: triangles} | None)"""
    saved = asm._MODULES
    asm._MODULES = {}
    rng = None
    w = None
    began = False
    try:
        mod = asm.load_asset(asset_id)
        core.clear_scene()
        core._reset_state(params["seed"])
        rng = core.Rng(f"{asset_id}/{params['seed']}")
        levels._begin(0, where=where)
        began = True
        mod.build(core.P(params), rng)
        if where:
            w = final_levels.breakdown()
    finally:
        rec0 = levels._end(rng) if began else None
        asm._MODULES = saved
    return rec0, w


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _copy(src, dst):
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(fi.read())


def _holes_of(c):
    """the sidecar's numbers of "no holes" (docs/LOD.md) for a candidate: "holes" - the share of the street views'
    samples that look through the level, "holes_worst" - in the worst single view, "closed" - the triangles put
    back to close them (not held against the budget), "holes_at" - where the rest is"""
    h = ((c or {}).get("m") or {}).get("holes")
    out = {}
    if h:
        out.update(holes=h["holes"], holes_worst=h["worst"])
        if h.get("why") and h.get("where"):
            out["holes_at"] = h["where"][:3]
    if (c or {}).get("closed"):
        out["closed"] = int(c["closed"])
    return out


def build_levels(mod, asset_id, vname, overrides, out_dir, base=None, cmp=False, auto=False):
    """Build with the cheaper levels: level 0 exactly as build_one, then per level the candidates - the modelled
    level (build() again at tk.lod() == 1 / 2: assets with META["lod"], or auto), the reduced copy (tk.reduce, from
    the un-merged full model) and the decimated copy - and tk.final_levels.pick() says which one ships."""
    t0 = time.time()
    core.clear_scene()
    params = asm.variant_params(mod, base if base is not None else (vname if overrides is None else None), overrides)
    core._reset_state(params["seed"])
    rng = core.Rng(f"{asset_id}/{params['seed']}")
    meta = dict(getattr(mod, "META", {}))
    far_only = meta.get("lod") == "far" and not auto    # lod 2 modelled; lod 0 and lod 1 as without META["lod"]
    modelled = (bool(meta.get("lod")) or bool(auto)) and not far_only
    auth = (1, 2) if modelled else (2,) if far_only else ()
    glb = os.path.join(out_dir, asset_id, vname + ".glb")
    stem = os.path.splitext(glb)[0]
    keep, notes, T = {}, [], {}
    snap = rec0 = where = None
    # a modelled level that keeps its contract ships and a broken one gives way to the decimated copy: the reduced
    # copy of a modelled asset is only built where it is shown (cmp) or where the switches let it ship
    want_reduced = not modelled or cmp or final_levels.AUTHORED_YIELDS or final_levels.REJECTED_SHIPS != "decimated"
    if modelled:
        levels._begin(0, where=cmp)     # records the toolkit's stream ids; the ids themselves are the usual ones
    try:
        mod.build(core.P(params), rng)
        bt = time.time() - t0
        if modelled and cmp:
            where = final_levels.breakdown()
        if want_reduced:
            try:
                snap = reduce.snapshot()    # the un-merged full model, for the reducer (finalize() merges it)
            except Exception as e:
                notes.append(f"no reduced levels: snapshot failed ({type(e).__name__}: {e})")
        info = final.finalize(asset_id, vname, params, meta, glb, bt, lods=False, _keep=keep)
    finally:
        if modelled:
            rec0 = levels._end(rng)
    t1 = time.time()
    t2 = time.time()
    ref = final_levels.reference(info, keep)
    ref_s = round(time.time() - t2, 2)
    c0 = time.process_time()
    mount = params.get("mount") if params.get("mount") in ("hang", "wall", "ground", "wall_ground") else meta.get("mount", "ground")
    dec = final._decimated_lods(keep["merged"], keep["tris"], keep["diag"], glb, keep["kw"], info["warnings"], suffix="dec",
                                measure=(lambda li, made: lodimage.text_measure(li, ref["text"], final_levels.text_tris(made)))
                                if ref["text"] else None, mount=mount)
    T["dec_s"] = time.process_time() - c0

    # ---- the candidates: decimated, reduced (before the modelled levels - their builds clear the scene), modelled
    cands = {1: [], 2: []}
    for n in (1, 2):
        cands[n].append(final_levels.candidate("decimated", "decimated", None, f"{stem}.dec{n}.glb", tris=dec[n - 1]["tris"],
                                               budget=final_levels.budget(n, ref["tris"]), glb_bytes=dec[n - 1]["glb_bytes"],
                                               text=dec[n - 1].get("text"), closed=dec[n - 1].get("closed", 0)))
        for name in reduce.CAND:
            _remove(reduce.cand_file(stem, name, n))
        if snap is not None:
            try:
                cands[n] += reduce.level_candidates(n, snap, ref, meta, params, stem, T)[0]
            except Exception as e:
                notes.append(f"lod {n}: no reduced level ({type(e).__name__}: {e})")
    if snap is not None:
        reduce.release(snap)
    res = {}
    if far_only:
        try:
            rec0, where = _record_run(asset_id, params, cmp)
        except Exception as e:
            auth = ()
            notes.append(f"lod 2: no modelled level (the recording build failed: {type(e).__name__}: {e})")
    if auth:
        c0 = time.time()
        for n in auth:
            res[n] = _level_run(asset_id, n, rec0, ref, params, meta, glb, cmp, auto)
        T["auth_s"] = time.time() - c0
        for r in res.values():
            wrote = bool(r["glb_bytes"]) and (r["ok"] or cmp)
            cands[r["level"]].append(final_levels.candidate("authored", "authored", r, f"{stem}.lod{r['level']}.glb" if wrote else None))

    # ---- their pictures, and what ships
    c0 = time.process_time()
    picks = {}
    try:
        M = lodimage.Metric(glb, mount, sockets=info.get("sockets"))
        for n in (1, 2):
            for c in cands[n]:
                c["m"] = M.score(n, c["file"]) if c["file"] and os.path.exists(c["file"]) else None
                if c["m"] is not None and c.get("text") is not None:
                    c["m"]["text"] = c["text"]          # lettering against lod 0 (tk.lodimage.text_measure)
                if c["m"] is not None and c.get("open") is not None:
                    c["m"]["open"] = c["open"]          # share of the tiled fields laid bare (tk.reduce._open)
            picks[n] = final_levels.pick(n, cands[n])
    except Exception as e:      # never lose a build over the pictures: the rule from before them
        notes.append(f"levels were chosen without their pictures ({type(e).__name__}: {e})")
        for n in (1, 2):
            a = next((c for c in cands[n] if c["kind"] == "authored" and c["res"]["ok"]), None)
            w = a or cands[n][0]
            picks[n] = {"level": n, "winner": w["name"], "kind": w["kind"], "allowance": False, "over": 0, "why": {},
                        "clean": False, "img": None, "kept": None, "better": None}
    T["metric_s"] = time.process_time() - c0
    lod_s = round(time.time() - t1, 2)
    del ref

    lods, lod = [], []
    for n in (1, 2):
        pk = picks[n]
        by = {c["name"]: c for c in cands[n]}
        win = by[pk["winner"]]
        dst = f"{stem}.lod{n}.glb"
        r = res.get(n)
        lost = lambda name: " | ".join(pk["why"].get(name) or [])       # noqa: E731
        gated = {k: (win.get("m") or {}).get(v) for k, v in (("text", "text"), ("zfight", "zf")) if (win.get("m") or {}).get(v)}
        gated.update(_holes_of(win))
        hw = ((win.get("m") or {}).get("holes") or {}).get("why")
        if hw and not (pk["kind"] == "authored" and pk.get("kept")):
            info["warnings"].append(f"lod {n}: " + hw[0])
        if r is not None:
            for w in r.get("warn", []):
                info["warnings"].append(w if w.startswith(f"lod {n}") else f"lod {n}: {w}")
            if not r["ok"]:
                info["warnings"].append(f"lod {n}: modelled level REJECTED - " + " | ".join(r["why"])
                                        + ("" if cmp else f" -> the {pk['kind']} copy ships instead"))
            elif pk["kind"] != "authored":      # (tk.final_levels.AUTHORED_YIELDS)
                info["warnings"].append(f"lod {n}: the {pk['kind']} copy {'would ship' if cmp else 'ships'} instead of the modelled "
                                        f"level - {lost('authored')}")
            elif pk.get("kept"):
                b = by.get(pk.get("better"))
                info["warnings"].append(f"lod {n}: " + " | ".join(pk["kept"]) + ("" if b is None else
                                        f" - the {b['kind']} copy passes (image difference {b['m']['img']:.4f} against {pk['img']:.4f})"))
            lod.append(dict({k: v for k, v in r.items() if k not in ("trace", "where")}, ships=pk["kind"]))
        if cmp and r is not None:
            am = (by.get("authored") or {}).get("m") or {}
            lods.append(dict({"tris": r["tris"], "glb_bytes": r["glb_bytes"], "mode": "authored" if r["ok"] else "rejected"},
                             **{k: am[v] for k, v in (("text", "text"), ("zfight", "zf")) if am.get(v)}, **_holes_of(by.get("authored"))))
        elif cmp:
            # tools/tsq lod of an asset without levels of its own: the reduced copy is the 'new' level of the sheet
            red = win if pk["kind"] == "reduced" else by.get("reduced")
            ok = pk["kind"] == "reduced"
            if red is None:
                why = [x for x in notes if x.startswith((f"lod {n}:", "no reduced"))] or ["no reduced level"]
                lod.append({"level": n, "ok": False, "why": why, "warn": [], "tris": win["tris"], "glb_bytes": win["glb_bytes"],
                            "budget": win["budget"], "ships": pk["kind"]})
                lods.append(dict({"tris": win["tris"], "glb_bytes": win["glb_bytes"], "mode": pk["kind"]}, **gated))
                _copy(win["file"], dst)
            else:
                rr = red["res"]
                lod.append(dict({k: v for k, v in rr.items() if k not in ("trace", "where")}, ok=ok, ships=pk["kind"],
                                why=[] if ok else ([lost(red["name"])] if lost(red["name"]) else ["the decimated copy is the better picture"])))
                lods.append(dict({"tris": red["tris"], "glb_bytes": red["glb_bytes"], "mode": "reduced" if ok else "rejected"},
                                 **{k: (red.get("m") or {}).get(v) for k, v in (("text", "text"), ("zfight", "zf")) if (red.get("m") or {}).get(v)},
                                 **_holes_of(red)))
                _copy(red["file"], dst)
        else:
            if win["file"] != dst:
                os.replace(win["file"], dst)
            if win["kind"] == "reduced":
                for w in win["res"].get("warn", []):
                    info["warnings"].append(w if w.startswith(f"lod {n}") else f"lod {n}: {w}")
            lods.append(dict({"tris": win["tris"], "glb_bytes": win["glb_bytes"], "mode": win["kind"]}, **gated))
            for c in cands[n]:
                if c["kind"] != "authored" and c["file"]:
                    _remove(c["file"])
    for x in notes:
        info["warnings"].append(x if x.startswith("lod ") else f"lod: {x}")
    info["stats"]["lods"] = lods
    info["stats"]["lod_s"] = lod_s                # levels 1 + 2: every candidate, the reference (ref_s) and the pictures
    info["stats"]["lod_ref_s"] = ref_s
    info["stats"]["lod_cpu"] = {k: round(v, 2) for k, v in T.items()}
    info["stats"]["hash_pinned"] = not sys.flags.hash_randomization     # PYTHONHASHSEED reached Blender (tools/tsq, town.py)
    if cmp:
        info["stats"]["dec"] = dec
    if lod:
        info["lod"] = lod
    info["lod_pick"] = [{"level": n, "ships": picks[n]["winner"], "kind": picks[n]["kind"], "allowance": picks[n]["allowance"],
                         "over": picks[n].get("over", 0),
                         "img": picks[n]["img"], "clean": picks[n]["clean"], "why": picks[n]["why"],
                         "better": picks[n].get("better"),
                         "cand": {c["name"]: dict({"tris": c["tris"], "relaxed": bool(c.get("relaxed")), "closed": int(c.get("closed") or 0)},
                                                  **{k: (c["m"] or {}).get(k) for k in ("img", "sil", "tear", "lum", "glow", "text", "zf", "open", "holes")})
                                  for c in cands[n]}} for n in (1, 2)]
    if where is not None:
        rs = [res.get(n) or {} for n in (1, 2)]
        info["_where"] = {k: [where.get(k, 0)] + [r.get("where", {}).get(k, 0) for r in rs]
                          for k in set(where).union(*[set(r.get("where", {})) for r in rs])}
    with open(stem + ".json", "w") as f:
        json.dump(info, f, indent=1)
    info["_lod_trace"] = [r["trace"] for r in res.values() if r.get("trace")]
    return info


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = parse(argv)
    asset_id = a["asset"]
    try:
        mod = asm.load_asset(asset_id)
    except Exception:
        print("TSQ_RESULT " + json.dumps({"id": asset_id, "variant": "*", "ok": False, "error": traceback.format_exc()}))
        return
    variants = getattr(mod, "VARIANTS", {}) or {}
    jobs = []
    if a["jobs"] is not None:
        # town builds: a JSON file [{"name": "_ab12", "variant": "twin" | null, "params": {...}}, ...]
        jobs = [(j["name"], j.get("params") or {}, j.get("variant")) for j in json.load(open(a["jobs"]))]
    elif a["params"] is not None:
        jobs.append((a["name"] or "custom", json.loads(a["params"])))
    elif a["variants"] == "all":
        names = list(variants) or ["default"]
        if "default" not in names:
            names = ["default"] + names
        jobs = [(n, None) for n in names]
    else:
        jobs = [(n, None) for n in a["variants"].split(",")]
    for job in jobs:
        vname, ov = job[0], job[1]
        base = job[2] if len(job) > 2 else None
        notes = []
        if len(job) > 2:
            # town builds are tolerant: unknown variant / params / choices are dropped with a note
            if base is not None and base not in variants and base != "default":
                notes.append(f"unknown variant {base!r} (using defaults)")
                base = None
            schema = dict(getattr(mod, "PARAMS", {}))
            clean = {}
            for k, v in (ov or {}).items():
                spec = schema.get(k)
                if k == "seed":
                    clean[k] = v
                elif spec is None:
                    notes.append(f"unknown param {k!r} dropped")
                elif "choices" in spec and v not in spec["choices"]:
                    notes.append(f"param {k}={v!r} not in choices (default used)")
                elif isinstance(v, (int, float)) and not isinstance(v, bool) and (
                        ("min" in spec and v < spec["min"] - 1e-9) or ("max" in spec and v > spec["max"] + 1e-9)):
                    notes.append(f"param {k}={v} clamped to [{spec.get('min')}, {spec.get('max')}]")
                    clean[k] = v
                else:
                    clean[k] = v
            ov = clean
        try:
            jb = base if len(job) > 2 else None
            if a["lods"] in ("1", "cmp"):
                info = build_levels(mod, asset_id, vname, ov, a["out"], base=jb, cmp=a["lods"] == "cmp", auto=a["auto"] == "1")
            else:
                info = build_one(mod, asset_id, vname, ov, a["out"], base=jb)
            res = {"id": asset_id, "variant": vname, "ok": not info["errors"], "tris": info["stats"]["tris"],
                   "size": info["bounds"]["size"], "warnings": info["warnings"], "errors": info["errors"],
                   "seconds": info["stats"]["build_s"] + info["stats"]["finalize_s"], "notes": notes}
            if info["stats"].get("lods"):
                res["lods"] = info["stats"]["lods"]
                res["lod_s"] = info["stats"].get("lod_s")
                res["dec"] = info["stats"].get("dec")
                res["lod"] = info.get("lod")
                res["lod_pick"] = info.get("lod_pick")
                res["lod_trace"] = info.get("_lod_trace")
                res["where"] = info.get("_where")
        except Exception:
            res = {"id": asset_id, "variant": vname, "ok": False, "error": traceback.format_exc(), "notes": notes}
        print("TSQ_RESULT " + json.dumps(res))
        sys.stdout.flush()


main()
