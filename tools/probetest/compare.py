#!/usr/bin/env python3
"""Numbers of the probe test, from the files the other scripts wrote (host python, no numpy):

    python3 tools/probetest/compare.py

  out/probetest/reference.json   Cycles: white sphere at each spot (bl_ref.py mode=spots)
  out/probetest/browser.json     three.js: irradiance its SHADER reconstructs at the spots, per probe source (shots.py)
  out/probetest/probes_*.bin     the grids themselves (Blender bakes with other seeds / sample counts, three's own bake)

All values are irradiance E in linear Rec.709 (a white Lambert surface shows E / pi). -> out/probetest/compare.json
"""
import json, os, struct, math

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "out", "probetest")
LUM = (0.2126, 0.7152, 0.0722)


def lum(c):
    return sum(a * b for a, b in zip(c, LUM))


def load(stem):
    p = os.path.join(OUT, stem + ".bin")
    if not os.path.exists(p):
        return None
    raw = open(p, "rb").read()
    return struct.unpack("<%df" % (len(raw) // 4), raw)


def irr(sh, i, n):
    """three's getShIrradianceAt for probe i, normal n (three space)"""
    x, y, z = n
    w = (0.886227, 2 * 0.511664 * y, 2 * 0.511664 * z, 2 * 0.511664 * x, 2 * 0.429043 * x * y, 2 * 0.429043 * y * z,
         0.743125 * z * z - 0.247708, 2 * 0.429043 * x * z, 0.429043 * (x * x - y * y))
    o = i * 27
    return [sum(w[k] * sh[o + k * 3 + c] for k in range(9)) for c in range(3)]


AXES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def grid_diff(a, b, valid):
    """relative difference of the irradiance luminance over all valid probes x 6 axis normals"""
    rel, ratio = [], []
    for i, ok in enumerate(valid):
        if not ok:
            continue
        for n in AXES:
            ea, eb = lum(irr(a, i, n)), lum(irr(b, i, n))
            if eb > 1e-4:
                rel.append(abs(ea - eb) / eb)
                ratio.append(ea / eb)
    rel.sort()
    ratio.sort()
    return {"n": len(rel), "median_abs_rel": round(rel[len(rel) // 2], 4), "p90_abs_rel": round(rel[int(len(rel) * 0.9)], 4),
            "rms_rel": round(math.sqrt(sum(r * r for r in rel) / len(rel)), 4), "median_ratio": round(ratio[len(ratio) // 2], 4)}


ref = json.load(open(os.path.join(OUT, "reference.json")))
br = json.load(open(os.path.join(OUT, "browser.json")))
meta = json.load(open(os.path.join(OUT, "probes_blender.json")))
res = {"units": ref["units"], "spots": [], "grid": {}}

print("%-26s %-5s | %-22s | %-22s %6s | %-22s %6s | %-22s | %s" % (
    "spot", "n", "Cycles sphere (truth)", "three + BLENDER bake", "err", "three + THREE bake", "err", "as shaded (Blender)", "ideal SH2 at spot"))
errs = {"blender_probe": [], "three_probe": [], "blender_between": [], "three_between": [], "sh2": []}
for row in ref["spots"]:
    name = row["name"]
    on_probe = name.startswith("probe_")
    for nk in ("+Y", "-Y", "wall"):
        truth = row["E"]["cycles_sphere_indirect"][nk]
        eb = br["modes"]["blender"]["E"][name]["shader"][nk]
        et = br["modes"]["three"]["E"][name]["shader"][nk]
        es = br["modes"]["blender"]["E"][name]["surface"][nk]
        sh2 = row["E"]["cycles_capture_sh2"][nk]
        rb, rt, r2 = lum(eb) / lum(truth) - 1, lum(et) / lum(truth) - 1, lum(sh2) / lum(truth) - 1
        errs["blender_probe" if on_probe else "blender_between"].append(abs(rb))
        errs["three_probe" if on_probe else "three_between"].append(abs(rt))
        errs["sh2"].append(abs(r2))
        f = lambda c: "%.3f %.3f %.3f" % tuple(c)
        print("%-26s %-5s | %-22s | %-22s %+5.0f%% | %-22s %+5.0f%% | %-22s | %s (%+.0f%%)" % (
            name, nk, f(truth), f(eb), 100 * rb, f(et), 100 * rt, f(es), f(sh2), 100 * r2))
        res["spots"].append({"spot": name, "normal": nk, "on_probe": on_probe, "cycles_sphere_indirect": truth,
                             "cycles_sphere_total": row["E"]["cycles_sphere_total"][nk], "three_shader_blender_bake": eb,
                             "three_shader_three_bake": et, "three_as_shaded_blender_bake": es, "ideal_sh2_at_spot": sh2,
                             "rel_err_lum_blender": round(rb, 4), "rel_err_lum_three": round(rt, 4), "rel_err_lum_ideal_sh2": round(r2, 4)})
res["mean_abs_rel_err_luminance"] = {k: round(sum(v) / len(v), 4) for k, v in errs.items() if v}
res["max_abs_rel_err_luminance"] = {k: round(max(v), 4) for k, v in errs.items() if v}
res["median_abs_rel_err_luminance"] = {k: round(sorted(v)[len(v) // 2], 4) for k, v in errs.items() if v}
print("median |err|", res["median_abs_rel_err_luminance"])
print("mean |err|", res["mean_abs_rel_err_luminance"])
print("max  |err|", res["max_abs_rel_err_luminance"])

base = load("probes_blender")
valid = meta["valid"]
for other, label in (("probes_three", "blender_vs_three_own_bake"), ("probes_seed1", "noise_seed0_vs_seed1_64spp"),
                     ("probes_spp16", "spp16_vs_64spp"), ("probes_hi", "64spp_64x32_vs_1024spp_128x64")):
    o = load(other)
    if o is not None and len(o) == len(base):
        res["grid"][label] = grid_diff(base, o, valid) if other != "probes_three" else grid_diff(o, base, valid)
        print("%-34s %s" % (label, res["grid"][label]))
hi = load("probes_hi")
s16 = load("probes_spp16")
if hi and s16:
    res["grid"]["spp16_vs_1024spp_128x64"] = grid_diff(s16, hi, valid)
    print("%-34s %s" % ("spp16_vs_1024spp_128x64", res["grid"]["spp16_vs_1024spp_128x64"]))

# probes inside geometry: what they held before the fill
raw = load("probes_blender_raw")
if raw:
    inv = [i for i, ok in enumerate(valid) if not ok]
    lr = [lum(irr(raw, i, (0, 1, 0))) for i in inv]
    lf = [lum(irr(base, i, (0, 1, 0))) for i in inv]
    res["invalid"] = {"count": len(inv), "of": len(valid), "raw_E_up_lum_min_max": [round(min(lr), 4), round(max(lr), 4)],
                      "filled_E_up_lum_min_max": [round(min(lf), 4), round(max(lf), 4)],
                      "backface_share_min_of_invalid": round(min(meta["backface"][i] for i in inv), 3),
                      "backface_share_max_of_valid": round(max(meta["backface"][i] for i, ok in enumerate(valid) if ok), 3)}
    print("invalid probes", res["invalid"])
with open(os.path.join(OUT, "compare.json"), "w") as f:
    json.dump(res, f, indent=1)
