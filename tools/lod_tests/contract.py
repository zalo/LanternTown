"""Contract tests for modelled levels of detail (docs/LOD.md, rules a-f). Run inside Blender:

    blender --background --factory-startup --python tools/lod_tests/contract.py

Writes throw-away generators t_*.py and their GLBs into .scratch/lodtests/ (env TSQ_LOD_WORK overrides), never into
assets/ or out/. Every generator is one small lamp post that breaks exactly one rule; the checks say which gate of
tk.final_levels.finalize_level must reject it, that the decimated copy ships instead, and that lod 0 is the same as
the build of the same generator without META["lod"]. Prints pass / FAIL per check, then ALL PASSED or FAILED: ...
Run it after any change to tk/levels.py, tk/final_levels.py, tk/final.py or tools/bl_build.py."""
import sys, os, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = (os.environ.get("TSQ_LOD_WORK") or os.path.join(ROOT, ".scratch", "lodtests")).rstrip("/") + "/"
D = S + "tests/"
os.makedirs(D, exist_ok=True)

# ---------------------------------------------------------------- the generators
HEAD = '''from tk import *
META = {"name": "t", "class": "prop", "mount": "ground", "tri_budget": 200000, "lod": True}
PARAMS = {"h": {"default": 1.2, "min": 0.5, "max": 2.0, "desc": "height"}}
VARIANTS = {}


def build(p, rng):
    iron = mat("plain", "#333333")
    glow = mat("emissive", "#ffcc88")
    post = cylinder("post", 0.08, POST_H, loc=(0, 0, POST_H / 2), verts=48, bevel=0.01)
    assign(post, iron)
    base = box("base", (0.6, 0.6, 0.1), loc=(0, 0, 0.05), bevel=0.01, segments=3)
    assign(base, iron)
RIVETS
    x, y = rng.u(-0.2, 0.2), rng.u(-0.2, 0.2)          # drawn AFTER the block a level skips
    knob = sphere("knob", 0.05, loc=(x, y, 0.15), segments=24, rings=12)
    assign(knob, iron)
    socket("after", "attach", (x, y, 0.2))
    xs = [rng.u(-0.25, 0.25) for _ in range(12)]       # draw all, emit a subset
    for i, x_ in enumerate(lod_thin(xs, 0.5, 0.25)):
        assign(box(f"bar{i}", (0.01, 0.01, 0.3), loc=(x_, 0.28, 0.25)), iron)
LAMP
    arm = box("arm", (0.3, 0.02, 0.02), loc=(0.15, 0, 1.0), bevel=0.004)
    assign(arm, iron)
PART
    light("lamp", LIGHT_AT, intensity=60)
    socket("top", "attach", TOP_AT, role="prop_top")
    z = lod_memo(MEMO_KEY, lambda: world_bbox([knob])[1].z)   # measured on the mesh: finer at lod 0 than at 2
    socket("measured", "attach", (x, y, round(z, 6)))
EXTRA
'''
BASE = {
    "POST_H": "1.2",
    "RIVETS": '''    with lod_upto(0):                                  # draws rng; its meshes are emptied above lod 0
        for i in range(40):
            a, z_ = rng.u(0, 6.283), rng.u(0.1, 1.1)
            assign(sphere(f"rivet{i}", 0.012, loc=(0.085 * cos(a), 0.085 * sin(a), z_), segments=12, rings=6), iron)''',
    "LAMP": '''    assign(sphere("lamp", 0.07, loc=(0, 0, 1.3), segments=32, rings=16), glow)
    with lod_upto(0):                                  # glow is never emptied
        assign(sphere("lamp2", 0.03, loc=(0.2, 0, 0.5), segments=16, rings=8), glow)''',
    "PART": '''    part([arm], "arm", pivot=(0, 0, 1.0), motion="swing", amp=3)''',
    "LIGHT_AT": "(0, 0, 1.3)", "TOP_AT": "(0, 0, 1.4)", "MEMO_KEY": '"knob_top"', "EXTRA": "",
}
PLAIN_RIVETS = '''    if lod() == 0:                                     # WRONG: the draws happen at lod 0 only
        for i in range(40):
            a, z_ = rng.u(0, 6.283), rng.u(0.1, 1.1)
            assign(sphere(f"rivet{i}", 0.012, loc=(0.085 * cos(a), 0.085 * sin(a), z_), segments=12, rings=6), iron)'''
FORK_RIVETS = '''    if lod() == 0:                                     # fine: its own stream
        rr = rng.fork("rivets")
        for i in range(40):
            a, z_ = rr.u(0, 6.283), rr.u(0.1, 1.1)
            assign(sphere(f"rivet{i}", 0.012, loc=(0.085 * cos(a), 0.085 * sin(a), z_), segments=12, rings=6), iron)'''
T = {
    "t_ok": {},
    "t_drift": {"RIVETS": PLAIN_RIVETS},
    "t_fork": {"RIVETS": FORK_RIVETS},
    "t_socket": {"TOP_AT": "(0, 0, 1.4 + 0.01 * lod())"},
    "t_light": {"LIGHT_AT": "(0, 0, 1.3 + 0.05 * lod())"},
    "t_part": {"PART": '''    if lod() < 2:
        part([arm], "arm", pivot=(0, 0, 1.0), motion="swing", amp=3)'''},
    "t_pivot": {"PART": '''    part([arm], "arm", pivot=(0, 0, 1.0 + 0.1 * lod()), motion="swing", amp=3)'''},
    "t_glow": {"LAMP": '''    if lod() == 0:                                     # WRONG: the lamp goes dark at the levels
        assign(sphere("lamp", 0.07, loc=(0, 0, 1.3), segments=32, rings=16), glow)'''},
    "t_budget": {"EXTRA": '''    assign(grid("slab", (2.0, 2.0), cuts=(lod_exact(60), lod_exact(60)), loc=(0, 0, 0.11)), iron)'''},
    "t_bounds": {"POST_H": "(1.2 * lod_pick(1.0, 1.5))"},
    "t_crash": {"EXTRA": '''    if lod() == 2:
        raise ValueError("boom")'''},
    "t_memo_miss": {"MEMO_KEY": '("knob_top", lod())'},
    "t_nomemo": {"EXTRA": '''    socket("unmemoed", "attach", (0, 0, round(world_bbox([knob])[0].x, 6)))'''},
    "t_use": {"EXTRA": '''    w = use("window_casement")                       # converted child: modelled at the level
    w.place(loc=(1.5, 0, 0.2))
    d = use("door_plank")                              # un-converted child: built in full, reduced inside
    d.place(loc=(-1.8, 0, 0.0))
    socket("after_use", "attach", (rng.u(-1, 1), 0, 0))'''},
}
for name, over in T.items():
    s = HEAD
    for k, v in dict(BASE, **over).items():
        s = s.replace(k, v)
    open(D + name + ".py", "w").write(s)
s = HEAD
for k, v in BASE.items():
    s = s.replace(k, v)
L = D[:-1] + "_legacy/"
os.makedirs(L, exist_ok=True)
for f in glob.glob(D + "t_*.py"):                # the same generators without the flag: today's path
    open(L + os.path.basename(f), "w").write(open(f).read().replace(', "lod": True', ""))

# ---------------------------------------------------------------- the checks
src = open(ROOT + "/tools/bl_build.py").read()
assert src.rstrip().endswith("main()")
src = src.rstrip()[:-len("main()")].replace('os.path.dirname(os.path.dirname(os.path.abspath(__file__)))', repr(ROOT))
G = {"__name__": "bl_build_t", "__file__": ROOT + "/tools/bl_build.py"}
exec(compile(src, "bl_build.py", "exec"), G)
asm = G["asm"]
real = asm.find_asset
LEG = [False]
asm.find_asset = lambda a: (S + ("tests_legacy/" if LEG[0] else "tests/") + a + ".py") if a.startswith("t_") else real(a)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glbdiff import load
OUT = S + "t_out"

def nodes(p):
    _, j = load(p)
    return sorted((n.get("name", ""), tuple(round(x, 5) for x in n.get("translation", [0, 0, 0]))) for n in j["nodes"] if "mesh" not in n)

def run(aid, cmp=False, legacy=False):
    asm._MODULES.clear()
    LEG[0] = legacy
    try:
        mod = asm.load_asset(aid)
        if mod.META.get("lod"):
            info = G["build_levels"](mod, aid, "default", None, OUT, cmp=cmp)
        else:
            info = G["build_one"](mod, aid, "default", None, OUT + "_legacy", lods=True)
    finally:
        LEG[0] = False
        asm._MODULES.clear()
    return info

def same_lod0(aid, j):
    """lod 0 of the level-aware build == the build of the same generator without META['lod'] (today's path)."""
    k = run(aid, legacy=True)
    return (k["stats"]["tris"] == j["stats"]["tris"] and json.dumps(k["sockets"], sort_keys=True) == json.dumps(j["sockets"], sort_keys=True)
            and json.dumps(k["parts"], sort_keys=True) == json.dumps(j["parts"], sort_keys=True) and k["bounds"] == j["bounds"]
            and geo(f"{OUT}/{aid}/default.glb") == geo(f"{OUT}_legacy/{aid}/default.glb")), k

import hashlib
def geo(p):
    out, _ = load(p)
    import numpy as np
    h = hashlib.sha1()
    for k in sorted(out):
        d = out[k]
        C = np.ascontiguousarray(np.hstack([d["POSITION"], d["TEXCOORD_0"]]).astype(np.float32))
        Tt = C[d["idx"].ravel()].reshape(len(d["idx"].ravel()) // 3, -1)
        h.update(Tt[np.lexsort(Tt.T[::-1])].tobytes())
    return h.hexdigest()[:12]
fails = []
def check(name, cond, note=""):
    print(("  pass  " if cond else "  FAIL  ") + name + ("   " + note if note else ""))
    if not cond:
        fails.append(name)

def modes(info):
    return [l.get("mode") for l in info["stats"]["lods"]]
def why(info, n):
    return " | ".join(info["lod"][n - 1].get("why", []))

print("== t_ok: a block skipped with lod_upto, draw-all-emit-fewer, glow, part, light, memo")
i = run("t_ok")
t0 = i["stats"]["tris"]
print("   tris", t0, [(l["tris"], l["mode"]) for l in i["stats"]["lods"]], "warnings", [w for w in i["warnings"] if "lod" in w])
check("both levels authored", modes(i) == ["authored", "authored"], why(i, 1) + why(i, 2))
check("budgets held (40 % / 10 %, floors 400 / 150)", i["stats"]["lods"][0]["tris"] <= max(0.4 * t0, 400) and i["stats"]["lods"][1]["tris"] <= max(0.1 * t0, 150))
check("stats.lods entries carry tris and glb_bytes", all(l["tris"] > 0 and l["glb_bytes"] > 0 for l in i["stats"]["lods"]))
n0, n1, n2 = (nodes(f"{OUT}/t_ok/default{s}.glb") for s in ("", ".lod1", ".lod2"))
check("level files exist next to the lod 0 file", all(os.path.exists(f"{OUT}/t_ok/default{s}.glb") for s in (".lod1", ".lod2")))
print("   lod 0 nodes:", n0)
check("socket / light / part nodes of the three GLBs identical (name + position)", n0 == n1 == n2 if (n1 and n2) else True,
      "" if (n1 or n2) else "(levels carry no nodes: the sidecar of lod 0 is the one source)")
ok_sockets = json.dumps(i["sockets"], sort_keys=True)
ok_parts = json.dumps(i["parts"], sort_keys=True)
meshes = {s: load(f"{OUT}/t_ok/default{s}.glb")[0] for s in ("", ".lod1", ".lod2")}
def has_glow(m):
    return any("emissive" in k for k in m)
check("the lamp glows at every level", all(has_glow(m) for m in meshes.values()), str({s: sorted(k.split("/")[1] for k in m) for s, m in meshes.items()}))
check("masks exported at every level", all(all("_MASK" in d for d in m.values()) for m in meshes.values()))
check("uv exported at every level", all(all("TEXCOORD_0" in d for d in m.values()) for m in meshes.values()))

ok0, il = same_lod0("t_ok", i)
check("lod 0 identical (tris, sockets, parts, bounds, geometry + uv hash) to the build without META['lod']", ok0)
check("modelled levels are cheaper than the decimated ones", all(a["tris"] <= b["tris"] for a, b in zip(i["stats"]["lods"], il["stats"]["lods"])),
      f"{[l['tris'] for l in i['stats']['lods']]} vs {[l['tris'] for l in il['stats']['lods']]}")

def expect_reject(aid, levels_bad, word, good_mode="authored"):
    print(f"== {aid}")
    j = run(aid)
    m = modes(j)
    for n in (1, 2):
        if n in levels_bad:
            check(f"lod {n} rejected ({word})", (not j["lod"][n - 1]["ok"]) and word in why(j, n), why(j, n)[:150])
            check(f"lod {n} ships the decimated copy", m[n - 1] == "decimated" and os.path.exists(f"{OUT}/{aid}/default.lod{n}.glb"))
        else:
            check(f"lod {n} stays modelled", m[n - 1] == good_mode, why(j, n)[:150])
    check("a WARN names the rejection", sum("REJECTED" in w for w in j["warnings"]) == len(levels_bad))
    ok_, k = same_lod0(aid, j)
    check("lod 0 identical to the build without META['lod']", ok_)
    # without the triangles that closed holes (docs/LOD.md rule g, "closed"): that repair follows the rays that
    # happen to pass through, and two builds of the same post differ in it (20 against 152 triangles seen)
    net = lambda x: [l["tris"] - int(l.get("closed") or 0) for l in x["stats"]["lods"]]
    check("the level that fell back has the triangles of today's decimated level", all(abs(net(j)[n - 1] - net(k)[n - 1]) <= 2 for n in levels_bad),
          f"{net(j)} vs {net(k)} (closed {[l.get('closed') for l in j['stats']['lods']]} / {[l.get('closed') for l in k['stats']['lods']]})")
    return j

expect_reject("t_drift", (1, 2), "random stream")
print("== t_fork: the same skipped block on rng.fork")
j = run("t_fork")
check("both levels authored", modes(j) == ["authored", "authored"], why(j, 1) + why(j, 2))
check("lod 0 identical to the build without META['lod']", same_lod0("t_fork", j)[0])
expect_reject("t_socket", (1, 2), "sockets")
expect_reject("t_light", (1, 2), "sockets / lights")
expect_reject("t_part", (2,), "parts differ")
expect_reject("t_pivot", (1, 2), "pivot")
expect_reject("t_glow", (1, 2), "glow")
expect_reject("t_budget", (1, 2), "budget")
expect_reject("t_bounds", (1, 2), "bounds")
expect_reject("t_crash", (2,), "build failed")
expect_reject("t_nomemo", (2,), "sockets")
print("== t_memo_miss: a memo key that lod 0 never recorded")
j = run("t_memo_miss")
print("   ", modes(j), [w for w in j["warnings"] if "lod" in w][:4])
check("a warning names the missing key", any("memo" in w for w in j["warnings"]))
print("== t_use: a converted and an un-converted child")
j = run("t_use", cmp=True)
print("   tris", j["stats"]["tris"], [(l["tris"], l["mode"]) for l in j["stats"]["lods"]], "warn", [w for w in j["warnings"] if "lod" in w][:4])
wh = j.get("_where") or {}
for k in sorted(wh, key=lambda k: -wh[k][0])[:4]:
    print("     ", k, wh[k])
check("both levels pass their checks", all(l["ok"] for l in j["lod"]), why(j, 1) + why(j, 2))
kw = [k for k in wh if "window_casement" in k]
kd = [k for k in wh if "door_plank" in k]
check("the converted child was modelled at the level", bool(kw) and sum(wh[k][1] for k in kw) < 0.3 * sum(wh[k][0] for k in kw) and sum(wh[k][2] for k in kw) < 0.06 * sum(wh[k][0] for k in kw))
check("the un-converted child was reduced inside the parent", bool(kd) and sum(wh[k][1] for k in kd) < 0.5 * sum(wh[k][0] for k in kd) and sum(wh[k][2] for k in kd) < 0.2 * sum(wh[k][0] for k in kd))
print("FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
