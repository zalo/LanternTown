"""Browser side of the probe test, run INSIDE browser-use (its helpers are globals):

    BU_NAME=probetest BU_CDP_URL=http://localhost:9231 browser-use < tools/probetest/shots.py

Opens web/probetest/ in the one tab of that Chrome for every mode x camera, saves 1280x720 stills to
out/renders/_probetest/, and collects the numbers the page measures (irradiance the shader reconstructs at the spots,
three's own baked atlas read back from the GPU) into out/probetest/browser.json + probes_three.bin/.json.
"""
import json, time, base64, struct, os

ROOT = os.environ.get("TSQ_ROOT") or os.getcwd()   # (run from the project root: there is no __file__ on stdin)
PAGE = "http://localhost:8765/probetest/"
OUT = os.path.join(ROOT, "out", "probetest")
SHOTS = os.path.join(ROOT, "out", "renders", "_probetest")
os.makedirs(SHOTS, exist_ok=True)


def tab():
    for t in list_tabs():
        if "/probetest/" in t["url"] or "/gallery/" in t["url"]:
            switch_tab(t["targetId"])
            return
    new_tab("http://localhost:8765/gallery/")


def load(query, timeout=90):
    goto_url(PAGE + "?" + query + "&still=1&v=%d" % int(time.time() * 10))
    wait_for_load()
    cdp("Page.bringToFront")
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = json.loads(js("JSON.stringify({ready: window.__probetest && window.__probetest.ready, failed: window.__probetest && window.__probetest.failed})") or "{}")
        if st.get("ready") or st.get("failed"):
            break
        time.sleep(0.3)
    if not st.get("ready"):
        print("NOT READY", query, st, js("JSON.stringify(window.__log)"))
    return st


def snap(name):
    d = js("window.__probetest.snap()")
    with open(os.path.join(SHOTS, name), "wb") as f:
        f.write(base64.b64decode(d.split(",", 1)[1]))


tab()
cdp("Emulation.setDeviceMetricsOverride", width=1280, height=720, deviceScaleFactor=1, mobile=False)
adapter = None
for _ in range(40):        # a fresh Chrome can answer null for its first seconds (the page would then run on WebGL2)
    adapter = js("(async () => { const a = await navigator.gpu?.requestAdapter(); return a ? a.info.vendor + ' ' + a.info.architecture : null })()")
    if adapter:
        break
    time.sleep(0.5)
res = {"adapter": adapter, "modes": {}}
extra = os.environ.get("PT_QUERY", "")
for mode in ("off", "blender", "three"):
    for cam in ("1", "2"):
        q = "probes=%s&cam=%s%s" % (mode, cam, extra) + ("&measure=1" if cam == "1" and mode != "off" else "")
        load(q)
        snap("three_%s_cam%s.png" % (mode, cam))
        if cam == "1":
            m = json.loads(js("JSON.stringify({backend: __probetest.backend, tris: __probetest.tris, setup_ms: __probetest.probeSetupMs, E: __probetest.E || null, atlasError: __probetest.atlasError || null, log: window.__log})"))
            res["modes"][mode] = m
            if mode == "three":
                atlas = json.loads(js("JSON.stringify(window.__probetest.atlas || null)"))
                if atlas:
                    grid = json.loads(js("JSON.stringify(window.__probetest.grid)"))
                    with open(os.path.join(OUT, "probes_three.bin"), "wb") as f:
                        f.write(struct.pack("<%df" % len(atlas), *atlas))
                    with open(os.path.join(OUT, "probes_three.json"), "w") as f:
                        json.dump({"format": "as probes_blender.bin; read back from three's own LightProbeGrid.bake() atlas", "grid": grid,
                                   "probes": len(atlas) // 27, "bake": q, "bake_ms": m.get("setup_ms")}, f)
            print("MODE", mode, json.dumps({k: m.get(k) for k in ("backend", "tris", "setup_ms", "atlasError", "log")}))
# reflection-probe side test: Cycles equirect EXR as envMap of two chrome balls (sharp, rough)
if os.path.exists(os.path.join(OUT, "refl_probe.exr")):
    load("probes=blender&cam=3&figures=off&refl=1")
    snap("three_reflection_probe_cam3.png")
    res["refl"] = json.loads(js("JSON.stringify({refl: __probetest.refl || null, log: window.__log})"))
    print("REFL", res["refl"])
# the walker half way behind the red wall (dynamic object, never in any bake)
load("probes=blender&cam=2&figures=move&t=0.86")
snap("three_blender_cam2_walker.png")
with open(os.path.join(OUT, "browser.json"), "w") as f:
    json.dump(res, f, indent=1)
print("adapter", res["adapter"])
print(list_tabs())
