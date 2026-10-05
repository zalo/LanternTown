#!/bin/bash
# usage: [TSQ_PERF_PORT=9229] tools/perf_measure.sh "<query>" [path, default town/]   -> loads the live town page in the :9224 GPU Chrome and prints memory / fps
cd "$(dirname "$0")/.." || exit 1
Q="$1"; P="${2:-town/}"
PORT="${TSQ_PERF_PORT:-9224}"
export TSQ_CHROME_PORT=$PORT TSQ_CHROME_PROFILE=$PWD/.scratch/chrome/tsq-chrome-$PORT
tools/gpu_chrome.sh >/dev/null 2>&1
BU_NAME=tsqperf$PORT BU_CDP_URL=http://localhost:$PORT TSQ_PERF_PROFILE=tsq-chrome-$PORT browser-use <<PY
import time, json, subprocess, os
tid = new_tab("http://127.0.0.1:8765/${P}?${Q}&t=%d" % time.time())
t0 = time.time(); st = None
while time.time() - t0 < 240:
    time.sleep(3)
    try:
        st = js("(() => { const t = window.town, e = window.eng; if (!t || !e) return null; return JSON.stringify({q: t.queue ? t.queue.length : -1, inflight: t.inflight ? t.inflight.size : -1, msg: document.getElementById('msg')?.textContent}); })()")
    except Exception as ex:
        st = None
    if st:
        s = json.loads(st)
        if s["q"] == 0 and s["inflight"] == 0 and time.time() - t0 > 20: break
print("LOAD", "load_wait_s", round(time.time() - t0, 1), st)
cdp("Page.bringToFront")
time.sleep(8)
F = "(() => { const e = window.eng, i = e.renderer.info; return JSON.stringify({frame: i.frame, calls: i.render.calls, t: performance.now(), heapMB: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null, draws: i.render.drawCalls, tris: i.render.triangles, geometries: i.memory.geometries, textures: i.memory.textures, gpu: e.isWebGPU ? 'webgpu' : 'webgl2', size: [e.renderer.domElement.width, e.renderer.domElement.height], vis: document.visibilityState}); })()"
a = json.loads(js(F)); time.sleep(5); b = json.loads(js(F))
b["fps"] = round(((b.get("frame") or b.get("calls") or 0) - (a.get("frame") or a.get("calls") or 0)) / ((b["t"] - a["t"]) / 1000), 1)
r = json.dumps(b)
print("PAGE", r)
print(subprocess.run(["python3", os.path.join(os.getcwd(), "tools", "perf_vram.py")], capture_output=True, text=True).stdout.strip())
new_tab("about:blank")
cdp("Target.closeTarget", targetId=tid)
PY
tools/gpu_chrome.sh stop >/dev/null 2>&1
