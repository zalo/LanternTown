#!/usr/bin/env python3
"""Measure the town page (performance branch by default) in this branch's own GPU Chrome (CDP :9229).

    python3 tools/perf_bench.py quality=phone [webgl=1] [path=perf/town/] [query='baked=0']
                                [shots=spawn,05_fountain_close,07_terrace_shops,11_canal] [throttle=6]
                                [device=phone|phone-land] [label=..] [json=out/web/bench.jsonl] [keep=1]

Per viewpoint (a fresh navigation each, HTTP cache off): seconds to the first walkable frame, bytes requested before
it / in total once streaming has settled, draw calls and triangles of a frame, canvas size, JS heap, renderer RSS,
GPU memory of Chrome's GPU process (minus what it held before the page), frame rate, and - with the main thread
throttled `throttle` times (CDP Emulation.setCPUThrottlingRate) - frame rate and CPU ms per frame. `gpu_ms` is the
uncapped time per frame on this machine's GPU (render + wait for the queue), a relative number only.
device=phone emulates a 412x915 @2.625 touch viewport (the tier is still chosen by quality=).
"""
import sys, os, json, time, subprocess, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = os.environ.get("TSQ_PERF_PORT", "9229")
# the browser profile lives on disk (never /tmp or /dev/shm: both are RAM)
PROFILE = os.path.abspath(os.environ.get("TSQ_PERF_PROFILE") or os.path.join(ROOT, ".scratch", "chrome", "tsq-chrome-" + PORT))

INIT = r"""
try { performance.setResourceTimingBufferSize(30000); } catch (e) {}
window.__res = (before) => { let b = 0, n = 0; for (const e of performance.getEntriesByType('resource')) { if (before != null && e.startTime > before) continue; b += e.encodedBodySize || 0; n++; }
  const nav = performance.getEntriesByType('navigation')[0]; if (nav) b += nav.encodedBodySize || 0; return { b, n }; };
(() => { const t = setInterval(() => { const g = document.getElementById('go'); if (g && g.style.display === 'block') { clearInterval(t); window.__walk = performance.now(); } }, 8); })();
"""

DRIVER = r'''
import time, json, subprocess, re
CFG = json.loads(%(cfg)r)
def J(expr):
    try:
        r = js(expr)
        return json.loads(r) if r else None
    except Exception as ex:
        return None
def procs():
    ps = subprocess.run(["ps", "-eo", "pid,rss,args"], capture_output=True, text=True).stdout.splitlines()
    mine = [l.split(None, 2) for l in ps if CFG["profile"] in l]
    gpu = [p for p, r, a in mine if "type=gpu-process" in a]
    smi = dict(l.replace(" ", "").split(",") for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines() if l)
    vram = sum(int(re.sub(r"\D", "", smi.get(p, "0")) or 0) for p in gpu)
    rend = sorted(int(r) for p, r, a in mine if "type=renderer" in a)
    return dict(vram=vram, rss=(rend[-1] // 1024 if rend else 0), gpu_rss=sum(int(r) for p, r, a in mine if "type=gpu-process" in a) // 1024)
tid = new_tab("about:blank")
cdp("Page.enable"); cdp("Network.enable"); cdp("Network.setCacheDisabled", cacheDisabled=True)
cdp("Page.addScriptToEvaluateOnNewDocument", source=CFG["init"])
if CFG.get("device"):
    d = CFG["device"]
    cdp("Emulation.setDeviceMetricsOverride", width=d[0], height=d[1], deviceScaleFactor=d[2], mobile=True)
    cdp("Emulation.setTouchEmulationEnabled", enabled=True)
time.sleep(1.5)
base = procs()
print("BENCH_BASE", json.dumps(base))
STAT = "(() => { const e = window.eng, t = window.town; if (!e || !t) return null; const i = e.renderer.info; const c = e.renderer.domElement; const vs = t.visibleStats ? t.visibleStats() : null; return JSON.stringify({frame: i.frame, t: performance.now(), draws: i.render.drawCalls, tris: i.render.triangles, heap: performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1e6) : null, geometries: i.memory.geometries, textures: i.memory.textures, gpu: e.isWebGPU ? 'webgpu' : 'webgl2', size: [c.width, c.height], pending: (vs && vs.pending != null) ? vs.pending : t.pending, lite: !!e.lite, vs: vs && { lods: vs.lods, tris: vs.tris, bytes: vs.bytes }, walk: window.__walk || null, warn: (t.warnings || []).slice(0, 3)}); })()"
# (eng.manual: outside the animation loop the scene pass only re-renders when the node frame is advanced by hand)
CPU = "(() => { const e = window.eng, t = window.town; const n = 20; const m = e.manual; e.manual = true; e.render(); const a = performance.now(); for (let k = 0; k < n; k++) { e.update(k * 0.016); t.update(e.camera, k * 0.016); e.render(); } const ms = (performance.now() - a) / n; e.manual = m; return JSON.stringify({ms}); })()"
GPU = "(async () => { const e = window.eng, n = 40; const dev = e.renderer.backend.device, gl = e.renderer.backend.gl; const m = e.manual; e.manual = true; e.render(); if (dev) await dev.queue.onSubmittedWorkDone(); else if (gl) gl.finish(); const a = performance.now(); for (let k = 0; k < n; k++) e.render(); e.manual = m; if (dev) await dev.queue.onSubmittedWorkDone(); else if (gl) { gl.finish(); const px = new Uint8Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); } return JSON.stringify({ms: (performance.now() - a) / n}); })()"
def fps(sec):
    a = J(STAT); time.sleep(sec); b = J(STAT)
    if not a or not b: return None
    return round((b["frame"] - a["frame"]) / ((b["t"] - a["t"]) / 1000), 1)
for shot in CFG["shots"]:
    url = CFG["url"] + ("" if shot == "spawn" else "&shot=" + shot) + "&t=%%d" %% time.time()
    cdp("Emulation.setCPUThrottlingRate", rate=1)
    cdp("Page.bringToFront")
    goto_url(url)
    t0 = time.time(); st = None; quiet = 0
    while time.time() - t0 < CFG["timeout"]:
        time.sleep(0.5)
        st = J(STAT)
        if st and st["walk"] and st["pending"] == 0:
            quiet += 1
            if quiet >= 6: break
        else: quiet = 0
    if not st or not st.get("walk"):
        print("BENCH_ROW", json.dumps(dict(shot=shot, failed=True, msg=J("JSON.stringify(document.getElementById('msg')?.textContent || '')"), st=st)))
        continue
    cdp("Page.bringToFront")
    time.sleep(2)
    r = dict(shot=shot)
    st = J(STAT)
    walk = st["walk"]
    first = J("JSON.stringify(window.__res(%%f))" %% walk); total = J("JSON.stringify(window.__res())")
    r.update(walk_s=round(walk / 1000, 2), mb_first=round(first["b"] / 1e6, 1), mb_total=round(total["b"] / 1e6, 1), requests=total["n"],
        draws=st["draws"], tris=st["tris"], heap=st["heap"], size=st["size"], gpu=st["gpu"], lite=st["lite"], lods=(st["vs"] or {}).get("lods"), warn=st["warn"])
    r["fps"] = fps(3)
    p = procs()
    r.update(vram=p["vram"] - base["vram"], vram_abs=p["vram"], rss=p["rss"], gpu_rss=p["gpu_rss"])
    g = J(GPU); r["gpu_ms"] = g and round(g["ms"], 2)
    c = J(CPU); r["cpu_ms"] = c and round(c["ms"], 2)
    if CFG["throttle"] > 1:
        cdp("Emulation.setCPUThrottlingRate", rate=CFG["throttle"])
        time.sleep(1.5)
        r["fps_thr"] = fps(4)
        c = J(CPU); r["cpu_ms_thr"] = c and round(c["ms"], 2)
        cdp("Emulation.setCPUThrottlingRate", rate=1)
    print("BENCH_ROW", json.dumps(r))
if CFG.get("slow"):
    # the load again with a slow main thread (decode / upload cost a phone pays)
    cdp("Emulation.setCPUThrottlingRate", rate=CFG["load_throttle"])
    cdp("Page.bringToFront")
    goto_url(CFG["url"] + "&t=%%d" %% time.time())
    t0 = time.time(); st = None
    while time.time() - t0 < CFG["timeout"]:
        time.sleep(0.5)
        st = J(STAT)
        if st and st["walk"]: break
    cdp("Emulation.setCPUThrottlingRate", rate=1)
    print("BENCH_SLOWLOAD", json.dumps(dict(rate=CFG["load_throttle"], walk_s=st and st.get("walk") and round(st["walk"] / 1000, 2))))
goto_url("about:blank")
'''


def main():
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    quality = kw.get("quality", "phone")
    q = "quality=" + quality if quality != "auto" else "auto=1"  # quality=auto: let the page pick its default tier
    if kw.get("webgl"):
        q += "&webgl=1"
    if kw.get("query"):
        q += "&" + kw["query"].strip("&")
    path = kw.get("path", "perf/town/")
    devices = {"phone": [412, 915, 2.625], "phone-land": [915, 412, 2.625]}
    cfg = dict(
        url=f"http://127.0.0.1:{os.environ.get('TSQ_PORT', '8765')}/{path}?{q}",
        shots=(kw.get("shots") or "spawn,05_fountain_close,07_terrace_shops,11_canal").split(","),
        throttle=float(kw.get("throttle", 6)), load_throttle=float(kw.get("load_throttle", 4)), timeout=int(kw.get("timeout", 240)),
        profile="--user-data-dir=" + PROFILE, init=INIT, device=devices.get(kw.get("device", "")),
    )
    label = kw.get("label") or (path.strip("/").replace("/", ".") + " " + q + (" " + kw["device"] if kw.get("device") else ""))
    os.makedirs(os.path.dirname(PROFILE), exist_ok=True)
    cenv = dict(os.environ, TSQ_CHROME_PORT=PORT, TSQ_CHROME_PROFILE=PROFILE)
    chrome = os.path.join(ROOT, "tools", "gpu_chrome.sh")
    env = dict(os.environ, BU_NAME="tsqperfbench", BU_CDP_URL=f"http://localhost:{PORT}")
    out = ""
    # a fresh browser per viewpoint: memory of an earlier page would otherwise count against the next one
    runs = [dict(cfg, shots=[s_], slow=False) for s_ in cfg["shots"]]
    if cfg["load_throttle"] > 1:
        runs.append(dict(cfg, shots=[], slow=True))
    try:
        for run in runs:
            for attempt in range(2):
                subprocess.run([chrome, "stop"], env=cenv, capture_output=True)
                time.sleep(1.5)
                subprocess.run([chrome], env=cenv, check=True, capture_output=True)
                try:
                    p = subprocess.run(["browser-use"], input=DRIVER % {"cfg": json.dumps(run)}, capture_output=True, text=True, env=env, timeout=cfg["timeout"] * 2 + 120)
                except subprocess.TimeoutExpired:
                    continue
                if "BENCH_ROW" in p.stdout or "BENCH_SLOWLOAD" in p.stdout:
                    out += p.stdout
                    break
                err = p.stdout[-800:] + p.stderr[-800:]
            else:
                print("bench FAILED for", run["shots"] or "slow load", "\n" + err)
    finally:
        if not kw.get("keep"):
            subprocess.run([chrome, "stop"], env=cenv, capture_output=True)
    rows = []
    slow = None
    for line in out.splitlines():
        if line.startswith("BENCH_ROW "):
            rows.append(json.loads(line[10:]))
        elif line.startswith("BENCH_SLOWLOAD "):
            slow = json.loads(line[15:])
    print(f"== {label}" + (f"   (first walkable frame with the CPU throttled {slow['rate']:g}x: {slow['walk_s']} s)" if slow else ""))
    print(f"{'viewpoint':<20}{'walk s':>7}{'MB 1st':>8}{'MB all':>8}{'draws':>7}{'tris k':>8}{'VRAM MB':>9}{'RSS MB':>8}{'heap':>6}{'fps':>6}{'gpu ms':>8}{'cpu ms':>8}{'thr fps':>8}{'thr ms':>8}  canvas")
    for r in rows:
        if r.get("failed"):
            print(f"{r['shot']:<20} FAILED {json.dumps(r)[:300]}")
            continue
        print(f"{r['shot']:<20}{r['walk_s']:>7}{r['mb_first']:>8}{r['mb_total']:>8}{r['draws']:>7}{r['tris'] / 1e3:>8.0f}{r['vram']:>9}{r['rss']:>8}{str(r['heap']):>6}{str(r['fps']):>6}{str(r['gpu_ms']):>8}{str(r['cpu_ms']):>8}{str(r.get('fps_thr')):>8}{str(r.get('cpu_ms_thr')):>8}  {r['size'][0]}x{r['size'][1]} {r['gpu']}{' baked' if r['lite'] else ''} {r.get('lods')}")
        for w in r.get("warn") or []:
            print("   warn:", w)
    if kw.get("json"):
        with open(os.path.join(ROOT, kw["json"]), "a") as f:
            f.write(json.dumps(dict(label=label, t=int(time.time()), slow=slow, rows=rows)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
