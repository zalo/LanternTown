#!/usr/bin/env python3
"""The town page (web/town/) as a visitor gets it (interactive, 1920 x 1080), measured with its own numbers.

    python3 tools/measure.py --quality medium [--throttle 4] [--walk] [--q "fine=12"]

-> JSON: first frame (ms from opening the page), fps, draw calls and triangles in view (settled, the most while
turning on the spot, the most along a walk through the square), levels, bytes downloaded, and the GPU memory of this
Chrome's GPU process (nvidia-smi) before the page, settled, and after the walk.
Chrome: the capture Chrome on port 9229 (see shoot.py).

    python3 tools/measure.py --quality low --keys [--page perf/town] [--throttle 4]

--keys: the same visit measured from outside the page, so it reads any copy of the page as it is and all of them
the same way - a frame timer of its own (requestAnimationFrame), the renderer's own counts, and the walk driven
through the page's keyboard handlers (W + Shift, the arrow keys) from the camera's position. -> per phase (turn on
the spot, the walk): frames, the median / 95th / 99th percentile / longest frame (ms), frames over 25 and 33 ms,
main-thread ms per frame (CDP Performance: tasks, script), the most triangles and draw calls of a frame.
"""
import sys, os, json, subprocess, argparse, urllib.parse

ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

DRIVER = r"""
import json, time, subprocess
url = %(url)r
PID = %(pid)d
def smi():
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for l in o.splitlines():
            a = [x.strip() for x in l.split(",")]
            if a[0] == str(PID):
                return int(a[1])
    except Exception as e:
        return "smi: " + str(e)[:80]
    return None
tid = None
for t in cdp("Target.getTargets").get("targetInfos", []):
    if t.get("type") != "page":
        continue
    if tid is None:
        tid = t["targetId"]
    elif not str(t.get("url", "")).startswith("about:"):
        try:
            cdp("Target.closeTarget", targetId=t["targetId"])  # (one town page at a time: a second one doubles the GPU memory)
        except Exception:
            pass
if tid is None:
    tid = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
sid = cdp("Target.attachToTarget", targetId=tid, flatten=True)["sessionId"]
def ev(expr, wait=False):
    try:
        r = cdp("Runtime.evaluate", session_id=sid, expression=expr, returnByValue=True, awaitPromise=wait)
    except Exception as e:
        return {"busy": str(e)[:200]}
    if r.get("exceptionDetails"):
        return {"exception": str(r["exceptionDetails"])[:600]}
    return (r.get("result") or {}).get("value")
def stats():
    v = ev("JSON.stringify(window.__stats || null)")
    return json.loads(v) if isinstance(v, str) else None
cdp("Page.enable", session_id=sid)
try:
    cdp("Target.activateTarget", targetId=tid)
    cdp("Page.bringToFront", session_id=sid)
    cdp("Emulation.setFocusEmulationEnabled", session_id=sid, enabled=True)
except Exception as e:
    print("front:", e)
cdp("Emulation.setDeviceMetricsOverride", session_id=sid, width=%(w)d, height=%(h)d, deviceScaleFactor=1, mobile=False)
cdp("Emulation.setCPUThrottlingRate", session_id=sid, rate=1)
cdp("Page.navigate", session_id=sid, url="/".join(url.split("/", 3)[:3]) + "/gallery/")
gpu = None
for _ in range(40):
    time.sleep(0.5)
    gpu = ev("(async () => (await navigator.gpu?.requestAdapter())?.info.vendor || null)()", True)
    if isinstance(gpu, str) and gpu:
        break
cdp("Page.navigate", session_id=sid, url="about:blank")
time.sleep(4)
out = {"gpu": gpu, "pid": PID, "mib_before": smi()}
cdp("Emulation.setCPUThrottlingRate", session_id=sid, rate=%(throttle)d)
cdp("Page.navigate", session_id=sid, url=url)
t0 = time.time()
st = None
while time.time() - t0 < 60:
    time.sleep(0.5)
    st = stats()
    if not st or not st.get("first") or not st.get("town"):
        continue
    tw = st["town"]
    sharp = tw.get("sharp")
    if tw.get("pending") == 0 and (not sharp or sharp[2] > 0 and (not %(sharp)d or sharp[1] >= %(sharp)d)):
        break
time.sleep(3)
st = stats() or {}
out["settle_s"] = round(time.time() - t0, 1)
out["settled"] = {k: st.get(k) for k in ("first", "load_s", "fps", "draws", "tris", "worst", "tier", "baked", "town")}
out["mib_settled"] = smi()
out["size"] = ev("JSON.stringify([innerWidth, innerHeight, document.querySelector('canvas').width, document.querySelector('canvas').height])")
out["errors"] = ev("JSON.stringify((window.__log || []).filter(l => /^(error|onerror|rejection)|invalid|shader/i.test(l)).slice(0, 6))")
def watch(seconds, until=None):
    ev("window.__stats.worst = 0; window.__stats.peak = 0; window.__stats.peakDraws = 0; window.__w = []")
    t1 = time.time(); fps = []; res = 0; lods = None; mib = 0
    while time.time() - t1 < seconds:
        time.sleep(0.5)
        s = stats() or {}
        if s.get("fps") is not None: fps.append(s["fps"])
        tw = s.get("town") or {}
        res = max(res, tw.get("resident") or 0)
        lods = tw.get("lods")
        m = smi() or 0
        mib = max(mib, m)
        if until and ev(until):
            break
    s = stats() or {}
    return {"seconds": round(time.time() - t1, 1), "fps_min": min(fps) if fps else None, "fps_med": sorted(fps)[len(fps) // 2] if fps else None, "worst_ms": s.get("worst"),
        "peak_tris": s.get("peak"), "peak_draws": s.get("peakDraws"), "resident_max": res, "lods": lods, "mib_max": mib, "bytes": (s.get("town") or {}).get("bytes")}
ev("window.__auto = { spin: 0.9 }")
out["turn"] = watch(8)
ev("window.__auto = null")
if %(fadeshot)r:
    # a change of level caught half way: the fades slowed down, a few steps forward until some start, one screenshot
    import base64
    ev("window.town.fadeTime = 16")
    ev("window.__auto = { path: %(fadepath)s, speed: 6 }")
    n = 0
    for _ in range(100):
        time.sleep(0.2)
        n = ev("window.town.fades.size")
        if isinstance(n, int) and n > 0:
            break
    time.sleep(1.5)
    ev("window.__auto = null")
    time.sleep(6)
    out["fades"] = ev("JSON.stringify([...window.town.fades].map(c => [c.m.k, c.fade.from, c.fade.to, Math.round(c.d), +((performance.now() / 1000 - c.fade.t0) / window.town.fadeTime).toFixed(2)]))")
    out["fade_at"] = ev("JSON.stringify([window.__player.x, window.__player.y, window.__player.yaw].map(v => +(+v).toFixed(2)))")
    # (a screenshot of the tab leaves a WebGPU canvas blank here: the engine's own read-back instead)
    png = ev("(async () => { const img = await window.eng.capture(); const c = document.createElement('canvas'); c.width = img.width; c.height = img.height; c.getContext('2d').putImageData(img, 0, 0); return c.toDataURL('image/png').split(',')[1]; })()", True)
    if isinstance(png, str):
        open(%(fadeshot)r, "wb").write(base64.b64decode(png))
    else:
        out["fadeshot_error"] = str(png)[:300]
    ev("window.town.fadeTime = 0.45")
    time.sleep(1)
if %(walk)d:
    ev("window.__auto = { path: %(path)s, speed: %(speed)s }")
    out["walk"] = watch(150, "!window.__auto.path.length")
    out["walk"]["walked_m"] = ev("Math.round(window.__auto.walked || 0)")
    ev("window.__auto = null")
    time.sleep(2)
    st = stats() or {}
    out["after_walk"] = {k: st.get(k) for k in ("fps", "draws", "tris", "town")}
    out["mib_after_walk"] = smi()
cdp("Emulation.setCPUThrottlingRate", session_id=sid, rate=1)
cdp("Emulation.clearDeviceMetricsOverride", session_id=sid)
cdp("Page.navigate", session_id=sid, url="about:blank")
out["vis"] = ev("document.visibilityState")
print("MEASURE " + json.dumps(out))
"""

# --keys: nothing of the page's own is used but window.eng, window.town and its key handlers
KEYS_JS = r"""
(() => {
  const eng = window.eng, cam = eng.camera;
  const M = window.__m = { dts: [], slow: [], tris: 0, draws: 0, path: null, spin: 0, walked: 0, stuck: 0, done: false, keys: {} };
  const key = (code, on) => { if (!!M.keys[code] === on) return; M.keys[code] = on; window.dispatchEvent(new KeyboardEvent(on ? 'keydown' : 'keyup', { code, bubbles: true })); };
  const off = () => { for (const k in M.keys) key(k, false); };
  let last = 0, lx = cam.position.x, lz = cam.position.z;
  const tick = (t) => {
    if (window.__m !== M) { off(); return; }
    requestAnimationFrame(tick);
    if (last) { M.dts.push(t - last); if (t - last > 25 && M.slow.length < 200) M.slow.push([+(t - last).toFixed(1), +cam.position.x.toFixed(1), +(- cam.position.z).toFixed(1), Math.round(M.walked)]); }
    last = t;
    const info = eng.renderer.info.render;
    if (info.triangles > M.tris) M.tris = info.triangles;
    if (info.drawCalls > M.draws) M.draws = info.drawCalls;
    const moved = Math.hypot(cam.position.x - lx, cam.position.z - lz);
    lx = cam.position.x; lz = cam.position.z;
    if (M.spin) { key('ArrowLeft', true); return; }
    if (!M.path || !M.path.length) { off(); return; }
    M.walked += moved;
    const x = cam.position.x, y = - cam.position.z, ex = M.path[0][0] - x, ey = M.path[0][1] - y, d = Math.hypot(ex, ey);
    if (d < 0.7 || M.stuck > 90) { M.path.shift(); M.stuck = 0; if (!M.path.length) { off(); M.done = true; } return; }
    const turn = Math.atan2(Math.sin(Math.atan2(- ex, ey) - cam.rotation.y), Math.cos(Math.atan2(- ex, ey) - cam.rotation.y));
    key('ArrowLeft', turn > 0.08); key('ArrowRight', turn < - 0.08);
    const go = Math.abs(turn) < 0.5;
    key('KeyW', go); key('ShiftLeft', go);
    M.stuck = go && moved < 0.01 ? M.stuck + 1 : 0;
  };
  requestAnimationFrame(tick);
})()
"""

KEYS_SUM = r"""
(() => { const M = window.__m, a = M.dts.slice().sort((p, q) => p - q), n = a.length, at = (f) => n ? +a[Math.min(n - 1, Math.floor(n * f))].toFixed(2) : null;
  return JSON.stringify({ frames: n, med_ms: at(0.5), p95_ms: at(0.95), p99_ms: at(0.99), worst_ms: n ? +a[n - 1].toFixed(1) : null, mean_ms: n ? +(a.reduce((p, q) => p + q, 0) / n).toFixed(2) : null,
    over25: a.filter(v => v > 25).length, over33: a.filter(v => v > 33.4).length, peak_tris: M.tris, peak_draws: M.draws, walked_m: Math.round(M.walked), left: M.path ? M.path.length : 0, slow: M.slow.slice().sort((p, q) => q[0] - p[0]).slice(0, 8) }); })()
"""

DRIVER_KEYS = r"""
import json, time, subprocess
url = %(url)r
PID = %(pid)d
def smi():
    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for l in o.splitlines():
            a = [x.strip() for x in l.split(",")]
            if a[0] == str(PID):
                return int(a[1])
    except Exception as e:
        return "smi: " + str(e)[:80]
    return None
tid = None
for t in cdp("Target.getTargets").get("targetInfos", []):
    if t.get("type") != "page":
        continue
    if tid is None:
        tid = t["targetId"]
    elif not str(t.get("url", "")).startswith("about:"):
        try:
            cdp("Target.closeTarget", targetId=t["targetId"])  # (one town page at a time)
        except Exception:
            pass
if tid is None:
    tid = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
sid = cdp("Target.attachToTarget", targetId=tid, flatten=True)["sessionId"]
def ev(expr, wait=False):
    try:
        r = cdp("Runtime.evaluate", session_id=sid, expression=expr, returnByValue=True, awaitPromise=wait)
    except Exception as e:
        return {"busy": str(e)[:200]}
    if r.get("exceptionDetails"):
        return {"exception": str(r["exceptionDetails"])[:600]}
    return (r.get("result") or {}).get("value")
def cpu():
    try:
        m = {q["name"]: q["value"] for q in cdp("Performance.getMetrics", session_id=sid).get("metrics", [])}
        return m.get("TaskDuration", 0), m.get("ScriptDuration", 0)
    except Exception:
        return 0, 0
cdp("Page.enable", session_id=sid)
cdp("Performance.enable", session_id=sid)
try:
    cdp("Target.activateTarget", targetId=tid)
    cdp("Page.bringToFront", session_id=sid)
    cdp("Emulation.setFocusEmulationEnabled", session_id=sid, enabled=True)
except Exception as e:
    print("front:", e)
cdp("Emulation.setDeviceMetricsOverride", session_id=sid, width=%(w)d, height=%(h)d, deviceScaleFactor=1, mobile=False)
cdp("Emulation.setCPUThrottlingRate", session_id=sid, rate=1)
cdp("Page.navigate", session_id=sid, url="/".join(url.split("/", 3)[:3]) + "/gallery/")
gpu = None
for _ in range(40):
    time.sleep(0.5)
    gpu = ev("(async () => (await navigator.gpu?.requestAdapter())?.info.vendor || null)()", True)
    if isinstance(gpu, str) and gpu:
        break
cdp("Page.navigate", session_id=sid, url="about:blank")
time.sleep(4)
out = {"gpu": gpu, "pid": PID, "url": url, "throttle": %(throttle)d, "mib_before": smi()}
# (the first frame, seen from outside: the frame after the page shows its start button, which it does as its loop starts)
hook = cdp("Page.addScriptToEvaluateOnNewDocument", session_id=sid, source="(() => { const t = () => { const g = document.getElementById('go'); if (g && g.style.display === 'block') { requestAnimationFrame(() => { window.__first = Math.round(performance.now()); }); return; } requestAnimationFrame(t); }; requestAnimationFrame(t); })()")
cdp("Emulation.setCPUThrottlingRate", session_id=sid, rate=%(throttle)d)
cdp("Page.navigate", session_id=sid, url=url)
t0 = time.time()
while time.time() - t0 < 90:
    time.sleep(0.5)
    v = ev("JSON.stringify(window.__first && window.town && window.eng ? window.town.visibleStats() : null)")
    st = json.loads(v) if isinstance(v, str) else None
    if st and st.get("pending") == 0:
        break
try:
    cdp("Page.removeScriptToEvaluateOnNewDocument", session_id=sid, identifier=hook.get("identifier"))
except Exception:
    pass
time.sleep(3)
ev("document.getElementById('veil') && document.getElementById('veil').classList.add('hide')")  # (as the start button does, without asking for the pointer)
time.sleep(1)
v = ev("JSON.stringify({ first: window.__first, own_first: (window.__stats || {}).first || null, town: window.town.visibleStats(), draws: window.eng.renderer.info.render.drawCalls, tris: window.eng.renderer.info.render.triangles, hud: (document.getElementById('hud') || {}).textContent || '' })")
out["settle_s"] = round(time.time() - t0, 1)
out["settled"] = json.loads(v) if isinstance(v, str) else v
out["mib_settled"] = smi()
out["size"] = ev("JSON.stringify([innerWidth, innerHeight, document.querySelector('canvas').width, document.querySelector('canvas').height])")
out["errors"] = ev("JSON.stringify((window.__log || []).filter(l => /^(error|onerror|rejection)|invalid|shader/i.test(l)).slice(0, 6))")
ev(%(keys_js)r)
def phase(start, seconds, until=None):
    time.sleep(0.5)
    ev("window.__m.dts = []; window.__m.tris = 0; window.__m.draws = 0; window.__m.walked = 0; " + start)
    c0 = cpu(); t1 = time.time(); mib = 0
    while time.time() - t1 < seconds:
        time.sleep(0.5)
        mib = max(mib, smi() or 0)
        if until and ev(until):
            break
    c1 = cpu()
    v = ev(%(keys_sum)r)
    r = json.loads(v) if isinstance(v, str) else {"error": str(v)[:300]}
    n = max(1, r.get("frames") or 1)
    r.update({"seconds": round(time.time() - t1, 1), "task_ms_frame": round((c1[0] - c0[0]) * 1000 / n, 2), "script_ms_frame": round((c1[1] - c0[1]) * 1000 / n, 2), "mib_max": mib})
    return r
out["still"] = phase("", 5)
out["turn"] = phase("window.__m.spin = 1", 8)
ev("window.__m.spin = 0")
if %(walk)d:
    out["walk"] = phase("window.__m.done = false; window.__m.path = %(path)s", 200, "window.__m.done")
    time.sleep(2)
    v = ev("JSON.stringify({ town: window.town.visibleStats(), draws: window.eng.renderer.info.render.drawCalls, tris: window.eng.renderer.info.render.triangles })")
    out["after_walk"] = json.loads(v) if isinstance(v, str) else v
    out["mib_after_walk"] = smi()
ev("window.__m = null")
cdp("Emulation.setCPUThrottlingRate", session_id=sid, rate=1)
cdp("Emulation.clearDeviceMetricsOverride", session_id=sid)
cdp("Page.navigate", session_id=sid, url="about:blank")
print("MEASURE " + json.dumps(out))
"""

# through the square: spawn -> gate -> fountain -> market -> hall steps -> back out of the gate
PATH = [ [ - 1, - 22 ], [ - 0.8, - 13 ], [ - 6, - 4 ], [ - 11, 0 ], [ - 6, 5 ], [ 2.6, 7.6 ], [ 9, 8 ], [ 2, 9 ], [ - 5.5, 8 ], [ - 1, 13 ], [ - 1, 0 ], [ - 0.8, - 13 ], [ - 1, - 30 ] ]

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--quality', default = 'high' )
	ap.add_argument( '--q', default = '' )
	ap.add_argument( '--town', default = '' )
	ap.add_argument( '--throttle', type = int, default = 1 )
	ap.add_argument( '--walk', action = 'store_true' )
	ap.add_argument( '--speed', type = float, default = 4.5 )
	ap.add_argument( '--w', type = int, default = 1920 ); ap.add_argument( '--h', type = int, default = 1080 )
	ap.add_argument( '--page', default = 'town', help = "the page under web/ ('next/town': the frozen branch copy)" )
	ap.add_argument( '--fadeshot', default = '', help = 'png path: a screenshot with the changes of level held half way' )
	ap.add_argument( '--fadepath', default = '[[-0.8,-13],[-6,-4]]' )
	ap.add_argument( '--keys', action = 'store_true', help = 'measured from outside the page, the walk through its key handlers' )
	a = ap.parse_args()
	pid = 0
	for l in subprocess.run( [ 'pgrep', '-af', 'tsq-chrome-9229' ], capture_output = True, text = True ).stdout.splitlines():

		w = l.split()
		if len( w ) > 2 and 'chrom' in os.path.basename( w[ 1 ] ) and w[ 2 ] == '--type=gpu-process': pid = int( w[ 0 ] )

	q = dict( quality = a.quality )
	if a.town: q[ 'town' ] = a.town
	url = 'http://127.0.0.1:8765/' + a.page + '/?' + urllib.parse.urlencode( q ) + ( '&' + a.q if a.q else '' )
	sharp = 1024 if a.quality == 'high' and 'sharp=0' not in a.q else 0
	env = dict( os.environ, BU_NAME = 'nexttown', BU_CDP_URL = 'http://localhost:9229' )
	if a.keys:

		src = DRIVER_KEYS % dict( url = url, pid = pid, throttle = a.throttle, walk = 1, path = json.dumps( PATH ), w = a.w, h = a.h, page = a.page, keys_js = KEYS_JS, keys_sum = KEYS_SUM )

	else: src = DRIVER % dict( url = url, pid = pid, throttle = a.throttle, walk = 1 if a.walk else 0, path = json.dumps( PATH ), speed = a.speed, w = a.w, h = a.h, sharp = sharp, fadeshot = os.path.abspath( a.fadeshot ) if a.fadeshot else '', fadepath = a.fadepath )
	p = subprocess.run( [ 'browser-use' ], input = src, capture_output = True, text = True, env = env, cwd = ROOT )
	for line in p.stdout.splitlines():

		if line.startswith( 'MEASURE ' ):

			print( json.dumps( json.loads( line[ 8: ] ), indent = 1 ) )
			return

	sys.exit( 'measure: no result\n' + p.stdout[ - 2000: ] + p.stderr[ - 2000: ] )

if __name__ == '__main__': main()
