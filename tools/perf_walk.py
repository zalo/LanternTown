#!/usr/bin/env python3
"""Walk the town page with the keyboard, in this branch's own GPU Chrome (CDP :9229), and check the walk.

    python3 tools/perf_walk.py quality=phone [webgl=1] [path=perf/town/] [query='baked=0'] [town=town] [keep=1]
                               [loops=N] [series=1]

The route comes from the walkable regions of out/<town>/town.json: spawn -> gate -> plaza -> up the terrace stairs ->
terrace -> back down -> arcade -> canal quay, plus two pushes that must fail: into the largest obstacle on the plaza
and off the quay into the canal. An autopilot in the page presses the page's own keys (keydown / keyup of KeyW,
ShiftLeft, ArrowLeft / ArrowRight, KeyA / KeyD to slip round a post), so everything runs through the page's input,
walk and collision code; one stretch at the start is driven with real key events from CDP. Checked afterwards, here,
against town.json: every waypoint reached, the height on the terrace, the two pushes stopped, and no sample of the
trail inside a collider or outside the walkable regions. Also reported: frames over 50 ms, the most triangles and draw
calls in a frame, what the walk downloaded, and the GPU memory of the Chrome GPU process over a blank tab (at the end
and at most). loops=N walks back and round again N times in all (what a long walk leaves in memory); series=1 prints
[resident k triangles, GPU MiB] once a second.
"""
import sys, os, json, math, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = os.environ.get("TSQ_PERF_PORT", "9229")
PROFILE = os.path.abspath(os.environ.get("TSQ_PERF_PROFILE") or os.path.join(ROOT, ".scratch", "chrome", "tsq-chrome-" + PORT))

PILOT = r"""
window.__pilot = (route) => {
  const S = window.__pilotState = { i: 0, done: false, log: [], trail: [], frames: 0, maxTris: 0, maxDraws: 0, slow: 0 };
  const held = new Set();
  const set = (code, on) => { if (on === held.has(code)) return; if (on) held.add(code); else held.delete(code); dispatchEvent(new KeyboardEvent(on ? 'keydown' : 'keyup', { code, bubbles: true })); };
  const pose = () => { const c = window.eng.camera, e = c.matrixWorld.elements; return { x: c.position.x, y: -c.position.z, h: c.position.y, yaw: Math.atan2(e[8], e[10]) }; };
  let t0 = performance.now(), last = t0, lastTrail = 0, ref = null, refT = t0, slip = 0, slipKey = 'KeyD';
  const next = (ok, why) => { const p = pose(), w = route[S.i]; const vs = window.town.visibleStats ? window.town.visibleStats() : null;
    S.log.push({ name: w.name, ok, why, x: +p.x.toFixed(2), y: +p.y.toFixed(2), h: +p.h.toFixed(2), s: +((performance.now() - t0) / 1000).toFixed(1), lods: vs && vs.lods });
    for (const k of [...held]) set(k, false);
    S.i++; t0 = performance.now(); ref = null; slip = 0;
    if (S.i >= route.length) S.done = true; };
  const tick = () => {
    if (S.done) return;
    requestAnimationFrame(tick);
    const now = performance.now(), dt = now - last; last = now; S.frames++; if (dt > 50) S.slow++;
    const info = window.eng.renderer.info.render; S.maxTris = Math.max(S.maxTris, info.triangles); S.maxDraws = Math.max(S.maxDraws, info.drawCalls);
    const p = pose(), w = route[S.i];
    if (now - lastTrail > 100) { lastTrail = now; S.trail.push([+p.x.toFixed(3), +p.y.toFixed(3), +p.h.toFixed(3)]); }
    const dx = w.x - p.x, dy = w.y - p.y, d = Math.hypot(dx, dy);
    if (!w.push && d < (w.r || 0.6)) return next(true);
    if ((now - t0) / 1000 > (w.push || 45)) return next(!!w.push, w.push ? 'pushed' : 'timeout');
    let err = Math.atan2(-dx, dy) - p.yaw; err = Math.atan2(Math.sin(err), Math.cos(err));
    set('ArrowLeft', err > 0.05); set('ArrowRight', err < -0.05);
    const go = Math.abs(err) < 0.35;
    set('KeyW', go); set('ShiftLeft', go && d > 2 && !w.push);
    if (w.push) return;
    // not getting anywhere (a post, a table in the way): slip sideways for a moment, the other way next time
    if (!ref || !go) { ref = p; refT = now; }
    if (now < slip) return;
    set('KeyD', false); set('KeyA', false);
    if (now - refT > 700) { if (Math.hypot(p.x - ref.x, p.y - ref.y) < 0.25) { slipKey = slipKey === 'KeyD' ? 'KeyA' : 'KeyD'; set(slipKey, true); slip = now + 500 + Math.random() * 400; } ref = p; refT = now; }
  };
  requestAnimationFrame(tick);
};
"""

DRIVER = r'''
import time, json, subprocess, re
CFG = json.loads(%(cfg)r)
def vram():
    # MiB of this Chrome's GPU process (nvidia-smi)
    ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    gpu = [l.split(None, 1)[0] for l in ps if CFG["profile"] in l and "type=gpu-process" in l]
    smi = dict(l.replace(" ", "").split(",") for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines() if l)
    return sum(int(re.sub(r"\D", "", smi.get(p, "0")) or 0) for p in gpu)
def J(expr):
    try:
        r = js(expr)
        return json.loads(r) if r else None
    except Exception as ex:
        return None
tid = new_tab("about:blank")
cdp("Page.enable")
cdp("Page.addScriptToEvaluateOnNewDocument", source="try { performance.setResourceTimingBufferSize(30000); } catch (e) {}")
cdp("Page.bringToFront")
time.sleep(1.5)
BASE = vram()   # the blank tab: GPU memory is given over this, as tools/perf_bench.py does
goto_url(CFG["url"])
t0 = time.time(); ok = False
while time.time() - t0 < 180:
    time.sleep(0.5)
    st = J("JSON.stringify({go: document.getElementById('go')?.style.display === 'block', eng: !!window.eng, town: !!window.town})")
    if st and st["go"] and st["eng"]:
        ok = True; break
if not ok:
    print("WALK_FAIL", json.dumps(dict(msg=J("JSON.stringify(document.getElementById('msg')?.textContent || '')"))))
else:
    cdp("Page.bringToFront")
    time.sleep(1.0)
    POSE = "(() => { const c = window.eng.camera; return JSON.stringify({x: c.position.x, y: -c.position.z, h: c.position.y, gpu: window.eng.isWebGPU ? 'webgpu' : 'webgl2', lite: !!window.eng.lite}); })()"
    a = J(POSE)
    # real key events first: W held for a second and a half
    cdp("Input.dispatchKeyEvent", type="rawKeyDown", code="KeyW", key="w", windowsVirtualKeyCode=87, nativeVirtualKeyCode=87)
    time.sleep(1.5)
    cdp("Input.dispatchKeyEvent", type="keyUp", code="KeyW", key="w", windowsVirtualKeyCode=87, nativeVirtualKeyCode=87)
    time.sleep(0.3)
    b = J(POSE)
    print("WALK_KEYS", json.dumps(dict(a=a, b=b)))
    js(CFG["pilot"])
    js("window.__pilot(" + json.dumps(CFG["route"]) + "); 1")
    t0 = time.time(); st = None; peak = 0; series = []
    while time.time() - t0 < CFG["timeout"]:
        time.sleep(1.0)
        v = vram() - BASE; peak = max(peak, v)
        st = J("JSON.stringify({done: window.__pilotState.done, i: window.__pilotState.i, res: window.town.visibleStats ? window.town.visibleStats().resident || 0 : 0})")
        if st: series.append([round(st["res"] / 1e3), v])
        if st and st["done"]: break
    print("WALK_STATE", js("JSON.stringify(window.__pilotState)"))
    # what the walk downloaded (as sent: gzip counts; a file fetched again comes from the cache and counts once), what
    # it left resident, GPU memory
    end = J("(() => { let all = 0, bake = 0, n = 0; const seen = new Set(); for (const e of performance.getEntriesByType('resource').concat(performance.getEntriesByType('navigation'))) { n ++; if (seen.has(e.name)) continue; seen.add(e.name); const b = e.encodedBodySize || 0; all += b; if (/_bake/.test(e.name)) bake += b; } const vs = window.town.visibleStats ? window.town.visibleStats() : {}; return JSON.stringify({all, bake, n, resident: vs.resident || 0}); })()") or {}
    print("WALK_END", json.dumps(dict(end, vram=vram() - BASE, base=BASE, peak=peak, series=series)))
goto_url("about:blank")
'''


def main():
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    town = kw.get("town", "town")
    T = json.load(open(os.path.join(ROOT, "out", town, "town.json")))
    R = {r["name"]: r for r in T["regions"]}
    mid = lambda r: ((r["rect"][0] + r["rect"][2]) / 2, (r["rect"][1] + r["rect"][3]) / 2)
    route = []
    add = lambda name, x, y, **k: route.append(dict(name=name, x=round(x, 2), y=round(y, 2), **k))
    plaza, quay = R.get("plaza"), R.get("quay")
    probe = None
    if "gate" in R:
        add("gate", *mid(R["gate"]))
    if plaza:
        pr = plaza["rect"]
        gx = mid(R["gate"])[0] if "gate" in R else mid(plaza)[0]
        add("plaza", gx, pr[1] + 2.5)
        # the largest obstacle well inside the plaza: walk straight at it, it must stop the walker
        inside = [c for c in T["colliders"] if pr[0] + 4 < c["c"][0] < pr[2] - 4 and pr[1] + 4 < c["c"][1] < pr[3] - 4 and (not c.get("z") or (c["z"][0] < 0.3 and c["z"][1] > 1.0))]
        if inside:
            size = lambda c: c["rad"] if c["t"] == "c" else min(c["h"])
            probe = max(inside, key=size)
            add("push:obstacle", probe["c"][0], probe["c"][1], push=9)
    st = next((r for n, r in R.items() if r.get("ramp") and "terrace" in n), None)
    if st and "terrace" in R:
        q = st["ramp"]
        ym = (st["rect"][1] + st["rect"][3]) / 2 if abs(q["d"][0]) > abs(q["d"][1]) else None
        xm = (st["rect"][0] + st["rect"][2]) / 2 if ym is None else None
        foot = (q["o"][0] - q["d"][0] * 1.5, ym) if ym is not None else (xm, q["o"][1] - q["d"][1] * 1.5)
        top = (q["o"][0] + q["d"][0] * (q["len"] + 2.0), ym) if ym is not None else (xm, q["o"][1] + q["d"][1] * (q["len"] + 2.0))
        if q["z1"] < q["z0"]:
            foot, top = top, foot
        add("stairs foot", *foot)
        add("terrace", *top, z=max(q["z0"], q["z1"]))
        add("stairs foot again", *foot)
    if "arcade" in R:
        ar = R["arcade"]["rect"]
        ay = (ar[1] + ar[3]) / 2
        east, west = (ar[2] + 1.5, ay), (ar[0] + 1.0, ay)
        if quay and mid(quay)[0] > mid(R["arcade"])[0]:
            east, west = (ar[0] - 1.5, ay), (ar[2] - 1.0, ay)
        add("arcade", *east)
        add("arcade end", *west)
    if quay:
        qr = quay["rect"]
        qx = (qr[0] + qr[2]) / 2
        add("quay", qx, mid(R["arcade"])[1] if "arcade" in R else mid(quay)[1])
        # along the quay, then straight at the canal: the edge must hold (away from the bridge)
        water = -1 if not plaza or mid(plaza)[0] > qx else 1
        y2 = qr[1] + (qr[3] - qr[1]) * 0.25
        add("quay south", qx, y2)
        add("push:canal", qx + water * 8, y2, push=5)

    # loops=N: back the way it came and round again (without the pushes), to see what a long walk leaves in memory
    once = list(route)
    for k in range(1, int(kw.get("loops", 1))):
        back = [dict(w, name=w["name"] + " (back)") for w in reversed(once[:-1]) if not w.get("push")]
        route += back + [dict(w, name=w["name"] + " (%d)" % (k + 1)) for w in once if not w.get("push")]
    q = "quality=" + kw.get("quality", "phone")
    if kw.get("webgl"):
        q += "&webgl=1"
    if town != "town":
        q += "&town=" + town
    if kw.get("query"):
        q += "&" + kw["query"].strip("&")
    path = kw.get("path", "perf/town/")
    cfg = dict(profile=PROFILE, url=f"http://127.0.0.1:{os.environ.get('TSQ_PORT', '8765')}/{path}?{q}&t={int(time.time())}", route=route, pilot=PILOT, timeout=int(kw.get("timeout", 400)))
    os.makedirs(os.path.dirname(PROFILE), exist_ok=True)
    cenv = dict(os.environ, TSQ_CHROME_PORT=PORT, TSQ_CHROME_PROFILE=PROFILE)
    chrome = os.path.join(ROOT, "tools", "gpu_chrome.sh")
    env = dict(os.environ, BU_NAME="tsqperfwalk", BU_CDP_URL=f"http://localhost:{PORT}")
    if not kw.get("keep"):
        subprocess.run([chrome, "stop"], env=cenv, capture_output=True)  # a fresh GPU process: what an earlier page left pooled would hide in the baseline
    subprocess.run([chrome], env=cenv, check=True, capture_output=True)
    try:
        p = subprocess.run(["browser-use"], input=DRIVER % {"cfg": json.dumps(cfg)}, capture_output=True, text=True, env=env, timeout=cfg["timeout"] + 300)
    finally:
        if not kw.get("keep"):
            subprocess.run([chrome, "stop"], env=cenv, capture_output=True)
    keys = state = end = None
    for line in p.stdout.splitlines():
        if line.startswith("WALK_KEYS "):
            keys = json.loads(line[10:])
        elif line.startswith("WALK_STATE "):
            state = json.loads(line[11:])
        elif line.startswith("WALK_END "):
            end = json.loads(line[9:])
        elif line.startswith("WALK_FAIL "):
            print("the page did not come up:", line[10:])
    if not state:
        print("walk FAILED\n" + p.stdout[-1500:] + p.stderr[-1500:])
        sys.exit(1)

    # ---- checks, against town.json
    bad = []
    a, b = keys["a"], keys["b"]
    moved = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    print(f"== walk  {q}  ({b['gpu']}{', baked' if b['lite'] else ''})")
    print(f"  real key events (W for 1.5 s): moved {moved:.2f} m")
    if moved < 1.0:
        bad.append("real key events did not move the walker")
    eye = T["spawn"]["pos"][2] if len(T["spawn"]["pos"]) > 2 else 1.65
    for w, l in zip(route, state["log"]):
        note = ""
        if w.get("push"):
            if w["name"] == "push:obstacle":
                c = probe
                if c["t"] == "c":
                    gap = math.hypot(l["x"] - c["c"][0], l["y"] - c["c"][1]) - c["rad"]
                else:
                    an = math.radians(c.get("r") or 0)
                    dx, dy = l["x"] - c["c"][0], l["y"] - c["c"][1]
                    lx, ly = dx * math.cos(an) + dy * math.sin(an), -dx * math.sin(an) + dy * math.cos(an)
                    gap = math.hypot(max(abs(lx) - c["h"][0], 0), max(abs(ly) - c["h"][1], 0)) if (abs(lx) > c["h"][0] or abs(ly) > c["h"][1]) else -min(c["h"][0] - abs(lx), c["h"][1] - abs(ly))
                note = f"stopped {gap:.2f} m from the obstacle (body radius 0.30)"
                if gap < 0.25 or gap > 1.5:
                    bad.append("obstacle push: " + note)
            else:
                qr = quay["rect"]
                out = max(qr[0] - l["x"], l["x"] - qr[2], 0)
                note = f"stopped {out:.2f} m past the quay edge"
                if out > 0.05:
                    bad.append("canal push: " + note)
        elif not l["ok"]:
            bad.append(f"waypoint {w['name']} not reached ({l['why']})")
        if "z" in w:
            note = f"feet at {l['h'] - eye:.2f} m (terrace {w['z']:.2f})"
            if abs(l["h"] - eye - w["z"]) > 0.2:
                bad.append("terrace height: " + note)
        print(f"  {w['name']:18s} {'ok ' if l['ok'] else 'NO '} at ({l['x']:7.2f}, {l['y']:7.2f}) eye {l['h']:.2f}  {l['s']:5.1f} s  levels {l['lods']}  {note}")
    if len(state["log"]) < len(route):
        bad.append("route not finished")
    # the trail: never inside a collider, always over a walkable region
    worst = 0.0
    off = 0
    for x, y, h in state["trail"]:
        if not any(r["rect"][0] - 0.01 <= x <= r["rect"][2] + 0.01 and r["rect"][1] - 0.01 <= y <= r["rect"][3] + 0.01 for r in T["regions"]):
            off += 1
        z = h - eye
        for c in T["colliders"]:
            zr = c.get("z")
            if zr and (zr[1] < z + 0.45 or zr[0] > z + 1.55):  # (the eye height lags on stairs: a margin on the page's own test)
                continue
            if c["t"] == "c":
                pen = c["rad"] + 0.3 - math.hypot(x - c["c"][0], y - c["c"][1])
            else:
                an = math.radians(c.get("r") or 0)
                dx, dy = x - c["c"][0], y - c["c"][1]
                lx, ly = dx * math.cos(an) + dy * math.sin(an), -dx * math.sin(an) + dy * math.cos(an)
                ex, ey = abs(lx) - c["h"][0], abs(ly) - c["h"][1]
                pen = 0.3 - math.hypot(max(ex, 0), max(ey, 0)) if (ex > 0 or ey > 0) else 0.3 + min(-ex, -ey)
            worst = max(worst, pen)
    print(f"  trail: {len(state['trail'])} samples, deepest overlap with a collider {worst:.3f} m, {off} samples off the walkable regions")
    if worst > 0.03:
        bad.append(f"trail overlaps a collider by {worst:.2f} m")
    if off:
        bad.append(f"{off} trail samples off the walkable regions")
    print(f"  frames {state['frames']}, over 50 ms: {state['slow']}; most in a frame: {state['maxTris']} triangles, {state['maxDraws']} draw calls")
    if end:
        print(f"  downloaded {end['all'] / 1e6:.1f} MB as sent ({end['n']} requests; the baked town's files {end['bake'] / 1e6:.1f} MB); resident at the end {end['resident'] / 1e3:.0f}k triangles; "
              f"GPU memory over a blank tab ({end.get('base')} MiB) {end['vram']} MiB at the end, {end['peak']} MiB at most")
        if kw.get("series"):
            print("  resident k triangles / GPU MiB each second: " + " ".join(f"{r}/{v}" for r, v in end.get("series", [])))
    print("  RESULT:", "ok" if not bad else "FAILED - " + "; ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
