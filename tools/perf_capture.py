#!/usr/bin/env python3
"""Review captures of the performance branch's town page (web/perf/town/) through the page's own capture mode.

    python3 tools/perf_capture.py quality=phone out=perf/phone [shots=02_gate_view,04_cafe] [webgl=1]
                                  [path=perf/town/] [size=1600x900] [look='fog=0.01&baked=0'] [keep=1] [timeout=420]

Writes out/renders/_town/<out>/<shot>.png; `out` has to start with "perf" (the top-level _town/*.png files are the
art review's). Same driver as `tools/town.py render`, but against /perf/town/ (or path=town/ for a reference capture
of main) and in this branch's own GPU Chrome (CDP :9229, stopped afterwards unless keep=1).
"""
import sys, os, json, time, subprocess, importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = os.environ.get("TSQ_PERF_PORT", "9229")


def driver():
    spec = importlib.util.spec_from_file_location("tsq_town", os.path.join(ROOT, "tools", "town.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.DRIVER
    # the adapter warm-up page: the gallery of main (the branch copy has none)
    a = 'url.split("/town/")[0] + "/gallery/"'
    if a in d:
        d = d.replace(a, 'url.split("/perf/town/")[0].split("/town/")[0] + "/gallery/"')
    return d


def main():
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = kw.get("out", "")
    if not out.startswith("perf"):
        sys.exit(__doc__)
    token = open(os.path.join(ROOT, "out", ".upload_token")).read().strip()
    w, h = (kw.get("size") or "1600x900").split("x")
    q = f"capture=1&token={token}&quality={kw.get('quality', 'phone')}&w={w}&h={h}&t={int(time.time())}&out={out}"
    if kw.get("shots"):
        q += "&shots=" + kw["shots"]
    if kw.get("webgl"):
        q += "&webgl=1"
    if kw.get("look"):
        q += "&" + kw["look"].strip("&")
    url = f"http://127.0.0.1:{os.environ.get('TSQ_PORT', '8765')}/{kw.get('path', 'perf/town/')}?{q}"
    # the browser profile lives on disk (never /tmp or /dev/shm: both are RAM)
    profile = os.path.abspath(os.environ.get("TSQ_PERF_PROFILE") or os.path.join(ROOT, ".scratch", "chrome", "tsq-chrome-" + PORT))
    os.makedirs(os.path.dirname(profile), exist_ok=True)
    cenv = dict(os.environ, TSQ_CHROME_PORT=PORT, TSQ_CHROME_PROFILE=profile)
    chrome = os.path.join(ROOT, "tools", "gpu_chrome.sh")
    timeout = int(kw.get("timeout", 420))
    drv = driver()
    p = None
    try:
        for attempt in range(3):
            subprocess.run([chrome], env=cenv, check=True, capture_output=True)
            env = dict(os.environ, BU_NAME=f"tsqperfcap{attempt or ''}", BU_CDP_URL=f"http://localhost:{PORT}")
            p = subprocess.run(["browser-use"], input=drv % {"url": url, "timeout": timeout}, capture_output=True, text=True, env=env, timeout=timeout + 60)
            if "TOWN_RETRY" not in p.stdout:
                break
            subprocess.run([chrome, "stop"], env=cenv, capture_output=True)
            time.sleep(5)
    finally:
        if not kw.get("keep"):
            subprocess.run([chrome, "stop"], env=cenv, capture_output=True)
    for line in p.stdout.splitlines():
        if line.startswith("TOWN_RESULT "):
            r = json.loads(line[12:])
            if not r.get("done"):
                print("capture FAILED:", json.dumps(r)[:3000])
                return 1
            print(f"captured {len(r.get('shots', []))} shots in {r.get('seconds')} s  (load {r.get('load_s')} s, {r.get('gpu')}, {r.get('draws')} draw calls, "
                  f"{r.get('tris')} tris, {r.get('fps')} fps at the spawn; {r.get('cpu_ms')} ms CPU/frame, {r.get('objects')} render objects, {json.dumps(r.get('spawn'))})")
            for wmsg in r.get("warnings", [])[:10]:
                print("  warn:", wmsg)
            return 0
    print("capture FAILED (no result)\n" + p.stdout[-1500:] + p.stderr[-1500:])
    return 1


if __name__ == "__main__":
    sys.exit(main())
