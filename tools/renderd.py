# Render queue daemon. Run through the browser-use CLI so the CDP helpers are available:
#
#     tools/renderd.sh            (starts the private GPU Chrome on :9223 if needed, then this daemon)
#
# Consumes out/queue/*.json serially through ONE background tab of the project's GPU Chrome
# (web/render.html), writes results to out/queue/done/<job_id>.json and a heartbeat to
# out/queue/daemon.json. Never touches tabs it did not create.
import glob
import json
import os
import signal
import sys
import time

ROOT = os.environ.get("TSQ_ROOT") or os.getcwd()
PORT = int(os.environ.get("TSQ_PORT", "8765"))
PAGE = f"http://127.0.0.1:{PORT}/render.html"
Q = os.path.join(ROOT, "out", "queue")
DONE = os.path.join(Q, "done")
HB = os.path.join(Q, "daemon.json")
JOB_TIMEOUT = float(os.environ.get("TSQ_JOB_TIMEOUT", "240"))
RECYCLE = int(os.environ.get("TSQ_RECYCLE", "20"))  # jobs per tab before it is replaced (bounds heap / VRAM growth)
VRAM_LIMIT = int(os.environ.get("TSQ_VRAM_LIMIT_MB", "8000"))  # Chrome's GPU process never returns VRAM: restart Chrome above this
os.makedirs(DONE, exist_ok=True)


def chrome_vram():
    """MiB held by the GPU process of the project's private Chrome (0 if unknown)"""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        tot = 0
        for line in out.splitlines():
            pid, mb = [x.strip() for x in line.split(",")[:2]]
            try:
                cmd = open("/proc/%s/cmdline" % pid, "rb").read()
                if b"tsq-chrome-9223" in cmd or b"tsq-gpu-chrome-profile" in cmd:
                    tot += int(mb)
            except OSError:
                pass
        return tot
    except Exception:
        return 0

state = {"pid": os.getpid(), "started": time.time(), "jobs": 0, "failed": 0, "target": None, "status": "starting"}
tab = {"tid": None, "sid": None}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def heartbeat(**kw):
    state.update(kw)
    state["time"] = time.time()
    tmp = HB + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, HB)


def ev(expr, await_promise=False):
    r = cdp("Runtime.evaluate", session_id=tab["sid"], expression=expr, returnByValue=True, awaitPromise=await_promise)
    if r.get("exceptionDetails"):
        d = r["exceptionDetails"]
        raise RuntimeError("page exception: " + str((d.get("exception") or {}).get("description") or d.get("text")))
    return (r.get("result") or {}).get("value")


def targets():
    return cdp("Target.getTargets").get("targetInfos", [])


def open_tab(fresh=False):
    """(re)attach to our own tab; create it in the background if it does not exist.
    fresh=True drops the old tab first (new renderer process: frees its JS heap and GPU resources, and is
    the only way out when the renderer has crashed - CDP calls on a crashed target just time out)."""
    tid = tab["tid"]
    if fresh and tid is not None:
        try:
            if len([t for t in targets() if t.get("type") == "page"]) > 1:
                cdp("Target.closeTarget", targetId=tid)
        except Exception as e:
            log("close of old tab failed:", e)
        tid = tab["tid"] = None
        time.sleep(0.5)
    if tid is None and not fresh and os.path.exists(HB):
        try:
            tid = json.load(open(HB)).get("target")
        except Exception:
            tid = None
    live = {t["targetId"]: t for t in targets() if t.get("type") == "page"}
    if tid not in live or "render.html" not in live[tid].get("url", "") and live[tid].get("url") != "about:blank":
        tid = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
        log("created tab", tid)
    tab["tid"] = tid
    tab["sid"] = cdp("Target.attachToTarget", targetId=tid, flatten=True)["sessionId"]
    cdp("Page.enable", session_id=tab["sid"])
    cdp("Page.navigate", session_id=tab["sid"], url=PAGE + "?t=" + str(int(time.time())))
    t0 = time.time()
    while time.time() - t0 < 60:
        time.sleep(0.25)
        try:
            if ev("window.renderReady === true"):
                break
        except Exception:
            pass
    else:
        raise RuntimeError("render page did not become ready: " + str(page_log()))
    vendor = ev("(async () => (await navigator.gpu?.requestAdapter())?.info.vendor || null)()", True)
    log("render page ready, gpu adapter:", vendor)
    heartbeat(target=tid, gpu=vendor)
    if vendor != "nvidia" and not os.environ.get("TSQ_ALLOW_SOFTWARE"):
        raise RuntimeError("no hardware WebGPU adapter (vendor=%r): start tools/gpu_chrome.sh and point BU_CDP_URL at :9223" % vendor)


def page_log(n=12):
    try:
        return ev("JSON.stringify((window.__log||[]).slice(-%d))" % n)
    except Exception as e:
        return "<no log: %s>" % e


def versions():
    mats = {}
    for p in glob.glob(os.path.join(ROOT, "out", "mat", "*", "mat.json")):
        mats[os.path.basename(os.path.dirname(p))] = int(os.path.getmtime(p))
    return mats


def run(job):
    jid = job["job_id"]
    job["token"] = open(os.path.join(ROOT, "out", ".upload_token")).read().strip()
    job["matVersions"] = versions()
    if job.get("kind") == "asset":
        d = os.path.join(ROOT, "out", "glb", job["id"])
        if not job.get("variants"):
            job["variants"] = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(d, "*.glb")))
        job["versions"] = {v: int(os.path.getmtime(os.path.join(d, v + ".glb"))) for v in job["variants"]
                           if os.path.exists(os.path.join(d, v + ".glb"))}
        missing = [v for v in job["variants"] if v not in job["versions"]]
        if missing:
            return {"ok": False, "error": "missing GLB for variants: %s (build first)" % missing}
    ev("window.startJob(%s)" % json.dumps(job))
    t0 = time.time()
    while True:
        time.sleep(0.2)
        st = json.loads(ev("window.jobStatus(%s)" % json.dumps(jid)))
        if st["state"] == "done":
            return st["result"]
        if st["state"] == "unknown":
            return {"ok": False, "error": "job lost by the render page", "log": page_log()}
        if time.time() - t0 > JOB_TIMEOUT:
            lg = page_log()
            open_tab(fresh=True)  # new tab = clean state
            return {"ok": False, "error": "render timed out after %ds" % JOB_TIMEOUT, "log": lg}
        if time.time() - state["time"] > 5:
            heartbeat(status="rendering " + jid)


def finish(path, jid, res, t_sub):
    res["job_id"] = jid
    res["waited"] = round(time.time() - t_sub, 2)
    tmp = os.path.join(DONE, jid + ".tmp")
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, os.path.join(DONE, jid + ".json"))
    try:
        os.remove(path)
    except OSError:
        pass


def shutdown(*_):
    log("shutting down")
    try:
        pages = [t for t in targets() if t.get("type") == "page"]
        if tab["tid"]:
            if len(pages) <= 1:  # never close the last tab: Chrome would exit
                cdp("Page.navigate", session_id=tab["sid"], url="about:blank")
            else:
                cdp("Target.closeTarget", targetId=tab["tid"])
    except Exception as e:
        log("close failed", e)
    heartbeat(status="stopped", target=None)
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

open_tab(fresh=True)
strikes = 0
heartbeat(status="idle")
log("renderd up, queue:", Q)
while True:
    paths = sorted(p for p in glob.glob(os.path.join(Q, "*.json")) if os.path.basename(p) != "daemon.json")
    if not paths:
        if time.time() - state["time"] > 5:
            heartbeat(status="idle", pending=0)
        time.sleep(0.15)
        continue
    path = paths[0]
    jid = os.path.basename(path)[:-5]
    t_sub = os.path.getmtime(path)
    try:
        job = json.load(open(path))
        job["job_id"] = jid
    except Exception as e:
        finish(path, jid, {"ok": False, "error": "bad job file: %s" % e}, t_sub)
        continue
    heartbeat(status="rendering " + jid, pending=len(paths))
    t0 = time.time()
    try:
        res = run(job)
    except Exception as e:
        res = {"ok": False, "error": "daemon: %s" % e, "retry": True}
        try:
            open_tab(fresh=True)
            strikes = 0
        except Exception as e2:
            strikes += 1
            log("reopen failed (%d):" % strikes, e2)
            if strikes >= 2:
                # Chrome itself is unhealthy: leave the queue intact and let renderd.sh restart everything
                heartbeat(status="restarting chrome")
                sys.exit(3)
            time.sleep(3)
        if job.get("_tries", 0) < 2:
            # the page died under this job (not the job's fault as far as we know): requeue it once or twice
            job["_tries"] = job.get("_tries", 0) + 1
            job.pop("token", None)
            with open(path + ".tmp", "w") as f:
                json.dump(job, f)
            os.replace(path + ".tmp", path)
            os.utime(path, (t_sub, t_sub))
            log("requeued", jid, "try", job["_tries"])
            continue
    res.setdefault("seconds", round(time.time() - t0, 2))
    state["jobs"] += 1
    if not res.get("ok"):
        state["failed"] += 1
    log(("ok  " if res.get("ok") else "FAIL"), jid, "%.1fs" % (time.time() - t0), "" if res.get("ok") else str(res.get("error"))[:300])
    finish(path, jid, res, t_sub)
    heartbeat(status="idle", pending=len(paths) - 1)
    if state["jobs"] % 5 == 0:
        mb = chrome_vram()
        state["vram_mb"] = mb
        hold = os.path.join(Q, "town.lock")     # a town capture is using this Chrome: let it finish first
        held = os.path.exists(hold) and time.time() - os.path.getmtime(hold) < 900
        if mb > VRAM_LIMIT and not held:
            log("chrome GPU process holds %d MiB: restarting chrome" % mb)
            heartbeat(status="restarting chrome")
            sys.exit(3)
    if state["jobs"] % RECYCLE == 0:
        try:
            open_tab(fresh=True)
        except Exception as e:
            log("recycle failed:", e)
