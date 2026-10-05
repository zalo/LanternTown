#!/usr/bin/env python3
"""What visitors' browsers report back (web/engine/telemetry.js): kept as one JSON line per report, read here.

    POST /api/telemetry           tools/serve.py -> receive(): out/.telemetry/<day>.jsonl (a folder the server never serves)
    python3 tools/telemetry.py                 the visits of the last 2 days, one line each
    python3 tools/telemetry.py --days 7        ... of the last week
    python3 tools/telemetry.py --sid 3fa9      one visit in full (any prefix of its id): device, timings, frames, log
    python3 tools/telemetry.py --errors        every error / warning line, with how many visits logged it
    python3 tools/telemetry.py --devices       the devices seen, with their frame rates per page and preset

A report is what the page chose to send and nothing more: the server adds the time it arrived and stores no address.
Limits: a report of at most 256 kB, 64 MB a day, 512 MB in all - past that reports are refused (507) and the page stops.
"""
import os
import re
import sys
import json
import time
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "out", ".telemetry")
MAX_REPORT = 256 << 10
MAX_DAY = 64 << 20
MAX_ALL = 512 << 20
_lock = threading.Lock()


def _day(t=None):
    return time.strftime("%Y-%m-%d", time.localtime(t))


def _files():
    try:
        return sorted(n for n in os.listdir(DIR) if n.endswith(".jsonl"))
    except OSError:
        return []


def receive(raw, headers=None):
    """One report from a page -> (http status, reply). Kept only when it is a JSON object with a session id."""
    if not raw or len(raw) > MAX_REPORT:
        return 413, {"ok": False, "error": "too large"}
    try:
        rep = json.loads(raw.decode("utf-8"))
    except Exception:
        return 400, {"ok": False, "error": "not json"}
    if not isinstance(rep, dict) or not re.fullmatch(r"[0-9a-zA-Z]{6,32}", str(rep.get("sid", ""))):
        return 400, {"ok": False, "error": "not a report"}
    rep["_t"] = round(time.time(), 2)
    if headers is not None and "ua" not in (rep.get("device") or {}) and rep.get("seq") == 0:
        rep["_ua"] = str(headers.get("User-Agent", ""))[:400]
    line = json.dumps(rep, separators=(",", ":"), ensure_ascii=False) + "\n"
    with _lock:
        os.makedirs(DIR, exist_ok=True)
        path = os.path.join(DIR, _day() + ".jsonl")
        try:
            today = os.path.getsize(path)
        except OSError:
            today = 0
        if today + len(line) > MAX_DAY or sum(os.path.getsize(os.path.join(DIR, n)) for n in _files()) + len(line) > MAX_ALL:
            return 507, {"ok": False, "error": "log full"}
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    return 200, {"ok": True}


# ------------------------------------------------------------------------------------------ reading

def reports(days=2):
    keep = {_day(time.time() - 86400 * i) for i in range(max(1, days))}
    out = []
    for n in _files():
        if n[:-6] not in keep:
            continue
        with open(os.path.join(DIR, n), encoding="utf-8") as f:
            for line in f:
                try:
                    r = _clean(json.loads(line))
                except Exception:
                    r = None
                if r:
                    out.append(r)
    return out


def _num(x, d=0):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) < 1e12 else d


def _clean(r):
    """A stored line -> a report of the shape the page sends, or None: the endpoint takes whatever is posted to it, so
    nothing below this may trust a field's type."""
    if not isinstance(r, dict) or not re.fullmatch(r"[0-9a-zA-Z]{6,32}", str(r.get("sid", ""))):
        return None
    o = {"sid": str(r["sid"]), "_t": _num(r.get("_t")), "seq": _num(r.get("seq")), "at": _num(r.get("at"))}
    for k, n in (("tid", 40), ("page", 40), ("url", 600), ("kind", 16), ("_ua", 400), ("kindOfDevice", 16)):
        if isinstance(r.get(k), (str, int, float)):
            o[k] = str(r[k])[:n]
    dev = r.get("device")
    if isinstance(dev, dict):
        dev = {str(k): x for k, x in dev.items() if x is not None}
        for k in ("ua", "kind", "platform", "model"):
            if k in dev:
                dev[k] = str(dev[k])[:400]
        scr = dev.get("screen")
        if not (isinstance(scr, list) and len(scr) >= 2 and all(_num(x, None) is not None for x in scr[:2])):
            dev.pop("screen", None)
        o["device"] = dev
    fr = r.get("frames")
    if isinstance(fr, dict) and _num(fr.get("n")) > 0:
        o["frames"] = {k: (int(_num(fr.get(k))) if k in ("n", "worst", "stalls", "over33", "over50") else _num(fr.get(k)))
                       for k in ("n", "fps", "p50", "p95", "p99", "worst", "over33", "over50", "stalls")}
    if isinstance(r.get("logs"), list):
        o["logs"] = [[_num(l[0]), str(l[1])[:8], str(l[2])[:2000]] for l in r["logs"][:400] if isinstance(l, list) and len(l) == 3]
    for k in ("stats", "marks", "extra", "load"):
        if isinstance(r.get(k), dict):
            o[k] = r[k]
    if "stats" in o and "first" in o["stats"]:
        o["stats"] = dict(o["stats"], first=_num(o["stats"]["first"]))
    if "load" in o:
        res = o["load"].get("res")
        res = dict(res) if isinstance(res, dict) else {}
        res["slowest"] = [[str(x[0])[:300], int(_num(x[1])), _num(x[2])] for x in res.get("slowest") or [] if isinstance(x, list) and len(x) >= 3] if isinstance(res.get("slowest"), list) else []
        res["failed"] = [[str(x[0])[:300], str(x[1])[:80]] for x in res.get("failed") or [] if isinstance(x, list) and len(x) >= 2] if isinstance(res.get("failed"), list) else []
        o["load"] = {"nav": o["load"].get("nav"), "res": res}
    for k in ("hud", "ktxinfo", "msg", "heap", "canvas", "long", "errors"):
        if r.get(k) is not None:
            o[k] = r[k] if isinstance(r[k], (int, float)) else (r[k][:4000] if isinstance(r[k], str) else json.dumps(r[k], ensure_ascii=False)[:4000])
    return o


def visits(reps):
    """-> [visit]: the reports of one page view folded together, oldest visit first"""
    by = {}
    for r in sorted(reps, key=lambda r: (r.get("_t", 0), r.get("seq", 0))):
        v = by.setdefault(r["sid"], {"sid": r["sid"], "tid": r.get("tid", ""), "t": r["_t"], "page": r.get("page", "?"), "url": r.get("url", ""),
                                     "device": {}, "kinds": [], "frames": [], "logs": [], "stats": {}, "marks": {}, "extra": {}, "load": {}, "last": {}})
        v["kinds"].append(r.get("kind", "?"))
        v["end"] = r["_t"]
        v["at"] = r.get("at", 0)
        if r.get("device"):
            v["device"].update({k: x for k, x in r["device"].items() if x is not None})
        if r.get("_ua") and "ua" not in v["device"]:
            v["device"]["ua"] = r["_ua"]
        v["device"].setdefault("kind", r.get("kindOfDevice", "?"))
        if r.get("frames"):
            v["frames"].append(dict(r["frames"], at=r.get("at", 0)))
        v["logs"] += r.get("logs") or []
        for k in ("stats", "marks", "extra", "load"):
            if r.get(k):
                v[k] = r[k] if k in ("stats", "load") else dict(v[k], **r[k])
        for k in ("hud", "ktxinfo", "msg", "heap", "canvas", "long", "errors"):
            if r.get(k) is not None:
                v["last"][k] = r[k]
    return sorted(by.values(), key=lambda v: v["t"])


def what(dev):
    """a device in a few words: kind, model / OS, browser, GPU"""
    ua = dev.get("ua", "")
    m = re.search(r"iPhone OS (\d+)[_.](\d+)", ua) or re.search(r"CPU OS (\d+)[_.](\d+)", ua)
    if m:
        osn = ("iPadOS " if "iPad" in ua else "iOS ") + m.group(1) + "." + m.group(2)
    elif "Android" in ua:
        a = re.search(r"Android ([\d.]+)(?:; ([^;)]+))?", ua)
        osn = "Android " + a.group(1) + (" " + a.group(2).strip() if a.group(2) and a.group(2).strip() != "K" else "")
    elif "Mac OS X" in ua:
        osn = "macOS"
    elif "Windows" in ua:
        osn = "Windows"
    elif "CrOS" in ua:
        osn = "ChromeOS"
    elif "Linux" in ua:
        osn = "Linux"
    else:
        osn = dev.get("platform") or "?"
    if dev.get("model"):
        osn += " " + dev["model"]
    b = (re.search(r"(CriOS|FxiOS|EdgiOS|Edg|OPR|SamsungBrowser|Firefox)/(\d+)", ua) or re.search(r"(Chrome)/(\d+)", ua)
         or re.search(r"Version/(\d+)[\d.]* .*(Safari)", ua))
    if b:
        g = b.groups()
        name, ver = (g[1], g[0]) if g[1] == "Safari" else g
        name = {"CriOS": "Chrome", "FxiOS": "Firefox", "EdgiOS": "Edge", "Edg": "Edge", "OPR": "Opera", "SamsungBrowser": "Samsung"}.get(name, name)
        br = f"{name} {ver}"
    else:
        br = "?"
    wg, gl = dev.get("webgpu"), dev.get("webgl")
    gpu = ""
    if isinstance(gl, dict) and gl.get("renderer"):
        gpu = re.sub(r"^ANGLE \((.*)\)$", r"\1", str(gl["renderer"]))
        gpu = re.sub(r", Unspecified Version| \(0x[0-9A-Fa-f]+\)", "", gpu)[:72]
    elif isinstance(wg, dict):
        gpu = " ".join(str(wg.get(k) or "") for k in ("vendor", "arch", "name")).strip()[:48]
    api = "WebGPU" if isinstance(wg, dict) else "no WebGPU" if "webgpu" in dev else ""
    scr = dev.get("screen")
    size = f"{scr[0]}x{scr[1]}@{dev.get('dpr', 1)}" if isinstance(scr, list) and len(scr) >= 2 else ""    # a report is whatever a browser sent
    mem = f"{dev.get('cores', '?')}c" + (f"/{dev['memory']}G" if dev.get("memory") else "")
    return " · ".join(x for x in (dev.get("kind", "?"), osn, br, gpu, api, size, mem) if x)


def frame_line(v):
    """the visit's frames: the reports weighted by their frame counts"""
    fr = [f for f in v["frames"] if f.get("n")]
    if not fr:
        return None
    n = sum(f["n"] for f in fr)
    return {"n": n, "fps": round(sum(f["fps"] * f["n"] for f in fr) / n, 1), "p50": round(sum(f["p50"] * f["n"] for f in fr) / n, 1),
            "p95": max(f["p95"] for f in fr), "p99": max(f["p99"] for f in fr), "worst": max(f["worst"] for f in v["frames"]),
            "stalls": sum(f.get("stalls", 0) for f in v["frames"]), "over50": sum(f.get("over50", 0) for f in fr)}


def ended(v):
    k = v["kinds"]
    if k == ["start"]:
        return "DIED LOADING?" if time.time() - v["end"] > 60 else "loading"
    if "bye" in k:
        return "left"
    return "open" if time.time() - v["end"] < 150 else "no goodbye"


def tier(v):
    s = v["stats"]
    q = dict(p.split("=", 1) for p in v["url"].split("?", 1)[1].split("&") if "=" in p) if "?" in v["url"] else {}
    name = str(s.get("tier") or q.get("quality") or "")[:12]
    return name + (" baked" if s.get("baked") else " STREAMING" if "baked" in s else "")


def cmd_list(vs):
    if not vs:
        print(f"no reports yet in {os.path.relpath(DIR, ROOT)}/")
        return
    print(f"{len(vs)} visits, {len({v['tid'] for v in vs})} browsers")
    for v in vs:
        f = frame_line(v)
        s = v["stats"]
        bits = [time.strftime("%m-%d %H:%M", time.localtime(v["t"])), v["sid"][:6], f"{v['at'] / 1000:5.0f}s", f"{ended(v):13}",
                f"{v['page']:8}", f"{tier(v):12}"]
        bits.append(f"first frame {s['first'] / 1000:.1f}s" if s.get("first") else "first frame -   ")
        bits.append(f"{f['fps']:5.1f} fps p95 {f['p95']:.0f} ms worst {f['worst']} ms" + (f" {f['stalls']} stalls" if f["stalls"] else "") if f else "no frames")
        err = sum(1 for l in v["logs"] if l[1] in ("error", "lost"))
        if err:
            bits.append(f"{err} ERRORS")
        print("  ".join(bits))
        print("      " + what(v["device"]) + "   " + v["url"][:110])


def cmd_sid(vs, pre):
    hit = [v for v in vs if v["sid"].startswith(pre)]
    if not hit:
        print("no visit with that id")
        return
    for v in hit:
        print(f"visit {v['sid']} (browser {v['tid']})  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v['t']))}  {ended(v)}  reports: {' '.join(v['kinds'])}")
        print("page   ", v["page"], v["url"])
        print("device ", what(v["device"]))
        print("        " + json.dumps({k: x for k, x in v["device"].items() if k != "ua"}, ensure_ascii=False))
        print("        " + v["device"].get("ua", ""))
        ld = v["load"]
        if ld:
            print("load   ", json.dumps(ld.get("nav")), "| downloads", json.dumps({k: x for k, x in ld.get("res", {}).items() if k not in ("slowest", "failed")}))
            for r in ld.get("res", {}).get("slowest", []):
                print(f"         {r[1]:6d} ms {r[2] / 1e6:7.2f} MB  {r[0]}")
            for r in ld.get("res", {}).get("failed", []):
                print(f"         FAILED {r[1]}  {r[0]}")
        if v["marks"]:
            print("marks  ", json.dumps(v["marks"]))
        if v["stats"]:
            print("stats  ", json.dumps(v["stats"], ensure_ascii=False)[:1500])
        for k in ("hud", "ktxinfo", "msg", "heap", "canvas", "long"):
            if k in v["last"]:
                print(f"{k:7}", str(v["last"][k]).replace("\n", "\n        "))
        for f in v["frames"]:
            print(f"frames  at {f['at'] / 1000:6.0f}s  " + json.dumps({k: x for k, x in f.items() if k != "at"}))
        print(f"log     {len(v['logs'])} lines")
        for t, level, text in v["logs"]:
            print(f"  {t / 1000:8.1f}s {level:5} " + str(text).replace("\n", "\n" + " " * 18))


def cmd_errors(vs):
    seen = {}
    for v in vs:
        for t, level, text in v["logs"]:
            if level in ("error", "lost", "warn"):
                key = (level, re.sub(r"\d+(\.\d+)?", "#", str(text).split("\n")[0])[:200])
                e = seen.setdefault(key, {"n": 0, "sids": set(), "devs": set(), "text": str(text)[:400]})
                e["n"] += 1
                e["sids"].add(v["sid"][:6])
                e["devs"].add(v["device"].get("kind", "?"))
    if not seen:
        print("no warnings or errors logged")
    for (level, _), e in sorted(seen.items(), key=lambda kv: (kv[0][0] != "error", -len(kv[1]["sids"]))):
        print(f"{level:5} x{e['n']} in {len(e['sids'])} visits ({', '.join(sorted(e['devs']))}; {', '.join(sorted(e['sids'])[:6])})")
        print("      " + e["text"].replace("\n", "\n      "))


def cmd_devices(vs):
    by = {}
    for v in vs:
        by.setdefault((v["tid"], what(v["device"])), []).append(v)
    for (tid, name), group in by.items():
        print(f"{name}   [{tid}, {len(group)} visits]")
        rows = {}
        for v in group:
            f = frame_line(v)
            if f:
                rows.setdefault((v["page"], tier(v)), []).append((f, v))
        for (page, t), fs in sorted(rows.items()):
            n = sum(f["n"] for f, _ in fs)
            first = [v["stats"]["first"] / 1000 for _, v in fs if v["stats"].get("first")]
            print(f"    {page:8} {t:14} {sum(f['fps'] * f['n'] for f, _ in fs) / n:5.1f} fps  p95 {max(f['p95'] for f, _ in fs):5.1f} ms  worst {max(f['worst'] for f, _ in fs):5d} ms"
                  + (f"  first frame {min(first):.1f}-{max(first):.1f} s" if first else "") + f"  ({len(fs)} visits)")
        died = sum(1 for v in group if ended(v) == "DIED LOADING?")
        if died:
            print(f"    {died} visits sent nothing after opening the page")


def main():
    a = sys.argv[1:]
    days = int(a[a.index("--days") + 1]) if "--days" in a else 2
    vs = visits(reports(days))
    if "--sid" in a:
        cmd_sid(vs, a[a.index("--sid") + 1])
    elif "--errors" in a:
        cmd_errors(vs)
    elif "--devices" in a:
        cmd_devices(vs)
    else:
        cmd_list(vs)


if __name__ == "__main__":
    main()
