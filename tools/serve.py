#!/usr/bin/env python3
"""Static server for the town-square project (stdlib only).

    python3 tools/serve.py [port=8765]

URL layout (all pages use RELATIVE urls so they work behind the /lp/<port>/ proxy):
    /                 -> web/                (gallery/, town/, render.html, engine/)
    /out/...          -> out/                (glb, mat, renders, manifest.json)
    /node_modules/... -> node_modules/
    /docs/, /reviews/, /town-data/ -> docs/, reviews/, town/
    GET  /api/manifest   live project manifest for the gallery (tools/manifest.py)
    GET  /api/source.zip[?refs=1]   the project's source as one zip, built from the files as they are (tools/srczip.py)
    POST /api/upload?path=renders/<...>.png|webp|json&token=...   (render page -> out/renders)
    POST /api/telemetry  what a visitor's browser reports back (web/engine/telemetry.js -> tools/telemetry.py, out/.telemetry/)
A page that asks for it (<script type="importmap" data-stamp>, the town's) goes out with every script of its own folder
and of web/engine named by its date (`stamped`), so a kept copy of a script can never be paired with a newer page.
Everything is served with no-cache headers (revalidated with Last-Modified), except the files of a baked town
asked for by version (/out/web/_next*/[lean/]<file>?v=<baked> and /out/web/_bake*/<file>?v=<baked>, see
tools/webbake.py and tools/webbake_v1.py): those URLs never change content, so
they may be kept for good. Text, the bake's .bin files and the models (.glb; TSQ_GZIP_GLB=0 in the environment: not
those) go out gzip-compressed when the client takes it and it makes them smaller (compressed once per file version,
kept in memory).
A request for one byte range (`Range: bytes=a-b`) is answered 206 with that part of the file as it is on disk, never
gzip-compressed: what a loader sends to read one level out of a large .glb.
"""
import io
import os
import sys
import gzip
import re
import json
import time
import datetime
import threading
import email.utils
import secrets
import mimetypes
import posixpath
import urllib.parse
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOUNTS = {
    "out": os.path.join(ROOT, "out"),
    "node_modules": os.path.join(ROOT, "node_modules"),
    "docs": os.path.join(ROOT, "docs"),
    "reviews": os.path.join(ROOT, "reviews"),
    "town-data": os.path.join(ROOT, "town"),
}
WEB = os.path.join(ROOT, "web")
TOKEN_FILE = os.path.join(ROOT, "out", ".upload_token")
try:
    TOKEN = open(TOKEN_FILE).read().strip() or secrets.token_hex(16)   # stable across restarts (render daemon keeps working)
except OSError:
    TOKEN = secrets.token_hex(16)
_tools = {}     # name -> [module, mtime]
_tools_lock = threading.Lock()


def tool(name):
    """tools/<name>.py as a module, hot-reloaded when edited"""
    import importlib.util
    p = os.path.join(ROOT, "tools", name + ".py")
    mt = os.path.getmtime(p)
    with _tools_lock:
        e = _tools.get(name)
        if e is None or mt != e[1]:
            spec = importlib.util.spec_from_file_location("tsq_" + name, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            e = _tools[name] = [mod, mt]
        return e[0]


def manifest_bytes():
    return tool("manifest").get()

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


# --- gzip: which files are worth trying, and a small cache of the results
GZIP_EXT = {".json", ".js", ".mjs", ".bin", ".html", ".css", ".svg", ".txt", ".md", ".gltf", ".wasm", ".glb"}
if os.environ.get("TSQ_GZIP_GLB") == "0":   # the models as they were sent before (float data: gzip takes about half)
    GZIP_EXT.discard(".glb")
GZIP_MIN = 1024              # smaller files go out as they are
GZIP_GAIN = 0.95             # ... and so do files that do not shrink below this share of their size
GZIP_CACHE_MAX = 384 << 20   # bytes of compressed bodies kept (oldest use goes first)
GZIP_GLB_MAX = 256 << 20     # ... the .glb bodies have a budget of their own: they never push the other files out
_gz = {}                     # path -> [mtime_ns, size, body or None (does not shrink), last use]
_gz_lock = threading.Lock()
_gz_busy = {}                # path -> lock held while that file is being compressed


def gzipped(path, st):
    """-> the gzip-compressed content of a file, or None when compressing it gains nothing"""
    key = (st.st_mtime_ns, st.st_size)
    with _gz_lock:
        e = _gz.get(path)
        if e and (e[0], e[1]) == key:
            e[3] = time.monotonic()
            return e[2]
        busy = _gz_busy.setdefault(path, threading.Lock())
    with busy:
        with _gz_lock:
            e = _gz.get(path)
            if e and (e[0], e[1]) == key:
                return e[2]
        with open(path, "rb") as f:
            raw = f.read()
        body = gzip.compress(raw, 6, mtime=0)
        if len(raw) != st.st_size:
            return None        # being written right now: not this time
        if len(body) > len(raw) * GZIP_GAIN:
            body = None
        with _gz_lock:
            _gz[path] = [key[0], key[1], body, time.monotonic()]
            _gz_busy.pop(path, None)
            glb = path.lower().endswith(".glb")
            pool = [k for k in _gz if k.lower().endswith(".glb") == glb]
            total = sum(len(_gz[k][2]) for k in pool if _gz[k][2])
            for k in sorted(pool, key=lambda k: _gz[k][3]):
                if total <= (GZIP_GLB_MAX if glb else GZIP_CACHE_MAX):
                    break
                if k != path and _gz[k][2]:
                    total -= len(_gz[k][2])
                    del _gz[k]
        return body


def versioned(url_path, query):
    """a file of a baked town asked for with its version: the same URL always has the same content"""
    parts = [p for p in url_path.split("/") if p]
    if len(parts) < 4 or parts[:2] != ["out", "web"] or "v" not in urllib.parse.parse_qs(query):
        return False
    if len(parts) == 4:
        return parts[2].startswith(("_bake", "_next"))
    return len(parts) == 5 and parts[2].startswith("_next") and parts[3] == "lean"  # low / phone read the lean bake


STAMP = re.compile(rb'(<script type="importmap" data-stamp>)(.*?)(</script>)', re.S)
MODSRC = re.compile(rb'(<script type="module" src=")([^"?]+\.js)(\?[^"]*)?(")')


def stamped(path):
    """-> a page with its scripts named by their dates, or None when the page does not ask for it.

    `no-cache` was not enough: a phone, behind the proxy, ran the day's page with a script kept from before (the menu
    had an entry its handler did not know). So the page names what it loads: `<script type="module" src>` gets
    `v=<date of the file>`, and the import map gets an entry for every .js beside the page and in web/engine
    ("../engine/lite.js" -> "../engine/lite.js?v=..."), which every `import` in those files then follows. A file
    that did not change keeps its name and stays in the browser's cache."""
    with open(path, "rb") as f:
        html = f.read()
    m = STAMP.search(html)
    if not m:
        return None
    here, dates = os.path.dirname(path), {}
    for d in (here, os.path.join(WEB, "engine")):
        for name in sorted(os.listdir(d)):
            if name.endswith(".js"):
                p = os.path.join(d, name)
                rel = os.path.relpath(p, here).replace(os.sep, "/")
                dates[rel if rel.startswith(".") else "./" + rel] = "%x" % (os.stat(p).st_mtime_ns // 1000000)
    imap = json.loads(m.group(2))
    for rel, v in dates.items():
        imap["imports"].setdefault(rel, rel + "?v=" + v)
    html = html[:m.start(2)] + json.dumps(imap, indent="\t").encode() + html[m.end(2):]

    def src(mm):
        name = mm.group(2).decode()
        v = dates.get(name if name.startswith(".") else "./" + name)
        if not v:
            return mm.group(0)
        q = mm.group(3) or b""
        return mm.group(1) + mm.group(2) + q + (b"&" if q else b"?") + b"v=" + v.encode() + mm.group(4)
    return MODSRC.sub(src, html)


class _Part:
    """the next `n` bytes of an open file, for the base class to copy out"""

    def __init__(self, f, n):
        self.f, self.n = f, n

    def read(self, k=-1):
        k = self.n if k is None or k < 0 else min(k, self.n)
        data = self.f.read(k) if k else b""
        self.n -= len(data)
        return data

    def close(self):
        self.f.close()


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    _immutable = False
    _vary = False

    def log_message(self, fmt, *args):
        pass

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        path = posixpath.normpath(urllib.parse.unquote(path))
        parts = [p for p in path.split("/") if p and p not in (".", "..")]
        if any(p.startswith(".") for p in parts):
            return os.path.join(WEB, "__forbidden__")
        if parts and parts[0] in MOUNTS:
            return os.path.join(MOUNTS[parts[0]], *parts[1:])
        return os.path.join(WEB, *parts)

    def end_headers(self):
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if self._immutable else "no-cache")
        if self._vary:
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self._immutable = self._vary = False
        super().end_headers()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path.endswith("/api/telemetry"):                 # no token: any visitor's page (tools/telemetry.py keeps the limits)
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n <= 0 or n > 256 * 1024:
                self.close_connection = True                  # the body is not read
                return self._reply(413, {"ok": False, "error": "bad length"})
            try:
                code, obj = tool("telemetry").receive(self.rfile.read(n), self.headers)
            except Exception as e:
                code, obj = 500, {"ok": False, "error": str(e)}
            return self._reply(code, obj)
        if not u.path.endswith("/api/upload"):
            return self._reply(404, {"ok": False, "error": "not found"})
        if q.get("token", [""])[0] != TOKEN:
            return self._reply(403, {"ok": False, "error": "bad token"})
        rel = posixpath.normpath(q.get("path", [""])[0])
        parts = rel.split("/")
        if (not rel.startswith("renders/") or ".." in parts or any(p.startswith(".") for p in parts)
                or os.path.splitext(rel)[1].lower() not in (".png", ".webp", ".jpg", ".json")):
            return self._reply(400, {"ok": False, "error": "bad path"})
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > 64 * 1024 * 1024:
            return self._reply(400, {"ok": False, "error": "bad length"})
        data = self.rfile.read(n)
        dst = os.path.join(ROOT, "out", *parts)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
        return self._reply(200, {"ok": True, "bytes": n})

    def _reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0].endswith("/api/manifest"):
            try:
                body = manifest_bytes()
            except Exception as e:
                return self._reply(500, {"ok": False, "error": str(e)})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if "gzip" in self.headers.get("Accept-Encoding", "").lower():
                body = gzip.compress(body, 5, mtime=0)
                self.send_header("Content-Encoding", "gzip")
            self._vary = True
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split("?", 1)[0].endswith("/api/source.zip"):
            # the project's source as it is now (tools/srczip.py); ?refs=1: with the reference images
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                body, name = tool("srczip").build(q.get("refs", ["0"])[0] not in ("", "0"))
            except Exception as e:
                return self._reply(500, {"ok": False, "error": str(e)})
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def _range(self, path, header, vary):
        """-> the reply to a single `Range: bytes=..` (206, or 416 when it lies outside the file); False: not one"""
        m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header)
        if not m or not (m.group(1) or m.group(2)):
            return False
        try:
            f = open(path, "rb")
        except OSError:
            return False
        st = os.fstat(f.fileno())
        size = st.st_size
        if m.group(1):
            lo, hi = int(m.group(1)), min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:                                   # bytes=-N: the last N
            lo, hi = max(0, size - int(m.group(2))), size - 1
        self._vary = vary
        if lo > hi or lo >= size:
            f.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f.seek(lo)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.send_header("Content-Length", str(hi - lo + 1))
        self.send_header("Last-Modified", self.date_time_string(st.st_mtime))
        self.end_headers()
        return _Part(f, hi - lo + 1)

    def send_head(self):
        # directory without trailing slash -> relative redirect (keeps the proxy prefix intact)
        path = self.translate_path(self.path)
        if os.path.isdir(path) and not self.path.split("?", 1)[0].endswith("/"):
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            last = self.path.split("?", 1)[0].rstrip("/").split("/")[-1]
            self.send_header("Location", last + "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        self._immutable = self._vary = False
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        if not os.path.isfile(path):
            return super().send_head()
        if path.endswith(".html"):
            try:
                body = stamped(path)
            except (OSError, ValueError, KeyError):
                body = None
            if body is not None:      # (no Last-Modified: its content follows the scripts' dates, not its own)
                self._vary = True
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", self.guess_type(path))
                if "gzip" in self.headers.get("Accept-Encoding", "").lower():
                    body = gzip.compress(body, 6, mtime=0)
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return io.BytesIO(body)
        u = urllib.parse.urlsplit(self.path)
        self._immutable = versioned(posixpath.normpath(urllib.parse.unquote(u.path)), u.query)
        zip_ext = os.path.splitext(path)[1].lower() in GZIP_EXT
        if self.headers.get("Range"):
            # a part of the file as it is on disk (never of the gzip copy): what a loader asks for when it reads one
            # level out of a large .glb. More than one range, or a unit other than bytes: the whole file, as before
            got = self._range(path, self.headers["Range"], zip_ext)
            if got is not False:
                return got
        if not zip_ext:
            return super().send_head()
        self._vary = True
        if "gzip" not in self.headers.get("Accept-Encoding", "").lower():
            return super().send_head()
        try:
            st = os.stat(path)
            body = gzipped(path, st) if st.st_size >= GZIP_MIN else None
        except OSError:
            body = None
        if body is None:
            return super().send_head()
        # (as the base class: 304 when the client's copy is not older than the file)
        since = self.headers.get("If-Modified-Since")
        if since and "If-None-Match" not in self.headers:
            try:
                t = email.utils.parsedate_to_datetime(since)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=datetime.timezone.utc)
                if int(st.st_mtime) <= t.timestamp():
                    self.send_response(HTTPStatus.NOT_MODIFIED)
                    self.end_headers()
                    return None
            except (TypeError, IndexError, OverflowError, ValueError):
                pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Last-Modified", self.date_time_string(st.st_mtime))
        self.end_headers()
        return io.BytesIO(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        f.write(TOKEN)
    os.chmod(TOKEN_FILE, 0o600)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    print(f"serving {ROOT} on http://127.0.0.1:{port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
