#!/usr/bin/env python3
"""Review captures of the next branch's page (web/next/town/) through the capture Chrome on port 9229.

    python3 tools/next/shoot.py --quality high --shots 02_gate_view,09_hall --out next/high
    python3 tools/next/shoot.py --quality medium --cam "a:0,0,1.65,10,0,1.5,60" --out next/tmp --q "fine=0"

-> out/renders/_town/<out>/<shot>.png and the page's own numbers (fps, draws, triangles, levels) as JSON.
The Chrome is started as tools/town.py starts its own:
    TSQ_CHROME_PORT=9229 TSQ_CHROME_PROFILE=<project>/.scratch/chrome/tsq-chrome-9229 tools/gpu_chrome.sh
"""
import sys, os, json, subprocess, argparse, urllib.parse

ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

DRIVER = r'''
import json, time
url = %(url)r
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
cdp("Page.enable", session_id=sid)
cdp("Emulation.setCPUThrottlingRate", session_id=sid, rate=%(throttle)d)
cdp("Page.navigate", session_id=sid, url=url.split("/", 3)[0] + "//" + url.split("/", 3)[2] + "/gallery/")
gpu = None
for _ in range(40):
    time.sleep(0.5)
    gpu = ev("(async () => (await navigator.gpu?.requestAdapter())?.info.vendor || null)()", True)
    if isinstance(gpu, str) and gpu:
        break
cdp("Page.navigate", session_id=sid, url=url)
t0 = time.time()
res = None
while time.time() - t0 < %(timeout)d:
    time.sleep(0.7)
    v = ev(%(poll)r)
    if isinstance(v, str):
        res = json.loads(v)
        if res and (res.get("done") or res.get("failed")):
            break
print("SHOOT " + json.dumps({"gpu": gpu, "res": res, "seconds": round(time.time() - t0, 1)}))
'''

def run( url, poll, timeout = 300, throttle = 1 ):

	env = dict( os.environ, BU_NAME = 'nexttown', BU_CDP_URL = 'http://localhost:9229' )
	p = subprocess.run( [ 'browser-use' ], input = DRIVER % dict( url = url, timeout = timeout, poll = poll, throttle = throttle ), capture_output = True, text = True, env = env, cwd = ROOT )
	for line in p.stdout.splitlines():

		if line.startswith( 'SHOOT ' ): return json.loads( line[ 6: ] )

	sys.exit( 'shoot: no result\n' + p.stdout[ - 2000: ] + p.stderr[ - 2000: ] )

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--quality', default = 'high' )
	ap.add_argument( '--shots', default = '02_gate_view,05_fountain_close,09_hall,13_market' )
	ap.add_argument( '--cam', default = '' )
	ap.add_argument( '--out', default = 'next/tmp' )
	ap.add_argument( '--w', type = int, default = 1600 ); ap.add_argument( '--h', type = int, default = 900 )
	ap.add_argument( '--q', default = '', help = 'more of the query string (fine=20&u_glint=1)' )
	ap.add_argument( '--town', default = '' )
	ap.add_argument( '--page', default = 'next/town', help = "'town': the live page (read only: to compare its low / phone tiers), captures still under --out" )
	ap.add_argument( '--throttle', type = int, default = 1, help = 'CPU throttling rate (4 = a four times slower main thread)' )
	ap.add_argument( '--timeout', type = int, default = 300 )
	a = ap.parse_args()
	token = open( os.path.join( ROOT, 'out/.upload_token' ) ).read().strip()
	q = dict( capture = 1, token = token, quality = a.quality, w = a.w, h = a.h, out = a.out )
	if a.cam: q[ 'cam' ] = a.cam
	else: q[ 'shots' ] = a.shots
	if a.town: q[ 'town' ] = a.town
	url = 'http://127.0.0.1:8765/' + a.page.strip( '/' ) + '/?' + urllib.parse.urlencode( q ) + ( '&' + a.q if a.q else '' )
	r = run( url, 'JSON.stringify(window.__capture || null)', a.timeout, a.throttle )
	res = r[ 'res' ] or {}
	if not res.get( 'failed' ): res.pop( 'log', None )
	print( json.dumps( r, indent = 1 ) )
	sys.exit( 0 if res.get( 'done' ) else 1 )

if __name__ == '__main__': main()
