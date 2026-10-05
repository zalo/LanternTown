#!/usr/bin/env python3
# Baked light in the vertices of the reduced levels, the comparison sheet: the same four views of the town with every
# variant's bundle (tools/next/lightmap.py) forced onto every cell, as the page (web/next/town/) shows them.
#
#   python3 tools/next/lsvb_sheet.py --level mid --capture     captures (the capture Chrome of port 9224 must be up:
#                                                              TSQ_CHROME_PORT=9224 tools/gpu_chrome.sh) + the sheet
#   python3 tools/next/lsvb_sheet.py --level mid               the sheet again from the captures there are
#   --cols "label=bundle,..."    other columns ("ref" = the top levels as shipped, no bundle)
#   --rows lit,flat,heat,sign    lit: as the page shows it; flat: the baked light alone on grey (?flat=1);
#                                heat: what is left against the full model's light, a face at a time (<bundle>_res);
#                                sign: the same signed - blue darker than the full model, red brighter (<bundle>_sgn)
#   --views pool,wall1           only these views
#
# -> out/renders/_town/next/lsvb/<level>/<column>_<row>/<view>.png and out/renders/_town/next/lsvb/sheet_<level>.png
#
# A level is forced with the page's own distances (fine=-1&near=-1[&mid=-1]): every cell shows it, also the ones a
# tier would show at a higher level - nearer than the level is ever seen in the town (what a view is for is in VIEWS).
# Every browser call goes to the daemon "lsvb" on port 9224 and checks that before it navigates anything.

import sys, os, json, subprocess, argparse, urllib.parse, shutil

ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
REN = os.path.join( ROOT, 'out/renders/_town' )
LRAW = os.path.join( ROOT, 'out/web/_next/lm/.raw' )
PORT = 9224

# town frame, z up: eye, target, field of view
VIEWS = [
	( 'pool', '-11.25,0,1.65,-1,0,1.5,55', 'lamp pools on paving (10 m)' ),
	( 'facade', '21.4,-6.5,3.75,22.5,14,5.5,56', 'shop fronts, window light on walls (20 m)' ),
	( 'props', '2.63,7.62,1.65,13.5,8.2,2.0,58', 'market stalls: contact shadows (11 m)' ),
	( 'roofs', '26,-13,24,2,12,4,50', 'roofs, far (30 - 60 m)' ),
	( 'wall1', '17.5,-1,4.5,25,1,5,60', 'one row of shop fronts filling the frame (7 m)' ),
	( 'wall2', '0,4,3,0,20,5,60', 'the church front across the square (16 m)' ),
]
ROWQ = dict( lit = ( '', 0 ), flat = ( '', 1 ), heat = ( '_res', 1 ), sign = ( '_sgn', 1 ) ) # row: bundle suffix, ?flat
FORCE = dict( mid = 'fine=-1&near=-1', lo = 'fine=-1&near=-1&mid=-1', near = 'fine=-1' )
COLS = dict(
	mid = 'ref=,a shipped=mid_std_a,b refit=mid_std_b,c refit+split=mid_std_c,d relit+colour-aware+refit=mid_rlc_b,f lightmap 10 cm=mid_std_s10',
	lo = 'ref=,a shipped=lo_std_a,b refit=lo_std_b,c refit+split=lo_std_c,d relit+colour-aware+refit=lo_rlc_b,f lightmap 20 cm=lo_std_s20' )

DRIVER = r'''
import json, time
import urllib.request
url = %(url)r
mine = set(t["id"] for t in json.loads(urllib.request.urlopen("http://localhost:%(port)d/json/list", timeout=5).read()))
seen = [t for t in cdp("Target.getTargets").get("targetInfos", []) if t.get("type") == "page"]
print("TABS " + json.dumps([[t["targetId"][:8], t.get("url", "")[:120]] for t in seen]))
if not seen or any(t["targetId"] not in mine for t in seen):
    raise SystemExit("lsvb_sheet: this daemon is not on the Chrome of port %(port)d - nothing touched")
tid = None
for t in seen:
    if tid is None:
        tid = t["targetId"]
    elif not str(t.get("url", "")).startswith("about:"):
        try:
            cdp("Target.closeTarget", targetId=t["targetId"])
        except Exception:
            pass
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
    v = ev("JSON.stringify(window.__capture || null)")
    if isinstance(v, str):
        res = json.loads(v)
        if res and (res.get("done") or res.get("failed")):
            break
print("SHOOT " + json.dumps({"gpu": gpu, "res": res, "seconds": round(time.time() - t0, 1)}))
'''

def shoot( out, query, quality, w, h, timeout = 300 ):

	token = open( os.path.join( ROOT, 'out/.upload_token' ) ).read().strip()
	q = dict( capture = 1, token = token, quality = quality, w = w, h = h, out = out, cam = ';'.join( '%s:%s' % ( v[ 0 ], v[ 1 ] ) for v in VIEWS ) )
	url = 'http://127.0.0.1:8765/next/town/?' + urllib.parse.urlencode( q ) + ( '&' + query if query else '' )
	env = dict( os.environ, BU_NAME = 'lsvb', BU_CDP_URL = 'http://localhost:%d' % PORT )
	p = subprocess.run( [ 'browser-use' ], input = DRIVER % dict( url = url, timeout = timeout, port = PORT ), capture_output = True, text = True, env = env, cwd = ROOT )
	for line in p.stdout.splitlines():

		if line.startswith( 'SHOOT ' ):

			r = json.loads( line[ 6: ] ); res = r[ 'res' ] or {}
			if not res.get( 'failed' ): res.pop( 'log', None )
			if not res.get( 'done' ) or r[ 'gpu' ] != 'nvidia' or res.get( 'warnings' ): print( 'lsvb_sheet: look at this capture: %s' % json.dumps( r )[ :1500 ], flush = True )
			return r

	sys.exit( 'lsvb_sheet: no result\n' + p.stdout[ - 2000: ] + p.stderr[ - 2000: ] )

def stat_line( bundle ):

	fn = os.path.join( LRAW, 'stats.%s.json' % bundle )
	if not bundle or not os.path.exists( fn ): return ''
	st = json.load( open( fn ) ); a = st[ 'fit' ][ 'all' ]; b = ( st.get( 'fit_blur' ) or {} ).get( 'all' ); w = st[ 'fit' ].get( 'wall' ) or {}
	return 'rms %.1f  p95 %.1f%s%s  %dk verts' % ( a[ 'rms' ], a[ 'p95' ], '  25 cm rms %.1f' % b[ 'rms' ] if b else '', '  walls: rms %.1f bias %+.1f' % ( w[ 'rms' ], w[ 'bias' ] ) if 'bias' in w else '', round( st[ 'verts' ] / 1000 ) )

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--level', default = 'mid' )
	ap.add_argument( '--cols', default = '' )
	ap.add_argument( '--rows', default = 'lit,flat,heat' )
	ap.add_argument( '--capture', action = 'store_true' )
	ap.add_argument( '--only', default = '', help = 'capture only these columns (labels up to the first blank)' )
	ap.add_argument( '--quality', default = 'high' )
	ap.add_argument( '--w', type = int, default = 1280 ); ap.add_argument( '--h', type = int, default = 720 )
	ap.add_argument( '--tile', type = int, default = 640 )
	ap.add_argument( '--name', default = '' )
	ap.add_argument( '--views', default = '' )
	ap.add_argument( '--crop', default = '', help = 'the middle part of every capture at full size instead of all of it scaled down: WxH' )
	a = ap.parse_args()
	cols = [ c.split( '=', 1 ) for c in ( a.cols or COLS[ a.level ] ).split( ',' ) ]
	rows = a.rows.split( ',' )
	if a.views: VIEWS[ : ] = [ v for v in VIEWS if v[ 0 ] in a.views.split( ',' ) ]
	base = 'next/lsvb/%s' % a.level
	def cdir( label, row ): return '%s/%s_%s' % ( base, label.split( ' ' )[ 0 ], row )
	force = FORCE[ a.level ]
	if a.capture:

		log = {}
		lf = os.path.join( REN, base, 'captures.json' )
		if os.path.exists( lf ): log = json.load( open( lf ) )
		for label, bundle in cols:

			if a.only and label.split( ' ' )[ 0 ] not in a.only.split( ',' ): continue
			for row in rows:

				sfx, flat = ROWQ[ row ]
				if sfx and not bundle: continue
				if bundle and not os.path.exists( os.path.join( ROOT, 'out/web/_next/lm', bundle + sfx + '.json' ) ):

					print( 'lsvb_sheet: no bundle %s for %s' % ( bundle, row ), flush = True ); continue

				# (the reference: the top levels as far as the tier's distances go, no bundle)
				q = [ 'pr=1' ] + ( [ 'lm=%s' % ( bundle + sfx ), force ] if bundle else [] ) + ( [ 'flat=1' ] if flat else [] )
				r = shoot( cdir( label, row ), '&'.join( q ), a.quality, a.w, a.h )
				log[ cdir( label, row ) ] = dict( query = '&'.join( q ), gpu = r[ 'gpu' ], seconds = r[ 'seconds' ], res = r[ 'res' ] )
				print( 'lsvb_sheet: %s %s %.0fs' % ( cdir( label, row ), '&'.join( q ), r[ 'seconds' ] ), flush = True )
				os.makedirs( os.path.join( REN, base ), exist_ok = True )
				json.dump( log, open( lf, 'w' ), indent = 1 )

	# the sheet: a block of rows a view, a column a variant
	magick = shutil.which( 'magick' ) or sys.exit( 'lsvb_sheet: no magick' )
	tw = a.tile; th = round( a.tile * a.h / a.w )
	if a.crop: tw, th = [ int( v ) for v in a.crop.split( 'x' ) ]
	work = os.path.join( ROOT, '.scratch/lsvb/sheet_%s%s' % ( a.level, a.name ) ); os.makedirs( work, exist_ok = True )
	strips = []
	def run( args ):

		r = subprocess.run( [ magick ] + args, capture_output = True, text = True )
		if r.returncode: sys.exit( 'lsvb_sheet: magick: ' + r.stderr[ - 1500: ] )

	def text( s, w, h, size, out, bg = '#101014' ):

		run( [ '-size', '%dx%d' % ( w, h ), 'xc:' + bg, '-font', 'Adwaita-Sans', '-pointsize', str( size ), '-fill', '#e8e8e8', '-gravity', 'West', '-annotate', '+8+0', s, out ] )
		return out

	W = ( tw + 4 ) * len( cols )
	head = os.path.join( work, 'head.png' )
	args = []
	for i, ( label, bundle ) in enumerate( cols ):

		t = text( label + ( '\n' + bundle if bundle else '\nthe top levels as shipped' ) + '\n' + stat_line( bundle ), tw + 4, 66, 15, os.path.join( work, 'h%d.png' % i ) )
		args.append( t )

	run( args + [ '+append', head ] ); strips.append( head )
	for vi, ( view, cam, what ) in enumerate( VIEWS ):

		strips.append( text( '%s - %s   [level %s forced on every cell; camera %s]' % ( view, what, a.level, cam ), W, 26, 16, os.path.join( work, 'v%d.png' % vi ), '#26262e' ) )
		for row in rows:

			args = []
			for label, bundle in cols:

				fn = os.path.join( REN, cdir( label, row ), view + '.png' )
				if os.path.exists( fn ):

					args += [ '(', fn ] + ( [ '-gravity', 'Center', '-crop', '%dx%d+0+0' % ( tw, th ), '+repage' ] if a.crop else [ '-resize', '%dx%d!' % ( tw, th ) ] ) + [ '-bordercolor', '#101014', '-border', '2x2', ')' ]

				else: args += [ '(', '-size', '%dx%d' % ( tw + 4, th + 4 ), 'xc:#101014', ')' ]

			out = os.path.join( work, 'r_%d_%s.png' % ( vi, row ) )
			run( args + [ '+append', out ] ); strips.append( out )

	strips.append( text( 'rows of a view: %s.   lit = as the page shows it;  flat = the baked light alone on grey;  heat = what is left against the full model\'s light, face by face (rms of the points on it, display steps of 255): black 2 and less, blue 4, green 8, yellow 16, orange 32, red 64 and more;  sign = the mean of it signed: grey none, blue darker than the full model, red brighter (full colour at 32).\nNumbers in the head: all nine cells, every point that is in the open on the full model; bias = mean signed error on walls (below 0: darker).' % ', '.join( rows ),
		W, 48, 14, os.path.join( work, 'foot.png' ) ) )
	out = os.path.join( REN, 'next/lsvb/sheet_%s%s.png' % ( a.level, a.name ) )
	run( strips + [ '-append', out ] )
	shutil.rmtree( work )
	print( 'lsvb_sheet: %s' % out )

if __name__ == '__main__': main()
