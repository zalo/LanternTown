#!/usr/bin/env python3
"""What can be seen from where: the visibility sets of a baked town (pvs.json beside each manifest).

    python3 tools/pvs.py                       # out/web/_next: the full bake and its lean one
    python3 tools/pvs.py --dir _next_far       # a trial bake beside it (out/web/<name>, as the page's ?bakedir=)
    python3 tools/pvs.py --only lean           # one of the two
    python3 tools/pvs.py --check [--n 80]      # test the sets: every tier, from places off the grid (exit 1 on a miss
                                               # of more than --allow pixels; smaller ones are listed as slivers)
    python3 tools/pvs.py --off                 # take the sets out again (the page then draws all that is in view)
    python3 tools/pvs.py --holes [--limit 600] # where the reduced levels let the eye through: <bake>/.raw/holes.json,
                                               # which the packer mends them with (webbake.py --pack-only; then the sets again)

The town page does the work (web/engine/pvs.js, ?pvs=bake): in a GPU Chrome of its own it draws the bake in cell
colours from eye points all over the walkable ground, at every level of detail, and sends back per square of ground
the cells that show from there. This tool files that as <bake>/pvs.json and names it in the manifest ("pvs"), which
is what makes the page use it (BakedTown._occluded of web/engine/lite.js: a cell that is not in the set of the
camera's square is not drawn). tools/webbake.py writes a new manifest without it, so sets never outlive their bake:
run this after every bake. Needs tools/serve.py running (as the render tools do).
"""
import os, sys, json, time, argparse, subprocess

ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

DRIVER = r'''
import json, time
url = %(url)r
tid = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
sid = cdp("Target.attachToTarget", targetId=tid, flatten=True)["sessionId"]
def ev(expr, wait=False):
    try:
        r = cdp("Runtime.evaluate", session_id=sid, expression=expr, returnByValue=True, awaitPromise=wait)
    except Exception as e:
        return None
    return (r.get("result") or {}).get("value")
cdp("Page.enable", session_id=sid)
# (a freshly launched Chrome can answer requestAdapter() with null for its first seconds: wait on a plain page)
cdp("Page.navigate", session_id=sid, url=url.split("/town/")[0] + "/gallery/")
for _ in range(40):
    time.sleep(0.5)
    if ev("(async () => (await navigator.gpu?.requestAdapter())?.info.vendor || null)()", True): break
cdp("Page.navigate", session_id=sid, url=url)
t0, res, said = time.time(), None, None
while time.time() - t0 < %(timeout)d:
    time.sleep(2.0)
    v = ev("JSON.stringify(window.__pvs || null)")
    st = json.loads(v) if isinstance(v, str) else None
    if st and st.get("done"):
        res = st
        break
    if st and st.get("stage") != said:
        said = st.get("stage")
        print("PVS_STAGE " + str(said), flush=True)
if res is None:
    res = {"done": False, "error": "timeout", "state": ev("JSON.stringify(window.__pvs || null)"), "log": ev("JSON.stringify((window.__log || []).slice(-12))")}
try: cdp("Target.closeTarget", targetId=tid)
except Exception: pass
print("PVS_RESULT " + json.dumps(res))
'''


def page( query, timeout, keep ):
	"""Open the town page with this query in the tool's own GPU Chrome -> window.__pvs when it is done."""
	url = 'http://127.0.0.1:%s/town/?%s' % ( os.environ.get( 'TSQ_PORT', '8765' ), query )
	port = os.environ.get( 'TSQ_TOWN_CHROME_PORT', '9224' )
	cenv = dict( os.environ, TSQ_CHROME_PORT = port, TSQ_CHROME_PROFILE = os.environ.get( 'TSQ_TOWN_CHROME_PROFILE' ) or os.path.join( ROOT, '.scratch', 'chrome', 'tsq-chrome-' + port ) )
	chrome = os.path.join( ROOT, 'tools', 'gpu_chrome.sh' )
	try:
		subprocess.run( [ chrome ], env = cenv, check = True, capture_output = True )
		env = dict( os.environ, BU_NAME = 'tsqpvs' + port, BU_CDP_URL = 'http://localhost:' + port )
		p = subprocess.run( [ 'browser-use' ], input = DRIVER % dict( url = url, timeout = timeout ), capture_output = True, text = True, env = env, timeout = timeout + 90 )
	finally:
		if not keep: subprocess.run( [ chrome, 'stop' ], env = cenv, capture_output = True )
	for line in p.stdout.splitlines():
		if line.startswith( 'PVS_RESULT ' ): return json.loads( line[ 11: ] )
	return dict( done = False, error = ( p.stdout + p.stderr )[ -2000: ] )


def main():
	ap = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
	ap.add_argument( '--dir', default = '_next', help = 'the bake: a folder of out/web (default _next)' )
	ap.add_argument( '--only', choices = ( 'both', 'full', 'lean' ), default = 'both' )
	ap.add_argument( '--check', action = 'store_true', help = 'test the sets that are there, in every tier' )
	ap.add_argument( '--off', action = 'store_true', help = 'remove the sets' )
	ap.add_argument( '--holes', action = 'store_true', help = 'the census of holes in the reduced levels (holesPVS of web/engine/pvs.js) -> <bake>/.raw/holes.json' )
	ap.add_argument( '--limit', type = int, default = 600, help = '--holes: eye points looked from' )
	ap.add_argument( '--levels', default = '', help = '--holes: only these levels (lo,mid,near,ring)' )
	ap.add_argument( '--name', default = 'holes.json', help = '--holes: the file name (another than holes.json: a census to compare, the packer does not read it)' )
	ap.add_argument( '--n', type = int, default = 80, help = '--check: places per tier' )
	ap.add_argument( '--seed', type = int, default = 1 )
	ap.add_argument( '--tiers', default = '', help = '--check: only these (high,medium,low,phone)' )
	ap.add_argument( '--allow', type = int, default = 8, help = '--check: a cell left out that shows in no more pixels than this (of the 512 x 512 x 6 around a place) is listed as a sliver, not failed on' )
	ap.add_argument( '--face', type = int, default = 512, help = 'pixels across a face of the cube an eye point looks through' )
	ap.add_argument( '--step', type = float, default = 0.5, help = 'grid of the eye points (m)' )
	ap.add_argument( '--square', type = float, default = 1.0, help = 'the squares of ground the sets are kept for (m)' )
	ap.add_argument( '--pad', type = float, default = 1.5, help = 'how far outside its square an eye point still counts for it (m)' )
	ap.add_argument( '--eyes', default = '1.45,1.65,1.85', help = 'eye heights over the ground the town is looked at from (m)' )
	ap.add_argument( '--near', type = float, default = 8.0, help = 'cells this close to a square are always in its set (m)' )
	ap.add_argument( '--timeout', type = int, default = 3600 )
	ap.add_argument( '--keep', action = 'store_true', help = 'leave the Chrome running' )
	a = ap.parse_args()
	base = os.path.join( ROOT, 'out', 'web', a.dir )
	if not os.path.isfile( os.path.join( base, 'manifest.json' ) ): sys.exit( 'pvs: no bake in ' + base )
	bakedir = '' if a.dir == '_next' else '&bakedir=' + a.dir
	token = open( os.path.join( ROOT, 'out', '.upload_token' ) ).read().strip()
	full = json.load( open( os.path.join( base, 'manifest.json' ) ) )
	# (which: [ the folder, the tier whose page loads it ])
	which = [ w for w in ( [ 'full', base, 'high' ], [ 'lean', os.path.join( base, str( full.get( 'lean' ) or 'lean' ) ), 'low' ] )
		if a.only in ( 'both', w[ 0 ] ) and os.path.isfile( os.path.join( w[ 1 ], 'manifest.json' ) ) ]

	if a.off:
		for name, d, tier in which:
			mp = os.path.join( d, 'manifest.json' )
			man = json.load( open( mp ) )
			if man.pop( 'pvs', None ) is not None: json.dump( man, open( mp, 'w' ), separators = ( ',', ':' ) )
			if os.path.isfile( os.path.join( d, 'pvs.json' ) ): os.remove( os.path.join( d, 'pvs.json' ) )
			print( 'pvs: %s: off' % name )
		return 0

	if a.holes:
		for name, d, tier in which:
			out = 'holes_%s_%s' % ( a.dir.strip( '_' ), name )
			got = os.path.join( ROOT, 'out', 'renders', '_town', out, 'holes.json' )
			if os.path.isfile( got ): os.remove( got )
			r = page( 'quality=%s&pvs=holes&out=%s&token=%s&limit=%d&face=%d%s%s' % ( tier, out, token, a.limit, a.face, '&levels=' + a.levels if a.levels else '', bakedir ), a.timeout, a.keep )
			if r.get( 'failed' ) or not r.get( 'path' ) or not os.path.isfile( got ):
				print( 'pvs: %s: FAILED: %s' % ( name, json.dumps( r )[ :3000 ] ) )
				return 1
			h = json.load( open( got ) )
			if h.get( 'baked' ) != json.load( open( os.path.join( d, 'manifest.json' ) ) ).get( 'baked' ):
				print( 'pvs: %s: the page looked at another bake than %s' % ( name, d ) )
				return 1
			to = os.path.join( d, '.raw' if os.path.isdir( os.path.join( d, '.raw' ) ) else '', a.name )
			os.replace( got, to )
			os.rmdir( os.path.dirname( got ) )
			print( 'pvs: %s: holes against the %s level from %d eye points, %.0f s -> %s' % ( name, h[ 'top' ], h[ 'points' ], h[ 'seconds' ], os.path.relpath( to, ROOT ) ) )
			for l, v in h[ 'levels' ].items():
				cells = sorted( v[ 'cells' ].items(), key = lambda kv: - kv[ 1 ][ 'n' ] )
				print( '        %-5s %7.1f pixels in a million show the sky or another cell well behind (%d of %d; %.1f with any surface further behind), %d places in %d cells; most: %s' % (
					l, 1e6 * v.get( 'through', v[ 'pixels' ] ) / max( 1, v[ 'of' ] ), v.get( 'through', v[ 'pixels' ] ), v[ 'of' ], 1e6 * v[ 'pixels' ] / max( 1, v[ 'of' ] ), sum( len( c[ 'pts' ] ) // 3 for k, c in cells ), len( cells ),
					', '.join( '%s %d' % ( k, c[ 'n' ] ) for k, c in cells[ :6 ] ) ) )
		return 0

	if a.check:
		bad = 0
		for tier in [ t for t in ( [ 'high', 'medium' ] if a.only != 'lean' else [] ) + ( [ 'low', 'phone' ] if a.only != 'full' else [] ) if not a.tiers or t in a.tiers.split( ',' ) ]:
			r = page( 'quality=%s&pvs=check&n=%d&seed=%d&face=512%s' % ( tier, a.n, a.seed, bakedir ), a.timeout, a.keep )
			c = r.get( 'check' )
			if not c or c.get( 'error' ):
				print( 'pvs: %s: FAILED: %s' % ( tier, json.dumps( r )[ :1500 ] ) ); bad += 1; continue
			print( 'pvs: %-6s %d places (%d without a set): %.1f of %d cells left out on average, %.0f %% of the triangles the page holds for the place; %d cells seen that the sets leave out (largest: %d of %d pixels), %.0f s' % (
				tier, c[ 'points' ], c[ 'no_set' ], c[ 'hidden' ] / max( 1, c[ 'points' ] - c[ 'no_set' ] ), c[ 'cells' ], 100 * c[ 'hidden_tris' ] / max( 1, c[ 'drawn_tris' ] ),
				len( c[ 'misses' ] ), c[ 'worst' ], c[ 'pixels' ], c[ 'seconds' ] ) )
			for m in c[ 'misses' ][ :12 ]: print( '        ' + ( 'sliver ' if m[ 'pixels' ] <= a.allow else 'MISS   ' ) + json.dumps( m ) )
			bad += sum( 1 for m in c[ 'misses' ] if m[ 'pixels' ] > a.allow )
		return 1 if bad else 0

	for name, d, tier in which:
		t0 = time.time()
		out = 'pvs_%s_%s' % ( a.dir.strip( '_' ), name )
		got = os.path.join( ROOT, 'out', 'renders', '_town', out, 'pvs.json' )
		if os.path.isfile( got ): os.remove( got )
		r = page( 'quality=%s&pvs=bake&out=%s&token=%s&face=%d&step=%g&square=%g&pad=%g&near=%g&eyes=%s%s' % ( tier, out, token, a.face, a.step, a.square, a.pad, a.near, a.eyes, bakedir ), a.timeout, a.keep )
		if r.get( 'failed' ) or not r.get( 'path' ) or not os.path.isfile( got ):
			print( 'pvs: %s: FAILED: %s' % ( name, json.dumps( r )[ :3000 ] ) )
			return 1
		mp = os.path.join( d, 'manifest.json' )
		man, pvs = json.load( open( mp ) ), json.load( open( got ) )
		if pvs.get( 'baked' ) != man.get( 'baked' ) or pvs.get( 'keys' ) != [ c[ 'k' ] for c in man[ 'chunks' ] ]:
			print( 'pvs: %s: the page looked at another bake than %s (lean without a folder of its own?)' % ( name, d ) )
			return 1
		os.replace( got, os.path.join( d, 'pvs.json' ) )
		os.rmdir( os.path.dirname( got ) )
		s = pvs[ 'stats' ]
		man[ 'pvs' ] = dict( file = 'pvs.json', v = int( time.time() ), squares = s[ 'squares' ], seen = s[ 'seen' ] )
		json.dump( man, open( mp, 'w' ), separators = ( ',', ':' ) )
		print( 'pvs: %s: %d eye points x %d heights x %d levels and %d tiers\' mixes of them, %d squares of %g m; from a square %.0f %% of the %d cells show on average (%d from nowhere), %d kB, %.0f s' % (
			name, s[ 'points' ], len( s[ 'eyes' ] ), len( s[ 'levels' ] ) - len( s.get( 'bands' ) or [] ), len( s.get( 'bands' ) or [] ), s[ 'squares' ], a.square, 100 * s[ 'seen' ], s[ 'cells' ], s[ 'never' ], os.path.getsize( os.path.join( d, 'pvs.json' ) ) // 1024, time.time() - t0 ) )
		if s.get( 'stray' ): print( 'pvs: %s: %d runs of pixels in a colour that is no cell (the cell colours did not come back as they went in?)' % ( name, s[ 'stray' ] ) )
	return 0


if __name__ == '__main__':
	sys.exit( main() )
