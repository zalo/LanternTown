#!/usr/bin/env python3
"""Contact occlusion of the ground of a baked town: <bake>/gao.png, named in the manifest ("gao") - and, beside it,
the lanterns' shadows on that ground: <bake>/gsh.webp ("gsh").

    python3 tools/aomap.py                     # out/web/_next (and its lean bake: the same file beside its manifest)
    python3 tools/aomap.py --dir _next_m       # a trial bake beside it (out/web/<name>, as the page's ?bakedir=)
    python3 tools/aomap.py --off               # take it out again
    python3 tools/aomap.py --texel 0.0625 --radius 1.0 --rays 48 --lift 0.05 --jobs 12 --level top

The bake's ambient occlusion is one number per vertex, and the ground has a vertex every 0.6 m at best: the dark rim
where a barrel, a stall or a wall stands on the paving falls between them. This is that rim, as a picture of the
town's ground from above - one texel every --texel metres, read by the ground's own pixels at their place in the
world (BakedTown._materials of web/engine/lite.js): no geometry, no draw, nothing transparent to sort or to fight
for depth. Per texel, from the bake's working data (<bake>/.raw):

  the terrain's height there (the triangles of ground vertices, the topmost), and the floor on it: the first surface
  under a point --floor above the terrain (a slab, a kerb, a laid floor - or the terrain itself);
  --rays rays over the hemisphere above the floor (cosine-weighted, turned per texel), --radius long, from --lift
  above it (the joints of modelled setts are below that and stay out of it): a ray that ends counts the more the
  sooner it ends. A texel inside something - under a crate, in a wall - comes out black, which is what runs a dark
  line along every foot.

red: the open share (255: nothing near); green: the terrain's height (h0 + green x hs), by which a pixel knows it is
of the ground and not of a table top above it; blue: 255 where there is terrain. tools/webbake.py runs this as its
last step when the working data is kept (--keep-raw / --pack-only; --no-gao: not); every bake writes a new manifest
without it, so after a bake that kept none, run it by hand once there is some. A quarter of a minute.

The shadows (gsh.webp) are not made here: the bake itself lights the ground's texels (ground_shadows() of
tools/webbake_bl.py, BK gsm*; <bake>/.raw/gsm_v.bin, gsm_f.bin, gsm.json) and gives the ground's vertices the lanterns'
light WITHOUT their shadows - the page takes it away again per pixel by this picture, so a bake made with it needs it
(--off leaves its ground without lantern shadows; --set gsm=0 bakes them into the vertices as before). Here the two
bytes a texel are laid on the grid of gao.png (red: sqrt of the share of the lanterns' light that arrives, green: sqrt
of the share of the texel's light that is not a lantern's) and stored without loss; the lean bake gets every second
texel, as it reads gao.png. The page loads both into one texture.
"""
import os, sys, json, time, math, shutil, subprocess

ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
OPT = dict( texel = 0.0625, radius = 1.0, rays = 48, lift = 0.05, floor = 0.2, jobs = 12, level = 'top' )

def find_blender():

	for c in ( os.environ.get( 'TSQ_BLENDER' ), os.environ.get( 'BLENDER' ), shutil.which( 'blender' ), os.path.expanduser( '~/.local/bin/blender' ) ):
		if c and os.path.exists( c ): return c
	sys.exit( 'aomap: blender not found (set TSQ_BLENDER=/path/to/blender)' )

def manifests( base ):

	out = [ base ]
	man = json.load( open( os.path.join( base, 'manifest.json' ) ) )
	if isinstance( man.get( 'lean' ), str ) and os.path.isfile( os.path.join( base, man[ 'lean' ], 'manifest.json' ) ): out.append( os.path.join( base, man[ 'lean' ] ) )
	return out

def shadow_map( raw, work, g, dirs ):

	# the bake's shadow picture (ground_shadows() of tools/webbake_bl.py) on the grid of gao.png -> the manifest's
	# "gsh", or None when the bake made none; dirs: the bake and its lean one (half the size there)
	jp = os.path.join( raw, 'gsm.json' )
	if not os.path.isfile( jp ): return None
	m = json.load( open( jp ) )
	s = g[ 'texel' ]
	if abs( m[ 'texel' ] - s ) > 1e-9:

		print( 'aomap: the shadow picture has texels of %.4f m, the occlusion map of %.4f: no shadow map (bake with --set gsm_texel=%s, or --texel)' % ( m[ 'texel' ], s, s ), flush = True )
		return None

	W, H, w, h = g[ 'w' ], g[ 'h' ], m[ 'w' ], m[ 'h' ]
	ox = int( round( ( m[ 'x0' ] - g[ 'x0' ] ) / s ) ); oz = int( round( ( m[ 'z0' ] - g[ 'z0' ] ) / s ) )
	left = max( ox, 0 ); src = max( - ox, 0 ); n = max( 0, min( w - src, W - left ) )
	out = bytearray( W * H * 3 )
	for c, fn in enumerate( ( 'gsm_v.bin', 'gsm_f.bin' ) ):

		b = open( os.path.join( raw, fn ), 'rb' ).read()
		if len( b ) != w * h: sys.exit( 'aomap: %s is not %d x %d' % ( fn, w, h ) )
		# (where the picture has nothing: all of the light arrives, none of it a lantern's)
		a = bytearray( b'\xff' * ( W * H ) )
		for j in range( H ):

			jj = j - oz
			if 0 <= jj < h and n: a[ j * W + left:j * W + left + n ] = b[ jj * w + src:jj * w + src + n ]

		out[ c::3 ] = a

	ppm = os.path.join( work, 'gsh.ppm' )
	with open( ppm, 'wb' ) as f: f.write( b'P6\n%d %d\n255\n' % ( W, H ) ); f.write( out )
	del out
	rec = { k: m[ k ] for k in ( 'band', 'cF', 'cD', 'leak', 'rays', 'lit', 'shadowed', 'mean' ) }
	rec.update( file = 'gsh.webp', v = int( time.time() ) )
	res = []
	for i, d in enumerate( dirs ):

		webp = os.path.join( d, 'gsh.webp' ); half = i > 0
		size = ( ( W + 1 ) // 2, ( H + 1 ) // 2 ) if half else ( W, H )
		cmd = [ shutil.which( 'magick' ) or 'magick', ppm ] + ( [ '-filter', 'Box', '-resize', '%dx%d!' % size ] if half else [] ) + [ '-define', 'webp:lossless=true', '-define', 'webp:method=6', webp ]
		r = subprocess.run( cmd )
		if r.returncode != 0 or not os.path.isfile( webp ): sys.exit( 'aomap: no shadow picture (ImageMagick)' )
		res.append( dict( rec, w = size[ 0 ], h = size[ 1 ], bytes = os.path.getsize( webp ) ) )

	return res

def main():

	import argparse
	ap = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
	ap.add_argument( '--dir', default = '_next' )
	ap.add_argument( '--off', action = 'store_true' )
	for k, v in OPT.items(): ap.add_argument( '--' + k, type = type( v ), default = v )
	a = ap.parse_args()
	base = os.path.join( ROOT, 'out/web', a.dir )
	if not os.path.isfile( os.path.join( base, 'manifest.json' ) ): sys.exit( 'aomap: no bake in ' + base )
	if a.off:

		for d in manifests( base ):

			mp = os.path.join( d, 'manifest.json' ); man = json.load( open( mp ) )
			g = man.pop( 'gao', None ); sh = man.pop( 'gsh', None )
			if g is not None or sh is not None: json.dump( man, open( mp, 'w' ), separators = ( ',', ':' ) )
			for q in ( g, sh ):

				if q and os.path.isfile( os.path.join( d, q[ 'file' ] ) ): os.remove( os.path.join( d, q[ 'file' ] ) )

			if sh: print( 'aomap: %s was baked for its shadow map: its ground has no lantern shadows now' % os.path.relpath( d, ROOT ) )

		print( 'aomap: off' ); return

	raw = os.path.join( base, '.raw' )
	if not os.path.isfile( os.path.join( raw, 'raw.json' ) ): sys.exit( 'aomap: the bake has no working data (.raw: bake with --keep-raw)' )
	opt = { k: getattr( a, k ) for k in OPT }
	work = os.path.join( raw, 'gao_work' ); os.makedirs( work, exist_ok = True )
	t0 = time.time()
	r = subprocess.run( [ find_blender(), '--background', '--factory-startup', '--python', os.path.abspath( __file__ ), '--', raw, work, json.dumps( opt ) ] )
	info = os.path.join( work, 'gao.json' )
	if r.returncode != 0 or not os.path.isfile( info ): sys.exit( 'aomap: failed' )
	g = json.load( open( info ) )
	png = os.path.join( base, 'gao.png' )
	r = subprocess.run( [ shutil.which( 'magick' ) or 'magick', os.path.join( work, 'gao.ppm' ), '-define', 'png:compression-level=9', 'PNG24:' + png ] )
	if r.returncode != 0 or not os.path.isfile( png ): sys.exit( 'aomap: no picture (ImageMagick)' )
	g[ 'file' ] = 'gao.png'; g[ 'v' ] = int( time.time() ); g[ 'bytes' ] = os.path.getsize( png )
	dirs = manifests( base )
	sh = shadow_map( raw, work, g, dirs )
	for i, d in enumerate( dirs ):

		if d != base: shutil.copyfile( png, os.path.join( d, 'gao.png' ) )
		mp = os.path.join( d, 'manifest.json' ); man = json.load( open( mp ) )
		man[ 'gao' ] = g
		old = man.pop( 'gsh', None )
		if sh: man[ 'gsh' ] = sh[ i ]
		elif old and os.path.isfile( os.path.join( d, old[ 'file' ] ) ): os.remove( os.path.join( d, old[ 'file' ] ) )
		json.dump( man, open( mp, 'w' ), separators = ( ',', ':' ) )

	shutil.rmtree( work )
	if sh: print( 'aomap: %s: the lanterns\' shadows: %.0f %% of the ground in their light, %.2f of it arrives; %.2f MB%s' % ( a.dir, 100 * sh[ 0 ][ 'lit' ], sh[ 0 ][ 'mean' ], sh[ 0 ][ 'bytes' ] / 1e6, ( ' (lean: %.2f MB)' % ( sh[ 1 ][ 'bytes' ] / 1e6 ) ) if len( sh ) > 1 else '' ) )
	print( 'aomap: %s: %d x %d texels of %.4f m, %.0f %% of the ground with something near, %.2f MB, %.0fs' % ( a.dir, g[ 'w' ], g[ 'h' ], g[ 'texel' ], 100 * g[ 'near' ], g[ 'bytes' ] / 1e6, time.time() - t0 ) )

# ---------------------------------------------------------------------------------------- what follows runs in Blender

T0 = time.time()
def log( *a ): print( '[aomap %6.1fs]' % ( time.time() - T0 ), *a, flush = True )

def make_bvh( P, TRI ):

	# (as tools/webbake_bl.py)
	import bpy
	from mathutils.bvhtree import BVHTree
	used, inv = np.unique( TRI.ravel(), return_inverse = True )
	co = P[ used ].astype( np.float32 )
	tri = inv.reshape( - 1, 3 ).astype( np.int32 )
	me = bpy.data.meshes.new( 'occ' )
	me.vertices.add( len( co ) ); me.vertices.foreach_set( 'co', co.ravel() )
	me.loops.add( len( tri ) * 3 ); me.loops.foreach_set( 'vertex_index', tri.ravel() )
	me.polygons.add( len( tri ) ); me.polygons.foreach_set( 'loop_start', np.arange( 0, len( tri ) * 3, 3, dtype = np.int32 ) )
	me.update( calc_edges = True )
	ob = bpy.data.objects.new( 'occ', me )
	bpy.context.scene.collection.objects.link( ob )
	dg = bpy.context.evaluated_depsgraph_get()
	bvh = BVHTree.FromObject( ob, dg )
	bpy.data.objects.remove( ob ); bpy.data.meshes.remove( me )
	return bvh

def hemi( n, turns ):

	# cosine-weighted, stratified, about +y: `turns` copies, each turned a little further -> lists of tuples
	g = ( math.sqrt( 5 ) - 1 ) / 2
	out = []
	for t in range( turns ):

		a0 = 2 * math.pi * t / turns; d = []
		for i in range( n ):

			u = ( i + ( ( t * 7 ) % turns + 0.5 ) / turns ) / n; ph = a0 + 2 * math.pi * ( ( i * g ) % 1 )
			r = math.sqrt( u ); d.append( ( r * math.cos( ph ), math.sqrt( max( 1 - u, 0 ) ), r * math.sin( ph ) ) )

		out.append( d )

	return out

def rows( j0, j1, G, gb, ab, nb, VIS, HT, NEAR ):

	W = G[ 'w' ]; x0 = G[ 'x0' ]; z0 = G[ 'z0' ]; s = G[ 'texel' ]; R = G[ 'radius' ]; lift = G[ 'lift' ]; fl = G[ 'floor' ]
	H = hemi( G[ 'rays' ], 64 ); n = G[ 'rays' ]
	down = ( 0.0, - 1.0, 0.0 )
	gr = gb.ray_cast; ar = ab.ray_cast; near = nb.find_nearest
	for j in range( j0, j1 ):

		z = z0 + ( j + 0.5 ) * s
		vis = np.full( W, 255, np.uint8 ); ht = np.full( W, np.nan, np.float32 ); nr = 0
		for i in range( W ):

			x = x0 + ( i + 0.5 ) * s
			r = gr( ( x, 60.0, z ), down )
			if r[ 0 ] is None: continue
			h0 = r[ 0 ][ 1 ]; ht[ i ] = h0
			r = ar( ( x, h0 + fl, z ), down, fl + 0.12 )
			o = ( x, ( r[ 0 ][ 1 ] if r[ 0 ] is not None else h0 ) + lift, z )
			if near( o, R )[ 0 ] is None: continue
			nr += 1
			occ = 0.0
			for d in H[ ( i * 37 + j * 101 + ( i * j ) % 13 ) & 63 ]:

				r = ar( o, d, R )
				if r[ 0 ] is not None: occ += 1.0 - r[ 3 ] / R

			vis[ i ] = int( 255 * ( 1 - occ / n ) + 0.5 )

		VIS[ j ] = vis; HT[ j ] = ht; NEAR[ j ] = nr

def bake( raw, work, opt ):

	man = json.load( open( os.path.join( raw, 'raw.json' ) ) )
	Ps = []; Ts = []; Gs = []; nv0 = 0
	for c in man[ 'chunks' ]:

		lv = [ l for l in ( ( 'fine', 'near', 'mid' ) if opt[ 'level' ] == 'top' else ( opt[ 'level' ], 'near', 'mid' ) ) if l in c ][ 0 ]
		m = c[ lv ]; w = m.get( 'w', 10 ); nv, ni = m[ 'nv' ], m[ 'ni' ]
		buf = np.fromfile( os.path.join( raw, m[ 'raw' ] ), np.uint8 )
		rec = buf[ :nv * w * 2 ].view( '<u2' ).reshape( nv, w )
		idx = buf[ nv * w * 2:nv * w * 2 + ni * 4 ].view( '<u4' )
		flag = buf[ nv * w * 2 + ni * 4:nv * w * 2 + ni * 4 + nv ]
		tri = np.concatenate( [ idx[ s[ 1 ]:s[ 1 ] + s[ 2 ] ] for s in m[ 'sub' ] if s[ 0 ] == 0 ] + [ np.zeros( 0, '<u4' ) ] ).reshape( - 1, 3 ).astype( np.int64 )
		Ps.append( ( np.array( m[ 'min' ], np.float64 )[ None ] + rec[ :, :3 ].astype( np.float64 ) * m[ 'q' ] ).astype( np.float32 ) )
		Ts.append( tri + nv0 ); Gs.append( ( flag[ tri ] & 4 ).all( 1 ) ); nv0 += nv

	P = np.concatenate( Ps ); TRI = np.concatenate( Ts ); gnd = np.concatenate( Gs )
	del Ps, Ts, Gs
	# (slivers of no area make a tree no better)
	a = np.linalg.norm( np.cross( P[ TRI[ :, 1 ] ] - P[ TRI[ :, 0 ] ], P[ TRI[ :, 2 ] ] - P[ TRI[ :, 0 ] ] ), axis = 1 )
	TRI = TRI[ a > 1e-10 ]; gnd = gnd[ a > 1e-10 ]
	log( '%d cells, %d opaque triangles, %d of the ground' % ( len( man[ 'chunks' ] ), len( TRI ), int( gnd.sum() ) ) )
	Q = P[ TRI[ gnd ] ].reshape( - 1, 3 )
	s = float( opt[ 'texel' ] )
	x0 = math.floor( Q[ :, 0 ].min() / s ) * s; z0 = math.floor( Q[ :, 2 ].min() / s ) * s
	W = int( math.ceil( ( Q[ :, 0 ].max() - x0 ) / s ) ); Hh = int( math.ceil( ( Q[ :, 2 ].max() - z0 ) / s ) )
	G = dict( opt, w = W, h = Hh, x0 = x0, z0 = z0, sx = W * s, sz = Hh * s )
	gb = make_bvh( P, TRI[ gnd ] ); nb = make_bvh( P, TRI[ ~ gnd ] ); ab = make_bvh( P, TRI )
	log( 'trees; %d x %d texels' % ( W, Hh ) )
	del P, TRI
	VIS = np.memmap( os.path.join( work, 'vis.bin' ), np.uint8, 'w+', shape = ( Hh, W ) )
	HT = np.memmap( os.path.join( work, 'ht.bin' ), np.float32, 'w+', shape = ( Hh, W ) )
	NEAR = np.memmap( os.path.join( work, 'near.bin' ), np.int32, 'w+', shape = ( Hh, ) )
	jobs = max( 1, int( opt[ 'jobs' ] ) ); step = 8
	sl = [ ( j, min( Hh, j + step ) ) for j in range( 0, Hh, step ) ]
	pids = []
	for k in range( jobs ):

		pid = os.fork() if jobs > 1 else 0
		if pid == 0:

			code = 0
			try:

				for q in range( k, len( sl ), jobs ): rows( sl[ q ][ 0 ], sl[ q ][ 1 ], G, gb, ab, nb, VIS, HT, NEAR )
				VIS.flush(); HT.flush(); NEAR.flush()

			except BaseException:

				import traceback; traceback.print_exc(); code = 1

			if jobs > 1: os._exit( code )

		else: pids.append( pid )

	bad = 0
	for pid in pids:

		_, st = os.waitpid( pid, 0 )
		if st != 0: bad += 1

	if bad: raise RuntimeError( '%d workers failed' % bad )
	vis = np.array( VIS, np.float32 ); ht = np.array( HT ); has = np.isfinite( ht )
	log( 'rays: %d texels of ground, %d with something near' % ( int( has.sum() ), int( NEAR.sum() ) ) )
	# the rays' noise: a 3 x 3 binomial (the texture filter does the rest)
	p = np.pad( vis, 1, mode = 'edge' )
	p = ( p[ :, :- 2 ] + 2 * p[ :, 1:- 1 ] + p[ :, 2: ] ) / 4
	vis = ( p[ :- 2 ] + 2 * p[ 1:- 1 ] + p[ 2: ] ) / 4
	# the height where there is no terrain: its neighbours', spread outwards (a filtered read at the rim is then theirs)
	h0 = float( np.nanmin( ht ) ) - 0.25; h1 = float( np.nanmax( ht ) ) + 0.25
	hh = np.where( has, ht, 0 ).astype( np.float32 ); hw = has.astype( np.float32 )
	for it in range( 6 ):

		ps = np.pad( hh * hw, 1 ); pw = np.pad( hw, 1 )
		ns = ps[ :- 2, 1:- 1 ] + ps[ 2:, 1:- 1 ] + ps[ 1:- 1, :- 2 ] + ps[ 1:- 1, 2: ]; nw = pw[ :- 2, 1:- 1 ] + pw[ 2:, 1:- 1 ] + pw[ 1:- 1, :- 2 ] + pw[ 1:- 1, 2: ]
		grow = ( hw == 0 ) & ( nw > 0 )
		hh[ grow ] = ns[ grow ] / nw[ grow ]; hw[ grow ] = 1

	hh[ hw == 0 ] = h0
	out = np.zeros( ( Hh, W, 3 ), np.uint8 )
	out[ :, :, 0 ] = np.clip( vis + 0.5, 0, 255 ).astype( np.uint8 )
	out[ :, :, 1 ] = np.clip( ( hh - h0 ) / ( h1 - h0 ) * 255 + 0.5, 0, 255 ).astype( np.uint8 )
	out[ :, :, 2 ] = has * 255
	with open( os.path.join( work, 'gao.ppm' ), 'wb' ) as f: f.write( b'P6\n%d %d\n255\n' % ( W, Hh ) ); f.write( out.tobytes() )
	del VIS, HT
	G = { k: G[ k ] for k in ( 'w', 'h', 'x0', 'z0', 'sx', 'sz', 'texel', 'radius', 'rays', 'lift', 'floor' ) }
	G.update( h0 = round( h0, 4 ), hs = round( h1 - h0, 4 ), near = round( float( NEAR.sum() ) / max( int( has.sum() ), 1 ), 4 ), mean = round( float( vis[ has ].mean() ) / 255, 4 ) )
	del NEAR
	for n in ( 'vis.bin', 'ht.bin', 'near.bin' ): os.remove( os.path.join( work, n ) )
	json.dump( G, open( os.path.join( work, 'gao.json' ), 'w' ) )
	log( 'done: open share %.3f on average' % G[ 'mean' ] )

try: import bpy
except ImportError: bpy = None

if bpy is not None and '--' in sys.argv:

	import numpy as np
	try: bake( sys.argv[ sys.argv.index( '--' ) + 1 ], sys.argv[ sys.argv.index( '--' ) + 2 ], json.loads( sys.argv[ sys.argv.index( '--' ) + 3 ] ) )
	except BaseException:

		import traceback; traceback.print_exc(); sys.exit( 1 )

elif __name__ == '__main__': main()
