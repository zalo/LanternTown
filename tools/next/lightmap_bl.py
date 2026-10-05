# Lightmap prototype, the Blender step (numpy + mathutils), driven by tools/next/lightmap.py:
#
#   blender --background --factory-startup --python tools/next/lightmap_bl.py -- <config.json>
#
# The baked light of a few cells in texture pages instead of vertex colours, to compare the two (not a rollout).
#
# mode "bake": for every mesh the node stage "dump" left in <lm>/.raw (a level of a prototype cell, simplified as the
#   packer does it or harder) and every bundle of the config (a texel size):
#   - unwrap(): charts = faces joined over welded edges that share their dominant axis; a chart is projected straight
#     along that axis onto a WORLD lattice of the texel size (texel centres at whole multiples of it), so the paving or
#     a wall carries the same texels on both sides of a cell border, and the lattices of 5 / 10 / 20 cm nest
#   - every texel is ss x ss samples; a sample is a point of the low mesh (or, in the gutter around a chart, the plane
#     of its nearest face carried on) which is moved onto the surface of the FULL model (the fine soup, LOD0) - the
#     nearest triangle of the same material that faces the same way - and lit there by run_lighting() + final_colour()
#     of the bake itself. The light in a texel is therefore one function of the place, whatever level the mesh is.
#   - a sample that sits inside something (a ray along the surface meets a back face within a texel) does not count:
#     the texels at the foot of a wall take the light in front of it, not the dark behind it
#   - the lit colour (rgb) and the gloss (a: where the wet film lies) go to <lm>/.raw/<bundle>.npz, the meshes with
#     uv2 in their colour slots to <lm>/.raw/<mesh>.<bundle>.bin
#   "dry": no fine soup - every sample takes the low mesh's own vertex light (checks the unwrap and the page alone)
# mode "encode": the texels of a bundle as RGBA8 pages (rgb = ( light / range ) ^ ( 1 / gamma ), dithered; a = gloss)
#
#   a bundle of the kind "fit" keeps the light in the vertices (tools/next/lsvb_fit.py, part of this file): the low mesh
#   is sampled over its faces, the samples lit on the full model as above (kept in <lm>/.raw/smp.*.npz: mode "bake"
#   with `sample`), and the vertex colours are the least-squares fit to them (Kavan, Bargteil, Sloan 2011) - mode "fit",
#   which needs neither the full model nor a ray. What is left on each face goes out as a false-colour mesh (_res).
#
# Glowing surfaces (windows, signs, lantern mantles), glass, water, falls and leaf cards keep their vertex light.
# Reads out/town/town.json and out/glb for minutes (mode "bake"): the driver holds the BAKING marker meanwhile.

import sys, os, json, time, math
import numpy as np

_CFG = json.load( open( sys.argv[ sys.argv.index( '--' ) + 1 ] ) )
LM = _CFG[ 'lm' ]
LDIR = LM[ 'dir' ]; LRAW = os.path.join( LDIR, '.raw' )
PAGE = int( LM.get( 'page', 1024 ) )
DRY = bool( LM.get( 'dry' ) )

# (refit: the split bundles again from the samples the last bake left - no full model, no rays, nothing read of out/glb)
HAVE_FINE = LM[ 'mode' ] == 'bake' and not DRY and not LM.get( 'refit' )
LUM = np.array( [ 0.2126, 0.7152, 0.0722 ] )

if HAVE_FINE:

	_BAKE = os.path.join( _CFG[ 'root' ], 'tools/next/webbake_bl.py' )
	exec( compile( open( _BAKE ).read(), _BAKE, 'exec' ) )

else:

	T0 = time.time()
	K_OPAQUE = 0
	JOBS = int( _CFG.get( 'jobs', 8 ) )
	def log( *a ): print( '[lightmap %6.1fs]' % ( time.time() - T0 ), *a, flush = True )

def llog( *a ): print( '[lightmap %6.1fs]' % ( time.time() - T0 ), *a, flush = True )

UA = np.array( [ 2, 0, 0 ] ); VA = np.array( [ 1, 2, 1 ] ) # the two axes of a chart's plane, by its dominant axis

# ------------------------------------------------------------------------------------------ meshes

def load_mesh( m, path = None ):

	b = open( path or os.path.join( LRAW, m[ 'raw' ] ), 'rb' ).read()
	nv, ni = m[ 'nv' ], m[ 'ni' ]
	rec = np.frombuffer( b, '<u2', nv * 10 ).reshape( nv, 10 ).copy()
	tri = np.frombuffer( b, '<u4', ni, nv * 20 ).reshape( - 1, 3 ).astype( np.int64 )
	flags = np.frombuffer( b, np.uint8, nv, nv * 20 + ni * 4 ).copy()
	return dict( m, rec = rec, tri = tri, flags = flags )

def mesh_pos( M ):

	return np.array( M[ 'min' ], np.float64 )[ None ] + M[ 'rec' ][ :, :3 ].astype( np.float64 ) * M[ 'q' ]

def vertex_light( rec ):

	c = rec[ :, 6:9 ].astype( np.float64 ) / 65535
	return c * c * 64, np.minimum( rec[ :, 9 ].astype( np.float64 ) / 32767, 1 )

# ------------------------------------------------------------------------------------------ unwrap + samples

def unwrap( M, B ):

	s = float( B[ 's' ] ); ss = int( B.get( 'ss', 1 ) ); g = 1
	rec, tri = M[ 'rec' ], M[ 'tri' ]
	nv, nt = len( rec ), len( tri )
	P = mesh_pos( M )
	opq = np.zeros( nt, bool )
	for kind, start, count in M[ 'sub' ]:

		if kind == K_OPAQUE: opq[ start // 3:( start + count ) // 3 ] = True

	glow = rec[ :, 9 ] == 65535
	fn = np.cross( P[ tri[ :, 1 ] ] - P[ tri[ :, 0 ] ], P[ tri[ :, 2 ] ] - P[ tri[ :, 0 ] ] )
	a2 = np.linalg.norm( fn, axis = 1 )
	lmf = opq & ~ glow[ tri ].any( 1 ) & ( a2 > 1e-7 )
	nrm = fn / np.maximum( a2, 1e-20 )[ :, None ]
	ax = np.abs( nrm ).argmax( 1 )
	cls = ax * 2 + ( nrm[ np.arange( nt ), ax ] < 0 )
	key = rec[ :, 0 ].astype( np.int64 ) | ( rec[ :, 1 ].astype( np.int64 ) << 16 ) | ( rec[ :, 2 ].astype( np.int64 ) << 32 )
	_, weld = np.unique( key, return_inverse = True ); NW = int( weld.max() ) + 1 if nv else 1

	def charts_of( f ):

		# -> chart of each face of f, whether each of its edges is on the outline of its chart
		nf = len( f )
		wt = weld[ tri[ f ] ]
		e0 = wt.ravel(); e1 = wt[ :, [ 1, 2, 0 ] ].ravel()
		ek = ( np.minimum( e0, e1 ) * NW + np.maximum( e0, e1 ) ) * 6 + np.repeat( cls[ f ], 3 )
		ef = np.repeat( np.arange( nf ), 3 )
		o = np.argsort( ek, kind = 'stable' ); eks = ek[ o ]; efs = ef[ o ]
		same = eks[ 1: ] == eks[ :- 1 ]
		par = list( range( nf ) )
		def find( x ):

			while par[ x ] != x: par[ x ] = par[ par[ x ] ]; x = par[ x ]
			return x

		for x, y in zip( efs[ :- 1 ][ same ].tolist(), efs[ 1: ][ same ].tolist() ):

			rx, ry = find( x ), find( y )
			if rx != ry: par[ ry ] = rx

		root = np.array( [ find( x ) for x in range( nf ) ], np.int64 ) if nf else np.zeros( 0, np.int64 )
		_, chart = np.unique( root, return_inverse = True )
		_, inv, cnt = np.unique( ek, return_inverse = True, return_counts = True )
		return chart, ( cnt[ inv ] == 1 ).reshape( nf, 3 )

	f = np.nonzero( lmf )[ 0 ]
	chart, bedge = charts_of( f )
	nc = int( chart.max() ) + 1 if len( f ) else 0
	if B.get( 'min_area', 0 ) > 0 and nc:

		# only the large surfaces (paving, walls): the rest keeps its vertex light
		big = np.bincount( chart, a2[ f ] * 0.5, nc ) >= B[ 'min_area' ]
		f = f[ big[ chart ] ]
		chart, bedge = charts_of( f )
		nc = int( chart.max() ) + 1 if len( f ) else 0

	nf = len( f )
	fa = ax[ f ]
	PF = P[ tri[ f ] ] # nf, 3, 3
	ar = np.arange( nf )
	pu = PF[ ar[ :, None ], np.arange( 3 )[ None, : ], UA[ fa ][ :, None ] ]; pv = PF[ ar[ :, None ], np.arange( 3 )[ None, : ], VA[ fa ][ :, None ] ]
	# texel size of a chart: coarser on what lies above the eaves, and where a chart would not fit a page
	ymin = np.full( nc, 1e9 ); np.minimum.at( ymin, chart, PF[ :, :, 1 ].min( 1 ) )
	sc = np.where( ymin >= B.get( 'roof_h', 6.0 ), s * B.get( 'roof_mul', 1 ), s )
	def extent( a ):

		lo = np.full( nc, 1e18 ); hi = np.full( nc, - 1e18 )
		np.minimum.at( lo, chart, a.min( 1 ) ); np.maximum.at( hi, chart, a.max( 1 ) )
		return lo, hi

	u0, u1 = extent( pu ); v0, v1 = extent( pv )
	for _ in range( 6 ):

		big = np.maximum( u1 - u0, v1 - v0 ) / sc > PAGE - 24
		if not big.any(): break
		sc[ big ] *= 2

	def axis( lo, hi ):

		# -> origin (texels), node count, collapsed: a chart thinner than half a texel is one node across, at its middle
		lo = lo / sc; hi = hi / sc
		thin = hi - lo < 0.5
		o = np.where( thin, ( lo + hi ) * 0.5, np.floor( lo + 1e-4 ) )
		n = np.where( thin, 1, np.ceil( hi - 1e-4 ) - np.floor( lo + 1e-4 ) + 1 ).astype( np.int64 )
		return o, n, thin

	ou, nu, tu = axis( u0, u1 ); ov, nvv, tv = axis( v0, v1 )
	gc = np.where( tu & tv, 0, g ).astype( np.int64 )
	w = nu + 2 * gc; h = nvv + 2 * gc
	Wc = w * ss; Hc = h * ss
	base = np.concatenate( [ [ 0 ], np.cumsum( Wc * Hc ) ] ).astype( np.int64 )
	tbase = np.concatenate( [ [ 0 ], np.cumsum( w * h ) ] ).astype( np.int64 )
	NS = int( base[ - 1 ] )
	# the faces on the sub-sample lattice of their chart: sample m of a texel row sits at texel ( m + 0.5 ) / ss - 0.5
	fc = chart
	def sub( p, o, thin ):

		t = np.where( thin[ fc ][ :, None ], 0.0, p / sc[ fc ][ :, None ] - o[ fc ][ :, None ] ) + gc[ fc ][ :, None ]
		return t, ( t + 0.5 ) * ss - 0.5

	tU, mU = sub( pu, ou, tu ); tV, mV = sub( pv, ov, tv )
	# (a collapsed axis: the samples still spread over the face, about its middle)
	mUr = np.where( tu[ fc ][ :, None ], ( pu / sc[ fc ][ :, None ] - ou[ fc ][ :, None ] + gc[ fc ][ :, None ] + 0.5 ) * ss - 0.5, mU )
	mVr = np.where( tv[ fc ][ :, None ], ( pv / sc[ fc ][ :, None ] - ov[ fc ][ :, None ] + gc[ fc ][ :, None ] + 0.5 ) * ss - 0.5, mV )
	edge_face = bedge.any( 1 ) | tu[ fc ] | tv[ fc ]
	pad = np.where( edge_face, ( g + 0.5 ) * ss, 0.02 )
	lo = np.stack( [ np.floor( mUr.min( 1 ) - pad ), np.floor( mVr.min( 1 ) - pad ) ], 1 ).astype( np.int64 )
	hi = np.stack( [ np.ceil( mUr.max( 1 ) + pad ), np.ceil( mVr.max( 1 ) + pad ) ], 1 ).astype( np.int64 )
	lo = np.maximum( lo, 0 ); hi = np.minimum( hi, np.stack( [ Wc[ fc ] - 1, Hc[ fc ] - 1 ], 1 ) )
	dims = hi - lo + 1
	best_s = np.full( NS, - 1e9, np.float32 ); best_f = np.full( NS, - 1, np.int32 ); best_b = np.zeros( ( NS, 2 ), np.float32 )
	over = 0
	ok = dims.min( 1 ) > 0
	kb = np.ceil( np.log2( np.maximum( dims.max( 1 ), 1 ) ) ).astype( np.int64 )
	p0 = np.stack( [ mUr[ :, 0 ], mVr[ :, 0 ] ], 1 ); p1 = np.stack( [ mUr[ :, 1 ], mVr[ :, 1 ] ], 1 ); p2 = np.stack( [ mUr[ :, 2 ], mVr[ :, 2 ] ], 1 )
	A2 = ( p1[ :, 0 ] - p0[ :, 0 ] ) * ( p2[ :, 1 ] - p0[ :, 1 ] ) - ( p1[ :, 1 ] - p0[ :, 1 ] ) * ( p2[ :, 0 ] - p0[ :, 0 ] )
	ok &= np.abs( A2 ) > 1e-9
	for e in np.unique( kb[ ok ] ) if ok.any() else []:

		idx = np.nonzero( ok & ( kb == e ) )[ 0 ]
		k = 1 << int( e )
		gy, gx = np.mgrid[ 0:k, 0:k ]; gx = gx.ravel()[ None, : ]; gy = gy.ravel()[ None, : ]
		step = max( 1, 1200000 // ( k * k ) )
		for i0 in range( 0, len( idx ), step ):

			fi = idx[ i0:i0 + step ]
			x = lo[ fi, 0 ][ :, None ] + gx; y = lo[ fi, 1 ][ :, None ] + gy
			inb = ( x <= hi[ fi, 0 ][ :, None ] ) & ( y <= hi[ fi, 1 ][ :, None ] )
			q0, q1, q2 = p0[ fi ], p1[ fi ], p2[ fi ]
			sg = np.sign( A2[ fi ] )
			def dist( a, b ):

				ex = b[ :, 0 ] - a[ :, 0 ]; ey = b[ :, 1 ] - a[ :, 1 ]
				return ( ex[ :, None ] * ( y - a[ :, 1 ][ :, None ] ) - ey[ :, None ] * ( x - a[ :, 0 ][ :, None ] ) ) * ( sg / np.maximum( np.hypot( ex, ey ), 1e-12 ) )[ :, None ]

			d = np.minimum( np.minimum( dist( q0, q1 ), dist( q1, q2 ) ), dist( q2, q0 ) )
			keep = inb & ( d >= - pad[ fi ][ :, None ] )
			r, c = np.nonzero( keep )
			if not len( r ): continue
			xx = x[ r, c ].astype( np.float64 ); yy = y[ r, c ].astype( np.float64 ); fr = fi[ r ]
			b1 = ( ( xx - p0[ fr, 0 ] ) * ( p2[ fr, 1 ] - p0[ fr, 1 ] ) - ( yy - p0[ fr, 1 ] ) * ( p2[ fr, 0 ] - p0[ fr, 0 ] ) ) / A2[ fr ]
			b2 = ( ( p1[ fr, 0 ] - p0[ fr, 0 ] ) * ( yy - p0[ fr, 1 ] ) - ( p1[ fr, 1 ] - p0[ fr, 1 ] ) * ( xx - p0[ fr, 0 ] ) ) / A2[ fr ]
			sid = base[ fc[ fr ] ] + y[ r, c ] * Wc[ fc[ fr ] ] + x[ r, c ]
			sco = d[ r, c ].astype( np.float32 )
			o = np.lexsort( ( - sco, sid ) ); sid = sid[ o ]
			first = np.ones( len( sid ), bool ); first[ 1: ] = sid[ 1: ] != sid[ :- 1 ]
			# (two faces of one chart over the same sample, both well inside: the chart folds over itself)
			over += int( ( ~ first & ( sco[ o ] > 0.3 ) ).sum() )
			o = o[ first ]; sid = sid[ first ]
			better = sco[ o ] > best_s[ sid ]
			o = o[ better ]; sid = sid[ better ]
			best_s[ sid ] = sco[ o ]; best_f[ sid ] = fr[ o ]; best_b[ sid, 0 ] = b1[ o ]; best_b[ sid, 1 ] = b2[ o ]

	act = np.nonzero( best_f >= 0 )[ 0 ]
	sf = best_f[ act ].astype( np.int64 ); b1 = best_b[ act, 0 ].astype( np.float64 ); b2 = best_b[ act, 1 ].astype( np.float64 )
	covered = best_s[ act ] >= - 0.02
	X = PF[ sf, 0 ] * ( 1 - b1 - b2 )[ :, None ] + PF[ sf, 1 ] * b1[ :, None ] + PF[ sf, 2 ] * b2[ :, None ]
	# the texel of each sample
	sc_ = fc[ sf ]; rel = act - base[ sc_ ]
	my = rel // Wc[ sc_ ]; mx = rel - my * Wc[ sc_ ]
	tex = tbase[ sc_ ] + ( my // ss ) * w[ sc_ ] + ( mx // ss )
	# the mesh with its vertices split by chart
	cid = np.full( nt, - 1, np.int64 ); cid[ f ] = chart
	pair = tri * ( nc + 1 ) + ( cid[ :, None ] + 1 )
	up, inv = np.unique( pair.ravel(), return_inverse = True )
	old = up // ( nc + 1 ); vch = up % ( nc + 1 ) - 1
	lmv = vch >= 0
	vc = vch[ lmv ]; Pv = P[ old[ lmv ] ]
	cax = np.zeros( nc, np.int64 ); cax[ chart ] = fa
	pu_v = Pv[ np.arange( len( vc ) ), UA[ cax[ vc ] ] ]; pv_v = Pv[ np.arange( len( vc ) ), VA[ cax[ vc ] ] ]
	tu_v = np.where( tu[ vc ], 0.0, pu_v / sc[ vc ] - ou[ vc ] ) + gc[ vc ]
	tv_v = np.where( tv[ vc ], 0.0, pv_v / sc[ vc ] - ov[ vc ] ) + gc[ vc ]
	# low mesh's own light at the samples (the fallback, and all there is in a dry run)
	lit, gl = vertex_light( rec )
	c1 = np.clip( b1, 0, 1 ); c2 = np.clip( b2, 0, 1 ); c0 = np.clip( 1 - c1 - c2, 0, 1 ); cs = np.maximum( c0 + c1 + c2, 1e-9 )
	tv3 = tri[ f ][ sf ]
	own = ( lit[ tv3[ :, 0 ] ] * c0[ :, None ] + lit[ tv3[ :, 1 ] ] * c1[ :, None ] + lit[ tv3[ :, 2 ] ] * c2[ :, None ] ) / cs[ :, None ]
	owng = ( gl[ tv3[ :, 0 ] ] * c0 + gl[ tv3[ :, 1 ] ] * c1 + gl[ tv3[ :, 2 ] ] * c2 ) / cs
	NT = int( tbase[ - 1 ] )
	# texel -> its chart and place in it
	tch = np.repeat( np.arange( nc ), w * h ); trel = np.arange( NT ) - tbase[ tch ]
	ty = trel // w[ tch ]; tx = trel - ty * w[ tch ]
	return dict( s = s, ss = ss, nc = nc, w = w, h = h, sc = sc, area = float( a2[ f ].sum() * 0.5 ), area_all = float( a2[ opq ].sum() * 0.5 ), faces = nf, faces_opaque = int( opq.sum() ), over = over,
		X = X, N = nrm[ f ][ sf ], LAY = rec[ tv3[ :, 0 ], 3 ].astype( np.int64 ), L = sc[ sc_ ], covered = covered, tex = tex, own = own, owng = owng,
		NT = NT, tch = tch, tx = tx, ty = ty, tw = w[ tch ], th = h[ tch ],
		new_tri = inv.reshape( - 1, 3 ), old = old, lmv = lmv, vc = vc, tu = tu_v, tv = tv_v )

def texels( U, C, G, tier ):

	# samples -> texels: the mean of the best kind of sample a texel has (2: on the full model and in the open,
	# 1: inside something or not found there), then the texels without a good sample take their good neighbours' mean
	NT = U[ 'NT' ]; tex = U[ 'tex' ]
	val = np.zeros( ( NT, 4 ), np.float64 ); tt = np.zeros( NT, np.int8 )
	for k in ( 1, 2 ):

		m = tier == k
		if not m.any(): continue
		cnt = np.bincount( tex[ m ], minlength = NT )
		has = cnt > 0
		acc = np.stack( [ np.bincount( tex[ m ], C[ m, j ], NT ) for j in range( 3 ) ] + [ np.bincount( tex[ m ], G[ m ], NT ) ], 1 )
		val[ has ] = acc[ has ] / cnt[ has ][ :, None ]; tt[ has ] = k

	idx = np.arange( NT ); tx, ty, tw, th = U[ 'tx' ], U[ 'ty' ], U[ 'tw' ], U[ 'th' ]
	filled = 0
	for it in range( 2 ):

		need = np.nonzero( tt < 2 )[ 0 ]
		if not len( need ): break
		acc = np.zeros( ( len( need ), 4 ) ); cnt = np.zeros( len( need ) )
		for dx in ( - 1, 0, 1 ):

			for dy in ( - 1, 0, 1 ):

				if not dx and not dy: continue
				x = tx[ need ] + dx; y = ty[ need ] + dy
				ok = ( x >= 0 ) & ( x < tw[ need ] ) & ( y >= 0 ) & ( y < th[ need ] )
				j = need[ ok ] + dx + dy * tw[ need ][ ok ]
				good = tt[ j ] == 2
				k = np.nonzero( ok )[ 0 ][ good ]
				acc[ k ] += val[ j[ good ] ]; cnt[ k ] += 1

		got = cnt > 0
		val[ need[ got ] ] = acc[ got ] / cnt[ got ][ :, None ]; tt[ need[ got ] ] = 3
		filled += int( got.sum() )
		tt[ tt == 3 ] = 2

	return val, tt, filled

# ------------------------------------------------------------------------------------------ the full model

FINE = {}

def fine_setup():

	cells = setup()
	S = build_soup( 0, cells ); _GLB.clear()
	llog( 'fine soup: %d tris, %d verts' % ( len( S.t[ 'TRI' ] ), len( S.v[ 'P' ] ) ) )
	bvh = light_soup( S, 0 )
	for k in ( 'UV', 'GRP', 'PROP' ): del S.v[ k ]
	V, Tt = S.v, S.t
	sel = np.nonzero( Tt[ 'KIND' ] == K_OPAQUE )[ 0 ]
	T = Tt[ 'TRI' ][ sel ]
	sel = sel[ V[ 'EMF' ][ T ].max( 1 ) < 0.5 ] # (never onto a glowing surface)
	T = Tt[ 'TRI' ][ sel ]
	cen = ( V[ 'P' ][ T[ :, 0 ] ] + V[ 'P' ][ T[ :, 1 ] ] + V[ 'P' ][ T[ :, 2 ] ] ) / 3
	for k in ( 'OCC', 'CELL', 'TERR', 'KIND' ): del Tt[ k ]
	FINE.update( S = S, bvh = bvh, sel = sel, cen = cen, cache = {} )
	llog( 'fine surface to project on: %d triangles' % len( sel ) )

def fine_near( key, lo, hi ):

	# the full model's opaque triangles about a cell, as a tree of their own
	c = FINE[ 'cache' ]
	if key in c: return c[ key ]
	c.clear() # (one at a time: memory)
	cen = FINE[ 'cen' ]
	m = ( cen >= ( np.array( lo ) - 1.5 )[ None ] ).all( 1 ) & ( cen <= ( np.array( hi ) + 1.5 )[ None ] ).all( 1 )
	sel = FINE[ 'sel' ][ m ]
	V = FINE[ 'S' ].v; T = FINE[ 'S' ].t[ 'TRI' ][ sel ]
	P0 = V[ 'P' ][ T[ :, 0 ] ].astype( np.float64 ); P1 = V[ 'P' ][ T[ :, 1 ] ].astype( np.float64 ); P2 = V[ 'P' ][ T[ :, 2 ] ].astype( np.float64 )
	fn = np.cross( P1 - P0, P2 - P0 ); fn /= np.maximum( np.linalg.norm( fn, axis = 1 ), 1e-20 )[ :, None ]
	up = np.where( np.abs( fn[ :, 1:2 ] ) < 0.9, np.array( [ [ 0.0, 1.0, 0.0 ] ] ), np.array( [ [ 1.0, 0.0, 0.0 ] ] ) )
	t1 = np.cross( up, fn ); t1 /= np.maximum( np.linalg.norm( t1, axis = 1 ), 1e-20 )[ :, None ]
	t2 = np.cross( fn, t1 )
	t0 = time.time()
	# (make_bvh() keeps the order of the triangles it is given: its polygon i is sel[ i ])
	c[ key ] = dict( bvh = make_bvh( V[ 'P' ], T ), sel = sel, T = T, FN = fn.tolist(), T1 = t1.tolist(), T2 = t2.tolist(), LAY = V[ 'LAY' ][ T[ :, 0 ] ].tolist() )
	llog( '  %s: %d triangles of the full model, tree %.1fs' % ( key, len( sel ), time.time() - t0 ) )
	return c[ key ]

def project( pb, X, N, LAY, L, R ):

	# every sample onto the full model -> ( triangle of pb or -1, point x y z, 1 = in the open )
	near = pb[ 'bvh' ].find_nearest; rng = pb[ 'bvh' ].find_nearest_range; rc = FINE[ 'bvh' ].ray_cast
	FN, T1, T2, LT = pb[ 'FN' ], pb[ 'T1' ], pb[ 'T2' ], pb[ 'LAY' ]
	def fn( s, e ):

		out = np.zeros( ( e - s, 5 ), np.float64 ); out[ :, 0 ] = - 1
		xs = X[ s:e ].tolist(); ns = N[ s:e ].tolist(); ls = LAY[ s:e ].tolist(); Ls = L[ s:e ].tolist()
		for i in range( e - s ):

			x = xs[ i ]; n = ns[ i ]; la = ls[ i ]
			r = near( x, R )
			if r[ 0 ] is None: continue
			t = - 1
			j = r[ 2 ]; f = FN[ j ]
			if LT[ j ] == la and f[ 0 ] * n[ 0 ] + f[ 1 ] * n[ 1 ] + f[ 2 ] * n[ 2 ] > 0.3: t = j; loc = r[ 0 ]
			else:

				bd = 1e9
				for c in rng( x, R ):

					j = c[ 2 ]
					if LT[ j ] != la: continue
					f = FN[ j ]; d = f[ 0 ] * n[ 0 ] + f[ 1 ] * n[ 1 ] + f[ 2 ] * n[ 2 ]
					if d <= 0: continue
					m = c[ 3 ] + 0.25 * R * ( 1 - d ) # nearer, and facing more the same way
					if m < bd: bd = m; t = j; loc = c[ 0 ]

			if t < 0: continue
			f = FN[ t ]
			o = ( loc[ 0 ] + f[ 0 ] * 0.01, loc[ 1 ] + f[ 1 ] * 0.01, loc[ 2 ] + f[ 2 ] * 0.01 )
			valid = 1.0; reach = 1.2 * Ls[ i ]
			a = T1[ t ]; b = T2[ t ]
			for d in ( a, ( - a[ 0 ], - a[ 1 ], - a[ 2 ] ), b, ( - b[ 0 ], - b[ 1 ], - b[ 2 ] ) ):

				hh = rc( o, d, reach )
				if hh[ 0 ] is not None:

					hn = hh[ 1 ]
					if hn[ 0 ] * d[ 0 ] + hn[ 1 ] * d[ 1 ] + hn[ 2 ] * d[ 2 ] > 0.1: valid = 0.0; break

			out[ i ] = ( t, loc[ 0 ], loc[ 1 ], loc[ 2 ], valid )

		return out

	return parallel( fn, len( X ), np.float64, ( 5, ), 4096, 'proj' )

def free_rays( pb, pr, K, far ):

	# of K rays over the half space of each projected point (rows of project() that hit): how many go far[ j ] metres
	# free or leave the town - a ray that ends on the back of a face does not count however far it went (last column:
	# how many did end on a back)
	rc = FINE[ 'bvh' ].ray_cast
	FN, T1, T2 = pb[ 'FN' ], pb[ 'T1' ], pb[ 'T2' ]
	dirs = []
	for k in range( K ):

		u = ( k + 0.5 ) / K; r = math.sqrt( u ); ph = k * 2.399963229728653
		dirs.append( ( r * math.cos( ph ), r * math.sin( ph ), math.sqrt( 1 - u ) ) )

	def fn( s, e ):

		out = np.zeros( ( e - s, len( far ) + 1 ), np.uint8 ); nb = len( far )
		ts = pr[ s:e, 0 ].astype( np.int64 ).tolist(); ps = pr[ s:e, 1:4 ].tolist()
		for i in range( e - s ):

			t = ts[ i ]; f = FN[ t ]; a = T1[ t ]; b = T2[ t ]; p = ps[ i ]
			o = ( p[ 0 ] + f[ 0 ] * 0.01, p[ 1 ] + f[ 1 ] * 0.01, p[ 2 ] + f[ 2 ] * 0.01 )
			n = [ 0 ] * ( nb + 1 )
			for lx, ly, lz in dirs:

				d = ( a[ 0 ] * lx + b[ 0 ] * ly + f[ 0 ] * lz, a[ 1 ] * lx + b[ 1 ] * ly + f[ 1 ] * lz, a[ 2 ] * lx + b[ 2 ] * ly + f[ 2 ] * lz )
				hh = rc( o, d, 40.0 )
				if hh[ 0 ] is None: dist = 1e9
				else:

					hn = hh[ 1 ]
					if hn[ 0 ] * d[ 0 ] + hn[ 1 ] * d[ 1 ] + hn[ 2 ] * d[ 2 ] >= 0: n[ nb ] += 1; continue
					dist = hh[ 3 ]

				for j in range( nb ):

					if dist >= far[ j ]: n[ j ] += 1

			out[ i ] = n

		return out

	return parallel( fn, len( pr ), np.uint8, ( len( far ) + 1, ), 4096, 'vis' )

class _Fake: pass

def fine_light( pb, pr, tag ):

	# the bake's light at the projected points -> colour (n, 3), gloss (n); pr: rows of project() that hit
	V = FINE[ 'S' ].v
	T = pb[ 'T' ][ pr[ :, 0 ].astype( np.int64 ) ]
	p = pr[ :, 1:4 ]
	a = V[ 'P' ][ T[ :, 0 ] ].astype( np.float64 ); b = V[ 'P' ][ T[ :, 1 ] ].astype( np.float64 ); c = V[ 'P' ][ T[ :, 2 ] ].astype( np.float64 )
	v0 = b - a; v1 = c - a; v2 = p - a
	d00 = ( v0 * v0 ).sum( 1 ); d01 = ( v0 * v1 ).sum( 1 ); d11 = ( v1 * v1 ).sum( 1 ); d20 = ( v2 * v0 ).sum( 1 ); d21 = ( v2 * v1 ).sum( 1 )
	den = np.maximum( d00 * d11 - d01 * d01, 1e-30 )
	w1 = np.clip( ( d11 * d20 - d01 * d21 ) / den, 0, 1 ); w2 = np.clip( ( d00 * d21 - d01 * d20 ) / den, 0, 1 )
	w0 = np.clip( 1 - w1 - w2, 0, 1 ); ws = np.maximum( w0 + w1 + w2, 1e-9 )
	w0 /= ws; w1 /= ws; w2 /= ws
	def at( k ):

		x = V[ k ]
		if x.ndim == 1: return ( x[ T[ :, 0 ] ] * w0 + x[ T[ :, 1 ] ] * w1 + x[ T[ :, 2 ] ] * w2 ).astype( np.float32 )
		return ( x[ T[ :, 0 ] ] * w0[ :, None ] + x[ T[ :, 1 ] ] * w1[ :, None ] + x[ T[ :, 2 ] ] * w2[ :, None ] ).astype( np.float32 )

	F = _Fake()
	N = at( 'N' ); N /= np.maximum( np.linalg.norm( N, axis = 1 ), 1e-9 )[ :, None ]
	F.v = dict( P = p.astype( np.float32 ), N = N.astype( np.float32 ), VAO = at( 'VAO' ), FSKY = at( 'FSKY' ), MUL = at( 'MUL' ), GLOW = at( 'GLOW' ), EM = at( 'EM' ), LIT0 = V[ 'LIT0' ][ T[ :, 0 ] ] )
	fol = np.ascontiguousarray( V[ 'FOL' ][ T[ :, 0 ] ] )
	need = ( F.v[ 'LIT0' ] == 0 ) & ( F.v[ 'MUL' ].max( 1 ) > 0 )
	E, A = run_lighting( FINE[ 'bvh' ], F.v[ 'P' ], F.v[ 'N' ], fol, need, tag )
	return final_colour( F, E, A ).astype( np.float64 ), np.clip( at( 'AUX' ), 0, 1 ).astype( np.float64 )

# ------------------------------------------------------------------------------------------ bundles

def shelf_pack( w, h ):

	# charts on shelves, tallest first -> page, x, y of each
	n = len( w )
	order = np.argsort( - h, kind = 'stable' )
	page = np.zeros( n, np.int64 ); px = np.zeros( n, np.int64 ); py = np.zeros( n, np.int64 )
	p = x = y = sh = 0
	wl = w.tolist(); hl = h.tolist()
	for c in order.tolist():

		if x + wl[ c ] > PAGE: y += sh; x = 0; sh = 0
		if y + hl[ c ] > PAGE: p += 1; x = y = sh = 0
		page[ c ] = p; px[ c ] = x; py[ c ] = y
		x += wl[ c ]; sh = max( sh, hl[ c ] )

	return page, px, py, p + 1 if n else 0

def bake_bundle( B, meshes ):

	t00 = time.time()
	res = []
	for m in meshes:

		t0 = time.time()
		M = load_mesh( m )
		U = unwrap( M, B )
		n = len( U[ 'X' ] )
		t1 = time.time()
		st = dict( charts = U[ 'nc' ], texels = U[ 'NT' ], samples = n, faces = U[ 'faces' ], faces_opaque = U[ 'faces_opaque' ], area = round( U[ 'area' ], 1 ), area_opaque = round( U[ 'area_all' ], 1 ), fold = U[ 'over' ] )
		if DRY:

			C = U[ 'own' ]; G = U[ 'owng' ]; tier = np.where( U[ 'covered' ], 2, 1 ).astype( np.int8 )

		else:

			P = mesh_pos( M )
			pb = fine_near( m[ 'key' ], P.min( 0 ), P.max( 0 ) )
			R = float( B.get( 'radius', { 'lo': 0.6, 'mid': 0.35, 'near': 0.25 }[ m[ 'level' ] ] ) )
			pr = project( pb, U[ 'X' ], U[ 'N' ], U[ 'LAY' ], U[ 'L' ], R )
			hit = pr[ :, 0 ] >= 0
			C = U[ 'own' ].copy(); G = U[ 'owng' ].copy()
			tier = np.where( U[ 'covered' ], 1, 0 ).astype( np.int8 ) # not found: the mesh's own light, where it is a point of the mesh
			if hit.any():

				c, gl = fine_light( pb, pr[ hit ], 'lm' )
				C[ hit ] = c; G[ hit ] = gl
				tier[ hit ] = np.where( pr[ hit, 4 ] > 0.5, 2, 1 )

			cov = U[ 'covered' ]
			st.update( missed = round( float( ( ~ hit & cov ).sum() ) / max( 1, int( cov.sum() ) ), 4 ), inside = round( float( ( hit & ( pr[ :, 4 ] < 0.5 ) & cov ).sum() ) / max( 1, int( cov.sum() ) ), 4 ),
				moved = round( float( np.linalg.norm( pr[ hit, 1:4 ] - U[ 'X' ][ hit ], axis = 1 ).mean() ) if hit.any() else 0, 4 ) )

		val, tt, filled = texels( U, C, G, tier )
		st.update( texels_own = int( ( np.bincount( U[ 'tex' ][ U[ 'covered' ] ], minlength = U[ 'NT' ] ) > 0 ).sum() ), filled = filled, verts_in = len( M[ 'rec' ] ), verts = len( U[ 'old' ] ), unwrap_s = round( t1 - t0, 1 ), light_s = round( time.time() - t1, 1 ) )
		llog( '  %s %s %s @ %s: %s' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ], B[ 'name' ], json.dumps( st ) ) )
		res.append( dict( m = m, M = M, U = U, val = val.astype( np.float32 ), tt = tt, st = st ) )

	# all charts of the bundle on its pages
	w = np.concatenate( [ r[ 'U' ][ 'w' ] for r in res ] ); h = np.concatenate( [ r[ 'U' ][ 'h' ] for r in res ] )
	page, px, py, npages = shelf_pack( w, h )
	c0 = 0; out = []; T = dict( val = [], page = [], x = [], y = [], tt = [] )
	for r in res:

		U, M, m = r[ 'U' ], r[ 'M' ], r[ 'm' ]
		nc = U[ 'nc' ]
		cp, cx, cy = page[ c0:c0 + nc ], px[ c0:c0 + nc ], py[ c0:c0 + nc ]; c0 += nc
		T[ 'val' ].append( r[ 'val' ] ); T[ 'tt' ].append( r[ 'tt' ] ); T[ 'page' ].append( cp[ U[ 'tch' ] ] ); T[ 'x' ].append( cx[ U[ 'tch' ] ] + U[ 'tx' ] ); T[ 'y' ].append( cy[ U[ 'tch' ] ] + U[ 'ty' ] )
		rec = M[ 'rec' ][ U[ 'old' ] ].copy()
		lv = np.nonzero( U[ 'lmv' ] )[ 0 ]; vc = U[ 'vc' ]
		rec[ lv, 3 ] |= 0x8000
		rec[ lv, 6 ] = np.clip( np.round( ( cx[ vc ] + U[ 'tu' ] + 0.5 ) / PAGE * 65535 ), 0, 65535 )
		rec[ lv, 7 ] = np.clip( np.round( ( cy[ vc ] + U[ 'tv' ] + 0.5 ) / PAGE * 65535 ), 0, 65535 )
		rec[ lv, 8 ] = cp[ vc ] * 256 + 128
		fn = '%s.%s.%s.%s.bin' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ], B[ 'name' ] )
		with open( os.path.join( LRAW, fn ), 'wb' ) as f:

			f.write( rec.astype( '<u2' ).tobytes() ); f.write( U[ 'new_tri' ].astype( '<u4' ).tobytes() ); f.write( M[ 'flags' ][ U[ 'old' ] ].tobytes() )

		out.append( dict( key = m[ 'key' ], level = m[ 'level' ], variant = m[ 'variant' ], raw = fn, min = m[ 'min' ], q = m[ 'q' ], nv = int( len( rec ) ), ni = int( U[ 'new_tri' ].size ), sub = m[ 'sub' ], lm = r[ 'st' ] ) )

	T = { k: np.concatenate( v ) for k, v in T.items() }
	np.savez( os.path.join( LRAW, B[ 'name' ] + '.npz' ), val = T[ 'val' ].astype( np.float16 ), tt = T[ 'tt' ], page = T[ 'page' ].astype( np.int16 ), x = T[ 'x' ].astype( np.int16 ), y = T[ 'y' ].astype( np.int16 ), npages = npages )
	meta = dict( name = B[ 'name' ], level = B[ 'level' ], texel = B[ 's' ], ss = B.get( 'ss', 1 ), npages = int( npages ), size = PAGE, meshes = out, dry = DRY,
		stats = dict( texels = int( len( T[ 'tt' ] ) ), texels_used = int( ( T[ 'tt' ] > 0 ).sum() ), seconds = round( time.time() - t00, 1 ) ) )
	json.dump( meta, open( os.path.join( LRAW, B[ 'name' ] + '.meta.json' ), 'w' ) )
	llog( '%s: %d meshes, %d texels (%d used) on %d pages, %.1fs' % ( B[ 'name' ], len( out ), len( T[ 'tt' ] ), int( ( T[ 'tt' ] > 0 ).sum() ), npages, time.time() - t00 ) )

def bake_va( v ):

	# vertex light from the full model's surface: every opaque vertex of an unsimplified level, a step inside its faces
	t0 = time.time()
	M = load_mesh( dict( nv = v[ 'nv' ], ni = v[ 'ni' ] ), v[ 'path' ] )
	M.update( min = v[ 'min' ], q = v[ 'q' ] )
	rec, tri = M[ 'rec' ], M[ 'tri' ]
	P = mesh_pos( M )
	opq = np.zeros( len( tri ), bool )
	for kind, start, count in v[ 'sub' ]:

		if kind == K_OPAQUE: opq[ start // 3:( start + count ) // 3 ] = True

	t = tri[ opq ]
	fn = np.cross( P[ t[ :, 1 ] ] - P[ t[ :, 0 ] ], P[ t[ :, 2 ] ] - P[ t[ :, 0 ] ] )
	cen = ( P[ t[ :, 0 ] ] + P[ t[ :, 1 ] ] + P[ t[ :, 2 ] ] ) / 3
	N = np.zeros( ( len( rec ), 3 ) ); Cn = np.zeros( ( len( rec ), 3 ) ); cnt = np.zeros( len( rec ) )
	for k in range( 3 ): np.add.at( N, t[ :, k ], fn ); np.add.at( Cn, t[ :, k ], cen ); np.add.at( cnt, t[ :, k ], 1 )
	use = ( cnt > 0 ) & ( rec[ :, 9 ] != 65535 ) & ( np.linalg.norm( N, axis = 1 ) > 1e-12 )
	vi = np.nonzero( use )[ 0 ]
	n = N[ vi ] / np.linalg.norm( N[ vi ], axis = 1 )[ :, None ]
	d = Cn[ vi ] / cnt[ vi ][ :, None ] - P[ vi ]; dl = np.maximum( np.linalg.norm( d, axis = 1 ), 1e-9 )
	X = P[ vi ] + d * ( np.minimum( dl, 0.01 ) / dl )[ :, None ]
	pb = fine_near( v[ 'key' ], P.min( 0 ), P.max( 0 ) )
	pr = project( pb, X, n, rec[ vi, 3 ].astype( np.int64 ), np.full( len( vi ), 0.02 ), float( v.get( 'radius', 0.35 ) ) )
	hit = pr[ :, 0 ] >= 0
	if hit.any():

		c, gl = fine_light( pb, pr[ hit ], 'va' )
		rec[ vi[ hit ], 6:9 ] = np.round( np.sqrt( np.clip( c, 0, 64 ) / 64 ) * 65535 )
		rec[ vi[ hit ], 9 ] = np.round( np.clip( gl, 0, 1 ) * 32767 )

	fnm = '%s.%s.va.bin' % ( v[ 'key' ], v[ 'src' ] )
	with open( os.path.join( LRAW, fnm ), 'wb' ) as f: f.write( rec.astype( '<u2' ).tobytes() ); f.write( tri.astype( '<u4' ).tobytes() ); f.write( M[ 'flags' ].tobytes() )
	llog( 'va %s %s: %d of %d vertices lit on the full model, %.1fs' % ( v[ 'key' ], v[ 'src' ], int( hit.sum() ), len( rec ), time.time() - t0 ) )
	return dict( key = v[ 'key' ], src = v[ 'src' ], raw = fnm, levels = v[ 'levels' ] )

# ------------------------------------------------------------------------------------------ light kept in the vertices, fitted

# (tone(), the samples on the faces, fit_values(), evaluate(), run_fit(): tools/next/lsvb_fit.py, part of this file)
_FIT = os.path.join( _CFG[ 'root' ], 'tools/next/lsvb_fit.py' )
exec( compile( open( _FIT ).read(), _FIT, 'exec' ) )

# ------------------------------------------------------------------------------------------ pages

def encode( B ):

	z = np.load( os.path.join( LRAW, B[ 'name' ] + '.npz' ) )
	meta = json.load( open( os.path.join( LRAW, B[ 'name' ] + '.meta.json' ) ) )
	val = z[ 'val' ].astype( np.float32 ); page = z[ 'page' ].astype( np.int64 ); x = z[ 'x' ].astype( np.int64 ); y = z[ 'y' ].astype( np.int64 ); tt = z[ 'tt' ]
	npages = int( z[ 'npages' ] )
	R = float( B.get( 'range', 8 ) ); gamma = float( B.get( 'gamma', 2 ) )
	img = np.zeros( ( max( npages, 1 ), PAGE, PAGE, 4 ), np.uint8 )
	used = tt > 0
	v = np.clip( val[ :, :3 ] / R, 0, 1 ) ** ( 1 / gamma ) * 255
	rs = np.random.RandomState( 7 )
	d = ( rs.uniform( - 0.5, 0.5, v.shape ) * float( B.get( 'dither', 1 ) ) ).astype( np.float32 )
	q = np.clip( np.floor( v + 0.5 + d ), 0, 255 ).astype( np.uint8 )
	a = np.clip( np.round( val[ :, 3 ] * 255 ), 0, 255 ).astype( np.uint8 )
	img[ page[ used ], y[ used ], x[ used ], :3 ] = q[ used ]
	img[ page[ used ], y[ used ], x[ used ], 3 ] = a[ used ]
	d = os.path.join( LDIR, B[ 'name' ] ); os.makedirs( d, exist_ok = True )
	files = []
	for p in range( npages ):

		fn = '%s/p%d.bin' % ( B[ 'name' ], p )
		with open( os.path.join( LDIR, fn ), 'wb' ) as f: f.write( img[ p ].tobytes() )
		files.append( fn )

	clip = float( ( val[ used, :3 ].max( 1 ) > R ).mean() ) if used.any() else 0
	meta.update( range = R, gamma = gamma, gloss = 1, pages = files )
	meta[ 'stats' ].update( clipped = round( clip, 5 ), max = round( float( val[ used, :3 ].max() ) if used.any() else 0, 3 ) )
	llog( 'encode %s: %d pages, range %g gamma %g, %.3f%% of the texels over the range (max %.2f)' % ( B[ 'name' ], npages, R, gamma, 100 * clip, meta[ 'stats' ][ 'max' ] ) )
	return meta

# ------------------------------------------------------------------------------------------ how to store a texel

def study( B ):

	# the texels of a bundle through several encodings: what each is off by on the screen (steps of 255 after exposure,
	# tone curve and display gamma, on a mid-grey surface), how coarse its steps are in the dark, and what its pages weigh
	import subprocess, shutil
	z = np.load( os.path.join( LRAW, B[ 'name' ] + '.npz' ) )
	tt = z[ 'tt' ]; used = tt > 0
	val = z[ 'val' ].astype( np.float64 )[ used ]; c = val[ :, :3 ]
	page = z[ 'page' ].astype( np.int64 )[ used ]; x = z[ 'x' ].astype( np.int64 )[ used ]; y = z[ 'y' ].astype( np.int64 )[ used ]
	ref = tone( c ); lum = c @ np.array( [ 0.2126, 0.7152, 0.0722 ] )
	dark = ( lum > 1e-4 ) & ( lum <= np.percentile( lum[ lum > 1e-4 ], 25 ) ) # the darkest quarter of what is lit at all
	rs = np.random.RandomState( 7 )
	res = dict( name = B[ 'name' ], texels = int( used.sum() ), max = round( float( c.max() ), 2 ), p999 = round( float( np.percentile( c.max( 1 ), 99.9 ) ), 2 ), median = round( float( np.median( lum ) ), 4 ), enc = {} )
	def err( dec, tag, extra = None ):

		e = np.abs( tone( dec ) - ref ).max( 1 ) * 255
		r = dict( rms = round( float( np.sqrt( ( e ** 2 ).mean() ) ), 3 ), p99 = round( float( np.percentile( e, 99 ) ), 2 ), max = round( float( e.max() ), 2 ),
			over1 = round( float( ( e > 1 ).mean() ), 4 ), dark_rms = round( float( np.sqrt( ( e[ dark ] ** 2 ).mean() ) ), 3 ) )
		if extra: r.update( extra )
		res[ 'enc' ][ tag ] = r
		return r

	def codes( R, g, bits, dither ):

		n = ( 1 << bits ) - 1
		v = np.clip( c / R, 0, 1 ) ** ( 1 / g ) * n
		d = rs.uniform( - 0.5, 0.5, v.shape ) * dither
		q = np.clip( np.floor( v + 0.5 + d ), 0, n )
		# the step between two neighbouring codes, on the screen, where the dark texels are
		st = ( tone( ( ( np.minimum( q[ dark ] + 1, n ) / n ) ** g ) * R ) - tone( ( q[ dark ] / n ) ** g * R ) ).max( 1 ) * 255
		return ( q / n ) ** g * R, dict( dark_step = round( float( np.median( st ) ), 2 ), dark_step_p95 = round( float( np.percentile( st, 95 ) ), 2 ), bytes_per_texel = 4 if bits == 8 else 8 )

	R99 = float( np.ceil( np.percentile( c.max( 1 ), 99.9 ) * 4 ) / 4 )
	for tag, R, g, bits, di in ( ( 'rgba8 linear /8', 8, 1, 8, 0 ), ( 'rgba8 root /8', 8, 2, 8, 0 ), ( 'rgba8 root /8 dithered', 8, 2, 8, 1 ), ( 'rgba8 root /%g (99.9 %% of this bundle)' % R99, R99, 2, 8, 1 ),
			( 'rgba8 cube root /8 dithered', 8, 3, 8, 1 ), ( 'rgba8 gamma 2.2 /8 dithered', 8, 2.2, 8, 1 ), ( 'rgba16 root /64', 64, 2, 16, 0 ) ):

		dec, ex = codes( R, g, bits, di ); err( dec, tag, ex )

	# RGBM: the colour over its largest channel in rgb, that channel (over 8, root) in a - the gloss needs a page of its own
	m = np.clip( np.sqrt( c.max( 1 ) / 8 ), 1 / 255, 1 ); mq = np.ceil( m * 255 ) / 255
	rgb = np.clip( np.round( c / ( mq ** 2 * 8 )[ :, None ] * 255 ), 0, 255 ) / 255
	err( rgb * ( mq ** 2 * 8 )[ :, None ], 'rgbm8 (m root /8)', dict( bytes_per_texel = 5 ) )
	# half float: the bake's own values as they are kept (float16)
	err( z[ 'val' ][ used ][ :, :3 ].astype( np.float64 ), 'rgba16f', dict( bytes_per_texel = 8 ) )
	# the pages as files
	d = os.path.join( LDIR, B[ 'name' ] ); tmp = os.path.join( LDIR, '.work', 'study' ); os.makedirs( tmp, exist_ok = True )
	mg = shutil.which( 'magick' )
	npg = int( z[ 'npages' ] ); some = list( range( min( npg, int( B.get( 'study_pages', 3 ) ) ) ) )
	size = dict( pages = npg, measured = len( some ), raw = 0 )
	Rr = float( B.get( 'range', 8 ) )
	for pi in some:

		src = os.path.join( d, 'p%d.bin' % pi )
		if not os.path.exists( src ) or not mg: break
		size[ 'raw' ] += os.path.getsize( src )
		on = page == pi
		img = np.frombuffer( open( src, 'rb' ).read(), np.uint8 ).reshape( PAGE, PAGE, 4 )
		own = ( img[ y[ on ], x[ on ], :3 ].astype( np.float64 ) / 255 ) ** 2 * Rr
		for tag, ext, args, lossy in ( ( 'png', 'png', [ '-define', 'png:compression-level=9' ], 0 ), ( 'webp lossless', 'webp', [ '-define', 'webp:lossless=true', '-define', 'webp:exact=true', '-define', 'webp:method=6' ], 0 ),
				( 'webp q90', 'webp', [ '-quality', '90', '-define', 'webp:exact=true', '-define', 'webp:alpha-quality=100', '-define', 'webp:method=6' ], 1 ),
				( 'webp q75', 'webp', [ '-quality', '75', '-define', 'webp:exact=true', '-define', 'webp:alpha-quality=100', '-define', 'webp:method=6' ], 1 ),
				( 'bc3 (dxt5, 1 B / texel)', 'dds', [ '-define', 'dds:compression=dxt5', '-define', 'dds:mipmaps=0' ], 1 ), ( 'bc1 (dxt1, no gloss, 0.5 B / texel)', 'dds', [ '-alpha', 'off', '-define', 'dds:compression=dxt1', '-define', 'dds:mipmaps=0' ], 1 ) ):

			out = os.path.join( tmp, 'p.' + ext ); back = os.path.join( tmp, 'back.rgba' )
			r = subprocess.run( [ mg, '-size', '%dx%d' % ( PAGE, PAGE ), '-depth', '8', 'rgba:' + src ] + args + [ out ], capture_output = True )
			if r.returncode or not os.path.exists( out ): size[ tag ] = None; continue
			size[ tag ] = size.get( tag, 0 ) + os.path.getsize( out )
			if lossy:

				subprocess.run( [ mg, out, '-depth', '8', 'rgba:' + back ], capture_output = True )
				bk = np.frombuffer( open( back, 'rb' ).read(), np.uint8 ).reshape( PAGE, PAGE, 4 )
				dec = ( bk[ y[ on ], x[ on ], :3 ].astype( np.float64 ) / 255 ) ** 2 * Rr
				e = np.abs( tone( dec ) - tone( own ) ).max( 1 ) * 255
				k = res[ 'enc' ].setdefault( 'file: ' + tag, dict( rms = [], p99 = [], max = [] ) )
				k[ 'rms' ].append( round( float( np.sqrt( ( e ** 2 ).mean() ) ), 2 ) ); k[ 'p99' ].append( round( float( np.percentile( e, 99 ) ), 2 ) ); k[ 'max' ].append( round( float( e.max() ), 1 ) )
				os.remove( back )

			os.remove( out )

	res[ 'size' ] = size
	llog( 'study %s: %s' % ( B[ 'name' ], json.dumps( res ) ) )
	return res

# ------------------------------------------------------------------------------------------ main

def lm_main():

	os.makedirs( LRAW, exist_ok = True )
	dump = json.load( open( os.path.join( LRAW, 'dump.json' ) ) )[ 'meshes' ] if os.path.exists( os.path.join( LRAW, 'dump.json' ) ) else []
	def pick( B ): return [ m for m in dump if m[ 'level' ] == B[ 'level' ] and m[ 'variant' ] == B[ 'variant' ] and ( not B.get( 'cells' ) or m[ 'key' ] in B[ 'cells' ] ) ]
	fitted = lambda B: B.get( 'kind' ) in ( 'own', 'fit' )
	if LM[ 'mode' ] == 'bake':

		if not DRY: fine_setup()
		# the points on the faces of the meshes whose light stays in the vertices, lit on the full model (a cell at a time)
		todo = sorted( [ ( m[ 'key' ], m[ 'level' ], m[ 'variant' ], m, S ) for S in LM.get( 'sample' ) or [] for m in pick( S ) ], key = lambda t: t[ :3 ] )
		if todo and not DRY:

			# (a mesh at a time, each kept as it is done: the run can be asked to stop between two - the file `stop` - and
			# picks up where it was when it is started again)
			for key, level, variant, m, S in todo:

				if LM.get( 'stop' ) and os.path.exists( LM[ 'stop' ] ): llog( 'asked to stop: %s' % LM[ 'stop' ] ); break
				sample_meshes( [ m ], S )

		for B in LM[ 'bundles' ]:

			if fitted( B ): continue
			ms = pick( B )
			if ms: bake_bundle( B, ms )

		va = [ bake_va( v ) for v in LM.get( 'va' ) or [] ] if not DRY else []
		if LM.get( 'va' ) is not None and not DRY: json.dump( va, open( os.path.join( LRAW, 'va.json' ), 'w' ) )

	elif LM[ 'mode' ] == 'fit':

		# (no full model, no rays: everything from the samples the bake left)
		for B in LM[ 'bundles' ]:

			ms = pick( B )
			if not ms: continue
			if fitted( B ): run_fit( B, ms )
			elif os.path.exists( os.path.join( LDIR, B[ 'name' ], 'p0.bin' ) ): eval_lightmap( B, ms )

	elif LM[ 'mode' ] == 'relight':

		# (no full model, no rays) the unsimplified meshes of the dump with the raw level's light: into the dump as "rawrl"
		dj = json.load( open( os.path.join( LRAW, 'dump.json' ) ) )
		new = [ relight( m ) for m in dump if m[ 'variant' ] == 'raw' and ( not LM.get( 'cells' ) or m[ 'key' ] in LM[ 'cells' ] ) ]
		got = set( ( m[ 'key' ], m[ 'level' ] ) for m in new )
		dj[ 'meshes' ] = [ m for m in dj[ 'meshes' ] if not ( m[ 'variant' ] == 'rawrl' and ( m[ 'key' ], m[ 'level' ] ) in got ) ] + new
		json.dump( dj, open( os.path.join( LRAW, 'dump.json' ), 'w' ) )

	elif LM[ 'mode' ] == 'encode':

		out = []
		for B in LM[ 'bundles' ]:

			for name in ( ( B[ 'name' ], B[ 'name' ] + '_res' ) if fitted( B ) else ( B[ 'name' ], ) ):

				if not os.path.exists( os.path.join( LRAW, name + ( '.meta.json' if fitted( B ) else '.npz' ) ) ): continue
				out.append( json.load( open( os.path.join( LRAW, name + '.meta.json' ) ) ) if fitted( B ) else encode( B ) )

			# (a bundle in pages: what the fit stage measured of it, as a mesh of its own)
			rn = os.path.join( LRAW, B[ 'name' ] + '_res.meta.json' )
			if not fitted( B ) and os.path.exists( rn ): out.append( json.load( open( rn ) ) )

		va = json.load( open( os.path.join( LRAW, 'va.json' ) ) ) if os.path.exists( os.path.join( LRAW, 'va.json' ) ) else []
		json.dump( dict( bundles = out, va = va ), open( os.path.join( LRAW, 'bake.json' ), 'w' ) )

	elif LM[ 'mode' ] == 'hybrid':

		# vertex light by default, texels only where it leaves something to see (Schäfer et al. 2012), counted: the lit
		# faces of a fitted bundle V whose error (rms of the points on the face) stays over t display steps, by kind of
		# surface, and what is left when those faces take the light of the bundle in pages L instead (same faces)
		out = []
		for V, L in LM[ 'pairs' ]:

			fv = os.path.join( LRAW, 'faces.%s.npz' % V )
			if not os.path.exists( fv ): continue
			zv = np.load( fv ); n = len( zv.files ) // 3
			cat = lambda z, k: np.concatenate( [ z[ '%s_%d' % ( k, i ) ] for i in range( n ) ] ).astype( np.float64 )
			ev, ar, kd = cat( zv, 'err' ), cat( zv, 'area' ), cat( zv, 'kind' ).astype( np.int64 )
			el = None; fl = os.path.join( LRAW, 'faces.%s.npz' % L )
			if L and os.path.exists( fl ):

				zl = np.load( fl )
				if len( zl.files ) == len( zv.files ) and all( len( zl[ 'err_%d' % i ] ) == len( zv[ 'err_%d' % i ] ) for i in range( n ) ): el = cat( zl, 'err' )

			row = dict( vertex = V, pages = L if el is not None else None, by = {} )
			if L and os.path.exists( os.path.join( LRAW, 'stats.%s.json' % L ) ):

				sl = json.load( open( os.path.join( LRAW, 'stats.%s.json' % L ) ) )
				row[ 'texels_m2' ] = round( sl[ 'texels' ] / max( sl[ 'paged_area' ], 1e-9 ), 1 )

			rms = lambda e, w: round( float( np.sqrt( ( e * e * w ).sum() / max( w.sum(), 1e-30 ) ) ), 3 )
			for t in LM.get( 'over', ( 4, 8 ) ):

				for ki, kn in [ ( - 1, 'all' ) ] + list( enumerate( KINDS ) ):

					s = np.ones( len( ev ), bool ) if ki < 0 else kd == ki
					if not s.any(): continue
					bad = s & ( ev > t )
					d = dict( area = round( float( ar[ s ].sum() ), 1 ), faces = int( s.sum() ), over_area = round( float( ar[ bad ].sum() ), 1 ), over_faces = int( bad.sum() ), rms = rms( ev[ s ], ar[ s ] ) )
					if el is not None:

						use = bad & ( el < ev )
						mix = np.where( use, el, ev )
						d.update( paged_area = round( float( ar[ use ].sum() ), 1 ), rms_mixed = rms( mix[ s ], ar[ s ] ), rms_pages = rms( el[ s ], ar[ s ] ), left_area = round( float( ar[ s & ( mix > t ) ].sum() ), 1 ) )

					row[ 'by' ][ '%s_%d' % ( kn, t ) ] = d

			out.append( row )

		json.dump( out, open( os.path.join( LRAW, 'hybrid.json' ), 'w' ), indent = 1 )

	elif LM[ 'mode' ] == 'study':

		out = [ study( B ) for B in LM[ 'bundles' ] if not fitted( B ) and os.path.exists( os.path.join( LRAW, B[ 'name' ] + '.npz' ) ) ]
		json.dump( out, open( os.path.join( LRAW, 'study.json' ), 'w' ), indent = 1 )

	llog( 'done' )

lm_main()
