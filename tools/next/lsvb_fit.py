# Baked light kept in the vertices of the reduced levels, fitted - part of tools/next/lightmap_bl.py (run by it with
# exec: its globals are this file's), driven by tools/next/lightmap.py.
#
#   samples   mesh_samples(): two sets of points on the lit faces of a reduced mesh, each moved onto the FULL model
#             (LOD0) and lit there by the bake's own ray casting (project() + fine_light()):
#               fit   a lattice over every face (what a fit may look at)
#               eval  random points, even over the area (what every variant is judged by - never shown to a fit)
#             and for both the light the densest RAW level of the cell has there (its vertex light interpolated: what a
#             fit could use without casting a ray). Kept in <lm>/.raw/smp.*.npz: all fits run from them without rays.
#   fit       fit_values(): the least-squares vertex light of Kavan, Bargteil, Sloan 2011 - ( A + alpha R ) x = b, A the
#             mass matrix as the samples weigh it, R the edge term (eq. 13: the change of the gradient over an edge two
#             faces of one flat piece share). The page squares the stored colour in the VERTEX stage, so what the GPU
#             interpolates over a face is linear light: the unknowns are linear light and the hat functions are exact.
#             The samples are weighed by the slope of the page's tone curve at their value, squared: least squares on
#             what the screen shows, not on radiance.
#   variants  keep: the vertices as they are, only their values; split: the corners of a place grouped by what their
#             faces want; refine: the faces the fit cannot carry cut in four, worst first (Li et al. 2012), fitted again
#   measure   evaluate(): the mesh AS WRITTEN (colours rounded as the packer rounds them) against the eval set, in
#             display steps of 255: | tone( mesh ) - tone( full model ) |, the largest of the three channels, weighed by
#             area; points inside something or not found on the full model left out (their share is reported)
#
# Who may not move: a ground vertex (one ground at every level: bit 4, unless ground_fit), a ground vertex two cells
# share (bit 1), and every vertex that also stands in a neighbouring cell's mesh (walls run over cell borders: both
# cells carry the same vertex with the same light, and a fit of one cell alone would open a seam there).

LUMW = np.array( [ 0.2126, 0.7152, 0.0722 ] )
BRAW = os.path.join( os.path.dirname( LDIR.rstrip( '/' ) ), '.raw' ) # the unpacked bake: read, never written
KINDS = ( 'ground', 'wall', 'prop', 'other' )

def tone( c ):

	# about what the page shows of a lit colour on a mid-grey surface (exposure 1.25, ACES, display gamma): 0 .. 1
	x = np.maximum( c, 0 ) * ( 1.25 * 0.5 )
	a = np.clip( x * ( 2.51 * x + 0.03 ) / ( x * ( 2.43 * x + 0.59 ) + 0.14 ), 0, 1 )
	return a ** ( 1 / 2.2 )

def heat( e ):

	# an error in display steps (of 255) as a colour the page can show: 2 and less near black, 4 blue, 8 green,
	# 16 yellow, 32 orange, 64 and more red (as lit colour: it goes through the page's tone curve)
	t = np.clip( np.log2( np.maximum( e, 1e-6 ) ), 1, 6 ) - 1
	stops = np.array( [ [ 0.01, 0.015, 0.06 ], [ 0.03, 0.10, 0.9 ], [ 0.08, 0.9, 0.12 ], [ 1.6, 1.3, 0.05 ], [ 2.4, 0.45, 0.03 ], [ 2.4, 0.03, 0.03 ] ] )
	i = np.clip( np.floor( t ).astype( np.int64 ), 0, 4 ); fr = ( t - i )[ :, None ]
	return stops[ i ] * ( 1 - fr ) + stops[ i + 1 ] * fr

def heat_signed( d ):

	# a signed error (display steps: the mesh's light less the full model's) as a colour: grey none, blue darker than
	# the full model, red brighter; full colour at 32 steps
	t = ( np.clip( np.abs( d ) / 32, 0, 1 ) ** 0.7 )[ :, None ]
	end = np.where( ( d < 0 )[ :, None ], np.array( [ [ 0.02, 0.12, 2.6 ] ] ), np.array( [ [ 2.6, 0.06, 0.02 ] ] ) )
	return np.array( [ [ 0.22, 0.22, 0.22 ] ] ) * ( 1 - t ) + end * t

def enc16( c ): return np.round( np.sqrt( np.clip( c, 0, 64 ) / 64 ) * 65535 )

def lin13( c16 ):

	# the linear light the page makes of the stored colour slots: rounded to 13 bits by the packer, squared in the vertex stage
	v = np.minimum( 65528, np.round( c16.astype( np.float64 ) / 8 ) * 8 )
	return ( v / 65535 ) ** 2 * 64

def pkey( rec ): return rec[ :, 0 ].astype( np.int64 ) | ( rec[ :, 1 ].astype( np.int64 ) << 16 ) | ( rec[ :, 2 ].astype( np.int64 ) << 32 )

def wkeys( mn, q, rec3 ):

	# a place in the world as one number, whatever cell's grid it is on
	w = np.round( ( np.array( mn, np.float64 )[ None ] + rec3.astype( np.float64 ) * q ) * 1024 ).astype( np.int64 ) + ( 1 << 20 )
	return ( w[ :, 0 ] << 42 ) | ( w[ :, 1 ] << 21 ) | w[ :, 2 ]

_RJ = {}
def raw_chunks():

	if not _RJ: _RJ.update( { c[ 'k' ]: c for c in json.load( open( os.path.join( BRAW, 'raw.json' ) ) )[ 'chunks' ] } )
	return _RJ

_NB = {}
def neighbour_keys( key ):

	# every place a neighbouring cell has a vertex at (its unsimplified mid level: all its reduced levels are of it)
	if key in _NB: return _NB[ key ]
	_NB.clear()
	i, j = [ int( v ) for v in key.split( '_' )[ 1: ] ]
	out = [ np.zeros( 0, np.int64 ) ]
	for di in ( - 1, 0, 1 ):

		for dj in ( - 1, 0, 1 ):

			c = raw_chunks().get( 'c_%d_%d' % ( i + di, j + dj ) )
			if not c or ( not di and not dj ): continue
			m = c[ 'mid' ]
			r = np.fromfile( os.path.join( BRAW, m[ 'raw' ] ), '<u2', m[ 'nv' ] * 10 ).reshape( - 1, 10 )
			out.append( wkeys( m[ 'min' ], m[ 'q' ], r[ :, :3 ] ) )

	_NB[ key ] = np.unique( np.concatenate( out ) )
	return _NB[ key ]

def hold_vertices( m, rec, flags, B ):

	hold = ( flags & 1 ) > 0
	if not B.get( 'ground_fit' ): hold |= ( flags & 4 ) > 0
	if B.get( 'neighbours', 1 ): hold |= np.isin( wkeys( m[ 'min' ], m[ 'q' ], rec[ :, :3 ] ), neighbour_keys( m[ 'key' ] ) )
	return hold

# ------------------------------------------------------------------------------------------ points on the faces

def lattice( k ):

	# the middles of the k * k parts of a triangle (one part: the three points that integrate a square exactly);
	# a small one also has its corners, pulled in
	if k == 1: return np.array( [ [ 2 / 3, 1 / 6, 1 / 6 ], [ 1 / 6, 2 / 3, 1 / 6 ], [ 1 / 6, 1 / 6, 2 / 3 ] ] )
	out = []
	for i in range( k ):

		for j in range( k - i ):

			out.append( ( ( i + 1 / 3 ) / k, ( j + 1 / 3 ) / k ) )
			if i + j < k - 1: out.append( ( ( i + 2 / 3 ) / k, ( j + 2 / 3 ) / k ) )

	if k <= 2: out += [ ( 0.15, 0.15 ), ( 0.7, 0.15 ), ( 0.15, 0.7 ) ]
	b = np.array( out, np.float64 )
	return np.concatenate( [ 1 - b.sum( 1, keepdims = True ), b ], 1 )

def face_samples( P, tri, faces, s, kmax ):

	# -> face (index into tri), barycentric (n, 3), point, face normal, spacing of the samples there
	a = P[ tri[ faces, 0 ] ]; b = P[ tri[ faces, 1 ] ]; c = P[ tri[ faces, 2 ] ]
	fn = np.cross( b - a, c - a ); a2 = np.linalg.norm( fn, axis = 1 )
	nrm = fn / np.maximum( a2, 1e-20 )[ :, None ]
	longest = np.sqrt( np.maximum( np.maximum( ( ( b - a ) ** 2 ).sum( 1 ), ( ( c - b ) ** 2 ).sum( 1 ) ), ( ( a - c ) ** 2 ).sum( 1 ) ) )
	k = np.clip( np.ceil( np.sqrt( a2 ) / s ), 1, kmax ).astype( np.int64 )
	k = np.maximum( k, np.minimum( np.ceil( longest / ( 4 * s ) ), 4 ).astype( np.int64 ) )
	F, Bc = [], []
	for kk in np.unique( k ).tolist():

		idx = np.nonzero( k == kk )[ 0 ]; L = lattice( kk )
		F.append( np.repeat( idx, len( L ) ) ); Bc.append( np.tile( L, ( len( idx ), 1 ) ) )

	fl = np.concatenate( F ) if F else np.zeros( 0, np.int64 ); bc = np.concatenate( Bc ) if Bc else np.zeros( ( 0, 3 ) )
	X = a[ fl ] * bc[ :, 0:1 ] + b[ fl ] * bc[ :, 1:2 ] + c[ fl ] * bc[ :, 2:3 ]
	Ls = np.clip( np.sqrt( a2 ) / k * 0.5, 0.02, s )
	return faces[ fl ], bc, X, nrm[ fl ], Ls[ fl ]

def rand_samples( P, tri, faces, rho, seed ):

	# random points, rho a square metre and one a face at least -> face, barycentric, point, normal, the area each stands for
	a = P[ tri[ faces, 0 ] ]; b = P[ tri[ faces, 1 ] ]; c = P[ tri[ faces, 2 ] ]
	fn = np.cross( b - a, c - a ); a2 = np.linalg.norm( fn, axis = 1 )
	nrm = fn / np.maximum( a2, 1e-20 )[ :, None ]
	rs = np.random.RandomState( seed )
	n = np.maximum( np.floor( a2 * 0.5 * rho + rs.uniform( size = len( faces ) ) ).astype( np.int64 ), 1 )
	fl = np.repeat( np.arange( len( faces ) ), n )
	r1 = np.sqrt( rs.uniform( size = len( fl ) ) ); r2 = rs.uniform( size = len( fl ) )
	bc = np.stack( [ 1 - r1, r1 * ( 1 - r2 ), r1 * r2 ], 1 )
	X = a[ fl ] * bc[ :, 0:1 ] + b[ fl ] * bc[ :, 1:2 ] + c[ fl ] * bc[ :, 2:3 ]
	return faces[ fl ], bc, X, nrm[ fl ], ( a2 * 0.5 / n )[ fl ]

def leader( ids, v, tol ):

	# of the corners with one id, those within tol of the lowest value stay together, the rest start again
	o = np.lexsort( ( v, ids ) ); gi = ids[ o ]; vv = v[ o ]
	n = len( o ); out = np.full( n, - 1, np.int64 ); nxt = 0
	left = np.arange( n )
	while len( left ):

		g = gi[ left ]
		first = np.ones( len( left ), bool ); first[ 1: ] = g[ 1: ] != g[ :- 1 ]
		st = np.maximum.accumulate( np.where( first, np.arange( len( left ) ), 0 ) )
		take = vv[ left ] - vv[ left ][ st ] <= tol
		cid = np.cumsum( first ) - 1 + nxt
		out[ left[ take ] ] = cid[ take ]
		nxt += int( first.sum() )
		left = left[ ~ take ]

	res = np.empty( n, np.int64 ); res[ o ] = out
	return np.unique( res, return_inverse = True )[ 1 ].ravel()

def cut_faces( rec, flags, tri, ent, pick, nocut = None ):

	# the picked triangles in four, by the middles of their edges (pick: one flag a triangle), or only the picked edges
	# of them (pick: three flags a triangle - a long thin face is cut along its length alone); whoever shares a cut edge
	# is cut along with them (in two or three), so no vertex ends up in the middle of somebody's edge. An edge along the
	# border of the cell, between two vertices of nocut (who may not move: the next cell or level has the same edge
	# whole) or at a glowing vertex stays whole. A new vertex has the light the edge had there (linear light).
	# -> rec, flags, tri, ent, the triangles that are new, the triangle each came from (in rising order)
	nv, nt = len( rec ), len( tri )
	weld = np.unique( pkey( rec ), return_inverse = True )[ 1 ].ravel(); NW = int( weld.max() ) + 1
	wl = np.zeros( NW, bool ); wl[ weld[ ( flags & 1 ) > 0 ] ] = True
	if nocut is not None: wl[ weld[ nocut ] ] = True
	wg = np.zeros( NW, bool ); wg[ weld[ rec[ :, 9 ] == 65535 ] ] = True
	wt = weld[ tri ]; ea = wt; eb = wt[ :, [ 1, 2, 0 ] ]
	ek = np.minimum( ea, eb ) * NW + np.maximum( ea, eb )
	bad = ( wl[ ea ] & wl[ eb ] ) | wg[ ea ] | wg[ eb ] | ( ea == eb )
	if pick.ndim == 1: pick = np.repeat( pick[ :, None ], 3, 1 )
	mk = np.isin( ek, np.unique( ek[ pick & ~ bad ] ) ) & ~ bad
	# the new vertices: one for each pair of vertex records an edge to be cut runs between
	va = tri; vb = tri[ :, [ 1, 2, 0 ] ]
	pk = np.minimum( va, vb ) * nv + np.maximum( va, vb )
	up, inv = np.unique( pk[ mk ], return_inverse = True )
	a = up // nv; b = up % nv
	new = rec[ a ].copy()
	new[ :, 0:3 ] = ( rec[ a, 0:3 ].astype( np.int64 ) + rec[ b, 0:3 ] ) // 2
	new[ :, 4:6 ] = ( rec[ a, 4:6 ].astype( np.int64 ) + rec[ b, 4:6 ] ) // 2
	new[ :, 6:9 ] = np.round( np.sqrt( ( rec[ a, 6:9 ].astype( np.float64 ) ** 2 + rec[ b, 6:9 ].astype( np.float64 ) ** 2 ) * 0.5 ) )
	new[ :, 9 ] = ( rec[ a, 9 ].astype( np.int64 ) + rec[ b, 9 ] ) // 2
	mid = np.full( ( nt, 3 ), - 1, np.int64 ); mid[ mk ] = nv + inv.ravel()
	code = mk[ :, 0 ] + 2 * mk[ :, 1 ] + 4 * mk[ :, 2 ]
	T, Pa = [ tri[ code == 0 ] ], [ np.nonzero( code == 0 )[ 0 ] ]
	def emit( idx, *ts ):

		for t in ts: T.append( np.stack( t, 1 ) ); Pa.append( idx )

	idx = np.nonzero( code == 7 )[ 0 ]
	if len( idx ):

		A, Bv, C = tri[ idx, 0 ], tri[ idx, 1 ], tri[ idx, 2 ]; m0, m1, m2 = mid[ idx, 0 ], mid[ idx, 1 ], mid[ idx, 2 ]
		emit( idx, ( A, m0, m2 ), ( m0, Bv, m1 ), ( m2, m1, C ), ( m0, m1, m2 ) )

	for j in range( 3 ):

		# one edge: j, between the corners j and j + 1
		idx = np.nonzero( code == ( 1 << j ) )[ 0 ]
		if len( idx ):

			p, q, r = tri[ idx, j ], tri[ idx, ( j + 1 ) % 3 ], tri[ idx, ( j + 2 ) % 3 ]; m = mid[ idx, j ]
			emit( idx, ( p, m, r ), ( m, q, r ) )

		# two edges: all but u
		u = j
		idx = np.nonzero( code == ( 7 ^ ( 1 << u ) ) )[ 0 ]
		if len( idx ):

			p, q, r = tri[ idx, ( u + 2 ) % 3 ], tri[ idx, u ], tri[ idx, ( u + 1 ) % 3 ]
			mpq, mrp = mid[ idx, ( u + 2 ) % 3 ], mid[ idx, ( u + 1 ) % 3 ]
			emit( idx, ( p, mpq, mrp ), ( mpq, q, r ), ( mpq, r, mrp ) )

	T = np.concatenate( T ); Pa = np.concatenate( Pa )
	o = np.argsort( Pa, kind = 'stable' ); T = T[ o ]; Pa = Pa[ o ]
	nf = ( flags[ a ] & flags[ b ] ) & 0xfe
	return np.concatenate( [ rec, new ] ), np.concatenate( [ flags, nf.astype( np.uint8 ) ] ), T, ent[ Pa ], code[ Pa ] > 0, Pa

def bary( A, Bp, C, X ):

	# X in the plane of the triangles A B C -> its three weights (not held to the triangle)
	v0 = Bp - A; v1 = C - A; v2 = X - A
	d00 = ( v0 * v0 ).sum( 1 ); d01 = ( v0 * v1 ).sum( 1 ); d11 = ( v1 * v1 ).sum( 1 ); d20 = ( v2 * v0 ).sum( 1 ); d21 = ( v2 * v1 ).sum( 1 )
	den = d00 * d11 - d01 * d01; den = np.where( np.abs( den ) < 1e-30, 1e-30, den )
	w1 = ( d11 * d20 - d01 * d21 ) / den; w2 = ( d00 * d21 - d01 * d20 ) / den
	return np.stack( [ 1 - w1 - w2, w1, w2 ], 1 )

def move_samples( S, P, T, Pa ):

	# after cut_faces(): every sample (of both sets) to the piece of its face it lies in - the reference is as dense as
	# it was: what a cut face is judged by next is still the full model's light all over it
	out = dict( S )
	for kf, kb, kx in ( ( 'face', 'b', 'X' ), ( 'eface', 'eb', 'eX' ) ):

		pf = S[ kf ]
		c0 = np.searchsorted( Pa, pf, 'left' ); cn = np.searchsorted( Pa, pf, 'right' ) - c0
		face = c0.copy(); b = S[ kb ].copy()
		idx = np.nonzero( cn > 1 )[ 0 ]
		best = np.full( len( idx ), - 1e9 )
		X = S[ kx ][ idx ]
		for c in range( 4 ):

			use = np.nonzero( cn[ idx ] > c )[ 0 ]
			if not len( use ): break
			cand = c0[ idx[ use ] ] + c
			w = bary( P[ T[ cand, 0 ] ], P[ T[ cand, 1 ] ], P[ T[ cand, 2 ] ], X[ use ] )
			sc = w.min( 1 )
			bt = sc > best[ use ]
			u = use[ bt ]
			best[ u ] = sc[ bt ]; face[ idx[ u ] ] = cand[ bt ]
			wc = np.clip( w[ bt ], 0, 1 ); b[ idx[ u ] ] = wc / np.maximum( wc.sum( 1, keepdims = True ), 1e-9 )

		out[ kf ] = face; out[ kb ] = b

	return out

def lit_faces( rec, tri, ent, kinds, P ):

	# the faces whose light is baked: opaque, not glowing, not a sliver of no area -> flags, twice their area
	a2 = np.linalg.norm( np.cross( P[ tri[ :, 1 ] ] - P[ tri[ :, 0 ] ], P[ tri[ :, 2 ] ] - P[ tri[ :, 0 ] ] ), axis = 1 )
	return ( kinds[ ent ] == K_OPAQUE ) & ~ ( rec[ :, 9 ] == 65535 )[ tri ].any( 1 ) & ( a2 > 1e-7 ), a2

def face_kind( flags, tri, f, P ):

	tf = tri[ f ]; fl = flags[ tf ]
	n = np.cross( P[ tf[ :, 1 ] ] - P[ tf[ :, 0 ] ], P[ tf[ :, 2 ] ] - P[ tf[ :, 0 ] ] ); n /= np.maximum( np.linalg.norm( n, axis = 1 ), 1e-20 )[ :, None ]
	return np.where( ( ( fl & 4 ) > 0 ).all( 1 ), 0, np.where( ( ( fl & 2 ) > 0 ).all( 1 ), 2, np.where( np.abs( n[ :, 1 ] ) < 0.5, 1, 3 ) ) )

# ------------------------------------------------------------------------------------------ the raw bake as a target

_RAWL = {}
def raw_level( key ):

	# the densest level of a cell as the bake left it (its vertex light is what a fit can have without a ray)
	if key in _RAWL: return _RAWL[ key ]
	_RAWL.clear()
	from mathutils.bvhtree import BVHTree
	ch = raw_chunks()[ key ]
	lv = 'fine' if 'fine' in ch else 'near' if 'near' in ch else 'mid'
	c = ch[ lv ]; w = c.get( 'w', 10 ); nv, ni = c[ 'nv' ], c[ 'ni' ]
	b = open( os.path.join( BRAW, c[ 'raw' ] ), 'rb' ).read()
	rec = np.frombuffer( b, '<u2', nv * w ).reshape( nv, w )[ :, :10 ]
	tri = np.frombuffer( b, '<u4', ni, nv * w * 2 ).reshape( - 1, 3 ).astype( np.int64 )
	opq = np.zeros( len( tri ), bool )
	for kind, start, count in c[ 'sub' ]:

		if kind == K_OPAQUE: opq[ start // 3:( start + count ) // 3 ] = True

	P = np.array( c[ 'min' ], np.float64 )[ None ] + rec[ :, :3 ].astype( np.float64 ) * c[ 'q' ]
	fn = np.cross( P[ tri[ :, 1 ] ] - P[ tri[ :, 0 ] ], P[ tri[ :, 2 ] ] - P[ tri[ :, 0 ] ] ); a2 = np.linalg.norm( fn, axis = 1 )
	T = tri[ opq & ~ ( rec[ :, 9 ] == 65535 )[ tri ].any( 1 ) & ( a2 > 1e-9 ) ]
	fn = np.cross( P[ T[ :, 1 ] ] - P[ T[ :, 0 ] ], P[ T[ :, 2 ] ] - P[ T[ :, 0 ] ] ); fn /= np.maximum( np.linalg.norm( fn, axis = 1 ), 1e-20 )[ :, None ]
	used, inv = np.unique( T.ravel(), return_inverse = True )
	t0 = time.time()
	bvh = BVHTree.FromPolygons( P[ used ].tolist(), inv.reshape( - 1, 3 ).tolist(), all_triangles = True )
	lit, gl = vertex_light( rec )
	_RAWL[ key ] = dict( level = lv, bvh = bvh, T = T, P = P, FN = fn.tolist(), LAY = rec[ T[ :, 0 ], 3 ].tolist(), L4 = np.concatenate( [ lit, gl[ :, None ] ], 1 ) )
	llog( '  %s: raw level %s, %d triangles, tree %.1fs' % ( key, lv, len( T ), time.time() - t0 ) )
	return _RAWL[ key ]

def raw_target( key, X, N, LAY, R ):

	# the raw level's light at the points X (the nearest face of the same material that looks the same way)
	# -> light + gloss (n, 4), found
	rl = raw_level( key )
	near = rl[ 'bvh' ].find_nearest; rng = rl[ 'bvh' ].find_nearest_range
	FN, LT = rl[ 'FN' ], rl[ 'LAY' ]
	def fn( s, e ):

		out = np.zeros( ( e - s, 4 ), np.float64 ); out[ :, 0 ] = - 1
		xs = X[ s:e ].tolist(); ns = N[ s:e ].tolist(); ls = LAY[ s:e ].tolist()
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
					m = c[ 3 ] + 0.25 * R * ( 1 - d )
					if m < bd: bd = m; t = j; loc = c[ 0 ]

			if t >= 0: out[ i ] = ( t, loc[ 0 ], loc[ 1 ], loc[ 2 ] )

		return out

	if HAVE_FINE: rows = parallel( fn, len( X ), np.float64, ( 4, ), 4096, 'rawt' )
	else: rows = fn( 0, len( X ) )
	hit = rows[ :, 0 ] >= 0
	y = np.zeros( ( len( X ), 4 ) )
	if hit.any():

		T = rl[ 'T' ][ rows[ hit, 0 ].astype( np.int64 ) ]; P = rl[ 'P' ]
		w = np.clip( bary( P[ T[ :, 0 ] ], P[ T[ :, 1 ] ], P[ T[ :, 2 ] ], rows[ hit, 1:4 ] ), 0, 1 ); w /= np.maximum( w.sum( 1, keepdims = True ), 1e-9 )
		L4 = rl[ 'L4' ]
		y[ hit ] = L4[ T[ :, 0 ] ] * w[ :, 0:1 ] + L4[ T[ :, 1 ] ] * w[ :, 1:2 ] + L4[ T[ :, 2 ] ] * w[ :, 2:3 ]

	return y, hit

def relight( m ):

	# the unsimplified level of a cell with the densest raw level's light at its vertices (no ray: a lookup): every lit
	# corner asks a little inside its face (which side of an edge it means), a vertex takes the mean of its corners.
	# Who may not move (hold_vertices) keeps what it has. -> the mesh as the variant "rawrl", same vertices and faces
	t0 = time.time()
	M, rec, tri, flags, ent, kinds, P, lmf, a2 = open_mesh( m )
	f = np.nonzero( lmf )[ 0 ]; tf = tri[ f ]
	Pf = P[ tf ]; cen = Pf.mean( 1, keepdims = True )
	d = np.linalg.norm( cen - Pf, axis = 2, keepdims = True )
	X = ( Pf + ( cen - Pf ) * np.minimum( 0.1, 0.02 / np.maximum( d, 1e-9 ) ) ).reshape( - 1, 3 )
	n = np.cross( Pf[ :, 1 ] - Pf[ :, 0 ], Pf[ :, 2 ] - Pf[ :, 0 ] ); n /= np.maximum( np.linalg.norm( n, axis = 1 ), 1e-20 )[ :, None ]
	yr, hr = raw_target( m[ 'key' ], X, np.repeat( n, 3, 0 ), np.repeat( rec[ tf[ :, 0 ], 3 ].astype( np.int64 ), 3 ), 0.35 )
	r = tf.ravel(); nv = len( rec )
	cnt = np.bincount( r, hr, nv )
	acc = np.stack( [ np.bincount( r, hr * yr[ :, c ], nv ) for c in range( 3 ) ], 1 )
	move = ( cnt > 0 ) & ~ hold_vertices( m, rec, flags, {} ) & ( rec[ :, 9 ] != 65535 )
	out = rec.copy()
	out[ move, 6:9 ] = enc16( acc[ move ] / cnt[ move ][ :, None ] )
	fnm = '%s.%s.rawrl.bin' % ( m[ 'key' ], m[ 'level' ] )
	with open( os.path.join( LRAW, fnm ), 'wb' ) as fh: fh.write( out.astype( '<u2' ).tobytes() ); fh.write( tri.astype( '<u4' ).tobytes() ); fh.write( flags.astype( np.uint8 ).tobytes() )
	llog( '  relit %s %s: %d of %d vertices from the raw level (%d corners, %.1f %% found), %.0fs' % ( m[ 'key' ], m[ 'level' ], int( move.sum() ), nv, len( r ), 100 * float( hr.mean() ) if len( hr ) else 0, time.time() - t0 ) )
	return dict( { k: v for k, v in m.items() if k not in ( 'vbytes', 'vnv' ) }, variant = 'rawrl', raw = fnm )

# ------------------------------------------------------------------------------------------ samples

def smp_file( m ): return os.path.join( LRAW, 'smp.%s.%s.%s.npz' % ( m[ 'key' ], m[ 'level' ], m.get( 'smp', m[ 'variant' ] ) ) ) # (`smp`: a variant with the faces of another is judged on that one's points)
SMP_VER = 2
def smp_have( m, nt ):

	fn = smp_file( m )
	if not os.path.exists( fn ): return None
	z = np.load( fn )
	return z if int( z[ 'nt' ] ) == nt and 'ver' in z.files and int( z[ 'ver' ] ) == SMP_VER else None

def sample_points( m, P, rec, tri, f, B ):

	# the lattice and the points to judge by on the lit faces f (the same every time they are asked for), and both in
	# one row for project(): place, normal, spacing, material, radius to look in
	s = float( B.get( 's', { 'lo': 0.10, 'mid': 0.07, 'near': 0.05 }[ m[ 'level' ] ] ) )
	face, bc, X, N, Ls = face_samples( P, tri, f, s, int( B.get( 'kmax', 32 ) ) )
	seed = ( sum( ord( ch ) * ( i + 1 ) for i, ch in enumerate( m[ 'key' ] + m[ 'level' ] + m[ 'variant' ] ) ) * 7919 ) % ( 1 << 31 )
	eface, eb, eX, eN, ew = rand_samples( P, tri, f, float( B.get( 'rho', 150 ) ), seed )
	aface = np.concatenate( [ face, eface ] )
	return ( face, bc, X, eface, eb, eX, ew, len( face ), aface, np.concatenate( [ bc, eb ] ), np.concatenate( [ X, eX ] ), np.concatenate( [ N, eN ] ), np.concatenate( [ Ls, np.full( len( eface ), 0.04 ) ] ),
		rec[ tri[ aface ][ :, 0 ], 3 ].astype( np.int64 ), float( B.get( 'radius', { 'lo': 0.6, 'mid': 0.35, 'near': 0.25 }[ m[ 'level' ] ] ) ) )

def vis_file( m ): return os.path.join( LRAW, 'vis.%s.%s.%s.npz' % ( m[ 'key' ], m[ 'level' ], m.get( 'smp', m[ 'variant' ] ) ) )
VIS_RAYS = 16; VIS_FAR = ( 0.3, 1.0, 3.0 ); VIS_USE = 1
def mesh_vis( m, P, rec, tri, f, B, S ):

	# Which points can be seen at all: project() only knows a point within a few cm of the edge of what covers it; a wall
	# that stands inside the next house, or behind a stall pushed against it, is dark on the full model and nobody
	# ever looks at it - no target for a fit, and nothing to judge by. free_rays(): of VIS_RAYS rays over the half
	# space of the point on the full model, how many go VIS_FAR metres without meeting anything, or leave the town
	# (a ray that ends on the back of a face started inside something), one column for each distance of VIS_FAR and one
	# with the rays that ended on a back. 0 = hidden; 255 = not found on the full model.
	# -> v (the lattice), ev (the points to judge by); kept beside the samples
	fn = vis_file( m )
	if os.path.exists( fn ):

		z = np.load( fn )
		if z[ 'v' ].ndim == 2 and len( z[ 'v' ] ) == len( S[ 'face' ] ) and len( z[ 'ev' ] ) == len( S[ 'eface' ] ): return z[ 'v' ], z[ 'ev' ]

	if not HAVE_FINE: return None, None
	t0 = time.time()
	face, bc, X, eface, eb, eX, ew, nfit, aface, ab, aX, aN, aL, LAY, R = sample_points( m, P, rec, tri, f, B )
	assert len( face ) == len( S[ 'face' ] ) and len( eface ) == len( S[ 'eface' ] ) and np.abs( eX - S[ 'eX' ] ).max() < 1e-3, 'the points of %s %s %s are not the ones that were sampled' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ] )
	v = np.full( ( len( aface ), len( VIS_FAR ) + 1 ), 255, np.uint8 )
	if len( aface ):

		pb = fine_near( m[ 'key' ], P.min( 0 ), P.max( 0 ) )
		pr = project( pb, aX, aN, LAY, aL, R )
		hit = pr[ :, 0 ] >= 0
		if hit.any(): v[ hit ] = free_rays( pb, pr[ hit ], VIS_RAYS, VIS_FAR )

	np.savez( fn, v = v[ :nfit ], ev = v[ nfit: ], rays = VIS_RAYS, far = VIS_FAR )
	llog( '  seen %s %s %s: %d points; no ray of %d free for %s m: %s %%; all rays end on a back: %.1f %%; not found %.1f %%; %.0fs' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ], len( v ), VIS_RAYS,
		' / '.join( '%g' % d for d in VIS_FAR ), ' / '.join( '%.1f' % ( 100 * float( ( v[ :, j ] == 0 ).mean() ) ) for j in range( len( VIS_FAR ) ) ),
		100 * float( ( v[ :, -1 ] == VIS_RAYS ).mean() ), 100 * float( ( v[ :, 0 ] == 255 ).mean() ), time.time() - t0 ) )
	return v[ :nfit ], v[ nfit: ]

def mesh_samples( m, P, rec, tri, f, B ):

	# the two sets of points on the lit faces f of a mesh, with the full model's light:
	#   face, b, X, y (rgb + gloss), q (1 in the open, 0.1 inside something, 0.05 not found on the full model: the
	#   mesh's own light), yr / qr (the raw level's light, found there) - the lattice a fit may use
	#   eface, eb, eX, ey, eq, eyr, eqr, ew (area) - the points every variant is judged by
	# The bake's light at a point is a few rays (3 to a lantern's 14 cm source, 12 of the sky's occlusion, turned by a hash
	# of the place): a noisy estimate. The lattice has it as the bake would (y: a fit averages it out). The points to
	# judge by have it twice: eyn as the bake would, and ey with `truth_rays` times the rays - the reference.
	fn = smp_file( m )
	z = smp_have( m, len( tri ) )
	if z is not None:

		S = { k: z[ k ].astype( np.int64 if k in ( 'face', 'eface' ) else np.float64 ) for k in z.files if k not in ( 'nt', 'secs', 'ver' ) }
		S[ 'secs' ] = z[ 'secs' ].tolist()
		# (what no eye can reach - mesh_vis() - is 0.2 from here on: no target, nothing to judge by. `vis`: which distance
		# of VIS_FAR a ray has to go free, - 1 = project()'s few cm only)
		# (`fit_vis`: the same for what the FIT may look at alone - qf / qrf - while every variant is judged by `vis`:
		# a fit with a cheaper test of what is hidden, or none, measured on the same points as the others)
		j = int( B.get( 'vis', VIS_USE ) ); jf = int( B.get( 'fit_vis', j ) )
		S[ 'qf' ] = S[ 'q' ].copy(); S[ 'qrf' ] = S[ 'qr' ].copy()
		if ( j >= 0 or jf >= 0 ) and os.path.exists( vis_file( m ) ):

			zv = np.load( vis_file( m ) )
			if zv[ 'v' ].ndim == 2 and len( zv[ 'v' ] ) == len( S[ 'face' ] ) and len( zv[ 'ev' ] ) == len( S[ 'eface' ] ):

				for kv, ks, jj in ( ( 'v', ( 'q', 'qr' ), j ), ( 'ev', ( 'eq', 'eqr' ), j ), ( 'v', ( 'qf', 'qrf' ), jf ) ):

					if jj < 0: continue
					h = zv[ kv ][ :, jj ] == 0
					for k in ks: S[ k ] = np.where( h & ( S[ k ] > 0.5 ), 0.2, S[ k ] )

		return S

	if not HAVE_FINE: sys.exit( 'lightmap: no samples of %s %s %s - run the sample stage' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ] ) )
	t0 = time.time()
	face, bc, X, eface, eb, eX, ew, nfit, aface, ab, aX, aN, aL, LAY, R = sample_points( m, P, rec, tri, f, B )
	lit, gl = vertex_light( rec ); own4 = np.concatenate( [ lit, gl[ :, None ] ], 1 ); tv = tri[ aface ]
	y = own4[ tv[ :, 0 ] ] * ab[ :, 0:1 ] + own4[ tv[ :, 1 ] ] * ab[ :, 1:2 ] + own4[ tv[ :, 2 ] ] * ab[ :, 2:3 ]
	q = np.full( len( aface ), 0.05 )
	secs = [ 0.0, 0.0, 0.0, 0.0 ] # project, light, raw target, the reference's rays
	yc = None
	if len( aface ):

		pb = fine_near( m[ 'key' ], P.min( 0 ), P.max( 0 ) )
		t1 = time.time()
		pr = project( pb, aX, aN, LAY, aL, R )
		hit = pr[ :, 0 ] >= 0
		t2 = time.time()
		if hit.any():

			c, g = fine_light( pb, pr[ hit ], 'vs' )
			y[ hit, :3 ] = c; y[ hit, 3 ] = g
			q[ hit ] = np.where( pr[ hit, 4 ] > 0.5, 1.0, 0.1 )

		t3 = time.time()
		yc = y[ nfit: ].copy(); eh = hit[ nfit: ]
		mul = int( B.get( 'truth_rays', 6 ) )
		if eh.any() and mul > 1:

			keep = ( BK[ 'light_rays' ], BK[ 'ao_rays' ] )
			BK[ 'light_rays' ] = keep[ 0 ] * mul; BK[ 'ao_rays' ] = keep[ 1 ] * mul
			try: yc[ eh, :3 ] = fine_light( pb, pr[ nfit: ][ eh ], 've' )[ 0 ]
			finally: BK[ 'light_rays' ], BK[ 'ao_rays' ] = keep

		t4 = time.time()
		yr, hr = raw_target( m[ 'key' ], aX, aN, LAY, R )
		secs = [ t2 - t1, t3 - t2, time.time() - t4, t4 - t3 ]

	else: yr = np.zeros( ( 0, 4 ) ); hr = np.zeros( 0, bool )
	f32 = np.float32
	if yc is None: yc = y[ nfit: ]
	np.savez( fn, nt = len( tri ), ver = SMP_VER, secs = np.array( secs ), eyn = y[ nfit: ].astype( f32 ),
		face = face.astype( np.int32 ), b = bc.astype( f32 ), X = X.astype( f32 ), y = y[ :nfit ].astype( f32 ), q = q[ :nfit ].astype( f32 ), yr = yr[ :nfit ].astype( f32 ), qr = hr[ :nfit ].astype( f32 ),
		eface = eface.astype( np.int32 ), eb = eb.astype( f32 ), eX = eX.astype( f32 ), ey = yc.astype( f32 ), eq = q[ nfit: ].astype( f32 ), eyr = yr[ nfit: ].astype( f32 ), eqr = hr[ nfit: ].astype( f32 ), ew = ew.astype( f32 ) )
	llog( '  samples %s %s %s: %d on the lattice, %d to judge by; found %.1f %%, in the open %.1f %%, on the raw level %.1f %%; project %.0fs light %.0fs raw %.0fs reference %.0fs' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ], nfit, len( eface ),
		100 * float( ( q > 0.07 ).mean() ) if len( q ) else 0, 100 * float( ( q > 0.5 ).mean() ) if len( q ) else 0, 100 * float( hr.mean() ) if len( hr ) else 0, secs[ 0 ], secs[ 1 ], secs[ 2 ], secs[ 3 ] ) )
	return mesh_samples( m, P, rec, tri, f, B )

# ------------------------------------------------------------------------------------------ the fit

def tone_slope( v ):

	# the slope of the page's tone curve about a value (a secant over a quarter of it: finite in the dark)
	d = np.maximum( 0.25 * v, 0.004 ); lo = np.maximum( v - d, 0 )
	return ( tone( v + d ) - tone( lo ) ) / ( v + d - lo )

def fit_values( rec, flags, tri, f, P, a2, S, B, hold_v ):

	# -> dict( cv: variable of each corner of the lit faces f (nf, 3), x: light + gloss of each variable (K, 4),
	#          ferr: what the lattice says is left on each face (display steps, rms), stats )
	nf = len( f )
	fmap = np.full( len( tri ), - 1, np.int64 ); fmap[ f ] = np.arange( nf )
	sf = fmap[ S[ 'face' ] ]; ok = sf >= 0
	sf = sf[ ok ]; b = S[ 'b' ][ ok ]
	lit, gl = vertex_light( rec ); own4 = np.concatenate( [ lit, gl[ :, None ] ], 1 )
	if B.get( 'target' ) == 'raw':

		# (no ray: the raw level's light; where it has none, the mesh's own, which counts little)
		tv = tri[ f ][ sf ]
		mine = own4[ tv[ :, 0 ] ] * b[ :, 0:1 ] + own4[ tv[ :, 1 ] ] * b[ :, 1:2 ] + own4[ tv[ :, 2 ] ] * b[ :, 2:3 ]
		got = S[ 'qrf' ][ ok ] > 0.5
		y = np.where( got[ :, None ], S[ 'yr' ][ ok ], mine ); q = np.where( got, 1.0, np.where( S[ 'qrf' ][ ok ] > 0.07, float( B.get( 'hidden', 0 ) ), 0.05 ) )

	else:

		# (a point inside something, or behind it, is no target: `hidden` is what it counts - 0. The points not found on
		# the full model have the mesh's own light: they hold a face nothing else speaks for where it is)
		y = S[ 'y' ][ ok ]; q = S[ 'qf' ][ ok ]
		q = np.where( q > 0.5, 1.0, np.where( q > 0.07, float( B.get( 'hidden', 0 ) ), 0.05 ) )

	ar = a2[ f ] * 0.5
	nper = np.bincount( sf, minlength = nf )
	aw = ar[ sf ] / np.maximum( nper[ sf ], 1 )
	# What a sample counts (the unknowns are linear light; the eye is not):
	#   none   nothing: least squares on radiance - the mean light of a face is kept (what a far pixel sums up), the dark
	#          next to a light takes some of it
	#   tone   the slope of the page's tone curve AT THE SAMPLE, squared: least squares on what the screen shows, as far as
	#          a line through the target tells it - steep in the dark (near 1000 x between no light and a lit wall), so a
	#          vertex between light and dark goes to the dark: the lit side loses more than the measure sees
	#   sqrt   the same with the square root the colours are stored in (1 / 4 y: milder)
	#   face   the tone curve's slope at the MEAN light of the sample's face: one weight over a face (its mean is kept),
	#          a dark wall still counts as much as a lit one
	#   irls rounds > 0 (tone, sqrt): again with the slope between what the fit shows and what it should - the true
	#          least squares in that space, symmetric between too dark and too bright
	how = B.get( 'weight', 'none' )
	curve = ( lambda v: np.sqrt( np.maximum( v, 0 ) ) ) if how == 'sqrt' else tone
	wpow = float( B.get( 'wpow', 1 ) ) # (face: 1 = the slope squared, 0.5 = the slope, 0 = none)
	def slope_at( v ):

		if how == 'sqrt': d = np.maximum( 0.25 * v, 0.004 ); lo = np.maximum( v - d, 0 ); return ( np.sqrt( v + d ) - np.sqrt( lo ) ) / ( v + d - lo )
		return tone_slope( v )

	qa = q * aw
	fw = np.maximum( np.bincount( sf, qa, nf ), 1e-30 )
	ybar = np.stack( [ np.bincount( sf, qa * y[ :, c ], nf ) for c in range( 4 ) ], 1 ) / fw[ :, None ] # the mean light of each face's samples
	def weights( at ):

		w = np.ones( ( len( sf ), 4 ) )
		if how in ( 'tone', 'sqrt' ): w[ :, :3 ] = slope_at( y[ :, :3 ] ) ** 2
		elif how == 'face': w[ :, :3 ] = ( tone_slope( ybar[ :, :3 ] ) ** ( 2 * wpow ) )[ sf ]
		elif how == 'lum': w[ :, :3 ] = ( 1 / ( y[ :, :3 ] @ LUMW + 0.03 ) )[ :, None ]
		if at is not None and how in ( 'tone', 'sqrt' ):

			d = np.abs( at[ :, :3 ] - y[ :, :3 ] ); dm = np.maximum( 0.05 * y[ :, :3 ], 0.002 )
			sl = np.abs( curve( at[ :, :3 ] ) - curve( y[ :, :3 ] ) ) / np.maximum( d, 1e-12 )
			w[ :, :3 ] = np.where( d > dm, sl, slope_at( y[ :, :3 ] ) ) ** 2

		w[ :, 3 ] = float( B.get( 'gloss_weight', 0.05 ) )
		return w * qa[ :, None ]

	def gram( w ):

		G = np.zeros( ( nf, 4, 3, 3 ) ); h = np.zeros( ( nf, 3, 4 ) )
		for c in range( 4 ):

			for i in range( 3 ):

				wb = w[ :, c ] * b[ :, i ]
				h[ :, i, c ] = np.bincount( sf, wb * y[ :, c ], nf )
				for j in range( i, 3 ): G[ :, c, i, j ] = G[ :, c, j, i ] = np.bincount( sf, wb * b[ :, j ], nf )

		sw = np.stack( [ np.bincount( sf, w[ :, c ], nf ) for c in range( 4 ) ], 1 )
		mu = float( B.get( 'mean', 1 ) )
		if mu > 0:

			# the mean of a face: ( x0 + x1 + x2 ) / 3 held to the mean light of its samples, mu times as heavy as they
			# are together at the weight the face's mean light has
			wm = mu * fw[ :, None ] * np.concatenate( [ tone_slope( ybar[ :, :3 ] ) ** ( 2 * wpow if how == 'face' else 2 ) if how != 'none' else np.ones( ( nf, 3 ) ), np.full( ( nf, 1 ), float( B.get( 'gloss_weight', 0.05 ) ) ) ], 1 )
			G += ( wm / 9 )[ :, :, None, None ]
			h += ( wm * ybar / 3 )[ :, None, :]

		return G, h, sw

	w = weights( None )
	G, h, sw = gram( w )
	has = sw[ :, 0 ] > 0
	tf = tri[ f ]; r = tf.ravel()
	place = np.unique( pkey( rec ), return_inverse = True )[ 1 ].ravel()
	hold = hold_v[ r ]
	mode = B.get( 'mode', 'keep' )
	if mode == 'keep':

		# the vertices as they are: the records of one place that carry one light today stay one value
		ids = np.unique( np.concatenate( [ place[ r ][ :, None ], rec[ r, 6:10 ].astype( np.int64 ) ], 1 ), axis = 0, return_inverse = True )[ 1 ].ravel()
		xv = own4[ r ]

	else:

		# each face on its own first: what its three corners would like to be (held to the range of its samples) ...
		loc = own4[ tf ].copy()
		if has.any():

			ym = np.stack( [ np.bincount( sf, w[ :, c ] * y[ :, c ], nf ) for c in range( 4 ) ], 1 )[ has ] / np.maximum( sw[ has ], 1e-30 )
			lo = np.full( ( nf, 4 ), 1e9 ); hi = np.full( ( nf, 4 ), - 1e9 )
			np.minimum.at( lo, sf, y ); np.maximum.at( hi, sf, y )
			trc = np.maximum( np.trace( G[ has ], axis1 = 2, axis2 = 3 ), 1e-30 ) / 3 # (nh, 4)
			eps = 0.03
			sol = np.linalg.solve( G[ has ] + ( eps * trc )[ :, :, None, None ] * np.eye( 3 )[ None, None ], ( h[ has ].transpose( 0, 2, 1 ) + ( eps * trc )[ :, :, None ] * ym[ :, :, None ] )[ ..., None ] )[ ..., 0 ]
			loc[ has ] = np.clip( sol.transpose( 0, 2, 1 ), lo[ has ][ :, None, : ], hi[ has ][ :, None, : ] )

		if B.get( 'wish' ) == 'near':

			# (or: what the full model shows NEAR the corner inside this face - the mean of the face's samples, each by
			# its weight of that corner to a power. A face's own best line overshoots at its corners wherever the light
			# bends, and two faces of one smooth wall then ask for different values at the vertex they share: a split
			# there shows as a hard step in a soft gradient. The light near the corner differs between two faces only
			# where the light itself has an edge between them)
			pw = float( B.get( 'near_pow', 4 ) )
			for i in range( 3 ):

				wi = w * ( b[ :, i:i + 1 ] ** pw )
				num = np.stack( [ np.bincount( sf, wi[ :, c ] * y[ :, c ], nf ) for c in range( 4 ) ], 1 )
				den = np.stack( [ np.bincount( sf, wi[ :, c ], nf ) for c in range( 4 ) ], 1 )
				loc[ :, i, : ] = np.where( den > 1e-30, num / np.maximum( den, 1e-30 ), loc[ :, i, : ] )

		# ... then the corners of a place grouped by that, within tol display steps (a corner that may not move: by the light it has)
		val = np.where( hold[ :, None ], own4[ r ], loc.reshape( - 1, 4 ) )
		d = tone( val[ :, :3 ] )
		tol = np.where( hold, float( B.get( 'keep_tol', 1 ) ), max( float( B.get( 'tol', 6 ) ), 1e-3 ) ) / 255
		ids = place[ r ] * 2 + hold
		ids = leader( ids, ( d @ LUMW ) / tol, 1.0 )
		ids = leader( ids, ( d[ :, 0 ] - d[ :, 2 ] ) / ( tol * 1.5 ), 1.0 )
		ids = leader( ids, val[ :, 3 ] / np.where( hold, 0.02, float( B.get( 'gloss_tol', 0.25 ) ) ), 1.0 )
		xv = val

	K = int( ids.max() ) + 1 if len( ids ) else 0
	cv = ids.reshape( nf, 3 )
	cnt = np.maximum( np.bincount( ids, minlength = K ), 1 )
	def acc( idx, v ): return np.stack( [ np.bincount( idx, v[ :, c ], K ) for c in range( v.shape[ 1 ] ) ], 1 )
	x0 = acc( ids, xv ) / cnt[ :, None ]
	fixed = np.bincount( ids, hold, K ) > 0
	# the edge term: pairs of faces over an edge both run along with the same two variables, in one flat piece
	alpha = float( B.get( 'alpha', 0.1 ) ); reg = B.get( 'reg', 'lin' )
	Mp = None; npairs = 0
	if alpha > 0 and nf:

		Pf = P[ tf ]
		fnv = np.cross( Pf[ :, 1 ] - Pf[ :, 0 ], Pf[ :, 2 ] - Pf[ :, 0 ] )
		nrm = fnv / np.maximum( np.linalg.norm( fnv, axis = 1 ), 1e-20 )[ :, None ]
		el = Pf[ :, [ 1, 2, 0 ] ] - Pf # edge k: from corner k to k + 1
		slim = ( el ** 2 ).sum( 2 ).max( 1 ) / np.maximum( 2 * ar, 1e-20 )
		e0 = cv; e1 = cv[ :, [ 1, 2, 0 ] ]
		ek = ( np.minimum( e0, e1 ) * K + np.maximum( e0, e1 ) ).ravel()
		o = np.argsort( ek, kind = 'stable' )
		uq, first, n = np.unique( ek[ o ], return_index = True, return_counts = True )
		two = n == 2
		i0 = o[ first[ two ] ]; i1 = o[ first[ two ] + 1 ]
		t = i0 // 3; kt = i0 % 3; u = i1 // 3; ku = i1 % 3
		good = ( cv[ t, kt ] != cv[ t, ( kt + 1 ) % 3 ] ) & ( cv[ t, kt ] == cv[ u, ( ku + 1 ) % 3 ] ) & ( t != u ) & ( ( nrm[ t ] * nrm[ u ] ).sum( 1 ) > 0.9 ) & ( slim[ t ] < 200 ) & ( slim[ u ] < 200 )
		t, kt, u, ku = t[ good ], kt[ good ], u[ good ], ku[ good ]
		npairs = len( t )
		if npairs:

			# the gradient of the hat of corner m of a face: n x ( the edge opposite, from m + 1 to m + 2 ) / 2 a
			gh = np.cross( nrm[ :, None, : ], el[ :, [ 1, 2, 0 ] ] ) / np.maximum( 2 * ar, 1e-20 )[ :, None, None ]
			dd = el[ t, kt ]; dd = dd / np.maximum( np.linalg.norm( dd, axis = 1 ), 1e-20 )[ :, None ]
			def grad( fc ):

				# ... in the frame of the pair: along the shared edge, and across it (the two faces laid out flat)
				pp = np.cross( nrm[ fc ], dd )
				g = np.stack( [ np.einsum( 'pmx,px->pm', gh[ fc ], dd ), np.einsum( 'pmx,px->pm', gh[ fc ], pp ) ], 1 )
				# (the paper's eq. 13 scales each face's gradient by the root of its area; "lin" does not - a light that
				# is one linear function over both faces then costs nothing whatever their sizes)
				return np.sqrt( ar[ fc ] )[ :, None, None ] * g if reg == 'paper' else g

			Mp = np.concatenate( [ grad( t ), - grad( u ) ], 2 ) # pairs, 2, 6
			idx6 = np.concatenate( [ cv[ t ], cv[ u ] ], 1 )
			wgeo = ( ar[ t ] + ar[ u ] ) if reg == 'paper' else 0.5 * ( ar[ t ] + ar[ u ] ) ** 2

	lam = float( B.get( 'lam', 1e-4 ) )
	free = ( ~ fixed )[ :, None ] & np.ones( ( 1, 4 ), bool )
	x = x0.copy(); its = 0
	def solve( x, free, G, h, sw ):

		dm = acc( ids, np.stack( [ G[ :, :, 0, 0 ], G[ :, :, 1, 1 ], G[ :, :, 2, 2 ] ], 1 ).reshape( - 1, 4 ) )
		D0 = np.maximum( dm, 1e-6 * np.maximum( dm.mean( 0 ), 1e-30 )[ None ] )
		Dj = dm + lam * D0
		if Mp is not None:

			ew = sw / np.maximum( ar, 1e-20 )[ :, None ]; ew = np.where( has[ :, None ], ew, ew[ has ].mean( 0 )[ None ] if has.any() else 1.0 )
			Wp = alpha * wgeo[ :, None ] * 0.5 * ( ew[ t ] + ew[ u ] ) # pairs, 4
			Dj = Dj + acc( idx6.ravel(), ( ( Mp ** 2 ).sum( 1 )[ :, :, None ] * Wp[ :, None, : ] ).reshape( - 1, 4 ) )

		def mv( v ):

			out = acc( ids, np.einsum( 'tcij,tjc->tic', G, v[ cv ] ).reshape( - 1, 4 ) ) + lam * D0 * v
			if Mp is not None:

				g = np.einsum( 'pij,pjc->pic', Mp, v[ idx6 ] ) * Wp[ :, None, : ]
				out += acc( idx6.ravel(), np.einsum( 'pij,pic->pjc', Mp, g ).reshape( - 1, 4 ) )

			return out

		rhs = acc( ids, h.reshape( - 1, 4 ) ) + lam * D0 * x0
		rr = ( rhs - mv( x ) ) * free; z = rr / Dj; p = z.copy(); rz = ( rr * z ).sum( 0 ); r0 = np.maximum( rz, 1e-30 )
		it = 0
		for it in range( int( B.get( 'cg', 500 ) ) ):

			Ap = mv( p ) * free; al = rz / np.maximum( ( p * Ap ).sum( 0 ), 1e-30 )
			x = x + al[ None ] * p; rr = rr - al[ None ] * Ap
			z = rr / Dj; rz2 = ( rr * z ).sum( 0 )
			if ( rz2 / r0 ).max() < 1e-10: break
			p = z + ( rz2 / np.maximum( rz, 1e-30 ) )[ None ] * p; rz = rz2

		return x, it + 1

	def at_samples( v ): return v[ cv[ sf, 0 ] ] * b[ :, 0:1 ] + v[ cv[ sf, 1 ] ] * b[ :, 1:2 ] + v[ cv[ sf, 2 ] ] * b[ :, 2:3 ]
	clamped = 0.0
	if K:

		vlo = np.zeros( ( K, 4 ) ); vhi = np.full( ( K, 4 ), 64.0 ); vhi[ :, 3 ] = 1
		if B.get( 'clamp' ) == 'range':

			# (the paper clamps to the valid range; this, to what the samples about the vertex show - no light where none is)
			vlo = np.full( ( K, 4 ), 1e9 ); vhi = np.full( ( K, 4 ), - 1e9 )
			for k in range( 3 ): np.minimum.at( vlo, cv[ sf, k ], y ); np.maximum.at( vhi, cv[ sf, k ], y )
			none = vlo[ :, 0 ] > vhi[ :, 0 ]
			vlo[ none ] = x0[ none ]; vhi[ none ] = x0[ none ]
			vlo = np.maximum( vlo, 0 )

		for rnd in range( 1 + ( int( B.get( 'irls', 1 ) ) if how in ( 'tone', 'sqrt' ) else 0 ) ):

			if rnd: G, h, sw = gram( weights( at_samples( np.clip( x, vlo, vhi ) ) ) )
			fr = free
			for cl in range( 3 ):

				x, n = solve( np.where( fr, x, np.clip( x, vlo, vhi ) ), fr, G, h, sw ); its += n
				# over- and undershoot (next to a hard edge of the light): held, and the others solved again around it
				out = ( ( x < vlo ) | ( x > vhi ) ) & fr
				if rnd == 0 and cl == 0: clamped = float( out[ :, :3 ].any( 1 ).sum() ) / max( 1, int( ( ~ fixed ).sum() ) )
				if not out.any(): break
				fr = fr & ~ out

			x = np.where( free, np.clip( x, vlo, vhi ), x0 )

	# what is left on each face, by the lattice (what a refinement may go by)
	e = np.abs( tone( at_samples( x )[ :, :3 ] ) - tone( y[ :, :3 ] ) ).max( 1 ) * 255
	wa = aw * ( q > 0.5 )
	ferr = np.sqrt( np.bincount( sf, wa * e * e, nf ) / np.maximum( np.bincount( sf, wa, nf ), 1e-30 ) )
	return dict( cv = cv, x = x, ferr = ferr, stats = dict( vars = K, fixed = int( fixed.sum() ), pairs = int( npairs ), cg = its, clamped = round( clamped, 4 ) ) )

# ------------------------------------------------------------------------------------------ the measure

def evaluate( rec, tri, S, target = 'ey' ):

	# the mesh as the page shows it against the full model's light at the points to judge by
	# -> error of each point (display steps), whether it counts (in the open on the full model)
	lit = lin13( rec[ :, 6:9 ] )
	tv = tri[ S[ 'eface' ] ]; b = S[ 'eb' ]
	g = lit[ tv[ :, 0 ] ] * b[ :, 0:1 ] + lit[ tv[ :, 1 ] ] * b[ :, 1:2 ] + lit[ tv[ :, 2 ] ] * b[ :, 2:3 ]
	e = np.abs( tone( g ) - tone( S[ target ][ :, :3 ] ) ).max( 1 ) * 255
	return e, ( S[ 'eq' ] > 0.5 ) if target == 'ey' else ( S[ 'eqr' ] > 0.5 ), g

def face_normals( m, rec, tri ):

	P = np.array( m[ 'min' ], np.float64 )[ None ] + rec[ :, :3 ].astype( np.float64 ) * m[ 'q' ]
	n = np.cross( P[ tri[ :, 1 ] ] - P[ tri[ :, 0 ] ], P[ tri[ :, 2 ] ] - P[ tri[ :, 0 ] ] )
	return n / np.maximum( np.linalg.norm( n, axis = 1 ), 1e-20 )[ :, None ]

def blur_error( g, y, X, n, w, ok, kind, h ):

	# the same over patches: the mean light of the mesh and of the full model over the points of one box of h metres
	# that face the same way and are of one kind - what is left when nothing finer than h shows (a level seen from far,
	# or the eye that does not miss a detail but does see a wall of the wrong brightness)
	# -> error of each patch, its area, its kind
	i = np.nonzero( ok )[ 0 ]
	if not len( i ): return np.zeros( 0 ), np.zeros( 0 ), np.zeros( 0, np.int64 )
	c = np.floor( X[ i ] / h ).astype( np.int64 ); c -= c.min( 0 )[ None ]
	ax = np.abs( n[ i ] ).argmax( 1 )
	ax = ax * 2 + ( n[ i ][ np.arange( len( i ) ), ax ] < 0 )
	key = ( ( c[ :, 0 ] * 4096 + c[ :, 1 ] ) * 4096 + c[ :, 2 ] ) * 32 + ax * 4 + kind[ i ]
	u, inv = np.unique( key, return_inverse = True ); inv = inv.ravel(); K = len( u )
	W = np.bincount( inv, w[ i ], K )
	mg = np.stack( [ np.bincount( inv, w[ i ] * g[ i, k ], K ) for k in range( 3 ) ], 1 ) / W[ :, None ]
	my = np.stack( [ np.bincount( inv, w[ i ] * y[ i, k ], K ) for k in range( 3 ) ], 1 ) / W[ :, None ]
	return np.abs( tone( mg ) - tone( my ) ).max( 1 ) * 255, W, u % 4

def wstats( e, w ):

	if not len( e ) or w.sum() <= 0: return None
	o = np.argsort( e ); es = e[ o ]; cw = np.cumsum( w[ o ] ); W = float( cw[ - 1 ] )
	pc = lambda p: round( float( es[ min( len( es ) - 1, int( np.searchsorted( cw, p * W ) ) ) ] ), 2 )
	sh = lambda t: round( float( w[ e > t ].sum() ) / W, 4 )
	return dict( area = round( W, 1 ), mean = round( float( ( e * w ).sum() / W ), 3 ), rms = round( float( np.sqrt( ( e * e * w ).sum() / W ) ), 3 ), p50 = pc( 0.5 ), p95 = pc( 0.95 ), p99 = pc( 0.99 ),
		over2 = sh( 2 ), over4 = sh( 4 ), over8 = sh( 8 ) )

class Tally:

	# the errors of a bundle over all its meshes, all together and by the kind of surface
	# (with the light itself - g the mesh's, y what it is judged by - also the signed error: `bias`, the mean of
	# tone( g ) - tone( y ) in display steps (luminance; below 0: the mesh is darker), and `lin`, the mesh's sum of
	# linear light over the other's - what a far pixel makes of it)
	def __init__( self ): self.e = []; self.w = []; self.k = []; self.inside = 0.0; self.total = 0.0; self.sg = np.zeros( ( 4, 4 ) )
	def patches( self, r ): self.add( r[ 0 ], np.ones( len( r[ 0 ] ), bool ), r[ 1 ], r[ 2 ] )
	def add( self, e, ok, w, kind, g = None, y = None ):

		self.e.append( e[ ok ] ); self.w.append( w[ ok ] ); self.k.append( kind[ ok ] )
		self.inside += float( w[ ~ ok ].sum() ); self.total += float( w.sum() )
		if g is not None:

			wo = w[ ok ]; ko = kind[ ok ]
			for i, v in enumerate( ( np.ones( len( wo ) ), signed( g[ ok ], y[ ok ] ), g[ ok ][ :, :3 ] @ LUMW, y[ ok ][ :, :3 ] @ LUMW ) ): self.sg[ :, i ] += np.bincount( ko, wo * v, 4 )

	def stats( self ):

		e = np.concatenate( self.e ) if self.e else np.zeros( 0 ); w = np.concatenate( self.w ) if self.w else np.zeros( 0 ); k = np.concatenate( self.k ) if self.k else np.zeros( 0, np.int64 )
		out = dict( all = wstats( e, w ), left_out = round( self.inside / max( self.total, 1e-30 ), 4 ) )
		for i, name in enumerate( KINDS ): out[ name ] = wstats( e[ k == i ], w[ k == i ] )
		if self.sg[ :, 0 ].sum() > 0:

			for name, r in [ ( 'all', self.sg.sum( 0 ) ) ] + [ ( name, self.sg[ i ] ) for i, name in enumerate( KINDS ) ]:

				if out.get( name ) and r[ 0 ] > 0: out[ name ].update( bias = round( float( r[ 1 ] / r[ 0 ] ), 3 ), lin = round( float( r[ 2 ] / max( r[ 3 ], 1e-30 ) ), 4 ) )

		return out

def signed( g, y ): return ( tone( g[ :, :3 ] ) @ LUMW - tone( y[ :, :3 ] ) @ LUMW ) * 255

def face_signed( g, ok, S, nt ):

	# the mean signed error of the points on each face (display steps; below 0: darker than the full model)
	w = S[ 'ew' ] * ok
	return np.bincount( S[ 'eface' ], w * signed( g, S[ 'ey' ] ), nt ) / np.maximum( np.bincount( S[ 'eface' ], w, nt ), 1e-30 )

def face_error( e, ok, S, nt ):

	# rms of the points on each face (a face with none that counts: 0)
	w = S[ 'ew' ] * ok
	return np.sqrt( np.bincount( S[ 'eface' ], w * e * e, nt ) / np.maximum( np.bincount( S[ 'eface' ], w, nt ), 1e-30 ) )

# ------------------------------------------------------------------------------------------ meshes out

def mesh_sub( ent, kinds ):

	sub = []
	for i in range( len( kinds ) ):

		w_ = np.nonzero( ent == i )[ 0 ]
		if len( w_ ): assert w_[ - 1 ] - w_[ 0 ] + 1 == len( w_ ); sub.append( [ int( kinds[ i ] ), int( w_[ 0 ] ) * 3, len( w_ ) * 3 ] )

	return sub

def emit_mesh( m, bname, rec, flags, tri, f, ent, kinds, nrec, nfl, cidx, st, write = True ):

	# the mesh with the records nrec (indexed by cidx: len( f ), 3) for its lit faces, the others as they were
	K = len( nrec )
	ntri = tri + K; ntri[ f ] = cidx
	up, inv = np.unique( ntri.ravel(), return_inverse = True )
	allrec = np.concatenate( [ nrec, rec ] )[ up ]; allfl = np.concatenate( [ nfl, flags ] )[ up ]
	ntri = inv.reshape( - 1, 3 )
	fnm = '%s.%s.%s.%s.bin' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ], bname )
	if write:

		with open( os.path.join( LRAW, fnm ), 'wb' ) as fh: fh.write( allrec.astype( '<u2' ).tobytes() ); fh.write( ntri.astype( '<u4' ).tobytes() ); fh.write( allfl.astype( np.uint8 ).tobytes() )

	return dict( key = m[ 'key' ], level = m[ 'level' ], variant = m[ 'variant' ], raw = fnm, min = m[ 'min' ], q = m[ 'q' ], nv = int( len( allrec ) ), ni = int( ntri.size ), sub = mesh_sub( ent, kinds ), lm = st ), allrec, ntri

def res_mesh( m, name, rec, flags, tri, f, ent, kinds, col ):

	# what is left on each lit face as a colour (heat() / heat_signed() of it), its corners its own
	r = tri[ f ].ravel()
	nrec = rec[ r ].copy()
	nrec[ :, 6:9 ] = enc16( np.repeat( col, 3, 0 ) ); nrec[ :, 9 ] = 0
	return emit_mesh( m, name, rec, flags, tri, f, ent, kinds, nrec, flags[ r ], np.arange( len( r ) ).reshape( - 1, 3 ), {} )[ 0 ]

# ------------------------------------------------------------------------------------------ a bundle of fitted meshes

def open_mesh( m ):

	M = load_mesh( m )
	rec, tri, flags = M[ 'rec' ], M[ 'tri' ], M[ 'flags' ]
	ent = np.zeros( len( tri ), np.int64 )
	for i, ( kind, start, count ) in enumerate( m[ 'sub' ] ): ent[ start // 3:( start + count ) // 3 ] = i
	kinds = np.array( [ q[ 0 ] for q in m[ 'sub' ] ], np.int64 )
	P = mesh_pos( M )
	lmf, a2 = lit_faces( rec, tri, ent, kinds, P )
	return M, rec, tri, flags, ent, kinds, P, lmf, a2

def sample_meshes( meshes, B ):

	# (the sample stage: the points of every mesh lit on the full model and kept)
	for m in sorted( meshes, key = lambda m: m[ 'key' ] ):

		if smp_have( m, m[ 'ni' ] // 3 ) is not None and os.path.exists( vis_file( m ) ): continue
		M, rec, tri, flags, ent, kinds, P, lmf, a2 = open_mesh( m )
		S = mesh_samples( m, P, rec, tri, np.nonzero( lmf )[ 0 ], B )
		mesh_vis( m, P, rec, tri, np.nonzero( lmf )[ 0 ], B, S )

def run_fit( B, meshes ):

	# kind "own": the meshes as they came, only measured; kind "fit": fitted (keep / split, refined), written, measured
	t00 = time.time()
	out = []; res = []; sgn = []; hid = np.zeros( ( 4, 4 ) )
	T1 = Tally(); T0 = Tally(); TR = Tally(); TS = Tally(); TN = Tally(); T1b = Tally(); T0b = Tally(); TSb = Tally(); TRb = Tally()
	hb = float( B.get( 'blur', 0.25 ) )
	tot = dict( verts = 0, tris = 0, verts_in = 0, tris_in = 0, places = 0, fit_s = 0.0, sample_s = 0.0, samples = 0, vars = 0, fixed = 0, clamped = 0.0, lit_area = 0.0 )
	faces = [] # per mesh: what is left on each lit face, its area, its kind (for the hybrid)
	for m in meshes:

		t0 = time.time()
		M, rec, tri, flags, ent, kinds, P, lmf, a2 = open_mesh( m )
		f = np.nonzero( lmf )[ 0 ]
		S = mesh_samples( m, P, rec, tri, f, B )
		tot[ 'verts_in' ] += len( rec ); tot[ 'tris_in' ] += len( tri ); tot[ 'samples' ] += len( S[ 'face' ] ); tot[ 'sample_s' ] += sum( S[ 'secs' ] ) * len( S[ 'face' ] ) / max( 1, len( S[ 'face' ] ) + len( S[ 'eface' ] ) )
		kd0 = np.zeros( len( tri ), np.int64 ); kd0[ f ] = face_kind( flags, tri, f, P )
		# (the ground has its light from the bake's own rule, not from the rays of the place found for it - under the setts,
		# where no ray gets out: nothing of the ground is unseen)
		for kq, kf in ( ( 'q', 'face' ), ( 'qr', 'face' ), ( 'qf', 'face' ), ( 'qrf', 'face' ), ( 'eq', 'eface' ), ( 'eqr', 'eface' ) ):

			kg = kd0[ S[ kf ] ] == 0
			S[ kq ] = np.where( kg & ( S[ kq ] > 0.15 ) & ( S[ kq ] < 0.5 ), 1.0, S[ kq ] )

		e0, ok0, g0 = evaluate( rec, tri, S )
		T0.add( e0, ok0, S[ 'ew' ], kd0[ S[ 'eface' ] ], g0, S[ 'ey' ] )
		T0b.patches( blur_error( g0, S[ 'ey' ], S[ 'eX' ], face_normals( m, rec, tri )[ S[ 'eface' ] ], S[ 'ew' ], ok0, kd0[ S[ 'eface' ] ], hb ) )
		# (how far the mesh's light is from the raw level's, where both are known: what changes when the level does)
		st = dict( verts_in = len( rec ), tris_in = len( tri ) )
		t1 = time.time()
		if B.get( 'kind' ) == 'own':

			om = dict( key = m[ 'key' ], level = m[ 'level' ], variant = m[ 'variant' ], raw = m[ 'raw' ], min = m[ 'min' ], q = m[ 'q' ], nv = m[ 'nv' ], ni = m[ 'ni' ], sub = m[ 'sub' ], lm = st )
			nrec2, ntri2 = rec, tri

		else:

			hold = hold_vertices( m, rec, flags, B )
			target = int( B[ 'refine_to' ][ m[ 'key' ] ] ) if B.get( 'refine_to' ) else int( len( tri ) * ( 1 + B.get( 'refine', 0 ) ) )
			rounds = int( B.get( 'rounds', 4 ) ) if target > len( tri ) else 0
			s = float( B.get( 'min_edge', { 'lo': 0.3, 'mid': 0.2, 'near': 0.12 }[ m[ 'level' ] ] ) )
			for rnd in range( rounds + 1 ):

				f = np.nonzero( lmf )[ 0 ]
				F = fit_values( rec, flags, tri, f, P, a2, S, B, hold )
				if rnd == rounds: break
				# the faces the fit cannot carry (a pool of light in the middle of a wall), by the lattice all over them:
				# cut, worst first, as far as this round's share of the triangles allows - and fitted again
				budget = target - len( tri )
				can = ( np.sqrt( a2[ f ] ) >= 2 * s ) & ( F[ 'ferr' ] > float( B.get( 'refine_tol', 3 ) ) )
				cand = np.nonzero( can )[ 0 ]
				# (a cut face makes 4 of 1 and its three neighbours 2 of 1 each: about 6 more triangles)
				cand = cand[ np.argsort( - F[ 'ferr' ][ cand ] ** 2 * a2[ f ][ cand ] ) ][ :max( 0, int( budget / ( 5.0 * ( rounds - rnd ) ** 0.5 ) ) ) ]
				if not len( cand ) or budget < 8: break
				pick = np.zeros( len( tri ), bool ); pick[ f[ cand ] ] = True
				rec, flags, tri, ent, fresh, parent = cut_faces( rec, flags, tri, ent, pick, hold )
				M[ 'rec' ] = rec; P = mesh_pos( M )
				S = move_samples( S, P, tri, parent )
				lmf, a2 = lit_faces( rec, tri, ent, kinds, P )
				hold = hold_vertices( m, rec, flags, B )
				st[ 'cut_%d' % rnd ] = int( len( cand ) )

			cv, x = F[ 'cv' ], F[ 'x' ]
			st.update( F[ 'stats' ] )
			tot[ 'vars' ] += F[ 'stats' ][ 'vars' ]; tot[ 'fixed' ] += F[ 'stats' ][ 'fixed' ]; tot[ 'clamped' ] += F[ 'stats' ][ 'clamped' ] * ( F[ 'stats' ][ 'vars' ] - F[ 'stats' ][ 'fixed' ] )
			# the mesh: a record for every ( variable, material + uv ) its lit corners have, the others as they were
			nf = len( f ); r = tri[ f ].ravel()
			uvg = np.unique( np.ascontiguousarray( rec[ :, 0:6 ] ).view( np.dtype( ( np.void, 12 ) ) ).ravel(), return_inverse = True )[ 1 ].ravel()
			NU = int( uvg.max() ) + 1
			up, fi, inv = np.unique( cv.ravel() * NU + uvg[ r ], return_index = True, return_inverse = True )
			var = up // NU
			nrec = rec[ r[ fi ] ].copy()
			nrec[ :, 6:9 ] = enc16( x[ var, :3 ] )
			nrec[ :, 9 ] = np.round( np.clip( x[ var, 3 ], 0, 1 ) * 32767 )
			nfl = np.zeros( len( up ), np.uint8 ); np.bitwise_or.at( nfl, inv.ravel(), flags[ r ] )
			om, nrec2, ntri2 = emit_mesh( m, B[ 'name' ], rec, flags, tri, f, ent, kinds, nrec, nfl, inv.reshape( nf, 3 ), st )

		tot[ 'fit_s' ] += time.time() - t1
		kd = np.zeros( len( tri ), np.int64 ); kd[ f ] = face_kind( flags, tri, f, P )
		e1, ok1, g1 = evaluate( nrec2, ntri2, S )
		ke = kd[ S[ 'eface' ] ]; ne = face_normals( m, nrec2, ntri2 )[ S[ 'eface' ] ]
		T1.add( e1, ok1, S[ 'ew' ], ke, g1, S[ 'ey' ] )
		# (the points that do not count, by kind: inside something on the full model / not found on it)
		for i, v in enumerate( ( np.ones( len( ke ) ), ( S[ 'eq' ] > 0.07 ) & ( S[ 'eq' ] < 0.15 ), S[ 'eq' ] < 0.07, ( S[ 'eq' ] > 0.15 ) & ( S[ 'eq' ] < 0.5 ) ) ): hid[ :, i ] += np.bincount( ke, S[ 'ew' ] * v, 4 )
		T1b.patches( blur_error( g1, S[ 'ey' ], S[ 'eX' ], ne, S[ 'ew' ], ok1, ke, hb ) )
		# (the reference against the bake's own few rays at the same points: the noise a point-sampled vertex carries)
		TN.add( np.abs( tone( S[ 'eyn' ][ :, :3 ] ) - tone( S[ 'ey' ][ :, :3 ] ) ).max( 1 ) * 255, ok1, S[ 'ew' ], ke )
		# the same mesh against the raw level's light: what the eye meets when the cell changes level (the top level IS that light)
		er, okr, _ = evaluate( nrec2, ntri2, S, 'eyr' ); TR.add( er, okr, S[ 'ew' ], ke, g1, S[ 'eyr' ] )
		TRb.patches( blur_error( g1, S[ 'eyr' ], S[ 'eX' ], ne, S[ 'ew' ], okr, ke, hb ) )
		# ... and the raw level's light itself against the full model's (the floor of anything fitted to it)
		es = np.abs( tone( S[ 'eyr' ][ :, :3 ] ) - tone( S[ 'ey' ][ :, :3 ] ) ).max( 1 ) * 255; TS.add( es, ok1 & okr, S[ 'ew' ], ke, S[ 'eyr' ], S[ 'ey' ] )
		TSb.patches( blur_error( S[ 'eyr' ], S[ 'ey' ], S[ 'eX' ], ne, S[ 'ew' ], ok1 & okr, ke, hb ) )
		fe = face_error( e1, ok1, S, len( tri ) )
		faces.append( dict( err = fe[ f ].astype( np.float32 ), area = ( a2[ f ] * 0.5 ).astype( np.float32 ), kind = kd[ f ].astype( np.int8 ) ) )
		st.update( verts = om[ 'nv' ], tris = om[ 'ni' ] // 3, lit_faces = int( len( f ) ), places = int( len( np.unique( pkey( nrec2 ) ) ) ), seconds = round( time.time() - t0, 1 ),
			own = wstats( e0[ ok0 ], S[ 'ew' ][ ok0 ] ), fit = wstats( e1[ ok1 ], S[ 'ew' ][ ok1 ] ) )
		tot[ 'verts' ] += om[ 'nv' ]; tot[ 'tris' ] += om[ 'ni' ] // 3; tot[ 'places' ] += st[ 'places' ]; tot[ 'lit_area' ] += float( a2[ f ].sum() * 0.5 )
		out.append( om )
		if B.get( 'res', 1 ):

			rfl = flags if B.get( 'kind' ) == 'own' else np.zeros( len( nrec2 ), np.uint8 )
			res.append( res_mesh( m, B[ 'name' ] + '_res', nrec2, rfl, ntri2, f, ent, kinds, heat( fe[ f ] ) ) )
			sgn.append( res_mesh( m, B[ 'name' ] + '_sgn', nrec2, rfl, ntri2, f, ent, kinds, heat_signed( face_signed( g1, ok1, S, len( tri ) )[ f ] ) ) )

		llog( '  %s %s %s @ %s: %d tris %d verts; as it came rms %.2f p95 %.2f -> rms %.2f p95 %.2f; %s' % ( m[ 'key' ], m[ 'level' ], m[ 'variant' ], B[ 'name' ], st[ 'tris' ], st[ 'verts' ],
			st[ 'own' ][ 'rms' ], st[ 'own' ][ 'p95' ], st[ 'fit' ][ 'rms' ], st[ 'fit' ][ 'p95' ], json.dumps( { k: v for k, v in st.items() if k not in ( 'own', 'fit' ) } ) ) )

	tot[ 'clamped' ] = round( tot[ 'clamped' ] / max( 1, tot[ 'vars' ] - tot[ 'fixed' ] ), 4 )
	tot = { k: ( round( v, 1 ) if isinstance( v, float ) and k != 'clamped' else v ) for k, v in tot.items() }
	stats = dict( name = B[ 'name' ], level = B[ 'level' ], variant = B[ 'variant' ], settings = { k: v for k, v in B.items() if k not in ( 'refine_to', ) }, seconds = round( time.time() - t00, 1 ), **tot,
		fit = T1.stats(), own = T0.stats(), vs_raw = TR.stats(), raw_vs_full = TS.stats(), noise = TN.stats(),
		fit_blur = T1b.stats(), own_blur = T0b.stats(), vs_raw_blur = TRb.stats(), raw_vs_full_blur = TSb.stats(),
		hidden = { name: dict( inside = round( float( r[ 1 ] / max( r[ 0 ], 1e-30 ) ), 4 ), not_found = round( float( r[ 2 ] / max( r[ 0 ], 1e-30 ) ), 4 ), unseen = round( float( r[ 3 ] / max( r[ 0 ], 1e-30 ) ), 4 ) ) for name, r in [ ( 'all', hid.sum( 0 ) ) ] + [ ( n, hid[ i ] ) for i, n in enumerate( KINDS ) ] } )
	np.savez( os.path.join( LRAW, 'faces.%s.npz' % B[ 'name' ] ), **{ '%s_%d' % ( k, i ): v for i, d in enumerate( faces ) for k, v in d.items() } )
	json.dump( stats, open( os.path.join( LRAW, 'stats.%s.json' % B[ 'name' ] ), 'w' ), indent = 1 )
	for name, ms in ( ( B[ 'name' ], out ), ( B[ 'name' ] + '_res', res ), ( B[ 'name' ] + '_sgn', sgn ) ):

		if ms: json.dump( dict( name = name, level = B[ 'level' ], kind = 'fit', meshes = ms, stats = { k: v for k, v in stats.items() if k in ( 'fit', 'own', 'verts', 'tris' ) } if name == B[ 'name' ] else {} ), open( os.path.join( LRAW, name + '.meta.json' ), 'w' ) )

	a, o = stats[ 'fit' ][ 'all' ], stats[ 'own' ][ 'all' ]
	ab, ob = stats[ 'fit_blur' ][ 'all' ], stats[ 'own_blur' ][ 'all' ]
	llog( '%s: %d meshes, %d tris, %d verts, %.1fs; as it came: mean %.2f rms %.2f p95 %.2f; this: mean %.2f rms %.2f p95 %.2f; over %d cm patches: rms %.2f p95 %.2f -> rms %.2f p95 %.2f; by kind (rms as it came -> this): %s' % ( B[ 'name' ], len( out ), tot[ 'tris' ], tot[ 'verts' ], time.time() - t00,
		o[ 'mean' ], o[ 'rms' ], o[ 'p95' ], a[ 'mean' ], a[ 'rms' ], a[ 'p95' ], round( hb * 100 ), ob[ 'rms' ], ob[ 'p95' ], ab[ 'rms' ], ab[ 'p95' ], ', '.join( '%s %.2f -> %.2f' % ( k, stats[ 'own' ][ k ][ 'rms' ], stats[ 'fit' ][ k ][ 'rms' ] ) for k in KINDS if stats[ 'fit' ][ k ] ) ) )
	return stats

# ------------------------------------------------------------------------------------------ the light in pages, measured

def eval_lightmap( B, meshes ):

	# a bundle whose light is in texture pages (bake_bundle() + encode()) judged as the fitted ones are: the page's
	# bytes read as the shader reads them (the four texels about the point mixed, squared, times the range) at the same
	# points on the same faces. A face that kept its vertex light is judged by that.
	meta = json.load( open( os.path.join( LRAW, B[ 'name' ] + '.meta.json' ) ) )
	R = float( B.get( 'range', 8 ) )
	pages = {}
	def page( i ):

		if i not in pages: pages[ i ] = np.frombuffer( open( os.path.join( LDIR, B[ 'name' ], 'p%d.bin' % i ), 'rb' ).read(), np.uint8 ).reshape( PAGE, PAGE, 4 )
		return pages[ i ]

	T1 = Tally(); T0 = Tally(); T1b = Tally(); T0b = Tally(); hb = float( B.get( 'blur', 0.25 ) ); res = []
	tot = dict( verts = 0, tris = 0, texels = 0, paged_area = 0.0, lit_area = 0.0 ); faces = []
	for m in meshes:

		om = next( ( o for o in meta[ 'meshes' ] if o[ 'key' ] == m[ 'key' ] ), None )
		if om is None: continue
		M, rec, tri, flags, ent, kinds, P, lmf, a2 = open_mesh( m )
		f = np.nonzero( lmf )[ 0 ]
		S = mesh_samples( m, P, rec, tri, f, B )
		kd = np.zeros( len( tri ), np.int64 ); kd[ f ] = face_kind( flags, tri, f, P )
		e0, ok0, g0 = evaluate( rec, tri, S ); T0.add( e0, ok0, S[ 'ew' ], kd[ S[ 'eface' ] ] )
		ne = face_normals( m, rec, tri )[ S[ 'eface' ] ]
		T0b.patches( blur_error( g0, S[ 'ey' ], S[ 'eX' ], ne, S[ 'ew' ], ok0, kd[ S[ 'eface' ] ], hb ) )
		L = load_mesh( om ); lrec, ltri = L[ 'rec' ], L[ 'tri' ]
		assert len( ltri ) == len( tri )
		tv = ltri[ S[ 'eface' ] ]; b = S[ 'eb' ]
		paged = ( ( lrec[ :, 3 ] & 0x8000 ) > 0 )[ tv ].all( 1 )
		lit = lin13( lrec[ :, 6:9 ] )
		g = lit[ tv[ :, 0 ] ] * b[ :, 0:1 ] + lit[ tv[ :, 1 ] ] * b[ :, 1:2 ] + lit[ tv[ :, 2 ] ] * b[ :, 2:3 ]
		uv = lrec[ :, 6:8 ].astype( np.float64 ) / 65535 * PAGE - 0.5
		u = uv[ tv[ :, 0 ] ] * b[ :, 0:1 ] + uv[ tv[ :, 1 ] ] * b[ :, 1:2 ] + uv[ tv[ :, 2 ] ] * b[ :, 2:3 ]
		pg = ( lrec[ tv[ :, 0 ], 8 ] // 256 ).astype( np.int64 )
		for i in np.unique( pg[ paged ] ).tolist():

			w = np.nonzero( paged & ( pg == i ) )[ 0 ]; img = page( i )
			x0 = np.floor( u[ w, 0 ] ).astype( np.int64 ); y0 = np.floor( u[ w, 1 ] ).astype( np.int64 ); fx = ( u[ w, 0 ] - x0 )[ :, None ]; fy = ( u[ w, 1 ] - y0 )[ :, None ]
			cl = lambda v: np.clip( v, 0, PAGE - 1 )
			t = lambda xx, yy: img[ cl( yy ), cl( xx ), :3 ].astype( np.float64 ) / 255
			v = ( t( x0, y0 ) * ( 1 - fx ) + t( x0 + 1, y0 ) * fx ) * ( 1 - fy ) + ( t( x0, y0 + 1 ) * ( 1 - fx ) + t( x0 + 1, y0 + 1 ) * fx ) * fy
			g[ w ] = v * v * R

		e1 = np.abs( tone( g ) - tone( S[ 'ey' ][ :, :3 ] ) ).max( 1 ) * 255; ok1 = S[ 'eq' ] > 0.5
		T1.add( e1, ok1, S[ 'ew' ], kd[ S[ 'eface' ] ], g, S[ 'ey' ] )
		T1b.patches( blur_error( g, S[ 'ey' ], S[ 'eX' ], ne, S[ 'ew' ], ok1, kd[ S[ 'eface' ] ], hb ) )
		fe = face_error( e1, ok1, S, len( tri ) )
		faces.append( dict( err = fe[ f ].astype( np.float32 ), area = ( a2[ f ] * 0.5 ).astype( np.float32 ), kind = kd[ f ].astype( np.int8 ) ) )
		res.append( res_mesh( m, B[ 'name' ] + '_res', rec, flags, tri, f, ent, kinds, heat( fe[ f ] ) ) )
		tot[ 'verts' ] += om[ 'nv' ]; tot[ 'tris' ] += om[ 'ni' ] // 3; tot[ 'texels' ] += om[ 'lm' ][ 'texels' ]; tot[ 'paged_area' ] += float( S[ 'ew' ][ paged ].sum() ); tot[ 'lit_area' ] += float( S[ 'ew' ].sum() )

	stats = dict( name = B[ 'name' ], level = B[ 'level' ], variant = B[ 'variant' ], settings = B, fit_s = 0, pages = int( meta.get( 'npages', 0 ) ), bake_s = meta[ 'stats' ][ 'seconds' ], **tot,
		fit = T1.stats(), own = T0.stats(), fit_blur = T1b.stats(), own_blur = T0b.stats(), vs_raw = dict( all = None ), raw_vs_full = dict( all = None ) )
	np.savez( os.path.join( LRAW, 'faces.%s.npz' % B[ 'name' ] ), **{ '%s_%d' % ( k, i ): v for i, d in enumerate( faces ) for k, v in d.items() } )
	json.dump( stats, open( os.path.join( LRAW, 'stats.%s.json' % B[ 'name' ] ), 'w' ), indent = 1 )
	if res: json.dump( dict( name = B[ 'name' ] + '_res', level = B[ 'level' ], kind = 'fit', meshes = res, stats = {} ), open( os.path.join( LRAW, B[ 'name' ] + '_res.meta.json' ), 'w' ) )
	a = stats[ 'fit' ][ 'all' ]
	llog( '%s (pages): %d tris, %d verts, %d texels; mean %.2f rms %.2f p95 %.2f' % ( B[ 'name' ], tot[ 'tris' ], tot[ 'verts' ], tot[ 'texels' ], a[ 'mean' ], a[ 'rms' ], a[ 'p95' ] ) )
	return stats
