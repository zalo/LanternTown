#!/usr/bin/env python3
# The light of the reduced levels, taken from the top level of the same bake - a lookup, no ray. Runs between the
# raw bake (webbake_bl.py) and the packer (webbake_pack.mjs): one call in webbake.py's bake_one().
#
# Why: the bake lights every level on its own geometry. The lower levels are other meshes (LOD1, LOD2), so their
# light differs from the top level's by more than their shape does - a wall of the mid level is darker by a tenth of
# its light than the same wall at the top level, and the picture jumps when a cell changes level. Here every vertex
# of a lower level asks the TOP level of its own cell (fine; near in a bake without fine) what light it has at that
# place, and takes it. The packer then simplifies as before (its lowest level is made of mid: it follows).
#
#   corner    every lit corner asks a little inside its face (which side of an edge it means): the nearest face of
#             the top level with the same material that looks the same way, within `refit_reach`
#   vertex    the corners of one place, material and direction share the mean of what they found. Where they carry
#             DIFFERENT colours of their own (a tint painted into the vertices: the stripes of an awning), each
#             keeps its share of their mean - the level's own tint with the top level's light
#   held      as the bake left them, byte for byte: the ground (one ground at every level - the packer copies the
#             top level's), every vertex at a place a neighbouring cell also has one at (no seam), what glows, glass
#             and every vertex with no face of the top level in reach
#
# In:  <out>/.raw/raw.json + the levels' files. Out: <cell>.<level>.refit.bin beside them (the same bytes but for
# the colour slots of the vertices that moved) and raw.json pointing at them (`raw`; the bake's own file stays, as
# `raw0`). Run again it starts from `raw0`: the result does not depend on how often it ran. `--set refit=0` (a bake
# setting of webbake.py) puts the bake's own files back.
#
# numpy alone, no Blender process: this python if it has numpy, else the python Blender ships, run by itself.
#
#   python3 tools/next/webbake.py ...                           (calls run())
#   <python with numpy> tools/next/lsvb_refit.py -- <out>/.raw ['{"refit_jobs":8}']
#
# For the trials of tools/next/lightmap.py (never in a bake): "dest" writes the files there as the variant
# rawrf<tag> of its dump.json and leaves the bake alone; "top" names the level to take the light from; "cells";
# "judge": [ variants ] measures meshes of that dump against the top level instead (display steps, no ray).

import sys, os, json, time, glob, subprocess

OPT = dict(
	refit = 1, # 0: the levels as the bake lit them
	refit_levels = 'mid,near', # the raw levels that take the top level's light
	refit_reach = dict( mid = 0.35, near = 0.25 ), # how far from a corner the top level's surface may be (m)
	refit_inset = 0.02, # a corner asks this far inside its face (m; a tenth of the way to the middle at most)
	refit_turn = 0.96, # corners of one place whose faces look the same way within this cosine are one vertex
	refit_tint = 1, # 1: corners of one vertex with colours of their own keep their share of them
	refit_jobs = 0 ) # cells at once (0: the bake's --jobs)
K_OPAQUE = 0
TOPBOX = 2.0 # the largest boxes of the search (m)
BLOCK = 200000 # pairs of a point and a face at once (what stays in the processor's cache: five times faster than 2 M)

def python( blender ):

	try:

		import numpy
		return [ sys.executable ]

	except ImportError: pass
	py = sorted( p for p in glob.glob( os.path.join( os.path.dirname( os.path.realpath( blender ) ), '*/python/bin/python3*' ) ) if not p.endswith( 'config' ) ) if blender else []
	if py: return [ py[ 0 ] ]
	if not blender: sys.exit( 'lsvb_refit: numpy is needed (this python has none and no Blender was named)' )
	return [ blender, '--background', '--factory-startup', '--python' ]

def run( out, bake = None, jobs = 4, blender = None, env = None ):

	# webbake.py's bake_one(), after the bake and before the packer -> whether the raw bake is as `refit` says
	opt = { k: v for k, v in ( bake or {} ).items() if k in OPT }
	if not opt.get( 'refit_jobs' ): opt[ 'refit_jobs' ] = jobs
	r = subprocess.run( python( blender ) + [ os.path.abspath( __file__ ), '--', os.path.join( out, '.raw' ), json.dumps( opt ) ], env = env )
	return r.returncode == 0

# ------------------------------------------------------------------------------------------ what follows needs numpy

T0 = time.time()
def log( *a ): print( '[refit %6.1fs]' % ( time.time() - T0 ), *a, flush = True )

def read( raw, m, own = False ):

	# a level as the BAKE wrote it (`raw0` once this ran) -> its bytes, records (u16 view of them), triangles, flags
	w = m.get( 'w', 10 ); nv, ni = m[ 'nv' ], m[ 'ni' ]
	buf = np.fromfile( os.path.join( raw, m[ 'raw' ] if own else m.get( 'raw0', m[ 'raw' ] ) ), np.uint8 )
	if len( buf ) < nv * w * 2 + ni * 4 + nv: sys.exit( 'lsvb_refit: %s is shorter than raw.json says' % m[ 'raw' ] )
	rec = buf[ :nv * w * 2 ].view( '<u2' ).reshape( nv, w )
	tri = buf[ nv * w * 2:nv * w * 2 + ni * 4 ].view( '<u4' ).reshape( - 1, 3 ).astype( np.int64 )
	return buf, rec, tri, buf[ nv * w * 2 + ni * 4:nv * w * 2 + ni * 4 + nv ]

def pos( m, rec ): return np.array( m[ 'min' ], np.float64 )[ None ] + rec[ :, :3 ].astype( np.float64 ) * m[ 'q' ]
def lin( c16 ): return ( c16.astype( np.float64 ) / 65535 ) ** 2 * 64
def enc16( c ): return np.round( np.sqrt( np.clip( c, 0, 64 ) / 64 ) * 65535 )

def lit( m, rec, tri, P, eps ):

	# the faces whose light is baked: opaque, nothing of them glowing, not a sliver -> which, the faces' normals
	opq = np.zeros( len( tri ), bool )
	for kind, start, count in m[ 'sub' ]:

		if kind == K_OPAQUE: opq[ start // 3:( start + count ) // 3 ] = True

	n = np.cross( P[ tri[ :, 1 ] ] - P[ tri[ :, 0 ] ], P[ tri[ :, 2 ] ] - P[ tri[ :, 0 ] ] ); a2 = np.linalg.norm( n, axis = 1 )
	return opq & ~ ( rec[ :, 9 ] == 65535 )[ tri ].any( 1 ) & ( a2 > eps ), n / np.maximum( a2, 1e-20 )[ :, None ], a2

def wkeys( m, rec ):

	# a place in the world as one number, whatever cell's grid it is on
	w = np.round( ( np.array( m[ 'min' ], np.float64 )[ None ] + rec[ :, :3 ].astype( np.float64 ) * m[ 'q' ] ) * 1024 ).astype( np.int64 ) + ( 1 << 20 )
	return ( w[ :, 0 ] << 42 ) | ( w[ :, 1 ] << 21 ) | w[ :, 2 ]

def shared( raw, chunks, key, lv, rec ):

	# the vertices at a place where another cell has one too, at this level (walls run over cell borders: both cells
	# carry the vertex with the same light, and a cell that changed it alone would open a seam)
	m = chunks[ key ][ lv ]; out = np.zeros( len( rec ), bool ); mine = None
	for k, c in chunks.items():

		o = c.get( lv )
		if k == key or not o or any( o[ 'min' ][ i ] > m[ 'max' ][ i ] + 0.01 or o[ 'max' ][ i ] < m[ 'min' ][ i ] - 0.01 for i in range( 3 ) ): continue
		if mine is None: mine = wkeys( m, rec )
		r = np.memmap( os.path.join( raw, o.get( 'raw0', o[ 'raw' ] ) ), '<u2', 'r', shape = ( o[ 'nv' ], o.get( 'w', 10 ) ) )
		out |= np.isin( mine, np.unique( wkeys( o, np.asarray( r[ :, :3 ] ) ) ) )

	return out

def closest( p, a, b, c ):

	# the point of the triangle a b c nearest to p (Ericson, Real-Time Collision Detection 5.1.5) -> how far, its
	# weights of b and c
	ab = b - a; ac = c - a; ap = p - a
	d1 = ( ab * ap ).sum( 1 ); d2 = ( ac * ap ).sum( 1 )
	bp = p - b; d3 = ( ab * bp ).sum( 1 ); d4 = ( ac * bp ).sum( 1 )
	cp = p - c; d5 = ( ab * cp ).sum( 1 ); d6 = ( ac * cp ).sum( 1 )
	va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
	with np.errstate( divide = 'ignore', invalid = 'ignore' ):

		den = va + vb + vc; v = vb / den; w = vc / den # inside
		k = ( va <= 0 ) & ( d4 - d3 >= 0 ) & ( d5 - d6 >= 0 ); t = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) ); v = np.where( k, 1 - t, v ); w = np.where( k, t, w ) # edge bc
		k = ( vb <= 0 ) & ( d2 >= 0 ) & ( d6 <= 0 ); t = d2 / ( d2 - d6 ); v = np.where( k, 0, v ); w = np.where( k, t, w ) # edge ac
		k = ( d6 >= 0 ) & ( d5 <= d6 ); v = np.where( k, 0, v ); w = np.where( k, 1, w ) # c
		k = ( vc <= 0 ) & ( d1 >= 0 ) & ( d3 <= 0 ); t = d1 / ( d1 - d3 ); v = np.where( k, t, v ); w = np.where( k, 0, w ) # edge ab
		k = ( d3 >= 0 ) & ( d4 <= d3 ); v = np.where( k, 1, v ); w = np.where( k, 0, w ) # b
		k = ( d1 <= 0 ) & ( d2 <= 0 ); v = np.where( k, 0, v ); w = np.where( k, 0, w ) # a

	v = np.nan_to_num( v ); w = np.nan_to_num( w )
	d = ap - ab * v[ :, None ] - ac * w[ :, None ]
	return np.sqrt( ( d * d ).sum( 1 ) ), v, w

class Top:

	# the lit faces of a cell's top level, to be asked for the light at a point

	def __init__( self, m, rec, tri ):

		P = pos( m, rec )
		ok, n, a2 = lit( m, rec, tri, P, 1e-9 )
		T = tri[ ok ]; n = n[ ok ]
		self.T = T; self.N = n; self.col = lin( rec[ :, 6:9 ] ); self.lay = rec[ T[ :, 0 ], 3 ].astype( np.int64 )
		self.A = P[ T[ :, 0 ] ]; self.B = P[ T[ :, 1 ] ]; self.C = P[ T[ :, 2 ] ]
		self.org = ( P.min( 0 ) if len( P ) else np.zeros( 3 ) ) - 8.0
		# what tells whether a face comes near a box, one row a face: its bounds, then four planes ( unit normal, offset,
		# the normal's |x| + |y| + |z| ) - its own and, outward, the three through its edges
		a = self.A - self.org; b = self.B - self.org; c = self.C - self.org
		lo = np.minimum( np.minimum( a, b ), c ); hi = np.maximum( np.maximum( a, b ), c )
		self.FB = np.concatenate( [ ( lo + hi ) / 2, ( hi - lo ) / 2 ], 1 ).astype( np.float32 ); self.lo = lo; self.hi = hi
		F = np.zeros( ( len( T ), 4, 5 ), np.float32 )
		for k, ( u, o ) in enumerate( ( ( n, a ), ( np.cross( b - a, n ), a ), ( np.cross( c - b, n ), b ), ( np.cross( a - c, n ), c ) ) ):

			u = u / np.maximum( np.linalg.norm( u, axis = 1, keepdims = True ), 1e-20 )
			F[ :, k, 0:3 ] = u; F[ :, k, 3 ] = ( u * o ).sum( 1 ); F[ :, k, 4 ] = np.abs( u ).sum( 1 )

		self.F = F; self.n32 = n.astype( np.float32 ); self.pairs = 0; self.tests = 0

	def reach( self, f, cell, h, g ):

		# whether face f may come within g of the box of side h whose low corner is `cell` (never no when it does)
		e = np.float32( g + 1e-4 ); h2 = np.float32( h / 2 ); mid = cell + h2
		B = self.FB[ f ]
		ok = ( np.abs( mid - B[ :, 0:3 ] ) <= B[ :, 3:6 ] + ( e + h2 ) ).all( 1 )
		i = np.nonzero( ok )[ 0 ]; F = self.F[ f[ i ] ]; mid = mid[ i ]
		d = np.einsum( 'ikj,ij->ik', F[ :, :, 0:3 ], mid ) - F[ :, :, 3 ]; d[ :, 0 ] = np.abs( d[ :, 0 ] )
		ok[ i ] = ( d <= e + h2 * F[ :, :, 4 ] ).all( 1 )
		self.tests += len( f )
		return ok

	def near( self, X, LAY, g, h ):

		# the faces that may be within g of each point, as ( points, faces ) a block at a time. The points' boxes of
		# side h, and the boxes of twice, four times .. that side they are in: a face is given to the largest boxes its
		# bounds touch and goes on only into the halves it still reaches
		nq = len( X ); L = max( 0, int( np.ceil( np.log2( TOPBOX / h ) ) ) )
		c = np.clip( np.floor( ( X - self.org ) / h ), 0, 65535 ).astype( np.int64 )
		key = lambda c, lay: ( lay << 51 ) | ( c[ :, 0 ] << 34 ) | ( c[ :, 1 ] << 17 ) | c[ :, 2 ]
		xyz = lambda k: np.stack( [ ( k >> 34 ) & 0x1ffff, ( k >> 17 ) & 0x1ffff, k & 0x1ffff ], 1 )
		cells = [ np.unique( key( c >> ( L - l ), LAY ) ) for l in range( L + 1 ) ]
		def under( k, sub ):

			# the members `sub` of every box k (sub's numbers, in the order of k's): where each k's start, how many
			o = np.argsort( sub, kind = 'stable' ); n = np.bincount( sub, minlength = len( k ) )
			return o, np.cumsum( n ) - n, n

		kids = [ under( cells[ l ], np.searchsorted( cells[ l ], key( xyz( cells[ l + 1 ] ) >> 1, cells[ l + 1 ] >> 51 ) ) ) for l in range( L ) ]
		pts = under( cells[ L ], np.searchsorted( cells[ L ], key( c, LAY ) ) )
		def spread( ci, f, sub ):

			o, start, n = sub; cnt = n[ ci ]; tot = int( cnt.sum() )
			return o[ np.repeat( start[ ci ] - ( np.cumsum( cnt ) - cnt ), cnt ) + np.arange( tot ) ], np.repeat( f, cnt )

		# the largest boxes: every face of a material that is asked for, into those its bounds ( + g ) touch
		H = h * 2.0 ** L; c0 = xyz( cells[ 0 ] ); lo0 = c0.min( 0 ); hi0 = c0.max( 0 )
		f = np.nonzero( np.isin( self.lay, np.unique( LAY ) ) )[ 0 ]
		i0 = np.maximum( np.floor( ( self.lo[ f ] - g - 1e-4 ) / H ).astype( np.int64 ), lo0 ); i1 = np.minimum( np.floor( ( self.hi[ f ] + g + 1e-4 ) / H ).astype( np.int64 ), hi0 )
		n3 = np.maximum( i1 - i0 + 1, 0 ); cnt = n3.prod( 1 ); tot = int( cnt.sum() )
		r = np.arange( tot ) - np.repeat( np.cumsum( cnt ) - cnt, cnt ); ff = np.repeat( f, cnt ); n3 = np.repeat( n3, cnt, 0 ); i0 = np.repeat( i0, cnt, 0 )
		k = key( np.stack( [ i0[ :, 0 ] + r // ( n3[ :, 1 ] * n3[ :, 2 ] ), i0[ :, 1 ] + ( r // n3[ :, 2 ] ) % n3[ :, 1 ], i0[ :, 2 ] + r % n3[ :, 2 ] ], 1 ), self.lay[ ff ] ) if tot else np.zeros( 0, np.int64 )
		ci = np.minimum( np.searchsorted( cells[ 0 ], k ), len( cells[ 0 ] ) - 1 ); ok = cells[ 0 ][ ci ] == k if len( k ) else np.zeros( 0, bool )
		ci = ci[ ok ]; ff = ff[ ok ]
		o = np.argsort( ci, kind = 'stable' ); ci = ci[ o ]; ff = ff[ o ]
		def down( l, ci, ff ):

			# ( boxes of level l, faces ) -> ( points, faces ), never more than a block at once
			hl = h * 2.0 ** ( L - l )
			for s in range( 0, len( ci ), BLOCK ):

				a = ci[ s:s + BLOCK ]; b = ff[ s:s + BLOCK ]
				ok = self.reach( b, ( xyz( cells[ l ][ a ] ) * hl ).astype( np.float32 ), hl, g )
				a = a[ ok ]; b = b[ ok ]
				if not len( a ): continue
				sub = kids[ l ] if l < L else pts
				# (as many at a time as spread to a block)
				cum = np.cumsum( sub[ 2 ][ a ] ); e0 = 0
				while e0 < len( a ):

					e1 = max( e0 + 1, int( np.searchsorted( cum, ( cum[ e0 - 1 ] if e0 else 0 ) + BLOCK, 'right' ) ) )
					x, y = spread( a[ e0:e1 ], b[ e0:e1 ], sub ); e0 = e1
					if l < L: yield from down( l + 1, x, y )
					else: yield x, y

		yield from down( 0, ci, ff )

	def stage( self, X, N, LAY, g, R, h = None ):

		# the best face within g of each point: same material, looking the same way; least of distance + what it is
		# turned by ( a quarter of R at a right angle ), of equals the first -> face ( - 1 none ), weights, that measure
		nq = len( X ); ot = np.full( nq, - 1, np.int64 ); ob = np.zeros( ( nq, 3 ) ); om = np.full( nq, 1e9 )
		N32 = N.astype( np.float32 )
		for q, t in self.near( X, LAY, g, h or 2 * g ):

			self.pairs += len( q )
			keep = ( self.n32[ t ] * N32[ q ] ).sum( 1 ) > 0
			q = q[ keep ]; t = t[ keep ]
			if not len( q ): continue
			d, v, w = closest( X[ q ], self.A[ t ], self.B[ t ], self.C[ t ] )
			ok = d <= g
			q = q[ ok ]; t = t[ ok ]; v = v[ ok ]; w = w[ ok ]
			if not len( q ): continue
			m = d[ ok ] + 0.25 * R * ( 1 - ( self.N[ t ] * N[ q ] ).sum( 1 ) )
			o = np.lexsort( ( t, m, q ) ); first = np.ones( len( o ), bool ); first[ 1: ] = q[ o ][ 1: ] != q[ o ][ :- 1 ]
			o = o[ first ]; q = q[ o ]; better = ( m[ o ] < om[ q ] ) | ( ( m[ o ] == om[ q ] ) & ( t[ o ] < ot[ q ] ) )
			o = o[ better ]; q = q[ better ]
			ot[ q ] = t[ o ]; om[ q ] = m[ o ]; ob[ q ] = np.stack( [ 1 - v[ o ] - w[ o ], v[ o ], w[ o ] ], 1 )

		return ot, ob, om

	def find( self, X, N, LAY, R ):

		# near first: a point whose best face within g measures g or less cannot do better farther out
		nq = len( X ); bt = np.full( nq, - 1, np.int64 ); bb = np.zeros( ( nq, 3 ) )
		left = np.arange( nq )
		for g in [ r for r in ( 0.02, 0.06, 0.18 ) if r < R ] + [ R ]:

			if not len( left ): break
			t, b, m = self.stage( X[ left ], N[ left ], LAY[ left ], g, R )
			done = ( t >= 0 ) & ( ( m <= g ) | ( g >= R ) )
			bt[ left[ done ] ] = t[ done ]; bb[ left[ done ] ] = b[ done ]
			left = left[ ~ done ]

		return bt, bb

	def light( self, X, N, LAY, R ):

		# -> the light at the place found for each point (its face's vertex light, as the page interpolates it), found
		t, b = self.find( X, N, LAY, R )
		hit = t >= 0; tv = self.T[ t[ hit ] ]; b = b[ hit ]
		y = np.zeros( ( len( X ), 3 ) )
		y[ hit ] = self.col[ tv[ :, 0 ] ] * b[ :, 0:1 ] + self.col[ tv[ :, 1 ] ] * b[ :, 1:2 ] + self.col[ tv[ :, 2 ] ] * b[ :, 2:3 ]
		return y, hit

def clusters( gid, n, turn ):

	# the corners of one group (a place and a material) whose faces look one way: the first that is left leads, who
	# is within `turn` of it goes with it, the rest start again -> a number for every corner
	o = np.argsort( gid, kind = 'stable' ); g = gid[ o ]; nn = n[ o ]
	out = np.full( len( o ), - 1, np.int64 ); nxt = 0
	left = np.arange( len( o ) )
	while len( left ):

		gl = g[ left ]
		first = np.ones( len( left ), bool ); first[ 1: ] = gl[ 1: ] != gl[ :- 1 ]
		lead = left[ np.maximum.accumulate( np.where( first, np.arange( len( left ) ), 0 ) ) ]
		take = ( nn[ left ] * nn[ lead ] ).sum( 1 ) >= turn
		take |= first
		out[ left[ take ] ] = ( np.cumsum( first ) - 1 + nxt )[ take ]
		nxt += int( first.sum() )
		left = left[ ~ take ]

	res = np.empty( len( o ), np.int64 ); res[ o ] = out
	return res

def tone( c ):

	# about what the page shows of a lit colour on a mid-grey surface (exposure 1.25, ACES, display gamma): 0 .. 1
	x = np.maximum( c, 0 ) * ( 1.25 * 0.5 )
	return np.clip( x * ( 2.51 * x + 0.03 ) / ( x * ( 2.43 * x + 0.59 ) + 0.14 ), 0, 1 ) ** ( 1 / 2.2 )

def refit_level( raw, chunks, key, lv, top, opt, dest ):

	t0 = time.time()
	m = chunks[ key ][ lv ]
	buf, rec, tri, flags = read( raw, m ); P = pos( m, rec )
	ok, fn, a2 = lit( m, rec, tri, P, 1e-7 )
	tf = tri[ ok ]; r = tf.ravel()
	hold = ( ( flags & 5 ) > 0 ) | ( rec[ :, 9 ] == 65535 ) | shared( raw, chunks, key, lv, rec )
	for kind, start, count in m[ 'sub' ]:

		if kind != K_OPAQUE: hold[ tri[ start // 3:( start + count ) // 3 ].ravel() ] = True # (glass, water, cut-outs)

	ask = ~ hold[ r ]
	Pf = P[ tf ]; cen = Pf.mean( 1, keepdims = True ); d = np.linalg.norm( cen - Pf, axis = 2, keepdims = True )
	X = ( Pf + ( cen - Pf ) * np.minimum( 0.1, opt[ 'refit_inset' ] / np.maximum( d, 1e-9 ) ) ).reshape( - 1, 3 )[ ask ]
	N = np.repeat( fn[ ok ], 3, 0 )[ ask ]; rc = r[ ask ]; LAY = rec[ rc, 3 ].astype( np.int64 )
	reach = opt[ 'refit_reach' ]; reach = float( reach.get( lv, 0.35 ) if isinstance( reach, dict ) else reach )
	y, hit = top.light( X, N, LAY, reach )
	st = dict( verts = len( rec ), held = int( hold.sum() ), corners = int( len( rc ) ), found = int( hit.sum() ) )
	y = y[ hit ]; rc = rc[ hit ]; N = N[ hit ]; LAY = LAY[ hit ]
	own = lin( rec[ rc, 6:9 ] )
	place = rec[ rc, 0 ].astype( np.int64 ) | ( rec[ rc, 1 ].astype( np.int64 ) << 16 ) | ( rec[ rc, 2 ].astype( np.int64 ) << 32 ) | ( LAY << 48 )
	cl = clusters( np.unique( place, return_inverse = True )[ 1 ].ravel(), N, opt[ 'refit_turn' ] )
	K = int( cl.max() ) + 1 if len( cl ) else 0
	cnt = np.maximum( np.bincount( cl, minlength = K ), 1 )[ :, None ]
	Y = np.stack( [ np.bincount( cl, y[ :, c ], K ) for c in range( 3 ) ], 1 ) / cnt
	val = Y[ cl ]
	if opt[ 'refit_tint' ] and K:

		# (one place, one material, one direction: the bake's light is the same for all of them - what differs between
		# their colours is tint. Each keeps its share of the mean colour, of the top level's mean light there)
		C = np.stack( [ np.bincount( cl, own[ :, c ], K ) for c in range( 3 ) ], 1 ) / cnt
		dev = np.zeros( K ); np.maximum.at( dev, cl, np.abs( own - C[ cl ] ).max( 1 ) )
		tinted = dev > 0.04 * C.max( 1 ) + 1e-5
		share = np.clip( own / np.maximum( C[ cl ], 1e-6 ), 0.125, 8 )
		val = np.where( tinted[ cl ][ :, None ], val * share, val )
		st[ 'tinted' ] = int( tinted[ cl ].sum() )

	nv = len( rec ); n = np.bincount( rc, minlength = nv )
	new = np.stack( [ np.bincount( rc, val[ :, c ], nv ) for c in range( 3 ) ], 1 )
	move = n > 0
	c16 = enc16( new[ move ] / n[ move ][ :, None ] ).astype( '<u2' )
	st[ 'moved' ] = int( move.sum() )
	st[ 'changed' ] = round( float( np.abs( tone( lin( c16 ) ) - tone( lin( rec[ move, 6:9 ] ) ) ).max( 1 ).mean() * 255 ), 2 ) if move.any() else 0.0
	out = buf.copy()
	w = m.get( 'w', 10 ); orec = out[ :nv * w * 2 ].view( '<u2' ).reshape( nv, w )
	orec[ move, 6:9 ] = c16
	name = '%s.%s.rawrf%s.bin' % ( key, lv, opt.get( 'tag', '' ) ) if dest else os.path.splitext( m.get( 'raw0', m[ 'raw' ] ) )[ 0 ] + '.refit.bin'
	path = os.path.join( dest or raw, name )
	out.tofile( path + '.tmp' ); os.replace( path + '.tmp', path )
	st.update( file = name, seconds = round( time.time() - t0, 1 ) )
	return st

def judge_mesh( m, rec, tri, flags, top, R, s ):

	# a mesh against the top level's light, at a lattice over its lit faces (no ray, nothing left out for being hidden)
	# -> per point: kind (0 ground, 1 wall, 2 prop, 3 other), the area it stands for, the mesh's light, the top level's
	P = pos( m, rec ); ok, fn, a2 = lit( m, rec, tri, P, 1e-7 )
	f = np.nonzero( ok )[ 0 ]; tf = tri[ f ]; fl = flags[ tf ]
	kind = np.where( ( ( fl & 4 ) > 0 ).all( 1 ), 0, np.where( ( ( fl & 2 ) > 0 ).all( 1 ), 2, np.where( np.abs( fn[ f ][ :, 1 ] ) < 0.5, 1, 3 ) ) )
	k = np.clip( np.ceil( np.sqrt( a2[ f ] ) / s ), 1, 6 ).astype( np.int64 )
	F, Bc = [], []
	for kk in np.unique( k ).tolist():

		pts = []
		for i in range( kk ):

			for j in range( kk - i ):

				pts.append( ( ( i + 1 / 3 ) / kk, ( j + 1 / 3 ) / kk ) )
				if i + j < kk - 1: pts.append( ( ( i + 2 / 3 ) / kk, ( j + 2 / 3 ) / kk ) )

		b = np.array( pts ); b = np.concatenate( [ 1 - b.sum( 1, keepdims = True ), b ], 1 )
		idx = np.nonzero( k == kk )[ 0 ]
		F.append( np.repeat( idx, len( b ) ) ); Bc.append( np.tile( b, ( len( idx ), 1 ) ) )

	fi = np.concatenate( F ); b = np.concatenate( Bc ); tv = tf[ fi ]
	X = P[ tv[ :, 0 ] ] * b[ :, 0:1 ] + P[ tv[ :, 1 ] ] * b[ :, 1:2 ] + P[ tv[ :, 2 ] ] * b[ :, 2:3 ]
	c = lin( np.minimum( 65528, np.round( rec[ :, 6:9 ].astype( np.float64 ) / 8 ) * 8 ) ) # (as the packer rounds it)
	g = c[ tv[ :, 0 ] ] * b[ :, 0:1 ] + c[ tv[ :, 1 ] ] * b[ :, 1:2 ] + c[ tv[ :, 2 ] ] * b[ :, 2:3 ]
	y, hit = top.light( X, fn[ f ][ fi ], rec[ tv[ :, 0 ], 3 ].astype( np.int64 ), R )
	w = ( a2[ f ] * 0.5 / np.bincount( fi, minlength = len( f ) ) )[ fi ]
	return kind[ fi ][ hit ], w[ hit ], g[ hit ], y[ hit ]

def numbers( kind, w, g, y ):

	# display steps of 255: mean, rms, p95 of the largest channel's difference; the mean signed difference of the
	# luminance (< 0: the mesh is darker); the mesh's linear light over the top level's
	LUM = np.array( [ 0.2126, 0.7152, 0.0722 ] ); out = {}
	e = np.abs( tone( g ) - tone( y ) ).max( 1 ) * 255; sg = ( tone( g ) @ LUM - tone( y ) @ LUM ) * 255
	for name, sel in ( ( 'all', kind >= 0 ), ( 'no ground', kind > 0 ), ( 'wall', kind == 1 ), ( 'prop', kind == 2 ), ( 'other', kind == 3 ) ):

		if not sel.any(): continue
		ee = e[ sel ]; ww = w[ sel ]; o = np.argsort( ee ); cw = np.cumsum( ww[ o ] )
		out[ name ] = dict( mean = float( ( ee * ww ).sum() / ww.sum() ), rms = float( np.sqrt( ( ee * ee * ww ).sum() / ww.sum() ) ), p95 = float( ee[ o ][ min( len( o ) - 1, int( np.searchsorted( cw, 0.95 * cw[ - 1 ] ) ) ) ] ),
			bias = float( ( sg[ sel ] * ww ).sum() / ww.sum() ), lin = float( ( ( g[ sel ] @ LUM ) * ww ).sum() / max( 1e-30, ( ( y[ sel ] @ LUM ) * ww ).sum() ) ), area = float( ww.sum() ) )

	return out

def top_of( ch, opt ): return opt.get( 'top' ) or ( 'fine' if 'fine' in ch else 'near' if 'near' in ch else None )

def work( job ):

	import resource
	raw, chunks, key, opt, dest = job
	t0 = time.time(); ch = chunks[ key ]; tl = top_of( ch, opt )
	_, rec, tri, _ = read( raw, ch[ tl ] )
	top = Top( ch[ tl ], rec, tri )
	res = dict( key = key, top = tl, levels = {} )
	if opt.get( 'judge' ):

		# (a trial: meshes of the dump - or the bake's own level, variant "bake" - against the top level)
		dump = json.load( open( os.path.join( dest, 'dump.json' ) ) )[ 'meshes' ]
		for v, lv in opt[ 'judge' ]:

			src = 'near' if lv == 'near' else 'mid'
			if v == 'bake': m = ch[ lv ]; _, r2, t2, f2 = read( raw, m ); r2 = r2[ :, :10 ]
			else:

				m = next( ( d for d in dump if d[ 'key' ] == key and d[ 'level' ] == lv and d[ 'variant' ] == v ), None )
				if not m: continue
				_, r2, t2, f2 = read( dest, dict( m, w = 10 ) )

			res[ 'levels' ][ v + ':' + lv ] = [ a.astype( np.float32 ) for a in judge_mesh( m, r2, t2, f2, top, float( OPT[ 'refit_reach' ][ src ] ), dict( lo = 0.14, mid = 0.10, near = 0.07 )[ lv ] ) ]

	else:

		for lv in opt[ 'refit_levels' ].split( ',' ):

			if lv in ch and lv != tl and ( tl == 'fine' or lv == 'mid' ): res[ 'levels' ][ lv ] = refit_level( raw, chunks, key, lv, top, opt, dest )

	res.update( seconds = round( time.time() - t0, 1 ), pairs = top.pairs, faces = len( top.T ), rss = resource.getrusage( resource.RUSAGE_SELF ).ru_maxrss // 1024 )
	return res

def main( argv ):

	global np
	import numpy as np, multiprocessing
	raw = os.path.abspath( argv[ 0 ] ); opt = dict( OPT, **( json.loads( argv[ 1 ] ) if len( argv ) > 1 else {} ) )
	dest = opt.get( 'dest' )
	rj = os.path.join( raw, 'raw.json' )
	man = json.load( open( rj ) ); chunks = { c[ 'k' ]: c for c in man[ 'chunks' ] }
	def save():

		json.dump( man, open( rj + '.tmp', 'w' ), separators = ( ',', ':' ) ); os.replace( rj + '.tmp', rj )

	if not opt[ 'refit' ]:

		# the bake's own files back (a kept bake packed again with refit=0)
		n = 0
		for c in man[ 'chunks' ]:

			for lv in ( 'mid', 'near' ):

				if lv in c and 'raw0' in c[ lv ]:

					if os.path.exists( os.path.join( raw, c[ lv ][ 'raw' ] ) ): os.remove( os.path.join( raw, c[ lv ][ 'raw' ] ) )
					c[ lv ][ 'raw' ] = c[ lv ].pop( 'raw0' ); n += 1

		if n or 'refit' in man: man.pop( 'refit', None ); save()
		log( 'off (refit=0): %d levels as the bake lit them' % n )
		return

	keys = [ k for k, c in chunks.items() if top_of( c, opt ) in c and any( lv in c and lv != top_of( c, opt ) for lv in opt[ 'refit_levels' ].split( ',' ) ) and ( not opt.get( 'cells' ) or k in opt[ 'cells' ] ) ]
	keys.sort( key = lambda k: ( - chunks[ k ][ top_of( chunks[ k ], opt ) ][ 'ni' ], k ) ) # the big ones first
	jobs = max( 1, min( int( opt[ 'refit_jobs' ] ) or 4, len( keys ) ) )
	todo = [ ( raw, chunks, k, opt, dest ) for k in keys ]
	if jobs > 1:

		with multiprocessing.get_context( 'fork' ).Pool( jobs, maxtasksperchild = 1 ) as pool: res = list( pool.imap_unordered( work, todo, 1 ) )

	else: res = [ work( j ) for j in todo ]
	res.sort( key = lambda r: r[ 'key' ] )
	rss = sorted( ( r[ 'rss' ] for r in res ), reverse = True )
	if opt.get( 'judge' ):

		tab = {}
		for name in sorted( { n for r in res for n in r[ 'levels' ] } ):

			parts = [ r[ 'levels' ][ name ] for r in res if name in r[ 'levels' ] ]
			tab[ name ] = numbers( *[ np.concatenate( [ p[ i ] for p in parts ] ).astype( np.float64 ) for i in range( 4 ) ] )
			tab[ name ][ 'cells' ] = len( parts )
			log( '%-18s %s' % ( name, ' | '.join( '%s %5.1f %5.1f %5.1f %+5.1f %.3f' % ( k[ :4 ], v[ 'mean' ], v[ 'rms' ], v[ 'p95' ], v[ 'bias' ], v[ 'lin' ] ) for k, v in tab[ name ].items() if k != 'cells' ) ) )

		json.dump( tab, open( os.path.join( dest, 'judge%s.json' % opt.get( 'tag', '' ) ), 'w' ), indent = 1 )
		log( 'judged against the top level (mean rms p95 bias lin): %d cells, %.1fs' % ( len( res ), time.time() - T0 ) )
		return

	tot = {}
	for r in res:

		for lv, st in r[ 'levels' ].items():

			t = tot.setdefault( lv, dict( cells = 0, verts = 0, held = 0, corners = 0, found = 0, moved = 0, tinted = 0, changed = 0.0 ) )
			t[ 'cells' ] += 1
			for k in ( 'verts', 'held', 'corners', 'found', 'moved', 'tinted' ): t[ k ] += st.get( k, 0 )
			t[ 'changed' ] += st[ 'changed' ] * st[ 'moved' ]

	for lv, t in tot.items():

		t[ 'changed' ] = round( t[ 'changed' ] / max( 1, t[ 'moved' ] ), 2 )
		log( '%s: %d cells, %d of %d vertices take the top level\'s light (%d held; %.1f %% of %d corners found a face; %d corners keep a tint of their own), %.1f display steps on average' % ( lv, t[ 'cells' ], t[ 'moved' ], t[ 'verts' ], t[ 'held' ],
			100 * t[ 'found' ] / max( 1, t[ 'corners' ] ), t[ 'corners' ], t[ 'tinted' ], t[ 'changed' ] ) )

	info = dict( settings = { k: opt[ k ] for k in OPT }, levels = tot, seconds = round( time.time() - T0, 1 ), jobs = jobs, peak_mb = sum( rss[ :jobs ] ) )
	if dest:

		# (a trial: into the dump of tools/next/lightmap.py as the variant rawrf<tag>, the bake untouched)
		dj = os.path.join( dest, 'dump.json' ); d = json.load( open( dj ) ) if os.path.exists( dj ) else dict( meshes = [] ); v = 'rawrf' + opt.get( 'tag', '' )
		new = [ dict( key = r[ 'key' ], level = lv, variant = v, raw = st[ 'file' ], **{ k: chunks[ r[ 'key' ] ][ lv ][ k ] for k in ( 'min', 'q', 'nv', 'ni', 'sub' ) } ) for r in res for lv, st in r[ 'levels' ].items() ]
		got = { ( m[ 'key' ], m[ 'level' ] ) for m in new }
		d[ 'meshes' ] = [ m for m in d[ 'meshes' ] if not ( m[ 'variant' ] == v and ( m[ 'key' ], m[ 'level' ] ) in got ) ] + new
		json.dump( d, open( dj + '.tmp', 'w' ) ); os.replace( dj + '.tmp', dj )
		json.dump( dict( info, cells = res ), open( os.path.join( dest, 'refit%s.json' % opt.get( 'tag', '' ) ), 'w' ), indent = 1 )

	else:

		for r in res:

			for lv, st in r[ 'levels' ].items():

				m = chunks[ r[ 'key' ] ][ lv ]
				m.setdefault( 'raw0', m[ 'raw' ] ); m[ 'raw' ] = st[ 'file' ]

		man[ 'refit' ] = info; save()

	log( 'done: %d cells, %d at once, %.1fs, the largest %d of them together %d MB' % ( len( res ), jobs, time.time() - T0, jobs, info[ 'peak_mb' ] ) )

if __name__ == '__main__': main( sys.argv[ sys.argv.index( '--' ) + 1: ] if '--' in sys.argv else sys.argv[ 1: ] )
