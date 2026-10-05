#!/usr/bin/env python3
# The light of the reduced levels, fitted to the top level of the same bake - a lookup and a least-squares fit, no ray.
#
# Why: the bake lights every level at its own vertices and the packer then removes most of them (simplify() keeps a
# subset; the light on a face is whatever its three survivors happened to have: smudged, and another smudge at each
# level, so the picture jumps when a cell changes level). Here the vertices that are left take the values that, as
# the page interpolates them over the level's faces, come nearest to the light the TOP level of the cell shows at
# the same places (fine; near in a bake without fine) - Least Squares Vertex Baking (Kavan, Bargteil, Sloan 2011)
# with the top level's vertex light as the target instead of rays.
#
#   points    a lattice over every lit face of the level (`refit_step` apart); each asks the top level for the light
#             at the nearest face of the same material that looks the same way, within `refit_reach`. A point that
#             finds none says what the level has there itself, and counts a twentieth
#   covered   a point with a face of the top level a short way OUT along its normal (within `refit_cover`: same
#             material, looking the same way) takes that face's light, the outermost one's: what lies under a
#             detail layer was lit under it (the deck under laid roof tiles is black at the top level, where nobody
#             sees it), and a level that has lost the layer shows what was under it
#   unknowns  one for the records of a place that carry one colour today (what is one vertex stays one, a crease
#             stays a crease): no vertex is added, no file grows
#   fit       least squares on linear light over the points, + the mean of each face held to the mean of its points,
#             + a little against a change of slope over an edge inside a flat piece (`refit_smooth`), clamped to
#             what a colour can store; conjugate gradients, numpy alone
#   held      as the bake has them, byte for byte: the ground (one ground at every level - the packer copies the top
#             level's), every vertex at a place a neighbouring cell also has one at (no seam at a cell border), what
#             the bake marked to stay, what glows, glass / water / cut-outs
#   tint      a level's face over top-level faces whose vertices carry a pattern of their own (two colours at one
#             place, one material, one direction: the stripes of an awning) cannot show that pattern with three
#             values, and a fit would paint the mean of it (an even pink). Its vertices stay as baked (`refit_tint`)
#   two looks a bake with a second look (LOOK2 of webbake_bl.py: a second colour in every vertex) is fitted in six
#             channels - the same points, the same faces of the top level, the same unknowns; a fit file then holds
#             six values a vertex. What is said of the error is the first look's
#
# It runs between two passes of the packer, because a reduced level exists only as the packer's index list
# (tools/lsvb_pack.mjs). webbake.py's bake_one(), before it calls the packer:
#
#   if pack.get( 'refit' ): pack = lsvb_refit.run( ROOT, out, pack, a.jobs, find_blender(), env ) or fail( 'refit' )
#
#   pass 1   node tools/webbake_pack.mjs <out> { .., refit: 'sim' }   -> .raw/<cell>.<level>.sim.bin + sim.json
#   the fit  <python with numpy> tools/lsvb_refit.py -- <out>/.raw    -> .raw/<cell>.<level>.fit.bin + fit.json
#   pass 2   the caller's own packer run, with refit: 1 (the settings run() returns)
#
# Everything is made again from the raw bake on every run; nothing is kept from an earlier bake. Off unless the
# packer's settings have `refit` (python3 tools/webbake.py --set pack.refit=1).
# numpy alone, no Blender process: this python if it has numpy, else the python Blender ships, run by itself.

import sys, os, json, time, glob, shutil, struct, hashlib, subprocess

OPT = dict(
	refit = 0, # 1: fit the reduced levels' light to the top level's
	refit_levels = 'lo,mid,near,ring', # the packed levels that are fitted (never a cell's top level)
	refit_step = dict( lo = 0.10, mid = 0.07, near = 0.05, ring = 0.05 ), # the lattice: about this far between two points (m)
	refit_reach = dict( lo = 0.6, mid = 0.35, near = 0.25, ring = 0.25 ), # how far from a point the top level's surface may be (m)
	refit_cover = 0.08, # a face of the top level this far out along a point's normal covers it: the point takes its light (0: off)
	refit_smooth = 0.1, # the weight of the edge term (0: none)
	refit_tint = 0.3, # a face with more than this share of its points on patterned faces of the top level stays as baked (0: fit all)
	refit_turn = 0.96, # corners of one place whose faces look the same way within this cosine are one direction
	refit_jobs = 0 ) # cells at once (0: the bake's --jobs, 4 at most - more of them wait for each other's memory, and the largest cells take 2 GB each)
K_OPAQUE = 0
MAGIC = 0x4656534c; VERSION = 1 # "LSVF" (tools/lsvb_pack.mjs reads it)
CG_MAX = 200; CG_TOL = 1e-8 # the solver: steps at most, and the residual (squared, of what it started with) it stops at
KMAX = 32 # a face is cut in no more than KMAX * KMAX parts
TOPBOX = 2.0 # the largest boxes of the search (m)
COVER_COS = 0.8; COVER_MIN = 0.004 # a covering face: looks the same way within this cosine, and lies at least this far out (m)
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

def run( root, out, pack, jobs = 4, blender = None, env = None, packer = None ):

	# webbake.py's bake_one(), before the packer -> the packer's settings for the pass that packs (None: failed)
	t0 = time.time()
	node = shutil.which( 'node' ); raw = os.path.join( out, '.raw' ); man = os.path.join( out, 'manifest.json' )
	if not node: return None
	r = subprocess.run( [ node, packer or os.path.join( root, 'tools/webbake_pack.mjs' ), out, json.dumps( dict( pack, refit = 'sim', keep_raw = True ) ) ], cwd = root, env = env, stdout = subprocess.DEVNULL )
	if r.returncode != 0 or not os.path.exists( os.path.join( raw, 'sim.json' ) ): return None
	if os.path.exists( man ): os.remove( man ) # (the first pass's: not this bake's manifest yet)
	opt = { k: v for k, v in pack.items() if k.startswith( 'refit' ) }
	if not opt.get( 'refit_jobs' ): opt[ 'refit_jobs' ] = max( 1, min( 4, jobs ) )
	print( '[refit] the levels as the packer leaves them: %.1fs' % ( time.time() - t0 ), flush = True )
	r = subprocess.run( python( blender ) + [ os.path.abspath( __file__ ), '--', raw, json.dumps( opt ) ], env = env )
	if r.returncode != 0 or not os.path.exists( os.path.join( raw, 'fit.json' ) ): return None
	return dict( pack, refit = 1 )

# ------------------------------------------------------------------------------------------ what follows needs numpy

T0 = time.time()
def log( *a ): print( '[refit %6.1fs]' % ( time.time() - T0 ), *a, flush = True )

def read( raw, m ):

	# a level as the bake wrote it -> records (u16, nv x w), triangles, flags
	w = m.get( 'w', 10 ); nv, ni = m[ 'nv' ], m[ 'ni' ]
	buf = np.fromfile( os.path.join( raw, m[ 'raw' ] ), np.uint8 )
	if len( buf ) < nv * w * 2 + ni * 4 + nv: sys.exit( 'lsvb_refit: %s is shorter than raw.json says' % m[ 'raw' ] )
	rec = buf[ :nv * w * 2 ].view( '<u2' ).reshape( nv, w )
	tri = buf[ nv * w * 2:nv * w * 2 + ni * 4 ].view( '<u4' ).reshape( - 1, 3 ).astype( np.int64 )
	return rec, tri, buf[ nv * w * 2 + ni * 4:nv * w * 2 + ni * 4 + nv ]

def pos( m, rec ): return np.array( m[ 'min' ], np.float64 )[ None ] + rec[ :, :3 ].astype( np.float64 ) * m[ 'q' ]
def lin( c16 ): return ( c16.astype( np.float64 ) / 65535 ) ** 2 * 64 # a stored colour -> linear light x tint
def col2( rec ): return { 14: 10, 18: 12 }.get( rec.shape[ 1 ], 0 ) # the word a second look's colour starts at (0: none)
def cols( rec ): return rec[ :, 6:9 ] if not col2( rec ) else np.concatenate( [ rec[ :, 6:9 ], rec[ :, col2( rec ):col2( rec ) + 3 ] ], 1 ) # the stored colours: 3, or 6 with a second look
def enc16( c ): return np.round( np.sqrt( np.clip( c, 0, 64 ) / 64 ) * 65535 )
def pkey( rec ): return rec[ :, 0 ].astype( np.int64 ) | ( rec[ :, 1 ].astype( np.int64 ) << 16 ) | ( rec[ :, 2 ].astype( np.int64 ) << 32 )

def tone( c ):

	# about what the page shows of a lit colour on a mid-grey surface (exposure 1.25, ACES, display gamma): 0 .. 1
	x = np.maximum( c, 0 ) * ( 1.25 * 0.5 )
	return np.clip( x * ( 2.51 * x + 0.03 ) / ( x * ( 2.43 * x + 0.59 ) + 0.14 ), 0, 1 ) ** ( 1 / 2.2 )

def opaque( sub, nt ):

	out = np.zeros( nt, bool )
	for s in sub:

		if s[ 0 ] == K_OPAQUE: out[ s[ 1 ] // 3:( s[ 1 ] + s[ 2 ] ) // 3 ] = True

	return out

def lit( sub, rec, tri, P, eps ):

	# the faces whose light is baked: opaque, nothing of them glowing, not a sliver -> which, the faces' normals, twice their areas
	n = np.cross( P[ tri[ :, 1 ] ] - P[ tri[ :, 0 ] ], P[ tri[ :, 2 ] ] - P[ tri[ :, 0 ] ] ); a2 = np.linalg.norm( n, axis = 1 )
	return opaque( sub, len( tri ) ) & ~ ( ( rec[ :, 9 ] == 65535 ) | ( rec[ :, 9 ] == 65533 ) )[ tri ].any( 1 ) & ( a2 > eps ), n / np.maximum( a2, 1e-20 )[ :, None ], a2

def wkeys( m, rec ):

	# a place in the world as one number, whatever cell's grid it is on
	w = np.round( ( np.array( m[ 'min' ], np.float64 )[ None ] + rec[ :, :3 ].astype( np.float64 ) * m[ 'q' ] ) * 1024 ).astype( np.int64 ) + ( 1 << 20 )
	return ( w[ :, 0 ] << 42 ) | ( w[ :, 1 ] << 21 ) | w[ :, 2 ]

def shared( raw, chunks, key, lv, rec ):

	# the vertices at a place where another cell has one too, at this raw level (walls run over cell borders: both
	# cells carry the vertex with the same light, and a cell that changed it alone would open a seam)
	m = chunks[ key ][ lv ]; out = np.zeros( len( rec ), bool ); mine = None
	for k, c in chunks.items():

		o = c.get( lv )
		if k == key or not o or any( o[ 'min' ][ i ] > m[ 'max' ][ i ] + 0.01 or o[ 'max' ][ i ] < m[ 'min' ][ i ] - 0.01 for i in range( 3 ) ): continue
		if mine is None: mine = wkeys( m, rec )
		r = np.memmap( os.path.join( raw, o[ 'raw' ] ), '<u2', 'r', shape = ( o[ 'nv' ], o.get( 'w', 10 ) ) )
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

class Top:

	# the lit faces of a cell's top level, to be asked for the light at a point

	def __init__( self, m, rec, tri ):

		P = pos( m, rec )
		ok, n, a2 = lit( m[ 'sub' ], rec, tri, P, 1e-9 )
		T = tri[ ok ]; n = n[ ok ]
		self.T = T; self.N = n; self.col = lin( cols( rec ) ); self.lay = rec[ T[ :, 0 ], 3 ].astype( np.int64 )
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

		self.F = F; self.n32 = n.astype( np.float32 ); self.pairs = 0
		self.place = pkey( rec[ T.ravel() ] ) | ( np.repeat( self.lay, 3 ) << 48 )

	def patterned( self, turn ):

		# the faces with a corner at a place where the records of one material and one direction carry different
		# colours: the bake's light is one there, so what differs is tint painted into the vertices
		if not len( self.T ): return np.zeros( 0, bool )
		cl = clusters( np.unique( self.place, return_inverse = True )[ 1 ].ravel(), np.repeat( self.N, 3, 0 ), turn )
		own = self.col[ self.T.ravel(), :3 ]
		o = np.argsort( cl, kind = 'stable' ); start = np.nonzero( np.r_[ True, cl[ o ][ 1: ] != cl[ o ][ :- 1 ] ] )[ 0 ]
		hi = np.maximum.reduceat( own[ o ], start, 0 ); lo = np.minimum.reduceat( own[ o ], start, 0 )
		return ( ( hi - lo ).max( 1 ) > 0.08 * hi.max( 1 ) + 2e-5 )[ cl ].reshape( - 1, 3 ).any( 1 )

	def reach( self, f, cell, h, g ):

		# whether face f may come within g of the box of side h whose low corner is `cell` (never no when it does)
		e = np.float32( g + 1e-4 ); h2 = np.float32( h / 2 ); mid = cell + h2
		B = self.FB[ f ]
		ok = ( np.abs( mid - B[ :, 0:3 ] ) <= B[ :, 3:6 ] + ( e + h2 ) ).all( 1 )
		i = np.nonzero( ok )[ 0 ]; F = self.F[ f[ i ] ]; mid = mid[ i ]
		d = np.einsum( 'ikj,ij->ik', F[ :, :, 0:3 ], mid ) - F[ :, :, 3 ]; d[ :, 0 ] = np.abs( d[ :, 0 ] )
		ok[ i ] = ( d <= e + h2 * F[ :, :, 4 ] ).all( 1 )
		return ok

	def near( self, X, LAY, g, h ):

		# the faces that may be within g of each point, as ( points, faces ) a block at a time. The points' boxes of
		# side h, and the boxes of twice, four times .. that side they are in: a face is given to the largest boxes its
		# bounds touch and goes on only into the halves it still reaches
		L = max( 0, int( np.ceil( np.log2( TOPBOX / h ) ) ) )
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

	def stage( self, X, N, LAY, g, R ):

		# the best face within g of each point: same material, looking the same way; least of distance + what it is
		# turned by ( a quarter of R at a right angle ), of equals the first -> face ( - 1 none ), weights, that measure
		nq = len( X ); ot = np.full( nq, - 1, np.int64 ); ob = np.zeros( ( nq, 3 ) ); om = np.full( nq, 1e9 )
		N32 = N.astype( np.float32 )
		for q, t in self.near( X, LAY, g, 2 * g ):

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

	def cover( self, X, N, LAY, c ):

		# the outermost face met within c along each point's normal: same material, looking the same way -> face
		# ( - 1 none ), weights
		nq = len( X ); ot = np.full( nq, - 1, np.int64 ); ob = np.zeros( ( nq, 3 ) ); os_ = np.zeros( nq )
		for q, t in self.near( X, LAY, c, 2 * c ):

			self.pairs += len( q )
			dn = ( self.N[ t ] * N[ q ] ).sum( 1 )
			keep = dn > COVER_COS
			q = q[ keep ]; t = t[ keep ]; dn = dn[ keep ]
			if not len( q ): continue
			s = ( ( self.A[ t ] - X[ q ] ) * self.N[ t ] ).sum( 1 ) / dn
			keep = ( s > COVER_MIN ) & ( s <= c )
			q = q[ keep ]; t = t[ keep ]; s = s[ keep ]
			if not len( q ): continue
			a = self.A[ t ]; e0 = self.B[ t ] - a; e1 = self.C[ t ] - a; e2 = X[ q ] + s[ :, None ] * N[ q ] - a
			d00 = ( e0 * e0 ).sum( 1 ); d01 = ( e0 * e1 ).sum( 1 ); d11 = ( e1 * e1 ).sum( 1 ); d20 = ( e2 * e0 ).sum( 1 ); d21 = ( e2 * e1 ).sum( 1 )
			den = np.maximum( d00 * d11 - d01 * d01, 1e-30 )
			v = ( d11 * d20 - d01 * d21 ) / den; w = ( d00 * d21 - d01 * d20 ) / den
			keep = ( v >= - 1e-4 ) & ( w >= - 1e-4 ) & ( v + w <= 1 + 1e-4 )
			q = q[ keep ]; t = t[ keep ]; s = s[ keep ]; v = np.clip( v[ keep ], 0, 1 ); w = np.clip( w[ keep ], 0, 1 )
			if not len( q ): continue
			o = np.lexsort( ( t, - s, q ) ); first = np.ones( len( o ), bool ); first[ 1: ] = q[ o ][ 1: ] != q[ o ][ :- 1 ]
			o = o[ first ]; q = q[ o ]; better = s[ o ] > os_[ q ]
			o = o[ better ]; q = q[ better ]
			ot[ q ] = t[ o ]; os_[ q ] = s[ o ]; ob[ q ] = np.stack( [ np.maximum( 0, 1 - v[ o ] - w[ o ] ), v[ o ], w[ o ] ], 1 )

		return ot, ob

	def light( self, X, N, LAY, R, cover = 0 ):

		# -> the light at the place found for each point (its face's vertex light, as the page interpolates it), the
		# face ( - 1: none in reach ). Near first: a point whose best face within g measures g or less cannot do
		# better farther out
		nq = len( X ); bt = np.full( nq, - 1, np.int64 ); bb = np.zeros( ( nq, 3 ) )
		left = np.arange( nq )
		for g in [ r for r in ( 0.02, 0.06, 0.18 ) if r < R ] + [ R ]:

			if not len( left ): break
			t, b, m = self.stage( X[ left ], N[ left ], LAY[ left ], g, R )
			done = ( t >= 0 ) & ( ( m <= g ) | ( g >= R ) )
			bt[ left[ done ] ] = t[ done ]; bb[ left[ done ] ] = b[ done ]
			left = left[ ~ done ]

		self.covered = 0
		if cover > 0 and nq:

			ct, cb = self.cover( X, N, LAY, cover )
			over = ct >= 0; self.covered = int( over.sum() )
			bt[ over ] = ct[ over ]; bb[ over ] = cb[ over ]

		hit = bt >= 0; tv = self.T[ bt[ hit ] ]; b = bb[ hit ]
		y = np.zeros( ( nq, self.col.shape[ 1 ] ) )
		y[ hit ] = self.col[ tv[ :, 0 ] ] * b[ :, 0:1 ] + self.col[ tv[ :, 1 ] ] * b[ :, 1:2 ] + self.col[ tv[ :, 2 ] ] * b[ :, 2:3 ]
		return y, bt

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

def face_points( Pf, faces, s ):

	# a lattice over the faces `faces` of Pf (n, 3, 3), about s apart -> face, weights of its corners, the point
	a = Pf[ faces, 0 ]; b = Pf[ faces, 1 ]; c = Pf[ faces, 2 ]
	a2 = np.linalg.norm( np.cross( b - a, c - a ), axis = 1 )
	longest = np.sqrt( np.maximum( np.maximum( ( ( b - a ) ** 2 ).sum( 1 ), ( ( c - b ) ** 2 ).sum( 1 ) ), ( ( a - c ) ** 2 ).sum( 1 ) ) )
	k = np.clip( np.ceil( np.sqrt( a2 ) / s ), 1, KMAX ).astype( np.int64 )
	k = np.maximum( k, np.minimum( np.ceil( longest / ( 4 * s ) ), 4 ).astype( np.int64 ) ) # (a long thin one: along its length too)
	F, Bc = [], []
	for kk in np.unique( k ).tolist():

		idx = np.nonzero( k == kk )[ 0 ]; L = lattice( kk )
		F.append( np.repeat( idx, len( L ) ) ); Bc.append( np.tile( L, ( len( idx ), 1 ) ) )

	fl = np.concatenate( F ) if F else np.zeros( 0, np.int64 ); bc = np.concatenate( Bc ) if Bc else np.zeros( ( 0, 3 ) )
	o = np.argsort( fl, kind = 'stable' ); fl = fl[ o ]; bc = bc[ o ]
	return faces[ fl ], bc, a[ fl ] * bc[ :, 0:1 ] + b[ fl ] * bc[ :, 1:2 ] + c[ fl ] * bc[ :, 2:3 ]

def solve( cv, K, fixed, x0, Pf, sf, b, y, q, alpha ):

	# the values of the K unknowns (linear light, 3 channels - 6 with a second look) that come nearest to y at the points ( face sf of cv,
	# weights b, each counting q times its share of the face's area ) -> x (K, 3), steps of the solver
	nf = len( cv ); ids = cv.ravel(); NC = y.shape[ 1 ]
	fnv = np.cross( Pf[ :, 1 ] - Pf[ :, 0 ], Pf[ :, 2 ] - Pf[ :, 0 ] ); a2 = np.linalg.norm( fnv, axis = 1 )
	nrm = fnv / np.maximum( a2, 1e-20 )[ :, None ]; ar = a2 * 0.5
	qa = q * ar[ sf ] / np.maximum( np.bincount( sf, minlength = nf )[ sf ], 1 )
	fw = np.bincount( sf, qa, nf ); has = fw > 0
	ybar = np.stack( [ np.bincount( sf, qa * y[ :, c ], nf ) for c in range( NC ) ], 1 ) / np.maximum( fw, 1e-30 )[ :, None ]
	G = np.zeros( ( nf, 3, 3 ) ); h = np.zeros( ( nf, 3, NC ) ) # per face: corner x corner, corner x channel
	for i in range( 3 ):

		wb = qa * b[ :, i ]
		for c in range( NC ): h[ :, i, c ] = np.bincount( sf, wb * y[ :, c ], nf )
		for j in range( i, 3 ): G[ :, i, j ] = G[ :, j, i ] = np.bincount( sf, wb * b[ :, j ], nf )

	# the mean of a face: ( x0 + x1 + x2 ) / 3 held to the mean light of its points, as heavy as they are together
	G += ( fw / 9 )[ :, None, None ]; h += ( fw[ :, None ] * ybar / 3 )[ :, None, : ]
	def acc( idx, v ): return np.stack( [ np.bincount( idx, v[ :, c ], K ) for c in range( v.shape[ 1 ] ) ], 1 )
	# the edge term: pairs of faces over an edge both run along with the same two unknowns, in one flat piece - the
	# slope of the light on one side less that on the other, squared (a light that is one line over both costs nothing)
	Mp = None
	live = ( ~ fixed )[ cv ].any( 1 )
	if alpha > 0 and nf:

		el = Pf[ :, [ 1, 2, 0 ] ] - Pf # edge k: from corner k to k + 1
		slim = ( el ** 2 ).sum( 2 ).max( 1 ) / np.maximum( 2 * ar, 1e-20 )
		e0 = cv; e1 = cv[ :, [ 1, 2, 0 ] ]
		ek = ( np.minimum( e0, e1 ) * K + np.maximum( e0, e1 ) ).ravel()
		o = np.argsort( ek, kind = 'stable' )
		uq, first, n = np.unique( ek[ o ], return_index = True, return_counts = True )
		two = n == 2
		i0 = o[ first[ two ] ]; i1 = o[ first[ two ] + 1 ]
		t = i0 // 3; kt = i0 % 3; u = i1 // 3; ku = i1 % 3
		good = ( cv[ t, kt ] != cv[ t, ( kt + 1 ) % 3 ] ) & ( cv[ t, kt ] == cv[ u, ( ku + 1 ) % 3 ] ) & ( t != u ) & ( ( nrm[ t ] * nrm[ u ] ).sum( 1 ) > 0.9 ) & ( slim[ t ] < 200 ) & ( slim[ u ] < 200 ) & ( live[ t ] | live[ u ] )
		t, kt, u, ku = t[ good ], kt[ good ], u[ good ], ku[ good ]
		if len( t ):

			# the gradient of the hat of corner m of a face: n x ( the edge opposite, from m + 1 to m + 2 ) / 2 a
			gh = np.cross( nrm[ :, None, : ], el[ :, [ 1, 2, 0 ] ] ) / np.maximum( 2 * ar, 1e-20 )[ :, None, None ]
			dd = el[ t, kt ]; dd = dd / np.maximum( np.linalg.norm( dd, axis = 1 ), 1e-20 )[ :, None ]
			def grad( fc ):

				# ... in the frame of the pair: along the shared edge, and across it (the two faces laid out flat)
				pp = np.cross( nrm[ fc ], dd )
				return np.stack( [ np.einsum( 'pmx,px->pm', gh[ fc ], dd ), np.einsum( 'pmx,px->pm', gh[ fc ], pp ) ], 1 )

			Mp = np.concatenate( [ grad( t ), - grad( u ) ], 2 ) # pairs, 2, 6
			idx6 = np.concatenate( [ cv[ t ], cv[ u ] ], 1 )
			ew = np.where( has, fw / np.maximum( ar, 1e-20 ), 1.0 )
			Wp = alpha * 0.5 * ( ar[ t ] + ar[ u ] ) ** 2 * 0.5 * ( ew[ t ] + ew[ u ] )
			MpT = np.ascontiguousarray( ( Mp * Wp[ :, None, None ] ).transpose( 0, 2, 1 ) )

	lam = 1e-4
	dm = np.bincount( ids, np.stack( [ G[ :, 0, 0 ], G[ :, 1, 1 ], G[ :, 2, 2 ] ], 1 ).ravel(), K )
	D0 = np.maximum( dm, 1e-6 * max( float( dm.mean() ) if K else 0.0, 1e-30 ) )
	Dj = dm + lam * D0
	if Mp is not None: Dj = Dj + np.bincount( idx6.ravel(), ( ( Mp ** 2 ).sum( 1 ) * Wp[ :, None ] ).ravel(), K )
	Dj = Dj[ :, None ]; D0 = D0[ :, None ]
	def mv( v ):

		out = acc( ids, np.matmul( G, v[ cv ] ).reshape( - 1, NC ) ) + lam * D0 * v
		if Mp is not None:

			out += acc( idx6.ravel(), np.matmul( MpT, np.matmul( Mp, v[ idx6 ] ) ).reshape( - 1, NC ) )

		return out

	rhs = acc( ids, h.reshape( - 1, NC ) ) + lam * D0 * x0
	def cg( x, free ):

		rr = ( rhs - mv( x ) ) * free; z = rr / Dj; p = z.copy(); rz = ( rr * z ).sum( 0 ); r0 = np.maximum( rz, 1e-30 )
		it = 0
		for it in range( CG_MAX ):

			Ap = mv( p ) * free; al = rz / np.maximum( ( p * Ap ).sum( 0 ), 1e-30 )
			x = x + al[ None ] * p; rr = rr - al[ None ] * Ap
			z = rr / Dj; rz2 = ( rr * z ).sum( 0 )
			if ( rz2 / r0 ).max() < CG_TOL: break
			p = z + ( rz2 / np.maximum( rz, 1e-30 ) )[ None ] * p; rz = rz2

		return x, it + 1

	free = ( ~ fixed )[ :, None ] & np.ones( ( 1, NC ), bool )
	x = x0.copy(); its = 0; fr = free
	for rnd in range( 3 ):

		x, n = cg( np.where( fr, x, np.clip( x, 0, 64 ) ), fr ); its += n
		# over- and undershoot (next to a hard edge of the light): held at the limit, the others solved again around it
		out = ( ( x < 0 ) | ( x > 64 ) ) & fr
		if not out.any(): break
		fr = fr & ~ out

	return np.where( free, np.clip( x, 0, 64 ), x0 ), its

def errors( g, y, w ):

	# what the page shows of g against y, in display steps of 255 (the largest channel's): rms, mean signed difference
	if not len( w ) or w.sum() <= 0: return [ 0.0, 0.0, 0.0 ]
	e = ( tone( g ) - tone( y ) ) * 255; m = np.abs( e ).max( 1 )
	return [ float( ( m * m * w ).sum() ), float( ( e.mean( 1 ) * w ).sum() ), float( w.sum() ) ]

def fit_level( raw, chunks, e, top, pat, opt, cache, done ):

	t0 = time.time()
	key, name, src = e[ 'k' ], e[ 'name' ], e[ 'src' ]; m = chunks[ key ][ src ]
	if src not in cache:

		rec, rtri, flags = read( raw, m )
		hold = ( ( flags & 5 ) > 0 ) | ( rec[ :, 9 ] == 65535 ) | ( rec[ :, 9 ] == 65533 ) | shared( raw, chunks, key, src, rec )
		hold[ rtri[ ~ opaque( m[ 'sub' ], len( rtri ) ) ].ravel() ] = True # (glass, water, cut-outs)
		cache.clear(); cache[ src ] = ( rec, flags, hold, pos( m, rec ) )

	rec, flags, hold, P = cache[ src ]
	nv = len( rec )
	ib = np.fromfile( os.path.join( raw, e[ 'file' ] ), '<u4' )
	if nv != e[ 'nv' ] or len( ib ) != e[ 'ni' ] or hashlib.sha1( ib.tobytes() ).hexdigest() != e[ 'ib' ]: sys.exit( 'lsvb_refit: %s is not what sim.json says' % e[ 'file' ] )
	tri = ib.astype( np.int64 ).reshape( - 1, 3 )
	file = '%s.%s.fit.bin' % ( key, name ); path = os.path.join( raw, file )
	def write( idx, c16 ):

		with open( path + '.tmp', 'wb' ) as fh:

			fh.write( struct.pack( '<5I', MAGIC, VERSION, nv, len( ib ), len( idx ) ) + bytes.fromhex( e[ 'ib' ] ) + bytes.fromhex( e[ 'vb' ] ) )
			fh.write( idx.astype( '<u4' ).tobytes() ); fh.write( c16.astype( '<u2' ).tobytes() )

		os.replace( path + '.tmp', path )

	# a level whose faces are all faces of one already fitted (the ring is near without its smallest parts, in another
	# order) takes that one's values: the two show one light, and the work is done once
	tk = ( tri[ :, 0 ] * nv + tri[ :, 1 ] ) * nv + tri[ :, 2 ]
	for other, ( osrc, otk, oidx, oc16 ) in done.items():

		if osrc == src and np.isin( tk, otk ).all():

			used = np.isin( oidx, ib )
			write( oidx[ used ], oc16[ used ] )
			return dict( verts = int( len( np.unique( ib ) ) ), tris = len( tri ), moved = int( used.sum() ), same_as = other, file = file, seconds = round( time.time() - t0, 1 ), err = dict( all = [ 0.0 ] * 6, wall = [ 0.0 ] * 6 ) )

	ok, fn, a2 = lit( e[ 'sub' ], rec, tri, P, 1e-7 )
	tf = tri[ ok ]; fn = fn[ ok ]; r = tf.ravel(); nf = len( tf )
	st = dict( verts = int( len( np.unique( ib ) ) ), tris = len( tri ), lit = nf )
	# the unknowns: the records of one place with one colour and gloss
	col = rec[ r, 6:10 ].astype( np.int64 )
	ck = np.unique( col[ :, 0 ] | ( col[ :, 1 ] << 16 ) | ( col[ :, 2 ] << 32 ) | ( col[ :, 3 ] << 48 ), return_inverse = True )
	if col2( rec ):

		# (a second look: one unknown is one colour in both)
		c2 = rec[ r, col2( rec ):col2( rec ) + 3 ].astype( np.int64 )
		k2 = np.unique( c2[ :, 0 ] | ( c2[ :, 1 ] << 16 ) | ( c2[ :, 2 ] << 32 ), return_inverse = True )
		ck = np.unique( ck[ 1 ].ravel() * len( k2[ 0 ] ) + k2[ 1 ].ravel(), return_inverse = True )

	pk = np.unique( pkey( rec[ r ] ), return_inverse = True )
	ids = np.unique( pk[ 1 ].ravel() * len( ck[ 0 ] ) + ck[ 1 ].ravel(), return_inverse = True )
	K = len( ids[ 0 ] ); ids = ids[ 1 ].ravel(); cv = ids.reshape( nf, 3 )
	x0 = np.zeros( ( K, 6 if col2( rec ) else 3 ) ); x0[ ids ] = lin( cols( rec[ r ] ) )
	fixed = np.bincount( ids, hold[ r ], K ) > 0
	# the points, on every face with an unknown that may move
	Pf = P[ tf ]
	sf, b, X = face_points( Pf, np.nonzero( ( ~ fixed )[ cv ].any( 1 ) )[ 0 ], float( opt[ 'refit_step' ][ name ] ) )
	y, t = top.light( X, fn[ sf ], rec[ tf[ sf, 0 ], 3 ].astype( np.int64 ), float( opt[ 'refit_reach' ][ name ] ), float( opt.get( 'refit_cover' ) or 0 ) )
	hit = t >= 0
	mine = x0[ cv[ sf, 0 ] ] * b[ :, 0:1 ] + x0[ cv[ sf, 1 ] ] * b[ :, 1:2 ] + x0[ cv[ sf, 2 ] ] * b[ :, 2:3 ]
	y = np.where( hit[ :, None ], y, mine )
	st.update( points = len( sf ), found = int( hit.sum() ), covered = top.covered )
	if pat is not None and len( sf ):

		share = np.bincount( sf[ hit ], pat[ t[ hit ] ], nf ) / np.maximum( np.bincount( sf[ hit ], minlength = nf ), 1 )
		keep = share > float( opt[ 'refit_tint' ] )
		was = int( fixed.sum() ); fixed[ cv[ keep ].ravel() ] = True
		st.update( tint_faces = int( keep.sum() ), tint_vars = int( fixed.sum() ) - was )

	x, its = solve( cv, K, fixed, x0, Pf, sf, b, y, np.where( hit, 1.0, 0.05 ), float( opt[ 'refit_smooth' ] ) )
	# what moved: the vertices of the free unknowns whose stored colour is another now
	var = np.full( nv, - 1, np.int64 ); var[ r ] = ids
	idx = np.nonzero( ( var >= 0 ) & ~ fixed[ np.maximum( var, 0 ) ] )[ 0 ]
	c16 = enc16( x[ var[ idx ] ] ).astype( '<u2' )
	ch = ( c16 != cols( rec[ idx ] ) ).any( 1 ); idx = idx[ ch ]; c16 = c16[ ch ]
	# against the top level at the points, before and after (the page rounds a colour to 13 bits: so does this)
	r13 = lambda v: lin( np.minimum( 65528, np.round( enc16( v ) / 8 ) * 8 ) )
	at = lambda v: v[ cv[ sf, 0 ] ] * b[ :, 0:1 ] + v[ cv[ sf, 1 ] ] * b[ :, 1:2 ] + v[ cv[ sf, 2 ] ] * b[ :, 2:3 ]
	w = ( a2[ ok ] * 0.5 / np.maximum( np.bincount( sf, minlength = nf ), 1 ) )[ sf ] * hit
	g0 = at( r13( x0 ) )[ :, :3 ]; g1 = at( r13( x ) )[ :, :3 ]; y = y[ :, :3 ] # (the first look's)
	wall = ( np.abs( fn[ sf, 1 ] ) < 0.5 ) & ~ ( ( flags[ tf[ sf ] ] & 2 ) > 0 ).all( 1 )
	st.update( vars = K, fixed = int( fixed.sum() ), cg = its, moved = int( len( idx ) ),
		err = dict( all = errors( g0, y, w ) + errors( g1, y, w ), wall = errors( g0[ wall ], y[ wall ], w[ wall ] ) + errors( g1[ wall ], y[ wall ], w[ wall ] ) ) )
	write( idx, c16 ); done[ name ] = ( src, np.unique( tk ), idx, c16 )
	st.update( file = file, seconds = round( time.time() - t0, 1 ) )
	return st

def work( job ):

	import resource
	raw, chunks, key, entries, opt = job
	t0 = time.time(); ch = chunks[ key ]; tl = entries[ 0 ][ 'top' ]
	rec, tri, _ = read( raw, ch[ tl ] )
	top = Top( ch[ tl ], rec, tri ); del rec, tri
	pat = top.patterned( float( opt[ 'refit_turn' ] ) ) if float( opt[ 'refit_tint' ] ) > 0 else None
	cache = {}; done = {}
	levels = { e[ 'name' ]: fit_level( raw, chunks, e, top, pat, opt, cache, done ) for e in sorted( entries, key = lambda e: ( e[ 'src' ], e[ 'name' ] ) ) }
	return dict( key = key, top = tl, levels = levels, faces = len( top.T ), patterned = int( pat.sum() ) if pat is not None else 0, pairs = top.pairs,
		seconds = round( time.time() - t0, 1 ), rss = resource.getrusage( resource.RUSAGE_SELF ).ru_maxrss // 1024 )

def main( argv ):

	global np
	import numpy as np, multiprocessing
	raw = os.path.abspath( argv[ 0 ] ); opt = dict( OPT, **( json.loads( argv[ 1 ] ) if len( argv ) > 1 else {} ) )
	for k in ( 'refit_step', 'refit_reach' ):

		# (one number for every level, or some levels named: the rest as here)
		opt[ k ] = dict( OPT[ k ], **opt[ k ] ) if isinstance( opt[ k ], dict ) else { n: opt[ k ] for n in OPT[ k ] }

	chunks = { c[ 'k' ]: c for c in json.load( open( os.path.join( raw, 'raw.json' ) ) )[ 'chunks' ] }
	sj = os.path.join( raw, 'sim.json' )
	if not os.path.exists( sj ): sys.exit( 'lsvb_refit: no %s - the packer leaves it when run with refit: "sim" (see run())' % sj )
	sim = json.load( open( sj ) )
	want = str( opt[ 'refit_levels' ] ).split( ',' ); cells = {}
	for e in sim[ 'levels' ]:

		if e[ 'name' ] in want and ( not opt.get( 'refit_cells' ) or e[ 'k' ] in opt[ 'refit_cells' ] ): cells.setdefault( e[ 'k' ], [] ).append( e )

	for f in glob.glob( os.path.join( glob.escape( raw ), '*.fit.bin' ) ) + [ os.path.join( raw, 'fit.json' ) ]:

		if os.path.exists( f ): os.remove( f )

	keys = sorted( cells, key = lambda k: ( - chunks[ k ][ cells[ k ][ 0 ][ 'top' ] ][ 'ni' ], k ) ) # the big ones first
	jobs = max( 1, min( int( opt[ 'refit_jobs' ] ) or 4, len( keys ) ) )
	todo = [ ( raw, chunks, k, cells[ k ], opt ) for k in keys ]
	if jobs > 1:

		with multiprocessing.get_context( 'fork' ).Pool( jobs, maxtasksperchild = 1 ) as pool: res = list( pool.imap_unordered( work, todo, 1 ) )

	else: res = [ work( j ) for j in todo ]
	res.sort( key = lambda r: r[ 'key' ] )
	rss = sorted( ( r[ 'rss' ] for r in res ), reverse = True )
	tot = {}
	for r in res:

		for name, st in r[ 'levels' ].items():

			t = tot.setdefault( name, dict( cells = 0, same = 0, verts = 0, moved = 0, points = 0, found = 0, covered = 0, tint_faces = 0, lit = 0, seconds = 0.0, all = [ 0.0 ] * 6, wall = [ 0.0 ] * 6 ) )
			t[ 'cells' ] += 1; t[ 'same' ] += 'same_as' in st
			for k in ( 'verts', 'moved', 'points', 'found', 'covered', 'tint_faces', 'lit', 'seconds' ): t[ k ] += st.get( k, 0 )
			for k in ( 'all', 'wall' ): t[ k ] = [ a + b for a, b in zip( t[ k ], st[ 'err' ][ k ] ) ]

	for name in [ n for n in ( 'lo', 'mid', 'near', 'ring' ) if n in tot ]:

		t = tot[ name ]
		for k in ( 'all', 'wall' ):

			s = t[ k ]; t[ k ] = dict( rms = [ round( ( s[ 0 ] / max( s[ 2 ], 1e-30 ) ) ** 0.5, 2 ), round( ( s[ 3 ] / max( s[ 5 ], 1e-30 ) ) ** 0.5, 2 ) ], bias = [ round( s[ 1 ] / max( s[ 2 ], 1e-30 ), 2 ), round( s[ 4 ] / max( s[ 5 ], 1e-30 ), 2 ) ] )

		t[ 'seconds' ] = round( t[ 'seconds' ], 1 )
		if t[ 'same' ] == t[ 'cells' ]: log( '%-4s %d cells: %d of %d vertices moved - the values of near, whose faces these are' % ( name, t[ 'cells' ], t[ 'moved' ], t[ 'verts' ] ) ); continue
		log( '%-4s %d cells: %d of %d vertices moved (%d points, %.1f %% found the top level, %.1f %% under a covering face; %d of %d faces stay for their pattern). Against the top level, display steps rms: all %.1f -> %.1f, walls %.1f -> %.1f (bias %+.1f -> %+.1f)' % (
			name, t[ 'cells' ], t[ 'moved' ], t[ 'verts' ], t[ 'points' ], 100 * t[ 'found' ] / max( 1, t[ 'points' ] ), 100 * t[ 'covered' ] / max( 1, t[ 'points' ] ), t[ 'tint_faces' ], t[ 'lit' ],
			t[ 'all' ][ 'rms' ][ 0 ], t[ 'all' ][ 'rms' ][ 1 ], t[ 'wall' ][ 'rms' ][ 0 ], t[ 'wall' ][ 'rms' ][ 1 ], t[ 'wall' ][ 'bias' ][ 0 ], t[ 'wall' ][ 'bias' ][ 1 ] ) )

	info = dict( version = VERSION, settings = { k: opt[ k ] for k in OPT }, totals = tot, seconds = round( time.time() - T0, 1 ), jobs = jobs, peak_mb = sum( rss[ :jobs ] ),
		levels = { '%s.%s' % ( r[ 'key' ], name ): dict( file = st[ 'file' ], moved = st[ 'moved' ] ) for r in res for name, st in r[ 'levels' ].items() } )
	if opt.get( 'refit_report' ): info[ 'cells' ] = res
	fj = os.path.join( raw, 'fit.json' )
	json.dump( info, open( fj + '.tmp', 'w' ), separators = ( ',', ':' ) ); os.replace( fj + '.tmp', fj )
	if not opt.get( 'refit_keep' ):

		# (the index lists were for this: the fit files name them by their sums)
		for e in sim[ 'levels' ]:

			if os.path.exists( os.path.join( raw, e[ 'file' ] ) ): os.remove( os.path.join( raw, e[ 'file' ] ) )

		os.remove( sj )

	log( 'done: %d cells, %d at once, %.1fs, the largest %d of them together %d MB' % ( len( res ), jobs, time.time() - T0, jobs, info[ 'peak_mb' ] ) )

if __name__ == '__main__': main( sys.argv[ sys.argv.index( '--' ) + 1: ] if '--' in sys.argv else sys.argv[ 1: ] )
