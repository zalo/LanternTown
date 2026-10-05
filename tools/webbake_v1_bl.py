# Web bake worker: runs inside Blender's python (numpy + mathutils), driven by tools/webbake_v1.py.
#
#   blender --background --factory-startup --python tools/webbake_v1_bl.py -- <config.json>
#
# Reads out/town/town.json + the built GLBs (parsed directly, no bpy import) and writes the "baked" web build:
# the static town merged into spatial chunks (one vertex stream, a handful of index ranges), every vertex carrying
# its final lit colour (all lantern lights with BVH shadow rays + moon + sky ambient + ambient occlusion, weathering
# and tint folded in), plus small texture-array pages of pre-shaded albedo and a manifest with the navigation data.
#
# Nothing here names an asset or an item: the bake follows town.json, so it is re-run after every town rebuild.
#
# Coordinates: town.json is Blender's (X east, Y north, Z up); everything below is three's (x, z, -y), which is
# also what the GLBs hold.

import sys, os, re, json, struct, time, math, hashlib, zlib
import numpy as np

T0 = time.time()
CFG = json.load( open( sys.argv[ sys.argv.index( '--' ) + 1 ] ) )
ROOT = CFG[ 'root' ]
OUT = CFG[ 'out' ]
# work files (the maps the forked workers write their results to): on disk beside the output, never /tmp or /dev/shm
WORK = CFG.get( 'work' ) or os.path.join( OUT, '.work' )
os.makedirs( WORK, exist_ok = True )
JOBS = int( CFG.get( 'jobs', 8 ) )

def log( *a ):

	print( '[webbake %6.1fs]' % ( time.time() - T0 ), *a, flush = True )

# ------------------------------------------------------------------------------------------ look constants
# The look set up by web/town/main.js + web/engine/* for the walkable town (the "high" tier is the reference).
# tools/webbake_v1.py reads every one of these from those files (LOOK_SRC there) and passes them in, so the numbers below
# are only what a run without the driver gets; a change of the look needs a re-bake, nothing else.
LOOK = dict(
	lamp = 0.5, ambient = 1.3, moon = 2.0,
	sky = ( 0.33, 0.40, 0.60 ), ground = ( 0.16, 0.15, 0.2 ), moon_col = ( 0.62, 0.68, 0.87 ), moon_dir = ( - 0.45, 0.62, 0.35 ),
	source_radius = 0.6, ao_direct = 0.55,
	cap_up = 0.25, roof_fill = 2.5, # engine: CAP_UP of town.js (capped lanterns), roofFill (sky fill above the eaves)
	wet = 0.68, grime = 0.8, wear = 1.0, special = 1.0, macro = 1.0, soft = 1.0, window_glow = 0.75, emissive = 1.0,
	puddle_rough = 0.09, film_rough = 0.17, pool_cap = 1.0, tide_h = 0.4,
)
LOOK.update( CFG.get( 'look', {} ) )

# bake parameters
BK = dict(
	cell = 16.0, far_cell = 64.0, near_zone = 56.0, # chunk grid (m); items further than near_zone from the walkable area: coarse cells, base level only
	refine_k = 0.5, refine_min = 0.6, refine_emin = 0.08, refine_iters = 7, # tessellation of large faces near lanterns (edge <= k * distance)
	shadow_leak = 0.06, light_eps = 0.003, light_skip = 0.42, light_jitter = 0.14, light_rays = 3,
	ao_rays = 12, ao_radius = 3.0, ao_strength = 0.7,
	metal_diffuse = 0.35, # how much of the lit colour a metal keeps (stand-in for its specular)
	layer_min_area = 20.0, # m2 of surface below which a textured material is drawn in its mean colour instead of taking a texture layer
	cull_height = 7.0, # faces that cannot be seen from anywhere above the walkable area up to this height are dropped (0 = keep all)
	tex_sizes = [ 512, 256 ],
	prop_size = 2.5, # things smaller than this across (m) are flagged for the packer: they keep their shape in the lowest level
)
BK.update( CFG.get( 'bake', {} ) )

# ------------------------------------------------------------------------------------------ helpers

def lin( h ):

	h = h.lstrip( '#' )
	if len( h ) == 3: h = ''.join( c * 2 for c in h )
	c = np.array( [ int( h[ i:i + 2 ], 16 ) / 255 for i in ( 0, 2, 4 ) ], np.float64 )
	return np.where( c <= 0.04045, c / 12.92, ( ( c + 0.055 ) / 1.055 ) ** 2.4 )

def to_srgb( x ):

	x = np.clip( x, 0, 1 )
	return np.where( x <= 0.0031308, x * 12.92, 1.055 * np.power( x, 1 / 2.4 ) - 0.055 )

def from_srgb( c ):

	return np.where( c <= 0.04045, c / 12.92, ( ( c + 0.055 ) / 1.055 ) ** 2.4 )

def sstep( e0, e1, x ):

	t = np.clip( ( x - e0 ) / ( e1 - e0 ), 0, 1 )
	return t * t * ( 3 - 2 * t )

def mix( a, b, t ):

	return a + ( b - a ) * t

def lum( c ):

	return c[ ..., 0 ] * 0.2126 + c[ ..., 1 ] * 0.7152 + c[ ..., 2 ] * 0.0722

def make_noise( size = 256 ):

	# the engine's makeNoiseTexture (web/engine/materials.js), value for value: world-space weathering, pools and
	# macro variation then fall where the full renderer puts them
	data = np.zeros( ( size, size, 4 ), np.float64 )
	seed = 1234567
	for c in range( 4 ):

		acc = np.zeros( ( size, size ) )
		amp = 1.0
		cells = 8
		while cells <= 64:

			g = np.empty( cells * cells )
			for i in range( cells * cells ):

				seed = ( seed * 1664525 + 1013904223 ) & 0xffffffff
				g[ i ] = seed / 4294967296

			g = g.reshape( cells, cells )
			k = size / cells
			f = np.arange( size ) / k
			i0 = np.floor( f ).astype( int ); t = f - i0; s = t * t * ( 3 - 2 * t ); i1 = ( i0 + 1 ) % cells
			a = g[ np.ix_( i0, i0 ) ]; b = g[ np.ix_( i0, i1 ) ]; d = g[ np.ix_( i1, i0 ) ]; e = g[ np.ix_( i1, i1 ) ]
			sx = s[ None, : ]; sy = s[ :, None ]
			acc += amp * ( a + ( b - a ) * sx + ( d - a ) * sy + ( a - b - d + e ) * sx * sy )
			amp *= 0.5
			cells *= 2

		lo, hi = acc.min(), acc.max()
		data[ :, :, c ] = np.round( 255 * ( acc - lo ) / ( hi - lo ) ) / 255

	return data.astype( np.float32 )

NOISE = make_noise()

def noise_at( u, v ):

	# bilinear, repeat wrapping (rows = v)
	s = NOISE.shape[ 0 ]
	x = u * s - 0.5; y = v * s - 0.5
	x0 = np.floor( x ); y0 = np.floor( y )
	fx = ( x - x0 )[ :, None ].astype( np.float32 ); fy = ( y - y0 )[ :, None ].astype( np.float32 )
	x0 = x0.astype( np.int64 ) % s; y0 = y0.astype( np.int64 ) % s
	x1 = ( x0 + 1 ) % s; y1 = ( y0 + 1 ) % s
	a = NOISE[ y0, x0 ]; b = NOISE[ y0, x1 ]; c = NOISE[ y1, x0 ]; d = NOISE[ y1, x1 ]
	return ( a + ( b - a ) * fx ) * ( 1 - fy ) + ( c + ( d - c ) * fx ) * fy

# ------------------------------------------------------------------------------------------ GLB reader

CT = { 5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32 }
NC = { 'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16 }

def glb_json( path ):

	with open( path, 'rb' ) as f:

		f.read( 12 )
		cl, _ = struct.unpack( '<II', f.read( 8 ) )
		return json.loads( f.read( cl ) )

def node_matrix( nd ):

	if 'matrix' in nd: return np.array( nd[ 'matrix' ], np.float64 ).reshape( 4, 4 ).T
	m = np.eye( 4 )
	s = nd.get( 'scale', [ 1, 1, 1 ] ); q = nd.get( 'rotation', [ 0, 0, 0, 1 ] ); t = nd.get( 'translation', [ 0, 0, 0 ] )
	x, y, z, w = q
	r = np.array( [ [ 1 - 2 * ( y * y + z * z ), 2 * ( x * y - z * w ), 2 * ( x * z + y * w ) ],
		[ 2 * ( x * y + z * w ), 1 - 2 * ( x * x + z * z ), 2 * ( y * z - x * w ) ],
		[ 2 * ( x * z - y * w ), 2 * ( y * z + x * w ), 1 - 2 * ( x * x + y * y ) ] ] )
	m[ :3, :3 ] = r * np.array( s )[ None, : ]
	m[ :3, 3 ] = t
	return m

def glb_nodes( j ):

	# ( node index, world matrix ) for every node reachable from the scene
	out = []
	def walk( i, pm ):

		nd = j[ 'nodes' ][ i ]
		m = pm @ node_matrix( nd )
		out.append( ( i, m ) )
		for c in nd.get( 'children', [] ): walk( c, m )

	sc = j[ 'scenes' ][ j.get( 'scene', 0 ) ]
	for i in sc[ 'nodes' ]: walk( i, np.eye( 4 ) )
	return out

_GLB = {}

def load_glb( path ):

	# -> list of primitives in asset space: dict( P, N, UV, MASK, I, lib, tint, params )
	if path in _GLB: return _GLB[ path ]
	b = open( path, 'rb' ).read()
	cl = struct.unpack_from( '<I', b, 12 )[ 0 ]
	j = json.loads( b[ 20:20 + cl ] )
	off = 20 + cl + 8
	def acc( i ):

		a = j[ 'accessors' ][ i ]; bv = j[ 'bufferViews' ][ a[ 'bufferView' ] ]
		n = NC[ a[ 'type' ] ]; dt = np.dtype( CT[ a[ 'componentType' ] ] )
		o = off + bv.get( 'byteOffset', 0 ) + a.get( 'byteOffset', 0 )
		st = bv.get( 'byteStride', 0 )
		if st and st != n * dt.itemsize:

			raw = np.frombuffer( b, np.uint8, st * a[ 'count' ], o ).reshape( a[ 'count' ], st )[ :, :n * dt.itemsize ]
			arr = np.ascontiguousarray( raw ).view( dt )

		else: arr = np.frombuffer( b, dt, a[ 'count' ] * n, o )
		if a.get( 'normalized' ) and dt.kind in 'iu': arr = arr.astype( np.float32 ) / np.iinfo( dt ).max
		return arr.reshape( a[ 'count' ], n ) if n > 1 else arr

	prims = []
	for ni, m in glb_nodes( j ):

		nd = j[ 'nodes' ][ ni ]
		if 'mesh' not in nd: continue
		ident = np.allclose( m, np.eye( 4 ) )
		nm = np.linalg.inv( m[ :3, :3 ] ).T
		flip = np.linalg.det( m[ :3, :3 ] ) < 0
		for pr in j[ 'meshes' ][ nd[ 'mesh' ] ][ 'primitives' ]:

			if pr.get( 'mode', 4 ) != 4 or 'POSITION' not in pr[ 'attributes' ]: continue
			at = pr[ 'attributes' ]
			P = acc( at[ 'POSITION' ] ).astype( np.float32 )
			n = len( P )
			if 'indices' in pr: I = acc( pr[ 'indices' ] ).astype( np.int32 ).reshape( - 1, 3 )
			else: I = np.arange( n, dtype = np.int32 ).reshape( - 1, 3 )
			if 'NORMAL' in at: N = acc( at[ 'NORMAL' ] ).astype( np.float32 )
			else:

				fn = np.cross( P[ I[ :, 1 ] ] - P[ I[ :, 0 ] ], P[ I[ :, 2 ] ] - P[ I[ :, 0 ] ] )
				N = np.zeros_like( P )
				for k in range( 3 ): np.add.at( N, I[ :, k ], fn )
				N /= np.maximum( np.linalg.norm( N, axis = 1, keepdims = True ), 1e-12 )

			UV = acc( at[ 'TEXCOORD_0' ] ).astype( np.float32 ) if 'TEXCOORD_0' in at else np.zeros( ( n, 2 ), np.float32 )
			MASK = acc( at[ '_MASK' ] ).astype( np.float32 ) if '_MASK' in at else np.tile( np.array( [ 1, 0, 0, 0 ], np.float32 ), ( n, 1 ) )
			if not ident:

				P = ( P.astype( np.float64 ) @ m[ :3, :3 ].T + m[ :3, 3 ] ).astype( np.float32 )
				N = N.astype( np.float64 ) @ nm.T
				N = ( N / np.maximum( np.linalg.norm( N, axis = 1, keepdims = True ), 1e-12 ) ).astype( np.float32 )
				if flip: I = I[ :, ::- 1 ]

			mat = j[ 'materials' ][ pr[ 'material' ] ] if 'material' in pr else {}
			ex = mat.get( 'extras', {} )
			params = ex.get( 'tsq_params' ) or {}
			if isinstance( params, str ):

				try: params = json.loads( params )
				except Exception: params = {}

			prims.append( dict( P = P, N = N, UV = UV, MASK = MASK, I = I, lib = ex.get( 'tsq_lib' ) or 'plain', tint = ex.get( 'tsq_tint' ), params = params ) )

	_GLB[ path ] = prims
	return prims

def glb_sockets( path ):

	j = glb_json( path )
	out = []
	for ni, m in glb_nodes( j ):

		ex = j[ 'nodes' ][ ni ].get( 'extras', {} )
		if ex.get( 'tsq_socket' ):

			pr = ex.get( 'tsq_props' ) or {}
			if isinstance( pr, str ):

				try: pr = json.loads( pr )
				except Exception: pr = {}

			out.append( dict( kind = ex[ 'tsq_socket' ], name = ex.get( 'tsq_name', '' ), props = pr, pos = m[ :3, 3 ].copy() ) )

	return out

# ------------------------------------------------------------------------------------------ town + materials

TOWN_PATH = CFG.get( 'town' ) or os.path.join( ROOT, 'out/town/town.json' ) # tools/webbake_v1.py --town
TOWN_RAW = open( TOWN_PATH, 'rb' ).read()
TOWN = json.loads( TOWN_RAW )
CATALOG = { m[ 'name' ]: m for m in json.load( open( os.path.join( ROOT, 'docs/materials.json' ) ) )[ 'materials' ] }
INFO = {}
for name, e in CATALOG.items():

	p = os.path.join( ROOT, 'out/mat', name, 'mat.json' )
	INFO[ name ] = json.load( open( p ) ) if e.get( 'kind' ) == 'baked' and os.path.exists( p ) else None

SPECIAL = {
	'plain': dict( tint = '#808080', rough = 0.7 ), 'metal_plain': dict( tint = '#c9c9c9', rough = 0.35 ),
	'emissive': dict( tint = '#ffb45a', glow = 8 ), 'neon': dict( tint = '#ffb45a', glow = 14 ),
	'glass_frosted': dict( tint = '#efe6d2', glow = 3, rough = 0.5 ), 'glass_clear': dict( tint = '#b9c9c2', opacity = 0.16 ),
	'glass_lantern': dict( tint = '#ffdca8', glow = 0.5, opacity = 0.22 ), 'glass_stained': dict( tint = '#c89a3c', glow = 1.6 ),
	'window_dark': dict( tint = '#ffc27a' ), 'window_lit': dict( tint = '#ffc27a', glow = 2.4 ),
	'water': dict( tint = '#0b1512' ), 'water_fall': dict( tint = '#cfe0e6', opacity = 0.7 ),
}
NO_OCCLUDE = { 'emissive', 'neon', 'glass_frosted', 'glass_clear', 'glass_lantern', 'water_fall' }
K_OPAQUE, K_GLASS, K_WATER, K_FALL, K_CUT = 0, 1, 2, 3, 4 # K_CUT: leaf cards, cut out by the layer's relief height
L_WHITE, L_WINDOW = 0, 1
WIN_CELL = ( 2.9, 2.7 ) # metres of wall one fake room of the window layer covers (the layer holds 2 x 2 rooms)

REGIONS = TOWN.get( 'regions', [] )
WATER_Z = TOWN.get( 'frame', {} ).get( 'water_z' )
TIDE_Y = WATER_Z if isinstance( WATER_Z, ( int, float ) ) else - 1e4

def b2t( p ):

	return np.array( [ p[ 0 ], p[ 2 ] if len( p ) > 2 else 0.0, - p[ 1 ] ], np.float64 )

def region_dist( x, z ):

	# distance (m) of three-space points (x, z) to the walkable area
	d = np.full( np.shape( x ), 1e9 )
	for r in REGIONS:

		x0, y0, x1, y1 = r[ 'rect' ]
		dx = np.maximum( np.maximum( x0 - x, x - x1 ), 0 ); dz = np.maximum( np.maximum( - y1 - z, z + y0 ), 0 )
		d = np.minimum( d, np.hypot( dx, dz ) )

	return d

def cell_ids( x, z ):

	# chunk cell of points: near cells are even ids, far cells odd (two interleaved grids)
	near = region_dist( x, z ) <= BK[ 'near_zone' ]
	c = np.where( near, BK[ 'cell' ], BK[ 'far_cell' ] )
	ix = np.floor( x / c ).astype( np.int64 ) + 512; iz = np.floor( z / c ).astype( np.int64 ) + 512
	return ( ( ix * 1024 + iz ) * 2 + np.where( near, 0, 1 ) ).astype( np.int64 )

def cell_key( cid ):

	far = cid & 1; q = cid >> 1
	return '%s_%d_%d' % ( 'f' if far else 'c', ( q // 1024 ) - 512, ( q % 1024 ) - 512 )

# ------------------------------------------------------------------------------------------ texture layers

class Layers:

	def __init__( self ):

		self.names = [ 'white', 'window' ]
		self.index = { 'white': L_WHITE, 'window': L_WINDOW }
		self.stripes = {} # key -> dict( layer, c1, c2, cmax )
		self.libs = set()

	def lib( self, name ):

		return self.index.get( name, L_WHITE )

	def add_lib( self, name ):

		if name not in self.index:

			self.index[ name ] = len( self.names ); self.names.append( name ); self.libs.add( name )

	def stripe( self, c1, c2 ):

		key = 'stripe:' + ','.join( '%.3f' % v for v in list( c1 ) + list( c2 ) )
		s = self.stripes.get( key )
		if not s:

			cmax = np.maximum( np.maximum( c1, c2 ), 1e-4 )
			s = self.stripes[ key ] = dict( layer = len( self.names ), c1 = np.array( c1 ), c2 = np.array( c2 ), cmax = cmax )
			self.index[ key ] = s[ 'layer' ]; self.names.append( key )

		return s

LAYERS = Layers()

# ------------------------------------------------------------------------------------------ vertex shading inputs

def weather( entry, alb, rough, metal, mask, nLo, wp, ny ):

	# vertex-rate version of MaterialLib._weather with the relief height at its mean (0.5) and the fine noise at its
	# mean; the metre-scale noise is the real one. alb: (n,3) linear. Returns alb, rough, metal.
	g = entry.get( 'grime' ); w = entry.get( 'wear' ); sp = entry.get( 'special' )
	WET = LOOK[ 'wet' ]
	n = len( alb )
	one = np.ones( n, np.float32 )

	def pools( alb, rough ):

		q0 = wp[ :, 0 ] + wp[ :, 1 ] * 0.37; q1 = wp[ :, 2 ] + wp[ :, 1 ] * 0.61
		nP = noise_at( q0 * 0.031 + 0.53, q1 * 0.031 + 0.17 )
		zone = np.minimum( ( nP[ :, 0 ] + nP[ :, 2 ] * 0.35 - 0.8 ) * 5 + ( WET - 0.5 ), LOOK[ 'pool_cap' ] )
		level = zone + ( nLo[ :, 3 ] - 0.5 ) * 2.4 - 0.25
		lvl = sstep( 0.985, 0.998, ny ) * sstep( 0.05, 0.3, WET )
		depth = level - 0.5 * 0.8
		water = sstep( 0.0, 0.5, depth ) * lvl
		halo = sstep( - 1.0, 0.1, depth ) * lvl
		alb = alb * ( mix( 1, 0.78, halo ) * mix( 1, 0.62, water ) )[ :, None ]
		rough = mix( rough * mix( 1, 0.7, halo ), LOOK[ 'puddle_rough' ], water )
		return alb, rough

	if w:

		amt = mask[ :, 2 ] * LOOK[ 'wear' ]
		k = sstep( 0.35, 0.8, amt * 1.25 ) * sstep( 0.02, 0.2, amt )
		wc = lin( w[ 'color' ] )[ None, : ]
		wk = mix( 1, 0.45, LOOK[ 'soft' ] )
		hi = np.minimum( alb * 1.6, np.maximum( wc, alb ) ) if w.get( 'keep_hue' ) else mix( wc, np.minimum( alb * 1.5, np.maximum( wc, alb ) ), LOOK[ 'soft' ] * 0.6 )
		alb = mix( alb, hi, ( k * w.get( 'amount', 0.85 ) * wk )[ :, None ] )
		rough = mix( rough, w.get( 'rough', 0.5 ), k * 0.9 * wk )
		metal = mix( metal, w.get( 'metal', 0 ) or 0, k )

	wet_kind = bool( sp and sp.get( 'kind' ) == 'wet' )
	if sp and sp.get( 'kind' ) and sp[ 'kind' ] != 'none':

		amt = mask[ :, 3 ] * LOOK[ 'special' ]
		sc = lin( sp.get( 'color', '#000000' ) )[ None, : ]
		if sp[ 'kind' ] == 'wet':

			amt = np.clip( amt + WET, 0, 1 )
			k = sstep( 0.15, 0.75, amt + ( nLo[ :, 1 ] - 0.5 ) * 0.35 ) * sstep( 0.02, 0.2, amt )
			puddle = k * sstep( 0.55, 0.2, 0.5 + ( 1 - amt ) * 0.4 )
			alb = alb * mix( 1, 0.5, k )[ :, None ]
			rough = mix( mix( rough, rough * 0.45, k ), sp.get( 'rough', 0.08 ), puddle )
			alb, rough = pools( alb, rough )

		elif sp[ 'kind' ] == 'fade':

			k = sstep( 0.2, 0.8, amt + ( nLo[ :, 2 ] - 0.5 ) * 0.4 ) * sstep( 0.02, 0.2, amt )
			l = lum( alb )[ :, None ]
			alb = mix( alb, mix( l, sc, 0.25 ) * 1.25 + 0.02, ( k * 0.65 )[ :, None ] )
			rough = mix( rough, sp.get( 'rough', 0.9 ), k * 0.6 )

		else:

			k = sstep( 0.42, 0.62, amt * 1.15 + ( nLo[ :, 0 ] - 0.5 ) * 0.45 ) * sstep( 0.02, 0.2, amt )
			alb = mix( alb, sc * 1.0, ( k * 0.92 )[ :, None ] )
			rough = mix( rough, sp.get( 'rough', 0.9 ), k )
			metal = metal * ( 1 - k )

	if not wet_kind:

		amt = WET * sstep( 0.5, 0.92, ny )
		k = sstep( 0.15, 0.75, amt + ( nLo[ :, 1 ] - 0.5 ) * 0.35 ) * sstep( 0.02, 0.2, amt )
		puddle = k * sstep( 0.55, 0.2, 0.5 + ( 1 - amt ) * 0.4 ) * sstep( 0.97, 0.995, ny )
		film = k * sstep( 0.9, 0.99, ny ) * sstep( 0.3, 0.75, nLo[ :, 2 ] * 0.7 + 0.15 + ( WET - 0.5 ) )
		alb = alb * mix( 1, 0.58, k )[ :, None ]
		rough = mix( mix( mix( rough, rough * mix( 0.72, 0.45, sstep( 0.85, 0.98, ny ) ), k ), np.minimum( rough, LOOK[ 'film_rough' ] ), film * 0.85 ), 0.07, puddle )
		alb, rough = pools( alb, rough )

	if g:

		amt = mask[ :, 1 ] * LOOK[ 'grime' ]
		wall = sstep( 0.35, 0.8, 1 - np.abs( ny ) )
		streak = noise_at( ( wp[ :, 0 ] * 1.1 + wp[ :, 2 ] * 0.9 ) * 0.45, wp[ :, 1 ] * 0.035 )[ :, 1 ]
		k0 = sstep( 0.3, 0.85, amt * 1.3 + ( nLo[ :, 3 ] - 0.5 ) * 0.3 )
		k1 = sstep( 0.35, 0.9, amt * 1.1 + ( 1 - mask[ :, 0 ] ) * 0.5 + ( streak - 0.5 ) * 0.55 * wall + ( nLo[ :, 3 ] - 0.5 ) * 0.12 * ( 1 - wall ) )
		k = mix( k0, k1, LOOK[ 'soft' ] ) * sstep( 0.02, 0.2, amt )
		gc = lin( g[ 'color' ] )[ None, : ]
		dirty = mix( mix( alb * gc * 2.2, gc, 0.45 ), alb * 0.42 + gc * 0.6, LOOK[ 'soft' ] )
		alb = mix( alb, dirty, ( k * mix( 0.85, 0.7, LOOK[ 'soft' ] ) )[ :, None ] )
		rough = mix( rough, g.get( 'rough', 0.9 ), k * 0.85 )
		metal = metal * ( 1 - k * 0.8 )

	top = TIDE_Y + LOOK[ 'tide_h' ] * ( nLo[ :, 2 ] * 0.9 + 0.15 + 0.35 )
	band = sstep( top, TIDE_Y + 0.02, wp[ :, 1 ] )
	alb = mix( alb, alb * np.array( [ 0.36, 0.42, 0.32 ] )[ None, : ], ( band * 0.9 )[ :, None ] )
	rough = mix( rough, np.minimum( rough, 0.3 ), band * 0.8 )
	return alb, rough, metal

def shade_prim( pr, wp, wn ):

	# everything about a primitive instance that does not depend on the lighting.
	# -> dict( kind, occ, double, fol, UV, MUL (albedo multiplier of the layer texture), GLOW (emissive, in units of MUL),
	#          EM (absolute emissive), VAO, AUX (gloss, or opacity for blended kinds), EMF (1 = glowing surface), LAY ) or None
	lib = pr[ 'lib' ] if pr[ 'lib' ] in CATALOG else 'plain'
	entry = CATALOG[ lib ]
	params = pr[ 'params' ]
	n = len( wp )
	mask = pr[ 'MASK' ]
	tint = np.array( pr[ 'tint' ][ :3 ], np.float64 ) if pr[ 'tint' ] is not None else None
	baked = entry.get( 'kind' ) == 'baked'
	info = INFO.get( lib ) if baked else None
	scale = float( params.get( 'scale' ) or 1 )
	uvm = np.stack( [ pr[ 'UV' ][ :, 0 ], 1 - pr[ 'UV' ][ :, 1 ] ], 1 ).astype( np.float64 ) / scale # metres, V up
	q0 = wp[ :, 0 ] + wp[ :, 1 ] * 0.37; q1 = wp[ :, 2 ] + wp[ :, 1 ] * 0.61
	nLo = noise_at( q0 * 0.11, q1 * 0.11 )
	ny = wn[ :, 1 ]
	vao = mask[ :, 0 ].astype( np.float32 )
	o = dict( kind = K_OPAQUE, occ = lib not in NO_OCCLUDE, double = bool( params.get( 'double' ) ), fol = False,
		UV = np.zeros( ( n, 2 ), np.float32 ), GLOW = np.zeros( n, np.float32 ), EM = np.zeros( ( n, 3 ), np.float32 ), VAO = vao,
		AUX = np.zeros( n, np.float32 ), EMF = np.zeros( n, np.float32 ), LAY = L_WHITE )
	glow_p = params.get( 'glow' )

	def finish( alb, rough, metal, mean = None ):

		alb = np.maximum( alb, 0 )
		md = ( 1 - metal ) + BK[ 'metal_diffuse' ] * metal
		mul = alb * np.reshape( md, ( - 1, 1 ) )
		if mean is not None: mul = mul / mean[ None, : ]
		o[ 'MUL' ] = mul.astype( np.float32 )
		o[ 'AUX' ] = ( sstep( 0.5, 0.06, np.clip( rough, 0.04, 1 ) ) * np.ones( n ) ).astype( np.float32 )

	if baked and info:

		mean = np.maximum( np.array( info[ 'preview' ][ :3 ], np.float64 ), 0.004 )
		col = np.minimum( 6, tint / mean ) if ( tint is not None and entry.get( 'tintable' ) ) else np.ones( 3 )
		tile = info.get( 'tile', [ 1, 1 ] )
		alb0 = mean * col
		layered = lib in LAYERS.index
		stripe = None
		if lib == 'canvas' and params.get( 'stripe' ) and entry.get( 'tintable' ):

			c2 = lin( params[ 'tint2' ] ) if params.get( 'tint2' ) else lin( '#d9d2c0' )
			f2 = np.minimum( 6, c2 / mean )
			stripe = LAYERS.stripe( mean * col, mean * f2 )
			alb0 = 0.5 * ( mean * col + mean * f2 )

		# foliage as in MaterialLib._baked: leaf cards (`leafy`) wear one of two leaf colours (mask.w) and darken towards
		# the heart of the crown (mask.z); a green far above what a leaf reflects is pulled back onto a leaf's range
		foliage = lib in ( 'leaf', 'hedge' )
		leafy = float( params.get( 'leafy' ) or 0 ) if foliage else 0.0
		went = entry
		alb = np.tile( alb0, ( n, 1 ) )
		if leafy:

			went = dict( entry, wear = None, special = None )
			if params.get( 'tint2' ):

				f2 = np.minimum( 6, lin( params[ 'tint2' ] ) / mean )
				k2 = sstep( 0.38, 0.62, mask[ :, 3 ] * 0.75 + 0.125 )
				alb = mix( alb, ( mean * f2 )[ None, : ], k2[ :, None ] )

			alb = alb * mix( 1, 0.4, mask[ :, 2 ] )[ :, None ]

		if foliage:

			l = lum( alb ) + 0.0005
			green = sstep( 0.0, 0.25, ( alb[ :, 1 ] - np.maximum( alb[ :, 0 ], alb[ :, 2 ] ) ) / l )
			kg = np.minimum( 1, ( np.maximum( l - 0.1, 0 ) * 0.15 + 0.1 ) / l )
			alb = mix( alb, mix( alb, l[ :, None ], ( sstep( 0.08, 0.3, l ) * 0.45 )[ :, None ] ) * kg[ :, None ], green[ :, None ] )

		macro = 1 + ( ( nLo[ :, 0 ] - 0.5 ) * 0.3 + ( nLo[ :, 2 ] - 0.5 ) * 0.14 ) * LOOK[ 'macro' ]
		alb = alb * macro[ :, None ]
		rough = info.get( 'mean_rough', 0.7 ) * ( 1 + ( nLo[ :, 1 ] - 0.5 ) * 0.16 * LOOK[ 'macro' ] )
		metal = np.full( n, float( info.get( 'mean_metal', 0 ) ) )
		alb, rough, metal = weather( went, alb, rough, metal, mask, nLo, wp, ny )
		if stripe:

			finish( alb * ( stripe[ 'cmax' ] / np.maximum( alb0, 1e-4 ) )[ None, : ], rough, metal )
			o[ 'LAY' ] = stripe[ 'layer' ]
			o[ 'UV' ][ :, 0 ] = uvm[ :, 0 ] / float( params[ 'stripe' ] )
			o[ 'UV' ][ :, 1 ] = - uvm[ :, 1 ] / tile[ 1 ]

		elif layered:

			finish( alb, rough, metal, mean )
			o[ 'LAY' ] = LAYERS.index[ lib ]
			if leafy:

				# one card = one cell of the uv: the uv is stored in cells (the cut-out needs the cell), the factor from
				# cells to texture tiles rides in aux. One copy, drawn two-sided, lit as a spray of blades at every angle.
				o[ 'kind' ] = K_CUT; o[ 'double' ] = False; o[ 'fol' ] = True
				o[ 'UV' ][ :, 0 ] = uvm[ :, 0 ] * scale / leafy
				o[ 'UV' ][ :, 1 ] = - uvm[ :, 1 ] * scale / leafy
				o[ 'AUX' ][ : ] = min( 1.0, leafy / ( scale * tile[ 0 ] ) / CUT_AUX )

			else:

				o[ 'UV' ][ :, 0 ] = uvm[ :, 0 ] / tile[ 0 ]
				o[ 'UV' ][ :, 1 ] = - uvm[ :, 1 ] / tile[ 1 ]

		else: finish( alb, rough, metal )
		if glow_p: o[ 'GLOW' ][ : ] = float( glow_p ) * LOOK[ 'emissive' ]; o[ 'EMF' ][ : ] = 1 if float( glow_p ) > 0.25 else 0
		if glow_p and float( glow_p ) > 0: o[ 'occ' ] = False
		return o

	name = lib if not baked else 'plain' # a library material that is not baked yet is drawn as its flat stand-in
	d = SPECIAL.get( name, SPECIAL[ 'plain' ] )
	T = tint if tint is not None else ( np.array( [ 0.3, 0.3, 0.3 ] ) if baked else lin( d[ 'tint' ] ) )
	glow = float( params.get( 'glow', d.get( 'glow', 0 ) ) if 'glow' in d else ( glow_p or 0 ) )
	c_rough = float( params.get( 'rough', d.get( 'rough', 0.7 ) ) )
	opacity = float( params.get( 'opacity', d.get( 'opacity', 1 ) ) )
	dirt_lo = ( nLo[ :, 0 ] - 0.5 ) * 0.25

	if name in ( 'plain', 'metal_plain' ):

		metal0 = float( params.get( 'metal', 1 if name == 'metal_plain' else 0 ) )
		alb = T[ None, : ] * ( 1 + ( nLo[ :, 0 ] - 0.5 ) * 0.12 )[ :, None ]
		rough = c_rough + ( nLo[ :, 1 ] - 0.5 ) * 0.1
		alb, rough, metal = weather( entry, alb, rough, np.full( n, metal0 ), mask, nLo, wp, ny )
		finish( alb, rough, metal )
		if glow: o[ 'GLOW' ][ : ] = glow * LOOK[ 'emissive' ]; o[ 'EMF' ][ : ] = 1 if glow > 0.25 else 0; o[ 'occ' ] = False

	elif name in ( 'emissive', 'neon' ):

		e = T * glow * LOOK[ 'emissive' ]
		if name == 'neon': e = mix( e, np.full( 3, glow * LOOK[ 'emissive' ] ), 0.2 )
		o[ 'MUL' ] = np.zeros( ( n, 3 ), np.float32 )
		o[ 'EM' ][ : ] = e
		o[ 'EMF' ][ : ] = 1

	elif name == 'glass_frosted':

		o[ 'MUL' ] = ( T[ None, : ] * 0.8 * mix( 1, vao, 0.4 )[ :, None ] ).astype( np.float32 )
		dirt = ( 1 - mask[ :, 1 ] * 0.7 ) * 0.975
		o[ 'EM' ][ : ] = T[ None, : ] * ( glow * 0.75 * LOOK[ 'emissive' ] ) * dirt[ :, None ]
		o[ 'EMF' ][ : ] = 1 if glow > 0.25 else 0

	elif name in ( 'glass_clear', 'glass_lantern' ):

		lantern = name == 'glass_lantern'
		o[ 'two' ] = True
		dirt = np.clip( mask[ :, 1 ] * 0.8 + dirt_lo, 0, 1 )
		colr = mix( ( T * ( 0.02 if lantern else 0.06 ) )[ None, : ], np.array( [ 0.12, 0.11, 0.09 ] )[ None, : ], ( dirt * ( 0.35 if lantern else 1 ) )[ :, None ] )
		alpha = mix( opacity, 0.9, np.maximum( 0.06, dirt * 0.7 ) )
		if alpha.max() < 0.09 and not ( lantern and glow > 0.05 ): return None # a clean, clear pane: nothing to draw
		o[ 'kind' ] = K_GLASS
		o[ 'MUL' ] = colr.astype( np.float32 )
		# the flame's glare in the panes (as in the engine, for a pane seen at 60 degrees; the pane is blended, so the
		# glow is divided by its cover): a lit lantern reads as a lamp, not as a candle in a clear box
		if lantern: o[ 'EM' ][ : ] = ( T * np.array( [ 1.0, 0.76, 0.46 ] ) )[ None, : ] * glow * LOOK[ 'emissive' ] * ( ( dirt * 0.6 + 1.05 ) * ( nLo[ :, 1 ] * 0.5 + 0.75 ) / np.maximum( alpha, 0.05 ) )[ :, None ]
		o[ 'AUX' ] = alpha.astype( np.float32 )

	elif name == 'glass_stained':

		accent = lin( params.get( 'accent' ) or '#7a1f1a' )
		pane = 0.82 * T * 0.9 + 0.18 * accent
		o[ 'MUL' ] = np.tile( ( pane * 0.2 * 0.85 ).astype( np.float32 ), ( n, 1 ) )
		o[ 'EM' ][ : ] = ( pane * 0.85 * glow * LOOK[ 'window_glow' ] )[ None, : ] * ( nLo[ :, 1 ] * 0.6 + 0.7 )[ :, None ]
		o[ 'AUX' ][ : ] = 0.5
		o[ 'EMF' ][ : ] = 1 if glow > 0.25 else 0

	elif name in ( 'window_dark', 'window_lit' ):

		dirt = np.clip( mask[ :, 1 ] * 0.8, 0, 1 )
		o[ 'MUL' ] = mix( np.array( [ 0.012, 0.014, 0.018 ] )[ None, : ], np.array( [ 0.07, 0.065, 0.055 ] )[ None, : ], dirt[ :, None ] ).astype( np.float32 )
		o[ 'AUX' ] = sstep( 0.5, 0.06, mix( 0.06, 0.5, dirt ) ).astype( np.float32 )
		if name == 'window_lit':

			seed = mask[ :, 3 ].astype( np.float64 )
			level = mix( 0.55, 1.25, np.mod( seed * 13.37, 1 ) )
			# the window layer holds the fake room (normalised, see window_layer()); MUL is not used for it, so the
			# emissive goes through GLOW with MUL as the light colour and the dark glass is dropped
			o[ 'MUL' ] = ( T[ None, : ] * ( glow * level * ( 1 - dirt * 0.7 ) * LOOK[ 'window_glow' ] * WINDOW_GAIN )[ :, None ] ).astype( np.float32 )
			o[ 'GLOW' ][ : ] = 1
			o[ 'LIT0' ] = True # no lighting on top of the emissive
			o[ 'LAY' ] = L_WINDOW
			o[ 'UV' ][ :, 0 ] = uvm[ :, 0 ] / ( WIN_CELL[ 0 ] * 2 ) + np.floor( seed * 7.0 ) * 0.5
			o[ 'UV' ][ :, 1 ] = - uvm[ :, 1 ] / ( WIN_CELL[ 1 ] * 2 ) + np.floor( np.mod( seed * 5.3, 1 ) * 2 ) * 0.5
			o[ 'EMF' ][ : ] = 1
			o[ 'AUX' ][ : ] = 0

	elif name == 'water':

		o[ 'kind' ] = K_WATER
		o[ 'MUL' ] = np.tile( ( T * 0.6 ).astype( np.float32 ), ( n, 1 ) )
		o[ 'AUX' ][ : ] = 1

	elif name == 'water_fall':

		o[ 'kind' ] = K_FALL
		o[ 'occ' ] = False
		o[ 'MUL' ] = np.tile( T.astype( np.float32 ), ( n, 1 ) )
		o[ 'GLOW' ][ : ] = ( glow + 0.12 ) * LOOK[ 'emissive' ] * 0.3
		o[ 'AUX' ] = ( np.full( n, opacity ) ).astype( np.float32 )
		# break-up and foam hints ride in the uv's fractional precision budget: the fall shader only needs metres
		o[ 'UV' ][ : ] = uvm
		o[ 'UV' ][ :, 1 ] *= - 1

	else:

		alb, rough, metal = weather( entry, np.tile( T, ( n, 1 ) ), np.full( n, c_rough ), np.zeros( n ), mask, nLo, wp, ny )
		finish( alb, rough, metal )

	return o

CUT_AUX = 4.0 # leaf cards: aux = ( texture tiles per uv cell ) / CUT_AUX
WINDOW_GAIN = 1.6 # the window layer stores room / WINDOW_GAIN
UV_RANGE = 32.0 # uv (in tiles) is stored as u16 over [-32, 32)
UV_SPLIT = 40.0; UV_BLOCK = 16.0

# ------------------------------------------------------------------------------------------ soup

class Soup:

	# all static geometry of one level, concatenated. Per vertex: P N UV MUL GLOW EM VAO AUX EMF LAY GRP FOL LIT0;
	# per triangle: TRI KIND OCC CELL
	VF = [ ( 'P', 3, np.float32 ), ( 'N', 3, np.float32 ), ( 'UV', 2, np.float32 ), ( 'MUL', 3, np.float32 ), ( 'GLOW', 1, np.float32 ),
		( 'EM', 3, np.float32 ), ( 'VAO', 1, np.float32 ), ( 'AUX', 1, np.float32 ), ( 'EMF', 1, np.float32 ), ( 'LAY', 1, np.uint16 ),
		( 'GRP', 1, np.int32 ), ( 'FOL', 1, np.uint8 ), ( 'LIT0', 1, np.uint8 ), ( 'PROP', 1, np.uint8 ) ]

	def __init__( self ):

		self.v = { k: [] for k, _, _ in self.VF }
		self.t = dict( TRI = [], KIND = [], OCC = [], CELL = [], TERR = [] )
		self.nv = 0
		self.ngrp = 0
		self.area = {}

	def add( self, pr, wp, wn, cell, terrain = False, prop = False ):

		# pr: primitive in asset space (for uv / mask / material), wp / wn: its world positions / normals,
		# cell: one id for the whole primitive or None = by triangle centroid; prop: part of a small thing (see BK prop_size)
		o = shade_prim( pr, wp, wn )
		if o is None: return
		n = len( wp )
		I = pr[ 'I' ]
		if cell is None:

			c = ( wp[ I[ :, 0 ] ] + wp[ I[ :, 1 ] ] + wp[ I[ :, 2 ] ] ) / 3
			tc = cell_ids( c[ :, 0 ], c[ :, 2 ] )

		else: tc = np.full( len( I ), cell, np.int64 )
		grp = np.zeros( n, np.int64 ); ngrp = 1
		if np.ptp( o[ 'UV' ], axis = 0 ).max() > UV_SPLIT:

			# a surface that runs over more tiles than the stored uv range: cut into blocks, each with its own offset
			c = I.ravel()
			for k in ( 'UV', 'MUL', 'GLOW', 'EM', 'VAO', 'AUX', 'EMF' ): o[ k ] = o[ k ][ c ]
			wp = wp[ c ]; wn = wn[ c ]; n = len( c )
			I = np.arange( n ).reshape( - 1, 3 )
			blk = np.floor( o[ 'UV' ].reshape( - 1, 3, 2 ).mean( 1 ) / UV_BLOCK ).astype( np.int64 )
			u, gi = np.unique( blk[ :, 0 ] * 100003 + blk[ :, 1 ], return_inverse = True )
			grp = np.repeat( gi, 3 ); ngrp = len( u )

		sides = [ ( wn, I, o[ 'occ' ] ) ]
		if o[ 'double' ] and o[ 'kind' ] == K_OPAQUE: sides.append( ( - wn, I[ :, ::- 1 ], False ) )
		for nn, ii, occ in sides:

			v = self.v
			v[ 'P' ].append( wp ); v[ 'N' ].append( nn.astype( np.float32 ) ); v[ 'UV' ].append( o[ 'UV' ] ); v[ 'MUL' ].append( o[ 'MUL' ] )
			v[ 'GLOW' ].append( o[ 'GLOW' ] ); v[ 'EM' ].append( o[ 'EM' ] ); v[ 'VAO' ].append( o[ 'VAO' ] ); v[ 'AUX' ].append( o[ 'AUX' ] )
			v[ 'EMF' ].append( o[ 'EMF' ] ); v[ 'LAY' ].append( np.full( n, o[ 'LAY' ], np.uint16 ) ); v[ 'GRP' ].append( ( grp + self.ngrp ).astype( np.int32 ) )
			v[ 'FOL' ].append( np.full( n, 1 if o[ 'fol' ] else 2 if o.get( 'two' ) else 0, np.uint8 ) ); v[ 'LIT0' ].append( np.full( n, 1 if o.get( 'LIT0' ) else 0, np.uint8 ) )
			v[ 'PROP' ].append( np.full( n, 1 if prop else 0, np.uint8 ) )
			t = self.t
			t[ 'TRI' ].append( ii.astype( np.int64 ) + self.nv ); t[ 'KIND' ].append( np.full( len( ii ), o[ 'kind' ], np.uint8 ) )
			t[ 'OCC' ].append( np.full( len( ii ), occ, bool ) ); t[ 'CELL' ].append( tc ); t[ 'TERR' ].append( np.full( len( ii ), terrain, bool ) )
			self.nv += n
			self.ngrp += ngrp

	def close( self ):

		for k, c, dt in self.VF:

			a = np.concatenate( self.v[ k ] ) if self.v[ k ] else np.zeros( ( 0, c ) if c > 1 else 0, dt )
			self.v[ k ] = np.ascontiguousarray( a.astype( dt, copy = False ) )

		for k in self.t: self.t[ k ] = np.concatenate( self.t[ k ] ) if self.t[ k ] else np.zeros( ( 0, 3 ) if k == 'TRI' else 0, np.int64 )
		return self

def item_xform( it ):

	a = math.radians( it.get( 'rot' ) or 0 ); c, s = math.cos( a ), math.sin( a ); sc = it.get( 's', 1 ) or 1
	R = np.array( [ [ c, 0, s ], [ 0, 1, 0 ], [ - s, 0, c ] ] )
	return R, sc, b2t( it[ 'pos' ] )

def item_center( it, p ):

	R, sc, T = item_xform( it )
	mn = np.array( p[ 'min' ] ); mx = np.array( p[ 'max' ] )
	c = ( mn + mx ) * 0.5 * sc # blender local
	return R @ np.array( [ c[ 0 ], c[ 2 ], - c[ 1 ] ] ) + T

def proto_path( key, p, lod ):

	k = max( 0, min( lod, len( p.get( 'tris' ) or [ 0 ] ) - 1 ) )
	return os.path.join( ROOT, 'out/glb', key + ( '' if k == 0 else '.lod%d' % k ) + '.glb' )

def build_soup( level, cells_of_items ):

	S = Soup()
	missing = 0
	for ii, it in enumerate( TOWN[ 'items' ] ):

		p = TOWN[ 'protos' ].get( it[ 'p' ] )
		if not p: continue
		cid = cells_of_items[ ii ]
		if level == 1 and ( cid & 1 ): continue # far cells: base level only
		path = proto_path( it[ 'p' ], p, level )
		if not os.path.exists( path ):

			missing += 1
			continue

		R, sc, T = item_xform( it )
		# a small thing of its own (a bollard, a crate, a barrel): the packer keeps its shape in the lowest level
		prop = 'min' in p and 'max' in p and float( np.linalg.norm( ( np.array( p[ 'max' ] ) - np.array( p[ 'min' ] ) ) * sc ) ) < BK[ 'prop_size' ]
		for pr in load_glb( path ):

			if not len( pr[ 'I' ] ): continue
			wp = ( ( pr[ 'P' ].astype( np.float64 ) * sc ) @ R.T + T ).astype( np.float32 )
			wn = ( pr[ 'N' ].astype( np.float64 ) @ R.T ).astype( np.float32 )
			S.add( pr, wp, wn, cid, prop = prop )

	for t in TOWN.get( 'terrain', [] ):

		path = os.path.join( ROOT, 'out/glb', t[ 'file' ] + '.glb' )
		if not os.path.exists( path ):

			missing += 1
			continue

		for pr in load_glb( path ):

			if not len( pr[ 'I' ] ): continue
			if level == 1:

				# the near level only re-bakes the ground of near cells
				c = ( pr[ 'P' ][ pr[ 'I' ][ :, 0 ] ] + pr[ 'P' ][ pr[ 'I' ][ :, 1 ] ] + pr[ 'P' ][ pr[ 'I' ][ :, 2 ] ] ) / 3
				keep = ( cell_ids( c[ :, 0 ], c[ :, 2 ] ) & 1 ) == 0
				if not keep.any(): continue
				pr = dict( pr, I = pr[ 'I' ][ keep ] )

			S.add( pr, pr[ 'P' ], pr[ 'N' ], None, True )

	if missing: log( 'level %d: %d GLBs missing (skipped)' % ( level, missing ) )
	return S.close()

# ------------------------------------------------------------------------------------------ lights

def town_lights():

	L = TOWN.get( 'lights', [] )
	pos = np.array( [ b2t( l[ 'pos' ] ) for l in L ], np.float64 ).reshape( - 1, 3 )
	col = np.array( [ from_srgb( np.array( [ int( ( l.get( 'color' ) or '#ffbf6a' ).lstrip( '#' )[ i:i + 2 ], 16 ) / 255 for i in ( 0, 2, 4 ) ] ) ) for l in L ] ).reshape( - 1, 3 )
	inten = np.array( [ l.get( 'intensity', 0 ) * LOOK[ 'lamp' ] for l in L ], np.float64 )
	rng = np.array( [ l.get( 'range' ) or 9 for l in L ], np.float64 )
	near = np.array( [ l.get( 'near' ) or 0.03 for l in L ], np.float64 )
	# capped lanterns (Town._lights of the engine): most of their light leaves downwards and sideways. `up` = the share
	# that goes up: the record's own, or read off the flame - lantern flames flicker 0.07..0.35; open fires, paper
	# shades and the spill helpers of doors and tables are bare
	bare = re.compile( r'^(brazier|cafe_table|paper_lantern|parasol|door_plank|bicycle|neon_glyph|neon_sign_vertical|festoon_lights|bunting)/' )
	items = TOWN.get( 'items', [] )
	def up( l ):

		if l.get( 'up' ) is not None: return float( l[ 'up' ] )
		f = l.get( 'flicker' ) or 0
		it = l.get( 'item' )
		proto = items[ it ].get( 'p', '' ) if isinstance( it, int ) and 0 <= it < len( items ) else ''
		return LOOK[ 'cap_up' ] if 0.065 < f < 0.36 and not bare.match( proto ) else 1.0

	return dict( pos = pos, col = col, I = inten, range = rng, near = near, up = np.array( [ up( l ) for l in L ], np.float64 ),
		src = np.array( [ 0 if l.get( 'src' ) == 0 else 1 for l in L ], np.int32 ) )

LIGHTS = town_lights()

# ------------------------------------------------------------------------------------------ parallel map

def parallel( fn, n, dtype, tail, step, tag ):

	# out[ s:e ] = fn( s, e ) over 0..n in forked workers (the arrays fn closes over are shared copy-on-write)
	if n == 0: return np.zeros( ( 0, ) + tuple( tail ), dtype )
	slices = [ ( s, min( n, s + step ) ) for s in range( 0, n, step ) ]
	jobs = max( 1, min( JOBS, len( slices ) ) )
	if jobs == 1: return np.concatenate( [ np.asarray( fn( s, e ), dtype ) for s, e in slices ] )
	path = os.path.join( WORK, 'webbake_%d_%s.bin' % ( os.getpid(), tag ) )
	mm = np.memmap( path, dtype, 'w+', shape = ( n, ) + tuple( tail ) )
	pids = []
	for j in range( jobs ):

		pid = os.fork()
		if pid == 0:

			code = 0
			try:

				for k in range( j, len( slices ), jobs ): mm[ slices[ k ][ 0 ]:slices[ k ][ 1 ] ] = fn( *slices[ k ] )
				mm.flush()

			except BaseException:

				import traceback; traceback.print_exc(); code = 1

			os._exit( code )

		pids.append( pid )

	bad = sum( 1 for pid in pids if os.waitpid( pid, 0 )[ 1 ] != 0 )
	out = np.array( mm )
	del mm
	os.remove( path )
	if bad: raise RuntimeError( '%d workers failed (%s)' % ( bad, tag ) )
	return out

# ------------------------------------------------------------------------------------------ refinement

def refine( S ):

	# Large faces near lanterns are split (edge midpoints, conforming) until an edge is no longer than k * its
	# distance to the lantern: vertex lighting then carries the pools of light on walls and paving.
	V, Tt = S.v, S.t
	Lp = LIGHTS[ 'pos' ].astype( np.float32 )
	Ilum = LIGHTS[ 'I' ] * lum( LIGHTS[ 'col' ] )
	rsig = np.minimum( LIGHTS[ 'range' ], np.sqrt( np.maximum( Ilum, 0 ) / BK[ 'refine_emin' ] ) ).astype( np.float32 )
	use = rsig > BK[ 'refine_min' ]
	Lp, rsig = Lp[ use ], rsig[ use ]
	K, LMIN = BK[ 'refine_k' ], BK[ 'refine_min' ]
	if not len( Lp ): return
	lmin = Lp.min( 0 ) - rsig.max(); lmax = Lp.max( 0 ) + rsig.max()
	t_in = len( Tt[ 'TRI' ] )
	fresh = np.ones( t_in, bool ) # a face that passed once never needs another look: its edges do not change

	def edge_limit( A, B ):

		# longest edge allowed between A and B: k * distance to the nearest lantern that still lights it
		out = np.full( len( A ), np.inf, np.float32 )
		AB = B - A
		d2 = np.maximum( ( AB * AB ).sum( 1 ), 1e-12 )
		for s0 in range( 0, len( A ), 4096 ):

			A3 = A[ s0:s0 + 4096 ]; AB3 = AB[ s0:s0 + 4096 ]
			W = Lp[ None, :, : ] - A3[ :, None, : ] # e, L, 3
			tt = np.clip( np.einsum( 'elk,ek->el', W, AB3 ) / d2[ s0:s0 + 4096, None ], 0, 1 )
			C = W - tt[ :, :, None ] * AB3[ :, None, : ]
			dist = np.sqrt( np.einsum( 'elk,elk->el', C, C ) )
			out[ s0:s0 + 4096 ] = np.where( dist < rsig[ None, : ], np.maximum( K * dist, LMIN ), np.inf ).min( 1 )

		return out

	for it in range( BK[ 'refine_iters' ] ):

		P = V[ 'P' ]; TRI = Tt[ 'TRI' ]
		ft = np.nonzero( fresh )[ 0 ]
		tr = TRI[ ft ]
		lit = ( V[ 'LIT0' ][ tr[ :, 0 ] ] == 0 ) & ( Tt[ 'KIND' ][ ft ] != K_GLASS ) & ( Tt[ 'KIND' ][ ft ] != K_FALL ) & ( Tt[ 'KIND' ][ ft ] != K_CUT )
		a = P[ tr ] # T,3,3
		el = np.linalg.norm( a - a[ :, [ 1, 2, 0 ] ], axis = 2 ) # edge k: v_k -> v_k+1
		inb = np.all( ( a.max( 1 ) >= lmin ) & ( a.min( 1 ) <= lmax ), axis = 1 )
		csel = np.nonzero( lit & inb & ( el.max( 1 ) > LMIN ) )[ 0 ]
		fresh[ : ] = False
		if not len( csel ): break
		cand = ft[ csel ]; a = a[ csel ]; el = el[ csel ]
		A = a.reshape( - 1, 3 ); B = a[ :, [ 1, 2, 0 ] ].reshape( - 1, 3 ); ln = el.ravel()
		sel = np.nonzero( ln > LMIN )[ 0 ]
		A = A[ sel ]; B = B[ sel ]
		# canonical direction, so both faces of an edge take the same decision
		sw = ( A[ :, 0 ] > B[ :, 0 ] ) | ( ( A[ :, 0 ] == B[ :, 0 ] ) & ( ( A[ :, 1 ] > B[ :, 1 ] ) | ( ( A[ :, 1 ] == B[ :, 1 ] ) & ( A[ :, 2 ] > B[ :, 2 ] ) ) ) )
		A2 = np.ascontiguousarray( np.where( sw[ :, None ], B, A ) ); B2 = np.ascontiguousarray( np.where( sw[ :, None ], A, B ) )
		lim = parallel( lambda s0, s1: edge_limit( A2[ s0:s1 ], B2[ s0:s1 ] ), len( sel ), np.float32, (), 16384, 'rf' )
		flags = np.zeros( len( cand ) * 3, bool )
		flags[ sel ] = ln[ sel ] > lim
		flags = flags.reshape( - 1, 3 )

		anyf = flags.any( 1 )
		if not anyf.any(): break
		cand = cand[ anyf ]; flags = flags[ anyf ]
		tri = TRI[ cand ]
		# midpoint vertices, shared between faces that share both vertex indices
		e0 = tri; e1 = tri[ :, [ 1, 2, 0 ] ]
		lo = np.minimum( e0, e1 ); hi = np.maximum( e0, e1 )
		nv = len( P )
		keys = ( lo.astype( np.int64 ) * ( nv + 1 ) + hi )[ flags ]
		uk, inv = np.unique( keys, return_inverse = True )
		i0 = ( uk // ( nv + 1 ) ); i1 = ( uk % ( nv + 1 ) )
		mid = np.full( flags.shape, - 1, np.int64 )
		mid[ flags ] = nv + inv
		for kname, c, dt in Soup.VF:

			arr = V[ kname ]
			if kname in ( 'LAY', 'GRP', 'FOL', 'LIT0', 'PROP' ): new = arr[ i0 ]
			else:

				new = ( arr[ i0 ].astype( np.float64 ) + arr[ i1 ] ) * 0.5
				if kname == 'N': new /= np.maximum( np.linalg.norm( new, axis = 1, keepdims = True ), 1e-12 )
				new = new.astype( dt )

			V[ kname ] = np.concatenate( [ arr, new ] )

		nf = flags.sum( 1 )
		out_tri = []; out_src = []
		# 3 edges split
		s = np.nonzero( nf == 3 )[ 0 ]
		if len( s ):

			v = tri[ s ]; m = mid[ s ]
			for q in ( ( v[ :, 0 ], m[ :, 0 ], m[ :, 2 ] ), ( m[ :, 0 ], v[ :, 1 ], m[ :, 1 ] ), ( m[ :, 2 ], m[ :, 1 ], v[ :, 2 ] ), ( m[ :, 0 ], m[ :, 1 ], m[ :, 2 ] ) ):

				out_tri.append( np.stack( q, 1 ) ); out_src.append( cand[ s ] )

		# 1 edge split
		s = np.nonzero( nf == 1 )[ 0 ]
		if len( s ):

			k = np.argmax( flags[ s ], 1 ); r = np.arange( len( s ) )
			va = tri[ s ][ r, k ]; vb = tri[ s ][ r, ( k + 1 ) % 3 ]; vc = tri[ s ][ r, ( k + 2 ) % 3 ]; m = mid[ s ][ r, k ]
			out_tri += [ np.stack( ( va, m, vc ), 1 ), np.stack( ( m, vb, vc ), 1 ) ]; out_src += [ cand[ s ], cand[ s ] ]

		# 2 edges split: k = the edge left whole
		s = np.nonzero( nf == 2 )[ 0 ]
		if len( s ):

			k = np.argmin( flags[ s ], 1 ); r = np.arange( len( s ) )
			va = tri[ s ][ r, k ]; vb = tri[ s ][ r, ( k + 1 ) % 3 ]; vc = tri[ s ][ r, ( k + 2 ) % 3 ]
			mb = mid[ s ][ r, ( k + 1 ) % 3 ]; mc = mid[ s ][ r, ( k + 2 ) % 3 ]
			out_tri += [ np.stack( ( vc, mc, mb ), 1 ), np.stack( ( va, vb, mb ), 1 ), np.stack( ( va, mb, mc ), 1 ) ]; out_src += [ cand[ s ] ] * 3

		new_tri = np.concatenate( out_tri ); src = np.concatenate( out_src )
		keep = np.ones( len( TRI ), bool ); keep[ cand ] = False
		for kname in list( Tt.keys() ):

			if kname == 'TRI': continue
			Tt[ kname ] = np.concatenate( [ Tt[ kname ][ keep ], Tt[ kname ][ src ] ] )

		Tt[ 'TRI' ] = np.concatenate( [ TRI[ keep ], new_tri ] )
		fresh = np.zeros( len( Tt[ 'TRI' ] ), bool ); fresh[ int( keep.sum() ): ] = True

	log( '  refine: %d -> %d tris, %d verts' % ( t_in, len( Tt[ 'TRI' ] ), len( V[ 'P' ] ) ) )

# ------------------------------------------------------------------------------------------ visibility cull

def view_points():

	pts = []
	H = BK[ 'cull_height' ]
	for r in REGIONS:

		x0, y0, x1, y1 = r[ 'rect' ]
		zs = [ r.get( 'z', 0 ) ]
		if r.get( 'ramp' ): zs += [ r[ 'ramp' ].get( 'z0', 0 ), r[ 'ramp' ].get( 'z1', 0 ) ]
		if r.get( 'arch' ): zs.append( r.get( 'z', 0 ) + r[ 'arch' ].get( 'rise', 0 ) )
		for x in ( x0 - 0.6, x1 + 0.6 ):

			for y in ( y0 - 0.6, y1 + 0.6 ):

				for z in ( min( zs ) + 0.9, max( zs ) + H ): pts.append( b2t( [ x, y, z ] ) )

	for s in TOWN.get( 'shots', [] ): pts.append( b2t( s[ 'pos' ] ) )
	pts.append( b2t( TOWN.get( 'spawn', {} ).get( 'pos', [ 0, 0, 1.65 ] ) ) )
	return np.array( pts, np.float32 )

def cull_hidden( S ):

	# a single-sided face whose front half-space holds no point a camera can reach is never drawn: drop it
	if BK[ 'cull_height' ] <= 0: return
	V, Tt = S.v, S.t
	P = V[ 'P' ]; TRI = Tt[ 'TRI' ]
	a = P[ TRI[ :, 0 ] ]
	n = np.cross( P[ TRI[ :, 1 ] ] - a, P[ TRI[ :, 2 ] ] - a )
	ln = np.linalg.norm( n, axis = 1 )
	n /= np.maximum( ln, 1e-20 )[ :, None ]
	vis = ( Tt[ 'KIND' ] != K_OPAQUE ) | ( ln < 1e-12 )
	for v in view_points():

		vis |= ( ( v[ None, : ] - a ) * n ).sum( 1 ) > - 0.03

	k = int( ( ~ vis ).sum() )
	for name in list( Tt.keys() ): Tt[ name ] = Tt[ name ][ vis ]
	log( '  cull: %d faces can never face a camera (%.1f%%)' % ( k, 100 * k / max( 1, len( vis ) ) ) )

# ------------------------------------------------------------------------------------------ lighting

def make_bvh( P, TRI ):

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

def _hemi_dirs( n ):

	# cosine-weighted, stratified
	out = []
	g = ( math.sqrt( 5 ) - 1 ) / 2
	for i in range( n ):

		u = ( i + 0.5 ) / n; ph = 2 * math.pi * ( ( i * g ) % 1 )
		r = math.sqrt( u ); out.append( ( r * math.cos( ph ), r * math.sin( ph ), math.sqrt( 1 - u ) ) )

	return np.array( out, np.float32 )

def pos_hash( p ):

	q = np.floor( p.astype( np.float64 ) * 997.0 ).astype( np.int64 )
	return ( ( q[ :, 0 ] * 73856093 ) ^ ( q[ :, 1 ] * 19349663 ) ^ ( q[ :, 2 ] * 83492791 ) ) & 0xfffff

def light_range( bvh, P, N, FOL, s, e, Ed, AO ):

	# direct irradiance (lanterns + moon, shadowed) and sky visibility of vertices s..e
	rc = bvh.ray_cast
	Lp = LIGHTS[ 'pos' ].astype( np.float32 ); Lc = ( LIGHTS[ 'col' ] * LIGHTS[ 'I' ][ :, None ] ).astype( np.float32 )
	Ll = lum( Lc ).astype( np.float32 ); Lr = LIGHTS[ 'range' ].astype( np.float32 ); Lup = LIGHTS[ 'up' ].astype( np.float32 )
	Lskip = np.maximum( BK[ 'light_skip' ], LIGHTS[ 'near' ] + 0.25 ).astype( np.float32 )
	sr2 = LOOK[ 'source_radius' ] ** 2
	leak = BK[ 'shadow_leak' ]
	rs = np.random.RandomState( 12345 )
	JIT = rs.normal( size = ( 4096, 3 ) ).astype( np.float32 )
	JIT *= ( BK[ 'light_jitter' ] * rs.uniform( 0.3, 1, 4096 ) ** ( 1 / 3 ) / np.linalg.norm( JIT, axis = 1 ) )[ :, None ].astype( np.float32 )
	md = np.array( LOOK[ 'moon_dir' ], np.float64 ); md /= np.linalg.norm( md )
	mcol = np.array( LOOK[ 'moon_col' ], np.float32 ) * LOOK[ 'moon' ]
	H = _hemi_dirs( BK[ 'ao_rays' ] )
	KR = BK[ 'light_rays' ]
	B = 2048
	for b0 in range( s, e, B ):

		b1 = min( e, b0 + B )
		n = N[ b0:b1 ]; p = P[ b0:b1 ] + n * 0.03
		fol = FOL[ b0:b1 ] == 1
		two = FOL[ b0:b1 ] == 2 # thin panes: lit from either side (a lantern's glass glows with the flame behind it)
		hv = pos_hash( P[ b0:b1 ] ) # sampling pattern by position: coincident vertices come out identical and weld
		m = b1 - b0
		E = np.zeros( ( m, 3 ), np.float32 )
		if len( Lp ):

			D = Lp[ None, :, : ] - p[ :, None, : ]
			d2 = np.einsum( 'vlk,vlk->vl', D, D ); d = np.sqrt( d2 )
			ndl = np.einsum( 'vlk,vk->vl', D, n ) / np.maximum( d, 1e-6 )
			ndl = np.where( fol[ :, None ], 0.3 + 0.7 * np.maximum( ndl, - 0.6 * ndl ), np.where( two[ :, None ], np.abs( ndl ), ndl ) )
			w = ( d2 / ( d2 + sr2 ) ) / np.maximum( d2, 0.01 ) * np.clip( 1 - ( d / Lr[ None, : ] ) ** 4, 0, 1 ) ** 2 * np.clip( ndl, 0, None )
			# the lantern's cap (capFactor of ClusteredLightsNode.js): full light below ~25 degrees under its horizon, `up` above
			w *= 1 + ( Lup[ None, : ] - 1 ) * sstep( - 0.42, 0.3, - D[ :, :, 1 ] / np.maximum( d, 1e-3 ) )
			wl = w * Ll[ None, : ]
			vi, lj = np.nonzero( wl > BK[ 'light_eps' ] )
			if len( vi ):

				strong = wl[ vi, lj ] > 0.03
				reps = np.where( strong, KR, 1 )
				pv = np.repeat( np.arange( len( vi ) ), reps ) # pair of each ray
				rv = vi[ pv ]; rl = lj[ pv ]
				ordn = np.arange( len( pv ) ) - ( np.cumsum( reps ) - reps )[ pv ]
				tgt = Lp[ rl ] + JIT[ ( hv[ rv ] + rl * 31 + ordn * 7 ) % 4096 ] * ( reps[ pv ] > 1 )[ :, None ]
				dr = tgt - p[ rv ]
				dl = np.linalg.norm( dr, axis = 1 )
				dr /= np.maximum( dl, 1e-6 )[ :, None ]
				tm = dl - Lskip[ rl ]
				O = p[ rv ].tolist(); Dd = dr.tolist(); Tm = tm.tolist()
				hit = np.zeros( len( pv ), np.float32 )
				for i in range( len( pv ) ):

					t = Tm[ i ]
					if t > 0.02 and rc( O[ i ], Dd[ i ], t )[ 0 ] is not None: hit[ i ] = 1.0

				occ = np.bincount( pv, weights = hit, minlength = len( vi ) ) / reps
				vis = 1 - occ
				vis = np.where( fol[ vi ], 0.4 + 0.6 * vis, vis )
				ww = w[ vi, lj ] * ( leak + ( 1 - leak ) * vis )
				np.add.at( E, vi, ww[ :, None ].astype( np.float32 ) * Lc[ lj ] )

		# moon
		ndm = n @ md.astype( np.float32 )
		ndm = np.where( fol, 0.3 + 0.7 * np.maximum( ndm, - 0.6 * ndm ), np.where( two, np.abs( ndm ), ndm ) )
		mi = np.nonzero( ndm > 0.01 )[ 0 ]
		if len( mi ):

			jd = md[ None, : ] + JIT[ ( hv[ mi ] * 3 + 1 ) % 4096 ] * ( 0.02 / BK[ 'light_jitter' ] )
			jd /= np.linalg.norm( jd, axis = 1, keepdims = True )
			O = p[ mi ].tolist(); Dd = jd.tolist()
			hit = np.zeros( len( mi ), np.float32 )
			for i in range( len( mi ) ):

				if rc( O[ i ], Dd[ i ], 400.0 )[ 0 ] is not None: hit[ i ] = 1.0

			vis = 1 - hit
			vis = np.where( fol[ mi ], 0.4 + 0.6 * vis, vis )
			E[ mi ] += ( ndm[ mi ] * ( 0.04 + 0.96 * vis ) )[ :, None ] * mcol[ None, : ]

		Ed[ b0:b1 ] = E
		# ambient occlusion
		up = np.where( np.abs( n[ :, 1:2 ] ) < 0.9, np.array( [ [ 0, 1, 0 ] ], np.float32 ), np.array( [ [ 1, 0, 0 ] ], np.float32 ) )
		tx = np.cross( up, n ); tx /= np.maximum( np.linalg.norm( tx, axis = 1, keepdims = True ), 1e-9 )
		ty = np.cross( n, tx )
		ang = ( ( hv % 1024 ) * ( 2 * math.pi / 1024 ) ).astype( np.float32 ); ca, sa = np.cos( ang ), np.sin( ang )
		occ = np.zeros( m, np.float32 )
		R = BK[ 'ao_radius' ]
		O = p.tolist()
		for h in H:

			hx = h[ 0 ] * ca - h[ 1 ] * sa; hy = h[ 0 ] * sa + h[ 1 ] * ca
			dd = ( tx * hx[ :, None ] + ty * hy[ :, None ] + n * h[ 2 ] ).tolist()
			for i in range( m ):

				r = rc( O[ i ], dd[ i ], R )
				if r[ 0 ] is not None: occ[ i ] += 1.0 - 0.5 * r[ 3 ] / R # near blockers count more

		ao = 1 - BK[ 'ao_strength' ] * occ / len( H )
		AO[ b0:b1 ] = np.where( fol, 0.5 + 0.5 * ao, ao )

def light_soup( S, tag ):

	V, Tt = S.v, S.t
	occ = Tt[ 'OCC' ]
	t0 = time.time()
	bvh = make_bvh( V[ 'P' ], Tt[ 'TRI' ][ occ ] )
	log( '  bvh: %d occluder tris, %.1fs' % ( int( occ.sum() ), time.time() - t0 ) )
	return bvh

def run_lighting( bvh, P, N, FOL, need, tag ):

	# fork-parallel over the vertices that need light; results come back through shared memory maps
	idx = np.nonzero( need )[ 0 ]
	n = len( idx )
	Pn = np.ascontiguousarray( P[ idx ] ); Nn = np.ascontiguousarray( N[ idx ] ); Fn = np.ascontiguousarray( FOL[ idx ] )
	fE = os.path.join( WORK, 'webbake_%d_%s_E.bin' % ( os.getpid(), tag ) ); fA = os.path.join( WORK, 'webbake_%d_%s_A.bin' % ( os.getpid(), tag ) )
	Ed = np.memmap( fE, np.float32, 'w+', shape = ( max( n, 1 ), 3 ) ); AO = np.memmap( fA, np.float32, 'w+', shape = ( max( n, 1 ), ) )
	t0 = time.time()
	jobs = max( 1, min( JOBS, n // 4096 + 1 ) )
	step = 16384
	slices = [ ( s, min( n, s + step ) ) for s in range( 0, n, step ) ]
	pids = []
	for j in range( jobs ):

		pid = os.fork() if jobs > 1 else 0
		if pid == 0:

			code = 0
			try:

				for k in range( j, len( slices ), jobs ): light_range( bvh, Pn, Nn, Fn, slices[ k ][ 0 ], slices[ k ][ 1 ], Ed, AO )
				Ed.flush(); AO.flush()

			except BaseException:

				import traceback; traceback.print_exc(); code = 1

			if jobs > 1: os._exit( code )

		else: pids.append( pid )

	bad = 0
	for pid in pids:

		_, st = os.waitpid( pid, 0 )
		if st != 0: bad += 1

	if bad: raise RuntimeError( '%d lighting workers failed' % bad )
	E = np.zeros( ( len( P ), 3 ), np.float32 ); A = np.ones( len( P ), np.float32 )
	E[ idx ] = Ed[ :n ]; A[ idx ] = AO[ :n ]
	del Ed, AO
	os.remove( fE ); os.remove( fA )
	log( '  lighting: %d verts, %d workers, %.1fs' % ( n, jobs, time.time() - t0 ) )
	return E, A

def final_colour( S, E, A ):

	V = S.v
	n = V[ 'N' ]
	sky = np.array( LOOK[ 'sky' ], np.float32 ); gnd = np.array( LOOK[ 'ground' ], np.float32 )
	# (the sky half of the fill grows from the eaves up, as in the engine: streets are canyons, roofs see all of the sky)
	skyv = sky[ None, : ] * ( 1 + LOOK[ 'roof_fill' ] * sstep( 5.5, 13, V[ 'P' ][ :, 1 ] ) )[ :, None ]
	hemi = mix( gnd[ None, : ], skyv, ( 0.5 * n[ :, 1 ] + 0.5 )[ :, None ] ) * LOOK[ 'ambient' ]
	vao = V[ 'VAO' ]
	direct = E * mix( 1, vao, LOOK[ 'ao_direct' ] )[ :, None ]
	amb = hemi * ( vao * A )[ :, None ]
	lit = ( direct + amb ) / math.pi
	lit[ V[ 'LIT0' ] > 0 ] = 0
	return V[ 'MUL' ] * ( lit + V[ 'GLOW' ][ :, None ] ) + V[ 'EM' ]

# ------------------------------------------------------------------------------------------ chunk writer


def encode_chunk( S, C, LOCK, tsel ):

	# -> ( vertex bytes, index bytes (u32), flag bytes, meta, clamped uv groups ) for the triangles tsel of soup S (colours C).
	# Flags per vertex: 1 = must stay (LOCK), 2 = part of a small thing, 4 = ground (the terrain: see border_lock()).
	# Vertex record: 10 x u16 = position xyz on a power-of-two grid + texture layer, uv, colour (sqrt of value / 64), aux.
	V, Tt = S.v, S.t
	tri = Tt[ 'TRI' ][ tsel ]; kind = Tt[ 'KIND' ][ tsel ]
	order = np.argsort( kind, kind = 'stable' )
	tri = tri[ order ]; kind = kind[ order ]
	used, inv = np.unique( tri.ravel(), return_inverse = True )
	tri = inv.reshape( - 1, 3 )
	P = V[ 'P' ][ used ].astype( np.float64 )
	# positions snap to a grid shared by all chunks (1/1024 m unless the chunk is too large): ground vertices on a
	# cell border land on the same point in both cells
	q = 1 / 1024
	while True:

		mn = np.floor( P.min( 0 ) / q ) * q
		if ( ( P.max( 0 ) - mn ) / q ).max() <= 65534: break
		q *= 2

	rec = np.zeros( ( len( used ), 10 ), np.uint16 )
	rec[ :, 0:3 ] = np.round( ( P - mn ) / q )
	rec[ :, 3 ] = V[ 'LAY' ][ used ]
	# uv: whole tiles are dropped per primitive so the values fit the fixed range
	uv = V[ 'UV' ][ used ].astype( np.float64 )
	grp = V[ 'GRP' ][ used ]
	go = np.argsort( grp, kind = 'stable' )
	gs = grp[ go ]
	st = np.nonzero( np.concatenate( [ [ True ], gs[ 1: ] != gs[ :- 1 ] ] ) )[ 0 ]
	lo = np.minimum.reduceat( uv[ go ], st, axis = 0 ); hi = np.maximum.reduceat( uv[ go ], st, axis = 0 )
	shift = np.round( ( lo + hi ) * 0.5 )
	over = int( ( ( hi - lo ) > 2 * UV_RANGE - 2 ).any( 1 ).sum() )
	gi = np.empty( len( used ), np.int64 ); gi[ go ] = np.repeat( np.arange( len( st ) ), np.diff( np.concatenate( [ st, [ len( gs ) ] ] ) ) )
	uv -= shift[ gi ]
	rec[ :, 4:6 ] = np.clip( np.round( ( uv + UV_RANGE ) * ( 65536 / ( 2 * UV_RANGE ) ) ), 0, 65535 )
	col = C[ used ]
	rec[ :, 6:9 ] = np.round( np.sqrt( np.clip( col, 0, 64 ) / 64 ) * 65535 )
	aux = V[ 'AUX' ][ used ].astype( np.float64 )
	# opaque: gloss in the lower half, a glowing surface (higher highlight ceiling) flagged in the upper half
	opq = np.zeros( len( used ), bool ); opq[ tri[ kind == K_OPAQUE ].ravel() ] = True
	a16 = np.where( opq, np.where( V[ 'EMF' ][ used ] > 0.5, 65535, np.round( np.clip( aux, 0, 1 ) * 32767 ) ), np.round( np.clip( aux, 0, 1 ) * 65535 ) )
	rec[ :, 9 ] = a16
	lock = LOCK[ used ] | ( V[ 'PROP' ][ used ] << 1 )
	# weld identical records
	view = np.ascontiguousarray( rec ).view( np.dtype( ( np.void, 20 ) ) ).ravel()
	_, first, winv = np.unique( view, return_index = True, return_inverse = True )
	# keep first-use order (locality) rather than the sorted order of np.unique
	rank = np.argsort( np.argsort( first ) )
	keep = np.sort( first )
	rec = rec[ keep ]
	lk = np.zeros( len( keep ), np.uint8 ); np.bitwise_or.at( lk, rank[ winv ], lock )
	tri = rank[ winv ][ tri ]
	ok = ( tri[ :, 0 ] != tri[ :, 1 ] ) & ( tri[ :, 1 ] != tri[ :, 2 ] ) & ( tri[ :, 0 ] != tri[ :, 2 ] )
	tri = tri[ ok ]; kind = kind[ ok ]
	sub = []
	for k in ( K_OPAQUE, K_GLASS, K_WATER, K_FALL, K_CUT ):

		w = np.nonzero( kind == k )[ 0 ]
		if len( w ): sub.append( [ int( k ), int( w[ 0 ] ) * 3, int( len( w ) ) * 3 ] )

	mx = mn + rec[ :, 0:3 ].max( 0 ) * q
	meta = dict( min = [ float( x ) for x in mn ], max = [ round( float( x ), 4 ) for x in mx ], q = q, nv = int( len( rec ) ), ni = int( len( tri ) * 3 ), sub = sub )
	return rec.tobytes(), tri.astype( np.uint32 ).tobytes(), lk.tobytes(), meta, over

def border_lock( S ):

	# ground vertices shared by two cells: the far level must keep them, or the ground would open along the seam
	V, Tt = S.v, S.t
	lock = np.zeros( len( V[ 'P' ] ), np.uint8 )
	tt = np.nonzero( Tt[ 'TERR' ] )[ 0 ]
	if not len( tt ): return lock
	vi = Tt[ 'TRI' ][ tt ].ravel(); ci = np.repeat( Tt[ 'CELL' ][ tt ], 3 )
	# 4: a vertex of the ground. The packer keeps them all at the levels above the lowest and gives them one colour
	# and gloss for all levels: the pool of a lantern on the paving, its streaks and the wet film are then the same
	# at every level of a cell (a simplifier steered by the colour alone folds the gloss - where the film lies - into
	# other triangles at each level)
	lock[ vi ] |= 4
	q = np.round( V[ 'P' ][ vi ].astype( np.float64 ) * 1024 ).astype( np.int64 )
	_, pid = np.unique( q, axis = 0, return_inverse = True )
	pid = pid.ravel()
	o = np.lexsort( ( ci, pid ) )
	ps = pid[ o ]; cs = ci[ o ]
	n = pid.max() + 1
	cmin = np.full( n, np.iinfo( np.int64 ).max ); cmax = np.full( n, - 1 )
	np.minimum.at( cmin, ps, cs ); np.maximum.at( cmax, ps, cs )
	lock[ vi[ ( cmin != cmax )[ pid ] ] ] |= 1
	return lock

# ------------------------------------------------------------------------------------------ textures

def window_layer( size ):

	# the fake lit room of MaterialLib 'window_lit' without the parallax: 2 x 2 rooms, each WIN_CELL metres of wall
	S = size
	yy, xx = np.mgrid[ 0:S, 0:S ].astype( np.float64 )
	u = ( xx + 0.5 ) / S * 2; v = ( 1 - ( yy + 0.5 ) / S ) * 2 # cells, v up
	cx = np.floor( u ); cy = np.floor( v )
	fx = u - cx; fy = v - cy
	r1 = np.mod( np.sin( cx * 91.7 + cy * 37.3 + 3.1 ) * 43758.5453, 1 ); r2 = np.mod( np.sin( cx * 47.3 + cy * 71.9 + 1.7 ) * 24634.6345, 1 )
	wy = fy * WIN_CELL[ 1 ]
	wall = mix( np.array( [ 0.55, 0.36, 0.17 ] ), np.array( [ 0.62, 0.46, 0.3 ] ), r1[ ..., None ] ) * mix( 1, 0.45, sstep( 0.2, 1.6, wy - 0.25 ) )[ ..., None ]
	lx = r1 * 0.6 + 0.2; ly = r2 * 0.5 - 0.25
	ld2 = ( ( fx - lx ) * WIN_CELL[ 0 ] ) ** 2 + ( ( fy - 0.5 ) * WIN_CELL[ 1 ] - ly ) ** 2
	lamp = np.exp( - 1.6 * ld2 ) * 2.2
	frame = ( np.abs( ( fx - 0.7 ) * WIN_CELL[ 0 ] ) < 0.28 ) * ( np.abs( ( fy - 0.62 ) * WIN_CELL[ 1 ] ) < 0.2 )
	furn = sstep( 0.1, 0.0, fy - ( r2 * 0.25 + 0.12 ) ) * ( fx < 0.45 )
	room = wall * ( lamp + 0.35 )[ ..., None ] * ( 1 - frame * 0.55 )[ ..., None ] * ( 1 - furn * 0.75 )[ ..., None ]
	xm = u * WIN_CELL[ 0 ]
	fold = np.sin( xm * 55 + np.sin( xm * 13 ) * 2 ) * 0.5 + 0.5
	cur = sstep( 0.0, 0.12, np.sin( xm * 2.2 + cy * 2.0 ) * 0.5 + fold * 0.15 - 0.12 )
	ccol = mix( np.array( [ 0.85, 0.62, 0.36 ] ), np.array( [ 0.7, 0.3, 0.16 ] ), ( r2 > 0.6 )[ ..., None ] ) * ( fold * 0.5 + 0.5 )[ ..., None ] * 0.75
	room = mix( room, ccol, ( cur * 0.82 )[ ..., None ] )
	out = np.ones( ( S, S, 4 ), np.float32 )
	out[ ..., :3 ] = np.clip( room / WINDOW_GAIN, 0, 1 )
	return out

def stripe_layer( size, s ):

	u = ( np.arange( size ) + 0.5 ) / size
	band = sstep( 0.47, 0.53, np.abs( u - 0.5 ) * 2 )
	row = mix( s[ 'c1' ][ None, : ], s[ 'c2' ][ None, : ], band[ :, None ] ) / s[ 'cmax' ][ None, : ]
	out = np.ones( ( size, size, 4 ), np.float32 )
	out[ ..., :3 ] = row[ None, :, : ]
	return out

def resize_axis( a, s, axis ):

	# box-filter resample of one axis to s samples (any ratio): differences of the running sum at the new pixel borders
	a = np.moveaxis( a, axis, 0 )
	n = a.shape[ 0 ]
	if n != s:

		C = np.concatenate( [ np.zeros( ( 1, ) + a.shape[ 1: ] ), np.cumsum( a, 0, dtype = np.float64 ) ] )
		b = np.arange( s + 1 ) * ( n / s )
		i0 = np.minimum( np.floor( b ).astype( int ), n ); i1 = np.minimum( i0 + 1, n )
		fr = ( b - i0 ).reshape( ( - 1, ) + ( 1, ) * ( a.ndim - 1 ) )
		I = C[ i0 ] + ( C[ i1 ] - C[ i0 ] ) * fr
		a = ( I[ 1: ] - I[ :- 1 ] ) / ( n / s )

	return np.moveaxis( a, 0, axis )

def resize( a, s ):

	return resize_axis( resize_axis( a, s, 0 ), s, 1 )

def lib_layer( name, sizes ):

	# pre-shaded albedo of a library material (linear, top row first): albedo x cavity AO x a soft relief light
	# from above, with the relief height in alpha. The lean runtime samples nothing else.
	import bpy
	d = os.path.join( ROOT, 'out/mat', name )
	info = INFO[ name ]
	def load( f, ch = 4 ):

		img = bpy.data.images.load( os.path.join( d, f ), check_existing = False )
		img.colorspace_settings.name = 'Non-Color'
		w, h = img.size
		a = np.empty( w * h * img.channels, np.float32 )
		img.pixels.foreach_get( a )
		a = a.reshape( h, w, img.channels )[ ::- 1 ] # top row first
		bpy.data.images.remove( img )
		return a

	alb = from_srgb( load( info[ 'maps' ][ 'albedo' ] )[ ..., :3 ].astype( np.float64 ) )
	orm = load( info[ 'maps' ][ 'orm' ] )
	nrm = load( info[ 'maps' ][ 'normal' ] )
	ao = orm[ ..., 0:1 ].astype( np.float64 )
	ny = nrm[ ..., 1 ].astype( np.float64 ) * 2 - 1
	hgt = nrm[ ..., 3 ] if nrm.shape[ 2 ] > 3 else np.full( nrm.shape[ :2 ], 0.5, np.float32 )
	relief = 1 + CFG.get( 'relief', 0.55 ) * CFG.get( 'relief_sign', - 1.0 ) * ny
	relief /= relief.mean()
	lean = alb * mix( 1, ao, 0.75 ) * relief[ ..., None ]
	rgba = np.concatenate( [ lean, hgt[ ..., None ].astype( np.float64 ) ], 2 )
	out = {}
	for s in sizes: out[ s ] = resize( rgba, s ).astype( np.float32 )

	return out

def write_pages( layers_px, size, cols, rows, quality ):

	# layers_px: list of (S,S,4) float arrays: rgb linear, alpha 0..1 -> page images (sRGB bytes; alpha kept >= 0.5 so
	# a premultiplied canvas read-back loses no colour precision)
	import bpy
	per = cols * rows
	files = []
	for p0 in range( 0, len( layers_px ), per ):

		page = np.zeros( ( rows * size, cols * size, 4 ), np.float32 )
		page[ ..., 3 ] = 1
		for i, a in enumerate( layers_px[ p0:p0 + per ] ):

			r, c = divmod( i, cols )
			t = np.empty( ( size, size, 4 ), np.float32 )
			t[ ..., :3 ] = to_srgb( a[ ..., :3 ] ); t[ ..., 3 ] = 0.5 + 0.5 * np.clip( a[ ..., 3 ], 0, 1 )
			page[ r * size:( r + 1 ) * size, c * size:( c + 1 ) * size ] = t

		name = 'tex%d_%d' % ( size, p0 // per )
		img = bpy.data.images.new( name, cols * size, rows * size, alpha = True, float_buffer = False )
		img.colorspace_settings.name = 'Non-Color'
		img.alpha_mode = 'STRAIGHT'
		img.pixels.foreach_set( np.ascontiguousarray( page[ ::- 1 ] ).ravel() )
		png = os.path.join( OUT, name + '.png' )
		img.filepath_raw = png; img.file_format = 'PNG'
		img.save()
		bpy.data.images.remove( img )
		files.append( name )

	return files

def bake_textures():

	sizes = BK[ 'tex_sizes' ]
	names = LAYERS.names
	cache_dir = os.path.join( OUT, '.texcache' )
	os.makedirs( cache_dir, exist_ok = True )
	per_size = { s: [] for s in sizes }
	t0 = time.time()
	for i, name in enumerate( names ):

		if name == 'white': px = { s: np.ones( ( s, s, 4 ), np.float32 ) for s in sizes }
		elif name == 'window': px = { s: window_layer( s ) for s in sizes }
		elif name.startswith( 'stripe:' ): px = { s: stripe_layer( s, LAYERS.stripes[ name ] ) for s in sizes }
		else:

			d = os.path.join( ROOT, 'out/mat', name )
			info = INFO[ name ]
			sig = hashlib.sha1( json.dumps( [ [ f, os.path.getmtime( os.path.join( d, f ) ), os.path.getsize( os.path.join( d, f ) ) ] for f in sorted( info[ 'maps' ].values() ) ] + [ sizes, CFG.get( 'relief', 0.55 ), CFG.get( 'relief_sign', - 1.0 ), 4 ] ).encode() ).hexdigest()[ :16 ]
			cf = os.path.join( cache_dir, '%s_%s.npz' % ( name, sig ) )
			if os.path.exists( cf ):

				z = np.load( cf ); px = { s: z[ 's%d' % s ].astype( np.float32 ) / 255 for s in sizes }
				for s in sizes: px[ s ][ ..., :3 ] = from_srgb( px[ s ][ ..., :3 ] )

			else:

				px = lib_layer( name, sizes )
				for f in os.listdir( cache_dir ):

					if f.startswith( name + '_' ) and f[ len( name ) + 1: ].count( '_' ) == 0: os.remove( os.path.join( cache_dir, f ) )

				sv = {}
				for s in sizes:

					q = px[ s ].copy(); q[ ..., :3 ] = to_srgb( q[ ..., :3 ] )
					sv[ 's%d' % s ] = np.round( np.clip( q, 0, 1 ) * 255 ).astype( np.uint8 )

				np.savez_compressed( cf, **sv )

		for s in sizes: per_size[ s ].append( px[ s ] )

	tex = {}
	for s in sizes:

		cols, rows = ( 4, 2 ) if s >= 512 else ( 8, 4 )
		files = write_pages( per_size[ s ], s, cols, rows, 85 )
		tex[ str( s ) ] = dict( size = s, cols = cols, rows = rows, pages = files )

	log( 'textures: %d layers x %s, %.1fs' % ( len( names ), sizes, time.time() - t0 ) )
	return tex

# ------------------------------------------------------------------------------------------ main

def main():

	os.makedirs( OUT, exist_ok = True )
	protos = TOWN[ 'protos' ]; items = TOWN[ 'items' ]
	log( 'town: %d items, %d protos, %d lights' % ( len( items ), len( protos ), len( LIGHTS[ 'pos' ] ) ) )

	# chunk cell of every item (by the centre of its bounds)
	centres = np.array( [ item_center( it, protos[ it[ 'p' ] ] ) if it[ 'p' ] in protos else [ 0, 0, 0 ] for it in items ] ).reshape( - 1, 3 )
	cells = cell_ids( centres[ :, 0 ], centres[ :, 2 ] ) if len( items ) else np.zeros( 0, np.int64 )

	# which materials earn a texture layer: surface area over the base level
	area = {}
	for ii, it in enumerate( items ):

		p = protos.get( it[ 'p' ] )
		if not p: continue
		path = proto_path( it[ 'p' ], p, 2 )
		if not os.path.exists( path ): continue
		sc = it.get( 's', 1 ) or 1
		for pr in load_glb( path ):

			P, I = pr[ 'P' ], pr[ 'I' ]
			a = 0.5 * np.linalg.norm( np.cross( P[ I[ :, 1 ] ] - P[ I[ :, 0 ] ], P[ I[ :, 2 ] ] - P[ I[ :, 0 ] ] ), axis = 1 ).sum() * sc * sc
			area[ pr[ 'lib' ] ] = area.get( pr[ 'lib' ], 0 ) + float( a )

	for t in TOWN.get( 'terrain', [] ):

		path = os.path.join( ROOT, 'out/glb', t[ 'file' ] + '.glb' )
		if not os.path.exists( path ): continue
		for pr in load_glb( path ):

			P, I = pr[ 'P' ], pr[ 'I' ]
			area[ pr[ 'lib' ] ] = area.get( pr[ 'lib' ], 0 ) + float( 0.5 * np.linalg.norm( np.cross( P[ I[ :, 1 ] ] - P[ I[ :, 0 ] ], P[ I[ :, 2 ] ] - P[ I[ :, 0 ] ] ), axis = 1 ).sum() )

	for lib, a in sorted( area.items(), key = lambda kv: - kv[ 1 ] ):

		# (leaf cards are cut out by their layer's height: foliage always gets its layer)
		if CATALOG.get( lib, {} ).get( 'kind' ) == 'baked' and INFO.get( lib ) and ( a >= BK[ 'layer_min_area' ] or lib in ( 'leaf', 'hedge' ) ): LAYERS.add_lib( lib )

	log( 'texture layers: %d materials of %d used' % ( len( LAYERS.libs ), len( area ) ) )

	raw = os.path.join( OUT, '.raw' )
	if os.path.isdir( raw ):

		for fn in os.listdir( raw ): os.remove( os.path.join( raw, fn ) )

	os.makedirs( raw, exist_ok = True )
	chunks = {} # key -> entry
	stats = {}
	for level, lname in ( ( 2, 'mid' ), ( 1, 'near' ) ):

		log( 'level %s (LOD%d)' % ( lname, level ) )
		S = build_soup( level, cells )
		nt0 = len( S.t[ 'TRI' ] )
		log( '  soup: %d tris, %d verts' % ( nt0, len( S.v[ 'P' ] ) ) )
		bvh = light_soup( S, level )
		cull_hidden( S )
		refine( S )
		V = S.v
		need = np.zeros( len( V[ 'P' ] ), bool ); need[ S.t[ 'TRI' ].ravel() ] = True
		need &= ( V[ 'LIT0' ] == 0 ) & ( V[ 'MUL' ].max( 1 ) > 0 )
		E, A = run_lighting( bvh, V[ 'P' ], V[ 'N' ], V[ 'FOL' ], need, 'l%d' % level )
		del bvh
		C = final_colour( S, E, A )
		LOCK = border_lock( S )
		cid = S.t[ 'CELL' ]
		order = np.argsort( cid, kind = 'stable' )
		cs = cid[ order ]
		st = np.nonzero( np.concatenate( [ [ True ], cs[ 1: ] != cs[ :- 1 ] ] ) )[ 0 ]
		en = np.concatenate( [ st[ 1: ], [ len( cs ) ] ] )
		tv = tt = 0; over = 0
		for a, b in zip( st, en ):

			c = int( cs[ a ] )
			key = cell_key( c )
			if level == 1:

				# a cell of ground alone gains nothing from the near level
				if ( c & 1 ) or key not in chunks or not ( cells == c ).any(): continue

			vb, ib, lk, meta, ov = encode_chunk( S, C, LOCK, order[ a:b ] )
			over += ov
			tv += meta[ 'nv' ]; tt += meta[ 'ni' ] // 3
			fn = '%s.%s.bin' % ( key, lname )
			with open( os.path.join( raw, fn ), 'wb' ) as f: f.write( vb ); f.write( ib ); f.write( lk )
			if level == 2: chunks[ key ] = dict( k = key, outer = bool( c & 1 ) )
			chunks[ key ][ lname ] = dict( raw = fn, **meta )

		if over: log( '  WARNING: %d surfaces span more than %d texture tiles (uv clamped)' % ( over, int( 2 * UV_RANGE - 2 ) ) )
		stats[ lname ] = dict( tris_in = int( nt0 ), tris = int( tt ), verts = int( tv ) )
		log( '  written: %d tris, %d verts' % ( tt, tv ) )
		del S, C, E, A

	# nearest the spawn first: the packer keeps this order in the base file, so a streamed download shows the surroundings early
	sp = b2t( TOWN.get( 'spawn', {} ).get( 'pos', [ 0, 0, 0 ] ) )
	def dist( ch ):

		c = ( np.array( ch[ 'mid' ][ 'min' ] ) + np.array( ch[ 'mid' ][ 'max' ] ) ) * 0.5
		return float( np.hypot( c[ 0 ] - sp[ 0 ], c[ 2 ] - sp[ 2 ] ) )

	ordered = sorted( chunks.values(), key = dist )

	tex = bake_textures()

	# lantern glows + flames. One glow per fitting, where its burner is seen - never at a helper light (shop-front spill,
	# window fill, one light standing in for a row of lamps): nothing is there to look at and the glow hangs in the air.
	# town.json: `halo` = the output the glow stands for (a fitting whose light is split over helpers, like the globe
	# post), `halo_at` = the burners when the light sits off them (the hall's clock lanterns) or stands for several,
	# `halo_r` = the radius of an opaque glowing body round the burner. A row: x, y, z, r, g, b, size, 1, lift (m towards the eye)
	halos = []
	for i, l in enumerate( TOWN.get( 'lights', [] ) ):

		I = l.get( 'halo', l.get( 'intensity', 0 ) if LIGHTS[ 'src' ][ i ] else 0 ) * LOOK[ 'lamp' ]
		if I < 12: continue
		at = [ b2t( p ) for p in l[ 'halo_at' ] ] if l.get( 'halo_at' ) else [ LIGHTS[ 'pos' ][ i ] ] # several: the globes of one head, each with its share
		size = max( 0.5, round( math.sqrt( I / len( at ) / 40 ) * 1.5 * 4 ) / 4 )
		lift = round( float( l.get( 'halo_r', 0 ) ) * 1.15 + 0.01, 3 ) if l.get( 'halo_r' ) else 0 # in front of an opal globe, which would hide the hot centre
		for p in at: halos.append( [ round( float( v ), 3 ) for v in p ] + [ round( float( v ), 4 ) for v in LIGHTS[ 'col' ][ i ] ] + [ size, 1, lift ] )

	flames = []
	sock_cache = {}
	for ii, it in enumerate( items ):

		p = protos.get( it[ 'p' ] )
		if not p or not p.get( 'fx' ): continue
		path = proto_path( it[ 'p' ], p, 0 )
		if path not in sock_cache: sock_cache[ path ] = glb_sockets( path ) if os.path.exists( path ) else []
		R, sc, T = item_xform( it )
		for s in sock_cache[ path ]:

			pr = s[ 'props' ]
			if s[ 'kind' ] != 'fx' or not ( pr.get( 'fx' ) == 'flame' or pr.get( 'kind' ) == 'flame' or 'flame' in s[ 'name' ] ): continue
			wp = R @ ( s[ 'pos' ] * sc ) + T
			c = lin( pr.get( 'color' ) or '#ffb860' )
			flames.append( [ round( float( v ), 3 ) for v in wp ] + [ round( float( v ), 4 ) for v in c ] + [ float( pr.get( 'size', 0.09 ) ) * sc, float( pr.get( 'glow', 9 ) ), float( pr.get( 'speed', 1 ) ) ] )

	lights = [ [ round( float( v ), 3 ) for v in LIGHTS[ 'pos' ][ i ] ] + [ round( float( v ), 4 ) for v in LIGHTS[ 'col' ][ i ] * LIGHTS[ 'I' ][ i ] ] + [ float( LIGHTS[ 'range' ][ i ] ), float( LIGHTS[ 'up' ][ i ] ) ]
		for i in range( len( LIGHTS[ 'pos' ] ) ) ]

	man = dict(
		version = 1,
		town = dict( name = CFG.get( 'name', 'town' ), sha1 = hashlib.sha1( TOWN_RAW ).hexdigest(), bytes = len( TOWN_RAW ), built = TOWN.get( 'stats', {} ).get( 'built' ) ),
		baked = int( time.time() ),
		look = LOOK, bake = BK,
		stride = 20, uv_range = UV_RANGE, colour_scale = 64, window_gain = WINDOW_GAIN, cut_aux = CUT_AUX,
		layers = LAYERS.names, tex = tex,
		chunks = ordered,
		halos = halos, flames = flames, lights = lights,
		stats = stats,
		# what the walk needs of town.json (navigation is analytic)
		data = { k: TOWN.get( k ) for k in ( 'seed', 'frame', 'spawn', 'shots', 'regions', 'colliders', 'missing', 'stats' ) },
	)
	json.dump( man, open( os.path.join( raw, 'raw.json' ), 'w' ), separators = ( ',', ':' ) )
	log( 'raw: %d chunks (%d with a near level), %d halos, %d flames' % ( len( ordered ), sum( 1 for c in ordered if 'near' in c ), len( halos ), len( flames ) ) )

main()
