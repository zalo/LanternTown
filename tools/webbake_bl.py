# Web bake worker: runs inside Blender's python (numpy + mathutils), driven by tools/webbake.py.
#
#   blender --background --factory-startup --python tools/webbake_bl.py -- <config.json>
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
# tools/webbake.py reads every one of these from those files (LOOK_SRC there) and passes them in, so the numbers below
# are only what a run without the driver gets; a change of the look needs a re-bake, nothing else.
LOOK = dict(
	lamp = 0.5, ambient = 1.3, moon = 2.0,
	sky = ( 0.33, 0.40, 0.60 ), ground = ( 0.16, 0.15, 0.2 ), moon_col = ( 0.62, 0.68, 0.87 ), moon_dir = ( - 0.45, 0.62, 0.35 ),
	source_radius = 0.6, ao_direct = 0.55,
	reach = 4.5, # pools of light: a lantern's light x 1 / ( 1 + ( d / reach )^2 ) (lampFalloff of ClusteredLightsNode.js; 0 = pure inverse square)
	cap_up = 0.25, roof_fill = 2.5, # engine: CAP_UP of town.js (capped lanterns), roofFill (sky fill above the eaves)
	wet = 0.68, grime = 0.8, wear = 1.0, special = 1.0, macro = 1.0, soft = 1.0, window_glow = 0.75, emissive = 1.0,
	puddle_rough = 0.09, film_rough = 0.17, pool_cap = 1.0, tide_h = 0.4,
	# foliage and bark at night (FoliageLighting + the fol* uniforms of web/engine/materials.js): extra sky floor (x the
	# sky fill), its share on faces looking down, wrap / transmitted share of every light, how much of the reach falloff
	# a crown above the lamp head keeps, the share of a capped lantern's withheld light that comes back up from the
	# paving, floor of the painted occlusion, share of the ambient occlusion left off
	fol_sky = 6.0, fol_under = 0.35, fol_wrap = 0.3, fol_back = 0.3, fol_reach = 0.5, fol_bounce = 0.3, fol_ao = 0.45, fol_gtao = 0.6,
	stained_back = 0.16, stained_reach = 1.6, stained_pale = 0.65, # StainedLighting: stained glass with a lamp at it
	# signs: the page draws a sign's tubes, bulbs and faces this many times as bright as their materials say (HDR.neon of
	# web/engine/lite.js), so the light a sign throws on its wall and the paving (the town's `neon...` lights) is as many
	# times what the town declares. 1 = as declared
	neon = 1.0,
)
LOOK.update( CFG.get( 'look', {} ) )

# bake parameters
BK = dict(
	cell = 16.0, far_cell = 64.0, near_zone = 56.0, # chunk grid (m); items further than near_zone from the walkable area: coarse cells, base level only
	refine_k = 0.5, refine_min = 0.6, refine_emin = 0.08, refine_iters = 7, # tessellation of large faces near lanterns (edge <= k * distance)
	shadow_leak = 0.06, light_eps = 0.003, light_skip = 0.42, light_jitter = 0.14, light_rays = 3,
	ao_rays = 12, ao_radius = 3.0, ao_strength = 0.7,
	# one bounce: the pool a lantern throws on the paving lights what stands around it (0 = off). bounce = the paving's
	# albedo; its light dies out bounce_range metres from the pool. fol_vis: how much of a lantern's light foliage keeps
	# where its own crown shadows it (the engine's lanterns cast no shadow inside a crown)
	bounce = 0.3, bounce_range = 12.0, fol_vis = 0.85,
	metal_diffuse = 0.35, # how much of the lit colour a metal keeps (stand-in for its specular)
	layer_min_area = 20.0, # m2 of surface below which a textured material is drawn in its mean colour instead of taking a texture layer
	cull_height = 7.0, # faces that cannot be seen from anywhere above the walkable area up to this height are dropped (0 = keep all)
	tex_sizes = [ 512, 256 ],
	prop_size = 2.5, # things smaller than this across (m) are flagged for the packer: they keep their shape in the lowest level
	# --- next branch
	fine = 1, # also bake the "fine" level (LOD0 geometry) of the near cells: the top level of the medium / high tiers
	# layers of their own file at a larger size (l<size>_<layer>.webp), for the tiers that draw the near field sharp:
	# 1024 for every library material, 2048 for those whose maps are that large. big_tile: the layers with tiles of
	# at least this many metres come first after the 2048 ones (a tier may then give that run of layers the larger size)
	layer_sizes = [ 1024, 2048 ], big_tile = 0.8,
	lod_masks = 1, # the reduced models take their baked occlusion / dirt from the full model's surface (see lod_masks())
	# ... and their large faces are split where the corners do not tell what the full model has in between (lod_split()):
	# edges longer than lod_split_min metres (LOD1, LOD2; 0 = off) while the mask is off by more than lod_split_tol, or longer than lod_split_max (0 = no limit)
	lod_split_min = [ 0.75, 1.5 ], lod_split_max = 0, lod_split_tol = 0.1, lod_split_iters = 6,
	# the fine level carries where its light comes from (light_dir(): 4 bytes per vertex), so the page can light the
	# relief of a wall or a cobble per pixel. dir_max: the longest direction vector kept (1 / cosine of the steepest light)
	# dircheck: also measure that model, and light probes on a grid, against the bake's own light on turned normals (dir_check())
	dir = 1, dir_max = 4.0, dircheck = 0,
	pane_r = 0.64, # lantern panes carry the vector to their own burner (x 127 / pane_r): the page draws the mantle's glare
	pane_uv = 1, # ... in the uv slots of glass vertices, at every level (pane_bytes(); told to the page, not a switch)
	nr_sizes = [ 512, 1024 ], # the normal + roughness layers of the library materials, for the same tiers
	single_layers = 1, # 0: no l<size>_<layer> files (the lean bake: its tiers draw the pages; the layers keep their order all the same)
	# the position grid and what glows (encode_chunk()): 1 = a glowing surface is rounded outward along its normal and
	# glass a step further, so the grid cannot lay a lit card onto the wall behind it or a pane into its lit room; 0 = all to nearest
	glow_out = 0,
	# glow_audit: '' = off, 'all' or a list of cell keys 'c_1_-1,c_0_0': what the grid does to the distance between a
	# glowing surface and what lies within glow_reach in front of and behind it (glow_audit(): a report, nothing changes)
	glow_audit = '', glow_reach = 0.012, glow_sample = 6000,
	# what only glows inside a glowing body the town declares round a burner (`halo_at` + `halo_r`: an opal globe) is at
	# least this bright, in the hue of its light (body_glow(); 0 = off, the material's own glow alone)
	halo_body = 3.0,
	# The lanterns' shadows on the ground as a picture from above (ground_shadows(); 0 = off: the ground's vertices hold
	# them, as coarse as the ground's mesh). The ground's vertices then take the lanterns' light without its shadows -
	# smooth, the same at every level - and the picture holds, per gsm_texel metres, how much of that light arrives
	# (gsm_rays rays to every lantern that counts there, spread over its mantle) and how much of the texel's light is
	# not a lantern's (gsm_ao_rays: the occlusion of that fill). gsm_floor / gsm_lift: as --floor / --lift of
	# tools/aomap.py, which stores the picture beside its own (the page reads both in one fetch). gsm_grow: texels
	# inside a solid take their neighbours' values so many texels inwards (a filtered read at a wall's foot is then
	# the paving's); gsm_fill: the radius (texels) the fill is smoothed over, as the vertices smooth it
	gsm = 1, gsm_texel = 0.0625, gsm_rays = 8, gsm_ao_rays = 6, gsm_floor = 0.2, gsm_lift = 0.05, gsm_grow = 3, gsm_fill = 5,
)
BK.update( CFG.get( 'bake', {} ) )
# A second look lit onto the same mesh (tools/webbake.py --set look2.KEY=VALUE): what its lanterns give out (lamp), how
# they pool (reach) and how far they carry (range_scale); what is not given is as in the first. Every vertex then
# carries a second colour and, at the fine level, a second direction (encode_chunk()), the manifest the second look's
# lanterns and glows (look2): the page goes from one look to the other with a uniform, on the one mesh
LOOK2 = CFG.get( 'look2' ) or None

def bk( level, key ):

	# a bake parameter for one level: BK[ '<level>_<key>' ] when given (--set fine_refine_k=0.35), else BK[ key ]
	return BK.get( level + '_' + key, BK[ key ] )

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

	# -> list of primitives in asset space: dict( P, N, UV, MASK, I, lib, tint, params [, KEEP] )
	# KEEP (bool per vertex, only when the primitive carries the custom attribute _TSQ_KEEP or, the default of the models, TEXCOORD_1.x; > 0.5 = set): a vertex of
	# what the thing is known by - the bold hands and hour marks a generator puts in for a clock's, lettering. It ends up
	# as the must-stay flag of the vertex at every level (encode_chunk()), which the packer never collapses, prunes or
	# sorts out by size. (The GLB is parsed here, not imported by Blender: any accessor type reads the same way.)
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
			KEEP = None
			ka = '_TSQ_KEEP' if '_TSQ_KEEP' in at else 'TEXCOORD_1' if 'TEXCOORD_1' in at else None # TEXCOORD_1.x: the mark as tk.final.mark_keep() writes it by default (the models have no second uv set of their own)
			if ka:

				KEEP = np.asarray( acc( at[ ka ] ), np.float32 ).reshape( n, - 1 )[ :, 0 ] > 0.5 if j[ 'accessors' ][ at[ ka ] ][ 'count' ] == n else None
				if KEEP is None: log( 'WARNING: %s: %s does not have one value per vertex (ignored)' % ( os.path.basename( path ), ka ) )
				elif not KEEP.any(): KEEP = None

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

			prims.append( dict( P = P, N = N, UV = UV, MASK = MASK, I = I, lib = ex.get( 'tsq_lib' ) or 'plain', tint = ex.get( 'tsq_tint' ), params = params, **( dict( KEEP = KEEP ) if KEEP is not None else {} ) ) )

	_GLB[ path ] = prims
	return prims

_MASKED = set()
MASK_T = [ 0.0, 0, 0, 0 ] # seconds, vertices, vertices changed, triangles added

def lod_split( prims, lookup, lmin, lmax, tol, iters ):

	# The reduced models keep a wall as a few large faces whose corners all sit in its creases: the occlusion and dirt
	# of the crease (and the light sampled there) would then run across the whole wall, and the wall is darker than at
	# the level above. An edge longer than lmin is split (midpoints, the same decision for every face on it) while the
	# full model's mask at its middle - or at the middle of the face - is further than tol from what the corners give,
	# or while it is longer than lmax (0: no such limit). prims: the primitives of one reduced model, changed in place.
	prims = [ pr for pr in prims if len( pr[ 'I' ] ) ]
	if not prims: return 0
	off = np.cumsum( [ 0 ] + [ len( pr[ 'P' ] ) for pr in prims ] )
	P = np.concatenate( [ pr[ 'P' ] for pr in prims ] ).astype( np.float32 )
	N = np.concatenate( [ pr[ 'N' ] for pr in prims ] ).astype( np.float32 )
	UV = np.concatenate( [ pr[ 'UV' ] for pr in prims ] ).astype( np.float32 )
	M = np.concatenate( [ pr[ 'MASK' ] for pr in prims ] ).astype( np.float32 )
	K = np.concatenate( [ pr[ 'KEEP' ] if 'KEEP' in pr else np.zeros( len( pr[ 'P' ] ), bool ) for pr in prims ] ) # (_TSQ_KEEP: see load_glb())
	I = np.concatenate( [ pr[ 'I' ].astype( np.int64 ) + off[ i ] for i, pr in enumerate( prims ) ] )
	PR = np.concatenate( [ np.full( len( pr[ 'I' ] ), i, np.int32 ) for i, pr in enumerate( prims ) ] )
	t_in = len( I )
	fresh = np.ones( t_in, bool )
	def unit( v ): return v / np.maximum( np.linalg.norm( v, axis = 1, keepdims = True ), 1e-12 )
	for it in range( iters ):

		ft = np.nonzero( fresh )[ 0 ]
		fresh[ : ] = False
		if not len( ft ): break
		tr = I[ ft ]
		a = P[ tr ].astype( np.float64 )
		el = np.linalg.norm( a[ :, [ 1, 2, 0 ] ] - a, axis = 2 ) # edge k: v_k -> v_k+1
		big = el > lmin
		cs = np.nonzero( big.any( 1 ) )[ 0 ]
		if not len( cs ): break
		cand = ft[ cs ]; tr = tr[ cs ]; el = el[ cs ]; big = big[ cs ]
		e0 = tr; e1 = tr[ :, [ 1, 2, 0 ] ]
		flags = big & ( el > lmax ) if lmax > 0 else np.zeros_like( big )
		ti, ki = np.nonzero( big & ~ flags )
		if len( ti ):

			va = e0[ ti, ki ]; vb = e1[ ti, ki ]
			ok, m = lookup( ( P[ va ].astype( np.float64 ) + P[ vb ] ) * 0.5, unit( N[ va ].astype( np.float64 ) + N[ vb ] ) )
			flags[ ti, ki ] = ok & ( np.abs( m - ( M[ va ] + M[ vb ] ) * 0.5 ).max( 1 ) > tol )

		ci = np.nonzero( big.all( 1 ) & ~ flags.all( 1 ) )[ 0 ]
		if len( ci ):

			v = tr[ ci ]
			ok, m = lookup( P[ v ].astype( np.float64 ).mean( 1 ), unit( N[ v ].astype( np.float64 ).sum( 1 ) ) )
			flags[ ci[ ok & ( np.abs( m - M[ v ].mean( 1 ) ).max( 1 ) > tol ) ] ] = True

		# every face on an edge takes the same decision, whichever vertices of a seam it uses: by position
		_, pid = np.unique( np.round( P.astype( np.float64 ) * 1e4 ).astype( np.int64 ), axis = 0, return_inverse = True )
		pid = pid.ravel().astype( np.int64 ); npid = int( pid.max() ) + 2
		keys = np.minimum( pid[ e0 ], pid[ e1 ] ) * npid + np.maximum( pid[ e0 ], pid[ e1 ] )
		uk, inv = np.unique( keys[ big ], return_inverse = True )
		hit = np.zeros( len( uk ), bool )
		np.logical_or.at( hit, inv, flags[ big ] )
		flags[ big ] = hit[ inv ]
		anyf = flags.any( 1 )
		if not anyf.any(): break
		cand = cand[ anyf ]; flags = flags[ anyf ]; tri = I[ cand ]
		e0 = tri; e1 = tri[ :, [ 1, 2, 0 ] ]
		nv = len( P )
		vk = ( np.minimum( e0, e1 ) * ( nv + 1 ) + np.maximum( e0, e1 ) )[ flags ]
		uk, inv = np.unique( vk, return_inverse = True )
		i0 = uk // ( nv + 1 ); i1 = uk % ( nv + 1 )
		mid = np.full( flags.shape, - 1, np.int64 )
		mid[ flags ] = nv + inv
		nP = ( ( P[ i0 ].astype( np.float64 ) + P[ i1 ] ) * 0.5 ).astype( np.float32 )
		nN = unit( N[ i0 ].astype( np.float64 ) + N[ i1 ] ).astype( np.float32 )
		ok, m = lookup( nP.astype( np.float64 ), nN.astype( np.float64 ) )
		nM = np.where( ok[ :, None ], m, ( M[ i0 ] + M[ i1 ] ) * 0.5 ).astype( np.float32 )
		P = np.concatenate( [ P, nP ] ); N = np.concatenate( [ N, nN ] ); M = np.concatenate( [ M, nM ] ); K = np.concatenate( [ K, K[ i0 ] & K[ i1 ] ] )
		UV = np.concatenate( [ UV, ( ( UV[ i0 ].astype( np.float64 ) + UV[ i1 ] ) * 0.5 ).astype( np.float32 ) ] )
		nf = flags.sum( 1 )
		out_tri = []; out_src = []
		s = np.nonzero( nf == 3 )[ 0 ]
		if len( s ):

			v = tri[ s ]; m = mid[ s ]
			for q in ( ( v[ :, 0 ], m[ :, 0 ], m[ :, 2 ] ), ( m[ :, 0 ], v[ :, 1 ], m[ :, 1 ] ), ( m[ :, 2 ], m[ :, 1 ], v[ :, 2 ] ), ( m[ :, 0 ], m[ :, 1 ], m[ :, 2 ] ) ):

				out_tri.append( np.stack( q, 1 ) ); out_src.append( cand[ s ] )

		s = np.nonzero( nf == 1 )[ 0 ]
		if len( s ):

			k = np.argmax( flags[ s ], 1 ); r = np.arange( len( s ) )
			va = tri[ s ][ r, k ]; vb = tri[ s ][ r, ( k + 1 ) % 3 ]; vc = tri[ s ][ r, ( k + 2 ) % 3 ]; m = mid[ s ][ r, k ]
			out_tri += [ np.stack( ( va, m, vc ), 1 ), np.stack( ( m, vb, vc ), 1 ) ]; out_src += [ cand[ s ], cand[ s ] ]

		s = np.nonzero( nf == 2 )[ 0 ] # k = the edge left whole
		if len( s ):

			k = np.argmin( flags[ s ], 1 ); r = np.arange( len( s ) )
			va = tri[ s ][ r, k ]; vb = tri[ s ][ r, ( k + 1 ) % 3 ]; vc = tri[ s ][ r, ( k + 2 ) % 3 ]
			mb = mid[ s ][ r, ( k + 1 ) % 3 ]; mc = mid[ s ][ r, ( k + 2 ) % 3 ]
			out_tri += [ np.stack( ( vc, mc, mb ), 1 ), np.stack( ( va, vb, mb ), 1 ), np.stack( ( va, mb, mc ), 1 ) ]; out_src += [ cand[ s ] ] * 3

		new_tri = np.concatenate( out_tri ); src = np.concatenate( out_src )
		keep = np.ones( len( I ), bool ); keep[ cand ] = False
		PR = np.concatenate( [ PR[ keep ], PR[ src ] ] )
		I = np.concatenate( [ I[ keep ], new_tri ] )
		fresh = np.zeros( len( I ), bool ); fresh[ int( keep.sum() ): ] = True

	if len( I ) == t_in: return 0
	for i, pr in enumerate( prims ):

		tr = I[ PR == i ]
		used, inv = np.unique( tr.ravel(), return_inverse = True )
		pr[ 'P' ] = np.ascontiguousarray( P[ used ] ); pr[ 'N' ] = np.ascontiguousarray( N[ used ] ); pr[ 'UV' ] = np.ascontiguousarray( UV[ used ] )
		pr[ 'MASK' ] = np.ascontiguousarray( M[ used ] ); pr[ 'I' ] = inv.reshape( - 1, 3 ).astype( np.int32 )
		if 'KEEP' in pr: pr[ 'KEEP' ] = np.ascontiguousarray( K[ used ] )

	return len( I ) - t_in

def lod_masks( key, p ):

	# The reduced models carry the baked occlusion / dirt (_MASK) of whichever vertices the collapse happened to keep:
	# often the hidden twin of a wall's vertex, and the wall turns black at the change of level. Their masks are taken
	# from the full model's surface at the same place, facing the same way - and where a face is too large for its
	# corners to say what the full model has in between, it is split first (lod_split).
	if key in _MASKED: return
	_MASKED.add( key )
	p0 = proto_path( key, p, 0 )
	paths = []
	for l in ( 1, 2 ):

		q = proto_path( key, p, l )
		if q != p0 and q not in [ a for a, _ in paths ] and os.path.exists( q ): paths.append( ( q, l ) )

	if not paths or not os.path.exists( p0 ): return
	t0 = time.time()
	from mathutils.bvhtree import BVHTree
	prims0 = [ pr for pr in load_glb( p0 ) if len( pr[ 'I' ] ) ]
	if not prims0: return
	off = np.cumsum( [ 0 ] + [ len( pr[ 'P' ] ) for pr in prims0 ] )
	P0 = np.concatenate( [ pr[ 'P' ] for pr in prims0 ] ).astype( np.float64 )
	I0 = np.concatenate( [ pr[ 'I' ] + off[ i ] for i, pr in enumerate( prims0 ) ] )
	M0 = np.concatenate( [ pr[ 'MASK' ] for pr in prims0 ] ).astype( np.float64 )
	bvh = BVHTree.FromPolygons( P0.tolist(), I0.tolist() )
	size = float( np.linalg.norm( P0.max( 0 ) - P0.min( 0 ) ) )
	reach = max( 0.05, 0.03 * size )

	def lookup( Pq, Nq ):

		# -> found (n), the full model's mask (n, 4) at the points Pq of a surface facing Nq
		n = len( Pq )
		tri = np.full( n, - 1, np.int64 ); loc = np.zeros( ( n, 3 ) )
		Pl = np.asarray( Pq, np.float64 ).tolist(); Nl = np.asarray( Nq, np.float64 ).tolist()
		for i in range( n ):

			q = Pl[ i ]; nn = Nl[ i ]
			r = bvh.find_nearest( q, reach )
			if r[ 0 ] is None: continue
			fn = r[ 1 ]
			if fn[ 0 ] * nn[ 0 ] + fn[ 1 ] * nn[ 1 ] + fn[ 2 ] * nn[ 2 ] < 0.2:

				# the nearest face looks the other way (the back of a thin part, an inner shell): the nearest that agrees
				best = None
				for c in bvh.find_nearest_range( q, min( reach, r[ 3 ] + 0.04 ) ):

					fn = c[ 1 ]
					if fn[ 0 ] * nn[ 0 ] + fn[ 1 ] * nn[ 1 ] + fn[ 2 ] * nn[ 2 ] >= 0.2 and ( best is None or c[ 3 ] < best[ 3 ] ): best = c

				if best is None: continue
				r = best

			tri[ i ] = r[ 2 ]; loc[ i ] = r[ 0 ]

		ok = tri >= 0
		T = I0[ np.maximum( tri, 0 ) ]
		A = P0[ T[ :, 0 ] ]; e1 = P0[ T[ :, 1 ] ] - A; e2 = P0[ T[ :, 2 ] ] - A; w = loc - A
		d11 = ( e1 * e1 ).sum( 1 ); d12 = ( e1 * e2 ).sum( 1 ); d22 = ( e2 * e2 ).sum( 1 ); w1 = ( w * e1 ).sum( 1 ); w2 = ( w * e2 ).sum( 1 )
		den = np.maximum( d11 * d22 - d12 * d12, 1e-20 )
		v = np.clip( ( d22 * w1 - d12 * w2 ) / den, 0, 1 ); u = np.clip( ( d11 * w2 - d12 * w1 ) / den, 0, 1 )
		sm = np.maximum( 1, u + v ); u /= sm; v /= sm
		return ok, ( M0[ T[ :, 0 ] ] * ( 1 - u - v )[ :, None ] + M0[ T[ :, 1 ] ] * v[ :, None ] + M0[ T[ :, 2 ] ] * u[ :, None ] ).astype( np.float32 )

	nt0 = len( I0 )
	for path, l in paths:

		prims = load_glb( path )
		if sum( len( pr[ 'I' ] ) for pr in prims ) == nt0: continue # no reduced model: the full one under another name
		for pr in prims:

			n = len( pr[ 'P' ] )
			if not n: continue
			ok, new = lookup( pr[ 'P' ], pr[ 'N' ] )
			MASK_T[ 1 ] += n
			M = np.array( pr[ 'MASK' ], np.float32 )
			MASK_T[ 2 ] += int( ( ok & ( np.abs( M[ :, 0 ] - new[ :, 0 ] ) > 0.25 ) ).sum() )
			M[ ok ] = new[ ok ]
			pr[ 'MASK' ] = M

		lmin = BK[ 'lod_split_min' ][ l - 1 ]
		if lmin > 0: MASK_T[ 3 ] += lod_split( prims, lookup, lmin, BK[ 'lod_split_max' ], BK[ 'lod_split_tol' ], BK[ 'lod_split_iters' ] )

	MASK_T[ 0 ] += time.time() - t0

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

TOWN_PATH = CFG.get( 'town' ) or os.path.join( ROOT, 'out/town/town.json' ) # tools/webbake.py --town
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
		self.big = [ 0, 0 ] # how many library layers (from layer 2 on) have 2048 maps, and how many more have large tiles

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
		AUX = np.zeros( n, np.float32 ), EMF = np.zeros( n, np.float32 ), LAY = L_WHITE, FSKY = np.zeros( n, np.float32 ), fk = 0 )
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
		bark = lib == 'bark'
		if foliage or bark:

			# night lighting of trees, as the engine's FoliageLighting: the painted occlusion keeps a floor (the heart of a
			# crown is near black in it), and the sky floor of light_parts() is weighted by how open the point is and by
			# its height above the plant's own foot (a crown stands in the open sky, a hedge in the street's canyon)
			yl = pr[ 'P' ][ :, 1 ].astype( np.float64 )
			high = mix( 0.2, 0.6, sstep( 2.0, 4.5, yl ) ) + 0.8 * sstep( 4.5, 8.5, yl )
			opn = mix( 1, 0.35, mask[ :, 2 ] ) if leafy else mix( 0.5, 1, mask[ :, 0 ] )
			o[ 'FSKY' ] = ( opn * high * ( 0.5 if bark else 1 ) ).astype( np.float32 )
			o[ 'VAO' ] = mix( LOOK[ 'fol_ao' ], 1, vao ).astype( np.float32 )
			o[ 'fk' ] = 4 if bark else 3
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
				o[ 'kind' ] = K_CUT; o[ 'double' ] = False; o[ 'fol' ] = True; o[ 'fk' ] = 1
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
		# a sign (EMF 2: the page draws it brighter, see LOOK[ 'neon' ]): what its asset says - mat( ..., sign = 1 ) for the
		# bulbs and lit letters of one, sign = 0 for a tube that is none - else a neon tube in a colour no flame has (an
		# opal globe is `neon` too, in the hue of its lamp)
		sg = params.get( 'sign' )
		if sg is None: sg = name == 'neon' and max( float( T[ 2 ] - T[ 1 ] ), float( T[ 1 ] - T[ 0 ] ) ) > 0.02 * max( float( np.max( T ) ), 1e-6 )
		o[ 'EMF' ][ : ] = 2 if float( sg ) > 0 else 1

	elif name == 'glass_frosted':

		o[ 'MUL' ] = ( T[ None, : ] * 0.8 * mix( 1, vao, 0.4 )[ :, None ] ).astype( np.float32 )
		dirt = ( 1 - mask[ :, 1 ] * 0.7 ) * 0.975
		o[ 'EM' ][ : ] = T[ None, : ] * ( glow * 0.75 * LOOK[ 'emissive' ] ) * dirt[ :, None ]
		o[ 'EMF' ][ : ] = 1 if glow > 0.25 else 0

	elif name in ( 'glass_clear', 'glass_lantern' ):

		lantern = name == 'glass_lantern'
		# (only the lantern's own amber sheet is lit from behind: a clear pane in front of a flame or a lit dial would turn
		# into a white board - the street clock's face, the mantle of the gas lamps)
		o[ 'two' ] = lantern
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
		# (StainedLighting of the engine) the glass passes the light of a lamp at it whatever its facing: the panes nearest
		# the lamp burn pale and hot, and the glow falls off across the glazing with the lamp's own falloff
		sr = LOOK[ 'stained_reach' ]
		Lp = LIGHTS[ 'pos' ]
		if len( Lp ) and LOOK[ 'stained_back' ] > 0:

			c0 = wp.min( 0 ).astype( np.float64 ); c1 = wp.max( 0 ).astype( np.float64 )
			glass = pane * 0.85 * min( 1.0, max( 0.0, glow * 4 ) ) * LOOK[ 'window_glow' ]
			pale = mix( glass, np.array( [ 1.0, 0.9, 0.62 ] ) * glass.max(), LOOK[ 'stained_pale' ] )
			if LOOK2: o[ 'EM2' ] = o[ 'EM' ].copy()
			for j in np.nonzero( ( ( Lp > c0[ None, : ] - sr ) & ( Lp < c1[ None, : ] + sr ) ).all( 1 ) )[ 0 ]:

				D = Lp[ j ][ None, : ] - wp.astype( np.float64 )
				d2 = ( D * D ).sum( 1 ); d = np.sqrt( d2 )
				near = sstep( sr * sr, sr * sr * 0.25, d2 )
				for li, ( LI, LR, reach ) in enumerate( look_sets() ): # (the second look's lamp: into EM2)

					att = ( d2 / ( d2 + LOOK[ 'source_radius' ] ** 2 ) ) / np.maximum( d2, 0.01 ) * np.clip( 1 - ( d / LR[ j ] ) ** 4, 0, 1 ) ** 2
					att *= 1 + ( LIGHTS[ 'up' ][ j ] - 1 ) * sstep( - 0.42, 0.3, - D[ :, 1 ] / np.maximum( d, 1e-3 ) )
					if reach > 0: att /= 1 + d2 / reach ** 2
					o[ 'EM2' if li else 'EM' ] += ( ( att * near * LOOK[ 'stained_back' ] * LI[ j ] )[ :, None ] * ( LIGHTS[ 'col' ][ j ] * pale )[ None, : ] ).astype( np.float32 )

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
		o[ 'two' ] = True # a sheet of water has no back: lit from either side
		# the fall shader's masks (the engine's geometry contract: _mask.z break-up, _mask.w foam) ride in the layer slot
		# of the record, 4 bits each - a fall samples no texture layer
		o[ 'LAY' ] = ( np.round( np.clip( mask[ :, 2 ], 0, 1 ) * 15 ) + 16 * np.round( np.clip( mask[ :, 3 ], 0, 1 ) * 15 ) ).astype( np.uint16 )
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
		( 'GRP', 1, np.int32 ), ( 'FOL', 1, np.uint8 ), ( 'LIT0', 1, np.uint8 ), ( 'PROP', 1, np.uint8 ),
		( 'FSKY', 1, np.float32 ), ( 'KEEP', 1, np.uint8 ) ] + ( [ ( 'EM2', 3, np.float32 ) ] if LOOK2 else [] ) # EM2: what glows in the second look (LOOK2), FOL: 0 plain, 1 leaf cards, 2 thin pane (lit from either side), 3 hedge, 4 bark; FSKY: weight of the foliage sky floor; KEEP: 1 = _TSQ_KEEP of the model (load_glb())

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
		o[ 'LAY' ] = np.full( n, o[ 'LAY' ], np.uint16 ) if np.ndim( o[ 'LAY' ] ) == 0 else np.asarray( o[ 'LAY' ], np.uint16 )
		if cell is None:

			c = ( wp[ I[ :, 0 ] ] + wp[ I[ :, 1 ] ] + wp[ I[ :, 2 ] ] ) / 3
			tc = cell_ids( c[ :, 0 ], c[ :, 2 ] )

		else: tc = np.full( len( I ), cell, np.int64 )
		grp = np.zeros( n, np.int64 ); ngrp = 1
		keep = np.asarray( pr[ 'KEEP' ], np.uint8 ) if 'KEEP' in pr else np.zeros( n, np.uint8 )
		if np.ptp( o[ 'UV' ], axis = 0 ).max() > UV_SPLIT:

			# a surface that runs over more tiles than the stored uv range: cut into blocks, each with its own offset
			c = I.ravel()
			for k in ( 'UV', 'MUL', 'GLOW', 'EM', 'VAO', 'AUX', 'EMF', 'LAY', 'FSKY' ) + ( ( 'EM2', ) if 'EM2' in o else () ): o[ k ] = o[ k ][ c ]
			wp = wp[ c ]; wn = wn[ c ]; keep = keep[ c ]; n = len( c )
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
			v[ 'EMF' ].append( o[ 'EMF' ] ); v[ 'LAY' ].append( o[ 'LAY' ] ); v[ 'GRP' ].append( ( grp + self.ngrp ).astype( np.int32 ) )
			v[ 'FOL' ].append( np.full( n, o[ 'fk' ] or ( 2 if o.get( 'two' ) else 0 ), np.uint8 ) ); v[ 'FSKY' ].append( o[ 'FSKY' ] ); v[ 'LIT0' ].append( np.full( n, 1 if o.get( 'LIT0' ) else 0, np.uint8 ) )
			v[ 'PROP' ].append( np.full( n, 1 if prop else 0, np.uint8 ) ); v[ 'KEEP' ].append( keep )
			if LOOK2: v[ 'EM2' ].append( o.get( 'EM2', o[ 'EM' ] ) )
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
		if level <= 1 and ( cid & 1 ): continue # far cells: base level only
		path = proto_path( it[ 'p' ], p, level )
		if level > 0 and BK[ 'lod_masks' ]: lod_masks( it[ 'p' ], p )
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
			if level <= 1:

				# the near and fine levels only re-bake the ground of near cells
				c = ( pr[ 'P' ][ pr[ 'I' ][ :, 0 ] ] + pr[ 'P' ][ pr[ 'I' ][ :, 1 ] ] + pr[ 'P' ][ pr[ 'I' ][ :, 2 ] ] ) / 3
				keep = ( cell_ids( c[ :, 0 ], c[ :, 2 ] ) & 1 ) == 0
				if not keep.any(): continue
				pr = dict( pr, I = pr[ 'I' ][ keep ] )

			S.add( pr, pr[ 'P' ], pr[ 'N' ], None, True )

	if missing: log( 'level %d: %d GLBs missing (skipped)' % ( level, missing ) )
	if level > 0 and MASK_T[ 1 ]: log( '  lod masks from the full models: %d vertices, %d changed by more than 0.25, %d triangles added to their large faces (LOD1 + LOD2 models), %.1fs so far' % ( MASK_T[ 1 ], MASK_T[ 2 ], MASK_T[ 3 ], MASK_T[ 0 ] ) )
	return S.close()

# ------------------------------------------------------------------------------------------ lights

def town_lights( lamp, range_scale ):

	L = TOWN.get( 'lights', [] )
	pos = np.array( [ b2t( l[ 'pos' ] ) for l in L ], np.float64 ).reshape( - 1, 3 )
	col = np.array( [ from_srgb( np.array( [ int( ( l.get( 'color' ) or '#ffbf6a' ).lstrip( '#' )[ i:i + 2 ], 16 ) / 255 for i in ( 0, 2, 4 ) ] ) ) for l in L ] ).reshape( - 1, 3 )
	sign = np.array( [ 1 if SIGN_LIGHT.match( str( l.get( 'name' ) or '' ) ) else 0 for l in L ], np.int32 ) # the light of a sign: LOOK[ 'neon' ]
	inten = np.array( [ l.get( 'intensity', 0 ) * lamp for l in L ], np.float64 ) * np.where( sign > 0, float( LOOK.get( 'neon', 1 ) ), 1.0 )
	rng = np.array( [ l.get( 'range' ) or 9 for l in L ], np.float64 ) * float( range_scale ) # (range_scale: a trial of lanterns that carry further than the town declares; 1 = as declared)
	near =np.array( [ l.get( 'near' ) or 0.03 for l in L ], np.float64 )
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

	# src: 1 = the light sits at a visible burner, 0 = a helper light with nothing to look at (shop-front spill, window fill)
	return dict( pos = pos, col = col, I = inten, range = rng, near = near, up = np.array( [ up( l ) for l in L ], np.float64 ),
		src = np.array( [ 0 if l.get( 'src' ) == 0 else 1 for l in L ], np.int32 ), sign = sign )

SIGN_LIGHT = re.compile( r'^neon' )
LIGHTS = town_lights( LOOK[ 'lamp' ], BK.get( 'range_scale', 1 ) )
# the second look's lanterns, and the two as the lighting pass reads them: LOOK and LIGHTS hold the first look's unless
# use_look( 1 ) has put the second's there (the workers of a pass are forked after that and see what was set)
LIGHTS2 = town_lights( LOOK2.get( 'lamp', LOOK[ 'lamp' ] ), LOOK2.get( 'range_scale', BK.get( 'range_scale', 1 ) ) ) if LOOK2 else None
_LOOK1 = ( dict( LIGHTS ), LOOK[ 'lamp' ], LOOK.get( 'reach', 0 ) )
LOOK_I = [ 0 ]

def use_look( i ):

	L, lamp, reach = ( LIGHTS2, LOOK2.get( 'lamp', _LOOK1[ 1 ] ), LOOK2.get( 'reach', _LOOK1[ 2 ] ) ) if i else _LOOK1
	LIGHTS.update( L ); LOOK[ 'lamp' ] = lamp; LOOK[ 'reach' ] = reach; LOOK_I[ 0 ] = i

def look_sets():

	# -> ( the lanterns' intensities, their ranges, the reach ) of the first look and, when there is one, of the second
	return [ ( _LOOK1[ 0 ][ 'I' ], _LOOK1[ 0 ][ 'range' ], _LOOK1[ 2 ] ) ] + ( [ ( LIGHTS2[ 'I' ], LIGHTS2[ 'range' ], LOOK2.get( 'reach', _LOOK1[ 2 ] ) ) ] if LOOK2 else [] )

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

def refine( S, lname = '' ):

	# Large faces near lanterns are split (edge midpoints, conforming) until an edge is no longer than k * its
	# distance to the lantern: vertex lighting then carries the pools of light on walls and paving.
	V, Tt = S.v, S.t
	Lp = LIGHTS[ 'pos' ].astype( np.float32 )
	Ilum = LIGHTS[ 'I' ] * lum( LIGHTS[ 'col' ] )
	rsig = np.minimum( LIGHTS[ 'range' ], np.sqrt( np.maximum( Ilum, 0 ) / bk( lname, 'refine_emin' ) ) ).astype( np.float32 )
	# (one mesh for both looks: split as far as either look's lantern asks)
	if LOOK2: rsig = np.maximum( rsig, np.minimum( LIGHTS2[ 'range' ], np.sqrt( np.maximum( LIGHTS2[ 'I' ] * lum( LIGHTS2[ 'col' ] ), 0 ) / bk( lname, 'refine_emin' ) ) ).astype( np.float32 ) )
	use = rsig > bk( lname, 'refine_min' )
	Lp, rsig = Lp[ use ], rsig[ use ]
	K, LMIN = bk( lname, 'refine_k' ), bk( lname, 'refine_min' )
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

	for it in range( bk( lname, 'refine_iters' ) ):

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
			elif kname == 'KEEP': new = arr[ i0 ] & arr[ i1 ]
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

_POOLS = {}
def light_pools( bvh ):

	# The pool of light under every lantern as one small lamp lying on the paving and shining upwards (one bounce):
	# position, colour x intensity straight up, luminance of that, (height of the lantern)^2 = the pool's size.
	# The light a lantern at height h puts on the ground is 2 pi I x f, f = 1 - ( h / a ) x ( pi / 2 - atan( h / a ) )
	# with the reach falloff a (1 without); a diffuse ground of albedo 1 sends I' = 2 I f x cos straight back up.
	k = ( id( bvh ), LOOK_I[ 0 ] )
	if k in _POOLS: return _POOLS[ k ]
	rc = bvh.ray_cast
	a = LOOK.get( 'reach', 0 )
	pos = []; col = []; h2 = []
	for i in range( len( LIGHTS[ 'pos' ] ) ):

		if LIGHTS[ 'src' ][ i ] != 1 or LIGHTS[ 'I' ][ i ] <= 0: continue
		p = [ float( v ) for v in LIGHTS[ 'pos' ][ i ] ]
		h = 0.0
		for dx, dz in ( ( 0.5, 0 ), ( - 0.5, 0 ), ( 0, 0.5 ), ( 0, - 0.5 ) ): # (past its own post and bracket)

			r = rc( ( p[ 0 ] + dx, p[ 1 ], p[ 2 ] + dz ), ( 0, - 1, 0 ), 12.0 )
			if r[ 0 ] is not None: h = max( h, r[ 3 ] )

		if h < 1.0: continue
		f = 1 - ( h / a ) * ( math.pi / 2 - math.atan( h / a ) ) if a > 0 else 1.0
		pos.append( ( p[ 0 ], p[ 1 ] - h + 0.05, p[ 2 ] ) ); h2.append( h * h )
		col.append( LIGHTS[ 'col' ][ i ] * LIGHTS[ 'I' ][ i ] * 2 * f )

	if not pos: _POOLS[ k ] = None; return None
	col = np.array( col, np.float32 )
	_POOLS[ k ] = dict( pos = np.array( pos, np.float32 ), col = col, lum = lum( col ).astype( np.float32 ), h2 = np.array( h2, np.float32 ) )
	return _POOLS[ k ]

def light_range( bvh, P, N, FOL, s, e, Ed, AO, Gd = None, Sh = None ):

	# direct irradiance (lanterns + moon, shadowed) and sky visibility of vertices s..e
	# Gd: where that light comes from - the sum of luminance x direction over the lights a vertex faces, so that the
	# luminance of the direct light on a normal n' near the vertex's own is n' . G (see light_dir())
	# Sh: what the lanterns' shadows took (3 columns: their light without shadows less what arrived; 6: and that light
	# without shadows) - ground_weight() gives it back to the ground's vertices, ground_shadows() makes its picture of it
	rc = bvh.ray_cast
	Lp = LIGHTS[ 'pos' ].astype( np.float32 ); Lc = ( LIGHTS[ 'col' ] * LIGHTS[ 'I' ][ :, None ] ).astype( np.float32 )
	Ll = lum( Lc ).astype( np.float32 ); Lr = LIGHTS[ 'range' ].astype( np.float32 ); Lup = LIGHTS[ 'up' ].astype( np.float32 )
	Lskip = np.maximum( BK[ 'light_skip' ], LIGHTS[ 'near' ] + 0.25 ).astype( np.float32 )
	sr2 = LOOK[ 'source_radius' ] ** 2; reach2 = np.float32( LOOK.get( 'reach', 0 ) ** 2 )
	leak = BK[ 'shadow_leak' ]
	rs = np.random.RandomState( 12345 )
	JIT = rs.normal( size = ( 4096, 3 ) ).astype( np.float32 )
	JIT *= ( BK[ 'light_jitter' ] * rs.uniform( 0.3, 1, 4096 ) ** ( 1 / 3 ) / np.linalg.norm( JIT, axis = 1 ) )[ :, None ].astype( np.float32 )
	md = np.array( LOOK[ 'moon_dir' ], np.float64 ); md /= np.linalg.norm( md )
	mcol = np.array( LOOK[ 'moon_col' ], np.float32 ) * LOOK[ 'moon' ]
	H = _hemi_dirs( BK[ 'ao_rays' ] )
	KR = BK[ 'light_rays' ]
	B = 2048
	# foliage and bark (FoliageLighting of the engine), by FOL code: wrap and transmitted share of a light, tint of what
	# passed through a blade, share of the paving's bounce
	fw = np.array( [ 0, 1.0, 0, 0.6, 0.8 ], np.float32 ) * LOOK[ 'fol_wrap' ]
	fb = np.array( [ 0, 1.0, 0, 0.6, 0.0 ], np.float32 ) * LOOK[ 'fol_back' ]
	fbn = np.array( [ 0, 1.0, 0, 1.0, 0.5 ], np.float32 ) * LOOK[ 'fol_bounce' ]
	ftint = np.array( [ [ 1, 1, 1 ], [ 1, 0.95, 0.6 ], [ 1, 1, 1 ], [ 1, 0.95, 0.6 ], [ 1, 1, 1 ] ], np.float32 )
	bcol = np.array( [ 1.0, 0.84, 0.62 ], np.float32 )
	md32 = md.astype( np.float32 )
	VP = light_pools( bvh ) if BK[ 'bounce' ] > 0 else None
	for b0 in range( s, e, B ):

		b1 = min( e, b0 + B )
		n = N[ b0:b1 ]; p = P[ b0:b1 ] + n * 0.03
		fk = FOL[ b0:b1 ]
		leaf = fk == 1
		fol = leaf | ( fk >= 3 ) # lit as foliage: leaf cards, hedges, bark
		two = fk == 2 # thin panes: lit from either side (a lantern's glass glows with the flame behind it)
		kw = fw[ fk ]; kb = fb[ fk ]; tint = ftint[ fk ]
		# (how much a point faces the paving: a crown of cards has no normal that says so)
		kbn = fbn[ fk ] * ( 0.4 + 0.6 * np.where( leaf, 0.7, 0.5 - 0.5 * n[ :, 1 ] ) )
		hv = pos_hash( P[ b0:b1 ] ) # sampling pattern by position: coincident vertices come out identical and weld
		m = b1 - b0
		E = np.zeros( ( m, 3 ), np.float32 )
		G = np.zeros( ( m, 3 ), np.float32 )
		if Sh is not None: Sh[ b0:b1 ] = 0
		if len( Lp ):

			D = Lp[ None, :, : ] - p[ :, None, : ]
			d2 = np.einsum( 'vlk,vlk->vl', D, D ); d = np.sqrt( d2 )
			nl = np.einsum( 'vlk,vk->vl', D, n ) / np.maximum( d, 1e-6 )
			# foliage: a card stands for a spray of blades at every angle and a blade lets light through - every card takes
			# a share of each light whatever its facing (wrap), more when lit from behind (back)
			lam = np.clip( nl, 0, 1 )
			wr = ( 1 - lam ) * kw[ :, None ] + np.clip( - nl, 0, 1 ) * kb[ :, None ]
			ndl = np.where( fol[ :, None ], lam + wr, np.where( two[ :, None ], np.abs( nl ), nl ) )
			# the lantern's cap (capFactor of ClusteredLightsNode.js): full light below ~25 degrees under its horizon, `up` above
			above = sstep( - 0.42, 0.3, - D[ :, :, 1 ] / np.maximum( d, 1e-3 ) )
			cap = 1 + ( Lup[ None, : ] - 1 ) * above
			geo = ( d2 / ( d2 + sr2 ) ) / np.maximum( d2, 0.01 ) * np.clip( 1 - ( d / Lr[ None, : ] ) ** 4, 0, 1 ) ** 2 * cap
			if reach2 > 0:

				# reachFactor of ClusteredLightsNode.js. What hangs above the lamp head is no part of a pool of light on the
				# paving: foliage there keeps part of what the cut took
				R = 1 / ( 1 + d2 / reach2 )
				geo = geo * np.where( fol[ :, None ], mix( R, mix( 1, R, LOOK[ 'fol_reach' ] ), above ), R )

			w = geo * np.clip( ndl, 0, None )
			# what the cap withheld went down onto the paving and part of it comes back up, warm and diffuse (foliage only)
			bn = geo * ( 1 - cap ) / np.maximum( cap, 0.05 ) * kbn[ :, None ]
			wl = ( w + bn ) * Ll[ None, : ]
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
				vis = np.where( fol[ vi ], BK[ 'fol_vis' ] + ( 1 - BK[ 'fol_vis' ] ) * vis, vis )
				gp = geo[ vi, lj ] * ( leak + ( 1 - leak ) * vis )
				cv = np.where( fol[ vi ][ :, None ], lam[ vi, lj ][ :, None ] + tint[ vi ] * wr[ vi, lj ][ :, None ], np.clip( ndl[ vi, lj ], 0, None )[ :, None ] )
				np.add.at( E, vi, ( gp[ :, None ] * cv + bn[ vi, lj ][ :, None ] * bcol[ None, : ] ).astype( np.float32 ) * Lc[ lj ] )
				if Sh is not None:

					T = np.zeros( ( m, Sh.shape[ 1 ] ), np.float32 )
					np.add.at( T[ :, :3 ], vi, ( ( geo[ vi, lj ] - gp )[ :, None ] * cv ).astype( np.float32 ) * Lc[ lj ] )
					if Sh.shape[ 1 ] > 3: np.add.at( T[ :, 3: ], vi, ( geo[ vi, lj ][ :, None ] * cv + bn[ vi, lj ][ :, None ] * bcol[ None, : ] ).astype( np.float32 ) * Lc[ lj ] )
					Sh[ b0:b1 ] = T

				if Gd is not None:

					fr = nl[ vi, lj ] > 0 # (the part of the light that turns with the normal)
					np.add.at( G, vi[ fr ], ( gp[ fr ] * Ll[ lj[ fr ] ] / np.maximum( d[ vi[ fr ], lj[ fr ] ], 1e-6 ) )[ :, None ].astype( np.float32 ) * D[ vi[ fr ], lj[ fr ] ] )

		# moon
		nm = n @ md32
		lam = np.clip( nm, 0, 1 ); wr = ( 1 - lam ) * kw + np.clip( - nm, 0, 1 ) * kb
		ndm = np.where( fol, lam + wr, np.where( two, np.abs( nm ), nm ) )
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
			cv = np.where( fol[ mi ][ :, None ], lam[ mi ][ :, None ] + tint[ mi ] * wr[ mi ][ :, None ], ndm[ mi ][ :, None ] )
			E[ mi ] += ( cv * ( 0.04 + 0.96 * vis )[ :, None ] * mcol[ None, : ] ).astype( np.float32 )
			G[ mi ] += ( float( lum( mcol ) ) * ( 0.04 + 0.96 * vis ) * ( nm[ mi ] > 0 ) )[ :, None ].astype( np.float32 ) * md32[ None, : ]

		Eb = Gb = None
		if VP is not None:

			# one bounce: every pool as a diffuse lamp on the paving, as wide as its lantern is high. No shadow ray - what
			# stands in the way is left to the occlusion of the vertex (below). Foliage has its own bounce term (bn).
			Dv = VP[ 'pos' ][ None, :, : ] - p[ :, None, : ]
			dv2 = np.einsum( 'vlk,vlk->vl', Dv, Dv ); dv = np.sqrt( dv2 ) + 1e-6
			cu = np.clip( - Dv[ :, :, 1 ] / dv, 0, 1 ) # the pool's cosine
			cr = np.clip( ( np.einsum( 'vlk,vk->vl', Dv, n ) / dv + 0.25 ) / 1.25, 0, 1 ) # the vertex's, wrapped: a pool is wide
			g = cu * cr / ( dv2 + VP[ 'h2' ][ None, : ] ) * np.clip( 1 - ( dv / BK[ 'bounce_range' ] ) ** 4, 0, 1 ) ** 2
			g *= ( BK[ 'bounce' ] * ( fk == 0 ) )[ :, None ]
			Eb = ( g @ VP[ 'col' ] ) * bcol[ None, : ]
			Gb = np.einsum( 'vl,vlk->vk', g * VP[ 'lum' ][ None, : ] / dv, Dv ) * float( lum( bcol ) )

		Ed[ b0:b1 ] = E
		if Gd is not None: G[ leaf | two ] = 0; Gd[ b0:b1 ] = G
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
		if Eb is not None:

			Ed[ b0:b1 ] += ( Eb * ao[ :, None ] ).astype( np.float32 )
			if Gd is not None: Gd[ b0:b1 ] += ( Gb * ao[ :, None ] ).astype( np.float32 )

		AO[ b0:b1 ] = np.where( leaf | ( fk == 3 ), 1 - ( 1 - ao ) * ( 1 - LOOK[ 'fol_gtao' ] ), ao ) # (leaves: part of the occlusion is left off)

def light_soup( S, tag ):

	V, Tt = S.v, S.t
	occ = Tt[ 'OCC' ]
	t0 = time.time()
	bvh = make_bvh( V[ 'P' ], Tt[ 'TRI' ][ occ ] )
	log( '  bvh: %d occluder tris, %.1fs' % ( int( occ.sum() ), time.time() - t0 ) )
	if BK[ 'bounce' ] > 0:

		vp = light_pools( bvh )
		log( '  one bounce: %d pools of light on the paving (lanterns %.1f m up on average)' % ( ( len( vp[ 'pos' ] ), float( np.sqrt( vp[ 'h2' ] ).mean() ) ) if vp else ( 0, 0 ) ) )

	return bvh

def run_lighting( bvh, P, N, FOL, need, tag, with_dir = False, sh = 0 ):

	# fork-parallel over the vertices that need light; results come back through shared memory maps
	# -> E, A (, G with with_dir) (, what the shadows took with sh = 3 or 6 columns: see light_range())
	idx = np.nonzero( need )[ 0 ]
	n = len( idx )
	Pn = np.ascontiguousarray( P[ idx ] ); Nn = np.ascontiguousarray( N[ idx ] ); Fn = np.ascontiguousarray( FOL[ idx ] )
	fE = os.path.join( WORK, 'webbake_%d_%s_E.bin' % ( os.getpid(), tag ) ); fA = os.path.join( WORK, 'webbake_%d_%s_A.bin' % ( os.getpid(), tag ) )
	Ed = np.memmap( fE, np.float32, 'w+', shape = ( max( n, 1 ), 3 ) ); AO = np.memmap( fA, np.float32, 'w+', shape = ( max( n, 1 ), ) )
	fG = os.path.join( WORK, 'webbake_%d_%s_G.bin' % ( os.getpid(), tag ) )
	Gd = np.memmap( fG, np.float32, 'w+', shape = ( max( n, 1 ), 3 ) ) if with_dir else None
	fS = os.path.join( WORK, 'webbake_%d_%s_S.bin' % ( os.getpid(), tag ) )
	Sd = np.memmap( fS, np.float32, 'w+', shape = ( max( n, 1 ), sh ) ) if sh else None
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

				for k in range( j, len( slices ), jobs ): light_range( bvh, Pn, Nn, Fn, slices[ k ][ 0 ], slices[ k ][ 1 ], Ed, AO, Gd, Sd )
				Ed.flush(); AO.flush()
				if Gd is not None: Gd.flush()
				if Sd is not None: Sd.flush()

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
	G = None
	if with_dir: G = np.zeros( ( len( P ), 3 ), np.float32 ); G[ idx ] = Gd[ :n ]
	SH = None
	if sh: SH = np.zeros( ( len( P ), sh ), np.float32 ); SH[ idx ] = Sd[ :n ]
	del Ed, AO, Gd, Sd
	os.remove( fE ); os.remove( fA )
	if with_dir: os.remove( fG )
	if sh: os.remove( fS )
	log( '  lighting: %d verts, %d workers, %.1fs' % ( n, jobs, time.time() - t0 ) )
	return ( E, A ) + ( ( G, ) if with_dir else () ) + ( ( SH, ) if sh else () )

def gsm_band():

	# the heights over the terrain (m) between which a surface that faces up is the ground of the shadow picture: full
	# from the second to the third, none below the first and above the last. The top is under gsm_floor - what lies
	# higher is not the floor ground_shadows() finds, and keeps its shadows in its vertices. The page weighs every
	# pixel by exactly this (manifest: gsh.band), ground_weight() every vertex
	return [ - 0.3, - 0.15, BK[ 'gsm_floor' ] - 0.05, BK[ 'gsm_floor' ] - 0.01 ]

def ground_weight( S, need ):

	# -> per vertex, how much of it is ground to the shadow picture (0..1): it faces up (or down) and lies in gsm_band()
	# about the terrain - the triangles of the ground, the topmost at its place. The same test the page makes per pixel
	# (BakedTown._materials of web/engine/lite.js: `band` x `up` of the shadows); a vertex takes back that share of the
	# light the lanterns' shadows took from it, and the page takes it away again, per pixel, by the picture.
	V, Tt = S.v, S.t
	P = V[ 'P' ]
	w = np.zeros( len( P ), np.float32 )
	tt = Tt[ 'TRI' ][ Tt[ 'TERR' ] ]
	if not len( tt ): return w
	t0 = time.time()
	b = gsm_band()
	up = sstep( 0.5, 0.8, np.abs( V[ 'N' ][ :, 1 ] ) )
	isg = np.zeros( len( P ), bool ); isg[ tt.ravel() ] = True
	y0 = float( P[ isg, 1 ].min() ); y1 = float( P[ isg, 1 ].max() )
	# (only what the page draws with the ground's factor: opaque triangles - no leaf, no pane, no water)
	opq = np.zeros( len( P ), bool ); opq[ Tt[ 'TRI' ][ Tt[ 'KIND' ] == K_OPAQUE ].ravel() ] = True
	opq[ Tt[ 'TRI' ][ Tt[ 'KIND' ] != K_OPAQUE ].ravel() ] = False
	cand = np.nonzero( need & opq & ( V[ 'FOL' ] == 0 ) & ( up > 0 ) & ( P[ :, 1 ] > y0 + b[ 0 ] ) & ( P[ :, 1 ] < y1 + b[ 3 ] ) )[ 0 ]
	gb = make_bvh( P, tt )
	rc = gb.ray_cast
	X = P[ cand, 0 ].tolist(); Y = P[ cand, 1 ].tolist(); Z = P[ cand, 2 ].tolist(); own = isg[ cand ].tolist()
	dh = np.full( len( cand ), 9.0, np.float32 )
	down = ( 0.0, - 1.0, 0.0 )
	for i in range( len( cand ) ):

		r = rc( ( X[ i ], 60.0, Z[ i ] ), down )
		if r[ 0 ] is not None: dh[ i ] = Y[ i ] - r[ 0 ][ 1 ]
		elif own[ i ]: dh[ i ] = 0.0 # (a vertex on the rim of the terrain: the ray down passes beside it)

	w[ cand ] = up[ cand ] * sstep( b[ 0 ], b[ 1 ], dh ) * ( 1 - sstep( b[ 2 ], b[ 3 ], dh ) )
	del gb
	log( '  ground of the shadow picture: %d of %d vertices that face up near the terrain (%d of them the terrain\'s own), %.1fs' % ( int( ( w > 0.5 ).sum() ), len( cand ), int( ( w[ isg ] > 0.5 ).sum() ), time.time() - t0 ) )
	return w

def _gsm_rows( j0, j1, G, gb, bvh, Y, IN ):

	# per texel of rows j0..j1: the height of the point its light is taken at (nan: no terrain there) and whether that
	# point is inside something solid - what it sees first upwards and to the four sides is mostly the back of a face
	W = G[ 'w' ]; x0 = G[ 'x0' ]; z0 = G[ 'z0' ]; s = G[ 'texel' ]; fl = G[ 'floor' ]; lift = G[ 'lift' ]
	down = ( 0.0, - 1.0, 0.0 ); gr = gb.ray_cast; rc = bvh.ray_cast
	probes = ( ( ( 0.0, 1.0, 0.0 ), 40.0 ), ( ( 1.0, 0.0, 0.0 ), 1.5 ), ( ( - 1.0, 0.0, 0.0 ), 1.5 ), ( ( 0.0, 0.0, 1.0 ), 1.5 ), ( ( 0.0, 0.0, - 1.0 ), 1.5 ) )
	for j in range( j0, j1 ):

		z = z0 + ( j + 0.5 ) * s
		y = np.full( W, np.nan, np.float32 ); ins = np.zeros( W, np.uint8 )
		for i in range( W ):

			x = x0 + ( i + 0.5 ) * s
			r = gr( ( x, 60.0, z ), down )
			if r[ 0 ] is None: continue
			h0 = r[ 0 ][ 1 ]
			r = rc( ( x, h0 + fl, z ), down, fl + 0.12 )
			yy = ( r[ 0 ][ 1 ] if r[ 0 ] is not None else h0 ) + lift
			y[ i ] = yy
			o = ( x, yy, z ); back = 0
			for d, t in probes:

				r = rc( o, d, t )
				if r[ 0 ] is not None and r[ 1 ][ 0 ] * d[ 0 ] + r[ 1 ][ 1 ] * d[ 1 ] + r[ 1 ][ 2 ] * d[ 2 ] > 0: back += 1

			if back >= 3: ins[ i ] = 1

		Y[ j ] = y; IN[ j ] = ins

def _box( a, r ):

	# mean over ( 2 r + 1 )^2 texels, the rim repeated
	if r <= 0: return a
	for ax in ( 0, 1 ):

		p = np.pad( a, [ ( r + 1, r ) if k == ax else ( 0, 0 ) for k in ( 0, 1 ) ], mode = 'edge' ).astype( np.float64 )
		c = np.cumsum( p, axis = ax )
		a = ( ( c[ 2 * r + 1: ] - c[ :- 2 * r - 1 ] ) if ax == 0 else ( c[ :, 2 * r + 1: ] - c[ :, :- 2 * r - 1 ] ) ) / ( 2 * r + 1 )

	return a.astype( np.float32 )

def ground_shadows( S, bvh, raw ):

	# The lanterns' shadows on the ground, as a picture of the town from above (BK gsm*): <raw>/gsm_v.bin, gsm_f.bin
	# (a byte per texel, rows from z0) and gsm.json. The ground has a vertex every 0.6 m at best, so the shadow of a
	# lamp post, a bench or a railing in its vertices is a smudge - and another one at every level. Here it is taken
	# per texel, by the bake's own lighting (light_range(): the same falloff, cap, leak and rays spread over the
	# mantle, so the edge is as soft as the lantern is wide), at the floor point tools/aomap.py takes its occlusion at:
	#   v = luminance of the lanterns' light that arrives / of their light without shadows  (gsm_v: sqrt( v ) x 255)
	#   f = luminance of the fill - moon, sky, the pools' bounce, as the vertices hold them - / of fill + lanterns'
	#       light without shadows  (gsm_f: sqrt( f ) x 255)
	# and two hues for the whole town, cF (fill) and cD (lanterns), of luminance 1. The ground's vertices hold fill +
	# unshadowed lanterns (ground_weight()), so a pixel's light is its vertices' times
	#   ( f cF + ( 1 - f ) v cD ) / ( f cF + ( 1 - f ) cD )        per channel: a shadow is not only darker, it is the fill's blue
	V, Tt = S.v, S.t
	tt = Tt[ 'TRI' ][ Tt[ 'TERR' ] ]
	if not len( tt ): return
	t0 = time.time()
	s = float( BK[ 'gsm_texel' ] )
	Q = V[ 'P' ][ np.unique( tt ) ]
	x0 = round( float( Q[ :, 0 ].min() ) / s ) * s; z0 = round( float( Q[ :, 2 ].min() ) / s ) * s
	W = int( round( ( float( Q[ :, 0 ].max() ) - x0 ) / s ) ); H = int( round( ( float( Q[ :, 2 ].max() ) - z0 ) / s ) )
	G = dict( w = W, h = H, x0 = x0, z0 = z0, texel = s, floor = float( BK[ 'gsm_floor' ] ), lift = float( BK[ 'gsm_lift' ] ) )
	gb = make_bvh( V[ 'P' ], tt )
	fy = os.path.join( WORK, 'webbake_%d_gsm_y.bin' % os.getpid() ); fi = os.path.join( WORK, 'webbake_%d_gsm_in.bin' % os.getpid() )
	Y = np.memmap( fy, np.float32, 'w+', shape = ( H, W ) ); IN = np.memmap( fi, np.uint8, 'w+', shape = ( H, W ) )
	step = 8
	sl = [ ( j, min( H, j + step ) ) for j in range( 0, H, step ) ]
	jobs = max( 1, JOBS ); pids = []
	for k in range( jobs ):

		pid = os.fork() if jobs > 1 else 0
		if pid == 0:

			code = 0
			try:

				for q in range( k, len( sl ), jobs ): _gsm_rows( sl[ q ][ 0 ], sl[ q ][ 1 ], G, gb, bvh, Y, IN )
				Y.flush(); IN.flush()

			except BaseException:

				import traceback; traceback.print_exc(); code = 1

			if jobs > 1: os._exit( code )

		else: pids.append( pid )

	bad = 0
	for pid in pids:

		_, st = os.waitpid( pid, 0 )
		if st != 0: bad += 1

	if bad: raise RuntimeError( '%d workers of the shadow picture failed' % bad )
	yy = np.array( Y ); ins = np.array( IN ) > 0
	del Y, IN, gb
	os.remove( fy ); os.remove( fi )
	has = np.isfinite( yy )
	jj, ii = np.nonzero( has )
	n = len( jj )
	log( '  shadow picture: %d x %d texels of %.4f m, %d on the ground (%d inside something), %.1fs' % ( W, H, s, n, int( ( ins & has ).sum() ), time.time() - t0 ) )
	# (light_range() lifts a point 0.03 along its normal)
	P = np.stack( [ x0 + ( ii + 0.5 ) * s, yy[ jj, ii ] - 0.03, z0 + ( jj + 0.5 ) * s ], 1 ).astype( np.float32 )
	N = np.zeros( ( n, 3 ), np.float32 ); N[ :, 1 ] = 1
	keep = ( BK[ 'light_rays' ], BK[ 'ao_rays' ] )
	BK[ 'light_rays' ] = int( BK[ 'gsm_rays' ] ); BK[ 'ao_rays' ] = int( BK[ 'gsm_ao_rays' ] )
	try: E, A, SH = run_lighting( bvh, P, N, np.zeros( n, np.uint8 ), np.ones( n, bool ), 'gsm', False, 6 )
	finally: BK[ 'light_rays' ], BK[ 'ao_rays' ] = keep
	del P, N
	U = SH[ :, 3: ]; T = SH[ :, :3 ]
	lu = lum( U ); lt = lum( T )
	# the fill as a vertex of the ground holds it: moon and bounce (what is left of E), the sky's share by the occlusion
	F = ( E - ( U - T ) ) + ( np.array( LOOK[ 'sky' ], np.float32 ) * LOOK[ 'ambient' ] )[ None, : ] * A[ :, None ]
	lf = np.maximum( lum( F ), 1e-6 )
	wt = np.maximum( lt, 0 ).astype( np.float64 )
	hue = lambda C, l: ( C.astype( np.float64 ) * ( wt / np.maximum( l, 1e-6 ) )[ :, None ] ).sum( 0 )
	cD = hue( U, lu ); cF = hue( F, lf )
	cD = ( cD / max( float( lum( cD ) ), 1e-9 ) ) if wt.sum() > 0 else np.ones( 3 ); cF = ( cF / max( float( lum( cF ) ), 1e-9 ) ) if wt.sum() > 0 else np.ones( 3 )
	v = np.ones( ( H, W ), np.float32 ); v[ jj, ii ] = np.where( lu > 1e-5, 1 - lt / np.maximum( lu, 1e-9 ), 1 )
	LF = np.zeros( ( H, W ), np.float32 ); LF[ jj, ii ] = lf
	LU = np.zeros( ( H, W ), np.float32 ); LU[ jj, ii ] = lu
	del E, A, SH, U, T, F
	# inside a solid (under a crate, in a wall): the neighbours' values, grown inwards - a filtered read at its foot
	# is then the paving's own and no dark line of the map's making runs along a lit wall
	ok = has & ~ ins
	wk = ok.astype( np.float32 ); va = v * wk; fa = LF * wk
	for it in range( int( BK[ 'gsm_grow' ] ) ):

		pw = np.pad( wk, 1 ); pv = np.pad( va, 1 ); pf = np.pad( fa, 1 )
		nb = lambda p: p[ :- 2, 1:- 1 ] + p[ 2:, 1:- 1 ] + p[ 1:- 1, :- 2 ] + p[ 1:- 1, 2: ]
		nw = nb( pw ); grow = ( wk == 0 ) & ( nw > 0 ) & has
		va[ grow ] = nb( pv )[ grow ] / nw[ grow ]; fa[ grow ] = nb( pf )[ grow ] / nw[ grow ]; wk[ grow ] = 1

	grown = ( wk > 0 ) & ~ ok
	v[ grown ] = va[ grown ]; LF[ grown ] = fa[ grown ]
	# (the fill is smoothed over the texels that have one: the ground, and not the inside of what stands on it)
	hf = ( wk > 0 ).astype( np.float32 )
	del va, fa, wk
	p = np.pad( v, 1, mode = 'edge' )
	p = ( p[ :, :- 2 ] + 2 * p[ :, 1:- 1 ] + p[ :, 2: ] ) / 4
	v = ( p[ :- 2 ] + 2 * p[ 1:- 1 ] + p[ 2: ] ) / 4
	r = int( BK[ 'gsm_fill' ] )
	hw = _box( _box( hf, r ), r )
	LFs = np.where( hw > 1e-4, _box( _box( LF * hf, r ), r ) / np.maximum( hw, 1e-4 ), LF )
	f = np.where( has, LFs / np.maximum( LFs + LU, 1e-9 ), 1 )
	leak = float( BK[ 'shadow_leak' ] )
	v = np.clip( v, leak, 1 ); f = np.clip( f, 0, 1 )
	( np.sqrt( v ) * 255 + 0.5 ).astype( np.uint8 ).tofile( os.path.join( raw, 'gsm_v.bin' ) )
	( np.sqrt( f ) * 255 + 0.5 ).astype( np.uint8 ).tofile( os.path.join( raw, 'gsm_f.bin' ) )
	lit = has & ( LU > 0.02 )
	info = dict( G, band = [ round( float( x ), 4 ) for x in gsm_band() ], rays = int( BK[ 'gsm_rays' ] ), leak = leak,
		cF = [ round( float( x ), 4 ) for x in cF ], cD = [ round( float( x ), 4 ) for x in cD ],
		lit = round( float( lit.sum() ) / max( int( has.sum() ), 1 ), 4 ), shadowed = round( float( ( v[ lit ] < 0.5 ).sum() ) / max( int( lit.sum() ), 1 ), 4 ), mean = round( float( v[ lit ].mean() ) if lit.any() else 1.0, 4 ) )
	json.dump( info, open( os.path.join( raw, 'gsm.json' ), 'w' ) )
	log( '  shadow picture: %.0f %% of the ground in lantern light, %.0f %% of that more than half in shadow (%.2f of the light arrives on average); hues: fill %s, lanterns %s; %.1fs' % (
		100 * info[ 'lit' ], 100 * info[ 'shadowed' ], info[ 'mean' ], info[ 'cF' ], info[ 'cD' ], time.time() - t0 ) )

def light_parts( S, E, A, N = None ):

	# -> the direct and the ambient irradiance of every vertex as they go into its colour (N: other normals than its own)
	V = S.v
	n = V[ 'N' ] if N is None else N
	sky = np.array( LOOK[ 'sky' ], np.float32 ); gnd = np.array( LOOK[ 'ground' ], np.float32 )
	# (the sky half of the fill grows from the eaves up, as in the engine: streets are canyons, roofs see all of the sky)
	skyv = sky[ None, : ] * ( 1 + LOOK[ 'roof_fill' ] * sstep( 5.5, 13, V[ 'P' ][ :, 1 ] ) )[ :, None ]
	hemi = mix( gnd[ None, : ], skyv, ( 0.5 * n[ :, 1 ] + 0.5 )[ :, None ] ) * LOOK[ 'ambient' ]
	vao = V[ 'VAO' ]
	# foliage and bark: a floor of sky / moon light on top of the fill, strongest on what faces up, a transmitted share
	# below; thinned towards the heart of the crown and growing with height (FSKY), but not by the occlusion
	up = 0.5 * n[ :, 1 ] + 0.5
	floor = sky[ None, : ] * ( LOOK[ 'ambient' ] * LOOK[ 'fol_sky' ] * mix( LOOK[ 'fol_under' ], 1, up * up ) * V[ 'FSKY' ] )[ :, None ]
	return E * mix( 1, vao, LOOK[ 'ao_direct' ] )[ :, None ], hemi * ( vao * A )[ :, None ] + floor.astype( np.float32 )

def light_dir( S, E, A, G ):

	# The light of a vertex as a direction, for the per-pixel relief of the fine level: 4 bytes per vertex.
	#   g (3 x int8, x 127 / dir_max): the direct light on a normal n' near the vertex's own is its baked value times
	#     1 + ( n' - n ) . g   (g = G / luminance of the direct light: the mean direction to the lights over the cosine)
	#   r (2 x 4 bit, red and blue): how much of the baked colour is direct light - what is left is the fill from the
	#     sky, which a turned normal does not lose. A crevice turned away from the lantern goes dark AND cool.
	# Nothing for leaves and thin panes (lit from both sides) and for what is not lit at all.
	V = S.v
	direct, amb = light_parts( S, E, A )
	le = lum( E )
	off = ( V[ 'LIT0' ] > 0 ) | ( V[ 'FOL' ] == 1 ) | ( V[ 'FOL' ] == 2 ) | ( le < 1e-5 )
	g = G / np.maximum( le, 1e-6 )[ :, None ]
	g *= np.minimum( 1, BK[ 'dir_max' ] / np.maximum( np.linalg.norm( g, axis = 1 ), 1e-9 ) )[ :, None ]
	r = direct / np.maximum( direct + amb, 1e-9 )
	g[ off ] = 0; r[ off ] = 0
	out = np.zeros( ( len( g ), 4 ), np.uint8 )
	out[ :, :3 ] = np.clip( np.round( g * ( 127 / BK[ 'dir_max' ] ) ), - 127, 127 ) + 128
	out[ :, 3 ] = ( np.round( np.clip( r[ :, 0 ], 0, 1 ) * 15 ).astype( np.uint8 ) << 4 ) | np.round( np.clip( r[ :, 2 ], 0, 1 ) * 15 ).astype( np.uint8 )
	return out

def pane_bytes( S ):

	# Lantern panes (thin glass lit from either side, a burner within pane_r): four bytes per vertex - the vector from
	# the vertex to that burner (3 x int8, x 127 / pane_r) and its hue (green and blue over red, 4 bits each; never 0
	# in the upper half, which is how the page tells such a pane; all zero for any other vertex). The page draws the
	# mantle's glare around the line of sight to the burner: a hot core in clear glass instead of an evenly glowing
	# card. For every level alike (they ride in the uv slots of glass vertices, which glass has no use for: see
	# encode_chunk), so a lantern looks the same whatever level its cell is drawn at.
	V = S.v
	out = np.zeros( ( len( V[ 'P' ] ), 4 ), np.uint8 )
	pr_ = float( BK.get( 'pane_r' ) or 0 )
	gl = np.zeros( len( out ), bool ); gl[ S.t[ 'TRI' ][ S.t[ 'KIND' ] == K_GLASS ].ravel() ] = True
	gi = np.nonzero( gl & ( V[ 'FOL' ] == 2 ) )[ 0 ]
	src = np.nonzero( ( LIGHTS[ 'src' ] == 1 ) & ( LIGHTS[ 'I' ] > 0 ) )[ 0 ]
	npane = 0; heads = set()
	if pr_ > 0 and len( gi ) and len( src ):

		Lp = LIGHTS[ 'pos' ][ src ]
		for b0 in range( 0, len( gi ), 4096 ):

			ii = gi[ b0:b0 + 4096 ]
			D = Lp[ None, :, : ] - V[ 'P' ][ ii ].astype( np.float64 )[ :, None, : ]
			j = np.argmin( ( D * D ).sum( 2 ), 1 )
			vec = D[ np.arange( len( ii ) ), j ]
			ok = np.abs( vec ).max( 1 ) <= pr_
			hue = LIGHTS[ 'col' ][ src[ j ] ]; hue = hue / np.maximum( hue.max( 1, keepdims = True ), 1e-6 )
			hb = ( np.maximum( 2, np.round( np.clip( hue[ :, 1 ], 0, 1 ) * 15 ) ).astype( np.uint8 ) << 4 ) | np.round( np.clip( hue[ :, 2 ], 0, 1 ) * 15 ).astype( np.uint8 )
			out[ ii[ ok ], :3 ] = np.clip( np.round( vec[ ok ] * ( 127 / pr_ ) ), - 127, 127 ) + 128
			out[ ii[ ok ], 3 ] = hb[ ok ]
			npane += int( ok.sum() ); heads.update( int( x ) for x in src[ j[ ok ] ] )

	log( '  lantern panes: %d of %d glass vertices have a burner within %.2f m (%d of %d burners behind glass)' % ( npane, int( gl.sum() ), pr_, len( heads ), len( src ) ) )
	return out

def dir_decode( B ):

	# what the page makes of light_dir()'s bytes: -> g, r (red, green, blue)
	g = ( B[ :, :3 ].astype( np.float32 ) - 128 ) * ( BK[ 'dir_max' ] / 127 )
	rr = ( B[ :, 3 ] >> 4 ).astype( np.float32 ) / 15; rb = ( B[ :, 3 ] & 15 ).astype( np.float32 ) / 15
	return g, np.stack( [ rr, mix( rr, rb, 0.3 ), rb ], 1 )

def probe_light( bvh, X, normals ):

	# dir_check(): light probes at X the way a probe grid would hold them - the radiance of every lantern and the moon
	# (one shadow ray each from the probe), as L2 spherical harmonics - evaluated as irradiance for each set of normals.
	# (SH is linear: the irradiance of a blend of probes is the blend of their irradiances, so no coefficients are kept.
	# A point light of perpendicular irradiance E gives E x K( n . l ), K the L2 fit of the clamped cosine.)
	rc = bvh.ray_cast
	Lp = LIGHTS[ 'pos' ].astype( np.float32 ); Lc = ( LIGHTS[ 'col' ] * LIGHTS[ 'I' ][ :, None ] ).astype( np.float32 )
	Ll = lum( Lc ).astype( np.float32 ); Lr = LIGHTS[ 'range' ].astype( np.float32 ); Lup = LIGHTS[ 'up' ].astype( np.float32 )
	Lskip = np.maximum( BK[ 'light_skip' ], LIGHTS[ 'near' ] + 0.25 ).astype( np.float32 )
	sr2 = LOOK[ 'source_radius' ] ** 2; reach2 = np.float32( LOOK.get( 'reach', 0 ) ** 2 ); leak = BK[ 'shadow_leak' ]
	md = np.array( LOOK[ 'moon_dir' ], np.float64 ); md /= np.linalg.norm( md ); md = md.astype( np.float32 )
	mcol = np.array( LOOK[ 'moon_col' ], np.float32 ) * LOOK[ 'moon' ]
	K = lambda c: 0.25 + 0.5 * c + 0.15625 * ( 3 * c * c - 1 )
	out = [ np.zeros( ( len( X ), 3 ), np.float32 ) for _ in normals ]
	rays = 0
	for b0 in range( 0, len( X ), 2048 ):

		b1 = min( len( X ), b0 + 2048 )
		p = X[ b0:b1 ].astype( np.float32 )
		D = Lp[ None, :, : ] - p[ :, None, : ]
		d2 = np.einsum( 'vlk,vlk->vl', D, D ); d = np.sqrt( d2 )
		w = ( d2 / ( d2 + sr2 ) ) / np.maximum( d2, 0.01 ) * np.clip( 1 - ( d / Lr[ None, : ] ) ** 4, 0, 1 ) ** 2
		w *= 1 + ( Lup[ None, : ] - 1 ) * sstep( - 0.42, 0.3, - D[ :, :, 1 ] / np.maximum( d, 1e-3 ) )
		if reach2 > 0: w /= 1 + d2 / reach2
		vi, lj = np.nonzero( w * Ll[ None, : ] > BK[ 'light_eps' ] )
		dr = D[ vi, lj ] / np.maximum( d[ vi, lj ], 1e-6 )[ :, None ]
		O = p[ vi ].tolist(); Dd = dr.tolist(); Tm = ( d[ vi, lj ] - Lskip[ lj ] ).tolist()
		hit = np.zeros( len( vi ), np.float32 )
		for i in range( len( vi ) ):

			if Tm[ i ] > 0.02 and rc( O[ i ], Dd[ i ], Tm[ i ] )[ 0 ] is not None: hit[ i ] = 1.0

		rays += len( vi ) + len( p )
		ww = w[ vi, lj ] * ( leak + ( 1 - leak ) * ( 1 - hit ) )
		O = p.tolist(); mdl = md.tolist()
		mh = np.array( [ 1.0 if rc( o, mdl, 400.0 )[ 0 ] is not None else 0.0 for o in O ], np.float32 )
		for k, Nn in enumerate( normals ):

			n = Nn[ b0:b1 ]
			np.add.at( out[ k ][ b0:b1 ], vi, ( ww * K( np.einsum( 'ij,ij->i', dr, n[ vi ] ) ) )[ :, None ] * Lc[ lj ] )
			out[ k ][ b0:b1 ] += ( K( n @ md ) * ( 0.04 + 0.96 * ( 1 - mh ) ) )[ :, None ] * mcol[ None, : ]

	return out, rays

def probe_valid( bvh, X ):

	# a probe inside a wall: two or more of the six axis rays leave through the back of a face within a metre
	rc = bvh.ray_cast
	bad = np.zeros( len( X ), np.int32 )
	O = X.tolist()
	for dv in ( ( 1, 0, 0 ), ( - 1, 0, 0 ), ( 0, 1, 0 ), ( 0, - 1, 0 ), ( 0, 0, 1 ), ( 0, 0, - 1 ) ):

		for i in range( len( O ) ):

			r = rc( O[ i ], dv, 1.0 )
			if r[ 0 ] is not None and r[ 1 ][ 0 ] * dv[ 0 ] + r[ 1 ][ 1 ] * dv[ 1 ] + r[ 1 ][ 2 ] * dv[ 2 ] > 0: bad[ i ] += 1

	return bad < 2

def dir_check( S, bvh, E, A, G, DIR, need ):

	# How well does a baked direction stand in for the light on a turned normal (a bump of the relief)? Vertices of
	# lit opaque surfaces, their normal turned by 15 and 30 degrees; the bake's own light on that normal (shadow rays
	# and all) against
	#   flat      the baked colour as it is (the lean tiers)
	#   vertex    the baked colour x the 4-byte direction of light_dir(), evaluated as the page does
	#   probes    L2 light probes on a grid (BK dircheck_grid metres), trilinear, no visibility test (three's
	#             LightProbeGrid), used as they are: irradiance from the probes x the vertex's own occlusion
	#   probes/r  the baked colour x probes( n' ) / probes( n )
	#   .. valid  the same with the probes inside walls left out of the blend
	# Errors are in luminance, as a share of the brighter of the true and the baked luminance; red/blue: the median error of that ratio.
	V = S.v
	t0 = time.time()
	rs = np.random.RandomState( 777 )
	cand = np.nonzero( need & ( V[ 'FOL' ] == 0 ) & ( lum( E ) > 0.02 ) )[ 0 ]
	pick = np.sort( cand[ rs.choice( len( cand ), min( int( BK.get( 'dircheck_n', 6000 ) ), len( cand ) ), replace = False ) ] )
	m = len( pick )
	P = V[ 'P' ][ pick ].astype( np.float32 ); n0 = V[ 'N' ][ pick ].astype( np.float32 ); vao = V[ 'VAO' ][ pick ]; A0 = A[ pick ]
	kd = mix( 1, vao, LOOK[ 'ao_direct' ] )
	sky = np.array( LOOK[ 'sky' ], np.float32 ); gnd = np.array( LOOK[ 'ground' ], np.float32 )
	skyv = sky[ None, : ] * ( 1 + LOOK[ 'roof_fill' ] * sstep( 5.5, 13, P[ :, 1 ] ) )[ :, None ]
	def lit( Ex, n ): return ( Ex * kd[ :, None ] + mix( gnd[ None, : ], skyv, ( 0.5 * n[ :, 1 ] + 0.5 )[ :, None ] ) * LOOK[ 'ambient' ] * ( vao * A0 )[ :, None ] ) / math.pi
	base = lit( E[ pick ], n0 )
	lb = np.maximum( lum( base ), 1e-6 )
	# the turned normals and the bake's light on them
	Ns = [ n0 ]; true = [ base ]
	fz = np.zeros( m, V[ 'FOL' ].dtype )
	for th in ( 15, 30 ):

		t = np.cross( n0, rs.normal( size = ( m, 3 ) ).astype( np.float32 ) ); t /= np.maximum( np.linalg.norm( t, axis = 1, keepdims = True ), 1e-9 )
		nt = ( math.cos( math.radians( th ) ) * n0 + math.sin( math.radians( th ) ) * t ).astype( np.float32 )
		Ed = np.zeros( ( m, 3 ), np.float32 ); AOd = np.zeros( m, np.float32 )
		light_range( bvh, P, nt, fz, 0, m, Ed, AOd )
		Ns.append( nt ); true.append( lit( Ed, nt ) )

	# the vertex model as the page has it
	g, r = dir_decode( DIR[ pick ] )
	def vertex( nt ):

		dn = nt - n0
		ka = 0.5 * ( sky - gnd )[ None, : ] / mix( gnd[ None, : ], sky[ None, : ], ( 0.5 * n0[ :, 1 ] + 0.5 )[ :, None ] )
		return base * np.clip( 1 + np.einsum( 'ij,ij->i', dn, g )[ :, None ] * r + dn[ :, 1:2 ] * ka * ( 1 - r ), 0, 3 )

	# probes
	h = float( BK.get( 'dircheck_grid', 1.0 ) )
	c0 = np.floor( P / h ); f = P / h - c0
	acc = [ np.zeros( ( m, 3 ), np.float32 ) for _ in Ns ]; accv = [ np.zeros( ( m, 3 ), np.float32 ) for _ in Ns ]; wv = np.zeros( m, np.float32 )
	rays = 0; cells = []; nvalid = 0
	for ci in range( 8 ):

		o = np.array( [ ci & 1, ci >> 1 & 1, ci >> 2 & 1 ], np.float32 )
		X = ( ( c0 + o ) * h ).astype( np.float32 )
		w = np.prod( np.where( o[ None, : ] > 0, f, 1 - f ), 1 ).astype( np.float32 )
		outs, nr = probe_light( bvh, X, Ns ); rays += nr
		ok = probe_valid( bvh, X ); nvalid += int( ok.sum() )
		cells.append( c0 + o )
		for k in range( len( Ns ) ): acc[ k ] += w[ :, None ] * outs[ k ]; accv[ k ] += ( w * ok )[ :, None ] * outs[ k ]
		wv += w * ok

	for k in range( len( Ns ) ): accv[ k ] = np.where( wv[ :, None ] > 1e-4, accv[ k ] / np.maximum( wv, 1e-4 )[ :, None ], acc[ k ] )
	nprobe = len( np.unique( np.concatenate( cells ).astype( np.int64 ), axis = 0 ) )
	tp = time.time() - t0
	def err( model, tr ):

		# (a share of the brighter of the two: a normal turned towards a lantern the vertex's own did not face has no bound otherwise)
		e = ( lum( model ) - lum( tr ) ) / np.maximum( np.maximum( lum( tr ), lb ), 1e-6 )
		rb = lambda c: c[ :, 0 ] / np.maximum( c[ :, 2 ], 1e-9 )
		hue = np.abs( rb( model ) / np.maximum( rb( tr ), 1e-9 ) - 1 )
		a = np.abs( e )
		return 'mean %.3f median %.3f p90 %.3f bias %+.3f red/blue %.3f' % ( float( a.mean() ), float( np.median( a ) ), float( np.percentile( a, 90 ) ), float( e.mean() ), float( np.median( hue ) ) )

	log( '  dircheck: %d vertices, %d probes of a %.2g m grid around them (%d%% of the corners valid), %d probe rays, %.1fs' % ( m, nprobe, h, round( 100 * nvalid / ( 8 * m ) ), rays, tp ) )
	log( '  dircheck  0 deg | probes %s | probes valid %s' % ( err( lit( acc[ 0 ], n0 ), base ), err( lit( accv[ 0 ], n0 ), base ) ) )
	for k, th in ( ( 1, 15 ), ( 2, 30 ) ):

		ratio = lambda a: base * np.clip( lit( a[ k ], Ns[ k ] ) / np.maximum( lit( a[ 0 ], n0 ), 1e-6 ), 0, 3 )
		log( '  dircheck %2d deg | flat %s | vertex %s | probes %s | probes/r %s | probes valid %s | probes/r valid %s' % ( th,
			err( base, true[ k ] ), err( vertex( Ns[ k ] ), true[ k ] ), err( lit( acc[ k ], Ns[ k ] ), true[ k ] ), err( ratio( acc ), true[ k ] ),
			err( lit( accv[ k ], Ns[ k ] ), true[ k ] ), err( ratio( accv ), true[ k ] ) ) )

def final_colour( S, E, A, EM = None ):

	V = S.v
	direct, amb = light_parts( S, E, A )
	lit = ( direct + amb ) / math.pi
	lit[ V[ 'LIT0' ] > 0 ] = 0
	return body_glow( S, V[ 'MUL' ] * ( lit + V[ 'GLOW' ][ :, None ] ) + ( V[ 'EM' ] if EM is None else EM ) )

def halo_bodies():

	# -> ( centres n x 3 (three's frame), radii n, hues n x 3 ) of the opaque glowing bodies town.json declares round a
	# burner: every `halo_at` point of a light that has a `halo_r` (the opal globes of a globe post)
	c = []; r = []; h = []
	for i, l in enumerate( TOWN.get( 'lights', [] ) ):

		if not l.get( 'halo_r' ) or not l.get( 'halo_at' ): continue
		hue = LIGHTS[ 'col' ][ i ] / max( float( LIGHTS[ 'col' ][ i ].max() ), 1e-6 )
		for p in l[ 'halo_at' ]: c.append( b2t( p ) ); r.append( float( l[ 'halo_r' ] ) ); h.append( hue )

	return np.array( c, np.float64 ).reshape( - 1, 3 ), np.array( r, np.float64 ), np.array( h, np.float64 ).reshape( - 1, 3 )

BODIES = [ None, 0 ]

def body_glow( S, C ):

	# The body of a lamp is its burner: a surface that only glows (emissive, neon: no lit colour of its own) and lies
	# inside a declared glowing body takes at least BK[ 'halo_body' ] x the hue of its light (a fifth of it white, as
	# neon is). An opal globe with the glow its material has (0.8) comes out grey on the page beside a lantern's
	# glass, and greyer under its clear skin: it read as switched off without its sprite. 0 = as the material says
	k = float( BK[ 'halo_body' ] )
	if k <= 0: return C
	if BODIES[ 0 ] is None: BODIES[ 0 ] = halo_bodies()
	bc, br, bh = BODIES[ 0 ]
	if not len( bc ): return C
	V = S.v
	em = np.nonzero( ( V[ 'EMF' ] > 0 ) & ( V[ 'MUL' ].max( 1 ) <= 0 ) )[ 0 ]
	if not len( em ): return C
	P = V[ 'P' ][ em ].astype( np.float64 )
	n = 0
	for c, r, hue in zip( bc, br, bh ):

		near = np.nonzero( ( np.abs( P - c[ None, : ] ) <= r * 1.03 + 0.005 ).all( 1 ) )[ 0 ]
		if not len( near ): continue
		near = near[ np.linalg.norm( P[ near ] - c[ None, : ], axis = 1 ) <= r * 1.03 + 0.005 ]
		if not len( near ): continue
		floor = ( mix( hue, np.ones( 3 ), 0.2 ) * k ).astype( C.dtype )
		C[ em[ near ] ] = np.maximum( C[ em[ near ] ], floor[ None, : ] )
		n += len( near )

	BODIES[ 1 ] += n
	if n: log( '  glowing bodies: %d vertices inside %d declared bodies take at least %g x their light\'s hue' % ( n, len( bc ), k ) )
	return C

# ------------------------------------------------------------------------------------------ chunk writer


AUDIT = {} # level -> rows of glow_audit()
AUDIT_LEVEL = [ '' ]

def glow_audit( name, P, N, tri, kind, glow, q, rules ):

	# What the position grid does to the room around a glowing surface. From a sample of the glowing vertices a ray
	# along the normal and one against it (glow_reach long) on the chunk as modelled: the distance d0 to the surface in
	# front / behind. Then the same distance with the vertex and the three corners of the surface hit moved as each
	# rule of rounding moves them. rules: name -> displacement per vertex (metres). A third rule is tried here alone,
	# 'aware': the vertex takes, of the grid points around it, the one that keeps the smaller of its two distances
	# largest (everything else to nearest). Rows: side (0 front, 1 behind), what was hit (0 opaque, 1 glowing, 2 glass),
	# d0, then the distance after each rule.
	from mathutils.bvhtree import BVHTree
	from mathutils import Vector
	g = np.nonzero( glow )[ 0 ]
	if not len( g ): return
	if len( g ) > BK[ 'glow_sample' ]: g = np.random.default_rng( 7 ).choice( g, BK[ 'glow_sample' ], replace = False )
	bvh = BVHTree.FromPolygons( P.tolist(), tri.tolist(), all_triangles = True )
	reach = BK[ 'glow_reach' ]; eps = 5e-5
	rows = []
	for v in g:

		n = N[ v ]; l = float( np.linalg.norm( n ) )
		if l < 0.5: continue
		n = n / l
		for side, sgn in ( ( 0, 1.0 ), ( 1, - 1.0 ) ):

			d = n * sgn
			o = P[ v ] + d * eps
			loc, _, idx, dist = bvh.ray_cast( Vector( o.tolist() ), Vector( d.tolist() ), reach )
			if loc is None: continue
			rows.append( ( v, side, idx, dist + eps, d[ 0 ], d[ 1 ], d[ 2 ], loc[ 0 ], loc[ 1 ], loc[ 2 ] ) )

	if not rows: return
	R = np.array( rows, np.float64 )
	v = R[ :, 0 ].astype( np.int64 ); side = R[ :, 1 ].astype( np.int64 ); t = R[ :, 2 ].astype( np.int64 ); d0 = R[ :, 3 ]; D = R[ :, 4:7 ]; L = R[ :, 7:10 ]
	T = tri[ t ]
	# (a surface the vertex itself belongs to is no neighbour)
	ok = ( T != v[ :, None ] ).all( 1 )
	a, b, c = P[ T[ :, 0 ] ], P[ T[ :, 1 ] ], P[ T[ :, 2 ] ]
	e0, e1, e2 = b - a, c - a, L - a
	d00 = ( e0 * e0 ).sum( 1 ); d01 = ( e0 * e1 ).sum( 1 ); d11 = ( e1 * e1 ).sum( 1 ); d20 = ( e2 * e0 ).sum( 1 ); d21 = ( e2 * e1 ).sum( 1 )
	den = d00 * d11 - d01 * d01
	ok &= den > 1e-18
	den = np.where( den > 1e-18, den, 1 )
	bv = ( d11 * d20 - d01 * d21 ) / den; bw = ( d00 * d21 - d01 * d20 ) / den
	B = np.clip( np.stack( [ 1 - bv - bw, bv, bw ], 1 ), 0, 1 )
	what = np.where( kind[ t ] == K_GLASS, 2, np.where( glow[ T ].all( 1 ), 1, 0 ) )
	cols = [ side, what, d0 ]
	near = None
	for rn, U in rules.items():

		move = ( B[ :, :, None ] * U[ T ] ).sum( 1 ) - U[ v ]
		cols.append( d0 + ( D * move ).sum( 1 ) )
		if rn == 'nearest': near = U

	# 'aware': the eight grid points around the vertex (those along an axis its normal has no part in stay nearest)
	if near is not None:

		X = P[ v ] / q
		fl = np.floor( X ); best = np.full( len( v ), - 1e9 ); bg = np.zeros( ( len( v ), 2 ) )
		hit = ( B[ :, :, None ] * near[ T ] ).sum( 1 )
		# both rays of a vertex decide together: the worse of the two per choice
		order = np.argsort( v, kind = 'stable' ); vs = v[ order ]
		st = np.nonzero( np.concatenate( [ [ True ], vs[ 1: ] != vs[ :- 1 ] ] ) )[ 0 ]
		grp = np.empty( len( v ), np.int64 ); grp[ order ] = np.repeat( np.arange( len( st ) ), np.diff( np.concatenate( [ st, [ len( vs ) ] ] ) ) )
		free = np.abs( N[ v ] ) > 0.1 * np.linalg.norm( N[ v ], axis = 1, keepdims = True )
		choice = np.zeros( len( v ) ); score = np.full( len( st ), - 1e9 ); pick = np.zeros( len( st ), np.int64 ); gaps = []
		for k in range( 8 ):

			up = np.array( [ k & 1, k >> 1 & 1, k >> 2 & 1 ], np.float64 )
			cand = np.where( free, fl + up, np.round( X ) ) * q - P[ v ]
			gk = d0 + ( D * ( hit - cand ) ).sum( 1 )
			gaps.append( gk )
			# (no credit for more room than there was; ties go to the point nearest)
			sk = np.minimum( gk, d0 ) - 1e-9 * np.abs( cand ).sum( 1 ) / q
			sk = np.where( ok, sk, 1e9 )
			worst = np.full( len( st ), 1e9 ); np.minimum.at( worst, grp, sk )
			better = worst > score
			score = np.where( better, worst, score ); pick = np.where( better, k, pick )

		cols.append( np.stack( gaps, 1 )[ np.arange( len( v ) ), pick[ grp ] ] )

	AUDIT.setdefault( name, [] ).append( np.stack( cols, 1 )[ ok ] )

def glow_report( name ):

	if name not in AUDIT: return
	R = np.concatenate( AUDIT.pop( name ) )
	names = [ 'nearest', 'outward', 'aware' ][ :R.shape[ 1 ] - 3 ]
	log( '  glow audit (%s): %d rays hit something within %.0f mm of a glowing vertex' % ( name, len( R ), BK[ 'glow_reach' ] * 1000 ) )
	for side, sn in ( ( 0, 'in front' ), ( 1, 'behind' ) ):

		for what, wn in ( ( 0, 'opaque' ), ( 1, 'glowing' ), ( 2, 'glass' ) ):

			for lo, hi in ( ( 0, 0.0005 ), ( 0.0005, 0.0015 ), ( 0.0015, 0.0035 ), ( 0.0035, 1 ) ):

				m = ( R[ :, 0 ] == side ) & ( R[ :, 1 ] == what ) & ( R[ :, 2 ] >= lo ) & ( R[ :, 2 ] < hi )
				if not m.any(): continue
				txt = []
				for i, rn in enumerate( names ):

					gp = R[ m, 3 + i ] * 1000
					txt.append( '%s: through %d, under 0.25 mm %d, least %.2f, 5%% %.2f, median %.2f' % ( rn, ( gp < - 1e-6 ).sum(), ( gp < 0.25 ).sum(), gp.min(), np.percentile( gp, 5 ), np.median( gp ) ) )

				log( '    %-8s %-7s %.1f-%.1f mm as modelled: %6d (median %.2f) | %s' % ( sn, wn, lo * 1000, min( hi, BK[ 'glow_reach' ] ) * 1000, m.sum(), np.median( R[ m, 2 ] ) * 1000, ' | '.join( txt ) ) )

def encode_chunk( S, C, LOCK, tsel, DIR = None, PANE = None, key = None, C2 = None, DIR2 = None ):

	# -> ( vertex bytes, index bytes (u32), flag bytes, meta, clamped uv groups ) for the triangles tsel of soup S (colours C).
	# Flags per vertex: 1 = must stay (LOCK), 2 = part of a small thing, 4 = ground (the terrain: see border_lock()).
	# Vertex record: 10 x u16 = position xyz on a power-of-two grid + texture layer, uv, colour (sqrt of value / 64), aux.
	# With DIR (light_dir(): the fine level) 12 x u16 - its four bytes follow.
	# With C2 (a second look, LOOK2) its colour follows all that as rgb + a spare word, then its direction DIR2 where
	# there is a DIR: 14 x u16 (second colour at word 10) or 18 x u16 (second colour at 12, second direction at 16).
	# With PANE (pane_bytes()) the uv slots of glass vertices hold its four bytes (glass is drawn without a texture);
	# a vertex that glass shares with anything else is given to each on its own.
	V, Tt = S.v, S.t
	tri = Tt[ 'TRI' ][ tsel ]; kind = Tt[ 'KIND' ][ tsel ]
	order = np.argsort( kind, kind = 'stable' )
	tri = tri[ order ]; kind = kind[ order ]
	NV = len( V[ 'P' ] )
	if PANE is not None: tri = tri + ( kind == K_GLASS )[ :, None ].astype( tri.dtype ) * NV
	used, inv = np.unique( tri.ravel(), return_inverse = True )
	isg = used >= NV; used = used % NV
	tri = inv.reshape( - 1, 3 )
	P = V[ 'P' ][ used ].astype( np.float64 )
	# positions snap to a grid shared by all chunks (1/1024 m unless the chunk is too large): ground vertices on a
	# cell border land on the same point in both cells
	# What glows is rounded outward along its normal instead (per axis, by the sign of the normal: its distance to
	# whatever lies behind it can then lose half a step at most where rounding both to nearest loses a whole one, and
	# on a wall along the axes nothing), and glass one step further than that: a pane a fraction of a millimetre in
	# front of a lit card stays in front of it. Nothing else moves, so the ground and the cell borders are as before.
	out = np.zeros( len( used ), np.int64 )
	aud = BK[ 'glow_audit' ] and ( BK[ 'glow_audit' ] == 'all' or key in BK[ 'glow_audit' ].split( ',' ) )
	glow = ( V[ 'GLOW' ][ used ] > 0 ) | ( V[ 'EM' ][ used ].max( 1 ) > 0 ) | ( V[ 'EMF' ][ used ] > 0.5 )
	if BK[ 'glow_out' ] or aud:

		out[ glow ] = 1
		gl = np.zeros( len( used ), bool ); gl[ tri[ kind == K_GLASS ].ravel() ] = True
		out[ gl ] = 2

	w = np.nonzero( out )[ 0 ]
	if len( w ):

		n = V[ 'N' ][ used[ w ] ].astype( np.float64 )
		sg = np.where( n > 0.1, 1.0, np.where( n < - 0.1, - 1.0, 0.0 ) )
		# vertices at one point move as one where their normals do not pull apart (the edge of a glowing box stays
		# closed; the two faces of a card drawn from both sides go their own ways)
		_, gi = np.unique( np.ascontiguousarray( P[ w ] ).view( np.dtype( ( np.void, 24 ) ) ).ravel(), return_inverse = True )
		gi = gi.ravel()
		hi = np.zeros( ( gi.max() + 1, 3 ) ); lo = np.zeros( ( gi.max() + 1, 3 ) )
		np.maximum.at( hi, gi, sg ); np.minimum.at( lo, gi, sg )
		sg = np.where( ( hi[ gi ] > 0 ) & ( lo[ gi ] < 0 ), sg, hi[ gi ] + lo[ gi ] )
		two = ( out[ w ] == 2 )[ :, None ]

	def grid( w ):

		q = 1 / 1024
		while True:

			mn = np.floor( P.min( 0 ) / q ) * q
			G = ( P - mn ) / q
			fit = G.max() <= 65534
			G = np.round( G )
			if len( w ):

				X = ( P[ w ] - mn ) / q
				G[ w ] = np.where( sg > 0, np.ceil( X ), np.where( sg < 0, np.floor( X ), G[ w ] ) ) + sg * two
				# (a step beyond the box of the chunk: the box moves with it)
				g0 = np.minimum( G.min( 0 ), 0 )
				mn = mn + g0 * q; G -= g0
				fit = fit and G.max() <= 65535

			if fit: return q, mn, G
			q *= 2

	q, mn, G = grid( w if BK[ 'glow_out' ] else w[ :0 ] )
	if aud:

		rules = {}
		for rn, ww in ( ( 'nearest', w[ :0 ] ), ( 'outward', w ) ):

			q2, mn2, G2 = grid( ww )
			rules[ rn ] = G2 * q2 + mn2 - P

		glow_audit( AUDIT_LEVEL[ 0 ], P, V[ 'N' ][ used ].astype( np.float64 ), tri, kind, glow, q, rules )

	W = ( 10 if DIR is None else 12 ) + ( 0 if C2 is None else 4 if DIR is None else 6 )
	rec = np.zeros( ( len( used ), W ), np.uint16 )
	rec[ :, 0:3 ] = G
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
	if PANE is not None: rec[ isg, 4:6 ] = np.ascontiguousarray( PANE[ used[ isg ] ] ).view( '<u2' )
	col = C[ used ]
	rec[ :, 6:9 ] = np.round( np.sqrt( np.clip( col, 0, 64 ) / 64 ) * 65535 )
	aux = V[ 'AUX' ][ used ].astype( np.float64 )
	# opaque: gloss in the lower half, a glowing surface (higher highlight ceiling) flagged in the upper half: 65535, a
	# sign 65533 (the packer marks those of the lowest level that are not to be pulled forward one lower: 65534, 65532)
	opq = np.zeros( len( used ), bool ); opq[ tri[ kind == K_OPAQUE ].ravel() ] = True
	a16 = np.where( opq, np.where( V[ 'EMF' ][ used ] > 1.5, 65533, np.where( V[ 'EMF' ][ used ] > 0.5, 65535, np.round( np.clip( aux, 0, 1 ) * 32767 ) ) ), np.round( np.clip( aux, 0, 1 ) * 65535 ) )
	rec[ :, 9 ] = a16
	if DIR is not None: rec[ :, 10:12 ] = np.ascontiguousarray( DIR[ used ] ).view( '<u2' )
	if C2 is not None:

		c2 = 10 if DIR is None else 12
		rec[ :, c2:c2 + 3 ] = np.round( np.sqrt( np.clip( C2[ used ], 0, 64 ) / 64 ) * 65535 )
		if DIR is not None: rec[ :, 16:18 ] = np.ascontiguousarray( DIR2[ used ] ).view( '<u2' )

	lock = LOCK[ used ] | ( V[ 'PROP' ][ used ] << 1 ) | ( V[ 'KEEP' ][ used ] & 1 ) # (KEEP: _TSQ_KEEP of the model = must stay, at every level)
	# weld identical records
	view = np.ascontiguousarray( rec ).view( np.dtype( ( np.void, W * 2 ) ) ).ravel()
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
	meta = dict( min = [ float( x ) for x in mn ], max = [ round( float( x ), 4 ) for x in mx ], q = q, nv = int( len( rec ) ), ni = int( len( tri ) * 3 ), sub = sub, **( dict( w = W ) if W != 10 else {} ) )
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
	# other triangles at each level, and the mirrored town switched with the level).
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
	# ... and for the tiers that light the near field per pixel (BK nr_sizes): the normal map's x and y with the
	# roughness in blue, as bytes
	n3 = nrm[ ..., :3 ].astype( np.float64 ) * 2 - 1
	for s in BK[ 'nr_sizes' ]:

		n = resize( n3, s ); n /= np.maximum( np.linalg.norm( n, axis = 2, keepdims = True ), 1e-6 )
		q = np.concatenate( [ n[ ..., :2 ] * 0.5 + 0.5, resize( orm[ ..., 1:2 ].astype( np.float64 ), s ), np.ones( ( s, s, 1 ) ) ], 2 )
		out[ 'n%d' % s ] = np.round( np.clip( q, 0, 1 ) * 255 ).astype( np.uint8 )

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

def write_layer( a, name ):

	# one layer as its own image (same encoding as a page)
	import bpy
	size = a.shape[ 0 ]
	t = np.empty( ( size, size, 4 ), np.float32 )
	t[ ..., :3 ] = to_srgb( a[ ..., :3 ] ); t[ ..., 3 ] = 0.5 + 0.5 * np.clip( a[ ..., 3 ], 0, 1 )
	img = bpy.data.images.new( name, size, size, alpha = True, float_buffer = False )
	img.colorspace_settings.name = 'Non-Color'
	img.alpha_mode = 'STRAIGHT'
	img.pixels.foreach_set( np.ascontiguousarray( t[ ::- 1 ] ).ravel() )
	img.filepath_raw = os.path.join( OUT, name + '.png' ); img.file_format = 'PNG'
	img.save()
	bpy.data.images.remove( img )
	return name

def write_bytes( a, name ):

	# one layer of plain bytes (no colour space) as its own image
	import bpy
	size = a.shape[ 0 ]
	img = bpy.data.images.new( name, size, size, alpha = True, float_buffer = False )
	img.colorspace_settings.name = 'Non-Color'
	img.alpha_mode = 'STRAIGHT'
	img.pixels.foreach_set( np.ascontiguousarray( a[ ::- 1 ].astype( np.float32 ) / 255 ).ravel() )
	img.filepath_raw = os.path.join( OUT, name + '.png' ); img.file_format = 'PNG'
	img.save()
	bpy.data.images.remove( img )
	return name

def bake_textures():

	# -> ( pages per size, single layers per size ). The pages hold every layer at the small sizes; the single files
	# hold a layer at 1024 (all of them) and at 2048 (the first LAYERS.big[ 0 ] library layers, whose maps are that large)
	psizes = BK[ 'tex_sizes' ]
	lsizes = sorted( BK[ 'layer_sizes' ] ) if BK[ 'single_layers' ] else []
	top = set( range( 2, 2 + LAYERS.big[ 0 ] ) )
	texl = { str( s ): dict( size = s, files = [] ) for s in lsizes }
	texn = { str( s ): dict( size = s, files = [] ) for s in BK[ 'nr_sizes' ] } # normal + roughness (library materials only)
	names = LAYERS.names
	cache_dir = os.path.join( OUT, '.texcache' )
	os.makedirs( cache_dir, exist_ok = True )
	per_size = { s: [] for s in psizes }
	t0 = time.time()
	for i, name in enumerate( names ):

		# the sizes of this layer: the largest only where the maps have it
		mine = [ s for s in lsizes if s < lsizes[ - 1 ] or i in top ]
		sizes = psizes + [ s for s in mine if s not in psizes ]
		nr = None
		if name == 'white': px = { s: np.ones( ( s, s, 4 ), np.float32 ) for s in sizes }
		elif name == 'window': px = { s: window_layer( s ) for s in sizes }
		elif name.startswith( 'stripe:' ): px = { s: stripe_layer( s, LAYERS.stripes[ name ] ) for s in sizes }
		else:

			d = os.path.join( ROOT, 'out/mat', name )
			info = INFO[ name ]
			sig = hashlib.sha1( json.dumps( [ [ f, os.path.getmtime( os.path.join( d, f ) ), os.path.getsize( os.path.join( d, f ) ) ] for f in sorted( info[ 'maps' ].values() ) ] + [ sizes, CFG.get( 'relief', 0.55 ), CFG.get( 'relief_sign', - 1.0 ), 5, BK[ 'nr_sizes' ] ] ).encode() ).hexdigest()[ :16 ]
			cf = os.path.join( cache_dir, '%s_%s.npz' % ( name, sig ) )
			if os.path.exists( cf ):

				z = np.load( cf ); px = { s: z[ 's%d' % s ].astype( np.float32 ) / 255 for s in sizes }
				for s in sizes: px[ s ][ ..., :3 ] = from_srgb( px[ s ][ ..., :3 ] )
				nr = { s: z[ 'n%d' % s ] for s in BK[ 'nr_sizes' ] }

			else:

				px = lib_layer( name, sizes )
				nr = { s: px.pop( 'n%d' % s ) for s in BK[ 'nr_sizes' ] }
				for f in os.listdir( cache_dir ):

					if f.startswith( name + '_' ) and f[ len( name ) + 1: ].count( '_' ) == 0: os.remove( os.path.join( cache_dir, f ) )

				sv = {}
				for s in sizes:

					q = px[ s ].copy(); q[ ..., :3 ] = to_srgb( q[ ..., :3 ] )
					sv[ 's%d' % s ] = np.round( np.clip( q, 0, 1 ) * 255 ).astype( np.uint8 )

				for s in BK[ 'nr_sizes' ]: sv[ 'n%d' % s ] = nr[ s ]
				np.savez_compressed( cf, **sv )

		for s in psizes: per_size[ s ].append( px[ s ] )
		for s in lsizes: texl[ str( s ) ][ 'files' ].append( write_layer( px[ s ], 'l%d_%d' % ( s, i ) ) if s in mine else None )
		for s in BK[ 'nr_sizes' ]: texn[ str( s ) ][ 'files' ].append( write_bytes( nr[ s ], 'n%d_%d' % ( s, i ) ) if nr else None )
		del px, nr

	tex = {}
	for s in psizes:

		cols, rows = ( 4, 2 ) if s >= 512 else ( 8, 4 )
		files = write_pages( per_size[ s ], s, cols, rows, 85 )
		tex[ str( s ) ] = dict( size = s, cols = cols, rows = rows, pages = files )

	log( 'textures: %d layers x %s in pages, %s single, %.1fs' % ( len( names ), psizes, { s: sum( 1 for f in texl[ str( s ) ][ 'files' ] if f ) for s in lsizes }, time.time() - t0 ) )
	return tex, texl, texn

# ------------------------------------------------------------------------------------------ main

def setup():

	# the chunk cell of every item + the texture layers (needed before any build_soup())
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

	def tex_class( lib ):

		# 0: maps of the largest layer size, 1: large tiles (few texels per metre at the common size), 2: the rest
		info = INFO.get( lib ) or {}
		if min( info.get( 'res', [ 0, 0 ] ) ) >= max( BK[ 'layer_sizes' ] ): return 0
		return 1 if max( info.get( 'tile', [ 0, 0 ] ) ) >= BK[ 'big_tile' ] else 2

	for lib, a in sorted( area.items(), key = lambda kv: ( tex_class( kv[ 0 ] ), - kv[ 1 ] ) ):

		# (leaf cards are cut out by their layer's height: foliage always gets its layer)
		if CATALOG.get( lib, {} ).get( 'kind' ) == 'baked' and INFO.get( lib ) and ( a >= BK[ 'layer_min_area' ] or lib in ( 'leaf', 'hedge' ) ):

			LAYERS.add_lib( lib )
			if tex_class( lib ) < 2: LAYERS.big[ tex_class( lib ) ] += 1

	log( 'texture layers: %d materials of %d used (%d with %d px maps, %d more with large tiles)' % ( len( LAYERS.libs ), len( area ), LAYERS.big[ 0 ], max( BK[ 'layer_sizes' ] ), LAYERS.big[ 1 ] ) )
	return cells

def main():

	protos = TOWN[ 'protos' ]; items = TOWN[ 'items' ]
	cells = setup()
	raw = os.path.join( OUT, '.raw' )
	if os.path.isdir( raw ):

		for fn in os.listdir( raw ): os.remove( os.path.join( raw, fn ) )

	os.makedirs( raw, exist_ok = True )
	chunks = {} # key -> entry
	stats = {}
	for level, lname in ( ( 2, 'mid' ), ( 1, 'near' ) ) + ( ( ( 0, 'fine' ), ) if BK[ 'fine' ] else () ):

		log( 'level %s (LOD%d)' % ( lname, level ) )
		# (the models stay loaded from one level to the next: lod_masks() corrects the reduced ones of both levels at once)
		S = build_soup( level, cells )
		if level == 0 or not BK[ 'fine' ] and level == 1: _GLB.clear()
		nt0 = len( S.t[ 'TRI' ] )
		log( '  soup: %d tris, %d verts' % ( nt0, len( S.v[ 'P' ] ) ) )
		if S.v[ 'KEEP' ].any(): log( '  %d vertices carry _TSQ_KEEP: they stay at every level' % int( S.v[ 'KEEP' ].sum() ) )
		bvh = light_soup( S, level )
		cull_hidden( S )
		refine( S, lname )
		V = S.v
		need = np.zeros( len( V[ 'P' ] ), bool ); need[ S.t[ 'TRI' ].ravel() ] = True
		need &= ( V[ 'LIT0' ] == 0 ) & ( V[ 'MUL' ].max( 1 ) > 0 )
		wd = level == 0 and bool( BK[ 'dir' ] )
		# the lanterns' shadows on the ground go into a picture (ground_shadows(), from the top level's geometry); its
		# vertices, at every level, take the lanterns' light without them (GW: how much of a vertex is ground)
		gs = 3 if BK[ 'gsm' ] else 0
		r = run_lighting( bvh, V[ 'P' ], V[ 'N' ], V[ 'FOL' ], need, 'l%d' % level, wd, gs )
		E, A = r[ 0 ], r[ 1 ]; G = r[ 2 ] if wd else None
		GW = None
		if gs:

			GW = ground_weight( S, need )
			E += r[ - 1 ] * GW[ :, None ]
			if level == 0 or not BK[ 'fine' ] and level == 1: ground_shadows( S, bvh, raw )

		del r
		DIR = light_dir( S, E, A, G ) if wd else None
		PANE = pane_bytes( S )
		if wd and BK[ 'dircheck' ]: dir_check( S, bvh, E, A, G, DIR, need )
		C = final_colour( S, E, A )
		C2 = DIR2 = None
		if LOOK2:

			# the second look: the same vertices lit again, by its lanterns
			use_look( 1 )
			# (the shadow picture is the first look's: the second look's ground is without the lanterns' shadows all the
			# same, and the page darkens it by the first's picture)
			r = run_lighting( bvh, V[ 'P' ], V[ 'N' ], V[ 'FOL' ], need, 'l%db' % level, wd, gs )
			E2, A2 = r[ 0 ], r[ 1 ]; G2 = r[ 2 ] if wd else None
			if gs: E2 += r[ - 1 ] * GW[ :, None ]
			del r
			DIR2 = light_dir( S, E2, A2, G2 ) if wd else None
			C2 = final_colour( S, E2, A2, V[ 'EM2' ] )
			use_look( 0 )
			del E2, A2, G2

		del bvh, G
		LOCK = border_lock( S )
		cid = S.t[ 'CELL' ]
		order = np.argsort( cid, kind = 'stable' )
		cs = cid[ order ]
		st = np.nonzero( np.concatenate( [ [ True ], cs[ 1: ] != cs[ :- 1 ] ] ) )[ 0 ]
		en = np.concatenate( [ st[ 1: ], [ len( cs ) ] ] )
		tv = tt = 0; over = 0
		AUDIT_LEVEL[ 0 ] = lname
		for a, b in zip( st, en ):

			c = int( cs[ a ] )
			key = cell_key( c )
			if level <= 1:

				# a cell of ground alone gains nothing from the near and fine levels
				if ( c & 1 ) or key not in chunks or not ( cells == c ).any(): continue

			vb, ib, lk, meta, ov = encode_chunk( S, C, LOCK, order[ a:b ], DIR, PANE, key, C2, DIR2 )
			over += ov
			tv += meta[ 'nv' ]; tt += meta[ 'ni' ] // 3
			fn = '%s.%s.bin' % ( key, lname )
			with open( os.path.join( raw, fn ), 'wb' ) as f: f.write( vb ); f.write( ib ); f.write( lk )
			if level == 2: chunks[ key ] = dict( k = key, outer = bool( c & 1 ) )
			chunks[ key ][ lname ] = dict( raw = fn, **meta )

		if over: log( '  WARNING: %d surfaces span more than %d texture tiles (uv clamped)' % ( over, int( 2 * UV_RANGE - 2 ) ) )
		stats[ lname ] = dict( tris_in = int( nt0 ), tris = int( tt ), verts = int( tv ) )
		log( '  written: %d tris, %d verts' % ( tt, tv ) )
		glow_report( lname )
		del S, C, E, A, DIR, PANE, C2, DIR2

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
	def halo_rows( lamp ):

		halos = []
		for i, l in enumerate( TOWN.get( 'lights', [] ) ):

			I = l.get( 'halo', l.get( 'intensity', 0 ) if LIGHTS[ 'src' ][ i ] else 0 ) * lamp
			if I < 12: continue
			at = [ b2t( p ) for p in l[ 'halo_at' ] ] if l.get( 'halo_at' ) else [ LIGHTS[ 'pos' ][ i ] ] # several: the globes of one head, each with its share
			size = max( 0.5, round( math.sqrt( I / len( at ) / 40 ) * 1.5 * 4 ) / 4 )
			lift = round( float( l.get( 'halo_r', 0 ) ) * 1.15 + 0.01, 3 ) if l.get( 'halo_r' ) else 0 # in front of an opal globe, which would hide the hot centre
			for p in at: halos.append( [ round( float( v ), 3 ) for v in p ] + [ round( float( v ), 4 ) for v in LIGHTS[ 'col' ][ i ] ] + [ size, 1, lift ] )

		return halos

	halos = halo_rows( LOOK[ 'lamp' ] )

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

	light_rows = lambda L: [ [ round( float( v ), 3 ) for v in L[ 'pos' ][ i ] ] + [ round( float( v ), 4 ) for v in L[ 'col' ][ i ] * L[ 'I' ][ i ] ] + [ float( L[ 'range' ][ i ] ), float( L[ 'up' ][ i ] ) ]
		for i in range( len( L[ 'pos' ] ) ) ]
	lights = light_rows( LIGHTS )
	# the second look, as the page needs it beside the second colour of the vertices: its numbers, its glows, its lanterns
	look2 = dict( lamp = LOOK2.get( 'lamp', LOOK[ 'lamp' ] ), reach = LOOK2.get( 'reach', LOOK.get( 'reach', 0 ) ), range_scale = LOOK2.get( 'range_scale', BK.get( 'range_scale', 1 ) ),
		halos = halo_rows( LOOK2.get( 'lamp', LOOK[ 'lamp' ] ) ), lights = light_rows( LIGHTS2 ) ) if LOOK2 else None

	man = dict(
		version = 1,
		town = dict( name = CFG.get( 'name', 'town' ), sha1 = hashlib.sha1( TOWN_RAW ).hexdigest(), bytes = len( TOWN_RAW ), built = TOWN.get( 'stats', {} ).get( 'built' ) ),
		baked = int( time.time() ),
		look = LOOK, bake = BK,
		stride = 20, uv_range = UV_RANGE, colour_scale = 64, window_gain = WINDOW_GAIN, cut_aux = CUT_AUX,
		layers = LAYERS.names, layer_big = LAYERS.big, tex = tex[ 0 ], texl = tex[ 1 ],
		# metres of relief the height in a layer's alpha stands for (its material's height_range)
		relief = [ round( float( ( INFO.get( nm ) or {} ).get( 'height_range', 0.025 ) ), 5 ) for nm in LAYERS.names ],
		# ... and its normal + roughness layers (n<size>_<layer>), with the material's mean roughness and metalness
		texn = tex[ 2 ],
		rough = [ round( float( ( INFO.get( nm ) or {} ).get( 'mean_rough', 0.6 ) ), 3 ) for nm in LAYERS.names ],
		metal = [ round( float( ( INFO.get( nm ) or {} ).get( 'mean_metal', 0 ) ), 3 ) for nm in LAYERS.names ],
		chunks = ordered,
		halos = halos, flames = flames, lights = lights,
		# signs: the rows of `lights` that are a sign's, already times k (LOOK[ 'neon' ]); their surfaces carry aux 65533 / 65532
		signs = dict( k = float( LOOK.get( 'neon', 1 ) ), lights = [ int( i ) for i in np.nonzero( LIGHTS[ 'sign' ] )[ 0 ] ] ),
		**( dict( look2 = look2 ) if look2 else {} ),
		stats = stats,
		# what the walk needs of town.json (navigation is analytic)
		data = { k: TOWN.get( k ) for k in ( 'seed', 'frame', 'spawn', 'shots', 'regions', 'colliders', 'missing', 'stats' ) },
	)
	json.dump( man, open( os.path.join( raw, 'raw.json' ), 'w' ), separators = ( ',', ':' ) )
	log( 'raw: %d chunks (%d with a near level), %d halos, %d flames' % ( len( ordered ), sum( 1 for c in ordered if 'near' in c ), len( halos ), len( flames ) ) )

# ( 'lib' in the config: tools/next/lightmap_bl.py runs this file for its functions alone)
if not CFG.get( 'lib' ): main()
