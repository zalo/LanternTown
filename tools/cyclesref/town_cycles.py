# Path-traced reference of the real-time town: assembles out/town/town.json in Blender 4.1 and renders named shots
# with Cycles. A ground truth for the page's lighting, not new art direction.
#
#   cd <project> && blender --background --factory-startup --threads 14 --python tools/cyclesref/town_cycles.py -- \
#       shots=02_gate_view,05_fountain_close out=out/renders/_town/cycles spp=256
#
# options (key=value after --):
#   shots=a,b        shot names of town.json                     out=dir     output folder (default out/renders/_town/cycles)
#   spp=256          samples per pixel                           size=1600x900
#   device=GPU       GPU (OptiX; the script stops with an error rather than fall back to the CPU) | CPU only when asked for
#   texlimit=        1024 / 2048 ...: Cycles' texture size limit, to make room on the card     need=6.8  GB of RAM Cycles is expected to take
#   far=0            metres: protos farther than this from every requested camera use their .lod1.glb (0 = all finest)
#   sky=1 moon=1 lamp=1   multipliers on the calibrated sky / moon / lantern scale       helpers=1  0 drops town.json src 0 lights
#   agx=0            exposure (stops) of the AgX pictures        dry=1       build only, print material statistics
#
# Runs in two stages inside ONE command: the render (GPU), then the same Blender is exec'ed again with stage=post to
# make the pictures (post.py), so the card and the scene's memory are released as soon as the last frame is on disk.
# Written per shot: <shot>.exr (linear, denoised Combined + raw 'Noisy Image' + raw light passes + light groups),
# <shot>_page.png, <shot>_page_nofog.png, <shot>_agx.png, <shot>_agx_glare.png, <shot>_passes.png, passes/<shot>_<pass>.png,
# and cycles_stats.json (shares of the image-mean luminance from the RAW passes, device proof, memory, times).
#
# What is the page's own: geometry (finest GLB of every proto, terrain, water; loaders of tools/webbake_bl.py), baked
# material maps and the weathering / wet paving / tint logic of web/engine/materials.js as Cycles nodes with the
# page's own tiling noise texture (mats.py), emissive colours and strengths, lantern positions and colours, moon
# direction and colour, sky colours, cameras. What is the path tracer's: all light transport.
# Light calibration (calib.py measured Cycles' units): sun strength = three.js DirectionalLight irradiance (1:1);
# point light watts = 4 pi x candela; a world radiance L lights an open surface with pi L.
import bpy, sys, os, json, math, time, re, resource, subprocess, threading
import numpy as np
from mathutils import Matrix, Vector

T0 = time.time()
HERE = os.path.dirname( os.path.abspath( __file__ ) )
ROOT = os.path.abspath( os.path.join( HERE, '..', '..' ) )
SCRATCH = os.path.join( ROOT, '.scratch', 'cyclesref' )
ARGS = dict( a.split( '=', 1 ) for a in sys.argv[ sys.argv.index( '--' ) + 1: ] if '=' in a ) if '--' in sys.argv else {}
ARGV_USER = list( sys.argv[ sys.argv.index( '--' ) + 1: ] ) if '--' in sys.argv else []
OUT = os.path.abspath( os.path.join( ROOT, ARGS.get( 'out', 'out/renders/_town/cycles' ) ) )
SPP = int( ARGS.get( 'spp', 256 ) )
W, H = [ int( t ) for t in ARGS.get( 'size', '1600x900' ).split( 'x' ) ]
FAR = float( ARGS.get( 'far', 0 ) )
os.makedirs( SCRATCH, exist_ok = True ); os.makedirs( OUT, exist_ok = True )

def log( *a ):

	print( '[cyclesref %6.1fs]' % ( time.time() - T0 ), *a ); sys.stdout.flush()

def mem_available_gb():

	for l in open( '/proc/meminfo' ):

		if l.startswith( 'MemAvailable' ): return int( l.split()[ 1 ] ) / 1048576

	return 0.0

def rss_gb():

	for l in open( '/proc/self/status' ):

		if l.startswith( 'VmRSS' ): return int( l.split()[ 1 ] ) / 1048576

	return 0.0

def trim():

	# the loaders leave a lot of freed python heap behind: hand it back before Cycles takes its share
	import gc, ctypes
	gc.collect()
	try: ctypes.CDLL( 'libc.so.6' ).malloc_trim( 0 )
	except Exception: pass

def die( msg ):

	# a failure nobody can miss: message on both streams, non-zero exit
	for f in ( sys.stdout, sys.stderr ): print( '[cyclesref] FAILED: ' + msg, file = f ); f.flush()
	os._exit( 2 )

def gpu_apps():

	# the processes that hold memory on the card (a CPU render does not appear here)
	try: return [ l.strip() for l in subprocess.run( [ 'nvidia-smi', '--query-compute-apps=pid,used_memory,name', '--format=csv,noheader' ], capture_output = True, text = True, timeout = 10 ).stdout.splitlines() if l.strip() ]
	except Exception: return []

def vram_used():

	try: return int( subprocess.run( [ 'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits' ], capture_output = True, text = True, timeout = 10 ).stdout.split()[ 0 ] )
	except Exception: return - 1

# ------------------------------------------------------------------ the page's constants + the bake's loaders
sys.path.insert( 0, os.path.join( ROOT, 'tools' ) ); sys.path.insert( 0, HERE )
import webbake
LOOK_PAGE = webbake.engine_look()
cfg_path = os.path.join( SCRATCH, 'wb_cfg.json' )
json.dump( dict( root = ROOT, out = os.path.join( SCRATCH, 'wb_out' ), work = os.path.join( SCRATCH, 'wb_work' ), lib = 1, look = LOOK_PAGE ), open( cfg_path, 'w' ) )
_argv = sys.argv
sys.argv = [ 'blender', '--', cfg_path ]
import webbake_bl as wb
sys.argv = _argv
import mats, post
log( 'loaders imported: process memory %.2f GB' % rss_gb() )

def page_number( src, pat, default ):

	m = re.search( pat, src )
	return float( m.group( 1 ) ) if m else default

MAIN_JS = open( os.path.join( ROOT, 'web/town/main.js' ) ).read()
PG = dict( post.PAGE )
for k, pat in dict( exposure = r'\bexposure:\s*([\d.]+)', fog = r'\bfog:\s*([\d.]+)', haze = r'\bhaze:\s*([\d.]+)', contrast = r'P\.contrast(?:\.value)?\s*=\s*([\d.]+)',
	saturation = r'P\.saturation(?:\.value)?\s*=\s*([\d.]+)', knee = r'P\.knee(?:\.value)?\s*=\s*([\d.]+)', knee_glow = r'P\.kneeGlow(?:\.value)?\s*=\s*([\d.]+)', bloom = r'\bbloom:\s*([\d.]+)' ).items():
	PG[ k ] = page_number( MAIN_JS, pat, PG[ k ] )
SKY = dict( cover = page_number( MAIN_JS, r'\bcover:\s*([\d.]+)', 0.46 ), cloudGain = page_number( MAIN_JS, r'\bcloudGain:\s*([\d.]+)', 1.0 ),
	skyGlow = page_number( MAIN_JS, r'\bskyGlow:\s*([\d.]+)', 1.0 ), clouds = page_number( MAIN_JS, r'\bclouds:\s*([\d.]+)', 1.0 ) )
wb.LOOK[ 'wet_flat' ] = page_number( MAIN_JS, r'wetFlat(?:\.value)?\s*=\s*([\d.]+)', 0.7 )
log( 'page look', json.dumps( LOOK_PAGE ) )
log( 'page picture', json.dumps( PG ), json.dumps( SKY ) )

TOWN = wb.TOWN
SHOTS = { s[ 'name' ]: s for s in ( TOWN[ 'shots' ] if isinstance( TOWN[ 'shots' ], list ) else TOWN[ 'shots' ].values() ) }
WANT = [ s for s in ARGS.get( 'shots', '02_gate_view' ).split( ',' ) if s ]
for s in WANT:

	if s not in SHOTS: raise SystemExit( 'unknown shot ' + s )

OUT_STATS = os.path.join( OUT, 'cycles_stats.json' )
PENDING = os.path.join( SCRATCH, 'pending.json' )
AGX_EXP = float( ARGS.get( 'agx', 0.0 ) )

def make_pictures():

	# second stage, in a fresh process that never touched the card: the page / AgX pictures, the pass pictures and the shares
	pend = json.load( open( PENDING ) )
	ALL = json.load( open( OUT_STATS ) ) if os.path.exists( OUT_STATS ) else {}
	log( 'pictures: this process on the card: %r' % ( [ l for l in gpu_apps() if l.split( ',' )[ 0 ].strip() == str( os.getpid() ) ], ) )
	for name, exr, extra in pend[ 'done' ]:

		res = post.process( exr, name, OUT, SHOTS[ name ][ 'pos' ], PG, AGX_EXP, log )
		res.update( extra )
		ALL[ name ] = res; ALL[ '_scene' ] = pend[ 'scene' ]
		json.dump( ALL, open( OUT_STATS, 'w' ), indent = 1 )
		log( '%s shares %s' % ( name, json.dumps( res[ 'share' ] ) ) ); log( '%s light groups %s' % ( name, json.dumps( res[ 'lightgroups' ] ) ) )

	os.remove( PENDING )
	log( 'done' )

if ARGS.get( 'stage' ) == 'post': make_pictures(); sys.stdout.flush(); os._exit( 0 )

C3 = np.array( [ [ 1, 0, 0 ], [ 0, 0, - 1 ], [ 0, 1, 0 ] ], float ) # three (x, y, z) -> Blender (x, -z, y)

def t2b( P ): return np.stack( [ P[ :, 0 ], - P[ :, 2 ], P[ :, 1 ] ], 1 )

# ------------------------------------------------------------------ scene
for o in list( bpy.data.objects ): bpy.data.objects.remove( o )
for m in list( bpy.data.meshes ): bpy.data.meshes.remove( m )
scene = bpy.context.scene
LIB = mats.Lib( wb, ROOT )
COLL = bpy.data.collections.new( 'town' ); scene.collection.children.link( COLL )
PK = set()
STATS = dict( tris = 0, tris_instanced = 0, protos = 0, items = 0, missing = [], lod1 = [] )

def build_mesh( name, prims ):

	prims = [ p for p in prims if len( p[ 'I' ] ) ]
	if not prims: return None
	nv = sum( len( p[ 'P' ] ) for p in prims ); nt = sum( len( p[ 'I' ] ) for p in prims )
	P = np.empty( ( nv, 3 ), np.float32 ); N = np.empty( ( nv, 3 ), np.float32 ); UV = np.empty( ( nv, 2 ), np.float32 ); MK = np.empty( ( nv, 4 ), np.float32 )
	I = np.empty( ( nt, 3 ), np.int32 ); MI = np.empty( nt, np.int32 )
	me = bpy.data.meshes.new( name )
	slots = {}
	v0 = t0 = 0
	for p in prims:

		n = len( p[ 'P' ] ); k = len( p[ 'I' ] )
		P[ v0:v0 + n ] = t2b( p[ 'P' ] ); N[ v0:v0 + n ] = t2b( p[ 'N' ] )
		UV[ v0:v0 + n, 0 ] = p[ 'UV' ][ :, 0 ]; UV[ v0:v0 + n, 1 ] = 1 - p[ 'UV' ][ :, 1 ]
		MK[ v0:v0 + n ] = p[ 'MASK' ][ :, :4 ] if p[ 'MASK' ].shape[ 1 ] >= 4 else np.pad( p[ 'MASK' ], ( ( 0, 0 ), ( 0, 4 - p[ 'MASK' ].shape[ 1 ] ) ) )
		I[ t0:t0 + k ] = p[ 'I' ] + v0
		r0 = rss_gb()
		mat = LIB.get( p[ 'lib' ], p[ 'tint' ], p[ 'params' ] or {} )
		STATS[ 'mat_gb' ] = STATS.get( 'mat_gb', 0.0 ) + rss_gb() - r0
		PK.add( ( p[ 'lib' ], json.dumps( p[ 'params' ] or {}, sort_keys = True ) ) )
		if mat.name not in slots: slots[ mat.name ] = len( slots ); me.materials.append( mat )
		MI[ t0:t0 + k ] = slots[ mat.name ]
		v0 += n; t0 += k

	# degenerate triangles upset the tangent pass and the validator: drop them here
	ok = ( I[ :, 0 ] != I[ :, 1 ] ) & ( I[ :, 1 ] != I[ :, 2 ] ) & ( I[ :, 0 ] != I[ :, 2 ] )
	if not ok.all(): I = I[ ok ]; MI = MI[ ok ]; nt = len( I )
	me.vertices.add( nv ); me.vertices.foreach_set( 'co', P.ravel() )
	me.loops.add( nt * 3 ); me.loops.foreach_set( 'vertex_index', I.ravel() )
	me.polygons.add( nt ); me.polygons.foreach_set( 'loop_start', np.arange( nt, dtype = np.int32 ) * 3 )
	try: me.polygons.foreach_set( 'loop_total', np.full( nt, 3, np.int32 ) )
	except Exception: pass
	me.polygons.foreach_set( 'material_index', MI )
	me.polygons.foreach_set( 'use_smooth', np.ones( nt, bool ) )
	uv = me.uv_layers.new( name = 'UVMap' )
	uv.data.foreach_set( 'uv', UV[ I.ravel() ].ravel() )
	at = me.attributes.new( 'tsq_mask', 'BYTE_COLOR', 'POINT' ) # (8 bits, as the page's vertex attribute; a float colour costs 1 GB more through Blender and Cycles)
	at.data.foreach_set( 'color', np.clip( MK, 0, 1 ).ravel() )
	me.update( calc_edges = True )
	ln = np.linalg.norm( N, axis = 1, keepdims = True )
	N = np.where( ln > 1e-6, N / np.maximum( ln, 1e-6 ), np.array( [ [ 0, 0, 1 ] ], np.float32 ) )
	me.normals_split_custom_set_from_vertices( N.tolist() )
	return me, nt

def camera_positions(): return [ np.array( SHOTS[ s ][ 'pos' ], float ) for s in WANT ]

def build_town():

	cams = camera_positions()
	items_of = {}
	for it in TOWN[ 'items' ]: items_of.setdefault( it[ 'p' ], [] ).append( it )
	meshes = {}
	for key, its in items_of.items():

		p = TOWN[ 'protos' ].get( key )
		if not p: continue
		lod = 0
		if FAR > 0 and 'min' in p and 'max' in p:

			rad = 0.5 * float( np.linalg.norm( np.array( p[ 'max' ] ) - np.array( p[ 'min' ] ) ) )
			dmin = 1e9
			for it in its:

				c = C3 @ wb.item_center( it, p )
				dmin = min( dmin, min( float( np.linalg.norm( c - cp ) ) for cp in cams ) - rad * ( it.get( 's', 1 ) or 1 ) )

			if dmin > FAR and os.path.exists( wb.proto_path( key, p, 1 ) ) and len( p.get( 'tris' ) or [] ) > 1: lod = 1; STATS[ 'lod1' ].append( key )

		path = wb.proto_path( key, p, lod )
		if not os.path.exists( path ): STATS[ 'missing' ].append( key ); continue
		r = build_mesh( key.replace( '/', '.' ), wb.load_glb( path ) )
		wb._GLB.pop( path, None )
		if not r: continue
		me, nt = r
		meshes[ key ] = me
		STATS[ 'tris' ] += nt; STATS[ 'protos' ] += 1
		for it in its:

			R, sc, T = wb.item_xform( it )
			M = np.eye( 4 ); M[ :3, :3 ] = ( C3 @ R @ C3.T ) * sc; M[ :3, 3 ] = C3 @ np.asarray( T, float )
			ob = bpy.data.objects.new( key.split( '/' )[ 0 ], me )
			ob.matrix_world = Matrix( M.tolist() )
			ob.lightgroup = 'emit'
			COLL.objects.link( ob )
			STATS[ 'tris_instanced' ] += nt; STATS[ 'items' ] += 1

		if STATS[ 'protos' ] % 40 == 0: log( '  %d protos, %.2f M triangles, %d materials, %.1f GB free' % ( STATS[ 'protos' ], STATS[ 'tris' ] / 1e6, len( LIB.cache ), mem_available_gb() ) )

	terr = {}
	for t in TOWN.get( 'terrain', [] ):

		path = os.path.join( ROOT, 'out/glb', t[ 'file' ] + '.glb' )
		if not os.path.exists( path ): STATS[ 'missing' ].append( t[ 'file' ] ); continue
		r = build_mesh( t[ 'file' ].replace( '/', '.' ), wb.load_glb( path ) )
		wb._GLB.pop( path, None )
		if not r: continue
		ob = bpy.data.objects.new( t[ 'file' ].split( '/' )[ - 1 ], r[ 0 ] ); ob.lightgroup = 'emit'
		COLL.objects.link( ob ); terr[ t[ 'file' ] ] = ob
		STATS[ 'tris' ] += r[ 1 ]; STATS[ 'tris_instanced' ] += r[ 1 ]

	return terr

# ------------------------------------------------------------------ lights
def page_falloff( h, rng, L ):

	# what the page multiplies the inverse square by, for a point straight below a lamp (ClusteredLightsNode / webbake_bl light_range)
	sr = L[ 'source_radius' ]; reach = L[ 'reach' ]
	return ( h * h / ( h * h + sr * sr ) ) * max( 0.0, 1 - ( h / rng ) ** 4 ) ** 2 / ( 1 + h * h / ( reach * reach ) if reach > 0 else 1 )

def build_lights( terr ):

	L = wb.LOOK
	lights = TOWN.get( 'lights', [] )
	# ONE global lantern scale: the page's falloff relative to the inverse square on the paving straight below each
	# street lamp, intensity-weighted median over the lamps that stand 1.5 - 8 m over the terrain
	ground = terr.get( '_town/terrain' )
	hs = []
	if ground is not None:

		# (a BVH of the terrain alone: Object.ray_cast would evaluate the whole scene, 1.6 GB of copies the render never uses)
		from mathutils.bvhtree import BVHTree
		gm = ground.data; gv = np.empty( len( gm.vertices ) * 3, np.float32 ); gm.vertices.foreach_get( 'co', gv )
		bvh = BVHTree.FromPolygons( [ tuple( v ) for v in gv.reshape( - 1, 3 ).tolist() ], [ tuple( p.vertices ) for p in gm.polygons ] )
		inv = ground.matrix_world.inverted()
		for l in lights:

			if l.get( 'src' ) == 0: continue
			loc, nrm, idx, dist = bvh.ray_cast( inv @ Vector( l[ 'pos' ] ), Vector( ( 0, 0, - 1 ) ) )
			ok = loc is not None
			if ok: loc = ground.matrix_world @ loc
			if ok and 1.5 <= l[ 'pos' ][ 2 ] - loc[ 2 ] <= 8.0: hs.append( ( page_falloff( l[ 'pos' ][ 2 ] - loc[ 2 ], l.get( 'range' ) or 9, L ), l.get( 'intensity', 0 ), l[ 'pos' ][ 2 ] - loc[ 2 ] ) )

	if hs:

		hs.sort(); tot = sum( w for _, w, _ in hs ); acc = 0; k_lamp = hs[ - 1 ][ 0 ]
		for f, w, h in hs:

			acc += w
			if acc >= tot * 0.5: k_lamp = f; break

		h_med = float( np.median( [ h for _, _, h in hs ] ) )

	else: k_lamp = page_falloff( 3.2, 10, L ); h_med = 3.2
	k_user = float( ARGS.get( 'lamp', 1 ) )
	datas = {}
	n_h = 0
	helpers = ARGS.get( 'helpers', '1' ) != '0'
	lc = bpy.data.collections.new( 'lights' ); scene.collection.children.link( lc )
	for i, l in enumerate( lights ):

		src = 0 if l.get( 'src' ) == 0 else 1
		if not src and not helpers: continue
		col = wb.lin( l.get( 'color' ) or '#ffbf6a' )
		cd = l.get( 'intensity', 0 ) * L[ 'lamp' ] * k_lamp * k_user # candela, three.js units
		if cd <= 0: continue
		near = float( l.get( 'near' ) or 0.03 )
		key = ( round( cd, 4 ), tuple( round( float( c ), 4 ) for c in col ), round( near, 3 ) )
		d = datas.get( key )
		if d is None:

			d = datas[ key ] = bpy.data.lights.new( 'lamp', 'POINT' )
			d.color = tuple( float( c ) for c in col[ :3 ] )
			d.energy = 4 * math.pi * cd
			d.shadow_soft_size = max( near, 0.03 ) # the page's `near`: what is closer to the burner than this casts no shadow
			try: d.use_soft_falloff = True
			except Exception: pass

		ob = bpy.data.objects.new( 'L%03d_%s' % ( i, l.get( 'name' ) or '' ), d )
		ob.location = l[ 'pos' ]
		ob.visible_camera = False
		ob.lightgroup = 'lanterns' if src else 'helpers'
		lc.objects.link( ob )
		n_h += 0 if src else 1

	# moon: three.js DirectionalLight intensity x colour = irradiance = Cycles sun strength
	m3 = np.array( L[ 'moon_dir' ], float ); m3 /= np.linalg.norm( m3 )
	mb = C3 @ m3 # direction towards the moon
	sd = bpy.data.lights.new( 'moon', 'SUN' )
	sd.color = tuple( L[ 'moon_col' ] ); sd.energy = L[ 'moon' ] * float( ARGS.get( 'moon', 1 ) ); sd.angle = math.radians( 0.6 )
	so = bpy.data.objects.new( 'moon', sd )
	so.rotation_euler = Vector( mb ).to_track_quat( 'Z', 'Y' ).to_euler()
	so.lightgroup = 'moon'; so.visible_camera = False
	lc.objects.link( so )
	return dict( lantern_scale = k_lamp * k_user, lantern_scale_page = k_lamp, lamps_measured = len( hs ), lamp_height_median = h_med, lights = len( lc.objects ) - 1, helpers = n_h,
		light_datablocks = len( datas ), moon_strength = sd.energy, moon_dir_blender = [ float( t ) for t in mb ] )

def build_world():

	L = wb.LOOK
	w = bpy.data.worlds.new( 'tsq_night' ); w.use_nodes = True
	nt = w.node_tree; nt.nodes.clear()
	g = mats.G( nt )
	out = g.new( 'ShaderNodeOutputWorld' )
	tc = g.new( 'ShaderNodeTexCoord' )
	dx, dy, dz = g.sep( tc.outputs[ 'Generated' ] )
	sky = np.array( L[ 'sky' ], float ); gnd = np.array( L[ 'ground' ], float ); amb = L[ 'ambient' ] * float( ARGS.get( 'sky', 1 ) )
	# the radiance field whose irradiance is three.js' HemisphereLight: E(n) = ambient * mix( ground, sky, 0.5 n.up + 0.5 )
	# <=> L(w) = c0 + c1 (w.up). An open, level surface receives exactly ambient * sky from it; the ground half of
	# the page's fill is left to the real bounce off the real ground (which hides the lower half of this field).
	c0 = amb * ( sky + gnd ) / ( 2 * math.pi ); c1 = amb * ( sky - gnd ) * 3 / ( 4 * math.pi )
	light = g.va( c0, g.vs( c1, dz ) )
	# what the camera sees: the page's sky (createSky of engine.js: gradient, town glow, cloud deck from the same noise
	# texture at time 0; no stars, no moon disc)
	h = dz; hs = g.sat( h ); hh = g.mx( h, 0.0 )
	horizon = np.array( [ 0.018, 0.034, 0.085 ] ); zenith = np.array( [ 0.004, 0.009, 0.03 ] )
	glow = np.array( [ 0.026, 0.016, 0.008 ] ) * SKY[ 'skyGlow' ]; cloudc = np.array( [ 0.05, 0.075, 0.14 ] )
	col = g.vmix( horizon, zenith, g.pw( hs, 0.42 ) )
	col = g.va( col, g.va( g.vs( glow, g.ex( g.mul( hh, - 7.0 ) ) ), g.vs( glow * np.array( [ 0.3, 0.42, 0.85 ] ), g.ex( g.mul( hh, - 2.4 ) ) ) ) )
	col = g.vmix( col, np.array( [ 0.006, 0.009, 0.018 ] ), g.ss( 0.0, - 0.25, h ) )
	den = g.add( hh, 0.2 )
	qx = g.div( dx, den ); qy = g.div( g.mul( dy, - 1.0 ), den )
	def nz( u, v ):

		c_, a_ = g.tex( LIB.noise, u, v )
		r_, g_, b_ = g.sep( c_ )
		return r_, g_, b_, a_

	def dens( px, py ):

		a = nz( g.mul( px, 0.16 ), g.mul( py, 0.16 ) )[ 0 ]
		b = nz( g.mul( g.add( g.mul( px, 0.44 ), g.mul( py, 0.9 ) ), 0.42 ), g.mul( g.sub( g.mul( py, 0.44 ), g.mul( px, 0.9 ) ), 0.42 ) )[ 1 ]
		return g.add( g.mul( a, 0.6 ), g.mul( b, 0.4 ) )

	m3 = np.array( L[ 'moon_dir' ], float ); m3 /= np.linalg.norm( m3 )
	mxz = np.array( [ m3[ 0 ], m3[ 2 ] ] ); mxz = mxz / np.linalg.norm( mxz ) * 0.3
	n0 = dens( qx, qy ); n1 = dens( g.add( qx, float( mxz[ 0 ] ) ), g.add( qy, float( mxz[ 1 ] ) ) )
	fine = nz( g.mul( qx, 1.3 ), g.mul( qy, 1.3 ) )
	nn = g.add( n0, g.ma( fine[ 2 ], 0.14, - 0.07 ) )
	cover = SKY[ 'cover' ]
	cloud = g.mul( g.mul( g.ss( cover, cover + 0.2, nn ), g.ss( - 0.02, 0.06, h ) ), SKY[ 'clouds' ] )
	thin = g.om( g.ss( cover + 0.08, cover + 0.3, nn ) )
	side = g.sat( g.ma( g.sub( n0, n1 ), 7.0, 0.45 ) )
	md = g.add( g.add( g.mul( dx, float( m3[ 0 ] ) ), g.mul( dz, float( m3[ 1 ] ) ) ), g.mul( dy, - float( m3[ 2 ] ) ) )
	to_moon = g.pw( g.sat( g.ma( md, 0.5, 0.5 ) ), 3.0 )
	k = g.add( g.mul( g.ma( g.mul( g.mul( side, g.ma( thin, 0.75, 0.25 ) ), g.ma( fine[ 3 ], 0.9, 0.55 ) ), 1.1, 0.16 ), g.ma( to_moon, 1.4, 0.75 ) ),
		g.mul( g.mul( g.pw( g.sat( md ), 14.0 ), thin ), 2.5 ) )
	lit = g.va( g.vs( cloudc, k ), g.vs( glow * 0.8, g.ex( g.mul( hh, - 4.5 ) ) ) )
	col = g.vmix( col, g.vs( lit, SKY[ 'cloudGain' ] ), cloud )
	b1 = g.new( 'ShaderNodeBackground' ); g.put( b1.inputs[ 0 ], light ); b1.inputs[ 1 ].default_value = 1.0
	b2 = g.new( 'ShaderNodeBackground' ); g.put( b2.inputs[ 0 ], col ); b2.inputs[ 1 ].default_value = 1.0
	lp = g.new( 'ShaderNodeLightPath' )
	mx = g.new( 'ShaderNodeMixShader' )
	nt.links.new( lp.outputs[ 'Is Camera Ray' ], mx.inputs[ 0 ] ); nt.links.new( b1.outputs[ 0 ], mx.inputs[ 1 ] ); nt.links.new( b2.outputs[ 0 ], mx.inputs[ 2 ] )
	nt.links.new( mx.outputs[ 0 ], out.inputs[ 0 ] )
	w.lightgroup = 'sky'
	w.light_settings.distance = 1.0 # ambient occlusion pass: 1 m
	scene.world = w
	return dict( sky_up_radiance = [ float( t ) for t in c0 + c1 ], sky_horizon_radiance = [ float( t ) for t in c0 ], open_level_irradiance = [ float( t ) for t in amb * sky ] )

REN_INFO = {}
VRAM_CAP = 14000 # MiB, the whole card while this renders (ComfyUI shares it)
VRAM_SCENE = 6500 # MiB this town takes on the card (measured: 5.7 - 6.1 GB)

def setup_render():

	r = scene.render; c = scene.cycles
	r.engine = 'CYCLES'
	r.resolution_x = W; r.resolution_y = H; r.resolution_percentage = 100
	r.use_persistent_data = True
	r.use_motion_blur = False
	r.film_transparent = False
	r.use_compositing = False; r.use_sequencer = False
	c.samples = SPP; c.use_adaptive_sampling = False
	c.max_bounces = 12; c.diffuse_bounces = 8; c.glossy_bounces = 8; c.transmission_bounces = 12; c.transparent_max_bounces = 48; c.volume_bounces = 0
	c.caustics_reflective = False; c.caustics_refractive = False
	c.sample_clamp_indirect = 10.0; c.sample_clamp_direct = 0.0
	c.use_light_tree = True
	c.use_denoising = True; c.denoiser = 'OPENIMAGEDENOISE'; c.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
	try: c.denoising_prefilter = 'ACCURATE'
	except Exception: pass
	c.pixel_filter_type = 'BLACKMAN_HARRIS'; c.filter_width = 1.5
	scene.view_settings.view_transform = 'Standard'
	vl = scene.view_layers[ 0 ]
	for a in ( 'use_pass_combined', 'use_pass_z', 'use_pass_normal', 'use_pass_position', 'use_pass_diffuse_direct', 'use_pass_diffuse_indirect', 'use_pass_diffuse_color',
		'use_pass_glossy_direct', 'use_pass_glossy_indirect', 'use_pass_glossy_color', 'use_pass_transmission_direct', 'use_pass_transmission_indirect', 'use_pass_transmission_color',
		'use_pass_emit', 'use_pass_environment', 'use_pass_ambient_occlusion' ): setattr( vl, a, True )
	for n in ( 'lanterns', 'helpers', 'emit', 'moon', 'sky' ): vl.lightgroups.add( name = n )
	r.image_settings.file_format = 'OPEN_EXR_MULTILAYER'; r.image_settings.color_depth = '16'; r.image_settings.exr_codec = 'PIZ'
	dev = ARGS.get( 'device', 'AUTO' ).upper()
	used = vram_used()
	if dev == 'AUTO': dev = 'GPU' # the GPU always; when the card is short of room the answer is a leaner scene (far=, texlimit=), not the CPU
	name = 'CPU'
	if ARGS.get( 'texlimit' ): r.use_simplify = True; c.texture_limit_render = str( ARGS[ 'texlimit' ] )
	if dev == 'GPU':

		if used < 0: die( 'device=GPU: nvidia-smi does not answer' )
		if used + VRAM_SCENE > VRAM_CAP: die( 'device=GPU: %d MiB of the card are in use, this town takes about %d more and the total has to stay under %d MiB. Rerun with texlimit=1024 or far=<m>, not on the CPU' % ( used, VRAM_SCENE, VRAM_CAP ) )
		prefs = bpy.context.preferences.addons[ 'cycles' ].preferences
		prefs.compute_device_type = 'OPTIX'; prefs.get_devices()
		for d in prefs.devices: d.use = d.type == 'OPTIX'
		on = [ d.name for d in prefs.devices if d.use ]
		if not on: die( 'device=GPU: no OptiX device could be enabled (devices: %r)' % ( [ ( d.name, d.type ) for d in prefs.devices ], ) )
		c.device = 'GPU'
		name = 'GPU OptiX: ' + ', '.join( on )
		c.denoising_use_gpu = True
		oidn_gpu = None
		try:

			import _cycles
			oidn_gpu = [ tuple( t ) for t in _cycles.available_devices( 'OPTIX' ) ]

		except Exception as e: oidn_gpu = repr( e )
		REN_INFO[ 'available_devices' ] = oidn_gpu

	elif dev == 'CPU': c.device = 'CPU'; name = 'CPU'
	else: die( 'device=%s: GPU or CPU' % dev )
	if c.denoiser != 'OPENIMAGEDENOISE':

		# (this build would not take OpenImageDenoise)
		try: c.denoiser = 'OPTIX'
		except Exception as e: die( 'no denoiser: %r' % ( e, ) )

	REN_INFO.update( denoiser_setting = c.denoiser, denoising_use_gpu = bool( getattr( c, 'denoising_use_gpu', False ) ), denoising_input_passes = c.denoising_input_passes, want = dev )
	return dict( REN_INFO, device = name, vram_used_before_mib = used, spp = SPP, size = [ W, H ],
		bounces = dict( total = c.max_bounces, diffuse = c.diffuse_bounces, glossy = c.glossy_bounces, transmission = c.transmission_bounces, transparent = c.transparent_max_bounces ),
		clamp_indirect = c.sample_clamp_indirect, light_tree = c.use_light_tree, caustics = False, denoiser = ( 'OpenImageDenoise' if c.denoiser == 'OPENIMAGEDENOISE' else 'OptiX' ) + ' (albedo + normal guides, %s)' % ( 'on the GPU' if getattr( c, 'denoising_use_gpu', False ) and c.device == 'GPU' else 'on the CPU' ) )

def set_camera( shot ):

	cam = scene.camera
	if cam is None:

		cd = bpy.data.cameras.new( 'cam' ); cam = bpy.data.objects.new( 'cam', cd ); scene.collection.objects.link( cam ); scene.camera = cam

	cd = cam.data
	cd.sensor_fit = 'VERTICAL'; cd.angle = math.radians( float( shot.get( 'fov' ) or 62 ) ) # three.js fov = vertical
	cd.clip_start = 0.08; cd.clip_end = 1500
	cd.dof.use_dof = False
	pos = Vector( shot[ 'pos' ] ); tgt = Vector( shot[ 'target' ] )
	cam.location = pos
	cam.rotation_euler = ( tgt - pos ).to_track_quat( '-Z', 'Y' ).to_euler()
	if shot.get( 'shift' ): log( 'NOTE: shot %s has a lens shift (%r), not applied' % ( shot[ 'name' ], shot[ 'shift' ] ) )

class VramWatch( threading.Thread ):

	def __init__( self ): super().__init__( daemon = True ); self.peak = vram_used(); self.min_ram = mem_available_gb(); self.on = True; self.rendering = False; self.mine = ( 0, '' )
	def run( self ):

		pid = str( os.getpid() )
		while self.on:

			self.peak = max( self.peak, vram_used() ); self.min_ram = min( self.min_ram, mem_available_gb() )
			if self.rendering:

				for l in gpu_apps():

					f = [ t.strip() for t in l.split( ',' ) ]
					if f and f[ 0 ] == pid:

						try: mib = int( f[ 1 ].split()[ 0 ] )
						except Exception: mib = 0
						if mib >= self.mine[ 0 ]: self.mine = ( mib, l )

			time.sleep( 1.0 )

# ------------------------------------------------------------------ run
log( 'building the town (%d items, %d protos)' % ( len( TOWN[ 'items' ] ), len( TOWN[ 'protos' ] ) ) )
terr = build_town()
log( 'geometry: %d protos, %d items, %.2f M unique triangles (%.2f M instanced), %d materials, missing %r, lod1 %d' % ( STATS[ 'protos' ], STATS[ 'items' ], STATS[ 'tris' ] / 1e6, STATS[ 'tris_instanced' ] / 1e6, len( LIB.cache ), STATS[ 'missing' ], len( STATS[ 'lod1' ] ) ) )
log( 'process memory after the build %.2f GB' % rss_gb() ); trim(); log( 'after trim %.2f GB, %.1f GB free' % ( rss_gb(), mem_available_gb() ) )
if ARGS.get( 'dry' ):

	import collections
	nn = [ ( len( m.node_tree.nodes ), m.name ) for m in bpy.data.materials if m.node_tree ]
	log( 'materials %d, nodes %d, largest %r, memory taken while making materials %.2f GB, distinct (lib, params) %d' % ( len( nn ), sum( n for n, _ in nn ), sorted( nn )[ - 5: ], STATS.get( 'mat_gb', 0 ), len( PK ) ) )
	log( 'by lib: %r' % ( collections.Counter( k[ 0 ] for k in LIB.cache ).most_common( 60 ), ) )
	raise SystemExit( 0 )

REN = setup_render()
LI = build_lights( terr )
WO = build_world()
log( 'lights', json.dumps( LI ) ); log( 'world', json.dumps( WO ) ); log( 'render', json.dumps( REN ) )
NEED_GB = float( ARGS.get( 'need', 6.8 ) ) # what Cycles takes on top of the Blender scene for this town (measured)
if mem_available_gb() - NEED_GB < 4.0: die( 'rendering would leave less than 4 GB of memory free (%.1f GB free now, Cycles needs about %.1f more): not started' % ( mem_available_gb(), NEED_GB ) )
if ARGS.get( 'save' ): bpy.ops.wm.save_as_mainfile( filepath = os.path.join( SCRATCH, 'town.blend' ) )

watch = VramWatch(); watch.start()
DONE = []
for name in WANT:

	shot = SHOTS[ name ]
	set_camera( shot )
	exr = os.path.join( OUT, name + '.exr' )
	scene.render.filepath = exr
	t = time.time()
	cprefs = bpy.context.preferences.addons[ 'cycles' ].preferences
	proof = dict( compute_device_type = cprefs.compute_device_type, enabled_devices = [ '%s (%s)' % ( d.name, d.type ) for d in cprefs.devices if d.use ], scene_cycles_device = scene.cycles.device,
		use_denoising = scene.cycles.use_denoising, denoiser = scene.cycles.denoiser, denoising_use_gpu = bool( getattr( scene.cycles, 'denoising_use_gpu', False ) ), denoising_input_passes = scene.cycles.denoising_input_passes )
	log( 'before render.render: %s' % json.dumps( proof ) )
	if REN[ 'want' ] == 'GPU' and ( scene.cycles.device != 'GPU' or cprefs.compute_device_type != 'OPTIX' or not any( d.use and d.type == 'OPTIX' for d in cprefs.devices ) ): die( 'device=GPU asked for, but Cycles is set to %s / %s' % ( scene.cycles.device, cprefs.compute_device_type ) )
	if not scene.cycles.use_denoising: die( 'the denoiser is off' )
	log( 'rendering %s (%d spp, %dx%d, %s)' % ( name, SPP, W, H, REN[ 'device' ] ) )
	watch.mine = ( 0, '' ); watch.rendering = True
	bpy.ops.render.render( write_still = True )
	watch.rendering = False
	dt = time.time() - t
	proof[ 'nvidia_smi_compute_app_during_render' ] = watch.mine[ 1 ]; proof[ 'own_vram_peak_mib' ] = watch.mine[ 0 ]
	log( 'nvidia-smi --query-compute-apps during the render (pid %d): %r' % ( os.getpid(), watch.mine[ 1 ] ) )
	if REN[ 'want' ] == 'GPU' and not watch.mine[ 1 ] and dt > 4: die( 'device=GPU asked for, but this process held no memory on the card while rendering %s' % name )
	log( 'rendered %s in %.1fs; peak RSS %.2f GB, peak VRAM (whole card) %d MiB, least free RAM %.1f GB' % ( name, dt, resource.getrusage( resource.RUSAGE_SELF ).ru_maxrss / 1048576, watch.peak, watch.min_ram ) )
	DONE.append( ( name, exr, dict( seconds = dt, spp = SPP, device = REN[ 'device' ], peak_rss_gb = resource.getrusage( resource.RUSAGE_SELF ).ru_maxrss / 1048576, peak_vram_card_mib = watch.peak,
		vram_before_mib = REN[ 'vram_used_before_mib' ], min_free_ram_gb = watch.min_ram, agx_exposure_stops = AGX_EXP, gpu_proof = proof, camera = dict( pos = shot[ 'pos' ], target = shot[ 'target' ], fov = shot.get( 'fov' ) or 62 ) ) ) )

watch.on = False
SCENE_INFO = dict( geometry = { k: ( v if not isinstance( v, list ) else v[ :50 ] ) for k, v in STATS.items() }, materials = len( LIB.cache ), lights = LI, world = WO, render = REN, page = PG, look = LOOK_PAGE, sky = SKY, args = ARGS )
json.dump( dict( done = DONE, scene = SCENE_INFO ), open( PENDING, 'w' ) )
if not DONE: die( 'nothing rendered' )
# The pictures are made by a second stage that REPLACES this process (exec, so there is never a second Blender): the card
# is released the moment the last frame is written (the CUDA context alone would hold 1.6 GB of it while numpy works),
# and so are the 4.6 GB of memory a Blender that has rendered this town keeps after the scene is gone.
log( 'rendered %d shots; handing over to the picture stage (this process, and its hold on the card, ends here)' % len( DONE ) )
sys.stdout.flush(); sys.stderr.flush()
try:

	import ctypes
	ctypes.CDLL( 'libc.so.6' ).fflush( None )

except Exception: pass
exe = bpy.app.binary_path
os.execv( exe, [ exe, '--background', '--factory-startup', '--threads', '14', '--python', os.path.abspath( __file__ ), '--' ] + [ a for a in ARGV_USER if not a.startswith( 'stage=' ) ] + [ 'stage=post' ] )
