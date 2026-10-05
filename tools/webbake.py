#!/usr/bin/env python3
"""Bake the town into the web-friendly build the light tiers of the walkable demo load.

    python3 tools/webbake.py                 # out/town/town.json + out/glb/** + out/mat/** -> out/web/_next/
    python3 tools/webbake.py --town town2    # another assembly: out/town2/town.json -> out/web/_next_town2/
                                             # (--town also takes a path to a town.json; --out names the directory)
    python3 tools/webbake.py --jobs 16       # lighting workers (default: cores / 2, at most 16)
    python3 tools/webbake.py --check         # exit 1 when the bake is missing or older than town.json
    python3 tools/webbake.py --keep-raw      # keep the unpacked chunks (out/web/_next/.raw, ~250 MB) ...
    python3 tools/webbake.py --pack-only     # ... to try packing parameters (--set pack.KEY=VALUE) without baking again
    python3 tools/webbake.py --only full     # one of the two bakes of a run: full (medium / high) or lean (low / phone)
    python3 tools/webbake.py --no-glb        # the packed files only (the page then loads those)
    python3 tools/webbake.py --keep-packed   # keep the packed .bin files beside the .glb files (?fmt=packed, to compare)
    python3 tools/webbake.py --pvs           # then bake the visibility sets too (tools/pvs.py: needs the page server and the GPU Chrome)
    python3 tools/webbake.py --no-gao        # without the ground's contact occlusion map (tools/aomap.py; it needs the working data,
                                             # so it is only made with --keep-raw / --pack-only)

One command, no hand steps: it follows town.json (nothing here names an asset), so it is simply re-run after
every `tools/town.py all`. The heavy part (tools/webbake_bl.py) runs inside Blender's python for numpy and the
BVH, tools/webbake_pack.mjs (node + meshoptimizer) simplifies and compresses the chunks; this file runs both,
converts the texture pages to WebP and, last, has tools/bake2glb.mjs write the geometry once more as .glb files.

Output (out/web/_next/):
    manifest.json     chunk table, texture layers, lantern glows / flames, navigation data, town fingerprint
    gao.png           the ground's contact occlusion seen from above (tools/aomap.py; "gao" in the manifest), also in lean/
    gsh.webp          the lanterns' shadows on the ground seen from above (the bake's ground_shadows(), stored by tools/aomap.py;
                      "gsh" in the manifest; half the size in lean/). The ground's vertices hold the lanterns' light without
                      shadows then: --set gsm=0 bakes them into the vertices as before (no picture)
    base.bin          every chunk at its lowest level (simplified LOD2 geometry), nearest the spawn first
    m_<cell>.bin      mid level (LOD2 geometry) of one 16 m cell, fetched while walking
    n_<cell>.bin      near level (LOD1 geometry) of one 16 m cell, fetched while walking (laptop tier)
    r_<cell>.bin      the same for the phone tier: in clusters, each with its parts in order of size, so the page
                      draws a cell down to what shows at its distance and what its triangle budget holds
    f_<cell>.bin      fine level (LOD0 geometry) of one 16 m cell: the top level of the medium / high tiers
    base.glb, m_ / n_ / r_ / f_<cell>.glb
                      the same geometry, file for file, as glTF 2.0 binaries: what the town page loads (?fmt=packed:
                      the .bin files). A .glb holds the .bin's meshopt streams byte for byte as its BIN chunk
                      (KHR_meshopt_compression, the bake's own vertex layout; base.glb: every cell's lowest level in
                      the manifest's order, so it shows cell by cell as it arrives) under a glTF JSON that names
                      each cell's level, and the bake's stamp. manifest.json lists them under "glb" ({ v, base: {
                      file, bytes, json }, files, bytes }) once every one is written and checked against its .bin;
                      a manifest without it - a bake cut short there, --no-glb - has the page load the .bin files.
                      The .bin files are then deleted (--drop-packed of that tool; "only": true in the record) - they
                      are the same bytes twice - unless --keep-packed; the exporter reads a dropped level from its .glb.
                      Written by `node tools/bake2glb.mjs <this directory> --live --verify` (also by hand, on any
                      bake directory: it skips what is up to date), read by web/engine/lite_glb.js
    tex<size>_<n>.webp  texture-array pages (grids of pre-shaded albedo layers)
    l<size>_<layer>.webp  one layer at 1024 (all) or 2048 (the materials with maps that large): the sharp textures
    n<size>_<layer>.webp  normal (x, y) + roughness of a library material: the per-pixel light of medium / high
                      the medium / high tiers swap in after the first frame
    lean/             the bake of the low / phone tiers, a directory like this one without the fine level and the
                      single layers: the same town lit with the lanterns' pure inverse square (LEAN below - the look
                      those tiers have had from the start; medium / high pool the light as the full engine does).
                      Every run makes both; manifest.json names it ("lean") and the page loads it for those tiers
"""
import sys, os, re, json, math, subprocess, time, argparse, hashlib, shutil, concurrent.futures

ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
VERSION = 1 # of the bake's format (manifest.version; web/engine/lite.js reads exactly this one)
LEVEL_FILES = ( 'mid', 'near', 'ring', 'fine' ) # chunk levels kept in files of their own
# the second bake of a run, for the tiers without the fine level (low / phone): directory under the output, what it
# overrides of the look and of the bake parameters (--set lean.look.KEY=VALUE / lean.KEY=VALUE / lean.pack.KEY=VALUE change it)
LEAN = dict( dir = 'lean', look = dict( reach = 0 ), bake = dict( fine = 0, dir = 0, single_layers = 0, nr_sizes = [] ), pack = {} )
# A second look in the same bake (--set look2.KEY=VALUE, lean.look2.KEY=VALUE for the lean bake alone): the same mesh
# lit a second time with these lanterns, its light a second colour (and direction) in every vertex, so the page
# switches between the two with a uniform. Only what the lighting pass alone reads can differ:
LOOK2_KEYS = ( 'lamp', 'reach', 'range_scale' )
# The look the town ships with ("far-reaching lamps", chosen over the engine's pools of light): every lantern carries
# three times as far as the town declares with its pure inverse square, turned down to suit. It is what a run makes
# when nothing else is asked for; the pools are --set look.lamp=0.7 --set look.reach=4.5 --set range_scale=1
# --set lean.look.lamp=0.7 (or, beside this one in the same mesh, the same values as look2.KEY).
FAR = dict( full = dict( look = dict( lamp = 0.43, reach = 0 ), bake = dict( range_scale = 3 ) ), lean = dict( look = dict( lamp = 0.6 ), bake = dict( range_scale = 3 ) ) )
# the lock two agents keep on this branch: no bake while the towns are rebuilt (out/glb and out/town change under
# it), and a marker for the other side while a bake reads them
BAKING = os.path.join( ROOT, '.scratch/next/BAKING' ); REBUILDING = os.path.join( ROOT, '.scratch/main/REBUILDING' )

GLB_FAILED = [] # the bakes of this run whose .glb files could not be written (their packed files stand)

def glb_export( out, label, keep = False ):

	# the last step of a bake: its geometry as .glb files beside the packed ones (see Output above). The packed bake
	# is complete before this starts and stays as it is whatever happens here: a manifest that names no .glb has the
	# page load the .bin files. A failure is said here and makes the run end with status 3
	node = shutil.which( 'node' )
	cmd = [ node, os.path.join( ROOT, 'tools/bake2glb.mjs' ), out, '--live', '--verify', '--prune', '--no-lean' ] + ( [] if keep else [ '--drop-packed' ] )
	if not node or subprocess.run( cmd, cwd = ROOT ).returncode != 0:

		GLB_FAILED.append( label )
		print( 'webbake: the .glb files of %s (%s) were NOT written: the page loads its packed files. Again: node tools/bake2glb.mjs %s --live --verify' % (
			os.path.relpath( out, ROOT ), label, os.path.relpath( out, ROOT ) ), flush = True )

class bake_lock:

	# with bake_lock(): ... - around everything that reads out/glb or out/town for longer than a moment. It nests (one
	# marker for a whole run of several bakes), and the marker is written again every few seconds while it is held: a
	# clean-up of the scratch folder by somebody else cannot leave a running bake unmarked.
	depth = 0; stop = None

	@staticmethod
	def mark():

		os.makedirs( os.path.dirname( BAKING ), exist_ok = True )
		open( BAKING, 'w' ).write( '%d %s\n' % ( os.getpid(), time.strftime( '%Y-%m-%d %H:%M:%S' ) ) )

	def __enter__( self ):

		if os.path.exists( REBUILDING ): sys.exit( 'webbake: the towns are being rebuilt (%s exists): no bake now' % os.path.relpath( REBUILDING, ROOT ) )
		bake_lock.depth += 1
		if bake_lock.depth == 1:

			import threading
			bake_lock.mark()
			bake_lock.stop = stop = threading.Event()
			def keep():

				while not stop.wait( 5 ):

					if not os.path.exists( BAKING ): bake_lock.mark()

			threading.Thread( target = keep, daemon = True ).start()

		return self

	def __exit__( self, *exc ):

		bake_lock.depth -= 1
		if bake_lock.depth == 0:

			bake_lock.stop.set()
			if os.path.exists( BAKING ): os.remove( BAKING )

		return False

def town_paths( town, out = None ):

	# -> ( name, town.json, output directory ); `town` is an assembly name (out/<name>/town.json) or a path to its json
	if town.endswith( '.json' ) or os.sep in town:

		path = os.path.abspath( town )
		name = os.path.basename( os.path.dirname( path ) ) or 'town'

	else: name = town; path = os.path.join( ROOT, 'out', town, 'town.json' )
	return name, path, os.path.abspath( out ) if out else os.path.join( ROOT, 'out/web', '_next' if name == 'town' else '_next_' + name )

def find_blender():

	for c in ( os.environ.get( 'TSQ_BLENDER' ), os.environ.get( 'BLENDER' ), shutil.which( 'blender' ), os.path.expanduser( '~/.local/bin/blender' ) ):

		if c and os.path.exists( c ): return c

	sys.exit( 'webbake: blender not found (set TSQ_BLENDER=/path/to/blender)' )

def town_sha( path ):

	return hashlib.sha1( open( path, 'rb' ).read() ).hexdigest()

# The look the bake lights the town with is the night rig of the full engine. It is read here from the very lines that
# set it for the high tier (so a retune there reaches the next bake, and --check reports a bake made with the old one):
# ( key of LOOK in webbake_bl.py, file, pattern whose group is the number / the numbers )
NUM = r'([-+.\de ]+)'
NIGHT = r"name === 'night'[\s\S]*?"
LOOK_SRC = (
	( 'sky', 'web/engine/engine.js', NIGHT + r'hemi\.color\.setRGB\(([^)]*)\)' ),
	( 'ground', 'web/engine/engine.js', NIGHT + r'hemi\.groundColor\.setRGB\(([^)]*)\)' ),
	( 'moon_col', 'web/engine/engine.js', NIGHT + r'moon\.color\.setRGB\(([^)]*)\)' ),
	( 'moon_dir', 'web/engine/engine.js', r'const moonDir = new THREE\.Vector3\(([^)]*)\)' ),
	( 'roof_fill', 'web/engine/engine.js', r'const roofFill = uniform\(' + NUM + r'\)' ),
	( 'cap_up', 'web/engine/town.js', r'const CAP_UP = ' + NUM + ';' ),
	( 'ambient', 'web/town/main.js', r'const LOOK = \{[^}]*?\bambient:' + NUM ),
	( 'moon', 'web/town/main.js', r'const LOOK = \{[^}]*?\bmoon:' + NUM ),
	( 'lamp', 'web/town/main.js', r'const LOOK = \{[^}]*?\blamp:' + NUM ),
	( 'source_radius', 'web/town/main.js', r'sourceRadius\.value = [^;]*:' + NUM + ';' ),
	( 'reach', 'web/town/main.js', r"const reach = Q\.has\( 'reach' \) \? [^;]*:" + NUM + ';' ),
	( 'neon', 'web/engine/lite.js', r"neon: v\( 'neon'," + NUM + ',' ), # (signs: as much brighter in the bake's light as on the page)
) + tuple(
	# material uniforms: what the page sets, else the default of the material library
	( key, ( 'web/town/main.js', 'web/engine/materials.js' ), ( r'lib\.u\.%s\.value = ' % u + NUM + ';', r'\b%s: uniform\(' % u + NUM + r'\)' ) )
	for key, u in ( ( 'wet', 'wet' ), ( 'grime', 'grime' ), ( 'wear', 'wear' ), ( 'special', 'special' ), ( 'macro', 'macro' ), ( 'soft', 'soft' ), ( 'window_glow', 'windowGlow' ),
		( 'emissive', 'emissive' ), ( 'ao_direct', 'aoDirect' ), ( 'puddle_rough', 'puddleRough' ), ( 'film_rough', 'filmRough' ), ( 'pool_cap', 'poolCap' ), ( 'tide_h', 'tideH' ) )
)

def engine_look():

	# -> the look constants as the engine and the page have them now; stops when one cannot be found (the line was
	# rewritten: adjust LOOK_SRC), because a bake lit with a guessed value is the drift this is here to prevent
	src = {}; look = {}
	def read( f ):

		if f not in src: src[ f ] = open( os.path.join( ROOT, f ), encoding = 'utf8' ).read()
		return src[ f ]

	for key, files, pats in LOOK_SRC:

		if isinstance( files, str ): files, pats = ( files, ), ( pats, )
		for f, pat in zip( files, pats ):

			m = re.search( pat, read( f ) )
			if m: break

		if not m: sys.exit( 'webbake: look constant %r not found in %s (see LOOK_SRC in tools/webbake.py)' % ( key, ' / '.join( files ) ) )
		try: v = [ float( x.replace( ' ', '' ) ) for x in m.group( 1 ).split( ',' ) ]
		except ValueError: sys.exit( 'webbake: look constant %r in %s is not a number: %r' % ( key, f, m.group( 1 ) ) )
		look[ key ] = v if len( v ) > 1 else v[ 0 ]

	return look

def look_drift( baked, now ):

	# -> the keys of the engine's look that differ from what a bake was lit with
	same = lambda a, b: all( abs( x - y ) < 1e-6 for x, y in zip( a, b ) ) if isinstance( a, list ) and isinstance( b, list ) and len( a ) == len( b ) else ( not isinstance( a, list ) and not isinstance( b, list ) and b is not None and abs( a - b ) < 1e-6 )
	return [ k for k, v in now.items() if not same( v, baked.get( k ) ) ]

def check( town, out, look = None ):

	# -> None when the bake in `out` is current, else one line saying why it is not (look: what this bake overrides of the engine's)
	rel = lambda p: os.path.relpath( p, ROOT )
	stamp = lambda t: time.strftime( '%d %b %H:%M', time.localtime( t ) ) if t else '?'
	again = ': run python3 tools/webbake.py' + ( '' if os.path.basename( os.path.dirname( town ) ) == 'town' else ' --town ' + os.path.basename( os.path.dirname( town ) ) )
	try: man = json.load( open( os.path.join( out, 'manifest.json' ) ) )
	except Exception: return 'no bake in ' + rel( out ) + again
	if not os.path.exists( town ): return 'no ' + rel( town )
	if man.get( 'version' ) != VERSION: return 'the bake in %s has format %s, the page reads %s%s' % ( rel( out ), man.get( 'version' ), VERSION, again )
	if man.get( 'town', {} ).get( 'sha1' ) != town_sha( town ):

		return '%s (%s) is newer than the bake in %s (%s)%s' % ( rel( town ), stamp( os.path.getmtime( town ) ), rel( out ), stamp( man.get( 'baked' ) ), again )

	drift = look_drift( man.get( 'look', {} ), dict( engine_look(), **( look or {} ) ) )
	if drift: return 'the look changed since the bake in %s (%s)%s' % ( rel( out ), ', '.join( drift ), again )
	for f, n in [ ( man[ 'base' ][ 'file' ], man[ 'base' ][ 'bytes' ] ) ] + [ ( c[ l ][ 'file' ], c[ l ][ 'len' ] ) for c in man[ 'chunks' ] for l in LEVEL_FILES if l in c ]:

		p = os.path.join( out, f )
		if not os.path.exists( p ) or os.path.getsize( p ) != n: return '%s of the bake in %s is missing or cut short%s' % ( f, rel( out ), again )

	return None

LANDMARK_CLS = ( 'landmark', )

def landmark_boxes( town ):

	# -> [ [ x0, z0, x1, z1 ], .. ]: the ground box of every item of a landmark class, in the page's frame (x = town x,
	# z = - town y), for the packer (OPT.landmark_boxes of webbake_pack.mjs; it does nothing with them unless one of
	# its landmark_* values is set). The transform is item_xform() of webbake_bl.py
	t = json.load( open( town ) )
	out = []
	for it in t.get( 'items', [] ):

		p = t.get( 'protos', {} ).get( it.get( 'p' ) )
		if not p or p.get( 'cls' ) not in LANDMARK_CLS or 'min' not in p or 'max' not in p: continue
		r = math.radians( it.get( 'rot' ) or 0 ); c, s = math.cos( r ), math.sin( r ); sc = it.get( 's', 1 ) or 1
		xs = []; zs = []
		for bx in ( p[ 'min' ][ 0 ], p[ 'max' ][ 0 ] ):

			for by in ( p[ 'min' ][ 1 ], p[ 'max' ][ 1 ] ):

				lx, lz = bx * sc, - by * sc
				xs.append( c * lx + s * lz + it[ 'pos' ][ 0 ] ); zs.append( - s * lx + c * lz - it[ 'pos' ][ 1 ] )

		out.append( [ round( min( xs ), 3 ), round( min( zs ), 3 ), round( max( xs ), 3 ), round( max( zs ), 3 ) ] )

	return out

def bake_one( a, name, town, out, bake, pack, look, label, look2 = None ):

	# one bake: Blender (unless --pack-only), the packer, the texture pages -> manifest.json of `out`
	t0 = time.time()
	os.makedirs( out, exist_ok = True )
	# work files of every step (and the TMPDIR of Blender, node and ImageMagick) stay on disk beside the output:
	# /tmp and /dev/shm are RAM on the build machine
	work = os.path.join( out, '.work' )
	shutil.rmtree( work, ignore_errors = True )
	os.makedirs( work )
	env = dict( os.environ, TMPDIR = work, TMP = work, TEMP = work, MAGICK_TMPDIR = work )
	cfg = dict( root = ROOT, out = out, work = work, jobs = a.jobs, bake = bake, look = look, town = town, name = name, look2 = look2 or None )
	cfg_path = os.path.join( out, '.config.json' )
	json.dump( cfg, open( cfg_path, 'w' ) )
	# the manifest goes last: a reader never sees a manifest that points at half-written chunks
	man = os.path.join( out, 'manifest.json' )
	if os.path.exists( man ): os.replace( man, man + '.prev' )
	def fail( what ):

		if os.path.exists( man + '.prev' ) and not os.path.exists( man ): os.replace( man + '.prev', man )
		shutil.rmtree( work, ignore_errors = True )
		sys.exit( 'webbake: %s failed (%s)' % ( what, label ) )

	have_raw = os.path.exists( os.path.join( out, '.raw/raw.json' ) )
	if a.pack_only and not have_raw: fail( '--pack-only: no kept bake (run once with --keep-raw);' )
	if a.pack_only and json.load( open( os.path.join( out, '.raw/raw.json' ) ) ).get( 'town', {} ).get( 'sha1' ) != town_sha( town ): fail( '--pack-only: the kept bake is older than its town.json;' )
	if not a.pack_only:

		if os.path.exists( REBUILDING ): fail( 'the towns are being rebuilt (%s exists): the bake' % os.path.relpath( REBUILDING, ROOT ) )
		with bake_lock():

			r = subprocess.run( [ find_blender(), '--background', '--factory-startup', '--python', os.path.join( ROOT, 'tools/webbake_bl.py' ), '--', cfg_path ], env = env )

		if r.returncode != 0 or not os.path.exists( os.path.join( out, '.raw/raw.json' ) ): fail( 'bake' )

	if a.keep_raw or a.pack_only: pack = dict( pack, keep_raw = True )
	if 'landmark_boxes' not in pack: pack = dict( pack, landmark_boxes = landmark_boxes( town ) )
	node = shutil.which( 'node' )
	if not node: fail( 'node not found: packing' )
	if pack.get( 'refit' ): # --set pack.refit=1: the reduced levels' vertex light fitted to the top level's, see tools/lsvb_refit.py
		import lsvb_refit
		pack = lsvb_refit.run( ROOT, out, pack, a.jobs, find_blender(), env ) or fail( 'refit' )
	r = subprocess.run( [ node, os.path.join( ROOT, 'tools/webbake_pack.mjs' ), out, json.dumps( pack ) ], cwd = ROOT, env = env )
	if r.returncode != 0 or not os.path.exists( man ): fail( 'packing' )

	m = json.load( open( man ) )
	magick = shutil.which( 'magick' ) or shutil.which( 'convert' )
	if not magick: fail( 'ImageMagick (magick) is needed to write the WebP texture pages: converting them' )
	keep = set()
	tex_bytes = {}
	def webp_of( name ):

		png = os.path.join( out, name + '.png' ); webp = os.path.join( out, name + '.webp' )
		if os.path.exists( png ) or not os.path.exists( webp ):

			subprocess.run( [ magick, '-quiet', png, '-quality', str( a.quality ), '-define', 'webp:method=6', '-define', 'webp:alpha-quality=90', '-define', 'webp:exact=true', webp ], check = True, env = env )
			os.remove( png )

		return os.path.getsize( webp )

	# the pages (every layer at the small sizes) and the single layers (l<size>_<layer>, the sharp sizes of medium / high)
	sets = [ ( size, t, t[ 'pages' ] ) for size, t in m[ 'tex' ].items() ] + [ ( size + ' layers', t, [ f for f in t[ 'files' ] if f ] ) for size, t in m.get( 'texl', {} ).items() ] + \
		[ ( size + ' normal + roughness', t, [ f for f in t[ 'files' ] if f ] ) for size, t in m.get( 'texn', {} ).items() ]
	with concurrent.futures.ThreadPoolExecutor( max( 1, min( a.jobs, 16 ) ) ) as ex:

		for label_, t, names in sets:

			sizes = list( ex.map( webp_of, names ) )
			keep.update( n + '.webp' for n in names )
			t[ 'bytes' ] = tex_bytes[ label_ ] = sum( sizes )
			if 'files' in t: t[ 'each' ] = dict( zip( names, sizes ) )

	for fn in os.listdir( out ):

		if re.match( r'(tex\d+_|l\d+_|n\d+_)\d+\.(webp|png)$', fn ) and fn not in keep: os.remove( os.path.join( out, fn ) )

	m[ 'seconds' ] = round( time.time() - t0, 1 )
	if label == 'full' and lean_current( town, out ): m[ 'lean' ] = LEAN[ 'dir' ]
	json.dump( m, open( man + '.tmp', 'w' ), separators = ( ',', ':' ) )
	os.replace( man + '.tmp', man )
	if os.path.exists( man + '.prev' ): os.remove( man + '.prev' )
	shutil.rmtree( work, ignore_errors = True )
	st = m[ 'stats' ]
	print( 'webbake: %s -> %s (%s, reach %g): %d chunks, base %.1f MB (%.0fk tris), mid %.1f MB, near %.1f MB, ring %.1f MB, fine %.1f MB (%.2f M tris), textures %s, manifest %.0f kB, %.0f s' % (
		os.path.relpath( town, ROOT ), os.path.relpath( out, ROOT ), label, m.get( 'look', {} ).get( 'reach', 0 ), len( m[ 'chunks' ] ), m[ 'base' ][ 'bytes' ] / 1e6, st.get( 'lo', {} ).get( 'tris', 0 ) / 1e3, st[ 'mid' ][ 'bytes' ] / 1e6, st[ 'near' ][ 'bytes' ] / 1e6, st.get( 'ring', {} ).get( 'bytes', 0 ) / 1e6,
		st.get( 'fine', {} ).get( 'bytes', 0 ) / 1e6, st.get( 'fine', {} ).get( 'tris', 0 ) / 1e6,
		', '.join( '%s: %.1f MB' % ( s, b / 1e6 ) for s, b in tex_bytes.items() ), os.path.getsize( man ) / 1e3, time.time() - t0 ), flush = True )
	if not a.no_glb: glb_export( out, label, a.keep_packed )

def lean_current( town, out ):

	# is there a lean bake of this town.json under `out`?
	try: return json.load( open( os.path.join( out, LEAN[ 'dir' ], 'manifest.json' ) ) ).get( 'town', {} ).get( 'sha1' ) == town_sha( town )
	except Exception: return False

def name_lean( town, out ):

	# the full bake's manifest names the lean one exactly while there is a current one
	man = os.path.join( out, 'manifest.json' )
	try: m = json.load( open( man ) )
	except Exception: return
	want = LEAN[ 'dir' ] if lean_current( town, out ) else None
	if m.get( 'lean' ) == want: return
	if want: m[ 'lean' ] = want
	else: m.pop( 'lean', None )
	json.dump( m, open( man + '.tmp', 'w' ), separators = ( ',', ':' ) )
	os.replace( man + '.tmp', man )

def main():

	ap = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
	ap.add_argument( '--jobs', type = int, default = max( 1, min( 16, ( os.cpu_count() or 2 ) // 2 ) ) )
	ap.add_argument( '--town', default = 'town', help = 'assembly to bake: a name (out/<name>/town.json) or a path to a town.json' )
	ap.add_argument( '--out', default = None, help = 'output directory (default out/web/_next, or out/web/_next_<name> for another assembly)' )
	ap.add_argument( '--check', action = 'store_true' )
	ap.add_argument( '--keep-raw', action = 'store_true' )
	ap.add_argument( '--pack-only', action = 'store_true' )
	ap.add_argument( '--no-glb', action = 'store_true', help = 'do not write the .glb files (tools/bake2glb.mjs --live): the page loads the packed files' )
	ap.add_argument( '--pvs', action = 'store_true', help = 'last of all: the visibility sets of the new bake (tools/pvs.py), without which the page culls by the view frustum alone' )
	ap.add_argument( '--no-gao', action = 'store_true', help = 'do not make the ground occlusion map (tools/aomap.py)' )
	ap.add_argument( '--keep-packed', action = 'store_true', help = 'keep the packed .bin files once the .glb files are written and checked (default: they are deleted)' )
	ap.add_argument( '--only', choices = ( 'both', 'full', 'lean' ), default = 'both', help = 'which of the two bakes to make: full (every level, medium / high), lean (low / phone: <out>/lean), default both' )
	ap.add_argument( '--quality', type = int, default = 86, help = 'WebP quality of the texture pages' )
	ap.add_argument( '--set', action = 'append', default = [], metavar = 'KEY=VALUE', help = 'override a bake parameter (BK in webbake_bl.py), as pack.KEY a packing parameter (OPT in webbake_pack.mjs), as look.KEY a look constant (LOOK in webbake_bl.py), as look2.KEY (lamp, reach, range_scale) the second look lit into the same mesh; as lean.KEY / lean.pack.KEY / lean.look.KEY / lean.look2.KEY for the lean bake alone' )
	a = ap.parse_args()

	name, town, out = town_paths( a.town, a.out )
	lean_out = os.path.join( out, LEAN[ 'dir' ] )
	sets = dict( full = dict( bake = dict( FAR[ 'full' ][ 'bake' ] ), pack = {}, look = dict( FAR[ 'full' ][ 'look' ] ), look2 = {} ),
		lean = dict( bake = dict( FAR[ 'lean' ][ 'bake' ], **LEAN[ 'bake' ] ), pack = dict( LEAN[ 'pack' ] ), look = dict( FAR[ 'lean' ][ 'look' ], **LEAN[ 'look' ] ), look2 = {} ) )
	for kv in sorted( a.set, key = lambda kv: kv.startswith( 'lean.' ) ): # (what is meant for the lean bake alone goes last, whatever the order given)

		k, v = kv.split( '=', 1 )
		which = ( 'lean', ) if k.startswith( 'lean.' ) else ( 'full', 'lean' )
		if k.startswith( 'lean.' ): k = k[ 5: ]
		for w in which:

			# (what the lean bake fixes itself - its reach, no fine level - is not changed by a setting meant for the full one)
			if len( which ) == 2 and w == 'lean' and ( k in LEAN[ 'bake' ] or k.startswith( 'look.' ) and k[ 5: ] in LEAN[ 'look' ] ): continue
			if k.startswith( 'look2.' ):

				if k[ 6: ] not in LOOK2_KEYS: sys.exit( 'webbake: look2.%s: the second look differs in %s alone' % ( k[ 6: ], ', '.join( LOOK2_KEYS ) ) )
				sets[ w ][ 'look2' ][ k[ 6: ] ] = json.loads( v )

			elif k.startswith( 'pack.' ): sets[ w ][ 'pack' ][ k[ 5: ] ] = json.loads( v )
			elif k.startswith( 'look.' ): sets[ w ][ 'look' ][ k[ 5: ] ] = json.loads( v )
			else: sets[ w ][ 'bake' ][ k ] = json.loads( v )

	if a.check:

		why = check( town, out, sets[ 'full' ][ 'look' ] ) or check( town, lean_out, sets[ 'lean' ][ 'look' ] )
		print( 'webbake: ' + ( why or 'up to date' ) )
		sys.exit( 1 if why else 0 )

	if not os.path.exists( town ): sys.exit( 'webbake: no ' + town )
	if not a.pack_only and os.path.exists( REBUILDING ): sys.exit( 'webbake: the towns are being rebuilt (%s exists): no bake now' % os.path.relpath( REBUILDING, ROOT ) )
	look = engine_look()
	class _none:

		def __enter__( self ): return self
		def __exit__( self, *exc ): return False

	# (the marker for the whole run, the packing between the two bakes included)
	with ( _none() if a.pack_only else bake_lock() ):

		# the lean bake first: the full bake's manifest, written last of all, names it
		if a.only in ( 'both', 'lean' ) and not ( a.pack_only and a.only == 'both' and not os.path.exists( os.path.join( lean_out, '.raw/raw.json' ) ) ):

			bake_one( a, name, town, lean_out, sets[ 'lean' ][ 'bake' ], sets[ 'lean' ][ 'pack' ], dict( look, **sets[ 'lean' ][ 'look' ] ), 'lean', sets[ 'lean' ][ 'look2' ] )

		if a.only in ( 'both', 'full' ): bake_one( a, name, town, out, sets[ 'full' ][ 'bake' ], sets[ 'full' ][ 'pack' ], dict( look, **sets[ 'full' ][ 'look' ] ), 'full', sets[ 'full' ][ 'look2' ] )
		name_lean( town, out )

	if GLB_FAILED: sys.exit( 3 )
	rel = os.path.relpath( os.path.join( ROOT, out ), os.path.join( ROOT, 'out/web' ) )
	# the ground's contact occlusion (tools/aomap.py: a picture of the town from above, read by the ground's pixels):
	# from the working data, so with --keep-raw / --pack-only; a quarter of a minute
	if not a.no_gao and not ( os.sep in rel or rel.startswith( '.' ) ):

		if not os.path.exists( os.path.join( ROOT, out, '.raw/raw.json' ) ): print( 'webbake: no ground occlusion map (it is made from the working data: bake with --keep-raw, or python3 tools/aomap.py --dir %s after one)' % rel, flush = True )
		elif subprocess.run( [ sys.executable, os.path.join( ROOT, 'tools/aomap.py' ), '--dir', rel, '--jobs', str( a.jobs ) ], cwd = ROOT ).returncode != 0: sys.exit( 5 )

	# a bake starts with a new manifest: the visibility sets of the last one (occlusion culling) are gone with it
	pvs = [ sys.executable, os.path.join( ROOT, 'tools/pvs.py' ), '--dir', rel, '--only', a.only ]
	if not a.pvs or os.sep in rel or rel.startswith( '.' ):

		print( 'webbake: no visibility sets yet (the page culls by the view frustum alone). To bake them: python3 tools/pvs.py --dir %s' % rel, flush = True )

	elif subprocess.run( pvs, cwd = ROOT ).returncode != 0: sys.exit( 4 )

if __name__ == '__main__': main()
