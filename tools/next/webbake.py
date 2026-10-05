#!/usr/bin/env python3
"""Bake the town into the web-friendly build the light tiers of the walkable demo load.

    python3 tools/next/webbake.py                 # out/town/town.json + out/glb/** + out/mat/** -> out/web/_next/
    python3 tools/next/webbake.py --town town2    # another assembly: out/town2/town.json -> out/web/_next_town2/
                                             # (--town also takes a path to a town.json; --out names the directory)
    python3 tools/next/webbake.py --jobs 16       # lighting workers (default: cores / 2, at most 16)
    python3 tools/next/webbake.py --check         # exit 1 when the bake is missing or older than town.json
    python3 tools/next/webbake.py --keep-raw      # keep the unpacked chunks (out/web/_next/.raw, ~250 MB) ...
    python3 tools/next/webbake.py --pack-only     # ... to try packing parameters (--set pack.KEY=VALUE) without baking again
    python3 tools/next/webbake.py --only full     # one of the two bakes of a run: full (medium / high) or lean (low / phone)

One command, no hand steps: it follows town.json (nothing here names an asset), so it is simply re-run after
every `tools/town.py all`. The heavy part (tools/next/webbake_bl.py) runs inside Blender's python for numpy and the
BVH, tools/next/webbake_pack.mjs (node + meshoptimizer) simplifies and compresses the chunks; this file runs both and
converts the texture pages to WebP.

Output (out/web/_next/):
    manifest.json     chunk table, texture layers, lantern glows / flames, navigation data, town fingerprint
    base.bin          every chunk at its lowest level (simplified LOD2 geometry), nearest the spawn first
    m_<cell>.bin      mid level (LOD2 geometry) of one 16 m cell, fetched while walking
    n_<cell>.bin      near level (LOD1 geometry) of one 16 m cell, fetched while walking (laptop tier)
    r_<cell>.bin      the same for the phone tier: in clusters, each with its parts in order of size, so the page
                      draws a cell down to what shows at its distance and what its triangle budget holds
    f_<cell>.bin      fine level (LOD0 geometry) of one 16 m cell: the top level of the medium / high tiers
    tex<size>_<n>.webp  texture-array pages (grids of pre-shaded albedo layers)
    l<size>_<layer>.webp  one layer at 1024 (all) or 2048 (the materials with maps that large): the sharp textures
    n<size>_<layer>.webp  normal (x, y) + roughness of a library material: the per-pixel light of medium / high
                      the medium / high tiers swap in after the first frame
    lean/             the bake of the low / phone tiers, a directory like this one without the fine level and the
                      single layers: the same town lit with the lanterns' pure inverse square (LEAN below - the look
                      those tiers have had from the start; medium / high pool the light as the full engine does).
                      Every run makes both; manifest.json names it ("lean") and the page loads it for those tiers
"""
import sys, os, re, json, subprocess, time, argparse, hashlib, shutil, concurrent.futures

ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) ) # tools/next/ -> the project
VERSION = 1 # of the bake's format (manifest.version; web/engine/lite.js reads exactly this one)
LEVEL_FILES = ( 'mid', 'near', 'ring', 'fine' ) # chunk levels kept in files of their own
# the second bake of a run, for the tiers without the fine level (low / phone): directory under the output, what it
# overrides of the look and of the bake parameters (--set lean.look.KEY=VALUE / lean.KEY=VALUE / lean.pack.KEY=VALUE change it)
LEAN = dict( dir = 'lean', look = dict( reach = 0 ), bake = dict( fine = 0, dir = 0, single_layers = 0, nr_sizes = [] ), pack = {} )
# the lock two agents keep on this branch: no bake while the towns are rebuilt (out/glb and out/town change under
# it), and a marker for the other side while a bake reads them
BAKING = os.path.join( ROOT, '.scratch/next/BAKING' ); REBUILDING = os.path.join( ROOT, '.scratch/main/REBUILDING' )

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

	for c in ( os.environ.get( 'BLENDER' ), os.path.expanduser( '~/.local/bin/blender' ), shutil.which( 'blender' ) ):

		if c and os.path.exists( c ): return c

	sys.exit( 'webbake: blender not found (set BLENDER=/path/to/blender)' )

def town_sha( path ):

	return hashlib.sha1( open( path, 'rb' ).read() ).hexdigest()

# The look the bake lights the town with is the night rig of the full engine. It is read here from the very lines that
# set it for the high tier (so a retune there reaches the next bake, and --check reports a bake made with the old one):
# ( key of LOOK in webbake_bl.py, file, pattern whose group is the number / the numbers )
NUM = r'([-+.\de ]+)'
NIGHT = r"name === 'night'[\s\S]*?"
LOOK_SRC = (
	( 'sky', 'web/next/engine/engine.js', NIGHT + r'hemi\.color\.setRGB\(([^)]*)\)' ),
	( 'ground', 'web/next/engine/engine.js', NIGHT + r'hemi\.groundColor\.setRGB\(([^)]*)\)' ),
	( 'moon_col', 'web/next/engine/engine.js', NIGHT + r'moon\.color\.setRGB\(([^)]*)\)' ),
	( 'moon_dir', 'web/next/engine/engine.js', r'const moonDir = new THREE\.Vector3\(([^)]*)\)' ),
	( 'roof_fill', 'web/next/engine/engine.js', r'const roofFill = uniform\(' + NUM + r'\)' ),
	( 'cap_up', 'web/next/engine/town.js', r'const CAP_UP = ' + NUM + ';' ),
	( 'ambient', 'web/next/town/main.js', r'const LOOK = \{[^}]*?\bambient:' + NUM ),
	( 'moon', 'web/next/town/main.js', r'const LOOK = \{[^}]*?\bmoon:' + NUM ),
	( 'lamp', 'web/next/town/main.js', r'const LOOK = \{[^}]*?\blamp:' + NUM ),
	( 'source_radius', 'web/next/town/main.js', r'sourceRadius\.value = [^;]*:' + NUM + ';' ),
	( 'reach', 'web/next/town/main.js', r"const reach = Q\.has\( 'reach' \) \? [^;]*:" + NUM + ';' ),
) + tuple(
	# material uniforms: what the page sets, else the default of the material library
	( key, ( 'web/next/town/main.js', 'web/next/engine/materials.js' ), ( r'lib\.u\.%s\.value = ' % u + NUM + ';', r'\b%s: uniform\(' % u + NUM + r'\)' ) )
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

		if not m: sys.exit( 'webbake: look constant %r not found in %s (see LOOK_SRC in tools/next/webbake.py)' % ( key, ' / '.join( files ) ) )
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
	again = ': run python3 tools/next/webbake.py' + ( '' if os.path.basename( os.path.dirname( town ) ) == 'town' else ' --town ' + os.path.basename( os.path.dirname( town ) ) )
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

def bake_one( a, name, town, out, bake, pack, look, label ):

	# one bake: Blender (unless --pack-only), the packer, the texture pages -> manifest.json of `out`
	t0 = time.time()
	os.makedirs( out, exist_ok = True )
	# work files of every step (and the TMPDIR of Blender, node and ImageMagick) stay on disk beside the output:
	# /tmp and /dev/shm are RAM on the build machine
	work = os.path.join( out, '.work' )
	shutil.rmtree( work, ignore_errors = True )
	os.makedirs( work )
	env = dict( os.environ, TMPDIR = work, TMP = work, TEMP = work, MAGICK_TMPDIR = work )
	cfg = dict( root = ROOT, out = out, work = work, jobs = a.jobs, bake = bake, look = look, town = town, name = name )
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

			r = subprocess.run( [ find_blender(), '--background', '--factory-startup', '--python', os.path.join( ROOT, 'tools/next/webbake_bl.py' ), '--', cfg_path ], env = env )

		if r.returncode != 0 or not os.path.exists( os.path.join( out, '.raw/raw.json' ) ): fail( 'bake' )

	if a.keep_raw or a.pack_only: pack = dict( pack, keep_raw = True )
	node = shutil.which( 'node' )
	if not node: fail( 'node not found: packing' )
	r = subprocess.run( [ node, os.path.join( ROOT, 'tools/next/webbake_pack.mjs' ), out, json.dumps( pack ) ], cwd = ROOT, env = env )
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
	ap.add_argument( '--only', choices = ( 'both', 'full', 'lean' ), default = 'both', help = 'which of the two bakes to make: full (every level, medium / high), lean (low / phone: <out>/lean), default both' )
	ap.add_argument( '--quality', type = int, default = 86, help = 'WebP quality of the texture pages' )
	ap.add_argument( '--set', action = 'append', default = [], metavar = 'KEY=VALUE', help = 'override a bake parameter (BK in webbake_bl.py), as pack.KEY a packing parameter (OPT in webbake_pack.mjs), as look.KEY a look constant (LOOK in webbake_bl.py); as lean.KEY / lean.pack.KEY / lean.look.KEY for the lean bake alone' )
	a = ap.parse_args()

	name, town, out = town_paths( a.town, a.out )
	lean_out = os.path.join( out, LEAN[ 'dir' ] )
	sets = dict( full = dict( bake = {}, pack = {}, look = {} ), lean = dict( bake = dict( LEAN[ 'bake' ] ), pack = dict( LEAN[ 'pack' ] ), look = dict( LEAN[ 'look' ] ) ) )
	for kv in a.set:

		k, v = kv.split( '=', 1 )
		which = ( 'lean', ) if k.startswith( 'lean.' ) else ( 'full', 'lean' )
		if k.startswith( 'lean.' ): k = k[ 5: ]
		for w in which:

			# (what the lean bake fixes itself - its reach, no fine level - is not changed by a setting meant for the full one)
			if len( which ) == 2 and w == 'lean' and ( k in LEAN[ 'bake' ] or k.startswith( 'look.' ) and k[ 5: ] in LEAN[ 'look' ] ): continue
			if k.startswith( 'pack.' ): sets[ w ][ 'pack' ][ k[ 5: ] ] = json.loads( v )
			elif k.startswith( 'look.' ): sets[ w ][ 'look' ][ k[ 5: ] ] = json.loads( v )
			else: sets[ w ][ 'bake' ][ k ] = json.loads( v )

	if a.check:

		why = check( town, out ) or check( town, lean_out, sets[ 'lean' ][ 'look' ] )
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

			bake_one( a, name, town, lean_out, sets[ 'lean' ][ 'bake' ], sets[ 'lean' ][ 'pack' ], dict( look, **sets[ 'lean' ][ 'look' ] ), 'lean' )

		if a.only in ( 'both', 'full' ): bake_one( a, name, town, out, sets[ 'full' ][ 'bake' ], sets[ 'full' ][ 'pack' ], dict( look, **sets[ 'full' ][ 'look' ] ), 'full' )
		name_lean( town, out )

if __name__ == '__main__': main()
