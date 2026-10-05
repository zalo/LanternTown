#!/usr/bin/env python3
"""The first bake, kept until the towns are baked again by tools/webbake.py: what web/engine/lite_v1.js reads
(web/town/ with ?legacy=1 and ?town=town2). The live tiers are baked by tools/webbake.py.

Bake the town into the web-friendly build the light tiers of the walkable demo load.

    python3 tools/webbake_v1.py                 # out/town/town.json + out/glb/** + out/mat/** -> out/web/_bake/
    python3 tools/webbake_v1.py --town town2    # another assembly: out/town2/town.json -> out/web/_bake_town2/
                                             # (--town also takes a path to a town.json; --out names the directory)
    python3 tools/webbake_v1.py --jobs 16       # lighting workers (default: cores / 2, at most 16)
    python3 tools/webbake_v1.py --check         # exit 1 when the bake is missing or older than town.json
    python3 tools/webbake_v1.py --keep-raw      # keep the unpacked chunks (out/web/_bake/.raw, ~250 MB) ...
    python3 tools/webbake_v1.py --pack-only     # ... to try packing parameters (--set pack.KEY=VALUE) without baking again
    python3 tools/webbake_v1.py --no-glb        # the packed files only (the page then loads those)

One command, no hand steps: it follows town.json (nothing here names an asset), so it is simply re-run after
every `tools/town.py all`. The heavy part (tools/webbake_v1_bl.py) runs inside Blender's python for numpy and the
BVH, tools/webbake_v1_pack.mjs (node + meshoptimizer) simplifies and compresses the chunks; this file runs both and
converts the texture pages to WebP.

Output (out/web/_bake/):
    manifest.json     chunk table, texture layers, lantern glows / flames, navigation data, town fingerprint
    base.bin          every chunk at its lowest level (simplified LOD2 geometry), nearest the spawn first
    m_<cell>.bin      mid level (LOD2 geometry) of one 16 m cell, fetched while walking
    n_<cell>.bin      near level (LOD1 geometry) of one 16 m cell, fetched while walking (laptop tier)
    r_<cell>.bin      the same for the phone tier: in clusters, each with its parts in order of size, so the page
                      draws a cell down to what shows at its distance and what its triangle budget holds
    base.glb, m_ / n_ / r_<cell>.glb
                      the same geometry, file for file, as glTF 2.0 binaries (the .bin's meshopt streams under a
                      glTF JSON): what the town page loads, listed in manifest.json under "glb"; ?fmt=packed and a
                      manifest without "glb" load the .bin files. The last step here, and by hand on any bake
                      directory: node tools/bake2glb.mjs <directory> --live --verify (see tools/webbake.py)
    tex<size>_<n>.webp  texture-array pages (grids of pre-shaded albedo layers)
"""
import sys, os, re, json, subprocess, time, argparse, hashlib, shutil

ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
VERSION = 1 # of the bake's format (manifest.version; web/engine/lite_v1.js reads exactly this one)
LEVEL_FILES = ( 'mid', 'near', 'ring' ) # chunk levels kept in files of their own

def town_paths( town, out = None ):

	# -> ( name, town.json, output directory ); `town` is an assembly name (out/<name>/town.json) or a path to its json
	if town.endswith( '.json' ) or os.sep in town:

		path = os.path.abspath( town )
		name = os.path.basename( os.path.dirname( path ) ) or 'town'

	else: name = town; path = os.path.join( ROOT, 'out', town, 'town.json' )
	return name, path, os.path.abspath( out ) if out else os.path.join( ROOT, 'out/web', '_bake' if name == 'town' else '_bake_' + name )

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

		if not m: sys.exit( 'webbake: look constant %r not found in %s (see LOOK_SRC in tools/webbake_v1.py)' % ( key, ' / '.join( files ) ) )
		try: v = [ float( x.replace( ' ', '' ) ) for x in m.group( 1 ).split( ',' ) ]
		except ValueError: sys.exit( 'webbake: look constant %r in %s is not a number: %r' % ( key, f, m.group( 1 ) ) )
		look[ key ] = v if len( v ) > 1 else v[ 0 ]

	return look

def look_drift( baked, now ):

	# -> the keys of the engine's look that differ from what a bake was lit with
	same = lambda a, b: all( abs( x - y ) < 1e-6 for x, y in zip( a, b ) ) if isinstance( a, list ) and isinstance( b, list ) and len( a ) == len( b ) else ( not isinstance( a, list ) and not isinstance( b, list ) and b is not None and abs( a - b ) < 1e-6 )
	return [ k for k, v in now.items() if not same( v, baked.get( k ) ) ]

def check( town, out ):

	# -> None when the bake in `out` is current, else one line saying why it is not
	rel = lambda p: os.path.relpath( p, ROOT )
	stamp = lambda t: time.strftime( '%d %b %H:%M', time.localtime( t ) ) if t else '?'
	again = ': run python3 tools/webbake_v1.py' + ( '' if os.path.basename( os.path.dirname( town ) ) == 'town' else ' --town ' + os.path.basename( os.path.dirname( town ) ) )
	try: man = json.load( open( os.path.join( out, 'manifest.json' ) ) )
	except Exception: return 'no bake in ' + rel( out ) + again
	if not os.path.exists( town ): return 'no ' + rel( town )
	if man.get( 'version' ) != VERSION: return 'the bake in %s has format %s, the page reads %s%s' % ( rel( out ), man.get( 'version' ), VERSION, again )
	if man.get( 'town', {} ).get( 'sha1' ) != town_sha( town ):

		return '%s (%s) is newer than the bake in %s (%s)%s' % ( rel( town ), stamp( os.path.getmtime( town ) ), rel( out ), stamp( man.get( 'baked' ) ), again )

	drift = look_drift( man.get( 'look', {} ), engine_look() )
	if drift: return 'the look changed since the bake in %s (%s)%s' % ( rel( out ), ', '.join( drift ), again )
	for f, n in [ ( man[ 'base' ][ 'file' ], man[ 'base' ][ 'bytes' ] ) ] + [ ( c[ l ][ 'file' ], c[ l ][ 'len' ] ) for c in man[ 'chunks' ] for l in LEVEL_FILES if l in c ]:

		p = os.path.join( out, f )
		if not os.path.exists( p ) or os.path.getsize( p ) != n: return '%s of the bake in %s is missing or cut short%s' % ( f, rel( out ), again )

	return None

def main():

	ap = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
	ap.add_argument( '--jobs', type = int, default = max( 1, min( 16, ( os.cpu_count() or 2 ) // 2 ) ) )
	ap.add_argument( '--town', default = 'town', help = 'assembly to bake: a name (out/<name>/town.json) or a path to a town.json' )
	ap.add_argument( '--out', default = None, help = 'output directory (default out/web/_bake, or out/web/_bake_<name> for another assembly)' )
	ap.add_argument( '--check', action = 'store_true' )
	ap.add_argument( '--keep-raw', action = 'store_true' )
	ap.add_argument( '--pack-only', action = 'store_true' )
	ap.add_argument( '--no-glb', action = 'store_true', help = 'do not write the .glb files (tools/bake2glb.mjs --live): the page loads the packed files' )
	ap.add_argument( '--quality', type = int, default = 86, help = 'WebP quality of the texture pages' )
	ap.add_argument( '--set', action = 'append', default = [], metavar = 'KEY=VALUE', help = 'override a bake parameter (BK in webbake_bl.py), as pack.KEY a packing parameter (OPT in webbake_pack.mjs), as look.KEY a look constant (LOOK in webbake_bl.py)' )
	a = ap.parse_args()

	name, town, out = town_paths( a.town, a.out )
	if a.check:

		why = check( town, out )
		print( 'webbake: ' + ( why or 'up to date' ) )
		sys.exit( 1 if why else 0 )

	t0 = time.time()
	if not os.path.exists( town ): sys.exit( 'webbake: no ' + town )
	os.makedirs( out, exist_ok = True )
	bake = {}; pack = {}; look = engine_look()
	for kv in a.set:

		k, v = kv.split( '=', 1 )
		if k.startswith( 'pack.' ): pack[ k[ 5: ] ] = json.loads( v )
		elif k.startswith( 'look.' ): look[ k[ 5: ] ] = json.loads( v )
		else: bake[ k ] = json.loads( v )

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
		sys.exit( 'webbake: %s failed' % what )

	have_raw = os.path.exists( os.path.join( out, '.raw/raw.json' ) )
	if a.pack_only and not have_raw: fail( '--pack-only: no kept bake (run once with --keep-raw);' )
	if a.pack_only and json.load( open( os.path.join( out, '.raw/raw.json' ) ) ).get( 'town', {} ).get( 'sha1' ) != town_sha( town ): fail( '--pack-only: the kept bake is older than its town.json;' )
	if not a.pack_only:

		r = subprocess.run( [ find_blender(), '--background', '--factory-startup', '--python', os.path.join( ROOT, 'tools/webbake_v1_bl.py' ), '--', cfg_path ], env = env )
		if r.returncode != 0 or not os.path.exists( os.path.join( out, '.raw/raw.json' ) ): fail( 'bake' )

	if a.keep_raw or a.pack_only: pack[ 'keep_raw' ] = True
	node = shutil.which( 'node' )
	if not node: fail( 'node not found: packing' )
	r = subprocess.run( [ node, os.path.join( ROOT, 'tools/webbake_v1_pack.mjs' ), out, json.dumps( pack ) ], cwd = ROOT, env = env )
	if r.returncode != 0 or not os.path.exists( man ): fail( 'packing' )

	m = json.load( open( man ) )
	magick = shutil.which( 'magick' ) or shutil.which( 'convert' )
	if not magick: fail( 'ImageMagick (magick) is needed to write the WebP texture pages: converting them' )
	keep = set()
	tex_bytes = {}
	for size, t in m[ 'tex' ].items():

		n = 0
		for name in t[ 'pages' ]:

			png = os.path.join( out, name + '.png' ); webp = os.path.join( out, name + '.webp' )
			if os.path.exists( png ) or not os.path.exists( webp ):

				subprocess.run( [ magick, '-quiet', png, '-quality', str( a.quality ), '-define', 'webp:method=6', '-define', 'webp:alpha-quality=90', '-define', 'webp:exact=true', webp ], check = True, env = env )
				os.remove( png )

			keep.add( name + '.webp' )
			n += os.path.getsize( webp )

		tex_bytes[ size ] = n
		t[ 'bytes' ] = n

	for fn in os.listdir( out ):

		if fn.startswith( 'tex' ) and fn.endswith( ( '.webp', '.png' ) ) and fn not in keep: os.remove( os.path.join( out, fn ) )

	m[ 'seconds' ] = round( time.time() - t0, 1 )
	json.dump( m, open( man + '.tmp', 'w' ), separators = ( ',', ':' ) )
	os.replace( man + '.tmp', man )
	if os.path.exists( man + '.prev' ): os.remove( man + '.prev' )
	shutil.rmtree( work, ignore_errors = True )
	st = m[ 'stats' ]
	print( 'webbake: %s -> %s: %d chunks, base %.1f MB, mid %.1f MB, near %.1f MB, ring %.1f MB, textures %s, manifest %.0f kB, %.0f s' % (
		os.path.relpath( town, ROOT ), os.path.relpath( out, ROOT ), len( m[ 'chunks' ] ), m[ 'base' ][ 'bytes' ] / 1e6, st[ 'mid' ][ 'bytes' ] / 1e6, st[ 'near' ][ 'bytes' ] / 1e6, st.get( 'ring', {} ).get( 'bytes', 0 ) / 1e6,
		', '.join( '%s: %.1f MB' % ( s, b / 1e6 ) for s, b in tex_bytes.items() ), os.path.getsize( man ) / 1e3, time.time() - t0 ) )
	# last: the geometry once more as .glb files beside the packed ones, each checked against its .bin. The packed bake
	# stands whatever happens here (a manifest that names no .glb has the page load the .bin files)
	if not a.no_glb and subprocess.run( [ node, os.path.join( ROOT, 'tools/bake2glb.mjs' ), out, '--live', '--verify', '--prune' ], cwd = ROOT ).returncode != 0:

		sys.exit( 'webbake: the .glb files of %s were NOT written: the page loads its packed files. Again: node tools/bake2glb.mjs %s --live --verify' % ( os.path.relpath( out, ROOT ), os.path.relpath( out, ROOT ) ) )

if __name__ == '__main__': main()
