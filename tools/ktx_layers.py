#!/usr/bin/env python3
"""KTX2 / Basis texture layers of a baked town (experiment: GPU-compressed textures for web/engine/lite.js).

    python3 tools/ktx_layers.py                                  # out/web/_next -> out/web/_nextktx (+ lean/)
    python3 tools/ktx_layers.py --out out/web/_nextktx_e1s --colour etc1s,q=255,c=4 --nr etc1s,q=255,c=4
    python3 tools/ktx_layers.py --out out/web/_nextktx_kai --mips kaiser --nrmips toksvig

Reads the bake's manifest and, where there is one, the lossless source of a layer (the bake's .texcache/<name>_<sig>.npz:
s<size> = sRGB colour + relief height, n<size> = normal x, y + roughness); the layers without one (white, window,
stripes) are decoded from the bake's WebP files. Builds the mip chain itself and has `ktx create` (KTX-Software,
Apache-2.0) encode it:

    c<size>.ktx2          every layer at a page size, ONE array file (the first frame's textures: tex256 / tex512 pages)
    l<size>_<layer>.ktx2  one colour layer (rgb pre-shaded albedo sRGB, a = 0.5 + 0.5 x relief height), all mips
    n<size>_<layer>.ktx2  one normal x, y + roughness layer (linear), all mips
    ktx.json              what is there (web/ktx/engine/lite_ktx.js reads it), `baked` = the bake it was made from

Needs numpy (and ImageMagick for the layers without a cache). It never writes into the bake.
"""
import sys, os, re, json, time, argparse, subprocess, shutil, tempfile, concurrent.futures
import numpy as np

ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

def find_ktx( given ):

	cands = [ given, os.environ.get( 'KTX' ), os.path.join( ROOT, '.scratch/ktx/KTX-Software-4.4.2-Linux-x86_64/bin/ktx' ), shutil.which( 'ktx' ) ]
	for c in cands:

		if c and os.path.exists( c ): return os.path.abspath( c )

	sys.exit( 'ktx_layers: no `ktx` (KTX-Software >= 4.3): pass --ktx or set $KTX' )

def from_srgb( c ): return np.where( c <= 0.04045, c / 12.92, ( ( c + 0.055 ) / 1.055 ) ** 2.4 )
def to_srgb( c ): return np.where( c <= 0.0031308, c * 12.92, 1.055 * np.maximum( c, 1e-12 ) ** ( 1 / 2.4 ) - 0.055 )

# 2 : 1 reduction kernels (taps on the finer level, centred between its two middle samples), tiles wrap
def kernel( kind ):

	if kind == 'box': w = np.array( [ 1, 1 ], np.float64 )
	elif kind == 'tent': w = np.array( [ 1, 3, 3, 1 ], np.float64 )
	elif kind == 'kaiser':

		# Kaiser-windowed sinc, cut at the coarser level's Nyquist: 8 taps (radius 4 fine texels), beta 4
		x = np.arange( 8 ) - 3.5
		w = np.sinc( x / 2 ) * np.i0( 4.0 * np.sqrt( np.maximum( 0, 1 - ( x / 4 ) ** 2 ) ) )

	else: sys.exit( 'ktx_layers: mip filter %r?' % kind )
	return w / w.sum()

def down_axis( a, w, axis ):

	n = a.shape[ axis ]
	if n == 1: return a
	half = len( w ) // 2 - 1
	out = 0
	for k, wk in enumerate( w ):

		r = np.roll( a, half - k, axis ) if k != half else a
		out = out + wk * np.take( r, np.arange( 0, n, 2 ), axis )

	return out

def down( a, kind ):

	w = kernel( kind )
	return down_axis( down_axis( a, w, 0 ), w, 1 )

def colour_levels( rgba8, kind, hkind, hfade ):

	# rgba8: (S, S, 4) bytes, sRGB colour + alpha as the page has it. -> levels as bytes, down to 1 x 1.
	# colour filtered in linear light (what the GPU's generateMipmaps of an sRGB texture does with `box`)
	a = rgba8.astype( np.float64 ) / 255
	lin = np.concatenate( [ from_srgb( a[ ..., :3 ] ), a[ ..., 3: ] ], 2 )
	hm = lin[ ..., 3 ].mean()
	out = [ rgba8 ]
	cur_c, cur_h = lin[ ..., :3 ], lin[ ..., 3: ]
	lv = 0
	while cur_c.shape[ 0 ] > 1:

		lv += 1
		cur_c = np.clip( down( cur_c, kind ), 0, 1 ); cur_h = np.clip( down( cur_h, hkind ), 0, 1 )
		h = cur_h
		if hfade:

			# the relief height towards the layer's mean from level `hfade[ 0 ]` on, flat by level `hfade[ 1 ]`
			k = min( 1, max( 0, ( lv - hfade[ 0 ] + 1 ) / max( 1e-6, hfade[ 1 ] - hfade[ 0 ] + 1 ) ) )
			h = hm + ( cur_h - hm ) * ( 1 - k )

		q = np.concatenate( [ to_srgb( cur_c ), h ], 2 )
		out.append( np.round( np.clip( q, 0, 1 ) * 255 ).astype( np.uint8 ) )

	return out

def nr_levels( nr8, kind ):

	# nr8: (S, S, >= 3) bytes: normal x, y and roughness. -> levels of (S, S, 3) bytes.
	# box: the three channels averaged, as the GPU's own mips are. toksvig: the normals averaged as vectors and
	# renormalised, the variance the average lost (its shortening) added to the roughness: what was relief a level
	# up is roughness a level down, so a distant highlight is wide and dim instead of sparkling
	a = nr8[ ..., :3 ].astype( np.float64 ) / 255
	out = [ np.ascontiguousarray( nr8[ ..., :3 ] ) ]
	if kind != 'toksvig':

		cur = a
		while cur.shape[ 0 ] > 1:

			cur = np.clip( down( cur, 'box' if kind == 'box' else kind ), 0, 1 )
			out.append( np.round( cur * 255 ).astype( np.uint8 ) )

		return out

	xy = a[ ..., :2 ] * 2 - 1
	z = np.sqrt( np.maximum( 1 - ( xy ** 2 ).sum( 2, keepdims = True ), 0.05 ) ) # (as the shader rebuilds it)
	n = np.concatenate( [ xy, z ], 2 ); n /= np.linalg.norm( n, axis = 2, keepdims = True )
	a2 = a[ ..., 2: ] ** 4 # GGX alpha^2 of roughness r: ( r^2 )^2
	while n.shape[ 0 ] > 1:

		n = down( n, 'box' ); a2 = down( a2, 'box' )
		ln = np.maximum( np.linalg.norm( n, axis = 2, keepdims = True ), 1e-4 )
		var = np.minimum( ( 1 - ln ) / ln, 0.5 )
		r = np.clip( a2 + 2 * var, 0, 1 ) ** 0.25
		q = np.concatenate( [ n[ ..., :2 ] / ln * 0.5 + 0.5, r ], 2 )
		out.append( np.round( np.clip( q, 0, 1 ) * 255 ).astype( np.uint8 ) )

	return out

def webp_rgba( path, crop = None ):

	magick = shutil.which( 'magick' ) or shutil.which( 'convert' )
	if not magick: sys.exit( 'ktx_layers: ImageMagick is needed for the layers without a cache' )
	cmd = [ magick, '-quiet', path ] + ( [ '-crop', '%dx%d+%d+%d' % crop, '+repage' ] if crop else [] ) + [ '-depth', '8', 'RGBA:-' ]
	b = subprocess.run( cmd, check = True, capture_output = True ).stdout
	s = int( round( ( len( b ) / 4 ) ** 0.5 ) )
	return np.frombuffer( b, np.uint8 ).reshape( s, s, 4 )

def codec_args( spec, srgb, ch ):

	p = spec.split( ',' ); kv = dict( x.split( '=' ) for x in p[ 1: ] )
	fmt = ( 'R8G8B8A8' if ch == 4 else 'R8G8B8' ) + ( '_SRGB' if srgb else '_UNORM' )
	a = [ '--format', fmt ]
	if p[ 0 ] == 'uastc':

		a += [ '--encode', 'uastc', '--uastc-quality', kv.get( 'q', '2' ) ]
		if float( kv.get( 'rdo', 0 ) ) > 0: a += [ '--uastc-rdo', '--uastc-rdo-l', kv[ 'rdo' ], '--uastc-rdo-m' ]
		if int( kv.get( 'zstd', 18 ) ) > 0: a += [ '--zstd', kv.get( 'zstd', '18' ) ]

	elif p[ 0 ] == 'etc1s':

		a += [ '--encode', 'basis-lz', '--qlevel', kv.get( 'q', '255' ), '--clevel', kv.get( 'c', '4' ) ]
		if kv.get( 'rdo', '1' ) == '0': a += [ '--no-endpoint-rdo', '--no-selector-rdo' ]

	else: sys.exit( 'ktx_layers: codec %r?' % spec )
	return a

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--bake', default = 'out/web/_next' )
	ap.add_argument( '--out', default = 'out/web/_nextktx' )
	ap.add_argument( '--ktx', default = '' )
	ap.add_argument( '--sets', default = '', help = 'c256,c512,l1024,l2048,n512,n1024 (default: all the bake has)' )
	ap.add_argument( '--colour', default = 'uastc,q=2,rdo=1.0,zstd=18' )
	ap.add_argument( '--nr', default = 'uastc,q=2,rdo=0.5,zstd=18' )
	ap.add_argument( '--mips', default = 'box', help = 'colour: box | tent | kaiser (in linear light)' )
	ap.add_argument( '--hmips', default = '', help = 'relief height: box | tent | kaiser (default: as --mips)' )
	ap.add_argument( '--hfade', default = '', help = 'a,b: the height goes to its mean from mip level a, flat at b' )
	ap.add_argument( '--nrmips', default = 'box', help = 'box | toksvig' )
	ap.add_argument( '--jobs', type = int, default = 8 )
	ap.add_argument( '--lean', type = int, default = 1, help = 'also <out>/lean: the page sets for the bake of low / phone' )
	ap.add_argument( '--scratch', default = os.path.join( ROOT, '.scratch/ktx/enc' ) )
	a = ap.parse_args()
	ktx = find_ktx( a.ktx )
	env = dict( os.environ, LD_LIBRARY_PATH = os.path.join( os.path.dirname( os.path.dirname( ktx ) ), 'lib' ) + ':' + os.environ.get( 'LD_LIBRARY_PATH', '' ) )
	bake = os.path.join( ROOT, a.bake ) if not os.path.isabs( a.bake ) else a.bake
	out = os.path.join( ROOT, a.out ) if not os.path.isabs( a.out ) else a.out
	if not re.match( r'_nextktx\w*$', os.path.basename( out ) ): sys.exit( 'ktx_layers: --out must be a folder named _nextktx...' )
	man = json.load( open( os.path.join( bake, 'manifest.json' ) ) )
	names = man[ 'layers' ]; N = len( names )
	os.makedirs( out, exist_ok = True ); os.makedirs( a.scratch, exist_ok = True )
	tmp = tempfile.mkdtemp( prefix = 'run', dir = a.scratch )
	hfade = [ float( v ) for v in a.hfade.split( ',' ) ] if a.hfade else None
	hk = a.hmips or a.mips

	cache = {}
	cd = os.path.join( bake, '.texcache' )
	for f in ( sorted( os.listdir( cd ), key = lambda f: os.path.getmtime( os.path.join( cd, f ) ) ) if os.path.isdir( cd ) else [] ):

		m = re.match( r'(.*)_([0-9a-f]{16})\.npz$', f )
		if m: cache[ m.group( 1 ) ] = os.path.join( cd, f ) # (the newest of a name wins)

	want = set( a.sets.split( ',' ) ) if a.sets else None
	on = lambda k: want is None or k in want
	src_note = { 'cache': 0, 'webp': 0 }

	def colour_src( i, size ):

		# the page's bytes of layer i at `size`: (S, S, 4), sRGB colour, alpha = 0.5 + 0.5 x height
		cf = cache.get( names[ i ] )
		if cf:

			z = np.load( cf )
			if 's%d' % size in z:

				px = z[ 's%d' % size ].copy()
				px[ ..., 3 ] = np.round( 127.5 + 0.5 * px[ ..., 3 ].astype( np.float64 ) ).astype( np.uint8 ) # (write_layer of webbake_bl.py)
				src_note[ 'cache' ] += 1
				return px

		src_note[ 'webp' ] += 1
		tl = man.get( 'texl', {} ).get( str( size ) )
		if tl and tl[ 'files' ][ i ]: return webp_rgba( os.path.join( bake, tl[ 'files' ][ i ] + '.webp' ) )
		t = man[ 'tex' ][ str( size ) ]; per = t[ 'cols' ] * t[ 'rows' ]
		k = i % per; col = k % t[ 'cols' ]; row = k // t[ 'cols' ]
		return webp_rgba( os.path.join( bake, t[ 'pages' ][ i // per ] + '.webp' ), ( size, size, col * size, row * size ) )

	def nr_src( i, size ):

		cf = cache.get( names[ i ] )
		if cf:

			z = np.load( cf )
			if 'n%d' % size in z: src_note[ 'cache' ] += 1; return z[ 'n%d' % size ]

		src_note[ 'webp' ] += 1
		return webp_rgba( os.path.join( bake, man[ 'texn' ][ str( size ) ][ 'files' ][ i ] + '.webp' ) )

	def encode( levels_per_layer, name, spec, srgb, threads ):

		# levels_per_layer: [ layer ][ level ] byte arrays -> <out>/<name>.ktx2 ; -> bytes
		L = len( levels_per_layer ); nl = len( levels_per_layer[ 0 ] ); size = levels_per_layer[ 0 ][ 0 ].shape[ 0 ]; ch = levels_per_layer[ 0 ][ 0 ].shape[ 2 ]
		files = []
		for lv in range( nl ):

			for ly in range( L ):

				f = os.path.join( tmp, '%s_%d_%d.raw' % ( name, lv, ly ) )
				np.ascontiguousarray( levels_per_layer[ ly ][ lv ] ).tofile( f ); files.append( f )

		dst = os.path.join( out, name + '.ktx2' )
		cmd = [ ktx, 'create', '--raw', '--width', str( size ), '--height', str( size ), '--levels', str( nl ) ] + ( [ '--layers', str( L ) ] if L > 1 else [] ) + \
			codec_args( spec, srgb, ch ) + [ '--threads', str( threads ) ] + files + [ dst + '.tmp' ]
		r = subprocess.run( cmd, env = env, capture_output = True, text = True )
		for f in files: os.remove( f )
		if r.returncode != 0: raise RuntimeError( '%s: %s' % ( name, ( r.stderr or r.stdout )[ - 600: ] ) )
		os.replace( dst + '.tmp', dst )
		return os.path.getsize( dst )

	t0 = time.time()
	J = dict( version = 1, baked = man[ 'baked' ], built = int( time.time() ), layers = N, tool = 'ktx_layers.py + ' + subprocess.run( [ ktx, '--version' ], env = env, capture_output = True, text = True ).stdout.strip(),
		colour = a.colour, nr = a.nr, mips = dict( colour = a.mips, height = hk, hfade = hfade, nr = a.nrmips ), c = {}, l = {}, n = {}, seconds = {} )
	cmds = {}

	# the page sets: one array file each
	for size in sorted( int( s ) for s in man[ 'tex' ] ):

		if not on( 'c%d' % size ): continue
		t1 = time.time()
		with concurrent.futures.ThreadPoolExecutor( a.jobs ) as ex: lv = list( ex.map( lambda i: colour_levels( colour_src( i, size ), a.mips, hk, hfade ), range( N ) ) )
		b = encode( lv, 'c%d' % size, a.colour, True, a.jobs )
		J[ 'c' ][ str( size ) ] = dict( size = size, file = 'c%d' % size, layers = N, levels = len( lv[ 0 ] ), bytes = b )
		J[ 'seconds' ][ 'c%d' % size ] = round( time.time() - t1, 1 )
		print( 'c%d: %d layers, %.2f MB, %.1fs' % ( size, N, b / 1e6, time.time() - t1 ), flush = True )
		del lv

	# the single layers
	def one( kind, size, i ):

		if kind == 'l': return encode( [ colour_levels( colour_src( i, size ), a.mips, hk, hfade ) ], 'l%d_%d' % ( size, i ), a.colour, True, 1 )
		return encode( [ nr_levels( nr_src( i, size ), a.nrmips ) ], 'n%d_%d' % ( size, i ), a.nr, False, 1 )

	for kind, key in ( ( 'l', 'texl' ), ( 'n', 'texn' ) ):

		for size in sorted( int( s ) for s in man.get( key, {} ) ):

			if not on( '%s%d' % ( kind, size ) ): continue
			t1 = time.time()
			have = [ i for i, f in enumerate( man[ key ][ str( size ) ][ 'files' ] ) if f ]
			with concurrent.futures.ThreadPoolExecutor( a.jobs ) as ex: sizes = list( ex.map( lambda i: one( kind, size, i ), have ) )
			files = [ None ] * N; each = [ 0 ] * N
			for i, b in zip( have, sizes ): files[ i ] = '%s%d_%d' % ( kind, size, i ); each[ i ] = b
			J[ kind ][ str( size ) ] = dict( size = size, files = files, each = each, bytes = sum( sizes ) )
			J[ 'seconds' ][ '%s%d' % ( kind, size ) ] = round( time.time() - t1, 1 )
			print( '%s%d: %d layers, %.2f MB, %.1fs' % ( kind, size, len( have ), sum( sizes ) / 1e6, time.time() - t1 ), flush = True )

	J[ 'seconds' ][ 'all' ] = round( time.time() - t0, 1 )
	J[ 'sources' ] = src_note
	# (a run of some sets keeps what an earlier run of the same settings wrote)
	jf = os.path.join( out, 'ktx.json' )
	if want is not None and os.path.exists( jf ):

		old = json.load( open( jf ) )
		for k in ( 'c', 'l', 'n' ):

			for s, v in old.get( k, {} ).items(): J[ k ].setdefault( s, v )

	json.dump( J, open( jf, 'w' ), separators = ( ',', ':' ) )
	if a.lean and man.get( 'lean' ) and J[ 'c' ]:

		# low / phone run a bake of their own (<bake>/lean) with the same pages: its folder gets the page sets
		lm = os.path.join( bake, man[ 'lean' ], 'manifest.json' )
		if os.path.exists( lm ):

			ld = os.path.join( out, man[ 'lean' ] ); os.makedirs( ld, exist_ok = True )
			for s, v in J[ 'c' ].items(): shutil.copyfile( os.path.join( out, v[ 'file' ] + '.ktx2' ), os.path.join( ld, v[ 'file' ] + '.ktx2' ) )
			json.dump( dict( J, baked = json.load( open( lm ) )[ 'baked' ], l = {}, n = {} ), open( os.path.join( ld, 'ktx.json' ), 'w' ), separators = ( ',', ':' ) )

	shutil.rmtree( tmp )
	print( 'ktx_layers: %s  colour %s  nr %s  mips %s  %.1fs  sources %s' % ( os.path.relpath( out, ROOT ), a.colour, a.nr, json.dumps( J[ 'mips' ] ), time.time() - t0, src_note ) )

if __name__ == '__main__': main()
