# numpy post for the Cycles reference: the page's exposure / tone map / grade on the linear EXR, AgX versions,
# pass pictures, a contact sheet and the share of the image-mean luminance each light pass carries.
# Imported by town_cycles.py (Blender's python: numpy + OpenImageIO); can also be run on existing EXRs:
#   blender --background --factory-startup --python tools/cyclesref/post.py -- exr=<file.exr> shot=<name> out=<dir> cam=x,y,z
import os, sys, json, math
import numpy as np
import OpenImageIO as oiio

LW = np.array( [ 0.2126729, 0.7151522, 0.0721750 ], np.float32 )

# the page's picture controls (web/town/main.js LOOK + eng.P, web/engine/engine.js night rig); town_cycles.py re-reads them
PAGE = dict( exposure = 0.72, vignette = 0.35, contrast = 1.06, saturation = 1.03, knee = 1.6, knee_glow = 2.8,
	fog = 0.0075, fog_height = 18.0, haze = 0.35, bloom = 0.3 )

def lum( a ): return a[ ..., :3 ] @ LW

def read_exr( path ):

	inp = oiio.ImageInput.open( path )
	if inp is None: raise RuntimeError( 'cannot open ' + path + ': ' + oiio.geterror() )
	spec = inp.spec()
	a = inp.read_image( 'float' )
	inp.close()
	groups = {}
	for i, n in enumerate( spec.channelnames ):

		parts = n.split( '.' )
		key = parts[ - 2 ] if len( parts ) >= 2 else 'Combined'
		if len( parts ) >= 3 and parts[ 0 ] == 'Composite': key = 'Composite'
		groups.setdefault( key, {} )[ parts[ - 1 ] ] = i

	out = {}
	for k, ch in groups.items():

		order = [ c for c in ( 'R', 'G', 'B', 'A', 'X', 'Y', 'Z', 'V' ) if c in ch ]
		out[ k ] = np.stack( [ a[ :, :, ch[ c ] ] for c in order ], - 1 ).astype( np.float32 )

	return out

def write_png( path, rgb ):

	a = np.clip( np.nan_to_num( rgb ), 0, 1 )
	a = ( a * 255 + 0.5 ).astype( np.uint8 )
	h, w = a.shape[ :2 ]
	buf = oiio.ImageBuf( oiio.ImageSpec( w, h, 3, 'uint8' ) )
	buf.set_pixels( oiio.ROI( 0, w, 0, h ), np.ascontiguousarray( a[ ..., :3 ] ) )
	if not buf.write( path ): raise RuntimeError( buf.geterror() )

def srgb( x ):

	x = np.clip( x, 0, 1 )
	return np.where( x <= 0.0031308, x * 12.92, 1.055 * np.power( np.maximum( x, 1e-8 ), 1 / 2.4 ) - 0.055 )

def aces_three( c, exposure ):

	# three.js ACESFilmicToneMapping, value for value (the / 0.6 is three's own)
	A = np.array( [ [ 0.59719, 0.35458, 0.04823 ], [ 0.07600, 0.90834, 0.01566 ], [ 0.02840, 0.13383, 0.83777 ] ], np.float32 )
	B = np.array( [ [ 1.60475, - 0.53108, - 0.07367 ], [ - 0.10208, 1.10813, - 0.00605 ], [ - 0.00327, - 0.07276, 1.07602 ] ], np.float32 )
	v = ( c * ( exposure / 0.6 ) ) @ A.T
	v = ( v * ( v + 0.0245786 ) - 0.000090537 ) / ( v * ( 0.983729 * v + 0.4329510 ) + 0.238081 )
	return np.clip( v @ B.T, 0, 1 )

def knee( c, emit, surf, P ):

	# engine.js kneeOf(): lit surfaces roll off towards P.knee, glowing ones towards P.knee_glow (hue kept); not the sky
	if not P[ 'knee' ]: return c
	e = np.clip( lum( emit ) * 2, 0, 1 ) if emit is not None else 0.0
	m = P[ 'knee' ] + ( max( P[ 'knee_glow' ], P[ 'knee' ] ) - P[ 'knee' ] ) * e
	t = m * 0.35
	Lm = np.maximum( c.max( - 1 ), 1e-4 )
	x = np.maximum( Lm - t, 0 ) / np.maximum( m - t, 1e-4 )
	Lk = np.minimum( Lm, t ) + ( m - t ) * ( x / ( x + 1 ) )
	f = np.where( surf, Lk / Lm, 1.0 )
	return c * f[ ..., None ]

def fog( c, pos, surf, cam, P ):

	# engine.js: exp2 distance fog that thins with altitude; colour = the night rig's fog colour lifted by `haze`
	if pos is None or not P[ 'fog' ]: return c
	col = np.array( [ 0.016, 0.026, 0.058 ] ) * ( 1 - P[ 'haze' ] ) + np.array( [ 0.085, 0.07, 0.075 ] ) * P[ 'haze' ]
	d = np.linalg.norm( pos[ ..., :3 ] - np.asarray( cam, np.float32 )[ None, None, : ], axis = - 1 )
	thin = np.exp( - np.maximum( pos[ ..., 2 ], 0 ) / P[ 'fog_height' ] ) * 0.8 + 0.2
	fx = d * P[ 'fog' ] * thin
	f = np.where( surf, 1 - np.exp( - fx * fx ), 0.0 )[ ..., None ]
	return c * ( 1 - f ) + col.astype( np.float32 )[ None, None, : ] * f

def grade( c, P, vignette = True ):

	# engine.js grade(): vignette, contrast about 0.18, saturation, then tone map + sRGB (no bloom, grain, FXAA, dither)
	h, w = c.shape[ :2 ]
	if vignette and P[ 'vignette' ]:

		yy, xx = np.mgrid[ 0:h, 0:w ].astype( np.float32 )
		vx = ( xx + 0.5 ) / w - 0.5; vy = ( yy + 0.5 ) / h - 0.5
		c = c * ( 1 - ( vx * vx + vy * vy ) * P[ 'vignette' ] * 1.6 )[ ..., None ]

	c = np.power( np.maximum( c, 0 ) / 0.18, P[ 'contrast' ] ) * 0.18
	l = lum( c )[ ..., None ]
	c = l + ( c - l ) * P[ 'saturation' ]
	return srgb( aces_three( c, P[ 'exposure' ] ) )

def page( comb, emit, pos, surf, cam, P, with_fog = True, with_knee = True ):

	c = comb[ ..., :3 ].astype( np.float32 )
	if with_knee: c = knee( c, emit, surf, P )
	if with_fog: c = fog( c, pos, surf, cam, P )
	return grade( c, P )

_OCIO = [ None ]

def agx( lin, exposure = 0.0, view = 'AgX', look = None ):

	# Blender's own OpenColorIO config: scene-linear Rec.709 -> sRGB display, AgX view
	import PyOpenColorIO as ocio
	if _OCIO[ 0 ] is None:

		import bpy
		_OCIO[ 0 ] = ocio.Config.CreateFromFile( os.path.join( bpy.utils.system_resource( 'DATAFILES' ), 'colormanagement', 'config.ocio' ) )

	cfg = _OCIO[ 0 ]
	tr = ocio.DisplayViewTransform( src = 'Linear Rec.709', display = 'sRGB', view = view )
	if look:

		lt = ocio.LegacyViewingPipeline()
		lt.setDisplayViewTransform( tr ); lt.setLooksOverrideEnabled( True ); lt.setLooksOverride( look )
		proc = lt.getProcessor( cfg, cfg.getCurrentContext() )

	else: proc = cfg.getProcessor( tr )
	cpu = proc.getDefaultCPUProcessor()
	a = np.ascontiguousarray( np.maximum( lin[ ..., :3 ], 0 ) * ( 2.0 ** exposure ), np.float32 )
	h, w = a.shape[ :2 ]
	a = a.reshape( - 1, 3 ).copy()
	cpu.applyRGB( a )
	return a.reshape( h, w, 3 )

def half( a ):

	h, w = a.shape[ 0 ] // 2 * 2, a.shape[ 1 ] // 2 * 2
	a = a[ :h, :w ]
	return ( a[ 0::2, 0::2 ] + a[ 1::2, 0::2 ] + a[ 0::2, 1::2 ] + a[ 1::2, 1::2 ] ) * 0.25

def bilinear( a, H, W ):

	h, w = a.shape[ :2 ]
	y = ( np.arange( H ) + 0.5 ) * h / H - 0.5; x = ( np.arange( W ) + 0.5 ) * w / W - 0.5
	y0 = np.clip( np.floor( y ).astype( int ), 0, h - 1 ); x0 = np.clip( np.floor( x ).astype( int ), 0, w - 1 )
	y1 = np.minimum( y0 + 1, h - 1 ); x1 = np.minimum( x0 + 1, w - 1 )
	fy = np.clip( y - y0, 0, 1 )[ :, None, None ].astype( np.float32 ); fx = np.clip( x - x0, 0, 1 )[ None, :, None ].astype( np.float32 )
	top = a[ y0 ][ :, x0 ] * ( 1 - fx ) + a[ y0 ][ :, x1 ] * fx
	bot = a[ y1 ][ :, x0 ] * ( 1 - fx ) + a[ y1 ][ :, x1 ] * fx
	return top * ( 1 - fy ) + bot * fy

def blur( a, taps = ( 1, 4, 6, 4, 1 ) ):

	k = np.array( taps, np.float32 ); k /= k.sum(); r = len( k ) // 2
	p = np.pad( a, ( ( r, r ), ( 0, 0 ), ( 0, 0 ) ), mode = 'edge' )
	a = sum( k[ i ] * p[ i:i + a.shape[ 0 ] ] for i in range( len( k ) ) )
	p = np.pad( a, ( ( 0, 0 ), ( r, r ), ( 0, 0 ) ), mode = 'edge' )
	return sum( k[ i ] * p[ :, i:i + a.shape[ 1 ] ] for i in range( len( k ) ) )

def glare( lin, threshold = 1.0, strength = 0.06 ):

	# a modest fog-glow: what is brighter than `threshold` spreads over a 5-level pyramid
	c = np.maximum( lin[ ..., :3 ], 0 ).astype( np.float32 )
	l = np.maximum( c.max( - 1 ), 1e-6 )
	b = c * ( np.maximum( l - threshold, 0 ) / l )[ ..., None ]
	b = np.minimum( b, 60.0 )
	H, W = c.shape[ :2 ]
	acc = np.zeros_like( c ); cur = b
	for i in range( 6 ):

		cur = blur( half( cur ) )
		if i >= 1: acc += bilinear( cur, H, W )

	return c + acc * strength

def contributions( L ):

	z = None
	def g( n ):

		a = L.get( n )
		return a[ ..., :3 ] if a is not None else None

	def prod( a, b ):

		if g( a ) is None or g( b ) is None: return None
		return g( a ) * g( b )

	C = {}
	C[ 'diffuse_direct' ] = prod( 'DiffDir', 'DiffCol' )
	C[ 'diffuse_indirect' ] = prod( 'DiffInd', 'DiffCol' )
	C[ 'glossy_direct' ] = prod( 'GlossDir', 'GlossCol' )
	C[ 'glossy_indirect' ] = prod( 'GlossInd', 'GlossCol' )
	td, ti = prod( 'TransDir', 'TransCol' ), prod( 'TransInd', 'TransCol' )
	C[ 'transmission' ] = None if td is None else td + ( ti if ti is not None else 0 )
	C[ 'emission' ] = g( 'Emit' )
	C[ 'environment' ] = g( 'Env' )
	return { k: v for k, v in C.items() if v is not None }

def process( exr, shot, out_dir, cam, P = PAGE, agx_exposure = 0.0, log = print ):

	L = read_exr( exr )
	comb = L[ 'Combined' ][ ..., :3 ]
	emit = L[ 'Emit' ][ ..., :3 ] if 'Emit' in L else None
	pos = L.get( 'Position' )
	depth = L.get( 'Depth' )
	surf = ( depth[ ..., 0 ] < 1e9 ) if depth is not None else np.ones( comb.shape[ :2 ], bool )
	os.makedirs( os.path.join( out_dir, 'passes' ), exist_ok = True )
	write_png( os.path.join( out_dir, shot + '_page.png' ), page( comb, emit, pos, surf, cam, P ) )
	write_png( os.path.join( out_dir, shot + '_page_nofog.png' ), page( comb, emit, pos, surf, cam, P, with_fog = False ) )
	try:

		write_png( os.path.join( out_dir, shot + '_agx.png' ), agx( comb, agx_exposure ) )
		write_png( os.path.join( out_dir, shot + '_agx_glare.png' ), agx( glare( comb ), agx_exposure ) )

	except Exception as e: log( 'AgX conversion failed: %r' % ( e, ) )

	C = contributions( L )
	total = sum( C.values() )
	mt = float( lum( total ).mean() )
	res = dict( shot = shot, mean_luminance_combined = float( lum( comb ).mean() ), mean_luminance_sum_of_passes = mt,
		surface_pixels = float( surf.mean() ), share = {}, share_surface_only = {}, lightgroups = {}, mean_rgb = {} )
	ms = float( lum( total )[ surf ].mean() ) if surf.any() else 1.0
	for k, v in C.items():

		res[ 'share' ][ k ] = float( lum( v ).mean() ) / max( mt, 1e-9 )
		res[ 'share_surface_only' ][ k ] = ( float( lum( v )[ surf ].mean() ) / max( ms, 1e-9 ) ) if surf.any() else 0.0
		res[ 'mean_rgb' ][ k ] = [ float( t ) for t in v.reshape( - 1, 3 ).mean( 0 ) ]

	groups = { k[ len( 'Combined_' ): ]: v[ ..., :3 ] for k, v in L.items() if k.startswith( 'Combined_' ) }
	gt = sum( float( lum( v ).mean() ) for v in groups.values() ) or 1.0
	for k, v in groups.items(): res[ 'lightgroups' ][ k ] = float( lum( v ).mean() ) / gt
	if 'AO' in L: res[ 'ao_mean_surface' ] = float( lum( L[ 'AO' ] )[ surf ].mean() ) if surf.any() else 1.0

	# pass pictures: half size, the page's exposure and tone map (no fog / knee / vignette) so they compare with each other
	def tone( a ): return srgb( aces_three( np.maximum( half( a ), 0 ), P[ 'exposure' ] ) )
	tiles = [ ( 'combined (page tone map)', tone( comb ) ) ]
	for k in ( 'diffuse_direct', 'diffuse_indirect', 'glossy_direct', 'glossy_indirect', 'emission', 'transmission' ):

		if k in C:

			t = tone( C[ k ] )
			write_png( os.path.join( out_dir, 'passes', '%s_%s.png' % ( shot, k ) ), t )
			tiles.append( ( '%s  %.1f%%' % ( k.replace( '_', ' ' ), 100 * res[ 'share' ][ k ] ), t ) )

	if 'AO' in L:

		t = srgb( np.clip( half( L[ 'AO' ][ ..., :3 ] ), 0, 1 ) )
		write_png( os.path.join( out_dir, 'passes', '%s_ao.png' % shot ), t )
		tiles.append( ( 'ambient occlusion (1 m)', t ) )

	for k in ( 'lanterns', 'helpers', 'emit', 'moon', 'sky' ):

		if k in groups:

			t = tone( groups[ k ] )
			write_png( os.path.join( out_dir, 'passes', '%s_light_%s.png' % ( shot, k ) ), t )
			tiles.append( ( 'light group: %s  %.1f%%' % ( k, 100 * res[ 'lightgroups' ][ k ] ), t ) )

	cols = 4; rows = ( len( tiles ) + cols - 1 ) // cols
	th, tw = tiles[ 0 ][ 1 ].shape[ 0 ] // 2, tiles[ 0 ][ 1 ].shape[ 1 ] // 2
	sheet = np.zeros( ( rows * th, cols * tw, 3 ), np.float32 )
	for i, ( name, t ) in enumerate( tiles ):

		r, c = divmod( i, cols )
		sheet[ r * th:( r + 1 ) * th, c * tw:( c + 1 ) * tw ] = half( t )[ :th, :tw ]

	a8 = ( np.clip( sheet, 0, 1 ) * 255 + 0.5 ).astype( np.uint8 )
	buf = oiio.ImageBuf( oiio.ImageSpec( sheet.shape[ 1 ], sheet.shape[ 0 ], 3, 'uint8' ) )
	buf.set_pixels( oiio.ROI( 0, sheet.shape[ 1 ], 0, sheet.shape[ 0 ] ), a8 )
	sheet_path = os.path.join( out_dir, shot + '_passes.png' )
	buf.write( sheet_path )
	labelled = False
	try:

		# OpenImageIO in Blender has no FreeType: label the tiles with Blender's own font drawing into the picture
		import blf, imbuf
		ib = imbuf.load( sheet_path )
		with blf.bind_imbuf( 0, ib ):

			blf.size( 0, 15.0 ); blf.color( 0, 1.0, 1.0, 1.0, 1.0 )
			for i, ( name, t ) in enumerate( tiles ):

				r, c = divmod( i, cols )
				blf.position( 0, c * tw + 8, sheet.shape[ 0 ] - r * th - 20, 0 )
				blf.draw_buffer( 0, name )

		imbuf.write( ib, filepath = sheet_path )
		labelled = True

	except Exception as e: log( 'contact sheet left unlabelled (%r); tile order, row by row: %s' % ( e, '; '.join( n for n, _ in tiles ) ) )
	res[ 'sheet_tiles' ] = [ n for n, _ in tiles ]
	res[ 'sheet_labelled' ] = labelled
	return res

if __name__ == '__main__':

	A = dict( a.split( '=', 1 ) for a in sys.argv[ sys.argv.index( '--' ) + 1: ] ) if '--' in sys.argv else {}
	if A.get( 'exr' ):

		cam = [ float( t ) for t in A.get( 'cam', '0,0,0' ).split( ',' ) ]
		r = process( A[ 'exr' ], A.get( 'shot', 'shot' ), A.get( 'out', os.path.dirname( A[ 'exr' ] ) ), cam, PAGE, float( A.get( 'agx', 0 ) ) )
		print( json.dumps( r, indent = 1 ) )
