# Cycles node port of the page's material library (web/engine/materials.js: _common, _weather, _baked, _special),
# with the constants and the per-library logic of tools/webbake_bl.py (shade_prim / weather) as the reference.
# Runs inside Blender 4.1. See town_cycles.py for what is exact and what is approximated.
import bpy, os, json, math
import numpy as np

SOCK = bpy.types.NodeSocket

def K( x ): return not isinstance( x, SOCK ) # a python constant (not a node socket)

def ss_k( e0, e1, x ):

	if e0 > e1: return 1.0 - ss_k( e1, e0, x )
	t = min( 1.0, max( 0.0, ( x - e0 ) / ( e1 - e0 ) ) )
	return t * t * ( 3 - 2 * t )

class G:

	# small expression builder over a shader node tree: python numbers fold, sockets make nodes
	def __init__( self, nt ): self.nt = nt
	def new( self, t, **kw ):

		n = self.nt.nodes.new( t )
		for k, v in kw.items(): setattr( n, k, v )
		return n

	def put( self, inp, v ):

		if not K( v ): self.nt.links.new( v, inp ); return
		if inp.type == 'VALUE': inp.default_value = float( v )
		else:

			a = np.broadcast_to( np.asarray( v, float ).ravel()[ :3 ] if np.ndim( v ) else np.asarray( v, float ), ( 3, ) )
			inp.default_value = ( a[ 0 ], a[ 1 ], a[ 2 ] ) if inp.type == 'VECTOR' else ( a[ 0 ], a[ 1 ], a[ 2 ], 1.0 )

	# ---- floats
	def m( self, op, a, b = None, c = None, clamp = False ):

		n = self.new( 'ShaderNodeMath', operation = op ); n.use_clamp = clamp
		self.put( n.inputs[ 0 ], a )
		if b is not None: self.put( n.inputs[ 1 ], b )
		if c is not None: self.put( n.inputs[ 2 ], c )
		return n.outputs[ 0 ]

	def add( self, a, b ):

		if K( a ) and K( b ): return a + b
		if K( a ) and a == 0: return b
		if K( b ) and b == 0: return a
		return self.m( 'ADD', a, b )

	def sub( self, a, b ):

		if K( a ) and K( b ): return a - b
		if K( b ) and b == 0: return a
		return self.m( 'SUBTRACT', a, b )

	def mul( self, a, b ):

		if K( a ) and K( b ): return a * b
		for p, q in ( ( a, b ), ( b, a ) ):

			if K( p ):

				if p == 0: return 0.0
				if p == 1: return q

		return self.m( 'MULTIPLY', a, b )

	def div( self, a, b ):

		if K( b ): return self.mul( a, 1.0 / b )
		return self.m( 'DIVIDE', a, b )

	def ma( self, a, b, c ): # a * b + c

		if not K( a ) and K( b ) and K( c ): return self.m( 'MULTIPLY_ADD', a, b, c )
		return self.add( self.mul( a, b ), c )

	def mn( self, a, b ): return min( a, b ) if K( a ) and K( b ) else self.m( 'MINIMUM', a, b )
	def mx( self, a, b ):

		if K( a ) and K( b ): return max( a, b )
		if K( a ) and a == 0 and False: return b
		return self.m( 'MAXIMUM', a, b )

	def pw( self, a, b ): return a ** b if K( a ) and K( b ) else self.m( 'POWER', a, b )
	def ab( self, a ): return abs( a ) if K( a ) else self.m( 'ABSOLUTE', a )
	def fr( self, a ): return a - math.floor( a ) if K( a ) else self.m( 'FRACT', a )
	def fl( self, a ): return math.floor( a ) if K( a ) else self.m( 'FLOOR', a )
	def ex( self, a ): return math.exp( a ) if K( a ) else self.m( 'EXPONENT', a )
	def sat( self, a ): return min( 1.0, max( 0.0, a ) ) if K( a ) else self.m( 'ADD', a, 0.0, clamp = True )
	def om( self, a ): return self.sub( 1.0, a )
	def gt( self, a, b ): return float( a > b ) if K( a ) and K( b ) else self.m( 'GREATER_THAN', a, b )
	def ss( self, e0, e1, x ):

		if K( e0 ) and K( e1 ) and K( x ): return ss_k( e0, e1, x )
		n = self.new( 'ShaderNodeMapRange', interpolation_type = 'SMOOTHSTEP' )
		self.put( n.inputs[ 0 ], x ); self.put( n.inputs[ 1 ], e0 ); self.put( n.inputs[ 2 ], e1 )
		n.inputs[ 3 ].default_value = 0.0; n.inputs[ 4 ].default_value = 1.0
		return n.outputs[ 0 ]

	def mix( self, a, b, t ):

		if K( t ):

			if t == 0: return a
			if t == 1: return b

		if K( a ) and K( b ):

			if K( t ): return a + ( b - a ) * t
			return self.m( 'MULTIPLY_ADD', t, b - a, a )

		n = self.new( 'ShaderNodeMix', data_type = 'FLOAT', clamp_factor = False )
		self.put( n.inputs[ 0 ], t ); self.put( n.inputs[ 2 ], a ); self.put( n.inputs[ 3 ], b )
		return n.outputs[ 0 ]

	# ---- vectors / colours (constants are numpy arrays of 3)
	def vm( self, op, a, b = None, scale = None, out = 0 ):

		n = self.new( 'ShaderNodeVectorMath', operation = op )
		self.put( n.inputs[ 0 ], a )
		if b is not None: self.put( n.inputs[ 1 ], b )
		if scale is not None: self.put( n.inputs[ 3 ], scale )
		return n.outputs[ out ]

	def vs( self, v, f ): # vector * float

		if K( f ):

			if f == 1: return v
			if K( v ): return np.asarray( v, float ) * f

		return self.vm( 'SCALE', v, scale = f )

	def vv( self, a, b ): # vector * vector

		if K( a ) and K( b ): return np.asarray( a, float ) * np.asarray( b, float )
		return self.vm( 'MULTIPLY', a, b )

	def va( self, a, b ):

		if K( a ) and K( b ): return np.asarray( a, float ) + np.asarray( b, float )
		return self.vm( 'ADD', a, b )

	def vmin( self, a, b ): return np.minimum( a, b ) if K( a ) and K( b ) else self.vm( 'MINIMUM', a, b )
	def vmax( self, a, b ): return np.maximum( a, b ) if K( a ) and K( b ) else self.vm( 'MAXIMUM', a, b )
	def vmix( self, a, b, t ):

		if K( t ):

			if t == 0: return a
			if t == 1: return b
			if K( a ) and K( b ): return np.asarray( a, float ) * ( 1 - t ) + np.asarray( b, float ) * t

		n = self.new( 'ShaderNodeMix', data_type = 'VECTOR', clamp_factor = False )
		self.put( n.inputs[ 0 ], t ); self.put( n.inputs[ 4 ], a ); self.put( n.inputs[ 5 ], b )
		return n.outputs[ 1 ]

	def lum( self, v ):

		w = ( 0.2126729, 0.7151522, 0.0721750 )
		if K( v ): return float( np.dot( np.asarray( v, float ), w ) )
		return self.vm( 'DOT_PRODUCT', v, w, out = 1 )

	def vec( self, x, y, z ):

		if K( x ) and K( y ) and K( z ): return np.array( [ x, y, z ], float )
		n = self.new( 'ShaderNodeCombineXYZ' )
		self.put( n.inputs[ 0 ], x ); self.put( n.inputs[ 1 ], y ); self.put( n.inputs[ 2 ], z )
		return n.outputs[ 0 ]

	def sep( self, v ):

		if K( v ): return tuple( float( t ) for t in np.asarray( v, float ) )
		n = self.new( 'ShaderNodeSeparateXYZ' ); self.put( n.inputs[ 0 ], v )
		return n.outputs[ 0 ], n.outputs[ 1 ], n.outputs[ 2 ]

	def tex( self, img, u, v, interp = 'Linear' ):

		n = self.new( 'ShaderNodeTexImage', image = img, interpolation = interp, extension = 'REPEAT' )
		self.put( n.inputs[ 0 ], self.vec( u, v, 0.0 ) if not ( K( u ) and K( v ) ) else self._const_vec( u, v ) )
		return n.outputs[ 0 ], n.outputs[ 1 ]

	def _const_vec( self, u, v ):

		n = self.new( 'ShaderNodeCombineXYZ' ); n.inputs[ 0 ].default_value = u; n.inputs[ 1 ].default_value = v
		return n.outputs[ 0 ]

class Common:

	# MaterialLib._common in Blender's frame: three (x, y, z) = Blender (x, z, -y)
	def __init__( self, g, noise_img, scale = 1.0 ):

		self.g = g; self.img = noise_img; self._c = {}
		geo = g.new( 'ShaderNodeNewGeometry' )
		bx, by, bz = g.sep( geo.outputs[ 'Position' ] )
		self.wx = bx; self.wy = bz; self.wz = g.mul( by, - 1.0 )
		self.q0 = g.add( bx, g.mul( bz, 0.37 ) ); self.q1 = g.add( self.wz, g.mul( bz, 0.61 ) )
		self.ny = g.sep( geo.outputs[ 'Normal' ] )[ 2 ]
		at = g.new( 'ShaderNodeAttribute', attribute_name = 'tsq_mask' )
		r, gg, b = g.sep( at.outputs[ 'Color' ] )
		self.mask = ( r, gg, b, at.outputs[ 'Alpha' ] )
		uv = g.new( 'ShaderNodeUVMap', uv_map = 'UVMap' )
		u, v, _ = g.sep( uv.outputs[ 0 ] )
		self.uvx = g.mul( u, 1.0 / scale ); self.uvy = g.mul( v, 1.0 / scale ) # metres, V up

	def noise( self, u, v ):

		col, a = self.g.tex( self.img, u, v )
		r, gg, b = self.g.sep( col )
		return ( r, gg, b, a )

	def _get( self, k, fn ):

		if k not in self._c: self._c[ k ] = fn()
		return self._c[ k ]

	def nLo( self ): g = self.g; return self._get( 'lo', lambda: self.noise( g.mul( self.q0, 0.11 ), g.mul( self.q1, 0.11 ) ) )
	def nHi( self ): g = self.g; return self._get( 'hi', lambda: self.noise( g.ma( self.q0, 0.93, 0.31 ), g.ma( self.q1, 0.93, 0.77 ) ) )
	def nP( self ): g = self.g; return self._get( 'p', lambda: self.noise( g.ma( self.q0, 0.031, 0.53 ), g.ma( self.q1, 0.031, 0.17 ) ) )
	def _along( self ): g = self.g; return self._get( 'al', lambda: g.add( g.mul( self.wx, 1.1 ), g.mul( self.wz, 0.9 ) ) )
	def streak( self ): g = self.g; return self._get( 'st', lambda: self.noise( g.mul( self._along(), 0.45 ), g.mul( self.wy, 0.035 ) )[ 1 ] )
	def broad( self ): g = self.g; return self._get( 'br', lambda: self.noise( g.ma( self._along(), 0.034, 0.21 ), g.ma( self.wy, 0.05, 0.63 ) ) )
	def objy( self ):

		def f():

			n = self.g.new( 'ShaderNodeTexCoord' )
			return self.g.sep( n.outputs[ 'Object' ] )[ 2 ]

		return self._get( 'oy', f )

MASONRY = ( 'ashlar', 'rubble', 'brick', 'stone', 'plaster', 'mortar' )
WASH = dict( foot = 0.9, footGain = 0.9, runGain = 0.6, patch = 0.45 )

def weather( g, c, entry, s, L, lin, tide_y, rain_dark = 0.58 ):

	# MaterialLib._weather, per pixel. s: dict( alb, rough, metal, h, flat ). (`fine`, the pixel-footprint term that
	# turns the per-tile damp into broad patches from afar, is 1: the path tracer supersamples instead.)
	gr = entry.get( 'grime' ); w = entry.get( 'wear' ); sp = entry.get( 'special' )
	WET = L[ 'wet' ]; soft = L[ 'soft' ]
	mx_, my_, mz_, mw_ = c.mask
	lo = c.nLo(); hi = c.nHi(); ny = c.ny; h = s[ 'h' ]
	add, sub, mul, mix, ss = g.add, g.sub, g.mul, g.mix, g.ss

	def pools():

		nP = c.nP()
		zone = g.mn( add( mul( add( add( nP[ 0 ], mul( nP[ 2 ], 0.35 ) ), - 0.8 ), 5.0 ), WET - 0.5 ), L[ 'pool_cap' ] )
		level = add( add( add( zone, g.ma( lo[ 3 ], 2.4, - 1.2 ) ), g.ma( hi[ 3 ], 0.3, - 0.15 ) ), - 0.25 )
		lvl = mul( ss( 0.985, 0.998, ny ), ss_k( 0.05, 0.3, WET ) )
		depth = sub( level, mul( h, 0.8 ) )
		water = mul( ss( 0.0, 0.5, depth ), lvl ); halo = mul( ss( - 1.0, 0.1, depth ), lvl )
		s[ 'alb' ] = g.vs( s[ 'alb' ], mul( mix( 1.0, 0.78, halo ), mix( 1.0, 0.62, water ) ) )
		s[ 'rough' ] = mix( mul( s[ 'rough' ], mix( 1.0, 0.7, halo ) ), L[ 'puddle_rough' ], water )
		s[ 'flat' ] = g.mx( s[ 'flat' ], water )
		s[ 'water' ] = water

	if w:

		amt = mul( mz_, L[ 'wear' ] )
		k = mul( ss( 0.35, 0.8, add( add( mul( amt, 1.25 ), mul( sub( h, 0.5 ), 0.45 ) ), g.ma( hi[ 3 ], 0.5, - 0.25 ) ) ), ss( 0.02, 0.2, amt ) )
		wc = lin( w[ 'color' ] ); wk = 1.0 + ( 0.45 - 1.0 ) * soft
		if w.get( 'keep_hue' ): hc = g.vmin( g.vs( s[ 'alb' ], 1.6 ), g.vmax( wc, s[ 'alb' ] ) )
		else: hc = g.vmix( wc, g.vmin( g.vs( s[ 'alb' ], 1.5 ), g.vmax( wc, s[ 'alb' ] ) ), soft * 0.6 )
		s[ 'alb' ] = g.vmix( s[ 'alb' ], hc, mul( k, ( w.get( 'amount' ) if w.get( 'amount' ) is not None else 0.85 ) * wk ) )
		s[ 'rough' ] = mix( s[ 'rough' ], float( w.get( 'rough', 0.5 ) ), mul( k, 0.9 * wk ) )
		s[ 'metal' ] = mix( s[ 'metal' ], float( w.get( 'metal' ) or 0 ), k )

	wet_kind = bool( sp and sp.get( 'kind' ) == 'wet' )
	if sp and sp.get( 'kind' ) and sp[ 'kind' ] != 'none':

		amt = mul( mw_, L[ 'special' ] )
		sc = lin( sp.get( 'color', '#000000' ) )
		if sp[ 'kind' ] == 'wet':

			amt = g.sat( add( amt, WET ) )
			k = mul( ss( 0.15, 0.75, sub( add( add( amt, g.ma( lo[ 1 ], 0.35, - 0.175 ) ), g.ma( hi[ 1 ], 0.18, - 0.09 ) ), mul( sub( h, 0.5 ), 0.2 ) ) ), ss( 0.02, 0.2, amt ) )
			puddle = mul( k, ss( 0.55, 0.2, add( h, mul( g.om( amt ), 0.4 ) ) ) )
			s[ 'alb' ] = g.vs( s[ 'alb' ], mix( 1.0, 0.5, k ) )
			sheet = mul( mul( k, L.get( 'wet_flat', 0.7 ) ), ss( 0.9, 0.99, ny ) )
			r45 = mul( s[ 'rough' ], 0.45 )
			s[ 'rough' ] = mix( mix( mix( s[ 'rough' ], r45, k ), g.mn( r45, L[ 'film_rough' ] ), sheet ), float( sp.get( 'rough', 0.08 ) ), puddle )
			s[ 'flat' ] = g.mx( s[ 'flat' ], g.mx( mul( puddle, 0.95 ), mul( sheet, 0.9 ) ) )
			pools()

		elif sp[ 'kind' ] == 'fade':

			k = mul( ss( 0.2, 0.8, add( amt, g.ma( lo[ 2 ], 0.4, - 0.2 ) ) ), ss( 0.02, 0.2, amt ) )
			l = g.lum( s[ 'alb' ] )
			grey = g.vec( l, l, l )
			s[ 'alb' ] = g.vmix( s[ 'alb' ], g.va( g.vs( g.vmix( grey, sc, 0.25 ), 1.25 ), np.full( 3, 0.02 ) ), mul( k, 0.65 ) )
			s[ 'rough' ] = mix( s[ 'rough' ], float( sp.get( 'rough', 0.9 ) ), mul( k, 0.6 ) )

		else:

			k = mul( ss( 0.42, 0.62, add( add( add( mul( amt, 1.15 ), mul( sub( 0.5, h ), 0.3 ) ), g.ma( lo[ 0 ], 0.45, - 0.225 ) ), g.ma( hi[ 0 ], 0.35, - 0.175 ) ) ), ss( 0.02, 0.2, amt ) )
			var = g.ma( hi[ 1 ], 0.7, 0.65 )
			s[ 'alb' ] = g.vmix( s[ 'alb' ], g.vs( sc, var ), mul( k, 0.92 ) )
			s[ 'rough' ] = mix( s[ 'rough' ], float( sp.get( 'rough', 0.9 ) ), k )
			s[ 'metal' ] = mix( s[ 'metal' ], 0.0, k )
			if sp[ 'kind' ] == 'moss': s[ 'flat' ] = g.mx( s[ 'flat' ], mul( k, 0.6 ) )

	if not wet_kind:

		amt = mul( ss( 0.5, 0.92, ny ), WET )
		k = mul( ss( 0.15, 0.75, add( add( amt, g.ma( lo[ 1 ], 0.35, - 0.175 ) ), sub( g.ma( hi[ 1 ], 0.18, - 0.09 ), mul( sub( h, 0.5 ), 0.2 ) ) ) ), ss( 0.02, 0.2, amt ) )
		puddle = mul( mul( k, ss( 0.55, 0.2, add( h, mul( g.om( amt ), 0.4 ) ) ) ), ss( 0.97, 0.995, ny ) )
		film = mul( mul( k, ss( 0.9, 0.99, ny ) ), ss( 0.3, 0.75, add( add( mul( lo[ 2 ], 0.7 ), mul( hi[ 3 ], 0.3 ) ), WET - 0.5 ) ) )
		s[ 'alb' ] = g.vs( s[ 'alb' ], mix( 1.0, rain_dark, k ) )
		s[ 'rough' ] = mix( mix( mix( s[ 'rough' ], mul( s[ 'rough' ], mix( 0.72, 0.45, ss( 0.85, 0.98, ny ) ) ), k ), g.mn( s[ 'rough' ], L[ 'film_rough' ] ), mul( film, 0.85 ) ), 0.07, puddle )
		s[ 'flat' ] = g.mx( s[ 'flat' ], g.mx( mul( puddle, 0.95 ), mul( film, L.get( 'wet_flat', 0.7 ) * 0.9 ) ) )
		pools()

	if gr:

		amt = mul( my_, L[ 'grime' ] )
		hterm = mul( sub( 0.5, h ), 0.4 )
		k0 = ss( 0.3, 0.85, add( add( add( mul( amt, 1.3 ), hterm ), g.ma( hi[ 2 ], 0.45, - 0.225 ) ), g.ma( lo[ 3 ], 0.3, - 0.15 ) ) )
		wall = ss( 0.35, 0.8, g.om( g.ab( ny ) ) )
		streak = c.streak()
		k1 = ss( 0.35, 0.9, add( add( add( add( add( mul( amt, 1.1 ), mul( g.om( mx_ ), 0.5 ) ), hterm ), g.ma( hi[ 2 ], 0.3, - 0.15 ) ),
			mul( g.ma( streak, 0.55, - 0.275 ), wall ) ), mul( g.ma( lo[ 3 ], 0.12, - 0.06 ), g.om( wall ) ) ) )
		k = mul( mix( k0, k1, soft ), ss( 0.02, 0.2, amt ) )
		gc = lin( gr[ 'color' ] )
		if soft >= 1: dirty = g.va( g.vs( s[ 'alb' ], 0.42 ), gc * 0.6 )
		else: dirty = g.vmix( g.vmix( g.vs( g.vv( s[ 'alb' ], gc ), 2.2 ), gc, 0.45 ), g.va( g.vs( s[ 'alb' ], 0.42 ), gc * 0.6 ), soft )
		s[ 'alb' ] = g.vmix( s[ 'alb' ], dirty, mul( k, 0.85 + ( 0.7 - 0.85 ) * soft ) )
		s[ 'rough' ] = mix( s[ 'rough' ], float( gr.get( 'rough', 0.9 ) ), mul( k, 0.85 ) )
		s[ 'metal' ] = mix( s[ 'metal' ], 0.0, mul( k, 0.8 ) )
		if entry.get( 'name', '' ).startswith( MASONRY ):

			broad = c.broad()
			top = g.ma( lo[ 2 ], WASH[ 'foot' ], 0.3 )
			foot = mul( ss( top, mul( top, 0.12 ), c.objy() ), wall )
			runs = mul( ss( 0.52, 0.86, add( add( streak, g.ma( broad[ 0 ], 0.7, - 0.35 ) ), mul( g.om( mx_ ), 0.6 ) ) ), wall )
			kw = mul( g.sat( add( mul( mul( foot, g.ma( hi[ 2 ], 0.5, 0.75 ) ), WASH[ 'footGain' ] ), mul( runs, WASH[ 'runGain' ] ) ) ), L[ 'grime' ] )
			s[ 'alb' ] = g.vs( g.vmix( s[ 'alb' ], g.va( g.vv( s[ 'alb' ], np.array( [ 0.4, 0.42, 0.37 ] ) ), gc * 0.6 ), kw ),
				g.ma( broad[ 1 ], WASH[ 'patch' ] * L[ 'grime' ], 1.0 - 0.5 * WASH[ 'patch' ] * L[ 'grime' ] ) )
			s[ 'rough' ] = mix( s[ 'rough' ], float( gr.get( 'rough', 0.9 ) ), mul( kw, 0.6 ) )

	top = add( tide_y, mul( add( add( mul( lo[ 2 ], 0.9 ), mul( hi[ 0 ], 0.3 ) ), 0.35 ), L[ 'tide_h' ] ) )
	band = ss( top, tide_y + 0.02, c.wy )
	s[ 'alb' ] = g.vmix( s[ 'alb' ], g.vv( s[ 'alb' ], np.array( [ 0.36, 0.42, 0.32 ] ) ), mul( band, 0.9 ) )
	s[ 'rough' ] = mix( s[ 'rough' ], g.mn( s[ 'rough' ], 0.3 ), mul( band, 0.8 ) )
	return s

class Lib:

	def __init__( self, wb, root ):

		self.wb = wb; self.root = root; self.L = dict( wb.LOOK ); self.cache = {}; self.imgs = {}
		self.noise = self._array_image( 'tsq_noise', wb.NOISE )
		self.room = self._array_image( 'tsq_room', wb.window_layer( 512 )[ ::- 1 ] )
		self.count = 0

	def _array_image( self, name, a ):

		h, w = a.shape[ :2 ]
		img = bpy.data.images.new( name, w, h, alpha = True, float_buffer = True, is_data = True )
		img.alpha_mode = 'CHANNEL_PACKED'
		img.pixels.foreach_set( np.ascontiguousarray( a, np.float32 ).ravel() )
		img.pack()
		return img

	def image( self, lib, kind ):

		k = ( lib, kind )
		if k not in self.imgs:

			img = bpy.data.images.load( os.path.join( self.root, 'out/mat', lib, kind + '.png' ), check_existing = True )
			img.colorspace_settings.name = 'sRGB' if kind == 'albedo' else 'Non-Color'
			img.alpha_mode = 'CHANNEL_PACKED'
			self.imgs[ k ] = img

		return self.imgs[ k ]

	def get( self, lib, tint, params ):

		wb = self.wb
		lib = lib if lib in wb.CATALOG else 'plain'
		key = ( lib, None if tint is None else tuple( round( float( t ), 4 ) for t in tint[ :3 ] ), json.dumps( params, sort_keys = True ) )
		if key in self.cache: return self.cache[ key ]
		mat = bpy.data.materials.new( 'tsq_%s_%d' % ( lib, len( self.cache ) ) )
		mat.use_nodes = True
		nt = mat.node_tree
		nt.nodes.clear()
		g = G( nt )
		out = g.new( 'ShaderNodeOutputMaterial' )
		entry = dict( wb.CATALOG[ lib ], name = lib )
		T = np.array( tint[ :3 ], float ) if tint is not None else None
		baked = entry.get( 'kind' ) == 'baked'
		info = wb.INFO.get( lib ) if baked else None
		if baked and info and os.path.exists( os.path.join( self.root, 'out/mat', lib, 'albedo.png' ) ): sh = self._baked( g, lib, entry, info, T, params )
		else: sh = self._special( g, lib if not baked else 'plain', entry, T, params, baked )
		nt.links.new( sh, out.inputs[ 0 ] )
		self.cache[ key ] = mat
		return mat

	# ---- shader ends
	def _principled( self, g, base, rough, metal, normal = None, emit = None, ior = 1.5, trans = 0.0 ):

		p = g.new( 'ShaderNodeBsdfPrincipled' )
		g.put( p.inputs[ 'Base Color' ], base )
		g.put( p.inputs[ 'Roughness' ], rough if K( rough ) else g.m( 'MAXIMUM', g.m( 'MINIMUM', rough, 1.0 ), 0.04 ) )
		g.put( p.inputs[ 'Metallic' ], metal if K( metal ) else g.sat( metal ) )
		p.inputs[ 'IOR' ].default_value = ior
		if trans: p.inputs[ 'Transmission Weight' ].default_value = trans
		if normal is not None: g.nt.links.new( normal, p.inputs[ 'Normal' ] )
		if emit is not None:

			g.put( p.inputs[ 'Emission Color' ], emit ); p.inputs[ 'Emission Strength' ].default_value = 1.0

		return p.outputs[ 0 ]

	def _mixsh( self, g, fac, a, b ):

		n = g.new( 'ShaderNodeMixShader' )
		g.put( n.inputs[ 0 ], fac ); g.nt.links.new( a, n.inputs[ 1 ] ); g.nt.links.new( b, n.inputs[ 2 ] )
		return n.outputs[ 0 ]

	def _transparent( self, g, col = ( 1, 1, 1 ) ):

		n = g.new( 'ShaderNodeBsdfTransparent' ); g.put( n.inputs[ 0 ], np.asarray( col, float ) )
		return n.outputs[ 0 ]

	def _no_shadow( self, g, sh ):

		# the page's NO_OCCLUDE kinds (lamp glass, glowing things, falls): no shadow of their own, so the lamp inside shines out
		lp = g.new( 'ShaderNodeLightPath' )
		return self._mixsh( g, lp.outputs[ 'Is Shadow Ray' ], sh, self._transparent( g ) )

	def _emission( self, g, col ):

		n = g.new( 'ShaderNodeEmission' ); g.put( n.inputs[ 0 ], col ); n.inputs[ 1 ].default_value = 1.0
		return n.outputs[ 0 ]

	# ---- MaterialLib._baked
	def _baked( self, g, lib, entry, info, T, params ):

		wb = self.wb; L = self.L
		scale = float( params.get( 'scale' ) or 1 )
		c = Common( g, self.noise, scale )
		tile = info.get( 'tile', [ 1, 1 ] )
		tu = g.mul( c.uvx, 1.0 / tile[ 0 ] ); tv = g.mul( c.uvy, 1.0 / tile[ 1 ] )
		uvt = g.vec( tu, tv, 0.0 )
		def tx( kind ):

			n = g.new( 'ShaderNodeTexImage', image = self.image( lib, kind ), extension = 'REPEAT' )
			g.nt.links.new( uvt, n.inputs[ 0 ] )
			return n.outputs[ 0 ], n.outputs[ 1 ]

		aT, _ = tx( 'albedo' ); nT, nA = tx( 'normal' ); oT, _ = tx( 'orm' )
		oR, oG, oB = g.sep( oT )
		mean = np.maximum( np.array( info[ 'preview' ][ :3 ], float ), 0.004 )
		alb = aT
		lo = c.nLo(); hi = c.nHi()
		if entry.get( 'tintable' ):

			if T is not None: alb = g.vv( alb, np.minimum( 6, T / mean ) )
			if lib == 'canvas' and params.get( 'stripe' ):

				f2 = np.minimum( 6, ( wb.lin( params[ 'tint2' ] ) if params.get( 'tint2' ) else wb.lin( '#d9d2c0' ) ) / mean )
				band = g.ss( 0.47, 0.53, g.mul( g.ab( g.sub( g.fr( g.mul( c.uvx, 1.0 / float( params[ 'stripe' ] ) ) ), 0.5 ) ), 2.0 ) )
				alb = g.vmix( alb, g.vv( aT, f2 ), band )

		foliage = lib in ( 'leaf', 'hedge' )
		leafy = float( params.get( 'leafy' ) or 0 ) if foliage else 0.0
		went = entry; rain_dark = 0.58
		if foliage: rain_dark = 0.88 if leafy else 0.7
		if leafy:

			went = dict( entry, wear = None, special = None )
			if params.get( 'tint2' ):

				f2 = np.minimum( 6, wb.lin( params[ 'tint2' ] ) / mean )
				k2 = g.ss( 0.38, 0.62, g.add( g.mul( c.mask[ 3 ], 0.75 ), g.mul( hi[ 1 ], 0.25 ) ) )
				alb = g.vmix( alb, g.vv( aT, f2 ), k2 )

			# (the page also darkens a card by how deep it hangs in the crown, mask.z: painted occlusion, left to the path tracer)
			alb = g.vs( alb, g.ma( hi[ 2 ], 0.5, 0.75 ) )

		if foliage:

			l = g.add( g.lum( alb ), 0.0005 )
			r_, g_, b_ = g.sep( alb )
			green = g.ss( 0.0, 0.25, g.div( g.sub( g_, g.mx( r_, b_ ) ), l ) )
			kg = g.mn( 1.0, g.div( g.ma( g.mx( g.sub( l, 0.1 ), 0.0 ), 0.15, 0.1 ), l ) )
			alb = g.vmix( alb, g.vs( g.vmix( alb, g.vec( l, l, l ), g.mul( g.ss( 0.08, 0.3, l ), 0.45 ) ), kg ), green )

		M = L[ 'macro' ]
		alb = g.vs( alb, g.add( g.add( g.ma( lo[ 0 ], 0.3 * M, - 0.15 * M ), g.ma( lo[ 2 ], 0.14 * M, - 0.07 * M ) ), 1.0 ) )
		s = dict( alb = alb, rough = g.mul( oG, g.ma( lo[ 1 ], 0.16 * M, 1.0 - 0.08 * M ) ), metal = oB, h = nA, flat = 0.0 )
		s = weather( g, c, went, s, L, wb.lin, wb.TIDE_Y, rain_dark )
		ns = float( params.get( 'bump' ) if params.get( 'bump' ) is not None else 1 )
		if lib.startswith( 'plaster' ): ns *= 1.0 + ( 0.5 - 1.0 ) * L[ 'soft' ]
		nrm = nT
		if ns != 1: nrm = g.va( g.vv( g.va( nrm, np.full( 3, - 0.5 ) ), np.array( [ ns, ns, 1.0 ] ) ), np.full( 3, 0.5 ) )
		nrm = g.vmix( nrm, np.array( [ 0.5, 0.5, 1.0 ] ), s[ 'flat' ] )
		nm = g.new( 'ShaderNodeNormalMap', space = 'TANGENT', uv_map = 'UVMap' )
		g.put( nm.inputs[ 'Color' ], nrm ); nm.inputs[ 'Strength' ].default_value = 1.0
		base = g.vs( s[ 'alb' ], g.mix( 1.0, oR, 0.5 ) ) # the texture's own occlusion at half strength, as the page; the painted (vertex) occlusion is left to the path tracer
		glow = params.get( 'glow' )
		emit = g.vs( s[ 'alb' ], float( glow ) * L[ 'emissive' ] ) if glow else None
		nrm_out = nm.outputs[ 0 ]
		if leafy:

			# leaf cards carry the normals of the crown as a volume (a real-time device: the page shades both faces with
			# them through its own FoliageLighting). A path tracer wants the card as it is: its true normal, half of the
			# light reflected and half passed through, and the crown shades itself
			nrm_out = g.new( 'ShaderNodeNewGeometry' ).outputs[ 'True Normal' ]

		sh = self._principled( g, base, s[ 'rough' ], s[ 'metal' ], nrm_out, emit )
		if foliage:

			tr = g.new( 'ShaderNodeBsdfTranslucent' )
			g.put( tr.inputs[ 0 ], g.vv( base, np.array( [ 1.0, 0.95, 0.6 ] ) ) ); g.nt.links.new( nrm_out, tr.inputs[ 'Normal' ] )
			sh = self._mixsh( g, 0.5 if leafy else 0.15, sh, tr.outputs[ 0 ] )

		if leafy:

			cu = g.ma( g.fr( g.mul( c.uvx, scale / leafy ) ), 2.0, - 1.0 ); cv = g.ma( g.fr( g.mul( c.uvy, scale / leafy ) ), 2.0, - 1.0 )
			d = g.add( g.m( 'SQRT', g.add( g.mul( cu, cu ), g.mul( cv, cv ) ) ), g.ma( hi[ 0 ], 0.3, - 0.15 ) )
			keep = g.gt( nA, g.ma( g.ss( 0.45, 1.0, d ), 0.8, 0.33 ) )
			sh = self._mixsh( g, keep, self._transparent( g ), sh )

		if glow and float( glow ) > 0: sh = self._no_shadow( g, sh )
		return sh

	# ---- MaterialLib._special
	def _special( self, g, name, entry, T, params, baked ):

		wb = self.wb; L = self.L
		d = wb.SPECIAL.get( name, wb.SPECIAL[ 'plain' ] )
		if T is None: T = np.array( [ 0.3, 0.3, 0.3 ] ) if baked else np.asarray( wb.lin( d[ 'tint' ] ), float )
		glow_p = params.get( 'glow' )
		glow = float( params.get( 'glow', d.get( 'glow', 0 ) ) if 'glow' in d else ( glow_p or 0 ) )
		c_rough = float( params.get( 'rough', d.get( 'rough', 0.7 ) ) )
		opacity = float( params.get( 'opacity', d.get( 'opacity', 1 ) ) )
		c = Common( g, self.noise, float( params.get( 'scale' ) or 1 ) )
		lo = c.nLo()
		mx_, my_, mz_, mw_ = c.mask
		EM = L[ 'emissive' ]

		def weathered( alb, rough, metal ):

			s = weather( g, c, entry, dict( alb = alb, rough = rough, metal = metal, h = 0.5, flat = 0.0 ), L, wb.lin, wb.TIDE_Y )
			return s

		if name in ( 'plain', 'metal_plain' ):

			metal0 = float( params.get( 'metal', 1 if name == 'metal_plain' else 0 ) )
			s = weathered( g.vs( T, g.ma( lo[ 0 ], 0.12, 0.94 ) ), g.ma( lo[ 1 ], 0.1, c_rough - 0.05 ), metal0 )
			sh = self._principled( g, s[ 'alb' ], s[ 'rough' ], s[ 'metal' ], None, g.vs( s[ 'alb' ], glow * EM ) if glow else None )
			return self._no_shadow( g, sh ) if glow else sh

		if name in ( 'emissive', 'neon' ):

			e = T * glow * EM
			if name == 'neon': e = e * 0.8 + np.full( 3, glow * EM ) * 0.2
			return self._no_shadow( g, self._emission( g, e ) )

		if name == 'glass_frosted':

			dirt = g.ma( my_, - 0.7 * 0.975, 0.975 )
			sh = self._principled( g, T * 0.8, c_rough, 0.0, None, g.vs( T * ( glow * 0.75 * EM ), dirt ) )
			return self._no_shadow( g, sh )

		if name in ( 'glass_clear', 'glass_lantern' ):

			# real thin glass: straight-through transmission, a Fresnel mirror on top, dirt as a diffuse veil
			lantern = name == 'glass_lantern'
			dirt = g.sat( g.add( g.mul( my_, 0.8 ), g.ma( lo[ 0 ], 0.25, - 0.125 ) ) )
			veil = g.mul( dirt, 0.7 * ( 0.35 if lantern else 1.0 ) )
			tcol = ( 1 - 0.3 ) + 0.3 * np.clip( T / max( T.max(), 1e-3 ), 0, 1 )
			dif = g.new( 'ShaderNodeBsdfDiffuse' ); g.put( dif.inputs[ 0 ], np.array( [ 0.12, 0.11, 0.09 ] ) )
			body = self._mixsh( g, veil, self._transparent( g, tcol ), dif.outputs[ 0 ] )
			gl = g.new( 'ShaderNodeBsdfGlossy' ); g.put( gl.inputs[ 0 ], np.ones( 3 ) ); g.put( gl.inputs[ 'Roughness' ], g.ma( dirt, 0.4, 0.03 ) )
			fr = g.new( 'ShaderNodeFresnel' ); fr.inputs[ 0 ].default_value = 1.5
			sh = self._mixsh( g, fr.outputs[ 0 ], body, gl.outputs[ 0 ] )
			if lantern and glow > 0:

				# the flame's glare in the panes, view-averaged as tools/webbake_bl.py takes it ( alpha * EM of shade_prim )
				e = g.vs( T * np.array( [ 1.0, 0.76, 0.46 ] ) * glow * EM, g.mul( g.ma( dirt, 0.6, 1.05 ), g.ma( lo[ 1 ], 0.5, 0.75 ) ) )
				ad = g.new( 'ShaderNodeAddShader' ); g.nt.links.new( sh, ad.inputs[ 0 ] ); g.nt.links.new( self._emission( g, e ), ad.inputs[ 1 ] )
				sh = ad.outputs[ 0 ]

			return self._no_shadow( g, sh )

		if name == 'glass_stained':

			accent = np.asarray( wb.lin( params.get( 'accent' ) or '#7a1f1a' ), float )
			pane = 0.82 * T * 0.9 + 0.18 * accent
			ps = float( params.get( 'pane' ) or 0.085 )
			a = g.fr( g.mul( g.add( c.uvx, c.uvy ), 1.0 / ps ) ); b = g.fr( g.mul( g.sub( c.uvx, c.uvy ), 1.0 / ps ) )
			lead = g.ss( 0.8, 0.9, g.mul( g.mx( g.ab( g.sub( a, 0.5 ) ), g.ab( g.sub( b, 0.5 ) ) ), 2.0 ) ) # leaded diamond quarries (approximation of the page's lattice)
			e = g.vs( pane * ( 0.85 * glow * L[ 'window_glow' ] ), g.mul( g.ma( lo[ 1 ], 0.6, 0.7 ), g.om( lead ) ) )
			return self._principled( g, g.vs( pane * 0.2 * 0.85, g.ma( lead, - 0.8, 1.0 ) ), g.ma( lead, 0.4, 0.2 ), 0.0, None, e )

		if name in ( 'window_dark', 'window_lit' ):

			dirt = g.sat( g.mul( my_, 0.8 ) )
			base = g.vmix( np.array( [ 0.012, 0.014, 0.018 ] ), np.array( [ 0.07, 0.065, 0.055 ] ), dirt )
			e = None
			if name == 'window_lit':

				# the page's fake lit room without its parallax (the flat room layer of tools/webbake_bl.py window_layer())
				level = g.ma( g.fr( g.mul( mw_, 13.37 ) ), 1.25 - 0.55, 0.55 )
				WC = wb.WIN_CELL
				ru = g.add( g.mul( c.uvx, 1.0 / ( WC[ 0 ] * 2 ) ), g.mul( g.fl( g.mul( mw_, 7.0 ) ), 0.5 ) )
				rv = g.sub( g.mul( c.uvy, 1.0 / ( WC[ 1 ] * 2 ) ), g.mul( g.fl( g.mul( g.fr( g.mul( mw_, 5.3 ) ), 2.0 ) ), 0.5 ) )
				room, _ = g.tex( self.room, ru, rv )
				e = g.vs( g.vv( room, T * ( glow * L[ 'window_glow' ] * wb.WINDOW_GAIN ) ), g.mul( level, g.ma( dirt, - 0.7, 1.0 ) ) )

			return self._principled( g, base, g.ma( dirt, 0.44, 0.06 ), 0.0, None, e )

		if name == 'water':

			# ripples: two octaves of the page's own noise as a bump (the page animates wave trains; the scale is the same order)
			n1 = c.noise( g.mul( c.q0, 1.3 ), g.mul( c.q1, 1.3 ) ); n2 = c.noise( g.mul( c.q0, 3.1 ), g.mul( c.q1, 3.1 ) )
			bp = g.new( 'ShaderNodeBump' ); bp.inputs[ 'Strength' ].default_value = 0.25; bp.inputs[ 'Distance' ].default_value = 0.02
			g.put( bp.inputs[ 'Height' ], g.add( n1[ 0 ], g.mul( n2[ 2 ], 0.5 ) ) )
			if params.get( 'depth' ):

				# a shallow basin: refractive, the bed shows through
				p = g.new( 'ShaderNodeBsdfPrincipled' )
				g.put( p.inputs[ 'Base Color' ], np.clip( 0.8 + T * 2.0, 0, 1 ) ); p.inputs[ 'Roughness' ].default_value = 0.05
				p.inputs[ 'IOR' ].default_value = 1.33; p.inputs[ 'Transmission Weight' ].default_value = 1.0
				g.nt.links.new( bp.outputs[ 0 ], p.inputs[ 'Normal' ] )
				return self._no_shadow( g, p.outputs[ 0 ] )

			p = g.new( 'ShaderNodeBsdfPrincipled' )
			g.put( p.inputs[ 'Base Color' ], T * 0.6 ); p.inputs[ 'Roughness' ].default_value = 0.08; p.inputs[ 'IOR' ].default_value = 1.33
			g.nt.links.new( bp.outputs[ 0 ], p.inputs[ 'Normal' ] )
			return p.outputs[ 0 ]

		if name == 'water_fall':

			# the page's falling water at time 0: flutes, rents, ropes and threads cut from the same noise (mask.z = break-up,
			# mask.w = foam); clear where it is a film (straight-through + a Fresnel mirror), a white diffuser where it foams
			u0 = c.uvx; v = c.uvy
			brk = g.sat( mz_ ); white = g.sat( mw_ )
			u = g.add( u0, g.mul( g.sub( c.noise( g.mul( u0, 0.9 ), g.mul( v, 0.31 ) )[ 3 ], 0.5 ), g.ma( brk, 0.05, 0.012 ) ) )
			s1 = c.noise( g.mul( u, 3.1 ), g.mul( v, 0.23 ) ); s2 = c.noise( g.ma( u, 8.3, 0.37 ), g.mul( v, 0.5 ) ); s3 = c.noise( g.ma( u, 1.3, 0.71 ), g.mul( v, 0.62 ) )
			body = g.add( g.add( g.mul( s1[ 0 ], g.ma( brk, - 0.2, 0.47 ) ), g.mul( s2[ 1 ], 0.25 ) ), g.mul( s3[ 2 ], g.ma( brk, 0.2, 0.28 ) ) )
			thr = g.sub( g.ma( g.mul( brk, g.ma( brk, 0.35, 0.65 ) ), 0.61 - 0.16, 0.16 ), g.mul( white, 0.14 ) )
			a = g.ss( thr, g.add( thr, 0.06 ), body )
			rim = g.mul( a, g.om( g.ss( g.add( thr, 0.05 ), g.add( thr, 0.16 ), body ) ) )
			fib = g.ss( 0.42, 0.7, g.add( g.mul( s2[ 1 ], 0.55 ), g.mul( s1[ 2 ], 0.45 ) ) )
			air = g.mul( brk, g.ma( brk, 0.5, 0.5 ) )
			foam = g.sat( g.add( g.add( g.add( white, g.ma( fib, 0.26, 0.06 ) ), g.mul( air, g.ma( fib, 0.4, 0.36 ) ) ), g.mul( g.mul( rim, brk ), 0.5 ) ) )
			bead = g.mul( g.mul( g.ss( 0.8, 0.92, g.add( g.mul( s2[ 2 ], 0.6 ), g.mul( s1[ 3 ], 0.4 ) ) ), a ), g.ma( brk, 0.8, 0.2 ) )
			white_sh = self._principled( g, g.vs( T, g.ma( foam, 0.8, 0.03 ) ), g.ma( foam, 0.52, 0.08 ), 0.0, None, g.vs( T * ( glow * EM ), bead ) if glow else None, ior = 1.33 )
			gl = g.new( 'ShaderNodeBsdfGlossy' ); g.put( gl.inputs[ 0 ], np.ones( 3 ) ); gl.inputs[ 'Roughness' ].default_value = 0.08
			fr = g.new( 'ShaderNodeFresnel' ); fr.inputs[ 0 ].default_value = 1.33
			clear = self._mixsh( g, fr.outputs[ 0 ], self._transparent( g, 0.7 + 0.3 * T ), gl.outputs[ 0 ] )
			sh = self._mixsh( g, foam, clear, white_sh )
			return self._no_shadow( g, self._mixsh( g, g.mul( a, opacity ), self._transparent( g ), sh ) )

		s = weathered( T, c_rough, 0.0 )
		return self._principled( g, s[ 'alb' ], s[ 'rough' ], s[ 'metal' ] )
