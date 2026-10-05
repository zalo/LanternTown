# _TSQ_KEEP (float scalar or normalised integer vertex attribute of a level GLB, 1 = identity vertex: bold clock hands,
# hour marks, lettering) through tools/webbake_bl.py: GLB -> load_glb -> lod_split -> Soup.add -> refine -> encode_chunk,
# where it must come out as bit 1 (must stay) of the chunk's flag column - which tools/webbake_pack.mjs holds at every
# level, the lowest and the ring included. Runs in Blender's bundled python (numpy; no Blender process, no bake):
#   <blender>/4.1/python/bin/python3.11 tools/webbake_keep_test.py [glb with the attribute]
# Without an argument: copies of out/lod_eval/clock_tower_hall/default.lod2.glb (its own mark taken out) with the attribute written into them
# (the top tenth of the model; once as float, once as normalised u8). With one: the same checks on that GLB as it is.
# Work folder .scratch/next/keeptest, removed at the end. Prints 'all passed' or the failed checks (exit code 1).
import sys, os, json, struct, shutil
import numpy as np
ROOT = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
WORK = ROOT + '/.scratch/next/keeptest'
os.makedirs( WORK, exist_ok = True )
real = sys.argv[ 1 ] if len( sys.argv ) > 1 else None
SRC = ROOT + '/out/lod_eval/clock_tower_hall/default.lod2.glb'

def stripped( src, dst ):

	# a copy of the GLB without any keep mark (the models carry one of their own since tk.final.mark_keep(): TEXCOORD_1)
	b = open( src, 'rb' ).read()
	cl = struct.unpack_from( '<I', b, 12 )[ 0 ]
	j = json.loads( b[ 20:20 + cl ] )
	for m in j[ 'meshes' ]:

		for pr in m[ 'primitives' ]:

			for k in ( '_TSQ_KEEP', 'TEXCOORD_1' ): pr[ 'attributes' ].pop( k, None )

	js = json.dumps( j, separators = ( ',', ':' ) ).encode()
	js += b' ' * ( - len( js ) % 4 )
	rest = b[ 20 + cl: ]
	open( dst, 'wb' ).write( struct.pack( '<III', 0x46546C67, 2, 12 + 8 + len( js ) + len( rest ) ) + struct.pack( '<II', len( js ), 0x4E4F534A ) + js + rest )
	return dst

SRC = stripped( SRC, WORK + '/plain.glb' )

def with_keep( src, dst, rule, as_u8 = False ):

	# a copy of the GLB whose primitives carry _TSQ_KEEP (float scalar, or normalised u8) = rule( positions )
	b = open( src, 'rb' ).read()
	cl = struct.unpack_from( '<I', b, 12 )[ 0 ]
	j = json.loads( b[ 20:20 + cl ] )
	bl = struct.unpack_from( '<I', b, 20 + cl )[ 0 ]
	bn = bytearray( b[ 28 + cl:28 + cl + bl ] )
	total = 0
	for m in j[ 'meshes' ]:

		for pr in m[ 'primitives' ]:

			a = j[ 'accessors' ][ pr[ 'attributes' ][ 'POSITION' ] ]; bv = j[ 'bufferViews' ][ a[ 'bufferView' ] ]
			o = bv.get( 'byteOffset', 0 ) + a.get( 'byteOffset', 0 ); st = bv.get( 'byteStride', 0 ) or 12
			P = np.frombuffer( bytes( bn ), np.uint8, st * ( a[ 'count' ] - 1 ) + 12, o )
			P = np.stack( [ np.frombuffer( P[ i * st:i * st + 12 ].tobytes(), np.float32 ) for i in range( a[ 'count' ] ) ] ) if st != 12 else np.frombuffer( P.tobytes(), np.float32 ).reshape( - 1, 3 )
			k = rule( P )
			total += int( k.sum() )
			while len( bn ) % 4: bn.append( 0 )
			data = ( k.astype( np.uint8 ) * 255 ).tobytes() if as_u8 else k.astype( np.float32 ).tobytes()
			j[ 'bufferViews' ].append( dict( buffer = 0, byteOffset = len( bn ), byteLength = len( data ) ) )
			bn += data
			j[ 'accessors' ].append( dict( bufferView = len( j[ 'bufferViews' ] ) - 1, componentType = 5121 if as_u8 else 5126, count = a[ 'count' ], type = 'SCALAR', **( dict( normalized = True ) if as_u8 else {} ) ) )
			pr[ 'attributes' ][ '_TSQ_KEEP' ] = len( j[ 'accessors' ] ) - 1

	while len( bn ) % 4: bn.append( 0 )
	j[ 'buffers' ][ 0 ][ 'byteLength' ] = len( bn )
	js = json.dumps( j, separators = ( ',', ':' ) ).encode()
	js += b' ' * ( - len( js ) % 4 )
	out = struct.pack( '<III', 0x46546C67, 2, 12 + 8 + len( js ) + 8 + len( bn ) ) + struct.pack( '<II', len( js ), 0x4E4F534A ) + js + struct.pack( '<II', len( bn ), 0x004E4942 ) + bytes( bn )
	open( dst, 'wb' ).write( out )
	return total

json.dump( dict( root = ROOT, out = WORK, lib = 1, jobs = 2 ), open( WORK + '/cfg.json', 'w' ) )
sys.argv = [ 'keep_test', '--', WORK + '/cfg.json' ]
sys.path.insert( 0, ROOT + '/tools' )
import webbake_bl as B

fails = []
def check( name, ok, note = '' ):

	print( ( 'ok   ' if ok else 'FAIL ' ) + name + ( ': ' + str( note ) if note != '' else '' ) )
	if not ok: fails.append( name )

# 0. no attribute: no KEEP key in any primitive (today's behaviour), and the flag column of the chunk has no bit from it
plain = B.load_glb( SRC )
check( 'GLB without the attribute: no primitive has KEEP', not any( 'KEEP' in pr for pr in plain ), '%d primitives' % len( plain ) )

if real:

	paths = [ ( real, None ) ]

else:

	# the rule of the stand-in: everything in the top tenth of the model (its spire and finial): float in one copy, normalised u8 in the other
	allP = np.concatenate( [ pr[ 'P' ] for pr in plain ] )
	y0 = float( allP[ :, 1 ].min() + 0.9 * np.ptp( allP[ :, 1 ] ) )
	rule = lambda P: P[ :, 1 ] > y0
	n_f = with_keep( SRC, WORK + '/keep_f32.glb', rule )
	n_u = with_keep( SRC, WORK + '/keep_u8.glb', rule, True )
	paths = [ ( WORK + '/keep_f32.glb', n_f ), ( WORK + '/keep_u8.glb', n_u ) ]

def chunk( prims ):

	# the primitives as one cell of a soup, through refine() and encode_chunk(): -> ( positions (m), flags, soup )
	S = B.Soup()
	for pr in prims:

		if len( pr[ 'I' ] ): S.add( pr, pr[ 'P' ], pr[ 'N' ], 0 )

	S.close()
	n0 = len( S.v[ 'P' ] ); k0 = int( S.v[ 'KEEP' ].sum() )
	B.refine( S, 'mid' )
	C = np.ones( ( len( S.v[ 'P' ] ), 3 ), np.float32 )
	vb, ib, lk, meta, ov = B.encode_chunk( S, C, B.border_lock( S ), np.arange( len( S.t[ 'TRI' ] ) ), None, None, 'c_0_0' )
	W = meta.get( 'w', 10 )
	rec = np.frombuffer( vb, '<u2' ).reshape( - 1, W )
	pos = np.array( meta[ 'min' ] ) + rec[ :, :3 ].astype( np.float64 ) * meta[ 'q' ]
	return pos, np.frombuffer( lk, np.uint8 ), ( n0, k0, len( S.v[ 'P' ] ), int( S.v[ 'KEEP' ].sum() ) ), S

pos0, fl0, st0, S0 = chunk( plain )
check( 'chunk without the attribute: no vertex has bit 1', int( ( fl0 & 1 ).sum() ) == 0, '%d vertices, flags %s' % ( len( fl0 ), sorted( set( fl0.tolist() ) ) ) )

for path, want in paths:

	name = os.path.basename( path )
	prims = B.load_glb( path )
	got = sum( int( pr[ 'KEEP' ].sum() ) for pr in prims if 'KEEP' in pr )
	check( name + ': load_glb reads the attribute', got > 0 and ( want is None or got == want ), '%d vertices set of %d%s' % ( got, sum( len( pr[ 'P' ] ) for pr in prims ), '' if want is None else ' (written: %d)' % want ) )
	kp = np.concatenate( [ pr[ 'P' ][ pr[ 'KEEP' ] ] for pr in prims if 'KEEP' in pr ] )
	lo, hi = kp.min( 0 ) - 1e-4, kp.max( 0 ) + 1e-4
	# lod_split with a lookup that finds nothing: every edge over 1.5 m is split (lmax), the masks interpolate
	t0 = sum( len( pr[ 'I' ] ) for pr in prims )
	added = B.lod_split( prims, lambda Pq, Nq: ( np.zeros( len( Pq ), bool ), np.zeros( ( len( Pq ), 4 ), np.float32 ) ), 0.5, 1.5, 0.1, 3 )
	ok = all( ( 'KEEP' not in pr ) or len( pr[ 'KEEP' ] ) == len( pr[ 'P' ] ) for pr in prims )
	got2 = sum( int( pr[ 'KEEP' ].sum() ) for pr in prims if 'KEEP' in pr )
	kp2 = np.concatenate( [ pr[ 'P' ][ pr[ 'KEEP' ] ] for pr in prims if 'KEEP' in pr ] )
	inside = bool( ( ( kp2 >= lo ) & ( kp2 <= hi ) ).all() )
	check( name + ': lod_split keeps the flag with its vertices', ok and got2 >= got and inside, '%d triangles added to %d, %d vertices set after (%d before), all inside the box of the marked ones: %s' % ( added, t0, got2, got, inside ) )
	pos, fl, st, S = chunk( prims )
	k = ( fl & 1 ) > 0
	# every position marked in the model is flagged in the chunk, and nothing outside their box is
	# (marked vertices that a triangle uses; the chunk's positions are on its grid: within 2 mm)
	mk = np.concatenate( [ pr[ 'P' ][ np.intersect1d( np.nonzero( pr[ 'KEEP' ] )[ 0 ], pr[ 'I' ].ravel() ) ] for pr in prims if 'KEEP' in pr ] ).astype( np.float64 )
	d = np.sqrt( ( ( mk[ :, None, : ] - pos[ k ][ None, :, : ] ) ** 2 ).sum( 2 ) ).min( 1 ) if k.any() else np.full( len( mk ), 9.0 )
	missing = int( ( d > 2e-3 ).sum() )
	# ... and the other way round: no flagged vertex further than 2 mm from a marked one
	d2 = np.sqrt( ( ( pos[ k ][ :, None, : ] - mk[ None, :, : ] ) ** 2 ).sum( 2 ) ).min( 1 ) if k.any() else np.zeros( 0 )
	extra = int( ( d2 > 2e-3 ).sum() )
	out = int( ( ~ ( ( pos[ k ] >= lo - 2e-3 ) & ( pos[ k ] <= hi + 2e-3 ) ).all( 1 ) ).sum() )
	check( name + ': encode_chunk sets bit 1 on them', k.sum() > 0 and missing == 0 and out == 0 and extra == 0, 'soup %d vertices (%d set) -> after refine %d (%d set) -> chunk %d vertices, %d with bit 1; of %d marked vertices in use not flagged: %d; flagged but not marked: %d; flag values %s' % ( st + ( len( fl ), int( k.sum() ), len( mk ), missing, extra, sorted( set( fl.tolist() ) ) ) ) )
	check( name + ': the rest of the chunk is what it is without the attribute', len( fl ) == len( fl0 ) if want is not None and not added else True, '%d vs %d vertices' % ( len( fl ), len( fl0 ) ) )

shutil.rmtree( WORK )
print( 'FAILED: ' + ', '.join( fails ) if fails else 'all passed' )
sys.exit( 1 if fails else 0 )
