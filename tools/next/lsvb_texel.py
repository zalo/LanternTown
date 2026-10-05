# f / g: what light in texels would do on the trial cells, from the ray samples already there (no new ray):
# texels of side s made of the lattice samples (the bake's own noisy rays, a few per texel, as a texel bake has them),
# read back with 2 x 2 filtering at the OTHER set of points (the 6x reference every variant is judged by), stored in
# 8 bits as the prototype's pages are (root of light / 8). A texel here = a cube of side s cut by the surface, per
# facing (dominant axis and its sign): on axis-aligned walls that is a square of s x s; the texels in use are counted.
# Beside it: the level as shipped and with the production fit (std's mesh with the pf colours), same points, and the
# hybrid: texels only on the faces whose vertex light (pf) stays over t display steps.
# <numpy python> tools/next/lsvb_texel.py <level> <s,s,...> [cells]
import sys, os, json, numpy as np
ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) ); LRAW = ROOT + '/out/web/_next/lm/.raw'
lv = sys.argv[ 1 ]; SS = [ float( x ) for x in sys.argv[ 2 ].split( ',' ) ]; cells = sys.argv[ 3 ].split( ',' ) if len( sys.argv ) > 3 else None
dump = json.load( open( LRAW + '/dump.json' ) )[ 'meshes' ]
LUMW = np.array( [ 0.2126, 0.7152, 0.0722 ] )
def tone( c ):
    x = np.maximum( c, 0 ) * ( 1.25 * 0.5 ); a = np.clip( x * ( 2.51 * x + 0.03 ) / ( x * ( 2.43 * x + 0.59 ) + 0.14 ), 0, 1 ); return a ** ( 1 / 2.2 )
def lin13( c16 ): v = np.minimum( 65528, np.round( c16.astype( np.float64 ) / 8 ) * 8 ); return ( v / 65535 ) ** 2 * 64
def load( m ):
    nv, ni = m[ 'nv' ], m[ 'ni' ]; b = np.fromfile( LRAW + '/' + m[ 'raw' ], np.uint8 )
    return b[ : nv * 20 ].view( '<u2' ).reshape( nv, 10 ), b[ nv * 20: nv * 20 + ni * 4 ].view( '<u4' ).reshape( - 1, 3 ).astype( np.int64 ), b[ nv * 20 + ni * 4: nv * 20 + ni * 4 + nv ]
FACE = bool( os.environ.get( 'LM_FACE' ) ) # (a face's texels its own: no two surfaces in one texel, no texel over an edge - the bound from below)
def chart( n, X, s, f = None ):
    # per point: its facing and the cube it is in -> the facing + slice (one number), the place in the slice in texels
    a = np.abs( n ).argmax( 1 ); sg = ( n[ np.arange( len( n ) ), a ] > 0 ).astype( np.int64 )
    Y = X / s + 0.37
    sl = np.floor( Y[ np.arange( len( n ) ), a ] ).astype( np.int64 )
    u = Y[ np.arange( len( n ) ), ( a + 1 ) % 3 ]; v = Y[ np.arange( len( n ) ), ( a + 2 ) % 3 ]
    return ( f if FACE else ( a * 2 + sg ) + 8 * ( sl + ( 1 << 15 ) ) ), u, v
def tkey( c, iu, iv ): return ( c << 34 ) | ( ( iu + ( 1 << 16 ) ) << 17 ) | ( iv + ( 1 << 16 ) )
names = [ 'shipped', 'fit' ] + [ 'tex%g' % ( s * 100 ) for s in SS ]
T = { n: dict( e = [], g = [] ) for n in names }; W = []; K = []; FE = { n: [] for n in names }; FA = []; FK = []
texels = { s: 0 for s in SS }; area = 0.0; nofind = { s: 0.0 for s in SS }; clipped = { s: [ 0, 0 ] for s in SS }
for m in [ m for m in dump if m[ 'variant' ] == 'std' and m[ 'level' ] == lv ]:
    k = m[ 'key' ]; mp = next( ( x for x in dump if x[ 'variant' ] == 'pf' and x[ 'level' ] == lv and x[ 'key' ] == k ), None )
    if ( cells and k not in cells ) or mp is None: continue
    rec, tri, fl = load( m ); prec = load( mp )[ 0 ]; assert len( prec ) == len( rec )
    S = np.load( LRAW + '/smp.%s.%s.std.npz' % ( k, lv ) ); assert int( S[ 'nt' ] ) == len( tri )
    P = np.array( m[ 'min' ] )[ None ] + rec[ :, :3 ].astype( np.float64 ) * m[ 'q' ]
    fn = np.cross( P[ tri[ :, 1 ] ] - P[ tri[ :, 0 ] ], P[ tri[ :, 2 ] ] - P[ tri[ :, 0 ] ] ); fn /= np.maximum( np.linalg.norm( fn, axis = 1 ), 1e-20 )[ :, None ]
    ef = S[ 'eface' ].astype( np.int64 ); tf = tri[ ef ]; fk = fl[ tf ]; nn = fn[ ef ]
    kind = np.where( ( ( fk & 4 ) > 0 ).all( 1 ), 0, np.where( ( ( fk & 2 ) > 0 ).all( 1 ), 2, np.where( np.abs( nn[ :, 1 ] ) < 0.5, 1, 3 ) ) )
    ok = S[ 'eq' ] > 0.5; eb = S[ 'eb' ].astype( np.float64 ); y = S[ 'ey' ][ :, :3 ].astype( np.float64 ); ew = S[ 'ew' ].astype( np.float64 )
    zv = np.load( LRAW + '/vis.%s.%s.std.npz' % ( k, lv ) ); ok = ok & ~ ( ( zv[ 'ev' ][ :, 1 ] == 0 ) & ( kind != 0 ) )
    G = {}
    for name, rr in ( ( 'shipped', rec ), ( 'fit', prec ) ):
        lit = lin13( rr[ :, 6:9 ] ); G[ name ] = lit[ tf[ :, 0 ] ] * eb[ :, 0:1 ] + lit[ tf[ :, 1 ] ] * eb[ :, 1:2 ] + lit[ tf[ :, 2 ] ] * eb[ :, 2:3 ]
    lf = S[ 'face' ].astype( np.int64 ); lX = S[ 'X' ].astype( np.float64 ); ly = S[ 'y' ][ :, :3 ].astype( np.float64 ); lq = S[ 'q' ].astype( np.float64 )
    lq = np.where( ( zv[ 'v' ][ :, 1 ] == 0 ) & ( lq > 0.5 ), 0.2, lq )
    eX = S[ 'eX' ].astype( np.float64 )
    for s in SS:
        c, u, v = chart( fn[ lf ], lX, s, lf )
        tk, inv = np.unique( tkey( c, np.floor( u ).astype( np.int64 ), np.floor( v ).astype( np.int64 ) ), return_inverse = True )
        wsum = np.bincount( inv, lq, len( tk ) )
        val = np.stack( [ np.bincount( inv, lq * ly[ :, j ], len( tk ) ) for j in range( 3 ) ], 1 ) / np.maximum( wsum, 1e-30 )[ :, None ]
        clipped[ s ][ 0 ] += int( ( val.max( 1 ) > 8 ).sum() ); clipped[ s ][ 1 ] += len( val )
        val = ( np.round( np.sqrt( np.clip( val / 8, 0, 1 ) ) * 255 ) / 255 ) ** 2 * 8
        texels[ s ] += len( tk )
        c, u, v = chart( nn, eX, s, ef ); u -= 0.5; v -= 0.5
        iu = np.floor( u ).astype( np.int64 ); iv = np.floor( v ).astype( np.int64 ); fu = u - iu; fv = v - iv
        g = np.zeros( ( len( c ), 3 ) ); ws = np.zeros( len( c ) )
        for du, dv, w in ( ( 0, 0, ( 1 - fu ) * ( 1 - fv ) ), ( 1, 0, fu * ( 1 - fv ) ), ( 0, 1, ( 1 - fu ) * fv ), ( 1, 1, fu * fv ) ):
            kk = tkey( c, iu + du, iv + dv ); p = np.minimum( np.searchsorted( tk, kk ), len( tk ) - 1 ); hit = tk[ p ] == kk
            g += np.where( hit, w, 0 )[ :, None ] * val[ p ]; ws += np.where( hit, w, 0 )
        miss = ws < 1e-6
        g = np.where( miss[ :, None ], G[ 'fit' ], g / np.maximum( ws, 1e-6 )[ :, None ] )
        nofind[ s ] += float( ew[ miss & ok ].sum() )
        G[ 'tex%g' % ( s * 100 ) ] = g
    lit_f = np.unique( ef[ ok ] ); fi = np.full( len( tri ), - 1, np.int64 ); fi[ lit_f ] = np.arange( len( lit_f ) )
    fa = np.bincount( fi[ ef[ ok ] ], ew[ ok ], len( lit_f ) ); FA.append( fa ); FK.append( np.bincount( fi[ ef[ ok ] ], kind[ ok ], len( lit_f ) ) / np.maximum( np.bincount( fi[ ef[ ok ] ], minlength = len( lit_f ) ), 1 ) )
    for name in names:
        g = G[ name ]; e = np.abs( tone( g ) - tone( y ) ).max( 1 ) * 255
        T[ name ][ 'e' ].append( e[ ok ] ); T[ name ][ 'g' ].append( ( ( tone( g ) - tone( y ) ) @ LUMW * 255 )[ ok ] )
        FE[ name ].append( np.sqrt( np.bincount( fi[ ef[ ok ] ], ( e * e * ew )[ ok ], len( lit_f ) ) / np.maximum( fa, 1e-30 ) ) )
    W.append( ew[ ok ] ); K.append( kind[ ok ] ); area += float( ew.sum() )
W = np.concatenate( W ); K = np.concatenate( K ); FA = np.concatenate( FA ); FK = np.round( np.concatenate( FK ) ).astype( np.int64 )
SEL = ( ( 'all', np.ones( len( W ), bool ) ), ( 'wall', K == 1 ), ( 'prop', K == 2 ), ( 'other', K == 3 ), ( 'ground', K == 0 ) )
print( '%s, %s: lit area %.0f m2, judged (seen) %.0f m2; per kind: mean rms p95 bias' % ( lv, ','.join( cells ) if cells else 'the 9 trial cells', area, W.sum() ) )
for name in names:
    e = np.concatenate( T[ name ][ 'e' ] ); sg = np.concatenate( T[ name ][ 'g' ] ); row = []
    for label, sel in SEL:
        ee, ww = e[ sel ], W[ sel ]; ws = ww.sum(); o = np.argsort( ee ); cw = np.cumsum( ww[ o ] )
        row.append( '%s %5.1f %5.1f %5.1f %+5.1f' % ( label[ :4 ], ( ee * ww ).sum() / ws, np.sqrt( ( ee * ee * ww ).sum() / ws ), ee[ o ][ min( len( ee ) - 1, np.searchsorted( cw, 0.95 * ws ) ) ], ( sg[ sel ] * ww ).sum() / ws ) )
    print( '%-8s | %s' % ( name, ' | '.join( row ) ) )
for s in SS:
    print( 'tex%g: %d texels in use = %.1f a m2 of lit area (a square grid of this side: %.1f); %.2f %% of the judged area found no texel (kept the fit); %.3f %% of the texels over the 8 bit range' % (
        s * 100, texels[ s ], texels[ s ] / area, 1 / ( s * s ), 100 * nofind[ s ] / W.sum(), 100 * clipped[ s ][ 0 ] / max( 1, clipped[ s ][ 1 ] ) ) )
# the hybrid: the fit's faces over t steps (rms on the face) take the texels where those are better
fe = np.concatenate( FE[ 'fit' ] ); rms = lambda e, w: np.sqrt( ( e * e * w ).sum() / max( w.sum(), 1e-30 ) )
for s in SS:
    ft = np.concatenate( FE[ 'tex%g' % ( s * 100 ) ] )
    for label, sel in ( ( 'all', np.ones( len( fe ), bool ) ), ( 'wall', FK == 1 ) ):
        for t in ( 8, 16, 24 ):
            use = sel & ( fe > t ) & ( ft < fe ); mix = np.where( use, ft, fe )
            print( 'hybrid tex%g %-4s over %2d: %5.1f %% of the area over (%6d faces); in texels %7.1f m2 (%4.1f %%, ~%d texels); face rms %5.2f -> %5.2f (all in texels %5.2f); still over %4.1f %%' % (
                s * 100, label, t, 100 * FA[ sel & ( fe > t ) ].sum() / FA[ sel ].sum(), int( ( sel & ( fe > t ) ).sum() ), FA[ use ].sum(), 100 * FA[ use ].sum() / FA[ sel ].sum(), round( FA[ use ].sum() * texels[ s ] / area ),
                rms( fe[ sel ], FA[ sel ] ), rms( mix[ sel ], FA[ sel ] ), rms( ft[ sel ], FA[ sel ] ), 100 * FA[ sel & ( mix > t ) ].sum() / FA[ sel ].sum() ) )
