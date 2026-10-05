#!/usr/bin/env python3
# The jump of the picture when a cell changes level, as the bake ships it and with the lower levels' light taken from
# the top level (tools/lsvb_refit.py; here as bundles <level>_pf_v / <level>_pfn_v cut from a fitted pack of the same bake).
#
# The same camera with every cell forced to one level and then to the next one down: the difference of the two
# pictures is what a cell does at that boundary. A level's own shape differs too (the same with either light): read
# shipped against refit, not against 0. The ground's light is one over the levels in both (held), so the views from
# eye height are cut to what is above the paving.
#
#   python3 tools/next/lsvb_jump.py --capture      (the capture Chrome of port 9224 must be up) + the table
#   python3 tools/next/lsvb_jump.py                the table again
#
# high: fine | near | mid | lo, the bake as it is.  low: near | mid | lo with ?lean=0 - the full bake's levels with
# the low tier's distances and near on top; refit there = mid lit from near (what the bake without fine can do).

import sys, os, json, subprocess, argparse
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
import lsvb_sheet as L

TOP = '0,0,1280,446' # what is above the paving, from eye height
VIEWS = [ # name, camera (town frame, z up), metres to what it looks at, the part measured
	( 'wall1', '17.5,-1,4.5,25,1,5,60', 7, TOP ), ( 'wall2', '0,4,3,0,20,5,60', 16, TOP ),
	( 'j20', '0,0,3,0,20,5,60', 20, TOP ), ( 'k20', '5,-3,4.5,25,1,5,60', 20, TOP ),
	( 'j30', '0,-10,3,0,20,5,60', 30, TOP ), ( 'k30', '-5,-5,4.5,25,1,5,60', 30, TOP ),
	( 'j72', '0,-52,20,0,20,5,60', 72, '' ), ( 'k72', '-45,-12,18,25,1,5,60', 72, '' ),
	( 'j180', '0,-160,45,0,20,5,60', 180, '' ) ]
BIG = 'tris=30000000&resident=60000000' # (no budget in the way of a level forced on every cell)
TIERS = dict(
	high = dict( q = 'quality=high', lm = 'near_pf_v,mid_pf_v,lo_pf_v',
		states = dict( F = 'fine=9999&near=9999&mid=9999', N = 'fine=-1&near=9999&mid=9999', M = 'fine=-1&near=-1&mid=9999', L = 'fine=-1&near=-1&mid=-1' ),
		pairs = [ ( 'fine | near', 'F', 'N', 30 ), ( 'near | mid', 'N', 'M', 72 ), ( 'mid | lo', 'M', 'L', 180 ) ] ),
	low = dict( q = 'quality=low&lean=0', lm = 'mid_pfn_v,lo_pfn_v',
		states = dict( N = 'near=9999&mid=9999', M = 'near=-1&mid=9999', L = 'near=-1&mid=-1' ),
		pairs = [ ( 'near | mid', 'N', 'M', 20 ), ( 'mid | lo', 'M', 'L', 64 ) ] ) )
ROWS = dict( lit = '', flat = '&flat=1' )

def cdir( tier, state, var, row ): return 'next/lsvb/jump/%s_%s%s_%s' % ( tier, state, var, row )

def diff( a, b, crop ):

	c = [ '-crop', '%sx%s+%s+%s' % tuple( crop.split( ',' )[ i ] for i in ( 2, 3, 0, 1 ) ), '+repage' ] if crop else []
	base = [ 'magick', '(', a ] + c + [ ')', '(', b ] + c + [ ')', '-compose', 'difference', '-composite', '-colorspace', 'gray' ]
	run = lambda x: float( subprocess.run( base + x, capture_output = True, text = True, check = True ).stdout )
	return run( [ '-format', '%[fx:mean*255]', 'info:' ] ), run( [ '-threshold', '3.14%', '-format', '%[fx:mean*100]', 'info:' ] )

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--capture', action = 'store_true' )
	ap.add_argument( '--tiers', default = 'high,low' )
	ap.add_argument( '--rows', default = 'lit,flat' )
	a = ap.parse_args()
	L.VIEWS[ : ] = [ ( v[ 0 ], v[ 1 ], '' ) for v in VIEWS ]
	for tier in a.tiers.split( ',' ):

		T = TIERS[ tier ]; top = next( iter( T[ 'states' ] ) )
		if a.capture:

			for row in a.rows.split( ',' ):

				for state, force in T[ 'states' ].items():

					for var in ( 's', 'r' ):

						if var == 'r' and state == top: continue # (the top level is the bake's in both)
						q = '&'.join( [ 'pr=1', BIG, force ] + T[ 'q' ].split( '&' )[ 1: ] + ( [ 'lm=' + T[ 'lm' ] ] if var == 'r' else [] ) ) + ROWS[ row ]
						r = L.shoot( cdir( tier, state, var, row ), q, T[ 'q' ].split( '&' )[ 0 ].split( '=' )[ 1 ], 1280, 720 )
						# (quality goes in by shoot(); what else the tier asks for by the query)
						print( 'lsvb_jump: %s %s %.0fs' % ( cdir( tier, state, var, row ), q, r[ 'seconds' ] ), flush = True )

		for row in a.rows.split( ',' ):

			print( '\n%s tier, %s: mean |difference| in 8 bit steps / %% of the pixels over 8 steps   (* the view at about the boundary\'s distance)' % ( tier, row ) )
			print( '%-13s %-12s %-18s %-18s %s' % ( 'boundary', 'view (m)', 'shipped', 'refit', 'refit / shipped' ) )
			for name, s0, s1, dist in T[ 'pairs' ]:

				for v, cam, m, crop in VIEWS:

					f = lambda state, var: os.path.join( L.REN, cdir( tier, state, 's' if state == top else var, row ), v + '.png' )
					if not all( os.path.exists( f( s, var ) ) for s in ( s0, s1 ) for var in ( 's', 'r' ) ): continue
					ds = diff( f( s0, 's' ), f( s1, 's' ), crop ); dr = diff( f( s0, 'r' ), f( s1, 'r' ), crop )
					near = abs( m - dist ) <= 0.25 * dist
					print( '%-13s %-12s %5.2f / %5.1f %%    %5.2f / %5.1f %%    %.2f' % ( name, '%s%s (%d)' % ( '*' if near else ' ', v, m ), ds[ 0 ], ds[ 1 ], dr[ 0 ], dr[ 1 ], dr[ 0 ] / max( ds[ 0 ], 1e-9 ) ) )

if __name__ == '__main__': main()
