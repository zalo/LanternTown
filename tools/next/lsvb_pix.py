#!/usr/bin/env python3
# The picture's own measure for tools/next/lsvb_sheet.py: every variant's capture against the full-detail capture
# from the same camera (the `ref` column), in 8 bit steps of the page as it is shown -
#   d     the mean absolute difference of the pixels (grey)
#   bias  the variant's mean minus the reference's (< 0: the picture is darker)
# A level's own geometry differs from the full model's too (the same for every variant of one level): read the
# columns against each other, not against 0. Sky and whatever else is equal in both only thins the numbers out -
# --crop x,y,w,h (in pixels of the capture) takes a part.
#
#   python3 tools/next/lsvb_pix.py --level mid --cols a,b,c --row flat --views wall1,wall2 [--crop 0,200,1280,400]

import sys, os, subprocess, argparse

ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
CAP = os.path.join( ROOT, 'out/renders/_town/next/lsvb' )

def measure( ref, img, crop ):

	c = [ '-crop', '%sx%s+%s+%s' % ( crop[ 2 ], crop[ 3 ], crop[ 0 ], crop[ 1 ] ), '+repage' ] if crop else []
	d = float( subprocess.run( [ 'magick', '(', ref ] + c + [ ')', '(', img ] + c + [ ')', '-compose', 'difference', '-composite', '-colorspace', 'gray', '-format', '%[fx:mean*255]', 'info:' ],
		capture_output = True, text = True, check = True ).stdout )
	m = [ float( subprocess.run( [ 'magick', x ] + c + [ '-colorspace', 'gray', '-format', '%[fx:mean*255]', 'info:' ], capture_output = True, text = True, check = True ).stdout ) for x in ( ref, img ) ]
	return d, m[ 1 ] - m[ 0 ]

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--level', default = 'mid' )
	ap.add_argument( '--cols', default = 'a,b,c' )
	ap.add_argument( '--row', default = 'flat' )
	ap.add_argument( '--views', default = 'pool,facade,props,roofs,wall1,wall2' )
	ap.add_argument( '--crop', default = '' )
	ap.add_argument( '--ref', default = 'ref' )
	a = ap.parse_args()
	crop = [ int( v ) for v in a.crop.split( ',' ) ] if a.crop else None
	views = a.views.split( ',' ); cols = a.cols.split( ',' )
	print( '%s, %s captures against %s%s: mean |d| / bias (8 bit steps)' % ( a.level, a.row, a.ref, ', crop %s' % a.crop if crop else '' ) )
	print( '%-10s' % '' + ''.join( '%-16s' % v for v in views ) + 'all' )
	for col in cols:

		row = []; line = '%-10s' % col
		for v in views:

			ref = os.path.join( CAP, a.level, '%s_%s' % ( a.ref, a.row ), v + '.png' ); img = os.path.join( CAP, a.level, '%s_%s' % ( col, a.row ), v + '.png' )
			if not ( os.path.exists( ref ) and os.path.exists( img ) ): line += '%-16s' % '-'; continue
			d, b = measure( ref, img, crop ); row.append( ( d, b ) )
			line += '%-16s' % ( '%5.2f / %+5.2f' % ( d, b ) )

		if row: line += '%5.2f / %+5.2f' % ( sum( r[ 0 ] for r in row ) / len( row ), sum( r[ 1 ] for r in row ) / len( row ) )
		print( line )

if __name__ == '__main__': main()
