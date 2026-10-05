#!/usr/bin/env python3
# Lightmap prototype, the driver: baked light in texture pages instead of vertex colours on a few cells of the
# branch bake (out/web/_next), to compare the two. Nothing of the bake is changed; all output is out/web/_next/lm/.
#
#   python3 tools/next/lightmap.py --stage dump            the cells' levels in their geometry variants (node)
#   python3 tools/next/lightmap.py --stage sample [--pages] points on the faces of every variant lit on the full model
#                                                          (Blender; reads out/glb and out/town for minutes: holds the
#                                                          BAKING marker); --pages: also the bundles in texture pages
#   python3 tools/next/lightmap.py --stage fit             the fitted vertex light of every bundle, measured (Blender's
#                                                          python for numpy: no full model, no rays)
#   python3 tools/next/lightmap.py --stage bake [--dry]    unwrap + light the texels on the full model (as sample --pages)
#   python3 tools/next/lightmap.py --stage encode          texels -> RGBA8 pages (Blender's python, for numpy)
#   python3 tools/next/lightmap.py --stage pack            the meshes -> bundles (node)
#   python3 tools/next/lightmap.py --stage table           the numbers of the fit stage
#   python3 tools/next/lightmap.py --stage relight         the unsimplified dumped meshes with the raw bake's densest
#                                                          light looked up at their vertices (no ray) -> variant rawrl
#   python3 tools/next/lightmap.py --stage hybrid --pairs mid_std_c:mid_std_s10
#                                                          the faces a fitted bundle leaves over 4 / 8 display steps,
#                                                          by kind of surface, and what texels on only those would do
# The sample stage keeps every mesh's points as it finishes it (.raw/smp.*.npz) and stops between two meshes when the
# file .scratch/lsvb/STOP exists (to let go of the BAKING marker); started again it goes on where it was.
#   --bundles a,b   only these;  --cells c_0_2,...   other cells;  --set KEY=VALUE   into the lm settings
#
# The page reads a bundle with ?lm=<bundle>[,<bundle>...] (web/next/engine/lite.js).

import sys, os, json, subprocess, argparse, shutil, time

HERE = os.path.dirname( os.path.abspath( __file__ ) )
sys.path.insert( 0, HERE )
import webbake as WB

ROOT = WB.ROOT
OUT = os.path.join( ROOT, 'out/web/_next' )
LMD = os.path.join( OUT, 'lm' )

# (a block of 3 x 3 cells about the square: lamp pools on paving, shop fronts, the market's props, roofs)
CELLS = [ 'c_%d_%d' % ( i, j ) for j in ( 0, - 1, - 2 ) for i in ( - 1, 0, 1 ) ]
# geometry variants per level: std = as the packer ships it; h = no colour in the error and the ground free
# (the light is not in the vertices any more); hh = h at a lower triangle budget;
# cw* = the light weighs more in the simplifier's error, at std's triangle count (by the error bound);
# g = the light weighs nothing, at 85 % of std's triangles (the rest is spent where the fit asks for it)
_V = dict( std = {}, h = dict( hard = 1 ), hh = dict( hard = 1, ratio = 0.5, error = 1.5 ), m = dict( match = 'h' ), mm = dict( match = 'hh' ) )
# pf / pfn = what ships: std's meshes with the colours tools/lsvb_refit.py gives them in a fitted pack (pf: a bake with
# fine; pfn: without) - not made by the dump stage (name them in --variants there no more than `raw`): put beside the
# dumps from a fitted pack, judged on std's points (`smp` in their entries of dump.json)
PF = dict( match = 'std' )
VARIANTS = dict( near = dict( _V, rf = dict( src = 'rawrf' ) ),
	lo = dict( std = {}, pf = PF, pfn = PF, rf = dict( src = 'rawrf' ), rfn = dict( src = 'rawrfn' ), cw1 = dict( colour = 0.6, match = 'std', wide = 1 ), cw2 = dict( colour = 2.5, match = 'std', wide = 1 ), g = dict( colour = 0, match = 'std', frac = 0.85 ),
		rl = dict( src = 'rawrl', match = 'std', wide = 1 ), rlc = dict( src = 'rawrl', colour = 2.5, match = 'std', wide = 1 ) ),
	mid = dict( std = {}, pf = PF, pfn = PF, rf = dict( src = 'rawrf' ), rfn = dict( src = 'rawrfn' ), raw = dict( raw = 1 ), cw2 = dict( colour = 2, match = 'std', wide = 1 ), g = dict( colour = 0, match = 'std', frac = 0.85 ),
		rl = dict( src = 'rawrl', match = 'std', wide = 1 ), rlc = dict( src = 'rawrl', colour = 2, match = 'std', wide = 1 ) ) )
# raw = the level as the bake left it, not simplified (mid only: lo is made of the same); rl / rlc = simplified from that
# with the light of the densest raw level looked up at its vertices first (stage relight: no ray) - rl as std, rlc with
# the light weighing more - both at std's triangle count
def B( level, variant, cm, ss, **kw ): return dict( name = '%s_%s_s%d' % ( level, variant, cm ) + kw.pop( 'tag', '' ), level = level, variant = variant, s = cm / 100, ss = ss, roof_h = 6.5, roof_mul = 2, **kw )
# rf = what tools/next/lsvb_refit.py makes of a bake (its trial files, rawrf: the raw level with the top level's light),
# simplified as the packer does it; rfn = the same with near as the top level (a bake without fine: the lean one)
# the light kept in the vertices (tools/next/lsvb_fit.py): a = the mesh as it comes, only measured; the rest fitted to
# the full model's light over the faces -
#   b    the vertices as they are, new values (bg: the ground's too; br / bgr: fitted to the raw bake, no ray)
#   c    a vertex split where its faces want different light (tol display steps); cf: every corner its own (the floor)
#   e    the faces the fit cannot carry cut, worst first (on g: up to std's triangle count; on std: 12 % more)
def F( level, variant, tag, **kw ): return dict( name = '%s_%s_%s' % ( level, variant, tag ), kind = kw.pop( 'kind', 'fit' ), level = level, variant = variant, **kw )
def FITS( level ):

	out = [ F( level, 'std', 'a', kind = 'own' ), F( level, 'std', 'b' ), F( level, 'std', 'br', target = 'raw' ), F( level, 'std', 'bg', ground_fit = 1 ), F( level, 'std', 'bgr', ground_fit = 1, target = 'raw' ) ]
	# (how the fit is made: each against b)
	out += [ F( level, 'std', 'b_a%s' % str( a ).replace( '.', '' ), alpha = a, res = 0 ) for a in ( 0, 0.01, 0.03, 0.3, 1 ) ]
	out += [ F( level, 'std', 'b_paper', reg = 'paper', res = 0 ), F( level, 'std', 'b_range', clamp = 'range', res = 0 ), F( level, 'std', 'b_free', neighbours = 0, res = 0 ) ]
	# (what a sample counts - lsvb_fit.py, fit_values(): w0 = the first fit of this work, which came out too dark)
	# (the same with what no eye reaches left out - vis: 1 = a ray has to go 1 m free)
	out += [ F( level, 'std', 'a_v%d' % j if j >= 0 else 'a_vn', kind = 'own', vis = j, res = 0 ) for j in ( - 1, 0, 1, 2 ) ]
	# (c2: a split only where the light near the vertex differs between its faces - lsvb_fit.py, wish = near)
	out += [ F( level, 'std', 'c2', mode = 'split', wish = 'near', tol = 8 ), F( level, 'std', 'c2_12', mode = 'split', wish = 'near', tol = 12, res = 0 ), F( level, 'std', 'c2_16', mode = 'split', wish = 'near', tol = 16 ) ] + [ F( level, 'std', 'c2_%d' % t, mode = 'split', wish = 'near', tol = t, res = 0 ) for t in ( 24, 32, 48 ) ]
	# (c3: ... by the samples right at the vertex only - two faces of a smooth wall then never differ by the light's slope)
	out += [ F( level, 'std', 'c3_24', mode = 'split', wish = 'near', near_pow = 24, tol = 24, res = 0 ) ]
	# (b5 / c5: the weight between none and the tone curve's - the slope, not its square)
	out += [ F( level, 'std', 'b5', weight = 'face', wpow = 0.5 ), F( level, 'std', 'c5', mode = 'split', tol = 6, weight = 'face', wpow = 0.5 ) ]
	out += [ F( level, 'std', 'b_vnone_m', weight = 'none', mean = 1 ), F( level, 'std', 'b_vface5_m', weight = 'face', wpow = 0.5, mean = 1, res = 0 ), F( level, 'std', 'b_vtone', weight = 'tone', irls = 1, res = 0 ),
		F( level, 'std', 'b_vnone_m0', weight = 'none', mean = 1, vis = 0, res = 0 ), F( level, 'std', 'b_vnone_m2', weight = 'none', mean = 1, vis = 2, res = 0 ) ]
	out += [ F( level, 'std', 'b_w0', weight = 'tone', irls = 1, hidden = 0.1 ), F( level, 'std', 'b_wtone', weight = 'tone', irls = 1, res = 0 ), F( level, 'std', 'b_wtone4', weight = 'tone', irls = 4, res = 0 ),
		F( level, 'std', 'b_wsqrt', weight = 'sqrt', irls = 0, res = 0 ), F( level, 'std', 'b_wsqrt4', weight = 'sqrt', irls = 4, res = 0 ), F( level, 'std', 'b_wnone', weight = 'none', res = 0 ), F( level, 'std', 'b_wnone_h', weight = 'none', hidden = 0.1, res = 0 ),
		F( level, 'std', 'b_wface', weight = 'face', res = 0 ), F( level, 'std', 'b_wface_m', weight = 'face', mean = 1, res = 0 ), F( level, 'std', 'b_wnone_m', weight = 'none', mean = 1, res = 0 ), F( level, 'std', 'b_wtone4_m', weight = 'tone', irls = 4, mean = 1, res = 0 ) ]
	# (the fit to the raw level with no test of what is hidden - fit_vis - 1 - judged as the others are)
	out += [ F( level, 'std', 'br_nm', target = 'raw', fit_vis = - 1, res = 0 ), F( level, 'std', 'b_nm', fit_vis = - 1, res = 0 ) ]
	out += [ F( level, 'std', 'c', mode = 'split', tol = 6 ), F( level, 'std', 'c3', mode = 'split', tol = 3, res = 0 ), F( level, 'std', 'c12', mode = 'split', tol = 12, res = 0 ), F( level, 'std', 'cf', mode = 'split', tol = 0, res = 0 ),
		F( level, 'std', 'cg', mode = 'split', tol = 6, ground_fit = 1 ) ]
	out += [ F( level, 'std', 'e12', refine = 0.12 ), F( level, 'std', 'eg12', refine = 0.12, ground_fit = 1 ), F( level, 'g', 'a', kind = 'own' ), F( level, 'g', 'b', res = 0 ), F( level, 'g', 'e', refine_to = 'std' ), F( level, 'g', 'eg', refine_to = 'std', ground_fit = 1 ), F( level, 'g', 'c', mode = 'split', tol = 6, res = 0 ) ]
	for v in VARIANTS[ level ]:

		if v.startswith( 'cw' ): out += [ F( level, v, 'a', kind = 'own' ), F( level, v, 'b' ), F( level, v, 'bg', ground_fit = 1, res = 0 ) ]
		if v.startswith( 'rl' ): out += [ F( level, v, 'a', kind = 'own' ), F( level, v, 'b' ), F( level, v, 'br', target = 'raw', res = 0 ), F( level, v, 'c', mode = 'split', tol = 6, res = 0 ) ]
		if v == 'raw': out += [ F( level, v, 'a', kind = 'own', res = 0 ) ]
		if v.startswith( 'rf' ): out += [ F( level, v, 'a', kind = 'own' ), F( level, v, 'b', res = 0 ) ]
		if v.startswith( 'pf' ): out += [ F( level, v, 'a', kind = 'own' ) ]

	return out

BUNDLES = [ B( 'lo', 'std', 20, 2 ), B( 'mid', 'std', 10, 1 ), B( 'mid', 'std', 20, 2 ) ] + FITS( 'lo' ) + FITS( 'mid' )

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--stage', required = True )
	ap.add_argument( '--dry', action = 'store_true' )
	ap.add_argument( '--bundles', default = '' )
	ap.add_argument( '--cells', default = '' )
	ap.add_argument( '--levels', default = 'lo,mid' )
	ap.add_argument( '--no-va', action = 'store_true' )
	ap.add_argument( '--jobs', type = int, default = 2 )
	ap.add_argument( '--variants', default = '' )
	ap.add_argument( '--pages', action = 'store_true', help = 'sample: also the bundles in texture pages' )
	ap.add_argument( '--pairs', default = '', help = 'hybrid: vertex bundle:bundle in pages, ...' )
	ap.add_argument( '--set', action = 'append', default = [] )
	ap.add_argument( '--tag', default = '', help = 'fit: added to the names of the bundles (a run over other cells keeps its numbers apart)' )
	a = ap.parse_args()
	cells = a.cells.split( ',' ) if a.cells else CELLS
	levels = a.levels.split( ',' )
	man = json.load( open( os.path.join( OUT, 'manifest.json' ) ) )
	raw = json.load( open( os.path.join( OUT, '.raw/raw.json' ) ) )
	extra = {}
	for kv in a.set:

		k, v = kv.split( '=', 1 )
		extra[ k ] = json.loads( v )

	fitted = lambda b: b.get( 'kind' ) in ( 'own', 'fit' )
	lmk = ( 'range', 'gamma', 'dither', 'radius' )
	bundles = [ dict( b, **{ k: v for k, v in extra.items() if fitted( b ) or k in lmk } ) for b in BUNDLES if b[ 'level' ] in levels and ( not a.bundles or b[ 'name' ] in a.bundles.split( ',' ) ) ]
	if a.cells: bundles = [ dict( b, cells = cells ) for b in bundles ]
	if a.tag: bundles = [ dict( b, name = b[ 'name' ] + a.tag ) for b in bundles ]
	os.makedirs( os.path.join( LMD, '.raw' ), exist_ok = True )
	work = os.path.join( LMD, '.work' ); os.makedirs( work, exist_ok = True )
	env = dict( os.environ, TMPDIR = work, TMP = work, TEMP = work, MAGICK_TMPDIR = work )
	t0 = time.time()

	def node( lm ):

		n = shutil.which( 'node' ) or sys.exit( 'lightmap: node not found' )
		r = subprocess.run( [ n, os.path.join( ROOT, 'tools/next/webbake_pack.mjs' ), OUT, json.dumps( dict( man[ 'pack' ], lm = lm ) ) ], cwd = ROOT, env = env )
		if r.returncode: sys.exit( 'lightmap: the node stage failed' )

	def blender( lm ):

		cfg = json.load( open( os.path.join( OUT, '.config.json' ) ) )
		# (--jobs workers and Blender itself: the machine is shared, four Blender processes at most on it)
		cfg.update( out = os.path.join( LMD, '.blout' ), work = work, lib = 1, jobs = a.jobs, lm = dict( lm, dir = LMD ) )
		path = os.path.join( LMD, '.config.json' )
		json.dump( cfg, open( path, 'w' ) )
		r = subprocess.run( [ WB.find_blender(), '--background', '--factory-startup', '--python', os.path.join( ROOT, 'tools/next/lightmap_bl.py' ), '--', path ], env = env )
		if r.returncode: sys.exit( 'lightmap: the Blender stage failed' )

	def numpy( lm ):

		# the stages that cast no ray need numpy alone: the python Blender ships, run by itself (no Blender process)
		import glob
		py = sorted( glob.glob( os.path.join( os.path.dirname( os.path.realpath( WB.find_blender() ) ), '*/python/bin/python3*' ) ) )
		if not py: return blender( lm )
		cfg = json.load( open( os.path.join( OUT, '.config.json' ) ) )
		cfg.update( out = os.path.join( LMD, '.blout' ), work = work, lib = 1, jobs = 1, lm = dict( lm, dir = LMD ) )
		path = os.path.join( LMD, '.config.%d.json' % os.getpid() )
		json.dump( cfg, open( path, 'w' ) )
		try: r = subprocess.run( [ py[ 0 ], os.path.join( ROOT, 'tools/next/lightmap_bl.py' ), '--', path ], env = env )
		finally: os.remove( path )
		if r.returncode: sys.exit( 'lightmap: the numpy stage failed' )

	if a.stage == 'dump':

		vs = a.variants.split( ',' ) if a.variants else None
		node( dict( stage = 'dump', cells = cells, levels = levels, variants = { l: { k: v for k, v in VARIANTS[ l ].items() if not vs or k in vs } for l in VARIANTS } ) )

	elif a.stage == 'relight':

		blender( dict( mode = 'relight', bundles = [], cells = cells ) )

	elif a.stage == 'bake':

		va = []
		if not a.no_va and not a.dry:

			for key in cells:

				ch = next( c for c in raw[ 'chunks' ] if c[ 'k' ] == key )
				for src, lv in ( ( 'mid', [ l for l in ( 'lo', 'mid' ) if l in levels ] ), ( 'near', [ l for l in ( 'near', ) if l in levels ] ) ):

					if lv and src in ch: va.append( dict( key = key, src = src, levels = lv, path = os.path.join( OUT, '.raw', ch[ src ][ 'raw' ] ), **{ k: ch[ src ][ k ] for k in ( 'nv', 'ni', 'min', 'q', 'sub' ) } ) )

		lm = dict( mode = 'bake', dry = a.dry, bundles = bundles, va = None if a.no_va else va, **{ k: v for k, v in extra.items() if k in ( 'page', ) } )
		if a.dry: blender( lm )
		else:

			# (the full model is read from out/glb: not while the towns are rebuilt, and marked for the other side)
			with WB.bake_lock(): blender( lm )

	elif a.stage == 'sample':

		# the points on the faces of every mesh variant whose light stays in the vertices, lit on the full model
		vs = a.variants.split( ',' ) if a.variants else None
		smp = [ dict( level = l, variant = v, cells = cells, **{ k: x for k, x in extra.items() if k in ( 's', 'rho', 'kmax', 'radius', 'truth_rays' ) } ) for l in levels for v in VARIANTS[ l ] if v in ( vs or [ x for x in VARIANTS[ l ] if x != 'raw' ] ) and v not in ( 'h', 'hh', 'm', 'mm' ) ]
		with WB.bake_lock(): blender( dict( mode = 'bake', bundles = [ b for b in bundles if not fitted( b ) ] if a.pages else [], sample = smp, va = None, stop = os.path.join( ROOT, '.scratch/lsvb/STOP' ) ) )

	elif a.stage == 'fit':

		dump = json.load( open( os.path.join( LMD, '.raw/dump.json' ) ) )[ 'meshes' ]
		fb = []
		for b in bundles:

			if b.get( 'refine_to' ): b = dict( b, refine_to = { m[ 'key' ]: m[ 'ni' ] // 3 for m in dump if m[ 'level' ] == b[ 'level' ] and m[ 'variant' ] == b[ 'refine_to' ] } )
			fb.append( b )

		numpy( dict( mode = 'fit', bundles = fb ) )

	elif a.stage == 'encode':

		# (the bundles in pages need numpy: Blender; the fitted ones are their meta files alone)
		bj = os.path.join( LMD, '.raw/bake.json' )
		paged = [ b for b in bundles if not fitted( b ) and os.path.exists( os.path.join( LMD, '.raw', b[ 'name' ] + '.npz' ) ) ]
		if paged: numpy( dict( mode = 'encode', bundles = paged ) ); out = json.load( open( bj ) )
		else: out = dict( bundles = [], va = [] )
		for b in bundles:

			for name in ( b[ 'name' ], b[ 'name' ] + '_res', b[ 'name' ] + '_sgn' ) if fitted( b ) else ():

				fn = os.path.join( LMD, '.raw', name + '.meta.json' )
				if os.path.exists( fn ): out[ 'bundles' ].append( json.load( open( fn ) ) )

		json.dump( out, open( bj, 'w' ) )
		print( 'lightmap: %d bundles to pack' % len( out[ 'bundles' ] ) )

	elif a.stage == 'hybrid':

		# --pairs vertex:pages,...  (the vertex bundle's faces over 4 / 8 display steps, and with those faces in pages)
		pairs = [ p.split( ':' ) + [ '' ] for p in a.pairs.split( ',' ) if p ]
		numpy( dict( mode = 'hybrid', bundles = [], pairs = [ p[ :2 ] for p in pairs ] ) )
		for row in json.load( open( os.path.join( LMD, '.raw/hybrid.json' ) ) ):

			print( '%s%s' % ( row[ 'vertex' ], ' + pages of %s (%.0f texels a m2 there)' % ( row[ 'pages' ], row.get( 'texels_m2', 0 ) ) if row[ 'pages' ] else '' ) )
			for k, d in row[ 'by' ].items():

				kn, t = k.rsplit( '_', 1 )
				print( '  over %s  %-6s  area %8.1f m2  over: %7.1f m2 (%5.2f %%, %6d faces)  face rms %5.2f' % ( t, kn, d[ 'area' ], d[ 'over_area' ], 100 * d[ 'over_area' ] / max( d[ 'area' ], 1e-9 ), d[ 'over_faces' ], d[ 'rms' ] )
					+ ( '  -> in pages %7.1f m2: rms %5.2f, still over %6.1f m2 (all in pages: %5.2f)%s' % ( d[ 'paged_area' ], d[ 'rms_mixed' ], d[ 'left_area' ], d[ 'rms_pages' ],
						'  ~%d texels' % round( d[ 'paged_area' ] * row[ 'texels_m2' ] ) if row.get( 'texels_m2' ) else '' ) if 'rms_mixed' in d else '' ) )

	elif a.stage == 'study':

		numpy( dict( mode = 'study', bundles = bundles, **extra ) )

	elif a.stage == 'pack':

		node( dict( stage = 'pack', **{ k: v for k, v in extra.items() if k in ( 'va_colour', ) } ) )

	elif a.stage == 'table':

		# what the fit stage measured (display steps of 255; area weighed; the points in the open on the full model)
		print( '%-14s %8s %8s | %6s %6s %6s %6s %6s | %s | %-13s %s | %6s %6s | %5s' % ( 'bundle', 'tris', 'verts', 'mean', 'rms', 'p95', 'p99', '>4 %', ' '.join( '%-13s' % ( k + ' rms/p95' ) for k in ( 'ground', 'wall', 'prop', 'other' ) ),
			'patch rms/p95', ' '.join( '%-6s' % k for k in ( 'ground', 'wall', 'prop', 'other' ) ), 'vs raw', 'ground', 'fit s' ) + ' | bias (display steps; < 0 darker) all wall prop other | sum of linear light / the full model\'s: all wall prop other' )
		for b in bundles:

			fn = os.path.join( LMD, '.raw/stats.%s.json' % b[ 'name' ] )
			if not os.path.exists( fn ): continue
			st = json.load( open( fn ) ); f = st[ 'fit' ]; x = f[ 'all' ]
			kinds = ' '.join( '%5.2f / %5.2f' % ( f[ k ][ 'rms' ], f[ k ][ 'p95' ] ) if f.get( k ) else '      -      ' for k in ( 'ground', 'wall', 'prop', 'other' ) )
			vr = st[ 'vs_raw' ]; fb = st.get( 'fit_blur' ) or {}
			blur = ( '%5.2f / %5.2f ' % ( fb[ 'all' ][ 'rms' ], fb[ 'all' ][ 'p95' ] ) if fb.get( 'all' ) else '      -       ' ) + ' '.join( '%6.2f' % fb[ k ][ 'rms' ] if fb.get( k ) else '     -' for k in ( 'ground', 'wall', 'prop', 'other' ) )
			print( '%-14s %8d %8d | %6.2f %6.2f %6.2f %6.2f %6.2f | %s | %s | %6.2f %6.2f | %5.0f' % ( st[ 'name' ], st[ 'tris' ], st[ 'verts' ], x[ 'mean' ], x[ 'rms' ], x[ 'p95' ], x[ 'p99' ], 100 * x[ 'over4' ], kinds, blur,
				vr[ 'all' ][ 'rms' ] if vr.get( 'all' ) else 0, vr[ 'ground' ][ 'rms' ] if vr.get( 'ground' ) else 0, st[ 'fit_s' ] )
				+ ( ' | ' + ' '.join( '%+6.2f' % f[ k ][ 'bias' ] for k in ( 'all', 'wall', 'prop', 'other' ) ) + ' | ' + ' '.join( '%.3f' % f[ k ][ 'lin' ] for k in ( 'all', 'wall', 'prop', 'other' ) ) if 'bias' in x else '' ) )

	else: sys.exit( 'lightmap: --stage dump | sample | bake | fit | encode | study | pack | table' )
	print( 'lightmap: %s done in %.0fs' % ( a.stage, time.time() - t0 ), flush = True )

if __name__ == '__main__': main()
