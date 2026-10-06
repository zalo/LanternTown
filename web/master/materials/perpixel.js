// Light per pixel: what only the fine level has.
//
// The baked light is one colour per vertex: right for the face, blind to the relief on it. The fine level's
// vertices bring 4 bytes more (`_g`): g, the direction their direct light comes from over the cosine it arrives
// at, and r, the share of the baked colour that is direct light (red, blue). The light on the pixel's own
// normal is then the baked one x ( 1 + ( N - face normal ) . g ) for the direct share, and the sky's fill for
// the rest, which only cares how far the normal looks up. A joint turned away from the lantern goes dark and
// cool, the flank of a cobble towards it bright and warm.
//
// The pixel's normal is the material's own normal map where the layer has one (the grain of a door, the pores
// of a brick, rust on a lamp post), with its roughness - and with them a highlight from the lanterns on every
// surface, not only the glossy ones. The tangent frame comes from the derivatives of the uv. Wet paving keeps
// the relief of its height instead: the film and the mirror are built on it.
//
// All of this is a matter of the pixel's distance, not of its cell's level: it has faded to the picture every
// level draws before the nearest cell that is not at this level (U.fineR, set by levels.js). A cell given or
// refused the level then changes nothing but its mesh.

import { attribute, varying, texture, vec2, vec3, float, int, mix, step, max, dot, sqrt, floor, clamp, length, smoothstep, oneMinus, dFdx, dFdy, cameraPosition, positionWorld } from 'three/tsl';
import { unit } from './frame.js';
import { perLayer } from './layers.js';

// N: the relief normal; Nf: the face normal; both turned to the eye. Returns the pixel's normal, roughness and
// f0, `spec` (1 where the layer's own maps are in use) and `light`, the factor on the baked colour.
export function perPixelLight( ctx, { frame, N, Nf, gloss, rough, wetK } ) {

	const { bake, U, NR, tx } = ctx;
	const { layer, tiles } = ctx.inputs;
	const { R1, R2 } = frame;
	const kd = oneMinus( smoothstep( U.fineR.mul( 0.6 ), U.fineR.add( 0.01 ), length( cameraPosition.xz.sub( positionWorld.xz ) ) ) );

	// --- the material's normal map and roughness
	const li = floor( layer.add( 0.5 ) );
	const has = step( NR.n0, li ).mul( oneMinus( step( NR.n1, li ) ) ).mul( NR.on ).mul( oneMinus( wetK ) ).mul( kd );
	const nq = texture( NR.blank, tiles ).depth( int( clamp( li.sub( NR.n0 ), 0, NR.count - 1 ) ) );
	NR.nodes.push( nq );
	const mx = nq.x.mul( 2 ).sub( 1 ).mul( U.nmap ), my = nq.y.mul( 2 ).sub( 1 ).mul( U.nmap ).mul( U.nflip ), mz = sqrt( max( oneMinus( mx.mul( mx ) ).sub( my.mul( my ) ), 0.05 ) );
	const du = dFdx( tiles ), dv = dFdy( tiles );
	const Tu = R1.mul( du.x ).add( R2.mul( dv.x ) ), Tv = R1.mul( du.y ).add( R2.mul( dv.y ) );
	const sc = float( 1 ).div( sqrt( max( max( dot( Tu, Tu ), dot( Tv, Tv ) ), 1e-30 ) ) );
	const nm = Tu.mul( mx.mul( sc ) ).add( Tv.mul( my.mul( sc ) ) ).add( Nf.mul( mz ) );
	const Np = unit( mix( N, unit( nm ), has ) );
	const mean = perLayer( bake.rough, layer );
	const roughP = clamp( mix( rough, mix( max( mean, 0.5 ), float( 0.08 ), gloss ).add( nq.z.sub( mean ) ), has ), 0.04, 1 );
	const f0 = mix( vec3( 0.04 ), tx.rgb, perLayer( bake.metal, layer ).mul( has ) );

	// --- the baked light on that normal
	const ga = attribute( '_g', 'vec2' ).mul( 65535 );
	const hi = floor( ga.add( 0.5 ).div( 256 ) ), lo = ga.sub( hi.mul( 256 ) );
	const gv = varying( vec3( lo.x, hi.x, lo.y ).sub( 128 ).mul( bake.bake.dir_max / 127 ), 'vDirG' );
	const rr = floor( hi.y.add( 0.5 ).div( 16 ) );
	const rv = varying( vec2( rr, hi.y.sub( rr.mul( 16 ) ) ).div( 15 ), 'vDirR' );
	const dn = Np.sub( Nf );
	const r3 = vec3( rv.x, mix( rv.x, rv.y, 0.3 ), rv.y );
	const sk = bake.look.sky, gd = bake.look.ground;
	const fill = mix( vec3( gd[ 0 ], gd[ 1 ], gd[ 2 ] ), vec3( sk[ 0 ], sk[ 1 ], sk[ 2 ] ), Nf.y.mul( 0.5 ).add( 0.5 ) );
	const ka = vec3( ( sk[ 0 ] - gd[ 0 ] ) / 2, ( sk[ 1 ] - gd[ 1 ] ) / 2, ( sk[ 2 ] - gd[ 2 ] ) / 2 ).div( fill );
	const light = clamp( dot( dn, gv ).mul( r3 ).add( dn.y.mul( ka ).mul( oneMinus( r3 ) ) ).mul( U.dir.mul( kd ) ).add( 1 ), 0, 3 );

	return { N: Np, rough: roughP, f0, spec: has, light };

}
