// Highlights: the lanterns of the light grid (lightgrid.js) and a little of the moon, mirrored by a surface.
//
// This is all the lighting the page computes; everything diffuse is baked.

import { vec2, vec3, float, int, ivec2, mix, max, dot, sqrt, floor, clamp, saturate, smoothstep, oneMinus, normalize, textureLoad, positionWorld } from 'three/tsl';
import { GLINT_H2, GLINT_FADE } from './lightgrid.js';

// the highlight of a light in direction Lw on a surface with normal N seen from V (times the cosine)
export function brdf( N, V, Lw, rough, f0 ) {

	const a = rough.mul( rough ), a2 = a.mul( a );
	const H = normalize( Lw.add( V ) );
	const nh = saturate( dot( N, H ) ), nl = saturate( dot( N, Lw ) ), lh = saturate( dot( Lw, H ) );
	const den = nh.mul( nh ).mul( a2.sub( 1 ) ).add( 1 );
	const F = oneMinus( lh ).pow( 5 ).mul( oneMinus( f0 ) ).add( f0 );
	return a2.div( den.mul( den ).mul( max( lh.mul( lh ), 0.1 ) ).mul( a.add( 0.5 ) ).mul( 4 * Math.PI ) ).mul( nl ).mul( F );

}

// glints( N, V, rough, f0, own ): specular radiance of the cell's lanterns leaving a surface towards V.
// own (the fine level's variant): how far the baked light says "lit here", 0 .. 1 - the highlights have no
// shadows of their own, so a surface in baked shadow gets none. Except within a pace of a burner: that is the
// lamp's own metalwork - glazing bars, the rim, the roof's underside, black paint that holds next to no baked
// light however hard the mantle shines on it - and nothing stands between the two; the bracket that carries the
// heads is the same case, hence the wide end. There the highlight is whole, and with the bare inverse square
// (U.headSoft 1 would put the soft near field back).
export function createGlints( { stage, bake, grid, U } ) {

	const g = grid, look = bake.look, rf = g.rf;
	const sr2 = look.source_radius ** 2;
	const m = look.moon_col, mk = look.moon * 0.5;
	const moonDir = vec3( stage.moonDir.x, stage.moonDir.y, stage.moonDir.z );

	return ( N, V, rough, f0, own = null ) => {

		const shine = ( Lw ) => brdf( N, V, Lw, rough, f0 );
		const gp = positionWorld.xz.sub( vec2( g.x0, g.z0 ) ).div( g.G );
		const c = clamp( floor( gp ), vec2( 0, 0 ), vec2( g.nx - 1, g.nz - 1 ) );
		const cf = saturate( gp.sub( c ) ); // place within the cell: the threshold is interpolated between its corners
		const ix = int( c.x ).mul( g.W ), iz = int( c.y );
		const T = textureLoad( g.tex, ivec2( ix, iz ) );
		const tau = mix( mix( T.x, T.y, cf.x ), mix( T.z, T.w, cf.x ), cf.y );
		const band = max( tau.mul( GLINT_FADE - 1 ), 1e-6 );
		let sum = vec3( 0 );
		for ( let k = 0; k < g.K; k ++ ) {

			const A = textureLoad( g.tex, ivec2( ix.add( 1 + 2 * k ), iz ) ), B = textureLoad( g.tex, ivec2( ix.add( 2 + 2 * k ), iz ) );
			const Lv = A.xyz.sub( positionWorld );
			// how much this lantern counts here: by its light on the floor at this place
			const h2 = dot( Lv.xz, Lv.xz ).add( GLINT_H2 ), hq = h2.div( A.w.mul( A.w ) ), hw = saturate( oneMinus( hq.mul( hq ) ) );
			const e = dot( B.rgb, vec3( 0.2126, 0.7152, 0.0722 ) ).mul( hw ).mul( hw ).div( h2.mul( h2.mul( rf ).add( 1 ) ) );
			const win = smoothstep( 0, 1, e.sub( tau ).div( band ) );
			const d2 = max( dot( Lv, Lv ), 0.01 ), d = sqrt( d2 );
			const q = d.div( A.w ), w = saturate( oneMinus( q.mul( q ).mul( q ).mul( q ) ) );
			const cap = mix( float( 1 ), B.w, smoothstep( - 0.42, 0.3, Lv.y.negate().div( d ) ) ); // a capped lantern sends less up
			let one = B.rgb.mul( w.mul( w ).div( d2.mul( d2.mul( rf ).add( 1 ) ) ) ).mul( cap ).mul( win ).mul( shine( Lv.div( d ) ) );
			if ( own ) {

				const head = oneMinus( smoothstep( 0.5, 2.0, d2 ) );
				one = one.mul( d2.div( d2.add( head.mul( sr2 ).mul( U.headSoft ) ) ) ).mul( max( own, head ) );

			}

			sum = sum.add( one );

		}

		const moon = vec3( m[ 0 ] * mk, m[ 1 ] * mk, m[ 2 ] * mk ).mul( shine( moonDir ) );
		return sum.add( own ? moon.mul( own ) : moon );

	};

}
