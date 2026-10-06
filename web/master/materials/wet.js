// The town just after rain, per pixel - and what of it the mirror pass is told.
//
// The bake has the rain per vertex, with the relief at its mean: which stone is wet, how dark, how glossy on
// average - not where the water stands. Here, on level wet paving, the joints flood first (puddle), a film
// stands over the stones themselves (sheet), and low zones a few metres across hold pools whose edge follows
// the relief (water). Standing water is one smooth surface: the relief normal gives way to the face normal
// there, so a lantern is mirrored as one long streak across many cobbles instead of a speck on each, and its
// roughness is the water's, not the stone's.
//
// Water lies level: on a facet within a few degrees of level - the top of a modelled sett, whatever is left of
// it at a lower level, flat paving alike - the film's normal is the vertical, so the mirror image and the
// streaks on a paved place are the same at every level of its cell and a faceted sett is not a faceted mirror.
//
// The mirror image (stage.mirrorOut: strength, roughness and normal in a second colour target, which mirror.js
// turns into the reflected town). Standing water mirrors along the face normal and its ripples; the wet crown
// of a cobble along its own relief - every stone picks its own piece of the town, and a lit doorway comes out
// as a streak of lit crowns towards the viewer. Strength: the film's fresnel at the mean normal (per pixel it
// would stipple), less where the stone has drained, least in baked shadow.
//
// What the mirror is cut from - which pixel is joint, which crown, which way a crown tilts - are steps and a
// normal taken of the relief height, and a step of a height that is only filtered across the pixel's short side
// (the anisotropic filter keeps receding paving sharp) is a line a pixel thin wherever the courses of a brick
// floor get close to the pixel rows: the joints of a street stood out as a moire of bright and dark rows and
// crawled with every step. The mirror therefore reads the height once more, filtered over the pixel's long side
// in both directions (U.mband times it): up close that is the same texel, far off the mean of the course.

import { texture, vec2, vec3, float, mix, max, min, pow, dot, length, saturate, smoothstep, oneMinus, dFdx, dFdy, time, positionWorld } from 'three/tsl';
import { unit } from './frame.js';

// N: the pixel's normal; Nf: the face normal turned to the eye; wetK: how far this is wet, level paving;
// lum: the luminance of its baked light. Returns the film's normal and roughness, `darken` (a factor on the
// colour), fresnel( n.v ) and the node for the mirror target.
export function wetFilm( ctx, { frame, N, Nf, rough, lum, wetK } ) {

	const { stage, bake, U, tx, sample, relief, GAO } = ctx;
	const { tiles, V } = ctx.inputs;
	const look = bake.look, amt = look.wet;

	const lvl = smoothstep( 0.95, 0.985, Nf.y ).mul( U.level );
	const Ng = mix( Nf, vec3( 0, 1, 0 ), lvl ).normalize(); // the film on the face
	const tip = Ng.sub( Nf ).mul( wetK ); // (the relief stands on the film's plane, not the facet's)
	const wp = positionWorld;
	const q = vec2( wp.x.add( wp.y.mul( 0.37 ) ), wp.z.add( wp.y.mul( 0.61 ) ) );
	const nLo = texture( stage.noise, q.mul( 0.11 ) ), nHi = texture( stage.noise, q.mul( 0.93 ).add( vec2( 0.31, 0.77 ) ) );
	const nP = texture( stage.noise, q.mul( 0.031 ).add( vec2( 0.53, 0.17 ) ) );
	const sheet = wetK.mul( U.wetFlat ).mul( smoothstep( 0.25, 0.7, nLo.b.mul( 0.7 ).add( nHi.a.mul( 0.3 ) ).add( amt - 0.5 ) ).mul( 0.5 ).add( 0.5 ) );
	const zone = min( nP.r.add( nP.b.mul( 0.35 ) ).sub( 0.8 ).mul( 5 ).add( amt - 0.5 ), look.pool_cap );
	const low = zone.add( nLo.a.sub( 0.5 ).mul( 2.4 ) ).add( nHi.a.sub( 0.5 ).mul( 0.3 ) ).sub( 0.25 ); // how deep a pool stands over a stone of mean height
	const tt = time.mul( 0.03 );
	const r1 = texture( stage.noise, q.mul( 1.3 ).add( vec2( tt, tt.mul( 0.7 ) ) ) ).rg.sub( 0.5 );
	const r2 = texture( stage.noise, q.mul( 3.1 ).add( vec2( tt.mul( - 1.3 ), tt.mul( 0.9 ) ) ) ).ba.sub( 0.5 );

	// where the water stands, given the relief height (a layer's alpha holds it in its upper half)
	const standing = ( alpha ) => {

		const hgt = alpha.mul( 2 ).sub( 1 );
		const puddle = wetK.mul( smoothstep( 0.55, 0.2, hgt.add( ( 1 - amt ) * 0.4 ) ) );
		const water = smoothstep( 0.0, 0.5, low.sub( hgt.mul( 0.8 ) ) ).mul( smoothstep( 0.985, 0.998, Nf.y ) ).mul( wetK );
		return {
			hgt, puddle, water,
			flat: max( max( puddle.mul( 0.95 ), sheet.mul( 0.9 ) ), water ),
			rough: mix( mix( mix( rough, min( rough, float( look.film_rough ) ), sheet ), float( 0.08 ), puddle ), float( look.puddle_rough ), water ),
			ripple: r1.add( r2.mul( 0.5 ) ).mul( U.ripple ).mul( max( water, sheet.mul( 0.5 ) ) ),
		};

	};

	// --- as the eye sees it
	const seen = standing( tx.a );
	const Nw = unit( mix( N.add( tip ).normalize(), Ng, seen.flat ).add( vec3( seen.ripple.x, 0, seen.ripple.y ) ) );
	// wet stone is darker where the water stands than where a cobble's crown has drained (mean kept: the bake's
	// wet darkening is the average of the two)
	const darken = mix( float( 1 ), mix( U.crown, U.sunk, max( seen.puddle, seen.water ) ), wetK );
	// (steeper than Schlick: a thin film over a rough base keeps its long streaks)
	const fresnel = ( nv ) => {

		const s2 = oneMinus( nv.mul( nv ) );
		return mix( pow( oneMinus( nv ), 5 ), s2.mul( s2 ), wetK ).mul( 0.96 ).add( 0.04 );

	};

	// --- as the mirror takes it: the same from the height filtered over the pixel's long side
	const gm = max( length( dFdx( tiles ) ), length( dFdy( tiles ) ) ).mul( U.mband );
	const txm = sample( tiles, [ vec2( gm, 0 ), vec2( 0, gm ) ] );
	const m = standing( txm.a );
	const Nm = unit( mix( unit( frame.bump( txm.a.mul( relief ) ) ).add( tip ).normalize(), Ng, max( max( m.puddle, m.water ), sheet.mul( U.mflat ) ) ).add( vec3( m.ripple.x, 0, m.ripple.y ) ) );
	const nvm = saturate( dot( Ng, V ) ), s2 = oneMinus( nvm.mul( nvm ) );
	const shade = mix( U.mshade, float( 1 ), smoothstep( 0.004, 0.05, lum ) );
	// (a joint is a crevice: seen from eye height it mirrors next to nothing unless a pool stands over it)
	const proud = mix( U.joint, float( 1 ), max( smoothstep( 0.2, 0.6, m.hgt ), m.water ) );
	const strength = s2.mul( s2 ).mul( 0.96 ).add( 0.04 ).mul( wetK ).mul( mix( U.mdry, float( 1 ), m.flat ) ).mul( shade ).mul( proud ).mul( oneMinus( stage.fogAmount() ) ).mul( U.mirror ).mul( GAO.fm );

	return { N: Nw, rough: seen.rough, darken, fresnel, mrtNode: stage.mirrorOut( strength, m.rough, Nm ) };

}
