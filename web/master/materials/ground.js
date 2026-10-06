// The ground's maps: contact occlusion and the lanterns' shadows, one picture of the town seen from above.
//
// The baked light is one number per vertex and the ground has a vertex every 0.6 m at best: the dark rim where
// a barrel, a bench or a wall stands on the paving falls between them, and a shadow can only smear. The picture
// (bake.gao + bake.gsh, read at the pixel's place in the world: no geometry, no draw, nothing to sort) holds
//   red    the open share of the hemisphere within a metre of the floor,
//   green  the terrain's height (what faces up within a hand of it is ground; a table top above it is not),
//   blue   sqrt of v, the share of the lanterns' light that arrives at the texel,
//   alpha  sqrt of f, the share of its light that is no lantern's (moon, bounce, sky).
// The ground's vertices hold the lanterns' light WITHOUT shadows (smooth, the same at every level) and a pixel
// takes the shadow from the picture, sharp to 6 cm:
//     x ( f cF + ( 1 - f ) v cD ) / ( f cF + ( 1 - f ) ( v + w ( 1 - v ) ) cD )
// cF, cD: the hues of fill and lanterns (town-wide, luminance 1); w: how much of the pixel is ground.
//
// That measured share alone does not show: many lamps fill each other's shadows, and wet paving at night is
// glints and mirror images, which a factor on the diffuse light leaves alone. So a shadow is deepened where it
// is LOCAL: against vb, the same v from a coarse level of the picture (texels of about `span` metres),
//     c = clamp( v / ( bias vb ), 0, 1 ) ^ pow
// is 1 on open ground, whatever its v, and falls under what stands close by. v c replaces v above, and c also
// gates the glints (fg), the mirror (fm) and a share of the fill (`all`).
//
// The picture arrives late: `node` samples a blank until textures.js sets its value and `on`.

import * as THREE from 'three';
import { texture, uniform, vec2, vec3, float, mix, max, pow, abs, clamp, cross, length, smoothstep, oneMinus, dFdx, dFdy, positionWorld } from 'three/tsl';

export function createGroundMaps( { bake, lit } ) {

	const g = bake.gao, sh = bake.gsh;
	const blank = new THREE.DataTexture( new Uint8Array( [ 255, 0, 255, 255 ] ), 1, 1, THREE.RGBAFormat );
	blank.magFilter = blank.minFilter = THREE.LinearFilter; blank.unpackAlignment = 1; blank.needsUpdate = true;

	// --- contact occlusion: the open share to the power `pow` (3: a rim that reads on wet paving at night)
	const GAO = { k: uniform( 0.6 ), pow: uniform( 3 ), spec: uniform( 1 ), on: uniform( 0 ), tex: null, sh: null };
	const sm = GAO.node = texture( blank, positionWorld.xz.sub( vec2( g.x0, g.z0 ) ).div( vec2( g.sx, g.sz ) ) );
	const dh = positionWorld.y.sub( sm.g.mul( g.hs ).add( g.h0 ) );
	const nr = cross( dFdx( positionWorld ), dFdy( positionWorld ) );
	const up = smoothstep( 0.5, 0.8, abs( nr.y ).div( max( length( nr ), 1e-20 ) ) );
	const band = smoothstep( - 0.3, - 0.15, dh ).mul( oneMinus( smoothstep( g.floor + 0.03, g.floor + 0.12, dh ) ) );
	GAO.f = mix( float( 1 ), pow( sm.r, GAO.pow ), band.mul( up ).mul( GAO.k ).mul( GAO.on ) );
	// (what the wet paving mirrors is most of its light at night: a corner shut in from all sides mirrors less of
	// it too, by `spec` of the same factor)
	GAO.fs = mix( float( 1 ), GAO.f, GAO.spec );

	// --- the lanterns' shadows
	const b = sh.band, S = GAO.sh = { on: uniform( 1 ), spec: uniform( 1 ), pow: uniform( 5 ), mir: uniform( 1 ), all: uniform( 0.5 ), bias: uniform( 0.95 ), span: 3 };
	const w = smoothstep( b[ 0 ], b[ 1 ], dh ).mul( oneMinus( smoothstep( b[ 2 ], b[ 3 ], dh ) ) ).mul( up );
	const v = sm.b.mul( sm.b ), f = sm.a.mul( sm.a );
	const fill = vec3( sh.cF[ 0 ], sh.cF[ 1 ], sh.cF[ 2 ] ).mul( f ), lan = vec3( sh.cD[ 0 ], sh.cD[ 1 ], sh.cD[ 2 ] ).mul( oneMinus( f ) );
	const lv = Math.max( 0, Math.log2( S.span / g.texel ) );
	const sb = sm.level( lv ).b, vb = sb.mul( sb );
	const c = mix( float( 1 ), pow( clamp( v.div( max( vb.mul( S.bias ), 0.02 ) ), 0, 1 ), S.pow ), w.mul( S.on ) );
	const vs = mix( float( 1 ), v, S.on ).mul( c );
	S.c = c;
	S.v = mix( float( 1 ), vs, w );
	S.f = fill.add( lan.mul( mix( v, vs, w ) ) ).div( max( fill.add( lan.mul( mix( v, float( 1 ), w ) ) ), vec3( 1e-5 ) ) ).mul( mix( float( 1 ), c, S.all ) );
	GAO.fg = GAO.fs.mul( mix( float( 1 ), S.v, S.spec ) ); // what gates the lanterns' glints
	GAO.fm = GAO.fs.mul( mix( float( 1 ), c, S.mir ) ); // and the mirror image

	// the baked light of an opaque surface, with both
	return { GAO, litGround: lit.mul( GAO.f ).mul( S.f ) };

}
