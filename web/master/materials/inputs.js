// What every material of the town reads from a vertex, and the fog they all end in.
//
// A vertex of the bake is 16 bit throughout: `_pl` is the position again with the texture layer in .w, `uv`
// counts texture tiles around the middle of its range, and `_col` holds ALL the light that arrives there,
// already times the surface tint, as a square root (so the dark end keeps its steps) - plus one more number,
// `aux`: the gloss of an opaque surface (0 .. 0.5), "this glows" (from 0.75 up), the cover of glass.
// Nothing is lit at run time: a material multiplies its texture by `lit` and adds what cannot be baked.

import { attribute, uv, varying, step, mix, normalize, cameraPosition, positionWorld } from 'three/tsl';

export function createInputs( { stage, bake } ) {

	const pl = attribute( '_pl', 'vec4' ), cv = attribute( '_col', 'vec4' );
	const aux = varying( cv.w, 'vAux' );

	return {
		pl, cv, aux,
		layer: varying( pl.w.mul( 65535 ), 'vLayer' ),
		tiles: varying( uv().mul( 65535 / 65536 ).sub( 0.5 ).mul( 2 * bake.uv_range ), 'vTiles' ),
		lit: varying( cv.rgb.mul( cv.rgb ).mul( bake.colour_scale ), 'vLit' ),
		glowing: step( 0.75, aux ),
		V: normalize( cameraPosition.sub( positionWorld ) ), // towards the eye
		fogged: ( c ) => mix( c, stage.FOG.color, stage.fogAmount() ),
	};

}
