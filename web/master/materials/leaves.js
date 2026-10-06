// Leaf cards: the trees' foliage.
//
// The uv counts cards, `aux` scales it to texture tiles. The relief height of the leaf mosaic cuts the card
// into leaves, and the cut rises towards the card's edge - a ragged spray, never a straight edge. A mask, not
// a blend: the cards write depth and need no sorting.

import * as THREE from 'three';
import { float, fract, length, smoothstep } from 'three/tsl';

export function createLeaves( ctx ) {

	const { stage, bake, sample } = ctx;
	const { tiles, lit, aux, fogged } = ctx.inputs;
	const material = new THREE.MeshBasicNodeMaterial( { fog: false, side: THREE.DoubleSide } );
	material.name = 'baked leaves';

	const t = sample( tiles.mul( aux.mul( bake.cut_aux ) ) );
	const cell = fract( tiles ).sub( 0.5 ).mul( 2 );
	const h = t.a.mul( 2 ).sub( 1 ); // the layer keeps the height in the upper half of alpha
	material.maskNode = h.greaterThan( smoothstep( 0.45, 1.0, length( cell ) ).mul( 0.8 ).add( 0.33 ) );
	material.colorNode = stage.knee( fogged( t.rgb.mul( lit ) ), float( 0 ) );
	return material;

}
