// The texture arrays the materials sample: colour + height, and normal + roughness.
//
// Every surface of the town takes its texture from one layer of an array; which one is a number in the vertex.
// The arrays arrive after the materials are built (textures.js fills them in), so each sample here starts on a
// one-texel blank and remembers its node: swapping a texture in is setting `.value` on the nodes of a list.

import * as THREE from 'three';
import { texture, uniform, uniformArray, int, floor, step, clamp, max, mix, oneMinus } from 'three/tsl';

// Colour (rgb) and relief height (alpha). The layers b0 .. b1 live in a second, sharper array (the "big" one:
// the paving and the walls one stands next to); those after them sit `shift` places further down in the first.
// Until the big array is in, b1 = b0 and everything reads the first.
// sample( at, g ): g - the two sides of the footprint to filter over instead of the pixel's own.
export function createLayerSampler( layers, layer ) {

	const blank = new THREE.DataArrayTexture( new Uint8Array( 4 ).fill( 255 ), 1, 1, 1 );
	blank.colorSpace = THREE.SRGBColorSpace; blank.wrapS = blank.wrapT = THREE.RepeatWrapping;
	blank.magFilter = blank.minFilter = THREE.LinearFilter; blank.needsUpdate = true;
	const TX = { a: [], b: [], u: { b0: uniform( 2 ), b1: uniform( 2 ), shift: uniform( 0 ) } };
	const sample = ( at, g = null ) => {

		const lf = floor( layer.add( 0.5 ) ), u = TX.u;
		let a = texture( layers, at ).depth( int( lf.sub( step( u.b1, lf ).mul( u.shift ) ) ) );
		let b = texture( blank, at ).depth( int( clamp( lf.sub( u.b0 ), 0, max( u.b1.sub( u.b0 ).sub( 1 ), 0 ) ) ) );
		if ( g ) { a = a.grad( g[ 0 ], g[ 1 ] ); b = b.grad( g[ 0 ], g[ 1 ] ); }
		TX.a.push( a ); TX.b.push( b );
		return mix( a, b, step( u.b0, lf ).mul( oneMinus( step( u.b1, lf ) ) ) );

	};

	return { TX, sample };

}

// The materials' own normal maps and roughness (the n<size>_<layer> files of the bake), one array for the layers
// n0 .. n1 that have them. `on` is 0 until the array is in: a surface is then lit by its relief alone.
export function createNormalMaps( bake, tier ) {

	const files = bake.texn[ tier.nr ].files;
	const n0 = files.findIndex( ( f ) => f ), n1 = files.length - [ ...files ].reverse().findIndex( ( f ) => f );
	const blank = new THREE.DataArrayTexture( new Uint8Array( [ 128, 128, 153, 255 ] ), 1, 1, 1 );
	blank.wrapS = blank.wrapT = THREE.RepeatWrapping; blank.magFilter = blank.minFilter = THREE.LinearFilter; blank.needsUpdate = true;
	return { n0, n1, count: n1 - n0, size: bake.texn[ tier.nr ].size, files: files.slice( n0, n1 ), blank, nodes: [], on: uniform( 0 ), tex: null };

}

// A number per layer (the bake's `relief`, `rough`, `metal`), read at this pixel's layer.
export const perLayer = ( values, layer ) => uniformArray( values, 'float' ).element( int( layer.add( 0.5 ) ) );
