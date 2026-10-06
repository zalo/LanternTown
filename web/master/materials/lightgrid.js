// The K lanterns that light each ground cell most, as a small float texture.
//
// The diffuse light of every lantern is in the vertex colours; this only feeds the view-dependent part - the
// glints on wet cobbles, polished stone and the canal - which cannot be baked. Per cell: one texel of
// thresholds, then for each entry position + range and colour x intensity + cap.
//
// A list of the K strongest per cell alone leaves a straight edge in the floor's shine wherever a lantern is in
// one cell's list and not in the next one's. So the choice is made a continuous function of the place on the
// ground: with e the light a lantern sends to a point of the floor, a lantern counts in full where e is over
// GLINT_FADE x the threshold there and not at all where it is under it. The threshold is interpolated between
// the cell corners, and a corner's is the largest "strongest light of a lantern left out" of the four cells
// around it - so a lantern has always faded to nothing by the time it reaches a cell that does not list it,
// and it fades where it matters least: where K others outshine it.
//
// Lanterns of one colour closer together than the tier's `merge` (the heads of one post, the bulbs of one
// festoon) share an entry: with few entries per cell a street of many small lights would otherwise keep only a
// few of them and lose most of its shine.

import * as THREE from 'three';

export const GLINT_H2 = 2.25; // floor to lantern, squared (m^2): the same figure here and in the shader
export const GLINT_FADE = 2;

const luminance = ( l ) => 0.2126 * l[ 3 ] + 0.7152 * l[ 4 ] + 0.0722 * l[ 5 ];
// (a blue sign keeps its own entry)
const alike = ( a, b ) => {

	const u = luminance( a ) + 1e-6, v = luminance( b ) + 1e-6;
	return Math.max( Math.abs( a[ 3 ] / u - b[ 3 ] / v ), Math.abs( a[ 4 ] / u - b[ 4 ] / v ), Math.abs( a[ 5 ] / u - b[ 5 ] / v ) ) < 0.3;

};

// The lanterns of the bake: rows of [ x, y, z, r, g, b, range, cap ]. A sign's light shines on what is wet or
// polished as much brighter as its tubes are (stage.HDR.neon, as it stands now); the bake names its signs'
// lights and the factor they are lit with already.
function lanterns( stage, bake ) {

	const named = new Set( bake.signs.lights ), nk = stage.HDR.neon.value / ( bake.signs.k || 1 );
	if ( nk === 1 ) return bake.lights;
	return bake.lights.map( ( l, i ) => ( named.has( i ) ? [ l[ 0 ], l[ 1 ], l[ 2 ], l[ 3 ] * nk, l[ 4 ] * nk, l[ 5 ] * nk, ...l.slice( 6 ) ] : l ) );

}

// Entries: lanterns folded together, strongest first. A burner with a fitting (a halo row with its flag) is
// never folded into another's entry: its own metalwork takes a highlight from it (glints.js), and from the
// middle of a post's two heads that would be none.
function entries( src, halos, merge ) {

	const L = [], taken = new Uint8Array( src.length );
	const fit = halos.filter( ( h ) => h[ 7 ] );
	const strong = fit.length ? src.map( ( l ) => fit.some( ( h ) => Math.abs( h[ 0 ] - l[ 0 ] ) < 0.03 && Math.abs( h[ 1 ] - l[ 1 ] ) < 0.03 && Math.abs( h[ 2 ] - l[ 2 ] ) < 0.03 ) ) : null;
	const order = src.map( ( l, i ) => i ).sort( ( a, b ) => luminance( src[ b ] ) - luminance( src[ a ] ) || a - b );
	for ( const i of order ) {

		if ( taken[ i ] ) continue;
		const a = src[ i ], m = [ 0, 0, 0, 0, 0, 0, 0, 0 ];
		let sum = 0;
		for ( const j of order ) {

			const b = src[ j ];
			if ( taken[ j ] || Math.hypot( a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] ) > merge || ! alike( a, b ) ) continue;
			if ( strong && j !== i && strong[ j ] ) continue;
			taken[ j ] = 1;
			const u = luminance( b ) + 1e-6;
			sum += u;
			for ( const c of [ 0, 1, 2 ] ) m[ c ] += b[ c ] * u;
			for ( const c of [ 3, 4, 5 ] ) m[ c ] += b[ c ];
			m[ 6 ] = Math.max( m[ 6 ], b[ 6 ] );
			m[ 7 ] += b[ 7 ] * u; // share of the light a capped lantern sends up

		}

		for ( const c of [ 0, 1, 2, 7 ] ) m[ c ] /= sum;
		L.push( m );

	}

	return L;

}

export function createLightGrid( { stage, bake, tier } ) {

	const K = tier.glints;
	const L = entries( lanterns( stage, bake ), bake.halos, tier.merge );
	// pools of light: x 1 / ( 1 + d2 * rf ), as the bake (0: the pure inverse square)
	const reach = bake.look.reach, rf = reach > 0 ? 1 / ( reach * reach ) : 0;

	let x0 = Infinity, z0 = Infinity, x1 = - Infinity, z1 = - Infinity;
	for ( const l of L ) {

		x0 = Math.min( x0, l[ 0 ] - l[ 6 ] ); x1 = Math.max( x1, l[ 0 ] + l[ 6 ] );
		z0 = Math.min( z0, l[ 2 ] - l[ 6 ] ); z1 = Math.max( z1, l[ 2 ] + l[ 6 ] );

	}

	const G = Math.max( 1.25, ( x1 - x0 ) / 320, ( z1 - z0 ) / 320 );
	const nx = Math.max( 1, Math.ceil( ( x1 - x0 ) / G ) ), nz = Math.max( 1, Math.ceil( ( z1 - z0 ) / G ) );
	const W = 2 * K + 1;
	const data = new Float32Array( nx * W * nz * 4 );
	const lum = L.map( luminance );
	const list = new Int32Array( nx * nz * K ).fill( - 1 ), tau = new Float32Array( nx * nz );
	const best = [], row = [];
	for ( let iz = 0; iz < nz; iz ++ ) {

		const cz0 = z0 + iz * G;
		row.length = 0;
		for ( let i = 0; i < L.length; i ++ ) if ( L[ i ][ 2 ] > cz0 - L[ i ][ 6 ] && L[ i ][ 2 ] < cz0 + G + L[ i ][ 6 ] ) row.push( i );
		for ( let ix = 0; ix < nx; ix ++ ) {

			// the most each lantern sends to any point of the cell
			const cx0 = x0 + ix * G;
			best.length = 0;
			for ( const i of row ) {

				const l = L[ i ];
				const dx = Math.max( cx0 - l[ 0 ], 0, l[ 0 ] - cx0 - G ), dz = Math.max( cz0 - l[ 2 ], 0, l[ 2 ] - cz0 - G );
				const d2 = dx * dx + dz * dz + GLINT_H2;
				const w = Math.max( 0, 1 - ( d2 / ( l[ 6 ] * l[ 6 ] ) ) ** 2 );
				const e = lum[ i ] * w * w / d2 / ( 1 + d2 * rf );
				if ( e > 0 ) best.push( [ e, i ] );

			}

			best.sort( ( a, b ) => b[ 0 ] - a[ 0 ] || a[ 1 ] - b[ 1 ] );
			for ( let k = 0; k < K && k < best.length; k ++ ) list[ ( iz * nx + ix ) * K + k ] = best[ k ][ 1 ];
			tau[ iz * nx + ix ] = best.length > K ? best[ K ][ 0 ] : 0;

		}

	}

	const at = ( ix, iz ) => ( ix < 0 || iz < 0 || ix >= nx || iz >= nz ? 0 : tau[ iz * nx + ix ] );
	const corner = ( cx, cz ) => Math.max( at( cx - 1, cz - 1 ), at( cx, cz - 1 ), at( cx - 1, cz ), at( cx, cz ) );
	for ( let iz = 0; iz < nz; iz ++ ) for ( let ix = 0; ix < nx; ix ++ ) {

		let o = ( iz * nx + ix ) * W * 4;
		data.set( [ corner( ix, iz ), corner( ix + 1, iz ), corner( ix, iz + 1 ), corner( ix + 1, iz + 1 ) ], o );
		for ( let k = 0; k < K; k ++ ) {

			const i = list[ ( iz * nx + ix ) * K + k ], l = i >= 0 ? L[ i ] : null;
			o += 8;
			if ( l ) data.set( [ l[ 0 ], l[ 1 ], l[ 2 ], l[ 6 ], l[ 3 ], l[ 4 ], l[ 5 ], l[ 7 ] ], o - 4 );
			else data.set( [ 0, - 1e4, 0, 1, 0, 0, 0, 1 ], o - 4 );

		}

	}

	const tex = new THREE.DataTexture( data, nx * W, nz, THREE.RGBAFormat, THREE.FloatType );
	tex.minFilter = tex.magFilter = THREE.NearestFilter;
	tex.generateMipmaps = false;
	tex.needsUpdate = true;
	return { tex, K, G, x0, z0, nx, nz, W, rf, lights: L.length };

}
