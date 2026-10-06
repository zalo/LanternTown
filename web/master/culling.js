// What of the town is left out of a frame: the meshes outside the view, and the cells behind the houses.
//
// The view. The renderer culls by bounding sphere, and for these meshes that is a loose fit: 16 m of facade under a
// tower has a sphere of 20 m and more, which a view from next to it never misses, whichever way it looks. So each
// mesh of a level that is shown is tested here first, as its box (and CULL_PAD around it, for what sways or is
// pulled forward) against the four sides of the view, and what is outside is left out by its layers. The view budget
// (budget.js) goes on counting by the spheres, so what is drawn can only be less than what it allows.
//
// Occlusion. The bake has, for every square of ground one can stand on, the cells of which anything at all can be
// seen from there, whichever way one looks (pvs.json, made by tools/pvs.py). A cell that is not in the set of the
// camera's square is left out like one outside the view. The sets hold for a camera at walking height over a square
// that was looked from; any other camera - a shot from a roof - sees everything. So does a walker squeezed between
// two obstacles: Nav.resolve still moves the place, so the eye can be inside one of them, where no eye point of the
// bake was.
//
// apply() is for the frame's own view and runs after the levels are chosen. The mirror image is another view: it
// keeps the spheres (TownLevels.mirrorView lets its meshes through), so apply() comes after it is drawn.

import { Matrix4 } from 'three';

const CULL_PAD = 0.5; // m: what a mesh may draw outside its box
const EYE = 1.65; // m: the walker's eye over the ground

const bytes = ( s ) => Uint8Array.from( atob( s ), ( c ) => c.charCodeAt( 0 ) );

export class TownCulling {

	// levels: the TownLevels; nav: the Nav the walker moves by; pvs: pvs.json as parsed, or null; baked: the stamp of
	// the bake the town is of (sets of another bake are not used)
	constructor( { levels, nav, pvs = null, baked } ) {

		this.chunks = levels.chunks;
		this.nav = nav;
		this.matrix = new Matrix4();
		this.planes = new Float64Array( 16 );
		this.pvs = null;
		if ( pvs && pvs.v === 1 && pvs.baked === baked ) {

			// sq: the set of each square of the grid (0: none); z: the heights each set holds between; bits: the sets, a
			// bit per cell; bit: which bit a cell of the town is; hid: the cells hidden from the set the camera is in (at)
			const at = new Map( pvs.keys.map( ( k, i ) => [ k, i ] ) );
			this.pvs = { s: pvs.s, x0: pvs.x0, y0: pvs.y0, nx: pvs.nx, ny: pvs.ny, nb: pvs.nb, z: pvs.z, sq: new Uint16Array( bytes( pvs.sq ).buffer ), bits: bytes( pvs.bits ),
				bit: this.chunks.map( ( c ) => ( at.has( c.m.k ) ? at.get( c.m.k ) : - 1 ) ), hid: new Uint8Array( this.chunks.length ), at: - 1, hidden: 0, q: { x: 0, y: 0 } };

		}

	}

	apply( camera ) {

		camera.updateMatrixWorld();
		const e = this.matrix.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse ).elements, P = this.planes;
		// (clip x = row 0 . p, y = row 1 . p, w = row 3 . p, and inside is - w <= x, y <= w: the depth convention does not come into it)
		for ( let k = 0; k < 4; k ++ ) for ( let i = 0; i < 4; i ++ ) P[ k * 4 + i ] = e[ i * 4 + 3 ] + ( k & 1 ? - 1 : 1 ) * e[ i * 4 + ( k >> 1 ) ];
		for ( let k = 0; k < 16; k += 4 ) P[ k + 3 ] += CULL_PAD * Math.hypot( P[ k ], P[ k + 1 ], P[ k + 2 ] );
		const hid = this.pvs ? this._occluded( camera ) : null, C = this.chunks;
		for ( let i = 0; i < C.length; i ++ ) for ( const mesh of C[ i ].mesh ) {

			if ( ! mesh || ! mesh.visible ) continue;
			for ( const o of mesh.userData.all ) {

				const b = o.userData.wb;
				let inside = hid !== null && hid[ i ] === 1 ? 0 : 1;
				for ( let k = 0; k < 16 && inside; k += 4 )
					if ( P[ k ] * ( P[ k ] > 0 ? b[ 3 ] : b[ 0 ] ) + P[ k + 1 ] * ( P[ k + 1 ] > 0 ? b[ 4 ] : b[ 1 ] ) + P[ k + 2 ] * ( P[ k + 2 ] > 0 ? b[ 5 ] : b[ 2 ] ) + P[ k + 3 ] < 0 ) inside = 0;
				o.layers.mask = inside;

			}

		}

	}

	// -> per cell 1: hidden from where the camera is, or null: the sets do not hold for it
	_occluded( camera ) {

		// (the bake's ground is x, y with z up)
		const p = this.pvs, x = camera.position.x, y = - camera.position.z, h = camera.position.y;
		const i = Math.floor( ( x - p.x0 ) / p.s ), j = Math.floor( ( y - p.y0 ) / p.s );
		let e = i >= 0 && j >= 0 && i < p.nx && j < p.ny ? p.sq[ j * p.nx + i ] : 0;
		if ( e && ( h < p.z[ e * 2 - 2 ] || h > p.z[ e * 2 - 1 ] ) ) e = 0;
		if ( e ) {

			const q = p.q; q.x = x; q.y = y;
			if ( this.nav.resolve( q, h - EYE ) && Math.hypot( q.x - x, q.y - y ) > 0.05 ) e = 0;

		}

		if ( e !== p.at ) {

			p.at = e; p.hidden = 0;
			const o = ( e - 1 ) * p.nb;
			for ( let k = 0; k < p.hid.length; k ++ ) {

				const b = p.bit[ k ];
				p.hidden += p.hid[ k ] = e && b >= 0 && ! ( p.bits[ o + ( b >> 3 ) ] >> ( b & 7 ) & 1 ) ? 1 : 0;

			}

		}

		return e ? p.hid : null;

	}

	// cells left out by the visibility sets (- 1: a camera they do not hold for; null: no sets)
	get hidden() {

		return this.pvs ? ( this.pvs.at ? this.pvs.hidden : - 1 ) : null;

	}

}
