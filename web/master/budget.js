// The triangle budget of a view (the tier's `budget: 'arc'`, `tris`, `detail`, `first`, `sure`).
//
// A frame costs what the camera has in view, not what the town holds: so the ceiling is kept per view. From where the
// camera stands, 64 headings x 8 pitches are counted - level, looking up (which brings the tall things at the sides
// in) and looking down - and whichever way the walker turns next, what the renderer's culling lets through stays
// under `tris`. A mesh is in a view when the sphere around it passes that view's side planes, its near and upper
// plane and, looking past the edge of the picture, the plane that then cuts off what is straight ahead; every arc is
// half a heading step wider. The mirror image of the wet paving is drawn in the same frame - the town at its lowest
// level, seen from the camera's reflection - and counts with it.
//
// The rule: every cell starts at its lowest level and the second goes to all that ask for it, nearest first; then
// level by level, nearest first, a cell climbs while every view that sees it has the room. Coverage comes before
// detail: no cell is left at the far level to pay for the top one of another. Two exceptions put what is within
// arm's reach first (`first`, `sure`: see plan()). The top level counts cluster by cluster, each drawn down to the
// parts that still show at its distance (`detail`: size / distance). A cell that did not have a level the last time
// needs 6 % to spare, so nothing flickers at the edge of the budget.

const HEADINGS = 64;
const PITCHES = [ 0, 15, 30, 45, 60, 75, - 25, - 50 ].map( ( d ) => d * Math.PI / 180 );
const VIEWS = HEADINGS * PITCHES.length;
const STEP = 2 * Math.PI / HEADINGS;
const PART_MAX = 0.3; // m: no part larger than this is ever left out

// index count of a cluster drawn down to parts of size s (m): its range is in order of part size, and `steps` has
// the count down to each size
function cut( steps, s ) {

	for ( const [ size, n ] of steps ) if ( size <= s ) return n;
	return steps[ steps.length - 1 ][ 1 ];

}

export class ViewBudget {

	// names: the levels, lowest first; mirror: { wide, level } of the planar mirror image (how many times the
	// camera's view it spans, the height of its plane) or null; cellSize: the grid of the cells (m)
	constructor( tier, names, { mirror = null, cellSize = 16 } = {} ) {

		this.tier = tier;
		this.names = names;
		this.mirror = mirror;
		this.cellSize = cellSize;
		this.load = new Float64Array( VIEWS ); // triangles in each view
		this.inMirror = new Float32Array( VIEWS ); // ... of them in the mirror image
		this.tmp = new Float32Array( VIEWS );
		this.bearing = new Float32Array( HEADINGS );
		this.most = 0; // what the fullest view holds
		this.mostBy = null; // [ heading (degrees), pitch, triangles at each level.., in the mirror image ]

	}

	// -> ( sphere [ x, z, radius, , height ], out ): which views of a camera `wide` times as broad have it in them
	_arc( camera, wide ) {

		const x = camera.position.x, y = camera.position.y, z = camera.position.z, B = this.bearing, hs = STEP / 2;
		const th = Math.tan( camera.fov * Math.PI / 360 ) * wide, tw = th * camera.aspect;
		const sw = Math.sqrt( 1 + tw * tw ), sh = Math.sqrt( 1 + th * th );
		return ( s, out ) => {

			const dx = s[ 0 ] - x, dz = s[ 1 ] - z, d = Math.max( Math.hypot( dx, dz ), 1e-3 ), r = s[ 2 ], a = Math.atan2( dx, dz );
			for ( let h = 0; h < HEADINGS; h ++ ) {

				const b = Math.abs( h * STEP - a ) % ( 2 * Math.PI );
				B[ h ] = b > Math.PI ? 2 * Math.PI - b : b; // how far heading h is off the bearing of the sphere

			}

			for ( let k = 0; k < PITCHES.length; k ++ ) {

				// (looking down is looking up with the heights mirrored)
				const p = Math.abs( PITCHES[ k ] ), sp = Math.sin( p ), cp = Math.cos( p ), dy = PITCHES[ k ] < 0 ? y - s[ 4 ] : s[ 4 ] - y;
				// side planes: d ( sin b - A cos b ) <= r sw + tw dy sp, A = tw cos p
				const A = tw * cp, phi = Math.atan( A ), rho = ( r * sw + tw * dy * sp ) / ( d * Math.sqrt( 1 + A * A ) );
				let w1 = 4, w2 = 0; // in view up to w1 off the bearing, and again from w2 on (around the back)
				if ( rho <= - 1 ) { w1 = - 1; w2 = 4; } else if ( rho < 1 ) { const t = Math.asin( rho ); w1 = phi + t; w2 = Math.PI + phi - t; }
				// near plane: d cos b cp + dy sp >= - r
				const cn = - ( r + dy * sp ) / ( d * cp ), wn = cn >= 1 ? - 1 : cn <= - 1 ? 4 : Math.acos( cn );
				// upper plane: d cos b ( sp + th cp ) >= dy ( cp - th sp ) - r sh
				const ct = ( dy * ( cp - th * sp ) - r * sh ) / ( d * ( sp + th * cp ) ), wt = ct >= 1 ? - 1 : ct <= - 1 ? 4 : Math.acos( ct );
				// the lower plane, once it is above the horizon: only what is far enough off to the side stays above it
				const q = sp - th * cp, cl = q > 0 ? ( dy * ( cp + th * sp ) + r * sh ) / ( d * q ) : 1, wl = cl >= 1 ? 0 : cl <= - 1 ? 4 : Math.acos( cl );
				for ( let h = 0, o = k * HEADINGS; h < HEADINGS; h ++ ) {

					const b = B[ h ];
					out[ o + h ] = ( b <= w1 + hs || b >= w2 - hs ) && b <= wn + hs && b <= wt + hs && b >= wl - hs ? 1 : 0;

				}

			}

			return out;

		};

	}

	// what the budget keeps of a cell between passes: the views each of its levels is in, and what they see of it now
	_slot( c ) {

		return c.view || ( c.view = { arc: this.names.map( () => new Uint8Array( VIEWS ) ), seen: new Float32Array( VIEWS ), mirror: new Uint8Array( VIEWS ) } );

	}

	// chunks: every cell; order: the cells that ask for more than their lowest level (`want`), nearest first.
	// Leaves in `want` the level each of them gets, and in the `n` of its top level's clusters how far each is drawn.
	plan( camera, chunks, order ) {

		const T = this.tier, names = this.names, N = names.length, x = camera.position.x, z = camera.position.z;
		const arc = this._arc( camera, 1 ), load = this.load, tmp = this.tmp, mir = this.inMirror;
		const M = this.mirror, arcM = M ? this._arc( camera, M.wide ) : null, ms = [ 0, 0, 0, 0, 0 ];
		load.fill( 0 );
		mir.fill( 0 );
		for ( const c of chunks ) {

			const v = this._slot( c ), A = arc( c.sph[ 0 ], v.arc[ 0 ] ), n = c.sph[ 0 ][ 3 ];
			for ( let h = 0; h < VIEWS; h ++ ) load[ h ] += v.seen[ h ] = A[ h ] ? n : 0;
			if ( ! M || ! c.mesh[ 0 ] ) continue;
			// (the same sphere with its height mirrored in the paving)
			ms[ 0 ] = c.sph[ 0 ][ 0 ]; ms[ 1 ] = c.sph[ 0 ][ 1 ]; ms[ 2 ] = c.sph[ 0 ][ 2 ]; ms[ 4 ] = 2 * M.level - c.sph[ 0 ][ 4 ];
			const R = arcM( ms, v.mirror );
			for ( let h = 0; h < VIEWS; h ++ ) if ( R[ h ] ) mir[ h ] += n;

		}

		if ( M ) for ( let h = 0; h < VIEWS; h ++ ) load[ h ] += mir[ h ];

		for ( const c of order ) { c.ask = c.want; c.want = 0; }
		const full = T.tris, spare = T.tris * 0.94, K = T.detail || 0;
		// cell c one level up, to l, if every view that sees more of it then has the room
		const up = ( c, l ) => {

			if ( ! c.m[ names[ l ] ] || c.failed & ( 1 << l ) ) return false;
			const v = c.view, A = arc( c.sph[ l ], v.arc[ l ] ), seen = v.seen, cap = c.was < l ? spare : full, parts = l === N - 1 && c.top.length > 0;
			const n = parts ? c.rest : c.sph[ l ][ 3 ];
			for ( let h = 0; h < VIEWS; h ++ ) tmp[ h ] = A[ h ] ? n : 0;
			if ( parts ) for ( const q of c.top ) {

				const d = Math.hypot( Math.max( q.box[ 0 ] - x, 0, x - q.box[ 2 ] ), Math.max( q.box[ 1 ] - z, 0, z - q.box[ 3 ] ) );
				// (a part is left out a little further off than it comes back: no flicker at a step)
				let t = K > 0 ? cut( q.steps, Math.min( PART_MAX, d * K ) ) : q.full;
				if ( K > 0 && c.was === l && q.n > t && q.n <= cut( q.steps, Math.min( PART_MAX, d * K * 0.85 ) ) ) t = q.n;
				q.t = t;
				const Q = arc( q.sph, q.arc || ( q.arc = new Uint8Array( VIEWS ) ) );
				for ( let h = 0; h < VIEWS; h ++ ) if ( Q[ h ] ) tmp[ h ] += t / 3;

			}

			for ( let h = 0; h < VIEWS; h ++ ) if ( tmp[ h ] > seen[ h ] && load[ h ] + tmp[ h ] - seen[ h ] > cap ) return false;
			for ( let h = 0; h < VIEWS; h ++ ) { load[ h ] += tmp[ h ] - seen[ h ]; seen[ h ] = tmp[ h ]; }
			if ( parts ) for ( const q of c.top ) q.n = q.t;
			c.want = l;
			return true;

		};

		for ( const c of order ) up( c, 1 );
		// What is within arm's reach comes before the rule: a cell within `first` metres (a fifth more for one that has
		// it) gets its top level before a facade 40 m off gets its second. Where the top level of three cells is most
		// of the budget (Medium) that left a facade 20 m off at its far level, which is made to be seen from 45 m on:
		// with `sure` the cell one stands in comes first, then the second level of all within `sure` metres, and only
		// then the top level of the neighbours.
		if ( T.sure > 0 ) {

			const gx = Math.floor( x / this.cellSize ), gz = Math.floor( z / this.cellSize );
			for ( const c of order ) if ( c.gx === gx && c.gz === gz ) for ( let l = 2; l <= c.ask && c.want === l - 1; l ++ ) up( c, l );
			for ( const c of order ) if ( c.want === 1 && c.ask >= 2 && c.d < ( c.was >= 2 ? T.sure * 1.2 : T.sure ) ) up( c, 2 );

		}

		if ( T.first > 0 ) for ( const c of order ) if ( c.d < ( c.was === N - 1 ? T.first * 1.2 : T.first ) ) for ( let l = Math.max( 2, c.want + 1 ); l <= c.ask && c.want === l - 1; l ++ ) up( c, l );
		for ( let l = 2; l < N; l ++ ) for ( const c of order ) if ( c.want === l - 1 && l <= c.ask ) up( c, l );

		let most = 0, at = 0;
		for ( let h = 0; h < VIEWS; h ++ ) if ( load[ h ] > most ) { most = load[ h ]; at = h; }
		this.most = most;
		const by = this.mostBy = [ Math.round( ( at % HEADINGS ) * 360 / HEADINGS ), Math.round( PITCHES[ at / HEADINGS | 0 ] * 180 / Math.PI ), ...names.map( () => 0 ), Math.round( mir[ at ] ) ];
		for ( const c of chunks ) if ( c.view.seen[ at ] > 0 ) by[ 2 + c.want ] += Math.round( c.view.seen[ at ] );

	}

}
