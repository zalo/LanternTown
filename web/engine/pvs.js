// What can be seen from where: the baked town's visibility sets (pvs.json beside a bake's manifest, read by
// BakedTown._occluded of lite.js). Run by the town page with ?pvs=bake (tools/pvs.py drives it and files the result).
//
// The town is drawn in flat cell colours - every cell its own number - from eye points all over the ground one can
// walk on, five faces of a cube each (the four sides and up; what is steeply below is within `near` metres, and
// those cells are always in the set), and the numbers that come back in the picture are the cells of which
// something shows from there. That is done once for every level of detail in the bake, each time with the whole town
// at that level (a level's silhouette is not the next one's), then once more for every tier that loads the bake with
// each cell at the level that tier has it at from there (near cells fine, far ones coarse: a far cell at its lowest
// level shows past a near one at its top level by slivers that no one level has) and with the top level down to
// the parts the tier's triangle budget can leave it with (Phone drops parts of up to 0.3 m from the cells around
// the camera when a view is full: what stood behind them shows), and all of it is added up: per square of `square`
// metres the cells seen from any eye point in it or within `pad` of it, at any level. Being wrong on that side only
// costs triangles; a cell wrongly left out would be a hole in the town, so everything errs towards "seen":
//   - glass, water and the falls hide nothing (they are not drawn here), leaf cards are cut as in the picture
//   - an eye point is a place a walker can be: a grid point inside a wall or a stall is moved out of it as the
//     walker would be (Nav.resolve), and dropped if that is further than the grid is wide
//   - three eye heights, the walker's and 0.2 m either side of it, each cube turned by its own angle
//   - a square nobody looked from, or a camera at another height than a walker's there, has no set: all is drawn
import * as THREE from 'three';
import { uniform, vec4, positionWorld } from 'three/tsl';
import { BAKED_TIERS } from './lite.js';

const K_OPAQUE = 0, K_CUT = 4; // (the kinds of lite.js that hide what is behind them)
const LEVELS = [ 'lo', 'mid', 'near', 'ring', 'fine' ];
const FACES = 5, COLS = 3, ROWS = 2;

const b64 = ( u8 ) => {

	let s = '';
	for ( let i = 0; i < u8.length; i += 0x8000 ) s += String.fromCharCode.apply( null, u8.subarray( i, i + 0x8000 ) );
	return btoa( s );

};

// the eye points: [ x, y (town frame), ground height ] of every place on the grid a walker can stand
export function eyePoints( nav, step ) {

	const out = [];
	let x0 = Infinity, y0 = Infinity, x1 = - Infinity, y1 = - Infinity;
	for ( const r of nav.regions ) { x0 = Math.min( x0, r.rect[ 0 ] ); y0 = Math.min( y0, r.rect[ 1 ] ); x1 = Math.max( x1, r.rect[ 2 ] ); y1 = Math.max( y1, r.rect[ 3 ] ); }
	for ( let j = Math.ceil( y0 / step ); j * step <= y1; j ++ ) for ( let i = Math.ceil( x0 / step ); i * step <= x1; i ++ ) {

		const x = i * step, y = j * step, hs = [];
		for ( const r of nav.regions ) {

			const R = r.rect;
			if ( x < R[ 0 ] || x > R[ 2 ] || y < R[ 1 ] || y > R[ 3 ] ) continue;
			const h = nav.heightOf( r, x, y );
			if ( ! hs.some( ( g ) => Math.abs( g - h ) < 0.3 ) ) hs.push( h );

		}

		for ( const h of hs ) {

			const p = { x, y };
			for ( let k = 0; k < 4 && nav.resolve( p, h ); k ++ );
			if ( Math.hypot( p.x - x, p.y - y ) > step ) continue; // (inside a house)
			out.push( [ p.x, p.y, nav.groundAt( p.x, p.y, h + 0.05 ) ?? h ] );

		}

	}

	return { points: out, rect: [ x0, y0, x1, y1 ] };

}

// the town in cell colours: a scene of its own, a cube's five faces side by side in one target, and what comes back
// (where: not colours but floats - the place in the world that shows in a pixel, and the cell + 1 beside it; holesPVS)
function idRig( town, N, where = false ) {

	const renderer = town.eng.renderer, C = town.chunks;
	const kindOf = new Map();
	for ( const M of [ town.mats, town.matsDir, town.fmats, town.fmatsDir ] ) if ( M ) for ( const k in M ) kindOf.set( M[ k ], + k );

	// the cell colours: the number of the mesh's cell, + 1, in red and green
	const idU = uniform( new THREE.Vector3() ).onObjectUpdate( ( { object }, self ) => { const n = object.userData.pvsId; self.value.set( ( n & 255 ) / 255, ( n >> 8 ) / 255, 0 ); } );
	const solid = new THREE.MeshBasicNodeMaterial( { fog: false } );
	solid.colorNode = vec4( idU, 1 );
	const leaves = new THREE.MeshBasicNodeMaterial( { fog: false, side: THREE.DoubleSide } );
	leaves.colorNode = vec4( idU, 1 );
	leaves.maskNode = town.mats[ K_CUT ].maskNode;
	const none = new THREE.MeshBasicNodeMaterial( { fog: false } );
	none.visible = false;
	for ( const m of [ solid, leaves ] ) m.toneMapped = false;
	if ( where ) {

		// (outputNode: past the opaque material's alpha of 1 and its clamp at 0; the leaves' cut-out stays)
		const idF = uniform( 0 ).onObjectUpdate( ( { object } ) => object.userData.pvsId );
		solid.outputNode = vec4( positionWorld, idF );
		leaves.outputNode = vec4( positionWorld, idF.negate() ); // (leaf cards: told apart by the sign)

	}

	const idMat = ( m ) => ( Array.isArray( m ) ? m.map( idMat ) : kindOf.get( m ) === K_OPAQUE ? solid : kindOf.get( m ) === K_CUT ? leaves : none );

	const scene = new THREE.Scene();
	// the meshes of one level of a cell (BakedTown._build), again, in its colour (and as much of each as is drawn now)
	const again = ( mesh, n ) => {

		const g = new THREE.Group();
		g.matrixAutoUpdate = false;
		g.matrix.copy( mesh.matrix );
		for ( const o of mesh.userData.all ) {

			const q = new THREE.Mesh( o.geometry, idMat( o.userData.mat || o.material ) );
			q.userData.pvsId = n + 1;
			q.matrixAutoUpdate = false;
			// (a level in one mesh is shown or hidden as a whole by the page - the lowest level of a cell that is at a higher one
			// now is hidden - and here it is always the level that is asked for; the parts of a level in clusters: as drawn now)
			q.visible = o === mesh || o.visible;
			g.add( q );

		}

		g.updateMatrixWorld( true );
		scene.add( g );
		return g;

	};

	const W = COLS * N, H = ROWS * N;
	const rt = new THREE.RenderTarget( W, H, { depthBuffer: true, samples: 0, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, type: where ? THREE.FloatType : THREE.UnsignedByteType } );
	rt.texture.colorSpace = THREE.NoColorSpace;
	const cam = new THREE.PerspectiveCamera( 90, 1, 0.05, 2000 );
	const keep = { rt: renderer.getRenderTarget(), auto: renderer.autoClear, scissor: renderer.getScissorTest(), mrt: renderer.getMRT(), color: renderer.getClearColor( new THREE.Color() ), alpha: renderer.getClearAlpha() };
	const begin = () => {

		renderer.setMRT( null );
		renderer.setClearColor( 0x000000, 0 );
		renderer.setScissorTest( true );
		renderer.setRenderTarget( rt );

	};

	const end = () => {

		renderer.setRenderTarget( keep.rt );
		renderer.autoClear = keep.auto;
		renderer.setScissorTest( keep.scissor );
		renderer.setMRT( keep.mrt );
		renderer.setClearColor( keep.color, keep.alpha );

	};

	// the cube around an eye (town frame; turn: about the vertical) -> its pixels (r | g << 8: the cell + 1, 0: sky)
	const shoot = ( x, y, z, turn ) => {

		cam.position.set( x, z, - y );
		for ( let f = 0; f < FACES; f ++ ) {

			if ( f < 4 ) cam.rotation.set( 0, turn + f * Math.PI / 2, 0, 'YXZ' ); else cam.rotation.set( Math.PI / 2, turn, 0, 'YXZ' );
			cam.updateMatrixWorld();
			rt.viewport.set( ( f % COLS ) * N, Math.floor( f / COLS ) * N, N, N );
			rt.scissor.copy( rt.viewport );
			renderer.autoClear = f === 0; // (a clear is of the whole target, whatever the scissor)
			renderer.render( scene, cam );

		}

		return renderer.readRenderTargetPixelsAsync( rt, 0, 0, W, H );

	};

	// where a pixel of the cube looks: a narrow view (fov, degrees) along it, and what that shows - { dir (town frame),
	// n: its pixels of the cell `id`, frame: Map id -> pixels }. Which way up the rows come back is found by trying:
	// the view that has the cell in it is the one
	const spot = async ( x, y, z, turn, i, id, fov = 6 ) => {

		const col = i % W, row = Math.floor( i / W ), u = ( ( col % N ) + 0.5 ) / N * 2 - 1, v = ( ( row % N ) + 0.5 ) / N * 2 - 1;
		const d = new THREE.Vector3(), e = new THREE.Euler();
		let found = null;
		for ( const flip of [ 0, 1 ] ) for ( const sv of [ - 1, 1 ] ) {

			const f = ( flip ? ROWS - 1 - Math.floor( row / N ) : Math.floor( row / N ) ) * COLS + Math.floor( col / N );
			if ( f >= FACES || found ) continue;
			if ( f < 4 ) e.set( 0, turn + f * Math.PI / 2, 0, 'YXZ' ); else e.set( Math.PI / 2, turn, 0, 'YXZ' );
			d.set( u, sv * v, - 1 ).normalize().applyEuler( e );
			cam.position.set( x, z, - y );
			cam.fov = fov; cam.updateProjectionMatrix();
			cam.lookAt( x + d.x, z + d.y, - y + d.z );
			cam.updateMatrixWorld();
			rt.viewport.set( 0, 0, N, N );
			rt.scissor.copy( rt.viewport );
			renderer.autoClear = true;
			renderer.render( scene, cam );
			cam.fov = 90; cam.updateProjectionMatrix();
			const px = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, W, H ), frame = new Map();
			for ( let k = 0; k < px.length; k += 4 ) { const c = px[ k ] | px[ k + 1 ] << 8; if ( c ) frame.set( c, ( frame.get( c ) || 0 ) + 1 ); }
			if ( frame.get( id ) ) found = { dir: [ d.x, - d.z, d.y ], n: frame.get( id ), frame };

		}

		return found;

	};

	// a copy that is done with: the renderer keeps what it made for a mesh - and through that its geometry - until the
	// mesh or its material says 'dispose' (the geometry saying so is not enough, and the materials here are shared)
	const drop = ( g ) => { scene.remove( g ); for ( const q of g.children ) q.dispatchEvent( { type: 'dispose' } ); };

	return { scene, again, shoot, spot, begin, end, drop, W, H, dispose: () => { rt.dispose(); for ( const m of [ solid, leaves, none ] ) m.dispose(); } };

}

// -> the object that becomes pvs.json. face: pixels across one face of the cube; step: the grid of eye points (m);
// square: the squares the sets are kept for; pad: how far outside its square an eye point still counts for it;
// eyes: eye heights over the ground; near: cells this close to a square are always in its set; slack: how far from
// the walker's eye height (1.65 m over the ground of its square) a camera may be and still use the set
export async function bakePVS( town, { face = 512, step = 0.5, square = 1, pad = 1.5, eyes = [ 1.45, 1.65, 1.85 ], near = 8, slack = 0.5, eye = 1.65, levels = null, bands = null, limit = 0, debug = - 1, on = () => {} } = {} ) {

	const C = town.chunks, t0 = performance.now();
	const rig = idRig( town, face ), { W, H } = rig;
	const E = eyePoints( town.nav, step );
	let pts = E.points;
	if ( limit > 0 && pts.length > limit ) { const k = pts.length / limit; pts = Array.from( { length: limit }, ( _, i ) => pts[ Math.floor( i * k ) ] ); } // (a trial run)
	const nb = Math.ceil( C.length / 8 ), seen = new Uint8Array( pts.length * nb ), count = new Uint32Array( C.length + 1 );
	const names = ( levels || LEVELS ).filter( ( l ) => C.some( ( c ) => c.m[ l ] ) );
	// bands: the tiers whose mix of levels is looked at as well (default: every tier that loads this bake - the fine ones or the others)
	const tiers = ( bands || Object.keys( BAKED_TIERS ).filter( ( t ) => ( BAKED_TIERS[ t ].top === 'fine' ) === ( town.T.top === 'fine' ) ) ).filter( ( t ) => BAKED_TIERS[ t ] );
	const stats = { points: pts.length, cubes: 0, levels: {}, bands: [], face, step, square, pad, eyes, near };
	const src = {}; // level -> per cell, [ the mesh built for it, its entry in the manifest ] (not for a cell's lowest level: the page's own mesh)
	const at = {}, kept = []; // level -> per cell, its copy in cell colours; the meshes built for them (held to the end when there are tiers to mix them for)
	let pass = 0;
	const pending = new Set();
	const read = ( i, shot ) => {

		const p = shot.then( ( px ) => {

			if ( i === debug ) ( bakePVS.debug || ( bakePVS.debug = [] ) ).push( { W, H, px, at: pts[ i ] } ); // (?debug=<eye point>: its pictures, to look at)
			const o = i * nb;
			let last = 0;
			for ( let k = 0; k < px.length; k += 4 ) {

				const n = px[ k ] | px[ k + 1 ] << 8;
				if ( n === last ) continue;
				last = n;
				if ( n > 0 && n <= C.length ) seen[ o + ( n - 1 >> 3 ) ] |= 1 << ( n - 1 & 7 );
				else if ( n ) count[ 0 ] ++; // (a colour that is no cell: none expected)

			}

			pending.delete( p );

		} );
		pending.add( p );

	};

	// every eye point's cubes of what the scene holds (set: what it is to hold for this one). Each pass turns its cubes its own way.
	const sweep = async ( what, set = null ) => {

		const turn = ( pass ++ ) * 0.381966;
		for ( let i = 0; i < pts.length; i ++ ) {

			const [ x, y, g ] = pts[ i ];
			if ( set ) set( x, y );
			for ( let k = 0; k < eyes.length; k ++ ) {

				read( i, rig.shoot( x, y, g + eyes[ k ], ( ( ( i * 2 + k ) * 0.61803398875 + turn ) % 1 ) * Math.PI / 2 ) );
				stats.cubes ++;

			}

			while ( pending.size > 12 ) await Promise.race( pending );
			if ( i % 200 === 0 ) on( `${what}: ${i} / ${pts.length} eye points, ${( ( performance.now() - t0 ) / 1000 ).toFixed( 0 )} s` );

		}

		await Promise.all( pending );

	};

	try {

		rig.begin();
		for ( const name of names ) {

			// the whole town at this level (a cell without it: at the one below that it has)
			on( `level ${name}: loading` );
			const t1 = performance.now(), built = [], groups = at[ name ] = new Array( C.length ), from = src[ name ] = new Array( C.length );
			let next = 0, tris = 0;
			const one = async () => {

				for ( ;; ) {

					const n = next ++;
					if ( n >= C.length ) return;
					const c = C[ n ];
					let l = LEVELS.indexOf( name );
					if ( name === 'ring' && ! c.m.ring ) l = LEVELS.indexOf( 'near' ); // (ring is near without its smallest parts)
					while ( l > 0 && ! c.m[ LEVELS[ l ] ] ) l --;
					const m = c.m[ LEVELS[ l ] ];
					tris += m.ni / 3;
					if ( l === 0 ) { groups[ n ] = rig.again( c.mesh[ 0 ], n ); continue; }
					const mesh = town._build( await town._levelBytes( c, LEVELS[ l ] ), m, null, 1 );
					built.push( mesh );
					from[ n ] = [ mesh, m ];
					groups[ n ] = rig.again( mesh, n );

				}

			};

			await Promise.all( Array.from( { length: 6 }, one ) );
			const t2 = performance.now();
			await sweep( `level ${name}` );
			stats.levels[ name ] = { tris: Math.round( tris ), load_s: + ( ( t2 - t1 ) / 1000 ).toFixed( 1 ), render_s: + ( ( performance.now() - t2 ) / 1000 ).toFixed( 1 ) };
			if ( tiers.length ) { for ( const g of groups ) g.visible = false; kept.push( ...built ); continue; }
			for ( const g of groups ) rig.drop( g );
			for ( const mesh of built ) for ( const o of mesh.userData.all ) o.geometry.dispose();

		}

		// a tier's mix: every cell at the level the tier's distances give it from the eye point (BakedTown.update,
		// without what the budget and the late letting-go change about it: those are single levels' pictures again)
		for ( const tier of tiers ) {

			const T = BAKED_TIERS[ tier ], t1 = performance.now();
			const Lt = T.top === 'fine' ? [ 'lo', 'mid', 'near', 'fine' ] : [ 'lo', 'mid', T.top || 'near' ], Rt = T.top === 'fine' ? [ Infinity, T.mid, T.near, T.fine ] : [ Infinity, T.mid, T.near ];
			if ( Lt.some( ( l ) => ! at[ l ] ) ) continue; // (a level this bake does not have, or was not asked for)
			// the top level's clusters without the parts the budget may leave out: up to 0.3 m with the ladder of the view
			// budget, with the arc budget what T.detail makes of the furthest a top level is held (_viewBudget, _arcBudget)
			const size = T.budget === 'view' ? 0.3 : T.budget === 'arc' ? Math.min( 0.3, ( Rt[ Rt.length - 1 ] * 1.2 + 2 ) * ( T.detail || 0 ) ) : 0;
			for ( const s of src[ Lt[ Lt.length - 1 ] ] ) if ( s && s[ 0 ].userData.parts ) for ( const part of s[ 0 ].userData.parts ) {

				const p = s[ 1 ].parts.find( ( p ) => s[ 1 ].sub[ p[ 0 ] ][ 1 ] === part.userData.start );
				part.geometry.setDrawRange( part.userData.start, size > 0 ? town._cut( p[ 1 ], size ) : s[ 1 ].sub[ p[ 0 ] ][ 2 ] );

			}

			const cur = new Int8Array( C.length ).fill( - 1 );
			await sweep( `mix ${tier}`, ( x, y ) => {

				for ( let n = 0; n < C.length; n ++ ) {

					const d = town._dist( C[ n ], x, - y );
					let l = 0;
					for ( let k = Lt.length - 1; k > 0 && ! l; k -- ) if ( d < Rt[ k ] ) l = k;
					if ( l === cur[ n ] ) continue;
					if ( cur[ n ] >= 0 ) at[ Lt[ cur[ n ] ] ][ n ].visible = false;
					at[ Lt[ l ] ][ n ].visible = true;
					cur[ n ] = l;

				}

			} );
			cur.forEach( ( l, n ) => { if ( l >= 0 ) at[ Lt[ l ] ][ n ].visible = false; } );
			stats.bands.push( tier );
			stats.levels[ 'mix ' + tier ] = { parts_from: + size.toFixed( 3 ), render_s: + ( ( performance.now() - t1 ) / 1000 ).toFixed( 1 ) };

		}

		if ( tiers.length ) for ( const name in at ) for ( const g of at[ name ] ) rig.drop( g );
		for ( const mesh of kept ) for ( const o of mesh.userData.all ) o.geometry.dispose();

	} finally {

		rig.end();
		rig.dispose();

	}

	// per square: the eye points in it or within `pad` of it, their sets added up, and the cells next to it
	const x0 = Math.floor( ( E.rect[ 0 ] - pad ) / square ) * square, y0 = Math.floor( ( E.rect[ 1 ] - pad ) / square ) * square;
	const nx = Math.ceil( ( E.rect[ 2 ] + pad - x0 ) / square ), ny = Math.ceil( ( E.rect[ 3 ] + pad - y0 ) / square );
	const of = new Map(); // square -> { bits, z0, z1 }
	pts.forEach( ( [ x, y, g ], i ) => {

		for ( let b = Math.floor( ( y - pad - y0 ) / square ); b <= Math.floor( ( y + pad - y0 ) / square ); b ++ )
			for ( let a = Math.floor( ( x - pad - x0 ) / square ); a <= Math.floor( ( x + pad - x0 ) / square ); a ++ ) {

				if ( a < 0 || b < 0 || a >= nx || b >= ny ) continue;
				let s = of.get( b * nx + a );
				if ( ! s ) of.set( b * nx + a, s = { bits: new Uint8Array( nb ), z0: Infinity, z1: - Infinity } );
				for ( let k = 0; k < nb; k ++ ) s.bits[ k ] |= seen[ i * nb + k ];
				s.z0 = Math.min( s.z0, g ); s.z1 = Math.max( s.z1, g );

			}

	} );
	const sq = new Uint16Array( nx * ny ), bits = new Uint8Array( of.size * nb ), z = [];
	let e = 0, shown = 0, added = 0;
	for ( const [ at, s ] of [ ...of ].sort( ( a, b ) => a[ 0 ] - b[ 0 ] ) ) {

		const a = at % nx, b = Math.floor( at / nx ), sx = x0 + a * square, sy = y0 + b * square;
		C.forEach( ( c, n ) => {

			// (c.box: x and z of three's frame, z = - y)
			const dx = Math.max( c.box[ 0 ] - ( sx + square ), 0, sx - c.box[ 2 ] ), dy = Math.max( c.box[ 1 ] + sy, 0, - ( sy + square ) - c.box[ 3 ] );
			// (and the skyline - the cells that only have a lowest level, far outside the walks - in every set: such a cell
			// shows through a gap between two roofs from a hand's breadth of positions, which a grid of eye points misses -
			// 18 pixels of one, 93 m off, from one place on the terrace - and all fifteen together are 17,600 triangles)
			if ( ( Math.hypot( dx, dy ) < near || ! ( c.m.mid || c.m.near ) ) && ! ( s.bits[ n >> 3 ] >> ( n & 7 ) & 1 ) ) { s.bits[ n >> 3 ] |= 1 << ( n & 7 ); added ++; }
			if ( s.bits[ n >> 3 ] >> ( n & 7 ) & 1 ) { shown ++; count[ n + 1 ] ++; }

		} );
		bits.set( s.bits, e * nb );
		z.push( + ( s.z0 + eye - slack ).toFixed( 2 ), + ( s.z1 + eye + slack ).toFixed( 2 ) );
		sq[ at ] = ++ e;

	}

	Object.assign( stats, { squares: of.size, cells: C.length, seen: + ( shown / ( of.size * C.length ) ).toFixed( 4 ), near_added: added, never: C.filter( ( c, n ) => ! count[ n + 1 ] ).length,
		stray: count[ 0 ], seconds: + ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 ) } );
	return { v: 1, baked: town.bake.baked, keys: C.map( ( c ) => c.m.k ), s: square, x0, y0, nx, ny, nb, z, sq: b64( new Uint8Array( sq.buffer ) ), bits: b64( bits ), stats };

}

// The test of the sets: from `n` places a walker can be - anywhere, not on the grid - the town as the page's tier
// shows it there (its levels, its parts, settled) in cell colours at `face` pixels, against the set of the square.
// -> { points, misses: [ { at, cell, pixels, of } .. ] (cells in the picture that the set leaves out), .. }
export async function checkPVS( town, { n = 60, face = 512, eye = 1.65, seed = 1, on = () => {} } = {} ) {

	const C = town.chunks, nav = town.nav, rig = idRig( town, face ), t0 = performance.now();
	let s = seed >>> 0 || 1;
	const rnd = () => ( ( s = Math.imul( s ^ s >>> 15, 0x2c1b3c6d ) + 0x9e3779b9 | 0, ( ( s ^ s >>> 13 ) >>> 0 ) / 4294967296 ) );
	const area = nav.regions.map( ( r ) => ( r.rect[ 2 ] - r.rect[ 0 ] ) * ( r.rect[ 3 ] - r.rect[ 1 ] ) ), total = area.reduce( ( a, b ) => a + b, 0 );
	const cam = town.eng.camera, res = { points: 0, no_set: 0, misses: [], hidden: 0, hidden_tris: 0, drawn_tris: 0, worst: 0 };
	await town.pvsReady;
	if ( ! town.pvs ) return { ...res, error: 'the bake has no visibility sets' };
	const made = new Map(); // a level's mesh -> its copy in cell colours (once: every copy is render objects the renderer keeps)
	const held = new Map(); // .. -> [ its cell, its level ]: a copy goes when the town lets the level go, or it would keep its geometry
	for ( let i = 0; i < n; i ++ ) {

		let t = rnd() * total, r = nav.regions[ 0 ];
		for ( let k = 0; k < area.length; k ++ ) { if ( t < area[ k ] ) { r = nav.regions[ k ]; break; } t -= area[ k ]; }
		const p = { x: r.rect[ 0 ] + rnd() * ( r.rect[ 2 ] - r.rect[ 0 ] ), y: r.rect[ 1 ] + rnd() * ( r.rect[ 3 ] - r.rect[ 1 ] ) }, x0 = p.x, y0 = p.y;
		const h = nav.heightOf( r, p.x, p.y );
		for ( let k = 0; k < 4 && nav.resolve( p, h ); k ++ );
		if ( Math.hypot( p.x - x0, p.y - y0 ) > 0.5 ) { i --; continue; } // (inside a house: another)
		const g = nav.groundAt( p.x, p.y, h + 0.05 ) ?? h;
		cam.position.set( p.x, g + eye, - p.y );
		cam.rotation.set( 0, rnd() * Math.PI * 2, 0, 'YXZ' );
		cam.updateMatrixWorld();
		await town.settle( cam );
		const hid = town._occluded( cam );
		res.points ++;
		if ( ! hid ) { res.no_set ++; continue; }
		const groups = [];
		C.forEach( ( c, k ) => {

			const mesh = c.mesh[ c.level ];
			if ( ! mesh ) return;
			let q = made.get( mesh );
			if ( q ) { mesh.userData.all.forEach( ( o, j ) => { q.children[ j ].visible = o.visible; } ); rig.scene.add( q ); } else { made.set( mesh, q = rig.again( mesh, k ) ); held.set( mesh, [ c, c.level ] ); }
			groups.push( q );
			const tris = mesh.userData.shown ?? mesh.userData.tris;
			res.drawn_tris += tris;
			if ( hid[ k ] ) { res.hidden ++; res.hidden_tris += tris; }

		} );
		rig.begin();
		const turn = rnd() * Math.PI / 2, px = await rig.shoot( p.x, p.y, g + eye, turn );
		rig.end();
		for ( const q of groups ) rig.scene.remove( q );
		for ( const [ mesh, [ c, l ] ] of held ) if ( c.mesh[ l ] !== mesh ) { rig.drop( made.get( mesh ) ); made.delete( mesh ); held.delete( mesh ); }
		const count = new Uint32Array( C.length + 1 );
		for ( let k = 0; k < px.length; k += 4 ) { const id = px[ k ] | px[ k + 1 ] << 8; if ( id <= C.length ) count[ id ] ++; }
		const missed = [];
		C.forEach( ( c, k ) => {

			if ( ! hid[ k ] || ! count[ k + 1 ] ) return;
			missed.push( { at: [ + p.x.toFixed( 2 ), + p.y.toFixed( 2 ), + g.toFixed( 2 ) ], cell: c.m.k, pixels: count[ k + 1 ], d: + c.d.toFixed( 1 ), level: town.L[ c.level ], k } );
			res.worst = Math.max( res.worst, count[ k + 1 ] );

		} );
		if ( missed.length ) {

			// to tell why: what stands in those pixels when every cell is at its lowest level (the cell that hides it
			// there, and the level the page has that one at), and the levels the page holds
			const lows = C.map( ( c, k ) => {

				const mesh = c.mesh[ 0 ];
				let q = made.get( mesh );
				if ( q ) rig.scene.add( q ); else { made.set( mesh, q = rig.again( mesh, k ) ); held.set( mesh, [ c, 0 ] ); }
				return q;

			} );
			rig.begin();
			const low = await rig.shoot( p.x, p.y, g + eye, turn );
			rig.end();
			for ( const q of lows ) rig.scene.remove( q );
			const held_ = town.L.map( ( l, i ) => l + ' ' + C.filter( ( c ) => c.level === i ).length ).join( ', ' );
			// .. and where: the direction it shows in, and the cells around it in a narrow view along that
			for ( const q of groups ) rig.scene.add( q );
			rig.begin();
			for ( const m of missed ) {

				let i0 = - 1;
				for ( let k = 0, j = 0; k < px.length; k += 4 ) if ( ( px[ k ] | px[ k + 1 ] << 8 ) === m.k + 1 && j ++ >= ( m.pixels >> 1 ) ) { i0 = k >> 2; break; }
				const w = await rig.spot( p.x, p.y, g + eye, turn, i0, m.k + 1 );
				m.eye = [ + p.x.toFixed( 3 ), + p.y.toFixed( 3 ), + ( g + eye ).toFixed( 3 ) ];
				if ( w ) { m.dir = w.dir.map( ( v ) => + v.toFixed( 5 ) ); m.around = [ ...w.frame ].sort( ( a, b ) => b[ 1 ] - a[ 1 ] ).slice( 0, 6 ).map( ( [ id, n ] ) => `${C[ id - 1 ].m.k} ${n} (${town.L[ C[ id - 1 ].level ]}, ${C[ id - 1 ].d.toFixed( 0 )} m)` ); }

			}

			rig.end();
			for ( const q of groups ) rig.scene.remove( q );
			for ( const m of missed ) {

				const by = new Map();
				for ( let k = 0; k < px.length; k += 4 ) if ( ( px[ k ] | px[ k + 1 ] << 8 ) === m.k + 1 ) { const id = low[ k ] | low[ k + 1 ] << 8; by.set( id, ( by.get( id ) || 0 ) + 1 ); }
				m.behind = [ ...by ].sort( ( a, b ) => b[ 1 ] - a[ 1 ] ).map( ( [ id, n ] ) => id === 0 ? `sky ${n}` : `${C[ id - 1 ].m.k} ${n} (has ${town.L[ C[ id - 1 ].level ]}, wants ${town.L[ C[ id - 1 ].want ]}, ${C[ id - 1 ].d.toFixed( 0 )} m)` );
				m.held = held_; m.pending = town.pending;
				delete m.k;
				res.misses.push( m );

			}

		}
		if ( i % 10 === 0 ) on( `check ${i} / ${n}, ${res.misses.length} misses, ${( ( performance.now() - t0 ) / 1000 ).toFixed( 0 )} s (${made.size} copies, ${( town.bytes / 1e6 ).toFixed( 0 )} MB of levels, heap ${performance.memory ? ( performance.memory.usedJSHeapSize / 1e6 ).toFixed( 0 ) : '?'} MB)` );

	}

	for ( const q of made.values() ) rig.drop( q );
	rig.dispose();
	res.pixels = rig.W * rig.H;
	res.cells = C.length;
	res.seconds = + ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 );
	return res;

}


// The holes of the reduced levels: places where the top level of the bake shows a surface and a lower level lets the
// eye through to something well behind it (or to the sky). From `limit` eye points of the walkable ground (a grid of
// `step` m, the walker's eye height) the whole town is drawn at its top level and then at each level below, a cube's
// five faces of floats each - the place in the world in every pixel - and the two are compared pixel by pixel:
// a hole is where the level's surface is more than max( tol, rel x distance ) behind the top level's, and the top
// level stands as close as that `erode` pixels to either side as well (so the rim of a thing that shrank, a rail or
// a pole that went, do not count: only what opens inside a surface). What comes back names, per level and cell, the
// places on the top level's surface that show through (one per `voxel` m), which is what the packer mends the level
// with (holes.json beside the raw bake: OPT.holes of tools/webbake_pack.mjs), and the worst view of each.
// Leaf cards hide what is behind them but are no surface to mend (a crown's cards differ from level to level), and a
// cell counts for a level only from as far as some tier may show it at that level (`from`, metres to the cell).
// -> { top, levels: { <level>: { pixels, of, cells: { <cell>: { n, pts: [ x, y, z .. ] (three's frame, mm), view } } } } }
export async function holesPVS( town, { limit = 600, step = 1, face = 512, eye = 1.65, levels = null, top = null, tol = 0.4, rel = 0.03, erode = 2, voxel = 0.06, from = { lo: 20, mid: 12, near: 12, ring: 0 }, through = true, deep = 1.5, on = () => {} } = {} ) {

	// (through: a place to mend is only one where the sky shows behind, or another cell more than `deep` m further off -
	// a window frame gone with its own room behind it is the level doing its job, and so is one whose window is a thing
	// of its own in the next cell, with the house's room right behind it)
	const C = town.chunks, t0 = performance.now(), N = face;
	const rig = idRig( town, N, true ), { W, H } = rig;
	let pts = eyePoints( town.nav, step ).points;
	if ( limit > 0 && pts.length > limit ) { const k = pts.length / limit; pts = Array.from( { length: limit }, ( _, i ) => pts[ Math.floor( i * k ) ] ); }
	const have = LEVELS.filter( ( l ) => C.some( ( c ) => c.m[ l ] ) );
	const topName = top || have[ have.length - 1 ], names = ( levels || have ).filter( ( l ) => l !== topName && have.includes( l ) );
	const at = {}, built = [], res = { top: topName, points: pts.length, face, tol, rel, erode, voxel, through, deep, levels: {} };
	const offs = [ - erode, erode, - erode * W, erode * W ].map( ( o ) => o * 4 );

	try {

		for ( const name of [ topName, ...names ] ) {

			// the whole town at this level (a cell without it: at the one below that it has), as bakePVS has it
			on( `level ${name}: loading` );
			const groups = at[ name ] = new Array( C.length );
			let next = 0;
			const one = async () => {

				for ( ;; ) {

					const n = next ++;
					if ( n >= C.length ) return;
					const c = C[ n ];
					let l = LEVELS.indexOf( name );
					if ( name === 'ring' && ! c.m.ring ) l = LEVELS.indexOf( 'near' );
					while ( l > 0 && ! c.m[ LEVELS[ l ] ] ) l --;
					if ( l === 0 ) { groups[ n ] = rig.again( c.mesh[ 0 ], n ); groups[ n ].visible = false; continue; }
					const mesh = town._build( await town._levelBytes( c, LEVELS[ l ] ), c.m[ LEVELS[ l ] ], null, 1 );
					built.push( mesh );
					groups[ n ] = rig.again( mesh, n );
					groups[ n ].visible = false;

				}

			};

			await Promise.all( Array.from( { length: 6 }, one ) );
			if ( name !== topName ) res.levels[ name ] = { pixels: 0, through: 0, of: 0, cells: {} };

		}

		const show = ( name, v ) => { for ( const g of at[ name ] ) g.visible = v; };
		const found = {}; // level -> cell -> { n, vox: Map, view }
		for ( const name of names ) found[ name ] = new Map();
		rig.begin();
		for ( let i = 0; i < pts.length; i ++ ) {

			const [ x, y, g ] = pts[ i ], z = g + eye, turn = ( ( i * 0.61803398875 ) % 1 ) * Math.PI / 2;
			const ex = x, ey = z, ez = - y; // (three's frame)
			const far = C.map( ( c ) => town._dist( c, x, - y ) );
			show( topName, true );
			const T = await rig.shoot( x, y, z, turn );
			show( topName, false );
			for ( const name of names ) {

				show( name, true );
				const P = await rig.shoot( x, y, z, turn );
				show( name, false );
				const L = res.levels[ name ], F = found[ name ], lim = from[ name ] || 0, here = new Map(); // (here: cell -> [ pixels, sum x y z ] from this eye point)
				for ( let f = 0; f < FACES; f ++ ) {

					const fx = ( f % COLS ) * N, fy = Math.floor( f / COLS ) * N;
					for ( let v = erode; v < N - erode; v ++ ) for ( let u = erode; u < N - erode; u ++ ) {

						const k = ( ( fy + v ) * W + fx + u ) * 4, a = T[ k + 3 ];
						if ( ! ( a > 0 ) || far[ Math.round( a ) - 1 ] < lim ) continue;
						L.of ++;
						const dT = Math.hypot( T[ k ] - ex, T[ k + 1 ] - ey, T[ k + 2 ] - ez );
						const dP = P[ k + 3 ] ? Math.hypot( P[ k ] - ex, P[ k + 1 ] - ey, P[ k + 2 ] - ez ) : Infinity;
						const m = Math.max( tol, rel * dT );
						if ( dP <= dT + m ) continue;
						let inside = true;
						for ( const o of offs ) {

							const j = k + o;
							if ( ! T[ j + 3 ] || Math.hypot( T[ j ] - ex, T[ j + 1 ] - ey, T[ j + 2 ] - ez ) + m >= dP ) { inside = false; break; }

						}

						if ( ! inside ) continue;
						L.pixels ++;
						const id = Math.round( a ) - 1, thru = ! P[ k + 3 ] || ( Math.round( Math.abs( P[ k + 3 ] ) ) !== id + 1 && dP > dT + Math.max( deep, m ) );
						if ( thru ) L.through ++;
						else if ( through ) continue;
						let e = F.get( id );
						if ( ! e ) F.set( id, e = { n: 0, vox: new Map(), view: null } );
						e.n ++;
						const key = Math.round( T[ k ] / voxel ) + ',' + Math.round( T[ k + 1 ] / voxel ) + ',' + Math.round( T[ k + 2 ] / voxel );
						if ( ! e.vox.has( key ) ) e.vox.set( key, [ T[ k ], T[ k + 1 ], T[ k + 2 ] ] );
						let h = here.get( id );
						if ( ! h ) here.set( id, h = [ 0, 0, 0, 0 ] );
						h[ 0 ] ++; h[ 1 ] += T[ k ]; h[ 2 ] += T[ k + 1 ]; h[ 3 ] += T[ k + 2 ];

					}

				}

				for ( const [ id, h ] of here ) {

					const e = F.get( id );
					if ( ! e.view || h[ 0 ] > e.view.n ) e.view = { n: h[ 0 ], eye: [ + x.toFixed( 2 ), + y.toFixed( 2 ), + z.toFixed( 2 ) ], target: [ + ( h[ 1 ] / h[ 0 ] ).toFixed( 2 ), + ( - h[ 3 ] / h[ 0 ] ).toFixed( 2 ), + ( h[ 2 ] / h[ 0 ] ).toFixed( 2 ) ] }; // (town frame)

				}

			}

			if ( i % 20 === 0 ) on( `holes ${i} / ${pts.length}: ${names.map( ( l ) => l + ' ' + res.levels[ l ].through + ' / ' + res.levels[ l ].pixels ).join( ', ' )} pixels through / behind, ${( ( performance.now() - t0 ) / 1000 ).toFixed( 0 )} s` );

		}

		for ( const name of names ) for ( const [ id, e ] of found[ name ] ) {

			const pts = [];
			for ( const p of e.vox.values() ) pts.push( Math.round( p[ 0 ] * 1000 ), Math.round( p[ 1 ] * 1000 ), Math.round( p[ 2 ] * 1000 ) );
			res.levels[ name ].cells[ C[ id ].m.k ] = { n: e.n, pts, view: e.view };

		}

	} finally {

		rig.end();
		for ( const name in at ) for ( const g of at[ name ] ) if ( g ) rig.drop( g );
		for ( const mesh of built ) for ( const o of mesh.userData.all ) o.geometry.dispose();
		rig.dispose();

	}

	res.seconds = + ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 );
	return res;

}
