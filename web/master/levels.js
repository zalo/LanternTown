// Which level of detail each cell of the town shows from where the camera stands (the High and Medium presets).
//
// The policy is the walk page's (engine/lite.js, BakedTown.update), unchanged:
// - a level is wanted inside its radius (the tier's `mid`, `near`, `fine`) and kept a fifth and 2 m further out
//   than it came, so a step back and forth swaps nothing;
// - `resident` bounds what the GPU holds: over it the farthest cells go down a level;
// - the view budget (budget.js) bounds what a frame draws, whichever way the walker turns next;
// - until a wanted level is in, a cell shows the best one there is;
// - a change of level is a dithered fade of `fade` seconds, both levels drawn; the cells going down take their turn
//   first, since the dear level draws for as long as they wait;
// - `fineR` tells the materials how far out the top level reaches, so its per-pixel light ends where it does.
//
// The LOD is only the request trigger. The loader delivers each cell as a THREE.LOD that fetches a level (a byte
// range of the one file) when the LOD first selects it, and a LOD selects by one distance per level. The town's
// choice is not a function of one distance - it depends on the budgets, on what the cell had before, on its
// neighbours - and it shows two levels at once during a fade. The loader has no other way to be asked for a level
// than update() selecting it, so on every pass the level distances of each LOD are set so that update() selects
// exactly the level the policy wants (see _request), which queues it or lets a request that is no longer wanted
// lapse; then every level group is made visible again. What is drawn is decided here alone, by the `visible` of the
// level objects themselves (and their meshes' layers, see culling.js); `autoUpdate` is off, so the renderer never
// selects.
//
// What the loader does not do is made up for here: it does not say when a level failed (an awaited level counts
// as failed after PATIENCE seconds in which nothing arrived), and it cannot unload (a level that is left gives up
// its copy on the GPU, see free(); the arrays stay with the loader).

import { InterleavedBuffer, InterleavedBufferAttribute, Matrix4 } from 'three';
import { ViewBudget } from './budget.js';

const LEVELS = [ 'lo', 'mid', 'near', 'fine' ], N = LEVELS.length;
const NEVER = Number.MAX_VALUE; // a level distance no camera is at (LOD's hysteresis term makes NaN of Infinity)
const PATIENCE = 20; // s
const STEP = 0.75; // m the camera moves before the policy looks again
const SLACK = 0.08; // of the budget: what fades may draw on top of it

// from a point on the ground to a box [ x0, z0, x1, z1 ]
function distance( box, x, z ) {

	return Math.hypot( Math.max( box[ 0 ] - x, 0, x - box[ 2 ] ), Math.max( box[ 1 ] - z, 0, z - box[ 3 ] ) );

}

// Give up a level's copy on the GPU: three uploads the arrays again if the level is shown once more.
//
// Disposing the geometries is all it should take. But three (r186) does not get a disposed interleaved attribute
// back on the GPU: its WebGPU back-end keeps its record of the buffer it destroyed and binds that, its WebGL one
// uploads only the first attribute of a buffer and draws without the others. So each geometry then gets new
// attribute objects over the same arrays - nothing is copied - which three has never seen.
//
// three's WebGL back-end also never deletes a vertex array object, and a deleted buffer lives on for as long as
// one points at it: a walk would leave every level it ever showed on the GPU. The ones over this level's buffers
// are deleted here.
function free( renderer, mesh ) {

	const b = renderer.backend, gl = b.isWebGLBackend && b.vaoCache ? b.gl : null, ids = new Set();
	const fresh = new Map(); // attribute or buffer -> its successor (the meshes of a level share theirs)
	const successor = ( old, make ) => fresh.get( old ) || fresh.set( old, make() ).get( old );
	const renew = ( a ) => ! a.isInterleavedBufferAttribute ? a : successor( a, () => {

		const data = successor( a.data, () => new InterleavedBuffer( a.data.array, a.data.stride ).setUsage( a.data.usage ) );
		const next = new InterleavedBufferAttribute( data, a.itemSize, a.offset, a.normalized );
		next.name = a.name;
		return next;

	} );

	if ( gl ) mesh.traverse( ( o ) => {

		if ( o.geometry ) for ( const a of Object.values( o.geometry.attributes ) ) { const id = b.get( a ).id; if ( id !== undefined ) ids.add( String( id ) ); }

	} );
	mesh.traverse( ( o ) => o.geometry && o.geometry.dispose() );
	mesh.traverse( ( o ) => { // (after all of them: three finds what to delete through the attributes)

		if ( o.geometry ) for ( const k in o.geometry.attributes ) o.geometry.setAttribute( k, renew( o.geometry.attributes[ k ] ) );

	} );
	if ( ids.size ) for ( const key of Object.keys( b.vaoCache ) ) {

		if ( ! key.split( ':' ).some( ( id ) => ids.has( id ) ) ) continue;
		const vao = b.vaoCache[ key ];
		if ( b.state.currentVAO === vao ) b.state.resetVertexState();
		gl.deleteVertexArray( vao );
		delete b.vaoCache[ key ];

	}

}

// what the policy keeps of a cell (m: its entry in the manifest, one record per level)
function record( cell ) {

	const m = cell.m, box = [ m.lo.bmin[ 0 ], m.lo.bmin[ 2 ], m.lo.bmax[ 0 ], m.lo.bmax[ 2 ] ];
	for ( const l of LEVELS.slice( 1 ) ) if ( m[ l ] ) {

		box[ 0 ] = Math.min( box[ 0 ], m[ l ].bmin[ 0 ] ); box[ 1 ] = Math.min( box[ 1 ], m[ l ].bmin[ 2 ] );
		box[ 2 ] = Math.max( box[ 2 ], m[ l ].bmax[ 0 ] ); box[ 3 ] = Math.max( box[ 3 ], m[ l ].bmax[ 2 ] );

	}

	// per level, for the view budget: centre (x, z), radius, triangles and centre height of its bounding sphere
	const sph = LEVELS.map( ( l ) => m[ l ] && [ ( m[ l ].bmin[ 0 ] + m[ l ].bmax[ 0 ] ) / 2, ( m[ l ].bmin[ 2 ] + m[ l ].bmax[ 2 ] ) / 2,
		0.5 * Math.hypot( m[ l ].bmax[ 0 ] - m[ l ].bmin[ 0 ], m[ l ].bmax[ 1 ] - m[ l ].bmin[ 1 ], m[ l ].bmax[ 2 ] - m[ l ].bmin[ 2 ] ), m[ l ].ni / 3,
		( m[ l ].bmin[ 1 ] + m[ l ].bmax[ 1 ] ) / 2 ] );
	// the clusters of the top level's opaque range, each a mesh of its own that is drawn down to a part size: sphere,
	// box on the ground, [ part size, index count down to it ] steps; rest: the triangles outside them
	const t = m[ LEVELS[ N - 1 ] ];
	const top = ( t && t.parts || [] ).map( ( [ sub, steps, b ] ) => ( {
		sph: [ ( b[ 0 ] + b[ 3 ] ) / 2, ( b[ 2 ] + b[ 5 ] ) / 2, 0.5 * Math.hypot( b[ 3 ] - b[ 0 ], b[ 4 ] - b[ 1 ], b[ 5 ] - b[ 2 ] ), 0, ( b[ 1 ] + b[ 4 ] ) / 2 ],
		box: [ b[ 0 ], b[ 2 ], b[ 3 ], b[ 5 ] ], steps, full: t.sub[ sub ][ 2 ], n: t.sub[ sub ][ 2 ] } ) );
	const rest = t ? t.ni / 3 - top.reduce( ( n, q ) => n + q.full / 3, 0 ) : 0;
	const g = /^c_(-?\d+)_(-?\d+)$/.exec( m.k ); // (the cell of the grid it is: its things may reach into the next)
	// mesh: the level objects that have arrived; level: the one shown; want: the one the policy chose; failed: bits
	// of the levels given up; awaited, since: the level waited for, and from when
	return { cell, m, box, sph, top, rest, mesh: LEVELS.map( () => null ), level: 0, want: 0, was: 0, d: 0, fade: null, failed: 0, awaited: 0, since: 0,
		gx: g ? + g[ 1 ] : NaN, gz: g ? + g[ 2 ] : NaN };

}

export class TownLevels {

	// cells: [ { k, m, lod, lodIndex } ] of the plugin; tier: the preset's row; materials: for U.fineR; renderer: see
	// free(); mirror: { wide, level } of the planar mirror image or null, cellSize: the cell grid (m) - see budget.js
	constructor( { cells, tier, materials, renderer, mirror = null, cellSize = 16 } ) {

		this.tier = tier;
		this.R = [ Infinity, tier.mid, tier.near, tier.fine ]; // the distance inside which each level is shown
		this.fineR = materials.U.fineR;
		this.renderer = renderer;
		this.budget = new ViewBudget( tier, LEVELS, { mirror, cellSize } );
		this.chunks = cells.map( record );
		this.byCell = new Map( this.chunks.map( ( c ) => [ c.cell, c ] ) );
		this.loTris = this.chunks.reduce( ( n, c ) => n + c.m.lo.ni / 3, 0 );
		this.baseLeft = this.chunks.length; // lowest levels still to come
		this.fades = new Set();
		this.waiters = [];
		this.warnings = [];
		this.eye = { matrixWorld: new Matrix4(), zoom: 1 }; // the camera a LOD is asked through (see _request)
		this.swapped = []; // (see mirrorView)
		this._arrived = 0;
		this._dirty = true;

	}

	// the loader has built level `name` of a cell (`object`: its meshes, see the plugin)
	arrive( cell, name, object ) {

		const c = this.byCell.get( cell ), l = LEVELS.indexOf( name );
		c.mesh[ l ] = object;
		c.failed &= ~ ( 1 << l ); // (late is not lost)
		object.visible = l === 0 && c.level === 0;
		object.userData.held = l === 0; // on the GPU, or about to be: shown since it was last freed (the lowest always)
		if ( l === 0 ) this.baseLeft --;
		this._arrived = performance.now() / 1000;
		this._dirty = true;
		for ( const f of this.waiters.splice( 0 ) ) f();

	}

	// choose the level of every cell for this camera, ask for what is missing, swap and fade
	update( camera, { force = false } = {} ) {

		const x = camera.position.x, z = camera.position.z, clock = performance.now() / 1000, at = this._at;
		if ( this.fades.size ) this._fadeTick( clock, force );
		if ( ! force ) this._fineReach( x, z, clock, false );
		this._giveUp( clock );
		if ( ! force && ! this._dirty && at && Math.hypot( x - at[ 0 ], z - at[ 1 ] ) < STEP && camera.aspect === at[ 2 ] ) return;
		this._at = [ x, z, camera.aspect ]; this._dirty = false;
		const order = this._choose( x, z );
		this.budget.plan( camera, this.chunks, order );
		this._request( clock );
		this._show( clock, force );
		if ( force ) this._fineReach( x, z, clock, true );

	}

	// the level each cell asks for by its distance, under the ceiling of what the GPU holds -> those above their
	// lowest, nearest first
	_choose( x, z ) {

		const T = this.tier, R = this.R, order = [];
		for ( const c of this.chunks ) {

			c.d = distance( c.box, x, z );
			const keep = ( r, l ) => c.d < ( c.want >= l ? r + r * 0.2 + 2 : r );
			let want = 0;
			for ( let l = N - 1; l > 0 && ! want; l -- ) if ( c.m[ LEVELS[ l ] ] && R[ l ] > 0 && keep( R[ l ], l ) ) want = l;
			while ( want > 0 && ( c.failed & ( 1 << want ) || ! c.m[ LEVELS[ want ] ] ) ) want --;
			c.was = c.want; c.want = want;
			if ( want > 0 ) order.push( c );

		}

		order.sort( ( a, b ) => a.d - b.d );
		// what the GPU holds is every wanted level whole, in view or not (the view budget only narrows what is drawn
		// of it): over the ceiling, first the cells that are only still held for being left late go down a level,
		// farthest first, then the farthest of the top level and of the one below it
		let held = this.loTris;
		for ( const c of order ) held += c.m[ LEVELS[ c.want ] ].ni / 3;
		const down = ( c ) => {

			held -= c.m[ LEVELS[ c.want ] ].ni / 3;
			do c.want --; while ( c.want > 0 && ( c.failed & ( 1 << c.want ) || ! c.m[ LEVELS[ c.want ] ] ) );
			if ( c.want > 0 ) held += c.m[ LEVELS[ c.want ] ].ni / 3;

		};

		for ( let i = order.length - 1; i >= 0 && held > T.resident; i -- ) {

			const c = order[ i ];
			while ( c.want > 0 && held > T.resident && c.d >= R[ c.want ] ) down( c );

		}

		for ( let top = N - 1; top >= 2; top -- ) for ( let i = order.length - 1; i >= 0 && held > T.resident; i -- ) if ( order[ i ].want === top ) down( order[ i ] );
		return order.filter( ( c ) => c.want > 0 );

	}

	// Tell the loader what is wanted. The levels of a LOD are finest first: with distance 0 down to the wanted one and
	// NEVER beyond it, update() selects that one, whatever the distance - and queues it if it is not in, or lets a
	// queued request for another lapse. The loader takes its queue nearest first, by the camera's distance to the
	// LOD's origin (the middle of the cell's bounds, at whatever height); the policy's distance is to the cell's box
	// on the ground. So the LOD is asked through a stand-in camera that is as far from its origin as the policy says
	// the cell is, and the queue follows the policy.
	_request( clock ) {

		for ( const c of this.chunks ) {

			const lod = c.cell.lod; // (none: a cell of one level, or a file whose scene is still to come)
			const missing = lod && c.want > 0 && ! c.mesh[ c.want ] ? c.want : 0;
			if ( missing !== c.awaited ) { c.awaited = missing; c.since = clock; }
			if ( ! lod ) continue;
			const k = c.cell.lodIndex[ LEVELS[ c.want ] ], levels = lod.levels, e = lod.matrixWorld.elements;
			for ( let i = 0; i < levels.length; i ++ ) { levels[ i ].distance = i <= k ? 0 : NEVER; levels[ i ].hysteresis = 0; }
			this.eye.matrixWorld.setPosition( e[ 12 ] + c.d, e[ 13 ], e[ 14 ] );
			lod.update( this.eye );
			for ( const level of levels ) level.object.visible = true; // (the loader's own choice of what to show)

		}

	}

	// A level that fails to load never arrives, and the loader does not try again: one that is still awaited PATIENCE
	// seconds after it was asked for and after anything last arrived (a long queue on a slow line is not a failure)
	// is given up, so the policy settles for the level below.
	_giveUp( clock ) {

		for ( const c of this.chunks ) {

			if ( ! c.awaited || c.mesh[ c.awaited ] || clock - Math.max( c.since, this._arrived ) < PATIENCE ) continue;
			c.failed |= 1 << c.awaited;
			this.warnings.push( `town: ${c.m.k} ${LEVELS[ c.awaited ]} did not arrive` );
			c.awaited = 0;
			this._dirty = true;

		}

	}

	// show the wanted level of every cell or, until it is in, the best one there is; start the fades
	_show( clock, force ) {

		const T = this.tier;
		// (a fade draws two levels of a cell. A cell that waits to go down a level draws the dear one all the while, so
		// holding it back saves nothing - it only keeps the picture over the budget for longer. So the cells going down
		// take their turn first: what their fades add, the cheap level, is kept under 8 % of the budget. The ones going
		// up wait - nearest first, at the level they have - while what is drawn, fades and cells still to come down
		// included, would pass what is wanted by more than those 8 %)
		let under = 0, drawn = 0, wanted = 0;
		for ( const c of this.fades ) if ( c.fade.down ) under += c.fade.cost;
		if ( ! force ) for ( const c of this.chunks ) {

			for ( const m of c.mesh ) if ( m && m.visible ) drawn += m.userData.shown ?? m.userData.tris;
			const m = c.mesh[ c.want ] || c.mesh[ c.level ];
			if ( m ) wanted += m.userData.tris;

		}

		const turn = force ? this.chunks : this.chunks.slice().sort( ( a, b ) => ( a.want < a.level ? 0 : 1 ) - ( b.want < b.level ? 0 : 1 ) || a.d - b.d );
		for ( const c of turn ) {

			if ( c.fade ) continue; // (one change at a time: the next starts when this one is through)
			// the one shown now, else the nearest below, else above
			let show = c.want;
			if ( ! c.mesh[ show ] ) {

				if ( c.mesh[ c.level ] ) show = c.level;
				else {

					show = 0;
					for ( let l = c.want - 1; l > 0 && ! show; l -- ) if ( c.mesh[ l ] ) show = l;
					for ( let l = c.want + 1; l < N && ! show; l ++ ) if ( c.mesh[ l ] ) show = l;

				}

			}

			const a = c.mesh[ c.level ], b = c.mesh[ show ];
			if ( show !== c.level && ! force && a && b && a.visible ) {

				const ca = a.userData.shown ?? a.userData.tris, cb = b.userData.tris;
				const down = cb < ca, cost = down ? cb : ca;
				if ( down ) {

					if ( under > 0 && under + cost > T.tris * SLACK ) continue;
					under += cost;

				} else if ( this.fades.size > 0 && drawn + cb > wanted + T.tris * SLACK ) continue;

				drawn += cb;
				c.fade = { from: c.level, to: show, t0: clock, cost, down };
				this.fades.add( c );
				for ( const o of a.userData.draws ) { o.material = o.userData.fmat; o.userData.fade = - 1e-3; }
				for ( const o of b.userData.draws ) { o.material = o.userData.fmat; o.userData.fade = 1e-3; }
				b.visible = b.userData.held = true;
				this._parts( c, b );
				c.level = show;
				continue;

			}

			for ( let l = 0; l < N; l ++ ) {

				const mesh = c.mesh[ l ];
				if ( ! mesh ) continue;
				mesh.visible = l === show;
				if ( l === show ) { mesh.userData.held = true; this._parts( c, mesh ); }
				// (a level that is neither shown nor wanted goes; not twice without having been shown in between)
				else if ( l > 0 && l !== c.want && mesh.userData.held ) { mesh.userData.held = false; free( this.renderer, mesh ); }

			}

			c.level = show;

		}

	}

	// the top level: each cluster down to the part size the budget gave it
	_parts( c, mesh ) {

		const u = mesh.userData;
		if ( ! u.parts ) return;
		u.shown = c.rest;
		u.parts.forEach( ( part, i ) => {

			const n = c.top[ i ].n;
			part.geometry.setDrawRange( part.userData.start, n );
			part.visible = n > 0;
			u.shown += n / 3;

		} );

	}

	// a change of level in progress: both levels draw, dithered against each other, for `fade` seconds
	_fadeTick( clock, finish ) {

		for ( const c of this.fades ) {

			const f = c.fade, t = finish ? 1 : ( clock - f.t0 ) / this.tier.fade;
			const a = c.mesh[ f.from ], b = c.mesh[ f.to ];
			if ( t < 1 ) {

				const k = Math.max( 1e-3, t * t * ( 3 - 2 * t ) );
				for ( const o of b.userData.draws ) o.userData.fade = k;
				for ( const o of a.userData.draws ) o.userData.fade = - k;
				continue;

			}

			for ( const m of [ a, b ] ) for ( const o of m.userData.draws ) { o.material = o.userData.mat; o.userData.fade = 1; }
			a.visible = false;
			c.fade = null;
			this.fades.delete( c );
			this._dirty = true; // (the level it left can go now)

		}

	}

	// fineR: how far from the camera every cell shows its top level - the distance to the nearest cell that could
	// have it and does not (too far, still loading, refused by the budget, in a fade), `fine` at most. Eased, quicker
	// down than up, so a cell the budget takes the level from has lost the per-pixel light before its fade is half
	// through.
	_fineReach( x, z, clock, snap ) {

		const u = this.fineR;
		let r = this.tier.fine;
		for ( const c of this.chunks ) {

			if ( ! c.m[ LEVELS[ N - 1 ] ] || c.failed & ( 1 << ( N - 1 ) ) || ( c.level === N - 1 && ! c.fade ) ) continue;
			r = Math.min( r, distance( c.box, x, z ) );

		}

		const dt = Math.min( clock - ( this._fineAt ?? clock ), 0.1 );
		this._fineAt = clock;
		u.value = snap ? r : u.value + ( r - u.value ) * ( 1 - Math.exp( - dt / ( r < u.value ? 0.12 : 0.4 ) ) );

	}

	// cells within this many metres whose wanted level has not arrived (the textures give way to these)
	awaited( within = Infinity ) {

		let n = 0;
		for ( const c of this.chunks ) if ( c.want > 0 && ! c.mesh[ c.want ] && c.d < within ) n ++;
		return n;

	}

	get pending() { return this.awaited(); }

	// resolve when the surroundings of this camera are in (walking: only the cells right around it hold up the first
	// frame, and not for long) or, with `all`, everything its view asks for (a capture)
	async settle( camera, { all = false } = {} ) {

		const t0 = performance.now();
		for ( ;; ) {

			this.update( camera, { force: true } );
			const wait = all ? this.baseLeft > 0 || this.pending > 0
				: this.awaited( 6 ) > 0 && performance.now() - t0 < 2500;
			if ( ! wait ) break;
			// (the next arrival, or a look at the clock: a level that failed wakes nobody)
			await new Promise( ( r ) => { this.waiters.push( r ); setTimeout( r, 250 ); } );

		}

		this.update( camera, { force: true } );

	}

	// The mirror image sees the town at its lowest level (always resident, a tenth of the triangles), so it costs a
	// fraction of a frame, not a second one. on: before it is drawn; off: after. It is another view than the one
	// culling.js looked through, so the lowest level's meshes are all let through here - cull again before the frame.
	mirrorView( on ) {

		const swapped = this.swapped;
		if ( ! on ) { for ( const m of swapped ) m.visible = ! m.visible; swapped.length = 0; return; }
		for ( const c of this.chunks ) {

			const lo = c.mesh[ 0 ];
			if ( ! lo ) continue;
			for ( const o of lo.userData.all ) o.layers.mask = 1;
			let any = false;
			for ( let l = 1; l < N; l ++ ) { const m = c.mesh[ l ]; if ( m && m.visible ) { m.visible = false; swapped.push( m ); any = true; } }
			if ( any && ! lo.visible ) { lo.visible = true; swapped.push( lo ); }

		}

	}

	// lods: cells shown at [ fine, near, mid, lo ]; tris: drawn before culling; resident: on the GPU; loaded: in
	// memory (the loader keeps every level it has read); view, by: what the fullest view from here holds, and of what
	stats() {

		const lods = LEVELS.map( () => 0 ), B = this.budget;
		let items = 0, tris = 0, resident = 0, loaded = 0;
		for ( const c of this.chunks ) {

			const shown = c.mesh[ c.level ];
			if ( shown ) { items ++; lods[ N - 1 - c.level ] ++; tris += shown.userData.shown ?? shown.userData.tris; }
			for ( const mesh of c.mesh ) if ( mesh ) { loaded += mesh.userData.tris; if ( mesh.userData.held ) resident += mesh.userData.tris; }

		}

		return { items, lods, pending: this.pending + this.baseLeft, tris, resident, loaded, ...( B.most ? { view: B.most, by: B.mostBy } : {} ) };

	}

}
