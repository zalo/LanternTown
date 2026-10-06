// Walkable town runtime: streams the composed scene described by out/town/town.json (written by tools/town.py)
// into an engine: terrain, per-item clones of the built assets with LOD streaming + switching, lantern lights
// (clustered pool + a few shadow casters that follow the camera), flames / halos, collision + navigation.
//
// Coordinates in town.json are Blender's (X east, Y north, Z up, metres); three is (x, z, -y).

import { THREE, createFlame, createHalo, createContactShadows, capLight } from './engine.js';
import { fetchSafe } from './net.js';

export const TIERS = {
	// lod: scales the LOD switch distances. pool: shadow-casting lantern lights. lights: clustered light budget.
	high: { pixelRatio: 1.5, ao: true, shadow: 4096, pool: 4, poolSize: 1024, lights: 448, lod: 1, bldLod0: true, parts: 45, cull: 90, load: 8, shadowEvery: 1.0 },
	medium: { pixelRatio: 1, ao: true, shadow: 2048, pool: 2, poolSize: 512, lights: 224, lod: 0.6, bldLod0: true, parts: 30, cull: 60, load: 6, shadowEvery: 2.5 },
	low: { pixelRatio: 0.75, ao: false, ssr: false, shadow: 2048, pool: 0, poolSize: 512, lights: 96, lod: 0.34, bldLod0: false, parts: 18, cull: 38, load: 4, shadowEvery: 1e9, tex: 1024 },
	// phones / integrated GPUs: tex caps the material textures, minLod drops the full-detail meshes entirely,
	// probes: false skips the reflection cube captures
	phone: { pixelRatio: 0.6, ao: false, ssr: false, shadow: 1024, pool: 0, poolSize: 512, lights: 48, lod: 0.22, bldLod0: false, parts: 8, cull: 26, load: 2, shadowEvery: 1e9, tex: 512, minLod: 1, probes: false, aniso: 2 },
};

const LOD0_DIST = 16, LOD1_DIST = 44; // metres to the item's bounds at lod scale 1
const NONE = 0, LOADING = 1, READY = 2, FAILED = 3;
const BIG = new Set( [ 'building', 'landmark', 'backdrop' ] );

const b2t = ( p, out = new THREE.Vector3() ) => out.set( p[ 0 ], p[ 2 ] ?? 0, - p[ 1 ] );
const CAP_UP = 0.25; // light a capped lantern sends upwards, relative to what it sends down
const parseJSON = ( s, d ) => {

	if ( s === undefined || s === null ) return d;
	if ( typeof s !== 'string' ) return s;
	try {

		return JSON.parse( s );

	} catch ( e ) {

		return d;

	}

};

// ------------------------------------------------------------------------------------------ navigation

export class Nav {

	constructor( town ) {

		this.regions = town.regions || [];
		this.colliders = ( town.colliders || [] ).map( ( c ) => {

			if ( c.t !== 'b' ) return c;
			const a = THREE.MathUtils.degToRad( c.r || 0 );
			return { ...c, cos: Math.cos( a ), sin: Math.sin( a ) };

		} );
		this.radius = 0.3;
		this.step = 0.5; // max step up
		this.drop = 0.6; // max step down
		this.head = 1.9;

	}

	heightOf( r, x, y ) {

		if ( r.ramp ) {

			const q = r.ramp;
			const t = THREE.MathUtils.clamp( ( ( x - q.o[ 0 ] ) * q.d[ 0 ] + ( y - q.o[ 1 ] ) * q.d[ 1 ] ) / q.len, 0, 1 );
			return q.z0 + ( q.z1 - q.z0 ) * t;

		}

		if ( r.arch ) {

			const R = r.rect;
			const t = r.arch.axis === 'x' ? ( x - R[ 0 ] ) / ( R[ 2 ] - R[ 0 ] ) : ( y - R[ 1 ] ) / ( R[ 3 ] - R[ 1 ] );
			return r.z + r.arch.rise * ( 1 - ( 2 * t - 1 ) ** 2 );

		}

		return r.z;

	}

	// highest walkable surface at (x, y) reachable from height z; null if there is none or something blocks the body
	groundAt( x, y, z ) {

		let best = null;
		for ( const r of this.regions ) {

			const R = r.rect;
			if ( x < R[ 0 ] || x > R[ 2 ] || y < R[ 1 ] || y > R[ 3 ] ) continue;
			const h = this.heightOf( r, x, y );
			if ( h <= z + this.step ) {

				if ( best === null || h > best ) best = h;

			} else if ( h < z + this.head ) return null; // stair flank / ledge at body height

		}

		if ( best !== null && best < z - this.drop ) return null;
		return best;

	}

	// push the point out of every collider that overlaps the body at height z; returns true if it moved
	resolve( p, z ) {

		const rad = this.radius;
		let hit = false;
		for ( const c of this.colliders ) {

			const zr = c.z;
			if ( zr && ( zr[ 1 ] < z + 0.3 || zr[ 0 ] > z + 1.7 ) ) continue;
			if ( c.t === 'c' ) {

				const dx = p.x - c.c[ 0 ], dy = p.y - c.c[ 1 ], R = c.rad + rad;
				const d2 = dx * dx + dy * dy;
				if ( d2 >= R * R ) continue;
				const d = Math.sqrt( d2 ) || 1e-6;
				p.x = c.c[ 0 ] + dx / d * R; p.y = c.c[ 1 ] + dy / d * R;
				hit = true;

			} else {

				// to box space
				const dx = p.x - c.c[ 0 ], dy = p.y - c.c[ 1 ];
				const lx = dx * c.cos + dy * c.sin, ly = - dx * c.sin + dy * c.cos;
				const hx = c.h[ 0 ], hy = c.h[ 1 ];
				const qx = Math.max( - hx, Math.min( hx, lx ) ), qy = Math.max( - hy, Math.min( hy, ly ) );
				let ex = lx - qx, ey = ly - qy;
				const d2 = ex * ex + ey * ey;
				if ( d2 >= rad * rad ) continue;
				let nx, ny;
				if ( d2 > 1e-10 ) {

					const d = Math.sqrt( d2 );
					nx = qx + ex / d * rad; ny = qy + ey / d * rad;

				} else {

					// centre is inside: leave through the nearest face
					const px = hx - Math.abs( lx ), py = hy - Math.abs( ly );
					if ( px < py ) {

						nx = Math.sign( lx || 1 ) * ( hx + rad ); ny = ly;

					} else {

						nx = lx; ny = Math.sign( ly || 1 ) * ( hy + rad );

					}

				}

				p.x = c.c[ 0 ] + nx * c.cos - ny * c.sin;
				p.y = c.c[ 1 ] + nx * c.sin + ny * c.cos;
				hit = true;

			}

		}

		return hit;

	}

	// move a body standing at (pos.x, pos.y, pos.z = feet height) by (dx, dy); slides along obstacles
	move( pos, dx, dy ) {

		const n = Math.max( 1, Math.ceil( Math.hypot( dx, dy ) / 0.2 ) );
		const p = { x: 0, y: 0 };
		for ( let i = 0; i < n; i ++ ) {

			for ( const [ sx, sy ] of [ [ dx / n, dy / n ], [ dx / n, 0 ], [ 0, dy / n ] ] ) {

				if ( sx === 0 && sy === 0 ) continue;
				p.x = pos.x + sx; p.y = pos.y + sy;
				this.resolve( p, pos.z );
				this.resolve( p, pos.z );
				const g = this.groundAt( p.x, p.y, pos.z );
				if ( g === null ) continue;
				pos.x = p.x; pos.y = p.y; pos.z = g;
				break;

			}

		}

		return pos;

	}

}

// ------------------------------------------------------------------------------------------ town

// Static batching. Every lit render object costs the CPU ~15 us per frame in the WebGPU renderer, so a town of a
// few hundred items x ~6 materials is CPU-bound. All static meshes that share a material (and vertex layout) go
// into ONE BatchedMesh instead: one render object per material, per-instance visibility (LOD switching), geometry
// shared between the instances of an asset. A batch that runs out of room is rebuilt at twice the size (instance
// and geometry ids are handed out in insertion order, so they survive the rebuild). Per-instance frustum culling is
// off on purpose: it re-uploads a texture per batch per frame, which costs more than the GPU saves.
class Batcher {

	constructor( group ) {

		this.group = group;
		this.keys = new Map();
		this.instances = 0;
		this.rebuilds = 0;
		this._sig = new WeakMap();
		this._ids = new WeakMap();
		this._n = 0;

	}

	get chunks() {

		return this.keys.size;

	}

	static ok( o ) {

		if ( ! o.isMesh || o.isSkinnedMesh || o.isInstancedMesh || o.isBatchedMesh || Array.isArray( o.material ) || ! o.material ) return false;
		const g = o.geometry;
		if ( ! g || ! g.attributes.position || g.morphAttributes?.position ) return false;
		for ( const k in g.attributes ) if ( g.attributes[ k ].isInterleavedBufferAttribute ) return false;
		return true;

	}

	_signature( g ) {

		let s = this._sig.get( g );
		if ( s === undefined ) {

			s = Object.keys( g.attributes ).sort().map( ( k ) => {

				const a = g.attributes[ k ];
				return `${k}:${a.itemSize}:${a.array.constructor.name}:${a.normalized ? 1 : 0}`;

			} ).join( ',' ) + ( g.index ? '|i' : '|n' );
			this._sig.set( g, s );

		}

		return s;

	}

	_id( o ) {

		let i = this._ids.get( o );
		if ( i === undefined ) this._ids.set( o, i = ++ this._n );
		return i;

	}

	// (re)create the BatchedMesh of a key at its current capacities and replay geometries + instances in order
	_build( K ) {

		const old = K.mesh, src = K.src;
		const bm = new THREE.BatchedMesh( K.cap, K.vCap, K.iCap, src.material );
		bm.name = 'batch:' + ( src.material.name || src.material.uuid.slice( 0, 8 ) );
		bm.frustumCulled = false;
		bm.perObjectFrustumCulled = false;
		bm.sortObjects = false;
		bm.castShadow = src.castShadow;
		bm.receiveShadow = src.receiveShadow;
		bm.renderOrder = src.renderOrder;
		bm.matrixAutoUpdate = false;
		for ( const g of K.geos ) bm.addGeometry( g );
		for ( const r of K.insts ) {

			const id = bm.addInstance( r.g );
			bm.setMatrixAt( id, r.m );
			bm.setVisibleAt( id, r.on );

		}

		if ( old ) {

			this.group.remove( old );
			old.dispose();
			this.rebuilds ++;

		}

		this.group.add( bm );
		K.mesh = bm;

	}

	// one hidden instance of mesh.geometry with the world matrix; returns a handle for show()
	add( mesh, matrix ) {

		const g = mesh.geometry;
		const key = `${this._id( mesh.material )}|${this._signature( g )}|${mesh.castShadow ? 1 : 0}${mesh.receiveShadow ? 1 : 0}|${mesh.renderOrder}`;
		let K = this.keys.get( key );
		if ( ! K ) this.keys.set( key, K = { mesh: null, src: { material: mesh.material, castShadow: mesh.castShadow, receiveShadow: mesh.receiveShadow, renderOrder: mesh.renderOrder },
			geos: [], geoId: new Map(), insts: [], v: 0, i: 0, vCap: 0, iCap: 0, cap: 0 } );
		let gid = K.geoId.get( g ), grow = false;
		if ( gid === undefined ) {

			const nv = g.attributes.position.count, ni = g.index ? g.index.count : 0;
			gid = K.geos.length;
			K.geos.push( g );
			K.geoId.set( g, gid );
			K.v += nv;
			K.i += ni;
			if ( K.v > K.vCap || K.i > K.iCap ) {

				K.vCap = Math.max( 2048, Math.ceil( K.v * 2 ) );
				K.iCap = Math.max( 6144, Math.ceil( K.i * 2 ) );
				grow = true;

			} else K.mesh.addGeometry( g );

		}

		const id = K.insts.length;
		K.insts.push( { g: gid, m: matrix.clone(), on: false } );
		if ( id >= K.cap ) {

			K.cap = Math.max( 16, K.cap * 2 );
			grow = true;

		}

		if ( grow ) this._build( K );
		else {

			const got = K.mesh.addInstance( gid );
			if ( got !== id ) console.warn( '[town] batch instance id drift', got, id );
			K.mesh.setMatrixAt( id, matrix );
			K.mesh.setVisibleAt( id, false );

		}

		this.instances ++;
		return { K, id };

	}

	show( h, on ) {

		const r = h.K.insts[ h.id ];
		if ( r.on === on ) return;
		r.on = on;
		h.K.mesh.setVisibleAt( h.id, on );

	}

}

const _m1 = new THREE.Matrix4(), _m2 = new THREE.Matrix4(), _q1 = new THREE.Quaternion(), _v1 = new THREE.Vector3(), _up = new THREE.Vector3( 0, 1, 0 );
const isPart = ( o ) => !! ( o.userData && o.userData.tsq_part && o.userData.tsq_part !== 'body' );

export class Town {

	constructor( eng, { base = '.', tier = 'high', lamp = 1, batch = true, cap = CAP_UP } = {} ) {

		this.lampGain = lamp;
		this.capUp = cap;
		this.batching = batch;

		this.eng = eng;
		this.base = base;
		this.tierName = tier;
		this.tier = TIERS[ tier ] || TIERS.high;
		this.group = new THREE.Group();
		this.group.name = 'town';
		eng.scene.add( this.group );
		this.batcher = new Batcher( this.group );
		this.items = [];
		this.protos = new Map();
		this.queue = [];
		this.inflight = new Set();
		this.lights = [];
		this.slots = [];
		this.shadowSlots = [];
		this.warnings = [];
		this.stats = { loaded: 0, failed: 0, bytes: 0, clones: 0 };
		this._flames = new Map();
		this._halos = new Map();
		this._shadowDirty = true;
		this._shadowAt = - 1e9;
		this._slowAt = - 1e9;
		this._frame = 0;

	}

	warn( m ) {

		if ( this.warnings.length < 60 ) this.warnings.push( String( m ) );
		console.warn( '[town]', m );

	}

	// fetch town.json, build everything that does not need streaming, then stream the coarsest LOD of every asset
	async load( url, onProgress = () => {} ) {

		const eng = this.eng, T = this.tier;
		// the GLBs as released (engine/glbrelease.js, tools/glbopt.mjs), or eng.assets itself; asked for alongside town.json
		const release = import( './glbrelease.js' ).then( ( m ) => m.openRelease( eng.assets, url ) ).catch( () => eng.assets );
		const data = this.data = await fetchSafe( url, { cache: 'no-cache' } ).then( ( r ) => r.json() );
		this.assets = await release;
		this.nav = new Nav( data );

		// world bounds from the items -> static moon shadow over the whole town
		// (the moon shadow covers the walkable town; the far skyline ring would stretch its texels threefold)
		const bounds = this.bounds = new THREE.Box3(), core = new THREE.Box3();
		for ( const [ key, p ] of Object.entries( data.protos ) ) {

			const maxLod = Math.max( 0, ( p.tris?.length || 1 ) - 1 );
			let minLod = Math.min( T.minLod || 0, maxLod );
			if ( BIG.has( p.cls ) && ! T.bldLod0 ) minLod = Math.max( minLod, Math.min( 1, maxLod ) );
			this.protos.set( key, { key, ...p, maxLod, minLod, lod: [ 0, 1, 2 ].map( () => ( { state: NONE, root: null, wanted: 1e9 } ) ) } );

		}

		const v = new THREE.Vector3();
		for ( const it of data.items ) {

			const p = this.protos.get( it.p );
			if ( ! p ) continue;
			const a = THREE.MathUtils.degToRad( it.rot || 0 ), c = Math.cos( a ), s = Math.sin( a ), sc = it.s ?? 1;
			const box = new THREE.Box3();
			for ( let i = 0; i < 8; i ++ ) {

				const lx = ( i & 1 ? p.max[ 0 ] : p.min[ 0 ] ) * sc, ly = ( i & 2 ? p.max[ 1 ] : p.min[ 1 ] ) * sc, lz = ( i & 4 ? p.max[ 2 ] : p.min[ 2 ] ) * sc;
				box.expandByPoint( v.set( it.pos[ 0 ] + lx * c - ly * s, it.pos[ 2 ] + lz, - ( it.pos[ 1 ] + lx * s + ly * c ) ) );

			}

			const size = box.getSize( new THREE.Vector3() );
			bounds.union( box );
			if ( p.cls !== 'backdrop' && p.cls !== 'building' ) core.union( box );
			this.items.push( { p, pos: b2t( it.pos ), rot: a, s: sc, tag: it.tag, box, small: size.length() < 1.6 && ! p.lights,
				cur: - 1, want: 2, d: 1e9, objs: [ null, null, null ], parts: [ null, null, null ] } );

		}

		// contact pads under the small things that stand on the ground (chairs, tables, benches, barrels, posts)
		const PADS = new Set( [ 'furniture', 'prop', 'prop_small', 'street_furniture' ] ), pads = [];
		for ( const it of this.items ) {

			const p = it.p;
			if ( ! PADS.has( p.cls ) || p.mount !== 'ground' || Math.abs( p.min[ 2 ] ) > 0.06 ) continue;
			const h = p.max[ 2 ] * it.s;
			let hx = ( p.max[ 0 ] - p.min[ 0 ] ) * 0.5 * it.s, hz = ( p.max[ 1 ] - p.min[ 1 ] ) * 0.5 * it.s;
			if ( h < 0.3 || Math.min( hx, hz ) < 0.12 || Math.max( hx, hz ) > 1.3 ) continue; // covers, pigeons, table lanterns, barriers
			if ( h > 2.5 ) hx = hz = Math.min( hx, hz, 0.3 ); // posts: the box is the lantern arms, the foot is a plinth
			const cx = ( p.max[ 0 ] + p.min[ 0 ] ) * 0.5 * it.s, cy = ( p.max[ 1 ] + p.min[ 1 ] ) * 0.5 * it.s, c = Math.cos( it.rot ), s = Math.sin( it.rot );
			pads.push( { x: it.pos.x + cx * c - cy * s, y: it.pos.y + 0.015, z: it.pos.z - ( cx * s + cy * c ), rot: it.rot,
				hx: hx * 1.3 + 0.1, hz: hz * 1.3 + 0.1, k: h > 2.5 ? 0.5 : 0.7 } );

		}

		if ( pads.length ) this.group.add( createContactShadows( pads ) );

		if ( bounds.isEmpty() ) bounds.set( new THREE.Vector3( - 40, - 2, - 40 ), new THREE.Vector3( 40, 20, 40 ) );
		eng.moon.shadow.mapSize.set( T.shadow, T.shadow );
		eng.moon.shadow.autoUpdate = false;
		if ( core.isEmpty() ) eng.fitShadow( bounds, 4 );
		else eng.fitShadow( core, 8, { tight: true } );

		onProgress( 0, 'terrain' );
		await Promise.all( ( data.terrain || [] ).map( async ( t ) => {

			const [ id, name ] = t.file.split( '/' );
			try {

				const inst = await this.assets.load( id, name, { version: t.v, sidecar: false } );
				inst.root.traverse( ( o ) => {

					o.matrixAutoUpdate = false;
					if ( o.isMesh ) o.castShadow = false; // flat ground: receives only

				} );
				inst.root.updateMatrixWorld( true );
				this.group.add( inst.root );

			} catch ( e ) {

				this.warn( `terrain ${t.file}: ${e.message || e}` );

			}

		} ) );

		this._blockouts( data.blockouts || [] );
		this._lights( data.lights || [], data.items );

		// coarsest LOD of everything first: the whole town is there (and casts moon shadows) before detail streams in
		const protos = [ ...this.protos.values() ];
		let done = 0;
		const first = protos.map( ( p ) => this._load( p, p.maxLod ).then( () => onProgress( ++ done / protos.length, p.asset ) ) );
		await this._throttle( first );
		this._shadowDirty = true;
		this._probeSetup( data );
		return this;

	}

	async _throttle( promises ) {

		await Promise.all( promises );

	}

	_blockouts( list ) {

		if ( ! list.length ) return;
		const mat = new THREE.MeshStandardNodeMaterial( { color: 0x4b4640, roughness: 0.92, metalness: 0 } );
		const geo = new THREE.BoxGeometry( 1, 1, 1 );
		geo.translate( 0, 0.5, 0 );
		for ( const b of list ) {

			const m = new THREE.Mesh( geo, mat );
			m.position.set( b.c[ 0 ], b.c[ 2 ] || 0, - b.c[ 1 ] );
			m.rotation.y = THREE.MathUtils.degToRad( b.r || 0 );
			m.scale.set( b.h[ 0 ] * 2, b.height || 8, b.h[ 1 ] * 2 );
			m.castShadow = m.receiveShadow = true;
			m.matrixAutoUpdate = false;
			m.updateMatrix();
			m.name = 'blockout:' + b.asset;
			this.group.add( m );

		}

	}

	_lights( list, items = [] ) {

		const eng = this.eng, T = this.tier;
		// lanterns with a cap throw most of their light down and sideways: walls above them fall off towards the
		// moonlit blue instead of being lit to the eaves. A light record may say so itself (up: 0..1, the share of the
		// light that leaves above the cap). Otherwise it is read off the flame: the assets give lantern flames a flicker
		// of 0.07..0.35, steadier lights are windows, signs, globes and dials, livelier ones bare candles - and open
		// fires, paper shades and the bounce / spill helpers of doors and tables are bare whatever they flicker.
		const BARE = /^(brazier|cafe_table|paper_lantern|parasol|door_plank|bicycle|neon_glyph|neon_sign_vertical|festoon_lights|bunting)\//;
		const capped = ( l ) => ( l.flicker ?? 0 ) > 0.065 && l.flicker < 0.36 && ! BARE.test( items[ l.item ]?.p || '' );
		this.lights = list.map( ( l, i ) => ( { ...l, up: l.up ?? ( capped( l ) ? this.capUp : 1 ), intensity: l.intensity * this.lampGain, pos: b2t( l.pos ), col: new THREE.Color( l.color || '#ffbf6a' ), phase: ( i * 2.399 ) % 6.283,
			slot: null, shadowed: false, d: 0 } ) );
		const n = Math.min( this.lights.length, T.lights );
		for ( let i = 0; i < n; i ++ ) {

			const L = new THREE.PointLight( 0xffbf6a, 0, 9, 2 );
			this.group.add( L );
			const rec = { light: L, base: 0, flicker: 0, phase: 0 };
			eng.lanterns.push( rec );
			this.slots.push( { light: L, rec, l: null } );

		}

		for ( let i = 0; i < T.pool; i ++ ) {

			const L = new THREE.PointLight( 0xffbf6a, 0, 9, 2 );
			L.castShadow = true;
			capLight( L );
			L.shadow.mapSize.set( T.poolSize, T.poolSize );
			L.shadow.bias = - 0.002;
			L.shadow.normalBias = 0.02;
			L.shadow.autoUpdate = false;
			L.position.set( 0, - 50, 0 );
			this.group.add( L );
			const rec = { light: L, base: 0, flicker: 0, phase: 0 };
			eng.lanterns.push( rec );
			this.shadowSlots.push( { light: L, rec, l: null } );

		}

		// glow sprites for every lantern, whether or not it currently owns a light. One per fitting, where its burner is
		// seen - none at a helper light (town.json src 0: shop-front spill, window fill, one light for a row of lamps),
		// which has nothing to look at. `halo` = the output the glow stands for, `halo_at` = the burner when the light
		// sits off it (tk.asm light())
		for ( const l of this.lights ) {

			const I = l.halo !== undefined ? l.halo * this.lampGain : ( l.src !== 0 ? l.intensity : 0 );
			if ( I < 12 ) continue;
			const at = l.halo_at ? l.halo_at.map( ( p ) => b2t( p ) ) : [ l.pos ]; // several: the globes of one head, each with its share
			const size = Math.max( 0.5, Math.round( Math.sqrt( I / at.length / 40 ) * 1.5 * 4 ) / 4 );
			const key = l.col.getHexString() + '|' + size;
			let tpl = this._halos.get( key );
			if ( ! tpl ) this._halos.set( key, tpl = createHalo( '#' + l.col.getHexString(), size, 0.1, true ) );
			for ( const p of at ) {

				const h = tpl.clone();
				h.position.copy( p );
				h.matrixAutoUpdate = false;
				h.updateMatrix();
				this.group.add( h );

			}

		}

		this._assignLights( new THREE.Vector3(), true );

	}

	_applySlot( s ) {

		const l = s.l;
		if ( ! l ) {

			s.rec.base = 0; s.light.intensity = 0;
			return;

		}

		s.light.position.copy( l.pos );
		s.light.color.copy( l.col );
		s.light.distance = l.range ?? 9;
		s.light.userData.up = l.up;
		s.rec.base = l.shadowed ? 0 : l.intensity;
		s.rec.flicker = l.flicker ?? 0.1;
		s.rec.phase = l.phase;

	}

	// give the light pool to the lanterns that matter most from here (all of them when the budget allows)
	_assignLights( cam, force = false ) {

		const L = this.lights, S = this.slots;
		if ( ! S.length ) return;
		if ( S.length >= L.length ) {

			if ( force ) L.forEach( ( l, i ) => {

				S[ i ].l = l; l.slot = S[ i ];
				this._applySlot( S[ i ] );

			} );
			return;

		}

		for ( const l of L ) l.d = l.pos.distanceTo( cam ) - ( l.range ?? 9 ) * 0.5;
		const keep = new Set( [ ...L ].sort( ( a, b ) => a.d - b.d ).slice( 0, S.length ) );
		const free = [];
		for ( const s of S ) {

			if ( s.l && keep.has( s.l ) ) keep.delete( s.l );
			else {

				if ( s.l ) s.l.slot = null;
				s.l = null;
				free.push( s );

			}

		}

		for ( const l of keep ) {

			const s = free.pop();
			if ( ! s ) break;
			s.l = l; l.slot = s;
			this._applySlot( s );

		}

		for ( const s of free ) this._applySlot( s );

	}

	// the few shadow-casting lights follow the camera: nearest strong lanterns, at most one swap per call
	_assignShadows( cam ) {

		const SS = this.shadowSlots;
		if ( ! SS.length ) return;
		const cand = this.lights.filter( ( l ) => l.intensity >= 24 );
		for ( const l of cand ) l.ds = l.pos.distanceTo( cam );
		cand.sort( ( a, b ) => a.ds - b.ds );
		const want = cand.slice( 0, SS.length );
		const free = SS.filter( ( s ) => ! s.l );
		for ( const l of want ) {

			if ( l.shadowed ) continue;
			let s = free.pop();
			if ( ! s ) {

				// replace the slot whose lantern is now farthest, if it is clearly farther than the newcomer
				const far = SS.reduce( ( a, b ) => ( a.l.ds > b.l.ds ? a : b ) );
				if ( far.l.ds < l.ds + 2.5 ) continue;
				far.l.shadowed = false;
				if ( far.l.slot ) this._applySlot( far.l.slot );
				s = far;

			}

			s.l = l; l.shadowed = true;
			if ( l.slot ) this._applySlot( l.slot );
			s.light.position.copy( l.pos );
			s.light.color.copy( l.col );
			s.light.distance = l.range ?? 9;
			s.light.userData.up = l.up;
			s.light.shadow.camera.near = Math.max( 0.03, l.near ?? 0.1 );
			s.light.shadow.camera.far = l.range ?? 9;
			s.light.shadow.camera.updateProjectionMatrix();
			s.light.shadow.needsUpdate = true;
			s.rec.base = l.intensity; s.rec.flicker = l.flicker ?? 0.1; s.rec.phase = l.phase;
			return true; // one cube shadow render per call keeps frame times even

		}

		return false;

	}

	// ---------------------------------------------------------------------------------- streaming

	_load( p, k ) {

		const L = p.lod[ k ];
		if ( L.state !== NONE ) return L.promise || Promise.resolve();
		L.state = LOADING;
		const name = p.key.split( '/' )[ 1 ] + ( k ? '.lod' + k : '' );
		L.promise = this.assets.load( p.asset, name, { version: p.v, bytes: p.bytes?.[ k ], sidecar: false } ).then( ( inst ) => {

			L.root = inst.root;
			L.state = READY;
			this.stats.loaded ++;
			this.stats.bytes += p.bytes?.[ k ] || 0;

		} ).catch( ( e ) => {

			L.state = FAILED;
			this.stats.failed ++;
			this.warn( `load ${p.key} lod${k}: ${e.message || e}` );

		} );
		return L.promise;

	}

	_request( p, k, d ) {

		const L = p.lod[ k ];
		if ( L.state !== NONE ) return;
		if ( ! L.queued ) {

			L.queued = true;
			this.queue.push( { p, k, L } );

		}

		L.wanted = Math.min( L.wanted, d );

	}

	_pump() {

		const Q = this.queue;
		if ( ! Q.length || this.inflight.size >= this.tier.load ) return;
		Q.sort( ( a, b ) => b.L.wanted - a.L.wanted ); // nearest last -> pop()
		while ( Q.length && this.inflight.size < this.tier.load ) {

			const q = Q.pop();
			const pr = this._load( q.p, q.k ).then( () => {

				this.inflight.delete( pr );
				q.L.queued = false;

			} );
			this.inflight.add( pr );

		}

		for ( const q of Q ) q.L.wanted = 1e9;

	}

	get pending() {

		return this.queue.length + this.inflight.size;

	}

	_flame( pr, name ) {

		const size = pr.size ?? 0.09, color = pr.color || '#ffb860', glow = pr.glow ?? 9, speed = pr.speed ?? 1;
		const ph = ( this._flameN = ( ( this._flameN || 0 ) + 1 ) % 5 );
		const key = [ size, color, glow, speed, ph ].join( '|' );
		let tpl = this._flames.get( key );
		if ( ! tpl ) this._flames.set( key, tpl = createFlame( this.eng.lib.noise, { size, color, glow, phase: ph * 0.2 + 0.07, speed } ) );
		return tpl.clone();

	}

	// split a loaded LOD once: static meshes (-> batches) are taken out of the tree, what is left (animated parts,
	// flame sockets, anything unbatchable) is cloned per item
	_plan( p, k ) {

		const L = p.lod[ k ];
		if ( L.plan ) return L.plan;
		const root = L.root, stat = [];
		root.updateMatrixWorld( true );
		const inv = new THREE.Matrix4().copy( root.matrixWorld ).invert();
		root.traverse( ( o ) => {

			if ( ! this.batching || ! Batcher.ok( o ) || o.matrixWorld.determinant() < 0 ) return;
			for ( let a = o; a && a !== root; a = a.parent ) if ( isPart( a ) ) return;
			stat.push( { mesh: o, rel: new THREE.Matrix4().multiplyMatrices( inv, o.matrixWorld ) } );

		} );
		for ( const { mesh: o } of stat ) {

			const par = o.parent;
			if ( ! par ) continue; // the root itself is a mesh: nothing to detach from
			if ( o.children.length ) {

				const g = new THREE.Object3D();
				g.name = o.name;
				g.userData = o.userData;
				g.position.copy( o.position );
				g.quaternion.copy( o.quaternion );
				g.scale.copy( o.scale );
				for ( const c of [ ...o.children ] ) g.add( c );
				par.add( g );

			}

			par.remove( o );

		}

		let dynamic = false;
		root.traverse( ( o ) => {

			if ( stat.length && o === root && o.isMesh ) return;
			if ( o.isMesh || o.isSprite || o.isLine || o.isPoints || ( o.userData && o.userData.tsq_socket === 'fx' ) ) dynamic = true;

		} );
		return ( L.plan = { stat, dynamic, rootIsStatic: stat.some( ( e ) => e.mesh === root ) } );

	}

	// one item at one LOD: batched instances for the static meshes + a clone of the dynamic remainder
	_instantiate( it, k ) {

		const plan = this._plan( it.p, k );
		_m1.compose( it.pos, _q1.setFromAxisAngle( _up, it.rot ), _v1.setScalar( it.s ) );
		const handles = plan.stat.map( ( e ) => this.batcher.add( e.mesh, _m2.multiplyMatrices( _m1, e.rel ) ) );
		let obj = null;
		const parts = [];
		if ( plan.dynamic && ! plan.rootIsStatic ) {

			obj = it.p.lod[ k ].root.clone( true );
			obj.position.copy( it.pos );
			obj.rotation.y = it.rot;
			obj.scale.setScalar( it.s );
			const sockets = [];
			obj.traverse( ( o ) => {

				const ud = o.userData || {};
				if ( isPart( o ) ) parts.push( { name: ud.tsq_part, node: o, props: parseJSON( ud.tsq_part_props, {} ), rest: o.quaternion.clone() } );
				if ( ud.tsq_socket === 'fx' ) sockets.push( o );

			} );
			for ( const o of sockets ) {

				const pr = parseJSON( o.userData.tsq_props, {} );
				const nm = o.userData.tsq_name || '';
				if ( pr.fx === 'flame' || pr.kind === 'flame' || nm.includes( 'flame' ) ) o.add( this._flame( pr, nm ) );

			}

			this.group.add( obj );
			obj.updateMatrixWorld( true );
			// static: only animated parts keep recomputing their matrices
			obj.traverse( ( o ) => {

				o.matrixAutoUpdate = false;

			} );
			for ( const p of parts ) p.node.matrixAutoUpdate = true;
			obj.visible = false;
			this.stats.clones ++;

		}

		it.parts[ k ] = parts;
		return ( it.objs[ k ] = { obj, handles } );

	}

	_vis( rec, on ) {

		if ( rec.obj ) rec.obj.visible = on;
		for ( const h of rec.handles ) this.batcher.show( h, on );

	}

	_show( it, k ) {

		if ( it.cur >= 0 && it.objs[ it.cur ] ) this._vis( it.objs[ it.cur ], false );
		if ( k >= 0 ) this._vis( it.objs[ k ] || this._instantiate( it, k ), true );
		it.cur = k;
		this._shadowDirty = true;

	}

	// per frame: LOD selection, streaming requests, light pools, static shadow refresh
	update( camera, now = performance.now() / 1000, { force = false } = {} ) {

		const T = this.tier, cp = camera.position;
		const d0 = LOD0_DIST * T.lod, d1 = LOD1_DIST * T.lod;
		const slow = force || now - this._slowAt > 0.4;
		this._frame ++;
		for ( const it of this.items ) {

			const p = it.p;
			const d = it.d = it.box.distanceToPoint( cp );
			// hysteresis around the switch distances
			let want = it.want;
			if ( want === 0 ) want = d > d0 * 1.08 ? ( d > d1 * 1.08 ? 2 : 1 ) : 0;
			else if ( want === 1 ) want = d < d0 * 0.94 ? 0 : d > d1 * 1.08 ? 2 : 1;
			else want = d < d0 * 0.94 ? 0 : d < d1 * 0.94 ? 1 : 2;
			it.want = want;
			const k = Math.min( Math.max( want, p.minLod ), p.maxLod );
			let use = - 1;
			if ( ! ( it.small && d > T.cull ) ) {

				if ( p.lod[ k ].state === READY ) use = k;
				else {

					this._request( p, k, d );
					for ( let j = k + 1; j <= p.maxLod && use < 0; j ++ ) if ( p.lod[ j ].state === READY ) use = j;
					for ( let j = k - 1; j >= 0 && use < 0; j -- ) if ( p.lod[ j ].state === READY ) use = j;

				}

			}

			if ( use !== it.cur ) this._show( it, use );

		}

		this._pump();
		if ( slow ) {

			this._slowAt = now;
			this._assignLights( cp );
			this._assignShadows( cp );
			if ( ! force ) this._probeRefresh( now );
			const ps = this.eng.partSets;
			ps.length = 0;
			for ( const it of this.items ) {

				const parts = it.cur >= 0 ? it.parts[ it.cur ] : null;
				if ( parts && parts.length && it.d < T.parts ) ps.push( parts );

			}

		}

		if ( this._shadowDirty && ( force || now - this._shadowAt > T.shadowEvery || this._shadowAt < 0 ) ) {

			this._shadowDirty = false;
			this._shadowAt = now;
			this.eng.moon.shadow.needsUpdate = true;
			for ( const s of this.shadowSlots ) if ( s.l ) s.light.shadow.needsUpdate = true;

		}

	}

	// reflection probes: two along the canal (either side of the bridge, box-projected onto the canal corridor)
	// and one over the square for basins. Positions come from the layout frame, nothing is placed by hand.
	_probeSetup( data ) {

		if ( this.tier.probes === false ) return;
		const PR = this.eng.lib.probe, F = data.frame || {}, reg = ( n ) => ( data.regions || [] ).find( ( r ) => r.name === n )?.rect;
		const quay = reg( 'quay' ), far = reg( 'farbank' ), bridge = reg( 'bridge' ), wz = F.water_z ?? - 1.2;
		this._probes = [];
		if ( F.canal_x && quay && far ) {

			const x0 = Math.min( far[ 0 ], quay[ 0 ] ), x1 = Math.max( far[ 2 ], quay[ 2 ] ), y0 = quay[ 1 ] - 2, y1 = quay[ 3 ] + 2;
			const by = bridge ? ( bridge[ 1 ] + bridge[ 3 ] ) / 2 : ( y0 + y1 ) / 2, cx = ( F.canal_x[ 0 ] + F.canal_x[ 1 ] ) / 2;
			PR.min.value.set( x0, wz, - y1 );
			PR.max.value.set( x1, wz + 18, - y0 );
			PR.splitZ.value = - by;
			PR.p[ 0 ].pos.value.set( cx, wz + 1.4, - ( y0 + by ) / 2 );
			PR.p[ 1 ].pos.value.set( cx, wz + 1.4, - ( y1 + by ) / 2 );
			this._probes.push( 0, 1 );

		}

		const f = F.fountain || [ 0, 0 ];
		PR.p[ 2 ].pos.value.set( f[ 0 ] + 4, 2.6, - ( f[ 1 ] - 3 ) );
		this._probes.push( 2 );
		this._probeScore = [ 1e9, 1e9, 1e9 ];
		this._probeAt = - 1e9; // first capture: when the first settle / idle update has put the town on screen

	}

	// a probe is re-captured when the detail around it is better than at its last capture (LODs streamed in)
	_probeRefresh( now, all = false ) {

		if ( ! this._probes || this.pending || ( ! all && now - this._probeAt < 6 ) ) return;
		const PR = this.eng.lib.probe;
		for ( const i of this._probes ) {

			const c = PR.p[ i ].pos.value;
			let score = 0;
			for ( const it of this.items ) if ( it.box.distanceToPoint( c ) < 35 ) score += it.cur < 0 ? 3 : it.cur;
			if ( score >= this._probeScore[ i ] ) continue;
			this._probeScore[ i ] = score;
			this._probeAt = now;
			this.eng.captureProbe( i );
			if ( ! all ) return; // one probe per call keeps the hitch short

		}

	}

	// capture / teleport: bring everything this camera wants fully in before the next frame
	async settle( camera, now = performance.now() / 1000 ) {

		for ( let i = 0; i < 40; i ++ ) {

			this.update( camera, now, { force: true } );
			if ( ! this.pending ) break;
			await Promise.all( [ ...this.inflight ] );

		}

		for ( let i = 0; i < this.shadowSlots.length + 1; i ++ ) this._assignShadows( camera.position );
		this.update( camera, now, { force: true } );
		this._shadowDirty = false;
		this.eng.moon.shadow.needsUpdate = true;
		for ( const s of this.shadowSlots ) if ( s.l ) s.light.shadow.needsUpdate = true;
		this._probeRefresh( now, true );

	}

	visibleStats() {

		let tris = 0;
		const lods = [ 0, 0, 0 ];
		for ( const it of this.items ) if ( it.cur >= 0 ) {

			tris += it.p.tris?.[ it.cur ] || 0;
			lods[ it.cur ] ++;

		}

		return { items: this.items.length, lods, tris, pending: this.pending, loaded: this.stats.loaded, clones: this.stats.clones,
			batches: this.batcher.chunks, instances: this.batcher.instances, rebuilds: this.batcher.rebuilds,
			mb: Math.round( this.stats.bytes / 1e5 ) / 10 };

	}

}
