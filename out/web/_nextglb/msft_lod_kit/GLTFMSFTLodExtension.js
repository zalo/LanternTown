// MSFT_lod for three.js GLTFLoader, as a plugin - a reference, written to be lifted into the loader.
//
//   import { GLTFMSFTLodExtension } from './GLTFMSFTLodExtension.js';
//   loader.register( ( parser ) => new GLTFMSFTLodExtension( parser ) );             // tier A: "transparent"
//   loader.register( ( parser ) => new GLTFMSFTLodExtension( parser, { lazy: true } ) ); // tier B: lazy levels
//   const gltf = await loadRangedGLB( loader, url );  // tier B over ONE .glb, fetched by HTTP range requests
//
// It depends on three and on nothing else.
//
// TIER A (default, no options): load() / parse() work as before and everything is loaded before they answer. The one
// difference: where a node carries MSFT_lod, its parent gets a THREE.LOD in the node's place:
//
//   parent
//     LOD                 position = middle of the bounds of level 0 (in the parent's space); no name
//       Group             level 0 holder, position = - that middle; holds the node itself (name, transform,
//         <node>            userData / extras, children, associations: all as GLTFLoader makes them)
//       Group             level 1 holder; holds the node ids[ 0 ] names
//       ...
//       Object3D          only when the last MSFT_screencoverage value is above 0: an empty level ("nothing drawn")
//
// three.js switches a THREE.LOD by itself while it renders (LOD.autoUpdate): nothing for the application to call.
// Material-level MSFT_lod: every mesh drawn with such a material becomes an LOD of copies of the mesh (one geometry,
// one material per level).
//
// Distances. THREE.LOD switches on the distance from the camera to the LOD's own origin; MSFT_screencoverage gives
// screen coverage. The mapping used here (a choice; the spec calls the values a hint):
//
//   coverage( d ) = area of the bounding sphere's disc / area of the screen
//                 = PI r^2 / ( 4 aspect tan^2( fov / 2 ) d^2 )
//   distance( c ) = r sqrt( PI / ( 4 aspect tan^2( fov / 2 ) c ) )
//
// r: half the diagonal of the bounding box of level 0 (from the accessors' min / max, with the transforms of the
// nodes above it in the file), fov / aspect: a reference camera, options.fov (50) and options.aspect (16 / 9).
// Level i ( i >= 1 ) starts at distance( coverage[ i - 1 ] ); below coverage[ last ] nothing is drawn. Without
// MSFT_screencoverage: coverage[ i ] = 0.25 ^ ( i + 1 ), the last level is never dropped.
//
// TIER B ({ lazy: true }): parse() answers when the coarsest level (that has geometry) of every LOD is there. A finer
// level is asked for when the camera first needs it: GLTFLazyLOD (a subclass of LOD) overrides update( camera ) -
// the call the renderer makes for every LOD it meets - and starts the load from there. At most options.maxInFlight
// levels load at a time (all files together), the nearest LOD first; the level that is there keeps being drawn until
// the wanted one has arrived. Per level the file needs buffers of its own for this to save anything (one .bin per
// level, or loadRangedGLB() below). options.releaseAfter (seconds, default 0 = never) drops a level that has not
// been drawn for that long. LOD chains that animations or skins point into are loaded whole, as in tier A.

import { Box3, Group, LOD, Matrix4, Object3D, Quaternion, Vector3 } from 'three';

const EXT = 'MSFT_lod';
const COVERAGE = 'MSFT_screencoverage';
const SHARED = '*';
const MESHOPT = [ 'EXT_meshopt_compression', 'KHR_meshopt_compression' ];
const COMPONENT_MAX = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };

const NONE = 0, LOADING = 1, READY = 2, FAILED = 3;

const _v1 = new Vector3(), _v2 = new Vector3(), _m = new Matrix4();

/** distance at which a sphere of `radius` covers `coverage` (0..1) of a screen with this fov (degrees) and aspect */
export function coverageToDistance( coverage, radius, fov = 50, aspect = 16 / 9 ) {

	const t = Math.tan( fov * Math.PI / 360 );
	return radius * Math.sqrt( Math.PI / ( 4 * aspect * t * t * coverage ) );

}

/** the screen coverage values taken when a file has none: [ 0.25, 0.0625, ..., 0 ] */
export function defaultCoverage( levels ) {

	const c = [];
	for ( let i = 0; i < levels; i ++ ) c.push( i < levels - 1 ? Math.pow( 0.25, i + 1 ) : 0 );
	return c;

}

// ---------------------------------------------------------------------------------------------- loading on demand

// one queue for all files: what is in flight is bounded for the page, not per model
const queue = {

	max: 2, inFlight: 0, peak: 0, waiting: new Set(),

	pump() {

		while ( this.inFlight < this.max && this.waiting.size ) {

			const now = performance.now();
			let best = null;
			for ( const lod of this.waiting ) {

				const want = lod._want;
				// (an LOD that left the scene is not updated any more: its wish lapses)
				if ( want < 0 || lod._state[ want ] !== NONE || now - lod._seen > 1000 ) { this.waiting.delete( lod ); continue; }
				if ( best === null || lod._distance < best._distance ) best = lod;

			}

			if ( best === null ) return;
			this.waiting.delete( best );
			best._ext._start( best, best._want );

		}

	}

};

/**
 * The LOD of tier B: a THREE.LOD whose levels arrive when they are first needed.
 * levels[ i ].object is a Group that is empty until its node has been loaded.
 */
export class GLTFLazyLOD extends LOD {

	constructor() {

		super();
		this.isGLTFLazyLOD = true;
		this._ext = null; // the plugin (null: a copy - what is loaded stays, nothing more is fetched)
		this._chain = null;
		this._state = [];
		this._last = []; // when a level was last drawn (ms)
		this._want = - 1;
		this._distance = 0;
		this._seen = 0;
		this._swept = 0;

	}

	/** NONE 0, LOADING 1, READY 2, FAILED 3 */
	levelState( level ) { return this._state[ level ]; }

	update( camera ) {

		super.update( camera );
		if ( this._ext !== null ) this._ext._afterUpdate( this, camera );

	}

}

// ---------------------------------------------------------------------------------------------- one file by ranges

const ranged = new Map(); // token in the JSON -> RangedSource (until the parser has picked it up)
let rangedCount = 0;

/**
 * A file read through HTTP range requests. Requests made in the same task are merged when they touch (or nearly).
 * A server that answers 200 with the whole body is taken at its word once: the body is kept and serves every
 * later request, nothing is fetched again.
 */
export class RangedSource {

	constructor( url, { gap = 16384, requestInit = {} } = {} ) {

		this.url = url; this.gap = gap; this.requestInit = requestInit;
		this.chunks = []; // { start, end, bytes } kept (the probe)
		this.full = null; // Promise<Uint8Array> once a server has sent everything
		this.mode = 'unknown'; // 'ranges' | 'full'
		this.pending = []; this.timer = null; this.gate = null;
		this.log = []; // { start, end, status, bytes } per request made
		this.offset = 0; // where buffer 0 starts in the file (.glb: behind the BIN chunk header)

	}

	async _fetch( start, end ) {

		const headers = { ...( this.requestInit.headers || {} ), Range: `bytes=${start}-${end - 1}` };
		const res = await fetch( this.url, { ...this.requestInit, headers } );
		if ( res.status === 206 ) {

			const bytes = new Uint8Array( await res.arrayBuffer() );
			this.mode = 'ranges';
			this.log.push( { start, end, status: 206, bytes: bytes.length } );
			if ( bytes.length < end - start && this.size === undefined ) this.size = start + bytes.length; // (asked past the end)
			return { start, bytes };

		}

		if ( res.status === 200 ) {

			// no range support: this answer is the whole file
			if ( this.full === null ) this.full = res.arrayBuffer().then( ( b ) => new Uint8Array( b ) );
			this.mode = 'full';
			const bytes = await this.full;
			this.log.push( { start, end, status: 200, bytes: bytes.length } );
			return { start: 0, bytes };

		}

		throw new Error( `RangedSource: ${this.url} answered ${res.status} to bytes=${start}-${end - 1}` );

	}

	/** bytes [ start, start + length ) of the file -> Promise<Uint8Array> (a view: copy it if you keep it) */
	request( start, length ) {

		const end = start + length;
		if ( this.full !== null ) return this.full.then( ( b ) => b.subarray( start, end ) );
		for ( const c of this.chunks ) if ( start >= c.start && end <= c.end ) return Promise.resolve( c.bytes.subarray( start - c.start, end - c.start ) );
		return new Promise( ( resolve, reject ) => {

			this.pending.push( { start, end, resolve, reject } );
			if ( this.timer === null ) this.timer = setTimeout( () => this._flush(), 0 );

		} );

	}

	async _flush() {

		this.timer = null;
		const list = this.pending.sort( ( a, b ) => a.start - b.start );
		this.pending = [];
		const groups = [];
		for ( const r of list ) {

			const g = groups[ groups.length - 1 ];
			if ( g && r.start - g.end <= this.gap ) { g.end = Math.max( g.end, r.end ); g.items.push( r ); } else groups.push( { start: r.start, end: r.end, items: [ r ] } );

		}

		const run = async ( g ) => {

			try {

				if ( this.full !== null ) { const b = await this.full; for ( const r of g.items ) r.resolve( b.subarray( r.start, r.end ) ); return; }
				const got = await this._fetch( g.start, g.end );
				for ( const r of g.items ) r.resolve( got.bytes.subarray( r.start - got.start, r.end - got.start ) );

			} catch ( e ) { for ( const r of g.items ) r.reject( e ); }

		};

		// until it is known that the server does ranges, one request at a time: a server that does not must not be
		// asked for the whole file once per group
		if ( this.mode === 'unknown' ) {

			if ( this.gate === null ) this.gate = run( groups.shift() );
			await this.gate;

		}

		await Promise.all( groups.map( run ) );

	}

	/** the first `bytes` of the file, kept for later requests */
	async probe( bytes ) {

		const got = await this._fetch( 0, bytes );
		if ( this.mode === 'ranges' ) this.chunks.push( { start: 0, end: got.bytes.length, bytes: got.bytes } );
		return got.bytes;

	}

}

/**
 * Loads a .glb through range requests: the header and the JSON first (options.probe bytes, default 64 KiB, asked
 * for in one request - with the coarse levels at the front of the BIN chunk that request is also the first picture),
 * then the bufferViews as the parser asks for them. Needs GLTFMSFTLodExtension registered on the loader (with
 * { lazy: true }, or everything is fetched at once, range by range). A server without range support answers the
 * probe with the whole file: it is then parsed like any .glb, from that one answer.
 * -> Promise<gltf>; gltf.userData.rangedSource is the RangedSource (its .log lists the requests made).
 */
export async function loadRangedGLB( loader, url, { probe = 65536, path, gap, requestInit } = {} ) {

	const src = new RangedSource( url, { gap, requestInit } );
	const base = path !== undefined ? path : url.slice( 0, url.lastIndexOf( '/' ) + 1 );
	let head = await src.probe( probe );
	let gltf;
	if ( src.mode === 'full' ) {

		const whole = await src.full;
		gltf = await loader.parseAsync( whole.buffer.slice( whole.byteOffset, whole.byteOffset + whole.byteLength ), base );

	} else {

		const dv = new DataView( head.buffer, head.byteOffset, head.byteLength );
		if ( head.length < 20 || dv.getUint32( 0, true ) !== 0x46546C67 ) throw new Error( 'loadRangedGLB: not a .glb: ' + url );
		const jsonLength = dv.getUint32( 12, true );
		if ( dv.getUint32( 16, true ) !== 0x4E4F534A ) throw new Error( 'loadRangedGLB: the first chunk is not JSON' );
		if ( 20 + jsonLength > head.length ) {

			// the JSON is longer than the probe: one more request for the rest
			const rest = await src.request( head.length, 20 + jsonLength - head.length );
			const all = new Uint8Array( 20 + jsonLength );
			all.set( head ); all.set( rest, head.length );
			head = all;

		}

		const json = JSON.parse( new TextDecoder().decode( head.subarray( 20, 20 + jsonLength ) ) );
		src.offset = 20 + jsonLength + 8;
		const token = 'ranged:' + ( ++ rangedCount );
		ranged.set( token, src );
		const b0 = json.buffers && json.buffers[ 0 ];
		if ( ! b0 || b0.uri !== undefined ) throw new Error( 'loadRangedGLB: buffer 0 is not the BIN chunk' );
		b0.extras = { ...( b0.extras || {} ), rangedSource: token };
		// A compressed view of the BIN chunk: the stock meshopt plugin would ask for the whole buffer. Its extension is
		// set aside here; loadBufferView() of the plugin below fetches the range and calls the same decoder.
		for ( const view of json.bufferViews || [] ) {

			for ( const name of MESHOPT ) {

				const def = view.extensions && view.extensions[ name ];
				if ( def && def.buffer === 0 ) {

					view.extras = { ...( view.extras || {} ), rangedMeshopt: def };
					const { [ name ]: _, ...others } = view.extensions;
					view.extensions = others;

				}

			}

		}

		gltf = await loader.parseAsync( JSON.stringify( json ), base );
		ranged.delete( token );

	}

	gltf.userData.rangedSource = src;
	return gltf;

}

// ---------------------------------------------------------------------------------------------- the plugin

export class GLTFMSFTLodExtension {

	constructor( parser, options = {} ) {

		this.name = EXT;
		this.parser = parser;
		this.options = {
			lazy: false,
			fov: 50, aspect: 16 / 9, // the reference camera of the coverage -> distance mapping
			hysteresis: 0, // THREE.LOD's hysteresis (a fraction of the distance) for every level
			distances: null, // ( info ) => { distance: [ 0, d1, .. ], hysteresis?: [ .. ], center?: [ x, y, z ], cull?: d } | null
			maxInFlight: 2, // lazy: levels loading at a time (all files together)
			releaseAfter: 0, // lazy: seconds a level may stay undrawn before it is dropped (0: never)
			keepBuffers: false, // lazy: keep the fetched (still compressed) buffer of a level in the parser's cache
			onLevel: null, // lazy: ( lod, level, object ) when a level has arrived (before it is first drawn)
			...options
		};
		this.chains = new Map(); // node index of level 0 -> chain
		this.materialChains = new Map();
		this.lods = []; // { lod, chain }
		this.sources = new Map(); // buffer index -> RangedSource
		if ( this.options.lazy ) queue.max = this.options.maxInFlight;
		this._prepare();

	}

	// ---- reading the JSON

	_prepare() {

		const json = this.parser.json;
		const nodes = json.nodes || [];
		const valid = ( def, list ) => {

			const ext = def.extensions && def.extensions[ EXT ];
			if ( ! ext || ! Array.isArray( ext.ids ) || ext.ids.length === 0 ) return null;
			if ( ext.ids.some( ( id ) => ! Number.isInteger( id ) || id < 0 || id >= list.length || list[ id ] === def ) ) { console.warn( 'GLTFMSFTLodExtension: MSFT_lod.ids out of range, ignored' ); return null; }
			return ext.ids;

		};

		const coverageOf = ( def, levels ) => {

			const c = def.extras && def.extras[ COVERAGE ];
			if ( ! Array.isArray( c ) || c.length !== levels ) return null;
			for ( let i = 0; i < levels; i ++ ) if ( typeof c[ i ] !== 'number' || c[ i ] < 0 || c[ i ] > 1 || ( i > 0 && c[ i ] > c[ i - 1 ] ) || ( i < levels - 1 && c[ i ] === 0 ) ) return null;
			return c;

		};

		( json.buffers || [] ).forEach( ( b, i ) => {

			const token = b.extras && b.extras.rangedSource;
			if ( token && ranged.has( token ) ) this.sources.set( i, ranged.get( token ) );

		} );

		( json.materials || [] ).forEach( ( def, i ) => {

			const ids = valid( def, json.materials );
			if ( ids ) this.materialChains.set( i, { ids: [ i, ...ids ], coverage: coverageOf( def, ids.length + 1 ) } );

		} );

		const lower = new Set();
		nodes.forEach( ( def, i ) => { const ids = valid( def, nodes ); if ( ids ) for ( const id of ids ) lower.add( id ); } );
		nodes.forEach( ( def, i ) => {

			const ids = valid( def, nodes );
			// (a level of another LOD that has levels of its own: taken as a plain node)
			if ( ids && ! lower.has( i ) ) this.chains.set( i, { base: i, ids: [ i, ...ids ], def, coverage: coverageOf( def, ids.length + 1 ) } );

		} );
		if ( this.chains.size === 0 ) return;

		const parent = new Map();
		nodes.forEach( ( def, i ) => { for ( const c of def.children || [] ) parent.set( c, i ); } );

		// who uses what: a level owns a mesh / accessor / bufferView / buffer when nothing else in the file touches it
		const owners = { node: new Map(), mesh: new Map(), accessor: new Map(), view: new Map(), buffer: new Map() };
		const add = ( map, key, owner ) => { if ( key === undefined || key === null ) return; if ( ! map.has( key ) ) map.set( key, new Set() ); map.get( key ).add( owner ); };
		const subtree = ( i, out = [] ) => { out.push( i ); for ( const c of nodes[ i ].children || [] ) subtree( c, out ); return out; };
		for ( const chain of this.chains.values() ) {

			chain.nodes = chain.ids.map( ( id ) => subtree( id ) );
			chain.hasMesh = chain.nodes.map( ( list ) => list.some( ( n ) => nodes[ n ].mesh !== undefined ) );
			chain.nodes.forEach( ( list, level ) => { for ( const n of list ) add( owners.node, n, chain.base + ':' + level ); } );

		}

		nodes.forEach( ( def, i ) => { for ( const o of owners.node.get( i ) || [ SHARED ] ) add( owners.mesh, def.mesh, o ); } );
		( json.meshes || [] ).forEach( ( def, i ) => {

			for ( const o of owners.mesh.get( i ) || [ SHARED ] ) for ( const p of def.primitives || [] ) {

				for ( const k in p.attributes ) add( owners.accessor, p.attributes[ k ], o );
				add( owners.accessor, p.indices, o );
				for ( const t of p.targets || [] ) for ( const k in t ) add( owners.accessor, t[ k ], o );

			}

		} );
		for ( const s of json.skins || [] ) add( owners.accessor, s.inverseBindMatrices, SHARED );
		for ( const a of json.animations || [] ) for ( const s of a.samplers || [] ) { add( owners.accessor, s.input, SHARED ); add( owners.accessor, s.output, SHARED ); }
		( json.accessors || [] ).forEach( ( def, i ) => {

			for ( const o of owners.accessor.get( i ) || [ SHARED ] ) {

				add( owners.view, def.bufferView, o );
				if ( def.sparse ) { add( owners.view, def.sparse.indices.bufferView, o ); add( owners.view, def.sparse.values.bufferView, o ); }

			}

		} );
		for ( const im of json.images || [] ) add( owners.view, im.bufferView, SHARED );
		( json.bufferViews || [] ).forEach( ( def, i ) => {

			for ( const o of owners.view.get( i ) || [ SHARED ] ) {

				add( owners.buffer, def.buffer, o );
				for ( const name of MESHOPT ) if ( def.extensions && def.extensions[ name ] ) add( owners.buffer, def.extensions[ name ].buffer, o );
				if ( def.extras && def.extras.rangedMeshopt ) add( owners.buffer, def.extras.rangedMeshopt.buffer, o );

			}

		} );
		this.owners = owners;

		// what animations and skins point at: such chains are loaded whole
		const moving = new Set();
		for ( const a of json.animations || [] ) for ( const ch of a.channels || [] ) if ( ch.target && ch.target.node !== undefined ) moving.add( ch.target.node );
		for ( const s of json.skins || [] ) { for ( const j of s.joints || [] ) moving.add( j ); if ( s.skeleton !== undefined ) moving.add( s.skeleton ); }
		nodes.forEach( ( def, i ) => { if ( def.skin !== undefined ) moving.add( i ); } );

		for ( const chain of this.chains.values() ) {

			chain.eager = chain.nodes.some( ( list ) => list.some( ( n ) => moving.has( n ) ) );

			// bounds of level 0 in the parent's space (the first level that has any, should level 0 be empty)
			const box = new Box3();
			for ( let level = 0; level < chain.ids.length && box.isEmpty(); level ++ ) this._expand( box, chain.ids[ level ], new Matrix4() );
			chain.center = box.isEmpty() ? new Vector3() : box.getCenter( new Vector3() );
			chain.radius = box.isEmpty() ? 1 : Math.max( box.getSize( _v1 ).length() / 2, 1e-9 );
			// the scale of what is above, as far as the file says
			const world = new Matrix4();
			for ( let p = parent.get( chain.base ); p !== undefined; p = parent.get( p ) ) world.premultiply( this._local( nodes[ p ] ) );
			chain.scale = world.getMaxScaleOnAxis();
			Object.assign( chain, this._distances( chain.def, chain.ids.length, chain.coverage, chain.radius * chain.scale, chain.center ) );

		}

	}

	_local( def ) {

		const m = new Matrix4();
		if ( def.matrix !== undefined ) return m.fromArray( def.matrix );
		return m.compose( new Vector3().fromArray( def.translation || [ 0, 0, 0 ] ), new Quaternion().fromArray( def.rotation || [ 0, 0, 0, 1 ] ), new Vector3().fromArray( def.scale || [ 1, 1, 1 ] ) );

	}

	_expand( box, index, above ) {

		const json = this.parser.json, def = json.nodes[ index ];
		const m = above.clone().multiply( this._local( def ) );
		if ( def.mesh !== undefined ) for ( const p of json.meshes[ def.mesh ].primitives || [] ) {

			const a = json.accessors[ p.attributes.POSITION ];
			if ( ! a || ! a.min || ! a.max ) continue;
			const k = a.normalized ? 1 / COMPONENT_MAX[ a.componentType ] : 1;
			for ( let c = 0; c < 8; c ++ ) box.expandByPoint( _v1.set( ( c & 1 ? a.max : a.min )[ 0 ] * k, ( c & 2 ? a.max : a.min )[ 1 ] * k, ( c & 4 ? a.max : a.min )[ 2 ] * k ).applyMatrix4( m ) );

		}

		for ( const c of def.children || [] ) this._expand( box, c, m );

	}

	// -> { distance: [ 0, d1, .. ], hysteresis: [ .. ], cull: distance beyond which nothing is drawn | Infinity, center }
	_distances( def, levels, coverage, radius, center ) {

		const o = this.options;
		const cov = coverage || defaultCoverage( levels );
		const distance = [ 0 ], hysteresis = [ 0 ];
		for ( let i = 1; i < levels; i ++ ) { distance.push( Math.max( distance[ i - 1 ], coverageToDistance( cov[ i - 1 ], radius, o.fov, o.aspect ) ) ); hysteresis.push( o.hysteresis ); }
		const out = { distance, hysteresis, cull: cov[ levels - 1 ] > 0 ? coverageToDistance( cov[ levels - 1 ], radius, o.fov, o.aspect ) : Infinity, center, usedCoverage: cov };
		const own = o.distances ? o.distances( { def, levels, coverage, radius, center, parser: this.parser } ) : null;
		if ( own && Array.isArray( own.distance ) && own.distance.length === levels ) {

			out.distance = own.distance.slice();
			if ( Array.isArray( own.hysteresis ) ) out.hysteresis = own.hysteresis.slice();
			if ( own.center ) out.center = new Vector3().fromArray( own.center );
			out.cull = own.cull !== undefined ? own.cull : Infinity;

		}

		return out;

	}

	// ---- parser hooks

	beforeRoot() {

		// A lower level that is also listed in a scene or as somebody's child would be drawn on top of the LOD: it
		// belongs to its LOD only.
		if ( this.chains.size === 0 ) return null;
		const json = this.parser.json, lower = new Set();
		for ( const chain of this.chains.values() ) for ( let i = 1; i < chain.ids.length; i ++ ) lower.add( chain.ids[ i ] );
		const strip = ( list ) => ( list && list.some( ( n ) => lower.has( n ) ) ? list.filter( ( n ) => ! lower.has( n ) ) : list );
		for ( const s of json.scenes || [] ) s.nodes = strip( s.nodes );
		for ( const n of json.nodes ) if ( n.children ) n.children = strip( n.children );
		return null;

	}

	loadNode( index ) {

		const chain = this.chains.get( index );
		if ( chain === undefined ) return null;
		return this.options.lazy && ! chain.eager ? this._loadLazy( chain ) : this._loadEager( chain );

	}

	// bufferViews of a file that is read by ranges (loadRangedGLB)
	loadBufferView( index ) {

		if ( this.sources.size === 0 ) return null;
		const parser = this.parser, view = parser.json.bufferViews[ index ];
		const packed = view.extras && view.extras.rangedMeshopt;
		const src = this.sources.get( packed ? packed.buffer : view.buffer );
		if ( src === undefined ) return null;
		if ( ! packed ) return src.request( src.offset + ( view.byteOffset || 0 ), view.byteLength ).then( ( bytes ) => bytes.slice().buffer );
		const decoder = parser.options.meshoptDecoder;
		if ( ! decoder || ! decoder.supported ) throw new Error( 'THREE.GLTFLoader: setMeshoptDecoder must be called before loading compressed files' );
		return src.request( src.offset + ( packed.byteOffset || 0 ), packed.byteLength ).then( ( source ) => {

			// (what the meshopt plugin of GLTFLoader does with its slice of the whole buffer)
			if ( decoder.decodeGltfBufferAsync ) return decoder.decodeGltfBufferAsync( packed.count, packed.byteStride, source, packed.mode, packed.filter ).then( ( res ) => res.buffer );
			return decoder.ready.then( () => {

				const result = new ArrayBuffer( packed.count * packed.byteStride );
				decoder.decodeGltfBuffer( new Uint8Array( result ), packed.count, packed.byteStride, source, packed.mode, packed.filter );
				return result;

			} );

		} );

	}

	afterRoot( result ) {

		const pending = [];
		if ( this.materialChains.size ) for ( const scene of result.scenes ) pending.push( this._materialLods( scene ) );
		// Animations found the LOD where they asked for the node (the parser's cache holds the LOD under the node's
		// index): point their tracks at the node itself.
		for ( const { lod, chain } of this.lods ) {

			if ( ! chain.eager && this.options.lazy ) continue;
			const node = lod.levels[ 0 ].object.children[ 0 ];
			if ( ! node ) continue;
			const from = lod.uuid + '.', to = ( node.name ? node.name : node.uuid ) + '.';
			for ( const clip of result.animations || [] ) for ( const track of clip.tracks ) if ( track.name.startsWith( from ) ) track.name = to + track.name.slice( from.length );

		}

		return pending.length ? Promise.all( pending ) : null;

	}

	// ---- building

	// the node of a level as GLTFLoader makes it; level 0 without coming back here
	_loadLevel( chain, level ) {

		const parser = this.parser;
		let pending;
		if ( level > 0 ) pending = parser.getDependency( 'node', chain.ids[ level ] );
		else {

			pending = null;
			for ( const ext of Object.values( parser.plugins ) ) {

				if ( ext === this || ! ext.loadNode ) continue;
				pending = ext.loadNode( chain.base );
				if ( pending ) break;

			}

			if ( ! pending ) pending = parser.loadNode( chain.base );

		}

		return this.materialChains.size ? pending.then( ( node ) => this._materialLods( node ) ) : pending;

	}

	_setup( lod, chain, nodes ) {

		lod.position.copy( chain.center );
		for ( let i = 0; i < chain.ids.length; i ++ ) {

			const holder = new Group();
			holder.position.copy( chain.center ).negate();
			if ( nodes[ i ] ) holder.add( nodes[ i ] );
			lod.addLevel( holder, chain.distance[ i ], chain.hysteresis[ i ] || 0 );

		}

		if ( chain.cull !== Infinity ) lod.addLevel( new Object3D(), Math.max( chain.cull, chain.distance[ chain.ids.length - 1 ] ), this.options.hysteresis );
		this.lods.push( { lod, chain } );
		return lod;

	}

	_loadEager( chain ) {

		return Promise.all( chain.ids.map( ( _, level ) => this._loadLevel( chain, level ) ) ).then( ( nodes ) => this._setup( new LOD(), chain, nodes ) );

	}

	_loadLazy( chain ) {

		const n = chain.ids.length;
		let first = n - 1;
		while ( first > 0 && ! chain.hasMesh[ first ] ) first --; // the coarsest level that draws something
		if ( ! chain.hasMesh[ first ] ) first = n - 1;
		// (levels without geometry cost nothing: they are made at once)
		const now = chain.ids.map( ( _, level ) => ( level === first || ! chain.hasMesh[ level ] ? this._loadLevel( chain, level ) : null ) );
		return Promise.all( now ).then( ( nodes ) => {

			const lod = new GLTFLazyLOD();
			lod._ext = this; lod._chain = chain; lod._first = first;
			lod._state = nodes.map( ( node ) => ( node ? READY : NONE ) );
			lod._last = nodes.map( () => performance.now() );
			this._setup( lod, chain, nodes );
			nodes.forEach( ( node, level ) => { if ( node ) this._settled( chain, level ); } );
			return lod;

		} );

	}

	// ---- tier B: called from GLTFLazyLOD.update(), i.e. by the renderer, once per LOD and frame

	_afterUpdate( lod, camera ) {

		const levels = lod.levels, state = lod._state, n = state.length, now = performance.now();
		const want = lod.getCurrentLevel();
		lod._seen = now;
		if ( want >= n || state[ want ] === READY ) { lod._want = - 1; if ( want < n ) lod._last[ want ] = now; } else {

			lod._want = want;
			lod._distance = _v1.setFromMatrixPosition( camera.matrixWorld ).distanceTo( _v2.setFromMatrixPosition( lod.matrixWorld ) ) / camera.zoom;
			if ( state[ want ] === NONE ) queue.waiting.add( lod );
			// until it is there: the next coarser level that is, else the next finer one
			let shown = - 1;
			for ( let i = want + 1; i < n && shown < 0; i ++ ) if ( state[ i ] === READY ) shown = i;
			for ( let i = want - 1; i >= 0 && shown < 0; i -- ) if ( state[ i ] === READY ) shown = i;
			if ( shown >= 0 ) { levels[ shown ].object.visible = true; lod._last[ shown ] = now; }

		}

		queue.pump();

		const after = this.options.releaseAfter * 1000;
		if ( after > 0 && now - lod._swept > 1000 ) {

			lod._swept = now;
			for ( let i = 0; i < n; i ++ ) if ( state[ i ] === READY && i !== lod._first && lod._chain.hasMesh[ i ] && now - lod._last[ i ] > after ) this._release( lod, i );

		}

	}

	_start( lod, level ) {

		const chain = lod._chain;
		lod._state[ level ] = LOADING;
		queue.inFlight ++; queue.peak = Math.max( queue.peak, queue.inFlight );
		this._loadLevel( chain, level ).then( ( node ) => {

			if ( this.options.onLevel ) this.options.onLevel( lod, level, node );
			lod.levels[ level ].object.add( node );
			lod._state[ level ] = READY;
			lod._last[ level ] = performance.now();
			this._settled( chain, level );

		} ).catch( ( e ) => {

			lod._state[ level ] = FAILED;
			console.error( 'GLTFMSFTLodExtension: level ' + level + ' of node ' + chain.base + ' failed to load', e );

		} ).then( () => { queue.inFlight --; queue.pump(); } );

	}

	_owned( kind, level, chain ) {

		const me = chain.base + ':' + level, out = [];
		for ( const [ key, set ] of this.owners[ kind ] ) if ( set.size === 1 && set.has( me ) ) out.push( key );
		return out;

	}

	// a level has been built: the file's bytes it came from are not needed again
	_settled( chain, level ) {

		if ( this.options.keepBuffers ) return;
		for ( const b of this._owned( 'buffer', level, chain ) ) this.parser.cache.remove( 'buffer:' + b );

	}

	_release( lod, level ) {

		const parser = this.parser, chain = lod._chain, holder = lod.levels[ level ].object;
		const meshes = new Set( this._owned( 'mesh', level, chain ) );
		const prims = new Set();
		for ( const m of meshes ) for ( const p of parser.json.meshes[ m ].primitives ) prims.add( p );
		for ( const node of holder.children.slice() ) {

			node.traverse( ( o ) => { const a = o.geometry && parser.associations.get( o ); if ( a && meshes.has( a.meshes ) ) o.geometry.dispose(); } );
			holder.remove( node );

		}

		// forget what the parser remembers of the level, or the next request would hand out the same objects
		for ( const n of chain.nodes[ level ] ) { if ( n !== chain.base ) parser.cache.remove( 'node:' + n ); delete parser.nodeCache[ n ]; }
		for ( const m of meshes ) parser.cache.remove( 'mesh:' + m );
		for ( const key in parser.primitiveCache ) if ( prims.has( parser.primitiveCache[ key ].primitive ) ) delete parser.primitiveCache[ key ];
		for ( const a of this._owned( 'accessor', level, chain ) ) parser.cache.remove( 'accessor:' + a );
		for ( const v of this._owned( 'view', level, chain ) ) parser.cache.remove( 'bufferView:' + v );
		for ( const b of this._owned( 'buffer', level, chain ) ) parser.cache.remove( 'buffer:' + b );
		lod._state[ level ] = NONE;

	}

	// ---- material-level MSFT_lod: one geometry, a material per level

	async _materialLods( root ) {

		const parser = this.parser, found = [];
		root.traverse( ( o ) => {

			if ( ! o.isMesh || Array.isArray( o.material ) || ( o.parent && o.parent.userData.materialLodHolder ) ) return;
			const a = parser.associations.get( o.material );
			if ( a && this.materialChains.has( a.materials ) ) found.push( [ o, this.materialChains.get( a.materials ) ] );

		} );
		for ( const [ mesh, chain ] of found ) {

			const materials = await Promise.all( chain.ids.slice( 1 ).map( ( id ) => parser.getDependency( 'material', id ) ) );
			// (the same sphere as for a node: around the bounding box, in the parent's space)
			if ( mesh.geometry.boundingBox === null ) mesh.geometry.computeBoundingBox();
			mesh.updateMatrix();
			const box = new Box3().copy( mesh.geometry.boundingBox ).applyMatrix4( mesh.matrix );
			const center = box.getCenter( new Vector3() );
			let scale = 1;
			for ( let p = mesh.parent; p; p = p.parent ) { p.updateMatrix(); scale *= p.matrix.getMaxScaleOnAxis(); }
			const plan = this._distances( parser.json.materials[ chain.ids[ 0 ] ], chain.ids.length, chain.coverage, Math.max( box.getSize( _v1 ).length() / 2, 1e-9 ) * scale, center );
			const lod = new LOD();
			lod.position.copy( plan.center );
			const parent = mesh.parent, at = parent ? parent.children.indexOf( mesh ) : - 1;
			for ( let i = 0; i < chain.ids.length; i ++ ) {

				let object = mesh;
				if ( i > 0 ) {

					// (children of the mesh - child nodes of a node that is the mesh - stay with level 0)
					object = mesh.clone( false );
					object.material = materials[ i - 1 ];
					parser.assignFinalMaterial( object );
					if ( parser.associations.has( mesh ) ) parser.associations.set( object, parser.associations.get( mesh ) );

				}

				const holder = new Group();
				holder.userData.materialLodHolder = true;
				holder.position.copy( plan.center ).negate();
				holder.add( object );
				lod.addLevel( holder, plan.distance[ i ], plan.hysteresis[ i ] || 0 );

			}

			if ( plan.cull !== Infinity ) lod.addLevel( new Object3D(), Math.max( plan.cull, plan.distance[ chain.ids.length - 1 ] ), this.options.hysteresis );
			if ( parent ) { parent.add( lod ); parent.children.splice( parent.children.indexOf( lod ), 1 ); parent.children.splice( at, 0, lod ); } else root = lod;

		}

		return root;

	}

}

/** numbers of the shared load queue: { max, inFlight, peak, waiting } */
export function lodQueueStats() { return { max: queue.max, inFlight: queue.inFlight, peak: queue.peak, waiting: queue.waiting.size }; }

export default GLTFMSFTLodExtension;
