// bench.js - the baked town loaded from its packed format and from the glTF files tools/bake2glb.mjs makes of it,
// through the same engine (web/engine/lite.js, not modified), one variant per page load. See index.html for the
// parameters; the numbers are in window.__bench (and on the page) when __bench.done is set.
//
//   v=packed    the page's own loader: manifest.json + base.bin + per cell level files (BakedTown) - of a bake that keeps
//               no packed files (the live one: bake2glb --drop-packed), the .glb files that hold the same streams
//   v=exact     the same bytes as GLB: exact_lo.glb, cells/exact_<level>_<cell>.glb, read by GLTFLoader + the meshopt
//               decoder; the decoded views go into BakedTown's own mesh builder, so materials, budgets and partial
//               draws are the page's and the picture is the same
//   v=portable  the portable GLB (float positions, vertex light, one unlit material per texture layer) as it comes
//               out of GLTFLoader, no level logic; file=NAME.glb: another file of the folder (the dcc profile's)
//   v=lod       ONE glTF of the whole town with the levels of a cell linked by MSFT_lod, as THREE.LOD objects from the
//               kit's plugin, switched by three.js; form=glb (one file, eager), gltf (a .bin per level), ranged (one
//               .glb read by HTTP range requests); lazy=0 (tier A: everything up front) or 1 (tier B: on demand)
//   scene=walk  (default) the town as the page streams it at the spawn, then a walk; lo: only the lowest level of the
//               whole town; near / fine: after that, that level of the cells within 20 m of the spawn
import * as THREE from 'three';
import * as LITE from '../engine/lite.js';
import { Nav } from '../engine/town.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

const Q = new URLSearchParams( location.search );
const BASE = '..';
const GLB = Q.get( 'glb' ) || `${BASE}/out/web/_nextglb`;
const PLUGIN = Q.get( 'plugin' ) || `${BASE}/out/web/_nextglb/msft_lod_kit/GLTFMSFTLodExtension.js`;
const RANGED = Q.get( 'ranged' ) || `http://${location.hostname}:8791/bench/town_lod.glb`; // (a server that answers Range: the kit's server.mjs)
const V = Q.get( 'v' ) || 'packed';
const SCENE = Q.get( 'scene' ) || 'walk';
const FORM = Q.get( 'form' ) || 'glb';
const LAZY = Q.get( 'lazy' ) === '1';
const TIER = Q.get( 'tier' ) || 'high';
const WALK = Q.has( 'walk' ) ? parseFloat( Q.get( 'walk' ) ) : 12; // seconds
const SPEED = 4; // m / s
const SHOT = Q.get( 'shot' ) === '1';
const FILE = /^[\w./-]+\.glb$/.test( Q.get( 'file' ) || '' ) && ! Q.get( 'file' ).includes( '..' ) ? Q.get( 'file' ) : null; // (v=portable: another file of the folder)
const $ = ( id ) => document.getElementById( id );
const say = ( s ) => { $( 'msg' ).textContent = s; publish(); };

// ------------------------------------------------------------------------------------------------ instruments

const R = window.__bench = { hidden: document.visibilityState !== 'visible' || undefined, v: V, scene: SCENE, form: V === 'lod' ? FORM : undefined, lazy: V === 'lod' ? LAZY : undefined, tier: TIER, marks: {}, at: {}, done: false, errors: [] };
const mark = ( k ) => ( R.marks[ k ] = Math.round( performance.now() ) );
// what a visitor's report carries (engine/telemetry.js reads window.__stats): this load's row, without the pictures
const STATS = window.__stats = { page: 'bakeglb-bench' };
const publish = () => {

	for ( const k of [ 'v', 'scene', 'form', 'lazy', 'tier', 'backend', 'adapter', 'marks', 'at', 'settled', 'level', 'spawn', 'walk', 'end', 'decode', 'fetched', 'ranges', 'warnings', 'done', 'failed' ] ) if ( R[ k ] !== undefined ) STATS[ k ] = R[ k ];
	STATS.errors = R.errors.slice( 0, 3 ).map( ( e ) => e.slice( 0, 200 ) );
	// (the report takes 6000 characters: the walk's detail goes first)
	if ( JSON.stringify( STATS ).length > 5500 ) { delete STATS.walk; delete STATS.end; }

};
window.addEventListener( 'error', ( e ) => R.errors.push( String( e.message ) ) );
window.addEventListener( 'unhandledrejection', ( e ) => R.errors.push( String( e.reason && e.reason.message || e.reason ) ) );

// requests, from Resource Timing: wire = bytes of the body as sent (gzip where the server does it), raw = decoded
performance.setResourceTimingBufferSize( 20000 );
const isGeometry = ( u ) => /\/out\/web\/_next\/[^?]*\.(bin|glb)|\/_nextglb\/[^?]*\.(glb|gltf|bin)|:8791\//.test( u ) && ! /\.js(\?|$)/.test( u );
function fetched() {

	const o = { geo_n: 0, geo_wire: 0, geo_raw: 0, other_n: 0, other_wire: 0, other_raw: 0 };
	for ( const e of performance.getEntriesByType( 'resource' ) ) {

		const k = isGeometry( e.name ) ? 'geo' : 'other';
		o[ k + '_n' ] ++; o[ k + '_wire' ] += e.encodedBodySize; o[ k + '_raw' ] += e.decodedBodySize;

	}

	return o;

}

// main-thread tasks of 50 ms and more
const LONG = [];
try { new PerformanceObserver( ( l ) => { for ( const e of l.getEntries() ) LONG.push( [ e.startTime, e.duration ] ); } ).observe( { type: 'longtask', buffered: true } ); } catch ( e ) { /* not there */ }
function longTasks( t0, t1 ) {

	let n = 0, total = 0, longest = 0;
	for ( const [ s, d ] of LONG ) if ( s + d > t0 && s < t1 ) { n ++; total += d; longest = Math.max( longest, d ); }
	return { n, total: Math.round( total ), longest: Math.round( longest ) };

}

// the meshopt decoder: time on the main thread (sync) and, for what goes to the workers, call to answer (async: the
// sum over calls that overlap, and busy: the wall time during which at least one was out)
const DEC = { sync_ms: 0, sync_n: 0, async_ms: 0, async_n: 0, busy_ms: 0, out: 0, since: 0, bytes: 0 };
for ( const k of [ 'decodeVertexBuffer', 'decodeIndexBuffer', 'decodeIndexSequence', 'decodeGltfBuffer' ] ) {

	const f = MeshoptDecoder[ k ];
	MeshoptDecoder[ k ] = function ( ...a ) { const t = performance.now(); const r = f.apply( this, a ); DEC.sync_ms += performance.now() - t; DEC.sync_n ++; return r; };

}

{

	const f = MeshoptDecoder.decodeGltfBufferAsync;
	MeshoptDecoder.decodeGltfBufferAsync = function ( count, size, source, ...a ) {

		const t = performance.now();
		if ( DEC.out ++ === 0 ) DEC.since = t;
		DEC.async_n ++; DEC.bytes += count * size;
		const end = () => { const n = performance.now(); DEC.async_ms += n - t; if ( -- DEC.out === 0 ) DEC.busy_ms += n - DEC.since; };
		const p = f.call( this, count, size, source, ...a );
		p.then( end, end );
		return p;

	};

}

const decode = () => ( { main_ms: Math.round( DEC.sync_ms ), main_calls: DEC.sync_n, worker_ms_sum: Math.round( DEC.async_ms ), worker_busy_ms: Math.round( DEC.busy_ms ), worker_calls: DEC.async_n } );
const heap = () => ( performance.memory ? Math.round( performance.memory.usedJSHeapSize / 1048576 ) : null );
const snap = ( k ) => { R.at[ k ] = { ms: mark( k ), ...fetched(), long: longTasks( 0, performance.now() ), decode: decode(), heap_mb: heap() }; };
// (a hidden tab gets no animation frames: the timer keeps the run going there, 4 frames a second)
const raf = () => new Promise( ( r ) => { let d = false; const f = ( t ) => { if ( ! d ) { d = true; r( t ); } }; requestAnimationFrame( f ); setTimeout( f, 250 ); } );
const stat = ( a ) => { const s = a.slice().sort( ( x, y ) => x - y ), q = ( p ) => + ( s[ Math.min( s.length - 1, Math.floor( p * s.length ) ) ] || 0 ).toFixed( 2 ); return { median: q( 0.5 ), p95: q( 0.95 ), max: q( 1 ) }; };

function makeLoader() {

	const loader = new GLTFLoader();
	loader.setMeshoptDecoder( MeshoptDecoder );
	loader.setDRACOLoader( new DRACOLoader().setDecoderPath( `${BASE}/node_modules/three/examples/jsm/libs/draco/gltf/` ) ); // (the dcc profile's files; fetched only when a file asks for it)
	return loader;

}

// the two decoded views of a level (its vertex records and its indices) out of a parsed exact-mode file
async function levelViews( parser, node ) {

	const json = parser.json, p = json.meshes[ node.mesh ].primitives[ 0 ];
	const [ v, i ] = await Promise.all( [ parser.getDependency( 'bufferView', json.accessors[ p.attributes.POSITION ].bufferView ), parser.getDependency( 'bufferView', json.accessors[ p.indices ].bufferView ) ] );
	return [ new Uint8Array( v ), new Uint8Array( i ) ];

}

// ------------------------------------------------------------------------------------------------ v=exact

class GlbTown extends LITE.BakedTown {

	// the lowest level of all cells: one GLB instead of base.bin (nothing shows before the whole file is in)
	async _streamBase( onChunk ) {

		const gltf = await makeLoader().loadAsync( `${GLB}/exact_lo.glb?v=${this.bake.baked}` );
		const of = new Map();
		for ( const n of gltf.parser.json.nodes ) if ( n.extras && n.extras.baked && n.extras.baked.level === 'lo' ) of.set( n.extras.baked.cell, n );
		for ( let i = 0; i < this.chunks.length; i ++ ) {

			const c = this.chunks[ i ], m = c.m.lo;
			c.mesh[ 0 ] = this._build( null, m, await levelViews( gltf.parser, of.get( c.m.k ) ) );
			c.mesh[ 0 ].visible = c.level === 0;
			this.group.add( c.mesh[ 0 ] );
			this._dirty = true;
			this.baseLeft --;
			onChunk( i + 1 );

		}

		this._wake();

	}

	// a level of a cell: one GLB instead of the level file
	async _fetch( c, l ) {

		const lv = this.L[ l ], m = c.m[ lv ], key = m.file;
		try {

			const gltf = await makeLoader().loadAsync( `${GLB}/bench/cells/exact_${lv}_${c.m.k}.glb?v=${this.bake.baked}` );
			const node = gltf.parser.json.nodes.find( ( n ) => n.extras && n.extras.baked && n.extras.baked.level === lv );
			const decoded = await levelViews( gltf.parser, node );
			if ( c.want === l && ! c.mesh[ l ] ) {

				c.mesh[ l ] = this._build( null, m, decoded, l );
				c.mesh[ l ].visible = false;
				this.group.add( c.mesh[ l ] );

			}

		} catch ( e ) {

			c.failed = ( c.failed || 0 ) | ( 1 << l );
			R.errors.push( 'exact: ' + ( e.message || e ) );

		} finally {

			this.inflight.delete( key );
			this._dirty = true;
			this._wake();

		}

	}

}

// ------------------------------------------------------------------------------------------------ v=lod

// the town as GLTFLoader + the MSFT_lod plugin make it: a THREE.LOD per cell, switched by the renderer. BakedTown is
// only kept for its textures and materials (and the engine's hooks); its own level logic has nothing to do.
class LodTown extends LITE.BakedTown {

	constructor( eng, bake, o ) {

		super( eng, bake, o );
		this.R = this.R.map( ( r, i ) => ( i ? 0 : r ) );
		this.arrived = 0;

	}

	async _streamBase( onChunk ) {

		const kit = this.kit = await import( new URL( PLUGIN, location.href ).href );
		const loader = makeLoader();
		loader.register( ( parser ) => new kit.GLTFMSFTLodExtension( parser, {
			lazy: LAZY,
			// the distances the exporter wrote from the page's own tier table (extras.lod), not the coverage formula
			distances: ( { def } ) => ( def.extras && def.extras.lod ? { distance: def.extras.lod.distance, hysteresis: def.extras.lod.hysteresis } : null ),
			onLevel: ( lod, level, object ) => { this.arrived ++; object.traverse( ( o ) => o.isMesh && this._bind( o ) ); } } ) );
		const gltf = this.gltf = FORM === 'ranged' ? await kit.loadRangedGLB( loader, RANGED ) : await loader.loadAsync( FORM === 'gltf' ? `${GLB}/bench/town_lod_levels/town_lod.gltf` : `${GLB}/bench/town_lod.glb` );
		gltf.scene.traverse( ( o ) => o.isMesh && this._bind( o ) );
		this.lods = [];
		gltf.scene.traverse( ( o ) => o.isLOD && this.lods.push( o ) );
		this.group.add( gltf.scene );
		this.baseLeft = 0;
		onChunk( this.chunks.length );
		this._wake();

	}

	// a mesh of GLTFLoader's (one per draw range of the level) gets the page's material: the same interleaved buffer
	// under the attribute names and sizes the material reads
	_bind( o ) {

		const g = o.geometry, pos = g.attributes.position;
		if ( o.userData.bound || ! pos || ! pos.isInterleavedBufferAttribute ) return;
		o.userData.bound = true;
		const buf = pos.data, W = buf.stride, dir = W >= 12 && !! this.matsDir, M = dir ? this.matsDir : this.mats;
		const A = buf._baked || ( buf._baked = {
			position: new THREE.InterleavedBufferAttribute( buf, 4, 0, true ), _pl: new THREE.InterleavedBufferAttribute( buf, 4, 0, true ),
			uv: new THREE.InterleavedBufferAttribute( buf, 2, 4, true ), _col: new THREE.InterleavedBufferAttribute( buf, 4, 6, true ),
			_g: dir ? new THREE.InterleavedBufferAttribute( buf, 2, 10, true ) : null } );
		for ( const k of Object.keys( g.attributes ) ) g.deleteAttribute( k );
		for ( const k in A ) if ( A[ k ] ) g.setAttribute( k, A[ k ] );
		if ( g.index && g.index.array instanceof Uint16Array ) g.index.normalized = true; // (as BakedTown does: stays 16 bit on the GPU)
		const b = o.userData.baked || o.parent.userData.baked, kind = o.material.userData.baked_kind || 0;
		// (the box of this draw range, where the exporter wrote one: culled on its own, as the page's clusters are)
		const box = g.userData.box;
		if ( box && b ) {

			const s = b.q * 65535;
			g.boundingBox = new THREE.Box3( new THREE.Vector3( ( box[ 0 ] - b.min[ 0 ] ) / s, ( box[ 1 ] - b.min[ 1 ] ) / s, ( box[ 2 ] - b.min[ 2 ] ) / s ), new THREE.Vector3( ( box[ 3 ] - b.min[ 0 ] ) / s, ( box[ 4 ] - b.min[ 1 ] ) / s, ( box[ 5 ] - b.min[ 2 ] ) / s ) );
			g.boundingSphere = g.boundingBox.getBoundingSphere( new THREE.Sphere() );

		}

		o.material = M[ kind ];
		o.userData.pull = this.PULL[ b ? this.L.indexOf( b.level ) : 0 ] || null;

	}

	_fetch() { return Promise.resolve(); }

	get pending() {

		if ( ! this.kit || ! this.lods ) return 1;
		const q = this.kit.lodQueueStats();
		return q.inFlight + q.waiting;

	}

	lodStats() {

		const shown = [ 0, 0, 0, 0 ]; // LODs showing [ fine, near, mid, lo ] (a cell with two levels has mid and lo)
		let tris = 0, bytes = 0, meshes = 0;
		const seen = new Set();
		for ( const lod of this.lods ) { const i = lod.levels.findIndex( ( l ) => l.object.visible ); if ( i >= 0 ) shown[ Math.max( 0, 4 - lod.levels.length + i ) ] ++; }
		this.gltf.scene.traverse( ( o ) => {

			if ( ! o.isMesh ) return;
			meshes ++; tris += o.geometry.index.count / 3;
			for ( const a of [ o.geometry.attributes.position.data, o.geometry.index ] ) if ( ! seen.has( a ) ) { seen.add( a ); bytes += a.array.byteLength || a._bytes || 0; }

		} );
		return { lods: this.lods.length, shown, resident: tris, meshes, geometry_mb: + ( bytes / 1048576 ).toFixed( 1 ), arrived: this.arrived };

	}

	level( lod ) { return lod.levels.findIndex( ( l ) => l.object.visible ); }

}

// ------------------------------------------------------------------------------------------------ run

async function main() {

	const canvas = $( 'gl' );
	const W = + Q.get( 'w' ) || 1280, H = + Q.get( 'h' ) || 720;
	const bake = await LITE.loadBake( BASE, TIER );
	if ( ! bake ) throw new Error( 'no bake: ' + LITE.loadBake.note );
	mark( 'manifest' );
	const eng = window.eng = await LITE.createLiteEngine( { canvas, base: BASE, width: W, height: H, pixelRatio: 1, fov: 62, forceWebGL: Q.get( 'webgl' ) === '1', bake } );
	eng.setSize( W, H, 1 );
	eng.setRig( 'night', { fog: 0.0075, ambient: 0.8, moon: 1.3, exposure: 0.72, bloom: 0.3, lamp: 0.7, clouds: 1.0, cover: 0.36, cloudGain: 2.4, skyGlow: 4, haze: 0.35, glare: 24 } );
	eng.P.contrast.value = 1.06; eng.P.saturation.value = 1.03; eng.P.knee.value = 1.6; eng.P.kneeGlow.value = 2.8;
	eng.camera.near = 0.08; eng.camera.far = 1500; eng.camera.updateProjectionMatrix();
	mark( 'engine' );
	R.backend = eng.renderer.backend.isWebGPUBackend ? 'webgpu' : 'webgl';
	try { R.adapter = ( await navigator.gpu.requestAdapter() ).info.vendor; } catch ( e ) { R.adapter = null; }
	const gpuDone = async () => { await eng.renderer.backend.device?.queue.onSubmittedWorkDone?.(); };
	const nav = new Nav( bake.data );

	const Town = { packed: LITE.BakedTown, exact: GlbTown, lod: LodTown }[ V ];
	const town = window.town = Town ? new Town( eng, bake, { wait: SHOT ? 'all' : 'near' } ) : null;
	if ( ! town && eng.probe ) { eng.probe.ground = () => null; eng.probe.swap = () => {}; }
	const liveR = town ? town.R.slice() : null;
	if ( town && SCENE !== 'walk' ) town.R = town.R.map( ( r, i ) => ( i ? 0 : r ) );
	let portable = null;
	const loadPortable = async ( file ) => {

		const gltf = await makeLoader().loadAsync( `${GLB}/${file}` );
		eng.scene.add( gltf.scene );
		let tris = 0, meshes = 0, bytes = 0, texels = 0;
		const mats = new Set(), maps = new Set();
		gltf.scene.traverse( ( o ) => {

			if ( ! o.isMesh ) return;
			meshes ++; tris += ( o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count ) / 3;
			for ( const a of [ ...Object.values( o.geometry.attributes ), o.geometry.index ] ) if ( a ) bytes += a.array.byteLength;
			mats.add( o.material ); if ( o.material.map ) maps.add( o.material.map );

		} );
		for ( const t of maps ) if ( t.image ) texels += t.image.width * t.image.height;
		// (texture_mb: RGBA8 with its mip chain, what the GPU holds for the file's images)
		portable = { file, meshes, tris, materials: mats.size, textures: maps.size, geometry_mb: + ( bytes / 1048576 ).toFixed( 1 ), texture_mb: + ( texels * 4 * 4 / 3 / 1048576 ).toFixed( 1 ) };

	};

	say( 'loading' );
	if ( town ) await town.load( `${BASE}/out/town/town.json`, ( f, what ) => say( 'loading ' + ( what || '' ) + ' ' + Math.round( f * 100 ) + '%' ) );
	else await loadPortable( FILE || ( SCENE === 'lo' || SCENE === 'walk' ? 'portable_lo.glb' : `bench/portable_${SCENE}_spawn.glb` ) );
	mark( 'loaded' );

	// the spawn, as the page stands there
	const sp = bake.data.spawn, P = { x: sp.pos[ 0 ], y: sp.pos[ 1 ], yaw: THREE.MathUtils.degToRad( sp.yaw || 0 ) };
	let ground = nav.groundAt( P.x, P.y, 50 ) ?? 0;
	const place = () => {

		const g = nav.groundAt( P.x, P.y, ground + 1 );
		if ( g !== null && g !== undefined ) ground = g;
		eng.camera.position.set( P.x, ground + 1.65, - P.y );
		eng.camera.rotation.set( 0, P.yaw, 0, 'YXZ' );

	};

	let t = 0;
	const frame = () => { t += 1 / 60; eng.update( t ); if ( town ) town.update( eng.camera, t ); eng.render(); };
	place();
	if ( town ) await town.settle( eng.camera );
	frame();
	await gpuDone();
	snap( 'first' );
	say( 'first frame at ' + R.marks.first + ' ms' );

	// every level the view from here asks for is in and shown
	const idle = () => ! town || ( town.baseLeft <= 0 && town.pending === 0 && town.fades.size === 0 );
	const untilIdle = async ( limit = 180000 ) => {

		const t0 = performance.now();
		for ( let calm = 0; calm < 4 && performance.now() - t0 < limit; ) { await raf(); frame(); calm = idle() ? calm + 1 : 0; }
		await gpuDone();
		return performance.now() - t0 < limit;

	};

	R.settled = await untilIdle();
	snap( 'all' );

	if ( town && V !== 'lod' && ( SCENE === 'near' || SCENE === 'fine' ) ) {

		// that level of the cells around the spawn, asked for now: request to shown
		const before = fetched(), d0 = decode(), t0 = performance.now();
		town.R[ town.L.indexOf( SCENE ) ] = 20;
		town._dirty = true;
		R.settled = await untilIdle();
		const after = fetched(), d1 = decode();
		R.level = { ms: Math.round( performance.now() - t0 ), cells: town.chunks.filter( ( c ) => c.level === town.L.indexOf( SCENE ) ).length,
			n: after.geo_n - before.geo_n, wire: after.geo_wire - before.geo_wire, raw: after.geo_raw - before.geo_raw,
			decode_main_ms: d1.main_ms - d0.main_ms, decode_worker_busy_ms: d1.worker_busy_ms - d0.worker_busy_ms, long: longTasks( t0, performance.now() ) };
		snap( 'level' );

	}

	// what one frame draws from here, all passes of the frame (the mirror / probe passes come and go: the median of 9 frames)
	const DRAW = { n: 0, tris: 0 };
	{

		const info = eng.renderer.info, update = info.update;
		info.update = function ( object, count, instanceCount ) { DRAW.n ++; if ( object.isMesh || object.isSprite ) DRAW.tris += instanceCount * ( count / 3 ); return update.call( this, object, count, instanceCount ); };

	}

	const counts = async () => {

		const d = [], n = [];
		for ( let i = 0; i < 9; i ++ ) { await raf(); DRAW.n = DRAW.tris = 0; frame(); d.push( DRAW.n ); n.push( DRAW.tris ); } // (one frame per animation frame: the scene's pass is drawn once in each)
		return { draws: stat( d ).median, tris: Math.round( stat( n ).median ) };

	};

	const residency = () => {

		if ( V === 'lod' ) return town.lodStats();
		if ( ! town ) return portable;
		let bytes = 0;
		for ( const c of town.chunks ) c.mesh.forEach( ( mesh, l ) => { if ( mesh ) { const m = c.m[ town.L[ l ] ]; bytes += m.nv * ( m.w || 10 ) * 2 + m.ni * m.it; } } );
		const s = town.visibleStats();
		return { shown: s.lods, resident: s.resident, shown_tris: Math.round( s.tris ), geometry_mb: + ( bytes / 1048576 ).toFixed( 1 ) };

	};

	R.spawn = { ...( await counts() ), ...residency() };
	if ( town && town.warnings.length ) R.warnings = town.warnings.slice( 0, 5 );

	if ( SHOT ) {

		// the settled picture from the spawn and from a second place, for comparing variants pixel by pixel
		R.shots = [];
		for ( const [ x, y, yaw ] of [ [ P.x, P.y, P.yaw ], [ P.x + 6, P.y + 14, P.yaw + 2.2 ] ] ) {

			P.x = x; P.y = y; P.yaw = yaw; place();
			if ( town ) { await town.settle( eng.camera ); if ( town.sharp ) await town.sharp; if ( town.detail ) await town.detail; }
			await untilIdle( 30000 );
			// the same clock in every run: the flames and the water move with the engine's time
			// (three.js advances the shaders' time in NodeFrame.update() from the wall clock: held still here)
			const nf = eng.renderer._nodes.nodeFrame;
			nf.update = function () { this.frameId ++; this.deltaTime = 0; this.time = 1000; };
			t = 1000; for ( let i = 0; i < 6; i ++ ) { await raf(); frame(); }
			await gpuDone();
			const img = await eng.capture();
			const c = document.createElement( 'canvas' ); c.width = img.width; c.height = img.height;
			c.getContext( '2d' ).putImageData( img, 0, 0 );
			R.shots.push( c.toDataURL( 'image/png' ) );

		}

	} else if ( SCENE === 'walk' && WALK > 0 ) {

		// a walk straight on from the spawn, in real time
		say( 'walking' );
		const cpu = [], gap = [], w0 = performance.now(), b0 = fetched(), y0 = P.y;
		let last = w0, switches = 0, worstPending = 0;
		const levelOf = V === 'lod' ? () => town.lods.map( ( l ) => town.level( l ) ) : town ? () => town.chunks.map( ( c ) => c.level ) : () => [];
		let was = levelOf();
		for ( ;; ) {

			const now = await raf(), el = ( performance.now() - w0 ) / 1000;
			if ( el >= WALK ) break;
			P.y = y0 + SPEED * el;
			place();
			const c0 = performance.now();
			frame();
			cpu.push( performance.now() - c0 );
			gap.push( performance.now() - last ); last = performance.now();
			const is = levelOf();
			for ( let i = 0; i < is.length; i ++ ) if ( is[ i ] !== was[ i ] ) switches ++;
			was = is;
			if ( town ) worstPending = Math.max( worstPending, town.pending );

		}

		const w1 = performance.now(), b1 = fetched();
		R.walk = { seconds: WALK, metres: WALK * SPEED, frames: cpu.length, fps: + ( cpu.length / ( ( w1 - w0 ) / 1000 ) ).toFixed( 1 ), cpu_ms: stat( cpu ), frame_ms: stat( gap ), long: longTasks( w0, w1 ),
			level_switches: switches, most_pending: worstPending, pending_at_end: town ? town.pending : 0,
			geo_n: b1.geo_n - b0.geo_n, geo_wire: b1.geo_wire - b0.geo_wire, geo_raw: b1.geo_raw - b0.geo_raw };
		await untilIdle( 30000 );
		snap( 'walked' );
		R.end = { ...( await counts() ), ...residency() };

	}

	// (the sharp texture layers come in behind the geometry, the same for every variant: memory is read after them)
	if ( town ) { await town.sharp; await town.detail; for ( let i = 0; i < 5; i ++ ) { await raf(); frame(); } await gpuDone(); mark( 'textures' ); }
	R.decode = decode();
	R.fetched = fetched();
	if ( FORM === 'ranged' && town && town.gltf && town.gltf.userData.rangedSource ) R.ranges = town.gltf.userData.rangedSource.log.length;
	R.done = true;
	say( 'done' );
	$( 'out' ).textContent = JSON.stringify( { ...R, shots: R.shots ? R.shots.length : undefined }, null, 1 );

}

main().catch( ( e ) => { R.errors.push( String( e && e.stack || e ) ); R.done = true; R.failed = true; say( 'failed: ' + ( e.message || e ) ); } );
