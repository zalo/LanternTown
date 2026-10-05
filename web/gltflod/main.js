// Loads the town's MSFT_lod export with the patched GLTFLoader (lib/, copied from the two branches in
// upstream/three.js) and shows what the loading cost: time to the first frame, time until the view has the
// levels it asks for, bytes and requests. Times count from the call of loader.load().
//
// The default file is the portable one: stock glTF materials (KHR_materials_unlit, textures, vertex colours),
// Draco geometry, and nothing on this page touches what the loader made of it. The two "exact" files are the
// town's own vertex format and need the adapter below (?src=glb, ?src=bins).
//
//   ?tier=a        MSFT_lod as a plain THREE.LOD, everything loaded with the file
//   ?tier=b        (default) setLazyLOD( true ), and setRangeRequests( true ) for a .glb
//   ?src=portable|glb|bins  the portable .glb (default) | the exact format as a single .glb | with one .bin per level
//   ?lazy=0 ?range=0   switch one of the two tier B options off
//   ?all=1         load every level right after the first frame (and time it)
//   ?cam=x,y,z,tx,ty,tz | ?view=roofs|street|far|away      ?tint=1 colours the levels      ?file=URL any other file
//   ?workers=0     decode meshopt data on the main thread (default: in workers; Draco always decodes in workers)
//   ?tex=0         without the texture layers of the town's bake (see "materials" below)
//   ?lib=name      another build of the loader: ./lib/<name>.js (for measuring a change)
//
// The HUD says in plain words how the file came: in byte ranges (206) or, where the server or a proxy in between
// ignores Range, as one full download (200). Everything is addressed relatively. window.__gltflod has the numbers
// for scripts.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';

const params = new URLSearchParams( location.search );
const TIER = params.get( 'tier' ) === 'a' ? 'a' : 'b';
const SOURCES = { portable: '../out/web/_nextglb/portable_lod_spawn.glb', glb: '../out/web/_nextglb/exact_lod_spawn.glb', bins: '../out/web/_nextglb/exact_lodbake_spawn.gltf' };
const SRC = params.get( 'src' ) in SOURCES ? params.get( 'src' ) : 'portable';
const FILE = params.get( 'file' ) || SOURCES[ SRC ];
// The model files are regenerated under the same name. A range of an old copy in a cache put beside ranges of the new
// file would be garbage, so every page load asks under a URL of its own, and nothing of it is read from or kept in the
// HTTP cache (see fetch below).
const FILE_URL = FILE + ( FILE.includes( '?' ) ? '&' : '?' ) + 'v=' + Date.now().toString( 36 );
// The same for the code: index.html imports this module under a new URL on every page load, and the loader build is
// imported with that version (a phone was seen running a main.js of an hour before out of its cache).
const BUILD = new URL( import.meta.url ).searchParams.get( 'v' ) || Date.now().toString( 36 );
// ?refine=hop|step: experiments with the order of the levels that are NOT in the branch (lib/GLTFLoader.tierB.hop.js and
// .step.js: the branch's loader with three statements changed, see .scratch/gltflod/build_lib.mjs and docs/gltfloader/).
// The branch loads the level the camera asks for and nothing else. hop: a LOD that shows its coarsest level gets the
// next one first. step: every level on the way, one after the other; the LOD that is most levels behind goes first.
const REFINE = TIER === 'b' && params.get( 'lazy' ) !== '0' && [ 'hop', 'step' ].includes( params.get( 'refine' ) ) ? params.get( 'refine' ) : 'direct';
// ?dist=0.5 multiplies the distances at which the LODs of the file switch (three.js has no global LOD bias; the levels'
// distances are public). 1: as the file's MSFT_screencoverage says.
const DIST = Math.min( 4, Math.max( 0.05, Number( params.get( 'dist' ) ) || 1 ) );
const LIB = /^[\w./-]+$/.test( params.get( 'lib' ) || '' ) && ! ( params.get( 'lib' ) || '' ).includes( '..' ) ? params.get( 'lib' ) : null;
const WORKERS = params.get( 'workers' ) === '0' ? 0 : Math.max( 1, Math.min( 4, ( navigator.hardwareConcurrency || 2 ) - 1 ) );
const LAZY = TIER === 'b' && params.get( 'lazy' ) !== '0';
const RANGE = TIER === 'b' && params.get( 'range' ) !== '0' && /\.glb(\?|$)/.test( FILE );
const VIEWS = { roofs: [ 34, 26, - 8, 0, 0, 30 ], street: [ - 12, 6, - 2, 4, 5, 30 ], far: [ 120, 80, - 110, 0, 0, 30 ], away: [ 330, 200, - 330, 0, 0, 30 ] };
const VIEW = params.get( 'view' ) in VIEWS ? params.get( 'view' ) : 'roofs';
const LEVEL_NAMES = [ 'fine', 'near', 'mid', 'lo' ];
const LEVEL_TINTS = { fine: [ 1.6, 0.45, 0.4 ], near: [ 1.5, 1.2, 0.35 ], mid: [ 0.45, 1.5, 0.5 ], lo: [ 0.45, 0.7, 1.7 ] };

const hud = document.getElementById( 'hud' ), bar = document.getElementById( 'bar' );
const M = window.__gltflod = { tier: TIER, file: FILE, lazy: LAZY, range: RANGE, workers: WORKERS, lib: LIB, messages: [], error: null };

// ---- network: what the model files cost ----

const resources = [];
const isModelData = ( name ) => /\.(gltf|glb|bin)(\?|$)/.test( name );

// What the server answered, as it happens: every request for model data goes through here, with its Range header,
// the status of the answer and the bytes of the body as the loader reads them (a browser's Resource Timing has an
// entry only when a request is over, not every browser puts the status in it, and a request that the loader cancels
// half way has to be counted too). Bytes are those of the body; what gzip saved on the wire is shown beside them.
const transfers = [];
let transfersPeak = 0; // the most requests that were open at the same time
const nativeFetch = window.fetch.bind( window );
window.fetch = function ( input, init ) {

	const url = typeof input === 'string' ? input : input.url;
	if ( ! isModelData( url ) ) return nativeFetch( input, init );
	const headers = input instanceof Request ? input.headers : new Headers( init && init.headers );
	const t = { name: url.split( '/' ).pop().split( '?' )[ 0 ], range: headers.get( 'Range' ), start: performance.now(), status: 0, bytes: 0, end: undefined, encoding: null, contentRange: null, failed: false, cancelled: false, log: [] };
	transfers.push( t );
	transfersPeak = Math.max( transfersPeak, transfers.filter( ( x ) => x.end === undefined ).length );
	return nativeFetch( input, Object.assign( {}, init, { cache: 'no-store' } ) ).then( ( response ) => {

		t.status = response.status; t.encoding = response.headers.get( 'Content-Encoding' ); t.contentRange = response.headers.get( 'Content-Range' );
		t.total = t.encoding ? 0 : Number( response.headers.get( 'Content-Length' ) ) || 0; // of a compressed answer the length of the body is not known in advance
		if ( ! response.body || typeof ReadableStream === 'undefined' ) { t.end = performance.now(); t.bytes = Number( response.headers.get( 'Content-Length' ) ) || 0; t.log.push( t.end, t.bytes ); return response; }
		const reader = response.body.getReader();
		const counted = new ReadableStream( {

			pull( controller ) {

				return reader.read().then( ( { done, value } ) => {

					if ( done ) { t.end = performance.now(); controller.close(); return; }
					t.bytes += value.byteLength; t.log.push( performance.now(), t.bytes );
					controller.enqueue( value );

				}, ( error ) => { t.end = performance.now(); t.failed = true; controller.error( error ); } );

			},
			cancel( reason ) { t.end = performance.now(); t.cancelled = true; return reader.cancel( reason ); }

		}, { highWaterMark: 0 } );
		const wrapped = new Response( counted, { status: response.status, statusText: response.statusText, headers: response.headers } );
		Object.defineProperty( wrapped, 'url', { value: response.url } );
		return wrapped;

	}, ( error ) => { t.end = performance.now(); t.failed = true; throw error; } );

};

// -> how the model data came, for the HUD and for scripts
function transport() {

	const ranged = transfers.filter( ( t ) => t.range ), answered = transfers.filter( ( t ) => t.status );
	const out = { requests: transfers.length, inFlight: transfers.filter( ( t ) => t.end === undefined ).length, bytes: transfers.reduce( ( s, t ) => s + t.bytes, 0 ), announced: transfers.reduce( ( s, t ) => s + ( t.cancelled || t.failed ? t.bytes : Math.max( t.total || 0, t.bytes ) ), 0 ), peak: transfersPeak, cancelled: transfers.filter( ( t ) => t.cancelled ).length,
		ranged: ranged.length, partial: ranged.filter( ( t ) => t.status === 206 ).length, ignored: ranged.filter( ( t ) => t.status === 200 ).length, failed: transfers.filter( ( t ) => t.failed || t.status >= 400 ).length,
		gzip: answered.some( ( t ) => t.encoding ), protocol: [ ...new Set( performance.getEntriesByType( 'resource' ).filter( ( e ) => isModelData( e.name ) && e.nextHopProtocol ).map( ( e ) => e.nextHopProtocol ) ) ].join( ', ' ) };
	out.fileBytes = transfers.reduce( ( n, t ) => Math.max( n, t.contentRange ? Number( t.contentRange.split( '/' )[ 1 ] ) || 0 : 0 ), 0 ) || ( /\.glb(\?|$)/.test( FILE ) ? transfers.reduce( ( n, t ) => Math.max( n, t.status === 200 ? t.total || 0 : 0 ), 0 ) : 0 ) || null; // the size of the file, as the Content-Range of a 206 or the Content-Length of a 200 says it
	out.mode = ranged.length === 0 ? ( answered.length ? 'whole' : 'pending' ) : out.ignored > 0 ? 'fallback' : out.partial > 0 ? 'ranges' : 'pending';
	return out;

}
performance.setResourceTimingBufferSize( 4000 );
new PerformanceObserver( ( list ) => {

	for ( const e of list.getEntries() ) if ( isModelData( e.name ) ) resources.push( { name: e.name.split( '/' ).pop(), end: e.responseEnd, wire: e.transferSize, body: e.encodedBodySize, decoded: e.decodedBodySize, status: e.responseStatus } );

} ).observe( { type: 'resource', buffered: true } );

// -> requests started and bytes read up to a moment; wire: what the finished requests cost on the wire (Resource Timing)
function network( until = Infinity ) {

	const out = { requests: 0, wire: 0, body: 0, decoded: 0, saved: 0 };
	for ( const t of transfers ) {

		if ( t.start > until ) continue;
		out.requests ++;
		let bytes = 0;
		for ( let i = 0; i < t.log.length && t.log[ i ] <= until; i += 2 ) bytes = t.log[ i + 1 ];
		out.body += bytes;

	}

	out.wire = out.decoded = out.body;
	for ( const r of resources ) if ( r.end <= until && r.decoded > r.body ) out.saved += r.decoded - r.body;
	return out;

}

// A visitor who leaves while files download aborts the requests, and the browser reports each as a failed fetch just
// before the page goes. That is not an error of the page: what looks like one is held back for a moment and dropped
// if the page is on its way out by then.
let leaving = false;
addEventListener( 'pagehide', () => { leaving = true; } );
addEventListener( 'pageshow', () => { leaving = false; } );
const ABORTED = /Load failed|Failed to fetch|NetworkError|aborted|cancell?ed/i;
const unlessLeaving = ( report ) => setTimeout( () => { if ( leaving === false ) report(); }, 400 );

for ( const kind of [ 'warn', 'error' ] ) {

	const original = console[ kind ];
	console[ kind ] = ( ...args ) => {

		const text = args.map( ( a ) => a && a.message || String( a ) ).join( ' ' ).slice( 0, 240 );
		const report = () => { M.messages.push( kind + ': ' + text ); original.apply( console, args ); };
		if ( ABORTED.test( text ) ) unlessLeaving( report ); else report();

	};

}

// ---- scene ----

const renderer = new THREE.WebGLRenderer( { antialias: true, powerPreference: 'high-performance' } );
renderer.setPixelRatio( Math.min( devicePixelRatio, 1 ) ); // (pixel ratio 1 everywhere: the user's call)
renderer.setSize( innerWidth, innerHeight );
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
document.body.prepend( renderer.domElement );

const scene = new THREE.Scene();
scene.background = new THREE.Color( 0x0b1020 );
scene.add( new THREE.HemisphereLight( 0xbfd4ff, 0x2a2622, 2.2 ) );
const sun = new THREE.DirectionalLight( 0xffffff, 2 ); sun.position.set( - 0.45, 0.62, 0.35 ); scene.add( sun );

const camera = new THREE.PerspectiveCamera( 50, innerWidth / innerHeight, 0.3, 3000 );
const controls = new OrbitControls( camera, renderer.domElement );
controls.enableDamping = true; controls.dampingFactor = 0.12; controls.maxPolarAngle = Math.PI * 0.495;

function setCamera( x, y, z, tx, ty, tz ) {

	camera.position.set( x, y, z ); controls.target.set( tx, ty, tz ); controls.update();

}

M.setCamera = setCamera;
const cam = ( params.get( 'cam' ) || '' ).split( ',' ).map( Number );
setCamera( ...( cam.length === 6 && cam.every( Number.isFinite ) ? cam : VIEWS[ VIEW ] ) );

addEventListener( 'resize', () => { renderer.setSize( innerWidth, innerHeight ); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); } );

// keys: WASD / arrows move over the ground, Q E down and up, shift faster, L colours the levels
const keys = new Set();
addEventListener( 'keydown', ( e ) => { keys.add( e.code ); if ( e.code === 'KeyL' ) setTint( ! tinted ); } );
addEventListener( 'keyup', ( e ) => keys.delete( e.code ) );
addEventListener( 'blur', () => keys.clear() );
const _forward = new THREE.Vector3(), _right = new THREE.Vector3(), _move = new THREE.Vector3();

function moveByKeys( dt ) {

	if ( keys.size === 0 ) return;
	camera.getWorldDirection( _forward ); _forward.y = 0; _forward.normalize(); _right.crossVectors( _forward, camera.up );
	const f = ( keys.has( 'KeyW' ) || keys.has( 'ArrowUp' ) ? 1 : 0 ) - ( keys.has( 'KeyS' ) || keys.has( 'ArrowDown' ) ? 1 : 0 );
	const r = ( keys.has( 'KeyD' ) || keys.has( 'ArrowRight' ) ? 1 : 0 ) - ( keys.has( 'KeyA' ) || keys.has( 'ArrowLeft' ) ? 1 : 0 );
	const u = ( keys.has( 'KeyE' ) ? 1 : 0 ) - ( keys.has( 'KeyQ' ) ? 1 : 0 );
	_move.set( 0, 0, 0 ).addScaledVector( _forward, f ).addScaledVector( _right, r ); _move.y = u;
	if ( _move.lengthSq() === 0 ) return;
	_move.multiplyScalar( ( keys.has( 'ShiftLeft' ) || keys.has( 'ShiftRight' ) ? 60 : 15 ) * dt );
	camera.position.add( _move ); controls.target.add( _move );

}

// ---- materials ----
// A file with stock glTF materials is drawn as GLTFLoader returns it. The town's "exact" export is not such a file:
// it has no textures and no normals. Its meshes carry the bake's records - the light on every vertex (_COL), the
// index of a texture layer (the fourth component of _PL) and tile coordinates (TEXCOORD_0) - and its five materials
// only name the kind of surface (extras.baked_kind: 0 opaque, 1 glass, 2 water, 3 falling water, 4 leaf cards).
// The colour of the surfaces is in 48 texture layers that are not part of the glTF file: WebP pages next to the
// bake, listed in the scene's extras. This adapter draws light x layer as the town's runtime does at its simplest
// (web/next/engine/lite.js), with the leaf cards cut out by the height in the leaf layer's alpha. ?tex=0: light alone.

let tinted = params.get( 'tint' ) === '1';
const bakedMaterials = new Map();
const TEX = params.get( 'tex' ) !== '0';
const layers = { value: null }; // one uniform for all materials: the layer array, a white texel until the pages are in

{

	const blank = new THREE.DataArrayTexture( new Uint8Array( 4 ).fill( 255 ), 1, 1, 1 );
	blank.colorSpace = THREE.SRGBColorSpace; blank.wrapS = blank.wrapT = THREE.RepeatWrapping; blank.needsUpdate = true;
	layers.value = blank;

}

const BAKED_VERTEX = `
	attribute vec4 _col; attribute vec4 _pl;
	uniform float uScale; uniform float uUvRange;
	varying vec3 vLit; varying float vAux; varying float vLayer; varying vec2 vTiles;
	void main() {
		vLit = _col.rgb * _col.rgb * uScale; vAux = _col.a; vLayer = _pl.w * 65535.0;
		vTiles = ( uv * ( 65535.0 / 65536.0 ) - 0.5 ) * 2.0 * uUvRange;
		gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
	}`;
const BAKED_FRAGMENT = `
	uniform sampler2DArray uLayers; uniform vec3 uTint; uniform float uCutAux; uniform float uTextured;
	varying vec3 vLit; varying float vAux; varying float vLayer; varying vec2 vTiles;
	void main() {
		vec3 c = vLit; float a = 1.0;
		#if KIND == 0
			c *= mix( vec3( 0.6 ), texture( uLayers, vec3( vTiles, floor( vLayer + 0.5 ) ) ).rgb, uTextured );
		#elif KIND == 1
			a = vAux;
		#elif KIND == 2
			c = c * 0.3 + vec3( 0.018, 0.034, 0.085 ) * 0.25;
		#elif KIND == 3
			c *= 0.5; a = 0.45 * vAux;
		#elif KIND == 4
			vec4 t = texture( uLayers, vec3( vTiles * vAux * uCutAux, floor( vLayer + 0.5 ) ) );
			vec2 cell = ( fract( vTiles ) - 0.5 ) * 2.0;
			if ( uTextured > 0.5 && t.a * 2.0 - 1.0 <= smoothstep( 0.45, 1.0, length( cell ) ) * 0.8 + 0.33 ) discard;
			c *= mix( vec3( 0.6 ), t.rgb, uTextured );
		#endif
		gl_FragColor = vec4( c * uTint, a );
		#include <tonemapping_fragment>
		#include <colorspace_fragment>
	}`;
const textured = { value: 0 };

function bakedMaterial( source, levelName, baked ) {

	const key = source.uuid + ':' + levelName;
	let material = bakedMaterials.get( key );
	if ( material ) return material;
	const kind = source.userData.baked_kind | 0;
	material = new THREE.ShaderMaterial( {
		name: source.name + ':' + levelName,
		defines: { KIND: kind },
		uniforms: { uScale: { value: baked.colour_scale || 64 }, uUvRange: { value: baked.uv_range || 32 }, uCutAux: { value: baked.cut_aux || 4 }, uLayers: layers, uTextured: textured, uTint: { value: new THREE.Vector3( 1, 1, 1 ) } },
		vertexShader: BAKED_VERTEX, fragmentShader: BAKED_FRAGMENT,
		transparent: kind === 1 || kind === 3, depthWrite: ! ( kind === 1 || kind === 3 ), side: kind >= 3 ? THREE.DoubleSide : source.side
	} );
	material.userData.levelName = levelName;
	bakedMaterials.set( key, material );
	applyTint( material );
	return material;

}

// the bake's texture layers: pages of cols x rows layers each -> one array texture (as _loadLayers() of lite.js)
async function loadLayers( baked ) {

	const sizes = Object.keys( baked.tex || {} ).map( Number ).sort( ( x, y ) => x - y );
	if ( ! TEX || sizes.length === 0 || ! baked.layers ) return;
	const t = baked.tex[ matchMedia( '(pointer: coarse)' ).matches ? sizes[ 0 ] : sizes[ sizes.length - 1 ] ]; // a phone takes the small pages
	const size = t.size, n = baked.layers.length, per = t.cols * t.rows;
	const data = new Uint8Array( size * size * 4 * n );
	M.layers = { size, count: n, pages: t.pages.length, bytes: 0, loaded: false };
	await Promise.all( t.pages.map( async ( name, p ) => {

		const r = await fetch( new URL( '../_next/' + name + '.webp?v=' + baked.baked, new URL( FILE, location.href ) ) );
		if ( ! r.ok ) throw new Error( name + '.webp: ' + r.status );
		const blob = await r.blob();
		M.layers.bytes += blob.size;
		const bmp = await createImageBitmap( blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
		const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( bmp.width, bmp.height ) : Object.assign( document.createElement( 'canvas' ), { width: bmp.width, height: bmp.height } );
		const cx = cv.getContext( '2d', { willReadFrequently: true } );
		cx.globalCompositeOperation = 'copy';
		cx.drawImage( bmp, 0, 0 );
		if ( bmp.close ) bmp.close();
		for ( let i = p * per; i < Math.min( n, ( p + 1 ) * per ); i ++ ) {

			const k = i - p * per, col = k % t.cols, row = ( k - col ) / t.cols;
			data.set( cx.getImageData( col * size, row * size, size, size ).data, i * size * size * 4 );

		}

	} ) );
	const tex = new THREE.DataArrayTexture( data, size, size, n );
	tex.colorSpace = THREE.SRGBColorSpace; tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter; tex.generateMipmaps = true;
	tex.anisotropy = Math.min( 8, renderer.capabilities.getMaxAnisotropy() );
	tex.needsUpdate = true;
	layers.value.dispose(); layers.value = tex; textured.value = 1;
	M.layers.loaded = true;

}

function applyTint( material ) {

	const t = tinted && LEVEL_TINTS[ material.userData.levelName ] || [ 1, 1, 1 ];
	material.uniforms.uTint.value.set( t[ 0 ], t[ 1 ], t[ 2 ] );

}

const stockMeshes = [], tintedCopies = new Map();

function tintedCopy( material, levelName ) {

	const tint = LEVEL_TINTS[ levelName ];
	if ( ! tint || ! material.color ) return material;
	const key = material.uuid + levelName;
	if ( ! tintedCopies.has( key ) ) { const copy = material.clone(); copy.color.r *= tint[ 0 ]; copy.color.g *= tint[ 1 ]; copy.color.b *= tint[ 2 ]; tintedCopies.set( key, copy ); }
	return tintedCopies.get( key );

}

function setTint( on ) {

	tinted = on;
	for ( const material of bakedMaterials.values() ) applyTint( material );
	for ( const entry of stockMeshes ) entry.mesh.material = on ? tintedCopy( entry.material, entry.levelName ) : entry.material;
	buildBar();

}

// ---- load ----

let gltf = null, lods = [], lodInfo = new Map(), totalLevels = 0, levelNameOfNode = new Map();
const _a = new THREE.Vector3(), _b = new THREE.Vector3();
const probe = new THREE.PerspectiveCamera();

function levelNameOf( def, level ) {

	// "c_0_1:near" in the exact files, "c_0_1_near" in the portable one, where the finest level has the bare name
	const tail = /(?:[:_])(fine|near|mid|lo)$/.exec( String( def && def.name || '' ) );
	return tail ? tail[ 1 ] : 'L' + level;

}

function describe( json ) {

	// which LOD node and level every level node belongs to (from the file alone)
	const byNode = new Map();
	( json.nodes || [] ).forEach( ( def, index ) => {

		const ids = def.extensions && def.extensions.MSFT_lod && def.extensions.MSFT_lod.ids;
		if ( ! ids ) return;
		const names = [ index, ...ids ].map( ( n, level ) => levelNameOf( json.nodes[ n ], level ) );
		if ( names[ 0 ] === 'L0' && names.length > 1 && LEVEL_NAMES.includes( names[ names.length - 1 ] ) ) names[ 0 ] = names.length === 2 ? 'mid' : 'fine';
		[ index, ...ids ].forEach( ( n, level ) => byNode.set( n, { node: index, level, count: ids.length + 1, names } ) );
		totalLevels += ids.length + 1;

	} );
	return byNode;

}

// Meshes arrive with the file or, lazily, any time later: give the new ones their material.
function adoptMeshes() {

	const baked = gltf.scene.userData.baked || {};
	gltf.scene.traverse( ( o ) => {

		if ( ! o.isMesh || o.userData.adopted ) return;
		o.userData.adopted = true;
		let holder = o;
		while ( holder.parent && ! holder.parent.isLOD ) holder = holder.parent;
		const lod = holder.parent, info = lod && lodInfo.get( lod );
		const level = lod ? lod.levels.findIndex( ( l ) => l.object === holder ) : - 1;
		const levelName = info && info.names[ level ] || 'L' + level;
		if ( ! o.geometry.attributes._col || ! o.geometry.attributes._pl ) {

			// stock materials stay as the loader made them; only "colour levels" swaps in a tinted copy
			stockMeshes.push( { mesh: o, material: o.material, levelName } );
			if ( tinted ) o.material = tintedCopy( o.material, levelName );
			return;

		}

		if ( ! M.adapter ) {

			M.adapter = true;
			if ( baked.layers ) loadLayers( baked ).catch( ( e ) => console.warn( 'texture layers: ' + ( e && e.message || e ) ) );

		}

		o.material = bakedMaterial( o.material, levelName, baked );

	} );

}

function inspectLods() {

	const associations = gltf.parser.associations;
	for ( const lod of lods ) {

		if ( lodInfo.has( lod ) ) continue;
		for ( const level of lod.levels ) {

			const child = level.object.children[ 0 ], mapping = child && associations.get( child );
			const info = mapping && levelNameOfNode.get( mapping.nodes );
			if ( info ) { lodInfo.set( lod, info ); break; }

		}

	}

}

function countLevels() {

	let loaded = 0;
	for ( const lod of lods ) {

		const info = lodInfo.get( lod ), count = info ? info.count : lod.levels.length;
		for ( let i = 0; i < count && i < lod.levels.length; i ++ ) if ( lod.levels[ i ].object.children.length > 0 ) loaded ++;

	}

	return loaded;

}

// the level LOD.update() picks for the camera
function wantedLevel( lod, cam ) {

	const d = _a.setFromMatrixPosition( cam.matrixWorld ).distanceTo( _b.setFromMatrixPosition( lod.matrixWorld ) ) / cam.zoom;
	let k = 0;
	for ( let i = 1; i < lod.levels.length; i ++ ) { if ( d >= lod.levels[ i ].distance ) k = i; else break; }
	return k;

}

function viewState() {

	// deficit: how many steps too coarse the picture is, summed over the LODs; nearest: the same for the LOD next to the camera
	const showing = {}; let missing = 0, deficit = 0, worst = 0, nearest = 0, nearestDistance = Infinity;
	for ( const lod of lods ) {

		const info = lodInfo.get( lod ), count = info ? info.count : lod.levels.length;
		const wanted = wantedLevel( lod, camera ), current = lod.getCurrentLevel();
		if ( wanted < count && lod.levels[ wanted ].object.children.length === 0 ) missing ++;
		const steps = Math.max( 0, current - wanted ), d = _a.setFromMatrixPosition( camera.matrixWorld ).distanceTo( _b.setFromMatrixPosition( lod.matrixWorld ) );
		deficit += steps; worst = Math.max( worst, steps );
		if ( d < nearestDistance ) { nearestDistance = d; nearest = steps; }
		const name = info && info.names[ current ] || ( current < count ? 'L' + current : 'none' );
		showing[ name ] = ( showing[ name ] || 0 ) + 1;

	}

	return { showing, missing, deficit, worst, nearest };

}

// "Load all levels": with the public API alone - a LOD loads the level its update() is asked for, so ask each
// LOD for its missing levels with a camera at the right distance, the coarsest missing one first.
let sweeping = false, sweepProgress = 0;

function sweep( now ) {

	let pending = 0;
	const loaded = countLevels();
	if ( loaded !== sweep.loaded ) { sweep.loaded = loaded; sweepProgress = now; }
	for ( const lod of lods ) {

		const info = lodInfo.get( lod ), count = info ? info.count : lod.levels.length;
		let target = - 1;
		for ( let i = count - 1; i >= 0; i -- ) if ( lod.levels[ i ].object.children.length === 0 ) { target = i; break; }
		if ( target === - 1 ) { lod.autoUpdate = true; continue; }
		pending ++;
		lod.autoUpdate = false;
		_b.setFromMatrixPosition( lod.matrixWorld );
		probe.position.copy( _b ); probe.position.y += target === 0 ? 0 : lod.levels[ target ].distance * 1.0001;
		probe.updateMatrixWorld();
		lod.update( probe );

	}

	if ( pending === 0 || now - sweepProgress > 60000 ) {

		sweeping = false;
		for ( const lod of lods ) lod.autoUpdate = true;
		if ( pending > 0 ) M.messages.push( 'error: load all levels gave up, ' + pending + ' LODs still have a missing level' );
		buildBar();

	}

}

M.loadAll = () => { if ( gltf && countLevels() < totalLevels ) { sweeping = true; sweepProgress = performance.now(); M.tSweepStart = M.tSweepStart || sweepProgress; buildBar(); } };

async function load() {

	const { GLTFLoader } = await import( ( LIB ? './lib/' + LIB + '.js' : TIER === 'a' ? './lib/GLTFLoader.tierA.js' : REFINE !== 'direct' ? './lib/GLTFLoader.tierB.' + REFINE + '.js' : './lib/GLTFLoader.tierB.js' ) + '?v=' + BUILD );
	await MeshoptDecoder.ready;
	// with workers the loader's decoder calls (decodeGltfBufferAsync) leave the main thread: the frames keep coming while a level of some megabytes is unpacked
	if ( WORKERS > 0 ) MeshoptDecoder.useWorkers( WORKERS );
	// the decoder (a .wasm of 190 kB) is fetched now, beside the file, and not when the first Draco mesh turns up
	const draco = new DRACOLoader().setDecoderPath( new URL( '../node_modules/three/examples/jsm/libs/draco/gltf/', location.href ).href );
	draco.preload();
	const loader = new GLTFLoader().setMeshoptDecoder( MeshoptDecoder ).setDRACOLoader( draco );
	if ( LAZY ) loader.setLazyLOD( true );
	if ( RANGE ) loader.setRangeRequests( true );
	M.tStart = performance.now();
	gltf = await loader.loadAsync( FILE_URL );
	M.tLoaded = performance.now();
	levelNameOfNode = describe( gltf.parser.json );
	gltf.scene.traverse( ( o ) => { if ( o.isLOD ) lods.push( o ); } );
	M.lods = lods.length; M.totalLevels = totalLevels;
	if ( DIST !== 1 ) for ( const lod of lods ) for ( const level of lod.levels ) level.distance *= DIST;
	inspectLods();
	scene.add( gltf.scene );
	M.gltf = gltf;
	M.draco = ( gltf.parser.json.extensionsUsed || [] ).includes( 'KHR_draco_mesh_compression' );

}

// ---- frame loop ----

let last = performance.now(), frames = 0, fps = 0, fpsAt = last, hudAt = 0;

function frame( now ) {

	requestAnimationFrame( frame );
	const dt = Math.min( ( now - last ) / 1000, 0.1 ); last = now;
	moveByKeys( dt );
	controls.update();

	if ( gltf ) {

		adoptMeshes();
		if ( sweeping ) sweep( now );

	}

	renderer.render( scene, camera );

	if ( gltf ) {

		const done = performance.now();
		if ( M.tFirstFrame === undefined ) {

			M.tFirstFrame = done;
			window.__tele?.mark( 'first-frame' );
			M.levelsAtFirstFrame = countLevels();
			M.worstAtFirstFrame = viewState().worst;
			M.trianglesAtFirstFrame = renderer.info.render.triangles;
			if ( params.get( 'all' ) === '1' ) M.loadAll();

		}

		const loaded = countLevels(), view = viewState();
		M.levels = loaded; M.showing = view.showing; M.missing = view.missing; M.worst = view.worst;
		// on the way to the complete view: when no LOD was more than 2 levels, and more than 1 level, coarser than the camera asks for
		// (where fine is due: mid or better, near or better), and when the LOD next to the camera had its level
		if ( M.within === undefined ) M.within = {};
		for ( const k of [ 2, 1 ] ) if ( M.within[ k ] === undefined && view.worst <= k ) M.within[ k ] = done;
		if ( M.tNearest === undefined && view.nearest === 0 ) M.tNearest = done;
		// the way to the complete view: [ ms since load(), steps too coarse in sum, of the nearest LOD ] whenever it changes
		if ( M.timeline === undefined ) M.timeline = [];
		const lastStep = M.timeline[ M.timeline.length - 1 ];
		if ( M.timeline.length < 400 && ( ! lastStep || lastStep[ 1 ] !== view.deficit || lastStep[ 2 ] !== view.nearest ) ) M.timeline.push( [ Math.round( done - M.tStart ), view.deficit, view.nearest ] );
		// the view is settled when every LOD has the level the camera asks for
		if ( view.missing === 0 ) { if ( M.tSettled === undefined ) { M.tSettled = done; window.__tele?.mark( 'view-complete' ); } if ( M.unsettledSince !== undefined ) { M.lastSettle = done - M.unsettledSince; M.unsettledSince = undefined; } } else if ( M.tSettled !== undefined && M.unsettledSince === undefined ) M.unsettledSince = done;
		if ( loaded === totalLevels && M.tAll === undefined ) M.tAll = done;

	}

	frames ++;
	if ( now - fpsAt >= 500 ) { fps = frames * 1000 / ( now - fpsAt ); frames = 0; fpsAt = now; }
	if ( now - hudAt >= 200 ) { hudAt = now; drawHud(); }

}

const MB = ( n ) => ( n / 1048576 ).toFixed( n < 10485760 ? 2 : 1 ) + ' MB';
const MS = ( t ) => t === undefined || Number.isNaN( t ) ? '...' : t < 10000 ? Math.round( t ) + ' ms' : ( t / 1000 ).toFixed( 1 ) + ' s';
const T = ( t ) => MS( t === undefined ? undefined : t - M.tStart ); // times count from the call of loader.load()
const NET = ( n ) => `${MB( n.body )} in ${n.requests} request${n.requests === 1 ? '' : 's'}`;

// What a visitor's report carries (engine/telemetry.js reads window.__stats at every report): the same facts as the
// HUD, as numbers. One object, updated in place.
const stats = window.__stats = { page: 'gltflod', tier: TIER, src: params.get( 'file' ) ? 'file' : SRC, file: FILE.split( '/' ).pop().split( '?' )[ 0 ], file_bytes: null, lazy: LAZY, range: RANGE, lib: LIB || null, refine: REFINE };

function publish() {

	const tr = M.transport || transport(), since = ( t ) => t === undefined ? null : Math.round( t - M.tStart );
	const at = ( n ) => n ? { requests: n.requests, bytes: n.body } : null;
	Object.assign( stats, {
		transport: tr.mode, // 'ranges': 206 answers | 'fallback': a Range request was answered 200 with the whole file | 'whole': no ranges asked for | 'pending'
		range_requests: tr.ranged, ranges_honoured_206: tr.partial, fell_back_to_full_200: tr.ignored,
		requests: tr.requests, requests_in_flight: tr.inFlight, concurrent_peak: tr.peak, requests_failed: tr.failed, requests_stopped_early: tr.cancelled,
		file_bytes: tr.fileBytes, bytes_fetched: tr.bytes, gzip: tr.gzip, protocol: tr.protocol || null,
		onload_ms: since( M.tLoaded ), first_frame_ms: since( M.tFirstFrame ), view_complete_ms: since( M.tSettled ), all_levels_ms: since( M.tAll ),
		within2_ms: since( M.within && M.within[ 2 ] ), within1_ms: since( M.within && M.within[ 1 ] ), nearest_right_ms: since( M.tNearest ), // no LOD over 2 / over 1 levels coarser than the camera asks for; the LOD next to the camera has its level
		at_first_frame: at( M.networkAtFirstFrame ), at_within2: at( M.networkWithin && M.networkWithin[ 2 ] ), at_within1: at( M.networkWithin && M.networkWithin[ 1 ] ), at_view_complete: at( M.networkAtSettled ),
		levels_too_coarse_at_first_frame: M.worstAtFirstFrame ?? null, levels_too_coarse_now: M.worst ?? null, lod_distance_scale: DIST, build: BUILD,
		lods: M.lods ?? null, levels_total: M.totalLevels ?? null, levels_loaded: M.levels ?? null, levels_at_first_frame: M.levelsAtFirstFrame ?? null, lods_waiting_for_a_level: M.missing ?? null, showing: M.showing || null,
		decoder: M.gltf ? ( M.draco ? 'draco' : 'meshopt' ) : null, adapter: !! M.adapter,
		level_errors: M.messages.filter( ( m ) => m.includes( 'level of detail' ) ).length, errors: M.messages.filter( ( m ) => m.startsWith( 'error' ) ).length, warnings: M.messages.filter( ( m ) => m.startsWith( 'warn' ) ).length,
		error: M.error, fps: Math.round( fps ), triangles: renderer.info.render.triangles
	} );

}

function drawHud() {

	const lines = [];
	const what = TIER === 'a' ? 'tier A: LOD, all levels loaded with the file' : 'tier B: ' + ( [ LAZY && 'levels on demand', RANGE && 'range requests' ].filter( Boolean ).join( ' + ' ) || 'both options off' );
	lines.push( `<b>${what}</b>\n<span class="dim">${FILE.split( '/' ).pop()}${SRC === 'bins' && ! params.get( 'file' ) ? ' (one .bin per level)' : ''}${M.adapter ? '   the town\'s own vertex format: drawn by this page\'s adapter, not by stock materials' : ''}</span>` );
	if ( M.error ) lines.push( `<span class="bad">${M.error}</span>` );
	const tr = M.transport = transport();
	const how = tr.protocol ? ` <span class="dim">${tr.protocol}</span>` : '';
	if ( tr.mode === 'ranges' ) lines.push( `transport      <b>byte ranges honoured</b>: ${tr.partial} answered 206${how}` );
	else if ( tr.mode === 'fallback' ) lines.push( `<span class="bad">transport      RANGE IGNORED: the server (or a proxy) answered 200 with the whole\n               file. One download of all of it, read as it arrives: a level shows when its bytes are in.</span>${how}` );
	else if ( tr.mode === 'whole' ) lines.push( `transport      whole files, no ranges asked for${tr.gzip ? ' (gzip)' : ''}${how}` );
	else lines.push( 'transport      waiting for the first answer ...' );
	const now = network();
	M.network = now; M.networkAtFirstFrame = M.tFirstFrame !== undefined ? network( M.tFirstFrame ) : null; M.networkAtSettled = M.tSettled !== undefined ? network( M.tSettled ) : null; M.networkAtAll = M.tAll !== undefined ? network( M.tAll ) : null;
	lines.push( `first frame    <b>${T( M.tFirstFrame )}</b>${M.networkAtFirstFrame ? '   ' + NET( M.networkAtFirstFrame ) + `   <span class="dim">onLoad ${T( M.tLoaded )}, ${M.levelsAtFirstFrame}/${totalLevels} levels</span>` : ''}` );
	M.networkWithin = {}; for ( const k of [ 2, 1 ] ) M.networkWithin[ k ] = M.within && M.within[ k ] !== undefined ? network( M.within[ k ] ) : null;
	if ( M.tFirstFrame === undefined && tr.requests > 0 ) lines.push( `downloading    <b>${MB( tr.bytes )}</b>${tr.announced > tr.bytes ? ` of ${MB( tr.announced )} (${Math.floor( 100 * tr.bytes / tr.announced )} %)` : ''}${TIER === 'a' || ! LAZY ? '   <span class="dim">all of the file before the first frame</span>' : ''}` );
	if ( M.worstAtFirstFrame >= 2 ) lines.push( `mid or better  <b>${T( M.within[ 2 ] )}</b>${M.networkWithin[ 2 ] ? '   ' + NET( M.networkWithin[ 2 ] ) : ''}   <span class="dim">no LOD over 2 levels too coarse</span>` );
	if ( M.worstAtFirstFrame >= 1 ) lines.push( `near or better <b>${T( M.within[ 1 ] )}</b>${M.networkWithin[ 1 ] ? '   ' + NET( M.networkWithin[ 1 ] ) : ''}   <span class="dim">none over 1 level; next to the camera right at ${T( M.tNearest )}</span>` );
	lines.push( `view complete  <b>${T( M.tSettled )}</b>${M.networkAtSettled ? '   ' + NET( M.networkAtSettled ) : ''}${M.unsettledSince !== undefined ? '   <span class="dim">loading for the new view ...</span>' : M.lastSettle !== undefined ? `   <span class="dim">last move: ${MS( M.lastSettle )}</span>` : ''}` );
	lines.push( `all levels     <b>${M.tAll !== undefined ? T( M.tAll ) : ( M.levels ?? 0 ) + '/' + totalLevels + ' loaded'}</b>${M.networkAtAll ? '   ' + NET( M.networkAtAll ) : ''}` );
	lines.push( `fetched so far <b>${MB( tr.bytes )}</b> in ${tr.requests} request${tr.requests === 1 ? '' : 's'}${tr.inFlight ? `, <b>${tr.inFlight} in flight</b>` : ''}${tr.cancelled ? `, ${tr.cancelled} stopped early by the loader` : ''}${tr.failed ? `, <span class="bad">${tr.failed} failed</span>` : ''}${now.saved ? `   <span class="dim">gzip saved ${MB( now.saved )} on the wire</span>` : ''}` );
	if ( gltf ) {

		const s = M.showing || {};
		const rank = ( k ) => LEVEL_NAMES.includes( k ) ? LEVEL_NAMES.indexOf( k ) : 9;
		const order = Object.keys( s ).sort( ( x, y ) => rank( x ) - rank( y ) || x.localeCompare( y ) );
		lines.push( `showing        ${order.map( ( k ) => `${k} ${s[ k ]}` ).join( '  ' ) || 'no LOD in this file'}${M.missing ? `   <span class="dim">${M.missing} waiting for a finer level</span>` : ''}` );
		lines.push( `<span class="dim">${M.adapter ? ( M.layers ? `materials: the bake's light x ${M.layers.count} texture layers of ${M.layers.size} px${M.layers.loaded ? ` (${MB( M.layers.bytes )}, beside the file)` : ' (loading)'}` : 'materials: the bake\'s light alone, no textures' ) : 'materials: as in the glTF file'}${DIST !== 1 ? '   LOD distances x' + DIST : ''}${REFINE !== 'direct' ? '   level order: ' + ( REFINE === 'hop' ? 'one step off the coarsest level first' : 'every level on the way' ) + ' (experiment, not in the branch)' : ''}   decoding: ${M.draco ? 'Draco, in DRACOLoader\'s workers' : WORKERS ? 'meshopt, ' + WORKERS + ' worker' + ( WORKERS === 1 ? '' : 's' ) : 'meshopt, main thread'}</span>` );
		lines.push( `<span class="dim">${( renderer.info.render.triangles / 1e6 ).toFixed( 2 )} M triangles  ${renderer.info.memory.geometries} geometries  ${fps.toFixed( 0 )} fps  camera ${camera.position.toArray().map( ( v ) => v.toFixed( 0 ) ).join( ' ' )}\ntimes from loader.load(), ${MS( M.tStart )} after the page started</span>` );

	}

	const message = M.messages[ M.messages.length - 1 ];
	if ( message ) lines.push( `<span class="bad">${message.replace( /</g, '&lt;' )}</span>` );
	hud.innerHTML = lines.join( '\n' );
	publish();

}

function link( label, changes, on ) {

	const q = new URLSearchParams( location.search );
	for ( const k in changes ) { if ( changes[ k ] === null ) q.delete( k ); else q.set( k, changes[ k ] ); }
	const a = document.createElement( 'a' );
	a.textContent = label; a.href = '?' + q.toString(); if ( on ) a.className = 'on';
	return a;

}

function span( label ) {

	const e = document.createElement( 'span' );
	e.textContent = label;
	return e;

}

function button( label, action, on ) {

	const b = document.createElement( 'button' );
	b.textContent = label; b.onclick = action; if ( on ) b.className = 'on';
	return b;

}

function buildBar() {

	bar.textContent = '';
	bar.append( link( 'tier A', { tier: 'a' }, TIER === 'a' ), link( 'tier B', { tier: 'b' }, TIER === 'b' ) );
	if ( ! params.get( 'file' ) ) bar.append( link( 'portable .glb', { src: null }, SRC === 'portable' ), link( 'exact .glb (adapter)', { src: 'glb' }, SRC === 'glb' ), link( 'exact .bin per level (adapter)', { src: 'bins' }, SRC === 'bins' ) );
	for ( const name in VIEWS ) bar.append( button( name, () => setCamera( ...VIEWS[ name ] ) ) );
	bar.append( button( 'colour levels', () => setTint( ! tinted ), tinted ) );
	if ( TIER === 'b' && LAZY && ! LIB ) bar.append( span( 'levels:' ), link( 'wanted only', { refine: null }, REFINE === 'direct' ), link( 'one step first', { refine: 'hop' }, REFINE === 'hop' ), link( 'every step', { refine: 'step' }, REFINE === 'step' ) );
	if ( TIER === 'b' ) bar.append( button( sweeping ? 'loading all ...' : 'load all levels', () => M.loadAll(), sweeping ) );

}

buildBar();
requestAnimationFrame( frame );
load().catch( ( e ) => {

	const report = () => { M.error = String( e && e.message || e ); console.error( e ); publish(); };
	if ( ABORTED.test( String( e && e.message || e ) ) ) unlessLeaving( report ); else report();

} );
