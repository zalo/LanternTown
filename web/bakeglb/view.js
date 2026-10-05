// The portable glTF files of tools/bake2glb.mjs (--profile dcc) in three.js with nothing of this project's own:
// WebGPURenderer, the stock GLTFLoader and DRACOLoader, the materials the loader makes. No shader of the town's page.
//   ?file=NAME.glb        a file of out/web/_nextglb/ (default portable_lod_spawn.glb)
//   ?loader=stock|a|b     stock (default): three/addons GLTFLoader, which shows an MSFT_lod file's finest level;
//                         a, b: the MSFT_lod loaders of web/gltflod/lib (read only here): a = levels into THREE.LOD,
//                         b = a + lazy levels, a .glb read by HTTP Range (&lazy=0, &range=0 turn those off)
//   ?cam=x,y,z,tx,ty,tz   camera and what it looks at (default: the bake's spawn point)   ?fov=62
//   ?tone=aces|neutral|none (default aces)   ?exp=0.72   ?bg=rrggbb   ?w=&h= (a fixed canvas, for pictures)   ?pr=
//   ?sky=mesh|background|none   the file's sky: mesh (default) = its node `sky` as any viewer shows it, a shell around the
//                         town; background = the same image as the scene's background, infinitely far (what
//                         scenes[0].extras.baked.sky is for), the node hidden; none = the node hidden, ?bg behind
//   ?shot=1               a settled picture as a PNG data URL in window.__view.shot
// window.__view: { done, error, ms: { fetch+parse (loaded), first frame }, bytes, requests, calls, triangles, ... }
// window.__stats: the same numbers without the picture, kept up to date for a visitor's report (engine/telemetry.js)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { texture, equirectUV, positionWorldDirection, vec2, vec3 } from 'three/tsl';

const Q = new URLSearchParams( location.search );
const FILE = /^[\w./-]+\.glb$/.test( Q.get( 'file' ) || '' ) && ! Q.get( 'file' ).includes( '..' ) ? Q.get( 'file' ) : 'portable_lod_spawn.glb';
const BASE = /^[\w./-]+$/.test( Q.get( 'base' ) || '' ) ? Q.get( 'base' ) : '../out/web/_nextglb/';
const LOADER = [ 'a', 'b' ].includes( Q.get( 'loader' ) ) ? Q.get( 'loader' ) : 'stock';
const SKY = [ 'background', 'none' ].includes( Q.get( 'sky' ) ) ? Q.get( 'sky' ) : 'mesh';
const num = ( k, d ) => ( isFinite( parseFloat( Q.get( k ) ) ) ? parseFloat( Q.get( k ) ) : d );
const V = window.__view = { done: false, error: null, file: FILE, loader: LOADER, ms: {}, requests: [] };
const hud = document.getElementById( 'hud' ), canvas = document.getElementById( 'gl' );
const t0 = performance.now();
// the bytes that came for the model. The ranged loader reads the file's front as one open-ended request and cancels it
// when it has what it asked for: that request counts as far as it was read (source.byteLength), not as its Content-Length
let source = null;
const wire = () => ( source ? source.byteLength + V.requests.reduce( ( s, r ) => s + ( r.status === 206 && /^bytes=[1-9]\d*-\d+$/.test( r.range || '' ) ? r.bytes : 0 ), 0 ) : V.requests.reduce( ( s, r ) => s + r.bytes, 0 ) );
const S = window.__stats = { page: 'bakeglb-view', file: FILE, loader: LOADER };
const publish = () => {

	Object.assign( S, { lazy: V.lazy, ranged: V.ranged, bytes: V.bytes, requests: V.requests.length, fetch_ms: V.ms.fetched, parse_ms: V.ms.fetched === undefined || V.ms.loaded === undefined ? undefined : V.ms.loaded - V.ms.fetched,
		loaded_ms: V.ms.loaded, loaded_bytes: V.ready && V.ready.bytes, loaded_requests: V.ready && V.ready.requests, gpu_upload_ms: V.ms.first === undefined ? undefined : V.ms.first - V.ms.loaded, first_frame_ms: V.ms.first, first_frame_bytes: V.first && V.first.bytes, first_frame_requests: V.first && V.first.requests, triangles: V.triangles, draw_calls: V.calls, fps: V.fps === undefined ? undefined : Math.round( V.fps ), render_cpu_ms: V.cpu === undefined ? undefined : + V.cpu.toFixed( 2 ),
		meshes: V.meshes, materials: V.materials, lods: V.lods, sky: V.sky, textures: V.mem && V.mem.textures, geometries: V.mem && V.mem.geometries, tone: V.tone, backend: V.backend, extensions: V.extensions, done: V.done, error: V.error ? V.error.slice( 0, 300 ) : undefined } );

};

// what goes over the wire for the model, request by request (Range and all)
const fetch0 = window.fetch.bind( window );
window.fetch = async ( input, init ) => {

	const url = typeof input === 'string' ? input : input.url;
	const res = await fetch0( input, init );
	if ( /\.(glb|bin)(\?|$)/.test( url ) ) {

		const h = new Headers( ( init && init.headers ) || ( typeof input !== 'string' ? input.headers : undefined ) );
		V.requests.push( { range: h.get( 'Range' ), status: res.status, bytes: Number( res.headers.get( 'Content-Length' ) ) || 0, at: Math.round( performance.now() - t0 ) } );

	}

	return res;

};

try {

	const renderer = new THREE.WebGPURenderer( { canvas, antialias: true, forceWebGL: Q.get( 'webgl' ) === '1' } );
	await renderer.init();
	const fixed = Q.get( 'w' ) && Q.get( 'h' );
	const size = () => { const w = fixed ? num( 'w', 1280 ) : window.innerWidth, h = fixed ? num( 'h', 720 ) : window.innerHeight; renderer.setPixelRatio( num( 'pr', 1 ) ); renderer.setSize( w, h, ! fixed ); camera.aspect = w / h; camera.updateProjectionMatrix(); };
	const tone = Q.get( 'tone' ) || 'aces';
	V.tone = tone; V.backend = renderer.backend.isWebGPUBackend ? 'webgpu' : 'webgl';
	renderer.toneMapping = tone === 'none' ? THREE.NoToneMapping : tone === 'neutral' ? THREE.NeutralToneMapping : THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = num( 'exp', 0.72 );
	const scene = new THREE.Scene();
	scene.background = new THREE.Color( parseInt( /^[0-9a-f]{6}$/i.test( Q.get( 'bg' ) || '' ) ? Q.get( 'bg' ) : '070a14', 16 ) );
	const camera = new THREE.PerspectiveCamera( num( 'fov', 62 ), 16 / 9, 0.1, 4000 );
	if ( fixed ) { canvas.style.width = num( 'w', 1280 ) + 'px'; canvas.style.height = num( 'h', 720 ) + 'px'; }
	size();
	if ( ! fixed ) window.addEventListener( 'resize', size );
	const controls = new OrbitControls( camera, canvas );
	controls.enableDamping = true; controls.maxDistance = 1500; controls.listenToKeyEvents( window ); controls.keyPanSpeed = 30;

	const mod = LOADER === 'stock' ? await import( 'three/addons/loaders/GLTFLoader.js' ) : await import( LOADER === 'a' ? '../gltflod/lib/GLTFLoader.tierA.js' : '../gltflod/lib/GLTFLoader.tierB.js' );
	const loader = new mod.GLTFLoader();
	loader.setDRACOLoader( new DRACOLoader().setDecoderPath( '../node_modules/three/examples/jsm/libs/draco/gltf/' ) );
	const ranged = LOADER === 'b' && Q.get( 'range' ) !== '0', lazy = LOADER === 'b' && Q.get( 'lazy' ) !== '0';
	if ( lazy ) loader.setLazyLOD( true );
	if ( ranged ) loader.setRangeRequests( true );
	V.lazy = lazy; V.ranged = ranged;
	const gltf = await loader.loadAsync( BASE + FILE, ( e ) => { if ( e.total && e.loaded >= e.total ) V.ms.fetched = Math.round( performance.now() - t0 ); if ( e.total ) hud.textContent = `loading ${FILE}: ${( e.loaded / 1048576 ).toFixed( 1 )} of ${( e.total / 1048576 ).toFixed( 1 )} MB`; } );
	V.ms.loaded = Math.round( performance.now() - t0 );
	source = ranged && gltf.parser.extensions.KHR_binary_glTF && gltf.parser.extensions.KHR_binary_glTF.byteLength !== undefined && ! gltf.parser.extensions.KHR_binary_glTF.complete ? gltf.parser.extensions.KHR_binary_glTF : null;
	V.ready = { bytes: wire(), requests: V.requests.length }; // (what had to come before anything could be drawn)
	scene.add( gltf.scene );
	gltf.scene.updateMatrixWorld( true );
	const lods = []; let meshes = 0, mats = new Set();
	gltf.scene.traverse( ( o ) => { if ( o.isLOD ) lods.push( o ); if ( o.isMesh ) { meshes ++; mats.add( o.material ); } } );
	V.lods = lods.length; V.meshes = meshes; V.materials = mats.size;
	V.extensions = gltf.parser.json.extensionsUsed; V.required = gltf.parser.json.extensionsRequired;
	V.bytes = wire(); publish();

	// the sky: the file's shell as it is, or its image behind everything. The image is three.js' equirectangular layout
	// with +v down (a glTF texture is not flipped), about +y by `yaw`: the direction turned back, v the other way up
	const skyInfo = ( gltf.parser.json.scenes[ 0 ].extras || {} ).baked && gltf.parser.json.scenes[ 0 ].extras.baked.sky;
	let skyNode = null;
	gltf.scene.traverse( ( o ) => { if ( o.userData && o.userData.sky === true ) skyNode = o; } );
	V.sky = skyNode ? SKY : 'absent';
	if ( skyNode && SKY !== 'mesh' ) {

		skyNode.visible = false;
		if ( SKY === 'background' && skyInfo && skyInfo.projection === 'equirect' ) {

			const c = Math.cos( skyInfo.yaw || 0 ), s = Math.sin( skyInfo.yaw || 0 ), d = positionWorldDirection;
			const uv = equirectUV( vec3( d.x.mul( c ).sub( d.z.mul( s ) ), d.y, d.x.mul( s ).add( d.z.mul( c ) ) ) );
			scene.backgroundNode = texture( skyNode.material.map, vec2( uv.x, uv.y.oneMinus() ) ).level( 0 );

		}

	}

	// the camera: ?cam, or the bake's spawn point looking north along the street
	const ex = ( gltf.parser.json.scenes[ 0 ].extras || {} ), sp = ex.baked && ex.baked.spawn && ex.baked.spawn.pos;
	const cam = ( Q.get( 'cam' ) || '' ).split( ',' ).map( Number );
	if ( cam.length === 6 && cam.every( isFinite ) ) { camera.position.set( cam[ 0 ], cam[ 1 ], cam[ 2 ] ); controls.target.set( cam[ 3 ], cam[ 4 ], cam[ 5 ] ); }
	else if ( sp ) { camera.position.set( sp[ 0 ], sp[ 2 ], - sp[ 1 ] ); controls.target.set( sp[ 0 ], sp[ 2 ] - 0.1, - sp[ 1 ] - 6 ); }
	else { const b = new THREE.Box3().setFromObject( gltf.scene ), c = b.getCenter( new THREE.Vector3() ); camera.position.set( c.x, b.max.y + 20, b.max.z + 30 ); controls.target.copy( c ); }
	controls.update();

	let calls = 0, tris = 0, frames = 0, last = performance.now(), fps = 0, cpu = 0;
	const show = () => {

		hud.textContent = `${FILE}  ${LOADER === 'stock' ? 'stock GLTFLoader' : 'MSFT_lod loader, tier ' + LOADER.toUpperCase() + ( lazy ? ', lazy' : '' ) + ( ranged ? ', by range' : '' )}\n`
			+ `${( wire() / 1048576 ).toFixed( 2 )} MB in ${V.requests.length} request${V.requests.length === 1 ? '' : 's'}; loaded ${V.ms.loaded} ms, first frame ${V.ms.first ?? '...'} ms\n`
			+ `${calls} draw calls, ${( tris / 1e6 ).toFixed( 2 )} M triangles, ${meshes} meshes, ${mats.size} materials${lods.length ? ', ' + lods.length + ' LODs' : ''}, ${fps.toFixed( 0 )} fps\n`
			+ `camera ${camera.position.toArray().map( ( v ) => v.toFixed( 1 ) ).join( ',' )},${controls.target.toArray().map( ( v ) => v.toFixed( 1 ) ).join( ',' )}`;

	};

	await renderer.compileAsync( scene, camera );
	renderer.setAnimationLoop( () => {

		controls.update();
		const r0 = performance.now();
		renderer.render( scene, camera );
		cpu += performance.now() - r0;
		calls = renderer.info.render.drawCalls; tris = renderer.info.render.triangles;
		frames ++;
		if ( V.ms.first === undefined ) { V.ms.first = Math.round( performance.now() - t0 ); V.first = { bytes: wire(), requests: V.requests.length }; }
		const now = performance.now();
		if ( now - last > 500 ) { fps = frames * 1000 / ( now - last ); V.cpu = cpu / frames; cpu = 0; frames = 0; last = now; V.calls = calls; V.triangles = tris; V.fps = fps; V.bytes = wire(); V.mem = renderer.info.memory; show(); publish(); }

	} );
	// settled: a few frames on, and nothing loading any more (lazy levels arrive by themselves)
	await new Promise( ( k ) => setTimeout( k, num( 'settle', 2500 ) ) );
	V.calls = calls; V.triangles = tris; V.bytes = wire();
	if ( Q.get( 'shot' ) === '1' ) { renderer.render( scene, camera ); V.shot = canvas.toDataURL( 'image/png' ); }
	V.done = true; publish();

	const bar = document.getElementById( 'bar' );
	const link = ( text, set ) => { const q = new URLSearchParams( location.search ); for ( const k in set ) { if ( set[ k ] === null ) q.delete( k ); else q.set( k, set[ k ] ); } const a = document.createElement( 'a' ); a.textContent = text; a.href = '?' + q; if ( Object.keys( set ).every( ( k ) => ( Q.get( k ) || null ) === set[ k ] ) ) a.className = 'on'; return a; };
	bar.append( link( 'stock GLTFLoader', { loader: null } ), link( 'MSFT_lod tier A', { loader: 'a' } ), link( 'MSFT_lod tier B (lazy, ranged)', { loader: 'b' } ),
		link( 'ACES', { tone: null } ), link( 'no tone mapping', { tone: 'none' } ), link( 'sky: mesh', { sky: null } ), link( 'sky: background', { sky: 'background' } ), link( 'no sky', { sky: 'none' } ), Object.assign( document.createElement( 'a' ), { textContent: 'download ' + FILE, href: BASE + FILE, download: FILE } ) );

} catch ( e ) {

	V.error = String( ( e && e.stack ) || e ); V.done = true; publish();
	hud.textContent = 'failed: ' + V.error;

}
