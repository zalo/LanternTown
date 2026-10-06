// The town from ONE GLB and nothing else: three's stock GLTFLoader (+ DRACOLoader), stock materials, no town engine -
// no halo sprites, no bloom, no levels of detail, no streaming. What is drawn is what the file holds, the sky too
// (a mesh node `sky` around the town with the panorama on it).
// (tools/bake2glb.mjs --profile dcc: KHR_materials_unlit + COLOR_0 with the baked light, Draco, WebP).
// Only walking is added: the walkable regions and the colliders of out/town/town.json, as on the live page.
//
//   ?file=town_best.glb     any GLB of out/web/_nextglb/ (town_best.glb: every cell at its finest level; town_near.glb: the
//                           level below it, 3.1 M triangles; town_mid.glb: the middle level, 1.2 M; town_lo.glb: the lowest level, for a small device)
//   ?sky=mesh|background|none   the file's sky as its mesh (default), as a background drawn from the same picture by view
//                           direction (scenes[0].extras.baked.sky says how it is laid out; it does not slide as one walks), or hidden
//   ?bg=ff00ff              background colour (with ?sky=none: holes in the town show in it)
//   ?tone=none|neutral      tone mapping (default ACES at 0.72, as the live page)   ?exp=0.72
//   ?pr=1.5  ?pixels=2e6    pixel ratio cap and the cap on drawn pixels (touch devices: 2 M)
//   ?webgl=1                the WebGL 2 backend
//   ?walk=bvh               (a prototype, off unless asked for) walk by triangles instead of town.json's regions and colliders:
//                           a capsule against a three-mesh-bvh MeshBVH, see bvhwalk.js. town.json is then only asked for
//                           its spawn, and may be missing ( ?spawn=drop: as if it were ). What is collided with:
//                             the file's own solid triangles; over ?budget=1000000 of them (0: no limit) only those
//                             at the heights ?band=-3,6 around the spawn's ground ( ?band=all: never ) - which for the
//                             8.5 M triangles of town_best.glb is still 5.6 M, 200 MB and 2 s: give that one a coarse level,
//                             ?collide=town_lo.glb (another file of the folder, loaded and not drawn);
//                             a file with MSFT_lod levels collides with the last level of each node ( ?collide=self: not that )
//                           ?bvhshow=1  draws the collision triangles as a wireframe
// window.__stats: file, bytes, timings, triangles, draw calls, fps (a visitor's report carries it: engine/telemetry.js)

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { texture, equirectUV, positionWorldDirection, vec2, vec3 } from 'three/tsl';
import { Nav } from './nav.js';

const Q = new URLSearchParams( location.search );
const $ = ( id ) => document.getElementById( id );
const FILES = { 'town_best.glb': 'Finest level (52 MB)', 'town_near.glb': 'Near level (31 MB)', 'town_mid.glb': 'Middle level (12 MB)', 'town_lo.glb': 'Lowest level (6 MB)' };
const FILE = /^[\w.-]+\.glb$/.test( Q.get( 'file' ) || '' ) ? Q.get( 'file' ) : 'town_best.glb';
const BASE = '../out/web/_nextglb/';
const BVH = Q.get( 'walk' ) === 'bvh';
const num = ( k, d ) => ( isFinite( parseFloat( Q.get( k ) ) ) ? parseFloat( Q.get( k ) ) : d );
const touch = matchMedia( '(pointer: coarse)' ).matches || 'ontouchstart' in window;
if ( touch ) document.body.classList.add( 'touch' );
const canvas = $( 'gl' ), hud = $( 'hud' );
const S = window.__stats = { page: 'glbtown', file: FILE, loader: 'stock', done: false, error: null };
const t0 = performance.now();

for ( const f of Object.keys( FILES ).concat( FILES[ FILE ] ? [] : [ FILE ] ) ) $( 'file' ).append( Object.assign( document.createElement( 'option' ), { value: f, textContent: FILES[ f ] || f, selected: f === FILE } ) );
$( 'file' ).onchange = ( e ) => { const u = new URL( location.href ); u.searchParams.set( 'file', e.target.value ); location.href = u.href; };

try {

	const renderer = new THREE.WebGPURenderer( { canvas, antialias: true, forceWebGL: Q.get( 'webgl' ) === '1' } );
	await renderer.init();
	S.backend = renderer.backend.isWebGPUBackend ? 'webgpu' : 'webgl';
	const tone = Q.get( 'tone' ) || 'aces';
	renderer.toneMapping = tone === 'none' ? THREE.NoToneMapping : tone === 'neutral' ? THREE.NeutralToneMapping : THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = num( 'exp', 0.72 );
	const scene = new THREE.Scene();
	scene.background = new THREE.Color( /^[0-9a-f]{6}$/i.test( Q.get( 'bg' ) || '' ) ? parseInt( Q.get( 'bg' ), 16 ) : 0x070a14 );
	const camera = new THREE.PerspectiveCamera( 62, 16 / 9, 0.08, 1500 );
	const size = () => {

		const w = innerWidth, h = innerHeight;
		const pr = Math.min( num( 'pr', 1 ), devicePixelRatio, Math.sqrt( num( 'pixels', touch ? 2e6 : 8e6 ) / ( w * h ) ) );
		renderer.setPixelRatio( pr ); renderer.setSize( w, h, false );
		camera.aspect = w / h; camera.updateProjectionMatrix();
		S.canvas = [ Math.round( w * pr ), Math.round( h * pr ) ];

	};

	size();
	addEventListener( 'resize', size );

	// ---------------------------------------------------------------------------------- the file, and where one may walk
	const loader = new GLTFLoader();
	loader.setDRACOLoader( new DRACOLoader().setDecoderPath( '../node_modules/three/examples/jsm/libs/draco/gltf/' ) );
	const stamp = await fetch( BASE + FILE, { method: 'HEAD', cache: 'no-store' } ).then( ( r ) => ( r.ok ? ( r.headers.get( 'Last-Modified' ) || '' ) + ( r.headers.get( 'Content-Length' ) || '' ) : '' ) ).catch( () => '' );
	const url = BASE + FILE + ( stamp ? '?v=' + encodeURIComponent( stamp.replace( /\W+/g, '' ) ) : '' ); // (the file changes under its name: never an old copy)
	const [ gltf, town ] = await Promise.all( [
		loader.loadAsync( url, ( e ) => {

			if ( ! e.total ) return;
			S.bytes = e.total;
			if ( e.loaded >= e.total && ! S.fetch_ms ) S.fetch_ms = Math.round( performance.now() - t0 );
			$( 'bar' ).firstElementChild.style.width = Math.round( e.loaded / e.total * 100 ) + '%';
			$( 'msg' ).textContent = `${FILE}: ${( e.loaded / 1048576 ).toFixed( 1 )} of ${( e.total / 1048576 ).toFixed( 1 )} MB` + ( e.loaded >= e.total ? ' · decoding…' : '' );

		} ),
		fetch( '../out/town/town.json' ).then( ( r ) => r.json() ).catch( ( e ) => { if ( BVH ) return null; throw e; } ), // (?walk=bvh needs none)
	] );
	S.loaded_ms = Math.round( performance.now() - t0 );
	S.parse_ms = S.loaded_ms - ( S.fetch_ms || 0 );
	scene.add( gltf.scene );
	gltf.scene.updateMatrixWorld( true );
	let meshes = 0; const mats = new Set();
	gltf.scene.traverse( ( o ) => { if ( o.isMesh ) { meshes ++; mats.add( o.material ); o.matrixAutoUpdate = false; } } );
	S.meshes = meshes; S.materials = mats.size; S.extensions = gltf.parser.json.extensionsUsed;
	// ?vc=0: the file without its baked light - the vertex colours (COLOR_0) ignored, the albedo textures alone, unlit
	// (?lit=1: lit instead by a plain sky light and a sun, the unlit materials swapped for standard ones, to see the shapes)
	if ( Q.get( 'vc' ) === '0' ) {

		const lit = Q.get( 'lit' ) === '1', swap = new Map();
		gltf.scene.traverse( ( o ) => {

			if ( ! o.isMesh || o.userData.sky === true ) return; // (the sky keeps its picture)
			const m = o.material;
			if ( lit ) {

				if ( ! swap.has( m ) ) swap.set( m, new THREE.MeshStandardMaterial( { map: m.map, color: m.color, transparent: m.transparent, opacity: m.opacity, alphaTest: m.alphaTest, side: m.side, depthWrite: m.depthWrite, roughness: 0.85, metalness: 0, name: m.name } ) );
				o.material = swap.get( m );

			} else { m.vertexColors = false; m.needsUpdate = true; }

		} );
		if ( lit ) {

			scene.add( new THREE.HemisphereLight( 0xbfd4ff, 0x3a2a1a, num( 'skylight', 1.2 ) ) );
			const sun = new THREE.DirectionalLight( 0xfff1d6, num( 'sun', 2.2 ) );
			sun.position.set( 40, 30, 60 );
			scene.add( sun );

		}

		S.vc = lit ? 'off, lit' : 'off';

	}

	// the sky: a node of the file marked extras.sky. As a background, the same picture is read by view direction
	// (GLTFLoader textures have flipY off, hence the node and not scene.background = texture)
	let sky = null;
	gltf.scene.traverse( ( o ) => { if ( o.userData.sky === true ) sky = o; } );
	const skyInfo = ( ( ( gltf.parser.json.scenes || [] )[ 0 ] || {} ).extras || {} ).baked?.sky, skyAs = Q.get( 'sky' ) || 'mesh';
	S.sky = ! sky ? 'none in the file' : skyAs === 'none' ? 'none' : skyAs === 'background' && skyInfo && sky.material.map ? 'background' : 'mesh';
	if ( sky && S.sky !== 'mesh' ) sky.visible = false;
	if ( S.sky === 'background' ) {

		const c = Math.cos( skyInfo.yaw || 0 ), s = Math.sin( skyInfo.yaw || 0 ), d = positionWorldDirection;
		const uv = equirectUV( vec3( d.x.mul( c ).sub( d.z.mul( s ) ), d.y, d.x.mul( s ).add( d.z.mul( c ) ) ) );
		scene.backgroundNode = texture( sky.material.map, vec2( uv.x, uv.y.oneMinus() ) ).level( 0 );

	}

	const nav = town ? new Nav( town ) : null;

	// ---------------------------------------------------------------------------------- player (as on the live page)
	const spawn = BVH && ( ! town || Q.get( 'spawn' ) === 'drop' ) ? null : town.spawn;
	const P = { x: spawn ? spawn.pos[ 0 ] : 0, y: spawn ? spawn.pos[ 1 ] : 0, z: 0, yaw: THREE.MathUtils.degToRad( ( spawn && spawn.yaw ) || 0 ), pitch: 0, eye: 1.65, bob: 0 };
	if ( ! BVH ) P.z = nav.groundAt( P.x, P.y, 50 ) ?? 0;
	// ?walk=bvh: the triangles one may bump into, and the capsule that does (bvhwalk.js; loaded only when asked for)
	let walker = null;
	if ( BVH ) {

		$( 'msg' ).textContent = `${FILE}: what one can bump into…`;
		const { collect, Walker } = await import( './bvhwalk.js' );
		const t1 = performance.now(), json = gltf.parser.json, top = ( json.scenes[ json.scene || 0 ] || {} ).nodes || [];
		const lod = ( i ) => json.nodes[ i ].extensions?.MSFT_lod?.ids;
		const other = /^[\w.-]+\.glb$/.test( Q.get( 'collide' ) || '' ) ? Q.get( 'collide' ) : null;
		let from = gltf.scene, what = FILE;
		if ( other ) {

			// a coarse file of the same town: loaded, never drawn
			from = ( await loader.loadAsync( BASE + other ) ).scene; what = other;

		} else if ( Q.get( 'collide' ) !== 'self' && top.some( lod ) ) {

			// MSFT_lod: the stock loader builds only the first level of a node; its last is asked of the parser here
			from = await Promise.all( top.map( ( i ) => gltf.parser.getDependency( 'node', lod( i ) ? lod( i )[ lod( i ).length - 1 ] : i ) ) );
			what = FILE + ', last MSFT_lod level';

		}

		// the band is counted from the ground at the spawn; with no spawn there is no height to count from
		const band = ( Q.get( 'band' ) || '-3,6' ).split( ',' ).map( parseFloat ), foot = spawn ? ( spawn.pos[ 2 ] ?? P.eye ) - P.eye : null;
		const banded = foot !== null && band.length === 2 && band.every( isFinite );
		const { geometry, stats } = collect( from, {
			budget: num( 'budget', 1e6 ), band: ! banded ? false : Q.get( 'band' ) ? true : 'auto',
			lo: banded ? foot + band[ 0 ] : undefined, hi: banded ? foot + band[ 1 ] : undefined,
		} );
		if ( other ) from.traverse( ( o ) => { if ( o.isMesh ) { o.geometry.dispose(); o.material.map?.dispose(); } } );
		walker = new Walker( geometry, stats );
		walker.place( P, spawn );
		S.bvh = Object.assign( stats, { from: what, source_ms: Math.round( performance.now() - t1 ) - stats.collect_ms - stats.build_ms } );
		if ( Q.get( 'bvhshow' ) === '1' ) scene.add( walker.helper() );
		if ( stats.over_budget ) console.warn( `walk=bvh: ${stats.triangles} triangles to collide with, ${( stats.bytes / 1048576 ).toFixed( 0 )} MB: give a coarse level, ?collide=town_lo.glb` );

	}

	let eyeZ = P.z;
	const keys = new Set(), stick = { x: 0, y: 0 };
	const placeCamera = () => {

		// town coordinates are X east, Y north, Z up; three is ( x, z, -y )
		camera.position.set( P.x, eyeZ + P.eye + Math.sin( P.bob * 2 ) * 0.018, - P.y );
		camera.rotation.set( P.pitch, P.yaw, 0, 'YXZ' );

	};

	const walk = ( dt ) => {

		let f = 0, s = 0;
		if ( keys.has( 'KeyW' ) || keys.has( 'ArrowUp' ) ) f += 1;
		if ( keys.has( 'KeyS' ) || keys.has( 'ArrowDown' ) ) f -= 1;
		if ( keys.has( 'KeyD' ) ) s += 1;
		if ( keys.has( 'KeyA' ) ) s -= 1;
		if ( keys.has( 'ArrowLeft' ) ) P.yaw += 1.8 * dt;
		if ( keys.has( 'ArrowRight' ) ) P.yaw -= 1.8 * dt;
		f += - stick.y; s += stick.x;
		const m = Math.hypot( f, s );
		if ( m > 0.05 ) {

			const sp = ( keys.has( 'ShiftLeft' ) || keys.has( 'ShiftRight' ) || m > 1.25 ? 4.5 : 2.2 ) * Math.min( 1, m ) * dt;
			f /= m; s /= m;
			// yaw 0 looks north (+Y); right is +X
			const dx = ( - Math.sin( P.yaw ) * f + Math.cos( P.yaw ) * s ) * sp, dy = ( Math.cos( P.yaw ) * f + Math.sin( P.yaw ) * s ) * sp;
			const ox = P.x, oy = P.y;
			if ( walker ) walker.step( P, dx, dy, dt ); else nav.move( P, dx, dy );
			P.bob += Math.hypot( P.x - ox, P.y - oy ) * 2.1;

		} else if ( walker ) walker.step( P, 0, 0, dt ); // (standing still, one still falls)

		eyeZ += ( P.z - eyeZ ) * Math.min( 1, dt * 10 ); // steps are eased, not jumped

	};

	const look = ( dx, dy, k ) => { P.yaw -= dx * k; P.pitch = THREE.MathUtils.clamp( P.pitch - dy * k, - 1.35, 1.35 ); };
	addEventListener( 'keydown', ( e ) => { keys.add( e.code ); if ( e.code === 'KeyH' ) hud.style.display = hud.style.display === 'none' ? '' : 'none'; } );
	addEventListener( 'keyup', ( e ) => keys.delete( e.code ) );
	addEventListener( 'blur', () => keys.clear() );
	canvas.addEventListener( 'click', () => { if ( ! touch ) canvas.requestPointerLock?.(); } );
	addEventListener( 'mousemove', ( e ) => { if ( document.pointerLockElement === canvas ) look( e.movementX, e.movementY, 0.0022 ); } );

	// touch: left stick walks, dragging anywhere else looks
	const st = $( 'stick' ), knob = st.firstElementChild;
	let stickId = null, lookId = null, lx = 0, ly = 0;
	st.addEventListener( 'pointerdown', ( e ) => { stickId = e.pointerId; st.setPointerCapture( e.pointerId ); e.preventDefault(); } );
	st.addEventListener( 'pointermove', ( e ) => {

		if ( e.pointerId !== stickId ) return;
		const r = st.getBoundingClientRect();
		let x = ( e.clientX - r.left - r.width / 2 ) / ( r.width / 2 ), y = ( e.clientY - r.top - r.height / 2 ) / ( r.height / 2 );
		const m = Math.hypot( x, y );
		if ( m > 1 ) { x /= m; y /= m; }
		stick.x = x * 1.3; stick.y = y * 1.3;
		knob.style.transform = `translate(${x * 36}px, ${y * 36}px)`;

	} );
	const stickEnd = ( e ) => { if ( e.pointerId !== stickId ) return; stickId = null; stick.x = stick.y = 0; knob.style.transform = ''; };
	st.addEventListener( 'pointerup', stickEnd );
	st.addEventListener( 'pointercancel', stickEnd );
	canvas.addEventListener( 'pointerdown', ( e ) => { if ( e.pointerType === 'mouse' ) return; lookId = e.pointerId; lx = e.clientX; ly = e.clientY; } );
	canvas.addEventListener( 'pointermove', ( e ) => { if ( e.pointerId !== lookId ) return; look( e.clientX - lx, e.clientY - ly, 0.0045 ); lx = e.clientX; ly = e.clientY; } );
	const lookEnd = ( e ) => { if ( e.pointerId === lookId ) lookId = null; };
	canvas.addEventListener( 'pointerup', lookEnd );
	canvas.addEventListener( 'pointercancel', lookEnd );

	// ---------------------------------------------------------------------------------- frames
	placeCamera();
	$( 'msg' ).textContent = `${FILE}: on to the GPU…`;
	await renderer.compileAsync( scene, camera );
	let frames = 0, last = performance.now(), prev = last, worst = 0;
	renderer.setAnimationLoop( () => {

		const now = performance.now(), dt = Math.min( 0.1, ( now - prev ) / 1000 );
		if ( S.first_frame_ms !== undefined ) worst = Math.max( worst, now - prev );
		prev = now;
		walk( dt );
		placeCamera();
		renderer.render( scene, camera );
		frames ++;
		if ( S.first_frame_ms === undefined ) {

			S.first_frame_ms = S.first = Math.round( now - t0 ); // (first: the name the live page uses)
			$( 'veil' ).classList.add( 'hide' );
			window.__tele?.mark( 'first-frame' );

		}

		if ( now - last > 500 ) {

			S.fps = Math.round( frames * 1000 / ( now - last ) ); S.worst_ms = Math.round( worst );
			S.draw_calls = renderer.info.render.drawCalls; S.triangles = renderer.info.render.triangles;
			S.geometries = renderer.info.memory.geometries; S.textures = renderer.info.memory.textures;
			S.at = [ + P.x.toFixed( 1 ), + P.y.toFixed( 1 ) ];
			frames = 0; last = now; worst = 0;
			hud.textContent = `${S.fps} fps  ${S.draw_calls} draws  ${( S.triangles / 1e6 ).toFixed( 2 )} M tris\n`
				+ `${FILE}  ${( ( S.bytes || 0 ) / 1048576 ).toFixed( 1 )} MB  stock GLTFLoader · ${S.backend === 'webgpu' ? 'WebGPU' : 'WebGL 2'}\n`
				+ `download ${( ( S.fetch_ms || 0 ) / 1000 ).toFixed( 1 )} s  decode ${( S.parse_ms / 1000 ).toFixed( 1 )} s  first frame ${( S.first_frame_ms / 1000 ).toFixed( 1 )} s\n`
				+ `${meshes} meshes  ${mats.size} materials  ${S.canvas[ 0 ]}x${S.canvas[ 1 ]}`;
			if ( walker ) {

				// what the collision costs a frame (the whole of step(): every sub-step and its shapecasts)
				S.bvh.step_ms = + ( walker.ms / Math.max( 1, walker.calls ) ).toFixed( 3 ); S.bvh.step_worst_ms = + walker.worst.toFixed( 2 );
				S.bvh.fell_out = walker.fellOut; S.bvh.on_ground = walker.ground; S.at[ 2 ] = + P.z.toFixed( 2 );
				walker.ms = walker.calls = walker.worst = 0;
				hud.textContent += `\nwalk=bvh  ${( S.bvh.triangles / 1e3 ).toFixed( 0 )} k tris of ${S.bvh.from}  built ${S.bvh.collect_ms}+${S.bvh.build_ms} ms  ${( S.bvh.bytes / 1048576 ).toFixed( 1 )} MB  step ${S.bvh.step_ms} ms`;

			}

		}

	} );
	$( 'help' ).textContent = touch ? '' : 'click to look · WASD / arrows walk · Shift run · H stats';
	// for a check from outside: the player, and the picture (a WebGPU canvas only reads back in the task that drew it)
	window.__view = { P, nav, shot: () => { placeCamera(); renderer.render( scene, camera ); return canvas.toDataURL( 'image/jpeg', 0.9 ); } };
	if ( walker ) Object.assign( window.__view, { walker, scene, keys } ); // (keys: to hold a key from outside)
	S.done = true;

} catch ( e ) {

	S.error = String( ( e && e.stack ) || e ); S.done = true;
	$( 'msg' ).textContent = 'could not show ' + FILE + ': ' + ( ( e && e.message ) || e );
	console.error( e );

}
