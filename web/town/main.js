// Walkable lantern town square. Loads out/town/town.json (tools/town.py) through the shared engine.
//   ?quality=high|medium|low|phone   render tier, each a baked build (out/web/_next, tools/webbake.py; default: high, on every device; the selector remembers a choice)
//   ?quality=studio|studio-medium    the streaming engine (every lantern a light, screen-space reflections; about 4 GB of GPU memory):
//                                the picture tools/town.py render judges. Only ever from the URL / the selector, never remembered
//   ?shot=<name>                 start at a review viewpoint
//   ?baked=0                     the streaming engine with this tier's settings instead of the baked build
//   ?baked=stale                 the baked build even when town.json has been rebuilt since. That is what a visitor gets
//                                anyway (a bake a rebuild behind beats the streaming engine on a phone); review renders
//                                (?capture=1) and ?baked=fresh stream the town then, so a picture never shows an old town unasked
//   ?legacy=1                    the page as it was before Medium / High were baked: phone / low on the first baked runtime
//                                (engine/lite_v1.js, out/web/_bake, tools/webbake_v1.py), medium / high streaming.
//                                ?town=town2 always runs this way: it has no bake of the new kind yet
//   ?fmt=packed                  the baked build's geometry from its packed .bin files instead of the .glb files beside them
//                                (engine/lite_glb.js; node tools/bake2glb.mjs <bake dir> --live writes them, the bake runs it).
//                                ?fmt=glb is the default wherever a bake has them; __stats.fmt / __stats.geo say what was loaded
//   ?capture=1&token=..&w=..&h=..[&shots=a,b][&out=dir]   review renders: every shot -> out/renders/_town/[dir/]<shot>.png

import { createEngine, THREE, sourceRadius, lampFalloff, haloCore } from '../engine/engine.js';
import { Town, TIERS } from '../engine/town.js';
import * as LITE from '../engine/lite.js';

const BASE = '..';
const Q = new URLSearchParams( location.search );
const TOWN_ID = /^town\d+$/.test( Q.get( 'town' ) || '' ) ? Q.get( 'town' ) : 'town'; // ?town=town2: a second town assembly (out/town2/town.json, renders -> out/renders/_town2/)
const CAPTURE = Q.get( 'capture' ) === '1';
const PVS = Q.get( 'pvs' ) === 'bake' || Q.get( 'pvs' ) === 'check' || Q.get( 'pvs' ) === 'holes'; // (tools/pvs.py: work out what can be seen from where, see engine/pvs.js)
const touch = matchMedia( '(pointer: coarse)' ).matches;
// the baked tiers are the default (High, phones too); the streaming engine (4 GB of GPU memory) has names of its own and is opt-in through
// the selector / ?quality=studio: it is never read from the remembered choice, so it cannot become what a phone opens
const STUDIO = { studio: 'high', 'studio-medium': 'medium' }; // -> the streaming engine's tier
const quality = Object.hasOwn( STUDIO, Q.get( 'quality' ) || '' ) ? Q.get( 'quality' ) : [ Q.get( 'quality' ), localStorage.getItem( 'tsq.quality.v2' ), 'high' ].find( ( t ) => t && Object.hasOwn( TIERS, t ) );
// (what is neither is not kept: a page of before wrote the menu's "flare" entry here and into the address)
if ( Q.has( 'quality' ) && Q.get( 'quality' ) !== quality ) { const u = new URL( location.href ); u.searchParams.delete( 'quality' ); history.replaceState( null, '', u.href ); }
if ( ! Object.hasOwn( TIERS, localStorage.getItem( 'tsq.quality.v2' ) || quality ) ) localStorage.removeItem( 'tsq.quality.v2' );
const tierName = STUDIO[ quality ] || quality; // the row of TIERS (streaming) and of BAKED_TIERS (baked)
const T = TIERS[ tierName ];
const LIGHT = T.ssr === false; // low + phone
const LEGACY = Q.get( 'legacy' ) === '1' || TOWN_ID !== 'town'; // the page as it was: the first baked runtime for phone / low, medium / high streaming
let lite = LITE; // the baked runtime (LEGACY: engine/lite_v1.js, fetched in main())
const BAKED = ! STUDIO[ quality ] && ( LEGACY ? LIGHT : !! LITE.BAKED_TIERS[ tierName ] ) && Q.get( 'baked' ) !== '0'; // every tier runs a baked build (?baked=0: the streaming engine)
const $ = ( id ) => document.getElementById( id );
document.body.classList.toggle( 'touch', touch && ! CAPTURE );
document.body.classList.toggle( 'capture', CAPTURE );
$( 'quality' ).value = quality;

const tick = () => new Promise( ( r ) => {

	const c = new MessageChannel();
	c.port1.onmessage = () => r();
	c.port2.postMessage( 0 );

} );

const size = () => ( CAPTURE ? [ + ( Q.get( 'w' ) || 1600 ), + ( Q.get( 'h' ) || 900 ) ] : [ innerWidth, innerHeight ] );
let bake = null; // manifest of the baked build the light tiers run (null: none, the town streams asset by asset)
// (review renders are drawn at the tier's own pixel ratio - what a high-dpi screen shows - and reduced to the frame
// size: thin railings, bunting strings and roof tiles get the samples the live picture has; ?pr= overrides)
const pixelRatio = () => ( CAPTURE ? ( parseFloat( Q.get( 'pr' ) ) || ( bake ? bake.tier.capture : T.pixelRatio ) || 1 ) : bake ? lite.bakedPixelRatio( bake, innerWidth, innerHeight ) : Math.min( 1, devicePixelRatio || 1, T.pixelRatio ) );

async function main() {

	const t0 = performance.now();
	const [ w, h ] = size();
	const canvas = $( 'gl' );
	if ( BAKED && LEGACY ) lite = await import( '../engine/lite_v1.js' );
	const loadBake = lite.loadBake;
	if ( BAKED ) bake = await loadBake( BASE, tierName, TOWN_ID, { stale: Q.get( 'baked' ) === 'stale' || ( ! CAPTURE && Q.get( 'baked' ) !== 'fresh' ) } );
	if ( BAKED && loadBake.note ) console.warn( 'town: ' + loadBake.note );
	const eng = window.eng = bake ? await lite.createLiteEngine( { canvas, base: BASE, width: w, height: h, pixelRatio: pixelRatio(), fov: 62, forceWebGL: Q.get( 'webgl' ) === '1', gpuTime: Q.get( 'gputime' ) === '1', bake } ) : await createEngine( { canvas, base: BASE, width: w, height: h, pixelRatio: pixelRatio(), fov: 62, forceWebGL: Q.get( 'webgl' ) === '1',
		anisotropy: T.aniso || ( LIGHT ? 4 : 8 ), maxTextureSize: T.tex || 2048, reflections: T.ssr !== false && Q.get( 'ssr' ) !== '0', envProbe: T.probes !== false } );
	if ( Q.has( 'ssr' ) ) eng.P.ssr.value = parseFloat( Q.get( 'ssr' ) );
	if ( Q.has( 'env' ) ) eng.P.env.value = parseFloat( Q.get( 'env' ) );
	// look: ?moon= ?ambient= ?exposure= ?bloom= ?fog= ?lamp= override the defaults (for tuning from the URL)
	// a moonlit night: enough cool fill that roofs and shadow sides read navy with detail, lamps held under the shoulder
	// (the fill at street level is a third lower than the roofs get - engine roofFill - and the lanterns reach less far,
	// see lampFalloff below: pools of lamplight with cool, darker paving between them instead of one amber mid-level)
	const LOOK = { fog: 0.0075, ambient: 0.8, moon: 1.3, exposure: 0.72, bloom: 0.3, lamp: 0.7, clouds: 1.0, cover: 0.36, cloudGain: 2.4, skyGlow: 4, haze: 0.35, glare: 24, mist: 0, mistHeight: 5, fogGain: 1 };
	for ( const k in LOOK ) if ( Q.has( k ) && isFinite( parseFloat( Q.get( k ) ) ) ) LOOK[ k ] = parseFloat( Q.get( k ) );
	haloCore.value = LOOK.glare;
	eng.setRig( 'night', LOOK );
	// (no grade on top of the bake: the lanterns reach as far as real ones and their light overlaps over the whole
	// square, and a steeper curve to pull the pools apart - 1.5 under exposure 0.56 with the ceiling at 4, then 1.25
	// with 2.4 - was each time judged too harsh on an HDR phone, where what is over white is shown brighter still.
	// ?contrast= ?knee= to try one)
	eng.P.contrast.value = 1; eng.P.saturation.value = 1.03;
	eng.P.knee.value = 2; eng.P.kneeGlow.value = 2.8; // highlight roll-off: lit walls / glowing glass (scene-linear ceilings)
	eng.lib.u.windowGlow.value = 0.75;
	eng.lib.u.soft.value = 1; // weathering tuned for street distance (thin wear, crevice / rain-streak grime)
	// grade / tone overrides for tuning: ?tm=agx&contrast=1.05&saturation=1&vignette=0.3&lift=0.002
	if ( Q.get( 'tm' ) === 'agx' ) eng.renderer.toneMapping = THREE.AgXToneMapping;
	for ( const k of [ 'contrast', 'saturation', 'vignette', 'lift', 'knee', 'kneeGlow', 'tame', 'free', 'roofFill', 'ao' ] ) if ( Q.has( k ) && isFinite( parseFloat( Q.get( k ) ) ) ) eng.P[ k ].value = parseFloat( Q.get( k ) );
	eng.lib.u.wet.value = 0.68; // the town is seen just after rain
	eng.lib.u.grime.value = 0.8;
	if ( eng.lib.u.wetFlat ) eng.lib.u.wetFlat.value = 0.7; // the soaked paving mirrors lanterns and lit windows as streaks, not as specks on every cobble
	sourceRadius.value = Q.has( 'srcr' ) ? parseFloat( Q.get( 'srcr' ) ) : 0.6; // lanterns light as small area sources
	{

		// pools of light: a lantern's reach in metres (its light x 1 / ( 1 + ( d / reach )^2 ), 0 = pure inverse square)
		const reach = Q.has( 'reach' ) ? parseFloat( Q.get( 'reach' ) ) : 4.5;
		lampFalloff.value = reach > 0 ? 1 / ( reach * reach ) : 0;

	}

	// material look overrides for tuning: ?wet=0.3&grime=0.7&wear=1 (global uniforms shared by every material)
	for ( const k of [ 'wet', 'grime', 'wear', 'special', 'windowGlow', 'emissive', 'puddleRough', 'filmRough', 'ripple', 'wetFlat', 'poolCap', 'soft', 'folSky', 'folUnder', 'folWrap', 'folBack', 'folReach', 'folBounce', 'folAO', 'folGTAO' ] ) if ( Q.has( k ) && isFinite( parseFloat( Q.get( k ) ) ) && eng.lib.u[ k ] ) eng.lib.u[ k ].value = parseFloat( Q.get( k ) );
	if ( ! T.ao ) eng.setPost( true, { ao: false } );
	eng.camera.near = 0.08; eng.camera.far = 1500;

	const town = window.town = bake ? new lite.BakedTown( eng, bake, { wait: CAPTURE || PVS || Q.has( 'shot' ) ? 'all' : 'near' } ) : new Town( eng, { base: BASE, tier: tierName, lamp: LOOK.lamp, batch: Q.get( 'batch' ) !== '0', cap: Q.has( 'cap' ) ? parseFloat( Q.get( 'cap' ) ) : undefined } );
	if ( BAKED && loadBake.note ) town.warnings.push( loadBake.note ); // (the render tool prints these)
	if ( bake && bake.look2 && Q.get( 'look' ) === '2' ) town.setLook( 1 ); // (?look=2: the bake's second look, see the selector)
	await town.load( `${BASE}/out/${TOWN_ID}/town.json?t=${Date.now()}`, ( f, what ) => {

		$( 'bar' ).firstElementChild.style.width = Math.round( f * 100 ) + '%';
		$( 'msg' ).textContent = 'lighting the lamps… ' + ( what || '' );

	} );
	const data = town.data, nav = town.nav;
	if ( data.frame && isFinite( data.frame.water_z ) && Q.get( 'tide' ) !== '0' ) eng.lib.u.tideY.value = data.frame.water_z; // wet band on the canal walls
	const loadS = ( performance.now() - t0 ) / 1000;

	// ------------------------------------------------------------------ player
	const P = { x: data.spawn.pos[ 0 ], y: data.spawn.pos[ 1 ], z: 0, yaw: THREE.MathUtils.degToRad( data.spawn.yaw || 0 ), pitch: 0, eye: 1.65, bob: 0 };
	// ?at=x,y,z,yaw,pitch: where to stand instead of the spawn (what the selectors pass on when they reload the page)
	const AT = ( Q.get( 'at' ) || '' ).split( ',' ).map( parseFloat );
	const atOK = AT.length === 5 && AT.every( isFinite );
	if ( atOK ) { P.x = AT[ 0 ]; P.y = AT[ 1 ]; P.yaw = AT[ 3 ]; P.pitch = AT[ 4 ]; }
	P.z = atOK ? nav.groundAt( P.x, P.y, AT[ 2 ] + 0.05 ) ?? AT[ 2 ] : nav.groundAt( P.x, P.y, 50 ) ?? 0;
	let eyeZ = P.z;
	const keys = new Set();
	const stick = { x: 0, y: 0 };
	const SHIFT = new Set( ( Q.get( 'shift' ) || '' ).split( ',' ).filter( Boolean ) );
	let free = null; // {pos, target, fov}: a fixed viewpoint (tour / capture) overrides the walk camera until the player moves

	const placeCamera = () => {

		const cam = eng.camera;
		if ( free ) {

			cam.position.set( free.pos[ 0 ], free.pos[ 2 ], - free.pos[ 1 ] );
			cam.fov = free.fov || 60;
			// review renders only: a shot marked shift (or named in ?shift=a,b) is taken like a shift lens - the camera
			// stays level and the frame slides to the target, so a high viewpoint keeps its verticals upright
			const rise = free.target[ 2 ] - free.pos[ 2 ], run = Math.hypot( free.target[ 0 ] - free.pos[ 0 ], free.target[ 1 ] - free.pos[ 1 ] );
			if ( CAPTURE && run > 0.01 && ( free.shift || SHIFT.has( free.name ) ) ) {

				const [ fw, fh ] = size();
				cam.setViewOffset( fw, fh, 0, - ( rise / run ) / Math.tan( THREE.MathUtils.degToRad( cam.fov ) / 2 ) * fh / 2, fw, fh );
				cam.lookAt( free.target[ 0 ], free.pos[ 2 ], - free.target[ 1 ] );
				return;

			}

			if ( cam.view ) cam.clearViewOffset();
			cam.updateProjectionMatrix();
			cam.lookAt( free.target[ 0 ], free.target[ 2 ], - free.target[ 1 ] );
			return;

		}

		if ( cam.fov !== 62 ) {

			cam.fov = 62; cam.updateProjectionMatrix();

		}

		cam.position.set( P.x, eyeZ + P.eye + Math.sin( P.bob * 2 ) * 0.018, - P.y );
		cam.rotation.set( P.pitch, P.yaw, 0, 'YXZ' );

	};

	const leaveView = () => {

		if ( ! free ) return;
		// step out of a fixed viewpoint: keep looking the same way, drop to the nearest walkable ground
		const d = new THREE.Vector3( free.target[ 0 ] - free.pos[ 0 ], free.target[ 1 ] - free.pos[ 1 ], free.target[ 2 ] - free.pos[ 2 ] );
		P.yaw = Math.atan2( - d.x, d.y );
		P.pitch = Math.atan2( d.z, Math.hypot( d.x, d.y ) );
		const g = nav.groundAt( free.pos[ 0 ], free.pos[ 1 ], free.pos[ 2 ] );
		if ( g !== null && free.pos[ 2 ] - g < 2.6 ) {

			P.x = free.pos[ 0 ]; P.y = free.pos[ 1 ]; P.z = eyeZ = g;

		}

		free = null;

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

			leaveView();
			const sp = ( keys.has( 'ShiftLeft' ) || keys.has( 'ShiftRight' ) || m > 1.25 ? 4.5 : 2.2 ) * Math.min( 1, m ) * dt;
			f /= m; s /= m;
			// yaw 0 looks north (+Y); right is +X
			const dx = ( - Math.sin( P.yaw ) * f + Math.cos( P.yaw ) * s ) * sp, dy = ( Math.cos( P.yaw ) * f + Math.sin( P.yaw ) * s ) * sp;
			const ox = P.x, oy = P.y;
			nav.move( P, dx, dy );
			P.bob += Math.hypot( P.x - ox, P.y - oy ) * 2.1;

		}

		// a scripted walk, for measuring from outside: window.__auto = { path: [ [ x, y ], .. ], speed, spin }
		const A = window.__auto;
		if ( A && A.path && A.path.length ) {

			leaveView();
			const [ tx, ty ] = A.path[ 0 ], ex = tx - P.x, ey = ty - P.y, d = Math.hypot( ex, ey ), sp = ( A.speed || 4.5 ) * dt;
			if ( d < Math.max( 0.5, sp ) || ( A.stuck || 0 ) > 40 ) { A.path.shift(); A.stuck = 0; } else {

				const turn = Math.atan2( Math.sin( Math.atan2( - ex, ey ) - P.yaw ), Math.cos( Math.atan2( - ex, ey ) - P.yaw ) );
				P.yaw += turn * Math.min( 1, dt * 4 ); P.pitch *= 1 - Math.min( 1, dt * 4 );
				const ox = P.x, oy = P.y;
				nav.move( P, ex / d * sp, ey / d * sp );
				const moved = Math.hypot( P.x - ox, P.y - oy );
				A.stuck = moved < sp * 0.2 ? ( A.stuck || 0 ) + 1 : 0;
				A.walked = ( A.walked || 0 ) + moved;
				P.bob += moved * 2.1;

			}

		} else if ( A && A.spin ) { leaveView(); P.yaw += A.spin * dt; }

		eyeZ += ( P.z - eyeZ ) * Math.min( 1, dt * 10 );

	};

	// ------------------------------------------------------------------ the bake's visibility sets (tools/pvs.py)
	if ( PVS && bake ) {

		// ?pvs=bake&out=<folder>&token=..[&face=512&step=0.5&square=1&pad=1.5&near=8&eyes=1.45,1.65,1.85&levels=lo,mid&bands=phone,low&limit=200&debug=<eye point>]
		eng.manual = true;
		const res = window.__pvs = { done: false, stage: 'start' };
		try {

			placeCamera();
			await town.settle( eng.camera, 0 );
			eng.update( 0 ); town.update( eng.camera, 0 ); eng.render(); // (one frame: the lowest level of every cell is on the GPU)
			const { bakePVS, checkPVS, holesPVS } = await import( '../engine/pvs.js' );
			const opt = { on: ( s ) => { res.stage = s; } };
			for ( const k of [ 'face', 'step', 'square', 'pad', 'near', 'limit', 'debug', 'n', 'seed', 'tol', 'rel', 'erode', 'voxel' ] ) if ( isFinite( parseFloat( Q.get( k ) ) ) ) opt[ k ] = parseFloat( Q.get( k ) );
			// ?pvs=check[&n=60&face=512&seed=1]: the sets of the bake against what this tier shows from places off the grid
			if ( Q.get( 'pvs' ) === 'check' ) { res.check = await checkPVS( town, opt ); res.done = true; return; }
			if ( Q.get( 'levels' ) ) opt.levels = Q.get( 'levels' ).split( ',' );
			// ?pvs=holes&out=<folder>&token=..[&limit=600&step=1&levels=lo,mid&top=fine&tol=0.4]: where the reduced levels let the eye through (holes.json)
			if ( Q.get( 'pvs' ) === 'holes' ) {

				if ( Q.get( 'top' ) ) opt.top = Q.get( 'top' );
				if ( Q.get( 'through' ) === '0' ) opt.through = false; // (every place with a surface further behind, not only those that show another cell or the sky)
				const holes = await holesPVS( town, opt );
				res.holes = { top: holes.top, points: holes.points, seconds: holes.seconds, levels: Object.fromEntries( Object.entries( holes.levels ).map( ( [ l, v ] ) => [ l, { pixels: v.pixels, of: v.of, cells: Object.keys( v.cells ).length } ] ) ) };
				const path = `renders/_${TOWN_ID}/${Q.get( 'out' ) || 'holes'}/holes.json`;
				const up = await fetch( `${BASE}/api/upload?path=${encodeURIComponent( path )}&token=${Q.get( 'token' )}`, { method: 'POST', body: JSON.stringify( Object.assign( holes, { baked: town.bake.baked } ) ) } );
				if ( ! up.ok ) throw new Error( `upload ${path}: ${up.status}` );
				res.path = path; res.done = true;
				return;

			}

			if ( Q.has( 'bands' ) ) opt.bands = Q.get( 'bands' ).split( ',' ).filter( Boolean ); // (bands=: no tier's mix of levels, the single levels alone)
			if ( Q.get( 'eyes' ) ) opt.eyes = Q.get( 'eyes' ).split( ',' ).map( parseFloat ).filter( isFinite );
			const pvs = await bakePVS( town, opt );
			if ( bakePVS.debug ) window.__pvsDebug = bakePVS.debug;
			res.stats = pvs.stats;
			const path = `renders/_${TOWN_ID}/${Q.get( 'out' ) || 'pvs'}/pvs.json`;
			const up = await fetch( `${BASE}/api/upload?path=${encodeURIComponent( path )}&token=${Q.get( 'token' )}`, { method: 'POST', body: JSON.stringify( pvs ) } );
			if ( ! up.ok ) throw new Error( `upload ${path}: ${up.status}` );
			res.path = path;

		} catch ( e ) {

			res.failed = true; res.error = String( e && e.stack || e );

		}

		res.warnings = town.warnings;
		res.done = true;
		return;

	}

	// ------------------------------------------------------------------ capture mode (review renders)
	if ( CAPTURE ) {

		eng.manual = true;
		const res = window.__capture = { done: false, shots: [], warnings: town.warnings, load_s: + loadS.toFixed( 1 ), stage: 'start' };
		try {

			const token = Q.get( 'token' ), out = Q.get( 'out' );
			const only = Q.get( 'shots' ) ? new Set( Q.get( 'shots' ).split( ',' ) ) : null;
			res.gpu = ( await navigator.gpu?.requestAdapter() )?.info.vendor || ( eng.isWebGPU ? 'webgpu' : 'webgl' );
			const c2 = document.createElement( 'canvas' );
			c2.width = w; c2.height = h;
			const ctx = c2.getContext( '2d' );
			let t = 20, cpu = 0;
			const frame = async ( n ) => {

				for ( let i = 0; i < n; i ++ ) {

					t += 1 / 30;
					const c0 = performance.now();
					eng.update( t );
					town.update( eng.camera, t );
					eng.render();
					cpu += performance.now() - c0;
					await tick();

				}

			};

			// frame rate at the spawn (what a visitor gets first), all detail in
			placeCamera();
			await town.settle( eng.camera, t );
			await frame( 12 );
			await eng.renderer.backend.device?.queue.onSubmittedWorkDone?.();
			const f0 = performance.now();
			cpu = 0;
			await frame( 20 );
			res.cpu_ms = Math.round( cpu / 20 * 10 ) / 10; // main-thread cost of one frame (update + render submit)
			let objects = 0;
			eng.scene.traverseVisible( ( o ) => {

				if ( o.isMesh || o.isSprite || o.isPoints || o.isLine ) objects ++;

			} );
			res.objects = objects;
			if ( Q.get( 'probe' ) === '1' ) {

				// where the frame goes: time each stage, then the render with parts of the scene hidden
				const T = ( fn, n = 10 ) => {

					const a = performance.now();
					for ( let i = 0; i < n; i ++ ) fn( i );
					return Math.round( ( performance.now() - a ) / n * 100 ) / 100;

				};

				const pr = res.probe = {};
				pr.update = T( () => eng.update( t += 0.033 ) );
				pr.town = T( () => town.update( eng.camera, t += 0.033 ) );
				pr.render = T( () => eng.render() );
				// inclusive time per renderer-internal method over 10 frames (wrappers removed afterwards)
				{

					const R = eng.renderer, acc = {}, undo = [];
					const targets = { nodes: R._nodes, geometries: R._geometries, bindings: R._bindings, pipelines: R._pipelines, objects: R._objects, backend: R.backend, renderer: R, lists: R._renderLists, attributes: R._attributes, textures: R._textures };
					for ( const [ tn, obj ] of Object.entries( targets ) ) {

						if ( ! obj ) continue;
						const seen = new Set();
						for ( let pr0 = obj; pr0 && pr0 !== Object.prototype; pr0 = Object.getPrototypeOf( pr0 ) ) for ( const k of Object.getOwnPropertyNames( pr0 ) ) {

							if ( seen.has( k ) || k === 'constructor' ) continue;
							seen.add( k );
							const d = Object.getOwnPropertyDescriptor( pr0, k );
							if ( ! d || typeof d.value !== 'function' ) continue;
							const fn = d.value, name = tn + '.' + k, own = Object.prototype.hasOwnProperty.call( obj, k ), prev = own ? obj[ k ] : undefined;
							const a = acc[ name ] = { t: 0, n: 0 };
							obj[ k ] = function ( ...args ) {

								const t0 = performance.now();
								try {

									return fn.apply( this, args );

								} finally {

									a.t += performance.now() - t0; a.n ++;

								}

							};
							undo.push( () => {

								if ( own ) obj[ k ] = prev; else delete obj[ k ];

							} );

						}

					}

					const bp = THREE.BatchedMesh.prototype, obr = bp.onBeforeRender, ab = acc[ 'batched.onBeforeRender' ] = { t: 0, n: 0 };
					bp.onBeforeRender = function ( ...args ) {

						const t0 = performance.now();
						obr.apply( this, args );
						ab.t += performance.now() - t0; ab.n ++;

					};
					const NF = R._nodes.nodeFrame, ubn = NF.updateBeforeNode, byType = {};
					NF.updateBeforeNode = function ( node ) {

						const t0 = performance.now();
						ubn.call( this, node );
						const k = ( node.constructor?.name || node.type || '?' ) + ':' + node.updateBeforeType;
						const a = byType[ k ] ??= { t: 0, n: 0 };
						a.t += performance.now() - t0; a.n ++;

					};
					for ( let i = 0; i < 10; i ++ ) eng.render();
					NF.updateBeforeNode = ubn;
					pr.before = Object.entries( byType ).sort( ( a, b ) => b[ 1 ].t - a[ 1 ].t ).slice( 0, 12 ).map( ( [ k, v ] ) => `${k} ${( v.t / 10 ).toFixed( 2 )}ms x${Math.round( v.n / 10 )}` );
					bp.onBeforeRender = obr;
					for ( const u of undo ) u();
					pr.hot = Object.entries( acc ).filter( ( e ) => e[ 1 ].t > 2 ).sort( ( a, b ) => b[ 1 ].t - a[ 1 ].t ).slice( 0, 28 )
						.map( ( [ k, v ] ) => `${k} ${( v.t / 10 ).toFixed( 2 )}ms x${Math.round( v.n / 10 )}` );

				}

				const batches = town.group.children.filter( ( o ) => o.isBatchedMesh ), rest = town.group.children.filter( ( o ) => ! o.isBatchedMesh );
				town.group.visible = false;
				pr.render_noTown = T( () => eng.render() );
				town.group.visible = true;
				batches.forEach( ( b ) => ( b.visible = false ) );
				pr.render_noBatches = T( () => eng.render() );
				batches.forEach( ( b ) => ( b.visible = true ) );
				const vis = rest.map( ( o ) => o.visible );
				rest.forEach( ( o ) => ( o.visible = false ) );
				pr.render_onlyBatches = T( () => eng.render() );
				rest.forEach( ( o, i ) => ( o.visible = vis[ i ] ) );
				pr.batches = batches.length; pr.rest = rest.length;
				pr.batchesDrawn = batches.filter( ( b ) => b._multiDrawCount > 0 ).length;
				let lights = 0;
				eng.scene.traverse( ( o ) => {

					if ( o.isPointLight && o.intensity > 0 ) lights ++;

				} );
				pr.lights = lights;

			}

			await eng.renderer.backend.device?.queue.onSubmittedWorkDone?.();
			res.fps = Math.round( 20000 / ( performance.now() - f0 ) );
			// what one frame draws, all passes (the renderer otherwise counts per render call: the last is the output quad)
			const counts = () => {

				const info = eng.renderer.info, was = info.autoReset;
				info.autoReset = false;
				eng.render();
				info.autoReset = was;
				return { draws: info.render.drawCalls, tris: info.render.triangles };

			};

			// (the streaming engine is captured exactly as before: no extra render between its frames)
			Object.assign( res, bake ? counts() : { draws: eng.renderer.info.render.drawCalls, tris: eng.renderer.info.render.triangles } );
			res.spawn = town.visibleStats();
			res.view = {};

			// ad-hoc review cameras: cam=name:x,y,z,tx,ty,tz[,fov];... (town frame, Z up) replace the shot list
			const cams = ( Q.get( 'cam' ) || '' ).split( ';' ).filter( Boolean ).map( ( c ) => {

				const [ name, v ] = c.split( ':' ), n = v.split( ',' ).map( Number );
				return { name, pos: n.slice( 0, 3 ), target: n.slice( 3, 6 ), fov: n[ 6 ] || 62 };

			} );
			for ( const s of cams.length ? cams : data.shots ) {

				if ( ! cams.length && only && ! only.has( s.name ) ) continue;
				res.stage = s.name;
				free = s;
				placeCamera();
				await town.settle( eng.camera, t );
				res.stage = s.name + ':frames';
				await frame( 8 );
				if ( bake ) { const v = town.visibleStats(); res.view[ s.name ] = { ...counts(), lods: v.lods, shown: Math.round( v.tris ), budget: Math.round( v.view || 0 ), by: v.by }; }
				res.stage = s.name + ':capture';
				const img = await eng.capture();
				res.stage = s.name + ':encode';
				if ( img.width === w && img.height === h ) ctx.putImageData( img, 0, 0 );
				else {

					const bmp = await createImageBitmap( img );
					ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
					ctx.drawImage( bmp, 0, 0, w, h );
					bmp.close();

				}

				const path = `_${TOWN_ID}/${out ? out + '/' : ''}${s.name}.png`;
				const blob = await new Promise( ( r ) => c2.toBlob( r, 'image/png' ) );
				const up = await fetch( `${BASE}/api/upload?path=${encodeURIComponent( 'renders/' + path )}&token=${token}`, { method: 'POST', body: blob } );
				if ( ! up.ok ) throw new Error( `upload ${path}: ${up.status}` );
				res.shots.push( path );

			}

			res.stats = town.visibleStats();
			res.seconds = + ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 );
			res.log = window.__log.slice( - 12 );
			// GPU validation / shader errors of the session go first in the warnings the render tool prints
			const gpu = [ ...new Set( window.__log.filter( ( l ) => /^(error|onerror|rejection)|OutputType|INVALID|invalid|shader/i.test( l ) ).map( ( l ) => 'console ' + l.slice( 0, 240 ) ) ) ];
			if ( gpu.length ) res.warnings = [ ...gpu.slice( 0, 6 ), ...town.warnings ];
			res.done = true;

		} catch ( e ) {

			res.error = String( e && e.stack || e );
			res.log = window.__log.slice( - 20 );
			res.failed = true;

		}

		$( 'veil' ).classList.add( 'hide' );
		return;

	}

	// ------------------------------------------------------------------ interactive
	const startShot = data.shots.find( ( s ) => s.name === Q.get( 'shot' ) );
	if ( startShot ) free = startShot;
	placeCamera();
	await town.settle( eng.camera );

	addEventListener( 'resize', () => eng.setSize( innerWidth, innerHeight, pixelRatio() ) );
	addEventListener( 'keydown', ( e ) => {

		keys.add( e.code );
		if ( e.code === 'KeyV' ) nextView();
		if ( e.code === 'KeyH' ) $( 'hud' ).style.display = $( 'hud' ).style.display === 'none' ? '' : 'none';

	} );
	addEventListener( 'keyup', ( e ) => keys.delete( e.code ) );
	addEventListener( 'blur', () => keys.clear() );
	let viewI = - 1;
	const nextView = () => {

		viewI = ( viewI + 1 ) % data.shots.length;
		free = data.shots[ viewI ];

	};

	$( 'tour' ).onclick = nextView;
	// the selectors reload the page (a tier, a bake and the post chain are each built once): at the place and heading of the walk
	const reloadWith = ( set ) => {

		const u = new URL( location.href );
		for ( const k in set ) if ( set[ k ] === null ) u.searchParams.delete( k ); else u.searchParams.set( k, set[ k ] );
		u.searchParams.delete( 'shot' );
		u.searchParams.set( 'at', [ P.x, P.y, P.z, P.yaw, P.pitch ].map( ( v ) => + v.toFixed( 2 ) ).join( ',' ) );
		location.href = u.href;

	};
	// the lens flare: an entry of the same menu, but switched at once (a post effect - its passes are skipped while
	// it is off, nothing is built again). Off unless the visitor chose it; ?nb_flare= sets its strength
	const flare = { u: eng.nbPass && eng.nbPass._flare ? eng.nbPass.flareIntensity : null, on: false, strength: 0.25 };
	if ( flare.u ) {

		const asked = parseFloat( Q.get( 'nb_flare' ) ), saved = localStorage.getItem( 'tsq.flare' );
		if ( asked > 0 ) flare.strength = asked;
		flare.set = ( on ) => {

			flare.on = on;
			flare.u.value = on ? flare.strength : 0;
			$( 'flare' ).textContent = 'Lens flare: ' + ( on ? 'on' : 'off' );
			if ( window.__tele && window.__fx ) { window.__fx.flare = on; window.__tele.set( { fx: window.__fx } ); }

		};
		flare.set( isFinite( asked ) ? asked > 0 : saved === '1' );

	} else $( 'flare' )?.remove(); // (taken out, not hidden: a phone's picker shows hidden entries)

	// the lanterns' shadows on the ground: a picture the ground's pixels read (a bake with `gsh`, see BakedTown._materials)
	// - one number in the shader, so switched at once as well. On unless the visitor chose otherwise; ?gsh=0 | 1
	const gsh = { can: !! ( bake && town.groundShadows !== null && town.groundShadows !== undefined ), on: true };
	if ( gsh.can ) {

		const asked = Q.get( 'gsh' ), saved = localStorage.getItem( 'tsq.gsh' );
		gsh.set = ( on ) => {

			gsh.on = on;
			town.setGroundShadows( on );
			if ( $( 'gshadow' ) ) $( 'gshadow' ).textContent = 'Ground shadows: ' + ( on ? 'on' : 'off' );
			if ( window.__tele && window.__fx ) { window.__fx.gsh = on; window.__tele.set( { fx: window.__fx } ); }

		};
		gsh.set( asked === '0' || asked === '1' ? asked === '1' : saved !== '0' );

	} else $( 'gshadow' )?.remove();
	if ( $( 'fx' ) && ! $( 'fx' ).children.length ) $( 'fx' ).remove();

	$( 'quality' ).onchange = ( e ) => {

		if ( e.target.value === 'flare' || e.target.value === 'gsh' ) {

			if ( e.target.value === 'flare' ) {

				if ( flare.u ) flare.set( ! flare.on );
				localStorage.setItem( 'tsq.flare', flare.on ? '1' : '0' );

			} else {

				if ( gsh.can ) gsh.set( ! gsh.on );
				localStorage.setItem( 'tsq.gsh', gsh.on ? '1' : '0' );

			}

			e.target.value = quality; e.target.blur();
			return;

		}

		if ( ! STUDIO[ e.target.value ] ) localStorage.setItem( 'tsq.quality.v2', e.target.value ); // (the streaming engine: for this visit only)
		reloadWith( { quality: e.target.value } );

	};
	if ( bake && ! LEGACY ) {

		// a second look: a bake that has one in its own vertices (`look2`, tools/webbake.py --set look2.KEY) switches
		// at once, with one uniform. (The town ships with one look - the far-reaching lamps - and no selector.)
		if ( bake.look2 ) {

			$( 'look' ).value = town.lookI ? '_next_look2' : '';
			$( 'look' ).title = 'how the lanterns light the town: both are in the bake, the switch is immediate (key L)';
			const setLook = ( i ) => {

				$( 'look' ).value = town.setLook( i ) ? '_next_look2' : '';
				const u = new URL( location.href );
				if ( town.lookI ) u.searchParams.set( 'look', '2' ); else u.searchParams.delete( 'look' );
				history.replaceState( null, '', u.href );

			};
			$( 'look' ).onchange = ( e ) => { setLook( e.target.value ? 1 : 0 ); e.target.blur(); };
			addEventListener( 'keydown', ( e ) => { if ( e.code === 'KeyL' && ! e.repeat && ! e.ctrlKey && ! e.metaKey && ! e.altKey ) setLook( town.lookI ? 0 : 1 ); } );
			$( 'look' ).hidden = false;

		}

	}

	const look = ( dx, dy, k ) => {

		leaveView();
		P.yaw -= dx * k;
		P.pitch = THREE.MathUtils.clamp( P.pitch - dy * k, - 1.35, 1.35 );

	};

	canvas.addEventListener( 'click', () => {

		if ( ! touch ) canvas.requestPointerLock?.();

	} );
	addEventListener( 'mousemove', ( e ) => {

		if ( document.pointerLockElement === canvas ) look( e.movementX, e.movementY, 0.0022 );

	} );

	// touch: left stick walks, dragging anywhere else looks
	const st = $( 'stick' ), knob = st.firstElementChild;
	let stickId = null, lookId = null, lx = 0, ly = 0;
	st.addEventListener( 'pointerdown', ( e ) => {

		stickId = e.pointerId; st.setPointerCapture( e.pointerId );
		e.preventDefault();

	} );
	st.addEventListener( 'pointermove', ( e ) => {

		if ( e.pointerId !== stickId ) return;
		const r = st.getBoundingClientRect();
		let x = ( e.clientX - r.left - r.width / 2 ) / ( r.width / 2 ), y = ( e.clientY - r.top - r.height / 2 ) / ( r.height / 2 );
		const m = Math.hypot( x, y );
		if ( m > 1 ) {

			x /= m; y /= m;

		}

		stick.x = x * 1.3; stick.y = y * 1.3;
		knob.style.transform = `translate(${x * 36}px, ${y * 36}px)`;

	} );
	const stickEnd = ( e ) => {

		if ( e.pointerId !== stickId ) return;
		stickId = null; stick.x = stick.y = 0; knob.style.transform = '';

	};

	st.addEventListener( 'pointerup', stickEnd );
	st.addEventListener( 'pointercancel', stickEnd );
	canvas.addEventListener( 'pointerdown', ( e ) => {

		if ( e.pointerType === 'mouse' ) return;
		lookId = e.pointerId; lx = e.clientX; ly = e.clientY;

	} );
	canvas.addEventListener( 'pointermove', ( e ) => {

		if ( e.pointerId !== lookId ) return;
		look( e.clientX - lx, e.clientY - ly, 0.0045 );
		lx = e.clientX; ly = e.clientY;

	} );
	const lookEnd = ( e ) => {

		if ( e.pointerId === lookId ) lookId = null;

	};

	canvas.addEventListener( 'pointerup', lookEnd );
	canvas.addEventListener( 'pointercancel', lookEnd );

	// the town is there: the veil goes, the help line says how to walk (a pointer lock needs a click: the first on the canvas)
	$( 'msg' ).textContent = touch ? 'left stick to walk · drag to look' : 'click to look around · WASD to walk · Shift to run';
	$( 'bar' ).style.display = 'none';
	$( 'veil' ).classList.add( 'hide' );

	const clock = new THREE.Clock();
	let acc = 0, frames = 0, fps = 0, tSec = 0;
	// the page's own numbers, for measuring from outside (window.__stats; worst: the longest frame so far, ms - reset it to watch a walk)
	const ST = window.__stats = { first: 0, worst: 0, frames: 0, load_s: + loadS.toFixed( 2 ), tier: quality, baked: !! bake, legacy: LEGACY };
	// baked: which geometry files (?fmt=glb | packed) and their load times - a live object, engine/lite_glb.js keeps it
	if ( bake && town.geo ) { ST.fmt = town.geo.fmt; ST.geo = town.geo; }
	window.__player = P;
	// (what the picture is made with, for the visitors' reports: blur levels of the natural bloom, HDR emitters, HDR canvas)
	if ( window.__tele && eng.HDR ) window.__tele.set( { fx: window.__fx = { nb: eng.nbPass ? eng.nbPass.levels + '/' + eng.nbPass.divisor : 0, flare: !! ( eng.nbPass && eng.nbPass._flare && eng.nbPass.flareIntensity.value > 0 ), gsh: bake ? town.groundShadows : null, hdr: eng.HDR.on, hdrOut: eng.HDR.out } } );
	window.__town = town; // (measuring from outside: e.g. the baked town's visibility sets on and off in one session)
	window.__view = ( f ) => { free = f; }; // a fixed viewpoint from outside: { pos, target, fov } in the town's frame, null: back to the walk
	const GPUTIME = Q.get( 'gputime' ) === '1' && eng.renderer.backend && eng.renderer.backend.trackTimestamp ? { sum: 0, n: 0, busy: false } : null;
	// ?rep=K (measuring): K whole frames per displayed one with the camera swaying a little, so that the GPU is kept
	// busy (an idle one clocks down and its times mean nothing) and what only happens while one moves does happen
	const REP = Math.max( 1, Math.min( 64, + Q.get( 'rep' ) || 1 ) );
	let sway = 0;
	eng.renderer.setAnimationLoop( () => {

		const dt = Math.min( 0.05, clock.getDelta() );
		tSec += dt;
		walk( dt );
		placeCamera();
		eng.update( tSec );
		town.update( eng.camera, tSec );
		eng.render();
		// (three's passes draw once per frame of its own count: step that count as its animation loop does)
		for ( let i = 1; i < REP; i ++ ) { P.yaw += 0.004 * Math.cos( ( sway ++ ) * 0.02 ); placeCamera(); const nf = eng.renderer._nodes.nodeFrame; nf.update(); eng.renderer.info.frame = nf.frameId; eng.update( tSec ); town.update( eng.camera, tSec ); eng.render(); }
		// ?gputime=1: what a frame's render passes took on the GPU (ms, all passes of the last frame drawn added up;
		// needs timestamp queries) -> __stats.gpu
		if ( GPUTIME && ! GPUTIME.busy ) {

			GPUTIME.busy = true;
			eng.renderer.resolveTimestampsAsync( 'render' ).then( () => { const ms = eng.renderer.info.render.timestamp; if ( ms > 0 ) { GPUTIME.sum += ms; GPUTIME.n ++; } } ).catch( () => {} ).then( () => { GPUTIME.busy = false; } );

		}

		if ( ! ST.first ) ST.first = Math.round( performance.now() ); // ms from opening the page to the first frame
		const ms = performance.now() - ( ST.last || performance.now() ); ST.last = performance.now();
		if ( ms > ST.worst ) ST.worst = Math.round( ms );
		{ const r = eng.renderer.info.render; if ( r.triangles > ( ST.peak || 0 ) ) ST.peak = r.triangles; if ( r.drawCalls > ( ST.peakDraws || 0 ) ) ST.peakDraws = r.drawCalls; }
		acc += dt; frames ++;
		if ( acc > 0.5 ) {

			fps = Math.round( frames / acc ); acc = 0; frames = 0;
			const s = town.visibleStats(), info = eng.renderer.info.render;
			Object.assign( ST, { fps, draws: info.drawCalls, tris: info.triangles, town: s, frames: ST.frames + 1 } );
			if ( GPUTIME && GPUTIME.n ) { ST.gpu = + ( GPUTIME.sum / GPUTIME.n ).toFixed( 2 ); GPUTIME.sum = GPUTIME.n = 0; }
			$( 'hud' ).textContent = `${fps} fps  ${info.drawCalls} draws  ${( info.triangles / 1e6 ).toFixed( 2 )} M tris\n` +
				`${s.items} items  LOD ${s.lods.join( '/' )}  ${s.pending ? 'streaming ' + s.pending : ''}\n` +
				`${quality}${bake ? ' baked' : ''}${bake && LEGACY ? ' v1' : ''} · ${eng.isWebGPU ? 'WebGPU' : 'WebGL2'}${eng.HDR && eng.HDR.out ? ' HDR' : ''} · ${data.missing?.length || 0} assets pending` +
				( BAKED && loadBake.note ? '\n' + loadBake.note : '' );

		}

	} );

}

main().catch( ( e ) => {

	console.error( e );
	document.getElementById( 'msg' ).textContent = 'failed to start: ' + ( e.message || e );
	if ( CAPTURE ) window.__capture = { done: false, failed: true, error: String( e && e.stack || e ), log: window.__log.slice( - 20 ) };

} );
