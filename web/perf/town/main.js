// Walkable lantern town square. Loads out/town/town.json (tools/town.py) through the shared engine.
//   ?quality=high|medium|low|phone   render tier (default: phone on touch / weak devices, low elsewhere; the selector remembers a choice)
//   ?shot=<name>                 start at a review viewpoint
//   ?baked=0                     low / phone: the streaming path instead of the baked build (out/web/_bake, tools/webbake.py)
//   ?capture=1&token=..&w=..&h=..[&shots=a,b][&out=dir]   review renders: every shot -> out/renders/_town/[dir/]<shot>.png

import { createEngine, THREE, sourceRadius } from '../engine/engine.js';
import { Town, TIERS } from '../engine/town.js';
import { loadBake, createLiteEngine, BakedTown, bakedPixelRatio } from '../engine/lite.js';

const BASE = '../..';
const Q = new URLSearchParams( location.search );
const TOWN_ID = /^town\d+$/.test( Q.get( 'town' ) || '' ) ? Q.get( 'town' ) : 'town'; // ?town=town2: a second town assembly (out/town2/town.json, renders -> out/renders/_town2/)
const CAPTURE = Q.get( 'capture' ) === '1';
const touch = matchMedia( '(pointer: coarse)' ).matches;
// the light tiers are the default: the full scene (4 GB of GPU memory) is opt-in through the selector / ?quality=high
const weak = touch || ! navigator.gpu || ( navigator.deviceMemory || 8 ) <= 4; // phones, tablets, no WebGPU, little memory
const tierName = [ Q.get( 'quality' ), localStorage.getItem( 'tsq.quality.v2' ), weak ? 'phone' : 'low' ].find( ( t ) => TIERS[ t ] );
const T = TIERS[ tierName ];
const LIGHT = T.ssr === false; // low + phone
const $ = ( id ) => document.getElementById( id );
document.body.classList.toggle( 'touch', touch && ! CAPTURE );
document.body.classList.toggle( 'capture', CAPTURE );
$( 'quality' ).value = tierName;

const tick = () => new Promise( ( r ) => {

	const c = new MessageChannel();
	c.port1.onmessage = () => r();
	c.port2.postMessage( 0 );

} );

const size = () => ( CAPTURE ? [ + ( Q.get( 'w' ) || 1600 ), + ( Q.get( 'h' ) || 900 ) ] : [ innerWidth, innerHeight ] );
let bake = null; // manifest of the baked build the light tiers run (null: none, the town streams asset by asset)
// (review renders are drawn at the tier's own pixel ratio - what a high-dpi screen shows - and reduced to the frame
// size: thin railings, bunting strings and roof tiles get the samples the live picture has; ?pr= overrides)
const pixelRatio = () => ( CAPTURE ? ( parseFloat( Q.get( 'pr' ) ) || ( bake ? 1 : T.pixelRatio ) || 1 ) : bake ? bakedPixelRatio( bake, innerWidth, innerHeight ) : Math.min( devicePixelRatio || 1, T.pixelRatio * ( touch ? 1 : 1 ) ) * ( LIGHT ? Math.min( 1, 1.5 / ( devicePixelRatio || 1 ) ) : 1 ) );

async function main() {

	const t0 = performance.now();
	const [ w, h ] = size();
	const canvas = $( 'gl' );
	if ( LIGHT && Q.get( 'baked' ) !== '0' ) bake = await loadBake( BASE, tierName, TOWN_ID );
	const eng = window.eng = bake ? await createLiteEngine( { canvas, base: BASE, width: w, height: h, pixelRatio: pixelRatio(), fov: 62, forceWebGL: Q.get( 'webgl' ) === '1', bake } ) : await createEngine( { canvas, base: BASE, width: w, height: h, pixelRatio: pixelRatio(), fov: 62, forceWebGL: Q.get( 'webgl' ) === '1',
		anisotropy: T.aniso || ( LIGHT ? 4 : 8 ), maxTextureSize: T.tex || 2048, reflections: T.ssr !== false && Q.get( 'ssr' ) !== '0', envProbe: T.probes !== false } );
	if ( Q.has( 'ssr' ) ) eng.P.ssr.value = parseFloat( Q.get( 'ssr' ) );
	if ( Q.has( 'env' ) ) eng.P.env.value = parseFloat( Q.get( 'env' ) );
	// look: ?moon= ?ambient= ?exposure= ?bloom= ?fog= ?lamp= override the defaults (for tuning from the URL)
	// a moonlit night: enough cool fill that roofs and shadow sides read navy with detail, lamps held under the shoulder
	const LOOK = { fog: 0.0075, ambient: 1.3, moon: 2.0, exposure: 0.72, bloom: 0.3, lamp: 0.5, clouds: 1.0, cover: 0.36, cloudGain: 2.4, skyGlow: 4, haze: 0.35 };
	for ( const k in LOOK ) if ( Q.has( k ) && isFinite( parseFloat( Q.get( k ) ) ) ) LOOK[ k ] = parseFloat( Q.get( k ) );
	eng.setRig( 'night', LOOK );
	eng.P.contrast.value = 1.06; eng.P.saturation.value = 1.03;
	eng.P.knee.value = 1.6; eng.P.kneeGlow.value = 2.2; // highlight roll-off: lit walls / glowing glass (scene-linear ceilings)
	eng.lib.u.windowGlow.value = 0.75;
	eng.lib.u.soft.value = 1; // weathering tuned for street distance (thin wear, crevice / rain-streak grime)
	// grade / tone overrides for tuning: ?tm=agx&contrast=1.05&saturation=1&vignette=0.3&lift=0.002
	if ( Q.get( 'tm' ) === 'agx' ) eng.renderer.toneMapping = THREE.AgXToneMapping;
	for ( const k of [ 'contrast', 'saturation', 'vignette', 'lift', 'knee', 'kneeGlow', 'tame', 'free', 'roofFill' ] ) if ( Q.has( k ) && isFinite( parseFloat( Q.get( k ) ) ) ) eng.P[ k ].value = parseFloat( Q.get( k ) );
	eng.lib.u.wet.value = 0.68; // the town is seen just after rain
	eng.lib.u.grime.value = 0.8;
	sourceRadius.value = Q.has( 'srcr' ) ? parseFloat( Q.get( 'srcr' ) ) : 0.6; // lanterns light as small area sources
	// material look overrides for tuning: ?wet=0.3&grime=0.7&wear=1 (global uniforms shared by every material)
	for ( const k of [ 'wet', 'grime', 'wear', 'special', 'windowGlow', 'emissive', 'puddleRough', 'filmRough', 'ripple', 'poolCap', 'soft' ] ) if ( Q.has( k ) && isFinite( parseFloat( Q.get( k ) ) ) && eng.lib.u[ k ] ) eng.lib.u[ k ].value = parseFloat( Q.get( k ) );
	if ( ! T.ao ) eng.setPost( true, { ao: false } );
	eng.camera.near = 0.08; eng.camera.far = 1500;

	const town = window.town = bake ? new BakedTown( eng, bake, { wait: CAPTURE || Q.has( 'shot' ) ? 'all' : 'near' } ) : new Town( eng, { base: BASE, tier: tierName, lamp: LOOK.lamp, batch: Q.get( 'batch' ) !== '0', cap: Q.has( 'cap' ) ? parseFloat( Q.get( 'cap' ) ) : undefined } );
	await town.load( `${BASE}/out/${TOWN_ID}/town.json?t=${Date.now()}`, ( f, what ) => {

		$( 'bar' ).firstElementChild.style.width = Math.round( f * 100 ) + '%';
		$( 'msg' ).textContent = 'lighting the lamps… ' + ( what || '' );

	} );
	const data = town.data, nav = town.nav;
	if ( data.frame && isFinite( data.frame.water_z ) && Q.get( 'tide' ) !== '0' ) eng.lib.u.tideY.value = data.frame.water_z; // wet band on the canal walls
	const loadS = ( performance.now() - t0 ) / 1000;

	// ------------------------------------------------------------------ player
	const P = { x: data.spawn.pos[ 0 ], y: data.spawn.pos[ 1 ], z: 0, yaw: THREE.MathUtils.degToRad( data.spawn.yaw || 0 ), pitch: 0, eye: 1.65, bob: 0 };
	P.z = nav.groundAt( P.x, P.y, 50 ) ?? 0;
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

		eyeZ += ( P.z - eyeZ ) * Math.min( 1, dt * 10 );

	};

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
			res.draws = eng.renderer.info.render.drawCalls;
			res.tris = eng.renderer.info.render.triangles;
			res.spawn = town.visibleStats();

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
	$( 'quality' ).onchange = ( e ) => {

		localStorage.setItem( 'tsq.quality.v2', e.target.value );
		const u = new URL( location.href );
		u.searchParams.set( 'quality', e.target.value );
		location.href = u.href;

	};

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

	$( 'msg' ).textContent = touch ? 'left stick to walk · drag to look' : 'click to look around · WASD to walk · Shift to run';
	$( 'bar' ).style.display = 'none';
	$( 'go' ).style.display = 'block';
	$( 'go' ).onclick = () => {

		$( 'veil' ).classList.add( 'hide' );
		if ( ! touch ) canvas.requestPointerLock?.();

	};

	const clock = new THREE.Clock();
	let acc = 0, frames = 0, fps = 0, tSec = 0;
	eng.renderer.setAnimationLoop( () => {

		const dt = Math.min( 0.05, clock.getDelta() );
		tSec += dt;
		walk( dt );
		placeCamera();
		eng.update( tSec );
		town.update( eng.camera, tSec );
		eng.render();
		acc += dt; frames ++;
		if ( acc > 0.5 ) {

			fps = Math.round( frames / acc ); acc = 0; frames = 0;
			const s = town.visibleStats(), info = eng.renderer.info.render;
			$( 'hud' ).textContent = `${fps} fps  ${info.drawCalls} draws  ${( info.triangles / 1e6 ).toFixed( 2 )} M tris\n` +
				`${s.items} items  LOD ${s.lods.join( '/' )}  ${s.pending ? 'streaming ' + s.pending : ''}\n` +
				`${tierName}${bake ? ' baked' : ''} · ${eng.isWebGPU ? 'WebGPU' : 'WebGL2'} · ${data.missing?.length || 0} assets pending`;

		}

	} );

}

main().catch( ( e ) => {

	console.error( e );
	document.getElementById( 'msg' ).textContent = 'failed to start: ' + ( e.message || e );
	if ( CAPTURE ) window.__capture = { done: false, failed: true, error: String( e && e.stack || e ), log: window.__log.slice( - 20 ) };

} );
