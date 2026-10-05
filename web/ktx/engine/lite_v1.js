// The first baked runtime, as it went live: kept for ?legacy=1 (phone / low) and for ?town=town2 of web/town/ until
// the towns are baked again for web/engine/lite.js. Reads out/web/_bake (tools/webbake_v1.py).
//
// Lean runtime for the baked town (tools/webbake_v1.py -> out/web/_bake/): what the light tiers of the walkable demo run.
//
// The bake has already done the expensive part offline: every lantern, the moon, the sky fill and the occlusion are
// folded into a per-vertex colour, the static town is merged into one mesh per 16 m cell (three levels of detail) and
// the materials are pre-shaded layers of one small texture array. So this path needs no lights, no shadow maps, no
// reflection probes, no G-buffer: one unlit draw per visible cell, two instanced sprite draws (lantern glows, flames),
// the sky, and a small post chain. createLiteEngine() and BakedTown mirror the parts of createEngine() / Town that
// the town page uses, so the page code stays the same for both paths.

import * as THREE from 'three';
import {
	pass, attribute, uv, varying, texture, vec2, vec3, vec4, float, int, mix, step, max, min, exp, length, dot, pow,
	smoothstep, saturate, oneMinus, uniform, positionWorld, positionView, renderOutput, screenUV, screenCoordinate,
	interleavedGradientNoise, time, fract, sin, abs, instancedBufferAttribute, cameraPosition, normalize,
	textureLoad, ivec2, dFdx, dFdy, cross, sign, floor, clamp, sqrt, positionLocal, modelWorldMatrixInverse
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { createSky, MaterialLib, haloCore } from './engine.js';
import { Nav } from './town.js';

// what each baked tier draws. mid / near: distance (m) from the camera to a cell inside which that level is shown
// (a cell leaves a level 20 % + 2 m further out than it entered it - `hold` m at most - so walking along a border
// swaps nothing);
// top: the level shown inside `near` - "near" (LOD1 geometry within 2 cm) or "ring" (the same in clusters that are
// culled on their own, each with its parts in order of size, so it can be drawn down to the parts that still show
// at its distance: the phone's);
// tris: ceiling on the triangles shown, all cells at their level (so also on what one view can hold, however the
// town grows) - or with budget 'view', on what any one heading and pitch from where the camera stands can have in view,
// spent on the top level around the camera first and on mid after it (the phone: its ring is what one sees up
// close, and what is behind the camera costs memory, not frame time); reach: how far mid should get before the
// ring gives up small parts, and then its farthest cells, for it; resident: ceiling on the triangles held on the GPU (a level
// is held whole, in view or not), kept by giving up the farthest cells of a level;
// pixels: frame buffer budget;
// glints: lanterns per ground cell whose highlight on wet / polished surfaces is evaluated per pixel (0 = none)
// merge: lanterns closer together than this (m) count as one of them
export const BAKED_TIERS = {
	phone: { tex: 256, mid: 40, near: 14, top: 'ring', tris: 0.975e6, budget: 'view', reach: 20, hold: 3, resident: 1.8e6, pixels: 0.8e6, bloom: false, fxaa: true, aniso: 4, fetch: 2, glints: 2, merge: 5, halo: 0.3 },
	low: { tex: 512, mid: 64, near: 20, top: 'near', tris: 2.7e6, pixels: 1.7e6, bloom: true, fxaa: true, aniso: 8, fetch: 3, glints: 4, merge: 4 },
};

const K_OPAQUE = 0, K_GLASS = 1, K_WATER = 2, K_FALL = 3, K_CUT = 4;
// the glint light grid (_lightGrid): lanterns closer than this share an entry (m, a tier's `merge` overrides); floor
// to lantern, squared, when ranking them; a lantern fades in between 1 x and GLINT_FADE x the threshold of the place
const GLINT_MERGE = 2, GLINT_H2 = 2.25, GLINT_FADE = 2;
// camera pitches the view budget is kept for (radians; tools/perf_sim.py reads this line)
const VIEW_PITCH = [ 0, 15, 30, 45, 60, 75, - 25, - 50 ].map( ( d ) => d * Math.PI / 180 );

// Is the bake still the town? Its manifest carries the fingerprint of the town.json it was made from; when the
// assembly beside it (out/<town>/town.json) has been rebuilt since, the bake shows a town that no longer exists.
// -> a line saying so, or null (current, or no assembly to compare with: a deployed copy of the bake alone)
async function bakeAge( base, town, man ) {

	try {

		const r = await fetch( `${base}/out/${town}/town.json`, { cache: 'no-cache' } );
		if ( ! r.ok || ! man.town ) return null;
		const buf = await r.arrayBuffer();
		let same = buf.byteLength === man.town.bytes;
		if ( same && man.town.sha1 && globalThis.crypto?.subtle ) {

			const h = new Uint8Array( await crypto.subtle.digest( 'SHA-1', buf ) );
			same = Array.from( h, ( b ) => b.toString( 16 ).padStart( 2, '0' ) ).join( '' ) === man.town.sha1;

		}

		const cmd = 'python3 tools/webbake_v1.py' + ( town === 'town' ? '' : ' --town ' + town );
		return same ? null : `the baked build is older than out/${town}/town.json (re-bake: ${cmd})`;

	} catch ( e ) {

		return null;

	}

}

// the bake's manifest, or null when there is no usable bake - none at all, one in another format, or one made from an
// older town.json: the page then falls back to the streaming path, and loadBake.note says why (for the console / HUD).
// `town`: the assembly (tools/webbake_v1.py --town <name>); the first one lives in out/web/_bake, others in _bake_<name>.
// stale: use a bake that is older than its town all the same (?baked=stale)
export async function loadBake( base, tier, town = 'town', { stale = false } = {} ) {

	loadBake.note = null;
	if ( ! BAKED_TIERS[ tier ] || ! /^\w+$/.test( town ) ) return null;
	// ?bakedir1=_bake_stage: a trial bake beside the real one (out/web/<name>, a name that starts with _bake_stage; the towns
	// after the first in <name>_<town>) - for comparing a new bake with the live one
	const trial = new URLSearchParams( location.search ).get( 'bakedir1' ) || '';
	const BAKE_DIR = 'out/web/' + ( /^_bake_stage\w*$/.test( trial ) ? ( town === 'town' ? trial : trial + '_' + town ) : town === 'town' ? '_bake' : '_bake_' + town );
	try {

		const r = await fetch( `${base}/${BAKE_DIR}/manifest.json`, { cache: 'no-cache' } );
		if ( ! r.ok ) return null;
		const man = await r.json();
		if ( man.version !== 1 || man.codec !== 'meshopt' || ! man.tex[ BAKED_TIERS[ tier ].tex ] ) {

			loadBake.note = 'the baked build has another format than this page reads (re-bake: python3 tools/webbake_v1.py)';
			return null;

		}

		const old = await bakeAge( base, town, man );
		if ( old ) {

			loadBake.note = old + ( stale ? '' : ': streaming the town instead' );
			if ( ! stale ) return null;

		}

		man.dir = `${base}/${BAKE_DIR}`;
		man.tier = { ...BAKED_TIERS[ tier ] };
		// tuning from the URL: ?mid=30&near=10&tris=500000&pixels=1e6&glints=0
		const q = new URLSearchParams( location.search );
		for ( const k of [ 'mid', 'near', 'tris', 'pixels', 'glints', 'merge', 'halo', 'aniso' ] ) if ( q.has( k ) && isFinite( parseFloat( q.get( k ) ) ) ) man.tier[ k ] = parseFloat( q.get( k ) );
		if ( q.get( 'top' ) === 'near' || q.get( 'top' ) === 'ring' ) man.tier.top = q.get( 'top' );
		return man;

	} catch ( e ) {

		return null;

	}

}

// frame buffer scale for a baked tier: a pixel budget instead of a fixed ratio (the unlit shader is cheap, so a
// phone gets a sharp picture; a 4K desktop does not get four times the work)
export function bakedPixelRatio( bake, w, h ) {

	return Math.max( 0.5, Math.min( devicePixelRatio || 1, Math.sqrt( bake.tier.pixels / Math.max( 1, w * h ) ) ) );

}

// ---------------------------------------------------------------------------------------- engine

export async function createLiteEngine( { canvas, base = '.', width = 1280, height = 720, pixelRatio = 1, fov = 62, forceWebGL = false, bake } ) {

	const T = bake.tier;
	// reversed float depth: the usual 24 bit buffer with the near plane at 5 cm resolves 1 mm at 30 m and 4 mm at 60 m,
	// and the lit rooms behind the windows are cards a few millimetres apart - they shimmered as one walked (rz=0: off)
	const reversedDepthBuffer = new URLSearchParams( location.search ).get( 'rz' ) !== '0';
	const renderer = new THREE.WebGPURenderer( { canvas, antialias: false, alpha: false, forceWebGL, reversedDepthBuffer } );
	renderer.setPixelRatio( pixelRatio );
	renderer.setSize( width, height, false );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.25;
	await renderer.init();
	const isWebGPU = renderer.backend.isWebGPUBackend === true;

	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera( fov, width / height, 0.05, 2000 );
	const noise = new MaterialLib( { base } ).noise; // the engine's tiling noise (sky, flames, water)
	const L = bake.look;
	const moonDir = new THREE.Vector3( ...L.moon_dir ).normalize();
	const sky = createSky( moonDir, noise );
	scene.add( sky );

	// same night haze as the full engine; the baked materials apply it themselves (before the highlight knee)
	const FOG = { color: uniform( new THREE.Color( 0.012, 0.015, 0.024 ) ), density: uniform( 0.0 ), height: uniform( 18 ) };
	const fogAmount = () => {

		const thin = exp( max( positionWorld.y, 0 ).div( FOG.height ).negate() ).mul( 0.8 ).add( 0.2 );
		const fx = length( positionView ).mul( FOG.density ).mul( thin );
		return oneMinus( exp( fx.mul( fx ).negate() ) );

	};

	const P = {
		ao: uniform( 0 ), bloom: uniform( 0.3 ), vignette: uniform( 0.35 ), grain: uniform( 0.012 ),
		tint: uniform( new THREE.Color( 1, 1, 1 ) ), lift: uniform( 0.0 ), contrast: uniform( 1.12 ), saturation: uniform( 1.08 ),
		ssr: uniform( 0 ), env: uniform( 0 ), knee: uniform( 0.0 ), kneeGlow: uniform( 3.0 ),
		tame: uniform( 0 ), free: uniform( 0 ), roofFill: uniform( 0 ), // of the full engine only: taken from the URL, not used
	};
	// highlight knee of the full engine: lit surfaces roll off towards a ceiling, glowing ones towards a higher one
	const knee = ( c, glowing ) => {

		const m = mix( P.knee, max( P.kneeGlow, P.knee ), glowing );
		const t = m.mul( 0.35 );
		const Lm = max( max( c.r, c.g ), max( c.b, 1e-4 ) );
		const x = Lm.sub( t ).max( 0 ).div( m.sub( t ).max( 1e-4 ) );
		const Lk = min( Lm, t ).add( m.sub( t ).mul( x.div( x.add( 1 ) ) ) );
		return c.mul( mix( float( 1 ), Lk.div( Lm ), step( 0.001, P.knee ) ) );

	};

	// --- post: scene -> (bloom) -> grade -> tone map -> (FXAA) -> dither. No G-buffer, no AO / reflection passes.
	const pipeline = new THREE.RenderPipeline( renderer );
	const scenePass = pass( scene, camera );
	const colorTex = scenePass.getTextureNode( 'output' );
	let comp = colorTex.rgb;
	let bloomPass = null;
	if ( T.bloom ) {

		bloomPass = bloom( comp, 1.0, 0.55, 1.1 );
		comp = comp.add( bloomPass.rgb.mul( P.bloom ) );

	}

	const vg = screenUV.sub( 0.5 );
	comp = comp.mul( oneMinus( dot( vg, vg ).mul( P.vignette ).mul( 1.6 ) ) ).mul( P.tint ).add( P.lift );
	comp = pow( max( comp, 0 ).div( 0.18 ), vec3( P.contrast ) ).mul( 0.18 );
	comp = mix( vec3( dot( comp, vec3( 0.2126, 0.7152, 0.0722 ) ) ), comp, P.saturation ).max( 0 );
	const gr = fract( sin( dot( screenUV.mul( 917.3 ).add( fract( time ) ), vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) ).sub( 0.5 );
	comp = comp.add( gr.mul( P.grain ).mul( comp.add( 0.02 ) ) );
	pipeline.outputColorTransform = false;
	const dither = vec4( vec3( interleavedGradientNoise( screenCoordinate.xy ).sub( 0.5 ).div( 160 ) ), 0 );
	const shown = renderOutput( vec4( comp, 1 ) );
	pipeline.outputNode = ( T.fxaa ? fxaa( shown ) : shown ).add( dither );

	let captureRT = null;
	const stub = () => ( { value: 0 } );
	const eng = {
		lite: true, renderer, scene, camera, pipeline, sky, moonDir, P, FOG, fogAmount, knee, noise, bloomPass, isWebGPU, width, height,
		lanterns: [], flames: [], partSets: [],
		// the material look is baked: these only keep the page's look code working (it sets them for the streaming path)
		lib: { noise, warnings: new Set(), u: { wet: stub(), aoDirect: stub(), grime: stub(), wear: stub(), special: stub(), macro: stub(), windowGlow: stub(), emissive: stub(),
			puddleRough: stub(), filmRough: stub(), ripple: stub(), poolCap: stub(), tideY: stub(), tideH: stub(), soft: stub() } },

		setSize( w, h, pr = pixelRatio ) {

			eng.width = w; eng.height = h;
			renderer.setPixelRatio( pr );
			renderer.setSize( w, h, false );
			camera.aspect = w / h;
			camera.updateProjectionMatrix();

		},

		update() {

			sky.position.copy( camera.position );

		},

		// the lighting half of the look (ambient, moon, lamp) is in the bake; the rest is live
		setRig( name, o = {} ) {

			const SU = sky.userData.U;
			SU.gain.value = 1;
			renderer.toneMappingExposure = o.exposure ?? 1.25;
			P.bloom.value = o.bloom ?? 0.32;
			P.vignette.value = o.vignette ?? 0.35;
			FOG.color.value.setRGB( 0.016, 0.026, 0.058 ).lerp( new THREE.Color( 0.085, 0.07, 0.075 ), o.haze ?? 0 );
			FOG.density.value = o.fog ?? 0.0;
			SU.glowGain.value = o.skyGlow ?? 1.0;
			SU.cover.value = o.cover ?? 0.46;
			SU.cloudGain.value = o.cloudGain ?? 1.0;
			SU.horizon.value.setRGB( 0.018, 0.034, 0.085 );
			SU.cloud.value.setRGB( 0.05, 0.075, 0.14 );
			SU.clouds.value = o.clouds ?? 1.0;
			eng.rig = name;

		},

		setPost() {},

		manual: false,
		tick() {

			if ( eng.manual ) {

				renderer.info.reset();
				renderer._nodes.nodeFrame.update();
				renderer.info.frame = renderer._nodes.nodeFrame.frameId;

			}

		},

		render() {

			eng.tick();
			pipeline.render();

		},

		// offscreen render -> ImageData (sRGB, top row first); same contract as the full engine's capture()
		async capture() {

			const pr = renderer.getPixelRatio();
			const w = Math.round( eng.width * pr ), h = Math.round( eng.height * pr );
			if ( ! isWebGPU ) {

				eng.tick();
				renderer.setRenderTarget( null );
				pipeline.render();
				const gl = renderer.backend.gl;
				const bw = gl.drawingBufferWidth, bh = gl.drawingBufferHeight;
				const raw = new Uint8Array( bw * bh * 4 ), img = new Uint8ClampedArray( bw * bh * 4 );
				gl.bindFramebuffer( gl.FRAMEBUFFER, null );
				gl.readPixels( 0, 0, bw, bh, gl.RGBA, gl.UNSIGNED_BYTE, raw );
				for ( let y = 0; y < bh; y ++ ) img.set( raw.subarray( y * bw * 4, ( y + 1 ) * bw * 4 ), ( bh - 1 - y ) * bw * 4 );
				for ( let i = 3; i < img.length; i += 4 ) img[ i ] = 255;
				return new ImageData( img, bw, bh );

			}

			if ( ! captureRT || captureRT.width !== w || captureRT.height !== h ) {

				captureRT?.dispose();
				captureRT = new THREE.RenderTarget( w, h, { type: THREE.UnsignedByteType, colorSpace: THREE.NoColorSpace, depthBuffer: false } );

			}

			eng.tick();
			renderer.setRenderTarget( captureRT );
			pipeline.render();
			renderer.setRenderTarget( null );
			const px = await renderer.readRenderTargetPixelsAsync( captureRT, 0, 0, w, h );
			const out = new Uint8ClampedArray( w * h * 4 );
			if ( px.length === out.length ) out.set( px );
			else {

				const stride = Math.ceil( w * 4 / 256 ) * 256;
				for ( let y = 0; y < h; y ++ ) out.set( px.subarray( y * stride, y * stride + w * 4 ), y * w * 4 );

			}

			return new ImageData( out, w, h );

		},
	};

	eng.setRig( 'night' );
	return eng;

}

// ---------------------------------------------------------------------------------------- town

const EMPTY16 = new Uint16Array( 0 );

export class BakedTown {

	// wait: 'all' (review captures: every level the view asks for is in before a frame counts) or 'near' (walking:
	// start as soon as the surroundings of the camera are in, the rest streams)
	constructor( eng, bake, { wait = 'near' } = {} ) {

		this.eng = eng;
		this.bake = bake;
		this.T = bake.tier;
		this.wait = wait;
		this.data = bake.data;
		this.nav = new Nav( this.data );
		this.group = new THREE.Group();
		this.group.name = 'town';
		eng.scene.add( this.group );
		this.warnings = [];
		this.bytes = 0; // downloaded so far (manifest excluded)
		this.inflight = new Map(); // chunk level key -> promise
		this.waiters = [];
		this.loTris = bake.chunks.reduce( ( n, m ) => n + m.lo.ni / 3, 0 );
		// the three levels of a cell: lowest, mid, and the tier's own top one (a bake without the phone's ring: none there)
		const L = this.L = [ 'lo', 'mid', this.T.top || 'near' ];
		this.chunks = bake.chunks.map( ( m ) => {

			const b0 = m.lo.bmin, b1 = m.lo.bmax;
			const box = [ b0[ 0 ], b0[ 2 ], b1[ 0 ], b1[ 2 ] ];
			for ( const l of [ 'mid', L[ 2 ] ] ) if ( m[ l ] ) {

				box[ 0 ] = Math.min( box[ 0 ], m[ l ].bmin[ 0 ] ); box[ 1 ] = Math.min( box[ 1 ], m[ l ].bmin[ 2 ] );
				box[ 2 ] = Math.max( box[ 2 ], m[ l ].bmax[ 0 ] ); box[ 3 ] = Math.max( box[ 3 ], m[ l ].bmax[ 2 ] );

			}

			// per level, for the view budget: centre (x, z), radius, triangles and centre height of the sphere the renderer
			// culls it by
			const sph = L.map( ( l ) => m[ l ] && [ ( m[ l ].bmin[ 0 ] + m[ l ].bmax[ 0 ] ) / 2, ( m[ l ].bmin[ 2 ] + m[ l ].bmax[ 2 ] ) / 2,
				0.5 * Math.hypot( m[ l ].bmax[ 0 ] - m[ l ].bmin[ 0 ], m[ l ].bmax[ 1 ] - m[ l ].bmin[ 1 ], m[ l ].bmax[ 2 ] - m[ l ].bmin[ 2 ] ), m[ l ].ni / 3,
				( m[ l ].bmin[ 1 ] + m[ l ].bmax[ 1 ] ) / 2 ] );
			// the clusters of the top level's opaque range (the phone's ring: each a mesh of its own, drawn down to a part
			// size): sphere, box on the ground, [ part size, index count down to it ] steps; rest: the triangles outside them
			const t = m[ L[ 2 ] ];
			const top = ( t && t.parts || [] ).map( ( [ sub, steps, b ] ) => ( {
				sph: [ ( b[ 0 ] + b[ 3 ] ) / 2, ( b[ 2 ] + b[ 5 ] ) / 2, 0.5 * Math.hypot( b[ 3 ] - b[ 0 ], b[ 4 ] - b[ 1 ], b[ 5 ] - b[ 2 ] ), 0, ( b[ 1 ] + b[ 4 ] ) / 2 ],
				box: [ b[ 0 ], b[ 2 ], b[ 3 ], b[ 5 ] ], steps, full: t.sub[ sub ][ 2 ], n: t.sub[ sub ][ 2 ], d: 0 } ) );
			const rest = t ? t.ni / 3 - top.reduce( ( n, q ) => n + q.full / 3, 0 ) : 0;
			return { m, box, sph, top, rest, f: 0, mesh: [ null, null, null ], level: 0, want: 0, d: 0 };

		} );
		this.baseLeft = this.chunks.length;

	}

	_dist( c, x, z ) {

		const dx = Math.max( c.box[ 0 ] - x, 0, x - c.box[ 2 ] ), dz = Math.max( c.box[ 1 ] - z, 0, z - c.box[ 3 ] );
		return Math.hypot( dx, dz );

	}

	async load( url, onProgress = () => {} ) {

		const bake = this.bake, eng = this.eng;
		await MeshoptDecoder.ready;
		onProgress( 0.02, 'textures' );
		const texP = this._loadLayers();
		this._materials( await texP );
		this._sprites();
		// base.bin holds every chunk's lowest level, nearest the spawn first: chunks appear as their bytes arrive
		const sp = this.data.spawn.pos;
		const need = this.wait === 'all' ? Infinity : 90;
		let first = 0;
		this.chunks.forEach( ( c, i ) => {

			if ( this._dist( c, sp[ 0 ], - sp[ 1 ] ) < need ) first = i + 1;

		} );
		let ready;
		const enough = new Promise( ( r ) => ( ready = r ) );
		this.baseDone = this._streamBase( ( n ) => {

			onProgress( 0.1 + 0.9 * Math.min( 1, n / first ), 'town' );
			if ( n >= first ) ready();

		} ).then( ready, ( e ) => {

			this.warnings.push( 'baked town: ' + ( e.message || e ) );
			ready();
			throw e;

		} );
		await enough;
		if ( this.wait === 'all' ) await this.baseDone;
		else this.baseDone.catch( () => {} );
		return this;

	}

	async _streamBase( onChunk ) {

		const bake = this.bake;
		const r = await fetch( `${bake.dir}/${bake.base.file}?v=${bake.baked}` );
		if ( ! r.ok ) throw new Error( `${bake.base.file}: ${r.status}` );
		let buf = new Uint8Array( bake.base.bytes );
		let got = 0, next = 0;
		const flush = () => {

			while ( next < this.chunks.length ) {

				const c = this.chunks[ next ], m = c.m.lo;
				if ( m.off + m.len > got ) break;
				c.mesh[ 0 ] = this._build( buf.subarray( m.off, m.off + m.len ), m, 0 );
				c.mesh[ 0 ].visible = c.level === 0;
				this.group.add( c.mesh[ 0 ] );
				this._dirty = true;
				next ++; this.baseLeft --;
				onChunk( next );

			}

		};

		if ( r.body && r.body.getReader ) {

			const rd = r.body.getReader();
			for ( ;; ) {

				const { done, value } = await rd.read();
				if ( done ) break;
				if ( got + value.length > buf.length ) throw new Error( 'base.bin is larger than the manifest says (stale bake?)' );
				buf.set( value, got ); got += value.length;
				flush();

			}

		} else {

			buf.set( new Uint8Array( await r.arrayBuffer() ) ); got = buf.length;
			flush();

		}

		this.bytes += got;
		if ( next < this.chunks.length ) throw new Error( 'base.bin is shorter than the manifest says (stale bake?)' );
		buf = null;
		this._wake();

	}

	// texture array of pre-shaded albedo layers (rgb) with the relief height in alpha, from the WebP pages
	async _loadLayers() {

		const bake = this.bake, t = bake.tex[ this.T.tex ], size = t.size, n = bake.layers.length, per = t.cols * t.rows;
		const data = new Uint8Array( size * size * 4 * n );
		const page = async ( name ) => {

			const r = await fetch( `${bake.dir}/${name}.webp?v=${bake.baked}` );
			if ( ! r.ok ) throw new Error( `${name}.webp: ${r.status}` );
			const blob = await r.blob();
			this.bytes += blob.size;
			const bmp = await createImageBitmap( blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
			const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( bmp.width, bmp.height ) : Object.assign( document.createElement( 'canvas' ), { width: bmp.width, height: bmp.height } );
			const cx = cv.getContext( '2d', { willReadFrequently: true } );
			cx.globalCompositeOperation = 'copy';
			cx.drawImage( bmp, 0, 0 );
			bmp.close?.();
			return cx;

		};

		await Promise.all( t.pages.map( async ( name, p ) => {

			const cx = await page( name );
			const hx = t.hpages ? await page( t.hpages[ p ] ) : null;
			for ( let i = p * per; i < Math.min( n, ( p + 1 ) * per ); i ++ ) {

				const k = i - p * per, col = k % t.cols, row = ( k - col ) / t.cols;
				const px = cx.getImageData( col * size, row * size, size, size ).data;
				if ( hx ) {

					const hp = hx.getImageData( col * size, row * size, size, size ).data;
					for ( let j = 0; j < px.length; j += 4 ) px[ j + 3 ] = hp[ j ];

				}

				data.set( px, i * size * size * 4 );

			}

		} ) );
		const tex = new THREE.DataArrayTexture( data, size, size, n );
		tex.format = THREE.RGBAFormat; tex.type = THREE.UnsignedByteType;
		tex.colorSpace = THREE.SRGBColorSpace;
		tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
		tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
		tex.generateMipmaps = true;
		tex.anisotropy = this.T.aniso;
		tex.needsUpdate = true;
		this.layers = tex;
		return tex;

	}

	// The K lanterns that light each ground cell most, as a small float texture ( per cell: one texel of thresholds,
	// then position + range and colour x intensity + cap of each ). The diffuse light of every lantern is in the vertex
	// colours; this only feeds the view-dependent part - the glints on wet cobbles, polished stone and the canal -
	// which cannot be baked.
	// A list of the K strongest per cell alone leaves a straight edge in the floor's shine wherever a lantern is in one
	// cell's list and not in the next one's (squares of wet paving with and without a lamp's highlight). So the choice
	// is made a continuous function of the place on the ground: with e the light a lantern sends to a point of the
	// floor, a lantern counts in full where e is over GLINT_FADE x the threshold there and not at all where it is under
	// it. The threshold is interpolated between the cell corners, and a corner's is the largest "strongest light of a
	// lantern left out" of the four cells around it - so a lantern has always faded to nothing by the time it reaches
	// a cell that does not list it, and it fades where it matters least: where K others outshine it.
	// Lanterns of one colour closer together than the tier's `merge` (the heads of one post, the bulbs of one festoon,
	// the lamps of one shop front) share an entry: with few entries per cell, a street of many small lights would
	// otherwise keep only a few of them and lose most of its shine.
	_lightGrid( K ) {

		const src = this.bake.lights || [];
		if ( ! src.length || ! ( K > 0 ) ) return null;
		const lu = ( l ) => 0.2126 * l[ 3 ] + 0.7152 * l[ 4 ] + 0.0722 * l[ 5 ];
		const L = [], taken = new Uint8Array( src.length );
		const merge = this.T.merge ?? GLINT_MERGE;
		const alike = ( a, b ) => { const u = lu( a ) + 1e-6, v = lu( b ) + 1e-6; return Math.max( Math.abs( a[ 3 ] / u - b[ 3 ] / v ), Math.abs( a[ 4 ] / u - b[ 4 ] / v ), Math.abs( a[ 5 ] / u - b[ 5 ] / v ) ) < 0.3; }; // a blue sign keeps its own
		const order = src.map( ( l, i ) => i ).sort( ( a, b ) => lu( src[ b ] ) - lu( src[ a ] ) || a - b );
		for ( const i of order ) {

			if ( taken[ i ] ) continue;
			const a = src[ i ], m = [ 0, 0, 0, 0, 0, 0, 0, 0 ];
			let sum = 0;
			for ( const j of order ) {

				const b = src[ j ];
				if ( taken[ j ] || Math.hypot( a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] ) > merge || ! alike( a, b ) ) continue;
				taken[ j ] = 1;
				const u = lu( b ) + 1e-6;
				sum += u;
				for ( const c of [ 0, 1, 2 ] ) m[ c ] += b[ c ] * u;
				for ( const c of [ 3, 4, 5 ] ) m[ c ] += b[ c ];
				m[ 6 ] = Math.max( m[ 6 ], b[ 6 ] );
				m[ 7 ] += ( b[ 7 ] ?? 1 ) * u; // share of the light a capped lantern sends up

			}

			for ( const c of [ 0, 1, 2, 7 ] ) m[ c ] /= sum;
			L.push( m );

		}

		let x0 = Infinity, z0 = Infinity, x1 = - Infinity, z1 = - Infinity;
		for ( const l of L ) {

			x0 = Math.min( x0, l[ 0 ] - l[ 6 ] ); x1 = Math.max( x1, l[ 0 ] + l[ 6 ] );
			z0 = Math.min( z0, l[ 2 ] - l[ 6 ] ); z1 = Math.max( z1, l[ 2 ] + l[ 6 ] );

		}

		const G = Math.max( 1.25, ( x1 - x0 ) / 320, ( z1 - z0 ) / 320 );
		const nx = Math.max( 1, Math.ceil( ( x1 - x0 ) / G ) ), nz = Math.max( 1, Math.ceil( ( z1 - z0 ) / G ) );
		const lum = L.map( lu );
		const list = new Int32Array( nx * nz * K ).fill( - 1 ), tau = new Float32Array( nx * nz );
		const best = [], row = [];
		for ( let iz = 0; iz < nz; iz ++ ) {

			const cz0 = z0 + iz * G;
			row.length = 0;
			for ( let i = 0; i < L.length; i ++ ) if ( L[ i ][ 2 ] > cz0 - L[ i ][ 6 ] && L[ i ][ 2 ] < cz0 + G + L[ i ][ 6 ] ) row.push( i );
			for ( let ix = 0; ix < nx; ix ++ ) {

				// the most each lantern sends to any point of the cell (GLINT_H2: floor to lantern, squared - the same
				// figure as in _glints, or the two would not agree on where a lantern ends)
				const cx0 = x0 + ix * G;
				best.length = 0;
				for ( const i of row ) {

					const l = L[ i ];
					const dx = Math.max( cx0 - l[ 0 ], 0, l[ 0 ] - cx0 - G ), dz = Math.max( cz0 - l[ 2 ], 0, l[ 2 ] - cz0 - G );
					const d2 = dx * dx + dz * dz + GLINT_H2;
					const w = Math.max( 0, 1 - ( d2 / ( l[ 6 ] * l[ 6 ] ) ) ** 2 );
					const e = lum[ i ] * w * w / d2;
					if ( e > 0 ) best.push( [ e, i ] );

				}

				best.sort( ( a, b ) => b[ 0 ] - a[ 0 ] || a[ 1 ] - b[ 1 ] );
				for ( let k = 0; k < K && k < best.length; k ++ ) list[ ( iz * nx + ix ) * K + k ] = best[ k ][ 1 ];
				tau[ iz * nx + ix ] = best.length > K ? best[ K ][ 0 ] : 0;

			}

		}

		const at = ( ix, iz ) => ix < 0 || iz < 0 || ix >= nx || iz >= nz ? 0 : tau[ iz * nx + ix ];
		const corner = ( cx, cz ) => Math.max( at( cx - 1, cz - 1 ), at( cx, cz - 1 ), at( cx - 1, cz ), at( cx, cz ) );
		const W = 2 * K + 1;
		const data = new Float32Array( nx * W * nz * 4 );
		for ( let iz = 0; iz < nz; iz ++ ) for ( let ix = 0; ix < nx; ix ++ ) {

			let o = ( iz * nx + ix ) * W * 4;
			data.set( [ corner( ix, iz ), corner( ix + 1, iz ), corner( ix, iz + 1 ), corner( ix + 1, iz + 1 ) ], o );
			for ( let k = 0; k < K; k ++ ) {

				const i = list[ ( iz * nx + ix ) * K + k ], l = i >= 0 ? L[ i ] : null;
				o += 8;
				if ( l ) data.set( [ l[ 0 ], l[ 1 ], l[ 2 ], l[ 6 ], l[ 3 ], l[ 4 ], l[ 5 ], l[ 7 ] ], o - 4 );
				else data.set( [ 0, - 1e4, 0, 1, 0, 0, 0, 1 ], o - 4 );

			}

		}

		const tex = new THREE.DataTexture( data, nx * W, nz, THREE.RGBAFormat, THREE.FloatType );
		tex.minFilter = tex.magFilter = THREE.NearestFilter;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		return { tex, K, G, x0, z0, nx, nz, W, lights: L.length };

	}

	// specular radiance of the cell's lanterns (and a little of the moon) leaving a surface with normal N towards V
	_glints( N, V, rough, f0 ) {

		const g = this.grid, look = this.bake.look;
		const a = rough.mul( rough ), a2 = a.mul( a );
		const brdf = ( Lw ) => {

			const H = normalize( Lw.add( V ) );
			const nh = saturate( dot( N, H ) ), nl = saturate( dot( N, Lw ) ), lh = saturate( dot( Lw, H ) );
			const den = nh.mul( nh ).mul( a2.sub( 1 ) ).add( 1 );
			const F = oneMinus( lh ).pow( 5 ).mul( oneMinus( f0 ) ).add( f0 );
			return a2.div( den.mul( den ).mul( max( lh.mul( lh ), 0.1 ) ).mul( a.add( 0.5 ) ).mul( 4 * Math.PI ) ).mul( nl ).mul( F );

		};

		const gp = positionWorld.xz.sub( vec2( g.x0, g.z0 ) ).div( g.G );
		const c = clamp( floor( gp ), vec2( 0, 0 ), vec2( g.nx - 1, g.nz - 1 ) );
		const cf = saturate( gp.sub( c ) ); // place within the cell: the threshold is interpolated between its corners
		const ix = int( c.x ).mul( g.W ), iz = int( c.y );
		const T = textureLoad( g.tex, ivec2( ix, iz ) );
		const tau = mix( mix( T.x, T.y, cf.x ), mix( T.z, T.w, cf.x ), cf.y );
		const band = max( tau.mul( GLINT_FADE - 1 ), 1e-6 );
		let sum = vec3( 0 );
		for ( let k = 0; k < g.K; k ++ ) {

			const A = textureLoad( g.tex, ivec2( ix.add( 1 + 2 * k ), iz ) ), B = textureLoad( g.tex, ivec2( ix.add( 2 + 2 * k ), iz ) );
			const Lv = A.xyz.sub( positionWorld );
			// how much this lantern counts here (see _lightGrid): by its light on the floor at this place
			const h2 = dot( Lv.xz, Lv.xz ).add( GLINT_H2 ), hq = h2.div( A.w.mul( A.w ) ), hw = saturate( oneMinus( hq.mul( hq ) ) );
			const e = dot( B.rgb, vec3( 0.2126, 0.7152, 0.0722 ) ).mul( hw ).mul( hw ).div( h2 );
			const win = smoothstep( 0, 1, e.sub( tau ).div( band ) );
			const d2 = max( dot( Lv, Lv ), 0.01 ), d = sqrt( d2 );
			const q = d.div( A.w ), w = saturate( oneMinus( q.mul( q ).mul( q ).mul( q ) ) );
			const cap = mix( float( 1 ), B.w, smoothstep( - 0.42, 0.3, Lv.y.negate().div( d ) ) ); // capFactor of ClusteredLightsNode.js
			sum = sum.add( B.rgb.mul( w.mul( w ).div( d2 ) ).mul( cap ).mul( win ).mul( brdf( Lv.div( d ) ) ) );

		}

		const m = look.moon_col || [ 0.6, 0.7, 1 ], mk = ( look.moon ?? 1 ) * 0.5;
		return sum.add( vec3( m[ 0 ] * mk, m[ 1 ] * mk, m[ 2 ] * mk ).mul( brdf( vec3( this.eng.moonDir.x, this.eng.moonDir.y, this.eng.moonDir.z ) ) ) );

	}

	_materials( tex ) {

		const eng = this.eng, bake = this.bake;
		const pl = attribute( '_pl', 'vec4' ), cv = attribute( '_col', 'vec4' );
		const layer = varying( pl.w.mul( 65535 ), 'vLayer' );
		const tiles = varying( uv().mul( 65535 / 65536 ).sub( 0.5 ).mul( 2 * bake.uv_range ), 'vTiles' );
		const lit = varying( cv.rgb.mul( cv.rgb ).mul( bake.colour_scale ), 'vLit' ); // vertex colour: all light, already times the surface tint
		const aux = varying( cv.w, 'vAux' );
		const fogged = ( c ) => mix( c, eng.FOG.color, eng.fogAmount() );
		const tx = texture( tex, tiles ).depth( int( layer.add( 0.5 ) ) );
		const V = normalize( cameraPosition.sub( positionWorld ) );
		const SKY = vec3( 0.018, 0.034, 0.085 );
		this.grid = this._lightGrid( this.T.glints );
		const U = this.U = { relief: uniform( 0.05 ), glint: uniform( 1.5 ), sheen: uniform( 1.0 ), town: uniform( 0.4 ), swell: uniform( 0.3 ), wglow: uniform( 4 ) };
		for ( const k in U ) { const v = parseFloat( new URLSearchParams( location.search ).get( 'u_' + k ) ); if ( isFinite( v ) ) U[ k ].value = v; } // ?u_glint=2 (tuning)

		// A lit card sits millimetres from its wall, and the lowest level is simplified well past that: there the page
		// draws the glowing faces and the glass a little towards the eye (a metres at 1 m, less with distance: constant
		// on screen; + k of the distance), so they never share depth with the wall behind. Per level [ a, k ]; ?pull=lo,mid,top
		// and ?pullk= override. The levels above keep the models' own clearance (webbake_v1_pack.mjs *_glow_guard).
		const PULL = this.PULL = { 0: [ 0.06, 0.0005 ], 1: [ 0, 0 ], 2: [ 0, 0 ] };
		{

			const q = new URLSearchParams( location.search ), ks = [ 0, 1, 2 ];
			[ 'pull', 'pullk' ].forEach( ( name, j ) => ( q.get( name ) || '' ).split( ',' ).forEach( ( t, i ) => { const v = parseFloat( t ); if ( isFinite( v ) && i < ks.length ) PULL[ ks[ i ] ][ j ] = v; } ) );

		}

		const pullObj = uniform( new THREE.Vector3() ).onObjectUpdate( ( { object }, self ) => {

			const p = object.userData.pull;
			self.value.set( p ? p[ 0 ] : 0, p ? p[ 1 ] : 0, object.matrixWorld.elements[ 0 ] ); // (.z: metres per unit of the mesh)

		} );
		const pulled = ( which ) => {

			const to = modelWorldMatrixInverse.mul( vec4( cameraPosition, 1 ) ).xyz.sub( positionLocal );
			const d = length( to ).mul( pullObj.z );
			const t = min( pullObj.x.div( max( d, 1e-3 ) ).add( pullObj.y ), 0.5 );
			return positionLocal.add( to.mul( which === null ? t : t.mul( which ) ) );

		};

		const opaque = new THREE.MeshBasicNodeMaterial( { fog: false } );
		const glowing = step( 0.75, aux );
		let col = tx.rgb.mul( lit );
		if ( this.grid ) {

			// wet cobbles, polished stone, glazed tiles: the face normal (from the position derivatives), roughened by the
			// relief in the layer's alpha, mirrors the nearest lanterns and a little of the sky. Where the baked light says
			// "in shadow" the highlight goes too.
			const dpx = dFdx( positionWorld ), dpy = dFdy( positionWorld );
			const nr = cross( dpx, dpy );
			const N0 = nr.div( max( length( nr ), 1e-20 ) );
			const R1 = cross( dpy, N0 ), R2 = cross( N0, dpx ), det = dot( dpx, R1 );
			const h = tx.a.mul( U.relief );
			const nb = N0.mul( det ).mul( sign( dot( N0, V ) ) ).sub( R1.mul( dFdx( h ) ).add( R2.mul( dFdy( h ) ) ) );
			const N = nb.div( max( length( nb ), 1e-20 ) );
			const gloss = min( aux.mul( 2 ), 1 );
			const rough = mix( float( 0.5 ), float( 0.08 ), gloss );
			const lum = dot( lit, vec3( 0.2126, 0.7152, 0.0722 ) );
			const gate = smoothstep( 0.03, 0.3, gloss ).mul( oneMinus( glowing ) ).mul( smoothstep( 0.008, 0.08, lum ) );
			// what the wet film mirrors besides the lanterns: the sky where the reflected ray points up, the lit town
			// (about as bright as the light arriving here) where it stays low
			const nv = saturate( dot( N, V ) );
			const fres = pow( oneMinus( nv ), 5 ).mul( 0.96 ).add( 0.04 );
			const ry = N.y.mul( nv ).mul( 2 ).sub( V.y );
			const env = mix( lit.mul( U.town ), SKY.mul( U.sheen ), smoothstep( 0.02, 0.45, ry ) );
			col = col.add( this._glints( N, V, rough, float( 0.04 ) ).mul( U.glint ).add( env.mul( fres ).mul( gloss ) ).mul( gate ) );

		}

		opaque.colorNode = eng.knee( fogged( col ), glowing );
		opaque.positionNode = pulled( step( 0.75, cv.w ) );
		opaque.name = 'baked';

		const glass = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false } );
		glass.colorNode = fogged( lit );
		glass.opacityNode = aux;
		glass.positionNode = pulled( null );
		glass.name = 'baked glass';

		// canal / basin water at night: a dark body that mirrors. Steep reflected rays find the navy sky, low ones the
		// town (dark, warm); the lanterns of the cell show twice - as the sharp glint of the flame, torn into glitter by
		// the ripples, and as the broad warm patch of the wall they light. One slow swell and a finer ripple scroll
		// across it. No pass, no target: three noise taps and the light grid the wet ground already reads.
		const water = new THREE.MeshBasicNodeMaterial( { fog: false } );
		{

			const t = time.mul( 0.05 );
			const wp = positionWorld.xz;
			const n1 = texture( eng.noise, wp.mul( 0.33 ).add( vec2( t, t.mul( 0.6 ) ) ) ).rg.sub( 0.5 );
			const n2 = texture( eng.noise, wp.mul( 1.1 ).add( vec2( t.mul( - 0.7 ), t.mul( 1.1 ) ) ) ).ba.sub( 0.5 );
			const n3 = texture( eng.noise, wp.mul( 4.3 ).add( vec2( t.mul( 2.1 ), t.mul( - 1.6 ) ) ) ).rg.sub( 0.5 );
			const swell = n1.add( n2.mul( 0.45 ) ).mul( U.swell );
			const rip = swell.add( n3.mul( U.swell ).mul( 0.6 ) );
			const Ns = normalize( vec3( swell.x, 1, swell.y ) ), N = normalize( vec3( rip.x, 1, rip.y ) );
			const nv = saturate( dot( N, V ) );
			const fres = pow( oneMinus( nv ), 5 ).mul( 0.98 ).add( 0.02 );
			const ry = N.y.mul( nv ).mul( 2 ).sub( V.y ); // height of the reflected ray
			const env = mix( vec3( 0.028, 0.019, 0.012 ).mul( U.town ).mul( 2.5 ), SKY.mul( U.sheen ), smoothstep( 0.1, 0.5, ry ) );
			let c = lit.mul( 0.3 ).add( env.mul( fres ) );
			if ( this.grid ) c = c.add( this._glints( N, V, float( 0.07 ), float( 0.02 ) ).mul( U.glint ) ).add( this._glints( Ns, V, float( 0.4 ), float( 0.02 ) ).mul( U.wglow ) );
			water.colorNode = fogged( c );

		}

		water.name = 'baked water';

		// falling water: the engine's torn sheet (strands scrolling down V), lit by the baked colour
		const fall = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false, side: THREE.DoubleSide } );
		{

			const t = time.mul( 1.6 );
			const u = tiles.x, v = tiles.y.negate();
			const s1 = texture( eng.noise, vec2( u.mul( 7 ), v.mul( 0.5 ).add( t.mul( 0.6 ) ) ) );
			const s2 = texture( eng.noise, vec2( u.mul( 19 ), v.mul( 1.2 ).add( t ) ) );
			const body = s1.r.mul( 0.5 ).add( s2.g.mul( 0.32 ) ).add( s1.b.mul( 0.18 ) );
			const a = smoothstep( 0.42, 0.58, body );
			const rim = a.mul( oneMinus( smoothstep( 0.52, 0.72, body ) ) );
			const foam = rim.mul( 0.75 ).add( 0.2 ).saturate();
			const bead = smoothstep( 0.78, 0.9, s2.b.mul( 0.6 ).add( s1.a.mul( 0.4 ) ).add( foam.mul( 0.12 ) ) ).mul( a );
			fall.colorNode = fogged( lit.mul( mix( float( 0.22 ), float( 0.8 ), foam ).add( bead.mul( 1.5 ) ) ) );
			fall.opacityNode = a.mul( mix( float( 0.4 ), float( 1 ), max( foam, bead ) ) ).mul( aux );

		}

		fall.name = 'baked fall';

		// leaf cards (the trees' foliage): the uv counts cards, aux scales it to texture tiles. As in the engine's foliage
		// material the relief height of the leaf mosaic cuts the card into leaves, and the cut rises towards the card's
		// edge - a ragged spray, never a straight edge.
		const cut = new THREE.MeshBasicNodeMaterial( { fog: false, side: THREE.DoubleSide } );
		{

			const t = texture( tex, tiles.mul( aux.mul( bake.cut_aux || 4 ) ) ).depth( int( layer.add( 0.5 ) ) );
			const cell = fract( tiles ).sub( 0.5 ).mul( 2 );
			const h = t.a.mul( 2 ).sub( 1 ); // the layer keeps the height in the upper half of alpha
			cut.maskNode = h.greaterThan( smoothstep( 0.45, 1.0, length( cell ) ).mul( 0.8 ).add( 0.33 ) );
			cut.colorNode = eng.knee( fogged( t.rgb.mul( lit ) ), float( 0 ) );

		}

		cut.name = 'baked leaves';
		this.mats = { [ K_OPAQUE ]: opaque, [ K_GLASS ]: glass, [ K_WATER ]: water, [ K_FALL ]: fall, [ K_CUT ]: cut };

	}

	// lantern glows and flames: one instanced sprite draw each
	_sprites() {

		const eng = this.eng, bake = this.bake;
		const inst = ( rows, from, n ) => {

			const a = new Float32Array( rows.length * n );
			rows.forEach( ( r, i ) => {

				for ( let k = 0; k < n; k ++ ) a[ i * n + k ] = r[ from + k ];

			} );
			return instancedBufferAttribute( new THREE.InstancedBufferAttribute( a, n ) );

		};

		if ( bake.halos.length ) {

			const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
			const H = bake.halos;
			m.positionNode = inst( H, 0, 3 );
			m.scaleNode = inst( H, 6, 1 );
			const r = length( uv().sub( 0.5 ) ).mul( 2 );
			const a = exp( r.mul( r ).mul( - 5 ) ).sub( 0.0067 ).max( 0 );
			const hs = varying( inst( H, 6, 1 ) );
			// same hot core as engine.js createHalo; row[ 7 ] = 0 marks a helper light with no burner to look at
			const core = exp( r.mul( r ).mul( hs.mul( hs ).mul( - 11 ) ) ).mul( H[ 0 ].length > 7 ? varying( inst( H, 7, 1 ) ) : 1 );
			m.colorNode = varying( inst( H, 3, 3 ) ).mul( a.add( core.mul( haloCore ) ) ).mul( this.T.halo || 0.1 );
			m.opacityNode = a.add( core.mul( haloCore.min( 1 ) ) ).min( 1 );
			const s = new THREE.Sprite( m );
			s.count = H.length;
			s.frustumCulled = false;
			s.renderOrder = 4;
			s.name = 'halos';
			this.group.add( s );
			this.halos = s;

		}

		if ( bake.flames.length ) {

			const F = bake.flames.map( ( f, i ) => [ ...f, ( i * 0.61803 ) % 1, f[ 6 ] * 0.8, f[ 6 ] * 1.7 ] ); // + phase, sprite size
			const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
			m.positionNode = inst( F, 0, 3 );
			m.scaleNode = inst( F, 10, 2 );
			const col = varying( inst( F, 3, 3 ) ), gs = varying( inst( F, 7, 3 ) ); // glow, speed, phase
			const glow = gs.x, phase = gs.z;
			const p = uv();
			const y = p.y, x = p.x.sub( 0.5 ).mul( 2 );
			const t = time.mul( gs.y ).add( phase.mul( 7.3 ) );
			const n1 = texture( eng.noise, vec2( x.mul( 0.12 ).add( phase ), y.mul( 0.3 ).sub( t.mul( 0.5 ) ) ) ).r.sub( 0.5 );
			const n2 = texture( eng.noise, vec2( x.mul( 0.35 ).add( phase.mul( 2.1 ) ), y.mul( 0.8 ).sub( t.mul( 1.05 ) ) ) ).g.sub( 0.5 );
			const sway = sin( t.mul( 2.1 ) ).mul( 0.06 ).add( n1.mul( 0.55 ) ).add( n2.mul( 0.22 ) );
			const xx = x.add( sway.mul( y ) );
			const hgt = y.div( n1.mul( 0.25 ).add( 0.9 ) );
			const w = min( pow( max( hgt, 0 ).div( 0.22 ), 0.55 ), pow( max( oneMinus( hgt ), 0 ).div( 0.78 ), 0.85 ) ).mul( 0.42 );
			const dd = abs( xx ).div( max( w, 0.001 ) );
			const body = smoothstep( 1.0, 0.25, dd ).mul( step( hgt, 1 ) );
			const core = smoothstep( 0.7, 0.0, dd ).mul( smoothstep( 0.95, 0.25, hgt ) );
			const outer = col.mul( vec3( 1, 0.55, 0.25 ) );
			let c = mix( outer, vec3( 1.0, 0.86, 0.6 ), core );
			c = mix( vec3( 0.06, 0.125, 0.45 ), c, smoothstep( 0.02, 0.2, hgt ) );
			m.colorNode = c.mul( body ).mul( glow );
			m.opacityNode = body;
			const s = new THREE.Sprite( m );
			s.count = F.length;
			s.center.set( 0.5, 0.02 );
			s.frustumCulled = false;
			s.renderOrder = 5;
			s.name = 'flames';
			this.group.add( s );
			this.flames = s;

		}

	}

	// one chunk level: meshopt streams -> a mesh drawing straight from the quantized vertex records
	_build( bytes, m, level = 0 ) {

		const vb = new Uint8Array( m.nv * 20 );
		MeshoptDecoder.decodeVertexBuffer( vb, m.nv, 20, bytes.subarray( 0, m.vlen ) );
		const ib = new Uint8Array( m.ni * m.it );
		MeshoptDecoder.decodeIndexBuffer( ib, m.ni, m.it, bytes.subarray( m.vlen ) );
		const buf = new THREE.InterleavedBuffer( new Uint16Array( vb.buffer ), 10 );
		const g = new THREE.BufferGeometry();
		g.setAttribute( 'position', new THREE.InterleavedBufferAttribute( buf, 4, 0, true ) );
		g.setAttribute( '_pl', new THREE.InterleavedBufferAttribute( buf, 4, 0, true ) ); // .w = texture layer
		g.setAttribute( 'uv', new THREE.InterleavedBufferAttribute( buf, 2, 4, true ) );
		g.setAttribute( '_col', new THREE.InterleavedBufferAttribute( buf, 4, 6, true ) );
		// (a 16-bit index is marked normalized: the WebGPU backend widens plain 16-bit integer attributes to 32 bits when
		// it uploads them, and for an index that is GPU memory for nothing; as it is, it goes up and is drawn as 16-bit)
		g.setIndex( new THREE.BufferAttribute( m.it === 2 ? new Uint16Array( ib.buffer ) : new Uint32Array( ib.buffer ), 1, m.it === 2 ) );
		const s = m.q * 65535;
		g.boundingBox = new THREE.Box3(
			new THREE.Vector3( ( m.bmin[ 0 ] - m.min[ 0 ] ) / s, ( m.bmin[ 1 ] - m.min[ 1 ] ) / s, ( m.bmin[ 2 ] - m.min[ 2 ] ) / s ),
			new THREE.Vector3( ( m.bmax[ 0 ] - m.min[ 0 ] ) / s, ( m.bmax[ 1 ] - m.min[ 1 ] ) / s, ( m.bmax[ 2 ] - m.min[ 2 ] ) / s ) );
		g.boundingSphere = g.boundingBox.getBoundingSphere( new THREE.Sphere() );
		const place = ( o ) => {

			o.position.fromArray( m.min );
			o.scale.setScalar( s );
			o.matrixAutoUpdate = false;
			o.updateMatrix();
			o.userData.tris = m.ni / 3;
			return o;

		};

		// the arrays are only needed until the first draw has put them on the GPU
		// (the index keeps its array type: the back-ends read the index format from it)
		const meshes = [];
		const free = () => {

			buf.array = EMPTY16;
			g.index.array = new g.index.array.constructor( 0 );
			for ( const o of meshes ) o.onAfterRender = THREE.Object3D.prototype.onAfterRender;

		};

		const mats = [], own = new Map( ( m.parts || [] ).map( ( q, i ) => [ q[ 0 ], i ] ) );
		const parts = [];
		m.sub.forEach( ( [ kind, start, count ], i ) => {

			if ( ! own.has( i ) ) { g.addGroup( start, count, mats.length ); mats.push( this.mats[ kind ] ); return; }
			// a cluster of the opaque range: the same buffers, its own range, bounds and so its own culling
			const b = m.parts[ own.get( i ) ][ 2 ], q = new THREE.BufferGeometry();
			for ( const name in g.attributes ) q.setAttribute( name, g.attributes[ name ] );
			q.setIndex( g.index );
			q.setDrawRange( start, count );
			q.boundingBox = new THREE.Box3(
				new THREE.Vector3( ( b[ 0 ] - m.min[ 0 ] ) / s, ( b[ 1 ] - m.min[ 1 ] ) / s, ( b[ 2 ] - m.min[ 2 ] ) / s ),
				new THREE.Vector3( ( b[ 3 ] - m.min[ 0 ] ) / s, ( b[ 4 ] - m.min[ 1 ] ) / s, ( b[ 5 ] - m.min[ 2 ] ) / s ) );
			q.boundingSphere = q.boundingBox.getBoundingSphere( new THREE.Sphere() );
			const part = new THREE.Mesh( q, this.mats[ kind ] );
			part.matrixAutoUpdate = false;
			part.userData.start = start;
			parts.push( part );

		} );
		let mesh = mats.length ? new THREE.Mesh( g, mats.length === 1 && ! parts.length ? mats[ 0 ] : mats ) : null; // (one material: the groups would not count)
		if ( parts.length ) {

			const all = new THREE.Group();
			if ( mesh ) { mesh.matrixAutoUpdate = false; all.add( mesh ); }
			all.add( ...parts );
			all.userData.parts = parts;
			meshes.push( ...all.children );
			mesh = all;

		} else meshes.push( mesh );

		for ( const o of meshes ) { o.onAfterRender = free; o.userData.pull = this.PULL[ level ] || null; }
		return place( mesh );

	}

	_drop( c, l ) {

		const mesh = c.mesh[ l ];
		if ( ! mesh ) return;
		this.group.remove( mesh );
		// three's WebGL back-end (r186) never deletes a vertex array object, and a deleted buffer lives on for as long as one
		// still points at it: a walk would leave every chunk it ever showed on the GPU. So the ones over this mesh's buffers
		// are deleted here.
		const b = this.eng.renderer.backend, gl = b.isWebGLBackend && b.vaoCache ? b.gl : null, ids = new Set();
		if ( gl ) mesh.traverse( ( o ) => {

			if ( ! o.geometry ) return;
			for ( const k in o.geometry.attributes ) { const id = b.get( o.geometry.attributes[ k ] ).id; if ( id !== undefined ) ids.add( String( id ) ); }

		} );
		mesh.traverse( ( o ) => o.geometry && o.geometry.dispose() );
		if ( ids.size ) for ( const key of Object.keys( b.vaoCache ) ) {

			if ( ! key.split( ':' ).some( ( id ) => ids.has( id ) ) ) continue;
			const vao = b.vaoCache[ key ];
			if ( b.state.currentVAO === vao ) b.state.resetVertexState();
			gl.deleteVertexArray( vao );
			delete b.vaoCache[ key ];

		}
		c.mesh[ l ] = null;

	}

	_wake() {

		const w = this.waiters; this.waiters = [];
		for ( const f of w ) f();

	}

	async _fetch( c, l ) {

		const m = c.m[ this.L[ l ] ], key = m.file;
		try {

			const r = await fetch( `${this.bake.dir}/${m.file}?v=${this.bake.baked}` );
			if ( ! r.ok ) throw new Error( `${m.file}: ${r.status}` );
			const bytes = new Uint8Array( await r.arrayBuffer() );
			this.bytes += bytes.length;
			if ( bytes.length !== m.len ) throw new Error( `${m.file}: size differs from the manifest (stale bake?)` );
			if ( c.want === l && ! c.mesh[ l ] ) {

				c.mesh[ l ] = this._build( bytes, m, l );
				c.mesh[ l ].visible = false;
				this.group.add( c.mesh[ l ] );

			}

		} catch ( e ) {

			c.failed = ( c.failed || 0 ) | ( 1 << l );
			const w = 'baked town: ' + ( e.message || e );
			if ( ! this.warnings.includes( w ) ) this.warnings.push( w );

		} finally {

			this.inflight.delete( key );
			this._dirty = true;
			this._wake();

		}

	}

	// the triangle budget on everything shown, nearest first: the mid level for every cell that asks for more than
	// lo, then near for those that asked for it (one near cell costs several mid ones - coverage comes before detail)
	_budget( order ) {

		const T = this.T, L = this.L;
		let tris = this.loTris;
		for ( const c of order ) {

			const n = c.m.mid ? ( c.m.mid.ni - c.m.lo.ni ) / 3 : 0;
			c.cost = 0;
			if ( c.m.mid && tris + n <= T.tris ) { tris += n; c.cost = c.m.mid.ni / 3; }
			else if ( c.m.mid || c.want < 2 ) c.want = 0;

		}

		for ( const c of order ) {

			if ( c.want < 2 ) continue;
			const n = c.m[ L[ 2 ] ].ni / 3 - ( c.cost || c.m.lo.ni / 3 );
			if ( tris + n <= T.tris ) tris += n;
			else c.want = c.cost > 0 ? 1 : 0;

		}

	}

	// index count of a cluster drawn down to parts of size s (m): its range is in order of part size, with the count
	// down to each size in the manifest
	_cut( steps, s ) {

		for ( const [ size, n ] of steps ) if ( size <= s ) return n;
		return steps[ steps.length - 1 ][ 1 ];

	}

	// the triangle budget on what a view can hold: whichever way a camera here looks, what the renderer's culling lets
	// through may not add up to more than T.tris. The sum is kept per view: 64 headings x the pitches of VIEW_PITCH
	// (level, looking up - which brings the tall things at the sides into view - and looking down). A mesh is in a
	// view when its sphere passes the side planes (heading within its bearing +- the half field of view at that pitch
	// + the angle the sphere takes up, or around the back for one overhead), the near and the upper plane and, looking
	// up or down past the edge of the picture, the plane that then cuts off what is straight ahead; each arc is half a heading
	// step wider.
	// The top level takes the room first, nearest cell first, each of its clusters drawn down to the parts that still
	// show at its distance (T.detail: size / distance, about 3 pixels), then mid for the rest. Where a top cell does
	// not fit, or mid would not reach T.reach, all top cells leave out larger parts (the next step of the ladder, up
	// to 0.3 m); past the last step the farthest top cells fall back to mid until mid reaches. What the tightest
	// view cost the cells it does not look at is then given back: nearest first, each top cell as fine as it now
	// fits. Anything a cell did not have the last time needs 6 % to spare, so nothing flickers at the edge of the
	// budget. (tools/perf_sim.py runs the same rule over the whole town.)
	_viewBudget( camera, order ) {

		const T = this.T, H = 64, NP = VIEW_PITCH.length, S = H * NP, x = camera.position.x, y = camera.position.y, z = camera.position.z;
		const step = 2 * Math.PI / H, hs = step / 2, th = Math.tan( camera.fov * Math.PI / 360 ), tw = th * camera.aspect;
		const sw = Math.sqrt( 1 + tw * tw ), sh = Math.sqrt( 1 + th * th );
		const { load, base, tmp, B } = this._heads || ( this._heads = { load: new Float64Array( S ), base: new Float64Array( S ), tmp: new Float32Array( S ), B: new Float32Array( H ) } );
		// the views (of S) that have a sphere [ x, z, radius, , height ] in them
		const arc = ( s, out ) => {

			const dx = s[ 0 ] - x, dz = s[ 1 ] - z, d = Math.max( Math.hypot( dx, dz ), 1e-3 ), r = s[ 2 ], a = Math.atan2( dx, dz );
			for ( let h = 0; h < H; h ++ ) {

				const b = Math.abs( h * step - a ) % ( 2 * Math.PI );
				B[ h ] = b > Math.PI ? 2 * Math.PI - b : b; // how far heading h is off the bearing of the sphere

			}

			for ( let k = 0; k < NP; k ++ ) {

				// (looking down is looking up with the heights mirrored)
				const p = Math.abs( VIEW_PITCH[ k ] ), sp = Math.sin( p ), cp = Math.cos( p ), dy = VIEW_PITCH[ k ] < 0 ? y - s[ 4 ] : s[ 4 ] - y;
				// side planes: d ( sin b - A cos b ) <= r sw + tw dy sp, A = tw cos p
				const A = tw * cp, phi = Math.atan( A ), rho = ( r * sw + tw * dy * sp ) / ( d * Math.sqrt( 1 + A * A ) );
				let w1 = 4, w2 = 0; // in view up to w1 off the bearing, and again from w2 on (around the back)
				if ( rho <= - 1 ) { w1 = - 1; w2 = 4; } else if ( rho < 1 ) { const t = Math.asin( rho ); w1 = phi + t; w2 = Math.PI + phi - t; }
				// near plane: d cos b cp + dy sp >= - r
				const cn = - ( r + dy * sp ) / ( d * cp ), wn = cn >= 1 ? - 1 : cn <= - 1 ? 4 : Math.acos( cn );
				// upper plane: d cos b ( sp + th cp ) >= dy ( cp - th sp ) - r sh
				const ct = ( dy * ( cp - th * sp ) - r * sh ) / ( d * ( sp + th * cp ) ), wt = ct >= 1 ? - 1 : ct <= - 1 ? 4 : Math.acos( ct );
				// the lower plane, once it is above the horizon: only what is far enough off to the side stays above it
				const q = sp - th * cp, cl = q > 0 ? ( dy * ( cp + th * sp ) + r * sh ) / ( d * q ) : 1, wl = cl >= 1 ? 0 : cl <= - 1 ? 4 : Math.acos( cl );
				for ( let h = 0, o = k * H; h < H; h ++ ) {

					const b = B[ h ];
					out[ o + h ] = ( b <= w1 + hs || b >= w2 - hs ) && b <= wn + hs && b <= wt + hs && b >= wl - hs ? 1 : 0;

				}

			}

			return out;

		};

		// per cell: lo (what each view sees of its lowest level), cv (... of the level it has in this plan)
		base.fill( 0 );
		for ( const c of this.chunks ) {

			if ( ! c.arc ) Object.assign( c, { arc: [ 0, 1, 2 ].map( () => new Uint8Array( S ) ), lo: new Float32Array( S ), cv: new Float32Array( S ) } );
			const A = arc( c.sph[ 0 ], c.arc[ 0 ] ), n = c.sph[ 0 ][ 3 ];
			for ( let h = 0; h < S; h ++ ) base[ h ] += c.lo[ h ] = A[ h ] ? n : 0;

		}

		for ( const c of order ) {

			c.ask = c.want; c.f0 = c.f;
			if ( c.m.mid ) arc( c.sph[ 1 ], c.arc[ 1 ] );
			if ( c.ask < 2 ) continue;
			arc( c.sph[ 2 ], c.arc[ 2 ] );
			for ( const q of c.top ) {

				arc( q.sph, q.arc || ( q.arc = new Uint8Array( S ) ) );
				q.d = Math.hypot( Math.max( q.box[ 0 ] - x, 0, x - q.box[ 2 ] ), Math.max( q.box[ 1 ] - z, 0, z - q.box[ 3 ] ) );

			}

		}

		// what the views see of a cell becomes nw, if all that see more of it have the room
		const put = ( c, nw, cap ) => {

			const cv = c.cv;
			for ( let h = 0; h < S; h ++ ) if ( nw[ h ] > cv[ h ] && load[ h ] + nw[ h ] - cv[ h ] > cap ) return false;
			for ( let h = 0; h < S; h ++ ) { load[ h ] += nw[ h ] - cv[ h ]; cv[ h ] = nw[ h ]; }
			return true;

		};

		const K = T.detail || 0.0055, reach = Math.min( T.mid, T.reach || T.mid ), F = [ 1, 1.6, 2.5, 4, 6.3, 10, 16 ];
		const full = T.tris, spare = T.tris * 0.94;
		const mid = ( c ) => {

			const A = c.arc[ 1 ], n = c.sph[ 1 ][ 3 ];
			for ( let h = 0; h < S; h ++ ) tmp[ h ] = A[ h ] ? n : 0;
			return put( c, tmp, c.was < 1 ? spare : full );

		};

		// the top level of a cell at step f of the ladder
		const top = ( c, f ) => {

			const A = c.arc[ 2 ], n = c.rest;
			for ( let h = 0; h < S; h ++ ) tmp[ h ] = A[ h ] ? n : 0;
			for ( const q of c.top ) {

				const n = ( q.t = this._cut( q.steps, Math.min( 0.3, q.d * K * f ) ) ) / 3, A = q.arc;
				for ( let h = 0; h < S; h ++ ) if ( A[ h ] ) tmp[ h ] += n;

			}

			if ( ! put( c, tmp, c.was < 2 || f < c.f0 ? spare : full ) ) return false;
			for ( const q of c.top ) q.n = q.t;
			c.f = f;
			return true;

		};

		const far = order.filter( ( c ) => c.ask === 2 && c.d > 0 ).reverse();
		let refused, gap;
		const solve = ( f, drop ) => {

			load.set( base );
			for ( const c of this.chunks ) c.cv.set( c.lo );
			for ( const c of order ) c.want = c.ask;
			for ( let i = 0; i < drop; i ++ ) far[ i ].want = 1;
			refused = 0; gap = Infinity;
			for ( const c of order ) if ( c.want === 2 && ! top( c, f ) ) { c.want = 1; refused ++; }
			for ( const c of order ) if ( c.want === 1 && ! ( c.m.mid && mid( c ) ) ) {

				c.want = 0;
				if ( c.d < gap ) gap = c.d;

			}

			return refused === 0 && gap >= reach;

		};

		let f = F[ 0 ];
		for ( const g of F ) { f = g; if ( solve( g, 0 ) ) break; }
		for ( let drop = 1; drop <= far.length && gap < reach; drop ++ ) solve( f, drop );
		for ( const c of order ) {

			if ( c.ask !== 2 ) continue;
			const now = c.want === 2 ? c.f : Infinity;
			for ( const g of F ) {

				if ( g >= now ) break;
				if ( top( c, g ) ) { c.want = 2; break; }

			}

		}

		this._f = f;

	}

	// choose the level of every cell for this camera, start / drop downloads, swap meshes
	update( camera, now, { force = false } = {} ) {

		const T = this.T, L = this.L, x = camera.position.x, z = camera.position.z;
		if ( ! force && ! this._dirty && this._at && Math.hypot( x - this._at[ 0 ], z - this._at[ 1 ] ) < 0.75 && camera.aspect === this._at[ 2 ] ) return;
		this._at = [ x, z, camera.aspect ]; this._dirty = false;
		const order = [];
		for ( const c of this.chunks ) {

			c.d = this._dist( c, x, z );
			const keep = ( r, l ) => c.d < ( c.want >= l ? r + Math.min( r * 0.2 + 2, T.hold || Infinity ) : r );
			let want = 0;
			if ( c.m[ L[ 2 ] ] && T.near > 0 && keep( T.near, 2 ) ) want = 2;
			else if ( c.m.mid && T.mid > 0 && keep( T.mid, 1 ) ) want = 1;
			while ( want > 0 && ( c.failed & ( 1 << want ) || ! c.m[ L[ want ] ] ) ) want --;
			c.was = c.want; c.want = want;
			if ( want > 0 ) order.push( c );

		}

		order.sort( ( a, b ) => a.d - b.d );
		if ( T.resident ) {

			// what the GPU holds is every wanted level whole, in view or not (the view budget only narrows what is
			// drawn of it): over the ceiling, first the cells that are only still held for being left late go down a
			// level, farthest first, then the farthest of the top level
			let held = this.loTris;
			for ( const c of order ) held += c.m[ L[ c.want ] ].ni / 3;
			const down = ( c ) => {

				held -= c.m[ L[ c.want ] ].ni / 3;
				do c.want --; while ( c.want > 0 && ( c.failed & ( 1 << c.want ) || ! c.m[ L[ c.want ] ] ) );
				if ( c.want > 0 ) held += c.m[ L[ c.want ] ].ni / 3;

			};

			for ( let i = order.length - 1; i >= 0 && held > T.resident; i -- ) {

				const c = order[ i ];
				while ( c.want > 0 && held > T.resident && c.d >= ( c.want === 2 ? T.near : T.mid ) ) down( c );

			}

			for ( let i = order.length - 1; i >= 0 && held > T.resident; i -- ) if ( order[ i ].want === 2 ) down( order[ i ] );
			for ( let i = order.length - 1; i >= 0; i -- ) if ( order[ i ].want === 0 ) order.splice( i, 1 );

		}

		if ( T.budget === 'view' ) this._viewBudget( camera, order );
		else this._budget( order );
		for ( const c of order ) {

			if ( c.want > 0 && ! c.mesh[ c.want ] ) {

				const key = c.m[ L[ c.want ] ].file;
				if ( ! this.inflight.has( key ) && this.inflight.size < T.fetch ) this.inflight.set( key, this._fetch( c, c.want ) );

			}

		}

		for ( const c of this.chunks ) {

			// show the wanted level, or until it is in the best one there is
			let show = c.want;
			if ( ! c.mesh[ show ] ) show = c.mesh[ c.level ] ? c.level : ( c.mesh[ 1 ] ? 1 : c.mesh[ 2 ] ? 2 : 0 );
			for ( let l = 0; l < 3; l ++ ) {

				const mesh = c.mesh[ l ];
				if ( ! mesh ) continue;
				mesh.visible = l === show;
				const u = mesh.userData;
				if ( u.parts && l === show ) {

					// the top level: each cluster down to the part size the budget gave it
					u.shown = c.rest;
					u.parts.forEach( ( part, i ) => {

						const n = c.top[ i ].n;
						part.geometry.setDrawRange( part.userData.start, n );
						part.visible = n > 0;
						u.shown += n / 3;

					} );

				}

				if ( l > 0 && l !== show && l !== c.want ) this._drop( c, l );

			}

			c.level = show;

		}

	}

	get pending() {

		let n = this.inflight.size;
		for ( const c of this.chunks ) if ( c.want > 0 && ! c.mesh[ c.want ] && ! this.inflight.has( c.m[ this.L[ c.want ] ].file ) ) n ++;
		return n;

	}

	// resolve when the view from this camera is complete ('all') or its surroundings are in ('near')
	async settle( camera ) {

		const t0 = performance.now();
		for ( ;; ) {

			this.update( camera, 0, { force: true } );
			const waitBase = this.wait === 'all' ? this.baseLeft > 0 : false;
			let waitLevels = this.pending > 0;
			if ( this.wait !== 'all' ) {

				// walking: only the cells right around the camera hold up the first frame, and not for long
				waitLevels = this.chunks.some( ( c ) => c.want > 0 && ! c.mesh[ c.want ] && c.d < 6 ) && performance.now() - t0 < 2500;

			}

			if ( ! waitBase && ! waitLevels ) break;
			await new Promise( ( r ) => this.waiters.push( r ) );

		}

		this.update( camera, 0, { force: true } );

	}

	visibleStats() {

		const lods = [ 0, 0, 0 ];
		let items = 0, tris = 0, gpu = 0;
		for ( const c of this.chunks ) {

			if ( c.mesh[ c.level ] ) {

				items ++; lods[ 2 - c.level ] ++;
				tris += c.mesh[ c.level ].userData.shown ?? c.mesh[ c.level ].userData.tris;

			}

			for ( const mesh of c.mesh ) if ( mesh ) gpu += mesh.userData.tris;

		}

		// lods: cells shown at [ near, mid, lo ]
		// step: how far the view budget has the top level leave out small parts (1: only what does not show anyway)
		return { items, lods, pending: this.pending + ( this.baseLeft > 0 ? this.baseLeft : 0 ), tris, resident: gpu, bytes: this.bytes, ...( this._f ? { step: this._f } : {} ) };

	}

}
