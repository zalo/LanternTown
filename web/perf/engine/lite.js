// Lean runtime for the baked town (tools/webbake.py -> out/web/_bake/): what the light tiers of the walkable demo run.
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
	textureLoad, ivec2, dFdx, dFdy, cross, sign, floor, clamp, sqrt
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { createSky, MaterialLib } from './engine.js';
import { Nav } from './town.js';

// what each baked tier draws. mid / near: distance (m) from the camera to a cell inside which that level is shown;
// tris: ceiling on the triangles shown, all cells at their level (so also on what one view can hold, however the town
// grows); pixels: frame buffer budget;
// glints: lanterns per ground cell whose highlight on wet / polished surfaces is evaluated per pixel (0 = none)
export const BAKED_TIERS = {
	phone: { tex: 256, mid: 20, near: 0, tris: 0.5e6, pixels: 0.8e6, bloom: false, fxaa: true, aniso: 2, fetch: 2, glints: 2, halo: 0.3 },
	low: { tex: 512, mid: 64, near: 20, tris: 2.7e6, pixels: 1.7e6, bloom: true, fxaa: true, aniso: 4, fetch: 3, glints: 4 },
};

const K_OPAQUE = 0, K_GLASS = 1, K_WATER = 2, K_FALL = 3, K_CUT = 4;

// the bake's manifest, or null when there is no (usable) bake: the page then falls back to the streaming path.
// `town`: the assembly (tools/webbake.py --town <name>); the first one lives in out/web/_bake, others in _bake_<name>
export async function loadBake( base, tier, town = 'town' ) {

	if ( ! BAKED_TIERS[ tier ] || ! /^\w+$/.test( town ) ) return null;
	const BAKE_DIR = 'out/web/' + ( town === 'town' ? '_bake' : '_bake_' + town );
	try {

		const r = await fetch( `${base}/${BAKE_DIR}/manifest.json`, { cache: 'no-cache' } );
		if ( ! r.ok ) return null;
		const man = await r.json();
		if ( man.version !== 1 || man.codec !== 'meshopt' || ! man.tex[ BAKED_TIERS[ tier ].tex ] ) return null;
		man.dir = `${base}/${BAKE_DIR}`;
		man.tier = { ...BAKED_TIERS[ tier ] };
		// tuning from the URL: ?mid=30&near=10&tris=500000&pixels=1e6&glints=0
		const q = new URLSearchParams( location.search );
		for ( const k of [ 'mid', 'near', 'tris', 'pixels', 'glints', 'halo' ] ) if ( q.has( k ) && isFinite( parseFloat( q.get( k ) ) ) ) man.tier[ k ] = parseFloat( q.get( k ) );
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
	const renderer = new THREE.WebGPURenderer( { canvas, antialias: false, alpha: false, forceWebGL } );
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
const LEVELS = [ 'lo', 'mid', 'near' ];

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
		this.chunks = bake.chunks.map( ( m ) => {

			const b0 = m.lo.bmin, b1 = m.lo.bmax;
			const box = [ b0[ 0 ], b0[ 2 ], b1[ 0 ], b1[ 2 ] ];
			for ( const l of [ 'mid', 'near' ] ) if ( m[ l ] ) {

				box[ 0 ] = Math.min( box[ 0 ], m[ l ].bmin[ 0 ] ); box[ 1 ] = Math.min( box[ 1 ], m[ l ].bmin[ 2 ] );
				box[ 2 ] = Math.max( box[ 2 ], m[ l ].bmax[ 0 ] ); box[ 3 ] = Math.max( box[ 3 ], m[ l ].bmax[ 2 ] );

			}

			return { m, box, mesh: [ null, null, null ], level: 0, want: 0, d: 0 };

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
				c.mesh[ 0 ] = this._build( buf.subarray( m.off, m.off + m.len ), m );
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

	// The K lanterns that light each 4 m ground cell most, as a small float texture ( 2 K texels per cell: position +
	// range, colour x intensity ). The diffuse light of every lantern is in the vertex colours; this only feeds the
	// view-dependent part - the glints on wet cobbles, polished stone and the canal - which cannot be baked.
	_lightGrid( K ) {

		const L = this.bake.lights || [];
		if ( ! L.length || ! ( K > 0 ) ) return null;
		let x0 = Infinity, z0 = Infinity, x1 = - Infinity, z1 = - Infinity;
		for ( const l of L ) {

			x0 = Math.min( x0, l[ 0 ] - l[ 6 ] ); x1 = Math.max( x1, l[ 0 ] + l[ 6 ] );
			z0 = Math.min( z0, l[ 2 ] - l[ 6 ] ); z1 = Math.max( z1, l[ 2 ] + l[ 6 ] );

		}

		const G = Math.max( 4, ( x1 - x0 ) / 96, ( z1 - z0 ) / 96 );
		const nx = Math.max( 1, Math.ceil( ( x1 - x0 ) / G ) ), nz = Math.max( 1, Math.ceil( ( z1 - z0 ) / G ) );
		const data = new Float32Array( nx * 2 * nz * K * 4 );
		const lum = L.map( ( l ) => 0.2126 * l[ 3 ] + 0.7152 * l[ 4 ] + 0.0722 * l[ 5 ] );
		const best = [];
		for ( let iz = 0; iz < nz; iz ++ ) for ( let ix = 0; ix < nx; ix ++ ) {

			const cx0 = x0 + ix * G, cz0 = z0 + iz * G;
			best.length = 0;
			for ( let i = 0; i < L.length; i ++ ) {

				const l = L[ i ];
				const dx = Math.max( cx0 - l[ 0 ], 0, l[ 0 ] - cx0 - G ), dz = Math.max( cz0 - l[ 2 ], 0, l[ 2 ] - cz0 - G );
				const d2 = dx * dx + dz * dz + 2.25;
				const w = Math.max( 0, 1 - ( d2 / ( l[ 6 ] * l[ 6 ] ) ) ** 2 );
				const e = lum[ i ] * w * w / d2;
				if ( e > 0.002 ) best.push( [ e, i ] );

			}

			best.sort( ( a, b ) => b[ 0 ] - a[ 0 ] );
			for ( let k = 0; k < K; k ++ ) {

				const o = ( ( iz * K + k ) * nx * 2 + ix * 2 ) * 4;
				const l = k < best.length ? L[ best[ k ][ 1 ] ] : null;
				if ( l ) data.set( [ l[ 0 ], l[ 1 ], l[ 2 ], l[ 6 ], l[ 3 ], l[ 4 ], l[ 5 ], l[ 7 ] ?? 1 ], o ); // .. , share of the light a capped lantern sends up
				else data.set( [ 0, - 1e4, 0, 1, 0, 0, 0, 1 ], o );

			}

		}

		const tex = new THREE.DataTexture( data, nx * 2, nz * K, THREE.RGBAFormat, THREE.FloatType );
		tex.minFilter = tex.magFilter = THREE.NearestFilter;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		return { tex, K, G, x0, z0, nx, nz };

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

		const c = clamp( floor( positionWorld.xz.sub( vec2( g.x0, g.z0 ) ).div( g.G ) ), vec2( 0, 0 ), vec2( g.nx - 1, g.nz - 1 ) );
		const ix = int( c.x ).mul( 2 ), iz = int( c.y ).mul( g.K );
		let sum = vec3( 0 );
		for ( let k = 0; k < g.K; k ++ ) {

			const A = textureLoad( g.tex, ivec2( ix, iz.add( k ) ) ), B = textureLoad( g.tex, ivec2( ix.add( 1 ), iz.add( k ) ) );
			const Lv = A.xyz.sub( positionWorld );
			const d2 = max( dot( Lv, Lv ), 0.01 ), d = sqrt( d2 );
			const q = d.div( A.w ), w = saturate( oneMinus( q.mul( q ).mul( q ).mul( q ) ) );
			const cap = mix( float( 1 ), B.w, smoothstep( - 0.42, 0.3, Lv.y.negate().div( d ) ) ); // capFactor of ClusteredLightsNode.js
			sum = sum.add( B.rgb.mul( w.mul( w ).div( d2 ) ).mul( cap ).mul( brdf( Lv.div( d ) ) ) );

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
		const U = this.U = { relief: uniform( 0.05 ), glint: uniform( 1.5 ), sheen: uniform( 1.0 ), town: uniform( 0.4 ) };
		for ( const k in U ) { const v = parseFloat( new URLSearchParams( location.search ).get( 'u_' + k ) ); if ( isFinite( v ) ) U[ k ].value = v; } // ?u_glint=2 (tuning)

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
		opaque.name = 'baked';

		const glass = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false } );
		glass.colorNode = fogged( lit );
		glass.opacityNode = aux;
		glass.name = 'baked glass';

		// canal / basin water: dark body, the navy of the sky towards grazing angles, a slow swell breaking up the
		// reflections of the lanterns
		const water = new THREE.MeshBasicNodeMaterial( { fog: false } );
		{

			const t = time.mul( 0.05 );
			const wp = positionWorld.xz;
			const n1 = texture( eng.noise, wp.mul( 0.33 ).add( vec2( t, t.mul( 0.6 ) ) ) ).rg.sub( 0.5 );
			const n2 = texture( eng.noise, wp.mul( 1.1 ).add( vec2( t.mul( - 0.7 ), t.mul( 1.1 ) ) ) ).ba.sub( 0.5 );
			const w = n1.add( n2.mul( 0.45 ) ).mul( 0.3 );
			const N = normalize( vec3( w.x, 1, w.y ) );
			const fres = pow( oneMinus( saturate( dot( N, V ) ) ), 5 ).mul( 0.98 ).add( 0.02 );
			let c = lit.mul( 0.3 ).add( SKY.mul( fres ).mul( 1.4 ) );
			if ( this.grid ) c = c.add( this._glints( N, V, float( 0.16 ), float( 0.02 ) ).mul( U.glint ) );
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
			m.colorNode = varying( inst( H, 3, 3 ) ).mul( a ).mul( this.T.halo || 0.1 );
			m.opacityNode = a;
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
	_build( bytes, m ) {

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
		g.setIndex( new THREE.BufferAttribute( m.it === 2 ? new Uint16Array( ib.buffer ) : new Uint32Array( ib.buffer ), 1 ) );
		const s = m.q * 65535;
		g.boundingBox = new THREE.Box3(
			new THREE.Vector3( ( m.bmin[ 0 ] - m.min[ 0 ] ) / s, ( m.bmin[ 1 ] - m.min[ 1 ] ) / s, ( m.bmin[ 2 ] - m.min[ 2 ] ) / s ),
			new THREE.Vector3( ( m.bmax[ 0 ] - m.min[ 0 ] ) / s, ( m.bmax[ 1 ] - m.min[ 1 ] ) / s, ( m.bmax[ 2 ] - m.min[ 2 ] ) / s ) );
		g.boundingSphere = g.boundingBox.getBoundingSphere( new THREE.Sphere() );
		const mats = [];
		for ( const [ kind, start, count ] of m.sub ) {

			g.addGroup( start, count, mats.length );
			mats.push( this.mats[ kind ] );

		}

		const mesh = new THREE.Mesh( g, mats.length === 1 ? mats[ 0 ] : mats );
		mesh.position.fromArray( m.min );
		mesh.scale.setScalar( s );
		mesh.matrixAutoUpdate = false;
		mesh.updateMatrix();
		mesh.userData.tris = m.ni / 3;
		// the arrays are only needed until the first draw has put them on the GPU
		// (the index keeps its array type: the WebGPU backend widens 16-bit indices on upload and reads the format from it)
		mesh.onAfterRender = () => {

			buf.array = EMPTY16;
			g.index.array = new g.index.array.constructor( 0 );
			mesh.onAfterRender = THREE.Object3D.prototype.onAfterRender;

		};

		return mesh;

	}

	_drop( c, l ) {

		const mesh = c.mesh[ l ];
		if ( ! mesh ) return;
		this.group.remove( mesh );
		mesh.geometry.dispose();
		c.mesh[ l ] = null;

	}

	_wake() {

		const w = this.waiters; this.waiters = [];
		for ( const f of w ) f();

	}

	async _fetch( c, l ) {

		const m = c.m[ LEVELS[ l ] ], key = m.file;
		try {

			const r = await fetch( `${this.bake.dir}/${m.file}?v=${this.bake.baked}` );
			if ( ! r.ok ) throw new Error( `${m.file}: ${r.status}` );
			const bytes = new Uint8Array( await r.arrayBuffer() );
			this.bytes += bytes.length;
			if ( bytes.length !== m.len ) throw new Error( `${m.file}: size differs from the manifest (stale bake?)` );
			if ( c.want === l && ! c.mesh[ l ] ) {

				c.mesh[ l ] = this._build( bytes, m );
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

	// choose the level of every cell for this camera, start / drop downloads, swap meshes
	update( camera, now, { force = false } = {} ) {

		const T = this.T, x = camera.position.x, z = camera.position.z;
		if ( ! force && ! this._dirty && this._at && Math.hypot( x - this._at[ 0 ], z - this._at[ 1 ] ) < 0.75 ) return;
		this._at = [ x, z ]; this._dirty = false;
		const order = [];
		for ( const c of this.chunks ) {

			c.d = this._dist( c, x, z );
			const keep = ( r, l ) => c.d < ( c.want >= l ? r * 1.2 + 2 : r );
			let want = 0;
			if ( c.m.near && T.near > 0 && keep( T.near, 2 ) ) want = 2;
			else if ( c.m.mid && T.mid > 0 && keep( T.mid, 1 ) ) want = 1;
			while ( want > 0 && ( c.failed & ( 1 << want ) || ! c.m[ LEVELS[ want ] ] ) ) want --;
			c.want = want;
			if ( want > 0 ) order.push( c );

		}

		// inside the triangle budget, nearest first: the mid level for every cell that asks for more than lo, then
		// near for those that asked for it (one near cell costs several mid ones - coverage comes before detail)
		order.sort( ( a, b ) => a.d - b.d );
		let tris = this.loTris;
		for ( const c of order ) {

			const n = c.m.mid ? ( c.m.mid.ni - c.m.lo.ni ) / 3 : 0;
			c.cost = 0;
			if ( c.m.mid && tris + n <= T.tris ) { tris += n; c.cost = c.m.mid.ni / 3; }
			else if ( c.m.mid || c.want < 2 ) c.want = 0;

		}

		for ( const c of order ) {

			if ( c.want < 2 ) continue;
			const n = c.m.near.ni / 3 - ( c.cost || c.m.lo.ni / 3 );
			if ( tris + n <= T.tris ) tris += n;
			else c.want = c.cost > 0 ? 1 : 0;

		}

		for ( const c of order ) {

			if ( c.want > 0 && ! c.mesh[ c.want ] ) {

				const key = c.m[ LEVELS[ c.want ] ].file;
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
				if ( l > 0 && l !== show && l !== c.want ) this._drop( c, l );

			}

			c.level = show;

		}

	}

	get pending() {

		let n = this.inflight.size;
		for ( const c of this.chunks ) if ( c.want > 0 && ! c.mesh[ c.want ] && ! this.inflight.has( c.m[ LEVELS[ c.want ] ].file ) ) n ++;
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
				tris += c.mesh[ c.level ].userData.tris;

			}

			for ( const mesh of c.mesh ) if ( mesh ) gpu += mesh.userData.tris;

		}

		// lods: cells shown at [ near, mid, lo ]
		return { items, lods, pending: this.pending + ( this.baseLeft > 0 ? this.baseLeft : 0 ), tris, resident: gpu, bytes: this.bytes };

	}

}
