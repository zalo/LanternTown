// Town-square engine core: WebGPU renderer, night environment, lantern lights with flicker, flame fx,
// post pipeline (GTAO + bloom + grade). Shared by the review render page and the walkable town.

import * as THREE from 'three';
import {
	pass, mrt, output, normalView, packNormalToRGB, unpackRGBToNormal, sample, vec2, vec3, vec4, float, mix,
	smoothstep, uniform, texture, uv, time, positionLocal, normalize, dot, pow, exp, max, min, abs, fract, floor,
	sin, step, oneMinus, length, screenUV, saturate, mx_fractal_noise_float, renderOutput, interleavedGradientNoise,
	screenCoordinate, metalness, roughness, fog, positionView, positionWorld, Fn, If, Loop, Break, Continue, int, bool,
	getViewPosition, getScreenPosition, perspectiveDepthToViewZ, passTexture, reflect, cross, cubeTexture, nodeObject,
	luminance, clamp, emissive, vogelDiskSample, reference, renderGroup, screenSize, log2, attribute, cameraPosition
} from 'three/tsl';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import SSRNode from 'three/addons/tsl/display/SSRNode.js';
import { ClusteredLighting } from './ClusteredLighting.js'; // vendored: soft near-field falloff
import { sourceRadius, capFactor, aboveFactor, lampFalloff, reachFactor } from './ClusteredLightsNode.js';
export { sourceRadius, lampFalloff };
import { MaterialLib, NO_NORMAL } from './materials.js';
import { AssetLoader, animateParts, ensureMask } from './assets.js';

export { THREE, MaterialLib, AssetLoader, animateParts, ensureMask };

// ---------------------------------------------------------------------------------------- sky

export function createSky( moonDir, noise ) {

	const U = {
		moonDir: uniform( moonDir.clone().normalize() ),
		zenith: uniform( new THREE.Color( 0.004, 0.009, 0.03 ) ),
		horizon: uniform( new THREE.Color( 0.012, 0.024, 0.062 ) ),
		glow: uniform( new THREE.Color( 0.026, 0.016, 0.008 ) ), // warm town glow hugging the horizon
		glowAloft: uniform( new THREE.Color( 0.0075, 0.0085, 0.0125 ) ), // the same band seen from over the roofs
		cloud: uniform( new THREE.Color( 0.03, 0.045, 0.085 ) ), // moonlit cloud tops
		clouds: uniform( 0.8 ),
		cover: uniform( 0.46 ), // noise level where cloud begins: lower = more sky covered
		glowGain: uniform( 1.0 ),
		cloudGain: uniform( 1.0 ),
		stars: uniform( 1.0 ),
		moon: uniform( 1.0 ),
		gain: uniform( 1.0 ),
	};
	const d = normalize( positionLocal );
	const h = d.y;
	let col = mix( U.horizon, U.zenith, pow( saturate( h ), 0.42 ) );
	// from above the roofs the picture's whole sky is the few degrees over the horizon, where the warm glow band lives:
	// what reads as a glow behind the roofs from the street turned that entire sky muddy brown-purple. Up there the
	// glow is the navy's own lift (same band, cool, dimmer) and only a trace of the warmth is left.
	const aloft = smoothstep( 9, 24, cameraPosition.y );
	const glow = mix( U.glow, U.glowAloft, aloft ).mul( U.glowGain );
	// town glow on the haze: a warm band hugging the roofs inside a wider, cooler lift of the navy
	col = col.add( glow.mul( exp( max( h, 0 ).mul( - 7 ) ).add( exp( max( h, 0 ).mul( - 2.4 ) ).mul( vec3( 0.3, 0.42, 0.85 ) ) ) ) );
	// below the horizon: dark ground haze
	col = mix( col, vec3( 0.006, 0.009, 0.018 ), smoothstep( 0.0, - 0.25, h ) );
	// stars: hashed cells on the direction sphere
	const p = d.mul( 170 );
	const cell = floor( p ), f = fract( p ).sub( 0.5 );
	const hsh = fract( sin( dot( cell, vec3( 127.1, 311.7, 74.7 ) ) ).mul( 43758.5453 ) );
	const hs2 = fract( hsh.mul( 91.345 ) );
	const star = step( 0.988, hsh ).mul( smoothstep( 0.22, 0.02, length( f.add( vec3( hs2.sub( 0.5 ).mul( 0.5 ) ) ) ) ) );
	const tw = sin( time.mul( 2.0 ).add( hsh.mul( 300 ) ) ).mul( 0.25 ).add( 0.75 );
	const starCol = mix( vec3( 0.75, 0.82, 1.0 ), vec3( 1.0, 0.86, 0.7 ), hs2 );
	// moonlit cloud deck over the whole dome: a layer projected from a plane overhead (so it foreshortens towards
	// the horizon), broken into banks with clear gaps. Each bank is lit from the moon's side (density difference
	// towards the moon), thin edges let the moonlight through, and the town's glow warms the bases near the horizon.
	const md = dot( d, U.moonDir );
	const hh = max( h, 0.0 );
	const q = d.xz.div( hh.add( 0.2 ) );
	const drift = vec2( time.mul( 0.0016 ), time.mul( 0.0006 ) );
	const dens = ( pq ) => {

		const a = texture( noise, pq.mul( 0.16 ).add( drift ) ).r;
		const b = texture( noise, vec2( pq.x.mul( 0.44 ).add( pq.y.mul( 0.9 ) ), pq.y.mul( 0.44 ).sub( pq.x.mul( 0.9 ) ) ).mul( 0.42 ).add( drift.mul( 1.7 ) ) ).g;
		return a.mul( 0.6 ).add( b.mul( 0.4 ) );

	};
	const n0 = dens( q ), n1 = dens( q.add( U.moonDir.xz.normalize().mul( 0.3 ) ) );
	const fine = texture( noise, q.mul( 1.3 ).add( drift.mul( 3 ) ) );
	const nn = n0.add( fine.b.sub( 0.5 ).mul( 0.14 ) );
	const cloud = smoothstep( U.cover, U.cover.add( 0.2 ), nn ).mul( smoothstep( - 0.02, 0.06, h ) ).mul( U.clouds );
	const thin = oneMinus( smoothstep( U.cover.add( 0.08 ), U.cover.add( 0.3 ), nn ) );
	const side = saturate( n0.sub( n1 ).mul( 7 ).add( 0.45 ) );
	const toMoon = pow( saturate( md.mul( 0.5 ).add( 0.5 ) ), 3 );
	const cloudLit = U.cloud.mul( side.mul( thin.mul( 0.75 ).add( 0.25 ) ).mul( fine.a.mul( 0.9 ).add( 0.55 ) ).mul( 1.1 ).add( 0.16 ) ).mul( toMoon.mul( 1.4 ).add( 0.75 ) )
		.add( U.cloud.mul( pow( saturate( md ), 14 ).mul( thin ).mul( 2.5 ) ) )
		.add( glow.mul( exp( hh.mul( - 4.5 ) ) ).mul( 0.8 ) );
	col = col.add( starCol.mul( star ).mul( tw ).mul( hs2.mul( 0.9 ).add( 0.25 ) ).mul( smoothstep( 0.02, 0.3, h ) ).mul( oneMinus( saturate( cloud.mul( 1.6 ) ) ) ).mul( U.stars ) );
	col = mix( col, cloudLit.mul( U.cloudGain ), cloud );
	// moon disc + halo
	const disc = smoothstep( 0.99965, 0.9998, md );
	const halo = exp( oneMinus( md ).mul( - 900 ) ).mul( 0.06 ).add( exp( oneMinus( md ).mul( - 60 ) ).mul( 0.012 ) );
	col = col.add( vec3( 0.85, 0.9, 1.0 ).mul( disc.mul( 2.2 ).add( halo ) ).mul( U.moon ) );
	const m = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide, depthWrite: false, fog: false } );
	m.colorNode = col.mul( U.gain );
	const mesh = new THREE.Mesh( new THREE.SphereGeometry( 1, 48, 24 ), m );
	mesh.scale.setScalar( 900 );
	mesh.frustumCulled = false;
	mesh.renderOrder = - 1000;
	mesh.name = 'sky';
	mesh.userData.U = U;
	return mesh;

}

// ---------------------------------------------------------------------------------------- capped lanterns

// The clustered lights read light.userData.up (ClusteredLightsNode capFactor); a shadow-casting PointLight goes through
// three's own light node, so it gets the same cap as a colour node. Call once per light; userData.up may change later.
const _caps = [];
export function capLight( L ) {

	// plain uniforms in the shared render group, synced once per frame from eng.tick(): as update nodes (or in the
	// per-object group) they cost a callback and a buffer rewrite for every lit object in every frame
	const c = { L, col: uniform( new THREE.Color() ).setGroup( renderGroup ), pos: uniform( new THREE.Vector3() ).setGroup( renderGroup ), up: uniform( 1 ).setGroup( renderGroup ) };
	_caps.push( c );
	const v = positionWorld.sub( c.pos );
	const sinElev = v.y.div( length( v ).max( 1e-3 ) );
	const lampCap = capFactor( sinElev, c.up ), lampReach = reachFactor( dot( v, v ) );
	L.colorNode = c.col.mul( lampCap ).mul( lampReach );
	L.lampNodes = { lampCap, lampReach, lampAbove: aboveFactor( sinElev ) }; // (for lighting models that take these factors differently: foliage)
	return L;

}

function syncCapLights() {

	for ( const c of _caps ) {

		if ( ! c.L.parent ) continue;
		c.L.updateWorldMatrix( true, false );
		c.col.value.copy( c.L.color ).multiplyScalar( c.L.intensity );
		c.pos.value.setFromMatrixPosition( c.L.matrixWorld );
		c.up.value = c.L.userData.up ?? 1;

	}

}

// ---------------------------------------------------------------------------------------- contact shadows

// Soft dark pads under small ground props (chairs, tables, benches, barrels). The cafe and street lamps are clustered
// lights without shadow maps and the AO radius is too wide for chair legs, so without these the furniture floats.
// list: [ { x, y, z, rot, hx, hz, k } ] (three.js world space, half extents in metres, k = strength 0..1)
export function createContactShadows( list ) {

	const m = new THREE.MeshBasicNodeMaterial( { transparent: true, depthWrite: false, fog: false, polygonOffset: true, polygonOffsetFactor: - 2, polygonOffsetUnits: - 4 } );
	const r = length( uv().sub( 0.5 ) ).mul( 2 );
	const a = oneMinus( smoothstep( 0.3, 1.0, r ) );
	const k = a.mul( a ).mul( attribute( 'aK', 'float' ) ).mul( oneMinus( smoothstep( 30, 70, positionView.z.negate() ) ) );
	m.colorNode = vec3( 0, 0, 0 );
	m.opacityNode = k;
	// (the surface buffer is blended towards rough under the pad: the wet-paving mirror there would show the underside
	// of the prop, not the lamps and sky)
	m.mrtNode = mrt( { normal: vec4( 0, 0, 0, 0 ), mr: vec4( 0, 1, 0, k.mul( 1.2 ).min( 1 ) ) } );
	const g = new THREE.PlaneGeometry( 2, 2 ).rotateX( - Math.PI / 2 );
	g.setAttribute( 'aK', new THREE.InstancedBufferAttribute( new Float32Array( list.map( ( o ) => o.k ) ), 1 ) );
	const mesh = new THREE.InstancedMesh( g, m, list.length );
	const d = new THREE.Object3D();
	list.forEach( ( o, i ) => {

		d.position.set( o.x, o.y, o.z );
		d.rotation.set( 0, o.rot, 0 );
		d.scale.set( o.hx, 1, o.hz );
		d.updateMatrix();
		mesh.setMatrixAt( i, d.matrix );

	} );
	mesh.frustumCulled = false;
	mesh.castShadow = mesh.receiveShadow = false;
	mesh.renderOrder = - 1; // under the other blended surfaces (water sheets, glass, flames)
	mesh.name = 'contact';
	return mesh;

}

// ---------------------------------------------------------------------------------------- flames

let _flameGeo = null;

export function createFlame( noiseTex, { size = 0.09, color = '#ffb860', glow = 9, phase = 0, speed = 1 } = {} ) {

	const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	m.mrtNode = NO_NORMAL;
	const p = uv();
	const y = p.y, x = p.x.sub( 0.5 ).mul( 2 );
	const t = time.mul( speed ).add( phase * 7.3 );
	const n1 = texture( noiseTex, vec2( x.mul( 0.12 ).add( phase ), y.mul( 0.3 ).sub( t.mul( 0.5 ) ) ) ).r.sub( 0.5 );
	const n2 = texture( noiseTex, vec2( x.mul( 0.35 ).add( phase * 2.1 ), y.mul( 0.8 ).sub( t.mul( 1.05 ) ) ) ).g.sub( 0.5 );
	const sway = sin( t.mul( 2.1 ) ).mul( 0.06 ).add( n1.mul( 0.55 ) ).add( n2.mul( 0.22 ) );
	const xx = x.add( sway.mul( y ) );
	const hgt = y.div( n1.mul( 0.25 ).add( 0.9 ) ); // tip height breathes
	const w = min( pow( max( hgt, 0 ).div( 0.22 ), 0.55 ), pow( max( oneMinus( hgt ), 0 ).div( 0.78 ), 0.85 ) ).mul( 0.42 );
	const dd = abs( xx ).div( max( w, 0.001 ) );
	const body = smoothstep( 1.0, 0.25, dd ).mul( step( hgt, 1 ) );
	const core = smoothstep( 0.7, 0.0, dd ).mul( smoothstep( 0.95, 0.25, hgt ) );
	const c = new THREE.Color( color );
	const outer = vec3( c.r, c.g * 0.55, c.b * 0.25 );
	const inner = vec3( 1.0, 0.86, 0.6 );
	let col = mix( outer, inner, core );
	col = mix( vec3( 0.12, 0.25, 0.9 ).mul( 0.5 ), col, smoothstep( 0.02, 0.2, hgt ) ); // blue root
	m.colorNode = col.mul( body ).mul( glow );
	m.opacityNode = body;
	const s = new THREE.Sprite( m );
	s.center.set( 0.5, 0.02 );
	s.scale.set( size * 0.8, size * 1.7, 1 );
	s.renderOrder = 5;
	s.name = 'flame';
	return s;

}

// glare of the bare mantle seen through clear lantern glass: a tight hot core in the middle of the halo of every
// light that sits at a visible burner (createHalo( .., core = true ); helper lights of windows and shop fronts: false).
// 0 = off (asset review renders); the town page sets it (?glare=).
export const haloCore = uniform( 0 );

// soft additive glow around a lantern (cheap stand-in for scattering in the night haze)
export function createHalo( color = '#ffbf6a', size = 1.2, strength = 0.12, core = true ) {

	const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	m.mrtNode = NO_NORMAL;
	const r = length( uv().sub( 0.5 ) ).mul( 2 );
	const a = exp( r.mul( r ).mul( - 5 ) ).sub( 0.0067 ).max( 0 );
	const c = new THREE.Color( color );
	const hot = core ? exp( r.mul( r ).mul( - 11 * size * size ) ) : null; // about 0.3 m across whatever the sprite size
	m.colorNode = vec3( c.r, c.g, c.b ).mul( hot ? a.add( hot.mul( haloCore ) ) : a ).mul( strength );
	m.opacityNode = hot ? a.add( hot.mul( haloCore.min( 1 ) ) ).min( 1 ) : a;
	const s = new THREE.Sprite( m );
	s.scale.set( size, size, 1 );
	s.renderOrder = 4;
	s.name = 'halo';
	return s;

}

// ---------------------------------------------------------------------------------------- reflections

// Screen-space reflections with a confidence channel. three's SSRNode (mirror / blur mode) returns whatever the
// ray march finds at full strength, so rays that leave the screen, pass behind thin objects or hit a surface edge-on
// leave detached smears. This keeps its targets and blur chain but traces with a bisected hit and returns
// ( colour * confidence, confidence ): the composite fades to the reflection probe where the ray is not to be trusted.
class ConfidentSSR extends SSRNode {

	constructor( colorNode, depthNode, normalNode, options ) {

		super( colorNode, depthNode, normalNode, options );
		// reflection blur from the surface roughness: a rain puddle (0.1) is a soft mirror, damp stone a wide glow
		const mips = this._blurRenderTarget.texture.mipmaps.length - 1;
		this.blur = uniform( 9 );
		this._blurredTextureNode = passTexture( this, this._blurRenderTarget.texture ).level( this.roughnessNode.mul( this.blur ).clamp( 0, mips ) );

	}

	setup( builder ) {

		const uvNode = uv();
		const res = this._resolution, proj = this._cameraProjectionMatrix, projInv = this._cameraProjectionMatrixInverse;
		const near = this._cameraNear, far = this._cameraFar;
		const viewZ = ( d ) => perspectiveDepthToViewZ( d, near, far );
		const lineDist = ( p, a, b ) => cross( p.sub( a ), p.sub( b ) ).length().div( b.sub( a ).length() );
		const depthAt = ( u ) => this.depthNode.sample( u ).r;

		const ssrFn = Fn( () => {

			const depth = depthAt( uvNode ).toVar();
			depth.greaterThanEqual( 1.0 ).discard(); // sky
			float( this.metalnessNode ).lessThanEqual( 0.003 ).discard(); // nothing to mirror
			const P0 = getViewPosition( uvNode, depth, projInv ).toVar();
			const N = this.normalNode.rgb.normalize().toVar();
			const I = normalize( P0 ).toVar();
			const R = reflect( I, N ).normalize().toVar();
			const P1 = P0.add( R.mul( this.maxDistance.div( dot( I.negate(), N ).max( 0.02 ) ) ) ).toVar();
			If( P1.z.greaterThan( near.negate() ), () => {

				P1.assign( P0.add( R.mul( near.negate().sub( P0.z ).div( R.z ) ) ) );

			} );
			const d0 = uvNode.mul( res ).toVar();
			const span = getScreenPosition( P1, proj ).mul( res ).sub( d0 ).toVar();
			const total = int( max( abs( span.x ), abs( span.y ) ).mul( this.quality ).clamp( 1, 900 ) ).toVar();
			const stepV = span.div( float( total ) ).toVar();
			const invRes = vec2( 1 ).div( res ).toVar();
			const rz0 = float( 1 ).div( P0.z ).toVar(), rz1 = float( 1 ).div( P1.z ).toVar();
			const rayZ = ( s ) => float( 1 ).div( rz0.add( s.mul( rz1.sub( rz0 ) ) ) );
			const found = bool( false ).toVar();
			const sLo = float( 0 ).toVar(), sHi = float( 0 ).toVar(), tkHit = float( 0 ).toVar();
			Loop( { start: int( 1 ), end: total }, ( { i } ) => {

				const s = float( i ).div( float( total ) ).toVar();
				const xy = d0.add( stepV.mul( float( i ) ) ).toVar();
				If( xy.x.lessThan( 0 ).or( xy.x.greaterThan( res.x ) ).or( xy.y.lessThan( 0 ) ).or( xy.y.greaterThan( res.y ) ), () => {

					Break();

				} );
				const uvS = xy.mul( invRes ).toVar();
				const d = depthAt( uvS ).toVar();
				If( rayZ( s ).lessThanEqual( viewZ( d ) ), () => {

					// the ray is behind the depth buffer here: a hit only if it is within the surface's thickness
					const vP = getViewPosition( uvS, d, projInv ).toVar();
					const away = lineDist( vP, P0, P1 ).toVar();
					const tk = max( getViewPosition( uvS.add( vec2( invRes.x, 0 ) ), d, projInv ).x.sub( vP.x ).mul( 3 ), this.thickness ).toVar();
					If( away.lessThanEqual( tk ), () => {

						const vN = this.normalNode.sample( uvS ).rgb.normalize();
						If( dot( R, vN ).greaterThanEqual( 0 ), () => {

							Continue(); // a back face

						} );
						found.assign( true );
						sLo.assign( float( i ).sub( 1 ).div( float( total ) ) );
						sHi.assign( s );
						tkHit.assign( tk );
						Break();

					} );

				} );

			} );
			const out = vec4( 0 ).toVar();
			If( found, () => {

				// bisect the bracketed crossing: no stair-steps from the march stride
				Loop( { start: int( 0 ), end: int( 5 ), type: 'int', condition: '<' }, () => {

					const sm = sLo.add( sHi ).mul( 0.5 ).toVar();
					const zm = viewZ( depthAt( d0.add( span.mul( sm ) ).mul( invRes ) ) );
					If( rayZ( sm ).lessThanEqual( zm ), () => {

						sHi.assign( sm );

					} ).Else( () => {

						sLo.assign( sm );

					} );

				} );
				const uvH = d0.add( span.mul( sHi ) ).mul( invRes ).toVar();
				const vP = getViewPosition( uvH, depthAt( uvH ), projInv ).toVar();
				const vN = this.normalNode.sample( uvH ).rgb.normalize();
				const height = dot( vP.sub( P0 ), N ); // of the hit above the mirror plane
				const reach = oneMinus( height.div( this.maxDistance ) ).saturate();
				const edge = min( min( uvH.x, oneMinus( uvH.x ) ), min( uvH.y, oneMinus( uvH.y ) ) );
				const conf = reach.mul( reach ) // far above the mirror
					.mul( smoothstep( 0.0, this.screenEdgeFade, edge ) ) // what lies beyond the frame is unknown
					.mul( smoothstep( 1.0, 0.85, sHi ) ) // end of the ray
					.mul( smoothstep( tkHit, tkHit.mul( 0.35 ), lineDist( vP, P0, P1 ) ) ) // depth mismatch: passed behind / thin occluder
					.mul( smoothstep( 0.0, - 0.25, dot( R, vN ) ) ) // surface seen edge-on from the ray
					.mul( smoothstep( 0.45, 0.1, R.z ) ); // ray coming back at the camera: it would see the hidden side
				const col = this.colorNode.sample( uvH ).rgb.toVar();
				col.mulAssign( this.maxLuminance.div( luminance( col ).max( 1e-4 ) ).min( 1 ) );
				out.assign( vec4( col.mul( conf ), conf ) );

			} );
			return out.max( 0 );

		} );

		this._ssrFn = ssrFn;
		this._sharedContext = builder.getSharedContext();
		this._buildSSRMaterial();
		this._buildBlurMaterial();
		this._copyMaterial.fragmentNode = texture( this._ssrRenderTarget.texture );
		this._copyMaterial.needsUpdate = true;
		return this.getTextureNode();

	}

}

// ---------------------------------------------------------------------------------------- engine

export async function createEngine( { canvas, base = '.', width = 1280, height = 720, pixelRatio = 1, clustered = true,
	anisotropy = 8, maxTextureSize = 2048, shadows = true, fov = 40, forceWebGL = false, reflections = false, envProbe = false } = {} ) {

	const renderer = new THREE.WebGPURenderer( { canvas, antialias: false, alpha: false, forceWebGL } );
	renderer.setPixelRatio( pixelRatio );
	renderer.setSize( width, height, false );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.25;
	renderer.shadowMap.enabled = shadows;
	renderer.shadowMap.type = THREE.PCFShadowMap;
	if ( clustered ) renderer.lighting = new ClusteredLighting( 512, 32, 24, 48 );
	await renderer.init();
	const isWebGPU = renderer.backend.isWebGPUBackend === true;

	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera( fov, width / height, 0.05, 2000 );
	const lib = await new MaterialLib( { base, anisotropy, maxTextureSize } ).init();
	const assets = new AssetLoader( lib, base );
	lib.refraction = reflections; // falling water shows the frame behind it (one framebuffer copy while a fall is in view)
	// static reflection probes (cube maps of the finished scene; the post pipeline mirrors them in every glossy surface
	// that screen-space reflections cannot resolve): 0/1 box-projected pair inside the box min..max, 2 everywhere else.
	// The box is empty until a scene sets it.
	lib.probe = { p: [ 0, 1, 2 ].map( () => {

		const rt = new THREE.CubeRenderTarget( reflections ? 512 : 256, { type: THREE.HalfFloatType, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter } );
		return { rt, cam: new THREE.CubeCamera( 0.1, 400, rt ), pos: uniform( new THREE.Vector3() ) };

	} ), min: uniform( new THREE.Vector3( 1e5, 1e5, 1e5 ) ), max: uniform( new THREE.Vector3( - 1e5, - 1e5, - 1e5 ) ), splitZ: uniform( 0 ), gain: uniform( 0 ), mats: new Set() };
	{

		const empty = new THREE.Scene(); // allocate the cube textures so materials can bind them before the first capture
		for ( const pr of lib.probe.p ) {

			// (allocate with the full mip chain first: CubeCamera.update() switches generateMipmaps off for its first
			// five faces, and a cube texture first created during one of those gets a single level for good - every
			// .level() lookup then reads the sharp level 0 and glossy paving is stippled with lamp and window texels)
			renderer.initRenderTarget( pr.rt );
			pr.cam.update( renderer, empty );

		}
		renderer.setRenderTarget( null );

	}

	// --- environment
	const moonDir = new THREE.Vector3( - 0.45, 0.62, 0.35 ).normalize();
	const sky = createSky( moonDir, lib.noise );
	scene.add( sky );
	const hemi = new THREE.HemisphereLight( new THREE.Color( 0.32, 0.42, 0.7 ), new THREE.Color( 0.22, 0.16, 0.1 ), 0.06 );
	scene.add( hemi );
	// the streets are canyons that see a strip of the night sky, the roofs see all of it: the sky half of the fill
	// grows with height (x ( 1 + roofFill ) from the eaves up), so the roofscape reads navy instead of brown-black
	// and upper storeys cool towards the moonlight while the street level keeps its balance
	const roofFill = uniform( 4.8 ).setGroup( renderGroup );
	const hemiSky = uniform( new THREE.Color() ).setGroup( renderGroup ); // synced in tick()
	const syncLights = () => {

		syncCapLights();
		hemiSky.value.copy( hemi.color ).multiplyScalar( hemi.intensity );
		lib.u.skyFill.value.copy( hemiSky.value ); // (foliage takes its own share of the sky: see FoliageLighting)

	};
	hemi.colorNode = hemiSky.mul( smoothstep( 5.5, 13, positionWorld.y ).mul( roofFill ).add( 1 ) );
	const moon = new THREE.DirectionalLight( new THREE.Color( 0.56, 0.68, 1.0 ), 0.22 );
	moon.position.copy( moonDir ).multiplyScalar( 60 );
	moon.castShadow = shadows;
	moon.shadow.mapSize.set( 2048, 2048 );
	moon.shadow.bias = - 0.0004;
	moon.shadow.normalBias = 0.03;
	if ( reflections ) {

		// (full tiers) a wider percentage-closer kernel than three's 5 taps: 12 taps on a per-pixel rotated disc,
		// radius moon.shadow.radius texels. A town-sized map has texels of a few centimetres, so railings and balusters
		// are at the limit of what it resolves: their shadows soften to bars instead of beating against the texel grid.
		moon.shadow.filterNode = Fn( ( { depthTexture, shadowCoord, shadow } ) => {

			const mapSize = reference( 'mapSize', 'vec2', shadow ).setGroup( renderGroup );
			const radius = reference( 'radius', 'float', shadow ).setGroup( renderGroup ).div( mapSize.x );
			const phi = interleavedGradientNoise( screenCoordinate.xy ).mul( 6.28318530718 );
			let sum = float( 0 );
			for ( let i = 0; i < 12; i ++ ) sum = sum.add( texture( depthTexture, shadowCoord.xy.add( vogelDiskSample( i, 12, phi ).mul( radius ) ) ).compare( shadowCoord.z ) );
			return sum.div( 12 );

		} );

	}

	scene.add( moon, moon.target );
	// night haze: exp2 distance fog that thins with altitude, so the streets sit in a soft veil while roofs, towers
	// and the far skyline stay cleaner and separate into layers
	const FOG = { color: uniform( new THREE.Color( 0.012, 0.015, 0.024 ) ), density: uniform( 0.0 ), height: uniform( 18 ) };
	{

		const thin = exp( max( positionWorld.y, 0 ).div( FOG.height ).negate() ).mul( 0.8 ).add( 0.2 );
		const fx = length( positionView ).mul( FOG.density ).mul( thin );
		scene.fogNode = fog( FOG.color, oneMinus( exp( fx.mul( fx ).negate() ) ) );

	}

	// --- post
	const pipeline = new THREE.RenderPipeline( renderer );
	const scenePass = pass( scene, camera );
	// highlight knee (opt-in, 0 = off): a lantern a pace from a wall lights it far past white, and the tone curve then
	// bleaches the plaster to the colour of the lamp itself. Lit surfaces roll off towards a ceiling (hue kept), glowing
	// ones (lit windows, clock faces, lamp glass) towards a higher one, so sources stay brighter than what they light.
	// Unlit materials (flames, halos, sky: no roughness) pass untouched.
	const kneeLit = uniform( 0.0 ), kneeGlow = uniform( 3.0 );
	// (keep: how much darker than its own clean stone a weathered surface is. The knee is taken on the clean surface
	// and the weathering goes on after it - under a lantern a wall sits far up the shoulder, where a third less albedo
	// is a twentieth less picture, and every lamp-lit wall came out the one clean beige whatever grime it carried.)
	const kneeOf = ( keep ) => Fn( () => {

		const c = keep ? output.rgb.div( keep ) : output.rgb;
		const m = mix( kneeLit, max( kneeGlow, kneeLit ), saturate( luminance( emissive ).mul( 2 ) ) );
		const t = m.mul( 0.35 );
		const L = max( max( c.r, c.g ), max( c.b, 1e-4 ) );
		const x = L.sub( t ).max( 0 ).div( m.sub( t ).max( 1e-4 ) );
		const Lk = min( L, t ).add( m.sub( t ).mul( x.div( x.add( 1 ) ) ) ); // slow shoulder: the falloff across the wall survives
		// (blended glass is exempt too: its glow reaches the frame scaled by its opacity)
		const on = step( 0.001, kneeLit ).mul( step( 0.01, roughness ) ).mul( step( 0.999, output.a ) );
		const ck = c.mul( mix( float( 1 ), Lk.div( L ), on ) );
		return vec4( keep ? ck.mul( keep ) : ck, output.a );

	} )();
	const kneed = kneeOf( null );
	lib.kneeWith = ( keep ) => mrt( { output: kneeOf( keep ) } );
	// (foliage: the spare channel of the surface buffer says how much of the screen-space occlusion to leave off. A
	// crown is a cloud of cards, each "occluded" by its neighbours; every other surface writes 0 = full occlusion)
	lib.aoRelief = ( k ) => mrt( { mr: vec4( metalness, roughness, k, 1 ) } );
	const sceneMRT = mrt( { output: kneed, normal: packNormalToRGB( normalView ), mr: vec4( metalness, roughness, 0, 1 ) } );
	// let blended materials (glass, flames, halos: NO_NORMAL) leave the normal / surface buffers untouched
	sceneMRT.setBlendMode( 'normal', new THREE.BlendMode( THREE.MaterialBlending ) );
	sceneMRT.setBlendMode( 'mr', new THREE.BlendMode( THREE.MaterialBlending ) );
	scenePass.setMRT( sceneMRT );
	const colorTex = scenePass.getTextureNode( 'output' );
	const depthTex = scenePass.getTextureNode( 'depth' );
	const normalTex = scenePass.getTextureNode( 'normal' );
	scenePass.getTexture( 'normal' ).type = reflections ? THREE.HalfFloatType : THREE.UnsignedByteType; // mirrors need finer normals than 8 bits
	scenePass.getTexture( 'mr' ).type = THREE.UnsignedByteType;
	const sceneNormal = sample( ( uvn ) => unpackRGBToNormal( normalTex.sample( uvn ) ) );
	const aoPass = ao( depthTex, sceneNormal, camera );
	aoPass.resolutionScale = 1;
	aoPass.radius.value = 0.45;
	aoPass.thickness.value = 1.0;
	aoPass.scale.value = 1.0;
	aoPass.samples.value = 16;
	const P = {
		ao: uniform( 0.85 ), bloom: uniform( 0.3 ), vignette: uniform( 0.35 ), grain: uniform( 0.012 ),
		tint: uniform( new THREE.Color( 1, 1, 1 ) ), lift: uniform( 0.0 ), contrast: uniform( 1.12 ), saturation: uniform( 1.08 ),
		ssr: uniform( 1.0 ), env: uniform( 1.0 ), knee: kneeLit, kneeGlow, tame: uniform( 0.35 ), free: uniform( 0.7 ), roofFill,
	};
	// screen-space reflections (opt-in): smooth surfaces - rain puddles, wet sheen, water, polished metal - mirror what
	// is on screen. The mirror strength comes from the surface roughness alone, so no material has to opt in.
	let ssrPass = null;
	const mrTex = scenePass.getTextureNode( 'mr' );
	const gloss = smoothstep( 0.5, 0.06, mrTex.g );
	// how far the surface normals scatter within a pixel's neighbourhood (0 = smooth; relief finer than the pixel grid:
	// distant cobbles, rippled puddles). A mirror ray per pixel through such normals lands somewhere else every pixel
	// and prints the brightest things in the square - lamps, stained glass - over the paving as coloured confetti;
	// instead the mirror image is taken along the mean normal and blurred by the scatter.
	const pxl = vec2( 1 ).div( screenSize );
	// (a 3 x 3 pixel tent in four bilinear taps: neighbouring pixels share most of it, so the mean and the scatter
	// vary smoothly)
	const nBox = ( uvn ) => {

		const t = ( dx, dy ) => normalTex.sample( uvn.add( vec2( dx, dy ).mul( pxl ) ) ).rgb;
		return t( 0.5, 0.5 ).add( t( - 0.5, 0.5 ) ).add( t( 0.5, - 0.5 ) ).add( t( - 0.5, - 0.5 ) ).mul( 0.5 ).sub( 1 );

	};
	const nSum = nBox( screenUV ).toVar();
	const nScatter = oneMinus( length( nSum ) ).max( 0 ); // ~ half the squared spread angle
	const gTap = ( dx, dy ) => mrTex.sample( screenUV.add( vec2( dx, dy ).mul( pxl ) ) ).g;
	const roughMean = mrTex.g.add( gTap( 1.5, 0.5 ) ).add( gTap( - 0.5, 1.5 ) ).add( gTap( - 1.5, - 0.5 ) ).add( gTap( 0.5, - 1.5 ) ).div( 5 );
	if ( reflections ) {

		// (rays leave along the neighbourhood's mean normal: the image in wet paving is a coherent smear, not a
		// different hit for every pixel)
		const meanNormal = sample( ( uvn ) => nBox( uvn ).normalize() );
		ssrPass = nodeObject( new ConfidentSSR( nodeObject( colorTex ), nodeObject( depthTex ), nodeObject( meanNormal ), { metalnessNode: gloss.mul( gloss ), roughnessNode: mrTex.g.add( nScatter.sqrt().mul( 3.0 ) ), camera } ) );
		ssrPass.maxDistance.value = 30;
		ssrPass.thickness.value = 0.25;
		ssrPass.quality.value = 0.5;
		ssrPass.maxLuminance.value = 40;
		ssrPass.screenEdgeFade.value = 0.08;

	}

	// what a glossy surface mirrors: the screen-space hit where the ray can be trusted, else the reflection probes
	// (the canal pair box-projected onto its corridor, the square's probe by direction everywhere else)
	const camWorld = uniform( camera.matrixWorld ), camProjInv = uniform( camera.projectionMatrixInverse );
	const reflection = ( aoNode ) => {

		const PR = lib.probe;
		const depth = depthTex.r;
		const vP = getViewPosition( screenUV, depth, camProjInv );
		const Nv = unpackRGBToNormal( normalTex.rgb ).normalize();
		const Iv = normalize( vP );
		// (strength from the neighbourhood's mean normal and gloss: per pixel it flickers between 4% and full over
		// fine relief and stipples the paving with the colour of whatever the probe holds in that direction)
		const glossM = smoothstep( 0.5, 0.06, roughMean );
		const ndv = saturate( dot( nSum.normalize(), Iv.negate() ) );
		const sin2 = oneMinus( ndv.mul( ndv ) );
		// mirror strength: none looking straight down, full at grazing angles (steeper than Schlick: wet paving is a
		// thin film over a rough base and should keep its long lamp streaks)
		const fres = sin2.mul( sin2 ).mul( 0.96 ).add( 0.04 );
		const w = glossM.mul( glossM ).mul( fres ).mul( step( depth, 0.999999 ) );
		const Pw = camWorld.mul( vec4( vP, 1 ) ).xyz, Nw = camWorld.mul( vec4( Nv, 0 ) ).xyz;
		const R0 = camWorld.mul( vec4( reflect( Iv, nSum.normalize() ), 0 ) ).xyz;
		// level surfaces never mirror what is below the horizon
		const R = normalize( vec3( R0.x, mix( R0.y, max( R0.y, 0.03 ), smoothstep( 0.6, 0.9, Nw.y ) ), R0.z ) );
		// probe mip: by roughness, and no finer than the scatter of the normals lets neighbouring pixels agree on
		// (a 512 cube: 0.003 rad per texel at level 0)
		const lod = max( mrTex.g.mul( 8 ), log2( nScatter.mul( 2 ).sqrt().mul( 330 ).max( 1 ) ) );
		const inv = vec3( 1 ).div( R.add( vec3( 1e-5 ) ) );
		const tm = max( PR.max.sub( Pw ).mul( inv ), PR.min.sub( Pw ).mul( inv ) );
		const hit = Pw.add( R.mul( min( tm.x, min( tm.y, tm.z ) ) ) );
		const inBox = smoothstep( 0.0, 1.5, min( min( Pw.x.sub( PR.min.x ), PR.max.x.sub( Pw.x ) ), min( Pw.z.sub( PR.min.z ), PR.max.z.sub( Pw.z ) ) ) );
		const split = smoothstep( - 1.5, 1.5, PR.splitZ.sub( Pw.z ) );
		const probe = mix( cubeTexture( PR.p[ 2 ].rt.texture, R ).level( lod.add( 1 ).max( 2.4 ) ).rgb.mul( P.free ),
			mix( cubeTexture( PR.p[ 0 ].rt.texture, hit.sub( PR.p[ 0 ].pos ) ).level( lod ).rgb, cubeTexture( PR.p[ 1 ].rt.texture, hit.sub( PR.p[ 1 ].pos ) ).level( lod ).rgb, split ), inBox );
		// a probe is a stand-in, not an image (no parallax, one viewpoint): keep its light and dark, mute its colour so
		// stained glass and shop signs do not tint paving they could never be mirrored in
		const env = mix( probe, vec3( luminance( probe ) ), P.tame ).mul( PR.gain ).mul( P.env );
		const refl = ssrPass ? env.mul( oneMinus( ssrPass.a.mul( P.ssr ).saturate() ) ).add( ssrPass.rgb.mul( P.ssr ) ) : env;
		// the surface is seen through the night haze, and so is its reflection; crevices do not mirror the sky
		const thin = exp( max( Pw.y, 0 ).div( FOG.height ).negate() ).mul( 0.8 ).add( 0.2 );
		const fx = length( vP ).mul( FOG.density ).mul( thin );
		return refl.mul( w ).mul( exp( fx.mul( fx ).negate() ) ).mul( aoNode ? mix( float( 1 ), aoNode, 0.7 ) : 1 );

	};

	const aoTex = aoPass.getTextureNode();
	const grade = ( lit ) => {

		const bl = bloom( lit, 1.0, 0.55, 1.1 );
		let comp = lit.add( bl.rgb.mul( P.bloom ) );
		const vg = screenUV.sub( 0.5 );
		comp = comp.mul( oneMinus( dot( vg, vg ).mul( P.vignette ).mul( 1.6 ) ) ).mul( P.tint ).add( P.lift );
		// grade in scene-linear, pivoting on mid grey, before tone mapping
		comp = pow( max( comp, 0 ).div( 0.18 ), vec3( P.contrast ) ).mul( 0.18 );
		comp = mix( vec3( dot( comp, vec3( 0.2126, 0.7152, 0.0722 ) ) ), comp, P.saturation ).max( 0 );
		const gr = fract( sin( dot( screenUV.mul( 917.3 ).add( fract( time ) ), vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) ).sub( 0.5 );
		comp = comp.add( gr.mul( P.grain ).mul( comp.add( 0.02 ) ) );
		return { node: vec4( comp, 1 ), bloom: bl };

	};

	const withRefl = ( c, aoNode = null ) => ( ssrPass || envProbe ? c.add( reflection( aoNode ) ) : c );
	const graded = grade( withRefl( colorTex.rgb.mul( mix( float( 1 ), aoTex.r, P.ao.mul( oneMinus( mrTex.b ) ) ) ), aoTex.r ) );
	const finalNode = graded.node, bloomPass = graded.bloom;
	let noAO = null; // same grade without the GTAO pass (phone tier), built on first use
	// tone map + encode explicitly so FXAA runs on display values and the result can be dithered
	// (dark night gradients band badly at 8 bits otherwise)
	pipeline.outputColorTransform = false;
	const dither = vec4( vec3( interleavedGradientNoise( screenCoordinate.xy ).sub( 0.5 ).div( 160 ) ), 0 );
	const display = ( n, aa = true ) => ( aa ? fxaa( renderOutput( n ) ) : renderOutput( n ) ).add( dither );
	pipeline.outputNode = display( finalNode );
	const plainNode = vec4( colorTex.rgb, 1 );

	// --- lantern lights / fx bookkeeping
	const lanterns = []; // {light, base, flicker, phase}
	const flames = [];
	const partSets = [];
	let captureRT = null;

	const eng = {
		renderer, scene, camera, lib, assets, pipeline, sky, hemi, moon, moonDir, P, FOG, aoPass, bloomPass, ssrPass, isWebGPU,
		lanterns, flames, partSets, width, height,

		setSize( w, h, pr = pixelRatio ) {

			eng.width = w; eng.height = h;
			renderer.setPixelRatio( pr );
			renderer.setSize( w, h, false );
			camera.aspect = w / h;
			camera.updateProjectionMatrix();

		},

		// instantiate the lights / flames / halos of a loaded asset (sockets) under its root
		activate( inst, { lights = true, fx = true, halos = true, shadowLights = 0, intensity = 1 } = {} ) {

			const made = [];
			let nShadow = 0;
			for ( const s of inst.sockets ) {

				if ( s.kind === 'light' && lights ) {

					const pr = s.props;
					const L = new THREE.PointLight( new THREE.Color( pr.color || '#ffbf6a' ), ( pr.intensity ?? 40 ) * intensity, pr.range ?? 9, 2 );
					L.position.copy( s.position );
					if ( ( pr.shadow || nShadow < shadowLights ) && nShadow < Math.max( shadowLights, 0 ) ) {

						L.castShadow = true;
						L.shadow.mapSize.set( 1024, 1024 );
						L.shadow.bias = - 0.002;
						L.shadow.normalBias = 0.02;
						L.shadow.camera.near = pr.shadow_near ?? 0.03;
						L.shadow.camera.far = pr.range ?? 9;
						nShadow ++;

					}

					inst.root.add( L );
					const rec = { light: L, base: L.intensity, flicker: pr.flicker ?? 0.1, phase: ( lanterns.length * 2.399 ) % 6.283 };
					lanterns.push( rec );
					made.push( L );
					if ( halos ) {

						const hsz = Math.sqrt( ( pr.intensity ?? 40 ) / 40 ) * 1.5;
						const hl = createHalo( pr.color || '#ffbf6a', hsz, 0.1 );
						hl.position.copy( s.position );
						inst.root.add( hl );
						rec.halo = hl;
						made.push( hl );

					}

				} else if ( s.kind === 'fx' && fx ) {

					const pr = s.props;
					if ( pr.fx === 'flame' || pr.kind === 'flame' || s.name.includes( 'flame' ) ) {

						const fl = createFlame( lib.noise, { size: pr.size ?? 0.09, color: pr.color || '#ffb860', glow: pr.glow ?? 9,
							phase: ( flames.length * 0.618 ) % 1, speed: pr.speed ?? 1 } );
						fl.position.copy( s.position );
						inst.root.add( fl );
						flames.push( fl );
						made.push( fl );

					}

				}

			}

			if ( inst.parts.length ) partSets.push( inst.parts );
			inst.active = made;
			return made;

		},

		deactivate( inst ) {

			for ( const o of inst.active || [] ) {

				o.parent?.remove( o );
				if ( o.isLight ) {

					const i = lanterns.findIndex( ( r ) => r.light === o );
					if ( i >= 0 ) lanterns.splice( i, 1 );
					o.dispose?.();

				} else {

					const i = flames.indexOf( o );
					if ( i >= 0 ) flames.splice( i, 1 );
					o.material?.dispose?.();

				}

			}

			const pi = partSets.indexOf( inst.parts );
			if ( pi >= 0 ) partSets.splice( pi, 1 );
			inst.active = [];

		},

		// (re)capture reflection probe i from the scene as it is now; surfaces that sample the probes sit out
		captureProbe( i ) {

			syncLights();

			const PR = lib.probe, pr = PR.p[ i ], hidden = [], swapped = [];
			scene.traverseVisible( ( o ) => {

				// halo / flame sprites write through the scene pass's MRT layout and cannot compile for the cube target
				if ( ( o.isMesh && ( PR.mats.has( o.material ) || o.name === 'contact' ) ) || o.isSprite ) hidden.push( o );
				else if ( o.isMesh && ! Array.isArray( o.material ) ) {

					// blended library materials (glass, falling water) name the scene pass's extra MRT outputs and some
					// sample the probe being drawn: they are captured through a variant that does neither
					const v = lib.probeSafe( o.material );
					if ( v ) {

						swapped.push( o, o.material );
						o.material = v;

					}

				}

			} );
			for ( const o of hidden ) o.visible = false;
			pr.cam.position.copy( pr.pos.value );
			pr.cam.updateMatrixWorld( true );
			const rt0 = renderer.getRenderTarget();
			pr.cam.update( renderer, scene );
			renderer.setRenderTarget( rt0 );
			for ( const o of hidden ) o.visible = true;
			for ( let k = 0; k < swapped.length; k += 2 ) swapped[ k ].material = swapped[ k + 1 ];
			PR.gain.value = 1;

		},

		// t seconds: light flicker + swinging parts
		update( t, wind = 1 ) {

			for ( const r of lanterns ) {

				const f = r.flicker;
				if ( f > 0 ) {

					const n = Math.sin( t * 11.3 + r.phase ) * 0.35 + Math.sin( t * 23.7 + r.phase * 3.1 ) * 0.2 + Math.sin( t * 5.1 + r.phase * 1.7 ) * 0.45;
					r.light.intensity = r.base * ( 1 + f * n );

				}

			}

			for ( const ps of partSets ) animateParts( ps, t, wind );
			sky.position.copy( camera.position );

		},

		// look presets. night: lantern-lit town look. studio: neutral daylight for material/form checks.
		setRig( name, o = {} ) {

			const SU = sky.userData.U;
			if ( name === 'night' ) {

				SU.gain.value = 1;
				hemi.color.setRGB( 0.33, 0.40, 0.60 ); hemi.groundColor.setRGB( 0.16, 0.15, 0.2 ); // grey-blue moonlight (0.26, 0.38, 0.78 made every high shadow-side wall cobalt), same luminance
				hemi.intensity = o.ambient ?? 0.4;
				moon.color.setRGB( 0.62, 0.68, 0.87 );
				moon.intensity = o.moon ?? 0.6;
				moon.position.copy( moonDir ).multiplyScalar( 60 ).add( moon.target.position );
				renderer.toneMappingExposure = o.exposure ?? 1.25;
				P.bloom.value = o.bloom ?? 0.32;
				P.ao.value = o.ao ?? 0.85;
				P.vignette.value = o.vignette ?? 0.35;
				// haze: how much of the town's own light the night air carries (lifts and warms the fog colour, so far
				// roofs step back from near ones instead of sinking into the sky)
				FOG.color.value.setRGB( 0.016, 0.026, 0.058 ).lerp( new THREE.Color( 0.085, 0.07, 0.075 ), o.haze ?? 0 );
				FOG.density.value = o.fog ?? 0.0;
				SU.glowGain.value = o.skyGlow ?? 1.0;
				SU.cover.value = o.cover ?? 0.46;
				SU.cloudGain.value = o.cloudGain ?? 1.0;
				SU.horizon.value.setRGB( 0.018, 0.034, 0.085 );
				SU.cloud.value.setRGB( 0.05, 0.075, 0.14 );
				SU.clouds.value = o.clouds ?? 1.0;

			} else if ( name === 'studio' ) {

				SU.gain.value = 0.0;
				hemi.color.setRGB( 0.75, 0.8, 0.9 ); hemi.groundColor.setRGB( 0.42, 0.4, 0.38 );
				hemi.intensity = o.ambient ?? 1.1;
				moon.color.setRGB( 1.0, 0.96, 0.9 );
				moon.intensity = o.key ?? 2.6;
				const kd = o.keyDir || new THREE.Vector3( - 0.5, 0.75, 0.55 );
				moon.position.copy( kd ).normalize().multiplyScalar( 60 ).add( moon.target.position );
				renderer.toneMappingExposure = o.exposure ?? 0.8;
				P.bloom.value = 0.04;
				P.ao.value = o.ao ?? 0.8;
				P.vignette.value = 0.0;
				FOG.density.value = o.fog ?? 0;
				if ( o.fogColor ) FOG.color.value.set( o.fogColor );

			}

			eng.rig = name;

		},

		setBackground( c ) {

			scene.background = c === null ? null : new THREE.Color( c );
			sky.visible = c === null;

		},

		// fit the moon/key shadow camera around a box
		// tight: fit the light-space rectangle of the box (not its bounding sphere), keep the depth range short and
		// derive the biases and the filter radius from the texel size the fit ends up with (a scene-sized map: the
		// receiver is pushed two texels along its normal, which is what stops the texel grid printing on walls)
		fitShadow( box, pad = 1.5, { tight = false } = {} ) {

			const c = box.getCenter( new THREE.Vector3() );
			const r = box.getSize( new THREE.Vector3() ).length() * 0.5 + pad;
			const dir = moon.position.clone().sub( moon.target.position ).normalize();
			const sc = moon.shadow.camera;
			moon.target.position.copy( c );
			if ( tight ) {

				const right = new THREE.Vector3( 0, 1, 0 ).cross( dir ).normalize(), up = dir.clone().cross( right );
				let hx = 0, hy = 0;
				const v = new THREE.Vector3();
				for ( let i = 0; i < 8; i ++ ) {

					v.set( i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z ).sub( c );
					hx = Math.max( hx, Math.abs( v.dot( right ) ) ); hy = Math.max( hy, Math.abs( v.dot( up ) ) );

				}

				hx += pad; hy += pad;
				const D = r * 1.5 + 40; // casters this far towards the moon still reach the map (towers outside the box)
				moon.position.copy( c ).addScaledVector( dir, D );
				sc.left = - hx; sc.right = hx; sc.top = hy; sc.bottom = - hy;
				sc.near = 0.1; sc.far = D + r;
				const texel = Math.max( 2 * hx / moon.shadow.mapSize.x, 2 * hy / moon.shadow.mapSize.y );
				moon.shadow.normalBias = texel * 2.0;
				moon.shadow.bias = - texel * 1.2 / ( sc.far - sc.near );
				moon.shadow.radius = 2.2;
				eng.shadowTexel = texel;

			} else {

				moon.position.copy( c ).addScaledVector( dir, r * 3 );
				sc.left = - r; sc.right = r; sc.top = r; sc.bottom = - r;
				sc.near = 0.1; sc.far = r * 6;

			}

			sc.updateProjectionMatrix();
			moon.target.updateMatrixWorld();

		},

		setPost( on, { ao: withAO = true } = {} ) {

			if ( on && ! withAO ) noAO ??= grade( withRefl( colorTex.rgb ) ).node;
			pipeline.outputNode = on ? display( withAO ? finalNode : noAO ) : display( plainNode, false );
			pipeline.needsUpdate = true;

		},

		// three advances its node frame (per-frame pass updates, time) only inside setAnimationLoop;
		// when frames are driven by hand (render service) it has to be ticked explicitly.
		manual: false,
		tick() {

			syncLights();
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

		// offscreen render -> ImageData (sRGB, top row first). Works without canvas presentation
		// (hidden tabs, headless), which is what the review render service needs.
		async capture() {

			const pr = renderer.getPixelRatio();
			const w = Math.round( eng.width * pr ), h = Math.round( eng.height * pr );
			if ( ! captureRT || captureRT.width !== w || captureRT.height !== h ) {

				captureRT?.dispose();
				// the pipeline already writes display-encoded values: keep the target a plain unorm (no second sRGB encode)
				captureRT = new THREE.RenderTarget( w, h, { type: THREE.UnsignedByteType, colorSpace: THREE.NoColorSpace, depthBuffer: false } );

			}

			if ( ! isWebGPU ) {

				// WebGL2 fallback: read the drawing buffer straight after the draw. three's async read-back polls a
				// fence that never signals while the page is driven frame by frame (capture mode), so it would stall.
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

			eng.tick();
			renderer.setRenderTarget( captureRT );
			pipeline.render();
			renderer.setRenderTarget( null );
			const px = await renderer.readRenderTargetPixelsAsync( captureRT, 0, 0, w, h );
			const out = new Uint8ClampedArray( w * h * 4 );
			if ( px.length === out.length ) out.set( px );
			else {

				const flip = ! isWebGPU;
				const stride = isWebGPU ? Math.ceil( w * 4 / 256 ) * 256 : w * 4;
				for ( let y = 0; y < h; y ++ ) out.set( px.subarray( y * stride, y * stride + w * 4 ), ( flip ? h - 1 - y : y ) * w * 4 );

			}

			if ( ! isWebGPU && px.length === out.length ) {

				for ( let y = 0; y < h >> 1; y ++ ) {

					const a = y * w * 4, b = ( h - 1 - y ) * w * 4;
					const t = out.slice( a, a + w * 4 );
					out.copyWithin( a, b, b + w * 4 );
					out.set( t, b );

				}

			}

			return new ImageData( out, w, h );

		},
	};

	eng.setRig( 'night' );
	return eng;

}
