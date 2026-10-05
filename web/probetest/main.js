// Probe test: a courtyard of the project's built GLBs, lit by two real-time point lights, plus a light probe grid
// that was baked EITHER by Blender/Cycles (tools/probetest/bl_bake.py) OR by three's own LightProbeGrid.bake().
//
//   ?probes=blender | three | off      which probe data feeds three's LightProbeGrid (default blender)
//   ?cam=1 | 2 | orbit                 fixed cameras of out/probetest/scene.json, or mouse orbit (default 1)
//   ?figures=stations | move | off     grey test figures standing at the stations, or one walking the path
//   ?t=0..1                            with figures=move: freeze the walker at this point of the path
//   ?file=probes_blender               stem of the Blender bake in out/probetest/
//   ?bounces=2&cube=16                 three's own bake: extra bounce passes, cube face size
//   ?mat=standard | lambert            surface model (lambert = exactly the diffuse BSDF Cycles used)
//   ?shadows=1 | 0   ?exposure=1.5   ?helper=1 (three's probe spheres)   ?intensity=1 (probe grid gain)
//   ?measure=1                         read back the irradiance the SHADER reconstructs at scene.json's spots
//   ?refl=1&cam=3&figures=off          side test: a Cycles equirect capture (refl_probe.exr) as envMap of two chrome balls
//   ?still=1                          no nav, sets window.__probetest.ready when the frame is final
//
// Everything is linear Rec.709; the page shows linear * exposure -> sRGB, the same transfer as the Cycles stills.
// Asset space (scene.json) is Blender's: three (X, Y, Z) = (x, z, -y).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LightProbeGrid } from 'three/addons/lighting/LightProbeGrid.js';
import { LightProbeGridNode, ATLAS_PADDING } from 'three/addons/tsl/lighting/LightProbeGridNode.js';
import { LightProbeGridHelper } from 'three/addons/helpers/LightProbeGridHelper.js';

const Q = new URLSearchParams( location.search );
const BASE = '..';          // the page lives in /probetest/: '..' is the server root (out/, node_modules/), also behind /lp/<port>/
const MODE = Q.get( 'probes' ) || 'blender';
const CAM = Q.get( 'cam' ) || '1';
const FIGURES = Q.get( 'figures' ) || 'stations';
const FILE = Q.get( 'file' ) || 'probes_blender';
const EXPOSURE = parseFloat( Q.get( 'exposure' ) || '1.5' );
const STILL = Q.get( 'still' ) === '1';
const state = window.__probetest = { ready: false, mode: MODE, log: window.__log };
const hud = document.getElementById( 'hud' );
if ( STILL ) document.body.classList.add( 'still' );

const b2t = ( p ) => new THREE.Vector3( p[ 0 ], p[ 2 ], - p[ 1 ] );

// ---- probe data: file -> three's atlas layout -------------------------------------------------------------
// LightProbeGrid keeps 9 RGB radiance-SH coefficients per probe = 27 floats, in 7 RGBA texels: float k of the
// flat [coef][rgb] array goes to sub-volume k >> 2, channel k & 3. The 7 sub-volumes are stacked along Z of ONE
// 3D texture nx * ny * 7 (nz + 2); each has a padding slice (copy of its edge slice) at both ends.
function packAtlas( sh, nx, ny, nz ) {

	const P = ATLAS_PADDING, slices = nz + 2 * P, depth = 7 * slices;
	const data = new Uint16Array( nx * ny * depth * 4 );

	for ( let t = 0; t < 7; t ++ ) for ( let s = 0; s < slices; s ++ ) {

		const iz = Math.min( nz - 1, Math.max( 0, s - P ) );

		for ( let iy = 0; iy < ny; iy ++ ) for ( let ix = 0; ix < nx; ix ++ ) {

			const p = ix + iy * nx + iz * nx * ny;
			const o = ( ( ( t * slices + s ) * ny + iy ) * nx + ix ) * 4;

			for ( let c = 0; c < 4; c ++ ) {

				const k = t * 4 + c;
				data[ o + c ] = THREE.DataUtils.toHalfFloat( k < 27 ? sh[ p * 27 + k ] : 0 );

			}

		}

	}

	const tex = new THREE.Data3DTexture( data, nx, ny, depth );
	tex.format = THREE.RGBAFormat;
	tex.type = THREE.HalfFloatType;
	tex.minFilter = tex.magFilter = THREE.LinearFilter;
	tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
	tex.generateMipmaps = false;
	tex.needsUpdate = true;
	return tex;

}

// the reverse, from the GPU: three's own bake -> flat [probe][coef][rgb] (to compare the two bakes number by number)
async function readAtlas( renderer, grid ) {

	const { x: nx, y: ny, z: nz } = grid.resolution;
	const P = ATLAS_PADDING, slices = nz + 2 * P;
	const out = new Float32Array( nx * ny * nz * 27 );
	const stride = Math.ceil( nx * 8 / 256 ) * 256 / 2;        // readback rows are padded to 256 bytes (uint16 units)

	for ( let t = 0; t < 7; t ++ ) for ( let iz = 0; iz < nz; iz ++ ) {

		const px = await renderer.readRenderTargetPixelsAsync( grid._renderTarget, 0, 0, nx, ny, 0, t * slices + P + iz );

		for ( let iy = 0; iy < ny; iy ++ ) for ( let ix = 0; ix < nx; ix ++ ) for ( let c = 0; c < 4; c ++ ) {

			const k = t * 4 + c;
			if ( k < 27 ) out[ ( ix + iy * nx + iz * nx * ny ) * 27 + k ] = THREE.DataUtils.fromHalfFloat( px[ iy * stride + ix * 4 + c ] );

		}

	}

	return out;

}

// three's getShIrradianceAt, on the CPU (SphericalHarmonics3.getIrradianceAt is the same polynomial)
function shIrradiance( sh, o, n ) {

	const { x, y, z } = n, e = [ 0, 0, 0 ];
	const w = [ 0.886227, 2 * 0.511664 * y, 2 * 0.511664 * z, 2 * 0.511664 * x, 2 * 0.429043 * x * y, 2 * 0.429043 * y * z,
		0.743125 * z * z - 0.247708, 2 * 0.429043 * x * z, 0.429043 * ( x * x - y * y ) ];
	for ( let k = 0; k < 9; k ++ ) for ( let c = 0; c < 3; c ++ ) e[ c ] += w[ k ] * sh[ o + k * 3 + c ];
	return e;

}

function shTrilinear( sh, grid, p ) {

	const res = grid.resolution.toArray(), mn = grid.boundingBox.min.toArray(), mx = grid.boundingBox.max.toArray(), pa = p.toArray();
	const i0 = [], w = [];
	for ( let a = 0; a < 3; a ++ ) {

		const f = Math.min( 1, Math.max( 0, ( pa[ a ] - mn[ a ] ) / ( mx[ a ] - mn[ a ] ) ) ) * ( res[ a ] - 1 );
		i0[ a ] = Math.min( Math.floor( f ), res[ a ] - 2 );
		w[ a ] = f - i0[ a ];

	}

	const out = new Float32Array( 27 );
	for ( let dz = 0; dz < 2; dz ++ ) for ( let dy = 0; dy < 2; dy ++ ) for ( let dx = 0; dx < 2; dx ++ ) {

		const k = ( i0[ 0 ] + dx ) + ( i0[ 1 ] + dy ) * res[ 0 ] + ( i0[ 2 ] + dz ) * res[ 0 ] * res[ 1 ];
		const wt = ( dx ? w[ 0 ] : 1 - w[ 0 ] ) * ( dy ? w[ 1 ] : 1 - w[ 1 ] ) * ( dz ? w[ 2 ] : 1 - w[ 2 ] );
		for ( let j = 0; j < 27; j ++ ) out[ j ] += wt * sh[ k * 27 + j ];

	}

	return out;

}

// ---- irradiance the SHADER reconstructs at a point for a normal -------------------------------------------
// A white Lambert card facing n is drawn into a float target with only the probe grid as light: radiance = E / pi.
// LightProbeGridNode samples at position + normal * spacing / 2, so the card is put half a cell BEHIND the point:
// the sample then lands exactly on the point.
// cancelOffset = false leaves the card ON the point: that is what a real surface there gets (the sample is taken half
// a cell in front of it - through a thin wall, if one is closer than that).
async function shaderIrradiance( renderer, grid, points, cancelOffset = true ) {

	const rt = new THREE.RenderTarget( 4, 4, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: true } );
	const sc = new THREE.Scene();
	const mat = new THREE.MeshLambertNodeMaterial();
	mat.color.setRGB( 1, 1, 1 );
	const card = new THREE.Mesh( new THREE.PlaneGeometry( 0.2, 0.2 ), mat );
	sc.add( card );
	// a 2 mm wide view of the card's centre: the grid's gradient across a wall is steep enough for centimetres to show
	const cam = new THREE.OrthographicCamera( - 0.001, 0.001, 0.001, - 0.001, 0.1, 5 );
	const parent = grid.parent;
	sc.add( grid );
	const spacing = new THREE.Vector3( grid.width, grid.height, grid.depth ).divide( grid.resolution.clone().subScalar( 1 ) );
	const prevTarget = renderer.getRenderTarget();
	const out = [];

	for ( const { p, n } of points ) {

		card.position.copy( p );
		if ( cancelOffset ) card.position.sub( n.clone().multiply( spacing ).multiplyScalar( 0.5 ) );
		card.quaternion.setFromUnitVectors( new THREE.Vector3( 0, 0, 1 ), n );
		cam.position.copy( card.position ).add( n );
		cam.up.set( 0, Math.abs( n.y ) > 0.9 ? 0 : 1, Math.abs( n.y ) > 0.9 ? 1 : 0 );
		cam.lookAt( card.position );
		sc.updateMatrixWorld( true );
		renderer.setRenderTarget( rt );
		renderer.render( sc, cam );
		const px = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, 4, 4 );
		out.push( [ px[ 0 ] * Math.PI, px[ 1 ] * Math.PI, px[ 2 ] * Math.PI ] );

	}

	renderer.setRenderTarget( prevTarget );
	if ( parent ) parent.add( grid ); else sc.remove( grid );
	rt.dispose();
	return out;

}

// ---- scene ---------------------------------------------------------------------------------------------------
function makeMaterial( spec ) {

	if ( spec.kind === 'emit' ) {

		const m = new THREE.MeshBasicNodeMaterial( { side: THREE.DoubleSide } );
		m.color.setRGB( spec.emission[ 0 ], spec.emission[ 1 ], spec.emission[ 2 ] );
		return m;

	}

	const m = Q.get( 'mat' ) === 'lambert' ? new THREE.MeshLambertNodeMaterial( { side: THREE.DoubleSide } )
		: new THREE.MeshStandardNodeMaterial( { roughness: 1, metalness: 0, side: THREE.DoubleSide } );
	m.color.setRGB( spec.albedo[ 0 ], spec.albedo[ 1 ], spec.albedo[ 2 ] );
	return m;

}

function pathPoint( path, t ) {

	const seg = [];
	let total = 0;
	for ( let i = 0; i < path.length - 1; i ++ ) { seg.push( Math.hypot( path[ i + 1 ][ 0 ] - path[ i ][ 0 ], path[ i + 1 ][ 1 ] - path[ i ][ 1 ] ) ); total += seg[ i ]; }
	let d = Math.min( 1, Math.max( 0, t ) ) * total;
	for ( let i = 0; i < seg.length; i ++ ) {

		if ( d <= seg[ i ] || i === seg.length - 1 ) {

			const k = Math.min( 1, d / seg[ i ] );
			return [ path[ i ][ 0 ] + k * ( path[ i + 1 ][ 0 ] - path[ i ][ 0 ] ), path[ i ][ 1 ] + k * ( path[ i + 1 ][ 1 ] - path[ i ][ 1 ] ) ];

		}

		d -= seg[ i ];

	}

}

async function main() {

	const canvas = document.getElementById( 'gl' );
	const renderer = new THREE.WebGPURenderer( { canvas, antialias: true } );
	await renderer.init();
	state.backend = renderer.backend.isWebGPUBackend ? 'webgpu' : 'webgl2';
	renderer.setPixelRatio( 1 );
	renderer.setSize( innerWidth, innerHeight, false );
	renderer.toneMapping = THREE.LinearToneMapping;         // = colour * exposure, clipped; then sRGB
	renderer.toneMappingExposure = EXPOSURE;
	renderer.shadowMap.enabled = Q.get( 'shadows' ) !== '0';

	const S = await ( await fetch( `${BASE}/out/probetest/scene.json`, { cache: 'no-cache' } ) ).json();
	const scene = new THREE.Scene();
	scene.background = new THREE.Color().setRGB( S.sky[ 0 ], S.sky[ 1 ], S.sky[ 2 ] );

	// assets
	const loader = new GLTFLoader();
	const cache = new Map(), mats = new Map();
	const matFor = ( name, spec ) => { if ( ! mats.has( name ) ) mats.set( name, makeMaterial( spec ) ); return mats.get( name ); };
	let tris = 0;
	await Promise.all( S.instances.map( async ( it ) => {

		if ( ! cache.has( it.glb ) ) cache.set( it.glb, loader.loadAsync( `${BASE}/${it.glb}` ) );
		const gltf = await cache.get( it.glb );
		const root = gltf.scene.clone( true );
		const drop = [];
		root.traverse( ( o ) => {

			if ( ! o.isMesh ) return;
			const spec = S.materials[ o.material.name ];
			if ( ! spec || spec.kind === 'hide' ) { drop.push( o ); return; }
			o.material = matFor( o.material.name, spec );
			o.castShadow = it.id !== 'lamp_post_gas' && spec.kind !== 'emit';     // a lamp does not shadow on its own post (as in the Cycles scene)
			o.receiveShadow = true;
			tris += ( o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count ) / 3;

		} );
		for ( const o of drop ) o.removeFromParent();
		root.position.copy( b2t( it.pos ) );
		root.rotation.y = it.rz;
		scene.add( root );

	} ) );
	for ( const b of S.boxes ) {

		const mn = b2t( b.min ), mx = b2t( b.max );
		const size = new THREE.Vector3().subVectors( mx, mn );
		const m = new THREE.Mesh( new THREE.BoxGeometry( Math.abs( size.x ), Math.abs( size.y ), Math.abs( size.z ) ), matFor( 'box_' + b.name, { kind: 'diffuse', albedo: b.albedo } ) );
		m.position.addVectors( mn, mx ).multiplyScalar( 0.5 );
		m.castShadow = m.receiveShadow = true;
		scene.add( m );

	}

	// lights: Blender watts -> three: PointLight.power = watts (intensity = watts / 4 pi), inverse-square, no cut-off
	for ( const L of S.lights ) {

		const pl = new THREE.PointLight( new THREE.Color().setRGB( L.color[ 0 ], L.color[ 1 ], L.color[ 2 ] ), L.watts / ( 4 * Math.PI ), 0, 2 );
		pl.position.copy( b2t( L.pos ) );
		pl.castShadow = renderer.shadowMap.enabled;
		pl.shadow.mapSize.set( 1024, 1024 );
		// r186 compares PERSPECTIVE depth for point lights, so a bias is ~ bias * z^2 / near metres: with near 0.1 and
		// bias -0.004 the wall 10 m from a lamp cast no shadow at all. Near 0.5 (nothing but the lamp's own post is
		// that close, and the post does not cast) and a small bias: 2 mm at 2 m, 9 cm at 12 m.
		pl.shadow.camera.near = 0.5;
		pl.shadow.camera.far = 80;
		pl.shadow.bias = - 0.0003;
		pl.shadow.normalBias = 0.04;
		scene.add( pl );

	}

	// camera
	const camera = new THREE.PerspectiveCamera( 50, innerWidth / innerHeight, 0.05, 400 );
	const cs = S.cameras[ CAM ] || S.cameras[ '1' ];
	camera.fov = cs.fov;
	camera.position.copy( b2t( cs.eye ) );
	camera.lookAt( b2t( cs.target ) );
	camera.updateProjectionMatrix();
	let controls = null;
	if ( CAM === 'orbit' ) { controls = new OrbitControls( camera, canvas ); controls.target.copy( b2t( cs.target ) ); controls.update(); }

	// ---- the probe grid
	let grid = null, shBlender = null, meta = null;
	const G = S.grid;
	state.grid = G;
	if ( MODE === 'blender' || Q.get( 'measure' ) === '1' ) {

		try {

			meta = await ( await fetch( `${BASE}/out/probetest/${FILE}.json`, { cache: 'no-cache' } ) ).json();
			shBlender = new Float32Array( await ( await fetch( `${BASE}/out/probetest/${FILE}.bin`, { cache: 'no-cache' } ) ).arrayBuffer() );

		} catch ( e ) { if ( MODE === 'blender' ) throw e; }

	}

	if ( MODE !== 'off' ) {

		const res = MODE === 'blender' ? meta.grid.res : G.res;
		grid = new LightProbeGrid( G.size[ 0 ], G.size[ 1 ], G.size[ 2 ], res[ 0 ], res[ 1 ], res[ 2 ] );
		grid.position.fromArray( G.center );
		grid.intensity = parseFloat( Q.get( 'intensity' ) || '1' );
		grid.updateBoundingBox();
		scene.add( grid );
		const t0 = performance.now();

		if ( MODE === 'blender' ) {

			// Blender's coefficients straight into the texture three's own light node samples: no custom shader
			grid.texture = packAtlas( shBlender, res[ 0 ], res[ 1 ], res[ 2 ] );
			if ( renderer.library.getLightNodeClass( LightProbeGrid ) === null ) renderer.library.addLight( LightProbeGridNode, LightProbeGrid );
			state.probeBytes = shBlender.byteLength;

		} else {

			// three's normal path: cube captures of this scene on the GPU, + bounce passes
			grid.bake( renderer, scene, { cubemapSize: parseInt( Q.get( 'cube' ) || '16' ), near: 0.05, far: 200, bounces: parseInt( Q.get( 'bounces' ) || '2' ) } );
			await renderer.backend.device?.queue.onSubmittedWorkDone();

		}

		state.probeSetupMs = Math.round( performance.now() - t0 );
		if ( Q.get( 'helper' ) === '1' ) scene.add( new LightProbeGridHelper( grid, 0.12 ) );

	}

	// ---- dynamic objects (never part of a bake)
	const F = S.figure;
	const figMat = makeMaterial( { kind: 'diffuse', albedo: F.albedo } );
	const figure = () => {

		const g = new THREE.Group();
		const cap = new THREE.Mesh( new THREE.CapsuleGeometry( F.capsule_radius, F.capsule_height - 2 * F.capsule_radius, 12, 32 ), figMat );
		cap.position.y = F.capsule_height / 2;
		const sph = new THREE.Mesh( new THREE.SphereGeometry( F.sphere_radius, 48, 24 ), figMat );
		sph.position.y = F.sphere_z;
		for ( const m of [ cap, sph ] ) { m.castShadow = m.receiveShadow = true; g.add( m ); }
		return g;

	};

	let walker = null;
	if ( FIGURES === 'stations' ) for ( const p of S.station_pos ) { const f = figure(); f.position.copy( b2t( [ p[ 0 ], p[ 1 ], 0 ] ) ); scene.add( f ); }
	if ( FIGURES === 'move' ) { walker = figure(); scene.add( walker ); }

	// ---- ?refl=1: the reflection-probe side test. A Cycles equirect capture (bl_ref.py mode=refl) as the envMap of a
	// chrome ball hanging where it was captured; three pre-filters it (PMREM) itself.
	if ( Q.get( 'refl' ) === '1' && S.refl ) {

		const { EXRLoader } = await import( 'three/addons/loaders/EXRLoader.js' );
		const env = await new EXRLoader().loadAsync( `${BASE}/out/probetest/refl_probe.exr` );
		env.mapping = THREE.EquirectangularReflectionMapping;
		state.refl = { w: env.image.width, h: env.image.height, type: env.type };
		for ( const [ dx, rough ] of [ [ 0, 0.03 ], [ - 1.1, 0.35 ] ] ) {

			const m = new THREE.MeshStandardNodeMaterial( { metalness: 1, roughness: rough, envMap: env } );
			m.color.setRGB( 0.95, 0.95, 0.95 );
			const ball = new THREE.Mesh( new THREE.SphereGeometry( S.refl.radius, 64, 32 ), m );
			ball.position.copy( b2t( S.refl.pos ) ).x += dx;
			scene.add( ball );

		}

	}

	Object.assign( state, { scene, camera, renderer, grid } );      // for poking at it from the console
	state.tris = Math.round( tris );
	hud.textContent = `probes: ${ { blender: 'BLENDER / Cycles bake (' + FILE + ')', three: "three's own in-browser bake", off: 'OFF (direct light only)' }[ MODE ] }`
		+ `   cam ${CAM}   ${state.backend}` + ( grid ? `   grid ${grid.resolution.toArray().join( 'x' )}  setup ${state.probeSetupMs} ms` : '' );

	// ---- numbers
	if ( grid && Q.get( 'measure' ) === '1' ) {

		const pts = [], keys = [];
		for ( const sp of S.spots ) for ( const [ k, n ] of [ [ '+Y', [ 0, 0, 1 ] ], [ '-Y', [ 0, 0, - 1 ] ], [ 'wall', sp.wall ] ] ) { pts.push( { p: b2t( sp.pos ), n: b2t( n ).normalize() } ); keys.push( [ sp.name, k ] ); }
		const E = await shaderIrradiance( renderer, grid, pts );
		const Es = await shaderIrradiance( renderer, grid, pts, false );
		const res = {};
		keys.forEach( ( [ name, k ], i ) => {

			res[ name ] ??= { shader: {}, surface: {}, cpu: {} };
			res[ name ].shader[ k ] = E[ i ].map( ( v ) => + v.toFixed( 5 ) );
			res[ name ].surface[ k ] = Es[ i ].map( ( v ) => + v.toFixed( 5 ) );
			if ( MODE === 'blender' ) res[ name ].cpu[ k ] = shIrradiance( shTrilinear( shBlender, grid, pts[ i ].p ), 0, pts[ i ].n ).map( ( v ) => + v.toFixed( 5 ) );

		} );
		state.E = res;
		if ( MODE === 'three' ) {

			try { state.atlas = Array.from( await readAtlas( renderer, grid ) ); } catch ( e ) { state.atlasError = String( e ); }

		}

	}

	// ---- still capture: this project's headless Chrome does not composite the WebGPU canvas into CDP screenshots, so
	// the frame is copied out of the canvas in the same task it was rendered in (label burned in)
	// the frame is rendered into a linear half-float target (4x MSAA), read back, and given the canvas' transfer on the
	// CPU: min( 1, linear * exposure ) -> sRGB. Also returns the mean linear colour (for comparing with Cycles).
	state.snap = async ( label = true ) => {

		const w = canvas.width, h = canvas.height;
		const rt = new THREE.RenderTarget( w, h, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, samples: 4, depthBuffer: true } );
		renderer.setRenderTarget( rt );
		renderer.render( scene, camera );
		renderer.setRenderTarget( null );
		const px = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, w, h );
		rt.dispose();
		const stride = Math.ceil( w * 8 / 256 ) * 256 / 2;
		const img = new ImageData( w, h );
		const lut = new Float32Array( 65536 );
		for ( let i = 0; i < 65536; i ++ ) lut[ i ] = THREE.DataUtils.fromHalfFloat( i );
		const mean = [ 0, 0, 0 ];
		for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) for ( let c = 0; c < 3; c ++ ) {

			const lin = lut[ px[ y * stride + x * 4 + c ] ];
			mean[ c ] += lin;
			const v = Math.min( 1, Math.max( 0, lin * EXPOSURE ) );
			img.data[ ( y * w + x ) * 4 + c ] = Math.round( 255 * ( v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow( v, 1 / 2.4 ) - 0.055 ) );
			img.data[ ( y * w + x ) * 4 + 3 ] = 255;

		}

		state.meanLinear = mean.map( ( v ) => v / ( w * h ) );
		const c2 = document.createElement( 'canvas' );
		c2.width = w; c2.height = h;
		const g = c2.getContext( '2d' );
		g.putImageData( img, 0, 0 );
		if ( label ) { g.font = '14px monospace'; g.fillStyle = '#000a'; g.fillRect( 0, 0, g.measureText( hud.textContent ).width + 24, 30 ); g.fillStyle = '#f0e6d2'; g.fillText( hud.textContent, 12, 20 ); }
		return c2.toDataURL( 'image/png' );

	};

	// ---- loop
	let frames = 0;
	const clock = new THREE.Clock();
	renderer.setAnimationLoop( () => {

		if ( walker ) {

			const t = Q.has( 't' ) ? parseFloat( Q.get( 't' ) ) : ( clock.getElapsedTime() / 40 ) % 1;
			const p = pathPoint( S.path, t );
			walker.position.copy( b2t( [ p[ 0 ], p[ 1 ], 0 ] ) );

		}

		if ( controls ) controls.update();
		renderer.render( scene, camera );
		if ( ++ frames === 6 ) state.ready = true;

	} );
	addEventListener( 'resize', () => { renderer.setSize( innerWidth, innerHeight, false ); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); } );

	const nav = document.getElementById( 'nav' );
	const link = ( label, set ) => { const u = new URL( location.href ); for ( const k in set ) u.searchParams.set( k, set[ k ] ); const a = document.createElement( 'a' ); a.href = u.search; a.textContent = label; nav.appendChild( a ); };
	link( 'probes off', { probes: 'off' } ); link( 'Blender bake', { probes: 'blender' } ); link( 'three bake', { probes: 'three' } );
	link( 'cam 1', { cam: '1' } ); link( 'cam 2', { cam: '2' } ); link( 'orbit', { cam: 'orbit' } ); link( 'walk', { figures: 'move' } ); link( 'stations', { figures: 'stations' } );

}

main().catch( ( e ) => { state.failed = String( e && e.stack || e ); hud.textContent = 'FAILED ' + state.failed; console.error( e ); } );
