// The stage: renderer, scene, camera, and the two things every material of the town shares - the night air
// (fog) and the way light is taken up (the HDR lift of what glows, the highlight knee).
//
// Nothing here knows about the file. The town's materials (materials/), the sky and the post chain (post.js) all
// read the same uniforms from the object this returns, so one look is set in one place (look.js) and holds for
// the whole frame: a wall, its image in the wet paving and the bloom around a lamp fog and roll off alike.

import * as THREE from 'three';
import { uniform, float, mix, step, max, min, exp, pow, length, oneMinus, positionView, positionWorld, cameraPosition } from 'three/tsl';
import { createSky, MaterialLib } from '../engine/engine.js';

export async function createStage( { canvas, fov = 62, forceWebGL = false, gpuTime = false } ) {

	// Reversed float depth: a 24 bit buffer with the near plane at 8 cm resolves 4 mm at 60 m, and the lit rooms
	// behind the windows are cards a few millimetres behind their openings - they shimmered as one walked.
	// HDR output where the display has range above white (WebGPU only): the canvas is 16 bit float and the tone
	// curve in post.js lets the lamps go up there.
	const hdrWanted = ! forceWebGL && !! navigator.gpu && matchMedia( '(dynamic-range: high)' ).matches;
	const renderer = new THREE.WebGPURenderer( { canvas, antialias: false, alpha: false, forceWebGL, reversedDepthBuffer: true, trackTimestamp: gpuTime,
		...( hdrWanted ? { outputType: THREE.HalfFloatType } : {} ) } );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	await renderer.init();
	const isWebGPU = renderer.backend.isWebGPUBackend === true;
	const reversed = renderer.reversedDepthBuffer === true; // (WebGL takes it back where EXT_clip_control is missing)
	const hdrOut = hdrWanted && isWebGPU && extendedRange( canvas );

	// A camera as the renderer leaves it after its first frame: the mirror image copies the projection before that
	// frame, and the renderer only rebuilds the projection of cameras it has not marked yet.
	const ownCamera = ( c ) => { c.coordinateSystem = renderer.coordinateSystem; c._reversedDepth = reversed; c.updateProjectionMatrix(); return c; };

	const scene = new THREE.Scene();
	const camera = ownCamera( new THREE.PerspectiveCamera( fov, 16 / 9, 0.08, 1500 ) );

	// --- the air. Two layers that thin out with height: a haze (`density`, squared in the exponent, so the first
	// tens of metres stay clear) and a ground mist (`mist`, plain extinction). Each is taken along the whole line
	// of sight (fogPath: the length of the line weighted by the layer's density on the way), not by the height of
	// the point looked at: a roof seen from the street is behind the mist it is seen through.
	const FOG = { color: uniform( new THREE.Color( 0.012, 0.015, 0.024 ) ), density: uniform( 0 ), height: uniform( 18 ), mist: uniform( 0 ), mistHeight: uniform( 5 ) };
	const fogPath = ( dist, y0, dy, H ) => {

		const t = dy.div( H ).toVar();
		const ts = t.add( step( 0, t ).mul( 2e-3 ).sub( 1e-3 ) ); // (never 0)
		const a = y0.div( H ).toVar();
		return dist.mul( exp( a.negate() ).sub( exp( a.add( ts ).negate() ) ) ).div( ts );

	};
	const fogDepth = ( dist, y0, dy, F = FOG ) => {

		const haze = fogPath( dist, y0, dy, F.height ).mul( F.density );
		return haze.mul( haze ).add( fogPath( dist, y0, dy, F.mistHeight ).mul( F.mist ) );

	};
	const fogAmount = () => oneMinus( exp( fogDepth( length( positionView ), cameraPosition.y, positionWorld.y.sub( cameraPosition.y ) ).negate() ) );

	// --- light. P: what the grade and the materials read. knee: lit surfaces roll off towards a ceiling (`knee`),
	// glowing ones towards a higher one (`kneeGlow`, times `lift` for a neon tube).
	const P = {
		vignette: uniform( 0.35 ), grain: uniform( 0.012 ), tint: uniform( new THREE.Color( 1, 1, 1 ) ), lift: uniform( 0 ),
		contrast: uniform( 1 ), saturation: uniform( 1 ), ssr: uniform( 1 ), knee: uniform( 0 ), kneeGlow: uniform( 3 ),
	};
	const knee = ( c, glowing, lift = 1 ) => {

		const m = mix( P.knee, max( P.kneeGlow, P.knee ).mul( lift ), glowing );
		const t = m.mul( 0.35 );
		const Lm = max( max( c.r, c.g ), max( c.b, 1e-4 ) );
		const x = Lm.sub( t ).max( 0 ).div( m.sub( t ).max( 1e-4 ) );
		const Lk = min( Lm, t ).add( m.sub( t ).mul( x.div( x.add( 1 ) ) ) );
		return c.mul( mix( float( 1 ), Lk.div( Lm ), step( 0.001, P.knee ) ) );

	};

	// What gives light is as bright as a light. The bake's glowing surfaces were set to read on a screen without
	// range above white - the brightest a stop over the paving under it - and a bloom that conserves energy has
	// nothing to spread with that. So a glowing surface is taken back out of the squeeze: x `glow`, what was
	// already over white to the power `hot` more, `top` at most; a mantle behind its pane x `core`, a flame
	// x `flame`, a coloured sign x `neon`. `peak`: how far over white an HDR canvas shows the brightest pixel.
	const HDR = { on: true, out: hdrOut, glow: uniform( 1.5 ), hot: uniform( 2.5 ), top: uniform( 48 ), core: uniform( 8 ), flame: uniform( 4 ), neon: uniform( 2.5 ),
		amb: uniform( 1 ), lamp: uniform( 1 ), peak: uniform( 4 ) };
	const emit = ( c, glowing ) => {

		const Lm = max( max( c.r, c.g ), max( c.b, 1e-4 ) );
		const k = min( HDR.glow.mul( pow( max( Lm, 1 ), HDR.hot ) ), max( HDR.top.div( Lm ), 1 ) );
		return c.mul( mix( float( 1 ), k, glowing ) );

	};

	const stage = {
		renderer, scene, camera, ownCamera, isWebGPU, reversed, FOG, fogDepth, fogAmount, P, knee, HDR, emit,
		noise: new MaterialLib().noise, // the engine's tiling noise (sky, flames, water)
		sky: null, moonDir: null, probe: null, mirror: null, mirrorOut: null, // (the sky: addSky(); the mirror: mirror.js)

		setSize( width, height, pixelRatio ) {

			renderer.setPixelRatio( pixelRatio );
			renderer.setSize( width, height, false );
			camera.aspect = width / height;
			camera.updateProjectionMatrix();

		},
	};
	return stage;

}

// The sky is a dome standing around the town (every cell is within 365 m of the middle) and in the same fog:
// clear overhead, gone into the mist low over the roofs. It has a depth like any front, so the fog's scattering
// in post.js needs no special case for it.
export function addSky( stage, moonDir ) {

	stage.moonDir = new THREE.Vector3( ...moonDir ).normalize();
	stage.sky = createSky( stage.moonDir, stage.noise, { shell: 600, fog: ( c ) => mix( c, stage.FOG.color, stage.fogAmount() ) } );
	stage.scene.add( stage.sky );
	return stage.sky;

}

// Did the canvas get the extended range? (A browser that does not know the tone mapping mode leaves it out.)
function extendedRange( canvas ) {

	try {

		const context = canvas.getContext( 'webgpu' ), config = context && context.getConfiguration ? context.getConfiguration() : null;
		return config ? !! ( config.toneMapping && config.toneMapping.mode === 'extended' ) : true;

	} catch ( e ) {

		return false;

	}

}
