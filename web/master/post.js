// The frame after the scene is drawn: one chain, each step reading the one before.
//
//   scene (colour, wet, velocity, depth) -> + mirror image on wet paving -> TRAA -> natural bloom and the fog's
//   scattering -> grade -> tone curve -> dither
//
// - TRAA comes before the bloom: a lit slit narrower than a pixel then counts for the share of the pixel it
//   covers, and the bloom has nothing to flash with.
// - Bloom, scattering and the optional lens flare read one pyramid of blurred copies of the frame (natbloom.js):
//   the bloom blends every pixel towards the blur by `intensity`, the scattering blends it again by the share of
//   fog on the pixel's line of sight - a lamp down the street smears into the air, what stands close stays sharp.
//   That share is worked out from the depth, along the same path as the fog the materials mix in, and rides in the
//   alpha of the picture TRAA resolves, so on a silhouette it is averaged over the frames with the colour.
// - No pass here is per-material or per-light; nothing is added for a tier that the tier's table does not name.

import * as THREE from 'three';
import {
	pass, uniform, uv, vec2, vec3, vec4, float, Fn, mix, max, min, exp, length, dot, pow, sin, fract, oneMinus, log2, smoothstep,
	screenUV, screenCoordinate, time, interleavedGradientNoise, getViewPosition, renderOutput, toneMappingExposure, acesFilmicToneMapping,
} from 'three/tsl';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { naturalBloom } from '../engine/natbloom.js';

const BLOOM = 0.14, SPREAD = 0.7, SCATTER = 0.7; // the glow, which levels carry it (0 tight .. 1 wide), how much of the fog's share is scattered
const FLARE = 0.25; // the lens flare when it is switched on

export function createPost( stage, tier, mirror ) {

	const { renderer, scene, camera, P, HDR } = stage;
	const pipeline = new THREE.RenderPipeline( renderer );
	const scenePass = pass( scene, camera );
	scenePass.setMRT( mirror.targets() );
	scenePass.getTexture( 'wet' ).type = THREE.UnsignedByteType;
	const depth = scenePass.getTextureNode( 'depth' );

	let color = scenePass.getTextureNode( 'output' ).rgb.add( mirror.reflection( scenePass ).mul( P.ssr ) );

	// the share of fog on each pixel's line of sight, with strengths of its own (the haze squared in the exponent:
	// little in the first tens of metres, most of it by the far side of the square). The sky is a dome with a
	// depth like any front, so its share falls out of the same sum.
	const AIR = { density: uniform( 0.04 ), height: uniform( 18 ), mist: uniform( 0 ), mistHeight: stage.FOG.mistHeight, max: uniform( 0.9 ) };
	const projInv = uniform( camera.projectionMatrixInverse ), world = uniform( camera.matrixWorld );
	const share = Fn( () => {

		const u = uv(), d = depth.sample( u ).r;
		// (view position as in the mirror pass)
		const clip = projInv.mul( vec4( u.x.mul( 2 ).sub( 1 ), u.y.oneMinus().mul( 2 ).sub( 1 ), d, 1 ) );
		const Pv = ( stage.isWebGPU || stage.reversed ? clip.xyz.div( clip.w ) : getViewPosition( u, d, projInv ) ).toVar();
		const tau = stage.fogDepth( length( Pv ), world.mul( vec4( 0, 0, 0, 1 ) ).y, world.mul( vec4( Pv, 0 ) ).y, AIR );
		return min( oneMinus( exp( tau.negate() ) ), AIR.max );

	} )();

	const resolved = traa( vec4( color, share ), depth, scenePass.getTextureNode( 'velocity' ), camera );
	// (the share is in its alpha: a material that counts as opaque would write 1 there)
	if ( resolved._resolveMaterial ) resolved._resolveMaterial.blending = THREE.NoBlending;

	// (the tier's levels are for a picture 720 high; one more for every doubling, or the widest glow would shrink with the screen)
	const more = Math.max( 0, Math.round( Math.log2( renderer.getDrawingBufferSize( new THREE.Vector2() ).y / 720 ) ) );
	const bloom = naturalBloom( resolved.getTextureNode(), { levels: Math.min( 8, tier.nb + more ), divisor: tier.nbdiv || 2, dirt: null, flare: true } );
	bloom.intensity.value = BLOOM; bloom.spread.value = SPREAD; bloom.fog.value = SCATTER;
	bloom.flareIntensity.value = 0; // (its passes run only while this is over 0)
	color = bloom.rgb;

	// --- grade: vignette, tint, contrast about middle grey, saturation, grain
	const fromCentre = screenUV.sub( 0.5 );
	color = color.mul( oneMinus( dot( fromCentre, fromCentre ).mul( P.vignette ).mul( 1.6 ) ) ).mul( P.tint ).add( P.lift );
	color = pow( max( color, 0 ).div( 0.18 ), vec3( P.contrast ) ).mul( 0.18 );
	color = mix( vec3( dot( color, vec3( 0.2126, 0.7152, 0.0722 ) ) ), color, P.saturation ).max( 0 );
	const grain = fract( sin( dot( screenUV.mul( 917.3 ).add( fract( time ) ), vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) ).sub( 0.5 );
	color = color.add( grain.mul( P.grain ).mul( color.add( 0.02 ) ) );

	// --- the tone curve. Standard: ACES to 0..1 (the renderer's). Extended (an HDR canvas): the same picture up to
	// white, times a gain that rises with how far the scene is over white - 1 at white, `peak` four stops over it:
	// a lamp's mantle and its streak on the wet paving sit above the page's white, the lit walls where they were.
	pipeline.outputColorTransform = false;
	let shown;
	if ( HDR.out ) {

		const exposed = color.mul( toneMappingExposure ), top = max( max( exposed.r, exposed.g ), max( exposed.b, 1e-4 ) );
		const over = smoothstep( 0, 1, log2( max( top, 1 ) ).div( 4 ) );
		shown = renderOutput( vec4( acesFilmicToneMapping( color, toneMappingExposure ).mul( mix( float( 1 ), HDR.peak, over ) ), 1 ), THREE.NoToneMapping, THREE.SRGBColorSpace );

	} else shown = renderOutput( vec4( color, 1 ) );
	// (a dither under the 8 bit steps of the night sky)
	pipeline.outputNode = shown.add( vec4( vec3( interleavedGradientNoise( screenCoordinate.xy ).sub( 0.5 ).div( 160 ) ), 0 ) );

	return {
		pipeline, scenePass, bloom, AIR,
		render() { pipeline.render(); },
		set flare( on ) { bloom.flareIntensity.value = on ? FLARE : 0; },
		get flare() { return bloom.flareIntensity.value > 0; },
	};

}
