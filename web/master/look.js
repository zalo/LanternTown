// The look of the night and the tier the page runs: the numbers, in one place.
//
// The lighting half of the look (ambient, moon, lanterns) is in the bake's vertices; what is live is the air, the
// sky, the exposure and the highlight roll-off. These are the walk page's values, not retuned: a picture from
// this page is meant to match the High demo's pixel for pixel.

import * as THREE from 'three';
import { haloCore } from '../engine/engine.js';

export const FOV = 62;

const LOOK = {
	exposure: 0.72, vignette: 0.35, contrast: 1, saturation: 1.03,
	knee: 2, kneeGlow: 2.8, // highlight roll-off: lit walls / glowing glass (scene-linear ceilings)
	fog: 0.0075, haze: 0.35, fogGain: 1, mist: 0, mistHeight: 5, // haze: how far the air's blue goes over to lamplit dust
	skyGlow: 4, cover: 0.36, cloudGain: 2.4, clouds: 1,
	glare: 24, // the centre of a lantern's glow
};

export function applyLook( stage ) {

	const { renderer, P, FOG, sky } = stage;
	renderer.toneMappingExposure = LOOK.exposure;
	P.vignette.value = LOOK.vignette; P.contrast.value = LOOK.contrast; P.saturation.value = LOOK.saturation;
	P.knee.value = LOOK.knee; P.kneeGlow.value = LOOK.kneeGlow;
	FOG.color.value.setRGB( 0.016, 0.026, 0.058 ).lerp( new THREE.Color( 0.085, 0.07, 0.075 ), LOOK.haze ).multiplyScalar( LOOK.fogGain );
	FOG.density.value = LOOK.fog; FOG.mist.value = LOOK.mist; FOG.mistHeight.value = LOOK.mistHeight;
	const S = sky.userData.U;
	S.gain.value = 1; S.glowGain.value = LOOK.skyGlow; S.cover.value = LOOK.cover; S.cloudGain.value = LOOK.cloudGain; S.clouds.value = LOOK.clouds;
	S.horizon.value.setRGB( 0.018, 0.034, 0.085 ); S.cloud.value.setRGB( 0.05, 0.075, 0.14 );
	haloCore.value = LOOK.glare;

}

// The tiers this file serves: the rows of the bake's own table (the exporter copies them into the file, so page
// and file cannot disagree about distances or texture sizes).
export const QUALITIES = [ 'high', 'medium' ];

export function pickQuality( query ) {

	return [ query.get( 'quality' ), localStorage.getItem( 'tsq.master.quality' ), 'high' ].find( ( q ) => QUALITIES.includes( q ) );

}

export function tierOf( tiers, quality ) {

	const tier = { ...tiers[ quality ] };
	// (a hand-held screen: its own frame buffer budget)
	const hand = matchMedia( '(pointer: coarse)' ).matches && Math.min( screen.width, screen.height ) < 600;
	if ( hand && tier.hand ) Object.assign( tier, tier.hand );
	return tier;

}

// pixel ratio 1 at most, less where the frame would go over the tier's pixel budget
export function pixelRatio( tier, width, height ) {

	return Math.max( 0.5, Math.min( devicePixelRatio || 1, tier.ratio || 1, Math.sqrt( tier.pixels / Math.max( 1, width * height ) ) ) );

}
