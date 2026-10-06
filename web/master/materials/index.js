// The materials of the town: five kinds of surface, and nothing else.
//
// All lighting is baked into the vertices, so every material is an unlit one (MeshBasicNodeMaterial) whose
// colour node is "texture x baked light" plus the little that turns with the eye. The whole town draws with
// six programs: opaque, opaque with light per pixel (the fine level), glass, water, falling water, leaves -
// and the same six again drawn through a dither while a cell changes its level.
//
//   inputs.js    what a vertex brings, the fog          layers.js    the texture arrays
//   ground.js    contact occlusion, ground shadows      pull.js      glowing cards kept in front
//   lightgrid.js + glints.js   the lanterns' highlights uniforms.js  the strengths, in one table
//   surface.js ( perpixel.js, wet.js )   glass.js   water.js   fall.js   leaves.js   fading.js
//
// Build order matters once: stage.mirrorOut (mirror.js) and stage.moonDir (addSky) must be set before this is
// called. Nothing here needs a call per frame - time, the camera and the per-mesh numbers (fade, pull) reach
// the shaders through three.js's own uniform updates.

import { createInputs } from './inputs.js';
import { createLayerSampler, createNormalMaps, perLayer } from './layers.js';
import { createLookUniforms } from './uniforms.js';
import { createPull } from './pull.js';
import { createGroundMaps } from './ground.js';
import { createLightGrid } from './lightgrid.js';
import { createGlints } from './glints.js';
import { createSurface } from './surface.js';
import { createGlass } from './glass.js';
import { createWater } from './water.js';
import { createFall } from './fall.js';
import { createLeaves } from './leaves.js';
import { createFading } from './fading.js';

// the kind of a primitive (mesh.userData.kind, from the file)
export const KIND = { OPAQUE: 0, GLASS: 1, WATER: 2, FALL: 3, CUT: 4 };

// stage: stage.js; bake: the manifest in the file (scenes[ 0 ].extras.baked); tier: its row of the tier table;
// layers: the first texture array
export function createTownMaterials( { stage, bake, tier, layers } ) {

	const U = createLookUniforms();
	const inputs = createInputs( { stage, bake } );
	const { TX, sample } = createLayerSampler( layers, inputs.layer );
	const NR = createNormalMaps( bake, tier );
	const { PULL, pullAll, pulled } = createPull();
	const { GAO, litGround } = createGroundMaps( { bake, lit: inputs.lit } );
	const grid = createLightGrid( { stage, bake, tier } );
	const glints = createGlints( { stage, bake, grid, U } );

	// what the makers share
	const ctx = {
		stage, bake, U, inputs, sample, NR, GAO, litGround, glints, pulled,
		tx: sample( inputs.tiles ), // the layer's colour and height at this pixel
		relief: perLayer( bake.relief, inputs.layer ).mul( 2 ).mul( U.rscale ), // metres of relief in the layer's alpha
	};

	const plain = { [ KIND.OPAQUE ]: createSurface( ctx, false ), [ KIND.GLASS ]: createGlass( ctx ), [ KIND.WATER ]: createWater( ctx ), [ KIND.FALL ]: createFall( ctx ), [ KIND.CUT ]: createLeaves( ctx ) };
	const fine = { ...plain, [ KIND.OPAQUE ]: createSurface( ctx, true ) };
	let plainFading = null, fineFading = null;
	if ( tier.fade > 0 ) {

		const fading = createFading();
		plainFading = {};
		for ( const k in plain ) plainFading[ k ] = fading( plain[ k ] );
		fineFading = { ...plainFading, [ KIND.OPAQUE ]: fading( fine[ KIND.OPAQUE ] ) };

	}

	// Give a primitive's mesh its material. A mesh whose vertices say where their light comes from (`_g`: the
	// fine level) takes the per-pixel variant; userData.mat / fmat are what levels.js swaps between during a fade.
	const dress = ( mesh ) => {

		const kind = mesh.userData.kind || KIND.OPAQUE, dir = mesh.geometry.hasAttribute( '_g' );
		mesh.material = mesh.userData.mat = ( dir ? fine : plain )[ kind ];
		if ( plainFading ) mesh.userData.fmat = ( dir ? fineFading : plainFading )[ kind ];

	};

	return { dress, TX, NR, GAO, U, PULL, pullAll, grid };

}
