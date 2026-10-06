// The look of the town's materials: every strength a material reads, in one table of uniforms.
//
// They are uniforms and not numbers so that one of them can be turned while the page runs without compiling
// a material again. The values are the ones the bake was judged with; only `fineR` moves by itself
// (levels.js: how far the fine level reaches this frame).

import { uniform } from 'three/tsl';

export function createLookUniforms() {

	return {
		// highlights: the lanterns' glints (with the mirror pass they share the paving with the mirrored town),
		// the same on a surface with its own normal map, the sky's and the town's share of what a gloss mirrors
		glint: uniform( 0.6 ), spec: uniform( 0.7 ), sheen: uniform( 1.0 ), town: uniform( 0.4 ),
		// the town after rain: the film on the stones, its ripples, how dark a drained crown and a flooded joint are,
		// how far a nearly level facet counts as level
		wetFlat: uniform( 1 ), ripple: uniform( 0.03 ), crown: uniform( 1.0 ), sunk: uniform( 0.9 ), level: uniform( 1 ),
		// the mirror image in it: strength, how flat the film lies, the share of drained stone, of baked shadow, of a
		// joint, and the footprint its height is filtered over (times the pixel's long side)
		mirror: uniform( 1 ), mflat: uniform( 1 ), mdry: uniform( 0.8 ), mshade: uniform( 0.3 ), joint: uniform( 0.8 ), mband: uniform( 3 ),
		// light per pixel (the fine level): its strength, the scale of a layer's relief, the normal map's strength and
		// the sign of its green, and the distance all of it has faded out by
		dir: uniform( 1 ), rscale: uniform( 1 ), nmap: uniform( 1 ), nflip: uniform( - 1 ), fineR: uniform( 0 ),
		// lantern panes: the glare as a whole, its core, body and even glow, how much a pixel's footprint widens it;
		// clear panes: the burner's light on their dirt, its highlight and the ceilings of both lobes, the least dirt
		pane: uniform( 1 ), paneCore: uniform( 1.2 ), paneBody: uniform( 0.5 ), paneEven: uniform( 0.3 ), paneBlur: uniform( 0.25 ),
		paneLit: uniform( 0.3 ), paneSpec: uniform( 0.5 ), paneDirt: uniform( 0.4 ), paneCap: uniform( 12 ), paneStar: uniform( 40 ),
		// the soft near field of a burner on its own metalwork (0: the bare inverse square)
		headSoft: uniform( 0 ),
		// canal water: the height of its ripples, the lanterns' wide glow on it, the silt's light, mirror and roughness
		swell: uniform( 0.55 ), wglow: uniform( 1.5 ), wbody: uniform( 0.5 ), wmirror: uniform( 2.5 ), wrough: uniform( 0.05 ),
		// falling water: the lanterns' lines on it, its lit foam, how far its light is cooled to silver
		fall: uniform( 1 ), fallLit: uniform( 1 ), fallCool: uniform( 0.5 ),
	};

}
