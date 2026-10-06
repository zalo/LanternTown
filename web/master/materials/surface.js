// The opaque surfaces: walls, paving, roofs, props - nearly every triangle of the town.
//
// Colour is the layer's texture times the baked light. On top comes what cannot be baked because it turns with
// the eye: wet cobbles, polished stone and glazed tiles mirror the nearest lanterns (glints.js) and a little of
// the sky, along the face normal roughened by the relief in the layer's alpha; where the baked light says "in
// shadow" the highlight goes too. Wet paving also tells the mirror pass what it mirrors (wet.js).
//
// Two variants: the plain one, and `dir` for the meshes whose vertices also say where their light comes from
// (the fine level): there the relief of a wall or a cobble is lit per pixel (perpixel.js).

import * as THREE from 'three';
import { vec3, float, mix, step, max, min, dot, sign, saturate, smoothstep, oneMinus } from 'three/tsl';
import { faceFrame, unit } from './frame.js';
import { perPixelLight } from './perpixel.js';
import { wetFilm } from './wet.js';

export function createSurface( ctx, dir ) {

	const { stage, U, tx, GAO, glints, pulled } = ctx;
	const { cv, layer, aux, glowing, V, fogged } = ctx.inputs;
	const lit = ctx.litGround; // the baked light under the ground's occlusion and shadows (ground.js)
	const material = new THREE.MeshBasicNodeMaterial( { fog: false } );
	material.name = dir ? 'baked (directional)' : 'baked';
	let col = tx.rgb.mul( lit );

	const frame = faceFrame( V );
	const Nf = frame.N0.mul( sign( dot( frame.N0, V ) ) );
	let N = unit( frame.bump( tx.a.mul( ctx.relief ) ) );
	const gloss = min( aux.mul( 2 ), 1 );
	let rough = mix( float( 0.5 ), float( 0.08 ), gloss );
	let f0 = float( 0.04 ), spec = null;
	const wetK = smoothstep( 0.2, 0.6, gloss ).mul( smoothstep( 0.9, 0.99, Nf.y ) ).mul( oneMinus( glowing ) ); // wet, level paving

	if ( dir ) {

		const fine = perPixelLight( ctx, { frame, N, Nf, gloss, rough, wetK } );
		( { N, rough, f0, spec } = fine );
		col = col.mul( mix( fine.light, vec3( 1 ), glowing ) );

	}

	const lum = dot( lit, vec3( 0.2126, 0.7152, 0.0722 ) );
	const shiny = ( spec ? max( smoothstep( 0.03, 0.3, gloss ), spec ) : smoothstep( 0.03, 0.3, gloss ) ).mul( oneMinus( glowing ) );
	const inLight = smoothstep( 0.008, 0.08, lum ); // (the highlights have no shadows: the baked light stands in)

	const wet = wetFilm( ctx, { frame, N, Nf, rough, lum, wetK } );
	( { N, rough } = wet );
	col = col.mul( wet.darken );
	material.mrtNode = wet.mrtNode;

	// what a gloss mirrors besides the lanterns: the sky where the reflected ray points up, the lit town (about as
	// bright as the light arriving here) where it stays low. Not on wet paving: the mirror pass has the real thing.
	const nv = saturate( dot( N, V ) );
	const ry = N.y.mul( nv ).mul( 2 ).sub( V.y );
	const env = mix( lit.mul( U.town ), vec3( 0.018, 0.034, 0.085 ).mul( U.sheen ), smoothstep( 0.02, 0.45, ry ) ).mul( oneMinus( wetK ) ).mul( wet.fresnel( nv ) ).mul( gloss );
	// (dir: a lamp's own metalwork takes its burner's highlight whatever its baked light, see glints.js)
	if ( dir ) col = col.add( glints( N, V, rough, f0, inLight ).mul( mix( U.glint, U.spec, spec ) ).add( env.mul( inLight ) ).mul( shiny.mul( GAO.fg ) ) );
	else col = col.add( glints( N, V, rough, f0 ).mul( U.glint ).add( env ).mul( shiny.mul( inLight ).mul( GAO.fg ) ) );

	// Neon and coloured signs: what glows with no texture of its own (layer 0) in a colour no flame has - a mantle,
	// a bulb, a lit room run red over green over blue, a tube of cyan, green, violet or pink does not. Amber and
	// yellow signs are not told from the lamps that way: the bake marks them (aux 65533, or 65532 where the packer
	// left a mark not to pull). A sign is brighter by HDR.neon, and so is its ceiling: a back-lit sign face is far
	// under the ceiling, a tube at it.
	const litM = max( max( lit.r, lit.g ), max( lit.b, 1e-4 ) );
	const hued = glowing.mul( step( layer, 0.5 ) ).mul( saturate( max( lit.b.sub( lit.g ), lit.g.sub( lit.r ) ).div( litM ).mul( 12 ) ) );
	const neon = max( hued, step( 65531.5 / 65535, aux ).mul( step( aux, 65533.5 / 65535 ) ) );
	const neonK = mix( float( 1 ), stage.HDR.neon, neon );
	material.colorNode = stage.knee( fogged( stage.emit( col, glowing ).mul( neonK ) ), glowing, neonK );

	// Only the glowing vertices the packer left at 65535 are pulled (pull.js) - those it marked 65534 (a lit card
	// with kept geometry right in front, a clock's hands) stay. A sign's vertices are the same pair one lower:
	// 65533 pulled, 65532 not.
	const a16 = cv.w.mul( 65535 );
	material.positionNode = pulled( step( 65534.5, a16 ).add( step( 65532.5, a16 ).mul( step( a16, 65533.5 ) ) ) );
	return material;

}
