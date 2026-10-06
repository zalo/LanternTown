// A change of level is dithered across.
//
// While it lasts (tier.fade seconds) the two meshes of a cell draw with these variants of their materials: the
// one coming in where the screen noise is under its progress, the one going out where it is over. Opaque stays
// opaque: no blending, no sorting, both meshes write depth. The progress is the mesh's own: levels.js writes
// mesh.userData.fade (positive: coming in, drawn where the noise is under it; negative: going out, drawn where
// the noise is over its size; none: drawn whole) and an object uniform hands it to the shader.

import * as THREE from 'three';
import { uniform, abs, interleavedGradientNoise, screenCoordinate } from 'three/tsl';

export function createFading() {

	const fadeU = uniform( 1 ).onObjectUpdate( ( { object } ) => object.userData.fade ?? 1 );
	const fadeMask = interleavedGradientNoise( screenCoordinate.xy ).sub( abs( fadeU ) ).mul( fadeU ).lessThan( 0 );

	// the same material, drawn through the mask
	return ( m ) => {

		const f = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: m.transparent, depthWrite: m.depthWrite, side: m.side } );
		f.colorNode = m.colorNode; f.opacityNode = m.opacityNode; f.mrtNode = m.mrtNode; f.positionNode = m.positionNode;
		f.maskNode = m.maskNode ? m.maskNode.and( fadeMask ) : fadeMask;
		f.forceSinglePass = m.forceSinglePass;
		f.name = m.name + ' (fading)';
		return f;

	};

}
