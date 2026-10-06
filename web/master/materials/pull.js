// Glowing vertices are pulled towards the eye.
//
// A lit room is a glowing card a few millimetres behind its window's opening, a lantern's mantle a card inside
// its cage. The levels below the fine one are simplified to centimetres (the lowest to 0.2 m), so the card and
// the wall it sits in can end up in one plane or pass through each other: which of the two a pixel shows then
// turns with every step, and the mirror image in the paving - the town at its lowest level, seen from a few
// metres - draws that out into a flickering streak. So glowing vertices (and the glass in front of them, which
// must stay in front) move along their line of sight: the place on the screen stays, only the depth changes.
//
// PULL[ level ] = [ metres, share of the distance ], level 0 the lowest. Only that one is pulled: what stands
// closer than the pull in front of a lit card goes behind it, and the levels above still have their glazing
// bars (there the packer keeps what lies at a glowing face as the model has it). The plugin gives each mesh its
// row (mesh.userData.pull); the mirror pass sets `pullAll` to PULL.mirror while it draws.

import * as THREE from 'three';
import { uniform, vec4, length, max, min, cameraPosition, positionLocal, modelWorldMatrixInverse } from 'three/tsl';

export function createPull() {

	const PULL = { 0: [ 0.06, 0.0005 ], 1: [ 0, 0 ], 2: [ 0, 0 ], 3: [ 0, 0 ], mirror: [ 0.12, 0 ] };
	const pullAll = uniform( new THREE.Vector2() );
	const pullObj = uniform( new THREE.Vector3() ).onObjectUpdate( ( { object }, self ) => {

		const p = object.userData.pull;
		self.value.set( p ? p[ 0 ] : 0, p ? p[ 1 ] : 0, object.matrixWorld.elements[ 0 ] ); // (.z: metres per unit of the mesh)

	} );

	// the vertex, moved; which: 0 .. 1 per vertex, or null for all of the mesh (glass)
	const pulled = ( which ) => {

		const to = modelWorldMatrixInverse.mul( vec4( cameraPosition, 1 ) ).xyz.sub( positionLocal );
		const d = length( to ).mul( pullObj.z );
		const t = min( max( pullObj.x, pullAll.x ).div( max( d, 1e-3 ) ).add( max( pullObj.y, pullAll.y ) ), 0.5 );
		return positionLocal.add( to.mul( which === null ? t : t.mul( which ) ) );

	};

	return { PULL, pullAll, pulled };

}
