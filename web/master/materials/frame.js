// The frame of a face, from screen derivatives.
//
// The bake's vertices carry no normals and no tangents (12 to 18 bytes each, and a level of detail that is
// simplified to centimetres would not keep them anyway). The face normal is the cross product of the position's
// derivatives, and a height given per pixel tilts it by its own derivatives (bump): relief, the thickness of
// falling water, the height of a wet sett's crown all go through here.

import { dFdx, dFdy, cross, dot, length, max, sign, positionWorld } from 'three/tsl';

export const unit = ( v ) => v.div( max( length( v ), 1e-20 ) );

export function faceFrame( V ) {

	const dpx = dFdx( positionWorld ), dpy = dFdy( positionWorld );
	const N0 = unit( cross( dpx, dpy ) );
	const R1 = cross( dpy, N0 ), R2 = cross( N0, dpx ), det = dot( dpx, R1 );
	// the normal of the face (turned to the eye) under a height h in metres, not normalised
	const bump = ( h ) => N0.mul( det ).mul( sign( dot( N0, V ) ) ).sub( R1.mul( dFdx( h ) ).add( R2.mul( dFdy( h ) ) ) );
	return { dpx, dpy, N0, R1, R2, det, bump };

}
