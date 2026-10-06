// Canal and basin water at night: black water that mirrors.
//
// What makes it water and not a dark floor or a bank of mist: nothing of it is lit like a surface (only the
// silt in it, faintly), it mirrors by its fresnel alone - next to nothing straight down, nearly everything at a
// glancing angle - and what it mirrors is broken up by ripples with crests: four sizes of them, each the tiling
// noise stretched along its crests and sliding its own way, so a lantern lies on it as a column of short bright
// dashes towards the viewer. It says so in the mirror target like the wet paving (wet.js), and the mirror pass
// finds the quay wall, the hulls and the lit windows in it; the lanterns of the cell glint on it directly.

import * as THREE from 'three';
import { texture, vec2, vec3, float, mix, pow, dot, length, normalize, saturate, smoothstep, oneMinus, time, cameraPosition, positionWorld } from 'three/tsl';

export function createWater( ctx ) {

	const { stage, U, glints } = ctx;
	const { lit, V, fogged } = ctx.inputs;
	const material = new THREE.MeshBasicNodeMaterial( { fog: false } );
	material.name = 'baked water';

	const wp = positionWorld.xz;
	// (ang: which way the crests face; fx / fz: tiles per metre across / along them; v: metres a second;
	// ba: which pair of the noise's channels; amp: the slope)
	const wave = ( ang, fx, fz, v, ba, amp ) => {

		const c = Math.cos( ang ), sn = Math.sin( ang );
		const q = vec2( wp.x.mul( c ).add( wp.y.mul( sn ) ), wp.y.mul( c ).sub( wp.x.mul( sn ) ) );
		const n = texture( stage.noise, vec2( q.x.mul( fx ).add( time.mul( v * fx ) ), q.y.mul( fz ).add( time.mul( v * fz * 0.3 ) ) ) );
		const g = ( ba ? n.ba : n.rg ).sub( 0.5 ).mul( vec2( amp, amp * fz / fx ) ); // the slope is across the crests
		return vec2( g.x.mul( c ).sub( g.y.mul( sn ) ), g.x.mul( sn ).add( g.y.mul( c ) ) );

	};
	const w1 = wave( 1.2, 0.22, 0.12, 0.03, 0, 0.14 ), w2 = wave( 1.8, 0.9, 0.36, 0.07, 1, 0.3 );
	const w3 = wave( 1.3, 2.6, 1.1, - 0.09, 0, 0.3 ), w4 = wave( 2.0, 7.0, 3.4, 0.1, 1, 0.22 );
	const slow = w1.add( w2 ).mul( U.swell ), all = slow.add( w3.add( w4 ).mul( U.swell ) );
	const Ns = normalize( vec3( slow.x, 1, slow.y ) ), N = normalize( vec3( all.x, 1, all.y ) );
	const nv = saturate( dot( Ns, V ) );
	const fres = pow( oneMinus( nv ), 5 ).mul( 0.98 ).add( 0.02 );

	// (far off the ripples are smaller than a pixel: what they did to the image becomes its roughness)
	const far = smoothstep( 4, 40, length( cameraPosition.sub( positionWorld ) ) );
	material.mrtNode = stage.mirrorOut( saturate( fres.mul( U.wmirror ) ).mul( oneMinus( stage.fogAmount() ) ), mix( U.wrough, U.wrough.mul( 3 ), far ), N );

	// the silt, the lanterns' sharp glints on every ripple and their wide glow on the swell
	const c = lit.mul( U.wbody ).add( glints( N, V, float( 0.08 ), float( 0.02 ) ).mul( U.glint ) ).add( glints( Ns, V, float( 0.2 ), float( 0.02 ) ).mul( U.wglow.mul( 0.5 ) ) );
	material.colorNode = fogged( c );
	return material;

}
