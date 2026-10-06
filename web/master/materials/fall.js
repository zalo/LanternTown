// Falling water: a torn sheet, lit by the baked colour.
//
// A fluted sheet at the lip that thickens into ropes, tears open between them and ends in white threads and
// drops: strands of the tiling noise scrolling down V. Its two masks - break-up and foam - ride in the layer
// slot of a fall's vertices, 4 bits each: without them every jet is an unbroken thin sheet, next to invisible.
//
// Clear water has no diffuse response: only the silvering and the foam are lit like a surface. The sheet is a
// veil blended over the frame, so both are taken over the veil's share - blended at the veil's cover they came
// out a third as bright on the clear film, which is most of a sheet's width. The baked colour alone is all
// lamp; a sheet of water is mostly what is seen through it plus the sky's share of its mirror, a cool silver
// (U.fallCool).
//
// The lanterns run as thin bright lines down the flutes and ropes: the relief of the water's thickness turns
// the face normal, the light grid the wet paving reads does the rest.

import * as THREE from 'three';
import { varying, texture, vec2, vec3, float, mix, max, dot, floor, smoothstep, oneMinus, time } from 'three/tsl';
import { faceFrame, unit } from './frame.js';

export function createFall( ctx ) {

	const { stage, U, glints } = ctx;
	const { pl, tiles, lit, aux, V, fogged } = ctx.inputs;
	const material = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false, side: THREE.DoubleSide } );
	material.name = 'baked fall';

	const lw = pl.w.mul( 65535 ).add( 0.5 ), wq = floor( lw.div( 16 ) );
	const fm = varying( vec2( floor( lw.sub( wq.mul( 16 ) ) ), wq ).div( 15 ), 'vFall' );
	const brk = fm.x.saturate(), white = fm.y.saturate();
	const t = time.mul( 1.6 );
	const u0 = tiles.x, v = tiles.y.negate();
	// strands wander sideways as they fall, more once they have torn free
	const u = u0.add( texture( stage.noise, vec2( u0.mul( 0.9 ), v.mul( 0.31 ).add( t.mul( 0.21 ) ) ) ).a.sub( 0.5 ).mul( brk.mul( 0.05 ).add( 0.012 ) ) );
	const s1 = texture( stage.noise, vec2( u.mul( 3.1 ), v.mul( 0.23 ).add( t.mul( 0.42 ) ) ) ); // flutes of the sheet
	const s2 = texture( stage.noise, vec2( u.mul( 8.3 ).add( 0.37 ), v.mul( 0.5 ).add( t.mul( 0.9 ) ) ) ); // threads
	const s3 = texture( stage.noise, vec2( u.mul( 1.3 ).add( 0.71 ), v.mul( 0.62 ).add( t.mul( 1.1 ) ) ) ); // surges travelling down
	const body = s1.r.mul( brk.mul( - 0.2 ).add( 0.47 ) ).add( s2.g.mul( 0.25 ) ).add( s3.b.mul( brk.mul( 0.2 ).add( 0.28 ) ) );
	const thr = mix( float( 0.16 ), float( 0.61 ), brk.mul( brk.mul( 0.35 ).add( 0.65 ) ) ).sub( white.mul( 0.14 ) );
	const a = smoothstep( thr, thr.add( 0.06 ), body );
	const rim = a.mul( oneMinus( smoothstep( thr.add( 0.05 ), thr.add( 0.16 ), body ) ) ); // the rolled edge of a rope
	const fib = smoothstep( 0.42, 0.7, s2.g.mul( 0.55 ).add( s1.b.mul( 0.45 ) ) );
	const air = brk.mul( brk.mul( 0.5 ).add( 0.5 ) );
	const foam = white.add( fib.mul( 0.26 ).add( 0.06 ) ).add( air.mul( fib.mul( 0.4 ).add( 0.36 ) ) ).add( rim.mul( brk ).mul( 0.5 ) ).saturate();
	const bead = smoothstep( 0.8, 0.92, s2.b.mul( 0.6 ).add( s1.a.mul( 0.4 ) ) ).mul( a ).mul( brk.mul( 0.8 ).add( 0.2 ) );

	const veil = mix( float( 0.22 ), float( 0.95 ), max( foam, bead ) );
	const flit = mix( lit, vec3( 0.81, 0.88, 0.9 ).mul( dot( lit, vec3( 0.2126, 0.7152, 0.0722 ) ) ), U.fallCool );
	const op = a.mul( veil ).mul( aux );
	let c = flit.mul( foam.mul( 0.8 ).add( 0.03 ) ).mul( U.fallLit ).add( vec3( 0.62, 0.74, 0.79 ).mul( bead ).mul( 0.3 ) ).div( veil );

	// the lanterns' lines (the water's thickness in metres, from the same noise)
	const hgt = body.mul( 1.6 ).add( smoothstep( thr.sub( 0.02 ), thr.add( 0.2 ), body ) ).mul( 0.012 );
	const sp = glints( unit( faceFrame( V ).bump( hgt ) ), V, mix( float( 0.08 ), float( 0.6 ), foam ), float( 0.02 ) ).mul( U.fall );
	c = c.add( sp.mul( a ).div( max( op, 0.05 ) ) ); // (blended: over its cover, so the line arrives whole)

	material.colorNode = fogged( c );
	material.opacityNode = op;
	return material;

}
