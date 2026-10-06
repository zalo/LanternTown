// Lantern glows and flames: one instanced sprite draw each.
//
// The bake lists them (rows of numbers in the file's manifest); a sprite per row would be a draw call per lamp,
// so each list is ONE Sprite with a count and its rows as instanced attributes. Both are additive and write no
// depth: they are light, not things.
//
//   halo   [ x, y, z, r, g, b, size, core, towards ]   flame   [ x, y, z, r, g, b, size, glow, speed ]

import * as THREE from 'three';
import { instancedBufferAttribute, varying, texture, uv, vec2, vec3, mix, step, max, min, exp, pow, abs, sin, length, normalize, smoothstep, oneMinus, time, cameraPosition } from 'three/tsl';
import { haloCore } from '../engine/engine.js';

const HALO = 0.1; // the strength of a glow

// n numbers of every row, from column `from`, as one instanced attribute
function column( rows, from, n ) {

	const a = new Float32Array( rows.length * n );
	rows.forEach( ( r, i ) => {

		for ( let k = 0; k < n; k ++ ) a[ i * n + k ] = r[ from + k ];

	} );
	return instancedBufferAttribute( new THREE.InstancedBufferAttribute( a, n ) );

}

// The soft glow around a burner, with a hot core (the same as the engine's createHalo; `haloCore` of engine.js
// is its strength, set with the look). `core` is 0 for a helper light with no burner to look at. `towards`: the
// glow is drawn that far towards the eye - a burner inside an opaque glowing body, an opal globe, would
// otherwise have its hot centre hidden and look switched off.
function createHalos( stage, H ) {

	const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	const P = column( H, 0, 3 );
	m.positionNode = P.add( normalize( cameraPosition.sub( P ) ).mul( column( H, 8, 1 ) ) );
	m.scaleNode = column( H, 6, 1 );
	const r = length( uv().sub( 0.5 ) ).mul( 2 );
	const a = exp( r.mul( r ).mul( - 5 ) ).sub( 0.0067 ).max( 0 );
	const hs = varying( column( H, 6, 1 ) );
	const core = exp( r.mul( r ).mul( hs.mul( hs ).mul( - 11 ) ) ).mul( varying( column( H, 7, 1 ) ) );
	m.colorNode = varying( column( H, 3, 3 ) ).mul( a.add( core.mul( haloCore ).mul( stage.HDR.core ) ) ).mul( HALO );
	m.opacityNode = a.add( core.mul( haloCore.min( 1 ) ) ).min( 1 );
	const s = new THREE.Sprite( m );
	s.count = H.length;
	s.frustumCulled = false;
	s.renderOrder = 4;
	s.name = 'halos';
	return s;

}

// Open flames (torches, braziers): a teardrop that sways and flickers with the tiling noise, blue at the root,
// each with its own phase. The sprite stands on its foot.
function createFlames( stage, rows ) {

	const F = rows.map( ( f, i ) => [ ...f, ( i * 0.61803 ) % 1, f[ 6 ] * 0.8, f[ 6 ] * 1.7 ] ); // + phase, sprite size
	const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	m.positionNode = column( F, 0, 3 );
	m.scaleNode = column( F, 10, 2 );
	const col = varying( column( F, 3, 3 ) ), gs = varying( column( F, 7, 3 ) ); // glow, speed, phase
	const glow = gs.x, phase = gs.z;
	const p = uv();
	const y = p.y, x = p.x.sub( 0.5 ).mul( 2 );
	const t = time.mul( gs.y ).add( phase.mul( 7.3 ) );
	const n1 = texture( stage.noise, vec2( x.mul( 0.12 ).add( phase ), y.mul( 0.3 ).sub( t.mul( 0.5 ) ) ) ).r.sub( 0.5 );
	const n2 = texture( stage.noise, vec2( x.mul( 0.35 ).add( phase.mul( 2.1 ) ), y.mul( 0.8 ).sub( t.mul( 1.05 ) ) ) ).g.sub( 0.5 );
	const sway = sin( t.mul( 2.1 ) ).mul( 0.06 ).add( n1.mul( 0.55 ) ).add( n2.mul( 0.22 ) );
	const xx = x.add( sway.mul( y ) );
	const hgt = y.div( n1.mul( 0.25 ).add( 0.9 ) );
	const w = min( pow( max( hgt, 0 ).div( 0.22 ), 0.55 ), pow( max( oneMinus( hgt ), 0 ).div( 0.78 ), 0.85 ) ).mul( 0.42 );
	const dd = abs( xx ).div( max( w, 0.001 ) );
	const body = smoothstep( 1.0, 0.25, dd ).mul( step( hgt, 1 ) );
	const core = smoothstep( 0.7, 0.0, dd ).mul( smoothstep( 0.95, 0.25, hgt ) );
	const outer = col.mul( vec3( 1, 0.55, 0.25 ) );
	let c = mix( outer, vec3( 1.0, 0.86, 0.6 ), core );
	c = mix( vec3( 0.06, 0.125, 0.45 ), c, smoothstep( 0.02, 0.2, hgt ) );
	m.colorNode = c.mul( body ).mul( glow ).mul( stage.HDR.flame );
	m.opacityNode = body;
	const s = new THREE.Sprite( m );
	s.count = F.length;
	s.center.set( 0.5, 0.02 );
	s.frustumCulled = false;
	s.renderOrder = 5;
	s.name = 'flames';
	return s;

}

// Adds both to the scene. They animate by themselves (TSL `time`).
export function createSprites( { stage, bake } ) {

	const halos = createHalos( stage, bake.halos ), flames = createFlames( stage, bake.flames );
	stage.scene.add( halos, flames );
	return { halos, flames };

}
