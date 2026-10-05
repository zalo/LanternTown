// GLB asset loading: resolves library materials, collects sockets (lights / fx / attach points) and
// animated parts from the node extras written by tk/final.py.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MaterialLib } from './materials.js';

export function ensureMask( geometry ) {

	if ( geometry.getAttribute( '_mask' ) ) return;
	const n = geometry.getAttribute( 'position' ).count;
	const a = new Float32Array( n * 4 );
	for ( let i = 0; i < n; i ++ ) a[ i * 4 ] = 1;
	geometry.setAttribute( '_mask', new THREE.BufferAttribute( a, 4 ) );

}

const parseJSON = ( s, d ) => {

	if ( s === undefined || s === null ) return d;
	if ( typeof s !== 'string' ) return s;
	try {

		return JSON.parse( s );

	} catch ( e ) {

		return d;

	}

};

export class AssetLoader {

	constructor( lib, base = '.' ) {

		this.lib = lib;
		this.base = base;
		this.gltf = new GLTFLoader();

	}

	// -> { root, sockets, parts, stats, meta }
	async load( id, variant = 'default', { version = 0, matVersions = {}, sidecar = true } = {} ) {

		const url = `${this.base}/out/glb/${id}/${variant}.glb?v=${version}`;
		const [ gltf, meta ] = await Promise.all( [
			this.gltf.loadAsync( url ),
			sidecar ? fetch( `${this.base}/out/glb/${id}/${variant}.json?v=${version}`, { cache: 'no-cache' } ).then( ( r ) => ( r.ok ? r.json() : null ) ).catch( () => null ) : null,
		] );
		const root = gltf.scene;
		root.name = `${id}/${variant}`;
		await this.lib.prepare( MaterialLib.libsOf( root ), matVersions );
		return this.adopt( root, meta );

	}

	adopt( root, meta = null ) {

		const sockets = [], parts = [];
		let tris = 0;
		root.updateMatrixWorld( true );
		const inv = new THREE.Matrix4().copy( root.matrixWorld ).invert();
		const old = new Set();
		root.traverse( ( o ) => {

			const ud = o.userData || {};
			if ( ud.tsq_socket ) {

				const m = new THREE.Matrix4().multiplyMatrices( inv, o.matrixWorld );
				const s = { kind: ud.tsq_socket, name: ud.tsq_name, props: parseJSON( ud.tsq_props, {} ), node: o,
					position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), scale: new THREE.Vector3() };
				m.decompose( s.position, s.quaternion, s.scale );
				sockets.push( s );

			}

			if ( ud.tsq_part && ud.tsq_part !== 'body' ) {

				parts.push( { name: ud.tsq_part, node: o, props: parseJSON( ud.tsq_part_props, {} ), rest: o.quaternion.clone() } );

			}

			if ( ! o.isMesh ) return;
			ensureMask( o.geometry );
			const src = o.material;
			old.add( src );
			const ex = src.userData || {};
			const mp = parseJSON( ex.tsq_params, {} );
			const mat = this.lib.apply( o, ex.tsq_lib || 'plain', ex.tsq_tint || null, mp );
			// glowing surfaces (bulbs, mantles, lit panes) must not shadow the light that sits inside them
			o.castShadow = ! mat.transparent && ! /^(emissive|neon|glass_|window_|water)/.test( ex.tsq_lib || '' ) && ! ( mp.glow > 0 );
			o.receiveShadow = true;
			if ( mat.transparent ) o.renderOrder = 2;
			const g = o.geometry;
			tris += ( g.index ? g.index.count : g.getAttribute( 'position' ).count ) / 3;

		} );
		for ( const m of old ) m.dispose();
		return { root, sockets, parts, tris, meta };

	}

}

// simple procedural part motion (lantern swing, sign creak, weathervane spin, flag flutter)
export function animateParts( parts, t, wind = 1 ) {

	for ( let i = 0; i < parts.length; i ++ ) {

		const p = parts[ i ], pr = p.props || {};
		if ( ! pr.motion ) continue; // motion: None = a separate but static part (gate leaves, door leaves, pump handles)
		const ax = ( p.axis ??= partAxis( p, pr ) );
		// flutter amplitudes below half a unit are metres of travel at the free edge, everything else is degrees
		const amp = ( pr.motion === 'flutter' && pr.amp < 0.5 ? pr.amp / p.drop : THREE.MathUtils.degToRad( pr.amp ?? 3 ) ) * wind;
		const speed = pr.speed ?? 1;
		const ph = ( p.phase ??= ( pr.phase !== undefined ? pr.phase * 6.2832 : i * 1.618 + ( p.node.id % 17 ) ) );
		let a = 0;
		if ( pr.motion === 'spin' ) a = t * speed * 0.6 + Math.sin( t * 0.31 + ph ) * 1.5;
		else if ( pr.motion === 'flutter' ) a = amp * ( Math.sin( t * 2.3 * speed + ph ) * 0.6 + Math.sin( t * 5.1 * speed + ph * 2 ) * 0.4 );
		else a = amp * ( Math.sin( t * 0.9 * speed + ph ) * 0.7 + Math.sin( t * 2.17 * speed + ph * 1.7 ) * 0.3 );
		_q.setFromAxisAngle( ax, a );
		p.node.quaternion.copy( p.rest ).multiply( _q );

	}

}

const _X = new THREE.Vector3( 1, 0, 0 ), _Y = new THREE.Vector3( 0, 1, 0 ), _Z = new THREE.Vector3( 0, 0, 1 );
const _q = new THREE.Quaternion(), _b = new THREE.Box3();

// hinge axis of a part in its own (glTF, Y-up) frame; the part's origin is its pivot.
// Authored axes are Blender's: X -> x, Y -> z, Z (up) -> y.
function partAxis( p, pr ) {

	_b.makeEmpty();
	p.node.traverse( ( o ) => {

		if ( ! o.geometry ) return;
		if ( ! o.geometry.boundingBox ) o.geometry.computeBoundingBox();
		_b.union( o.geometry.boundingBox );

	} );
	p.drop = _b.isEmpty() ? 1 : Math.max( 0.1, - _b.min.y, _b.max.y ); // how far the part reaches from its pivot
	const a = typeof pr.axis === 'string' ? pr.axis.toLowerCase() : '';
	if ( a === 'x' ) return _X;
	if ( a === 'y' ) return _Z;
	if ( a === 'z' || pr.motion === 'spin' ) return _Y;
	// no axis authored: a hanging thing turns about its long horizontal side (banner rod, sign bar, flag hoist),
	// so cloth and boards swing out of their own plane instead of sliding sideways within it
	return _b.isEmpty() || ( _b.max.x - _b.min.x ) >= ( _b.max.z - _b.min.z ) ? _X : _Z;

}
