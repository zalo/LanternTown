// Interactive single-asset viewer (linked from the gallery):  viewer.html?id=<asset>[&variant=<name>][&rig=night|studio]
// Same engine, materials and lights as the town; drag to orbit, wheel / pinch to zoom.

import { createEngine, THREE, ensureMask } from './engine/engine.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BASE = '.';
const Q = new URLSearchParams( location.search );
const $ = ( id ) => document.getElementById( id );
const touch = matchMedia( '(pointer: coarse)' ).matches;

function plane( w, h, uv, mask ) {

	const g = new THREE.PlaneGeometry( w, h, 1, 1 );
	const p = g.getAttribute( 'position' ), t = g.getAttribute( 'uv' );
	for ( let i = 0; i < p.count; i ++ ) t.setXY( i, ...uv( p.getX( i ), p.getY( i ) ) );
	const a = new Float32Array( p.count * 4 );
	for ( let i = 0; i < p.count; i ++ ) a.set( mask, i * 4 );
	g.setAttribute( '_mask', new THREE.BufferAttribute( a, 4 ) );
	return g;

}

async function main() {

	const canvas = $( 'gl' );
	const eng = window.eng = await createEngine( { canvas, base: BASE, width: innerWidth, height: innerHeight,
		pixelRatio: Math.min( devicePixelRatio || 1, 1 ), fov: 32 } ); // (pixel ratio 1 everywhere: the user's call)
	const M = await fetch( `${BASE}/api/manifest`, { cache: 'no-store' } ).then( ( r ) => r.json() );
	const matVersions = Object.fromEntries( ( M.materials || [] ).map( ( m ) => [ m.name, ( /v=(\d+)/.exec( m.preview || '' ) || [] )[ 1 ] || 0 ] ) );
	const A = M.assets.find( ( a ) => a.id === Q.get( 'id' ) ) || M.assets.find( ( a ) => a.variants?.length );
	if ( ! A || ! A.variants?.length ) {

		$( 'name' ).textContent = `${Q.get( 'id' ) || 'asset'}: not built yet`;
		return;

	}

	$( 'name' ).textContent = A.name || A.id;
	document.title = `${A.name || A.id} · Lantern Square`;
	$( 'variant' ).innerHTML = A.variants.map( ( v ) => `<option>${v.name}</option>` ).join( '' );
	$( 'rig' ).value = Q.get( 'rig' ) === 'studio' ? 'studio' : 'night';

	// stage: cobbled ground, a plaster wall for wall-mounted assets, an off-frame street lamp as the night key
	await eng.lib.prepare( [ 'cobble_granite', 'plaster_lime' ], matVersions );
	const ground = new THREE.Mesh( plane( 120, 120, ( x, y ) => [ x, 1 - y ], [ 1, 0.3, 0.25, 0.3 ] ).rotateX( - Math.PI / 2 ) );
	const wall = new THREE.Mesh( plane( 40, 14, ( x, y ) => [ x, 1 - y ], [ 1, 0.25, 0.1, 0.1 ] ) );
	eng.lib.apply( ground, 'cobble_granite' );
	eng.lib.apply( wall, 'plaster_lime', [ 0.55, 0.47, 0.36 ] );
	wall.position.y = 7;
	ground.receiveShadow = wall.receiveShadow = wall.castShadow = true;
	const key = new THREE.PointLight( new THREE.Color( '#ffbf6a' ), 60, 40, 2 );
	key.castShadow = true;
	key.shadow.mapSize.set( 2048, 2048 );
	key.shadow.bias = - 0.0015;
	key.shadow.normalBias = 0.02;
	const fill = new THREE.PointLight( new THREE.Color( '#ff9a52' ), 20, 40, 2 );
	eng.scene.add( ground, wall, key, fill );

	const controls = new OrbitControls( eng.camera, canvas );
	controls.enableDamping = true;
	controls.maxPolarAngle = Math.PI * 0.497;

	let inst = null, loading = 0;
	const cache = new Map();

	function rig() {

		if ( ! inst ) return;
		const night = $( 'rig' ).value === 'night';
		const box = new THREE.Box3().setFromObject( inst.root, true );
		const c = box.getCenter( new THREE.Vector3() ), r = Math.max( box.getSize( new THREE.Vector3() ).length() * 0.5, 0.35 );
		const own = inst.sockets.some( ( s ) => s.kind === 'light' );
		eng.setRig( night ? 'night' : 'studio', night ? { fog: 0.02 } : { fog: 0.02, fogColor: '#8d9299' } );
		eng.setBackground( night ? null : '#8d9299' );
		eng.fitShadow( box );
		key.visible = fill.visible = night;
		key.position.set( c.x - ( r * 1.1 + 0.6 ), Math.max( c.y + r * 0.9 + 0.5, Math.min( 3.2, box.max.y + 1.2 ) ), c.z + r * 1.5 + 0.9 );
		key.intensity = ( own ? 0.3 : 1.8 ) * key.position.distanceToSquared( c );
		key.distance = key.position.distanceTo( c ) * 6 + 10;
		key.shadow.camera.far = key.distance;
		fill.position.set( c.x + r * 1.6 + 1.2, c.y + r * 0.2 + 0.4, c.z + r * 1.2 + 1.5 );
		fill.intensity = ( own ? 0.08 : 0.25 ) * fill.position.distanceToSquared( c );
		fill.distance = key.distance;
		for ( const o of inst.active || [] ) o.visible = night;
		return { box, c, r };

	}

	async function show( name ) {

		const my = ++ loading;
		const v = A.variants.find( ( x ) => x.name === name ) || A.variants[ 0 ];
		const version = ( /v=(\d+)/.exec( v.glb || '' ) || [] )[ 1 ] || 0;
		$( 'info' ).textContent = 'loading…';
		let next = cache.get( v.name );
		if ( ! next ) cache.set( v.name, next = await eng.assets.load( A.id, v.name, { version, matVersions } ) );
		if ( my !== loading ) return;
		if ( inst ) {

			eng.deactivate( inst );
			eng.scene.remove( inst.root );

		}

		inst = next;
		const meta = inst.meta?.meta || {}, mount = meta.mount || 'ground';
		const mn = inst.meta?.bounds?.min?.[ 2 ] ?? 0;
		const lift = mount === 'wall' ? meta.mount_height ?? 2.6 : ( mount === 'hang' || mn < - 0.08 ) ? meta.mount_height ?? Math.max( 2.4, - mn + 1.2 ) : 0;
		wall.visible = mount === 'wall' || mount === 'wall_ground';
		inst.root.position.set( 0, lift, 0 );
		eng.scene.add( inst.root );
		inst.root.updateMatrixWorld( true );
		eng.activate( inst, { lights: true, fx: true, halos: true, shadowLights: 2 } );
		const { c, r } = rig();
		const d = r / Math.tan( THREE.MathUtils.degToRad( eng.camera.fov / 2 ) ) * 1.15;
		const az = THREE.MathUtils.degToRad( 36 ), el = THREE.MathUtils.degToRad( r > 4.5 ? 6 : 13 );
		eng.camera.position.set( c.x + Math.sin( az ) * Math.cos( el ) * d, Math.max( 0.25, c.y + Math.sin( el ) * d ), c.z + Math.cos( az ) * Math.cos( el ) * d );
		eng.camera.near = Math.max( 0.02, d * 0.01 );
		eng.camera.updateProjectionMatrix();
		controls.target.copy( c );
		controls.minDistance = r * 0.15;
		controls.maxDistance = r * 8 + 4;
		const b = inst.meta?.bounds?.size;
		$( 'info' ).textContent = `${A.id} / ${v.name}   ${Math.round( inst.tris ).toLocaleString()} tris` + ( b ? `   ${b.map( ( x ) => x.toFixed( 2 ) ).join( ' × ' )} m (w d h)` : '' ) +
			'\n' + Object.entries( v.params || {} ).map( ( [ k, x ] ) => `${k}=${typeof x === 'number' ? + x.toFixed( 3 ) : x}` ).join( '  ' );

	}

	$( 'variant' ).onchange = () => show( $( 'variant' ).value );
	$( 'rig' ).onchange = rig;
	addEventListener( 'resize', () => eng.setSize( innerWidth, innerHeight ) );
	const want = Q.get( 'variant' );
	if ( want && A.variants.some( ( v ) => v.name === want ) ) $( 'variant' ).value = want;
	await show( $( 'variant' ).value );
	const t0 = performance.now();
	eng.renderer.setAnimationLoop( () => {

		controls.update();
		eng.update( ( performance.now() - t0 ) / 1000 );
		eng.render();

	} );

}

main().catch( ( e ) => {

	console.error( e );
	$( 'info' ).textContent = 'viewer failed: ' + ( e?.message || e );

} );

void ensureMask;
