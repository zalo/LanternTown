// Review render service page. Driven by tools/renderd.py through CDP:
//   window.startJob(job) -> starts an async job;  window.jobStatus(job_id) -> JSON string
// Renders are composed onto 2D canvases and POSTed to /api/upload (out/renders/...).

import { createEngine, THREE, ensureMask, MaterialLib } from './engine/engine.js';

const BASE = '.';
const SS = 2; // supersampling factor for review renders
const PAGE_VERSION = 3; // bumped when a job kind is added or a sheet changes (3: lod sheets name what ships): tools/tsq resubmits until the daemon's tab runs this file
let eng = null;
let stage = null;
const jobs = {};
window.__jobs = jobs;

const tick = () => new Promise( ( r ) => {

	const c = new MessageChannel();
	c.port1.onmessage = () => r();
	c.port2.postMessage( 0 );

} );

async function getEngine() {

	if ( eng ) return eng;
	const canvas = document.getElementById( 'gl' );
	eng = await createEngine( { canvas, base: BASE, width: 800, height: 600, pixelRatio: SS } );
	eng.manual = true;
	window.eng = eng;
	stage = await buildStage( eng );
	return eng;

}

// ---------------------------------------------------------------------------------- stage

function metreUVs( geo, fn ) {

	const p = geo.getAttribute( 'position' ), uvA = geo.getAttribute( 'uv' );
	for ( let i = 0; i < p.count; i ++ ) {

		const [ u, v ] = fn( p.getX( i ), p.getY( i ), p.getZ( i ) );
		uvA.setXY( i, u, 1 - v );

	}

	uvA.needsUpdate = true;

}

function setMask( geo, ao, grime, wear, special ) {

	const n = geo.getAttribute( 'position' ).count;
	const a = new Float32Array( n * 4 );
	for ( let i = 0; i < n; i ++ ) {

		a[ i * 4 ] = ao; a[ i * 4 + 1 ] = grime; a[ i * 4 + 2 ] = wear; a[ i * 4 + 3 ] = special;

	}

	geo.setAttribute( '_mask', new THREE.BufferAttribute( a, 4 ) );

}

async function buildStage( eng ) {

	const g = new THREE.Group();
	g.name = 'stage';
	const ground = new THREE.Mesh( new THREE.PlaneGeometry( 120, 120, 1, 1 ).rotateX( - Math.PI / 2 ) );
	metreUVs( ground.geometry, ( x, y, z ) => [ x, - z ] );
	setMask( ground.geometry, 1, 0.3, 0.25, 0.3 );
	ground.receiveShadow = true;
	const wall = new THREE.Mesh( new THREE.PlaneGeometry( 40, 14, 1, 1 ) );
	wall.position.set( 0, 7, 0 );
	metreUVs( wall.geometry, ( x, y ) => [ x, y ] );
	setMask( wall.geometry, 1, 0.25, 0.1, 0.1 );
	wall.receiveShadow = true;
	wall.castShadow = true;
	g.add( ground, wall );
	eng.scene.add( g );
	// 1.8 m scale figure (clay views only)
	const fig = new THREE.Group();
	const body = new THREE.Mesh( new THREE.CapsuleGeometry( 0.17, 1.1, 6, 16 ) );
	body.position.y = 0.17 + 0.55 + 0.35;
	const legs = new THREE.Mesh( new THREE.CapsuleGeometry( 0.13, 0.75, 6, 12 ) );
	legs.position.y = 0.13 + 0.375;
	const head = new THREE.Mesh( new THREE.SphereGeometry( 0.115, 20, 14 ) );
	head.position.y = 1.685;
	fig.add( body, legs, head );
	fig.traverse( ( o ) => {

		if ( o.isMesh ) {

			ensureMask( o.geometry ); o.castShadow = true;

		}

	} );
	fig.visible = false;
	eng.scene.add( fig );
	// key lantern for the night rig (a street lamp just out of frame)
	const key = new THREE.PointLight( new THREE.Color( '#ffbf6a' ), 60, 40, 2 );
	key.castShadow = true;
	key.shadow.mapSize.set( 2048, 2048 );
	key.shadow.bias = - 0.0015;
	key.shadow.normalBias = 0.02;
	key.shadow.camera.near = 0.05;
	eng.scene.add( key );
	const fill = new THREE.PointLight( new THREE.Color( '#ff9a52' ), 20, 40, 2 );
	eng.scene.add( fill );
	return { group: g, ground, wall, fig, key, fill, mats: {} };

}

async function stageMaterials( eng, matVersions ) {

	await eng.lib.prepare( [ 'cobble_granite', 'plaster_lime' ], matVersions );
	eng.lib.apply( stage.ground, 'cobble_granite' );
	eng.lib.apply( stage.wall, 'plaster_lime', [ 0.55, 0.47, 0.36 ] );
	const clay = eng.lib.clay();
	stage.fig.traverse( ( o ) => {

		if ( o.isMesh ) o.material = clay;

	} );

}

// ---------------------------------------------------------------------------------- shots

function frame( eng, box, { az = 35, el = 14, zoom = 1, aspect = 1.6, target = null, fov = 32 } = {} ) {

	const cam = eng.camera;
	cam.fov = fov;
	cam.aspect = aspect;
	const c = target ? target.clone() : box.getCenter( new THREE.Vector3() );
	const size = box.getSize( new THREE.Vector3() );
	const a = THREE.MathUtils.degToRad( az ), e = THREE.MathUtils.degToRad( el );
	const dir = new THREE.Vector3( Math.sin( a ) * Math.cos( e ), Math.sin( e ), Math.cos( a ) * Math.cos( e ) );
	// fit the box corners into the frustum
	const right = new THREE.Vector3().crossVectors( new THREE.Vector3( 0, 1, 0 ), dir ).normalize();
	const up = new THREE.Vector3().crossVectors( dir, right ).normalize();
	const tv = Math.tan( THREE.MathUtils.degToRad( fov ) / 2 ), th = tv * aspect;
	let dist = 0.1;
	for ( let i = 0; i < 8; i ++ ) {

		const p = new THREE.Vector3( i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z ).sub( c );
		const x = Math.abs( p.dot( right ) ), y = Math.abs( p.dot( up ) ), z = p.dot( dir );
		dist = Math.max( dist, z + x / th, z + y / tv );

	}

	dist = dist * 1.12 / zoom;
	cam.position.copy( c ).addScaledVector( dir, dist );
	// keep the camera above the ground
	if ( cam.position.y < 0.25 ) cam.position.y = 0.25;
	cam.near = Math.max( 0.02, dist * 0.02 );
	cam.far = 1500;
	cam.lookAt( c );
	cam.updateProjectionMatrix();
	cam.updateMatrixWorld();
	return { dist, center: c, size };

}

function nightKey( eng, box, own, dir = 1 ) {

	const c = box.getCenter( new THREE.Vector3() );
	const r = Math.max( box.getSize( new THREE.Vector3() ).length() * 0.5, 0.35 );
	const k = stage.key, f = stage.fill;
	k.position.set( c.x - dir * ( r * 1.1 + 0.6 ), Math.max( c.y + r * 0.9 + 0.5, Math.min( 3.2, box.max.y + 1.2 ) ), c.z + r * 1.5 + 0.9 );
	const d2 = k.position.distanceToSquared( c );
	k.intensity = ( own ? 0.3 : 1.8 ) * d2;
	k.distance = Math.sqrt( d2 ) * 6 + 10;
	k.shadow.camera.far = k.distance;
	k.visible = true;
	f.position.set( c.x + dir * ( r * 1.6 + 1.2 ), c.y + r * 0.2 + 0.4, c.z + r * 1.2 + 1.5 );
	f.intensity = ( own ? 0.08 : 0.25 ) * f.position.distanceToSquared( c );
	f.distance = k.distance;
	f.visible = true;

}

async function shoot( eng, ctx, x, y, w, h, frames = 3 ) {

	eng.setSize( w * SS, h * SS, 1 );
	eng.camera.aspect = w / h;
	eng.camera.updateProjectionMatrix();
	for ( let i = 0; i < frames - 1; i ++ ) {

		eng.update( 10 + i * 0.016 );
		eng.render();
		await tick();

	}

	eng.update( 10.1 );
	const img = await eng.capture();
	if ( ! shotCanvas || shotCanvas.width !== img.width || shotCanvas.height !== img.height ) {

		shotCanvas = document.createElement( 'canvas' );
		shotCanvas.width = img.width; shotCanvas.height = img.height;

	}

	shotCanvas.getContext( '2d' ).putImageData( img, 0, 0 );
	ctx.imageSmoothingEnabled = true;
	ctx.imageSmoothingQuality = 'high';
	ctx.drawImage( shotCanvas, x, y, w, h );

}

let shotCanvas = null;

function label( ctx, x, y, text, { size = 15, color = '#f2e6d0', bg = 'rgba(0,0,0,0.55)', align = 'left' } = {} ) {

	ctx.font = `600 ${size}px system-ui, sans-serif`;
	const lines = String( text ).split( '\n' );
	const w = Math.max( ...lines.map( ( l ) => ctx.measureText( l ).width ) ) + 14, h = lines.length * ( size + 5 ) + 8;
	const x0 = align === 'right' ? x - w : x;
	ctx.fillStyle = bg;
	ctx.fillRect( x0, y, w, h );
	ctx.fillStyle = color;
	ctx.textBaseline = 'top';
	lines.forEach( ( l, i ) => ctx.fillText( l, x0 + 7, y + 5 + i * ( size + 5 ) ) );

}

async function upload( canvas, path, token, type = 'image/png', q ) {

	const blob = await new Promise( ( r ) => canvas.toBlob( r, type, q ) );
	const res = await fetch( `${BASE}/api/upload?path=${encodeURIComponent( path )}&token=${token}`, { method: 'POST', body: blob } );
	if ( ! res.ok ) throw new Error( `upload ${path} failed: ${res.status}` );
	return 'out/' + path;

}

const canvas2d = ( w, h ) => {

	const c = document.createElement( 'canvas' );
	c.width = w; c.height = h;
	const ctx = c.getContext( '2d' );
	ctx.fillStyle = '#0c0d10';
	ctx.fillRect( 0, 0, w, h );
	return [ c, ctx ];

};

const B2T = ( v ) => new THREE.Vector3( v[ 0 ], v[ 2 ], - v[ 1 ] ); // Blender Z-up -> three Y-up

// ---------------------------------------------------------------------------------- asset job

async function runAsset( job ) {

	const eng = await getEngine();
	const warnings = [];
	await stageMaterials( eng, job.matVersions || {} );
	const insts = {};
	for ( const v of job.variants ) {

		insts[ v ] = await eng.assets.load( job.id, v, { version: ( job.versions || {} )[ v ] ?? 0, matVersions: job.matVersions || {} } );

	}

	const names = job.variants.slice().sort( ( a, b ) => ( a === 'default' ? - 1 : b === 'default' ? 1 : 0 ) );
	const main = insts[ names[ 0 ] ];
	const meta = main.meta?.meta || {};
	const mount = meta.mount || 'ground';
	const wallMount = mount === 'wall' || mount === 'wall_ground';
	// wall-mounted assets hang on the stage wall; hanging assets (origin = suspension point, geometry below) are
	// lifted so they clear the ground
	const liftOf = ( inst ) => {

		if ( mount === 'wall' ) return meta.mount_height ?? 2.6;
		const mn = inst.meta?.bounds?.min?.[ 2 ] ?? 0;
		if ( mount === 'hang' || mn < - 0.08 ) return meta.mount_height ?? Math.max( 2.4, - mn + 1.2 );
		return 0;

	};

	const lift = liftOf( main );
	stage.wall.visible = wallMount;
	const files = [];
	// review exposure guard: an asset's own lights are scaled so the nearest lit stage surface is not blown out
	// (a lantern head reviewed at ground level would otherwise put >100 lux on the cobbles)
	const E_MAX = 10;
	const lightScale = ( inst, y0 ) => {

		let e = 0;
		for ( const sk of inst.sockets ) {

			if ( sk.kind !== 'light' ) continue;
			const I = sk.props.intensity ?? 40;
			const hgt = Math.max( sk.position.y + y0, 0.5 );
			e = Math.max( e, I / ( hgt * hgt ) );
			if ( wallMount ) e = Math.max( e, I / Math.max( sk.position.z, 0.6 ) ** 2 * 0.5 );

		}

		return e > E_MAX ? E_MAX / e : 1;

	};

	const show = ( inst, on, opts = {} ) => {

		if ( on ) {

			const y0 = liftOf( inst );
			inst.root.position.set( opts.x ?? 0, y0, 0 );
			eng.scene.add( inst.root );
			inst.root.updateMatrixWorld( true );
			eng.activate( inst, { lights: opts.lights ?? true, fx: true, halos: opts.lights ?? true, shadowLights: opts.shadowLights ?? 1,
				intensity: lightScale( inst, y0 ) } );

		} else {

			eng.deactivate( inst );
			eng.scene.remove( inst.root );

		}

	};

	const boxOf = ( inst ) => new THREE.Box3().setFromObject( inst.root, true );
	const info = ( inst ) => {

		const b = inst.meta?.bounds?.size;
		return `${inst.tris.toLocaleString()} tris` + ( b ? `  ${b[ 0 ].toFixed( 2 )} x ${b[ 1 ].toFixed( 2 )} x ${b[ 2 ].toFixed( 2 )} m (w d h)` : '' );

	};

	const hasOwn = ( inst ) => inst.sockets.some( ( s ) => s.kind === 'light' );
	const big = () => boxOf( main ).getSize( new THREE.Vector3() ).length() > 9;

	// ---- hero: night, 3/4 view
	{

		const [ c, ctx ] = canvas2d( 1600, 1000 );
		show( main, true, { shadowLights: 2 } );
		const box = boxOf( main );
		eng.setRig( 'night', { fog: 0.028 } );
		eng.setBackground( null );
		eng.fitShadow( box );
		nightKey( eng, box, hasOwn( main ) );
		frame( eng, box, { az: wallMount ? 32 : 36, el: big() ? 6 : 13, aspect: 1.6, zoom: 0.92 } );
		await shoot( eng, ctx, 0, 0, 1600, 1000, 4 );
		label( ctx, 14, 14, `${job.id} / ${names[ 0 ]}   night hero\n${info( main )}` );
		files.push( await upload( c, `renders/${job.id}/hero.png`, job.token ) );
		show( main, false );

	}

	// ---- sheet: studio front, studio back/side, close-up night, clay + scale figure
	{

		const W = 800, H = 600;
		const [ c, ctx ] = canvas2d( W * 2, H * 2 );
		// studio front 3/4
		show( main, true, { lights: false } );
		let box = boxOf( main );
		stage.key.visible = false; stage.fill.visible = false;
		eng.setRig( 'studio', { fog: 0.022, fogColor: '#8d9299' } );
		eng.setBackground( '#8d9299' );
		eng.fitShadow( box );
		frame( eng, box, { az: - 28, el: 16, aspect: W / H } );
		await shoot( eng, ctx, 0, 0, W, H );
		label( ctx, 10, 10, 'studio daylight - front 3/4' );
		// studio back (wall mounted: steep side view instead)
		frame( eng, box, { az: wallMount ? 78 : 152, el: 20, aspect: W / H } );
		await shoot( eng, ctx, W, 0, W, H );
		label( ctx, W + 10, 10, wallMount ? 'studio daylight - side' : 'studio daylight - back 3/4' );
		show( main, false );
		// close-up, night
		show( main, true, { shadowLights: 2 } );
		box = boxOf( main );
		eng.setRig( 'night', { fog: 0.028 } );
		eng.setBackground( null );
		eng.fitShadow( box );
		nightKey( eng, box, hasOwn( main ), - 1 );
		const sz = box.getSize( new THREE.Vector3() );
		// META focus: [x, y, z] in asset space (Blender axes) or a 0..1 fraction of the height
		const fh = typeof meta.focus === 'number' ? meta.focus : 0.68;
		const focus = Array.isArray( meta.focus ) ? B2T( meta.focus ).add( new THREE.Vector3( 0, lift, 0 ) )
			: new THREE.Vector3( ( box.min.x + box.max.x ) / 2, box.min.y + sz.y * fh, ( box.min.z + box.max.z ) / 2 + sz.z * 0.25 );
		frame( eng, box, { az: - 18, el: 8, aspect: W / H, zoom: 2.8, target: focus } );
		await shoot( eng, ctx, 0, H, W, H );
		label( ctx, 10, H + 10, 'close-up - night' );
		show( main, false );
		// clay + scale figure
		show( main, true, { lights: false } );
		box = boxOf( main );
		for ( const f of eng.flames ) f.visible = false;
		stage.key.visible = false; stage.fill.visible = false;
		stage.fig.visible = true;
		stage.fig.position.set( box.max.x + 0.55, 0, Math.min( box.max.z, 0.4 ) + 0.1 );
		const box2 = box.clone().expandByObject( stage.fig );
		eng.setRig( 'studio', { ao: 1, fog: 0.022, fogColor: '#6f737a' } );
		eng.setBackground( '#6f737a' );
		eng.fitShadow( box2 );
		eng.scene.overrideMaterial = eng.lib.clay();
		frame( eng, box2, { az: 24, el: 12, aspect: W / H } );
		await shoot( eng, ctx, W, H, W, H );
		eng.scene.overrideMaterial = null;
		stage.fig.visible = false;
		label( ctx, W + 10, H + 10, `clay - 1.8 m figure for scale\n${info( main )}` );
		show( main, false );
		files.push( await upload( c, `renders/${job.id}/sheet.png`, job.token ) );

	}

	// ---- variants: night, same framing scale
	{

		const vs = names.slice( 0, 6 );
		const cols = vs.length <= 3 ? vs.length : Math.ceil( vs.length / 2 ), rows = vs.length <= 3 ? 1 : 2;
		const W = Math.floor( 1600 / cols ), H = rows === 1 ? Math.min( 900, Math.round( W * 1.15 ) ) : 560;
		const [ c, ctx ] = canvas2d( W * cols, H * rows );
		// common box so relative scale is visible
		const common = new THREE.Box3();
		for ( const v of vs ) {

			insts[ v ].root.position.set( 0, liftOf( insts[ v ] ), 0 );
			insts[ v ].root.updateMatrixWorld( true );
			common.union( boxOf( insts[ v ] ) );

		}

		eng.setRig( 'night', { fog: 0.028 } );
		eng.setBackground( null );
		for ( let i = 0; i < vs.length; i ++ ) {

			const inst = insts[ vs[ i ] ];
			show( inst, true, { shadowLights: 1 } );
			eng.fitShadow( common );
			nightKey( eng, common, hasOwn( inst ) );
			frame( eng, common, { az: 30 - ( i % 2 ) * 55, el: big() ? 6 : 12, aspect: W / H } );
			const x = ( i % cols ) * W, y = Math.floor( i / cols ) * H;
			await shoot( eng, ctx, x, y, W, H );
			label( ctx, x + 10, y + 10, `${vs[ i ]}\n${info( inst )}`, { size: 13 } );
			show( inst, false );

		}

		files.push( await upload( c, `renders/${job.id}/variants.png`, job.token ) );

	}

	// free GPU memory of this asset
	for ( const v of job.variants ) {

		insts[ v ].root.traverse( ( o ) => {

			if ( o.isMesh ) o.geometry.dispose();

		} );

	}

	stage.key.visible = false; stage.fill.visible = false;
	for ( const w of eng.lib.warnings ) warnings.push( w );
	eng.lib.warnings.clear();
	return { files, warnings };

}

// ---------------------------------------------------------------------------------- material job

function sphereMetres( r ) {

	const g = new THREE.SphereGeometry( r, 96, 64 );
	const uvA = g.getAttribute( 'uv' );
	for ( let i = 0; i < uvA.count; i ++ ) uvA.setXY( i, uvA.getX( i ) * 2 * Math.PI * r, 1 - uvA.getY( i ) * Math.PI * r );
	return g;

}

function boxMetres( w, h, d ) {

	const g = new THREE.BoxGeometry( w, h, d );
	const p = g.getAttribute( 'position' ), n = g.getAttribute( 'normal' ), uvA = g.getAttribute( 'uv' );
	for ( let i = 0; i < p.count; i ++ ) {

		const nx = Math.abs( n.getX( i ) ), ny = Math.abs( n.getY( i ) );
		let u, v;
		if ( nx > 0.5 ) {

			u = p.getZ( i ); v = p.getY( i );

		} else if ( ny > 0.5 ) {

			u = p.getX( i ); v = - p.getZ( i );

		} else {

			u = p.getX( i ); v = p.getY( i );

		}

		uvA.setXY( i, u, 1 - v );

	}

	return g;

}

async function runMat( job ) {

	const eng = await getEngine();
	const name = job.id;
	await eng.lib.prepare( [ name ], job.matVersions || {} );
	const entry = eng.lib.catalog.get( name );
	if ( ! entry ) throw new Error( 'unknown material ' + name );
	const tints = [ null ];
	if ( entry.tintable && entry.kind === 'baked' ) tints.push( [ 0.05, 0.16, 0.1 ], [ 0.3, 0.05, 0.04 ] );
	const g = new THREE.Group();
	const floor = new THREE.Mesh( new THREE.PlaneGeometry( 60, 60 ).rotateX( - Math.PI / 2 ) );
	metreUVs( floor.geometry, ( x, y, z ) => [ x, - z ] );
	const wallG = new THREE.PlaneGeometry( 30, 8 );
	const wallM = new THREE.Mesh( wallG );
	wallM.position.set( 0, 4, - 1.6 );
	metreUVs( wallG, ( x, y ) => [ x, y ] );
	const sph = new THREE.Mesh( sphereMetres( 0.5 ) );
	sph.position.set( - 0.75, 0.5, 0 );
	const cube = new THREE.Mesh( boxMetres( 0.9, 0.9, 0.9 ) );
	cube.position.set( 0.7, 0.45, - 0.1 );
	cube.rotation.y = 0.5;
	// weathered copies show the grime / wear / special response
	const sph2 = new THREE.Mesh( sphereMetres( 0.3 ) );
	sph2.position.set( 0.05, 0.3, 0.75 );
	for ( const o of [ floor, wallM, sph, cube ] ) setMask( o.geometry, 1, 0.12, 0.1, 0.05 );
	{

		const geo = sph2.geometry, p = geo.getAttribute( 'position' ), a = new Float32Array( p.count * 4 );
		for ( let i = 0; i < p.count; i ++ ) {

			const yy = p.getY( i ) / 0.3, xx = p.getX( i ) / 0.3;
			a[ i * 4 ] = 1; a[ i * 4 + 1 ] = THREE.MathUtils.clamp( 0.6 - yy, 0, 1 );
			a[ i * 4 + 2 ] = THREE.MathUtils.clamp( xx * 1.2, 0, 1 ); a[ i * 4 + 3 ] = THREE.MathUtils.clamp( - xx * 1.2, 0, 1 );

		}

		geo.setAttribute( '_mask', new THREE.BufferAttribute( a, 4 ) );

	}

	for ( const o of [ floor, wallM, sph, cube, sph2 ] ) {

		o.castShadow = o !== floor; o.receiveShadow = true; g.add( o );

	}

	stage.group.visible = false;
	eng.scene.add( g );
	const W = 800, H = 560;
	const [ c, ctx ] = canvas2d( W * tints.length, H * 2 );
	const box = new THREE.Box3( new THREE.Vector3( - 1.3, 0, - 0.6 ), new THREE.Vector3( 1.3, 1.05, 1.0 ) );
	for ( let i = 0; i < tints.length; i ++ ) {

		for ( const o of [ floor, wallM, sph, cube, sph2 ] ) eng.lib.apply( o, name, tints[ i ], {}, true );
		eng.setRig( 'night', { fog: 0.028 } );
		eng.setBackground( null );
		eng.fitShadow( box );
		nightKey( eng, box, false );
		frame( eng, box, { az: 12, el: 16, aspect: W / H, zoom: 1.0 } );
		await shoot( eng, ctx, i * W, 0, W, H );
		label( ctx, i * W + 10, 10, `${name}${tints[ i ] ? ' (tinted)' : ''} - night lantern` );
		stage.key.visible = false; stage.fill.visible = false;
		eng.setRig( 'studio', { fog: 0.022, fogColor: '#8d9299' } );
		eng.setBackground( '#8d9299' );
		eng.fitShadow( box );
		frame( eng, box, { az: - 20, el: 22, aspect: W / H, zoom: 1.0 } );
		await shoot( eng, ctx, i * W, H, W, H );
		label( ctx, i * W + 10, H + 10, 'studio daylight (small sphere: grime below, wear right, special left)', { size: 12 } );

	}

	eng.scene.remove( g );
	for ( const o of [ floor, wallM, sph, cube, sph2 ] ) o.geometry.dispose();
	stage.group.visible = true;
	const warnings = [ ...eng.lib.warnings ];
	eng.lib.warnings.clear();
	return { files: [ await upload( c, `renders/_mat/${name}.png`, job.token ) ], warnings };

}

// ---------------------------------------------------------------------------------- lod comparison job
// tools/tsq lod <asset>: the decimated levels next to the ones in <variant>.lod1.glb / .lod2.glb (docs/LOD.md),
// lantern-lit like the asset review. Top row: level 0 and both LOD1 enlarged; bottom row: true-size crops (what a
// 1080p town view shows at the distance each level is used at) and both LOD2 enlarged. The right-hand column is
// titled with the mode the sidecar says ships at that level: "authored", "reduced" or "decimated ships".

const LOD_DIST = [ 0, 16, 44 ], TOWN_FOV = 62, TOWN_H = 1080, TOWN_FOG = 0.0075;

async function runLod( job ) {

	const eng = await getEngine();
	const warnings = [];
	const mv = job.matVersions || {};
	await stageMaterials( eng, mv );
	const v = job.variant || 'default', ver = job.version || 0;
	const url = ( suffix ) => `${BASE}/${job.dir}/${v}${suffix}?v=${ver}`;
	const meta0 = await fetch( url( '.json' ), { cache: 'no-cache' } ).then( ( r ) => ( r.ok ? r.json() : null ) ).catch( () => null );
	if ( ! meta0 ) throw new Error( `no sidecar ${job.dir}/${v}.json (run tools/tsq lod without --no-build)` );
	const loadOne = async ( suffix ) => {

		try {

			const gltf = await eng.assets.gltf.loadAsync( url( suffix ) );
			await eng.lib.prepare( MaterialLib.libsOf( gltf.scene ), mv );
			return eng.assets.adopt( gltf.scene, meta0 );

		} catch ( e ) {

			return null;

		}

	};

	const lod0 = await loadOne( '.glb' );
	if ( ! lod0 ) throw new Error( `missing ${job.dir}/${v}.glb` );
	// level files hold meshes only: lights / flames come from level 0, exactly as in the town runtime
	const level = async ( suffix ) => {

		const i = await loadOne( suffix );
		if ( i ) i.sockets = lod0.sockets;
		return i;

	};

	const dec = [ null, await level( '.dec1.glb' ), await level( '.dec2.glb' ) ];
	const mod = [ null, await level( '.lod1.glb' ), await level( '.lod2.glb' ) ];
	const meta = meta0.meta || {};
	const mount = meta.mount || 'ground';
	const wallMount = mount === 'wall' || mount === 'wall_ground';
	const mnz = meta0.bounds?.min?.[ 2 ] ?? 0;
	const lift = mount === 'wall' ? ( meta.mount_height ?? 2.6 ) : ( mount === 'hang' || mnz < - 0.08 ) ? ( meta.mount_height ?? Math.max( 2.4, - mnz + 1.2 ) ) : 0;
	stage.wall.visible = wallMount;
	const own = lod0.sockets.some( ( s ) => s.kind === 'light' );
	let scale = 1;
	{

		let e = 0;
		for ( const sk of lod0.sockets ) {

			if ( sk.kind !== 'light' ) continue;
			const I = sk.props.intensity ?? 40, hgt = Math.max( sk.position.y + lift, 0.5 );
			e = Math.max( e, I / ( hgt * hgt ) );
			if ( wallMount ) e = Math.max( e, I / Math.max( sk.position.z, 0.6 ) ** 2 * 0.5 );

		}

		if ( e > 10 ) scale = 10 / e;

	}

	const show = ( inst, on ) => {

		if ( on ) {

			inst.root.position.set( 0, lift, 0 );
			eng.scene.add( inst.root );
			inst.root.updateMatrixWorld( true );
			eng.activate( inst, { lights: true, fx: true, halos: true, shadowLights: 1, intensity: scale } );

		} else {

			eng.deactivate( inst );
			eng.scene.remove( inst.root );

		}

	};

	lod0.root.position.set( 0, lift, 0 );
	lod0.root.updateMatrixWorld( true );
	const box = new THREE.Box3().setFromObject( lod0.root, true );
	const size = box.getSize( new THREE.Vector3() );
	const big = size.length() > 9;
	const t0 = lod0.tris;
	const pct = ( t ) => `${t.toLocaleString()} tris (${( 100 * t / t0 ).toFixed( 0 )} %)`;
	const st = meta0.lod || [];
	// what the right-hand column shows, from the sidecar: the mode that ships at level n (stats.lods[].mode, lod[].ships
	// / .mode), how far over its budget (lod_pick[].over), and whether the file shown is a level that was turned down
	// (comparison builds keep it next to the decimated copy that ships instead)
	const shipped = ( n ) => {

		const r = st[ n - 1 ] || {}, sl = meta0.stats?.lods?.[ n - 1 ] || {};
		const pk = meta0.lod_pick?.[ n - 1 ] || meta0.lod_eval?.levels?.[ n ]?.pick || {};
		const mode = r.ships || r.mode || pk.kind || ( sl.mode !== 'rejected' ? sl.mode : null ) || null;
		const shown = sl.mode === 'rejected' ? ( meta0.meta?.lod ? 'authored' : 'reduced' ) : mode;
		const over = pk.over ? ` (${pk.over} x budget)` : '';
		if ( ! mode ) return { title: 'shipping', short: 'ships', turnedDown: false };
		if ( sl.mode === 'rejected' && shown !== mode ) return { title: `${shown} (turned down, ${mode} ships)`, short: `${shown}, turned down`, turnedDown: true };
		if ( mode === 'decimated' ) return { title: 'decimated ships', short: 'decimated ships', turnedDown: false };
		return { title: mode + over, short: mode, turnedDown: false };

	};

	const PW = 640, PH = 520;
	const [ c, ctx ] = canvas2d( PW * 3, PH * 2 );
	const night = ( fog ) => {

		eng.setRig( 'night', { fog } );
		eng.setBackground( null );
		eng.fitShadow( box );
		nightKey( eng, box, own );

	};

	const none = ( x, y, w, h, text ) => {

		ctx.fillStyle = '#17181c';
		ctx.fillRect( x, y, w, h );
		label( ctx, x + 10, y + h / 2 - 12, text, { size: 13, color: '#c9b8a0' } );

	};

	// ---- enlarged: the same framing for every level
	const enlarged = async ( inst, x, y, title, n, isNew ) => {

		if ( ! inst ) return none( x, y, PW, PH, `${title}: no file` );
		show( inst, true );
		night( 0.028 );
		frame( eng, box, { az: wallMount ? 32 : 36, el: big ? 6 : 13, aspect: PW / PH, zoom: 0.95 } );
		await shoot( eng, ctx, x, y, PW, PH, 3 );
		show( inst, false );
		label( ctx, x + 10, y + 10, `${title}\n${n ? pct( inst.tris ) : inst.tris.toLocaleString() + ' tris'}`, { size: 14, color: isNew ? '#bfe8b0' : '#f2e6d0' } );
		const r = isNew && n ? st[ n - 1 ] : null;
		if ( r && ! r.ok ) label( ctx, x + 10, y + PH - 52, ( shipped( n ).turnedDown ? 'REJECTED (decimated copy ships):\n' : 'no other level passed:\n' ) + String( ( r.why || [] )[ 0 ] || '' ).slice( 0, 84 ), { size: 12, color: '#ff9a8a' } );

	};

	await enlarged( lod0, 0, 0, `${job.id} / ${v}   LOD0 (full model)`, 0, false );
	await enlarged( dec[ 1 ], PW, 0, 'decimated LOD1', 1, false );
	await enlarged( mod[ 1 ], PW * 2, 0, `LOD1: ${shipped( 1 ).title}`, 1, true );
	await enlarged( dec[ 2 ], PW, PH, 'decimated LOD2', 2, false );
	await enlarged( mod[ 2 ], PW * 2, PH, `LOD2: ${shipped( 2 ).title}`, 2, true );

	// ---- true size: 1:1 pixels of a 1080p town view (fov 62) at the distance where the level takes over
	const CW = PW / 2, CH = PH / 2;
	const fovCrop = 2 * THREE.MathUtils.radToDeg( Math.atan( Math.tan( THREE.MathUtils.degToRad( TOWN_FOV / 2 ) ) * CH / TOWN_H ) );
	const center = box.getCenter( new THREE.Vector3() );
	const trueSize = async ( inst, x, y, title, n, isNew ) => {

		if ( ! inst ) return none( x, y, CW, CH, `${title}: no file` );
		show( inst, true );
		night( TOWN_FOG );
		const d = LOD_DIST[ n ] + Math.min( size.x, size.z ) * 0.5;
		const cam = eng.camera;
		// a walker's eye (1.7 m) looking at the part of the asset that fills the crop
		const hView = 2 * d * Math.tan( THREE.MathUtils.degToRad( fovCrop / 2 ) );
		const ty = size.y <= hView ? center.y : Math.min( box.max.y - hView / 2, Math.max( box.min.y + hView / 2, 1.7 + hView * 0.2 ) );
		const tgt = new THREE.Vector3( center.x, ty, center.z );
		const a = THREE.MathUtils.degToRad( wallMount ? 32 : 36 );
		cam.fov = fovCrop;
		cam.aspect = CW / CH;
		cam.position.set( tgt.x + Math.sin( a ) * d, 1.7, tgt.z + Math.cos( a ) * d );
		cam.near = 0.2; cam.far = 1500;
		cam.lookAt( tgt );
		cam.updateProjectionMatrix();
		cam.updateMatrixWorld();
		await shoot( eng, ctx, x, y, CW, CH, 3 );
		show( inst, false );
		label( ctx, x + 6, y + 6, `${title} @ ${LOD_DIST[ n ]} m  1:1\n${pct( inst.tris )}`, { size: 11, color: isNew ? '#bfe8b0' : '#f2e6d0' } );

	};

	await trueSize( dec[ 1 ], 0, PH, 'decimated LOD1', 1, false );
	await trueSize( mod[ 1 ], CW, PH, `LOD1 ${shipped( 1 ).short}`, 1, true );
	await trueSize( dec[ 2 ], 0, PH + CH, 'decimated LOD2', 2, false );
	await trueSize( mod[ 2 ], CW, PH + CH, `LOD2 ${shipped( 2 ).short}`, 2, true );
	ctx.strokeStyle = '#3a3c44';
	ctx.lineWidth = 1;
	for ( const [ x0, y0, x1, y1 ] of [ [ PW, 0, PW, PH * 2 ], [ PW * 2, 0, PW * 2, PH * 2 ], [ 0, PH, PW * 3, PH ], [ CW, PH, CW, PH * 2 ], [ 0, PH + CH, PW, PH + CH ] ] ) {

		ctx.beginPath(); ctx.moveTo( x0 + 0.5, y0 + 0.5 ); ctx.lineTo( x1 + 0.5, y1 + 0.5 ); ctx.stroke();

	}

	const files = [ await upload( c, `renders/_lod/${job.id}${v === 'default' ? '' : '.' + v}.png`, job.token ) ];
	const seen = new Set();
	for ( const inst of [ lod0, ...dec, ...mod ] ) {

		if ( ! inst ) continue;
		inst.root.traverse( ( o ) => {

			if ( o.isMesh && ! seen.has( o.geometry ) ) {

				seen.add( o.geometry ); o.geometry.dispose();

			}

		} );

	}

	stage.key.visible = false; stage.fill.visible = false;
	for ( const w of eng.lib.warnings ) warnings.push( w );
	eng.lib.warnings.clear();
	return { files, warnings, tris: { lod0: t0, dec: dec.slice( 1 ).map( ( i ) => i?.tris ?? null ), modelled: mod.slice( 1 ).map( ( i ) => i?.tris ?? null ) } };

}

// ---------------------------------------------------------------------------------- job api

window.startJob = ( job ) => {

	const rec = jobs[ job.job_id ] = { state: 'running', t0: performance.now() };
	if ( ( job.pageVersion || 0 ) > PAGE_VERSION ) {

		rec.state = 'done'; rec.result = { ok: false, error: 'stale page' };
		return 'started';

	}

	const fn = job.kind === 'asset' ? runAsset : job.kind === 'mat' ? runMat : job.kind === 'lod' ? runLod : null;
	if ( ! fn ) {

		rec.state = 'done'; rec.result = { ok: false, error: 'unknown job kind ' + job.kind };
		return 'started';

	}

	const mark = window.__log.length;
	fn( job ).then( ( r ) => {

		rec.result = { ok: true, ...r, seconds: +( ( performance.now() - rec.t0 ) / 1000 ).toFixed( 2 ), log: window.__log.slice( mark ).slice( 0, 12 ) };
		rec.state = 'done';

	} ).catch( ( e ) => {

		rec.result = { ok: false, error: String( e && e.stack || e ), log: window.__log.slice( mark ).slice( 0, 20 ) };
		rec.state = 'done';
		// leave the scene clean for the next job
		try {

			eng.scene.overrideMaterial = null;
			for ( const o of [ ...eng.scene.children ] ) if ( o !== stage.group && o !== stage.fig && o !== stage.key && o !== stage.fill && o !== eng.sky && o !== eng.hemi && o !== eng.moon && o !== eng.moon.target ) eng.scene.remove( o );
			eng.lanterns.length = 0; eng.flames.length = 0; eng.partSets.length = 0;
			stage.group.visible = true; stage.fig.visible = false;

		} catch ( e2 ) { /* ignore */ }

	} );
	return 'started';

};

window.jobStatus = ( id ) => {

	const rec = jobs[ id ];
	if ( ! rec ) return JSON.stringify( { state: 'unknown' } );
	if ( rec.state === 'done' ) {

		delete jobs[ id ];
		return JSON.stringify( { state: 'done', result: rec.result } );

	}

	return JSON.stringify( { state: rec.state } );

};

window.__dbg = { getEngine, frame, nightKey, stageMaterials, get stage() { return stage; } };
window.renderReady = true;
