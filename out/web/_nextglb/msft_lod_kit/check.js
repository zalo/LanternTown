// check.js - loads every model of expected.json with the loader under test and reports, per expectation, pass or fail.
// See check.html for the query parameters and README.md for what each expectation means.
//
// How it looks at a loader, so that it does not depend on how the loader is built:
//   - an LOD is found as the nearest ancestor with .isLOD of an object that carries the name of one of its levels
//   - "level i is drawn": an object named like the level's node exists, and it and everything above it is visible
//   - a frame: renderer.render( scene, camera ) of a WebGLRenderer - three.js itself calls LOD.update( camera )
//   - what is fetched: window.fetch is wrapped (GLTFLoader's FileLoader uses fetch), per URL below models/

import * as THREE from 'three';

const q = new URLSearchParams( location.search );
const abs = ( u ) => new URL( u, location.href ).href;
const TIER = ( q.get( 'tier' ) || 'b' ).toLowerCase() === 'a' ? 'a' : 'b';
const LOADER = q.get( 'loader' ) ? abs( q.get( 'loader' ) ) : 'three/addons/loaders/GLTFLoader.js';
const SETUP = q.get( 'setup' ) ? abs( q.get( 'setup' ) ) : null;
const ONLY = q.get( 'model' );
const WAIT = + ( q.get( 'wait' ) || 8000 ); // ms a lazy level may take to arrive
const MODELS = abs( 'models/' );

const sleep = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );
const near = ( a, b, eps = 1e-3 ) => a.length === b.length && a.every( ( x, i ) => Math.abs( x - b[ i ] ) <= eps * Math.max( 1, Math.abs( b[ i ] ) ) );
const sanitize = ( name ) => name.replace( /\s/g, '_' ).replace( /[\[\]\.:\/]/g, '' ); // as GLTFLoader names its objects

// ------------------------------------------------------------------------------------------------ network

const net = { log: [], inFlight: 0, delay: () => 0, strip: () => false };
const realFetch = window.fetch.bind( window );
window.fetch = async function ( input, init = {} ) {

	const isReq = typeof Request !== 'undefined' && input instanceof Request;
	const url = abs( isReq ? input.url : String( input ) );
	if ( ! url.startsWith( MODELS ) ) return realFetch( input, init );
	const headers = new Headers( init.headers || ( isReq ? input.headers : undefined ) );
	const entry = { key: url.slice( MODELS.length ).split( '?' )[ 0 ], range: headers.get( 'range' ), status: 0, bytes: 0, t: performance.now(), done: false };
	net.log.push( entry );
	net.inFlight ++;
	try {

		if ( net.strip( entry.key ) ) { headers.delete( 'range' ); entry.stripped = true; }
		const wait = net.delay( entry.key );
		if ( wait ) await sleep( wait );
		const res = await realFetch( url, { method: 'GET', headers, cache: 'no-store', signal: init.signal || ( isReq ? input.signal : undefined ) } );
		const body = await res.arrayBuffer();
		entry.status = res.status; entry.bytes = body.byteLength; entry.done = true;
		return new Response( body, { status: res.status, statusText: res.statusText, headers: res.headers } );

	} finally { net.inFlight --; }

};

const requests = ( from = 0 ) => { const n = {}; for ( const e of net.log.slice( from ) ) n[ e.key ] = ( n[ e.key ] || 0 ) + 1; return n; };

// ------------------------------------------------------------------------------------------------ view

const canvas = document.getElementById( 'view' );
let renderer = null;
try { renderer = new THREE.WebGLRenderer( { canvas, antialias: true } ); renderer.setSize( 640, 360, false ); } catch ( e ) { console.warn( 'no WebGL: LOD.update( camera ) is called by hand', e ); }
const view = new THREE.Scene();
view.add( new THREE.HemisphereLight( 0xffffff, 0x555566, 2.5 ) );
const camera = new THREE.PerspectiveCamera( 50, 16 / 9, 0.05, 1e7 );

function frame() {

	camera.updateMatrixWorld();
	if ( renderer ) renderer.render( view, camera );
	else { view.updateMatrixWorld(); view.traverse( ( o ) => { if ( o.isLOD && o.autoUpdate ) o.update( camera ); } ); }

}

async function until( cond, ms = WAIT ) {

	const t0 = performance.now();
	for ( ;; ) {

		frame();
		if ( cond() ) return true;
		if ( performance.now() - t0 > ms ) return false;
		await sleep( 16 );

	}

}

// frames until nothing is being fetched and what is drawn has not changed for a while
async function settle( what, quiet = 250, ms = WAIT ) {

	const t0 = performance.now();
	let last = null, since = t0;
	for ( ;; ) {

		frame();
		const now = performance.now(), s = JSON.stringify( what() );
		if ( s !== last || net.inFlight > 0 ) { last = s; since = now; }
		if ( now - since >= quiet || now - t0 > ms ) return JSON.parse( s );
		await sleep( 16 );

	}

}

const DIR = new THREE.Vector3( 0.25, 0.3, 1 ).normalize();
function look( center, distance ) {

	const c = new THREE.Vector3().fromArray( center );
	camera.position.copy( c ).addScaledVector( DIR, distance );
	camera.near = Math.max( 0.01, distance * 0.01 ); camera.far = distance * 10 + 1000;
	camera.updateProjectionMatrix();
	camera.lookAt( c );

}

// ------------------------------------------------------------------------------------------------ looking at a scene

const named = ( root, name ) => { const s = sanitize( name ), out = []; root.traverse( ( o ) => { if ( o.name === s ) out.push( o ); } ); return out; };
const shown = ( o, root ) => { for ( let p = o; p; p = p.parent ) { if ( ! p.visible ) return false; if ( p === root ) return true; } return false; };
const lodAbove = ( o ) => { for ( let p = o; p; p = p.parent ) if ( p.isLOD ) return p; return null; };

// the levels of an expected LOD that are drawn now: [ level index, ... ]
function drawn( root, lod ) {

	const out = [];
	lod.levels.forEach( ( name, i ) => {

		if ( lod.kind === 'material' ) { if ( named( root, lod.node_name ).some( ( o ) => o.isMesh && o.material && o.material.name === name && shown( o, root ) ) ) out.push( i ); }
		else { const n = named( root, name ).filter( ( o ) => shown( o, root ) ).length; for ( let k = 0; k < n; k ++ ) out.push( i ); }

	} );
	return out;

}

function lodObject( root, lod ) {

	const names = lod.kind === 'material' ? [ lod.node_name ] : lod.levels;
	for ( const name of names ) for ( const o of named( root, name ) ) { const l = lodAbove( o ); if ( l ) return l; }
	return null;

}

const TINT = [ 0xe0281a, 0x2ac23a, 0x2a50f0, 0xf0d820 ];
const tinted = new WeakSet();
function tint( root, model ) {

	// a model whose levels look alike (the town cells): level 0 red, 1 green, 2 blue, 3 yellow
	if ( ! model.needs ) return;
	for ( const lod of model.lods ) lod.levels.forEach( ( name, i ) => { for ( const o of named( root, name ) ) o.traverse( ( m ) => { if ( m.isMesh && ! tinted.has( m ) ) { tinted.add( m ); m.material = m.material.clone(); m.material.color.setHex( TINT[ i % 4 ] ); } } ); } );

}

// ------------------------------------------------------------------------------------------------ the checks of one model

async function checkModel( model, GLTFLoader, setup, meshopt ) {

	const results = [], obs = {};
	const add = ( id, tier, pass, detail = '', optional = false ) => results.push( { model: model.file, id, tier, pass: !! pass, detail: String( detail ), optional: optional || !! model.optional } );
	const url = MODELS + model.file;
	const lazy = TIER === 'b';
	const nodeLods = model.lods.filter( ( l ) => l.kind === 'node' );

	const newLoader = () => { const loader = new GLTFLoader(); if ( meshopt && model.needs ) loader.setMeshoptDecoder( meshopt ); if ( setup && setup.configure ) setup.configure( loader, { tier: TIER, THREE } ); return loader; };
	const load = async ( byRanges ) => ( byRanges ? setup.loadRanged( newLoader(), url, { probe: model.ranged.probe } ) : newLoader().loadAsync( url ) );
	const show = ( gltf ) => { for ( const c of view.children.slice() ) if ( ! c.isLight ) view.remove( c ); view.add( gltf.scene ); };
	const all = model.lods.map( ( l ) => l.world_center );
	const mid = all.length ? all.reduce( ( s, c ) => s.map( ( x, i ) => x + c[ i ] / all.length ), [ 0, 0, 0 ] ) : [ 0, 0, 0 ];
	const size = Math.max( 1, ...model.lods.map( ( l ) => l.world_radius + Math.hypot( ...l.world_center.map( ( x, i ) => x - mid[ i ] ) ) ) );
	const far = () => look( mid, size * 2000 );
	const close = ( lod ) => look( lod.world_center, lod.world_radius * 1.05 );

	net.log.length = 0; net.delay = () => 0; net.strip = () => false;
	const ranged = lazy && !! model.ranged;
	let gltf;
	try {

		if ( ranged && ! ( setup && setup.loadRanged ) ) { add( 'ranged-entry', 'b', false, 'the adapter module exports no loadRanged( loader, url, options )' ); gltf = await load( false ); } else gltf = await load( ranged );

	} catch ( e ) { add( 'loads', 'a', false, e && e.message ? e.message : e ); return { results, obs }; }

	const root = gltf.scene;
	const atLoad = net.log.length, loadBytes = net.log.reduce( ( s, e ) => s + e.bytes, 0 );
	show( gltf );
	let lods = 0, meshes = 0;
	root.traverse( ( o ) => { if ( o.isLOD ) lods ++; if ( o.isMesh ) meshes ++; } );
	obs.after_load = { LOD_objects: lods, meshes, requests: requests() };

	// ---- tier B: what is fetched, and when
	if ( lazy && ( model.lazy || ranged ) && nodeLods.length ) {

		const first = nodeLods[ 0 ];
		const fine = ranged ? [ model.file ] : first.level_uris[ 0 ];
		if ( model.lazy ) {

			const allowed = new Set( [ model.file, ...model.lazy.initial_buffers ] );
			const extra = Object.keys( requests() ).filter( ( k ) => ! allowed.has( k ) );
			add( 'initial-requests', 'b', extra.length === 0 && model.lazy.initial_buffers.every( ( b ) => requests()[ b ] === 1 ), extra.length ? 'fetched before the first frame: ' + extra.join( ', ' ) : 'only ' + [ ...allowed ].join( ', ' ) );

		}

		if ( ranged ) {

			const e0 = net.log[ 0 ];
			add( 'first-request-is-a-small-range', 'b', e0 && !! e0.range && e0.status === 206 && e0.bytes <= Math.max( model.ranged.probe, model.glb.head_bytes ) && e0.bytes < model.glb.bytes / 2, e0 ? `Range: ${e0.range}, ${e0.status}, ${e0.bytes} of ${model.glb.bytes} bytes` : 'no request seen' );
			add( 'bytes-before-first-frame', 'b', loadBytes <= model.ranged.initial_bytes_max, `${loadBytes} bytes in ${atLoad} requests (limit ${model.ranged.initial_bytes_max}; the file: ${model.glb.bytes})` );

		}

		far();
		for ( let i = 0; i < 20; i ++ ) { frame(); await sleep( 16 ); }
		add( 'far-fetches-nothing', 'b', net.log.length === atLoad, net.log.length === atLoad ? `${atLoad} requests, as after load` : 'fetched with the camera far away: ' + net.log.slice( atLoad ).map( ( e ) => e.key + ( e.range ? ' ' + e.range : '' ) ).join( ', ' ) );
		const farDrawn = drawn( root, first );
		add( 'far-draws-coarsest', 'b', farDrawn.length === 1 && farDrawn[ 0 ] === first.level_has_mesh.lastIndexOf( true ) || ( farDrawn.length === 1 && farDrawn[ 0 ] === first.levels.length - 1 ), 'drawn: ' + JSON.stringify( farDrawn.map( ( i ) => first.levels[ i ] ) ) );

		// the fine level held back for a moment: the coarse one must stay on screen
		const before = net.log.length;
		net.delay = ( key ) => ( fine.includes( key ) ? 600 : 0 );
		close( first );
		for ( let i = 0; i < 12; i ++ ) { frame(); await sleep( 16 ); }
		const meanwhile = drawn( root, first );
		add( 'coarse-stays-until-fine-arrives', 'b', meanwhile.length >= 1 && ! meanwhile.includes( 0 ), 'while the fine level loads: ' + JSON.stringify( meanwhile.map( ( i ) => first.levels[ i ] ) ) );
		const arrived = await until( () => { const d = drawn( root, first ); return d.length === 1 && d[ 0 ] === 0; } );
		net.delay = () => 0;
		const asked = net.log.slice( before );
		add( 'near-fetches-the-fine-level', 'b', arrived && asked.length > 0 && ( ranged ? asked.every( ( e ) => ! e.done || e.status === 206 ) : fine.every( ( f ) => asked.filter( ( e ) => e.key === f ).length === 1 ) ), ( arrived ? 'level 0 drawn after ' : 'level 0 NOT drawn; ' ) + asked.map( ( e ) => e.key + ( e.range ? ' ' + e.range : '' ) ).join( ', ' ).slice( 0, 300 ) );

		await settle( () => drawn( root, first ) );
		const mark = net.log.length;
		far(); await settle( () => drawn( root, first ) );
		close( first ); await settle( () => drawn( root, first ) );
		add( 'away-and-back-fetches-nothing', 'b', net.log.length === mark, net.log.length === mark ? 'no request' : net.log.slice( mark ).map( ( e ) => e.key + ( e.range ? ' ' + e.range : '' ) ).join( ', ' ).slice( 0, 300 ) );

	}

	// ---- tier A: an LOD in place of the node, switched by three.js
	add( 'lod-count', 'a', ( () => { let n = 0; root.traverse( ( o ) => { if ( o.isLOD ) n ++; } ); obs.LOD_objects = n; return n === model.lod_count; } )(), `${obs.LOD_objects} LOD objects in the scene, ${model.lod_count} expected` );

	for ( const lod of model.lods ) {

		const tag = model.lods.length > 1 ? ` [${lod.levels[ 0 ]}]` : '';
		const n = lod.levels.length, last = n - 1;
		const state = () => drawn( root, lod );
		const names = ( d ) => JSON.stringify( d.map( ( i ) => lod.levels[ i ] ) );

		close( lod );
		const dNear = await settle( state );
		tint( root, model );
		add( 'near-draws-level-0' + tag, 'a', dNear.length === 1 && dNear[ 0 ] === 0, 'drawn: ' + names( dNear ) );
		if ( lod === model.lods[ 0 ] ) obs.drawn_near = dNear.map( ( i ) => lod.levels[ i ] );

		look( lod.world_center, lod.world_radius * 5000 );
		const dFar = await settle( state );
		tint( root, model );
		const culled = lod.reference_cull !== null && lod.reference_cull !== undefined;
		add( 'far-draws-the-last-level' + tag, 'a', ( dFar.length === 1 && dFar[ 0 ] === last ) || ( culled && dFar.length === 0 ), 'drawn: ' + names( dFar ) );
		if ( culled ) add( 'below-the-last-coverage-nothing-is-drawn' + tag, 'a', dFar.length === 0, 'drawn: ' + names( dFar ), true );
		if ( lod === model.lods[ 0 ] ) obs.drawn_far = dFar.map( ( i ) => lod.levels[ i ] );

		const o = lodObject( root, lod );
		add( 'is-a-THREE.LOD' + tag, 'a', !! o, o ? `${o.type} with ${o.levels.length} levels at distances ${JSON.stringify( o.levels.map( ( l ) => + l.distance.toPrecision( 5 ) ) )}, autoUpdate ${o.autoUpdate}` : 'no object with .isLOD above the level' );

		// a walk from near to far: one level at a time, never back to a finer one, every level once
		const seen = [], steps = 24;
		let single = true, ordered = true, prev = - 1;
		for ( let s = 0; s < steps; s ++ ) {

			look( lod.world_center, lod.world_radius * 1.05 * Math.pow( 600, s / ( steps - 1 ) ) );
			const d = await settle( state, 120 );
			tint( root, model );
			if ( d.length > 1 || ( d.length === 0 && ! culled ) ) single = false;
			if ( d.length === 1 ) { if ( d[ 0 ] < prev ) ordered = false; prev = d[ 0 ]; if ( ! seen.includes( d[ 0 ] ) ) seen.push( d[ 0 ] ); }

		}

		add( 'walk-one-level-at-a-time' + tag, 'a', single, single ? '' : 'at some distance two levels (or none) were drawn' );
		add( 'walk-levels-in-order' + tag, 'a', ordered && seen.length === n, 'levels seen from near to far: ' + names( seen ) );

		// the distances of the reference mapping (README.md): a choice, hence optional
		if ( lod.reference_distances ) {

			let ok = true; const notes = [];
			const ref = lod.reference_distances;
			for ( let i = 0; i < ref.length; i ++ ) {

				for ( const [ f, want ] of [ [ 0.9, i ], [ 1.1, i + 1 ] ] ) {

					const dist = ref[ i ] * f;
					if ( ( f < 1 && i > 0 && ref[ i - 1 ] >= dist ) || ( f > 1 && ( ( i + 1 < ref.length && ref[ i + 1 ] <= dist ) || ( culled && lod.reference_cull <= dist ) ) ) ) continue;
					look( lod.world_center, dist );
					const d = await settle( state, 200 );
					if ( ! ( d.length === 1 && d[ 0 ] === want ) ) { ok = false; notes.push( `at ${dist.toFixed( 2 )}: ${names( d )}, expected ${lod.levels[ want ]}` ); }

				}

			}

			add( 'reference-distances' + tag, 'a', ok, ok ? 'switches at ' + JSON.stringify( ref ) : notes.join( '; ' ), true );

		}

		// level 0 is the node the file describes, untouched
		if ( lod.kind === 'node' ) {

			close( lod ); await settle( state );
			const node = named( root, lod.level0.name )[ 0 ];
			if ( ! node ) { add( 'level-0-node-intact' + tag, 'a', false, 'no object named ' + lod.level0.name ); continue; }
			root.updateMatrixWorld( true );
			const problems = [];
			if ( lod.level0.tag !== null && node.userData.tag !== lod.level0.tag ) problems.push( 'userData.tag is ' + node.userData.tag );
			if ( ! near( node.position.toArray(), lod.level0.translation ) ) problems.push( 'position ' + node.position.toArray() );
			if ( ! near( node.quaternion.toArray(), lod.level0.rotation ) ) problems.push( 'quaternion ' + node.quaternion.toArray() );
			if ( ! near( node.scale.toArray(), lod.level0.scale ) ) problems.push( 'scale ' + node.scale.toArray() );
			const wp = node.getWorldPosition( new THREE.Vector3() ).toArray();
			if ( ! near( wp, lod.level0.world_position ) ) problems.push( 'world position ' + wp.map( ( x ) => + x.toFixed( 4 ) ) + ', expected ' + lod.level0.world_position );
			for ( const c of lod.level0.children ) if ( ! node.children.some( ( k ) => k.name === sanitize( c ) ) ) problems.push( 'child ' + c + ' missing' );
			const a = gltf.parser.associations.get( node );
			if ( ! a || a.nodes !== lod.node ) problems.push( 'parser.associations: ' + JSON.stringify( a ) );
			let above = o ? o.parent : node.parent;
			while ( above && above !== root && ! above.name ) above = above.parent;
			const parentName = above && above !== root ? above.name : null;
			if ( parentName !== ( lod.parent === null ? null : sanitize( lod.parent ) ) ) problems.push( 'parent is ' + parentName + ', expected ' + lod.parent );
			add( 'level-0-node-intact' + tag, 'a', problems.length === 0, problems.join( '; ' ) || 'name, extras, transform, world position, children, associations, parent' );

		}

	}

	if ( model.tree ) {

		const dump = ( o ) => [ o.type, o.name, o.children.map( dump ) ];
		const got = JSON.stringify( root.children.map( dump ) ), want = JSON.stringify( model.tree );
		add( 'same-scene-as-the-stock-loader', 'a', got === want, got === want ? got : 'got ' + got + ', expected ' + want );

	}

	if ( model.animation ) {

		const clip = gltf.animations[ 0 ], an = model.animation;
		let detail = 'no animation in the result', ok = false;
		if ( clip ) {

			const mixer = new THREE.AnimationMixer( root );
			mixer.clipAction( clip ).play();
			mixer.setTime( an.at );
			root.updateMatrixWorld( true );
			const node = named( root, an.target )[ 0 ];
			const wp = node ? node.getWorldPosition( new THREE.Vector3() ).toArray() : null;
			ok = !! wp && near( wp, an.world_position );
			detail = `track ${clip.tracks.map( ( t ) => t.name ).join( ', ' )}; ${an.target} at t = ${an.at}: ${wp ? wp.map( ( x ) => + x.toFixed( 3 ) ) : 'not found'}, expected ${an.world_position}`;
			mixer.stopAllAction();

		}

		add( 'animation-moves-the-level-0-node', 'a', ok, detail );

	}

	// ---- tier B, at the end: nothing was fetched twice
	if ( lazy && model.lazy ) {

		const r = requests(), twice = Object.keys( r ).filter( ( k ) => r[ k ] > 1 ), missing = model.lazy.all_buffers.filter( ( b ) => ! r[ b ] );
		add( 'every-buffer-fetched-once', 'b', twice.length === 0 && missing.length === 0, twice.length ? 'more than once: ' + twice.map( ( k ) => `${k} x${r[ k ]}` ).join( ', ' ) : missing.length ? 'never fetched (after a walk through all levels): ' + missing.join( ', ' ) : `${Object.keys( r ).length} files, each once` );

	}

	if ( ranged && setup && setup.loadRanged ) {

		const total = net.log.reduce( ( s, e ) => s + e.bytes, 0 );
		add( 'total-bytes-not-above-the-file', 'b', total <= model.ranged.total_bytes_max && net.log.every( ( e ) => e.status === 206 ), `${total} bytes in ${net.log.length} range requests (the file: ${model.glb.bytes})` );
		obs.requests = net.log.map( ( e ) => `${e.range} ${e.status}` );

		// the same file from a server that ignores Range: one answer with everything, used for everything
		net.log.length = 0; net.strip = () => true;
		try {

			const g2 = await load( true );
			show( g2 );
			const first = nodeLods[ 0 ];
			far(); await settle( () => drawn( g2.scene, first ) );
			close( first );
			const ok = await until( () => { const d = drawn( g2.scene, first ); return d.length === 1 && d[ 0 ] === 0; } );
			far(); await settle( () => drawn( g2.scene, first ) );
			close( first ); await settle( () => drawn( g2.scene, first ) );
			add( 'no-range-support-one-full-fetch', 'b', ok && net.log.length === 1 && net.log[ 0 ].status === 200, `${net.log.length} request(s): ${net.log.map( ( e ) => e.status + ' ' + e.bytes + ' bytes' ).join( ', ' ).slice( 0, 200 )}; level 0 ${ok ? 'drawn' : 'NOT drawn'}` );

		} catch ( e ) { add( 'no-range-support-one-full-fetch', 'b', false, e && e.message ? e.message : e ); }

		net.strip = () => false;

	} else obs.requests = requests();

	return { results, obs };

}

// ------------------------------------------------------------------------------------------------ main

const out = document.getElementById( 'out' ), summary = document.getElementById( 'summary' );
const el = ( tag, cls, text ) => { const e = document.createElement( tag ); if ( cls ) e.className = cls; if ( text !== undefined ) e.textContent = text; return e; };

async function main() {

	const expected = await ( await realFetch( abs( 'expected.json' ), { cache: 'no-store' } ) ).json();
	const { GLTFLoader } = await import( LOADER );
	const setup = SETUP ? await import( SETUP ) : null;
	let meshopt = null;
	if ( expected.models.some( ( m ) => m.needs ) ) {

		try { meshopt = ( await import( 'three/addons/libs/meshopt_decoder.module.js' ) ).MeshoptDecoder; await meshopt.ready; if ( meshopt.useWorkers ) meshopt.useWorkers( 2 ); } catch ( e ) { console.warn( 'no meshopt decoder', e ); }

	}

	const all = [], observed = {};
	for ( const model of expected.models ) {

		if ( ONLY && ! model.file.includes( ONLY ) ) continue;
		summary.textContent = 'checking ' + model.file + ' ...';
		let r;
		try { r = await checkModel( model, GLTFLoader, setup, meshopt ); } catch ( e ) { console.error( e ); r = { results: [ { model: model.file, id: 'check-ran', tier: 'a', pass: false, detail: String( e && e.stack ? e.stack : e ), optional: !! model.optional } ], obs: {} }; }
		const shownRows = r.results.filter( ( x ) => TIER === 'b' || x.tier === 'a' );
		all.push( ...shownRows );
		observed[ model.file ] = r.obs;
		out.appendChild( el( 'div', 'model', model.file + ( model.optional ? '   (optional: not counted)' : '' ) ) );
		out.appendChild( el( 'div', 'about', model.about ) );
		const table = el( 'table' );
		for ( const x of shownRows ) {

			const tr = el( 'tr' );
			tr.appendChild( el( 'td', '', 'tier ' + x.tier.toUpperCase() ) );
			tr.appendChild( el( 'td', x.pass ? 'pass' : x.optional ? 'skip' : 'fail', ( x.pass ? 'pass' : 'FAIL' ) + ( x.optional ? ' (optional)' : '' ) ) );
			tr.appendChild( el( 'td', '', x.id ) );
			tr.appendChild( el( 'td', 'about', x.detail ) );
			table.appendChild( tr );

		}

		out.appendChild( table );
		out.appendChild( el( 'div', 'obs', 'observed: ' + JSON.stringify( r.obs ) ) );

	}

	const count = ( tier ) => { const rows = all.filter( ( x ) => x.tier === tier && ! x.optional ); return { pass: rows.filter( ( x ) => x.pass ).length, of: rows.length, failed: rows.filter( ( x ) => ! x.pass ).map( ( x ) => x.model + ': ' + x.id ) }; };
	const opt = all.filter( ( x ) => x.optional );
	const A = count( 'a' ), B = TIER === 'b' ? count( 'b' ) : null;
	const verdict = ( c ) => ( c === null ? 'not run' : c.of > 0 && c.pass === c.of ? 'PASS' : 'FAIL' );
	const result = { tier: TIER, loader: LOADER, setup: SETUP, three: THREE.REVISION, renderer: renderer ? 'WebGLRenderer' : 'none (LOD.update called by hand)', tierA: { verdict: verdict( A ), ...A }, tierB: B ? { verdict: verdict( B ), ...B } : { verdict: 'not run' }, optional: { pass: opt.filter( ( x ) => x.pass ).length, of: opt.length, failed: opt.filter( ( x ) => ! x.pass ).map( ( x ) => x.model + ': ' + x.id ) }, observed, rows: all };
	window.__result = result;
	document.title = `MSFT_lod check: A ${result.tierA.verdict}, B ${result.tierB.verdict}`;
	summary.innerHTML = '';
	summary.appendChild( el( 'div', '', `three r${THREE.REVISION}, loader ${LOADER}${SETUP ? ', adapter ' + SETUP : ' (no adapter)'}, run as tier ${TIER.toUpperCase()}` ) );
	summary.appendChild( el( 'div', A.pass === A.of ? 'pass' : 'fail', `tier A: ${result.tierA.verdict}  ${A.pass} / ${A.of}` ) );
	summary.appendChild( el( 'div', B === null ? 'skip' : B.pass === B.of && B.of ? 'pass' : 'fail', B === null ? 'tier B: not run (?tier=b)' : `tier B: ${result.tierB.verdict}  ${B.pass} / ${B.of}` ) );
	summary.appendChild( el( 'div', 'skip', `optional: ${result.optional.pass} / ${result.optional.of}` ) );

}

main().catch( ( e ) => { console.error( e ); summary.textContent = 'check.js failed: ' + ( e && e.stack ? e.stack : e ); window.__result = { error: String( e ) }; document.title = 'MSFT_lod check: ERROR'; } );
