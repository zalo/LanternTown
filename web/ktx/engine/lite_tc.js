// WebP on the wire, block-compressed on the client (experiment; ?tc=1).
//
// The layers are the same WebP files lite.js loads - but each is encoded in a worker to the block format this GPU
// takes, every mip level of it, and the arrays go to the GPU block-compressed: a quarter of RGBA8, as the KTX2 sets
// are (lite_ktx.js), without their download. The main thread copies the blocks into the array and uploads it once.
//
//   the format    BC7 where the GPU offers it (bc7m6.js: desktops), else ASTC 4 x 4 (astc4.js: phones, tablets, Apple),
//                 else nothing: the WebP path of lite.js, unchanged. Both are 8 bits a pixel. WebGPU: the features
//                 texture-compression-bc / -astc; WebGL2: EXT_texture_compression_bptc / WEBGL_compressed_texture_astc
//                 (three's hasFeature() answers for both). No ETC2: a WebGPU device has BC or ASTC by the standard,
//                 and a WebGL2 device with ETC2 alone is a phone of 2013 - it keeps the path it has.
//   the arrays    the sharp colour layers and the normal + roughness layers (medium, high), as the files arrive;
//                 and the first array (every tier: all a phone or `low` has). That one is shown as it is today and
//                 swapped for its compressed self once the cells near the player are in - the first frame waits for
//                 nothing new (on `high` the sharp array takes its place anyway: nothing to do).
//
//   ?tc=1            on (no format, no workers, or any failure on the way: the WebP path)
//   ?tcforce=        astc | bc: only that format; none: as on a device with neither (testing)
//   ?tcfirst=        swap (default) | show: the first array compressed before the first frame | 0: left as it is
//   ?tcgate=0        the sharp / normal files are asked for at once, not behind the cells near the player
//   ?tcworkers=n     1..4 (default: by the cores of the device, two at most - three on `high`)
//   ?tcrefine=0      no least-squares pass (a third less encoding time, a little less quality)
//   ?tchold=1        the workers too wait for the cells near the player (measured: buys nothing on four cores)
//   ?tcslow=n        every layer encoded n times (measuring: a core n times slower)
//
// Each function answers false / null when lite.js should go on with its own path.

import * as THREE from 'three';
import { encodeBC7 } from './bc7m6.js';
import { solidASTC } from './astc4.js';

const Q = new URLSearchParams( location.search );
const now = () => performance.now();
const HELD = Q.get( 'tchold' ) === '1';
const GATED = Q.get( 'tcgate' ) !== '0';
const FIRST = [ '0', 'show', 'swap' ].includes( Q.get( 'tcfirst' ) ) ? Q.get( 'tcfirst' ) : 'swap';
const FORCE = Q.get( 'tcforce' );
const SLOW = Math.max( 1, Math.min( 16, parseInt( Q.get( 'tcslow' ) ) || 1 ) );
const HOLD = 40; // m: the cells this near are in before a sharp layer is asked for
const HOLD_MAX = 90000; // ms: ... or this long has passed
const TRICKLE = 2; // sharp layers at a time while levels are still coming in
const REFINE = Q.get( 'tcrefine' ) === '0' ? 0 : 1;

// the block format of this device, and why
function pick( town ) {

	const r = town.eng.renderer, has = ( f ) => { try { return r.hasFeature( 'texture-compression-' + f ); } catch ( e ) { return false; } };
	const bc = has( 'bc' ), astc = has( 'astc' );
	if ( FORCE === 'none' ) return { why: 'switched off (tcforce=none): uncompressed' };
	if ( bc && FORCE !== 'astc' ) return { name: 'BC7', format: THREE.RGBA_BPTC_Format, astc: false, why: 'this GPU offers BC7' + ( FORCE === 'bc' ? ' (tcforce=bc)' : '' ) };
	if ( astc && FORCE !== 'bc' ) return { name: 'ASTC 4x4', format: THREE.RGBA_ASTC_4x4_Format, astc: true, why: bc ? 'tcforce=astc (this GPU offers BC7 too)' : 'this GPU has no BC7 and offers ASTC' };
	return { why: ( FORCE === 'astc' || FORCE === 'bc' ? `tcforce=${FORCE}, and this GPU does not offer it` : `this GPU offers neither BC7 nor ASTC${has( 'etc2' ) ? ' (ETC2 alone)' : ''}` ) + ': uncompressed' };

}

function open( town ) {

	if ( town.tc !== undefined ) return town.tc;
	town.tc = null;
	if ( Q.get( 'tc' ) !== '1' ) return null;
	const cores = navigator.hardwareConcurrency || 0;
	const stats = town.tcStats = { on: false, why: '', format: '', layers: 0, bytes: 0, held: 0, ms: { fetch: 0, queue: 0, decode: 0, mips: 0, encode: 0, copy: 0 }, sets: {},
		workers: 0, cores, refine: REFINE, first: FIRST, gate: '', hold: 0, gpuError: '' };
	try {

		const F = pick( town );
		stats.why = F.why;
		if ( ! F.format ) return null;
		if ( typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined' ) { stats.why = 'no workers with a canvas in this browser: uncompressed'; return null; }
		// (the workers share the cores with the page and with the workers that unpack the cells: half the cores less one,
		// and few - each holds a layer's pixels while it works)
		let n = Math.min( town.T.sharp ? 3 : 2, Math.max( 1, ( ( cores || 4 ) >> 1 ) - 1 ) );
		if ( navigator.deviceMemory && navigator.deviceMemory <= 2 ) n = 1;
		n = Math.max( 1, Math.min( 4, parseInt( Q.get( 'tcworkers' ) ) || n ) );
		// (started with the first job, gone when no array is being made: a phone has them for a few seconds)
		const idle = [], waiting = [], jobs = new Map();
		let workers = null, id = 0, users = 0;
		const next = () => { while ( idle.length && waiting.length ) { const w = idle.pop(), j = waiting.shift(); jobs.set( j.msg.id, { ...j, w } ); w.postMessage( j.msg, j.transfer ); } };
		const start = () => ( workers = Array.from( { length: n }, () => {

			const w = new Worker( new URL( './lite_tc_worker.js', import.meta.url ), { type: 'module' } );
			w.onmessage = ( e ) => { const j = jobs.get( e.data.id ); if ( ! j ) return; jobs.delete( e.data.id ); idle.push( w ); next(); if ( e.data.error ) j.no( new Error( e.data.error ) ); else j.ok( e.data ); };
			w.onerror = ( e ) => { for ( const [ k, j ] of jobs ) if ( j.w === w ) { jobs.delete( k ); j.no( new Error( 'worker: ' + ( e.message || 'failed' ) ) ); } };
			idle.push( w );
			return w;

		} ) );
		const pool = {
			run: ( msg, transfer = [] ) => new Promise( ( ok, no ) => { if ( ! workers ) start(); waiting.push( { msg: { ...msg, id: ++ id, refine: REFINE, astc: F.astc, slow: SLOW }, transfer, ok, no } ); next(); } ),
			hold: () => { users ++; },
			release: () => { if ( -- users <= 0 && workers ) { for ( const w of workers ) w.terminate(); workers = null; idle.length = waiting.length = 0; jobs.clear(); } },
		};
		// (what the GPU says to an upload - on a phone the readout of ?ktxinfo=1 is all there is to see)
		const b = r_backend( town );
		if ( b && b.device && b.device.addEventListener ) b.device.addEventListener( 'uncapturederror', ( e ) => { if ( ! stats.gpuError ) stats.gpuError = String( e.error && e.error.message || e ).slice( 0, 200 ); } );
		stats.on = true; stats.format = F.name; stats.workers = n;
		town.tc = { stats, F, pool, near: null, gate: null, failed: false };

	} catch ( e ) {

		stats.why = String( e.message || e );

	}

	return town.tc;

}

const r_backend = ( town ) => town.eng.renderer.backend || null;

// cells within HOLD metres that are to show a level they have not got (-1: the town has not looked yet)
function cellsNear( town ) {

	if ( ! town._at ) return - 1;
	let n = 0;
	for ( const c of town.chunks ) if ( c.d < HOLD && c.want > 0 && ! c.mesh[ c.want ] ) n ++;
	return n;

}

// resolves when the cells within HOLD metres show the level they are to show (at once: a review capture)
function near( town ) {

	const K = town.tc;
	if ( town.wait === 'all' ) return Promise.resolve();
	return K.near || ( K.near = new Promise( ( ok ) => {

		const t0 = now();
		const look = () => {

			K.stats.held = Math.round( now() - t0 );
			if ( cellsNear( town ) === 0 || now() - t0 > HOLD_MAX ) ok(); else setTimeout( look, 100 );

		};

		look();

	} ) );

}

// how many files of the sharp / normal sets may be on their way: none until the cells near the player are in, TRICKLE
// while any level is still wanted, then what the tier fetches at a time (as lite_ktx.js does for its files: the cells'
// levels are what the player sees coming in - on a slow line the layers must not take their place in the queue)
function gate( town ) {

	const K = town.tc, S = K.stats;
	if ( K.gate ) return K.gate;
	const full = Math.max( 2, town.T.fetch || 3 ), waiting = [], t0 = now(), free = ! GATED || town.wait === 'all';
	let busy = 0, open = free, timer = 0;
	S.gate = open ? 'open' : 'hold'; S.hold = 0;
	const room = () => {

		if ( free ) return full;
		if ( ! open ) {

			if ( cellsNear( town ) !== 0 && now() - t0 < HOLD_MAX ) { S.hold = Math.round( now() - t0 ); return 0; }
			open = true; S.hold = Math.round( now() - t0 ); S.opened = Math.round( now() );

		}

		const coming = town.pending > 0 || town.baseLeft > 0;
		S.gate = coming ? 'trickle' : 'open';
		return coming ? Math.min( TRICKLE, full ) : full;

	};

	const pump = () => {

		while ( waiting.length && busy < room() ) { busy ++; waiting.shift()(); }
		if ( ! waiting.length && timer ) { clearInterval( timer ); timer = 0; }

	};

	return ( K.gate = {
		enter: () => new Promise( ( r ) => { waiting.push( r ); if ( ! timer ) timer = setInterval( pump, 100 ); pump(); } ),
		leave: () => { busy --; pump(); },
	} );

}

// the levels of an array of n layers, empty
function levels( size, n ) {

	return Array.from( { length: Math.log2( size ) + 1 }, ( _, l ) => { const w = Math.max( 1, size >> l ), b = Math.ceil( w / 4 ); return { data: new Uint8Array( b * b * 16 * n ), width: w, height: w, each: b * b * 16 }; } );

}

function tally( S, t, wall = 0 ) {

	S.ms.decode += t.decode; S.ms.mips += t.mips; S.ms.encode += t.encode;
	if ( wall ) S.ms.queue += Math.max( 0, wall - t.decode - t.mips - t.encode );

}

function make( town, mips, size, n, srgb, set ) {

	const K = town.tc, S = K.stats;
	const tex = new THREE.CompressedArrayTexture( mips.map( ( m ) => ( { data: m.data, width: m.width, height: m.height } ) ), size, size, n, K.F.format, THREE.UnsignedByteType );
	tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = false;
	tex.anisotropy = town.T.aniso;
	set.format = tex.format; set.ready = Math.round( now() ); set.mb = mips.reduce( ( a, m ) => a + m.data.byteLength, 0 ) / 1e6;
	tex.onUpdate = () => {

		for ( const m of tex.mipmaps ) m.data = null; // on the GPU: the copy here has done its work
		tex.onUpdate = null; set.uploaded = Math.round( now() );
		const b = r_backend( town ), gl = b && b.gl;
		if ( gl && gl.getError ) { const e = gl.getError(); if ( e && ! S.gpuError ) S.gpuError = 'WebGL error 0x' + e.toString( 16 ) + ' at the upload of a ' + K.F.name + ' array'; }

	};

	tex.needsUpdate = true;
	return tex;

}

// An array that another has taken the place of. dispose() alone leaves it on the GPU (see retire() in lite_ktx.js: the
// bindings of a mesh that arrives later still name it, and three creates it again) - so what is left is one texel / block
function retire( old ) {

	old.dispose();
	if ( old.isCompressedTexture ) { old.mipmaps = [ { data: new Uint8Array( 16 ), width: 4, height: 4 } ]; old.image = { width: 4, height: 4, depth: 1 }; }
	else { old.image = { data: new Uint8Array( [ 0, 0, 0, 255 ] ), width: 1, height: 1, depth: 1 }; old.generateMipmaps = false; old.minFilter = THREE.LinearFilter; }
	old.onUpdate = null;

}

// files (null: a blank layer of `fill`) -> one CompressedArrayTexture with every mip level
function array( town, size, files, srgb, set, fill = [ 0, 0, 0, 255 ] ) {

	const K = town.tc, S = K.stats, bake = town.bake, n = files.length;
	const mips = levels( size, n );
	Object.assign( set, { size, n, got: 0, t0: Math.round( now() ) } );
	const slots = Array.from( { length: Math.max( 2, town.T.fetch || 3 ) }, () => Promise.resolve() ), G = gate( town );
	let turn = 0;
	return Promise.all( files.map( ( file, k ) => {

		if ( ! file ) {

			const block = K.F.astc ? solidASTC( ...fill ) : encodeBC7( new Uint8Array( 64 ).map( ( _, i ) => fill[ i % 4 ] ), 4, 4, new Uint8Array( 16 ) );
			for ( const m of mips ) for ( let o = k * m.each; o < ( k + 1 ) * m.each; o += 16 ) m.data.set( block, o );
			set.got ++;
			return null;

		}

		// (the files as lite.js asks for them; the encoding in its own queue, so that a file does not wait for a worker)
		const i = turn ++ % slots.length;
		const got = ( slots[ i ] = slots[ i ].then( async () => {

			await G.enter();
			try {

				const t = now();
				const r = await fetch( `${bake.dir}/${file}.webp?v=${bake.baked}`, GATED && town.wait !== 'all' ? { priority: 'low' } : {} );
				if ( ! r.ok ) throw new Error( `${file}.webp: ${r.status}` );
				const blob = await r.blob();
				town.bytes += blob.size; S.bytes += blob.size; S.ms.fetch += now() - t;
				return blob;

			} finally { G.leave(); }

		} ) );
		return got.then( async ( blob ) => {

			if ( HELD ) await near( town );
			const t = now();
			const res = await K.pool.run( { blob, size, srgb } );
			const t1 = now();
			if ( res.mips.length !== mips.length ) throw new Error( `${file}: ${res.mips.length} levels` );
			res.mips.forEach( ( m, l ) => mips[ l ].data.set( m, k * mips[ l ].each ) );
			S.ms.copy += now() - t1; tally( S, res.times, t1 - t );
			S.layers ++; set.got ++;

		} );

	} ) ).then( () => make( town, mips, size, n, srgb, set ) );

}

// _loadLayers() with ?tcfirst=show: the pages of the first array, decoded and encoded in the workers - the first frame
// waits for them. -> the array, or null
export async function tcLayers( town ) {

	const K = FIRST === 'show' ? open( town ) : null;
	if ( ! K ) return null;
	const bake = town.bake, t = bake.tex[ town.T.tex ], size = t.size, n = bake.layers.length, per = t.cols * t.rows, S = K.stats;
	const set = S.sets.first = { mode: 'show', size, n, got: 0, t0: Math.round( now() ) };
	K.pool.hold();
	try {

		const mips = levels( size, n );
		const blob = async ( name ) => {

			const r = await fetch( `${bake.dir}/${name}.webp?v=${bake.baked}` );
			if ( ! r.ok ) throw new Error( `${name}.webp: ${r.status}` );
			const b = await r.blob();
			town.bytes += b.size; S.bytes += b.size;
			return b;

		};

		await Promise.all( t.pages.map( async ( name, p ) => {

			const [ page, hpage ] = await Promise.all( [ blob( name ), t.hpages ? blob( t.hpages[ p ] ) : null ] );
			const count = Math.min( n, ( p + 1 ) * per ) - p * per, t0 = now();
			const res = await K.pool.run( { page, hpage, size, cols: t.cols, count, srgb: true } );
			const t1 = now();
			res.layers.forEach( ( lm, k ) => lm.forEach( ( m, l ) => mips[ l ].data.set( m, ( p * per + k ) * mips[ l ].each ) ) );
			S.ms.copy += now() - t1; tally( S, res.times, t1 - t0 );
			S.layers += count; set.got += count;

		} ) );
		const tex = make( town, mips, size, n, true, set );
		set.at = Math.round( now() );
		K.pool.release();
		return tex;

	} catch ( e ) {

		town.warnings.push( 'baked town: client-compressed first layers: ' + ( e.message || e ) + ' (WebP as it is instead)' );
		set.failed = String( e.message || e );
		K.pool.release();
		return null;

	}

}

// After the sharp and normal layers (lite.js load()): the first array - shown uncompressed, as ever - is encoded from
// the pixels it holds and swapped for the result, once the cells near the player are in. Where a `big` array covers
// some of its layers (medium) those are left out, as the sharp array of `high` leaves them out. -> true when swapped
export async function tcFirst( town ) {

	const K = open( town );
	if ( ! K ) return false;
	const S = K.stats, old = town.layers, TX = town.TX, nodes = TX ? TX.a : town.TX0;
	if ( FIRST !== 'swap' || K.failed || ! old || ! old.isDataArrayTexture || ! old.image.data || ! nodes || ! nodes.length ) return false;
	if ( TX && TX.u.shift.value !== 0 ) return false; // (high: the sharp array has taken its place)
	if ( town.T.sharp ) return false;
	const size = old.image.width, N = old.image.depth, each = size * size * 4;
	const set = S.sets.first = { mode: 'swap', size, n: N, got: 0, t0: Math.round( now() ), state: 'wait' };
	K.pool.hold();
	try {

		await near( town );
		set.state = 'encode'; set.t1 = Math.round( now() );
		const nb = TX && town.layersBig ? TX.u.b1.value - TX.u.b0.value : 0;
		const keep = Array.from( { length: N }, ( _, i ) => i ).filter( ( i ) => nb === 0 || i < TX.u.b0.value || i >= TX.u.b1.value );
		const mips = levels( size, keep.length );
		set.n = keep.length; set.of = N;
		// (a layer's pixels are copied for a worker only when one is free: a megabyte or four in flight, not the array twice)
		let at = 0;
		const lane = async () => {

			while ( at < keep.length ) {

				const k = at ++, px = old.image.data.slice( keep[ k ] * each, ( keep[ k ] + 1 ) * each ), t = now();
				const res = await K.pool.run( { px, size, srgb: true }, [ px.buffer ] );
				const t1 = now();
				res.mips.forEach( ( m, l ) => mips[ l ].data.set( m, k * mips[ l ].each ) );
				S.ms.copy += now() - t1; tally( S, res.times, t1 - t );
				S.layers ++; set.got ++;

			}

		};

		await Promise.all( Array.from( { length: S.workers }, lane ) );
		if ( town.layers !== old ) throw new Error( 'the first array changed meanwhile' );
		const tex = make( town, mips, size, keep.length, true, set );
		for ( const n of nodes ) n.value = tex;
		if ( nb ) TX.u.shift.value = nb;
		town.layers = tex;
		set.freed = ( each * N * 4 / 3 + each * N ) / 1e6; // (the RGBA8 array with its mips on the GPU, and the pixels kept here)
		retire( old );
		set.state = 'done'; set.at = Math.round( now() );
		K.pool.release();
		return true;

	} catch ( e ) {

		town.warnings.push( 'baked town: client-compressed first layers: ' + ( e.message || e ) + ' (left as they are)' );
		set.state = 'failed'; set.failed = String( e.message || e );
		K.pool.release();
		return false;

	}

}

// _sharpen(): as lite.js swaps its arrays in, each whole. -> true when done
export async function tcSharpen( town, nb ) {

	const K = open( town );
	if ( ! K ) return false;
	const T = town.T, bake = town.bake, N = bake.layers.length, TX = town.TX, S = K.stats.sets;
	K.pool.hold();
	try {

		const all = Array.from( { length: N }, ( _, i ) => i );
		const names = ( size, layers ) => { const t = bake.texl[ size ]; return t && layers.length && layers.every( ( i ) => t.files[ i ] ) ? layers.map( ( i ) => t.files[ i ] ) : null; };
		const bn = nb > 0 ? names( T.big[ 0 ], all.slice( 2, 2 + nb ) ) : null;
		const rn = T.sharp ? names( T.sharp, bn ? all.filter( ( i ) => i < 2 || i >= 2 + nb ) : all ) : null;
		if ( ( nb > 0 && ! bn ) || ( T.sharp && ! rn ) ) throw new Error( 'layers missing' );
		const big = bn ? array( town, T.big[ 0 ], bn, true, S.big = {} ) : null;
		const rest = rn ? array( town, T.sharp, rn, true, S.rest = {} ) : null;
		const jobs = [];
		if ( big ) jobs.push( big.then( ( tex ) => {

			for ( const n of TX.b ) n.value = tex;
			TX.u.b0.value = 2; TX.u.b1.value = 2 + nb;
			town.layersBig = tex;
			S.big.at = Math.round( now() );

		} ) );
		if ( rest ) jobs.push( Promise.all( [ rest, big ] ).then( ( [ tex ] ) => {

			const old = town.layers;
			for ( const n of TX.a ) n.value = tex;
			TX.u.shift.value = big ? nb : 0;
			town.layers = tex;
			retire( old );
			S.rest.at = Math.round( now() );

		} ) );
		await Promise.all( jobs );
		K.pool.release();
		return true;

	} catch ( e ) {

		town.warnings.push( 'baked town: client-compressed sharp layers: ' + ( e.message || e ) + ' (WebP as it is instead)' );
		K.stats.why = String( e.message || e );
		K.failed = true;
		K.pool.release();
		return false;

	}

}

// _loadNR(): the normal + roughness array. -> true when done
export async function tcNR( town ) {

	const NR = town.NR, K = NR ? open( town ) : null;
	if ( ! K || K.failed ) return false;
	K.pool.hold();
	try {

		const set = K.stats.sets.nr = {};
		const tex = await array( town, NR.size, NR.files, false, set, [ 128, 128, 153, 255 ] );
		for ( const n of NR.nodes ) n.value = tex;
		NR.on.value = 1;
		NR.tex = tex;
		set.at = Math.round( now() );
		K.pool.release();
		return true;

	} catch ( e ) {

		town.warnings.push( 'baked town: client-compressed normal layers: ' + ( e.message || e ) + ' (WebP as it is instead)' );
		K.failed = true;
		K.pool.release();
		return false;

	}

}

// the lines of ?ktxinfo=1 (lite_ktx.js): the format chosen and why, and where each array stands
export function tcText( town, sec ) {

	const C = town.tcStats;
	if ( ! C ) return [];
	if ( ! C.on ) return [ `client compression OFF - ${C.why || 'failed'}` ];
	const T = town.T, S = C.sets, lines = [];
	lines.push( `client compression: ${C.format} - ${C.why} · ${C.workers} worker${C.workers > 1 ? 's' : ''} (${C.cores || '?'} cores${navigator.deviceMemory ? ', ' + navigator.deviceMemory + ' GB' : ''})` +
		( C.refine ? '' : ' · tcrefine=0' ) + ( SLOW > 1 ? ` · tcslow=${SLOW}` : '' ) );
	const up = ( v ) => ( v.uploaded ? `, on the GPU ${sec( v.uploaded )} (${v.mb.toFixed( 1 )} MB)` : v.ready ? ', upload to come' : '' );
	const f = S.first;
	let first;
	if ( f && f.mode === 'show' ) first = f.failed ? `first ${f.size}: FAILED (${f.failed}) - WebP as it is` : f.at ? `first ${f.size}: ${C.format} before the first frame, ${sec( f.at )}${up( f )}` : `first ${f.size}: encoding ${f.got}/${f.n}`;
	else if ( f ) first = `first ${f.size}: shown uncompressed, ` + ( f.state === 'wait' ? `waits for the cells within ${HOLD} m to swap` : f.state === 'encode' ? `encoding ${f.got}/${f.n}` :
		f.state === 'done' ? `${C.format} swapped in at ${sec( f.at )} (${f.n}${f.of !== f.n ? ' of ' + f.of : ''} layers, encoding began ${sec( f.t1 )})${up( f )}, ${Math.round( f.freed )} MB let go` : `swap FAILED (${f.failed})` );
	else first = C.first === '0' ? `first ${T.tex}: left uncompressed (tcfirst=0)` : T.sharp ? `first ${T.tex}: uncompressed until the sharp array takes its place` : `first ${T.tex}: shown uncompressed, swap to come`;
	lines.push( first );
	const sets = [ 'big', 'rest', 'nr' ].filter( ( k ) => S[ k ] );
	if ( sets.length ) lines.push( sets.map( ( k ) => `${k} ${S[ k ].size} ${S[ k ].got}/${S[ k ].n} ${sec( S[ k ].at )}${S[ k ].uploaded ? ' on GPU' : ''}` ).join( ' · ' ) +
		( C.gate === 'hold' ? ` · files wait for the cells within ${HOLD} m (${( C.hold / 1000 ).toFixed( 1 )}s)` : C.hold ? ` · files waited ${( C.hold / 1000 ).toFixed( 1 )}s for the cells near` : '' ) );
	if ( C.layers ) lines.push( `${C.layers} layers: decode ${( C.ms.decode / 1000 ).toFixed( 1 )}s encode ${( ( C.ms.encode + C.ms.mips ) / 1000 ).toFixed( 1 )}s in workers (${Math.round( ( C.ms.encode + C.ms.mips ) / C.layers )} ms a layer), copy ${Math.round( C.ms.copy )} ms here` );
	if ( C.gpuError ) lines.push( '! GPU: ' + C.gpuError );
	if ( C.on && C.why && C.why !== town.tc.F.why ) lines.push( '! ' + C.why );
	return lines;

}
