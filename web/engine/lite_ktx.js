// KTX2 / Basis texture layers for the baked town (experiment; tools/ktx_layers.py -> out/web/_nextktx/).
//
// The same three texture arrays as lite.js builds from WebP files - the first frame's layers, the sharp layers
// (`big` and the rest) and the normal + roughness layers - but block-compressed on the GPU (BC7 on desktop, ASTC or
// ETC2 on Apple / Android: a quarter of RGBA8 and less), with the mip chain the bake made instead of the GPU's own,
// and decoded in workers (three's KTX2Loader: the Basis transcoder in a worker pool) instead of
// createImageBitmap -> canvas -> getImageData on the main thread.
//
//   ?ktx=1            on (anything else, a renderer without a compressed format, a folder that is not there or is of
//                     another bake, or any failure on the way: the WebP path of lite.js, unchanged)
//   ?ktxdir=_nextktx_x  another folder of out/web (comparing encodings / mip filters)
//   ?ktxhold=0        the sharp and normal sets start at once, six at a time (as it was: measuring what the order below is worth)
//   ?ktxinfo=1        the load, as text on the page (ktxInfo below)
//
// The cells go first. The KTX2 sets are 2.5 times the bytes of the WebP ones, and they were asked for before the first
// level of a cell was: on a slow line the cells around the player stayed at their far level for as long as the textures
// took their share of it. So the sharp and normal sets wait until the cells within HOLD metres have the level they are
// to show (gate() below), then come in two at a time, at low priority, while levels are still on their way, and only
// then as fast as the tier fetches.
//
// Each function answers null / false when lite.js should go on with its own path.

import * as THREE from 'three';
import { tcText } from './lite_tc.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';

const Q = new URLSearchParams( location.search );
const now = () => performance.now();
const HELD = Q.get( 'ktxhold' ) !== '0';
const HOLD = 40; // m: the cells this near are in before a sharp layer is asked for
const HOLD_MAX = 90000; // ms: ... or this long has passed
const TRICKLE = 2; // sharp layers at a time while levels are still coming in

// -> { dir, man, loader, stats } of a town, or null
async function open( town ) {

	if ( town.ktx !== undefined ) return town.ktx;
	town.ktx = null;
	if ( Q.get( 'ktx' ) !== '1' ) return null;
	const stats = town.ktxStats = { on: false, why: '', bytes: 0, files: 0, transcode_ms: 0, sets: {} };
	try {

		const r = town.eng.renderer, bake = town.bake;
		const fmt = [ 'texture-compression-astc', 'texture-compression-bc', 'texture-compression-etc2' ].filter( ( f ) => r.hasFeature( f ) );
		if ( ! fmt.length ) { stats.why = 'no compressed texture format'; return null; }
		const name = /^_nextktx\w*$/.test( Q.get( 'ktxdir' ) || '' ) ? Q.get( 'ktxdir' ) : null;
		const dir = bake.dir.replace( /\/_next(\w*)/, ( m, rest ) => '/' + ( name || '_nextktx' + rest ) );
		const res = await fetch( `${dir}/ktx.json`, { cache: 'no-cache' } );
		if ( ! res.ok ) { stats.why = `ktx.json: ${res.status}`; return null; }
		const man = await res.json();
		if ( man.version !== 1 || man.baked !== bake.baked || man.layers !== bake.layers.length ) { stats.why = 'ktx.json is of another bake'; return null; }
		const loader = new KTX2Loader();
		loader.setTranscoderPath( import.meta.resolve( 'three/addons/libs/basis/' ) );
		// (two at most: the cells' levels are unpacked in the tier's workers at the same time, on four to six cores)
		loader.setWorkerLimit( Math.max( 1, Math.min( 4, parseInt( Q.get( 'ktxworkers' ) ) || Math.min( HELD ? 2 : 4, town.T.workers || 1 ) ) ) );
		loader.detectSupport( r );
		// ?ktxforce=astc|etc2|bc|dxt|none (testing): transcode to that format whatever else the device has
		const force = Q.get( 'ktxforce' );
		if ( force ) {

			const key = { astc: 'astcSupported', etc2: 'etc2Supported', bc: 'bptcSupported', dxt: 'dxtSupported' }[ force ];
			for ( const k in loader.workerConfig ) loader.workerConfig[ k ] = k === key && loader.workerConfig[ k ];
			stats.force = force;

		}

		stats.on = true; stats.features = fmt; stats.dir = dir;
		town.ktx = { dir, man, loader, stats, left: 0 };

	} catch ( e ) {

		stats.why = String( e.message || e );

	}

	return town.ktx;

}

// one file -> CompressedTexture / CompressedArrayTexture (not yet on the GPU)
async function file( town, name, low = false ) {

	const K = town.ktx;
	const r = await fetch( `${K.dir}/${name}.ktx2?v=${K.man.built || K.man.baked}`, low ? { priority: 'low' } : {} );
	if ( ! r.ok ) throw new Error( `${name}.ktx2: ${r.status}` );
	const buf = await r.arrayBuffer();
	town.bytes += buf.byteLength; K.stats.bytes += buf.byteLength; K.stats.files ++;
	const t0 = now();
	const tex = await new Promise( ( ok, no ) => K.loader.parse( buf, ok, no ) ); // (the buffer goes to a worker)
	K.stats.transcode_ms += now() - t0; // (wall time of the tasks, queueing included)
	if ( ! tex.isCompressedTexture || tex.format === THREE.RGBAFormat ) throw new Error( `${name}.ktx2: not transcoded to a compressed format` );
	return tex;

}

function setup( town, tex, srgb ) {

	tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = false;
	tex.anisotropy = town.T.aniso;
	{ const m = tex.mipmaps[ tex.mipmaps.length - 1 ]; tex.userData.block = m.data.byteLength / ( tex.image.depth || 1 ) / ( Math.ceil( m.width / 4 ) * Math.ceil( m.height / 4 ) ); } // bytes of a 4 x 4 block
	tex.onUpdate = () => { for ( const m of tex.mipmaps ) m.data = null; tex.onUpdate = null; }; // on the GPU: the copy here has done its work
	tex.needsUpdate = true;
	town.ktx.stats.format = tex.format;
	return tex;

}

// The first array, when the sharp ones have taken its place. dispose() alone does not see it off: the bindings three
// compiled for a material name the texture of that moment, and every mesh that arrives later starts from a copy of
// them (Bindings._createBindings) before it follows the node to the new texture - so the renderer creates the
// disposed array again and uploads it. The WebP array has kept its pixels: it is uploaded a second time and stays on
// the GPU. This one gave its data away when it was first uploaded, and writeTexture( null ) threw in the middle of a
// frame (the frame was lost and the picture stayed dark: the renderer was left half way through a pass). So what is
// left of it is one 4 x 4 block: valid to create, nothing to hold.
function retire( tex ) {

	tex.dispose();
	tex.mipmaps = [ { data: new Uint8Array( tex.userData.block || 16 ), width: 4, height: 4 } ];
	tex.image = { width: 4, height: 4, depth: 1 };
	tex.onUpdate = null;

}

// cells within HOLD metres that are to show a level they have not got (-1: the town has not looked yet)
function cellsNear( town ) {

	if ( ! town._at ) return - 1;
	let n = 0;
	for ( const c of town.chunks ) if ( c.d < HOLD && c.want > 0 && ! c.mesh[ c.want ] ) n ++;
	return n;

}

// how many files of the sharp / normal sets may be on their way: none until the cells near the player are in, TRICKLE
// while any level is still wanted, then what the tier fetches at a time. enter() resolves when there is room.
function gate( town ) {

	const K = town.ktx, S = K.stats;
	if ( K.gate ) return K.gate;
	const full = Math.max( 2, town.T.fetch || 3 ), waiting = [], t0 = now();
	let busy = 0, open = ! HELD || town.wait === 'all', timer = 0; // (a review capture waits for everything anyway, and looks only then)
	S.gate = open ? 'open' : 'hold'; S.hold = 0;
	const room = () => {

		if ( ! HELD || town.wait === 'all' ) return full;
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

// single-layer files -> one CompressedArrayTexture, its layers in the order of `names`. A layer is copied into its
// place as it arrives (a memcpy of the compressed bytes); the array is uploaded once, whole.
function array( town, size, names, srgb, queue, set ) {

	let mips = null, format = 0;
	Object.assign( set, { size, n: names.length, got: 0 } );
	const G = gate( town );
	return Promise.all( names.map( ( name, k ) => queue( async () => {

		await G.enter();
		let t;
		try { t = await file( town, name, HELD ); } finally { G.leave(); }
		if ( t.image.width !== size || t.image.height !== size ) throw new Error( `${name}.ktx2: ${t.image.width} px, not ${size}` );
		if ( ! mips ) {

			format = t.format;
			mips = t.mipmaps.map( ( m ) => ( { data: new Uint8Array( m.data.byteLength * names.length ), width: m.width, height: m.height } ) );

		}

		// (ETC2 has one format with alpha and one without: the layers of an array must agree)
		if ( t.format !== format || t.mipmaps.length !== mips.length ) throw new Error( `${name}.ktx2: another format than the layers before it` );
		t.mipmaps.forEach( ( m, l ) => mips[ l ].data.set( m.data, k * m.data.byteLength ) );
		t.mipmaps.length = 0;
		set.got ++; set.format = format;

	} ) ) ).then( () => setup( town, new THREE.CompressedArrayTexture( mips, size, size, names.length, format, THREE.UnsignedByteType ), srgb ) );

}

function slots( town ) {

	const s = Array.from( { length: Math.max( 2, town.T.fetch || 3 ) }, () => Promise.resolve() );
	let turn = 0;
	return ( fn ) => { const i = turn ++ % s.length; return ( s[ i ] = s[ i ].then( fn ) ); };

}

// (the workers and the transcoder's memory go when the last array is in)
function done( town ) {

	const K = town.ktx;
	if ( K && -- K.left <= 0 && K.loader ) { K.loader.dispose(); K.loader = null; }

}

// the first frame's array: c<size>.ktx2, every layer in one file. -> the texture, or null
export async function ktxLayers( town ) {

	const K = await open( town );
	if ( ! K ) return null;
	const t0 = now();
	try {

		const c = K.man.c[ town.T.tex ];
		if ( ! c || c.layers !== town.bake.layers.length ) throw new Error( `no c${town.T.tex}` );
		// (what comes after it: the loader is kept until then)
		K.left = 1 + ( town.bake.texl && ( town.T.big || town.T.sharp ) ? 1 : 0 ) + ( town.T.dir && town.T.nr ? 1 : 0 );
		const set = K.stats.sets.first = { size: c.size, n: c.layers, got: 0 };
		const tex = setup( town, await file( town, c.file ), true );
		K.stats.first_ms = Math.round( now() - t0 );
		Object.assign( set, { got: c.layers, format: tex.format, at: Math.round( now() ) } );
		done( town );
		return tex;

	} catch ( e ) {

		town.warnings.push( 'baked town: KTX2 layers: ' + ( e.message || e ) + ' (WebP instead)' );
		K.stats.on = false; K.stats.why = String( e.message || e );
		if ( K.loader ) K.loader.dispose();
		town.ktx = null;
		return null;

	}

}

// _sharpen(): the `big` array and, with T.sharp, the rest - each swapped in whole. -> true when done
export async function ktxSharpen( town, nb ) {

	const K = town.ktx;
	if ( ! K ) return false;
	const T = town.T, N = town.bake.layers.length, TX = town.TX, t0 = now();
	try {

		const all = Array.from( { length: N }, ( _, i ) => i );
		const names = ( size, layers ) => { const t = K.man.l[ size ]; return t && layers.every( ( i ) => t.files[ i ] ) ? layers.map( ( i ) => t.files[ i ] ) : null; };
		const bn = nb > 0 ? names( T.big[ 0 ], all.slice( 2, 2 + nb ) ) : null;
		const rn = T.sharp ? names( T.sharp, bn ? all.filter( ( i ) => i < 2 || i >= 2 + nb ) : all ) : null;
		if ( ( nb > 0 && ! bn ) || ( T.sharp && ! rn ) ) throw new Error( 'layers missing' );
		const queue = slots( town );
		const S = K.stats.sets;
		const big = bn ? array( town, T.big[ 0 ], bn, true, queue, S.big = {} ) : null;
		const rest = rn ? array( town, T.sharp, rn, true, queue, S.rest = {} ) : null;
		const jobs = [];
		if ( big ) jobs.push( big.then( ( tex ) => {

			for ( const n of TX.b ) n.value = tex;
			TX.u.b0.value = 2; TX.u.b1.value = 2 + nb;
			town.layersBig = tex;
			K.stats.big_ms = Math.round( now() - t0 ); S.big.at = Math.round( now() );

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
		K.stats.sharp_ms = Math.round( now() - t0 );
		done( town );
		return true;

	} catch ( e ) {

		town.warnings.push( 'baked town: KTX2 sharp layers: ' + ( e.message || e ) + ' (WebP instead)' );
		for ( const k of [ 'big', 'rest' ] ) if ( K.stats.sets[ k ] ) K.stats.sets[ k ].webp = String( e.message || e );
		done( town );
		return false;

	}

}

// _loadNR(): the normal + roughness array. -> true when done
export async function ktxNR( town ) {

	const K = town.ktx, NR = town.NR;
	if ( ! K || ! NR ) return false;
	const t0 = now();
	try {

		const t = K.man.n[ NR.size ];
		const names = t ? t.files.slice( NR.n0, NR.n1 ) : [];
		if ( ! names.length || names.some( ( f ) => ! f ) ) throw new Error( 'layers missing' ); // (a gap would need a blank compressed layer)
		const set = K.stats.sets.nr = {};
		const tex = await array( town, NR.size, names, false, slots( town ), set );
		for ( const n of NR.nodes ) n.value = tex;
		NR.on.value = 1;
		NR.tex = tex;
		K.stats.nr_ms = Math.round( now() - t0 ); set.at = Math.round( now() );
		done( town );
		return true;

	} catch ( e ) {

		town.warnings.push( 'baked town: KTX2 normal layers: ' + ( e.message || e ) + ' (WebP instead)' );
		if ( K.stats.sets.nr ) K.stats.sets.nr.webp = String( e.message || e );
		done( town );
		return false;

	}

}

// ----------------------------------------------------------------------------------------------------------- load log
// What the page shows while it loads: town.loadLog (always, for measuring from outside) and, with ?ktxinfo=1, a few
// lines of text on the page - for a device without a console. Times: ms / s from opening the page.
//   tex: the first frame's textures are in; base: enough of the lowest level is in (the sharp textures may start);
//   settled: the page goes on to its first frame; gap: the longest the main thread was away before it [ ms, when ];
//   first: the first frame; mid / near / top: no cell that is to show that level (or a higher one) shows a lower one
//   any more; all: every cell shows its level; loClose: the last moment a cell within `close` metres showed its lowest
//   (far) level while a higher one was on its way; waitClose: the last moment one of them showed less than its level
const FORMAT = { 36492: 'BC7', 37808: 'ASTC4x4', 37496: 'ETC2-RGBA', 37492: 'ETC2-RGB', 33779: 'DXT5', 33776: 'DXT1', 33777: 'DXT1', 1023: 'RGBA8' };

export function ktxInfo( town ) {

	if ( town.loadLog || typeof window === 'undefined' ) return;
	const L = town.L, N = L.length, T = town.T, close = Math.min( 40, town.R[ 2 ] || 40 );
	const log = town.loadLog = { close, levels: L, tex: 0, base: 0, settled: 0, gap: [ 0, 0 ], first: 0, mid: 0, near: 0, top: 0, all: 0, loClose: 0, waitClose: 0, sharp: 0, detail: 0, samples: [], now: null };
	const r = town.eng.renderer;
	const feats = [ 'astc', 'bc', 'etc2' ].filter( ( f ) => { try { return r.hasFeature( 'texture-compression-' + f ); } catch ( e ) { return false; } } );
	const sec = ( ms ) => ( ms ? ( ms / 1000 ).toFixed( 1 ) + 's' : '…' );
	let hooked = false, el = null, lastSample = 0, lastText = 0, lastTick = 0;
	// (long: the main thread's tasks of 50 ms and more, [ start, ms ])
	log.long = [];
	try { new PerformanceObserver( ( l ) => { for ( const e of l.getEntries() ) if ( log.long.length < 600 ) log.long.push( [ Math.round( e.startTime ), Math.round( e.duration ) ] ); } ).observe( { type: 'longtask', buffered: true } ); } catch ( e ) { /* no long task timing here */ }
	if ( Q.get( 'ktxinfo' ) === '1' ) {

		el = document.createElement( 'pre' );
		el.id = 'ktxinfo';
		el.style.cssText = 'position:fixed;left:0;right:0;top:0;margin:0;padding:3px 5px;z-index:99999;pointer-events:none;font:10px/1.25 monospace;' +
			'color:#fff;background:rgba(0,0,0,.62);white-space:pre-wrap;overflow-wrap:anywhere';
		document.body.appendChild( el );

	}

	const text = ( s ) => {

		const K = town.ktxStats, o = s.shown.map( ( n, l ) => `${L[ l ]} ${n}` ).join( ' ' );
		const lines = [];
		lines.push( `${T.top === 'fine' ? ( T.sharp ? 'high' : 'medium' ) : T.tex === 256 ? 'phone' : 'low'} · ${town.eng.isWebGPU ? 'WebGPU' : 'WebGL2'} · gpu formats: ${feats.join( ' ' ) || 'none'} · ` +
			( K ? ( K.on ? `KTX2 on${K.force ? ' (forced ' + K.force + ')' : ''}` : `KTX2 OFF: ${K.why || 'failed'}` ) : 'KTX2 not asked for (WebP)' ) );
		if ( K && K.on ) lines.push( 'textures: ' + [ 'first', 'big', 'rest', 'nr' ].filter( ( k ) => K.sets[ k ] ).map( ( k ) => { const v = K.sets[ k ];
			return `${k} ${v.size} ${v.webp ? 'WebP (' + v.webp + ')' : ( FORMAT[ v.format ] || v.format || '' ) + ' ' + v.got + '/' + v.n} ${sec( v.at )}`; } ).join( ' · ' ) +
			( T.nr && ! K.sets.nr ? ' · nr to come' : '' ) +
			( K.gate === 'hold' ? `\nsharp textures wait for the cells within ${HOLD} m (${( K.hold / 1000 ).toFixed( 1 )}s so far)` :
				K.hold ? `\nsharp textures waited ${( K.hold / 1000 ).toFixed( 1 )}s for the cells within ${HOLD} m${K.gate === 'trickle' && ( s.wait || s.inflight || s.base > 0 ) ? ', now ' + TRICKLE + ' at a time while levels come in' : ''}` : '' ) );
		else lines.push( `textures: first ${T.tex} WebP · sharp ${T.big || T.sharp ? sec( log.sharp ) : 'none'} · nr ${T.nr ? sec( log.detail ) : 'none'}` );
		lines.push( ...tcText( town, sec ) ); // ?tc=1 (lite_tc.js): the format chosen and why, and where its arrays stand
		lines.push( `cells shown: ${o} · waiting for a level: ${s.wait} (in flight ${s.inflight}) · far level within ${close} m: ${s.loClose}` + ( s.base > 0 ? ` · base ${s.base} to come` : '' ) );
		lines.push( `first frame ${sec( log.first )} · mid in ${sec( log.mid )} · near in ${sec( log.near )}${N > 3 ? ' · fine in ' + sec( log.top ) : ''} · all in ${sec( log.all )} · ` +
			`near by (${close} m): far level until ${log.loClose ? sec( log.loClose ) : log.all ? '-' : '…'}, all levels in ${log.waitClose && ! s.waitClose ? sec( log.waitClose ) : log.all ? '-' : '…'} · now ${sec( s.t )} · ${( s.bytes / 1e6 ).toFixed( 1 )} MB` );
		if ( town.warnings.length ) lines.push( '! ' + town.warnings.slice( 0, 3 ).join( ' | ' ).slice( 0, 300 ) );
		return lines.join( '\n' );

	};

	const tick = () => {

		const t = Math.round( now() ), S = window.__stats;
		if ( ! log.first && lastTick && t - lastTick > log.gap[ 0 ] ) log.gap = [ t - lastTick, lastTick ];
		lastTick = t;
		if ( ! log.tex && town.layers ) log.tex = t;
		if ( ! log.settled && S ) log.settled = t;
		if ( ! hooked && town.sharp && town.detail ) {

			hooked = true; log.base = t;
			town.sharp.then( () => { log.sharp = Math.round( now() ); } );
			town.detail.then( () => { log.detail = Math.round( now() ); } );

		}

		if ( ! log.first && S && S.first ) log.first = S.first;
		const shown = L.map( () => 0 );
		let wait = 0, noMid = 0, noNear = 0, noTop = 0, loClose = 0, waitClose = 0;
		for ( const c of town.chunks ) {

			if ( c.mesh[ c.level ] ) shown[ c.level ] ++;
			if ( c.want > c.level ) { wait ++; if ( c.d < close ) waitClose ++; }
			if ( c.want >= 1 && c.level < 1 ) { noMid ++; if ( c.d < close ) loClose ++; }
			if ( c.want >= 2 && c.level < 2 ) noNear ++;
			if ( c.want >= 3 && c.level < 3 ) noTop ++;

		}

		const s = log.now = { t, shown, wait, noMid, noNear, noTop, loClose, waitClose, inflight: town.inflight.size, base: town.baseLeft, bytes: town.bytes };
		if ( log.first && ! log.all ) {

			if ( ! log.mid && ! noMid ) log.mid = t;
			if ( ! log.near && ! noNear ) log.near = t;
			if ( ! log.top && ! noTop ) log.top = t;
			if ( loClose ) log.loClose = t;
			if ( waitClose ) log.waitClose = t;
			if ( ! wait && ! town.inflight.size && town.baseLeft <= 0 ) log.all = t;
			if ( t - lastSample >= 250 && log.samples.length < 2400 ) { lastSample = t; log.samples.push( [ t, ...shown, wait, loClose, waitClose, Math.round( town.bytes / 1e5 ) / 10 ] ); }

		}

		if ( el && t - lastText >= 240 ) { lastText = t; el.textContent = text( s ); }

	};

	log.text = () => text( log.now );
	setInterval( tick, 100 );

}
