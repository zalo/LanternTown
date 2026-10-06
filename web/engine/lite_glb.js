// lite_glb.js - the baked town's geometry out of .glb files (tools/bake2glb.mjs <bake dir> --live), for the baked
// runtimes (lite.js, lite_v1.js).
//
// The files are plain glTF 2.0 binaries (KHR_meshopt_compression + KHR_mesh_quantization; any glTF reader opens
// them), written beside the packed ones of a bake:
//   base.glb             the lowest level of every cell, in the manifest's order (nearest the spawn first)
//   <m|n|r|f>_<cell>.glb one level of one cell (the name of its .bin with .glb)
// A level in them is the packer's two meshopt streams (vertex records, then indices) one after the other in the
// binary chunk - byte for byte what the packed file holds. So nothing is loaded through GLTFLoader here: this reads
// the 12-byte header and the JSON chunk, finds the two compressed bufferViews of a level, and hands the very same
// bytes to the runtime's own decode (BakedTown._build, the meshopt workers). What is drawn cannot differ, and the
// cost over the packed file is one JSON.parse of the file's table of contents.
// base.glb is read while it arrives: JSON first (it stands before the binary chunk), then cell after cell. It is
// asked for as soon as the manifest is known (glbAhead), while the textures load: its table of contents (some 20 kB
// on the wire, which the packed file does not have) and the round trip of the request are over by the time the
// first cell can be built.
//
//   ?fmt=packed | glb    which files to load. Default (DEFAULT_FMT): the .glb files, when the bake's manifest names them
//                        (manifest.glb, written by the export); a bake without them loads its packed files
//   ?glbahead=0          base.glb asked for only when the cells can be built, as the packed file is (for measuring)
// A bake keeps its packed files or not (manifest.glb.only, the export's --drop-packed: the .glb files are then all
// there is, and ?fmt=packed is not followed). Where they are kept, whatever goes wrong with a .glb (missing, cut
// short, of another bake) ends in the packed file of the same level and a line in town.warnings; where they are not,
// it is the error the packed file's would have been. town.geo says what was loaded and how long it took (the town
// page: __stats.geo).
// The files are read through net.js: a request that stops arriving (a tab back from the background, a link that
// dropped) is asked for again from where it stopped. base.glb is asked for until it is whole; a level that the
// network did not bring is an error with `net: true`, which the runtime asks for again later (BakedTown._fetch).

import { openBytes, netStats, FOREVER } from './net.js';

export const GLB_V = 1; // LIVE_V of tools/bake2glb.mjs: manifest.glb.v
export const DEFAULT_FMT = 'glb'; // ?fmt=packed: the packed files, of a bake that keeps them

const MAGIC = 0x46546C67, JSON_CHUNK = 0x4E4F534A, BIN_CHUNK = 0x004E4942;
export const glbName = ( file ) => file.replace( /\.bin$/, '' ) + '.glb';
const now = () => performance.now();

// town.geo: the format in use and the load's numbers (ms from the page's start unless said otherwise)
export function geoState( town ) {

	if ( town.geo ) return town.geo;
	const bake = town.bake, asked = new URLSearchParams( location.search ).get( 'fmt' );
	const has = !! ( bake.glb && bake.glb.v === GLB_V && bake.glb.base && bake.glb.base.file );
	const only = has && bake.glb.only === true; // no packed files beside the .glb
	const want = only ? 'glb' : asked === 'glb' || asked === 'packed' ? asked : DEFAULT_FMT;
	const G = town.geo = { fmt: want === 'glb' && has ? 'glb' : 'packed', asked, has, only, fallback: null,
		// base: the lowest level's file; t: asked for, answered, read from (the textures are in), table of contents read (glb),
		// first cell built, last cell built
		base: { file: null, bytes: 0, json: 0, t: { ask: 0, answer: 0, read: 0, toc: 0, first: 0, done: 0 } },
		// levels: the files fetched while walking; net: ms from asking to the last byte, build: ms from there to the mesh (sums)
		// last: when the latest of them stood
		levels: { files: 0, bytes: 0, json: 0, net: 0, build: 0, slowest: 0, last: 0 },
		io: netStats }; // (requests asked for again, see net.js)
	if ( asked === 'glb' && ! has ) fallback( town, new Error( 'the bake names no .glb files (node tools/bake2glb.mjs <bake dir> --live)' ) );
	if ( asked === 'packed' && only && town.warnings ) town.warnings.push( 'baked town: ?fmt=packed, but this bake keeps no packed files: loading its .glb files' );
	return G;

}

export const glbOn = ( town ) => geoState( town ).fmt === 'glb';

function fallback( town, e ) {

	const G = geoState( town );
	if ( G.only ) throw e; // (nothing to fall back to: the caller's error, as a packed file's would be)
	G.fmt = 'packed';
	if ( G.fallback ) return;
	G.fallback = String( e && e.message || e );
	const w = `baked town: ${G.fallback}: loading the packed files`;
	if ( town.warnings && ! town.warnings.includes( w ) ) town.warnings.push( w );

}

// for both formats: wraps the onChunk of _streamBase so that town.geo.base has the times of the first and the last cell
export function geoTimed( town, onChunk ) {

	const G = geoState( town ), B = G.base;
	if ( ! B.t.ask ) B.t.ask = Math.round( now() );
	B.t.read = Math.round( now() );
	if ( ! B.file ) B.file = G.fmt === 'glb' ? town.bake.glb.base.file : town.bake.base.file;
	return ( n ) => {

		if ( ! B.t.first ) B.t.first = Math.round( now() );
		if ( n >= town.chunks.length ) { B.t.done = Math.round( now() ); if ( G.fmt !== 'glb' ) B.bytes = town.bake.base.bytes; }
		onChunk( n );

	};

}

// for both formats: one level file is in (t0: when it was asked for, t1: when its bytes were there)
export function geoLevel( town, t0, t1, bytes ) {

	const L = geoState( town ).levels, t2 = now();
	L.files ++; L.bytes += bytes; L.net += t1 - t0; L.build += t2 - t1;
	if ( t2 - t0 > L.slowest ) L.slowest = Math.round( t2 - t0 );
	L.last = Math.round( t2 );
	L.net = Math.round( L.net * 10 ) / 10; L.build = Math.round( L.build * 10 ) / 10;

}

// ---------------------------------------------------------------------------------------------- the reader

// A .glb taken in pieces (push) or whole. level() gives the packed bytes of one level - the vertex stream and the
// index stream as one run - or null while they have not arrived.
export class GlbReader {

	constructor( name, stamp ) {

		this.name = name; this.stamp = stamp;
		this.buf = null; this.got = 0; this.total = 0;
		this.map = null; this.binAt = 0; this.jsonBytes = 0;
		this._head = [];

	}

	push( v ) {

		if ( ! this.buf ) {

			let h = v;
			if ( this._head.length || v.length < 12 ) {

				// (a first piece of under 12 bytes: wait for the header)
				this._head.push( v ); this.got += v.length;
				if ( this.got < 12 ) return;
				h = new Uint8Array( this.got );
				let o = 0;
				for ( const p of this._head ) { h.set( p, o ); o += p.length; }

			} else this.got = v.length;

			const dv = new DataView( h.buffer, h.byteOffset, h.byteLength );
			if ( dv.getUint32( 0, true ) !== MAGIC || dv.getUint32( 4, true ) !== 2 ) throw new Error( `${this.name}: not a .glb` );
			this.total = dv.getUint32( 8, true );
			if ( this.total < h.length ) throw new Error( `${this.name} is longer than its header says (stale bake?)` );
			// (the whole file in one piece is used where it is; pieces are gathered in one buffer of the file's size)
			if ( h.length === this.total ) this.buf = h; else { this.buf = new Uint8Array( this.total ); this.buf.set( h ); }
			this._head = null;

		} else {

			if ( this.got + v.length > this.total ) throw new Error( `${this.name} is longer than its header says (stale bake?)` );
			this.buf.set( v, this.got ); this.got += v.length;

		}

		if ( ! this.map && this.got >= 20 ) {

			const b = this.buf, dv = new DataView( b.buffer, b.byteOffset, b.byteLength ), n = dv.getUint32( 12, true );
			if ( dv.getUint32( 16, true ) !== JSON_CHUNK ) throw new Error( `${this.name}: not a .glb` );
			if ( this.got < 20 + n + 8 ) return;
			if ( dv.getUint32( 20 + n + 4, true ) !== BIN_CHUNK ) throw new Error( `${this.name}: no binary chunk` );
			this.map = this._levels( JSON.parse( new TextDecoder().decode( b.subarray( 20, 20 + n ) ) ) );
			this.binAt = 20 + n + 8; this.jsonBytes = n;

		}

	}

	get complete() { return !! this.buf && this.got === this.total; }

	// "cell:level" -> where its streams lie in the binary chunk
	_levels( j ) {

		const ex = j.scenes && j.scenes[ 0 ] && j.scenes[ 0 ].extras, stamp = ex && ex.baked && ex.baked.baked;
		if ( this.stamp !== undefined && stamp !== this.stamp ) throw new Error( `${this.name} is of another bake (${stamp}, the manifest's is ${this.stamp})` );
		const map = new Map();
		const stream = ( acc ) => {

			const v = j.bufferViews[ j.accessors[ acc ].bufferView ], e = v.extensions || {};
			const x = e.KHR_meshopt_compression || e.EXT_meshopt_compression;
			if ( ! x || x.buffer !== 0 ) throw new Error( `${this.name}: a level that is no meshopt stream of the binary chunk` );
			return x;

		};

		for ( const n of j.nodes ) {

			const b = n.extras && n.extras.baked;
			if ( ! b || n.mesh === undefined ) continue;
			const p = j.meshes[ n.mesh ].primitives[ 0 ], v = stream( p.attributes.POSITION ), i = stream( p.indices );
			map.set( b.cell + ':' + b.level, { at: v.byteOffset || 0, vlen: v.byteLength, nv: v.count, stride: v.byteStride, iat: i.byteOffset || 0, ilen: i.byteLength, ni: i.count, it: i.byteStride } );

		}

		return map;

	}

	level( cell, level, m ) {

		if ( ! this.map ) return null;
		const L = this.map.get( cell + ':' + level );
		if ( ! L ) throw new Error( `${this.name}: no ${cell} ${level} in it (stale bake?)` );
		if ( L.vlen !== m.vlen || L.vlen + L.ilen !== m.len || L.nv !== m.nv || L.ni !== m.ni || L.stride !== ( m.w || 10 ) * 2 || L.it !== m.it ) throw new Error( `${this.name}: ${cell} ${level} differs from the manifest (stale bake?)` );
		const a = this.binAt + L.at, i = this.binAt + L.iat;
		if ( L.iat === L.at + L.vlen ) return a + m.len <= this.got ? this.buf.subarray( a, a + m.len ) : null;
		// (a file laid out otherwise: the two streams are put together)
		if ( Math.max( a + L.vlen, i + L.ilen ) > this.got ) return null;
		const out = new Uint8Array( m.len );
		out.set( this.buf.subarray( a, a + L.vlen ) ); out.set( this.buf.subarray( i, i + L.ilen ), L.vlen );
		return out;

	}

}

// ---------------------------------------------------------------------------------------------- for the runtimes

// base.glb asked for now - the first thing of a load, before the textures - and read from in glbBase() when the cells
// can be built. (Until then the browser holds what arrives, up to its own limit: a few hundred kB.)
export function glbAhead( town ) {

	const G = geoState( town ), bake = town.bake;
	if ( G.fmt !== 'glb' || G.ahead || G.base.t.ask || new URLSearchParams( location.search ).get( 'glbahead' ) === '0' ) return;
	G.base.t.ask = Math.round( now() );
	G.ahead = openBytes( `${bake.dir}/${bake.glb.base.file}?v=${bake.baked}`, null, FOREVER );

}

// The lowest level of every cell from base.glb, cells built as their bytes arrive: put( c, bytes ) builds chunk c from
// its packed bytes, onChunk( n ) after each. -> true when it is done; false when nothing could be taken from the
// file (the caller then streams the packed base file). An error after the first cell goes to the caller, as the
// packed file's would.
export async function glbBase( town, put, onChunk ) {

	const bake = town.bake, G = geoState( town ), file = bake.glb.base.file, T = G.base.t;
	let next = 0;
	try {

		const b = G.ahead || openBytes( `${bake.dir}/${file}?v=${bake.baked}`, null, FOREVER );
		G.ahead = null;
		const r = await b.status();
		if ( ! r.ok ) throw new Error( `${file}: ${r.status}` );
		if ( ! T.answer ) T.answer = Math.round( b.answered || now() );
		let S = new GlbReader( file, bake.baked );
		const flush = () => {

			if ( S.map && ! T.toc ) T.toc = Math.round( now() );
			while ( next < town.chunks.length ) {

				const c = town.chunks[ next ], m = c.m.lo;
				const bytes = m.data || S.level( c.m.k, 'lo', m );
				if ( ! bytes ) break;
				put( c, bytes );
				next ++;
				onChunk( next );

			}

		};

		for ( ;; ) {

			const p = await b.read();
			if ( p.done ) break;
			// (the server sent the file from its start again: read anew; the cells built so far stand)
			if ( p.restart ) { S = new GlbReader( file, bake.baked ); continue; }
			S.push( p.value );
			flush();

		}

		town.bytes += S.got;
		G.base.file = file; G.base.bytes = S.got; G.base.json = S.jsonBytes;
		if ( next < town.chunks.length ) throw new Error( `${file} is shorter than its header says (stale bake?)` );
		return true;

	} catch ( e ) {

		if ( next > 0 ) throw e;
		fallback( town, e );
		G.base.file = bake.base.file; T.answer = T.toc = 0;
		return false;

	}

}

// One level of one cell from its .glb -> its packed bytes (as the .bin holds them), or null: the caller then fetches
// the packed file. m: the level's entry of the manifest
export async function glbLevel( town, cell, level, m ) {

	const bake = town.bake, G = geoState( town ), file = glbName( m.file );
	try {

		const b = openBytes( `${bake.dir}/${file}?v=${bake.baked}` );
		const r = await b.status();
		if ( ! r.ok ) throw new Error( `${file}: ${r.status}` );
		let S = new GlbReader( file, bake.baked );
		for ( ;; ) {

			const p = await b.read();
			if ( p.done ) break;
			if ( p.restart ) S = new GlbReader( file, bake.baked );
			else S.push( p.value );

		}

		const bytes = S.complete ? S.level( cell, level, m ) : null;
		if ( ! bytes ) throw new Error( `${file} is cut short (stale bake?)` );
		town.bytes += S.got;
		G.levels.json += S.jsonBytes;
		return bytes;

	} catch ( e ) {

		if ( e.net ) throw e; // (the network, not the file: the caller asks again later)
		fallback( town, e );
		return null;

	}

}
