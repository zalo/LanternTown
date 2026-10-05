#!/usr/bin/env node
// Lossless delivery copies of a town's GLBs (the streaming "Studio" engine reads them, web/engine/glbrelease.js).
//
//     node tools/glbopt.mjs [--town town] [--meshopt 1|0] [--struct 0|1] [--codec ext|khr] [--bundle 1|0] [--default] [--force] [--quiet]
//     node tools/glbopt.mjs --all                          every variant that is there, each with its own options
//     node tools/glbopt.mjs --verify [--town town] [--name <variant> | --all]
//     node tools/glbopt.mjs --stats  [--town town] [--name <variant> | --all]
//     (--json <town.json> --src <dir> --dst <dir>: the same on a copy, for trying the tool)
//
// A post-build step: reads out/<town>/town.json and the files it names under out/glb (every proto at every level +
// the terrain), NEVER writes there (Blender and the bake read out/glb as built), and writes
//     out/web/_glb/<town>/<variant>/<asset>/<name>.glb     one optimised file per source file
//     out/web/_glb/<town>/<variant>/coarse.<hash>.bin      --bundle: the files every visitor needs before the first
//                                                          frame (each proto's coarsest level + the terrain), end to end
//     out/web/_glb/<town>/<variant>/manifest.json          the tool's own state: what is there, from which source (hash)
//     out/web/_glb/<town>/<variant>/index.json             what the page reads: the look table, per file the source it
//                                                          was made from (mtime, size), its size, name tag and bundle offset
//     out/web/_glb/<town>/release.json                     --default: this variant's index.json, what the page uses when
//                                                          it is not told (no release.json: the page reads out/glb)
// Re-runnable and incremental: a source file whose content hash is in the manifest is not processed again, outputs
// are a function of the source bytes and the options only (no times, no order of arrival), files that left the town
// are removed. A source that changes while it is read is retried, then skipped and reported (exit code 2).
//
// What it may do - nothing here changes a single bit of a vertex:
//   --meshopt   EXT_meshopt_compression on the float32 / integer data as it is: vertex codec per attribute stream,
//               index codec per index stream, filter NONE (no quantisation). --codec khr writes
//               KHR_meshopt_compression with the v1 vertex codec instead (about 12 % smaller before gzip, 2.5 % after).
//   --struct    one material table for the town (manifest.materials; primitives name their look by id in
//               extras.tsq_m = [ id, vertices, id, vertices, .. ] and the file carries no materials), primitives of a
//               mesh that the runtime draws with one material joined (same library material + shader parameters +
//               shadow class: vertices appended in source order, indices offset - tint / glow / rough / opacity /
//               curtain stay per range, exactly what the runtime writes per vertex anyway), one bufferView per
//               attribute per file. Nodes are kept as they are (the files are flat: body + parts + sockets).
//               Off unless asked for: the data is the same, the picture is not quite. The renderer sorts opaque
//               meshes by depth, so two primitives it used to draw in an order of its own choosing are now drawn in
//               file order; where their surfaces meet at the same depth the other one shows (measured on the town:
//               445 and 21 of 1 440 000 pixels in two of four review shots, a sign's frame against its board).
//   --reorder   none (default) | tri | full.  Measured on this town: the exporter's vertex order compresses best -
//               any vertex permutation (full = meshopt's cache + fetch order) costs 16-19 % more bytes after gzip.
//               tri reorders triangles only (vertices untouched). Both keep the triangle order of blended materials
//               and of primitives with coincident triangles, where the order is visible.
//   --verify    decodes every output with the decoder the browser uses and compares it with its source: per mesh and
//               material the same multiset of full vertex records (bit for bit) and of triangles, in the same order
//               where the order was kept; the same nodes (names, extras, transforms), the same material extras.
//
// Variant names are made of the options: m = meshopt, s = struct, t / r = reorder tri / full, k = khr codec.
// Page switches: ?glb=raw | <variant>, ?meshopt=0, ?struct=0, ?bundle=0, ?mow=<decode workers> (web/engine/glbrelease.js);
// the page takes a file from the release only when it was made from the source town.json names, else from out/glb.
// Exit codes: 0, 1 (--verify: an output differs from its source), 2 (a source could not be read whole / an output is stale).

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { MeshoptEncoder } from 'meshoptimizer';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

const TOOL = 4; // part of every output's identity: raise it when the output of the same source + options changes
const ROOT = path.dirname( path.dirname( fileURLToPath( import.meta.url ) ) );

// ------------------------------------------------------------------------------------------ options

const A = { town: 'town', json: '', meshopt: '1', struct: '0', bundle: '1', reorder: 'none', codec: 'ext', name: '', src: '', dst: '', verify: false, stats: false, all: false, default: false, force: false, quiet: false };
for ( let i = 2; i < process.argv.length; i ++ ) {

	const a = process.argv[ i ];
	if ( ! a.startsWith( '--' ) ) continue;
	const [ k, v ] = a.slice( 2 ).split( '=' );
	if ( ! ( k in A ) ) {

		console.error( `unknown option --${k}` );
		process.exit( 1 );

	}

	if ( typeof A[ k ] === 'boolean' ) A[ k ] = v === undefined ? true : v !== '0';
	else A[ k ] = v !== undefined ? v : process.argv[ ++ i ];

}

let OPT = { meshopt: A.meshopt !== '0', struct: A.struct !== '0', reorder: A.reorder, codec: A.codec };
if ( ! OPT.meshopt ) OPT.codec = 'ext'; // (no codec in use: one name for the variant)
if ( ! [ 'none', 'tri', 'full' ].includes( OPT.reorder ) || ! [ 'ext', 'khr' ].includes( OPT.codec ) ) {

	console.error( '--reorder none|tri|full, --codec ext|khr' );
	process.exit( 1 );

}

const variantOf = ( o ) => ( o.meshopt ? 'm' : '' ) + ( o.struct ? 's' : '' ) + ( o.reorder === 'tri' ? 't' : o.reorder === 'full' ? 'r' : '' ) + ( o.meshopt && o.codec === 'khr' ? 'k' : '' ) || 'p';
const TOWN_JSON = A.json ? path.resolve( A.json ) : path.join( ROOT, 'out', A.town, 'town.json' );
const GLB = A.src ? path.resolve( A.src ) : path.join( ROOT, 'out', 'glb' ); // (--src / --dst: for trying the tool on a copy)
const REL = path.join( A.dst ? path.resolve( A.dst ) : path.join( ROOT, 'out', 'web', '_glb' ), A.town );
if ( path.resolve( REL ).startsWith( path.join( ROOT, 'out', 'glb' ) + path.sep ) ) {

	console.error( 'the release may not be written into out/glb' );
	process.exit( 1 );

}

await MeshoptEncoder.ready;
await MeshoptDecoder.ready;

// ------------------------------------------------------------------------------------------ reading a GLB

const COMP = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const NUM = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };
const MESHOPT = [ 'EXT_meshopt_compression', 'KHR_meshopt_compression' ];

class Unsupported extends Error {}

const need = ( ok, why ) => {

	if ( ! ok ) throw new Unsupported( why );

};

const only = ( o, keys, what ) => {

	for ( const k in o ) need( keys.includes( k ), `${what}.${k}` );

};

function parseGlb( buf ) {

	if ( buf.length < 20 || buf.readUInt32LE( 0 ) !== 0x46546C67 || buf.readUInt32LE( 4 ) !== 2 ) throw new Error( 'not a GLB' );
	if ( buf.readUInt32LE( 8 ) !== buf.length ) throw new Error( 'GLB length does not match the file (being written?)' );
	const jl = buf.readUInt32LE( 12 );
	if ( buf.readUInt32LE( 16 ) !== 0x4E4F534A || 20 + jl > buf.length ) throw new Error( 'GLB without a JSON chunk' );
	const json = JSON.parse( buf.subarray( 20, 20 + jl ).toString( 'utf8' ) );
	let bin = new Uint8Array( 0 );
	if ( 20 + jl + 8 <= buf.length ) {

		const bl = buf.readUInt32LE( 20 + jl );
		if ( buf.readUInt32LE( 24 + jl ) !== 0x004E4942 || 28 + jl + bl > buf.length ) throw new Error( 'GLB with a broken BIN chunk' );
		bin = new Uint8Array( buf.buffer, buf.byteOffset + 28 + jl, bl );

	}

	return { json, bin, views: new Map() };

}

// decoded bytes of a bufferView (meshopt views through the decoder three ships: what the browser runs)
function viewBytes( G, i ) {

	let b = G.views.get( i );
	if ( b ) return b;
	const bv = G.json.bufferViews[ i ];
	const name = MESHOPT.find( ( n ) => bv.extensions && bv.extensions[ n ] );
	if ( name ) {

		const x = bv.extensions[ name ];
		need( x.buffer === 0, 'meshopt view outside the BIN chunk' );
		b = new Uint8Array( x.count * x.byteStride );
		MeshoptDecoder.decodeGltfBuffer( b, x.count, x.byteStride, G.bin.subarray( x.byteOffset || 0, ( x.byteOffset || 0 ) + x.byteLength ), x.mode, x.filter || 'NONE' );
		if ( b.length !== bv.byteLength ) throw new Error( 'meshopt view decodes to another length than its bufferView' );

	} else {

		need( bv.buffer === 0 && ! G.json.buffers[ 0 ].uri, 'external buffer' );
		only( bv, [ 'buffer', 'byteOffset', 'byteLength', 'byteStride', 'target', 'name' ], 'bufferView' );
		b = G.bin.subarray( bv.byteOffset || 0, ( bv.byteOffset || 0 ) + bv.byteLength );
		if ( b.length !== bv.byteLength ) throw new Error( 'bufferView past the end of the BIN chunk' );

	}

	G.views.set( i, b );
	return b;

}

// tightly packed copy of an accessor's elements
function accessorBytes( G, i ) {

	const a = G.json.accessors[ i ];
	only( a, [ 'bufferView', 'byteOffset', 'componentType', 'count', 'type', 'min', 'max', 'normalized', 'name' ], 'accessor' );
	need( a.bufferView !== undefined && COMP[ a.componentType ] && NUM[ a.type ], 'accessor layout' );
	const el = COMP[ a.componentType ] * NUM[ a.type ], src = viewBytes( G, a.bufferView );
	const stride = G.json.bufferViews[ a.bufferView ].byteStride || el, o = a.byteOffset || 0;
	const out = new Uint8Array( a.count * el );
	if ( o + ( a.count ? ( a.count - 1 ) * stride + el : 0 ) > src.length ) throw new Error( 'accessor past the end of its bufferView' );
	if ( stride === el ) out.set( src.subarray( o, o + out.length ) );
	else for ( let k = 0; k < a.count; k ++ ) out.set( src.subarray( o + k * stride, o + k * stride + el ), k * el );
	return out;

}

// the part of a file this tool rewrites, as plain data: meshes -> primitives -> named attribute byte arrays + indices.
// Anything it does not know for certain how to carry over makes the file Unsupported (it is then copied as it is).
function readModel( G ) {

	const J = G.json;
	only( J, [ 'asset', 'scene', 'scenes', 'nodes', 'materials', 'meshes', 'accessors', 'bufferViews', 'buffers', 'extensionsUsed', 'extensionsRequired' ], 'gltf' );
	for ( const e of [ ...( J.extensionsUsed || [] ), ...( J.extensionsRequired || [] ) ] ) need( MESHOPT.includes( e ), 'extension ' + e );
	need( ( J.buffers || [] ).length >= 1, 'no buffer' );
	for ( const n of J.nodes || [] ) only( n, [ 'name', 'extras', 'translation', 'rotation', 'scale', 'matrix', 'children', 'mesh' ], 'node' );
	for ( const s of J.scenes || [] ) only( s, [ 'name', 'nodes', 'extras' ], 'scene' );
	for ( const m of J.materials || [] ) need( ! m.extensions, 'material extension' );
	const meshes = ( J.meshes || [] ).map( ( m ) => {

		only( m, [ 'name', 'primitives' ], 'mesh' );
		return { name: m.name, prims: m.primitives.map( ( p ) => {

			only( p, [ 'attributes', 'indices', 'material', 'mode', 'extras' ], 'primitive' );
			need( ( p.mode ?? 4 ) === 4 && p.indices !== undefined, 'primitive that is not an indexed triangle list' );
			const ia = J.accessors[ p.indices ];
			need( ia.type === 'SCALAR' && ( ia.componentType === 5123 || ia.componentType === 5125 ) && ia.count % 3 === 0, 'index accessor' );
			const ib = accessorBytes( G, p.indices );
			const idx = ia.componentType === 5123 ? Uint32Array.from( new Uint16Array( ib.buffer, 0, ia.count ) ) : new Uint32Array( ib.buffer, 0, ia.count );
			const attrs = [];
			let count = - 1;
			for ( const [ name, ai ] of Object.entries( p.attributes ) ) {

				const a = J.accessors[ ai ];
				need( ( COMP[ a.componentType ] * NUM[ a.type ] ) % 4 === 0, 'attribute element that is not a multiple of 4 bytes' );
				need( count < 0 || count === a.count, 'attributes of different lengths' );
				count = a.count;
				attrs.push( { name, type: a.type, comp: a.componentType, norm: !! a.normalized, el: COMP[ a.componentType ] * NUM[ a.type ], bytes: accessorBytes( G, ai ) } );

			}

			need( count >= 0, 'primitive without attributes' );
			for ( let k = 0; k < idx.length; k ++ ) if ( idx[ k ] >= count ) throw new Error( 'index past the vertices' );
			// (struct outputs carry their looks in extras.tsq_m instead of a material)
			return { attrs, count, idx, wide: ia.componentType === 5125, material: p.material, tsq_m: p.extras ? p.extras.tsq_m : undefined, extras: p.extras };

		} ) };

	} );
	return { json: J, meshes };

}

// ------------------------------------------------------------------------------------------ looks

const canon = ( v ) => Array.isArray( v ) ? v.map( canon ) : v && typeof v === 'object' ? Object.fromEntries( Object.keys( v ).sort().map( ( k ) => [ k, canon( v[ k ] ) ] ) ) : v;
const lookId = ( extras ) => crypto.createHash( 'sha256' ).update( JSON.stringify( canon( extras || {} ) ) ).digest( 'hex' ).slice( 0, 10 );

// what decides which material object and which shadow class the runtime gives a primitive (engine/assets.js adopt(),
// engine/materials.js get()): the library material and every parameter that is not written per vertex.
// tint, glow, rough, opacity and curtain are (materials.js vertexValues()); glow > 0 also takes the mesh out of the
// shadow pass. The page checks the material again for every joined range and falls back to the source file if not.
const PER_VERTEX = [ 'glow', 'rough', 'opacity', 'curtain' ];
function drawKey( extras, sig ) {

	const ex = extras || {};
	let p = ex.tsq_params;
	if ( typeof p === 'string' ) {

		try {

			p = JSON.parse( p );

		} catch ( e ) {

			return null;

		}

	}

	if ( p === undefined || p === null ) p = {};
	if ( typeof p !== 'object' || Array.isArray( p ) ) return null;
	const rest = {};
	for ( const k of Object.keys( p ).sort() ) if ( ! PER_VERTEX.includes( k ) && p[ k ] !== undefined && p[ k ] !== null ) rest[ k ] = p[ k ];
	return JSON.stringify( [ ex.tsq_lib || 'plain', canon( rest ), p.glow > 0, sig ] );

}

const BLENDED = /^(glass_|window_|water|neon|emissive)/;

// triangle order is visible where triangles blend or lie on top of each other
function orderMatters( prim, extras ) {

	const ex = extras || {};
	if ( BLENDED.test( ex.tsq_lib || '' ) || /opacity|glow/.test( typeof ex.tsq_params === 'string' ? ex.tsq_params : JSON.stringify( ex.tsq_params || {} ) ) ) return true;
	const pos = prim.attrs.find( ( a ) => a.name === 'POSITION' );
	if ( ! pos || pos.el !== 12 ) return true;
	const f = new Uint32Array( pos.bytes.buffer, pos.bytes.byteOffset, prim.count * 3 ), ids = new Map(), vid = new Uint32Array( prim.count );
	for ( let v = 0; v < prim.count; v ++ ) {

		const k = f[ v * 3 ] + ',' + f[ v * 3 + 1 ] + ',' + f[ v * 3 + 2 ];
		let id = ids.get( k );
		if ( id === undefined ) ids.set( k, id = ids.size );
		vid[ v ] = id;

	}

	const seen = new Set(), I = prim.idx;
	for ( let t = 0; t < I.length; t += 3 ) {

		const k = [ vid[ I[ t ] ], vid[ I[ t + 1 ] ], vid[ I[ t + 2 ] ] ].sort( ( a, b ) => a - b ).join( ',' );
		if ( seen.has( k ) ) return true;
		seen.add( k );

	}

	return false;

}

// --reorder: meshopt's vertex cache order for the triangles, and (full) its fetch order for the vertices
function reorder( prim, mode ) {

	const idx = prim.idx.slice();
	const [ remap ] = MeshoptEncoder.reorderMesh( idx, true, false );
	const n = prim.count, to = new Uint32Array( n ).fill( 0xffffffff );
	let used = 0;
	for ( let v = 0; v < remap.length; v ++ ) if ( remap[ v ] !== 0xffffffff ) {

		to[ v ] = remap[ v ];
		used ++;

	}

	for ( let v = 0; v < n; v ++ ) if ( to[ v ] === 0xffffffff ) to[ v ] = used ++; // vertices no triangle uses stay, at the end
	if ( mode === 'tri' ) {

		const back = new Uint32Array( n );
		for ( let v = 0; v < n; v ++ ) back[ to[ v ] ] = v;
		for ( let k = 0; k < idx.length; k ++ ) idx[ k ] = back[ idx[ k ] ];
		return { ...prim, idx };

	}

	const attrs = prim.attrs.map( ( a ) => {

		const bytes = new Uint8Array( a.bytes.length );
		for ( let v = 0; v < n; v ++ ) bytes.set( a.bytes.subarray( v * a.el, v * a.el + a.el ), to[ v ] * a.el );
		return { ...a, bytes };

	} );
	return { ...prim, attrs, idx };

}

// ------------------------------------------------------------------------------------------ writing a GLB

const concat = ( list ) => {

	const out = new Uint8Array( list.reduce( ( n, b ) => n + b.length, 0 ) );
	let o = 0;
	for ( const b of list ) {

		out.set( b, o );
		o += b.length;

	}

	return out;

};

// -> { glb: Buffer, looks: { id: extras }, stats }
function optimise( M, opt ) {

	const J = M.json, mats = J.materials || [], looks = {};
	const stats = { prims: 0, prims_out: 0, kept_order: 0, unused_vertices: 0 };
	const extrasOf = ( p ) => p.material === undefined ? {} : ( mats[ p.material ].extras || {} );

	// 1. primitives of each mesh as they will be written: [ { attrs, count, idx, wide, material, ranges } ]
	const meshes = M.meshes.map( ( m ) => {

		let prims = m.prims.map( ( p ) => {

			need( p.extras === undefined, 'primitive extras' );
			stats.prims ++;
			const used = new Uint8Array( p.count );
			for ( let k = 0; k < p.idx.length; k ++ ) used[ p.idx[ k ] ] = 1;
			for ( let v = 0; v < p.count; v ++ ) if ( ! used[ v ] ) stats.unused_vertices ++;
			if ( opt.reorder === 'none' ) return p;
			if ( orderMatters( p, extrasOf( p ) ) ) {

				stats.kept_order ++;
				return p;

			}

			return reorder( p, opt.reorder );

		} );
		if ( opt.struct ) {

			// join what the runtime draws with one material, in source order; a joined primitive keeps 16 bit indices
			const groups = new Map();
			prims.forEach( ( p, i ) => {

				const ex = extrasOf( p ), id = lookId( ex );
				looks[ id ] = canon( ex );
				const sig = p.attrs.map( ( a ) => `${a.name}:${a.type}:${a.comp}:${a.norm ? 1 : 0}` ).join( ',' );
				const key = drawKey( ex, sig ) ?? 'alone:' + i;
				let g = groups.get( key );
				if ( ! g ) groups.set( key, g = [] );
				let last = g[ g.length - 1 ];
				if ( ! last || ( ! last.wide && ! p.wide && last.count + p.count > 65535 ) || last.wide !== p.wide ) g.push( last = { at: i, members: [], count: 0, wide: p.wide } );
				last.members.push( { p, id } );
				last.count += p.count;

			} );
			prims = [ ...groups.values() ].flat().sort( ( a, b ) => a.at - b.at ).map( ( g ) => {

				const first = g.members[ 0 ].p, ranges = [];
				const idx = new Uint32Array( g.members.reduce( ( n, e ) => n + e.p.idx.length, 0 ) );
				let base = 0, at = 0;
				for ( const { p, id } of g.members ) {

					for ( let k = 0; k < p.idx.length; k ++ ) idx[ at + k ] = p.idx[ k ] + base;
					at += p.idx.length;
					base += p.count;
					ranges.push( id, p.count );

				}

				const attrs = first.attrs.map( ( a, ai ) => ( { ...a, bytes: g.members.length === 1 ? a.bytes : concat( g.members.map( ( e ) => e.p.attrs[ ai ].bytes ) ) } ) );
				return { attrs, count: base, idx, wide: g.wide, ranges };

			} );

		}

		stats.prims_out += prims.length;
		return { name: m.name, prims };

	} );

	// 2. accessors + views. struct: one view per kind of stream for the whole file; otherwise one per accessor, in the
	// exporter's order (the attributes of a primitive, then its indices). Equal data is written once.
	const accessors = [], views = [], byData = new Map();
	const streams = new Map(); // struct: key -> { parts, length, el, index, view }
	const sha = ( b ) => crypto.createHash( 'sha1' ).update( b ).digest( 'base64' );
	const put = ( bytes, def, el, index ) => {

		const key = `${index ? '' : def.name}|${def.type}|${def.componentType}|${def.normalized ? 1 : 0}|${def.count}|${sha( bytes )}`;
		let ai = byData.get( key );
		if ( ai !== undefined ) return ai;
		ai = accessors.length;
		if ( opt.struct ) {

			const sk = index ? 'i' + el : `${def.name}|${def.type}|${def.componentType}`;
			let s = streams.get( sk );
			if ( ! s ) {

				streams.set( sk, s = { parts: [], length: 0, el, index, view: views.length } );
				views.push( s );

			}

			accessors.push( { bufferView: s.view, byteOffset: s.length, ...def } );
			s.parts.push( bytes );
			s.length += bytes.length;

		} else {

			accessors.push( { bufferView: views.length, ...def } );
			views.push( { parts: [ bytes ], length: bytes.length, el, index, single: true } );

		}

		delete accessors[ ai ].name;
		byData.set( key, ai );
		return ai;

	};

	const sets = new Map(); // struct: accessor set -> looks -> its index accessor
	const outMeshes = meshes.map( ( m ) => ( { name: m.name, primitives: m.prims.map( ( p ) => {

		const attributes = {};
		for ( const a of p.attrs ) {

			const def = { name: a.name, componentType: a.comp, count: p.count, type: a.type };
			if ( a.norm ) def.normalized = true;
			if ( a.name === 'POSITION' && a.comp === 5126 && p.count ) {

				const f = new Float32Array( a.bytes.buffer, a.bytes.byteOffset, p.count * NUM[ a.type ] ), n = NUM[ a.type ];
				def.max = new Array( n ).fill( - Infinity );
				def.min = new Array( n ).fill( Infinity );
				for ( let k = 0; k < f.length; k ++ ) {

					const c = k % n;
					if ( f[ k ] > def.max[ c ] ) def.max[ c ] = f[ k ];
					if ( f[ k ] < def.min[ c ] ) def.min[ c ] = f[ k ];

				}

			}

			attributes[ a.name ] = put( a.bytes, def, a.el, false );

		}

		const ib = p.wide ? new Uint8Array( p.idx.buffer, p.idx.byteOffset, p.idx.byteLength ) : new Uint8Array( Uint16Array.from( p.idx ).buffer );
		const out = { attributes, indices: put( ib, { componentType: p.wide ? 5125 : 5123, count: p.idx.length, type: 'SCALAR' }, p.wide ? 4 : 2, true ) };
		if ( opt.struct ) {

			// GLTFLoader makes one geometry per set of accessors and hands the primitive's extras to it: the same data
			// with other looks gets an index accessor of its own (an entry in the JSON, no bytes)
			const set = JSON.stringify( [ attributes, out.indices ] ), looks = p.ranges.join();
			let by = sets.get( set );
			if ( ! by ) sets.set( set, by = new Map( [ [ looks, out.indices ] ] ) );
			else if ( by.has( looks ) ) out.indices = by.get( looks );
			else {

				accessors.push( { ...accessors[ out.indices ] } );
				by.set( looks, out.indices = accessors.length - 1 );

			}

			out.extras = { tsq_m: p.ranges };

		} else if ( p.material !== undefined ) out.material = p.material;
		return out;

	} ) } ) );

	// 3. the BIN chunk
	const ext = opt.codec === 'khr' ? 'KHR_meshopt_compression' : 'EXT_meshopt_compression';
	const chunks = [], bufferViews = [];
	let off = 0, plain = 0;
	for ( const v of views ) {

		const raw = v.parts.length === 1 ? v.parts[ 0 ] : concat( v.parts ), count = raw.length / v.el;
		const bv = { buffer: 0, byteOffset: 0, byteLength: raw.length };
		if ( ! v.index && ! v.single ) bv.byteStride = v.el; // (several accessors read this view)
		bv.target = v.index ? 34963 : 34962;
		let data = raw;
		if ( opt.meshopt && count > 0 ) {

			data = v.index ? MeshoptEncoder.encodeIndexBuffer( raw, count, v.el )
				: MeshoptEncoder.encodeVertexBufferLevel( raw, count, v.el, opt.codec === 'khr' ? 3 : 2, opt.codec === 'khr' ? 1 : 0 );
			bv.buffer = 1;
			bv.byteOffset = plain;
			bv.extensions = { [ ext ]: { buffer: 0, byteOffset: off, byteLength: data.length, byteStride: v.el, count, mode: v.index ? 'TRIANGLES' : 'ATTRIBUTES' } };
			plain += raw.length + ( 4 - raw.length % 4 ) % 4;

		} else bv.byteOffset = off;
		chunks.push( data );
		off += data.length;
		const pad = ( 4 - off % 4 ) % 4;
		if ( pad ) chunks.push( new Uint8Array( pad ) );
		off += pad;
		bufferViews.push( bv );

	}

	const out = { asset: { version: '2.0', generator: `tsq glbopt ${TOOL} (${J.asset?.generator || '?'})` } };
	const packed = opt.meshopt && plain > 0;
	if ( packed ) out.extensionsUsed = out.extensionsRequired = [ ext ];
	if ( J.scene !== undefined ) out.scene = J.scene;
	if ( J.scenes ) out.scenes = J.scenes;
	if ( J.nodes ) out.nodes = J.nodes;
	if ( ! opt.struct && J.materials ) out.materials = J.materials;
	out.meshes = outMeshes;
	out.accessors = accessors;
	out.bufferViews = bufferViews;
	out.buffers = [ { byteLength: off } ];
	if ( packed ) out.buffers.push( { byteLength: plain, extensions: { [ ext ]: { fallback: true } } } );

	let js = Buffer.from( JSON.stringify( out ), 'utf8' );
	if ( js.length % 4 ) js = Buffer.concat( [ js, Buffer.alloc( 4 - js.length % 4, 0x20 ) ] );
	const head = Buffer.alloc( 20 ), bh = Buffer.alloc( 8 );
	head.writeUInt32LE( 0x46546C67, 0 ); head.writeUInt32LE( 2, 4 ); head.writeUInt32LE( 28 + js.length + off, 8 );
	head.writeUInt32LE( js.length, 12 ); head.writeUInt32LE( 0x4E4F534A, 16 );
	bh.writeUInt32LE( off, 0 ); bh.writeUInt32LE( 0x004E4942, 4 );
	return { glb: Buffer.concat( [ head, js, bh, ...chunks.map( ( c ) => Buffer.from( c.buffer, c.byteOffset, c.byteLength ) ) ] ), looks, stats };

}

// ------------------------------------------------------------------------------------------ verification

// compares an output with its source; returns a list of differences (empty = the same content)
function compare( srcBuf, outBuf, table, opt ) {

	const bad = [], info = { ordered: 0, unordered: 0, vertices: 0, triangles: 0 };
	const S = readModel( parseGlb( srcBuf ) ), Oj = parseGlb( outBuf ), O = readModel( Oj );
	const sj = S.json, oj = O.json, eq = ( a, b ) => JSON.stringify( canon( a ) ) === JSON.stringify( canon( b ) );
	if ( ! eq( sj.scene, oj.scene ) || ! eq( sj.scenes, oj.scenes ) ) bad.push( 'scenes differ' );
	if ( ( sj.nodes || [] ).length !== ( oj.nodes || [] ).length ) bad.push( 'node count differs' );
	else ( sj.nodes || [] ).forEach( ( n, i ) => {

		if ( ! eq( n, oj.nodes[ i ] ) ) bad.push( `node ${i} (${n.name}) differs` ); // name, extras (sockets, parts), transform, mesh index, children

	} );
	if ( S.meshes.length !== O.meshes.length ) return [ ...bad, 'mesh count differs' ];
	for ( const x of oj.bufferViews || [] ) for ( const n of MESHOPT ) if ( x.extensions?.[ n ] && ( x.extensions[ n ].filter || 'NONE' ) !== 'NONE' ) bad.push( 'a meshopt filter is in use' );
	if ( ! opt.struct && ! eq( sj.materials, oj.materials ) ) bad.push( 'materials differ' );
	const smat = sj.materials || [];
	S.meshes.forEach( ( sm, mi ) => {

		const om = O.meshes[ mi ], where = `mesh ${mi} (${sm.name})`;
		if ( sm.name !== om.name ) bad.push( `${where}: name differs` );
		// full vertex records (every attribute of a vertex, in the source's attribute order), numbered by content
		const names = sm.prims[ 0 ] ? sm.prims[ 0 ].attrs.map( ( a ) => a.name ) : [];
		const ids = new Map();
		const records = ( p ) => {

			const at = names.map( ( n ) => p.attrs.find( ( a ) => a.name === n ) );
			if ( at.includes( undefined ) || p.attrs.length !== names.length ) return null;
			const el = at.reduce( ( n, a ) => n + a.el, 0 ), rec = Buffer.alloc( el ), out = new Uint32Array( p.count );
			for ( let v = 0; v < p.count; v ++ ) {

				let o = 0;
				for ( const a of at ) {

					rec.set( a.bytes.subarray( v * a.el, v * a.el + a.el ), o );
					o += a.el;

				}

				const k = rec.toString( 'latin1' );
				let id = ids.get( k );
				if ( id === undefined ) ids.set( k, id = ids.size );
				out[ v ] = id;

			}

			return out;

		};

		const sig = ( p ) => p.attrs.map( ( a ) => `${a.name}:${a.type}:${a.comp}:${a.norm ? 1 : 0}` ).sort().join( ',' );
		// triangles as content ids, turned so that the smallest comes first (the index codec may turn a triangle; its winding stays)
		const tris = ( rec, idx, from, to ) => {

			const out = [];
			for ( let t = 0; t < idx.length; t += 3 ) {

				const i0 = idx[ t ], i1 = idx[ t + 1 ], i2 = idx[ t + 2 ];
				const inside = i0 >= from && i0 < to;
				if ( inside !== ( i1 >= from && i1 < to ) || inside !== ( i2 >= from && i2 < to ) ) return null; // a triangle across two looks
				if ( ! inside ) continue;
				const a = rec[ i0 ], b = rec[ i1 ], c = rec[ i2 ];
				out.push( a <= b && a <= c ? `${a},${b},${c}` : b <= a && b <= c ? `${b},${c},${a}` : `${c},${a},${b}` );

			}

			return out;

		};

		// source: per look (material extras) the vertex ids and triangles of its primitives, in order
		const want = new Map(), got = new Map();
		const add = ( map, key, v, t, s ) => {

			let e = map.get( key );
			if ( ! e ) map.set( key, e = { v: [], t: [], n: 0, sig: s } );
			if ( e.sig !== s ) bad.push( `${where}: attribute layouts differ within a look` );
			for ( const x of v ) e.v.push( x );
			for ( const x of t ) e.t.push( x );
			e.n ++;

		};

		for ( const p of sm.prims ) {

			const rec = records( p );
			if ( ! rec ) return bad.push( `${where}: source primitives with different attributes` );
			const ex = p.material === undefined ? {} : ( smat[ p.material ].extras || {} );
			add( want, JSON.stringify( canon( ex ) ), rec, tris( rec, p.idx, 0, p.count ), sig( p ) );
			info.vertices += p.count;
			info.triangles += p.idx.length / 3;

		}

		const known = ids.size;
		for ( const p of om.prims ) {

			const rec = records( p );
			if ( ! rec ) return bad.push( `${where}: output attributes differ` );
			let ranges;
			if ( opt.struct ) {

				if ( ! Array.isArray( p.tsq_m ) || p.tsq_m.length % 2 || p.material !== undefined ) return bad.push( `${where}: output primitive without looks` );
				ranges = [];
				let v0 = 0;
				for ( let k = 0; k < p.tsq_m.length; k += 2 ) {

					if ( ! table[ p.tsq_m[ k ] ] ) return bad.push( `${where}: look ${p.tsq_m[ k ]} is not in the manifest` );
					ranges.push( [ JSON.stringify( canon( table[ p.tsq_m[ k ] ] ) ), v0, v0 + p.tsq_m[ k + 1 ] ] );
					v0 += p.tsq_m[ k + 1 ];

				}

				if ( v0 !== p.count ) bad.push( `${where}: looks do not cover the vertices` );

			} else ranges = [ [ JSON.stringify( canon( p.material === undefined ? {} : ( ( oj.materials || [] )[ p.material ].extras || {} ) ) ), 0, p.count ] ];
			let nt = 0;
			for ( const [ key, a, b ] of ranges ) {

				const t = tris( rec, p.idx, a, b );
				if ( ! t ) return bad.push( `${where}: a triangle spans two looks` );
				nt += t.length;
				add( got, key, rec.subarray( a, b ), t, sig( p ) );

			}

			if ( nt !== p.idx.length / 3 ) bad.push( `${where}: triangles outside their look` );

		}

		if ( ids.size !== known ) bad.push( `${where}: ${ids.size - known} vertex records that the source does not have` );
		if ( want.size !== got.size ) bad.push( `${where}: ${want.size} looks in the source, ${got.size} in the output` );
		for ( const [ key, w ] of want ) {

			const g = got.get( key );
			if ( ! g ) {

				bad.push( `${where}: look missing in the output: ${key.slice( 0, 120 )}` );
				continue;

			}

			if ( g.sig !== w.sig ) bad.push( `${where}: attribute layout differs` );
			const same = ( a, b ) => a.length === b.length && a.every( ( x, i ) => x === b[ i ] );
			const inOrder = same( w.v, g.v ) && same( w.t, g.t );
			if ( inOrder ) info.ordered ++;
			else {

				// the same multiset of vertices and of triangles
				const sv = ( a ) => Uint32Array.from( a ).sort(), st = ( a ) => [ ...a ].sort();
				if ( ! same( sv( w.v ), sv( g.v ) ) ) bad.push( `${where}: vertex set differs (${w.v.length} / ${g.v.length} vertices)` );
				else if ( ! same( st( w.t ), st( g.t ) ) ) bad.push( `${where}: triangle set differs (${w.t.length} / ${g.t.length})` );
				else if ( opt.reorder === 'none' ) bad.push( `${where}: order changed although --reorder none` );
				else info.unordered ++;

			}

		}

	} );
	// one geometry per accessor set in three: the looks of a set must be one
	const looksOf = new Map();
	for ( const m of oj.meshes || [] ) for ( const p of m.primitives ) {

		const set = JSON.stringify( [ p.attributes, p.indices ] ), looks = JSON.stringify( p.extras?.tsq_m ?? null );
		if ( looksOf.has( set ) && looksOf.get( set ) !== looks ) bad.push( 'primitives that share their accessors carry different looks' );
		looksOf.set( set, looks );

	}

	// what three reads besides the data: POSITION bounds
	for ( const m of oj.meshes || [] ) for ( const p of m.primitives ) {

		const a = oj.accessors[ p.attributes.POSITION ];
		if ( ! a || a.componentType !== 5126 || ! a.count ) continue;
		const f = new Float32Array( accessorBytes( Oj, p.attributes.POSITION ).buffer ), n = NUM[ a.type ];
		const mx = new Array( n ).fill( - Infinity ), mn = new Array( n ).fill( Infinity );
		for ( let k = 0; k < f.length; k ++ ) {

			mx[ k % n ] = Math.max( mx[ k % n ], f[ k ] );
			mn[ k % n ] = Math.min( mn[ k % n ], f[ k ] );

		}

		if ( ! eq( a.max, mx ) || ! eq( a.min, mn ) ) bad.push( 'POSITION bounds are not those of the data' );

	}

	if ( ! opt.struct ) ( sj.meshes || [] ).forEach( ( m, mi ) => m.primitives.forEach( ( p, pi ) => {

		const a = sj.accessors[ p.attributes.POSITION ], b = oj.accessors[ oj.meshes[ mi ].primitives[ pi ].attributes.POSITION ];
		if ( a && a.min && ( ! eq( a.min, b.min ) || ! eq( a.max, b.max ) ) ) bad.push( `mesh ${mi} primitive ${pi}: POSITION bounds differ from the source's` );

	} ) );
	return Object.assign( bad, { info } );

}

// ------------------------------------------------------------------------------------------ files

const sleep = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );
const sha16 = ( b ) => crypto.createHash( 'sha256' ).update( b ).digest( 'hex' ).slice( 0, 16 );
const gz = ( b ) => zlib.gzipSync( b, { level: 6 } ).length; // what tools/serve.py sends
const readJSON = ( p, d ) => {

	try {

		return JSON.parse( fs.readFileSync( p, 'utf8' ) );

	} catch ( e ) {

		return d;

	}

};

const writeAtomic = ( p, data ) => {

	fs.mkdirSync( path.dirname( p ), { recursive: true } );
	fs.writeFileSync( p + '.part', data );
	fs.renameSync( p + '.part', p );

};

// a source file that is not being written: the same size and mtime before and after the read, and a whole GLB
async function readStable( p ) {

	let why = 'missing';
	for ( let i = 0; i < 4; i ++ ) {

		if ( i ) await sleep( 400 * i );
		try {

			const a = fs.statSync( p ), buf = fs.readFileSync( p ), b = fs.statSync( p );
			if ( a.size !== b.size || a.mtimeMs !== b.mtimeMs || buf.length !== b.size ) {

				why = 'changing while read';
				continue;

			}

			if ( buf.length < 20 || buf.readUInt32LE( 8 ) !== buf.length ) {

				why = 'incomplete';
				continue;

			}

			return { buf, size: b.size, mtime: Math.floor( b.mtimeMs / 1000 ), mtimeMs: b.mtimeMs };

		} catch ( e ) {

			why = e.code || e.message;

		}

	}

	return { error: why };

}

// the files of the town: [ { key: '<asset>/<name>', coarse } ]  (coarse: wanted before the first frame)
function townFiles() {

	const town = JSON.parse( fs.readFileSync( TOWN_JSON, 'utf8' ) ), list = [];
	for ( const t of town.terrain || [] ) list.push( { key: t.file, coarse: true } );
	for ( const [ key, p ] of Object.entries( town.protos || {} ) ) {

		const n = Math.max( 1, ( p.tris || [] ).length );
		for ( let k = 0; k < n; k ++ ) list.push( { key: key + ( k ? '.lod' + k : '' ), coarse: k === n - 1 } );

	}

	return list.sort( ( a, b ) => ( a.key < b.key ? - 1 : a.key > b.key ? 1 : 0 ) );

}

const variants = () => {

	if ( ! A.all ) return [ A.name || variantOf( OPT ) ];
	try {

		return fs.readdirSync( REL, { withFileTypes: true } ).filter( ( d ) => d.isDirectory() && fs.existsSync( path.join( REL, d.name, 'manifest.json' ) ) ).map( ( d ) => d.name ).sort();

	} catch ( e ) {

		return [];

	}

};

const mb = ( n ) => ( n / 1e6 ).toFixed( 1 );
const levelOf = ( key ) => ( /\.lod(\d)$/.exec( key ) || [ 0, 0 ] )[ 1 ] | 0;

// the page's view of a manifest (web/engine/glbrelease.js): files: key -> [ source mtime, source size, size, hash, offset in the bundle or - 1 ]
const indexOf = ( man ) => ( { format: 1, tool: man.tool, town: man.town, variant: man.variant, opt: man.opt, materials: man.materials, bundle: man.bundles[ 0 ] || null,
	files: Object.fromEntries( Object.entries( man.files ).map( ( [ k, e ] ) => [ k, [ e.v, e.n, e.o, e.oh, e.b ?? - 1 ] ] ) ) } );

async function build( name, OPT, bundle ) {

	const dir = path.join( REL, name ), mpath = path.join( dir, 'manifest.json' );
	const idn = JSON.stringify( { tool: TOOL, ...OPT } );
	const old = readJSON( mpath, null ), reuse = old && JSON.stringify( { tool: old.tool, ...old.opt } ) === idn && ! A.force;
	const files = townFiles(), out = { tool: TOOL, town: A.town, variant: name, opt: OPT, materials: {}, bundles: [], files: {} };
	const skipped = [], t0 = Date.now(), tally = { made: 0, kept: 0, copied: 0 }, sum = { prims: 0, prims_out: 0, kept_order: 0, unused_vertices: 0 };
	for ( const f of files ) {

		const src = path.join( GLB, f.key + '.glb' ), dst = path.join( dir, f.key + '.glb' );
		const was = reuse ? old.files[ f.key ] : null;
		let st = null;
		try {

			st = fs.statSync( src );

		} catch ( e ) { /* reported below */ }

		// unchanged since the last run (size + mtime, the hash when only the mtime moved) and its output is still there
		const outOk = ( e ) => {

			try {

				return fs.statSync( dst ).size === e.o;

			} catch ( x ) {

				return false;

			}

		};

		let e = null, R = null;
		if ( was && st && was.n === st.size && was.t === st.mtimeMs && outOk( was ) ) e = was;
		else {

			R = await readStable( src );
			if ( R.error ) {

				skipped.push( `${f.key}: ${R.error}` );
				continue;

			}

			if ( was && was.h === sha16( R.buf ) && outOk( was ) ) e = { ...was, v: R.mtime, t: R.mtimeMs };

		}

		if ( e ) {

			tally.kept ++;
			for ( const id of e.m || [] ) out.materials[ id ] = old.materials[ id ];

		} else {

			e = { v: R.mtime, t: R.mtimeMs, n: R.size, h: sha16( R.buf ), sz: gz( R.buf ) };
			let res = null;
			try {

				res = optimise( readModel( parseGlb( R.buf ) ), OPT );
				const diff = compare( R.buf, res.glb, res.looks, OPT ); // nothing is written that does not decode back to its source
				if ( diff.length ) throw new Error( 'round trip: ' + diff.slice( 0, 3 ).join( '; ' ) );

			} catch ( x ) {

				if ( ! ( x instanceof Unsupported ) ) {

					skipped.push( `${f.key}: ${x.message}` );
					continue;

				}

				e.pass = x.message; // a structure this tool does not rewrite: delivered as built
				res = null;

			}

			const data = res ? res.glb : R.buf;
			writeAtomic( dst, data );
			e.o = data.length;
			e.z = gz( data );
			e.oh = sha16( data ).slice( 0, 10 );
			if ( res ) {

				e.m = Object.keys( res.looks ).sort();
				e.s = [ res.stats.prims, res.stats.prims_out, res.stats.kept_order, res.stats.unused_vertices ];
				Object.assign( out.materials, res.looks );
				tally.made ++;

			} else tally.copied ++;
			if ( ! A.quiet && ( tally.made + tally.copied ) % 200 === 0 ) console.log( `  ${tally.made + tally.copied} files written, ${( ( Date.now() - t0 ) / 1000 ).toFixed( 0 )} s` );

		}

		delete e.b;
		out.files[ f.key ] = e;
		if ( e.s ) [ 'prims', 'prims_out', 'kept_order', 'unused_vertices' ].forEach( ( k, i ) => ( sum[ k ] += e.s[ i ] ) );

	}

	out.materials = canon( out.materials );

	// the bundle: the coarse files end to end (a file that was skipped is simply not in it: the page asks for it alone)
	const stale = new Set( fs.existsSync( dir ) ? fs.readdirSync( dir ).filter( ( n ) => /^coarse\.[0-9a-f]+\.bin(\.part)?$/.test( n ) ) : [] );
	if ( bundle ) {

		const parts = [];
		let off = 0;
		for ( const f of files ) {

			const e = out.files[ f.key ];
			if ( ! f.coarse || ! e ) continue;
			const b = fs.readFileSync( path.join( dir, f.key + '.glb' ) );
			if ( b.length !== e.o ) throw new Error( `${f.key}: output changed under the bundle` );
			e.b = off;
			parts.push( b );
			off += b.length;

		}

		const all = Buffer.concat( parts ), bn = `coarse.${sha16( all )}.bin`;
		if ( ! stale.delete( bn ) ) writeAtomic( path.join( dir, bn ), all );
		out.bundles.push( { file: bn, bytes: all.length, files: parts.length } );

	}

	fs.mkdirSync( dir, { recursive: true } );
	const index = JSON.stringify( indexOf( out ) );
	writeAtomic( path.join( dir, 'index.json' ), index );
	writeAtomic( mpath, JSON.stringify( out ) );
	// the variant the page takes when it is not told: kept current once it has been chosen with --default
	const rel = path.join( REL, 'release.json' );
	if ( A.default || readJSON( rel, {} ).variant === name ) writeAtomic( rel, index );

	// tidy: bundles of earlier runs and outputs of files that left the town (only ever inside this variant's folder)
	let removed = 0;
	for ( const n of stale ) {

		fs.unlinkSync( path.join( dir, n ) );
		removed ++;

	}

	for ( const d of fs.readdirSync( dir, { withFileTypes: true } ) ) {

		if ( ! d.isDirectory() ) continue;
		for ( const n of fs.readdirSync( path.join( dir, d.name ) ) ) {

			if ( n.endsWith( '.glb' ) && out.files[ d.name + '/' + n.slice( 0, - 4 ) ] ) continue;
			if ( ! /\.glb(\.part)?$/.test( n ) ) continue;
			fs.unlinkSync( path.join( dir, d.name, n ) );
			removed ++;

		}

		if ( ! fs.readdirSync( path.join( dir, d.name ) ).length ) fs.rmdirSync( path.join( dir, d.name ) );

	}

	console.log( `glbopt ${A.town}/${name}: ${Object.keys( out.files ).length} of ${files.length} files (${tally.made} optimised, ${tally.kept} unchanged, ${tally.copied} copied as built), ` +
		`${removed} removed, ${Object.keys( out.materials ).length} looks, primitives ${sum.prims} -> ${sum.prims_out}` +
		( OPT.reorder !== 'none' ? `, order kept in ${sum.kept_order}` : '' ) + ( sum.unused_vertices ? `, ${sum.unused_vertices} unused vertices kept` : '' ) +
		`, ${( ( Date.now() - t0 ) / 1000 ).toFixed( 1 )} s` );
	if ( skipped.length ) {

		console.log( `SKIPPED ${skipped.length} (the page reads these from out/glb; run again when they are whole):` );
		for ( const s of skipped.slice( 0, 20 ) ) console.log( '  ' + s );

	}

	if ( ! A.quiet ) report( out );
	return skipped.length ? 2 : 0;

}

function report( man ) {

	const zero = () => ( { n: 0, src: 0, sgz: 0, out: 0, ogz: 0 } );
	const L = [ zero(), zero(), zero() ], C = zero(), all = zero();
	for ( const [ key, e ] of Object.entries( man.files ) ) {

		for ( const r of [ L[ levelOf( key ) ], all, ...( e.b !== undefined ? [ C ] : [] ) ] ) {

			r.n ++; r.src += e.n; r.sgz += e.sz; r.out += e.o; r.ogz += e.z;

		}

	}

	const row = ( n, r ) => console.log( `  ${n.padEnd( 14 )} ${String( r.n ).padStart( 5 )} files   source ${mb( r.src ).padStart( 6 )} MB (gzip ${mb( r.sgz ).padStart( 6 )})   ${man.variant.padEnd( 4 )} ${mb( r.out ).padStart( 6 )} MB (gzip ${mb( r.ogz ).padStart( 6 )})` );
	L.forEach( ( r, k ) => row( 'level ' + k, r ) );
	row( 'all levels', all );
	if ( man.bundles.length ) {

		row( 'coarse set', C );
		const b = fs.readFileSync( path.join( REL, man.variant, man.bundles[ 0 ].file ) );
		console.log( `  bundle ${man.bundles[ 0 ].file}: ${mb( b.length )} MB, gzip ${mb( gz( b ) )} MB in one request` );

	}

	const pass = Object.entries( man.files ).filter( ( e ) => e[ 1 ].pass );
	if ( pass.length ) console.log( `  ${pass.length} copied as built: ${[ ...new Set( pass.map( ( e ) => e[ 1 ].pass ) ) ].slice( 0, 5 ).join( ', ' )}` );

}

async function verify( name ) {

	const dir = path.join( REL, name ), man = readJSON( path.join( dir, 'manifest.json' ), null );
	if ( ! man ) {

		console.log( `verify ${A.town}/${name}: no manifest` );
		return 1;

	}

	const t0 = Date.now(), bad = [], stale = [], T = { ok: 0, pass: 0, ordered: 0, unordered: 0, vertices: 0, triangles: 0, bundled: 0 };
	const bundle = man.bundles.length ? fs.readFileSync( path.join( dir, man.bundles[ 0 ].file ) ) : null;
	if ( JSON.stringify( readJSON( path.join( dir, 'index.json' ), null ) ) !== JSON.stringify( indexOf( man ) ) ) bad.push( 'index.json is not the manifest\'s' );
	const rel = readJSON( path.join( REL, 'release.json' ), null );
	if ( rel && rel.variant === name && JSON.stringify( rel ) !== JSON.stringify( indexOf( man ) ) ) bad.push( 'release.json is not this variant\'s index' );
	for ( const f of townFiles() ) if ( ! man.files[ f.key ] ) stale.push( `${f.key}: not in the release` );
	for ( const [ key, e ] of Object.entries( man.files ) ) {

		const R = await readStable( path.join( GLB, key + '.glb' ) );
		if ( R.error || sha16( R.buf ) !== e.h ) {

			stale.push( `${key}: source ${R.error || 'changed since the release was made'}` );
			continue;

		}

		let o;
		try {

			o = fs.readFileSync( path.join( dir, key + '.glb' ) );

		} catch ( x ) {

			bad.push( `${key}: output missing` );
			continue;

		}

		if ( o.length !== e.o || sha16( o ).slice( 0, 10 ) !== e.oh ) bad.push( `${key}: output is not the one in the manifest` );
		if ( e.b !== undefined ) {

			if ( ! bundle || ! bundle.subarray( e.b, e.b + e.o ).equals( o ) ) bad.push( `${key}: the bundle's copy is not the file` );
			else T.bundled ++;

		}

		if ( e.pass ) {

			if ( ! o.equals( R.buf ) ) bad.push( `${key}: copied file differs from its source` );
			else T.pass ++;
			continue;

		}

		try {

			const d = compare( R.buf, o, man.materials, man.opt );
			if ( d.length ) bad.push( `${key}: ${d.slice( 0, 4 ).join( '; ' )}` );
			else T.ok ++;
			for ( const k of [ 'ordered', 'unordered', 'vertices', 'triangles' ] ) T[ k ] += d.info[ k ];

		} catch ( x ) {

			bad.push( `${key}: ${x.message}` );

		}

	}

	console.log( `verify ${A.town}/${name}: ${T.ok} files identical to their source (${T.vertices} vertices, ${T.triangles} triangles; ${T.ordered} primitive looks in source order, ${T.unordered} as the same set in another order)` +
		`, ${T.pass} byte copies, ${T.bundled} bundle slices equal to their files, ${bad.length} DIFFERENT, ${stale.length} stale, ${( ( Date.now() - t0 ) / 1000 ).toFixed( 0 )} s` );
	for ( const b of bad.slice( 0, 30 ) ) console.log( '  DIFFERENT ' + b );
	for ( const s of stale.slice( 0, 10 ) ) console.log( '  stale ' + s );
	return bad.length ? 1 : stale.length ? 2 : 0;

}

const codes = [];
if ( A.verify || A.stats || A.all ) {

	// --all: every variant that is there (build: each again with the options it was made with)
	for ( const name of variants() ) {

		const man = readJSON( path.join( REL, name, 'manifest.json' ), null );
		if ( A.verify ) codes.push( await verify( name ) );
		else if ( ! man ) continue;
		else if ( A.stats ) {

			console.log( `${A.town}/${name}  ${JSON.stringify( man.opt )}  ${Object.keys( man.materials ).length} looks` );
			report( man );

		} else codes.push( await build( name, man.opt, man.bundles.length > 0 ) );

	}

} else codes.push( await build( A.name || variantOf( OPT ), OPT, A.bundle !== '0' ) );
process.exit( codes.includes( 1 ) ? 1 : codes.includes( 2 ) ? 2 : 0 ); // 1: an output differs from its source, 2: files skipped / stale
