#!/usr/bin/env node
// masterglb.mjs - the baked town (tools/webbake.py -> out/web/_next) as ONE .glb that the walk page web/master/ loads
// with three.js's GLTFLoader, levels of detail and textures by HTTP range requests, and nothing beside it.
//
//   node tools/masterglb.mjs                      # out/web/_next -> out/web/_master/town_master.glb
//   node tools/masterglb.mjs --bake DIR --out FILE
//
// WHAT IS IN THE FILE
//   Geometry: every cell of the bake with its MSFT_lod chain (fine, near, mid, lo - the High tier's levels), written by
//   tools/bake2glb.mjs in its exact mode: the bake's own u16 vertex records and the packer's meshopt streams, byte for
//   byte (KHR_meshopt_compression, KHR_mesh_quantization). This tool runs that export, then lays the binary chunk out
//   again and adds what the page needs so that the file stands alone:
//     - the bake's pictures as glTF images (EXT_texture_webp; the ground occlusion map is a PNG): the atlas pages of
//       the first frame, the single colour layers, the normal + roughness layers, the two ground maps
//     - the visibility sets (pvs.json, as a plain bufferView)
//     - scenes[0].extras: `baked` (the manifest's globals with lanterns, halos, flames, signs, navigation and
//       collision data, the sky's light), `export` (bake2glb's record, plus `tiers`: the High and Medium rows of
//       BAKED_TIERS in web/engine/lite.js) and `master` (below)
//
// THE ORDER OF THE BINARY CHUNK (a reader that takes ranges gets a picture from the front of the file)
//   lo      the lowest level of all cells          \
//   pages   the atlas pages (512 px layers)         > the first frame
//   pvs     the visibility sets                     |
//   ground  ground occlusion, ground shadows       /
//   mid, near, fine                                 the levels a walk asks for, cell by cell
//   l1024, n1024, l2048, n512                       single layers, in the order High asks for them; then Medium's
//   Not in the file: the 256 px pages, the phone's `ring` level and the lean bake (Low / Phone).
//
// scenes[0].extras.master = { version: 1, baked: <the bake's stamp>, tool: 'masterglb',
//   files: { '<file name in the bake>': { image: <index> } | { bufferView: <index> } },   // 'tex512_0.webp', 'gao.png', 'pvs.json'
//   layout: [ { what, byteOffset, byteLength } ] }                                         // the table above, offsets in the chunk's data
// The page asks for a picture by the name the bake gave it, so its texture code reads like the bake's own.
//
// Every stream is compared with the export's and every embedded file with the bake's after writing, and
// `bake2glb.mjs --check` then reads the finished file back against the decoded bake (vertex and index bytes, draw
// ranges, placement). Any difference fails the run.

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname( fileURLToPath( import.meta.url ) );
const ROOT = path.resolve( HERE, '..' );
const BAKE2GLB = path.join( HERE, 'bake2glb.mjs' );
const TIERS_FROM = path.join( ROOT, 'web/engine/lite.js' );
const GLB = 0x46546C67, JSON_CHUNK = 0x4E4F534A, BIN_CHUNK = 0x004E4942;
const MIME = { '.webp': 'image/webp', '.png': 'image/png', '.jpg': 'image/jpeg' };

function die( msg ) { console.error( 'masterglb: ' + msg ); process.exit( 1 ); }

function parseArgs( argv ) {

	const o = { bake: path.join( ROOT, 'out/web/_next' ), out: path.join( ROOT, 'out/web/_master/town_master.glb' ) };
	for ( let i = 0; i < argv.length; i ++ ) {

		if ( ( argv[ i ] === '--bake' || argv[ i ] === '--out' ) && i + 1 < argv.length ) o[ argv[ i ].slice( 2 ) ] = path.resolve( argv[ ++ i ] );
		else die( `unknown argument ${argv[ i ]} (usage: node tools/masterglb.mjs [--bake DIR] [--out FILE])` );

	}

	return o;

}

// BAKED_TIERS of the town page: the object literal is cut out of the source and evaluated (plain data), as
// bake2glb.mjs does - the page's numbers are written down in one place only
function readTiers( file ) {

	const src = fs.readFileSync( file, 'utf8' );
	const at = src.indexOf( 'BAKED_TIERS =' );
	if ( at < 0 ) die( `no BAKED_TIERS in ${file}` );
	const a = src.indexOf( '{', at );
	let depth = 0, b = a;
	for ( ; b < src.length; b ++ ) { if ( src[ b ] === '{' ) depth ++; else if ( src[ b ] === '}' && -- depth === 0 ) break; }
	return Function( '"use strict"; return (' + src.slice( a, b + 1 ) + ');' )();

}

function readGlb( file ) {

	const raw = fs.readFileSync( file );
	if ( raw.length < 28 || raw.readUInt32LE( 0 ) !== GLB || raw.readUInt32LE( 4 ) !== 2 ) die( `${file}: not a glTF 2.0 binary file` );
	if ( raw.readUInt32LE( 8 ) !== raw.length ) die( `${file}: the header says ${raw.readUInt32LE( 8 )} bytes, the file has ${raw.length}` );
	const jn = raw.readUInt32LE( 12 );
	if ( raw.readUInt32LE( 16 ) !== JSON_CHUNK ) die( `${file}: the first chunk is not JSON` );
	const bo = 20 + jn, bn = raw.readUInt32LE( bo );
	if ( raw.readUInt32LE( bo + 4 ) !== BIN_CHUNK ) die( `${file}: the second chunk is not BIN` );
	if ( bo + 8 + bn !== raw.length ) die( `${file}: chunk lengths do not add up to the file's` );
	return { json: JSON.parse( raw.subarray( 20, 20 + jn ).toString( 'utf8' ) ), jsonBytes: jn, bin: raw.subarray( bo + 8, bo + 8 + bn ) };

}

function writeGlb( file, json, parts, length ) {

	let js = Buffer.from( JSON.stringify( json ), 'utf8' );
	if ( js.length % 4 ) js = Buffer.concat( [ js, Buffer.alloc( 4 - js.length % 4, 0x20 ) ] );
	const pad = ( 4 - length % 4 ) % 4, total = 12 + 8 + js.length + 8 + length + pad;
	if ( total >= 2 ** 32 ) die( `the file would have ${total} bytes: a .glb ends at 4 GB` );
	const head = Buffer.alloc( 20 ), bh = Buffer.alloc( 8 );
	head.writeUInt32LE( GLB, 0 ); head.writeUInt32LE( 2, 4 ); head.writeUInt32LE( total, 8 );
	head.writeUInt32LE( js.length, 12 ); head.writeUInt32LE( JSON_CHUNK, 16 );
	bh.writeUInt32LE( length + pad, 0 ); bh.writeUInt32LE( BIN_CHUNK, 4 );
	const fd = fs.openSync( file, 'w' );
	for ( const p of [ head, js, bh, ...parts, Buffer.alloc( pad ) ] ) { let o = 0; while ( o < p.byteLength ) o += fs.writeSync( fd, p, o, p.byteLength - o ); }
	fs.closeSync( fd );

}

const meshopt = ( view ) => view.extensions && ( view.extensions.KHR_meshopt_compression || view.extensions.EXT_meshopt_compression );

// bufferView -> the level it belongs to, through the nodes that carry the bake's level entry (nothing is taken from
// the order bake2glb happened to write)
function levelOfViews( json ) {

	const of = new Map();
	for ( const n of json.nodes ) {

		const level = n.extras && n.extras.baked && n.extras.baked.level;
		if ( ! level || n.mesh === undefined ) continue;
		for ( const p of json.meshes[ n.mesh ].primitives ) for ( const a of [ ...Object.values( p.attributes ), p.indices ] ) {

			const v = json.accessors[ a ].bufferView;
			if ( of.has( v ) && of.get( v ) !== level ) die( `bufferView ${v} is shared by the levels ${of.get( v )} and ${level}` );
			of.set( v, level );

		}

	}

	return of;

}

// ------------------------------------------------------------------------------------------------ main

const o = parseArgs( process.argv.slice( 2 ) );
const manFile = path.join( o.bake, 'manifest.json' );
if ( ! fs.existsSync( manFile ) ) die( `no manifest.json in ${o.bake}` );
const man = JSON.parse( fs.readFileSync( manFile, 'utf8' ) );
const tiers = readTiers( TIERS_FROM ), H = tiers.high, M = tiers.medium;
if ( ! H || ! M ) die( `${TIERS_FROM}: BAKED_TIERS has no high / medium row` );

// 1. the geometry: bake2glb's exact export of the whole town, lowest level first (checked against the bake at the end,
// on the finished file, so not here)
const tmp = path.join( path.dirname( o.out ), '.tmp' );
fs.rmSync( tmp, { recursive: true, force: true } );
fs.mkdirSync( tmp, { recursive: true } );
const run = ( args ) => { const r = spawnSync( process.execPath, [ BAKE2GLB, o.bake, ...args ], { stdio: 'inherit' } ); if ( r.status !== 0 ) die( `bake2glb.mjs ${args.join( ' ' )} failed` ); };
run( [ '--mode', 'exact', '--lods', 'auto', '--tier', 'high', '--cells', 'all', '--order', 'coarse', '--extras', 'full', '--out', tmp, '--name', 'exact' ] );
const src = readGlb( path.join( tmp, 'exact.glb' ) ), json = src.json;
const extras = json.scenes[ 0 ].extras;
if ( extras.baked.baked !== man.baked ) die( 'the bake changed while it was exported: run again' );
if ( json.buffers[ 0 ].uri || json.images || json.textures ) die( 'bake2glb\'s export is not what this tool expects (one buffer in the file, no images)' );

// 2. the binary chunk again, in the order of the header
const parts = [], layout = [], files = {}, streams = [], embedded = [];
let length = 0;
const put = ( bytes, align ) => {

	const pad = ( align - length % align ) % align;
	if ( pad ) { parts.push( Buffer.alloc( pad ) ); length += pad; }
	parts.push( bytes ); length += bytes.byteLength;
	return length - bytes.byteLength;

};
const section = ( what, fill ) => { put( Buffer.alloc( 0 ), 4 ); const at = length; fill(); layout.push( { what, byteOffset: at, byteLength: length - at } ); };

const levelOf = levelOfViews( json );
const geometry = ( level ) => section( level, () => {

	// (compressed streams need no alignment; they stay packed as bake2glb packed them)
	const views = json.bufferViews.map( ( v, i ) => ( { i, x: meshopt( v ) } ) ).filter( ( { i, x } ) => x && levelOf.get( i ) === level ).sort( ( a, b ) => a.x.byteOffset - b.x.byteOffset );
	for ( const { i, x } of views ) {

		if ( x.buffer !== 0 ) die( `bufferView ${i}: its stream is not in the file's own buffer` );
		const from = x.byteOffset || 0;
		x.byteOffset = put( src.bin.subarray( from, from + x.byteLength ), 1 );
		streams.push( { view: i, from, at: x.byteOffset, length: x.byteLength } );

	}

} );
const embed = ( name ) => {

	if ( files[ name ] ) return; // (a file two tiers read is in once)
	const file = path.join( o.bake, name );
	if ( ! fs.existsSync( file ) ) die( `the manifest names ${name}, the bake does not have it` );
	const bytes = fs.readFileSync( file ), mimeType = MIME[ path.extname( name ) ];
	const bufferView = json.bufferViews.push( { buffer: 0, byteOffset: put( bytes, 4 ), byteLength: bytes.byteLength } ) - 1;
	embedded.push( { name, bufferView } );
	if ( ! mimeType ) { files[ name ] = { bufferView }; return; }
	const image = ( json.images ||= [] ).push( { name, bufferView, mimeType } ) - 1;
	( json.textures ||= [] ).push( mimeType === 'image/webp' ? { name, extensions: { EXT_texture_webp: { source: image } } } : { name, source: image } );
	files[ name ] = { image };

};
// a table of the manifest by size ('l': single colour layers, 'n': normal + roughness layers): the files it names
const singles = ( kind, size ) => {

	const t = ( kind === 'l' ? man.texl : man.texn || {} )[ size ];
	if ( ! t ) die( `the bake has no ${kind}${size} layers (the tier table asks for them)` );
	return t.files.filter( ( f ) => f ).map( ( f ) => f + '.webp' );

};

const chain = extras.export.levels, lowest = chain[ chain.length - 1 ];
for ( const level of new Set( levelOf.values() ) ) if ( ! chain.includes( level ) ) die( `a level ${level} the export does not list` );
geometry( lowest );
section( 'pages', () => {

	for ( const size of new Set( [ H.tex, M.tex ] ) ) for ( const p of [ ...man.tex[ size ].pages, ...( man.tex[ size ].hpages || [] ) ] ) embed( p + '.webp' );

} );
if ( man.pvs ) section( 'pvs', () => embed( man.pvs.file ) );
section( 'ground', () => { for ( const g of [ man.gao, man.gsh ] ) if ( g ) embed( g.file ); } );
for ( const level of chain.slice( 0, - 1 ).reverse() ) geometry( level );
// single layers: High's sharp colour, its normals, its large-tiled layers at twice the size; then what only Medium reads
const wanted = [ [ 'l', H.sharp ], [ 'n', H.nr ], [ 'l', H.big && H.big[ 0 ] ], [ 'l', M.sharp ], [ 'l', M.big && M.big[ 0 ] ], [ 'n', M.nr ] ];
for ( const [ kind, size ] of wanted ) if ( size && ! layout.some( ( l ) => l.what === kind + size ) ) section( kind + size, () => singles( kind, size ).forEach( embed ) );

// 3. the JSON: the buffer's new length, the extensions the pictures use, what the page reads
json.buffers[ 0 ].byteLength = length;
if ( ( json.textures || [] ).some( ( t ) => t.extensions ) ) {

	// (required: these textures have no PNG / JPEG to fall back to, which is what the extension asks to be declared)
	json.extensionsUsed = [ ...new Set( [ ...json.extensionsUsed, 'EXT_texture_webp' ] ) ].sort();
	json.extensionsRequired = [ ...new Set( [ ...json.extensionsRequired, 'EXT_texture_webp' ] ) ].sort();

}

json.asset.generator = 'masterglb (threejsodango), geometry by ' + json.asset.generator;
extras.export.tiers = { high: H, medium: M };
extras.master = { version: 1, baked: man.baked, tool: 'masterglb', files, layout };

fs.mkdirSync( path.dirname( o.out ), { recursive: true } );
writeGlb( o.out, json, parts, length );

// 4. the finished file, read back
const fail = ( msg ) => { console.error( 'masterglb: CHECK FAILED: ' + msg ); process.exit( 2 ); };
const res = readGlb( o.out ), j = res.json;
if ( res.bin.length - j.buffers[ 0 ].byteLength < 0 || res.bin.length - j.buffers[ 0 ].byteLength > 3 ) fail( 'the BIN chunk is not buffer 0 (padded to four bytes)' );
j.bufferViews.forEach( ( v, i ) => {

	if ( ! j.buffers[ v.buffer ] || ( v.byteOffset || 0 ) + v.byteLength > j.buffers[ v.buffer ].byteLength ) fail( `bufferView ${i} reaches outside its buffer` );
	const x = meshopt( v );
	if ( x && ( x.byteOffset || 0 ) + x.byteLength > j.buffers[ x.buffer ].byteLength ) fail( `the stream of bufferView ${i} reaches outside its buffer` );

} );
for ( const s of streams ) {

	const x = meshopt( j.bufferViews[ s.view ] );
	if ( x.byteOffset !== s.at || x.byteLength !== s.length || Buffer.compare( res.bin.subarray( s.at, s.at + s.length ), src.bin.subarray( s.from, s.from + s.length ) ) !== 0 ) fail( `the stream of bufferView ${s.view} is not the export's` );

}

if ( streams.length !== j.bufferViews.filter( meshopt ).length ) fail( 'a compressed stream belongs to no level' );
for ( const e of embedded ) {

	const v = j.bufferViews[ e.bufferView ];
	if ( v.byteOffset % 4 || v.byteStride !== undefined || v.target !== undefined ) fail( `${e.name}: its bufferView is not a plain, aligned one` );
	if ( Buffer.compare( res.bin.subarray( v.byteOffset, v.byteOffset + v.byteLength ), fs.readFileSync( path.join( o.bake, e.name ) ) ) !== 0 ) fail( `${e.name} is not the bake's file` );
	const f = j.scenes[ 0 ].extras.master.files[ e.name ];
	if ( f.image === undefined ? f.bufferView !== e.bufferView : j.images[ f.image ].bufferView !== e.bufferView || j.images[ f.image ].name !== e.name ) fail( `${e.name}: master.files does not lead to it` );

}

for ( const l of j.scenes[ 0 ].extras.master.layout ) if ( l.byteOffset % 4 || l.byteOffset + l.byteLength > j.buffers[ 0 ].byteLength ) fail( `layout entry ${l.what} is off` );
if ( JSON.parse( fs.readFileSync( manFile, 'utf8' ) ).baked !== man.baked ) fail( 'the bake changed while the file was written: run again' );
run( [ '--check', o.out ] ); // every level decoded and compared with the decoded bake
fs.rmSync( tmp, { recursive: true, force: true } );

// 5. what was written
const total = fs.statSync( o.out ).size, mb = ( n ) => ( n / 1e6 ).toFixed( 2 ).padStart( 8 ) + ' MB', share = ( n ) => ( 100 * n / total ).toFixed( 1 ).padStart( 5 ) + ' %';
const levels = j.nodes.filter( ( n ) => n.mesh !== undefined && n.extras && n.extras.baked );
const tris = ( level ) => Math.round( levels.filter( ( n ) => ! level || n.extras.baked.level === level ).reduce( ( s, n ) => s + n.extras.baked.ni / 3, 0 ) );
console.log( `${path.relative( ROOT, o.out )}: ${total} bytes, bake ${man.baked}` );
console.log( `  ${'JSON'.padEnd( 7 )}${mb( res.jsonBytes )} ${share( res.jsonBytes )}` );
for ( const l of layout ) console.log( `  ${l.what.padEnd( 7 )}${mb( l.byteLength )} ${share( l.byteLength )}` + ( chain.includes( l.what ) ? `   ${tris( l.what )} triangles, ${( l.byteLength / tris( l.what ) ).toFixed( 1 )} bytes each` : '' ) );
console.log( `  ${j.nodes.filter( ( n ) => n.extras && n.extras.cell ).length} cells, ${levels.length} levels (${j.nodes.filter( ( n ) => n.extensions && n.extensions.MSFT_lod ).length} chains), ${tris()} triangles; ${( j.images || [] ).length} images, ${Object.keys( files ).length} files by name` );
console.log( `  checked: ${streams.length} streams identical to the export's, ${embedded.length} files identical to the bake's, every level decoded against the bake` );
