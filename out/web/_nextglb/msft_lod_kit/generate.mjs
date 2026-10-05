#!/usr/bin/env node
// generate.mjs - writes the synthetic MSFT_lod test models of this kit and expected.json. Node, no dependencies.
//
//   node generate.mjs            # (re)writes models/*.gltf|.bin|.glb and expected.json beside this file
//
// The models are plain glTF 2.0: float positions and normals, no extension but MSFT_lod. Every level has a colour
// of its own: level 0 red, level 1 green, level 2 blue, level 3 yellow (and white for a child node), so that a
// glance at the screen says which level is drawn.
// expected.json is derived from the JSON of every model listed in MODELS below - also of the ones this script does
// not write (the town cells: see README.md) - by the rules a loader has to follow, not by running a loader.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname( fileURLToPath( import.meta.url ) );
const OUT = path.join( HERE, 'models' );
const FOV = 50, ASPECT = 16 / 9; // the reference camera of the coverage -> distance mapping (README.md)

// ------------------------------------------------------------------------------------------------ geometry

function icosphere( subdivisions, radius = 1 ) {

	const t = ( 1 + Math.sqrt( 5 ) ) / 2;
	let v = [ [ - 1, t, 0 ], [ 1, t, 0 ], [ - 1, - t, 0 ], [ 1, - t, 0 ], [ 0, - 1, t ], [ 0, 1, t ], [ 0, - 1, - t ], [ 0, 1, - t ], [ t, 0, - 1 ], [ t, 0, 1 ], [ - t, 0, - 1 ], [ - t, 0, 1 ] ];
	let f = [ [ 0, 11, 5 ], [ 0, 5, 1 ], [ 0, 1, 7 ], [ 0, 7, 10 ], [ 0, 10, 11 ], [ 1, 5, 9 ], [ 5, 11, 4 ], [ 11, 10, 2 ], [ 10, 7, 6 ], [ 7, 1, 8 ],
		[ 3, 9, 4 ], [ 3, 4, 2 ], [ 3, 2, 6 ], [ 3, 6, 8 ], [ 3, 8, 9 ], [ 4, 9, 5 ], [ 2, 4, 11 ], [ 6, 2, 10 ], [ 8, 6, 7 ], [ 9, 8, 1 ] ];
	const norm = ( p ) => { const l = Math.hypot( p[ 0 ], p[ 1 ], p[ 2 ] ); return [ p[ 0 ] / l, p[ 1 ] / l, p[ 2 ] / l ]; };
	v = v.map( norm );
	for ( let s = 0; s < subdivisions; s ++ ) {

		const mid = new Map(), nf = [];
		const m = ( a, b ) => {

			const key = a < b ? a * 1e6 + b : b * 1e6 + a;
			if ( ! mid.has( key ) ) { v.push( norm( [ v[ a ][ 0 ] + v[ b ][ 0 ], v[ a ][ 1 ] + v[ b ][ 1 ], v[ a ][ 2 ] + v[ b ][ 2 ] ] ) ); mid.set( key, v.length - 1 ); }
			return mid.get( key );

		};

		for ( const [ a, b, c ] of f ) { const ab = m( a, b ), bc = m( b, c ), ca = m( c, a ); nf.push( [ a, ab, ca ], [ b, bc, ab ], [ c, ca, bc ], [ ab, bc, ca ] ); }
		f = nf;

	}

	return { position: new Float32Array( v.flat().map( ( x ) => x * radius ) ), normal: new Float32Array( v.flat() ), index: new Uint16Array( f.flat() ) };

}

function box( h = 0.8 ) {

	const p = [], n = [], idx = [];
	const faces = [ [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ], [ [ - 1, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ] ], [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ], [ [ 0, - 1, 0 ], [ 1, 0, 0 ], [ 0, 0, 1 ] ], [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ], [ [ 0, 0, - 1 ], [ 0, 1, 0 ], [ 1, 0, 0 ] ] ];
	for ( const [ N, U, V ] of faces ) {

		const b = p.length / 3;
		for ( const [ su, sv ] of [ [ - 1, - 1 ], [ 1, - 1 ], [ 1, 1 ], [ - 1, 1 ] ] ) { p.push( ...[ 0, 1, 2 ].map( ( k ) => ( N[ k ] + su * U[ k ] + sv * V[ k ] ) * h ) ); n.push( ...N ); }
		idx.push( b, b + 1, b + 2, b, b + 2, b + 3 );

	}

	return { position: new Float32Array( p ), normal: new Float32Array( n ), index: new Uint16Array( idx ) };

}

const COLOURS = { red: [ 0.9, 0.08, 0.05 ], green: [ 0.1, 0.75, 0.15 ], blue: [ 0.1, 0.25, 0.95 ], yellow: [ 0.95, 0.85, 0.1 ], white: [ 0.9, 0.9, 0.9 ] };

// ------------------------------------------------------------------------------------------------ a glTF document

class Doc {

	constructor() {

		this.json = { asset: { version: '2.0', generator: 'msft_lod_kit generate.mjs' }, scene: 0, scenes: [ { nodes: [] } ], nodes: [], meshes: [], materials: [], accessors: [], bufferViews: [], buffers: [] };
		this.bufs = [];

	}

	buffer( uri ) { this.bufs.push( { uri, parts: [], length: 0 } ); return this.bufs.length - 1; }

	_view( bi, array, target ) {

		const b = this.bufs[ bi ];
		const pad = ( 4 - b.length % 4 ) % 4;
		if ( pad ) { b.parts.push( Buffer.alloc( pad ) ); b.length += pad; }
		const bytes = Buffer.from( array.buffer, array.byteOffset, array.byteLength );
		this.json.bufferViews.push( { buffer: bi, byteOffset: b.length, byteLength: bytes.length, target } );
		b.parts.push( bytes ); b.length += bytes.length;
		return this.json.bufferViews.length - 1;

	}

	material( name, colour ) {

		const c = COLOURS[ colour ];
		// (emissive as well: the colour shows in a viewer without lights)
		this.json.materials.push( { name, pbrMetallicRoughness: { baseColorFactor: [ ...c, 1 ], metallicFactor: 0, roughnessFactor: 0.9 }, emissiveFactor: c.map( ( x ) => + ( x * 0.45 ).toFixed( 4 ) ) } );
		return this.json.materials.length - 1;

	}

	mesh( name, geo, material, bi ) {

		const mm = ( a ) => { const lo = [ Infinity, Infinity, Infinity ], hi = [ - Infinity, - Infinity, - Infinity ]; for ( let i = 0; i < a.length; i ++ ) { lo[ i % 3 ] = Math.min( lo[ i % 3 ], a[ i ] ); hi[ i % 3 ] = Math.max( hi[ i % 3 ], a[ i ] ); } return { min: lo, max: hi }; };
		const acc = ( def ) => { this.json.accessors.push( def ); return this.json.accessors.length - 1; };
		const P = acc( { bufferView: this._view( bi, geo.position, 34962 ), componentType: 5126, count: geo.position.length / 3, type: 'VEC3', ...mm( geo.position ) } );
		const N = acc( { bufferView: this._view( bi, geo.normal, 34962 ), componentType: 5126, count: geo.normal.length / 3, type: 'VEC3' } );
		const I = acc( { bufferView: this._view( bi, geo.index, 34963 ), componentType: 5123, count: geo.index.length, type: 'SCALAR' } );
		this.json.meshes.push( { name, primitives: [ { attributes: { POSITION: P, NORMAL: N }, indices: I, material } ] } );
		return this.json.meshes.length - 1;

	}

	node( def, inScene = false ) { this.json.nodes.push( def ); const i = this.json.nodes.length - 1; if ( inScene ) this.json.scenes[ 0 ].nodes.push( i ); return i; }

	write( stem, { glb = false, gltf = true } = {} ) {

		const j = this.json;
		const out = [];
		if ( gltf ) {

			j.buffers = this.bufs.map( ( b ) => ( { uri: b.uri, byteLength: b.length } ) );
			for ( const b of this.bufs ) { fs.writeFileSync( path.join( OUT, b.uri ), Buffer.concat( b.parts ) ); out.push( b.uri ); }
			fs.writeFileSync( path.join( OUT, stem + '.gltf' ), JSON.stringify( j, null, '\t' ) + '\n' );
			out.push( stem + '.gltf' );

		}

		if ( glb ) {

			if ( this.bufs.length !== 1 ) throw new Error( 'a .glb takes one buffer' );
			const bin0 = Buffer.concat( this.bufs[ 0 ].parts );
			const bin = Buffer.concat( [ bin0, Buffer.alloc( ( 4 - bin0.length % 4 ) % 4 ) ] );
			let js = Buffer.from( JSON.stringify( { ...j, buffers: [ { byteLength: bin0.length } ] } ), 'utf8' );
			js = Buffer.concat( [ js, Buffer.alloc( ( 4 - js.length % 4 ) % 4, 0x20 ) ] );
			const head = Buffer.alloc( 20 ), bh = Buffer.alloc( 8 );
			head.writeUInt32LE( 0x46546C67, 0 ); head.writeUInt32LE( 2, 4 ); head.writeUInt32LE( 28 + js.length + bin.length, 8 ); head.writeUInt32LE( js.length, 12 ); head.writeUInt32LE( 0x4E4F534A, 16 );
			bh.writeUInt32LE( bin.length, 0 ); bh.writeUInt32LE( 0x004E4942, 4 );
			fs.writeFileSync( path.join( OUT, stem + '.glb' ), Buffer.concat( [ head, js, bh, bin ] ) );
			out.push( stem + '.glb' );

		}

		return out;

	}

}

// the usual three levels: a fine sphere, a coarse sphere, a box
function threeLevels( d, prefix, buffers, { fine = 3, order = [ 0, 1, 2 ] } = {} ) {

	const geo = [ icosphere( fine ), icosphere( 1 ), box() ], colour = [ 'red', 'green', 'blue' ], tag = [ 'High', 'Mid', 'Low' ];
	const mesh = [];
	// (the order the data is written in; the node order stays High, Mid, Low)
	for ( const i of order ) mesh[ i ] = d.mesh( `${prefix}_${tag[ i ]}_mesh`, geo[ i ], d.material( `${prefix}_${colour[ i ]}`, colour[ i ] ), buffers[ i ] );
	return mesh;

}

// ------------------------------------------------------------------------------------------------ the models

function writeModels() {

	fs.mkdirSync( OUT, { recursive: true } );
	const written = [];

	{ // (a) + (f): node-level, three levels, MSFT_screencoverage; .gltf + .bin and .glb. One buffer: nothing in it can be left out.

		const d = new Doc(), b = d.buffer( 'a_node_lod.bin' );
		const m = threeLevels( d, 'A', [ b, b, b ] );
		d.node( { name: 'A_High', mesh: m[ 0 ], extensions: { MSFT_lod: { ids: [ 1, 2 ] } }, extras: { MSFT_screencoverage: [ 0.25, 0.05, 0.004 ], tag: 'a-level-0' } }, true );
		d.node( { name: 'A_Mid', mesh: m[ 1 ] } );
		d.node( { name: 'A_Low', mesh: m[ 2 ] } );
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'a_node_lod', { glb: true } ) );

	}

	{ // (b): the same without MSFT_screencoverage

		const d = new Doc(), b = d.buffer( 'b_no_coverage.bin' );
		const m = threeLevels( d, 'B', [ b, b, b ] );
		d.node( { name: 'B_High', mesh: m[ 0 ], extensions: { MSFT_lod: { ids: [ 1, 2 ] } }, extras: { tag: 'b-level-0' } }, true );
		d.node( { name: 'B_Mid', mesh: m[ 1 ] } );
		d.node( { name: 'B_Low', mesh: m[ 2 ] } );
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'b_no_coverage' ) );

	}

	{ // (c): an LOD node with a transform, a parent and a child; a second LOD in the same scene with four levels

		const d = new Doc(), b = d.buffer( 'c_transforms.bin' );
		const mat = ( c ) => d.material( 'C_' + c, c );
		const red = d.mesh( 'C_fine', icosphere( 3 ), mat( 'red' ), b ), green = d.mesh( 'C_coarse', icosphere( 1 ), mat( 'green' ), b );
		const blue = d.mesh( 'C_box', box(), mat( 'blue' ), b ), yellow = d.mesh( 'C_rough', icosphere( 0 ), mat( 'yellow' ), b ), white = d.mesh( 'C_knob', box( 1 ), mat( 'white' ), b );
		const s = Math.sin( Math.PI / 8 ), c = Math.cos( Math.PI / 8 ); // 45 degrees about y
		const trs = { translation: [ 0, 1, 0 ], rotation: [ 0, s, 0, c ], scale: [ 1.5, 1, 1 ] };
		d.node( { name: 'C_Left_Parent', translation: [ - 6, 0, 0 ], scale: [ 2, 2, 2 ], children: [ 1 ] }, true ); // 0
		d.node( { name: 'C_Left_High', mesh: red, ...trs, children: [ 2 ], extensions: { MSFT_lod: { ids: [ 3 ] } }, extras: { MSFT_screencoverage: [ 0.1, 0 ], tag: 'c-left-0' } } ); // 1
		d.node( { name: 'C_Left_Child', mesh: white, translation: [ 0, 1.3, 0 ], scale: [ 0.3, 0.3, 0.3 ] } ); // 2
		d.node( { name: 'C_Left_Low', mesh: green, ...trs } ); // 3
		d.node( { name: 'C_Right_High', mesh: red, translation: [ 6, 0, 0 ], extensions: { MSFT_lod: { ids: [ 5, 6, 7 ] } }, extras: { MSFT_screencoverage: [ 0.3, 0.1, 0.03, 0.002 ], tag: 'c-right-0' } }, true ); // 4
		d.node( { name: 'C_Right_Mid', mesh: green, translation: [ 6, 0, 0 ] } ); // 5
		d.node( { name: 'C_Right_Low', mesh: blue, translation: [ 6, 0, 0 ] } ); // 6
		d.node( { name: 'C_Right_Lowest', mesh: yellow, translation: [ 6, 0, 0 ] } ); // 7
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'c_transforms' ) );

	}

	{ // (d): material-level: one geometry, three materials

		const d = new Doc(), b = d.buffer( 'd_material_lod.bin' );
		const hi = d.material( 'D_Mat_High', 'red' ); d.material( 'D_Mat_Mid', 'green' ); d.material( 'D_Mat_Low', 'blue' );
		d.json.materials[ hi ].extensions = { MSFT_lod: { ids: [ 1, 2 ] } };
		d.json.materials[ hi ].extras = { MSFT_screencoverage: [ 0.25, 0.05, 0 ] };
		d.node( { name: 'D_Sphere', mesh: d.mesh( 'D_mesh', icosphere( 2 ), hi, b ), extras: { tag: 'd-node' } }, true );
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'd_material_lod' ) );

	}

	{ // (e): the last id is a node without a mesh ("nothing at a distance" said with a node, not with coverage)

		const d = new Doc(), b = d.buffer( 'e_empty_lowest.bin' );
		const red = d.mesh( 'E_fine', icosphere( 3 ), d.material( 'E_red', 'red' ), b ), green = d.mesh( 'E_coarse', icosphere( 1 ), d.material( 'E_green', 'green' ), b );
		d.node( { name: 'E_High', mesh: red, extensions: { MSFT_lod: { ids: [ 1, 2 ] } }, extras: { MSFT_screencoverage: [ 0.25, 0.02, 0 ], tag: 'e-level-0' } }, true );
		d.node( { name: 'E_Mid', mesh: green } );
		d.node( { name: 'E_Empty' } );
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'e_empty_lowest' ) );

	}

	{ // (g): the lazy case: a buffer per level, the fine level heavy (20480 triangles)

		const d = new Doc(), bufs = [ d.buffer( 'g_lazy.lod0.bin' ), d.buffer( 'g_lazy.lod1.bin' ), d.buffer( 'g_lazy.lod2.bin' ) ];
		const m = threeLevels( d, 'G', bufs, { fine: 5 } );
		d.node( { name: 'G_High', mesh: m[ 0 ], extensions: { MSFT_lod: { ids: [ 1, 2 ] } }, extras: { MSFT_screencoverage: [ 0.25, 0.05, 0 ], tag: 'g-level-0' } }, true );
		d.node( { name: 'G_Mid', mesh: m[ 1 ] } );
		d.node( { name: 'G_Low', mesh: m[ 2 ] } );
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'g_lazy' ) );

	}

	{ // (i): one .glb for HTTP range requests: the coarse levels at the front of the BIN chunk, the fine one (81920 triangles) behind

		const d = new Doc(), b = d.buffer( 'i_ranged.bin' );
		const m = threeLevels( d, 'I', [ b, b, b ], { fine: 6, order: [ 2, 1, 0 ] } );
		d.node( { name: 'I_High', mesh: m[ 0 ], extensions: { MSFT_lod: { ids: [ 1, 2 ] } }, extras: { MSFT_screencoverage: [ 0.25, 0.05, 0 ], tag: 'i-level-0' } }, true );
		d.node( { name: 'I_Mid', mesh: m[ 1 ] } );
		d.node( { name: 'I_Low', mesh: m[ 2 ] } );
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'i_ranged', { glb: true, gltf: false } ) );

	}

	{ // (k): an animation that moves the LOD's level 0 node

		const d = new Doc(), b = d.buffer( 'k_animated.bin' );
		const red = d.mesh( 'K_fine', icosphere( 3 ), d.material( 'K_red', 'red' ), b ), green = d.mesh( 'K_coarse', icosphere( 1 ), d.material( 'K_green', 'green' ), b );
		d.node( { name: 'K_High', mesh: red, extensions: { MSFT_lod: { ids: [ 1 ] } }, extras: { MSFT_screencoverage: [ 0.1, 0 ], tag: 'k-level-0' } }, true );
		d.node( { name: 'K_Low', mesh: green } );
		const times = d._view( b, new Float32Array( [ 0, 1 ] ) ), values = d._view( b, new Float32Array( [ 0, 0, 0, 0, 2, 0 ] ) );
		d.json.accessors.push( { bufferView: times, componentType: 5126, count: 2, type: 'SCALAR', min: [ 0 ], max: [ 1 ] }, { bufferView: values, componentType: 5126, count: 2, type: 'VEC3' } );
		const n = d.json.accessors.length;
		d.json.animations = [ { name: 'K_Rise', samplers: [ { input: n - 2, output: n - 1, interpolation: 'LINEAR' } ], channels: [ { sampler: 0, target: { node: 0, path: 'translation' } } ] } ];
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'k_animated' ) );

	}

	{ // (n): no MSFT_lod at all: the loader must give what the stock loader gives

		const d = new Doc(), b = d.buffer( 'n_plain.bin' );
		const red = d.mesh( 'N_fine', icosphere( 2 ), d.material( 'N_red', 'red' ), b ), white = d.mesh( 'N_knob', box( 0.3 ), d.material( 'N_white', 'white' ), b );
		d.node( { name: 'N_Root', translation: [ 0, 0.5, 0 ], children: [ 1, 2 ] }, true );
		d.node( { name: 'N_Sphere', mesh: red, extras: { tag: 'n' } } );
		d.node( { name: 'N_Knob', mesh: white, translation: [ 0, 1.3, 0 ] } );
		written.push( ...d.write( 'n_plain' ) );

	}

	{ // (x): outside the spec's promise: the lower levels are ALSO listed in the scene (a stock loader draws all three on top of each other)

		const d = new Doc(), b = d.buffer( 'x_levels_in_scene.bin' );
		const m = threeLevels( d, 'X', [ b, b, b ] );
		d.node( { name: 'X_High', mesh: m[ 0 ], extensions: { MSFT_lod: { ids: [ 1, 2 ] } }, extras: { MSFT_screencoverage: [ 0.25, 0.05, 0 ], tag: 'x-level-0' } }, true );
		d.node( { name: 'X_Mid', mesh: m[ 1 ] }, true );
		d.node( { name: 'X_Low', mesh: m[ 2 ] }, true );
		d.json.extensionsUsed = [ 'MSFT_lod' ];
		written.push( ...d.write( 'x_levels_in_scene' ) );

	}

	return written;

}

// ------------------------------------------------------------------------------------------------ expectations

// 4x4, column-major (as glTF and three.js store them)
const mul = ( a, b ) => { const o = new Array( 16 ).fill( 0 ); for ( let c = 0; c < 4; c ++ ) for ( let r = 0; r < 4; r ++ ) for ( let k = 0; k < 4; k ++ ) o[ c * 4 + r ] += a[ k * 4 + r ] * b[ c * 4 + k ]; return o; };
const IDENT = [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ];
function local( def ) {

	if ( def.matrix ) return def.matrix.slice();
	const [ x, y, z, w ] = def.rotation || [ 0, 0, 0, 1 ], [ sx, sy, sz ] = def.scale || [ 1, 1, 1 ], t = def.translation || [ 0, 0, 0 ];
	return [ ( 1 - 2 * ( y * y + z * z ) ) * sx, 2 * ( x * y + z * w ) * sx, 2 * ( x * z - y * w ) * sx, 0,
		2 * ( x * y - z * w ) * sy, ( 1 - 2 * ( x * x + z * z ) ) * sy, 2 * ( y * z + x * w ) * sy, 0,
		2 * ( x * z + y * w ) * sz, 2 * ( y * z - x * w ) * sz, ( 1 - 2 * ( x * x + y * y ) ) * sz, 0, t[ 0 ], t[ 1 ], t[ 2 ], 1 ];

}

const apply = ( m, p ) => [ m[ 0 ] * p[ 0 ] + m[ 4 ] * p[ 1 ] + m[ 8 ] * p[ 2 ] + m[ 12 ], m[ 1 ] * p[ 0 ] + m[ 5 ] * p[ 1 ] + m[ 9 ] * p[ 2 ] + m[ 13 ], m[ 2 ] * p[ 0 ] + m[ 6 ] * p[ 1 ] + m[ 10 ] * p[ 2 ] + m[ 14 ] ];
const maxScale = ( m ) => Math.sqrt( Math.max( m[ 0 ] ** 2 + m[ 1 ] ** 2 + m[ 2 ] ** 2, m[ 4 ] ** 2 + m[ 5 ] ** 2 + m[ 6 ] ** 2, m[ 8 ] ** 2 + m[ 9 ] ** 2 + m[ 10 ] ** 2 ) );
const COMPONENT_MAX = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };
const r6 = ( x ) => + x.toPrecision( 7 );

export function coverageToDistance( coverage, radius, fov = FOV, aspect = ASPECT ) {

	const t = Math.tan( fov * Math.PI / 360 );
	return radius * Math.sqrt( Math.PI / ( 4 * aspect * t * t * coverage ) );

}

function expectations( file ) {

	const raw = fs.readFileSync( path.join( OUT, file ) );
	const glb = raw.readUInt32LE( 0 ) === 0x46546C67;
	const json = JSON.parse( glb ? raw.subarray( 20, 20 + raw.readUInt32LE( 12 ) ).toString( 'utf8' ) : raw.toString( 'utf8' ) );
	const nodes = json.nodes || [];
	const parent = new Map();
	nodes.forEach( ( n, i ) => { for ( const c of n.children || [] ) parent.set( c, i ); } );
	const world = ( i ) => { let m = IDENT; for ( let p = parent.get( i ); p !== undefined; p = parent.get( p ) ) m = mul( local( nodes[ p ] ), m ); return m; }; // of the parent
	const bounds = ( i, above, box ) => {

		const m = mul( above, local( nodes[ i ] ) ), def = nodes[ i ];
		if ( def.mesh !== undefined ) for ( const p of json.meshes[ def.mesh ].primitives ) {

			const a = json.accessors[ p.attributes.POSITION ], k = a.normalized ? 1 / COMPONENT_MAX[ a.componentType ] : 1;
			for ( let c = 0; c < 8; c ++ ) { const q = apply( m, [ ( c & 1 ? a.max : a.min )[ 0 ] * k, ( c & 2 ? a.max : a.min )[ 1 ] * k, ( c & 4 ? a.max : a.min )[ 2 ] * k ] ); for ( let d = 0; d < 3; d ++ ) { box[ d ] = Math.min( box[ d ], q[ d ] ); box[ d + 3 ] = Math.max( box[ d + 3 ], q[ d ] ); } }

		}

		for ( const c of def.children || [] ) bounds( c, m, box );
		return box;

	};

	const subtree = ( i, out = [] ) => { out.push( i ); for ( const c of nodes[ i ].children || [] ) subtree( c, out ); return out; };
	// the files a set of nodes needs: the buffers behind their meshes (also behind a meshopt extension)
	const urisOf = ( list ) => {

		const uris = new Set();
		for ( const n of list ) if ( nodes[ n ].mesh !== undefined ) for ( const p of json.meshes[ nodes[ n ].mesh ].primitives ) for ( const a of [ ...Object.values( p.attributes ), p.indices ] ) {

			if ( a === undefined ) continue;
			const v = json.bufferViews[ json.accessors[ a ].bufferView ];
			const packed = v.extensions && ( v.extensions.EXT_meshopt_compression || v.extensions.KHR_meshopt_compression );
			const b = json.buffers[ packed ? packed.buffer : v.buffer ];
			uris.add( b.uri === undefined ? file : b.uri );

		}

		return [ ...uris ].sort();

	};

	// the bytes of a set of nodes in the file (compressed size where a view is compressed)
	const bytesOf = ( list ) => {

		const views = new Set();
		for ( const n of list ) if ( nodes[ n ].mesh !== undefined ) for ( const p of json.meshes[ nodes[ n ].mesh ].primitives ) for ( const a of [ ...Object.values( p.attributes ), p.indices ] ) if ( a !== undefined ) views.add( json.accessors[ a ].bufferView );
		let sum = 0;
		for ( const v of views ) { const d = json.bufferViews[ v ], packed = d.extensions && ( d.extensions.EXT_meshopt_compression || d.extensions.KHR_meshopt_compression ); sum += packed ? packed.byteLength : d.byteLength; }
		return sum;

	};

	const validCoverage = ( def, n ) => { const c = def.extras && def.extras.MSFT_screencoverage; return Array.isArray( c ) && c.length === n ? c : null; };
	const plan = ( coverage, n, radius ) => {

		const cov = coverage || Array.from( { length: n }, ( _, i ) => ( i < n - 1 ? 0.25 ** ( i + 1 ) : 0 ) );
		return { reference_distances: cov.slice( 0, n - 1 ).map( ( c ) => r6( coverageToDistance( c, radius ) ) ), reference_cull: cov[ n - 1 ] > 0 ? r6( coverageToDistance( cov[ n - 1 ], radius ) ) : null };

	};

	const lower = new Set();
	nodes.forEach( ( n ) => { const ids = n.extensions && n.extensions.MSFT_lod && n.extensions.MSFT_lod.ids; if ( ids ) for ( const id of ids ) lower.add( id ); } );
	const lods = [], inLod = new Set();
	nodes.forEach( ( def, i ) => {

		const ids = def.extensions && def.extensions.MSFT_lod && def.extensions.MSFT_lod.ids;
		if ( ! ids || lower.has( i ) ) return;
		const chain = [ i, ...ids ], pw = world( i );
		let box = null;
		for ( const id of chain ) { box = bounds( id, IDENT, [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ] ); if ( box[ 0 ] !== Infinity ) break; }
		const center = [ ( box[ 0 ] + box[ 3 ] ) / 2, ( box[ 1 ] + box[ 4 ] ) / 2, ( box[ 2 ] + box[ 5 ] ) / 2 ];
		const radius = Math.hypot( box[ 3 ] - box[ 0 ], box[ 4 ] - box[ 1 ], box[ 5 ] - box[ 2 ] ) / 2 * maxScale( pw );
		const coverage = validCoverage( def, chain.length );
		const trees = chain.map( ( id ) => subtree( id ) );
		for ( const t of trees ) for ( const n of t ) inLod.add( n );
		const node0 = apply( mul( pw, local( def ) ), [ 0, 0, 0 ] );
		lods.push( {
			kind: 'node', node: i, levels: chain.map( ( id ) => nodes[ id ].name ), level_nodes: chain, level_has_mesh: trees.map( ( t ) => t.some( ( n ) => nodes[ n ].mesh !== undefined ) ),
			parent: parent.has( i ) ? nodes[ parent.get( i ) ].name : null, coverage, world_center: apply( pw, center ).map( r6 ), world_radius: r6( radius ),
			...plan( coverage, chain.length, radius ),
			level0: { name: def.name, tag: def.extras && def.extras.tag !== undefined ? def.extras.tag : null, translation: def.translation || [ 0, 0, 0 ], rotation: def.rotation || [ 0, 0, 0, 1 ], scale: def.scale || [ 1, 1, 1 ], world_position: node0.map( r6 ), children: ( def.children || [] ).map( ( c ) => nodes[ c ].name ) },
			level_uris: trees.map( urisOf ), level_bytes: trees.map( bytesOf )
		} );

	} );

	( json.materials || [] ).forEach( ( def, i ) => {

		const ids = def.extensions && def.extensions.MSFT_lod && def.extensions.MSFT_lod.ids;
		if ( ! ids ) return;
		const chain = [ i, ...ids ];
		nodes.forEach( ( n, ni ) => {

			if ( n.mesh === undefined || ! json.meshes[ n.mesh ].primitives.some( ( p ) => p.material === i ) ) return;
			const pw = world( ni ), box = bounds( ni, IDENT, [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ] );
			// (a sphere around the geometry; the reference plugin takes the geometry's bounding sphere, which for these models is the same middle)
			const center = [ ( box[ 0 ] + box[ 3 ] ) / 2, ( box[ 1 ] + box[ 4 ] ) / 2, ( box[ 2 ] + box[ 5 ] ) / 2 ];
			const radius = Math.hypot( box[ 3 ] - box[ 0 ], box[ 4 ] - box[ 1 ], box[ 5 ] - box[ 2 ] ) / 2 * maxScale( pw );
			const coverage = validCoverage( def, chain.length );
			lods.push( { kind: 'material', node: ni, node_name: n.name, levels: chain.map( ( id ) => json.materials[ id ].name ), coverage, world_center: apply( pw, center ).map( r6 ), world_radius: r6( radius ), ...plan( coverage, chain.length, radius ) } );

		} );

	} );

	const model = { file, lod_count: lods.length, lods };
	const anim = ( json.animations || [] )[ 0 ];
	if ( anim ) model.animation = { name: anim.name, target: nodes[ anim.channels[ 0 ].target.node ].name, at: 0.5, world_position: [ 0, 1, 0 ] };
	if ( ! lods.length ) {

		// what GLTFLoader makes of the file: [ type, name, children ]
		const tree = ( i ) => [ nodes[ i ].mesh !== undefined ? 'Mesh' : 'Object3D', nodes[ i ].name || '', ( nodes[ i ].children || [] ).map( tree ) ];
		model.tree = json.scenes[ json.scene || 0 ].nodes.map( tree );

	}

	// tier B: what may be fetched before the first frame - the file, what is outside every LOD, and of every LOD the
	// coarsest level that draws something
	const nodeLods = lods.filter( ( l ) => l.kind === 'node' );
	const separate = nodeLods.length > 0 && nodeLods.every( ( l ) => { const first = l.level_has_mesh.lastIndexOf( true ); return l.level_uris.every( ( u, k ) => k === first || ! u.some( ( x ) => l.level_uris[ first ].includes( x ) ) ); } );
	if ( separate && ! glb && ! anim ) {

		const initial = new Set( urisOf( nodes.map( ( _, i ) => i ).filter( ( i ) => ! inLod.has( i ) ) ) );
		for ( const l of nodeLods ) for ( const u of l.level_uris[ l.level_has_mesh.lastIndexOf( true ) ] ) initial.add( u );
		model.lazy = { initial_buffers: [ ...initial ].sort(), all_buffers: ( json.buffers || [] ).filter( ( b ) => b.uri !== undefined && ! ( b.extensions && Object.values( b.extensions ).some( ( e ) => e.fallback ) ) ).map( ( b ) => b.uri ).sort() };

	}

	if ( glb && nodeLods.length ) {

		// by ranges: before the first frame the head of the file (header, JSON, one probe) and the coarsest level of every
		// LOD; in the end not more than the file (and the slack of merged requests)
		const head = 28 + raw.readUInt32LE( 12 ), coarse = nodeLods.reduce( ( s, l ) => s + l.level_bytes[ l.level_has_mesh.lastIndexOf( true ) ], 0 );
		model.glb = { bytes: raw.length, head_bytes: head, coarse_bytes: coarse, finest_bytes: nodeLods.reduce( ( s, l ) => s + l.level_bytes[ 0 ], 0 ) };

	}
	return model;

}

// ------------------------------------------------------------------------------------------------ main

const ABOUT = {
	'a_node_lod.gltf': '(a) node-level MSFT_lod, 3 levels, MSFT_screencoverage [0.25, 0.05, 0.004] (so: nothing drawn below 0.004). One buffer.',
	'a_node_lod.glb': '(f) the same model as a_node_lod.gltf in one .glb. One BIN chunk: can only be loaded whole (without range requests).',
	'b_no_coverage.gltf': '(b) as (a) without MSFT_screencoverage: the loader needs distances of its own.',
	'c_transforms.gltf': '(c) two LODs in one scene: C_Left_High (2 levels; translation, rotation, scale; a child node; under a scaled parent) and C_Right_High (4 levels).',
	'd_material_lod.gltf': '(d) material-level MSFT_lod: one mesh, materials D_Mat_High / _Mid / _Low.',
	'e_empty_lowest.gltf': '(e) the last id is a node without a mesh: at a distance nothing is drawn.',
	'g_lazy.gltf': '(g) lazy case: one .bin per level, the fine level heavy.',
	'i_ranged.glb': '(i) one .glb laid out for HTTP range requests: coarse levels at the front of the BIN chunk.',
	'k_animated.gltf': '(k) an animation channel targets the LOD\'s level 0 node.',
	'n_plain.gltf': '(n) no MSFT_lod: the result must equal the stock loader\'s.',
	'x_levels_in_scene.gltf': '(x) outside the spec: the lower levels are also listed in scene.nodes.',
	'town_cells.gltf': '(h) cells of a baked town, 4 levels each, meshopt + quantization, one .bin per cell and level (the lowest level of all cells in one .bin).',
	'town_cells.glb': '(h, ranged) the same cells in one .glb, coarse levels first, for HTTP range requests with meshopt.',
};

const PROBE = 65536; // the first range asked of a ranged .glb
const RANGED = { 'i_ranged.glb': true, 'town_cells.glb': true };
const NEEDS = { 'town_cells.gltf': [ 'meshopt' ], 'town_cells.glb': [ 'meshopt' ] };

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {

	const written = writeModels();
	const models = [];
	for ( const file of Object.keys( ABOUT ) ) {

		if ( ! fs.existsSync( path.join( OUT, file ) ) ) { console.log( `  (${file} is not here: left out of expected.json)` ); continue; }
		const m = { about: ABOUT[ file ], ...expectations( file ) };
		if ( RANGED[ file ] ) {

			const slack = 65536 * ( 1 + m.lods.length );
			m.ranged = { probe: PROBE, initial_bytes_max: m.glb.head_bytes + PROBE + m.glb.coarse_bytes + slack, total_bytes_max: m.glb.bytes + PROBE + slack };

		}
		if ( NEEDS[ file ] ) m.needs = NEEDS[ file ];
		if ( file === 'x_levels_in_scene.gltf' ) m.optional = true;
		models.push( m );

	}

	const expected = {
		about: 'Expectations for a GLTFLoader with MSFT_lod, read by check.js. Tier A: THREE.LOD in place of the node, everything loaded. Tier B: levels fetched when first needed. See README.md.',
		spec: 'https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/MSFT_lod',
		reference_camera: { fov: FOV, aspect: ASPECT, formula: 'distance = radius * sqrt( PI / ( 4 * aspect * tan( fov / 2 )^2 * coverage ) )' },
		models
	};
	fs.writeFileSync( path.join( HERE, 'expected.json' ), JSON.stringify( expected, null, '\t' ) + '\n' );
	console.log( `${written.length} model files written to ${path.relative( process.cwd(), OUT ) || '.'}, expected.json: ${models.length} models, ${models.reduce( ( s, m ) => s + m.lod_count, 0 )} LODs` );

}
