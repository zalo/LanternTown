// Web bake, packing step (run by tools/webbake.py after tools/webbake_bl.py):
//
//   node tools/webbake_pack.mjs <out dir> ['{"lo_ratio":0.22,...}']
//
// Reads the raw chunks the Blender step left in <out>/.raw and writes what the page loads:
//   base.bin        every chunk at its lowest level ("lo": the lit LOD2 geometry simplified with its colours and uvs
//                   as constraints, small loose parts pruned), nearest the spawn first
//   m_<cell>.bin    "mid" level of a cell (lit LOD2 geometry)
//   n_<cell>.bin    "near" level of a cell (lit LOD1 geometry, simplified within 2 cm: half the triangles, the same
//                   picture)
//   r_<cell>.bin    "ring" level of a cell: the near level of the phone tier (the same lit LOD1 geometry, simplified
//                   further - what a phone shows right around the camera has to fit its download budget)
//   manifest.json
// All vertex / index streams are meshopt-compressed (decoded by three/addons/libs/meshopt_decoder.module.js).

import fs from 'node:fs';
import path from 'node:path';
import { MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

const OUT = process.argv[ 2 ];
const OPT = Object.assign( {
	lo_ratio: 0.13, lo_error: 0.22, // inner cells: keep at most this share of the triangles, stop at this error (m)
	outer_ratio: 0.3, outer_error: 0.45, // cells beyond the walkable area's surroundings
	colour_weight: 0.15, uv_weight: 0.03,
	lo_glow_lock: 1, // lit windows / signs keep their outline in the lowest level
	// lanterns are what the town is about: in the lowest level of the inner cells their posts and fixtures (loose parts
	// at a light) are simplified apart from the rest, within this error (m), and their glass stays. 0: no special case
	lantern_error: 0.03, lantern_ratio: 0.5,
	// small things of their own (a bollard, a crate, a barrel: flagged by the bake) fold into shards at lo_error:
	// in the lowest level of the inner cells they get their own, tighter error (m). 0: no special case
	prop_error: 0.08, prop_ratio: 0.15,
	mid_error: 0.03, mid_ratio: 0.5, mid_colour_weight: 0.5, // the mid level (LOD2 geometry) is simplified too (m), where lighting and uvs allow; 0: as baked
	lock_border: 1, // mid / near: open edges stay (a pane, a shutter, a sign keep their outline)
	near_error: 0.02, near_ratio: 0.08, near_colour_weight: 0.3, // the near level: simplified within this error (m); 0: as baked
	// the phone's near level, from the same LOD1 geometry. An error bound hardly makes it smaller (it is small parts,
	// a few triangles each), so its triangles are put in order instead: the loose parts by size, largest first, with
	// the index count down to each size in the manifest ("parts") - the page draws a cell down to the size that still
	// shows at its distance, and a little less when the view is over its budget. Lit and lantern parts come first
	// (never left out); parts under ring_min_part (m) are left out of the file. A cell of ring_cluster_min triangles
	// or more is cut in ring_clusters lumps (a power of two) that are culled and drawn down on their own, next to
	// one of everything that spans ring_span (m) of ground or more. ring_error 0: no such level
	ring_error: 0.02, ring_ratio: 0.08, ring_colour_weight: 0.3, ring_prune: 0,
	// the fine level (LOD0 geometry, the top level of the medium / high tiers): only what cannot be seen from a step
	// away goes - mostly the triangles the bake's lamp refinement added where the light turned out even
	fine_error: 0.004, fine_ratio: 0.1, fine_colour_weight: 0.6, fine_prune: 0,
	// the fine level (LOD0 geometry, medium / high) is laid out like the ring: lumps of about fine_cluster_tris
	// triangles (16 at most) that are culled on their own, each with its loose parts in order of size, so the page
	// can leave out what is under a pixel or two at its distance. Nothing is left out of the file. fine_sizes null: one range
	fine_sizes: [ 0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025, 0.017, 0.012, 0 ], fine_clusters: 16, fine_cluster_tris: 45000, fine_span: 9,
	ring_sizes: [ 0.3, 0.22, 0.16, 0.12, 0.09, 0.065, 0.045, 0.03, 0 ], ring_min_part: 0.02, ring_clusters: 4, ring_span: 9, ring_cluster_min: 20000,
	vertex_version: 1, vertex_level: 3, // meshopt vertex codec (version 1 needs the decoder of meshoptimizer >= 0.23: three's is 1.1)
	// the light direction of the fine level (3 x int8 + 2 x 4 bit, see light_dir() of the bake): bits kept of each
	dir_bits: 6, dir_rbits: 3,
	// the ground (the terrain's vertices, flagged by the bake) is one thing at every level of a cell: ground_lock keeps
	// all of its vertices in the mid / near / ring / fine levels, ground_unify gives the mid and near levels the fine
	// level's colour and gloss at them. A pool of light, its streaks and the wet film on the paving are then the same
	// function of the place whatever level the cell is shown at. (A bake without the fine level - the lean tiers' own -
	// takes the near level's ground for the mid level.)
	ground_lock: 1, ground_unify: 1,
	// A lit room is a glowing card millimetres behind its window, a mantle a card inside its cage: the models keep a
	// clearance between a glowing face and what lies behind / in front of it, and these errors are ten times that, so
	// the simplifier would move a reveal, a glazing bar or the wall itself through the card (the two then flicker as
	// one walks). *_glow_guard (m, 0: off): the opaque triangles that pass within this of a glowing face keep their
	// vertices, as the glowing ones do - around every glowing face the level is the model, clearance and all. The
	// lowest level has none: at its error (0.2 m) that would be the whole of every shop front, and the page draws
	// its glowing faces a few centimetres towards the eye instead (PULL of lite.js).
	fine_glow_guard: 0.006, near_glow_guard: 0.03, ring_glow_guard: 0.03, mid_glow_guard: 0.045, lo_glow_guard: 0,
	// ... and at the lowest level of the inner cells: lo_ground 1 keeps the ground's triangles as the mid level has them
	// (a pool of light does not lose a quarter of itself where mid gives way to lo), 2 simplifies the ground on its own
	// within lo_ground_error (m) with the colour weighted lo_ground_colour, 0 simplifies it with everything else
	lo_ground: 1, lo_ground_error: 0.05, lo_ground_colour: 4,
	// a bake with a second look (LOOK2 of webbake_bl.py: a second lit colour in every vertex) is simplified by both: the
	// second colour counts this many times what the first does at every level (0: the mesh follows the first look alone)
	colour2_weight: 1,
	colour_bits: 13, // precision kept of the lit vertex colour (stored as its root: 13 bits are 1/8192 steps, and a tenth less to download)
	clear_panes: 1, // the clear glass around a burner with a fitting takes pane bytes too (clearPanes())
	burner_r: 1.5, // ... a glow set on the burner off its light (halo_at) goes by the nearest light this near (m); 0 = only a light at the row
	// ... 1 = also the burners inside an opaque glowing body (a row with a lift: an opal globe). Off: as a lit pane its clear
	// skin takes the dirt and the rim of a lamp's glazing and the globe behind it comes out 4 % darker and mottled (measured)
	burner_body: 0,
	// A landmark (cls 'landmark' of town.json: the hall that closes the square, the gate, the fountain) is what a view
	// is read by from far off, and lo_error is four pixels of it at the 40 m where the phone tier shows the lowest
	// level. landmark_boxes ([ x0, z0, x1, z1 ] on the ground, three frame, one per such item: tools/webbake.py fills
	// it in from town.json) says where they stand, and in the lowest level of the inner cells the loose parts that lie
	// inside a box (the landmark, and what hangs on it or stands on its steps) are simplified apart from the rest:
	// within landmark_lo_error (m; 0: no special case), down to landmark_lo_ratio of their triangles at most, and what
	// passes within landmark_lo_guard (m) of a glowing face keeps its vertices (the hands, hour marks and numerals
	// over a lit dial, the glazing bars over a lit window). With the mid level's values (0.03, 0.5, guard 0.045,
	// border 1) a landmark's lowest level is its mid level.
	landmark_boxes: null, landmark_lo_error: 0, landmark_lo_ratio: 0.1, landmark_lo_guard: 0, landmark_lo_colour: 0.5, landmark_lo_border: 0,
	// landmark_src 1: the mid level of a landmark's cell - the inner cell with most triangles inside its box - is made
	// from the near level's geometry (LOD1) instead of LOD2's, within landmark_mid_error (m) down to
	// landmark_mid_ratio: for a landmark whose LOD2 model has lost what it is read by. 2: its lowest level as well
	// (three times the triangles there: LOD1 is thousands of loose parts no bound joins).
	// The whole cell changes its source, so only boxes of landmark_src_min (m, the longer side) and more take part
	landmark_src: 0, landmark_src_min: 20, landmark_mid_error: 0.04, landmark_mid_ratio: 0.3, landmark_mid_prune: 0, // (prune 1: loose parts smaller than the error go)
	// lo_no_pull (m; 0: none): the page pulls the glowing faces of the lowest level towards the eye (PULL of lite.js),
	// and what stands closer than that in front of one - the hands of a clock - goes behind it. A glowing card of the
	// lowest level with kept geometry within this distance (kept by a glow guard, or marked to stay by the bake) is
	// marked as not to be pulled: aux 65534 instead of 65535. The page tells the two apart with ?pullsel=1; without
	// it both are pulled as before.
	lo_no_pull: 0,
	// landmark_lo_floor (triangles; 0: off): the landmark's loose parts are simplified one by one, each down to its
	// share of landmark_lo_ratio by the root of its area, none below the floor and none pruned
	landmark_lo_floor: 0,
	// lo_flags: further simplifier flags for the lowest level ('Regularize', 'Permissive', .. comma separated; trials)
	lo_flags: '',
	// landmark_stat 1: print, per cell, how far the landmarks' surface is from what the lowest level keeps of it
	landmark_stat: 0,
	// keep_whole 1: a triangle whose three corners the bake marks as "must stay" (flag 1: a part its model declares
	// with lod_keep / lod_standin, docs/LOD.md - the hands, hour marks and numerals of a dial) takes no part in a
	// simplification that prunes. Its corners are locked there anyway, nothing of it can be merged - but 'Prune'
	// takes whole loose parts under the error bound whatever their locks: the far dial's twelve hour marks went with
	// the landmark's 0.22 m, the strokes of I, II, III and IIII with the mid level's 0.08 m. 0: as before
	keep_whole: 1,
	// holes 1: mend where the census saw through a reduced level (tools/pvs.py --holes -> .raw/holes.json: the places,
	// on the top level's surface, where a lower level shows another cell or the sky from where it is drawn). The level's
	// source nearly always has the surface there - the simplifier took it: a shutter, a frame, a chimney pot went as a
	// whole, an eave or a verge was pulled back. A loose part of at most holes_tris triangles and holes_size metres with
	// a triangle within holes_reach of such a place stays as it is; of a larger part the vertices of the triangles
	// within holes_lock of one are locked. Nothing is added to a level, so nothing new can lie in the plane of something
	// else. The census must be of a pack made without it (holes 0 or 2). 2: count only (the log line), mend nothing.
	// holes_levels: the levels it is done for. Measured over 600 places to stand (full bake, pixels seen through in a
	// million, before -> after): lo 4048 -> 587 for 32 % more triangles; mid 675 -> 632 for 8 % more, near 367 -> 304 for
	// 9 % more - the lowest level is where it pays
	holes: 0, holes_reach: 0.08, holes_lock: 0.15, holes_tris: 300, holes_size: 3, holes_file: 'holes.json', holes_levels: 'lo',
}, JSON.parse( process.argv[ 3 ] || '{}' ) );
const RAW = path.join( OUT, '.raw' );
const man = JSON.parse( fs.readFileSync( path.join( RAW, 'raw.json' ), 'utf8' ) );
let W = 10; // u16 per vertex of the level in hand (12 where the bake added the light's direction: `w` of the raw level)
// ... 14 / 18: the same two with a second look - its colour (+ a spare word) at word 10 / 12, its direction at word 16
const col2 = ( w ) => w === 14 ? 10 : w === 18 ? 12 : 0, hasDir = ( w ) => w === 12 || w === 18;
const K_OPAQUE = 0, K_GLASS = 1;

await MeshoptEncoder.ready; await MeshoptSimplifier.ready; await MeshoptDecoder.ready;

function readRaw( m ) {

	const b = fs.readFileSync( path.join( RAW, m.raw ) );
	const ab = b.buffer.slice( b.byteOffset, b.byteOffset + b.byteLength );
	const vb = new Uint16Array( ab, 0, m.nv * W );
	const ib = new Uint32Array( ab, m.nv * W * 2, m.ni );
	// per vertex: bit 1 = must stay, bit 2 = part of a small thing (BK prop_size in webbake_bl.py), bit 4 = ground
	const flags = new Uint8Array( ab, m.nv * W * 2 + m.ni * 4, m.nv );
	if ( OPT.clear_panes ) clearPanes( vb, ib, m );
	return { vb, ib, flags, lock: flags.map( ( f ) => f & 1 ) };

}

// The clear glass of a lamp head (a gas lamp's panes around its mantle): the bake gives pane bytes - the vector to
// the burner and its hue, in the uv slots glass has no use for - only to a lantern's own amber sheet (pane_bytes()),
// and lights clear glass from its front alone, so the panes of the square's lamps were black glass around a white
// bulb where the engine shows them lit from inside. Here every glass vertex without pane bytes that has a burner
// with a fitting (a halo row with its flag set: the lights of 12 cd and more) within pane_r gets the same four
// bytes, and in its layer slot (glass has no texture) the peak of that burner's light x 64 - which is how the page
// tells such a pane from a lantern's sheet (0 there), and what it lights the dirt on the glass with (glassPane of
// lite.js). At every level alike.
let burnersOff = 0; // rows that are a burner off its light
const BURNERS = ( () => {

	const out = [];
	for ( const h of man.halos || [] ) {

		if ( ! h[ 7 ] || ( h[ 8 ] > 0 && ! OPT.burner_body ) ) continue;
		let best = null, bd = 0.05, at = null;
		for ( const l of man.lights || [] ) { const d = Math.hypot( l[ 0 ] - h[ 0 ], l[ 1 ] - h[ 1 ], l[ 2 ] - h[ 2 ] ); if ( d < bd ) { bd = d; best = l; } }
		if ( ! best && OPT.burner_r > 0 ) {

			// a glow set where the burner is seen, off its light (halo_at of town.json: the side globes of a post's head,
			// the hall's clock lanterns): the row is the burner, the nearest lit light within burner_r its colour and strength
			bd = OPT.burner_r; at = h;
			for ( const l of man.lights || [] ) {

				if ( ! ( Math.max( l[ 3 ], l[ 4 ], l[ 5 ] ) > 0 ) ) continue;
				const d = Math.hypot( l[ 0 ] - h[ 0 ], l[ 1 ] - h[ 1 ], l[ 2 ] - h[ 2 ] );
				if ( d < bd ) { bd = d; best = l; }

			}

		}

		if ( ! best ) continue;
		if ( ! at ) at = best; else burnersOff ++;
		const pk = Math.max( best[ 3 ], best[ 4 ], best[ 5 ] );
		if ( ! ( pk > 0 ) ) continue;
		const hue = ( Math.max( 2, Math.round( Math.min( 1, best[ 4 ] / pk ) * 15 ) ) << 4 ) | Math.round( Math.min( 1, best[ 5 ] / pk ) * 15 );
		out.push( { p: [ at[ 0 ], at[ 1 ], at[ 2 ] ], hue, code: Math.max( 1, Math.min( 32767, Math.round( pk * 64 ) ) ) } );

	}

	return out;

} )();
const paneStat = { verts: 0, heads: new Set() };

function clearPanes( vb, ib, m ) {

	const pr = man.bake && man.bake.pane_uv ? man.bake.pane_r || 0 : 0;
	if ( ! ( pr > 0 ) || ! BURNERS.length ) return;
	const near = BURNERS.filter( ( b ) => [ 0, 1, 2 ].every( ( c ) => b.p[ c ] > m.min[ c ] - pr && b.p[ c ] < m.max[ c ] + pr ) );
	if ( ! near.length ) return;
	const seen = new Uint8Array( m.nv );
	for ( const [ kind, start, count ] of m.sub ) {

		if ( kind !== K_GLASS ) continue;
		for ( let i = start; i < start + count; i ++ ) {

			const v = ib[ i ], o = v * W;
			if ( seen[ v ] ) continue;
			seen[ v ] = 1;
			if ( vb[ o + 5 ] >> 12 ) continue; // a lantern's sheet: the bake's own bytes
			const x = vb[ o ] * m.q + m.min[ 0 ], y = vb[ o + 1 ] * m.q + m.min[ 1 ], z = vb[ o + 2 ] * m.q + m.min[ 2 ];
			let best = null, bd = Infinity;
			for ( const b of near ) { const d = ( b.p[ 0 ] - x ) ** 2 + ( b.p[ 1 ] - y ) ** 2 + ( b.p[ 2 ] - z ) ** 2; if ( d < bd ) { bd = d; best = b; } }
			const vx = best.p[ 0 ] - x, vy = best.p[ 1 ] - y, vz = best.p[ 2 ] - z;
			if ( Math.max( Math.abs( vx ), Math.abs( vy ), Math.abs( vz ) ) > pr ) continue;
			const q = ( t ) => Math.max( - 127, Math.min( 127, Math.round( t * 127 / pr ) ) ) + 128;
			vb[ o + 4 ] = q( vx ) | q( vy ) << 8;
			vb[ o + 5 ] = q( vz ) | best.hue << 8;
			vb[ o + 3 ] = best.code;
			paneStat.verts ++; paneStat.heads.add( best );

		}

	}

}

// -> { blob, meta }: each range compacted and ordered for the vertex cache, then both streams compressed
function pack( vb, ib, sub, m ) {

	const vs = [], is = [], nsub = [], parts = [];
	let nv = 0, ni = 0, cur = null;
	for ( const [ kind, start, count, size, g ] of sub ) { // size, g: a step of cluster g, which goes on with the one before it (see ordered())

		if ( ! count && size === undefined ) continue;
		const idx = ib.slice( start, start + count );
		const [ remap, unique ] = count ? MeshoptEncoder.reorderMesh( idx, true, false ) : [ [], 0 ];
		const v = new Uint16Array( unique * W );
		for ( let i = 0; i < remap.length; i ++ ) {

			const r = remap[ i ];
			if ( r !== 0xffffffff ) v.set( vb.subarray( i * W, i * W + W ), r * W );

		}

		if ( nv ) for ( let i = 0; i < idx.length; i ++ ) idx[ i ] += nv;
		vs.push( v ); is.push( idx );
		if ( size === undefined ) { nsub.push( [ kind, ni, count ] ); cur = null; }
		else {

			if ( cur && cur.g === g ) nsub[ cur.sub ][ 2 ] += count;
			else { nsub.push( [ kind, ni, count ] ); parts.push( cur = { g, sub: nsub.length - 1, steps: [], lo: [ 65535, 65535, 65535 ], hi: [ 0, 0, 0 ] } ); }
			cur.steps.push( [ size, nsub[ cur.sub ][ 2 ] ] );
			for ( let i = 0; i < v.length; i += W ) for ( let c = 0; c < 3; c ++ ) {

				if ( v[ i + c ] < cur.lo[ c ] ) cur.lo[ c ] = v[ i + c ];
				if ( v[ i + c ] > cur.hi[ c ] ) cur.hi[ c ] = v[ i + c ];

			}

		}

		nv += unique; ni += count;

	}

	const V = new Uint16Array( nv * W ); let o = 0;
	for ( const v of vs ) { V.set( v, o ); o += v.length; }
	const it = nv <= 65535 ? 2 : 4;
	const I = it === 2 ? new Uint16Array( ni ) : new Uint32Array( ni ); o = 0;
	for ( const i of is ) { I.set( i, o ); o += i.length; }
	if ( OPT.colour_bits < 16 ) {

		// the lit colour ( square root of the light, 16 bit ) carries more precision than a screen shows: rounding it
		// leaves low bits that compress to nothing
		const s = 1 << ( 16 - OPT.colour_bits ), top = 65536 - s;
		for ( let i = 0; i < nv; i ++ ) for ( let c = 6; c < 9; c ++ ) V[ i * W + c ] = Math.min( top, Math.round( V[ i * W + c ] / s ) * s );
		const c2 = col2( W );
		if ( c2 ) for ( let i = 0; i < nv; i ++ ) for ( let c = c2; c < c2 + 3; c ++ ) V[ i * W + c ] = Math.min( top, Math.round( V[ i * W + c ] / s ) * s );

	}

	if ( hasDir( W ) && ( OPT.dir_bits < 8 || OPT.dir_rbits < 4 ) ) {

		const s = 1 << ( 8 - OPT.dir_bits ), rs = 1 << ( 4 - OPT.dir_rbits ), B = new Uint8Array( V.buffer );
		const q = ( v ) => Math.max( 1, Math.min( 255, 128 + Math.round( ( v - 128 ) / s ) * s ) ), qr = ( v ) => Math.min( 15, Math.round( v / rs ) * rs );
		for ( const at of W === 18 ? [ 20, 32 ] : [ 20 ] ) for ( let i = 0; i < nv; i ++ ) { // (32: the second look's direction)

			const o = i * W * 2 + at;
			B[ o ] = q( B[ o ] ); B[ o + 1 ] = q( B[ o + 1 ] ); B[ o + 2 ] = q( B[ o + 2 ] );
			B[ o + 3 ] = qr( B[ o + 3 ] >> 4 ) << 4 | qr( B[ o + 3 ] & 15 );

		}

	}

	const ve = MeshoptEncoder.encodeVertexBufferLevel( new Uint8Array( V.buffer ), nv, W * 2, OPT.vertex_level, OPT.vertex_version );
	const ie = MeshoptEncoder.encodeIndexBuffer( new Uint8Array( I.buffer ), ni, it );
	const blob = Buffer.concat( [ Buffer.from( ve.buffer, ve.byteOffset, ve.byteLength ), Buffer.from( ie.buffer, ie.byteOffset, ie.byteLength ) ] );
	// bounds of what is left
	let x0 = 65535, y0 = 65535, z0 = 65535, x1 = 0, y1 = 0, z1 = 0;
	for ( let i = 0; i < nv; i ++ ) {

		const x = V[ i * W ], y = V[ i * W + 1 ], z = V[ i * W + 2 ];
		if ( x < x0 ) x0 = x; if ( x > x1 ) x1 = x; if ( y < y0 ) y0 = y; if ( y > y1 ) y1 = y; if ( z < z0 ) z0 = z; if ( z > z1 ) z1 = z;

	}

	const r = ( v ) => Math.round( v * 1e4 ) / 1e4;
	const box = ( q ) => q.lo.map( ( v, c ) => r( m.min[ c ] + Math.min( v, q.hi[ c ] ) * m.q ) ).concat( q.hi.map( ( v, c ) => r( m.min[ c ] + Math.max( v, q.lo[ c ] ) * m.q ) ) );
	// parts: the clusters of the opaque range - [ range, [ [ size, index count down to parts of that size ], .. ], box ]
	const meta = { min: m.min, q: m.q, nv, ni, it, vlen: ve.byteLength, ...( W !== 10 ? { w: W } : {} ), sub: nsub, ...( parts.length ? { parts: parts.map( ( q ) => [ q.sub, q.steps, box( q ) ] ) } : {} ),
		bmin: nv ? [ r( m.min[ 0 ] + x0 * m.q ), r( m.min[ 1 ] + y0 * m.q ), r( m.min[ 2 ] + z0 * m.q ) ] : m.min,
		bmax: nv ? [ r( m.min[ 0 ] + x1 * m.q ), r( m.min[ 1 ] + y1 * m.q ), r( m.min[ 2 ] + z1 * m.q ) ] : m.min };
	return { blob, meta, V, I };

}

let checked = false;
function selfCheck( p ) {

	// the page decodes with three's copy of the decoder: make sure it reads what this encoder wrote
	if ( checked || ! p.meta.nv ) return;
	checked = true;
	const v = new Uint8Array( p.meta.nv * W * 2 );
	MeshoptDecoder.decodeVertexBuffer( v, p.meta.nv, W * 2, new Uint8Array( p.blob.buffer, p.blob.byteOffset, p.meta.vlen ) );
	const i = new Uint8Array( p.meta.ni * p.meta.it );
	MeshoptDecoder.decodeIndexBuffer( i, p.meta.ni, p.meta.it, new Uint8Array( p.blob.buffer, p.blob.byteOffset + p.meta.vlen, p.blob.byteLength - p.meta.vlen ) );
	const a = new Uint8Array( p.V.buffer ), b = new Uint8Array( p.I.buffer );
	for ( let k = 0; k < a.length; k ++ ) if ( a[ k ] !== v[ k ] ) throw new Error( 'meshopt vertex round trip failed' );
	for ( let k = 0; k < b.length; k += 3 * p.meta.it ) {

		// triangles may come back rotated: compare as sets
		const t = ( buf, o ) => { const r = []; for ( let c = 0; c < 3; c ++ ) r.push( p.meta.it === 2 ? buf[ o + c * 2 ] | buf[ o + c * 2 + 1 ] << 8 : ( buf[ o + c * 4 ] | buf[ o + c * 4 + 1 ] << 8 | buf[ o + c * 4 + 2 ] << 16 | buf[ o + c * 4 + 3 ] << 24 ) >>> 0 ); return r.sort().join(); };
		if ( t( b, k ) !== t( i, k ) ) throw new Error( 'meshopt index round trip failed' );

	}

}

// the loose parts of a range (triangles connected through vertices at the same position):
// -> { of: part of each triangle, parts: Map( part -> [ min x y z, max x y z, indices, thickness, area, volume ] ) }, world
// metres. thickness: the smaller of the lowest side of its box and 4 volume / area (a bar's width, whichever way it points)
function looseParts( vb, idx, m ) {

	const key = new Map(), id = new Int32Array( m.nv ).fill( - 1 ), par = [];
	const find = ( a ) => {

		while ( par[ a ] !== a ) { par[ a ] = par[ par[ a ] ]; a = par[ a ]; }
		return a;

	};

	for ( const i of idx ) {

		if ( id[ i ] >= 0 ) continue;
		const k = vb[ i * W ] + vb[ i * W + 1 ] * 65536 + vb[ i * W + 2 ] * 4294967296; // welded by position
		let w = key.get( k );
		if ( w === undefined ) { w = par.length; par.push( w ); key.set( k, w ); }
		id[ i ] = w;

	}

	for ( let t = 0; t < idx.length; t += 3 ) {

		const a = find( id[ idx[ t ] ] );
		par[ find( id[ idx[ t + 1 ] ] ) ] = a; par[ find( id[ idx[ t + 2 ] ] ) ] = a;

	}

	const parts = new Map(), of = new Int32Array( idx.length / 3 );
	const P = [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
	for ( let t = 0; t < idx.length; t += 3 ) {

		const r = find( id[ idx[ t ] ] );
		of[ t / 3 ] = r;
		let b = parts.get( r );
		if ( ! b ) parts.set( r, b = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity, 0, 0, 0, 0 ] );
		for ( let v = 0; v < 3; v ++ ) for ( let c = 0; c < 3; c ++ ) {

			const x = vb[ idx[ t + v ] * W + c ] * m.q;
			P[ v * 3 + c ] = x;
			if ( x < b[ c ] ) b[ c ] = x;
			if ( x > b[ 3 + c ] ) b[ 3 + c ] = x;

		}

		b[ 6 ] += 3;
		// area and signed volume (about the cell's corner; exact for a closed part)
		const ux = P[ 3 ] - P[ 0 ], uy = P[ 4 ] - P[ 1 ], uz = P[ 5 ] - P[ 2 ], vx = P[ 6 ] - P[ 0 ], vy = P[ 7 ] - P[ 1 ], vz = P[ 8 ] - P[ 2 ];
		const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
		b[ 8 ] += Math.hypot( nx, ny, nz ) / 2;
		b[ 9 ] += ( P[ 0 ] * nx + P[ 1 ] * ny + P[ 2 ] * nz ) / 6;

	}

	for ( const b of parts.values() ) {

		b[ 7 ] = Math.min( b[ 3 ] - b[ 0 ], b[ 4 ] - b[ 1 ], b[ 5 ] - b[ 2 ], 4 * Math.abs( b[ 9 ] ) / Math.max( b[ 8 ], 1e-9 ) );
		for ( let c = 0; c < 3; c ++ ) { b[ c ] += m.min[ c ]; b[ 3 + c ] += m.min[ c ]; }

	}

	return { of, parts };

}

// the loose parts that belong to a lantern: a post (tall, slim) with a light at it, or a small part right at a light
function lanternParts( lp, m ) {

	const lights = ( man.lights || [] ).filter( ( l ) => l[ 0 ] > m.min[ 0 ] - 2 && l[ 2 ] > m.min[ 2 ] - 2 && l[ 0 ] < m.min[ 0 ] + 65535 * m.q + 2 && l[ 2 ] < m.min[ 2 ] + 65535 * m.q + 2 );
	const at = ( b, pad ) => lights.some( ( l ) => l[ 0 ] > b[ 0 ] - pad && l[ 0 ] < b[ 3 ] + pad && l[ 1 ] > b[ 1 ] - pad && l[ 1 ] < b[ 4 ] + pad && l[ 2 ] > b[ 2 ] - pad && l[ 2 ] < b[ 5 ] + pad );
	const mine = new Set();
	for ( const [ r, b ] of lp.parts ) {

		const dx = b[ 3 ] - b[ 0 ], dy = b[ 4 ] - b[ 1 ], dz = b[ 5 ] - b[ 2 ];
		const post = dy >= 1.6 && Math.max( dx, dz ) <= 1.2 && b[ 6 ] <= 12000;
		if ( post ? at( b, 0.8 ) : Math.hypot( dx, dy, dz ) < 1.2 && at( b, 0.25 ) ) mine.add( r );

	}

	return mine;

}

// the triangles of an index list for which pick( part of the triangle ) holds, and the others
function split( idx, of, pick ) {

	const a = [], rest = [];
	for ( let t = 0; t < idx.length; t += 3 ) ( pick( of[ t / 3 ] ) ? a : rest ).push( idx[ t ], idx[ t + 1 ], idx[ t + 2 ] );
	return [ new Uint32Array( a ), new Uint32Array( rest ) ];

}

// squared distance of the point p to the triangle a b c (Ericson, Real-Time Collision Detection 5.1.5)
function pointTri2( px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz ) {

	const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
	const apx = px - ax, apy = py - ay, apz = pz - az;
	const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
	let qx, qy, qz;
	if ( d1 <= 0 && d2 <= 0 ) { qx = ax; qy = ay; qz = az; } else {

		const bpx = px - bx, bpy = py - by, bpz = pz - bz;
		const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
		if ( d3 >= 0 && d4 <= d3 ) { qx = bx; qy = by; qz = bz; } else {

			const vc = d1 * d4 - d3 * d2;
			if ( vc <= 0 && d1 >= 0 && d3 <= 0 ) { const v = d1 / ( d1 - d3 || 1 ); qx = ax + v * abx; qy = ay + v * aby; qz = az + v * abz; } else {

				const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
				const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
				if ( d6 >= 0 && d5 <= d6 ) { qx = cx; qy = cy; qz = cz; } else {

					const vb = d5 * d2 - d1 * d6;
					if ( vb <= 0 && d2 >= 0 && d6 <= 0 ) { const w = d2 / ( d2 - d6 || 1 ); qx = ax + w * acx; qy = ay + w * acy; qz = az + w * acz; } else {

						const va = d3 * d6 - d5 * d4;
						if ( va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0 ) {

							const w = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) || 1 );
							qx = bx + w * ( cx - bx ); qy = by + w * ( cy - by ); qz = bz + w * ( cz - bz );

						} else {

							const den = 1 / ( va + vb + vc || 1 ), v = vb * den, w = vc * den;
							qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w;

						}

					}

				}

			}

		}

	}

	return ( px - qx ) * ( px - qx ) + ( py - qy ) * ( py - qy ) + ( pz - qz ) * ( pz - qz );

}

// OPT.*_glow_guard: the vertices of every opaque triangle that passes within D (m) of a glowing one are marked in
// `lock`. Two tests, each over a grid: a corner of the triangle within D of a glowing face (small things at a large
// card: a glazing bar, a frame), and a point of a glowing face - its corners, and a 5 cm lattice over the large ones -
// within D of the triangle (large things at a card: the wall it is mounted on).
const guardStat = {};
function glowGuard( vb, ib, m, lock, D ) {

	const nv = m.nv, q = m.q, D2 = D * D, S = 8192;
	const glow = new Uint8Array( nv );
	const G = [], O = [];
	for ( const [ kind, start, count ] of m.sub ) {

		if ( kind !== K_OPAQUE ) continue;
		for ( let t = start; t < start + count; t += 3 ) {

			const n = GLOWS( vb[ ib[ t ] * W + 9 ] ) + GLOWS( vb[ ib[ t + 1 ] * W + 9 ] ) + GLOWS( vb[ ib[ t + 2 ] * W + 9 ] );
			if ( n === 3 ) { G.push( t ); glow[ ib[ t ] ] = glow[ ib[ t + 1 ] ] = glow[ ib[ t + 2 ] ] = 1; } else O.push( t );

		}

	}

	if ( ! G.length || ! O.length ) return;
	const X = ( i ) => vb[ i * W ] * q, Y = ( i ) => vb[ i * W + 1 ] * q, Z = ( i ) => vb[ i * W + 2 ] * q;
	const put = ( map, k, v ) => { const a = map.get( k ); if ( a ) a.push( v ); else map.set( k, [ v ] ); };
	// the glowing triangles by the cells (1/16 m) they reach, D included
	const H = 16, tg = new Map();
	for ( const t of G ) {

		const a = ib[ t ], b = ib[ t + 1 ], c = ib[ t + 2 ];
		const x0 = Math.floor( ( Math.min( X( a ), X( b ), X( c ) ) - D ) * H ), x1 = Math.floor( ( Math.max( X( a ), X( b ), X( c ) ) + D ) * H );
		const y0 = Math.floor( ( Math.min( Y( a ), Y( b ), Y( c ) ) - D ) * H ), y1 = Math.floor( ( Math.max( Y( a ), Y( b ), Y( c ) ) + D ) * H );
		const z0 = Math.floor( ( Math.min( Z( a ), Z( b ), Z( c ) ) - D ) * H ), z1 = Math.floor( ( Math.max( Z( a ), Z( b ), Z( c ) ) + D ) * H );
		for ( let z = z0; z <= z1; z ++ ) for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) put( tg, x + 1 + S * ( y + 1 + S * ( z + 1 ) ), t );

	}

	// (a) per vertex: 1 = within D of a glowing face, 2 = not
	const nearV = new Uint8Array( nv );
	const vnear = ( i ) => {

		if ( nearV[ i ] ) return nearV[ i ] === 1;
		const px = X( i ), py = Y( i ), pz = Z( i );
		const list = tg.get( Math.floor( px * H ) + 1 + S * ( Math.floor( py * H ) + 1 + S * ( Math.floor( pz * H ) + 1 ) ) );
		let hit = false;
		if ( list ) for ( const t of list ) {

			const a = ib[ t ], b = ib[ t + 1 ], c = ib[ t + 2 ];
			if ( pointTri2( px, py, pz, X( a ), Y( a ), Z( a ), X( b ), Y( b ), Z( b ), X( c ), Y( c ), Z( c ) ) < D2 ) { hit = true; break; }

		}

		nearV[ i ] = hit ? 1 : 2;
		return hit;

	};

	// (b) points of the glowing faces, by cells of 1/4 m
	const Hs = 4, step = 0.05, pts = [], pg = new Map();
	const point = ( x, y, z ) => { put( pg, Math.floor( x * Hs ) + 1 + S * ( Math.floor( y * Hs ) + 1 + S * ( Math.floor( z * Hs ) + 1 ) ), pts.length ); pts.push( x, y, z ); };
	for ( let i = 0; i < nv; i ++ ) if ( glow[ i ] ) point( X( i ), Y( i ), Z( i ) );
	for ( const t of G ) {

		const a = ib[ t ], b = ib[ t + 1 ], c = ib[ t + 2 ];
		const ax = X( a ), ay = Y( a ), az = Z( a ), bx = X( b ) - ax, by = Y( b ) - ay, bz = Z( b ) - az, cx = X( c ) - ax, cy = Y( c ) - ay, cz = Z( c ) - az;
		const L = Math.sqrt( Math.max( bx * bx + by * by + bz * bz, cx * cx + cy * cy + cz * cz, ( bx - cx ) ** 2 + ( by - cy ) ** 2 + ( bz - cz ) ** 2 ) );
		const n = Math.min( 32, Math.ceil( L / step ) );
		if ( n < 2 ) continue;
		for ( let i = 0; i <= n; i ++ ) for ( let j = 0; j <= n - i; j ++ ) {

			if ( ( i === 0 || i === n ) && ( j === 0 || j === n - i ) && ( i + j === 0 || i + j === n ) ) continue; // (the corners are in)
			const u = i / n, v = j / n;
			point( ax + bx * u + cx * v, ay + by * u + cy * v, az + bz * u + cz * v );

		}

	}

	let tris = 0, verts = 0;
	for ( const t of O ) {

		const a = ib[ t ], b = ib[ t + 1 ], c = ib[ t + 2 ];
		// (all three at glowing vertices or kept already: nothing to find out)
		if ( lock[ a ] && lock[ b ] && lock[ c ] ) continue;
		let hit = ( ! glow[ a ] && vnear( a ) ) || ( ! glow[ b ] && vnear( b ) ) || ( ! glow[ c ] && vnear( c ) );
		if ( ! hit ) {

			const ax = X( a ), ay = Y( a ), az = Z( a ), bx = X( b ), by = Y( b ), bz = Z( b ), cx = X( c ), cy = Y( c ), cz = Z( c );
			const lx = Math.min( ax, bx, cx ) - D, hx = Math.max( ax, bx, cx ) + D, ly = Math.min( ay, by, cy ) - D, hy = Math.max( ay, by, cy ) + D, lz = Math.min( az, bz, cz ) - D, hz = Math.max( az, bz, cz ) + D;
			const x0 = Math.floor( lx * Hs ), x1 = Math.floor( hx * Hs ), y0 = Math.floor( ly * Hs ), y1 = Math.floor( hy * Hs ), z0 = Math.floor( lz * Hs ), z1 = Math.floor( hz * Hs );
			scan: for ( let z = z0; z <= z1; z ++ ) for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) {

				const list = pg.get( x + 1 + S * ( y + 1 + S * ( z + 1 ) ) );
				if ( ! list ) continue;
				for ( const p of list ) {

					const px = pts[ p ], py = pts[ p + 1 ], pz = pts[ p + 2 ];
					if ( px < lx || px > hx || py < ly || py > hy || pz < lz || pz > hz ) continue;
					if ( pointTri2( px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz ) < D2 ) { hit = true; break scan; }

				}

			}

		}

		if ( ! hit ) continue;
		tris ++;
		for ( const i of [ a, b, c ] ) if ( ! lock[ i ] ) { lock[ i ] = 1; verts ++; }

	}

	const st = guardStat[ D ] || ( guardStat[ D ] = { glow: 0, tris: 0, verts: 0, of: 0 } );
	st.glow += G.length; st.tris += tris; st.verts += verts; st.of += O.length;

}

// OPT.landmark_boxes: is the box b (min x y z, max x y z, world metres) of a loose part inside one of them
const LMK = ( OPT.landmark_boxes || [] ).filter( ( b ) => b && b.length >= 4 );
const LMK_PAD = 0.25;
const inLandmark = ( b, boxes ) => boxes.some( ( k ) => b[ 0 ] > k[ 0 ] - LMK_PAD && b[ 2 ] > k[ 1 ] - LMK_PAD && b[ 3 ] < k[ 2 ] + LMK_PAD && b[ 5 ] < k[ 3 ] + LMK_PAD );
// a glowing surface of the bake: aux 65535, a sign's 65533 (the page draws signs brighter: `signs` of the manifest)
function GLOWS( a ) { return a === 65535 || a === 65533; }
const lmk = { lo: 0, loIn: 0, cells: [], unpull: 0, stat: [] }; // triangles of the landmarks in the lowest level, of how many; the cells they are in

// the opaque triangles of the raw level m that lie inside the box k (all three corners; the ground does not count)
function landmarkTris( m, k ) {

	if ( ! ( k[ 0 ] < m.max[ 0 ] && k[ 2 ] > m.min[ 0 ] && k[ 1 ] < m.max[ 2 ] && k[ 3 ] > m.min[ 2 ] ) ) return 0;
	const w = m.w || 10, q = m.q, x0 = m.min[ 0 ], z0 = m.min[ 2 ];
	const b = fs.readFileSync( path.join( RAW, m.raw ) );
	const ab = b.buffer.slice( b.byteOffset, b.byteOffset + b.byteLength );
	const vb = new Uint16Array( ab, 0, m.nv * w ), ib = new Uint32Array( ab, m.nv * w * 2, m.ni ), flags = new Uint8Array( ab, m.nv * w * 2 + m.ni * 4, m.nv );
	const inside = ( i ) => { const x = x0 + vb[ i * w ] * q, z = z0 + vb[ i * w + 2 ] * q; return x > k[ 0 ] - LMK_PAD && z > k[ 1 ] - LMK_PAD && x < k[ 2 ] + LMK_PAD && z < k[ 3 ] + LMK_PAD; };
	let n = 0;
	for ( const [ kind, start, count ] of m.sub ) {

		if ( kind !== K_OPAQUE ) continue;
		for ( let t = start; t < start + count; t += 3 ) if ( ! ( flags[ ib[ t ] ] & 4 ) && inside( ib[ t ] ) && inside( ib[ t + 1 ] ) && inside( ib[ t + 2 ] ) ) n ++;

	}

	return n;

}

// OPT.landmark_src: the cell of each landmark that is large enough - the inner cell with most triangles of its LOD2
// geometry inside the box (a model is baked into one cell whole, and what stands in front of it is small beside it)
function landmarkCells() {

	const of = new Map();
	for ( const k of LMK ) {

		let best = 0, cell = null;
		for ( const ch of man.chunks ) {

			if ( ch.outer ) continue;
			const n = landmarkTris( ch.mid, k );
			if ( n > best ) { best = n; cell = ch.k; }

		}

		if ( cell ) { if ( ! of.has( cell ) ) of.set( cell, [] ); of.get( cell ).push( k ); }

	}

	return of;

}

// OPT.holes: per triangle of idx the squared distance to the nearest of the points pts (xyz, the cell's metres) where
// that is under R (Infinity otherwise); `near`, if given: per point the smallest squared distance found, kept if smaller
function nearPoints( vb, idx, m, pts, R, near = null ) {

	const H = 1 / R, S = 8192, q = m.q, R2 = R * R, np = pts.length / 3;
	const out = new Float32Array( idx.length / 3 ).fill( Infinity ), grid = new Map();
	for ( let i = 0; i < np; i ++ ) {

		const k = Math.floor( pts[ i * 3 ] * H ) + 1 + S * ( Math.floor( pts[ i * 3 + 1 ] * H ) + 1 + S * ( Math.floor( pts[ i * 3 + 2 ] * H ) + 1 ) ), l = grid.get( k );
		if ( l ) l.push( i ); else grid.set( k, [ i ] );

	}

	const X = ( i, c ) => vb[ i * W + c ] * q;
	for ( let t = 0; t < idx.length; t += 3 ) {

		const a = idx[ t ], b = idx[ t + 1 ], c = idx[ t + 2 ];
		const ax = X( a, 0 ), ay = X( a, 1 ), az = X( a, 2 ), bx = X( b, 0 ), by = X( b, 1 ), bz = X( b, 2 ), cx = X( c, 0 ), cy = X( c, 1 ), cz = X( c, 2 );
		const lx = Math.min( ax, bx, cx ) - R, ly = Math.min( ay, by, cy ) - R, lz = Math.min( az, bz, cz ) - R, hx = Math.max( ax, bx, cx ) + R, hy = Math.max( ay, by, cy ) + R, hz = Math.max( az, bz, cz ) + R;
		const x0 = Math.floor( lx * H ), x1 = Math.floor( hx * H ), y0 = Math.floor( ly * H ), y1 = Math.floor( hy * H ), z0 = Math.floor( lz * H ), z1 = Math.floor( hz * H );
		let best = Infinity;
		const at = ( i ) => {

			const d = pointTri2( pts[ i * 3 ], pts[ i * 3 + 1 ], pts[ i * 3 + 2 ], ax, ay, az, bx, by, bz, cx, cy, cz );
			if ( d >= R2 ) return;
			if ( d < best ) best = d;
			if ( near && d < near[ i ] ) near[ i ] = d;

		};

		if ( ( x1 - x0 + 1 ) * ( y1 - y0 + 1 ) * ( z1 - z0 + 1 ) > np ) {

			// (a large triangle: the points one by one)
			for ( let i = 0; i < np; i ++ ) { const x = pts[ i * 3 ], y = pts[ i * 3 + 1 ], z = pts[ i * 3 + 2 ]; if ( x > lx && x < hx && y > ly && y < hy && z > lz && z < hz ) at( i ); }

		} else for ( let z = z0; z <= z1; z ++ ) for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) {

			const l = grid.get( x + 1 + S * ( y + 1 + S * ( z + 1 ) ) );
			if ( l ) for ( const i of l ) at( i );

		}

		out[ t / 3 ] = best;

	}

	return out;

}

// OPT.lo_no_pull: the glowing vertices among the triangles `tris` whose card - the glowing faces joined to theirs -
// has a kept vertex (lk; not glowing, not ground) within D: added to `out`
function noPullMarks( vb, tris, m, flags, lk, D, out ) {

	const q = m.q, H = 1 / Math.max( D, 0.05 ), S = 8192, D2 = D * D;
	const X = ( i, c ) => vb[ i * W + c ] * q;
	const cell = ( x, y, z ) => x + 1 + S * ( y + 1 + S * ( z + 1 ) );
	const kept = new Map(), seen = new Uint8Array( m.nv ), G = [];
	for ( let t = 0; t < tris.length; t += 3 ) {

		const a = tris[ t ], b = tris[ t + 1 ], c = tris[ t + 2 ];
		if ( GLOWS( vb[ a * W + 9 ] ) && GLOWS( vb[ b * W + 9 ] ) && GLOWS( vb[ c * W + 9 ] ) ) { G.push( t ); continue; }
		for ( const i of [ a, b, c ] ) {

			if ( seen[ i ] ) continue;
			seen[ i ] = 1;
			if ( ! lk[ i ] || GLOWS( vb[ i * W + 9 ] ) || flags[ i ] & 4 ) continue;
			const k = cell( Math.floor( X( i, 0 ) * H ), Math.floor( X( i, 1 ) * H ), Math.floor( X( i, 2 ) * H ) ), l = kept.get( k );
			if ( l ) l.push( i ); else kept.set( k, [ i ] );

		}

	}

	if ( ! G.length || ! kept.size ) return;
	const key = new Map(), par = [];
	const find = ( a ) => { while ( par[ a ] !== a ) { par[ a ] = par[ par[ a ] ]; a = par[ a ]; } return a; };
	const id = ( i ) => { const k = vb[ i * W ] + vb[ i * W + 1 ] * 65536 + vb[ i * W + 2 ] * 4294967296; let w = key.get( k ); if ( w === undefined ) { w = par.length; par.push( w ); key.set( k, w ); } return w; };
	for ( const t of G ) { const a = find( id( tris[ t ] ) ); par[ find( id( tris[ t + 1 ] ) ) ] = a; par[ find( id( tris[ t + 2 ] ) ) ] = a; }
	const hit = new Set();
	for ( const t of G ) {

		const r = find( id( tris[ t ] ) );
		if ( hit.has( r ) ) continue;
		const a = tris[ t ], b = tris[ t + 1 ], c = tris[ t + 2 ];
		const ax = X( a, 0 ), ay = X( a, 1 ), az = X( a, 2 ), bx = X( b, 0 ), by = X( b, 1 ), bz = X( b, 2 ), cx = X( c, 0 ), cy = X( c, 1 ), cz = X( c, 2 );
		const x0 = Math.floor( ( Math.min( ax, bx, cx ) - D ) * H ), x1 = Math.floor( ( Math.max( ax, bx, cx ) + D ) * H );
		const y0 = Math.floor( ( Math.min( ay, by, cy ) - D ) * H ), y1 = Math.floor( ( Math.max( ay, by, cy ) + D ) * H );
		const z0 = Math.floor( ( Math.min( az, bz, cz ) - D ) * H ), z1 = Math.floor( ( Math.max( az, bz, cz ) + D ) * H );
		scan: for ( let z = z0; z <= z1; z ++ ) for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) {

			const l = kept.get( cell( x, y, z ) );
			if ( l ) for ( const i of l ) if ( pointTri2( X( i, 0 ), X( i, 1 ), X( i, 2 ), ax, ay, az, bx, by, bz, cx, cy, cz ) < D2 ) { hit.add( r ); break scan; }

		}

	}

	for ( const t of G ) if ( hit.has( find( id( tris[ t ] ) ) ) ) { out.add( tris[ t ] ); out.add( tris[ t + 1 ] ); out.add( tris[ t + 2 ] ); }

}

// OPT.landmark_stat: n points spread over the triangles `a` by area, each one's distance to the nearest of the
// triangles `d` -> [ rms, the 95th percentile, the largest ] in metres
function surfaceStat( vb, m, a, d, n = 3000 ) {

	const q = m.q, P = ( i, c ) => vb[ i * W + c ] * q;
	const area = new Float64Array( a.length / 3 );
	let all = 0;
	for ( let t = 0; t < a.length; t += 3 ) {

		const ux = P( a[ t + 1 ], 0 ) - P( a[ t ], 0 ), uy = P( a[ t + 1 ], 1 ) - P( a[ t ], 1 ), uz = P( a[ t + 1 ], 2 ) - P( a[ t ], 2 );
		const vx = P( a[ t + 2 ], 0 ) - P( a[ t ], 0 ), vy = P( a[ t + 2 ], 1 ) - P( a[ t ], 1 ), vz = P( a[ t + 2 ], 2 ) - P( a[ t ], 2 );
		all += area[ t / 3 ] = Math.hypot( uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx ) / 2;

	}

	if ( ! ( all > 0 ) || ! d.length ) return [ 0, 0, 0 ];
	// the triangles of d by cells of 1 m
	const S = 256, grid = new Map(), T = new Float64Array( d.length * 3 );
	for ( let t = 0; t < d.length; t += 3 ) {

		for ( let v = 0; v < 3; v ++ ) for ( let c = 0; c < 3; c ++ ) T[ ( t + v ) * 3 + c ] = P( d[ t + v ], c );
		const lo = [ 0, 1, 2 ].map( ( c ) => Math.floor( Math.min( T[ t * 3 + c ], T[ t * 3 + 3 + c ], T[ t * 3 + 6 + c ] ) ) );
		const hi = [ 0, 1, 2 ].map( ( c ) => Math.floor( Math.max( T[ t * 3 + c ], T[ t * 3 + 3 + c ], T[ t * 3 + 6 + c ] ) ) );
		for ( let z = lo[ 2 ]; z <= hi[ 2 ]; z ++ ) for ( let y = lo[ 1 ]; y <= hi[ 1 ]; y ++ ) for ( let x = lo[ 0 ]; x <= hi[ 0 ]; x ++ ) {

			const k = x + 1 + S * ( y + 1 + S * ( z + 1 ) ), l = grid.get( k );
			if ( l ) l.push( t ); else grid.set( k, [ t ] );

		}

	}

	let seed = 12345;
	const rnd = () => ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296;
	const out = new Float64Array( n );
	let t = 0, acc = area[ 0 ];
	for ( let i = 0; i < n; i ++ ) {

		const at = ( i + 0.5 ) / n * all;
		while ( acc < at && t < area.length - 1 ) acc += area[ ++ t ];
		let u = rnd(), v = rnd();
		if ( u + v > 1 ) { u = 1 - u; v = 1 - v; }
		const i0 = a[ t * 3 ], i1 = a[ t * 3 + 1 ], i2 = a[ t * 3 + 2 ];
		const p = [ 0, 1, 2 ].map( ( c ) => P( i0, c ) * ( 1 - u - v ) + P( i1, c ) * u + P( i2, c ) * v );
		let best = Infinity;
		// rings of cells about the point, until the nearest found is closer than the next ring can be
		for ( let r = 0; r < 64 && ( best === Infinity || Math.sqrt( best ) > r - 1 ); r ++ ) {

			const cx = Math.floor( p[ 0 ] ), cy = Math.floor( p[ 1 ] ), cz = Math.floor( p[ 2 ] );
			for ( let z = cz - r; z <= cz + r; z ++ ) for ( let y = cy - r; y <= cy + r; y ++ ) for ( let x = cx - r; x <= cx + r; x ++ ) {

				if ( Math.max( Math.abs( x - cx ), Math.abs( y - cy ), Math.abs( z - cz ) ) !== r ) continue;
				const l = grid.get( x + 1 + S * ( y + 1 + S * ( z + 1 ) ) );
				if ( l ) for ( const k of l ) {

					const o = k * 3;
					const e = pointTri2( p[ 0 ], p[ 1 ], p[ 2 ], T[ o ], T[ o + 1 ], T[ o + 2 ], T[ o + 3 ], T[ o + 4 ], T[ o + 5 ], T[ o + 6 ], T[ o + 7 ], T[ o + 8 ] );
					if ( e < best ) best = e;

				}

			}

		}

		out[ i ] = Math.sqrt( best );

	}

	const s = Array.from( out ).sort( ( x, y ) => x - y );
	return [ Math.sqrt( s.reduce( ( x, y ) => x + y * y, 0 ) / n ), s[ Math.floor( n * 0.95 ) ], s[ n - 1 ] ];

}

// a level below the baked one: the opaque range simplified (positions in metres; colour and uv steer which edges go).
// glass: kept or dropped; lanterns / props: their parts are simplified on their own (see OPT.lantern_error, prop_error);
// landmark: { error, ratio, guard, colour, border } for the loose parts inside OPT.landmark_boxes, likewise.
// (tried and dropped: an error bound per loose part, in proportion to its thickness - a third of the near level's
// triangles sit in parts under 0.3 m of ~8 triangles each, which no error bound makes any smaller)
function simplify( raw, m, ratio, error, { glass = false, glowLock = false, colour = OPT.colour_weight, prune = true, border = false, lanterns = false, props = false, ground = false, groundKeep = 0, guard = 0, landmark = null, noPull = 0, extra = [], holes = null } = {} ) {

	const { vb, ib } = raw;
	const nv = m.nv;
	const unp = noPull > 0 ? new Set() : null;
	let lock = raw.lock;
	if ( glowLock || ground || guard > 0 ) {

		lock = lock.slice();
		if ( glowLock ) for ( let i = 0; i < nv; i ++ ) if ( GLOWS( vb[ i * W + 9 ] ) ) lock[ i ] = 1;
		if ( ground ) for ( let i = 0; i < nv; i ++ ) if ( raw.flags[ i ] & 4 ) lock[ i ] = 1;
		if ( guard > 0 ) glowGuard( vb, ib, m, lock, guard );

	}

	// (NA attributes a vertex: colour, uv and - a bake with a second look - its colour, so that an edge either look needs stays)
	const c2 = OPT.colour2_weight > 0 ? col2( W ) : 0, NA = c2 ? 8 : 5;
	const wts = ( cw, u ) => c2 ? [ cw, cw, cw, u, u, cw * OPT.colour2_weight, cw * OPT.colour2_weight, cw * OPT.colour2_weight ] : [ cw, cw, cw, u, u ];
	const pos = new Float32Array( nv * 3 ), att = new Float32Array( nv * NA );
	const cs = man.colour_scale, ur = man.uv_range;
	for ( let i = 0; i < nv; i ++ ) {

		const o = i * W;
		pos[ i * 3 ] = vb[ o ] * m.q; pos[ i * 3 + 1 ] = vb[ o + 1 ] * m.q; pos[ i * 3 + 2 ] = vb[ o + 2 ] * m.q;
		for ( let c = 0; c < 3; c ++ ) {

			const s = vb[ o + 6 + c ] / 65535;
			att[ i * NA + c ] = Math.min( 1.6, Math.pow( s * s * cs, 0.4 ) ); // roughly how bright it ends up on screen
			if ( c2 ) { const t = vb[ o + c2 + c ] / 65535; att[ i * NA + 5 + c ] = Math.min( 1.6, Math.pow( t * t * cs, 0.4 ) ); }
		}

		att[ i * NA + 3 ] = ( vb[ o + 4 ] / 65536 - 0.5 ) * 2 * ur; att[ i * NA + 4 ] = ( vb[ o + 5 ] / 65536 - 0.5 ) * 2 * ur;

	}

	const parts = [], sub = [];
	let ni = 0;
	for ( const [ kind, start, count ] of m.sub ) {

		if ( kind === K_GLASS && ! glass ) continue;
		let idx = ib.slice( start, start + count );
		if ( kind === K_OPAQUE && count > 60 ) {

			const w = colour, u = OPT.uv_weight;
			const run1 = ( idx, ratio, error, flags, lk = lock, cw = w ) => idx.length <= 36 ? idx : MeshoptSimplifier.simplifyWithAttributes( idx, pos, 3, att, NA, wts( cw, u ), lk, Math.max( 36, Math.floor( idx.length * ratio / 3 ) * 3 ), error, [ 'ErrorAbsolute', ...flags ] )[ 0 ];
			const run = ( idx, ratio, error, flags, lk, cw ) => {

				if ( ! OPT.keep_whole || idx.length <= 36 || ! flags.includes( 'Prune' ) ) return run1( idx, ratio, error, flags, lk, cw );
				// (OPT.keep_whole: what must stay - not the ground, whose border is locked for another reason - is not pruned)
				const of = new Uint8Array( idx.length / 3 );
				let n = 0;
				for ( let t = 0; t < idx.length; t += 3 ) { const f = raw.flags[ idx[ t ] ] & raw.flags[ idx[ t + 1 ] ] & raw.flags[ idx[ t + 2 ] ]; if ( ( f & 1 ) && ! ( f & 4 ) ) { of[ t / 3 ] = 1; n ++; } }
				if ( ! n ) return run1( idx, ratio, error, flags, lk, cw );
				const [ whole, rest ] = split( idx, of, ( r ) => r > 0 ), d = run1( rest, ratio, error, flags, lk, cw );
				const out = new Uint32Array( whole.length + d.length );
				out.set( d ); out.set( whole, d.length ); keptWhole += n;
				return out;

			};
			const flags = [ ...( prune ? [ 'Prune' ] : [] ), ...( border ? [ 'LockBorder' ] : [] ), ...extra ];
			const done = [];
			if ( groundKeep ) {

				// the ground on its own: as it is, or simplified by where its light changes (its outline stays: the cells meet there)
				const of = new Int32Array( idx.length / 3 );
				for ( let t = 0; t < idx.length; t += 3 ) of[ t / 3 ] = raw.flags[ idx[ t ] ] & raw.flags[ idx[ t + 1 ] ] & raw.flags[ idx[ t + 2 ] ] & 4;
				const [ a, rest ] = split( idx, of, ( r ) => r > 0 );
				if ( a.length ) {

					const g = OPT.lo_ground_colour;
					const d = groundKeep === 2 && a.length > 36 ? MeshoptSimplifier.simplifyWithAttributes( a, pos, 3, att, NA, wts( g, u ), lock, 36, OPT.lo_ground_error, [ 'ErrorAbsolute', 'LockBorder' ] )[ 0 ] : a;
					done.push( d ); idx = rest; gnd.loIn += a.length / 3; gnd.lo += d.length / 3;

				}

			}

			if ( holes && idx.length ) {

				// OPT.holes: the places where the census saw through this level
				const R2 = OPT.holes_reach ** 2, LR = Math.max( OPT.holes_reach, OPT.holes_lock );
				const d2 = nearPoints( vb, idx, m, holes.pts, LR, holes.src );
				if ( OPT.holes === 1 ) {

					const lp = looseParts( vb, idx, m ), whole = new Set();
					for ( let t = 0; t < d2.length; t ++ ) {

						if ( ! ( d2[ t ] < R2 ) || whole.has( lp.of[ t ] ) ) continue;
						const b = lp.parts.get( lp.of[ t ] );
						if ( b[ 6 ] / 3 <= OPT.holes_tris && Math.max( b[ 3 ] - b[ 0 ], b[ 4 ] - b[ 1 ], b[ 5 ] - b[ 2 ] ) <= OPT.holes_size ) whole.add( lp.of[ t ] );

					}

					lock = lock.slice();
					for ( let t = 0; t < d2.length; t ++ ) if ( d2[ t ] < Infinity && ! whole.has( lp.of[ t ] ) ) for ( let v = 0; v < 3; v ++ ) { const i = idx[ t * 3 + v ]; if ( ! lock[ i ] ) { lock[ i ] = 1; holes.stat.locked ++; } }
					if ( whole.size ) {

						const [ a, rest ] = split( idx, lp.of, ( r ) => whole.has( r ) );
						done.push( a ); idx = rest; holes.stat.parts += whole.size; holes.stat.whole += a.length / 3;

					}

				}

			}

			if ( lanterns && OPT.lantern_error > 0 ) {

				const lp = looseParts( vb, idx, m ), mine = lanternParts( lp, m );
				const [ a, rest ] = split( idx, lp.of, ( r ) => mine.has( r ) );
				if ( a.length ) { const d = run( a, OPT.lantern_ratio, OPT.lantern_error, [] ); done.push( d ); idx = rest; lanternTris += d.length / 3; }

			}

			if ( landmark && landmark.error > 0 && idx.length > 36 ) {

				const lp = looseParts( vb, idx, m ), mine = new Set();
				for ( const [ r, b ] of lp.parts ) if ( inLandmark( b, landmark.boxes ) ) mine.add( r );
				const [ a, rest ] = split( idx, lp.of, ( r ) => mine.has( r ) );
				if ( a.length ) {

					let lk = lock;
					if ( landmark.guard > 0 ) { lk = lock.slice(); glowGuard( vb, ib, m, lk, landmark.guard ); }
					const cw = landmark.colour === undefined ? w : landmark.colour, lf = [ ...( landmark.border ? [ 'LockBorder' ] : [] ), ...extra ];
					let d;
					if ( landmark.floor > 0 ) {

						// part by part: a share of the budget by the root of the area, and a floor
						const tris = new Map();
						for ( let t = 0; t < idx.length; t += 3 ) {

							const r = lp.of[ t / 3 ];
							if ( ! mine.has( r ) ) continue;
							const l = tris.get( r );
							if ( l ) l.push( idx[ t ], idx[ t + 1 ], idx[ t + 2 ] ); else tris.set( r, [ idx[ t ], idx[ t + 1 ], idx[ t + 2 ] ] );

						}

						let roots = 0;
						for ( const r of tris.keys() ) roots += Math.sqrt( lp.parts.get( r )[ 8 ] );
						const out = [];
						let n = 0;
						for ( const [ r, l ] of tris ) {

							const want = Math.max( landmark.floor, Math.round( a.length / 3 * landmark.ratio * Math.sqrt( lp.parts.get( r )[ 8 ] ) / Math.max( roots, 1e-9 ) ) ) * 3;
							const o = l.length <= want ? l : MeshoptSimplifier.simplifyWithAttributes( new Uint32Array( l ), pos, 3, att, NA, wts( cw, u ), lk, want, landmark.error, [ 'ErrorAbsolute', ...lf ] )[ 0 ];
							out.push( o ); n += o.length;

						}

						d = new Uint32Array( n );
						let at = 0;
						for ( const o of out ) { d.set( o, at ); at += o.length; }

					} else d = run( a, landmark.ratio, landmark.error, [ ...( prune ? [ 'Prune' ] : [] ), ...lf ], lk, cw );
					if ( unp ) noPullMarks( vb, a, m, raw.flags, lk, noPull, unp );
					if ( OPT.landmark_stat ) {

						const st = surfaceStat( vb, m, a, d ), left = new Set();
						const part = new Map(); // (a vertex -> its loose part)
						for ( let t = 0; t < idx.length; t += 3 ) if ( mine.has( lp.of[ t / 3 ] ) ) part.set( idx[ t ], lp.of[ t / 3 ] );
						for ( let t = 0; t < d.length; t += 3 ) for ( let v = 0; v < 3; v ++ ) if ( part.has( d[ t + v ] ) ) { left.add( part.get( d[ t + v ] ) ); break; }
						let goneArea = 0, allArea = 0;
						for ( const r of mine ) { allArea += lp.parts.get( r )[ 8 ]; if ( ! left.has( r ) ) goneArea += lp.parts.get( r )[ 8 ]; }
						lmk.stat.push( `${ a.length / 3 } -> ${ d.length / 3 } triangles, ${ mine.size - left.size } of ${ mine.size } loose parts gone (${ ( goneArea / Math.max( allArea, 1e-9 ) * 100 ).toFixed( 1 ) } % of the area); source to lowest level: rms ${ ( st[ 0 ] * 100 ).toFixed( 1 ) } cm, p95 ${ ( st[ 1 ] * 100 ).toFixed( 1 ) } cm, max ${ ( st[ 2 ] * 100 ).toFixed( 1 ) } cm` );

					}

					done.push( d ); idx = rest; lmk.lo += d.length / 3; lmk.loIn += a.length / 3;

				}

			}

			if ( props && OPT.prop_error > 0 ) {

				const of = new Int32Array( idx.length / 3 );
				for ( let t = 0; t < idx.length; t += 3 ) of[ t / 3 ] = raw.flags[ idx[ t ] ] & raw.flags[ idx[ t + 1 ] ] & raw.flags[ idx[ t + 2 ] ] & 2;
				const [ a, rest ] = split( idx, of, ( r ) => r > 0 );
				if ( a.length ) { const d = run( a, OPT.prop_ratio, OPT.prop_error, [] ); done.push( d ); idx = rest; propTris += d.length / 3; propIn += a.length / 3; }

			}

			if ( unp ) noPullMarks( vb, idx, m, raw.flags, lock, noPull, unp );
			done.push( run( idx, ratio, error, flags ) );

			idx = new Uint32Array( done.reduce( ( n, d ) => n + d.length, 0 ) );
			let at = 0;
			for ( const d of done ) { idx.set( d, at ); at += d.length; }
			if ( holes ) nearPoints( vb, idx, m, holes.pts, 0.3, holes.got );

		}

		if ( ! idx.length ) continue;
		parts.push( idx ); sub.push( [ kind, ni, idx.length ] ); ni += idx.length;

	}

	if ( holes ) {

		// (of the places whose source has the surface, within 3 cm: where the level has it as near, after this)
		const st = holes.stat;
		for ( let i = 0; i < holes.src.length; i ++ ) { st.places ++; if ( holes.src[ i ] < 0.0009 ) { st.src ++; const g = Math.sqrt( holes.got[ i ] ); if ( g < 0.03 ) st.got ++; else if ( g < 0.06 ) st.g6 ++; else if ( g < 0.1 ) st.g10 ++; else if ( g < 0.3 ) st.g30 ++; } }

	}

	const I = new Uint32Array( ni ); let o = 0;
	for ( const p of parts ) { I.set( p, o ); o += p.length; }
	return { ib: I, sub, unpull: unp };

}

// The ground of a cell, the same at every level: the vertices of the terrain of `raw` (a level of w u16 per vertex)
// take colour and gloss of the fine level's vertex at the same place and of the same layer (where two share a place -
// the top and the side of a kerb - the one nearest in colour: the levels differ by a few percent, a kerb's faces by more)
const gnd = { verts: 0, found: 0, moved: 0, lo: 0, loIn: 0 };
function groundKey( vb, o, m ) {

	const g = ( k ) => Math.round( ( m.min[ k ] + vb[ o + k ] * m.q ) * 1024 );
	return g( 0 ) + '_' + g( 1 ) + '_' + g( 2 ) + '_' + vb[ o + 3 ];

}

function groundMap( raw, m, w ) {

	const map = new Map();
	for ( let i = 0; i < m.nv; i ++ ) {

		if ( ! ( raw.flags[ i ] & 4 ) ) continue;
		const k = groundKey( raw.vb, i * w, m ), a = map.get( k );
		if ( a ) a.push( i ); else map.set( k, [ i ] );

	}

	return map;

}

function groundFrom( map, fine, wf, raw, m, w ) {

	const A = fine.vb, B = raw.vb;
	for ( let i = 0; i < m.nv; i ++ ) {

		if ( ! ( raw.flags[ i ] & 4 ) ) continue;
		gnd.verts ++;
		const o = i * w, c = map.get( groundKey( B, o, m ) );
		if ( ! c ) continue;
		let best = - 1, bd = Infinity;
		for ( const j of c ) {

			const p = j * wf, d = Math.abs( A[ p + 6 ] - B[ o + 6 ] ) + Math.abs( A[ p + 7 ] - B[ o + 7 ] ) + Math.abs( A[ p + 8 ] - B[ o + 8 ] ) + Math.abs( A[ p + 9 ] - B[ o + 9 ] );
			if ( d < bd ) { bd = d; best = p; }

		}

		gnd.found ++;
		if ( bd > 0 ) gnd.moved ++;
		for ( let k = 6; k < 10; k ++ ) B[ o + k ] = A[ best + k ];
		const ca = col2( wf ), cb = col2( w ); // (a second look: its colour there too)
		if ( ca && cb ) for ( let k = 0; k < 3; k ++ ) B[ o + cb + k ] = A[ best + ca + k ];

	}

}

// the opaque range of a simplified level as clusters of loose parts, each in order of part size (OPT.ring_sizes):
// -> { ib, sub } where that range is a run of [ kind, start, count, size, cluster ] ranges, "all parts of the cluster
// of at least this size end here". Cluster 0 holds what spans the cell (ground, long walls); the others are the
// rest cut in OPT.ring_clusters lumps of about as many triangles, side by side, so each is culled and drawn down to
// its smallest parts on its own
function ordered( raw, m, ms, P ) {

	const S = P.sizes, at = ms.sub.findIndex( ( s ) => s[ 0 ] === K_OPAQUE );
	if ( at < 0 ) return ms;
	const [ , start, count ] = ms.sub[ at ];
	const idx = ms.ib.subarray( start, start + count );
	const lp = looseParts( raw.vb, idx, m ), keep = lanternParts( lp, m );
	for ( let t = 0; t < idx.length; t += 3 ) for ( let v = 0; v < 3; v ++ ) {

		const i = idx[ t + v ];
		if ( GLOWS( raw.vb[ i * W + 9 ] ) || raw.lock[ i ] ) keep.add( lp.of[ t / 3 ] ); // lit (a bulb, a pane) or marked to stay

	}

	const bucket = new Map(), cluster = new Map(), small = [];
	for ( const [ r, b ] of lp.parts ) {

		const size = Math.hypot( b[ 3 ] - b[ 0 ], b[ 4 ] - b[ 1 ], b[ 5 ] - b[ 2 ] );
		bucket.set( r, keep.has( r ) ? 0 : size < P.min_part ? - 1 : S.findIndex( ( s ) => size >= s ) );
		cluster.set( r, 0 );
		if ( Math.hypot( b[ 3 ] - b[ 0 ], b[ 5 ] - b[ 2 ] ) < P.span && bucket.get( r ) >= 0 ) small.push( [ r, ( b[ 0 ] + b[ 3 ] ) / 2, ( b[ 2 ] + b[ 5 ] ) / 2, b[ 6 ] ] );

	}

	let G = 1;
	const cut = ( list, n ) => {

		if ( n < 2 ) { for ( const p of list ) cluster.set( p[ 0 ], G ); G ++; return; }
		let x0 = Infinity, x1 = - Infinity, z0 = Infinity, z1 = - Infinity, all = 0;
		for ( const p of list ) { x0 = Math.min( x0, p[ 1 ] ); x1 = Math.max( x1, p[ 1 ] ); z0 = Math.min( z0, p[ 2 ] ); z1 = Math.max( z1, p[ 2 ] ); all += p[ 3 ]; }
		const c = x1 - x0 > z1 - z0 ? 1 : 2;
		list.sort( ( a, b ) => a[ c ] - b[ c ] );
		let i = 0, sum = 0;
		while ( i < list.length - 1 && sum + list[ i ][ 3 ] / 2 < all / 2 ) sum += list[ i ++ ][ 3 ];
		cut( list.slice( 0, i ), n / 2 ); cut( list.slice( i ), n / 2 );

	};

	const loose = small.reduce( ( n, p ) => n + p[ 3 ], 0 ) / 3;
	const lumps = P.cluster_tris ? Math.min( P.clusters, 2 ** Math.ceil( Math.log2( Math.max( 1, loose / P.cluster_tris ) ) ) ) : P.clusters;
	if ( lumps > 1 && loose >= P.cluster_min ) cut( small, lumps );
	const out = Array.from( { length: G }, () => S.map( () => [] ) );
	for ( let t = 0; t < idx.length; t += 3 ) {

		const r = lp.of[ t / 3 ], b = bucket.get( r );
		if ( b >= 0 ) out[ cluster.get( r ) ][ b ].push( idx[ t ], idx[ t + 1 ], idx[ t + 2 ] );

	}

	const parts = [], sub = [];
	let ni = 0;
	ms.sub.forEach( ( [ kind, start, count ], i ) => {

		if ( i !== at ) { parts.push( ms.ib.subarray( start, start + count ) ); sub.push( [ kind, ni, count ] ); ni += count; return; }
		out.forEach( ( steps, g ) => { if ( steps.some( ( o ) => o.length ) ) steps.forEach( ( o, b ) => { parts.push( o ); sub.push( [ kind, ni, o.length, S[ b ], g ] ); ni += o.length; } ); } );

	} );
	const I = new Uint32Array( ni ); let o = 0;
	for ( const p of parts ) { I.set( p, o ); o += p.length; }
	return { ib: I, sub };

}

let lanternTris = 0, propTris = 0, propIn = 0;
let keptWhole = 0; // (OPT.keep_whole: triangles of must-stay parts that a pruning simplification was not given)
// OPT.holes: the census (tools/pvs.py --holes), and per level what was done about it
const HOLES = OPT.holes && fs.existsSync( path.join( RAW, OPT.holes_file ) ) ? JSON.parse( fs.readFileSync( path.join( RAW, OPT.holes_file ), 'utf8' ) ) : null;
const holeStat = {};
if ( OPT.holes && ! HOLES ) console.log( '[webbake pack] holes: no census (.raw/holes.json, tools/pvs.py --holes): nothing mended' );
// the places of the cell `ch` at `level`, in the metres of its raw level m (the census: millimetres, the page's frame)
function holesOf( ch, level, m ) {

	const c = HOLES && String( OPT.holes_levels ).split( ',' ).includes( level ) && HOLES.levels[ level ] && HOLES.levels[ level ].cells[ ch.k ];
	if ( ! c || ! c.pts.length ) return null;
	const pts = new Float32Array( c.pts.length );
	for ( let i = 0; i < pts.length; i ++ ) pts[ i ] = c.pts[ i ] / 1000 - m.min[ i % 3 ];
	const n = pts.length / 3;
	return { pts, src: new Float32Array( n ).fill( Infinity ), got: new Float32Array( n ).fill( Infinity ), stat: holeStat[ level ] || ( holeStat[ level ] = { places: 0, src: 0, got: 0, g6: 0, g10: 0, g30: 0, parts: 0, whole: 0, locked: 0 } ) };

}
const REFIT = OPT.refit ? ( await import( './lsvb_pack.mjs' ) ).default( { OPT, RAW } ) : null; // the reduced levels' light fitted to the top level's (tools/lsvb_refit.py)
if ( OPT.lm ) {

	// the lightmap prototype (tools/next/lightmap.py): its meshes go through this file's simplify() and pack(); nothing
	// of the bake itself is read for writing or written
	const { default: lm } = await import( './lightmap_pack.mjs' );
	await lm( { OPT, man, RAW, OUT, readRaw, simplify, pack, groundMap, groundFrom, setW: ( w ) => { W = w; } } );
	process.exit( 0 );

}

for ( const f of fs.readdirSync( OUT ) ) if ( /^(n_|m_|r_|f_|c_).*\.bin$/.test( f ) || f === 'base.bin' ) fs.unlinkSync( path.join( OUT, f ) );

const t0 = Date.now();
const lmOwn = landmarkCells(), lmOf = new Map( OPT.landmark_src ? [ ...lmOwn ].map( ( [ k, b ] ) => [ k, b.filter( ( x ) => Math.max( x[ 2 ] - x[ 0 ], x[ 3 ] - x[ 1 ] ) >= OPT.landmark_src_min ) ] ).filter( ( [ , b ] ) => b.length ) : [] );
const base = fs.openSync( path.join( OUT, 'base.bin' ), 'w' );
let off = 0;
const tot = { lo: [ 0, 0, 0 ], mid: [ 0, 0, 0 ], near: [ 0, 0, 0 ], ring: [ 0, 0, 0 ], fine: [ 0, 0, 0 ] }; // tris, verts, bytes
const chunks = [];
for ( const ch of man.chunks ) {

	const c = { k: ch.k };
	if ( ch.outer ) c.outer = 1;
	W = ch.mid.w || 10;
	let raw = readRaw( ch.mid );
	const raws = { mid: raw };
	// (the landmarks that reach into this cell; is it the cell of one whose mid level is made from the near level)
	const lmBoxes = ch.outer ? [] : LMK.filter( ( k ) => k[ 0 ] < ch.mid.max[ 0 ] && k[ 2 ] > ch.mid.min[ 0 ] && k[ 1 ] < ch.mid.max[ 2 ] && k[ 3 ] > ch.mid.min[ 2 ] );
	const lmNear = lmOf.has( ch.k ) && !! ch.near;
	if ( lmNear ) lmk.cells.push( ch.k );
	if ( lmOwn.has( ch.k ) ) c.lmk = 1; // (the cell of a landmark: the page may hold its levels further out, ?lmk= of lite.js)
	if ( lmNear && ! OPT.ground_unify ) { W = ch.near.w || 10; raws.near = readRaw( ch.near ); W = ch.mid.w || 10; }
	if ( OPT.ground_unify && ch.near ) {

		// (from the top level the bake has: fine, or near in a bake without it)
		const wn = ch.near.w || 10, wm = W;
		W = wn; raws.near = readRaw( ch.near );
		if ( ch.fine ) {

			const wf = ch.fine.w || 10;
			W = wf; raws.fine = readRaw( ch.fine );
			const map = groundMap( raws.fine, ch.fine, wf );
			groundFrom( map, raws.fine, wf, raw, ch.mid, wm );
			groundFrom( map, raws.fine, wf, raws.near, ch.near, wn );

		} else groundFrom( groundMap( raws.near, ch.near, wn ), raws.near, wn, raw, ch.mid, wm );

		W = wm;

	}

	// OPT.landmark_src 2: from here on the cell's mid level is its near level's geometry (1: after the lowest level)
	if ( lmNear && OPT.landmark_src > 1 ) { ch.mid = ch.near; raw = raws.mid = raws.near; W = ch.mid.w || 10; }
	const s = simplify( raw, ch.mid, ch.outer ? OPT.outer_ratio : OPT.lo_ratio, ch.outer ? OPT.outer_error : OPT.lo_error, { glowLock: !! OPT.lo_glow_lock && ! ch.outer, glass: ! ch.outer, lanterns: ! ch.outer, props: ! ch.outer, groundKeep: ch.outer ? 0 : OPT.lo_ground, guard: ch.outer ? 0 : OPT.lo_glow_guard || 0, landmark: ! lmBoxes.length || ! ( OPT.landmark_lo_error > 0 ) ? null : { boxes: lmBoxes, error: OPT.landmark_lo_error, ratio: OPT.landmark_lo_ratio, guard: OPT.landmark_lo_guard || 0, colour: OPT.landmark_lo_colour, border: !! OPT.landmark_lo_border, floor: OPT.landmark_lo_floor || 0 }, noPull: ch.outer ? 0 : OPT.lo_no_pull || 0, extra: ch.outer || ! OPT.lo_flags ? [] : String( OPT.lo_flags ).split( ',' ), holes: holesOf( ch, 'lo', ch.mid ) } );
	if ( s.unpull && s.unpull.size ) {

		// (a copy: the mid level is packed from the same vertices)
		raw = Object.assign( {}, raw, { vb: raw.vb.slice() } );
		for ( const i of s.unpull ) raw.vb[ i * W + 9 ] = raw.vb[ i * W + 9 ] === 65533 ? 65532 : 65534; // (a sign stays a sign)
		lmk.unpull += s.unpull.size;

	}
	if ( OPT.landmark_stat ) while ( lmk.stat.length ) console.log( `[webbake pack] landmark parts of ${ ch.k }: ${ lmk.stat.shift() }` );
	const lo = pack( REFIT ? REFIT.vb( ch, 'lo', raw.vb, s.ib, s.sub ) : raw.vb, s.ib, s.sub, ch.mid );
	selfCheck( lo );
	fs.writeSync( base, lo.blob );
	c.lo = Object.assign( { off, len: lo.blob.byteLength }, lo.meta );
	off += lo.blob.byteLength;
	tot.lo[ 0 ] += lo.meta.ni / 3; tot.lo[ 1 ] += lo.meta.nv; tot.lo[ 2 ] += lo.blob.byteLength;
	if ( lmNear && ch.mid !== ch.near ) { ch.mid = ch.near; raws.mid = raws.near; }
	for ( const [ name, prefix, src ] of [ [ 'mid', 'm_', 'mid' ], [ 'near', 'n_', 'near' ], [ 'ring', 'r_', 'near' ], [ 'fine', 'f_', 'fine' ] ] ) {

		if ( ! ch[ src ] || ( ch.outer && ! OPT.outer_mid ) ) continue;
		if ( name === 'ring' && ! ( OPT.ring_error > 0 ) ) continue;
		W = ch[ src ].w || 10;
		const r = raws[ src ] || ( raws[ src ] = readRaw( ch[ src ] ) );
		if ( src === 'fine' ) raws.near = null;
		const lm = name === 'mid' && lmNear; // (the mid level of a landmark's cell, from LOD1: its own bound)
		const ms = OPT[ name + '_error' ] > 0 || lm
			? simplify( r, ch[ src ], lm ? OPT.landmark_mid_ratio : OPT[ name + '_ratio' ], lm ? OPT.landmark_mid_error : OPT[ name + '_error' ], { glass: true, glowLock: true, prune: !! ( lm ? OPT.landmark_mid_prune : OPT[ name + '_prune' ] ), border: !! OPT.lock_border, colour: OPT[ name + '_colour_weight' ], ground: !! OPT.ground_lock, guard: OPT[ name + '_glow_guard' ] || 0, holes: name === 'ring' || name === 'fine' ? null : holesOf( ch, name, ch[ src ] ) } )
			: { ib: r.ib, sub: ch[ src ].sub };
		const os = name === 'ring' && OPT.ring_sizes ? ordered( r, ch[ src ], ms, { sizes: OPT.ring_sizes, min_part: OPT.ring_min_part, clusters: OPT.ring_clusters, span: OPT.ring_span, cluster_min: OPT.ring_cluster_min } )
			: name === 'fine' && OPT.fine_sizes ? ordered( r, ch[ src ], ms, { sizes: OPT.fine_sizes, min_part: 0, clusters: OPT.fine_clusters, cluster_tris: OPT.fine_cluster_tris, span: OPT.fine_span, cluster_min: 0 } )
			: ms;
		const p = pack( REFIT ? REFIT.vb( ch, name, r.vb, os.ib, os.sub ) : r.vb, os.ib, os.sub, ch[ src ] );
		const file = prefix + ch.k + '.bin';
		fs.writeFileSync( path.join( OUT, file ), p.blob );
		c[ name ] = Object.assign( { file, len: p.blob.byteLength }, p.meta );
		tot[ name ][ 0 ] += p.meta.ni / 3; tot[ name ][ 1 ] += p.meta.nv; tot[ name ][ 2 ] += p.blob.byteLength;

	}

	chunks.push( c );

}

fs.closeSync( base );
man.chunks = chunks;
man.base = { file: 'base.bin', bytes: off };
man.codec = 'meshopt';
man.pack = Object.fromEntries( Object.entries( OPT ).filter( ( [ k ] ) => k !== 'keep_raw' ) );
for ( const k in tot ) man.stats[ k ] = Object.assign( man.stats[ k ] || {}, { tris: tot[ k ][ 0 ], verts: tot[ k ][ 1 ], bytes: tot[ k ][ 2 ] } );
man.baked = Math.floor( Date.now() / 1000 ); // the page asks for the files as ?v=<baked>, and a versioned file is cached for good
fs.writeFileSync( path.join( OUT, 'manifest.json' ), JSON.stringify( man ) );
if ( ! OPT.keep_raw ) fs.rmSync( RAW, { recursive: true, force: true } );
const f = ( t ) => `${ ( t[ 0 ] / 1e3 ).toFixed( 0 ) }k tris ${ ( t[ 1 ] / 1e3 ).toFixed( 0 ) }k verts ${ ( t[ 2 ] / 1e6 ).toFixed( 1 ) } MB`;
console.log( `[webbake pack] lo: ${ f( tot.lo ) } (lanterns ${ Math.round( lanternTris / 1e3 ) }k, props ${ Math.round( propTris / 1e3 ) }k of ${ Math.round( propIn / 1e3 ) }k) | mid: ${ f( tot.mid ) } | near: ${ f( tot.near ) } | ring: ${ f( tot.ring ) } | fine: ${ f( tot.fine ) } | ${ ( ( Date.now() - t0 ) / 1e3 ).toFixed( 1 ) }s` );
for ( const d in guardStat ) console.log( `[webbake pack] glow guard ${ d } m: ${ guardStat[ d ].tris } opaque triangles of ${ guardStat[ d ].of } pass that near one of ${ guardStat[ d ].glow } glowing ones; ${ guardStat[ d ].verts } more vertices kept` );
if ( OPT.clear_panes ) console.log( `[webbake pack] clear panes: ${ paneStat.verts } glass vertices (all levels) at ${ paneStat.heads.size } of ${ BURNERS.length } burners with a fitting (${ burnersOff } of them off their light)` );
if ( OPT.ground_unify ) console.log( `[webbake pack] ground: ${ gnd.verts } vertices of the mid and near levels, ${ gnd.found } with the top level's colour and gloss (${ gnd.moved } changed by it)` );
if ( keptWhole ) console.log( `[webbake pack] kept parts: ${ keptWhole } triangles with the bake's must-stay mark at all three corners took no part in pruning (all levels)` );
for ( const l in holeStat ) { const h = holeStat[ l ]; console.log( `[webbake pack] holes, ${ l }: ${ h.places } places seen through, ${ h.src } with the surface in the level's source (within 3 cm); the level has it as near at ${ h.got } of those, within 6 / 10 / 30 cm at ${ h.g6 } / ${ h.g10 } / ${ h.g30 } more${ OPT.holes === 1 ? ` (${ h.parts } loose parts kept as they are: ${ h.whole } triangles; ${ h.locked } vertices locked)` : ' (counted only)' }` ); }
if ( lmk.unpull ) console.log( `[webbake pack] lowest level: ${ lmk.unpull } glowing vertices with kept geometry within ${ OPT.lo_no_pull } m are marked as not to be pulled` );
if ( lmk.loIn || lmk.cells.length ) console.log( `[webbake pack] landmarks (${ LMK.length } boxes): ${ lmk.loIn ? `${ lmk.lo } triangles in the lowest level of ${ lmk.loIn } before` : 'as everything else in the lowest level' }${ lmk.cells.length ? `; mid level from the near level's geometry in ${ lmk.cells.join( ', ' ) }` : '' }` );
if ( OPT.lo_ground ) console.log( `[webbake pack] ground of the lowest level (inner cells): ${ gnd.lo } triangles of ${ gnd.loIn } at mid` );
