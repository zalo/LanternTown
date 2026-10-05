// The packer's side of the refit (tools/lsvb_refit.py): the light of the reduced levels fitted to the top level's.
//
// A reduced level is an index list over its raw level's vertices (simplify() in webbake_pack.mjs chooses which stay;
// the lowest level and mid are two lists over ONE vertex buffer), so its light can only be fitted once the list is
// known, and the two lists need a vertex buffer each. The packer therefore runs twice:
//
//   refit: 'sim'   it packs as ever and leaves every such list beside the raw bake: <cell>.<level>.sim.bin + sim.json
//   (tools/lsvb_refit.py fits each list's vertices to the top level's light: <cell>.<level>.fit.bin + fit.json)
//   refit: 1       every level that has a fit is packed with the fit's colours in place of the bake's
//
// A fit names the index list and the vertex buffer it was made for (sha1 of both): packed with other settings or
// another bake, the packer stops instead of putting one mesh's light on another. Nothing but the three colour slots
// of the vertices a fit lists is changed; without `refit` this file is not loaded.
// A bake with a second look (records of 14 / 18 words: a second colour at word 10 / 12) has six values a vertex in
// its fits, the second look's three after the first's; both go into their slots.
//
// The three places of webbake_pack.mjs:
//   const REFIT = OPT.refit ? ( await import( './lsvb_pack.mjs' ) ).default( { OPT, RAW } ) : null;
//   pack( REFIT ? REFIT.vb( ch, 'lo', raw.vb, s.ib, s.sub ) : raw.vb, s.ib, s.sub, ch.mid )
//   pack( REFIT ? REFIT.vb( ch, name, r.vb, os.ib, os.sub ) : r.vb, os.ib, os.sub, ch[ src ] )

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const MAGIC = 0x4656534c; // "LSVF"
const VERSION = 1;
const HEAD = 60; // magic, version, vertices of the raw level, indices, vertices listed (u32 each), sha1 of the index list, of the vertex buffer
const sha = ( a ) => crypto.createHash( 'sha1' ).update( new Uint8Array( a.buffer, a.byteOffset, a.byteLength ) ).digest();

export default function refit( { OPT, RAW } ) {

	const levels = String( OPT.refit_levels || 'lo,mid,near,ring' ).split( ',' );
	const SRC = { lo: 'mid', mid: 'mid', near: 'near', ring: 'near', fine: 'fine' };
	// the raw level a level's vertices really come from. The table, but for a landmark's cell (OPT.landmark_src of
	// webbake_pack.mjs): the packer puts the near level in the place of mid there (`ch.mid = ch.near`: after the lowest
	// level with landmark_src 1, before it with 2), and what it then makes of `mid` - the mid level, and with 2 the
	// lowest too - is an index list over the NEAR level's vertices. sim.json and the fit must name that level: the
	// fitter reads the vertices of the level named (and stops when their number is another)
	const srcOf = ( ch, name ) => SRC[ name ] === 'mid' && ch.near && ch.mid === ch.near ? 'near' : SRC[ name ];
	// the level a cell's light is taken from: the densest the bake has. What the packer makes of that level itself
	// (near and ring in a bake without fine) is never fitted: what is seen from closest stays as baked
	const topOf = ( ch ) => ch.fine ? 'fine' : ch.near ? 'near' : null;
	const stale = () => { for ( const f of fs.readdirSync( RAW ) ) if ( /\.(sim|fit)\.bin$/.test( f ) || f === 'sim.json' || f === 'fit.json' ) fs.unlinkSync( path.join( RAW, f ) ); };

	if ( OPT.refit === 'sim' ) {

		stale();
		const sim = { version: VERSION, levels: [] };
		process.on( 'exit', ( code ) => { if ( code === 0 ) fs.writeFileSync( path.join( RAW, 'sim.json' ), JSON.stringify( sim ) ); } );
		return { vb( ch, name, vb, ib, sub ) {

			const top = topOf( ch ), src = srcOf( ch, name );
			if ( top && src !== top && levels.includes( name ) && ch[ src ] ) {

				const file = `${ ch.k }.${ name }.sim.bin`;
				fs.writeFileSync( path.join( RAW, file ), new Uint8Array( ib.buffer, ib.byteOffset, ib.byteLength ) );
				sim.levels.push( { k: ch.k, name, src, top, nv: ch[ src ].nv, w: vb.length / ch[ src ].nv, ni: ib.length, sub, file, ib: sha( ib ).toString( 'hex' ), vb: sha( vb ).toString( 'hex' ) } );

			}

			return vb;

		} };

	}

	const fj = path.join( RAW, 'fit.json' );
	if ( ! fs.existsSync( fj ) ) throw new Error( 'lsvb_pack: `refit` is set and there is no fit beside the raw bake (.raw/fit.json): tools/lsvb_refit.py makes it - webbake.py runs it before this' );
	const fit = JSON.parse( fs.readFileSync( fj, 'utf8' ) );
	const stat = { levels: 0, verts: 0 };
	process.on( 'exit', ( code ) => { if ( code === 0 ) console.log( `[webbake pack] refit: ${ stat.levels } levels with the light fitted to their cell's top level, ${ stat.verts } vertices` ); } );
	return { vb( ch, name, vb, ib ) {

		const e = fit.levels[ ch.k + '.' + name ];
		if ( ! e ) return vb;
		const bad = ( why ) => { throw new Error( `lsvb_pack: ${ e.file } is not the fit of this ${ name } level of ${ ch.k } (${ why }): the bake or the packer's settings changed since - run tools/lsvb_refit.py again (webbake.py does)` ); };
		const b = fs.readFileSync( path.join( RAW, e.file ) );
		if ( b.length < HEAD ) bad( 'cut short' );
		const h = new Uint32Array( b.buffer.slice( b.byteOffset, b.byteOffset + 20 ) );
		const nv = ch[ srcOf( ch, name ) ].nv, n = h[ 4 ];
		if ( h[ 0 ] !== MAGIC || h[ 1 ] !== VERSION ) bad( 'not a fit file of this version' );
		const w = vb.length / nv, c2 = w === 14 ? 10 : w === 18 ? 12 : 0, C = c2 ? 6 : 3;
		if ( h[ 2 ] !== nv || h[ 3 ] !== ib.length || b.length !== HEAD + n * ( 4 + 2 * C ) ) bad( 'other sizes' );
		if ( ! b.subarray( 20, 40 ).equals( sha( ib ) ) ) bad( 'another index list' );
		if ( ! b.subarray( 40, 60 ).equals( sha( vb ) ) ) bad( 'other vertices' );
		const idx = new Uint32Array( b.buffer.slice( b.byteOffset + HEAD, b.byteOffset + HEAD + n * 4 ) );
		const col = new Uint16Array( b.buffer.slice( b.byteOffset + HEAD + n * 4, b.byteOffset + HEAD + n * ( 4 + 2 * C ) ) );
		const out = vb.slice();
		for ( let i = 0; i < n; i ++ ) {

			const o = idx[ i ] * w + 6, p = idx[ i ] * w + c2;
			out[ o ] = col[ i * C ]; out[ o + 1 ] = col[ i * C + 1 ]; out[ o + 2 ] = col[ i * C + 2 ];
			if ( c2 ) { out[ p ] = col[ i * C + 3 ]; out[ p + 1 ] = col[ i * C + 4 ]; out[ p + 2 ] = col[ i * C + 5 ]; }

		}

		stat.levels ++; stat.verts += n;
		return out;

	} };

}
