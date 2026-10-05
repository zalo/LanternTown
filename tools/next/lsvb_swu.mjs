// simplifyWithUpdate against simplifyWithAttributes on the mid level of the four sweep cells: the same locks, weights,
// error bound and triangle target; out: a private raw dir per variant that tools/lsvb_refit.py can judge / fit
//   node tools/next/lsvb_swu.mjs [colour weight] [tag]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { MeshoptSimplifier } from 'meshoptimizer';
await MeshoptSimplifier.ready;

const ROOT = path.resolve( path.dirname( new URL( import.meta.url ).pathname ), '..', '..' );
const KEPT = ROOT + '/out/web/_next/.raw', LM = ROOT + '/out/web/_next/lm/.raw', OUT = ROOT + '/.scratch/lsvb/swu';
const CELLS = [ 'c_1_0', 'c_-1_0', 'c_0_-1', 'c_1_-1' ];
const W = 10, CW = Number( process.argv[ 2 ] || 0.5 ), TAG = process.argv[ 3 ] || '', UW = 0.03, RATIO = 0.5, ERROR = 0.03;
const man = JSON.parse( fs.readFileSync( KEPT + '/raw.json', 'utf8' ) );
const cs = man.colour_scale, ur = man.uv_range;
const sha = ( a ) => crypto.createHash( 'sha1' ).update( new Uint8Array( a.buffer, a.byteOffset, a.byteLength ) ).digest( 'hex' );
const INPUT = { own: ( k ) => `${ KEPT }/${ k }.mid.bin`, rl: ( k ) => `${ LM }/${ k }.mid.rawrl.bin` };

for ( const input of [ 'own', 'rl' ] ) {

	let target = {}; // (the update variant aims at what the plain one reached: the same triangle count)
	for ( const method of [ 'att', 'upd' ] ) {

		const dir = `${ OUT }/${ input }_${ method }${ TAG }`;
		fs.mkdirSync( dir, { recursive: true } );
		const rj = JSON.parse( JSON.stringify( man ) ), sim = { version: 1, levels: [] };
		for ( const c of rj.chunks ) for ( const l of [ 'mid', 'near', 'fine' ] ) if ( c[ l ] ) c[ l ].raw = path.join( KEPT, c[ l ].raw );
		let tin = 0, tout = 0, moved = 0, recol = 0, maxd = 0;
		for ( const k of CELLS ) {

			const ch = rj.chunks.find( ( c ) => c.k === k ), m = ch.mid, nv = m.nv;
			const b = fs.readFileSync( INPUT[ input ]( k ) );
			const ab = b.buffer.slice( b.byteOffset, b.byteOffset + b.byteLength );
			const vb = new Uint16Array( ab, 0, nv * W ), ib = new Uint32Array( ab, nv * W * 2, m.ni ), flags = new Uint8Array( ab, nv * W * 2 + m.ni * 4, nv );
			const lock = new Uint8Array( nv );
			for ( let i = 0; i < nv; i ++ ) lock[ i ] = ( flags[ i ] & 5 ) || vb[ i * W + 9 ] === 65535 ? 1 : 0;
			for ( const [ kind, start, count ] of m.sub ) if ( kind !== 0 ) for ( let i = start; i < start + count; i ++ ) lock[ ib[ i ] ] = 1;
			const pos = new Float32Array( nv * 3 ), att = new Float32Array( nv * 5 );
			for ( let i = 0; i < nv; i ++ ) {

				const o = i * W;
				pos[ i * 3 ] = vb[ o ] * m.q; pos[ i * 3 + 1 ] = vb[ o + 1 ] * m.q; pos[ i * 3 + 2 ] = vb[ o + 2 ] * m.q;
				for ( let c = 0; c < 3; c ++ ) { const s = vb[ o + 6 + c ] / 65535; att[ i * 5 + c ] = Math.min( 1.6, Math.pow( s * s * cs, 0.4 ) ); }
				att[ i * 5 + 3 ] = ( vb[ o + 4 ] / 65536 - 0.5 ) * 2 * ur; att[ i * 5 + 4 ] = ( vb[ o + 5 ] / 65536 - 0.5 ) * 2 * ur;

			}

			const pos0 = pos.slice(), att0 = att.slice(), out = vb.slice();
			const parts = [], sub = [];
			let ni = 0;
			for ( const [ kind, start, count ] of m.sub ) {

				let idx = ib.slice( start, start + count );
				if ( kind === 0 ) {

					tin += count / 3;
					const want = method === 'upd' ? target[ k ] : Math.max( 36, Math.floor( count * RATIO / 3 ) * 3 );
					const args = [ idx, pos, 3, att, 5, [ CW, CW, CW, UW, UW ], lock, want, ERROR, [ 'ErrorAbsolute', 'LockBorder' ] ];
					if ( method === 'att' ) { idx = MeshoptSimplifier.simplifyWithAttributes( ...args )[ 0 ]; target[ k ] = idx.length; }
					else { const r = MeshoptSimplifier.simplifyWithUpdate( ...args ); idx = idx.slice( 0, r[ 0 ] ); }
					tout += idx.length / 3;

				}

				parts.push( idx ); sub.push( [ kind, ni, idx.length ] ); ni += idx.length;

			}

			const sib = new Uint32Array( ni );
			let at = 0;
			for ( const p of parts ) { sib.set( p, at ); at += p.length; }
			if ( method === 'upd' ) {

				const used = new Uint8Array( nv );
				for ( const i of sib ) used[ i ] = 1;
				for ( let i = 0; i < nv; i ++ ) {

					if ( ! used[ i ] ) continue;
					const o = i * W;
					let mv = false, rc = false;
					for ( let c = 0; c < 3; c ++ ) if ( pos[ i * 3 + c ] !== pos0[ i * 3 + c ] ) mv = true;
					for ( let c = 0; c < 3; c ++ ) if ( att[ i * 5 + c ] !== att0[ i * 5 + c ] ) rc = true;
					if ( mv ) {

						let d = 0;
						for ( let c = 0; c < 3; c ++ ) { out[ o + c ] = Math.max( 0, Math.min( 65535, Math.round( pos[ i * 3 + c ] / m.q ) ) ); d += ( pos[ i * 3 + c ] - pos0[ i * 3 + c ] ) ** 2; }
						moved ++; maxd = Math.max( maxd, Math.sqrt( d ) );

					}

					if ( rc ) {

						for ( let c = 0; c < 3; c ++ ) out[ o + 6 + c ] = Math.max( 0, Math.min( 65535, Math.round( Math.sqrt( Math.pow( Math.max( 0, att[ i * 5 + c ] ), 2.5 ) / cs ) * 65535 ) ) );
						recol ++;

					}

					for ( let c = 0; c < 2; c ++ ) if ( att[ i * 5 + 3 + c ] !== att0[ i * 5 + 3 + c ] ) out[ o + 4 + c ] = Math.max( 0, Math.min( 65535, Math.round( ( att[ i * 5 + 3 + c ] / ( 2 * ur ) + 0.5 ) * 65536 ) ) );

				}

			}

			const rawFile = `${ dir }/${ k }.mid.bin`;
			fs.writeFileSync( rawFile, Buffer.concat( [ Buffer.from( out.buffer, out.byteOffset, out.byteLength ), Buffer.from( ib.buffer, ib.byteOffset, ib.byteLength ), Buffer.from( flags.buffer, flags.byteOffset, flags.byteLength ) ] ) );
			m.raw = rawFile;
			const file = `${ k }.mid.sim.bin`;
			fs.writeFileSync( `${ dir }/${ file }`, new Uint8Array( sib.buffer, sib.byteOffset, sib.byteLength ) );
			sim.levels.push( { k, name: 'mid', src: 'mid', top: 'fine', nv, w: W, ni: sib.length, sub, file, ib: sha( sib ), vb: sha( out ) } );

		}

		fs.writeFileSync( dir + '/raw.json', JSON.stringify( rj ) );
		fs.writeFileSync( dir + '/sim.json', JSON.stringify( sim ) );
		console.log( `${ input }_${ method }${ TAG } colour weight ${ CW }: opaque triangles ${ tin } -> ${ tout }${ method === 'upd' ? `, ${ moved } vertices moved (the farthest ${ ( maxd * 100 ).toFixed( 1 ) } cm), ${ recol } recoloured` : '' }` );

	}

}
