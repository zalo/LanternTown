// Lightmap prototype, the two node stages (run through tools/next/webbake_pack.mjs with the option `lm`, which hands
// over its own functions: the meshes are simplified and packed by exactly the code of the bake).
//
//   stage "dump"  the levels of the prototype cells in their geometry variants, simplified as the packer would (or
//                 harder: no colour in the error, the ground free), written unpacked to <lm>/.raw/<cell>.<level>.<variant>.bin
//                 (the raw layout: 10 x u16 per vertex, u32 indices, one flag byte per vertex) for tools/next/lightmap_bl.py;
//                 each also packed with its vertex light as the bundle <level>_<variant>_v
//   stage "pack"  the meshes the Blender step wrote (light in pages, uv2 in the colour slots) packed as the bundles
//                 <lm>/<bundle>/<cell>.bin + <lm>/<bundle>.json; and the "va" meshes (vertex light taken from the
//                 full model's surface) simplified with a high colour weight
//
// Nothing here writes outside <out>/lm.

import fs from 'node:fs';
import path from 'node:path';

export default async function lm( { OPT, man, RAW, OUT, readRaw, simplify, pack, groundMap, groundFrom, setW } ) {

	const S = OPT.lm;
	const LM = path.join( OUT, 'lm' ), LRAW = path.join( LM, '.raw' );
	fs.mkdirSync( LRAW, { recursive: true } );
	const bits = OPT.colour_bits;
	const roundColours = ( vb, nv ) => {

		// what pack() does to the lit colour - but not to a vertex whose colour slots hold its place in a page
		const s = 1 << ( 16 - bits ), top = 65536 - s;
		for ( let i = 0; i < nv; i ++ ) if ( vb[ i * 10 + 3 ] < 32768 ) for ( let c = 6; c < 9; c ++ ) vb[ i * 10 + c ] = Math.min( top, Math.round( vb[ i * 10 + c ] / s ) * s );

	};

	const compact = ( vb, flags, s ) => {

		// the vertices a simplified index list still uses, in first-use order
		const map = new Int32Array( vb.length / 10 ).fill( - 1 ); let nv = 0;
		const ib = new Uint32Array( s.ib.length );
		for ( let i = 0; i < ib.length; i ++ ) { const v = s.ib[ i ]; if ( map[ v ] < 0 ) map[ v ] = nv ++; ib[ i ] = map[ v ]; }
		const V = new Uint16Array( nv * 10 ), F = new Uint8Array( nv );
		for ( let v = 0; v < map.length; v ++ ) if ( map[ v ] >= 0 ) { V.set( vb.subarray( v * 10, v * 10 + 10 ), map[ v ] * 10 ); F[ map[ v ] ] = flags[ v ]; }
		return { vb: V, ib, flags: F, nv };

	};

	const write = ( file, m ) => fs.writeFileSync( file, Buffer.concat( [ Buffer.from( m.vb.buffer, m.vb.byteOffset, m.vb.byteLength ), Buffer.from( m.ib.buffer, m.ib.byteOffset, m.ib.byteLength ), Buffer.from( m.flags.buffer, m.flags.byteOffset, m.flags.byteLength ) ] ) );
	const read = ( file, nv, ni ) => {

		const b = fs.readFileSync( file );
		const ab = b.buffer.slice( b.byteOffset, b.byteOffset + b.byteLength );
		const flags = new Uint8Array( ab, nv * 20 + ni * 4, nv );
		return { vb: new Uint16Array( ab, 0, nv * 10 ), ib: new Uint32Array( ab, nv * 20, ni ), flags, lock: flags.map( ( f ) => f & 1 ) };

	};

	// the opaque vertices that differ in nothing but their baked light made one (it goes into a page: what is left to
	// split a vertex is its texture layer and uv), so that the simplifier meets no seam where only the light changed
	const weld = ( raw, m ) => {

		const { vb, ib, flags } = raw, nv = m.nv;
		const plain = new Uint8Array( nv );
		for ( const [ kind, start, count ] of m.sub ) if ( kind === 0 ) for ( let i = start; i < start + count; i ++ ) plain[ ib[ i ] ] = 1;
		for ( const [ kind, start, count ] of m.sub ) if ( kind !== 0 ) for ( let i = start; i < start + count; i ++ ) plain[ ib[ i ] ] = 0;
		const seen = new Map(), map = new Int32Array( nv ); let n = 0;
		const first = [];
		for ( let v = 0; v < nv; v ++ ) {

			const o = v * 10;
			const key = plain[ v ] && vb[ o + 9 ] !== 65535 ? `${ vb[ o ] },${ vb[ o + 1 ] },${ vb[ o + 2 ] },${ vb[ o + 3 ] },${ vb[ o + 4 ] },${ vb[ o + 5 ] }` : 'v' + v;
			let i = seen.get( key );
			if ( i === undefined ) { i = n ++; seen.set( key, i ); first.push( v ); }
			map[ v ] = i;

		}

		const V = new Uint16Array( n * 10 ), F = new Uint8Array( n ), acc = new Float64Array( n * 5 );
		for ( let i = 0; i < n; i ++ ) V.set( vb.subarray( first[ i ] * 10, first[ i ] * 10 + 10 ), i * 10 );
		for ( let v = 0; v < nv; v ++ ) {

			const i = map[ v ]; F[ i ] |= flags[ v ];
			for ( let c = 0; c < 4; c ++ ) acc[ i * 5 + c ] += vb[ v * 10 + 6 + c ];
			acc[ i * 5 + 4 ] ++;

		}

		// (the light a welded vertex keeps for itself: the mean - used where a texel finds nothing on the full model)
		for ( let i = 0; i < n; i ++ ) if ( acc[ i * 5 + 4 ] > 1 ) for ( let c = 0; c < 4; c ++ ) V[ i * 10 + 6 + c ] = Math.round( acc[ i * 5 + c ] / acc[ i * 5 + 4 ] );
		return { raw: { vb: V, ib: ib.map( ( v ) => map[ v ] ), flags: F, lock: F.map( ( f ) => f & 1 ) }, m: { ...m, nv: n } };

	};

	// the settings of a level's simplification, as the packer's main loop has them, and their harder variants
	const settings = ( level, v ) => {

		const hard = v.hard ? { colour: 0, ground: false, groundKeep: 0 } : {};
		const col = v.colour !== undefined ? { colour: v.colour } : {};
		const r = v.ratio || 1, e = v.error || 1;
		if ( level === 'lo' ) return [ OPT.lo_ratio * r, OPT.lo_error * e, { glowLock: !! OPT.lo_glow_lock, glass: true, lanterns: true, props: true, groundKeep: OPT.lo_ground, guard: OPT.lo_glow_guard || 0, ...hard, ...col } ];
		return [ OPT[ level + '_ratio' ] * r, OPT[ level + '_error' ] * e, { glass: true, glowLock: true, prune: !! OPT[ level + '_prune' ], border: !! OPT.lock_border, colour: OPT[ level + '_colour_weight' ], ground: !! OPT.ground_lock, guard: OPT[ level + '_glow_guard' ] || 0, ...hard, ...col } ];

	};

	const bundles = {}; // name -> { name, level, cells: { key: meta } }
	const put = ( bundle, level, key, vb, ib, sub, m, extra = {} ) => {

		OPT.colour_bits = 16;
		const p = pack( vb, ib, sub, m );
		OPT.colour_bits = bits;
		const dir = path.join( LM, bundle );
		fs.mkdirSync( dir, { recursive: true } );
		fs.writeFileSync( path.join( dir, key + '.bin' ), p.blob );
		const b = bundles[ bundle ] || ( bundles[ bundle ] = { name: bundle, level, cells: {}, ...extra } );
		b.cells[ key ] = Object.assign( { file: `${ bundle }/${ key }.bin`, len: p.blob.byteLength }, p.meta );
		return p;

	};

	const finish = () => {

		for ( const b of Object.values( bundles ) ) {

			b.tris = Object.values( b.cells ).reduce( ( n, c ) => n + c.ni / 3, 0 );
			b.verts = Object.values( b.cells ).reduce( ( n, c ) => n + c.nv, 0 );
			b.bytes = Object.values( b.cells ).reduce( ( n, c ) => n + c.len, 0 );
			b.baked = Math.floor( Date.now() / 1000 );
			fs.writeFileSync( path.join( LM, b.name + '.json' ), JSON.stringify( b ) );
			console.log( `[lightmap pack] ${ b.name.padEnd( 16 ) } ${ String( b.tris ).padStart( 8 ) } tris ${ String( b.verts ).padStart( 8 ) } verts ${ String( b.bytes ).padStart( 9 ) } B${ b.pages ? `  ${ b.pages.length } pages` : '' }` );

		}

	};

	// a level's source: the raw chunk with the ground made one over the levels, as in the packer's main loop
	// (`other`: { level: file } - that level from another file, as tools/next/lsvb_refit.py leaves one in a bake)
	const sources = ( ch0, other = {} ) => {

		setW( 10 );
		const ch = { ...ch0 };
		for ( const [ l, f ] of Object.entries( other ) ) ch[ l ] = { ...ch[ l ], raw: path.relative( RAW, f ) };
		const raws = { mid: readRaw( ch.mid ) };
		if ( OPT.ground_unify && ch.near ) {

			raws.near = readRaw( ch.near );
			if ( ch.fine ) {

				const wf = ch.fine.w || 10;
				setW( wf ); const fine = readRaw( ch.fine ); setW( 10 );
				const map = groundMap( fine, ch.fine, wf );
				groundFrom( map, fine, wf, raws.mid, ch.mid, 10 );
				groundFrom( map, fine, wf, raws.near, ch.near, 10 );

			} else groundFrom( groundMap( raws.near, ch.near, 10 ), raws.near, 10, raws.mid, ch.mid, 10 );

		} else if ( ch.near ) raws.near = readRaw( ch.near );

		return raws;

	};

	if ( S.stage === 'dump' ) {

		const out = [];
		// (what an earlier run left: a variant may start from one of its meshes, or match its triangle count)
		const dj = path.join( LRAW, 'dump.json' );
		const prev = fs.existsSync( dj ) ? JSON.parse( fs.readFileSync( dj, 'utf8' ) ).meshes : [];
		const have = ( key, level, variant ) => out.find( ( x ) => x.key === key && x.level === level && x.variant === variant ) || prev.find( ( x ) => x.key === key && x.level === level && x.variant === variant );
		for ( const key of S.cells ) {

			const ch = man.chunks.find( ( c ) => c.k === key );
			if ( ! ch ) throw new Error( 'no chunk ' + key );
			if ( ( ch.mid.w || 10 ) !== 10 || ( ch.near && ( ch.near.w || 10 ) !== 10 ) ) throw new Error( 'the prototype reads 10 x u16 vertices' );
			const raws = sources( ch );
			for ( const level of S.levels ) {

				const src = level === 'near' ? 'near' : 'mid';
				if ( ! ch[ src ] ) continue;
				for ( const [ vname, v ] of Object.entries( S.variants[ level ] || {} ) ) {

					const [ ratio, error, o ] = settings( level, v );
					let w = v.hard ? weld( raws[ src ], ch[ src ] ) : { raw: raws[ src ], m: ch[ src ] };
					if ( v.src ) {

						// from a mesh of the dump (the unsimplified level with another light: stage "relight") instead of the bake's
						const d = have( key, src, v.src );
						if ( ! d ) throw new Error( `no ${ key } ${ src } ${ v.src } in the dump` );
						// (rawrf*: the bake's level with other colours - read as the packer reads the bake's, ground and panes)
						const r = v.src.startsWith( 'rawrf' ) ? sources( ch, { [ src ]: path.join( LRAW, d.raw ) } )[ src ] : read( path.join( LRAW, d.raw ), d.nv, d.ni );
						w = { raw: r, m: { ...ch[ src ], nv: d.nv, ni: d.ni, sub: d.sub } };

					}

					// (raw: as the bake left it, not simplified)
					let s = v.raw ? { ib: w.raw.ib.slice( 0, w.m.ni ), sub: w.m.sub } : simplify( w.raw, w.m, ratio, error, o );
					if ( v.match ) {

						// the triangle count of another variant (or `frac` of it), by the error bound alone: equal budgets for
						// meshes simplified by different rules (the bound may go below the level's own: 0.05 .. 40 times it)
						const want = Math.round( have( key, level, v.match ).ni * ( v.frac || 1 ) );
						let lo = v.frac || v.wide ? 0.05 : 1, hi = v.frac || v.wide ? 40 : 12;
						for ( let it = 0; it < 18 && Math.abs( s.ib.length - want ) > want * 0.01; it ++ ) {

							const e = Math.sqrt( lo * hi );
							s = simplify( w.raw, w.m, ratio * 0.2, error * e, o );
							if ( s.ib.length > want ) lo = e; else hi = e;

						}

					}

					const c = compact( w.raw.vb, w.raw.flags, s );
					const file = `${ key }.${ level }.${ vname }.bin`;
					write( path.join( LRAW, file ), c );
					const m = { key, level, variant: vname, raw: file, min: ch[ src ].min, q: ch[ src ].q, nv: c.nv, ni: c.ib.length, sub: s.sub };
					out.push( m );
					// ... and with the vertex light it has (rounded as the packer rounds it)
					const vb = c.vb.slice(); roundColours( vb, c.nv );
					const p = put( `${ level }_${ vname }_v`, level, key, vb, c.ib, s.sub, m );
					m.vbytes = p.blob.byteLength; m.vnv = p.meta.nv;
					console.log( `[lightmap dump] ${ key } ${ level } ${ vname }: ${ c.ib.length / 3 } tris, ${ c.nv } verts, ${ p.blob.byteLength } B packed with vertex light` );

				}

			}

		}

		const all = out.concat( prev.filter( ( p ) => ! out.find( ( x ) => x.key === p.key && x.level === p.level && x.variant === p.variant ) ) );
		fs.writeFileSync( dj, JSON.stringify( { meshes: all, colour_scale: man.colour_scale, uv_range: man.uv_range } ) );
		finish();
		return;

	}

	if ( S.stage === 'pack' ) {

		const bake = JSON.parse( fs.readFileSync( path.join( LRAW, 'bake.json' ), 'utf8' ) );
		for ( const b of bake.bundles ) {

			for ( const m of b.meshes ) {

				const r = read( path.join( LRAW, m.raw ), m.nv, m.ni );
				const vb = r.vb.slice(); roundColours( vb, m.nv );
				const p = put( b.name, b.level, m.key, vb, r.ib, m.sub, m, { range: b.range, gloss: b.gloss, size: b.size, pages: b.pages, texel: b.texel, stats: b.stats } );
				bundles[ b.name ].cells[ m.key ].lm = m.lm; // per mesh numbers of the bake (texels, charts, splits)

			}

		}

		// vertex light from the full model's surface on the unsimplified level, then simplified with the colour weighted high
		for ( const m of bake.va || [] ) {

			const ch = man.chunks.find( ( c ) => c.k === m.key );
			const r = read( path.join( LRAW, m.raw ), ch[ m.src ].nv, ch[ m.src ].ni );
			for ( const level of m.levels ) {

				const [ ratio, error, o ] = settings( level, { colour: S.va_colour ?? 2 } );
				const s = simplify( r, ch[ m.src ], ratio, error, o );
				const c = compact( r.vb, r.flags, s );
				roundColours( c.vb, c.nv );
				put( `${ level }_va`, level, m.key, c.vb, c.ib, s.sub, { min: ch[ m.src ].min, q: ch[ m.src ].q, nv: c.nv, ni: c.ib.length, sub: s.sub } );

			}

		}

		finish();
		return;

	}

	throw new Error( 'lm.stage: dump or pack' );

}
