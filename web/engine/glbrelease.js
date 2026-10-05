// The town's GLBs as tools/glbopt.mjs released them (out/web/_glb/<town>/): the same vertices, triangles, nodes and
// looks as the files under out/glb, but smaller on the wire (meshopt compression of the data as it is, no
// quantisation), and the files every visitor needs before the first frame (each asset's coarsest level + the
// terrain) in one request; in the variants with an s also the primitives the runtime draws with one material joined
// and the looks in one table for the town (fewer draws, but joined primitives are drawn in file order instead of the
// renderer's depth order, which shows in a few pixels where their surfaces meet - see tools/glbopt.mjs).
// A file is only taken from the release when it was made from the very source the town names (version and
// size, see load()); everything else - no release, a rebuilt asset, any error - is read from out/glb as before.
//
// Switches (the page's URL; each works alone):
//     ?glb=raw         no release at all: the page as it was
//     ?glb=<variant>   that folder of the release instead of the default (release.json)
//     ?meshopt=0       the default without the meshopt compression   (needs that variant: tools/glbopt.mjs --meshopt 0)
//     ?struct=0        the default without the joined primitives and the look table      (tools/glbopt.mjs --struct 0)
//     ?bundle=0        every file by its own request
//     ?mow=<n>         meshopt decode workers (0: decode on the main thread)

import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { MaterialLib } from './materials.js';

const FORMAT = 1; // of index.json / release.json
const WORKERS = 2; // meshopt decodes leave the main thread (one message per stream: cheap for the joined files)
const PARSERS = 4; // bundled files in the parser at a time
const SHADOWLESS = /^(emissive|neon|glass_|window_|water)/; // (as AssetLoader.adopt())

const parseJSON = ( s, d ) => {

	if ( s === undefined || s === null ) return d;
	if ( typeof s !== 'string' ) return s;
	try {

		return JSON.parse( s );

	} catch ( e ) {

		return d;

	}

};

const tick = () => new Promise( ( r ) => {

	const c = new MessageChannel();
	c.port1.onmessage = () => r();
	c.port2.postMessage( 0 );

} );

// the folder name tools/glbopt.mjs gives a set of options ('' = nothing to release: the source files)
const variantOf = ( o ) => ( ! o.meshopt && ! o.struct ? '' : ( o.meshopt ? 'm' : '' ) + ( o.struct ? 's' : '' ) + ( o.reorder === 'tri' ? 't' : o.reorder === 'full' ? 'r' : '' ) + ( o.meshopt && o.codec === 'khr' ? 'k' : '' ) );

// -> what Town loads its GLBs through: the release of the town whose town.json is at townUrl, or `assets` itself
export async function openRelease( assets, townUrl, query = location.search ) {

	try {

		const Q = new URLSearchParams( query ), m = /^(.*)\/out\/([^/]+)\/town\.json/.exec( townUrl );
		if ( ! m || Q.get( 'glb' ) === 'raw' ) return assets;
		const dir = `${m[ 1 ]}/out/web/_glb/${m[ 2 ]}`;
		const get = ( u ) => fetch( u, { cache: 'no-cache' } ).then( ( r ) => ( r.ok ? r.json() : null ) ).catch( () => null );
		let index = await get( Q.get( 'glb' ) ? `${dir}/${Q.get( 'glb' )}/index.json` : `${dir}/release.json` );
		if ( index && index.format === FORMAT && ! Q.get( 'glb' ) && ( Q.get( 'meshopt' ) === '0' || Q.get( 'struct' ) === '0' ) ) {

			const name = variantOf( { ...index.opt, meshopt: index.opt.meshopt && Q.get( 'meshopt' ) !== '0', struct: index.opt.struct && Q.get( 'struct' ) !== '0' } );
			if ( ! name ) return assets;
			if ( name !== index.variant ) index = await get( `${dir}/${name}/index.json` );

		}

		if ( ! index || index.format !== FORMAT || ! index.files || ! index.materials ) {

			if ( Q.get( 'glb' ) || Q.get( 'meshopt' ) === '0' || Q.get( 'struct' ) === '0' ) console.warn( '[glb] that release variant is not there (tools/glbopt.mjs builds it): reading out/glb' );
			return assets;

		}

		await MeshoptDecoder.ready;
		const workers = Q.has( 'mow' ) ? Math.max( 0, Math.min( 8, parseInt( Q.get( 'mow' ) ) || 0 ) ) : WORKERS;
		if ( index.opt.meshopt && workers ) MeshoptDecoder.useWorkers( workers );
		return new ReleaseAssets( assets, index, `${dir}/${index.variant}`, { bundle: Q.get( 'bundle' ) !== '0' } );

	} catch ( e ) {

		console.warn( '[glb] release not usable, reading out/glb:', e );
		return assets;

	}

}

class ReleaseAssets {

	constructor( assets, index, url, { bundle = true } = {} ) {

		this.raw = assets;
		this.lib = assets.lib;
		this.index = index;
		this.url = url;
		this.variant = index.variant;
		this.gltf = new GLTFLoader();
		this.gltf.setMeshoptDecoder( MeshoptDecoder );
		this.bundle = bundle && index.bundle ? index.bundle : null;
		this.stats = { release: 0, bundled: 0, raw: 0, failed: 0 }; // files by where they came from
		this._slices = null;
		this._parsing = 0;
		this._queue = [];
		this._breath = 0;

	}

	// as AssetLoader.load(); bytes: the size of the source file the caller means (town.json)
	load( id, variant = 'default', opts = {} ) {

		const key = `${id}/${variant}`, F = this.index.files;
		const e = F[ key ], e0 = F[ key.replace( /\.lod\d+$/, '' ) ];
		// the release of exactly this file: the asset's version is the mtime of its finest level (as tools/town.py
		// writes it), a level is told by its size
		if ( ! e || ! e0 || opts.sidecar !== false || e0[ 0 ] !== opts.version || ( opts.bytes !== undefined && e[ 1 ] !== opts.bytes ) ) {

			if ( this._slices ) this._slices.delete( key );
			this.stats.raw ++;
			return this.raw.load( id, variant, opts );

		}

		return this._load( key, e ).catch( ( err ) => {

			console.warn( `[glb] ${key}: ${err.message || err} - reading out/glb` );
			this.stats.failed ++;
			return this.raw.load( id, variant, opts );

		} );

	}

	async _load( key, e ) {

		const data = this.bundle && e[ 4 ] >= 0 ? await this._slice( key ) : null;
		const gltf = data ? await this._parse( data ) : await this.gltf.loadAsync( `${this.url}/${key}.glb?v=${e[ 3 ]}` );
		const root = gltf.scene, looks = this.index.materials, joined = [];
		root.name = key;
		// a file without materials names its looks per primitive: extras.tsq_m = [ look, vertices, look, vertices, .. ]
		root.traverse( ( o ) => {

			const r = o.isMesh && o.geometry.userData.tsq_m;
			if ( ! r ) return;
			if ( ! looks[ r[ 0 ] ] ) throw new Error( `look ${r[ 0 ]} is not in the release` );
			o.material = { userData: looks[ r[ 0 ] ], dispose() {} }; // (what adopt() reads of a glTF material)
			if ( r.length > 2 ) joined.push( o );

		} );
		await this.lib.prepare( MaterialLib.libsOf( root ), {} );
		const inst = this.raw.adopt( root, null );
		for ( const o of joined ) this._paint( o );
		this.stats[ data ? 'bundled' : 'release' ] ++;
		return inst;

	}

	// a joined primitive: adopt() wrote the first look over all its vertices, the other ranges get theirs
	_paint( o ) {

		const g = o.geometry, r = g.userData.tsq_m, lib = this.lib;
		if ( g.userData.tsqJoined ) return;
		const a = g.getAttribute( '_tint' ).array, b = g.getAttribute( '_prm' ).array;
		let v0 = r[ 1 ];
		for ( let k = 2; k < r.length; k += 2 ) {

			const ex = this.index.materials[ r[ k ] ], n = r[ k + 1 ];
			if ( ! ex ) throw new Error( `look ${r[ k ]} is not in the release` );
			const name = ex.tsq_lib || 'plain', mp = parseJSON( ex.tsq_params, {} );
			// the tool joined what the runtime draws alike (material, shadow class): made sure of here, file by file
			if ( lib.get( name, mp ) !== o.material || ( ! o.material.transparent && ! SHADOWLESS.test( name ) && ! ( mp.glow > 0 ) ) !== o.castShadow ) throw new Error( 'joined primitives that are not drawn alike' );
			const v = lib.vertexValues( name, ex.tsq_tint || null, mp );
			for ( let j = v0 * 4, end = ( v0 + n ) * 4; j < end; j += 4 ) {

				a[ j ] = v[ 0 ]; a[ j + 1 ] = v[ 1 ]; a[ j + 2 ] = v[ 2 ]; a[ j + 3 ] = v[ 3 ];
				b[ j ] = v[ 4 ]; b[ j + 1 ] = v[ 5 ]; b[ j + 2 ] = v[ 6 ]; b[ j + 3 ] = v[ 7 ];

			}

			v0 += n;

		}

		if ( v0 * 4 !== a.length ) throw new Error( 'looks that do not cover the vertices' );
		g.userData.tsqJoined = true;

	}

	// a few bundled files in the parser at a time, and a breath for the page between them
	async _parse( data ) {

		if ( this._parsing >= PARSERS ) await new Promise( ( r ) => this._queue.push( r ) );
		else this._parsing ++;
		try {

			if ( performance.now() - this._breath > 6 ) {

				await tick();
				this._breath = performance.now();

			}

			return await this.gltf.parseAsync( data, '' );

		} finally {

			const next = this._queue.shift();
			if ( next ) next();
			else this._parsing --;

		}

	}

	// the bytes of a bundled file (null: not to be had from the bundle - it is then asked for alone)
	_slice( key ) {

		if ( ! this._slices ) this._stream();
		const s = this._slices.get( key );
		this._slices.delete( key );
		return s ? s.promise : null;

	}

	// one request for the bundle; every file is handed out as soon as its last byte has arrived
	_stream() {

		const list = Object.entries( this.index.files ).filter( ( f ) => f[ 1 ][ 4 ] >= 0 ).sort( ( x, y ) => x[ 1 ][ 4 ] - y[ 1 ][ 4 ] );
		const slices = this._slices = new Map();
		for ( const [ key ] of list ) {

			const s = {};
			s.promise = new Promise( ( r ) => ( s.resolve = r ) );
			slices.set( key, s );

		}

		const give = ( i, buf ) => list[ i ][ 2 ].resolve( buf );
		for ( const f of list ) f.push( slices.get( f[ 0 ] ) ); // (handed out even when the map has let go of it)
		( async () => {

			let i = 0;
			try {

				const res = await fetch( `${this.url}/${this.bundle.file}` );
				if ( ! res.ok ) throw new Error( `bundle: ${res.status}` );
				const reader = res.body.getReader();
				let cur = null, fill = 0, at = 0; // the file being filled, bytes it has, position in the bundle
				for ( ;; ) {

					const { done, value } = await reader.read();
					if ( done ) break;
					let o = 0;
					while ( o < value.length && i < list.length ) {

						const [ , e ] = list[ i ];
						if ( at < e[ 4 ] ) {

							const skip = Math.min( value.length - o, e[ 4 ] - at );
							o += skip; at += skip;
							continue;

						}

						if ( ! cur ) {

							cur = new Uint8Array( e[ 2 ] );
							fill = 0;

						}

						const n = Math.min( value.length - o, e[ 2 ] - fill );
						cur.set( value.subarray( o, o + n ), fill );
						o += n; at += n; fill += n;
						if ( fill === e[ 2 ] ) {

							give( i ++, cur.buffer );
							cur = null;

						}

					}

					at += value.length - o;

				}

			} catch ( err ) {

				console.warn( `[glb] ${err.message || err} - its files are asked for one by one` );

			}

			for ( ; i < list.length; i ++ ) give( i, null );
			setTimeout( () => slices.clear(), 30000 ); // (files nobody asked for are not kept)

		} )();

	}

}
