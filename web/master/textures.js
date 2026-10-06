// The town's pictures, each read out of the master file by a byte range when it is needed - never before.
//
//   pages()    the first frame's colour layers: a few WebP pages of 512 px tiles -> one texture array
//   sharpen()  after the first frame: every layer as a file of its own at the tier's sizes, in two arrays (`big`:
//              the large-tiled walls, roofs and paving; the rest), each swapped in whole when its last layer is
//              made - one upload, no half-sharp town
//   normals()  the materials' normal + roughness layers
//   compact()  a tier without sharp layers keeps the first array: block-compressed where it stands
//   ground()   the ground's contact occlusion and the lanterns' shadows on it: two pictures of the town from above
//
// The arrays behind the first frame are block-compressed here (compress.js) where the GPU takes it, and they
// give way to the town's levels (yieldTo): a level is what a visitor sees coming in, a sharper texture is not.
//
// The materials hold the texture nodes (TX, NR, GAO of materials/): a new array is put into those nodes, nothing
// is rebuilt. `file( name )` is the plugin's: the bytes of one embedded file as a Blob.

import * as THREE from 'three';
import { createCompressor } from './compress.js';

const HOLD = 40; // m: the cells this near show their level before a picture is asked for ...
const HOLD_MAX = 90000; // ms: ... or this long has passed
const TRICKLE = 2; // pictures at a time while levels are still coming in

// compress: false keeps every array RGBA8 (?tc=0)
export function createTextures( { file, bake, tier, renderer, compress = true } ) {

	const N = bake.layers.length;
	let packer = compress ? createCompressor( renderer, tier ) : null;
	let traffic = null, gate = null;
	let held = 0; // bytes on the GPU, mip levels included

	// --- the order of things. `tier.fetch` pictures of an array at a time, in the order given - and of all arrays
	// together none until the cells within HOLD metres show the level they are to show, TRICKLE while any level
	// is awaited, then as many as the tier fetches.
	const enter = () => {

		if ( ! traffic ) return Promise.resolve( () => {} );
		if ( ! gate ) {

			const full = Math.max( 2, tier.fetch || 3 ), waiting = [], t0 = performance.now();
			let busy = 0, open = false, timer = 0;
			const room = () => {

				if ( ! open && traffic.awaited( HOLD ) > 0 && performance.now() - t0 < HOLD_MAX ) return 0;
				open = true;
				return traffic.awaited() + traffic.baseLeft > 0 ? Math.min( TRICKLE, full ) : full;

			};
			const pump = () => {

				while ( waiting.length && busy < room() ) { busy ++; waiting.shift()(); }
				if ( ! waiting.length && timer ) { clearInterval( timer ); timer = 0; }

			};
			const leave = () => { busy --; pump(); };
			gate = () => new Promise( ( r ) => { waiting.push( () => r( leave ) ); if ( ! timer ) timer = setInterval( pump, 100 ); pump(); } );

		}

		return gate();

	};
	const lanes = () => {

		const slots = Array.from( { length: Math.max( 2, tier.fetch || 3 ) }, () => Promise.resolve() );
		let turn = 0;
		return ( name ) => {

			const i = turn ++ % slots.length;
			return ( slots[ i ] = slots[ i ].catch( () => {} ).then( async () => { const leave = await enter(); try { return await file( name ); } finally { leave(); } } ) );

		};

	};

	// --- a picture -> RGBA bytes at w x h (as stored: no colour conversion, alpha not multiplied in)
	const draw = async ( blob, w, h ) => {

		const bitmap = await createImageBitmap( blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
		const width = w || bitmap.width, height = h || bitmap.height;
		const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( width, height ) : Object.assign( document.createElement( 'canvas' ), { width, height } );
		const context = canvas.getContext( '2d', { willReadFrequently: true } );
		context.globalCompositeOperation = 'copy'; context.imageSmoothingQuality = 'high';
		context.drawImage( bitmap, 0, 0, width, height );
		bitmap.close();
		return context;

	};

	const dress = ( tex, colorSpace, bytes ) => {

		tex.colorSpace = colorSpace;
		tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
		tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
		tex.anisotropy = tier.aniso;
		tex.needsUpdate = true;
		tex.userData.bytes = bytes; held += bytes;
		return tex;

	};
	// RGBA8, the GPU makes the mip levels. keep: the pixels stay here (the first array: compact() reads them)
	const plainArray = ( data, size, depth, colorSpace, keep = false ) => {

		const tex = new THREE.DataArrayTexture( data, size, size, depth );
		tex.format = THREE.RGBAFormat; tex.type = THREE.UnsignedByteType;
		tex.generateMipmaps = true;
		if ( ! keep ) tex.onUpdate = () => { tex.image.data = null; tex.onUpdate = null; }; // on the GPU: the copy here has done its work
		return dress( tex, colorSpace, data.length * 4 / 3 );

	};
	// block-compressed: the levels of `depth` layers, empty -> filled by layer -> the array
	const blocks = ( size, depth ) => Array.from( { length: Math.log2( size ) + 1 }, ( _, l ) => {

		const width = Math.max( 1, size >> l ), each = Math.ceil( width / 4 ) ** 2 * 16;
		return { data: new Uint8Array( each * depth ), width, height: width, each };

	} );
	const packedArray = ( mips, size, depth, colorSpace ) => {

		const tex = new THREE.CompressedArrayTexture( mips.map( ( { data, width, height } ) => ( { data, width, height } ) ), size, size, depth, packer.format, THREE.UnsignedByteType );
		tex.generateMipmaps = false;
		tex.onUpdate = () => { for ( const m of tex.mipmaps ) m.data = null; tex.onUpdate = null; };
		return dress( tex, colorSpace, mips.reduce( ( n, m ) => n + m.data.length, 0 ) );

	};
	// An array that another has taken the place of. dispose() alone would not do: the bindings of a mesh that
	// arrives later still name it and three would create it again - so what is left of it is one texel.
	const retire = ( old ) => {

		held -= old.userData.bytes;
		old.dispose();
		if ( old.isCompressedTexture ) { old.mipmaps = [ { data: new Uint8Array( 16 ), width: 4, height: 4 } ]; old.image = { width: 4, height: 4, depth: 1 }; }
		else { old.image = { data: new Uint8Array( [ 0, 0, 0, 255 ] ), width: 1, height: 1, depth: 1 }; old.generateMipmaps = false; old.minFilter = THREE.LinearFilter; }
		old.onUpdate = null;

	};

	// single-layer files (null: a layer of `fill`) -> one array
	const plainLayers = ( names, size, colorSpace, fill ) => {

		const px = size * size * 4, data = new Uint8Array( px * names.length ), fetch = lanes();
		if ( fill ) new Uint32Array( data.buffer ).fill( new Uint32Array( new Uint8Array( fill ).buffer )[ 0 ] );
		return Promise.all( names.map( async ( name, k ) => {

			if ( name ) data.set( ( await draw( await fetch( name + '.webp' ), size, size ) ).getImageData( 0, 0, size, size ).data, k * px );

		} ) ).then( () => plainArray( data, size, names.length, colorSpace ) );

	};
	const packedLayers = async ( names, size, colorSpace, fill ) => {

		const mips = blocks( size, names.length ), fetch = lanes(), srgb = colorSpace === THREE.SRGBColorSpace;
		const blank = fill ? packer.solid( fill ) : null;
		packer.hold();
		try {

			// (the fetching in its lanes, the encoding in the workers' own queue: a file does not wait for a worker)
			await Promise.all( names.map( async ( name, k ) => {

				if ( ! name ) { for ( const m of mips ) for ( let o = k * m.each; o < ( k + 1 ) * m.each; o += 16 ) m.data.set( blank, o ); return; }
				const made = await packer.chain( { blob: await fetch( name + '.webp' ), size, srgb } );
				made.mips.forEach( ( m, l ) => mips[ l ].data.set( m, k * mips[ l ].each ) );

			} ) );

		} finally { packer.release(); }

		return packedArray( mips, size, names.length, colorSpace );

	};
	// (a compression that fails - a worker that does not start, a picture it cannot read - is not tried again)
	const layersOf = ( names, size, colorSpace, fill = null ) => ( packer ? packedLayers( names, size, colorSpace, fill ).catch( ( e ) => {

		console.warn( `town: block compression: ${e.message || e} (RGBA8 instead)` );
		packer = null;
		return plainLayers( names, size, colorSpace, fill );

	} ) : plainLayers( names, size, colorSpace, fill ) );

	return {
		get held() { return held; },
		get compressed() { return packer !== null; },

		// traffic: { awaited( within ), baseLeft } - the town's levels (levels.js), which the pictures give way to
		yieldTo( levels ) { traffic = levels; },

		async pages() {

			const t = bake.tex[ tier.tex ], size = t.size, per = t.cols * t.rows;
			const data = new Uint8Array( size * size * 4 * N );
			await Promise.all( t.pages.map( async ( name, p ) => {

				const page = await draw( await file( name + '.webp' ) );
				for ( let i = p * per; i < Math.min( N, ( p + 1 ) * per ); i ++ ) {

					const k = i - p * per, col = k % t.cols, row = ( k - col ) / t.cols;
					data.set( page.getImageData( col * size, row * size, size, size ).data, i * size * size * 4 );

				}

			} ) );
			return plainArray( data, size, N, THREE.SRGBColorSpace, true );

		},

		// TX: { a, b: the texture nodes of the two arrays, u: { b0, b1: the layers the big array holds, shift } }
		async sharpen( TX, first ) {

			const groups = bake.layer_big || [ 0, 0 ];
			if ( ! bake.texl || ! ( tier.big || tier.sharp ) ) return;
			const nb = tier.big ? Math.min( N - 2, tier.big[ 1 ] === 'top' ? groups[ 0 ] : groups[ 0 ] + groups[ 1 ] ) : 0;
			const all = Array.from( { length: N }, ( _, i ) => i );
			const names = ( size, layers ) => ( bake.texl[ size ] && layers.length && layers.every( ( i ) => bake.texl[ size ].files[ i ] ) ? layers.map( ( i ) => bake.texl[ size ].files[ i ] ) : null );
			const bigNames = nb > 0 ? names( tier.big[ 0 ], all.slice( 2, 2 + nb ) ) : null;
			const restNames = tier.sharp ? names( tier.sharp, bigNames ? all.filter( ( i ) => i < 2 || i >= 2 + nb ) : all ) : null;
			const big = bigNames ? layersOf( bigNames, tier.big[ 0 ], THREE.SRGBColorSpace ) : null;
			const rest = restNames ? layersOf( restNames, tier.sharp, THREE.SRGBColorSpace ) : null;
			const jobs = [];
			if ( big ) jobs.push( big.then( ( tex ) => {

				for ( const n of TX.b ) n.value = tex;
				TX.u.b0.value = 2; TX.u.b1.value = 2 + nb;

			} ) );
			// (the pages' array stays until both are in: its layers under the big ones are what the remap of `rest` leaves out)
			if ( rest ) jobs.push( Promise.all( [ rest, big ] ).then( ( [ tex ] ) => {

				for ( const n of TX.a ) n.value = tex;
				TX.u.shift.value = big ? nb : 0;
				retire( first );

			} ) );
			await Promise.all( jobs );

		},

		// NR: { size, count, files: per layer of the array a name or null, nodes, on }. Until it is in the materials
		// shade the height relief alone; a layer without a file is flat and of middling roughness.
		async normals( NR ) {

			if ( ! NR ) return;
			const tex = await layersOf( NR.files, NR.size, THREE.NoColorSpace, [ 128, 128, 153, 255 ] );
			for ( const n of NR.nodes ) n.value = tex;
			NR.on.value = 1; NR.tex = tex;

		},

		// The first array of a tier that keeps it (no `sharp`): encoded from the pixels it holds once the cells
		// around the walker are in, and swapped for the result - without the layers a `big` array has taken over.
		async compact( TX, first ) {

			if ( ! packer || tier.sharp || ! first.image.data ) return;
			const size = first.image.width, each = size * size * 4;
			const keep = Array.from( { length: N }, ( _, i ) => i ).filter( ( i ) => i < TX.u.b0.value || i >= TX.u.b1.value );
			const mips = blocks( size, keep.length );
			packer.hold();
			try {

				( await enter() )();
				// (a layer's pixels are copied for a worker only when one is free: a megabyte in flight, not the array twice)
				let at = 0;
				const lane = async () => {

					while ( at < keep.length ) {

						const k = at ++, px = first.image.data.slice( keep[ k ] * each, ( keep[ k ] + 1 ) * each );
						const made = await packer.chain( { px, size, srgb: true }, [ px.buffer ] );
						made.mips.forEach( ( m, l ) => mips[ l ].data.set( m, k * mips[ l ].each ) );

					}

				};
				await Promise.all( Array.from( { length: packer.workers }, lane ) );

			} finally { packer.release(); }

			const tex = packedArray( mips, size, keep.length, THREE.SRGBColorSpace );
			for ( const n of TX.a ) n.value = tex;
			TX.u.shift.value = N - keep.length;
			retire( first );

		},

		// GAO: { node, on, sh: the shadows' switch or null }. Two bytes a texel, four with the shadows.
		async ground( GAO ) {

			if ( ! GAO ) return;
			const g = bake.gao, w = g.w, h = g.h, shadows = GAO.sh ? bake.gsh : null;
			const read = async ( q ) => ( await draw( await file( q.file ), w, h ) ).getImageData( 0, 0, w, h ).data;
			const [ px, ps ] = await Promise.all( [ read( g ), shadows ? read( shadows ) : null ] );
			const data = new Uint8Array( w * h * ( shadows ? 4 : 2 ) );
			if ( shadows ) for ( let i = 0; i < data.length; i += 4 ) { data[ i ] = px[ i ]; data[ i + 1 ] = px[ i + 1 ]; data[ i + 2 ] = ps[ i ]; data[ i + 3 ] = ps[ i + 1 ]; }
			else for ( let i = 0, j = 0; j < data.length; i += 4, j += 2 ) { data[ j ] = px[ i ]; data[ j + 1 ] = px[ i + 1 ]; }
			const tex = new THREE.DataTexture( data, w, h, shadows ? THREE.RGBAFormat : THREE.RGFormat, THREE.UnsignedByteType );
			tex.unpackAlignment = 1;
			tex.generateMipmaps = true;
			tex.onUpdate = () => { tex.image.data = null; tex.onUpdate = null; };
			dress( tex, THREE.NoColorSpace, data.length * 4 / 3 );
			tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
			GAO.node.value = tex; GAO.on.value = 1; GAO.tex = tex;

		},
	};

}
