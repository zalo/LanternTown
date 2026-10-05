// ?tc=1 (lite_tc.js): pixels -> the block-compressed mip chain of a layer (BC7: bc7m6.js; ASTC 4 x 4: astc4.js), all of
// it off the main thread. Three kinds of job:
//   { blob, size }                         one WebP layer file
//   { px, size }                           one layer's RGBA8 pixels (the first array, already decoded on the page)
//   { page, hpage, size, cols, count }     an atlas page of the first array (and its height page): `count` layers
import { encodeBC7, encodeChain } from './bc7m6.js';
import { encodeASTC } from './astc4.js';

async function pixels( blob, w, h ) {

	const bmp = await createImageBitmap( blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
	const cv = new OffscreenCanvas( w || bmp.width, h || bmp.height );
	const cx = cv.getContext( '2d', { willReadFrequently: true } );
	cx.globalCompositeOperation = 'copy';
	cx.drawImage( bmp, 0, 0, cv.width, cv.height );
	bmp.close();
	return cx;

}

self.onmessage = async ( e ) => {

	const { id, blob, px, page, hpage, size, cols, count, srgb, refine, astc, slow } = e.data;
	try {

		const enc = astc ? encodeASTC : encodeBC7;
		const times = { decode: 0, encode: 0, mips: 0 };
		// (slow: every chain made that many times - a slower core, for measuring on a fast one)
		const chain = ( p ) => { for ( let i = 1; i < ( slow || 1 ); i ++ ) encodeChain( p, size, srgb, refine, times, enc ); return encodeChain( p, size, srgb, refine, times, enc ); };
		let t0 = performance.now();
		if ( page ) {

			const cx = await pixels( page ), hx = hpage ? await pixels( hpage ) : null;
			times.decode = performance.now() - t0;
			const layers = [], buffers = [];
			for ( let k = 0; k < count; k ++ ) {

				t0 = performance.now();
				const col = k % cols, row = ( k - col ) / cols;
				const p = cx.getImageData( col * size, row * size, size, size ).data;
				if ( hx ) { const hp = hx.getImageData( col * size, row * size, size, size ).data; for ( let j = 0; j < p.length; j += 4 ) p[ j + 3 ] = hp[ j ]; }
				times.decode += performance.now() - t0;
				const mips = chain( p );
				layers.push( mips );
				for ( const m of mips ) buffers.push( m.buffer );

			}

			self.postMessage( { id, layers, times }, buffers );
			return;

		}

		const p = px || ( await pixels( blob, size, size ) ).getImageData( 0, 0, size, size ).data;
		times.decode = performance.now() - t0;
		const mips = chain( p );
		self.postMessage( { id, mips, times }, mips.map( ( m ) => m.buffer ) );

	} catch ( err ) {

		self.postMessage( { id, error: String( err && err.message || err ) } );

	}

};
