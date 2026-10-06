// WebP on the wire, block-compressed on the GPU.
//
// A texture layer travels as a WebP picture. Uploaded as it decodes it is RGBA8: 5.6 MB of GPU memory for one
// 1024 px layer with its mip levels, and the town has eighty. So each layer is decoded and encoded in a worker to
// the block format this GPU takes - BC7 on a desktop, ASTC 4 x 4 on a phone or an Apple chip, both 8 bits a
// pixel - mip levels included, and the arrays go to the GPU at a quarter of the size, without a second set of
// files to host. The encoders and their worker are the walk page's (engine/lite_tc_worker.js, bc7m6.js,
// astc4.js), used as they are.

import * as THREE from 'three';
import { encodeBC7 } from '../engine/bc7m6.js';
import { solidASTC } from '../engine/astc4.js';

// -> { format, chain( job, transfer ), solid( rgba ), hold(), release() }, or null: this GPU takes neither format
// or the browser has no workers with a canvas, and the arrays stay RGBA8
export function createCompressor( renderer, tier ) {

	const has = ( name ) => { try { return renderer.hasFeature( 'texture-compression-' + name ); } catch ( e ) { return false; } };
	const astc = ! has( 'bc' );
	if ( ( astc && ! has( 'astc' ) ) || typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined' ) return null;

	// The workers share the cores with the page and with the workers that unpack the levels: half the cores less
	// one, and few - each holds a layer's pixels while it works.
	let count = Math.min( tier.sharp ? 3 : 2, Math.max( 1, ( ( navigator.hardwareConcurrency || 4 ) >> 1 ) - 1 ) );
	if ( navigator.deviceMemory && navigator.deviceMemory <= 2 ) count = 1;

	// (started with the first job, gone when no array is being made: a phone has them for a few seconds)
	const idle = [], waiting = [], running = new Map();
	let workers = null, ids = 0, users = 0;
	const next = () => {

		while ( idle.length && waiting.length ) {

			const worker = idle.pop(), job = waiting.shift();
			running.set( job.message.id, { ...job, worker } );
			worker.postMessage( job.message, job.transfer );

		}

	};
	const start = () => Array.from( { length: count }, () => {

		const worker = new Worker( new URL( '../engine/lite_tc_worker.js', import.meta.url ), { type: 'module' } );
		worker.onmessage = ( { data } ) => {

			const job = running.get( data.id );
			if ( ! job ) return;
			running.delete( data.id );
			idle.push( worker );
			next();
			if ( data.error ) job.reject( new Error( data.error ) ); else job.resolve( data );

		};
		worker.onerror = ( e ) => {

			for ( const [ id, job ] of running ) if ( job.worker === worker ) { running.delete( id ); job.reject( new Error( 'worker: ' + ( e.message || 'failed' ) ) ); }

		};
		idle.push( worker );
		return worker;

	} );

	return {
		format: astc ? THREE.RGBA_ASTC_4x4_Format : THREE.RGBA_BPTC_Format,
		workers: count,

		// one layer -> { mips: its block-compressed levels, largest first }. job: { blob, size, srgb } a picture
		// file, or { px, size, srgb } RGBA8 pixels (then pass [ px.buffer ] as `transfer`)
		chain( job, transfer = [] ) {

			return new Promise( ( resolve, reject ) => {

				if ( ! workers ) workers = start();
				waiting.push( { message: { ...job, id: ++ ids, refine: 1, astc }, transfer, resolve, reject } );
				next();

			} );

		},

		// one block of a single colour (a layer without a picture)
		solid( rgba ) {

			return astc ? solidASTC( ...rgba ) : encodeBC7( new Uint8Array( 64 ).map( ( _, i ) => rgba[ i % 4 ] ), 4, 4, new Uint8Array( 16 ) );

		},

		// around the making of an array: the workers live while anyone holds them
		hold() { users ++; },
		release() {

			if ( -- users > 0 || ! workers ) return;
			for ( const worker of workers ) worker.terminate();
			workers = null; idle.length = waiting.length = 0; running.clear();

		},
	};

}
