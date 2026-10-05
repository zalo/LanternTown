// BC7 encoder, mode 6 only (experiment: ?tc=1, lite_tc.js): every 4 x 4 block is one line through RGBA space - two
// endpoints of 7 bits a channel plus a shared low bit each, and a 4-bit weight a pixel. No partitions, no search over
// modes: one principal axis, one least-squares pass. Smooth blocks (and the height in alpha, which is smooth) keep
// nearly 8 bits; a block with two unrelated colours, or a normal whose x, y and roughness go their own ways, pays.
// Plain JavaScript, no dependencies: runs in a worker (lite_tc_worker.js) and in node (measuring).

const WT = new Float64Array( [ 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 ] );
const P = new Float64Array( 64 ), E = new Float64Array( 8 ), V = new Float64Array( 4 ), M = new Float64Array( 4 ), C = new Float64Array( 16 );
const Q = new Uint8Array( 8 ), PB = new Uint8Array( 2 ), IX = new Uint8Array( 16 );

// endpoint k of E (floats) -> Q (8-bit values: 7 bits and the shared p bit), PB[ k ]
function quant( k ) {

	let best = 1e30, bp = 0;
	for ( let p = 0; p < 2; p ++ ) {

		let err = 0;
		for ( let c = 0; c < 4; c ++ ) {

			let q = Math.round( ( E[ k * 4 + c ] - p ) / 2 ); q = q < 0 ? 0 : q > 127 ? 127 : q;
			const d = q * 2 + p - E[ k * 4 + c ]; err += d * d;

		}

		if ( err < best ) { best = err; bp = p; }

	}

	PB[ k ] = bp;
	for ( let c = 0; c < 4; c ++ ) { let q = Math.round( ( E[ k * 4 + c ] - bp ) / 2 ); q = q < 0 ? 0 : q > 127 ? 127 : q; Q[ k * 4 + c ] = q * 2 + bp; }

}

// the weight of each pixel on the line Q0 -> Q1
function indices() {

	const d0 = Q[ 4 ] - Q[ 0 ], d1 = Q[ 5 ] - Q[ 1 ], d2 = Q[ 6 ] - Q[ 2 ], d3 = Q[ 7 ] - Q[ 3 ];
	const dd = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
	if ( dd === 0 ) { IX.fill( 0 ); return; }
	const s = 15 / dd;
	for ( let i = 0; i < 16; i ++ ) {

		const o = i * 4;
		let t = Math.round( ( ( P[ o ] - Q[ 0 ] ) * d0 + ( P[ o + 1 ] - Q[ 1 ] ) * d1 + ( P[ o + 2 ] - Q[ 2 ] ) * d2 + ( P[ o + 3 ] - Q[ 3 ] ) * d3 ) * s );
		IX[ i ] = t < 0 ? 0 : t > 15 ? 15 : t;

	}

}

// src: RGBA8, w x h -> out: ceil( w / 4 ) * ceil( h / 4 ) blocks of 16 bytes. dec (optional): what the blocks decode to
export function encodeBC7( src, w, h, out, refine = 1, dec = null ) {

	const bw = Math.ceil( w / 4 ), bh = Math.ceil( h / 4 );
	let o = 0;
	for ( let by = 0; by < bh; by ++ ) for ( let bx = 0; bx < bw; bx ++ ) {

		// the block, its mean and its box
		let m0 = 0, m1 = 0, m2 = 0, m3 = 0, lo0 = 255, lo1 = 255, lo2 = 255, lo3 = 255, hi0 = 0, hi1 = 0, hi2 = 0, hi3 = 0;
		for ( let y = 0; y < 4; y ++ ) {

			const sy = Math.min( h - 1, by * 4 + y ) * w;
			for ( let x = 0; x < 4; x ++ ) {

				const s = ( sy + Math.min( w - 1, bx * 4 + x ) ) * 4, k = ( y * 4 + x ) * 4;
				const r = src[ s ], g = src[ s + 1 ], b = src[ s + 2 ], a = src[ s + 3 ];
				P[ k ] = r; P[ k + 1 ] = g; P[ k + 2 ] = b; P[ k + 3 ] = a;
				m0 += r; m1 += g; m2 += b; m3 += a;
				if ( r < lo0 ) lo0 = r; if ( r > hi0 ) hi0 = r; if ( g < lo1 ) lo1 = g; if ( g > hi1 ) hi1 = g;
				if ( b < lo2 ) lo2 = b; if ( b > hi2 ) hi2 = b; if ( a < lo3 ) lo3 = a; if ( a > hi3 ) hi3 = a;

			}

		}

		M[ 0 ] = m0 / 16; M[ 1 ] = m1 / 16; M[ 2 ] = m2 / 16; M[ 3 ] = m3 / 16;
		if ( hi0 === lo0 && hi1 === lo1 && hi2 === lo2 && hi3 === lo3 ) {

			for ( let c = 0; c < 4; c ++ ) E[ c ] = E[ 4 + c ] = M[ c ];
			quant( 0 ); quant( 1 ); IX.fill( 0 );

		} else {

			// the principal axis: the covariance, a few rounds of power iteration from the box's diagonal
			C.fill( 0 );
			for ( let i = 0; i < 64; i += 4 ) {

				const a = P[ i ] - M[ 0 ], b = P[ i + 1 ] - M[ 1 ], c = P[ i + 2 ] - M[ 2 ], d = P[ i + 3 ] - M[ 3 ];
				C[ 0 ] += a * a; C[ 1 ] += a * b; C[ 2 ] += a * c; C[ 3 ] += a * d; C[ 5 ] += b * b; C[ 6 ] += b * c; C[ 7 ] += b * d; C[ 10 ] += c * c; C[ 11 ] += c * d; C[ 15 ] += d * d;

			}

			let v0 = hi0 - lo0, v1 = hi1 - lo1, v2 = hi2 - lo2, v3 = hi3 - lo3;
			// (the diagonal's signs: those of the covariance with the widest channel)
			{ const k = v0 >= v1 && v0 >= v2 && v0 >= v3 ? 0 : v1 >= v2 && v1 >= v3 ? 1 : v2 >= v3 ? 2 : 3;
				const row = k === 0 ? [ C[ 0 ], C[ 1 ], C[ 2 ], C[ 3 ] ] : k === 1 ? [ C[ 1 ], C[ 5 ], C[ 6 ], C[ 7 ] ] : k === 2 ? [ C[ 2 ], C[ 6 ], C[ 10 ], C[ 11 ] ] : [ C[ 3 ], C[ 7 ], C[ 11 ], C[ 15 ] ];
				if ( row[ 0 ] < 0 ) v0 = - v0; if ( row[ 1 ] < 0 ) v1 = - v1; if ( row[ 2 ] < 0 ) v2 = - v2; if ( row[ 3 ] < 0 ) v3 = - v3; }
			for ( let it = 0; it < 3; it ++ ) {

				const a = C[ 0 ] * v0 + C[ 1 ] * v1 + C[ 2 ] * v2 + C[ 3 ] * v3, b = C[ 1 ] * v0 + C[ 5 ] * v1 + C[ 6 ] * v2 + C[ 7 ] * v3;
				const c = C[ 2 ] * v0 + C[ 6 ] * v1 + C[ 10 ] * v2 + C[ 11 ] * v3, d = C[ 3 ] * v0 + C[ 7 ] * v1 + C[ 11 ] * v2 + C[ 15 ] * v3;
				const n = Math.max( Math.abs( a ), Math.abs( b ), Math.abs( c ), Math.abs( d ) );
				if ( n < 1e-9 ) break;
				v0 = a / n; v1 = b / n; v2 = c / n; v3 = d / n;

			}

			const vv = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
			let t0 = 1e30, t1 = - 1e30;
			for ( let i = 0; i < 64; i += 4 ) {

				const t = ( ( P[ i ] - M[ 0 ] ) * v0 + ( P[ i + 1 ] - M[ 1 ] ) * v1 + ( P[ i + 2 ] - M[ 2 ] ) * v2 + ( P[ i + 3 ] - M[ 3 ] ) * v3 ) / vv;
				if ( t < t0 ) t0 = t; if ( t > t1 ) t1 = t;

			}

			V[ 0 ] = v0; V[ 1 ] = v1; V[ 2 ] = v2; V[ 3 ] = v3;
			for ( let c = 0; c < 4; c ++ ) {

				const a = M[ c ] + V[ c ] * t0, b = M[ c ] + V[ c ] * t1;
				E[ c ] = a < 0 ? 0 : a > 255 ? 255 : a; E[ 4 + c ] = b < 0 ? 0 : b > 255 ? 255 : b;

			}

			quant( 0 ); quant( 1 ); indices();
			for ( let pass = 0; pass < refine; pass ++ ) {

				// the endpoints that fit these weights best (least squares, a channel at a time)
				let aa = 0, ab = 0, bb = 0;
				for ( let i = 0; i < 16; i ++ ) { const b = WT[ IX[ i ] ] / 64, a = 1 - b; aa += a * a; ab += a * b; bb += b * b; }
				const det = aa * bb - ab * ab;
				if ( det < 1e-4 ) break;
				for ( let c = 0; c < 4; c ++ ) {

					let ax = 0, bx2 = 0;
					for ( let i = 0; i < 16; i ++ ) { const b = WT[ IX[ i ] ] / 64, x = P[ i * 4 + c ]; ax += ( 1 - b ) * x; bx2 += b * x; }
					const e0 = ( bb * ax - ab * bx2 ) / det, e1 = ( aa * bx2 - ab * ax ) / det;
					E[ c ] = e0 < 0 ? 0 : e0 > 255 ? 255 : e0; E[ 4 + c ] = e1 < 0 ? 0 : e1 > 255 ? 255 : e1;

				}

				quant( 0 ); quant( 1 ); indices();

			}

		}

		// (the first pixel's weight is stored without its top bit: the endpoints the other way round if it has one)
		if ( IX[ 0 ] >= 8 ) {

			for ( let c = 0; c < 4; c ++ ) { const t = Q[ c ]; Q[ c ] = Q[ 4 + c ]; Q[ 4 + c ] = t; }
			const t = PB[ 0 ]; PB[ 0 ] = PB[ 1 ]; PB[ 1 ] = t;
			for ( let i = 0; i < 16; i ++ ) IX[ i ] = 15 - IX[ i ];

		}

		if ( dec ) for ( let y = 0; y < 4; y ++ ) for ( let x = 0; x < 4; x ++ ) {

			const X = bx * 4 + x, Y = by * 4 + y;
			if ( X >= w || Y >= h ) continue;
			const wt = WT[ IX[ y * 4 + x ] ], d = ( Y * w + X ) * 4;
			for ( let c = 0; c < 4; c ++ ) dec[ d + c ] = ( ( 64 - wt ) * Q[ c ] + wt * Q[ 4 + c ] + 32 ) >> 6;

		}

		// 128 bits, low bit first: mode (7), R0 R1 G0 G1 B0 B1 A0 A1 (7 each), P0 P1, the weights (3, then 15 x 4)
		let acc = 64, nb = 7;
		for ( let c = 0; c < 4; c ++ ) for ( let k = 0; k < 2; k ++ ) {

			acc |= ( Q[ k * 4 + c ] >> 1 ) << nb; nb += 7;
			while ( nb >= 8 ) { out[ o ++ ] = acc & 255; acc >>>= 8; nb -= 8; }

		}

		acc |= PB[ 0 ] << nb; nb ++; acc |= PB[ 1 ] << nb; nb ++;
		acc |= IX[ 0 ] << nb; nb += 3;
		while ( nb >= 8 ) { out[ o ++ ] = acc & 255; acc >>>= 8; nb -= 8; }
		for ( let i = 1; i < 16; i ++ ) {

			acc |= IX[ i ] << nb; nb += 4;
			while ( nb >= 8 ) { out[ o ++ ] = acc & 255; acc >>>= 8; nb -= 8; }

		}

	}

	return out;

}

// the next mip level: a 2 x 2 box, colour averaged as light when the layer is sRGB (as the GPU's own mips are)
const TO = new Float32Array( 256 ), FROM = new Uint8Array( 4097 );
for ( let i = 0; i < 256; i ++ ) { const c = i / 255; TO[ i ] = c <= 0.04045 ? c / 12.92 : Math.pow( ( c + 0.055 ) / 1.055, 2.4 ); }
for ( let i = 0; i <= 4096; i ++ ) { const c = i / 4096; FROM[ i ] = Math.round( 255 * ( c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow( c, 1 / 2.4 ) - 0.055 ) ); }

export function mipDown( src, w, h, srgb ) {

	const W = Math.max( 1, w >> 1 ), H = Math.max( 1, h >> 1 ), out = new Uint8Array( W * H * 4 );
	for ( let y = 0; y < H; y ++ ) {

		const y0 = Math.min( h - 1, y * 2 ) * w, y1 = Math.min( h - 1, y * 2 + 1 ) * w;
		for ( let x = 0; x < W; x ++ ) {

			const x0 = Math.min( w - 1, x * 2 ) * 4, x1 = Math.min( w - 1, x * 2 + 1 ) * 4;
			const a = y0 * 4 + x0, b = y0 * 4 + x1, c = y1 * 4 + x0, d = y1 * 4 + x1, o = ( y * W + x ) * 4;
			if ( srgb ) for ( let k = 0; k < 3; k ++ ) out[ o + k ] = FROM[ ( ( TO[ src[ a + k ] ] + TO[ src[ b + k ] ] + TO[ src[ c + k ] ] + TO[ src[ d + k ] ] ) * 1024 + 0.5 ) | 0 ];
			else for ( let k = 0; k < 3; k ++ ) out[ o + k ] = ( src[ a + k ] + src[ b + k ] + src[ c + k ] + src[ d + k ] + 2 ) >> 2;
			out[ o + 3 ] = ( src[ a + 3 ] + src[ b + 3 ] + src[ c + 3 ] + src[ d + 3 ] + 2 ) >> 2;

		}

	}

	return out;

}

// every level of a layer: [ Uint8Array of blocks ] from size x size down to 1 x 1 (enc: encodeBC7, or encodeASTC of
// astc4.js - both make 16 bytes of a 4 x 4 block)
export function encodeChain( px, size, srgb, refine = 1, times = null, enc = encodeBC7 ) {

	const mips = [];
	let w = size, h = size;
	for ( ;; ) {

		let t = times ? performance.now() : 0;
		mips.push( enc( px, w, h, new Uint8Array( Math.ceil( w / 4 ) * Math.ceil( h / 4 ) * 16 ), refine, null, srgb ) );
		if ( times ) times.encode += performance.now() - t;
		if ( w === 1 && h === 1 ) break;
		t = times ? performance.now() : 0;
		px = mipDown( px, w, h, srgb );
		if ( times ) times.mips += performance.now() - t;
		w = Math.max( 1, w >> 1 ); h = Math.max( 1, h >> 1 );

	}

	return mips;

}
