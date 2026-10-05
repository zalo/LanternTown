// ASTC encoder, 4 x 4 blocks, a small subset (?tc=1 where a device has no BC7: lite_tc.js). The same idea as
// bc7m6.js - every block is one line through colour space, two endpoints and a weight a pixel - in ASTC's terms:
// one partition, one weight a pixel (a 4 x 4 weight grid, no infill), LDR, and
//
//   a block without alpha (every alpha 255)   RGB endpoints (CEM 8), 192 levels a channel, 16 weight levels; in a
//                                              linear layer (normal x, y + roughness) also tried: two weight planes,
//                                              x and y on one and the roughness on its own (4 levels each, 160 levels
//                                              a channel) - kept where the block comes out closer
//   a block with alpha (the height)            RGBA endpoints (CEM 12), the split of the bits by how long the block's
//                                              line through colour space is:
//                                                short   192 levels a channel,  8 weight levels   (smooth blocks)
//                                                middle   96 levels a channel, 12 weight levels
//                                                long     48 levels a channel, 16 weight levels   (busy blocks)
//                                              and, where the line leaves an error and the alpha varies, two weight
//                                              planes (colour, alpha) of 4 levels each with 48 levels a channel -
//                                              kept where closer (the height that goes its own way: the relief)
//   a block of one colour                      a void-extent block: the colour itself, exact
//
// Not in it: partitions (two lines a block), a weight grid smaller than the block, the base + offset endpoint
// modes - what the offline encoders add, each for a search this one has no time for. Against BC7 mode 6 (bc7m6.js)
// the header costs ten bits more a block: the colour of the busiest layers comes out 1.5 - 2 dB lower, the height
// and the normals a little better (the two planes).
//
// ASTC stores the numbers of a block in whatever base fits the bits left ("integer sequence encoding": bits, or
// bits and a base-3 / base-5 digit shared between neighbours) - the tables below are built from the decoding rules
// of the format (Khronos data format specification, ASTC chapter), nothing is taken from another encoder.
// Plain JavaScript, no dependencies: runs in a worker (lite_tc_worker.js) and in node (measuring).

// ---- integer sequence encoding: five base-3 digits in 8 bits, three base-5 digits in 7 bits (the lowest code of each)
const TRIT = new Uint8Array( 243 ), QUINT = new Uint8Array( 125 );
for ( let T = 255; T >= 0; T -- ) {

	let c, t4, t3;
	if ( ( ( T >> 2 ) & 7 ) === 7 ) { c = ( ( T >> 5 ) << 2 ) | ( T & 3 ); t4 = t3 = 2; } else {

		c = T & 31;
		if ( ( ( T >> 5 ) & 3 ) === 3 ) { t4 = 2; t3 = T >> 7; } else { t4 = T >> 7; t3 = ( T >> 5 ) & 3; }

	}

	let t2, t1, t0;
	if ( ( c & 3 ) === 3 ) { t2 = 2; t1 = c >> 4; t0 = ( ( ( c >> 3 ) & 1 ) << 1 ) | ( ( ( c >> 2 ) & 1 ) & ~ ( ( c >> 3 ) & 1 ) ); }
	else if ( ( ( c >> 2 ) & 3 ) === 3 ) { t2 = 2; t1 = 2; t0 = c & 3; }
	else { t2 = c >> 4; t1 = ( c >> 2 ) & 3; t0 = ( ( ( c >> 1 ) & 1 ) << 1 ) | ( ( c & 1 ) & ~ ( ( c >> 1 ) & 1 ) ); }
	TRIT[ t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4 ] = T;

}

for ( let Q = 127; Q >= 0; Q -- ) {

	let q2, q1, q0;
	if ( ( ( Q >> 1 ) & 3 ) === 3 && ( ( Q >> 5 ) & 3 ) === 0 ) {

		const b = Q & 1;
		q2 = ( b << 2 ) | ( ( ( ( Q >> 4 ) & 1 ) & ~ b ) << 1 ) | ( ( ( Q >> 3 ) & 1 ) & ~ b ); q1 = q0 = 4;

	} else {

		let c;
		if ( ( ( Q >> 1 ) & 3 ) === 3 ) { q2 = 4; c = ( ( ( Q >> 3 ) & 3 ) << 3 ) | ( ( ~ ( Q >> 5 ) & 3 ) << 1 ) | ( Q & 1 ); } else { q2 = ( Q >> 5 ) & 3; c = Q & 31; }
		if ( ( c & 7 ) === 5 ) { q1 = 4; q0 = c >> 3; } else { q1 = c >> 3; q0 = c & 7; }

	}

	QUINT[ q0 + 5 * q1 + 25 * q2 ] = Q;

}

const REV = new Uint8Array( 256 );
for ( let i = 0; i < 256; i ++ ) { let r = 0; for ( let b = 0; b < 8; b ++ ) if ( i & ( 1 << b ) ) r |= 128 >> b; REV[ i ] = r; }

// n bits of v into buf at bit `pos`, low bit first -> the next position
function put( buf, pos, v, n ) {

	while ( n > 0 ) {

		const s = pos & 7, k = 8 - s < n ? 8 - s : n;
		buf[ pos >> 3 ] |= ( v & ( ( 1 << k ) - 1 ) ) << s;
		v >>= k; pos += k; n -= k;

	}

	return pos;

}

// kind: 0 bits only, 1 with a base-3 digit above the bits, 2 with a base-5 digit
function ise( buf, pos, v, n, bits, kind ) {

	const mask = ( 1 << bits ) - 1;
	if ( kind === 0 ) { for ( let i = 0; i < n; i ++ ) pos = put( buf, pos, v[ i ], bits ); return pos; }
	if ( kind === 1 ) {

		for ( let i = 0; i < n; i += 5 ) {

			const m = n - i;
			const T = TRIT[ ( v[ i ] >> bits ) + ( m > 1 ? 3 * ( v[ i + 1 ] >> bits ) : 0 ) + ( m > 2 ? 9 * ( v[ i + 2 ] >> bits ) : 0 ) + ( m > 3 ? 27 * ( v[ i + 3 ] >> bits ) : 0 ) + ( m > 4 ? 81 * ( v[ i + 4 ] >> bits ) : 0 ) ];
			pos = put( buf, pos, ( v[ i ] & mask ) | ( ( T & 3 ) << bits ), bits + 2 );
			if ( m > 1 ) pos = put( buf, pos, ( v[ i + 1 ] & mask ) | ( ( ( T >> 2 ) & 3 ) << bits ), bits + 2 );
			if ( m > 2 ) pos = put( buf, pos, ( v[ i + 2 ] & mask ) | ( ( ( T >> 4 ) & 1 ) << bits ), bits + 1 );
			if ( m > 3 ) pos = put( buf, pos, ( v[ i + 3 ] & mask ) | ( ( ( T >> 5 ) & 3 ) << bits ), bits + 2 );
			if ( m > 4 ) pos = put( buf, pos, ( v[ i + 4 ] & mask ) | ( ( ( T >> 7 ) & 1 ) << bits ), bits + 1 );

		}

		return pos;

	}

	for ( let i = 0; i < n; i += 3 ) {

		const m = n - i;
		const q = QUINT[ ( v[ i ] >> bits ) + ( m > 1 ? 5 * ( v[ i + 1 ] >> bits ) : 0 ) + ( m > 2 ? 25 * ( v[ i + 2 ] >> bits ) : 0 ) ];
		pos = put( buf, pos, ( v[ i ] & mask ) | ( ( q & 7 ) << bits ), bits + 3 );
		if ( m > 1 ) pos = put( buf, pos, ( v[ i + 1 ] & mask ) | ( ( ( q >> 3 ) & 3 ) << bits ), bits + 2 );
		if ( m > 2 ) pos = put( buf, pos, ( v[ i + 2 ] & mask ) | ( ( ( q >> 5 ) & 3 ) << bits ), bits + 2 );

	}

	return pos;

}

const iseBits = ( n, bits, kind ) => n * bits + ( kind === 1 ? Math.ceil( n * 8 / 5 ) : kind === 2 ? Math.ceil( n * 7 / 3 ) : 0 );

// ---- what a stored number stands for. Colour: 0 .. 255; a weight: 0 .. 64
function colourValue( v, bits, kind ) {

	if ( kind === 0 ) return bits === 8 ? v : ( ( v << ( 8 - bits ) ) | ( v >> ( 2 * bits - 8 ) ) ) & 255;
	const D = v >> bits, a = v & 1, b = ( v >> 1 ) & 1, c = ( v >> 2 ) & 1, d = ( v >> 3 ) & 1, e = ( v >> 4 ) & 1, f = ( v >> 5 ) & 1;
	let B, C;
	if ( kind === 1 ) {

		if ( bits === 4 ) { B = ( d << 8 ) | ( c << 7 ) | ( b << 6 ) | ( d << 2 ) | ( c << 1 ) | b; C = 22; }
		else if ( bits === 5 ) { B = ( e << 8 ) | ( d << 7 ) | ( c << 6 ) | ( b << 5 ) | ( e << 1 ) | d; C = 11; }
		else if ( bits === 6 ) { B = ( f << 8 ) | ( e << 7 ) | ( d << 6 ) | ( c << 5 ) | ( b << 4 ) | f; C = 5; }
		else throw new Error( 'colour range' );

	} else {

		if ( bits === 4 ) { B = ( d << 8 ) | ( c << 7 ) | ( b << 6 ) | ( d << 1 ) | c; C = 13; }
		else if ( bits === 5 ) { B = ( e << 8 ) | ( d << 7 ) | ( c << 6 ) | ( b << 5 ) | e; C = 6; }
		else throw new Error( 'colour range' );

	}

	const A = a ? 0x1ff : 0;
	let T = D * C + B; T ^= A;
	return ( A & 0x80 ) | ( T >> 2 );

}

function weightValue( v, bits, kind ) {

	let T;
	if ( kind === 0 ) T = bits === 4 ? ( v << 2 ) | ( v >> 2 ) : bits === 3 ? ( v << 3 ) | v : bits === 2 ? v * 21 : bits === 1 ? v * 63 : 0;
	else if ( bits === 0 ) T = kind === 1 ? [ 0, 32, 63 ][ v ] : [ 0, 16, 32, 47, 63 ][ v ];
	else {

		const D = v >> bits, a = v & 1, b = ( v >> 1 ) & 1;
		let B, C;
		if ( kind === 1 && bits === 1 ) { B = 0; C = 50; }
		else if ( kind === 2 && bits === 1 ) { B = 0; C = 28; }
		else if ( kind === 1 && bits === 2 ) { B = ( b << 6 ) | ( b << 2 ) | b; C = 23; }
		else throw new Error( 'weight range' );
		const A = a ? 0x7f : 0;
		T = D * C + B; T ^= A; T = ( A & 0x20 ) | ( T >> 2 );

	}

	return T > 32 ? T + 1 : T;

}

const COLOUR = [ [ 8, 0 ], [ 6, 1 ], [ 5, 2 ], [ 7, 0 ], [ 5, 1 ], [ 4, 2 ], [ 6, 0 ], [ 4, 1 ], [ 5, 0 ] ]; // most levels first: 256 192 160 128 96 80 64 48 32
const CT = new Map();
function colourTable( bits, kind ) {

	const key = bits * 4 + kind;
	if ( CT.has( key ) ) return CT.get( key );
	const n = ( kind === 1 ? 3 : kind === 2 ? 5 : 1 ) << bits, val = new Uint8Array( n ), inv = new Uint8Array( 256 );
	for ( let v = 0; v < n; v ++ ) val[ v ] = colourValue( v, bits, kind );
	for ( let x = 0; x < 256; x ++ ) { let best = 999, bi = 0; for ( let v = 0; v < n; v ++ ) { const d = Math.abs( val[ v ] - x ); if ( d < best ) { best = d; bi = v; } } inv[ x ] = bi; }
	const t = { bits, kind, n, val, inv };
	CT.set( key, t );
	return t;

}

const WEIGHT = { 3: [ 0, 1, 3, 0 ], 4: [ 2, 0, 4, 0 ], 5: [ 0, 2, 5, 0 ], 6: [ 1, 1, 6, 0 ], 8: [ 3, 0, 7, 0 ], 10: [ 1, 2, 2, 1 ], 12: [ 2, 1, 3, 1 ], 16: [ 4, 0, 4, 1 ] }; // levels: bits, kind, R, H of the block mode

// cem: 8 (RGB) or 12 (RGBA); wn: weight levels; dual: a second weight plane for the alpha
function config( cem, wn, dual ) {

	const [ wbits, wkind, R, H ] = WEIGHT[ wn ], nv = cem === 8 ? 6 : 8;
	const order = Array.from( { length: wn }, ( _, v ) => v ).sort( ( a, b ) => weightValue( a, wbits, wkind ) - weightValue( b, wbits, wkind ) );
	const wv = new Float64Array( order.map( ( v ) => weightValue( v, wbits, wkind ) / 64 ) ), wi = new Uint8Array( order ), wq = new Uint8Array( order.map( ( v ) => weightValue( v, wbits, wkind ) ) );
	const lut = new Uint8Array( 257 );
	for ( let t = 0; t <= 256; t ++ ) { let best = 9, bi = 0; for ( let k = 0; k < wn; k ++ ) { const d = Math.abs( wv[ k ] - t / 256 ); if ( d < best ) { best = d; bi = k; } } lut[ t ] = bi; }
	const wtotal = iseBits( dual ? 32 : 16, wbits, wkind ), left = 128 - 17 - wtotal - ( dual ? 2 : 0 );
	const [ cbits, ckind ] = COLOUR.find( ( [ b, k ] ) => iseBits( nv, b, k ) <= left );
	return { cem, nv, nch: nv / 2, wn, wbits, wkind, wv, wi, wq, lut, dual, wtotal, cq: colourTable( cbits, ckind ),
		head: ( ( R & 1 ) << 4 ) | ( ( R >> 1 ) & 3 ) | ( 2 << 5 ) | ( H << 9 ) | ( ( dual ? 1 : 0 ) << 10 ) | ( cem << 13 ) };

}

const RGB16 = config( 8, 16, false );  // 192 levels a channel
const A8 = config( 12, 8, false );    // 192
const A12 = config( 12, 12, false );  // 96
const A16 = config( 12, 16, false );  // 48
const A10 = config( 12, 10, false );  // 128
const DP4 = config( 12, 4, true );    // 48, colour and alpha each with weights of their own
const RGBDP4 = config( 8, 4, true );  // 160, r g and b (the roughness beside a normal) each with weights of their own
const SETS = { 8: [ A8 ], 16: [ A16 ], 12: [ A12 ], 10: [ A10 ] };

// ---- one block
const P = new Float64Array( 64 ), M = new Float64Array( 4 ), E = new Float64Array( 8 ), CV = new Float64Array( 16 );
const QE = new Float64Array( 8 ), QI = new Uint8Array( 8 ), K = new Uint8Array( 32 );
const BQE = new Float64Array( 8 ), BQI = new Uint8Array( 8 ), BK = new Uint8Array( 32 );
const VALS = new Uint8Array( 32 ), BLK = new Uint8Array( 16 ), WB = new Uint8Array( 16 );

function quant( cq, c0, c1 ) {

	for ( let k = 0; k < 2; k ++ ) for ( let c = c0; c < c1; c ++ ) {

		const e = E[ k * 4 + c ], i = cq.inv[ e <= 0 ? 0 : e >= 255 ? 255 : ( e + 0.5 ) | 0 ];
		QI[ k * 4 + c ] = i; QE[ k * 4 + c ] = cq.val[ i ];

	}

}

// the weight of each pixel on the line QE0 -> QE1 over the channels c0 .. c1 - 1, into K at `at` (step 1)
function weights( cfg, c0, c1, at ) {

	let dd = 0;
	for ( let c = c0; c < c1; c ++ ) { const d = QE[ 4 + c ] - QE[ c ]; dd += d * d; }
	if ( dd === 0 ) { for ( let i = 0; i < 16; i ++ ) K[ at + i ] = 0; return; }
	const s = 256 / dd, lut = cfg.lut;
	for ( let i = 0; i < 16; i ++ ) {

		let t = 0;
		for ( let c = c0; c < c1; c ++ ) t += ( P[ i * 4 + c ] - QE[ c ] ) * ( QE[ 4 + c ] - QE[ c ] );
		t = ( t * s + 0.5 ) | 0;
		K[ at + i ] = lut[ t < 0 ? 0 : t > 256 ? 256 : t ];

	}

}

// the endpoints that fit these weights best (least squares, a channel at a time) -> E; false: nothing to solve
function solve( cfg, c0, c1, at ) {

	const wv = cfg.wv;
	let aa = 0, ab = 0, bb = 0;
	for ( let i = 0; i < 16; i ++ ) { const b = wv[ K[ at + i ] ], a = 1 - b; aa += a * a; ab += a * b; bb += b * b; }
	const det = aa * bb - ab * ab;
	if ( det < 1e-4 ) return false;
	for ( let c = c0; c < c1; c ++ ) {

		let ax = 0, bx = 0;
		for ( let i = 0; i < 16; i ++ ) { const b = wv[ K[ at + i ] ], x = P[ i * 4 + c ]; ax += ( 1 - b ) * x; bx += b * x; }
		const e0 = ( bb * ax - ab * bx ) / det, e1 = ( aa * bx - ab * ax ) / det;
		E[ c ] = e0 < 0 ? 0 : e0 > 255 ? 255 : e0; E[ 4 + c ] = e1 < 0 ? 0 : e1 > 255 ? 255 : e1;

	}

	return true;

}

function error( cfg, c0, c1, at ) {

	const wv = cfg.wv;
	let err = 0;
	for ( let i = 0; i < 16; i ++ ) {

		const b = wv[ K[ at + i ] ];
		for ( let c = c0; c < c1; c ++ ) { const d = QE[ c ] + ( QE[ 4 + c ] - QE[ c ] ) * b - P[ i * 4 + c ]; err += d * d; }

	}

	return err;

}

// the principal axis of the channels c0 .. c1 - 1 through the mean -> E (both ends, clamped)
function axis( c0, c1, lo, hi ) {

	const n = c1 - c0;
	CV.fill( 0 );
	for ( let i = 0; i < 64; i += 4 ) for ( let a = 0; a < n; a ++ ) { const da = P[ i + c0 + a ] - M[ c0 + a ]; for ( let b = a; b < n; b ++ ) CV[ a * 4 + b ] += da * ( P[ i + c0 + b ] - M[ c0 + b ] ); }
	for ( let a = 1; a < n; a ++ ) for ( let b = 0; b < a; b ++ ) CV[ a * 4 + b ] = CV[ b * 4 + a ];
	// (start: the box's diagonal, its signs those of the covariance with the widest channel)
	let wide = 0;
	for ( let a = 1; a < n; a ++ ) if ( hi[ c0 + a ] - lo[ c0 + a ] > hi[ c0 + wide ] - lo[ c0 + wide ] ) wide = a;
	const v = [ 0, 0, 0, 0 ];
	for ( let a = 0; a < n; a ++ ) v[ a ] = ( hi[ c0 + a ] - lo[ c0 + a ] ) * ( CV[ wide * 4 + a ] < 0 ? - 1 : 1 );
	for ( let it = 0; it < 3; it ++ ) {

		const r = [ 0, 0, 0, 0 ];
		let m = 0;
		for ( let a = 0; a < n; a ++ ) { for ( let b = 0; b < n; b ++ ) r[ a ] += CV[ a * 4 + b ] * v[ b ]; if ( Math.abs( r[ a ] ) > m ) m = Math.abs( r[ a ] ); }
		if ( m < 1e-9 ) break;
		for ( let a = 0; a < n; a ++ ) v[ a ] = r[ a ] / m;

	}

	let vv = 0;
	for ( let a = 0; a < n; a ++ ) vv += v[ a ] * v[ a ];
	let t0 = 1e30, t1 = - 1e30;
	if ( vv > 0 ) for ( let i = 0; i < 64; i += 4 ) {

		let t = 0;
		for ( let a = 0; a < n; a ++ ) t += ( P[ i + c0 + a ] - M[ c0 + a ] ) * v[ a ];
		t /= vv;
		if ( t < t0 ) t0 = t; if ( t > t1 ) t1 = t;

	} else t0 = t1 = 0;
	for ( let a = 0; a < n; a ++ ) {

		const x = M[ c0 + a ] + v[ a ] * t0, y = M[ c0 + a ] + v[ a ] * t1;
		E[ c0 + a ] = x < 0 ? 0 : x > 255 ? 255 : x; E[ 4 + c0 + a ] = y < 0 ? 0 : y > 255 ? 255 : y;

	}

}

const keep = () => { BQE.set( QE ); BQI.set( QI ); BK.set( K ); };

// src: RGBA8, w x h -> out: ceil( w / 4 ) * ceil( h / 4 ) blocks of 16 bytes. dec (optional): what the blocks decode to
// (as 8 bits; srgb: the layer is uploaded as an sRGB format). opt: { rgba: [ weight levels to try ], dual: 0 | 1 } (measuring)
export function encodeASTC( src, w, h, out, refine = 1, dec = null, srgb = false, opt = null ) {

	const bw = Math.ceil( w / 4 ), bh = Math.ceil( h / 4 );
	const fixed = opt && opt.rgba ? opt.rgba.map( ( n ) => SETS[ n ][ 0 ] ) : null, dualOn = opt && opt.dual !== undefined ? opt.dual : 1;
	const T1 = opt && opt.t1 !== undefined ? opt.t1 : 1200, T2 = opt && opt.t2 !== undefined ? opt.t2 : 15600, ONE = [ null ];
	const rgbDual = ! srgb && ( opt && opt.rgbdual !== undefined ? opt.rgbdual : 1 ), BW = opt && opt.bw !== undefined ? opt.bw : 0.25, G = opt && opt.g !== undefined ? opt.g : 0;
	const lo = [ 0, 0, 0, 0 ], hi = [ 0, 0, 0, 0 ];
	const count = opt && opt.count ? opt.count : null, top8 = !! ( opt && opt.top8 ); // (top8: as astcenc writes a linear file to 8 bits)
	let o = 0;
	for ( let by = 0; by < bh; by ++ ) for ( let bx = 0; bx < bw; bx ++, o += 16 ) {

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

		if ( hi0 === lo0 && hi1 === lo1 && hi2 === lo2 && hi3 === lo3 ) {

			// one colour: a void-extent block (no extent given), the colour as 16 bits a channel
			out[ o ] = 0xfc; out[ o + 1 ] = 0xfd; for ( let i = 2; i < 8; i ++ ) out[ o + i ] = 0xff;
			out[ o + 8 ] = out[ o + 9 ] = lo0; out[ o + 10 ] = out[ o + 11 ] = lo1; out[ o + 12 ] = out[ o + 13 ] = lo2; out[ o + 14 ] = out[ o + 15 ] = lo3;
			if ( dec ) for ( let y = 0; y < 4; y ++ ) for ( let x = 0; x < 4; x ++ ) {

				const X = bx * 4 + x, Y = by * 4 + y;
				if ( X >= w || Y >= h ) continue;
				const d = ( Y * w + X ) * 4;
				dec[ d ] = lo0; dec[ d + 1 ] = lo1; dec[ d + 2 ] = lo2; dec[ d + 3 ] = lo3;

			}

			if ( count ) count.solid ++;
			continue;

		}

		M[ 0 ] = m0 / 16; M[ 1 ] = m1 / 16; M[ 2 ] = m2 / 16; M[ 3 ] = m3 / 16;
		lo[ 0 ] = lo0; lo[ 1 ] = lo1; lo[ 2 ] = lo2; lo[ 3 ] = lo3; hi[ 0 ] = hi0; hi[ 1 ] = hi1; hi[ 2 ] = hi2; hi[ 3 ] = hi3;
		let cfg;
		if ( lo3 === 255 ) {

			cfg = RGB16;
			axis( 0, 3, lo, hi );
			quant( cfg.cq, 0, 3 ); weights( cfg, 0, 3, 0 );
			for ( let pass = 0; pass < refine; pass ++ ) { if ( ! solve( cfg, 0, 3, 0 ) ) break; quant( cfg.cq, 0, 3 ); weights( cfg, 0, 3, 0 ); }
			// a normal (r, g) and its roughness (b): b with weights of its own where that is the smaller error (b counting BW)
			const e1 = rgbDual ? error( cfg, 0, 2, 0 ) + BW * error( cfg, 2, 3, 0 ) : 0;
			if ( e1 > G ) {

				keep();
				const d = RGBDP4;
				axis( 0, 2, lo, hi ); E[ 2 ] = lo2; E[ 6 ] = hi2;
				quant( d.cq, 0, 3 ); weights( d, 0, 2, 0 ); weights( d, 2, 3, 16 );
				for ( let pass = 0; pass < refine; pass ++ ) { const a = solve( d, 0, 2, 0 ), b = solve( d, 2, 3, 16 ); if ( ! a && ! b ) break; quant( d.cq, 0, 3 ); weights( d, 0, 2, 0 ); weights( d, 2, 3, 16 ); }
				if ( error( d, 0, 2, 0 ) + BW * error( d, 2, 3, 16 ) < e1 ) cfg = d; else { QE.set( BQE ); QI.set( BQI ); K.set( BK ); }

			}

			QE[ 3 ] = QE[ 7 ] = 255;

		} else {

			axis( 0, 4, lo, hi );
			// (a short line: fine endpoints and few weights; a long one: the other way round)
			let len = 0;
			for ( let c = 0; c < 4; c ++ ) len += ( E[ 4 + c ] - E[ c ] ) * ( E[ 4 + c ] - E[ c ] );
			const sets = fixed || ( ONE[ 0 ] = len < T1 ? A8 : len < T2 ? A12 : A16, ONE );
			const e0 = E[ 0 ], e1 = E[ 1 ], e2 = E[ 2 ], e3 = E[ 3 ], e4 = E[ 4 ], e5 = E[ 5 ], e6 = E[ 6 ], e7 = E[ 7 ];
			let best = 1e30, kept = false;
			for ( let s = 0; s < sets.length; s ++ ) {

				const c = sets[ s ];
				if ( s ) { E[ 0 ] = e0; E[ 1 ] = e1; E[ 2 ] = e2; E[ 3 ] = e3; E[ 4 ] = e4; E[ 5 ] = e5; E[ 6 ] = e6; E[ 7 ] = e7; }
				quant( c.cq, 0, 4 ); weights( c, 0, 4, 0 );
				for ( let pass = 0; pass < refine; pass ++ ) { if ( ! solve( c, 0, 4, 0 ) ) break; quant( c.cq, 0, 4 ); weights( c, 0, 4, 0 ); }
				if ( sets.length === 1 ) { cfg = c; break; }
				const err = error( c, 0, 4, 0 );
				if ( err < best ) { best = err; cfg = c; keep(); kept = true; }

			}

			// two planes when the line leaves a real error (its mean square a pixel and channel above 1): colour on one, alpha on the other
			if ( dualOn && hi3 > lo3 && ( kept ? best : ( best = error( cfg, 0, 4, 0 ) ) ) > 64 ) {

				const d = DP4;
				if ( ! kept ) { keep(); kept = true; }
				axis( 0, 3, lo, hi ); E[ 3 ] = lo3; E[ 7 ] = hi3;
				quant( d.cq, 0, 4 ); weights( d, 0, 3, 0 ); weights( d, 3, 4, 16 );
				for ( let pass = 0; pass < refine; pass ++ ) { const a = solve( d, 0, 3, 0 ), b = solve( d, 3, 4, 16 ); if ( ! a && ! b ) break; quant( d.cq, 0, 4 ); weights( d, 0, 3, 0 ); weights( d, 3, 4, 16 ); }
				const err = error( d, 0, 3, 0 ) + error( d, 3, 4, 16 );
				if ( err < best ) { best = err; cfg = d; keep(); }

			}

			if ( kept ) { QE.set( BQE ); QI.set( BQI ); K.set( BK ); }

		}

		if ( count ) count[ cfg === RGB16 ? 'rgb' : cfg === RGBDP4 ? 'rgbdual' : cfg.dual ? 'dual' : 'w' + cfg.wn ] ++;

		// (the decoder reads the endpoints the other way round, and mixes blue into red and green, when the first is the
		// brighter one: the brighter one last, the weights turned with it)
		const wn = cfg.wn, alphaPlane = cfg.dual && cfg.cem === 12, bluePlane = cfg.dual && cfg.cem === 8;
		if ( QE[ 4 ] + QE[ 5 ] + QE[ 6 ] < QE[ 0 ] + QE[ 1 ] + QE[ 2 ] ) {

			for ( let c = 0; c < 4; c ++ ) { const t = QE[ c ]; QE[ c ] = QE[ 4 + c ]; QE[ 4 + c ] = t; const u = QI[ c ]; QI[ c ] = QI[ 4 + c ]; QI[ 4 + c ] = u; }
			for ( let i = 0; i < ( cfg.dual ? 32 : 16 ); i ++ ) K[ i ] = wn - 1 - K[ i ];

		}

		if ( dec ) for ( let y = 0; y < 4; y ++ ) for ( let x = 0; x < 4; x ++ ) {

			const X = bx * 4 + x, Y = by * 4 + y;
			if ( X >= w || Y >= h ) continue;
			const i = y * 4 + x, d = ( Y * w + X ) * 4;
			for ( let c = 0; c < 4; c ++ ) {

				const wt = cfg.wq[ K[ ( ( alphaPlane && c === 3 ) || ( bluePlane && c === 2 ) ? 16 : 0 ) + i ] ];
				const a = srgb && c < 3 ? ( QE[ c ] << 8 ) | 0x80 : QE[ c ] * 257, b = srgb && c < 3 ? ( QE[ 4 + c ] << 8 ) | 0x80 : QE[ 4 + c ] * 257;
				const v = ( a * ( 64 - wt ) + b * wt + 32 ) >> 6; // 16 bits: the colour of an sRGB format is its top 8, anything else v / 65535
				dec[ d + c ] = ( srgb && c < 3 ) || top8 ? v >> 8 : ( v * 255 + 32767 ) / 65535 | 0;

			}

		}

		// 128 bits: block mode (11), partitions - 1 (2), endpoint mode (4), the endpoints upwards from bit 17; the weights
		// downwards from bit 127, and with two planes the channel of the second one (2 bits) below them
		BLK.fill( 0 ); WB.fill( 0 );
		put( BLK, 0, cfg.head, 17 );
		for ( let c = 0; c < cfg.nch; c ++ ) { VALS[ c * 2 ] = QI[ c ]; VALS[ c * 2 + 1 ] = QI[ 4 + c ]; }
		ise( BLK, 17, VALS, cfg.nv, cfg.cq.bits, cfg.cq.kind );
		if ( cfg.dual ) { for ( let i = 0; i < 16; i ++ ) { VALS[ i * 2 ] = cfg.wi[ K[ i ] ]; VALS[ i * 2 + 1 ] = cfg.wi[ K[ 16 + i ] ]; } put( BLK, 126 - cfg.wtotal, bluePlane ? 2 : 3, 2 ); }
		else for ( let i = 0; i < 16; i ++ ) VALS[ i ] = cfg.wi[ K[ i ] ];
		ise( WB, 0, VALS, cfg.dual ? 32 : 16, cfg.wbits, cfg.wkind );
		for ( let i = 0; i < 16; i ++ ) out[ o + i ] = BLK[ i ] | REV[ WB[ 15 - i ] ];

	}

	return out;

}

// a block of one colour (a blank layer)
export function solidASTC( r, g, b, a ) {

	return new Uint8Array( [ 0xfc, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, r, r, g, g, b, b, a, a ] );

}
