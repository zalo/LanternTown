// Walking a GLB by its own triangles, with no town.json ( ?walk=bvh ): a capsule moved against a three-mesh-bvh MeshBVH
// with `shapecast`, gravity and a few sub-steps a frame - the library's character-movement example
// (https://github.com/gkjohnson/three-mesh-bvh, example/characterMovement.js) in this page's terms. A prototype, opt-in.
//
// What is collided with is one small world-space geometry made by collect(), not the drawn meshes:
//   - meshes that are not solid are left out by their material: transparent, alpha-tested, not writing depth (glass,
//     the waterfall, cut-out leaves), the bake's own extras.kind (sky, water, glass, fall, cut) and soft things by
//     name (leaves, cloth, paper); the sky node (extras.sky / a node named `sky`) and everything below it
//   - slivers of no area (closestPointToSegment has no direction to push along on those)
//   - and, only when that is more than `budget` triangles, what is outside a height band around where one stands
//     (`lo`..`hi`): roofs and upper storeys cannot be touched.
// That is all the thinning there is. Measured on town_best_new.glb (8.57 M triangles, a desktop): everything solid is
// 7.2 M triangles, 260 MB and 2.0 s of building; the band -1..4.2 m still 4.8 M, 172 MB, 1.3 s. Dropping the small
// triangles as well (longest edge under 5 / 10 / 40 cm: 3.3 / 2.4 / 0.6 M) was tried and taken out again: under 10 cm
// the capsule already sticks in the terrace stairs, under 40 cm it falls through them. And a frame's collision, 0.1 -
// 0.3 ms nearly everywhere, is 7 ms where the fine file's paving is dense (the street at x -1, y -30). A file that size
// wants a coarse level of the same town to collide with (main.js: the last MSFT_lod level, or ?collide=town_lo.glb:
// 0.42 M triangles, 18 MB, 0.15 s, 0.1 - 0.3 ms a frame everywhere tried).
// The drawn geometry is neither changed nor kept: positions are copied (in world space, vertices no kept triangle uses dropped).

import * as THREE from 'three';
import { MeshBVH, estimateMemoryInBytes, CENTER, SAH } from 'three-mesh-bvh';

const NOT_SOLID = new Set( [ 'sky', 'glass', 'water', 'fall', 'cut' ] ); // materials[].extras.kind of tools/bake2glb.mjs
export const SOFT = /leaf|leaves|foliage|canvas|cloth|bunting|paper|stripe|water|glass/i; // by material name, for any file

export function solid( o, soft = SOFT ) {

	for ( let a = o; a; a = a.parent ) if ( a.userData.sky === true || a.name === 'sky' ) return false;
	const m = Array.isArray( o.material ) ? o.material[ 0 ] : o.material;
	if ( ! m || m.transparent || m.alphaTest > 0 || m.opacity < 1 || m.depthWrite === false ) return false;
	if ( NOT_SOLID.has( m.userData.kind ) || ( soft && soft.test( m.name ) ) ) return false;
	return true;

}

// -> { geometry, stats }: the triangles of `roots` (an Object3D or several) to collide with, in world space (three: y up)
//    opt: budget (triangles; 0 = no limit), lo, hi (world heights of the band: a triangle that reaches into it is kept),
//    band ('auto': only over budget | true | false), soft (RegExp | null)
export function collect( roots, opt = {} ) {

	const t0 = performance.now();
	const budget = opt.budget || Infinity, soft = opt.soft === undefined ? SOFT : opt.soft;
	const lo = opt.lo ?? - Infinity, hi = opt.hi ?? Infinity, hasBand = opt.band !== false && ( isFinite( lo ) || isFinite( hi ) );
	const parts = [], left = {};
	let seen = 0, inside = 0, outside = 0;
	// 1. every solid triangle: in the band (0), outside it (1), or of no area (2)
	for ( const root of [].concat( roots ) ) {

		root.updateMatrixWorld( true );
		root.traverse( ( o ) => {

			const g = o.isMesh && o.geometry, pos = g && g.attributes.position;
			if ( ! pos ) return;
			const index = g.index ? g.index.array : null, n = ( ( index ? index.length : pos.count ) / 3 ) | 0;
			seen += n;
			if ( ! solid( o, soft ) ) {

				const why = ( Array.isArray( o.material ) ? o.material[ 0 ] : o.material )?.name || o.name || 'unnamed';
				left[ why ] = ( left[ why ] || 0 ) + n;
				return;

			}

			// plain float positions are read from the array, anything else (interleaved, quantized) through the attribute
			const direct = ! pos.isInterleavedBufferAttribute && ! pos.normalized && pos.array instanceof Float32Array && pos.itemSize === 3;
			const a = direct ? pos.array : null, e = o.matrixWorld.elements, where = new Uint8Array( n );
			for ( let t = 0; t < n; t ++ ) {

				const i0 = index ? index[ t * 3 ] : t * 3, i1 = index ? index[ t * 3 + 1 ] : t * 3 + 1, i2 = index ? index[ t * 3 + 2 ] : t * 3 + 2;
				const ax = a ? a[ i0 * 3 ] : pos.getX( i0 ), ay = a ? a[ i0 * 3 + 1 ] : pos.getY( i0 ), az = a ? a[ i0 * 3 + 2 ] : pos.getZ( i0 );
				const bx = a ? a[ i1 * 3 ] : pos.getX( i1 ), by = a ? a[ i1 * 3 + 1 ] : pos.getY( i1 ), bz = a ? a[ i1 * 3 + 2 ] : pos.getZ( i1 );
				const cx = a ? a[ i2 * 3 ] : pos.getX( i2 ), cy = a ? a[ i2 * 3 + 1 ] : pos.getY( i2 ), cz = a ? a[ i2 * 3 + 2 ] : pos.getZ( i2 );
				const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
				const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
				if ( nx * nx + ny * ny + nz * nz < 1e-14 ) { where[ t ] = 2; continue; }
				if ( hasBand ) {

					const y0 = e[ 1 ] * ax + e[ 5 ] * ay + e[ 9 ] * az + e[ 13 ], y1 = e[ 1 ] * bx + e[ 5 ] * by + e[ 9 ] * bz + e[ 13 ], y2 = e[ 1 ] * cx + e[ 5 ] * cy + e[ 9 ] * cz + e[ 13 ];
					if ( Math.max( y0, y1, y2 ) < lo || Math.min( y0, y1, y2 ) > hi ) { where[ t ] = 1; outside ++; continue; }

				}

				inside ++;

			}

			parts.push( { o, pos, a, index, where } );

		} );

	}

	// 2. everything if it fits the budget, else the band
	const band = hasBand && ( opt.band === true || inside + outside > budget ), most = band ? 0 : 1, tris = inside + ( band ? 0 : outside );

	// 3. those triangles as one indexed geometry
	const stage = [];
	let verts = 0;
	for ( const { o, pos, a, index, where } of parts ) {

		const mark = new Uint32Array( pos.count ); // 0 = not used, else its new index + 1
		let used = 0, k = 0;
		for ( let t = 0; t < where.length; t ++ ) if ( where[ t ] <= most ) k ++;
		if ( ! k ) continue;
		const idx = new Uint32Array( k * 3 );
		k = 0;
		for ( let t = 0; t < where.length; t ++ ) {

			if ( where[ t ] > most ) continue;
			for ( let j = 0; j < 3; j ++ ) {

				const i = index ? index[ t * 3 + j ] : t * 3 + j;
				idx[ k ++ ] = verts + ( mark[ i ] || ( mark[ i ] = ++ used ) ) - 1;

			}

		}

		stage.push( { o, pos, a, mark, idx, base: verts } );
		verts += used;

	}

	const P = new Float32Array( verts * 3 ), I = new Uint32Array( tris * 3 );
	let i0 = 0;
	for ( const { o, pos, a, mark, idx, base } of stage ) {

		const e = o.matrixWorld.elements;
		for ( let i = 0, V = mark.length; i < V; i ++ ) {

			if ( ! mark[ i ] ) continue;
			const x = a ? a[ i * 3 ] : pos.getX( i ), y = a ? a[ i * 3 + 1 ] : pos.getY( i ), z = a ? a[ i * 3 + 2 ] : pos.getZ( i );
			const j = ( base + mark[ i ] - 1 ) * 3;
			P[ j ] = e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ];
			P[ j + 1 ] = e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ];
			P[ j + 2 ] = e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ];

		}

		I.set( idx, i0 ); i0 += idx.length;

	}

	const geometry = new THREE.BufferGeometry();
	geometry.setAttribute( 'position', new THREE.BufferAttribute( P, 3 ) );
	geometry.setIndex( new THREE.BufferAttribute( I, 1 ) );
	geometry.computeBoundingBox();
	return { geometry, stats: {
		file_triangles: seen, solid_triangles: inside + outside, triangles: tris, vertices: verts, meshes: stage.length, left_out: left,
		band: band ? [ lo, hi ] : 'all', over_budget: tris > budget, // (over: built all the same; slow and heavy, see the top of this file)
		collect_ms: Math.round( performance.now() - t0 ), geometry_bytes: P.byteLength + I.byteLength,
	} };

}

// the capsule and what moves it. Three coordinates inside; the page's player P (town coordinates: x east, y north,
// z up = the height of the feet) is read and written by step(), so P can be set from outside between frames.
export class Walker {

	constructor( geometry, stats, opt = {} ) {

		const t0 = performance.now();
		// (the BVH sorts the index of the geometry it is given: ours, not a drawn one)
		this.bvh = new MeshBVH( geometry, { strategy: opt.sah ? SAH : CENTER, targetLeafSize: opt.leaf || 10 } );
		stats.build_ms = Math.round( performance.now() - t0 );
		stats.bvh_bytes = estimateMemoryInBytes( this.bvh._roots );
		stats.bytes = stats.bvh_bytes + stats.geometry_bytes;
		this.geometry = geometry; this.bounds = geometry.boundingBox; this.stats = stats;
		this.radius = opt.radius || 0.3; // as nav.js
		this.height = opt.height || 1.75; // feet to crown (the eye is at 1.65)
		this.gravity = - 30; // the example's (a game's, not 9.8: it keeps the feet on stairs going down)
		this.maxFall = 25;
		this.steps = opt.steps || 5;
		this.v = new THREE.Vector3(); this.ground = false;
		this.fellOut = 0; this.safe = null; this.spawn = null; this.ms = 0; this.calls = 0; this.worst = 0;
		this._p = new THREE.Vector3(); this._d = new THREE.Vector3(); this._seg = new THREE.Line3(); this._box = new THREE.Box3();
		this._a = new THREE.Vector3(); this._b = new THREE.Vector3(); this._ray = new THREE.Ray();
		this._last = [ NaN, NaN, NaN ]; this._since = 0;
		const seg = this._seg, box = this._box, a = this._a, b = this._b;
		this._cast = {

			intersectsBounds: ( bb ) => bb.intersectsBox( box ),
			intersectsTriangle: ( tri ) => {

				// the triangle's point nearest the capsule's axis: closer than the radius, the capsule is pushed off it
				const d = tri.closestPointToSegment( seg, a, b );
				if ( d < this.radius ) {

					const depth = this.radius - d;
					b.sub( a ).normalize();
					seg.start.addScaledVector( b, depth ); seg.end.addScaledVector( b, depth );

				}

			},

		};

	}

	// push the capsule standing with its feet at p out of the triangles; -> this._d, how far it was pushed
	_resolve( p ) {

		const r = this.radius, seg = this._seg, box = this._box, sy = p.y + r;
		seg.start.set( p.x, sy, p.z ); seg.end.set( p.x, p.y + this.height - r, p.z );
		box.min.set( p.x - r, p.y, p.z - r ); box.max.set( p.x + r, p.y + this.height, p.z + r );
		this.bvh.shapecast( this._cast );
		// (measured on the segment itself: exactly zero when nothing touched it. A rounding residue here would be
		// normalized to a whole direction by _sub() and take the fall's speed away - one then floats down)
		return this._d.set( seg.start.x - p.x, seg.start.y - sy, seg.start.z - p.z );

	}

	// one sub-step of the example's updatePlayer(): gravity, the walk asked for (mx, mz), then out of the triangles
	_sub( p, mx, mz, h ) {

		const v = this.v, still = this.ground && mx === 0 && mz === 0;
		v.y = this.ground ? h * this.gravity : Math.max( - this.maxFall, v.y + h * this.gravity );
		p.addScaledVector( v, h );
		p.x += mx; p.z += mz;
		const d = this._resolve( p );
		// pushed mostly upwards: standing on something
		this.ground = d.y > Math.abs( h * v.y * 0.25 );
		const off = Math.max( 0, d.length() - 1e-5 );
		d.normalize();
		// (not in the example: standing still on a slope, the push off it has a part downhill and one creeps, 2 - 3 cm/s
		// on the bridge and on stairs. Lifted straight up instead, as far as that same push takes one off the slope)
		if ( still && this.ground && d.y > 0.7 ) p.y += off / d.y; else p.addScaledVector( d, off );
		if ( this.ground ) v.set( 0, 0, 0 ); else v.addScaledVector( d, - d.dot( v ) );
		return off;

	}

	// the height of the first surface below a point (three coordinates), null over a hole
	below( x, y, z ) {

		this._ray.origin.set( x, y, z ); this._ray.direction.set( 0, - 1, 0 );
		const hit = this.bvh.raycastFirst( this._ray, THREE.DoubleSide );
		return hit ? hit.point.y : null;

	}

	// a frame: walk (dx, dy) town metres from P, fall, land
	step( P, dx, dy, dt ) {

		const t0 = performance.now();
		const p = this._p.set( P.x, P.z, - P.y ), L = this._last;
		if ( P.x !== L[ 0 ] || P.y !== L[ 1 ] || P.z !== L[ 2 ] ) { this.v.set( 0, 0, 0 ); this.ground = false; } // put somewhere from outside
		const n = Math.max( this.steps, Math.ceil( dt / 0.008 ) ), h = dt / n; // a long frame takes more sub-steps: a fall never crosses a radius in one
		const mx = dx / n, mz = - dy / n;
		for ( let i = 0; i < n; i ++ ) {

			this._sub( p, mx, mz, h );

		}

		// out of the world (a hole in the baked ground, a gap between two walls): back to where one last stood
		this._since += dt;
		if ( this.ground && this._since > 0.5 ) { ( this.safe || ( this.safe = new THREE.Vector3() ) ).copy( p ); this._since = 0; }
		if ( p.y < this.bounds.min.y - 10 ) {

			const to = this.fellOut && performance.now() - this._fellAt < 4000 ? this.spawn : ( this.safe || this.spawn );
			if ( to ) p.copy( to ); else p.y = this.bounds.max.y + 2;
			p.y += 0.05; this.v.set( 0, 0, 0 ); this.ground = false; this.fellOut ++; this._fellAt = performance.now();

		}

		P.x = p.x; P.y = - p.z; P.z = p.y;
		L[ 0 ] = P.x; L[ 1 ] = P.y; L[ 2 ] = P.z;
		const ms = performance.now() - t0;
		this.ms += ms; this.calls ++; if ( ms > this.worst ) this.worst = ms;

	}

	// where to begin: `spawn` (town.json: pos = x, y and the eye's height) set down on what is below it; without one,
	// dropped onto the scene from above the middle of its bounding box (the nearest place with something below)
	place( P, spawn ) {

		const bb = this.bounds;
		let x, z, g = null;
		if ( spawn ) {

			x = spawn.pos[ 0 ]; z = - spawn.pos[ 1 ];
			g = this.below( x, spawn.pos[ 2 ] ?? bb.max.y + 1, z );

		}

		if ( g === null ) {

			const cx = spawn ? x : ( bb.min.x + bb.max.x ) / 2, cz = spawn ? z : ( bb.min.z + bb.max.z ) / 2;
			const far = Math.hypot( bb.max.x - bb.min.x, bb.max.z - bb.min.z ) / 2;
			search: for ( let r = 0; r <= far; r += 1 ) {

				for ( let k = 0, m = Math.max( 1, Math.round( r * 6 ) ); k < m; k ++ ) {

					x = cx + Math.cos( k / m * Math.PI * 2 ) * r; z = cz + Math.sin( k / m * Math.PI * 2 ) * r;
					g = this.below( x, bb.max.y + 1, z );
					if ( g !== null ) break search;

				}

			}

		}

		P.x = x; P.y = - z; P.z = ( g ?? bb.max.y ) + 0.02;
		this.spawn = new THREE.Vector3( P.x, P.z, - P.y );
		this._last = [ P.x, P.y, P.z ]; this.v.set( 0, 0, 0 ); this.ground = false;
		return P;

	}

	// the collision set as a wireframe, to see what blocks ( ?bvhshow=1 )
	helper() {

		const m = new THREE.Mesh( this.geometry, new THREE.MeshBasicMaterial( { color: 0x30ff90, wireframe: true, transparent: true, opacity: 0.35, depthWrite: false } ) );
		m.matrixAutoUpdate = false; m.renderOrder = 10;
		return m;

	}

}
