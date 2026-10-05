// Where one may walk: the walkable regions and the colliders of out/town/town.json (town coordinates: X east,
// Y north, Z up). The Nav class of ../engine/town.js, copied so that this page loads none of the town engine.

export class Nav {

	constructor( town ) {

		this.regions = town.regions || [];
		this.colliders = ( town.colliders || [] ).map( ( c ) => {

			if ( c.t !== 'b' ) return c;
			const a = ( c.r || 0 ) * Math.PI / 180;
			return { ...c, cos: Math.cos( a ), sin: Math.sin( a ) };

		} );
		this.radius = 0.3;
		this.step = 0.5; // max step up
		this.drop = 0.6; // max step down
		this.head = 1.9;

	}

	heightOf( r, x, y ) {

		if ( r.ramp ) {

			const q = r.ramp;
			const t = Math.max( 0, Math.min( 1, ( ( x - q.o[ 0 ] ) * q.d[ 0 ] + ( y - q.o[ 1 ] ) * q.d[ 1 ] ) / q.len ) );
			return q.z0 + ( q.z1 - q.z0 ) * t;

		}

		if ( r.arch ) {

			const R = r.rect;
			const t = r.arch.axis === 'x' ? ( x - R[ 0 ] ) / ( R[ 2 ] - R[ 0 ] ) : ( y - R[ 1 ] ) / ( R[ 3 ] - R[ 1 ] );
			return r.z + r.arch.rise * ( 1 - ( 2 * t - 1 ) ** 2 );

		}

		return r.z;

	}

	// highest walkable surface at (x, y) reachable from height z; null if there is none or something blocks the body
	groundAt( x, y, z ) {

		let best = null;
		for ( const r of this.regions ) {

			const R = r.rect;
			if ( x < R[ 0 ] || x > R[ 2 ] || y < R[ 1 ] || y > R[ 3 ] ) continue;
			const h = this.heightOf( r, x, y );
			if ( h <= z + this.step ) {

				if ( best === null || h > best ) best = h;

			} else if ( h < z + this.head ) return null; // stair flank / ledge at body height

		}

		if ( best !== null && best < z - this.drop ) return null;
		return best;

	}

	// push the point out of every collider that overlaps the body at height z; returns true if it moved
	resolve( p, z ) {

		const rad = this.radius;
		let hit = false;
		for ( const c of this.colliders ) {

			const zr = c.z;
			if ( zr && ( zr[ 1 ] < z + 0.3 || zr[ 0 ] > z + 1.7 ) ) continue;
			if ( c.t === 'c' ) {

				const dx = p.x - c.c[ 0 ], dy = p.y - c.c[ 1 ], R = c.rad + rad;
				const d2 = dx * dx + dy * dy;
				if ( d2 >= R * R ) continue;
				const d = Math.sqrt( d2 ) || 1e-6;
				p.x = c.c[ 0 ] + dx / d * R; p.y = c.c[ 1 ] + dy / d * R;
				hit = true;

			} else {

				// to box space
				const dx = p.x - c.c[ 0 ], dy = p.y - c.c[ 1 ];
				const lx = dx * c.cos + dy * c.sin, ly = - dx * c.sin + dy * c.cos;
				const hx = c.h[ 0 ], hy = c.h[ 1 ];
				const qx = Math.max( - hx, Math.min( hx, lx ) ), qy = Math.max( - hy, Math.min( hy, ly ) );
				let ex = lx - qx, ey = ly - qy;
				const d2 = ex * ex + ey * ey;
				if ( d2 >= rad * rad ) continue;
				let nx, ny;
				if ( d2 > 1e-10 ) {

					const d = Math.sqrt( d2 );
					nx = qx + ex / d * rad; ny = qy + ey / d * rad;

				} else {

					// centre is inside: leave through the nearest face
					const px = hx - Math.abs( lx ), py = hy - Math.abs( ly );
					if ( px < py ) {

						nx = Math.sign( lx || 1 ) * ( hx + rad ); ny = ly;

					} else {

						nx = lx; ny = Math.sign( ly || 1 ) * ( hy + rad );

					}

				}

				p.x = c.c[ 0 ] + nx * c.cos - ny * c.sin;
				p.y = c.c[ 1 ] + nx * c.sin + ny * c.cos;
				hit = true;

			}

		}

		return hit;

	}

	// move a body standing at (pos.x, pos.y, pos.z = feet height) by (dx, dy); slides along obstacles
	move( pos, dx, dy ) {

		const n = Math.max( 1, Math.ceil( Math.hypot( dx, dy ) / 0.2 ) );
		const p = { x: 0, y: 0 };
		for ( let i = 0; i < n; i ++ ) {

			for ( const [ sx, sy ] of [ [ dx / n, dy / n ], [ dx / n, 0 ], [ 0, dy / n ] ] ) {

				if ( sx === 0 && sy === 0 ) continue;
				p.x = pos.x + sx; p.y = pos.y + sy;
				this.resolve( p, pos.z );
				this.resolve( p, pos.z );
				const g = this.groundAt( p.x, p.y, pos.z );
				if ( g === null ) continue;
				pos.x = p.x; pos.y = p.y; pos.z = g;
				break;

			}

		}

		return pos;

	}

}
