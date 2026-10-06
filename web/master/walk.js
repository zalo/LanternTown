// Walking: keyboard, mouse look, a touch stick - and the two ways the page is driven from outside (a fixed
// viewpoint, a scripted path). The town's frame is X east, Y north, Z up; three's is Y up, so north is -Z.
//
// Collisions and the ground under the feet are the Nav's (the bake's walkable regions and colliders, read from
// the file); this module only turns input into steps and the walker into a camera.

import { MathUtils } from 'three';

const WALK = 2.2, RUN = 4.5; // m/s
const EYE = 1.65; // m

// nav: Nav; spawn: { pos, yaw } of the bake; at: [ x, y, z, yaw, pitch ] to stand at instead, or null;
// stick: the touch stick's element; fov: the walk's field of view
export function createWalk( { camera, canvas, nav, spawn, at = null, stick, fov, touch } ) {

	const P = { x: spawn.pos[ 0 ], y: spawn.pos[ 1 ], z: 0, yaw: MathUtils.degToRad( spawn.yaw || 0 ), pitch: 0, eye: EYE, bob: 0 };
	if ( at ) { P.x = at[ 0 ]; P.y = at[ 1 ]; P.yaw = at[ 3 ]; P.pitch = at[ 4 ]; }
	P.z = at ? nav.groundAt( P.x, P.y, at[ 2 ] + 0.05 ) ?? at[ 2 ] : nav.groundAt( P.x, P.y, 50 ) ?? 0;
	let eyeZ = P.z; // (eased: a kerb is a step, not a jolt)
	let free = null; // { pos, target, fov }: a fixed viewpoint, until the walker moves
	const keys = new Set(), push = { x: 0, y: 0 };

	// step out of a fixed viewpoint: keep looking the same way, drop to walkable ground if it is close below
	const leaveView = () => {

		if ( ! free ) return;
		const dx = free.target[ 0 ] - free.pos[ 0 ], dy = free.target[ 1 ] - free.pos[ 1 ], dz = free.target[ 2 ] - free.pos[ 2 ];
		P.yaw = Math.atan2( - dx, dy );
		P.pitch = Math.atan2( dz, Math.hypot( dx, dy ) );
		const ground = nav.groundAt( free.pos[ 0 ], free.pos[ 1 ], free.pos[ 2 ] );
		if ( ground !== null && free.pos[ 2 ] - ground < 2.6 ) { P.x = free.pos[ 0 ]; P.y = free.pos[ 1 ]; P.z = eyeZ = ground; }
		free = null;

	};

	const step = ( dx, dy ) => {

		const ox = P.x, oy = P.y;
		nav.move( P, dx, dy );
		const moved = Math.hypot( P.x - ox, P.y - oy );
		P.bob += moved * 2.1;
		return moved;

	};

	const look = ( dx, dy, k ) => {

		leaveView();
		P.yaw -= dx * k;
		P.pitch = MathUtils.clamp( P.pitch - dy * k, - 1.35, 1.35 );

	};

	// --- input
	addEventListener( 'keydown', ( e ) => keys.add( e.code ) );
	addEventListener( 'keyup', ( e ) => keys.delete( e.code ) );
	addEventListener( 'blur', () => keys.clear() );
	canvas.addEventListener( 'click', () => { if ( ! touch ) canvas.requestPointerLock?.(); } );
	addEventListener( 'mousemove', ( e ) => { if ( document.pointerLockElement === canvas ) look( e.movementX, e.movementY, 0.0022 ); } );

	// touch: the stick walks, dragging anywhere else looks
	const knob = stick.firstElementChild;
	let stickId = null, lookId = null, lx = 0, ly = 0;
	stick.addEventListener( 'pointerdown', ( e ) => { stickId = e.pointerId; stick.setPointerCapture( e.pointerId ); e.preventDefault(); } );
	stick.addEventListener( 'pointermove', ( e ) => {

		if ( e.pointerId !== stickId ) return;
		const r = stick.getBoundingClientRect();
		let x = ( e.clientX - r.left - r.width / 2 ) / ( r.width / 2 ), y = ( e.clientY - r.top - r.height / 2 ) / ( r.height / 2 );
		const m = Math.hypot( x, y );
		if ( m > 1 ) { x /= m; y /= m; }
		push.x = x * 1.3; push.y = y * 1.3; // (pushed to the rim: a run)
		knob.style.transform = `translate(${x * 36}px, ${y * 36}px)`;

	} );
	const stickEnd = ( e ) => { if ( e.pointerId === stickId ) { stickId = null; push.x = push.y = 0; knob.style.transform = ''; } };
	stick.addEventListener( 'pointerup', stickEnd );
	stick.addEventListener( 'pointercancel', stickEnd );
	canvas.addEventListener( 'pointerdown', ( e ) => { if ( e.pointerType !== 'mouse' ) { lookId = e.pointerId; lx = e.clientX; ly = e.clientY; } } );
	canvas.addEventListener( 'pointermove', ( e ) => {

		if ( e.pointerId !== lookId ) return;
		look( e.clientX - lx, e.clientY - ly, 0.0045 );
		lx = e.clientX; ly = e.clientY;

	} );
	const lookEnd = ( e ) => { if ( e.pointerId === lookId ) lookId = null; };
	canvas.addEventListener( 'pointerup', lookEnd );
	canvas.addEventListener( 'pointercancel', lookEnd );

	return {
		player: P,
		set view( f ) { free = f; }, // a fixed viewpoint { pos, target, fov } in the town's frame; null: back to the walk
		get view() { return free; },

		// one frame of walking, then the camera where the walker (or the fixed viewpoint) is
		update( dt ) {

			let f = 0, s = 0;
			if ( keys.has( 'KeyW' ) || keys.has( 'ArrowUp' ) ) f += 1;
			if ( keys.has( 'KeyS' ) || keys.has( 'ArrowDown' ) ) f -= 1;
			if ( keys.has( 'KeyD' ) ) s += 1;
			if ( keys.has( 'KeyA' ) ) s -= 1;
			if ( keys.has( 'ArrowLeft' ) ) P.yaw += 1.8 * dt;
			if ( keys.has( 'ArrowRight' ) ) P.yaw -= 1.8 * dt;
			f -= push.y; s += push.x;
			const m = Math.hypot( f, s );
			if ( m > 0.05 ) {

				leaveView();
				const d = ( keys.has( 'ShiftLeft' ) || keys.has( 'ShiftRight' ) || m > 1.25 ? RUN : WALK ) * Math.min( 1, m ) * dt;
				// yaw 0 looks north (+Y); right is +X
				step( ( - Math.sin( P.yaw ) * f + Math.cos( P.yaw ) * s ) / m * d, ( Math.cos( P.yaw ) * f + Math.sin( P.yaw ) * s ) / m * d );

			}

			// a scripted walk, for measuring from outside: window.__auto = { path: [ [ x, y ], .. ], speed, spin }
			const A = window.__auto;
			if ( A && A.path && A.path.length ) {

				leaveView();
				const ex = A.path[ 0 ][ 0 ] - P.x, ey = A.path[ 0 ][ 1 ] - P.y, d = Math.hypot( ex, ey ), reach = ( A.speed || RUN ) * dt;
				if ( d < Math.max( 0.5, reach ) || ( A.stuck || 0 ) > 40 ) { A.path.shift(); A.stuck = 0; } else {

					const heading = Math.atan2( - ex, ey ) - P.yaw, turn = Math.atan2( Math.sin( heading ), Math.cos( heading ) );
					P.yaw += turn * Math.min( 1, dt * 4 ); P.pitch *= 1 - Math.min( 1, dt * 4 );
					const moved = step( ex / d * reach, ey / d * reach );
					A.stuck = moved < reach * 0.2 ? ( A.stuck || 0 ) + 1 : 0;
					A.walked = ( A.walked || 0 ) + moved;

				}

			} else if ( A && A.spin ) { leaveView(); P.yaw += A.spin * dt; }

			eyeZ += ( P.z - eyeZ ) * Math.min( 1, dt * 10 );

			const want = free ? free.fov || 60 : fov;
			if ( camera.fov !== want ) { camera.fov = want; camera.updateProjectionMatrix(); }
			if ( free ) {

				camera.position.set( free.pos[ 0 ], free.pos[ 2 ], - free.pos[ 1 ] );
				camera.lookAt( free.target[ 0 ], free.target[ 2 ], - free.target[ 1 ] );

			} else {

				camera.position.set( P.x, eyeZ + P.eye + Math.sin( P.bob * 2 ) * 0.018, - P.y );
				camera.rotation.set( P.pitch, P.yaw, 0, 'YXZ' );

			}

		},
	};

}
