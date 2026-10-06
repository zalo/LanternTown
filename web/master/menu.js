// The page around the canvas: the veil it opens behind, the quality menu, the viewpoints.
//
// A tier is another set of textures, distances and targets: choosing one reloads the page at the place one
// stands. The effects in the same menu are uniforms and switch at once. What a visitor chose is kept under the
// walk page's own keys, so lens flare and ground shadows stay as they were set there.

const $ = ( id ) => document.getElementById( id );

export function createVeil() {

	return {
		progress( share ) { $( 'bar' ).firstElementChild.style.width = Math.round( share * 100 ) + '%'; },
		// the town is there: say how to walk, and wait for the visitor behind a button (a pointer lock needs a click)
		ready( canvas, touch ) {

			$( 'msg' ).textContent = touch ? 'left stick to walk · drag to look' : 'click to look around · WASD to walk · Shift to run';
			$( 'bar' ).style.display = 'none';
			$( 'go' ).style.display = 'block';
			$( 'go' ).onclick = () => {

				$( 'veil' ).classList.add( 'hide' );
				if ( ! touch ) canvas.requestPointerLock?.();

			};

		},
		fail( error ) { $( 'msg' ).textContent = 'failed to start: ' + ( error.message || error ); },
	};

}

// quality: the tier that runs; walk: walk.js; post: post.js (the flare); shadows: the ground shadows' switch
// (a uniform); shots: the bake's viewpoints; query: the page's
export function createMenu( { quality, walk, post, shadows, shots, query } ) {

	// --- effects: switched at once, remembered
	const effects = {
		flare: { element: $( 'flare' ), key: 'tsq.flare', text: 'Lens flare', on: localStorage.getItem( 'tsq.flare' ) === '1', set: ( on ) => { post.flare = on; } },
		gsh: { element: $( 'gshadow' ), key: 'tsq.gsh', text: 'Ground shadows', on: query.has( 'gsh' ) ? query.get( 'gsh' ) !== '0' : localStorage.getItem( 'tsq.gsh' ) !== '0', set: ( on ) => { shadows.value = on ? 1 : 0; } },
	};
	const show = ( e ) => { e.set( e.on ); e.element.textContent = `${e.text}: ${e.on ? 'on' : 'off'}`; };
	for ( const e of Object.values( effects ) ) show( e );

	// --- quality: reload where one stands
	const select = $( 'quality' );
	select.value = quality;
	select.onchange = () => {

		const effect = effects[ select.value ];
		if ( effect ) {

			effect.on = ! effect.on;
			localStorage.setItem( effect.key, effect.on ? '1' : '0' );
			show( effect );
			select.value = quality; select.blur(); // (the entries of the effects are switches, not a choice)
			return;

		}

		localStorage.setItem( 'tsq.master.quality', select.value );
		const P = walk.player, url = new URL( location.href );
		url.searchParams.set( 'quality', select.value );
		url.searchParams.set( 'at', [ P.x, P.y, P.z, P.yaw, P.pitch ].map( ( v ) => v.toFixed( 2 ) ).join( ',' ) );
		location.href = url.href;

	};

	// --- viewpoints (the button, key V) and the numbers (key H)
	let next = 0;
	const tour = () => { walk.view = shots[ next ]; next = ( next + 1 ) % shots.length; };
	$( 'tour' ).onclick = tour;
	addEventListener( 'keydown', ( e ) => {

		if ( e.code === 'KeyV' ) tour();
		if ( e.code === 'KeyH' ) $( 'hud' ).style.display = $( 'hud' ).style.display === 'none' ? '' : 'none';

	} );

}
