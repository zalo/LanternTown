// The page's numbers: the HUD (key H) and window.__stats, for measuring from outside - the same fields as the
// walk page's, plus what only this page can say: how much of the one file a visit has fetched.

// renderer; hud: the element; label: the HUD's last line; url, size, head: the master file, its bytes, those of
// its first part; describe(): the town's own numbers (cells, levels, pending); gpuTime: ?gputime=1
export function createStats( { renderer, hud, label, url, size, head, describe, gpuTime = false } ) {

	// first: ms from opening the page to the first frame; worst: the longest frame so far, ms (reset it to watch a walk)
	const ST = window.__stats = { first: 0, worst: 0, frames: 0, file: { size, requests: 0, bytes: 0, status: {} } };

	// every request of the master file as the browser saw it: no wrapper around fetch, the loader stays as it is
	const href = new URL( url, location.href ).href;
	new PerformanceObserver( ( list ) => {

		for ( const e of list.getEntries() ) {

			if ( e.name !== href ) continue;
			ST.file.requests ++;
			// (the loader reads the file's first part as a stream and stops when it has the lowest levels: the browser
			// reports no size for a stream that was left, so it is counted as what had to be read of it)
			ST.file.bytes += e.encodedBodySize || ( e.responseStatus === 206 ? head : 0 );
			ST.file.status[ e.responseStatus ] = ( ST.file.status[ e.responseStatus ] || 0 ) + 1;

		}

	} ).observe( { type: 'resource', buffered: true } );

	// what the frame's render passes took on the GPU (needs timestamp queries) -> __stats.gpu, ms
	const gpu = gpuTime && renderer.backend.trackTimestamp ? { sum: 0, n: 0, busy: false } : null;
	const MB = ( n ) => ( n / 1e6 ).toFixed( n < 1e8 ? 1 : 0 );
	let acc = 0, frames = 0, last = 0;

	return {
		ST,

		// after each frame
		frame( dt ) {

			const now = performance.now(), info = renderer.info.render;
			if ( ! ST.first ) ST.first = Math.round( now );
			if ( last && now - last > ST.worst ) ST.worst = Math.round( now - last );
			last = now;
			if ( info.triangles > ( ST.peak || 0 ) ) ST.peak = info.triangles;
			if ( info.drawCalls > ( ST.peakDraws || 0 ) ) ST.peakDraws = info.drawCalls;
			if ( gpu && ! gpu.busy ) {

				gpu.busy = true;
				renderer.resolveTimestampsAsync( 'render' ).then( () => { const ms = renderer.info.render.timestamp; if ( ms > 0 ) { gpu.sum += ms; gpu.n ++; } } )
					.catch( () => {} ).then( () => { gpu.busy = false; } );

			}

			acc += dt; frames ++;
			if ( acc <= 0.5 ) return;
			const fps = Math.round( frames / acc ), town = describe();
			acc = 0; frames = 0;
			Object.assign( ST, { fps, draws: info.drawCalls, tris: info.triangles, town, frames: ST.frames + 1 } );
			if ( gpu && gpu.n ) { ST.gpu = + ( gpu.sum / gpu.n ).toFixed( 2 ); gpu.sum = gpu.n = 0; }
			hud.textContent = `${fps} fps  ${info.drawCalls} draws  ${( info.triangles / 1e6 ).toFixed( 2 )} M tris\n` +
				`${town.items} cells  LOD ${town.lods.join( '/' )}  ${town.pending ? 'streaming ' + town.pending : ''}\n` +
				`${label} · ${MB( ST.file.bytes )} of ${MB( size )} MB in ${ST.file.requests} requests`;

		},
	};

}
