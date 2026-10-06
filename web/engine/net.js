// net.js - fetches that end: the baked town's files are asked for through this, so that a request which stops
// arriving is asked for again instead of being waited for forever.
//
// What goes wrong without it: a phone's tab that is put in the background while the town loads comes back with its
// connection dead - the responses on it neither arrive nor fail. fetch() has no timeout, so base.glb (the lowest
// level of every cell) stays where it stopped, the few level files in flight keep their places in the queue for
// good, and the rest of the town never loads. The same happens on a link that drops for a moment.
//
//   openBytes( url, init, tries )   the body in pieces, pulled: await read() -> { value } | { restart: true } | { done: true }
//   fetchSafe( url, init )          a drop-in for fetch(): a Response whose body is already here
//
// A request is asked for again when nothing of it has arrived for STALL ms (WAIT ms for the answer's first byte,
// REWAIT ms when it is asked for again: the browser may hand it a connection that is as dead as the last one),
// when it fails, and on 408 / 429 / 5xx. What had arrived is kept: the next request asks for the rest
// (`Range: bytes=<got>-`), and a server that sends the whole file again instead (200, not 206) makes read() say
// { restart: true } - the caller drops what it holds. After TRIES requests in a row that brought nothing, read()
// throws (`net: true` on the error: the network, not the file). Any other answer that is not ok is the caller's:
// status() gives it and read() ends at once.
// While the page is hidden nothing is timed (browsers hold a hidden tab's requests); when it shows again - or when
// the page itself stood still for seconds, as it does while a town is built - what is open has GRACE ms to bring a
// byte. A request that waits for its answer while others are arriving is in a queue behind them (HTTP/1 gives a host
// six connections), not lost: it is left alone for three times as long, and asking for it again is not counted.
// netStats: what happened, for __stats and the reports of telemetry.js. Each retry is one console.warn.
// ?net=0 on the page: nothing is timed (fetch as it is - to see what a stalled link does without this).

const STALL = 12000, WAIT = 20000, REWAIT = 8000, GRACE = 4000, BUSY = 4000, TRIES = 5, DELAY = [ 300, 1000, 2500, 5000 ];
export const FOREVER = 1e9; // openBytes( url, init, FOREVER ): never given up (a file the page cannot do without)
const now = () => performance.now();
const OFF = typeof location !== 'undefined' && /[?&]net=0\b/.test( location.search );

export const netStats = { open: 0, retries: 0, stalls: 0, errors: 0, resumed: 0, restarted: 0, gaveUp: 0, last: '' };

// ---------------------------------------------------------------------------------------------- the watchdog

const watched = new Set(); // { seen, limit, head, kill }
let timer = 0, ticked = 0, lastByte = 0; // (lastByte: when a piece of any request last arrived)

const grace = ( t ) => { for ( const w of watched ) w.seen = Math.max( w.seen, t - w.limit + GRACE ); };

function tick() {

	const t = now(), late = t - ticked > 3000;
	ticked = t;
	if ( typeof document !== 'undefined' && document.hidden ) return;
	if ( late ) return grace( t ); // (the page stood still: what arrived meanwhile has not been handed over yet)
	for ( const w of watched ) {

		const over = t - w.seen;
		if ( over <= w.limit ) continue;
		const queued = w.head && t - lastByte < BUSY;
		if ( queued && over <= w.limit * 3 ) continue;
		watched.delete( w );
		w.kill( queued );

	}

	if ( ! watched.size ) { clearInterval( timer ); timer = 0; }

}

if ( typeof document !== 'undefined' ) document.addEventListener( 'visibilitychange', () => {

	if ( ! document.hidden ) grace( ticked = now() );

} );

// p, or a rejection (`stalled: true`) when it has not settled `limit` ms from now, hidden time not counted
// (head: p is the wait for an answer, which may be a place in a queue)
function guard( p, limit, ctl, head ) {

	if ( OFF ) return p;
	return new Promise( ( resolve, reject ) => {

		const w = { seen: now(), limit, head: !! head, kill: ( queued ) => {

			const e = new Error( 'no answer' );
			e.stalled = true; e.queued = queued;
			reject( e );
			try { ctl.abort(); } catch ( x ) { /* (the rejection above is what counts) */ }

		} };
		watched.add( w );
		if ( ! timer ) { ticked = now(); timer = setInterval( tick, 1000 ); }
		p.then( ( v ) => { watched.delete( w ); resolve( v ); }, ( e ) => { watched.delete( w ); reject( e ); } );

	} );

}

const sleep = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );
const again = ( status ) => status === 408 || status === 429 || status >= 500;
const short = ( url ) => String( url ).split( '?' )[ 0 ].split( '/' ).pop();

// ---------------------------------------------------------------------------------------------- the body in pieces

class Bytes {

	constructor( url, init, tries ) {

		this.url = url; this.init = init || {}; this.max = tries || TRIES;
		this.answered = 0; // when the first answer came (ms of the page)
		this.got = 0; // bytes handed out (of the file as it is on the server: a compressed transfer is unpacked by then)
		this.tries = 0; // requests in a row that brought nothing
		this.info = null; // { ok, status, statusText, type } of the answer in use
		this._rd = null; this._ctl = null; this._restart = false; this._over = false;
		netStats.open ++;
		this._first = this._connect().catch( ( e ) => e ); // (asked for now; a failure waits for the first read())

	}

	async _connect() {

		const ctl = this._ctl = new AbortController(), from = this._whole ? 0 : this.got;
		const headers = from ? { ...( this.init.headers || {} ), Range: `bytes=${from}-` } : this.init.headers;
		const r = await guard( fetch( this.url, { ...this.init, ...( headers ? { headers } : {} ), signal: ctl.signal } ), this.tries ? REWAIT : WAIT, ctl, true );
		if ( again( r.status ) ) {

			try { ctl.abort(); } catch ( x ) { /* */ }

			throw new Error( `${r.status}` );

		}

		if ( from && r.status === 206 ) {

			// (the rest as the file has it, or nothing: a part of a compressed copy cannot be joined to what is here)
			if ( ! ( r.headers.get( 'Content-Range' ) || '' ).startsWith( `bytes ${from}-` ) || r.headers.get( 'Content-Encoding' ) ) {

				this._whole = true;
				try { ctl.abort(); } catch ( x ) { /* */ }

				throw new Error( 'not the part asked for' );

			}

			netStats.resumed ++;

		} else if ( this.got && r.ok ) { this._restart = true; netStats.restarted ++; } // (the whole file again)
		// (a 416 to the rest of a file: it had all arrived, and the end was what went missing)
		if ( from && r.status === 416 ) { this._rd = { read: async () => ( { done: true } ) }; return; }
		this.info = { ok: r.ok, status: r.status === 206 ? 200 : r.status, statusText: r.statusText, type: r.headers.get( 'Content-Type' ) || '' };
		if ( ! this.answered ) this.answered = now();
		if ( ! r.ok ) { this._over = true; netStats.open --; try { ctl.abort(); } catch ( x ) { /* */ } return; }
		if ( r.body && r.body.getReader ) this._rd = r.body.getReader();
		else {

			// (no streams: the body in one piece, with as long to arrive as a file of ours can take)
			let p = guard( r.arrayBuffer(), STALL * 10, ctl );
			this._rd = { read: async () => { const q = p; p = null; return q ? { value: new Uint8Array( await q ) } : { done: true }; } };

		}

	}

	// what the server said: { ok, status, statusText, type }. Throws when no request got an answer
	async status() {

		while ( ! this.info ) {

			const r = await this._step( true );
			if ( r && r.done ) break;

		}

		return this.info || { ok: false, status: 0, statusText: 'cancelled', type: '' };

	}

	async read() {

		for ( ;; ) {

			const r = await this._step( false );
			if ( r ) return r;

		}

	}

	// one step towards the next piece. -> a result of read(), or null: go on (head: stop once the answer is known)
	async _step( head ) {

		if ( this._over ) return { done: true };
		try {

			if ( this._first ) {

				const e = await this._first;
				this._first = null;
				if ( e ) throw e;

			} else if ( ! this._rd ) await this._connect();

			if ( this._over ) return { done: true };
			if ( this._restart ) { this._restart = false; this.got = 0; return { restart: true }; }
			if ( head ) return { head: true };
			const { done, value } = await guard( this._rd.read(), STALL, this._ctl );
			if ( done ) { this._over = true; netStats.open --; return { done: true }; }
			if ( ! value || ! value.length ) return null;
			this.got += value.length; this.tries = 0; lastByte = now();
			return { value };

		} catch ( e ) {

			if ( this._ctl ) try { this._ctl.abort(); } catch ( x ) { /* */ }

			this._rd = null;
			if ( e && e.stalled ) netStats.stalls ++; else netStats.errors ++;
			const why = e && e.stalled ? ( this.info && this.got ? 'stopped arriving' : 'got no answer' ) : `failed (${e && e.message || e})`;
			if ( ! ( e && e.queued ) ) this.tries ++;
			if ( this.tries >= this.max || this._cancelled ) {

				this._over = true; netStats.open --; netStats.gaveUp ++;
				netStats.last = `${short( this.url )} ${why} at ${this.got} bytes, given up`;
				const x = new Error( `${short( this.url )}: ${why}, ${this.tries} times` );
				x.net = true;
				throw x;

			}

			netStats.retries ++;
			netStats.last = `${short( this.url )} ${why} at ${this.got} bytes`;
			if ( this.tries < 12 ) console.warn( `[net] ${netStats.last}: asking again (${this.tries + 1}${this.max < FOREVER ? '/' + this.max : ''})` );
			await sleep( DELAY[ Math.max( 0, Math.min( this.tries, DELAY.length ) - 1 ) ] );
			return null;

		}

	}

	cancel() {

		this._cancelled = true;
		if ( ! this._over ) { this._over = true; netStats.open --; }
		if ( this._ctl ) try { this._ctl.abort(); } catch ( x ) { /* */ }

	}

}

export const openBytes = ( url, init, tries ) => new Bytes( url, init, tries );

// fetch(), with the body read here: -> a Response (ok or not, as the server said). Throws (`net: true`) when the
// network gave no whole answer in TRIES requests
export async function fetchSafe( url, init ) {

	const b = new Bytes( url, init );
	let parts = [];
	for ( ;; ) {

		const r = await b.read();
		if ( r.done ) break;
		if ( r.restart ) parts = [];
		else parts.push( r.value );

	}

	const info = b.info || { ok: false, status: 502, statusText: '', type: '' };
	return new Response( info.ok ? new Blob( parts, { type: info.type } ) : null, { status: info.status, statusText: info.statusText, headers: info.type ? { 'Content-Type': info.type } : {} } );

}
