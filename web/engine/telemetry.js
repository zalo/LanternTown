// Visitors report back: what device they are, how the page loaded, how it runs and what it logged.
//
//   <script type="module" src="../engine/telemetry.js?page=town"></script>     (before the page's own module)
//
// Reports go to <root>/api/telemetry (tools/serve.py -> out/.telemetry/<day>.jsonl; read them with
// `python3 tools/telemetry.py`). One when the page opens (so a tab that dies while loading still shows), one a few
// seconds in, then one a minute while the tab is visible, one when it is left, and one soon after any error.
// Sent: device (user agent, screen, cores, memory, GPU as WebGPU and WebGL name it, compressed-texture support,
// connection), the page's address without tokens, navigation and download timing, frame times measured here
// (requestAnimationFrame), window.__stats when the page keeps one, the text of its #hud / #ktxinfo / #msg, console
// warnings and errors (and the first lines of console.log), uncaught errors, lost GPU contexts.
// Not sent: anything typed, cookies, the address of the visitor (the server does not store it either).
//   ?telemetry=0   nothing is sent          ?telemetry=1   send from localhost too (off there: the tools' own browsers)
// A page can add to it: window.__tele.mark( 'first frame' ), window.__tele.log( 'note', text ), window.__tele.set( { k: v } )

const ME = new URL( import.meta.url );
const Q = new URLSearchParams( location.search );
const LOCAL = /^(localhost|127\.|\[::1\]|0\.0\.0\.0)/.test( location.host );
const ON = Q.get( 'telemetry' ) !== '0' && Q.get( 'capture' ) !== '1' && ( ! LOCAL || Q.get( 'telemetry' ) === '1' );
const ENDPOINT = new URL( '../api/telemetry', ME ).href;
const MAX_REPORTS = 40, MAX_LOG = 400, LINE = 600, LOG_LINES = 60; // per visit / per visit / chars a line / console.log lines kept

const T0 = performance.now();
const short = ( v, n = LINE ) => {

	let s;
	try {

		s = typeof v === 'string' ? v : v instanceof Error ? ( v.stack || v.name + ': ' + v.message ) : JSON.stringify( v );

	} catch ( e ) {

		s = String( v );

	}

	s = String( s );
	return s.length > n ? s.slice( 0, n ) + ' …' : s;

};

const state = { seq: 0, logs: [], kept: 0, plain: 0, marks: {}, extra: {}, errors: 0, sent: 0, gpu: null, dead: false };

function push( level, text ) {

	if ( state.kept >= MAX_LOG ) return;
	state.kept ++;
	state.logs.push( [ Math.round( performance.now() - T0 ), level, short( text ) ] );
	if ( level === 'error' || level === 'lost' ) {

		state.errors ++;
		soon( 'error' );

	}

}

function hookConsole() {

	for ( const level of [ 'log', 'info', 'warn', 'error' ] ) {

		const was = console[ level ];
		console[ level ] = function ( ...a ) {

			try {

				if ( level === 'warn' || level === 'error' || state.plain ++ < LOG_LINES ) push( level, a.map( ( v ) => short( v, 300 ) ).join( ' ' ) );

			} catch ( e ) { /* never in the page's way */ }

			return was.apply( this, a );

		};

	}

	addEventListener( 'error', ( e ) => {

		if ( e.target && e.target !== window && ( e.target.src || e.target.href ) ) push( 'error', 'failed to load ' + ( e.target.src || e.target.href ) );
		else push( 'error', ( e.message || 'error' ) + ( e.filename ? ` @ ${e.filename.split( '/' ).slice( - 2 ).join( '/' )}:${e.lineno}:${e.colno}` : '' ) + ( e.error?.stack ? '\n' + e.error.stack : '' ) );

	}, true );
	addEventListener( 'unhandledrejection', ( e ) => push( 'error', 'unhandled rejection: ' + short( e.reason ) ) );
	addEventListener( 'webglcontextlost', ( e ) => push( 'lost', 'WebGL context lost' + ( e.statusMessage ? ': ' + e.statusMessage : '' ) ), true );
	addEventListener( 'webglcontextrestored', () => push( 'warn', 'WebGL context restored' ), true );

}

// ---- the device

function deviceKind() {

	const ua = navigator.userAgent, touch = navigator.maxTouchPoints > 0;
	if ( /iPhone|iPod/.test( ua ) ) return 'phone';
	if ( /iPad/.test( ua ) || ( /Macintosh/.test( ua ) && navigator.maxTouchPoints > 1 ) ) return 'tablet';
	if ( /Android/.test( ua ) ) return /Mobile/.test( ua ) ? 'phone' : 'tablet';
	if ( touch && Math.min( screen.width, screen.height ) < 600 ) return 'phone';
	return touch && matchMedia( '(pointer: coarse)' ).matches ? 'tablet' : 'desktop';

}

function device() {

	const c = navigator.connection, d = navigator.userAgentData;
	return {
		kind: deviceKind(),
		ua: navigator.userAgent,
		platform: d?.platform || navigator.platform,
		mobile: d ? d.mobile : undefined,
		brands: d?.brands?.filter( ( b ) => ! /Not.A.Brand/i.test( b.brand ) ).map( ( b ) => b.brand + ' ' + b.version ).join( ', ' ),
		screen: [ screen.width, screen.height ],
		dpr: + ( devicePixelRatio || 1 ).toFixed( 2 ),
		view: [ innerWidth, innerHeight ],
		cores: navigator.hardwareConcurrency,
		memory: navigator.deviceMemory,
		touch: navigator.maxTouchPoints,
		coarse: matchMedia( '(pointer: coarse)' ).matches,
		lang: navigator.language,
		net: c ? { type: c.effectiveType, down: c.downlink, rtt: c.rtt, save: c.saveData } : undefined,
		standalone: matchMedia( '(display-mode: standalone)' ).matches || undefined,
		reducedMotion: matchMedia( '(prefers-reduced-motion: reduce)' ).matches || undefined,
		hdr: matchMedia( '(dynamic-range: high)' ).matches, // (the display has range above white; __hdrOut: what the canvas got)
		hdrOut: window.__hdrOut ? window.__hdrOut.on : undefined,
	};

}

// what the GPU calls itself, asked on an adapter / a 1 px context of our own: the page's renderer is not touched
async function gpu() {

	const out = {};
	try {

		if ( navigator.userAgentData?.getHighEntropyValues ) {

			const h = await navigator.userAgentData.getHighEntropyValues( [ 'model', 'platformVersion', 'architecture', 'fullVersionList' ] );
			out.model = h.model || undefined;
			out.os = h.platformVersion;
			out.arch = h.architecture || undefined;

		}

	} catch ( e ) { /* not offered */ }

	try {

		if ( ! navigator.gpu ) out.webgpu = false;
		else {

			const a = await navigator.gpu.requestAdapter();
			const i = a && ( a.info || ( a.requestAdapterInfo ? await a.requestAdapterInfo() : null ) );
			out.webgpu = ! a ? null : {
				vendor: i?.vendor, arch: i?.architecture, device: i?.device, name: i?.description,
				fallback: a.isFallbackAdapter || i?.isFallbackAdapter || undefined,
				tex: a.limits?.maxTextureDimension2D, layers: a.limits?.maxTextureArrayLayers,
				compressed: [ 'bc', 'astc', 'etc2' ].filter( ( f ) => a.features.has( 'texture-compression-' + f ) ),
			};

		}

	} catch ( e ) {

		out.webgpu = 'error: ' + e.message;

	}

	try {

		const c = document.createElement( 'canvas' );
		c.width = c.height = 1;
		const gl = c.getContext( 'webgl2' ) || c.getContext( 'webgl' );
		if ( ! gl ) out.webgl = null;
		else {

			const dbg = gl.getExtension( 'WEBGL_debug_renderer_info' );
			out.webgl = {
				v: typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext ? 2 : 1,
				renderer: gl.getParameter( dbg ? dbg.UNMASKED_RENDERER_WEBGL : gl.RENDERER ),
				vendor: gl.getParameter( dbg ? dbg.UNMASKED_VENDOR_WEBGL : gl.VENDOR ),
				tex: gl.getParameter( gl.MAX_TEXTURE_SIZE ),
				compressed: [ [ 'bc', 'EXT_texture_compression_bptc' ], [ 's3tc', 'WEBGL_compressed_texture_s3tc' ], [ 'astc', 'WEBGL_compressed_texture_astc' ], [ 'etc2', 'WEBGL_compressed_texture_etc' ] ].filter( ( f ) => gl.getExtension( f[ 1 ] ) ).map( ( f ) => f[ 0 ] ),
			};
			gl.getExtension( 'WEBGL_lose_context' )?.loseContext();

		}

	} catch ( e ) {

		out.webgl = 'error: ' + e.message;

	}

	return out;

}

// ---- how it runs: frame times of the tab (ms), measured between animation frames while it is visible

const frames = { ms: new Float32Array( 8192 ), n: 0, last: 0, stalls: 0, worst: 0, total: 0 };

function watchFrames() {

	const step = ( t ) => {

		if ( frames.last && document.visibilityState === 'visible' ) {

			const ms = t - frames.last;
			if ( ms > 250 ) frames.stalls ++; // the tab stood still: counted, kept out of the percentiles
			else if ( frames.n < frames.ms.length ) frames.ms[ frames.n ++ ] = ms;
			if ( ms > frames.worst ) frames.worst = ms;
			frames.total ++;

		}

		frames.last = t;
		requestAnimationFrame( step );

	};

	requestAnimationFrame( step );
	document.addEventListener( 'visibilitychange', () => {

		frames.last = 0;

	} );

}

const long = { n: 0, ms: 0 };
function watchLongTasks() {

	try {

		new PerformanceObserver( ( l ) => {

			for ( const e of l.getEntries() ) {

				long.n ++;
				long.ms += e.duration;

			}

		} ).observe( { type: 'longtask', buffered: true } );

	} catch ( e ) { /* Chromium only */ }

}

function frameStats() {

	const n = frames.n;
	if ( ! n ) return frames.stalls ? { n: 0, stalls: frames.stalls, worst: Math.round( frames.worst ) } : undefined;
	const a = Array.from( frames.ms.subarray( 0, n ) ).sort( ( x, y ) => x - y );
	const at = ( p ) => a[ Math.min( n - 1, Math.floor( p * n ) ) ];
	const sum = a.reduce( ( s, v ) => s + v, 0 );
	const out = {
		n, fps: + ( 1000 * n / sum ).toFixed( 1 ), p50: + at( 0.5 ).toFixed( 1 ), p95: + at( 0.95 ).toFixed( 1 ), p99: + at( 0.99 ).toFixed( 1 ),
		worst: Math.round( frames.worst ), over33: a.filter( ( v ) => v > 33.4 ).length, over50: a.filter( ( v ) => v > 50 ).length, stalls: frames.stalls,
	};
	frames.n = 0; frames.stalls = 0; frames.worst = 0;
	return out;

}

function loadStats() {

	const nav = performance.getEntriesByType( 'navigation' )[ 0 ];
	const res = performance.getEntriesByType( 'resource' );
	const r = ( v ) => Math.round( v );
	let bytes = 0, body = 0, cached = 0;
	for ( const e of res ) {

		bytes += e.transferSize || 0;
		body += e.decodedBodySize || 0;
		if ( e.transferSize === 0 && e.decodedBodySize > 0 ) cached ++;

	}

	const name = ( u ) => u.replace( /^https?:\/\/[^/]+/, '' ).replace( /[?&]token=[^&]*/, '' ).slice( - 90 );
	return {
		nav: nav ? { ttfb: r( nav.responseStart ), dom: r( nav.domContentLoadedEventEnd ), load: r( nav.loadEventEnd ), type: nav.type, proto: nav.nextHopProtocol } : undefined,
		res: {
			n: res.length, mb: + ( bytes / 1e6 ).toFixed( 2 ), body_mb: + ( body / 1e6 ).toFixed( 2 ), cached,
			last: res.length ? r( Math.max( ...res.map( ( e ) => e.responseEnd ) ) ) : 0,
			slowest: res.slice().sort( ( a, b ) => b.duration - a.duration ).slice( 0, 6 ).map( ( e ) => [ name( e.name ), r( e.duration ), e.transferSize || 0, e.responseStatus || undefined ] ),
			failed: res.filter( ( e ) => e.responseStatus >= 400 ).slice( 0, 8 ).map( ( e ) => [ name( e.name ), e.responseStatus ] ),
		},
	};

}

function pageStats() {

	const out = {};
	try {

		if ( window.__stats && typeof window.__stats === 'object' ) out.stats = JSON.parse( short( window.__stats, 6000 ).replace( / …$/, '' ) );

	} catch ( e ) { /* cut mid-way: left out */ }

	for ( const id of [ 'hud', 'ktxinfo', 'msg' ] ) {

		const el = document.getElementById( id );
		const text = el && el.textContent.trim();
		if ( text ) out[ id ] = short( text, 1500 );

	}

	const cv = document.querySelector( 'canvas' );
	if ( cv ) out.canvas = [ cv.width, cv.height ];
	return out;

}

// ---- sending

function where() {

	const u = new URL( location.href );
	for ( const k of [ 'token', 'key', 'auth' ] ) u.searchParams.delete( k );
	return u.pathname.replace( /^\/lp\/\d+/, '' ) + u.search;

}

const page = ME.searchParams.get( 'page' ) || location.pathname.split( '/' ).filter( Boolean ).slice( - 2 ).join( '/' ) || 'page';
const sid = ( crypto.randomUUID ? crypto.randomUUID() : String( Math.random() ).slice( 2 ) + Date.now() ).replace( /-/g, '' ).slice( 0, 16 );
let tid = '';
try {

	tid = localStorage.getItem( 'tsq.tid' ) || '';
	if ( ! tid ) localStorage.setItem( 'tsq.tid', tid = sid.slice( 0, 10 ) ); // this browser, so its visits can be read together

} catch ( e ) { /* private mode */ }

function report( kind ) {

	if ( ! ON || state.dead || state.sent >= MAX_REPORTS ) return;
	state.sent ++;
	const first = state.seq === 0;
	const body = {
		v: 1, sid, tid, seq: state.seq ++, kind, page, url: where(), at: Math.round( performance.now() - T0 ), when: new Date().toISOString(),
		visible: document.visibilityState === 'visible',
		device: first || kind === 'hello' ? Object.assign( device(), state.gpu || {} ) : undefined,
		kindOfDevice: deviceKind(),
		load: kind === 'start' ? undefined : loadStats(),
		frames: kind === 'start' ? undefined : frameStats(),
		long: long.n ? { n: long.n, ms: Math.round( long.ms ) } : undefined,
		heap: performance.memory ? Math.round( performance.memory.usedJSHeapSize / 1e6 ) : undefined,
		marks: Object.keys( state.marks ).length ? state.marks : undefined,
		extra: Object.keys( state.extra ).length ? state.extra : undefined,
		errors: state.errors || undefined,
		logs: state.logs.splice( 0 ),
		...( kind === 'start' ? {} : pageStats() ),
	};
	let text = JSON.stringify( body );
	if ( text.length > 200000 ) text = JSON.stringify( Object.assign( body, { logs: body.logs.slice( - 80 ), cut: true } ) );
	try {

		// (a string body: sent as text/plain, which needs no preflight and survives the tab being closed)
		if ( kind === 'bye' && navigator.sendBeacon && navigator.sendBeacon( ENDPOINT, text ) ) return;
		fetch( ENDPOINT, { method: 'POST', body: text, keepalive: text.length < 60000, credentials: 'same-origin' } ).then( ( r ) => {

			if ( r.status === 404 || r.status === 410 || r.status === 507 ) state.dead = true; // nobody listening / log full

		} ).catch( () => {} );

	} catch ( e ) { /* never in the page's way */ }

}

let pending = 0;
function soon( kind ) {

	if ( pending ) return;
	pending = setTimeout( () => {

		pending = 0;
		report( kind );

	}, 2500 );

}

function start() {

	hookConsole();
	watchFrames();
	watchLongTasks();
	report( 'start' ); // the page was opened: if nothing follows, it died while loading
	gpu().then( ( g ) => {

		state.gpu = g;

	} ).finally( () => setTimeout( () => report( 'hello' ), 4000 ) );
	for ( const s of [ 15, 30 ] ) setTimeout( () => report( 'tick' ), s * 1000 );
	setInterval( () => {

		if ( document.visibilityState === 'visible' ) report( 'tick' );

	}, 60000 );
	let left = 0;
	const bye = () => {

		if ( performance.now() - left < 1000 ) return; // pagehide follows visibilitychange
		left = performance.now();
		report( 'bye' );

	};

	document.addEventListener( 'visibilitychange', () => {

		if ( document.visibilityState === 'hidden' ) bye();

	} );
	addEventListener( 'pagehide', bye );

}

window.__tele = {
	on: ON, sid,
	mark( name ) {

		state.marks[ name ] = Math.round( performance.now() - T0 );

	},
	log( level, text ) {

		push( String( level ), text );

	},
	set( values ) {

		Object.assign( state.extra, values );

	},
	send() {

		report( 'tick' );

	},
};

if ( ON ) start();
