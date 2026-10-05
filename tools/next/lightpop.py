#!/usr/bin/env python3
"""Does a light change with the camera's distance? The next branch's page (interactive, 1920 x 1080), the camera moved
step by step towards / past / around one light, and at every step the level of the light's cell and the brightness of
(a) its head, (b) a fixed patch of its pool on the ground, (c) its reflection on the wet paving, read from the page's
own frame. A step change between two neighbouring steps is a pop.

    python3 tools/next/lightpop.py --quality high --case street --name before_high_street [--routes dolly,strafe,spin] [--q "u_pane=0"]

-> .scratch/next/lp/<name>.json (rows) and .scratch/next/lp/<name>/<route>_<step>_{head,ground}.png (tiles of a fixed
size in the world: 3 m around the head, the paving from the foot of the light to its mirror image), a table on stdout.
Every step is taken twice where the light's cell changes level: once with the change held half way ("mid"), once settled.
Chrome: the capture Chrome on port 9229 (see shoot.py).
"""
import sys, os, json, math, subprocess, argparse, urllib.parse, base64

ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

# town coordinates (x east, y north, z up). head: the burner; foot: the ground under it; start: where the dolly starts;
# eye: 'ground' (1.65 m over the walkable ground) or a height; clear: how far from the light the line of sight is free of
# props (tools: the town is dense - no light has more than 26 m of walkable ground with nothing within 0.5 m of the
# line; beyond `clear` a step may have a planter, a bench or a bollard in its measuring windows and is marked `x`)
CASES = {
	# a lamp post of the main street, seen from the square through the gate
	'street': dict( head = [ 2.48, - 32.194, 3.897 ], foot = 0.0, start = [ - 7.68, 14.0 ], eye = 'ground', clear = 30.5,
		strafe = dict( a = [ - 3.5, - 38.0 ], b = [ - 3.5, - 23.0 ], look = [ 1, 0 ] ), spin = [ - 1.0, - 24.5 ] ),
	# the wall lantern at the end of the alley off the terrace (a level flight at the terrace's eye height)
	'alley': dict( head = [ 31.58, - 7.0, 3.6 ], foot = 2.1, start = [ - 16.4, - 7.0 ], eye = 3.75, clear = 14.8,
		strafe = dict( a = [ 24.0, - 14.0 ], b = [ 24.0, 0.0 ], look = [ 1, 0 ] ), spin = [ 22.0, - 7.0 ] ),
	# a bulb of the cafe's festoon on the square, along the one free line across the square
	'festoon': dict( head = [ - 11.21, - 6.83, 2.44 ], foot = 0.0, start = [ 14.0, - 13.1 ], eye = 'ground', clear = 26.0,
		strafe = dict( a = [ - 5.0, - 15.0 ], b = [ - 5.0, 0.0 ], look = [ - 1, 0 ] ), spin = [ - 5.0, - 4.0 ] ),
	# the tailor's shop window on the terrace
	'window': dict( head = [ 24.0, 13.55, 3.3 ], foot = 2.1, start = [ - 16.0, 13.55 ], eye = 3.75, clear = 8.0,
		strafe = dict( a = [ 18.0, 6.0 ], b = [ 18.0, 20.5 ], look = [ 1, 0 ] ), spin = [ 17.0, 12.0 ] ),
}

PAGE = r"""
window.__lp = ( () => {

	const town = window.town, eng = window.eng, P = window.__player;
	const V3 = eng.camera.position.constructor;
	const raf = () => new Promise( ( r ) => requestAnimationFrame( r ) );
	const frames = async ( n ) => { for ( let i = 0; i < n; i ++ ) await raf(); };
	const cv = document.createElement( 'canvas' ), cx = cv.getContext( '2d' );
	const tile = document.createElement( 'canvas' ), tx = tile.getContext( '2d' );
	const cellAt = ( x, z ) => town.chunks.find( ( c ) => x >= c.box[ 0 ] && x <= c.box[ 2 ] && z >= c.box[ 1 ] && z <= c.box[ 3 ] && ! c.m.outer ) || town.chunks.find( ( c ) => x >= c.box[ 0 ] && x <= c.box[ 2 ] && z >= c.box[ 1 ] && z <= c.box[ 3 ] );
	const luma = ( D, i ) => 0.2126 * D[ i ] + 0.7152 * D[ i + 1 ] + 0.0722 * D[ i + 2 ];
	// mean and peak of the display luminance (0..255) in a box of the frame; null when its centre is out of the frame
	const box = ( img, c, hw, hh ) => {

		if ( ! c ) return null;
		const W = img.width, H = img.height, D = img.data;
		if ( c[ 0 ] < 0 || c[ 0 ] >= W || c[ 1 ] < 0 || c[ 1 ] >= H ) return null;
		const x0 = Math.max( 0, Math.round( c[ 0 ] - hw ) ), x1 = Math.min( W - 1, Math.round( c[ 0 ] + hw ) ), y0 = Math.max( 0, Math.round( c[ 1 ] - hh ) ), y1 = Math.min( H - 1, Math.round( c[ 1 ] + hh ) );
		let s = 0, m = 0, n = 0;
		for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) { const l = luma( D, ( y * W + x ) * 4 ); s += l; if ( l > m ) m = l; n ++; }
		return n ? [ + ( s / n ).toFixed( 2 ), Math.round( m ) ] : null;

	};
	// mean colour (display, 0..255) of a box: the pane of a lantern, white-hot or amber
	const tint = ( img, c, hw ) => {

		if ( ! c ) return null;
		const W = img.width, H = img.height, D = img.data;
		if ( c[ 0 ] < 0 || c[ 0 ] >= W || c[ 1 ] < 0 || c[ 1 ] >= H ) return null;
		const x0 = Math.max( 0, Math.round( c[ 0 ] - hw ) ), x1 = Math.min( W - 1, Math.round( c[ 0 ] + hw ) ), y0 = Math.max( 0, Math.round( c[ 1 ] - hw ) ), y1 = Math.min( H - 1, Math.round( c[ 1 ] + hw ) );
		const s = [ 0, 0, 0 ]; let n = 0;
		for ( let y = y0; y <= y1; y ++ ) for ( let x = x0; x <= x1; x ++ ) { const i = ( y * W + x ) * 4; s[ 0 ] += D[ i ]; s[ 1 ] += D[ i + 1 ]; s[ 2 ] += D[ i + 2 ]; n ++; }
		return s.map( ( v ) => Math.round( v / n ) );

	};
	const crop = ( c, hw, hh, w, h ) => {

		if ( ! c ) return null;
		tile.width = w; tile.height = h;
		tx.fillStyle = '#000'; tx.fillRect( 0, 0, w, h );
		tx.imageSmoothingQuality = 'high';
		tx.drawImage( cv, c[ 0 ] - hw, c[ 1 ] - hh, hw * 2, hh * 2, 0, 0, w, h );
		return tile.toDataURL( 'image/png' ).split( ',' )[ 1 ];

	};
	// head, pool: three coordinates; gy: height of the ground the light is mirrored in
	// lag: with the town's probe (what wet paving mirrors beyond the reach of its rays) as a walker has it at worst -
	// taken that many metres back along the line of sight - instead of the fresh one a capture takes
	const measure = async ( head, pool, gy, tiles, lag ) => {

		let img;
		if ( lag && eng.probe ) {

			const p = eng.camera.position, keep = p.clone(), f = new V3();
			eng.camera.getWorldDirection( f ); f.y = 0; f.normalize();
			p.addScaledVector( f, - lag ); eng.probe.now(); p.copy( keep );
			const now = eng.probe.now; eng.probe.now = () => {};
			try { img = await eng.capture(); } finally { eng.probe.now = now; }

		} else img = await eng.capture();
		const cam = eng.camera, W = img.width, H = img.height;
		cam.updateMatrixWorld();
		const px = ( p ) => {

			const v = new V3( p[ 0 ], p[ 1 ], p[ 2 ] );
			if ( v.clone().applyMatrix4( cam.matrixWorldInverse ).z > - 0.2 ) return null;
			v.project( cam );
			return [ ( v.x * 0.5 + 0.5 ) * W, ( 0.5 - v.y * 0.5 ) * H ];

		};
		const perM = ( p ) => H / ( 2 * Math.tan( cam.fov * Math.PI / 360 ) * Math.max( 0.5, new V3( p[ 0 ], p[ 1 ], p[ 2 ] ).applyMatrix4( cam.matrixWorldInverse ).z * - 1 ) );
		const c = cam.position, e = c.y - gy, h = head[ 1 ] - gy, f = e / Math.max( e + h, 0.1 );
		const mir = [ c.x + ( head[ 0 ] - c.x ) * f, gy, c.z + ( head[ 2 ] - c.z ) * f ]; // where the paving mirrors the head
		const foot = [ head[ 0 ], gy, head[ 2 ] ];
		const ph = px( head ), pp = px( pool ), pm = px( mir ), pf = px( foot );
		const sh = perM( head ), sp = perM( pool ), sm = perM( mir );
		const dip = Math.max( 0.12, e / Math.max( Math.hypot( pool[ 0 ] - c.x, pool[ 2 ] - c.z ), 0.5 ) ); // a level patch seen at a slant
		const out = { size: [ W, H ], d: + Math.hypot( head[ 0 ] - c.x, head[ 2 ] - c.z ).toFixed( 2 ),
			head: box( img, ph, Math.max( 3, 0.6 * sh ), Math.max( 3, 0.6 * sh ) ), core: box( img, ph, Math.max( 2, 0.18 * sh ), Math.max( 2, 0.18 * sh ) ),
			pool: box( img, pp, Math.max( 2, 0.5 * sp ), Math.max( 2, 0.5 * sp * Math.min( 1, dip ) ) ),
			refl: box( img, pm, Math.max( 2, 0.35 * sm ), Math.max( 3, 0.9 * sm * Math.min( 1, Math.max( 0.25, e / Math.max( Math.hypot( mir[ 0 ] - c.x, mir[ 2 ] - c.z ), 0.5 ) ) ) ) ),
			pane: tint( img, ph, Math.max( 1, 0.09 * sh ) ),
			frame: box( img, [ W / 2, H / 2 ], W / 2, H / 2 ) };
		if ( tiles ) {

			cv.width = W; cv.height = H; cx.putImageData( img, 0, 0 );
			out.tiles = { head: ph && ph[ 0 ] > 0 && ph[ 0 ] < W && ph[ 1 ] > 0 && ph[ 1 ] < H ? crop( ph, 1.5 * sh, 1.5 * sh, 132, 132 ) : null, ground: null };
			if ( pf && pm ) {

				const y0 = Math.min( pf[ 1 ], pm[ 1 ] ) - 0.6 * sh, y1 = Math.max( pf[ 1 ], pm[ 1 ] ) + 0.25 * ( Math.abs( pm[ 1 ] - pf[ 1 ] ) + 20 );
				const hh = Math.max( 12, ( y1 - y0 ) / 2 ), mid = [ ( pf[ 0 ] + pm[ 0 ] ) / 2, ( y0 + y1 ) / 2 ];
				if ( mid[ 1 ] - hh < H && mid[ 0 ] > 0 && mid[ 0 ] < W ) out.tiles.ground = crop( mid, hh * 0.75, hh, 132, 176 );

			}

		}

		return out;

	};
	const state = ( cell ) => {

		const s = town.visibleStats();
		const c = cell;
		return { level: c ? c.level : null, want: c ? c.want : null, ask: c ? c.ask ?? null : null, cd: c ? + c.d.toFixed( 1 ) : null, fading: c && c.fade ? [ c.fade.from, c.fade.to ] : null,
			shown: c && c.mesh[ c.level ] ? Math.round( c.mesh[ c.level ].userData.shown ?? c.mesh[ c.level ].userData.tris ) : null,
			fades: town.fades.size, pending: s.pending, lods: s.lods, tris: eng.renderer.info.render.triangles };

	};
	const run = async ( job ) => {

		const R = window.__lpres = { rows: [], done: false, error: null, cell: null };
		try {

			const head = job.head, pool = job.pool, gy = job.gy;
			const cell = cellAt( head[ 0 ], head[ 2 ] );
			R.cell = cell ? cell.m.k : null;
			const fadeTime = town.fadeTime;
			for ( const st of job.steps ) {

				// (town coordinates in the job: x east, y north)
				let z = st.eye;
				if ( z === null ) { const g = town.nav.groundAt( st.x, st.y, 50 ); z = ( g ?? P.z ) + 1.65; }
				P.x = st.x; P.y = st.y; P.z = z - P.eye; P.yaw = st.yaw; P.pitch = st.pitch; P.bob = 0;
				let mid = null, t0 = performance.now(), calm = 0;
				await frames( 3 );
				while ( performance.now() - t0 < 20000 ) {

					await raf();
					if ( cell && cell.fade && ! mid && fadeTime > 0 ) {

						// the change of level of the light's own cell, held half way
						town.fadeTime = 1e4;
						for ( const c of town.fades ) c.fade.t0 = performance.now() / 1000 - 0.5e4;
						await frames( 4 );
						mid = Object.assign( await measure( head, pool, gy, job.tiles, job.lag ), state( cell ) );
						town.fadeTime = fadeTime;
						for ( const c of town.fades ) c.fade.t0 = performance.now() / 1000 - fadeTime * 0.5;
						continue;

					}

					const quiet = town.pending === 0 && town.fades.size === 0 && Math.abs( eng.camera.position.y - z ) < 0.03;
					calm = quiet ? calm + 1 : 0;
					if ( calm > 24 && performance.now() - t0 > 700 ) break;

				}

				const row = Object.assign( { route: st.route, i: st.i, at: [ st.x, st.y, + z.toFixed( 2 ) ], yaw: + ( st.yaw * 180 / Math.PI ).toFixed( 0 ), ms: Math.round( performance.now() - t0 ) }, await measure( head, pool, gy, job.tiles, job.lag ), state( cell ) );
				if ( mid ) row.mid = mid;
				R.rows.push( row );

			}

		} catch ( e ) {

			R.error = String( e && e.stack || e );

		}

		R.done = true;

	};
	// A walk as a walker has it: `step` metres every `frames` frames along a line, looking ahead; nothing waits to
	// settle, the probe is the one the page has at that moment. Every frame: for each lamp in view its head, a patch of
	// its pool fixed in the world, and the place the paving mirrors it; the level of the cell under the pool; whether
	// the probe was renewed; a strip of the lower half of the frame.
	const walk = async ( job ) => {

		const R = window.__lpres = { rows: [], done: false, error: null, cell: null, lamps: [] };
		try {

			const [ x0, y0, x1, y1 ] = job.walk, L = Math.hypot( x1 - x0, y1 - y0 ), ux = ( x1 - x0 ) / L, uy = ( y1 - y0 ) / L;
			const yaw = Math.atan2( - ux, uy ), pitch = job.pitch;
			const lamps = [];
			// (nav.groundAt wants a height to come from: the walkable levels of the town, lowest first)
			const gAt = ( x, y ) => { for ( const z of [ 0.3, 2.4, 4.5 ] ) { const g = town.nav.groundAt( x, y, z ); if ( g !== null && g !== undefined ) return g; } return null; };
			job.lamps.forEach( ( h, id ) => {

				const lx = h[ 0 ], ly = - h[ 2 ];
				const t = Math.max( 0, Math.min( L, ( lx - x0 ) * ux + ( ly - y0 ) * uy ) );
				const cx = x0 + ux * t, cy = y0 + uy * t, off = Math.hypot( lx - cx, ly - cy );
				if ( off > job.side || h[ 1 ] > 8 ) return;
				// the pool: on the ground, 1.3 m from the foot towards the line walked
				const k = off > 0.01 ? Math.min( 1.3, off ) / off : 0;
				const px_ = lx + ( cx - lx ) * k, py_ = ly + ( cy - ly ) * k;
				let g = gAt( px_, py_ );
				if ( g === null || g > h[ 1 ] - 1 ) g = gAt( cx, cy );
				if ( g === null ) return;
				const c = cellAt( px_, - py_ );
				lamps.push( { id, head: [ h[ 0 ], h[ 1 ], h[ 2 ] ], gy: g, pool: [ px_, g + 0.02, - py_ ], cell: c, t } );
				R.lamps.push( { id, x: lx, y: ly, z: h[ 1 ], gy: g, t: + t.toFixed( 2 ), off: + off.toFixed( 2 ), cell: c ? c.m.k : null } );

			} );
			// patches of the ground fixed in the world: a 3 m lattice 9 m to either side of the line and 30 m beyond its end
			const grid = [];
			for ( let a = 0; a <= L + 30; a += 3 ) for ( let b = - 9; b <= 9; b += 3 ) {

				const x = x0 + ux * a - uy * b, y = y0 + uy * a + ux * b, g = gAt( x, y );
				if ( g !== null ) grid.push( [ x, g + 0.02, - y ] );

			}

			if ( job.pmove && eng.probe ) eng.probe.move = job.pmove;
			const place = ( sAt ) => {

				const x = x0 + ux * sAt, y = y0 + uy * sAt;
				const z = ( gAt( x, y ) ?? 0 ) + 1.65;
				P.x = x; P.y = y; P.z = z - P.eye; P.yaw = yaw; P.pitch = pitch; P.bob = 0;
				return z;

			};
			place( 0 );
			let t0 = performance.now(), calm = 0;
			while ( performance.now() - t0 < 20000 ) {

				await raf();
				calm = town.pending === 0 && town.fades.size === 0 ? calm + 1 : 0;
				if ( calm > 30 && performance.now() - t0 > 1500 ) break;

			}

			const now = eng.probe ? eng.probe.now : null;
			if ( eng.probe ) eng.probe.now = () => {};
			let faces = eng.probe ? eng.probe.faces : 0, into = eng.probe ? eng.probe.into : 0;
			try {

				for ( let i = 0, sAt = 0; sAt <= L; i ++, sAt += job.step ) {

					place( sAt );
					await frames( job.frames );
					const img = await eng.capture();
					const cam = eng.camera, W = img.width, H = img.height;
					cam.updateMatrixWorld();
					const view = ( p ) => new V3( p[ 0 ], p[ 1 ], p[ 2 ] ).applyMatrix4( cam.matrixWorldInverse );
					const px = ( p ) => {

						if ( view( p ).z > - 0.2 ) return null;
						const v = new V3( p[ 0 ], p[ 1 ], p[ 2 ] ).project( cam );
						return [ ( v.x * 0.5 + 0.5 ) * W, ( 0.5 - v.y * 0.5 ) * H ];

					};
					const perM = ( p ) => H / ( 2 * Math.tan( cam.fov * Math.PI / 360 ) * Math.max( 0.5, - view( p ).z ) );
					const c = cam.position;
					const row = { i, s: + sAt.toFixed( 2 ), at: [ + c.x.toFixed( 2 ), + ( - c.z ).toFixed( 2 ), + c.y.toFixed( 2 ) ], fades: town.fades.size, pending: town.pending,
						lods: town.visibleStats().lods, probe: eng.probe ? ( eng.probe.into !== into ? 1 : 0 ) : 0, face: eng.probe ? eng.probe.face : 0,
						frame: box( img, [ W / 2, H / 2 ], W / 2, H / 2 ), L: [],
						fineR: town.U && town.U.fineR ? [ + ( town.fineR ?? - 1 ).toFixed( 1 ), + town.U.fineR.value.toFixed( 1 ) ] : null };
					if ( eng.probe ) into = eng.probe.into;
					for ( const l of lamps ) {

						const d = Math.hypot( l.head[ 0 ] - c.x, l.head[ 2 ] - c.z );
						if ( d > job.far ) continue;
						const e = c.y - l.gy, h = l.head[ 1 ] - l.gy, f = e / Math.max( e + h, 0.1 );
						const mir = [ c.x + ( l.head[ 0 ] - c.x ) * f, l.gy, c.z + ( l.head[ 2 ] - c.z ) * f ];
						const ph = px( l.head ), pp = px( l.pool ), pm = px( mir );
						if ( ! ph && ! pp && ! pm ) continue;
						const sh = perM( l.head ), sp = perM( l.pool ), sm = perM( mir );
						const dip = ( q ) => Math.min( 1, Math.max( 0.12, e / Math.max( Math.hypot( q[ 0 ] - c.x, q[ 2 ] - c.z ), 0.5 ) ) );
						const m = { id: l.id, d: + d.toFixed( 2 ), lvl: l.cell ? l.cell.level : null, fade: l.cell && l.cell.fade ? 1 : 0,
							head: box( img, ph, Math.max( 3, 0.6 * sh ), Math.max( 3, 0.6 * sh ) ),
							pool: box( img, pp, Math.max( 2, 0.5 * sp ), Math.max( 2, 0.5 * sp * dip( l.pool ) ) ),
							refl: box( img, pm, Math.max( 2, 0.35 * sm ), Math.max( 3, 0.9 * sm * Math.max( 0.25, dip( mir ) ) ) ),
							hin: ph ? + Math.min( ph[ 0 ] / W, 1 - ph[ 0 ] / W, ph[ 1 ] / H, 1 - ph[ 1 ] / H ).toFixed( 3 ) : null };
						if ( m.head || m.pool || m.refl ) row.L.push( m );

					}

					// (the lattice: the patches 4 - 32 m from the camera that are in the frame, 2 m across)
					row.G = [];
					grid.forEach( ( p, k ) => {

						const v = view( p );
						if ( v.z > - 4 || v.z < - 32 ) return;
						const q = px( p );
						const sc = perM( p ), hw = Math.max( 2, sc ), hh = Math.max( 1, sc * Math.min( 1, ( c.y - p[ 1 ] ) / - v.z ) );
						if ( ! q || q[ 0 ] < hw + 2 || q[ 0 ] > W - hw - 2 || q[ 1 ] < hh + 2 || q[ 1 ] > H - hh - 2 ) return;
						const b = box( img, q, hw, hh );
						if ( b ) row.G.push( [ k, b[ 0 ] ] );

					} );
					if ( job.tiles ) {

						cv.width = W; cv.height = H; cx.putImageData( img, 0, 0 );
						tile.width = 640; tile.height = 200;
						tx.imageSmoothingQuality = 'high';
						tx.drawImage( cv, 0, H * 0.4, W, H * 0.6, 0, 0, 640, 200 );
						row.strip = tile.toDataURL( 'image/jpeg', 0.9 ).split( ',' )[ 1 ];

					}

					R.rows.push( row );

				}

			} finally { if ( eng.probe ) eng.probe.now = now; }

		} catch ( e ) {

			R.error = String( e && e.stack || e );

		}

		R.done = true;

	};
	return { run: ( job ) => job.walk ? walk( job ) : run( job ) };

} )();
"""

DRIVER = r"""
import json, time
url = %(url)r
tid = None
for t in cdp("Target.getTargets").get("targetInfos", []):
    if t.get("type") != "page":
        continue
    if tid is None:
        tid = t["targetId"]
    elif not str(t.get("url", "")).startswith("about:"):
        try:
            cdp("Target.closeTarget", targetId=t["targetId"])  # (one town page at a time: a second one doubles the GPU memory)
        except Exception:
            pass
if tid is None:
    tid = cdp("Target.createTarget", url="about:blank", background=True)["targetId"]
sid = cdp("Target.attachToTarget", targetId=tid, flatten=True)["sessionId"]
def ev(expr, wait=False):
    try:
        r = cdp("Runtime.evaluate", session_id=sid, expression=expr, returnByValue=True, awaitPromise=wait)
    except Exception as e:
        return {"busy": str(e)[:200]}
    if r.get("exceptionDetails"):
        return {"exception": str(r["exceptionDetails"])[:600]}
    return (r.get("result") or {}).get("value")
def stats():
    v = ev("JSON.stringify(window.__stats || null)")
    return json.loads(v) if isinstance(v, str) else None
cdp("Page.enable", session_id=sid)
try:
    cdp("Target.activateTarget", targetId=tid)
    cdp("Page.bringToFront", session_id=sid)
    cdp("Emulation.setFocusEmulationEnabled", session_id=sid, enabled=True)
except Exception as e:
    print("front:", e)
cdp("Emulation.setDeviceMetricsOverride", session_id=sid, width=1920, height=1080, deviceScaleFactor=1, mobile=False)
cdp("Page.navigate", session_id=sid, url=url.split("/next/town/")[0] + "/gallery/")
gpu = None
for _ in range(40):
    time.sleep(0.5)
    gpu = ev("(async () => (await navigator.gpu?.requestAdapter())?.info.vendor || null)()", True)
    if isinstance(gpu, str) and gpu:
        break
cdp("Page.navigate", session_id=sid, url=url)
t0 = time.time()
st = None
while time.time() - t0 < 60:
    time.sleep(0.5)
    st = stats()
    if not st or not st.get("first") or not st.get("town"):
        continue
    tw = st["town"]
    sharp = tw.get("sharp")
    if tw.get("pending") == 0 and (not sharp or sharp[2] > 0 and (not %(sharp)d or sharp[1] >= %(sharp)d)):
        break
time.sleep(2)
out = {"gpu": gpu, "tier": (st or {}).get("tier"), "baked": (st or {}).get("baked")}
print("inject:", ev(%(page)r + "; 'ok'"))
ev("window.__lp.run(" + %(job)r + "); 'started'")
t1 = time.time()
done = False
while time.time() - t1 < %(timeout)d:
    time.sleep(1.0)
    v = ev("window.__lpres ? [window.__lpres.done, window.__lpres.rows.length] : null")
    if isinstance(v, list) and v[0]:
        done = True
        break
out["done"] = done
out["error"] = ev("window.__lpres && window.__lpres.error")
out["cell"] = ev("window.__lpres && window.__lpres.cell")
out["lamps"] = json.loads(ev("JSON.stringify(window.__lpres && window.__lpres.lamps || [])") or "[]")
n = ev("window.__lpres ? window.__lpres.rows.length : 0") or 0
rows = []
for i in range(n):
    v = ev("JSON.stringify(window.__lpres.rows[%%d])" %% i)
    if isinstance(v, str):
        rows.append(json.loads(v))
out["rows"] = rows
out["errors"] = ev("JSON.stringify((window.__log || []).filter(l => /^(error|onerror|rejection)|invalid|shader/i.test(l)).slice(0, 6))")
cdp("Emulation.clearDeviceMetricsOverride", session_id=sid)
cdp("Page.navigate", session_id=sid, url="about:blank")
open(%(result)r, "w").write(json.dumps(out))
print("LIGHTPOP done", done, len(rows))
"""

def steps_of( case, routes, step ):

	C = CASES[ case ]
	hx, hy, hz = C[ 'head' ]
	eye = None if C[ 'eye' ] == 'ground' else float( C[ 'eye' ] )
	eye_z = ( C[ 'foot' ] + 1.65 ) if eye is None else eye
	out = []
	def aim( x, y, ez ):

		# look at the light, a little below its head so the paving in front of it is in the picture, the head never
		# further than 22 degrees above the middle
		d = max( math.hypot( hx - x, hy - y ), 0.1 )
		up = math.atan2( hz - ez, d )
		pitch = max( up - math.radians( 22 ), min( up - math.radians( 8 ), math.radians( - 4 ) ) )
		return math.atan2( - ( hx - x ), hy - y ), pitch

	if 'dolly' in routes:

		sx, sy = C[ 'start' ]
		D = math.hypot( sx - hx, sy - hy ); ux, uy = ( sx - hx ) / D, ( sy - hy ) / D
		d = D; i = 0
		while d > 3.9:

			x, y = hx + ux * d, hy + uy * d
			yaw, pitch = aim( x, y, eye_z )
			out.append( dict( route = 'dolly', i = i, x = round( x, 3 ), y = round( y, 3 ), eye = eye, yaw = yaw, pitch = pitch ) )
			d -= step; i += 1

		# ... and back out again: a level that is kept longer than it is given shows as a loop
		n = len( out )
		for k in range( n - 2, - 1, - 2 ): out.append( dict( out[ k ], route = 'back', i = out[ k ][ 'i' ] ) )

	if 'strafe' in routes:

		S = C[ 'strafe' ]
		ax, ay = S[ 'a' ]; bx, by = S[ 'b' ]
		n = int( round( math.hypot( bx - ax, by - ay ) / 1.5 ) )
		yaw = math.atan2( - S[ 'look' ][ 0 ], S[ 'look' ][ 1 ] )
		for i in range( n + 1 ):

			x, y = ax + ( bx - ax ) * i / n, ay + ( by - ay ) * i / n
			pitch = math.atan2( hz - eye_z, max( math.hypot( hx - x, hy - y ), 1 ) ) - math.radians( 12 )
			out.append( dict( route = 'strafe', i = i, x = round( x, 3 ), y = round( y, 3 ), eye = eye, yaw = yaw, pitch = pitch ) )

	if 'spin' in routes:

		x, y = C[ 'spin' ]
		y0, p0 = aim( x, y, eye_z )
		for i in range( 24 ): out.append( dict( route = 'spin', i = i, x = x, y = y, eye = eye, yaw = y0 + math.radians( 15 * i ), pitch = p0 ) )

	return out

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument( '--quality', default = 'high' )
	ap.add_argument( '--case', default = 'street' )
	ap.add_argument( '--name', default = '' )
	ap.add_argument( '--routes', default = 'dolly,strafe,spin' )
	ap.add_argument( '--step', type = float, default = 3.0 )
	ap.add_argument( '--q', default = '' )
	ap.add_argument( '--notiles', action = 'store_true' )
	ap.add_argument( '--lag', type = float, default = 0, help = 'metres: measure with the reflection probe taken this far back (what a walker sees just before it is renewed)' )
	ap.add_argument( '--timeout', type = int, default = 520 )
	ap.add_argument( '--spin', default = '', help = 'x,y (town): turn on this spot instead of the one of the case (the arc budget at the edge of the fine level)' )
	ap.add_argument( '--walk', default = '', help = 'x0,y0,x1,y1 (town): a walk along this line as a walker has it (probe, levels and fades as they come) instead of a case' )
	ap.add_argument( '--wstep', type = float, default = 0.25, help = 'walk: metres a frame taken' )
	ap.add_argument( '--wframes', type = int, default = 5, help = 'walk: frames drawn between two frames taken (5 at 0.25 m: 3 m/s at 60 Hz)' )
	ap.add_argument( '--pitch', type = float, default = - 6.0, help = 'walk: degrees' )
	ap.add_argument( '--pmove', type = float, default = 0, help = 'walk: the distance after which the probe is renewed (1e9: never)' )
	a = ap.parse_args()
	if a.walk: return walk( a )
	if a.spin: CASES[ a.case ][ 'spin' ] = [ float( v ) for v in a.spin.split( ',' ) ]
	C = CASES[ a.case ]
	name = a.name or ( a.quality + '_' + a.case )
	lp = os.path.join( ROOT, '.scratch/next/lp' )
	os.makedirs( os.path.join( lp, name ), exist_ok = True )
	hx, hy, hz = C[ 'head' ]
	sx, sy = C[ 'start' ]
	D = math.hypot( sx - hx, sy - hy )
	# the patch of the pool: on the ground 1.3 m from the foot towards where the dolly starts
	pool = [ hx + ( sx - hx ) / D * 1.3, C[ 'foot' ] + 0.02, - ( hy + ( sy - hy ) / D * 1.3 ) ]
	job = dict( head = [ hx, hz, - hy ], pool = pool, gy = C[ 'foot' ], tiles = not a.notiles, lag = a.lag, steps = steps_of( a.case, a.routes.split( ',' ), a.step ) )
	url = 'http://127.0.0.1:8765/next/town/?' + urllib.parse.urlencode( dict( quality = a.quality ) ) + ( '&' + a.q if a.q else '' )
	result = os.path.join( lp, name + '.raw.json' )
	if os.path.exists( result ): os.remove( result )
	env = dict( os.environ, BU_NAME = 'nexttown', BU_CDP_URL = 'http://localhost:9229' )
	src = DRIVER % dict( url = url, sharp = 1024 if a.quality == 'high' and 'sharp=0' not in a.q else 0, page = PAGE, job = json.dumps( job ), timeout = a.timeout, result = result )
	p = subprocess.run( [ 'browser-use' ], input = src, capture_output = True, text = True, env = env, cwd = ROOT )
	if not os.path.exists( result ): sys.exit( 'lightpop: no result\n' + p.stdout[ - 2000: ] + p.stderr[ - 2000: ] )
	out = json.load( open( result ) )
	os.remove( result )
	for r in out[ 'rows' ]:

		for which, row in ( ( '', r ), ( 'mid', r.get( 'mid' ) ) ):

			t = row.pop( 'tiles', None ) if row else None
			if not t: continue
			for k, b in t.items():

				if b: open( os.path.join( lp, name, '%s_%02d%s_%s.png' % ( r[ 'route' ], r[ 'i' ], which, k ) ), 'wb' ).write( base64.b64decode( b ) )

	out.update( case = a.case, quality = a.quality, q = a.q, name = name )
	json.dump( out, open( os.path.join( lp, name + '.json' ), 'w' ) )
	print( 'tier %s baked %s cell %s done %s error %s errors %s' % ( out.get( 'tier' ), out.get( 'baked' ), out.get( 'cell' ), out.get( 'done' ), out.get( 'error' ), out.get( 'errors' ) ) )
	table( out )

def walk( a ):

	name = a.name or ( a.quality + '_walk' )
	lp = os.path.join( ROOT, '.scratch/next/lp' )
	os.makedirs( os.path.join( lp, name ), exist_ok = True )
	man = json.load( open( os.path.join( ROOT, 'out/web/_next/manifest.json' ) ) )
	job = dict( walk = [ float( v ) for v in a.walk.split( ',' ) ], step = a.wstep, frames = a.wframes, pitch = math.radians( a.pitch ), side = 14.0, far = 70.0,
		lamps = [ h[ :3 ] for h in man[ 'halos' ] ], tiles = not a.notiles, pmove = a.pmove )
	url = 'http://127.0.0.1:8765/next/town/?' + urllib.parse.urlencode( dict( quality = a.quality ) ) + ( '&' + a.q if a.q else '' )
	result = os.path.join( lp, name + '.raw.json' )
	if os.path.exists( result ): os.remove( result )
	env = dict( os.environ, BU_NAME = 'nexttown', BU_CDP_URL = 'http://localhost:9229' )
	src = DRIVER % dict( url = url, sharp = 1024 if a.quality == 'high' and 'sharp=0' not in a.q else 0, page = PAGE, job = json.dumps( job ), timeout = a.timeout, result = result )
	p = subprocess.run( [ 'browser-use' ], input = src, capture_output = True, text = True, env = env, cwd = ROOT )
	if not os.path.exists( result ): sys.exit( 'lightpop: no result\n' + p.stdout[ - 2000: ] + p.stderr[ - 2000: ] )
	out = json.load( open( result ) )
	os.remove( result )
	for r in out[ 'rows' ]:

		b = r.pop( 'strip', None )
		if b: open( os.path.join( lp, name, 'walk_%03d.jpg' % r[ 'i' ] ), 'wb' ).write( base64.b64decode( b ) )

	out.update( walk = job[ 'walk' ], quality = a.quality, q = a.q, name = name, step = a.wstep )
	json.dump( out, open( os.path.join( lp, name + '.json' ), 'w' ) )
	print( 'tier %s baked %s done %s error %s errors %s frames %d lamps %d' % ( out.get( 'tier' ), out.get( 'baked' ), out.get( 'done' ), out.get( 'error' ), out.get( 'errors' ), len( out[ 'rows' ] ), len( out.get( 'lamps' ) or [] ) ) )
	walktable( out )

def walktable( out, only = None ):

	# per lamp: the steps between neighbouring frames of its pool and of its mirror image, largest first - a walk at
	# 0.25 m a frame changes a patch by a few percent; a step of a quarter is a pop
	rows = out[ 'rows' ]
	# the ground at large: for every lattice patch three neighbouring frames share, how far the middle one lies off the
	# line through the other two (relative, 12 of 255 at least under it) - nothing for a patch that brightens or dims
	# steadily as one walks, half the step for one that jumps. A frame where many do so at once is a pop of the paving.
	G = [ dict( ( k, v ) for k, v in r.get( 'G', [] ) ) for r in rows ]
	steps = []
	for i in range( 1, len( rows ) - 1 ):

		d = sorted( abs( v - ( G[ i - 1 ][ k ] + G[ i + 1 ][ k ] ) / 2 ) / max( v, 12.0 ) for k, v in G[ i ].items() if k in G[ i - 1 ] and k in G[ i + 1 ] )
		if d: steps.append( ( rows[ i ], sum( d ) / len( d ), d[ int( 0.9 * ( len( d ) - 1 ) ) ], d[ - 1 ], sum( 1 for v in d if v > 0.1 ), len( d ) ) )

	if steps:

		m = sorted( s[ 1 ] for s in steps ); p = sorted( s[ 2 ] for s in steps )
		print( 'ground lattice, jump of a patch against its two neighbouring frames: mean %.2f%%, median frame %.2f%%, worst frame %.2f%%; p90 patch: median frame %.1f%%, worst frame %.1f%%; patches jumping > 10%%: %d in %d frames' % (
			100 * sum( m ) / len( m ), 100 * m[ len( m ) // 2 ], 100 * m[ - 1 ], 100 * p[ len( p ) // 2 ], 100 * p[ - 1 ], sum( s[ 4 ] for s in steps ), len( steps ) ) )
		print( 'per frame (mean jump in 0.1%%, * probe renewed, f a level fading): ' + ' '.join( '%.0f%s%s' % ( 1000 * s[ 1 ], '*' if s[ 0 ][ 'probe' ] else '', 'f' if s[ 0 ][ 'fades' ] else '' ) for s in steps ) )

	if rows and rows[ 0 ].get( 'fineR' ):

		print( 'reach of the fine level\'s own light (m, as drawn; * where the nearest cell without the level is nearer than that): ' + ' '.join( '%.0f%s' % ( r[ 'fineR' ][ 1 ], '*' if r[ 'fineR' ][ 0 ] < r[ 'fineR' ][ 1 ] - 0.5 else '' ) for r in rows ) )

	ser = {}
	for r in rows:

		for m in r[ 'L' ]: ser.setdefault( m[ 'id' ], [] ).append( ( r, m ) )

	ev = []
	for id, S in ser.items():

		for k in ( 'pool', 'refl' ):

			for ( r0, m0 ), ( r1, m1 ) in zip( S, S[ 1: ] ):

				if r1[ 'i' ] != r0[ 'i' ] + 1 or not m0.get( k ) or not m1.get( k ): continue
				a, b = m0[ k ][ 0 ], m1[ k ][ 0 ]
				rel = ( b - a ) / max( a, b, 12.0 )
				ev.append( ( abs( rel ), id, k, r1, m0, m1, a, b ) )

	ev.sort( key = lambda e: - e[ 0 ] )
	print( 'largest steps between neighbouring frames (%.2f m apart)' % out.get( 'step', 0.25 ) )
	print( ' lamp what  frame     s      d   before -> after   rel   lvl fade  probe fades pend  head-in-frame' )
	for rel, id, k, r1, m0, m1, a, b in ev[ :12 ]:

		if only is not None and id not in only: continue
		print( ' %4d %-5s %4d %6.2f %6.2f   %6.1f -> %6.1f  %+4.0f%%   %s->%s %d     %d     %d    %d    %s -> %s' % ( id, k, r1[ 'i' ], r1[ 's' ], m1[ 'd' ], a, b, 100 * ( b - a ) / max( a, b, 12.0 ),
			m0[ 'lvl' ], m1[ 'lvl' ], m1[ 'fade' ], r1[ 'probe' ], r1[ 'fades' ], r1[ 'pending' ], m0.get( 'hin' ), m1.get( 'hin' ) ) )

def table( out ):

	f = lambda v, k = 0: '   -' if not v else '%5.1f' % v[ k ]
	print( 'route  i     d  lvl want  head(mean peak) core(mean peak)  pane(r g b)    pool   refl  frame   tris  ms' )
	prev = None
	clear = CASES.get( out.get( 'case' ), {} ).get( 'clear', 1e9 )
	for r in out[ 'rows' ]:

		rows = ( [ ( 'mid', r[ 'mid' ] ) ] if r.get( 'mid' ) else [] ) + [ ( '', r ) ]
		for tag, m in rows:

			flag = ''
			if prev and prev[ 0 ] == r[ 'route' ]:

				for k in ( 'head', 'core', 'pool', 'refl' ):

					a, b = prev[ 1 ].get( k ), m.get( k )
					if a and b and abs( a[ 0 ] - b[ 0 ] ) > max( 4, 0.18 * max( a[ 0 ], b[ 0 ] ) ): flag += ' %s %+.0f' % ( k, b[ 0 ] - a[ 0 ] )

				a, b = prev[ 1 ].get( 'pane' ), m.get( 'pane' )
				if a and b and max( abs( a[ c ] - b[ c ] ) for c in range( 3 ) ) > 14: flag += ' pane %+d' % ( b[ 2 ] - a[ 2 ] )

			print( '%-6s %2d %5.1f%s %s %s%s  %s %s   %s %s   %-13s %s  %s  %s  %5.2f %5d%s%s' % ( r[ 'route' ], r[ 'i' ], m[ 'd' ], 'x' if r[ 'route' ] in ( 'dolly', 'back' ) and m[ 'd' ] > clear else ' ', m[ 'level' ], m[ 'want' ], ' f' if m.get( 'fading' ) else '  ',
				f( m[ 'head' ] ), f( m[ 'head' ], 1 ), f( m[ 'core' ] ), f( m[ 'core' ], 1 ), ' '.join( '%3d' % v for v in m[ 'pane' ] ) if m.get( 'pane' ) else '  -', f( m[ 'pool' ] ), f( m[ 'refl' ] ), f( m[ 'frame' ] ), m[ 'tris' ] / 1e6, r[ 'ms' ], ' ' + tag if tag else '', '   <<' + flag if flag else '' ) )
			prev = ( r[ 'route' ], m )

if __name__ == '__main__':

	if len( sys.argv ) == 3 and sys.argv[ 1 ] == '--table':

		o = json.load( open( sys.argv[ 2 ] ) )
		walktable( o ) if o.get( 'walk' ) else table( o )

	else: main()
