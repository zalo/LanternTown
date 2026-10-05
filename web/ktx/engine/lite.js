// Lean runtime for the baked town (tools/webbake.py -> out/web/_next/): what the light tiers of the walkable demo run.
//
// The bake has already done the expensive part offline: every lantern, the moon, the sky fill and the occlusion are
// folded into a per-vertex colour, the static town is merged into one mesh per 16 m cell (three levels of detail) and
// the materials are pre-shaded layers of one small texture array. So this path needs no lights, no shadow maps, no
// reflection probes, no G-buffer: one unlit draw per visible cell, two instanced sprite draws (lantern glows, flames),
// the sky, and a small post chain. createLiteEngine() and BakedTown mirror the parts of createEngine() / Town that
// the town page uses, so the page code stays the same for both paths.

import * as THREE from 'three';
import {
	pass, attribute, uv, varying, texture, vec2, vec3, vec4, float, int, mix, step, max, min, exp, length, dot, pow,
	smoothstep, saturate, oneMinus, uniform, positionWorld, positionView, renderOutput, screenUV, screenCoordinate,
	interleavedGradientNoise, time, fract, sin, abs, instancedBufferAttribute, cameraPosition, normalize,
	textureLoad, ivec2, dFdx, dFdy, cross, sign, floor, clamp, sqrt,
	mrt, output, Fn, If, Loop, Break, bool, reflect, getViewPosition, getScreenPosition, rtt, cubeTexture, uniformArray, log2,
	positionLocal, modelWorldMatrixInverse, frontFacing
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { fxaa } from 'three/addons/tsl/display/FXAANode.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { createSky, MaterialLib, haloCore } from './engine.js';
import { Nav } from './town.js';
import { glbOn, glbBase, glbLevel } from '../../engine/lite_glb.js'; // (the bake's .glb files: it may keep no packed ones)
import { ktxLayers, ktxSharpen, ktxNR, ktxInfo } from './lite_ktx.js';
import { tcLayers, tcFirst, tcSharpen, tcNR } from './lite_tc.js';

// what each baked tier draws. mid / near: distance (m) from the camera to a cell inside which that level is shown
// (a cell leaves a level 20 % + 2 m further out than it entered it - `hold` m at most - so walking along a border
// swaps nothing);
// top: the level shown inside `near` - "near" (LOD1 geometry within 2 cm) or "ring" (the same in clusters that are
// culled on their own, each with its parts in order of size, so it can be drawn down to the parts that still show
// at its distance: the phone's);
// tris: ceiling on the triangles shown, all cells at their level (so also on what one view can hold, however the
// town grows) - or with budget 'view', on what any one heading and pitch from where the camera stands can have in view,
// spent on the top level around the camera first and on mid after it (the phone: its ring is what one sees up
// close, and what is behind the camera costs memory, not frame time); reach: how far mid should get before the
// ring gives up small parts, and then its farthest cells, for it; resident: ceiling on the triangles held on the GPU (a level
// is held whole, in view or not), kept by giving up the farthest cells of a level;
// pixels: frame buffer budget;
// workers: threads that unpack the files of a level (a cell's level is otherwise unpacked in the frame it arrives in:
// with the CPU at a quarter of its speed a walk on Low then had 6 frames over 25 ms and a worst one of 67 ms - the
// build before this branch 5 and 50 - and has 1 and 33 ms with one worker);
// glints: lanterns per ground cell whose highlight on wet / polished surfaces is evaluated per pixel (0 = none)
// merge: lanterns closer together than this (m) count as one of them
//
// medium / high (the next branch): the same baked town with a fourth level on top - top: 'fine' is the LOD0 geometry,
// shown inside `fine` metres, with LOD1 (near) and LOD2 (mid) behind it - and sharp textures:
// tex: the pages the first frame is drawn with; after it the single-layer files come in: `big` = [ size, which ]
// puts the first layers of the list in a second array at that size ('top': the materials with 2048 maps, 'large':
// those and the ones with tiles of 0.8 m and more), `sharp`: the size of all other layers (default: they stay `tex`);
// budget 'arc': tris is what any heading and pitch from where the camera stands can have in view - the mirror image
// of the wet paving, drawn in the same frame, included - spent level by level, nearest cell first (coverage before
// detail) - but for the cells within `first` metres, which get their top level before anything else (`sure`: after
// the cell stood in, and after every cell within that many metres has its second level); fade: seconds a change
// of level is dithered across. (The views are counted by the spheres around the cells and the frame's own pass is
// culled by their boxes, see _cull(); a cell that fades still draws the level it leaves, on top of tris: a walk
// through the square peaks at 4.65 M with High's 4.7 M and at 2.34 M with Medium at 2.3 M - Medium is 2.5 M since the far levels are modelled, not measured again);
// capture: pixel ratio of review renders (as the streaming tiers: what a high-dpi screen shows)
export const BAKED_TIERS = {
	phone: { tex: 256, mid: 40, near: 14, top: 'ring', tris: 0.975e6, budget: 'view', reach: 20, hold: 3, resident: 1.8e6, pixels: 0.8e6, bloom: false, fxaa: true, aniso: 4, fetch: 2, workers: 1, glints: 2, merge: 5, halo: 0.3 },
	low: { tex: 512, mid: 64, near: 20, top: 'near', tris: 2.7e6, pixels: 1.7e6, bloom: true, fxaa: true, aniso: 8, fetch: 3, workers: 1, glints: 4, merge: 4 },
	medium: { tex: 512, big: [ 1024, 'large' ], mid: 110, near: 44, fine: 16, top: 'fine', tris: 2.5e6, budget: 'arc', sure: 16, detail: 0.004, resident: 5.5e6, pixels: 3.7e6, capture: 1,
		bloom: true, fxaa: true, aniso: 8, fetch: 4, workers: 2, glints: 6, merge: 3, first: 6, fade: 0.45, sheet: 1, mirror: 10, probe: 64, planar: 384, dir: 1, nr: 512 },
	high: { tex: 512, sharp: 1024, big: [ 2048, 'top' ], mid: 180, near: 72, fine: 30, top: 'fine', tris: 4.7e6, budget: 'arc', detail: 0.0034, resident: 10e6, pixels: 8.3e6, capture: 1.5,
		bloom: true, fxaa: true, aniso: 16, fetch: 6, workers: 3, glints: 8, merge: 2, first: 10, fade: 0.45, sheet: 1, mirror: 14, probe: 128, planar: 512, dir: 1, nr: 1024 },
};

const K_OPAQUE = 0, K_GLASS = 1, K_WATER = 2, K_FALL = 3, K_CUT = 4;
// the glint light grid (_lightGrid): lanterns closer than this share an entry (m, a tier's `merge` overrides); floor
// to lantern, squared, when ranking them; a lantern fades in between 1 x and GLINT_FADE x the threshold of the place
const GLINT_MERGE = 2, GLINT_H2 = 2.25, GLINT_FADE = 2;
// camera pitches the view budget is kept for (radians; tools/perf_sim.py reads this line)
const VIEW_PITCH = [ 0, 15, 30, 45, 60, 75, - 25, - 50 ].map( ( d ) => d * Math.PI / 180 );
const CULL_PAD = 0.5; // m: what a mesh may draw outside its box (see _cull)

// Is the bake still the town? Its manifest carries the fingerprint of the town.json it was made from; when the
// assembly beside it (out/<town>/town.json) has been rebuilt since, the bake shows a town that no longer exists.
// -> a line saying so, or null (current, or no assembly to compare with: a deployed copy of the bake alone)
async function bakeAge( base, town, man ) {

	try {

		const r = await fetch( `${base}/out/${town}/town.json`, { cache: 'no-cache' } );
		if ( ! r.ok || ! man.town ) return null;
		const buf = await r.arrayBuffer();
		let same = buf.byteLength === man.town.bytes;
		if ( same && man.town.sha1 && globalThis.crypto?.subtle ) {

			const h = new Uint8Array( await crypto.subtle.digest( 'SHA-1', buf ) );
			same = Array.from( h, ( b ) => b.toString( 16 ).padStart( 2, '0' ) ).join( '' ) === man.town.sha1;

		}

		const cmd = 'python3 tools/webbake.py' + ( town === 'town' ? '' : ' --town ' + town );
		return same ? null : `the baked build is older than out/${town}/town.json (re-bake: ${cmd})`;

	} catch ( e ) {

		return null;

	}

}

const LEAN_DIR = 'lean'; // (LEAN.dir of tools/webbake.py: where the full bake's manifest will most likely point low / phone)
// the bake's manifest, or null when there is no usable bake - none at all, one in another format, or one made from an
// older town.json: the page then falls back to the streaming path, and loadBake.note says why (for the console / HUD).
// `town`: the assembly (tools/webbake.py --town <name>); the first one lives in out/web/_next, others in _next_<name>.
// stale: use a bake that is older than its town all the same (?baked=stale)
export async function loadBake( base, tier, town = 'town', { stale = false } = {} ) {

	loadBake.note = null;
	if ( ! BAKED_TIERS[ tier ] || ! /^\w+$/.test( town ) ) return null;
	const q = new URLSearchParams( location.search );
	// ?bakedir=_nextfar: a trial bake beside the real one (out/web/<name>, a name that starts with _next) - for comparing packings
	const BAKE_DIR = 'out/web/' + ( /^_next\w+$/.test( q.get( 'bakedir' ) || '' ) ? q.get( 'bakedir' ) : town === 'town' ? '_next' : '_next_' + town );
	try {

		let dir = `${base}/${BAKE_DIR}`;
		// low / phone: the bake of their own beside it, when there is one - the same town without the fine level, lit with
		// the lanterns' pure inverse square (LEAN of tools/webbake.py; ?lean=0: the full bake's levels all the same).
		// The full bake's manifest says whether it is there and current (man.lean: its folder); both are asked for at
		// once, so these tiers do not wait for one answer after the other before their first byte of the town
		const lean = BAKED_TIERS[ tier ].top !== 'fine' && q.get( 'lean' ) !== '0';
		const get = ( d ) => fetch( `${d}/manifest.json`, { cache: 'no-cache' } ).then( ( r ) => ( r.ok ? r.json() : null ) );
		const ahead = lean ? get( dir + '/' + LEAN_DIR ).catch( () => null ) : null;
		let man = await get( dir );
		if ( ! man ) return null;
		if ( lean && typeof man.lean === 'string' && /^\w+$/.test( man.lean ) ) {

			const ml = await ( man.lean === LEAN_DIR ? ahead : get( dir + '/' + man.lean ).catch( () => null ) );
			if ( ml ) { dir += '/' + man.lean; man = ml; }

		}

		if ( man.version !== 1 || man.codec !== 'meshopt' || ! man.tex[ BAKED_TIERS[ tier ].tex ] ) {

			loadBake.note = 'the baked build has another format than this page reads (re-bake: python3 tools/webbake.py)';
			return null;

		}

		const old = await bakeAge( base, town, man );
		if ( old ) {

			loadBake.note = old + ( stale ? '' : ': streaming the town instead' );
			if ( ! stale ) return null;

		}

		man.dir = dir;
		man.tier = { ...BAKED_TIERS[ tier ] };
		// tuning from the URL: ?mid=30&near=10&tris=500000&pixels=1e6&glints=0
		for ( const k of [ 'mid', 'near', 'fine', 'tris', 'resident', 'pixels', 'glints', 'merge', 'first', 'sure', 'halo', 'fade', 'aniso', 'detail', 'workers', 'flat', 'sheet', 'mirror', 'probe', 'planar', 'planarRate', 'taps', 'dir', 'nr' ] ) if ( q.has( k ) && isFinite( parseFloat( q.get( k ) ) ) ) man.tier[ k ] = parseFloat( q.get( k ) );
		if ( q.get( 'top' ) === 'near' || q.get( 'top' ) === 'ring' ) man.tier.top = q.get( 'top' );
		// a bake without the fine level or the single layers (the main branch's): medium / high run as far as it goes
		if ( man.tier.top === 'fine' && ! man.chunks.some( ( c ) => c.fine ) ) { man.tier.top = 'near'; man.tier.near = man.tier.fine; }
		if ( q.get( 'sharp' ) === '0' || ! man.texl ) { man.tier.big = null; man.tier.sharp = 0; } // ?sharp=0: the first frame's textures throughout
		// ?lm=<bundle>[,<bundle>...]: the lightmap prototype (tools/next/lightmap.py; always beside the full bake). The cells
		// of a bundle take its meshes at its level; a bundle with pages also has its baked light in them (see _materials)
		const lmNames = ( q.get( 'lm' ) || '' ).split( ',' ).filter( ( n ) => /^\w+$/.test( n ) );
		if ( lmNames.length ) {

			const root = `${base}/${BAKE_DIR}/lm`;
			const lm = { pages: [], range: 8, gamma: 2, size: 1024, bundles: [], cells: 0 };
			for ( const name of lmNames ) {

				const rb = await fetch( `${root}/${name}.json`, { cache: 'no-cache' } );
				if ( ! rb.ok ) { loadBake.note = `no lightmap bundle ${name}`; continue; }
				const b = await rb.json();
				const page0 = lm.pages.length;
				for ( const f of b.pages || [] ) lm.pages.push( `${root}/${f}?v=${b.baked}` );
				if ( b.pages && b.pages.length ) { lm.range = b.range || lm.range; lm.size = b.size || lm.size; }
				for ( const key in b.cells ) {

					const c = man.chunks.find( ( x ) => x.k === key );
					if ( ! c ) continue;
					const meta = { ...b.cells[ key ], url: `${root}/${b.cells[ key ].file}?v=${b.baked}`, lmPage: page0 };
					// (the lowest level comes in one file with the page: a cell of a bundle brings its own bytes)
					if ( b.level === 'lo' ) meta.data = new Uint8Array( await ( await fetch( meta.url ) ).arrayBuffer() );
					for ( const l of b.level === 'near' ? [ 'near', 'ring' ] : [ b.level ] ) if ( c[ l ] ) { c[ l ] = meta; lm.cells ++; }

				}

				lm.bundles.push( name );

			}

			if ( lm.pages.length ) man.lm = lm;
			man.lmBundles = lm.bundles;

		}

		return man;

	} catch ( e ) {

		return null;

	}

}

// frame buffer scale for a baked tier: a pixel budget instead of a fixed ratio (the unlit shader is cheap, so a
// phone gets a sharp picture; a 4K desktop does not get four times the work)
export function bakedPixelRatio( bake, w, h ) {

	return Math.max( 0.5, Math.min( devicePixelRatio || 1, Math.sqrt( bake.tier.pixels / Math.max( 1, w * h ) ) ) );

}

// ---------------------------------------------------------------------------------------- engine

export async function createLiteEngine( { canvas, base = '.', width = 1280, height = 720, pixelRatio = 1, fov = 62, forceWebGL = false, bake } ) {

	const T = bake.tier;
	// reversed float depth: the usual 24 bit buffer with the near plane at 5 cm resolves 1 mm at 30 m and 4 mm at 60 m,
	// and the lit rooms behind the windows are cards a few millimetres apart - they shimmered as one walked, and the
	// screen-space reflections drew that shimmer out into the streaks on the paving (rz=0: off). The planar mirror's
	// own pass was never short of depth (its near plane is the paving) and draws the lowest level: what flickers in
	// it is that level's geometry, not the depth buffer.
	const reversedDepthBuffer = new URLSearchParams( location.search ).get( 'rz' ) !== '0';
	const renderer = new THREE.WebGPURenderer( { canvas, antialias: false, alpha: false, forceWebGL, reversedDepthBuffer } );
	renderer.setPixelRatio( pixelRatio );
	renderer.setSize( width, height, false );
	renderer.toneMapping = THREE.ACESFilmicToneMapping;
	renderer.toneMappingExposure = 1.25;
	await renderer.init();
	const isWebGPU = renderer.backend.isWebGPUBackend === true;
	// (the WebGL backend takes it back where EXT_clip_control is missing)
	const reversed = renderer.reversedDepthBuffer === true;
	// a camera of ours as the renderer would leave it after its first frame: the projection is already the reversed one
	// when something copies it before that frame (the mirror image), and a matrix set by hand is not overwritten later
	// (Renderer._updateCamera() rebuilds the projection of every camera it has not yet marked)
	const ownCamera = ( c ) => { c.coordinateSystem = renderer.coordinateSystem; c._reversedDepth = reversed; c.updateProjectionMatrix(); return c; };

	const scene = new THREE.Scene();
	const camera = ownCamera( new THREE.PerspectiveCamera( fov, width / height, 0.05, 2000 ) );
	const noise = new MaterialLib( { base } ).noise; // the engine's tiling noise (sky, flames, water)
	const L = bake.look;
	const moonDir = new THREE.Vector3( ...L.moon_dir ).normalize();
	const sky = createSky( moonDir, noise );
	scene.add( sky );

	// same night haze as the full engine; the baked materials apply it themselves (before the highlight knee)
	const FOG = { color: uniform( new THREE.Color( 0.012, 0.015, 0.024 ) ), density: uniform( 0.0 ), height: uniform( 18 ) };
	const fogAmount = () => {

		const thin = exp( max( positionWorld.y, 0 ).div( FOG.height ).negate() ).mul( 0.8 ).add( 0.2 );
		const fx = length( positionView ).mul( FOG.density ).mul( thin );
		return oneMinus( exp( fx.mul( fx ).negate() ) );

	};

	const eng0 = {};
	const P = {
		ao: uniform( 0 ), bloom: uniform( 0.3 ), vignette: uniform( 0.35 ), grain: uniform( 0.012 ),
		tint: uniform( new THREE.Color( 1, 1, 1 ) ), lift: uniform( 0.0 ), contrast: uniform( 1.12 ), saturation: uniform( 1.08 ),
		ssr: uniform( 0 ), env: uniform( 0 ), knee: uniform( 0.0 ), kneeGlow: uniform( 3.0 ),
		tame: uniform( 0 ), free: uniform( 0 ), roofFill: uniform( 0 ), // of the full engine only: taken from the URL, not used
	};
	// highlight knee of the full engine: lit surfaces roll off towards a ceiling, glowing ones towards a higher one
	const knee = ( c, glowing ) => {

		const m = mix( P.knee, max( P.kneeGlow, P.knee ), glowing );
		const t = m.mul( 0.35 );
		const Lm = max( max( c.r, c.g ), max( c.b, 1e-4 ) );
		const x = Lm.sub( t ).max( 0 ).div( m.sub( t ).max( 1e-4 ) );
		const Lk = min( Lm, t ).add( m.sub( t ).mul( x.div( x.add( 1 ) ) ) );
		return c.mul( mix( float( 1 ), Lk.div( Lm ), step( 0.001, P.knee ) ) );

	};

	// --- post: scene -> (bloom) -> grade -> tone map -> (FXAA) -> dither. No G-buffer, no AO / reflection passes.
	const pipeline = new THREE.RenderPipeline( renderer );
	const scenePass = pass( scene, camera );
	const colorTex = scenePass.getTextureNode( 'output' );
	let comp = colorTex.rgb;
	// The mirror image in standing water (tiers with `mirror`: wet paving, the canal). A surface that mirrors says so
	// in a second colour target - ( strength, roughness, tilt of its mean normal ), see eng.mirrorOut - and one pass
	// follows the reflected ray of those pixels through the depth buffer: `mirror` steps, short at the foot of the ray
	// (what stands on the paving is mirrored where it touches it) and longer further on, then a bisection of the step
	// that crossed a surface. What the ray finds is taken from the frame itself, drawn out along the ray by the
	// roughness (a lit doorway becomes a streak towards the viewer, as it does on wet stone); a ray that leaves the
	// frame, passes behind something or returns towards the camera finds nothing, and the material's own stand-in
	// (sky above, the lit town low down) is all there is.
	let mirrorPass = null;
	// Where the ray finds nothing the water still mirrors the town: on the tiers with `probe` (the size of a face) the
	// scene around the camera is kept as a small cube map - one face a frame, and only after the camera has moved
	// (`move` metres); two maps, so the one being read is always whole - and the mirror pass reads it along the
	// reflected ray, blurred by the roughness (its mip levels).
	//
	// Tiers with `planar` (the width of the image) keep a mirror image instead: the town at its lowest level, drawn every
	// frame the camera moves from the camera's reflection in the paving it stands on (a flat mirror: the wet street).
	// A cube map taken every few steps is looked up by direction alone, so a lantern's streak sat where the lantern
	// was seen from the place of the last map and the whole sheen of the paving changed with every new one - streaks
	// and pools of light switching as one walked. The mirror image is exact for the paving at the camera's level
	// whatever stands where, moves with every step, and does not depend on the level of any cell.
	let probe = null;
	if ( T.mirror && T.probe && T.planar ) {

		const w = Math.max( 64, Math.round( T.planar ) ), h = Math.max( 36, Math.round( w * 9 / 16 ) );
		const rt = new THREE.RenderTarget( w, h, { type: THREE.HalfFloatType, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter } );
		rt.texture.name = 'output'; // (a material with outputs of its own finds its colour target by this name)
		const cam = ownCamera( new THREE.PerspectiveCamera() );
		const node = texture( rt.texture );
		const eye = new THREE.Vector3(), fwd = new THREE.Vector3(), up = new THREE.Vector3(), tgt = new THREE.Vector3(), rot = new THREE.Matrix4(), plane = new THREE.Plane(), cp = new THREE.Vector4(), qv = new THREE.Vector4();
		const last = new THREE.Matrix4().makeScale( 0, 0, 0 );
		probe = eng0.probe = { planar: true, size: w, width: w, height: h, rt, cam, node, vp: uniform( new THREE.Matrix4() ), g: uniform( 0 ), focal: uniform( h ), on: uniform( 0 ), faces: 0, into: 0, face: 0,
			wide: 1.3, idle: 0, every: 6, rate: T.planarRate || 2, phase: 0, hold: false, level: 0, move: 0, ground: null,
			// Nothing below the paving is in a mirror. `clip` is the mirror's plane for whoever draws into the image (the
			// town hangs under a ClippingGroup that is on only while the image is drawn: clip distances in the vertex
			// stage, a discard where the device has none), and the image keeps the camera's own depth range. The other
			// way - the near plane turned into the mirror's (oblique clipping, `obl=1`) - costs nothing, but under it
			// depth falls off with the height above the paving instead of with the distance: across a facade it runs
			// through a large part of its range, and two surfaces a millimetre or two apart that are cut into different
			// triangles (a window's card and the wall behind it at the lowest level) come out in a different order from
			// one step of the camera to the next - a lit window's streak on the wet paving flickered as one walked.
			clip: new THREE.Plane( new THREE.Vector3( 0, 1, 0 ), 0 ), oblique: new URLSearchParams( location.search ).get( 'obl' ) === '1',
			step( all = false ) {

				if ( probe.hold && ! all && probe.on.value ) return;
				// the image stands while the camera does (but for the flames in it: every few frames)
				camera.updateMatrixWorld();
				const still = camera.matrixWorld.equals( last );
				if ( ! all && still && probe.on.value && ( probe.idle = ( probe.idle + 1 ) % probe.every ) !== 0 ) return;
				// moving: every 2nd frame is enough. The lookup projects through the view-projection the image was drawn
				// with, so an image a frame old is still placed right (what is off is the parallax of one frame's step:
				// under a pixel of this image); `last` stays behind on a skipped frame, so the image of the place the
				// camera comes to rest at is always drawn.
				if ( ! all && ! still && probe.on.value && ( probe.phase = ( probe.phase + 1 ) % probe.rate ) !== 0 ) return;
				last.copy( camera.matrixWorld );
				const E = eye.setFromMatrixPosition( camera.matrixWorld );
				// the mirror: the paving under the camera
				let g = probe.ground ? probe.ground( E ) : null;
				if ( g === null || g === undefined ) g = probe.level;
				g = Math.min( g, E.y - 0.1 );
				probe.level = all || ! probe.on.value ? g : probe.level + ( g - probe.level ) * 0.2;
				g = Math.min( probe.level, E.y - 0.1 );
				probe.g.value = g;
				rot.extractRotation( camera.matrixWorld );
				fwd.set( 0, 0, - 1 ).applyMatrix4( rot );
				up.set( 0, 1, 0 ).applyMatrix4( rot );
				cam.coordinateSystem = camera.coordinateSystem;
				cam.position.set( E.x, 2 * g - E.y, E.z );
				cam.up.set( up.x, - up.y, up.z );
				cam.updateMatrixWorld( true ); // (lookAt reads the place from the world matrix)
				tgt.set( E.x + fwd.x, 2 * g - ( E.y + fwd.y ), E.z + fwd.z );
				cam.lookAt( tgt );
				cam.near = camera.near; cam.far = camera.far;
				cam.updateMatrix();
				cam.updateMatrixWorld( true );
				// a wider view than the camera's (a streak is drawn from what stands above and beside its place in the mirror)
				const pm = cam.projectionMatrix.copy( camera.projectionMatrix ), e = pm.elements;
				e[ 0 ] /= probe.wide; e[ 5 ] /= probe.wide;
				probe.focal.value = e[ 5 ] * h / 2;
				probe.clip.constant = - g;
				// (`obl=1`, for comparison: the near plane becomes the mirror's, as three's ReflectorNode)
				plane.set( up.set( 0, 1, 0 ), - g ).applyMatrix4( cam.matrixWorldInverse );
				cp.set( plane.normal.x, plane.normal.y, plane.normal.z, plane.constant );
				// (qv: the far corner of the view opposite the mirror, over its distance - the far plane is turned to pass through it)
				qv.set( ( Math.sign( cp.x ) + e[ 8 ] ) / e[ 0 ], ( Math.sign( cp.y ) + e[ 9 ] ) / e[ 5 ], - 1, reversed ? 1 / cam.far : ( 1 + e[ 10 ] ) / e[ 14 ] );
				cp.multiplyScalar( 1 / cp.dot( qv ) );
				if ( ! probe.oblique ) { /* (the camera's own depth; `clip` does the cutting) */ } else if ( reversed ) {

					// depth runs from 1 at the mirror to 0 at the far plane: the row is that of w less the mirror's plane
					e[ 2 ] = - cp.x; e[ 6 ] = - cp.y; e[ 10 ] = - 1 - cp.z; e[ 14 ] = - cp.w;

				} else { e[ 2 ] = cp.x; e[ 6 ] = cp.y; e[ 10 ] = renderer.coordinateSystem === THREE.WebGPUCoordinateSystem ? cp.z : cp.z + 1; e[ 14 ] = cp.w; }
				cam.projectionMatrixInverse.copy( pm ).invert();
				probe.vp.value.multiplyMatrices( pm, cam.matrixWorldInverse );
				const was = [ renderer.getRenderTarget(), renderer.getActiveCubeFace(), renderer.getActiveMipmapLevel() ];
				if ( probe.swap ) probe.swap( true );
				renderer.setRenderTarget( rt );
				renderer.render( scene, cam );
				if ( probe.swap ) probe.swap( false );
				renderer.setRenderTarget( ...was );
				probe.faces ++;
				probe.on.value = 1;

			},
			// (a capture: of this very place)
			now() { probe.step( true ); },
		};

	} else if ( T.mirror && T.probe ) {

		const size = Math.max( 16, Math.round( T.probe ) );
		const mk = () => new THREE.CubeRenderTarget( size, { type: THREE.HalfFloatType, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter } );
		const rts = [ mk(), mk() ];
		for ( const rt of rts ) rt.texture.name = 'output'; // (a material with outputs of its own finds its colour target by this name)
		const cam = new THREE.CubeCamera( 0.25, 900, rts[ 0 ] );
		cam.coordinateSystem = renderer.coordinateSystem;
		cam.updateCoordinateSystem();
		const node = cubeTexture( rts[ 1 ].texture );
		probe = eng0.probe = { size, levels: Math.log2( size ), rts, cam, node, face: 0, into: 0, at: new THREE.Vector3( 1e9, 0, 0 ), next: new THREE.Vector3(), move: 1.5, every: 1, wait: 0, on: uniform( 0 ), faces: 0,
			step( all = false ) {

				if ( probe.face === 0 ) {

					if ( camera.position.distanceToSquared( probe.at ) < probe.move * probe.move ) return;
					probe.next.copy( camera.position );
					cam.position.copy( camera.position );
					cam.updateMatrixWorld( true );

				}

				if ( ! all && ( probe.wait = ( probe.wait + 1 ) % probe.every ) !== 0 ) return;
				const was = [ renderer.getRenderTarget(), renderer.getActiveCubeFace(), renderer.getActiveMipmapLevel() ];
				const rt = rts[ probe.into ];
				if ( probe.swap ) probe.swap( true );
				do {

					renderer.setRenderTarget( rt, probe.face, 0 );
					renderer.render( scene, cam.children[ probe.face ] );
					probe.face ++; probe.faces ++;

				} while ( all && probe.face < 6 );
				if ( probe.swap ) probe.swap( false );
				renderer.setRenderTarget( ...was );
				if ( probe.face === 6 ) {

					probe.face = 0;
					probe.into ^= 1;
					node.value = rt.texture;
					probe.at.copy( probe.next );
					probe.on.value = 1;

				}

			},
			// (a capture: whole, and of this very place)
			now() { probe.face = 0; probe.at.set( 1e9, 0, 0 ); probe.step( true ); },
		};

	}

	if ( T.mirror ) {

		const sceneMRT = mrt( { output, wet: vec4( 0 ) } );
		sceneMRT.setBlendMode( 'wet', new THREE.BlendMode( THREE.MaterialBlending ) ); // blended materials (glass, glows) leave it as it is
		scenePass.setMRT( sceneMRT );
		scenePass.getTexture( 'wet' ).type = THREE.UnsignedByteType;
		const wetTex = scenePass.getTextureNode( 'wet' ), depthTex = scenePass.getTextureNode( 'depth' );
		const proj = uniform( camera.projectionMatrix ), projInv = uniform( camera.projectionMatrixInverse ), view = uniform( camera.matrixWorldInverse ), world = uniform( camera.matrixWorld );
		const MU = eng0.mirror = { reach: uniform( probe && probe.planar ? 10 : 30 ), thick: uniform( 0.2 ), smear: uniform( 1.5 ), max: uniform( 6 ), env: uniform( 1.2 ), blur: uniform( 12 ), pblur: uniform( 3 ), plow: uniform( 0.3 ),
			// (the mirror image) lobe: length of a streak over the square of the roughness (tangent of the angle above and below
			// its place); gap: blur of a tap over the distance to the next; tg / tk: how far away what is mirrored is taken to
			// be (m) - of no account on the paving the camera stands on, further for paving above or below it
			lobe: uniform( 2.2 ), gap: uniform( 0.6 ), tg: uniform( 8 ), tk: uniform( 30 ), penv: uniform( 1 ), lmin: uniform( 1 ), ssr: uniform( 1 ),
			// hit: with the mirror image behind it a ray's own find only counts close to the pixel (m, fading from half of
			// it): the foot of a bollard, a bench, a kerb. Further out the frame's picture and the mirror image differ in
			// what level they show a house at and in whether it is in the frame at all - a streak must not depend on either.
			hit: uniform( probe && probe.planar ? 8 : 1e6 ) };
		const STEPS = Math.max( 8, Math.round( T.mirror ) );
		const mirrorFn = Fn( () => {

			const uvN = uv();
			const wv = wetTex.sample( uvN ).toVar();
			const out = vec4( 0 ).toVar();
			If( wv.r.greaterThan( 0.006 ), () => {

				// (as getViewPosition(), which takes the depth of a WebGL frame for -1..1: with reversed depth it is 0..1 there too)
				const viewAt = isWebGPU || reversed ? ( u ) => {

					const p = projInv.mul( vec4( u.x.mul( 2 ).sub( 1 ), u.y.oneMinus().mul( 2 ).sub( 1 ), depthTex.sample( u ).r, 1 ) );
					return p.xyz.div( p.w );

				} : ( u ) => getViewPosition( u, depthTex.sample( u ).r, projInv );
				const P0 = viewAt( uvN ).toVar();
				const t = wv.ba.sub( 0.5 );
				const Nv = normalize( view.mul( vec4( t.x, sqrt( max( oneMinus( dot( t, t ) ), 0.01 ) ), t.y, 0 ) ).xyz ).toVar();
				const R = reflect( normalize( P0 ), Nv ).toVar();
				// the ray ends `reach` metres on, or short of the near plane when it comes towards the camera
				const len = MU.reach.toVar();
				If( R.z.greaterThan( 0 ), () => {

					len.assign( min( len, P0.z.negate().sub( 0.1 ).div( R.z ).max( 0 ) ) );

				} );
				const P1 = P0.add( R.mul( len ) ).toVar();
				const span = getScreenPosition( P1, proj ).sub( uvN ).toVar();
				const sp = span.add( step( 0, span ).mul( 2e-6 ).sub( 1e-6 ) );
				const exit = max( oneMinus( uvN ).div( sp ), uvN.negate().div( sp ) ); // where the ray leaves the frame
				const sMax = min( float( 1 ), min( exit.x, exit.y ) ).toVar();
				const rz0 = float( 1 ).div( P0.z ).toVar(), rz1 = float( 1 ).div( P1.z ).toVar();
				const rayZ = ( s ) => float( 1 ).div( rz0.add( s.mul( rz1.sub( rz0 ) ) ) );
				const jit = interleavedGradientNoise( screenCoordinate.xy );
				const found = bool( false ).toVar();
				const hit = vec4( 0 ).toVar(); // what the ray found, and how surely
				const sLo = float( 0 ).toVar(), sHi = float( 0 ).toVar(), zLo = P0.z.toVar();
				Loop( { start: int( 0 ), end: int( STEPS ) }, ( { i } ) => {

					const s = sMax.mul( pow( float( i ).add( jit ).add( 0.25 ).div( STEPS + 0.25 ), 1.7 ) ).toVar();
					const zr = rayZ( s ).toVar();
					const behind = viewAt( uvN.add( span.mul( s ) ) ).z.sub( zr ).toVar(); // > 0: the ray is behind what is drawn there
					If( behind.greaterThan( 0 ).and( behind.lessThan( zLo.sub( zr ).abs().add( MU.thick ) ) ), () => {

						found.assign( true );
						sHi.assign( s );
						Break();

					} );
					sLo.assign( s );
					zLo.assign( zr );

				} );
				If( found, () => {

					Loop( { start: int( 0 ), end: int( 5 ) }, () => {

						const sm = sLo.add( sHi ).mul( 0.5 ).toVar();
						If( viewAt( uvN.add( span.mul( sm ) ) ).z.greaterThan( rayZ( sm ) ), () => {

							sHi.assign( sm );

						} ).Else( () => {

							sLo.assign( sm );

						} );

					} );
					const uvH = uvN.add( span.mul( sHi ) ).toVar();
					const vP = viewAt( uvH ).toVar();
					const tk = MU.thick.add( vP.z.abs().mul( 0.01 ) );
					const edge = min( min( uvH.x, oneMinus( uvH.x ) ), min( uvH.y, oneMinus( uvH.y ) ) );
					const conf = smoothstep( tk, tk.mul( 0.35 ), vP.z.sub( rayZ( sHi ) ).abs() ) // passed behind something
						.mul( smoothstep( 0.0, 0.06, edge ) ) // what lies beyond the frame is unknown
						.mul( smoothstep( 1.0, 0.8, sHi.mul( step( 0.999, sMax ) ) ) ) // end of the ray
						.mul( smoothstep( 0.0, 0.06, dot( vP.sub( P0 ), Nv ) ) ) // the surface itself
						.mul( smoothstep( 0.45, 0.1, R.z ) ) // coming back at the camera: it would see the hidden side
						.mul( smoothstep( MU.hit, MU.hit.mul( 0.5 ), length( vP.sub( P0 ) ) ) ); // far finds: the mirror image's
					// drawn out along the ray by the roughness
					const sm = wv.g.mul( MU.smear ).mul( sHi );
					const tap = ( k, w ) => colorTex.sample( uvN.add( span.mul( sHi.add( sm.mul( k ) ) ) ) ).rgb.mul( w );
					const c = tap( 0, 0.3 ).add( tap( - 0.5, 0.2 ) ).add( tap( 0.5, 0.2 ) ).add( tap( - 1, 0.15 ) ).add( tap( 1, 0.15 ) ).toVar();
					const lm = max( max( c.r, c.g ), max( c.b, 1e-4 ) );
					c.mulAssign( min( MU.max.div( lm ), 1 ) );
					hit.assign( vec4( c, conf ) );

				} );
				// ... and where it found nothing, the town at large (the probe): sharp in standing water, on wet stone a
				// broad blur of what stands low around - lit walls, windows, lanterns: the warm sheen of all wet paving
				// (levels as of a 128 cube; never below the horizon)
				let far = vec3( 0 );
				if ( probe && probe.planar ) {

					// the mirror image: the point `tg` metres along the reflected ray as the mirrored camera sees it (on the
					// paving the camera stands on that is the pixel's own place in the image, wherever along the ray).
					// Roughness draws it out towards the viewer far more than sideways (seen at a glancing angle a tilt of
					// the surface along the line of sight moves the mirrored point up and down, a tilt across it hardly
					// at all): `taps` samples above and below, each blurred by the distance to the next.
					const Pw = world.mul( vec4( P0, 1 ) ).xyz, Rw = world.mul( vec4( R, 0 ) ).xyz;
					const tg = MU.tg.add( Pw.y.sub( probe.g ).abs().mul( MU.tk ) );
					const C0 = probe.vp.mul( vec4( Pw.add( vec3( Rw.x, 0, Rw.z ).mul( tg ) ), 1 ) ).toVar();
					const D = probe.vp.mul( vec4( 0, 1, 0, 0 ) ).mul( tg ).toVar();
					const dl = clamp( wv.g.mul( wv.g ).mul( MU.lobe ), 0.004, 0.7 ).toVar();
					const NT = Math.max( 1, Math.round( T.taps || 5 ) );
					const lv = max( log2( max( dl.mul( probe.focal ).mul( MU.gap ).mul( 2 / NT ), 1 ) ), MU.lmin ).toVar();
					let acc = vec3( 0 ), ws = 0;
					for ( let k = 0; k < NT; k ++ ) {

						const ek = ( k + 0.5 ) / NT * 2 - 1, wk = Math.exp( - 2.5 * ek * ek );
						const c = C0.add( D.mul( max( Rw.y.add( dl.mul( ek ) ), 0.01 ) ) );
						const u = c.xy.div( max( c.w, 1e-3 ) );
						acc = acc.add( probe.node.sample( vec2( u.x.mul( 0.5 ).add( 0.5 ), u.y.mul( - 0.5 ).add( 0.5 ) ) ).level( lv ).rgb.min( MU.max ).mul( wk ) );
						ws += wk;

					}

					far = acc.mul( MU.penv.div( ws ) ).mul( probe.on );

				} else if ( probe ) {

					const Rw = world.mul( vec4( R, 0 ) ).xyz;
					const stone = smoothstep( 0.1, 0.2, wv.g );
					const lv = min( wv.g.mul( MU.blur ).add( 1 ), MU.pblur.negate().add( 7 ) ).sub( 7 - probe.levels ).max( 0 );
					far = probe.node.sample( vec3( Rw.x, max( Rw.y, 0.03 ).mul( mix( float( 1 ), MU.plow, stone ) ), Rw.z ) ).level( lv ).rgb.min( MU.max ).mul( MU.env ).mul( probe.on );

				}

				out.assign( vec4( mix( far, hit.rgb, hit.a.mul( MU.ssr ) ).mul( wv.r ), wv.r ) );

			} );
			return out.max( 0 );

		} );
		mirrorPass = rtt( mirrorFn(), null, null, { type: THREE.HalfFloatType } );
		P.ssr.value = 1;
		comp = comp.add( mirrorPass.rgb.mul( P.ssr ) );
		// checking: ?show=mirror (the mirror image alone), ?show=wet (strength, roughness, tilt), ?m_reach=20 ...
		const q = new URLSearchParams( location.search ), show = q.get( 'show' );
		if ( show === 'mirror' ) comp = mirrorPass.rgb;
		if ( show === 'planar' && probe && probe.planar ) comp = probe.node.sample( vec2( screenUV.x, screenUV.y ) ).level( 0 ).rgb;
		if ( show === 'wet' ) comp = vec3( wetTex.r, wetTex.g, wetTex.b.sub( 0.5 ).abs().add( wetTex.a.sub( 0.5 ).abs() ).mul( 2 ) ).mul( 0.5 );
		for ( const k in MU ) { const v = parseFloat( q.get( 'm_' + k ) ); if ( isFinite( v ) ) MU[ k ].value = v; }

	}

	let bloomPass = null;
	if ( T.bloom ) {

		bloomPass = bloom( comp, 1.0, 0.55, 1.1 );
		comp = comp.add( bloomPass.rgb.mul( P.bloom ) );

	}

	const vg = screenUV.sub( 0.5 );
	comp = comp.mul( oneMinus( dot( vg, vg ).mul( P.vignette ).mul( 1.6 ) ) ).mul( P.tint ).add( P.lift );
	comp = pow( max( comp, 0 ).div( 0.18 ), vec3( P.contrast ) ).mul( 0.18 );
	comp = mix( vec3( dot( comp, vec3( 0.2126, 0.7152, 0.0722 ) ) ), comp, P.saturation ).max( 0 );
	const gr = fract( sin( dot( screenUV.mul( 917.3 ).add( fract( time ) ), vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) ).sub( 0.5 );
	comp = comp.add( gr.mul( P.grain ).mul( comp.add( 0.02 ) ) );
	pipeline.outputColorTransform = false;
	const dither = vec4( vec3( interleavedGradientNoise( screenCoordinate.xy ).sub( 0.5 ).div( 160 ) ), 0 );
	const shown = renderOutput( vec4( comp, 1 ) );
	pipeline.outputNode = ( T.fxaa ? fxaa( shown ) : shown ).add( dither );

	let captureRT = null;
	const stub = () => ( { value: 0 } );
	const eng = {
		lite: true, renderer, scene, camera, pipeline, sky, moonDir, P, FOG, fogAmount, knee, noise, bloomPass, isWebGPU, width, height,
		mirrorPass, mirror: eng0.mirror || null, probe,
		// what a material that mirrors writes beside its colour (tiers with `mirror`): k strength 0..1, the roughness,
		// N its mean normal in world space (near level: 30 degrees of tilt at most)
		mirrorOut: T.mirror ? ( k, rough, N ) => mrt( { output, wet: vec4( saturate( k ), rough, N.x.add( 0.5 ), N.z.add( 0.5 ) ) } ) : null,
		lanterns: [], flames: [], partSets: [],
		// the material look is baked: these only keep the page's look code working (it sets them for the streaming path)
		lib: { noise, warnings: new Set(), u: { wet: stub(), aoDirect: stub(), grime: stub(), wear: stub(), special: stub(), macro: stub(), windowGlow: stub(), emissive: stub(),
			puddleRough: stub(), filmRough: stub(), ripple: stub(), poolCap: stub(), tideY: stub(), tideH: stub(), soft: stub() } },

		setSize( w, h, pr = pixelRatio ) {

			eng.width = w; eng.height = h;
			renderer.setPixelRatio( pr );
			renderer.setSize( w, h, false );
			camera.aspect = w / h;
			camera.updateProjectionMatrix();

		},

		update() {

			sky.position.copy( camera.position );

		},

		// the lighting half of the look (ambient, moon, lamp) is in the bake; the rest is live
		setRig( name, o = {} ) {

			const SU = sky.userData.U;
			SU.gain.value = 1;
			renderer.toneMappingExposure = o.exposure ?? 1.25;
			P.bloom.value = o.bloom ?? 0.32;
			P.vignette.value = o.vignette ?? 0.35;
			FOG.color.value.setRGB( 0.016, 0.026, 0.058 ).lerp( new THREE.Color( 0.085, 0.07, 0.075 ), o.haze ?? 0 );
			FOG.density.value = o.fog ?? 0.0;
			SU.glowGain.value = o.skyGlow ?? 1.0;
			SU.cover.value = o.cover ?? 0.46;
			SU.cloudGain.value = o.cloudGain ?? 1.0;
			SU.horizon.value.setRGB( 0.018, 0.034, 0.085 );
			SU.cloud.value.setRGB( 0.05, 0.075, 0.14 );
			SU.clouds.value = o.clouds ?? 1.0;
			eng.rig = name;

		},

		setPost() {},

		manual: false,
		cull: null, // (the town's own culling for the frame's pass, see BakedTown._cull)
		tick() {

			if ( eng.cull ) eng.cull();
			if ( eng.manual ) {

				renderer.info.reset();
				renderer._nodes.nodeFrame.update();
				renderer.info.frame = renderer._nodes.nodeFrame.frameId;

			}

		},

		render() {

			if ( probe ) probe.step();
			eng.tick();
			pipeline.render();

		},

		// offscreen render -> ImageData (sRGB, top row first); same contract as the full engine's capture()
		async capture() {

			const pr = renderer.getPixelRatio();
			const w = Math.round( eng.width * pr ), h = Math.round( eng.height * pr );
			if ( probe ) probe.now();
			if ( ! isWebGPU ) {

				eng.tick();
				renderer.setRenderTarget( null );
				pipeline.render();
				const gl = renderer.backend.gl;
				const bw = gl.drawingBufferWidth, bh = gl.drawingBufferHeight;
				const raw = new Uint8Array( bw * bh * 4 ), img = new Uint8ClampedArray( bw * bh * 4 );
				gl.bindFramebuffer( gl.FRAMEBUFFER, null );
				gl.readPixels( 0, 0, bw, bh, gl.RGBA, gl.UNSIGNED_BYTE, raw );
				for ( let y = 0; y < bh; y ++ ) img.set( raw.subarray( y * bw * 4, ( y + 1 ) * bw * 4 ), ( bh - 1 - y ) * bw * 4 );
				for ( let i = 3; i < img.length; i += 4 ) img[ i ] = 255;
				return new ImageData( img, bw, bh );

			}

			if ( ! captureRT || captureRT.width !== w || captureRT.height !== h ) {

				captureRT?.dispose();
				captureRT = new THREE.RenderTarget( w, h, { type: THREE.UnsignedByteType, colorSpace: THREE.NoColorSpace, depthBuffer: false } );

			}

			eng.tick();
			renderer.setRenderTarget( captureRT );
			pipeline.render();
			renderer.setRenderTarget( null );
			const px = await renderer.readRenderTargetPixelsAsync( captureRT, 0, 0, w, h );
			const out = new Uint8ClampedArray( w * h * 4 );
			if ( px.length === out.length ) out.set( px );
			else {

				const stride = Math.ceil( w * 4 / 256 ) * 256;
				for ( let y = 0; y < h; y ++ ) out.set( px.subarray( y * stride, y * stride + w * 4 ), y * w * 4 );

			}

			return new ImageData( out, w, h );

		},
	};

	eng.setRig( 'night' );
	return eng;

}

// ---------------------------------------------------------------------------------------- town

const EMPTY16 = new Uint16Array( 0 );

export class BakedTown {

	// wait: 'all' (review captures: every level the view asks for is in before a frame counts) or 'near' (walking:
	// start as soon as the surroundings of the camera are in, the rest streams)
	constructor( eng, bake, { wait = 'near' } = {} ) {

		this.eng = eng;
		this.bake = bake;
		this.T = bake.tier;
		this.wait = wait;
		this.data = bake.data;
		this.nav = new Nav( this.data );
		// (under a mirror image the town is cut at the mirror's plane while that image is drawn, see `probe.clip`)
		const clip = eng.probe && eng.probe.clip && ! eng.probe.oblique ? eng.probe.clip : null;
		this.group = clip ? new THREE.ClippingGroup() : new THREE.Group();
		if ( clip ) { this.group.clippingPlanes = [ clip ]; this.group.enabled = false; }
		this.group.name = 'town';
		eng.scene.add( this.group );
		this.warnings = [];
		this.bytes = 0; // downloaded so far (manifest excluded)
		this.inflight = new Map(); // chunk level key -> promise
		this.waiters = [];
		this.loTris = bake.chunks.reduce( ( n, m ) => n + m.lo.ni / 3, 0 );
		if ( new URLSearchParams( location.search ).get( 'box' ) !== '0' ) { this._pv = new THREE.Matrix4(); this._planes = new Float64Array( 16 ); eng.cull = () => this._cull( eng.camera ); }
		// the levels of a cell: lowest, mid, and the tier's own top one (a bake without the phone's ring: none there) -
		// or, with the fine level on top, four; R: the distance inside which each is shown
		const T = this.T;
		const L = this.L = T.top === 'fine' ? [ 'lo', 'mid', 'near', 'fine' ] : [ 'lo', 'mid', T.top || 'near' ];
		this.R = T.top === 'fine' ? [ Infinity, T.mid, T.near, T.fine ] : [ Infinity, T.mid, T.near ];
		// ?lmk=2 (trial): the cells the packer marked as a landmark's (lmk in the manifest) keep each level that many times further out
		this.lmkK = Math.max( 0, parseFloat( new URLSearchParams( location.search ).get( 'lmk' ) ) || 0 );
		this.fadeTime = wait === 'all' ? 0 : T.fade || 0; // (review captures show the settled picture)
		this.fades = new Set();
		this.chunks = bake.chunks.map( ( m ) => {

			const b0 = m.lo.bmin, b1 = m.lo.bmax;
			const box = [ b0[ 0 ], b0[ 2 ], b1[ 0 ], b1[ 2 ] ];
			for ( const l of L.slice( 1 ) ) if ( m[ l ] ) {

				box[ 0 ] = Math.min( box[ 0 ], m[ l ].bmin[ 0 ] ); box[ 1 ] = Math.min( box[ 1 ], m[ l ].bmin[ 2 ] );
				box[ 2 ] = Math.max( box[ 2 ], m[ l ].bmax[ 0 ] ); box[ 3 ] = Math.max( box[ 3 ], m[ l ].bmax[ 2 ] );

			}

			// per level, for the view budget: centre (x, z), radius, triangles and centre height of the sphere the renderer
			// culls it by
			const sph = L.map( ( l ) => m[ l ] && [ ( m[ l ].bmin[ 0 ] + m[ l ].bmax[ 0 ] ) / 2, ( m[ l ].bmin[ 2 ] + m[ l ].bmax[ 2 ] ) / 2,
				0.5 * Math.hypot( m[ l ].bmax[ 0 ] - m[ l ].bmin[ 0 ], m[ l ].bmax[ 1 ] - m[ l ].bmin[ 1 ], m[ l ].bmax[ 2 ] - m[ l ].bmin[ 2 ] ), m[ l ].ni / 3,
				( m[ l ].bmin[ 1 ] + m[ l ].bmax[ 1 ] ) / 2 ] );
			// the clusters of the top level's opaque range (the phone's ring: each a mesh of its own, drawn down to a part
			// size): sphere, box on the ground, [ part size, index count down to it ] steps; rest: the triangles outside them
			const t = m[ L[ L.length - 1 ] ];
			const top = ( t && t.parts || [] ).map( ( [ sub, steps, b ] ) => ( {
				sph: [ ( b[ 0 ] + b[ 3 ] ) / 2, ( b[ 2 ] + b[ 5 ] ) / 2, 0.5 * Math.hypot( b[ 3 ] - b[ 0 ], b[ 4 ] - b[ 1 ], b[ 5 ] - b[ 2 ] ), 0, ( b[ 1 ] + b[ 4 ] ) / 2 ],
				box: [ b[ 0 ], b[ 2 ], b[ 3 ], b[ 5 ] ], steps, full: t.sub[ sub ][ 2 ], n: t.sub[ sub ][ 2 ], d: 0 } ) );
			const rest = t ? t.ni / 3 - top.reduce( ( n, q ) => n + q.full / 3, 0 ) : 0;
			const g = /^c_(-?\d+)_(-?\d+)$/.exec( m.k ); // (the cell of the grid it is: its things may reach into the next)
			return { m, box, sph, top, rest, f: 0, mesh: L.map( () => null ), level: 0, want: 0, d: 0, fade: null, gx: g ? + g[ 1 ] : NaN, gz: g ? + g[ 2 ] : NaN };

		} );
		this.baseLeft = this.chunks.length;

	}

	_dist( c, x, z ) {

		const dx = Math.max( c.box[ 0 ] - x, 0, x - c.box[ 2 ] ), dz = Math.max( c.box[ 1 ] - z, 0, z - c.box[ 3 ] );
		return Math.hypot( dx, dz );

	}

	async load( url, onProgress = () => {} ) {

		const bake = this.bake, eng = this.eng;
		ktxInfo( this ); // (town.loadLog; ?ktxinfo=1: on the page)
		await MeshoptDecoder.ready;
		if ( this.T.workers ) MeshoptDecoder.useWorkers( this.T.workers );
		onProgress( 0.02, 'textures' );
		const texP = this._loadLayers();
		if ( bake.lm ) await this._loadLM();
		this._materials( await texP );
		this._sprites();
		if ( eng.probe ) {

			// the reflection probe sees the town at its lowest level (always resident, a tenth of the triangles: its
			// faces are 64 or 128 pixels across), so a probe face costs a fraction of a frame's triangles, not a second frame
			const held = [];
			// (the mirror image: the highest walkable surface under the camera)
			eng.probe.ground = ( p ) => {

				let best = null;
				for ( const r of this.nav.regions ) {

					const R = r.rect;
					if ( p.x < R[ 0 ] || p.x > R[ 2 ] || - p.z < R[ 1 ] || - p.z > R[ 3 ] ) continue;
					const h = this.nav.heightOf( r, p.x, - p.z );
					if ( h <= p.y - 0.3 && ( best === null || h > best ) ) best = h;

				}

				return best;

			};
			eng.probe.swap = ( on ) => {

				// (the image is of the lowest level, seen from near: its glowing cards come forward by more, see PULL)
				this.pullAll.value.set( on ? this.PULL.mirror[ 0 ] : 0, on ? this.PULL.mirror[ 1 ] : 0 );
				if ( this.group.isClippingGroup ) this.group.enabled = on;
				if ( ! on ) { for ( const m of held ) m.visible = ! m.visible; held.length = 0; return; }
				for ( const c of this.chunks ) {

					const lo = c.mesh[ 0 ];
					if ( ! lo ) continue;
					for ( const o of lo.userData.all ) o.layers.mask = 1; // (this view is not the one _cull() looked through)
					let any = false;
					for ( let l = 1; l < c.mesh.length; l ++ ) { const m = c.mesh[ l ]; if ( m && m.visible ) { m.visible = false; held.push( m ); any = true; } }
					if ( any && ! lo.visible ) { lo.visible = true; held.push( lo ); }

				}

			};

		}
		// base.bin holds every chunk's lowest level, nearest the spawn first: chunks appear as their bytes arrive
		const sp = this.data.spawn.pos;
		const need = this.wait === 'all' ? Infinity : 90;
		let first = 0;
		this.chunks.forEach( ( c, i ) => {

			if ( this._dist( c, sp[ 0 ], - sp[ 1 ] ) < need ) first = i + 1;

		} );
		let ready;
		const enough = new Promise( ( r ) => ( ready = r ) );
		this.baseDone = this._streamBase( ( n ) => {

			onProgress( 0.1 + 0.9 * Math.min( 1, n / first ), 'town' );
			if ( n >= first ) ready();

		} ).then( ready, ( e ) => {

			this.warnings.push( 'baked town: ' + ( e.message || e ) );
			ready();
			throw e;

		} );
		await enough;
		// the sharp layers come in behind the first frame (a review capture waits for them)
		this.sharp = this._sharpen().catch( ( e ) => { this.warnings.push( 'baked town: sharp textures: ' + ( e.message || e ) ); } );
		this.detail = this.sharp.then( () => this._loadNR() ).catch( ( e ) => { this.warnings.push( 'baked town: normal layers: ' + ( e.message || e ) ); } );
		this.tcDone = this.detail.then( () => ! this.ktx && tcFirst( this ) ); // ?tc=1: the first array, compressed here and swapped in (lite_tc.js)
		if ( this.wait === 'all' ) { await this.baseDone; await this.sharp; await this.detail; await this.tcDone; }
		else this.baseDone.catch( () => {} );
		return this;

	}

	// lightmap prototype (?lm=): the pages of all bundles as one array of RGBA8 layers - rgb = ( light / range ) ^ ( 1 / 2 ),
	// a = gloss; no mips (a texel of 20 / 10 / 5 cm is a pixel at about the distance its level is shown from)
	async _loadLM() {

		const lm = this.bake.lm, S = lm.size, n = lm.pages.length;
		const data = new Uint8Array( S * S * 4 * n );
		await Promise.all( lm.pages.map( async ( url, i ) => {

			const r = await fetch( url );
			if ( ! r.ok ) throw new Error( `lightmap page ${i}: ${r.status}` );
			data.set( new Uint8Array( await r.arrayBuffer() ), i * S * S * 4 );
			this.bytes += S * S * 4;

		} ) );
		const t = new THREE.DataArrayTexture( data, S, S, n );
		t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType; t.colorSpace = THREE.NoColorSpace;
		t.magFilter = t.minFilter = new URLSearchParams( location.search ).get( 'lmf' ) === '0' ? THREE.NearestFilter : THREE.LinearFilter;
		t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.generateMipmaps = false; t.needsUpdate = true;
		lm.tex = t;

	}

	async _streamBase( onChunk ) {

		const bake = this.bake;
		// the same bytes out of base.glb where the bake has it (../../engine/lite_glb.js); false: the packed file then
		if ( glbOn( this ) && await glbBase( this, ( c, bytes ) => {

			c.mesh[ 0 ] = this._build( bytes, c.m.lo );
			c.mesh[ 0 ].visible = c.level === 0;
			this.group.add( c.mesh[ 0 ] );
			this._dirty = true;
			this.baseLeft --;

		}, onChunk ) ) { this._wake(); return; }

		const r = await fetch( `${bake.dir}/${bake.base.file}?v=${bake.baked}` );
		if ( ! r.ok ) throw new Error( `${bake.base.file}: ${r.status}` );
		let buf = new Uint8Array( bake.base.bytes );
		let got = 0, next = 0;
		const flush = () => {

			while ( next < this.chunks.length ) {

				const c = this.chunks[ next ], m = c.m.lo;
				if ( ! m.data && m.off + m.len > got ) break;
				c.mesh[ 0 ] = this._build( m.data || buf.subarray( m.off, m.off + m.len ), m );
				c.mesh[ 0 ].visible = c.level === 0;
				this.group.add( c.mesh[ 0 ] );
				this._dirty = true;
				next ++; this.baseLeft --;
				onChunk( next );

			}

		};

		if ( r.body && r.body.getReader ) {

			const rd = r.body.getReader();
			for ( ;; ) {

				const { done, value } = await rd.read();
				if ( done ) break;
				if ( got + value.length > buf.length ) throw new Error( 'base.bin is larger than the manifest says (stale bake?)' );
				buf.set( value, got ); got += value.length;
				flush();

			}

		} else {

			buf.set( new Uint8Array( await r.arrayBuffer() ) ); got = buf.length;
			flush();

		}

		this.bytes += got;
		if ( next < this.chunks.length ) throw new Error( 'base.bin is shorter than the manifest says (stale bake?)' );
		buf = null;
		this._wake();

	}

	// texture array of pre-shaded albedo layers (rgb) with the relief height in alpha, from the WebP pages
	async _loadLayers() {

		{ const k = await ktxLayers( this ); if ( k ) return ( this.layers = k ); } // ?ktx=1: the KTX2 array (lite_ktx.js)
		if ( ! this.ktx ) { const k = await tcLayers( this ); if ( k ) return ( this.layers = k ); } // ?tc=1&tcfirst=show (lite_tc.js)
		const bake = this.bake, t = bake.tex[ this.T.tex ], size = t.size, n = bake.layers.length, per = t.cols * t.rows;
		const data = new Uint8Array( size * size * 4 * n );
		const page = async ( name ) => {

			const r = await fetch( `${bake.dir}/${name}.webp?v=${bake.baked}` );
			if ( ! r.ok ) throw new Error( `${name}.webp: ${r.status}` );
			const blob = await r.blob();
			this.bytes += blob.size;
			const bmp = await createImageBitmap( blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
			const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( bmp.width, bmp.height ) : Object.assign( document.createElement( 'canvas' ), { width: bmp.width, height: bmp.height } );
			const cx = cv.getContext( '2d', { willReadFrequently: true } );
			cx.globalCompositeOperation = 'copy';
			cx.drawImage( bmp, 0, 0 );
			bmp.close?.();
			return cx;

		};

		await Promise.all( t.pages.map( async ( name, p ) => {

			const cx = await page( name );
			const hx = t.hpages ? await page( t.hpages[ p ] ) : null;
			for ( let i = p * per; i < Math.min( n, ( p + 1 ) * per ); i ++ ) {

				const k = i - p * per, col = k % t.cols, row = ( k - col ) / t.cols;
				const px = cx.getImageData( col * size, row * size, size, size ).data;
				if ( hx ) {

					const hp = hx.getImageData( col * size, row * size, size, size ).data;
					for ( let j = 0; j < px.length; j += 4 ) px[ j + 3 ] = hp[ j ];

				}

				data.set( px, i * size * size * 4 );

			}

		} ) );
		const tex = new THREE.DataArrayTexture( data, size, size, n );
		tex.format = THREE.RGBAFormat; tex.type = THREE.UnsignedByteType;
		tex.colorSpace = THREE.SRGBColorSpace;
		tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
		tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
		tex.generateMipmaps = true;
		tex.anisotropy = this.T.aniso;
		tex.needsUpdate = true;
		this.layers = tex;
		return tex;

	}

	// medium / high: the single-layer files, once the first frame is out. Two arrays: `big` (the first layers of the
	// list at the larger size - the large-tiled walls, roofs and paving, where a 512 layer is 4 mm and more a texel)
	// and, with T.sharp, every other layer at that size in place of the pages' array. Each is swapped in whole when
	// its last layer is decoded: one upload, no half-sharp town.
	async _sharpen() {

		const bake = this.bake, T = this.T, N = bake.layers.length, lb = bake.layer_big || [ 0, 0 ];
		if ( ! bake.texl || ! ( T.big || T.sharp ) || ! this.TX ) return;
		const nb = T.big ? Math.min( N - 2, T.big[ 1 ] === 'top' ? lb[ 0 ] : lb[ 0 ] + lb[ 1 ] ) : 0;
		if ( this.ktx && await ktxSharpen( this, nb ) ) return;
		if ( ! this.ktx && await tcSharpen( this, nb ) ) return; // ?tc=1: WebP, compressed here (lite_tc.js)
		let turn = 0;
		const jobs = [], slots = Array.from( { length: Math.max( 2, T.fetch || 3 ) }, () => Promise.resolve() );
		const queue = ( fn ) => { const i = turn ++ % slots.length; return ( slots[ i ] = slots[ i ].then( fn ) ); };
		const array = ( size, layers ) => {

			const t = bake.texl[ size ];
			if ( ! t || ! layers.length || layers.some( ( i ) => ! t.files[ i ] ) ) return null;
			const data = new Uint8Array( size * size * 4 * layers.length );
			return Promise.all( layers.map( ( i, k ) => queue( async () => {

				const r = await fetch( `${bake.dir}/${t.files[ i ]}.webp?v=${bake.baked}` );
				if ( ! r.ok ) throw new Error( `${t.files[ i ]}.webp: ${r.status}` );
				const blob = await r.blob();
				this.bytes += blob.size;
				const bmp = await createImageBitmap( blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
				const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( size, size ) : Object.assign( document.createElement( 'canvas' ), { width: size, height: size } );
				const cx = cv.getContext( '2d', { willReadFrequently: true } );
				cx.globalCompositeOperation = 'copy';
				cx.drawImage( bmp, 0, 0, size, size );
				bmp.close?.();
				data.set( cx.getImageData( 0, 0, size, size ).data, k * size * size * 4 );

			} ) ) ).then( () => {

				const tex = new THREE.DataArrayTexture( data, size, size, layers.length );
				tex.format = THREE.RGBAFormat; tex.type = THREE.UnsignedByteType;
				tex.colorSpace = THREE.SRGBColorSpace;
				tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
				tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
				tex.generateMipmaps = true;
				tex.anisotropy = T.aniso;
				tex.onUpdate = () => { tex.image.data = null; tex.onUpdate = null; }; // on the GPU: the copy here has done its work
				tex.needsUpdate = true;
				return tex;

			} );

		};

		const all = Array.from( { length: N }, ( _, i ) => i );
		const big = nb > 0 ? array( T.big[ 0 ], all.slice( 2, 2 + nb ) ) : null;
		const rest = T.sharp ? array( T.sharp, big ? all.filter( ( i ) => i < 2 || i >= 2 + nb ) : all ) : null;
		const TX = this.TX;
		if ( big ) jobs.push( big.then( ( tex ) => {

			for ( const n of TX.b ) n.value = tex;
			TX.u.b0.value = 2; TX.u.b1.value = 2 + nb;
			this.layersBig = tex;

		} ) );
		// (the pages' array stays until both are in: its layers under the big ones are what the remap of `rest` leaves out)
		if ( rest ) jobs.push( Promise.all( [ rest, big ] ).then( ( [ tex ] ) => {

			const old = this.layers;
			for ( const n of TX.a ) n.value = tex;
			TX.u.shift.value = big ? nb : 0;
			this.layers = tex;
			old.dispose();

		} ) );
		await Promise.all( jobs );

	}

	// T.nr: the materials' normal + roughness layers (n<size>_<layer>.webp: normal x, y and roughness in r, g, b), one
	// array, behind the sharp colour layers. Until it is in, the directional material shades the height relief alone.
	async _loadNR() {

		const NR = this.NR, bake = this.bake;
		if ( ! NR ) return;
		if ( this.ktx && await ktxNR( this ) ) return;
		if ( ! this.ktx && await tcNR( this ) ) return;
		const size = NR.size, px = size * size * 4;
		const data = new Uint8Array( px * NR.count );
		for ( let i = 0; i < data.length; i += 4 ) { data[ i ] = data[ i + 1 ] = 128; data[ i + 2 ] = 153; data[ i + 3 ] = 255; }
		const slots = Array.from( { length: Math.max( 2, this.T.fetch || 3 ) }, () => Promise.resolve() );
		let turn = 0;
		await Promise.all( NR.files.map( ( file, k ) => {

			if ( ! file ) return null;
			const i = turn ++ % slots.length;
			return ( slots[ i ] = slots[ i ].then( async () => {

				const r = await fetch( `${bake.dir}/${file}.webp?v=${bake.baked}` );
				if ( ! r.ok ) throw new Error( `${file}.webp: ${r.status}` );
				const blob = await r.blob();
				this.bytes += blob.size;
				const bmp = await createImageBitmap( blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
				const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( size, size ) : Object.assign( document.createElement( 'canvas' ), { width: size, height: size } );
				const cx = cv.getContext( '2d', { willReadFrequently: true } );
				cx.globalCompositeOperation = 'copy';
				cx.drawImage( bmp, 0, 0, size, size );
				bmp.close?.();
				data.set( cx.getImageData( 0, 0, size, size ).data, k * px );

			} ) );

		} ) );
		const tex = new THREE.DataArrayTexture( data, size, size, NR.count );
		tex.format = THREE.RGBAFormat; tex.type = THREE.UnsignedByteType;
		tex.colorSpace = THREE.NoColorSpace;
		tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
		tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearMipmapLinearFilter;
		tex.generateMipmaps = true;
		tex.anisotropy = this.T.aniso;
		tex.onUpdate = () => { tex.image.data = null; tex.onUpdate = null; };
		tex.needsUpdate = true;
		for ( const n of NR.nodes ) n.value = tex;
		NR.on.value = 1;
		NR.tex = tex;

	}

	// The K lanterns that light each ground cell most, as a small float texture ( per cell: one texel of thresholds,
	// then position + range and colour x intensity + cap of each ). The diffuse light of every lantern is in the vertex
	// colours; this only feeds the view-dependent part - the glints on wet cobbles, polished stone and the canal -
	// which cannot be baked.
	// A list of the K strongest per cell alone leaves a straight edge in the floor's shine wherever a lantern is in one
	// cell's list and not in the next one's (squares of wet paving with and without a lamp's highlight). So the choice
	// is made a continuous function of the place on the ground: with e the light a lantern sends to a point of the
	// floor, a lantern counts in full where e is over GLINT_FADE x the threshold there and not at all where it is under
	// it. The threshold is interpolated between the cell corners, and a corner's is the largest "strongest light of a
	// lantern left out" of the four cells around it - so a lantern has always faded to nothing by the time it reaches
	// a cell that does not list it, and it fades where it matters least: where K others outshine it.
	// Lanterns of one colour closer together than the tier's `merge` (the heads of one post, the bulbs of one festoon,
	// the lamps of one shop front) share an entry: with few entries per cell, a street of many small lights would
	// otherwise keep only a few of them and lose most of its shine.
	_lightGrid( K ) {

		const src = this.bake.lights || [];
		if ( ! src.length || ! ( K > 0 ) ) return null;
		const lu = ( l ) => 0.2126 * l[ 3 ] + 0.7152 * l[ 4 ] + 0.0722 * l[ 5 ];
		const L = [], taken = new Uint8Array( src.length );
		const merge = this.T.merge ?? GLINT_MERGE;
		// (T.dir: a burner with a fitting - a halo row with its flag - is never folded into another entry. Its own
		// metalwork takes a highlight from it, see _glints: from the middle of a post's two heads that would be none)
		const fit = this.T.dir ? ( this.bake.halos || [] ).filter( ( h ) => h[ 7 ] ) : [];
		const strong = fit.length ? src.map( ( l ) => fit.some( ( h ) => Math.abs( h[ 0 ] - l[ 0 ] ) < 0.03 && Math.abs( h[ 1 ] - l[ 1 ] ) < 0.03 && Math.abs( h[ 2 ] - l[ 2 ] ) < 0.03 ) ) : null;
		const alike = ( a, b ) => { const u = lu( a ) + 1e-6, v = lu( b ) + 1e-6; return Math.max( Math.abs( a[ 3 ] / u - b[ 3 ] / v ), Math.abs( a[ 4 ] / u - b[ 4 ] / v ), Math.abs( a[ 5 ] / u - b[ 5 ] / v ) ) < 0.3; }; // a blue sign keeps its own
		const order = src.map( ( l, i ) => i ).sort( ( a, b ) => lu( src[ b ] ) - lu( src[ a ] ) || a - b );
		for ( const i of order ) {

			if ( taken[ i ] ) continue;
			const a = src[ i ], m = [ 0, 0, 0, 0, 0, 0, 0, 0 ];
			let sum = 0;
			for ( const j of order ) {

				const b = src[ j ];
				if ( taken[ j ] || Math.hypot( a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] ) > merge || ! alike( a, b ) ) continue;
				if ( strong && j !== i && strong[ j ] ) continue;
				taken[ j ] = 1;
				const u = lu( b ) + 1e-6;
				sum += u;
				for ( const c of [ 0, 1, 2 ] ) m[ c ] += b[ c ] * u;
				for ( const c of [ 3, 4, 5 ] ) m[ c ] += b[ c ];
				m[ 6 ] = Math.max( m[ 6 ], b[ 6 ] );
				m[ 7 ] += ( b[ 7 ] ?? 1 ) * u; // share of the light a capped lantern sends up

			}

			for ( const c of [ 0, 1, 2, 7 ] ) m[ c ] /= sum;
			L.push( m );

		}

		let x0 = Infinity, z0 = Infinity, x1 = - Infinity, z1 = - Infinity;
		for ( const l of L ) {

			x0 = Math.min( x0, l[ 0 ] - l[ 6 ] ); x1 = Math.max( x1, l[ 0 ] + l[ 6 ] );
			z0 = Math.min( z0, l[ 2 ] - l[ 6 ] ); z1 = Math.max( z1, l[ 2 ] + l[ 6 ] );

		}

		const G = Math.max( 1.25, ( x1 - x0 ) / 320, ( z1 - z0 ) / 320 );
		const nx = Math.max( 1, Math.ceil( ( x1 - x0 ) / G ) ), nz = Math.max( 1, Math.ceil( ( z1 - z0 ) / G ) );
		const lum = L.map( lu );
		const rf = this._reach(); // pools of light: x 1 / ( 1 + d2 * rf ), as the bake and the engine's reachFactor
		const list = new Int32Array( nx * nz * K ).fill( - 1 ), tau = new Float32Array( nx * nz );
		const best = [], row = [];
		for ( let iz = 0; iz < nz; iz ++ ) {

			const cz0 = z0 + iz * G;
			row.length = 0;
			for ( let i = 0; i < L.length; i ++ ) if ( L[ i ][ 2 ] > cz0 - L[ i ][ 6 ] && L[ i ][ 2 ] < cz0 + G + L[ i ][ 6 ] ) row.push( i );
			for ( let ix = 0; ix < nx; ix ++ ) {

				// the most each lantern sends to any point of the cell (GLINT_H2: floor to lantern, squared - the same
				// figure as in _glints, or the two would not agree on where a lantern ends)
				const cx0 = x0 + ix * G;
				best.length = 0;
				for ( const i of row ) {

					const l = L[ i ];
					const dx = Math.max( cx0 - l[ 0 ], 0, l[ 0 ] - cx0 - G ), dz = Math.max( cz0 - l[ 2 ], 0, l[ 2 ] - cz0 - G );
					const d2 = dx * dx + dz * dz + GLINT_H2;
					const w = Math.max( 0, 1 - ( d2 / ( l[ 6 ] * l[ 6 ] ) ) ** 2 );
					const e = lum[ i ] * w * w / d2 / ( 1 + d2 * rf );
					if ( e > 0 ) best.push( [ e, i ] );

				}

				best.sort( ( a, b ) => b[ 0 ] - a[ 0 ] || a[ 1 ] - b[ 1 ] );
				for ( let k = 0; k < K && k < best.length; k ++ ) list[ ( iz * nx + ix ) * K + k ] = best[ k ][ 1 ];
				tau[ iz * nx + ix ] = best.length > K ? best[ K ][ 0 ] : 0;

			}

		}

		const at = ( ix, iz ) => ix < 0 || iz < 0 || ix >= nx || iz >= nz ? 0 : tau[ iz * nx + ix ];
		const corner = ( cx, cz ) => Math.max( at( cx - 1, cz - 1 ), at( cx, cz - 1 ), at( cx - 1, cz ), at( cx, cz ) );
		const W = 2 * K + 1;
		const data = new Float32Array( nx * W * nz * 4 );
		for ( let iz = 0; iz < nz; iz ++ ) for ( let ix = 0; ix < nx; ix ++ ) {

			let o = ( iz * nx + ix ) * W * 4;
			data.set( [ corner( ix, iz ), corner( ix + 1, iz ), corner( ix, iz + 1 ), corner( ix + 1, iz + 1 ) ], o );
			for ( let k = 0; k < K; k ++ ) {

				const i = list[ ( iz * nx + ix ) * K + k ], l = i >= 0 ? L[ i ] : null;
				o += 8;
				if ( l ) data.set( [ l[ 0 ], l[ 1 ], l[ 2 ], l[ 6 ], l[ 3 ], l[ 4 ], l[ 5 ], l[ 7 ] ], o - 4 );
				else data.set( [ 0, - 1e4, 0, 1, 0, 0, 0, 1 ], o - 4 );

			}

		}

		const tex = new THREE.DataTexture( data, nx * W, nz, THREE.RGBAFormat, THREE.FloatType );
		tex.minFilter = tex.magFilter = THREE.NearestFilter;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		return { tex, K, G, x0, z0, nx, nz, W, lights: L.length };

	}

	// 1 / reach^2 of the lanterns' pools of light (0: the bake was lit with the pure inverse square)
	_reach() {

		const r = this.bake.look.reach || 0;
		return r > 0 ? 1 / ( r * r ) : 0;

	}

	// the highlight of a light in direction Lw on a surface with normal N seen from V (times the cosine)
	_brdf( N, V, Lw, rough, f0 ) {

		const a = rough.mul( rough ), a2 = a.mul( a );
		const H = normalize( Lw.add( V ) );
		const nh = saturate( dot( N, H ) ), nl = saturate( dot( N, Lw ) ), lh = saturate( dot( Lw, H ) );
		const den = nh.mul( nh ).mul( a2.sub( 1 ) ).add( 1 );
		const F = oneMinus( lh ).pow( 5 ).mul( oneMinus( f0 ) ).add( f0 );
		return a2.div( den.mul( den ).mul( max( lh.mul( lh ), 0.1 ) ).mul( a.add( 0.5 ) ).mul( 4 * Math.PI ) ).mul( nl ).mul( F );

	}

	// specular radiance of the cell's lanterns (and a little of the moon) leaving a surface with normal N towards V.
	// own (the fine level's variant): how far the baked light says "lit here", 0 .. 1 - the highlights have no shadows
	// of their own, so a surface in baked shadow gets none. Except within a pace of a burner: that is the lamp's own
	// metalwork - glazing bars, the rim, the roof's underside, black paint that holds next to no baked light however
	// hard the mantle shines on it - and nothing stands between the two; the bracket that carries the heads is the
	// same case, hence the wide end. There the highlight is whole, and with the bare inverse square: the streaming
	// engine's nearest lamps (its shadow pool) have no soft near field either (U.headSoft 1 would put it back).
	_glints( N, V, rough, f0, own = null ) {

		const g = this.grid, look = this.bake.look, rf = this._reach();
		const sr2 = ( look.source_radius ?? 0.6 ) ** 2;
		const brdf = ( Lw ) => this._brdf( N, V, Lw, rough, f0 );

		const gp = positionWorld.xz.sub( vec2( g.x0, g.z0 ) ).div( g.G );
		const c = clamp( floor( gp ), vec2( 0, 0 ), vec2( g.nx - 1, g.nz - 1 ) );
		const cf = saturate( gp.sub( c ) ); // place within the cell: the threshold is interpolated between its corners
		const ix = int( c.x ).mul( g.W ), iz = int( c.y );
		const T = textureLoad( g.tex, ivec2( ix, iz ) );
		const tau = mix( mix( T.x, T.y, cf.x ), mix( T.z, T.w, cf.x ), cf.y );
		const band = max( tau.mul( GLINT_FADE - 1 ), 1e-6 );
		let sum = vec3( 0 );
		for ( let k = 0; k < g.K; k ++ ) {

			const A = textureLoad( g.tex, ivec2( ix.add( 1 + 2 * k ), iz ) ), B = textureLoad( g.tex, ivec2( ix.add( 2 + 2 * k ), iz ) );
			const Lv = A.xyz.sub( positionWorld );
			// how much this lantern counts here (see _lightGrid): by its light on the floor at this place
			const h2 = dot( Lv.xz, Lv.xz ).add( GLINT_H2 ), hq = h2.div( A.w.mul( A.w ) ), hw = saturate( oneMinus( hq.mul( hq ) ) );
			const e = dot( B.rgb, vec3( 0.2126, 0.7152, 0.0722 ) ).mul( hw ).mul( hw ).div( h2.mul( h2.mul( rf ).add( 1 ) ) );
			const win = smoothstep( 0, 1, e.sub( tau ).div( band ) );
			const d2 = max( dot( Lv, Lv ), 0.01 ), d = sqrt( d2 );
			const q = d.div( A.w ), w = saturate( oneMinus( q.mul( q ).mul( q ).mul( q ) ) );
			const cap = mix( float( 1 ), B.w, smoothstep( - 0.42, 0.3, Lv.y.negate().div( d ) ) ); // capFactor of ClusteredLightsNode.js
			let one = B.rgb.mul( w.mul( w ).div( d2.mul( d2.mul( rf ).add( 1 ) ) ) ).mul( cap ).mul( win ).mul( brdf( Lv.div( d ) ) );
			if ( own ) {

				const head = oneMinus( smoothstep( 0.5, 2.0, d2 ) );
				one = one.mul( d2.div( d2.add( head.mul( sr2 ).mul( this.U.headSoft ) ) ) ).mul( max( own, head ) );

			}

			sum = sum.add( one );

		}

		const m = look.moon_col || [ 0.6, 0.7, 1 ], mk = ( look.moon ?? 1 ) * 0.5;
		const moon = vec3( m[ 0 ] * mk, m[ 1 ] * mk, m[ 2 ] * mk ).mul( brdf( vec3( this.eng.moonDir.x, this.eng.moonDir.y, this.eng.moonDir.z ) ) );
		return sum.add( own ? moon.mul( own ) : moon );

	}

	_materials( tex ) {

		const eng = this.eng, bake = this.bake;
		const pl = attribute( '_pl', 'vec4' ), cv = attribute( '_col', 'vec4' );
		// (lightmap prototype: bit 15 of the layer = this vertex has its light in a page, at the place in its colour slots)
		const lmBit = bake.lm ? step( 32767.5, pl.w.mul( 65535 ) ) : null;
		const layer = varying( bake.lm ? pl.w.mul( 65535 ).sub( lmBit.mul( 32768 ) ) : pl.w.mul( 65535 ), 'vLayer' );
		const tiles = varying( uv().mul( 65535 / 65536 ).sub( 0.5 ).mul( 2 * bake.uv_range ), 'vTiles' );
		let lit = varying( cv.rgb.mul( cv.rgb ).mul( bake.colour_scale ), 'vLit' ); // vertex colour: all light, already times the surface tint
		let aux = varying( cv.w, 'vAux' );
		if ( bake.lm ) {

			const on = varying( lmBit, 'vLm' ), at = varying( cv.rgb, 'vLmAt' );
			const s = texture( bake.lm.tex, at.xy ).depth( int( at.z.mul( 65535 / 256 ) ) );
			const how = new URLSearchParams( location.search ).get( 'lmshow' ); // lmshow=1: the pages alone; 2: where they are
			lit = how === '2' ? mix( lit, vec3( 0.2, 1.5, 0.2 ), on.mul( 0.7 ) ) : mix( how === '1' ? vec3( 0 ) : lit, s.rgb.mul( s.rgb ).mul( bake.lm.range ), on );
			aux = mix( aux, s.a.mul( 32767 / 65535 ), on );

		}

		const fogged = ( c ) => mix( c, eng.FOG.color, eng.fogAmount() );
		// a layer of the array - or, on the tiers with sharp textures, of one of two: the layers b0 .. b1 live in the
		// "big" array (a blank until it is in: b1 = b0), those after them `shift` places further down in the first
		const first = this.TX0 = []; // (the nodes that sample the first array: ?tc=1 swaps it, lite_tc.js)
		let sample = ( at ) => { const a = texture( tex, at ).depth( int( layer.add( 0.5 ) ) ); first.push( a ); return a; };
		if ( this.T.big || this.T.sharp ) {

			const blank = new THREE.DataArrayTexture( new Uint8Array( 4 ).fill( 255 ), 1, 1, 1 );
			blank.colorSpace = THREE.SRGBColorSpace; blank.wrapS = blank.wrapT = THREE.RepeatWrapping;
			blank.magFilter = blank.minFilter = THREE.LinearFilter; blank.needsUpdate = true;
			const TX = this.TX = { a: [], b: [], u: { b0: uniform( 2 ), b1: uniform( 2 ), shift: uniform( 0 ) } };
			sample = ( at ) => {

				const lf = floor( layer.add( 0.5 ) ), u = TX.u;
				const a = texture( tex, at ).depth( int( lf.sub( step( u.b1, lf ).mul( u.shift ) ) ) );
				const b = texture( blank, at ).depth( int( clamp( lf.sub( u.b0 ), 0, max( u.b1.sub( u.b0 ).sub( 1 ), 0 ) ) ) );
				TX.a.push( a ); TX.b.push( b );
				return mix( a, b, step( u.b0, lf ).mul( oneMinus( step( u.b1, lf ) ) ) );

			};

		}

		const tx = this.T.flat ? vec4( 0.6, 0.6, 0.6, 0.5 ) : sample( tiles ); // ?flat=1: the baked light alone (checking the bake)
		// a change of level is dithered across (T.fade): while it lasts the two meshes draw with these variants, the one
		// coming in where the screen noise is under its progress, the one going out where it is over
		const fadeU = uniform( 1 ).onObjectUpdate( ( { object } ) => object.userData.fade ?? 1 );
		const fadeMask = interleavedGradientNoise( screenCoordinate.xy ).sub( abs( fadeU ) ).mul( fadeU ).lessThan( 0 );
		const V = normalize( cameraPosition.sub( positionWorld ) );
		const SKY = vec3( 0.018, 0.034, 0.085 );
		this.grid = this._lightGrid( this.T.glints );
		const wetOn = !! ( this.T.sheet && eng.mirrorOut ); // (with the mirror pass the lantern glints share the paving with the mirrored town: less of them)
		const U = this.U = { relief: uniform( 0.05 ), glint: uniform( wetOn ? 0.6 : 1.5 ), sheen: uniform( 1.0 ), town: uniform( 0.4 ), swell: uniform( 0.3 ), wglow: uniform( 4 ),
			wetFlat: uniform( 1 ), ripple: uniform( 0.03 ), crown: uniform( 1.0 ), sunk: uniform( 0.9 ),
			level: uniform( 1 ), mirror: uniform( 1 ), mflat: uniform( 1 ), mdry: uniform( 0.8 ), mshade: uniform( 0.3 ), joint: uniform( 0.8 ),
			dir: uniform( 1 ), rscale: uniform( 1 ), nmap: uniform( 1 ), nflip: uniform( - 1 ), spec: uniform( 0.7 ),
			pane: uniform( 1 ), paneCore: uniform( 1.2 ), paneBody: uniform( 0.5 ), paneEven: uniform( 0.3 ), paneBlur: uniform( 0.25 ), paneLit: uniform( 0.3 ), paneSpec: uniform( 0.5 ), paneDirt: uniform( 0.4 ), paneCap: uniform( 12 ), paneStar: uniform( 40 ), headSoft: uniform( 0 ), fall: uniform( 1 ), fallLit: uniform( 1 ), fallCool: uniform( 0.5 ),
			fineR: uniform( 0 ) };
		{ const v = parseFloat( new URLSearchParams( location.search ).get( 'u_fineR' ) ); this._fineFix = isFinite( v ) ? v : null; } // (?u_fineR=1e6: no fade, checking)
		// (T.dir: the height in a layer's alpha in metres, per layer - the bake's `relief`)
		// (T.nr: their normal + roughness layers, one array for the library materials - a blank until _loadNR() has it)
		const nt = this.T.dir && this.T.nr && bake.texn && bake.texn[ this.T.nr ];
		let NR = this.NR = null;
		if ( nt && bake.rough && bake.metal && nt.files.some( ( f ) => f ) ) {

			const n0 = nt.files.findIndex( ( f ) => f ), n1 = nt.files.length - [ ...nt.files ].reverse().findIndex( ( f ) => f );
			const blank = new THREE.DataArrayTexture( new Uint8Array( [ 128, 128, 153, 255 ] ), 1, 1, 1 );
			blank.wrapS = blank.wrapT = THREE.RepeatWrapping; blank.magFilter = blank.minFilter = THREE.LinearFilter; blank.needsUpdate = true;
			NR = this.NR = { n0, n1, count: n1 - n0, size: nt.size, files: nt.files.slice( n0, n1 ), blank, nodes: [], on: uniform( 0 ) };

		}

		const reliefOf = this.T.dir && bake.relief ? uniformArray( bake.relief, 'float' ).element( int( layer.add( 0.5 ) ) ) : null;
		for ( const k in U ) { const v = parseFloat( new URLSearchParams( location.search ).get( 'u_' + k ) ); if ( isFinite( v ) ) U[ k ].value = v; } // ?u_glint=2 (tuning)

		const glowing = step( 0.75, aux );
		// A lit room is a glowing card a few millimetres behind its window's opening, a lantern's mantle a card inside its
		// cage. The levels below the fine one are simplified to centimetres (the lowest to 0.2 m), so the card and the wall
		// it sits in can end up in one plane or pass through each other: which of the two a pixel shows then turns with
		// every step, and the mirror image of the paving - the town at its lowest level, seen from a few metres - draws
		// that out into a flickering streak. Glowing vertices (and the glass in front of them, which must stay in front)
		// are therefore pulled towards the eye along their line of sight - the place on the screen stays, only the depth
		// moves - by PULL[ level ] = [ metres, share of the distance ]; the mirror image by PULL.mirror. What stands
		// closer than that in front of a lit card goes behind it, so only the lowest level is pulled: its glazing bars
		// and frames are shards or gone. The levels above it still have them, a centimetre or less in front of the
		// card (a pull of 6 mm took the sash bars of a window at the near level): there the packer keeps what lies at a
		// glowing face as the model has it instead (*_glow_guard of webbake_pack.mjs).
		// ?pull=lo,mid,near,fine,mirror (m), ?pullk= the same (shares of the distance).
		const PULL = this.PULL = { 0: [ 0.06, 0.0005 ], 1: [ 0, 0 ], 2: [ 0, 0 ], 3: [ 0, 0 ], mirror: [ 0.12, 0 ] };
		{

			const q = new URLSearchParams( location.search ), ks = [ 0, 1, 2, 3, 'mirror' ];
			[ 'pull', 'pullk' ].forEach( ( name, j ) => ( q.get( name ) || '' ).split( ',' ).forEach( ( t, i ) => { const v = parseFloat( t ); if ( isFinite( v ) && i < ks.length ) PULL[ ks[ i ] ][ j ] = v; } ) );

		}

		const pullAll = this.pullAll = uniform( new THREE.Vector2() ); // (set while the mirror image is drawn)
		const pullObj = uniform( new THREE.Vector3() ).onObjectUpdate( ( { object }, self ) => {

			const p = object.userData.pull;
			self.value.set( p ? p[ 0 ] : 0, p ? p[ 1 ] : 0, object.matrixWorld.elements[ 0 ] ); // (.z: metres per unit of the mesh)

		} );
		const pulled = ( which ) => {

			const to = modelWorldMatrixInverse.mul( vec4( cameraPosition, 1 ) ).xyz.sub( positionLocal );
			const d = length( to ).mul( pullObj.z );
			const t = min( max( pullObj.x, pullAll.x ).div( max( d, 1e-3 ) ).add( max( pullObj.y, pullAll.y ) ), 0.5 );
			return positionLocal.add( to.mul( which === null ? t : t.mul( which ) ) );

		};
		// the opaque surfaces. dir: the variant for the meshes whose vertices also say where their light comes from (the
		// fine level of a tier with T.dir, see _build): the relief of a wall or a cobble is lit per pixel
		const makeOpaque = ( dir ) => {

			const opaque = new THREE.MeshBasicNodeMaterial( { fog: false } );
			let col = tx.rgb.mul( lit );
			if ( this.grid && ! this.T.flat ) {

				// wet cobbles, polished stone, glazed tiles: the face normal (from the position derivatives), roughened by the
				// relief in the layer's alpha, mirrors the nearest lanterns and a little of the sky. Where the baked light says
				// "in shadow" the highlight goes too.
				const dpx = dFdx( positionWorld ), dpy = dFdy( positionWorld );
				const nr = cross( dpx, dpy );
				const N0 = nr.div( max( length( nr ), 1e-20 ) );
				const R1 = cross( dpy, N0 ), R2 = cross( N0, dpx ), det = dot( dpx, R1 );
				// (metres of relief: one guess for all layers on the lean tiers, each material's own where the light is per pixel)
				const h = tx.a.mul( reliefOf ? reliefOf.mul( 2 ).mul( U.rscale ) : U.relief );
				const nb = N0.mul( det ).mul( sign( dot( N0, V ) ) ).sub( R1.mul( dFdx( h ) ).add( R2.mul( dFdy( h ) ) ) );
				let N = nb.div( max( length( nb ), 1e-20 ) );
				const gloss = min( aux.mul( 2 ), 1 );
				let rough = mix( float( 0.5 ), float( 0.08 ), gloss );
				let f0 = float( 0.04 ), spec = null;
				if ( dir ) {

					const Nf = N0.mul( sign( dot( N0, V ) ) );
					// What only this level has - the material's normals and roughness, the light on the pixel's own normal,
					// a highlight on dull surfaces - is a matter of the pixel's distance, not of its cell's level: all of it
					// is through fading to the picture every level draws before the nearest cell that is not at this
					// level (U.fineR, _fineReach()). A cell given or refused the level then changes nothing but its mesh.
					const kd = oneMinus( smoothstep( U.fineR.mul( 0.6 ), U.fineR.add( 0.01 ), length( cameraPosition.xz.sub( positionWorld.xz ) ) ) );
					if ( NR ) {

						// The material's own normal map and roughness (NR: the n<size>_<layer> files, in after the first frame)
						// where the layer has them: the grain of a door, the pores of a brick, rust on a lamp post - and with
						// them a highlight from the lanterns on every surface, not only the glossy ones. The tangent frame comes
						// from the derivatives of the uv (no tangents in a vertex). Wet paving keeps the relief of its height:
						// the film and the mirror are built on it.
						const wetP = this.T.sheet ? smoothstep( 0.2, 0.6, gloss ).mul( smoothstep( 0.9, 0.99, Nf.y ) ).mul( oneMinus( glowing ) ) : float( 0 );
						const li = floor( layer.add( 0.5 ) );
						const has = step( NR.n0, li ).mul( oneMinus( step( NR.n1, li ) ) ).mul( NR.on ).mul( oneMinus( wetP ) ).mul( kd );
						const nq = texture( NR.blank, tiles ).depth( int( clamp( li.sub( NR.n0 ), 0, NR.count - 1 ) ) );
						NR.nodes.push( nq );
						const mx = nq.x.mul( 2 ).sub( 1 ).mul( U.nmap ), my = nq.y.mul( 2 ).sub( 1 ).mul( U.nmap ).mul( U.nflip ), mz = sqrt( max( oneMinus( mx.mul( mx ) ).sub( my.mul( my ) ), 0.05 ) );
						const du = dFdx( tiles ), dv = dFdy( tiles );
						const Tu = R1.mul( du.x ).add( R2.mul( dv.x ) ), Tv = R1.mul( du.y ).add( R2.mul( dv.y ) );
						const sc = float( 1 ).div( sqrt( max( max( dot( Tu, Tu ), dot( Tv, Tv ) ), 1e-30 ) ) );
						const nm = Tu.mul( mx.mul( sc ) ).add( Tv.mul( my.mul( sc ) ) ).add( Nf.mul( mz ) );
						const nx = mix( N, nm.div( max( length( nm ), 1e-20 ) ), has );
						N = nx.div( max( length( nx ), 1e-20 ) );
						const mean = uniformArray( bake.rough, 'float' ).element( int( li ) );
						rough = clamp( mix( rough, mix( max( mean, 0.5 ), float( 0.08 ), gloss ).add( nq.z.sub( mean ) ), has ), 0.04, 1 );
						f0 = mix( vec3( 0.04 ), tx.rgb, uniformArray( bake.metal, 'float' ).element( int( li ) ).mul( has ) );
						spec = has;

					}

					// The baked light is one colour per vertex: right for the face, blind to the relief on it. The fine level
					// brings 4 bytes more (light_dir() of the bake): g, the direction its direct light comes from over the
					// cosine it arrives at, and r, the share of the baked colour that is direct light (red, blue). The light
					// on the pixel's own normal is then the baked one x ( 1 + ( N - face normal ) . g ) for the direct share,
					// and the sky's fill for the rest, which only cares how far the normal looks up. A joint turned away
					// from the lantern goes dark and cool, the flank of a cobble towards it bright and warm.
					const ga = attribute( '_g', 'vec2' ).mul( 65535 );
					const hi = floor( ga.add( 0.5 ).div( 256 ) ), lo = ga.sub( hi.mul( 256 ) );
					const gv = varying( vec3( lo.x, hi.x, lo.y ).sub( 128 ).mul( ( bake.bake.dir_max || 4 ) / 127 ), 'vDirG' );
					const rr = floor( hi.y.add( 0.5 ).div( 16 ) );
					const rv = varying( vec2( rr, hi.y.sub( rr.mul( 16 ) ) ).div( 15 ), 'vDirR' );
					const dn = N.sub( Nf );
					const r3 = vec3( rv.x, mix( rv.x, rv.y, 0.3 ), rv.y );
					const sk = bake.look.sky, gd = bake.look.ground;
					const fill = mix( vec3( gd[ 0 ], gd[ 1 ], gd[ 2 ] ), vec3( sk[ 0 ], sk[ 1 ], sk[ 2 ] ), Nf.y.mul( 0.5 ).add( 0.5 ) );
					const ka = vec3( ( sk[ 0 ] - gd[ 0 ] ) / 2, ( sk[ 1 ] - gd[ 1 ] ) / 2, ( sk[ 2 ] - gd[ 2 ] ) / 2 ).div( fill );
					const F = clamp( dot( dn, gv ).mul( r3 ).add( dn.y.mul( ka ).mul( oneMinus( r3 ) ) ).mul( U.dir.mul( kd ) ).add( 1 ), 0, 3 );
					col = col.mul( mix( F, vec3( 1 ), glowing ) );
					this.dirNodes = { F, gv, r3, N };

				}

				const lum = dot( lit, vec3( 0.2126, 0.7152, 0.0722 ) );
				const shiny = ( spec ? max( smoothstep( 0.03, 0.3, gloss ), spec ) : smoothstep( 0.03, 0.3, gloss ) ).mul( oneMinus( glowing ) );
				const inLight = smoothstep( 0.008, 0.08, lum ); // (the highlights have no shadows: the baked light stands in)
				const gate = shiny.mul( inLight );
				let fresAt = ( nv ) => pow( oneMinus( nv ), 5 ).mul( 0.96 ).add( 0.04 );
				if ( this.T.sheet ) {

					// The town just after rain, per pixel (the bake has it per vertex, with the relief at its mean: which stone
					// is wet, how dark, how glossy on average - not where the water stands). As MaterialLib._weather: on level
					// wet paving the joints flood first (puddle), a film stands over the stones themselves (sheet), and low
					// zones a few metres across hold pools whose edge follows the relief. Standing water is one smooth surface:
					// the relief normal gives way to the face normal there, so a lantern is mirrored as one long streak across
					// many cobbles instead of a speck on each, and its roughness is the water's, not the stone's.
					// (Ng: the face as modelled; Nl: the film on it. Water lies level: on a facet within a few degrees of level
					// - the top of a modelled sett, whatever is left of it at a lower level, flat paving alike - the film's
					// normal is the vertical, so the mirror image and the streaks on a paved place are the same at every
					// level of its cell and a faceted sett is not a faceted mirror)
					const Nf0 = N0.mul( sign( dot( N0, V ) ) );
					const lvl = smoothstep( 0.95, 0.985, Nf0.y ).mul( U.level );
					const Ng = mix( Nf0, vec3( 0, 1, 0 ), lvl ).normalize();
					const wp = positionWorld, amt = bake.look.wet ?? 0.68;
					const hgt = tx.a.mul( 2 ).sub( 1 ); // (the relief height: a layer's alpha holds it in its upper half)
					const q = vec2( wp.x.add( wp.y.mul( 0.37 ) ), wp.z.add( wp.y.mul( 0.61 ) ) );
					const nLo = texture( eng.noise, q.mul( 0.11 ) ), nHi = texture( eng.noise, q.mul( 0.93 ).add( vec2( 0.31, 0.77 ) ) );
					const nP = texture( eng.noise, q.mul( 0.031 ).add( vec2( 0.53, 0.17 ) ) );
					const wetK = smoothstep( 0.2, 0.6, gloss ).mul( smoothstep( 0.9, 0.99, Nf0.y ) ).mul( oneMinus( glowing ) );
					const puddle = wetK.mul( smoothstep( 0.55, 0.2, hgt.add( ( 1 - amt ) * 0.4 ) ) );
					const sheet = wetK.mul( U.wetFlat ).mul( smoothstep( 0.25, 0.7, nLo.b.mul( 0.7 ).add( nHi.a.mul( 0.3 ) ).add( amt - 0.5 ) ).mul( 0.5 ).add( 0.5 ) );
					const zone = min( nP.r.add( nP.b.mul( 0.35 ) ).sub( 0.8 ).mul( 5 ).add( amt - 0.5 ), bake.look.pool_cap ?? 1 );
					const depth = zone.add( nLo.a.sub( 0.5 ).mul( 2.4 ) ).add( nHi.a.sub( 0.5 ).mul( 0.3 ) ).sub( 0.25 ).sub( hgt.mul( 0.8 ) );
					const water = smoothstep( 0.0, 0.5, depth ).mul( smoothstep( 0.985, 0.998, Nf0.y ) ).mul( wetK );
					const flat = max( max( puddle.mul( 0.95 ), sheet.mul( 0.9 ) ), water );
					const tt = time.mul( 0.03 );
					const r1 = texture( eng.noise, q.mul( 1.3 ).add( vec2( tt, tt.mul( 0.7 ) ) ) ).rg.sub( 0.5 );
					const r2 = texture( eng.noise, q.mul( 3.1 ).add( vec2( tt.mul( - 1.3 ), tt.mul( 0.9 ) ) ) ).ba.sub( 0.5 );
					const rip = r1.add( r2.mul( 0.5 ) ).mul( U.ripple ).mul( max( water, sheet.mul( 0.5 ) ) );
					const tip = Ng.sub( Nf0 ).mul( wetK ); // (the relief stands on the film's plane, not the facet's)
					const nw = mix( N.add( tip ).normalize(), Ng, flat ).add( vec3( rip.x, 0, rip.y ) );
					N = nw.div( max( length( nw ), 1e-20 ) );
					rough = mix( mix( mix( rough, min( rough, float( bake.look.film_rough ?? 0.17 ) ), sheet ), float( 0.08 ), puddle ), float( bake.look.puddle_rough ?? 0.09 ), water );
					// wet stone is darker where the water stands than where a cobble's crown has drained (mean kept: the
					// bake's wet darkening is the average of the two)
					col = col.mul( mix( float( 1 ), mix( U.crown, U.sunk, max( puddle, water ) ), wetK ) );
					// (steeper than Schlick, as the engine's mirror: a thin film over a rough base keeps its long streaks)
					fresAt = ( nv ) => { const s2 = oneMinus( nv.mul( nv ) ); return mix( pow( oneMinus( nv ), 5 ), s2.mul( s2 ), wetK ).mul( 0.96 ).add( 0.04 ); };
					this.wetNodes = { wetK, flat, water, puddle, sheet };
					if ( eng.mirrorOut ) {

						// the mirror image (eng.mirrorOut). Standing water mirrors along the face normal and its ripples; the wet
						// crown of a cobble along its own relief - every stone picks its own piece of the town, and a lit
						// doorway comes out as a streak of lit crowns towards the viewer. Strength: the film's fresnel at the
						// mean normal (per pixel it would stipple), less where the stone has drained, least in baked shadow
						// (what the ray cannot find comes from a probe that knows nothing of the stall overhead).
						const nm = mix( nb.div( max( length( nb ), 1e-20 ) ).add( tip ).normalize(), Ng, max( max( puddle, water ), sheet.mul( U.mflat ) ) ).add( vec3( rip.x, 0, rip.y ) );
						const Nm = nm.div( max( length( nm ), 1e-20 ) );
						const nvm = saturate( dot( Ng, V ) ), s2 = oneMinus( nvm.mul( nvm ) );
						const shade = mix( U.mshade, float( 1 ), smoothstep( 0.004, 0.05, lum ) );
						// (a joint is a crevice: seen from eye height it mirrors next to nothing unless a pool stands over it)
						const proud = mix( U.joint, float( 1 ), max( smoothstep( 0.2, 0.6, hgt ), water ) );
						opaque.mrtNode = eng.mirrorOut( s2.mul( s2 ).mul( 0.96 ).add( 0.04 ).mul( wetK ).mul( mix( U.mdry, float( 1 ), flat ) ).mul( shade ).mul( proud ).mul( oneMinus( eng.fogAmount() ) ).mul( U.mirror ), rough, Nm );

					}

				}

				// what the wet film mirrors besides the lanterns: the sky where the reflected ray points up, the lit town
				// (about as bright as the light arriving here) where it stays low
				const nv = saturate( dot( N, V ) );
				const fres = fresAt( nv );
				const ry = N.y.mul( nv ).mul( 2 ).sub( V.y );
				let env = mix( lit.mul( U.town ), SKY.mul( U.sheen ), smoothstep( 0.02, 0.45, ry ) );
				if ( eng.probe && this.wetNodes ) env = env.mul( oneMinus( this.wetNodes.wetK ) ); // (wet paving: the mirror pass has the real thing)
				// (dir: a lamp's own metalwork takes its burner's highlight whatever its baked light, see _glints)
				if ( dir ) col = col.add( this._glints( N, V, rough, f0, inLight ).mul( spec ? mix( U.glint, U.spec, spec ) : U.glint ).add( env.mul( fres ).mul( gloss ).mul( inLight ) ).mul( shiny ) );
				else col = col.add( this._glints( N, V, rough, f0 ).mul( spec ? mix( U.glint, U.spec, spec ) : U.glint ).add( env.mul( fres ).mul( gloss ) ).mul( gate ) );

			}

			opaque.colorNode = eng.knee( fogged( col ), glowing );
			if ( dir && new URLSearchParams( location.search ).get( 'show' ) === 'dir' ) opaque.colorNode = this.dirNodes.F.mul( 0.25 ); // checking
			if ( dir && new URLSearchParams( location.search ).get( 'show' ) === 'normal' ) opaque.colorNode = this.dirNodes.N.mul( 0.5 ).add( 0.5 ).mul( 0.3 ); // checking
			if ( this.wetNodes && new URLSearchParams( location.search ).get( 'show' ) === 'masks' ) opaque.colorNode = vec3( this.wetNodes.puddle, this.wetNodes.sheet, this.wetNodes.water ).mul( 0.5 ); // checking
			if ( new URLSearchParams( location.search ).get( 'show' ) === 'alpha' ) opaque.colorNode = vec3( step( 0.25, tx.a ), step( 0.5, tx.a ), step( 0.75, tx.a ) ).mul( 0.3 ); // checking
			// only the glowing vertices the packer left at 65535 are pulled - those it marked 65534 (lo_no_pull of
			// webbake_pack.mjs: a lit card with kept geometry right in front, a clock's hands) stay. A bake without
			// such marks has none at 65534: the same picture as before. ?pullsel=0: both are pulled, as it was
			opaque.positionNode = pulled( step( new URLSearchParams( location.search ).get( 'pullsel' ) === '0' ? 0.75 : 0.99999, cv.w ) );
			opaque.name = 'baked';
			if ( dir ) opaque.name = 'baked (directional)';
			return opaque;

		};

		const opaque = makeOpaque( false ), opaqueDir = this.T.dir ? makeOpaque( true ) : null;

		const glass = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false } );
		glass.colorNode = fogged( lit );
		glass.opacityNode = aux;
		glass.positionNode = pulled( null );
		glass.name = 'baked glass';

		// Lantern panes: the mantle's glare (LanternLighting of the engine). A pane that glows evenly is a paper card; a
		// lit gas lantern is white-hot where the mantle is seen through the glass and falls to amber towards the glazing
		// bars. The bake gives such a pane the vector to its burner and the burner's hue in the uv slots of its vertices
		// (pane_bytes(): glass has no texture) - at EVERY level, and the glare has no distance term: a lantern must not
		// change its face when its cell changes level or the camera comes closer (it did: the glare was the fine level's
		// alone and faded out towards its radius, so lamps "switched on" at 20 - 30 m). A tight core over a wider body
		// around the line of sight to the burner (a third of the engine's gains: the halo's hot core lies over the same
		// head, and with both whole the pane facing the camera is a white card that hides the mantle and the glazing
		// bars), the even glow turned down to the amber of the pane's edges, more cover at a glancing angle. Far away the
		// core is smaller than a pixel: both lobes are widened by the pixel's footprint at the same energy, so the lamp
		// keeps its light without twinkling.
		let glassPane = null;
		if ( this.T.dir && bake.bake && bake.bake.pane_r && bake.bake.pane_uv ) {

			glassPane = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false } );
			const ga = attribute( 'uv', 'vec2' ).mul( 65535 );
			const hi = floor( ga.add( 0.5 ).div( 256 ) ), lo = ga.sub( hi.mul( 256 ) );
			const hg = floor( hi.y.add( 0.5 ).div( 16 ) );
			const has = step( 0.5, hg );
			const Lv = varying( vec3( lo.x, hi.x, lo.y ).sub( 128 ).mul( bake.bake.pane_r / 127 ).mul( has ), 'vPaneL' ); // pixel -> burner (m)
			const hv = varying( vec3( has, hg.div( 15 ), hi.y.sub( hg.mul( 16 ) ).div( 15 ) ), 'vPaneH' ); // has a burner, its green and blue over red
			const P = U.pane, PC = U.paneCore, PB = U.paneBody, PE = U.paneEven; // (look: LANTERN of the engine's materials.js)
			const d2 = dot( Lv, Lv );
			const along = dot( Lv, V ); // (V looks back at the camera: the sign drops out in the square)
			const r2 = d2.sub( along.mul( along ) ).max( 0 ); // (distance of the burner from this pixel's line of sight)^2
			const dpx = dFdx( positionWorld ), dpy = dFdy( positionWorld );
			const nr = cross( dpx, dpy );
			const ndv = abs( dot( nr.div( max( length( nr ), 1e-20 ) ), V ) );
			const dirt = aux.sub( 0.22 ).div( 0.68 * 0.7 ).saturate(); // (the bake's cover = mix( 0.22, 0.9, dirt x 0.7 ))
			const cover = max( aux, mix( float( 0.22 ), float( 0.9 ), pow( oneMinus( ndv ), 4 ) ) );
			const k = hv.x.mul( oneMinus( smoothstep( 0.2, 1.0, d2 ) ) ).mul( P );
			const hue = vec3( 1, hv.y, hv.z );
			// (the pixel's footprint across the line of sight, m^2: what the lobes are widened by)
			const ax = dot( dpx, V ), ay = dot( dpy, V );
			const f2 = max( dot( dpx, dpx ).sub( ax.mul( ax ) ), dot( dpy, dpy ).sub( ay.mul( ay ) ) ).max( 0 ).mul( U.paneBlur );
			const sc = 0.04 * 0.04, sb = 0.085 * 0.085;
			const core = exp( r2.div( f2.add( sc ).mul( - 2 ) ) ).mul( float( sc ).div( f2.add( sc ) ) );
			const body = pow( r2.div( f2.mul( 2 ).add( sb ) ).add( 1 ), - 1.5 ).mul( float( sb ).div( f2.mul( 2 ).add( sb ) ) );
			const glare = mix( hue, vec3( 1 ), 0.45 ).mul( core.mul( PC ) ).add( hue.mul( vec3( 1.0, 0.8, 0.52 ) ).mul( body.mul( PB ) ).mul( dirt.mul( 0.6 ).add( 0.8 ) ) );
			// (the baked pane is the old even glow over the bake's cover: here the engine's amber share of it over this pixel's cover)
			const even = vec3( 1.0, 0.82, 0.65 ).mul( PE ).mul( dirt.mul( 0.6 ).add( 0.8 ).add( ndv.mul( 0.5 ) ).div( dirt.mul( 0.6 ).add( 1.05 ) ) ).mul( aux.div( cover ) );
			// Clear glass around a burner (the panes of a gas lamp's head: clearPanes() of the packer gives them the same
			// bytes, and the peak of the burner's light in the layer slot, which glass has no use for). The engine lights
			// such a pane as any surface: the dirt on it takes the burner's light and the glass mirrors the burner - seen
			// from outside that is the inside of the far panes, a frosted amber wall behind the mantle with the burner's
			// highlight in it, while the pane in front stays clear. The same here per pixel (the bake lights a pane from
			// its front alone, which the burner is behind): the light of that one burner on the face normal turned to
			// the eye, its soft near field and its cap as in the bake, the dirt of the bake's cover with the grain a
			// vertex cannot hold, and the highlight no narrower than the pixel.
			const pk = varying( pl.w.mul( 65535 / 64 ), 'vPanePk' );
			const clear = step( 0.25, pk ).mul( hv.x );
			const Nn = nr.div( max( length( nr ), 1e-20 ) ), Ng = Nn.mul( sign( dot( Nn, V ) ) );
			const dl = sqrt( max( d2, 1e-4 ) ), Ld = Lv.div( dl );
			const capL = mix( float( 1 ), float( bake.look.cap_up ?? 1 ), smoothstep( - 0.42, 0.3, Ld.y.negate() ) );
			const rad = hue.mul( pk ).mul( capL ).div( max( d2, 0.01 ).add( U.headSoft.mul( ( bake.look.source_radius ?? 0.6 ) ** 2 ) ) );
			const wq = positionWorld;
			const grain = texture( eng.noise, vec2( wq.x.add( wq.z.mul( 0.73 ) ), wq.y.add( wq.z.mul( 0.41 ) ) ).mul( 9 ) );
			// (U.paneDirt: the least of it - the bake's dirt on these panes is a tenth of what the engine's texture gives them)
			const dirtC = max( aux.sub( 0.16 ).div( 0.74 * 0.7 ), U.paneDirt ).add( grain.r.sub( 0.5 ).mul( 0.15 ) ).add( grain.b.sub( 0.5 ).mul( 0.1 ) ).saturate();
			const roughC = max( mix( float( 0.05 ), float( 0.55 ), dirtC ), sqrt( f2 ).div( dl ).mul( 2 ) ).clamp( 0.12, 1 );
			const shine = this._brdf( Ng, V, Ld, roughC, float( 0.04 ) );
			// (the highlight is held to U.paneCap: a hand from the mantle the bare figure runs into the thousands, and the
			// bloom would take a whole facade with it - the mirror image of a mantle is a small thing)
			// bloom would take a whole facade with it - the mirror image of a mantle is a small thing. That image itself
			// is the second, narrow lobe off the glass under the dirt: the star at the mantle, held to U.paneStar)
			const W3 = vec3( 0.2126, 0.7152, 0.0722 );
			const hot = rad.mul( shine.mul( U.paneSpec ) ), hotL = dot( hot, W3 );
			const pin = rad.mul( this._brdf( Ng, V, Ld, sqrt( f2 ).div( dl ).mul( 2 ).clamp( 0.1, 1 ), float( 0.04 ) ).mul( U.paneSpec ) ), pinL = dot( pin, W3 );
			const lamp = rad.mul( vec3( 0.12, 0.11, 0.09 ).mul( dirtC ).mul( saturate( dot( Ng, Ld ) ) ).mul( 1 / Math.PI ).mul( U.paneLit ) ).add( hot.mul( U.paneCap.div( U.paneCap.add( hotL ).max( 1e-6 ) ) ) ).add( pin.mul( U.paneStar.div( U.paneStar.add( pinL ).max( 1e-6 ) ) ) );
			const coverC = max( aux, mix( float( 0.16 ), float( 0.9 ), max( pow( oneMinus( ndv ), 4 ), dirtC.mul( 0.7 ) ) ) );
			const kc = clear.mul( oneMinus( smoothstep( 0.2, 1.0, d2 ) ) ).mul( P );
			const sheet = lit.mul( mix( vec3( 1 ), even, k ) ).add( glare.mul( k ).div( cover ) );
			glassPane.colorNode = fogged( mix( sheet, lit.add( lamp.mul( kc ) ), step( 0.25, pk ) ) );
			glassPane.opacityNode = mix( mix( aux, cover, k ), mix( aux, coverC, kc ), step( 0.25, pk ) );
			if ( bake.pack && bake.pack.clear_panes ) {

				// (the far panes are seen from inside: both faces are drawn, in one pass - of anything but such a pane the
				// back stays out, as before)
				glassPane.side = THREE.DoubleSide; glassPane.forceSinglePass = true;
				glassPane.opacityNode = glassPane.opacityNode.mul( max( float( frontFacing ), step( 0.25, pk ) ) );

			}

			glassPane.positionNode = pulled( null );
			glassPane.name = 'baked glass (lantern glare)';
			if ( new URLSearchParams( location.search ).get( 'show' ) === 'pane' ) { glassPane.colorNode = vec3( hv.x, oneMinus( smoothstep( 0.2, 1.0, d2 ) ), sqrt( d2 ) ).mul( 0.5 ); glassPane.opacityNode = float( 1 ); } // checking

		}

		// canal / basin water at night: a dark body that mirrors. Steep reflected rays find the navy sky, low ones the
		// town (dark, warm); the lanterns of the cell show twice - as the sharp glint of the flame, torn into glitter by
		// the ripples, and as the broad warm patch of the wall they light. One slow swell and a finer ripple scroll
		// across it. No pass, no target: three noise taps and the light grid the wet ground already reads.
		const water = new THREE.MeshBasicNodeMaterial( { fog: false } );
		{

			const t = time.mul( 0.05 );
			const wp = positionWorld.xz;
			const n1 = texture( eng.noise, wp.mul( 0.33 ).add( vec2( t, t.mul( 0.6 ) ) ) ).rg.sub( 0.5 );
			const n2 = texture( eng.noise, wp.mul( 1.1 ).add( vec2( t.mul( - 0.7 ), t.mul( 1.1 ) ) ) ).ba.sub( 0.5 );
			const n3 = texture( eng.noise, wp.mul( 4.3 ).add( vec2( t.mul( 2.1 ), t.mul( - 1.6 ) ) ) ).rg.sub( 0.5 );
			const swell = n1.add( n2.mul( 0.45 ) ).mul( U.swell );
			const rip = swell.add( n3.mul( U.swell ).mul( 0.6 ) );
			const Ns = normalize( vec3( swell.x, 1, swell.y ) ), N = normalize( vec3( rip.x, 1, rip.y ) );
			const nv = saturate( dot( N, V ) );
			const fres = pow( oneMinus( nv ), 5 ).mul( 0.98 ).add( 0.02 );
			const ry = N.y.mul( nv ).mul( 2 ).sub( V.y ); // height of the reflected ray
			const env = mix( vec3( 0.028, 0.019, 0.012 ).mul( U.town ).mul( 2.5 ), SKY.mul( U.sheen ), smoothstep( 0.1, 0.5, ry ) );
			let c = lit.mul( 0.3 ).add( env.mul( fres ) );
			if ( this.grid ) c = c.add( this._glints( N, V, float( 0.07 ), float( 0.02 ) ).mul( U.glint ) ).add( this._glints( Ns, V, float( 0.4 ), float( 0.02 ) ).mul( U.wglow ) );
			water.colorNode = fogged( c );

		}

		water.name = 'baked water';

		// falling water: the engine's torn sheet (strands scrolling down V), lit by the baked colour
		const fall = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false, side: THREE.DoubleSide } );
		{

			// (the engine's water_fall: a fluted sheet at the lip that thickens into ropes, tears open between them and ends
			// in white threads and drops. Its two masks - break-up and foam, _mask.z / .w of the model - ride in the layer
			// slot of a fall's vertices, 4 bits each: without them every jet is an unbroken thin sheet, next to invisible)
			const lw = pl.w.mul( 65535 ).add( 0.5 ), wq = floor( lw.div( 16 ) );
			const fm = varying( vec2( floor( lw.sub( wq.mul( 16 ) ) ), wq ).div( 15 ), 'vFall' );
			const brk = fm.x.saturate(), white = fm.y.saturate();
			const t = time.mul( 1.6 );
			const u0 = tiles.x, v = tiles.y.negate();
			// strands wander sideways as they fall, more once they have torn free
			const u = u0.add( texture( eng.noise, vec2( u0.mul( 0.9 ), v.mul( 0.31 ).add( t.mul( 0.21 ) ) ) ).a.sub( 0.5 ).mul( brk.mul( 0.05 ).add( 0.012 ) ) );
			const s1 = texture( eng.noise, vec2( u.mul( 3.1 ), v.mul( 0.23 ).add( t.mul( 0.42 ) ) ) ); // flutes of the sheet
			const s2 = texture( eng.noise, vec2( u.mul( 8.3 ).add( 0.37 ), v.mul( 0.5 ).add( t.mul( 0.9 ) ) ) ); // threads
			const s3 = texture( eng.noise, vec2( u.mul( 1.3 ).add( 0.71 ), v.mul( 0.62 ).add( t.mul( 1.1 ) ) ) ); // surges travelling down
			const body = s1.r.mul( brk.mul( - 0.2 ).add( 0.47 ) ).add( s2.g.mul( 0.25 ) ).add( s3.b.mul( brk.mul( 0.2 ).add( 0.28 ) ) );
			const thr = mix( float( 0.16 ), float( 0.61 ), brk.mul( brk.mul( 0.35 ).add( 0.65 ) ) ).sub( white.mul( 0.14 ) );
			const a = smoothstep( thr, thr.add( 0.06 ), body );
			const rim = a.mul( oneMinus( smoothstep( thr.add( 0.05 ), thr.add( 0.16 ), body ) ) ); // the rolled edge of a rope
			const fib = smoothstep( 0.42, 0.7, s2.g.mul( 0.55 ).add( s1.b.mul( 0.45 ) ) );
			const air = brk.mul( brk.mul( 0.5 ).add( 0.5 ) );
			const foam = white.add( fib.mul( 0.26 ).add( 0.06 ) ).add( air.mul( fib.mul( 0.4 ).add( 0.36 ) ) ).add( rim.mul( brk ).mul( 0.5 ) ).saturate();
			const bead = smoothstep( 0.8, 0.92, s2.b.mul( 0.6 ).add( s1.a.mul( 0.4 ) ) ).mul( a ).mul( brk.mul( 0.8 ).add( 0.2 ) );
			// clear water has no diffuse response: only the silvering and the foam are lit like a surface
			// (the engine draws the sheet at full cover and lets the frame behind through itself: its silvering and foam
			// arrive whole. Here the sheet is a veil blended over the frame, so they are taken over the veil's share -
			// times the veil they were a third of the engine's on the clear film, which is most of a sheet's width)
			const veil = mix( float( 0.22 ), float( 0.95 ), max( foam, bead ) );
			// (U.fallCool: the engine's sheet is mostly the grey square seen through it plus the sky's share of its
			// mirror, a cool silver; the baked colour alone is all lamp)
			// (both on the tiers with T.dir only: Low and Phone keep the fall they have)
			const flit = mix( lit, vec3( 0.81, 0.88, 0.9 ).mul( dot( lit, vec3( 0.2126, 0.7152, 0.0722 ) ) ), U.fallCool );
			let c = this.T.dir ? flit.mul( foam.mul( 0.8 ).add( 0.03 ) ).mul( U.fallLit ).add( vec3( 0.62, 0.74, 0.79 ).mul( bead ).mul( 0.3 ) ).div( veil )
				: lit.mul( foam.mul( 0.8 ).add( 0.03 ) ).add( vec3( 0.62, 0.74, 0.79 ).mul( bead ).mul( 0.3 ) );
			const op = a.mul( veil ).mul( aux );
			if ( this.grid && this.T.dir ) {

				// the lanterns run as thin bright lines down the flutes and ropes: the relief of the water's thickness
				// (metres, from screen derivatives) turns the face normal, the light grid the wet paving reads does the rest
				const dpx = dFdx( positionWorld ), dpy = dFdy( positionWorld );
				const nr = cross( dpx, dpy );
				const N0 = nr.div( max( length( nr ), 1e-20 ) );
				const R1 = cross( dpy, N0 ), R2 = cross( N0, dpx ), det = dot( dpx, R1 );
				const hgt = body.mul( 1.6 ).add( smoothstep( thr.sub( 0.02 ), thr.add( 0.2 ), body ) ).mul( 0.012 );
				const nb = N0.mul( det ).mul( sign( dot( N0, V ) ) ).sub( R1.mul( dFdx( hgt ) ).add( R2.mul( dFdy( hgt ) ) ) );
				const sp = this._glints( nb.div( max( length( nb ), 1e-20 ) ), V, mix( float( 0.08 ), float( 0.6 ), foam ), float( 0.02 ) ).mul( U.fall );
				c = c.add( sp.mul( a ).div( max( op, 0.05 ) ) ); // (blended: over its cover, so the line arrives whole)

			}

			fall.colorNode = fogged( c );
			fall.opacityNode = op;

		}

		fall.name = 'baked fall';

		// leaf cards (the trees' foliage): the uv counts cards, aux scales it to texture tiles. As in the engine's foliage
		// material the relief height of the leaf mosaic cuts the card into leaves, and the cut rises towards the card's
		// edge - a ragged spray, never a straight edge.
		const cut = new THREE.MeshBasicNodeMaterial( { fog: false, side: THREE.DoubleSide } );
		{

			const t = sample( tiles.mul( aux.mul( bake.cut_aux || 4 ) ) );
			const cell = fract( tiles ).sub( 0.5 ).mul( 2 );
			const h = t.a.mul( 2 ).sub( 1 ); // the layer keeps the height in the upper half of alpha
			cut.maskNode = h.greaterThan( smoothstep( 0.45, 1.0, length( cell ) ).mul( 0.8 ).add( 0.33 ) );
			cut.colorNode = eng.knee( fogged( t.rgb.mul( lit ) ), float( 0 ) );

		}

		cut.name = 'baked leaves';
		this.mats = { [ K_OPAQUE ]: opaque, [ K_GLASS ]: glassPane || glass, [ K_WATER ]: water, [ K_FALL ]: fall, [ K_CUT ]: cut };
		this.matsDir = opaqueDir ? { ...this.mats, [ K_OPAQUE ]: opaqueDir } : null;
		this.fmats = this.fmatsDir = null;
		if ( this.fadeTime > 0 ) {

			const fading = ( m ) => {

				const f = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: m.transparent, depthWrite: m.depthWrite, side: m.side } );
				f.colorNode = m.colorNode; f.opacityNode = m.opacityNode; f.mrtNode = m.mrtNode; f.positionNode = m.positionNode;
				f.maskNode = m.maskNode ? m.maskNode.and( fadeMask ) : fadeMask;
				f.forceSinglePass = m.forceSinglePass;
				f.name = m.name + ' (fading)';
				return f;

			};

			this.fmats = {};
			for ( const k in this.mats ) this.fmats[ k ] = fading( this.mats[ k ] );
			if ( opaqueDir ) this.fmatsDir = { ...this.fmats, [ K_OPAQUE ]: fading( opaqueDir ) };

		}

	}

	// lantern glows and flames: one instanced sprite draw each
	_sprites() {

		const eng = this.eng, bake = this.bake;
		const inst = ( rows, from, n ) => {

			const a = new Float32Array( rows.length * n );
			rows.forEach( ( r, i ) => {

				for ( let k = 0; k < n; k ++ ) a[ i * n + k ] = r[ from + k ];

			} );
			return instancedBufferAttribute( new THREE.InstancedBufferAttribute( a, n ) );

		};

		if ( bake.halos.length ) {

			const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
			const H = bake.halos;
			// (a row's 9th number: the glow is drawn that far towards the eye - the burner sits inside an opaque glowing
			// body, an opal globe, which would hide the hot centre and leave a lamp that looks switched off)
			const P = inst( H, 0, 3 );
			m.positionNode = H[ 0 ].length > 8 ? P.add( normalize( cameraPosition.sub( P ) ).mul( inst( H, 8, 1 ) ) ) : P;
			m.scaleNode = inst( H, 6, 1 );
			const r = length( uv().sub( 0.5 ) ).mul( 2 );
			const a = exp( r.mul( r ).mul( - 5 ) ).sub( 0.0067 ).max( 0 );
			const hs = varying( inst( H, 6, 1 ) );
			// same hot core as engine.js createHalo; a row's 8th number is 0 for a helper light with no burner to look at (no core)
			const core = exp( r.mul( r ).mul( hs.mul( hs ).mul( - 11 ) ) ).mul( H[ 0 ].length > 7 ? varying( inst( H, 7, 1 ) ) : 1 );
			m.colorNode = varying( inst( H, 3, 3 ) ).mul( a.add( core.mul( haloCore ) ) ).mul( this.T.halo || 0.1 );
			m.opacityNode = a.add( core.mul( haloCore.min( 1 ) ) ).min( 1 );
			const s = new THREE.Sprite( m );
			s.count = H.length;
			s.frustumCulled = false;
			s.renderOrder = 4;
			s.name = 'halos';
			this.group.add( s );
			this.halos = s;

		}

		if ( bake.flames.length ) {

			const F = bake.flames.map( ( f, i ) => [ ...f, ( i * 0.61803 ) % 1, f[ 6 ] * 0.8, f[ 6 ] * 1.7 ] ); // + phase, sprite size
			const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
			m.positionNode = inst( F, 0, 3 );
			m.scaleNode = inst( F, 10, 2 );
			const col = varying( inst( F, 3, 3 ) ), gs = varying( inst( F, 7, 3 ) ); // glow, speed, phase
			const glow = gs.x, phase = gs.z;
			const p = uv();
			const y = p.y, x = p.x.sub( 0.5 ).mul( 2 );
			const t = time.mul( gs.y ).add( phase.mul( 7.3 ) );
			const n1 = texture( eng.noise, vec2( x.mul( 0.12 ).add( phase ), y.mul( 0.3 ).sub( t.mul( 0.5 ) ) ) ).r.sub( 0.5 );
			const n2 = texture( eng.noise, vec2( x.mul( 0.35 ).add( phase.mul( 2.1 ) ), y.mul( 0.8 ).sub( t.mul( 1.05 ) ) ) ).g.sub( 0.5 );
			const sway = sin( t.mul( 2.1 ) ).mul( 0.06 ).add( n1.mul( 0.55 ) ).add( n2.mul( 0.22 ) );
			const xx = x.add( sway.mul( y ) );
			const hgt = y.div( n1.mul( 0.25 ).add( 0.9 ) );
			const w = min( pow( max( hgt, 0 ).div( 0.22 ), 0.55 ), pow( max( oneMinus( hgt ), 0 ).div( 0.78 ), 0.85 ) ).mul( 0.42 );
			const dd = abs( xx ).div( max( w, 0.001 ) );
			const body = smoothstep( 1.0, 0.25, dd ).mul( step( hgt, 1 ) );
			const core = smoothstep( 0.7, 0.0, dd ).mul( smoothstep( 0.95, 0.25, hgt ) );
			const outer = col.mul( vec3( 1, 0.55, 0.25 ) );
			let c = mix( outer, vec3( 1.0, 0.86, 0.6 ), core );
			c = mix( vec3( 0.06, 0.125, 0.45 ), c, smoothstep( 0.02, 0.2, hgt ) );
			m.colorNode = c.mul( body ).mul( glow );
			m.opacityNode = body;
			const s = new THREE.Sprite( m );
			s.count = F.length;
			s.center.set( 0.5, 0.02 );
			s.frustumCulled = false;
			s.renderOrder = 5;
			s.name = 'flames';
			this.group.add( s );
			this.flames = s;

		}

	}

	// one chunk level: meshopt streams -> a mesh drawing straight from the quantized vertex records
	// (`decoded`: the two streams already unpacked by the decoder's workers, see _fetch)
	_build( bytes, m, decoded = null, level = 0 ) {

		const W = m.w || 10; // u16 per vertex: 12 in a level that also has the light's direction (the bake's light_dir())
		const dir = W >= 12 && !! this.matsDir, M = dir ? this.matsDir : this.mats, FM = dir ? this.fmatsDir : this.fmats;
		const vb = decoded ? decoded[ 0 ] : new Uint8Array( m.nv * W * 2 );
		if ( ! decoded ) MeshoptDecoder.decodeVertexBuffer( vb, m.nv, W * 2, bytes.subarray( 0, m.vlen ) );
		const ib = decoded ? decoded[ 1 ] : new Uint8Array( m.ni * m.it );
		if ( ! decoded ) MeshoptDecoder.decodeIndexBuffer( ib, m.ni, m.it, bytes.subarray( m.vlen ) );
		if ( m.lmPage ) {

			// (lightmap prototype: the vertices with their light in a page - bit 15 of the layer - say which page of their
			// bundle; the bundles share one array)
			const a = new Uint16Array( vb.buffer, vb.byteOffset, m.nv * W );
			for ( let i = 0; i < a.length; i += W ) if ( a[ i + 3 ] & 0x8000 ) a[ i + 8 ] += m.lmPage * 256;

		}

		const buf = new THREE.InterleavedBuffer( new Uint16Array( vb.buffer ), W );
		const g = new THREE.BufferGeometry();
		g.setAttribute( 'position', new THREE.InterleavedBufferAttribute( buf, 4, 0, true ) );
		g.setAttribute( '_pl', new THREE.InterleavedBufferAttribute( buf, 4, 0, true ) ); // .w = texture layer
		g.setAttribute( 'uv', new THREE.InterleavedBufferAttribute( buf, 2, 4, true ) );
		g.setAttribute( '_col', new THREE.InterleavedBufferAttribute( buf, 4, 6, true ) );
		if ( dir ) g.setAttribute( '_g', new THREE.InterleavedBufferAttribute( buf, 2, 10, true ) );
		// (a 16-bit index is marked normalized: the WebGPU backend widens plain 16-bit integer attributes to 32 bits when
		// it uploads them, and for an index that is GPU memory for nothing; as it is, it goes up and is drawn as 16-bit)
		g.setIndex( new THREE.BufferAttribute( m.it === 2 ? new Uint16Array( ib.buffer ) : new Uint32Array( ib.buffer ), 1, m.it === 2 ) );
		const s = m.q * 65535;
		g.boundingBox = new THREE.Box3(
			new THREE.Vector3( ( m.bmin[ 0 ] - m.min[ 0 ] ) / s, ( m.bmin[ 1 ] - m.min[ 1 ] ) / s, ( m.bmin[ 2 ] - m.min[ 2 ] ) / s ),
			new THREE.Vector3( ( m.bmax[ 0 ] - m.min[ 0 ] ) / s, ( m.bmax[ 1 ] - m.min[ 1 ] ) / s, ( m.bmax[ 2 ] - m.min[ 2 ] ) / s ) );
		g.boundingSphere = g.boundingBox.getBoundingSphere( new THREE.Sphere() );
		const place = ( o ) => {

			o.position.fromArray( m.min );
			o.scale.setScalar( s );
			o.matrixAutoUpdate = false;
			o.updateMatrix();
			o.userData.tris = m.ni / 3;
			return o;

		};

		// the arrays are only needed until the first draw has put them on the GPU
		// (the index keeps its array type: the back-ends read the index format from it)
		const meshes = [];
		const free = () => {

			buf.array = EMPTY16;
			g.index.array = new g.index.array.constructor( 0 );
			for ( const o of meshes ) o.onAfterRender = THREE.Object3D.prototype.onAfterRender;

		};

		const mats = [], fmats = [], own = new Map( ( m.parts || [] ).map( ( q, i ) => [ q[ 0 ], i ] ) );
		const parts = [];
		m.sub.forEach( ( [ kind, start, count ], i ) => {

			if ( ! own.has( i ) ) { g.addGroup( start, count, mats.length ); mats.push( M[ kind ] ); if ( FM ) fmats.push( FM[ kind ] ); return; }
			// a cluster of the opaque range: the same buffers, its own range, bounds and so its own culling
			const b = m.parts[ own.get( i ) ][ 2 ], q = new THREE.BufferGeometry();
			for ( const name in g.attributes ) q.setAttribute( name, g.attributes[ name ] );
			q.setIndex( g.index );
			q.setDrawRange( start, count );
			q.boundingBox = new THREE.Box3(
				new THREE.Vector3( ( b[ 0 ] - m.min[ 0 ] ) / s, ( b[ 1 ] - m.min[ 1 ] ) / s, ( b[ 2 ] - m.min[ 2 ] ) / s ),
				new THREE.Vector3( ( b[ 3 ] - m.min[ 0 ] ) / s, ( b[ 4 ] - m.min[ 1 ] ) / s, ( b[ 5 ] - m.min[ 2 ] ) / s ) );
			q.boundingSphere = q.boundingBox.getBoundingSphere( new THREE.Sphere() );
			const part = new THREE.Mesh( q, M[ kind ] );
			part.matrixAutoUpdate = false;
			part.userData.start = start;
			part.userData.wb = b;
			if ( FM ) Object.assign( part.userData, { mat: part.material, fmat: FM[ kind ] } );
			parts.push( part );

		} );
		let mesh = mats.length ? new THREE.Mesh( g, mats.length === 1 && ! parts.length ? mats[ 0 ] : mats ) : null; // (one material: the groups would not count)
		// (what it is drawn with while a change of level fades, and otherwise)
		if ( mesh && this.fmats ) Object.assign( mesh.userData, { mat: mesh.material, fmat: Array.isArray( mesh.material ) ? fmats : fmats[ 0 ] } );
		if ( parts.length ) {

			const all = new THREE.Group();
			if ( mesh ) { mesh.matrixAutoUpdate = false; all.add( mesh ); }
			all.add( ...parts );
			all.userData.parts = parts;
			meshes.push( ...all.children );
			mesh = all;

		} else {

			meshes.push( mesh );

		}

		for ( const o of meshes ) { o.onAfterRender = free; o.userData.pull = this.PULL[ level ] || null; if ( ! o.userData.wb ) o.userData.wb = [ ...m.bmin, ...m.bmax ]; } // (see PULL; wb: see _cull)
		mesh.userData.draws = this.fmats ? meshes : null; // (the meshes a fade switches)
		mesh.userData.all = meshes;
		return place( mesh );

	}

	// The renderer culls a mesh by its sphere, and the sphere around the box of a cell - or of one cluster of a cell -
	// is a loose fit: 16 m of facade under a tower has a sphere of 20 m and more, which a view from next to it never
	// misses, whichever way it looks. So for the frame's own pass each mesh is tested here first, as its box (and
	// CULL_PAD around it, for what sways or is pulled forward) against the four sides of the view, and what is
	// outside is left out by its layers; the budget goes on counting by the spheres, so what is drawn can only be
	// less than what it allows. (The mirror image is another view: it keeps the spheres, see probe.swap.)
	_cull( camera ) {

		camera.updateMatrixWorld();
		const e = this._pv.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse ).elements, P = this._planes;
		// (clip x = row 0 . p, y = row 1 . p, w = row 3 . p, and inside is - w <= x, y <= w: the depth convention does not come into it)
		for ( let k = 0; k < 4; k ++ ) for ( let i = 0; i < 4; i ++ ) P[ k * 4 + i ] = e[ i * 4 + 3 ] + ( k & 1 ? - 1 : 1 ) * e[ i * 4 + ( k >> 1 ) ];
		for ( let k = 0; k < 16; k += 4 ) P[ k + 3 ] += CULL_PAD * Math.hypot( P[ k ], P[ k + 1 ], P[ k + 2 ] );
		for ( const c of this.chunks ) for ( const mesh of c.mesh ) {

			if ( ! mesh || ! mesh.visible ) continue;
			for ( const o of mesh.userData.all ) {

				const b = o.userData.wb;
				let inside = 1;
				for ( let k = 0; k < 16 && inside; k += 4 )
					if ( P[ k ] * ( P[ k ] > 0 ? b[ 3 ] : b[ 0 ] ) + P[ k + 1 ] * ( P[ k + 1 ] > 0 ? b[ 4 ] : b[ 1 ] ) + P[ k + 2 ] * ( P[ k + 2 ] > 0 ? b[ 5 ] : b[ 2 ] ) + P[ k + 3 ] < 0 ) inside = 0;
				o.layers.mask = inside;

			}

		}

	}

	_drop( c, l ) {

		const mesh = c.mesh[ l ];
		if ( ! mesh ) return;
		this.group.remove( mesh );
		// three's WebGL back-end (r186) never deletes a vertex array object, and a deleted buffer lives on for as long as one
		// still points at it: a walk would leave every chunk it ever showed on the GPU. So the ones over this mesh's buffers
		// are deleted here.
		const b = this.eng.renderer.backend, gl = b.isWebGLBackend && b.vaoCache ? b.gl : null, ids = new Set();
		if ( gl ) mesh.traverse( ( o ) => {

			if ( ! o.geometry ) return;
			for ( const k in o.geometry.attributes ) { const id = b.get( o.geometry.attributes[ k ] ).id; if ( id !== undefined ) ids.add( String( id ) ); }

		} );
		mesh.traverse( ( o ) => o.geometry && o.geometry.dispose() );
		if ( ids.size ) for ( const key of Object.keys( b.vaoCache ) ) {

			if ( ! key.split( ':' ).some( ( id ) => ids.has( id ) ) ) continue;
			const vao = b.vaoCache[ key ];
			if ( b.state.currentVAO === vao ) b.state.resetVertexState();
			gl.deleteVertexArray( vao );
			delete b.vaoCache[ key ];

		}
		c.mesh[ l ] = null;

	}

	_wake() {

		const w = this.waiters; this.waiters = [];
		for ( const f of w ) f();

	}

	async _fetch( c, l ) {

		const m = c.m[ this.L[ l ] ], key = m.file;
		try {

			// (the level's packed bytes out of its .glb where the bake has it; null: the packed file then)
			let bytes = glbOn( this ) && ! m.url ? await glbLevel( this, c.m.k, this.L[ l ], m ) : null;
			if ( ! bytes ) {

				const r = await fetch( m.url || `${this.bake.dir}/${m.file}?v=${this.bake.baked}` );
				if ( ! r.ok ) throw new Error( `${m.file}: ${r.status}` );
				bytes = new Uint8Array( await r.arrayBuffer() );
				this.bytes += bytes.length;
				if ( bytes.length !== m.len ) throw new Error( `${m.file}: size differs from the manifest (stale bake?)` );

			}

			// (unpacked off the main thread, see `workers`: a fine cell is some 8 MB of vertices, a visible hitch on a slow CPU)
			const decoded = this.T.workers && c.want === l ? await Promise.all( [
				MeshoptDecoder.decodeGltfBufferAsync( m.nv, ( m.w || 10 ) * 2, bytes.subarray( 0, m.vlen ), 'ATTRIBUTES' ),
				MeshoptDecoder.decodeGltfBufferAsync( m.ni, m.it, bytes.subarray( m.vlen ), 'TRIANGLES' ) ] ) : null;
			if ( c.want === l && ! c.mesh[ l ] ) {

				c.mesh[ l ] = this._build( bytes, m, decoded, l );
				c.mesh[ l ].visible = false;
				this.group.add( c.mesh[ l ] );

			}

		} catch ( e ) {

			c.failed = ( c.failed || 0 ) | ( 1 << l );
			const w = 'baked town: ' + ( e.message || e );
			if ( ! this.warnings.includes( w ) ) this.warnings.push( w );

		} finally {

			this.inflight.delete( key );
			this._dirty = true;
			this._wake();

		}

	}

	// the triangle budget on everything shown, nearest first: the mid level for every cell that asks for more than
	// lo, then near for those that asked for it (one near cell costs several mid ones - coverage comes before detail)
	_budget( order ) {

		const T = this.T, L = this.L;
		let tris = this.loTris;
		for ( const c of order ) {

			const n = c.m.mid ? ( c.m.mid.ni - c.m.lo.ni ) / 3 : 0;
			c.cost = 0;
			if ( c.m.mid && tris + n <= T.tris ) { tris += n; c.cost = c.m.mid.ni / 3; }
			else if ( c.m.mid || c.want < 2 ) c.want = 0;

		}

		for ( const c of order ) {

			if ( c.want < 2 ) continue;
			const n = c.m[ L[ 2 ] ].ni / 3 - ( c.cost || c.m.lo.ni / 3 );
			if ( tris + n <= T.tris ) tris += n;
			else c.want = c.cost > 0 ? 1 : 0;

		}

	}

	// index count of a cluster drawn down to parts of size s (m): its range is in order of part size, with the count
	// down to each size in the manifest
	_cut( steps, s ) {

		for ( const [ size, n ] of steps ) if ( size <= s ) return n;
		return steps[ steps.length - 1 ][ 1 ];

	}

	// the views a budget is kept for: 64 headings x the pitches of VIEW_PITCH from where the camera stands.
	// -> S (how many), arc( sphere, out ) (which of them have a mesh with that bounding sphere in them), work arrays
	_views( camera ) {

		const H = 64, NP = VIEW_PITCH.length, S = H * NP, x = camera.position.x, y = camera.position.y, z = camera.position.z;
		const step = 2 * Math.PI / H, hs = step / 2, th0 = Math.tan( camera.fov * Math.PI / 360 );
		const { load, base, tmp, B } = this._heads || ( this._heads = { load: new Float64Array( S ), base: new Float64Array( S ), tmp: new Float32Array( S ), B: new Float32Array( H ) } );
		// the views (of S) that have a sphere [ x, z, radius, , height ] in them (wide: of a view that many times the camera's)
		const arcOf = ( wide ) => {

		const th = th0 * wide, tw = th * camera.aspect, sw = Math.sqrt( 1 + tw * tw ), sh = Math.sqrt( 1 + th * th );
		return ( s, out ) => {

			const dx = s[ 0 ] - x, dz = s[ 1 ] - z, d = Math.max( Math.hypot( dx, dz ), 1e-3 ), r = s[ 2 ], a = Math.atan2( dx, dz );
			for ( let h = 0; h < H; h ++ ) {

				const b = Math.abs( h * step - a ) % ( 2 * Math.PI );
				B[ h ] = b > Math.PI ? 2 * Math.PI - b : b; // how far heading h is off the bearing of the sphere

			}

			for ( let k = 0; k < NP; k ++ ) {

				// (looking down is looking up with the heights mirrored)
				const p = Math.abs( VIEW_PITCH[ k ] ), sp = Math.sin( p ), cp = Math.cos( p ), dy = VIEW_PITCH[ k ] < 0 ? y - s[ 4 ] : s[ 4 ] - y;
				// side planes: d ( sin b - A cos b ) <= r sw + tw dy sp, A = tw cos p
				const A = tw * cp, phi = Math.atan( A ), rho = ( r * sw + tw * dy * sp ) / ( d * Math.sqrt( 1 + A * A ) );
				let w1 = 4, w2 = 0; // in view up to w1 off the bearing, and again from w2 on (around the back)
				if ( rho <= - 1 ) { w1 = - 1; w2 = 4; } else if ( rho < 1 ) { const t = Math.asin( rho ); w1 = phi + t; w2 = Math.PI + phi - t; }
				// near plane: d cos b cp + dy sp >= - r
				const cn = - ( r + dy * sp ) / ( d * cp ), wn = cn >= 1 ? - 1 : cn <= - 1 ? 4 : Math.acos( cn );
				// upper plane: d cos b ( sp + th cp ) >= dy ( cp - th sp ) - r sh
				const ct = ( dy * ( cp - th * sp ) - r * sh ) / ( d * ( sp + th * cp ) ), wt = ct >= 1 ? - 1 : ct <= - 1 ? 4 : Math.acos( ct );
				// the lower plane, once it is above the horizon: only what is far enough off to the side stays above it
				const q = sp - th * cp, cl = q > 0 ? ( dy * ( cp + th * sp ) + r * sh ) / ( d * q ) : 1, wl = cl >= 1 ? 0 : cl <= - 1 ? 4 : Math.acos( cl );
				for ( let h = 0, o = k * H; h < H; h ++ ) {

					const b = B[ h ];
					out[ o + h ] = ( b <= w1 + hs || b >= w2 - hs ) && b <= wn + hs && b <= wt + hs && b >= wl - hs ? 1 : 0;

				}

			}

			return out;

		};

		};

		return { S, arc: arcOf( 1 ), arcOf, load, base, tmp };

	}

	// the fine tiers' budget: the same views, a plain rule. Every cell starts at its lowest level and mid goes to all
	// that ask for it, nearest first; then level by level (near for all that asked, then fine), nearest first, a cell
	// climbs while every view that sees it has the room - no cell is left at the far level to pay for the fine one
	// of another (a facade 35 m off in its far level shows more than a cell at 20 m without its top one), and what
	// does not fit stays a level down. The fine level counts cluster by cluster (each is culled on its own), drawn down to the parts that
	// still show at its distance (T.detail: size / distance). A cell that did not have a level the last time needs
	// 6 % to spare, so nothing flickers at the edge of the budget.
	_arcBudget( camera, order ) {

		const T = this.T, L = this.L, N = L.length, x = camera.position.x, z = camera.position.z;
		const { S, arc, arcOf, load, tmp } = this._views( camera );
		load.fill( 0 );
		// (the mirror image is drawn in the same frame: the town at its lowest level, from the camera's reflection in the
		// paving - the same views with the heights mirrored, `wide` times as broad. It is a twelfth of a full frame on
		// High and a sixth on Medium, and a walk's peak was the budget plus all of it: so it counts)
		const P = this.eng.probe && this.eng.probe.planar ? this.eng.probe : null, arcM = P ? arcOf( P.wide ) : null, ms = this._ms || ( this._ms = [ 0, 0, 0, 0, 0 ] );
		const mir = this._mir || ( this._mir = new Float32Array( S ) );
		mir.fill( 0 );
		for ( const c of this.chunks ) {

			if ( ! c.arc ) Object.assign( c, { arc: L.map( () => new Uint8Array( S ) ), cv: new Float32Array( S ), ma: new Uint8Array( S ) } );
			const A = arc( c.sph[ 0 ], c.arc[ 0 ] ), n = c.sph[ 0 ][ 3 ];
			for ( let h = 0; h < S; h ++ ) load[ h ] += c.cv[ h ] = A[ h ] ? n : 0;
			if ( ! P || ! c.mesh[ 0 ] ) continue;
			ms[ 0 ] = c.sph[ 0 ][ 0 ]; ms[ 1 ] = c.sph[ 0 ][ 1 ]; ms[ 2 ] = c.sph[ 0 ][ 2 ]; ms[ 4 ] = 2 * P.level - c.sph[ 0 ][ 4 ];
			const M = arcM( ms, c.ma );
			for ( let h = 0; h < S; h ++ ) if ( M[ h ] ) mir[ h ] += n;

		}

		if ( P ) for ( let h = 0; h < S; h ++ ) load[ h ] += mir[ h ];

		for ( const c of order ) { c.ask = c.want; c.want = 0; }
		const full = T.tris, spare = T.tris * 0.94, K = T.detail || 0;
		const up = ( c, l ) => {

			if ( ! c.m[ L[ l ] ] || c.failed & ( 1 << l ) ) return false;
			const A = arc( c.sph[ l ], c.arc[ l ] ), cv = c.cv, cap = c.was < l ? spare : full, parts = l === N - 1 && c.top.length > 0;
			const n = parts ? c.rest : c.sph[ l ][ 3 ];
			for ( let h = 0; h < S; h ++ ) tmp[ h ] = A[ h ] ? n : 0;
			if ( parts ) for ( const q of c.top ) {

				const d = Math.hypot( Math.max( q.box[ 0 ] - x, 0, x - q.box[ 2 ] ), Math.max( q.box[ 1 ] - z, 0, z - q.box[ 3 ] ) );
				// (a part is left out a little further off than it comes back: no flicker at a step)
				let t = K > 0 ? this._cut( q.steps, Math.min( 0.3, d * K ) ) : q.full;
				if ( K > 0 && c.was === l && q.n > t && q.n <= this._cut( q.steps, Math.min( 0.3, d * K * 0.85 ) ) ) t = q.n;
				q.t = t;
				const A = arc( q.sph, q.arc || ( q.arc = new Uint8Array( S ) ) );
				for ( let h = 0; h < S; h ++ ) if ( A[ h ] ) tmp[ h ] += t / 3;

			}

			for ( let h = 0; h < S; h ++ ) if ( tmp[ h ] > cv[ h ] && load[ h ] + tmp[ h ] - cv[ h ] > cap ) return false;
			for ( let h = 0; h < S; h ++ ) { load[ h ] += tmp[ h ] - cv[ h ]; cv[ h ] = tmp[ h ]; }
			if ( parts ) for ( const q of c.top ) q.n = q.t;
			c.want = l;
			return true;

		};

		for ( const c of order ) up( c, 1 );
		// (the cells one stands in or next to come before the rule: what is within arm's reach gets its top level before
		// a facade 40 m off gets its second - T.first metres, a fifth more for a cell that has it. Where the top level of
		// three cells is most of the budget - Medium - that left a facade 20 m off at its far level, which is made to
		// be seen from 45 m on: there the cell one stands in comes first, then the second level of all within T.sure
		// metres, and only then the top level of the neighbours)
		const first = () => { for ( const c of order ) if ( c.d < ( c.was === N - 1 ? T.first * 1.2 : T.first ) ) for ( let l = Math.max( 2, c.want + 1 ); l <= c.ask && c.want === l - 1; l ++ ) up( c, l ); };
		if ( T.sure > 0 ) {

			const cs = this.bake.bake && this.bake.bake.cell || 16, gx = Math.floor( x / cs ), gz = Math.floor( z / cs );
			for ( const c of order ) if ( c.gx === gx && c.gz === gz ) for ( let l = 2; l <= c.ask && c.want === l - 1; l ++ ) up( c, l );
			for ( const c of order ) if ( c.want === 1 && c.ask >= 2 && c.d < ( c.was >= 2 ? T.sure * 1.2 : T.sure ) ) up( c, 2 );

		}

		if ( T.first > 0 ) first();
		for ( let l = 2; l < N; l ++ ) for ( const c of order ) if ( c.want === l - 1 && l <= c.ask ) up( c, l );
		let most = 0, at = 0;
		for ( let h = 0; h < S; h ++ ) if ( load[ h ] > most ) { most = load[ h ]; at = h; }
		this._most = most; // what the fullest view holds
		// (and what of: [ heading (degrees), pitch, triangles at lo, mid, .., in the mirror image ] - for the capture's numbers)
		const by = this._mostBy = [ Math.round( ( at % 64 ) * 360 / 64 ), Math.round( VIEW_PITCH[ at / 64 | 0 ] * 180 / Math.PI ), ...L.map( () => 0 ), Math.round( mir[ at ] ) ];
		for ( const c of this.chunks ) if ( c.cv && c.cv[ at ] > 0 ) by[ 2 + c.want ] += Math.round( c.cv[ at ] );

	}

	// the triangle budget on what a view can hold: whichever way a camera here looks, what the renderer's culling lets
	// through may not add up to more than T.tris. The sum is kept per view: 64 headings x the pitches of VIEW_PITCH
	// (level, looking up - which brings the tall things at the sides into view - and looking down). A mesh is in a
	// view when its sphere passes the side planes (heading within its bearing +- the half field of view at that pitch
	// + the angle the sphere takes up, or around the back for one overhead), the near and the upper plane and, looking
	// up or down past the edge of the picture, the plane that then cuts off what is straight ahead; each arc is half a heading
	// step wider.
	// The top level takes the room first, nearest cell first, each of its clusters drawn down to the parts that still
	// show at its distance (T.detail: size / distance, about 3 pixels), then mid for the rest. Where a top cell does
	// not fit, or mid would not reach T.reach, all top cells leave out larger parts (the next step of the ladder, up
	// to 0.3 m); past the last step the farthest top cells fall back to mid until mid reaches. What the tightest
	// view cost the cells it does not look at is then given back: nearest first, each top cell as fine as it now
	// fits. Anything a cell did not have the last time needs 6 % to spare, so nothing flickers at the edge of the
	// budget. (tools/perf_sim.py runs the same rule over the whole town.)
	_viewBudget( camera, order ) {

		const T = this.T, x = camera.position.x, z = camera.position.z;
		const { S, arc, load, base, tmp } = this._views( camera );
		// per cell: lo (what each view sees of its lowest level), cv (... of the level it has in this plan)
		base.fill( 0 );
		for ( const c of this.chunks ) {

			if ( ! c.arc ) Object.assign( c, { arc: [ 0, 1, 2 ].map( () => new Uint8Array( S ) ), lo: new Float32Array( S ), cv: new Float32Array( S ) } );
			const A = arc( c.sph[ 0 ], c.arc[ 0 ] ), n = c.sph[ 0 ][ 3 ];
			for ( let h = 0; h < S; h ++ ) base[ h ] += c.lo[ h ] = A[ h ] ? n : 0;

		}

		for ( const c of order ) {

			c.ask = c.want; c.f0 = c.f;
			if ( c.m.mid ) arc( c.sph[ 1 ], c.arc[ 1 ] );
			if ( c.ask < 2 ) continue;
			arc( c.sph[ 2 ], c.arc[ 2 ] );
			for ( const q of c.top ) {

				arc( q.sph, q.arc || ( q.arc = new Uint8Array( S ) ) );
				q.d = Math.hypot( Math.max( q.box[ 0 ] - x, 0, x - q.box[ 2 ] ), Math.max( q.box[ 1 ] - z, 0, z - q.box[ 3 ] ) );

			}

		}

		// what the views see of a cell becomes nw, if all that see more of it have the room
		const put = ( c, nw, cap ) => {

			const cv = c.cv;
			for ( let h = 0; h < S; h ++ ) if ( nw[ h ] > cv[ h ] && load[ h ] + nw[ h ] - cv[ h ] > cap ) return false;
			for ( let h = 0; h < S; h ++ ) { load[ h ] += nw[ h ] - cv[ h ]; cv[ h ] = nw[ h ]; }
			return true;

		};

		const K = T.detail || 0.0055, reach = Math.min( T.mid, T.reach || T.mid ), F = [ 1, 1.6, 2.5, 4, 6.3, 10, 16 ];
		const full = T.tris, spare = T.tris * 0.94;
		const mid = ( c ) => {

			const A = c.arc[ 1 ], n = c.sph[ 1 ][ 3 ];
			for ( let h = 0; h < S; h ++ ) tmp[ h ] = A[ h ] ? n : 0;
			return put( c, tmp, c.was < 1 ? spare : full );

		};

		// the top level of a cell at step f of the ladder
		const top = ( c, f ) => {

			const A = c.arc[ 2 ], n = c.rest;
			for ( let h = 0; h < S; h ++ ) tmp[ h ] = A[ h ] ? n : 0;
			for ( const q of c.top ) {

				const n = ( q.t = this._cut( q.steps, Math.min( 0.3, q.d * K * f ) ) ) / 3, A = q.arc;
				for ( let h = 0; h < S; h ++ ) if ( A[ h ] ) tmp[ h ] += n;

			}

			if ( ! put( c, tmp, c.was < 2 || f < c.f0 ? spare : full ) ) return false;
			for ( const q of c.top ) q.n = q.t;
			c.f = f;
			return true;

		};

		const far = order.filter( ( c ) => c.ask === 2 && c.d > 0 ).reverse();
		let refused, gap;
		const solve = ( f, drop ) => {

			load.set( base );
			for ( const c of this.chunks ) c.cv.set( c.lo );
			for ( const c of order ) c.want = c.ask;
			for ( let i = 0; i < drop; i ++ ) far[ i ].want = 1;
			refused = 0; gap = Infinity;
			for ( const c of order ) if ( c.want === 2 && ! top( c, f ) ) { c.want = 1; refused ++; }
			// (?lmk=: a landmark's cell takes its mid level before the cells nearer by - it is what the view is of)
			if ( this.lmkK ) for ( const c of order ) if ( c.m.lmk && c.want === 1 && c.m.mid ) mid( c );
			for ( const c of order ) if ( c.want === 1 && ! ( c.m.mid && mid( c ) ) ) {

				c.want = 0;
				if ( c.d < gap ) gap = c.d;

			}

			return refused === 0 && gap >= reach;

		};

		let f = F[ 0 ];
		for ( const g of F ) { f = g; if ( solve( g, 0 ) ) break; }
		for ( let drop = 1; drop <= far.length && gap < reach; drop ++ ) solve( f, drop );
		for ( const c of order ) {

			if ( c.ask !== 2 ) continue;
			const now = c.want === 2 ? c.f : Infinity;
			for ( const g of F ) {

				if ( g >= now ) break;
				if ( top( c, g ) ) { c.want = 2; break; }

			}

		}

		this._f = f;

	}

	// the top level: each cluster down to the part size the budget gave it
	_parts( c, mesh ) {

		const u = mesh.userData;
		if ( ! u.parts ) return;
		u.shown = c.rest;
		u.parts.forEach( ( part, i ) => {

			const n = c.top[ i ].n;
			part.geometry.setDrawRange( part.userData.start, n );
			part.visible = n > 0;
			u.shown += n / 3;

		} );

	}

	// a change of level in progress: both meshes draw, dithered against each other, for T.fade seconds
	_fadeTick( clock, finish ) {

		for ( const c of this.fades ) {

			const f = c.fade, t = finish ? 1 : ( clock - f.t0 ) / this.fadeTime;
			const a = c.mesh[ f.from ], b = c.mesh[ f.to ];
			if ( t < 1 && a && b ) {

				const k = Math.max( 1e-3, t * t * ( 3 - 2 * t ) );
				for ( const o of b.userData.draws ) o.userData.fade = k;
				for ( const o of a.userData.draws ) o.userData.fade = - k;
				continue;

			}

			for ( const m of [ a, b ] ) if ( m ) for ( const o of m.userData.draws ) { o.material = o.userData.mat; o.userData.fade = 1; }
			if ( a ) a.visible = false;
			c.fade = null;
			this.fades.delete( c );
			this._dirty = true; // (the level it left can go now)

		}

	}

	// U.fineR: how far from the camera every cell shows its top level - the distance to the nearest cell that could
	// have it and does not (too far, still loading, refused by the budget, in a fade), T.fine at most. Eased, quicker
	// down than up, so a cell the budget takes the level from has lost the per-pixel light before its fade is half through.
	_fineReach( x, z, clock, snap ) {

		const u = this.U.fineR, N = this.L.length, top = this.L[ N - 1 ];
		let r = this.T.fine;
		for ( const c of this.chunks ) {

			if ( ! c.m[ top ] || c.failed & ( 1 << ( N - 1 ) ) || ( c.level === N - 1 && ! c.fade ) ) continue;
			const d = this._dist( c, x, z );
			if ( d < r ) r = d;

		}

		const dt = Math.min( clock - ( this._fineAt ?? clock ), 0.1 );
		this._fineAt = clock;
		this.fineR = r;
		u.value = this._fineFix ?? ( snap ? r : u.value + ( r - u.value ) * ( 1 - Math.exp( - dt / ( r < u.value ? 0.12 : 0.4 ) ) ) );

	}

	// choose the level of every cell for this camera, start / drop downloads, swap meshes
	update( camera, now, { force = false } = {} ) {

		const T = this.T, L = this.L, R = this.R, N = L.length, x = camera.position.x, z = camera.position.z;
		const clock = performance.now() / 1000;
		if ( this.fades.size ) this._fadeTick( clock, force );
		if ( this.matsDir && this.U && this.U.fineR && ! force ) this._fineReach( x, z, clock, false );
		if ( ! force && ! this._dirty && this._at && Math.hypot( x - this._at[ 0 ], z - this._at[ 1 ] ) < 0.75 && camera.aspect === this._at[ 2 ] ) return;
		this._at = [ x, z, camera.aspect ]; this._dirty = false;
		const order = [];
		for ( const c of this.chunks ) {

			c.d = this._dist( c, x, z );
			const keep = ( r, l ) => c.d < ( c.want >= l ? r + Math.min( r * 0.2 + 2, T.hold || Infinity ) : r );
			const lk = c.m.lmk && this.lmkK ? this.lmkK : 1; // (?lmk=)
			let want = 0;
			for ( let l = N - 1; l > 0 && ! want; l -- ) if ( c.m[ L[ l ] ] && R[ l ] > 0 && keep( R[ l ] * lk, l ) ) want = l;
			while ( want > 0 && ( c.failed & ( 1 << want ) || ! c.m[ L[ want ] ] ) ) want --;
			c.was = c.want; c.want = want;
			if ( want > 0 ) order.push( c );

		}

		order.sort( ( a, b ) => a.d - b.d );
		if ( T.resident ) {

			// what the GPU holds is every wanted level whole, in view or not (the view budget only narrows what is
			// drawn of it): over the ceiling, first the cells that are only still held for being left late go down a
			// level, farthest first, then the farthest of the top level (and of the one below it, where there are four)
			let held = this.loTris;
			for ( const c of order ) held += c.m[ L[ c.want ] ].ni / 3;
			const down = ( c ) => {

				held -= c.m[ L[ c.want ] ].ni / 3;
				do c.want --; while ( c.want > 0 && ( c.failed & ( 1 << c.want ) || ! c.m[ L[ c.want ] ] ) );
				if ( c.want > 0 ) held += c.m[ L[ c.want ] ].ni / 3;

			};

			for ( let i = order.length - 1; i >= 0 && held > T.resident; i -- ) {

				const c = order[ i ];
				while ( c.want > 0 && held > T.resident && c.d >= R[ c.want ] ) down( c );

			}

			for ( let top = N - 1; top >= 2; top -- ) for ( let i = order.length - 1; i >= 0 && held > T.resident; i -- ) if ( order[ i ].want === top ) down( order[ i ] );
			for ( let i = order.length - 1; i >= 0; i -- ) if ( order[ i ].want === 0 ) order.splice( i, 1 );

		}

		if ( T.budget === 'view' ) this._viewBudget( camera, order );
		else if ( T.budget === 'arc' || N > 3 ) this._arcBudget( camera, order );
		else this._budget( order );
		for ( const c of order ) {

			if ( c.want > 0 && ! c.mesh[ c.want ] ) {

				const key = c.m[ L[ c.want ] ].file;
				if ( ! this.inflight.has( key ) && this.inflight.size < T.fetch ) this.inflight.set( key, this._fetch( c, c.want ) );

			}

		}

		// (a fade draws two levels of a cell. A cell that waits to go down a level draws the dear one all the while, so
		// holding it back saves nothing - it only keeps the picture over the budget for longer, which is what a walk's
		// peak was made of. So the cells going down take their turn first: what their fades add, the cheap level, is kept
		// under 8 % of the budget. The ones going up wait - nearest first, at the level they have - while what is drawn,
		// fades and cells still to come down included, would pass what is wanted by more than those 8 %)
		let under = 0, drawn = 0, wanted = 0;
		for ( const c of this.fades ) if ( c.fade.down ) under += c.fade.cost;
		if ( this.fadeTime > 0 && ! force ) for ( const c of this.chunks ) {

			for ( const m of c.mesh ) if ( m && m.visible ) drawn += m.userData.shown ?? m.userData.tris;
			const m = c.mesh[ c.want ] || c.mesh[ c.level ];
			if ( m ) wanted += m.userData.tris;

		}

		const turn = this.fadeTime > 0 && ! force ? this.chunks.slice().sort( ( a, b ) => ( a.want < a.level ? 0 : 1 ) - ( b.want < b.level ? 0 : 1 ) || a.d - b.d ) : this.chunks;
		for ( const c of turn ) {

			if ( c.fade ) continue; // (one change at a time: the next starts when this one is through)
			// show the wanted level, or until it is in the best one there is: the one shown now, else the nearest below, else above
			let show = c.want;
			if ( ! c.mesh[ show ] ) {

				if ( c.mesh[ c.level ] ) show = c.level;
				else {

					show = 0;
					for ( let l = c.want - 1; l > 0 && ! show; l -- ) if ( c.mesh[ l ] ) show = l;
					for ( let l = c.want + 1; l < N && ! show; l ++ ) if ( c.mesh[ l ] ) show = l;

				}

			}

			const a = c.mesh[ c.level ], b = c.mesh[ show ];
			if ( show !== c.level && this.fadeTime > 0 && ! force && a && b && a.visible && a.userData.draws && b.userData.draws ) {

				const ca = a.userData.shown ?? a.userData.tris, cb = b.userData.tris;
				const down = cb < ca, cost = down ? cb : ca;
				if ( down ) {

					if ( under > 0 && under + cost > T.tris * 0.08 ) continue;
					under += cost;

				} else if ( this.fades.size > 0 && drawn + cb > wanted + T.tris * 0.08 ) continue;

				drawn += cb;

				c.fade = { from: c.level, to: show, t0: clock, cost, down };
				this.fades.add( c );
				for ( const o of a.userData.draws ) { o.material = o.userData.fmat; o.userData.fade = - 1e-3; }
				for ( const o of b.userData.draws ) { o.material = o.userData.fmat; o.userData.fade = 1e-3; }
				b.visible = true;
				this._parts( c, b );
				c.level = show;
				continue;

			}

			for ( let l = 0; l < N; l ++ ) {

				const mesh = c.mesh[ l ];
				if ( ! mesh ) continue;
				mesh.visible = l === show;
				const u = mesh.userData;
				if ( l === show ) this._parts( c, mesh );

				if ( l > 0 && l !== show && l !== c.want ) this._drop( c, l );

			}

			c.level = show;

		}

		if ( this.matsDir && this.U && this.U.fineR && force ) this._fineReach( x, z, clock, true );

	}

	get pending() {

		let n = this.inflight.size;
		for ( const c of this.chunks ) if ( c.want > 0 && ! c.mesh[ c.want ] && ! this.inflight.has( c.m[ this.L[ c.want ] ].file ) ) n ++;
		return n;

	}

	// resolve when the view from this camera is complete ('all') or its surroundings are in ('near')
	async settle( camera ) {

		const t0 = performance.now();
		for ( ;; ) {

			this.update( camera, 0, { force: true } );
			const waitBase = this.wait === 'all' ? this.baseLeft > 0 : false;
			let waitLevels = this.pending > 0;
			if ( this.wait !== 'all' ) {

				// walking: only the cells right around the camera hold up the first frame, and not for long
				waitLevels = this.chunks.some( ( c ) => c.want > 0 && ! c.mesh[ c.want ] && c.d < 6 ) && performance.now() - t0 < 2500;

			}

			if ( ! waitBase && ! waitLevels ) break;
			await new Promise( ( r ) => this.waiters.push( r ) );

		}

		this.update( camera, 0, { force: true } );

	}

	visibleStats() {

		const N = this.L.length, lods = this.L.map( () => 0 );
		let items = 0, tris = 0, gpu = 0;
		for ( const c of this.chunks ) {

			if ( c.mesh[ c.level ] ) {

				items ++; lods[ N - 1 - c.level ] ++;
				tris += c.mesh[ c.level ].userData.shown ?? c.mesh[ c.level ].userData.tris;

			}

			for ( const mesh of c.mesh ) if ( mesh ) gpu += mesh.userData.tris;

		}

		// lods: cells shown at [ near, mid, lo ] - or [ fine, near, mid, lo ]; view: what the fullest view from here holds
		// step: how far the view budget has the top level leave out small parts (1: only what does not show anyway)
		return { items, lods, pending: this.pending + ( this.baseLeft > 0 ? this.baseLeft : 0 ), tris, resident: gpu, bytes: this.bytes, ...( this._f ? { step: this._f } : {} ), ...( this._most ? { view: this._most, by: this._mostBy } : {} ),
			...( this.TX ? { sharp: [ this.TX.u.b1.value - this.TX.u.b0.value, this.layers.image.width, this.layersBig ? this.layersBig.image.width : 0 ] } : {} ) };

	}

}
