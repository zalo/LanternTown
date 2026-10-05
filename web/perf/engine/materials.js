// Shared PBR material library: maps (lib, tint, params) from GLB material extras to cached TSL node
// materials. Baked materials use the procedural texture sets in out/mat/<name>/; "special" materials
// (glass, lit windows, emissive, water ...) are built from nodes only.
//
// Geometry contract: uv in METRES (glTF V flipped), attribute _mask = (ao, grime, wear, special).
//
// Tint, glow, roughness, opacity and curtain are per-primitive values that do not change the shader, so they
// ride on two more vertex attributes written by apply():  _tint = sqrt( rgb / 8 ),  _prm = ( glow / 128,
// rough / 4, opacity, curtain ).  One node material then serves every tint of a library material (a town
// has ~800 (lib, tint, params) combinations but only ~80 distinct shaders).

import * as THREE from 'three';
import {
	texture, uv, attribute, positionWorld, normalWorld, vec2, vec3, vec4, float, mix, smoothstep, saturate,
	uniform, normalMap, luminance, time, fract, floor, sin, cos, dot, abs, max, min, pow, step, oneMinus,
	cameraPosition, normalize, cross, length, exp, clamp, mrt, cubeTexture, reflect, normalView
} from 'three/tsl';
import { viewportSharedTexture, screenUV, positionView, normalViewGeometry } from 'three/tsl'; // (water, reflections)
import { diffuseContribution } from 'three/tsl'; // (foliage lighting)

// blended surfaces must not write into the normal G-buffer (it feeds GTAO): alpha 0 leaves it untouched
export const NO_NORMAL = mrt( { normal: vec4( 0, 0, 0, 0 ), mr: vec4( 0, 0, 0, 0 ) } );

const lin = ( hex ) => new THREE.Color( hex ); // THREE.Color parses hex as sRGB -> linear working space
// constant colour node. NOT vec3( THREE.Color ): TSL reads .x/.y/.z of the value, which a Color lacks, and the
// constant silently compiles to black.
const col3 = ( hex ) => { const c = lin( hex ); return vec3( c.r, c.g, c.b ); };

// Foliage cards (leaf / hedge with the `leafy` param): a card stands for a spray of small blades at every angle,
// and a blade lets light through. On top of the Lambert term every card takes a share of each light whatever
// its own facing, and more of it (yellower: it has passed through the blade) when it is lit from behind - so a
// crown lit by a lantern from below shades as one soft volume, not as lit and unlit polygons side by side.
class FoliageLighting extends THREE.PhysicalLightingModel {

	direct( input, builder ) {

		super.direct( input, builder );
		const { lightDirection, lightColor, reflectedLight } = input;
		const nl = normalView.dot( lightDirection );
		const k = oneMinus( saturate( nl ) ).mul( 0.3 ).add( saturate( nl.negate() ).mul( 0.3 ) ).mul( 1 / Math.PI );
		reflectedLight.directDiffuse.addAssign( lightColor.mul( diffuseContribution ).mul( vec3( 1.0, 0.95, 0.6 ) ).mul( k ) );

	}

}

// params that change the shader graph (everything else is per-vertex data or only used by the Blender side)
const SHADER_PARAMS = [ 'scale', 'stripe', 'tint2', 'bump', 'double', 'metal', 'flicker', 'pane', 'accent', 'depth', 'speed', 'ripple', 'probe', 'leafy' ];
// per-vertex defaults of the node-only materials
const SPECIAL = {
	plain: { tint: '#808080', rough: 0.7 },
	metal_plain: { tint: '#c9c9c9', rough: 0.35 },
	emissive: { tint: '#ffb45a', glow: 8 },
	neon: { tint: '#ffb45a', glow: 14 },
	glass_frosted: { tint: '#efe6d2', glow: 3, rough: 0.5 },
	glass_clear: { tint: '#b9c9c2', opacity: 0.16 },
	glass_lantern: { tint: '#ffdca8', glow: 0.5, opacity: 0.22 },
	glass_stained: { tint: '#c89a3c', glow: 1.6 },
	window_dark: { tint: '#ffc27a' },
	window_lit: { tint: '#ffc27a', glow: 2.4 },
	water: { tint: '#0b1512' },
	water_fall: { tint: '#cfe0e6', opacity: 0.7 },
};
const u16 = ( x ) => Math.round( 65535 * Math.min( 1, Math.max( 0, x ) ) );

function makeNoiseTexture( size = 256 ) {

	// 4 independent channels of tiling fractal value noise
	const data = new Uint8Array( size * size * 4 );
	let seed = 1234567;
	const rnd = () => ( ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296 );
	for ( let c = 0; c < 4; c ++ ) {

		const acc = new Float32Array( size * size );
		let amp = 1, tot = 0;
		for ( let cells = 8; cells <= 64; cells *= 2 ) {

			const g = new Float32Array( cells * cells );
			for ( let i = 0; i < g.length; i ++ ) g[ i ] = rnd();
			const k = size / cells;
			for ( let y = 0; y < size; y ++ ) {

				const fy = y / k, y0 = Math.floor( fy ), ty = fy - y0, sy = ty * ty * ( 3 - 2 * ty );
				for ( let x = 0; x < size; x ++ ) {

					const fx = x / k, x0 = Math.floor( fx ), tx = fx - x0, sx = tx * tx * ( 3 - 2 * tx );
					const x1 = ( x0 + 1 ) % cells, y1 = ( y0 + 1 ) % cells;
					const a = g[ y0 * cells + x0 ], b = g[ y0 * cells + x1 ], d = g[ y1 * cells + x0 ], e = g[ y1 * cells + x1 ];
					acc[ y * size + x ] += amp * ( a + ( b - a ) * sx + ( d - a ) * sy + ( a - b - d + e ) * sx * sy );

				}

			}

			tot += amp;
			amp *= 0.5;

		}

		let lo = 1e9, hi = - 1e9;
		for ( let i = 0; i < acc.length; i ++ ) {

			lo = Math.min( lo, acc[ i ] ); hi = Math.max( hi, acc[ i ] );

		}

		for ( let i = 0; i < acc.length; i ++ ) data[ i * 4 + c ] = Math.round( 255 * ( acc[ i ] - lo ) / ( hi - lo ) );

	}

	const t = new THREE.DataTexture( data, size, size, THREE.RGBAFormat );
	t.wrapS = t.wrapT = THREE.RepeatWrapping;
	t.magFilter = THREE.LinearFilter;
	t.minFilter = THREE.LinearMipmapLinearFilter;
	t.generateMipmaps = true;
	t.colorSpace = THREE.NoColorSpace;
	t.needsUpdate = true;
	return t;

}

export class MaterialLib {

	constructor( { base = '.', anisotropy = 8, maxTextureSize = 2048 } = {} ) {

		this.base = base;
		this.anisotropy = anisotropy;
		this.maxTextureSize = maxTextureSize;
		this.catalog = null;
		this.info = new Map(); // name -> mat.json (or null if not baked)
		this.tex = new Map(); // name -> {albedo, normal, orm}
		this.versions = new Map();
		this.cache = new Map();
		this.pending = new Map(); // name -> { v, p } material loads in flight
		this.loader = new THREE.TextureLoader();
		this.noise = makeNoiseTexture();
		this.warnings = new Set();
		this.probe = null; // reflection probes (set by the engine): { p: [{rt, pos}], min, max, splitZ, gain, mats }
		// global look uniforms (shared by every material)
		this.u = {
			wet: uniform( 0.0 ), // global rain wetness 0..1 added to "wet" specials
			aoDirect: uniform( 0.55 ), // how much vertex AO also darkens direct light
			grime: uniform( 1.0 ),
			wear: uniform( 1.0 ),
			special: uniform( 1.0 ),
			macro: uniform( 1.0 ),
			windowGlow: uniform( 1.0 ),
			emissive: uniform( 1.0 ),
			puddleRough: uniform( 0.09 ), // roughness of standing rain water (a soft mirror, not a perfect one)
			filmRough: uniform( 0.17 ), // roughness of the water film on rain-wet level stone
			ripple: uniform( 0.03 ), // micro-normal of standing water (tangent slope)
			poolCap: uniform( 1.0 ),
			tideY: uniform( - 1e4 ), tideH: uniform( 0.4 ), // world height of standing water and how far above it masonry stays dark and wet (off by default)
			soft: uniform( 0.0 ), // 0..1 weathering as seen at street distance: thinner wear, grime that sits in crevices and runs in rain streaks, calmer plaster relief // how far a low zone may flood: caps the size of a pool
		};

	}

	async init() {

		const r = await fetch( `${this.base}/docs/materials.json`, { cache: 'no-cache' } );
		const j = await r.json();
		this.catalog = new Map( j.materials.map( ( m ) => [ m.name, m ] ) );
		return this;

	}

	// make sure textures + metadata of the given library materials are loaded (versions: name -> any)
	async prepare( names, versions = {} ) {

		const jobs = [];
		for ( const name of new Set( names ) ) {

			const entry = this.catalog.get( name );
			if ( ! entry || entry.kind !== 'baked' ) continue;
			const v = versions[ name ] ?? 0;
			// a load in flight has its metadata before its textures: wait for it rather than skip, or the
			// caller builds (and caches) the flat stand-in for a material that is about to be textured
			const cur = this.pending.get( name );
			if ( cur && cur.v === v ) { jobs.push( cur.p ); continue; }
			if ( ! cur && this.info.has( name ) && this.versions.get( name ) === v ) continue;
			const p = ( cur ? cur.p.then( () => this._load( name, v ) ) : this._load( name, v ) )
				.finally( () => { if ( this.pending.get( name )?.p === p ) this.pending.delete( name ); } );
			this.pending.set( name, { v, p } );
			jobs.push( p );

		}

		await Promise.all( jobs );

	}

	async _load( name, v ) {

		// drop stale variants of this material
		if ( this.info.has( name ) ) {

			for ( const [ k, m ] of this.cache ) if ( k.startsWith( name + '|' ) ) {

				m.dispose(); this.cache.delete( k );

			}

			const old = this.tex.get( name );
			if ( old ) for ( const t of Object.values( old ) ) t.dispose();

		}

		this.versions.set( name, v );
		let info = null;
		try {

			const r = await fetch( `${this.base}/out/mat/${name}/mat.json?v=${v}`, { cache: 'no-cache' } );
			if ( r.ok ) info = await r.json();

		} catch ( e ) { /* not baked */ }

		this.info.set( name, info );
		if ( ! info ) {

			this.tex.delete( name );
			this.warnings.add( `material '${name}' is not baked yet (flat placeholder used)` );
			return;

		}

		const load = async ( file, srgb ) => {

			const url = `${this.base}/out/mat/${name}/${file}?v=${v}`, cap = this.maxTextureSize;
			let t;
			if ( cap < 2048 && typeof createImageBitmap === 'function' ) {

				// low tiers: decode straight to the capped size (a 2048 map with mips is 22 MB of GPU memory, a 512 one
				// 1.4 MB) and drop the pixels once they are on the GPU, so the page holds no CPU copy either
				const bmp = await createImageBitmap( await ( await fetch( url ) ).blob(),
					{ resizeWidth: cap, resizeHeight: cap, resizeQuality: 'high', imageOrientation: 'flipY', premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
				t = new THREE.Texture( bmp );
				t.flipY = false;
				t.needsUpdate = true;
				t.onUpdate = () => {

					t.onUpdate = null; bmp.close();
					t.source.data = { width: cap, height: cap };

				};

			} else t = await this.loader.loadAsync( url );

			t.wrapS = t.wrapT = THREE.RepeatWrapping;
			t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
			t.anisotropy = this.anisotropy;
			t.minFilter = THREE.LinearMipmapLinearFilter;
			t.generateMipmaps = true;
			return t;

		};

		const [ albedo, normal, orm ] = await Promise.all( [
			load( info.maps.albedo, true ), load( info.maps.normal, false ), load( info.maps.orm, false ) ] );
		this.tex.set( name, { albedo, normal, orm } );

	}

	// names of library materials used by a loaded glTF scene
	static libsOf( root ) {

		const s = new Set();
		root.traverse( ( o ) => {

			if ( ! o.isMesh ) return;
			for ( const m of Array.isArray( o.material ) ? o.material : [ o.material ] ) {

				const lib = m?.userData?.tsq_lib;
				if ( lib ) s.add( lib );

			}

		} );
		return [ ...s ];

	}

	// the shared node material of (lib, shader params). Geometry drawn with it needs the _tint / _prm vertex
	// attributes, so meshes go through apply(); get() alone is only for callers that write those themselves.
	get( lib, params = {} ) {

		const sp = {};
		for ( const k of SHADER_PARAMS ) if ( params[ k ] !== undefined && params[ k ] !== null ) sp[ k ] = params[ k ];
		const key = `${lib}|${JSON.stringify( sp )}`;
		let m = this.cache.get( key );
		if ( m ) return m;
		let entry = this.catalog.get( lib );
		if ( ! entry ) {

			this.warnings.add( `unknown material '${lib}' (plain used)` );
			entry = this.catalog.get( 'plain' );

		}

		m = entry.kind === 'baked' ? this._baked( entry, sp ) : this._special( entry, sp );
		m.name = key;
		m.userData.tsq = { lib, params: sp };
		this.cache.set( key, m );
		return m;

	}

	// the variant of a library material that can be drawn into a reflection probe: blended materials name the scene
	// pass's extra MRT outputs (which a cube target lacks) and some sample the probes themselves. null = use as is.
	probeSafe( m ) {

		const ud = m.userData;
		if ( ud.safe !== undefined ) return ud.safe;
		ud.safe = null;
		const entry = ud.tsq && this.catalog.get( ud.tsq.lib );
		if ( entry && entry.kind !== 'baked' && ( m.mrtNode || ud.probe ) ) {

			this._safe = true;
			const v = this._special( entry, ud.tsq.params );
			this._safe = false;
			v.name = m.name + '|safe';
			v.userData.safe = null;
			ud.safe = v;

		}

		return ud.safe;

	}

	// per-vertex look values of (lib, tint, params), encoded for the _tint / _prm attributes
	vertexValues( lib, tint = null, params = {} ) {

		const entry = this.catalog.get( lib ) || this.catalog.get( 'plain' );
		const baked = entry.kind === 'baked';
		let col, d;
		if ( baked && this.info.get( entry.name ) && this.tex.get( entry.name ) ) {

			// albedo multiplier that moves the texture's mean colour onto the tint
			const mean = this.info.get( entry.name ).preview;
			col = tint && entry.tintable ? tint.map( ( t, i ) => Math.min( 6, t / Math.max( mean[ i ], 0.004 ) ) ) : [ 1, 1, 1 ];
			d = {};

		} else {

			d = SPECIAL[ baked ? 'plain' : entry.name ] || SPECIAL.plain;
			col = tint || ( baked ? [ 0.3, 0.3, 0.3 ] : lin( d.tint ).toArray() );

		}

		const glow = d.glow === undefined ? ( params.glow || 0 ) : ( params.glow ?? d.glow );
		return [ ...col.slice( 0, 3 ).map( ( x ) => u16( Math.sqrt( Math.max( 0, x ) / 8 ) ) ), 65535,
			u16( glow / 128 ), u16( ( params.rough ?? d.rough ?? 0.7 ) / 4 ), u16( params.opacity ?? d.opacity ?? 1 ), u16( params.curtain ?? 0.5 ) ];

	}

	// give a mesh its library material: writes the per-vertex look attributes and sets mesh.material.
	// A geometry shared by primitives with different looks is split (shallow copy + its own two attributes);
	// pass own = true to restyle a mesh's geometry in place instead.
	apply( mesh, lib, tint = null, params = {}, own = false ) {

		const v = this.vertexValues( lib, tint, params ), key = v.join( ',' );
		let g = mesh.geometry;
		if ( g.userData.tsqLook !== key ) {

			if ( g.userData.tsqLook !== undefined && ! own ) {

				const src = g;
				g = mesh.geometry = new THREE.BufferGeometry();
				for ( const k in src.attributes ) g.setAttribute( k, src.attributes[ k ] );
				g.setIndex( src.index );
				for ( const gr of src.groups ) g.addGroup( gr.start, gr.count, gr.materialIndex );
				g.boundingBox = src.boundingBox; g.boundingSphere = src.boundingSphere;
				g.userData = { ...src.userData };

			}

			const n = g.getAttribute( 'position' ).count;
			const a = new Uint16Array( n * 4 ), b = new Uint16Array( n * 4 );
			for ( let i = 0, j = 0; i < n; i ++, j += 4 ) {

				a[ j ] = v[ 0 ]; a[ j + 1 ] = v[ 1 ]; a[ j + 2 ] = v[ 2 ]; a[ j + 3 ] = v[ 3 ];
				b[ j ] = v[ 4 ]; b[ j + 1 ] = v[ 5 ]; b[ j + 2 ] = v[ 6 ]; b[ j + 3 ] = v[ 7 ];

			}

			g.setAttribute( '_tint', new THREE.BufferAttribute( a, 4, true ) );
			g.setAttribute( '_prm', new THREE.BufferAttribute( b, 4, true ) );
			g.userData.tsqLook = key;

		}

		return ( mesh.material = this.get( lib, params ) );

	}

	// ------------------------------------------------------------------ shared node pieces

	_common( params ) {

		const mask = attribute( '_mask', 'vec4' );
		const wp = positionWorld;
		const q = vec2( wp.x.add( wp.y.mul( 0.37 ) ), wp.z.add( wp.y.mul( 0.61 ) ) );
		const nLo = texture( this.noise, q.mul( 0.11 ) ); // ~1 m blotches
		const nHi = texture( this.noise, q.mul( 0.93 ).add( vec2( 0.31, 0.77 ) ) ); // ~12 cm
		const u0 = uv();
		const scale = params.scale ?? 1;
		const uvm = vec2( u0.x, oneMinus( u0.y ) ).div( scale ); // metres, V up (Blender convention)
		const vt = attribute( '_tint', 'vec4' ), prm = attribute( '_prm', 'vec4' );
		// foot: metres of surface under one pixel (0.03+ = roof tiles and courses are only a few pixels each)
		const fw = uvm.fwidth();
		return { mask, wp, q, nLo, nHi, uvm, foot: max( fw.x, fw.y ), tint: vt.rgb.mul( vt.rgb ).mul( 8 ), glow: prm.x.mul( 128 ), rough: prm.y.mul( 4 ), opacity: prm.z, curtain: prm.w };

	}

	// grime / wear / special blending. s: {alb, rough, metal, h (0..1 height), flat (normal flatten 0..1)}
	_weather( entry, c, s ) {

		const U = this.u;
		const g = entry.grime, w = entry.wear, sp = entry.special;
		const { mask, nLo, nHi } = c;
		// standing rain water: shallow pools a few metres across on level ground. The water level rises towards the
		// middle of a pool, so stones at its edge stand proud of it and the joints flood first.
		const pools = () => {

			const nP = texture( this.noise, c.q.mul( 0.031 ).add( vec2( 0.53, 0.17 ) ) );
			// a low zone a few metres across floods only so far (poolCap); metre-scale unevenness then breaks it into
			// several pools no more than a pace or two wide, and the relief decides their outline (joints first)
			const zone = min( nP.r.add( nP.b.mul( 0.35 ) ).sub( 0.8 ).mul( 5 ).add( U.wet.sub( 0.5 ) ), U.poolCap );
			const level = zone.add( nLo.a.sub( 0.5 ).mul( 2.4 ) ).add( nHi.a.sub( 0.5 ).mul( 0.3 ) ).sub( 0.25 );
			const lvl = smoothstep( 0.985, 0.998, normalWorld.y ).mul( smoothstep( 0.05, 0.3, U.wet ) );
			const depth = level.sub( s.h.mul( 0.8 ) );
			const water = smoothstep( 0.0, 0.5, depth ).mul( lvl ); // feathered over 10-20 cm
			const halo = smoothstep( - 1.0, 0.1, depth ).mul( lvl ); // damp, darker stone around it
			s.alb = s.alb.mul( mix( float( 1 ), float( 0.78 ), halo ) ).mul( mix( float( 1 ), float( 0.62 ), water ) );
			s.rough = mix( s.rough.mul( mix( float( 1 ), float( 0.7 ), halo ) ), U.puddleRough, water );
			s.flat = max( s.flat, water );
			// wind / drip ripples: a fine, slowly drifting micro-normal instead of a dead flat mirror
			const tt = time.mul( 0.03 );
			const r1 = texture( this.noise, c.q.mul( 1.3 ).add( vec2( tt, tt.mul( 0.7 ) ) ) ).rg.sub( 0.5 );
			const r2 = texture( this.noise, c.q.mul( 3.1 ).add( vec2( tt.mul( - 1.3 ), tt.mul( 0.9 ) ) ) ).ba.sub( 0.5 );
			s.rip = r1.add( r2.mul( 0.5 ) ).mul( U.ripple ).mul( water );

		};

		if ( w ) {

			const amt = mask.z.mul( U.wear );
			const k = smoothstep( 0.35, 0.8, amt.mul( 1.25 ).add( s.h.sub( 0.5 ).mul( 0.45 ) ).add( nHi.a.sub( 0.5 ).mul( 0.5 ) ) )
				.mul( smoothstep( 0.02, 0.2, amt ) );
			const wc = col3( w.color );
			// (soft: whole roof tiles / boards flipping to the bare wear colour read as white speckle from across a square)
			const wk = mix( float( 1 ), float( 0.45 ), U.soft );
			s.alb = mix( s.alb, w.keep_hue ? s.alb.mul( 1.6 ).min( wc.max( s.alb ) ) : mix( wc, s.alb.mul( 1.5 ).min( wc.max( s.alb ) ), U.soft.mul( 0.6 ) ), k.mul( w.amount ?? 0.85 ).mul( wk ) );
			s.rough = mix( s.rough, float( w.rough ), k.mul( 0.9 ).mul( wk ) );
			s.metal = mix( s.metal, float( w.metal ?? 0 ), k );

		}

		if ( sp && sp.kind && sp.kind !== 'none' ) {

			let amt = mask.w.mul( U.special );
			const sc = col3( sp.color );
			if ( sp.kind === 'wet' ) {

				amt = amt.add( U.wet ).saturate();
				// water gathers in the low parts of the relief first; fully wet = sheen everywhere + puddles in joints
				const k = smoothstep( 0.15, 0.75, amt.add( nLo.g.sub( 0.5 ).mul( 0.35 ) ).add( nHi.g.sub( 0.5 ).mul( 0.18 ) ).sub( s.h.sub( 0.5 ).mul( 0.2 ) ) ).mul( smoothstep( 0.02, 0.2, amt ) );
				const puddle = k.mul( smoothstep( 0.55, 0.2, s.h.add( oneMinus( amt ).mul( 0.4 ) ) ) );
				s.alb = s.alb.mul( mix( float( 1 ), float( 0.5 ), k ) );
				s.rough = mix( mix( s.rough, s.rough.mul( 0.45 ), k ), float( sp.rough ), puddle );
				s.flat = max( s.flat, puddle.mul( 0.95 ) );
				pools();

			} else if ( sp.kind === 'fade' ) {

				const k = smoothstep( 0.2, 0.8, amt.add( nLo.b.sub( 0.5 ).mul( 0.4 ) ) ).mul( smoothstep( 0.02, 0.2, amt ) );
				const l = luminance( s.alb );
				s.alb = mix( s.alb, mix( vec3( l ), sc, 0.25 ).mul( 1.25 ).add( 0.02 ), k.mul( 0.65 ) );
				s.rough = mix( s.rough, float( sp.rough ), k.mul( 0.6 ) );

			} else { // moss / rust / verdigris: patchy overgrowth that starts in the recesses

				const k = smoothstep( 0.42, 0.62, amt.mul( 1.15 ).add( float( 0.5 ).sub( s.h ).mul( 0.3 ) ).add( nLo.r.sub( 0.5 ).mul( 0.45 ) ).add( nHi.r.sub( 0.5 ).mul( 0.35 ) ) )
					.mul( smoothstep( 0.02, 0.2, amt ) );
				const var_ = nHi.g.mul( 0.7 ).add( 0.65 );
				s.alb = mix( s.alb, sc.mul( var_ ), k.mul( 0.92 ) );
				s.rough = mix( s.rough, float( sp.rough ), k );
				s.metal = mix( s.metal, float( 0 ), k );
				if ( sp.kind === 'moss' ) s.flat = max( s.flat, k.mul( 0.6 ) );

			}

		}

		if ( ! ( sp && sp.kind === 'wet' ) ) {

			// rain (U.wet, 0 unless a scene asks for it): everything that faces the sky darkens and takes a sheen, and water
			// stands in the low parts of near-level surfaces. Paving with a "wet" special handles this itself above.
			const ny = normalWorld.y;
			const amt = U.wet.mul( smoothstep( 0.5, 0.92, ny ) );
			// (seen from across the square a pitched roof is damp in broad patches: tile by tile, wet-dark next to
			// dry-pale at a few pixels each is salt and pepper)
			const fine = oneMinus( smoothstep( 0.008, 0.03, c.foot ).mul( oneMinus( smoothstep( 0.9, 0.99, ny ) ) ) );
			const k = smoothstep( 0.15, 0.75, amt.add( nLo.g.sub( 0.5 ).mul( 0.35 ) ).add( nHi.g.sub( 0.5 ).mul( 0.18 ).sub( s.h.sub( 0.5 ).mul( 0.2 ) ).mul( fine ) ) ).mul( smoothstep( 0.02, 0.2, amt ) );
			const puddle = k.mul( smoothstep( 0.55, 0.2, s.h.add( oneMinus( amt ).mul( 0.4 ) ) ) ).mul( smoothstep( 0.97, 0.995, ny ) );
			// level stone keeps a film of water in patches: the damp sheen that mirrors the lamps
			const film = k.mul( smoothstep( 0.9, 0.99, ny ) ).mul( smoothstep( 0.3, 0.75, nLo.b.mul( 0.7 ).add( nHi.a.mul( 0.3 ) ).add( U.wet.sub( 0.5 ) ) ) );
			s.alb = s.alb.mul( mix( float( 1 ), float( 0.58 ), k ) );
			// (a pitched roof sheds its water: less sheen than a level slab)
			s.rough = mix( mix( mix( s.rough, s.rough.mul( mix( float( 0.72 ), float( 0.45 ), smoothstep( 0.85, 0.98, ny ) ) ), k ), min( s.rough, U.filmRough ), film.mul( 0.85 ) ), float( 0.07 ), puddle );
			s.flat = max( s.flat, puddle.mul( 0.95 ) );
			pools();

		}

		if ( g ) {

			const amt = mask.y.mul( U.grime );
			const k0 = smoothstep( 0.3, 0.85, amt.mul( 1.3 ).add( float( 0.5 ).sub( s.h ).mul( 0.4 ) ).add( nHi.b.sub( 0.5 ).mul( 0.45 ) ).add( nLo.a.sub( 0.5 ).mul( 0.3 ) ) );
			// soft: metre-wide blotches read as brush smears on a facade. Dirt collects where the painted occlusion is
			// (under sills, in corners), in the low parts of the relief, and on walls it runs down in rain streaks.
			const wall = smoothstep( 0.35, 0.8, oneMinus( abs( normalWorld.y ) ) );
			const streak = texture( this.noise, vec2( c.wp.x.mul( 1.1 ).add( c.wp.z.mul( 0.9 ) ).mul( 0.45 ), c.wp.y.mul( 0.035 ) ) ).g;
			const k1 = smoothstep( 0.35, 0.9, amt.mul( 1.1 ).add( oneMinus( mask.x ).mul( 0.5 ) ).add( float( 0.5 ).sub( s.h ).mul( 0.4 ) ).add( nHi.b.sub( 0.5 ).mul( 0.3 ) )
				.add( streak.sub( 0.5 ).mul( 0.55 ).mul( wall ) ).add( nLo.a.sub( 0.5 ).mul( 0.12 ).mul( oneMinus( wall ) ) ) );
			const k = mix( k0, k1, U.soft ).mul( smoothstep( 0.02, 0.2, amt ) );
			const gc = col3( g.color );
			const dirty = mix( mix( s.alb.mul( gc ).mul( 2.2 ), gc, 0.45 ), s.alb.mul( 0.42 ).add( gc.mul( 0.6 ) ), U.soft );
			s.alb = mix( s.alb, dirty, k.mul( mix( float( 0.85 ), float( 0.7 ), U.soft ) ) );
			s.rough = mix( s.rough, float( g.rough ), k.mul( 0.85 ) );
			s.metal = mix( s.metal, float( 0 ), k.mul( 0.8 ) );

		}

		{

			// tide band: whatever stands in the water (quay walls, steps, hulls) is dark and wet for a hand or two above
			// the surface, with a ragged upper edge and a touch of algae at the waterline
			const top = U.tideY.add( U.tideH.mul( nLo.b.mul( 0.9 ).add( nHi.r.mul( 0.3 ) ).add( 0.35 ) ) );
			const band = smoothstep( top, U.tideY.add( 0.02 ), c.wp.y );
			s.alb = mix( s.alb, s.alb.mul( vec3( 0.36, 0.42, 0.32 ) ), band.mul( 0.9 ) );
			s.rough = mix( s.rough, min( s.rough, float( 0.3 ) ), band.mul( 0.8 ) );

		}

		return s;

	}

	_finish( m, c, s, texAO = null, nTex = null ) {

		const U = this.u;
		const vao = c.mask.x;
		const ao = texAO ? vao.mul( texAO ) : vao;
		const direct = mix( float( 1 ), vao, U.aoDirect ).mul( texAO ? mix( float( 1 ), texAO, 0.5 ) : 1 );
		m.colorNode = s.alb.mul( direct );
		// specular anti-aliasing: where the shading normal changes faster than the pixel grid can show (distant roof
		// tiles, cobbles, carved detail) the highlight widens instead of sparkling. Level ground is left alone: its
		// gloss also drives the mirror image added in post.
		const dx = normalView.dFdx(), dy = normalView.dFdy();
		// (also from the normal map itself: its mips average the relief away and the shortened vector says how much
		// (Toksvig); and by footprint: a tile or course a few pixels wide has one flat normal each, which no derivative
		// sees - glossy next to dull tile by tile is the same sparkle)
		let vari = dot( dx, dx ).add( dot( dy, dy ) ).mul( 0.5 ).add( smoothstep( 0.008, 0.04, c.foot ).mul( 0.07 ) );
		if ( nTex ) {

			const nl = nTex.rgb.mul( 2 ).sub( 1 ).length().clamp( 0.5, 1 );
			vari = vari.add( oneMinus( nl ).div( nl ).mul( 1.5 ) );

		}

		const kernel = min( vari, 0.2 ).mul( oneMinus( smoothstep( 0.9, 0.99, normalWorld.y ) ) );
		const r0 = clamp( s.rough, 0.04, 1.0 );
		m.roughnessNode = pow( pow( r0, 4 ).add( kernel ), 0.25 );
		m.metalnessNode = s.metal;
		m.aoNode = ao;
		return m;

	}

	// ------------------------------------------------------------------ baked texture materials

	_baked( entry, params ) {

		const info = this.info.get( entry.name );
		const tex = this.tex.get( entry.name );
		if ( ! info || ! tex ) return this._special( this.catalog.get( 'plain' ), params, entry );
		const U = this.u;
		const m = new THREE.MeshStandardNodeMaterial();
		const c = this._common( params );
		const uvt = c.uvm.div( vec2( info.tile[ 0 ], info.tile[ 1 ] ) );
		const aT = texture( tex.albedo, uvt );
		const nT = texture( tex.normal, uvt );
		const oT = texture( tex.orm, uvt );
		let alb = aT.rgb;
		if ( entry.tintable ) {

			const mean = info.preview;
			alb = alb.mul( c.tint );
			if ( entry.name === 'canvas' && params.stripe ) {

				// awning stripes across U (period = stripe metres), second colour tint2 (default off-white)
				const t2 = params.tint2 ? new THREE.Color( params.tint2 ) : new THREE.Color( '#d9d2c0' );
				const f2 = [ t2.r, t2.g, t2.b ].map( ( t, i ) => Math.min( 6, t / Math.max( mean[ i ], 0.004 ) ) );
				const band = smoothstep( 0.47, 0.53, abs( fract( c.uvm.x.div( params.stripe ) ).sub( 0.5 ) ).mul( 2 ) );
				alb = mix( alb, aT.rgb.mul( vec3( f2[ 0 ], f2[ 1 ], f2[ 2 ] ) ), band );

			}

		}

		// ---- foliage (leaf, hedge)
		const foliage = entry.name === 'leaf' || entry.name === 'hedge';
		const leafy = foliage && params.leafy ? params.leafy : 0;
		let wEntry = entry;
		if ( leafy ) {

			// leaf-cluster cards: mask.w = which of the two leaf colours a card wears (tint .. tint2), mask.z = how deep
			// inside the crown it hangs (0 skin .. 1 heart: darker), so wear / special are not weathering here
			wEntry = { ...entry, wear: null, special: null };
			if ( params.tint2 ) {

				const t2 = new THREE.Color( params.tint2 ), mean = info.preview;
				const f2 = [ t2.r, t2.g, t2.b ].map( ( t, i ) => Math.min( 6, t / Math.max( mean[ i ], 0.004 ) ) );
				const k2 = smoothstep( 0.38, 0.62, c.mask.w.mul( 0.75 ).add( c.nHi.g.mul( 0.25 ) ) );
				alb = mix( alb, aT.rgb.mul( vec3( f2[ 0 ], f2[ 1 ], f2[ 2 ] ) ), k2 );

			}

			alb = alb.mul( c.nHi.b.sub( 0.5 ).mul( 0.5 ).add( 1 ) ).mul( mix( float( 1 ), float( 0.4 ), c.mask.z ) );

		}

		if ( foliage ) {

			// albedo guard: a green tint far above what a leaf reflects (lime, self-lit looking under a lantern) is pulled
			// back onto a leaf's range and greyed a little; flowers and autumn colours (not green-dominant) pass
			const l = luminance( alb ).add( 0.0005 );
			const green = smoothstep( 0.0, 0.25, alb.g.sub( max( alb.r, alb.b ) ).div( l ) );
			const kg = min( float( 1 ), l.sub( 0.1 ).max( 0 ).mul( 0.15 ).add( 0.1 ).div( l ) );
			alb = mix( alb, mix( alb, vec3( l ), smoothstep( 0.08, 0.3, l ).mul( 0.45 ) ).mul( kg ), green );

		}

		// world-space macro variation hides the tile repeat and gives large surfaces life
		const macro = c.nLo.r.sub( 0.5 ).mul( 0.3 ).add( c.nLo.b.sub( 0.5 ).mul( 0.14 ) ).mul( U.macro ).add( 1 );
		alb = alb.mul( macro );
		const s = this._weather( wEntry, c, {
			alb, rough: oT.g.mul( c.nLo.g.sub( 0.5 ).mul( 0.16 ).mul( U.macro ).add( 1 ) ), metal: oT.b, h: nT.a, flat: float( 0 ) } );
		// (soft: trowelled plaster under a lantern a pace away shows every lump of its relief like crumpled paper)
		const ns = /^plaster/.test( entry.name ) ? mix( float( 1 ), float( 0.5 ), U.soft ).mul( params.bump ?? 1 ) : ( params.bump ?? 1 );
		const nFlat = mix( nT.rgb, vec3( 0.5, 0.5, 1 ), s.flat );
		m.normalNode = normalMap( s.rip ? nFlat.add( vec3( s.rip, 0 ) ) : nFlat, typeof ns === 'number' ? vec2( ns, - ns ) : vec2( ns, ns.negate() ) );
		this._finish( m, c, s, oT.r, nT );
		if ( leafy ) {

			// every card covers one cell (leafy metres square) of its uv. The height of the leaf mosaic cuts it into
			// separate leaves, and the cut rises towards the cell's edge: a card ends in a ragged spray of leaves, never
			// in a straight edge. (From afar the height averages out and the spray closes into one soft tuft.)
			const cell = fract( c.uvm.mul( params.scale ?? 1 ).div( leafy ) ).sub( 0.5 ).mul( 2 );
			const d = length( cell ).add( c.nHi.r.sub( 0.5 ).mul( 0.3 ) );
			m.maskNode = nT.a.greaterThan( smoothstep( 0.45, 1.0, d ).mul( 0.8 ).add( 0.33 ) );
			m.setupLightingModel = () => new FoliageLighting();

		}

		m.emissiveNode = s.alb.mul( c.glow ).mul( U.emissive );
		if ( params.double ) m.side = THREE.DoubleSide;
		return m;

	}

	// ------------------------------------------------------------------ node-only materials

	_special( entry, params, standIn = null ) {

		const U = this.u;
		const name = entry.name;
		const c = this._common( params );
		const T = () => c.tint; // the default colour of each material is resolved in vertexValues()
		const m = new THREE.MeshStandardNodeMaterial();
		const grain = c.nHi.r.sub( 0.5 );
		const weatherEntry = standIn || entry;

		if ( name === 'plain' || name === 'metal_plain' ) {

			const metal = name === 'metal_plain';
			const s = this._weather( weatherEntry, c, {
				alb: T( metal ? '#c9c9c9' : '#808080' ).mul( grain.mul( 0.1 ).add( c.nLo.r.sub( 0.5 ).mul( 0.12 ) ).add( 1 ) ),
				rough: c.rough.add( grain.mul( 0.14 ) ).add( c.nLo.g.sub( 0.5 ).mul( 0.1 ) ),
				metal: float( params.metal ?? ( metal ? 1 : 0 ) ), h: c.nHi.g, flat: float( 0 ) } );
			this._finish( m, c, s );
			m.emissiveNode = s.alb.mul( c.glow ).mul( U.emissive );

		} else if ( name === 'emissive' || name === 'neon' ) {

			const glow = c.glow;
			let e = T( '#ffb45a' ).mul( glow ).mul( U.emissive );
			if ( name === 'neon' ) {

				// hot core, saturated rim (tube seen side-on is whiter in the middle)
				const ndv = abs( dot( normalize( normalWorld ), normalize( cameraPosition.sub( positionWorld ) ) ) );
				e = mix( e, vec3( glow ).mul( U.emissive ), pow( ndv, 3 ).mul( 0.55 ) );
				if ( params.flicker ) e = e.mul( step( 0.04, fract( sin( floor( time.mul( 18 ) ).mul( 12.9898 ) ).mul( 43758.5453 ) ) ).mul( 0.5 ).add( 0.5 ) );

			}

			m.colorNode = vec3( 0 );
			m.roughnessNode = float( 0.4 );
			m.metalnessNode = float( 0 );
			m.emissiveNode = e;

		} else if ( name === 'glass_frosted' ) {

			const col = T( '#efe6d2' );
			m.colorNode = col.mul( 0.8 ).mul( mix( float( 1 ), c.mask.x, 0.4 ) );
			m.roughnessNode = c.rough.add( grain.mul( 0.1 ) );
			m.metalnessNode = float( 0 );
			// lit globe: brighter where facing the viewer (thicker glowing volume behind), dirt dims it
			const ndv = abs( dot( normalize( normalWorld ), normalize( cameraPosition.sub( positionWorld ) ) ) );
			const dirt = oneMinus( c.mask.y.mul( 0.7 ) ).mul( c.nHi.g.mul( 0.25 ).add( 0.85 ) );
			m.emissiveNode = col.mul( c.glow ).mul( ndv.mul( 0.6 ).add( 0.4 ) ).mul( dirt ).mul( U.emissive );

		} else if ( name === 'glass_clear' || name === 'glass_lantern' ) {

			const lantern = name === 'glass_lantern';
			const col = T( lantern ? '#ffdca8' : '#b9c9c2' );
			const ndv = abs( dot( normalize( normalWorld ), normalize( cameraPosition.sub( positionWorld ) ) ) );
			const fres = pow( oneMinus( ndv ), 4 );
			const dirt = c.mask.y.mul( 0.8 ).add( c.nLo.r.sub( 0.5 ).mul( 0.25 ) ).add( c.nHi.b.sub( 0.5 ).mul( 0.2 ) ).saturate();
			// clear glass has next to no diffuse response: a pane beside a flame must not light up like paper
			m.colorNode = mix( col.mul( lantern ? 0.02 : 0.06 ), vec3( 0.12, 0.11, 0.09 ), dirt.mul( lantern ? 0.35 : 1 ) );
			m.roughnessNode = mix( float( 0.05 ), float( 0.55 ), dirt );
			m.metalnessNode = float( 0 );
			const cover = mix( c.opacity, float( 0.9 ), max( fres, dirt.mul( 0.7 ) ) );
			m.opacityNode = cover;
			// the flame's glare in the panes (soot and dirt catch more of it than clean glass): a lit lantern reads as a
			// lamp, not as a candle in a clear box. The pane is blended, so the glow is divided by its cover - what
			// reaches the picture sits just under the bloom threshold whatever the opacity.
			if ( lantern ) m.emissiveNode = col.mul( vec3( 1.0, 0.76, 0.46 ) ).mul( c.glow ).mul( dirt.mul( 0.6 ).add( 0.8 ).add( ndv.mul( 0.5 ) ) ).mul( c.nLo.g.mul( 0.5 ).add( 0.75 ) ).div( cover ).mul( U.emissive );
			else if ( this.probe && ! this._safe ) {

				// a pane mirrors the lit square: the plaza probe along the mirror direction, by fresnel. (Blended, so
				// the post pipeline's reflection pass does not see it; no gain until the probes are captured.)
				const PR = this.probe;
				const Nw = normalize( normalWorld ), Vw = normalize( cameraPosition.sub( positionWorld ) );
				const Nf = Nw.mul( dot( Nw, Vw ).sign() );
				const R = reflect( Vw.negate(), Nf );
				const env = cubeTexture( PR.p[ 2 ].rt.texture, vec3( R.x, max( R.y, - 0.05 ), R.z ) ).level( 1.5 ).rgb;
				m.emissiveNode = env.mul( PR.gain ).mul( pow( oneMinus( ndv ), 3 ).mul( 0.9 ).add( 0.1 ) ).mul( oneMinus( dirt.mul( 0.8 ) ) );

			}
			m.transparent = true;
			if ( ! this._safe ) m.mrtNode = NO_NORMAL;
			m.depthWrite = false;
			m.side = THREE.DoubleSide;

		} else if ( name === 'glass_stained' ) {

			// leaded diamond quarries, each pane a slightly different shade, glowing from inside
			const p = c.uvm.div( params.pane ?? 0.085 );
			const d = vec2( p.x.add( p.y ), p.x.sub( p.y ) );
			const cell = floor( d ), f = fract( d );
			const edge = min( min( f.x, oneMinus( f.x ) ), min( f.y, oneMinus( f.y ) ) );
			const lead = smoothstep( 0.035, 0.07, edge );
			const h1 = fract( sin( dot( cell, vec2( 127.1, 311.7 ) ) ).mul( 43758.5453 ) );
			const h2 = fract( sin( dot( cell, vec2( 269.5, 183.3 ) ) ).mul( 43758.5453 ) );
			const base = T( '#c89a3c' );
			const accent = col3( params.accent ?? '#7a1f1a' );
			const paneCol = mix( base.mul( h1.mul( 0.7 ).add( 0.55 ) ), accent, step( 0.82, h2 ) );
			m.colorNode = mix( vec3( 0.03 ), paneCol.mul( 0.2 ), lead );
			m.roughnessNode = mix( float( 0.6 ), float( 0.12 ), lead );
			m.metalnessNode = float( 0 );
			m.emissiveNode = paneCol.mul( lead ).mul( c.glow ).mul( c.nLo.g.mul( 0.6 ).add( 0.7 ) ).mul( U.windowGlow );

		} else if ( name === 'window_dark' || name === 'window_lit' ) {

			// old glass: slightly wavy reflections
			// (hand-span waves across a pane: the mirror image bends but stays an image; finer noise turns it to confetti)
			const wav = texture( this.noise, c.uvm.mul( 0.25 ) ).level( 3 ).rg.sub( 0.5 );
			m.normalNode = normalMap( vec3( wav.x.mul( 0.03 ).add( 0.5 ), wav.y.mul( 0.03 ).add( 0.5 ), 1 ) );
			const dirt = c.mask.y.mul( 0.8 ).add( c.nHi.b.sub( 0.5 ).mul( 0.2 ) ).saturate();
			m.colorNode = mix( vec3( 0.012, 0.014, 0.018 ), vec3( 0.07, 0.065, 0.055 ), dirt );
			m.roughnessNode = mix( float( 0.06 ), float( 0.5 ), dirt );
			m.metalnessNode = float( 0 );
			if ( name === 'window_lit' ) {

				// fake interior: parallax-shifted back wall + lamp pool + sheer curtain layer.
				const N = normalize( normalWorld );
				const V = normalize( cameraPosition.sub( positionWorld ) );
				const up = vec3( 0, 1, 0 );
				const Tn = normalize( cross( up, N ) );
				const vz = max( dot( V, N ), 0.12 );
				const par = vec2( dot( V, Tn ), dot( V, up ) ).div( vz ).negate(); // ray slope going into the room
				const seed = c.mask.w; // per-window random 0..1 painted by the asset
				const depth = float( params.depth ?? 2.6 );
				const pw = c.uvm.add( par.mul( depth ) ); // hit on the back wall (metres)
				const cellW = float( 2.9 );
				const cx = floor( pw.x.div( cellW ).add( seed.mul( 7.0 ) ) );
				const fx = fract( pw.x.div( cellW ).add( seed.mul( 7.0 ) ) );
				const r1 = fract( sin( cx.mul( 91.7 ).add( seed.mul( 311.7 ) ) ).mul( 43758.5453 ) );
				const r2 = fract( sin( cx.mul( 47.3 ).add( seed.mul( 157.1 ) ) ).mul( 24634.6345 ) );
				// wall: warm wallpaper, darker towards the ceiling, dado rail, picture frame, shelf shadows
				const wy = pw.y;
				const wall = mix( vec3( 0.55, 0.36, 0.17 ), vec3( 0.62, 0.46, 0.3 ), r1 ).mul( mix( float( 1 ), float( 0.45 ), smoothstep( 0.2, 1.6, wy.sub( seed.mul( 0.5 ) ) ) ) );
				const lampP = vec2( r1.mul( 0.6 ).add( 0.2 ), r2.mul( 0.5 ).sub( 0.25 ) );
				const ld = length( vec2( fx.sub( lampP.x ).mul( cellW ), fract( wy.div( 2.7 ) ).sub( 0.5 ).mul( 2.7 ).sub( lampP.y ) ) );
				const lamp = exp( ld.mul( ld ).mul( - 1.6 ) ).mul( 2.2 );
				const frame = step( abs( fx.sub( 0.5 + 0.2 ).mul( cellW ) ), 0.28 ).mul( step( abs( fract( wy.div( 2.7 ) ).sub( 0.62 ).mul( 2.7 ) ), 0.2 ) );
				const furn = smoothstep( 0.1, 0.0, fract( wy.div( 2.7 ) ).sub( r2.mul( 0.25 ).add( 0.12 ) ) ).mul( step( fx, 0.45 ) );
				let room = wall.mul( lamp.add( 0.35 ) ).mul( oneMinus( frame.mul( 0.55 ) ) ).mul( oneMinus( furn.mul( 0.75 ) ) );
				// curtain layer just behind the glass (20 cm): folds + gathered to the sides
				const pc = c.uvm.add( par.mul( 0.2 ) );
				const fold = sin( pc.x.mul( 55 ).add( sin( pc.x.mul( 13 ) ).mul( 2 ) ) ).mul( 0.5 ).add( 0.5 );
				const cur = c.curtain;
				const curMask = smoothstep( 0.0, 0.12, cur.sub( 0.5 ).mul( 2.4 ).add( sin( pc.x.mul( 2.2 ).add( seed.mul( 40 ) ) ).mul( 0.5 ) ).add( fold.mul( 0.15 ) ) );
				const curCol = mix( vec3( 0.85, 0.62, 0.36 ), vec3( 0.7, 0.3, 0.16 ), step( 0.6, r2 ) ).mul( fold.mul( 0.5 ).add( 0.5 ) ).mul( 0.75 );
				room = mix( room, curCol, curMask.mul( 0.82 ) );
				const tintC = T( '#ffc27a' );
				const level = mix( float( 0.55 ), float( 1.25 ), fract( seed.mul( 13.37 ) ) );
				m.emissiveNode = room.mul( tintC ).mul( c.glow ).mul( level ).mul( oneMinus( dirt.mul( 0.7 ) ) ).mul( U.windowGlow );

			}

		} else if ( name === 'water' ) {

			const t = time.mul( params.speed ?? 0.05 );
			// Still water (canal, basins). UV in metres. Ripples a hand to an arm across drifting two ways over a slow
			// swell; the surface tilts 1-2 degrees (peaks ~4). Seen at a grazing angle that is all it takes to draw every
			// lamp out into a long wavering vertical streak, while the mirror image stays an image (12 degrees of
			// knee-sized blobs, as before, crumples it into foil). params.ripple scales the tilt (0.5 = canal).
			const rp = ( params.ripple ?? 0.5 ) * 2;
			const n0 = texture( this.noise, c.uvm.mul( 0.12 ).add( vec2( t.mul( 0.3 ), t.mul( - 0.2 ) ) ) ).gb.sub( 0.5 );
			const n1 = texture( this.noise, c.uvm.mul( 1.1 ).add( vec2( t, t.mul( 0.6 ) ) ) ).rg.sub( 0.5 );
			const n2 = texture( this.noise, c.uvm.mul( 2.9 ).add( vec2( t.mul( - 1.3 ), t.mul( 1.9 ) ) ) ).ba.sub( 0.5 );
			const w = n1.add( n2.mul( 0.7 ) ).add( n0.mul( 0.2 ) ).mul( rp * 0.06 );
			m.normalNode = normalMap( vec3( w.x.add( 0.5 ), w.y.add( 0.5 ), 1 ) );
			m.colorNode = T( '#0b1512' ).mul( 0.6 );
			// opaque, so it writes its rippled normal and gloss to the surface buffers and takes screen-space reflections.
			// Where the ripples fall below a pixel (far water) their tilt turns into blur instead of vanishing.
			const fw = c.uvm.fwidth();
			m.roughnessNode = mix( float( 0.07 ), float( 0.15 ), smoothstep( 0.015, 0.25, max( fw.x, fw.y ) ) );
			m.metalnessNode = float( 0 );
			// the mirror image of the town (screen-space hit or reflection probe) is added by the post pipeline from
			// the rippled normal and the gloss written here

		} else if ( name === 'water_fall' ) {

			// Falling water: a clear sheet that shows (and bends) what is behind it, glints along the rolled edges of its
			// strands, and is white only where it is torn to spray or painted as foam.
			// GEOMETRY CONTRACT - everything this shader reads from the mesh (keep these when remodelling a fall):
			//   uv        U = metres across the fall, V = metres along the flow, up from the foot. The pattern scrolls
			//             down V at 2-3 m/s; strands are ~4 cm wide, so U must be continuous across a sheet.
			//   _mask.z   (wear) break-up: 0 = glassy unbroken sheet at the lip .. 1 = threads and drops at the foot.
			//   _mask.w   (special) foam: 1 = opaque white water (impact ring, aerated foot). Keep it 0 on the sheet.
			//   material  tint = colour of the foam / faint tint of the body, glow = sparkle of the drops (0.2-0.4),
			//             opacity = overall cover of the strands (0.9-1), speed = flow rate (default 1.6).
			// Without the masks the sheet still tears into strands, evenly, with clear edges and no foam.
			const t = time.mul( params.speed ?? 1.6 );
			const u0 = c.uvm.x, v = c.uvm.y;
			const brk = c.mask.z.saturate(), white = c.mask.w.saturate();
			// strands wander sideways as they fall, more once they have torn free
			const u = u0.add( texture( this.noise, vec2( u0.mul( 0.9 ), v.mul( 0.31 ).add( t.mul( 0.21 ) ) ) ).a.sub( 0.5 ).mul( brk.mul( 0.05 ).add( 0.012 ) ) );
			const s1 = texture( this.noise, vec2( u.mul( 3.1 ), v.mul( 0.23 ).add( t.mul( 0.42 ) ) ) ); // ribs of the sheet (4 x 50 cm)
			const s2 = texture( this.noise, vec2( u.mul( 8.3 ).add( 0.37 ), v.mul( 0.55 ).add( t.mul( 0.9 ) ) ) ); // threads
			const s3 = texture( this.noise, vec2( u.mul( 1.3 ).add( 0.71 ), v.mul( 1.1 ).add( t.mul( 1.25 ) ) ) ); // tears travelling down
			const body = s1.r.mul( 0.45 ).add( s2.g.mul( 0.25 ) ).add( s3.b.mul( 0.3 ) );
			// erosion: the threshold climbs with the break-up, so the whole sheet opens into strands and then threads
			const thr = mix( float( 0.36 ), float( 0.58 ), brk );
			const a = smoothstep( thr, thr.add( 0.05 ), body );
			const rim = a.mul( oneMinus( smoothstep( thr.add( 0.04 ), thr.add( 0.13 ), body ) ) ); // the rolled edge of a strand
			// white water: painted foam, torn edges once the sheet has broken up, the aerated foot
			const foam = white.add( rim.mul( brk ).mul( 0.9 ) ).add( smoothstep( 0.7, 1.0, brk ).mul( 0.35 ) ).saturate();
			// relief of the strands (thickness gradient in UV metres, from screen derivatives): the highlight of a lamp
			// runs as a thin line down each edge instead of lighting the sheet evenly
			const hgt = smoothstep( thr.sub( 0.02 ), thr.add( 0.2 ), body );
			const hx = hgt.dFdx(), hy = hgt.dFdy(), ux = c.uvm.dFdx(), uy = c.uvm.dFdy();
			const det = ux.x.mul( uy.y ).sub( ux.y.mul( uy.x ) );
			const idet = det.sign().div( max( abs( det ), 1e-9 ) ).mul( 0.012 );
			const gu = clamp( hx.mul( uy.y ).sub( hy.mul( ux.y ) ).mul( idet ), - 0.6, 0.6 );
			const gv = clamp( hy.mul( ux.x ).sub( hx.mul( uy.x ) ).mul( idet ), - 0.6, 0.6 );
			const slope = vec2( gu.negate().add( s2.r.sub( 0.5 ).mul( 0.3 ) ), gv.add( s1.b.sub( 0.5 ).mul( 0.15 ) ) );
			m.normalNode = normalMap( vec3( slope.mul( 0.5 ).add( 0.5 ), 1 ) );
			// clear water has no diffuse response: only foam is lit like a surface
			m.colorNode = T( '#cfe0e6' ).mul( foam.mul( 0.8 ).add( 0.03 ) );
			m.roughnessNode = mix( float( 0.08 ), float( 0.6 ), foam );
			m.metalnessNode = float( 0 );
			// sparkle: a few bright drops riding the broken strands; glow is the asset's dial
			const bead = smoothstep( 0.8, 0.92, s2.b.mul( 0.6 ).add( s1.a.mul( 0.4 ) ) ).mul( a ).mul( brk.mul( 0.8 ).add( 0.2 ) );
			const spark = T( '#cfe0e6' ).mul( bead ).mul( c.glow ).mul( U.emissive );
			if ( this.refraction && this.probe && ! this._safe ) {

				// the strand is drawn at full cover and carries its own transparency: the frame behind it, bent by the
				// relief, plus the mirror image of the square by fresnel (strongest on the rolled edges)
				const PR = this.probe;
				const Nw = normalize( normalWorld ), Vw = normalize( cameraPosition.sub( positionWorld ) );
				const ndv = abs( dot( Nw, Vw ) );
				const fres = pow( oneMinus( ndv ), 5 ).mul( 0.96 ).add( 0.04 ).add( rim.mul( 0.1 ) ).min( 1 );
				const bend = normalView.xy.sub( normalViewGeometry.xy ).mul( vec2( 1, - 1 ) ).mul( float( 0.05 ).div( positionView.z.negate().max( 0.6 ) ) );
				const behind = viewportSharedTexture( screenUV.add( bend ) ).rgb;
				const R = reflect( Vw.negate(), Nw.mul( dot( Nw, Vw ).sign() ) );
				const env = cubeTexture( PR.p[ 2 ].rt.texture, vec3( R.x, max( R.y, - 0.05 ), R.z ) ).level( 2.5 ).rgb.mul( PR.gain );
				const clear = behind.mul( mix( vec3( 1 ), T( '#cfe0e6' ), 0.3 ) ).mul( oneMinus( fres ) ).add( env.mul( fres ) );
				m.emissiveNode = clear.mul( oneMinus( foam ) ).add( spark );
				m.opacityNode = a.mul( c.opacity );

			} else {

				// (light tiers, reflection probes: plain blending, a thin veil with opaque foam)
				m.emissiveNode = spark;
				m.opacityNode = a.mul( mix( float( 0.22 ), float( 0.95 ), max( foam, bead ) ) ).mul( c.opacity );

			}
			m.transparent = true;
			if ( ! this._safe ) m.mrtNode = NO_NORMAL;
			m.depthWrite = false;
			m.side = THREE.DoubleSide;

		} else {

			this.warnings.add( `special material '${name}' has no shader (plain used)` );
			return this._special( this.catalog.get( 'plain' ), params );

		}

		if ( params.double ) m.side = THREE.DoubleSide;
		return m;

	}

	// neutral clay for form review (vertex AO only)
	clay() {

		if ( this._clay ) return this._clay;
		const m = new THREE.MeshStandardNodeMaterial();
		const mask = attribute( '_mask', 'vec4' );
		m.colorNode = vec3( 0.5, 0.49, 0.47 ).mul( mix( float( 0.3 ), float( 1 ), mask.x ) );
		m.roughnessNode = float( 0.75 );
		m.metalnessNode = float( 0 );
		m.side = THREE.DoubleSide;
		this._clay = m;
		return m;

	}

	// debug view of the weathering masks: R grime, G wear, B special, value = AO
	maskDebug() {

		if ( this._maskDbg ) return this._maskDbg;
		const m = new THREE.MeshBasicNodeMaterial();
		const mask = attribute( '_mask', 'vec4' );
		m.colorNode = vec3( mask.y, mask.z, mask.w ).mul( 0.85 ).add( 0.15 ).mul( mask.x );
		m.side = THREE.DoubleSide;
		this._maskDbg = m;
		return m;

	}

}
