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
import { diffuseContribution, faceDirection } from 'three/tsl'; // (foliage lighting)
import { sourceRadius } from './ClusteredLightsNode.js'; // (lantern glare)
import { positionGeometry } from 'three/tsl'; // (grime at the foot of walls)

// blended surfaces must not write into the normal G-buffer (it feeds GTAO): alpha 0 leaves it untouched
export const NO_NORMAL = mrt( { normal: vec4( 0, 0, 0, 0 ), mr: vec4( 0, 0, 0, 0 ) } );

const lin = ( hex ) => new THREE.Color( hex ); // THREE.Color parses hex as sRGB -> linear working space
// constant colour node. NOT vec3( THREE.Color ): TSL reads .x/.y/.z of the value, which a Color lacks, and the
// constant silently compiles to black.
const col3 = ( hex ) => { const c = lin( hex ); return vec3( c.r, c.g, c.b ); };

// Foliage (leaf, hedge) and bark at night. Shaded as plain surfaces a tree is a black cut-out: its crown hangs above
// the lantern heads (the capped share of each lamp, then the reach falloff), the moon is shadowed by the crown's own
// cards and the sky fill is multiplied away by the painted occlusion of the crown. What this model adds:
//  - wrap and transmission: a card stands for a spray of small blades at every angle and a blade lets light through,
//    so every card takes a share of each light whatever its own facing, and more of it (yellower: it has passed
//    through the blade) when lit from behind - a crown lit from below shades as one soft volume;
//  - reach: a lamp's light is cut short for the pools on the paving (reachFactor); what hangs above the lamp head
//    is no part of a pool and keeps part of what the cut took (folReach: 1 = cut like everything else, 0 = pure
//    inverse square). Below the lamp head (hedges, flower beds, trunks) nothing changes;
//  - bounce: above a capped lantern only the share `up` of its light arrives directly. The rest went down onto the
//    paving and a part of it comes back up, warm and diffuse: undersides of a crown over a lit street glow dimly;
//  - sky: a floor of sky / moon light on top of the hemisphere fill, strongest on what faces up, a transmitted share
//    below, thinned towards the heart of the crown (open) but not by the occlusion that kills the hemisphere term.
//    Like the roofs (the engine's roofFill) a crown stands in the open sky while the street below sees a strip of
//    it: the floor grows with height above the plant's own foot (x 0.2 for a hedge or a flower bed, x 0.6 at the
//    underside of a street tree's crown, x 1.4 at its top 8 m up - which also gives the crown its form).
// Everything scales with the lights and the sky fill: no light, no foliage.
class FoliageLighting extends THREE.PhysicalLightingModel {

	constructor( U, o = {} ) {

		super();
		this.U = U;
		this.wrap = o.wrap ?? 1; // share of the wrap term (clipped masses and bark: less than loose leaf sprays)
		this.back = o.back ?? 1; // share of the transmitted term (bark: none)
		this.tint = o.tint ?? vec3( 1.0, 0.95, 0.6 ); // colour of light that passed through a blade
		this.open = o.open ?? float( 1 ); // how much of the sky this point sees (1 skin of the crown .. its heart)
		this.sky = o.sky ?? 1;
		this.bounce = o.bounce ?? 1;
		this.alb = o.alb ?? diffuseContribution; // albedo the sky floor uses (without the painted occlusion)
		this.under = o.under ?? null; // how much a point faces the paving (default: by its normal; a crown of cards has none that do)

	}

	direct( input, builder ) {

		const U = this.U;
		const { lightNode, lightDirection, reflectedLight } = input;
		// clustered lamps hand their reach / cap factors over, the shadow-casting pool lights carry them (capLight)
		const lamp = input.lampReach ? input : ( lightNode && lightNode.light && lightNode.light.lampNodes ) || null;
		let lightColor = input.lightColor;
		if ( lamp ) lightColor = lightColor.mul( mix( float( 1 ), mix( float( 1 ), lamp.lampReach, U.folReach ).div( lamp.lampReach.max( 1e-3 ) ), lamp.lampAbove ) );
		super.direct( lamp ? { ...input, lightColor } : input, builder );
		const nl = normalView.dot( lightDirection );
		const k = oneMinus( saturate( nl ) ).mul( U.folWrap.mul( this.wrap ) ).add( saturate( nl.negate() ).mul( U.folBack.mul( this.back ) ) ).mul( 1 / Math.PI );
		reflectedLight.directDiffuse.addAssign( lightColor.mul( diffuseContribution ).mul( this.tint ).mul( k ) );
		if ( lamp ) {

			// what the cap withheld, back from the paving: lightColor / cap is the uncapped lamp, ( 1 - cap ) its share
			// that went down instead. Diffuse (it comes from the whole lit pool, not from the lamp), most on what faces down.
			const held = oneMinus( lamp.lampCap ).div( lamp.lampCap.max( 0.05 ) );
			const down = this.under ?? normalWorld.y.mul( - 0.5 ).add( 0.5 );
			reflectedLight.directDiffuse.addAssign( lightColor.mul( held ).mul( vec3( 1.0, 0.84, 0.62 ) ).mul( diffuseContribution )
				.mul( mix( float( 0.4 ), float( 1 ), down ).mul( U.folBounce ).mul( this.bounce / Math.PI ) ) );

		}

	}

	indirect( builder ) {

		super.indirect( builder );
		const U = this.U;
		const up = normalWorld.y.mul( 0.5 ).add( 0.5 );
		const high = mix( float( 0.2 ), float( 0.6 ), smoothstep( 2.0, 4.5, positionGeometry.y ) ).add( smoothstep( 4.5, 8.5, positionGeometry.y ).mul( 0.8 ) );
		const w = mix( U.folUnder, float( 1 ), up.mul( up ) ).mul( this.open ).mul( high ).mul( this.sky );
		builder.context.reflectedLight.indirectDiffuse.addAssign( U.skyFill.mul( U.folSky ).mul( w ).mul( this.alb ).mul( 1 / Math.PI ) );

	}

}

// Lantern panes: the mantle's glare. A pane that glows evenly is a paper card; a lit gas lantern is white-hot where
// the mantle is seen through the glass and falls to amber towards the glazing bars. Every light closer than a pace
// (the lantern's own burner) adds a glare disc around its sightline: a tight core over a wider body that the dirt on
// the glass catches. The pane is blended, so the glow is divided by its cover like the emissive term.
class LanternLighting extends THREE.PhysicalLightingModel {

	constructor( cover, dirt, lit ) {

		super();
		this.cover = cover; this.dirt = dirt; this.lit = lit;

	}

	direct( input, builder ) {

		const { lightNode, lightColor, reflectedLight } = input;
		// clustered lights hand their light vector over; three's own point lights (the shadow-casting pool) are asked
		const Lv = input.lightVector || ( lightNode && lightNode.light && lightNode.light.isPointLight ? lightNode.getLightVector( builder ) : null );
		if ( ! Lv ) return super.direct( input, builder );
		const d2 = dot( Lv, Lv );
		// (the burner's own highlight on the inside of the far panes - two white dots in the corners, brighter than the
		// mantle - is left to the glare)
		super.direct( { ...input, lightColor: lightColor.mul( smoothstep( 0.1, 1.0, d2 ).mul( 0.9 ).add( 0.1 ) ) }, builder );
		const along = dot( Lv, normalize( positionView ) );
		const r2 = d2.sub( along.mul( along ) ).max( 0 ); // (distance of the burner from this pixel's line of sight)^2
		const peak = max( lightColor.r, max( lightColor.g, lightColor.b ) ).max( 1e-4 );
		const hue = lightColor.div( peak );
		const on = smoothstep( 0.05, 0.8, peak.mul( d2.add( sourceRadius.mul( sourceRadius ) ) ) ).mul( smoothstep( 1.0, 0.2, d2 ) ).mul( this.lit );
		const core = exp( r2.mul( - 1 / ( 2 * LANTERN.core * LANTERN.core ) ) );
		const body = pow( r2.div( LANTERN.body * LANTERN.body ).add( 1 ), - 1.5 );
		const glare = mix( hue, vec3( 1 ), 0.45 ).mul( core.mul( LANTERN.coreGain ) )
			.add( hue.mul( vec3( 1.0, 0.8, 0.52 ) ).mul( body.mul( LANTERN.bodyGain ) ).mul( this.dirt.mul( 0.6 ).add( 0.8 ) ) );
		reflectedLight.directDiffuse.addAssign( glare.mul( on ).div( this.cover ) );

	}

}

// Stained glass with a lamp at it (a glazed lantern like the monument's star, a window with its lamp a step away):
// the glass passes that lamp's light whatever its facing, so the panes nearest the lamp burn pale and hot and the
// glow falls off across the glazing with the lamp's own falloff, instead of every pane sitting at one flat level.
// Driven by the light, so a brighter lamp in the layout is a brighter lantern.
class StainedLighting extends THREE.PhysicalLightingModel {

	constructor( glass ) {

		super();
		this.glass = glass;

	}

	direct( input, builder ) {

		super.direct( input, builder );
		const { lightNode, lightColor, reflectedLight } = input;
		const Lv = input.lightVector || ( lightNode && lightNode.light && lightNode.light.isPointLight ? lightNode.getLightVector( builder ) : null );
		if ( ! Lv ) return;
		const near = smoothstep( STAINED.reach * STAINED.reach, STAINED.reach * STAINED.reach * 0.25, dot( Lv, Lv ) );
		const pale = mix( this.glass, vec3( 1.0, 0.9, 0.62 ).mul( max( this.glass.r, max( this.glass.g, this.glass.b ) ) ), STAINED.pale );
		reflectedLight.directDiffuse.addAssign( lightColor.mul( pale ).mul( near.mul( STAINED.back ) ) );

	}

}

// stained glass look: share of a lamp's light the glass passes, how far from the lamp (m), how much paler than the pane
const STAINED = { back: 0.16, reach: 1.6, pale: 0.65 };

// lantern glass look: glare core / body radius (m) and scene-linear gain, share of the old even pane glow that is kept
const LANTERN = { core: 0.04, body: 0.085, coreGain: 3.5, bodyGain: 1.2, base: 0.3 };

// weather on masonry and plaster that needs no painted mask: height of the damp foot (m, + 0.3), its and the rain
// runs' strength, depth of the metres-wide tonal patches
const MASONRY = /^(ashlar|rubble|brick|stone|plaster|mortar)/;
const WASH = { foot: 0.9, footGain: 0.9, runGain: 0.6, patch: 0.45 };

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
			wetFlat: uniform( 0.0 ), // 0..1 how far the film of rain water on level paving drowns its relief: at 0 every cobble glints on its own and a lantern is mirrored as a scatter of specks, towards 1 the wet stones share one surface and it is mirrored as a streak
			poolCap: uniform( 1.0 ),
			tideY: uniform( - 1e4 ), tideH: uniform( 0.4 ), // world height of standing water and how far above it masonry stays dark and wet (off by default)
			// foliage / bark lighting (FoliageLighting). skyFill is the sky half of the ambient fill, synced by the engine
			// (black where no engine does: no extra light)
			skyFill: uniform( new THREE.Color( 0, 0, 0 ) ),
			folSky: uniform( 6.0 ), // extra sky / moon floor on foliage, x skyFill (bark: x 0.5)
			folUnder: uniform( 0.35 ), // share of it that reaches faces looking down (through the leaves)
			folWrap: uniform( 0.3 ), folBack: uniform( 0.3 ), // wrap / transmitted share of every direct light
			folReach: uniform( 0.5 ), // how much of the lamp reach falloff foliage keeps (1 = all, like every other surface)
			folBounce: uniform( 0.3 ), // share of a capped lantern's withheld light that returns from the paving onto foliage above it
			folAO: uniform( 0.45 ), // floor of the painted occlusion on foliage (0 = as painted: the heart of a crown takes no fill)
			folGTAO: uniform( 0.6 ), // share of the screen-space occlusion left off foliage
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
		if ( entry && ( m.mrtNode || ( ud.probe && entry.kind !== 'baked' ) ) ) {

			this._safe = true;
			const v = entry.kind === 'baked' ? this._baked( entry, ud.tsq.params ) : this._special( entry, ud.tsq.params ); // (baked: masonry, whose knee is its own MRT output)
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
				// (sheet: the film that stands over the stones themselves where the paving is level and soaked)
				const sheet = k.mul( U.wetFlat ).mul( smoothstep( 0.9, 0.99, normalWorld.y ) );
				s.rough = mix( mix( mix( s.rough, s.rough.mul( 0.45 ), k ), min( s.rough.mul( 0.45 ), U.filmRough ), sheet ), float( sp.rough ), puddle );
				s.flat = max( s.flat, max( puddle.mul( 0.95 ), sheet.mul( 0.9 ) ) );
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
			s.alb = s.alb.mul( mix( float( 1 ), float( entry.rainDark ?? 0.58 ), k ) ); // (rainDark: wet leaves shine, they do not soak dark like stone)
			// (a pitched roof sheds its water: less sheen than a level slab)
			s.rough = mix( mix( mix( s.rough, s.rough.mul( mix( float( 0.72 ), float( 0.45 ), smoothstep( 0.85, 0.98, ny ) ) ), k ), min( s.rough, U.filmRough ), film.mul( 0.85 ) ), float( 0.07 ), puddle );
			s.flat = max( s.flat, max( puddle.mul( 0.95 ), film.mul( U.wetFlat ).mul( 0.9 ) ) );
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
			if ( MASONRY.test( entry.name ) ) {

				// what no asset paints: the weather a wall has stood in. Splash and rising damp at its foot (height in the
				// asset's own frame: a building stands on its origin), ragged at the top; rain runs down the face in
				// clusters, thickest under whatever sheds water onto it (the painted occlusion under sills and copings);
				// and a few metres across the stone is a little darker here, paler there, so no large face is one colour.
				const broad = texture( this.noise, vec2( c.wp.x.mul( 1.1 ).add( c.wp.z.mul( 0.9 ) ).mul( 0.034 ), c.wp.y.mul( 0.05 ) ).add( vec2( 0.21, 0.63 ) ) );
				const top = nLo.b.mul( WASH.foot ).add( 0.3 );
				const foot = smoothstep( top, top.mul( 0.12 ), positionGeometry.y ).mul( wall );
				const runs = smoothstep( 0.52, 0.86, streak.add( broad.r.sub( 0.5 ).mul( 0.7 ) ).add( oneMinus( mask.x ).mul( 0.6 ) ) ).mul( wall );
				const kw = foot.mul( nHi.b.mul( 0.5 ).add( 0.75 ) ).mul( WASH.footGain ).add( runs.mul( WASH.runGain ) ).saturate().mul( U.grime );
				s.alb = mix( s.alb, s.alb.mul( vec3( 0.4, 0.42, 0.37 ) ).add( gc.mul( 0.6 ) ), kw ).mul( broad.g.sub( 0.5 ).mul( WASH.patch ).mul( U.grime ).add( 1 ) );
				s.rough = mix( s.rough, float( g.rough ), kw.mul( 0.6 ) );

			}

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
		const bark = entry.name === 'bark';
		let wEntry = foliage ? { ...entry, rainDark: leafy ? 0.88 : 0.7 } : entry;
		if ( leafy ) {

			// leaf-cluster cards: mask.w = which of the two leaf colours a card wears (tint .. tint2), mask.z = how deep
			// inside the crown it hangs (0 skin .. 1 heart: darker), so wear / special are not weathering here
			wEntry = { ...wEntry, wear: null, special: null };
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
			// the cards carry the normals of the crown as a volume (outwards and up), whichever way a card happens to
			// be wound. Flipped on its back face (three's rule for two-sided surfaces) such a normal points into the
			// crown and down: half the cards of a moonlit top shade as undersides. Both faces keep the authored normal.
			if ( params.double ) m.normalNode = m.normalNode.mul( faceDirection );

		}

		if ( foliage || bark ) {

			// night lighting of trees (see FoliageLighting). The painted occlusion of a crown is near black at its heart
			// (median 0.1 on the cards of a street tree): as a factor on all light it leaves a cut-out, so it keeps a floor.
			const vao = mix( U.folAO, float( 1 ), c.mask.x );
			if ( foliage ) {

				m.colorNode = s.alb.mul( mix( float( 1 ), vao, U.aoDirect ) ).mul( mix( float( 1 ), oT.r, 0.5 ) );
				m.aoNode = vao.mul( mix( U.folAO, float( 1 ), oT.r ) );
				if ( this.aoRelief && ! this._safe ) m.mrtNode = this.aoRelief( U.folGTAO );

			} else {

				// (bark: the limbs inside the crown carry the same painted occlusion as its leaves)
				m.colorNode = s.alb.mul( mix( float( 1 ), vao, U.aoDirect ) ).mul( mix( float( 1 ), oT.r, 0.5 ) );
				m.aoNode = vao.mul( oT.r );

			}

			const opts = leafy ? { alb: s.alb, open: mix( float( 1 ), float( 0.35 ), c.mask.z ), under: float( 0.7 ) }
				: foliage ? { alb: s.alb, wrap: 0.6, back: 0.6, open: mix( float( 0.5 ), float( 1 ), c.mask.x ) }
					: { wrap: 0.8, back: 0, tint: vec3( 1, 1, 1 ), sky: 0.5, bounce: 0.5, open: mix( float( 0.5 ), float( 1 ), c.mask.x ) };
			m.setupLightingModel = () => new FoliageLighting( U, opts );

		}

		m.emissiveNode = s.alb.mul( c.glow ).mul( U.emissive );
		if ( params.double ) m.side = THREE.DoubleSide;
		if ( this.kneeWith && ! this._safe && MASONRY.test( entry.name ) ) {

			// lamp-lit stone keeps its joints, stains and block-to-block tone through the highlight knee (see the engine)
			const pv = info.preview, ref = entry.tintable ? luminance( vec3( pv[ 0 ], pv[ 1 ], pv[ 2 ] ).mul( c.tint ) ) : float( 0.2126 * pv[ 0 ] + 0.7152 * pv[ 1 ] + 0.0722 * pv[ 2 ] );
			m.mrtNode = this.kneeWith( clamp( luminance( m.colorNode ).div( ref.max( 0.02 ) ), 0.3, 1.15 ) );

		}

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
			// (the even part is the amber of the pane's edges; the hot middle is the mantle's glare, see LanternLighting.
			// Far lanterns may not own a light - fewer lights than lanterns on the smaller tiers - and are a few pixels
			// across: they keep the full even glow)
			if ( lantern ) {

				const even = mix( float( LANTERN.base ), float( 1 ), smoothstep( 35, 70, length( positionView ) ) );
				m.emissiveNode = col.mul( vec3( 1.0, 0.62, 0.3 ) ).mul( c.glow ).mul( dirt.mul( 0.6 ).add( 0.8 ).add( ndv.mul( 0.5 ) ) ).mul( c.nLo.g.mul( 0.5 ).add( 0.75 ) ).mul( even ).div( cover ).mul( U.emissive );
				const lit = saturate( c.glow.mul( 4 ) ).mul( U.emissive );
				m.setupLightingModel = () => new LanternLighting( cover, dirt, lit );

			} else if ( this.probe && ! this._safe ) {

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
			const glass = paneCol.mul( lead ).mul( saturate( c.glow.mul( 4 ) ) ).mul( U.windowGlow );
			m.setupLightingModel = () => new StainedLighting( glass );

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
			//
			// The calm surface is a handful of long smooth wave trains (0.35 - 2 m, crests bent by a slow noise so they
			// never line up) plus one soft noise layer: no fine grain, so the mirror image wavers instead of crumbling.
			// GEOMETRY CONTRACT (all optional; a mesh without them - the canal - is plain still water):
			//   _mask.z (wear)    agitation where falling water lands: 1 at the landing, falling LINEARLY to 0 over 1.2 m.
			//                     Rings run outwards along its gradient and the chop rises with it.
			//   _mask.w (special) foam: 1 = churned white water at a landing, fading to 0 over a hand or two (lacy edge).
			//   _mask.y (grime)   > 0.5 = light of a submerged lamp in the water (needs material glow > 0).
			//   material          ripple = tilt of the calm surface, depth (m) = shallow basin whose lit bed shows through,
			//                     glow = strength of the submerged-lamp light.
			const rp = ( params.ripple ?? 0.5 ) * 2;
			const fw = c.uvm.fwidth(), fwm = max( fw.x, fw.y );
			const agit = c.mask.z.saturate(), foamM = c.mask.w.saturate();
			const n0 = texture( this.noise, c.uvm.mul( 0.07 ).add( vec2( t.mul( 0.3 ), t.mul( - 0.2 ) ) ) ).bias( 2 );
			const n1q = texture( this.noise, c.uvm.mul( 0.45 ).add( vec2( t.mul( 0.5 ), t.mul( 0.3 ) ) ) ).bias( 1.5 );
			const n1 = n1q.rg.sub( 0.5 );
			let w = vec2( 0 );
			// [ direction, wavelength m, speed m/s, slope, crest wander (rad), gust channel ]: slow enough to read as still
			// water. Each train swells and dies in its own patches and its crests wander, the short ones most, so a bright
			// reflection breaks into loose squiggles and never into even stripes.
			for ( const [ dx, dy, lam, sp, amp, wob, gust ] of [ [ 0.92, 0.39, 1.9, 0.2, 0.006, 0.6, n0.b ], [ - 0.47, 0.88, 1.15, 0.16, 0.006, 1.0, n0.r ],
				[ 0.31, - 0.95, 0.63, 0.13, 0.0045, 1.8, n1q.b ], [ - 0.83, - 0.56, 0.37, 0.11, 0.003, 2.6, n1q.a ] ] ) {

				const k = 2 * Math.PI / lam;
				const ph = c.uvm.x.mul( dx * k ).add( c.uvm.y.mul( dy * k ) ).add( time.mul( sp * k ) ).add( n0.g.mul( 3 ).add( n0.a.mul( dx * 2 ) ) )
					.add( n1.x.mul( wob ).add( n1.y.mul( wob * dy ) ) );
				// (a train finer than two pixels would beat against the pixel grid: it fades into the gloss instead)
				w = w.add( vec2( dx, dy ).mul( cos( ph ).mul( amp ).mul( gust.mul( 1.3 ).add( 0.2 ) ).mul( smoothstep( 0.5, 0.15, fwm.div( lam ) ) ) ) );

			}

			// landings whip up a short chop
			w = w.add( n1.mul( 0.014 ) ).mul( rp * 0.5 ).mul( agit.mul( 1.5 ).add( 1 ) );
			// rings from the landings: phase runs with the distance from the landing, direction is the mask's gradient
			const ux = c.uvm.dFdx(), uy = c.uvm.dFdy();
			const det = ux.x.mul( uy.y ).sub( ux.y.mul( uy.x ) );
			const idet = det.sign().div( max( abs( det ), 1e-9 ) );
			const ax = agit.dFdx(), ay = agit.dFdy();
			const gr = vec2( ax.mul( uy.y ).sub( ay.mul( ux.y ) ), ay.mul( ux.x ).sub( ax.mul( uy.x ) ) ).mul( idet );
			const gdir = gr.div( max( length( gr ), 1e-4 ) ).mul( smoothstep( 0.05, 0.3, length( gr ) ) ); // towards the landing
			const RL = 0.11; // ring spacing (m); they travel at ~0.3 m/s and die away over a metre
			const rph = oneMinus( agit ).mul( 1.2 * 2 * Math.PI / RL ).sub( time.mul( 0.3 * 2 * Math.PI / RL ) ).add( n0.r.mul( 5 ) );
			const ringA = agit.mul( agit ).mul( smoothstep( 0.4, 0.12, fwm.div( RL ) ) );
			const ring = cos( rph ).mul( ringA ).mul( 0.1 );
			w = w.add( vec2( gdir.x, gdir.y.negate() ).mul( ring ) );
			// foam: a boiling lace of white water painted round the landings
			const tf = time.mul( 0.22 );
			const f1 = texture( this.noise, c.uvm.mul( 5.3 ).add( vec2( tf.mul( 0.9 ), tf.mul( 0.4 ) ) ) );
			const f2 = texture( this.noise, c.uvm.mul( 11.7 ).add( vec2( tf.mul( - 0.7 ), tf.mul( 1.1 ) ) ) );
			const fn = f1.r.mul( 0.55 ).add( f2.g.mul( 0.45 ) ).add( cos( rph ).mul( 0.06 ) );
			const foam = smoothstep( float( 0.72 ).sub( foamM.mul( 0.8 ) ), float( 1.02 ).sub( foamM.mul( 0.55 ) ), fn ).mul( smoothstep( 0.03, 0.3, foamM ) );
			w = w.add( vec2( f1.b.sub( 0.5 ), f2.a.sub( 0.5 ) ).mul( foam.mul( 0.2 ) ) );
			m.normalNode = normalMap( vec3( w.x.add( 0.5 ), w.y.add( 0.5 ), 1 ) );
			let body = T( '#0b1512' ).mul( 0.6 );
			if ( params.depth ) {

				// shallow basin: the bed shows where the eye looks down into it - mottled stone, shifted and bent by
				// the ripples, with the net of light the surface throws on it. Lit by the lamps like any surface.
				const Vw = normalize( cameraPosition.sub( positionWorld ) );
				const see = smoothstep( 0.1, 0.75, Vw.y );
				const bp = positionWorld.xz.sub( Vw.xz.div( max( Vw.y, 0.3 ) ).mul( params.depth * 0.75 ) ).add( w.mul( params.depth * 3 ) );
				const bed = texture( this.noise, bp.mul( 1.3 ) );
				let net = float( 0 );
				if ( this.refraction ) {

					// (full tiers) caustics: bright seams where two drifting noise fields cross
					const ca = texture( this.noise, bp.mul( 2.1 ).add( vec2( tf.mul( 0.5 ), tf.mul( 0.3 ) ) ) ).bias( 1 ).r;
					const cb = texture( this.noise, bp.mul( 2.7 ).add( vec2( tf.mul( - 0.4 ), tf.mul( 0.45 ) ) ) ).bias( 1 ).g;
					net = pow( oneMinus( abs( ca.sub( cb ) ).mul( 3.2 ).min( 1 ) ), 5 );

				}

				const bedCol = T( '#0b1512' ).mul( 2.4 ).add( vec3( 0.02, 0.023, 0.018 ) ).mul( bed.r.mul( 0.7 ).add( 0.65 ) ).mul( net.mul( 1.2 ).add( 1 ) );
				// (the rings focus the light on the bed: bright bands travelling outwards with them)
				body = mix( body, bedCol.mul( sin( rph ).mul( ringA ).mul( 0.7 ).add( 1 ) ), see );
				// light of the submerged lamps: a warm pool in the water round each, alive with the same net
				const halo = smoothstep( 0.5, 1.0, c.mask.y );
				m.emissiveNode = vec3( 1.0, 0.66, 0.3 ).mul( T( '#0b1512' ).mul( 16 ).add( 0.45 ) ).mul( halo.mul( halo ) ).mul( net.mul( 1.6 ).add( 0.5 ).mul( bed.g.mul( 0.6 ).add( 0.7 ) ) )
					.mul( see.mul( 0.7 ).add( 0.3 ) ).mul( oneMinus( foam ) ).mul( c.glow ).mul( U.emissive );

			}

			// (churned water is white where the bubbles are thick, grey-green where the dark water shows between them)
			m.colorNode = mix( body, vec3( 0.5, 0.55, 0.56 ).mul( f2.r.mul( 0.5 ).add( 0.45 ) ), foam.mul( foam.mul( 0.35 ).add( 0.6 ) ) );
			// opaque, so it writes its rippled normal and gloss to the surface buffers and takes screen-space reflections.
			// Where the ripples fall below a pixel (far water) their tilt turns into blur instead of vanishing.
			m.roughnessNode = mix( mix( float( 0.05 ), float( 0.15 ), smoothstep( 0.015, 0.25, fwm ) ), float( 0.65 ), foam );
			m.metalnessNode = float( 0 );
			// the mirror image of the town (screen-space hit or reflection probe) is added by the post pipeline from
			// the rippled normal and the gloss written here

		} else if ( name === 'water_fall' ) {

			// Falling water: a glassy fluted sheet at the lip - continuous, faintly silvered, bending what is behind it and
			// carrying the lamps as bright lines down its flutes - that thickens into ropes, tears open between them and
			// ends in white threads and drops. It is white where it is aerated, torn to spray or painted as foam.
			// GEOMETRY CONTRACT - everything this shader reads from the mesh (keep these when remodelling a fall):
			//   uv        U = metres across the fall, V = along the flow, up from the foot. The pattern scrolls down V at
			//             ~2.9 units/s: V in metres = water at 2.9 m/s throughout; better, advance V by
			//             ds * 2.9 / max( speed, 1.2 ) so the pattern keeps pace with the water (slow over the lip,
			//             drawn out as it accelerates). Flutes are ~4 cm wide, so U must be continuous across a sheet.
			//             Spray cards (crowns, splashes): V DEcreasing along the throw, V scaled 2-4x, _mask.z ~0.8-1.
			//   _mask.z   (wear) break-up: 0 = unbroken sheet .. 0.45 first rents .. 0.75 ropes .. 1 = threads and drops.
			//   _mask.w   (special) foam: 1 = opaque white water (aerated foot, froth). Keep it 0 on the sheet.
			//   material  tint = colour of the white water / faint tint of the body, glow = sparkle of the drops (0.1-0.4),
			//             opacity = overall cover (0.7-1), speed = flow rate (default 1.6).
			// Without the masks it is an unbroken silvery sheet.
			const t = time.mul( params.speed ?? 1.6 );
			const u0 = c.uvm.x, v = c.uvm.y;
			const brk = c.mask.z.saturate(), white = c.mask.w.saturate();
			// strands wander sideways as they fall, more once they have torn free
			const u = u0.add( texture( this.noise, vec2( u0.mul( 0.9 ), v.mul( 0.31 ).add( t.mul( 0.21 ) ) ) ).a.sub( 0.5 ).mul( brk.mul( 0.05 ).add( 0.012 ) ) );
			const s1 = texture( this.noise, vec2( u.mul( 3.1 ), v.mul( 0.23 ).add( t.mul( 0.42 ) ) ) ); // flutes of the sheet (4 x 50 cm)
			const s2 = texture( this.noise, vec2( u.mul( 8.3 ).add( 0.37 ), v.mul( 0.5 ).add( t.mul( 0.9 ) ) ) ); // threads
			const s3 = texture( this.noise, vec2( u.mul( 1.3 ).add( 0.71 ), v.mul( 0.62 ).add( t.mul( 1.1 ) ) ) ); // surges travelling down
			// thickness of the water, 0.5 +- 0.1: the long flutes rule the sheet, the surges take over as it breaks up
			// and chop the ropes into dashes and drops
			const body = s1.r.mul( brk.mul( - 0.2 ).add( 0.47 ) ).add( s2.g.mul( 0.25 ) ).add( s3.b.mul( brk.mul( 0.2 ).add( 0.28 ) ) );
			// erosion: nothing is cut out of the sheet at the lip; the threshold climbs through the thickness as it breaks
			// up, so rents open between the flutes, the flutes become ropes and the ropes threads and drops
			const thr = mix( float( 0.16 ), float( 0.61 ), brk.mul( brk.mul( 0.35 ).add( 0.65 ) ) ).sub( white.mul( 0.14 ) );
			const a = smoothstep( thr, thr.add( 0.06 ), body );
			const rim = a.mul( oneMinus( smoothstep( thr.add( 0.05 ), thr.add( 0.16 ), body ) ) ); // the rolled edge of a rope
			// white water: the film is faintly silvered along its flutes from the start (a moving sheet is never optically
			// clean), entrains air as it falls and is white at torn edges, at the foot and where it is painted as foam
			const fib = smoothstep( 0.42, 0.7, s2.g.mul( 0.55 ).add( s1.b.mul( 0.45 ) ) );
			const air = brk.mul( brk.mul( 0.5 ).add( 0.5 ) );
			const foam = white.add( fib.mul( 0.26 ).add( 0.06 ) ).add( air.mul( fib.mul( 0.4 ).add( 0.36 ) ) ).add( rim.mul( brk ).mul( 0.5 ) ).saturate();
			// relief of the flutes and ropes (thickness gradient in UV metres, from screen derivatives): the highlight of a
			// lamp runs as thin lines down the sheet instead of lighting it evenly
			const hgt = body.mul( 1.6 ).add( smoothstep( thr.sub( 0.02 ), thr.add( 0.2 ), body ) );
			const hx = hgt.dFdx(), hy = hgt.dFdy(), ux = c.uvm.dFdx(), uy = c.uvm.dFdy();
			const det = ux.x.mul( uy.y ).sub( ux.y.mul( uy.x ) );
			const idet = det.sign().div( max( abs( det ), 1e-9 ) ).mul( 0.012 );
			const gu = clamp( hx.mul( uy.y ).sub( hy.mul( ux.y ) ).mul( idet ), - 0.6, 0.6 );
			const gv = clamp( hy.mul( ux.x ).sub( hx.mul( uy.x ) ).mul( idet ), - 0.6, 0.6 );
			const slope = vec2( gu.negate().add( s2.r.sub( 0.5 ).mul( 0.3 ) ), gv.add( s1.b.sub( 0.5 ).mul( 0.15 ) ) );
			m.normalNode = normalMap( vec3( slope.mul( 0.5 ).add( 0.5 ), 1 ) );
			// clear water has no diffuse response: only the silvering and the foam are lit like a surface
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
