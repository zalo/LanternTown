// Natural bloom, fog light scattering and lens dirt on one blur pyramid (three.js TSL, WebGPU and WebGL 2).
//
// Adapted from Sonic Ether's "SE Natural Bloom & Dirty Lens" for Unity (Cody Darr, MIT licence,
// github.com/sonicether/SE-Natural-Bloom-Dirty-Lens): the picture is blurred into a row of ever smaller targets with
// no brightness threshold, and the bloom is a blend TOWARDS a weighted sum of those levels instead of light added on
// top - what a pixel gives to its halo it loses itself, so the picture keeps its energy and nothing has to be
// declared "bright" first. The same levels are then read a second time for the fog: a pixel whose line of sight
// passes through a share `a` of fog is blended towards the blurred picture by that share - fog that smears light
// instead of only hiding it - and a third time, where a dirt texture is given, for the smudges on the lens. One
// pyramid, three readers: the scattering costs what the bloom already cost. (His Minecraft pack does the same in its
// last pass with one line: the mix by 1 - exp( -depth * k ) after the bloom's own mix. Nothing here is taken from
// that pack - its licence forbids it - only the idea, which the MIT Unity effect has in full.)
//
//   const nb = naturalBloom( vec4( rgb, fogShare ), { levels: 6, divisor: 2 } );   // rgb linear, before the tone map
//   pipeline.outputNode = renderOutput( vec4( nb.rgb, 1 ) );
//
// Differences from the Unity effect: the first level is taken while downsampling (no full-size copy of the frame);
// one two-pass Gaussian per level, the next level reading the one before it at half its size; six levels from half
// size (or `divisor: 4, levels: 4` on a phone) instead of eight; the fog share comes in with the picture's alpha, so
// the caller decides what fog is (and can have it antialiased by its TRAA with the colour); and the guard against a
// bright thing nearby glowing into the far fog also takes back the blend where every level is spoilt by it (`guard`),
// where the original only moved weight between levels.
//
// `flare: true` adds a fourth reader, after Lena Piquet's (Froyok) write-up of a lens flare for Unreal and John
// Chapman's pseudo lens flare before it: ghosts - copies of what is bright, mirrored and scaled about the middle of
// the picture - and a halo, each with its colours pulled apart a little. What is bright is taken from the picture
// itself, at full size, before anything has thinned it (a lantern across a square is three pixels far over white; in
// a blurred level it is under the lit wall next to it, and a threshold there finds the wall), then kept at a quarter
// of the size and blurred a little: three small passes and the flare's own.

import { HalfFloatType, RenderTarget, Vector2, TempNode, QuadMesh, NodeMaterial, RendererUtils, NodeUpdateType, CanvasTexture, RepeatWrapping } from 'three/webgpu';
import { Fn, float, uv, uniform, texture, mix, vec2, vec3, vec4, max, saturate, dot, length, pow, smoothstep, context, convertToTexture, nodeObject } from 'three/tsl';

const _quad = /*@__PURE__*/ new QuadMesh();
const _size = /*@__PURE__*/ new Vector2();
let _state;

// share of each octave in the fog's blur (the Unity effect's table: octave 0 is half size, 1 a quarter, ...)
const FOG_OCTAVES = [ 0.1, 0.27, 0.45, 0.55, 0.45, 0.38, 0.23, 0.0 ];
const lum = ( c ) => dot( c, vec3( 1 / 3 ) );
// the ghosts: scale about the middle (negative: mirrored; under 1: larger than what it copies), tint, strength
const GHOSTS = [ [ - 1.5, [ 1.0, 0.8, 0.45 ], 0.5 ], [ 2.5, [ 1.0, 0.95, 0.6 ], 0.3 ], [ 0.7, [ 0.45, 0.8, 1.0 ], 0.22 ], [ - 0.4, [ 0.5, 1.0, 0.45 ], 0.12 ], [ - 0.2, [ 0.9, 0.5, 1.0 ], 0.06 ], [ - 0.1, [ 0.6, 0.7, 1.0 ], 0.03 ] ];

class NaturalBloomNode extends TempNode {

	static get type() {

		return 'NaturalBloomNode';

	}

	/**
	 * @param {TextureNode} inputNode - rgb: the picture, linear and before the tone map. a: the share of fog on the pixel's line of sight, 0..1.
	 * @param {Object} [o]
	 * @param {number} [o.levels=6] - blur levels, each half the size of the one before
	 * @param {number} [o.divisor=2] - size of the first level: 2 (half) or 4 (a quarter - no tight halo, a third of the work)
	 * @param {Texture} [o.dirt=null] - lens dirt (rgb: how much of the smeared picture each part of the lens lets in)
	 * @param {boolean} [o.flare=false] - ghosts and a halo from what is bright (`flareIntensity`)
	 */
	constructor( inputNode, { levels = 6, divisor = 2, dirt = null, flare = false } = {} ) {

		super( 'vec4' );
		this.inputNode = inputNode;
		this.levels = Math.max( 1, Math.min( 8, Math.round( levels ) ) );
		this.divisor = divisor >= 4 ? 4 : 2;
		this.dirtTexture = dirt;

		// bloom: the share of every pixel that goes to its halo (the Unity slider 0..0.4, default 0.05). spread 0..1:
		// which levels carry it - 0 the tight ones, 0.5 all alike, 1 the wide ones
		this.intensity = uniform( 0.05 );
		this.spread = uniform( 0.5 );
		// fog: how much of the input's fog share is scattered (1: all of it, 0: none - the fog only hides). fogSpread as spread.
		// guard 0..1: how much a blur that is lit from nearer than the pixel is kept out of it
		this.fog = uniform( 1 );
		this.fogSpread = uniform( 0.5 );
		this.guard = uniform( 1 );
		// lens dirt: only with a dirt texture
		this.dirtIntensity = uniform( 0.05 );
		this.dirtSpread = uniform( 0.5 );
		// lens flare: only with `flare`. What it is made of: the part of the picture over `flareThreshold` (scene units:
		// over what any lit surface reaches, so that only what gives light passes), rolled off so that no pixel gives
		// more than `flareRange`, blurred by `flareBlur` (1: the nine-tap Gaussian at a quarter of the picture's size).
		// flareCurve: the power that is taken to on the way out (1: as it is; 0.5: a lamp of a tenth of the light still
		// gives a third of the ghost - the far ones are seen and the near one does not glare).
		// chroma: how far red and blue part, as a share of the distance from the middle. haloWidth: the ring's radius
		// in screen heights
		this.flareIntensity = uniform( 0.5 );
		this.flareThreshold = uniform( 4 );
		this.flareRange = uniform( 2 );
		this.flareBlur = uniform( 1 );
		this.flareCurve = uniform( 0.5 );
		this.flareChroma = uniform( 0.012 );
		this.haloWidth = uniform( 0.42 );
		this.haloIntensity = uniform( 0.35 );
		this._fl = Math.min( this.divisor >= 4 ? 0 : 1, this.levels - 1 ); // the level it has the size of: a quarter of the picture
		this._wantFlare = !! flare;
		this._flareMaterial = null;
		this._brightMaterial = null;
		this._aspect = uniform( 1 );

		const target = ( name ) => {

			const rt = new RenderTarget( 1, 1, { depthBuffer: false, type: HalfFloatType } );
			rt.texture.name = 'NaturalBloom.' + name;
			rt.texture.generateMipmaps = false;
			return rt;

		};

		this._down = target( 'down' );
		this._flare = this._wantFlare ? target( 'flare' ) : null;
		this._bright = this._wantFlare ? target( 'bright' ) : null;
		this._brightH = this._wantFlare ? target( 'brightH' ) : null;
		this._h = []; this._v = []; this._wb = []; this._wf = []; this._wd = [];
		for ( let i = 0; i < this.levels; i ++ ) {

			this._h.push( target( 'h' + i ) );
			this._v.push( target( 'v' + i ) );
			this._wb.push( uniform( 0 ) ); this._wf.push( uniform( 0 ) ); this._wd.push( uniform( 0 ) );

		}

		this._texel = uniform( new Vector2() ); // one texel of the input
		this._downMaterial = null;
		this._blurMaterial = null;
		this.updateBeforeType = NodeUpdateType.FRAME;

	}

	/** The blurred picture of one level as a texture node (checking, or for another reader of the pyramid). */
	getLevel( i ) {

		return texture( this._v[ Math.max( 0, Math.min( this.levels - 1, i ) ) ].texture );

	}

	setSize( width, height ) {

		this._texel.value.set( 1 / width, 1 / height );
		this._aspect.value = width / height;
		let w = Math.max( 1, Math.floor( width / this.divisor ) ), h = Math.max( 1, Math.floor( height / this.divisor ) );
		this._down.setSize( w, h );
		for ( let i = 0; i < this.levels; i ++ ) {

			this._h[ i ].setSize( w, h );
			this._v[ i ].setSize( w, h );
			if ( this._flare && i === this._fl ) { this._flare.setSize( w, h ); this._bright.setSize( w, h ); this._brightH.setSize( w, h ); }
			w = Math.max( 1, Math.floor( w / 2 ) ); h = Math.max( 1, Math.floor( h / 2 ) );

		}

	}

	_weights() {

		// as the Unity script: pow( octave + 1, slider * 5 - 2.5 ) for the bloom and the dirt, the table times
		// pow( octave + 1, slider * 4 - 1 ) for the fog; the bloom's and the dirt's sum to one here, the fog's in the shader
		const o0 = this.divisor >= 4 ? 1 : 0, n = this.levels;
		const eb = this.spread.value * 5 - 2.5, ed = this.dirtSpread.value * 5 - 2.5, ef = this.fogSpread.value * 4 - 1;
		let sb = 0, sd = 0;
		const b = [], d = [];
		// (by octave, not by level: a pyramid that starts at a quarter has the same widths the same weights as one that starts at half)
		for ( let i = 0; i < n; i ++ ) { b.push( Math.pow( i + o0 + 1, eb ) ); d.push( Math.pow( i + o0 + 1, ed ) ); sb += b[ i ]; sd += d[ i ]; }
		for ( let i = 0; i < n; i ++ ) {

			this._wb[ i ].value = b[ i ] / sb;
			this._wd[ i ].value = d[ i ] / sd;
			// (the last octave of the table is zero: the widest level in use here keeps the share of the one before it)
			const t = FOG_OCTAVES[ Math.min( i + o0, FOG_OCTAVES.length - 2 ) ];
			this._wf[ i ].value = t * Math.pow( i + o0 + 1, ef );

		}

	}

	updateBefore( frame ) {

		const { renderer } = frame;
		_state = RendererUtils.resetRendererState( renderer, _state );

		const size = renderer.getDrawingBufferSize( _size );
		this.setSize( size.width, size.height );
		this._weights();

		// 1. the picture at the first level's size, its alpha the fog share weighted by the brightness that carries it
		renderer.setRenderTarget( this._down );
		_quad.material = this._downMaterial;
		_quad.name = 'NaturalBloom [ Down ]';
		_quad.render( renderer );

		// 2. blur each level, the next one reading it at half the size
		const m = this._blurMaterial;
		_quad.material = m;
		let src = this._down;
		for ( let i = 0; i < this.levels; i ++ ) {

			const h = this._h[ i ], v = this._v[ i ];
			m.colorTexture.value = src.texture;
			m.step.value.set( 1 / h.width, 0 );
			renderer.setRenderTarget( h );
			_quad.name = `NaturalBloom [ Blur H ${ i } ]`;
			_quad.render( renderer );

			m.colorTexture.value = h.texture;
			m.step.value.set( 0, 1 / v.height );
			renderer.setRenderTarget( v );
			_quad.name = `NaturalBloom [ Blur V ${ i } ]`;
			_quad.render( renderer );
			src = v;

		}

		// 3. the lens flare: what is bright, blurred, then its ghosts and halo (none of it while its strength is 0:
		// a switch on the page turns it on and off without anything being built again)
		if ( this._flare && this.flareIntensity.value > 0 ) {

			const b = this._bright, bh = this._brightH, k = this.flareBlur.value;
			renderer.setRenderTarget( b );
			_quad.material = this._brightMaterial;
			_quad.name = 'NaturalBloom [ Bright ]';
			_quad.render( renderer );
			_quad.material = m;
			m.colorTexture.value = b.texture;
			m.step.value.set( k / b.width, 0 );
			renderer.setRenderTarget( bh );
			_quad.name = 'NaturalBloom [ Bright H ]';
			_quad.render( renderer );
			m.colorTexture.value = bh.texture;
			m.step.value.set( 0, k / b.height );
			renderer.setRenderTarget( b );
			_quad.name = 'NaturalBloom [ Bright V ]';
			_quad.render( renderer );

			renderer.setRenderTarget( this._flare );
			_quad.material = this._flareMaterial;
			_quad.name = 'NaturalBloom [ Flare ]';
			_quad.render( renderer );

		}

		RendererUtils.restoreRendererState( renderer, _state );

	}

	setup( builder ) {

		const shared = context( builder.getSharedContext() );
		const inp = this.inputNode;
		const uvN = uv();

		// first level
		const down = Fn( () => {

			let c;
			if ( this.divisor >= 4 ) {

				// four taps between the texels: a 4 x 4 box, so that a lamp of one pixel is in every frame's sum
				const t = this._texel;
				c = inp.sample( uvN.add( t ) ).add( inp.sample( uvN.sub( t ) ) ).add( inp.sample( uvN.add( vec2( t.x, t.y.negate() ) ) ) ).add( inp.sample( uvN.add( vec2( t.x.negate(), t.y ) ) ) ).mul( 0.25 );

			} else c = inp.sample( uvN );
			const rgb = max( c.rgb, 0 ).min( 256 );
			return vec4( rgb, saturate( c.a ).mul( lum( rgb ) ) );

		} );
		this._downMaterial = this._downMaterial || new NodeMaterial();
		this._downMaterial.contextNode = shared;
		this._downMaterial.fragmentNode = down();
		this._downMaterial.name = 'NaturalBloom_down';
		this._downMaterial.needsUpdate = true;

		// nine-tap Gaussian in five fetches (pairs of taps merged into one bilinear fetch)
		const colorTexture = texture( null ), step = uniform( new Vector2() );
		const blur = Fn( () => {

			const sum = colorTexture.sample( uvN ).mul( 0.22702 ).toVar();
			for ( const [ o, w ] of [ [ 1.38461, 0.31624 ], [ 3.23076, 0.07087 ] ] ) {

				sum.addAssign( colorTexture.sample( uvN.add( step.mul( o ) ) ).add( colorTexture.sample( uvN.sub( step.mul( o ) ) ) ).mul( w ) );

			}

			return sum;

		} );
		this._blurMaterial = this._blurMaterial || new NodeMaterial();
		this._blurMaterial.contextNode = shared;
		this._blurMaterial.fragmentNode = blur();
		this._blurMaterial.name = 'NaturalBloom_blur';
		this._blurMaterial.needsUpdate = true;
		this._blurMaterial.colorTexture = colorTexture;
		this._blurMaterial.step = step;

		if ( this._flare ) {

			// what is bright: every tap over the threshold on its own (four taps between the input's texels - at a
			// quarter of the size a 4 x 4 box, so a lamp of one pixel is in every frame's sum and is not thinned
			// under the threshold first), rolled off towards `flareRange`
			const over = Fn( () => {

				const t = this._texel;
				const tap = ( o ) => {

					const c = max( inp.sample( uvN.add( o ) ).rgb, 0 ).min( 256 ), m = max( max( c.r, c.g ), max( c.b, 1e-4 ) ), x = max( m.sub( this.flareThreshold ), 0 );
					return c.div( m ).mul( x.mul( this.flareRange ).div( x.add( this.flareRange ) ) );

				};
				return vec4( tap( t ).add( tap( t.negate() ) ).add( tap( vec2( t.x, t.y.negate() ) ) ).add( tap( vec2( t.x.negate(), t.y ) ) ).mul( 0.25 ), 1 );

			} );
			this._brightMaterial = this._brightMaterial || new NodeMaterial();
			this._brightMaterial.contextNode = shared;
			this._brightMaterial.fragmentNode = over();
			this._brightMaterial.name = 'NaturalBloom_bright';
			this._brightMaterial.needsUpdate = true;

			const lvl = texture( this._bright.texture );
			const flare = Fn( () => {

				const cen = uvN.sub( 0.5 ), asp = vec2( this._aspect, 1 );
				// what is bright at p, fading out where p nears the picture's edge (a copy must not end at a straight line)
				const bright = ( p ) => {

					const c = lvl.sample( p ).rgb, m = max( max( c.r, c.g ), c.b ).div( this.flareRange );
					const e = smoothstep( 0, 0.12, p.x ).mul( smoothstep( 0, 0.12, p.x.oneMinus() ) ).mul( smoothstep( 0, 0.12, p.y ) ).mul( smoothstep( 0, 0.12, p.y.oneMinus() ) );
					// (the curve: a lantern a street away is a few pixels that the blur has spread to a tenth of the roll-off's
					// ceiling, the one overhead a disc at that ceiling - taken in proportion the first gave nothing and
					// the second a glare. This way they differ by the size of their ghosts more than by the brightness.)
					return c.mul( pow( max( m, 1e-3 ), this.flareCurve.sub( 1 ) ) ).mul( e );

				};
				// (red from a little further out, blue from a little further in: three fetches a ghost)
				const split = ( at ) => vec3( at( this.flareChroma.add( 1 ) ).r, at( float( 1 ) ).g, at( this.flareChroma.oneMinus() ).b );
				const sum = vec3( 0 ).toVar();
				for ( const [ s, tint, gain ] of GHOSTS ) sum.addAssign( split( ( k ) => bright( cen.mul( k.mul( s ) ).add( 0.5 ) ) ).mul( vec3( ...tint ) ).mul( gain ) );
				// the halo: every pixel looks `haloWidth` towards the middle, so a lamp near the middle becomes a ring around it
				const da = cen.negate().mul( asp ), r = length( da ), n = da.div( max( r, 1e-4 ) );
				const ring = ( k ) => {

					const p = uvN.add( n.mul( this.haloWidth.mul( k ) ).div( asp ) );
					return bright( p ).mul( pow( saturate( length( p.sub( 0.5 ).mul( asp ) ).div( 0.9 ).oneMinus() ), 3 ) );

				};
				sum.addAssign( split( ring ).mul( this.haloIntensity ).mul( smoothstep( 0, 0.15, r ) ) );
				return vec4( sum, 1 );

			} );
			this._flareMaterial = this._flareMaterial || new NodeMaterial();
			this._flareMaterial.contextNode = shared;
			this._flareMaterial.fragmentNode = flare();
			this._flareMaterial.name = 'NaturalBloom_flare';
			this._flareMaterial.needsUpdate = true;

		}

		// the readers
		const dirtTex = this.dirtTexture ? texture( this.dirtTexture, uvN ) : null;
		const out = Fn( () => {

			const src = inp.sample( uvN ).toVar();
			const f = saturate( src.a.mul( this.fog ) ).toVar();
			const bloom = vec3( 0 ).toVar(), haze = vec3( 0 ).toVar(), smudge = vec3( 0 ).toVar();
			const wsum = float( 0 ).toVar(), wall = float( 0 ).toVar();
			for ( let i = 0; i < this.levels; i ++ ) {

				const b = texture( this._v[ i ].texture, uvN ).toVar();
				bloom.addAssign( b.rgb.mul( this._wb[ i ] ) );
				if ( dirtTex ) smudge.addAssign( b.rgb.mul( this._wd[ i ] ) );
				// the fog share of what lit this level here. Less than the pixel's own: the light is from in front of
				// the fog this pixel is seen through, and has no business in its haze
				const lit = b.a.div( max( lum( b.rgb ), 1e-7 ) );
				const w = this._wf[ i ].mul( saturate( float( 1 ).sub( max( f.sub( lit ), 0 ).mul( 0.99 ) ) ) );
				haze.addAssign( b.rgb.mul( w ) );
				wsum.addAssign( w ); wall.addAssign( this._wf[ i ] );

			}

			const col = mix( src.rgb, bloom, this.intensity ).toVar();
			if ( dirtTex ) col.assign( mix( col, smudge, saturate( dirtTex.rgb.mul( this.dirtIntensity ) ) ) );
			const kept = mix( float( 1 ), wsum.div( max( wall, 1e-6 ) ), this.guard );
			col.assign( mix( col, haze.div( max( wsum, 1e-6 ) ), f.mul( kept ) ) );
			// (the flare is of the lens, not of the air: on top of everything, and brighter where the lens is dirty)
			if ( this._flare ) col.addAssign( texture( this._flare.texture, uvN ).rgb.mul( this.flareIntensity ).mul( dirtTex ? dirtTex.rgb.mul( 4 ).add( 1 ) : 1 ) );
			return vec4( max( col, 0 ), src.a );

		} );
		return out();

	}

	dispose() {

		super.dispose();
		this._down.dispose();
		for ( const rt of this._h ) rt.dispose();
		for ( const rt of this._v ) rt.dispose();
		if ( this._downMaterial ) this._downMaterial.dispose();
		if ( this._blurMaterial ) this._blurMaterial.dispose();
		if ( this._flare ) { this._flare.dispose(); this._bright.dispose(); this._brightH.dispose(); }
		if ( this._flareMaterial ) this._flareMaterial.dispose();
		if ( this._brightMaterial ) this._brightMaterial.dispose();

	}

}

export default NaturalBloomNode;

/**
 * @param {Node<vec4>} node - rgb the picture (linear), a the share of fog on the pixel's line of sight
 * @param {Object} [options] - levels, divisor, dirt, flare
 * @returns {NaturalBloomNode}
 */
export const naturalBloom = ( node, options ) => new NaturalBloomNode( convertToTexture( nodeObject( node ), null, null, { type: HalfFloatType } ), options );

/**
 * A lens-dirt texture drawn from a seed (no picture file): out-of-focus specks and dust rings, denser towards the
 * rim, and a few wiped smears. rgb: how much of the smeared picture that part of the lens lets in.
 */
export function lensDirt( seed = 7, width = 1024, height = 576 ) {

	let s = seed >>> 0;
	const rnd = () => { s = s + 0x6D2B79F5 | 0; let t = Math.imul( s ^ s >>> 15, 1 | s ); t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t; return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296; };
	const cv = document.createElement( 'canvas' );
	cv.width = width; cv.height = height;
	const g = cv.getContext( '2d' );
	g.fillStyle = '#000'; g.fillRect( 0, 0, width, height );
	g.globalCompositeOperation = 'lighter';
	const tint = ( a ) => `rgba(${ 200 + rnd() * 55 | 0 },${ 200 + rnd() * 55 | 0 },${ 200 + rnd() * 55 | 0 },${ a })`;
	// towards the rim: where a lens is wiped least
	const place = () => { for ( let k = 0; k < 8; k ++ ) { const x = rnd(), y = rnd(); if ( rnd() < 0.25 + Math.hypot( x - 0.5, y - 0.5 ) * 1.5 ) return [ x * width, y * height ]; } return [ rnd() * width, rnd() * height ]; };
	for ( let i = 0; i < 26; i ++ ) { // smears
		const [ x, y ] = place(), r = ( 0.08 + rnd() * 0.2 ) * height, a = rnd() * Math.PI;
		g.save(); g.translate( x, y ); g.rotate( a ); g.scale( 1 + rnd() * 2.5, 1 );
		const gr = g.createRadialGradient( 0, 0, 0, 0, 0, r );
		gr.addColorStop( 0, tint( 0.05 + rnd() * 0.07 ) ); gr.addColorStop( 1, 'rgba(0,0,0,0)' );
		g.fillStyle = gr; g.beginPath(); g.arc( 0, 0, r, 0, 6.2832 ); g.fill(); g.restore();
	}
	for ( let i = 0; i < 150; i ++ ) { // dust: discs with a brighter rim, as specks out of focus look
		const [ x, y ] = place(), r = ( 0.008 + rnd() * rnd() * 0.06 ) * height, a = 0.04 + rnd() * 0.16;
		const gr = g.createRadialGradient( x, y, 0, x, y, r );
		gr.addColorStop( 0, tint( a * 0.55 ) ); gr.addColorStop( 0.78, tint( a * 0.7 ) ); gr.addColorStop( 0.92, tint( a ) ); gr.addColorStop( 1, 'rgba(0,0,0,0)' );
		g.fillStyle = gr; g.beginPath(); g.arc( x, y, r, 0, 6.2832 ); g.fill();
	}
	for ( let i = 0; i < 500; i ++ ) { // grit
		const [ x, y ] = place(), r = ( 0.001 + rnd() * 0.004 ) * height;
		g.fillStyle = tint( 0.1 + rnd() * 0.3 ); g.beginPath(); g.arc( x, y, r, 0, 6.2832 ); g.fill();
	}
	const t = new CanvasTexture( cv );
	t.wrapS = t.wrapT = RepeatWrapping;
	t.name = 'lensDirt.' + seed;
	return t;

}
