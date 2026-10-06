// The wet paving: what standing water mirrors, in two parts that one full-screen pass puts together.
//
// A surface that mirrors says so in a second colour target of the scene pass (`wet`: strength, roughness, tilt of
// its mean normal - see out()). For those pixels the pass
//  1. follows the reflected ray through the frame's own depth buffer - short steps at the foot of the ray, longer
//     ones further on, then a bisection - and takes what it finds from the frame itself, drawn out along the ray
//     by the roughness: a bollard's foot, a bench, a kerb, mirrored where they touch the paving;
//  2. and where the ray finds nothing reads a mirror image of the town: the scene drawn from the camera's
//     reflection in the paving it stands on, at the lowest level of detail, into a small target whose mip levels
//     are the blur. It is exact for the paving at the camera's level whatever stands where, moves with every
//     step, and does not depend on the level any cell happens to show - so a lantern's streak neither jumps nor
//     switches as one walks, which a cube map taken every few steps did.
//
// No pass is added for this beyond the image itself: the tiers without `mirror` simply do not create it.

import * as THREE from 'three';
import {
	texture, uniform, uv, vec2, vec3, vec4, float, int, bool, Fn, If, Loop, Break, mrt, output, velocity, rtt,
	saturate, normalize, reflect, dot, sqrt, max, min, pow, step, length, clamp, log2, smoothstep, oneMinus,
	getViewPosition, getScreenPosition, interleavedGradientNoise, screenCoordinate,
} from 'three/tsl';

export function createMirror( stage, tier ) {

	const { renderer, scene, camera } = stage;
	const image = createImage( stage, tier );

	// reach / thick: the ray's length and how thick a surface counts (m); smear: how far roughness draws a find out;
	// max: the brightest thing a mirror shows. For the image - lobe: length of a streak over the square of the
	// roughness; gap: blur of a tap over the distance to the next; tg / tk: how far away what is mirrored is taken
	// to be (m); hit: a ray's own find only counts this close to the pixel (m) - further out the frame and the image
	// differ in the level they show a house at, and a streak must not depend on that.
	const MU = { reach: uniform( 10 ), thick: uniform( 0.2 ), smear: uniform( 1.5 ), max: uniform( 6 ), lobe: uniform( 2.2 ), gap: uniform( 0.6 ),
		tg: uniform( 8 ), tk: uniform( 30 ), penv: uniform( 1 ), lmin: uniform( 1 ), ssr: uniform( 1 ), hit: uniform( 8 ) };
	const STEPS = Math.max( 8, Math.round( tier.mirror ) ), TAPS = 5;

	// the reflected light of the frame `scenePass` has drawn: rgb to add to it
	const reflection = ( scenePass ) => {

		const colorTex = scenePass.getTextureNode( 'output' ), wetTex = scenePass.getTextureNode( 'wet' ), depthTex = scenePass.getTextureNode( 'depth' );
		const proj = uniform( camera.projectionMatrix ), projInv = uniform( camera.projectionMatrixInverse ), view = uniform( camera.matrixWorldInverse ), world = uniform( camera.matrixWorld );
		// (getViewPosition() takes the depth of a WebGL frame for -1..1; with reversed depth it is 0..1 there too)
		const viewAt = stage.isWebGPU || stage.reversed ? ( u ) => {

			const p = projInv.mul( vec4( u.x.mul( 2 ).sub( 1 ), u.y.oneMinus().mul( 2 ).sub( 1 ), depthTex.sample( u ).r, 1 ) );
			return p.xyz.div( p.w );

		} : ( u ) => getViewPosition( u, depthTex.sample( u ).r, projInv );

		const pass = Fn( () => {

			const uvN = uv();
			const wv = wetTex.sample( uvN ).toVar();
			const out = vec4( 0 ).toVar();
			If( wv.r.greaterThan( 0.006 ), () => {

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
				const jitter = interleavedGradientNoise( screenCoordinate.xy );
				const found = bool( false ).toVar();
				const hit = vec4( 0 ).toVar(); // what the ray found, and how surely
				const sLo = float( 0 ).toVar(), sHi = float( 0 ).toVar(), zLo = P0.z.toVar();
				Loop( { start: int( 0 ), end: int( STEPS ) }, ( { i } ) => {

					const s = sMax.mul( pow( float( i ).add( jitter ).add( 0.25 ).div( STEPS + 0.25 ), 1.7 ) ).toVar();
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
					const sure = smoothstep( tk, tk.mul( 0.35 ), vP.z.sub( rayZ( sHi ) ).abs() ) // passed behind something
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
					hit.assign( vec4( c, sure ) );

				} );

				// The mirror image: the point `tg` metres along the reflected ray as the mirrored camera sees it (on the
				// paving the camera stands on that is the pixel's own place in the image, wherever along the ray).
				// Roughness draws it out towards the viewer far more than sideways - seen at a glancing angle a tilt of
				// the surface along the line of sight moves the mirrored point up and down, a tilt across it hardly at
				// all: TAPS samples above and below, each blurred by the distance to the next.
				const Pw = world.mul( vec4( P0, 1 ) ).xyz, Rw = world.mul( vec4( R, 0 ) ).xyz;
				const tg = MU.tg.add( Pw.y.sub( image.g ).abs().mul( MU.tk ) );
				const C0 = image.vp.mul( vec4( Pw.add( vec3( Rw.x, 0, Rw.z ).mul( tg ) ), 1 ) ).toVar();
				const D = image.vp.mul( vec4( 0, 1, 0, 0 ) ).mul( tg ).toVar();
				const dl = clamp( wv.g.mul( wv.g ).mul( MU.lobe ), 0.004, 0.7 ).toVar();
				const lv = max( log2( max( dl.mul( image.focal ).mul( MU.gap ).mul( 2 / TAPS ), 1 ) ), MU.lmin ).toVar();
				let sum = vec3( 0 ), weights = 0;
				for ( let k = 0; k < TAPS; k ++ ) {

					const ek = ( k + 0.5 ) / TAPS * 2 - 1, wk = Math.exp( - 2.5 * ek * ek );
					const c = C0.add( D.mul( max( Rw.y.add( dl.mul( ek ) ), 0.01 ) ) );
					const u = c.xy.div( max( c.w, 1e-3 ) );
					sum = sum.add( image.node.sample( vec2( u.x.mul( 0.5 ).add( 0.5 ), u.y.mul( - 0.5 ).add( 0.5 ) ) ).level( lv ).rgb.min( MU.max ).mul( wk ) );
					weights += wk;

				}

				const far = sum.mul( MU.penv.div( weights ) ).mul( image.on );
				out.assign( vec4( mix( far, hit.rgb, hit.a.mul( MU.ssr ) ).mul( wv.r ), wv.r ) );

			} );
			return out.max( 0 );

		} );
		return rtt( pass(), null, null, { type: THREE.HalfFloatType } ).rgb;

	};

	const mirror = {
		MU, image,
		// what a material that mirrors writes beside its colour: k strength 0..1, the roughness, N its mean normal in world space
		out: ( k, rough, N ) => mrt( { output, wet: vec4( saturate( k ), rough, N.x.add( 0.5 ), N.z.add( 0.5 ) ) } ),
		// the scene pass's targets: colour, `wet`, and the velocity TRAA follows. Blended materials (glass, glows)
		// leave `wet` as it is.
		targets() {

			const targets = mrt( { output, wet: vec4( 0 ), velocity } );
			targets.setBlendMode( 'wet', new THREE.BlendMode( THREE.MaterialBlending ) );
			return targets;

		},
		reflection,
		update() { image.step(); },
	};
	// (the materials take these from the stage, as they take the fog and the knee)
	stage.mirror = MU; stage.probe = image; stage.mirrorOut = mirror.out;
	return mirror;

	function mix( a, b, t ) { return a.add( b.sub( a ).mul( t ) ); }

}

// The mirror image: the scene from the camera's reflection in the paving under it, every second frame while the
// camera moves and now and then while it stands (the flames in it). `ground( position )` says where the paving is
// and `swap( on )` puts the town into the state the image wants (lowest level, cut at the mirror's plane): both
// are the caller's, set once the town is there.
function createImage( stage, tier ) {

	const { renderer, scene, camera } = stage;
	const width = Math.max( 64, Math.round( tier.planar ) ), height = Math.max( 36, Math.round( width * 9 / 16 ) );
	// (multisampled: a pixel of this image is 8 cm of a facade 30 m off, and a bright thing narrower than that was
	// in a pixel or not from one step to the next - its streak on the paving flashed)
	const rt = new THREE.RenderTarget( width, height, { type: THREE.HalfFloatType, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, samples: 4 } );
	rt.texture.name = 'output'; // (a material with outputs of its own finds its colour target by this name)
	const cam = stage.ownCamera( new THREE.PerspectiveCamera() );
	const eye = new THREE.Vector3(), forward = new THREE.Vector3(), up = new THREE.Vector3(), target = new THREE.Vector3(), rotation = new THREE.Matrix4();
	const last = new THREE.Matrix4().makeScale( 0, 0, 0 );
	const EVERY = 6, RATE = 2; // frames between images: standing, moving
	let idle = 0, phase = 0;

	const image = {
		planar: true, width, height, rt, cam, node: texture( rt.texture ),
		vp: uniform( new THREE.Matrix4() ), g: uniform( 0 ), focal: uniform( height ), on: uniform( 0 ), count: 0,
		// wide: a wider view than the camera's - a streak is drawn from what stands above and beside its place;
		// level: the height of the mirror (the view budget counts what this image draws by these two, see budget.js)
		wide: 1.3, level: 0,
		// Nothing below the paving is in a mirror: `clip` is the mirror's plane, for a ClippingGroup that is on only
		// while the image is drawn. (Turning the near plane into the mirror's instead costs nothing, but then depth
		// falls off with the height above the paving: two surfaces a millimetre apart came out in a different order
		// from one step to the next, and a lit window's streak flickered.)
		clip: new THREE.Plane( new THREE.Vector3( 0, 1, 0 ), 0 ),
		ground: null, swap: null,

		step( now = false ) {

			camera.updateMatrixWorld();
			const still = camera.matrixWorld.equals( last );
			if ( ! now && image.on.value ) {

				if ( still && ( idle = ( idle + 1 ) % EVERY ) !== 0 ) return;
				// (the lookup goes through the view-projection the image was drawn with, so an image a frame old is still
				// placed right; `last` stays behind on a skipped frame, so the place the camera comes to rest at is drawn)
				if ( ! still && ( phase = ( phase + 1 ) % RATE ) !== 0 ) return;

			}

			last.copy( camera.matrixWorld );
			const E = eye.setFromMatrixPosition( camera.matrixWorld );
			// the mirror: the paving under the camera, eased so a kerb does not make the image jump
			let g = image.ground ? image.ground( E ) : null;
			if ( g === null || g === undefined ) g = image.level;
			g = Math.min( g, E.y - 0.1 );
			image.level = now || ! image.on.value ? g : image.level + ( g - image.level ) * 0.2;
			g = Math.min( image.level, E.y - 0.1 );
			image.g.value = g;
			rotation.extractRotation( camera.matrixWorld );
			forward.set( 0, 0, - 1 ).applyMatrix4( rotation );
			up.set( 0, 1, 0 ).applyMatrix4( rotation );
			cam.position.set( E.x, 2 * g - E.y, E.z );
			cam.up.set( up.x, - up.y, up.z );
			cam.updateMatrixWorld( true ); // (lookAt reads the place from the world matrix)
			cam.lookAt( target.set( E.x + forward.x, 2 * g - ( E.y + forward.y ), E.z + forward.z ) );
			cam.near = camera.near; cam.far = camera.far;
			cam.updateMatrix();
			cam.updateMatrixWorld( true );
			const pm = cam.projectionMatrix.copy( camera.projectionMatrix ), e = pm.elements;
			e[ 0 ] /= image.wide; e[ 5 ] /= image.wide;
			image.focal.value = e[ 5 ] * height / 2;
			image.clip.constant = - g;
			cam.projectionMatrixInverse.copy( pm ).invert();
			image.vp.value.multiplyMatrices( pm, cam.matrixWorldInverse );
			const was = [ renderer.getRenderTarget(), renderer.getActiveCubeFace(), renderer.getActiveMipmapLevel() ];
			if ( image.swap ) image.swap( true );
			renderer.setRenderTarget( rt );
			renderer.render( scene, cam );
			if ( image.swap ) image.swap( false );
			renderer.setRenderTarget( ...was );
			image.count ++;
			image.on.value = 1;

		},
	};
	return image;

}

// Where the paving is that a camera at `p` (three's frame) stands over: the highest walkable surface of the
// bake's navigation data below it, or null.
export function pavingUnder( nav ) {

	return ( p ) => {

		let best = null;
		for ( const region of nav.regions ) {

			const r = region.rect;
			if ( p.x < r[ 0 ] || p.x > r[ 2 ] || - p.z < r[ 1 ] || - p.z > r[ 3 ] ) continue;
			const h = nav.heightOf( region, p.x, - p.z );
			if ( h <= p.y - 0.3 && ( best === null || h > best ) ) best = h;

		}

		return best;

	};

}
