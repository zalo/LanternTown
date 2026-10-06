// Glass: window panes, and the panes of the lanterns with the mantle's glare in them.
//
// Glass has no texture: it is the baked light at the cover in `aux`. A pane that glows evenly is a paper card,
// though; a lit gas lantern is white-hot where the mantle is seen through the glass and falls to amber towards
// the glazing bars. The bake gives such a pane the vector to its burner and the burner's hue in the uv slots
// of its vertices - at EVERY level, and the glare has no distance term: a lantern must not change its face when
// its cell changes level or the camera comes closer. A tight core over a wider body around the line of sight to
// the burner (the halo's hot core lies over the same head: with both whole, the pane facing the camera is a
// white card that hides the mantle and the glazing bars), the even glow turned down to the amber of the pane's
// edges, more cover at a glancing angle. Far away the core is smaller than a pixel: both lobes are widened by
// the pixel's footprint at the same energy, so the lamp keeps its light without twinkling.
//
// Clear glass around a burner (the head of a gas lamp; the peak of the burner's light rides in the layer slot,
// which glass has no use for) is lit as any surface: the dirt on it takes the burner's light and the glass
// mirrors the burner - seen from outside that is the inside of the far panes, a frosted amber wall behind the
// mantle with the burner's highlight in it, while the pane in front stays clear. Per pixel: the light of that
// one burner on the face normal turned to the eye, the dirt of the bake's cover with the grain a vertex cannot
// hold, and the highlight no narrower than the pixel.

import * as THREE from 'three';
import { attribute, varying, texture, vec2, vec3, float, mix, step, max, exp, pow, abs, dot, sqrt, sign, floor, saturate, smoothstep, oneMinus, frontFacing, positionWorld } from 'three/tsl';
import { faceFrame } from './frame.js';
import { brdf } from './glints.js';

export function createGlass( ctx ) {

	const { stage, bake, U, pulled } = ctx;
	const { pl, aux, lit, V, fogged } = ctx.inputs;
	const material = new THREE.MeshBasicNodeMaterial( { fog: false, transparent: true, depthWrite: false } );
	material.name = 'baked glass (lantern glare)';

	const ga = attribute( 'uv', 'vec2' ).mul( 65535 );
	const hi = floor( ga.add( 0.5 ).div( 256 ) ), lo = ga.sub( hi.mul( 256 ) );
	const hg = floor( hi.y.add( 0.5 ).div( 16 ) );
	const has = step( 0.5, hg );
	const Lv = varying( vec3( lo.x, hi.x, lo.y ).sub( 128 ).mul( bake.bake.pane_r / 127 ).mul( has ), 'vPaneL' ); // pixel -> burner (m)
	const hv = varying( vec3( has, hg.div( 15 ), hi.y.sub( hg.mul( 16 ) ).div( 15 ) ), 'vPaneH' ); // has a burner, its green and blue over red
	const d2 = dot( Lv, Lv );
	const along = dot( Lv, V ); // (V looks back at the camera: the sign drops out in the square)
	const r2 = d2.sub( along.mul( along ) ).max( 0 ); // (distance of the burner from this pixel's line of sight)^2
	const { dpx, dpy, N0 } = faceFrame( V );
	const ndv = abs( dot( N0, V ) );
	const near = oneMinus( smoothstep( 0.2, 1.0, d2 ) ); // 1 within 0.45 m of the burner, 0 from 1 m
	const hue = vec3( 1, hv.y, hv.z );
	// (the pixel's footprint across the line of sight, m^2: what the lobes are widened by)
	const ax = dot( dpx, V ), ay = dot( dpy, V );
	const f2 = max( dot( dpx, dpx ).sub( ax.mul( ax ) ), dot( dpy, dpy ).sub( ay.mul( ay ) ) ).max( 0 ).mul( U.paneBlur );

	// --- a glowing pane: the mantle's glare
	const dirt = aux.sub( 0.22 ).div( 0.68 * 0.7 ).saturate(); // (the bake's cover = mix( 0.22, 0.9, dirt x 0.7 ))
	const cover = max( aux, mix( float( 0.22 ), float( 0.9 ), pow( oneMinus( ndv ), 4 ) ) );
	const k = hv.x.mul( near ).mul( U.pane );
	const sc = 0.04 * 0.04, sb = 0.085 * 0.085;
	const core = exp( r2.div( f2.add( sc ).mul( - 2 ) ) ).mul( float( sc ).div( f2.add( sc ) ) );
	const body = pow( r2.div( f2.mul( 2 ).add( sb ) ).add( 1 ), - 1.5 ).mul( float( sb ).div( f2.mul( 2 ).add( sb ) ) );
	const glare = mix( hue, vec3( 1 ), 0.45 ).mul( core.mul( U.paneCore ).mul( stage.HDR.core ) ).add( hue.mul( vec3( 1.0, 0.8, 0.52 ) ).mul( body.mul( U.paneBody ) ).mul( dirt.mul( 0.6 ).add( 0.8 ) ) );
	// (the baked pane is an even glow over the bake's cover: here the amber share of it over this pixel's cover)
	const even = vec3( 1.0, 0.82, 0.65 ).mul( U.paneEven ).mul( dirt.mul( 0.6 ).add( 0.8 ).add( ndv.mul( 0.5 ) ).div( dirt.mul( 0.6 ).add( 1.05 ) ) ).mul( aux.div( cover ) );
	const glowPane = lit.mul( mix( vec3( 1 ), even, k ) ).add( glare.mul( k ).div( cover ) );

	// --- a clear pane: the burner's light on its dirt, and the burner's image in the glass
	const pk = varying( pl.w.mul( 65535 / 64 ), 'vPanePk' ), isClear = step( 0.25, pk );
	const Ng = N0.mul( sign( dot( N0, V ) ) );
	const dl = sqrt( max( d2, 1e-4 ) ), Ld = Lv.div( dl );
	const capL = mix( float( 1 ), float( bake.look.cap_up ), smoothstep( - 0.42, 0.3, Ld.y.negate() ) );
	const rad = hue.mul( pk ).mul( capL ).div( max( d2, 0.01 ).add( U.headSoft.mul( bake.look.source_radius ** 2 ) ) );
	const wq = positionWorld;
	const grain = texture( stage.noise, vec2( wq.x.add( wq.z.mul( 0.73 ) ), wq.y.add( wq.z.mul( 0.41 ) ) ).mul( 9 ) );
	// (U.paneDirt: the least of it - the bake's dirt on these panes is a tenth of what a dirt texture would give them)
	const dirtC = max( aux.sub( 0.16 ).div( 0.74 * 0.7 ), U.paneDirt ).add( grain.r.sub( 0.5 ).mul( 0.15 ) ).add( grain.b.sub( 0.5 ).mul( 0.1 ) ).saturate();
	const roughC = max( mix( float( 0.05 ), float( 0.55 ), dirtC ), sqrt( f2 ).div( dl ).mul( 2 ) ).clamp( 0.12, 1 );
	// (the highlight is held to U.paneCap: a hand from the mantle the bare figure runs into the thousands, and the
	// bloom would take a whole facade with it. The mantle's image itself is the second, narrow lobe off the glass
	// under the dirt: the star at the mantle, held to U.paneStar)
	const W3 = vec3( 0.2126, 0.7152, 0.0722 );
	const hot = rad.mul( brdf( Ng, V, Ld, roughC, float( 0.04 ) ).mul( U.paneSpec ) ), hotL = dot( hot, W3 );
	const pin = rad.mul( brdf( Ng, V, Ld, sqrt( f2 ).div( dl ).mul( 2 ).clamp( 0.1, 1 ), float( 0.04 ) ).mul( U.paneSpec ) ), pinL = dot( pin, W3 );
	const lamp = rad.mul( vec3( 0.12, 0.11, 0.09 ).mul( dirtC ).mul( saturate( dot( Ng, Ld ) ) ).mul( 1 / Math.PI ).mul( U.paneLit ) ).add( hot.mul( U.paneCap.div( U.paneCap.add( hotL ).max( 1e-6 ) ) ) ).add( pin.mul( U.paneStar.div( U.paneStar.add( pinL ).max( 1e-6 ) ) ) );
	const coverC = max( aux, mix( float( 0.16 ), float( 0.9 ), max( pow( oneMinus( ndv ), 4 ), dirtC.mul( 0.7 ) ) ) );
	const kc = isClear.mul( hv.x ).mul( near ).mul( U.pane );

	material.colorNode = fogged( mix( glowPane, lit.add( lamp.mul( kc ) ), isClear ) );
	// (the far panes of a lamp's head are seen from inside: both faces are drawn, in one pass - of anything but
	// such a pane the back stays out)
	material.side = THREE.DoubleSide; material.forceSinglePass = true;
	material.opacityNode = mix( mix( aux, cover, k ), mix( aux, coverC, kc ), isClear ).mul( max( float( frontFacing ), isClear ) );
	material.positionNode = pulled( null ); // (glass must stay in front of the glowing card behind it)
	return material;

}
