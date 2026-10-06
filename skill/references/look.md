# The look: what made the night read, and what did not

> Everything here was found on a night scene lit by lamps. For a daylit or overcast environment the same methods
> apply (grey-card exposure check, path-traced reference, one fog path, HDR content before HDR effects, tune on one
> loaded page, the owner decides on their device); the night-specific items (glare sprites, lamp reach, neon level)
> become sun highlights, sky and shadow design.

Everything here was decided by pictures the owner saw on their own phone and laptop. Rules that fell out:
offer a softer and a stronger link beside every new setting; tune on one loaded page with uniforms, not reloads;
never ship a grade change to the live link before the owner has seen it on their HDR device; and when the owner
says a look is right, freeze it.

## Starting values (reached by over-correcting twice; start near them)

- Tone mapping ACES (chosen over AgX: "the golden look closest to the reference"); validate exposure numerically -
  an 18 % grey plane must land at pixel 128 (exposure 1.0, not the 1.7 first guessed).
- Dynamic Studio look: fog 0.0075, ambient 0.85 -> 0.8 after round 4, moon 2.0 -> 1.3, exposure 1.12, bloom 0.3,
  lamp < 1, `wet 0.5` ("the scene is seen just after rain"); night hemisphere (0.33, 0.40, 0.60), moon colour
  (0.62, 0.68, 0.87), `roofFill` 2.5; lamp reach falloff `1 / (1 + d² / 4.5²)`; lantern colour `#ffbf6a`.
- Lantern glass `mat("glass_lantern", "#ffa94e", glow=1.4, opacity=0.32)`; clear glass diffuse 0.02 (lanterns) /
  0.06 (other) or panes "blow out like paper"; emissive, glass, window and water materials never cast shadows, plus a
  per-light `shadow_near`.
- Baked tiers as shipped: contrast 1 (no grade), exposure 0.72, knee 2 (lit-wall ceiling), `kneeGlow` for glowing
  surfaces, `HDR.neon` 2.5, bloom 0.14, scatter 0.7, contact occlusion 0.6 at `pow 3`, ground shadows deepened
  (`gshpow 5, gshspan 3`), lens flare 0.25 **off by default**, mist none, pixel ratio 1, TRAA on everywhere.

## What worked, in the order it was found

1. **Wet paving with screen-space reflections** was the single biggest realism gain and the judge's top request;
   its first version was faulted for "razor-edged perfect mirrors": roughness 0.05-0.15 with rain micro-normals,
   puddles feathered over 10-20 cm with a damp darker halo, pool size capped and broken along joints, ring paving
   and flagstones given the same damp sheen as cobbles. "Confident SSR" rejects hits far more saturated than
   plausible (stained-glass "confetti"); probes need mips (`initRenderTarget` before the first capture) and
   roughness-driven LOD; contact-shadow pads under furniture also block the reflection beneath them.
2. **Soft near-field falloff** (`sourceRadius` in the vendored clustered lights) instead of dimmer lamps: walls
   beside street lamps burning to pale yellow went from 4.6 % clipped pixels to 0.04 % with the amber-on-navy
   contrast kept. Capped lanterns (`capFactor`: 25 % upward) stop facades being "lit to the eaves". "Ambient was not
   the flood; the lamps were."
3. **Cooler moon against warmer lanterns**, and the stone materials re-baked with a pitted limestone micro-normal
   instead of brushed grain; the plaza stairs brought into the same cream family (judges flag mismatched hue
   families side by side).
4. **Glare sprites only on fittings** (name matches `lantern|lamp|globe|fire|festoon|flame|candle|bulb|torch|
   mantle`, ~0.3 m additive, sized from sqrt(candela), at the fitting's declared position, lifted in front of a
   globe's glass). Clear-glass lamps with a small flame read as unlit from a distance without them; a glare at every
   light socket drew orbs in mid-air in front of shop windows; 22 of 113 glows once belonged to invisible helper
   lights. Globe lamp posts got one per globe ("The globe lamp posts look great").
5. **Lamp reach decided by a path-traced reference** (`tools/cyclesref/`: the same assets, lanterns and judged
   cameras in Cycles on the GPU, reusing the bake's loaders and look constants, with the page's exposure, knee and
   grade ported to numpy so the pictures compare). It overturned the prediction: true bounce is only 5-8 % of the
   light; the gap was lamp light cut off at 9 m with softened falloff ("warm pools in cold blue-grey darkness" while
   path-traced paving was ~2x brighter and warmer), then missing lamp sheen on wet paving (27-35 % of the
   path-traced picture), weak contact shadows, cold fill. Low was closer to Cycles than High because its bake used
   plain inverse-square falloff. Physical reach (`reach 0, lamp 0.43, range_scale 3`) moved the paving means to the
   Cycles level on every preset; the owner kept it ("I like the physical lamp reach") and asked for contrast from
   the grade instead.
6. **Natural bloom and scattering fog** (`web/engine/natbloom.js`, after Sonic Ether's MIT Unity effect; the
   Minecraft pack was read for the idea only - its licence forbids reuse): blur the frame into a pyramid with no
   threshold; bloom is a blend *towards* the blurred sum (`mix(colour, blurred, 0.14)`), not light added; the fog
   reads the same pyramid again, blending each pixel towards the blur by its fog share, so fog smears light instead
   of only hiding it. Weights by octave, not level index (or Phone looks washed out); the fog share rides in the
   alpha of TRAA's input and TRAA's resolve must have `blending = NoBlending` or the alpha vanishes silently; levels
   tied to a 720-high picture. The owner asked for 70 % of the first tuning. Lens dirt is a third reader.
7. **HDR content before HDR effects**: `?show=hdr` (grey under white, a colour per stop above) showed every lit
   surface under 1 and mantles at 1-2 stops - an energy-conserving bloom had nothing to spread. Expand emitters at
   run time: `k = min(glow * max(L, 1)^hot, max(top / L, 1))` (1.5, 2.5, 48) so the brighter a thing was authored,
   the more it gains; mantles then sit 3-5 stops over white, windows 1-2, and halos, wet streaks and fog glow follow
   without separate tuning. HDR canvas (`outputType: HalfFloatType`, `toneMapping: { mode: 'extended' }`, only when
   `matchMedia('(dynamic-range: high)')` matches): ACES times a gain rising with stops over white, then sRGB. A
   headless test machine cannot show it; the owner's iPhone did.
8. **One fog path and a sky in the world**: two layers thinning with height (night haze with an 18 m scale height,
   squared in the exponent so the first tens of metres stay clear; optional ground mist), integrated along the line
   of sight, shared by the materials, the scatter and the sky; the sky is a 600 m dome written to depth and drawn
   last, painted by direction, fogged like any front. Mist rows at 0.012-0.03 all read as "one blue-grey veil" - the
   owner chose none; flat mist at night only looks right where a lamp lights it, which is the volumetric pass a
   phone cannot pay.
9. **Signs and neon at their own level**: every glowing surface rolled off to one ceiling, so a neon tube ended no
   brighter than a lit window. `HDR.neon` 2.5 multiplies light and ceiling for what glows in a colour no flame has
   (red >= green >= blue is a flame; cyan, violet, pink are not), plus an asset-declared `sign=1` flag for gold
   letters and chaser bulbs in flame colours, and the same factor on the coloured entries of the bake's light list
   so their streaks follow. Raising only the ceiling changed nothing. Judge gold at 2.5x on the HDR phone; in SDR it
   rolls off to cream.
10. **Contact occlusion and lantern shadows on the ground** (`bake-runtime.md`): invisible at their measured
    strength on wet night paving; read only when applied to the mirror and glints as well and deepened locally
    against a coarse mip. The owner chose the softer contact setting and accepted the deliberately exaggerated
    shadow: "It all looks great."
11. **Lens flare**: threshold the full-size picture (four taps per quarter pixel, each at 4 scene units, the
    ceiling of every lit surface, so only halo cores and flames pass), roll each tap off towards a ceiling, one
    nine-tap Gaussian at quarter size, compress on the way out (`c * (max / R)^(curve - 1)`, curve 0.5) so a lamp
    with a tenth of the light gives a third of the ghost. The first version read a blurred level over a threshold:
    far lamps never flared, a lamp overhead glared ("it only activates when I get near a light and then it's too
    strong"). Judge any bright-source effect from directly under a lamp as well as from across the square. Shipped
    at 0.25, switched by a uniform in the quality menu, off by default after the owner lived with it.
12. **Water**: dark body with almost no diffuse response, Fresnel mirroring, several octaves of crested anisotropic
    ripples (mip-mapped so they fade instead of aliasing), lanterns broken into columns of dashes; calm ripples at
    about half the slope that looks right in a still (`swell` 0.55) give the long golden streaks, more reads as chop.
    Presets without the mirror pass draw each nearby lantern as its mirror image at constant radiance from a light
    list of their own (nearest lanterns, unmerged, counted far beyond lighting range). Knobs handed to the owner:
    `?u_swell u_wmirror u_wrough u_wglow u_wbody u_wwall`. Blending a sky colour and a scene colour by
    low-frequency noise reads as clouds; fountain water as "white strands" or "toothpaste" was flagged in every
    round until strands with irregular lower edges, splash rings, foam cards and wet-darkened lips. Boats are dry
    inside (modelled bilge water read as the canal coming through the hull) and declare a waterline outline that the
    terrain cuts out of the water sheet.
13. **Foliage lighting**: a hemisphere floor added after AO (cool sky on tops, warm bounce from lit paving below),
    wrap and transmission on both faces, `normalNode * faceDirection` so back faces keep the crown normal. Stacked
    occlusion terms (painted vertex AO x texture AO x GTAO x rain darkening x leaf albedo 0.03) had printed canopies
    at 0.001-0.08 % luminance against a 0.45 % sky; low ambient was not the cause. Sample pixel luminance per term
    before touching constants.
14. **TRAA on every preset** (the owner: no toggle; `?traa=0` only for debugging); pixel ratio 1 everywhere; the
    ground-mirror target antialiased (half-float, mips, linear filtering) - and still the belfry louvres had to be
    remodelled as wide dim bands because 26 mm glowing slits flash whatever the renderer does.

## What did not work (so it is not retried without new numbers)

- Lamp-lit volumetric fog: the top missing "AAA technique", ruled out by the owner for phones. Do not build; the
  scatter share, flare and grade carry the air. (If asked for god rays: bake volumetric shafts into a coarse 3D
  texture from the same lantern visibility the vertex bake uses and ray-march it at quarter resolution in the fog
  pass; screen-space radial shafts are a one-day fallback that fails off-screen.)
- Ground mist (any strength). Baked glow cards on far backdrop rows ("stickers"). Karis averaging (dims exactly the
  small bright lamps the picture is about). The 13-tap / tent pyramid (this one measures within 0.5 % and shows no
  blocks). Softening wet-paving glints with distance ("removed detail the reference actually has"). Lighting a globe
  from inside (darker and blotchy). A harsh grade (contrast 1.5, exposure 0.56, knee 4) that looked right in SDR
  captures: "too dark and too harsh" on the HDR phone four hours later - on an HDR display the display supplies the
  contrast; leave the exposure alone and hand over `?contrast=&knee=` links. A second lighting look as a shipped
  option (+34-43 % bytes for a choice the visitor does not need). Lightmaps (measured, `bake-runtime.md`).
  Light probes from Blender (2 % error, 0.01 s per probe, but three.js's grid has no visibility test: a surface
  behind a 0.4 m wall read 12x too bright).

## Reading the owner's feedback

- "Too X" on the phone beats any desktop capture; the two differ on sharpness (pixel ratio), on contrast (HDR) and
  on brightness of gold (roll-off). Reproduce at the owner's spot and canvas size (402 x 684 CSS px, FOV 62 via
  `Emulation.setDeviceMetricsOverride`) with the switch on and off side by side.
- The owner answers labelled rows by panel name ("I like n0a the best"), sends comparison strips back as images,
  and halves the first strength of a new effect more often than not (flare 0.5 -> 0.25, bloom scatter -30 %, contact
  occlusion 1.0 -> 0.6). Start new effects slightly under where they look right to you.
- "I like both looks" is an answer: offer the cheaper of (a) rebake with one, (b) switchable with a cost table.
- Ranked "what is missing" lists get picked from: volumetric fog (vetoed), lamp shadows on the ground and light
  hierarchy (both then requested).
