#!/usr/bin/env node
// bake2glb.mjs - the baked town (tools/webbake.py: manifest.json + base.bin + m_/n_/r_/f_<cell>.bin) as glTF 2.0.
//
// USAGE
//   node tools/bake2glb.mjs [<bake dir>] --mode exact|portable --level lo|mid|near|ring|fine [options]
//   node tools/bake2glb.mjs [<bake dir>] --mode exact|portable --lods auto|fine,near,mid,lo [--tier high] [options]
//
//   node tools/bake2glb.mjs <bake dir> --live [--verify] [--force] [--prune] [--no-lean] [--drop-packed]   # the files the town page loads
//
//   <bake dir>            folder with manifest.json (default out/web/_next; out/web/_next/lean is the low / phone bake)
//   --live                the bake's geometry as the .glb files the walk page loads (web/engine/lite_glb.js), written INTO
//                         the bake directory beside the packed files: base.glb (the lowest level of every cell, in the
//                         manifest's order, so the page builds cells while the file arrives) and one <m|n|r|f>_<cell>.glb
//                         per level file - exact mode, the packer's streams byte for byte. A file is written again only
//                         when it is missing, cut short, of another bake (the stamp in its JSON), or older than its
//                         .bin; --force writes all. The manifest gets "glb": { v, base: { file, bytes, json }, files,
//                         bytes } (the rest of its text is left as it is): that is what tells the page the files are
//                         there. With --verify every file (written or kept) is read back: decoded bytes, draw ranges
//                         and placement against the bake, and the streams in its binary chunk byte for byte against
//                         the packed file. A lean bake the manifest names (<bake dir>/lean) is done too unless
//                         --no-lean. --prune deletes .glb files of this mode that no level of the manifest stands for.
//                         --drop-packed (implies --verify): once every .glb is read back, the packed .bin files are
//                         deleted and the manifest's record gets "only": true - the page then has no packed files to
//                         fall back to, and this tool reads a level's bytes out of its .glb wherever it would have
//                         read the .bin (every export of such a bake works as before, --buffers bake aside).
//                         tools/webbake.py runs this at the end of every bake.
//   --mode exact          the page's own data, bit for bit: the u16 vertex records as accessors of ONE interleaved view
//                         (POSITION = slots 0-2 with KHR_mesh_quantization and the node's translation / scale,
//                         _PL = slots 0-3 (.w: texture layer), TEXCOORD_0 = slots 4-5, _COL = slots 6-9 (light, aux),
//                         _G = slots 10-11 on the fine level; a bake with a second look: _COL2 = its light, four
//                         slots from 10 - from 12 after _G - and on the fine level _G2 = slots 16-17), the packer's
//                         meshopt streams copied as they are,
//                         one primitive per draw range (extras: kind, size steps, box), the manifest's level entry in
//                         the node's extras, the manifest's globals in the scene's. Needs the page's material to look
//                         like the town; a plain viewer shows the grey geometry.
//   --mode portable       for any glTF tool: float POSITION (metres, relative to the cell's node), TEXCOORD_0 in
//                         texture tiles, COLOR_0 = the baked light decoded to linear, one KHR_materials_unlit material
//                         per texture layer with the layer as baseColorTexture, glass / falling water blended,
//                         leaf cards alpha-masked. No custom attributes, no compression unless asked for.
//   --level L             one level of every chosen cell (cells that do not have it are left out)
//   --lods LIST|auto      several levels per cell, linked by MSFT_lod: the finest is the node in the scene, the others
//                         hang on its extensions.MSFT_lod.ids; extras.MSFT_screencoverage and extras.lod (plain
//                         distances for THREE.LOD) come from the tier table and the switch rule of the page
//                         (BAKED_TIERS and update() of web/engine/lite.js). auto: the tier's own levels.
//   --tier T              phone|low|medium|high (default high): whose distances --lods writes
//   --lod-dist a=b,...    override distances, e.g. fine=30,near=72,mid=180 (metres on the ground to the cell's box)
//   --tiers-from FILE     where BAKED_TIERS is read from (default web/engine/lite.js beside this tool)
//   --cells all|K,K|spawn:R|at:X,Z,R
//                         which cells: all, a list of keys, or those within R metres (on the ground, to the cell's
//                         box) of the manifest's spawn point / of X,Z (three.js frame)
//   --split town|cell     one file for all chosen cells (default) or one per cell (+ <name>.index.json)
//   --out DIR             default out/web/_nextglb
//   --name STEM           file name stem (default <mode>_<level | lod-<tier>>)
//   --container glb|gltf  .glb (default) or .gltf + .bin
//   --buffers one|level|bake
//                         one: everything in one buffer (default). level (.gltf): the lowest level of all cells in one
//                         .bin, every other level of a cell in a .bin of its own - a loader that asks for buffers when
//                         it needs them can show the town before the finer levels are down. bake (.gltf, exact mode):
//                         no geometry is copied at all, the .gltf points at the bake's own base.bin and level files.
//   --order cell|coarse   the order of the data in the buffers (default cell: a cell's levels one after the other).
//                         coarse: the lowest level of all cells first, then level by level towards the finest, so
//                         the front of a single .glb is all that is needed for a first picture (HTTP range requests).
//   --meshopt auto|khr|ext|ext-v0|none
//                         exact: auto (default) keeps the packer's streams and names the extension after their codec
//                         version (version 1 -> KHR_meshopt_compression, version 0 -> EXT_meshopt_compression);
//                         khr / ext force the name on the same bytes (three.js decodes either; a version-1 stream
//                         under the EXT name is outside that extension's text); ext-v0 re-encodes the vertex stream
//                         at version 0 (larger, valid EXT); none stores the decoded bytes.
//                         portable: none (default), or ext / khr to compress the float streams (Blender cannot read
//                         either).
//   --textures auto|jpeg|png|webp|none   (portable)
//                         auto (default): JPEG, PNG for the layers a leaf material cuts by alpha - core glTF, read by
//                         everything. webp: EXT_texture_webp (required); single-layer files of the bake are embedded
//                         byte for byte, atlas tiles are cropped and re-encoded. none: vertex light only.
//   --tex-size N          layer size to take (default 512; sizes the bake has: manifest tex / texl)
//   --tex-quality Q       JPEG / WebP quality of converted layers (default 90)
//   --tex-store embed|external   in the file (default) or as files in <out>/tex/
//   --color float|u16|u8  (portable) COLOR_0 type (default float). The light is multiplied by --exposure and clamped
//                         at 1: glTF allows no vertex colour above 1 (the validator calls it an error).
//   --hdr                 (portable, float) keep the light above 1 as it is (the bake's range is 0 .. colour_scale;
//                         lamp glass, lit windows and the paving under a lantern are above 1). Not valid glTF, but
//                         three.js and Blender take the numbers as they are.
//   --exposure X          factor on the light before it is stored (default 1)
//   --lights none|punctual|empties   the manifest's lanterns within the chosen cells as KHR_lights_punctual point
//                         lights or as empty nodes (default none; the intensity is the bake's number, not candela)
//   --extras min|full     full adds the manifest's lights, halos, flames and data to the scene's extras
//   --verify              exact: read the written file back, decode every view and compare every vertex and index
//                         byte with the decoded level of the bake; fails (exit 2) on any difference.
//                         portable: recomputes triangle counts and bounds against the bake.
//   --check FILE          no export: verify an existing file of this tool against the bake (as --verify does)
//   --gzip                also print the gzip -6 size of every file written
//   --quiet
//
// EXAMPLES
//   node tools/bake2glb.mjs out/web/_next --live --verify                          # what the page loads: write / refresh, check
//   node tools/bake2glb.mjs --mode exact --level lo --verify                       # base.bin as one GLB
//   node tools/bake2glb.mjs --mode exact --level fine --cells spawn:20 --split cell --verify
//   node tools/bake2glb.mjs --mode portable --level near --cells spawn:20 --lights punctual
//   node tools/bake2glb.mjs --mode portable --lods auto --tier high --name town_lod  # whole town, one LOD chain per cell
//   node tools/bake2glb.mjs --mode exact --lods auto --container gltf --buffers bake # a .gltf over the bake's own files
//
// PROFILE dcc: the portable file for Blender AND three.js (stock importer / stock GLTFLoader + DRACOLoader, no custom code)
//   node tools/bake2glb.mjs [<bake dir>] --profile dcc --level lo|mid|near|fine|best [options]
//   node tools/bake2glb.mjs [<bake dir>] --profile dcc --lods auto|LIST [--tier high] [options]
//
//   Only what both programs read (tested, see docs): KHR_draco_mesh_compression or KHR_mesh_quantization +
//   KHR_texture_transform, KHR_materials_unlit, EXT_texture_webp (required, no fallback image), optionally
//   KHR_materials_emissive_strength, KHR_lights_punctual; MSFT_lod in extensionsUsed only. Triangles are merged by
//   material inside a spatial unit (--grain), every texture is stored once, the data lies coarse-first.
//   --level L             the whole town at one level; a cell that does not have L takes its next coarser one
//                         (best = fine: every cell at the finest level it has)
//   --lods LIST|auto      a chain per spatial unit, linked by MSFT_lod: the finest is the node in the scene, the coarser
//                         ones hang under the nodes LOD1_<level>, LOD2_<level> ... which are in no scene (the stock
//                         three.js loader never touches them; Blender imports them as hierarchies to hide or delete)
//   --grain town|region:N|cell
//                         what becomes one object (one primitive per material in it): the whole town, N x N cells, or
//                         a cell. Default: region:8 for --level lo, region:4 for the other levels, region:2 for --lods
//                         (a level switches per object). Larger objects: fewer draw calls and a quicker first
//                         frame in three.js, less culled by the view, a slower import in Blender.
//   --geometry draco|quant|float
//                         draco (default): KHR_draco_mesh_compression; positions on the bake's own 1/1024 m grid, UVs on
//                         1/1024 of a tile - nothing is lost against the bake but what --color takes. quant: uncompressed
//                         KHR_mesh_quantization (u16 positions, on the bake's grid while an object spans 64 m or less;
//                         u16 UVs through KHR_texture_transform), for gzip / brotli transfer. float: core glTF floats.
//   --draco-level N       0 .. 10 (default 7): smaller and slower to encode and decode
//   --pos-drop N          (draco) positions on a grid 2^N times coarser than the bake's (1: 2 mm, 2: 4 mm), --uv-drop N the
//                         same for the UVs (2: 1/256 of a tile). Lossy, so not with --verify. Every vertex is rounded on
//                         one grid for the whole town: what met before still meets. NOT for a file anyone looks at:
//                         two surfaces a millimetre or two apart (a sign on its wall, a card behind its window) land
//                         in one plane on the coarser grid and fight for the depth buffer. Kept for size experiments.
//   --textures webp|none  (default webp) one image per texture layer, alpha kept only where a leaf material cuts by it
//   --tex-size N          default 512 for --level lo and for --lods, 1024 otherwise; --tex-quality Q (default 80)
//   --color u8|u16        COLOR_0 = baked light * --exposure, clamped at 1. u8 (default): 8 bits a channel, 37 % less
//                         geometry than u16 in a Draco file; against u16 a picture differs by 0.4 / 255 on average
//                         (dark surfaces move by a step or two). u16: the bake's light to the last bit
//   --clip hue|soft|channel  light above 1 (after the layer's gain; blended glass first moves it into its alpha, which
//                         keeps colour * alpha): hue (default) divides the colour by its largest channel, so a lit
//                         window stays orange, as the page's highlight knee keeps it; soft also pales it towards white
//                         the brighter it was (a plain tone curve's look); channel clamps each channel by itself (a
//                         strong orange turns yellow, lamp glass green-yellow)
//   --lamps lit|baked     lantern panes (glass with a burner behind it). The page lights them in its shader: the mantle's
//                         glare around the line of sight, the burner's light on the dirt of clear panes - none of it
//                         is in the baked light, and clear panes are baked black. lit (default) writes that glare as
//                         seen square-on into the panes' COLOR_0 (white-hot before the mantle, amber at the glazing
//                         bars), clear panes as frosted ones, and lets the alpha rise with the light: a lit pane
//                         covers the dark inside of its lantern. On fine, ring and near the large triangles of a pane
//                         are cut (6 / 12 cm) so that its vertices can hold the hot centre - the only triangles the
//                         file has that the bake does not; on mid and lo a pane keeps its sum. Opal globes (the page
//                         draws their halo before them, the bake has them as pale surfaces) take their lamp's light.
//                         baked: all of it as the bake has it
//   --glow clamp|emissive glowing cards (lit rooms, lantern mantles): clamp (default) keeps them in COLOR_0 at 1;
//                         emissive moves them to <layer>_glow materials with KHR_materials_emissive_strength (one
//                         strength and colour per material: the light above 1 survives, its variation does not)
//   --cut-alpha X         alphaCutoff of the leaf materials (default: from the layer's alpha, same cover as the page)
//   --ground fold|shadows|ao|none
//                         the two pictures the page reads per pixel on the ground (bake.gao: the dark rim where things
//                         stand on the paving; bake.gsh: the lanterns' shadows - the ground's vertices hold their light
//                         WITHOUT shadows). A stock viewer has no such shader, so fold (default where the bake has the
//                         pictures) multiplies the page's factor into COLOR_0 of what faces up near the terrain, with
//                         the page's numbers (occlusion: 0.6 of the open share cubed; shadows deepened where they are
//                         local). On the finest level of the file (fine or near) the triangles are cut where a vertex
//                         every 0.6 m cannot hold the factor: an edge is halved while it is longer than --ground-edge
//                         (default 0.16 m) and the factor along it leaves the straight line between its ends by more
//                         than --ground-tol (default 0.04). That is decided from the edge's two places alone, so every
//                         triangle that has the edge cuts it alike and no gap opens. The coarser levels take the factor
//                         per vertex, averaged over 0.25 / 0.5 / 1 m (near / mid / lo). shadows, ao: one of the two;
//                         none: the bake's vertices as they are (the ground without the lanterns' shadows).
//                         --ground-edge 0: no cutting, every level per vertex.
//   --sky mesh|none       the night sky of the town page. glTF has no background or environment a stock importer
//                         shows, so mesh (default) writes it as one more object: the node `sky` in the scene (never in
//                         a LOD chain), an unlit material without COLOR_0, one WebP panorama of the page's sky (gradient,
//                         town glow, cloud deck, moon, stars; a still frame) and a shell just around the file's
//                         content: a box with rounded edges standing on a floor 1 m under the lowest point, 4 m clear
//                         of every cell, mapped by direction from eye height at the middle of the plaza (so it is
//                         right from there and slides by some degrees as one walks; the page's sky follows the camera).
//                         Its triangles face inwards (single-sided: seen from outside, three.js and Blender's material
//                         preview look through it). scenes[0].extras.baked.sky says where it is, for an application
//                         that would rather use the image as a background at infinity and hide the node.
//                         The image is a plain equirectangular panorama, zenith in the top row, so it also serves as
//                         one: in Blender, World > Environment Texture with this image and no rotation (checked in 4.1).
//   --sky-size W          width of the panorama (2 : 1, default 4096: a texel of 0.09 degrees, about a pixel of the page,
//                         which keeps the stars points; 2048 with --sky-quality 95 is 53 kB less and a quarter of the
//                         memory, with the stars as soft blobs); --sky-quality Q WebP quality (default 90; 100: lossless)
//   --sky-min D           the shell stays at least D metres from its centre (default 140: nearer, the sky slides
//                         more than 20 degrees on a walk across the plaza); only a file of a few cells is made larger by it
//   --lights, --exposure, --cells, --out, --name, --verify, --check, --gzip as above
//
//   node tools/bake2glb.mjs <bake dir> --profile dcc --level lo   --name town_lo   --verify --gzip
//   node tools/bake2glb.mjs <bake dir> --profile dcc --level best --name town_best --verify --gzip
//   node tools/bake2glb.mjs <bake dir> --profile dcc --lods auto  --name town_lod  --verify --gzip
//
// Texture conversion (portable: anything but single-layer webp; dcc: always) calls ImageMagick (`magick`, or `convert`).
// --geometry draco needs the npm package draco3dgltf (a devDependency of this project).
// Everything about the bake is read from its manifest; nothing here is tied to one bake.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';

const HERE = path.dirname( fileURLToPath( import.meta.url ) );
const ROOT = path.resolve( HERE, '..' );
const KIND = [ 'opaque', 'glass', 'water', 'fall', 'cut' ];
const K_OPAQUE = 0, K_GLASS = 1, K_WATER = 2, K_FALL = 3, K_CUT = 4;
const LEVELS = [ 'lo', 'mid', 'near', 'ring', 'fine' ];
const FOV = 62, ASPECT = 16 / 9; // the page's camera, for MSFT_screencoverage
const CUT_ALPHA = 0.665; // the leaf mask of the page at the middle of a card: alpha * 2 - 1 > 0.33

// ------------------------------------------------------------------------------------------------ arguments

function parseArgs( argv ) {

	const o = { dir: null, mode: 'exact', level: null, lods: null, tier: 'high', lodDist: {}, tiersFrom: path.join( ROOT, 'web/engine/lite.js' ),
		cells: 'all', split: 'town', out: path.join( ROOT, 'out/web/_nextglb' ), name: null, container: 'glb', buffers: 'one', order: 'cell', meshopt: null,
		textures: 'auto', texSize: 512, texQuality: 90, texStore: 'embed', color: 'float', exposure: 1, lights: 'none', extras: 'min',
		verify: false, gzip: false, quiet: false, check: null, hdr: false, live: false, force: false, prune: false, noLean: false, dropPacked: false,
		grain: null, geometry: 'draco', dracoLevel: 7, posDrop: 0, uvDrop: 0, glow: 'clamp', clip: 'hue', lamps: 'lit', cutAlpha: null, ground: null, groundEdge: 0.16, groundTol: 0.04, sky: null, skySize: 4096, skyQuality: 90, skyMin: 140, given: new Set() };
	const flags = { '--verify': 'verify', '--gzip': 'gzip', '--quiet': 'quiet', '--hdr': 'hdr', '--live': 'live', '--force': 'force', '--prune': 'prune', '--no-lean': 'noLean', '--drop-packed': 'dropPacked' };
	const vals = { '--mode': 'mode', '--level': 'level', '--lods': 'lods', '--msft-lod': 'lods', '--tier': 'tier', '--tiers-from': 'tiersFrom', '--cells': 'cells', '--split': 'split', '--out': 'out',
		'--name': 'name', '--container': 'container', '--buffers': 'buffers', '--order': 'order', '--meshopt': 'meshopt', '--textures': 'textures', '--tex-size': 'texSize', '--tex-quality': 'texQuality',
		'--tex-store': 'texStore', '--color': 'color', '--exposure': 'exposure', '--lights': 'lights', '--extras': 'extras', '--check': 'check',
		'--profile': 'mode', '--grain': 'grain', '--geometry': 'geometry', '--draco-level': 'dracoLevel', '--pos-drop': 'posDrop', '--uv-drop': 'uvDrop', '--glow': 'glow', '--clip': 'clip', '--lamps': 'lamps', '--cut-alpha': 'cutAlpha', '--ground': 'ground', '--ground-edge': 'groundEdge', '--ground-tol': 'groundTol', '--sky': 'sky', '--sky-size': 'skySize', '--sky-quality': 'skyQuality', '--sky-min': 'skyMin' };
	for ( let i = 0; i < argv.length; i ++ ) {

		const a = argv[ i ];
		if ( a === '-h' || a === '--help' ) { const L = fs.readFileSync( fileURLToPath( import.meta.url ), 'utf8' ).split( '\n' ); console.log( L.slice( 1, L.findIndex( ( l, i ) => i > 0 && ! l.startsWith( '//' ) ) ).map( ( l ) => l.replace( /^\/\/ ?/, '' ) ).join( '\n' ) ); process.exit( 0 ); }
		if ( flags[ a ] ) { o[ flags[ a ] ] = true; continue; }
		if ( a === '--lod-dist' ) { for ( const kv of argv[ ++ i ].split( ',' ) ) { const [ k, v ] = kv.split( '=' ); o.lodDist[ k ] = parseFloat( v ); } continue; }
		if ( vals[ a ] ) { if ( i + 1 >= argv.length ) die( `${a} needs a value` ); o[ vals[ a ] ] = argv[ ++ i ]; o.given.add( vals[ a ] ); continue; }
		if ( a.startsWith( '--' ) ) die( `unknown option ${a} (see --help)` );
		if ( o.dir ) die( `one bake dir, got also ${a}` );
		o.dir = a;

	}

	o.dir = path.resolve( o.dir || path.join( ROOT, 'out/web/_next' ) );
	o.out = path.resolve( o.out );
	o.tiersFrom = path.resolve( o.tiersFrom );
	for ( const k of [ 'texSize', 'texQuality', 'exposure' ] ) { o[ k ] = parseFloat( o[ k ] ); if ( ! isFinite( o[ k ] ) ) die( `--${k}: not a number` ); }
	const one = ( k, list ) => { if ( ! list.includes( o[ k ] ) ) die( `--${k}: one of ${list.join( ', ' )}` ); };
	one( 'mode', [ 'exact', 'portable', 'dcc' ] ); one( 'split', [ 'town', 'cell' ] ); one( 'container', [ 'glb', 'gltf' ] ); one( 'buffers', [ 'one', 'level', 'bake' ] ); one( 'order', [ 'cell', 'coarse' ] );
	one( 'textures', [ 'auto', 'jpeg', 'png', 'webp', 'none' ] ); one( 'texStore', [ 'embed', 'external' ] ); one( 'color', [ 'float', 'u16', 'u8' ] );
	one( 'lights', [ 'none', 'punctual', 'empties' ] ); one( 'extras', [ 'min', 'full' ] );
	if ( o.mode !== 'dcc' && ( o.given.has( 'ground' ) || o.given.has( 'groundEdge' ) || o.given.has( 'groundTol' ) ) ) die( '--ground, --ground-edge, --ground-tol belong to the dcc profile' );
	if ( o.mode === 'dcc' ) {

		// the dcc profile's own defaults and limits (see the header)
		if ( o.ground !== null ) one( 'ground', [ 'fold', 'shadows', 'ao', 'none' ] );
		for ( const k of [ 'groundEdge', 'groundTol' ] ) { o[ k ] = parseFloat( o[ k ] ); if ( ! isFinite( o[ k ] ) ) die( `--${k}: not a number` ); }
		if ( ! ( o.groundEdge === 0 || ( o.groundEdge >= 0.02 && o.groundEdge <= 10 ) ) ) die( '--ground-edge: 0 (no cutting) or 0.02 .. 10 m' );
		if ( ! ( o.groundTol >= 0.005 && o.groundTol <= 1 ) ) die( '--ground-tol: 0.005 .. 1' );
		if ( ! o.given.has( 'textures' ) || o.textures === 'auto' ) o.textures = 'webp';
		if ( ! o.given.has( 'color' ) ) o.color = 'u8';
		if ( ! o.given.has( 'texQuality' ) ) o.texQuality = 80;
		if ( ! o.given.has( 'texSize' ) ) o.texSize = o.level === 'lo' || o.lods ? 512 : 1024; // (a chain is read front first: 512 px keeps that front small)
		if ( o.level === 'best' ) o.level = 'fine';
		if ( o.grain === null ) o.grain = o.lods ? 'region:2' : o.level === 'lo' ? 'region:8' : 'region:4';
		const m = /^region:(\d+)$/.exec( o.grain );
		if ( m ) { o.grainN = parseInt( m[ 1 ] ); if ( o.grainN < 1 ) die( '--grain region:N: N from 1' ); } else if ( o.grain !== 'town' && o.grain !== 'cell' ) die( '--grain: town, region:N or cell' );
		one( 'geometry', [ 'draco', 'quant', 'float' ] ); one( 'glow', [ 'clamp', 'emissive' ] ); one( 'clip', [ 'hue', 'soft', 'channel' ] ); one( 'lamps', [ 'lit', 'baked' ] ); one( 'textures', [ 'webp', 'none' ] ); one( 'color', [ 'u16', 'u8' ] );
		o.dracoLevel = parseInt( o.dracoLevel ); if ( ! ( o.dracoLevel >= 0 && o.dracoLevel <= 10 ) ) die( '--draco-level: 0 .. 10' );
		for ( const k of [ 'posDrop', 'uvDrop' ] ) { o[ k ] = parseInt( o[ k ] ); if ( ! ( o[ k ] >= 0 && o[ k ] <= 4 ) ) die( '--pos-drop, --uv-drop: 0 .. 4' ); }
		if ( ( o.posDrop || o.uvDrop ) && ( o.geometry !== 'draco' || o.verify ) ) die( '--pos-drop, --uv-drop: with --geometry draco, and not with --verify (the file is no longer the bake\'s, vertex for vertex)' );
		if ( o.cutAlpha !== null ) { o.cutAlpha = parseFloat( o.cutAlpha ); if ( ! ( o.cutAlpha > 0 && o.cutAlpha < 1 ) ) die( '--cut-alpha: between 0 and 1' ); }
		if ( o.sky === null ) o.sky = o.textures === 'none' ? 'none' : 'mesh'; // (the sky is its image)
		if ( o.sky === 'mesh' && o.textures === 'none' ) die( '--sky mesh needs --textures webp' );
		one( 'sky', [ 'mesh', 'none' ] );
		for ( const k of [ 'skySize', 'skyQuality', 'skyMin' ] ) { o[ k ] = parseFloat( o[ k ] ); if ( ! isFinite( o[ k ] ) ) die( `--${k}: not a number` ); }
		if ( ! ( o.skySize >= 64 && o.skySize <= 16384 && o.skySize % 2 === 0 ) ) die( '--sky-size: an even width from 64 to 16384' );
		if ( ! ( o.skyQuality >= 1 && o.skyQuality <= 100 ) ) die( '--sky-quality: 1 .. 100' );
		if ( ! ( o.skyMin >= 0 && o.skyMin <= 900 ) ) die( '--sky-min: 0 .. 900 m' );
		if ( o.hdr ) die( '--hdr: not in the dcc profile (it is not valid glTF)' );
		if ( o.split !== 'town' || o.buffers !== 'one' || o.meshopt ) die( 'the dcc profile writes one file with one buffer and no meshopt (Blender reads neither meshopt extension)' );
		o.meshopt = 'none';

	} else if ( o.grain !== null || o.given.has( 'geometry' ) || o.given.has( 'glow' ) || o.given.has( 'clip' ) || o.given.has( 'lamps' ) || o.given.has( 'dracoLevel' ) || o.cutAlpha !== null || o.sky !== null || o.given.has( 'skySize' ) || o.given.has( 'skyQuality' ) || o.given.has( 'skyMin' ) ) die( '--grain, --geometry, --draco-level, --glow, --clip, --lamps, --cut-alpha, --sky belong to --profile dcc' );
	if ( o.meshopt === null ) o.meshopt = o.mode === 'exact' ? 'auto' : 'none';
	one( 'meshopt', o.mode === 'exact' ? [ 'auto', 'khr', 'ext', 'ext-v0', 'none' ] : o.mode === 'dcc' ? [ 'none' ] : [ 'none', 'ext', 'khr' ] );
	if ( o.live ) {

		// (everything about these files is fixed: the page's reader relies on it)
		for ( const k of [ 'mode', 'level', 'lods', 'cells', 'split', 'out', 'name', 'container', 'buffers', 'order', 'meshopt', 'check' ] ) if ( o.given.has( k ) ) die( `--live takes a bake directory and --verify, --force, --prune, --no-lean, --drop-packed, --quiet (not --${k})` );
		if ( o.dropPacked ) o.verify = true; // (nothing is deleted that was not read back)
		return o;

	}

	if ( o.force || o.prune || o.noLean || o.dropPacked ) die( '--force, --prune, --no-lean, --drop-packed belong to --live' );
	if ( o.check ) return o;
	if ( ! o.level && ! o.lods ) die( 'give --level or --lods (see --help)' );
	if ( o.level && o.lods ) die( '--level or --lods, not both' );
	if ( o.mode === 'dcc' && ( o.container !== 'glb' || o.texStore !== 'embed' || o.level === 'ring' ) ) die( 'the dcc profile: one .glb with its images inside; levels lo, mid, near, fine or best' );
	if ( o.level && ! LEVELS.includes( o.level ) ) die( `--level: one of ${LEVELS.join( ', ' )}` + ( o.mode === 'dcc' ? ', best' : '' ) );
	if ( o.hdr && o.color !== 'float' ) die( '--hdr needs --color float' );
	if ( o.buffers !== 'one' && o.container !== 'gltf' ) die( `--buffers ${o.buffers} needs --container gltf` );
	if ( o.buffers === 'bake' && ( o.mode !== 'exact' || ! [ 'auto', 'khr', 'ext' ].includes( o.meshopt ) ) ) die( '--buffers bake: exact mode with the packer\'s own streams only' );
	return o;

}

function die( msg, code = 1 ) { console.error( 'bake2glb: ' + msg ); process.exit( code ); }

// BAKED_TIERS of the page: the object literal is cut out of the source and evaluated (it is plain data)
function readTiers( file ) {

	const src = fs.readFileSync( file, 'utf8' );
	const at = src.indexOf( 'BAKED_TIERS =' );
	if ( at < 0 ) die( `no BAKED_TIERS in ${file}` );
	const a = src.indexOf( '{', at );
	let depth = 0, b = a;
	for ( ; b < src.length; b ++ ) { if ( src[ b ] === '{' ) depth ++; else if ( src[ b ] === '}' && -- depth === 0 ) break; }
	return Function( '"use strict"; return (' + src.slice( a, b + 1 ) + ');' )();

}

// ------------------------------------------------------------------------------------------------ the bake

class Bake {

	constructor( dir ) {

		this.dir = dir;
		const f = path.join( dir, 'manifest.json' );
		if ( ! fs.existsSync( f ) ) die( `no manifest.json in ${dir}` );
		this.man = JSON.parse( fs.readFileSync( f, 'utf8' ) );
		if ( this.man.version !== 1 || this.man.codec !== 'meshopt' ) die( `manifest version ${this.man.version}, codec ${this.man.codec}: this tool reads version 1 / meshopt` );
		this._base = null;

	}

	baseFile() { return path.join( this.dir, this.man.base.file ); }
	// the packed bytes of one level of a cell: a slice of base.bin (those with `off`) or a file of its own; of a bake
	// whose packed files were dropped (--live --drop-packed) the same bytes out of its .glb
	blob( c, level ) {

		const m = c[ level ], f = m.file ? path.join( this.dir, m.file ) : this.baseFile();
		if ( ! fs.existsSync( f ) ) return this._glbBlob( c, level, m, f );
		if ( m.file ) return new Uint8Array( fs.readFileSync( f ) );
		if ( ! this._base ) this._base = new Uint8Array( fs.readFileSync( f ) );
		return this._base.subarray( m.off, m.off + m.len );

	}

	// (found as web/engine/lite_glb.js finds them: the two meshopt streams of the level's node, one after the other in
	// the binary chunk)
	_glbBlob( c, level, m, packed ) {

		const name = liveName( m.file || this.man.base.file ), f = path.join( this.dir, name );
		let G = this._glb && this._glb.name === name ? this._glb : null;
		if ( ! G ) {

			if ( ! fs.existsSync( f ) ) die( `${packed} is missing, and so is ${name}` );
			const raw = fs.readFileSync( f ), n = raw.readUInt32LE( 12 ), j = JSON.parse( raw.subarray( 20, 20 + n ).toString( 'utf8' ) );
			const ex = j.scenes && j.scenes[ 0 ] && j.scenes[ 0 ].extras;
			if ( raw.readUInt32LE( 0 ) !== 0x46546C67 || ! ex || ! ex.baked || ex.baked.baked !== this.man.baked ) die( `${packed} is missing, and ${name} is not this bake's` );
			const at = new Map();
			for ( const node of j.nodes ) {

				const b = node.extras && node.extras.baked;
				if ( ! b || node.mesh === undefined ) continue;
				const pr = j.meshes[ node.mesh ].primitives[ 0 ];
				const [ v, i ] = [ pr.attributes.POSITION, pr.indices ].map( ( a ) => { const e = j.bufferViews[ j.accessors[ a ].bufferView ].extensions || {}; return e.KHR_meshopt_compression || e.EXT_meshopt_compression; } );
				if ( v && i ) at.set( b.cell + ':' + b.level, [ v.byteOffset || 0, v.byteLength, i.byteOffset || 0, i.byteLength ] );

			}

			G = { name, raw, binAt: 20 + n + 8, at };
			if ( ! m.file ) this._glb = G; // (base.glb is asked once per cell)

		}

		const L = G.at.get( c.k + ':' + level );
		if ( ! L || L[ 1 ] !== m.vlen || L[ 2 ] !== L[ 0 ] + m.vlen || m.vlen + L[ 3 ] !== m.len ) die( `${name}: ${c.k} ${level} is not in it as the manifest has it` );
		return new Uint8Array( G.raw.buffer, G.raw.byteOffset + G.binAt + L[ 0 ], m.len );

	}

	// where those bytes live, for --buffers bake
	place( c, level ) {

		const m = c[ level ], p = m.file ? { file: path.join( this.dir, m.file ), off: 0 } : { file: this.baseFile(), off: m.off };
		if ( ! fs.existsSync( p.file ) ) die( `--buffers bake points at the bake's packed files, and ${p.file} is not there (a bake exported with --drop-packed keeps none)` );
		return p;

	}

	// decoded as the page decodes them (BakedTown._build)
	decode( m, blob ) {

		const W = m.w || 10;
		const vb = new Uint8Array( m.nv * W * 2 ), ib = new Uint8Array( m.ni * m.it );
		MeshoptDecoder.decodeVertexBuffer( vb, m.nv, W * 2, blob.subarray( 0, m.vlen ) );
		MeshoptDecoder.decodeIndexBuffer( ib, m.ni, m.it, blob.subarray( m.vlen ) );
		return { W, vb, ib };

	}

	levelsOf( c ) { return LEVELS.filter( ( l ) => c[ l ] && typeof c[ l ] === 'object' ); }
	// the box of a cell: all its levels' boxes together (as the page measures distances to it)
	box( c ) {

		const b = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ];
		for ( const l of this.levelsOf( c ) ) for ( let j = 0; j < 3; j ++ ) { b[ j ] = Math.min( b[ j ], c[ l ].bmin[ j ] ); b[ j + 3 ] = Math.max( b[ j + 3 ], c[ l ].bmax[ j ] ); }
		return b;

	}

	groundDist( c, x, z ) {

		const b = this.box( c );
		return Math.hypot( Math.max( b[ 0 ] - x, 0, x - b[ 3 ] ), Math.max( b[ 2 ] - z, 0, z - b[ 5 ] ) );

	}

	select( spec ) {

		const all = this.man.chunks;
		if ( spec === 'all' ) return all;
		let m;
		if ( ( m = /^spawn:([\d.]+)$/.exec( spec ) ) ) {

			const sp = this.man.data && this.man.data.spawn && this.man.data.spawn.pos;
			if ( ! sp ) die( 'the manifest has no spawn point (use --cells at:X,Z,R)' );
			return all.filter( ( c ) => this.groundDist( c, sp[ 0 ], - sp[ 1 ] ) <= parseFloat( m[ 1 ] ) ); // (town frame x, y -> three.js x, -z)

		}

		if ( ( m = /^at:(-?[\d.]+),(-?[\d.]+),([\d.]+)$/.exec( spec ) ) ) return all.filter( ( c ) => this.groundDist( c, parseFloat( m[ 1 ] ), parseFloat( m[ 2 ] ) ) <= parseFloat( m[ 3 ] ) );
		const keys = spec.split( ',' );
		const out = keys.map( ( k ) => all.find( ( c ) => c.k === k ) || die( `no cell ${k} in the manifest` ) );
		return out;

	}

}

// ------------------------------------------------------------------------------------------------ glTF writer

class Gltf {

	constructor( generator ) {

		this.json = { asset: { version: '2.0', generator }, scene: 0, scenes: [ { name: 'town', nodes: [] } ], nodes: [], meshes: [], materials: [], accessors: [], bufferViews: [], buffers: [] };
		this.used = new Set(); this.required = new Set();
		this.bufs = []; // { parts, length, uri, extern, fallback }
		this.named = new Map();

	}

	use( name, required = false ) { this.used.add( name ); if ( required ) this.required.add( name ); }
	// a buffer by a name of the caller's: made the first time it is asked for
	buffer( key, make ) {

		if ( ! this.named.has( key ) ) { this.bufs.push( { parts: [], length: 0, uri: null, extern: false, fallback: null, ...make() } ); this.named.set( key, this.bufs.length - 1 ); }
		return this.named.get( key );

	}

	append( bi, bytes, align = 4 ) {

		const b = this.bufs[ bi ];
		const pad = ( align - b.length % align ) % align;
		if ( pad ) { b.parts.push( new Uint8Array( pad ) ); b.length += pad; }
		const at = b.length;
		b.parts.push( bytes instanceof Uint8Array ? bytes : new Uint8Array( bytes.buffer, bytes.byteOffset, bytes.byteLength ) );
		b.length += bytes.byteLength;
		return at;

	}

	reserve( bi, n, align = 4 ) { const b = this.bufs[ bi ]; b.length += ( align - b.length % align ) % align; const at = b.length; b.length += n; return at; }
	view( def ) { this.json.bufferViews.push( def ); return this.json.bufferViews.length - 1; }
	accessor( def ) { this.json.accessors.push( def ); return this.json.accessors.length - 1; }
	node( def, inScene = false ) { this.json.nodes.push( def ); const i = this.json.nodes.length - 1; if ( inScene ) this.json.scenes[ 0 ].nodes.push( i ); return i; }

	// writes <stem>.glb, or <stem>.gltf and its .bin files; returns [ { file, bytes } ]
	write( dir, stem, container ) {

		const j = this.json, out = [];
		j.buffers = this.bufs.map( ( b ) => ( { byteLength: b.length, ...( b.uri ? { uri: b.uri } : {} ), ...( b.fallback ? { extensions: { [ b.fallback ]: { fallback: true } } } : {} ) } ) );
		if ( this.used.size ) j.extensionsUsed = [ ...this.used ].sort();
		if ( this.required.size ) j.extensionsRequired = [ ...this.required ].sort();
		for ( const k of [ 'materials', 'textures', 'images', 'samplers', 'meshes', 'accessors', 'bufferViews', 'buffers' ] ) if ( j[ k ] && ! j[ k ].length ) delete j[ k ];
		const put = ( file, parts ) => {

			const fd = fs.openSync( file, 'w' );
			let n = 0;
			for ( const p of parts ) { let o = 0; while ( o < p.byteLength ) o += fs.writeSync( fd, p, o, Math.min( p.byteLength - o, 1 << 30 ) ); n += p.byteLength; }
			fs.closeSync( fd );
			out.push( { file, bytes: n } );

		};

		for ( let i = 0; i < this.bufs.length; i ++ ) {

			const b = this.bufs[ i ];
			if ( b.extern || b.fallback || ( container === 'glb' && i === 0 ) ) continue;
			if ( ! b.uri ) die( `internal: buffer ${i} has no file` );
			put( path.join( dir, b.uri ), b.parts );

		}

		if ( container === 'glb' ) {

			const b0 = this.bufs[ 0 ];
			if ( b0 && ( b0.uri || b0.fallback ) ) die( 'internal: buffer 0 of a .glb must be its own chunk' );
			let js = Buffer.from( JSON.stringify( j ), 'utf8' );
			if ( js.length % 4 ) js = Buffer.concat( [ js, Buffer.alloc( 4 - js.length % 4, 0x20 ) ] );
			const binLen = b0 ? b0.length + ( 4 - b0.length % 4 ) % 4 : 0;
			const total = 12 + 8 + js.length + ( b0 ? 8 + binLen : 0 );
			if ( total >= 2 ** 32 ) die( `the file would have ${total} bytes: a .glb ends at 4 GB (use --container gltf, or fewer cells / levels)` );
			const head = Buffer.alloc( 20 );
			head.writeUInt32LE( 0x46546C67, 0 ); head.writeUInt32LE( 2, 4 ); head.writeUInt32LE( total, 8 );
			head.writeUInt32LE( js.length, 12 ); head.writeUInt32LE( 0x4E4F534A, 16 );
			const parts = [ head, js ];
			if ( b0 ) {

				const bh = Buffer.alloc( 8 );
				bh.writeUInt32LE( binLen, 0 ); bh.writeUInt32LE( 0x004E4942, 4 );
				parts.push( bh, ...b0.parts );
				if ( binLen > b0.length ) parts.push( new Uint8Array( binLen - b0.length ) );

			}

			put( path.join( dir, stem + '.glb' ), parts );

		} else put( path.join( dir, stem + '.gltf' ), [ Buffer.from( JSON.stringify( j ), 'utf8' ) ] );

		return out;

	}

}

// a file read back from disk, by nothing but the glTF rules (for --verify)
function readGltf( file ) {

	const raw = fs.readFileSync( file );
	let json, bin = null;
	if ( raw.readUInt32LE( 0 ) === 0x46546C67 ) {

		let o = 12;
		while ( o < raw.length ) {

			const n = raw.readUInt32LE( o ), t = raw.readUInt32LE( o + 4 );
			if ( t === 0x4E4F534A ) json = JSON.parse( raw.subarray( o + 8, o + 8 + n ).toString( 'utf8' ) );
			else if ( t === 0x004E4942 ) bin = new Uint8Array( raw.buffer, raw.byteOffset + o + 8, n );
			o += 8 + n;

		}

	} else json = JSON.parse( raw.toString( 'utf8' ) );
	const cache = new Map();
	const buffer = ( i ) => {

		if ( ! cache.has( i ) ) { const b = json.buffers[ i ]; cache.set( i, b.uri ? new Uint8Array( fs.readFileSync( path.resolve( path.dirname( file ), decodeURI( b.uri ) ) ) ) : i === 0 ? bin : null ); }
		return cache.get( i );

	};

	// the bytes a bufferView stands for (decoded when it is compressed)
	const view = ( i ) => {

		const v = json.bufferViews[ i ];
		const x = v.extensions && ( v.extensions.KHR_meshopt_compression || v.extensions.EXT_meshopt_compression );
		if ( ! x ) return buffer( v.buffer ).subarray( v.byteOffset || 0, ( v.byteOffset || 0 ) + v.byteLength );
		const src = buffer( x.buffer ).subarray( x.byteOffset || 0, ( x.byteOffset || 0 ) + x.byteLength );
		const dst = new Uint8Array( x.count * x.byteStride );
		MeshoptDecoder.decodeGltfBuffer( dst, x.count, x.byteStride, src, x.mode, x.filter || 'NONE' );
		if ( dst.length !== v.byteLength ) throw new Error( `bufferView ${i}: ${dst.length} bytes decoded, ${v.byteLength} declared` );
		return dst;

	};

	return { json, view };

}

// ------------------------------------------------------------------------------------------------ levels of detail

// The page (BakedTown.update): a cell's distance d is measured on the ground from the camera to the cell's box; level
// l is taken when d < R[l] and kept while d < R[l] + min( 0.2 R[l] + 2, hold ). R = the tier's mid / near / fine.
// THREE.LOD (and MSFT_lod viewers) measure to one point instead. The cell's node sits in the middle of the box's
// footprint, at its foot; a camera at ground distance d from the box is at most d + h from that point (h: half the
// footprint's diagonal), so the thresholds here are R + h: never a coarser level than the page would show, a finer one
// when the camera stands off a long side. extras.lod = { levels, distance[i] (level i is shown from there on),
// hysteresis[i] (THREE.LOD's: the share of distance[i] the camera must come back in before level i - 1 returns) }.
// MSFT_screencoverage[i] = the share of the screen the cell's bounding sphere (radius r) covers at the distance where
// level i gives way to the next: pi r^2 / ( 4 aspect tan^2( fov / 2 ) d^2 ), at the page's 62 degrees and 16:9,
// capped at 1; the last level's is 0 (it is never dropped).
function lodPlan( o, tiers, bake, c, levels ) { return lodPlanBox( o, tiers, bake.box( c ), levels ); }

function lodPlanBox( o, tiers, b, levels ) {

	const T = tiers[ o.tier ];
	const R = { fine: T.fine, near: T.near, ring: T.near, mid: T.mid, ...o.lodDist };
	const h = Math.hypot( b[ 3 ] - b[ 0 ], b[ 5 ] - b[ 2 ] ) / 2, r = Math.hypot( b[ 3 ] - b[ 0 ], b[ 4 ] - b[ 1 ], b[ 5 ] - b[ 2 ] ) / 2;
	const distance = [ 0 ], hysteresis = [ 0 ], coverage = [], ground = [];
	const tan = Math.tan( FOV * Math.PI / 360 );
	for ( let i = 0; i < levels.length - 1; i ++ ) {

		const Rl = R[ levels[ i ] ];
		if ( ! isFinite( Rl ) ) die( `tier ${o.tier} has no distance for level ${levels[ i ]} (use --lod-dist ${levels[ i ]}=<m>)` );
		const margin = Math.min( Rl * 0.2 + 2, T.hold || Infinity );
		const d = Rl + margin + h;
		distance.push( round( d, 3 ) ); hysteresis.push( round( margin / d, 5 ) ); ground.push( Rl );
		coverage.push( round( Math.min( 1, Math.PI * r * r / ( 4 * ASPECT * tan * tan * ( Rl + h ) * ( Rl + h ) ) ), 5 ) );

	}

	coverage.push( 0 );
	return { levels, distance, hysteresis, coverage, ground, radius: round( r, 3 ), reach: round( h, 3 ) };

}

const round = ( v, n ) => { const k = 10 ** n; return Math.round( v * k ) / k; };
// the node of a cell: the middle of its footprint, at its foot
const pivotOf = ( bake, c ) => { const b = bake.box( c ); return [ ( b[ 0 ] + b[ 3 ] ) / 2, b[ 1 ], ( b[ 2 ] + b[ 5 ] ) / 2 ]; };

// ------------------------------------------------------------------------------------------------ exact mode

function meshoptName( o, blob, m ) {

	if ( o.meshopt === 'none' ) return null;
	if ( o.meshopt === 'khr' ) return 'KHR_meshopt_compression';
	if ( o.meshopt === 'ext' || o.meshopt === 'ext-v0' ) return 'EXT_meshopt_compression';
	// (a vertex stream starts with 0xA0 | version; EXT_meshopt_compression is version 0, KHR_meshopt_compression adds 1)
	const v = blob[ 0 ] & 0x0f;
	return v === 0 ? 'EXT_meshopt_compression' : 'KHR_meshopt_compression';

}

function exactMaterials( g ) {

	if ( g.exactMats ) return g.exactMats;
	const col = [ [ 0.6, 0.6, 0.6, 1 ], [ 0.5, 0.7, 0.9, 0.35 ], [ 0.1, 0.2, 0.3, 1 ], [ 0.7, 0.8, 0.9, 0.5 ], [ 0.2, 0.4, 0.15, 1 ] ];
	g.exactMats = KIND.map( ( name, k ) => {

		g.json.materials.push( { name: 'baked:' + name, pbrMetallicRoughness: { baseColorFactor: col[ k ], metallicFactor: 0, roughnessFactor: 1 },
			...( k === K_GLASS || k === K_FALL ? { alphaMode: 'BLEND' } : {} ), ...( k === K_FALL || k === K_CUT ? { doubleSided: true } : {} ), extras: { baked_kind: k } } );
		return g.json.materials.length - 1;

	} );
	return g.exactMats;

}

// one level of a cell -> a mesh node. origin: what the node's parent already places (the cell's pivot), or null
function emitExact( g, o, bake, c, level, bufKey, origin ) {

	const m = c[ level ], blob = bake.blob( c, level ), W = m.w || 10, stride = W * 2;
	const dec = bake.decode( m, blob );
	const ext = meshoptName( o, blob, m );
	const vraw = m.nv * stride, iraw = m.ni * m.it;
	let vview, iview;
	if ( ext ) {

		g.use( ext, true );
		const fb = g.buffer( 'fallback', () => ( { fallback: ext } ) );
		let vsrc = blob.subarray( 0, m.vlen ), isrc = blob.subarray( m.vlen ), bi, voff, ioff;
		if ( o.meshopt === 'ext-v0' ) vsrc = MeshoptEncoder.encodeVertexBufferLevel( dec.vb, m.nv, stride, ( bake.man.pack && bake.man.pack.vertex_level ) || 2, 0 );
		if ( o.buffers === 'bake' ) {

			const p = bake.place( c, level );
			bi = g.buffer( 'bake:' + p.file, () => ( { uri: encodeURI( path.relative( o.out, p.file ).split( path.sep ).join( '/' ) ), extern: true, length: fs.statSync( p.file ).size } ) );
			voff = p.off; ioff = p.off + m.vlen;

		} else {

			bi = bufKey();
			voff = g.append( bi, vsrc, 1 ); ioff = g.append( bi, isrc, 1 ); // (compressed streams need no alignment)

		}

		vview = g.view( { buffer: fb, byteOffset: g.reserve( fb, vraw ), byteLength: vraw, byteStride: stride, target: 34962,
			extensions: { [ ext ]: { buffer: bi, byteOffset: voff, byteLength: vsrc.byteLength, byteStride: stride, count: m.nv, mode: 'ATTRIBUTES' } } } );
		iview = g.view( { buffer: fb, byteOffset: g.reserve( fb, iraw ), byteLength: iraw, target: 34963,
			extensions: { [ ext ]: { buffer: bi, byteOffset: ioff, byteLength: isrc.byteLength, byteStride: m.it, count: m.ni, mode: 'TRIANGLES' } } } );

	} else {

		const bi = bufKey();
		vview = g.view( { buffer: bi, byteOffset: g.append( bi, dec.vb ), byteLength: vraw, byteStride: stride, target: 34962 } );
		iview = g.view( { buffer: bi, byteOffset: g.append( bi, dec.ib ), byteLength: iraw, target: 34963 } );

	}

	g.use( 'KHR_mesh_quantization', true );
	// POSITION wants its range, in the stored integers
	const rec = new Uint16Array( dec.vb.buffer, dec.vb.byteOffset, m.nv * W );
	const mn = [ 65535, 65535, 65535 ], mx = [ 0, 0, 0 ];
	for ( let i = 0; i < rec.length; i += W ) for ( let j = 0; j < 3; j ++ ) { const v = rec[ i + j ]; if ( v < mn[ j ] ) mn[ j ] = v; if ( v > mx[ j ] ) mx[ j ] = v; }
	const acc = ( type, off, extra = {} ) => g.accessor( { bufferView: vview, byteOffset: off, componentType: 5123, normalized: true, count: m.nv, type, ...extra } );
	const attributes = { POSITION: acc( 'VEC3', 0, { min: mn, max: mx } ), _PL: acc( 'VEC4', 0 ), TEXCOORD_0: acc( 'VEC2', 8 ), _COL: acc( 'VEC4', 12 ) };
	// (records of 14 and 18 slots: a second look - webbake_bl.py encode_chunk())
	if ( W === 12 || W === 18 ) attributes._G = acc( 'VEC2', 20 );
	if ( W === 14 || W === 18 ) attributes._COL2 = acc( 'VEC4', W === 14 ? 20 : 24 );
	if ( W === 18 ) attributes._G2 = acc( 'VEC2', 32 );
	if ( ! [ 10, 12, 14, 18 ].includes( W ) ) die( `${c.k} ${level}: a vertex record of ${W} slots - this tool knows 10, 12, 14 and 18` );
	const mats = exactMaterials( g );
	const parts = new Map( ( m.parts || [] ).map( ( p ) => [ p[ 0 ], p ] ) );
	const primitives = m.sub.map( ( [ kind, start, count ], i ) => {

		const p = parts.get( i );
		return { attributes, indices: g.accessor( { bufferView: iview, byteOffset: start * m.it, componentType: m.it === 2 ? 5123 : 5125, count, type: 'SCALAR' } ), material: mats[ kind ], mode: 4,
			extras: { kind, sub: i, ...( p ? { steps: p[ 1 ], box: p[ 2 ] } : {} ) } };

	} );
	g.json.meshes.push( { name: `${c.k}:${level}`, primitives } );
	const s = m.q * 65535;
	const { off, len, file, ...meta } = m;
	return g.node( { name: `${c.k}:${level}`, mesh: g.json.meshes.length - 1, translation: origin ? m.min.map( ( v, j ) => v - origin[ j ] ) : m.min, scale: [ s, s, s ],
		extras: { baked: { cell: c.k, level, ...meta, ...( origin ? { origin } : {} ) } } } );

}

// ------------------------------------------------------------------------------------------------ portable mode

class Textures {

	constructor( g, o, bake ) { this.g = g; this.o = o; this.bake = bake; this.done = new Map(); this.bytes = new Map(); this.sampler = null; }
	// [ bytes, mime ] of one layer, with or without its alpha
	_image( layer, alpha ) {

		const o = this.o, man = this.bake.man, size = o.texSize;
		const fmt = o.textures === 'auto' ? ( alpha ? 'png' : 'jpeg' ) : o.textures;
		const key = `${layer}:${fmt}`;
		if ( this.bytes.has( key ) ) return this.bytes.get( key );
		const single = man.texl && man.texl[ size ] && man.texl[ size ].files[ layer ];
		const pages = man.tex && man.tex[ size ];
		let res;
		if ( single && fmt === 'webp' ) res = [ new Uint8Array( fs.readFileSync( path.join( this.bake.dir, single + '.webp' ) ) ), 'image/webp' ]; // as it is
		else {

			const args = [];
			if ( single ) args.push( path.join( this.bake.dir, single + '.webp' ) );
			else if ( pages ) {

				const per = pages.cols * pages.rows, p = Math.floor( layer / per ), k = layer - p * per, col = k % pages.cols, row = ( k - col ) / pages.cols;
				const crop = [ '-crop', `${size}x${size}+${col * size}+${row * size}`, '+repage' ];
				args.push( path.join( this.bake.dir, pages.pages[ p ] + '.webp' ), ...crop );
				// (a bake with separate height pages keeps the layer's alpha there, in red)
				if ( pages.hpages && fmt !== 'jpeg' ) args.push( '(', path.join( this.bake.dir, pages.hpages[ p ] + '.webp' ), ...crop, '-channel', 'R', '-separate', ')', '-alpha', 'off', '-compose', 'CopyOpacity', '-composite' );

			} else die( `the bake has no textures of size ${size} (it has: ${[ ...Object.keys( man.tex || {} ), ...Object.keys( man.texl || {} ) ].join( ', ' )})` );
			if ( fmt === 'jpeg' ) args.push( '-alpha', 'off', '-quality', String( o.texQuality ), 'jpg:-' );
			else if ( fmt === 'png' ) args.push( 'png:-' );
			else args.push( '-quality', String( o.texQuality ), '-define', 'webp:alpha-quality=100', 'webp:-' );
			res = [ new Uint8Array( magick( args ) ), fmt === 'jpeg' ? 'image/jpeg' : fmt === 'png' ? 'image/png' : 'image/webp' ];

		}

		this.bytes.set( key, res );
		return res;

	}

	// glTF texture index of a layer (null: --textures none)
	get( layer, alpha, name ) {

		const g = this.g, o = this.o, j = g.json;
		if ( o.textures === 'none' ) return null;
		const key = `${layer}:${alpha && o.textures === 'auto' ? 'a' : ''}`;
		if ( this.done.has( key ) ) return this.done.get( key );
		const [ bytes, mime ] = this._image( layer, alpha );
		const ext = mime === 'image/jpeg' ? 'jpg' : mime === 'image/png' ? 'png' : 'webp';
		j.images = j.images || []; j.textures = j.textures || []; j.samplers = j.samplers || [];
		if ( this.sampler === null ) { j.samplers.push( { magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 } ); this.sampler = j.samplers.length - 1; }
		if ( o.texStore === 'external' ) {

			const dir = path.join( o.out, 'tex' ), file = `${o.texSize}_${layer}.${ext}`;
			fs.mkdirSync( dir, { recursive: true } );
			if ( ! this.done.has( 'file:' + file ) ) { fs.writeFileSync( path.join( dir, file ), bytes ); this.done.set( 'file:' + file, 1 ); ( this.files = this.files || [] ).push( { file: path.join( dir, file ), bytes: bytes.byteLength } ); }
			j.images.push( { name, uri: 'tex/' + file } );

		} else {

			const bi = this.o.container === 'glb' ? g.buffer( 'main', () => ( {} ) ) : g.buffer( 'tex', () => ( { uri: g.stem + '.tex.bin' } ) );
			j.images.push( { name, mimeType: mime, bufferView: g.view( { buffer: bi, byteOffset: g.append( bi, bytes ), byteLength: bytes.byteLength } ) } );

		}

		if ( ext === 'webp' ) { g.use( 'EXT_texture_webp', true ); j.textures.push( { sampler: this.sampler, extensions: { EXT_texture_webp: { source: j.images.length - 1 } } } ); }
		else j.textures.push( { sampler: this.sampler, source: j.images.length - 1 } );
		this.done.set( key, j.textures.length - 1 );
		return j.textures.length - 1;

	}

}

let MAGICK = null;
function magick( args, input ) {

	const run = ( bin ) => execFileSync( bin, args, { maxBuffer: 1 << 28, ...( input ? { input } : {} ), stdio: [ input ? 'pipe' : 'ignore', 'pipe', 'pipe' ] } );
	if ( MAGICK ) return run( MAGICK );
	for ( const bin of [ 'magick', 'convert' ] ) {

		try { const r = run( bin ); MAGICK = bin; return r; } catch ( e ) { if ( e.code !== 'ENOENT' ) throw e; }

	}

	die( 'converting textures needs ImageMagick (magick or convert) on the PATH; --textures webp with a --tex-size the bake has single layers of, or --textures none, do without' );

}

function portableMaterial( g, o, bake, tex, kind, layer ) {

	const key = kind === K_OPAQUE || kind === K_CUT ? `${kind}:${layer}` : `${kind}`;
	g.pmats = g.pmats || new Map();
	if ( g.pmats.has( key ) ) return g.pmats.get( key );
	g.use( 'KHR_materials_unlit' );
	const lname = ( bake.man.layers[ layer ] || 'layer' + layer ).replace( /^stripe:.*/, 'stripe_' + layer );
	const mat = { name: KIND[ kind ], pbrMetallicRoughness: { metallicFactor: 0, roughnessFactor: 1 }, extensions: { KHR_materials_unlit: {} }, extras: { baked_kind: kind } };
	if ( kind === K_OPAQUE || kind === K_CUT ) {

		mat.name = kind === K_CUT ? 'leaves_' + lname : lname;
		mat.extras.layer = layer;
		const t = tex.get( layer, kind === K_CUT, lname );
		if ( t !== null ) mat.pbrMetallicRoughness.baseColorTexture = { index: t };
		if ( kind === K_CUT ) { mat.doubleSided = true; if ( t !== null ) { mat.alphaMode = 'MASK'; mat.alphaCutoff = CUT_ALPHA; } }

	} else if ( kind === K_GLASS ) mat.alphaMode = 'BLEND';
	else if ( kind === K_WATER ) mat.pbrMetallicRoughness.baseColorFactor = [ 0.3, 0.3, 0.3, 1 ]; // (the page: 0.3 of the baked light, plus mirror)
	else if ( kind === K_FALL ) { mat.alphaMode = 'BLEND'; mat.doubleSided = true; }
	g.json.materials.push( mat );
	g.pmats.set( key, g.json.materials.length - 1 );
	return g.json.materials.length - 1;

}

// one level of a cell -> a mesh node with one primitive per material; vertices relative to `origin`
function emitPortable( g, o, bake, tex, c, level, bufKey, origin, stats ) {

	const man = bake.man, m = c[ level ], dec = bake.decode( m, bake.blob( c, level ) ), W = dec.W;
	const rec = new Uint16Array( dec.vb.buffer, dec.vb.byteOffset, m.nv * W );
	const idx = m.it === 2 ? new Uint16Array( dec.ib.buffer, dec.ib.byteOffset, m.ni ) : new Uint32Array( dec.ib.buffer, dec.ib.byteOffset, m.ni );
	// triangles by material: kind, and for surfaces with a texture the layer of the first corner
	const groups = new Map();
	for ( const [ kind, start, count ] of m.sub ) {

		const tex = kind === K_OPAQUE || kind === K_CUT;
		for ( let t = start; t < start + count; t += 3 ) {

			const layer = tex ? rec[ idx[ t ] * W + 3 ] : - 1;
			if ( tex && ( rec[ idx[ t + 1 ] * W + 3 ] !== layer || rec[ idx[ t + 2 ] * W + 3 ] !== layer ) ) stats.mixed ++;
			const key = kind * 65536 + layer + 1;
			let grp = groups.get( key );
			if ( ! grp ) groups.set( key, grp = { kind, layer, tris: [] } );
			grp.tris.push( t );

		}

	}

	const S = man.colour_scale * o.exposure, UR = 2 * man.uv_range, CA = man.cut_aux || 4;
	const cType = o.color === 'float' ? 5126 : o.color === 'u16' ? 5123 : 5121, cMax = o.color === 'u16' ? 65535 : 255;
	const P = [], UV = [], C3 = [], C4 = [], IX = []; // the pieces of each view, in order
	const len = { P: 0, UV: 0, C3: 0, C4: 0, IX: 0 };
	const prims = [];
	const remap = new Int32Array( m.nv );
	const wide = o.meshopt !== 'none'; // (a compressed index view has one index size)
	for ( const key of [ ...groups.keys() ].sort( ( a, b ) => a - b ) ) {

		const { kind, layer, tris } = groups.get( key );
		remap.fill( - 1 );
		const order = [];
		const n3 = tris.length * 3;
		const ix = new Uint32Array( n3 );
		for ( let i = 0; i < tris.length; i ++ ) for ( let j = 0; j < 3; j ++ ) {

			const v = idx[ tris[ i ] + j ];
			if ( remap[ v ] < 0 ) { remap[ v ] = order.length; order.push( v ); }
			ix[ i * 3 + j ] = remap[ v ];

		}

		const n = order.length, alpha = kind === K_GLASS || kind === K_FALL, hasUV = kind === K_OPAQUE || kind === K_CUT;
		const pos = new Float32Array( n * 3 ), uv = hasUV ? new Float32Array( n * 2 ) : null;
		const nc = o.color === 'float' ? ( alpha ? 4 : 3 ) : 4;
		const col = o.color === 'float' ? new Float32Array( n * nc ) : o.color === 'u16' ? new Uint16Array( n * 4 ) : new Uint8Array( n * 4 );
		const mn = [ Infinity, Infinity, Infinity ], mx = [ - Infinity, - Infinity, - Infinity ];
		for ( let i = 0; i < n; i ++ ) {

			const r = order[ i ] * W;
			for ( let j = 0; j < 3; j ++ ) {

				const v = Math.fround( m.min[ j ] + rec[ r + j ] * m.q - origin[ j ] );
				pos[ i * 3 + j ] = v; if ( v < mn[ j ] ) mn[ j ] = v; if ( v > mx[ j ] ) mx[ j ] = v;

			}

			const aux = rec[ r + 9 ] / 65535;
			if ( uv ) {

				// (the page: tiles = ( uv * 65535 / 65536 - 0.5 ) * 2 uv_range; leaf cards scale them by aux * cut_aux)
				const k = kind === K_CUT ? aux * CA : 1;
				uv[ i * 2 ] = ( rec[ r + 4 ] / 65536 - 0.5 ) * UR * k; uv[ i * 2 + 1 ] = ( rec[ r + 5 ] / 65536 - 0.5 ) * UR * k;

			}

			for ( let j = 0; j < 3; j ++ ) {

				const e = rec[ r + 6 + j ] / 65535, lit = e * e * S; // (stored as a square root of light / colour_scale)
				if ( lit > 1 ) stats.over ++;
				if ( lit > stats.max ) stats.max = lit;
				col[ i * nc + j ] = o.color === 'float' ? ( o.hdr ? lit : Math.min( lit, 1 ) ) : Math.round( Math.min( lit, 1 ) * cMax );

			}

			stats.channels += 3;
			if ( nc === 4 ) col[ i * 4 + 3 ] = o.color === 'float' ? ( alpha ? aux : 1 ) : Math.round( ( alpha ? aux : 1 ) * cMax );

		}

		const small = n <= 65535 && ! wide;
		const ib = small ? new Uint16Array( ix ) : ix;
		const prim = { kind, layer, n, pos: len.P, uv: uv ? len.UV : - 1, nc, col: nc === 3 ? len.C3 : len.C4, ix: len.IX, ni: n3, it: small ? 5123 : 5125, mn, mx };
		P.push( pos ); len.P += pos.byteLength;
		if ( uv ) { UV.push( uv ); len.UV += uv.byteLength; }
		if ( nc === 3 ) { C3.push( col ); len.C3 += col.byteLength; } else { C4.push( col ); len.C4 += col.byteLength; }
		IX.push( ib ); len.IX += ib.byteLength;
		if ( len.IX % 4 ) { IX.push( new Uint8Array( 2 ) ); len.IX += 2; }
		prims.push( prim );
		stats.tris += tris.length; stats.verts += n;

	}

	// one view per attribute for the whole level (compressed as a whole when asked for)
	const bi = bufKey();
	const ext = o.meshopt === 'ext' ? 'EXT_meshopt_compression' : o.meshopt === 'khr' ? 'KHR_meshopt_compression' : null;
	const join = ( list, total ) => { const a = new Uint8Array( total ); let at = 0; for ( const p of list ) { a.set( new Uint8Array( p.buffer, p.byteOffset, p.byteLength ), at ); at += p.byteLength; } return a; };
	const mkView = ( list, total, stride, index ) => {

		if ( ! total ) return - 1;
		const bytes = join( list, total );
		if ( ! ext ) return g.view( { buffer: bi, byteOffset: g.append( bi, bytes ), byteLength: total, ...( index ? { target: 34963 } : { byteStride: stride, target: 34962 } ) } );
		g.use( ext, true );
		const fb = g.buffer( 'fallback', () => ( { fallback: ext } ) );
		const count = total / stride;
		const enc = index ? MeshoptEncoder.encodeIndexBuffer( bytes, count, 4 ) : MeshoptEncoder.encodeVertexBufferLevel( bytes, count, stride, 2, ext === 'KHR_meshopt_compression' ? 1 : 0 );
		return g.view( { buffer: fb, byteOffset: g.reserve( fb, total ), byteLength: total, ...( index ? { target: 34963 } : { byteStride: stride, target: 34962 } ),
			extensions: { [ ext ]: { buffer: bi, byteOffset: g.append( bi, enc, 1 ), byteLength: enc.byteLength, byteStride: stride, count, mode: index ? 'TRIANGLES' : 'ATTRIBUTES' } } } );

	};

	const cb = o.color === 'float' ? 4 : o.color === 'u16' ? 2 : 1;
	const vP = mkView( P, len.P, 12 ), vUV = mkView( UV, len.UV, 8 ), vC3 = mkView( C3, len.C3, 12 ), vC4 = mkView( C4, len.C4, cb * 4 ), vIX = mkView( IX, len.IX, 4, true );
	const primitives = prims.map( ( p ) => {

		const attributes = { POSITION: g.accessor( { bufferView: vP, byteOffset: p.pos, componentType: 5126, count: p.n, type: 'VEC3', min: p.mn, max: p.mx } ) };
		if ( p.uv >= 0 ) attributes.TEXCOORD_0 = g.accessor( { bufferView: vUV, byteOffset: p.uv, componentType: 5126, count: p.n, type: 'VEC2' } );
		attributes.COLOR_0 = g.accessor( { bufferView: p.nc === 3 ? vC3 : vC4, byteOffset: p.col, componentType: cType, ...( cType !== 5126 ? { normalized: true } : {} ), count: p.n, type: p.nc === 3 ? 'VEC3' : 'VEC4' } );
		return { attributes, indices: g.accessor( { bufferView: vIX, byteOffset: p.ix, componentType: p.it, count: p.ni, type: 'SCALAR' } ), material: portableMaterial( g, o, bake, tex, p.kind, p.layer ), mode: 4 };

	} );
	g.json.meshes.push( { name: `${c.k}:${level}`, primitives } );
	return g.node( { name: `${c.k}:${level}`, mesh: g.json.meshes.length - 1, extras: { baked: { cell: c.k, level, nv: m.nv, ni: m.ni, bmin: m.bmin, bmax: m.bmax } } } );

}

// ------------------------------------------------------------------------------------------------ files

function lightsFor( g, o, bake, cells ) {

	const src = bake.man.lights || [];
	if ( o.lights === 'none' || ! src.length ) return 0;
	const boxes = cells.map( ( c ) => bake.box( c ) );
	const inside = ( l ) => boxes.some( ( b ) => l[ 0 ] >= b[ 0 ] && l[ 0 ] <= b[ 3 ] && l[ 2 ] >= b[ 2 ] && l[ 2 ] <= b[ 5 ] );
	const kids = [];
	if ( o.lights === 'punctual' ) g.use( 'KHR_lights_punctual' );
	const defs = [], seen = new Map();
	src.forEach( ( l, i ) => {

		if ( ! inside( l ) ) return;
		const node = { name: 'lamp_' + i, translation: [ l[ 0 ], l[ 1 ], l[ 2 ] ], extras: { baked_light: { rgb: [ l[ 3 ], l[ 4 ], l[ 5 ] ], range: l[ 6 ], cap: l[ 7 ] } } };
		if ( o.lights === 'punctual' ) {

			// colour = the row's rgb over its largest channel, intensity = that channel (the bake's unit), range as baked
			const k = Math.max( l[ 3 ], l[ 4 ], l[ 5 ], 1e-9 );
			const def = { type: 'point', color: [ round( l[ 3 ] / k, 5 ), round( l[ 4 ] / k, 5 ), round( l[ 5 ] / k, 5 ) ], intensity: k, ...( l[ 6 ] > 0 ? { range: l[ 6 ] } : {} ) };
			const key = JSON.stringify( def );
			if ( ! seen.has( key ) ) { defs.push( def ); seen.set( key, defs.length - 1 ); }
			node.extensions = { KHR_lights_punctual: { light: seen.get( key ) } };

		}

		kids.push( g.node( node ) );

	} );
	if ( ! kids.length ) return 0;
	if ( defs.length ) g.json.extensions = { ...( g.json.extensions || {} ), KHR_lights_punctual: { lights: defs } };
	g.node( { name: 'lamps', children: kids }, true );
	return kids.length;

}

function sceneExtras( o, bake, tiers, levels ) {

	const { chunks, lights, halos, flames, data, ...rest } = bake.man;
	if ( rest.look2 && o.extras !== 'full' ) { const { lights, halos, ...l2 } = rest.look2; rest.look2 = l2; } // (its lanterns: as the first look's)
	// a file of --live says whose it is and what it was made from; everything else is in the manifest beside it
	if ( o.extras === 'live' ) return { baked: { version: rest.version, codec: rest.codec, baked: rest.baked, town: rest.town && rest.town.name }, export: { tool: 'bake2glb', mode: o.mode, levels, live: LIVE_V, src: o.liveSrc } };
	return { baked: { ...rest, ...( o.extras === 'full' ? { lights, halos, flames, data } : { spawn: data && data.spawn } ) },
		export: { tool: 'bake2glb', mode: o.mode, levels, ...( o.lods ? { tier: o.tier, tier_table: tiers[ o.tier ], fov: FOV, aspect: ASPECT } : {} ),
			...( o.mode === 'portable' ? { color: o.color, hdr: o.hdr, exposure: o.exposure, uv: 'texture tiles', light: 'COLOR_0 = linear light = stored^2 * colour_scale * exposure' + ( o.hdr ? '' : ', clamped at 1' ) } : {} ) } };

}

// one output file for `cells`
function build( o, bake, tiers, cells, stem ) {

	const g = new Gltf( 'bake2glb (threejsodango)' );
	g.stem = stem;
	const chain = o.lods ? ( o.lods === 'auto' ? autoLevels( tiers[ o.tier ] ) : o.lods.split( ',' ) ) : [ o.level ];
	for ( const l of chain ) if ( ! LEVELS.includes( l ) ) die( `no level ${l} (levels: ${LEVELS.join( ', ' )})` );
	const lowest = chain[ chain.length - 1 ];
	const tex = o.mode === 'portable' ? new Textures( g, o, bake ) : null;
	const stats = { cells: 0, levels: 0, tris: 0, verts: 0, mixed: 0, over: 0, channels: 0, max: 0, chains: 0 };
	const written = []; // { cell, level, node } for --verify
	if ( o.container === 'glb' || o.buffers === 'one' ) g.buffer( 'main', () => ( o.container === 'glb' ? {} : { uri: stem + '.bin' } ) );
	const emitted = new Map();
	const emit = ( c, level ) => {

		const key = c.k + ':' + level;
		if ( emitted.has( key ) ) return emitted.get( key );
		const bufKey = () => ( o.buffers === 'one' ? g.buffer( 'main', () => ( {} ) )
			: level === lowest ? g.buffer( 'lowest', () => ( { uri: `${stem}.${level}.bin` } ) ) : g.buffer( `${c.k}:${level}`, () => ( { uri: `${stem}.${level}.${c.k}.bin` } ) ) );
		stats.levels ++;
		const n = o.mode === 'exact' ? emitExact( g, o, bake, c, level, bufKey, o.lods ? pivotOf( bake, c ) : null ) : emitPortable( g, o, bake, tex, c, level, bufKey, pivotOf( bake, c ), stats );
		if ( o.mode === 'exact' ) { stats.tris += c[ level ].ni / 3; stats.verts += c[ level ].nv; }
		written.push( { cell: c.k, level, node: n } );
		emitted.set( key, n );
		return n;

	};

	// --order coarse: the lowest level of every cell first, then level by level towards the finest - in a single
	// buffer the first bytes are then all a loader needs to show the whole town (HTTP range requests, progressive reads)
	if ( o.order === 'coarse' ) for ( let li = chain.length - 1; li >= 0; li -- ) for ( const c of cells ) if ( c[ chain[ li ] ] && typeof c[ chain[ li ] ] === 'object' ) emit( c, chain[ li ] );

	for ( const c of cells ) {

		const have = chain.filter( ( l ) => c[ l ] && typeof c[ l ] === 'object' );
		if ( ! have.length ) continue;
		stats.cells ++;
		const origin = pivotOf( bake, c );
		const nodes = have.map( ( level ) => emit( c, level ) );
		const base = g.json.nodes[ nodes[ 0 ] ];
		const { k, ...cellRest } = c;
		const cellMeta = { cell: c.k, box: bake.box( c ).map( ( v ) => round( v, 4 ) ) };
		for ( const key in cellRest ) if ( ! LEVELS.includes( key ) ) cellMeta[ key ] = cellRest[ key ]; // (whatever else the manifest says about the cell)
		if ( o.lods && have.length > 1 ) {

			const plan = lodPlan( o, tiers, bake, c, have );
			g.use( 'MSFT_lod' );
			base.extensions = { ...( base.extensions || {} ), MSFT_lod: { ids: nodes.slice( 1 ) } };
			base.extras = { ...( base.extras || {} ), MSFT_screencoverage: plan.coverage, lod: plan };
			stats.chains ++;

		}

		// exact single levels stand where the page puts them; everything else hangs on a node at the cell's pivot
		if ( o.mode === 'exact' && ! o.lods ) { base.extras.cell = cellMeta; g.json.scenes[ 0 ].nodes.push( nodes[ 0 ] ); }
		else g.node( { name: c.k, translation: origin, children: [ nodes[ 0 ] ], extras: { cell: cellMeta } }, true );

	}

	if ( ! stats.cells ) return null;
	stats.lights = lightsFor( g, o, bake, cells );
	g.json.scenes[ 0 ].extras = sceneExtras( o, bake, tiers, chain );
	fs.mkdirSync( o.out, { recursive: true } );
	const files = g.write( o.out, stem, o.container );
	if ( tex && tex.files ) files.push( ...tex.files );
	return { files, stats, written, main: files[ files.length - ( tex && tex.files ? tex.files.length : 0 ) - 1 ].file, json: g.json };

}

const autoLevels = ( T ) => ( T.top === 'fine' ? [ 'fine', 'near', 'mid', 'lo' ] : [ T.top || 'near', 'mid', 'lo' ] );

// ------------------------------------------------------------------------------------------------ verify

function verifyExact( bake, file, cells ) {

	const gl = readGltf( file ), j = gl.json;
	const parentT = new Map();
	j.nodes.forEach( ( n ) => ( n.children || [] ).forEach( ( ch ) => parentT.set( ch, n.translation || [ 0, 0, 0 ] ) ) );
	// (the levels behind MSFT_lod stand where their base node stands)
	j.nodes.forEach( ( n, i ) => { const ids = n.extensions && n.extensions.MSFT_lod && n.extensions.MSFT_lod.ids; if ( ids ) for ( const id of ids ) parentT.set( id, parentT.get( i ) || [ 0, 0, 0 ] ); } );
	let vbytes = 0, ibytes = 0, levels = 0;
	const fail = ( msg ) => { console.error( `bake2glb: VERIFY FAILED: ${path.basename( file )}: ${msg}` ); process.exit( 2 ); };
	const views = new Map();
	const view = ( i ) => { if ( ! views.has( i ) ) views.set( i, gl.view( i ) ); return views.get( i ); };
	j.nodes.forEach( ( n, ni ) => {

		const b = n.extras && n.extras.baked;
		if ( ! b || n.mesh === undefined ) return;
		const c = cells.find( ( x ) => x.k === b.cell );
		if ( ! c || ! c[ b.level ] ) fail( `node ${n.name}: the bake has no ${b.cell} ${b.level}` );
		const m = c[ b.level ], src = bake.decode( m, bake.blob( c, b.level ) ), W = src.W;
		const prims = j.meshes[ n.mesh ].primitives;
		const A = prims[ 0 ].attributes;
		// the vertex records: every byte, and every slot of a record reachable through an accessor
		const va = j.accessors[ A.POSITION ], vv = view( va.bufferView );
		if ( j.bufferViews[ va.bufferView ].byteStride !== W * 2 || va.count !== m.nv ) fail( `${n.name}: stride / count ${j.bufferViews[ va.bufferView ].byteStride} / ${va.count}, the bake has ${W * 2} / ${m.nv}` );
		if ( vv.length !== src.vb.length ) fail( `${n.name}: ${vv.length} vertex bytes, the bake has ${src.vb.length}` );
		for ( let i = 0; i < vv.length; i ++ ) if ( vv[ i ] !== src.vb[ i ] ) fail( `${n.name}: vertex byte ${i} (vertex ${Math.floor( i / ( W * 2 ) )}, slot ${Math.floor( i % ( W * 2 ) / 2 )}) is ${vv[ i ]}, the bake has ${src.vb[ i ]}` );
		const slots = new Uint8Array( W ), comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
		for ( const name in A ) {

			const a = j.accessors[ A[ name ] ];
			if ( a.bufferView !== va.bufferView || a.componentType !== 5123 || ! a.normalized || a.count !== m.nv ) fail( `${n.name}: accessor ${name} is not a normalized u16 view of the records` );
			for ( let s = 0; s < comps[ a.type ]; s ++ ) slots[ ( a.byteOffset || 0 ) / 2 + s ] = 1;

		}

		if ( slots.some( ( s ) => ! s ) ) fail( `${n.name}: slots ${[ ...slots ].map( ( s, i ) => ( s ? - 1 : i ) ).filter( ( i ) => i >= 0 )} of the record are behind no accessor` );
		vbytes += vv.length;
		// the indices: every byte, and the draw ranges
		const iv = view( j.accessors[ prims[ 0 ].indices ].bufferView );
		if ( iv.length !== src.ib.length ) fail( `${n.name}: ${iv.length} index bytes, the bake has ${src.ib.length}` );
		for ( let i = 0; i < iv.length; i ++ ) if ( iv[ i ] !== src.ib[ i ] ) fail( `${n.name}: index byte ${i} is ${iv[ i ]}, the bake has ${src.ib[ i ]}` );
		ibytes += iv.length;
		if ( prims.length !== m.sub.length ) fail( `${n.name}: ${prims.length} primitives, the bake has ${m.sub.length} draw ranges` );
		prims.forEach( ( p, i ) => {

			const a = j.accessors[ p.indices ], [ kind, start, count ] = m.sub[ i ];
			if ( ( a.byteOffset || 0 ) !== start * m.it || a.count !== count || a.componentType !== ( m.it === 2 ? 5123 : 5125 ) || p.extras.kind !== kind ) fail( `${n.name}: primitive ${i} is not draw range ${JSON.stringify( m.sub[ i ] )}` );
			const part = ( m.parts || [] ).find( ( q ) => q[ 0 ] === i );
			if ( JSON.stringify( part ? [ part[ 1 ], part[ 2 ] ] : null ) !== JSON.stringify( p.extras.steps ? [ p.extras.steps, p.extras.box ] : null ) ) fail( `${n.name}: primitive ${i}: size steps / box differ` );

		} );
		// where it stands: translation (+ the cell node's) = min, scale = q * 65535
		const pt = parentT.get( ni ) || [ 0, 0, 0 ], s = m.q * 65535;
		for ( let k = 0; k < 3; k ++ ) if ( Math.abs( n.translation[ k ] + pt[ k ] - m.min[ k ] ) > 1e-9 || n.scale[ k ] !== s ) fail( `${n.name}: placed at ${n.translation} x ${n.scale}, the bake says ${m.min} x ${s}` );
		const { off, len, file: f, ...meta } = m;
		for ( const key in meta ) if ( JSON.stringify( b[ key ] ) !== JSON.stringify( meta[ key ] ) ) fail( `${n.name}: extras.baked.${key} differs from the manifest` );
		levels ++;

	} );
	return { levels, vbytes, ibytes };

}

function verifyPortable( bake, file, cells ) {

	const gl = readGltf( file ), j = gl.json;
	const fail = ( msg ) => { console.error( `bake2glb: VERIFY FAILED: ${path.basename( file )}: ${msg}` ); process.exit( 2 ); };
	const parentT = new Map();
	j.nodes.forEach( ( n ) => ( n.children || [] ).forEach( ( ch ) => parentT.set( ch, n.translation || [ 0, 0, 0 ] ) ) );
	j.nodes.forEach( ( n, i ) => { const ids = n.extensions && n.extensions.MSFT_lod && n.extensions.MSFT_lod.ids; if ( ids ) for ( const id of ids ) parentT.set( id, parentT.get( i ) || [ 0, 0, 0 ] ); } );
	let levels = 0, tris = 0;
	j.nodes.forEach( ( n, ni ) => {

		const b = n.extras && n.extras.baked;
		if ( ! b || n.mesh === undefined ) return;
		const c = cells.find( ( x ) => x.k === b.cell ), m = c && c[ b.level ];
		if ( ! m ) fail( `node ${n.name}: the bake has no ${b.cell} ${b.level}` );
		const pt = parentT.get( ni ) || [ 0, 0, 0 ];
		let ni3 = 0;
		const mn = [ Infinity, Infinity, Infinity ], mx = [ - Infinity, - Infinity, - Infinity ];
		for ( const p of j.meshes[ n.mesh ].primitives ) {

			ni3 += j.accessors[ p.indices ].count;
			const a = j.accessors[ p.attributes.POSITION ];
			for ( let k = 0; k < 3; k ++ ) { mn[ k ] = Math.min( mn[ k ], a.min[ k ] + pt[ k ] ); mx[ k ] = Math.max( mx[ k ], a.max[ k ] + pt[ k ] ); }

		}

		if ( ni3 !== m.ni ) fail( `${n.name}: ${ni3} indices, the bake has ${m.ni}` );
		// (the manifest's box is rounded to 1e-4 and covers what is drawn)
		for ( let k = 0; k < 3; k ++ ) if ( mn[ k ] < m.bmin[ k ] - 2e-3 || mx[ k ] > m.bmax[ k ] + 2e-3 ) fail( `${n.name}: bounds ${mn} .. ${mx} leave the bake's box ${m.bmin} .. ${m.bmax}` );
		levels ++; tris += ni3 / 3;

	} );
	return { levels, tris };

}

// ------------------------------------------------------------------------------------------------ dcc profile

// One object per spatial unit (--grain) and level, one primitive per material in it; see the header. What goes into a
// vertex is whole numbers throughout, so a file can be compared with the bake triangle by triangle (--verify):
//   position  the bake's own grid (1/1024 m: every level's min and step are multiples of it), relative to the unit's pivot
//   uv        surfaces: the stored u16 (tiles = u16 * 2 uv_range / 65536 - uv_range); leaf cards: tiles * aux * cut_aux,
//             snapped to 1/4096 of a tile (the one rounding in this profile: 1/8 of a texel of a 512 px layer)
//   COLOR_0   round( min( 1, light * exposure / gain ) * 65535 ), alpha = the glass / waterfall opacity or 1
// gain: glTF has no light above 1, and a per-channel clamp turns a strongly tinted light on a dark texture (the hedges:
// albedo 0.03 .. 0.06 under a tint of 1 .. 3) into another hue. So a layer whose light goes above 1 gets its texture
// brightened by a factor G (as far as the texture has room: its 99.5th percentile texel reaches 1) and its light divided
// by G: the product, which is what is shown, stays the bake's.
const DCC_LEVELS = [ 'lo', 'mid', 'near', 'fine' ];
const CUT_UV_GRID = 4096;
const GLOW_AUX = Math.round( 0.75 * 65535 ); // (the page: a surface with aux from 0.75 glows)
let DRACO = null;
async function dracoModules() {

	if ( DRACO ) return DRACO;
	let d;
	try { d = createRequire( import.meta.url )( 'draco3dgltf' ); } catch ( e ) { die( 'Draco needs the npm package draco3dgltf (npm install --save-dev draco3dgltf); --geometry quant does without' ); }
	DRACO = { enc: await d.createEncoderModule( {} ), dec: await d.createDecoderModule( {} ) };
	return DRACO;

}

class Grow {

	constructor( T, n = 4096 ) { this.a = new T( n ); this.n = 0; }
	need( k ) { if ( this.n + k > this.a.length ) { const b = new this.a.constructor( Math.max( this.a.length * 2, this.n + k ) ); b.set( this.a.subarray( 0, this.n ) ); this.a = b; } return this.a; }
	get() { return this.a.subarray( 0, this.n ); }

}

const gcd = ( a, b ) => { a = Math.abs( a ); b = Math.abs( b ); while ( b ) { const t = a % b; a = b; b = t; } return a; };
const dccKey = ( kind, layer, glow ) => ( kind * 65536 + layer + 1 ) * 2 + glow;
const hmix = ( h, v ) => { h = Math.imul( h ^ v, 0x9E3779B1 ); return ( h ^ ( h >>> 15 ) ) >>> 0; };

// the bake's grid: the largest of 1/1024, 1/4096, 1/65536 m that every level's min and step sit on
function dccGrid( pairs ) {

	const whole = ( v ) => Math.abs( v - Math.round( v ) ) < 1e-6;
	for ( const K of [ 1024, 4096, 65536 ] ) if ( pairs.every( ( [ c, l ] ) => whole( c[ l ].q * K ) && c[ l ].min.every( ( v ) => whole( v * K ) ) ) ) return K;
	die( 'the positions of this bake are on no common grid of 1/1024, 1/4096 or 1/65536 m: the dcc profile needs one' );

}

// what the file is made of, measured before anything is written: the light of every layer (for its gain), which layers
// carry leaves (their images keep the alpha), the mean light of the glowing cards
function dccStats( bake, o, pairs ) {

	const man = bake.man, S = man.colour_scale * o.exposure;
	const hist = new Map(), cut = new Set(), layers = new Set(), glow = new Map();
	let verts = 0, over = 0, max = 0;
	for ( const [ c, level ] of pairs ) {

		const m = c[ level ], dec = bake.decode( m, bake.blob( c, level ) ), W = dec.W;
		const rec = new Uint16Array( dec.vb.buffer, dec.vb.byteOffset, m.nv * W );
		const idx = m.it === 2 ? new Uint16Array( dec.ib.buffer, dec.ib.byteOffset, m.ni ) : new Uint32Array( dec.ib.buffer, dec.ib.byteOffset, m.ni );
		const seen = new Uint8Array( m.nv );
		for ( const [ kind, start, count ] of m.sub ) {

			const surf = kind === K_OPAQUE || kind === K_CUT;
			for ( let t = start; t < start + count; t ++ ) {

				const v = idx[ t ]; if ( seen[ v ] ) continue; seen[ v ] = 1;
				const r = v * W, e = Math.max( rec[ r + 6 ], rec[ r + 7 ], rec[ r + 8 ] );
				verts ++;
				const lit = ( e / 65535 ) ** 2 * S; if ( lit > 1 ) over ++; if ( lit > max ) max = lit;
				if ( ! surf ) continue;
				const layer = rec[ r + 3 ];
				layers.add( layer ); if ( kind === K_CUT ) cut.add( layer );
				if ( o.glow === 'emissive' && kind === K_OPAQUE && rec[ r + 9 ] >= GLOW_AUX ) {

					let gl = glow.get( layer ); if ( ! gl ) glow.set( layer, gl = [ 0, 0, 0, 0 ] );
					for ( let j = 0; j < 3; j ++ ) gl[ j ] += ( rec[ r + 6 + j ] / 65535 ) ** 2 * S;
					gl[ 3 ] ++;
					continue;

				}

				let h = hist.get( layer ); if ( ! h ) hist.set( layer, h = new Uint32Array( 65536 ) );
				h[ e ] ++;

			}

		}

	}

	// the light of each layer at its 99.5th percentile
	const top = new Map();
	for ( const [ layer, h ] of hist ) {

		let n = 0; for ( let i = 0; i < 65536; i ++ ) n += h[ i ];
		let k = 0, e = 65535; for ( let i = 0; i < 65536; i ++ ) { k += h[ i ]; if ( k >= n * 0.995 ) { e = i; break; } }
		top.set( layer, ( e / 65535 ) ** 2 * S );

	}

	return { top, cut, layers, glow, verts, over, max };

}

const srgb2lin = ( v ) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ( ( v + 0.055 ) / 1.055 ) ** 2.4; };

// the images of the dcc profile: one WebP per layer, from the largest fitting source the bake has, brightened by the
// layer's gain, without the alpha (it is the relief's height) unless leaves cut by it
class DccTextures {

	constructor( g, o, bake, st ) {

		this.g = g; this.o = o; this.bake = bake; this.st = st;
		this.done = new Map(); this.pending = []; this.sampler = null; this.gain = new Map(); this.mean = new Map(); this.cutoff = new Map(); this.bytes = 0; this.info = [];

	}

	isBlank( layer ) { return this.bake.man.layers[ layer ] === 'white'; }
	// magick arguments that read the layer (alpha and all) at any size
	source( layer ) {

		const man = this.bake.man, size = this.o.texSize;
		const singles = Object.keys( man.texl || {} ).map( Number ).filter( ( s ) => man.texl[ s ].files && man.texl[ s ].files[ layer ] ).sort( ( a, b ) => a - b );
		const atlases = Object.keys( man.tex || {} ).map( Number ).filter( ( s ) => man.tex[ s ].pages ).sort( ( a, b ) => a - b );
		const s1 = singles.find( ( s ) => s >= size ), a1 = atlases.find( ( s ) => s >= size );
		// the single layer of the size asked for, or the next larger one; an atlas tile only when it is the better fit
		const pick = s1 !== undefined && ( a1 === undefined || s1 <= a1 || a1 < size ) ? [ 's', s1 ] : a1 !== undefined ? [ 'a', a1 ] : singles.length ? [ 's', singles[ singles.length - 1 ] ] : atlases.length ? [ 'a', atlases[ atlases.length - 1 ] ] : null;
		if ( ! pick ) die( 'the bake has no textures (--textures none does without)' );
		if ( pick[ 0 ] === 's' ) return { size: pick[ 1 ], rgba: [ path.join( this.bake.dir, man.texl[ pick[ 1 ] ].files[ layer ] + '.webp' ) ] };
		const pages = man.tex[ pick[ 1 ] ], s = pick[ 1 ], per = pages.cols * pages.rows, p = Math.floor( layer / per ), k = layer - p * per, col = k % pages.cols, row = ( k - col ) / pages.cols;
		const crop = [ '-crop', `${s}x${s}+${col * s}+${row * s}`, '+repage' ];
		const rgba = [ path.join( this.bake.dir, pages.pages[ p ] + '.webp' ), ...crop ];
		if ( pages.hpages ) rgba.push( '(', path.join( this.bake.dir, pages.hpages[ p ] + '.webp' ), ...crop, '-channel', 'R', '-separate', '+channel', ')', '-alpha', 'off', '-compose', 'CopyOpacity', '-composite' );
		return { size: s, rgba };

	}

	// the layer's gain and, for leaves, the alpha cutoff that covers as much as the page's mask does
	measure( layer ) {

		if ( this.gain.has( layer ) ) return;
		const o = this.o, top = this.st.top.get( layer ) || 0;
		if ( this.isBlank( layer ) || o.textures === 'none' ) { this.gain.set( layer, 1 ); this.mean.set( layer, 1 ); this.cutoff.set( layer, o.cutAlpha === null ? CUT_ALPHA : o.cutAlpha ); return; }
		const src = this.source( layer ), N = 128;
		const px = magick( [ ...src.rgba, '-resize', `${N}x${N}!`, '-depth', '8', 'rgba:-' ] );
		const mx = new Float32Array( N * N ), al = new Uint32Array( 256 );
		for ( let i = 0; i < N * N; i ++ ) { mx[ i ] = Math.max( srgb2lin( px[ i * 4 ] ), srgb2lin( px[ i * 4 + 1 ] ), srgb2lin( px[ i * 4 + 2 ] ) ); al[ px[ i * 4 + 3 ] ] ++; }
		mx.sort();
		const room = 1 / Math.max( mx[ Math.floor( N * N * 0.995 ) ], 1e-3 );
		let G = Math.min( top, room ); G = G < 1.05 ? 1 : Math.round( G * 1000 ) / 1000;
		this.gain.set( layer, G );
		{ let sum = 0; for ( let i = 0; i < N * N; i ++ ) sum += mx[ i ]; this.mean.set( layer, Math.round( Math.min( 1, sum / ( N * N ) * G ) * 1000 ) / 1000 ); } // (the mean texel, as bright as the image is stored)
		let cutoff = o.cutAlpha;
		if ( cutoff === null && this.st.cut.has( layer ) ) {

			// the page keeps a texel when alpha > 0.665 + 0.4 smoothstep( 0.45, 1, r ), r = the distance from the card's middle
			// (1 at the middle of an edge); a stock material has one cutoff: the one that keeps the same share of a card
			const keep = ( th ) => { let n = 0; for ( let a = Math.max( 0, Math.ceil( th * 255 + 1e-9 ) ); a < 256; a ++ ) n += al[ a ]; return n / ( N * N ); };
			let share = 0; const M = 64;
			for ( let y = 0; y < M; y ++ ) for ( let x = 0; x < M; x ++ ) {

				const r = Math.hypot( ( x + 0.5 ) / M * 2 - 1, ( y + 0.5 ) / M * 2 - 1 ), t = Math.min( 1, Math.max( 0, ( r - 0.45 ) / 0.55 ) );
				share += keep( CUT_ALPHA + 0.4 * t * t * ( 3 - 2 * t ) );

			}

			share /= M * M;
			cutoff = CUT_ALPHA; let best = Infinity;
			for ( let a = 128; a < 256; a ++ ) { const d = Math.abs( keep( ( a - 0.5 ) / 255 ) - share ); if ( d < best ) { best = d; cutoff = Math.round( ( a - 0.5 ) / 255 * 1000 ) / 1000; } }
			this.share = this.share || {}; this.share[ layer ] = { page: round( share, 4 ), middle: round( keep( CUT_ALPHA ), 4 ) };

		}

		this.cutoff.set( layer, cutoff === null ? CUT_ALPHA : cutoff );

	}

	// glTF texture index of a layer (null: none)
	get( layer, name ) {

		const g = this.g, o = this.o, j = g.json;
		if ( o.textures === 'none' || this.isBlank( layer ) ) return null;
		if ( this.done.has( layer ) ) return this.done.get( layer );
		this.measure( layer );
		const src = this.source( layer ), G = this.gain.get( layer ), alpha = this.st.cut.has( layer );
		const fit = src.size !== o.texSize ? [ '-resize', `${o.texSize}x${o.texSize}!` ] : [];
		const lift = G > 1 ? [ '-colorspace', 'RGB', '-evaluate', 'Multiply', String( G ), '-colorspace', 'sRGB' ] : [];
		const args = alpha
			? [ '(', ...src.rgba, '-alpha', 'off', ...fit, ...lift, ')', '(', ...src.rgba, '-alpha', 'extract', ...fit, ')', '-alpha', 'off', '-compose', 'CopyOpacity', '-composite', '-define', 'webp:alpha-quality=100', '-define', 'webp:exact=true' ]
			: [ ...src.rgba, '-alpha', 'off', ...fit, ...lift ];
		const bytes = new Uint8Array( magick( [ ...args, '-quality', String( o.texQuality ), '-define', 'webp:method=6', 'webp:-' ] ) );
		j.images = j.images || []; j.textures = j.textures || []; j.samplers = j.samplers || [];
		if ( this.sampler === null ) { j.samplers.push( { magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 } ); this.sampler = 0; }
		const view = g.view( { buffer: 0, byteOffset: 0, byteLength: bytes.byteLength } );
		this.pending.push( { view, bytes } );
		j.images.push( { name, mimeType: 'image/webp', bufferView: view } );
		g.use( 'EXT_texture_webp', true );
		j.textures.push( { name, sampler: this.sampler, extensions: { EXT_texture_webp: { source: j.images.length - 1 } } } );
		this.bytes += bytes.byteLength; this.info.push( { layer, name, bytes: bytes.byteLength, gain: G, alpha, from: src.size } );
		this.done.set( layer, j.textures.length - 1 );
		return j.textures.length - 1;

	}

	// the images made since the last call go into the buffer here
	flush() { for ( const p of this.pending ) this.g.json.bufferViews[ p.view ].byteOffset = this.g.append( 0, p.bytes ); this.pending.length = 0; }

}

function dccMaterial( g, o, bake, tex, st, kind, layer, glow ) {

	const key = dccKey( kind, kind === K_OPAQUE || kind === K_CUT ? layer : - 1, glow );
	g.dmats = g.dmats || new Map();
	if ( g.dmats.has( key ) ) return g.dmats.get( key );
	const lname = layer >= 0 ? ( bake.man.layers[ layer ] || 'layer' + layer ).replace( /^stripe:.*/, 'stripe_' + layer ) : '';
	const quantUV = o.geometry === 'quant' && kind === K_OPAQUE;
	const UR = 2 * bake.man.uv_range;
	const texInfo = ( t ) => ( { index: t, ...( quantUV ? { extensions: { KHR_texture_transform: { offset: [ - UR / 2, - UR / 2 ], scale: [ 65535 * UR / 65536, 65535 * UR / 65536 ] } } } : {} ) } );
	let mat;
	if ( glow ) {

		// a glowing card as an emitter: the layer's texture times one colour and strength (the mean of its baked light)
		const gl = st.glow.get( layer ) || [ 1, 1, 1, 1 ], mean = [ gl[ 0 ] / gl[ 3 ], gl[ 1 ] / gl[ 3 ], gl[ 2 ] / gl[ 3 ] ], top = Math.max( ...mean, 1e-6 );
		const t = tex.get( layer, lname ), G = t === null ? 1 : tex.gain.get( layer );
		g.use( 'KHR_materials_emissive_strength' );
		mat = { name: lname + '_glow', pbrMetallicRoughness: { baseColorFactor: [ 0, 0, 0, 1 ], metallicFactor: 0, roughnessFactor: 1 }, emissiveFactor: mean.map( ( v ) => round( v / top, 4 ) ),
			extensions: { KHR_materials_emissive_strength: { emissiveStrength: round( top / G, 4 ) } }, extras: { kind: 'glow', layer } };
		if ( t !== null ) mat.emissiveTexture = texInfo( t );

	} else {

		g.use( 'KHR_materials_unlit' );
		mat = { name: KIND[ kind ], pbrMetallicRoughness: { metallicFactor: 0, roughnessFactor: 1 }, extensions: { KHR_materials_unlit: {} }, extras: { kind: KIND[ kind ] } };
		if ( kind === K_OPAQUE || kind === K_CUT ) {

			mat.name = kind === K_CUT ? 'leaves_' + lname : lname;
			mat.extras.layer = layer;
			const t = tex.get( layer, lname );
			if ( t !== null ) mat.pbrMetallicRoughness.baseColorTexture = texInfo( t );
			if ( kind === K_CUT ) { mat.doubleSided = true; if ( t !== null ) { mat.alphaMode = 'MASK'; mat.alphaCutoff = tex.cutoff.get( layer ); } }

		} else if ( kind === K_GLASS ) mat.alphaMode = 'BLEND';
		else if ( kind === K_WATER ) mat.pbrMetallicRoughness.baseColorFactor = [ 0.3, 0.3, 0.3, 1 ]; // (the page: 0.3 of the baked light, plus its mirror)
		else if ( kind === K_FALL ) { mat.name = 'waterfall'; mat.alphaMode = 'BLEND'; mat.doubleSided = true; }

	}

	if ( quantUV && ( mat.pbrMetallicRoughness.baseColorTexture || mat.emissiveTexture ) ) g.use( 'KHR_texture_transform', true );
	g.json.materials.push( mat );
	g.dmats.set( key, g.json.materials.length - 1 );
	return g.json.materials.length - 1;

}

// Lantern panes as the page's shader lights them (lite.js, 'baked glass (lantern glare)'), for a file whose materials
// have no shader: glass has no texture, so the bake keeps the vector to the burner and the burner's hue in the uv slots
// of a pane's vertices, and for clear glass the peak of the burner's light in the layer slot.
function dccPane( man, lamps ) {

	if ( lamps !== 'lit' || ! man.bake || ! man.bake.pane_r || ! man.bake.pane_uv ) return null;
	// (tess: the longest edge a pane's triangle keeps on a level, m - the levels seen from near by; none: no splitting)
	// (bodies: lamps whose burner sits inside an opaque glowing body, an opal globe - the halo rows with a lift, which is
	// how far towards the eye the page draws their halo so that the globe does not hide it: [ x, y, z, reach^2, r, g, b ])
	const bodies = ( man.halos || [] ).filter( ( h ) => h.length > 8 && h[ 8 ] > 0 ).map( ( h ) => [ h[ 0 ], h[ 1 ], h[ 2 ], ( h[ 8 ] * 1.1 ) ** 2, h[ 3 ], h[ 4 ], h[ 5 ] ] );
	return { r: man.bake.pane_r / 127, tess: { fine: 0.06, ring: 0.12, near: 0.12 }, max: { fine: 6, ring: 4, near: 4 }, bodies, body: 3 };

}

const sstep = ( a, b, x ) => { const t = Math.min( 1, Math.max( 0, ( x - a ) / ( b - a ) ) ); return t * t * ( 3 - 2 * t ); };
// one vertex's burner: [ has, Lx, Ly, Lz (vertex -> burner, m), hue g, hue b, peak ]
function paneBurner( rec, r, PR, out ) {

	const ax = rec[ r + 4 ], ay = rec[ r + 5 ], hx = Math.floor( ( ax + 0.5 ) / 256 ), hy = Math.floor( ( ay + 0.5 ) / 256 ), hg = Math.floor( ( hy + 0.5 ) / 16 );
	out[ 0 ] = hg >= 1 ? 1 : 0;
	out[ 1 ] = ( ax - hx * 256 - 128 ) * PR; out[ 2 ] = ( hx - 128 ) * PR; out[ 3 ] = ( ay - hy * 256 - 128 ) * PR;
	out[ 4 ] = hg / 15; out[ 5 ] = ( hy - hg * 16 ) / 15; out[ 6 ] = rec[ r + 3 ] / 64;
	return out;

}

// The pane at one point, seen square-on (the eye on its normal n) with the page's default look: f = [ r, g, b (times
// the cover), cover ]. L: point -> burner; l: the baked light there; ax: the baked cover.
const PANE = { SC: 0.04 * 0.04, SB: 0.085 * 0.085, CORE: 1.2, BODY: 0.5, EVEN: 0.3, CLEAR_LIT: 0.23, CLEAR_COVER: 0.5 };
function paneEval( Lx, Ly, Lz, n, l0, l1, l2, ax, hueG, hueB, pk, f ) {

	const d2 = Lx * Lx + Ly * Ly + Lz * Lz, h = Lx * n[ 0 ] + Ly * n[ 1 ] + Lz * n[ 2 ], r2 = Math.max( 0, d2 - h * h );
	if ( pk >= 0.25 ) {

		// Clear glass around a burner is baked black: the page lights it per pixel (the burner on the dirt of the far
		// panes, its mirror image, the mantle seen through, a halo over the head). Here it is frosted glass, lit as the
		// bake lights the frosted panes of the other lanterns: its burner's hue at a share of the burner's peak
		const e = pk * PANE.CLEAR_LIT;
		l0 = Math.max( l0, e ); l1 = Math.max( l1, e * hueG ); l2 = Math.max( l2, e * hueB ); ax = Math.max( ax, PANE.CLEAR_COVER );

	}

	// (the sheet: the baked even glow turned down to the amber of the pane's edges, the mantle's glare over it)
	const k = 1 - sstep( 0.2, 1.0, d2 );
	const core = Math.exp( - r2 / ( 2 * PANE.SC ) ) * PANE.CORE, dirt = Math.min( 1, Math.max( 0, ( ax - 0.22 ) / ( 0.68 * 0.7 ) ) ), cv = Math.max( ax, 0.22 );
	const body = Math.pow( r2 / PANE.SB + 1, - 1.5 ) * PANE.BODY * ( dirt * 0.6 + 0.8 );
	const ev = PANE.EVEN * ( dirt * 0.6 + 0.8 + 0.5 ) / ( dirt * 0.6 + 1.05 ) * ( ax / cv );
	const cover = ax + ( cv - ax ) * k, gk = k / cv * cover;
	f[ 0 ] = l0 * ( 1 + ( ev - 1 ) * k ) * cover + ( core + body ) * gk;
	f[ 1 ] = l1 * ( 1 + ( 0.82 * ev - 1 ) * k ) * cover + ( ( hueG + ( 1 - hueG ) * 0.45 ) * core + hueG * 0.8 * body ) * gk;
	f[ 2 ] = l2 * ( 1 + ( 0.65 * ev - 1 ) * k ) * cover + ( ( hueB + ( 1 - hueB ) * 0.45 ) * core + hueB * 0.52 * body ) * gk;
	f[ 3 ] = cover;

}

// The lantern panes among the glass triangles `tris` of one level. per: Float32Array, 5 per vertex [ r, g, b (times the
// cover), cover, weight ] (weight 0: not a pane's vertex); n: per triangle, how many pieces along an edge it is cut
// into (0: not a pane); tri( i ): that triangle's corners { B (burners), p (positions, m), lit, aux, nrm }.
// On a level that cuts large panes (`edge`, `most`) a vertex takes the pane at its own place; on the others each
// triangle's light is shared out to its corners (so a pane of four vertices keeps its sum, without its hot centre).
// a light seen directly (lantern pane, mantle, filament, bulb, neon tube) as the town page shows it: its highlight knee
// for glowing surfaces (web/town/main.js: kneeGlow 2.8, keeps the hue), then three.js' ACES filmic curve at the page's
// exposure (0.72), which is what pales a strong light towards white there - scaled so the brightest channel is 1
const PAGE = { knee: 2.8, exposure: 0.72 };
const rrt = ( v ) => ( v * ( v + 0.0245786 ) - 0.000090537 ) / ( v * ( 0.983729 * v + 0.432951 ) + 0.238081 );
function pageTone( l0, l1, l2, f ) {

	const Lm = Math.max( l0, l1, l2, 1e-4 ), m = PAGE.knee, t = m * 0.35, x = Math.max( 0, Lm - t ) / ( m - t );
	const k = ( Math.min( Lm, t ) + ( m - t ) * x / ( x + 1 ) ) / Lm * PAGE.exposure / 0.6;
	l0 *= k; l1 *= k; l2 *= k;
	const a = rrt( 0.59719 * l0 + 0.35458 * l1 + 0.04823 * l2 ), b = rrt( 0.07600 * l0 + 0.90834 * l1 + 0.01566 * l2 ), c = rrt( 0.02840 * l0 + 0.13383 * l1 + 0.83777 * l2 );
	const r = Math.max( 0, 1.60475 * a - 0.53108 * b - 0.07367 * c ), g = Math.max( 0, - 0.10208 * a + 1.10813 * b - 0.00605 * c ), bl = Math.max( 0, - 0.00327 * a - 0.07276 * b + 1.07602 * c );
	const top = Math.max( r, g, bl, 1e-6 );
	f[ 0 ] = r / top; f[ 1 ] = g / top; f[ 2 ] = bl / top;

}

function paneLight( PN, m, rec, W, idx, tris, S, edge, most ) {

	const per = new Float32Array( m.nv * 5 ), cuts = new Uint8Array( tris.length );
	const T = { B: [ new Float64Array( 7 ), new Float64Array( 7 ), new Float64Array( 7 ) ], p: [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ], lit: [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ], aux: [ 0, 0, 0 ], nrm: [ 0, 0, 0 ], area: 0, edge: 0, v: [ 0, 0, 0 ] };
	const tri = ( i ) => {

		for ( let c = 0; c < 3; c ++ ) {

			const v = T.v[ c ] = idx[ tris[ i ] + c ], r = v * W;
			if ( ! paneBurner( rec, r, PN.r, T.B[ c ] )[ 0 ] ) return null;
			for ( let j = 0; j < 3; j ++ ) { T.p[ c ][ j ] = rec[ r + j ] * m.q; const e = rec[ r + 6 + j ] / 65535; T.lit[ c ][ j ] = e * e * S; }
			T.aux[ c ] = rec[ r + 9 ] / 65535;

		}

		const p = T.p, e1 = [ p[ 1 ][ 0 ] - p[ 0 ][ 0 ], p[ 1 ][ 1 ] - p[ 0 ][ 1 ], p[ 1 ][ 2 ] - p[ 0 ][ 2 ] ], e2 = [ p[ 2 ][ 0 ] - p[ 0 ][ 0 ], p[ 2 ][ 1 ] - p[ 0 ][ 1 ], p[ 2 ][ 2 ] - p[ 0 ][ 2 ] ];
		const n = T.nrm; n[ 0 ] = e1[ 1 ] * e2[ 2 ] - e1[ 2 ] * e2[ 1 ]; n[ 1 ] = e1[ 2 ] * e2[ 0 ] - e1[ 0 ] * e2[ 2 ]; n[ 2 ] = e1[ 0 ] * e2[ 1 ] - e1[ 1 ] * e2[ 0 ];
		const nl = Math.hypot( n[ 0 ], n[ 1 ], n[ 2 ] );
		if ( ! ( nl > 0 ) ) return null;
		T.area = nl / 2; n[ 0 ] /= nl; n[ 1 ] /= nl; n[ 2 ] /= nl;
		T.edge = Math.max( Math.hypot( e1[ 0 ], e1[ 1 ], e1[ 2 ] ), Math.hypot( e2[ 0 ], e2[ 1 ], e2[ 2 ] ), Math.hypot( e1[ 0 ] - e2[ 0 ], e1[ 1 ] - e2[ 1 ], e1[ 2 ] - e2[ 2 ] ) );
		return T;

	};

	// the pane at the point ( u, s, t ) of the triangle in T
	const at = ( u, s, t, f ) => { const B = T.B, l = T.lit; paneEval( u * B[ 0 ][ 1 ] + s * B[ 1 ][ 1 ] + t * B[ 2 ][ 1 ], u * B[ 0 ][ 2 ] + s * B[ 1 ][ 2 ] + t * B[ 2 ][ 2 ], u * B[ 0 ][ 3 ] + s * B[ 1 ][ 3 ] + t * B[ 2 ][ 3 ], T.nrm,
		u * l[ 0 ][ 0 ] + s * l[ 1 ][ 0 ] + t * l[ 2 ][ 0 ], u * l[ 0 ][ 1 ] + s * l[ 1 ][ 1 ] + t * l[ 2 ][ 1 ], u * l[ 0 ][ 2 ] + s * l[ 1 ][ 2 ] + t * l[ 2 ][ 2 ], u * T.aux[ 0 ] + s * T.aux[ 1 ] + t * T.aux[ 2 ], B[ 0 ][ 4 ], B[ 0 ][ 5 ], B[ 0 ][ 6 ], f ); };
	const f = [ 0, 0, 0, 0 ], W3 = [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ];
	let any = false;
	for ( let i = 0; i < tris.length; i ++ ) {

		if ( ! tri( i ) ) continue;
		any = true;
		if ( edge ) {

			cuts[ i ] = Math.min( most, Math.max( 1, Math.ceil( T.edge / edge ) ) );
			for ( let c = 0; c < 3; c ++ ) { const q = T.v[ c ] * 5; if ( per[ q + 4 ] ) continue; at( W3[ c ][ 0 ], W3[ c ][ 1 ], W3[ c ][ 2 ], f ); per[ q ] = f[ 0 ]; per[ q + 1 ] = f[ 1 ]; per[ q + 2 ] = f[ 2 ]; per[ q + 3 ] = f[ 3 ]; per[ q + 4 ] = 1; }
			continue;

		}

		cuts[ i ] = 1;
		const N = Math.min( 8, Math.max( 1, Math.ceil( T.edge / 0.02 ) ) ), sc = T.area / ( N * N ); // (samples about 2 cm apart: the core is 4 cm wide)
		for ( let a = 0; a < N; a ++ ) for ( let b = 0; a + b < N; b ++ ) for ( let up = 0; up < 2; up ++ ) {

			if ( up && a + b > N - 2 ) continue;
			const s = ( a + ( up ? 2 : 1 ) / 3 ) / N, t = ( b + ( up ? 2 : 1 ) / 3 ) / N, w = [ 1 - s - t, s, t ];
			at( w[ 0 ], s, t, f );
			for ( let c = 0; c < 3; c ++ ) { const q = T.v[ c ] * 5, wc = w[ c ] * sc; per[ q ] += f[ 0 ] * wc; per[ q + 1 ] += f[ 1 ] * wc; per[ q + 2 ] += f[ 2 ] * wc; per[ q + 3 ] += f[ 3 ] * wc; per[ q + 4 ] += wc; }

		}

	}

	return any ? { per, cuts, tri, at } : null;

}

// Falling water. On the page a fall is a torn sheet drawn by a shader (web/engine/lite.js, 'baked fall'): scrolling
// noise cuts it into flutes, ropes and threads, and what is left is a thin veil - cover = sheet * veil * aux, colour =
// baked light * ( foam * 0.8 + 0.03 ) plus a little bead glint. Its two masks (break-up, foam; 4 bits each) ride in the
// layer slot of a fall's vertices. A stock material has neither the noise nor the time, so the file carries the mean
// of that shader over the noise: FALL[ slot & 255 ] = [ cover, light kept * cover, bead * cover ]. Drawn blended, a
// vertex with alpha = aux * cover and colour = ( light * kept + bead colour * bead ) / cover puts the page's average
// picture on the screen (the page without the directional extras: the Low preset's fall).
// the engine's tiling noise (web/engine/materials.js makeNoiseTexture), number for number, with its mip levels
let NOISE = null;
function engineNoise() {

	if ( NOISE ) return NOISE;
	const size = 256, data = new Float32Array( size * size * 4 );
	let seed = 1234567;
	const rnd = () => ( ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296 );
	for ( let c = 0; c < 4; c ++ ) {

		const acc = new Float32Array( size * size );
		let amp = 1;
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

			amp *= 0.5;

		}

		let lo = 1e9, hi = - 1e9;
		for ( let i = 0; i < acc.length; i ++ ) { lo = Math.min( lo, acc[ i ] ); hi = Math.max( hi, acc[ i ] ); }
		for ( let i = 0; i < acc.length; i ++ ) data[ i * 4 + c ] = Math.round( 255 * ( acc[ i ] - lo ) / ( hi - lo ) ) / 255;

	}

	const mips = [ data ];
	for ( let s = size / 2; s >= 1; s /= 2 ) {

		const src = mips[ mips.length - 1 ], S = s * 2, m = new Float32Array( s * s * 4 );
		for ( let y = 0; y < s; y ++ ) for ( let x = 0; x < s; x ++ ) for ( let c = 0; c < 4; c ++ )
			m[ ( y * s + x ) * 4 + c ] = ( src[ ( 2 * y * S + 2 * x ) * 4 + c ] + src[ ( 2 * y * S + 2 * x + 1 ) * 4 + c ] + src[ ( ( 2 * y + 1 ) * S + 2 * x ) * 4 + c ] + src[ ( ( 2 * y + 1 ) * S + 2 * x + 1 ) * 4 + c ] ) / 4;
		mips.push( m );

	}

	return ( NOISE = { size, data, mips } );

}

const FALL_BEAD = [ 0.62 * 0.3, 0.74 * 0.3, 0.79 * 0.3 ];
let FALL = null;
function fallTable() {

	if ( FALL ) return FALL;
	const data = engineNoise().data;

	// the three taps of the shader land anywhere on the noise as the sheet scrolls: three texels drawn at random
	// (a generator of its own, so the table is the same in every run)
	let s2 = 987654321;
	const pick = () => ( ( s2 = ( s2 * 1664525 + 1013904223 ) >>> 0 ) >>> 16 ) * 4;
	const N = 8192, taps = new Float32Array( N * 6 );
	for ( let i = 0; i < N; i ++ ) {

		const a = pick(), b = pick(), d = pick();
		taps[ i * 6 ] = data[ a ]; taps[ i * 6 + 1 ] = data[ a + 2 ]; taps[ i * 6 + 2 ] = data[ a + 3 ]; // s1 .r .b .a
		taps[ i * 6 + 3 ] = data[ b + 1 ]; taps[ i * 6 + 4 ] = data[ b + 2 ]; // s2 .g .b
		taps[ i * 6 + 5 ] = data[ d + 2 ]; // s3 .b

	}

	const sat = ( x ) => Math.min( 1, Math.max( 0, x ) );
	FALL = new Float64Array( 256 * 3 );
	for ( let slot = 0; slot < 256; slot ++ ) {

		const brk = sat( ( slot & 15 ) / 15 ), white = sat( ( slot >> 4 ) / 15 );
		const thr = 0.16 + ( 0.61 - 0.16 ) * brk * ( brk * 0.35 + 0.65 ) - white * 0.14, air = brk * ( brk * 0.5 + 0.5 );
		let cover = 0, kept = 0, beads = 0;
		for ( let i = 0; i < N; i ++ ) {

			const r1 = taps[ i * 6 ], b1 = taps[ i * 6 + 1 ], a1 = taps[ i * 6 + 2 ], g2 = taps[ i * 6 + 3 ], b2 = taps[ i * 6 + 4 ], b3 = taps[ i * 6 + 5 ];
			const body = r1 * ( brk * - 0.2 + 0.47 ) + g2 * 0.25 + b3 * ( brk * 0.2 + 0.28 );
			const a = sstep( thr, thr + 0.06, body );
			if ( a === 0 ) continue;
			const rim = a * ( 1 - sstep( thr + 0.05, thr + 0.16, body ) );
			const fib = sstep( 0.42, 0.7, g2 * 0.55 + b1 * 0.45 );
			const foam = sat( white + fib * 0.26 + 0.06 + air * ( fib * 0.4 + 0.36 ) + rim * brk * 0.5 );
			const bead = sstep( 0.8, 0.92, b2 * 0.6 + a1 * 0.4 ) * a * ( brk * 0.8 + 0.2 );
			const veil = 0.22 + ( 0.95 - 0.22 ) * Math.max( foam, bead ), op = a * veil;
			cover += op; kept += ( foam * 0.8 + 0.03 ) * op; beads += bead * op;

		}

		FALL[ slot * 3 ] = cover / N; FALL[ slot * 3 + 1 ] = kept / N; FALL[ slot * 3 + 2 ] = beads / N;

	}

	return FALL;

}

// The ground's two pictures (bake.gao: tools/aomap.py; bake.gsh: ground_shadows() of tools/webbake_bl.py), which the
// page reads per pixel (lite.js, 'The ground's contact occlusion'): the dark rim where things stand on the paving,
// and the lanterns' shadows - the ground's vertices hold the lanterns' light WITHOUT them. Here the same factor goes
// into the vertices' colours. q: { ao, shadows, edge, tol, cut (levels whose triangles are cut), blur (level: m),
// lattice (grid units the cut's new vertices sit on), k, pow, sharp, bias, all, span (GROUND) }; K: grid units a metre.
// -> height( x, z ): the terrain's; at( x, z, dh, up, out ): the factor [ r, g, b ] at a place dh above the terrain
// that faces up by `up` (0 .. 1), as the page's pixel has it; around(): its mean within r metres;
// cuts( a, b ): whether the edge between two places (grid units) is to be halved
const GROUND = { k: 0.6, pow: 3, sharp: 5, bias: 0.95, all: 0.5, span: 3 }; // (the page's: GAO.k, GAO.pow, and of the shadows pow, bias, all, span)
function dccGround( bake, q, K ) {

	const man = bake.man, g = man.gao, s = q.shadows ? man.gsh : null;
	if ( ! g || ! g.file || ( q.shadows && ( ! s || ! s.file ) ) ) die( 'this bake does not have the ground pictures the file was made with (gao, gsh)' );
	const w = g.w, h = g.h, n = w * h;
	const raw = ( f ) => { const b = magick( [ path.join( bake.dir, f.file ), '-depth', '8', 'rgb:-' ] ); if ( b.length !== n * 3 ) die( `${f.file}: not ${w} x ${h}` ); return b; };
	// (red: the open share, green: the terrain's height; of the shadows red: sqrt v, green: sqrt f)
	const R = new Uint8Array( n ), H = new Uint8Array( n ), V = new Uint8Array( n ).fill( 255 ), F = new Uint8Array( n ).fill( 255 );
	{ const b = raw( g ); for ( let i = 0; i < n; i ++ ) { R[ i ] = b[ i * 3 ]; H[ i ] = b[ i * 3 + 1 ]; } }
	if ( s ) { if ( s.w !== w || s.h !== h ) die( `${s.file}: not on the grid of ${g.file}` ); const b = raw( s ); for ( let i = 0; i < n; i ++ ) { V[ i ] = b[ i * 3 ]; F[ i ] = b[ i * 3 + 1 ]; } }
	const bil = ( a, w, h, u, v ) => {

		u = u < 0 ? 0 : u > w - 1 ? w - 1 : u; v = v < 0 ? 0 : v > h - 1 ? h - 1 : v;
		const i = Math.min( Math.max( w - 2, 0 ), Math.floor( u ) ), j = Math.min( Math.max( h - 2, 0 ), Math.floor( v ) ), fu = u - i, fv = v - j, o = j * w + i, dx = w > 1 ? 1 : 0, dy = h > 1 ? w : 0;
		return ( a[ o ] * ( 1 - fu ) + a[ o + dx ] * fu ) * ( 1 - fv ) + ( a[ o + dy ] * ( 1 - fu ) + a[ o + dy + dx ] * fu ) * fv;

	};

	// v around a place: the level of the picture's mip chain with texels of about `span` m, as the page's sampler reads
	// it (each level the one before it at half the size, linear between two levels)
	let vb = null;
	if ( s ) {

		const lv = Math.max( 0, Math.log2( q.span / g.texel ) ), l0 = Math.floor( lv ), lev = [];
		let a = Float32Array.from( V ), aw = w, ah = h;
		for ( let l = 0; ; l ++ ) {

			if ( l >= l0 ) lev.push( { a, w: aw, h: ah } );
			if ( l > l0 ) break;
			const bw = Math.max( 1, aw >> 1 ), bh = Math.max( 1, ah >> 1 ), b = new Float32Array( bw * bh );
			for ( let y = 0; y < bh; y ++ ) for ( let x = 0; x < bw; x ++ ) b[ y * bw + x ] = bil( a, aw, ah, ( x + 0.5 ) * aw / bw - 0.5, ( y + 0.5 ) * ah / bh - 0.5 );
			a = b; aw = bw; ah = bh;

		}

		vb = { lev, fr: lv - l0 };

	}

	const sm = [ 0, 0, 0, 0 ]; let so = 0, w00 = 0, w10 = 0, w01 = 0, w11 = 0;
	const place = ( x, z ) => {

		let u = ( x - g.x0 ) / g.sx * w - 0.5, v = ( z - g.z0 ) / g.sz * h - 0.5;
		u = u < 0 ? 0 : u > w - 1 ? w - 1 : u; v = v < 0 ? 0 : v > h - 1 ? h - 1 : v;
		const i = Math.min( w - 2, Math.floor( u ) ), j = Math.min( h - 2, Math.floor( v ) ), fu = u - i, fv = v - j;
		so = j * w + i; w00 = ( 1 - fu ) * ( 1 - fv ) / 255; w10 = fu * ( 1 - fv ) / 255; w01 = ( 1 - fu ) * fv / 255; w11 = fu * fv / 255;

	};
	const of = ( a ) => a[ so ] * w00 + a[ so + 1 ] * w10 + a[ so + w ] * w01 + a[ so + w + 1 ] * w11;
	const height = ( x, z ) => { place( x, z ); return of( H ) * g.hs + g.h0; };
	const b = s ? s.band : null, f0 = g.floor;
	const at = ( x, z, dh, up, out ) => {

		place( x, z );
		let fa = 1;
		if ( q.ao ) fa = 1 + ( of( R ) ** q.pow - 1 ) * sstep( - 0.3, - 0.15, dh ) * ( 1 - sstep( f0 + 0.03, f0 + 0.12, dh ) ) * up * q.k;
		out[ 0 ] = out[ 1 ] = out[ 2 ] = fa;
		if ( ! s ) return out;
		const wt = sstep( b[ 0 ], b[ 1 ], dh ) * ( 1 - sstep( b[ 2 ], b[ 3 ], dh ) ) * up;
		if ( wt <= 0 ) return out;
		const sv = of( V ), sf = of( F ), v = sv * sv, f = sf * sf;
		const un = ( x - g.x0 ) / g.sx, vn = ( z - g.z0 ) / g.sz, L0 = vb.lev[ 0 ], L1 = vb.lev[ 1 ];
		const s0 = bil( L0.a, L0.w, L0.h, un * L0.w - 0.5, vn * L0.h - 0.5 ), s1 = bil( L1.a, L1.w, L1.h, un * L1.w - 0.5, vn * L1.h - 0.5 ), sb = ( s0 + ( s1 - s0 ) * vb.fr ) / 255;
		// (a shadow is deepened where it is local: against the share around the place)
		const c = 1 + ( Math.min( 1, Math.max( 0, v / Math.max( sb * sb * q.bias, 0.02 ) ) ) ** q.sharp - 1 ) * wt;
		const vs = v * c, half = 1 + ( c - 1 ) * q.all;
		for ( let k = 0; k < 3; k ++ ) { const fill = s.cF[ k ] * f, lan = s.cD[ k ] * ( 1 - f ); out[ k ] *= ( fill + lan * ( v + ( vs - v ) * wt ) ) / Math.max( fill + lan * ( v + ( 1 - v ) * wt ), 1e-5 ) * half; }
		return out;

	};

	// (13 places of a square grid inside the circle)
	const DISK = []; for ( let i = - 2; i <= 2; i ++ ) for ( let j = - 2; j <= 2; j ++ ) if ( i * i + j * j <= 4 ) DISK.push( [ i / 2, j / 2 ] );
	const tmp = [ 0, 0, 0 ];
	const around = ( x, z, dh, up, r, out ) => {

		let a0 = 0, a1 = 0, a2 = 0;
		for ( const [ dx, dz ] of DISK ) { at( x + dx * r, z + dz * r, dh, up, tmp ); a0 += tmp[ 0 ]; a1 += tmp[ 1 ]; a2 += tmp[ 2 ]; }
		out[ 0 ] = a0 / DISK.length; out[ 1 ] = a1 / DISK.length; out[ 2 ] = a2 / DISK.length;
		return out;

	};

	// Whether the factor along the edge a - b (as on flat ground: up 1) is more than a straight line between its ends
	// can hold - read along the edge at every texel, and beside it (a quarter of its length to both sides at the
	// middle, an eighth at the quarters: what crosses the triangles next to it). A matter of the two places alone, in
	// an order that does not depend on which is given first: every triangle that has the edge - in any object, cell
	// or material - cuts it or leaves it alike, and the cut opens no gap.
	const Fa = [ 0, 0, 0 ], Fb = [ 0, 0, 0 ], Fp = [ 0, 0, 0 ], Fq = [ 0, 0, 0 ];
	const BESIDE = [ [ 0.5, 0.25 ], [ 0.25, 0.125 ], [ 0.75, 0.125 ] ];
	const cuts = ( a, b ) => {

		if ( ( a[ 0 ] - b[ 0 ] || a[ 1 ] - b[ 1 ] || a[ 2 ] - b[ 2 ] ) > 0 ) { const t = a; a = b; b = t; }
		const ax = a[ 0 ] / K, ay = a[ 1 ] / K, az = a[ 2 ] / K, dx = ( b[ 0 ] - a[ 0 ] ) / K, dy = ( b[ 1 ] - a[ 1 ] ) / K, dz = ( b[ 2 ] - a[ 2 ] ) / K;
		const len = Math.sqrt( dx * dx + dy * dy + dz * dz );
		if ( ! ( len > q.edge ) ) return false;
		const n = Math.min( 16, Math.max( 2, Math.ceil( len / g.texel ) ) );
		at( ax, az, ay - height( ax, az ), 1, Fa ); at( ax + dx, az + dz, ay + dy - height( ax + dx, az + dz ), 1, Fb );
		for ( let i = 1; i < n; i ++ ) {

			const t = i / n, x = ax + dx * t, z = az + dz * t;
			at( x, z, ay + dy * t - height( x, z ), 1, Fp );
			for ( let k = 0; k < 3; k ++ ) if ( Math.abs( Fp[ k ] - ( Fa[ k ] + ( Fb[ k ] - Fa[ k ] ) * t ) ) > q.tol ) return true;

		}

		const lx = Math.sqrt( dx * dx + dz * dz );
		if ( lx > 1e-6 ) for ( const [ t, o ] of BESIDE ) {

			const x = ax + dx * t, z = az + dz * t, dh = ay + dy * t - height( x, z ), ox = - dz / lx * o * len, oz = dx / lx * o * len;
			at( x + ox, z + oz, dh, 1, Fp ); at( x - ox, z - oz, dh, 1, Fq );
			for ( let k = 0; k < 3; k ++ ) if ( Math.abs( ( Fp[ k ] + Fq[ k ] ) / 2 - ( Fa[ k ] + ( Fb[ k ] - Fa[ k ] ) * t ) ) > q.tol ) return true;

		}

		return false;

	};

	// (lo .. hi: the heights above the terrain within which the factor is not 1)
	const lo = Math.min( q.ao ? - 0.3 : Infinity, b ? b[ 0 ] : Infinity ), hi = Math.max( q.ao ? f0 + 0.12 : - Infinity, b ? b[ 3 ] : - Infinity );
	return { K, cut: q.cut || [], blur: q.blur || {}, lattice: q.lattice || 1, lo, hi, height, at, around, cuts };

}

// Every triangle of `pairs` ([ cell, level ]) by material: Map key -> { kind, layer, glow, n, pos (world, grid units),
// uv (whole numbers, see above; null: none), col (4 per vertex), ix, step (all positions are multiples of it) }.
// P: { K, S, gain( layer ), mean( layer ), glow, clip, pane (dccPane), cMax, UR, CA, noUV( layer ), ground (dccGround; null: not folded) }
function dccCollect( bake, P, pairs ) {

	const accs = new Map();
	let degenerate = 0, mixed = 0, over = 0, channels = 0, bright = 0, panes = 0, added = 0, globes = 0;
	const GR = P.ground || null, ground = { tris: 0, verts: 0, cut: 0, added: 0 };
	for ( const [ c, level ] of pairs ) {

		const m = c[ level ], dec = bake.decode( m, bake.blob( c, level ) ), W = dec.W;
		const rec = new Uint16Array( dec.vb.buffer, dec.vb.byteOffset, m.nv * W );
		const idx = m.it === 2 ? new Uint16Array( dec.ib.buffer, dec.ib.byteOffset, m.ni ) : new Uint32Array( dec.ib.buffer, dec.ib.byteOffset, m.ni );
		const step = Math.round( m.q * P.K ), mi = m.min.map( ( v ) => Math.round( v * P.K ) );
		const groups = new Map();
		for ( const [ kind, start, count ] of m.sub ) {

			const surf = kind === K_OPAQUE || kind === K_CUT;
			for ( let t = start; t < start + count; t += 3 ) {

				const a = idx[ t ], b = idx[ t + 1 ], d = idx[ t + 2 ];
				if ( a === b || b === d || a === d ) { degenerate ++; continue; } // (draws nothing)
				const layer = surf ? rec[ a * W + 3 ] : - 1;
				if ( surf && ( rec[ b * W + 3 ] !== layer || rec[ d * W + 3 ] !== layer ) ) mixed ++;
				const glow = P.glow && kind === K_OPAQUE && rec[ a * W + 9 ] >= GLOW_AUX ? 1 : 0;
				const key = dccKey( kind, layer, glow );
				let list = groups.get( key ); if ( ! list ) groups.set( key, list = { kind, layer, glow, tris: [] } );
				list.tris.push( t );

			}

		}

		const remap = new Int32Array( m.nv );
		let reachOf = null; // (per vertex: 1 within the heights the ground's pictures act on, 2 not)
		const bodies = P.pane ? P.pane.bodies.filter( ( b ) => [ 0, 1, 2 ].every( ( j ) => b[ j ] > m.bmin[ j ] - 0.5 && b[ j ] < m.bmax[ j ] + 0.5 ) ) : [];
		for ( const [ key, { kind, layer, glow, tris } ] of groups ) {

			let acc = accs.get( key );
			if ( ! acc ) accs.set( key, acc = { kind, layer, glow, n: 0, step: 0, pos: new Grow( Int32Array ), uv: ( kind === K_OPAQUE || kind === K_CUT ) && ! P.noUV( layer ) ? new Grow( Int32Array ) : null,
				col: new Grow( P.cMax === 255 ? Uint8Array : Uint16Array ), ix: new Grow( Uint32Array ) } );
			acc.step = gcd( gcd( gcd( acc.step, step ), gcd( mi[ 0 ], mi[ 1 ] ) ), mi[ 2 ] );
			remap.fill( - 1 );
			const G = layer >= 0 ? P.gain( layer ) : 1, TM = layer >= 0 ? P.mean( layer ) : 1, alpha = kind === K_GLASS || kind === K_FALL, cut = kind === K_CUT;
			const PL = kind === K_GLASS && P.pane ? paneLight( P.pane, m, rec, W, idx, tris, P.S, P.pane.tess[ level ] || 0, P.pane.max[ level ] || 1 ) : null;
			// one vertex's colour and alpha: light l (after the gain), alpha al; lamp: a lantern pane
			const tone = [ 0, 0, 0 ], source = !! P.pane && kind === K_OPAQUE && P.noUV( layer );
			const colour = ( l0, l1, l2, al, lamp ) => {

				const ca = acc.col.need( 4 ), cn = acc.col.n;
				const top = Math.max( l0, l1, l2 );
				if ( top > 1 ) {

					over += ( l0 > 1 ) + ( l1 > 1 ) + ( l2 > 1 ); bright ++;
					if ( P.clip !== 'channel' || lamp ) {

						let mx = top;
						// blended glass: what it adds to the picture is colour * alpha, so light the colour cannot hold goes
						// into the alpha (the pane hides that much more of what is behind it), exactly while alpha * light <= 1
						if ( alpha && al > 0 ) { const a2 = Math.min( 1, al * top ), k = al / a2; l0 *= k; l1 *= k; l2 *= k; mx *= k; al = a2; }
						if ( mx > 1 + 1e-9 ) {

							if ( lamp ) {

								// a light seen directly, white-hot on the page: its tone curve (pageTone), coming in over the first
								// unit above 1 so there is no step beside light that fits
								pageTone( l0, l1, l2, tone );
								const w = sstep( 1, 2, mx );
								l0 = l0 / mx; l1 = l1 / mx; l2 = l2 / mx;
								l0 += ( tone[ 0 ] - l0 ) * w; l1 += ( tone[ 1 ] - l1 ) * w; l2 += ( tone[ 2 ] - l2 ) * w;

							} else {

								// what is left above 1 keeps its hue (hue), and pales towards white the further it is above (soft:
								// what the page's tone curve does with it, judged at the layer's mean texel)
								const w = P.clip === 'soft' ? 1 - 1 / Math.sqrt( Math.max( 1, mx * TM ) ) : 0;
								l0 = l0 / mx; l1 = l1 / mx; l2 = l2 / mx;
								l0 += ( 1 - l0 ) * w; l1 += ( 1 - l1 ) * w; l2 += ( 1 - l2 ) * w;

							}

						}

					}

				}

				ca[ cn ] = Math.round( Math.min( l0, 1 ) * P.cMax ); ca[ cn + 1 ] = Math.round( Math.min( l1, 1 ) * P.cMax ); ca[ cn + 2 ] = Math.round( Math.min( l2, 1 ) * P.cMax );
				channels += 3;
				ca[ cn + 3 ] = Math.round( al * P.cMax );
				acc.col.n += 4;

			};

			// The ground's pictures in the opaque surfaces (P.ground, dccGround): what faces up near the terrain takes the
			// page's factor at its vertices, and on a level that is cut (GR.cut) its triangles are halved along every edge
			// GR.cuts() names, again and again, until the vertices can hold the factor. New vertices sit on the grid, with
			// the UV and the light of the middle of their edge.
			if ( GR && kind === K_OPAQUE ) {

				const nv = m.nv, cutting = GR.cut.includes( level ), blur = cutting ? 0 : GR.blur[ level ] || 0, L = GR.lattice, K = P.K;
				if ( cutting ) acc.step = gcd( acc.step, L );
				if ( ! reachOf ) reachOf = new Uint8Array( nv );
				const xp = new Grow( Int32Array, 768 ), xu = new Grow( Int32Array, 512 ), xl = new Grow( Float64Array, 1024 ), xr = new Grow( Uint8Array, 256 ); let nx = 0;
				const vers = new Map(), mids = new Map();
				// (an edge the cut itself drew inside a triangle: how many cuts deep. Such an edge belongs to that triangle's
				// pieces alone, and past GEN it stays whole - so a sliver whose point lies on its own long edge, which the
				// test may leave whole while it halves the parts of it, comes to an end. Edges of the bake's triangles and
				// their halves are never held back: their neighbours cut them by the same test.)
				const gens = new Map(), GEN = 8, ekey = ( a, b ) => ( a < b ? a * 67108864 + b : b * 67108864 + a ), gen = ( a, b ) => gens.get( ekey( a, b ) ) || 0;
				const pN = [ 0, 0, 0 ], pA = [ 0, 0, 0 ], pB = [ 0, 0, 0 ], pC = [ 0, 0, 0 ], pO = [ 0, 0, 0 ], lA = [ 0, 0, 0, 0 ], lB = [ 0, 0, 0, 0 ], lO = [ 0, 0, 0, 0 ], fO = [ 0, 0, 0 ];
				const posOf = ( v, o ) => {

					if ( v < nv ) { const r = v * W; o[ 0 ] = mi[ 0 ] + rec[ r ] * step; o[ 1 ] = mi[ 1 ] + rec[ r + 1 ] * step; o[ 2 ] = mi[ 2 ] + rec[ r + 2 ] * step; }
					else { const r = ( v - nv ) * 3, a = xp.a; o[ 0 ] = a[ r ]; o[ 1 ] = a[ r + 1 ]; o[ 2 ] = a[ r + 2 ]; }
					return o;

				};
				const reach = ( p ) => { const x = p[ 0 ] / K, z = p[ 2 ] / K, dh = p[ 1 ] / K - GR.height( x, z ); return dh > GR.lo && dh < GR.hi; };
				const near = ( v ) => { if ( v >= nv ) return xr.a[ v - nv ] === 1; let r = reachOf[ v ]; if ( ! r ) r = reachOf[ v ] = reach( posOf( v, pN ) ) ? 1 : 2; return r === 1; };
				// a vertex's light before the factor: [ r, g, b, 1: a light seen directly (2: an opal globe) ]
				const lightOf = ( v, o ) => {

					if ( v >= nv ) { const r = ( v - nv ) * 4, a = xl.a; o[ 0 ] = a[ r ]; o[ 1 ] = a[ r + 1 ]; o[ 2 ] = a[ r + 2 ]; o[ 3 ] = a[ r + 3 ]; return o; }
					const r = v * W;
					let body = null;
					if ( bodies.length && rec[ r + 9 ] >= GLOW_AUX ) {

						const x = m.min[ 0 ] + rec[ r ] * m.q, y = m.min[ 1 ] + rec[ r + 1 ] * m.q, z = m.min[ 2 ] + rec[ r + 2 ] * m.q;
						for ( const b of bodies ) if ( ( x - b[ 0 ] ) ** 2 + ( y - b[ 1 ] ) ** 2 + ( z - b[ 2 ] ) ** 2 <= b[ 3 ] ) { body = b; break; }

					}

					if ( body ) { const k = P.pane.body / G; o[ 0 ] = body[ 4 ] * k; o[ 1 ] = body[ 5 ] * k; o[ 2 ] = body[ 6 ] * k; o[ 3 ] = 2; return o; }
					const e0 = rec[ r + 6 ] / 65535, e1 = rec[ r + 7 ] / 65535, e2 = rec[ r + 8 ] / 65535;
					o[ 0 ] = e0 * e0 * P.S / G; o[ 1 ] = e1 * e1 * P.S / G; o[ 2 ] = e2 * e2 * P.S / G; o[ 3 ] = source && rec[ r + 9 ] >= GLOW_AUX ? 1 : 0;
					return o;

				};
				const uvOf = ( v, j ) => ( v < nv ? rec[ v * W + 4 + j ] : xu.a[ ( v - nv ) * 2 + j ] );
				// the vertex in the middle of the edge a - b (- 1: the edge stays whole)
				const split = ( a, b ) => {

					if ( ! near( a ) || ! near( b ) ) return - 1;
					const lo = a < b ? a : b, hi = a < b ? b : a, key = lo * 67108864 + hi;
					let mv = mids.get( key ); if ( mv !== undefined ) return mv;
					mv = - 1;
					const eg = gens.get( key ) || 0;
					posOf( lo, pA ); posOf( hi, pB );
					if ( eg <= GEN && GR.cuts( pA, pB ) ) {

						pN[ 0 ] = Math.floor( ( pA[ 0 ] + pB[ 0 ] ) / ( 2 * L ) + 0.5 ) * L; pN[ 1 ] = Math.floor( ( pA[ 1 ] + pB[ 1 ] ) / ( 2 * L ) + 0.5 ) * L; pN[ 2 ] = Math.floor( ( pA[ 2 ] + pB[ 2 ] ) / ( 2 * L ) + 0.5 ) * L;
						const same = ( p ) => p[ 0 ] === pN[ 0 ] && p[ 1 ] === pN[ 1 ] && p[ 2 ] === pN[ 2 ];
						if ( ! same( pA ) && ! same( pB ) ) {

							mv = nv + nx ++;
							if ( eg ) { gens.set( ekey( lo, mv ), eg ); gens.set( ekey( mv, hi ), eg ); }
							const xa = xp.need( 3 ); xa[ xp.n ] = pN[ 0 ]; xa[ xp.n + 1 ] = pN[ 1 ]; xa[ xp.n + 2 ] = pN[ 2 ]; xp.n += 3;
							const ra = xr.need( 1 ); ra[ xr.n ++ ] = reach( pN ) ? 1 : 2;
							if ( acc.uv ) { const u0 = Math.round( ( uvOf( lo, 0 ) + uvOf( hi, 0 ) ) / 2 ), u1 = Math.round( ( uvOf( lo, 1 ) + uvOf( hi, 1 ) ) / 2 ), ua = xu.need( 2 ); ua[ xu.n ] = u0; ua[ xu.n + 1 ] = u1; xu.n += 2; }
							lightOf( lo, lA ); lightOf( hi, lB );
							const la = xl.need( 4 ); la[ xl.n ] = ( lA[ 0 ] + lB[ 0 ] ) / 2; la[ xl.n + 1 ] = ( lA[ 1 ] + lB[ 1 ] ) / 2; la[ xl.n + 2 ] = ( lA[ 2 ] + lB[ 2 ] ) / 2; la[ xl.n + 3 ] = lA[ 3 ] && lB[ 3 ] ? 1 : 0; xl.n += 4;

						}

					}

					mids.set( key, mv );
					return mv;

				};
				// the file's vertex for v as a triangle that faces up by q eighths has it
				const out = ( v, q ) => {

					if ( q && ! near( v ) ) q = 0;
					const plain = q === 0 && v < nv, key = v * 16 + q;
					let o = plain ? remap[ v ] : vers.get( key );
					if ( o !== undefined && o >= 0 ) return o;
					o = acc.n ++;
					posOf( v, pO );
					const pa = acc.pos.need( 3 ), pn = acc.pos.n; pa[ pn ] = pO[ 0 ]; pa[ pn + 1 ] = pO[ 1 ]; pa[ pn + 2 ] = pO[ 2 ]; acc.pos.n += 3;
					if ( acc.uv ) { const ua = acc.uv.need( 2 ), un = acc.uv.n; ua[ un ] = uvOf( v, 0 ); ua[ un + 1 ] = uvOf( v, 1 ); acc.uv.n += 2; }
					lightOf( v, lO );
					if ( lO[ 3 ] === 2 ) globes ++;
					if ( q && ! lO[ 3 ] ) {

						const x = pO[ 0 ] / K, z = pO[ 2 ] / K, dh = pO[ 1 ] / K - GR.height( x, z );
						if ( blur ) GR.around( x, z, dh, q / 8, blur, fO ); else GR.at( x, z, dh, q / 8, fO );
						lO[ 0 ] *= fO[ 0 ]; lO[ 1 ] *= fO[ 1 ]; lO[ 2 ] *= fO[ 2 ]; ground.verts ++;

					}

					colour( lO[ 0 ], lO[ 1 ], lO[ 2 ], 1, lO[ 3 ] > 0 );
					if ( plain ) remap[ v ] = o; else vers.set( key, o );
					return o;

				};
				let leaves = 0;
				const leaf = ( a, b, e, q ) => { const i0 = out( a, q ), i1 = out( b, q ), i2 = out( e, q ), ixa = acc.ix.need( 3 ), n = acc.ix.n; ixa[ n ] = i0; ixa[ n + 1 ] = i1; ixa[ n + 2 ] = i2; acc.ix.n += 3; leaves ++; };
				const dist2 = ( a, b ) => { posOf( a, pA ); posOf( b, pB ); return ( pA[ 0 ] - pB[ 0 ] ) ** 2 + ( pA[ 1 ] - pB[ 1 ] ) ** 2 + ( pA[ 2 ] - pB[ 2 ] ) ** 2; };
				const emit = ( a, b, e, q, depth ) => {

					let mab = split( a, b ), mbe = split( b, e ), mea = split( e, a );
					const n = ( mab >= 0 ) + ( mbe >= 0 ) + ( mea >= 0 );
					if ( n === 0 ) return leaf( a, b, e, q );
					if ( depth > 200 ) die( `internal: the ground's cut does not end in ${c.k}:${level}` );
					const ng = 1 + Math.max( gen( a, b ), gen( b, e ), gen( e, a ) );
					if ( n === 3 ) { gens.set( ekey( mab, mea ), ng ); gens.set( ekey( mab, mbe ), ng ); gens.set( ekey( mea, mbe ), ng ); emit( a, mab, mea, q, depth + 1 ); emit( mab, b, mbe, q, depth + 1 ); emit( mea, mbe, e, q, depth + 1 ); emit( mab, mbe, mea, q, depth + 1 ); return; }
					if ( n === 2 ) {

						// (turned so that a - b and b - e are the cut edges; the four-sided rest by its shorter diagonal)
						if ( mab < 0 ) { const t = a; a = b; b = e; e = t; mab = mbe; mbe = mea; }
						else if ( mbe < 0 ) { const t = b; b = a; a = e; e = t; mbe = mab; mab = mea; }
						gens.set( ekey( mab, mbe ), ng );
						const first = dist2( a, mbe ) <= dist2( mab, e );
						gens.set( first ? ekey( a, mbe ) : ekey( mab, e ), ng );
						emit( mab, b, mbe, q, depth + 1 );
						if ( first ) { emit( a, mab, mbe, q, depth + 1 ); emit( a, mbe, e, q, depth + 1 ); }
						else { emit( a, mab, e, q, depth + 1 ); emit( mab, mbe, e, q, depth + 1 ); }
						return;

					}

					// (turned so that a - b is the cut edge)
					if ( mbe >= 0 ) { const t = a; a = b; b = e; e = t; mab = mbe; }
					else if ( mea >= 0 ) { const t = b; b = a; a = e; e = t; mab = mea; }
					gens.set( ekey( mab, e ), ng );
					emit( a, mab, e, q, depth + 1 ); emit( mab, b, e, q, depth + 1 );

				};
				for ( let i = 0; i < tris.length; i ++ ) {

					const t = tris[ i ], a = idx[ t ], b = idx[ t + 1 ], e = idx[ t + 2 ];
					const nb = near( a ) + near( b ) + near( e );
					// (how far it faces up, in eighths: the page's weight, from the face normal)
					let q = 0;
					if ( nb ) {

						posOf( a, pA ); posOf( b, pB ); posOf( e, pC );
						const ux = pB[ 0 ] - pA[ 0 ], uy = pB[ 1 ] - pA[ 1 ], uz = pB[ 2 ] - pA[ 2 ], vx = pC[ 0 ] - pA[ 0 ], vy = pC[ 1 ] - pA[ 1 ], vz = pC[ 2 ] - pA[ 2 ];
						const n0 = uy * vz - uz * vy, n1 = uz * vx - ux * vz, n2 = ux * vy - uy * vx, ln = Math.sqrt( n0 * n0 + n1 * n1 + n2 * n2 );
						q = ln > 0 ? Math.round( sstep( 0.5, 0.8, Math.abs( n1 ) / ln ) * 8 ) : 0;
						if ( q ) ground.tris ++;

					}

					leaves = 0;
					if ( cutting && nb >= 2 ) emit( a, b, e, q, 0 ); else leaf( a, b, e, q );
					if ( leaves > 1 ) { ground.cut ++; ground.added += leaves - 1; added += leaves - 1; }

				}

				continue;

			}

			const f = [ 0, 0, 0, 0 ];
			for ( let i = 0; i < tris.length; i ++ ) {

				const N = PL ? PL.cuts[ i ] : 1;
				if ( N > 1 ) {

					// a large pane, cut N x N: new vertices on the cell's grid, each with the pane's light at its place
					const T = PL.tri( i ), r0 = T.v[ 0 ] * W, r1 = T.v[ 1 ] * W, r2 = T.v[ 2 ] * W, first = acc.n;
					for ( let a = 0; a <= N; a ++ ) for ( let b = 0; a + b <= N; b ++ ) {

						const s = a / N, t = b / N, u = 1 - s - t;
						const pa = acc.pos.need( 3 ), pn = acc.pos.n;
						for ( let j = 0; j < 3; j ++ ) pa[ pn + j ] = mi[ j ] + Math.round( u * rec[ r0 + j ] + s * rec[ r1 + j ] + t * rec[ r2 + j ] ) * step;
						acc.pos.n += 3; acc.n ++;
						PL.at( u, s, t, f );
						const al = Math.min( 1, f[ 3 ] );
						colour( al > 0 ? f[ 0 ] / al : 0, al > 0 ? f[ 1 ] / al : 0, al > 0 ? f[ 2 ] / al : 0, al, true );

					}

					// (row a starts at a * ( N + 1 ) - a * ( a - 1 ) / 2)
					const row = ( a ) => first + a * ( N + 1 ) - a * ( a - 1 ) / 2;
					const ixa = acc.ix.need( N * N * 3 ); let ixn = acc.ix.n;
					for ( let a = 0; a < N; a ++ ) for ( let b = 0; a + b < N; b ++ ) {

						ixa[ ixn ++ ] = row( a ) + b; ixa[ ixn ++ ] = row( a + 1 ) + b; ixa[ ixn ++ ] = row( a ) + b + 1;
						if ( a + b < N - 1 ) { ixa[ ixn ++ ] = row( a + 1 ) + b; ixa[ ixn ++ ] = row( a + 1 ) + b + 1; ixa[ ixn ++ ] = row( a ) + b + 1; }

					}

					acc.ix.n = ixn; panes ++; added += N * N - 1;
					continue;

				}

				for ( let j = 0; j < 3; j ++ ) {

					const v = idx[ tris[ i ] + j ];
					if ( remap[ v ] < 0 ) {

						remap[ v ] = acc.n ++;
						const r = v * W, aux = rec[ r + 9 ] / 65535;
						const pa = acc.pos.need( 3 ), pn = acc.pos.n;
						pa[ pn ] = mi[ 0 ] + rec[ r ] * step; pa[ pn + 1 ] = mi[ 1 ] + rec[ r + 1 ] * step; pa[ pn + 2 ] = mi[ 2 ] + rec[ r + 2 ] * step; acc.pos.n += 3;
						if ( acc.uv ) {

							const ua = acc.uv.need( 2 ), un = acc.uv.n;
							if ( cut ) { const k = aux * P.CA * CUT_UV_GRID; ua[ un ] = Math.round( ( rec[ r + 4 ] / 65536 - 0.5 ) * P.UR * k ); ua[ un + 1 ] = Math.round( ( rec[ r + 5 ] / 65536 - 0.5 ) * P.UR * k ); }
							else { ua[ un ] = rec[ r + 4 ]; ua[ un + 1 ] = rec[ r + 5 ]; }
							acc.uv.n += 2;

						}

						if ( PL && PL.per[ v * 5 + 4 ] > 0 ) {

							// (a lantern pane: colour * cover -> colour, cover)
							const q = v * 5, wq = PL.per[ q + 4 ], al = Math.min( 1, PL.per[ q + 3 ] / wq ), k = al > 0 ? 1 / ( wq * al ) : 0;
							colour( PL.per[ q ] * k, PL.per[ q + 1 ] * k, PL.per[ q + 2 ] * k, al, true );

						} else {

							// (an opal globe around a burner: on the page the halo before it is what is lit, the globe itself is
							// baked as a pale surface - here the globe takes the lamp's light, evenly)
							let body = null;
							if ( bodies.length && kind === K_OPAQUE && rec[ r + 9 ] >= GLOW_AUX ) {

								const x = m.min[ 0 ] + rec[ r ] * m.q, y = m.min[ 1 ] + rec[ r + 1 ] * m.q, z = m.min[ 2 ] + rec[ r + 2 ] * m.q;
								for ( const b of bodies ) if ( ( x - b[ 0 ] ) ** 2 + ( y - b[ 1 ] ) ** 2 + ( z - b[ 2 ] ) ** 2 <= b[ 3 ] ) { body = b; break; }

							}

							// (stored as a square root of light / colour_scale)
							const e0 = rec[ r + 6 ] / 65535, e1 = rec[ r + 7 ] / 65535, e2 = rec[ r + 8 ] / 65535;
							if ( body ) { const k = P.pane.body / G; colour( body[ 4 ] * k, body[ 5 ] * k, body[ 6 ] * k, 1, true ); globes ++; }
							else if ( kind === K_FALL ) {

								// (the mean of the page's torn sheet at this vertex's masks: see fallTable)
								const F = fallTable(), q = ( Math.min( 15, rec[ r + 3 ] >> 4 ) * 16 + ( rec[ r + 3 ] & 15 ) ) * 3, cv = F[ q ], k = cv > 0 ? F[ q + 1 ] / cv : 0, kb = cv > 0 ? F[ q + 2 ] / cv : 0;
								colour( e0 * e0 * P.S * k + FALL_BEAD[ 0 ] * kb, e1 * e1 * P.S * k + FALL_BEAD[ 1 ] * kb, e2 * e2 * P.S * k + FALL_BEAD[ 2 ] * kb, aux * cv, false );

							} else colour( e0 * e0 * P.S / G, e1 * e1 * P.S / G, e2 * e2 * P.S / G, alpha ? aux : 1, source && rec[ r + 9 ] >= GLOW_AUX );

						}

					}

					const ixa = acc.ix.need( 1 ); ixa[ acc.ix.n ++ ] = remap[ v ];

				}

				if ( N === 1 && PL ) panes ++;

			}

		}

	}

	return { accs, degenerate, mixed, over, channels, bright, panes, added, globes, ground };

}

// a sum over the triangles that does not depend on their order, nor on which corner a triangle starts with: [ sum, sum, triangles with area ]
function dccHash( sum, pos, uv, col, ix, ni ) {

	const corner = ( v ) => {

		let h = 0x811C9DC5;
		h = hmix( h, pos[ v * 3 ] ); h = hmix( h, pos[ v * 3 + 1 ] ); h = hmix( h, pos[ v * 3 + 2 ] );
		if ( uv ) { h = hmix( h, uv[ v * 2 ] ); h = hmix( h, uv[ v * 2 + 1 ] ); }
		h = hmix( h, col[ v * 4 ] ); h = hmix( h, col[ v * 4 + 1 ] ); h = hmix( h, col[ v * 4 + 2 ] ); h = hmix( h, col[ v * 4 + 3 ] );
		return h;

	};

	const same = ( a, b ) => pos[ a * 3 ] === pos[ b * 3 ] && pos[ a * 3 + 1 ] === pos[ b * 3 + 1 ] && pos[ a * 3 + 2 ] === pos[ b * 3 + 2 ];
	for ( let t = 0; t < ni; t += 3 ) {

		const i0 = ix[ t ], i1 = ix[ t + 1 ], i2 = ix[ t + 2 ];
		if ( same( i0, i1 ) || same( i1, i2 ) || same( i0, i2 ) ) continue; // (two corners in one place: no area. Draco drops those)
		const a = corner( i0 ), b = corner( i1 ), c = corner( i2 );
		const h = ( Math.imul( a, b | 1 ) + Math.imul( b, c | 1 ) + Math.imul( c, a | 1 ) ) >>> 0;
		sum[ 0 ] = ( sum[ 0 ] + h ) >>> 0; sum[ 1 ] = ( sum[ 1 ] + hmix( h, 0x5bd1e995 ) ) >>> 0; sum[ 2 ] ++;

	}

	return sum;

}

function dracoEncode( D, o, p ) {

	const enc = new D.Encoder(), mb = new D.MeshBuilder(), mesh = new D.Mesh();
	mb.AddFacesToMesh( mesh, p.ix.length / 3, p.ix );
	const ids = { POSITION: mb.AddFloatAttribute( mesh, D.POSITION, p.n, 3, p.pos ) };
	enc.SetAttributeExplicitQuantization( D.POSITION, p.posQ.bits, 3, p.posQ.origin, p.posQ.range );
	if ( p.uv ) { ids.TEXCOORD_0 = mb.AddFloatAttribute( mesh, D.TEX_COORD, p.n, 2, p.uv ); enc.SetAttributeExplicitQuantization( D.TEX_COORD, p.uvQ.bits, 2, p.uvQ.origin, p.uvQ.range ); }
	if ( p.col ) ids.COLOR_0 = p.col instanceof Uint16Array ? mb.AddUInt16Attribute( mesh, D.COLOR, p.n, p.nc, p.col ) : mb.AddUInt8Attribute( mesh, D.COLOR, p.n, p.nc, p.col );
	enc.SetEncodingMethod( D.MESH_EDGEBREAKER_ENCODING );
	enc.SetSpeedOptions( 10 - o.dracoLevel, 10 - o.dracoLevel );
	enc.SetTrackEncodedProperties( true );
	const out = new D.DracoInt8Array();
	const len = enc.EncodeMeshToDracoBuffer( mesh, out );
	if ( ! len ) die( 'internal: Draco did not encode a primitive' );
	const bytes = new Uint8Array( len );
	for ( let i = 0; i < len; i ++ ) bytes[ i ] = out.GetValue( i );
	const res = { bytes, ids, points: enc.GetNumberOfEncodedPoints(), faces: enc.GetNumberOfEncodedFaces() };
	D.destroy( out ); D.destroy( mesh ); D.destroy( mb ); D.destroy( enc );
	return res;

}

function dracoDecode( D, bytes, ids, col16 ) {

	const dec = new D.Decoder(), buf = new D.DecoderBuffer();
	buf.Init( new Int8Array( bytes.buffer, bytes.byteOffset, bytes.byteLength ), bytes.byteLength );
	const mesh = new D.Mesh();
	if ( ! dec.DecodeBufferToMesh( buf, mesh ).ok() ) throw new Error( 'Draco does not decode it' );
	const nf = mesh.num_faces(), np = mesh.num_points();
	const read = ( id, type, T, comps ) => {

		const at = dec.GetAttributeByUniqueId( mesh, id ), n = comps || at.num_components(), len = np * n * T.BYTES_PER_ELEMENT, ptr = D._malloc( len );
		dec.GetAttributeDataArrayForAllPoints( mesh, at, type, len, ptr );
		const a = new T( D.HEAPU8.buffer.slice( ptr, ptr + len ) ); D._free( ptr );
		return { a, n };

	};

	const iptr = D._malloc( nf * 12 ); dec.GetTrianglesUInt32Array( mesh, nf * 12, iptr );
	const ix = new Uint32Array( D.HEAPU8.buffer.slice( iptr, iptr + nf * 12 ) ); D._free( iptr );
	const res = { np, nf, ix, pos: read( ids.POSITION, D.DT_FLOAT32, Float32Array ).a, uv: ids.TEXCOORD_0 !== undefined ? read( ids.TEXCOORD_0, D.DT_FLOAT32, Float32Array ).a : null,
		col: ids.COLOR_0 === undefined ? null : col16 ? read( ids.COLOR_0, D.DT_UINT16, Uint16Array ) : read( ids.COLOR_0, D.DT_UINT8, Uint8Array ) };
	D.destroy( mesh ); D.destroy( buf ); D.destroy( dec );
	return res;

}

// The night sky of the town page (web/engine/engine.js createSky, with the numbers web/engine/lite.js setRig and
// web/town/main.js LOOK give it), evaluated here because a stock material has no shader: the navy gradient, the town's
// glow on the haze, the moonlit cloud deck, the moon, the stars. As seen from the street (the page's `aloft` term is 0),
// at the moment SKY_TIME of the page's clock (the deck drifts, the stars twinkle): 2 s, where the page's review pictures are.
const SKY_LOOK = { zenith: [ 0.004, 0.009, 0.03 ], horizon: [ 0.018, 0.034, 0.085 ], glow: [ 0.026, 0.016, 0.008 ], cloud: [ 0.05, 0.075, 0.14 ], ground: [ 0.006, 0.009, 0.018 ],
	clouds: 1.0, cover: 0.36, glowGain: 4, cloudGain: 2.4, stars: 1, moon: 1 };
const SKY_TIME = 2, SKY_N = 8, SKY_MARGIN = 4, SKY_GRID = 64, SKY_UV = 16384;
const sat1 = ( x ) => Math.min( 1, Math.max( 0, x ) );

// one channel of the engine's noise as the GPU samples it: repeat, bilinear, between two mip levels
function noiseAt( N, u, v, lod, c ) {

	const top = N.mips.length - 1;
	lod = Math.min( top, Math.max( 0, lod ) );
	const l0 = Math.floor( lod ), f = lod - l0;
	const one = ( l ) => {

		const s = N.size >> l, m = N.mips[ l ];
		const x = u * s - 0.5, y = v * s - 0.5;
		const x0 = Math.floor( x ), y0 = Math.floor( y ), tx = x - x0, ty = y - y0;
		const xa = ( ( x0 % s ) + s ) % s, xb = ( xa + 1 ) % s, ya = ( ( y0 % s ) + s ) % s, yb = ( ya + 1 ) % s;
		const a = m[ ( ya * s + xa ) * 4 + c ], b = m[ ( ya * s + xb ) * 4 + c ], d = m[ ( yb * s + xa ) * 4 + c ], e = m[ ( yb * s + xb ) * 4 + c ];
		return a + ( b - a ) * tx + ( d - a ) * ty + ( a - b - d + e ) * tx * ty;

	};

	const a = one( l0 );
	return f > 0 && l0 < top ? a + ( one( l0 + 1 ) - a ) * f : a;

}

// everything but the stars in the direction d (unit, y up), into out[ 0 .. 2 ]; returns what a star there keeps
// (the clouds before it, the haze low down). pix: the angle of a pixel (for the mip level of the noise)
function skyBase( L, N, moon, time, pix, dx, dy, dz, out ) {

	const h = dy, hh = Math.max( h, 0 );
	const k = Math.pow( sat1( h ), 0.42 );
	const e7 = Math.exp( - 7 * hh ), e24 = Math.exp( - 2.4 * hh );
	const g0 = L.glow[ 0 ] * L.glowGain, g1 = L.glow[ 1 ] * L.glowGain, g2 = L.glow[ 2 ] * L.glowGain;
	let r = L.horizon[ 0 ] + ( L.zenith[ 0 ] - L.horizon[ 0 ] ) * k + g0 * ( e7 + e24 * 0.3 );
	let g = L.horizon[ 1 ] + ( L.zenith[ 1 ] - L.horizon[ 1 ] ) * k + g1 * ( e7 + e24 * 0.42 );
	let b = L.horizon[ 2 ] + ( L.zenith[ 2 ] - L.horizon[ 2 ] ) * k + g2 * ( e7 + e24 * 0.85 );
	const under = sstep( 0, - 0.25, h );
	r += ( L.ground[ 0 ] - r ) * under; g += ( L.ground[ 1 ] - g ) * under; b += ( L.ground[ 2 ] - b ) * under;
	const md = dx * moon[ 0 ] + dy * moon[ 1 ] + dz * moon[ 2 ];
	let cloud = 0;
	const low = sstep( - 0.02, 0.06, h );
	if ( low > 0 && L.clouds > 0 ) {

		const iv = 1 / ( hh + 0.2 ), qx = dx * iv, qy = dz * iv;
		// how fast q turns with the direction (the larger of along and across the horizon), for the mip level
		const ce = Math.sqrt( Math.max( 0, 1 - dy * dy ) );
		const rho = Math.max( iv, h > 0 ? Math.abs( dy ) * iv + ce * ce * iv * iv : iv ) * pix * N.size;
		const lod = ( s ) => Math.log2( Math.max( 1e-6, rho * s ) );
		const drx = time * 0.0016, dry = time * 0.0006;
		const dens = ( px, py ) => noiseAt( N, px * 0.16 + drx, py * 0.16 + dry, lod( 0.16 ), 0 ) * 0.6
			+ noiseAt( N, ( px * 0.44 + py * 0.9 ) * 0.42 + drx * 1.7, ( py * 0.44 - px * 0.9 ) * 0.42 + dry * 1.7, lod( 0.42 ), 1 ) * 0.4;
		const ml = Math.hypot( moon[ 0 ], moon[ 2 ] ) || 1;
		const n0 = dens( qx, qy ), n1 = dens( qx + moon[ 0 ] / ml * 0.3, qy + moon[ 2 ] / ml * 0.3 );
		const fu = qx * 1.3 + drx * 3, fv = qy * 1.3 + dry * 3, fl = lod( 1.3 );
		const fineB = noiseAt( N, fu, fv, fl, 2 ), fineA = noiseAt( N, fu, fv, fl, 3 );
		const nn = n0 + ( fineB - 0.5 ) * 0.14;
		cloud = sstep( L.cover, L.cover + 0.2, nn ) * low * L.clouds;
		if ( cloud > 0 ) {

			const thin = 1 - sstep( L.cover + 0.08, L.cover + 0.3, nn );
			const side = sat1( ( n0 - n1 ) * 7 + 0.45 );
			const toMoon = Math.pow( sat1( md * 0.5 + 0.5 ), 3 );
			const lit = ( side * ( thin * 0.75 + 0.25 ) * ( fineA * 0.9 + 0.55 ) * 1.1 + 0.16 ) * ( toMoon * 1.4 + 0.75 ) + Math.pow( sat1( md ), 14 ) * thin * 2.5;
			const base = Math.exp( - 4.5 * hh ) * 0.8;
			const c0 = ( L.cloud[ 0 ] * lit + g0 * base ) * L.cloudGain, c1 = ( L.cloud[ 1 ] * lit + g1 * base ) * L.cloudGain, c2 = ( L.cloud[ 2 ] * lit + g2 * base ) * L.cloudGain;
			r += ( c0 - r ) * cloud; g += ( c1 - g ) * cloud; b += ( c2 - b ) * cloud;

		}

	}

	const disc = sstep( 0.99965, 0.9998, md );
	const halo = Math.exp( - 900 * ( 1 - md ) ) * 0.06 + Math.exp( - 60 * ( 1 - md ) ) * 0.012;
	const m = ( disc * 2.2 + halo ) * L.moon;
	out[ 0 ] = r + 0.85 * m; out[ 1 ] = g + 0.9 * m; out[ 2 ] = b + m;
	return sstep( 0.02, 0.3, h ) * ( 1 - sat1( cloud * 1.6 ) ) * ( 1 - cloud ) * L.stars;

}

// the page's stars in the direction d, added to out: hashed cells of the direction sphere. (The hash is the shader's
// fract( sin( x ) * 43758.5453 ) in single precision; a GPU's sine of such an x is its own, so these are stars of the
// same kind and number as the page's, not the same stars.)
function skyStar( time, dx, dy, dz, out ) {

	const fr = Math.fround;
	const px = dx * 170, py = dy * 170, pz = dz * 170;
	const cx = Math.floor( px ), cy = Math.floor( py ), cz = Math.floor( pz );
	const s = fr( fr( Math.sin( fr( fr( cx * fr( 127.1 ) ) + fr( cy * fr( 311.7 ) ) + fr( cz * fr( 74.7 ) ) ) ) ) * fr( 43758.5453 ) );
	const hsh = s - Math.floor( s );
	if ( hsh < 0.988 ) return;
	const t2 = fr( hsh * fr( 91.345 ) ), hs2 = t2 - Math.floor( t2 );
	const o = ( hs2 - 0.5 ) * 0.5;
	const st = sstep( 0.22, 0.02, Math.hypot( px - cx - 0.5 + o, py - cy - 0.5 + o, pz - cz - 0.5 + o ) );
	if ( st <= 0 ) return;
	const tw = Math.sin( time * 2 + hsh * 300 ) * 0.25 + 0.75;
	const k = st * tw * ( hs2 * 0.9 + 0.25 );
	out[ 0 ] += ( 0.75 + 0.25 * hs2 ) * k; out[ 1 ] += ( 0.82 + 0.04 * hs2 ) * k; out[ 2 ] += ( 1.0 - 0.3 * hs2 ) * k;

}

// The panorama: W x W / 2 texels, 8 bits sRGB, in three.js' own equirectangular layout ( u = 0.5 + atan2( z, x ) / 2 pi,
// rows from the zenith down ). The smooth part is taken 2 x 2 times a texel with the noise filtered as on the page
// (a pixel of 62 degrees / 720), the stars 4 x 4 times. Light above 1 (the moon's disc, the cloud beside it) is capped
// at 1, as everything in the file is.
function skyPanorama( L, moon, W, time ) {

	const H = W / 2, N = engineNoise(), out = new Uint8Array( W * H * 3 ), c = [ 0, 0, 0 ], st = [ 0, 0, 0 ], ss = 2, SS = 4;
	const ml = Math.hypot( ...moon ), mo = moon.map( ( v ) => v / ml );
	const pix = Math.max( FOV / 720 * Math.PI / 180, 2 * Math.PI / W / ss );
	const enc = ( v ) => { v = Math.min( 1, Math.max( 0, v ) ); return Math.min( 255, Math.floor( ( v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow( v, 1 / 2.4 ) - 0.055 ) * 255 + 0.5 ) ); };
	for ( let j = 0; j < H; j ++ ) {

		const n = ( 0.5 - ( j + 0.5 ) / H ) * Math.PI < - 0.3 ? 1 : ss; // (far under the horizon: one colour)
		for ( let i = 0; i < W; i ++ ) {

			let r = 0, g = 0, b = 0, keep = 0;
			for ( let b2 = 0; b2 < n; b2 ++ ) for ( let a2 = 0; a2 < n; a2 ++ ) {

				const el = ( 0.5 - ( j + ( b2 + 0.5 ) / n ) / H ) * Math.PI, ph = ( ( i + ( a2 + 0.5 ) / n ) / W - 0.5 ) * 2 * Math.PI, ce = Math.cos( el );
				keep += skyBase( L, N, mo, time, pix, ce * Math.cos( ph ), Math.sin( el ), ce * Math.sin( ph ), c );
				r += c[ 0 ]; g += c[ 1 ]; b += c[ 2 ];

			}

			r /= n * n; g /= n * n; b /= n * n; keep /= n * n;
			if ( keep > 0 ) {

				st[ 0 ] = st[ 1 ] = st[ 2 ] = 0;
				for ( let b2 = 0; b2 < SS; b2 ++ ) for ( let a2 = 0; a2 < SS; a2 ++ ) {

					const el = ( 0.5 - ( j + ( b2 + 0.5 ) / SS ) / H ) * Math.PI, ph = ( ( i + ( a2 + 0.5 ) / SS ) / W - 0.5 ) * 2 * Math.PI, ce = Math.cos( el );
					skyStar( time, ce * Math.cos( ph ), Math.sin( el ), ce * Math.sin( ph ), st );

				}

				const k = keep / ( SS * SS );
				r += st[ 0 ] * k; g += st[ 1 ] * k; b += st[ 2 ] * k;

			}

			const q = ( j * W + i ) * 3;
			out[ q ] = enc( r ); out[ q + 1 ] = enc( g ); out[ q + 2 ] = enc( b );

		}

	}

	return out;

}

// The shell: a dome |X / a|^n + |Z / b|^n + ( Y / c )^n = 1 standing on a flat floor, around the middle of the content:
// the smallest of the content's proportions that holds every box of `boxes` grown by `margin` and a half sphere of
// `minDist` around C, the centre of the panorama. n = 2 would be half an ellipsoid; 8 is a box with rounded edges.
function skyFit( boxes, C, n, margin, minDist ) {

	const all = boxes.map( ( b ) => [ b[ 0 ] - margin, b[ 1 ], b[ 2 ] - margin, b[ 3 ] + margin, b[ 4 ] + margin, b[ 5 ] + margin ] );
	if ( minDist > 0 ) for ( let el = 0; el <= 90; el += 3 ) for ( let az = 0; az < 360; az += 3 ) {

		const ce = Math.cos( el * Math.PI / 180 ), x = C[ 0 ] + minDist * ce * Math.cos( az * Math.PI / 180 ), y = C[ 1 ] + minDist * Math.sin( el * Math.PI / 180 ), z = C[ 2 ] + minDist * ce * Math.sin( az * Math.PI / 180 );
		all.push( [ x, y, z, x, y, z ] );

	}

	const B = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ];
	for ( const b of all ) for ( let j = 0; j < 3; j ++ ) { B[ j ] = Math.min( B[ j ], b[ j ] ); B[ j + 3 ] = Math.max( B[ j + 3 ], b[ j + 3 ] ); }
	const floor = Math.floor( ( Math.min( B[ 1 ], C[ 1 ] - 2 ) - 1 ) * SKY_GRID ) / SKY_GRID; // (1 m under the lowest point)
	const cx = ( B[ 0 ] + B[ 3 ] ) / 2, cz = ( B[ 2 ] + B[ 5 ] ) / 2;
	const a0 = ( B[ 3 ] - B[ 0 ] ) / 2, b0 = ( B[ 5 ] - B[ 2 ] ) / 2, c0 = B[ 4 ] - floor;
	let s = 1;
	for ( const b of all ) {

		const X = Math.max( Math.abs( b[ 0 ] - cx ), Math.abs( b[ 3 ] - cx ) ), Z = Math.max( Math.abs( b[ 2 ] - cz ), Math.abs( b[ 5 ] - cz ) ), Y = b[ 4 ] - floor;
		s = Math.max( s, Math.pow( Math.pow( X / a0, n ) + Math.pow( Z / b0, n ) + Math.pow( Y / c0, n ), 1 / n ) );

	}

	return { n, centre: [ cx, floor, cz ], a: a0 * s, b: b0 * s, c: c0 * s, floor, margin, min: minDist };

}

// how far along the ray from C in the direction d (unit) the shell is
function skyHit( S, C, dx, dy, dz ) {

	const f = ( t ) => Math.pow( Math.abs( C[ 0 ] + dx * t - S.centre[ 0 ] ) / S.a, S.n ) + Math.pow( Math.abs( C[ 2 ] + dz * t - S.centre[ 2 ] ) / S.b, S.n ) + Math.pow( Math.max( C[ 1 ] + dy * t - S.floor, 0 ) / S.c, S.n ) - 1;
	let tf = Infinity;
	if ( dy < 0 ) { tf = ( C[ 1 ] - S.floor ) / - dy; if ( f( tf ) <= 0 ) return tf; }
	let lo = 0, hi = 1;
	while ( f( hi ) < 0 ) { lo = hi; hi *= 2; if ( hi > 1e7 ) die( 'internal: the sky\'s centre is not inside its shell' ); }
	for ( let i = 0; i < 60; i ++ ) { const m = ( lo + hi ) / 2; if ( f( m ) < 0 ) lo = m; else hi = m; }
	return Math.min( tf, ( lo + hi ) / 2 );

}

// The mesh: `cols` columns around (the seam, towards -x, twice: u = 0 and u = 1), rows of at most `step` degrees from the
// zenith to the rim (where the dome stands on the floor), floor rows from there to the nadir; at the two poles a vertex
// for every column. UV = the direction from C, as three.js reads an equirectangular image (+v down, as glTF has it).
// Positions on a 1/64 m grid, UVs on 1/16384. The triangles face C.
function skyMesh( S, C, cols = 128, step = 1.5 ) {

	const G = SKY_GRID, D2R = Math.PI / 180;
	const az = ( i ) => - Math.PI + 2 * Math.PI * i / cols;
	const hit = ( a, el ) => { const dx = Math.cos( el ) * Math.cos( a ), dy = Math.sin( el ), dz = Math.cos( el ) * Math.sin( a ), t = skyHit( S, C, dx, dy, dz ); return [ C[ 0 ] + dx * t, C[ 1 ] + dy * t, C[ 2 ] + dz * t ]; };
	const rim = [];
	let rimMin = 0, rimMax = - 90;
	for ( let i = 0; i <= cols; i ++ ) {

		// the lowest elevation at which the ray meets the dome and not the floor
		let lo = - 89.9 * D2R, hi = 0;
		for ( let k = 0; k < 50; k ++ ) { const m = ( lo + hi ) / 2; if ( hit( az( i ), m )[ 1 ] - S.floor < 1e-6 ) lo = m; else hi = m; }
		rim.push( lo ); rimMin = Math.min( rimMin, lo ); rimMax = Math.max( rimMax, lo );

	}

	const nd = Math.ceil( ( Math.PI / 2 - rimMin ) / ( step * D2R ) ), nf = Math.ceil( Math.log( ( Math.PI / 2 ) / - rimMax ) / Math.log( 1.2 ) );
	const pos = [], uv = [], ix = [];
	const vert = ( a, el, u ) => {

		let p = Math.abs( el ) >= Math.PI / 2 - 1e-9 ? [ C[ 0 ], C[ 1 ] + Math.sign( el ) * skyHit( S, C, 0, Math.sign( el ), 0 ), C[ 2 ] ] : hit( a, el );
		p = [ Math.round( p[ 0 ] * G ) / G, Math.max( S.floor, Math.round( p[ 1 ] * G ) / G ), Math.round( p[ 2 ] * G ) / G ];
		pos.push( ...p );
		// v from where the vertex came to lie; u is the column's, exactly
		const d = [ p[ 0 ] - C[ 0 ], p[ 1 ] - C[ 1 ], p[ 2 ] - C[ 2 ] ];
		uv.push( u, Math.round( ( 0.5 - Math.asin( d[ 1 ] / Math.hypot( ...d ) ) / Math.PI ) * SKY_UV ) / SKY_UV );
		return pos.length / 3 - 1;

	};

	// rows from the zenith down: 0 the pole, 1 .. nd the dome (nd: the rim), then the floor, nd + nf the pole below
	const rows = nd + nf, id = [];
	for ( let r = 0; r <= rows; r ++ ) {

		const row = [];
		if ( r === 0 || r === rows ) for ( let i = 0; i < cols; i ++ ) row.push( vert( 0, r === 0 ? Math.PI / 2 : - Math.PI / 2, ( i + 0.5 ) / cols ) );
		else for ( let i = 0; i <= cols; i ++ ) row.push( vert( az( i ), r <= nd ? Math.PI / 2 - ( Math.PI / 2 - rim[ i ] ) * r / nd : rim[ i ] * Math.pow( ( Math.PI / 2 ) / - rim[ i ], ( r - nd ) / nf ), i / cols ) );
		id.push( row );

	}

	const tri = ( a, b, c ) => {

		// counter-clockwise as seen from C
		const ax = pos[ a * 3 ], ay = pos[ a * 3 + 1 ], azz = pos[ a * 3 + 2 ];
		const ux = pos[ b * 3 ] - ax, uy = pos[ b * 3 + 1 ] - ay, uz = pos[ b * 3 + 2 ] - azz, vx = pos[ c * 3 ] - ax, vy = pos[ c * 3 + 1 ] - ay, vz = pos[ c * 3 + 2 ] - azz;
		const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
		if ( nx * nx + ny * ny + nz * nz < 1e-12 ) return;
		if ( nx * ( C[ 0 ] - ax ) + ny * ( C[ 1 ] - ay ) + nz * ( C[ 2 ] - azz ) > 0 ) ix.push( a, b, c ); else ix.push( a, c, b );

	};

	for ( let r = 0; r < rows; r ++ ) for ( let i = 0; i < cols; i ++ ) {

		if ( r === 0 ) tri( id[ 0 ][ i ], id[ 1 ][ i ], id[ 1 ][ i + 1 ] );
		else if ( r === rows - 1 ) tri( id[ rows ][ i ], id[ r ][ i ], id[ r ][ i + 1 ] );
		else { tri( id[ r ][ i ], id[ r + 1 ][ i ], id[ r + 1 ][ i + 1 ] ); tri( id[ r ][ i ], id[ r + 1 ][ i + 1 ], id[ r ][ i + 1 ] ); }

	}

	return { pos: new Float32Array( pos ), uv: new Float32Array( uv ), ix: new Uint32Array( ix ), n: pos.length / 3 };

}

// What a sky mesh is worth, from its numbers alone: `back` triangles that do not face C; `swim`, how far the image lies
// from its direction inside a triangle above the horizon (texels of a W-wide panorama, at the middle of every triangle
// and of its edges; `swimLow`: below it, where the image is haze); `near`, the least distance from C above eye height;
// `inside`, how far within the shell the nearest corner of `boxes` is, along its ray from C (not positive: it shows through)
function skyCheck( M, C, boxes, W ) {

	const { pos, uv, ix } = M;
	let swim = 0, swimLow = 0, back = 0, near = Infinity;
	const mn = [ Infinity, Infinity, Infinity ], mx = [ - Infinity, - Infinity, - Infinity ];
	for ( let i = 0; i < pos.length; i ++ ) { const j = i % 3; if ( pos[ i ] < mn[ j ] ) mn[ j ] = pos[ i ]; if ( pos[ i ] > mx[ j ] ) mx[ j ] = pos[ i ]; }
	const W4 = [ [ 1 / 3, 1 / 3, 1 / 3 ], [ 0.5, 0.5, 0 ], [ 0, 0.5, 0.5 ], [ 0.5, 0, 0.5 ] ];
	for ( let t = 0; t < ix.length; t += 3 ) {

		const a = ix[ t ], b = ix[ t + 1 ], c = ix[ t + 2 ];
		const ax = pos[ a * 3 ], ay = pos[ a * 3 + 1 ], az = pos[ a * 3 + 2 ], bx = pos[ b * 3 ], by = pos[ b * 3 + 1 ], bz = pos[ b * 3 + 2 ], cx = pos[ c * 3 ], cy = pos[ c * 3 + 1 ], cz = pos[ c * 3 + 2 ];
		const nx = ( by - ay ) * ( cz - az ) - ( bz - az ) * ( cy - ay ), ny = ( bz - az ) * ( cx - ax ) - ( bx - ax ) * ( cz - az ), nz = ( bx - ax ) * ( cy - ay ) - ( by - ay ) * ( cx - ax );
		if ( ! ( nx * ( C[ 0 ] - ax ) + ny * ( C[ 1 ] - ay ) + nz * ( C[ 2 ] - az ) > 0 ) ) back ++;
		for ( const [ wa, wb, wc ] of W4 ) {

			const dx = ax * wa + bx * wb + cx * wc - C[ 0 ], dy = ay * wa + by * wb + cy * wc - C[ 1 ], dz = az * wa + bz * wb + cz * wc - C[ 2 ], l = Math.hypot( dx, dy, dz ), el = Math.asin( dy / l );
			const u = uv[ a * 2 ] * wa + uv[ b * 2 ] * wb + uv[ c * 2 ] * wc, v = uv[ a * 2 + 1 ] * wa + uv[ b * 2 + 1 ] * wb + uv[ c * 2 + 1 ] * wc;
			let du = Math.abs( 0.5 + Math.atan2( dz, dx ) / ( 2 * Math.PI ) - u ); du = Math.min( du, 1 - du );
			const e = Math.hypot( du * W * Math.cos( el ), ( 0.5 - el / Math.PI - v ) * W / 2 );
			if ( el > 0 ) { swim = Math.max( swim, e ); near = Math.min( near, l ); } else swimLow = Math.max( swimLow, e );

		}

	}

	let inside = Infinity;
	for ( const bx of boxes ) for ( const x of [ bx[ 0 ], bx[ 3 ] ] ) for ( const y of [ bx[ 1 ], bx[ 4 ] ] ) for ( const z of [ bx[ 2 ], bx[ 5 ] ] ) {

		const d = [ x - C[ 0 ], y - C[ 1 ], z - C[ 2 ] ], l = Math.hypot( ...d ); if ( l < 1e-6 ) continue;
		inside = Math.min( inside, rayMesh( M, C, d[ 0 ] / l, d[ 1 ] / l, d[ 2 ] / l ) - l );

	}

	return { swim, swimLow, back, near, inside, min: mn, max: mx };

}

// the distance at which the ray from O along d (unit) meets the mesh (Infinity: nowhere)
function rayMesh( M, O, dx, dy, dz ) {

	const { pos, ix } = M;
	let best = Infinity;
	for ( let t = 0; t < ix.length; t += 3 ) {

		const a = ix[ t ] * 3, b = ix[ t + 1 ] * 3, c = ix[ t + 2 ] * 3;
		const e1x = pos[ b ] - pos[ a ], e1y = pos[ b + 1 ] - pos[ a + 1 ], e1z = pos[ b + 2 ] - pos[ a + 2 ], e2x = pos[ c ] - pos[ a ], e2y = pos[ c + 1 ] - pos[ a + 1 ], e2z = pos[ c + 2 ] - pos[ a + 2 ];
		const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
		const det = e1x * px + e1y * py + e1z * pz; if ( Math.abs( det ) < 1e-12 ) continue;
		const tx = O[ 0 ] - pos[ a ], ty = O[ 1 ] - pos[ a + 1 ], tz = O[ 2 ] - pos[ a + 2 ];
		const u = ( tx * px + ty * py + tz * pz ) / det; if ( u < - 1e-7 || u > 1 + 1e-7 ) continue;
		const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
		const v = ( dx * qx + dy * qy + dz * qz ) / det; if ( v < - 1e-7 || u + v > 1 + 1e-7 ) continue;
		const tt = ( e2x * qx + e2y * qy + e2z * qz ) / det;
		if ( tt > 1e-6 && tt < best ) best = tt;

	}

	return best;

}

// How many degrees the sky slides for someone who walks from A to B (eye positions): the angle between the two sight
// lines to the same point of the shell, over the directions of the panorama from 5 degrees up (the largest, the mean
// by solid angle), and for the moon. A sky at infinity: 0.
function skyParallax( S, C, A, B, moon ) {

	const D2R = Math.PI / 180;
	const at = ( dx, dy, dz ) => {

		const t = skyHit( S, C, dx, dy, dz ), p = [ C[ 0 ] + dx * t, C[ 1 ] + dy * t, C[ 2 ] + dz * t ];
		const a = [ p[ 0 ] - A[ 0 ], p[ 1 ] - A[ 1 ], p[ 2 ] - A[ 2 ] ], b = [ p[ 0 ] - B[ 0 ], p[ 1 ] - B[ 1 ], p[ 2 ] - B[ 2 ] ];
		return Math.acos( Math.min( 1, ( a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ] ) / ( Math.hypot( ...a ) * Math.hypot( ...b ) ) ) ) / D2R;

	};

	let max = 0, sum = 0, w = 0;
	for ( let el = 5; el <= 90; el += 1 ) for ( let az = 0; az < 360; az += 2 ) {

		const ce = Math.cos( el * D2R ), v = at( ce * Math.cos( az * D2R ), Math.sin( el * D2R ), ce * Math.sin( az * D2R ) );
		max = Math.max( max, v ); sum += v * ce; w += ce;

	}

	const ml = Math.hypot( ...moon );
	return { max: round( max, 1 ), mean: round( sum / w, 1 ), moon: round( at( moon[ 0 ] / ml, moon[ 1 ] / ml, moon[ 2 ] / ml ), 1 ), from: A.map( ( v ) => round( v, 2 ) ), to: B.map( ( v ) => round( v, 2 ) ) };

}

// the sky of a dcc file: the node, its mesh, material and image, appended where the caller stands in the buffer.
// Returns what scenes[0].extras.baked.sky says, and numbers for the log.
function dccSky( g, o, bake, cells ) {

	const man = bake.man, j = g.json, d = man.data || {};
	const boxes = cells.map( ( c ) => bake.box( c ) );
	const B = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ];
	for ( const b of boxes ) for ( let k = 0; k < 3; k ++ ) { B[ k ] = Math.min( B[ k ], b[ k ] ); B[ k + 3 ] = Math.max( B[ k + 3 ], b[ k + 3 ] ); }
	// the centre of the panorama: eye height at the middle of the plaza; the spawn, or the middle of the content, when
	// the file does not reach there (town x, y, z-up -> x, z, -y)
	const eye = d.spawn && d.spawn.pos ? d.spawn.pos[ 2 ] : B[ 1 ] + 1.65, fr = d.frame || {}, cands = [];
	if ( fr.fountain ) cands.push( [ fr.fountain[ 0 ], eye, - fr.fountain[ 1 ] ] );
	else if ( fr.plaza ) cands.push( [ ( fr.plaza[ 0 ] + fr.plaza[ 2 ] ) / 2, eye, - ( fr.plaza[ 1 ] + fr.plaza[ 3 ] ) / 2 ] );
	if ( d.spawn && d.spawn.pos ) cands.push( [ d.spawn.pos[ 0 ], eye, - d.spawn.pos[ 1 ] ] );
	cands.push( [ ( B[ 0 ] + B[ 3 ] ) / 2, eye, ( B[ 2 ] + B[ 5 ] ) / 2 ] );
	const C = cands.find( ( c ) => c[ 0 ] >= B[ 0 ] && c[ 0 ] <= B[ 3 ] && c[ 2 ] >= B[ 2 ] && c[ 2 ] <= B[ 5 ] ).map( ( v ) => round( v, 3 ) );
	const S = skyFit( boxes, C, SKY_N, SKY_MARGIN, o.skyMin ), M = skyMesh( S, C ), ck = skyCheck( M, C, boxes, o.skySize );
	if ( ck.back || ! ( ck.inside > 0 ) ) die( `internal: the sky's shell: ${ck.back} triangles face outwards, the content is inside by ${ck.inside} m` );
	const moon = ( man.look && man.look.moon_dir ) || [ - 0.45, 0.62, 0.35 ];
	const W = o.skySize, H = W / 2, at0 = g.bufs[ 0 ].length;
	const webp = new Uint8Array( magick( [ '-size', `${W}x${H}`, '-depth', '8', 'rgb:-', ...( o.skyQuality >= 100 ? [ '-define', 'webp:lossless=true', '-quality', '100' ] : [ '-quality', String( o.skyQuality ) ] ), '-define', 'webp:method=6', 'webp:-' ], Buffer.from( skyPanorama( SKY_LOOK, moon, W, SKY_TIME ).buffer ) ) );
	// geometry
	const attributes = {}; let prim;
	if ( o.geometry === 'draco' ) {

		const ext = Math.max( ck.max[ 0 ] - ck.min[ 0 ], ck.max[ 1 ] - ck.min[ 1 ], ck.max[ 2 ] - ck.min[ 2 ] ) * SKY_GRID, bits = Math.max( 1, Math.ceil( Math.log2( ext + 1 ) ) );
		const e = dracoEncode( DRACO.enc, o, { n: M.n, ix: M.ix, pos: M.pos, posQ: { bits, origin: ck.min, range: ( 2 ** bits - 1 ) / SKY_GRID }, uv: M.uv, uvQ: { bits: 15, origin: [ 0, 0 ], range: ( 2 ** 15 - 1 ) / SKY_UV } } );
		const view = g.view( { buffer: 0, byteOffset: g.append( 0, e.bytes ), byteLength: e.bytes.byteLength } );
		attributes.POSITION = g.accessor( { componentType: 5126, count: e.points, type: 'VEC3', min: ck.min, max: ck.max } );
		attributes.TEXCOORD_0 = g.accessor( { componentType: 5126, count: e.points, type: 'VEC2' } );
		prim = { attributes, indices: g.accessor( { componentType: e.points <= 65535 ? 5123 : 5125, count: e.faces * 3, type: 'SCALAR' } ), extensions: { KHR_draco_mesh_compression: { bufferView: view, attributes: e.ids } } };
		g.use( 'KHR_draco_mesh_compression', true );
		M.faces = e.faces;

	} else {

		const vp = g.view( { buffer: 0, byteOffset: g.append( 0, M.pos ), byteLength: M.pos.byteLength, target: 34962 } ), vu = g.view( { buffer: 0, byteOffset: g.append( 0, M.uv ), byteLength: M.uv.byteLength, target: 34962 } );
		const ib = M.n <= 65535 ? new Uint16Array( M.ix ) : M.ix, vi = g.view( { buffer: 0, byteOffset: g.append( 0, ib ), byteLength: ib.byteLength, target: 34963 } );
		attributes.POSITION = g.accessor( { bufferView: vp, componentType: 5126, count: M.n, type: 'VEC3', min: ck.min, max: ck.max } );
		attributes.TEXCOORD_0 = g.accessor( { bufferView: vu, componentType: 5126, count: M.n, type: 'VEC2' } );
		prim = { attributes, indices: g.accessor( { bufferView: vi, componentType: M.n <= 65535 ? 5123 : 5125, count: M.ix.length, type: 'SCALAR' } ) };
		M.faces = M.ix.length / 3;

	}

	const geometry = g.bufs[ 0 ].length - at0;
	// the image: u wraps around, v does not (no nadir in the zenith); no mip levels (it is never seen much smaller)
	j.images = j.images || []; j.textures = j.textures || []; j.samplers = j.samplers || [];
	j.samplers.push( { magFilter: 9729, minFilter: 9729, wrapS: 10497, wrapT: 33071 } );
	j.images.push( { name: 'sky', mimeType: 'image/webp', bufferView: g.view( { buffer: 0, byteOffset: g.append( 0, webp ), byteLength: webp.byteLength } ) } );
	g.use( 'EXT_texture_webp', true );
	j.textures.push( { name: 'sky', sampler: j.samplers.length - 1, extensions: { EXT_texture_webp: { source: j.images.length - 1 } } } );
	g.use( 'KHR_materials_unlit' );
	j.materials.push( { name: 'sky', pbrMetallicRoughness: { baseColorTexture: { index: j.textures.length - 1 }, metallicFactor: 0, roughnessFactor: 1 }, extensions: { KHR_materials_unlit: {} }, extras: { kind: 'sky' } } );
	prim.material = j.materials.length - 1;
	j.meshes.push( { name: 'sky', primitives: [ prim ] } );
	const node = g.node( { name: 'sky', mesh: j.meshes.length - 1, extras: { sky: true } }, true );
	// the walk the parallax is told for: from the spawn to the far side of the plaza
	let slide = null;
	if ( d.spawn && d.spawn.pos && fr.plaza ) {

		const A = [ d.spawn.pos[ 0 ], eye, - d.spawn.pos[ 1 ] ], far = Math.abs( fr.plaza[ 1 ] - d.spawn.pos[ 1 ] ) > Math.abs( fr.plaza[ 3 ] - d.spawn.pos[ 1 ] ) ? fr.plaza[ 1 ] : fr.plaza[ 3 ];
		slide = skyParallax( S, C, A, [ C[ 0 ], eye, - far ], moon );

	}

	const r3 = ( a ) => a.map( ( v ) => round( v, 3 ) );
	return { extras: { node, name: 'sky', material: prim.material, texture: j.textures.length - 1, image: j.images.length - 1, projection: 'equirect', centre: C, yaw: 0, size: [ W, H ],
		uv: 'the direction from centre (x, y, z of the file, y up): u = 0.5 + atan2( z, x ) / ( 2 pi ), v = 0.5 - asin( y ) / pi; three.js\' equirectangular layout, +v down as in glTF; yaw: a turn about +y to add (radians)',
		colour: 'sRGB; the page\'s linear light capped at 1, no tone curve',
		shell: { form: '|X / a|^n + |Z / b|^n + ( Y / c )^n = 1 about `centre`, on the floor Y = 0', n: S.n, centre: r3( S.centre ), a: round( S.a, 3 ), b: round( S.b, 3 ), c: round( S.c, 3 ), margin: S.margin, min: S.min },
		bounds: { content: r3( B ), sky: [ ...ck.min, ...ck.max ] }, ...( slide ? { slide } : {} ), time: SKY_TIME },
		stats: { tris: M.faces, verts: M.n, geometry, image: webp.byteLength, W, H, near: ck.near, swim: ck.swim, inside: ck.inside, content: B, min: ck.min, max: ck.max, C, slide } };

}

// the sky of a file read back: its triangles face the centre the extras name, its UVs are the directions from there,
// nothing of the cells in the file reaches the shell
function verifySky( bake, gl, fail ) {

	const j = gl.json, sky = j.scenes[ 0 ].extras.baked && j.scenes[ 0 ].extras.baked.sky;
	if ( ! sky ) return null;
	const node = j.nodes[ sky.node ];
	if ( ! node || node.name !== 'sky' || ! j.scenes[ 0 ].nodes.includes( sky.node ) ) fail( 'sky: the node the extras name is not `sky` in the scene' );
	for ( const n of j.nodes ) if ( ( n.children || [] ).includes( sky.node ) || ( n.extensions && n.extensions.MSFT_lod && n.extensions.MSFT_lod.ids.includes( sky.node ) ) ) fail( 'sky: it hangs in a hierarchy or a LOD chain' );
	const p = j.meshes[ node.mesh ].primitives[ 0 ], mat = j.materials[ p.material ];
	if ( p.attributes.COLOR_0 !== undefined || ! mat.extensions || ! mat.extensions.KHR_materials_unlit || mat.pbrMetallicRoughness.baseColorTexture.index !== sky.texture || j.textures[ sky.texture ].extensions.EXT_texture_webp.source !== sky.image ) fail( 'sky: not an unlit material with the image the extras name and no COLOR_0' );
	const img = gl.view( j.images[ sky.image ].bufferView );
	if ( Buffer.from( img.subarray( 0, 4 ) ).toString( 'latin1' ) !== 'RIFF' || Buffer.from( img.subarray( 8, 12 ) ).toString( 'latin1' ) !== 'WEBP' ) fail( 'sky: its image is not WebP' );
	const dx = p.extensions && p.extensions.KHR_draco_mesh_compression;
	let M;
	if ( dx ) {

		let dd; try { dd = dracoDecode( DRACO.dec, gl.view( dx.bufferView ), dx.attributes, false ); } catch ( e ) { fail( `sky: ${e.message}` ); }
		if ( j.accessors[ p.attributes.POSITION ].count !== dd.np || j.accessors[ p.indices ].count !== dd.nf * 3 ) fail( 'sky: the accessors do not say what Draco decodes' );
		M = { pos: dd.pos, uv: dd.uv, ix: dd.ix, n: dd.np };

	} else {

		const f32 = ( i ) => { const a = j.accessors[ i ], b = gl.view( a.bufferView ), n = a.type === 'VEC3' ? 3 : 2; return new Float32Array( b.buffer.slice( b.byteOffset + ( a.byteOffset || 0 ), b.byteOffset + ( a.byteOffset || 0 ) + a.count * n * 4 ) ); };
		const ia = j.accessors[ p.indices ], ib = gl.view( ia.bufferView ), raw = ib.buffer.slice( ib.byteOffset + ( ia.byteOffset || 0 ), ib.byteOffset + ( ia.byteOffset || 0 ) + ia.count * ( ia.componentType === 5123 ? 2 : 4 ) );
		const pos = f32( p.attributes.POSITION );
		M = { pos, uv: f32( p.attributes.TEXCOORD_0 ), ix: ia.componentType === 5123 ? new Uint32Array( new Uint16Array( raw ) ) : new Uint32Array( raw ), n: pos.length / 3 };

	}

	const byKey = new Map( bake.man.chunks.map( ( c ) => [ c.k, c ] ) ), keys = new Set();
	for ( const n of j.nodes ) if ( n.extras && Array.isArray( n.extras.baked ) ) for ( const s of n.extras.baked ) keys.add( s.split( ':' )[ 0 ] );
	const boxes = [ ...keys ].map( ( k ) => bake.box( byKey.get( k ) ) );
	const ck = skyCheck( M, sky.centre, boxes, sky.size[ 0 ] );
	const a = j.accessors[ p.attributes.POSITION ];
	for ( let k = 0; k < 3; k ++ ) if ( Math.abs( a.min[ k ] - ck.min[ k ] ) > 1e-3 || Math.abs( a.max[ k ] - ck.max[ k ] ) > 1e-3 ) fail( 'sky: the bounds of its accessor are not those of its vertices' );
	for ( let i = 0; i < M.uv.length; i ++ ) if ( ! ( M.uv[ i ] >= 0 && M.uv[ i ] <= 1 ) ) fail( 'sky: a UV outside 0 .. 1' );
	if ( ck.back ) fail( `sky: ${ck.back} triangles do not face its centre` );
	if ( ! ( ck.inside > 0 ) ) fail( `sky: a cell's box reaches ${( - ck.inside ).toFixed( 2 )} m through the shell` );
	if ( ck.swim > 1 ) fail( `sky: the image lies ${ck.swim.toFixed( 2 )} texels off its direction inside a triangle` );
	return { tris: M.ix.length / 3, swim: ck.swim, inside: ck.inside, near: ck.near };

}

// the spatial units of a file and what each shows at every step of the chain
function dccUnits( o, bake, cells, chain ) {

	const xz = ( c ) => { const m = /^c_(-?\d+)_(-?\d+)$/.exec( c.k ); return m ? [ parseInt( m[ 1 ] ), parseInt( m[ 2 ] ) ] : null; };
	let x0 = Infinity, z0 = Infinity;
	for ( const c of cells ) { const p = xz( c ); if ( p ) { x0 = Math.min( x0, p[ 0 ] ); z0 = Math.min( z0, p[ 1 ] ); } }
	const units = new Map();
	for ( const c of cells ) {

		if ( ! chain.some( ( l ) => c[ l ] && typeof c[ l ] === 'object' ) ) continue;
		const p = xz( c );
		const name = o.grain === 'town' ? 'town' : o.grain === 'cell' ? c.k : p ? `region_${Math.floor( ( p[ 0 ] - x0 ) / o.grainN )}_${Math.floor( ( p[ 1 ] - z0 ) / o.grainN )}` : 'backdrop';
		let u = units.get( name ); if ( ! u ) units.set( name, u = { name, cells: [] } );
		u.cells.push( c );

	}

	for ( const u of units.values() ) {

		// step i: every cell at the first level it has from chain[ i ] on; steps that show the same are one
		u.rungs = [];
		for ( let i = 0; i < chain.length; i ++ ) {

			const pairs = [];
			for ( const c of u.cells ) { const l = chain.slice( i ).find( ( l ) => c[ l ] && typeof c[ l ] === 'object' ); if ( l ) pairs.push( [ c, l ] ); }
			if ( ! pairs.length ) continue;
			const sig = pairs.map( ( [ c, l ] ) => c.k + ':' + l ).join( ' ' ), last = u.rungs[ u.rungs.length - 1 ];
			if ( last && last.sig === sig ) last.level = chain[ i ]; else u.rungs.push( { level: chain[ i ], pairs, sig } );

		}

		u.box = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ];
		for ( const c of u.cells ) { const b = bake.box( c ); for ( let j = 0; j < 3; j ++ ) { u.box[ j ] = Math.min( u.box[ j ], b[ j ] ); u.box[ j + 3 ] = Math.max( u.box[ j + 3 ], b[ j + 3 ] ); } }

	}

	return [ ...units.values() ];

}

// one output file of the dcc profile
function buildDcc( o, bake, tiers, cells, stem ) {

	const D = o.geometry === 'draco' ? DRACO.enc : null;
	const g = new Gltf( 'bake2glb (threejsodango), profile dcc' );
	g.stem = stem;
	const chain = o.lods ? ( o.lods === 'auto' ? autoLevels( tiers[ o.tier ] ).filter( ( l ) => l !== 'ring' ) : o.lods.split( ',' ) ) : DCC_LEVELS.slice( 0, DCC_LEVELS.indexOf( o.level ) + 1 ).reverse();
	for ( const l of chain ) if ( ! DCC_LEVELS.includes( l ) ) die( `the dcc profile has the levels ${DCC_LEVELS.join( ', ' )} (and best), not ${l}` );
	const units = dccUnits( o, bake, cells, chain );
	if ( ! units.length ) return null;
	if ( ! o.lods ) for ( const u of units ) u.rungs.length = 1; // (--level: the finest a cell has, up to that level)
	const all = []; for ( const u of units ) for ( const r of u.rungs ) all.push( ...r.pairs );
	const K = dccGrid( all );
	const st = dccStats( bake, o, all );
	const tex = new DccTextures( g, o, bake, st );
	for ( const layer of st.layers ) tex.measure( layer );
	const man = bake.man, UR = 2 * man.uv_range;
	const P = { K, S: man.colour_scale * o.exposure, gain: ( layer ) => tex.gain.get( layer ) || 1, mean: ( layer ) => tex.mean.has( layer ) ? tex.mean.get( layer ) : 1, glow: o.glow === 'emissive', clip: o.clip, pane: dccPane( man, o.lamps ), cMax: o.color === 'u16' ? 65535 : 255, UR, CA: man.cut_aux || 4,
		noUV: ( layer ) => tex.isBlank( layer ), ground: null };
	// the ground's pictures (--ground): the finest level of the file is cut, while that is one seen from near by
	let gq = null;
	if ( o.ground !== 'none' && man.gao && man.gao.file ) {

		const mode = o.ground || 'fold', sh = mode !== 'ao' && !! ( man.gsh && man.gsh.file );
		if ( mode === 'shadows' && ! sh ) die( '--ground shadows: this bake has its shadows in the vertices (no gsh)' );
		if ( mode !== 'shadows' || sh ) {

			gq = { ...GROUND, ao: mode !== 'shadows', shadows: sh, edge: o.groundEdge, tol: o.groundTol, cut: o.groundEdge > 0 && ( chain[ 0 ] === 'fine' || chain[ 0 ] === 'near' ) ? [ chain[ 0 ] ] : [], blur: { near: 0.25, mid: 0.5, lo: 1 }, lattice: 0,
				maps: [ man.gao.v || 0, sh ? man.gsh.v || 0 : 0 ] };
			for ( const [ c, l ] of all ) if ( gq.cut.includes( l ) ) { gq.lattice = gcd( gq.lattice, Math.round( c[ l ].q * K ) ); for ( const v of c[ l ].min ) gq.lattice = gcd( gq.lattice, Math.round( v * K ) ); }
			gq.lattice = gq.lattice || 1;
			P.ground = dccGround( bake, gq, K );

		}

	} else if ( o.ground && o.ground !== 'none' ) die( `--ground ${o.ground}: this bake has no ground pictures (gao)` );
	const stats = { cells: 0, levels: 0, tris: 0, verts: 0, mixed: 0, over: 0, channels: 0, max: st.max, chains: 0, degenerate: 0, prims: 0, floatPos: 0, bakeTris: 0, geometry: 0, panes: 0, added: 0, globes: 0, ground: gq ? { ...gq, tris: 0, verts: 0, split: 0, more: 0 } : null };
	stats.cells = new Set( all.map( ( [ c ] ) => c.k ) ).size;
	g.buffer( 'main', () => ( {} ) );
	const expect = new Map(); // node -> [ sum, sum, triangles ]
	const cType = o.color === 'u16' ? 5123 : 5121;

	const emit = ( u, ri ) => {

		const rung = u.rungs[ ri ], at0 = g.bufs[ 0 ].length;
		const { accs, degenerate, mixed, over, channels, panes, added, globes, ground } = dccCollect( bake, P, rung.pairs );
		if ( stats.ground ) { stats.ground.tris += ground.tris; stats.ground.verts += ground.verts; stats.ground.split += ground.cut; stats.ground.more += ground.added; }
		stats.degenerate += degenerate; stats.mixed += mixed; stats.over += over; stats.channels += channels; stats.levels ++; stats.panes += panes; stats.added += added; stats.globes += globes;
		for ( const [ c, l ] of rung.pairs ) stats.bakeTris += c[ l ].ni / 3;
		stats.bakeTris += added;
		const keys = [ ...accs.keys() ].sort( ( a, b ) => a - b );
		const sum = [ 0, 0, 0 ];
		// where the node stands: the middle of the unit's footprint, at its foot, on the grid (the town as one object: the origin)
		const oi = o.grain === 'town' ? [ 0, 0, 0 ] : [ Math.round( ( u.box[ 0 ] + u.box[ 3 ] ) / 2 * K ), Math.round( u.box[ 1 ] * K ), Math.round( ( u.box[ 2 ] + u.box[ 5 ] ) / 2 * K ) ];
		const node = { name: ri === 0 ? u.name : `${u.name}_${rung.level}` };
		const primitives = [];
		// --geometry quant: u16 positions for the whole object while it fits without loss, floats otherwise
		let quant = null;
		if ( o.geometry === 'quant' ) {

			let step = 0; const mn = [ Infinity, Infinity, Infinity ], mx = [ - Infinity, - Infinity, - Infinity ];
			for ( const k of keys ) { const a = accs.get( k ), pos = a.pos.get(); step = gcd( step, a.step ); for ( let i = 0; i < pos.length; i ++ ) { const j = i % 3; if ( pos[ i ] < mn[ j ] ) mn[ j ] = pos[ i ]; if ( pos[ i ] > mx[ j ] ) mx[ j ] = pos[ i ]; } }
			step = step || 1;
			if ( Math.max( mx[ 0 ] - mn[ 0 ], mx[ 1 ] - mn[ 1 ], mx[ 2 ] - mn[ 2 ] ) / step <= 65535 ) quant = { step, mn }; else stats.floatPos ++;

		}

		const views = { P: [], UVQ: [], UVF: [], C: [], IX: [] }, len = { P: 0, UVQ: 0, UVF: 0, C: 0, IX: 0 };
		const push = ( k, a ) => { const at = len[ k ]; views[ k ].push( a ); len[ k ] += a.byteLength; if ( len[ k ] % 4 ) { const pad = 4 - len[ k ] % 4; views[ k ].push( new Uint8Array( pad ) ); len[ k ] += pad; } return at; };
		const plain = [];
		for ( const key of keys ) {

			const a = accs.get( key ), n = a.n, pos = a.pos.get(), uvi = a.uv ? a.uv.get() : null, col = a.col.get(), ix = a.ix.get();
			dccHash( sum, pos, uvi, col, ix, ix.length );
			const alpha = a.kind === K_GLASS || a.kind === K_FALL, cut = a.kind === K_CUT;
			const material = dccMaterial( g, o, bake, tex, st, a.kind, a.layer, a.glow );
			const mn = [ Infinity, Infinity, Infinity ], mx = [ - Infinity, - Infinity, - Infinity ];
			// (--pos-drop / --uv-drop: onto the coarser grid first, in whole numbers and from the town's origin - a vertex two objects share lands on the same point in both)
			if ( D && o.posDrop ) { const gq = ( a.step || 1 ) * 2 ** o.posDrop; for ( let i = 0; i < n * 3; i ++ ) pos[ i ] = Math.floor( ( pos[ i ] + gq / 2 ) / gq ) * gq; }
			if ( D && o.uvDrop && uvi ) { const h = 2 ** o.uvDrop; for ( let i = 0; i < n * 2; i ++ ) uvi[ i ] = Math.floor( ( uvi[ i ] + h / 2 ) / h ) * h; }
			for ( let i = 0; i < n * 3; i ++ ) { const j = i % 3, v = pos[ i ]; if ( v < mn[ j ] ) mn[ j ] = v; if ( v > mx[ j ] ) mx[ j ] = v; }
			const uvStep = cut ? 1 / CUT_UV_GRID : UR / 65536, uvOff = cut ? 0 : - UR / 2;
			stats.tris += ix.length / 3; stats.verts += n; stats.prims ++;
			if ( D ) {

				const step = ( a.step || 1 ) * 2 ** o.posDrop;
				const p3 = new Float32Array( n * 3 ); for ( let i = 0; i < n * 3; i ++ ) p3[ i ] = ( pos[ i ] - oi[ i % 3 ] ) / K;
				const ext = Math.max( mx[ 0 ] - mn[ 0 ], mx[ 1 ] - mn[ 1 ], mx[ 2 ] - mn[ 2 ] ) / step;
				const bits = Math.max( 1, Math.ceil( Math.log2( ext + 1 ) ) );
				if ( bits > 30 ) die( `internal: ${u.name} spans ${ext} grid steps, more than Draco quantises` );
				const p = { n, ix, pos: p3, posQ: { bits, origin: [ 0, 1, 2 ].map( ( j ) => ( mn[ j ] - oi[ j ] ) / K ), range: ( 2 ** bits - 1 ) * step / K } };
				if ( uvi ) {

					let u0 = Infinity, v0 = Infinity, u1 = - Infinity, v1 = - Infinity;
					for ( let i = 0; i < n; i ++ ) { const x = uvi[ i * 2 ], y = uvi[ i * 2 + 1 ]; if ( x < u0 ) u0 = x; if ( x > u1 ) u1 = x; if ( y < v0 ) v0 = y; if ( y > v1 ) v1 = y; }
					const uh = 2 ** o.uvDrop, ub = Math.max( 1, Math.ceil( Math.log2( Math.max( u1 - u0, v1 - v0 ) / uh + 1 ) ) );
					p.uv = new Float32Array( n * 2 ); for ( let i = 0; i < n * 2; i ++ ) p.uv[ i ] = uvi[ i ] * uvStep + uvOff;
					p.uvQ = { bits: ub, origin: [ u0 * uvStep + uvOff, v0 * uvStep + uvOff ], range: ( 2 ** ub - 1 ) * uvStep * uh };

				}

				// (three colour channels where the alpha says nothing)
				if ( alpha ) { p.col = col; p.nc = 4; } else { p.nc = 3; p.col = new col.constructor( n * 3 ); for ( let i = 0; i < n; i ++ ) { p.col[ i * 3 ] = col[ i * 4 ]; p.col[ i * 3 + 1 ] = col[ i * 4 + 1 ]; p.col[ i * 3 + 2 ] = col[ i * 4 + 2 ]; } }
				const e = dracoEncode( D, o, p );
				stats.degenerate += ix.length / 3 - e.faces; stats.tris -= ix.length / 3 - e.faces; // (triangles with two corners in one place: Draco leaves them out)
				const view = g.view( { buffer: 0, byteOffset: g.append( 0, e.bytes ), byteLength: e.bytes.byteLength } );
				const attributes = { POSITION: g.accessor( { componentType: 5126, count: e.points, type: 'VEC3', min: [ 0, 1, 2 ].map( ( j ) => ( mn[ j ] - oi[ j ] ) / K ), max: [ 0, 1, 2 ].map( ( j ) => ( mx[ j ] - oi[ j ] ) / K ) } ) };
				if ( uvi ) attributes.TEXCOORD_0 = g.accessor( { componentType: 5126, count: e.points, type: 'VEC2' } );
				attributes.COLOR_0 = g.accessor( { componentType: cType, normalized: true, count: e.points, type: p.nc === 3 ? 'VEC3' : 'VEC4' } );
				primitives.push( { attributes, indices: g.accessor( { componentType: e.points <= 65535 ? 5123 : 5125, count: e.faces * 3, type: 'SCALAR' } ), material,
					extensions: { KHR_draco_mesh_compression: { bufferView: view, attributes: e.ids } } } );

			} else {

				const q = { n, ni: ix.length, material, hasUV: !! uvi };
				if ( quant ) {

					const p4 = new Uint16Array( n * 4 ); for ( let i = 0; i < n; i ++ ) for ( let j = 0; j < 3; j ++ ) p4[ i * 4 + j ] = ( pos[ i * 3 + j ] - quant.mn[ j ] ) / quant.step;
					q.pos = push( 'P', p4 ); q.mn = mn.map( ( v, j ) => ( v - quant.mn[ j ] ) / quant.step ); q.mx = mx.map( ( v, j ) => ( v - quant.mn[ j ] ) / quant.step );

				} else {

					const p3 = new Float32Array( n * 3 ); for ( let i = 0; i < n * 3; i ++ ) p3[ i ] = ( pos[ i ] - oi[ i % 3 ] ) / K;
					q.pos = push( 'P', p3 ); q.mn = mn.map( ( v, j ) => ( v - oi[ j ] ) / K ); q.mx = mx.map( ( v, j ) => ( v - oi[ j ] ) / K );

				}

				if ( uvi ) {

					q.uvq = o.geometry === 'quant' && ! cut;
					if ( q.uvq ) { const a2 = new Uint16Array( n * 2 ); a2.set( uvi ); q.uv = push( 'UVQ', a2 ); }
					else { const a2 = new Float32Array( n * 2 ); for ( let i = 0; i < n * 2; i ++ ) a2[ i ] = uvi[ i ] * uvStep + uvOff; q.uv = push( 'UVF', a2 ); }

				}

				q.col = push( 'C', col.slice() );
				q.it = n <= 65535 ? 5123 : 5125;
				q.ix = push( 'IX', q.it === 5123 ? new Uint16Array( ix ) : ix.slice() );
				plain.push( q );

			}

		}

		if ( ! D ) {

			const join = ( list, total ) => { const a = new Uint8Array( total ); let at = 0; for ( const p of list ) { a.set( new Uint8Array( p.buffer, p.byteOffset, p.byteLength ), at ); at += p.byteLength; } return a; };
			const mk = ( k, stride ) => ( len[ k ] ? g.view( { buffer: 0, byteOffset: g.append( 0, join( views[ k ], len[ k ] ) ), byteLength: len[ k ], ...( stride ? { byteStride: stride, target: 34962 } : { target: 34963 } ) } ) : - 1 );
			const cb = o.color === 'u16' ? 8 : 4;
			const vP = mk( 'P', quant ? 8 : 12 ), vUQ = mk( 'UVQ', 4 ), vUF = mk( 'UVF', 8 ), vC = mk( 'C', cb ), vIX = mk( 'IX', 0 );
			if ( o.geometry === 'quant' ) g.use( 'KHR_mesh_quantization', true );
			for ( const q of plain ) {

				const attributes = { POSITION: g.accessor( { bufferView: vP, byteOffset: q.pos, componentType: quant ? 5123 : 5126, count: q.n, type: 'VEC3', min: q.mn, max: q.mx } ) };
				if ( q.hasUV ) attributes.TEXCOORD_0 = g.accessor( q.uvq ? { bufferView: vUQ, byteOffset: q.uv, componentType: 5123, normalized: true, count: q.n, type: 'VEC2' } : { bufferView: vUF, byteOffset: q.uv, componentType: 5126, count: q.n, type: 'VEC2' } );
				attributes.COLOR_0 = g.accessor( { bufferView: vC, byteOffset: q.col, componentType: cType, normalized: true, count: q.n, type: 'VEC4' } );
				primitives.push( { attributes, indices: g.accessor( { bufferView: vIX, byteOffset: q.ix, componentType: q.it, count: q.ni, type: 'SCALAR' } ), material: q.material } );

			}

		} else g.use( 'KHR_draco_mesh_compression', true );

		if ( quant ) { node.translation = quant.mn.map( ( v ) => v / K ); node.scale = [ quant.step / K, quant.step / K, quant.step / K ]; }
		else if ( oi[ 0 ] || oi[ 1 ] || oi[ 2 ] ) node.translation = oi.map( ( v ) => v / K );
		g.json.meshes.push( { name: node.name, primitives } );
		node.mesh = g.json.meshes.length - 1;
		node.extras = { baked: rung.pairs.map( ( [ c, l ] ) => c.k + ':' + l ) };
		const ni = g.node( node );
		expect.set( ni, sum );
		stats.geometry += g.bufs[ 0 ].length - at0;
		rung.node = ni; rung.bytes = g.bufs[ 0 ].length - at0;
		return ni;

	};

	// the data in the order a viewer needs it: every unit's coarsest level, the textures, then level by level to the finest
	for ( const u of units ) emit( u, u.rungs.length - 1 );
	tex.flush();
	const sky = o.sky === 'mesh' ? dccSky( g, o, bake, [ ...new Set( all.map( ( [ c ] ) => c ) ) ] ) : null; // (with the first picture)
	const front = g.bufs[ 0 ].length;
	for ( const level of DCC_LEVELS ) {

		for ( const u of units ) for ( let ri = u.rungs.length - 2; ri >= 0; ri -- ) if ( u.rungs[ ri ].level === level ) emit( u, ri );
		tex.flush();

	}

	const parents = new Map();
	const perLevel = {};
	for ( const u of units ) {

		const base = g.json.nodes[ u.rungs[ 0 ].node ];
		g.json.scenes[ 0 ].nodes.push( u.rungs[ 0 ].node );
		u.rungs.forEach( ( r, i ) => { const k = i === 0 ? 'shown' : r.level; perLevel[ k ] = ( perLevel[ k ] || 0 ) + r.bytes; } );
		if ( u.rungs.length < 2 ) continue;
		const levels = u.rungs.map( ( r ) => r.level ), plan = lodPlanBox( o, tiers, u.box, levels );
		g.use( 'MSFT_lod' );
		base.extensions = { MSFT_lod: { ids: u.rungs.slice( 1 ).map( ( r ) => r.node ) } };
		base.extras = { ...base.extras, MSFT_screencoverage: plan.coverage, lod: plan };
		stats.chains ++;
		// the coarser levels: under one node per level, in no scene
		for ( const r of u.rungs.slice( 1 ) ) { if ( ! parents.has( r.level ) ) parents.set( r.level, [] ); parents.get( r.level ).push( r.node ); }

	}

	for ( const level of chain ) if ( parents.has( level ) ) g.node( { name: `LOD${chain.indexOf( level )}_${level}`, children: parents.get( level ) } );
	stats.lights = lightsFor( g, o, bake, cells );
	const gains = {}; for ( const [ layer, G ] of tex.gain ) if ( G !== 1 ) gains[ layer ] = G;
	const texel = {}; for ( const [ layer, v ] of tex.mean ) if ( v !== 1 ) texel[ layer ] = v;
	g.json.scenes[ 0 ].extras = { baked: { baked: man.baked, spawn: man.data && man.data.spawn, ...( sky ? { sky: sky.extras } : {} ) },
		export: { tool: 'bake2glb', mode: 'dcc', levels: chain, grain: o.grain, geometry: o.geometry, color: o.color, exposure: o.exposure, glow: o.glow, clip: o.clip, lamps: o.lamps, grid: K, cut_uv_grid: CUT_UV_GRID, gain: gains, texel, front, ...( gq ? { ground: gq } : {} ),
			light: 'shown = texture * COLOR_0 = the baked light * exposure on the bake\'s texture (gain: how much brighter a layer\'s image is than the bake\'s, its COLOR_0 that much darker); light above 1 after the gain: ' + ( o.clip === 'channel' ? 'each channel clamped at 1' : o.clip === 'hue' ? 'divided by its largest channel' : 'divided by its largest channel m, then mixed towards white by 1 - 1 / sqrt( max( 1, m * texel ) ), texel = the mean of the layer\'s image' ) + ( o.clip === 'channel' ? '' : '; blended glass first moves such light into its alpha (colour * alpha kept)' ),
			fall: 'falling water: the mean of the page\'s animated sheet (its noise averaged out): alpha = baked opacity * mean cover, colour = baked light * mean share lit, by the two masks in the vertex\'s layer slot',
			...( o.lods ? { tier: o.tier, tier_table: tiers[ o.tier ], fov: FOV, aspect: ASPECT } : {} ) } };
	fs.mkdirSync( o.out, { recursive: true } );
	const files = g.write( o.out, stem, 'glb' );
	stats.textures = tex.bytes; stats.images = tex.info; stats.front = front; stats.perLevel = perLevel; stats.share = tex.share; stats.over0 = st.over / Math.max( 1, st.verts ); stats.sky = sky && sky.stats;
	stats.json = files[ 0 ].bytes - g.bufs[ 0 ].length - 28 - ( 4 - g.bufs[ 0 ].length % 4 ) % 4;
	stats.cutoff = [ ...st.cut ].map( ( l ) => [ man.layers[ l ], tex.cutoff.get( l ) ] );
	return { files, stats, written: [], main: files[ 0 ].file, json: g.json, expect };

}

// a dcc file read back by the glTF rules against the bake: every object's triangles (positions, UVs, colours, as whole
// numbers) are those dccCollect makes of the cells and levels the node names
function verifyDcc( bake, file, expect ) {

	const gl = readGltf( file ), j = gl.json, ex = j.scenes[ 0 ].extras.export;
	const fail = ( msg ) => { console.error( `bake2glb: VERIFY FAILED: ${path.basename( file )}: ${msg}` ); process.exit( 2 ); };
	const man = bake.man, K = ex.grid, UR = 2 * man.uv_range, cMax = ex.color === 'u16' ? 65535 : 255;
	const blank = ( layer ) => man.layers[ layer ] === 'white';
	const P = { K, S: man.colour_scale * ex.exposure, gain: ( layer ) => ( ex.gain && ex.gain[ layer ] ) || 1, mean: ( layer ) => ( ex.texel && ex.texel[ layer ] !== undefined ? ex.texel[ layer ] : 1 ), glow: ex.glow === 'emissive', clip: ex.clip || 'channel', pane: dccPane( man, ex.lamps || 'baked' ), cMax, UR, CA: man.cut_aux || 4, noUV: blank, ground: ex.ground ? dccGround( bake, ex.ground, K ) : null };
	const byKey = new Map( man.chunks.map( ( c ) => [ c.k, c ] ) );
	const acc = ( i ) => {

		const a = j.accessors[ i ], n = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[ a.type ], T = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }[ a.componentType ];
		const bv = j.bufferViews[ a.bufferView ], bytes = gl.view( a.bufferView ), stride = bv.byteStride || n * T.BYTES_PER_ELEMENT, out = new T( a.count * n );
		const dv = new DataView( bytes.buffer, bytes.byteOffset + ( a.byteOffset || 0 ) );
		const get = T === Float32Array ? ( o ) => dv.getFloat32( o, true ) : T === Uint16Array ? ( o ) => dv.getUint16( o, true ) : T === Uint32Array ? ( o ) => dv.getUint32( o, true ) : ( o ) => dv.getUint8( o );
		for ( let i = 0; i < a.count; i ++ ) for ( let k = 0; k < n; k ++ ) out[ i * n + k ] = get( i * stride + k * T.BYTES_PER_ELEMENT );
		return { a: out, n };

	};

	let nodes = 0, tris = 0, bakeTris = 0, prims = 0, added = 0, cut = 0;
	j.nodes.forEach( ( node, ni ) => {

		if ( node.mesh === undefined || ! node.extras || ! node.extras.baked ) return;
		const pairs = node.extras.baked.map( ( s ) => { const [ k, l ] = s.split( ':' ); const c = byKey.get( k ); if ( ! c || ! c[ l ] ) fail( `${node.name}: the bake has no ${s}` ); return [ c, l ]; } );
		const want = dccCollect( bake, P, pairs ), wsum = [ 0, 0, 0 ];
		for ( const a of want.accs.values() ) dccHash( wsum, a.pos.get(), a.uv ? a.uv.get() : null, a.col.get(), a.ix.get(), a.ix.n );
		for ( const [ c, l ] of pairs ) bakeTris += c[ l ].ni / 3;
		bakeTris += want.added; added += want.added; cut += want.ground.added;
		const T = node.translation || [ 0, 0, 0 ], Sc = node.scale || [ 1, 1, 1 ];
		const sum = [ 0, 0, 0 ];
		const box = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ];
		for ( const [ c, l ] of pairs ) for ( let k = 0; k < 3; k ++ ) { box[ k ] = Math.min( box[ k ], c[ l ].bmin[ k ] ); box[ k + 3 ] = Math.max( box[ k + 3 ], c[ l ].bmax[ k ] ); }
		for ( const p of j.meshes[ node.mesh ].primitives ) {

			const mat = j.materials[ p.material ], cut = mat.extras.kind === 'cut';
			const dx = p.extensions && p.extensions.KHR_draco_mesh_compression;
			let pos, uv = null, col, ix, uvNorm = false, np;
			if ( dx ) {

				let d; try { d = dracoDecode( DRACO.dec, gl.view( dx.bufferView ), dx.attributes, cMax === 65535 ); } catch ( e ) { fail( `${node.name} / ${mat.name}: ${e.message}` ); }
				pos = d.pos; uv = d.uv; col = d.col; ix = d.ix; np = d.np;
				if ( j.accessors[ p.attributes.POSITION ].count !== d.np || j.accessors[ p.indices ].count !== d.nf * 3 ) fail( `${node.name} / ${mat.name}: the accessors say ${j.accessors[ p.attributes.POSITION ].count} vertices, ${j.accessors[ p.indices ].count} indices; Draco decodes ${d.np}, ${d.nf * 3}` );

			} else {

				pos = acc( p.attributes.POSITION ).a; np = pos.length / 3;
				if ( p.attributes.TEXCOORD_0 !== undefined ) { uv = acc( p.attributes.TEXCOORD_0 ).a; uvNorm = !! j.accessors[ p.attributes.TEXCOORD_0 ].normalized; }
				col = acc( p.attributes.COLOR_0 ); ix = acc( p.indices ).a;

			}

			const pi = new Int32Array( np * 3 );
			for ( let i = 0; i < np * 3; i ++ ) pi[ i ] = Math.round( ( pos[ i ] * Sc[ i % 3 ] + T[ i % 3 ] ) * K );
			const a = j.accessors[ p.attributes.POSITION ];
			for ( let k = 0; k < 3; k ++ ) if ( a.min[ k ] * Sc[ k ] + T[ k ] < box[ k ] - 2e-3 || a.max[ k ] * Sc[ k ] + T[ k ] > box[ k + 3 ] + 2e-3 ) fail( `${node.name} / ${mat.name}: its bounds leave the bake's box` );
			let ui = null;
			if ( uv ) { ui = new Int32Array( np * 2 ); const s = cut ? CUT_UV_GRID : 65536 / UR, off = cut ? 0 : UR / 2; for ( let i = 0; i < np * 2; i ++ ) ui[ i ] = uvNorm ? uv[ i ] : Math.round( ( uv[ i ] + off ) * s ); }
			let c4 = col.a;
			if ( col.n === 3 ) { c4 = new col.a.constructor( np * 4 ); for ( let i = 0; i < np; i ++ ) { c4[ i * 4 ] = col.a[ i * 3 ]; c4[ i * 4 + 1 ] = col.a[ i * 3 + 1 ]; c4[ i * 4 + 2 ] = col.a[ i * 3 + 2 ]; c4[ i * 4 + 3 ] = cMax; } }
			dccHash( sum, pi, ui, c4, ix, ix.length );
			prims ++;

		}

		if ( sum[ 2 ] !== wsum[ 2 ] ) fail( `${node.name}: ${sum[ 2 ]} triangles, the bake draws ${wsum[ 2 ]}` );
		if ( sum[ 0 ] !== wsum[ 0 ] || sum[ 1 ] !== wsum[ 1 ] ) fail( `${node.name}: its triangles are not the bake's (positions, UVs or colours differ)` );
		if ( expect && expect.has( ni ) ) { const e = expect.get( ni ); if ( e[ 0 ] !== sum[ 0 ] || e[ 1 ] !== sum[ 1 ] || e[ 2 ] !== sum[ 2 ] ) fail( `${node.name}: not what was written` ); }
		nodes ++; tris += sum[ 2 ];

	} );
	return { nodes, tris, degenerate: bakeTris - tris, prims, added, cut, ground: !! ex.ground, sky: verifySky( bake, gl, fail ) };

}

// ------------------------------------------------------------------------------------------------ live

const LIVE_V = 1; // of what web/engine/lite_glb.js reads (manifest.glb.v and scenes[0].extras.export.live)
const liveName = ( f ) => f.replace( /\.bin$/, '' ) + '.glb';

// the head of a .glb without its binary chunk: null when it is none, or cut short
function glbHead( file ) {

	let fd;
	try { fd = fs.openSync( file, 'r' ); } catch ( e ) { return null; }
	try {

		const h = Buffer.alloc( 20 ), st = fs.fstatSync( fd );
		if ( fs.readSync( fd, h, 0, 20, 0 ) < 20 || h.readUInt32LE( 0 ) !== 0x46546C67 || h.readUInt32LE( 16 ) !== 0x4E4F534A ) return null;
		const total = h.readUInt32LE( 8 ), n = h.readUInt32LE( 12 );
		if ( st.size !== total ) return null;
		const js = Buffer.alloc( n );
		if ( fs.readSync( fd, js, 0, n, 20 ) < n ) return null;
		return { total, jsonBytes: n, json: JSON.parse( js.toString( 'utf8' ) ), mtimeMs: st.mtimeMs };

	} catch ( e ) { return null; } finally { fs.closeSync( fd ); }

}

// what the page's reader relies on, beyond verifyExact: the stamp, every level there once, and in the binary chunk the
// two streams of a level one after the other, byte for byte the packed file's
function verifyLive( bake, file, cells, level ) {

	const fail = ( msg ) => { console.error( `bake2glb: VERIFY FAILED: ${path.basename( file )}: ${msg}` ); process.exit( 2 ); };
	const v = verifyExact( bake, file, cells );
	const raw = fs.readFileSync( file ), n = raw.readUInt32LE( 12 ), j = JSON.parse( raw.subarray( 20, 20 + n ).toString( 'utf8' ) ), binAt = 20 + n + 8;
	if ( raw.readUInt32LE( 8 ) !== raw.length || raw.readUInt32LE( 20 + n + 4 ) !== 0x004E4942 ) fail( 'not a .glb of a JSON and a BIN chunk' );
	const ex = j.scenes[ 0 ].extras;
	if ( ex.baked.baked !== bake.man.baked || ex.export.live !== LIVE_V ) fail( `stamp ${ex.baked.baked} / format ${ex.export.live}, the bake is ${bake.man.baked} / ${LIVE_V}` );
	const seen = new Set();
	let at = 0;
	for ( const node of j.nodes ) {

		const b = node.extras && node.extras.baked;
		if ( ! b || node.mesh === undefined ) continue;
		if ( b.level !== level || seen.has( b.cell ) ) fail( `${node.name}: twice, or not level ${level}` );
		seen.add( b.cell );
		const c = cells.find( ( x ) => x.k === b.cell ), m = c[ level ], src = bake.blob( c, level );
		const p = j.meshes[ node.mesh ].primitives[ 0 ];
		const xs = [ j.accessors[ p.attributes.POSITION ].bufferView, j.accessors[ p.indices ].bufferView ].map( ( i ) => { const e = j.bufferViews[ i ].extensions || {}; return e.KHR_meshopt_compression || e.EXT_meshopt_compression; } );
		if ( ! xs[ 0 ] || ! xs[ 1 ] || xs[ 0 ].buffer !== 0 || xs[ 1 ].buffer !== 0 ) fail( `${node.name}: its views are not meshopt streams of the binary chunk` );
		if ( xs[ 0 ].byteLength !== m.vlen || xs[ 0 ].byteOffset + m.vlen !== xs[ 1 ].byteOffset || m.vlen + xs[ 1 ].byteLength !== m.len || src.length !== m.len ) fail( `${node.name}: streams of ${xs[ 0 ].byteLength} + ${xs[ 1 ].byteLength} bytes at ${xs[ 0 ].byteOffset} / ${xs[ 1 ].byteOffset}, the bake has ${m.vlen} + ${m.len - m.vlen}` );
		if ( xs[ 0 ].byteOffset !== at ) fail( `${node.name}: its streams start at ${xs[ 0 ].byteOffset}, not where the cell before ends (${at})` );
		at += m.len;
		if ( Buffer.compare( raw.subarray( binAt + xs[ 0 ].byteOffset, binAt + xs[ 0 ].byteOffset + m.len ), src ) !== 0 ) fail( `${node.name}: the streams differ from the packed file's` );

	}

	if ( seen.size !== cells.length || cells.some( ( c ) => ! seen.has( c.k ) ) ) fail( `${seen.size} cells, the bake has ${cells.length}` );
	return v;

}

// where the value of a top-level key of a JSON text lies: [ start (with its comma), end ) - to change one key and
// leave every other byte of a manifest as its writer made it
function topKeySpan( text, key ) {

	let depth = 0, i = 0, comma = - 1;
	while ( i < text.length ) {

		const ch = text[ i ];
		if ( ch === '"' ) {

			let e = i + 1;
			while ( text[ e ] !== '"' ) e += text[ e ] === '\\' ? 2 : 1;
			if ( depth === 1 && text.slice( i + 1, e ) === key && /^\s*:/.test( text.slice( e + 1, e + 8 ) ) ) {

				// the value: to the comma or brace that closes it at this depth
				let k = text.indexOf( ':', e ) + 1, d = 0;
				for ( ; k < text.length; k ++ ) {

					const c2 = text[ k ];
					if ( c2 === '"' ) { k ++; while ( text[ k ] !== '"' ) k += text[ k ] === '\\' ? 2 : 1; }
					else if ( c2 === '{' || c2 === '[' ) d ++;
					else if ( c2 === '}' || c2 === ']' ) { if ( d === 0 ) break; d --; }
					else if ( c2 === ',' && d === 0 ) break;

				}

				return comma >= 0 ? [ comma, k ] : [ i, text[ k ] === ',' ? k + 1 : k ];

			}

			i = e + 1; continue;

		}

		if ( ch === '{' || ch === '[' ) depth ++;
		else if ( ch === '}' || ch === ']' ) depth --;
		if ( depth === 1 && ch === ',' ) comma = i; else if ( ch !== ' ' && ch !== '\n' && ch !== '\t' && ch !== '\r' ) comma = - 1;
		i ++;

	}

	return null;

}

function manifestGlb( dir, rec ) {

	const f = path.join( dir, 'manifest.json' ), raw = fs.readFileSync( f, 'utf8' ), man = JSON.parse( raw );
	if ( JSON.stringify( man.glb ) === JSON.stringify( rec ) ) return false;
	let text = raw;
	const span = topKeySpan( text, 'glb' );
	if ( span ) text = text.slice( 0, span[ 0 ] ) + text.slice( span[ 1 ] );
	const end = text.lastIndexOf( '}' );
	text = text.slice( 0, end ) + ',"glb":' + JSON.stringify( rec ) + text.slice( end );
	const back = JSON.parse( text ), { glb, ...a } = back, { glb: old, ...b } = man;
	if ( JSON.stringify( a ) !== JSON.stringify( b ) || JSON.stringify( glb ) !== JSON.stringify( rec ) ) die( `internal: could not add "glb" to ${f}` );
	fs.writeFileSync( f + '.glbtmp', text );
	fs.renameSync( f + '.glbtmp', f );
	return true;

}

function liveExport( o, dir ) {

	const bake = new Bake( dir ), man = bake.man, t0 = Date.now();
	const say = ( ...a ) => { if ( ! o.quiet ) console.log( ...a ); };
	const rel = path.relative( ROOT, dir ) || '.';
	const oo = { ...o, mode: 'exact', lods: null, out: dir, container: 'glb', buffers: 'one', order: 'cell', meshopt: 'auto', extras: 'live', lights: 'none', split: 'town', quiet: true };
	if ( man.chunks.some( ( c ) => ! c.lo || typeof c.lo !== 'object' || c.lo.file || c.lo.off === undefined ) ) die( `${rel}: not every cell has its lowest level in ${man.base.file}` );
	const jobs = [ { file: liveName( man.base.file ), level: 'lo', cells: man.chunks, src: man.base.file, bytes: man.base.bytes } ];
	for ( const c of man.chunks ) for ( const l of LEVELS ) { const m = c[ l ]; if ( l !== 'lo' && m && typeof m === 'object' && m.file ) jobs.push( { file: liveName( m.file ), level: l, cells: [ c ], src: m.file, bytes: m.len } ); }
	const names = new Set( jobs.map( ( j ) => j.file ) );
	if ( names.size !== jobs.length ) die( `${rel}: two levels of the manifest share a file` );
	// (what a run that was stopped left behind)
	for ( const fn of fs.readdirSync( dir ) ) if ( /^\.bake2glb_.*\.glb$/.test( fn ) ) fs.unlinkSync( path.join( dir, fn ) );
	let written = 0, kept = 0, total = 0, packed = 0, levels = 0, base = null, gone = 0;
	for ( const job of jobs ) {

		const dst = path.join( dir, job.file ), srcFile = path.join( dir, job.src );
		// (st null: the packed file was dropped, --drop-packed - its .glb then has to be this bake's)
		let st = null;
		try { st = fs.statSync( srcFile ); } catch ( e ) { gone ++; }
		if ( st && st.size !== job.bytes ) die( `${rel}: ${job.src} has ${st.size} bytes, the manifest says ${job.bytes}` );
		let head = o.force && st ? null : glbHead( dst );
		const ex = head && head.json.scenes && head.json.scenes[ 0 ] && head.json.scenes[ 0 ].extras;
		const fresh = !! ( ex && ex.export && ex.baked && ex.export.live === LIVE_V && ex.baked.baked === man.baked && ex.export.src && ex.export.src.file === job.src && ex.export.src.bytes === job.bytes && ( ! st || head.mtimeMs >= st.mtimeMs ) );
		if ( ! st && ! fresh ) die( `${rel}: ${job.src} is missing, and ${job.file} is not this bake's` );
		if ( fresh ) kept ++;
		else {

			const tmp = '.bake2glb_' + job.file.replace( /\.glb$/, '' );
			let res;
			try { res = build( { ...oo, level: job.level, liveSrc: { file: job.src, bytes: job.bytes } }, bake, {}, job.cells, tmp ); } catch ( e ) { die( `${rel}: ${job.src} does not decode (${e.message || e}): ${job.file} not written` ); }
			if ( ! res ) die( `${rel}: nothing to write for ${job.file}` );
			fs.renameSync( path.join( dir, tmp + '.glb' ), dst );
			head = glbHead( dst );
			written ++;

		}

		if ( o.verify ) levels += verifyLive( bake, dst, job.cells, job.level ).levels;
		total += head.total; packed += job.bytes;
		if ( job.level === 'lo' ) base = { file: job.file, bytes: head.total, json: head.jsonBytes };

	}

	// .glb files of this mode that stand for nothing in the manifest (cells of an earlier bake)
	const orphans = fs.readdirSync( dir ).filter( ( fn ) => fn.endsWith( '.glb' ) && ! names.has( fn ) ).filter( ( fn ) => { const h = glbHead( path.join( dir, fn ) ), ex = h && h.json.scenes && h.json.scenes[ 0 ] && h.json.scenes[ 0 ].extras; return !! ( ex && ex.export && ex.export.live ); } );
	if ( o.prune ) for ( const fn of orphans ) fs.unlinkSync( path.join( dir, fn ) );
	// "only": no packed file stands beside the .glb files (the page has nothing to fall back to, and says so). Written
	// before the first one is deleted
	const only = o.dropPacked || gone === jobs.length;
	const changed = manifestGlb( dir, { v: LIVE_V, base, files: jobs.length, bytes: total, ...( only ? { only: true } : {} ) } );
	let dropped = 0;
	if ( o.dropPacked ) for ( const job of jobs ) { const f = path.join( dir, job.src ); if ( fs.existsSync( f ) ) { fs.unlinkSync( f ); dropped ++; } }
	say( `${rel}: ${jobs.length} files (${base.file} of ${man.chunks.length} cells, ${base.bytes} bytes with ${base.json} of JSON; ${jobs.length - 1} level files): ${written} written, ${kept} up to date; ${total} bytes, the packed files ${packed} (+${( 100 * ( total - packed ) / packed ).toFixed( 2 )} %)`
		+ ( o.verify ? `; verified: ${levels} levels decode to the bake's bytes, streams identical to the packed files'` : '' )
		+ ( orphans.length ? `; ${orphans.length} .glb of cells the manifest does not have ${o.prune ? 'deleted' : '(--prune deletes them)'}` : '' )
		+ ( o.dropPacked ? `; ${dropped} packed files deleted` : gone ? `; ${gone} of the packed files are not there${only ? ' (a bake that keeps none)' : ''}` : '' )
		+ `; manifest.glb ${changed ? 'written' : 'unchanged'}; ${( ( Date.now() - t0 ) / 1000 ).toFixed( 1 )} s` );
	if ( ! o.noLean && man.lean && fs.existsSync( path.join( dir, man.lean, 'manifest.json' ) ) ) liveExport( o, path.join( dir, man.lean ) );

}

// ------------------------------------------------------------------------------------------------ main

const o = parseArgs( process.argv.slice( 2 ) );
await MeshoptEncoder.ready; await MeshoptDecoder.ready;
if ( o.live ) { liveExport( o, o.dir ); process.exit( 0 ); }
const bake = new Bake( o.dir );
if ( o.check ) {

	const mode = ( ( readGltf( o.check ).json.scenes[ 0 ].extras || {} ).export || {} ).mode;
	if ( mode === 'exact' ) { const v = verifyExact( bake, o.check, bake.man.chunks ); console.log( `verify: ${v.levels} levels, ${v.vbytes} vertex bytes and ${v.ibytes} index bytes identical to the decoded bake; draw ranges, placement and extras match` ); }
	else if ( mode === 'portable' ) { const v = verifyPortable( bake, o.check, bake.man.chunks ); console.log( `verify: ${v.levels} levels, ${v.tris} triangles, as many as the bake draws; bounds inside the bake's boxes` ); }
	else if ( mode === 'dcc' ) { await dracoModules(); const v = verifyDcc( bake, o.check, null ); console.log( `verify: ${v.nodes} objects, ${v.prims} primitives, ${v.tris} triangles: positions, UVs and colours (at --color's depth) are the bake's, triangle by triangle (${v.degenerate} triangles without area left out)` + ( v.sky ? `; the sky: ${v.sky.tris} triangles facing its centre, UVs within ${v.sky.swim.toFixed( 2 )} texels of the direction, the cells ${v.sky.inside.toFixed( 1 )} m inside the shell` : '' ) ); }
	else die( `${o.check}: not a file of this tool` );
	process.exit( 0 );

}

if ( o.mode === 'dcc' && ( o.geometry === 'draco' || o.verify ) ) await dracoModules();
const tiers = o.lods ? readTiers( o.tiersFrom ) : {};
if ( o.lods && ! tiers[ o.tier ] ) die( `no tier ${o.tier} in ${o.tiersFrom} (it has: ${Object.keys( tiers ).join( ', ' )})` );
const cells = bake.select( o.cells );
if ( ! cells.length ) die( `--cells ${o.cells}: no cell` );
const stem0 = o.name || `${o.mode}_${o.lods ? 'lod-' + o.tier : o.level}`;
const say = ( ...a ) => { if ( ! o.quiet ) console.log( ...a ); };
const jobs = o.split === 'cell' ? cells.map( ( c ) => [ [ c ], `${stem0}_${c.k}` ] ) : [ [ cells, stem0 ] ];
const index = [];
let total = 0, totalGz = 0, nfiles = 0;
const t0 = Date.now();
for ( const [ list, stem ] of jobs ) {

	const res = o.mode === 'dcc' ? buildDcc( o, bake, tiers, list, stem ) : build( o, bake, tiers, list, stem );
	if ( ! res ) continue;
	let line = '';
	for ( const f of res.files ) {

		total += f.bytes; nfiles ++;
		line += `  ${path.relative( ROOT, f.file )}  ${f.bytes} bytes`;
		if ( o.gzip ) { const gz = gzipSize( f.file ); totalGz += gz; f.gzip = gz; line += ` (gzip -6: ${gz})`; }
		line += '\n';

	}

	const s = res.stats;
	say( `${path.basename( res.main )}: ${s.cells} cells, ${s.levels} levels, ${Math.round( s.tris )} triangles, ${s.verts} vertices` + ( s.chains ? `, ${s.chains} LOD chains` : '' ) + ( s.lights ? `, ${s.lights} lamps` : '' ) );
	if ( o.mode === 'dcc' ) {

		say( `  ${s.prims} primitives in ${res.json.scenes[ 0 ].nodes.length - ( s.lights ? 1 : 0 ) - ( s.sky ? 1 : 0 )} objects shown, ${res.json.materials.length} materials, ${( res.json.images || [] ).length} images (${s.textures} bytes, ${o.texSize} px, WebP q${o.texQuality}); geometry ${s.geometry} bytes (${o.geometry}), JSON ${s.json} bytes` );
		say( `  bytes by level: ${Object.entries( s.perLevel ).map( ( [ k, v ] ) => `${k} ${v}` ).join( ', ' )}; the coarsest level of everything and the textures are the first ${s.front} bytes of the binary chunk (${s.front + s.json + 28} of the file)` );
		say( `  light: ${( 100 * s.over0 ).toFixed( 2 )} % of the vertices have light above 1 (max ${s.max.toFixed( 2 )}); after the layers' gains ${( 100 * s.over / Math.max( 1, s.channels ) ).toFixed( 2 )} % of the colour channels are above 1 (--clip ${o.clip})` + ( s.mixed ? `; ${s.mixed} triangles span two texture layers (first corner's taken)` : '' ) + ( s.degenerate ? `; ${s.degenerate} triangles without area left out` : '' ) + ( s.floatPos ? `; ${s.floatPos} objects span more than 65535 grid steps: float positions` : '' ) );
		if ( s.sky ) {

			const k = s.sky, f1 = ( a ) => a.map( ( v ) => v.toFixed( 1 ) ).join( ', ' ), sz = ( mn, mx ) => [ 0, 1, 2 ].map( ( i ) => ( mx[ i ] - mn[ i ] ).toFixed( 1 ) ).join( ' x ' );
			say( `  sky: the node sky, ${k.tris} triangles (${k.geometry} bytes), panorama ${k.W} x ${k.H} WebP ${o.skyQuality >= 100 ? 'lossless' : 'q' + o.skyQuality} (${k.image} bytes), from the direction of [ ${k.C.join( ', ' )} ]; the image lies within ${k.swim.toFixed( 2 )} texels of its direction above the horizon` );
			say( `  bounds: [ ${f1( k.content.slice( 0, 3 ) )} ] .. [ ${f1( k.content.slice( 3 ) )} ] (${sz( k.content.slice( 0, 3 ), k.content.slice( 3 ) )} m) -> with the sky [ ${f1( k.min )} ] .. [ ${f1( k.max )} ] (${sz( k.min, k.max )} m); the shell is ${k.near.toFixed( 0 )} m away at its nearest, the content ${k.inside.toFixed( 1 )} m inside it`
				+ ( k.slide ? `; it slides by up to ${k.slide.max} degrees (mean ${k.slide.mean}, the moon ${k.slide.moon}) on the walk from the spawn to the far side of the plaza` : '' ) );

		}

		if ( s.panes ) say( `  lamps: ${s.panes} lantern pane triangles lit as the page's shader lights them (--lamps ${o.lamps})` + ( s.added - ( s.ground ? s.ground.more : 0 ) ? `, cut into ${s.panes + s.added - ( s.ground ? s.ground.more : 0 )}: ${s.added - ( s.ground ? s.ground.more : 0 )} triangles more than the bake has` : '' ) + ( s.globes ? `; ${s.globes} vertices of opal globes lit` : '' ) );
		if ( s.ground ) { const G = s.ground; say( `  ground: ${[ G.ao && 'contact occlusion', G.shadows && 'the lanterns\' shadows' ].filter( Boolean ).join( ' and ' )} folded into COLOR_0 at ${G.verts} vertices of ${G.tris} triangles that face up near the terrain (--ground ${o.ground || 'fold'})`
			+ ( G.cut.length ? `; on ${G.cut.join( ', ' )} ${G.split} of them cut where the factor changes (edges over ${G.edge} m, tolerance ${G.tol}): ${G.more} triangles more than the bake has` : '; no level cut' ) ); }
		const gained = s.images.filter( ( i ) => i.gain !== 1 );
		if ( gained.length ) say( `  gain: ${gained.map( ( i ) => `${i.name} ${i.gain}` ).join( ', ' )}` );
		if ( s.cutoff.length ) say( `  leaves: alphaCutoff ${s.cutoff.map( ( [ n, c ] ) => `${n} ${c}` ).join( ', ' )}` + ( s.share ? ` (share of a card kept: ${Object.entries( s.share ).map( ( [ l, v ] ) => `${bake.man.layers[ l ]} ${v.page} on the page, ${v.middle} with the page's middle value ${CUT_ALPHA}` ).join( '; ' )})` : '' ) );

	}

	if ( o.mode === 'portable' ) say( `  light: ${( 100 * s.over / Math.max( 1, s.channels ) ).toFixed( 2 )} % of the colour channels are above 1 (max ${s.max.toFixed( 2 )})` + ( o.hdr ? ', kept (--hdr: not valid glTF)' : ', clamped at 1 (--hdr keeps them)' ) + ( s.mixed ? `; ${s.mixed} triangles span two texture layers (first corner's taken)` : '' ) );
	if ( ! o.quiet ) process.stdout.write( line );
	if ( o.verify ) {

		if ( o.mode === 'dcc' ) { const v = verifyDcc( bake, res.main, res.expect ); say( `  verify: ${v.nodes} objects, ${v.prims} primitives, ${v.tris} triangles read back: positions, UVs and colours (at --color's depth) are the bake's, triangle by triangle (${v.degenerate} of the bake's triangles have no area${v.ground ? '; the ground\'s factor as the bake\'s pictures give it' : ''}${v.added ? `; ${v.added} more than the bake's, from cutting lantern panes${v.cut ? ' and the ground' : ''}` : ''})` + ( v.sky ? `; the sky: ${v.sky.tris} triangles facing its centre, UVs within ${v.sky.swim.toFixed( 2 )} texels of the direction, the cells ${v.sky.inside.toFixed( 1 )} m inside the shell` : '' ) ); }
		else if ( o.mode === 'exact' ) { const v = verifyExact( bake, res.main, list ); say( `  verify: ${v.levels} levels, ${v.vbytes} vertex bytes and ${v.ibytes} index bytes identical to the decoded bake; draw ranges, placement and extras match` ); }
		else { const v = verifyPortable( bake, res.main, list ); say( `  verify: ${v.levels} levels, ${v.tris} triangles, as many as the bake draws; bounds inside the bake's boxes` ); }

	}

	index.push( { cells: list.map( ( c ) => c.k ), file: path.basename( res.main ), files: res.files.map( ( f ) => ( { file: path.relative( o.out, f.file ).split( path.sep ).join( '/' ), bytes: f.bytes, ...( f.gzip ? { gzip: f.gzip } : {} ) } ) ), tris: Math.round( s.tris ), verts: s.verts } );

}

if ( ! index.length ) die( `none of the chosen cells has ${o.lods ? 'those levels' : 'level ' + o.level}` );
if ( o.split === 'cell' ) {

	const f = path.join( o.out, stem0 + '.index.json' );
	fs.writeFileSync( f, JSON.stringify( { mode: o.mode, level: o.level, lods: o.lods, tier: o.lods ? o.tier : undefined, baked: bake.man.baked, files: index } ) );
	say( `index: ${path.relative( ROOT, f )}` );

}

say( `${nfiles} file${nfiles === 1 ? '' : 's'}, ${total} bytes${o.gzip ? ` (gzip -6: ${totalGz})` : ''}, ${( ( Date.now() - t0 ) / 1000 ).toFixed( 1 )} s` );

function gzipSize( file ) {

	const st = fs.statSync( file );
	if ( st.size < 2 ** 30 ) return zlib.gzipSync( fs.readFileSync( file ), { level: 6 } ).length;
	// larger: independent 256 MB members (a concatenation of gzip members is a gzip file; the count is within a few bytes)
	const fd = fs.openSync( file, 'r' ), buf = Buffer.alloc( 1 << 28 );
	let n = 0, r;
	while ( ( r = fs.readSync( fd, buf, 0, buf.length, null ) ) > 0 ) n += zlib.gzipSync( buf.subarray( 0, r ), { level: 6 } ).length;
	fs.closeSync( fd );
	return n;

}
