// A walk through the lantern town, out of ONE file.
//
// town_master.glb is plain glTF 2.0: every cell of the town with its chain of levels of detail (MSFT_lod),
// meshopt-packed, the texture layers as WebP images, and the bake's manifest in the scene's extras. The stock
// GLTFLoader with lazy levels of detail and range requests (the three.js pull request this page is a showcase
// for) reads it: the JSON first, then the lowest level of every cell, then each further level and each picture
// as a byte range of the same file when the walk asks for it. A visit fetches a part of the file, never all.
//
// This module only wires the page together; each part is a module with one job:
//
//   town-plugin.js  the GLTFLoader plugin: the file's extras, the town's materials on the loader's meshes
//   stage.js        renderer, scene, camera, fog and light response       look.js   the look's numbers, the tiers
//   materials/      the five TSL node materials of the baked town         sprites.js   lantern glows and flames
//   textures.js     texture arrays from the embedded pictures             mirror.js    the wet paving's mirror
//   compress.js     ... block-compressed for the GPU in workers           culling.js   view and visibility sets
//   levels.js       which level each cell shows (budget.js: per view)     walk.js      walking, viewpoints
//   post.js         TRAA, natural bloom, grade, tone curve                stats.js     HUD, window.__stats
//   menu.js         veil, quality and effects menu
//
// ?quality=high|medium  ?webgl=1  ?at=x,y,z,yaw,pitch  ?gputime=1  ?gsh=0  ?tc=0 (textures uncompressed)

import * as THREE from 'three';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { GLTFLoader } from '../gltflod/lib/GLTFLoader.tierB.js';
import { Nav } from '../glbtown/nav.js';
import { TownPlugin } from './town-plugin.js';
import { createStage, addSky } from './stage.js';
import { FOV, applyLook, pickQuality, tierOf, pixelRatio } from './look.js';
import { createTownMaterials } from './materials/index.js';
import { createSprites } from './sprites.js';
import { createTextures } from './textures.js';
import { createMirror, pavingUnder } from './mirror.js';
import { createPost } from './post.js';
import { TownLevels } from './levels.js';
import { TownCulling } from './culling.js';
import { createWalk } from './walk.js';
import { createVeil, createMenu } from './menu.js';
import { createStats } from './stats.js';

const FILE = '../out/web/_master/town_master.glb';

const query = new URLSearchParams( location.search ), $ = ( id ) => document.getElementById( id );
const canvas = $( 'gl' ), veil = createVeil();
const quality = pickQuality( query );
const touch = matchMedia( '(pointer: coarse)' ).matches;
document.body.classList.toggle( 'touch', touch );

// Called by the plugin as soon as the file's JSON is read, with what the file says about itself. Everything a
// level's mesh needs to become part of the town is built here - while the loader is already fetching the
// lowest levels, which wait for this at the plugin's door.
async function open( { bake, tiers, cells, file, size, head } ) {

	const tier = tierOf( tiers, quality );
	if ( tier.workers ) MeshoptDecoder.useWorkers( tier.workers ); // (before the first level is unpacked)
	const stage = await createStage( { canvas, fov: FOV, forceWebGL: query.get( 'webgl' ) === '1', gpuTime: query.get( 'gputime' ) === '1' } );
	const resize = () => stage.setSize( innerWidth, innerHeight, pixelRatio( tier, innerWidth, innerHeight ) );
	resize();
	addEventListener( 'resize', resize );

	addSky( stage, bake.look.moon_dir );
	applyLook( stage );
	const mirror = createMirror( stage, tier );
	const post = createPost( stage, tier, mirror );
	const textures = createTextures( { file, bake, tier, renderer: stage.renderer, compress: query.get( 'tc' ) !== '0' } );
	const pvs = file( bake.pvs.file ).then( ( blob ) => blob.text() ).then( JSON.parse ).catch( () => null ); // (without them: no occlusion, the same picture)
	const first = await textures.pages();
	const materials = createTownMaterials( { stage, bake, tier, layers: first } );
	createSprites( { stage, bake } );
	const levels = new TownLevels( { cells, tier, materials, renderer: stage.renderer, mirror: mirror.image, cellSize: bake.bake.cell } );
	textures.yieldTo( levels ); // a level coming in is seen, a sharper texture is not: the pictures wait for the levels

	return {
		bake, tier, size, head, stage, mirror, post, textures, first, materials, levels, pvs,
		arrive( cell, name, object ) {

			levels.arrive( cell, name, object );
			veil.progress( 1 - levels.baseLeft / cells.length );

		},
	};

}

async function main() {

	const loader = new GLTFLoader().setMeshoptDecoder( MeshoptDecoder ).setLazyLOD( true ).setRangeRequests( true );
	let plugin;
	loader.register( ( parser ) => ( plugin = new TownPlugin( parser, open ) ) );
	const gltf = await loader.loadAsync( FILE ); // the scene: every cell at its lowest level, the others on demand
	const { bake, tier, size, head, stage, mirror, post, textures, first, materials, levels, pvs } = await plugin.ready;
	const { renderer, scene, camera } = stage;

	// The town stands under a group that cuts it at the paving while the mirror image is drawn, and only then.
	const town = new THREE.ClippingGroup();
	town.name = 'town';
	town.clippingPlanes = [ mirror.image.clip ];
	town.enabled = false;
	town.add( gltf.scene );
	scene.add( town );

	const nav = new Nav( bake.data );
	const culling = new TownCulling( { levels, nav, pvs: await pvs, baked: bake.baked } );
	// the mirror image: of the lowest levels, their glowing cards pulled well forward, nothing from under the paving
	mirror.image.ground = pavingUnder( nav );
	mirror.image.swap = ( on ) => {

		materials.pullAll.value.set( on ? materials.PULL.mirror[ 0 ] : 0, on ? materials.PULL.mirror[ 1 ] : 0 );
		town.enabled = on;
		levels.mirrorView( on );

	};

	const at = ( query.get( 'at' ) || '' ).split( ',' ).map( parseFloat );
	const walk = createWalk( { camera, canvas, nav, spawn: bake.data.spawn, at: at.length === 5 && at.every( isFinite ) ? at : null, stick: $( 'stick' ), fov: FOV, touch } );
	createMenu( { quality, walk, post, shadows: materials.GAO.sh.on, shots: bake.data.shots, query } );
	const stats = createStats( { renderer, hud: $( 'hud' ), label: `${quality} · one file · ${stage.isWebGPU ? 'WebGPU' : 'WebGL2'}${stage.HDR.out ? ' HDR' : ''}`,
		url: FILE, size, head, describe: () => ( { ...levels.stats(), hidden: culling.hidden } ), gpuTime: query.get( 'gputime' ) === '1' } );

	// for driving the page from outside: a fixed viewpoint { pos, target, fov } in the town's frame (null: walk
	// again), the walker, the parts (window.__auto: a scripted walk, see walk.js)
	window.__view = ( view ) => { walk.view = view; };
	window.__player = walk.player;
	window.__town = { bake, tier, stage, levels, culling, textures, materials, mirror, post, done: null };

	// the levels around the place one stands in, before the first frame
	walk.update( 0 );
	await levels.settle( camera );
	veil.ready( canvas, touch );

	const timer = new THREE.Timer();
	renderer.setAnimationLoop( ( time ) => {

		const dt = Math.min( 0.05, timer.update( time ).getDelta() );
		walk.update( dt );
		levels.update( camera ); // asks the loader for what the walk needs next
		mirror.update();
		culling.apply( camera ); // (after the mirror image, which is another view)
		post.render();
		stats.frame( dt );

	} );

	// behind the first frame: every texture layer at its full size, the materials' normals, the ground's maps
	const late = ( what, job ) => job.catch( ( e ) => console.warn( `town: ${what}: ${e.message || e}` ) );
	window.__town.done = Promise.all( [
		late( 'sharp textures', textures.sharpen( materials.TX, first ) )
			.then( () => late( 'normal layers', textures.normals( materials.NR ) ) )
			.then( () => late( 'first textures', textures.compact( materials.TX, first ) ) ),
		late( 'ground maps', textures.ground( materials.GAO ) ),
	] );

}

main().catch( ( e ) => {

	console.error( e );
	veil.fail( e );

} );
