// The adapter between check.js and a loader: this one registers the reference plugin of this kit.
// An adapter exports
//   configure( loader, { tier } )            called once per GLTFLoader that check.js makes; tier is 'a' or 'b'
//   loadRanged( loader, url, { probe } )     (tier B, optional) -> Promise<gltf>: a .glb through HTTP range requests
// A loader with MSFT_lod built in needs no adapter for tier A; for tier B the adapter is where its option is set.

import { GLTFMSFTLodExtension, loadRangedGLB } from './GLTFMSFTLodExtension.js';

export function configure( loader, { tier } ) {

	loader.register( ( parser ) => new GLTFMSFTLodExtension( parser, { lazy: tier === 'b' } ) );

}

export function loadRanged( loader, url, options ) {

	return loadRangedGLB( loader, url, options );

}
