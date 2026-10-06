// The GLTFLoader plugin that makes a town of the master file: loader.register( ( parser ) => new TownPlugin( .. ) ).
//
// The loader does everything a loader should - the JSON, the byte ranges, meshopt, the MSFT_lod chains as lazy
// THREE.LOD objects. What is the town's own is added here, through the loader's plugin hooks and nothing else:
//
//   beforeRoot  reads what the file says about itself (scenes[ 0 ].extras: the bake's manifest, the tier table,
//               the embedded files) and hands it to the page, which starts building its renderer and materials
//               while the loader is already fetching the first levels;
//   loadMesh    lets the loader build a level's mesh, then makes it the town's: the town's node material in place
//               of the glTF one (the vertices carry baked light and a texture layer, not a PBR material), the
//               bounds of each cluster, and the level's arrival is told to the page - for the levels that come
//               with the file and for every one a LOD fetches later;
//   afterRoot   ties each cell to its LOD.
//
// Files embedded in the master file (texture layers, ground maps, visibility sets) are glTF images or plain
// buffer views; file() reads one by its byte range when the page asks for it.

const LEVEL_INDEX = { lo: 0, mid: 1, near: 2, fine: 3 };

export class TownPlugin {

	// open( town ) -> Promise of { materials, arrive( cell, name, object ) }: called once the file's JSON is read,
	// see main.js
	constructor( parser, open ) {

		this.name = 'TSQ_town_master'; // (a plugin needs a name; the file has no extension of it)
		this.parser = parser;
		this.open = open;
		this.cells = []; // in the file's order: { k, box, m: { k, lo, mid?, near?, fine? }, lod, lodIndex }
		this.levelOfMesh = new Map(); // glTF mesh -> { cell, name, m }
		this.ready = null;

	}

	beforeRoot() {

		const json = this.parser.json, scene = json.scenes[ json.scene || 0 ], extras = scene.extras;
		for ( const index of scene.nodes ) {

			const node = json.nodes[ index ], finest = json.nodes[ node.children[ 0 ] ];
			const chain = finest.extensions && finest.extensions.MSFT_lod ? finest.extensions.MSFT_lod.ids : [];
			const cell = { k: node.extras.cell.cell, box: node.extras.cell.box, m: { k: node.extras.cell.cell }, lod: null, lodIndex: {} };
			[ finest, ...chain.map( ( i ) => json.nodes[ i ] ) ].forEach( ( level, i ) => {

				const m = level.extras.baked;
				cell.m[ m.level ] = m;
				cell.lodIndex[ m.level ] = i; // (a LOD's levels are finest first)
				this.levelOfMesh.set( level.mesh, { cell, name: m.level, m } );

			} );
			this.cells.push( cell );

		}

		this.files = extras.master.files;
		// size: of the binary chunk, which is the file but for its JSON; head: of its first part, the lowest level of
		// every cell, which the loader reads with the JSON before anything is shown
		const lowest = extras.master.layout.find( ( part ) => part.what === 'lo' );
		// (not awaited here: the loader goes on to request the first levels while the page builds what they need)
		this.ready = Promise.resolve( this.open( { bake: extras.baked, tiers: extras.export.tiers, cells: this.cells, file: ( name ) => this.file( name ),
			size: json.buffers[ 0 ].byteLength, head: lowest.byteOffset + lowest.byteLength } ) );
		this.ready.catch( () => {} ); // (reported by the loads that wait for it)

	}

	loadMesh( meshIndex ) {

		const level = this.levelOfMesh.get( meshIndex );
		if ( ! level ) return null;
		return Promise.all( [ this.parser.loadMesh( meshIndex ), this.ready ] ).then( ( [ object, town ] ) => this.adopt( object, meshIndex, level, town ) );

	}

	// object: what the loader built of one level - a Mesh, or a Group of one Mesh per primitive. A primitive is one
	// draw range of the bake: a kind of surface, or on the top level one cluster of the opaque range.
	adopt( object, meshIndex, { cell, name, m }, { materials, arrive } ) {

		const primitives = this.parser.json.meshes[ meshIndex ].primitives;
		const meshes = object.isMesh ? [ object ] : object.children.slice();
		const partOf = new Map( ( m.parts || [] ).map( ( part, i ) => [ part[ 0 ], i ] ) ), parts = [];
		const unit = m.q * 65535; // (positions are normalized 16 bit: one unit of the mesh's own frame in metres)
		meshes.forEach( ( mesh, i ) => {

			const { kind, sub } = primitives[ i ].extras, geometry = mesh.geometry;
			// The renderer culls by these, culling.js by the box in world space: of the cluster, or of the level. (The
			// loader takes them from the POSITION accessor, which every primitive of a level shares.)
			const wb = partOf.has( sub ) ? m.parts[ partOf.get( sub ) ][ 2 ] : [ ...m.bmin, ...m.bmax ];
			geometry.boundingBox.min.set( wb[ 0 ] - m.min[ 0 ], wb[ 1 ] - m.min[ 1 ], wb[ 2 ] - m.min[ 2 ] ).divideScalar( unit );
			geometry.boundingBox.max.set( wb[ 3 ] - m.min[ 0 ], wb[ 4 ] - m.min[ 1 ], wb[ 5 ] - m.min[ 2 ] ).divideScalar( unit );
			geometry.boundingBox.getBoundingSphere( geometry.boundingSphere );
			// (a 16 bit index marked normalized: the WebGPU backend widens plain 16 bit integers to 32 when it uploads
			// them, which for an index is GPU memory for nothing)
			if ( geometry.index.array.BYTES_PER_ELEMENT === 2 ) geometry.index.normalized = true;
			// start: where its draw range begins (the loader gives each primitive an index of its own); fade: see levels.js
			Object.assign( mesh.userData, { kind, wb, pull: materials.PULL[ LEVEL_INDEX[ name ] ] || null, start: 0, fade: 1 } );
			materials.dress( mesh );
			if ( partOf.has( sub ) ) parts[ partOf.get( sub ) ] = mesh;

		} );
		Object.assign( object.userData, { tris: m.ni / 3, all: meshes, draws: meshes, parts: parts.length ? parts : undefined } );
		arrive( cell, name, object );
		return object;

	}

	afterRoot( gltf ) {

		// (the scene's children are the cells, in the file's order; a cell's child is its LOD, or its one mesh)
		gltf.scene.children.forEach( ( node, i ) => {

			const lod = node.children[ 0 ];
			if ( ! lod || ! lod.isLOD ) return;
			lod.autoUpdate = false; // the page chooses the levels: levels.js
			this.cells[ i ].lod = lod;

		} );

	}

	// one embedded file by its name in the bake -> Blob, read by a range request of its own
	file( name ) {

		const json = this.parser.json, entry = this.files[ name ];
		if ( ! entry ) return Promise.reject( new Error( `${name} is not in the file` ) );
		const image = entry.image !== undefined ? json.images[ entry.image ] : null;
		const view = json.bufferViews[ image ? image.bufferView : entry.bufferView ];
		return this.parser.loadBufferRange( view.buffer, view.byteOffset || 0, view.byteLength )
			.then( ( bytes ) => new Blob( [ bytes ], { type: image ? image.mimeType : 'application/octet-stream' } ) );

	}

}
