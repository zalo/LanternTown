import { DataTexture, FloatType, RGBAFormat, RedFormat, Vector2, Vector3, LightsNode, NodeUpdateType } from 'three/webgpu';

import {
	attributeArray, nodeProxy, int, float, vec3, vec4, ivec2, ivec4, uniform, Break, Loop, positionView,
	Fn, If, Return, textureLoad, instanceIndex, screenCoordinate, directPointLight,
	renderGroup,
	min, max, pow, log, clamp, dot, vec2, mix, smoothstep, oneMinus, floor, fract
} from 'three/tsl';

// vendored from three/examples/jsm/tsl/lighting/ClusteredLightsNode.js (r186) with two changes:
//  - SOURCE_RADIUS: soft near-field falloff (see setupLights); capFactor: capped lanterns; off-axis projections
//  - CPU clustering for the WebGL2 backend (see "WebGL2 fallback" below); the WebGPU path is three's, untouched
export const sourceRadius = uniform( 0 ); // m; 0 = point sources (three's behaviour)

// tsq: capped lanterns. light.userData.up (0..1, default 1 = bare source) is the share of the light that leaves above the
// cap: full strength into the pool below (steeper than ~25 degrees down), falling through the horizon of the flame to
// `up` from ~17 degrees above it - facades get a gradient instead of an even wash. sinElev = height of the lit point
// over the light / distance. (shared with the shadow-casting lanterns, see engine.js capLight)
export const capFactor = ( sinElev, up ) => mix( float( 1 ), up, smoothstep( - 0.42, 0.3, sinElev ) );
const _vector3 = /*@__PURE__*/ new Vector3();
const _size = /*@__PURE__*/ new Vector2();

// WebGL2 fallback. three's implementation assigns lights to clusters in a compute pass that writes per-cluster index
// lists into a storage buffer. The WebGL2 backend runs "compute" as transform feedback, which can neither write those
// lists nor leave the GL state intact here (it raises INVALID_OPERATION inside the scene pass and a third of the
// frame's draws then fail the same way: no sky dome, missing meshes). So on WebGL2 the clusters are filled on the CPU
// each frame and read from one float texture:
//   rows [ 0, tableRows )  one texel per cluster (+ 1): where its list starts in the index list; it ends where the
//                          next one starts
//   rows [ tableRows, .. ) the index list: light indexes (into _lightsTexture), cluster after cluster
// The grid is fixed in NDC (not in pixels) and its depth slices are fitted to the lights every frame, so it is
// independent of the drawing buffer size, of the render target (reflection probes) and of y-flip conventions.
const CPU_GRID = { x: 24, y: 14, z: 20 };
const CPU_TEX_WIDTH = 512;
const CPU_Z_START = 1.0; // m: everything nearer than this shares the first depth slice

/**
 * A custom version of `LightsNode` implementing Forward+ clustered shading:
 * the view frustum is subdivided into a 3D grid of clusters (X × Y screen tiles
 * times an exponentially-spaced set of Z depth slices), and each cluster holds
 * only the point lights whose spheres intersect it. At shading time each fragment
 * looks up its cluster and loops over just that cluster's lights. Unlike 2D tiled
 * lighting, clustered shading culls lights that share screen pixels but lie at
 * different depths — suitable for 3D scenes with real depth complexity.
 *
 * @augments LightsNode
 * @three_import import { clusteredLights } from 'three/addons/tsl/lighting/ClusteredLightsNode.js';
 */
class ClusteredLightsNode extends LightsNode {

	static get type() {

		return 'ClusteredLightsNode';

	}

	/**
	 * Constructs a new clustered lights node.
	 *
	 * @param {number} [maxLights=1024] - Maximum number of point lights.
	 * @param {number} [tileSize=32] - Screen tile size in pixels (cluster XY size).
	 * @param {number} [zSlices=24] - Number of exponential depth slices.
	 * @param {number} [maxLightsPerCluster=64] - Per-cluster light-list capacity.
	 */
	constructor( maxLights = 1024, tileSize = 32, zSlices = 24, maxLightsPerCluster = 64 ) {

		super();

		this.materialLights = [];
		this.clusteredLights = [];

		this.maxLights = maxLights;
		this.tileSize = tileSize;
		this.zSlices = zSlices;
		this.maxLightsPerCluster = maxLightsPerCluster;

		this._chunksPerCluster = Math.ceil( maxLightsPerCluster / 4 );

		this._bufferSize = null;
		this._lightIndexes = null;
		this._screenClusterIndex = null;
		this._compute = null;

		const lightsData = new Float32Array( maxLights * 4 * 2 );
		this._lightsTexture = new DataTexture( lightsData, lightsData.length / 8, 2, RGBAFormat, FloatType );

		// Per Z-slice light range for Z-culling (CPU-sorted, uploaded each frame)

		this._zSliceRangesData = new Float32Array( zSlices * 4 );
		this._zSliceRangesTexture = new DataTexture( this._zSliceRangesData, zSlices, 1, RGBAFormat, FloatType );

		this._lightViewZ = new Float32Array( maxLights );
		this._lightSortOrder = [];

		this._lightsCount = uniform( 0, 'int' );

		// Render-group uniforms: shared between compute and fragment passes,
		// updated manually each frame in updateBefore (compute lacks a camera context).
		this._cameraNear = uniform( 0 ).setName( 'clusteredCameraNear' ).setGroup( renderGroup );
		this._cameraFar = uniform( 0 ).setName( 'clusteredCameraFar' ).setGroup( renderGroup );
		this._cameraViewMatrix = uniform( 'mat4' ).setName( 'clusteredCameraViewMatrix' ).setGroup( renderGroup );
		this._cameraProjectionMatrix = uniform( 'mat4' ).setName( 'clusteredCameraProjectionMatrix' ).setGroup( renderGroup );

		this._gridDimensions = uniform( new Vector2() );

		// WebGL2 fallback (CPU clustering): null = backend not known yet
		this.cpuGrid = { ...CPU_GRID };
		this._cpu = null;
		this._clusterTexture = null;
		this._clusterRange = null;
		this._sliceScale = uniform( 1 ).setName( 'clusteredSliceScale' ).setGroup( renderGroup );
		this._sliceBias = uniform( 0 ).setName( 'clusteredSliceBias' ).setGroup( renderGroup );

		this.updateBeforeType = NodeUpdateType.RENDER;

	}

	customCacheKey() {

		return ( this._compute ? this._compute.getCacheKey() : 0 ) + super.customCacheKey();

	}

	updateLightsTexture( camera ) {

		const { _lightsTexture: lightsTexture, clusteredLights } = this;

		const data = lightsTexture.image.data;
		const lineSize = lightsTexture.image.width * 4;
		const count = clusteredLights.length;

		this._lightsCount.value = count;

		// Sort lights by view-space depth for Z-culling

		const viewZ = this._lightViewZ;
		const order = this._lightSortOrder;

		for ( let i = 0; i < count; i ++ ) {

			_vector3.setFromMatrixPosition( clusteredLights[ i ].matrixWorld );
			_vector3.applyMatrix4( camera.matrixWorldInverse );
			viewZ[ i ] = _vector3.z;
			order[ i ] = i;

		}

		order.length = count;
		order.sort( ( a, b ) => viewZ[ a ] - viewZ[ b ] );

		// Write sorted lights to texture

		for ( let i = 0; i < count; i ++ ) {

			const light = clusteredLights[ order[ i ] ];

			_vector3.setFromMatrixPosition( light.matrixWorld );

			const offset = i * 4;

			data[ offset + 0 ] = _vector3.x;
			data[ offset + 1 ] = _vector3.y;
			data[ offset + 2 ] = _vector3.z;
			data[ offset + 3 ] = light.distance;

			data[ lineSize + offset + 0 ] = light.color.r * light.intensity;
			data[ lineSize + offset + 1 ] = light.color.g * light.intensity;
			data[ lineSize + offset + 2 ] = light.color.b * light.intensity;
			// (decay is a whole number here: its fraction carries the cap, 0.5 = bare)
			data[ lineSize + offset + 3 ] = Math.round( light.decay ) + 0.5 * Math.min( 1, Math.max( 0, light.userData.up ?? 1 ) );

		}

		lightsTexture.needsUpdate = true;

		if ( this._cpu !== null ) return; // WebGL2 fallback: the Z ranges only feed the compute pass

		// Compute per Z-slice light ranges

		const zRanges = this._zSliceRangesData;

		const near = camera.near;
		const far = camera.far;
		const NZ = this.zSlices;

		for ( let z = 0; z < NZ; z ++ ) {

			// Exponential Z-slice bounds (view-space, negative values)
			const sliceNear = - ( near * Math.pow( far / near, z / NZ ) );
			const sliceFar = - ( near * Math.pow( far / near, ( z + 1 ) / NZ ) );

			let rangeStart = count;
			let rangeEnd = 0;

			for ( let i = 0; i < count; i ++ ) {

				const vz = viewZ[ order[ i ] ];
				const r = clusteredLights[ order[ i ] ].distance;
				const radius = r > 0 ? r : far;

				// Light sphere Z: [vz - radius, vz + radius]
				// Slice Z: [sliceFar, sliceNear] (both negative, sliceFar < sliceNear)
				if ( vz + radius >= sliceFar && vz - radius <= sliceNear ) {

					if ( i < rangeStart ) rangeStart = i;
					if ( i + 1 > rangeEnd ) rangeEnd = i + 1;

				}

			}

			if ( rangeStart >= count ) {

				rangeStart = 0;
				rangeEnd = 0;

			}

			zRanges[ z * 4 ] = rangeStart;
			zRanges[ z * 4 + 1 ] = rangeEnd;

		}

		this._zSliceRangesTexture.needsUpdate = true;

	}

	updateBefore( frame ) {

		const { renderer, camera } = frame;

		this.updateProgram( renderer );

		this.updateLightsTexture( camera );

		this._cameraNear.value = camera.near;
		this._cameraFar.value = camera.far;
		this._cameraViewMatrix.value = camera.matrixWorldInverse;
		this._cameraProjectionMatrix.value = camera.projectionMatrix;

		if ( this._cpu !== null ) this.updateClustersCPU( camera );
		else renderer.compute( this._compute );

	}

	/**
	 * WebGL2 fallback: assigns the lights to the cluster grid on the CPU and uploads the per-cluster lists.
	 * Per light only the clusters inside the depth-slice / tile range its sphere of influence can reach are
	 * visited (sphere vs. view-space cluster AABB, like the compute pass), so the cost follows the screen
	 * coverage of the lights rather than clusters x lights.
	 *
	 * @param {Camera} camera - The camera the scene is rendered with.
	 */
	updateClustersCPU( camera ) {

		const C = this._cpu;
		const { NX, NY, NZ, tableRows, counts, offsets, zEdge, dx2, dy2, view } = C;
		const W = CPU_TEX_WIDTH;
		const count = this.clusteredLights.length;
		const lights = this._lightsTexture.image.data, lineSize = this._lightsTexture.image.width * 4;
		const data = this._clusterTexture.image.data;
		const me = camera.matrixWorldInverse.elements, pe = camera.projectionMatrix.elements;
		// ndc.x = fx * x / w - ox, with w = depth for a perspective and 1 for an orthographic camera
		const ortho = pe[ 11 ] === 0;
		const fx = pe[ 0 ], fy = pe[ 5 ], ox = ortho ? - pe[ 12 ] : pe[ 8 ], oy = ortho ? - pe[ 13 ] : pe[ 9 ];
		const near = camera.near, far = camera.far;
		const maxPer = this.maxLightsPerCluster;

		// view-space positions of the (depth-sorted) lights that emit anything, and how deep they reach
		let n = 0, reach = 0;
		for ( let i = 0; i < count; i ++ ) {

			const o = i * 4;
			if ( lights[ lineSize + o ] <= 0 && lights[ lineSize + o + 1 ] <= 0 && lights[ lineSize + o + 2 ] <= 0 ) continue;
			const x = lights[ o ], y = lights[ o + 1 ], z = lights[ o + 2 ];
			const r = lights[ o + 3 ] > 0 ? lights[ o + 3 ] : far;
			const d = - ( me[ 2 ] * x + me[ 6 ] * y + me[ 10 ] * z + me[ 14 ] );
			if ( d + r <= near ) continue; // entirely behind the camera
			const v = n * 5;
			view[ v ] = me[ 0 ] * x + me[ 4 ] * y + me[ 8 ] * z + me[ 12 ];
			view[ v + 1 ] = me[ 1 ] * x + me[ 5 ] * y + me[ 9 ] * z + me[ 13 ];
			view[ v + 2 ] = d;
			view[ v + 3 ] = r;
			view[ v + 4 ] = i;
			if ( d + r > reach ) reach = d + r;
			n ++;

		}

		// exponential depth slices from CPU_Z_START to the deepest point any light reaches (first / last slice open-ended)
		const zA = Math.max( CPU_Z_START, near ), zB = Math.min( Math.max( reach, zA * 8 ), Math.max( far, zA * 8 ) );
		const scale = NZ / Math.log( zB / zA ), bias = - Math.log( zA ) * scale;
		this._sliceScale.value = scale;
		this._sliceBias.value = bias;
		for ( let s = 0; s <= NZ; s ++ ) zEdge[ s ] = zA * Math.pow( zB / zA, s / NZ );
		zEdge[ 0 ] = ortho ? - Infinity : Math.min( near, zA );
		zEdge[ NZ ] = Math.max( far, zB );

		// tile edges as view-space slopes ( x / w, y / w )
		if ( C.fx !== fx || C.fy !== fy || C.ox !== ox || C.oy !== oy ) {

			C.fx = fx; C.fy = fy; C.ox = ox; C.oy = oy;
			for ( let i = 0; i <= NX; i ++ ) C.uEdge[ i ] = ( i * 2 / NX - 1 + ox ) / fx;
			for ( let i = 0; i <= NY; i ++ ) C.vEdge[ i ] = ( i * 2 / NY - 1 + oy ) / fy;

		}

		const uEdge = C.uEdge, vEdge = C.vEdge;

		// pass 1: collect ( cluster, light ) pairs
		counts.fill( 0 );
		let pairs = C.pairs, np = 0;
		const tiles = NX * NY;
		for ( let l = 0; l < n; l ++ ) {

			const v = l * 5;
			const lx = view[ v ], ly = view[ v + 1 ], ld = view[ v + 2 ], r = view[ v + 3 ], li = view[ v + 4 ];
			const r2 = r * r;
			const s0 = Math.min( NZ - 1, Math.max( 0, Math.floor( Math.log( Math.max( ld - r, 1e-6 ) ) * scale + bias ) ) );
			const s1 = Math.min( NZ - 1, Math.max( 0, Math.floor( Math.log( Math.max( ld + r, 1e-6 ) ) * scale + bias ) ) );
			if ( pairs.length < ( np + ( s1 - s0 + 1 ) * tiles ) * 2 ) {

				const grown = new Int32Array( Math.max( pairs.length * 2, ( np + ( s1 - s0 + 1 ) * tiles ) * 2 ) );
				grown.set( pairs.subarray( 0, np * 2 ) );
				pairs = C.pairs = grown;

			}

			for ( let s = s0; s <= s1; s ++ ) {

				const zn = zEdge[ s ], zf = zEdge[ s + 1 ];
				const dz = ld < zn ? zn - ld : ld > zf ? ld - zf : 0;
				const q2 = r2 - dz * dz;
				if ( q2 < 0 ) continue;
				// tile range: bounds of the sphere's slab inside this slice, projected with the depth that widens them
				const q = Math.sqrt( q2 );
				const wn = ortho ? 1 : zn, wf = ortho ? 1 : zf; // w at the two faces of the slice
				const d0 = ortho ? 1 : Math.max( zn, ld - r, 1e-6 ), d1 = ortho ? 1 : Math.max( Math.min( zf, ld + r ), d0 );
				const xa = lx - q, xb = lx + q, ya = ly - q, yb = ly + q;
				const fx0 = ( ( xa / ( xa < 0 ? d0 : d1 ) * fx - ox ) * 0.5 + 0.5 ) * NX, fx1 = ( ( xb / ( xb > 0 ? d0 : d1 ) * fx - ox ) * 0.5 + 0.5 ) * NX;
				const fy0 = ( ( ya / ( ya < 0 ? d0 : d1 ) * fy - oy ) * 0.5 + 0.5 ) * NY, fy1 = ( ( yb / ( yb > 0 ? d0 : d1 ) * fy - oy ) * 0.5 + 0.5 ) * NY;
				if ( fx1 < 0 || fx0 >= NX || fy1 < 0 || fy0 >= NY ) continue;
				const tx0 = Math.max( 0, Math.floor( fx0 ) ), tx1 = Math.min( NX - 1, Math.floor( fx1 ) );
				const ty0 = Math.max( 0, Math.floor( fy0 ) ), ty1 = Math.min( NY - 1, Math.floor( fy1 ) );
				// squared distance from the light to each tile's view-space extent in this slice, per axis
				for ( let t = tx0; t <= tx1; t ++ ) {

					const a = uEdge[ t ], b = uEdge[ t + 1 ];
					const lo = Math.min( a * wn, a * wf ), hi = Math.max( b * wn, b * wf );
					const e = lx < lo ? lo - lx : lx > hi ? lx - hi : 0;
					dx2[ t ] = e * e;

				}

				for ( let t = ty0; t <= ty1; t ++ ) {

					const a = vEdge[ t ], b = vEdge[ t + 1 ];
					const lo = Math.min( a * wn, a * wf ), hi = Math.max( b * wn, b * wf );
					const e = ly < lo ? lo - ly : ly > hi ? ly - hi : 0;
					dy2[ t ] = e * e;

				}

				const base = s * tiles;
				for ( let ty = ty0; ty <= ty1; ty ++ ) {

					const rest = q2 - dy2[ ty ];
					if ( rest < 0 ) continue;
					const row = base + ty * NX;
					for ( let tx = tx0; tx <= tx1; tx ++ ) {

						if ( dx2[ tx ] > rest ) continue;
						const c = row + tx;
						if ( counts[ c ] >= maxPer ) continue;
						counts[ c ] ++;
						pairs[ np * 2 ] = c;
						pairs[ np * 2 + 1 ] = li;
						np ++;

					}

				}

			}

		}

		// pass 2: lay the lists out back to back and write table + lists
		let total = 0;
		const clusters = tiles * NZ;
		for ( let c = 0; c < clusters; c ++ ) {

			offsets[ c ] = total;
			data[ c ] = total;
			total += counts[ c ];

		}

		data[ clusters ] = total;

		const list = tableRows * W;
		for ( let p = 0; p < np; p ++ ) data[ list + offsets[ pairs[ p * 2 ] ] ++ ] = pairs[ p * 2 + 1 ];

		C.stats.lights = n; C.stats.pairs = np;

		this._clusterTexture.needsUpdate = true;

	}

	setLights( lights ) {

		const { clusteredLights, materialLights } = this;

		let materialIndex = 0;
		let clusteredIndex = 0;

		for ( const light of lights ) {

			if ( light.isPointLight === true && light.castShadow !== true ) {

				clusteredLights[ clusteredIndex ++ ] = light;

			} else {

				materialLights[ materialIndex ++ ] = light;

			}

		}

		materialLights.length = materialIndex;
		clusteredLights.length = clusteredIndex;

		return super.setLights( lights );

	}

	getBuiltinLights() {

		return this.materialLights;

	}

	getBlock() {

		return this._lightIndexes.element( this._screenClusterIndex.mul( int( this._chunksPerCluster ) ) );

	}

	getTile( element ) {

		element = int( element );

		const stride = int( 4 );
		const chunkOffset = element.div( stride );
		const idx = this._screenClusterIndex.mul( int( this._chunksPerCluster ) ).add( chunkOffset );

		return this._lightIndexes.element( idx ).element( element.mod( stride ) );

	}

	getClusterLightCount( zSliceNode ) {

		const getCount = Fn( ( [ zSliceNode ] ) => {

			const count = int( 0 ).toVar();

			const debugClusterIndex = this._screenClusterIndex.toVar();

			If( zSliceNode.greaterThanEqual( int( 0 ) ), () => {

				const tileSize = int( this.tileSize );
				const screenTile = screenCoordinate.div( tileSize ).floor();
				const NX = int( this._gridDimensions.x );
				const NY = int( this._gridDimensions.y );

				debugClusterIndex.assign(
					int( screenTile.x )
						.add( int( screenTile.y ).mul( NX ) )
						.add( zSliceNode.mul( NX.mul( NY ) ) )
				);

			} );

			Loop( this.maxLightsPerCluster, ( { i } ) => {

				const element = int( i );
				const stride = int( 4 );
				const chunkOffset = element.div( stride );
				const idx = debugClusterIndex.mul( int( this._chunksPerCluster ) ).add( chunkOffset );
				const lightIndex = this._lightIndexes.element( idx ).element( element.mod( stride ) );

				If( lightIndex.equal( int( 0 ) ), () => {

					Break();

				} );

				count.addAssign( int( 1 ) );

			} );

			return count;

		} );

		return getCount( zSliceNode );

	}

	getLightData( index ) {

		index = int( index );

		const dataA = textureLoad( this._lightsTexture, ivec2( index, 0 ) );
		const dataB = textureLoad( this._lightsTexture, ivec2( index, 1 ) );

		const position = dataA.xyz;
		const viewPosition = this._cameraViewMatrix.mul( vec4( position, 1.0 ) ).xyz;
		const distance = dataA.w;
		const color = dataB.rgb;
		const decay = floor( dataB.w );
		const up = fract( dataB.w ).mul( 2 );

		return {
			position,
			viewPosition,
			distance,
			color,
			decay,
			up
		};

	}

	setupLights( builder, lightNodes ) {

		this.updateProgram( builder.renderer );

		//

		const lightingModel = builder.context.reflectedLight;

		lightingModel.directDiffuse.toStack();
		lightingModel.directSpecular.toStack();

		super.setupLights( builder, lightNodes );

		const upView = this._cameraViewMatrix.mul( vec4( 0, 1, 0, 0 ) ).xyz;

		// one clustered light (index into the lights texture) -> direct lighting
		const shade = ( index ) => {

				const { color, decay, viewPosition, distance, up } = this.getLightData( index );

				const lightVector = viewPosition.sub( positionView );
				// tsq: a lantern is a small area source, not a point. 1 / ( d2 + r2 ) instead of 1 / d2 keeps a wall half a
				// metre from the flame from burning out while leaving everything beyond ~2 m as it was.
				const d2 = dot( lightVector, lightVector );
				const soft = d2.div( d2.add( sourceRadius.mul( sourceRadius ) ) )
					.mul( capFactor( dot( lightVector, upView ).negate().div( d2.max( 1e-4 ).sqrt() ), up ) );

				// Early-out: skip full BRDF if fragment is beyond the light's cutoff
				If( distance.equal( 0 ).or( dot( lightVector, lightVector ).lessThanEqual( distance.mul( distance ) ) ), () => {

					builder.lightsNode.setupDirectLight( builder, this, directPointLight( {
						color: color.mul( soft ),
						lightVector,
						cutoffDistance: distance,
						decayExponent: decay
					} ) );

				} );

		};

		if ( this._cpu !== null ) {

			// WebGL2 fallback: walk this fragment's cluster list in the CPU-filled texture
			const W = CPU_TEX_WIDTH, tableRows = this._cpu.tableRows;

			Fn( () => {

				const range = this._clusterRange;

				Loop( { start: range.x, end: range.y, type: 'int', condition: '<' }, ( { i } ) => {

					shade( int( textureLoad( this._clusterTexture, ivec2( i.mod( int( W ) ), i.div( int( W ) ).add( int( tableRows ) ) ) ).x ) );

				} );

			}, 'void' )();

			return;

		}

		Fn( () => {

			Loop( this.maxLightsPerCluster, ( { i } ) => {

				const lightIndex = this.getTile( i );

				If( lightIndex.equal( int( 0 ) ), () => {

					Break();

				} );

				shade( lightIndex.sub( 1 ) );

			} );

		}, 'void' )();

	}

	getBufferFitSize( value ) {

		const multiple = this.tileSize;

		return Math.ceil( value / multiple ) * multiple;

	}

	setSize( width, height ) {

		width = this.getBufferFitSize( width );
		height = this.getBufferFitSize( height );

		if ( ! this._bufferSize || this._bufferSize.width !== width || this._bufferSize.height !== height ) {

			this.create( width, height );

		}

		return this;

	}

	updateProgram( renderer ) {

		if ( renderer.backend.isWebGPUBackend !== true ) {

			if ( this._cpu === null ) this.createCPU();
			return;

		}

		renderer.getDrawingBufferSize( _size );

		const width = this.getBufferFitSize( _size.width );
		const height = this.getBufferFitSize( _size.height );

		if ( this._bufferSize === null ) {

			this.create( width, height );

		} else if ( this._bufferSize.width !== width || this._bufferSize.height !== height ) {

			this.create( width, height );

		}

	}

	create( width, height ) {

		const { tileSize, zSlices, maxLightsPerCluster, _chunksPerCluster: chunksPerCluster } = this;

		const bufferSize = new Vector2( width, height );

		const NX = Math.floor( bufferSize.width / tileSize );
		const NY = Math.floor( bufferSize.height / tileSize );
		const NZ = zSlices;
		const clusterCount = NX * NY * NZ;

		this._gridDimensions.value.set( NX, NY );

		// release the compute pipeline and bindings built for the previous grid

		if ( this._compute !== null ) this._compute.dispose();

		// Per-cluster light-index storage (ivec4 chunks).

		const lightIndexesLength = clusterCount * chunksPerCluster * 4;

		let lightIndexes = this._lightIndexes;

		if ( lightIndexes === null || lightIndexes.value.array.length < lightIndexesLength ) {

			// TODO: Free the outgoing buffer once the renderer destroys the GPU resources of
			// standalone storage attributes (`Bindings._destroyBindings()` skips storage buffers).

			const lightIndexesArray = new Int32Array( lightIndexesLength );
			lightIndexes = attributeArray( lightIndexesArray, 'ivec4' ).setName( 'lightIndexes' );

		}

		// compute-side accessors (use instanceIndex)

		const getClusterChunk = ( chunkIdx ) => {

			const idx = instanceIndex.mul( int( chunksPerCluster ) ).add( int( chunkIdx ) );

			return lightIndexes.element( idx );

		};

		const getClusterSlot = ( slotIdx ) => {

			slotIdx = int( slotIdx );

			const stride = int( 4 );
			const chunkOffset = slotIdx.div( stride );
			const idx = instanceIndex.mul( int( chunksPerCluster ) ).add( chunkOffset );

			return lightIndexes.element( idx ).element( slotIdx.mod( stride ) );

		};

		// compute: one thread per cluster

		const compute = Fn( () => {

			// view-space scale factors derived from the projection matrix:
			//   view_x = ndc_x * (-view_z) / focal_x = ndc_x * (-view_z) * invFocalX
			//   view_y = ndc_y * (-view_z) / focal_y = ndc_y * (-view_z) * invFocalY
			// where focal_x = projMatrix[0][0] and focal_y = projMatrix[1][1].
			const invFocalX = float( 1 ).div( this._cameraProjectionMatrix.element( 0 ).element( 0 ) );
			const invFocalY = float( 1 ).div( this._cameraProjectionMatrix.element( 1 ).element( 1 ) );

			// 3D cluster coordinates from instanceIndex
			const cx = instanceIndex.mod( NX );
			const cy = instanceIndex.div( NX ).mod( NY );
			const cz = instanceIndex.div( NX * NY );

			// NDC X/Y bounds of the cluster.
			// Y is flipped: cy=0 is the top screen row (fragment y=0), which is NDC y=+1.
			// tsq: plus the off-axis terms of the projection (0 for a centred frustum; a shift-lens review camera -
			// setViewOffset - slides the frame: ndc = focal * view / -z - offset)
			const offX = this._cameraProjectionMatrix.element( 2 ).element( 0 );
			const offY = this._cameraProjectionMatrix.element( 2 ).element( 1 );
			const ndcXmin = float( cx ).mul( 2.0 / NX ).sub( 1.0 ).add( offX );
			const ndcXmax = float( cx.add( int( 1 ) ) ).mul( 2.0 / NX ).sub( 1.0 ).add( offX );
			const ndcYmax = float( 1 ).sub( float( cy ).mul( 2.0 / NY ) ).add( offY );
			const ndcYmin = float( 1 ).sub( float( cy.add( int( 1 ) ) ).mul( 2.0 / NY ) ).add( offY );

			// View-space Z bounds (negative, exponential slicing)
			const farOverNear = this._cameraFar.div( this._cameraNear );
			const zNearCluster = this._cameraNear.mul( pow( farOverNear, float( cz ).mul( 1.0 / NZ ) ) ).negate();
			const zFarCluster = this._cameraNear.mul( pow( farOverNear, float( cz.add( int( 1 ) ) ).mul( 1.0 / NZ ) ) ).negate();

			const scaleNearX = zNearCluster.negate().mul( invFocalX );
			const scaleFarX = zFarCluster.negate().mul( invFocalX );
			const scaleNearY = zNearCluster.negate().mul( invFocalY );
			const scaleFarY = zFarCluster.negate().mul( invFocalY );

			const xMinNear = ndcXmin.mul( scaleNearX );
			const xMaxNear = ndcXmax.mul( scaleNearX );
			const xMinFar = ndcXmin.mul( scaleFarX );
			const xMaxFar = ndcXmax.mul( scaleFarX );

			const yMinNear = ndcYmin.mul( scaleNearY );
			const yMaxNear = ndcYmax.mul( scaleNearY );
			const yMinFar = ndcYmin.mul( scaleFarY );
			const yMaxFar = ndcYmax.mul( scaleFarY );

			// AABB of the 8 view-space corners (tile boundaries can straddle the view axis)
			const aabbMinX = min( xMinNear, xMinFar );
			const aabbMaxX = max( xMaxNear, xMaxFar );
			const aabbMinY = min( yMinNear, yMinFar );
			const aabbMaxY = max( yMaxNear, yMaxFar );

			const aabbMin = vec3( aabbMinX, aabbMinY, zFarCluster );
			const aabbMax = vec3( aabbMaxX, aabbMaxY, zNearCluster );

			// clear stale data from previous frame
			Loop( chunksPerCluster, ( { i } ) => {

				getClusterChunk( i ).assign( ivec4( 0 ) );

			} );

			const index = int( 0 ).toVar();

			// Z-culling: only test lights that can reach this cluster's Z-slice
			const zRange = textureLoad( this._zSliceRangesTexture, ivec2( cz, 0 ) );
			const rangeStart = int( zRange.x );
			const rangeEnd = int( zRange.y );

			Loop( this.maxLights, ( { i } ) => {

				const lightIdx = rangeStart.add( i );

				If( index.greaterThanEqual( int( maxLightsPerCluster ) ).or( lightIdx.greaterThanEqual( rangeEnd ) ), () => {

					Return();

				} );

				const { viewPosition, distance } = this.getLightData( lightIdx );

				// sphere-AABB intersection in view space
				const pos = viewPosition.xyz;
				const closest = max( aabbMin, min( pos, aabbMax ) );
				const diff = pos.sub( closest );
				const distSq = dot( diff, diff );

				If( distSq.lessThanEqual( distance.mul( distance ) ), () => {

					getClusterSlot( index ).assign( lightIdx.add( int( 1 ) ) );
					index.addAssign( int( 1 ) );

				} );

			} );

		} )().compute( clusterCount ).setName( 'Update Clustered Lights' );

		// shading-side: fragment → cluster index

		const getScreenClusterIndex = Fn( () => {

			const screenTile = screenCoordinate.div( tileSize ).floor();

			// view-space depth from positionView (negative in front); take magnitude
			const viewDepth = positionView.z.negate();

			// exponential Z slice: tz = floor( log(depth/near) / log(far/near) * NZ )
			const invLogFarOverNear = float( 1 ).div( log( this._cameraFar.div( this._cameraNear ) ) );
			const sliceFloat = log( viewDepth.div( this._cameraNear ) ).mul( invLogFarOverNear ).mul( float( NZ ) );
			const zSlice = clamp( sliceFloat.floor(), float( 0 ), float( NZ - 1 ) );

			return int( screenTile.x )
				.add( int( screenTile.y ).mul( int( NX ) ) )
				.add( int( zSlice ).mul( int( NX * NY ) ) );

		} );

		const screenClusterIndex = getScreenClusterIndex().toVar();

		// assigns

		this._bufferSize = bufferSize;
		this._lightIndexes = lightIndexes;
		this._screenClusterIndex = screenClusterIndex;
		this._compute = compute;

	}

	/**
	 * WebGL2 fallback: allocates the CPU cluster grid, its texture and the fragment-side cluster lookup.
	 */
	createCPU() {

		const { maxLights } = this;
		const NX = this.cpuGrid.x, NY = this.cpuGrid.y, NZ = this.cpuGrid.z;
		const W = CPU_TEX_WIDTH;
		const clusterCount = NX * NY * NZ;
		const tableRows = Math.ceil( ( clusterCount + 1 ) / W );
		// room for every cluster's list at its cap (offsets stay far below 2^24, so they are exact in a float)
		const listRows = Math.ceil( clusterCount * Math.min( this.maxLightsPerCluster, maxLights ) / W );

		const data = new Float32Array( W * ( tableRows + listRows ) );
		this._clusterTexture = new DataTexture( data, W, tableRows + listRows, RedFormat, FloatType );

		this._cpu = {
			NX, NY, NZ, tableRows,
			counts: new Uint16Array( clusterCount ), offsets: new Uint32Array( clusterCount ),
			zEdge: new Float64Array( NZ + 1 ), uEdge: new Float64Array( NX + 1 ), vEdge: new Float64Array( NY + 1 ),
			dx2: new Float64Array( NX ), dy2: new Float64Array( NY ),
			view: new Float64Array( maxLights * 5 ), pairs: new Int32Array( 1 << 15 ),
			fx: 0, fy: 0, ox: 0, oy: 0, stats: { lights: 0, pairs: 0 }
		};

		this._gridDimensions.value.set( NX, NY );

		// shading-side: fragment -> [ start, end ) of its cluster's list. The tile comes from the projected view
		// position rather than the fragment coordinate, so it matches the CPU grid for any target size / orientation.
		const getClusterRange = Fn( () => {

			const clip = this._cameraProjectionMatrix.mul( vec4( positionView, 1.0 ) );
			const tile = clamp( clip.xy.div( clip.w ).mul( 0.5 ).add( 0.5 ).mul( vec2( NX, NY ) ).floor(), vec2( 0, 0 ), vec2( NX - 1, NY - 1 ) );
			const slice = clamp( log( max( positionView.z.negate(), 1e-6 ) ).mul( this._sliceScale ).add( this._sliceBias ).floor(), float( 0 ), float( NZ - 1 ) );
			const index = int( tile.x ).add( int( tile.y ).mul( int( NX ) ) ).add( int( slice ).mul( int( NX * NY ) ) );

			const next = index.add( int( 1 ) );

			return ivec2(
				int( textureLoad( this._clusterTexture, ivec2( index.mod( int( W ) ), index.div( int( W ) ) ) ).x ),
				int( textureLoad( this._clusterTexture, ivec2( next.mod( int( W ) ), next.div( int( W ) ) ) ).x )
			);

		} );

		this._clusterRange = getClusterRange().toVar();

	}

	/**
	 * Frees the GPU resources allocated by this node.
	 */
	dispose() {

		if ( this._compute !== null ) this._compute.dispose();
		if ( this._clusterTexture !== null ) this._clusterTexture.dispose();

		this._lightsTexture.dispose();
		this._zSliceRangesTexture.dispose();

		super.dispose();

	}

}

export default ClusteredLightsNode;

/**
 * TSL function that creates a clustered lights node.
 *
 * @tsl
 * @function
 * @param {number} [maxLights=1024] - Maximum number of point lights.
 * @param {number} [tileSize=32] - Screen tile size in pixels.
 * @param {number} [zSlices=24] - Depth slice count.
 * @param {number} [maxLightsPerCluster=64] - Per-cluster light-list capacity.
 * @return {ClusteredLightsNode} The clustered lights node.
 */
export const clusteredLights = /*@__PURE__*/ nodeProxy( ClusteredLightsNode );
