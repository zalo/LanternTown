# Unit test of Cycles light units against three.js conventions (run once; town_cycles.py uses the measured constants).
#   blender --background --factory-startup --threads 14 --python tools/cyclesref/calib.py
# three.js: a light of irradiance E on a Lambert surface of albedo a gives radiance a * E / pi.
#   DirectionalLight: E = intensity * colour * N.L        PointLight (candela I): E = I * colour * N.L / d^2
import bpy, math, os, numpy as np
ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )

def render_value():

	sc = bpy.context.scene
	sc.render.filepath = os.path.join( ROOT, '.scratch', 'cyclesref', 'calib.exr' )
	sc.render.image_settings.file_format = 'OPEN_EXR'; sc.render.image_settings.color_depth = '32'
	bpy.ops.render.render( write_still = True )
	import OpenImageIO as oiio
	a = oiio.ImageBuf( sc.render.filepath ).get_pixels( oiio.FLOAT )
	h, w = a.shape[ :2 ]
	return a[ h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2, :3 ].mean( ( 0, 1 ) )

for o in list( bpy.data.objects ): bpy.data.objects.remove( o )
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 64; sc.cycles.use_denoising = False
sc.cycles.max_bounces = 0; sc.cycles.diffuse_bounces = 0
sc.render.resolution_x = sc.render.resolution_y = 32; sc.render.resolution_percentage = 100
sc.view_settings.view_transform = 'Standard'
bpy.ops.mesh.primitive_plane_add( size = 200 )
pl = bpy.context.object
m = bpy.data.materials.new( 'a' ); m.use_nodes = True
nt = m.node_tree; nt.nodes.clear()
d = nt.nodes.new( 'ShaderNodeBsdfDiffuse' ); d.inputs[ 0 ].default_value = ( 0.5, 0.5, 0.5, 1 )
o = nt.nodes.new( 'ShaderNodeOutputMaterial' ); nt.links.new( d.outputs[ 0 ], o.inputs[ 0 ] )
pl.data.materials.append( m )
cd = bpy.data.cameras.new( 'c' ); cd.lens = 200
cam = bpy.data.objects.new( 'c', cd ); cam.location = ( 0, 0, 10 ); sc.collection.objects.link( cam ); sc.camera = cam
w = bpy.data.worlds.new( 'w' ); w.use_nodes = True; sc.world = w
bg = w.node_tree.nodes[ 'Background' ]; bg.inputs[ 0 ].default_value = ( 0, 0, 0, 1 )

A = 0.5
sd = bpy.data.lights.new( 's', 'SUN' ); sd.energy = 1.0; sd.angle = math.radians( 1 )
so = bpy.data.objects.new( 's', sd ); sc.collection.objects.link( so )
r = render_value()[ 0 ]
print( 'CALIB sun: strength 1 -> radiance %.5f on albedo %.1f: irradiance per unit strength = %.4f (three.js: 1)' % ( r, A, r * math.pi / A ) )
bpy.data.objects.remove( so )

H = 3.0
pd = bpy.data.lights.new( 'p', 'POINT' ); pd.energy = 1000.0; pd.shadow_soft_size = 0.1
po = bpy.data.objects.new( 'p', pd ); po.location = ( 0, 0, H ); sc.collection.objects.link( po )
r = render_value()[ 0 ]
cd_per_w = r * math.pi / A * H * H / 1000.0
print( 'CALIB point: 1000 W at %.1f m -> radiance %.5f: candela per watt = %.6f, watts per candela = %.4f (4 pi^2 = %.4f, 4 pi = %.4f)' % ( H, r, cd_per_w, 1 / cd_per_w, 4 * math.pi ** 2, 4 * math.pi ) )
bpy.data.objects.remove( po )

bg.inputs[ 0 ].default_value = ( 1, 1, 1, 1 )
r = render_value()[ 0 ]
print( 'CALIB world: uniform radiance 1 -> radiance %.5f on albedo %.1f: irradiance = %.4f (pi = %.4f)' % ( r, A, r * math.pi / A, math.pi ) )

# can OpenImageIO apply Blender's AgX view transform?
import OpenImageIO as oiio, os
cfg = os.path.join( bpy.utils.system_resource( 'DATAFILES' ), 'colormanagement', 'config.ocio' )
print( 'CALIB ocio config', cfg, os.path.exists( cfg ) )
try:

	src = oiio.ImageBuf( oiio.ImageSpec( 4, 4, 3, oiio.FLOAT ) )
	oiio.ImageBufAlgo.fill( src, ( 0.18, 0.18, 0.18 ) )
	dst = oiio.ImageBufAlgo.ociodisplay( src, 'sRGB', 'AgX', fromspace = 'Linear Rec.709', colorconfig = cfg )
	print( 'CALIB agx of 0.18 ->', dst.get_pixels( oiio.FLOAT )[ 0, 0 ], dst.has_error and dst.geterror() )
	dst = oiio.ImageBufAlgo.ociodisplay( src, 'sRGB', 'Standard', fromspace = 'Linear Rec.709', colorconfig = cfg )
	print( 'CALIB standard of 0.18 ->', dst.get_pixels( oiio.FLOAT )[ 0, 0 ] )

except Exception as e: print( 'CALIB ocio failed', e )
