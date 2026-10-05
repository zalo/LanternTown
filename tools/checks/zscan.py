# emissive faces with another surface lying parallel within a few mm of them (depth-fight candidates), per built GLB
import bpy, sys, json, struct, collections
from mathutils import Vector
from mathutils.bvhtree import BVHTree
a = sys.argv[sys.argv.index('--') + 1:]
listing, out = a[0], a[1]
DEF = {'emissive': 8, 'neon': 14, 'glass_frosted': 3, 'glass_lantern': 0.5, 'glass_stained': 1.6, 'window_lit': 2.4}
SEE = ('glass_clear', 'glass_lantern', 'glass_frosted', 'glass_stained', 'water', 'water_fall')
R = {}
for path in json.load(open(listing)):
    f = open(path, 'rb'); f.read(12); n, _ = struct.unpack('<II', f.read(8)); g = json.loads(f.read(n)); f.close()
    info = {}
    for m in g.get('materials', []):
        e = m.get('extras', {}); lib = e.get('tsq_lib', '?'); prm = json.loads(e.get('tsq_params', '{}'))
        info[m['name']] = (lib, float(prm.get('glow', DEF.get(lib, 0))))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    V, F, FM, FN, FA = [], [], [], [], []
    for o in bpy.context.scene.objects:
        if o.type != 'MESH':
            continue
        me = o.data; mw = o.matrix_world; base = len(V)
        V.extend([mw @ v.co for v in me.vertices])
        n3 = mw.to_3x3().inverted().transposed()
        names = [m.name if m else '?' for m in me.materials] or ['?']
        for p in me.polygons:
            F.append([base + i for i in p.vertices]); FM.append(names[p.material_index]); FN.append((n3 @ p.normal).normalized()); FA.append(p.area)
    if not F:
        continue
    bvh = BVHTree.FromPolygons(V, F, all_triangles=False, epsilon=0.0)
    res = collections.defaultdict(float); where = {}
    for i in range(len(F)):
        lib, gl = info.get(FM[i], ('?', 0))
        if gl < 0.3 or FA[i] < 1e-5:
            continue
        vs = [V[k] for k in F[i]]
        c = sum(vs, Vector()) / len(vs)
        pts = [c] if FA[i] < 0.004 else [c] + [c * 0.4 + v * 0.6 for v in vs]
        wa = FA[i] / len(pts)
        for pt in pts:
            for sgn, side in ((1, 'front'), (-1, 'behind')):
                d = FN[i] * sgn
                hit = bvh.ray_cast(pt + d * 2e-5, d, 0.006)
                if hit[0] is None:
                    continue
                j = hit[2]
                if j == i or FM[j] == FM[i] or abs(FN[j].dot(FN[i])) < 0.9:
                    continue
                dist = hit[3] + 2e-5
                b = 'A' if dist < 0.0003 else 'B' if dist < 0.0015 else 'C' if dist < 0.003 else 'D'
                hl, hg = info.get(FM[j], ('?', 0))
                k = '|'.join((lib, '%.1f' % gl, side, hl + ('*' if hg >= 0.3 else ''), b))
                res[k] += wa
                w = where.setdefault(k, [])
                if len(w) < 4 and all((Vector(q) - pt).length > 0.3 for q in w):
                    w.append([round(x, 3) for x in pt])
    R[path] = {'rows': dict(res), 'where': where, 'glow_area': sum(FA[i] for i in range(len(F)) if info.get(FM[i], ('?', 0))[1] >= 0.3)}
json.dump(R, open(out, 'w'))
print('ZSCAN done', len(R))
