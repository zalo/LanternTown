# aggregate.py - medians and (min-max) per variant, scene and CPU throttle from the .jsonl of measure.py
#   python3 web/bakeglb/aggregate.py out/web/_nextglb/bench/results/pass1_quiet.jsonl
import json, sys, statistics as st
from collections import defaultdict
import os
P = os.environ.get('AGG') or (sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl')
L = [json.loads(l) for l in open(P)]
G = defaultdict(list)
bad = 0
for x in L:
    if not x.get('done') or x.get('errors') or x.get('adapter') != 'nvidia' or x.get('hidden'):
        bad += 1
        print('BAD', x.get('url'), x.get('driver', {}).get('rate'), str(x.get('errors'))[:200], x.get('done'), x.get('hidden'))
        continue
    x['driver']['gpu_delta'] = x['driver']['gpu_mb_after'] - x['driver']['gpu_mb_before']
    x['driver']['buf_delta'] = x['driver']['buffers_mb_peak'] - x['driver']['buffers_mb_end']
    k = (x['scene'], x['v'] + ('' if x['v'] != 'lod' else ':' + x['form'] + (':lazy' if x['lazy'] else ':eager')), x['driver']['rate'])
    G[k].append(x)
print('runs', len(L), 'bad', bad)

def get(x, path):
    for p in path.split('.'):
        if x is None: return None
        x = x.get(p) if isinstance(x, dict) else None
    return x

def cell(xs, path, scale=1, nd=0):
    v = [get(x, path) for x in xs]
    v = [a * scale for a in v if isinstance(a, (int, float))]
    if not v: return '-'
    m = st.median(v)
    f = '%.' + str(nd) + 'f'
    if min(v) == max(v): return f % m
    return (f + ' (' + f + '-' + f + ')') % (m, min(v), max(v))

MB = 1 / 1e6
order_v = ['packed', 'exact', 'portable', 'lod:glb:eager', 'lod:gltf:eager', 'lod:gltf:lazy', 'lod:ranged:lazy']
for scene in ['lo', 'near', 'fine', 'walk']:
    for rate in [1.0, 4.0]:
        rows = [(v, G[(scene, v, rate)]) for v in order_v if (scene, v, rate) in G]
        if not rows: continue
        print('\n== scene', scene, 'cpu x', rate)
        cols = [('n', None), ('loaded', 'marks.loaded'), ('first', 'marks.first'), ('all', 'marks.all'),
                ('req@first', 'at.first.geo_n'), ('wireMB@first', 'at.first.geo_wire'), ('rawMB@first', 'at.first.geo_raw'),
                ('req@all', 'at.all.geo_n'), ('wireMB@all', 'at.all.geo_wire'), ('rawMB@all', 'at.all.geo_raw'),
                ('long@all n', 'at.all.long.n'), ('long@all total', 'at.all.long.total'), ('longest', 'at.all.long.longest'),
                ('dec main ms', 'decode.main_ms'), ('dec worker busy', 'decode.worker_busy_ms'),
                ('heap peak', 'driver.heap_mb_peak'), ('buf peak', 'driver.buffers_mb_peak'), ('buf end', 'driver.buffers_mb_end'),
                ('gpu delta', 'driver.gpu_delta'), ('heap end', 'driver.heap_mb_end'), ('wall s', 'driver.wall_s'), ('textures ms', 'marks.textures'),
                ('draws', 'spawn.draws'), ('tris', 'spawn.tris'), ('resident tris', 'spawn.resident'), ('geom MB', 'spawn.geometry_mb'), ('shown', None)]
        if scene in ('near', 'fine'):
            cols += [('level ms', 'level.ms'), ('level n', 'level.n'), ('level wireMB', 'level.wire'), ('level rawMB', 'level.raw'),
                     ('level dec main', 'level.decode_main_ms'), ('level dec worker', 'level.decode_worker_busy_ms'),
                     ('level long n', 'level.long.n'), ('level long total', 'level.long.total'), ('level longest', 'level.long.longest')]
        if scene == 'walk':
            cols += [('walk fps', 'walk.fps'), ('cpu med', 'walk.cpu_ms.median'), ('cpu p95', 'walk.cpu_ms.p95'), ('cpu max', 'walk.cpu_ms.max'),
                     ('frame med', 'walk.frame_ms.median'), ('frame p95', 'walk.frame_ms.p95'), ('frame max', 'walk.frame_ms.max'),
                     ('walk long n', 'walk.long.n'), ('walk long total', 'walk.long.total'), ('walk longest', 'walk.long.longest'),
                     ('switches', 'walk.level_switches'), ('walk req', 'walk.geo_n'), ('walk wireMB', 'walk.geo_wire'), ('walk rawMB', 'walk.geo_raw'),
                     ('req@walked', 'at.walked.geo_n'), ('wireMB@walked', 'at.walked.geo_wire'), ('rawMB@walked', 'at.walked.geo_raw'),
                     ('end resident', 'end.resident'), ('end geom MB', 'end.geometry_mb'), ('end draws', 'end.draws')]
        for name, path in cols:
            out = []
            for v, xs in rows:
                if name == 'n': out.append(str(len(xs)))
                elif name == 'shown': out.append(str(get(xs[0], 'spawn.shown')))
                elif 'MB' in name and 'wire' in name or 'rawMB' in name: out.append(cell(xs, path, MB, 2))
                elif name in ('cpu med', 'cpu p95', 'cpu max', 'frame med', 'frame p95', 'frame max', 'walk fps', 'geom MB', 'end geom MB'): out.append(cell(xs, path, 1, 1))
                else: out.append(cell(xs, path))
            print('%-18s' % name + ' | ' + ' | '.join(out))
        print('%-18s' % 'variants' + ' | ' + ' | '.join(v for v, _ in rows))
