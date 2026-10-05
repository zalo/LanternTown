"""Seeded layout generator for the SECOND town ("town 2": the fountain square under the terrace, read off
assets/IMG_2863.jpeg) - host python, stdlib only. The Town machinery is a snapshot of town/layout.py (nothing is
imported from it); the plan (_generate) is this town's own.

    generate(seed=1, **overrides) -> dict   (written to out/town2/layout.json by TSQ_TOWN=town2 tools/town.py gen)

World frame = Blender: X east, Y north, Z up, metres. Assets face -Y; `rot` is degrees about Z.
Nothing is hand-placed: every item comes from a rule (rows, runs, rings, scatters, attachments) driven by
PARAMS and the seed; asset variants and seeds are drawn from the seeded stream.

Output:
  items    [{asset, variant, params, pos, rot, scale, collide, anchor?, on?, attach?, nav?, tag}]
  regions  walkable areas  {name, rect:[x0,y0,x1,y1], z}  or  {..., ramp:{axis,z0,z1}} / {..., arch:{axis,rise}}
  terrain  ground patches for tools/bl_terrain.py
  shots    review cameras, spawn, missing (assets that do not exist yet and were skipped)

Deferred placement (resolved by tools/town.py after the assets are built, from their real bounds / sockets):
  anchor "miny" | "maxy"   the item's local front / back bounding face (not its origin) is put on `pos`
  on     <item index>      stand on top of that item (z = its top)
  attach {to, role}        sit on a socket of another item
  nav    "stairs"          the item's footprint becomes a ramp region from its base to its top

Placement validation: every flight of stairs (+ STAIR_MARGIN) and every declared walking route (Town.route) is a
keep-out for ground props; bulky props also keep out of the review cameras' sight cones (Town.cover). One-off props
go through Town.place (sampled candidates, best valid one). Town.audit asserts the result at generate time and
check_resolved() repeats it in tools/town.py resolve against the real colliders of the built assets.
"""
import ast
import glob
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARAMS = {
    "seed": 1,
    "plaza_w": 25.0, "plaza_d": 23.0,       # lower fountain square (x, y): west fronts -> terrace wall, south fronts -> stair head
    "neck_w": 15.0, "neck_d": 14.0,          # the square's neck north to the hall, beside the terrace's north arm
    "court_d": 9.0,                          # hall court: north face of the terrace -> hall front
    "wall_x": 4.0, "wall_y": 1.0,            # where the terrace wall (x) and the head of the main stair (y) stand
    "terrace_h": 1.2, "terrace_w": 9.0,      # raised terrace: height (the picture's is a low wall with benches against it), width (wall face -> shop fronts)
    "stair_w": 4.38,                         # clear width of the main flight
    "street_w": 7.0, "street_len": 14.0,     # arrival street from the south, under the terrace wall
    "hall_w": 40.0, "hall_scale": 1.3,       # the clock tower hall: width in the plan, and how much larger than built it stands
    "fountain_d": 4.6,
    "row_depth": 10.0,                       # west row (inn / hotel) depth
    "bld_depth": 8.0,
    "keep_bearing": 48.0,                    # the keep behind the terrace houses: bearing from the reference viewpoint (deg, 0 = east)
    "far_landmark": True, "far_bearing": 30.0,     # a far castle on the skyline: on / off, bearing from the square (deg, 0 = east)
    "density": 1.0,                          # multiplies scatter counts
    "storeys_bias": 0,
}


# ------------------------------------------------------------------------------------------ asset info

_SPEC = {}
_VARIANT = {}          # asset -> {variant name: param overrides} (when VARIANTS is a literal)
STAIR_MARGIN = 0.6     # props stay this far from a flight of stairs
ASPECT = 16.0 / 9.0    # review frames
EYE = 1.65

# things that are walked over (never an obstacle), and the plan radius / height of the bulky props: those are the
# ones that can block a review frame, so they are tested against the cameras' sight cones
FLAT = {"drain_cover", "paving_patch", "pavement_weeds", "kerb_gutter", "tree_grate", "pigeons", "cellar_hatch"}
# bulky props: (plan radius, height, clear height under a canopy) - tested against the sight cones of the review shots
BULK = {"newspaper_kiosk": (2.05, 5.5, 0.0), "well_stone": (1.1, 2.5, 0.0), "bandstand": (2.5, 6.4, 0.0), "market_stall": (1.9, 2.8, 0.0),
        "street_clock": (0.45, 3.95, 0.0), "signpost_finger": (0.8, 3.1, 0.0), "parasol": (1.5, 2.7, 1.9), "pergola": (2.7, 2.8, 0.0),
        "hand_cart": (1.2, 2.3, 0.0), "statue_figure": (1.0, 4.2, 0.0), "water_pump": (0.8, 1.5, 0.0), "topiary": (0.4, 2.1, 0.0),
        "planter_box": (0.45, 1.8, 0.0), "planter_urn": (0.4, 1.8, 0.0), "birdbath_sundial": (0.35, 1.2, 0.0), "post_box": (0.35, 1.5, 0.0),
        "litter_bin": (0.35, 1.2, 0.0)}


def asset_path(aid):
    for base in filter(None, [os.environ.get("TSQ_ASSET_DIR"), "assets"]):   # released copy first (tools/town.py release)
        hits = glob.glob(os.path.join(ROOT, base, "*", aid + ".py"))
        if hits:
            return hits[0]
    return None


def released(aid):
    """True when the asset is in the pinned release (TSQ_ASSET_DIR): a working file its author has not released yet
    does not count (asset_path finds those too). Without a release directory the working tree is the release."""
    base = os.environ.get("TSQ_ASSET_DIR") or "assets"
    return bool(glob.glob(os.path.join(ROOT, base, "*", aid + ".py")))


def asset_spec(aid):
    """(PARAMS schema, [variant names]) parsed statically from the asset source ({} / [] when not a literal)."""
    if aid in _SPEC:
        return _SPEC[aid]
    params, variants = {}, []
    p = asset_path(aid)
    if p:
        try:
            for node in ast.parse(open(p).read()).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    n = node.targets[0].id
                    if n in ("PARAMS", "VARIANTS"):
                        try:
                            v = ast.literal_eval(node.value)
                        except Exception:
                            v = {}
                            if isinstance(node.value, ast.Dict):   # keep at least the keys
                                for k in node.value.keys:
                                    if isinstance(k, ast.Constant):
                                        v[k.value] = {}
                        if n == "PARAMS":
                            params = v
                        else:
                            variants = list(v)
                            _VARIANT[aid] = v if isinstance(v, dict) else {}
        except SyntaxError:
            pass
    _SPEC[aid] = (params, variants)
    return _SPEC[aid]


SEED_POOL = {"building": 0, "landmark": 0, "backdrop": 0, "structure": 3, "vegetation": 3}
# loose goods and accents the eye picks out when they repeat: each instance takes a proto (variant + seed) that no review
# sight cone it stands in has shown yet, the one least used in the town (Town.vary); one asset is the foreground accent of
# one shot only; and every instance is scaled a little differently
ACCENT = {"milk_churns", "crate_stack", "sack_pile", "barrel", "bucket_set", "flower_pots", "planter_box", "planter_urn", "luggage_stack",
          "hand_cart", "wheelbarrow", "bicycle", "ladder", "litter_bin", "cat", "pigeons"}
ACCENT_SEEDS = 3       # seeds per variant of an accent asset
CONE = 30.0            # depth of a review sight cone for that rule (m)
FOREGROUND = 7.5       # ... and of its foreground
_CLS = {}
_MOUNT = {}


def asset_class(aid):
    if not _CLS:
        try:
            import json
            c = json.load(open(os.path.join(ROOT, "docs", "catalog.json")))
            for a in (c["assets"] if isinstance(c, dict) else c):
                _CLS[a["id"]] = a["class"]
                _MOUNT[a["id"]] = a.get("mount", "ground")
        except Exception:
            _CLS["?"] = "prop"
    return _CLS.get(aid, "prop")


def asset_mount(aid):
    asset_class(aid)
    return _MOUNT.get(aid, "ground")


def build_name(it):
    """file stem of an item's build (must match tools/town.py item_name)"""
    import hashlib
    import json
    return "_" + hashlib.sha1(json.dumps([it["asset"], it.get("variant"), it.get("params") or {}], sort_keys=True).encode()).hexdigest()[:10]


def sidecar(it):
    """the build record of an item (bounds, sockets) when it has been built already, else None"""
    import json
    p = os.path.join(ROOT, "out", "glb", it["asset"], build_name(it) + ".json")
    try:
        return json.load(open(p))
    except Exception:
        return None


# ------------------------------------------------------------------------------------------ built geometry
# What an asset really occupies is read from its built GLB (not from its bounding box): the surface is sampled on a
# CELL grid in four height bands and the connected pieces of each band become plan boxes. A lamp post is then a
# thin column with a wide head, a parasol a pole under a canopy, a building its body plus every stoop, step, bay
# and jetty that stands proud of it. Cached in out/town2/geom.json (keyed by build + file time).
BANDS = ((0.12, 0.6), (0.6, 1.7), (1.7, 3.0), (3.0, 4.2), (4.2, 5.6), (5.6, 99.0))
SKIN = 0.1             # plinths, sills and mouldings this proud of a building's body count as the body
CELL = 0.2
TOL = 0.05             # two things may touch / sink into each other this far before it counts as an intersection
GEO_V = 3              # format of the cached measurements
_GEO = {"data": None, "dirty": False}
GEO_PATH = os.path.join(ROOT, "out", "town2", "geom.json")
# never an obstacle: growth on walls, strings of lights / flags, things lying flat (FLAT, below)
# never tested for intersections: growth and garlands that lie on / hang between other things, ground litter,
# flights of stairs (a keep-out of their own) and what floats
SOFT = {"ivy_patch", "vine_wisteria", "festoon_lights", "bunting", "stairs_plaza", "rowboat", "hanging_basket", "pavement_weeds", "pigeons",
        "paving_patch"}
STOOP = 0.25           # a part at the foot of a building at least this big both ways is a keep-out for props (Town.blocked)
STEP_OFF = (0.0, 0.2, 0.4)   # Town.line / Town.bays: how far a piece steps off its wall / edge until it stands clear
SLIDE = (0.0, 0.22, -0.22, 0.4, -0.4)   # Town.bays: ... and how far (of a bay) it moves along the edge when something stands in its bay
LEAN = 0.35            # a wall-mounted / wall-leaning piece may sit this deep in the wall it is put against
WALLS = {"retaining_wall", "wall_stone_run", "canal_wall", "balustrade"}


def _glb_prims(path):
    """[(material name, xs, ys, zs, indices)] of a GLB in the asset frame (X east, Y north, Z up)"""
    import json
    import struct
    from array import array
    with open(path, "rb") as f:
        data = f.read()
    jl = struct.unpack_from("<I", data, 12)[0]
    js = json.loads(data[20:20 + jl])
    b0 = 20 + jl + 8

    def acc(i, fmt, n):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        off = b0 + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        arr = array(fmt)
        size = arr.itemsize * n * a["count"]
        st = bv.get("byteStride") or arr.itemsize * n
        if st == arr.itemsize * n:
            arr.frombytes(data[off:off + size])
        else:
            for k in range(a["count"]):
                arr.frombytes(data[off + k * st:off + k * st + arr.itemsize * n])
        return arr

    def mat(n):
        if "matrix" in n:
            m = n["matrix"]
            return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
        t = n.get("translation", [0, 0, 0])
        x, y, z, w = n.get("rotation", [0, 0, 0, 1])
        s = n.get("scale", [1, 1, 1])
        R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        return [[R[r][0] * s[0], R[r][1] * s[1], R[r][2] * s[2], t[r]] for r in range(3)] + [[0, 0, 0, 1]]

    def mul(a, b):
        return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]

    out = []
    ident = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]

    def walk(ni, M):
        n = js["nodes"][ni]
        M = mul(M, mat(n))
        if "mesh" in n:
            for pr in js["meshes"][n["mesh"]]["primitives"]:
                if pr.get("mode", 4) != 4 or "POSITION" not in pr["attributes"]:
                    continue
                p = acc(pr["attributes"]["POSITION"], "f", 3)
                gx, gy, gz = p[0::3], p[1::3], p[2::3]
                if any(abs(M[r][c] - ident[r][c]) > 1e-9 for r in range(3) for c in range(3)):
                    gx, gy, gz = ([M[r][0] * a + M[r][1] * b + M[r][2] * c + M[r][3] for a, b, c in zip(gx, gy, gz)] for r in range(3))
                else:
                    gx, gy, gz = [a + M[0][3] for a in gx], [a + M[1][3] for a in gy], [a + M[2][3] for a in gz]
                if "indices" in pr:
                    ct = js["accessors"][pr["indices"]]["componentType"]
                    idx = acc(pr["indices"], {5121: "B", 5123: "H", 5125: "I"}[ct], 1)
                else:
                    idx = range(len(gx))
                name = js["materials"][pr["material"]].get("name", "") if "material" in pr else ""
                out.append((name, gx, [-a for a in gz], gy, idx))      # glTF (x, y, z) = asset (x, z, -y)
        for c in n.get("children", []):
            walk(c, M)

    for ni in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(ni, ident)
    return out


def _slices(path, nominal=(), only=None, top=len(BANDS)):
    """Plan boxes of a built asset: [[band, z0, z1, x0, y0, x1, y1]] - per height band the connected pieces of its
    surface (triangles sampled on the CELL grid). nominal = plan rects that are solid anyway (a building's body):
    only what stands outside them is reported. only = material prefix (a tree's bark). top = number of bands."""
    cells = [dict() for _ in BANDS]
    edges = [b[0] for b in BANDS]

    def inside(x, y):
        for (x0, y0, x1, y1) in nominal:
            if x0 - SKIN <= x <= x1 + SKIN and y0 - SKIN <= y <= y1 + SKIN:
                return True
        return False

    def add(x, y, z):
        b = 0
        while b + 1 < len(edges) and z >= edges[b + 1]:
            b += 1
        if b >= top:
            return
        k = (int(x // CELL), int(y // CELL))
        c = cells[b].get(k)
        if c is None:
            cells[b][k] = [x, y, x, y, z, z]
        else:
            if x < c[0]:
                c[0] = x
            elif x > c[2]:
                c[2] = x
            if y < c[1]:
                c[1] = y
            elif y > c[3]:
                c[3] = y
            if z < c[4]:
                c[4] = z
            elif z > c[5]:
                c[5] = z

    zmin = edges[0]
    for (name, xs, ys, zs, idx) in _glb_prims(path):
        if only and not name.startswith(only):
            continue
        keep = [z >= zmin and not (nominal and inside(x, y)) for x, y, z in zip(xs, ys, zs)]
        for x, y, z, k in zip(xs, ys, zs, keep):
            if k:
                add(x, y, z)
        it = iter(idx)
        for a, b, c in zip(it, it, it):
            xa, xb, xc = xs[a], xs[b], xs[c]
            ya, yb, yc = ys[a], ys[b], ys[c]
            za, zb, zc = zs[a], zs[b], zs[c]
            ex = max(xa, xb, xc) - min(xa, xb, xc)
            ey = max(ya, yb, yc) - min(ya, yb, yc)
            ez = max(za, zb, zc) - min(za, zb, zc)
            if (ex < CELL and ey < CELL and ez < 0.3) or max(za, zb, zc) < zmin:
                continue
            if nominal and not (keep[a] or keep[b] or keep[c]) and ez < 0.3:
                continue
            n = int(min(48, math.ceil(max(ex, ey, ez) / 0.1)))
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    u, v = i / n, j / n
                    w = 1.0 - u - v
                    z = za * w + zb * u + zc * v
                    if z < zmin:
                        continue
                    x, y = xa * w + xb * u + xc * v, ya * w + yb * u + yc * v
                    if nominal and inside(x, y):
                        continue
                    add(x, y, z)
    out = []
    for b, cc in enumerate(cells):
        todo = set(cc)
        while todo:
            k0 = todo.pop()
            stack, comp = [k0], [k0]
            while stack:
                i, j = stack.pop()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        k = (i + di, j + dj)
                        if k in todo:
                            todo.discard(k)
                            stack.append(k)
                            comp.append(k)
            # a piece that fills its bounding box is one box; an L, U or ring is cut into strips of cells
            i0, i1 = min(k[0] for k in comp), max(k[0] for k in comp)
            j0, j1 = min(k[1] for k in comp), max(k[1] for k in comp)
            rects = [comp]
            if len(comp) < 0.6 * (i1 - i0 + 1) * (j1 - j0 + 1) and (nominal or len(comp) > 30):
                have, runs = set(comp), {}
                for j in range(j0, j1 + 1):
                    i = i0
                    while i <= i1:
                        if (i, j) in have:
                            a = i
                            while (i + 1, j) in have:
                                i += 1
                            runs.setdefault((a, i), []).append(j)
                        i += 1
                rects = []
                for (a, e), js_ in runs.items():      # rows with the same run of cells, stacked while adjacent
                    grp = [js_[0]]
                    for j in js_[1:] + [None]:
                        if j is not None and j == grp[-1] + 1:
                            grp.append(j)
                            continue
                        rects.append([(i, jj) for i in range(a, e + 1) for jj in grp])
                        grp = [j]
                if len(rects) > 24:
                    rects = [comp]
            for rc in rects:
                c = [cc[k] for k in rc]
                x0, y0, x1, y1 = min(q[0] for q in c), min(q[1] for q in c), max(q[2] for q in c), max(q[3] for q in c)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2     # reach from the centre: a round piece is smaller than its box
                rad = max(math.hypot(max(q[2] - mx, mx - q[0]), max(q[3] - my, my - q[1])) for q in c)
                out.append([b, round(min(q[4] for q in c), 3), round(max(q[5] for q in c), 3), round(x0, 3), round(y0, 3), round(x1, 3),
                            round(y1, 3), round(rad + 0.005, 3)])
    return out


def _radial(path, only, top=2.6, step=0.1):
    """Reach of a trunk from the asset's axis per `step` of height: [max radius in slice k] (0 where it has no surface)"""
    prof = [0.0] * int(round(top / step))
    for (name, xs, ys, zs, idx) in _glb_prims(path):
        if only and not name.startswith(only):
            continue
        for x, y, z in zip(xs, ys, zs):
            k = int(z / step) if z >= 0 else 0
            if k < len(prof):
                r = math.hypot(x, y)
                if r > prof[k]:
                    prof[k] = r
    for k in range(len(prof) - 2, -1, -1):           # a slice between two rings of the mesh: at least the one above
        if prof[k] == 0.0:
            prof[k] = prof[k + 1]
    return [round(r, 3) for r in prof]


def _bore(path, only=("iron", "metal"), deck=0.12):
    """The clear opening of a ring-shaped piece around its axis: [radius of the hole in its plate (below `deck`),
    inner radius of what stands on it (None = nothing), height of that]"""
    hole, cage, top = None, None, 0.0
    for (name, xs, ys, zs, idx) in _glb_prims(path):
        if not name.startswith(only):
            continue
        for x, y, z in zip(xs, ys, zs):
            r = math.hypot(x, y)
            if z < deck:
                hole = r if hole is None else min(hole, r)
            else:
                cage = r if cage is None else min(cage, r)
                top = max(top, z)
    return [round(hole or 0.0, 3), None if cage is None else round(cage, 3), round(top, 3)]


def geo(it, kind="boxes"):
    """Measured geometry of an item's build, None until it is built: kind "boxes" -> _slices (a building's parts
    outside its body), "trunk" -> _radial of the bark."""
    import json
    name = build_name(it)
    p = os.path.join(ROOT, "out", "glb", it["asset"], name + ".glb")
    try:
        mt = int(os.path.getmtime(p))
    except OSError:
        return None
    if _GEO["data"] is None:
        try:
            _GEO["data"] = json.load(open(GEO_PATH))
        except Exception:
            _GEO["data"] = {}
    mode = it.get("collide", "auto") if it.get("collide") in ("building", "passage") else "auto"
    key = f"{GEO_V}/{it['asset']}/{name}/{mode if kind == 'boxes' else kind}"
    rec = _GEO["data"].get(key)
    if rec and rec["t"] == mt:
        return rec["v"]
    sc = sidecar(it)
    if sc is None:
        return None
    try:
        if kind == "trunk":
            v = _radial(p, "bark")
        elif kind == "bore":
            v = _bore(p)
        else:
            veg = asset_class(it["asset"]) == "vegetation"
            bark = veg and any(m.startswith("bark") for m in (sc.get("stats") or {}).get("materials", {}))
            v = _slices(p, nominal=nominal(it, sc), only="bark" if bark else None, top=3 if bark else 2 if veg else len(BANDS))
    except Exception as e:                           # an unreadable GLB is reported by the audit, not swallowed
        v = {"error": str(e)[:200]}
    _GEO["data"][key] = {"t": mt, "v": v}
    _GEO["dirty"] = True
    return v


def geo_save():
    if _GEO["dirty"]:
        import json
        os.makedirs(os.path.dirname(GEO_PATH), exist_ok=True)
        with open(GEO_PATH + ".tmp", "w") as f:
            json.dump(_GEO["data"], f)
        os.replace(GEO_PATH + ".tmp", GEO_PATH)
        _GEO["dirty"] = False


def nominal(it, sc):
    """the plan rects of a building's body in its own frame (what tools/town.py turns into its colliders)"""
    mode = it.get("collide")
    if mode not in ("building", "passage"):
        return []
    b0, b1 = sc["bounds"]["min"], sc["bounds"]["max"]
    P = it.get("params") or {}
    w, d = float(P.get("width", b1[0] - b0[0])), float(P.get("depth", b1[1] - b0[1]))
    if mode == "building":
        return [(-w / 2, 0.0, w / 2, d)]
    pw = float(P.get("passage_width", 3.0))
    return [(-w / 2, -d / 2, -pw / 2, d / 2), (pw / 2, -d / 2, w / 2, d / 2)]


def ring_fit(bore, prof, gs=1.0, ts=1.0, gap=0.03, play=0.08):
    """By how much a trunk (reach profile per 0.1 m, scale ts) is too thick for a ring-shaped piece around it
    (bore = [hole radius, inner radius of the guard | None, guard height], scale gs): the hole must clear the foot of
    the trunk by `gap`, the guard the trunk over its whole height by `play`. <= 0: it fits."""
    need = prof[0] * ts + gap - bore[0] * gs
    if bore[1] is not None:
        top = int(min(len(prof) - 1, (bore[2] * gs / ts + 0.1) / 0.1))
        need = max(need, max(prof[1:top + 1] or [0.0]) * ts + play - bore[1] * gs)
    return need


def pen(A, B):
    """How deep two turned plan boxes (cx, cy, hx, hy, rot deg) sit in each other (0 = apart): the smallest overlap
    over the four separating axes."""
    best = 1e9
    dx, dy = B[0] - A[0], B[1] - A[1]
    for a in (A[4], B[4]):
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        for (ux, uy) in ((c, s), (-s, c)):
            ra = sum(h * abs(ux * ex + uy * ey) for h, (ex, ey) in ((A[2], _ax(A[4], 0)), (A[3], _ax(A[4], 1))))
            rb = sum(h * abs(ux * ex + uy * ey) for h, (ex, ey) in ((B[2], _ax(B[4], 0)), (B[3], _ax(B[4], 1))))
            o = ra + rb - abs(dx * ux + dy * uy)
            if o <= 0:
                return 0.0
            best = min(best, o)
    return best


def _ax(deg, k):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return (c, s) if k == 0 else (-s, c)


def stair_plan(variant, width, rise):
    """(plan width, run) of a stairs_plaza flight, from the asset's own step rule (a little generous)"""
    schema, _ = asset_spec("stairs_plaza")
    over = _VARIANT.get("stairs_plaza", {}).get(variant or "", {})

    def val(k, d):
        v = over.get(k, schema.get(k, {}).get("default", d) if isinstance(schema.get(k), dict) else d)
        return d if v is None else v
    n = int(round(val("steps", 10)))
    n = max(int(math.ceil(rise / 0.185 - 1e-6)), min(n, int(math.floor(rise / 0.125 + 1e-6))), 2)
    run = n * val("going", 0.34) + (val("landing_depth", 1.3) if val("landing", False) and n >= 5 else 0.0) + 0.3
    return width + 2 * val("flare", 0.0) + 0.1, run


def rect_of(x, y, rot, lx0, ly0, lx1, ly1):
    """world AABB of a local rect of an item at (x, y) turned by rot"""
    pts = [rot2(lx, ly, rot) for lx in (lx0, lx1) for ly in (ly0, ly1)]
    return (x + min(p[0] for p in pts), y + min(p[1] for p in pts), x + max(p[0] for p in pts), y + max(p[1] for p in pts))


def rot2(x, y, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return (x * c - y * s, x * s + y * c)


def facing_rot(fx, fy):
    """rot (deg) that turns an asset's front (-Y) towards the world direction (fx, fy)"""
    return round(math.degrees(math.atan2(fx, -fy)), 4)


PROP_CLS = {"prop", "prop_small", "furniture", "street_furniture", "vegetation", "structure", "landmark"}
# never tested as loose props: the fabric of the town (walls, flights, water's edge), wall growth and roof pieces
FIXED = {"balustrade", "stairs_plaza", "retaining_wall", "iron_gate", "gate_pier", "canal_wall", "external_stair", "wall_fountain",
         "tunnel_mouth", "hedge_run", "rowboat", "table_lantern", "weathervane", "roof_finial", "ivy_patch", "vine_wisteria",
         "fountain_tiered", "bridge_stone", "gate_arch_town", "clock_tower_hall", "monument_star", "civic_hall_grand", "tower_keep_ornate"}
def LAMP_CD(span):
    """candela of ONE lantern standing in a space `span` m wide (street, terrace, quay): sized so its pool dies on
    the paving and the foot of the walls - the upper walls stay moonlit. The open square takes the asset's own output."""
    return round(max(22.0, min(60.0, 3.4 * span)), 1)


LAMP_ONE = ["default", "plinth", "short_plain"]   # lamp_post_gas with one lantern (a narrow space gets one head per post)
SPILL = 22.0           # backdrop rows: candela of a street lamp behind the town's own walls (it lights its own row only)
TABLE_CD = 1.6         # candle lantern on a cafe table: lights the cloth and the faces round it, no more
BANNER_H = 2.8         # a banner hangs from a lamp post's bracket only when the bracket is this high (its hem clears a head)
BANNERS = ["crimson_fleur", "green_key_long", "purple_star_dagged"]
SIGHT_PUT = 0.14       # a bulky prop put by a general rule may hide at most this fraction of a review frame
SIGHT_ONE = 0.05       # ... and a one-off (Town.place) this much, unless the rule says otherwise
LENS = 3.0             # nothing loose stands this close in front of a review camera
NEAR = 6.0             # near field of a review camera: a standing piece this close is wholly in the frame or wholly out of it


FOOT = 0.07            # ... and what stands in the picture below eye height keeps its foot this far (of the frame height) above the bottom edge


def cut_by(cam, x, y, z, r, h, pts=None):
    """True when the frame of a review camera {pos, target, fov} would cut a standing piece (plan radius r - or the
    plan outline pts of the piece as built - from z up to z + h) in its near field (< NEAR from the lens): across a
    side edge of the frame, or - for what rises above the eye - across its top (a lamp post whose shaft fills the
    edge of the picture, a bicycle halved by the corner). What the bottom edge cuts is ordinary foreground as long as
    most of it shows: a low piece of which only the top pokes into the picture (a saddle and a handlebar) is cut too,
    and so is any piece below eye height whose foot the bottom edge cuts off (it stands FOOT of the frame's height
    clear of that edge, or under the frame altogether)."""
    px, py, pz = cam["pos"]
    if abs(z - (pz - EYE)) > 1.0 or abs(x - px) > NEAR + 6.0 or abs(y - py) > NEAR + 6.0:
        return False                                  # (the near field is the ground the camera stands on)
    if not pts:
        pts = [(x + r * math.cos(k * math.pi / 4), y + r * math.sin(k * math.pi / 4)) for k in range(8)]
    ds = [math.hypot(qx - px, qy - py) for qx, qy in pts]
    d = min(ds + [math.hypot(x - px, y - py)])
    if d > NEAR + 6.0:
        return False
    if d < 0.9:
        return True
    near = d <= NEAR
    tx, ty, tz = cam["target"]
    L = math.hypot(tx - px, ty - py) or 1.0
    ux, uy = (tx - px) / L, (ty - py) / L
    vf = math.radians(cam["fov"])
    hf = 2 * math.atan(math.tan(vf / 2) * ASPECT)
    bs, front, back, far, fore = [], NEAR, 0.0, None, None
    for (qx, qy), dq in zip(pts, ds):
        along, lat = (qx - px) * ux + (qy - py) * uy, (qy - py) * ux - (qx - px) * uy
        if along > 0.2:
            b = math.atan2(lat, along)
            bs.append(b)
            if abs(b) < hf / 2:
                far = (along, lat) if dq > back else far
                fore = (along, lat) if fore is None or along < fore[0] else fore
                front, back = min(front, dq), max(back, dq)
        else:
            bs.append(math.copysign(math.pi / 2, lat))
    if min(bs) >= hf / 2 or max(bs) <= -hf / 2 or all(abs(b) >= math.pi / 2 for b in bs):
        return False                                  # wholly out of the frame / behind the lens
    pitch = math.atan2(tz - pz, L)
    dn, df = (max(0.5, front), max(0.5, back)) if back > 0.0 else (max(0.5, d), max(0.5, d))
    e0, e1 = math.atan2(z - pz, dn) - pitch, math.atan2(z + h - pz, dn) - pitch       # its near edge: foot and top
    if (e1 <= -vf / 2 and math.atan2(z + h - pz, df) - pitch <= -vf / 2) or e0 >= vf / 2:
        return False                                  # under the frame, far edge and all / over it
    if 0.5 <= h <= EYE and far:                       # its far edge, foot and top, in the picture plane
        cp, sp, t = math.cos(pitch), math.sin(pitch), math.tan(vf / 2)
        f0, f1 = [((zz - pz) * cp - far[0] * sp) / max(0.1, far[0] * cp + (zz - pz) * sp) for zz in (z, z + h)]
        if -t < f1 < t and f1 + t < 0.66 * (f1 - f0):
            return True                               # a sliver over the bottom edge
    if h <= EYE and fore:                             # its near edge: the foot under (or on) the bottom edge, the top in the picture
        cp, sp, t = math.cos(pitch), math.sin(pitch), math.tan(vf / 2)
        n0, n1 = [((zz - pz) * cp - fore[0] * sp) / max(0.1, fore[0] * cp + (zz - pz) * sp) for zz in (z, z + h)]
        if n1 > -t and n0 < -t + FOOT * 2 * t:
            return True                               # stands on the bottom edge: its foot is cut off
    return near and (min(bs) < -hf / 2 or max(bs) > hf / 2 or (h > EYE and e1 > vf / 2) or front < LENS)   # ... or right at the lens
OWN = {("hotel", "sign")}   # sockets that mark a part the building already carries (no param to tell)


class Town:
    def __init__(self, seed, P):
        self.P = P
        self.seed = seed
        self.items = []
        self.regions = []
        self.terrain = []
        self.shots = []
        self.lamps = []
        self.yields = set()      # items that give way to the review frames (loose props, one-offs a rule can do without)
        self.cuts = []           # audit: props a review frame cuts in its near field [shot, asset, tag]
        self.crowns = []         # tree canopies (x, y, z, radius): tall slim pieces (clocks, signposts) keep out from under them
        self.views = []          # review viewpoints (x, y) from the pass before: tree canopies keep clear of them
        self.shown, self.used, self.accents = {}, {}, {}   # Town.vary: protos shown per sight cone / used in all / shots an asset fronts
        self.cams = []           # review cameras of the pass before [{name, pos, target, fov}]: sight cones for bulky props
        self.missing = {}
        self.blockouts = []    # footprints of buildings whose asset does not exist yet (grey massing in the runtime)
        self.occ = []          # (x, y, r, z) occupied discs for scatter spacing
        self.stairs = []       # (name, x0, y0, x1, y1, z0, z1) plan footprints of the flights of stairs
        self.routes = []       # (name, x0, y0, x1, y1, z) walking routes: ground props keep off them
        self.dropped = []      # props a rule found no valid spot for (reported, not placed)
        self.nudged = []       # props a rule had to move off a route / flight / out of a review frame
        self.unbuilt = 0       # socket look-ups that failed because the asset has not been built yet
        self.known, self.seen = {}, {}   # measures of the pass before / of this pass (generate)
        self.keep = []         # (name, x0, y0, x1, y1, z) plan rects of what stands out at the foot of a building (stoops, steps)
        self.part = {}         # item index -> index of the piece it belongs to (chair -> table, basket -> lamp): they may touch
        self.rings = []        # (ring item, item standing in it): a grate / guard around a trunk, checked by Town.audit
        self.clashed = []      # pieces a rule left out because they would have stood in something else
        self._solid = {}
        self._undo = {}
        self._n = 0

    # ---------------------------------------------------------------- random
    def rng(self, *key):
        return random.Random(f"{self.seed}/" + "/".join(str(k) for k in key))

    # ---------------------------------------------------------------- keep-outs: stairs, walking routes, sight cones
    def route(self, name, x0, y0, x1, y1, z=0.0):
        """Declare a walking route: no ground prop may stand on this rect (at this level)."""
        self.routes.append((name, round(min(x0, x1), 3), round(min(y0, y1), 3), round(max(x0, x1), 3), round(max(y0, y1), 3), round(z, 3)))

    def stair(self, idx, name):
        """Register the plan footprint of a flight of stairs (item idx, back edge anchored on its pos) as a keep-out.
        Uses the real bounds once the flight is built, the asset's own step rule before that. Returns (rect, run)."""
        it = self.items[idx]
        x, y, z = it["pos"]
        rise = it["params"].get("rise", 1.0)
        sc = sidecar(it)
        if sc:
            b0, b1 = sc["bounds"]["min"], sc["bounds"]["max"]
            lx0, lx1, run = b0[0], b1[0], b1[1] - b0[1]
        else:
            w, run = stair_plan(it["variant"], it["params"].get("width", 3.0), rise)
            lx0, lx1 = -w / 2, w / 2
        rect = tuple(round(v, 3) for v in rect_of(x, y, it["rot"], lx0, -run, lx1, 0.0))
        self.stairs.append((name,) + rect + (round(z, 3), round(z + rise, 3)))
        return rect, run

    def blocked(self, x, y, r, z=0.0):
        """Name of the flight of stairs (+ STAIR_MARGIN), walking route or foot of a building (stoop, steps: Town.keep)
        a disc (x, y, r) standing at level z would intrude on; None when it is clear of all of them."""
        m = STAIR_MARGIN + r
        for (n, x0, y0, x1, y1, z0, z1) in self.stairs:
            if z0 - 0.6 < z < z1 + 0.6 and x0 - m < x < x1 + m and y0 - m < y < y1 + m:
                return "stairs_" + n
        for (n, x0, y0, x1, y1, rz) in self.routes:
            if abs(rz - z) < 1.0 and x0 - r < x < x1 + r and y0 - r < y < y1 + r:
                return n
        r = min(r, 0.45)                                # the foot of the piece: its exact shape is tested by Town.hits
        for (n, x0, y0, x1, y1, rz) in self.keep:
            if abs(rz - z) < 1.0 and x0 - r < x < x1 + r and y0 - r < y < y1 + r:
                return n
        return None

    # ---------------------------------------------------------------- built geometry: what stands in what
    def solid_of(self, it):
        """World boxes of an item as built [(cx, cy, hx, hy, rot, z0, z1, reach, body)]: the plan boxes of its parts
        per height band (geo; a backdrop row: its masses from 3 m up) + a building's body. [] for soft things,
        stacked pieces (the resolver poses them) and whatever is not built yet."""
        a = it["asset"]
        if a in SOFT or "on" in it or asset_class(a) == "ground":
            return []
        g, sc = geo(it), sidecar(it)
        if not isinstance(g, list) or sc is None:
            return []
        s, rot = it.get("scale", 1.0), it["rot"]
        x, y, z = it["pos"]
        b0, b1 = sc["bounds"]["min"], sc["bounds"]["max"]
        ox = -(b0[0] + b1[0]) / 2 * s if it.get("cx") else 0.0
        oy = -b0[1] * s if it.get("anchor") == "miny" else -b1[1] * s if it.get("anchor") == "maxy" else 0.0
        lo = 3 if self.scenery(it) else 0
        out = []
        for (b, z0, z1, x0, y0, x1, y1, rad) in g:
            if b >= lo:
                mx, my = rot2((x0 + x1) / 2 * s + ox, (y0 + y1) / 2 * s + oy, rot)
                out.append((x + mx, y + my, (x1 - x0) / 2 * s, (y1 - y0) / 2 * s, rot, z + z0 * s, z + z1 * s, rad * s, False))
        for (x0, y0, x1, y1) in nominal(it, sc):
            mx, my = rot2((x0 + x1) / 2 * s + ox, (y0 + y1) / 2 * s + oy, rot)
            out.append((x + mx, y + my, (x1 - x0) / 2 * s, (y1 - y0) / 2 * s, rot, z - 1.0, z + b1[2] * s, 1e9, True))
        return out

    def solid(self, i):
        """(plan AABB, boxes) of placed item i (cached)"""
        got = self._solid.get(i)
        if got is None:
            bx = self.solid_of(self.items[i])
            ext = [(abs(math.cos(math.radians(b[4]))) * b[2] + abs(math.sin(math.radians(b[4]))) * b[3],
                    abs(math.sin(math.radians(b[4]))) * b[2] + abs(math.cos(math.radians(b[4]))) * b[3]) for b in bx]
            box = (min(b[0] - e[0] for b, e in zip(bx, ext)), min(b[1] - e[1] for b, e in zip(bx, ext)),
                   max(b[0] + e[0] for b, e in zip(bx, ext)), max(b[1] + e[1] for b, e in zip(bx, ext))) if bx else None
            got = self._solid[i] = (box, bx)
        return got

    def root(self, i):
        while i in self.part:
            i = self.part[i]
        return i

    def scenery(self, it):
        """True for the town beyond the district: backdrop rows and houses that are only looked at (no collider)"""
        c = asset_class(it["asset"])
        return c == "backdrop" or (c == "building" and it.get("collide") == "none")

    def loose(self, it):
        """True for what the rules must fit around the rest: ground props, pieces hung on a socket / wall, scenery"""
        return "prop" in it or "attach" in it or self.scenery(it) or asset_mount(it["asset"]) in ("wall", "hang")

    def give(self, a, b, body):
        """How deep piece `a` may sit in piece `b` (a box of b's body when `body`) before it counts"""
        if asset_mount(a["asset"]) != "ground" and (body or b["asset"] in WALLS or b.get("collide") in ("building", "passage")):
            return LEAN
        return TOL

    def hits(self, it, of=None, me=None, upto=None):
        """What an item (placed: index `me`; or about to be: `of` = the piece it belongs to) stands in:
        [(depth, index)] over everything placed before `upto`, deepest first. Tested box by box per height band: two
        boxes intersect when they share height and overlap in plan (turned boxes, and their reach for round
        pieces) by more than the allowed give."""
        if me is not None:
            box, A = self.solid(me)
        else:
            A = self.solid_of(it)
            box = None
            if A:
                R = max(math.hypot(b[2], b[3]) for b in A)
                box = (min(b[0] for b in A) - R, min(b[1] for b in A) - R, max(b[0] for b in A) + R, max(b[1] for b in A) + R)
        if not A:
            return []
        mine = self.root(me) if me is not None else (self.root(of) if of is not None else None)
        back = self.scenery(it)
        out = []
        for j in range(len(self.items) if upto is None else upto):
            if j == me:
                continue
            ot = self.items[j]
            if mine is not None and self.root(j) == mine:
                continue
            if "attach" in it and it["attach"]["to"] == j:
                continue
            if back != self.scenery(ot) and "prop" not in ot and "prop" not in it and not (
                    ot.get("collide") in ("building", "passage") or it.get("collide") in ("building", "passage")):
                continue
            jb, B = self.solid(j)
            if jb is None or jb[0] > box[2] or jb[2] < box[0] or jb[1] > box[3] or jb[3] < box[1]:
                continue
            worst = 0.0
            for a in A:
                for b in B:
                    if min(a[6], b[6]) - max(a[5], b[5]) <= TOL or (a[8] and b[8]):
                        continue
                    tol = max(self.give(it, ot, b[8]), self.give(ot, it, a[8]))
                    d = a[7] + b[7] - math.hypot(a[0] - b[0], a[1] - b[1])
                    if d <= tol:
                        continue
                    d = min(d, pen(a[:5], b[:5]))
                    if d > tol and d > worst:
                        worst = d
            if worst:
                out.append((round(worst, 3), j))
        return sorted(out, reverse=True)

    def reach(self, it):
        """Plan radius of what a ground piece puts in a walker's way: the reach from its origin of its parts as built
        below head height (a lamp post's foot, not its arms; a bench's ends), from the plan bounds until it is
        built; capped at 1.2 m. 0 when nothing is known."""
        s = it.get("scale", 1.0)
        g = geo(it)
        if isinstance(g, list):
            low = [min(math.hypot(max(abs(b[3]), abs(b[5])), max(abs(b[4]), abs(b[6]))), math.hypot((b[3] + b[5]) / 2, (b[4] + b[6]) / 2) + b[7])
                   for b in g if b[0] <= 1]
            return min(1.2, max(low) if low else 0.0) * s
        sc = sidecar(it)
        if sc:
            b0, b1 = sc["bounds"]["min"], sc["bounds"]["max"]
            return min(1.2, math.hypot(max(abs(b0[0]), abs(b1[0])), max(abs(b0[1]), abs(b1[1])))) * s
        return 0.0

    def walk(self, it):
        """What a walker bumps into: the colliders the built town gives a ground piece as it stands
        (the rules of tools/town.py resolve, from its bounds as built): [(cx, cy, half x, half y, angle rad)];
        its origin until it is built."""
        x, y, z = it["pos"]
        s, rot, mode = it.get("scale", 1.0), it["rot"], it.get("collide", "auto")
        sc = sidecar(it)
        if not sc:
            return [(x, y, 0.15, 0.15, 0.0)]
        b0, b1 = sc["bounds"]["min"], sc["bounds"]["max"]
        sx, sy, sz = (b1[0] - b0[0]) * s, (b1[1] - b0[1]) * s, (b1[2] - b0[2]) * s
        ox, oy = rot2((b0[0] + b1[0]) / 2 * s, (b0[1] + b1[1]) / 2 * s, rot)
        if mode.startswith("circle"):
            rad = float(mode.split(":")[1]) if ":" in mode else min(sx, sy) / 2 * 0.9
        elif mode != "auto" or sz < 0.2 or b0[2] * s > 1.9:
            return [(x, y, 0.15, 0.15, 0.0)]
        elif max(sx, sy) < 1.3 and min(sx, sy) > 0.55 * max(sx, sy):
            rad = max(sx, sy) / 2 * 0.85
        else:
            return [(x + ox, y + oy, sx / 2, sy / 2, math.radians(rot))]
        return [(x + ox, y + oy, rad, rad, 0.0)]

    def on_way(self, it):
        """Name of the flight of stairs (+ 0.3 m) or walking route the colliders of a ground piece reach onto, as
        the built town is checked again (check_resolved); None when it stands clear of all of them."""
        z = it["pos"][2]
        for box in self.walk(it):
            for (n, x0, y0, x1, y1, z0, z1) in self.stairs:
                if z0 - 0.6 < z < z1 + 0.6 and box_rect(box, x0 - 0.35, y0 - 0.35, x1 + 0.35, y1 + 0.35):
                    return "stairs_" + n
            for (n, x0, y0, x1, y1, rz) in self.routes:
                if abs(rz - z) < 1.0 and box_rect(box, x0, y0, x1, y1):
                    return n
        return None

    def unput(self):
        """take the last item back (a rule found it does not fit after all)"""
        i = len(self.items) - 1
        occ, lamp = self._undo.pop(i, (False, False))
        self.items.pop()
        self._solid.pop(i, None)
        self.part.pop(i, None)
        self.yields.discard(i)
        if occ:
            self.occ.pop()
        if lamp:
            self.lamps.pop()

    def cover(self, cam, x, y, z, r, h):
        """Fraction of a review frame an upright prop (plan radius r, from z up to z + h) would hide: its angular
        width x height inside the camera's cone, + 0.5 when it stands on the line to the subject and rises through
        it; 1.0 when it stands on the camera itself."""
        px, py, pz = cam["pos"]
        tx, ty, tz = cam["target"]
        if pz > 12.0:
            return 0.0
        dx, dy = x - px, y - py
        d = math.hypot(dx, dy)
        if d > 24.0:
            return 0.0
        if d < r + 1.0:
            return 1.0
        for (sx, sy, s0, s1) in cam.get("see", ()):   # what the shot is there to show: nothing may stand on the lines to it
            Ls = math.hypot(sx - px, sy - py) or 1.0
            a_ = (dx * (sx - px) + dy * (sy - py)) / Ls
            if 0.0 < a_ < Ls - 0.5 and abs(dy * (sx - px) - dx * (sy - py)) / Ls < r + 0.25 and any(
                    z - 0.3 < pz + (sz - pz) * a_ / Ls < z + h + 0.1 for sz in (s0, (s0 + s1) / 2, s1)):
                return 0.6
        L = math.hypot(tx - px, ty - py) or 1.0
        ux, uy = (tx - px) / L, (ty - py) / L
        along, lat = dx * ux + dy * uy, dy * ux - dx * uy
        vf = math.radians(cam["fov"])
        hf = 2 * math.atan(math.tan(vf / 2) * ASPECT)
        b = math.atan2(lat, along)
        hw = math.asin(min(1.0, r / d))
        lo, hi = max(b - hw, -hf / 2), min(b + hw, hf / 2)
        if hi <= lo:
            return 0.0
        pitch = math.atan2(tz - pz, L)
        e0, e1 = math.atan2(z - pz, d) - pitch, math.atan2(z + h - pz, d) - pitch
        elo, ehi = max(e0, -vf / 2), min(e1, vf / 2)
        if ehi <= elo:
            return 0.0
        c = (hi - lo) / hf * (ehi - elo) / vf
        if abs(lat) < r + 0.3 and 0.0 < along < 0.7 * L and e0 < 0.0 < e1:
            c += 0.5
        return c

    def tall(self, it):
        """height of a piece as built (its bounds), the bulky props' table or 1.2 m until it is built"""
        sc = sidecar(it)
        return sc["bounds"]["max"][2] * it.get("scale", 1.0) if sc else BULK.get(it["asset"], (0, 1.2))[1]

    def outline(self, it, x=None, y=None):
        """plan outline of a piece as built (the corners of its turned bounds), stood at (x, y); None until it is built"""
        sc = sidecar(it)
        if not sc:
            return None
        (ax, ay, _), (bx, by, _) = sc["bounds"]["min"], sc["bounds"]["max"]
        k = it.get("scale", 1.0)
        x, y = it["pos"][0] if x is None else x, it["pos"][1] if y is None else y
        out = []
        for lx, ly in ((ax, ay), (bx, ay), (bx, by), (ax, by), ((ax + bx) / 2, ay), ((ax + bx) / 2, by)):
            wx, wy = rot2(lx * k, ly * k, it["rot"])
            out.append((x + wx, y + wy))
        return out

    def cut(self, x, y, z, r, h, it=None):
        """name of the review shot whose frame would cut a standing piece in its near field (cut_by), or None"""
        pts = self.outline(it, x, y) if it is not None else None
        for cam in self.cams:
            if cut_by(cam, x, y, z, r, h, pts):
                return cam["name"]
        return None

    def sight(self, x, y, z, r, h, lift=0.0):
        """(worst cover, shot name) of a prop over all review cameras; `lift` = clear height under a canopy"""
        worst, name = 0.0, None
        for cam in self.cams:
            c = self.cover(cam, x, y, z + lift, r, h - lift)
            if c > worst:
                worst, name = c, cam["name"]
        return worst, name

    def home(self, x, y, z):
        """the walkable region a ground point belongs to (rect) or None"""
        for rg in self.regions:
            r0, r1, r2, r3 = rg["rect"]
            if abs(rg.get("z", 0) - z) < 0.5 and r0 <= x <= r2 and r1 <= y <= r3:
                return rg["rect"]
        return None

    def is_prop(self, asset, collide, extra):
        """True for things that stand loose on the ground (and so must keep off stairs and walking routes)"""
        if asset in FLAT or asset in FIXED or collide in ("building", "passage") or any(k in extra for k in ("anchor", "on", "attach", "nav")):
            return False
        if asset == "column":
            return True
        return asset_mount(asset) in ("ground", "wall_ground") and asset_class(asset) in PROP_CLS

    def settle(self, asset, x, y, z, r, ok=None, along=None):
        """Nearest spot (<= 2.4 m, same piece of ground) where a prop stands clear of every flight of stairs and
        walking route, in nothing that is built there (`ok`) and, when bulky, hides no more than SIGHT_PUT of any
        review frame. along = (ux, uy): a piece of a row (lamp posts along a kerb) first slides along its row.
        None when there is none."""
        bulk = BULK.get(asset)
        home = self.home(x, y, z)
        spots = [(x, y, 0)]
        if along:
            spots += [(x + along[0] * 0.3 * k * sg, y + along[1] * 0.3 * k * sg, 1) for k in range(1, 9) for sg in (1, -1)]
        for ring in range(1, 9):
            for k in range(12):
                a = k * math.tau / 12 + 0.4 * ring
                spots.append((x + 0.3 * ring * math.cos(a), y + 0.3 * ring * math.sin(a), ring))
        for (px, py, ring) in spots:
            if True:
                if ring and home and not (home[0] + 0.3 <= px <= home[2] - 0.3 and home[1] + 0.3 <= py <= home[3] - 0.3):
                    continue
                if self.blocked(px, py, r, z) or (ring and not self.clear(px, py, r, z)):
                    continue
                if bulk and self.sight(px, py, z, *bulk)[0] > SIGHT_PUT:
                    continue
                if ok and not ok(px, py):
                    continue
                return px, py
        return None

    def place(self, asset, zones, r, z=0.0, want=None, limit=SIGHT_ONE, rot=None, key="", n=260, foot=None, **kw):
        """One-off prop by rule: seeded candidate spots inside the intended zone rect(s); a candidate is rejected when
        it touches another prop, a walking route or a flight of stairs, or when it would hide more than `limit` of a
        review frame. Of the rest the cheapest is taken: distance from `want` (x, y) + what it hides - elbow room.
        rot: degrees | callable(x, y) | None (seeded). Returns the item index; None (and a report line) when no
        candidate survives."""
        if not asset_path(asset):
            self.missing[asset] = self.missing.get(asset, 0) + 1
            return None
        R = self.rng("place", asset, key)
        zones = [zq for zq in zones if zq[2] - zq[0] >= 0 and zq[3] - zq[1] >= 0]
        if not zones:
            self.dropped.append(f"{asset} ({key}): empty zone")
            return None
        area = [max(0.05, (zq[2] - zq[0])) * max(0.05, (zq[3] - zq[1])) for zq in zones]
        bulk = BULK.get(asset, (r, 1.5, 0.0))
        best, why = None, {}
        for _ in range(n):
            zq = R.choices(zones, weights=area)[0]
            x, y = R.uniform(zq[0], zq[2]), R.uniform(zq[1], zq[3])
            tie = R.random()
            if not self.free(x, y, r, z) if foot is None else (not self.clear(x, y, r, z) or self.blocked(x, y, foot, z)):
                hit = self.blocked(x, y, foot or r, z) or "occupied"
                why[hit] = why.get(hit, 0) + 1
                continue
            c, shot = self.sight(x, y, z, *bulk)
            if c > limit:
                why["view " + str(shot)] = why.get("view " + str(shot), 0) + 1
                continue
            if bulk[1] >= 3.0 and any(abs(tz - z) < 2.5 and math.hypot(tx - x, ty - y) < tr + bulk[0] for tx, ty, tz, tr in self.crowns):
                why["under a tree"] = why.get("under a tree", 0) + 1
                continue
            clear = 3.0
            for (ox, oy, orr, oz) in self.occ:
                if abs(oz - z) < 1.0:
                    clear = min(clear, math.hypot(ox - x, oy - y) - orr - r)
            cost = 40.0 * c - 0.3 * clear + 0.01 * tie
            if want:
                cost += math.hypot(x - want[0], y - want[1])
            best = (best or []) + [(cost, x, y)]
        kw.setdefault("variant", "*")
        kw.setdefault("tag", key)
        kw.setdefault("seed", R.randrange(999))
        spin = R.uniform(0, 360)
        for (_, x, y) in sorted(best or [])[:24]:
            a = rot(x, y) if callable(rot) else (spin if rot is None else rot)
            idx = self.put(asset, x, y, z, a, r=r, foot=foot, fixed=True, quiet=True, **kw)
            if idx is None:                           # the piece as built would stand in something there
                why["built parts"] = why.get("built parts", 0) + 1
                continue
            reach = self.reach(self.items[idx])       # the piece as built (variant by position): its real reach must fit too
            if reach > (foot or r) and "prop" in self.items[idx]:
                hit = self.blocked(x, y, reach, z)
                if hit:
                    why[hit] = why.get(hit, 0) + 1
                    self.unput()
                    continue
                self.items[idx]["prop"] = round(reach, 2)
            return idx
        self.dropped.append(f"{asset} ({key}): no valid spot in zone ({', '.join(f'{k} x{v}' for k, v in sorted(why.items()))})")
        return None

    def bays(self, a, b, step, names, z=0.0, off=0.6, key="", r=0.7):
        """Regular furnishing of an edge: one bay every `step` m along a->b; in each bay the next asset of the
        (seeded start) cycle `names` stands `off` to the RIGHT of the direction of travel, facing right. names:
        asset | (asset, {put kwargs, turn=deg}) | None (empty bay). A bay whose spot is taken, on a route / flight of
        stairs or in a review frame stays empty."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < step * 0.6 or not names:
            return 0
        nx, ny = dy / L, -dx / L
        rot = facing_rot(nx, ny)
        R = self.rng("bays", key, a, b)
        n = max(1, int(round(L / step * min(1.0, self.P["density"]))))
        k0 = R.randrange(len(names))
        done = 0
        for i in range(n):
            nm = names[(k0 + i) % len(names)]
            jit = R.uniform(-0.12, 0.12) * step
            sd = R.randrange(999)
            if nm is None:
                continue
            asset, kw = (nm, {}) if isinstance(nm, str) else (nm[0], dict(nm[1]))
            if not asset_path(asset):
                self.missing[asset] = self.missing.get(asset, 0) + 1
                continue
            t = (i + 0.5) / n
            x, y = a[0] + dx * t + nx * off + dx / L * jit, a[1] + dy * t + ny * off + dy / L * jit
            kw.setdefault("variant", "*")
            kw.setdefault("seed", sd)
            turn = kw.pop("turn", 0.0)
            for sl, st in ((sl, st) for sl in SLIDE for st in STEP_OFF):   # the piece as built decides how close to the edge it fits;
                px, py = x + nx * st + dx / L * sl * step, y + ny * st + dy / L * sl * step   # a lamp post in its bay: it moves over
                if not self.free(px, py, r, z) or (asset in BULK and self.sight(px, py, z, *BULK[asset])[0] > SIGHT_PUT):
                    continue
                if self.put(asset, px, py, z, rot + turn, tag=key, r=r, fixed=True, quiet=True, **kw) is not None:
                    done += 1
                    break
        return done

    def audit(self):
        """Generate-time assertion: no ground prop stands on a flight of stairs (+ margin), on a walking route or at
        the foot of a building's steps; no ground prop, hung piece or backdrop row intersects anything built (Town.hits:
        building bodies and their stoops, other props, trunks, walls); every trunk is clear of the grate / guard
        around it. Raises with the full list. Returns (sight report [(cover, shot, asset, tag)] of the bulky props,
        counts of what was tested)."""
        bad, seen = [], []
        self.cuts = []
        n = {"props": 0, "pieces": 0, "pairs": 0, "rings": len(self.rings), "unmeasured": 0}
        for i, it in enumerate(self.items):
            if "prop" in it:
                x, y, z = it["pos"]
                hit = self.blocked(x, y, it["prop"], z)
                if hit:
                    bad.append(f"{it['asset']} ({it['tag']}) r={it['prop']} at ({x}, {y}, {z}) stands on {hit}")
                if it["asset"] in BULK and self.cams:
                    c, shot = self.sight(x, y, z, *BULK[it["asset"]])
                    if c > 0.08:
                        seen.append([round(c, 3), shot, it["asset"], it["tag"]])
                if asset_class(it["asset"]) != "vegetation":
                    shot = self.cut(x, y, z, it["prop"], self.tall(it), it)
                    if shot:
                        self.cuts.append([shot, it["asset"], it["tag"]])
            g = geo(it)
            if isinstance(g, dict):
                bad.append(f"{it['asset']} ({it['tag']}): unreadable build ({g.get('error')})")
            if not self.loose(it):
                continue
            n["props" if "prop" in it else "pieces"] += 1
            if not self.solid(i)[1]:
                n["unmeasured"] += g is None
                continue
            for d, j in self.hits(it, me=i):
                ot = self.items[j]
                n["pairs"] += 1
                if j < i and self.loose(ot):
                    continue                              # reported from the other side
                bad.append(f"{it['asset']} ({it['tag']}) at {it['pos']} stands {d} m deep in {ot['asset']} ({ot['tag']}) at {ot['pos']}")
        for (gi, ti) in self.rings:
            g, t = self.items[gi], self.items[ti]
            bore, prof = geo(g, "bore"), geo(t, "trunk")
            if not isinstance(bore, list) or not isinstance(prof, list):
                n["unmeasured"] += 1
                continue
            need = ring_fit(bore, prof, g.get("scale", 1.0), t.get("scale", 1.0))
            if need > 0:
                bad.append(f"{t['asset']} ({t['tag']}, {t['variant']}) at {t['pos']}: trunk {need:.2f} m too thick for {g['asset']} ({g['variant']})")
        for (rn, rx0, ry0, rx1, ry1, rz) in self.routes:         # the head / foot of a flight must not open onto a stoop or base
            if not rn.startswith("stair_"):
                continue
            n["approaches"] = n.get("approaches", 0) + 1
            for (kn, x0, y0, x1, y1, kz) in self.keep:
                if abs(kz - rz) < 1.0 and min(rx1, x1) - max(rx0, x0) > 0.15 and min(ry1, y1) - max(ry0, y0) > 0.15:
                    bad.append(f"{kn} stands {min(min(rx1, x1) - max(rx0, x0), min(ry1, y1) - max(ry0, y0)):.2f} m into the approach {rn}")
        if bad:
            raise AssertionError("town layout: props on stairs / walking routes / in each other:\n  " + "\n  ".join(bad))
        return sorted(seen, reverse=True), n

    # ---------------------------------------------------------------- sockets of built assets
    def param(self, idx, name, default=None):
        """effective value of a build param of an item: explicit > variant override > asset default"""
        it = self.items[idx]
        if name in it["params"]:
            return it["params"][name]
        schema, _ = asset_spec(it["asset"])
        over = _VARIANT.get(it["asset"], {}).get(it["variant"] or "", {})
        if isinstance(over, dict) and name in over:
            return over[name]
        sp = schema.get(name)
        return sp.get("default", default) if isinstance(sp, dict) else default

    def sockets(self, idx, role=None):
        """Free attach sockets of an item in world space: [{name, role, x, y, z, rot, props}]. Read from the build
        record, so [] until the asset has been built once (tools/town.py all then lays the town out again). A socket
        named after a switched-on param of the asset (hanging_sign, awning ...) marks a part the building already
        carries and is not offered."""
        if idx is None:
            return []
        it = self.items[idx]
        sc = sidecar(it)
        if sc is None:
            self.unbuilt += 1
            return []
        schema, _ = asset_spec(it["asset"])
        s = it.get("scale", 1.0)
        out = []
        for q in sc.get("sockets", []):
            if q.get("kind") != "attach":
                continue
            ro = (q.get("props") or {}).get("role")
            if role and ro != role:
                continue
            if (it["asset"], q["name"]) in OWN or (q["name"] in schema and self.param(idx, q["name"])):
                continue
            ox, oy = rot2(q["pos"][0] * s, q["pos"][1] * s, it["rot"])
            out.append({"name": q["name"], "role": ro, "x": it["pos"][0] + ox, "y": it["pos"][1] + oy, "z": it["pos"][2] + q["pos"][2] * s,
                        "rot": (it["rot"] + (q.get("rot") or [0, 0, 0])[2]) % 360.0, "props": q.get("props") or {}})
        return out

    def lanterns(self, idx, top):
        """How many lights an item carries itself no more than `top` m above its base (its build record's light
        sockets; 0 until it has been built once)."""
        sc = sidecar(self.items[idx]) if idx is not None else None
        s = self.items[idx].get("scale", 1.0) if idx is not None else 1.0
        return sum(1 for q in (sc or {}).get("sockets", []) if q.get("kind") == "light" and q["pos"][2] * s <= top)

    def doors(self, idx, name):
        """walking-route rects in front of the door sockets of a building (1.5 m square, on the door's side)"""
        n = 0
        for q in self.sockets(idx, "door"):
            if abs(q["z"] - self.items[idx]["pos"][2]) > 1.6:
                continue
            fx, fy = rot2(0, -1, q["rot"])
            mx, my = q["x"] + fx * 0.8, q["y"] + fy * 0.8
            self.route(f"door_{name}{n or ''}", mx - 0.75, my - 0.75, mx + 0.75, my + 0.75, self.items[idx]["pos"][2])
            n += 1
        return n

    def fit(self, idx, q, asset, tag="fit", attach=True, **kw):
        """Fit an asset to socket q (one of Town.sockets(idx)): hung things are attached (the resolver takes the
        final pose from the built parent), standing things (attach=False) are put on the socket's spot and
        validated like any other prop."""
        it = self.items[idx]
        kw.setdefault("variant", "*")
        if not attach:
            return self.put(asset, q["x"], q["y"], q["z"], q["rot"], tag=tag, **kw)
        sc = sidecar(it) or {}
        same = [s for s in sc.get("sockets", []) if s.get("kind") == "attach" and ((s.get("props") or {}).get("role") == q["name"] or s["name"] == q["name"])]
        if len(same) == 1:
            kw["attach"] = {"to": idx, "role": q["name"]}
        return self.put(asset, q["x"], q["y"], q["z"], q["rot"], tag=tag, collide="none", **kw)

    def front(self, idx, d0, d1, inset=0.5):
        """zone rect in front of a row building: from d0 to d1 m off its front line, `inset` in from its side walls"""
        it = self.items[idx]
        w = it["params"].get("width", 4.0) / 2 - inset
        return rect_of(it["pos"][0], it["pos"][1], it["rot"], -w, -d1, w, -d0)

    # ---------------------------------------------------------------- items
    def put(self, asset, x, y, z=0.0, rot=0.0, variant=None, collide="auto", scale=1.0, tag="", r=None, foot=None, fixed=False,
            of=None, quiet=False, along=None, matched=False, **params):
        """Place one asset. variant='*' draws one of the asset's own VARIANTS from the seed; an int `seed` param is
        added automatically. Ground props are moved to the nearest spot clear of stairs / walking routes / review
        frames and of everything built there (Town.settle) unless `fixed` (the calling rule chose the spot: the piece
        is then left out when it would stand in something); a prop with no such spot is dropped and reported. A piece
        hung on a socket is left out likewise. r = spacing radius (other props keep that far), foot = radius of the
        body itself when that is smaller (tree trunks), of = index of the piece this one belongs to (they may touch).
        Returns the item index, or None when the asset does not exist yet / was dropped."""
        self._n += 1
        if not asset_path(asset):
            self.missing[asset] = self.missing.get(asset, 0) + 1
            if collide in ("building", "passage") and params.get("width"):
                self.blockouts.append({"asset": asset, "pos": [round(x, 3), round(y, 3), round(z, 3)], "rot": round(rot % 360.0, 3),
                                       "width": params["width"], "depth": params.get("depth", 8.0),
                                       "height": 3.2 * params.get("storeys", 3) + 2.5, "centered": collide == "passage",
                                       "passage_width": params.get("passage_width", 0)})
            return None
        extra = {}
        for k in ("anchor", "on", "attach", "nav", "cx"):
            if k in params:
                extra[k] = params.pop(k)
        schema, variants = asset_spec(asset)
        R = self.rng("item", asset, tag, round(x, 2), round(y, 2), round(z, 2))
        cand = None                                           # the variants the rule allows (None: the one it names)
        if variant == "*":
            cand = ["default"] + variants
            variant = R.choice(cand) if variants else None
        elif isinstance(variant, (list, tuple)):
            have = [v for v in variant if v in variants or v == "default"]
            cand = have or ["default"] + variants
            variant = R.choice(have) if have else (R.choice(["default"] + variants) if variants else None)
        if variant == "default":
            variant = None
        pr = {}
        for k, v in params.items():
            if v is None:
                continue
            pr[k] = round(v, 3) if isinstance(v, float) else v
        # a small pool of seeds per asset keeps the number of distinct meshes (GPU memory, build time) bounded:
        # repeated props share geometry, buildings and landmarks stay one-offs
        pool = SEED_POOL.get(asset_class(asset), 2)
        if pool and "seed" in pr:
            pr["seed"] = 1 + int(pr["seed"]) % pool
        # (a piece of a matched set - the pair at a door, the tubs closing a parapet row - is the design its rule names:
        # the set repeating is the point)
        accent = asset in ACCENT and not matched and not any(k in extra for k in ("on", "attach"))
        if accent:
            vs = ([v if v != "default" else None for v in cand] if cand else [variant])
            got = self.vary(asset, x, y, vs, R, quiet)
            if got is None:
                return None                                   # every proto of it already shows in a cone here / it is another shot's accent
            (variant, pr["seed"]), book = got[0], got
            if scale == 1.0 and asset != "pigeons":
                scale = round(R.uniform(0.9, 1.1), 2)
        prop = self.is_prop(asset, collide, extra)
        rr = (foot or r or BULK.get(asset, (0.3,))[0]) if prop else None
        it = {"asset": asset, "variant": variant, "params": pr, "pos": [round(x, 3), round(y, 3), round(z, 3)],
              "rot": round(rot % 360.0, 3), "scale": scale, "collide": collide, "tag": tag}
        it.update(extra)
        if prop:
            it["prop"] = round(rr, 2)

        def stands_in(px, py):
            """what the piece as built would intersect / what way its collider would reach onto at (px, py):
            'asset (tag)' / the name of the way, or None"""
            it["pos"][0], it["pos"][1] = round(px, 3), round(py, 3)
            h = self.hits(it, of)
            return f"{self.items[h[0][1]]['asset']} ({self.items[h[0][1]]['tag']})" if h else (self.on_way(it) if prop else None)
        if prop and collide == "auto":
            # top-heavy pieces (the arms of a lamp post, a parasol's canopy) are walked under: such a piece collides
            # as its body below head height, not as the box of everything it carries overhead
            low, sc = self.reach(it), sidecar(it)
            if sc and isinstance(geo(it), list) and low > 0.0:
                b0, b1 = sc["bounds"]["min"], sc["bounds"]["max"]
                if max(b1[0] - b0[0], b1[1] - b0[1]) / 2 * scale > 1.6 * low + 0.2 and b1[2] * scale > BANDS[2][0] + 0.5:
                    collide = it["collide"] = "circle:%.2f" % max(0.2, low)
        if prop and not fixed:
            # the body as built: a circle collider's radius, else its reach below head height (Town.reach)
            if isinstance(collide, str) and collide.startswith("circle:"):
                rr = max(rr, float(collide.split(":")[1]))
            else:
                rr = max(rr, self.reach(it))
            it["prop"] = round(rr, 2)
            hh = self.tall(it)
            at = self.settle(asset, x, y, z, rr, ok=lambda px, py: stands_in(px, py) is None and self.cut(px, py, z, rr, hh, it) is None, along=along)
            if at is None:
                if not quiet:                                 # (a quiet rule tries elsewhere / can do without)
                    self.dropped.append(f"{asset} ({tag}) at ({x:.2f}, {y:.2f}): {self.blocked(x, y, rr, z) or stands_in(x, y) or 'in a review frame'}, no free spot nearby")
                return None
            if at != (x, y):
                why = self.blocked(x, y, rr, z) or stands_in(x, y) or ("cut by the frame of " + str(self.cut(x, y, z, rr, hh, it)) if self.cut(x, y, z, rr, hh, it) else "review frame")
                self.nudged.append(f"{asset} ({tag}) ({x:.2f}, {y:.2f}) -> ({at[0]:.2f}, {at[1]:.2f}): {why}")
            x, y = at
            it["pos"][0], it["pos"][1] = round(x, 3), round(y, 3)
        elif self.loose(it):
            hit = stands_in(x, y)
            if not hit and prop and quiet and self.cut(x, y, z, rr, self.tall(it), it):
                return None                                   # dressing a rule can do without: not where a frame would cut it
            if hit and asset_mount(asset) == "wall" and "attach" not in it:
                ax, ay = rot2(1, 0, rot)                      # a wall fixture slides along its wall to clear what stands there
                for d in (0.7, -0.7, 1.4, -1.4):
                    if stands_in(x + ax * d, y + ay * d) is None:
                        self.nudged.append(f"{asset} ({tag}) ({x:.2f}, {y:.2f}) -> ({x + ax * d:.2f}, {y + ay * d:.2f}): {hit}")
                        x, y, hit = x + ax * d, y + ay * d, None
                        break
            if hit:
                if not quiet:
                    self.clashed.append(f"{asset} ({tag}) at ({x:.2f}, {y:.2f}, {z:.2f}): would stand in {hit}")
                return None
            it["pos"][0], it["pos"][1] = round(x, 3), round(y, 3)
        i = len(self.items)
        self.items.append(it)
        if accent:                                            # Town.vary: the proto now shows in these cones
            key = (asset,) + book[0]
            self.used[key] = self.used.get(key, 0) + 1
            for c in book[1]:
                self.shown.setdefault(c, set()).add(key)
            if book[2]:
                self.accents.setdefault(asset, set()).update(book[2])
        if of is not None:
            self.part[i] = of
        if (prop and (quiet or not fixed)) or of in self.yields or extra.get("on") in self.yields:
            self.yields.add(i)                                # gives way to the review frames, alone or with what it belongs to
        if r:
            self.occ.append((x, y, r, z))
        if asset.startswith("lamp_post"):
            self.lamps.append((x, y, z))
        self._undo[i] = (bool(r), asset.startswith("lamp_post"))
        if collide in ("building", "passage"):            # what stands out at its foot (stoops, steps, bay bases): a keep-out
            for b in self.solid(i)[1]:
                if not b[8] and b[5] - z < BANDS[1][1] - 0.1 and min(b[2], b[3]) >= STOOP / 2:
                    c, sn = abs(math.cos(math.radians(b[4]))), abs(math.sin(math.radians(b[4])))
                    ex, ey = c * b[2] + sn * b[3], sn * b[2] + c * b[3]
                    self.keep.append((f"foot of {asset} ({tag})", round(b[0] - ex, 3), round(b[1] - ey, 3), round(b[0] + ex, 3), round(b[1] + ey, 3), round(z, 3)))
        return i

    def vary(self, asset, x, y, variants, R, quiet):
        """(variant, seed) of an accent piece at (x, y): of the protos the rule allows, one that no review sight cone
        holding the spot (eye-level cameras of the pass before: in frame, within CONE) has shown yet - the one least
        used in the town so far, ties by the piece's own seed. None for a piece its rule can do without (`quiet`) when
        every proto already shows there, or when the asset already is a foreground accent (within FOREGROUND of
        a camera) - one kind of thing fronts one shot, once. Returns (proto, cones, foregrounds) for Town.put to book once the piece stands."""
        cones, fore = [], []
        for cam in self.cams:
            px, py, pz = cam["pos"]
            d = math.hypot(x - px, y - py)
            if pz > 12.0 or d > CONE:
                continue
            tx, ty = cam["target"][0], cam["target"][1]
            hf = 2 * math.atan(math.tan(math.radians(cam["fov"]) / 2) * ASPECT)
            b = math.atan2(y - py, x - px) - math.atan2(ty - py, tx - px)
            if abs((b + math.pi) % math.tau - math.pi) < hf / 2 + 0.06:
                cones.append(cam["name"])
                if d < FOREGROUND:
                    fore.append(cam["name"])
        if quiet and fore and self.accents.get(asset):
            return None                                       # it already fronts a shot (this one or another)
        protos = [(v, sd) for v in variants for sd in range(1, ACCENT_SEEDS + 1)]
        R.shuffle(protos)
        shown = set().union(*[self.shown.get(c, set()) for c in cones]) if cones else set()
        free = [q for q in protos if (asset,) + q not in shown]
        if not free and quiet:
            return None
        return min(free or protos, key=lambda q: self.used.get((asset,) + q, 0)), cones, fore

    def stands(self, asset, x, y, z=0.0, rot=0.0, r=None, foot=None, **kw):
        """Put a ground piece exactly where its rule wants it, or not at all: None unless the spot is free of props,
        flights, walking routes and building feet and the piece as built stands in nothing (never moved, never
        reported: the calling rule decides what a refusal means)."""
        if not self.free(x, y, foot or r or 0.3, z):
            return None
        return self.put(asset, x, y, z, rot, r=r, foot=foot, fixed=True, quiet=True, **kw)

    def pair(self, asset, ax, y, dx, z=0.0, rot=0.0, pull=(0.0,), tag="", seed=0, **kw):
        """A mirror pair about the axis x = ax (the two sides of a monument, a gate): both pieces stand (Town.stands)
        dx either side of it - pulled in towards the axis by each of `pull` in turn - or neither does. One variant
        for both, drawn from the rule's own stream. Returns the two indices, or None (reported as left out)."""
        v = kw.pop("variant", "*")
        if asset_path(asset) and (v == "*" or isinstance(v, (list, tuple))):
            names = ["default"] + asset_spec(asset)[1]
            v = self.rng("pair", asset, tag, round(ax, 2), round(y, 2)).choice(names if v == "*" else [q for q in v if q in names] or names)
        for d in pull:
            a = self.stands(asset, ax - (dx - d), y, z, rot, variant=v, tag=tag, seed=seed, **kw)
            if a is None:
                continue
            b = self.stands(asset, ax + (dx - d), y, z, (-rot) % 360.0, variant=v, tag=tag, seed=seed + 1, **kw)
            if b is not None:
                return a, b
            self.unput()
        if asset_path(asset):
            self.clashed.append(f"pair of {asset} ({tag}) {dx:.2f} m either side of x={ax:.2f} at y={y:.2f}: no free mirror spots")
        return None

    def tree_spot(self, x, y, R, z=0.0, r=1.2, canopy=3.6, reach=2.4, wall=2.4):
        """A free spot for a tree near (x, y) whose canopy keeps clear of every lamp post placed so far (a lantern
        inside the foliage blows the leaves out) and whose trunk stands at least `wall` from the edge of its piece
        of ground (house fronts, retaining walls). Seeded spiral search; None when nothing within `reach` fits."""
        home = self.home(x, y, z)
        for k in range(40):
            d = reach * k / 39.0
            a = R.uniform(0, 2 * math.pi)
            tx, ty = x + d * math.cos(a), y + d * math.sin(a)
            if home and not (home[0] + wall <= tx <= home[2] - wall and home[1] + wall <= ty <= home[3] - wall):
                continue
            if not self.clear(tx, ty, r, z) or self.blocked(tx, ty, 0.6, z) or self.in_view(tx, ty, z, canopy):
                continue
            if all(abs(lz - z) > 2.5 or (lx - tx) ** 2 + (ly - ty) ** 2 >= canopy ** 2 for lx, ly, lz in self.lamps):
                return tx, ty
        return None

    def tree(self, asset, x, y, z, rot, R, tag="tree", grate="tree_grate", smax=1.6, canopy=3.6, **kw):
        """A street tree by rule: it gets an iron grate - with a guard when the trunk leaves room for one - only
        when its trunk AS BUILT (bark reach per 0.1 m of height, measured from the tree's own build without a
        surround) passes through it with clearance (ring_fit); the grate is opened to the widest hole its asset
        allows and scaled up to `smax` for that. A trunk too thick for every grate keeps the tree's own surround
        (setts / soil) instead. Until the tree has been built once it stands bare (tools/town.py all lays the town
        out again after the build). Returns the tree's index."""
        ti = self.put(asset, x, y, z, rot, tag=tag, fixed=True, surround="none", **kw)
        if ti is not None:
            self.crowns.append((x, y, z, canopy))
            self.yields.add(ti)                               # stood by rule clear of the review views (tree_spot)
        if ti is None or not asset_path(grate):
            return ti
        prof = geo(self.items[ti], "trunk")
        if not isinstance(prof, list):
            return ti
        schema, names = asset_spec(grate)
        over = _VARIANT.get(grate, {})

        def val(v, k):
            return over.get(v, {}).get(k, schema.get(k, {}).get("default"))
        hole = schema.get("hole", {}).get("max", 0.6)
        ts = self.items[ti].get("scale", 1.0)
        order = ["default"] + list(names)
        R.shuffle(order)
        order.sort(key=lambda v: not val(v, "guard"))        # guards first, the plain grates after them
        grot = R.uniform(0, 360)
        for v in order:
            guard = bool(val(v, "guard"))
            # what the asset promises for these params: a hole of `hole` diameter, the guard's bars just outside it
            bore = [hole / 2, hole / 2 + 0.01 if guard else None, val(v, "guard_height") or 0.0]
            sc = next((q / 20.0 for q in range(20, int(smax * 20) + 1) if ring_fit(bore, prof, q / 20.0, ts) <= 0), None)
            if sc is None:
                continue
            size = max(val(v, "size") or 1.3, 1.3)           # a frame wide enough for the widest hole
            gi = self.put(grate, x, y, z, grot, variant=v, tag=tag, collide="none", scale=sc, of=ti, hole=hole, size=size, seed=R.randrange(99))
            real = geo(self.items[gi], "bore")               # the grate as built, once it is
            if isinstance(real, list) and ring_fit(real, prof, sc, ts) > 0:
                self.unput()
                continue
            self.rings.append((gi, ti))
            return ti
        self.unput()                                          # no grate takes this trunk: the tree keeps its own surround
        return self.put(asset, x, y, z, rot, tag=tag, fixed=True, **kw)

    def backdrop(self, tag, x, y, rot, length=None, z=0.0, reach=26.0, step=2.0, asset="backdrop_row", fill=False, **kw):
        """A piece of scenery by rule (a backdrop row `length` long, or a far house): it stands at (x, y) facing
        `rot` unless its masses as built (walls / roofs from 3 m up) would cut into a building or other scenery; then
        it is pushed away from the square and / or slid along its own line to the nearest clear pose (`step` grid,
        nearest first, within `reach`) and left out when there is none (reported, unless it is only there to `fill`
        a gap its neighbours may not leave). Variant and seed
        come from the nominal pose, so the build never depends on the pose tried. Returns the item index or None."""
        if not asset_path(asset):
            return self.put(asset, x, y, z, rot, tag=tag)
        names = asset_spec(asset)[1]
        v = kw.pop("variant", None) or "*"
        if not isinstance(v, str) or v == "*":
            v = self.rng("item", asset, tag, round(x, 2), round(y, 2), round(z, 2)).choice(["default"] + names if v == "*" else list(v))
        ox, oy = rot2(0, 1, rot)                              # away from what the row faces
        ax, ay = rot2(1, 0, rot)
        n = int(reach / step)
        for p, s in sorted(((p * step, s * step) for p in range(n + 1) for s in range(-n, n + 1)), key=lambda c: (c[0] + abs(c[1]), abs(c[1]), c[1])):
            if p + abs(s) > reach:
                break
            px, py = x + ox * p + ax * s, y + oy * p + ay * s
            i = self.put(asset, px, py, z, rot, variant=v, tag=tag, collide="none", quiet=True, length=length, **kw)
            if i is not None:
                if p or s:
                    self.nudged.append(f"{asset} ({tag}) ({x:.2f}, {y:.2f}) -> ({px:.2f}, {py:.2f}): pushed {p:.0f} m back, slid {s:.0f} m to clear its neighbours")
                return i
        if not fill:
            self.clashed.append(f"{asset} ({tag}) at ({x:.2f}, {y:.2f}, {z:.2f}): no clear pose within {reach:.0f} m")
        return None

    def in_view(self, x, y, z, canopy):
        """True when a tree at (x, y, z) would stand over a review viewpoint or in the near part of its sightline
        (views come from the layout pass before: (x, y, ground z, unit view direction))."""
        for vx, vy, vz, ux, uy in self.views:
            dx, dy = x - vx, y - vy
            if z > vz + 1.0:
                continue
            if dx * dx + dy * dy < (canopy + 1.2) ** 2:
                return True
            along, lat = dx * ux + dy * uy, abs(dy * ux - dx * uy)
            if 0.0 < along < 11.0 and lat < max(1.4, canopy + 0.4 - 0.22 * along):
                return True
        return False

    def pick(self, R, names):
        """a seeded choice among the assets of `names` (asset | (asset, variant)) that exist (None when none do)"""
        have = [n for n in names if asset_path(n if isinstance(n, str) else n[0])]
        for n in names:
            if n not in have:
                self.missing.setdefault(n if isinstance(n, str) else n[0], 0)
        return R.choice(have) if have else None

    def line(self, a, b, n, names, z=0.0, off=0.55, key="", r=0.8, wall=("ladder", "cellar_hatch", "notice_board"), jitter=14.0, spread=0.25):
        """Dress the foot of a wall: up to n props at seeded positions along a->b, standing `off` to the RIGHT of
        the direction of travel and facing right (away from the wall). Wall-leaning assets go against the wall."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 0.5:
            return 0
        nx, ny = dy / L, -dx / L
        rot = facing_rot(nx, ny)
        R = self.rng("line", key, a, b)
        n = max(1, int(round(n * self.P["density"])))
        done = 0
        for _ in range(n * 25):
            if done >= n:
                break
            asset = self.pick(R, names)
            if asset is None:
                break
            var = "*"
            if not isinstance(asset, str):
                asset, var = asset
            t = R.uniform(0.04, 0.96)
            o = 0.04 if asset in wall else off + R.uniform(-0.1, spread)
            x, y = a[0] + dx * t + nx * o, a[1] + dy * t + ny * o
            jr = 0.0 if asset in wall else R.uniform(-jitter, jitter)
            sd = R.randrange(999)
            for st in ((0.0,) if asset in wall else STEP_OFF):      # the piece as built decides how close to the wall it fits
                px, py = x + nx * st, y + ny * st
                if not self.free(px, py, r, z) or (asset in BULK and self.sight(px, py, z, *BULK[asset])[0] > SIGHT_PUT):
                    continue
                if self.put(asset, px, py, z, rot + jr, variant=var, tag=key, r=r, fixed=True, quiet=True, seed=sd) is not None:
                    done += 1
                    break
        return done

    def clear(self, x, y, r, z=0.0):
        """True when a disc is clear of every placed prop"""
        for (ox, oy, orr, oz) in self.occ:
            if abs(oz - z) < 1.0 and (ox - x) ** 2 + (oy - y) ** 2 < (orr + r) ** 2:
                return False
        return True

    def free(self, x, y, r, z=0.0):
        """True when a disc is clear of every placed prop, flight of stairs and walking route"""
        return self.clear(x, y, r, z) and self.blocked(x, y, r, z) is None

    def block(self, x, y, r, z=0.0):
        self.occ.append((x, y, r, z))

    def scatter(self, rect, n, fn, r=0.6, z=0.0, key="", tries=30):
        """Call fn(x, y, R, i) at up to n free spots inside rect (seeded rejection sampling)."""
        R = self.rng("scatter", key, rect)
        done = 0
        n = max(0, int(round(n * self.P["density"]))) if n > 1 else n
        if rect[2] - rect[0] < 2 * r or rect[3] - rect[1] < 2 * r:
            return 0
        for _ in range(n * tries):
            if done >= n:
                break
            x = R.uniform(rect[0] + r, rect[2] - r)
            y = R.uniform(rect[1] + r, rect[3] - r)
            if not self.free(x, y, r, z):
                continue
            if fn(x, y, R, done) is not False:
                self.block(x, y, r, z)
                done += 1
        return done

    def rows(self, rect, pitch, fn, r=0.6, z=0.0, key=""):
        """Call fn(x, y, R, i) at the free points of a grid of `pitch` m set out in rect (cafe tables stand in rows,
        each a little off its mark: seeded, by at most what keeps two neighbours r apart). Returns how many took."""
        R = self.rng("rows", key, rect)
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if w < 2 * r or h < 2 * r:
            return 0
        nx, ny = max(1, int(w / pitch + 1e-6)), max(1, int(h / pitch + 1e-6))
        jit = max(0.0, min(w / nx, h / ny) / 2 - r)
        done = 0
        for j in range(ny):
            for i in range(nx):
                x, y = rect[0] + (i + 0.5) * w / nx + R.uniform(-jit, jit), rect[1] + (j + 0.5) * h / ny + R.uniform(-jit, jit)
                if R.random() >= self.P["density"] or not self.free(x, y, r, z):
                    continue
                if fn(x, y, R, done) is not False:
                    self.block(x, y, r, z)
                    done += 1
        return done

    def run(self, asset, a, b, z=0.0, seeds=2, variant=None, lenkey="length", tag="", collide="auto", anchor=None,
            gap=0.0, **params):
        """A straight run of segments from a to b. The asset's local +X follows a->b and its front (-Y) faces the
        RIGHT of the direction of travel. Segment length is chosen inside the asset's own length range."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 0.05:
            return []
        schema, _ = asset_spec(asset)
        spec = schema.get(lenkey, {}) if isinstance(schema.get(lenkey), dict) else {}
        lo, hi = spec.get("min", 1.0), spec.get("max", 8.0)
        hi = min(hi, params.pop("max_seg", 8.0))
        n = max(1, math.ceil(L / hi - 1e-6))
        seg = L / n
        if seg < lo:                       # too short even as one piece: build it at min length and scale it
            n, seg = 1, L
        rot = math.degrees(math.atan2(dy, dx))
        out = []
        R = self.rng("run", asset, tag, a, b)
        for i in range(n):
            t = (i + 0.5) / n
            kw = dict(params)
            kw[lenkey] = max(lo, round(seg - gap, 3))
            kw["seed"] = 1 + (R.randrange(seeds) if seeds > 1 else 0) + params.get("seed", 0)
            sc = 1.0 if seg >= lo else seg / lo
            extra = {"anchor": anchor, "cx": True} if anchor else {"cx": True}
            out.append(self.put(asset, a[0] + dx * t, a[1] + dy * t, z, rot, variant=variant, tag=tag, collide=collide,
                                scale=sc, **kw, **extra))
        return [o for o in out if o is not None]

    def row(self, start, facing, picks, z=0.0, tag="row"):
        """Buildings side by side. start = front-line point where the row begins, facing = (fx, fy) the fronts look at;
        the row advances along the buildings' local +X. picks: [(asset, width, depth, {params})]."""
        fx, fy = facing
        rot = facing_rot(fx, fy)
        ax, ay = -fy, fx
        d = 0.0
        out = []
        for i, (asset, w, depth, kw) in enumerate(picks):
            c = d + w / 2
            kw = dict(kw or {})
            R = self.rng("row", tag, i, asset)
            kw.setdefault("seed", R.randrange(1, 9000))
            variant = kw.pop("variant", "*")
            x, y = start[0] + ax * c, start[1] + ay * c
            if asset is not None:
                out.append(self.put(asset, x, y, z, rot, variant=variant, tag=f"{tag}{i}", collide="building",
                                    width=w, depth=depth, **kw))
            d += w
        return out

    def region(self, name, x0, y0, x1, y1, z=0.0, **kw):
        r = {"name": name, "rect": [round(min(x0, x1), 3), round(min(y0, y1), 3), round(max(x0, x1), 3), round(max(y0, y1), 3)],
             "z": round(z, 3)}
        r.update(kw)
        self.regions.append(r)
        return r

    def ground(self, name, x0, y0, x1, y1, z, mat, **kw):
        t = {"kind": "rect", "name": name, "rect": [round(min(x0, x1), 3), round(min(y0, y1), 3), round(max(x0, x1), 3), round(max(y0, y1), 3)],
             "z": round(z, 3), "mat": mat}
        t.update(kw)
        self.terrain.append(t)

    def skirt(self, name, a, b, z0, z1, mat="ashlar_limestone"):
        """vertical face from a to b (visible from the right of the direction of travel)"""
        self.terrain.append({"kind": "skirt", "name": name, "a": [round(a[0], 3), round(a[1], 3)], "b": [round(b[0], 3), round(b[1], 3)],
                             "z0": round(z0, 3), "z1": round(z1, 3), "mat": mat})

    def shot(self, name, pos, target, fov=55, note="", see=()):
        """a review camera; see = [(x, y, z0, z1)]: what the shot is there to show - no prop may stand on the lines to it"""
        self.shots.append({"name": name, "pos": [round(v, 2) for v in pos], "target": [round(v, 2) for v in target], "fov": fov, "note": note,
                           "_see": [tuple(round(v, 2) for v in q) for q in see]})


# ------------------------------------------------------------------------------------------ resolve-time check


def box_rect(box, x0, y0, x1, y1):
    """turned box (centre x, y, half sizes, angle rad) against an axis-aligned rect: separating axes of both"""
    bx, by, hx, hy, a = box
    ca, sa = abs(math.cos(a)), abs(math.sin(a))
    ex, ey = hx * ca + hy * sa, hx * sa + hy * ca
    if bx + ex <= x0 or bx - ex >= x1 or by + ey <= y0 or by - ey >= y1:
        return False
    c, s_ = math.cos(a), math.sin(a)
    mx, my, rx, ry = (x0 + x1) / 2 - bx, (y0 + y1) / 2 - by, (x1 - x0) / 2, (y1 - y0) / 2
    return abs(mx * c + my * s_) < hx + rx * ca + ry * sa and abs(-mx * s_ + my * c) < hy + rx * sa + ry * ca


def check_resolved(layout, town, src):
    """The placement assertion again, against the built town: `src[i]` is the layout index of resolved item i.
    Every ground prop's real colliders (or its origin, when it has none) must stay off the real stairs_* ramps
    (+ 0.3 m) and off the declared walking routes. Returns the list of violations (empty = pass)."""
    stairs = [(rg["name"], rg["rect"], rg["ramp"]["z0"], rg["ramp"]["z1"]) for rg in town["regions"] if rg["name"].startswith("stairs_") and "ramp" in rg]
    routes = (layout.get("placement") or {}).get("routes", [])
    boxes = {}
    for c in town["colliders"]:
        if c.get("i", -1) < 0:
            continue
        if c["t"] == "c":
            boxes.setdefault(c["i"], []).append((c["c"][0], c["c"][1], c["rad"], c["rad"], 0.0))
        else:
            boxes.setdefault(c["i"], []).append((c["c"][0], c["c"][1], c["h"][0], c["h"][1], math.radians(c["r"])))

    hits = box_rect
    bad = []
    for i, li in enumerate(src):
        it = layout["items"][li]
        if "prop" not in it:
            continue
        x, y, z = town["items"][i]["pos"]
        for box in boxes.get(i) or [(x, y, 0.15, 0.15, 0.0)]:
            for (n, (x0, y0, x1, y1), z0, z1) in stairs:
                if z0 - 0.6 < z < z1 + 0.6 and hits(box, x0 - 0.3, y0 - 0.3, x1 + 0.3, y1 + 0.3):
                    bad.append(f"{it['asset']} ({it['tag']}) at ({x}, {y}) overlaps {n}")
            for (n, x0, y0, x1, y1, rz) in routes:
                if abs(rz - z) < 1.0 and hits(box, x0 + 0.04, y0 + 0.04, x1 - 0.04, y1 - 0.04):
                    bad.append(f"{it['asset']} ({it['tag']}) at ({x}, {y}) stands on route {n}")
    return sorted(set(bad))


# ------------------------------------------------------------------------------------------ the town


def generate(seed=1, **overrides):
    """Lay the town out. The rules are run until the review cameras stop moving (2-6 passes): trees and bulky props
    are placed against the viewpoints / sight cones of the pass before, and a rule that must know a piece placed
    after it (the monument's depth for the flight beside it) reads what the pass before measured (Town.known /
    Town.seen). The result carries the placement report."""
    overrides.pop("_views", None)
    views = cams = None
    known = {}
    for _ in range(6):
        res, T = _generate(seed, views, cams, overrides, known)
        new = [{"name": sh["name"], "pos": sh["pos"], "target": sh["target"], "fov": sh["fov"], "see": sh.get("_see") or []} for sh in T.shots]
        views = [(sh["pos"][0], sh["pos"][1], sh["pos"][2] - EYE) + sh["_dir"] for sh in T.shots if sh["pos"][2] < 12]
        same = cams is not None and all(a["pos"] == b["pos"] for a, b in zip(cams, new)) and known == T.seen
        cams, known = new, T.seen
        if same:
            break
    for sh in T.shots:
        for k in ("_dir", "_dir0", "_nom", "_see"):
            sh.pop(k, None)
    T.cams = cams
    try:
        sight, tested = T.audit()     # raises when a prop stands on a flight of stairs / a walking route / in something built
    finally:
        geo_save()
    res["placement"] = {"stairs": T.stairs, "routes": T.routes, "keep": T.keep, "props": sum(1 for it in T.items if "prop" in it),
                        "nudged": T.nudged, "dropped": T.dropped, "left_out": T.clashed, "sight": sight, "cut": T.cuts, "unbuilt": T.unbuilt,
                        "audit": tested}
    return res


def _generate(seed, _views, _cams, overrides, _known=None):
    P = dict(PARAMS)
    P.update(overrides)
    P["seed"] = seed
    T = Town(seed, P)
    T.known = dict(_known or {})
    T.views = list(_views or [])
    T.cams = list(_cams or [])
    G = T.rng("global")
    eye = EYE

    # ---- frame --------------------------------------------------------------------------------
    # The plan is read off assets/IMG_2863.jpeg: an L-shaped lower square (the fountain square + a neck running north
    # to the clock tower hall) wrapped round a raised terrace. The terrace is an L too: a south arm along the shop
    # row (behind a solid parapet, east of the square) and a north arm that turns west at the head of the stair to the
    # kiosk corner. The stair descends SOUTH, parallel to the shop fronts, into the mouth of the arrival street.
    TH = P["terrace_h"]
    XM, YS = P["wall_x"], P["wall_y"]                                # terrace south arm: wall face (looks west) / stair head line
    X0 = XM - P["plaza_w"]                                           # west building line (inn, hotel)
    Y0 = YS - P["plaza_d"]                                           # south building line
    TX1 = XM + P["terrace_w"]                                        # shop fronts (east)
    XW = X0 + P["neck_w"]                                            # terrace north arm: west wall face (looks west)
    YN = YS + P["neck_d"]                                            # terrace north arm: north wall face (looks north)
    YC = YN + P["court_d"]                                           # the hall court's north edge: the foot of the hall and of whatever stands out before it
    hall_grand = released("civic_hall_grand")                        # THE HALL's switch (see the block below the flights)
    HALL_OUT = 5.0 if hall_grand else 0.0                            # the grand hall's porch and steps stand out this far before its arcade (its contract's bound)
    YH = YC + HALL_OUT                                               # hall front: the building line of the court's north side
    GD = 4.0                                                         # the terrace runs this far south past the building line
    TS = Y0 + GD                                                     # cafe front (closes the terrace's south end)
    RD, BD = P["row_depth"], P["bld_depth"]
    SW = P["street_w"]
    SX1 = XM
    SX0 = SX1 - SW                                                   # arrival street: its east side is the terrace wall
    SY0 = Y0 - P["street_len"]
    scx = (SX0 + SX1) / 2
    COPE = 0.42                                                      # upper paving starts this far behind a wall face
    M = 0.45                                                         # lower walkable ground stops this far short of a wall face
    OUT = -0.07                                                      # ground level outside the walls
    FD = P["fountain_d"]
    nhw = (P["neck_w"] - M) / 2                                     # the neck: half its clear width, and its axis ...
    nax = X0 + nhw
    fx_, fy_ = nax, Y0 + 15.5 + G.uniform(-0.5, 0.5)                 # ... on which the fountain stands
    AP = FD / 2 + 0.6                                                # apothem of the railing round the basin
    sb = P["storeys_bias"]
    EX = TX1 + BD + 2.0                                              # the terrace fill runs under the shop row to here

    # ---- level changes: the two flights (first: the walkable ground is cut round them) -------------------------
    st_w = P["stair_w"]
    n_st = max(3, int(round(TH / 0.15)))                              # 15 cm risers: a broad, gentle flight whatever the terrace's height
    si = T.put("stairs_plaza", XM - (st_w + 1.12) / 2, YS, 0, 0, variant=None, tag="stair_main", collide="none", anchor="maxy", nav="stairs",
               width=st_w, rise=TH, steps=n_st, going=0.34, cheek="raked", cheek_height=round(max(0.45, min(0.95, 0.42 * TH)), 2), handrail=False, lanterns="all",
               intensity=LAMP_CD(SW) / 1.5, seed=G.randrange(1, 99))
    sc = sidecar(T.items[si])
    if sc:                                                            # its east cheek stands against the terrace wall
        T.items[si]["pos"][0] = round(XM - 0.02 - sc["bounds"]["max"][0], 3)
    ms, m_run = T.stair(si, "stair_main")                             # (x0, y0, x1, y1): the foot is y0
    st_x = (ms[0] + ms[2]) / 2
    XS = ms[0]
    ls_w = 2.2
    ls_x = XM + 4.0 + G.uniform(-0.8, 0.8)
    li_ = T.put("stairs_plaza", ls_x, YN, 0, 180, variant=None, tag="stair_lane", collide="none", anchor="maxy", nav="stairs",
                width=ls_w, rise=TH, steps=n_st, going=0.32, cheek="stepped", handrail=True, lanterns="bottom", intensity=LAMP_CD(P["court_d"]) / 2,
                seed=G.randrange(1, 99))
    ls, l_run = T.stair(li_, "stair_lane")                            # the foot is y1

    # ==== THE HALL ===================================================================================================
    # ONE block, ONE switch (`hall_grand`, set with the frame). `civic_hall_grand` - the picture's wide pale hall under
    # its red roof, modelled at its true size - stands here at scale 1 as soon as it is in the release; until then the
    # stand-in is the clock tower hall at its widest, scaled up. Whatever hangs off the hall (the house beside it, the
    # court's ground, lamp axes, planters, notice board, routes, shots, the roofscape behind) reads HALL, never the asset:
    #   i       item index (None: asset missing)       x0, x1   its ends along the court      e   where the next house begins
    #   tower   x of the clock tower (it closes the neck's axis), tower_y its front, top the height of the spire's tip
    #   door    (x, y) foot of the entrance steps      porch    (x0, x1) of what stands out south of YH, or None
    #   step    the neck's walk stops this far short of YH      back   y of the rear wall
    #   wall    (x0, x1) plain front wall at eye height (notice board), or None        urns   how its planters are set
    if hall_grand:
        # contract (tools/tsq brief civic_hall_grand): front -Y; the UPPER hall's front wall on local y = 0, its body
        # back to +depth; the arcade storey and its terrace stand out terrace_depth to -Y, porch and steps <= 5 m
        # more; the tower on x = tower_offset; end pavilions bulge 1.7 m past +-width/2. Everything else: the author's defaults
        g_spec, _ = asset_spec("civic_hall_grand")

        def g_def(k, d):
            v = g_spec.get(k)
            return v.get("default", d) if isinstance(v, dict) else d
        g_w = max(30.0, min(44.0, P["hall_w"]))                       # its own width range: never scaled
        g_td, g_d, g_pw = 5.0, 12.0, 7.0                              # arcade terrace depth, depth of the hall behind it, porch width
        g_x1 = min(nax + g_w / 2, TX1 - 7.0 - 1.75)                   # (a house's width is left between its east pavilion and the terrace corner)
        g_cx = g_x1 - g_w / 2
        g_to = max(-min(11.0, g_w / 2 - 9.5), min(11.0, g_w / 2 - 9.5, nax - g_cx))  # the tower closes the neck's axis
        g_po = max(-14.0, min(14.0, g_to + 8.0, ls[0] - 2.0 - g_pw / 2 - g_cx))   # the porch beside it as designed, west of the lane stair
        hall_i = T.put("civic_hall_grand", round(g_cx, 3), YH + g_td, 0, 0, variant=None, tag="hall", collide="building", width=round(g_w, 2),
                       depth=g_d, terrace_depth=g_td, tower_offset=round(g_to, 2), porch_offset=round(g_po, 2), porch_width=g_pw,
                       end_pavilions="right", seed=G.randrange(1, 999))
        g_px = g_cx + g_po
        HALL = {"i": hall_i, "grand": True, "x0": g_cx - g_w / 2, "x1": g_x1, "e": g_x1 + 1.75, "tower": g_cx + g_to, "tower_y": YH + g_td - 1.6,
                "top": g_def("tower_height", 25.0) + g_def("spire_height", 11.0) + 1.5, "door": (g_px, YC),
                "porch": (g_px - g_pw / 2 - 0.9, g_px + g_pw / 2 + 0.9), "step": 0.6, "back": YH + g_td + g_d, "wall": None, "urns": "sockets"}
    else:
        hall_s = max(1.0, min(P.get("hall_scale", 1.3), 1.3))        # the picture's hall dwarfs the houses: the asset at its widest and tallest, scaled up
        hall_w = max(22.0 * hall_s, min(P["hall_w"], 32.0 * hall_s, 2 * (TX1 - 7.0 - nax)))
        hx = nax                                                      # its door and tower close the neck's axis (its west wing runs on behind the last house of the west row)
        # (the picture's hall is cream stone glowing warm: pale render between pale dressings, glass and lanterns turned
        # up; the dial glows no brighter than its numerals can stand - at 10 it read as a blank disc from the street)
        hall_i = T.put("clock_tower_hall", hx, YH, 0, 0, variant=None, tag="hall", collide="building", scale=hall_s, width=round(hall_w / hall_s, 2),
                       depth=13.0, turret="right", tower_height=17.5, spire_height=8.2,
                       stone_tint="#f2e6c8", wall_finish="render", wall_tint="#fff0d0", roof="clay", dormers=3, window_glow=10.0, dial_glow=4.5,
                       lantern_intensity=130.0, moss=0.1, seed=G.randrange(1, 999))
        # the tower's front between the portal and the belfry is the hall's one face no lamp reaches: a wall lantern either
        # side of its window washes the pale wall (the picture's tower stands lit over the square)
        if hall_i is not None:
            for sgn in (-1, 1):
                T.put("wall_lantern", hx + sgn * 2.0 * hall_s, YH - 0.4 * hall_s, 9.3 * hall_s, 0, variant="grand_scroll", tag="hall_lantern", collide="none",
                      scale=2.0, of=hall_i, quiet=True, intensity=40.0, seed=3)
        HALL = {"i": hall_i, "grand": False, "x0": hx - hall_w / 2, "x1": hx + hall_w / 2, "e": hx + hall_w / 2, "tower": hx, "tower_y": YH,
                "top": (17.5 + 8.2) * hall_s, "door": (hx, YH), "porch": None, "step": 2.6, "back": YH + 13.0,
                "wall": (hx + hall_w / 2 - 6.0, hx + hall_w / 2 - 1.0), "urns": (round(4.2 * hall_s, 2), round(0.72 * hall_s, 2))}
    # ==== (end of THE HALL) ==========================================================================================

    # ---- walkable regions (the lower ones never reach under a terrace or a flight) ----------------------------
    T.region("plaza", X0, Y0, XS - 0.1, YS - M, 0)
    T.region("stair_bay", XS - 0.1, Y0, XM - M, ms[1] + 0.15, 0)
    T.region("neck", X0, YS - M, XW - M, YN + M, 0)
    if HALL["porch"]:                                                 # a porch stands in the court: the walkable ground is cut round it
        T.region("court", X0, YN + M, ls[0] - 0.1, YC, 0)
        T.region("court_hw", X0, YC, max(X0 + 0.5, HALL["porch"][0]), YH, 0)
        T.region("court_he", min(ls[0] - 0.6, HALL["porch"][1]), YC, ls[0] - 0.1, YH, 0)
    else:
        T.region("court", X0, YN + M, ls[0] - 0.1, YH, 0)
    T.region("court_n", ls[0] - 0.1, ls[3] - 0.15, ls[2] + 0.1, YH, 0)
    T.region("court_e", ls[2] + 0.1, YN + M, TX1, YH, 0)
    T.region("street", SX0, SY0, SX1, Y0 - GD, 0)
    T.region("street_n", SX0, Y0 - GD, XM - M, Y0, 0)
    T.region("terrace_s", XM, TS, TX1, YS, TH)
    T.region("terrace_n", XW, YS, TX1, YN, TH)

    # ---- ground --------------------------------------------------------------------------------
    T.ground("plaza", X0 - 1, Y0 - 0.3, XM + COPE, YS + COPE, 0, "cobble_granite", wet=0.62)
    T.ground("neck", X0 - 1, YS + COPE, XW + COPE, YN - COPE, 0, "cobble_granite", wet=0.55)
    T.ground("court", X0 - 1, YN - COPE, TX1 + 1, YH + 1, 0, "cobble_granite", wet=0.5)
    T.ground("street", SX0 - 1, SY0 - 1, SX1 + 1, Y0 - 0.3, 0, "sett_basalt", wet=0.62)
    T.ground("terrace_s", XM + COPE, Y0 - GD - 0.3, EX, YS + COPE, TH, "flagstone_york", wet=0.38)
    T.ground("terrace_n", XW + COPE, YS + COPE, EX, YN - COPE, TH, "flagstone_york", wet=0.38)
    T.ground("stair_landing", XS - 0.3, YS - 0.02, XM + COPE, YS + COPE, TH, "flagstone_york")
    T.ground("lane_landing", ls[0] - 0.3, YN - COPE, ls[2] + 0.3, YN + 0.02, TH, "flagstone_york")
    T.ground("base", -120, -130, 120, 120, OUT, "soil", step=6.0, bump=0)
    # sealing faces behind the wall assets (visible from the right of the direction of travel)
    T.skirt("sk_ts_w", (XM + COPE, YS + COPE), (XM + COPE, Y0 - GD - 0.3), -0.2, TH)
    T.skirt("sk_ts_s", (XM + COPE, Y0 - GD - 0.3), (EX, Y0 - GD - 0.3), -0.2, TH)
    T.skirt("sk_t_e", (EX, Y0 - GD - 0.3), (EX, YN - COPE), -0.2, TH)
    T.skirt("sk_tn_n", (EX, YN - COPE), (XW + COPE, YN - COPE), -0.2, TH)
    T.skirt("sk_tn_w", (XW + COPE, YN - COPE), (XW + COPE, YS + COPE), -0.2, TH)
    T.skirt("sk_tn_s", (XW + COPE, YS + COPE), (XM + COPE, YS + COPE), -0.2, TH)

    # ---- terrace walls: ONE low ashlar wall per edge, from the square's paving up to a seat-high parapet ---------------
    # (the picture's terrace stands behind a low wall under a thin coping, with benches against it - not a rampart: the
    # wall is built at its true height in 30 cm courses, its piers carry the lanterns that light its foot and the
    # terrace's edge, and nothing is stacked on its coping)
    WALL_T = 0.44                                                     # its back face reaches the terrace paving (COPE)
    WALL_UP = 0.45                                                    # the parapet: seat height above the terrace
    WALL = {"thickness": WALL_T, "facing": "ashlar", "coping": "flat", "kind": "freestanding", "stone_size": 1.1, "plinth": 0.16,
            "pier_width": 0.5, "pier_rise": 0.3, "pier_cap": "lantern", "lantern_cd": LAMP_CD(8.0), "moss": 0.4, "age": 0.6}

    def edge(a, b, z_low, h, tag, piers="left", seg=5.4):
        """the terrace wall along a->b (its face to the right of travel, on the line a->b): piers with a lantern at
        the start of every bay (`piers`: which ends of each bay)"""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        nx, ny = dy / L, -dx / L                                      # right of travel = low side
        off = -WALL_T / 2
        a2, b2 = (a[0] + nx * off, a[1] + ny * off), (b[0] + nx * off, b[1] + ny * off)
        return T.run("wall_stone_run", a2, b2, z_low, tag=tag, seeds=3, max_seg=seg, height=round(h + WALL_UP, 2),
                     **dict(WALL, pier_ends=piers))

    edge((XM, YS), (XM, Y0 - GD), 0, TH, "tw_a")                      # travel south: faces west (square, street)
    edge((XW, YS), (XS, YS), 0, TH, "tw_b", piers="both")             # travel east: faces south (square): the kiosk corner
    edge((XW, YN), (XW, YS), 0, TH, "tw_c")                           # travel south: faces west (neck)
    edge((TX1, YN), (ls[2], YN), 0, TH, "tw_d1", piers="right")       # travel west: faces north (hall court); its pier at the lane stair
    edge((ls[0], YN), (XW, YN), 0, TH, "tw_d2")

    def growth(a, b, n, key, taken, z=0.0, h=(1.6, 2.6), w=(1.0, 2.0), variant="*"):
        """Climbers on a wall face: up to n ivy patches at seeded spots along a->b (the face looks to the RIGHT of
        the direction of travel) that keep clear of what already hangs on that wall (`taken`: [(distance along, half
        width)], extended with the new patches)."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        R = T.rng("growth", key, a, b)
        rot = facing_rot(dy / L, -dx / L)
        done = 0
        for _ in range(n * 30):
            if done >= n:
                break
            d, ww, hh = R.uniform(0.8, L - 0.8), R.uniform(*w), R.uniform(*h)
            if any(abs(d - td) < tw + ww / 2 + 0.25 for td, tw in taken):
                continue
            taken.append((d, ww / 2))
            T.put("ivy_patch", a[0] + dx * d / L, a[1] + dy * d / L, z, rot, variant=variant, tag=key, collide="none", width=ww, height=hh,
                  seed=R.randrange(99))
            done += 1
        return done

    # the long west faces: the wall's own piers and lanterns set its rhythm (the picture's wall is plain ashlar under a
    # coping: no added buttresses); the stair bay is divided into even bays for its benches, ivy between them
    a_taken, c_taken, a_buts = [(YS - (YS + ms[1]) / 2, (YS - ms[1]) / 2 + 0.5)], [], []
    IVY_H = (0.8, max(0.9, min(2.1, TH + WALL_UP - 0.25)))            # the ivy stays under the coping
    for i, yy in enumerate([ms[1] - 3.2, ms[1] - 8.6, ms[1] - 14.0, Y0 - GD / 2]):
        if Y0 - GD + 1.0 < yy < ms[1] - 1.0:
            a_buts.append(yy)
    a_taken += [(YS - (ya + yb) / 2, 1.2) for ya, yb in zip(a_buts, a_buts[1:])]          # the ivy grows between the benches / urns of the wall foot
    c_n = max(1, int(round((YN - YS - 1.8) / 3.2 * min(1.0, P["density"]))))
    c_taken += [(1.2 + (i + 0.5) * (YN - YS - 1.8) / c_n, 0.75) for i in range(c_n)]
    growth((XM - 0.04, YS), (XM - 0.04, Y0 - GD), 3, "tw_ivy_a", a_taken, h=IVY_H, w=(0.9, 1.7), variant=["young", "old_leaning"])
    growth((XW - 0.04, YN), (XW - 0.04, YS), 2, "tw_ivy_c", c_taken, h=IVY_H, w=(0.9, 1.7), variant=["young", "old_leaning"])
    growth((TX1, YN + 0.04), (ls[2] + 0.4, YN + 0.04), 1, "tw_ivy_d", [], h=IVY_H, w=(1.0, 1.8))

    # ---- buildings -------------------------------------------------------------------------------
    # north: the hall closes the neck (THE HALL, above); its tower stands over the kiosk corner seen from the square
    # north-east: a tower house closes the court beside the hall, a stone house closes its east end
    ne_w = TX1 - HALL["e"]
    ne_picks = [("tower_house", min(11.0, ne_w) if ne_w - 11.0 < 3.5 else 9.0, BD, {"storeys": 3, "roof": "tile", "plaster": "#f0e2c0", "lit": 0.9})]
    if ne_w - ne_picks[0][1] >= 3.5:
        ne_picks.append(("house_stone" if ne_w - 9.0 >= 5.0 else "townhouse_narrow", ne_w - 9.0, BD, {"storeys": 3}))
    ne_row = T.row((HALL["e"], YH), (0, -1), ne_picks, tag="ne")
    lane_len = YH - YN                                                # (a court deeper than one house is wide: a narrow house beside it)
    lane_picks = [("townhouse_narrow", 4.5, BD, {"storeys": 3})] if lane_len > 12.0 else []
    lane_picks.append(("house_stone", lane_len - sum(p_[1] for p_ in lane_picks), BD, {"storeys": 3, "lit": 0.85, "lantern_cd": 24.0}))
    lane_row = T.row((TX1, YH), (-1, 0), lane_picks, tag="lane")
    # east, on the terrace (faces west), north -> south: the awning house, LUGGAGE, DRESS & SUIT, the big timber house
    sh_len = YN - TS
    w_lug, w_tai = round(G.uniform(6.6, 7.6), 1), round(G.uniform(7.6, 8.6), 1)
    w_hs = round(max(5.5, min(8.5, sh_len - 10.8 + G.uniform(0.0, 1.0) - w_lug - w_tai)), 1)
    w_tim = sh_len - w_hs - w_lug - w_tai
    # (the picture's shop terrace is the warm side of the square: every room lit, the door lanterns turned up)
    shop_row = T.row((TX1, YN), (-1, 0), [("house_stone", w_hs, BD, {"storeys": 2 + max(0, sb), "lit": 0.9, "lantern_cd": 30.0}),
                                          ("shop_luggage", w_lug, BD, {"variant": None, "storeys": 2, "lit": 0.95, "lantern_cd": 15.0}),
                                          ("shop_tailor", w_tai, BD, {"variant": None, "storeys": 2, "tints": 2, "wear": 0.65, "lit": 1.0, "lantern_cd": 24.0}),
                                          ("house_timber", w_tim, BD, {"variant": None, "storeys": 3, "roof": "eaves_front", "dormers": 3,
                                                                       "window_boxes": 4, "pattern": "close_studding", "lit_ratio": 0.9,
                                                                       "lantern_cd": 40.0})], z=TH, tag="shop")
    # the cafe with its gallery closes the south end of the terrace (faces north, up the promenade)
    cafe_w = max(8.0, P["terrace_w"] - 1.0)
    cafe_i = T.row((TX1, TS), (0, 1), [("cafe_gallery", cafe_w, BD, {"storeys": 2, "gallery_depth": 3.0, "light": 1.4})], z=TH, tag="cafe")[0]
    # west row (faces east), south -> north: the timber inn in the corner, the hotel, then the houses of the neck
    w_len = YC - Y0
    inn_w = 10.0
    hotel_w = round(min(20.0, max(12.0, P["plaza_d"] - inn_w + 1.0 + G.uniform(-0.5, 0.5))), 1)
    rest = w_len - inn_w - hotel_w
    n1, n3 = round(G.uniform(4.6, 5.4), 1), round(G.uniform(5.6, 6.6), 1)
    n4 = round(G.uniform(3.8, 4.6), 1)
    n2 = rest - n1 - n3 - n4
    west = [("house_timber", inn_w, RD, {"variant": "eaves_inn", "storeys": 2, "window_boxes": 3}),
            ("hotel", hotel_w, RD, {"storeys": 3 + max(0, sb), "sign_side": "right", "balconies": "centre"}),
            ("townhouse_narrow", n1, BD, {"storeys": 3}),
            ("house_stone" if 5.0 <= n2 <= 12.0 else "house_timber", n2, BD, {"storeys": 3}),
            ("house_timber", n3, BD, {"storeys": 2, "window_boxes": 2}),
            ("townhouse_narrow", n4, BD, {"storeys": 4})]
    if YH - YC >= 3.5:                                                # (the west row runs on to the hall's line)
        west.append(("townhouse_narrow" if YH - YC <= 5.5 else "house_stone", YH - YC, BD, {"storeys": 3}))
    west_row = T.row((X0, Y0), (1, 0), west, tag="west")
    inn_i, hotel_i = west_row[0], west_row[1]
    hotel_y0 = Y0 + inn_w
    # south row (faces north), east -> west: a corner shop on the street, a house in the corner
    s_len = SX0 - X0
    south_row = T.row((SX0, Y0), (0, 1), [("shop_corner", round(s_len * 0.5, 1), BD, {"storeys": 2}),
                                          ("house_stone", s_len - round(s_len * 0.5, 1), BD, {"storeys": 3})], tag="south")
    # the arrival street: houses below the cafe on its east side, houses behind the corner shop on its west side
    e_len = (Y0 - GD) - SY0
    e_picks = [("house_timber", 5.5, BD, {"storeys": 2, "window_boxes": 1})]
    if e_len - 5.5 > 9.0:                                              # (a row of two or three houses, by the street's length)
        e_picks.append(("townhouse_narrow", 4.5, BD, {"storeys": 3}))
    e_rest = e_len - sum(p_[1] for p_ in e_picks)
    e_picks.append(("townhouse_narrow" if e_rest <= 5.5 else "house_stone", e_rest, BD, {"storeys": 3}))
    street_e = T.row((SX1, Y0 - GD), (-1, 0), e_picks, tag="street_e")
    sw_len = (Y0 - BD) - SY0
    w_picks = [("townhouse_narrow", 4.5, BD, {"storeys": 4})] if sw_len - 4.5 >= 3.5 else []
    w_rest = sw_len - sum(p_[1] for p_ in w_picks)
    w_picks.append(("house_stone" if w_rest >= 5.0 else "townhouse_narrow", w_rest, BD, {"storeys": 2 if w_picks else 3}))
    street_w = T.row((SX0, SY0), (1, 0), w_picks, tag="street_w")
    # street end: wall, piers and a closed iron gate
    gw = 3.2
    # (the wall runs up to the gate's own posts: piers out here, 30 m from the nearest camera, cost 100 k triangles)
    T.run("wall_stone_run", (scx - gw / 2 - 0.05, SY0 + 0.3), (SX0, SY0 + 0.3), 0, tag="st_end", height=3.0)   # travel west: faces north
    T.run("wall_stone_run", (SX1, SY0 + 0.3), (scx + gw / 2 + 0.05, SY0 + 0.3), 0, tag="st_end", height=3.0)
    T.put("iron_gate", scx, SY0 + 0.3, 0, 180, variant="*", tag="st_end", width=gw, height=2.9, intensity=LAMP_CD(SW) / 1.5, seed=3)
    # the HOTEL blade: on the hotel's south corner, low over the square (the picture's left edge)
    neon_y = hotel_y0 + 1.0
    # (the hotel carries the blade itself, on its south corner - `sign_side` - so no separate neon is hung)

    rows = [b for b in [hall_i] + ne_row + lane_row + shop_row + [cafe_i] + west_row + south_row + street_e + street_w if b is not None]

    # ---- the two views the scenery answers to (the review cameras themselves are set at the end) ------------------
    # the camera of the picture: high over the square's south-west corner, looking over the fountain to the stair and
    # the shop terrace - hall upper left, the hotel's blade on the left edge. Its tilt and lens are a rule: the frame
    # holds the foot of the fountain's railing at the bottom and the tip of the hall's spire at the top (the
    # longest lens of 64-76 deg that does)
    REF = (X0 + 2.0, Y0 + 2.0)
    cw_y = (ls[3] + YC) / 2                                           # the court's walk passes north of the lane stair's foot
    over = ((X0 - 3.0, Y0 - 15.0, 36.0), (XM - 8.0, YS - 1.0, 2.0), 44)   # the bird's eye (position, target, fov)

    def frame_v(pos, at, fov, x, y, z):
        """height of a point in a review frame (-1 bottom edge .. 1 top edge; None behind the lens)"""
        fx, fy, fz = at[0] - pos[0], at[1] - pos[1], at[2] - pos[2]
        L = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / L, fy / L, fz / L
        h = math.hypot(fx, fy) or 1e-6
        rx, ry = fy / h, -fx / h                                      # right; up = right x forward
        ux, uy, uz = ry * fz, -rx * fz, rx * fy - ry * fx
        dx, dy, dz = x - pos[0], y - pos[1], z - pos[2]
        d = dx * fx + dy * fy + dz * fz
        return None if d < 0.1 else (dx * ux + dy * uy + dz * uz) / (d * math.tan(math.radians(fov) / 2))

    ref_pos, ref_at = (REF[0], REF[1], 11.0), (XW + 5.0, YS + 3.0)
    d_f = math.hypot(fx_ - REF[0], fy_ - REF[1]) or 1.0
    r_f = AP / math.cos(math.pi / 8) + 0.3                            # the railing's corners and their lanterns
    f_near = (fx_ - (fx_ - REF[0]) / d_f * r_f, fy_ - (fy_ - REF[1]) / d_f * r_f, 0.0)
    for ref_fov in range(64, 78, 2):
        lo, hi = -14.0, 16.0                                          # the height aimed at: higher = the fountain sinks in the frame
        for _ in range(28):
            ref_z = (lo + hi) / 2
            v = frame_v(ref_pos, ref_at + (ref_z,), ref_fov, *f_near)
            lo, hi = (ref_z, hi) if v is not None and v > -0.92 else (lo, ref_z)
        v = frame_v(ref_pos, ref_at + (ref_z,), ref_fov, HALL["tower"], HALL["tower_y"], HALL["top"])
        if v is not None and v <= 0.97:
            break

    # ---- the roofscape beyond ---------------------------------------------------------------------
    # the bird's-eye camera the scenery is laid out for (position, target, fov)
    bz = 0
    BR = 8.6
    # ==== THE KEEP ===================================================================================================
    # ONE block, ONE switch: the second tower of the picture, close behind the houses north-east of the square on the
    # bearing the reference view sees it over the terrace roofs; it stands before the rows are laid so they slide
    # clear of it. `tower_keep_ornate` - the banded sandstone keep modelled at its true size - stands here at scale 1,
    # as its author built it, as soon as it is in the release (origin: centre of its square plan, show face -Y turned to
    # the reference view; it carries its own windows and lanterns: nothing is hung on it). Until then the stand-in is
    # the town gate built without its gatehouse and stood end-on at tower scale, a stained window and two lanterns on
    # the end that looks at the square.
    keep_i = None
    ka = math.radians(P.get("keep_bearing", 48.0))
    ref = REF                                                         # the reference viewpoint
    keep_grand = released("tower_keep_ornate")
    if keep_grand:
        k_spec, _ = asset_spec("tower_keep_ornate")
        k_w = (k_spec.get("width") or {}).get("default", 8.5) if isinstance(k_spec.get("width"), dict) else 8.5
        kr = math.hypot(TX1 - ref[0], YH + BD - ref[1]) + 2.0 + 0.71 * k_w + 1.0        # (its corners, turned to the view, and its bartizans)
        keep_i = T.backdrop("bd_landmark", ref[0] + kr * math.cos(ka), ref[1] + kr * math.sin(ka), facing_rot(-math.cos(ka), -math.sin(ka)), None, bz,
                            asset="tower_keep_ornate", variant="default", reach=12.0, step=3.0, seed=7)
    elif asset_path("gate_arch_town"):
        ks, kw_, kd_, kh_ = 3.0, 8.0, 5.0, 10.0                       # scale; the gate's width (the keep's long side), depth, height as built
        KH = kh_ * ks                                                 # the keep's height
        kr = math.hypot(TX1 - ref[0], YH + BD - ref[1]) + 2.0 + kw_ * ks / 2
        keep_i = T.backdrop("bd_landmark", ref[0] + kr * math.cos(ka), ref[1] + kr * math.sin(ka), math.degrees(ka) + 180.0, None, bz,
                            asset="gate_arch_town", variant="plain_wallwalk", scale=ks, reach=12.0, step=3.0, width=kw_, depth=kd_, height=kh_,
                            passage_width=3.0, passage_height=4.0, gatehouse=False, banners=False, posters=False, lantern_intensity=30.0,
                            stone="#e2d2a8", moss=0.15, seed=7)
        if keep_i is not None:
            kp = T.items[keep_i]["pos"]
            ux, uy = -math.cos(ka), -math.sin(ka)                     # from the keep towards the reference view
            face = kw_ * ks / 2 + 0.05
            wrot = facing_rot(ux, uy)
            kb = (sidecar(T.items[keep_i]) or {}).get("bounds")           # the end wall's middle as built (the piers' buttresses are not symmetric)
            mid_ = (kb["min"][1] + kb["max"][1]) / 2 * ks if kb else -0.056 * kd_ * ks
            half = kd_ * ks / 2

            def on_end(lat, zf):
                """a point on the keep's end wall: `lat` m off its middle, at the fraction zf of its height"""
                return kp[0] + ux * face - uy * (mid_ + lat), kp[1] + uy * face + ux * (mid_ + lat), bz + zf * KH
            # the upper stage, the part that stands over the roofs: a tall stained window in the middle with a lantern
            # either side of it, a small lit window in each flank
            if asset_path("window_stained_arch"):
                wx, wy, wz = on_end(0.0, 0.64)
                T.put("window_stained_arch", wx, wy, wz, wrot, variant=None, tag="bd_keep", collide="none", scale=round(0.73 * ks, 2), of=keep_i, quiet=True,
                      width=1.6, height=3.4, tracery="twin", palette="jewel", glow=5.0, intensity=30.0, moss=0.2, seed=3)
                for sgn in (-1, 1):
                    wx, wy, wz = on_end(sgn * 0.64 * half, 0.70)
                    T.put("window_stained_arch", wx, wy, wz, wrot, variant=None, tag="bd_keep", collide="none", scale=round(0.68 * ks, 2), of=keep_i, quiet=True,
                          width=0.8, height=1.8, tracery="single", palette="ember", glow=4.0, intensity=0.0, moss=0.2, seed=5)
            for sgn in (-1, 1):
                wx, wy, wz = on_end(sgn * 0.37 * half, 0.69)
                T.put("wall_lantern", wx, wy, wz, wrot, variant="grand_scroll", tag="bd_keep", collide="none", scale=round(0.9 * ks, 2), of=keep_i, quiet=True,
                      intensity=40.0, seed=3)
    # ==== (end of THE KEEP) ==========================================================================================
    n_x = (X0 - RD + TX1) / 2
    # The first two rings stand close enough to be read as houses: they are only ever built from the asset's near
    # variants at no more than 1.35 x (a far-layer hull - plain, sunk in haze, no windows in backs and ends - belongs to
    # the third ring; a row scaled further shows metre-long roof courses and giant windows from above)
    asset_spec("backdrop_row")
    BD_NEAR = ["default"] + [n for n, v in _VARIANT.get("backdrop_row", {}).items() if v.get("layer", "near") != "far"]
    inner = [("bd_n", n_x, HALL["back"] + BR, 0, 56.0, 11), ("bd_e", TX1 + BD + BR, (TS + YN) / 2 - 1.5, -90, 48.0, 12),
             ("bd_w", X0 - RD - BR, (Y0 + YH) / 2, 90, 56.0, 13), ("bd_s", scx, SY0 - 8.0, 180, 48.0, 14),
             ("bd_sw", (X0 - RD + SX0 - BD) / 2 - 1.5, Y0 - BD - BR, 180, 18.0, 15), ("bd_se", SX1 + BD + BR, (SY0 + Y0 - GD) / 2 - 4.0, -90, 16.0, 16)]
    for (tg, bx, by, br, bl, bs) in inner:
        # (the east row carries no tower of its own where the keep stands over it: the keep is the landmark there)
        keep = tg == "bd_e" and keep_i is not None
        T.backdrop(tg, bx, by, br, bl, bz, step=1.0, variant=BD_NEAR, spill=SPILL, seed=bs, **({"landmark": "none"} if keep else {}))
    R3 = T.rng("ring3")
    R2 = T.rng("ring2")
    # (no spires in the second ring: the hall's clock tower and the keep are the landmarks of the skyline)
    far = {"lamps": 0, "clock": False, "height_min": 8.0, "height_max": 12.0, "depth": 10.0, "landmark_height": 3.0, "variant": BD_NEAR}
    for (tg, bx, by, br, bl, bs) in inner[:4]:
        ox, oy = rot2(0, 1, br)                                       # away from the square
        ax, ay = rot2(1, 0, br)
        for k, side in enumerate((-1, 1)):
            ln = round(R2.uniform(30.0, 44.0), 1)
            push, slide = R2.uniform(17.0, 21.0), side * (bl * 0.5 - R2.uniform(4.0, 10.0))
            T.backdrop(tg + "_far", bx + ox * push + ax * slide, by + oy * push + ay * slide, br + R2.uniform(-7, 7), ln, bz,
                       scale=round(R2.uniform(1.2, 1.35), 2), reach=44.0, seed=R2.randrange(100, 999),
                       **dict(far, landmark="none"))
    mid = ((X0 - RD + TX1 + BD) / 2, (SY0 + YC + 13.0) / 2)
    GAP = 3

    def skyline(view, vz, span):
        """bearing (whole degrees of `span`, seen from `view` at height vz) -> distance of the nearest row of the
        first two rings whose roofs reach the skyline there (top no more than 5 deg under the horizon)"""
        cov = {}
        for it in T.items:
            if not it["tag"].startswith("bd_") or it["tag"] == "bd_sky":
                continue
            pts, k = T.outline(it), it.get("scale", 1.0)
            top = T.tall(it) if pts else (it["params"].get("height_max", 14.0) + 5.0) * k
            if not pts and it["asset"] == "backdrop_row":             # not built yet: the row as its params describe it
                hl, hd = it["params"].get("length", 30.0) / 2 * k, it["params"].get("depth", 11.0) / 2 * k + 1.5
                pts = [(it["pos"][0] + wx, it["pos"][1] + wy) for wx, wy in (rot2(lx, ly, it["rot"]) for lx in (-hl, hl) for ly in (-hd, hd))]
            d = math.hypot(it["pos"][0] - view[0], it["pos"][1] - view[1])
            if not pts or d < 30.0 or math.degrees(math.atan2(top - vz, d)) < -5.0:
                continue
            bs = [math.degrees(math.atan2(py - view[1], px - view[0])) for px, py in pts]
            bs = [bs[0] + ((q - bs[0] + 180.0) % 360.0 - 180.0) for q in bs]
            for k in range(int(math.floor(min(bs))), int(math.ceil(max(bs))) + 1):
                if k % 360 in span:
                    cov[k % 360] = min(cov.get(k % 360, 1e9), d)
        return cov

    def close_ring(view, vz, b0, b1, key):
        span = [k % 360 for k in range(int(round(b0)), int(round(b1)))]
        RG = T.rng("gap", key)
        for _ in range(8):
            cov = skyline(view, vz, set(span))
            runs, cur = [], []
            for k in span + ([span[0]] if len(span) == 360 else []):
                if k in cov:
                    if len(cur) >= GAP:
                        runs.append(cur)
                    cur = []
                else:
                    cur.append(k)
            if len(cur) >= GAP:
                runs.append(cur)
            runs = [r for r in runs if len(r) < 300]
            if not runs:
                return
            run = max(runs, key=len)
            near = [cov[k] for k in ((run[0] - 1) % 360, (run[-1] + 1) % 360) if k in cov]
            d = max(60.0, min(near) if near else 90.0)
            sc_ = round(RG.uniform(1.25, 1.4), 2)
            a = math.radians(run[0] + (len(run) - 1) / 2.0)
            ln = max(12.0, min(40.0, 4.0 * math.ceil(2 * d * math.tan(math.radians(len(run) + 4) / 2) / sc_ / 4.0)))   # (whole 4 m: few distinct builds)
            qx, qy = view[0] + d * math.cos(a), view[1] + d * math.sin(a)
            if T.backdrop("bd_gap", qx, qy, facing_rot(view[0] - qx, view[1] - qy), round(ln, 1), bz, scale=sc_, fill=True, reach=44.0,
                          lit_ratio=0.5, seed=RG.randrange(100, 999), **far) is None:
                return
    close_ring(mid, 0.0, 0, 360, "mid")
    oyaw = math.degrees(math.atan2(over[1][1] - over[0][1], over[1][0] - over[0][0]))
    ohf = math.degrees(math.atan(math.tan(math.radians(over[2]) / 2) * ASPECT))
    close_ring(over[0][:2], over[0][2], oyaw - ohf, oyaw + ohf, "over")
    sky_r = max(math.hypot(TX1 + BD - mid[0], YH + 13 - mid[1]), math.hypot(X0 - RD - mid[0], SY0 - mid[1])) + 70.0
    sky_s, sky_d = 1.5, 22.0                                          # row scale, depth of a far row as built at that scale
    # the far layer: a castle on its hill well beyond the third ring (which hides the hill), on the bearing the
    # reference view looks over the shop terrace's roofs - the lowest stretch of its skyline, right of the keep
    if P.get("far_landmark", True) and released("castle_far"):
        fa, fd = math.radians(P.get("far_bearing", 30.0)), max(sky_r + 100.0, 250.0)
        fx, fy = mid[0] + fd * math.cos(fa), mid[1] + fd * math.sin(fa)
        T.backdrop("bd_far_landmark", fx, fy, facing_rot(mid[0] - fx, mid[1] - fy), None, bz, asset="castle_far",
                   variant="default", scale=1.5, reach=40.0, seed=1, floodlit=0.3, haze=0.85, lit_ratio=0.45, glow=14.0)
    sky_n = max(7, int(math.ceil(2 * math.pi * sky_r / (60.0 * sky_s))))
    a0 = R3.uniform(0, 360)
    for k in range(sky_n):
        a = math.radians(a0 + 360.0 * k / sky_n)
        rr = sky_r + (k % 2) * (sky_d + 3.0) + R3.uniform(-2.0, 2.0)
        qx, qy = mid[0] + rr * math.cos(a), mid[1] + rr * math.sin(a)
        ln = min(60.0, 2 * (rr - sky_d / 2) * math.tan(math.pi / sky_n) / sky_s * (1.0 if k % 2 else 0.92))
        T.backdrop("bd_sky", qx, qy, facing_rot(mid[0] - qx, mid[1] - qy), round(ln, 1), bz, scale=sky_s, layer="far",
                   landmark=R3.choice(("spire", "tower", "none", "none")), clock=False, lamps=0, height_min=round(R3.uniform(5.0, 6.5), 1), height_max=round(R3.uniform(8.5, 11.5), 1),
                   lit_ratio=round(R3.uniform(0.42, 0.6), 2), glow=9.0, haze=0.75, seed=R3.randrange(100, 999))

    # ---- walking routes: ground props keep off them (Town.blocked) -----------------------------------
    foot = 2.0                                                        # clear approach in front of a flight of stairs
    T.route("street_walk", scx - 1.4, SY0 + 1.5, scx + 1.4, Y0)
    T.route("bay_walk", st_x - 1.4, Y0, min(st_x + 1.4, XM - 1.45), ms[1] - foot)                   # street -> foot of the stair (clear of the wall's benches)
    T.route("stair_main_foot", ms[0], ms[1] - foot, ms[2], ms[1])
    T.route("stair_main_head", ms[0], YS, ms[2], YS + 2.0, TH)
    T.route("fountain_e", fx_ + AP + 0.8, fy_ - 1.2, st_x - 1.4, fy_ + 1.2)                         # stair / street -> fountain
    T.route("fountain_w", X0 + 2.2, fy_ - 1.2, fx_ - AP - 0.8, fy_ + 1.2)                           # fountain -> hotel
    T.route("fountain_n", fx_ - 1.2, fy_ + AP + 0.8, fx_ + 1.3, YS - M)                             # fountain -> neck
    T.route("neck_walk", nax - 1.4, YS - M, nax + 1.4, YH - HALL["step"])                                    # neck -> hall
    T.route("court_walk", nax + 1.4, cw_y - 1.1, ls[2] + 0.4, cw_y + 1.1)                           # hall front -> lane stair
    T.route("stair_lane_foot", ls[0], ls[3], ls[2], ls[3] + foot)
    T.route("stair_lane_head", ls[0], YN - 2.0, ls[2], YN, TH)
    T.route("lane_link", ls[2], YN - 2.0, TX1 - 2.0, YN - 0.75, TH)                                  # lane stair -> promenade
    T.route("terrace_cross", st_x - 1.3, YS + 2.0, TX1 - 2.9, YS + 4.2, TH)                         # stair head -> shop fronts
    T.route("promenade", TX1 - 2.9, TS + 4.8, TX1 - 1.9, YN - 0.75, TH)                              # along the shop fronts
    for b in rows:                                                    # in front of every door the buildings declare
        T.doors(b, T.items[b]["tag"])

    # ---- the fountain: a low round basin inside an iron railing on a stone kerb, lantern posts at its corners -------
    T.put("paving_patch", fx_, fy_, 0.0, G.uniform(0, 360), variant="*", tag="centre", collide="none", diameter=FD + 5.0, sett=0.16, seed=G.randrange(99))
    T.put("fountain_tiered", fx_, fy_, 0, G.uniform(0, 360), variant=None, tag="centre", collide="circle", diameter=FD, plan="round", tiers=2,
          steps=1, rim_lanterns=0, pool_lights=2, seed=G.randrange(99), r=FD / 2 + 0.2)
    RV = AP / math.cos(math.pi / 8)
    ring_a0 = 22.5                                                    # its sides face the walks and the diagonals
    verts = [(fx_ + RV * math.cos(math.radians(ring_a0 + 45 * k)), fy_ + RV * math.sin(math.radians(ring_a0 + 45 * k))) for k in range(8)]
    for k in range(8):                                                # counter-clockwise: the railing's front looks outwards
        T.run("iron_railing", verts[k], verts[(k + 1) % 8], 0, tag="ft_rail", height=0.95, seeds=2, max_seg=6.0, gap=0.62, collide="none")
        vx, vy = verts[k]
        T.put("ground_lantern", vx, vy, 0, ring_a0 + 45 * k + 90, variant=["stone_octagon"], tag="ft_post", collide="none", fixed=True, r=0.3,
              height=1.1, intensity=LAMP_CD(P["plaza_w"]) / (3 if k % 2 == 0 else 5), seed=2)

    def basket(li, k):
        """a flower basket on the lamp post's bracket (no candle: the lamp is the light)"""
        if li is not None and T.sockets(li, "planter"):
            T.put("hanging_basket", T.items[li]["pos"][0], T.items[li]["pos"][1], T.items[li]["pos"][2], T.items[li]["rot"], variant="*",
                  tag="basket", collide="none", attach={"to": li, "role": "planter"}, candle=False, seed=k)

    # ---- the kiosk on the terrace corner over the square (the picture's stone kiosk under a copper roof) ---------
    # (a landmark of the plan: it stands on the corner by rule and the review cameras give way to it)
    # (it steps in from the corner along the diagonal until its body as built stands clear of the two parapets)
    kiosk_i, kiosk_seed = None, G.randrange(99)
    for k in range(8):
        n_left = len(T.clashed)
        kiosk_i = T.put("newspaper_kiosk", XW + 2.7 + 0.2 * k, YS + 2.7 + 0.2 * k, TH, facing_rot(0.25, -1), variant=["bell_grand", "tent_copper"],
                        tag="kiosk", r=2.2, foot=1.5, fixed=True, roof_metal="copper", corner_lamps=True, intensity=70.0, seed=kiosk_seed)
        if kiosk_i is not None:
            break
        del T.clashed[n_left:]
    if kiosk_i is None:
        T.dropped.append("newspaper_kiosk (kiosk): does not fit the terrace corner")

    # ---- lamps: ONE design per zone, set out as mirrored pairs / even rows on the plan's axes (the calling rule
    # chooses the spots: they are fixed, the review cameras give way to them) ----------------------------------
    # the fountain: four twin-armed posts on its diagonals (the picture's banner lamp)
    ring_r = AP + 3.0
    for i in range(4):
        a = math.radians(45.0 + 90.0 * i)
        li = T.put("lamp_post_gas", fx_ + ring_r * math.cos(a), fy_ + ring_r * math.sin(a), 0, facing_rot(-math.cos(a), -math.sin(a)), variant="twin",
                   tag="ring_lamp", r=0.45, fixed=True, intensity=85.0, seed=70)
        basket(li, i)
    # the square's south side: a pair mirrored about the fountain's axis (the town clock stands between them)
    for sx in (-1, 1):
        li = T.put("lamp_post_gas", fx_ + sx * 5.5, Y0 + 2.0, 0, 0, variant="plinth", tag="edge_lamp", r=0.4, fixed=True,
                   intensity=LAMP_CD(P["plaza_w"]), seed=100)
        basket(li, 10 + sx)
    # the neck: a pair mirrored about its axis half way up (the hall's own lanterns light its door); the court: one post
    # in the middle of its east half
    for sx in (-1, 1):
        li = T.put("lamp_post_gas", nax + sx * (nhw - 2.8), (YS + YN) / 2, 0, 90 if sx < 0 else -90, variant=None, tag="neck_lamp", r=0.4, fixed=True,
                   intensity=LAMP_CD(P["neck_w"]), seed=110)
        basket(li, 20 + sx)
    T.put("lamp_post_gas", (ls[2] + TX1) / 2, (YN + YH) / 2, 0, 0, variant=None, tag="court_lamp", r=0.4, fixed=True, intensity=LAMP_CD(P["court_d"]), seed=110)
    # the court before the hall: posts of the same design in one row along its axis, at the pitch of the neck's pair and
    # in line with it (the row carries the neck's two lines of lamps on to the hall and along its east wing to the lane stair)
    hp = 2 * (nhw - 2.8)
    for k in range(-1, int((ls[0] - 1.5 - HALL["tower"]) / hp + 0.5)):
        T.put("lamp_post_gas", HALL["tower"] + (k + 0.5) * hp, (YN + YH) / 2, 0, 0, variant=None, tag="court_lamp", r=0.4, fixed=True, intensity=90.0,
              seed=110)                                              # turned up: they wash the hall's pale front (the picture's hall glows)

    def parapet_row(a, b, n, z, key, sd, seat="bench_park", seat_v=("plain_weathered",), tub="planter_box", tub_v=("flower_tub",)):
        """The furnishing of a terrace edge by ONE rule: n lamp posts of the terrace's design at an even pitch along
        a->b (the parapet is on the left), a bench centred between each two (n = 1: a bench either side of the post), a
        flower tub closing either end. Everything stands in one line, squared to the wall, looking away from it."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        nx, ny = uy, -ux
        rot = facing_rot(nx, ny)
        at = lambda t, o: (a[0] + dx * t + nx * o, a[1] + dy * t + ny * o)
        for i in range(n):
            x, y = at((i + 0.5) / n, 0.7)
            li = T.put("lamp_post_gas", x, y, z, rot, variant="short_plain", tag="ter_lamp", r=0.4, fixed=True, intensity=round(1.6 * LAMP_CD(P["terrace_w"]), 1), seed=sd)
            if i % 2 == 0:
                basket(li, sd + i)
        for t in ([i / n for i in range(1, n)] if n > 1 else [0.5 - 2.1 / L, 0.5 + 2.1 / L]):
            x, y = at(t, 0.7)
            if T.free(x, y, 0.55, z):
                T.put(seat, x, y, z, rot, variant=list(seat_v), tag=key, r=0.55, fixed=True, quiet=True, seed=sd)
        for t in (0.35 / L, 1 - 0.35 / L):
            x, y = at(t, 0.7)
            if T.free(x, y, 0.45, z):
                T.put(tub, x, y, z, rot, variant=list(tub_v), tag=key, r=0.45, fixed=True, quiet=True, matched=True, size=0.7, seed=sd)

    # the terrace: the south arm's parapet (the picture's benches behind the wall), the two parapets of the north arm
    parapet_row((XM + 0.48, TS + 4.6), (XM + 0.48, YS - 0.4), 3, TH, "ter_a", 120)                                  # travel north: looks east
    parapet_row((XW + 0.48, YN - 6.8), (XW + 0.48, YN - 0.8), 2, TH, "ter_c", 124)
    parapet_row((XW + 2.4, YN - 0.48), (ls[0] - 0.8, YN - 0.48), 2, TH, "ter_d", 126)                               # travel east: looks south
    # the street: posts on alternate sides at one pitch
    s_n = max(2, int(round((Y0 - SY0 - 3.0) / 5.6)))
    for k in range(s_n):
        side = 1 if k % 2 else -1
        ly = SY0 + 6.0 + (Y0 - 1.5 - SY0 - 6.0) * k / max(1, s_n - 1)
        ly = next((ly + q for q in (0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4) if T.free(scx + side * (SW / 2 - 1.0), ly + q, 0.5, 0)), ly)   # clear of a door
        li = T.put("lamp_post_gas", scx + side * (SW / 2 - 1.0), ly, 0, 90 if side < 0 else -90,
                   variant=None, tag="street_lamp", r=0.4, fixed=True, intensity=LAMP_CD(SW), seed=140)
        if k % 2 == 0:
            basket(li, 40 + k)

    # ---- trees: a plane on the west side of the neck between its lamp and the hall's (the picture's tree before the
    # hall; the hotel's blade stays in view from the terrace); a small tree in the pocket at the foot of the stair, one
    # on the terrace behind the kiosk -----
    for i, t in ((1, 0.8),):
        tx, ty = nax - (nhw - 2.8), YS + (YN - YS) * t
        at = None
        for k in range(5):
            at = T.tree_spot(tx, ty, T.rng("tree", i, k), reach=0.6 * k, wall=2.0)
            if at is not None:
                break
        if at is not None:
            T.tree("tree_plane", at[0], at[1], 0, G.uniform(0, 360), T.rng("tree_grate", i), variant=["young_slender"], collide="circle:0.35",
                   seed=90, r=1.2, foot=0.45)
    small = [((XS - 2.6, ms[1] - 0.72), 0.0), ((XM + 2.2, YN - 4.2), TH)]
    for i, ((tx, ty), tz) in enumerate(small):
        at = None
        for k in range(5):
            at = T.tree_spot(tx, ty, T.rng("small_tree", i, k), z=tz, r=1.0, canopy=2.4, reach=0.5 * k, wall=1.2)
            if at is not None:
                break
        if at is not None:
            ti = T.put("tree_ornamental", at[0], at[1], tz, G.uniform(0, 360), variant=["default"], tag="small_tree", collide="circle:0.3",
                       r=1.0, foot=0.45, fixed=True, seed=5, blossom=0.12, height=4.6, crown_width=3.0)   # the picture's small round green trees
            if ti is not None:
                T.yields.add(ti)
                T.crowns.append((at[0], at[1], tz, 2.4))
                if asset_path("tree_grate"):                          # a slender trunk: the grate and its guard go round it as built
                    T.put("tree_grate", at[0], at[1], tz, 0.0, variant=["round_scroll", "small_plain"], paint="#24302a", tag="small_tree", collide="none", of=ti,
                          quiet=True, seed=3)

    # ---- dressing rules: nothing stands singly -----------------------------------------------------------
    WIDE = {"bicycle": 1.9, "hand_cart": 2.3, "bench_park": 2.1, "luggage_stack": 1.3, "crate_stack": 1.2, "sack_pile": 1.3, "planter_box": 1.1,
            "milk_churns": 1.1, "flower_pots": 1.2}

    def front_pt(b, t):
        """point on the front line of building b at fraction t of its width, and the direction along the front that
        has the wall on its left (what Town.line / cluster expect)"""
        it = T.items[b]
        ox, oy = rot2((t - 0.5) * it["params"].get("width", 6.0), 0.0, it["rot"])
        return (it["pos"][0] + ox, it["pos"][1] + oy), rot2(1, 0, it["rot"])

    def cluster(c, along, names, key, z=0.0, n=3, off=0.55):
        """Loose dressing as ONE group against a wall: up to n different pieces of `names` (seeded order) side by side
        about c, `off` in front of the wall (the wall is on the left of `along`), each squared to it (a few degrees
        off true). c is the spot or a list of spots in order of preference: the group stands at the first where every
        piece fits (clear of doors, steps, ways and the edges of the review frames), else where most of it does. A
        piece that does not fit there is left out - it is never moved off on its own."""
        ux, uy = along
        nx, ny = uy, -ux
        R = T.rng("cluster", key)
        have = [m for m in names if asset_path(m)]
        R.shuffle(have)
        have = have[:max(1, int(round(n * min(1.0, P["density"]))))]
        total = sum(WIDE.get(m, 0.9) for m in have)
        draw = [(R.randrange(999), R.uniform(-5, 5)) for _ in have]

        def spots(c_):
            out, d = [], -total / 2
            for m in have:
                w = WIDE.get(m, 0.9)
                o = 0.04 if m == "ladder" else off
                x, y = c_[0] + ux * (d + w / 2) + nx * o, c_[1] + uy * (d + w / 2) + ny * o
                d += w
                rr = min(0.45, w / 2)
                out.append((m, rr, next(((x + nx * st, y + ny * st) for st in STEP_OFF
                                         if T.free(x + nx * st, y + ny * st, rr, z) and not T.cut(x + nx * st, y + ny * st, z, rr, 1.0)), None)))
            return out
        best = max((spots(c_) for c_ in (c if isinstance(c, list) else [c])), key=lambda s: sum(1 for q in s if q[2]))
        done = 0
        for (m, rr, at), (sd, jr) in zip(best, draw):
            if at and T.put(m, at[0], at[1], z, facing_rot(nx, ny) + jr, variant="*", tag=key, r=rr, fixed=True, quiet=True, seed=sd) is not None:
                done += 1
        return done

    def flank(b, asset, half, d=0.6, key="pair", r=0.5, **kw):
        """A mirrored pair: the same piece `half` m either side of building b's door (its middle when it declares no
        door), d m out of the wall, squared to it. Both stand or neither does."""
        if b is None or not asset_path(asset):
            return 0
        it = T.items[b]
        ux, uy = rot2(1, 0, it["rot"])
        lx = 0.0
        doors = [q for q in T.sockets(b, "door") if abs(q["z"] - it["pos"][2]) < 1.6]
        if doors:
            lx = min(((q["x"] - it["pos"][0]) * ux + (q["y"] - it["pos"][1]) * uy for q in doors), key=abs)
        z = it["pos"][2]
        wmax = it["params"].get("width", 2 * half + 4.0) / 2 - r - 0.3
        # the spacing the rule asks for first, then wider / closer; then the pair's centre slid along the front (a
        # stoop, a flight of steps or a walking line where one of the two would stand)
        for sh in (0.0, -0.6, 0.6, -1.2, 1.2, -1.8, 1.8, -2.4, 2.4):
            for dh in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, 1.6, 2.0):
                h = half + dh
                if h < r + 0.45 or abs(lx + sh) + h > wmax:
                    continue
                pts = [(it["pos"][0] + ox, it["pos"][1] + oy) for ox, oy in (rot2(lx + sh + s * h, -d, it["rot"]) for s in (-1, 1))]
                if any(not T.free(x, y, r, z) or T.cut(x, y, z, r, 1.0) for x, y in pts):
                    continue
                got = 0
                for x, y in pts:
                    if T.put(asset, x, y, z, it["rot"], tag=key, fixed=True, quiet=True, matched=True, r=r, **kw) is None:
                        break
                    got += 1
                if got == 2:
                    return 1
                if got == 1:
                    T.unput()
        return 0

    # ---- feature pieces: each on an axis of the plan or anchored to its building ------------------------------
    # the town clock on the fountain's axis before the south fronts, between the two lamps; the finger post at the
    # mouth of the street
    T.put("street_clock", fx_, Y0 + 2.4, 0, facing_rot(0, 1), variant=["station", "jeweller"], tag="clock", r=0.5, fixed=True, seed=3)
    T.put("signpost_finger", SX0 - 1.0, Y0 + 1.1, 0, 20.0, variant="*", tag="signpost", r=0.6, fixed=True, seed=3)
    # a figure against the court's terrace wall, in the middle of its west run, between two benches and two urns
    stx = (XW + ls[0]) / 2
    T.put("statue_figure", stx, YN + 1.6, 0, facing_rot(0, 1), variant=["scholar_low", "scholar_hooded", "star_pilgrim"], tag="statue", r=1.2, fixed=True,
          seed=G.randrange(99))
    for sx in (-1, 1):
        T.put("bench_stone", stx + sx * 3.0, YN + 0.95, 0, facing_rot(0, 1), variant=["backless_paw"], tag="statue", r=0.9, fixed=True, quiet=True, seed=2)
        T.put("planter_urn", stx + sx * 4.9, YN + 0.95, 0, facing_rot(0, 1), variant=["fern_vase"], tag="statue", r=0.5, fixed=True, quiet=True, seed=2)
    T.place("post_box", [(X0 + 0.5, hotel_y0 + hotel_w - 1.25, X0 + 0.8, hotel_y0 + hotel_w - 0.45)], 0.4, key="post_box", limit=0.03, rot=90)
    if HALL["wall"]:                                                  # (a hall with a plain stretch of front wall at eye height)
        T.place("notice_board", [(HALL["wall"][0], YH - 0.05, HALL["wall"][1], YH - 0.05)], 0.45, key="notice", limit=0.05, rot=0)
    T.place("litter_bin", [(SX0 - 2.6, Y0 + 0.5, SX0 - 1.9, Y0 + 0.8)], 0.4, key="bin", rot=180)
    T.place("menu_board", [(XM + 2.0, TS + 4.3, XM + 2.8, TS + 5.0)], 0.5, z=TH, key="menu", limit=0.05, rot=200)
    shop = {T.items[b]["asset"]: b for b in shop_row if b is not None}
    if "shop_luggage" in shop:
        T.place("luggage_stack", [T.front(shop["shop_luggage"], 0.7, 1.25, inset=-0.8)], 0.7, z=TH, key="luggage", rot=-90)   # at a corner of the front, off its step
    if "shop_tailor" in shop:
        T.place("shop_mannequin", [T.front(shop["shop_tailor"], 0.85, 1.2)], 0.4, z=TH, key="tailor", rot=-90)
    # a flower seller on the cobbles south-east of the fountain, squared to the square, her stock in a row before the cart
    fz = [(fx_ + 3.6, Y0 + 5.0, min(fx_ + 7.5, SX0 - 1.5), fy_ - AP - 4.0)]
    fc = T.place("hand_cart", fz, 1.3, key="flowers", limit=0.14, variant="flower_seller", lantern=True, intensity=LAMP_CD(P["plaza_w"]) / 5,
                 rot=lambda x, y: facing_rot(-1, 0))
    if fc is not None:
        ox, oy = T.items[fc]["pos"][0], T.items[fc]["pos"][1]
        for k, nm in enumerate(["flower_pots", "bucket_set", "flower_pots"]):
            T.put(nm, ox - 1.75, oy + (k - 1) * 1.0, 0, facing_rot(-1, 0), variant="*", tag="flowers", r=0.45, fixed=True, quiet=True, seed=k)

    def strew(rect, n, key, z=0.0, r=0.7):
        """Paving that is not swept bare: n drifts of fallen leaves and grit caught in the joints of the open paving in
        rect (seeded). They lie flat and are walked over, so the walks take them too; nothing stands on them."""
        R = T.rng("strew", key, rect)
        done, lie = 0, []
        for _ in range(n * 25):
            if done >= int(round(n * min(1.0, P["density"]))):
                break
            x, y = R.uniform(rect[0] + r, rect[2] - r), R.uniform(rect[1] + r, rect[3] - r)
            hit = T.blocked(x, y, r, z)
            if (not T.clear(x, y, r, z) or (hit and (hit.startswith("stairs") or hit.startswith("foot of")))
                    or any(math.hypot(x - lx, y - ly) < 3 * r for lx, ly in lie)):
                continue
            if T.put("pavement_weeds", x, y, z, R.uniform(0, 360), variant=["parched_tufts"], tag="leaves", collide="none",
                     habit="joints", litter=round(R.uniform(0.8, 1.0), 2), spread=round(R.uniform(1.1, 1.5), 2), seed=R.randrange(3)) is not None:
                lie.append((x, y))
                done += 1
        return done

    # ---- the square: its middle stays open cobbles ---------------------------------------------------------
    # benches with their backs to the fountain's railing on its four diagonals, each looking out at its lamp
    for i, a_ in enumerate((45.0, 135.0, 225.0, 315.0)):
        a = math.radians(a_)
        T.put("bench_park", fx_ + (AP + 0.75) * math.cos(a), fy_ + (AP + 0.75) * math.sin(a), 0, facing_rot(math.cos(a), math.sin(a)),
              variant=["plain_weathered"], tag="ring_bench", r=1.0, foot=0.9, fixed=True, seed=80)     # the town's one bench design
    # the foot of the terrace walls: stone benches and urns in even bays under the kiosk corner (the picture's bench),
    # along the neck and down the stair bay
    BENCH, URN = ("bench_stone", {"variant": ["backless_paw"], "seed": 2}), ("planter_urn", {"variant": ["flower_tazza"], "seed": 2})
    T.bays((XW + 0.9, YS - 0.2), (XS - 0.5, YS - 0.2), 2.4, [BENCH, URN], off=0.62, key="foot_b", r=0.8)                # travel east: props face south
    T.bays((XW - 0.2, YN - 1.2), (XW - 0.2, YS + 0.6), 3.2, [URN, BENCH], off=0.62, key="foot_c", r=0.8)                # travel south: face west
    # the stair bay's wall (the picture's benches at the wall foot): a stone bench in the middle of each bay between two
    # buttresses, an urn in the short bay left before the south row
    buts = sorted(a_buts, reverse=True)
    for ya, yb in zip(buts, buts[1:]):
        T.bays((XM - 0.2, ya), (XM - 0.2, yb), ya - yb, [BENCH], off=0.62, key="foot_a", r=0.6)
    if buts and buts[-1] - (Y0 + 0.6) > 2.4:
        T.bays((XM - 0.2, buts[-1]), (XM - 0.2, Y0 + 0.6), buts[-1] - (Y0 + 0.6), [URN], off=0.62, key="foot_a", r=0.6)
    T.scatter((fx_ + AP + 1.0, fy_ - 4.0, XS - 3.0, fy_ + 4.0), 2,
              lambda x, y, R, i: not T.cut(x, y, 0, 0.8, 0.3) and T.put("pigeons", x, y, 0, R.uniform(0, 360), variant="*", tag="centre", collide="none", seed=R.randrange(99)) is not None,
              r=0.8, key="pigeons")
    # the fronts: by every house ONE group of the loose things it deals in, at a party-wall end of its front, and
    # mirrored pairs flanking the doors that matter
    GOODS = {"house_timber": ["barrel", "crate_stack", "milk_churns", "sack_pile"], "house_stone": ["flower_pots", "planter_box", "bicycle", "bucket_set"],
             "townhouse_narrow": ["flower_pots", "bucket_set", "cat"], "shop_corner": ["crate_stack", "sack_pile", "barrel", "milk_churns"],
             "tower_house": ["barrel", "ladder", "crate_stack"], "hotel": ["luggage_stack", "bicycle", "planter_box"],
             "shop_luggage": ["planter_box", "flower_pots"], "shop_tailor": ["flower_pots", "planter_box"]}
    T.put("cellar_hatch", X0 + 0.05, hotel_y0 + hotel_w - 2.8, 0, 90, variant="*", tag="hotel", collide="none", r=0.5, seed=4)
    flank(hotel_i, "planter_urn", 1.7, key="hotel_pair", variant="grand_cone", seed=1)
    if HALL["urns"] == "sockets":                                     # the hall declares where planters stand: the two nearest its door
        qs = sorted(T.sockets(hall_i, "planter"), key=lambda q: math.hypot(q["x"] - HALL["door"][0], q["y"] - HALL["door"][1]))[:2]
        for q in (qs if len(qs) == 2 else []):
            T.fit(hall_i, q, "planter_urn", tag="hall_pair", attach=False, of=hall_i, fixed=True, quiet=True, matched=True, r=0.22,
                  variant="fern_vase", scale=0.85, seed=2)
    else:                                                             # against the wall either side of the porch steps
        flank(hall_i, "planter_urn", HALL["urns"][0], d=HALL["urns"][1], key="hall_pair", r=0.22, variant="fern_vase", scale=0.85, seed=2)
    flank(inn_i, "bench_park", 2.4, key="inn_pair", r=0.9, variant="plain_weathered", seed=1)
    if "house_timber" in shop:
        flank(shop["house_timber"], "bench_park", 1.8, key="shop_pair", r=0.9, variant="plain_weathered", seed=120)
    if "house_stone" in shop:
        flank(shop["house_stone"], "planter_box", 1.5, key="shop_pair", variant="flower_tub", size=0.7, seed=1)    # low: under its window boxes
    for n, b in enumerate(rows):
        it = T.items[b]
        if it["asset"] not in GOODS:
            continue
        t0 = 0.16 if n % 2 or b == hotel_i else 0.84
        ts = (t0, 1.0 - t0, 0.3 if t0 < 0.5 else 0.7, 0.7 if t0 < 0.5 else 0.3, 0.5)
        cluster([front_pt(b, t)[0] for t in ts], front_pt(b, t0)[1], GOODS[it["asset"]], f"goods{n}", z=it["pos"][2],
                n=3 if it["params"].get("width", 6.0) >= 6.0 else 2)
    # drain covers down the fall of the square, the neck and the court, with the leaves caught on them
    RD_ = T.rng("drains")
    for i in range(3):
        T.put("drain_cover", fx_ + (i - 1) * 6.5, Y0 + 3.4, 0.0, 0, variant="*", tag="drain", collide="none", seed=i, leaves=RD_.randrange(0, 7))
    for i, (dx_, dy_) in enumerate([(st_x, ms[1] - 4.0), (nax, (YS + YN) / 2), (nax, YN + 1.0), (ls_x - 3.0, cw_y)]):
        T.put("drain_cover", dx_, dy_, 0.0, 0, variant="*", tag="drain", collide="none", seed=20 + i, leaves=RD_.randrange(4, 10))
    for k, rc in enumerate([(X0 + 0.2, Y0 + 0.2, XS - 0.3, YS - M - 0.1), (X0 + 0.2, YS, XW - M - 0.1, YN), (X0 + 0.2, YN + M + 0.1, TX1 - 0.2, YH - 0.2)]):
        T.scatter(rc, (1, 0, 0)[k],
                  lambda x, y, R, i, rc=rc: T.put("pavement_weeds", x, y, 0, R.uniform(0, 360), variant="*", tag="weeds", collide="none", seed=R.randrange(3)) is not None
                  if (min(x - rc[0], rc[2] - x, y - rc[1], rc[3] - y) < 0.9) else False, r=0.3, key=f"weeds{k}", tries=200)
    strew((X0 + 1.0, Y0 + 1.0, XS - 0.5, YS - 1.0), 1, "plaza")
    strew((X0 + 1.0, YS, XW - 1.0, YN), 0, "neck")
    strew((X0 + 1.0, YN + 1.0, TX1 - 1.0, YH - 1.0), 0, "court")

    # ---- cafe tables ---------------------------------------------------------------------------------
    SHADE_D, SET_R, SET_P = 2.4, 1.0, 2.25   # parasol canopy; a table with its chairs pushed back; the pitch of the rows of tables
    shades = []

    def cafe_set(x, y, R, i, z=0.0, tag="cafe", shade=0):
        """a table with 2-3 chairs and a candle lantern on it - the table's one light, so every table of the town glows
        (TABLE_CD); every `shade`-th set stands under a parasol with one lantern at its pole, where the canopy neither
        hides a view nor is cut by a frame"""
        if T.cut(x, y, z, 1.0, 1.0):                  # the set as one piece: wholly in a near frame or wholly out of it
            return False
        ti = T.put("cafe_table", x, y, z, R.uniform(0, 360), variant="*", tag=tag, collide="circle:0.4", fixed=True, lamp="none", seed=R.randrange(999))
        if ti is None:
            return False
        T.yields.add(ti)
        n = 2 + 0 * R.choice([2, 2, 3])                  # two chairs to a table, facing each other
        a0 = R.uniform(0, 360)
        for k in range(n):
            a1 = a0 + k * 360 / n + R.uniform(-15, 15)
            d, turn, sd_ = 0.72 + R.uniform(-0.05, 0.1), R.uniform(-25, 25), R.randrange(999)
            for da in (0.0, 40.0, -40.0, 80.0, -80.0):        # (turned round the table when the next table's chair is in the way)
                a = math.radians(a1 + da)
                if T.put("cafe_chair", x + d * math.cos(a), y + d * math.sin(a), z, facing_rot(-math.cos(a), -math.sin(a)) + turn,
                         variant="*", tag=tag, collide="circle:0.25", fixed=True, quiet=True, of=ti, seed=sd_) is not None:
                    break
        T.put("table_lantern", x, y, z, R.uniform(0, 360), variant="*", tag=tag, collide="none", on=ti, intensity=TABLE_CD, seed=R.randrange(999))
        sd = R.randrange(999)
        if (shade and i % shade == 0 and T.sight(x, y, z, SHADE_D / 2, *BULK["parasol"][1:])[0] <= SIGHT_PUT and not T.cut(x, y, z, SHADE_D / 2, 2.9)
                and all(abs(z - sz) > 1.0 or math.hypot(x - sx, y - sy) > SHADE_D + 0.3 for sx, sy, sz in shades)):    # canopies never overlap
            if T.put("parasol", x + 0.05, y, z, R.uniform(0, 360), variant=["bistro_stripe"] if tag == "kiosk_cafe" else ["market_green"], tag=tag, collide="none",
                     fixed=True, of=ti, foot=0.3, diameter=SHADE_D, lanterns=1, rim_lanterns=0, intensity=2 * TABLE_CD, seed=sd) is not None:
                shades.append((x, y, z))
        return True
    # round the kiosk (the picture's tables on the terrace corner) and before the cafe's gallery
    T.rows((XW + 2.1, YS + 5.6, min(XW + 8.9, XM - 1.0), min(YS + 10.1, YN - 3.6)), SET_P,
           lambda x, y, R, i: cafe_set(x, y, R, i, z=TH, tag="kiosk_cafe", shade=2), r=SET_R, z=TH, key="kiosk_cafe")
    T.rows((XM + 1.8, TS + 6.3, TX1 - 3.0, TS + 10.8), SET_P - 0.15, lambda x, y, R, i: cafe_set(x, y, R, i, z=TH, tag="ter_cafe", shade=3),
           r=SET_R, z=TH, key="ter_cafe")

    # ---- terrace floor (its parapets are furnished with the lamps: parapet_row) ---------------------------
    RL = T.rng("ter_floor")
    pr0, pr1 = TS + 5.0, YN - 1.0
    n_g = int(round((pr1 - pr0) / 8.0))
    for i in range(n_g):
        T.put("drain_cover", TX1 - 2.6, pr0 + (i + 0.5) * (pr1 - pr0) / n_g, TH, 0, variant=["gully_grate"], tag="ter_drain",
              collide="none", leaves=RL.randrange(4, 10), seed=30 + i)
    strew((XM + 1.2, TS + 4.2, TX1 - 0.8, YS), 1, "terrace_s", z=TH)
    strew((XW + 1.2, YS + 1.0, TX1 - 0.8, YN - 1.0), 0, "terrace_n", z=TH)

    # ---- what the buildings offer: signs, awnings, banners, window boxes on their sockets ----------------
    RS = T.rng("fitout")
    SIGNS = ["cobbler_oval", "locksmith_cutout", "bookshop_rect", "wine_shield"]
    TRADES = [("boot", "BOOTS"), ("key", "KEYS"), ("book", "BOOKS"), ("bottle", "VINO"), ("cup", "CAFE"), ("star", "INN"), ("suitcase", "BAGS")]
    RS.shuffle(SIGNS)
    RS.shuffle(TRADES)
    n_sign, hung = 0, []
    for n, b in enumerate(sorted(rows, key=lambda b_: T.items[b_]["asset"] not in ("shop_tailor", "shop_luggage"))):
        asset = T.items[b]["asset"]
        on_street = b in street_w or b in street_e
        signs = [q for q in T.sockets(b, "sign") if q["name"] in ("sign", "sign_front", "hanging_sign")]
        if signs and any(math.hypot(signs[0]["x"] - hx_, signs[0]["y"] - hy_) < 4.0 for hx_, hy_ in hung):
            signs = []
        if signs and asset not in ("clock_tower_hall", "civic_hall_grand") and (not on_street or n % 2 == 0):
            hung.append((signs[0]["x"], signs[0]["y"]))
            if asset == "shop_tailor":
                T.fit(b, signs[0], "hanging_sign", tag="sign", variant="tailor_scissors", lamp="none", board_w=0.5, board_h=0.6, seed=RS.randrange(99))
            else:
                em, tx_ = ("suitcase", "BAGS") if asset == "shop_luggage" else TRADES[n_sign % len(TRADES)]
                T.fit(b, signs[0], "hanging_sign", tag="sign", variant=SIGNS[n_sign % len(SIGNS)], lamp="none", emblem=em, text=tx_, board_w=0.5,
                      board_h=0.6, seed=RS.randrange(99))
                n_sign += 1
        for q in T.sockets(b, "awning")[:1]:
            T.fit(b, q, "awning", tag="awning", width=q["props"].get("width", 3.0), variant=["regency_green", "tabac_ochre"], seed=RS.randrange(99))
        boxes = T.sockets(b, "window_box") + [q for q in T.sockets(b, "planter") if "planter" in q["name"] and q["z"] - T.items[b]["pos"][2] < 5.2]
        if b == hall_i and HALL["urns"] == "sockets":                 # (its planter sockets stand on the ground: urns, above)
            boxes = []
        for q in boxes[:2]:
            T.fit(b, q, "window_box", tag="window_box", width=min(1.8, max(0.5, q["props"].get("width", 0.9))), seed=RS.randrange(99))
    # festival banners: every lamp post whose bracket is high enough to walk under (BANNER_H) flies a small crown
    # banner, the colours taking turns from post to post; the hall and the cafe hang long ones
    k = 0
    for li, it in enumerate(list(T.items)):
        if it["asset"] in ("lamp_post_gas", "lamp_post_globe"):
            for q in T.sockets(li, "banner")[:1]:
                if q["z"] - it["pos"][2] >= BANNER_H and it["tag"] in ("ring_lamp", "edge_lamp", "street_lamp"):
                    k += T.fit(li, q, "banner_crown", tag="lamp_banner", width=0.5, drop=1.0,
                               variant=[BANNERS[("ring_lamp", "edge_lamp", "street_lamp").index(it["tag"]) % len(BANNERS)]], seed=2) is not None
    for b, (bw, bd_) in ((hall_i, (1.3, 3.4)), (cafe_i, (0.9, 2.0))):
        for k, q in enumerate(q for q in (T.sockets(b, "banner") if b is not None else []) if not q["name"].startswith("bunting")):
            T.fit(b, q, "banner_crown", tag="banner", width=bw, drop=bd_, variant=[BANNERS[0]], seed=3)

    # ---- the arrival street ----------------------------------------------------------------------------
    # (its houses get their groups of goods with the other fronts; a barrow stands by the corner shop's flank at the mouth)
    cluster((SX0, Y0 - BD / 2), (0, 1), ["hand_cart", "sack_pile", "barrel"], "street_goods", n=3, off=0.8)   # the corner shop's flank
    for i in range(2):
        T.put("drain_cover", scx, SY0 + 3.5 + i * (Y0 - SY0 - 5.0), 0, 0, variant="*", tag="drain", collide="none", seed=10 + i)
    for i, sx in enumerate((-1, 1)):
        kx = scx + sx * (SW / 2 - 1.7)
        T.run("kerb_gutter", (kx, SY0 + 0.8) if sx < 0 else (kx, Y0 - GD - 0.3), (kx, Y0 - GD - 0.3) if sx < 0 else (kx, SY0 + 0.8), 0, tag="kerb",
              collide="none", seeds=2, max_seg=6.0)
    T.scatter((SX0, SY0, SX1, Y0), 0,
              lambda x, y, R, i: T.put("pavement_weeds", x, y, 0, R.uniform(0, 360), variant="*", tag="weeds", collide="none", seed=R.randrange(3)) is not None
              if (min(x - SX0, SX1 - x) < 0.7) else False, r=0.3, key="weeds_st", tries=200)
    # garlands across the street: flags, and one string of lights half way up
    for i, t in enumerate((0.25, 0.5, 0.75)):
        yy = SY0 + (Y0 - GD - SY0) * t
        T.put("bunting" if i != 1 else "festoon_lights", SX0 + 0.25, yy, 5.6, 0, variant="*", tag="street_bunting", collide="none",
              span=SW - 0.5, seed=5 + i, **({"intensity": 7.0} if i == 1 else {}))

    # ---- trees outside the walls -----------------------------------------------------------------------
    RO = T.rng("outside")
    outside = [(HALL["x1"] + 4.0, YH + BD + 4.5)]
    for i, (tx, ty) in enumerate(outside):
        T.put("tree_plane", tx + RO.uniform(-0.8, 0.8), ty + RO.uniform(-0.8, 0.8), OUT - 0.1, RO.uniform(0, 360), variant=["young_slender"],
              tag="outside", collide="none", seed=60 + i)       # not fixed: the trunk moves clear of the walls it stands behind

    # ---- review cameras ---------------------------------------------------------------------------
    T.spawn = {"pos": [scx, SY0 + 5.0, eye], "yaw": 0.0}
    # the camera of the picture (set out with the scenery, above)
    T.shot("01_reference", ref_pos, ref_at + (round(ref_z, 2),), ref_fov, "the reference view: fountain, stair, terrace shops, hall, HOTEL blade",
           see=[(fx_, fy_, 0.6, 2.6)])
    T.shot("02_street", (scx + 0.6, SY0 + 2.2, eye), (st_x, YS, 3.2), 60, "arrival: up the street to the stair, the cafe on the wall, the shops above")
    T.shot("03_enter", (SX0 + 1.0, Y0 + 2.6, eye), (fx_ + 1.0, YN, 9.5), 54, "entering the square: fountain, kiosk corner, hall tower beyond",
           see=[(fx_, fy_, 0.6, 2.6)])
    T.shot("04_fountain", (fx_ + 1.6, fy_ - 9.4, eye), (fx_, fy_ + 9.0, 5.4), 58, "across the fountain to the neck and the hall")
    T.shot("05_stair_shops", (XS - 1.6, ms[1] - 7.4, eye), (TX1, YS + 2.5, TH + 4.2), 58, "up the stair bay: the flight, the wall and its benches, the shop row above")
    # (at the east parapet, in the middle of its second bay from the stair: the pier lanterns of the wall stand out of
    # the frame on either side and the coping under it; the kiosk corner has a lantern pier at every pace)
    bay_a = (YS - (Y0 - GD)) / math.ceil((YS - (Y0 - GD)) / 5.4)
    T.shot("06_overlook", (XM + 0.75, YS - 1.5 * bay_a, TH + eye), (fx_ - 3.0, fy_ + 1.5, 3.0), 54, "over the terrace parapet: the square, the fountain, the hotel")
    # (it stands a pace clear of the gallery's posts, whatever bay rhythm the cafe's variant has)
    T.shot("07_promenade", (TX1 - 4.0, TS + 3.0 + 1.0, TH + eye), (TX1 - 4.0, YN - 2.0, TH + 2.4), 56, "from the cafe gallery up the terrace: tables, parapet lamps, the shop fronts")
    T.shot("08_cafe", (XM + 5.4, YS - 3.2, TH + eye), (TX1 - 4.6, TS, TH + 2.6), 58, "down the south arm of the terrace: the tables before the cafe gallery")
    T.shot("09_kiosk", (XM + 4.0, YS + 7.0, TH + eye), (XW + 1.5, YS + 7.0, TH + 3.0), 56, "the kiosk corner: its tables, the parapet, the hall tower beyond")
    T.shot("10_neck_hall", (nax + 0.9, YS - 2.2, eye), (HALL["tower"], YH, 9.5), 70, "up the neck to the clock tower hall")
    T.shot("11_court", (X0 + 5.5, YN + 2.0, eye), (TX1 - 2.0, cw_y, 6.5), 54, "the hall court: lane stair, tower house, the terrace balustrade")
    T.shot("12_hotel", (X0 + 6.5, Y0 + 2.6, eye), (X0 + 0.4, hotel_y0 + 0.68 * hotel_w, 4.6), 60, "inn and hotel fronts, the HOTEL blade")
    T.shot("13_shops", (XM + 0.6, YS + 2.2, TH + eye), (TX1, YS + 2.2, TH + 3.6), 66, "across the terrace to the LUGGAGE and DRESS SUIT fronts")
    T.shot("14_overview", over[0], over[1], over[2], "bird's-eye of the whole district")

    # every eye-level camera is settled by rule: the free spot nearest its nominal position that stands clear of
    # props and has no lamp post / trunk in the near field of its frame (stays on the same piece of ground)
    occd = {(round(o[0], 2), round(o[1], 2)) for o in T.occ}
    # (a camera gives way to what is fixed only: loose dressing gives way to the camera in the next pass - Town.yields)
    soft = {(round(T.items[i]["pos"][0], 2), round(T.items[i]["pos"][1], 2)) for i in T.yields}
    hard = {(round(it["pos"][0], 2), round(it["pos"][1], 2)) for i, it in enumerate(T.items) if i not in T.yields}
    obst = [o for o in T.occ if (round(o[0], 2), round(o[1], 2)) in hard - soft]
    obst += [(lx, ly, 0.2, lz) for lx, ly, lz in T.lamps if (round(lx, 2), round(ly, 2)) not in occd and (round(lx, 2), round(ly, 2)) not in soft]
    things = [(it["pos"][0], it["pos"][1], it["pos"][2], it["prop"], T.tall(it), T.outline(it)) for i, it in enumerate(T.items)
              if "prop" in it and i not in T.yields and asset_class(it["asset"]) != "vegetation"]
    for sh in T.shots:
        x0, y0, z = sh["pos"]
        zg = z - eye
        gx, gy = sh["target"][0], sh["target"][1]
        sh["_nom"] = [x0, y0, z]

        def aim(px, py):
            L = math.hypot(gx - px, gy - py) or 1.0
            return (gx - px) / L, (gy - py) / L

        def cost(px, py):
            ux, uy = aim(px, py)
            c = math.hypot(px - x0, py - y0)
            for ox, oy, orr, oz in obst:
                if abs(oz - zg) > 1.0:
                    continue
                dx, dy = ox - px, oy - py
                d = math.hypot(dx, dy)
                if d < orr + 0.9:
                    return None
                along, lat = dx * ux + dy * uy, abs(dy * ux - dx * uy)
                if 0.0 < along < 3.0 and lat < orr + 0.2 + 0.95 * along:         # anywhere in the frame, right at the lens
                    c += 8.0 + 24.0 * (1.0 - along / 3.0)
                elif 0.0 < along < 7.0 and lat < orr + 0.5 + 0.08 * along:       # dead centre a little further out
                    c += 12.0
            cam = {"pos": (px, py, z), "target": sh["target"], "fov": sh["fov"]}
            for ox, oy, oz, orr, oh, opts in things:
                if abs(ox - px) < NEAR + 6.0 and abs(oy - py) < NEAR + 6.0 and abs(oz - zg) < 1.0 and cut_by(cam, ox, oy, oz, orr, oh, opts):
                    c += 6.0                                                     # a near piece cut by the edge of the frame
            return c

        sh["_dir0"] = aim(x0, y0)
        if z < 12:
            home = [rg["rect"] for rg in T.regions if abs(rg.get("z", 0) - zg) < 0.5 and rg["rect"][0] <= x0 <= rg["rect"][2] and rg["rect"][1] <= y0 <= rg["rect"][3]]
            best = None
            for ring in range(0, 9):
                for k in range(1 if ring == 0 else 16):
                    a = k * math.tau / 16
                    px, py = x0 + math.cos(a) * ring * 0.45, y0 + math.sin(a) * ring * 0.45
                    if ring and home and not any(r0 + 0.5 <= px <= r2 - 0.5 and r1 + 0.5 <= py <= r3 - 0.5 for r0, r1, r2, r3 in home):
                        continue
                    c = cost(px, py)
                    if c is not None and (best is None or c < best[0] - 1e-6):
                        best = (c, px, py)
            if best:
                x0, y0 = best[1], best[2]
        sh["pos"] = [round(x0, 2), round(y0, 2), z]
        sh["_dir"] = aim(x0, y0)

    return {
        "seed": seed, "params": P, "items": T.items, "regions": T.regions, "terrain": T.terrain, "shots": T.shots,
        "spawn": T.spawn, "missing": T.missing, "blockouts": T.blockouts,
        "frame": {"plaza": [X0, Y0, XM, YS], "terrace_x": [XM, TX1], "fountain": [fx_, fy_]},
    }, T


if __name__ == "__main__":
    import json
    import sys
    t = generate(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(json.dumps({"items": len(t["items"]), "missing": t["missing"], "placement": {k: v for k, v in t["placement"].items() if k not in ("stairs", "routes")}}, indent=1))
