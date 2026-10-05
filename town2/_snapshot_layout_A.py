"""Seeded layout generator for the lantern town square (host python, stdlib only).

    generate(seed=1, **overrides) -> dict   (written to out/town/layout.json by tools/town.py gen)

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
    "plaza_w": 32.0, "plaza_d": 30.0,       # lower square (x, y)
    "axis_x": -1.0,                          # gate - fountain - clock tower axis
    "terrace_h": 2.1, "terrace_w": 10.0,     # raised east terrace
    "fore_h": 0.9, "fore_d": 6.0,            # hall forecourt (north)
    "row_depth": 10.0,                       # west row (cafe / arcade / hotel) depth
    "bld_depth": 8.0,
    "quay_w": 4.0, "canal_w": 6.0, "canal_len": 40.0, "water_z": -1.5, "canal_depth": 2.4,
    "street_w": 9.0, "street_len": 20.0,
    "fountain_d": 5.5,
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
# and jetty that stands proud of it. Cached in out/town/geom.json (keyed by build + file time).
BANDS = ((0.12, 0.6), (0.6, 1.7), (1.7, 3.0), (3.0, 4.2), (4.2, 5.6), (5.6, 99.0))
SKIN = 0.1             # plinths, sills and mouldings this proud of a building's body count as the body
CELL = 0.2
TOL = 0.05             # two things may touch / sink into each other this far before it counts as an intersection
GEO_V = 3              # format of the cached measurements
_GEO = {"data": None, "dirty": False}
GEO_PATH = os.path.join(ROOT, "out", "town", "geom.json")
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
         "fountain_tiered", "bridge_stone", "gate_arch_town", "clock_tower_hall", "monument_star"}
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
            of=None, quiet=False, along=None, **params):
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
        accent = asset in ACCENT and not any(k in extra for k in ("on", "attach"))
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
    cx = P["axis_x"]
    X0, X1 = cx - P["plaza_w"] / 2, cx + P["plaza_w"] / 2            # plaza west / east edges
    Y0, Y1 = -P["plaza_d"] / 2, P["plaza_d"] / 2                     # plaza south / north edges
    TH, FH = P["terrace_h"], P["fore_h"]
    TX1 = X1 + P["terrace_w"]                                        # terrace east edge (shop fronts)
    FY1 = Y1 + P["fore_d"]                                           # forecourt north edge (hall front)
    RD, BD = P["row_depth"], P["bld_depth"]
    QX1 = X0 - RD                                                    # quay east edge (backs of the west row)
    QX0 = QX1 - P["quay_w"]                                          # canal east edge
    CX0 = QX0 - P["canal_w"]                                         # canal west edge
    BX0 = CX0 - P["quay_w"]                                          # far bank west edge (warehouse fronts)
    CY = P["canal_len"] / 2
    WZ = P["water_z"]
    SW = P["street_w"]
    SX0, SX1 = cx - SW / 2, cx + SW / 2
    GD = 4.0                                                         # gate depth
    SY1 = Y0 - GD                                                    # street north end (gate back)
    SY0 = SY1 - P["street_len"]                                      # street south end
    COPE = 0.42                                                      # upper paving starts this far behind a wall face
    FD = P["fountain_d"]
    fx_, fy_ = cx, 0.5 + G.uniform(-0.5, 0.5)                        # fountain centre
    mx_ = (X1 + TX1) / 2                                             # monument axis (terrace centre line)
    OUT = -0.07                                                      # ground level outside the walls

    # ---- walkable regions ----------------------------------------------------------------------
    T.region("plaza", X0, Y0, X1, Y1, 0)
    T.region("forecourt", X0, Y1, X1, FY1, FH)
    T.region("terrace", X1, Y0, TX1, FY1, TH)
    pw_gate, pw_arc = 4.0, 3.0
    T.region("gate", cx - pw_gate / 2 + 0.1, SY1, cx + pw_gate / 2 - 0.1, Y0, 0)
    T.region("street", SX0, SY0, SX1, SY1, 0)
    T.region("arcade", QX1, -pw_arc / 2 + 0.1, X0, pw_arc / 2 - 0.1, 0)
    T.region("quay", QX0, -CY, QX1, CY, 0)
    bridge_w, bridge_rise = 3.2, 0.7
    T.region("bridge", CX0, -bridge_w / 2 + 0.45, QX0, bridge_w / 2 - 0.45, 0, arch={"axis": "x", "rise": bridge_rise})
    T.region("farbank", BX0, -CY, CX0, CY, 0)
    alley_y1 = None   # set with the east row

    # ---- ground --------------------------------------------------------------------------------
    T.ground("plaza", X0 - 1, Y0 - 1, X1 + COPE, Y1 + COPE, 0, "cobble_granite", wet=0.55, crown=0.04, center=[fx_, fy_])
    T.ground("forecourt", X0 - 1, Y1 + COPE, X1 + COPE, FY1 + 1, FH, "flagstone_york", wet=0.3)
    T.ground("terrace", X1 + COPE, Y0 - 1, TX1 + 1, FY1 + 1, TH, "flagstone_york", wet=0.35)
    T.ground("gate_floor", SX0, SY1 - 0.5, SX1, Y0 - 1, 0, "flagstone_york", wet=0.2)
    T.ground("street", SX0 - 1, SY0 - 1, SX1 + 1, SY1 - 0.5, 0, "sett_basalt", wet=0.6, crown=0.05)
    # (the arcade passage lays its own floor between the quay and the plaza paving)
    T.ground("quay", QX0 + COPE, -CY - 1, QX1 + 1, CY + 1, 0, "sett_basalt", wet=0.7)
    T.ground("farbank", BX0 - 1, -CY - 1, CX0 - COPE, CY + 1, 0, "cobble_granite", wet=0.6)
    T.ground("canal_bed", CX0 - 0.6, -CY - 6, QX0 + 0.6, CY + 6, WZ - 1.0, "soil")
    T.ground("canal_water", CX0 - 0.5, -CY - 6, QX0 + 0.5, CY + 6, WZ, "water", water=True)
    T.ground("base_e", QX0 + COPE, -70, 70, 70, OUT, "soil", step=6.0, bump=0)
    T.ground("base_w", -80, -70, CX0 - COPE, 70, OUT, "soil", step=6.0, bump=0)
    # sealing faces behind the wall assets
    T.skirt("sk_terrace", (X1 + COPE, FY1 + 1), (X1 + COPE, Y0 - 1), -0.2, TH)
    T.skirt("sk_fore", (X1 + COPE, Y1 + COPE), (X0 - 1, Y1 + COPE), -0.2, FH)
    T.skirt("sk_quay", (QX0 + COPE, -CY - 1), (QX0 + COPE, CY + 1), WZ - 1.0, 0)
    T.skirt("sk_bank", (CX0 - COPE, CY + 1), (CX0 - COPE, -CY - 1), WZ - 1.0, 0)

    # ---- level changes: stairs, retaining walls, balustrades ---------------------------------------
    # every flight registers its plan footprint (Town.stair): a keep-out (+ STAIR_MARGIN) for all ground props
    st_w, st_y = 5.0, -2.0 + G.uniform(-1.0, 1.0)                     # main terrace stair (rises east)
    si = T.put("stairs_plaza", X1, st_y, 0, -90, variant="*", tag="stair_terrace", collide="none", anchor="maxy", nav="stairs",
               width=st_w, rise=TH, seed=G.randrange(1, 99))
    ts, _ = T.stair(si, "stair_terrace")                              # (x0, y0, x1, y1) of the flight; its foot is x0
    T.ground("stair_landing", X1 - 0.02, st_y - st_w / 2 - 0.4, X1 + COPE + 0.05, st_y + st_w / 2 + 0.4, TH, "flagstone_york")
    fs_w = 8.0                                                        # forecourt steps (rise north)
    si = T.put("stairs_plaza", cx, Y1, 0, 0, variant="*", tag="stair_fore", collide="none", anchor="maxy", nav="stairs",
               width=fs_w, rise=FH, seed=G.randrange(1, 99))
    fs, _ = T.stair(si, "stair_fore")                                 # its foot is y0
    T.ground("fore_landing", cx - fs_w / 2 - 0.4, Y1 - 0.02, cx + fs_w / 2 + 0.4, Y1 + COPE + 0.05, FH, "flagstone_york")
    # forecourt -> terrace steps (rise east): the flight lands south of the monument's base (its depth as built:
    # measured in the pass before) as far as the forecourt allows
    ns_w = 2.6
    ns_y = max(Y1 + ns_w / 2 + 0.1, min((Y1 + FY1) / 2 + 0.3, FY1 - T.known.get("mon_d", 3.1) - 0.3 - ns_w / 2))
    si = T.put("stairs_plaza", X1, ns_y, FH, -90, variant="*", tag="stair_ne", collide="none", anchor="maxy", nav="stairs",
               width=ns_w, rise=TH - FH, seed=G.randrange(1, 99))
    ns, _ = T.stair(si, "stair_ne")
    T.ground("ne_landing", X1 - 0.02, ns_y - ns_w / 2 - 0.3, X1 + COPE + 0.05, ns_y + ns_w / 2 + 0.3, TH, "flagstone_york")

    def edge(a, b, z_low, h, tag, bal=True, lamps=True):
        """retaining wall (front to the right of a->b) with a balustrade on its coping"""
        T.run("retaining_wall", a, b, z_low, tag=tag, anchor="miny", height=h, collide="none", seeds=3)
        if bal:
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            nx, ny = dy / L, -dx / L                                  # right of travel = low side
            off = -0.24
            T.run("balustrade", (a[0] + nx * off, a[1] + ny * off), (b[0] + nx * off, b[1] + ny * off), z_low + h, tag=tag,
                  seeds=2, max_seg=6.5)

    # terrace edge x = X1, wall faces west: travel north -> right is east; so travel SOUTH (right = west)
    edge((X1, Y1), (X1, st_y + st_w / 2), 0, TH, "tw_n")
    edge((X1, st_y - st_w / 2), (X1, Y0), 0, TH, "tw_s")
    edge((X1, FY1), (X1, ns_y + ns_w / 2), FH, TH - FH, "tw_ne1")
    edge((X1, ns_y - ns_w / 2), (X1, Y1), FH, TH - FH, "tw_ne2")
    # forecourt edge y = Y1, faces south: travel west -> right is north; travel EAST? east -> right is south. yes.
    edge((X0, Y1), (cx - fs_w / 2, Y1), 0, FH, "fw_w")
    edge((cx + fs_w / 2, Y1), (X1, Y1), 0, FH, "fw_e")

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

    # buttresses + wall dressing along the tall terrace wall. A buttress dies into the wall below the coping: the asset
    # is built at the lowest height it offers and scaled down to the wall when that is lower still; a short one is a
    # single shallow stage of clean stone (deep, mossy weatherings on a 2 m wall read as a green ramp)
    but_h = max(TH * 0.92, (asset_spec("buttress")[0].get("height") or {}).get("min", 2.2)) if asset_path("buttress") else TH * 0.92
    tw_taken = [(Y1 - st_y, st_w / 2 + 0.6)]                          # distances along the wall, measured south from Y1
    for i, yy in enumerate([st_y + st_w / 2 + 2.2, st_y + st_w / 2 + 7.0, st_y - st_w / 2 - 2.2, st_y - st_w / 2 - 7.0]):
        if Y0 + 1.5 < yy < Y1 - 1.5:
            T.put("buttress", X1 - 0.02, yy, 0, -90, variant="*", tag="tw_but", height=but_h, scale=round(TH * 0.92 / but_h, 3),
                  depth=round(min(0.36 * but_h, 0.4 * TH), 2), stages=1 if but_h < 3.0 else 2, moss=round(0.1 + 0.03 * i, 2), r=0.5, seed=3 + i)
            tw_taken.append((Y1 - yy, 0.55))
    T.put("wall_fountain", X1 - 0.05, st_y + st_w / 2 + 4.6, 0, -90, variant="*", tag="tw", r=1.0, seed=G.randrange(99))
    tw_taken.append((Y1 - (st_y + st_w / 2 + 4.6), 1.3))
    for i, yy in enumerate([st_y + st_w / 2 + 1.0, st_y - st_w / 2 - 1.0, Y1 - 3.0, Y0 + 3.5]):
        T.put("wall_lantern", X1 - 0.03, yy, min(1.9, TH - 0.3), -90, variant="*", tag="tw_l", collide="none", seed=10 + i)
        tw_taken.append((Y1 - yy, 0.45))
    for i, yy in enumerate([Y0 + 6.2, Y1 - 6.5, st_y - st_w / 2 - 4.4]):
        T.put("ivy_patch", X1 - 0.04, yy, 0, -90, variant="*", tag="tw_ivy", collide="none", seed=20 + i)
        tw_taken.append((Y1 - yy, 1.0))
    growth((X1 - 0.04, Y1), (X1 - 0.04, Y0), 4, "tw_ivy2", tw_taken, h=(1.5, 2.5), w=(0.9, 1.7), variant=["young", "old_leaning"])

    # ---- buildings -------------------------------------------------------------------------------
    sb = P["storeys_bias"]
    # west row (faces east), south -> north
    cafe_w = 11.5
    arc_w = 7.0
    hotel_w = (Y1 + 0.5) - (Y0 + cafe_w + arc_w)
    arc_y = Y0 + cafe_w + arc_w / 2
    T.regions[5]["rect"][1] = round(arc_y - pw_arc / 2 + 0.1, 3)
    T.regions[5]["rect"][3] = round(arc_y + pw_arc / 2 - 0.1, 3)
    cafe_i = T.row((X0, Y0), (1, 0), [("cafe_gallery", cafe_w, RD, {"storeys": 2, "gallery_depth": 3.2})], tag="west_cafe")[0]
    arc_i = T.put("arcade_passage", X0 - RD / 2, arc_y, 0, 90, variant="*", tag="arcade", collide="passage",
                  width=arc_w, depth=RD, passage_width=pw_arc, storeys=3, seed=G.randrange(1, 999))
    hotel_i = T.row((X0, Y0 + cafe_w + arc_w), (1, 0), [("hotel", hotel_w, RD, {"storeys": 3 + sb})], tag="west_hotel")[0]
    # the north-west corner between the hotel and the hall is closed by a tall house on the forecourt
    nw_w = FY1 - (Y1 + 0.5)
    nw_row = T.row((X0, Y1 + 0.5), (1, 0), [("townhouse_narrow" if 3.5 <= nw_w <= 5.5 else "house_stone", nw_w, RD, {"storeys": 4})], z=FH, tag="nw")
    # north: clock tower hall on the forecourt
    hall_i = T.put("clock_tower_hall", cx, FY1, FH, 0, variant="*", tag="hall", collide="building", width=P["plaza_w"], depth=13.0,
                   seed=G.randrange(1, 999))
    # monument at the north end of the terrace
    mon_i = T.put("monument_star", mx_, FY1, TH, 0, variant=None, tag="monument", collide="building",
                  width=min(10.0, P["terrace_w"]), height=7.6, palette=P.get("monument_palette", "amber"), star_points=5, glow=LAMP_CD(TX1 - X1) * 1.2,
                  seed=G.randrange(1, 999))
    mon_sc = sidecar(T.items[mon_i]) if mon_i is not None else None
    mon_w = T.items[mon_i]["params"]["width"] if mon_i is not None else 0.0
    mon_d = round(-mon_sc["bounds"]["min"][1], 2) if mon_sc else 3.1  # how far its base / steps / basins stand out
    T.seen["mon_d"] = mon_d
    # east row (faces west), north -> south, with an alley
    alley_w = 3.0
    east = [("house_stone", 6.0, BD, {"storeys": 2}), ("shop_tailor", 7.0, BD, {"storeys": 3}), ("shop_luggage", 6.0, BD, {"storeys": 2}),
            ("shop_corner", 7.5, BD, {"storeys": 3}), (None, alley_w, 0, None), ("house_timber", None, BD, {"storeys": 2})]
    used = sum(w for (_, w, _, _) in east if w)
    east[-1] = ("house_timber", (FY1 - Y0) - used, BD, {"storeys": 2, "window_boxes": 4})
    east_row = T.row((TX1, FY1), (-1, 0), east, z=TH, tag="east")
    alley_y1 = FY1 - sum(w for (_, w, _, _) in east[:4])
    alley_y0 = alley_y1 - alley_w
    T.region("alley", TX1, alley_y0 + 0.15, TX1 + BD - 0.6, alley_y1 - 0.15, TH)
    T.ground("alley", TX1 + 0.9, alley_y0 - 0.5, TX1 + BD + 0.5, alley_y1 + 0.5, TH, "sett_basalt", wet=0.7)
    T.run("wall_stone_run", (TX1 + BD - 0.3, alley_y1), (TX1 + BD - 0.3, alley_y0), TH, tag="alley_end", height=3.4, thickness=0.5)
    T.put("hanging_lantern", TX1 + 2.4, (alley_y0 + alley_y1) / 2, TH + 3.6, 0, variant="*", tag="alley", collide="none", seed=4)
    T.put("wall_lantern", TX1 + BD - 0.62, (alley_y0 + alley_y1) / 2, TH + 2.3, -90, variant="*", tag="alley", collide="none", seed=6)
    # south row (faces north), east -> west, with the town gate on the axis
    gate_w = SW
    s_east = [("house_stone", P["terrace_w"], BD, {"storeys": 2})]
    south_t = T.row((TX1, Y0), (0, 1), s_east, z=TH, tag="south_t")
    e_len = X1 - SX1
    south_e = T.row((X1, Y0), (0, 1), [("tower_house", 7.0, BD, {"storeys": 3}), ("townhouse_narrow", e_len - 7.0, BD, {"storeys": 3})], tag="south_e")
    gate_i = T.put("gate_arch_town", cx, Y0 - GD / 2, 0, 0, variant="*", tag="gate", collide="passage", width=gate_w, depth=GD,
                   passage_width=pw_gate, seed=G.randrange(1, 999))
    w_len = SX0 - X0
    south_w = T.row((SX0, Y0), (0, 1), [("house_timber", 6.5, BD, {"storeys": 3, "window_boxes": 4}), ("townhouse_narrow", w_len - 6.5, BD, {"storeys": 4})],
                    tag="south_w")
    # street outside the gate
    s_len = (Y0 - BD) - SY0
    street_w = T.row((SX0, SY0), (1, 0), [("townhouse_narrow", 5.0, BD, {"storeys": 3}), ("house_stone", s_len - 10.0, BD, {"storeys": 2}),
                                          ("townhouse_narrow", 5.0, BD, {"storeys": 4})], tag="street_w")
    street_e = T.row((SX1, Y0 - BD), (-1, 0), [("house_timber", 5.5, BD, {"storeys": 2, "window_boxes": 4}), ("townhouse_narrow", 5.0, BD, {"storeys": 3}),
                                               ("townhouse_narrow", s_len - 10.5, BD, {"storeys": 3})], tag="street_e")
    # street end: wall, piers and a closed iron gate
    gw = 3.2
    T.run("wall_stone_run", (cx - gw / 2 - 0.45, SY0 + 0.3), (SX0, SY0 + 0.3), 0, tag="st_end", height=3.2)   # travel west: faces north
    T.run("wall_stone_run", (SX1, SY0 + 0.3), (cx + gw / 2 + 0.45, SY0 + 0.3), 0, tag="st_end", height=3.2)
    for sx in (-1, 1):
        T.put("gate_pier", cx + sx * (gw / 2 + 0.2), SY0 + 0.3, 0, 180, variant="*", tag="st_end", height=3.4, r=0.5, intensity=LAMP_CD(SW) / 2, seed=5)
    T.put("iron_gate", cx, SY0 + 0.3, 0, 180, variant="*", tag="st_end", width=gw, height=2.9, intensity=LAMP_CD(SW) / 1.5, seed=3)
    # far bank warehouses (face east), south -> north: most of their windows are lit, they are the canal's far wall
    wl = 2 * CY
    T.row((BX0, -CY), (1, 0), [("warehouse", round(wl * 0.33, 2), 9.0, {"storeys": 3, "lit_ratio": 0.8}),
                               ("warehouse", round(wl * 0.36, 2), 9.0, {"storeys": 4, "lit_ratio": 0.85}),
                               ("warehouse", wl - round(wl * 0.33, 2) - round(wl * 0.36, 2), 9.0, {"storeys": 3, "lit_ratio": 0.75})], tag="bank")
    # perimeter closures (stone walls facing in)
    T.run("wall_stone_run", (QX1, Y0), (QX1, -CY), 0, tag="close_q", height=3.8)                             # travel south: faces west
    T.run("wall_stone_run", (QX1, CY), (QX1, Y1 + 0.5), 0, tag="close_q", height=3.8)
    T.run("wall_stone_run", (QX0, CY), (QX1, CY), 0, tag="close_qn", height=3.8)                             # travel east: faces south
    T.run("wall_stone_run", (BX0, CY), (CX0, CY), 0, tag="close_qn", height=3.8)
    T.run("wall_stone_run", (QX1, -CY), (QX0, -CY), 0, tag="close_qs", height=3.8)                           # travel west: faces north
    T.run("wall_stone_run", (CX0, -CY), (BX0, -CY), 0, tag="close_qs", height=3.8)
    # two views the scenery is laid out for: the vista up the terrace promenade through the monument, and the
    # bird's-eye camera high over the street end (position, target, fov)
    vista = ((TX1 - 3.6, Y0 + 8.5), (mx_, FY1))
    over = ((cx + 10.0, Y0 - 41.0, 30.0), (cx - 1.0, Y1 - 5.0, 9.5), 50)
    # backdrop roofscape (Town.backdrop: no row may cut into a building or another row): an inner ring behind the rows ...
    bz = 0
    # a row stands its own front reach (BR: half its depth + stagger + eaves) plus a yard behind the backs it closes;
    # the rows behind the street houses run from the front of the row across the street end to the south row's backs
    BR = 8.6
    s_lo, s_hi = SY0 - 8.0 + BR + 0.6, Y0 - BD - 1.6
    s_mid, s_run = (s_lo + s_hi) / 2, max(12.0, s_hi - s_lo)
    inner = [("bd_n", cx + 4, FY1 + 13 + BR, 0, 60.0, 11), ("bd_e", TX1 + BD + BR, 3, -90, 56.0, 12), ("bd_w", BX0 - 9 - BR, 0, 90, 56.0, 13),
             ("bd_s", cx, SY0 - 8.0, 180, 48.0, 14), ("bd_sw", SX0 - BD - BR, s_mid, 90, s_run, 15), ("bd_se", SX1 + BD + BR, s_mid, -90, s_run, 16)]
    for (tg, bx, by, br, bl, bs) in inner:
        T.backdrop(tg, bx, by, br, bl, bz, step=1.0, spill=SPILL, seed=bs)
    # ... a row beyond each end of the canal (the quay and the far bank must not end in a wall and empty sky) ...
    for i, sy in enumerate((1, -1)):
        T.backdrop("bd_canal", (QX0 + CX0) / 2 - 3.0, sy * (CY + 15.0), 0 if sy > 0 else 180, 20.0, bz, lamps=1, spill=SPILL, lit_ratio=0.7, seed=17 + i)
    # ... a second, taller ring further out: behind every inner row two longer / shorter ones pushed back and
    # slid sideways, plus a short one across each corner the ring leaves open, so the district reads as part of a
    # town from above ...
    # ... one landmark over them: a tall tower house at cathedral scale on the axis of the terrace promenade's vista
    # (the line from the promenade through the monument), right behind the inner ring. It stands BEFORE the second
    # ring is laid, so those rows slide / fall back clear of it: the vista ends on the tower from its eaves to its cap,
    # never on the flank of a row (far enough out that the engine only ever draws its lowest LOD) ...
    R3 = T.rng("ring3")
    lm_y = FY1 + 13 + 2 * BR + 5.0
    lm_x = vista[0][0] + (vista[1][0] - vista[0][0]) * (lm_y - vista[0][1]) / (vista[1][1] - vista[0][1])
    T.backdrop("bd_landmark", lm_x, lm_y, 0, None, bz, asset="tower_house", variant=["copper_cap_tall", "octagon_slate"], scale=2.2,
               lit=0.85, lantern=False, seed=7)
    # ... a second, taller ring further out: behind every inner row two longer / shorter ones pushed back and
    # slid sideways, so the district reads as part of a town from above ...
    R2 = T.rng("ring2")
    far = {"lamps": 0, "clock": False, "height_min": 9.0, "height_max": 15.0, "depth": 11.0}
    for (tg, bx, by, br, bl, bs) in inner[:4]:
        ox, oy = rot2(0, 1, br)                                       # away from the square
        ax, ay = rot2(1, 0, br)
        for k, side in enumerate((-1, 1)):
            ln = round(R2.uniform(30.0, 44.0), 1)
            push, slide = R2.uniform(17.0, 21.0), side * (bl * 0.5 - R2.uniform(4.0, 10.0))
            T.backdrop(tg + "_far", bx + ox * push + ax * slide, by + oy * push + ay * slide, br + R2.uniform(-7, 7), ln, bz,
                       scale=round(R2.uniform(1.25, 1.45), 2), reach=44.0, seed=R2.randrange(100, 999), **far)
    mid = ((QX0 + TX1) / 2, (SY0 + FY1) / 2)

    # ... and the ring is closed by rule: seen from the middle of the district (all round) and from the bird's-eye
    # camera (across its frame) no run of bearing wider than GAP may be left without a row that reaches the skyline;
    # every such run gets a short row across it, facing the viewpoint, at the depth of the nearer row beside it and
    # pushed out along the sight line until it stands clear (Town.backdrop) ...
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
            sc = round(RG.uniform(1.35, 1.6), 2)
            a = math.radians(run[0] + (len(run) - 1) / 2.0)
            ln = max(12.0, min(40.0, 4.0 * math.ceil(2 * d * math.tan(math.radians(len(run) + 4) / 2) / sc / 4.0)))   # (whole 4 m: few distinct builds)
            qx, qy = view[0] + d * math.cos(a), view[1] + d * math.sin(a)
            if T.backdrop("bd_gap", qx, qy, facing_rot(view[0] - qx, view[1] - qy), round(ln, 1), bz, scale=sc, fill=True, reach=44.0,
                          lit_ratio=0.5, seed=RG.randrange(100, 999), **far) is None:
                return
    close_ring(mid, 0.0, 0, 360, "mid")
    oyaw = math.degrees(math.atan2(over[1][1] - over[0][1], over[1][0] - over[0][0]))
    ohf = math.degrees(math.atan(math.tan(math.radians(over[2]) / 2) * ASPECT))
    close_ring(over[0][:2], over[0][2], oyaw - ohf, oyaw + ohf, "over")
    # ... the hook of the far landmark on the north axis beyond the hall, outside the second ring: stands only once
    # the asset `castle_far` is released (the third ring is laid after it and gives way to it) ...
    sky_r = max(math.hypot(TX1 - mid[0], FY1 - mid[1]), math.hypot(BX0 - mid[0], SY0 - mid[1])) + 70.0
    sky_s, sky_d = 1.5, 22.0                                          # row scale, depth of a far row as built at that scale
    if asset_path("castle_far"):
        T.backdrop("bd_far_landmark", cx, mid[1] + sky_r - 22.0, 0, None, bz, asset="castle_far", variant="*", reach=40.0, seed=1)
    # ... and the town beyond: a closed third ring of the asset's far layer (tall plain hulls sunk in night haze, lit
    # windows only, no lamps = no lights) well outside the second, its rows the sides of a polygon around the
    # district, alternately nearer and further so they overlap in depth from every review camera
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
    ring_r = FD / 2 + 3.6
    foot = 2.0                                                        # clear approach in front of a flight of stairs
    T.route("gate_passage", cx - pw_gate / 2 + 0.1, SY1, cx + pw_gate / 2 - 0.1, Y0)
    T.route("gate_walk", cx - 1.6, Y0, cx + 1.6, fy_ - ring_r - 1.2)                                # gate -> fountain -> hall axis
    T.route("fountain_s", cx - 1.2, fy_ - ring_r - 1.2, cx + 1.2, fy_ - FD / 2 - 0.3)
    T.route("fountain_n", cx - 1.2, fy_ + FD / 2 + 0.3, cx + 1.2, fy_ + ring_r + 1.2)
    T.route("hall_walk", cx - 1.6, fy_ + ring_r + 1.2, cx + 1.6, fs[1] - foot)
    T.route("stair_fore_foot", cx - fs_w / 2, fs[1] - foot, cx + fs_w / 2, fs[1])
    T.route("stair_fore_head", cx - fs_w / 2, Y1, cx + fs_w / 2, Y1 + 1.5, FH)
    T.route("fore_walk", cx - 1.6, Y1 + 1.5, cx + 1.6, FY1, FH)
    fc_y = (Y1 + FY1) / 2 + 0.3                                                                     # mid forecourt
    T.route("fore_cross", cx + 1.6, fc_y - 0.6, ns[0] - 1.6, fc_y + 0.6, FH)                        # hall front -> foot of the terrace steps
    T.route("stair_ne_foot", ns[0] - 1.6, ns_y - ns_w / 2, ns[0], ns_y + ns_w / 2, FH)
    T.route("stair_ne_head", X1, ns_y - ns_w / 2, X1 + 1.8, ns_y + ns_w / 2, TH)
    T.route("fountain_w", fx_ - ring_r - 1.5, fy_ - 1.2, fx_ - FD / 2 - 0.3, fy_ + 1.2)
    T.route("west_walk", X0 + 5.0, min(arc_y, fy_) - 1.2, fx_ - ring_r - 1.5, max(arc_y, fy_) + 1.2)   # arcade -> fountain
    T.route("arcade_mouth", X0, arc_y - 1.6, X0 + 5.0, arc_y + 1.6)
    T.route("arcade", QX1, arc_y - 0.7, X0, arc_y + 0.7)
    T.route("arcade_quay", QX0 + 0.4, arc_y - 1.6, QX1, arc_y + 1.6)
    T.route("bridge", CX0, -bridge_w / 2 + 0.45, QX0 + 0.4, bridge_w / 2 - 0.45)
    T.route("bridge_west", CX0 - 2.6, -1.6, CX0, 1.6)
    T.route("quay_walk", QX0 + 1.85, -CY + 1.0, QX0 + 2.2, CY - 1.0)
    T.route("bank_walk", CX0 - 2.2, -CY + 1.0, CX0 - 1.85, CY - 1.0)
    T.route("fountain_e", fx_ + FD / 2 + 0.3, fy_ - 1.2, fx_ + ring_r + 1.5, fy_ + 1.2)
    T.route("east_walk", fx_ + ring_r + 1.5, min(st_y, fy_) - 1.2, ts[0] - foot, max(st_y, fy_) + 1.2)   # fountain -> terrace stair
    T.route("stair_terrace_foot", ts[0] - foot, st_y - st_w / 2, ts[0], st_y + st_w / 2)
    T.route("stair_terrace_head", X1, st_y - st_w / 2, X1 + 2.2, st_y + st_w / 2, TH)
    T.route("terrace_cross", X1 + 2.2, st_y - 1.3, TX1 - 1.6, st_y + 1.3, TH)                       # stair head -> shop fronts
    T.route("terrace_walk", mx_ + 0.2, Y0 + 6.5, mx_ + 2.6, FY1 - 5.6, TH)                          # promenade along the shops
    T.route("alley_mouth", TX1 - 2.0, alley_y0, TX1, alley_y1, TH)
    T.route("alley", TX1, (alley_y0 + alley_y1) / 2 - 0.7, TX1 + BD - 1.5, (alley_y0 + alley_y1) / 2 + 0.7, TH)
    T.route("street_walk", cx - 1.5, SY0 + 1.5, cx + 1.5, SY1)
    rows = [b for b in [hotel_i, hall_i] + nw_row + east_row + south_t + south_e + south_w + street_w + street_e if b is not None]
    for b in rows:                                                    # in front of every door the buildings declare
        T.doors(b, T.items[b]["tag"])

    # ---- canal -----------------------------------------------------------------------------------
    cd = P["canal_depth"]
    # east bank wall faces west: travel north -> right = east; travel SOUTH -> right = west
    T.run("canal_wall", (QX0, CY), (QX0, bridge_w / 2), 0, tag="cw_e", anchor="miny", depth=cd, collide="none", seeds=3)
    T.run("canal_wall", (QX0, -bridge_w / 2), (QX0, -CY), 0, tag="cw_e", anchor="miny", depth=cd, collide="none", seeds=3)
    T.run("canal_wall", (CX0, -CY), (CX0, -bridge_w / 2), 0, tag="cw_w", anchor="miny", depth=cd, collide="none", seeds=3)
    T.run("canal_wall", (CX0, bridge_w / 2), (CX0, CY), 0, tag="cw_w", anchor="miny", depth=cd, collide="none", seeds=3)
    T.put("bridge_stone", (QX0 + CX0) / 2, 0, 0, 0, variant="*", tag="bridge", collide="none", span=P["canal_w"], width=bridge_w,
          rise=bridge_rise, seed=G.randrange(1, 99))
    T.put("tunnel_mouth", (QX0 + CX0) / 2, CY, 0, 0, variant="*", tag="tunnel_n", collide="none", width=P["canal_w"], height=2.4, seed=2)
    T.put("tunnel_mouth", (QX0 + CX0) / 2, -CY, 0, 180, variant="*", tag="tunnel_s", collide="none", width=P["canal_w"], height=2.4, seed=5)
    for (ya, yb, tg) in ((bridge_w / 2 + 0.5, CY - 0.6, "n"), (-CY + 0.6, -bridge_w / 2 - 0.5, "s")):
        T.run("iron_railing", (QX0 + 0.22, ya), (QX0 + 0.22, yb), 0, tag="rail_e" + tg, height=1.05, seeds=2, max_seg=6.0)
        T.run("iron_railing", (CX0 - 0.22, yb), (CX0 - 0.22, ya), 0, tag="rail_w" + tg, height=1.05, seeds=2, max_seg=6.0)
    for i, yy in enumerate([-CY + 5.5, -8.0, 9.0, CY - 5.0]):
        T.put("mooring_post", QX0 + 0.7, yy + G.uniform(-0.8, 0.8), 0, G.uniform(0, 360), variant="*", tag="quay", r=0.25, seed=i)
    boats = []
    for i, (yy, side) in enumerate([(-11.0, 1), (7.5, -1), (15.0, 1)]):
        bx = (QX0 - 1.2) if side > 0 else (CX0 + 1.2)
        by = yy + G.uniform(-1.5, 1.5)
        T.put("rowboat", bx, by, WZ - 0.12, 90 + G.uniform(-8, 8), variant="*", tag="canal", collide="none", seed=3 + i)
        boats.append((side, by))
    for i, yy in enumerate([-15.0, -5.5, 5.5, 14.5]):
        T.put("lamp_post_gas", QX1 - 0.9 if i % 2 else QX0 + 0.95, yy + G.uniform(-0.6, 0.6), 0, 90, variant=None, tag="quay_lamp", r=0.3,
              height=3.3, along=(0, 1), seed=30 + i)
        T.put("lamp_post_globe", CX0 - 0.95 if i % 2 else BX0 + 0.9, -yy + G.uniform(-0.6, 0.6), 0, 90, variant="*", tag="bank_lamp", r=0.3, along=(0, 1), seed=40 + i)
    # exterior stairs on the quay: one up to every door a building declares above the ground on its canal side, its
    # landing (the stair's own door socket) brought to that door - never a flight up to a blank wall
    sp_rise = asset_spec("external_stair")[0].get("rise", {}) if asset_path("external_stair") else {}
    for b in (hotel_i, arc_i, cafe_i, *nw_row):
        for i, q in enumerate(T.sockets(b, "door") if b is not None else []):
            rise = q["z"] - 0.0
            if abs(q["x"] - QX1) > 0.8 or not (sp_rise.get("min", 2.2) <= rise <= sp_rise.get("max", 3.8)):
                continue
            si = T.put("external_stair", QX1 - 0.02, q["y"], 0, facing_rot(-1, 0), variant="*", tag="quay_stair", rise=round(rise, 2), seed=i + 1, r=1.9)
            own = T.sockets(si, "door") if si is not None else []
            if own:                                                   # slide the flight along the wall: landing to door
                T.items[si]["pos"][1] = round(T.items[si]["pos"][1] + q["y"] - own[0]["y"], 3)
                T.occ[-1] = (T.occ[-1][0], T.items[si]["pos"][1]) + tuple(T.occ[-1][2:])
    # more boats moored along both banks: seeded berths clear of the bridge and of each other
    RB = T.rng("boats")
    for _ in range(40):
        if len(boats) >= 5:
            break
        side, by = RB.choice((1, -1)), RB.uniform(-CY + 4.0, CY - 4.0)
        if abs(by) < bridge_w / 2 + 3.5 or any(s == side and abs(by - b) < 5.5 for s, b in boats):
            continue
        T.put("rowboat", (QX0 - 1.2) if side > 0 else (CX0 + 1.2), by, WZ - 0.12, 90 + RB.uniform(-8, 8) + RB.choice((0, 180)), variant="*",
              tag="canal", collide="none", seed=RB.randrange(99))
        boats.append((side, by))

    # ---- plaza centre ----------------------------------------------------------------------------
    T.put("paving_patch", fx_, fy_, 0.0, G.uniform(0, 360), variant="*", tag="centre", collide="none", diameter=FD + 6.0, seed=G.randrange(99))
    T.put("fountain_tiered", fx_, fy_, 0, G.uniform(0, 360), variant="*", tag="centre", collide="circle", diameter=FD, seed=G.randrange(99), r=FD / 2 + 0.3)
    # ring of lamps and benches: set between the four walks that cross at the fountain (never on them)
    n_ring = 8
    a0 = 180.0 / n_ring + (G.uniform(0, 360) / 360.0 - 0.5) * 4.0
    ring_a0 = a0
    ring_lamps = []
    for i in range(n_ring):
        a = math.radians(a0 + i * 360 / n_ring)
        x, y = fx_ + ring_r * math.cos(a), fy_ + ring_r * math.sin(a)
        rot = facing_rot(-math.cos(a), -math.sin(a))                 # front (-Y) towards the fountain
        if i % 2 == 0:
            li = T.put("lamp_post_gas", x, y, 0, rot, variant=["twin", "triple"], tag="ring_lamp", r=0.45, seed=70 + i)
            if li is not None:
                T.put("hanging_basket", x, y, 0, rot, variant="*", tag="ring_basket", collide="none", attach={"to": li, "role": "planter"}, seed=i)
        else:
            T.put("bench_park", x, y, 0, rot, variant="*", tag="ring_bench", r=1.1, foot=0.95, seed=80 + i)

    # round raised beds on the four diagonals between the walks, inside the ring (each only where no frame cuts it)
    for i in range(4):
        a = math.radians(45.0 + 90.0 * i)
        T.stands("flower_bed", fx_ + (FD / 2 + 1.7) * math.cos(a), fy_ + (FD / 2 + 1.7) * math.sin(a), 0, G.uniform(0, 360),
                 variant=["round_rustic_young", "round_moonlight"], tag="centre_bed", r=1.1, seed=60 + i)

    def basket(li, k):
        """a flower basket on the lamp post's bracket (no candle: the lamp is the light)"""
        if li is not None and T.sockets(li, "planter"):
            T.put("hanging_basket", T.items[li]["pos"][0], T.items[li]["pos"][1], T.items[li]["pos"][2], T.items[li]["rot"], variant="*",
                  tag="basket", collide="none", attach={"to": li, "role": "planter"}, candle=False, seed=k)

    # ---- fixed furniture of the rules below (lamps first: everything else keeps clear of them) ------------
    # lamp posts along the edges of the square; the ones at the stairs stand just outside the flights' keep-out
    lm = STAIR_MARGIN + 1.0
    edge_lamps = [(X0 + 3.4, Y1 - 1.6), (X1 - 1.4, Y1 - 1.5), (X1 - 1.3, Y0 + 1.4), (X0 + 3.4, Y0 + 1.5),
                  (fs[0] - lm, fs[1] - 0.4), (fs[2] + lm, fs[1] - 0.4), (ts[0] + 1.6, ts[1] - lm), (ts[0] + 1.6, ts[3] + lm)]
    for i, (lx, ly) in enumerate(edge_lamps):
        li = T.put("lamp_post_gas" if i < 4 or i > 5 else "lamp_post_globe", lx, ly, 0, G.uniform(0, 360), variant="*", tag="edge_lamp", r=0.4, seed=100 + i)
        if i < 4 or i > 5:
            basket(li, i)
    cafe_lamps = [T.put("lamp_post_globe", X0 + 9.0, Y0 + cafe_w - 0.2, 0, 0, variant="*", tag="cafe", r=0.4, seed=3),
                  T.put("lamp_post_globe", X0 + 9.0, Y0 + 0.9, 0, 0, variant="*", tag="cafe", r=0.4, seed=4)]
    # market stalls below the terrace wall
    mz = (X1 - 6.4, st_y + st_w / 2 + 1.2, X1 - 0.9, Y1 - 4.2)
    for i in range(2):
        my = mz[1] + 2.0 + i * 4.6
        if my < mz[3]:
            T.put("market_stall", X1 - 2.6, my, 0, -90 + G.uniform(-6, 6), variant="*", tag="market", r=1.9, seed=G.randrange(99))
    # forecourt: lamps flanking the head of the steps, hedges and topiary against the hall
    for k, sx in enumerate((-1, 1)):
        li = T.put("lamp_post_gas", (fs[0] - lm) if sx < 0 else (fs[2] + lm), Y1 + 1.0, FH, 0, variant=["triple", "twin"], tag="fore_lamp", r=0.4, seed=12)
        basket(li, 20 + k)
        T.run("hedge_run", (cx + sx * 6.5, FY1 - 0.75), (cx + sx * 14.5, FY1 - 0.75), FH, tag="fore_hedge", height=0.9, seeds=2, max_seg=4.0)
        T.put("topiary", cx + sx * 5.2, FY1 - 1.0, FH, 0, variant="*", tag="fore", r=0.6, seed=3)
    # terrace: lamps along the balustrade, each with a basket and (where the bracket is high enough) a banner
    ter_lamps = []
    for i, yy in enumerate([Y0 + 3.0, Y0 + 11.0, st_y + st_w / 2 + 5.0, Y1 - 1.0]):
        li = T.put("lamp_post_gas", X1 + 1.0, yy, TH, G.uniform(0, 360), variant=LAMP_ONE, tag="ter_lamp", r=0.4, along=(0, 1),
                   intensity=LAMP_CD(TX1 - X1), seed=120 + i)
        ter_lamps.append(li)
        basket(li, 30 + i)
    # monument court, mirrored about the monument's axis - a pair stands on both sides or not at all (Town.pair), so a
    # door stoop or the head of the flight on one side never leaves half a court: benches on the seats the monument
    # offers (a hand off its base, pulled in up to 1 m), lanterns by the steps, a chained-off front between planters,
    # and cypresses / a column / a plaque only where the monument leaves wall beside it
    seats = sorted(T.sockets(mon_i, "seat"), key=lambda q: q["x"])
    if len(seats) == 2:
        T.pair("bench_stone", mx_, seats[0]["y"] - 0.1, (seats[1]["x"] - seats[0]["x"]) / 2, TH, pull=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
               tag="monument", r=1.1, foot=0.65, seed=5)
    # (the court's light comes from lanterns one can see: the monument's own on its piers and steps - only a monument
    # built without lanterns at walking height gets a pair of open-flamed ones stood by its steps)
    if mon_i is not None and sidecar(T.items[mon_i]) is not None and not T.lanterns(mon_i, 2.0):
        T.pair("ground_lantern", mx_, FY1 - mon_d - 0.55, 1.7, TH, variant=["stone_octagon", "iron_ogee"], tag="monument", r=0.3,
               intensity=LAMP_CD(TX1 - X1) / 2, seed=6)
    cb = T.stands("chain_barrier", mx_, FY1 - 5.0, TH, 0, variant=["open_gate"], tag="monument", collide="none", length=5.6, seed=2, r=0.3)   # walked through: its middle span stands open
    if cb is not None:
        T.pair("planter_box", mx_, FY1 - 5.0, 4.45, TH, pull=(0.0, 0.5, 1.0), variant=["flower_tub", "ball_cream"], tag="monument", r=0.5, seed=4)
    if (TX1 - X1) - mon_w >= 2.4:
        side = mon_w / 2 + ((TX1 - X1) - mon_w) / 4
        T.pair("tree_cypress", mx_, FY1 - 0.9, side, TH, variant="*", tag="monument", collide="circle:0.35", r=0.7, seed=4)
        T.put("plaque", mx_ + side, FY1 - 0.06, TH + 1.5, 0, variant="*", tag="monument", collide="none", seed=3)
        T.stands("column", X1 + 0.75, FY1 - 0.75, TH, 0, variant="*", tag="monument", height=3.2, diameter=0.42, r=0.4, seed=2)
    # gate: braziers outside, urns inside, bollards and planters lining the walk from the gate to the fountain
    for sx in (-1, 1):
        T.put("brazier", cx + sx * (pw_gate / 2 + 1.1), SY1 - 0.9, 0, 0, variant="*", tag="gate_out", r=0.5, seed=4 + sx)
        T.put("planter_urn", cx + sx * (pw_gate / 2 + 0.8), Y0 + 0.7, 0, 0, variant="*", tag="gate_in", r=0.45, seed=6 + sx)
        a, b = (cx + sx * 2.05, Y0 + 2.2), (cx + sx * 2.05, fy_ - ring_r - 1.6)
        T.bays(a if sx > 0 else b, b if sx > 0 else a, 2.9, ["bollard", "bollard", ("planter_urn", {"variant": ["flower_tazza", "fern_vase"]})],
               off=0.0, key=f"gate_walk{sx}", r=0.3)
    # street lamps
    k = 0
    yy = SY0 + 2.6
    while yy < SY1 - 2.0:
        side = 1 if k % 2 else -1
        li = T.put("lamp_post_gas", cx + side * (SW / 2 - 1.0), yy, 0, 90 if side < 0 else -90, variant=LAMP_ONE, tag="street_lamp", r=0.4, along=(0, 1),
                   intensity=LAMP_CD(SW), seed=140 + k)
        if k % 2 == 0:
            basket(li, 40 + k)
        yy += 5.2 + G.uniform(-0.5, 0.5)
        k += 1

    # the largest one-off first (its zone is the tightest): the kiosk by the south-east houses
    kiosk_x1 = X1 - 4.0 - BULK["newspaper_kiosk"][0]                  # stays 4 m clear of the terrace retaining wall
    T.place("newspaper_kiosk", [(cx + gate_w / 2 + 2.6, Y0 + 2.6, kiosk_x1, Y0 + 6.0)], BULK["newspaper_kiosk"][0] * 1.42 + 0.05, key="kiosk", limit=0.02, rot=lambda x, y: facing_rot(fx_ - x, fy_ - y))
    # plaza trees: once the lamps stand (the canopies keep clear of them), before the loose furniture (which keeps
    # clear of the trunks); never under the cafe gallery
    spots = [(X1 - 3.4, Y1 - 3.4), (X0 + 4.0, Y0 + 9.5), (X1 - 4.0, Y0 + 8.5), (cx + 6.5, Y1 - 3.0), (fs[0] - 3.4, Y1 - 3.2), (X1 - 7.5, Y0 + 3.2)]
    for i, (tx, ty) in enumerate(spots):
        tx, ty = tx + G.uniform(-0.6, 0.6), ty + G.uniform(-0.6, 0.6)
        at = None
        for k in range(6):
            at = T.tree_spot(max(tx, X0 + 4.6) if k else tx, ty, T.rng("tree", i, k), reach=4.5 if k == 0 else 2.0)
            if at is not None and not (at[0] < X0 + 3.2 + 0.9 and Y0 < at[1] < Y0 + cafe_w):
                break
            at = None
        if at is None:
            continue
        tx, ty = at
        T.tree("tree_plane", tx, ty, 0, G.uniform(0, 360), T.rng("tree_grate", i), variant="*", collide="circle:0.35", seed=90 + i, r=1.2, foot=0.45)

    # ---- one-off pieces: each takes the best valid spot of its zone (Town.place) ------------------------
    away = lambda x, y: facing_rot(x - fx_, y - fy_)                  # back to the fountain
    to_centre = lambda x, y: facing_rot(fx_ - x, fy_ - y)
    T.place("bandstand", [(X0 + 4.4, arc_y + 5.2, X0 + 8.6, Y1 - 4.4)], 3.2, key="bandstand", limit=0.1, rot=0, collide="none", diameter=5.0)
    T.place("well_stone", [(cx + 3.6, Y0 + 5.5, X1 - 4.6, ts[1] - 2.6)], 1.3, key="well", limit=0.05)
    T.place("street_clock", [(fs[2] + 2.0, Y1 - 7.6, X1 - 6.0, Y1 - 2.4), (X0 + 7.0, Y1 - 7.6, fs[0] - 2.0, Y1 - 2.4)], 0.5, key="clock", limit=0.012,
            rot=lambda x, y: facing_rot(0, -1))
    T.place("signpost_finger", [(cx + sx * 2.3 if sx > 0 else cx - 6.5, Y0 + 3.2, cx + 6.5 if sx > 0 else cx - 2.3, Y0 + 8.0) for sx in (1, -1)], 0.5,
            key="signpost", limit=0.012)
    T.place("water_pump", [(X0 + 1.0, arc_y + 2.4, X0 + 2.4, Y1 - 1.6), (X1 - 9.0, Y1 - 2.6, X1 - 5.0, Y1 - 1.3)], 0.8, key="pump", limit=0.02,
            rot=lambda x, y: facing_rot(1, 0) if x < cx else facing_rot(0, -1))
    T.place("post_box", [(cx + pw_gate / 2 + 0.7, Y0 + 0.5, cx + gate_w / 2 + 2.0, Y0 + 0.8)], 0.4, key="post_box", limit=0.03, rot=180)
    T.place("notice_board", [(cx - gate_w / 2 + 0.5, Y0 + 0.05, cx - pw_gate / 2 - 0.3, Y0 + 0.05)], 0.45, key="notice", limit=0.05, rot=180)
    T.place("menu_board", [(X0 + 3.6, Y0 + cafe_w - 0.6, X0 + 5.2, Y0 + cafe_w + 0.5)], 0.5, key="menu", limit=0.03, rot=60)
    for sx in (-1, 1):                                               # statues on the forecourt, looking out over the square
        T.place("statue_figure", [(cx + sx * 6.0 if sx > 0 else cx - 11.0, Y1 + 1.25, cx + 11.0 if sx > 0 else cx - 6.0, Y1 + 1.5)], 1.0, z=FH,
                key=f"statue{sx}", limit=0.08, rot=to_centre)
    # terrace one-offs
    T.place("birdbath_sundial", [(X1 + 0.9, ts[3] + 1.5, X1 + 2.6, Y1 - 1.5), (X1 + 0.9, Y0 + 4.5, X1 + 2.6, ts[1] - 1.5)], 0.6, z=TH, key="sundial",
            limit=0.006)
    pg = T.place("pergola", [(X1 + 3.5, Y0 + 2.3, X1 + 5.4, Y0 + 3.0)], 2.9, z=TH, key="pergola", limit=0.14, rot=0, collide="none",
                 length=5.0, width=2.6, height=2.5)
    if pg is not None:
        px, py = T.items[pg]["pos"][0], T.items[pg]["pos"][1]
        T.put("vine_wisteria", px, py + 1.3, TH, 180, variant="*", tag="pergola", collide="none", seed=G.randrange(99))
        T.put("bench_park", px, py + 0.2, TH, 0, variant="*", tag="pergola", fixed=True, of=pg, seed=31)
    T.place("litter_bin", [(X1 + 0.8, ts[3] + 0.5, X1 + 1.5, ts[3] + 2.2), (X1 + 0.8, ts[1] - 2.2, X1 + 1.5, ts[1] - 0.5)], 0.4, z=TH, key="bin", rot=90)
    shop = {T.items[b]["asset"]: b for b in east_row if b is not None}
    if "shop_luggage" in shop:
        T.place("luggage_stack", [T.front(shop["shop_luggage"], 0.6, 1.2)], 0.7, z=TH, key="luggage", rot=-90)
    if "shop_tailor" in shop:
        T.place("shop_mannequin", [T.front(shop["shop_tailor"], 0.5, 0.9)], 0.4, z=TH, key="tailor", rot=-90)
    if "shop_corner" in shop:
        T.place("fire_hydrant", [T.front(shop["shop_corner"], 0.6, 1.1, inset=0.3)], 0.35, z=TH, key="hydrant", rot=0)
    # a flower seller on the open paving between the gate and the cafe, with her stock around the cart
    fz = [(cx - 9.5, Y0 + 5.0, cx - 2.9, fy_ - ring_r - 2.0)]
    fc = T.place("hand_cart", fz, 1.3, key="flowers", limit=0.06, variant="flower_seller", lantern=False, rot=to_centre)
    if fc is not None:
        RF = T.rng("flowers")
        ox, oy = T.items[fc]["pos"][0], T.items[fc]["pos"][1]
        for k, nm in enumerate(["flower_pots", "bucket_set", "flower_pots"]):
            for _ in range(20):
                a, d = RF.uniform(0, math.tau), RF.uniform(1.7, 2.3)
                x, y = ox + d * math.cos(a), oy + d * math.sin(a)
                if T.free(x, y, 0.75) and T.put(nm, x, y, 0, RF.uniform(0, 360), variant="*", tag="flowers", r=0.75, fixed=True, quiet=True, seed=RF.randrange(99)) is not None:
                    break

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
            if T.put("pavement_weeds", x, y, z, R.uniform(0, 360), variant=["moss_joints", "parched_tufts"], tag="leaves", collide="none",
                     habit="joints", litter=round(R.uniform(0.8, 1.0), 2), spread=round(R.uniform(1.1, 1.5), 2), seed=R.randrange(99)) is not None:
                lie.append((x, y))
                done += 1
        return done

    # ---- canal dressing --------------------------------------------------------------------------
    QUAY = ["barrel", "crate_stack", "sack_pile", "barrel", "milk_churns", "bucket_set", "wheelbarrow", "ladder", "cat", "crate_stack"]
    # (the quay under the hotel / cafe backs is the far bank of the canal views: a piece to every 2.6 m of wall, flowers among the cargo)
    T.line((QX1, CY - 0.6), (QX1, -CY + 0.6), int(round((2 * CY - 1.2) / 2.6)), QUAY + ["flower_pots", "planter_box", "bench_park"], key="quay_e", r=0.7)
    T.line((BX0, -CY + 0.6), (BX0, CY - 0.6), 13, QUAY, key="quay_w", r=0.7)
    # the water's edge: mooring posts and small cargo between the railings and the walks
    EDGE = [("mooring_post", {"variant": ["bollard", "tall_capped", "painted_tarred_line", "mooring_stone"]}), "barrel",
            ("mooring_post", {"variant": ["bollard", "tall_capped", "painted_tarred_line", "mooring_stone"]}), "milk_churns", None]
    for (ya, yb, tg) in ((bridge_w / 2 + 1.2, CY - 1.0, "n"), (-CY + 1.0, -bridge_w / 2 - 1.2, "s")):
        T.bays((QX0 + 0.75, yb), (QX0 + 0.75, ya), 2.8, EDGE, off=0.0, key="edge_e" + tg, r=0.45)
        # (the far bank is a working wharf: cargo landed at the rail in every bay between the posts)
        T.bays((CX0 - 0.75, ya), (CX0 - 0.75, yb), 2.2, EDGE[:3] + ["crate_stack", EDGE[0], "sack_pile", "bucket_set"], off=0.0, key="edge_w" + tg, r=0.45)
    for i, yy in enumerate([-17.5, -6.5, 6.5, 18.0]):
        T.put("wall_lantern", QX1 - 0.02, yy, 2.6, facing_rot(-1, 0), variant="*", tag="quay_l", collide="none", seed=50 + i)
    # the warehouse fronts carry a lantern to every 7 m of wharf: its pool lies on the bank and climbs the brick
    n_bl = max(2, int(round(2 * CY / 7.0)))
    for i in range(n_bl):
        T.put("wall_lantern", BX0 + 0.02, -CY + (i + 0.5) * 2 * CY / n_bl, 2.8, facing_rot(1, 0), variant="*", tag="bank_l", collide="none",
              intensity=LAMP_CD(P["quay_w"]) / 2, seed=55 + i)
    strew((BX0 + 0.3, -CY + 0.5, CX0 - 0.4, CY - 0.5), 5, "bank")
    # the hotel advertises itself to the canal: a bulb-bordered marquee flat on its back wall, face-on to the far bank
    # and the bridge (a blade sign there is edge-on from both), slid along the wall by rule to clear what is built there
    hy = T.items[hotel_i]["pos"][1] if hotel_i is not None else arc_y + 8.0
    T.put("marquee_sign", QX1 - 0.03, hy, 6.4, facing_rot(-1, 0), variant=["grand_gable"], tag="hotel_back", collide="none",
          text="HOTEL", width=3.4, height=0.95, pediment_height=0.6, monogram="H", intensity=7.0, dead_bulbs=0.0, seed=3)
    growth((QX1 - 0.34, CY), (QX1 - 0.34, Y1 + 0.5), 1, "quay_ivy_n", [], h=(2.4, 3.4), w=(1.4, 2.4))
    growth((QX1 - 0.34, Y0), (QX1 - 0.34, -CY), 1, "quay_ivy_s", [], h=(2.4, 3.4), w=(1.4, 2.4))
    # arcade passage: the building dresses its own bays (shop windows, stalls, awnings, lamps, lanterns, floor); the
    # town only pastes posters on the plain bay walls it offers, stands one piece of goods by each wall where the
    # bays as built leave room, and hangs a projecting sign over each mouth
    for k, q in enumerate(sorted(T.sockets(arc_i, "poster"), key=lambda q: q["name"])):
        T.fit(arc_i, q, "wall_posters", tag="arc", variant="*", seed=2 + 3 * k)
    ARC = ["barrel", "bucket_set", "flower_pots", "cat"]            # (a different thing by each wall)
    T.put("drain_cover", (QX1 + X0) / 2, arc_y, 0.0, 0, variant=["gully_grate"], tag="arc_drain", collide="none", leaves=6, seed=5)   # the passage drains at its middle
    for k, (xa, xb) in enumerate(((QX1 - 1.6, QX1 + 1.4), (X0 - 1.4, X0 + 1.6))):          # leaves blow in at both mouths
        strew((xa, arc_y - pw_arc / 2 + 0.1, xb, arc_y + pw_arc / 2 - 0.1), 2, f"arcade{k}", r=0.5)
    T.line((QX1 + 0.8, arc_y + pw_arc / 2), (X0 - 0.8, arc_y + pw_arc / 2), 1, ARC[:2], key="arc_n", off=0.4, spread=0.02, r=0.32)   # travel east: off the north wall
    T.line((X0 - 0.8, arc_y - pw_arc / 2), (QX1 + 0.8, arc_y - pw_arc / 2), 1, ARC[2:], key="arc_s", off=0.4, spread=0.02, r=0.32)
    for k, q in enumerate(T.sockets(arc_i, "sign")):
        quay = q["x"] < X0 - RD / 2
        T.fit(arc_i, q, "hanging_sign", tag="arc_sign", variant="wine_shield" if quay else "bookshop_rect", lamp="swan" if quay else "none",
              text="QUAY INN" if quay else "MARKET", seed=4 + k)

    # ---- plaza -----------------------------------------------------------------------------------
    T.scatter((fx_ - ring_r, fy_ - ring_r, fx_ + ring_r, fy_ + ring_r), 3,
              lambda x, y, R, i: not T.cut(x, y, 0, 0.8, 0.3) and T.put("pigeons", x, y, 0, R.uniform(0, 360), variant="*", tag="centre", collide="none", seed=R.randrange(99)) is not None,
              r=0.8, key="pigeons")

    # ---- cafe terrace ----------------------------------------------------------------------------
    cz = (X0 + 3.6, Y0 + 1.2, X0 + 8.6, Y0 + cafe_w - 0.8)

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
        n = R.choice([2, 2, 3])
        a0 = R.uniform(0, 360)
        for k in range(n):
            a = math.radians(a0 + k * 360 / n + R.uniform(-15, 15))
            d = 0.72 + R.uniform(-0.05, 0.1)
            T.put("cafe_chair", x + d * math.cos(a), y + d * math.sin(a), z, facing_rot(-math.cos(a), -math.sin(a)) + R.uniform(-25, 25),
                  variant="*", tag=tag, collide="circle:0.25", fixed=True, of=ti, seed=R.randrange(999))
        T.put("table_lantern", x, y, z, R.uniform(0, 360), variant="*", tag=tag, collide="none", on=ti, intensity=TABLE_CD, seed=R.randrange(999))
        sd = R.randrange(999)
        if (shade and i % shade == 0 and T.sight(x, y, z, SHADE_D / 2, *BULK["parasol"][1:])[0] <= SIGHT_PUT and not T.cut(x, y, z, SHADE_D / 2, 2.9)
                and all(abs(z - sz) > 1.0 or math.hypot(x - sx, y - sy) > SHADE_D + 0.3 for sx, sy, sz in shades)):    # canopies never overlap
            if T.put("parasol", x + 0.05, y, z, R.uniform(0, 360), variant=["bistro_stripe", "market_green", "harlequin"], tag=tag, collide="none",
                     fixed=True, of=ti, foot=0.3, diameter=SHADE_D, lanterns=1, rim_lanterns=0, intensity=2 * TABLE_CD, seed=sd) is not None:
                shades.append((x, y, z))
        return True
    T.rows(cz, SET_P, lambda x, y, R, i: cafe_set(x, y, R, i, shade=2), r=SET_R, key="cafe")
    T.rows((X0 + 0.5, Y0 + 1.0, X0 + 3.0, Y0 + cafe_w - 1.0), SET_P, cafe_set, r=SET_R, key="cafe_in")       # under the gallery
    for i, yy in enumerate([Y0 + 0.8, Y0 + cafe_w - 0.7]):
        T.put("planter_box", X0 + 6.2, yy, 0, 0, variant="*", tag="cafe", r=0.9, seed=i + 2)
        T.put("barrel", X0 + 3.4, yy + (0.5 if i == 0 else -0.5), 0, G.uniform(0, 360), variant="*", tag="cafe", r=0.5, seed=i + 7)
    T.put("festoon_lights", X0 + 3.35, Y0 + 0.5, 3.0, 90, variant="*", tag="cafe", collide="none", span=cafe_w - 1.0, seed=1)
    for i, (y_a, y_b) in enumerate([(Y0 + cafe_w * 0.5, Y0 + cafe_w - 0.2), (Y0 + cafe_w * 0.5, Y0 + 0.9)]):
        dxx, dyy = 9.0 - 3.3, y_b - y_a
        T.put("festoon_lights", X0 + 3.3, y_a, 3.2, math.degrees(math.atan2(dyy, dxx)), variant="*", tag="cafe", collide="none",
              span=math.hypot(dxx, dyy), seed=2 + i)

    # ---- market corner (below the terrace wall) ----------------------------------------------------
    MARKET = ["crate_stack", "sack_pile", "barrel", "milk_churns", "hand_cart", "bucket_set", "flower_pots", "crate_stack"]

    def market_dress(x, y, R, i, names=MARKET):
        a = T.pick(R, names)
        return a is not None and T.put(a, x, y, 0, R.uniform(0, 360), variant="*", tag="market", r=0.75, quiet=True, seed=R.randrange(999)) is not None
    T.scatter(mz, 8, market_dress, r=0.75, key="market")
    # the market spills onto the open paving in front of the stalls (low goods only: the stalls stay in view)
    T.scatter((X1 - 10.4, mz[1] + 0.4, X1 - 6.9, mz[3] - 0.4), 5,
              lambda x, y, R, i: market_dress(x, y, R, i, ["crate_stack", "sack_pile", "barrel", "milk_churns", "bucket_set", "flower_pots"]),
              r=0.9, key="market_spill")

    # ---- general plaza dressing ---------------------------------------------------------------------
    WALL = ["barrel", "planter_box", "flower_pots", "bicycle", "crate_stack", "litter_bin", "planter_urn", "bucket_set", "cat", "ladder",
            "bench_park", "milk_churns"]
    # the hotel's cellar hatch first: what is leant and stood along its front keeps off the flaps
    T.put("cellar_hatch", X0 + 0.05, Y0 + cafe_w + arc_w + 2.5, 0, 90, variant="*", tag="hotel", collide="none", r=0.9, seed=4)
    T.line((X0, Y0 + cafe_w + arc_w + 0.6), (X0, Y1 - 0.6), 5, WALL, key="hotel_front")                     # travel north: props face east
    T.line((cx - gate_w / 2 - 0.6, Y0), (X0 + 0.8, Y0), 4, WALL, key="south_w_front")                      # travel west: props face north
    T.line((X1 - 5.5, Y0), (cx + gate_w / 2 + 0.6, Y0), 5, WALL, key="south_e_front")
    T.line((X0 + 0.6, Y1), (fs[0] - 1.0, Y1), 4, WALL, key="fore_wall_w", off=0.75)                         # travel east: props face south
    T.line((fs[2] + 1.0, Y1), (X1 - 2.0, Y1), 3, WALL, key="fore_wall_e", off=0.75)
    # drain covers: a row across the south of the square, a pair on the gate walk and one by the market, with fallen leaves
    RD_ = T.rng("drains")
    for i in range(5):
        dx_ = X0 + 3.0 + i * (P["plaza_w"] - 6.0) / 4 + G.uniform(-1, 1)
        T.put("drain_cover", dx_, Y0 + 2.6 + G.uniform(-0.4, 0.4), 0.0, G.choice([0, 90]), variant="*", tag="drain", collide="none", seed=i,
              leaves=RD_.randrange(0, 7))
    for i, (dx_, dy_) in enumerate([(cx + RD_.uniform(-0.9, 0.9), Y0 + 5.0 + RD_.uniform(-0.6, 0.6)), (cx + RD_.uniform(-0.9, 0.9), fy_ - ring_r - 2.2),
                                    (X1 - 8.6 + RD_.uniform(-0.8, 0.8), mz[1] + RD_.uniform(1.0, 5.0))]):
        T.put("drain_cover", dx_, dy_, 0.0, RD_.choice([0, 90]), variant="*", tag="drain", collide="none", seed=20 + i, leaves=RD_.randrange(4, 10))
    T.scatter((X0 + 0.2, Y0 + 0.2, X1 - 0.2, Y1 - 0.2), 14,
              lambda x, y, R, i: T.put("pavement_weeds", x, y, 0, R.uniform(0, 360), variant="*", tag="weeds", collide="none", seed=R.randrange(99)) is not None
              if (min(x - X0, X1 - x, y - Y0, Y1 - y) < 0.9) else False, r=0.3, key="weeds", tries=200)
    T.scatter((cx - 9.0, Y0 + 3.0, cx + 9.0, fy_ - ring_r - 1.0), 2,
              lambda x, y, R, i: not T.cut(x, y, 0, 0.8, 0.3) and T.put("pigeons", x, y, 0, R.uniform(0, 360), variant="*", tag="plaza", collide="none", seed=R.randrange(99)) is not None,
              r=0.8, key="pigeons_s")
    # bunting on the inner face of the gate: one swag on each side of the crest, none across it
    for sx in (-1, 1):
        x0b = cx + sx * 2.4 if sx > 0 else cx - gate_w / 2 - 2.0
        T.put("bunting", x0b, Y0 + 0.2, 5.6, 0, variant="*", tag="bunting", collide="none", span=gate_w / 2 + 2.0 - 2.4, seed=3 + sx)
    # garlands across the street: none in the stretch before the gate, and each one hangs above the line of sight
    # from the arrival point to the top of the gate's name board (so the board always reads)
    gate_top = T.param(gate_i, "passage_height", 5.0) + 1.9 if gate_i is not None else 6.9
    arrive = SY0 + 4.5
    for i, t in enumerate((0.2, 0.44, 0.66)):
        yy = SY0 + (SY1 - SY0) * t
        zg = 5.4
        if yy > arrive:
            zg = max(zg, eye + (gate_top - eye) * (yy - arrive) / (SY1 - arrive) + 0.75)
        if zg <= 6.4:
            T.put("bunting" if i != 1 else "festoon_lights", SX0 + 0.25, yy, round(zg, 2), 0, variant="*", tag="street_bunting", collide="none",
                  span=SW - 0.5, seed=5 + i, **({"intensity": 7.0} if i == 1 else {}))

    # ---- forecourt (hall front): benches, planters and beds along the balustrade ---------------------
    BED = {"variant": ["tall_royal", "rustic_sunset"], "length": 2.0}      # a raised bed one bay long
    FORE = [("flower_bed", dict(BED, turn=0.0)), ("bench_stone", {"turn": 0.0}), ("planter_box", {"variant": ["flower_tub", "ball_cream", "topiary_cone"]}),
            ("flower_bed", dict(BED, turn=0.0)), "planter_urn"]
    T.bays((fs[0] - 1.6, Y1 + 0.42), (X0 + 1.0, Y1 + 0.42), 2.9, FORE, z=FH, off=0.62, key="fore_w", r=0.85)        # travel west: props face north
    T.bays((ns[0] - 2.2, Y1 + 0.42), (fs[2] + 1.6, Y1 + 0.42), 2.9, FORE, z=FH, off=0.62, key="fore_e", r=0.85)

    # ---- terrace -----------------------------------------------------------------------------------
    # along the balustrade: a bench, a planter, a bed ... in every bay (the lamps are already there)
    TER = [("flower_bed", dict(BED, turn=90.0)), "bench_park", ("planter_box", {"variant": ["flower_tub", "ball_cream", "topiary_cone"]}),
           ("flower_bed", dict(BED, turn=90.0)), "planter_urn"]
    T.bays((X1 + 0.42, Y0 + 1.2), (X1 + 0.42, ts[1] - 0.2), 2.6, TER, z=TH, off=0.62, key="ter_s", r=0.8)            # travel north: props face east
    T.bays((X1 + 0.42, ts[3] + 0.2), (X1 + 0.42, Y1 - 0.2), 2.6, TER, z=TH, off=0.62, key="ter_n", r=0.8)
    T.bays((X1 + 0.42, Y1 + 0.4), (X1 + 0.42, ns[1] - 0.2), 2.2, TER[2:4], z=TH, off=0.62, key="ter_ne", r=0.7)
    # the terrace's trees (after the lamps and the balustrade's furniture: canopies and trunks keep clear of them;
    # before the cafe tables, which are set out round the trunks)
    for i, (tx, ty) in enumerate([(X1 + 2.9, Y0 + 7.4), (X1 + 2.9, st_y + st_w / 2 + 8.8), (TX1 - 2.6, Y0 + 2.2)]):
        at = T.tree_spot(tx, ty, T.rng("ter_tree", i), z=TH, r=1.0, canopy=2.6, reach=1.6)
        if at is not None:
            ti = T.put("tree_ornamental", at[0], at[1], TH, G.uniform(0, 360), variant="*", tag="terrace", collide="circle:0.3", r=1.0, foot=0.45,
                       fixed=True, seed=G.randrange(99))
            if ti is not None:
                T.yields.add(ti)
                T.crowns.append((at[0], at[1], TH, 2.6))
    # cafe tables under parasols between the balustrade and the promenade
    # (on both sides of the stair head, a file of sets along the balustrade side)
    for k, (y0_, y1_) in enumerate(((ts[3] + 1.0, Y1 - 1.0), (Y0 + 4.6, ts[1] - 1.0))):
        T.rows((X1 + 2.0, y0_, mx_ + 0.1, y1_), SET_P, lambda x, y, R, i: cafe_set(x, y, R, i, z=TH, tag="ter_cafe", shade=2),
               r=SET_R, z=TH, key="ter_cafe" if k == 0 else "ter_cafe_s")
    # the promenade's flags are not swept bare: a run of gully grates down its middle, one to every 5 m, with the
    # leaves caught on them, and drifts of leaf litter over the whole terrace
    RL = T.rng("ter_floor")
    pr0, pr1 = Y0 + 6.5, FY1 - 5.6
    n_g = int(round((pr1 - pr0) / 5.0))
    for i in range(n_g):
        T.put("drain_cover", mx_ + 1.4, pr0 + (i + 0.5) * (pr1 - pr0) / n_g + RL.uniform(-0.5, 0.5), TH, 0, variant=["gully_grate"], tag="ter_drain",
              collide="none", leaves=RL.randrange(4, 10), seed=30 + i)
    strew((X1 + 1.4, Y0 + 2.0, TX1 - 0.8, FY1 - 4.5), 12, "terrace", z=TH)
    SHOP = ["luggage_stack", "flower_pots", "planter_box", "barrel", "crate_stack", "bicycle", "planter_urn", "sack_pile", "ladder", "bench_park"]
    T.line((TX1, Y1 + 5.0), (TX1, Y0 + 0.5), 11, SHOP, z=TH, key="shops", r=0.85)                            # travel south: props face west
    T.line((TX1 + BD - 1.6, alley_y0 + 0.15), (TX1 + 1.0, alley_y0 + 0.15), 2, ["barrel", ("crate_stack", "pair_slatted"), "milk_churns", "cat"],
           z=TH, key="alley", off=0.4, spread=0.02, r=0.35)

    # ---- what the buildings offer: signs, awnings, banners, window boxes on their sockets ----------------
    RS = T.rng("fitout")
    SIGNS = ["cobbler_oval", "locksmith_cutout", "bookshop_rect", "tailor_scissors", "wine_shield"]
    for n, b in enumerate(rows):
        asset = T.items[b]["asset"]
        on_street = b in street_w or b in street_e
        signs = [q for q in T.sockets(b, "sign") if q["name"] in ("sign", "sign_front", "hanging_sign")]
        if signs and asset not in ("clock_tower_hall",) and (not on_street or n % 2 == 0):
            T.fit(b, signs[0], "hanging_sign", tag="sign", variant="tailor_scissors" if asset == "shop_tailor" else RS.choice(SIGNS), lamp="none",
                  seed=RS.randrange(99))
        for q in T.sockets(b, "awning")[:1]:
            T.fit(b, q, "awning", tag="awning", width=q["props"].get("width", 3.0), seed=RS.randrange(99))
        boxes = T.sockets(b, "window_box") + [q for q in T.sockets(b, "planter") if "planter" in q["name"] and q["z"] - T.items[b]["pos"][2] < 5.2]
        for q in boxes[:3]:
            T.fit(b, q, "window_box", tag="window_box", width=min(1.8, max(0.5, q["props"].get("width", 0.9))), seed=RS.randrange(99))
    # a flag on the corner house of the forecourt (after its sign: the flag slides along the wall to clear it)
    T.put("flag_wall", X0 + 0.3, Y1 + 3.0, FH + 3.2, 90, variant="*", tag="fore", collide="none", seed=2)
    # festival banners: every lamp post whose bracket is high enough to walk under (BANNER_H) flies a small crown
    # banner, the colours taking turns from post to post; the hall, the arcade and the cafe hang long ones
    k = 0
    for li, it in enumerate(list(T.items)):
        if it["asset"] in ("lamp_post_gas", "lamp_post_globe"):
            for q in T.sockets(li, "banner")[:1]:
                if q["z"] - it["pos"][2] >= BANNER_H:
                    k += T.fit(li, q, "banner_crown", tag="lamp_banner", width=0.5, drop=1.0, variant=[BANNERS[k % len(BANNERS)]], seed=2 + k) is not None
    for b, (bw, bd_) in ((hall_i, (1.3, 3.4)), (arc_i, (1.1, 2.8)), (cafe_i, (0.9, 2.0))):
        for k, q in enumerate(q for q in T.sockets(b, "banner") if not q["name"].startswith("bunting")):
            T.fit(b, q, "banner_crown", tag="banner", width=bw, drop=bd_, variant=BANNERS, seed=3 + k)

    # ---- gate + street ---------------------------------------------------------------------------
    T.put("banner_crown", cx - 2.9, SY1 - 0.08, 4.6, 0, variant="*", tag="gate_out", collide="none", seed=2)
    T.put("crest_cartouche", cx, Y0 + 0.06, 5.3, 180, variant="*", tag="gate_in", collide="none", seed=2)
    T.put("wall_posters", cx - pw_gate / 2 + 0.02, Y0 - GD / 2, 1.3, 90, variant="*", tag="gate", collide="none", seed=9)
    T.put("wall_clock", cx + 2.9, SY1 - 0.08, 4.4, 0, variant="*", tag="gate_out", collide="none", seed=2)
    T.put("ivy_patch", cx + gate_w / 2 - 0.9, Y0 + 0.03, 0, 180, variant=["young", "old_leaning"], tag="gate_ivy", collide="none", width=1.3, height=3.6, seed=4)
    T.put("ivy_patch", cx + gate_w / 2 - 0.8, SY1 - 0.03, 0, 0, variant=["young", "old_leaning"], tag="gate_ivy", collide="none", width=1.2, height=3.0, seed=6)
    STREET = ["barrel", "crate_stack", "planter_box", "flower_pots", "bicycle", "hand_cart", "litter_bin", "sack_pile", "bench_park",
              "milk_churns", "cat", "ladder"]
    T.line((SX0, SY0 + 1.2), (SX0, SY1 - 0.6), 7, STREET, key="street_w", r=0.85)                            # travel north: props face east
    T.line((SX1, SY1 - 0.6), (SX1, SY0 + 1.2), 7, STREET, key="street_e", r=0.85)                            # travel south: props face west
    for i in range(3):
        T.put("drain_cover", cx + G.uniform(-2.5, 2.5), SY0 + 3 + i * 6.0, 0, 0, variant="*", tag="drain", collide="none", seed=10 + i)
    for i, sx in enumerate((-1, 1)):
        T.run("kerb_gutter", (cx + sx * (SW / 2 - 2.1), SY0 + 0.8) if sx < 0 else (cx + sx * (SW / 2 - 2.1), SY1 - 0.3),
              (cx + sx * (SW / 2 - 2.1), SY1 - 0.3) if sx < 0 else (cx + sx * (SW / 2 - 2.1), SY0 + 0.8), 0, tag="kerb", collide="none", seeds=2, max_seg=6.0)
    T.scatter((SX0, SY0, SX1, SY1), 8,
              lambda x, y, R, i: T.put("pavement_weeds", x, y, 0, R.uniform(0, 360), variant="*", tag="weeds", collide="none", seed=R.randrange(99)) is not None
              if (min(x - SX0, SX1 - x) < 0.7) else False, r=0.3, key="weeds_st", tries=200)
    T.put("paper_lantern", cx - 1.6, SY1 - 0.6, 4.3, 0, variant="*", tag="gate_out", collide="none", intensity=LAMP_CD(SW) / 2, seed=3)
    T.put("paper_lantern", cx + 1.6, SY1 - 0.6, 4.3, 0, variant="*", tag="gate_out", collide="none", intensity=LAMP_CD(SW) / 2, seed=7)

    # ---- trees outside the walls -----------------------------------------------------------------------
    # trees outside the walls: behind the canal's end walls, in the yard north of the corner house, behind the monument
    RO = T.rng("outside")
    outside = [(QX0 + 2.0, CY + 3.6), (BX0 + 2.2, CY + 4.2), (QX0 + 2.2, -CY - 3.8), (BX0 + 2.0, -CY - 4.4),
               (X0 - 4.5, FY1 + 3.5), (X0 - 9.5, FY1 + 6.5), (mx_ - 3.0, FY1 + 4.0), (mx_ + 3.5, FY1 + 5.0)]
    for i, (tx, ty) in enumerate(outside):
        T.put("tree_plane", tx + RO.uniform(-0.8, 0.8), ty + RO.uniform(-0.8, 0.8), OUT - 0.1, RO.uniform(0, 360), variant=["veteran_broad", "autumn", "pollard"],
              tag="outside", collide="none", seed=60 + i)       # not fixed: the trunk moves clear of the walls it stands behind

    # ---- review cameras ---------------------------------------------------------------------------
    T.spawn = {"pos": [cx, SY0 + 5.0, eye], "yaw": 0.0}
    T.shot("01_approach", (cx + 0.6, arrive, eye), (cx, Y0, 3.4), 60, "arrival: street, gate, clock tower beyond")
    T.shot("02_gate_view", (cx, Y0 + 1.5, eye), (cx, FY1, 6.0), 62, "entering the square: fountain, forecourt steps, hall",
           see=[(fx_, fy_, 0.6, 3.2)])
    T.shot("03_plaza_from_terrace", (X1 + 0.35, st_y + 0.4, TH + eye), (fx_, fy_ + 1.0, 1.6), 64, "from the head of the terrace stair across the whole square")
    T.shot("04_cafe", (X0 + 11.5, Y0 + 2.2, eye), (X0, Y0 + cafe_w * 0.6, 2.4), 58, "cafe terrace and gallery")
    # in the gap between a lamp and a bench of the ring (its south-east one), just outside it: the pair frames the basin
    a5 = math.radians(ring_a0 + (n_ring - 1.5) * 360 / n_ring)
    T.shot("05_fountain_close", (fx_ + (ring_r + 0.6) * math.cos(a5), fy_ + (ring_r + 0.6) * math.sin(a5), eye), (fx_, fy_, 1.5), 55, "fountain, benches, ring lamps")
    T.shot("06_stairs_shops", (X1 - 9.5, st_y - 3.5, eye), (TX1, st_y + 4, TH + 2.5), 60, "terrace stair and the shop row above")
    # on the promenade, looking up it: the shop fronts on the right, the cafe tables on the left, the monument closing the view
    T.shot("07_terrace_shops", (vista[0][0], vista[0][1], TH + eye), (mx_ + 2.5, Y1 - 1.0, TH + 3.4), 56, "along the shop fronts on the terrace",
           see=[(mx_ + k * 0.3 * mon_w, FY1 - 0.5, TH + 0.8, TH + 4.5) for k in (-1, 0, 1)])
    # (far enough back that the court's furniture - chain, planters, lanterns - stands whole above the bottom edge)
    T.shot("08_monument", (mx_ - 1.2, FY1 - 10.5, TH + eye), (mx_, FY1, TH + 2.6), 55, "monument at the terrace end")
    T.shot("09_hall", (cx - 7.5, Y1 - 7.0, eye), (cx + 1.5, FY1, 8.5), 66, "clock tower hall from the square")
    # (at the mouth itself: the two near piers frame the passage, they do not fill the picture)
    T.shot("10_arcade_to_canal", (X0 + 0.4, arc_y + 0.3, eye), (QX0, arc_y, 1.6), 55, "through the arcade towards the canal")
    # on the centre line of the quay, looking up the canal: the water and both banks share the frame
    T.shot("11_canal", (QX0 + 1.9, -CY + 10.5, eye), ((QX0 + CX0) / 2 + 1.5, 6.0, 1.0), 62, "quay, canal, bridge, warehouses")
    # at the far bank's rail upstream of the bridge, looking down the canal: the water, the bridge and the hotel's lit back
    T.shot("12_bridge_back", (CX0 - 0.7, hy + 7.0, eye), (QX1, hy - 6.0, 2.4), 62, "from the far bank back to the hotel / cafe backs",
           see=[(QX1, hy, 6.0, 7.4)])
    T.shot("13_market", (X1 - 10.5, Y1 - 8.6, eye), (X1 - 1.5, st_y + st_w / 2 + 4, 1.4), 58, "market corner under the terrace wall")
    # high over the street end, on the axis side of the gate: the square sits in the middle of the frame
    # (low enough that the horizon, the sky and the far rings of roofs close the top of the frame)
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
        "frame": {"plaza": [X0, Y0, X1, Y1], "terrace_x": [X1, TX1], "fore_y": [Y1, FY1], "canal_x": [CX0, QX0], "street": [SX0, SY0, SX1, SY1],
                  "water_z": WZ, "fountain": [fx_, fy_]},
    }, T


if __name__ == "__main__":
    import json
    import sys
    t = generate(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(json.dumps({"items": len(t["items"]), "missing": t["missing"], "placement": {k: v for k, v in t["placement"].items() if k not in ("stairs", "routes")}}, indent=1))
