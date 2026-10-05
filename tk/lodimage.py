"""tk.lodimage - how a cheaper level LOOKS next to the full model, measured on images, and the choice between
the candidates of a level (modelled / reduced / decimated).

The metric is a point-splat render, not the real renderer: every model is sampled into surface points (a few per
pixel of the distance where the level takes over: 16 m for lod 1, 44 m for lod 2), the points are shaded with the
library material's mean colour, the vertex masks and one fixed sun, culled like the runtime culls (single-sided
unless the material is glass / water / double) and splatted orthographically from a ring of views.  Per candidate:

    sil    1 - |A and B| / |A or B| of the silhouettes                      (mean over the views; sil_max the worst)
    img    mean absolute difference of the blurred sRGB images over the union of both silhouettes
    lum    lit brightness of the candidate / of lod 0 where both cover
    glow   share of the glowing pixels of lod 0 that still glow (None when lod 0 hardly glows; glow_px: the share
           of the lod-0 picture that glows - under GLOW_PX the coverage is not judged)
    tear   share of the lod-0 picture torn open over more than the level's size limit; loose: pieces that touched
           in lod 0 and no longer do
    text   lettering (faces tagged by tk.geo text()), every group drawn alone at a finer scale: see text_measure
    zf     depth fight: glowing area with a parallel face of another material within 3 mm, against lod 0 (zfight;
           the measure of tools/checks/zscan.py)
    ident  smallest share of its lod-0 pixels any material kept (materials with >= 1 % of the picture)
    Local measures (a defect in one place hardly moves a mean over the picture; each is a gate against lod 0 and,
    where the decimated copy is given, only when that copy is clearly better - gates()):
    tile     worst tile of the worst view: mean image difference inside it (TILE; TILE_IDENT times wider for a copy
             carrying tk.lod_standin stand-ins) - a torn dial, a hole
    sil_new  share of the level's silhouette lying outside lod 0's silhouette grown by two pixels, worst view
             (SIL_NEW) - a shard, a filled hoop
    glow_min glow of the view where the least of it is left (GLOW_VIEW) - a lit face gone dark on one side
    leak     share of the directions around a light of the asset in which lod 0 casts a shadow and the level lets
             the light land on something a viewer sees (LEAK; shadow_casts, light_leak) - a missing shadow caster
    uvbad  share of the surface whose texture density is off by more than 2x (reported, never a gate)

    m = Metric("out/x/default.glb", mount="ground")
    a = m.score(1, "out/x/default.lod1.glb"); b = m.score(1, "out/x/default.dec1.glb")
    pick = choose(1, [{"name": "reduced", "kind": "reduced", "hard": [], "tris": 900, "budget": 1000, "m": a},
                      {"name": "decimated", "kind": "decimated", "tris": 1000, "budget": 1000, "m": b}])

Needs numpy only (no bpy), GLB files as written by tk.final (Y-up, _MASK attribute, material extras tsq_lib/tsq_tint).
Probe is the in-memory cousin used by tk.reduce to see whether a reduction step damaged the outline.
"""
import json
import math
import os
import struct
import time
import numpy as np

GLOW = ("emissive", "neon", "window_lit", "glass_stained", "glass_frosted", "glass_lantern")
PLANT = ("moss", "hedge", "leaf")
TWO_SIDED = ("glass_clear", "glass_frosted", "glass_lantern", "water", "water_fall")

DIST = (0.0, 16.0, 44.0)        # metres at which lod 1 / lod 2 take over (web/engine/town.js LOD0_DIST / LOD1_DIST)
PX_PER_M = 0.0008625            # size of a pixel per metre of distance (13.8 mm at 16 m, 38 mm at 44 m)
MAXPX = 400                     # a view is never larger than this on its long side (big assets get coarser pixels)
SPP = 4.0                       # surface samples per pixel ...
SPP_MAX = 12.0                  # ... and up to this many for small models, whose rim pixels count for much
NMAX = 700000                   # samples of lod 0 per level (coarser pixels beyond)
BG = 0.5                        # sRGB grey behind the model
SUN = np.array([-0.42, -0.55, 0.72]) / np.linalg.norm([-0.42, -0.55, 0.72])

# ---- gates (see choose) ----
SIL = (0.0, 0.03, 0.08)         # silhouette difference a level may have
LUM = (0.92, 1.08)              # lit brightness against lod 0
GLOW_MIN = 0.8                  # glow coverage
GLOW_PX = 0.005                 # ... judged only when this share of the lod-0 picture shows a glowing face: below it
                                #     the glow sits behind glass (a globe lamp, a clock dial) and the few pixels are
                                #     where the two fight in lod 0 - lit brightness is the measure there
LUM_WIDE = (0.85, 1.18)         # ... of a reduced level whose decimated copy is inside LUM: allowed when its image
LUM_GAIN = 0.15                 #     difference is lower than the decimated copy's by at least this share

# ---- local defects (round 3b): a mean over the picture and over all views hides what is wrong in one place ----
SEE = ("glass_clear",)          # drawn only where nothing else lies in the pixel: what stands behind clear glass (a
                                # clock dial, a shop display) is seen, as on screen
NOSHADOW = ("emissive", "neon", "glass_", "window_", "water")   # materials that cast no shadow (web/engine/assets.js)
TILE = (0.0, 0.085, 0.11)       # worst tile of the worst view: mean image difference it may have ...
TILE_IDENT = 1.5                # ... times this for a copy that carries the generator's own stand-ins (tk.lod_standin):
                                # a dial with bold hands and no minute ticks is meant to differ where it sits
TILE_DEC = 1.15                 # ... unless the decimated copy's worst tile is no better than this share of it
SIL_NEW = (0.0, 0.012, 0.03)    # pixels outside the lod-0 outline (grown by 2 px), share of the outline, worst view
GLOW_VIEW = 0.85                # glow coverage in the worst view that shows a glowing face over GLOW_VIEW_PX of it
GLOW_VIEW_PX = 0.005
LEAK = (0.0, 0.005, 0.01)       # share of the directions around a light of the asset in which lod 0 casts a shadow
                                # and the level lets a beam out onto something a viewer sees (lod 2 stands beyond
                                # 44 m, where the town gives no lantern a shadow map: web/engine/town.js)
LEAK_EPS = 0.05                 # ... landing this far (m) behind where lod 0 stopped the light
LEAK_DIRS = 3000                # directions around a light
LEAK_LIGHTS = 8                 # lights of an asset that are tested (shadow-flagged first, then the strongest)
LOCAL = ("local defect", "new outline", "a lit face", "light leak")     # how the reasons of these gates start
TEAR = (0.0, 0.003, 0.005)      # share of the lod-0 picture that may be torn open (see Metric.score: "tear"; the torn
                                #     crown of the first tower_keep_ornate lod 2 measured 0.0126, whole levels <= 0.0006)
OPEN = 0.05                     # share of the tiled fields (ashlar, slates: tk.reduce "open") that may lie bare - the
                                #     wall a few cm behind a skin of blocks is too near for the picture to tell
OPEN_LAST = 0.10                # ... and up to this ships when the decimated copy is no fallback (see choose())
DEC_MARGIN = 0.002              # the decimated copy only wins when its image difference is lower by more than this
ALLOW2 = 1.5                    # a reduced lod 2 may use up to this x its budget when it is clean and beats decimated
ALLOW2B = 2.0                   # ... a building / landmark up to this, when nothing cheaper keeps its outline whole
ALLOW_GAIN = 0.15               # ... and over a level inside the budget that passes every gate: only when its image
                                #     difference is lower by at least this share
BIG = ("building", "landmark")  # META["class"] of those
TEXT_PX = 3.0                   # lettering lower than this many pixels at the level's distance is not gated
TEXT_DEV = 0.10                 # share of a line of lettering that may be off (missing or extra, beyond one cell) ...
TEXT_WORST = 0.30               # ... and of its worst letter
TEXT_COV = (0.85, 1.15)         # ... and its covered area against lod 0
# ---- glowing faces that lie on another surface (they fight for the depth buffer; see zfight) ----
Z_DEF = {"emissive": 8, "neon": 14, "glass_frosted": 3, "glass_lantern": 0.5, "glass_stained": 1.6, "window_lit": 2.4}
Z_GLOW = 0.3                    # a material glows from this "glow" on (its tsq_params, else the library default above)
Z_CARD = ("window_lit", "window_dark")      # the room card of a window counts whatever its glow: the dark card of an
                                #   unlit room lies as close to the wall as the lit one and fights it just the same
Z_KEEP = 0.004                  # ... and a card (glowing or dark) keeps this much, less Z_EDGE, to the faces about it:
                                #   "z4", gated like "z3" (the generators give every card 4 mm; Z_EDGE is for rounding)
Z_SEE = ("glass_clear", "glass_lantern", "glass_frosted", "glass_stained", "water", "water_fall")   # see-through partners
Z_BUCKET = (0.0003, 0.0015, 0.003)      # A | B | C: distance of the parallel face of another material; D beyond
Z_FAR = 0.006                   # ... looked for up to here
Z_EDGE = 0.00025                # lod 0 is read this much wider (a gap of exactly 3 mm falls on either side by rounding)
Z_TOL = (0.0005, 0.004, 0.001)  # a level may have this much more card area with a partner under Z_BUCKET[2] than
                                #     lod 0: 0.1 % of the card area, at least 5 cm2 and at most 40 cm2 (what is left of
                                #     edges and corners; the levels are meant to have none that lod 0 has not)
Z_MISS = (1.5, 1.5)             # when the decimated copy fails that gate itself, a reduced one that keeps it ships
                                # with a silhouette up to this x the limit and an image difference up to this x the
                                # decimated copy's (see choose())

_SPECIAL = {"emissive": (1.0, 0.72, 0.42), "neon": (1.0, 0.45, 0.5), "window_lit": (1.0, 0.76, 0.48),
            "glass_stained": (0.75, 0.5, 0.3), "glass_frosted": (0.85, 0.8, 0.7), "glass_lantern": (1.0, 0.8, 0.55),
            "glass_clear": (0.07, 0.08, 0.09), "water": (0.05, 0.08, 0.08), "water_fall": (0.5, 0.56, 0.58)}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MATS = None


def _lin(h):
    h = h.lstrip("#")
    c = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def materials():
    """{library name: {"base": linear rgb, "grime": linear rgb, "tintable", "special"}} from docs/materials.json"""
    global _MATS
    if _MATS is None:
        _MATS = {}
        try:
            with open(os.path.join(_ROOT, "docs", "materials.json")) as f:
                data = json.load(f)["materials"]
        except Exception:  # pragma: no cover
            data = []
        for m in data:
            base = np.array(m.get("preview", [0.5, 0.5, 0.5]), dtype=np.float64)
            g = (m.get("grime") or {}).get("color")
            _MATS[m["name"]] = {"base": base, "grime": _lin(g) if g else base * 0.4, "tintable": bool(m.get("tintable")),
                                "special": m.get("kind") == "special"}
    return _MATS


def base_colour(lib, tint):
    """-> (albedo, grime colour), linear"""
    m = materials().get(lib)
    if lib in _SPECIAL:
        base = np.array(_SPECIAL[lib])
        if tint is not None and lib not in ("glass_clear", "water", "water_fall"):
            base = np.asarray(tint, dtype=np.float64)[:3] * (1.0 if lib.startswith(GLOW) else base)
        return base, base * 0.4
    if m is None:
        return np.array([0.5, 0.5, 0.5]), np.array([0.2, 0.2, 0.2])
    base = m["base"] * (np.asarray(tint, dtype=np.float64)[:3] if (tint is not None and m["tintable"]) else 1.0)
    return base, m["grime"]


# ----------------------------------------------------------------------------
# GLB
# ----------------------------------------------------------------------------

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_GLB = {}


def _node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in n:
        M = np.diag(list(n["scale"]) + [1.0]) @ M
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        R = np.eye(4)
        R[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        M = R @ M
    if "translation" in n:
        T = np.eye(4)
        T[:3, 3] = n["translation"]
        M = T @ M
    return M


def load_glb(path, cache=True):
    """-> [primitive]: {"P" (n,3) z-up metres, "N", "UV" | None, "MASK" (n,4), "I" (m,3), "lib", "tint", "name", "double"}"""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if cache and key in _GLB:
        return _GLB[key]
    with open(path, "rb") as f:
        data = f.read()
    off, js, blob = 12, None, b""
    while off < len(data):
        n, typ = struct.unpack_from("<I4s", data, off)
        off += 8
        if typ == b"JSON":
            js = json.loads(data[off:off + n])
        elif typ[:3] == b"BIN":
            blob = data[off:off + n]
        off += n

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        dt = np.dtype(_CT[a["componentType"]])
        nc = _NC[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or dt.itemsize * nc
        cnt = a["count"]
        if stride == dt.itemsize * nc:
            arr = np.frombuffer(blob, dtype=dt, count=cnt * nc, offset=start).reshape(cnt, nc)
        else:
            raw = np.frombuffer(blob, dtype=np.uint8, count=stride * (cnt - 1) + dt.itemsize * nc, offset=start)
            idx = (np.arange(cnt)[:, None] * stride + np.arange(dt.itemsize * nc)[None, :]).ravel()
            arr = raw[idx].view(dt).reshape(cnt, nc)
        if a.get("normalized") and dt.kind in "ui":
            arr = arr.astype(np.float32) / float(np.iinfo(dt).max)
        return arr

    prims = []
    mats = js.get("materials", [])

    def walk(ni, M):
        n = js["nodes"][ni]
        M = M @ _node_matrix(n)
        if "mesh" in n:
            for pr in js["meshes"][n["mesh"]]["primitives"]:
                at = pr["attributes"]
                if "POSITION" not in at or pr.get("mode", 4) != 4:
                    continue
                p = acc(at["POSITION"]).astype(np.float64)
                p = p @ M[:3, :3].T + M[:3, 3]
                nrm = acc(at["NORMAL"]).astype(np.float64) @ np.linalg.inv(M[:3, :3]) if "NORMAL" in at else None
                idx = acc(pr["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in pr else np.arange(len(p)).reshape(-1, 3)
                m = mats[pr["material"]] if "material" in pr and pr["material"] < len(mats) else {}
                ex = m.get("extras") or {}
                par = ex.get("tsq_params")
                if isinstance(par, str):
                    try:
                        par = json.loads(par)
                    except Exception:
                        par = {}
                lib = str(ex.get("tsq_lib", ""))
                prims.append({
                    "P": np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1),
                    "N": None if nrm is None else np.stack([nrm[:, 0], -nrm[:, 2], nrm[:, 1]], axis=1),
                    "UV": acc(at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else None,
                    "MASK": acc(at["_MASK"]).astype(np.float64) if "_MASK" in at else None,
                    "I": idx, "lib": lib, "tint": ex.get("tsq_tint"), "name": m.get("name", lib),
                    "double": bool((par or {}).get("double")) or lib.startswith(TWO_SIDED),
                    "zglow": z_glows(lib, par), "zdark": z_dark(lib, par)})
        for c in n.get("children", []):
            walk(c, M)

    sc = js["scenes"][js.get("scene", 0)] if js.get("scenes") else {"nodes": list(range(len(js.get("nodes", []))))}
    for ni in sc["nodes"]:
        walk(ni, np.eye(4))
    if cache:
        if len(_GLB) > 24:
            _GLB.clear()
        _GLB[key] = prims
    return prims


def tris_of(prims):
    return int(sum(len(p["I"]) for p in prims))


def bounds_of(prims):
    mn = np.min([p["P"].min(0) for p in prims if len(p["P"])], axis=0)
    mx = np.max([p["P"].max(0) for p in prims if len(p["P"])], axis=0)
    return mn, mx


# ----------------------------------------------------------------------------
# Sampling, views, splatting
# ----------------------------------------------------------------------------

def _points(V, T, px, spp, nmax, seed=7, sc=None):
    """Surface samples of triangles T over vertices V -> (triangle of each sample, barycentric weights (k,3),
    unit face normals (m,3), areas (m,)). The number of samples of a triangle is its area x the density, rounded up
    or down at random: the same surface gets the same coverage however finely it is cut (one sample for every
    triangle, however small, would draw a finely cut model fatter than its reduced copy).
    sc: fixed sampling scale; the samples then follow a fixed pattern per triangle (the same triangle gets the same
    samples whatever else is in T), which is what Probe needs to compare a soup with a part of itself."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    fn = np.cross(b - a, c - a)
    l = np.linalg.norm(fn, axis=1)
    ar = 0.5 * l
    want = ar * (spp / (px * px))
    fixed = sc is not None
    if sc is None:
        sc = _scale(want, len(T), nmax)
    if fixed:
        u = np.abs(np.sin((a + b + c) @ np.array([12.9898, 78.233, 37.719])) * 43758.5453) % 1.0
    else:
        rs = np.random.RandomState(seed)
        u = rs.random_sample(len(T))
    cnt = np.floor(want * sc + u).astype(np.int64)
    cnt[l <= 1e-14] = 0
    tri = np.repeat(np.arange(len(T)), cnt)
    if fixed:
        j = np.arange(len(tri)) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 1
        r = np.stack([(u[tri] + j * 0.7548776662466927) % 1.0, (0.5 + j * 0.5698402909980532) % 1.0], axis=1)
    else:
        r = rs.random_sample((len(tri), 2))
    s = np.sqrt(r[:, 0])
    W = np.stack([1.0 - s, s * (1.0 - r[:, 1]), s * r[:, 1]], axis=1)
    return tri, W, fn / np.maximum(l, 1e-20)[:, None], ar


def _scale(want, ntris, nmax):
    return min(1.0, nmax / max(float(want.sum()) + 0.5 * ntris, 1.0))


def _estimate(prims, px, spp):
    n = 0.0
    for p in prims:
        P, I = p["P"], p["I"]
        if len(I):
            n += 0.5 * np.linalg.norm(np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]), axis=1).sum() * spp / (px * px)
            n += 0.5 * len(I)
    return n


class Cloud:
    """point samples of a model: P positions, C lit linear colour, G glows, L library id, F face normal, two (sided)"""

    def __init__(self, prims, px, libid, spp=SPP, nmax=NMAX):
        Ps, Cs, Gs, Ls, Fs, Ws, Ss = [], [], [], [], [], [], []
        tot = max(_estimate(prims, px, spp), 1.0)
        for i, p in enumerate(prims):
            P, I = p["P"], p["I"]
            if not len(I):
                continue
            share = max(int(nmax * (_estimate([p], px, spp) / tot)), len(I))
            tri, W, fn, _ = _points(P, I, px, spp, share, seed=7 + i)
            if not len(tri):
                continue
            i0, i1, i2 = I[tri, 0], I[tri, 1], I[tri, 2]
            pos = P[i0] * W[:, 0:1] + P[i1] * W[:, 1:2] + P[i2] * W[:, 2:3]
            f = fn[tri]
            if p["N"] is not None:
                N = p["N"]
                n = N[i0] * W[:, 0:1] + N[i1] * W[:, 1:2] + N[i2] * W[:, 2:3]
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            else:
                n = f
            lib = p["lib"]
            base, grime = base_colour(lib, p["tint"])
            glow = lib.startswith(GLOW)
            if glow:
                col = np.broadcast_to(np.minimum(base * 2.2, 1.0), (len(tri), 3)).copy()
            else:
                if p["MASK"] is not None:
                    M = p["MASK"]
                    ao = M[i0, 0] * W[:, 0] + M[i1, 0] * W[:, 1] + M[i2, 0] * W[:, 2]
                    gr = M[i0, 1] * W[:, 0] + M[i1, 1] * W[:, 1] + M[i2, 1] * W[:, 2]
                else:
                    ao, gr = np.ones(len(tri)), np.zeros(len(tri))
                g = (0.7 * gr)[:, None]
                col = (base[None, :] * (1.0 - g) + grime[None, :] * g) * (0.35 + 0.65 * ao)[:, None]
                ns = n @ SUN
                if p["double"]:
                    ns = np.abs(ns)
                col = col * (0.42 + 0.62 * np.maximum(ns, 0.0) + 0.08 * np.maximum(-ns, 0.0))[:, None]
            Ps.append(pos.astype(np.float32))
            Cs.append(col.astype(np.float32))
            Gs.append(np.full(len(tri), glow, dtype=bool))
            Ls.append(np.full(len(tri), libid.setdefault(lib, len(libid)), dtype=np.int16))
            Fs.append(f.astype(np.float32))
            Ws.append(np.full(len(tri), bool(p["double"]), dtype=bool))
            Ss.append(np.full(len(tri), lib.startswith(SEE), dtype=bool))
        if Ps:
            self.P, self.C, self.G = np.vstack(Ps), np.vstack(Cs), np.concatenate(Gs)
            self.L, self.F, self.two = np.concatenate(Ls), np.vstack(Fs), np.concatenate(Ws)
            self.S = np.concatenate(Ss)
        else:
            self.P = self.C = self.F = np.zeros((0, 3), dtype=np.float32)
            self.G = self.two = self.S = np.zeros(0, dtype=bool)
            self.L = np.zeros(0, dtype=np.int16)


class View:
    """orthographic view: d looks into the scene, r right, u up; the window is the lod-0 bounds plus a margin"""

    def __init__(self, az, el, mn, mx, px):
        a, e = math.radians(az), math.radians(el)
        self.d = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), -math.sin(e)], dtype=np.float32)
        self.r = np.array([math.cos(a), -math.sin(a), 0.0], dtype=np.float32)
        self.u = np.cross(self.r, self.d).astype(np.float32)
        c = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        us, vs = c @ self.r, c @ self.u
        pad = 4 * px + 0.06 * max(us.max() - us.min(), vs.max() - vs.min())
        self.u0, self.v0 = float(us.min() - pad), float(vs.min() - pad)
        self.W = max(int(math.ceil((us.max() - us.min() + 2 * pad) / px)), 4)
        self.H = max(int(math.ceil((vs.max() - vs.min() + 2 * pad) / px)), 4)
        self.px = px


def views_for(mount, mn, mx):
    """[(azimuth, elevation)]: a ring at eye level plus a few from above; wall-mounted things from their front only"""
    if mount in ("wall", "wall_ground"):
        front = 180.0 if (mx[1] + mn[1]) > 0 else 0.0
        az = [front + a for a in (-70.0, -35.0, 0.0, 35.0, 70.0)]
    else:
        az = [45.0 * i for i in range(8)]
    return [(a, 8.0) for a in az] + [(a + 20.0, 32.0) for a in az[::2]]


def _zwin(P, F, two, vw, see=None):
    """nearest sample per pixel -> (win (H*W,) sample index or -1, depth of every sample, samples outside the window)
    see: (n,) bool, samples of clear glass - one wins a pixel only where no other sample lies in it"""
    z = P @ vw.d
    x = np.floor((P @ vw.r - vw.u0) / vw.px).astype(np.int32)
    y = np.floor((P @ vw.u - vw.v0) / vw.px).astype(np.int32)
    ok = (x >= 0) & (x < vw.W) & (y >= 0) & (y < vw.H)
    out = int((~ok).sum())
    ok &= two | ((F @ vw.d) < 0.0)
    idx = np.nonzero(ok)[0]
    pix = y[idx] * vw.W + x[idx]
    order = np.argsort(-z[idx])
    win = np.full(vw.W * vw.H, -1, dtype=np.int32)
    win[pix[order]] = idx[order]
    if see is not None and len(idx) and see[idx].any():
        k = ~see[idx]
        i2, p2 = idx[k], pix[k]
        o2 = np.argsort(-z[i2])
        w2 = np.full(vw.W * vw.H, -1, dtype=np.int32)
        w2[p2[o2]] = i2[o2]
        win = np.where(w2 >= 0, w2, win)
    return win, z, out


def _pad_sum(a):
    """3x3 box sum (zero outside)"""
    p = np.zeros((a.shape[0] + 2, a.shape[1] + 2) + a.shape[2:], dtype=np.float32)
    p[1:-1, 1:-1] = a
    H, W = a.shape[0], a.shape[1]
    out = np.zeros(a.shape, dtype=np.float32)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out += p[dy:dy + H, dx:dx + W]
    return out


def dilate(m):
    return _pad_sum(m.astype(np.float32)) > 0.5


def erode(m):
    p = np.ones((m.shape[0] + 2, m.shape[1] + 2), dtype=np.float32)
    p[1:-1, 1:-1] = m
    H, W = m.shape
    out = np.zeros(m.shape, dtype=np.float32)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out += p[dy:dy + H, dx:dx + W]
    return out > 8.5


def size_limit(level, diag):
    """What counts as small at a level (m): loose pieces under it may go, a hole under it is no tear. Grows with the
    asset (the same rule as the smallest island tk.final keeps in its decimated copies)."""
    k = float(np.clip(diag / 6.0, 0.6, 2.5))
    return (0.0, 0.035 * k, 0.14 * k)[level]


def _components(nv, ev):
    """vertex -> smallest vertex id of its connected component (ev: (m,2) edges)"""
    lab = np.arange(nv, dtype=np.int64)
    if len(ev) == 0:
        return lab
    src = np.concatenate([ev[:, 0], ev[:, 1]]).astype(np.int64)
    dst = np.concatenate([ev[:, 1], ev[:, 0]]).astype(np.int64)
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o]
    st = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
    us = src[st]
    while True:
        new = lab.copy()
        new[us] = np.minimum(lab[us], np.minimum.reduceat(lab[dst], st))
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        if np.array_equal(new, lab):
            return lab
        lab = new


def _shade(cl, vw):
    """one view of a cloud -> dict(c closed coverage, b blurred sRGB image, lum linear luminance, g glow, lib, out,
    cov raw coverage, z depth of the nearest sample)"""
    win, _z, out = _zwin(cl.P, cl.F, cl.two, vw, cl.S)
    H, W = vw.H, vw.W
    cov = (win >= 0).reshape(H, W)
    w = np.maximum(win, 0)
    col = np.where(cov[..., None], cl.C[w].reshape(H, W, 3), 0.0).astype(np.float32)
    closed = erode(dilate(cov))
    hole = closed & ~cov
    if hole.any():
        cnt = _pad_sum(cov.astype(np.float32))
        fill = _pad_sum(col) / np.maximum(cnt, 1.0)[..., None]
        col = np.where(hole[..., None], fill, col)
    c = cov | hole
    lum = (col @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)) * c
    im = np.where(c[..., None], np.clip(col, 0.0, 1.0) ** (1.0 / 2.2), np.float32(BG)).astype(np.float32)
    p = np.full((H + 2, W + 2, 3), BG, dtype=np.float32)
    p[1:-1, 1:-1] = im
    b = np.zeros((H, W, 3), dtype=np.float32)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            b += p[dy:dy + H, dx:dx + W]
    b /= 9.0
    g = cov & cl.G[w].reshape(H, W)
    lib = np.where(cov, cl.L[w].reshape(H, W), -1)
    return {"c": c, "b": b, "lum": lum, "g": g, "lib": lib, "out": out, "cov": cov,
            "z": np.where(cov, _z[w].reshape(H, W), np.float32(1e9))}


# ----------------------------------------------------------------------------
# Connectivity: what touched in lod 0 still touches
# ----------------------------------------------------------------------------

VOX_MAX = 250000                # voxels of lod 0 at most (the cell grows beyond)
_VM = 1 << 20
_VOFF = [dx + dy * _VM + dz * _VM * _VM for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
         if (dz, dy, dx) != (0, 0, 0)]


def vox_cell(level, diag, prims):
    """the cell of the touch test (m): half the level's size limit, at least 1.5 px at its distance"""
    area = 0.0
    for p in prims:
        P, I = p["P"], p["I"]
        if len(I):
            area += float(0.5 * np.linalg.norm(np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]), axis=1).sum())
    return max(0.5 * size_limit(level, diag), 1.5 * DIST[level] * PX_PER_M, math.sqrt(area / VOX_MAX))


def _vox_samples(P, I, c, org):
    """the cells the triangles pass through (sampled every c / 2) -> yields (triangle indices, keys (len, points))"""
    P = P.astype(np.float64)
    A, B, C = P[I[:, 0]], P[I[:, 1]], P[I[:, 2]]
    e = np.maximum(np.maximum(np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1)), np.linalg.norm(A - C, axis=1))
    n = np.clip(np.ceil(e / (0.5 * c)), 1, 4000).astype(np.int64)
    for q in np.unique(n):
        ii, jj = np.meshgrid(np.arange(q + 1), np.arange(q + 1), indexing="ij")
        ok = ii + jj <= q
        u, v = ii[ok] / q, jj[ok] / q
        wa = np.flatnonzero(n == q)
        step = max(1, 1500000 // len(u))
        for s0 in range(0, len(wa), step):
            w = wa[s0:s0 + step]
            pts = (A[w][:, None, :] * (1.0 - u - v)[None, :, None] + B[w][:, None, :] * u[None, :, None]
                   + C[w][:, None, :] * v[None, :, None])
            g = np.clip(np.floor((pts - org) / c).astype(np.int64), 0, _VM - 2)
            yield w, g[:, :, 0] + g[:, :, 1] * _VM + g[:, :, 2] * _VM * _VM


def voxels(prims, c, mn):
    """The cells of size c the surface passes through (plants left out) -> sorted keys, the connected group of
    each (cells that share a face, an edge or a corner are connected: surfaces that touch, or come closer than c,
    are one group; surfaces 2 c apart never are)."""
    keys = []
    org = np.asarray(mn, dtype=np.float64) - 2.0 * c
    for p in prims:
        if not len(p["I"]) or p["lib"].startswith(PLANT):
            continue
        for _w, k in _vox_samples(p["P"], p["I"], c, org):
            keys.append(np.unique(k))
    if not keys:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    K = np.unique(np.concatenate(keys))
    ev = []
    for off in _VOFF[13:]:
        pos = np.minimum(np.searchsorted(K, K + off), len(K) - 1)
        hit = np.flatnonzero(K[pos] == K + off)
        ev.append(np.stack([hit, pos[hit]], axis=1))
    return K, _components(len(K), np.concatenate(ev) if ev else np.zeros((0, 2), dtype=np.int64))


def _in(K, q):
    """q in the sorted keys K -> bool mask"""
    if not len(K):
        return np.zeros(len(q), dtype=bool)
    pos = np.minimum(np.searchsorted(K, q), len(K) - 1)
    return K[pos] == q


def tris_in_cells(P, I, c, mn, cells):
    """the triangles that pass through one of the cells (sorted keys of voxels(..., c, mn)) -> bool mask"""
    out = np.zeros(len(I), dtype=bool)
    if not len(I) or not len(cells):
        return out
    org = np.asarray(mn, dtype=np.float64) - 2.0 * c
    xyz = np.stack([cells % _VM, (cells // _VM) % _VM, cells // (_VM * _VM)], axis=1)
    lo, hi = org + xyz.min(axis=0) * c, org + (xyz.max(axis=0) + 1) * c
    T = P[I]
    near = np.flatnonzero(((T.max(axis=1) >= lo) & (T.min(axis=1) <= hi)).all(axis=1))
    if not len(near):
        return out
    for w, k in _vox_samples(P, I[near], c, org):
        out[near[w]] = _in(cells, k.ravel()).reshape(k.shape).any(axis=1)
    return out


def _loose_groups(K0, L0, K1, L1):
    """-> group of every cell of the level (0..n-1), cells per group, cells along the longest side of every group, their
    low and high corners (cells), the loose groups (largest first within their group of lod 0)"""
    l0 = np.full(len(K1), -1, dtype=np.int64)
    for off in [0] + _VOFF:
        w = np.flatnonzero(l0 < 0)
        if not len(w):
            break
        pos = np.minimum(np.searchsorted(K0, K1[w] + off), len(K0) - 1)
        hit = K0[pos] == K1[w] + off
        l0[w[hit]] = L0[pos[hit]]
    u1, c1 = np.unique(L1, return_inverse=True)
    n1 = len(u1)
    cnt = np.bincount(c1, minlength=n1)
    ok = l0 >= 0
    home = np.full(n1, -1, dtype=np.int64)
    if ok.any():
        u0, c0 = np.unique(l0[ok], return_inverse=True)
        pair = c1[ok] * len(u0) + c0
        up, pc = np.unique(pair, return_counts=True)
        o = np.lexsort((pc, up // len(u0)))             # per piece the most frequent group of lod 0 last
        last = np.r_[(up[o] // len(u0))[1:] != (up[o] // len(u0))[:-1], True]
        home[(up[o] // len(u0))[last]] = (up[o] % len(u0))[last]
    xyz = np.stack([K1 % _VM, (K1 // _VM) % _VM, K1 // (_VM * _VM)], axis=1)
    o = np.argsort(c1, kind="stable")
    st = np.flatnonzero(np.r_[True, c1[o][1:] != c1[o][:-1]])
    lo_ = np.minimum.reduceat(xyz[o], st, axis=0)
    hi_ = np.maximum.reduceat(xyz[o], st, axis=0)
    ext = (hi_ - lo_ + 1).max(axis=1)
    out = []
    for h in np.unique(home[home >= 0]):
        g = np.flatnonzero(home == h)
        if len(g) < 2:
            continue
        g = g[np.argsort(-cnt[g], kind="stable")][1:]
        out.extend(g[ext[g] >= 2].tolist())
    return c1, cnt, ext, lo_, hi_, out


def loose_pieces(K0, L0, K1, L1):
    """the loose pieces of loose() -> [indices into K1 of the cells of one piece, ...], largest first"""
    if not len(K0) or not len(K1):
        return []
    c1, cnt, ext, lo_, hi_, g = _loose_groups(K0, L0, K1, L1)
    g = sorted(g, key=lambda i: -int(cnt[i]))
    return [np.flatnonzero(c1 == i) for i in g]


def bridge(M, K1, piece, maxd=64):
    """The shortest way from a loose piece (indices into K1) to the rest of the level through the cells M that lod 0
    fills and the level does not (sorted keys) -> the keys of the cells on it, None when there is none"""
    if not len(M):
        return None
    pk = K1[piece]
    rest = np.ones(len(K1), dtype=bool)
    rest[piece] = False
    other = K1[rest]
    if not len(other):
        return None

    def around(keys):
        return np.unique(np.concatenate([keys + o for o in _VOFF]))

    def in_m(keys):
        pos = np.minimum(np.searchsorted(M, keys), len(M) - 1)
        return pos[M[pos] == keys]

    dist = np.full(len(M), -1, dtype=np.int64)
    f = in_m(around(pk))
    dist[f] = 0
    d = 0
    while len(f) and d <= maxd:
        ck = M[f]
        t = np.zeros(len(f), dtype=bool)
        for o in _VOFF:
            t |= _in(other, ck + o)
        if t.any():
            cur = int(f[t][0])
            path = [cur]
            for dd in range(d - 1, -1, -1):
                nb = in_m(M[cur] + np.array(_VOFF, dtype=np.int64))
                nb = nb[dist[nb] == dd]
                if not len(nb):
                    break
                cur = int(nb[0])
                path.append(cur)
            return M[np.array(path, dtype=np.int64)]
        n = in_m(around(ck))
        n = n[dist[n] < 0]
        d += 1
        dist[n] = d
        f = n
    return None


def loose(K0, L0, K1, L1, c):
    """Pieces of a level that hang loose where lod 0 was one: its connected groups are matched to the groups of lod
    0 they lie in; of several in one, all but the largest are loose (those at least two cells long count).
    -> number of loose pieces, their share of the level's cells, the size of the largest (m), its centre (cells)"""
    if not len(K0) or not len(K1):
        return 0, 0.0, 0.0, None
    c1, cnt, ext, lo_, hi_, g = _loose_groups(K0, L0, K1, L1)
    if not g:
        return 0, 0.0, 0.0, None
    g = np.array(g, dtype=np.int64)
    i = g[int(np.argmax(ext[g]))]
    return int(len(g)), float(cnt[g].sum()) / float(len(K1)), round(float(ext[i]) * c, 3), 0.5 * (lo_[i] + hi_[i] + 1)


# ----------------------------------------------------------------------------
# Texture density
# ----------------------------------------------------------------------------

def _uv_density(p):
    P, I, UV = p["P"], p["I"], p["UV"]
    a = 0.5 * np.linalg.norm(np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]), axis=1)
    if UV is None:
        return a, np.zeros(len(I))
    e1, e2 = UV[I[:, 1]] - UV[I[:, 0]], UV[I[:, 2]] - UV[I[:, 0]]
    ua = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return a, np.sqrt(ua / np.maximum(a, 1e-14))


def _wmedian(v, w):
    if not len(v):
        return 0.0
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(c, 0.5 * c[-1]), len(v) - 1)])


def _uv_exempt(lib):
    m = materials().get(lib)
    return lib.startswith(GLOW) or lib.startswith(PLANT) or lib in _SPECIAL or (m is not None and m["special"]) or not lib


def uv_reference(prims):
    """{material name | ("lib", library): area-weighted median texture density} of lod 0"""
    by, bylib = {}, {}
    for p in prims:
        if _uv_exempt(p["lib"]) or p["UV"] is None:
            continue
        a, d = _uv_density(p)
        by.setdefault(p["name"], []).append((d, a))
        bylib.setdefault(p["lib"], []).append((d, a))
    out = {}
    for src, pre in ((by, None), (bylib, "lib")):
        for k, lst in src.items():
            d = np.concatenate([x[0] for x in lst])
            a = np.concatenate([x[1] for x in lst])
            out[k if pre is None else (pre, k)] = _wmedian(d, a)
    return out


def uv_bad(prims, uref):
    """share of the surface whose texture density is off by more than 2x from lod 0's for that material"""
    bad = tot = 0.0
    for p in prims:
        if _uv_exempt(p["lib"]):
            continue
        a, d = _uv_density(p)
        tot += float(a.sum())
        r = uref.get(p["name"], uref.get(("lib", p["lib"])))
        if r is None or r <= 1e-9:
            continue
        q = d / r
        bad += float(a[(q > 2.0) | (q < 0.5)].sum())
    return bad / tot if tot > 0 else 0.0


# ----------------------------------------------------------------------------
# Glowing faces lying on another surface
# ----------------------------------------------------------------------------

def z_glows(lib, par):
    """does a material count as a card for the depth-fight measure: it glows (the rule of tools/checks/zscan.py), or
    it is the room card of a window, lit or dark (Z_CARD; zscan.py does not count the dark ones)"""
    if lib in Z_CARD:
        return True
    try:
        return float((par or {}).get("glow", Z_DEF.get(lib, 0))) >= Z_GLOW
    except (TypeError, ValueError, AttributeError):
        return Z_DEF.get(lib, 0) >= Z_GLOW


def z_dark(lib, par):
    """is a material the card of an unlit room: of Z_CARD, and counted for that alone"""
    if lib not in Z_CARD:
        return False
    try:
        return float((par or {}).get("glow", Z_DEF.get(lib, 0))) < Z_GLOW
    except (TypeError, ValueError, AttributeError):
        return Z_DEF.get(lib, 0) < Z_GLOW


def z_rays(V, T, mat, glow, far=Z_FAR, bvh=None):
    """The measure of tools/checks/zscan.py on a triangle soup. From every glowing triangle (its centre; from 0.004 m2
    on also three points towards its corners) a ray goes out along the normal and one against it; it counts when the
    first thing it meets within `far` is a parallel face (|cos| >= 0.9) of another material.
    V (n, 3), T (m, 3), mat (m,) material number of every triangle, glow (m,) bool.
    -> (glowing triangle, triangle met, distance, area the ray stands for, +1 in front | -1 behind): arrays, one row
       per such ray. None without mathutils (outside Blender)."""
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        return None
    e = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64))
    if not len(T) or not np.any(glow):
        return e
    V = np.asarray(V, dtype=np.float64)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    area = 0.5 * ln
    n = n / np.maximum(ln, 1e-30)[:, None]
    gi = np.flatnonzero(np.asarray(glow, dtype=bool) & (area >= 1e-5))
    if not len(gi):
        return e
    ctr = (a[gi] + b[gi] + c[gi]) / 3.0
    big = area[gi] >= 0.004
    pts = [ctr]
    own = [gi]
    for q in (a, b, c):
        pts.append(ctr[big] * 0.4 + q[gi][big] * 0.6)
        own.append(gi[big])
    pts, own = np.vstack(pts), np.concatenate(own)
    wa = (area[gi] / np.where(big, 4.0, 1.0))
    wa = np.concatenate([wa] + [wa[big]] * 3)
    if bvh is None:
        bvh = BVHTree.FromPolygons(V.tolist(), np.asarray(T).tolist(), all_triangles=True, epsilon=0.0)
    rc = bvh.ray_cast
    G, J, D, W, S = [], [], [], [], []
    nn = n[own]
    for sgn in (1, -1):
        d = nn * sgn
        o = pts + d * 2e-5
        for k, (o_, d_) in enumerate(zip(o.tolist(), d.tolist())):
            h = rc(o_, d_, far)
            if h[0] is None:
                continue
            G.append(k)
            J.append(h[2])
            D.append(h[3])
            S.append(sgn)
    if not G:
        return e
    G, J, D, S = np.array(G), np.array(J, dtype=np.int64), np.array(D) + 2e-5, np.array(S, dtype=np.int64)
    i = own[G]
    ok = (J != i) & (mat[J] != mat[i]) & (np.abs((n[J] * n[i]).sum(axis=1)) >= 0.9)
    return i[ok], J[ok], D[ok], wa[G][ok], S[ok]


def _clip(P, D):
    """the part of the polygon P (list of 3-tuples) where D (a number per corner, linear over it) is <= 0"""
    out = []
    k = len(P)
    for a in range(k):
        b = a + 1 if a + 1 < k else 0
        da, db = D[a], D[b]
        if da <= 0.0:
            out.append(P[a])
        if (da < 0.0 < db) or (db < 0.0 < da):
            t = da / (da - db)
            pa, pb = P[a], P[b]
            out.append((pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2])))
    return out


def z_pairs(V, T, mat, card, V2=None, T2=None, mat2=None, reach=Z_KEEP - Z_EDGE, bvh2=None):
    """What z_rays samples, measured: for every card triangle (V, T, mat, the mask `card`) and every triangle of
    another material that is parallel to it (|cos| >= 0.9) - of the soup V2, T2, mat2, default the same one - the
    area of the card over which that triangle lies within Z_BUCKET[2] / within `reach` of it along the card's
    normal. No sample falls between two rays here: a bar that crosses a pane at a slant is found however thin the
    strip. (Unlike a ray it does not stop at the first face: two partners over the same spot count twice.)
    -> (card triangle, triangle met, least distance, area under Z_BUCKET[2], area under reach, +1 in front of the
        card | -1 behind): arrays, one row per pair. None without mathutils (outside Blender)."""
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        return None
    e = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0),
         np.zeros(0, dtype=np.int64))
    same = V2 is None
    if same:
        V2, T2, mat2 = V, T, mat
    card = np.asarray(card, dtype=bool)
    if not len(T) or not len(T2) or not card.any():
        return e
    V = np.asarray(V, dtype=np.float64)
    V2 = V if same else np.asarray(V2, dtype=np.float64)
    T = np.asarray(T)
    T2 = np.asarray(T2)
    ci = np.flatnonzero(card)
    A, B, C = V[T[ci, 0]], V[T[ci, 1]], V[T[ci, 2]]
    n = np.cross(B - A, C - A)
    ln = np.linalg.norm(n, axis=1)
    ok = 0.5 * ln >= 1e-5
    if not ok.any():
        return e
    ci, A, B, C = ci[ok], A[ok], B[ok], C[ok]
    n = n[ok] / ln[ok][:, None]
    k = len(ci)
    h = 1.02 * reach
    # every card as a slab: what cuts its skin or its middle is a candidate ...
    up, dn = n * h, -n * h
    PV = np.vstack([A + up, B + up, C + up, A + dn, B + dn, C + dn, A, B, C])
    ix = np.arange(k)
    a1, b1, c1, a0, b0, c0, am, bm, cm = (ix + q * k for q in range(9))
    PT = np.vstack([np.stack(x, axis=1) for x in ((a1, b1, c1), (a0, b0, c0), (am, bm, cm), (a1, b1, b0), (a1, b0, a0),
                                                  (b1, c1, c0), (b1, c0, b0), (c1, a1, a0), (c1, a0, c0))])
    slab = BVHTree.FromPolygons(PV.tolist(), PT.tolist(), all_triangles=True, epsilon=0.0)
    if bvh2 is None:
        bvh2 = BVHTree.FromPolygons(V2.tolist(), T2.tolist(), all_triangles=True, epsilon=0.0)
    ov = slab.overlap(bvh2)
    I = [np.array([x[0] for x in ov], dtype=np.int64) % k] if ov else []
    J = [np.array([x[1] for x in ov], dtype=np.int64)] if ov else []
    # ... and what lies wholly inside one: its centre is within reach of the card
    P2 = V2[T2]
    ctr = P2.mean(axis=1)
    lo, hi = np.minimum(np.minimum(A, B), C).min(axis=0) - h, np.maximum(np.maximum(A, B), C).max(axis=0) + h
    inb = np.flatnonzero((ctr >= lo).all(axis=1) & (ctr <= hi).all(axis=1))
    if len(inb):
        cards = BVHTree.FromPolygons(PV[6 * k:].tolist(), np.stack([ix, ix + k, ix + 2 * k], axis=1).tolist(),
                                     all_triangles=True, epsilon=0.0)
        fr = cards.find_nearest_range
        i2, j2 = [], []
        for j, c in zip(inb.tolist(), ctr[inb].tolist()):
            for hit in fr(c, h):
                i2.append(hit[2])
                j2.append(j)
        if i2:
            I.append(np.array(i2, dtype=np.int64))
            J.append(np.array(j2, dtype=np.int64))
    if not I:
        return e
    I, J = np.concatenate(I), np.concatenate(J)
    key = np.unique(I * len(T2) + J)
    I, J = key // len(T2), key % len(T2)
    okp = np.asarray(mat)[ci[I]] != np.asarray(mat2)[J]
    I, J = I[okp], J[okp]
    if not len(I):
        return e
    Q = P2[J]
    n2 = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
    l2 = np.linalg.norm(n2, axis=1)
    sd = ((Q - A[I][:, None, :]) * n[I][:, None, :]).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        okp = (l2 > 1e-12) & (np.abs((n2 * n[I]).sum(axis=1)) >= 0.9 * l2)
    okp &= (sd.min(axis=1) < reach) & (sd.max(axis=1) > -reach)
    I, J, Q, sd = I[okp], J[okp], Q[okp], sd[okp]
    if not len(I):
        return e
    r3 = Z_BUCKET[2]
    oi, oj, od, o3, o4, osd = [], [], [], [], [], []
    for q, (i, j) in enumerate(zip(I.tolist(), J.tolist())):
        a, b, c, nn = A[i], B[i], C[i], n[i]
        ax, ay, az = a.tolist()
        nx, ny, nz = nn.tolist()
        poly = [tuple(x) for x in Q[q].tolist()]
        for p0, p1 in ((a, b), (b, c), (c, a)):
            ex, ey, ez = np.cross(nn, p1 - p0).tolist()             # into the card
            px, py, pz = p0.tolist()
            poly = _clip(poly, [-((x - px) * ex + (y - py) * ey + (z - pz) * ez) for x, y, z in poly])
            if len(poly) < 3:
                break
        if len(poly) < 3:
            continue
        ar = []
        dm = None
        for r in (reach, r3):
            pl = poly
            for sg in (1.0, -1.0):
                pl = _clip(pl, [sg * ((x - ax) * nx + (y - ay) * ny + (z - az) * nz) - r for x, y, z in pl])
                if len(pl) < 3:
                    break
            if len(pl) < 3:
                ar.append(0.0)
                continue
            sx = sy = sz = 0.0
            x0, y0, z0 = pl[0]
            for u in range(1, len(pl) - 1):
                x1, y1, z1 = pl[u][0] - x0, pl[u][1] - y0, pl[u][2] - z0
                x2, y2, z2 = pl[u + 1][0] - x0, pl[u + 1][1] - y0, pl[u + 1][2] - z0
                sx += y1 * z2 - z1 * y2
                sy += z1 * x2 - x1 * z2
                sz += x1 * y2 - y1 * x2
            ar.append(0.5 * abs(sx * nx + sy * ny + sz * nz))
            if dm is None:
                ds = [(x - ax) * nx + (y - ay) * ny + (z - az) * nz for x, y, z in pl]
                dm = 0.0 if min(ds) < 0.0 < max(ds) else min(abs(v) for v in ds)
        if ar[0] < 1e-7:
            continue
        oi.append(int(ci[i]))
        oj.append(j)
        od.append(dm)
        o4.append(ar[0])
        o3.append(ar[1])
        osd.append(1 if float(sd[q].mean()) >= 0.0 else -1)
    if not oi:
        return e
    return (np.array(oi, dtype=np.int64), np.array(oj, dtype=np.int64), np.array(od), np.array(o3), np.array(o4),
            np.array(osd, dtype=np.int64))


def zfight(prims):
    """Area of the cards of a model (load_glb; z_glows: what glows, and the room cards of windows lit or dark) that
    has a parallel face of another material closer than each of Z_BUCKET
    -> {"a", "b", "c", "d": m2 per bucket, "z3": a + b + c, "glass": the share of z3 whose partner is see-through,
        "z3w": the area under Z_BUCKET[2] + Z_EDGE, "z4": the area under Z_KEEP - Z_EDGE, "glow_area": area of all
        cards, "dark_area": of the cards of unlit rooms among them, "dark4": their share of z4,
        "x3", "x4", "xdark4": as z3, z4 and dark4 but measured, not sampled (z_pairs: the area over which a parallel
        face of another material lies within 3 mm / within Z_KEEP - Z_EDGE of the card)}; None outside Blender."""
    names, Vs, Ts, Ms, Gs, Ss, Ds, off = {}, [], [], [], [], [], [], 0
    for p in prims:
        if not len(p["I"]):
            continue
        k = names.setdefault(p["name"], len(names))
        Vs.append(p["P"])
        Ts.append(p["I"] + off)
        Ms.append(np.full(len(p["I"]), k, dtype=np.int64))
        Gs.append(np.full(len(p["I"]), bool(p.get("zglow")), dtype=bool))
        Ss.append(np.full(len(p["I"]), p["lib"] in Z_SEE, dtype=bool))
        Ds.append(np.full(len(p["I"]), bool(p.get("zdark")), dtype=bool))
        off += len(p["P"])
    out = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "z3": 0.0, "glass": 0.0, "z3w": 0.0, "z4": 0.0, "glow_area": 0.0,
           "dark_area": 0.0, "dark4": 0.0, "x3": 0.0, "x4": 0.0, "xdark4": 0.0}
    if not Vs:
        return out
    V, T, M, G, SEE = np.vstack(Vs), np.vstack(Ts), np.concatenate(Ms), np.concatenate(Gs), np.concatenate(Ss)
    DARK = np.concatenate(Ds)
    if not G.any():
        return out
    r = z_rays(V, T, M, G)
    if r is None:
        return None
    a, b, c = V[T[G, 0]], V[T[G, 1]], V[T[G, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    out["glow_area"] = round(float(ar.sum()), 4)
    out["dark_area"] = round(float(ar[DARK[G]].sum()), 4)
    i, j, d, w, sg = r
    bk = np.searchsorted(np.array(Z_BUCKET), d, side="right")
    for k, nm in enumerate("abcd"):
        out[nm] = round(float(w[bk == k].sum()), 4)
    out["z3"] = round(float(w[bk < 3].sum()), 4)
    out["glass"] = round(float(w[(bk < 3) & SEE[j]].sum()), 4)
    out["z3w"] = round(float(w[d < Z_BUCKET[2] + Z_EDGE].sum()), 4)
    out["z4"] = round(float(w[d < Z_KEEP - Z_EDGE].sum()), 4)
    out["dark4"] = round(float(w[(d < Z_KEEP - Z_EDGE) & DARK[i]].sum()), 4)
    xi, xj, xd, x3, x4, xs = z_pairs(V, T, M, G)
    out["x3"] = round(float(x3.sum()), 4)
    out["x4"] = round(float(x4.sum()), 4)
    out["xdark4"] = round(float(x4[DARK[xi]].sum()), 4)
    return out


def z_tol(glow_area):
    """how much more glowing area with a partner under 3 mm than lod 0 a level may have"""
    return float(np.clip(Z_TOL[2] * glow_area, Z_TOL[0], Z_TOL[1]))


# ----------------------------------------------------------------------------
# Local defects: the worst tile, and the shadows of the asset's own lights
# ----------------------------------------------------------------------------

def runs(a):
    """per pixel of the mask `a` the length of the run of set pixels it lies in along its row (0 where not set)"""
    h, w = a.shape
    z = np.zeros((h, 1), dtype=np.int8)
    d = np.diff(np.concatenate([z, a.astype(np.int8), z], axis=1), axis=1)
    ys, xs = np.nonzero(d == 1)
    ye, xe = np.nonzero(d == -1)
    add = np.zeros((h, w + 1), dtype=np.int32)
    add[ys, xs] += xe - xs
    add[ye, xe] -= xe - xs
    return np.cumsum(add, axis=1)[:, :w]


def slits(t, n):
    """the pixels of the mask `t` that lie in a run of at least n pixels along a row or a column"""
    if not t.any():
        return t
    return t & ((runs(t) >= n) | (runs(t.T).T >= n))


def tile_max(d, U):
    """The worst tile of a difference picture d (H, W) over the pixels U: square tiles of an eighth of the picture's
    long side (8 px at least), half a tile apart; the mean of d over the tile's pixels in U, a tile counting as a
    quarter covered at least (a few rim pixels are no tile)."""
    H, W = d.shape
    ts = max(8, int(round(max(H, W) / 8.0)))
    st = max(1, ts // 2)

    def integ(a):
        c = np.zeros((H + 1, W + 1), dtype=np.float64)
        c[1:, 1:] = np.cumsum(np.cumsum(a, axis=0, dtype=np.float64), axis=1)
        return c

    S, N = integ(np.where(U, d, 0.0)), integ(U.astype(np.float64))
    ys = np.array(sorted(set(range(0, max(H - ts, 0) + 1, st)) | {max(H - ts, 0)}))
    xs = np.array(sorted(set(range(0, max(W - ts, 0) + 1, st)) | {max(W - ts, 0)}))
    y1, x1 = np.minimum(ys + ts, H), np.minimum(xs + ts, W)

    def box(c):
        return c[y1][:, x1] - c[ys][:, x1] - c[y1][:, xs] + c[ys][:, xs]

    return float((box(S) / np.maximum(box(N), 0.25 * ts * ts)).max())


def lights_of(sockets, n=LEAK_LIGHTS):
    """The light sockets of an asset (sidecar "sockets") -> [(name, (x, y, z), range, shadow)], n at
    most: those flagged to cast shadows (shadow True) first, then the strongest. (In the town any light may be given
    a shadow map: web/engine/town.js.)"""
    L = []
    for s in sockets or []:
        pos = s.get("pos")
        if s.get("kind") != "light" or not pos or len(pos) < 3:
            continue
        pr = s.get("props") or {}
        L.append((0 if pr.get("shadow") else 1, -float(pr.get("intensity") or 0.0), len(L), str(s.get("name")),
                  tuple(float(x) for x in pos[:3]), float(pr.get("range") or 0.0) or 10.0))
    L.sort()
    return [(x[3], x[4], x[5], x[0] == 0) for x in L[:n]]


_DIRS = {}


def _light_dirs(n=LEAK_DIRS):
    """n directions spread evenly over the sphere, and for each the 13 nearest (itself first among them)"""
    if n not in _DIRS:
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        r = np.sqrt(1.0 - z * z)
        phi = i * math.pi * (3.0 - math.sqrt(5.0))
        D = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
        _DIRS[n] = (D, np.argpartition(-(D @ D.T), 13, axis=1)[:, :13])
    return _DIRS[n]


def shadow_casts(prims, lights, n=LEAK_DIRS, ref=None, mount="ground", front=1.0):
    """The shadows of the asset's lights -> (stopped (len(lights), n) bool, dist (len(lights), n), leak or None):
    does the light meet a shadow caster within its range in each of n directions, and how far away. As the engine
    draws its shadow maps (three.js, single-sided materials render their back faces into the map): a face casts when
    it looks away from the light or is two-sided, and its material is none of NOSHADOW / GLOW.
    ref: the result for lod 0 - leak (len(lights), n) bool is then where lod 0 stops the light and this model lets
    it land on something a viewer sees: on a face of its own that looks at the light, LEAK_EPS or more behind where
    lod 0 stopped it, at a point that is open to the outside (face_seen below), or (nothing stops it) on the ground - the wall, for a wall-mounted asset (front: the sign of
    y on the asset's side of it) - at a point that can be seen past the model. Light that ends inside a hollow
    building, or runs out into the open, is not counted.
    None without mathutils (outside Blender)."""
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        return None
    out = np.zeros((len(lights), n), dtype=bool)
    dist = np.full((len(lights), n), np.inf)
    leak = np.zeros((len(lights), n), dtype=bool) if ref is not None else None
    Vs, Ts, Ws, off = [], [], [], 0
    for p in prims:
        if p["lib"].startswith(NOSHADOW) or p["lib"].startswith(GLOW) or not len(p["I"]):
            continue
        Vs.append(np.asarray(p["P"], dtype=np.float64))
        Ts.append(np.asarray(p["I"], dtype=np.int64) + off)
        Ws.append(np.full(len(p["I"]), bool(p["double"]), dtype=bool))
        off += len(p["P"])
    if not Vs or not len(lights):
        return out, dist, leak
    V, T = np.vstack(Vs), np.vstack(Ts)
    W = np.concatenate(Ws).tolist()
    N = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]).tolist()
    rc = BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True, epsilon=0.0).ray_cast
    D = _light_dirs(n)[0].tolist()
    far = 2.0 * float(np.linalg.norm(V.max(0) - V.min(0))) + 10.0
    ground, wall = mount in ("ground", "wall_ground"), mount in ("wall", "wall_ground")
    UP = [(0.0, 0.0, 1.0), (0.7071, 0.0, 0.7071), (-0.7071, 0.0, 0.7071), (0.0, 0.7071, 0.7071), (0.0, -0.7071, 0.7071)]
    FR = [(0.0, front, 0.0), (0.7071, 0.7071 * front, 0.0), (-0.7071, 0.7071 * front, 0.0),
          (0.0, 0.7071 * front, 0.7071), (0.0, 0.7071 * front, -0.7071)]

    def seen(p, dirs):
        return any(rc(p, d_, far)[0] is None for d_ in dirs)

    def face_seen(p, nn, flip):
        # is the point p of a face (normal nn, looked at from the other side when flip) open to the outside: one of
        # five ways around its normal leaves the model (not into the ground, not into the wall)
        l = math.sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2])
        if l <= 0.0:
            return False
        s_ = -1.0 / l if flip else 1.0 / l
        a = (nn[0] * s_, nn[1] * s_, nn[2] * s_)
        b = (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0)
        u = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        lu = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
        u = (u[0] / lu, u[1] / lu, u[2] / lu)
        v = (a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0])
        q = (p[0] + a[0] * 2e-3, p[1] + a[1] * 2e-3, p[2] + a[2] * 2e-3)
        for cu, cv in ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
            d_ = (a[0] + cu * u[0] + cv * v[0], a[1] + cu * u[1] + cv * v[1], a[2] + cu * u[2] + cv * v[2])
            ld = math.sqrt(d_[0] * d_[0] + d_[1] * d_[1] + d_[2] * d_[2])
            d_ = (d_[0] / ld, d_[1] / ld, d_[2] / ld)
            if (ground and d_[2] < 0.0) or (wall and d_[1] * front < 0.0):
                continue
            if rc(q, d_, far)[0] is None:
                return True
        return False

    for li, (_name, pos, rng_, *_sh) in enumerate(lights):
        row, drow = out[li], dist[li]
        r0 = ref[0][li].tolist() if ref is not None else None
        t0s = ref[1][li].tolist() if ref is not None else None
        for di, d in enumerate(D):
            o, left, gone, lit = pos, rng_, 0.0, False
            t0 = t0s[di] if (r0 is not None and r0[di]) else None
            for _ in range(32):
                h = rc(o, d, left)
                if h[0] is None:
                    break
                t = h[2]
                nn = N[t]
                gone += h[3]
                back = nn[0] * d[0] + nn[1] * d[1] + nn[2] * d[2] > 0.0
                if t0 is not None and not lit and gone > t0 + LEAK_EPS and (W[t] or not back):
                    lit = face_seen(h[0], nn, back)
                if W[t] or back:
                    row[di] = True
                    drow[di] = gone
                    break
                s = h[3] + 1e-4
                left -= s
                gone += 1e-4
                if left <= 0.0:
                    break
                o = (o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s)
            if t0 is not None:
                if not lit and not row[di]:
                    if ground and d[2] < -1e-6 and t0 < -pos[2] / d[2] < rng_:
                        tg = -pos[2] / d[2]
                        lit = seen((pos[0] + d[0] * tg, pos[1] + d[1] * tg, 1e-3), UP)
                    elif wall and d[1] * front < -1e-6 and t0 < -pos[1] / d[1] < rng_:
                        tg = -pos[1] / d[1]
                        lit = seen((pos[0] + d[0] * tg, 1e-3 * front, pos[2] + d[2] * tg), FR)
                leak[li, di] = lit
    return out, dist, leak


def light_leak(c0, c1):
    """shadow_casts() of lod 0 and of a level (made with ref = that of lod 0) -> (share, index of the light): the
    share of the directions around the worst light in which the level lets a beam out - the direction leaks, it lies
    inside a shadow of lod 0 (three quarters of its 13 neighbours are stopped there) and a neighbour leaks with it (an
    edge that moved by a hair is no beam)."""
    NB = _light_dirs(c0[0].shape[1])[1]
    best = (0.0, None)
    for i in range(len(c0[0])):
        op = c1[2][i]
        if not op.any():
            continue
        beam = op & (c0[0][i][NB].mean(axis=1) >= 0.75) & (op[NB].sum(axis=1) >= 2)
        s = float(beam.sum()) / op.shape[0]
        if s > best[0]:
            best = (s, i)
    return best


# ----------------------------------------------------------------------------
# The metric
# ----------------------------------------------------------------------------

class Metric:
    """Image comparison against one lod-0 file. The lod-0 renders are made once per level and reused for every
    candidate. sockets: the asset's sockets (default: read from the sidecar next to glb0) - the shadows of its
    lights are compared too (m["leak"])."""

    def __init__(self, glb0, mount="ground", sockets=None):
        self.prims0 = load_glb(glb0)
        if sockets is None and isinstance(glb0, str):
            try:
                with open(os.path.splitext(glb0)[0] + ".json") as f:
                    sockets = json.load(f).get("sockets")
            except Exception:
                sockets = None
        self.lights = lights_of(sockets)
        self.cast0 = shadow_casts(self.prims0, self.lights) if self.lights else None
        self.mount = mount or "ground"
        self.mn, self.mx = bounds_of(self.prims0)
        self.views = views_for(self.mount, self.mn, self.mx)
        self.libid = {}
        self.uref = uv_reference(self.prims0)
        self.tris0 = tris_of(self.prims0)
        self._lv = {}
        self.seconds = 0.0
        self.diag = float(np.linalg.norm(self.mx - self.mn))
        self.z0 = zfight(self.prims0)
        self._holes = None

    def _level(self, level):
        if level not in self._lv:
            ext = float((self.mx - self.mn).max())
            px = max(DIST[level] * PX_PER_M, ext / MAXPX)
            est = _estimate(self.prims0, px, SPP)
            if est > NMAX:
                px *= math.sqrt(est / NMAX)
            spp = float(np.clip(0.5 * NMAX / max(est / SPP, 1.0), SPP, SPP_MAX))
            vws = [View(a, e, self.mn, self.mx, px) for a, e in self.views]
            cl = Cloud(self.prims0, px, self.libid, spp=spp)
            # a tear: a hole wider than the level's size limit (the mask is eroded that often before it counts)
            #         (a hole of w px leaves w - 2 px uncovered beyond one pixel; e erosions take 2 e + 1 of those)
            er = max(1, int(round((size_limit(level, self.diag) / px - 3.0) / 2.0)))
            vc = vox_cell(level, self.diag, self.prims0)
            K0, L0 = voxels(self.prims0, vc, self.mn)
            self._lv[level] = {"px": px, "spp": spp, "views": vws, "ref": [_shade(cl, v) for v in vws], "er": er,
                               "vc": vc, "vox": (K0, L0), "loose0": loose(K0, L0, K0, L0, vc)}
        return self._lv[level]

    def score(self, level, glb):
        """-> metrics dict of the level file `glb` against lod 0"""
        t0 = time.process_time()
        prims = glb if isinstance(glb, list) else load_glb(glb)
        L = self._level(level)
        cl = Cloud(prims, L["px"], self.libid, spp=L["spp"], nmax=int(NMAX * 1.5))
        sil, img, gl, tear = [], [], [], []
        tile, snew, glv, slit, thin = [], [], [], [], []
        gpx = [0.0, 0.0]                                # lod 0: pixels that show a glowing face, pixels in all
        plant = [i for k, i in self.libid.items() if k.startswith(PLANT)]
        l0 = l1 = 0.0
        nlib = max(len(self.libid), 1)
        n0, n1 = np.zeros(nlib + 1), np.zeros(nlib + 1)
        out = 0
        for vw, R in zip(L["views"], L["ref"]):
            K = _shade(cl, vw)
            out += K["out"]
            U = R["c"] | K["c"]
            I = R["c"] & K["c"]
            nu = int(U.sum())
            sil.append(1.0 - float(I.sum()) / nu if nu else 0.0)
            d_ = np.abs(K["b"] - R["b"]).mean(axis=2)
            img.append(float(d_[U].mean()) if nu else 0.0)
            tile.append(tile_max(d_, U) if nu else 0.0)
            n0c = float(R["c"].sum())
            snew.append(float((K["c"] & ~dilate(dilate(R["c"]))).sum()) / max(n0c, 1.0))
            l0 += float(R["lum"][I].sum())
            l1 += float(K["lum"][I].sum())
            g0 = R["g"]
            gpx[0] += float(g0.sum())
            gpx[1] += float(R["c"].sum())
            if int(g0.sum()) >= 2:
                gl.append(float((g0 & dilate(K["g"])).sum()) / float(g0.sum()))
                if float(g0.sum()) >= max(8.0, GLOW_VIEW_PX * n0c):
                    glv.append(gl[-1])
            # torn open: lod 0 shows a surface where the level shows nothing within a pixel, or only something that
            # lies more than 3 px behind it - over an area wider than the size limit (thinned plants do not count)
            r0, k0 = R["cov"], K["cov"]
            t = r0 & (~dilate(k0) | (k0 & (K["z"] - R["z"] > 3.0 * L["px"])))
            if plant:
                t &= ~np.isin(R["lib"], plant)
            # thin gaps (recorded as "slit" / "thin", no gate: they do not tell a course that was flattened from
            # small parts dropped on purpose): inside the lod-0 picture the level shows nothing at all - "thin": or
            # only something more than 3 px behind - in a line as long as the size limit, however thin (a tear has
            # to be that wide both ways)
            e0 = erode(r0)
            if plant:
                e0 &= ~np.isin(R["lib"], plant)
            nrun = 2 * L["er"] + 3
            slit.append(float(slits(e0 & ~k0, nrun).sum()) / max(float(r0.sum()), 1.0))
            thin.append(float(slits(e0 & (~k0 | (K["z"] - R["z"] > 3.0 * L["px"])), nrun).sum()) / max(float(r0.sum()), 1.0))
            for _ in range(L["er"]):
                if not t.any():
                    break
                t = erode(t)
            tear.append(float(t.sum()) / max(float(r0.sum()), 1.0))
            n0 += np.bincount(R["lib"][R["lib"] >= 0], minlength=nlib + 1)[:nlib + 1]
            n1 += np.bincount(K["lib"][K["lib"] >= 0], minlength=nlib + 1)[:nlib + 1]
        K1, L1 = voxels(prims, L["vc"], self.mn)
        lo = loose(L["vox"][0], L["vox"][1], K1, L1, L["vc"])
        names = {v: k for k, v in self.libid.items()}
        tot0, tot1 = max(n0.sum(), 1.0), max(n1.sum(), 1.0)
        ident, ident_lib = 1.0, None
        for i in range(nlib):
            if n0[i] >= 0.01 * tot0:
                s = min(n1[i] / n0[i], 1.0)
                if s < ident:
                    ident, ident_lib = float(s), names.get(i)
        new = [names.get(i) for i in range(nlib) if n1[i] >= 0.01 * tot1 and n0[i] == 0]
        m = {"sil": round(float(np.mean(sil)), 4), "sil_max": round(float(np.max(sil)), 4),
             "img": round(float(np.mean(img)), 5), "lum": round(l1 / l0, 4) if l0 > 0 else 1.0,
             "glow": round(float(np.mean(gl)), 3) if gl else None, "glow_px": round(gpx[0] / max(gpx[1], 1.0), 5),
             "ident": round(ident, 3), "ident_lib": ident_lib,
             "new_lib": new, "uvbad": round(uv_bad(prims, self.uref), 4), "outside": int(out),
             "tris": tris_of(prims), "px": round(L["px"], 5),
             "tear": round(float(np.mean(tear)), 5) if tear else 0.0, "tear_max": round(float(np.max(tear)), 5) if tear else 0.0,
             "loose": lo[0], "loose_share": round(lo[1], 5), "loose_m": lo[2], "cell": round(L["vc"], 4),
             "loose_at": None if lo[3] is None else [round(float(x), 2) for x in (self.mn - 2.0 * L["vc"] + lo[3] * L["vc"])]}
        # local defects, each from the worst view: the worst tile, pixels outside the lod-0 outline, a glowing face
        # going dark, and a light of the asset shining out where lod 0 casts a shadow
        m.update(tile=round(float(np.max(tile)), 4) if tile else 0.0,
                 tile_at=list(self.views[int(np.argmax(tile))]) if tile else None,
                 sil_new=round(float(np.max(snew)), 5) if snew else 0.0,
                 glow_min=round(float(np.min(glv)), 3) if glv else None,
                 slit=round(float(np.max(slit)), 5) if slit else 0.0, thin=round(float(np.max(thin)), 5) if thin else 0.0)
        if self.cast0 is not None:
            c1 = shadow_casts(prims, self.lights, ref=self.cast0, mount=self.mount,
                              front=-1.0 if (self.mx[1] + self.mn[1]) < 0 else 1.0)
            if c1 is not None:
                s_, i_ = light_leak(self.cast0, c1)
                m["leak"] = round(s_, 5)
                m["leak_at"] = self.lights[i_][0] if i_ is not None else None
        if self.z0 is not None and (self.z0["glow_area"] > 0 or any(p.get("zglow") for p in prims)):
            # cards lying on another surface: against lod 0 ("z0": its area under 3 mm, "z0w": under 3.25 mm,
            # "z40": under Z_KEEP less Z_EDGE, "dark0": its area of cards of unlit rooms, "tol": the slack)
            z = zfight(prims)
            if z is not None:
                z.pop("z3w", None)
                m["zf"] = dict(z, z0=self.z0["z3"], z0w=self.z0["z3w"], z40=self.z0["z4"], dark0=self.z0["dark_area"],
                               x30=self.z0["x3"], x40=self.z0["x4"],
                               tol=round(z_tol(self.z0["glow_area"]), 4))
        if os.environ.get("TSQ_HOLES_GATE", "1") != "0":
            # "no holes" (docs/LOD.md, tk/holes.py): the level as the street sees it - single-sided, close, from eye height
            try:
                from . import holes as _ho
                if self._holes is None:
                    self._holes = _ho.of_prims(self.prims0, self.mount)
                h = _ho.measure_prims(self._holes, prims, level)
                m["holes"] = dict(_ho.brief(h), why=_ho.ok(level, h))
            except ImportError:             # outside Blender (no mathutils): not measured
                pass
        dt = time.process_time() - t0
        self.seconds += dt
        m["s"] = round(dt, 2)
        return m


# ----------------------------------------------------------------------------
# Lettering: the tagged faces of tk.geo.text() on their own, letter by letter
# ----------------------------------------------------------------------------

def _tri_cells(P2, X, Y):
    """cells (centres X x Y) covered by the 2-D triangles P2 (m,3,2) -> (len(Y), len(X)) bool"""
    out = np.zeros(len(Y) * len(X), dtype=bool)
    if not len(P2):
        return out.reshape(len(Y), len(X))
    gx, gy = np.meshgrid(X, Y)
    px, py = gx.ravel()[:, None], gy.ravel()[:, None]
    step = max(1, int(2e6 // max(len(out), 1)))
    for i in range(0, len(P2), step):
        T = P2[i:i + step]
        sg = []
        for j in range(3):
            a, b = T[:, j], T[:, (j + 1) % 3]
            sg.append((b[:, 0] - a[:, 0])[None] * (py - a[:, 1][None]) - (b[:, 1] - a[:, 1])[None] * (px - a[:, 0][None]))
        e = 1e-12
        ins = ((sg[0] >= -e) & (sg[1] >= -e) & (sg[2] >= -e)) | ((sg[0] <= e) & (sg[1] <= e) & (sg[2] <= e))
        out |= ins.any(axis=1)
    return out.reshape(len(Y), len(X))


def _tri_normals(P):
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-20)[:, None], 0.5 * a


def text_prepare(T):
    """Lettering of lod 0, ready to compare levels against.
    T: {"P": (n,3,3) world triangles of the tagged faces, "g": (n,) number of their text() call, "k": (n,) 1 front |
    2 side | 3 back} (tk.final_levels.text_tris). -> list of groups {"id", "shape": the face type that carries the
    letter shape (1 raised or free-standing letters, 3 the floor of engraved ones), "h": letter height (m), "n", "u",
    "v": frame of the lettering plane, "glyphs": [{"P2", "nrm", "mn", "mx", "area"}]}; groups without flat faces are
    left out."""
    out = []
    if T is None or not len(T["P"]):
        return out
    P, g, k = np.asarray(T["P"], dtype=np.float64), np.asarray(T["g"]), np.asarray(T["k"])
    for gid in np.unique(g):
        sel = g == gid
        shape = 1 if (sel & (k == 1)).any() else 3 if (sel & (k == 3)).any() else 0
        if not shape:
            continue
        Q = P[sel & (k == shape)]
        nr, ar = _tri_normals(Q)
        Q, nr, ar = Q[ar > 1e-12], nr[ar > 1e-12], ar[ar > 1e-12]
        if not len(Q):
            continue
        n = (nr * ar[:, None]).sum(axis=0)
        ln = float(np.linalg.norm(n))
        if ln < 1e-9 * float(ar.sum()) or ln < 1e-20:
            n = nr[int(np.argmax(ar))]              # lettering wrapped all the way round: any plane does
        else:
            n = n / ln
        u = np.cross(n, (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0))
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        # letters = connected pieces of the shape faces
        key = np.round(Q.reshape(-1, 3) / 1e-5).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1, 3)
        lab = _components(int(inv.max()) + 1, np.concatenate([inv[:, :2], inv[:, 1:]]))[inv[:, 0]]
        glyphs, hs = [], []
        for l in np.unique(lab):
            m = lab == l
            q3 = Q[m]
            P2 = np.stack([q3 @ u, q3 @ v], axis=-1)
            pts = P2.reshape(-1, 2)
            side = 1e9
            for i in range(12):                      # side of the smallest square round the letter
                c, s_ = math.cos(i * math.pi / 12), math.sin(i * math.pi / 12)
                ea = pts @ np.array((c, s_))
                eb = pts @ np.array((-s_, c))
                side = min(side, max(float(ea.max() - ea.min()), float(eb.max() - eb.min())))
            gn = (nr[m] * ar[m][:, None]).sum(axis=0)
            gn /= max(float(np.linalg.norm(gn)), 1e-20)
            glyphs.append({"P2": P2, "nrm": gn, "mn": pts.min(axis=0), "mx": pts.max(axis=0), "area": float(ar[m].sum()),
                           "side": side})
            hs.append(side)
        hs = np.array(hs)
        big = hs >= 0.35 * hs.max()                 # dots and accents do not set the height
        A2 = np.stack([Q @ u, Q @ v], axis=-1)
        out.append({"id": int(gid), "shape": shape, "h": float(np.percentile(hs[big], 75)), "n": n, "u": u, "v": v,
                    "glyphs": glyphs, "tris": int(sel.sum()), "all": (A2, A2.min(axis=1), A2.max(axis=1), nr)})
    return out


def text_small(level, h):
    """is lettering of height h (m) under TEXT_PX pixels at the level's distance"""
    return h < TEXT_PX * DIST[level] * PX_PER_M


def text_measure(level, prep, T):
    """Lettering of a level against lod 0's (prep = text_prepare(lod-0 faces), T as there for the level).
    Every letter is drawn flat in its own plane on a grid of at least 24 cells across (never coarser than a quarter of
    the level's pixel) - lod 0's faces and the level's faces of the same text and type - and compared with one cell of
    tolerance: dev = (cells of the letter the level lacks + cells the level adds where lod 0 has no lettering) / cells
    of the letter. A line of lettering
    passes with dev <= TEXT_DEV, no letter over TEXT_WORST and its area within TEXT_COV. Lettering under TEXT_PX
    pixels high is not held to that: it has to be whole ("kept") or gone ("dropped"), "partial" fails; engraved
    lettering that small is free.
    -> None when lod 0 has no lettering, else {"ok", "groups", "gated", "dev", "worst", "cov", "tris", "small": {state:
       count}, "why": [...], "rows": [{"id", "h", "shape", "small", "dev", "worst", "cov", "state"}]}"""
    if not prep:
        return None
    px = DIST[level] * PX_PER_M
    if T is None or not len(T["P"]):
        P, g, k = np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    else:
        P, g, k = np.asarray(T["P"], dtype=np.float64), np.asarray(T["g"]), np.asarray(T["k"])
    nr, ar = _tri_normals(P) if len(P) else (np.zeros((0, 3)), np.zeros(0))
    rows, why = [], []
    small = {}
    gated = 0
    dev_w = worst_w = 0.0
    cov_lo, cov_hi = 1.0, 1.0
    for G in prep:
        sel = (g == G["id"]) & (k == G["shape"]) & (ar > 1e-12)
        Q, qn = P[sel], nr[sel]
        Q2 = np.stack([Q @ G["u"], Q @ G["v"]], axis=-1) if len(Q) else np.zeros((0, 3, 2))
        qmn = Q2.min(axis=1) if len(Q) else np.zeros((0, 2))
        qmx = Q2.max(axis=1) if len(Q) else np.zeros((0, 2))
        n0 = n1 = off = 0.0
        worst = 0.0
        for gl in G["glyphs"]:
            ext = float((gl["mx"] - gl["mn"]).max())
            q = max(px / 4.0, min(ext, 1.5 * G["h"]) / 24.0, 1e-6)
            lo, hi = gl["mn"] - 2.5 * q, gl["mx"] + 2.5 * q
            X = np.arange(lo[0], hi[0] + q, q)
            Y = np.arange(lo[1], hi[1] + q, q)
            m0 = _tri_cells(gl["P2"], X, Y)
            c0 = float(m0.sum())
            if c0 < 1:
                continue
            A2, amn, amx, an = G["all"]             # the neighbours reach into the letter's window
            nb = (amx >= lo[None]).all(axis=1) & (amn <= hi[None]).all(axis=1) & (an @ gl["nrm"] > 0.3)
            ma = _tri_cells(A2[nb], X, Y) | m0
            near = ((qmx >= lo[None]).all(axis=1) & (qmn <= hi[None]).all(axis=1) & (qn @ gl["nrm"] > 0.3)) if len(Q) else np.zeros(0, dtype=bool)
            m1 = _tri_cells(Q2[near], X, Y)
            bad = float((m0 & ~dilate(m1)).sum() + (m1 & ~dilate(ma)).sum())
            w = q * q
            n0 += float(ma.sum()) * w
            n1 += float(m1.sum()) * w
            off += bad * w
            worst = max(worst, bad / c0)
        if n0 <= 0:
            continue
        dev, cov = off / max(sum(float(gl["area"]) for gl in G["glyphs"]), 1e-12), n1 / n0
        sm = bool(text_small(level, G["h"]))
        row = {"id": G["id"], "h": round(G["h"], 4), "shape": G["shape"], "small": sm, "dev": round(dev, 4),
               "worst": round(worst, 4), "cov": round(cov, 4)}
        if sm:
            st = ("free" if G["shape"] == 3 else "kept" if (cov >= 0.7 and dev <= 0.3) else "dropped" if cov <= 0.02
                  else "partial")
            small[st] = small.get(st, 0) + 1
            if st == "partial":
                why.append(f"small lettering {G['id']} ({1000 * G['h']:.0f} mm) is neither whole nor gone (area {cov:.2f}, off {dev:.2f})")
        else:
            gated += 1
            ok = dev <= TEXT_DEV and worst <= TEXT_WORST and TEXT_COV[0] <= cov <= TEXT_COV[1]
            st = "ok" if ok else "bad"
            dev_w, worst_w = max(dev_w, dev), max(worst_w, worst)
            cov_lo, cov_hi = min(cov_lo, cov), max(cov_hi, cov)
            if not ok:
                why.append(f"lettering {G['id']} ({1000 * G['h']:.0f} mm): off {dev:.2f} (limit {TEXT_DEV}), worst letter {worst:.2f} "
                           f"(limit {TEXT_WORST}), area {cov:.2f}")
        row["state"] = st
        rows.append(row)
    return {"ok": not why, "groups": len(rows), "gated": gated, "dev": round(dev_w, 4), "worst": round(worst_w, 4),
            "cov": round(cov_lo if (1.0 - cov_lo) >= (cov_hi - 1.0) else cov_hi, 4), "tris": int(len(P)), "small": small,
            "why": why, "rows": rows}


# ----------------------------------------------------------------------------
# Gates and the choice
# ----------------------------------------------------------------------------

def _text_worse(tx, td):
    """does the lettering of a candidate (tx) fail harder than that of the decimated copy (td)"""
    return td is None or td["ok"] or len(tx["why"]) > len(td["why"]) or tx["dev"] > td["dev"] + 1e-6


def _z_bad(m, key=False):
    """the depth-fight gate in absolute terms -> the reason it fails, or None (key=True: -> which of "z3" / "z4" fails)"""
    z = (m or {}).get("zf")
    if z and z["z3"] > max(z["z0"], z.get("z0w", 0.0)) + z["tol"]:
        return "z3" if key else (f"depth fight: {z['z3']:.3f} m2 of glowing faces and window cards lie within 3 mm of "
                                 f"another surface (lod 0 has {z['z0']:.3f}, {z['tol']:.3f} more is allowed)")
    if z and z.get("z40") is not None and z.get("z4", 0.0) > z["z40"] + z["tol"]:
        return "z4" if key else (f"depth fight: {z['z4']:.3f} m2 of glowing faces and window cards lie within "
                                 f"{1000 * Z_KEEP:.0f} mm of another surface (lod 0 has {z['z40']:.3f}, {z['tol']:.3f} "
                                 f"more is allowed; {z.get('dark4', 0.0):.3f} of it on cards of unlit rooms)")
    for k_, k0, mm in (("x3", "x30", 1000 * Z_BUCKET[2]), ("x4", "x40", 1000 * Z_KEEP)):
        if z and z.get(k0) is not None and z.get(k_, 0.0) > z[k0] + z["tol"]:
            return k_ if key else (f"depth fight: a face of another material lies within {mm:.0f} mm over {z[k_]:.4f} m2 "
                                   f"of glowing faces and window cards (lod 0 has {z[k0]:.4f}, {z['tol']:.4f} more is "
                                   f"allowed; {z.get('xdark4', 0.0):.4f} of it on cards of unlit rooms)")
    return None


def gates(level, m, dec=None):
    """Image gates of a level -> reasons it fails (empty = passes). With `dec` (the metrics of the decimated copy,
    which is what ships otherwise) a gate only fails when the candidate is also worse than the decimated copy there;
    dec=None gives the absolute verdict.
    Whole: the silhouette; no hole wider than the level's size limit where lod 0 is closed ("torn open"); what touched
    in lod 0 still touches (see loose()); in absolute terms also no more than OPEN of the tiled fields bare ("torn
    field", m["open"] from tk.reduce - the decimated copy has no such measure, so this one never counts against it:
    it keeps a reduced level over its budget out and lets a building's 2 x candidate win over a cheaper torn one).
    No holes (docs/LOD.md rule g, m["holes"] from tk.holes): the street does not look through the level where lod 0
    is closed - single-sided, from eye height close by, which the views of this metric (from above, far) do not see.
    Depth fight: the glowing area with a parallel face of another material closer than 3 mm (zfight) may exceed that
    of lod 0 by z_tol() at most.
    Lit brightness: outside LUM fails on its own; against a decimated copy that is outside LUM too the image
    difference decides; against one inside LUM the candidate passes only within LUM_WIDE and with an image difference
    at least LUM_GAIN lower."""
    why = []
    lim = max(SIL[level], 1.5 * dec["sil"]) if dec else SIL[level]
    if m["sil"] > lim:
        why.append(f"silhouette differs by {m['sil']:.3f} (limit {lim:.3f})")
    t = m.get("tear") or 0.0
    lim = max(TEAR[level], 1.5 * (dec.get("tear") or 0.0)) if dec else TEAR[level]
    if t > lim:
        why.append(f"torn open: {100 * t:.2f} % of the picture shows through (limit {100 * lim:.2f} %)")
    h = m.get("holes")
    if h and h.get("why") and (dec is None or h["holes"] > ((dec.get("holes") or {}).get("holes") or 0.0)):
        why += h["why"]                 # tk.holes.ok(): the street looks through the model
    op = m.get("open") or 0.0
    if dec is None and op > OPEN:
        why.append(f"torn field: {100 * op:.0f} % of the tiled surface lies bare (limit {100 * OPEN:.0f} %)")
    n = m.get("loose") or 0
    if n and (dec is None or (m["loose_share"] > 1.5 * (dec.get("loose_share") or 0.0)
                              and (n > (dec.get("loose") or 0) or m["loose_m"] > (dec.get("loose_m") or 0.0)))):
        why.append(f"loose pieces: {n} that touched in lod 0 no longer do (the largest {m['loose_m']:.2f} m, "
                   f"gap over {m.get('cell', 0.0):.3f} m)")
    if not (LUM[0] <= m["lum"] <= LUM[1]):
        if dec is None:
            why.append(f"lit brightness is {m['lum']:.2f} x lod 0 (allowed {LUM[0]}-{LUM[1]})")
        elif LUM[0] <= dec["lum"] <= LUM[1]:
            if not (LUM_WIDE[0] <= m["lum"] <= LUM_WIDE[1]):
                why.append(f"lit brightness is {m['lum']:.2f} x lod 0 (allowed {LUM_WIDE[0]}-{LUM_WIDE[1]} against a "
                           f"decimated copy at {dec['lum']:.2f})")
            elif m["img"] > (1.0 - LUM_GAIN) * dec["img"]:
                why.append(f"lit brightness is {m['lum']:.2f} x lod 0 and the image difference {m['img']:.4f} is not "
                           f"{100 * LUM_GAIN:.0f} % under the decimated copy's {dec['img']:.4f}")
    g = m.get("glow")
    if g is not None and g < GLOW_MIN and m.get("glow_px", 1.0) >= GLOW_PX:
        gd = dec.get("glow") if dec else None
        if gd is None or g < gd - 0.05:
            why.append(f"only {100 * g:.0f} % of the glowing pixels still glow")
    zw = _z_bad(m)
    if zw:
        zk = _z_bad(m, key=True)
        zd = (dec.get("zf") or {}).get(zk) if dec else None
        if dec is None or zd is None or m["zf"][zk] > zd + 0.25 * m["zf"]["tol"]:
            why.append(zw)
    tx = m.get("text")
    if tx and not tx["ok"] and (dec is None or _text_worse(tx, dec.get("text"))):
        why.append("lettering: " + tx["why"][0] + (f" (+{len(tx['why']) - 1} more)" if len(tx["why"]) > 1 else ""))
    # local defects (the reasons start with LOCAL)
    tl = m.get("tile")
    tlim = TILE[level] * (TILE_IDENT if m.get("native") is True else 1.0)
    if tl is not None and tl > tlim and (dec is None or dec.get("tile") is None or tl > TILE_DEC * dec["tile"]):
        why.append(f"local defect: the worst tile of the picture differs by {tl:.3f} (limit {tlim:.3g}"
                   + (f", decimated {dec['tile']:.3f}" if dec and dec.get("tile") is not None else "")
                   + (f"; view {m['tile_at']}" if m.get("tile_at") else "") + ")")
    sn = m.get("sil_new")
    if sn is not None and sn > SIL_NEW[level] and (dec is None or sn > 1.5 * (dec.get("sil_new") or 0.0)):
        why.append(f"new outline: {100 * sn:.1f} % of the picture lies outside the outline of lod 0 in one view "
                   f"(limit {100 * SIL_NEW[level]:.1f} %)")
    gv = m.get("glow_min")
    if gv is not None and gv < GLOW_VIEW and (dec is None or dec.get("glow_min") is None or gv < dec["glow_min"] - 0.05):
        why.append(f"a lit face goes dark: only {100 * gv:.0f} % of the glowing pixels of one view still glow")
    lk = m.get("leak")
    if lk is not None and lk > LEAK[level] and (dec is None or lk > max(1.5 * (dec.get("leak") or 0.0), LEAK[level])):
        why.append(f"light leak: the light {m.get('leak_at')} shines out past a shadow of lod 0 in {100 * lk:.2f} % "
                   f"of the directions around it (limit {100 * LEAK[level]:.1f} %)")
    return why


def shape_gates(level, m):
    """the absolute gates on the outline alone: silhouette, tears and loose pieces"""
    return [w for w in gates(level, m, None) if w.startswith(("silhouette", "torn", "loose"))]


CLOSE_FREE = 0.25               # choose(): share of the budget the triangles that closed holes may take for free


def choose(level, candidates):
    """Pick what ships at a level.

    candidates: [{"name", "kind": "authored" | "reduced" | "decimated", "tris", "budget", "hard": [reasons from
    tk.final_levels.finalize_level other than the budget], "m": Metric.score(...) (+ "text": text_measure(...)),
    "allow": how far over the budget this asset may go at this level (default ALLOW2 at lod 2, else 1)}].
    Order: hard gates, then the image gates, then the lowest image difference wins; the decimated copy (always
    eligible - it is the fallback) only wins when it is better by more than DEC_MARGIN, and never over a candidate
    whose lettering (or depth-fight gate) passes when its own does not, nor over a reduced copy of an asset that
    declares identity parts ("ident": True - tk.lod_standin / lod_keep) and passes its gates.
    A reduced lod 2 over its budget stays eligible when it passes every gate in absolute terms (the brightness gate
    as above, against the decimated copy) and beats the decimated copy on image difference: up to ALLOW2 x the budget
    it wins when it is better than the best candidate inside the budget by more than DEC_MARGIN - and, when that one
    passes every gate in absolute terms, by at least ALLOW_GAIN of its image difference; between ALLOW2 and its "allow" (ALLOW2B for buildings and landmarks) only when
    nothing cheaper passes or the best cheaper one fails the silhouette or tear gate in absolute terms.
    Last resort at lod 2 ("last": True): when nothing passes and the decimated copy fails its lettering or the
    depth-fight gate in absolute terms, the reduced candidate inside its allowance with the lowest image difference
    ships that passes every gate but the torn-field one and has no more than OPEN_LAST of its fields bare.
    At either level ("last": True too): when nothing passes and the decimated copy fails the depth-fight gate in
    absolute terms, the reduced candidate inside its allowance with the lowest image difference ships that passes
    that gate, its lettering, tears, loose pieces and glow in absolute terms, with a silhouette within Z_MISS[0] x
    the limit, a brightness within LUM_WIDE and an image difference within Z_MISS[1] x the decimated copy's.
    And ("last": True): when nothing passes and the decimated copy fails the no-holes gate, the reduced candidate
    with the lowest image difference ships that passes that gate (or shows through half as much at most) and every other one against the decimated copy, has
    no more triangles than it and an image difference within Z_MISS[1] x its own.
    -> {"level", "winner": name, "kind", "allowance": bool, "over": 0 | 1.5 | 2.0, "why": {name: [reasons it lost]},
        "clean": the winner passes the absolute image gates, "img": its image difference}"""
    # the triangles that closed holes ("closed": tk.reduce._close, tk.final._close_holes) are not held against the
    # budget - up to CLOSE_FREE of it: a candidate that needs more than that to be whole is not a cheap one
    candidates = [dict(c, raw=c["tris"], tris=c["tris"] - min(int(c["closed"]), int(CLOSE_FREE * c["budget"])))
                  if c.get("closed") else c for c in candidates]
    dec = next((c for c in candidates if c["kind"] == "decimated"), None)
    dm = dec.get("m") if dec else None
    for c in candidates:
        if c.get("ident") and c.get("m") is not None:
            c["m"]["native"] = True                     # gates(): the local-tile limit is TILE_IDENT times wider
    out = {"level": level, "winner": None, "kind": None, "allowance": False, "over": 0, "why": {}, "clean": False,
           "img": None}
    tiers = {0: None, 1: None, 2: None}
    for c in candidates:
        if c is dec:
            continue
        why = list(c.get("hard") or [])
        m = c.get("m")
        if m is None:
            why.append("no image (level file missing)")
        else:
            why += gates(level, m, dm)
        tier = 0
        if c["tris"] > c["budget"]:
            allow = c.get("allow") or (ALLOW2 if level == 2 else 1.0)
            tier = 1 if c["tris"] <= ALLOW2 * c["budget"] else 2
            if not (c["kind"] == "reduced" and level == 2 and c["tris"] <= allow * c["budget"]):
                why.append(f"{c['tris']} tris is over the budget of {c['budget']}")
            elif not why:
                ab = [w for w in gates(level, m, None) if not w.startswith(("lit brightness", "light leak"))]
                if dm is None or not m["img"] < dm["img"]:
                    why.append(f"over the budget ({c['tris']} / {c['budget']}) and not a better picture than decimated")
                elif ab:
                    why.append(f"over the budget ({c['tris']} / {c['budget']}) and {ab[0]}")
        out["why"][c["name"]] = why
        if not why and (tiers[tier] is None or m["img"] < tiers[tier][0]):
            tiers[tier] = (m["img"], c, tier)
    best = tiers[0]
    if tiers[1] is not None:
        # half as many triangles again have to buy something: a level inside the budget that passes every gate in
        # absolute terms only gives way to a clearly better picture
        need = DEC_MARGIN if best is None or gates(level, best[1]["m"], None) else max(DEC_MARGIN, ALLOW_GAIN * best[0])
        if best is None or tiers[1][0] + need < best[0]:
            best = tiers[1]
        else:
            out["why"][tiers[1][1]["name"]] = [f"over the budget and not better than {best[1]['name']} by more than {need:.4f}"]
    if tiers[2] is not None:
        if best is None or shape_gates(level, best[1]["m"]):
            best = tiers[2]
        else:
            out["why"][tiers[2][1]["name"]] = [f"over 1.5 x the budget and {best[1]['name']} keeps the outline whole"]
    if dec is not None and best is not None and dm is not None and dm["img"] + DEC_MARGIN < best[0]:
        tb, td = best[1]["m"].get("text"), dm.get("text")
        if best[1].get("ident"):
            # tk.lod_standin / lod_keep: the generator said what its identity parts look like at this level. A
            # stand-in is meant to differ from lod 0, so the picture alone does not hand the level to the decimated
            # copy (which collapses those parts like everything else); a gate the reduced copy fails still does.
            out["why"][dec["name"]] = ["lower image difference, but the reduced copy carries the generator's own identity parts"]
        elif td and not td["ok"] and tb and tb["ok"]:
            out["why"][dec["name"]] = ["lower image difference, but its lettering fails: " + td["why"][0]]
        elif _z_bad(dm) and not _z_bad(best[1]["m"]):
            out["why"][dec["name"]] = ["lower image difference, but " + _z_bad(dm)]
        else:
            out["why"][best[1]["name"]] = [f"image difference {best[0]:.4f} is not better than decimated ({dm['img']:.4f})"]
            best = None
    if dec is not None and best is None and dm is not None and level == 2:
        # nothing passes, and the decimated copy breaks its lettering or lays a card onto a wall: a reduced copy
        # inside its allowance that does neither and misses the torn-field limit alone (by up to OPEN_LAST) ships
        td = dm.get("text")
        if (td and not td["ok"]) or _z_bad(dm):
            for c in candidates:
                m = c.get("m")
                if c is dec or c["kind"] != "reduced" or m is None or c.get("hard"):
                    continue
                if c["tris"] > (c.get("allow") or ALLOW2) * c["budget"] or not m["img"] < dm["img"]:
                    continue
                ab = [w for w in gates(level, m, None) if not w.startswith("lit brightness")]
                if gates(level, m, dm) or [w for w in ab if not w.startswith("torn field")] or (m.get("open") or 0.0) > OPEN_LAST:
                    continue
                if best is None or m["img"] < best[0]:
                    best = (m["img"], c, 0 if c["tris"] <= c["budget"] else 1 if c["tris"] <= ALLOW2 * c["budget"] else 2)
            if best is not None:
                out["why"][dec["name"]] = ["its lettering fails: " + td["why"][0] if (td and not td["ok"]) else _z_bad(dm)]
                out["why"][best[1]["name"]] = []
                out["last"] = True
    if dec is not None and best is None and dm is not None and _z_bad(dm):
        # the decimated copy lays a card onto another surface (it flickers on screen), so it is no shelter: a
        # reduced copy inside its allowance that keeps the clearance, its lettering and every other gate in
        # absolute terms, and misses the outline or the brightness band narrowly alone (Z_MISS), ships instead
        for c in candidates:
            m = c.get("m")
            if c is dec or c["kind"] != "reduced" or m is None or c.get("hard"):
                continue
            if c["tris"] > ((c.get("allow") or ALLOW2) if level == 2 else 1.0) * c["budget"]:
                continue
            rest = [w for w in gates(level, m, None) if not w.startswith(("silhouette", "lit brightness", "torn field"))]
            if (rest or m["sil"] > Z_MISS[0] * SIL[level] or not (LUM_WIDE[0] <= m["lum"] <= LUM_WIDE[1])
                    or m["img"] > Z_MISS[1] * dm["img"] or (m.get("open") or 0.0) > OPEN_LAST):
                continue
            if best is None or m["img"] < best[0]:
                best = (m["img"], c, 0 if c["tris"] <= c["budget"] else 1 if c["tris"] <= ALLOW2 * c["budget"] else 2)
        if best is not None:
            out["why"][dec["name"]] = [_z_bad(dm)]
            out["why"][best[1]["name"]] = []
            out["last"] = True
    if dec is not None and best is None and dm is not None and [w for w in gates(level, dm, None) if w.startswith(LOCAL)]:
        # both fail: the decimated copy shows a local defect of its own, so it is no shelter from one - the reduced
        # copy inside its allowance ships that misses nothing but local-defect gates against it and has the lower
        # image difference (out["local"] lists what the winner misses)
        for c in candidates:
            m = c.get("m")
            if c is dec or c["kind"] != "reduced" or m is None or c.get("hard"):
                continue
            if c["tris"] > ((c.get("allow") or ALLOW2) if level == 2 else 1.0) * c["budget"] or not m["img"] < dm["img"]:
                continue
            if [w for w in gates(level, m, dm) if not w.startswith(LOCAL)]:
                continue
            if c["tris"] > c["budget"] and [w for w in gates(level, m, None)
                                            if not w.startswith(("lit brightness",) + LOCAL)]:
                continue
            if best is None or m["img"] < best[0]:
                best = (m["img"], c, 0 if c["tris"] <= c["budget"] else 1 if c["tris"] <= ALLOW2 * c["budget"] else 2)
        if best is not None:
            out["why"][dec["name"]] = [w for w in gates(level, dm, None) if w.startswith(LOCAL)][:1]
            out["why"][best[1]["name"]] = []
            out["last"] = True
    if dec is not None and best is None and dm is not None and (dm.get("holes") or {}).get("why"):
        # the decimated copy shows through (rule g: its repair costs triangles too), so it is no shelter from a
        # budget or from a slightly better picture: the reduced copy ships that does not show through, has no more
        # triangles than it and passes every other gate against it
        for c in candidates:
            m = c.get("m")
            if c is dec or c["kind"] != "reduced" or m is None or c.get("hard"):
                continue
            h = m.get("holes") or {}
            if (h.get("why") and h["holes"] > 0.5 * dm["holes"]["holes"]) or c.get("raw", c["tris"]) > dec.get("raw", dec["tris"]):
                continue
            if gates(level, m, dm) or m["img"] > Z_MISS[1] * dm["img"]:
                continue
            if best is None or m["img"] < best[0]:
                best = (m["img"], c, 0 if c["tris"] <= c["budget"] else 1 if c["tris"] <= ALLOW2 * c["budget"] else 2)
        if best is not None:
            out["why"][dec["name"]] = list(dm["holes"]["why"])[:1]
            out["why"][best[1]["name"]] = []
            out["last"] = True
    if dec is not None and best is None:
        best = (dm["img"] if dm else None, dec, 0)
    if best is not None:
        c = best[1]
        out.update(winner=c["name"], kind=c["kind"], allowance=bool(best[2]), over=(0, ALLOW2, ALLOW2B)[best[2]],
                   img=best[0], clean=bool(c.get("m")) and not gates(level, c["m"], None))
        # the local-defect gates the winner misses in absolute terms (it ships because nothing does better there)
        out["local"] = [w for w in gates(level, c["m"], None) if w.startswith(LOCAL)] if c.get("m") else []
    return out


# ----------------------------------------------------------------------------
# Probe: the same splat on in-memory triangles (tk.reduce checks its own steps with it)
# ----------------------------------------------------------------------------

class Probe:
    """Outline / depth / glow of a triangle soup against a reference soup, from a handful of views.

        pr = Probe(V0, T0, glow0, two0, mount, level)
        pr.measure(V, T, glow, two, tone) -> {"sil", "hole", "lost", "glow", "lum", "tone"}
    tone: the albedo luminance of every triangle (optional); lum is then the brightness of the picture over the
    reference's, unlit (what dropping dark details off a light ground, or the reverse, does to it), and "tone" the
    mean difference of the two pictures pixel by pixel (gamma space) - which two shifts in opposite directions
    cannot hide.
    sil: silhouette difference as in Metric; hole: share of the reference pixels where the surface now lies deeper
    than `dtol` (something in front went missing); lost: share of the reference pixels with nothing left within one
    pixel (a real gap - shaving a sub-pixel rim off thin things does not count); glow: share of the reference's
    glowing pixels that still glow."""

    def __init__(self, V, T, glow, two, mount, level, maxpx=240, spp=3.0, nmax=220000, dtol=None, tone=None, see=None):
        self.spp, self.nmax = spp, nmax
        P = V[np.unique(T)] if len(T) else np.zeros((1, 3))
        self.mn, self.mx = P.min(0), P.max(0)
        ext = float((self.mx - self.mn).max())
        px = max(DIST[level] * PX_PER_M, ext / maxpx)
        a = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1).sum() if len(T) else 0.0
        est = a * spp / (px * px)
        if est > nmax:
            px *= math.sqrt(est / nmax)
        self.px = px
        # small and thin things: more samples per pixel, so that the rim pixels (hit by chance) settle
        self.spp = spp = float(np.clip(0.6 * nmax / max(a / (px * px), 1.0), spp, SPP_MAX))
        self.sc = min(1.0, nmax / max(a * spp / (px * px) + 0.5 * len(T), 1.0))
        self.dtol = dtol if dtol is not None else 3.0 * px
        if mount in ("wall", "wall_ground"):
            f = 180.0 if (self.mx[1] + self.mn[1]) > 0 else 0.0
            va = [(f - 55.0, 10.0), (f, 10.0), (f + 55.0, 10.0), (f + 20.0, 38.0)]
        else:
            va = [(25.0, 10.0), (115.0, 10.0), (205.0, 10.0), (295.0, 10.0), (70.0, 38.0), (250.0, 38.0)]
        self.views = [View(a_, e_, self.mn, self.mx, px) for a_, e_ in va]
        self.ref = self._render(V, T, glow, two, tone, see)

    def _render(self, V, T, glow, two, tone=None, see=None):
        if not len(T):
            return [(np.zeros((v.H, v.W), dtype=bool),) * 2 + (np.zeros((v.H, v.W), dtype=np.float32), np.zeros((v.H, v.W), dtype=bool),
                                                               np.zeros((v.H, v.W), dtype=np.float32)) for v in self.views]
        tri, W, fn, _ = _points(V, T, self.px, self.spp, self.nmax, sc=self.sc)
        P = (V[T[tri, 0]] * W[:, 0:1] + V[T[tri, 1]] * W[:, 1:2] + V[T[tri, 2]] * W[:, 2:3]).astype(np.float32)
        F = fn[tri].astype(np.float32)
        tw = np.asarray(two, dtype=bool)[tri]
        gs = np.asarray(glow, dtype=bool)[tri]
        tn = np.asarray(tone, dtype=np.float32)[tri] if tone is not None else np.ones(len(tri), dtype=np.float32)
        sw = np.asarray(see, dtype=bool)[tri] if see is not None else None
        out = []
        for vw in self.views:
            win, z, _ = _zwin(P, F, tw, vw, sw)
            cov = (win >= 0).reshape(vw.H, vw.W)
            w = np.maximum(win, 0)
            out.append((cov, erode(dilate(cov)), np.where(cov, z[w].reshape(vw.H, vw.W), np.float32(1e9)),
                        cov & gs[w].reshape(vw.H, vw.W), np.where(cov, tn[w].reshape(vw.H, vw.W), np.float32(0.0))))
        return out

    def wins(self, V, T, two, spp=8.0, see=None):
        """How many pixels each triangle of the soup wins over all views (densely sampled, so that hardly anything
        shows through the gaps between the samples of what lies in front) -> (len(T),) int array
        see: clear glass (per triangle) - what lies behind it wins its pixels as well as the glass"""
        out = np.zeros(len(T), dtype=np.int64)
        if not len(T):
            return out
        a = 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
        sc = min(1.0, 3.0 * self.nmax / max(float(a.sum()) * spp / (self.px * self.px) + 0.5 * len(T), 1.0))
        tri, W, fn, _ = _points(V, T, self.px, spp, None, sc=sc)
        P = (V[T[tri, 0]] * W[:, 0:1] + V[T[tri, 1]] * W[:, 1:2] + V[T[tri, 2]] * W[:, 2:3]).astype(np.float32)
        F = fn[tri].astype(np.float32)
        tw = np.asarray(two, dtype=bool)[tri]
        sw = np.asarray(see, dtype=bool)[tri] if see is not None else None
        for vw in self.views:
            win, _z, _ = _zwin(P, F, tw, vw)
            out += np.bincount(tri[win[win >= 0]], minlength=len(T))
            if sw is not None and sw.any():
                w2 = _zwin(P, F, tw, vw, sw)[0]
                w2 = w2[(w2 >= 0) & (w2 != win)]
                out += np.bincount(tri[w2], minlength=len(T))
        return out

    def measure(self, V, T, glow, two, tone=None, see=None):
        cur = self._render(V, T, glow, two, tone, see)
        sil, hole, lost, gl = [], [], [], []
        l0 = l1 = td = nb = 0.0
        for (c0, cc0, z0, g0, t0), (c1, cc1, z1, g1, t1) in zip(self.ref, cur):
            U = cc0 | cc1
            nu = int(U.sum())
            sil.append(1.0 - float((cc0 & cc1).sum()) / nu if nu else 0.0)
            both = c0 & c1
            n0 = int(c0.sum())
            hole.append(float((both & (z1 - z0 > self.dtol)).sum()) / n0 if n0 else 0.0)
            lost.append(float((c0 & ~dilate(c1)).sum()) / n0 if n0 else 0.0)
            if int(g0.sum()) >= 2:
                gl.append(float((g0 & dilate(g1)).sum()) / float(g0.sum()))
            a0, a1 = t0[both], t1[both]
            l0 += float(a0.sum())
            l1 += float(a1.sum())
            td += float(np.abs(np.power(a1, 1.0 / 2.2) - np.power(a0, 1.0 / 2.2)).sum())
            nb += float(len(a0))
        return {"sil": float(np.mean(sil)), "hole": float(np.mean(hole)), "lost": float(np.mean(lost)),
                "glow": float(np.mean(gl)) if gl else None, "lum": l1 / l0 if l0 > 0 else 1.0,
                "tone": td / nb if nb > 0 else 0.0}
