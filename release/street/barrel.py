"""Coopered oak barrel. Built the way a cooper builds it: 20-30 individual tapered staves of uneven
width (each a separate bulged board with eased edges, a V joint to its neighbour and its own end grain),
drawn together by riveted wrought-iron hoops that follow the taper, heads made of parallel boards let
into the croze 4 cm below the chime (the cooper's mark burnt into the front head), a bung in the widest
stave and a tap.

Poses: 'stand' (on its end; a tapped cask is stood on two oak bearers so a jug fits under the spout),
'side' (on a pegged oak stillage with chocks, tap in the front head), 'half' (sawn through at the bilge
and planted) and 'table' (a round plank top nailed to two battens that lie across the chime, dressed
with a candle and a stoneware tankard like the cafe terrace of the reference).

Weathering is resolved in the mesh: the staves carry extra rows under every hoop and extra columns in
the bung / tap staves so iron-tannate bands, drip runs, the damp foot and the bleached upper staves are
painted at 2-4 cm frequency; the brand, tap drip and mug rings are fine-grid overlays on the head boards.

Levels (docs/LOD.md): lod 1 keeps every stave as one flat board with its V joints, on the few rows the outline
and the stains need; the hoops are bands that bend from stave to stave (one flat to each: a turned ring of
fewer sides would cut into the boards), the brand and the mug ring stay on a coarser grid, the planter keeps
every other leaf and floret, larger and plainer. lod 2 is thirteen-odd double staves on five rows under plain
hoop bands, plain head boards, a few big leaves. Nothing under a cask that stands on bearers is built at 1 / 2.
"""
from tk import *
from math import ceil, acos, asin, exp, sin, cos, pi, radians, degrees, sqrt
import numpy as np
import bpy

# auto_wear is halved: full edge wear burnishes every rivet head and stave edge, which reads noisy at night.
META = {"name": "Oak barrel", "class": "prop", "mount": "ground", "focus": 0.56, "auto_wear": 0.45, "lod": True}
PARAMS = {
    "height": {"default": 0.92, "min": 0.55, "max": 1.3,
               "desc": "stave length = height of the standing barrel (m); the half barrel is cut from a barrel this tall"},
    "pose": {"default": "stand", "choices": ["stand", "side", "half", "table"],
             "desc": "standing / lying on a cradle / half-barrel planter / standing with a plank table top"},
    "girth": {"default": 0.76, "min": 0.6, "max": 0.95, "desc": "bilge diameter / height"},
    "bulge": {"default": 0.19, "min": 0.08, "max": 0.3, "desc": "belly: 1 - head diameter / bilge diameter"},
    "hoops": {"default": 6, "choices": [4, 6, 8], "desc": "number of iron hoops on the full barrel"},
    "wood": {"default": "#7f6b55", "desc": "oak tone of the staves (weathered oak; stain / age colour)"},
    "rust": {"default": 0.35, "min": 0.0, "max": 1.0,
             "desc": "rust on the hoops (0 = dark oiled iron, 1 = orange scale all over) and iron stain in the wood under them"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0,
            "desc": "condition of the cooperage: open joints, proud / sunken staves, slipped hoops, silvered staves, stains, moss"},
    "tap": {"default": "brass", "choices": ["brass", "wood", "none"],
            "desc": "spigot (side: in the front head; stand / table: low in a stave, and the cask is stood on two bearers)"},
    "brand": {"default": "XXX", "desc": "cooper's mark burnt into the head (empty = none)"},
    "planting": {"default": "geranium", "choices": ["geranium", "box", "soil"], "desc": "pose half: what grows in the planter"},
    "bloom": {"default": "#b3262b", "desc": "pose half: flower colour of the geraniums"},
    "top_width": {"default": 0.84, "min": 0.6, "max": 1.2, "desc": "pose table: diameter of the plank top (m)"},
    "candle": {"default": True, "desc": "pose table: lit candle in a brass chamberstick on the top"},
    "tankard": {"default": True, "desc": "pose table: stoneware tankard left on the top"},
}
VARIANTS = {
    "side_cradle": {"pose": "side", "height": 0.95, "wood": "#7a6048", "brand": "PORT", "rust": 0.5, "seed": 4},
    "half_planter": {"pose": "half", "height": 0.9, "girth": 0.8, "wood": "#786a5a", "rust": 0.7, "age": 0.8, "seed": 11},
    "cafe_table": {"pose": "table", "height": 0.98, "girth": 0.7, "hoops": 8, "wood": "#927c62", "tap": "none", "rust": 0.15,
                   "age": 0.3, "seed": 7},
    "box_planter": {"pose": "half", "planting": "box", "height": 1.0, "wood": "#6c5a48", "hoops": 8, "rust": 0.5, "seed": 5},
    "old_keg": {"pose": "stand", "height": 0.64, "girth": 0.82, "bulge": 0.24, "hoops": 4, "wood": "#8b8272", "rust": 0.95,
                "age": 1.0, "tap": "none", "brand": "IX", "seed": 23},
}




# ---------------------------------------------------------------------------------------------
# small local mesh builder: explicit verts / faces with metre UVs and per-vertex base masks
# ---------------------------------------------------------------------------------------------
class _MB:
    def __init__(self):
        self.v, self.vuv, self.g, self.w, self.s = [], [], [], [], []
        self.f, self.fuv, self.mi = [], [], []

    def vert(self, co, uv=(0.0, 0.0), g=0.0, w=0.0, s=0.0):
        self.v.append((float(co[0]), float(co[1]), float(co[2])))
        self.vuv.append((float(uv[0]), float(uv[1])))
        self.g.append(float(g))
        self.w.append(float(w))
        self.s.append(float(s))
        return len(self.v) - 1

    def face(self, idx, mi=0, uvf=None):
        idx = tuple(idx)
        self.f.append(idx)
        self.fuv.append([uvf(self.v[i]) for i in idx] if uvf else [self.vuv[i] for i in idx])
        self.mi.append(mi)

    def obj(self, name, mats, smooth_angle=40.0):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update(calc_edges=True)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.array([c for uvs in self.fuv for uv in uvs for c in uv], dtype=np.float32))
        me.polygons.foreach_set("material_index", np.array(self.mi, dtype=np.int32))
        for m in mats:
            me.materials.append(m)
        for nm, vals in (("m_grime", self.g), ("m_wear", self.w), ("m_special", self.s)):
            me.attributes.new(nm, 'FLOAT', 'POINT').data.foreach_set("value", np.clip(np.array(vals, dtype=np.float32), 0.0, 1.0))
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        o["tsq_uv"] = "keep"
        o["tsq_smooth"] = float(smooth_angle)
        return o


def _rows(musts, wants, sep, max_gap):
    """z stations: all `musts`, the `wants` that are not crowding another row, filled to max_gap."""
    zs = sorted(set(round(z, 5) for z in musts))
    lo, hi = zs[0], zs[-1]
    for z in wants:
        if lo < z < hi and all(abs(z - q) > sep for q in zs):
            zs.append(z)
    zs.sort()
    out = [zs[0]]
    for a, b in zip(zs, zs[1:]):
        n = int(ceil((b - a) / max_gap - 1e-6))
        out += [a + (b - a) * i / n for i in range(1, n)] + [b]
    return out


def _staves(name, rad, zs, edges, st, joints, th, Rb, top, bot, zin_top, zin_bot, cb, mats, wfn, shelf=(), swell=1.0):
    """Separate staves. Section per row: groove bottom, outer face (edge, the stave's own columns, edge - all
    on the hoop circle), groove bottom; plus the inside face where it can be seen (above the head / soil).
    Grain (U) runs along z. wfn(i, column, a, z) -> (grime, wear, special) paints the weathering per vertex.
    Levels: shelf = [(z, faces up)] - at each a ring from inside the wood half way to the axis, one face to each
    stave face: the head lies on it. (The staves are flats there and the head a round of few sides: without it
    one looks past the head's rim into the cask.)  swell: lod 2, the radius of the faces' edges x this."""
    mb = _MB()
    n, nz = len(st), len(zs)
    z_lo, z_hi = zs[0], zs[-1]
    Lv = lod()                                                    # levels: 1 = one flat face to a stave, 2 = to two
    if Lv == 2 and swell != 1.0:
        rad = (lambda f: lambda z: f(z) * swell)(rad)
    fs = list(range(0, n, 2))
    for i in (range(n) if Lv < 2 else fs):
        s = st[i]
        a0, a1 = edges[i], edges[i + 1 if Lv < 2 else min(i + 2, n)]
        am = 0.5 * (a0 + a1)
        jl, jr = joints[i], joints[(i + 1) % n]
        dl = min(s["dr"], st[i - 1]["dr"]) - jl["d"]              # groove bottoms are shared with the neighbours
        dr = min(s["dr"], st[(i + 1) % n]["dr"]) - jr["d"]
        offs = s["cols"] if Lv == 0 else []
        nf = len(offs) + 2 if Lv < 2 else 0                       # face columns
        grid = []
        if Lv:                                                    # the faces on either side
            sa, sb = (st[i - 1], st[(i + 1) % n]) if Lv == 1 else (st[fs[fs.index(i) - 1]], st[fs[(fs.index(i) + 1) % len(fs)]])
        if Lv == 2:
            # far off the faces have no grooves between them: each meets its neighbours on a whole edge (radius and
            # stave end of the lower of the two), or one would look between two faces into the cask from the side
            ends = [(a0, min(s["dr"], sa["dr"]), min(s["et"], sa["et"]), min(s["eb"], sa["eb"])),
                    (a1, min(s["dr"], sb["dr"]), min(s["et"], sb["et"]), min(s["eb"], sb["eb"]))]
            for k, z in enumerate(zs):
                r = rad(z)
                the = th * 0.5 if (top == "chime" and k == nz - 1) or (bot == "chime" and k == 0) else th
                row = [None] * 4
                for j, (a, d, et, eb) in enumerate(ends):
                    zz = z
                    if top == "chime" and z >= z_hi - cb - 1e-6:
                        zz += et
                    if bot == "chime" and z <= z_lo + cb + 1e-6:
                        zz -= eb
                    g, w, sp = wfn(i, 0, a, z)
                    row[j + 1] = mb.vert(((r + d) * cos(a), (r + d) * sin(a), zz), (zz + s["uo"], (a - am) * Rb + s["vo"]), g, w, sp)
                    if z >= zin_top - 1e-6 or z <= zin_bot + 1e-6:
                        ri = r + d - the
                        row[3 * j] = mb.vert((ri * cos(a), ri * sin(a), zz),
                                             (zz + s["uo"], (a - am) * Rb + s["vo"] + (2 * j - 1) * the), max(s["g"], 0.3), s["w"])
                grid.append(row)
        for k, z in enumerate(zs if Lv < 2 else ()):
            r = rad(z)
            zz, the = z, th
            if top == "chime" and z >= z_hi - cb - 1e-6:
                zz += s["et"]
            if bot == "chime" and z <= z_lo + cb + 1e-6:
                zz -= s["eb"]
            if (top == "chime" and k == nz - 1) or (bot == "chime" and k == 0):
                the = th * 0.5                                    # chime bevel on the inside of the stave end
            R = r + s["dr"]
            angs = [a0, a0 + jl["hw"] / r] + [am + o for o in offs] + [a1 - jr["hw"] / r, a1]
            rads = [r + dl] + [R] * nf + [r + dr]
            zl = zr = zz
            ril = rir = R - the
            if Lv == 1:
                # lod 1 has no sides in the joints: there the stave ends and the inside faces take the lower of the
                # two neighbours, as the groove bottoms do, so that nothing stands open between two staves
                zl = zr = z
                if top == "chime" and z >= z_hi - cb - 1e-6:
                    zl, zr = z + min(s["et"], sa["et"]), z + min(s["et"], sb["et"])
                if bot == "chime" and z <= z_lo + cb + 1e-6:
                    zl, zr = zl - min(s["eb"], sa["eb"]), zr - min(s["eb"], sb["eb"])
                ril, rir = r + min(s["dr"], sa["dr"]) - the, r + min(s["dr"], sb["dr"]) - the
            row = [None] * (nf + 4)
            for j in range(nf + 2):
                a = angs[j]
                g, w, sp = wfn(i, max(min(max(j - 1, 0), nf - 1), 0), a, z)
                zj = zl if j == 0 else zr if j == nf + 1 else zz
                row[j + 1] = mb.vert((rads[j] * cos(a), rads[j] * sin(a), zj), (zj + s["uo"], (a - am) * Rb + s["vo"]), g, w, sp)
            if z >= zin_top - 1e-6 or z <= zin_bot + 1e-6:
                gi = max(s["g"], 0.3)
                row[0] = mb.vert((ril * cos(a0), ril * sin(a0), zl), (zl + s["uo"], (a0 - am) * Rb + s["vo"] - the), gi, s["w"])
                row[nf + 3] = mb.vert((rir * cos(a1), rir * sin(a1), zr), (zr + s["uo"], (a1 - am) * Rb + s["vo"] + the), gi, s["w"])
            grid.append(row)
        for k in range(nz - 1):
            A, B = grid[k], grid[k + 1]
            for j in range(1, nf + 2):
                mb.face((A[j], A[j + 1], B[j + 1], B[j]), s["mi"])
            if A[0] is not None and B[0] is not None:
                if Lv == 0:                                       # (the sides in the joints: hidden by the neighbours)
                    mb.face((A[0], A[1], B[1], B[0]), s["mi"])
                    mb.face((A[nf + 2], A[nf + 3], B[nf + 3], B[nf + 2]), s["mi"])
                mb.face((A[nf + 3], A[0], B[0], B[nf + 3]), s["mi"])
        for zq, up in shelf:
            r = rad(zq)
            q = [mb.vert(((r + d - 0.5 * th) * cos(a), (r + d - 0.5 * th) * sin(a), zq), (zq + s["uo"], (a - am) * Rb + s["vo"]),
                         max(s["g"], 0.3), s["w"]) for a, d in ((a0, min(s["dr"], sa["dr"])), (a1, min(s["dr"], sb["dr"])))]
            q += [mb.vert((0.5 * (r - th) * cos(a), 0.5 * (r - th) * sin(a), zq), (zq + s["uo"] - 0.5 * r, (a - am) * Rb + s["vo"]),
                          max(s["g"], 0.3), s["w"]) for a in (a1, a0)]
            mb.face(q if up else q[::-1], s["mi"])
        eo = (s["uo"] * 0.37, s["vo"] * 0.41)
        end_uv = lambda c, eo=eo: (c[0] + eo[0], c[1] + eo[1])
        if top in ("chime", "cut"):
            mb.face(grid[-1], 2, end_uv)
        if bot == "chime":
            mb.face(list(reversed(grid[0])), 2, end_uv)
    return mb.obj(name, mats, 60.0)


def _band(name, prof, angs, Rb, m, off=None):
    """Levels: a hoop band [(radius, z), ...] as flats from angle to angle (angs: once round, last = first + 2 pi).
    off: {row of prof: per angle, added to the radius} (the staves' own step there)."""
    hb = _MB()
    k = len(angs) - 1
    off = off or {}
    ring = [[hb.vert(((r + d) * cos(a), (r + d) * sin(a), z)) for a, d in zip(angs[:k], off.get(j) or [0.0] * k)]
            for j, (r, z) in enumerate(prof)]
    for (A, (ra, za)), (B, (rb, zb)) in zip(zip(ring, prof), zip(ring[1:], prof[1:])):
        for q in range(k):
            hb.face((A[q], A[(q + 1) % k], B[(q + 1) % k], B[q]))
            hb.fuv[-1] = [(angs[q] * Rb, za), (angs[q + 1] * Rb, za), (angs[q + 1] * Rb, zb), (angs[q] * Rb, zb)]
    return hb.obj(name, [m], 35.0)


def _fewer(ms, mid=(2, 1.38), far=(4, 1.9)):
    """Levels: every n-th of the stamped things (matrices), that much larger - the same leaf on fewer leaves.
    Every one is drawn at every level; only what is stamped differs."""
    Lv = lod()
    if Lv == 0:
        return ms
    n, k = far if Lv == 2 else mid
    return [M_ @ Matrix.Scale(k, 4) for M_ in ms[::n]]


def _disc(name, R, nb, r, mats, age, thick=0.0, jd=0.004, ease=0.0, dphi=12.0, phi_min=4.0, nrow=8, tone=0.12, rim_g=0.0,
          stains=()):
    """Round panel of nb parallel boards (joints along Y, grain along Y) in the z = 0 plane facing +Z.
    thick = 0: top faces + joint chamfers only (a barrel head let into the croze - its rim is hidden).
    thick > 0: free-standing top with eased rim, end grain on the board ends, and an underside.
    tone = largest per-board grime / wear shift (kept small: boards of one head are cut from the same oak).
    rim_g = dirt gathered in the croze. stains = [(fn(X, Y) -> 0..1, (x0, x1, y0, y1), cell)]: marks burnt /
    soaked into the boards (brand, drip, mug ring), laid as a fine-grid overlay that carries the board's own
    material, UVs and tone so only the stain itself shows; it breaks and fades at every board joint.
    mats = [face wood, end grain]. Returns (object, board edge x list, highest board offset)."""
    mb = _MB()
    Lv = lod()
    if Lv:                                        # levels: fewer rows and columns to a board (the rim of a head is hidden)
        nrow = (4, 2)[Lv - 1]
        dphi = dphi * ((2.0, 5.0) if thick <= 0 else (1.5, 3.0))[Lv - 1]
    Rg = R - (ease if Lv < 2 else 0.0)            # (far off there is no eased arris)
    kx = R / Rg
    wd = [r.vary(1.0, 0.2) for _ in range(nb)]
    if nb > 2:
        wd[0] *= 0.82
        wd[-1] *= 0.82
    tot = sum(wd)
    xe = [-R]
    for w_ in wd:
        xe.append(xe[-1] + 2 * R * w_ / tot)
    xe[-1] = R
    xlim = Rg * cos(radians(phi_min))
    c = 0.0012 + 0.0012 * age if Lv < 2 else 0.0  # (far off there are no joints: the boards meet edge to edge)
    sv = []
    for q in range(nrow):                                         # rows gathered towards the board ends (croze dirt)
        u = -1.0 + 2.0 * q / (nrow - 1)
        sv.append((1.0 if u >= 0 else -1.0) * (1.0 - (1.0 - abs(u)) ** 1.5))

    def yg(x):
        return sqrt(max(Rg * Rg - x * x, 0.0))

    def yr(x):
        return sqrt(max(R * R - x * x, 0.0))

    top_dz = 0.0
    for b in range(nb):
        xa, xb = xe[b], xe[b + 1]
        jl, jr = b > 0, b < nb - 1
        xl = xa + c if jl else -xlim
        xr = xb - c if jr else xlim
        pa, pb = acos(clamp(xl / Rg, -1.0, 1.0)), acos(clamp(xr / Rg, -1.0, 1.0))
        nc = max(1, int(ceil(abs(pa - pb) / radians(dphi))))
        xs = [Rg * cos(pa + (pb - pa) * t / nc) for t in range(nc + 1)]
        xs[0], xs[-1] = xl, xr
        dz = r.jitter(0.0, 0.0009 * age)
        tilt = r.jitter(0.0, 0.006 * age)
        top_dz = max(top_dz, dz + abs(tilt) * (xb - xa) * 0.5)
        if Lv == 2:                                               # ... and lie in one plane, or one looks between them
            dz = tilt = 0.0
        g = r.u(0.25, 1.0) * tone * (0.4 + 0.6 * age) if r.chance(0.5) else 0.0
        w = r.u(0.25, 1.0) * tone if r.chance(0.35) else 0.0
        kb = r.u(0.72, 1.0)                                       # how well the brand took on this board
        uo, vo = r.u(0, 3), r.u(0, 3)
        xm = 0.5 * (xa + xb)

        def zt(x):
            return dz + tilt * (x - xm)

        def V(x, y, z):
            gv = g
            if rim_g > 0:
                gv = max(g, rim_g * smoothstep(0.8 * R, 0.985 * R, sqrt(x * x + y * y)))
            return mb.vert((x, y, z), (y + uo, x + vo), gv, w)

        cols = [[V(x, s * yg(x), zt(x)) for s in sv] for x in xs]
        for ci in range(nc):
            for ri in range(nrow - 1):
                mb.face((cols[ci][ri], cols[ci + 1][ri], cols[ci + 1][ri + 1], cols[ci][ri + 1]))
        for fn, (bx0, bx1, by0, by1), step in stains:
            if Lv == 2:                                           # far off: no mark can be read
                continue
            if Lv == 1:                                           # the mark stays, on a coarser grid
                step = step * 1.8
            a_, b_ = max(xl + 0.0015, bx0), min(xr - 0.0015, bx1)
            if b_ - a_ < 0.004 or by1 - by0 < 0.004:
                continue
            nx = max(1, int(ceil((b_ - a_) / step)))
            ny = max(1, int(ceil((by1 - by0) / step)))
            X, Y = np.meshgrid(np.linspace(a_, b_, nx + 1), np.linspace(by0, by1, ny + 1), indexing="ij")
            Mv = np.clip(fn(X.ravel(), Y.ravel()), 0.0, 1.0).reshape(X.shape) * kb
            if jl:
                Mv *= np.clip((X - xa) / 0.007, 0.5, 1.0)
            if jr:
                Mv *= np.clip((xb - X) / 0.007, 0.5, 1.0)
            Mv[np.hypot(X, Y) > Rg - 0.005] = 0.0
            keep = np.maximum(np.maximum(Mv[:-1, :-1], Mv[1:, :-1]), np.maximum(Mv[:-1, 1:], Mv[1:, 1:])) > 0.06
            vid = {}

            def dv(ix, iy):
                key = (int(ix), int(iy))
                if key not in vid:
                    x, y, m = float(X[key]), float(Y[key]), float(Mv[key])
                    gv = g
                    if rim_g > 0:
                        gv = max(g, rim_g * smoothstep(0.8 * R, 0.985 * R, sqrt(x * x + y * y)))
                    vid[key] = mb.vert((x, y, zt(x) + 0.0005), (y + uo, x + vo), gv + (1.0 - gv) * m, w * (1.0 - m))
                return vid[key]

            for ix, iy in zip(*np.nonzero(keep)):
                mb.face((dv(ix, iy), dv(ix + 1, iy), dv(ix + 1, iy + 1), dv(ix, iy + 1)))
        zb = -thick if thick > 0 else None
        GL = GR = None
        if jl and Lv == 0:
            GL = [V(xa, s * yg(xa), zt(xa) - c) for s in sv]
            SL = [V(xa, s * (yr(xa) if thick > 0 else yg(xa)), zb if thick > 0 else zt(xa) - jd) for s in sv]
            for ri in range(nrow - 1):
                mb.face((GL[ri], cols[0][ri], cols[0][ri + 1], GL[ri + 1]))
                mb.face((SL[ri], GL[ri], GL[ri + 1], SL[ri + 1]))
        if jr and Lv == 0:
            GR = [V(xb, s * yg(xb), zt(xb) - c) for s in sv]
            SR = [V(xb, s * (yr(xb) if thick > 0 else yg(xb)), zb if thick > 0 else zt(xb) - jd) for s in sv]
            for ri in range(nrow - 1):
                mb.face((cols[-1][ri], GR[ri], GR[ri + 1], cols[-1][ri + 1]))
                mb.face((GR[ri], SR[ri], SR[ri + 1], GR[ri + 1]))
        if Lv == 1:                                               # lod 1: the joint is one V, no chamfer (far off: none)
            for on, x_, col in ((jl, xa, cols[0]), (jr, xb, cols[-1])):
                if on:
                    SV_ = [V(x_, s * yg(x_), zt(x_) - jd) for s in sv]
                    for ri in range(nrow - 1):
                        mb.face((SV_[ri], col[ri], col[ri + 1], SV_[ri + 1]) if x_ == xa else (col[ri], SV_[ri], SV_[ri + 1], col[ri + 1]))
        if thick <= 0:
            continue
        eg_uv = lambda co: (co[0] + vo, co[2] + uo)               # end grain on the sawn board ends
        lg_uv = lambda co: (co[1] + uo, co[2] + vo)               # long grain on the cants' curved edge
        for sgn, ri in ((-1, 0), (1, nrow - 1)):
            bl = []
            if GL is not None:
                bl.append((GL[ri], xa, sgn * yr(xa), zt(xa) - c))
            for ci, x in enumerate(xs):
                if Lv == 1 and ((ci == 0 and jl) or (ci == nc and jr)):
                    # lod 1: the rim runs on to the joint line, where the neighbour's rim and the underside begin
                    xj = xa if ci == 0 else xb
                    bl.append((cols[ci][ri], xj, sgn * yr(xj), zt(x)))
                    continue
                bl.append((cols[ci][ri], x * kx, sgn * yg(x) * kx, zt(x)))
            if GR is not None:
                bl.append((GR[ri], xb, sgn * yr(xb), zt(xb) - c))
            # (far off the eased arris is left out: the rim drops straight from the face)
            E = [mb.vert((x, y, z - ease), (y + uo, x + vo), g, w) for (_, x, y, z) in bl] if Lv < 2 else [b_[0] for b_ in bl]
            Bv = [mb.vert((x, y, zb), (y + uo, x + vo), g, w) for (_, x, y, z) in bl]
            for q in range(len(bl) - 1):
                endg = abs(bl[q][2] + bl[q + 1][2]) * 0.5 > 0.55 * R
                mi, uvf = (1, eg_uv) if endg else (0, lg_uv)
                if sgn > 0:
                    if Lv < 2:
                        mb.face((bl[q][0], bl[q + 1][0], E[q + 1], E[q]))
                    mb.face((E[q], E[q + 1], Bv[q + 1], Bv[q]), mi, uvf)
                else:
                    if Lv < 2:
                        mb.face((bl[q + 1][0], bl[q][0], E[q], E[q + 1]))
                    mb.face((E[q + 1], E[q], Bv[q], Bv[q + 1]), mi, uvf)
        for is_left, col, x0 in ((True, cols[0], xl), (False, cols[-1], xr)):
            if (is_left and jl) or (not is_left and jr):
                continue
            E = [mb.vert((x0 * kx, s * yg(x0) * kx, zt(x0) - ease), (0, 0), g, w) for s in sv] if Lv < 2 else list(col)
            Bv = [mb.vert((x0 * kx, s * yg(x0) * kx, zb), (0, 0), g, w) for s in sv]
            for ri in range(nrow - 1):
                if is_left:
                    if Lv < 2:
                        mb.face((col[ri], col[ri + 1], E[ri + 1], E[ri]))
                    mb.face((E[ri], E[ri + 1], Bv[ri + 1], Bv[ri]), 0, lg_uv)
                else:
                    if Lv < 2:
                        mb.face((col[ri + 1], col[ri], E[ri], E[ri + 1]))
                    mb.face((E[ri + 1], E[ri], Bv[ri], Bv[ri + 1]), 0, lg_uv)
        bx = [xa if jl else -xlim * kx] + [x * kx for x in xs[1:-1]] + [xb if jr else xlim * kx]
        under = [[mb.vert((x, s * yr(x), zb), (s * yr(x) + uo + 0.7, x + vo), g, w) for s in sv] for x in bx]
        for ci in range(len(bx) - 1):
            for ri in range(nrow - 1):
                mb.face((under[ci][ri], under[ci][ri + 1], under[ci + 1][ri + 1], under[ci + 1][ri]))
    return mb.obj(name, mats, 35.0), xe, top_dz


def _text_mask(txt, font, width, hmax):
    """Rasteriser for a burnt mark: returns (cover(Q (N,2), blur radius) -> 0..1, half size (x, y)) of `txt`
    scaled to fit width x hmax and centred on the origin. The soft edge comes from averaging the glyph
    coverage over a small disc, exactly what a hot iron does to the wood around it."""
    with lod_as(0):                               # (the glyphs are only measured: the full curves, one mark at every level)
        o = text("brand_tmp", txt, size=0.1, depth=0.002, rot=(0, 0, 0), font=font, spacing=1.12)
    me = o.data
    me.calc_loop_triangles()
    co = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    tri = np.array([t.vertices[:] for t in me.loop_triangles if t.normal.z > 0.5], dtype=np.int64)
    delete(o)
    P = co[:, :2]
    lo, hi = P.min(0), P.max(0)
    k = min(width / max(hi[0] - lo[0], 1e-6), hmax / max(hi[1] - lo[1], 1e-6))
    P = (P - 0.5 * (lo + hi)) * k
    A, B, C = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    half = 0.5 * (hi - lo) * k

    def inside(Q):
        out = np.zeros(len(Q))
        for s in range(0, len(Q), 3000):
            q = Q[s:s + 3000][:, None, :]
            d1 = (B[None, :, 0] - A[None, :, 0]) * (q[:, :, 1] - A[None, :, 1]) - (B[None, :, 1] - A[None, :, 1]) * (q[:, :, 0] - A[None, :, 0])
            d2 = (C[None, :, 0] - B[None, :, 0]) * (q[:, :, 1] - B[None, :, 1]) - (C[None, :, 1] - B[None, :, 1]) * (q[:, :, 0] - B[None, :, 0])
            d3 = (A[None, :, 0] - C[None, :, 0]) * (q[:, :, 1] - C[None, :, 1]) - (A[None, :, 1] - C[None, :, 1]) * (q[:, :, 0] - C[None, :, 0])
            ins = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
            out[s:s + 3000] = ins.any(1)
        return out

    def cover(Q, rad_):
        offs = [(0.0, 0.0)] + [(rad_ * cos(t * pi / 4), rad_ * sin(t * pi / 4)) for t in range(8)] + \
               [(0.5 * rad_ * cos(t * pi / 2 + 0.4), 0.5 * rad_ * sin(t * pi / 2 + 0.4)) for t in range(4)]
        acc = np.zeros(len(Q))
        for ox, oy in offs:
            acc += inside(Q + np.array([ox, oy]))
        return acc / len(offs)

    return cover, half


def _ring_fn(cx, cy, rr, ph, k=0.62):
    """Mug ring: a thin dark annulus, stronger on one side where the drip ran down the tankard."""
    def fn(X, Y):
        dx, dy = X - cx, Y - cy
        d = np.hypot(dx, dy)
        return k * np.exp(-((d - rr) / 0.0038) ** 2) * (0.55 + 0.45 * np.cos(np.arctan2(dy, dx) - ph))
    return fn


def _frame(origin, x, z):
    """4x4 with the given origin, +X and +Z directions (Y completes the right-handed frame)."""
    x, z = Vector(x).normalized(), Vector(z).normalized()
    y = z.cross(x).normalized()
    z = x.cross(y).normalized()
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0], m[i][1], m[i][2], m[i][3] = x[i], y[i], z[i], origin[i]
    return m


def _zframe(origin, z, spin=0.0):
    """4x4 whose +Z points along `z` (for stamping things that grow out of a surface)."""
    z = Vector(z).normalized()
    ref = Vector((0, 0, 1)) if abs(z.z) < 0.9 else Vector((1, 0, 0))
    x = ref.cross(z).normalized()
    return _frame(origin, x, z) @ Matrix.Rotation(spin, 4, 'Z')


def _put(o, M):
    o.matrix_basis = M @ o.matrix_basis
    return o


def _timber(name, L, w, h, cham, M, wood, endg, bevel=0.004, both=False):
    """Squared timber along local X (width w across Y, height h up from z = 0) with sawn chamfers on the
    upper end corners (both = also the lower ones), eased arrises and end grain. M places it."""
    c = cham
    pts = [(-L / 2, c if both else 0.0), (-L / 2 + c, 0.0), (L / 2 - c, 0.0), (L / 2, c)] if both else [(-L / 2, 0.0), (L / 2, 0.0)]
    pts += [(L / 2, h - c), (L / 2 - c, h), (-L / 2 + c, h), (-L / 2, h - c)]
    o = extrude_profile(name, pts, w, loc=(0, w / 2, 0), rot=(90, 0, 0), bevel=bevel)
    assign(o, wood)
    assign_faces(o, endg, lambda c_, n: abs(n.x) > 0.6)
    uv_grain(o, 'X')
    smooth(o, 35)
    return _put(o, M)


def _tap(kind, k, metal, wood):
    """Spigot in its own frame: origin on the wood surface, +X out of the cask, +Z up."""
    S = lambda pts: [tuple(c * k for c in pt) for pt in pts]
    body = lathe("tap_body", S([(0.0095, -0.014), (0.0105, 0.003), (0.0155, 0.004), (0.0162, 0.0065), (0.0155, 0.009), (0.0112, 0.011),
                                (0.0104, 0.03), (0.0135, 0.034), (0.0142, 0.042), (0.0135, 0.05), (0.0104, 0.054), (0.0088, 0.064)]),
                 segments=lod_pick(10, 6, 4), rot=(0, 90, 0))
    spout = sweep("tap_spout", S([(0.06, 0, 0), (0.08, 0, 0), (0.096, 0, -0.007), (0.103, 0, -0.024), (0.104, 0, -0.047)]),
                  0.0078 * k, sides=lod_pick(8, 5, 3), step=0.0065 * k)
    lip = lod_drop(torus("tap_lip", 0.0082 * k, 0.0021 * k, loc=(0.104 * k, 0, -0.0455 * k), major_seg=8, minor_seg=4))
    key = lathe("tap_key", S([(0.0, -0.019), (0.006, -0.019), (0.0078, -0.016), (0.0078, 0.0155), (0.0108, 0.0175), (0.0108, 0.0205),
                              (0.0055, 0.0235), (0.005, 0.031), (0.0, 0.031)]), segments=lod_pick(8, 5, 3), loc=(0.042 * k, 0, 0))
    bow = [(0.0, -0.0045), (0.012, -0.0058), (0.021, -0.0105), (0.0265, -0.0085), (0.0285, 0.0), (0.0265, 0.0085), (0.021, 0.0105),
           (0.012, 0.0058), (0.0, 0.0045)]
    bow = bow + [(-x, -y) for x, y in bow[1:-1]]                   # butterfly key, point-symmetric
    handle = extrude_profile("tap_handle", S(bow), 0.0062 * k, loc=(0.0389 * k, 0, 0.039 * k), rot=(90, 0, 90), bevel=0.0012 * k)
    smooth(handle, 50)
    parts = [body, spout, lip, key, handle]
    assign(parts, metal if kind == "brass" else wood)
    return join(parts, "tap")


def build(p, rng):
    H = p.height
    pose = p.pose
    sc = clamp(H / 0.92, 0.7, 1.3)
    age, rust = p.age, p.rust
    upright = pose != "side"
    Rb = H * p.girth / 2.0                      # outer radius at the bilge
    Rh = Rb * (1.0 - p.bulge)                   # outer radius at the stave ends
    th = 0.027 * sc                             # stave thickness
    cd = 0.042 * sc                             # chime depth: recess of the head below the stave ends
    cb = 0.013 * sc                             # chime bevel
    Ztop = H * 0.5 if pose == "half" else H

    def rad(z):
        s = 2.0 * z / H - 1.0
        return Rb - (Rb - Rh) * s * s

    # ------------------------------------------------------------------ materials
    cw = hex_rgb(p.wood)
    lum = 0.3 * cw[0] + 0.55 * cw[1] + 0.15 * cw[2]
    grey = (lum * 1.03, lum, lum * 0.94)

    def tone(k, desat=0.0):
        """the oak tone, k x brighter and pulled towards silver-grey by desat"""
        return tuple(min(0.9, (c * (1.0 - desat) + g_ * desat) * k) for c, g_ in zip(cw, grey))

    oak_a = mat("wood_oak", tone(1.0), rough=0.9)                       # sound oak
    oak_b = mat("wood_weathered", tone(0.92, 0.4), rough=1.05)          # silvered staves: same tone within 10 %, drier
    endg = mat("wood_endgrain", tone(0.84, 0.15))
    iron = mat("iron_rusty")                    # hoop faces: base = rust scale, grime + wear = dark oiled iron
    # hoop arrises: rubbed back to bare grey metal on a sound hoop (the edge that catches the light), rusted through on an old one
    steel = iron if rust > 0.6 else mat("lead")
    brass = mat("brass", rough=1.5)

    # ------------------------------------------------------------------ hoops (z stations first: staves need them)
    fr = {4: [0.04, 0.27], 6: [0.04, 0.16, 0.31], 8: [0.04, 0.125, 0.225, 0.335]}[int(p.hoops)]
    r_h = rng.fork("hoops")
    hoops = []                                   # (z centre, width)
    for side in (0, 1):
        for j, f in enumerate(fr):
            w = (0.046 if j == 0 else 0.036) * sc * r_h.vary(1.0, 0.06)
            z = f * H + (r_h.jitter(0, 0.005 * age * sc) if j else 0.0)
            if j == 0:
                z = max(z, w / 2 + 0.007)        # the chime hoop always sits 7 mm inside the stave end
            hoops.append((z if side == 0 else H - z, w))
    if pose == "half":                           # only the sawn half barrel loses the hoops above the cut
        hoops = [h for h in hoops if h[0] + h[1] / 2 < Ztop - 0.01]
    t_h = 0.003                                  # hoop iron thickness

    # ------------------------------------------------------------------ stave layout
    if pose == "side":
        a_up = radians(90)
        a_bung = a_up + radians(rng.jitter(0, 7))
        a_lap = a_up - radians(48)
        a_tap = -a_up
    else:
        a_front = radians(-90)
        a_bung = a_front + radians(17 + rng.jitter(0, 4))
        a_lap = a_front - radians(8)
        a_tap = a_front - radians(12)
    r_s = rng.fork("staves")
    n_st = max(14, int(round(2 * pi * Rb / (0.084 * sc ** 0.5))))
    wd = [r_s.vary(1.0, 0.3) for _ in range(n_st)]
    wd[0] = 1.5                                  # the bung stave is the widest
    tot = sum(wd)
    edges = [a_bung - wd[0] / tot * pi]
    for w in wd:
        edges.append(edges[-1] + w / tot * 2 * pi)
    mids = [0.5 * (edges[i] + edges[i + 1]) for i in range(n_st)]
    drm = 0.0003 + 0.0011 * age
    st, joints = [], []
    for i in range(n_st):
        proud = r_s.chance(0.12 * age)
        st.append({"dr": r_s.jitter(0, drm) if not proud else r_s.pick([-1, 1]) * drm * 1.0,
                   "et": r_s.jitter(0, 0.0018 * age) if pose in ("stand", "side") else 0.0,
                   "eb": r_s.jitter(0, 0.0018 * age) if pose == "side" else 0.0,
                   "g": r_s.u(0.03, 0.13) * (0.4 + 0.6 * age) if r_s.chance(0.45) else 0.0,
                   "w": r_s.u(0.04, 0.14) if r_s.chance(0.3) else 0.0,
                   "mi": 1 if r_s.chance(0.2 + 0.45 * age) else 0,
                   "uo": r_s.u(0, 4), "vo": r_s.u(0, 4), "cols": [0.0]})
        d = (0.0022 + 0.0022 * age * r_s.u(0.3, 1.0)) * sc
        joints.append({"d": d, "hw": d * r_s.u(0.75, 1.05)})

    def renv(z):
        return rad(z) + drm

    # ------------------------------------------------------------------ tap position; a tapped upright cask stands on bearers
    has_tap = p.tap != "none" and pose != "half"
    has_bung = pose != "half"
    kt = sc * (1.0 if p.tap == "brass" else 1.28)
    tap_i, zt_l, lift = None, 0.0, 0.0
    if has_tap and upright:
        tap_i = min(range(1, n_st), key=lambda i: abs(((mids[i] - a_tap + pi) % (2 * pi)) - pi))
        (za, wa), (zb2, wb) = sorted(hoops)[:2]
        zt_l = min(max(0.5 * (za + wa / 2 + zb2 - wb / 2), 0.085 * H), zb2 - wb / 2 - 0.021 * kt)
        lift = max(0.1 * sc, 0.19 - (zt_l - 0.0475 * kt))            # spout mouth clears the paving by 19 cm
    if pose == "side":
        zc = 0.15 * sc + Rb + 0.005
        M = Matrix.Translation((0, H / 2, zc)) @ Matrix.Rotation(radians(90), 4, 'X')
    else:
        M = Matrix.Translation((0, 0, lift))
    Mr = np.array(M.to_3x3())
    Mt = np.array(M.translation)

    def lmask(fn):
        """mask written in the barrel's own frame (axis = z, base at z = 0) -> world-space paint() callable"""
        return lambda co, n: fn((co - Mt) @ Mr, n @ Mr)

    # extra columns in the bung and tap staves so the drip runs under them are 2-4 cm wide
    hw0 = 0.5 * (edges[1] - edges[0])
    d0 = min(0.016 * sc / Rb, 0.4 * hw0)
    if has_bung:
        st[0]["cols"] = [-2 * d0, -d0, 0.0, d0, 2 * d0]
    if tap_i is not None:
        d1 = min(0.015 * sc / Rb, 0.5 * 0.5 * (edges[tap_i + 1] - edges[tap_i]))
        st[tap_i]["cols"] = [-d1, 0.0, d1]

    # ------------------------------------------------------------------ rows of the stave mesh
    top = {"stand": "chime", "side": "chime", "half": "cut", "table": "chime"}[pose]
    bot = "chime" if (pose == "side" or lift > 0) else "ground"
    soil_d = 0.055 * sc
    zin_top = {"chime": H - cd - 0.006, "cut": Ztop - soil_d - 0.012}[top]
    zin_bot = cd + 0.006 if bot == "chime" else -1e9
    musts = [0.0, Ztop]

    def must(z):
        if 0.0 < z < Ztop and all(abs(z - q) > 0.004 for q in musts):
            musts.append(z)

    if top == "chime":
        must(H - cb)
    if bot == "chime":
        must(cb)
    wants = []
    spans = [(z - w / 2, z + w / 2) for z, w in hoops]
    for lo, hi in spans:                         # rows that carry the stain below (lying cask: either side of) each hoop
        must(lo)
        must(hi)
        if upright:
            wants += [lo - 0.03 * sc, lo - 0.075 * sc]
    bung_keys = (-0.031, -0.011, 0.011, 0.031)
    if has_bung:
        if upright:
            wants += [H / 2 + 0.04 * sc, H / 2 - 0.04 * sc, H / 2 - 0.08 * sc]
        else:
            for dk in bung_keys:
                must(H / 2 + dk * sc)
    if upright:
        wants += [0.034 * sc, 0.072 * sc]        # damp foot
    if tap_i is not None:
        wants += [zt_l + 0.012 * sc]
    wants = [q for q in wants if not any(lo - 0.003 < q < hi + 0.003 for lo, hi in spans)] + [0.5 * H]
    Lv = lod()
    if Lv == 0:
        zs = _rows(musts, wants, 0.012 * sc, 0.075 * H)
    else:
        # levels: the rows the outline and the weathering need - the stave ends, at lod 1 the lower edge of every
        # hoop (the stain starts there) and the damp foot - filled to a sixth / a quarter of the height
        mid = [lo for lo, hi in spans if 0.03 * sc < lo < Ztop - 0.03 * sc] if Lv == 1 else []
        foot = [0.072 * sc] if (upright and Lv == 1) else [0.1 * sc] if (upright and pose != "half") else []
        zs = _rows([0.0, Ztop] + mid, foot, 0.04 * sc, (0.16 if Lv == 1 else 0.27) * H)
    zin_top = max(z for z in zs if z <= zin_top + 0.002)      # the inside faces start on an existing row
    if bot == "chime":
        zin_bot = min(z for z in zs if z >= zin_bot - 0.002)

    # ------------------------------------------------------------------ weathering of the staves (cask frame, per vertex)
    RT = np.random.default_rng(int(p.seed) * 7919 + 11).random((n_st, 9, 6))
    r_w = rng.fork("stains")

    def runs(ncol):
        """drip runs down one stave: (strength, relative length) per face column"""
        tab = [[0.0, 0.0] for _ in range(ncol)]
        m = ncol // 2
        tab[m] = [1.0, r_w.u(0.75, 1.0)]
        for q in (m - 1, m + 1):
            tab[q] = [r_w.u(0.15, 0.35), r_w.u(0.3, 0.6)]
        if ncol >= 7:
            q = m + r_w.pick([-2, 2])
            tab[q] = [r_w.u(0.5, 0.75), r_w.u(0.3, 0.6)]
        return tab

    bung_runs = runs(7)
    tap_runs = runs(5)
    bung_rows = {dk: (s_, r_w.u(0.5, 1.0) * l_, r_w.u(0.5, 1.0) * l_)       # lying cask: runs around the belly (rad)
                 for dk, s_, l_ in zip(bung_keys, (0.0, 1.0, r_w.u(0.45, 0.8), 0.0), (0, 1.9, 1.3, 0))}
    damp = 0.35 + 0.65 * age if pose == "half" else 0.15 + 0.6 * age
    foot_k = 1.0 if lift == 0 else 0.55
    k_rust = 0.8 + 0.2 * rust
    k_bleach = clamp((age - 0.42) / 0.45)
    r_bung = 0.03 * sc

    def wfn(i, jc, a, z):
        s_ = st[i]
        R_ = RT[i, jc]
        Rs = RT[i, 8]                                   # per stave
        g, w, sp = s_["g"], s_["w"], 0.0
        for hk, (lo, hi) in enumerate(spans):           # iron-tannate stain: dark streaked band under every hoop
            rr = RT[i, jc, hk % 3]
            if upright:
                if z >= hi - 1e-5:
                    continue
                m = 1.0 if z > lo + 1e-5 else max(0.0, 1.0 - (lo - z) / ((0.06 + 0.05 * rr) * sc * (0.8 + 0.4 * rust))) ** 0.9
            else:                                       # lying: the rust creeps out either side and runs to the underside
                d = max(lo - z, z - hi, 0.0)
                low = 0.5 - 0.5 * sin(a)
                m = (0.35 + 0.65 * rr) * (0.55 + 0.45 * low) if d < 1e-4 else 0.0
            g = max(g, m * k_rust * (0.6 + 0.25 * R_[3]))
        if upright:
            zf = (0.1 + 0.07 * R_[4]) * sc * (1.0 if lift == 0 else 0.75)   # damp, green foot (rises above the chime hoop)
            m = max(0.0, 1.0 - z / zf)
            g = max(g, 0.92 * foot_k * m ** 0.7)
            sp = max(sp, foot_k * damp * m * (1.0 if R_[5] > 0.3 else 0.3))
            t = smoothstep(Ztop * 0.62, Ztop * 0.93, z)  # sun and rain bleach the upper third silver
        else:
            g = max(g, 0.3 * (0.5 - 0.5 * sin(a)) ** 3)
            t = smoothstep(0.2, 0.95, sin(a))
        w = max(w, 0.6 * t * k_bleach * (0.62 + 0.38 * Rs[0]) * (0.8 + 0.2 * R_[5]))
        roll = exp(-((z - H / 2) / (0.085 * H)) ** 2) * (0.08 + 0.2 * age) * (0.35 + 0.65 * Rs[1])
        w = max(w, roll)                                # the bilge is where a cask is rolled and chafed
        if has_bung:
            kb_ = 0.62 + 0.35 * age
            if upright and i == 0:                      # spill from the bung hole runs down the belly
                s0, ln = bung_runs[jc + 0] if jc < 7 else (0.0, 0.0)
                d = H / 2 - z
                if s0 > 0 and d > -0.02 * sc:
                    m = 1.0 if d < r_bung else max(0.0, 1.0 - (d - r_bung) / (ln * 0.42 * H)) ** 0.6
                    g = max(g, s0 * m * kb_)
            elif not upright:
                dzk = (z - H / 2) / sc
                da = (a - a_bung + pi) % (2 * pi) - pi
                for dk, (s0, lp, ln) in bung_rows.items():
                    if s0 > 0 and abs(dzk - dk) < 0.004:
                        m = max(0.0, 1.0 - abs(da) / (lp if da > 0 else ln)) ** 0.6
                        g = max(g, s0 * m * kb_)
        if tap_i is not None and i == tap_i:            # leak under the tap flange
            s0, ln = tap_runs[jc] if jc < 5 else (0.0, 0.0)
            d = zt_l - z
            if s0 > 0 and d > -0.004:
                g = max(g, 0.9 * s0 * max(0.0, 1.0 - d / (ln * 1.5 * zt_l + 0.02)) ** 0.5)
        return g, w, sp

    # (levels: nobody sees under a cask that stands on its bearers - the stave ends and the recess there are left open)
    open_foot = Lv > 0 and upright and bot == "chime"
    # lod 2 of the lying cask: its faces, two staves wide, lie up to a centimetre inside the round, and under the belly
    # one sees that much more of the ground between the saddles. Their edges stand out far enough to keep the girth
    # (the area of the section); the hoops go with them. A standing cask keeps its edges on the round
    kf = 1.0
    if Lv == 2 and pose == "side":
        qf = 2 * pi / len(range(0, n_st, 2))
        kf = (qf / sin(qf)) ** 0.5
    shelf = []
    if Lv:                                       # just under the boards of a head (at lod 1 under their joints too)
        dq = (0.006, 0.002)[Lv - 1]
        shelf = [(H - cd - dq, True)] * (top == "chime") + [(cd + dq, False)] * (bot == "chime" and not open_foot)
    staves = _staves("staves", rad, zs, edges, st, joints, th, Rb, top, "ground" if open_foot else bot, zin_top,
                     -1e9 if open_foot else zin_bot, cb, [oak_a, oak_b, endg], wfn, shelf, kf)
    barrel = [staves]                            # everything that moves with the cask

    def rface(z):
        """lod 2: the staves' faces at z (before their own step) - straight from row to row"""
        k = max(q for q in range(len(zs) - 1) if zs[q] <= z + 1e-9)
        return rad(zs[k]) + (rad(zs[k + 1]) - rad(zs[k])) * (z - zs[k]) / (zs[k + 1] - zs[k])

    # ------------------------------------------------------------------ hoops: tapered bands + riveted lap
    rivet = lathe("rivet", [(0.0078, 0.0), (0.0072, 0.0022), (0.0048, 0.0042), (0.0, 0.0052)], segments=7)
    laps = _MB()
    riv_m = []
    rims = []
    e_h, rim_h = 0.0012, 0.0075 * sc
    for hi_, (z, w) in enumerate(hoops):
        z0, z1 = z - w / 2, z + w / 2
        prof = [(renv(z0) - 0.002, z0), (renv(z0 + e_h) + t_h, z0 + e_h), (renv(z0 + rim_h) + t_h + 0.0004, z0 + rim_h),
                (renv(z1 - rim_h) + t_h + 0.0004, z1 - rim_h), (renv(z1 - e_h) + t_h, z1 - e_h), (renv(z1) - 0.002, z1)]
        spin = r_h.u(0, 6.28)
        if Lv == 0:
            for nm_, pr_, grp in (("hoop", prof[2:4], barrel), ("hoop_rim_a", prof[0:3], rims), ("hoop_rim_b", prof[3:6], rims)):
                hp = lathe(f"{nm_}{hi_}", pr_, segments=32, smooth_angle=35)
                hp.rotation_euler.z = spin
                grp.append(hp)
        elif Lv == 1:
            # levels: the hoop bends from stave to stave as the iron does - one flat to each, so it lies on them
            # (a turned ring of fewer sides than there are staves cuts into the boards); its bright arrises stay -
            # on a standing cask only one: the other looks at the ground from below eye height
            for nm_, pr_, grp in (("hoop", prof[2:4], barrel), ("hoop_rim_a", [prof[0], prof[2]], rims), ("hoop_rim_b", [prof[3], prof[5]], rims)):
                if not (upright and nm_ == ("hoop_rim_a" if prof[0][1] < prof[5][1] else "hoop_rim_b")):
                    grp.append(_band(f"{nm_}{hi_}", pr_, edges, Rb, iron))
        else:
            pr_, off = [(renv(z0) * kf + t_h, z0), (renv(z1) * kf + t_h, z1)], None
            if pose == "side":
                # the staves are flat from row to row there, inside the turned outline the band keeps: it stands off
                # them, and from either end of a lying cask one looks under it at the back of the iron. Its two
                # edges go down into the wood (on a standing cask nobody stands where he could look along the staves)
                dq = [min(st[i]["dr"], st[i - 2 if i else (n_st - 1) // 2 * 2]["dr"]) for i in range(0, n_st, 2)]
                pr_ = [(rface(z0 - 0.002) * kf - 0.0006, z0 - 0.002)] + pr_ + [(rface(z1 + 0.002) * kf - 0.0006, z1 + 0.002)]
                off = {0: dq, 3: dq}
            barrel.append(_band(f"hoop{hi_}", pr_, edges[::2] + ([edges[-1]] if n_st % 2 else []), Rb, iron, off))
        # lap: the strip overlaps itself for ~3 widths and is closed with two rivets
        a0 = a_lap + radians(r_h.jitter(0, 9))
        L = (0.085 + r_h.u(0, 0.03)) * sc
        da = L / renv(z)
        ro = lambda q: renv(q) + t_h - 0.0004
        za, zb_ = z0 + 0.0012, z1 - 0.0012
        cols = []
        nq = lod_pick(4, 2)
        for q in range(nq + 1):
            a = a0 + da * q / nq
            cols.append([laps.vert((r_ * cos(a), r_ * sin(a), z_), (a * Rb, z_)) for r_, z_ in
                         ((ro(za), za), (ro(za) + t_h + 0.001, za), (ro(zb_) + t_h + 0.001, zb_), (ro(zb_), zb_))])
        for q in range(nq):
            A, B = cols[q], cols[q + 1]
            for j in range(3):
                laps.face((A[j], B[j], B[j + 1], A[j + 1]))
        laps.face(cols[0])
        laps.face(list(reversed(cols[-1])))
        for tq in (0.24, 0.72):
            a = a0 + da * tq
            zr = z + r_h.jitter(0, w * 0.07)
            rr = ro(zr) + t_h + 0.0007
            riv_m.append(trs((rr * cos(a), rr * sin(a), zr), (0, 90, degrees(a)), sc))
    lap_o = laps.obj("hoop_laps", [iron], 35.0)
    rivets = instances_to_mesh("rivets", rivet, riv_m, keep_base=False)
    barrel += [lap_o, rivets]
    hoop_objs = barrel[1:]
    assign(hoop_objs, iron)
    if rims:
        assign(rims, steel)
    barrel += rims

    # ------------------------------------------------------------------ heads (boards let into the croze)
    r_d = rng.fork("heads")
    Rhead = rad(H - cd) - th + 0.004
    nbh = max(4, int(round(2 * Rhead / (0.105 * sc))))
    head_rot = radians(r_d.jitter(0, 14))
    ch_, sh_ = cos(-head_rot), sin(-head_rot)

    def to_head(x, y):                           # cask xy -> the head's own board frame
        return x * ch_ - y * sh_, x * sh_ + y * ch_

    heads = []
    hd_dz = 0.0
    if top == "chime":
        stains = []
        r_b = r_d.fork("brand")
        txt = p.brand.strip()[:6]
        if txt and pose in ("stand", "side"):
            # cooper's mark burnt in with a hot iron: charred core, scorched halo, uneven pressure, weak over the joints
            by = Rhead * (0.3 if (has_tap and pose == "side") else 0.0)
            cover, half = _text_mask(txt, "sans", 0.6 * 2 * Rhead, (0.23 if by else 0.3) * 2 * Rhead)
            th_b = radians(r_b.jitter(0, 5)) - head_rot
            cxl, cyl = to_head(0.0, by)
            cb_, sb_ = cos(th_b), sin(th_b)
            ph, sd = r_b.u(0, 6.28), r_b.pick([-1, 1])

            def brand_fn(X, Y):
                dx, dy = X - cxl, Y - cyl
                qx, qy = dx * cb_ + dy * sb_, -dx * sb_ + dy * cb_
                Q = np.stack([qx, qy], 1)
                core = np.clip(cover(Q, 0.0045 * sc) * 2.0 - 0.2, 0.0, 1.0)       # the iron is ~2 cm wide and chars past its edge
                m = np.maximum(core, 0.55 * cover(Q, 0.013 * sc))
                blot = 0.5 + 0.27 * np.sin(qx * 41 + ph) + 0.23 * np.sin(qy * 67 + qx * 13 + 2 * ph)
                press = 1.0 - 0.3 * np.clip(sd * qx / half[0], 0, 1)
                return m * (0.8 + 0.2 * blot) * press

            hx = abs(half[0] * cb_) + abs(half[1] * sb_) + 0.02 * sc
            hy = abs(half[0] * sb_) + abs(half[1] * cb_) + 0.02 * sc
            stains.append((brand_fn, (cxl - hx, cxl + hx, cyl - hy, cyl + hy), clamp(2 * half[1] / 16, 0.0055, 0.008)))
        if has_tap and pose == "side":           # wet run from the tap flange down the head
            tx, ty = to_head(0.0, -(Rhead - 0.075 * sc))
            gx, gy = to_head(0.0, -1.0)
            o2 = r_b.pick([-1, 1]) * 0.019 * sc

            def drip_fn(X, Y):
                dx, dy = X - tx, Y - ty
                down, side = dx * gx + dy * gy, -dx * gy + dy * gx
                run = np.exp(-(side / (0.011 * sc)) ** 2) * (down > -0.012 * sc) * 0.85
                run2 = np.exp(-((side - o2) / (0.006 * sc)) ** 2) * (down > 0) * np.clip(1 - down / (0.05 * sc), 0, 1) * 0.6
                halo = np.clip(1 - np.hypot(dx, dy) / (0.036 * sc), 0, 1) * 0.6
                return np.maximum(np.maximum(run, run2), halo)

            stains.append((drip_fn, (tx - 0.05 * sc, tx + 0.05 * sc, ty - 0.1 * sc, ty + 0.045 * sc), 0.005))
        if pose == "stand" and r_b.chance(0.75):  # somebody left a tankard on it
            ra = radians(r_b.u(65, 115))
            mx, my = to_head(0.62 * Rhead * cos(ra), 0.62 * Rhead * sin(ra))
            stains.append((_ring_fn(mx, my, 0.04, r_b.u(0, 6.28)), (mx - 0.052, mx + 0.052, my - 0.052, my + 0.052), 0.005))
        hd, _, hd_dz = _disc("head_top", Rhead, nbh, r_d.fork("t"), [oak_a, endg], age, nrow=9,
                             rim_g=0.6 if pose == "stand" else 0.35, stains=stains)
        _put(hd, Matrix.Translation((0, 0, H - cd)) @ Matrix.Rotation(head_rot, 4, 'Z'))
        heads.append(hd)
    if bot == "chime":
        hd, _, _ = _disc("head_bottom", Rhead, nbh, r_d.fork("b"), [oak_a, endg], age, rim_g=0.35)
        _put(hd, Matrix.Translation((0, 0, cd)) @ Matrix.Rotation(radians(180), 4, 'X') @ Matrix.Rotation(head_rot + 0.5, 4, 'Z'))
        heads.append(hd)
    barrel += heads

    # spile (vent peg) knocked into the head of the standing cask
    spile = None
    if pose == "stand":
        sa = radians(r_d.u(215, 250))
        sr = Rhead * 0.74
        spile = cylinder("spile", 0.0075 * sc, 0.034 * sc, loc=(sr * cos(sa), sr * sin(sa), H - cd + 0.011 * sc),
                         rot=(r_d.jitter(0, 9), r_d.jitter(0, 9), 0), verts=7, radius2=0.0055 * sc, bevel=0.001)
        assign(spile, endg)
        barrel.append(spile)

    # ------------------------------------------------------------------ bung in the bilge of the widest stave
    bung = None
    if has_bung:
        kb = sc
        bung = lathe("bung", [(0.026 * kb, -0.008 * kb), (0.028 * kb, 0.0), (0.0305 * kb, 0.015 * kb), (0.0292 * kb, 0.0185 * kb),
                              (0.027 * kb, 0.02 * kb), (0.0, 0.0206 * kb)], segments=16)
        assign(bung, endg)
        uv_box(bung)
        rbg = rad(H / 2) + st[0]["dr"]
        _put(bung, _zframe((rbg * cos(a_bung), rbg * sin(a_bung), H / 2), (cos(a_bung), sin(a_bung), 0), 0.4))
        barrel.append(bung)

    # ------------------------------------------------------------------ tap
    tap = None
    tap_l = None                                  # (origin, out, up) in the barrel frame
    if has_tap:
        tap = _tap(p.tap, kt, brass, endg)
        if pose == "side":
            org = Vector((0.0, -(Rhead - 0.075 * sc), H - cd + hd_dz))
            tap_l = (org, Vector((0, 0, 1)), Vector((0, 1, 0)))
        else:                                     # low in a front stave, between the chime hoop and the next one
            at = mids[tap_i]
            rt = rad(zt_l) + st[tap_i]["dr"] - 0.003
            tap_l = (Vector((rt * cos(at), rt * sin(at), zt_l)), Vector((cos(at), sin(at), 0)), Vector((0, 0, 1)))
        _put(tap, _frame(*tap_l))
        barrel.append(tap)

    # ------------------------------------------------------------------ pose everything that belongs to the cask
    for o in barrel:
        _put(o, M)

    # ------------------------------------------------------------------ weathering of iron, heads, fittings
    lo_r = 0.8 - 0.62 * rust
    n1 = noise_mask(0.085 * sc, seed=p.seed + 1, lo=lo_r, hi=lo_r + 0.13)
    n2 = noise_mask(0.028 * sc, seed=p.seed + 2, lo=0.25, hi=0.75)
    n3 = noise_mask(0.05 * sc, seed=p.seed + 5, lo=0.3, hi=0.7)

    def hoop_patch(co, n):                        # where the scale has broken out
        low = np.clip(1.0 - co[:, 2] / (0.22 * sc), 0.0, 1.0)              # splash zone rusts first
        return np.clip(n1(co, n) * (0.55 + 0.45 * n2(co, n)) * 1.35 + (0.15 + 0.5 * rust) * low * n3(co, n) + rust ** 3 * 0.5, 0.0, 1.0)

    # iron_rusty's own albedo is the bright mottled scale: leave it bare where the rust has broken out, and lay
    # grime + dark-metal wear over the rest (sound, oiled iron). A little `special` only varies the rust tone.
    def hoop_g(co, n):
        pt = hoop_patch(co, n)
        return np.clip((1.0 - pt) * (0.52 + 0.14 * n3(co, n)) + pt * 0.3 * (1.0 - n3(co, n)) * (1.0 - n2(co, n)), 0.0, 1.0)

    paint(hoop_objs, special=lambda co, n: hoop_patch(co, n) * 0.3 * n2(co, n), grime=hoop_g,
          wear=lambda co, n: (1.0 - hoop_patch(co, n)) * 0.3)
    lod_drop([lap_o, rivets], above=1)            # far off: bands only
    if not rims:
        pass
    elif rust > 0.6:
        paint(rims, special=lambda co, n: hoop_patch(co, n) * 0.35 * n3(co, n),
              grime=lambda co, n: np.clip(hoop_g(co, n) + 0.12, 0.0, 1.0), wear=lambda co, n: (1.0 - hoop_patch(co, n)) * 0.45)
    else:
        paint(rims, grime=lambda co, n: np.clip(hoop_patch(co, n) * 0.85 + (1.0 - hoop_patch(co, n)) * (0.1 + (1.0 - n3(co, n)) * 0.32), 0, 1),
              wear=lambda co, n: (1.0 - hoop_patch(co, n)) * (0.1 + 0.4 * n3(co, n)))

    for hd in heads:
        paint(hd, grime=mask_mul(noise_mask(0.1, seed=p.seed + 3, lo=0.4, hi=0.8), 0.08 + 0.14 * age))
    if pose == "stand" and age > 0.55:            # rain stands on the head of an old cask: green in the croze
        zh_w = lift + H - cd
        paint(heads[0], special=lambda co, n: np.clip((np.hypot(co[:, 0], co[:, 1]) / Rhead - 0.8) / 0.18, 0, 1)
              * (age - 0.5) * 1.6 * (np.abs(co[:, 2] - zh_w) < 0.02))
    if spile is not None:
        paint(spile, grime=0.3, wear=0.4)
        lod_drop(spile, above=1)
    if bung is not None:
        paint(bung, grime=lambda co, n: 0.15 + 0.5 * age * (np.abs(n @ Mr[:, 2]) > 0.5), wear=0.3)
    if tap is not None:
        Tw = np.array(M @ _frame(*tap_l))
        Ti = np.linalg.inv(Tw)

        def tap_l_co(co):
            return co @ Ti[:3, :3].T + Ti[:3, 3]

        def tap_wear(co, n):                      # hands polish the key, the jug rim polishes the spout mouth
            c = tap_l_co(co) / kt
            return np.clip(np.maximum.reduce([0.95 * (c[:, 2] > 0.027), 0.85 * (c[:, 0] > 0.088), 0.35 + 0.0 * c[:, 0]]), 0, 1)

        paint(tap, wear=tap_wear,
              special=lambda co, n: np.clip(1.0 - np.linalg.norm(tap_l_co(co), axis=1) / (0.026 * kt), 0, 1) * 0.7)

    # ------------------------------------------------------------------ pose-specific furniture
    if lift > 0:
        _bearers(p, rng.fork("bearers"), sc, lift, Rh, tone, endg)
    if pose == "side":
        _cradle(p, rng.fork("cradle"), H, sc, Rb, zc, renv, fr, tone, endg)
    elif pose == "half":
        _planting(p, rng.fork("plant"), sc, Ztop, soil_d, rad(Ztop - soil_d) - th + 0.003, rad(Ztop) + drm)
        socket("plant", "attach", (0, 0, Ztop - soil_d), role="planter")
    elif pose == "table":
        _table(p, rng.fork("table"), H, lift, sc, Rh, tone, endg, iron, brass)
    else:
        socket("top", "attach", (0, 0, lift + H - cd + 0.002), role="prop_top")


# ---------------------------------------------------------------------------------------------
# pose furniture
# ---------------------------------------------------------------------------------------------
def _bearers(p, r, sc, lift, Rh, tone, endg):
    """Two squared oak bearers (a low stillage) under a tapped standing cask: the spout clears a jug."""
    wood = mat("wood_oak", tone(0.84, 0.25), rough=1.0)
    w = 0.088 * sc
    L = 2 * Rh + 0.2 * sc
    objs = []
    for sx in (-1, 1):
        Mb = Matrix.Translation((sx * 0.56 * Rh + r.jitter(0, 0.006), r.jitter(0, 0.012), 0)) @ \
            Matrix.Rotation(radians(90 + r.jitter(0, 1.5)), 4, 'Z')
        o = _timber(f"bearer{sx}", L * r.vary(1.0, 0.03), w, lift, 0.036 * sc, Mb, wood, endg, bevel=0.0035)
        objs.append(o)
    paint(objs, grime=mask_mul(height_ramp(0.0, 0.09), 0.85),
          special=mask_mul(height_ramp(0.0, 0.05), noise_mask(0.07, seed=p.seed + 9, lo=0.3, hi=0.7), 0.2 + 0.6 * p.age),
          wear=mask_mul(facing((0, 0, 1)), noise_mask(0.06, seed=p.seed + 8, lo=0.3, hi=0.8), 0.6))


def _cradle(p, r, H, sc, Rb, zc, renv, fr, tone, endg):
    """Stillage: two oak saddles cut to the belly with chamfered shoulders, a pair of wedge chocks on each
    against the bilge, tied by two through-tenoned rails locked with pegs."""
    wood = mat("wood_oak", tone(0.84, 0.25), rough=1.0)
    cand = [0.5 * (a + b) for a, b in zip(fr, fr[1:])] + [fr[-1] + 0.075]
    fs = min(cand, key=lambda f: abs(f - 0.24))
    j = max(i for i, f in enumerate(fr) if f < fs)
    gap = (fr[j + 1] - fr[j]) * H - 0.04 * sc if j + 1 < len(fr) else 0.1
    sth = clamp(gap - 0.014, 0.034, 0.062 * sc)
    hw = Rb * 1.0
    fw, ah = 0.11 * sc, 0.034 * sc
    ch = 0.03 * sc
    spl = 0.016 * sc                                       # the cheeks lean in a little
    te = radians(41)
    objs, chocks = [], []
    for sgn in (-1, 1):
        zl = H / 2 + sgn * (0.5 - fs) * H                  # station in the cask frame
        zl_in = zl - sgn * sth * 0.2                       # fit the cut to the fatter (bilge side) end of the saddle
        rs = renv(zl_in) + 0.0005
        xe = rs * sin(te)
        zh = zc - rs * cos(te)                             # flat ledges either side of the cut
        ht = hw - spl
        pts = [(-hw, 0.0), (-hw + fw, 0.0), (-hw + fw + 0.035 * sc, ah), (hw - fw - 0.035 * sc, ah), (hw - fw, 0.0), (hw, 0.0),
               (ht, zh - ch), (ht - ch, zh)]
        nseg = 12
        pts += [(rs * sin(te - 2 * te * q / nseg), zc - rs * cos(te - 2 * te * q / nseg)) for q in range(nseg + 1)]
        pts += [(-ht + ch, zh), (-ht, zh - ch)]
        yw = H / 2 - zl + r.jitter(0, 0.004)
        sd = extrude_profile(f"saddle{sgn}", pts, sth, loc=(0, yw + sth / 2, 0), rot=(90, 0, 0), bevel=lod_pick(0.0065 * sc, 0.0))
        uv_grain(sd, 'X')
        smooth(sd, 35)
        objs.append(sd)
        # wedge chocks knocked in between ledge and bilge
        cth = sth * 0.62
        for sx in (-1, 1):
            dq = radians(11 + r.jitter(0, 1.5))
            rc = rs + 0.0012
            arc = [(rc * sin(te + dq * q / 4), zc - rc * cos(te + dq * q / 4)) for q in range(5)]
            xo = max(ht - ch - 0.004, arc[-1][0] + 0.014 * sc)
            wp = [(arc[0][0] + 0.001, zh), (xo, zh), (xo, zh + 0.014 * sc), (arc[-1][0] + 0.011 * sc, arc[-1][1])] + arc[:0:-1]
            if sx < 0:
                wp = [(-x, z) for x, z in reversed(wp)]
            ck = extrude_profile(f"chock{sgn}{sx}", wp, cth, loc=(0, yw + cth / 2 + r.jitter(0, 0.004), 0.0), rot=(90, 0, 0),
                                 bevel=0.003 * sc)
            uv_grain(ck, 'X')
            smooth(ck, 35)
            chocks.append(ck)
    ys = (0.5 - fs) * H
    rail_len = 2 * ys + sth + 0.17 * sc
    xr = hw * 0.55
    rh_, rw_ = 0.06 * sc, 0.045 * sc
    zr = ah + 0.014 * sc                                    # underside of the rails
    pegs = []
    for sx in (-1, 1):
        Mr_ = Matrix.Translation((sx * xr, r.jitter(0, 0.006), zr)) @ Matrix.Rotation(radians(90), 4, 'Z')
        objs.append(_timber(f"rail{sx}", rail_len, rw_, rh_, 0.012 * sc, Mr_, wood, endg, bevel=0.004, both=True))
        for sy in (-1, 1):
            pg = cylinder(f"peg{sx}{sy}", 0.0085 * sc, 0.108 * sc, loc=(sx * xr, sy * (ys + sth / 2 + 0.028 * sc), zr + rh_ / 2 + 0.008),
                          rot=(r.jitter(0, 5), r.jitter(0, 5), 0), verts=6, radius2=0.0062 * sc, bevel=0.0015)
            pegs.append(pg)
    assign(objs[:2] + chocks, wood)
    assign(pegs, endg)
    allo = objs + pegs + chocks
    paint(allo, grime=mask_mul(height_ramp(0.0, 0.14), 0.85),
          special=mask_mul(height_ramp(0.0, 0.05), noise_mask(0.07, seed=p.seed + 9, lo=0.3, hi=0.7), 0.15 + 0.6 * p.age),
          wear=mask_mul(facing((0, 0, 1)), noise_mask(0.06, seed=p.seed + 8, lo=0.25, hi=0.75), 0.7))
    paint(chocks, wear=0.35)


def _leaf_base(kind, m, T=0.25):
    """Unit leaf lying in XY, petiole at the origin, tip towards +X; m_grime carries the dark zone.
    kind 0 / 1: zonal pelargonium (round, scalloped, cupped, dark horseshoe band); 2: ivy-leaved (five lobes)."""
    lb = _MB()
    Lv = lod()

    def V(x, y, z, g=0.0):
        return lb.vert((x, y, z), (x / 2 * T, (y + 1) / 2 * T), g)

    if Lv == 2:                                   # far off: one cupped quad (round leaf) / one diamond (ivy)
        lb.face([V(0.25, 0.0, 0.02, 0.3), V(0.85, -0.8, -0.15, 0.5), V(1.9, 0.0, 0.1, 0.5), V(0.85, 0.8, -0.15, 0.5)] if kind == 2 else
                [V(1.0 + 0.95 * cos(a), 0.95 * sin(a), 0.3, 0.3) for a in (radians(45), radians(135), radians(225), radians(315))])
        return lb.obj(f"leaf_base{kind}", [m], 180.0)
    if Lv == 1 and kind != 2:                     # lod 1: a cupped, rippled fan of seven - the scallops are gone
        c0 = V(1.0, 0.0, -0.04, 0.12)
        cup, rip = (0.3, 0.08) if kind == 0 else (0.46, 0.13)
        ring = [V(1.0 + 0.97 * cos(a), 0.97 * sin(a), cup * 0.94 + rip * sin(3.0 * a + kind), 0.4)
                for a in [pi + (q + 0.5) / 7 * 2 * pi for q in range(7)]]
        for q in range(7):
            lb.face((c0, ring[q], ring[(q + 1) % 7]))
        return lb.obj(f"leaf_base{kind}", [m], 180.0)
    if kind == 2:
        c0 = V(0.85, 0.0, 0.1, 0.62)
        ring = []
        for q in range(10):
            a = pi + (q + 0.5) / 10 * 2 * pi
            rr = (1.0 if q % 2 else 0.6) * (1.12 if q in (4, 5) else 1.0)
            zk = 0.1 * rr * rr - (0.16 if q % 2 else 0.0) - 0.34 * rr * abs(sin(a)) + 0.12 * rr * cos(a)   # keel + curled lobes
            ring.append(V(0.85 + rr * cos(a) * 0.9, rr * sin(a), zk, 0.5 if q % 2 else 0.68))
        for q in range(9):
            lb.face((c0, ring[q], ring[q + 1]))
        o0 = V(0.25, 0.0, 0.02, 0.3)
        lb.face((c0, ring[-1], o0))
        lb.face((c0, o0, ring[0]))
        return lb.obj("leaf_base2", [m], 180.0)
    n_in = 7 if kind == 0 else 6
    cup = 0.3 if kind == 0 else 0.46
    rip = 0.08 if kind == 0 else 0.13
    c0 = V(1.0, 0.0, -0.04, 0.12)
    inner, outer = [], []
    for q in range(n_in):
        a = pi + (q + 0.5) / n_in * 2 * pi
        ri = 0.5
        inner.append(V(1.0 + ri * cos(a), ri * sin(a), cup * ri * ri, 0.9))
    for q in range(2 * n_in):
        a = pi + (q + 0.5) / (2 * n_in) * 2 * pi
        ro = (1.0 if q % 2 else 0.86) * (0.96 + 0.04 * cos(2 * a))
        outer.append(V(1.0 + ro * cos(a), ro * sin(a), cup * ro * ro + rip * sin(3.0 * a + kind) - (0.0 if q % 2 else 0.05), 0.22))
    for q in range(n_in - 1):
        lb.face((c0, inner[q], inner[q + 1]))
    for q in range(n_in):
        if q < n_in - 1:
            lb.face((inner[q], outer[2 * q], outer[2 * q + 1]))
            lb.face((inner[q], outer[2 * q + 1], inner[q + 1]))
            lb.face((inner[q + 1], outer[2 * q + 1], outer[2 * q + 2]))
        else:
            lb.face((inner[q], outer[2 * q], outer[2 * q + 1]))
    o0 = V(0.22, 0.0, 0.03, 0.3)                           # the notch where the petiole joins
    lb.face((c0, inner[-1], o0))
    lb.face((c0, o0, inner[0]))
    lb.face((inner[-1], outer[-1], o0))
    lb.face((o0, outer[0], inner[0]))
    return lb.obj(f"leaf_base{kind}", [m], 180.0)


def _floret_base(m, T=0.25):
    """Pelargonium floret: five rounded petals (two upper narrower), throat darker, slightly cupped."""
    fb = _MB()
    if lod():                                     # levels: a five-pointed cup (far off: flat)
        tips = [fb.vert(((1.0 if q in (0, 2, 3) else 0.86) * cos(q * 2 * pi / 5), (1.0 if q in (0, 2, 3) else 0.86) * sin(q * 2 * pi / 5), 0.3),
                        (0.95 * T, (0.1 + 0.2 * q) * T)) for q in range(5)]
        if lod() == 2:
            fb.face(tips)
        else:
            fc = fb.vert((0, 0, 0), (0.05 * T, 0.5 * T), 0.6)
            for q in range(5):
                fb.face((fc, tips[q], tips[(q + 1) % 5]))
        return fb.obj("floret_base", [m], 180.0)
    fc = fb.vert((0, 0, 0), (0.05 * T, 0.5 * T), 0.6)
    for q in range(5):
        a = q * 2 * pi / 5
        L_ = 1.0 if q in (0, 2, 3) else 0.86
        hwid = 0.56
        v = [fb.vert((0.66 * L_ * cos(a - hwid), 0.66 * L_ * sin(a - hwid), 0.18), (0.6 * T, 0.2 * T)),
             fb.vert((L_ * cos(a), L_ * sin(a), 0.3), (0.95 * T, 0.5 * T)),
             fb.vert((0.66 * L_ * cos(a + hwid), 0.66 * L_ * sin(a + hwid), 0.18), (0.6 * T, 0.8 * T))]
        fb.face((fc, v[0], v[1], v[2]))
    return fb.obj("floret_base", [m], 180.0)


def _umbel(r, flo_m, c, up, hr, n, size):
    """n florets packed on a spherical cap of radius hr around c (golden-angle spiral) - one flower head."""
    ref = Vector((1, 0, 0)) if abs(up.x) < 0.8 else Vector((0, 1, 0))
    e1 = up.cross(ref).normalized()
    e2 = up.cross(e1)
    ph0 = r.u(0, 6.28)
    for q in range(n):
        ct = 1.0 - 1.32 * (q + 0.5) / n                    # down to ~108 deg from the axis
        st_ = sqrt(max(0.0, 1.0 - ct * ct))
        ph = ph0 + q * 2.39996
        d = (up * ct + (e1 * cos(ph) + e2 * sin(ph)) * st_ + Vector((r.jitter(0, 0.12), r.jitter(0, 0.12), r.jitter(0, 0.12)))).normalized()
        flo_m.append(_zframe(c + d * hr * r.u(0.86, 1.06), d, r.u(0, 6.28)) @ Matrix.Scale(size * r.u(0.85, 1.15), 4))


def _planting(p, r, sc, ztop, soil_d, rs, r_out):
    zs = ztop - soil_d
    soil = lathe("soil", [(rs, zs - 0.012), (rs * 0.82, zs + 0.004), (rs * 0.5, zs + 0.014), (rs * 0.2, zs + 0.02), (0.0, zs + 0.021)],
                 segments=20, smooth_angle=70)
    displace_noise(soil, strength=0.007, scale=0.06, seed=p.seed)
    uv_box(soil)
    assign(soil, mat("soil"))
    paint(soil, special=mask_mul(noise_mask(0.15, seed=p.seed, lo=0.4, hi=0.7), 0.7))
    if p.planting == "soil":
        return
    up = Vector((0, 0, 1))
    if p.planting == "box":
        # clipped box ball on a short stem: lumpy shears-cut mass + a shell of leaf sprigs that breaks the outline
        rb = rs * 0.84
        zc = zs + rb * 0.96 + 0.055 * sc
        trunk = cylinder("trunk", 0.017 * sc, zc - zs, loc=(0.01, 0, (zc + zs) / 2), verts=7, radius2=0.012 * sc)
        assign(trunk, mat("bark"))
        ball = sphere("box_ball", rb, loc=(0, 0, zc), segments=lod_pick(30, 16, 8), rings=lod_pick(17, 9, 5), scale=(1.0, 1.0, 0.93))
        realize(ball)
        displace_noise(ball, strength=0.03, scale=rb * 0.55, seed=p.seed + 1)
        displace_noise(ball, strength=0.02, scale=rb * 0.2, seed=p.seed + 2)
        displace_noise(ball, strength=0.012, scale=rb * 0.085, seed=p.seed + 3)
        uv_box(ball)
        assign(ball, mat("hedge", "#2e4220", wear=0.25))
        paint(ball, grime=mask_mul(noise_mask(rb * 0.2, seed=p.seed + 4, lo=0.45, hi=0.75), 0.6),
              wear=mask_mul(facing((0, 0, 1)), noise_mask(rb * 0.3, seed=p.seed + 5, lo=0.3, hi=0.8), 0.35))
        boxleaf = mat("leaf", "#34491f", double=1, wear=0, rough=1.5)
        T = 0.25
        sb = _MB()
        for q in range(lod_pick(3, 3, 1)):                  # sprig: three small oval leaves on a twig end (far off: one)
            a = q * 2.0944 + 0.3
            tl = 0.95
            d = Vector((cos(a) * sin(tl), sin(a) * sin(tl), cos(tl)))
            sdv = Vector((-sin(a), cos(a), 0))
            nrm = d.cross(sdv)
            Lf = 1.0 - 0.12 * q
            v = [sb.vert(d * 0.06, (0.04 * T, 0.5 * T), 0.5),
                 sb.vert(d * 0.55 * Lf + sdv * 0.27 + nrm * 0.06, (0.5 * T, 0.2 * T)),
                 sb.vert(d * Lf - nrm * 0.1, (0.96 * T, 0.5 * T)),
                 sb.vert(d * 0.55 * Lf - sdv * 0.27 + nrm * 0.06, (0.5 * T, 0.8 * T))]
            sb.face(v)
        sprig = sb.obj("sprig_base", [boxleaf], 180.0)
        Mb = ball.matrix_basis.copy()
        # (the sprigs stand on the vertices of the full ball at every level: the same draws, the same places)
        vs = lod_memo("box_ball_verts", lambda: [(Mb @ v.co, (Mb.to_3x3() @ v.normal).normalized()) for v in ball.data.vertices])
        sm = []
        for q in range(250):
            co, n = vs[int(r.u(0, len(vs) - 0.001))]
            if n.z < -0.55:
                continue
            tw = Vector((r.jitter(0, 0.4), r.jitter(0, 0.4), r.jitter(0, 0.4)))
            off = Vector((r.jitter(0, 0.012), r.jitter(0, 0.012), r.jitter(0, 0.012)))
            sm.append(_zframe(co + off - n * 0.004, (n + tw).normalized(), r.u(0, 6.28)) @ Matrix.Scale(r.u(0.02, 0.031) * sc, 4))
        instances_to_mesh("box_sprigs", sprig, _fewer(sm, (2, 1.4), (4, 2.4)), uv_jitter=False, keep_base=False)
        return
    green = mat("leaf", "#42602b", double=1, wear=0, rough=1.8)
    green2 = mat("leaf", "#364f2c", double=1, wear=0, rough=2.3)
    petal = mat("leaf", p.bloom, double=1, wear=0, rough=1.5)
    leaves = [_leaf_base(0, green), _leaf_base(1, green2), _leaf_base(2, green2)]
    floret = _floret_base(petal)

    n_stem = max(9, int(round(17 * (rs / 0.3))))
    stems, flo_m = [], []
    leaf_m = [[], [], []]
    for i in range(n_stem):
        a = i * 2.39996 + r.jitter(0, 0.4)
        rr = rs * (0.12 + 0.66 * sqrt((i + 0.5) / n_stem))
        base = Vector((rr * cos(a), rr * sin(a), zs + 0.004))
        out = Vector((cos(a + r.jitter(0, 0.5)), sin(a + r.jitter(0, 0.5)), 0))
        hgt = (0.15 + 0.17 * r.u() * (1.0 - 0.45 * rr / rs)) * sc
        lean = (0.03 + 0.2 * (rr / rs) ** 1.5 * r.u(0.5, 1.0)) * sc
        tip = base + out * lean + up * hgt
        pts = bezier_path(base, base + up * hgt * 0.45, tip - out * lean * 0.45 - up * hgt * 0.1, tip, n=5)
        stems.append(sweep(f"stem{i}", pts, 0.0042 * sc, sides=lod_pick(4, 3), smooth_path=False, scale_fn=lambda t: 1.0 - 0.4 * t))
        nl = 8 + int(r.u(0, 3))
        for q in range(nl):
            t = 0.2 + 0.78 * (q + r.u(0, 0.8)) / nl
            pt = Vector(pts[min(len(pts) - 1, int(t * (len(pts) - 1)))])
            ya = a + q * 2.4 + r.jitter(0, 0.6)
            d = Vector((cos(ya), sin(ya), 0))
            droop = r.u(-0.3, 0.5) + 0.45 * (1 - t)
            size = (0.031 + 0.021 * r.u()) * sc * (0.8 + 0.45 * (1 - t))
            xdir = d * cos(droop) - up * sin(droop)
            zdir = up * cos(droop) + d * sin(droop)
            kind = 0 if r.chance(0.6) else 1
            leaf_m[kind].append(_frame(pt + d * 0.014 * sc, xdir, zdir) @ Matrix.Rotation(r.jitter(0, 0.4), 4, 'X') @ Matrix.Scale(size, 4))
        if r.chance(0.7):                                   # flower head on a bare stalk held above the leaves
            axis = (up + out * r.u(0.15, 0.6) + Vector((r.jitter(0, 0.15), r.jitter(0, 0.15), 0))).normalized()
            Ls = (0.045 + 0.05 * r.u()) * sc
            hr = (0.02 + 0.008 * r.u()) * sc
            c = tip + axis * Ls
            stems.append(sweep(f"stalk{i}", [tip - axis * 0.01, tip + axis * Ls * 0.5 + out * 0.004, c], 0.0024 * sc, sides=lod_pick(4, 3), smooth_path=False))
            _umbel(r, flo_m, c, axis, hr, 20 + int(r.u(0, 5)), 0.0125 * sc)
    for i in range(max(12, int(round(27 * rs / 0.3)))):     # skirt of big basal leaves that hides the soil and laps the rim
        a = i * 2.39996 + r.jitter(0, 0.5)
        rr = rs * r.u(0.3, 0.97)
        d = Vector((cos(a), sin(a), 0))
        droop = r.u(0.05, 0.45) + 0.5 * (rr / rs) ** 3
        size = (0.036 + 0.02 * r.u()) * sc
        pt = d * (rr - size) + up * (zs + (0.03 + 0.06 * r.u()) * sc)
        leaf_m[0 if r.chance(0.6) else 1].append(_frame(pt, d * cos(droop) - up * sin(droop), up * cos(droop) + d * sin(droop))
                                                 @ Matrix.Rotation(r.jitter(0, 0.3), 4, 'X') @ Matrix.Scale(size, 4))
    # ivy-leaved trailers spill over the rim and hang down the staves
    n_tr = 5 + int(r.u(0, 3))
    a0 = r.u(0, 6.28)
    for i in range(n_tr):
        a = a0 + (i + r.jitter(0, 0.3)) * 2 * pi / n_tr
        out = Vector((cos(a), sin(a), 0))
        tan = Vector((-sin(a), cos(a), 0))
        L = (0.1 + 0.2 * r.u()) * sc
        q0 = out * rs * 0.72 + up * (zs + 0.004)
        q1 = out * (r_out + 0.004) + up * (ztop + 0.011)
        q3 = out * (r_out + 0.024) + tan * r.jitter(0, 0.06) + up * (ztop - L)
        q2 = out * (r_out + 0.03) + tan * r.jitter(0, 0.035) + up * (ztop - L * 0.45)
        pts = bezier_path(q0, q0 + up * 0.05 * sc, q1 - out * 0.05 * sc + up * 0.012, q1, n=4)
        pts += bezier_path(q1, q1 + out * 0.03 * sc - up * 0.004, q2, q3, n=5)[1:]
        stems.append(sweep(f"trail{i}", pts, 0.003 * sc, sides=lod_pick(4, 3), smooth_path=False, scale_fn=lambda t: 1.0 - 0.5 * t))
        nl = int(L / (0.03 * sc)) + 3
        for q in range(nl):
            t = 0.3 + 0.7 * (q + r.u(0, 0.6)) / nl
            pt = Vector(pts[min(len(pts) - 1, int(round(t * (len(pts) - 1))))])
            sd = 1 if q % 2 else -1
            rad_ = Vector((pt.x, pt.y, 0)).normalized()
            xdir = tan * sd * r.u(0.2, 1.0) - up * r.u(0.2, 1.0) + rad_ * r.u(0.0, 0.5)
            size = (0.015 + 0.017 * r.u()) * sc
            nrm = rad_ + up * r.u(0.1, 0.9) + tan * r.jitter(0, 0.4)
            leaf_m[2 if r.chance(0.75) else 1].append(_frame(pt + rad_ * 0.005 + tan * sd * 0.006, xdir, nrm) @ Matrix.Scale(size, 4))
        if r.chance(0.6):
            tipv = Vector(pts[-1])
            rad_ = Vector((tipv.x, tipv.y, 0)).normalized()
            _umbel(r, flo_m, tipv + rad_ * 0.016 * sc, (rad_ + up * 0.3).normalized(), 0.015 * sc, 10, 0.0105 * sc)
    assign(stems, green2)
    lod_drop(join(stems, "stems"), above=1)       # far off the stems are lost in the leaves
    for k in range(3):
        if leaf_m[k]:
            instances_to_mesh(f"leaves{k}", leaves[k], _fewer(leaf_m[k]), uv_jitter=False, keep_base=False)
        else:
            delete(leaves[k])
    instances_to_mesh("florets", floret, _fewer(flo_m, (2, 1.4), (5, 2.2)), uv_jitter=False, keep_base=False)


def _table(p, r, H, lift, sc, Rh, tone, endg, iron, brass):
    """Round plank top nailed to two 60 x 30 mm battens that lie across the chime; candle and tankard on it."""
    Rt = p.top_width / 2
    tk_ = 0.034 * sc
    bt, bw = 0.03, 0.06
    pale = mat("wood_oak", tone(1.1, 0.15), rough=0.9)
    nb = max(4, int(round(2 * Rt / 0.135)))
    rot = radians(r.jitter(0, 20))
    zu = lift + H + bt                                     # underside of the top = upper face of the battens
    zt = zu + tk_
    r_t = r.fork("dress")
    # where things stand (table frame, before the top is turned by `rot`)
    ca, cr = r_t.u(0, 6.28), Rt * r_t.u(0.08, 0.3)
    ta = ca + radians(r_t.u(115, 245))
    tr = Rt * r_t.u(0.5, 0.66)
    rings = []
    for q in range(2):                                     # rings left by earlier tankards
        qa = ta + radians(r_t.u(50, 130)) * (1 if q else -1)
        qr = Rt * r_t.u(0.35, 0.68)
        mx, my = qr * cos(qa), qr * sin(qa)
        rings.append((_ring_fn(mx, my, 0.041, r_t.u(0, 6.28), 0.5 + 0.25 * q), (mx - 0.053, mx + 0.053, my - 0.053, my + 0.053), 0.0052))
    top, xe, top_dz = _disc("table_top", Rt, nb, r.fork("top"), [pale, endg], p.age, thick=tk_, ease=0.003, dphi=9.0,
                            phi_min=1.5, nrow=8, tone=0.14, stains=rings)
    Mtop = Matrix.Translation((0, 0, zt)) @ Matrix.Rotation(rot, 4, 'Z')
    _put(top, Mtop)
    paint(top, wear=mask_mul(lambda co, n: np.clip(1.0 - np.hypot(co[:, 0], co[:, 1]) / (Rt * 0.9), 0, 1) ** 0.6,
                             noise_mask(0.2, seed=p.seed + 4, lo=0.3, hi=0.8), 0.35),
          grime=mask_mul(lambda co, n: np.clip((np.hypot(co[:, 0], co[:, 1]) - Rt * 0.6) / (Rt * 0.4), 0, 1),
                         noise_mask(0.13, seed=p.seed + 5, lo=0.35, hi=0.7), 0.2 + 0.4 * p.age))
    # the two battens the boards are nailed to; they rest on the chime
    yb = 0.5 * Rh
    Lb = 2 * sqrt(max((Rt - 0.045) ** 2 - yb * yb, 0.01))
    bat = []
    for sy in (-1, 1):
        Mb = Matrix.Translation((0, 0, zu)) @ Matrix.Rotation(rot, 4, 'Z') @ Matrix.Translation((r.jitter(0, 0.008), sy * yb, 0)) @ \
            Matrix.Rotation(radians(180), 4, 'X')
        bat.append(_timber(f"batten{sy}", Lb, bw, bt, 0.014, Mb, pale, endg, bevel=0.003))
    paint(bat, grime=0.25 + 0.2 * p.age)
    # clout nails through every board into the battens, two per crossing
    nail = lathe("nail", [(0.0052, 0.0), (0.0047, 0.0012), (0.0028, 0.0021), (0.0, 0.0024)], segments=6)
    nm = []
    for sy in (-1, 1):
        for b in range(nb):
            for f in (0.27, 0.73):
                x = xe[b] + (xe[b + 1] - xe[b]) * f + r.jitter(0, 0.004)
                y = sy * yb + r.jitter(0, 0.006)
                if abs(x) < Lb / 2 - 0.02:
                    nm.append(Mtop @ trs((x, y, top_dz * 0.3 - 0.0004), (r.jitter(0, 6), r.jitter(0, 6), r.u(0, 360)), sc))
    nails = instances_to_mesh("nails", nail, nm, keep_base=False)
    assign(nails, iron)
    paint(nails, special=0.3 + 0.5 * p.rust, wear=0.5)
    lod_drop(nails)
    socket("top", "attach", (0, 0, zt + 0.002), role="prop_top")

    if p.candle:
        cx, cy = cr * cos(ca), cr * sin(ca)
        dish = lathe("chamberstick", [(0.0, 0.0), (0.04, 0.0), (0.05, 0.005), (0.053, 0.012), (0.0505, 0.0125), (0.041, 0.0045),
                                      (0.017, 0.004), (0.0165, 0.02), (0.019, 0.022), (0.019, 0.0245), (0.0125, 0.025), (0.0122, 0.012),
                                      (0.0, 0.012)], segments=18, loc=(cx, cy, zt))
        da = r_t.u(0, 6.28)
        ring = torus("chamber_ring", 0.012, 0.0026, loc=(cx + 0.058 * cos(da), cy + 0.058 * sin(da), zt + 0.011),
                     rot=(90, 0, degrees(da)), major_seg=12, minor_seg=5)
        assign([dish, ring], brass)
        paint([dish, ring], grime=0.2, wear=mask_mul(noise_mask(0.02, seed=p.seed + 6, lo=0.2, hi=0.7), 0.8))
        ch = 0.058 + 0.035 * r_t.u()
        z0 = zt + 0.012
        cprof = [(0.0108, z0), (0.011, z0 + ch * 0.82), (0.0118, z0 + ch * 0.9), (0.0112, z0 + ch), (0.0085, z0 + ch + 0.001),
                 (0.004, z0 + ch - 0.004), (0.0, z0 + ch - 0.005)]
        cand = lathe("candle", cprof, segments=10, loc=(cx, cy, 0))
        displace_noise(cand, strength=0.0006, scale=0.012, seed=p.seed)
        assign(cand, mat("wax", "#e9dcc0"))
        zf = z0 + ch - 0.003
        wick = cylinder("wick", 0.0009, 0.009, loc=(cx, cy, zf + 0.0035), verts=4)
        assign(wick, iron)
        lod_drop(wick)
        flame = lathe("flame", [(0.0, zf + 0.004), (0.0034, zf + 0.009), (0.0046, zf + 0.015), (0.003, zf + 0.025), (0.0, zf + 0.036)],
                      segments=lod_pick(8, 5, 4), loc=(cx, cy, 0))
        assign(flame, mat("emissive", "#ffb45e", glow=7))
        light("candle", (cx, cy, zf + 0.05), intensity=0.75, range=4.0, flicker=0.5, shadow_near=0.08)
        fx("flame", "flame", (cx, cy, zf + 0.006), size=0.03)
    if p.tankard:
        # salt-glazed stoneware tankard: turned foot, slightly bellied body, two cordons, rolled rim
        mx, my = tr * cos(ta), tr * sin(ta)
        stone = mat("ceramic_glazed", "#6f6858", rough=2.4)
        hm = 0.128
        mug = lathe("tankard", [(0.0, 0.0), (0.041, 0.0), (0.0435, 0.003), (0.0435, 0.007), (0.041, 0.0105), (0.0395, 0.0125),
                                (0.0415, 0.03), (0.0423, 0.055), (0.0405, 0.085), (0.0395, 0.092), (0.0408, 0.0945), (0.0408, 0.0975),
                                (0.0392, 0.1), (0.0378, hm - 0.012), (0.0392, hm - 0.008), (0.0396, hm - 0.003), (0.038, hm),
                                (0.0352, hm - 0.001), (0.0345, hm - 0.02), (0.0, hm - 0.022)], segments=20, loc=(mx, my, zt))
        ha = ta + radians(r_t.u(-70, 70))
        hp = [(0.039, 0, hm * 0.8), (0.058, 0, hm * 0.84), (0.071, 0, hm * 0.62), (0.062, 0, hm * 0.3), (0.04, 0, hm * 0.2)]
        handle = sweep("tankard_handle", hp, rounded_rect(0.014, 0.007, 0.003, 2), up=(0, 1, 0), step=0.012)
        _put(handle, Matrix.Translation((mx, my, zt)) @ Matrix.Rotation(ha, 4, 'Z'))
        assign([mug, handle], stone)
        paint([mug, handle], grime=lambda co, n: np.clip(np.maximum(1.0 - (co[:, 2] - zt) / 0.03, (co[:, 2] - zt - hm + 0.035) / 0.035 * 0.75), 0, 1),
              wear=mask_mul(noise_mask(0.025, seed=p.seed + 7, lo=0.45, hi=0.8), 0.5))
