"""Columnar Italian cypress (Cupressus sempervirens 'Stricta') for the square. A fluted, buttressed bark
trunk with surface roots and snapped twig stubs stands in a soil tree pit (optionally ringed by rough stone
setts; with pit "none" the stem is cut clean at z = 0 to drop into a planter); a few bare limbs fork off the
stem and rise steeply into the ragged, tapering skirt of a dense flame-shaped crown. The crown is a noisy,
leaning envelope (plus secondary leaders and splayed side branches) broken into upright staggered plumes.
Three layers build it, each inside the next: a core that only blocks the light, a closed thatch of long
pointed slivers (a jittered brick lattice, roots deep, tips lifted so each row covers the roots of the next
and no smooth skin is ever seen) and several thousand sprays on a second lattice rooted in that thatch -
narrow brushes of 5-7 thin slivers at a 13-21 degree splay (~15 x 4 cm), swept up within ~25 degrees of
vertical and overlapping the row above - which give the surface its fine upright grain. Only the crown
surface that can be seen is built, and its element size comes from a triangle budget. Strays are short bark
twigs carrying a few sprays; every leader ends in a tapering brush of sprays.
All live foliage is ONE hedge material: hue from the tint; the depth of the layering (baked AO) carries the
value, the grime vertex mask only adds broad, faint patches. Die-back is a few low hollows: dead grey-brown
slivers deep inside, bare grey twigs in front, a little dead spray on the rim; cones sit on the sprays.
Irregularity adds lobes, lean, strays, thin dents, side branches and secondary leaders.
"""
from tk import *
import numpy as np
from math import sin, cos, pi, atan2, hypot, sqrt, radians, ceil

# Grain of the foliage: the hedge tile (2 cm leaves with dark gaps) is mapped much finer across a blade than
# along it, so its leaves become thin upright cords of scale foliage and the tile averages out at a distance.
UV_ACROSS = 5.0
UV_ALONG = 1.6
# The runtime moves the mean colour of the hedge texture onto the tint (linear). A cypress is a dark, dull
# green: the `foliage` colour is pulled towards its own grey. The value looks high, but the dense layering
# is darkened a lot by the baked AO and its own shadows: sunlit tips come out near #2b3d20.
FOL_DESAT = 0.5
FOL_VALUE = 2.5
FOL_COOL = (0.86, 1.0, 0.8)     # less red and blue than the grey mix: stays green, not mustard, under amber lanterns
PL_AMP = 0.06     # relief (m) of the upright plumes the foliage surface is broken into
# value (grime mask) of the layers: a very narrow range - the baked AO already darkens the deep layers, and
# the lantern gradient has to carry the form, not the speckle
G_TIP, G_ROOT = 0.0, 0.03       # a spray: tip .. root
G_VAR = 0.04                    # broad patches of slightly darker sprays
G_SCALE = 0.03                  # thatch slivers (plus G_ROOT at their roots)
G_CORE = 0.1                    # the core behind everything
FOL_TRIS = 1640.0               # foliage triangles per m2 of visible crown surface (tuft = 1) ...
FOL_CAP = 23500.0               # ... up to about this many per tree; bigger crowns get coarser foliage instead

META = {"name": "Columnar cypress", "class": "vegetation", "mount": "ground", "focus": 0.22}
PARAMS = {
    "height": {"default": 6.0, "min": 4.0, "max": 8.0, "desc": "overall height (m)"},
    "width": {"default": 1.3, "min": 0.8, "max": 2.0, "desc": "nominal crown diameter at the widest point (m); ragged trees spread a little wider"},
    "irregularity": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "0 = clipped and regular, 1 = ragged: lobes, lean, stray sprays, split leaders"},
    "shape": {"default": "flame", "choices": ["flame", "column", "spire"], "desc": "crown silhouette"},
    "trunk": {"default": 0.8, "min": 0.25, "max": 1.2, "desc": "clear trunk height below the foliage (m)"},
    "foliage": {"default": "#33501f", "desc": "hue of the foliage; the built colour is a darker, duller green of this hue (see FOL_DESAT / FOL_VALUE)"},
    "tuft": {"default": 1.0, "min": 0.7, "max": 1.5, "desc": "relative size of the foliage sprays (1 = ~15 x 4 cm sprays over 26 cm slivers)"},
    "dieback": {"default": 0.25, "min": 0.0, "max": 1.0, "desc": "amount of grey-brown die-back: low hollow patches of bare twigs and a little dead spray"},
    "cones": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of seed cones"},
    "pit": {"default": "kerb", "choices": ["kerb", "soil", "none"], "desc": "tree pit: soil ringed by stone setts / bare soil / none (planter)"},
}
VARIANTS = {
    "tall_spire": {"height": 8.0, "width": 1.05, "shape": "spire", "irregularity": 0.25, "trunk": 1.0, "foliage": "#2c4a20",
                   "dieback": 0.1, "seed": 4},
    "young_golden": {"height": 4.2, "width": 0.95, "shape": "column", "irregularity": 0.12, "trunk": 0.5, "foliage": "#6c6612",
                     "dieback": 0.0, "cones": 0.15, "pit": "soil", "tuft": 0.85, "seed": 11},
    "old_ragged": {"height": 7.2, "width": 1.9, "shape": "flame", "irregularity": 1.0, "trunk": 1.1, "foliage": "#2f4a22",
                   "dieback": 1.0, "cones": 1.0, "seed": 23},
    "stout_planter": {"height": 4.8, "width": 1.7, "shape": "column", "irregularity": 0.6, "trunk": 0.55, "foliage": "#33581d",
                      "dieback": 0.35, "pit": "none", "tuft": 1.15, "seed": 37},
}

# crown silhouettes: (height fraction of the widest point, taper power, taper roundness, skirt fraction)
SHAPES = {"flame": (0.30, 1.7, 0.9, 0.5), "column": (0.2, 3.2, 0.8, 0.72), "spire": (0.16, 1.2, 1.0, 0.55)}
UP = Vector((0.0, 0.0, 1.0))


class _Flame:
    """A flame-shaped foliage envelope around a wandering axis; radius(t, a) is vectorised."""

    def __init__(self, base, z0, z1, width, lean, irr, shape, rng):
        self.bx, self.by = base
        self.z0, self.z1 = z0, z1
        self.hz = z1 - z0                     # height over which foliage elements are rooted (set in build)
        self.w = width
        self.lx, self.ly = lean
        self.irr = irr
        self.bt, self.p, self.q, self.bf = SHAPES[shape]
        self.lobes = [(m, rng.u(-0.7, 0.7), rng.u(0, 2 * pi), amp) for m, amp in ((1, 0.12), (2, 0.15), (3, 0.13), (5, 0.08), (8, 0.05))]
        self.vert = [(rng.u(1.2, 2.2), rng.u(0, 1), 0.08), (rng.u(3.0, 5.0), rng.u(0, 1), 0.05)]
        self.wob = [(rng.u(0, 2 * pi), rng.u(0.8, 1.7), rng.u(-1, 1)) for _ in range(2)]
        self.plume = [(m, rng.u(-1.4, 1.4), rng.u(0, 2 * pi), amp) for m, amp in ((11, 0.05), (17, 0.035))]

    def prof(self, t):
        t = np.clip(t, 0.0, 1.0)
        lo = self.bf + (1.0 - self.bf) * np.sin(0.5 * pi * np.minimum(1.0, t / self.bt))
        s = np.clip((t - self.bt) / (1.0 - self.bt), 0.0, 1.0)
        return lo * (1.0 - s ** self.p) ** self.q

    def axis(self, t):
        k = self.irr * self.w * 0.09
        (p0, f0, a0), (p1, f1, a1) = self.wob
        x = self.bx + self.lx * t + k * a0 * (np.sin(p0 + 2 * pi * f0 * t) - sin(p0)) * t
        y = self.by + self.ly * t + k * a1 * (np.sin(p1 + 2 * pi * f1 * t) - sin(p1)) * t
        return x, y

    def radius(self, t, a):
        n = 0.0
        for m, w, ph, amp in self.lobes:
            n = n + amp * np.sin(m * a + ph + w * 2 * pi * t)
        v = 1.0
        for f, ph, amp in self.vert:
            v = v + self.irr * amp * np.sin(2 * pi * (t * f + ph))
        f = 0.0                                   # upright plumes: fine vertical ridges and furrows
        for m, w, ph, amp in self.plume:
            f = f + amp * np.sin(m * a + ph + w * 2 * pi * t)
        n = 0.42 * np.tanh(self.irr * (1.0 + 0.4 * self.irr) * n / 0.42)          # soft limit: never thinner / fatter than ~40 %
        return 0.5 * self.w * self.prof(t) * v * (1.0 + n + (0.45 + 0.55 * self.irr) * f)

    def area(self):
        ts = np.linspace(0, 1, 60)
        return float(np.mean(self.prof(ts)) * pi * self.w * (self.z1 - self.z0))


def _spray_templates(rng, L0, n=18):
    """Unit-length foliage sprays: the upright tip of a cypress branchlet, a narrow brush / hand fan about a
    third as wide as it is long. 5-7 thin wedge slivers (one slender triangle each, its point at the root)
    leave the root at a tight 13-21 degree total splay; each is pitched and rolled a little out of the plane
    and ends in its own slanted cut at its own height, so the brush is crossed in 3D, its top is a ragged row
    of points and its widest, lightest part is the part that shows when sprays overlap like thatch.
    Base at the origin, tip at +Z, face towards +Y. UVs run along each sliver (see UV_ALONG / UV_ACROSS)."""
    Y = Vector((0.0, 1.0, 0.0))
    ku, kv = L0 * UV_ALONG, L0 * UV_ACROSS
    out = []
    for k in range(n):
        ns = rng.weighted([(5, 0.5), (6, 0.35), (7, 0.15)])
        S = radians(rng.u(13, 21))
        skew = rng.u(-0.2, 0.2) * S
        V, UV, T = [], [], []
        for j in range(ns):
            th = skew + S * ((j + 0.5 + rng.u(-0.3, 0.3)) / ns - 0.5)
            hw = 0.5 * S / ns * rng.u(0.8, 1.35)                     # half angular width of the wedge
            c = 1.0 - 0.3 * (2.0 * (j + 0.5) / ns - 1.0) ** 2             # the middle slivers are the longest
            l = c * rng.u(0.72, 1.0)
            u = Vector((sin(th), rng.u(-0.10, 0.13), cos(th))).normalized()
            side = Y.cross(u).normalized()
            roll = radians(rng.pick([-1, 1]) * rng.u(5, 28))
            sd = side * cos(roll) + u.cross(side).normalized() * sin(roll)
            ka, kb = (1.0, rng.u(0.78, 0.95)) if rng.chance(0.5) else (rng.u(0.78, 0.95), 1.0)
            s0 = rng.u(0.0, 0.2)
            i0 = len(V)
            V.extend([u * s0, u * (l * ka) + sd * (l * hw), u * (l * kb) - sd * (l * hw)])
            T.append((i0, i0 + 2, i0 + 1))                     # wound to face +Y (outwards): the export bakes AO along the normal
            u0, v0 = rng.u(0, 4), rng.u(0, 4)
            UV.extend([(u0 + s0 * ku, v0), (u0 + l * ka * ku, v0 + l * hw * kv), (u0 + l * kb * ku, v0 - l * hw * kv)])
        co = np.array([tuple(v) for v in V], dtype=np.float64)
        co[:, 2] /= co[:, 2].max()
        out.append({"co": co, "uv": np.array(UV, dtype=np.float64), "tris": np.array(T, dtype=np.int32), "tip": np.clip(co[:, 2], 0.0, 1.0)})
    return out


def _blade_templates(rng, L0, W0, n=10):
    """Unit plume blades for the thatch under the sprays: one long pointed triangle, wide at the root; x is
    scaled by the blade width, y / z by its length. Root at the origin, tip at +Z, face towards +Y."""
    out = []
    for k in range(n):
        co = np.array([(0.5, 0.0, rng.u(0.0, 0.12)), (-0.5, 0.0, rng.u(0.0, 0.12)), (rng.u(-0.3, 0.3), rng.u(-0.02, 0.03), 1.0)], dtype=np.float64)
        uv = np.stack([rng.u(0, 4) + co[:, 2] * L0 * UV_ALONG, rng.u(0, 4) + co[:, 0] * W0 * UV_ACROSS], axis=1)
        out.append({"co": co, "uv": uv, "tris": np.array([(0, 1, 2)], dtype=np.int32), "tip": np.clip(co[:, 2], 0.0, 1.0)})
    return out


class _Sprays:
    """Collects instances of unit templates and bakes them into one mesh with per-vertex grime (dark root,
    lighter tip) and blade-aligned UVs. (Custom normals do not survive the export's weighted-normal pass, so
    the blades themselves are kept close to the envelope tangent plane instead.)"""

    def __init__(self, templates, rng):
        self.tp = templates
        self.rng = rng
        self.items = [[] for _ in templates]

    def add(self, org, x, y, d, g_base, g_root=G_ROOT):
        k = self.rng.pick(range(len(self.tp)))
        self.items[k].append((tuple(x) + tuple(y) + tuple(d), tuple(org), g_base, g_root, self.rng.u(0, 7), self.rng.u(0, 7)))

    def count(self):
        return sum(len(i) for i in self.items)

    def tris(self):
        return sum(len(i) * len(tp["tris"]) for tp, i in zip(self.tp, self.items))

    def bake(self, name, material):
        CO, TR, UVS, GR = [], [], [], []
        base = 0
        for tp, its in zip(self.tp, self.items):
            if not its:
                continue
            m, nv = len(its), len(tp["co"])
            A = np.array([i[0] for i in its], dtype=np.float64).reshape(m, 3, 3)       # rows: x, y, d (scaled)
            org = np.array([i[1] for i in its], dtype=np.float64)
            gb = np.array([i[2] for i in its], dtype=np.float64)
            gr = np.array([i[3] for i in its], dtype=np.float64)
            uo = np.array([(i[4], i[5]) for i in its], dtype=np.float64)
            co = np.einsum("vj,mji->mvi", tp["co"], A) + org[:, None, :]
            g = gb[:, None] + gr[:, None] * (1.0 - tp["tip"][None, :]) ** 1.4
            CO.append(co.reshape(-1, 3))
            GR.append(np.clip(g, 0.0, 1.0).reshape(-1))
            UVS.append((tp["uv"][None, :, :] + uo[:, None, :]).reshape(-1, 2))
            TR.append((tp["tris"][None, :, :] + (base + np.arange(m) * nv)[:, None, None]).reshape(-1, 3))
            base += m * nv
        if not CO:
            return None
        co, gr, uv, tr = np.concatenate(CO), np.concatenate(GR), np.concatenate(UVS), np.concatenate(TR)
        o = mesh_object(name, [], [])
        me = o.data
        me.vertices.add(len(co))
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        me.loops.add(tr.size)
        me.loops.foreach_set("vertex_index", tr.astype(np.int32).ravel())
        me.polygons.add(len(tr))
        me.polygons.foreach_set("loop_start", np.arange(0, tr.size, 3, dtype=np.int32))
        me.update(calc_edges=True)
        me.polygons.foreach_set("use_smooth", np.ones(len(tr), dtype=bool))
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", uv[tr.ravel()].astype(np.float32).ravel())
        me.attributes.new("m_grime", 'FLOAT', 'POINT').data.foreach_set("value", gr.astype(np.float32))
        me.update()
        o["tsq_uv"] = "keep"
        o["tsq_smooth"] = "keep"
        assign(o, material)
        return o


def _foliage_look(hexcol):
    """Foliage colour param -> tint (linear). The hue of the param is kept, but the colour is pulled towards
    its own grey: a cypress is a sombre, dull green, and under warm lantern light a saturated albedo turns
    mustard."""
    c = np.array(hex_rgb(hexcol), dtype=np.float64)
    y = float(c @ np.array([0.2126, 0.7152, 0.0722]))
    t = FOL_VALUE * (y + (c - y) * (1.0 - FOL_DESAT)) * np.array(FOL_COOL)
    return tuple(float(x) for x in t)


def _grain_uv(o, fl):
    """UVs of a shell around envelope fl: metres around the girth (finer) x metres of height (stretched), so
    the foliage texture of the shell has the same upright grain as the sprays lying on it."""
    me = o.data
    vi = np.zeros(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ax, ay = fl.axis((co[:, 2] - fl.z0) / (fl.z1 - fl.z0))
    a = np.arctan2(co[:, 1] - ay, co[:, 0] - ax)[vi]
    lt = np.zeros(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    ls = np.zeros(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    ref = np.repeat(a[ls], lt)                              # unwrap the angle within each face
    a = ref + (a - ref + pi) % (2 * pi) - pi
    uv = np.stack([a * (0.36 * fl.w) * UV_ACROSS, co[vi, 2] * UV_ALONG], axis=1)
    (me.uv_layers.active or me.uv_layers.new(name="UVMap")).data.foreach_set("uv", uv.astype(np.float32).ravel())
    o["tsq_uv"] = "keep"


def _frame(d, ref, roll):
    """Orthonormal (x, y) around direction d with y as close as possible to `ref` (outwards), rolled."""
    x = ref.cross(d)
    if x.length < 1e-4:
        x = Vector((1.0, 0.0, 0.0)).cross(d)
    x.normalize()
    y = d.cross(x)
    return x * cos(roll) + y * sin(roll), y * cos(roll) - x * sin(roll)


def build(p, rng):
    H, W, irr = p.height, p.width, p.irregularity
    zb = min(p.trunk, H * 0.2)
    Hf = H - zb
    pit = p.pit

    # ------------------------------------------------------------------ crown envelopes
    r_ax = rng.fork("axis")
    la = r_ax.u(0, 2 * pi)
    lm = irr * r_ax.u(0.4, 1.0) * 0.04 * Hf
    main = _Flame((0.0, 0.0), zb, H, W * (1.0 - 0.1 * irr), (lm * cos(la), lm * sin(la)), irr, p.shape, rng.fork("main"))
    flames = [main]
    r_sec = rng.fork("leaders")
    nsec = 0 if irr < 0.3 else (1 if irr < 0.7 else 2)
    a_sec = r_sec.u(0, 2 * pi)
    for k in range(nsec):
        a = a_sec + k * r_sec.u(2.0, 4.0)
        t0 = r_sec.u(0.30, 0.50)
        R0 = float(main.radius(t0, a))
        ax, ay = main.axis(t0)
        top = H - r_sec.u(0.10, 0.24) * Hf
        t1 = (top - zb) / Hf
        ws = R0 * r_sec.u(1.0, 1.3)
        base = (float(ax) + cos(a) * R0 * 0.45, float(ay) + sin(a) * R0 * 0.45)
        tx, ty = main.axis(t1)
        tipr = float(main.radius(t1, a)) + ws * 0.36 + irr * 0.08
        lean = (float(tx) + cos(a) * tipr - base[0], float(ty) + sin(a) * tipr - base[1])
        flames.append(_Flame(base, zb + t0 * Hf, top, ws, lean, irr * 0.6, "spire", r_sec.fork("f%d" % k)))

    # splayed side branches that have fallen out of the column (old, unclipped trees)
    r_sb = rng.fork("side")
    for k in range(int(round(irr * 6))):
        a = r_sb.u(0, 2 * pi)
        t0 = r_sb.u(0.08, 0.66)
        R0 = float(main.radius(t0, a))
        ax, ay = main.axis(t0)
        z0s = zb + t0 * Hf
        top = min(z0s + r_sb.u(0.12, 0.2) * Hf, H - 0.3)
        t1 = (top - zb) / Hf
        ws = r_sb.u(0.28, 0.42) * sqrt(W / 1.3)
        base = (float(ax) + cos(a) * R0 * 0.78, float(ay) + sin(a) * R0 * 0.78)
        tx, ty = main.axis(t1)
        tipr = float(main.radius(t1, a)) + r_sb.u(0.12, 0.32) * (0.4 + 0.6 * irr) * sqrt(W / 1.3) * min(1.0, Hf / 5.0)
        lean = (float(tx) + cos(a) * tipr - base[0], float(ty) + sin(a) * tipr - base[1])
        flames.append(_Flame(base, z0s, top, ws, lean, irr * 0.5, "spire", r_sb.fork("s%d" % k)))

    # ------------------------------------------------------------------ materials (one foliage, one dead / cone)
    m_fol = mat("hedge", _foliage_look(p.foliage), double=1, bump=0.6)
    m_dead = mat("wood_weathered", "#5e5444", rough=1.2)
    m_bark = mat("bark")

    # ------------------------------------------------------------------ element sizes (provisional, see below)
    areas = [f.area() for f in flames]
    ks = p.tuft
    for fl in flames:
        fl.hz = max(fl.z1 - fl.z0 - 0.3 * ks, 0.25)

    # ------------------------------------------------------------------ die-back patches and thin dents
    r_die = rng.fork("dieback")
    patches = []
    for k in range(int(round(p.dieback * 4))):
        hi = 0.5 if k == 3 else 0.34                        # mostly low on the crown
        patches.append((r_die.u(0.07, hi), r_die.u(0, 2 * pi), r_die.u(0.28, 0.42) * sqrt(W / 1.3) * (0.85 + 0.3 * p.dieback)))
    r_gap = rng.fork("gaps")                 # dents where a branch has gone: the foliage is thin and set deep
    gaps = [(r_gap.u(0.12, 0.8), r_gap.u(0, 2 * pi), r_gap.u(0.2, 0.36) * sqrt(W / 1.3)) for _ in range(int(round(max(0.0, irr - 0.55) * 4.5)))]

    def hollow(t, a, R, hf):
        """(die-back, dent) amounts 0..1 at a point of the main crown."""
        dv = gv = 0.0
        for (pt, pa, pr) in patches:
            da = (a - pa + pi) % (2 * pi) - pi
            dv = max(dv, 1.0 - hypot((t - pt) * hf, da * max(R, 0.1)) / (pr * (0.82 + 0.2 * sin(7 * a + 23 * t))))
        for (gt, ga, gr) in gaps:
            da = (a - ga + pi) % (2 * pi) - pi
            gv = max(gv, 1.0 - hypot((t - gt) * hf * 0.6, da * max(R, 0.1)) / gr)
        return max(dv, 0.0), max(gv, 0.0)

    def bare(dv, t, a):
        """Inside a die-back hollow: no live foliage at all (the rim is ragged)."""
        return dv > 0.16 + 0.08 * sin(9 * a + 31 * t)

    # ------------------------------------------------------------------ the foliage surface
    # skirt: the crown does not end in a level cut - its lower edge hangs in lobes and tapers in to the limbs
    r_sk = rng.fork("skirt")
    sk = [(m, r_sk.u(0, 2 * pi), amp) for m, amp in ((2, 0.5), (3, 0.3), (5, 0.2))]
    sk_lo, sk_amp = 0.03, (0.12 + 0.2 * irr) * min(1.0, Hf / 5.0)

    def zl(a):
        """Height (m above the crown base) of the lower edge of the foliage in direction a."""
        return sk_lo + sk_amp * (0.5 + 0.5 * sum(amp * sin(m * a + ph) for m, ph, amp in sk))

    # plume relief: upright, staggered pillows (the ascending branch ends of a cypress) with furrows between
    r_pl = rng.fork("plumes")
    ptab = []
    for fl in flames:
        ncol = max(4, int(round(2 * pi * 0.42 * fl.w / 0.34)))
        ptab.append([(r_pl.u(0, 3), r_pl.u(0.55, 1.0) * (0.75 + 0.05 * H)) for _ in range(ncol)])

    def plume(fi, t, a, R):
        fl, cols = flames[fi], ptab[fi]
        z = t * (fl.z1 - fl.z0)
        c = ((a + 0.1 * sin(2.1 * z + fi)) / (2 * pi)) % 1.0 * len(cols)
        ph, P = cols[int(c) % len(cols)]
        sv = ((z + ph) / P) % 1.0
        q = 1.0 - (abs(2.0 * (c - int(c)) - 1.0) / (1.0 - 0.6 * smoothstep(0.45, 1.0, sv))) ** 2
        pil = max(q, 0.0) ** 0.7 * smoothstep(0.0, 0.18, sv) * (1.0 - smoothstep(0.82, 1.0, sv))
        return PL_AMP * (0.7 + 0.5 * irr) * min(1.0, R / 0.35) * (pil - 0.45)

    def surf(fi, t, a):
        """Radius of the foliage surface (sprays' tips) of flame fi at height fraction t, direction a."""
        fl = flames[fi]
        R = float(fl.radius(t, a))
        if fi == 0:
            R *= 0.74 + 0.26 * smoothstep(0.0, 0.45, t * fl.hz - zl(a))
        else:
            R *= 0.3 + 0.7 * smoothstep(0.0, 0.16, t)
        return max(R + plume(fi, t, a, R), 0.03 * ks * smoothstep(0.5, 0.8, t))        # the leader never thins to a bare stem

    def slope(fi, t, a):
        dt = 0.05 / flames[fi].hz
        t0, t1 = max(t - dt, 0.0), min(t + dt, 1.0)
        return (surf(fi, t1, a) - surf(fi, t0, a)) / ((t1 - t0) * flames[fi].hz)

    def buried(fi, P, margin):
        """True if a point of flame fi lies well inside another flame (hidden foliage is not built)."""
        for fj, fl in enumerate(flames):
            if fj == fi or not (fl.z0 < P.z < fl.z0 + fl.hz):
                continue
            t = (P.z - fl.z0) / fl.hz
            ax, ay = fl.axis(t)
            dx, dy = P.x - float(ax), P.y - float(ay)
            if surf(fj, t, atan2(dy, dx)) - hypot(dx, dy) > margin:
                return True
        return False

    def at(fi, t, a, r):
        fl = flames[fi]
        ax, ay = fl.axis(t)
        return Vector((float(ax) + cos(a) * r, float(ay) + sin(a) * r, fl.z0 + t * fl.hz))

    # ------------------------------------------------------------------ element sizes from the triangle budget
    # Only crown surface that can be seen is built (side leaders are half buried in the main flame), so the
    # budget is spread over the visible area: big / old crowns get coarser foliage instead of more triangles.
    r_ar = rng.fork("area")
    vis = []
    for fi, fl in enumerate(flames):
        ts = (np.arange(24) + 0.5) / 24.0
        rad = fl.prof(ts) * 0.5 * fl.w
        wt = (rad + 0.05) * np.sqrt(1.0 + np.clip(np.gradient(rad, ts * fl.hz), -4.0, 4.0) ** 2)
        seen = 0.0
        for t, w in zip(ts, wt):
            t = float(t)
            for j in range(8):
                a = 2 * pi * (j + r_ar.u(0, 1)) / 8.0
                if fi == 0 and t * fl.hz < zl(a):
                    continue
                if not buried(fi, at(fi, t, a, max(surf(fi, t, a) - 0.08 * ks, 0.0)), 0.09):
                    seen += w / 8.0
        vis.append(areas[fi] * seen / float(wt.sum()))
    A = sum(vis)
    t_want = FOL_TRIS * A / p.tuft ** 2
    t_fol = min(t_want, FOL_CAP * (max(A, 12.0) / 12.0) ** 0.25 / sqrt(p.tuft), 31000.0)
    ks = p.tuft * sqrt(t_want / t_fol)
    L0 = 0.15 * ks                                # spray length
    Ls, Ws = 0.26 * ks, 0.055 * ks                # thatch blade length / width
    Ltop = 1.1 * Ls                               # everything is rooted this far below a leader's tip
    for fl in flames:
        fl.hz = max(fl.z1 - fl.z0 - Ltop, 0.25)

    sprays = _spray_templates(rng.fork("shapes"), L0)
    live = _Sprays(sprays, rng.fork("pick_live"))
    dead = _Sprays(sprays, rng.fork("pick_dead"))
    blades = _blade_templates(rng.fork("blades"), Ls, Ws)
    thatch = _Sprays(blades, rng.fork("pick_blade"))
    husk = _Sprays(blades, rng.fork("pick_husk"))
    ang8 = np.linspace(0.0, 2 * pi, 8, endpoint=False)

    def lattice(fi, dz, dcol, r, dens=None):
        """Jittered brick lattice over flame fi (rows dz apart along the surface, columns dcol apart round the
        girth): even cover with no clumps and no holes. Yields (t, a)."""
        fl = flames[fi]
        a0 = r.u(0, 2 * pi)
        z, row = 0.0, 0
        while z < fl.hz + 0.4 * dz:
            tr = clamp(z / fl.hz)
            Rn = float(np.mean(fl.radius(tr, ang8)))
            sn = (float(np.mean(fl.radius(min(tr + 0.05 / fl.hz, 1.0), ang8))) - Rn) / 0.05
            k = dens(tr) if dens else 1.0
            n_a = max(3, int(ceil(2 * pi * max(Rn - 0.06 * ks, 0.02) / dcol)))
            for j in range(n_a):
                yield clamp((z + r.u(-0.3, 0.3) * dz / k) / fl.hz), a0 + 2 * pi * (j + 0.5 * (row % 2) + r.u(-0.28, 0.28)) / n_a
            z += dz / k / min(sqrt(1.0 + sn * sn), 3.0)      # rows follow the surface, so domed tops are covered too
            row += 1

    # ------------------------------------------------------------------ thatch: a closed coat of upright slivers
    # Long pointed slivers, rooted deep and lifted at the tip so each row covers the roots of the next like
    # thatch: the solid, deep green, upright-grained ground the sprays stand in. Nothing behind it is seen.
    r_sc = rng.fork("thatch")
    for fi, fl in enumerate(flames):
        for t, a in lattice(fi, Ls / 3.0, 0.65 * Ws, r_sc):
            if fi == 0 and t * fl.hz < zl(a):
                continue
            R = surf(fi, t, a)
            dv, gv = hollow(t, a, R, fl.hz) if fi == 0 else (0.0, 0.0)
            L = Ls * r_sc.u(0.8, 1.15)
            w = Ws * r_sc.u(0.8, 1.25) * clamp(R / 0.14, 0.5, 1.0)
            if bare(dv, t, a):
                # the back of a die-back hollow: dead grey-brown slivers still hanging deep inside, behind the twigs
                if r_sc.chance(0.7):
                    inset = min((r_sc.u(0.1, 0.17) + 0.05 * smoothstep(0.0, 0.6, dv)) * ks + 0.05, 0.7 * R)
                    out, tan = Vector((cos(a), sin(a), 0.0)), Vector((-sin(a), cos(a), 0.0))
                    d = (out * (clamp(slope(fi, t, a), -1.0, 0.5) + r_sc.u(-0.1, 0.25)) + tan * r_sc.u(-0.3, 0.3) + UP * r_sc.pick([1.0, 1.0, 0.5])).normalized()
                    x, y = _frame(d, out, r_sc.u(-0.6, 0.6))
                    husk.add(at(fi, t, a, R - inset), x * (w * 0.8), y * L, d * (L * r_sc.u(0.6, 0.9)), 0.25 + r_sc.u(0.0, 0.2), 0.2)
                continue
            inset = min((r_sc.u(0.055, 0.085) + 0.05 * gv) * ks, 0.62 * R)
            kout = clamp(slope(fi, t, a), -1.2, 0.5) + (inset - 0.02 * ks) / L
            out, tan = Vector((cos(a), sin(a), 0.0)), Vector((-sin(a), cos(a), 0.0))
            d = (out * kout + tan * r_sc.u(-0.12, 0.12) + UP).normalized()
            org = at(fi, t, a, R - inset)
            if buried(fi, org + d * (0.5 * L), 0.09):
                continue
            x, y = _frame(d, out, r_sc.u(-0.3, 0.3))
            thatch.add(org, x * w, y * L, d * L, G_SCALE + 0.03 * sin(2 * a + 7 * t) + r_sc.u(-0.02, 0.02))

    # ------------------------------------------------------------------ sprays: the upright grain
    def size_at(t):
        return (0.9 + 0.2 * smoothstep(0.08, 0.55, t)) * (1.0 - 0.25 * smoothstep(0.85, 1.0, t))

    def dens_at(t):
        return 1.0 + 0.5 * smoothstep(0.58, 0.8, t)             # closer rows towards the leader

    cone_pts = []
    twigs = []
    r_t = rng.fork("tufts")
    for fi, fl in enumerate(flames):
        hz = fl.hz
        cph = r_t.u(0, 2 * pi)
        for t, a in lattice(fi, 0.5 * L0, 0.34 * L0, r_t, dens_at):
            zr = t * hz - zl(a) if fi == 0 else 1.0           # height above the skirt edge
            if zr < 0.0:
                continue
            R = surf(fi, t, a)
            dv, gv = hollow(t, a, R, hz) if fi == 0 else (0.0, 0.0)
            if bare(dv, t, a) or r_t.chance(0.6 * smoothstep(0.0, 0.4, gv)):
                continue
            is_dead = dv > 0.0 and r_t.chance(clamp(2.5 * dv, 0.0, 0.5))      # a thin grey-brown rim round a hollow
            L = L0 * size_at(t) * r_t.u(0.82, 1.2)
            # About one spray in eight is a fin: turned edge-on to the surface, so the outline is a feathered
            # fringe rooted in the mass (flat-lying sprays vanish edge-on at the limb).
            fin = r_t.chance(0.12)
            # thatch: every spray is swept up within ~25 degrees of vertical, its root tucked into the slivers
            # and under the sprays below, its tip lifted clear
            kout = clamp(clamp(slope(fi, t, a), -0.5, 0.45) + (r_t.u(0.12, 0.3) if fin else r_t.u(0.05, 0.26)), -0.45, 0.47)
            if zr < 0.12:
                kout += r_t.u(0.0, 0.2)                       # the skirt flares a little
            ktan = r_t.u(-0.12, 0.12)
            depth = min(((r_t.u(0.025, 0.045) if fin else r_t.u(0.01, 0.045)) + 0.04 * gv) * ks, 0.55 * R)
            out, tan = Vector((cos(a), sin(a), 0.0)), Vector((-sin(a), cos(a), 0.0))
            d = (out * kout + tan * ktan + UP).normalized()
            org = at(fi, t, a, R - depth)
            if buried(fi, org + d * (0.5 * L), 0.07):
                continue
            roll = r_t.pick([-1, 1]) * (0.5 * pi + r_t.u(-0.35, 0.35)) if fin else r_t.u(-0.35, 0.35)
            x, y = _frame(d, out, roll)
            # value: low-frequency patches (not salt-and-pepper), a little darker when set deep
            gb = G_TIP + G_VAR * (0.5 + 0.3 * sin(2 * a + 7 * t + cph) + 0.2 * sin(5 * a - 11 * t + 2 * cph)) + r_t.u(-0.01, 0.01) + 0.5 * depth
            if is_dead:
                dead.add(org, x * (L * 0.8), y * L, d * (L * r_t.u(0.75, 1.0)), 0.1, 0.2)
                continue
            live.add(org, x * (L * r_t.vary(1.0, 0.12)), y * L, d * L, gb)
            if not fin and gv <= 0.0 and dv <= 0.0 and depth > 0.03 * ks:
                cone_pts.append((org + d * (L * 0.15) + y * 0.008, y))

        # the leader: a tapering brush of sprays up the last stretch of the stem, closing to a single point
        ax, ay = fl.axis(1.0)
        n_l = 18
        for j in range(n_l):
            f = j / (n_l - 1.0)                               # 0 at the foot of the brush, 1 at the tip
            L = L0 * lerp(1.0, 0.75, f) * r_t.u(0.85, 1.1)
            a = j * 2.39996 + r_t.u(-0.5, 0.5)
            out = Vector((cos(a), sin(a), 0.0))
            d = (out * lerp(r_t.u(0.1, 0.3), r_t.u(0.0, 0.08), f) + UP).normalized()
            org = Vector((float(ax), float(ay), fl.z0 + hz + lerp(-0.4 * L0, Ltop - L, f))) + out * 0.01
            if f < 0.6 and buried(fi, org + d * (0.5 * L), 0.07):
                continue
            x, y = _frame(d, out, r_t.u(-0.5, 0.5))
            live.add(org, x * (L * lerp(1.0, 0.7, f)), y * L, d * L, G_TIP + 0.5 * G_VAR)

        # strays: a short bare twig leaning out of the thatch with a few overlapping sprays along it
        for k in range(int(round((1.0 + 12.0 * irr) * vis[fi] / 12.0 + r_t.u(0, 1)))):
            t, a = r_t.u(0.12, 0.88), r_t.u(0, 2 * pi)
            if fi == 0 and t * hz < zl(a) + 0.2:
                continue
            R = surf(fi, t, a)
            dv, gv = hollow(t, a, R, hz) if fi == 0 else (0.0, 0.0)
            if dv > 0.0 or R < 0.12:
                continue
            out, tan = Vector((cos(a), sin(a), 0.0)), Vector((-sin(a), cos(a), 0.0))
            ds = (out * (r_t.u(0.3, 0.55) * (0.6 + 0.5 * irr)) + tan * r_t.u(-0.25, 0.25) + UP).normalized()
            P0 = at(fi, t, a, R - min(0.09 * ks, 0.5 * R))
            if buried(fi, P0, 0.0):
                continue
            ln = r_t.u(0.16, 0.26) * sqrt(ks)
            P1 = P0 + ds * (0.5 * ln) + out * (0.03 * ln)
            P2 = P0 + ds * ln + UP * (0.06 * ln)
            tw = sweep("stray%d_%d" % (fi, k), [tuple(P0), tuple(P1), tuple(P2)], r_t.u(0.004, 0.006), sides=3, smooth_path=False,
                       scale_fn=lambda u: 1.0 - 0.5 * u, cap=False)
            uv_transform(tw, rot_deg=90)
            assign(tw, m_bark)
            twigs.append(tw)
            ns = r_t.pick([4, 5, 5, 6])
            for j in range(ns):
                f = lerp(0.25, 0.97, j / (ns - 1.0))
                q = P0.lerp(P1, f / 0.5) if f < 0.5 else P1.lerp(P2, (f - 0.5) / 0.5)
                Lj = L0 * r_t.u(0.85, 1.15) * (1.1 - 0.25 * f)
                dj = (ds + Vector(r_t.vec(0.16)) + UP * 0.15).normalized()
                xj, yj = _frame(dj, out, r_t.u(-1.3, 1.3))
                live.add(q - dj * (0.05 * Lj), xj * Lj, yj * Lj, dj * Lj, G_TIP + 0.3 * G_VAR)

    # the skirt: a ragged ring of slivers and sprays flaring out of the lower edge, some of them drooping
    r_k = rng.fork("skirt_ring")
    hz0 = main.hz
    n_sk = max(8, int(round(2 * pi * 0.74 * float(np.mean(main.radius(0.02, ang8))) / (0.06 * ks))))
    for i in range(n_sk):
        a = 2 * pi * (i + r_k.u(-0.3, 0.3)) / n_sk
        t = (zl(a) + r_k.u(0.0, 0.03)) / hz0
        R = surf(0, t, a)
        dv, gv = hollow(t, a, R, hz0)
        if bare(dv, t, a):
            continue
        out, tan = Vector((cos(a), sin(a), 0.0)), Vector((-sin(a), cos(a), 0.0))
        droop = r_k.chance(0.3)
        d = (out * r_k.u(0.4, 0.75) + tan * r_k.u(-0.3, 0.3) + UP * (r_k.u(-0.45, -0.05) if droop else r_k.u(0.4, 1.0))).normalized()
        L = Ls * r_k.u(0.45, 0.7)
        org = at(0, t, a, max(R - 0.1 * ks, 0.45 * R)) + UP * r_k.u(0.02, 0.06)
        ref = out if droop else (out * 0.5 + UP).normalized()
        x, y = _frame(d, ref, r_k.u(-0.3, 0.3))
        thatch.add(org, x * (Ws * r_k.u(0.9, 1.3)), y * L, d * L, G_SCALE + 0.04)
        for j in range(r_k.pick([1, 2, 2])):
            Lj = L0 * r_k.u(0.8, 1.15)
            dj = (d + Vector(r_k.vec(0.3))).normalized()
            xj, yj = _frame(dj, ref, r_k.u(-0.8, 0.8))
            live.add(org + d * (L * r_k.u(0.2, 0.7)) + y * 0.006, xj * Lj, yj * Lj, dj * Lj, G_TIP + G_VAR)

    # die-back: bare grey twigs reaching up through the hollow, a few of them still carrying a thin dead spray
    r_tw = rng.fork("twigs")
    for pi_, (pt, pa, pr) in enumerate(patches):
        for k in range(int(round(8 + 22 * pr))):
            rho, phi = pr * 0.85 * sqrt(r_tw.u(0, 1)), r_tw.u(0, 2 * pi)
            t = clamp(pt + rho * sin(phi) / hz0, 0.03, 0.95)
            R0 = max(surf(0, t, pa), 0.1)
            a = pa + rho * cos(phi) / R0
            R = surf(0, t, a)
            ca, sa = cos(a), sin(a)
            ax, ay = main.axis(t)
            z1 = main.z0 + t * hz0
            ln = r_tw.u(0.24, 0.42)
            r1 = max(R - r_tw.u(0.02, 0.08), 0.02)
            r0 = max(R * 0.4, r1 - ln * 0.8)
            P0 = Vector((float(ax) + ca * r0, float(ay) + sa * r0, z1 - ln * 0.8))
            P2 = Vector((float(ax) + ca * r1, float(ay) + sa * r1, z1))
            P1 = P0.lerp(P2, 0.55) + Vector((ca, sa, 0.0)) * (ln * 0.12) + Vector(r_tw.vec(0.02))
            tw = sweep("twig%d_%d" % (pi_, k), [tuple(P0), tuple(P1), tuple(P2)], r_tw.u(0.009, 0.015), sides=3, smooth_path=False,
                       scale_fn=lambda u: 1.0 - 0.6 * u, cap=False)
            uv_transform(tw, rot_deg=90)
            assign(tw, m_bark)
            twigs.append(tw)
            dend = (P2 - P1).normalized()
            for j in range(r_tw.pick([0, 0, 1, 1, 2])):
                f = 1.0 if j == 0 else r_tw.u(0.45, 0.85)
                q = P1.lerp(P2, f) if f > 0.55 else P0.lerp(P1, f / 0.55)
                L = L0 * r_tw.u(0.6, 0.9)
                dj = (dend + Vector(r_tw.vec(0.0 if j == 0 else 0.4)) + Vector((ca, sa, 0.0)) * 0.15).normalized()
                xj, yj = _frame(dj, Vector((ca, sa, 0.0)), r_tw.u(-0.6, 0.6))
                dead.add(q - dj * (L * 0.08), xj * (L * 0.8), yj * L, dj * L, 0.15, 0.25)
    if twigs:
        paint(twigs, grime=0.3)

    live.bake("sprays", m_fol)
    thatch.bake("thatch", m_fol)
    dead.bake("sprays_dead", m_dead)
    husk.bake("husk", m_dead)
    print("DBG area %.1f of %.1f m2  ks %.2f  slivers %d  sprays %d  dead %d  twigs %d  foliage tris %d (budget %d)" % (
        A, sum(areas), ks, thatch.count(), live.count(), dead.count(), len(twigs), thatch.tris() + live.tris() + dead.tris() + husk.tris(), t_fol))

    # ------------------------------------------------------------------ core: only blocks the light behind the thatch
    nz = noise_mask(0.16, seed=p.seed + 31, lo=0.25, hi=0.75)
    zc = sk_lo + sk_amp + 0.14                         # the underside is a hollow the limbs rise into
    for fi, fl in enumerate(flames):

        def core_fn(u, v, fl=fl, fi=fi):
            a = 2 * pi * u
            if fi == 0:
                t_lo = zl(a) / fl.hz
                if v < 0.1:
                    s = v / 0.1
                    R = surf(0, t_lo, a)
                    r = lerp(0.02, max(R - min(0.115 * ks, 0.6 * R), 0.02), s)
                    return (fl.bx + r * cos(a), fl.by + r * sin(a), fl.z0 + lerp(zc, zl(a) + 0.03, s * s))
                t = lerp(t_lo, 1.0, (v - 0.1) / 0.9)
            else:
                t = v
            R = surf(fi, t, a)
            dv, gv = hollow(t, a, R, fl.hz) if fi == 0 else (0.0, 0.0)
            Rc = max(R - min(0.115 * ks, 0.6 * R) - (0.1 * ks + 0.06) * smoothstep(0.0, 0.5, dv), 0.25 * R, 0.012)
            if fi:
                Rc *= min(1.0, v / 0.08)
            ax, ay = fl.axis(t)
            return (float(ax) + Rc * cos(a), float(ay) + Rc * sin(a), fl.z0 + t * fl.hz + 0.03)

        nu, nv = (26, max(16, int(round(fl.hz / 0.2)))) if fi == 0 else (10, max(5, int(round(fl.hz / 0.3))))
        core = surface("core%d" % fi, core_fn, nu, nv, closed_u=True)
        recalc_normals(core)
        smooth(core, 180)
        _grain_uv(core, fl)
        assign(core, m_fol)

        def core_grime(co, n, fl=fl, fi=fi):
            g = G_CORE + 0.1 * nz(co, n)
            if fi == 0 and patches:                      # the hollow behind dead foliage is dark and bare
                for i, (x, y, z) in enumerate(co):
                    t = clamp((z - fl.z0) / fl.hz)
                    ax, ay = fl.axis(t)
                    a = atan2(y - float(ay), x - float(ax))
                    g[i] = max(g[i], 0.7 * smoothstep(0.0, 0.5, hollow(t, a, surf(0, t, a), fl.hz)[0]))
            return g

        paint(core, grime=core_grime)

    # ------------------------------------------------------------------ trunk: buttressed, fluted, slightly bowed
    r_tr = rng.fork("trunk")
    rb = 0.022 + 0.018 * H
    flare, butt = (0.7, 0.55) if pit != "none" else (0.12, 0.08)       # planter stock: clean straight stem, no root flare
    zt = zb + max(0.45, 0.1 * Hf)
    zs = [0.0, 0.02, 0.045, 0.08, 0.13, 0.2, 0.3]
    while zs[-1] < zt - 0.16:
        zs.append(zs[-1] + 0.14)
    zs.append(zt)

    def r_trunk(z):
        return rb * (1.0 + flare * np.exp(-z / 0.085)) * lerp(1.0, 0.7, z / zt)

    trunk = lathe("trunk", [(float(r_trunk(z)), z) for z in zs] + [(0.0, zt + 0.01)], segments=22, smooth_angle=70)
    nroot = r_tr.pick([4, 5, 5, 6])
    root_a = [2 * pi * (k + r_tr.u(-0.25, 0.25)) / nroot for k in range(nroot)]
    flute = [(m, r_tr.u(0, 2 * pi), amp) for m, amp in ((3, 0.07), (5, 0.05), (9, 0.025))]
    bow = (r_tr.u(-1, 1) * 0.03, r_tr.u(-1, 1) * 0.03)

    def warp_trunk(co):
        a = atan2(co.y, co.x)
        z = co.z
        k = 1.0
        for m, ph, amp in flute:
            k += amp * sin(m * a + ph + z * 1.3)
        but = 0.0
        for ra in root_a:
            dd = (a - ra + pi) % (2 * pi) - pi
            but = max(but, np.exp(-(dd / 0.33) ** 2))
        k += butt * but * np.exp(-z / 0.11)
        f = (z / zt) ** 2
        return Vector((co.x * k + bow[0] * f, co.y * k + bow[1] * f, z))

    transform_verts(trunk, warp_trunk)
    displace_noise(trunk, strength=0.006, scale=0.09, seed=p.seed, axis_mask=(1, 1, 0))
    assign(trunk, m_bark)
    woody = [trunk]

    # surface roots running out through the soil
    rp = max(0.42, 0.34 * W + 0.12)                      # tree pit radius
    if pit != "none":
        for k, ra in enumerate(root_a):
            reach = min(rp * 0.92, r_tr.u(0.45, 0.85) * rp + rb)
            bend = r_tr.u(-0.35, 0.35)
            pts = []
            for i in range(7):
                u = i / 6.0
                rr = lerp(rb * 1.25, reach, u)
                aa = ra + bend * u * u
                pts.append((rr * cos(aa), rr * sin(aa), lerp(0.062, 0.016, u ** 0.7)))
            r0 = rb * r_tr.u(0.28, 0.4)
            root = sweep("root%d" % k, pts, r0, sides=7, smooth_path=False, scale_fn=lambda u: lerp(1.0, 0.25, u ** 0.8))
            transform_verts(root, lambda co: Vector((co.x, co.y, max(co.z, 0.002))))
            uv_transform(root, rot_deg=90)
            assign(root, m_bark)
            woody.append(root)

    # bare limbs forking off the stem and rising steeply into the skirt: the foliage is visibly carried by them
    r_br = rng.fork("branches")
    nbr = r_br.pick([3, 4, 4, 5])
    a_br = r_br.u(0, 2 * pi)
    for k in range(nbr):
        a = a_br + 2 * pi * (k + r_br.u(-0.3, 0.3)) / nbr
        zr = zb * r_br.u(0.3, 0.75)
        Rk = surf(0, zl(a) / main.hz, a) * r_br.u(0.72, 0.92)
        rt = float(r_trunk(zr)) * 0.7
        ze = zb + zl(a) + r_br.u(0.12, 0.3)
        a1 = a + r_br.u(-0.3, 0.3)
        rm = lerp(rt, Rk, 0.6)
        pts = bezier_path((rt * cos(a), rt * sin(a), zr), (rm * cos(a), rm * sin(a), zr + 0.08),
                          (Rk * 0.95 * cos(a1), Rk * 0.95 * sin(a1), lerp(zr, ze, 0.45)), (Rk * cos(a1), Rk * sin(a1), ze), n=8)
        br = sweep("branch%d" % k, pts, rb * r_br.u(0.2, 0.3), sides=6, smooth_path=False, scale_fn=lambda u: 1.0 - 0.5 * u)
        uv_transform(br, rot_deg=90)
        assign(br, m_bark)
        woody.append(br)
    # snapped dead twig stubs on the clear trunk
    for k in range(r_br.pick([1, 2, 3])):
        a = r_br.u(0, 2 * pi)
        zr = zb * r_br.u(0.3, 0.85)
        rt = float(r_trunk(zr)) * 0.75
        ln = r_br.u(0.05, 0.13)
        st = sweep("stub%d" % k, [(rt * cos(a), rt * sin(a), zr), ((rt + ln * 0.6) * cos(a), (rt + ln * 0.6) * sin(a), zr + ln * 0.35),
                                  ((rt + ln) * cos(a + 0.1), (rt + ln) * sin(a + 0.1), zr + ln * 0.8)], 0.011, sides=5,
                   smooth_path=False, scale_fn=lambda u: 1.0 - 0.45 * u)
        uv_transform(st, rot_deg=90)
        assign(st, m_bark)
        woody.append(st)
    north = facing((0, 1, 0))
    for o in woody:
        paint(o, grime=height_ramp(0.0, 0.5),
              special=mask_mul(height_ramp(0.0, zb + 0.2), lambda co, n: 0.35 + 0.65 * north(co, n), noise_mask(0.12, seed=p.seed + 3, lo=0.3, hi=0.7)))

    # ------------------------------------------------------------------ seed cones in small clusters
    r_c = rng.fork("cones")
    ncl = int(round(p.cones * 30 * sqrt(sum(areas) / 15.0)))
    if ncl > 0 and cone_pts:
        cone = sphere("cone", 0.5, segments=5, rings=3)
        Ms = []
        for _ in range(ncl):
            pos, out = r_c.pick(cone_pts)                # sits on a spray's stem, the cluster touching
            for j in range(r_c.pick([1, 2, 2, 3])):
                s = r_c.u(0.026, 0.036)
                q = pos + Vector(r_c.vec(0.016)) * (1.0 if j else 0.0)
                Ms.append(trs(q, (r_c.u(0, 360), r_c.u(0, 360), r_c.u(0, 360)), (s, s, s * r_c.u(0.85, 1.1))))
        if pit != "none":                    # a few shed cones lying in the tree pit
            for _ in range(r_c.pick([4, 6, 8])):
                a, r = r_c.u(0, 2 * pi), r_c.u(rb * 2.0, rp * 0.9)
                s = r_c.u(0.026, 0.036)
                Ms.append(trs((r * cos(a), r * sin(a), lerp(0.05, 0.022, r / rp) + s * 0.3), (r_c.u(0, 360), r_c.u(0, 360), 0), s))
        cones = instances_to_mesh("cones", cone, Ms, keep_base=False)
        smooth(cones, 80)
        assign(cones, m_dead)

    # ------------------------------------------------------------------ tree pit: soil mound, litter, sett ring
    if pit != "none":
        r_p = rng.fork("pit")
        edge = rp + (0.03 if pit == "kerb" else 0.0)
        prof = [(0.0, 0.05), (rb * 1.5, 0.05), (edge * 0.45, 0.04), (edge * 0.75, 0.03), (edge * 0.92, 0.019 if pit == "kerb" else 0.011),
                (edge, 0.012 if pit == "kerb" else 0.0)]
        soil = lathe("soil", prof, segments=30, smooth_angle=80)
        subdivide(soil, 1)
        ragged = [(m, r_p.u(0, 2 * pi), amp) for m, amp in ((2, 0.05), (3, 0.05), (7, 0.03), (11, 0.02))] if pit == "soil" else []

        def warp_soil(co):
            r = hypot(co.x, co.y)
            a = atan2(co.y, co.x)
            k = 1.0 + sum(amp * sin(m * a + ph) for m, ph, amp in ragged) * (r / edge) ** 2
            z = co.z + 0.007 * sin(9 * co.x + 3.1) * sin(8 * co.y + 1.7) * min(1.0, (edge - r) / 0.1) if r < edge else co.z
            return Vector((co.x * k, co.y * k, max(z, 0.0)))

        transform_verts(soil, warp_soil)
        displace_noise(soil, strength=0.005, scale=0.05, seed=p.seed + 5, axis_mask=(0, 0, 1))
        transform_verts(soil, lambda co: Vector((co.x, co.y, max(co.z, 0.0))))
        assign(soil, mat("soil"))
        uv_box(soil)
        paint(soil, special=mask_mul(noise_mask(0.25, seed=p.seed + 8), 0.6))
        # shed brown sprays lying on the soil
        litter = _Sprays(sprays, rng.fork("litter"))
        up = Vector((0.0, 0.0, 1.0))
        for _ in range(r_p.pick([7, 9, 12])):
            a, r = r_p.u(0, 2 * pi), r_p.u(rb * 2.2, rp * 0.9)
            L = L0 * r_p.u(0.5, 0.8)
            b_ = r_p.u(0, 2 * pi)
            d = Vector((cos(b_), sin(b_), r_p.u(-0.04, 0.06))).normalized()
            x, y = _frame(d, up, r_p.u(-0.15, 0.15))
            litter.add(Vector((r * cos(a), r * sin(a), 0.05 - 0.03 * r / rp + 0.012)), x * L, y * (L * 0.35), d * L, 0.3, 0.2)
        lit = litter.bake("litter", m_dead)
        transform_verts(lit, lambda co: Vector((co.x, co.y, max(co.z, 0.006))))
        if pit == "kerb":
            st = mat("stone_rough", "#a39a8a")
            rk = rp + 0.065
            nk = max(8, int(round(2 * pi * rk / 0.19)))
            a_k = r_p.u(0, 2 * pi)
            for k in range(nk):
                a = a_k + 2 * pi * (k + r_p.u(-0.06, 0.06)) / nk
                hk = r_p.u(0.066, 0.084)
                lk = 2 * pi * rk / nk - r_p.u(0.008, 0.02)
                dk = r_p.u(0.105, 0.125)
                rr = rk + r_p.u(-0.008, 0.008)
                b = box("sett%d" % k, (dk, lk, hk), loc=(rr * cos(a), rr * sin(a), hk / 2 + 0.006),
                        rot=(r_p.u(-2, 2), r_p.u(-2.5, 2.5), np.degrees(a) + r_p.u(-3, 3)), bevel=r_p.u(0.009, 0.016), segments=2)
                displace_noise(b, strength=0.005, scale=0.07, seed=p.seed + 20 + k)
                smooth(b, 50)
                assign(b, st)
                paint(b, grime=height_ramp(0.0, 0.07), special=mask_mul(lambda co, n: 0.3 + 0.7 * north(co, n), noise_mask(0.15, seed=p.seed + 9, lo=0.35, hi=0.7)))
