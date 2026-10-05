"""Canal quay wall segment: a run of bull-nosed coping stones at pavement level (top exactly z = 0, cramped
together with let-in iron dogs) on a wall of individually modelled coursed blocks - fine ashlar, rock-faced
ashlar with drafted margins, or snecked squared rubble - bedded in recessed lime mortar. The wall drops to the
water with a projecting footing course and a graded tide zone (damp stone, a green algae line at the mean
level, dark slime below, limescale runs), iron mooring rings on ragbolts, forged quay lanterns and one
feature: a wrought-iron ladder with wall stays, a flight of one-piece water stairs cut into the wall, or an
arched culvert outfall (voussoirs, keystone, grate) set in the coursed masonry. Face in the plane y = 0,
water towards -Y, the run spans exactly x = -length/2 .. +length/2 and ends on half joints so runs butt.
"""
from tk import *
import numpy as np
from math import radians, degrees, cos, sin, pi, sqrt, exp, ceil, atan2, hypot
from types import SimpleNamespace as NS
from mathutils import Vector, Matrix, noise as mnoise

META = {"name": "Canal quay wall segment", "class": "structure", "mount": "ground",
        "allow_below_ground": True, "focus": 0.6,
        # the exporter's automatic cavity / ground-splash grime treats everything below z = 0 as buried and
        # blotches the whole wall black, so the grime story is painted by hand instead
        "auto_grime": 0.0, "lod": "far"}
PARAMS = {
    "length": {"default": 8.0, "min": 2.0, "max": 16.0, "desc": "run length along X (m); spans exactly +-length/2"},
    "depth": {"default": 2.0, "min": 1.0, "max": 3.0, "desc": "drop from the coping top (z = 0) to the wall foot (m)"},
    "feature": {"default": "steps", "choices": ["none", "ladder", "steps", "outfall"],
                "desc": "iron ladder, recessed water stairs (falls back to a ladder when the run is too short) or culvert outfall"},
    "feature_pos": {"default": -0.35, "min": -1.0, "max": 1.0, "desc": "where the feature sits along the run (-1 left .. 1 right)"},
    "rings": {"default": 2, "min": 0, "max": 8, "desc": "number of iron mooring rings on the face"},
    "moss": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "amount of algae / moss at the tide line and in the joints"},
    "masonry": {"default": "rockfaced", "choices": ["ashlar", "rockfaced", "rubble"], "desc": "wall masonry"},
    "course": {"default": 0.32, "min": 0.22, "max": 0.45, "desc": "nominal course height (m)"},
    "waterline": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "water surface height above the wall foot (m): tide band, landing, outfall invert"},
    "outfall_size": {"default": 0.75, "min": 0.4, "max": 1.3, "desc": "clear width of the culvert outfall (m)"},
    "stone": {"default": "#a69c88", "desc": "wall stone colour"},
    "coping_stone": {"default": "#b9b09e", "desc": "colour of the coping / dressed stone"},
    "lantern": {"default": True, "desc": "forged quay lanterns on the wall face (the first beside the feature)"},
    "lantern_every": {"default": 4.5, "min": 3.0, "max": 16.0, "desc": "one lantern per this many metres of run"},
    "intensity": {"default": 4.0, "min": 1.0, "max": 40.0, "desc": "lantern intensity (cd)"},
}
VARIANTS = {
    "ashlar_ladder": {"feature": "ladder", "length": 10.0, "depth": 2.4, "masonry": "ashlar", "rings": 3, "moss": 0.35,
                      "feature_pos": -0.2, "stone": "#b3a68c", "seed": 4},
    "outfall_rubble": {"feature": "outfall", "length": 7.0, "depth": 2.6, "masonry": "rubble", "rings": 2, "moss": 0.9,
                       "stone": "#8f8a7c", "outfall_size": 0.95, "feature_pos": -0.25, "seed": 9},
    "low_plain": {"feature": "none", "length": 5.0, "depth": 1.2, "masonry": "ashlar", "course": 0.26, "rings": 4,
                  "moss": 0.15, "waterline": 0.2, "stone": "#b8ad9a", "feature_pos": 0.3, "seed": 13},
    "deep_long": {"feature": "ladder", "length": 14.0, "depth": 3.0, "course": 0.4, "rings": 6, "moss": 0.7,
                  "waterline": 0.6, "feature_pos": 0.4, "stone": "#978d7c", "seed": 21},
}

# jw = mortar joint between block sides, ch = arris chamfer (visible joint = jw + 2 ch), rec = mortar set back
# from the arris, margin = drafted margin, step = run of the pitched edge up to the rock face, bulge = relief
STYLES = {
    "ashlar": NS(jw=0.006, ch=0.002, rec=0.006, margin=0.0, step=0.03, bulge=0.0016, cell=0.2, tilt=0.002, offj=0.0018,
                 jit=0.0, fine=0.0008, lmin=1.7, lmax=2.9, cscale=1.0, spall=0.06, spd=0.005, sm=40, lib="stone_smooth"),
    "rockfaced": NS(jw=0.009, ch=0.003, rec=0.008, margin=0.026, step=0.015, bulge=0.036, cell=0.085, tilt=0.003, offj=0.002,
                    jit=0.0, fine=0.004, lmin=1.5, lmax=2.5, cscale=1.0, spall=0.07, spd=0.002, sm=34, lib="stone_rough"),
    "rubble": NS(jw=0.015, ch=0.004, rec=0.005, margin=0.0, step=0.02, bulge=0.036, cell=0.075, tilt=0.012, offj=0.008,
                 jit=0.008, fine=0.005, lmin=0.9, lmax=1.9, cscale=1.08, spall=0.0, spd=0.0, sm=34, lib="stone_rough"),
}
ZV = Vector((0, 0, 1))
COP_TH = 0.17          # coping thickness
COP_Y0, COP_Y1 = -0.055, 0.56
BODY_Y = 0.5           # back of the wall body
STAIR_Y = 0.42         # back wall of the stair recess


def shade(hexc, k, g=(1.0, 1.0, 1.0)):
    h = hexc.lstrip("#")
    c = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(clamp(c[i] * k * g[i] / 255.0) * 255) for i in range(3))


def course_lines(top, bottom, c):
    """Course joint heights from the top down; independent of the seed so neighbouring runs line up."""
    pat = [1.0, 1.13, 0.9, 1.06, 1.2, 0.94, 1.1, 0.87]
    zs, i = [top], 0
    while zs[-1] - c * pat[i % 8] > bottom + 0.6 * c:
        zs.append(zs[-1] - c * pat[i % 8])
        i += 1
    zs.append(bottom)
    return zs


def split_run(xa, xb, lmin, lmax, r):
    xs, x, first = [xa], xa, True
    while True:
        l = r.u(lmin, lmax) * (r.u(0.45, 1.0) if first else 1.0)
        first = False
        if xb - (x + l) < lmin * 0.6:
            break
        x += l
        xs.append(x)
    xs.append(xb)
    return xs


class Blocks:
    """Collects individually shaped block faces. Each block is a shell: a side dropping into the joint behind
    the mortar plane, a chamfered arris, an optional drafted margin, a pitched edge and the (rock) face."""

    def __init__(self, st, rng, tide, damp_top):
        self.st, self.r, self.tide, self.damp_top = st, rng, tide, damp_top
        self.groups = {k: NS(V=[], F=[], G=[], W=[], K=[], UV=[]) for k in ("a", "w")}
        self.beds = []          # (O, U, N, u0, u1, v) bed joints, for the moss
        self.extra = {}
        self.far = lod() == 2   # far level (META "lod": "far"): every stone is one face at its mean relief ...
        self.skip = False       # ... and a course wholly under water one face per run (run())

    def add(self, O, U, N, u0, u1, v0, v1, flush=(False, False), proud=0.0, xl=None, xr=None):
        st, r = self.st, self.r
        j = st.jw * 0.5
        a0 = u0 + (0.0 if flush[0] else j)
        a1 = u1 - (0.0 if flush[1] else j)
        b0, b1 = v0 + j, v1 - j
        w, h = a1 - a0, b1 - b0
        if w < 0.03 or h < 0.03:
            return
        under = v1 < self.tide - 0.02                       # wholly under water: coarser
        zc_ = 0.5 * (v0 + v1)                               # "w" = the darker, damper stone
        g = self.groups["w" if (zc_ < self.damp_top or (zc_ < self.damp_top + 0.4 and r.chance(0.4)) or r.chance(0.14)) else "a"]
        curved = xl is not None or xr is not None
        if curved:
            def M(s, t):
                v = lerp(b0, b1, t)
                return (lerp(xl(v) + j if xl else a0, xr(v) - j if xr else a1, s), v)
        else:
            jt = st.jit
            jl, jr = (0.0 if flush[0] else 1.0), (0.0 if flush[1] else 1.0)
            c00 = (a0 + jl * r.u(0, jt), b0 + r.u(0, jt))
            c10 = (a1 - jr * r.u(0, jt), b0 + r.u(0, jt))
            c01 = (a0 + jl * r.u(0, jt), b1 - r.u(0, jt))
            c11 = (a1 - jr * r.u(0, jt), b1 - r.u(0, jt))

            def M(s, t):
                return (lerp(lerp(c00[0], c10[0], s), lerp(c01[0], c11[0], s), t),
                        lerp(lerp(c00[1], c10[1], s), lerp(c01[1], c11[1], s), t))

        lim = min(w, h)
        ch, mg, sp = min(st.ch, 0.06 * lim), min(st.margin, 0.16 * lim), min(st.step, 0.14 * lim)
        cl_, cr_ = (0.0 if flush[0] else ch), (0.0 if flush[1] else ch)
        gl, gr, gb = cl_ + mg + sp, cr_ + mg + sp, ch + mg + sp
        wg, hg = w - gl - gr, h - 2 * gb
        cell = st.cell * (1.7 if under else 1.0) * max(1.0, h / 0.34)
        nui = int(clamp(round(wg / cell), 1, 8))
        nvi = int(clamp(round(hg / cell), 1, 4))
        if curved:
            nvi = int(clamp(ceil(h / 0.07), nvi, 7))
        FB = [0.0, 1.0]
        FR = [k / nvi for k in range(nvi + 1)] if xr else FB
        FL = [k / nvi for k in range(nvi + 1)] if xl else FB

        any_flush = flush[0] or flush[1]
        amp = st.bulge * r.u(0.7, 1.15)
        tu = 0.0 if any_flush else r.jitter(0.0, st.tilt)
        tv = 0.0 if any_flush else r.jitter(0.0, st.tilt)
        off = proud + (0.0 if any_flush else r.jitter(0.0, st.offj))
        ox, oy = r.u(0, 50), r.u(0, 50)
        t_ = r.u()
        if t_ < 0.07:
            gj, wj = -0.3, r.u(0.3, 0.45)                   # replaced stone: newer, clean and paler
        elif t_ < 0.17:
            gj, wj = r.u(0.28, 0.4), 0.0                    # a stone that holds the dirt
        elif t_ < 0.45:
            gj, wj = r.u(0.14, 0.24), 0.0
        else:
            gj, wj = r.u(0.0, 0.1), r.pick((0.0, 0.0, 0.1, 0.2))
        spall = None
        if st.spall > 0 and not under and r.chance(st.spall):
            cs, ct = r.pick((0.0, 1.0)), r.pick((0.0, 1.0))
            spall = (lerp(a0, a1, cs if r.chance(0.6) else r.u()), lerp(b0, b1, ct), r.u(0.08, 0.15))
        rot, uo, vo = int(r.u(0, 4)) % 4, r.u(0, 7), r.u(0, 7)
        V, F = g.V, g.F

        def emit(s, t, d, k=0.0, wx=0.0):
            u, v = M(s, t)
            V.append(O + U * u + ZV * v + N * d)
            g.G.append(gj)
            g.W.append(wj + wx)
            g.K.append(k)
            pu, pv = u - a0, v - b0
            if rot == 1:
                pu, pv = -pv, pu
            elif rot == 2:
                pu, pv = -pu, -pv
            elif rot == 3:
                pu, pv = pv, -pu
            g.UV.append((pu + uo, pv + vo))
            return len(V) - 1

        def base(s, t):
            return off + tu * (s - 0.5) + tv * (t - 0.5)

        def loop(il, ir, ib, flat):
            sl, sr, tb, tt = il / w, 1 - ir / w, ib / h, 1 - ib / h
            pts = [(lerp(sl, sr, f), tb) for f in FB[:-1]]
            pts += [(sr, lerp(tb, tt, f)) for f in FR[:-1]]
            pts += [(lerp(sr, sl, f), tt) for f in FB[:-1]]
            pts += [(sl, lerp(tt, tb, f)) for f in FL[:-1]]
            if flat is None:
                return [emit(s, t, -(st.rec + 0.003), 1.0) for (s, t) in pts]
            return [emit(s, t, base(s, t) + flat) for (s, t) in pts]

        if self.far:                                        # the random stream has drawn what lod 0 draws
            if not self.skip:
                d = off + 0.55 * amp
                F.append((emit(0, 0, d), emit(1, 0, d), emit(1, 1, d), emit(0, 1, d)))
            self.beds.append((O, U, N, u0, u1, v0))
            return
        rings = [loop(0.0, 0.0, 0.0, None), loop(cl_, cr_, ch, 0.0)]
        if mg > 0:
            rings.append(loop(cl_ + mg, cr_ + mg, ch + mg, 0.0006))
        n = len(rings[0])
        for a, b in zip(rings[:-1], rings[1:]):
            for q in range(n):
                q2 = (q + 1) % n
                F.append((a[q], a[q2], b[q2], b[q]))
        # the face grid
        gi = {}
        for k in range(nvi + 1):
            for i in range(nui + 1):
                fs, ft = i / nui, k / nvi
                s, t = lerp(gl / w, 1 - gr / w, fs), lerp(gb / h, 1 - gb / h, ft)
                u, v = M(s, t)
                de = min(fs * wg, (1 - fs) * wg, ft * hg, (1 - ft) * hg)
                m = smoothstep(0.0, 0.08, de)
                n1 = mnoise.noise(Vector((u * 4.0 + ox, v * 4.0 + oy, 0.3)))
                n2 = mnoise.noise(Vector((u * 13.0 + oy, v * 13.0 + ox, 1.7)))
                n3 = 1.0 - 2.0 * abs(mnoise.noise(Vector((u * 7.0 + ox, v * 7.0 - oy, 3.1))))
                rel = max(amp * (0.62 + 0.38 * m) * (0.8 + 0.45 * n1 + 0.25 * n3), 0.3 * amp) + st.fine * n2
                wx = 0.0
                if spall:
                    ds = hypot(u - spall[0], v - spall[1])
                    if ds < spall[2]:
                        f = smoothstep(0.25 * spall[2], spall[2], ds)
                        rel = rel * f - st.spd * (1 - f)
                        wx = 0.4 * (1 - f)
                gi[(i, k)] = emit(s, t, base(s, t) + rel, 0.0, wx)
        for k in range(nvi):
            for i in range(nui):
                F.append((gi[(i, k)], gi[(i + 1, k)], gi[(i + 1, k + 1)], gi[(i, k + 1)]))
        # pitched edge: last ring -> boundary of the grid (quads where the side is subdivided, else a fan)
        last = rings[-1]
        nsr, nsl = len(FR) - 1, len(FL) - 1
        sides = [([last[0], last[1]], [gi[(i, 0)] for i in range(nui + 1)]),
                 ([last[1 + q] for q in range(nsr + 1)], [gi[(nui, k)] for k in range(nvi + 1)]),
                 ([last[1 + nsr], last[2 + nsr]], [gi[(i, nvi)] for i in range(nui, -1, -1)]),
                 ([last[(2 + nsr + q) % n] for q in range(nsl + 1)], [gi[(0, k)] for k in range(nvi, -1, -1)])]
        for rs, gs in sides:
            ng = len(gs) - 1
            if len(rs) == len(gs):
                for q in range(ng):
                    F.append((rs[q], rs[q + 1], gs[q + 1], gs[q]))
            else:
                a, b, hm = rs[0], rs[-1], ng // 2
                for q in range(ng):
                    F.append((a if q < hm else b, gs[q + 1], gs[q]))
                F.append((a, b, gs[hm]))
        self.beds.append((O, U, N, u0, u1, v0))

    def side(self, x, sg, v0, v1, key):
        """Far level: the ends of the courses at an end of the run as one face (what shows from along the wall)."""
        g = self.groups[key]
        b = len(g.V)
        pts = [(self.st.rec + 0.003, v0), (-0.5 * self.st.bulge, v0), (-0.5 * self.st.bulge, v1), (self.st.rec + 0.003, v1)]
        for (y, v) in (pts if sg < 0 else pts[::-1]):
            g.V.append(Vector((x, y, v)))
            g.G.append(0.3)
            g.W.append(0.0)
            g.K.append(0.0)
            g.UV.append((y, v))
        g.F.append((b, b + 1, b + 2, b + 3))

    def run(self, O, U, N, u0, u1, v0, v1, proud):
        """Far level: the damp stones of one course under water as a single face."""
        g = self.groups["w"]
        d, b = proud + 0.5 * self.st.bulge, len(g.V)
        for (u, v) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            g.V.append(O + U * u + ZV * v + N * d)
            g.G.append(0.1)
            g.W.append(0.0)
            g.K.append(0.0)
            g.UV.append((u, v))
        g.F.append((b, b + 1, b + 2, b + 3))

    def rubble_run(self, O, U, N, z0, z1, xa, xb, fl, fr, cl, cr, proud):
        """Snecked squared rubble: risers, levellers and small snecks making up one course band."""
        r, H = self.r, z1 - z0
        units, x = [], xa
        while xb - x > 1e-6:
            typ = (r.weighted([("riser", 3), ("stack2", 4), ("sneck", 3.5), ("stack3", 1.5)]) if H > 0.24
                   else r.weighted([("riser", 5), ("stack2", 2)]))
            l = {"riser": r.u(0.2, 0.42), "stack2": r.u(0.32, 0.62), "sneck": r.u(0.45, 0.78), "stack3": r.u(0.28, 0.45)}[typ]
            if xb - x - l < 0.18:
                l = xb - x
                if l > 0.5 and typ in ("riser", "stack3"):
                    typ = "stack2"
                if l < 0.3 and typ == "sneck":
                    typ = "riser"
            units.append((x, x + l, typ))
            x += l
        for q, (x0, x1, typ) in enumerate(units):
            f0, f1 = fl and q == 0, fr and q == len(units) - 1
            c0, c1 = (cl if q == 0 else None), (cr if q == len(units) - 1 else None)

            def A(xa_, xb_, za, zb):
                e0, e1 = xa_ == x0, xb_ == x1
                self.add(O, U, N, xa_, xb_, za, zb, (f0 and e0, f1 and e1), proud, xl=c0 if e0 else None, xr=c1 if e1 else None)

            l = x1 - x0
            if typ == "riser":
                A(x0, x1, z0, z1)
            elif typ == "stack2":
                zm = lerp(z0, z1, r.u(0.36, 0.64))
                lo_split = r.chance(0.5)
                for (za, zb, sp) in ((z0, zm, lo_split), (zm, z1, not lo_split)):
                    if sp and l > 0.42 and r.chance(0.65):
                        xm = lerp(x0, x1, r.u(0.35, 0.65))
                        A(x0, xm, za, zb)
                        A(xm, x1, za, zb)
                    else:
                        A(x0, x1, za, zb)
            elif typ == "stack3":
                za, zb = lerp(z0, z1, r.u(0.27, 0.36)), lerp(z0, z1, r.u(0.63, 0.73))
                A(x0, x1, z0, za)
                A(x0, x1, za, zb)
                A(x0, x1, zb, z1)
            else:
                hL = H * r.u(0.3, 0.4)
                s = r.u(0.11, 0.16)
                if r.chance(0.5):
                    A(x0, x1, z1 - hL, z1)
                    zr0, zr1 = z0, z1 - hL
                else:
                    A(x0, x1, z0, z0 + hL)
                    zr0, zr1 = z0 + hL, z1
                if r.chance(0.5):
                    xs_, sn, big = x0 + s, (x0, x0 + s), (x0 + s, x1)
                else:
                    xs_, sn, big = x1 - s, (x1 - s, x1), (x0, x1 - s)
                A(big[0], big[1], zr0, zr1)
                if zr1 - zr0 > 0.19:
                    zm = lerp(zr0, zr1, r.u(0.42, 0.58))
                    A(sn[0], sn[1], zr0, zm)
                    A(sn[0], sn[1], zm, zr1)
                else:
                    A(sn[0], sn[1], zr0, zr1)

    def course(self, O, U, N, z0, z1, intervals, proud=0.0):
        st, r = self.st, self.r
        h = z1 - z0
        self.skip = self.far and z1 < self.tide - 0.02
        for iv in intervals:
            xa, xb, fl, fr = iv[:4]
            cl, cr = (iv[4], iv[5]) if len(iv) > 4 else (None, None)
            if xb - xa < 0.05:
                continue
            if self.skip:
                self.run(O, U, N, xa, xb, z0, z1, proud)
            if st is STYLES["rubble"] and z1 > self.tide - 0.2:
                self.rubble_run(O, U, N, z0, z1, xa, xb, fl, fr, cl, cr, proud)
                continue
            xs = split_run(xa, xb, clamp(st.lmin * h, 0.2, 1.0), clamp(st.lmax * h, 0.4, 1.5), r)
            for q in range(len(xs) - 1):
                first, lastq = q == 0, q == len(xs) - 2
                self.add(O, U, N, xs[q], xs[q + 1], z0, z1, (fl and first, fr and lastq), proud,
                         xl=cl if first else None, xr=cr if lastq else None)
        self.skip = False

    def realize(self, mats):
        out = []
        for key, g in self.groups.items():
            if not g.F:
                continue
            o = mesh_object("wall_" + key, g.V, g.F)
            me = o.data
            uvl = me.uv_layers.new(name="UVMap")            # per-block texture offset / rotation
            li = np.empty(len(me.loops), dtype=np.int32)
            me.loops.foreach_get("vertex_index", li)
            uvl.data.foreach_set("uv", np.asarray(g.UV, dtype=np.float32)[li].ravel())
            uv_keep(o)
            assign(o, mats[key])
            smooth(o, self.st.sm)
            self.extra[o.name] = (np.array(g.G), np.array(g.W), np.array(g.K))
            out.append(o)
        return out


def nut(name, s, h, loc, axis="y"):
    """Square forged bolt head / nut as a plain little block (12 tris)."""
    return box(name, (s, h, s) if axis == "y" else (s, s, h), loc=loc, rot=(0, 17, 0) if axis == "y" else (0, 0, 17))


def link_chain(name, path, link_len=0.11, wire=0.009):
    """Light chain of oval links along a path lying in a vertical plane y = const."""
    P = [Vector(q) for q in path]
    step, out, acc = link_len * 0.72, [], 0.0
    for a, b in zip(P[:-1], P[1:]):
        l = (b - a).length
        while acc <= l:
            out.append((a.lerp(b, acc / l), (b - a).normalized()))
            acc += step
        acc -= l
    hl_, hw_ = link_len / 2 - wire, link_len * 0.3 - wire
    pts = [(hl_ * 0.55, hw_, 0), (hl_, hw_ * 0.55, 0), (hl_, -hw_ * 0.55, 0), (hl_ * 0.55, -hw_, 0),
           (-hl_ * 0.55, -hw_, 0), (-hl_, -hw_ * 0.55, 0), (-hl_, hw_ * 0.55, 0), (-hl_ * 0.55, hw_, 0)]
    base = sweep(name + "_link", pts, wire, closed=True, sides=3, smooth_path=False)
    ms = []
    for i, (c, t) in enumerate(out):
        b, nn = Vector((0, 1, 0)), Vector((-t.z, 0, t.x))
        if i % 2:
            b, nn = nn, -b
        ms.append(Matrix(((t.x, b.x, nn.x, c.x), (t.y, b.y, nn.y, c.y), (t.z, b.z, nn.z, c.z), (0, 0, 0, 1))))
    o = instances_to_mesh(name, base, ms, keep_base=False)
    smooth(o, 180)
    return o


def coping_stone(name, x0, x1, y0, y1, th, r, nose=0.035, dish=(None, None), chips=True):
    """Bull-nosed coping stone running along X, top exactly at z = 0 (worn hollows and chips only go down).
    dish = profile index of the top point that is let down at each end for an iron cramp."""
    ny = max(2, int(round((y1 - y0) / 0.15)))
    na = 4
    prof = [(y1 - (y1 - y0 - nose) * i / ny, 0.0) for i in range(ny)]
    for k in range(na + 1):
        a = radians(90 + 90 * k / na)
        prof.append((y0 + nose + nose * cos(a), -nose + nose * sin(a)))
    prof += [(y0, -th + 0.02), (y0 + 0.02, -th), (y1 - 0.01, -th), (y1, -th + 0.01)]
    npf = len(prof)
    ln = x1 - x0
    nxs = max(1, int(round(ln / 0.5)))
    xst = [x0 + ln * s / nxs for s in range(nxs + 1)]
    chip_at = {}
    if ln > 0.3:                                       # extra stations by the ends keep the joint dirt narrow
        xst = [x0, x0 + 0.035] + xst[1:-1] + [x1 - 0.035, x1]
        if chips:
            for e, xe_, xi in ((0, x0, x0 + 0.035), (1, x1, x1 - 0.035)):
                if r.chance(0.35):                     # knocked corner at the joint
                    d = r.u(0.008, 0.02)
                    chip_at[xe_], chip_at[xi] = d, 0.3 * d
            if ln > 0.6 and r.chance(0.5):             # a chip out of the nosing
                xc = r.u(x0 + 0.15, x1 - 0.15)
                hw = r.u(0.025, 0.045)
                xst = sorted([x for x in xst if abs(x - xc) > hw + 0.03] + [xc - hw, xc, xc + hw])
                chip_at[xc] = r.u(0.008, 0.018)
    nx = len(xst) - 1
    wk = (0.45, 1.0, 0.85, 0.45, 0.1)
    wz = (1.0, 0.6, 0.15, 0.0, 0.0)
    V, F = [], []
    for x in xst:
        d = chip_at.get(x, 0.0)
        for q, (y, z) in enumerate(prof):
            if d and ny <= q <= ny + na:
                y, z = y + d * wk[q - ny], z - 0.6 * d * wz[q - ny]
            if dish[0] is not None and q == dish[0] and x - x0 < 0.04:
                z -= 0.0045
            if dish[1] is not None and q == dish[1] and x1 - x < 0.04:
                z -= 0.0045
            V.append((x, y, z))
    for s in range(nx):
        for q in range(npf):
            a, b = s * npf + q, s * npf + (q + 1) % npf
            F.append((a, b, b + npf, a + npf))
    F.append(tuple(range(npf)))
    F.append(tuple(reversed(range(nx * npf, (nx + 1) * npf))))
    o = mesh_object(name, V, F)
    recalc_normals(o)
    bevel(o, 0.006, 1, angle=50)
    displace_noise(o, strength=0.002, scale=0.4, seed=int(r.u(0, 1000)), axis_mask=(1, 1, 0))
    transform_verts(o, lambda co: Vector((co.x, co.y, min(co.z, 0.0))))
    place(o, loc=(0, r.jitter(0, 0.003), 0))
    smooth(o, 40)
    return o


def coping_far(name, x0, x1, y0, y1, th, nose, caps, wall=None):
    """Far level: a run of coping as one prism - top, chamfered nosing, face, soffit and back; caps = which ends
    are closed. wall = where the wall face meets the soffit: the soffit in front of it then falls a little towards
    the wall. A level soffit faces no camera and the town bake drops it; the short face above it is then a strip
    with a free lower edge, and with open ends too the lowest level simplifies it away."""
    pr = [(y1, 0.0), (y0 + nose, 0.0), (y0, -nose), (y0, -th), (y1, -th)]
    if wall is not None:
        pr.insert(4, (wall, -th - 0.35 * (wall - y0)))
    n = len(pr)
    V = [(x, y, z) for x in (x0, x1) for (y, z) in pr]
    F = [(q, (q + 1) % n, (q + 1) % n + n, q + n) for q in range(n)]
    if caps[0]:
        F.append(tuple(range(n - 1, -1, -1)))
    if caps[1]:
        F.append(tuple(range(n, 2 * n)))
    return mesh_object(name, V, F)


def step_far(name, xa, xb, zt, rise, zbot, y0, y1, riser, jw):
    """Far level: a step as its tread, its riser and its dressed end in the wall face, one face each."""
    j = jw / 2
    x0, xr = xa + j, xb - j
    xe = xb + 0.011 if riser else xr
    zb = zbot + j
    V = [(x0, y0, zt), (xe, y0, zt), (xe, y1, zt), (x0, y1, zt),
         (x0, y0, zb), (xr, y0, zb), (xr, y0, zt), (x0, y0, zt)]
    F = [(0, 1, 2, 3), (4, 5, 6, 7)]
    if riser:                                           # the tread's front edge is cut where the dressed end stops,
        zf = zt - rise - 0.016                          # so tread, end and riser share whole edges
        V += [(xe, y0, zf), (xe, y1, zf), (xe, y1, zt), (xe, y0, zt)]
        F = [(3, 0, 6), (3, 6, 1), (3, 1, 2), (4, 5, 6, 7), (8, 9, 10, 11)]
    return mesh_object(name, V, F)


def step_stone(name, xa, xb, zt, rise, zbot, y0, y1, ysk, seed, riser=True, jw=0.008, nose=0.022, rn=0.026, wear=0.011):
    """One-piece stone step running back into the wall (along Y): tread hollowed where feet land, worn rounded
    nosing, raked riser. Its dressed end shows in the wall face at y0 as a block reaching down to the bed joint
    `zbot`, so the flight is bonded into the coursed masonry."""
    j = jw / 2
    x0, xr = xa + j, xb - j
    xe = xb + nose if riser else xr
    xt = xe - rn if riser else xe
    zf = zt - rise - 0.016
    nt = max(2, int(round((xe - x0) / 0.11)))
    sec = [(lerp(x0, xt, i / nt), zt, 0.35 + 0.65 * i / nt, 0.0) for i in range(nt + 1)]
    if riser:
        for k in range(1, 5):
            a = radians(90 - 90 * k / 4)
            sec.append((xe - rn + rn * cos(a), zt - rn + rn * sin(a), 1.0 - 0.22 * k, 1.0))
        sec.append((xe - 0.005, zt - 0.55 * rise, 0.0, 0.3))
        sec.append((xe - 0.014, zf, 0.0, 0.0))
    ns = len(sec)
    ny = max(3, int(round((y1 - y0) / 0.08)))
    ym = y0 + 0.46 * (y1 - y0)
    V, F = [], []
    for k in range(ny + 1):
        y = lerp(y0, y1, k / ny)
        b = exp(-((y - ym) / 0.13) ** 2) if k else 0.0
        V += [(x - 0.009 * b * wr, y, z - wear * b * wd) for (x, z, wd, wr) in sec]
    for k in range(ny):
        for i in range(ns - 1):
            F.append((k * ns + i, k * ns + i + 1, (k + 1) * ns + i + 1, (k + 1) * ns + i))
    # the end in the wall face: step end and the block under it are one stone
    zb = zbot + j
    nl = max(1, int(round((zt - zb) / 0.12)))
    side = [(x0, lerp(zt, zb, k / nl)) for k in range(1, nl + 1)]
    ztr = zf if riser else zt
    nr = max(1, int(round((ztr - zb) / 0.12)))
    side += [(xr, lerp(zb, ztr, k / nr)) for k in range(0, nr + (1 if riser else 0))]
    base = len(V)
    V += [(x, y0, z) for (x, z) in side]
    ring = [0] + list(range(base, base + len(side))) + ([] if riser else [ns - 1])
    F.append(tuple(reversed(range(ns))) + tuple(range(base, base + len(side))))
    b2 = len(V)
    V += [(V[q][0], ysk, V[q][2]) for q in ring]
    for q in range(len(ring) - 1):
        F.append((ring[q], b2 + q, b2 + q + 1, ring[q + 1]))
    o = mesh_object(name, V, F)
    bevel(o, 0.006, 1, angle=60)
    displace_noise(o, strength=0.0015, scale=0.2, seed=seed)
    smooth(o, 50)
    return o


def quay_lantern(name, x, y0, ztop, yl, iron, intensity, far=False):
    """Forged gallows bracket (short reach, the whole fitting stays within 0.3 m of the face) with a hanging
    four-pane oil lantern about 0.21 m square and 0.33 m tall; wall plate at y0, lantern axis at yl."""
    za = ztop - 0.06
    reach = y0 - yl
    if far:                                             # wall plate, arm, rod, a plain pyramid roof, floor and corner
        yp = y0 - 0.008                                 # bars; the four panes as they are (they are what is lit)
        iron.append(mesh_object("lan_plate", [(x - 0.03, yp, za - 0.31), (x + 0.03, yp, za - 0.31), (x + 0.03, yp, za + 0.07),
                                              (x - 0.03, yp, za + 0.07)], [(0, 1, 2, 3)]))
        iron.append(sweep("lan_arm", [(x, y0, za), (x, yl - 0.02, za)], 0.012, sides=3, smooth_path=False))
        a, b = 0.092, 0.062
        z1 = za - 0.185
        z0 = z1 - 0.2
        iron.append(lathe("lan_roof", [(0.0, 0.0), (0.146, 0.0), (0.0, 0.107)], segments=4, loc=(x, yl, z1), rot=(0, 0, 45), smooth_angle=20))
        iron.append(sweep("lan_rod", [(x, yl, z1 + 0.1), (x, yl, za)], 0.006, sides=3, smooth_path=False))
        iron.append(box("lan_base", (2 * b + 0.018, 2 * b + 0.018, 0.014), loc=(x, yl, z0 - 0.007)))
        gv, gf = [], []
        for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):   # the corner bars: they shade the light
            q = [(x + sx * (r_ + t_ + w_), yl + sy * (r_ + t_ - w_), z_) for t_ in (0.0012, -0.0012)
                 for (r_, z_) in ((b, z0), (a, z1)) for w_ in (0.007, -0.007)]
            fq = [(0, 1, 3, 2), (4, 6, 7, 5)] if sx * sy > 0 else [(0, 2, 3, 1), (4, 5, 7, 6)]
            iron.append(mesh_object(f"lan_bar{k}", q, fq))   # a slab across the corner: a face out, a face in
            gv += [(x + sx * (b - 0.004), yl + sy * (b - 0.004), z0), (x + sx * (a - 0.004), yl + sy * (a - 0.004), z1)]
        for k in range(4):
            k2 = (k + 1) % 4
            gf.append((2 * k, 2 * k2, 2 * k2 + 1, 2 * k + 1))
        glass = mesh_object("lan_glass", gv, gf)
        assign(glass, mat("glass_lantern", glow=1.0))
        flame = lathe("lan_flame", [(0.0, 0.0), (0.024, 0.026), (0.0, 0.075)], segments=4, loc=(x, yl, z0 + 0.05))
        assign(flame, mat("emissive", "#ffb45e", glow=8))
        light(name, (x, yl - 0.2, z0 + 0.1), intensity=intensity, range=8.0, flicker=0.3, shadow_near=0.34)
        return z0 + 0.09
    iron.append(box("lan_plate", (0.06, 0.03, 0.38), loc=(x, y0 + 0.007, za - 0.12)))
    for dz in (0.035, -0.275):
        iron.append(nut("lan_bolt", 0.022, 0.01, (x, y0 - 0.012, za + dz)))
    iron.append(box("lan_arm", (0.02, reach + 0.05, 0.02), loc=(x, y0 - (reach + 0.05) / 2, za)))
    stay = bezier_path((x, y0 - 0.01, za - 0.27), (x, y0 - 0.05, za - 0.25), (x, y0 - reach * 0.5, za - 0.15), (x, y0 - reach * 0.72, za - 0.012), n=7)
    flat = [(-0.012, -0.004), (0.012, -0.004), (0.012, 0.004), (-0.012, 0.004)]
    iron.append(sweep("lan_stay", stay, flat, up=(1, 0, 0)))
    curl = spiral_path(0.007, 0.034, 1.3, n=14, a0_deg=-90, plane="YZ", center=(x, yl - 0.05, za + 0.036))
    iron.append(sweep("lan_curl", curl, flat, up=(1, 0, 0), scale_fn=lambda t: 0.5 + 0.5 * t))
    iron.append(torus("lan_hook", 0.016, 0.005, loc=(x, yl, za - 0.026), rot=(0, 90, 0), major_seg=8, minor_seg=4))
    iron.append(cylinder("lan_rod", 0.005, 0.05, loc=(x, yl, za - 0.062), verts=5))
    zt = za - 0.08
    a, b, hb = 0.092, 0.062, 0.2
    z1 = zt - 0.105
    z0 = z1 - hb
    iron.append(lathe("lan_roof", [(0.0, 0.0), (0.143, 0.0), (0.148, 0.006), (0.088, 0.05), (0.05, 0.062), (0.042, 0.084), (0.018, 0.093),
                                   (0.015, 0.102), (0.0, 0.107)], segments=4, loc=(x, yl, z1), rot=(0, 0, 45), smooth_angle=20))
    iron.append(box("lan_base", (2 * b + 0.018, 2 * b + 0.018, 0.014), loc=(x, yl, z0 - 0.007)))
    iron.append(sphere("lan_finial", 0.014, loc=(x, yl, z0 - 0.024), segments=6, rings=4))
    iron.append(cylinder("lan_burner", 0.017, 0.045, loc=(x, yl, z0 + 0.0225), verts=6))
    gv, gf = [], []
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        iron.append(sweep(f"lan_bar{k}", [(x + sx * b, yl + sy * b, z0), (x + sx * a, yl + sy * a, z1)], 0.0055, sides=4, smooth_path=False))
        gv += [(x + sx * (b - 0.004), yl + sy * (b - 0.004), z0), (x + sx * (a - 0.004), yl + sy * (a - 0.004), z1)]
    for k in range(4):
        k2 = (k + 1) % 4
        gf.append((2 * k, 2 * k2, 2 * k2 + 1, 2 * k + 1))
    glass = mesh_object("lan_glass", gv, gf)
    assign(glass, mat("glass_lantern", glow=1.0))
    flame = lathe("lan_flame", [(0.0, 0.0), (0.014, 0.012), (0.017, 0.03), (0.009, 0.056), (0.0, 0.075)], segments=8, loc=(x, yl, z0 + 0.05))
    assign(flame, mat("emissive", "#ffb45e", glow=8))
    light(name, (x, yl - 0.2, z0 + 0.1), intensity=intensity, range=8.0, flicker=0.3, shadow_near=0.34)
    return z0 + 0.09


def build(p, rng):
    L, D = float(p.length), float(p.depth)
    hl = L / 2
    st = STYLES[p.masonry]
    rec = st.rec
    # lod 2 (META "lod": "far", docs/LOD.md) is modelled here: the wall as one face per stone on a closed mortar face,
    # the coping as one prism per run, the steps as tread / riser / end, chain, ring and lantern as a few faces.
    # Every random stream draws what it draws at lod 0; the water stairs are the feature that is worked out, a
    # ladder or an outfall keeps its lod 0 parts.
    far = lod() == 2
    top = -COP_TH
    tz = -D + min(p.waterline, D - 0.45)             # water level
    zs = course_lines(top, -D, p.course * st.cscale)
    FO, FU, FN = Vector((0, 0, 0)), Vector((1, 0, 0)), Vector((0, -1, 0))

    m_stone = {"a": mat(st.lib, p.stone, wear=2.5), "w": mat(st.lib, shade(p.stone, 0.9, (1.0, 0.99, 0.955)), rough=0.5, wear=2.5)}
    m_cop = mat("stone_smooth", p.coping_stone)
    m_mortar = mat("mortar", "#9b9280")
    m_iron = mat("iron_wrought")
    m_moss = mat("moss")

    step_objs, cop_objs, dress_objs, iron_objs, dark_objs = [], [], [], [], []
    zones, stains, joints, cramps, lans = [], [], [], [], []
    B = Blocks(st, rng.fork("blocks"), tz, tz + 0.3)

    MV, MF = [], []

    def fan(V, F, *pts):
        """Far level: a face as triangles about its first point (corners given outright, shared by position)."""
        b = len(V)
        V.extend(Vector(q) for q in pts)
        F.extend((b, b + k, b + k + 1) for k in range(1, len(pts) - 1))

    def mortar_rect(O, U, N, u0, u1, v0, v1, d=None):
        """Mortar bed behind the blocks; rows on the course lines so every vertex sits in a bed joint."""
        if u1 - u0 < 1e-3 or v1 - v0 < 1e-3:
            return
        d = -rec if d is None else d
        if far:
            fan(MV, MF, *[O + U * u + ZV * v + N * d for (u, v) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
            return
        nu = max(1, int(ceil((u1 - u0) / 0.4)))
        vs = sorted([v0, v1] + [z for z in zs if v0 + 0.03 < z < v1 - 0.03])
        base = len(MV)
        for v in vs:
            for i in range(nu + 1):
                MV.append(O + U * lerp(u0, u1, i / nu) + ZV * v + N * d)
        for k in range(len(vs) - 1):
            for i in range(nu):
                a = base + k * (nu + 1) + i
                MF.append((a, a + 1, a + nu + 2, a + nu + 1))

    BV, BF = [], []

    def quad(*pts):
        BV.extend(pts)
        BF.append(tuple(range(len(BV) - 4, len(BV))))

    # ------------------------------------------------------------------ resolve the feature
    feat = p.feature
    cols, zbs = [], []
    if feat == "steps":
        drop = max(0.3, -(tz + 0.12))
        nst = max(2, int(round(drop / 0.175)))
        rise = drop / nst
        going, land = 0.28, 0.75
        R = (nst - 1) * going + land
        if R > L - 0.7:
            going, land = 0.23, 0.6
            R = (nst - 1) * going + land
        if R > L - 0.7:
            feat = "ladder"
        else:
            xh = -R / 2 + p.feature_pos * ((L - R) / 2 - 0.3)
            xs1 = xh + R
            for i in range(nst):
                xa = xh + i * going
                xb = xs1 if i == nst - 1 else xa + going
                cols.append((xa, xb, -(i + 1) * rise))
                # the end of each step stone shows in the face down to the next bed joint; coursed masonry below
                zbs.append(max([z for z in zs if z <= -(i + 2) * rise - 0.06] or [-D]))
            z_land = cols[-1][2]
            zones.append((xh - 0.15, xs1 + 0.15))
    if feat == "outfall":
        z_inv = tz + 0.1
        ro = p.outfall_size / 2
        ro = min(ro, ((top - 0.06) - z_inv) / 1.95, (L - 0.5) / 2 / 1.45)
        if ro < 0.12:
            feat = "none"
        else:
            vw = max(0.14, 0.42 * ro)
            spring = 0.5 * ro
            zc = z_inv + spring
            Ro = ro + vw
            Rb = ro + 0.55 * vw                         # the coursed blocks run in behind the arch ring to here
            xo = p.feature_pos * max(0.0, hl - Ro - 0.25)
            xa_o, xb_o = xo - Ro, xo + Ro
            ztopf = min([z for z in zs if z >= zc + Ro + 0.02] or [top])
            zbot = max([z for z in zs if z <= z_inv - 0.1] or [-D])
            zones.append((xa_o - 0.15, xb_o + 0.15))

            def xe(v):
                dz = v - zc
                return Rb if dz <= 0 else (0.0 if dz >= Rb else sqrt(Rb * Rb - dz * dz))
    if feat == "ladder":
        xl = p.feature_pos * max(0.0, hl - 0.5)
        zones.append((xl - 0.42, xl + 0.42))

    # ------------------------------------------------------------------ front wall blocks + mortar bed
    for ci in range(len(zs) - 1):
        z1, z0 = zs[ci], zs[ci + 1]
        proud = 0.03 if (ci == len(zs) - 2 and len(zs) > 2) else 0.0
        ivs = [(-hl, hl, False, False)]
        if feat == "steps":
            xfull = next((c[0] for c, zb_ in zip(cols, zbs) if zb_ < z1 - 1e-6), None)
            if xfull is not None:
                ivs = [(-hl, xfull, False, False), (xs1, hl, z1 > z_land + 0.03, False)]
        elif feat == "outfall" and z0 >= zbot - 1e-6 and z1 <= ztopf + 1e-6:
            xem = xe(z0)
            ivs = [(-hl, xo - xem, False, False, None, lambda v: xo - xe(v)),
                   (xo + xem, hl, False, False, lambda v: xo + xe(v), None)]
        B.course(FO, FU, FN, z0, z1, ivs, proud)
    if far:                                             # the ends of the courses at either end of the run
        for sg in (-1, 1):
            B.side(sg * (hl - st.jw * 0.5), sg, -D, min(tz + 0.3, top), "w")
            if tz + 0.3 < top:
                B.side(sg * (hl - st.jw * 0.5), sg, tz + 0.3, top, "a")

    zcap = top + 0.004
    # Far level: the mortar face runs on up inside the coping to just under the quay, and face and recess (under
    # the treads, behind the risers, back and end wall) are one sheet that shares its edges corner for corner, so
    # no edge ends on the middle of another. The town's lowest level may flatten or drop a thin part in front of
    # it (kerb, coping, a step): what shows then is mortar, never the sky.
    mt = -0.012 if far else top
    if feat == "steps" and far:
        # under the treads by 6 cm and behind the risers by 5: well clear of the stones at any viewing distance
        yf, yb, xH, xE = rec, STAIR_Y + rec, xh - 0.05, xs1 + rec
        zc = [zt - 0.06 for (xa, xb, zt) in cols]
        xc = [xH] + [c[0] - 0.05 for c in cols[1:]] + [xE]
        fan(MV, MF, (xH, yf, zc[0]), (xH, yf, mt), (-hl, yf, mt), (-hl, yf, zcap), (-hl, yf, -D), (xH, yf, -D))
        for i in range(nst):
            xa, xb = xc[i], xc[i + 1]
            if i < nst - 1:
                fan(MV, MF, (xa, yf, zc[i]), (xa, yf, -D), (xb, yf, -D), (xb, yf, zc[i + 1]), (xb, yf, zc[i]))
            else:
                fan(MV, MF, (xa, yf, zc[i]), (xa, yf, -D), (xb, yf, -D), (xb, yf, zc[i]))
            fan(MV, MF, (xa, yf, zc[i]), (xb, yf, zc[i]), (xb, yb, zc[i]), (xa, yb, zc[i]))           # under the tread
            zr = mt if i == 0 else zc[i - 1]                                                          # behind the riser
            fan(MV, MF, (xa, yf, zc[i]), (xa, yb, zc[i]), (xa, yb, zr), (xa, yf, zr))
        fan(MV, MF, (xE, yf, zc[-1]), (xE, yf, -D), (hl, yf, -D), (hl, yf, zcap), (hl, yf, mt), (xE, yf, mt))
        prof = [(xH, yb, mt)]                                                                         # back of the recess
        for i in range(nst):
            prof += [(xc[i], yb, zc[i]), (xc[i + 1], yb, zc[i])]
        fan(MV, MF, (xE, yb, mt), *prof)
        fan(MV, MF, (xE, yb, zc[-1]), (xE, yf, zc[-1]), (xE, yf, mt), (xE, yb, mt))                   # its end wall
        caps = [(-hl, xH, rec, COP_Y1), (xE, hl, rec, COP_Y1), (xH, xE, yb, COP_Y1)]
    elif feat == "steps":
        mortar_rect(FO, FU, FN, -hl, xh, -D, top)
        for (xa, xb, zt) in cols:
            mortar_rect(FO, FU, FN, xa, xb, -D, zt - 0.015)
        mortar_rect(FO, FU, FN, xs1, hl, -D, top)
        mortar_rect(Vector((0, STAIR_Y, 0)), FU, FN, xh, xs1, z_land - rise, top)
        mortar_rect(Vector((xs1, 0, 0)), Vector((0, -1, 0)), Vector((-1, 0, 0)), -STAIR_Y, 0.0, z_land - rise, top)
        caps = [(-hl, xh, rec, COP_Y1), (xs1, hl, rec, COP_Y1), (xh, xs1, STAIR_Y + rec, COP_Y1)]
    elif feat == "outfall":
        mortar_rect(FO, FU, FN, -hl, xa_o, -D, mt)
        mortar_rect(FO, FU, FN, xb_o, hl, -D, mt)
        mortar_rect(FO, FU, FN, xa_o, xb_o, -D, zbot)
        mortar_rect(FO, FU, FN, xa_o, xb_o, ztopf, mt)
        Rm = 0.5 * (ro + Ro)
        xk = [-Ro] + [-Rm * cos(radians(a)) for a in range(0, 181, 15)] + [Ro]
        for q in range(len(xk) - 1):                    # strips whose lower edge hides behind the arch ring
            d0, d1 = sorted((abs(xk[q]), abs(xk[q + 1])))
            if d1 > Rm + 1e-6:
                v0 = zc
            else:
                zi = sqrt(ro * ro - d0 * d0) if d0 < ro else 0.0
                v0 = zc + 0.5 * (zi + sqrt(Ro * Ro - d1 * d1))
            mortar_rect(FO, FU, FN, xo + xk[q], xo + xk[q + 1], v0, ztopf)
        caps = [(-hl, hl, rec, COP_Y1)]
    else:
        mortar_rect(FO, FU, FN, -hl, hl, -D, mt)
        caps = [(-hl, hl, rec, COP_Y1)]
    # wall body: top bed under the coping, back and ends (buried in the quay / against the next run)
    for (x0, x1, y0, y1) in caps:
        quad((x0, y0, zcap), (x1, y0, zcap), (x1, y1, zcap), (x0, y1, zcap))
    quad((-hl, BODY_Y, -D), (-hl, BODY_Y, zcap), (hl, BODY_Y, zcap), (hl, BODY_Y, -D))
    quad((hl, rec, -D), (hl, BODY_Y, -D), (hl, BODY_Y, zcap), (hl, rec, zcap))
    quad((-hl, rec, zcap), (-hl, BODY_Y, zcap), (-hl, BODY_Y, -D), (-hl, rec, -D))

    # ------------------------------------------------------------------ coping
    rc = rng.fork("coping")
    cop_runs = [(-hl, hl)] if feat != "steps" else [(-hl, xh), (xs1, hl)]
    ci = 0
    pointing = []
    for (xa, xb) in cop_runs:
        if xb - xa < 0.05:
            continue
        xs = split_run(xa, xb, 0.85, 1.35, rc)
        dq = [None] + [rc.pick((1, 2)) for _ in xs[1:-1]] + [None]
        if far:
            cop_objs.append(coping_far(f"coping_run{len(cop_objs)}", xa, xb, COP_Y0, COP_Y1, COP_TH, 0.035, (True, True)))
        for q in range(len(xs) - 1):
            o = coping_stone(f"coping{ci}", xs[q] + 0.004, xs[q + 1] - 0.004, COP_Y0, COP_Y1, COP_TH, rc, dish=(dq[q], dq[q + 1]))
            if far:
                delete(o)                               # built for the sake of the random stream only
            else:
                cop_objs.append(o)
            ci += 1
            if q > 0:
                joints.append(xs[q])
                cramps.append((xs[q], COP_Y1 - (COP_Y1 - COP_Y0 - 0.035) * dq[q] / 4))
                pointing.append((xs[q], COP_Y0 + 0.014, COP_Y1 - 0.01))
    if feat == "steps":
        xs = split_run(xh, xs1, 0.9, 1.4, rc)
        if far:                                         # a closed prism: its ends lie against the coping to either side
            cop_objs.append(coping_far("kerb_run", xh, xs1, STAIR_Y - 0.06, COP_Y1, COP_TH, 0.02, (True, True), wall=STAIR_Y + rec))
        for q in range(len(xs) - 1):
            o = coping_stone(f"kerb{q}", xs[q] + 0.004, xs[q + 1] - 0.004, STAIR_Y - 0.06, COP_Y1, COP_TH, rc, nose=0.02)
            if far:
                delete(o)
            else:
                cop_objs.append(o)
        pointing += [(x, STAIR_Y - 0.05, COP_Y1 - 0.01) for x in xs]
    for (xj, ya, yb) in ([] if far else pointing):      # recessed pointing in the coping joints
        quad((xj - 0.006, ya, -0.006), (xj + 0.006, ya, -0.006), (xj + 0.006, yb, -0.006), (xj - 0.006, yb, -0.006))
        quad((xj - 0.006, ya, top), (xj + 0.006, ya, top), (xj + 0.006, ya, -0.006), (xj - 0.006, ya, -0.006))
    # iron dog cramps let into the top across every coping joint
    for (xj, yc) in cramps:
        cl_, ca_ = rc.u(0.2, 0.25), rc.jitter(0, 3)
        if not far:
            iron_objs.append(box("cramp", (cl_, 0.03, 0.01), loc=(xj, yc, -0.006), rot=(0, 0, ca_)))

    # ------------------------------------------------------------------ features
    y_rock = -(st.bulge * 0.9)                          # where fittings meet the block faces
    lan_x = None
    rung_obj = None
    if feat == "ladder":
        rl = rng.fork("ladder")
        wl, yo = 0.42, -0.15
        zb = -D + 0.06
        bar = rounded_rect(0.046, 0.012, 0.003, 2)      # flat-bar stiles, wide face to the water
        for sx in (-1, 1):
            x = xl + sx * wl / 2
            pts = [(x, yo, zb), (x, yo, (zb - 0.1) / 2), (x, yo, -0.1)]
            pts += bezier_path((x, yo, -0.1), (x, yo, 0.3), (x, 0.24, 0.3), (x, 0.24, 0.004), n=14)[1:]
            iron_objs.append(sweep(f"stile{sx}", pts, bar, up=(0, -1, 0), smooth_path=False))
            iron_objs.append(box("foot", (0.1, 0.11, 0.008), loc=(x, 0.24, 0.004)))
            z = -0.36
            while z > zb + 0.15:                        # stays: flat bar back to a bolted wall plate
                iron_objs.append(box("stay", (0.012, -yo - 0.004 + y_rock + 0.02, 0.042), loc=(x, (yo + 0.004 + y_rock + 0.02) / 2, z)))
                iron_objs.append(box("stay_plate", (0.1, 0.012, 0.13), loc=(x, y_rock - 0.002, z)))
                for dz in (-0.045, 0.045):
                    iron_objs.append(nut("stay_bolt", 0.02, 0.009, (x + sx * 0.03, y_rock - 0.012, z + dz)))
                z -= rl.u(0.74, 0.86)
        rung = cylinder("rung", 0.0115, wl, verts=8)
        ms, z = [], -0.16
        while z > zb + 0.1:
            ms.append(trs((xl + rl.jitter(0, 0.002), yo, z), (0, 90, rl.jitter(0, 1.5))))
            z -= 0.27
        rung_obj = instances_to_mesh("rungs", rung, ms, keep_base=False)
        iron_objs.append(rung_obj)
        for sx in (-1, 1):
            for (dx, dy) in ((-0.032, 0.035), (0.032, -0.035)):
                iron_objs.append(nut("ladder_bolt", 0.02, 0.008, (xl + sx * wl / 2 + dx, 0.24 + dy, 0.012), axis="z"))
        lan_x = xl + 0.8 if xl + 0.8 < hl - 0.3 else (xl - 0.8 if xl - 0.8 > -hl + 0.3 else None)

    if feat == "steps":
        rs = rng.fork("steps")
        BO = Vector((0, STAIR_Y, 0))
        # back wall of the recess (blocks start at the riser where the course first shows)
        for ci in range(len(zs) - 1):
            z1, z0 = zs[ci], zs[ci + 1]
            x0 = next((xa for (xa, xb, zt) in cols if zt < z1 - 0.03), None)
            if x0 is not None and z1 > z_land - rise:
                B.course(BO, FU, FN, z0, z1, [(x0, xs1, False, False)])
        # end wall of the recess above the landing (faces -X): the return of the pier, quoined to the front
        EO, EU, EN = Vector((xs1, 0, 0)), Vector((0, -1, 0)), Vector((-1, 0, 0))
        for ci in range(len(zs) - 1):
            z1, z0 = zs[ci], zs[ci + 1]
            if z1 > z_land + 0.03:
                B.add(EO, EU, EN, -STAIR_Y, 0.0, max(z0, z_land - 0.02), z1, flush=(False, True))
        # one-piece steps running into the back wall; their ends are bonded into the face masonry
        y_sf = -(0.012 + 0.6 * st.bulge)
        for i, (xa, xb, zt) in enumerate(cols):
            y0_, sd_ = y_sf - 0.005 * (i % 2) - rs.u(0, 0.002), int(rs.u(0, 1000))
            if far:
                step_objs.append(step_far(f"step{i}", xa, xb, zt, rise, zbs[i], y0_, STAIR_Y + 0.03, i < nst - 1, st.jw))
            else:
                step_objs.append(step_stone(f"step{i}", xa, xb, zt, rise, zbs[i], y0_, STAIR_Y + 0.03,
                                            rec + 0.003, sd_, riser=i < nst - 1, jw=st.jw))
        # grab chain swagged between eye bolts along the back wall
        yc_ = STAIR_Y + y_rock - 0.035

        def zhand(x):
            zt = next((c[2] for c in cols if c[0] <= x <= c[1] + 1e-6), z_land)
            return min(top - 0.2, zt + 0.9)

        ne = max(2, int(round((R - 0.5) / 0.95)) + 1)
        eyes = [xh + 0.3 + (R - 0.5) * k / (ne - 1) for k in range(ne)]
        eyes = [(x, yc_, zhand(x)) for x in eyes]
        cpath = [eyes[0]]
        fpath = [eyes[0]]                               # far level: the chain as a band through the low point of each swag
        for (ea, eb) in zip(eyes[:-1], eyes[1:]):
            sag = 0.09 + rs.u(0, 0.03)
            fpath += [(lerp(ea[0], eb[0], 0.5), yc_, lerp(ea[2], eb[2], 0.5) - 0.75 * sag), eb]
            cpath += bezier_path(ea, (lerp(ea[0], eb[0], 0.3), yc_, lerp(ea[2], eb[2], 0.3) - sag),
                                 (lerp(ea[0], eb[0], 0.7), yc_, lerp(ea[2], eb[2], 0.7) - sag), eb, n=8)[1:]
        if far:
            iron_objs.append(mesh_object("grab_chain", [(x, y, z + s) for (x, y, z) in fpath for s in (-0.014, 0.014)],
                                         [(2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1) for k in range(len(fpath) - 1)]))
        else:
            iron_objs.append(link_chain("grab_chain", cpath))
        for (x, y, z) in eyes:
            if not far:
                iron_objs.append(cylinder("chain_boss", 0.028, STAIR_Y - y, loc=(x, (STAIR_Y + y) / 2, z + 0.035), rot=(90, 0, 0), verts=6, radius2=0.02))
                iron_objs.append(torus("chain_eye", 0.026, 0.007, loc=(x, y, z + 0.014), rot=(0, 90, 0), major_seg=8, minor_seg=4))
            stains.append((x, z, 0.05, 0.5))
        lan_x = xs1 + 0.55 if xs1 + 0.55 < hl - 0.25 else (xh - 0.55 if xh - 0.55 > -hl + 0.25 else None)

    if feat == "outfall":
        ro_ = rng.fork("outfall")
        dv, yr = 0.5, 0.05                              # barrel depth, projection of the arch ring
        sill_h = z_inv - zbot
        dress_objs.append(stone_block("sill", (2 * Ro + 0.05, dv + yr + 0.025, sill_h - 0.006), loc=(xo, (dv - yr - 0.025) / 2, zbot + sill_h / 2),
                                      chamfer=0.016, rng=ro_, detail=0.16))
        for sx in (-1, 1):
            dress_objs.append(stone_block("jamb", (vw - 0.006, dv + yr, spring - 0.004), loc=(xo + sx * (ro + vw / 2), (dv - yr) / 2, z_inv + spring / 2),
                                          chamfer=0.012, rng=ro_, detail=0.16))
        nvz = max(5, int(round(pi * (ro + vw * 0.5) / 0.24)))
        nvz += 1 - nvz % 2
        zk = ztopf - zc - 0.006
        for k in range(nvz):
            a0, a1 = pi * k / nvz + 0.004 / Ro, pi * (k + 1) / nvz - 0.004 / Ro
            key = k == nvz // 2
            Rk = Ro + ro_.jitter(0, 0.005)
            pts = [(ro * cos(a0), ro * sin(a0))]
            if key:                                     # tall keystone rising to the next bed joint
                pts += [(zk / np.tan(a0), zk), (zk / np.tan(a1), zk)]
            else:
                pts += [(Rk * cos(a), Rk * sin(a)) for a in np.linspace(a0, a1, 4)]
            pts += [(ro * cos(a), ro * sin(a)) for a in np.linspace(a1, a0, 4)[:-1]]
            dp = dv + yr + (0.028 if key else ro_.jitter(0, 0.004))
            dress_objs.append(extrude_profile(f"voussoir{k}", pts, dp, loc=(xo, dv, zc), rot=(90, 0, 0), bevel=0.008, segments=1))
        # dark culvert behind the grate
        dk = mesh_object("culvert_back", [(xo - ro - 0.03, dv - 0.02, z_inv - 0.03), (xo + ro + 0.03, dv - 0.02, z_inv - 0.03),
                                          (xo + ro + 0.03, dv - 0.02, zc + ro + 0.03), (xo - ro - 0.03, dv - 0.02, zc + ro + 0.03)], [(0, 1, 2, 3)])
        dark_objs.append(dk)
        # grate: square bars leaded into sill and arch, flat rails, a hoop following the arch
        yg = 0.03
        nb = max(2, int(round(2 * ro / 0.12))) - 1
        for k in range(nb):
            x = -ro + 2 * ro * (k + 1) / (nb + 1)
            zt = zc + sqrt(max(ro * ro - x * x, 0.0)) + 0.015
            iron_objs.append(box(f"grate_bar{k}", (0.022, 0.022, zt - z_inv + 0.02), loc=(xo + x, yg, (zt + z_inv - 0.02) / 2), rot=(0, 0, 45)))
        for zr_ in ((z_inv + 0.11, zc + ro * 0.35) if spring + ro > 0.4 else (z_inv + (spring + ro) * 0.45,)):
            hw = sqrt(max(ro * ro - max(zr_ - zc, 0.0) ** 2, 0.0)) + 0.02
            iron_objs.append(box("grate_rail", (2 * hw, 0.014, 0.045), loc=(xo, yg - 0.017, zr_), bevel=0.002))
        hoop = [(xo - ro + 0.012, yg, z_inv)] + [(xo - (ro - 0.012) * cos(a), yg, zc + (ro - 0.012) * sin(a)) for a in np.linspace(0, pi, 15)] + [(xo + ro - 0.012, yg, z_inv)]
        iron_objs.append(sweep("grate_hoop", hoop, rounded_rect(0.012, 0.04, 0.002, 2), up=(0, -1, 0), smooth_path=False))
        # trickle of water over the sill lip
        tw = 0.02 + 0.035 * ro
        path = [(xo, 0.42, z_inv + 0.004), (xo, 0.1, z_inv + 0.004), (xo, -yr - 0.02, z_inv + 0.004), (xo, -yr - 0.045, z_inv - 0.012),
                (xo, -yr - 0.06, z_inv - 0.07), (xo, -yr - 0.068, tz - 0.03)]
        tr = sweep("trickle", path, [(-tw, 0.0), (tw, 0.0), (tw, 0.004), (-tw, 0.004)], up=(0, 0, 1), smooth_path=False)
        assign(tr, mat("water_fall", flow=0.7))
        fx("outfall_drip", "drip", (xo, -yr - 0.06, z_inv - 0.02), size=tw * 2, rate=6)
        stains.append((xo, z_inv, ro * 0.75, 0.85))
        lan_x = xb_o + 0.55 if xb_o + 0.55 < hl - 0.3 else (xa_o - 0.55 if xa_o - 0.55 > -hl + 0.3 else None)

    if feat == "none":
        lan_x = p.feature_pos * max(0.0, hl - 0.45)

    # ------------------------------------------------------------------ lanterns (the first beside the feature)
    ztop_l = top - 0.04
    if lan_x is not None and ztop_l - 0.52 > tz + 0.1:
        def lan_ok(x):
            return -hl + 0.35 <= x <= hl - 0.35 and all(not (a - 0.2 < x < b + 0.2) for (a, b) in zones)

        nl = int(clamp(round(L / p.lantern_every), 1, 5))
        for k in range(nl):
            x = (lan_x + k * L / nl + hl) % L - hl
            if k > 0:
                x = next((x + d for d in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6) if lan_ok(x + d)), None)
                if x is None:
                    continue
            if p.lantern:
                zf = quay_lantern("quay_lantern" + (str(k + 1) if k else ""), x, y_rock - 0.004, ztop_l, -0.186, iron_objs, p.intensity, far)
                lans.append((x, zf))
            else:
                socket("lantern" + (str(k + 1) if k else ""), "attach", (x, 0.0, -0.3), role="lantern")
            zones.append((x - 0.3, x + 0.3))

    # ------------------------------------------------------------------ mooring rings
    rr = rng.fork("rings")
    zr = (zs[1] + zs[2]) / 2 if len(zs) > 3 else (zs[0] + zs[1]) / 2
    zr = max(zr, tz + 0.3)
    placed = []

    def free(x):
        return (-hl + 0.2 <= x <= hl - 0.2 and all(not (a < x < b) for (a, b) in zones)
                and all(abs(x - q) > 0.4 for q in placed))

    nr = int(p.rings)
    for k in range(nr):
        x = -hl + (k + 0.5) * L / nr
        if not free(x):
            cands = [e for (a, b) in zones for e in (a - 0.2, b + 0.2) if free(e)]
            if not cands:
                continue
            x = min(cands, key=lambda e: abs(e - x))
        placed.append(x)
        tl = rr.u(4, 10)
        xr_, yr_ = x + rr.jitter(0, 0.006), rr.jitter(0, 5)
        if far:                                         # the ring as a flat six-sided band hanging close to the face
            ro_, ri_, yq = 0.1, 0.07, -0.062
            V = [(xr_ + q * cos(k * pi / 3), yq, zr - 0.105 + q * sin(k * pi / 3)) for k in range(6) for q in (ri_, ro_)]
            iron_objs.append(mesh_object("ring", V, [(2 * k, 2 * k + 1, 2 * ((k + 1) % 6) + 1, 2 * ((k + 1) % 6)) for k in range(6)]))
        else:
            iron_objs.append(cylinder("ring_boss", 0.05, 0.07, loc=(x, -0.025, zr), rot=(90, 0, 0), verts=8, radius2=0.04))
            iron_objs.append(torus("ring_eye", 0.03, 0.0095, loc=(x, -0.068, zr), rot=(0, 90, 0), major_seg=8, minor_seg=4))
            iron_objs.append(torus("ring", 0.085, 0.0115, loc=(xr_, -0.086 - 0.07 * sin(radians(tl)), zr - 0.105),
                                   rot=(90 - tl, yr_, 0), major_seg=14, minor_seg=5))
        stains.append((x, zr - 0.05, 0.07, 0.8))
        socket(f"mooring{len(placed)}", "attach", (x, -0.1, zr - 0.1), role="mooring")

    # ------------------------------------------------------------------ realize, materials
    wall_objs = B.realize(m_stone)
    mortar_obj = mesh_object("mortar_bed", MV, MF)
    assign(mortar_obj, m_mortar)
    body_obj = mesh_object("wall_body", BV, BF)
    assign(body_obj, m_mortar)
    for o in cop_objs + dress_objs:
        assign(o, m_cop)
    for i, o in enumerate(step_objs):
        assign(o, m_stone["w" if cols[i][2] < tz + 0.5 else "a"])
    for o in dark_objs:
        assign(o, m_mortar)
    for o in iron_objs:
        assign(o, m_iron)
        smooth(o, 40)

    # ------------------------------------------------------------------ moss: low clumps hugging the damp joints
    rm = rng.fork("moss")
    beds = [b for b in B.beds if tz - 0.02 < b[5] < tz + 0.8 and b[5] > -D + 0.02]
    tm = []
    if beds and p.moss > 0.03:
        for _ in range(int(min(p.moss * L * 3.0, 26))):
            (O, U, N, u0, u1, v) = rm.pick(beds)
            u = lerp(u0, u1, rm.pick((0.0, 1.0)) if rm.chance(0.4) else rm.u())
            if abs(U.x) > 0.5:
                u = clamp(u, -hl + 0.13, hl - 0.13)
            k = 1.0 - 0.5 * clamp((v - tz) / 0.8)
            nbl = rm.pick((2, 3, 3, 4))
            ang = degrees(atan2(U.y, U.x))
            for q in range(nbl):
                sx = rm.u(0.016, 0.04) * k
                c = O + U * (u + (q - (nbl - 1) / 2) * rm.u(0.02, 0.045)) + ZV * (v + rm.u(-0.006, 0.004)) + N * rm.u(-0.006, 0.003)
                tm.append(trs(tuple(c), (rm.jitter(0, 8), rm.jitter(0, 14), ang), (sx, rm.u(0.007, 0.013), rm.u(0.007, 0.016) * k)))
    if feat == "steps" and p.moss > 0.05:
        for (xa, xb, zt) in cols:                       # in the back corner of the low treads
            if zt < tz + 0.9 and rm.chance(0.3 + 0.7 * p.moss):
                x = rm.u(xa + 0.05, xb - 0.05)
                for q in range(rm.pick((2, 3))):
                    tm.append(trs((x + q * rm.u(0.03, 0.05), STAIR_Y + y_rock - 0.012, zt), (0, 0, rm.jitter(0, 20)),
                                  (rm.u(0.02, 0.045), rm.u(0.012, 0.022), rm.u(0.006, 0.012))))
    for xj in joints:
        if rm.chance(p.moss * 0.8):
            y = rm.u(0.24, 0.4)
            for q in range(rm.pick((1, 2, 3))):
                tm.append(trs((xj, y + q * rm.u(0.03, 0.05), -0.006), (0, 0, 90 + rm.jitter(0, 10)), (rm.u(0.016, 0.035), rm.u(0.005, 0.008), rm.u(0.005, 0.009))))
    mo = None
    if tm and not far:                                  # far level: no tufts (the green of the tide line is in the masks)
        tuft = sphere("tuft", 1.0, segments=5, rings=3)
        mo = instances_to_mesh("moss_tufts", tuft, tm, keep_base=False)
        displace_noise(mo, strength=0.005, scale=0.025, seed=p.seed)
        assign(mo, m_moss)
        smooth(mo, 80)

    # ------------------------------------------------------------------ weathering
    ma = float(p.moss)
    nm = noise_mask(0.45, seed=p.seed + 11, lo=0.3, hi=0.7)
    nm2 = noise_mask(0.13, seed=p.seed + 12, lo=0.35, hi=0.7)
    nmw = noise_mask(0.8, seed=p.seed + 13, lo=0.0, hi=1.0)
    nmf = noise_mask(0.045, seed=p.seed + 14, lo=0.25, hi=0.8)
    rw = rng.fork("scale")
    zmid = [z for z in zs[1:-1] if tz + 0.2 < z < tz + 1.1] or [tz + 0.5]
    runs = [(rw.u(-hl + 0.2, hl - 0.2), rw.pick(zmid), rw.u(0.45, 1.0), rw.u(0.022, 0.05), rw.u(0.45, 0.8)) for _ in range(int(L * 0.7) + 1)]

    def tide_h(co, n):                                                            # height above a wavy water mark
        return co[:, 2] - tz + 0.07 * (nmw(co, n) - 0.5)

    def streaks(co):
        x, z = co[:, 0], co[:, 2]
        g = np.zeros(len(x))
        for xj in joints:                                                         # rain streaks under the coping joints
            g = np.maximum(g, 0.55 * np.exp(-((x - xj) / 0.05) ** 2) * np.clip(1.0 + (z + COP_TH) / 1.3, 0.0, 1.0) * (z < top + 0.02))
        for (sx_, sz_, sw_, ss_) in stains:                                       # rust / outfall stains
            g = np.maximum(g, ss_ * np.exp(-((x - sx_) / sw_) ** 2) * np.clip(1.0 - (sz_ - z) / 1.1, 0.0, 1.0) * (z < sz_))
        return g

    def soot(co):                                                                 # lamp black above each lantern
        x, z = co[:, 0], co[:, 2]
        g = np.zeros(len(x))
        for (lx, lz) in lans:
            g = np.maximum(g, 0.45 * np.exp(-((x - lx) / 0.13) ** 2) * np.clip((z - lz - 0.12) / 0.1, 0.0, 1.0))
        return g

    def scale_runs(co):                                                           # pale limescale / efflorescence runs
        x, z = co[:, 0], co[:, 2]
        wv = np.zeros(len(x))
        for (sx_, s0, sl, sw_, ss_) in runs:
            wv = np.maximum(wv, ss_ * np.exp(-((x - sx_) / sw_) ** 2) * np.clip((s0 - z) / 0.04, 0.0, 1.0) * np.clip(1.0 - (s0 - z) / sl, 0.0, 1.0) ** 0.6)
        return wv * (co[:, 1] < 0.1)

    def algae(co, n):                                                             # the bright line at the mean water level
        h = tide_h(co, n)
        return np.exp(-((h - 0.03) / 0.09) ** 2)

    def wall_grime(co, n):
        z = co[:, 2]
        h = tide_h(co, n)
        damp = np.where(h > 0, 0.58 * np.clip(1.0 - h / 0.42, 0.0, 1.0) ** 1.15,  # damp-darkened above the water mark
                        0.5 + 0.13 * nm(co, n))                                    # slime below, broken, never solid black
        damp = damp * (1.0 - 0.8 * algae(co, n)) * (1.0 - 0.85 * scale_runs(co))
        g = np.maximum(damp, 0.22 * np.clip((z + 0.45) / 0.28, 0.0, 1.0))         # dirt washed off the quay
        g = np.maximum(g, 0.2 * nm(co, n) * nm2(co, n))                           # general soot / damp patches
        return np.maximum(np.maximum(g, streaks(co)), soot(co))

    def wall_moss(co, n):
        h = tide_h(co, n)
        line = algae(co, n) * (0.72 + 0.2 * nm2(co, n) + 0.2 * (ma - 0.5))
        splash = (0.45 + 0.3 * ma) * np.clip(1.0 - h / 0.45, 0.0, 1.0) * (h > 0) * (0.4 + 0.6 * nm(co, n))
        sub = (0.46 + 0.26 * nm2(co, n) + 0.12 * ma) * (h <= 0)                     # olive slime below, with breaks
        hi = (0.3 + 0.5 * ma) * nm(co, n) * nm2(co, n) * (h < 1.3)
        m = np.maximum(np.maximum(line, splash), np.maximum(sub, hi))
        m = np.maximum(m, (0.55 + 0.3 * ma) * streaks(co) * (0.4 + 0.6 * nm2(co, n)))
        return np.clip(m * (1.0 - 0.8 * scale_runs(co)), 0.0, 1.0) * (1.0 if ma > 0.01 else 0.0)

    def wall_wear(co, n):
        return scale_runs(co) * (0.5 + 0.5 * nmf(co, n))

    for o in wall_objs:
        G, W, K = B.extra[o.name]

        def bg(co, n, G=G, K=K):
            g = wall_grime(co, n)
            return np.clip(np.maximum(g, 0.3 * K) + G * (1.0 - 1.4 * np.clip(g, 0.0, 0.7)), 0.0, 1.0) if len(G) == len(g) else g

        def bw(co, n, W=W):
            wv = wall_wear(co, n)
            return np.clip(wv + W, 0.0, 1.0) if len(W) == len(wv) else wv

        paint(o, grime=bg, wear=bw, special=wall_moss)
    paint(mortar_obj, grime=lambda co, n: np.clip(0.15 + 0.2 * nm2(co, n) + 0.6 * wall_grime(co, n), 0.0, 0.58),
          wear=wall_wear, special=lambda co, n: np.clip(1.15 * wall_moss(co, n), 0.0, 1.0))
    paint(body_obj, grime=lambda co, n: 0.2 + 0.25 * nm(co, n))
    for o in dark_objs:
        paint(o, grime=1.0)

    def step_grime(co, n):
        y, z = co[:, 1], co[:, 2]
        g = np.maximum(wall_grime(co, n), 0.5 * np.clip((tz + 0.95 - z) / 0.85, 0.0, 1.0))     # wetter towards the water
        g = np.maximum(g, 0.5 * np.clip((y - (STAIR_Y - 0.14)) / 0.12, 0.0, 1.0))              # dirt in the back corner
        return np.maximum(g, 0.16 * nm2(co, n))

    def step_wear(co, n):
        y, z = co[:, 1], co[:, 2]
        tread = (n[:, 2] > 0.5) * np.exp(-((y - 0.19) / 0.13) ** 2)
        return np.maximum(0.45 * tread * np.clip((z - tz) / 0.9, 0.1, 1.0), wall_wear(co, n))

    def step_moss(co, n):
        y, z = co[:, 1], co[:, 2]
        prog = np.clip((tz + 0.8 - z) / 0.65, 0.0, 1.0)                                         # bottom 3-4 treads and the landing
        foot = 1.0 - 0.55 * (n[:, 2] > 0.5) * np.exp(-((y - 0.19) / 0.13) ** 2)
        m = prog * (0.3 + 0.7 * nm2(co, n)) * foot * (0.35 + 1.1 * ma)
        return np.clip(np.maximum(m, wall_moss(co, n)), 0.0, 1.0) * (1.0 if ma > 0.01 else 0.0)

    for o in step_objs:
        paint(o, grime=step_grime, wear=step_wear, special=step_moss)

    def cop_wear(co, n):
        y, z = co[:, 1], co[:, 2]
        tread = (n[:, 2] > 0.6) * np.exp(-((y - 0.16) / 0.2) ** 2) * 0.22 * (0.5 + 0.5 * nm(co, n))
        nose = np.clip((COP_Y0 + 0.1 - y) / 0.1, 0.0, 1.0) * (z > -0.07) * (y < 0.1) * (0.3 + 0.3 * nmf(co, n))
        return np.maximum(tread, nose)

    def cop_grime(co, n):
        x, y, z = co[:, 0], co[:, 1], co[:, 2]
        g = wall_grime(co, n) * (z < -0.05)
        g = np.maximum(g, 0.34 * np.clip((y - 0.36) / 0.16, 0.0, 1.0) * (z > -0.05))            # where nobody walks
        g = np.maximum(g, 0.38 * nm(co, n) * (0.4 + 0.6 * nm2(co, n)))                           # soot blotches
        g = np.maximum(g, 0.3 * np.clip((-0.07 - z) / 0.06, 0.0, 1.0) * (y < 0.0))               # drip line under the nosing
        for (lx, lz) in lans:                                                                     # lamp smoke on the nosing above
            g = np.maximum(g, 0.4 * np.exp(-((x - lx) / 0.2) ** 2) * (y < 0.03))
        for xj in joints:
            g = np.maximum(g, 0.6 * np.exp(-((x - xj) / 0.025) ** 2))
        return g

    def cop_moss(co, n):
        y, z = co[:, 1], co[:, 2]
        back = np.clip((y - 0.34) / 0.2, 0.0, 1.0) * (0.3 + 0.7 * nm(co, n)) * (z > -0.05) * 0.8
        return np.maximum(wall_moss(co, n) * (z < -0.1), back * (0.5 + ma)) * (1.0 if ma > 0.01 else 0.0)

    for o in cop_objs:
        paint(o, grime=cop_grime, wear=cop_wear, special=cop_moss)
    for o in dress_objs:                                                          # outfall dressings: tide + dark towards the barrel
        paint(o, grime=lambda co, n: np.maximum(np.maximum(wall_grime(co, n), 0.12 + 0.2 * nm2(co, n)), 0.95 * np.clip((co[:, 1] - 0.0) / 0.22, 0.0, 1.0)),
              wear=lambda co, n: wall_wear(co, n), special=wall_moss)
    if mo is not None:
        paint(mo, grime=lambda co, n: 0.35 + 0.3 * nm2(co, n))

    def rust(co, n):
        z = co[:, 2]
        return np.clip(0.22 + 0.36 * np.clip((tz + 1.0 - z) / 1.0, 0.0, 1.0) * (0.4 + 0.6 * nm2(co, n)), 0.0, 1.0)

    for o in iron_objs:
        grate = o.name.startswith("grate")
        paint(o, special=(lambda co, n: 0.75 + 0.25 * nm2(co, n)) if grate else rust,
              grime=lambda co, n: np.clip((tz + 0.3 - co[:, 2]) / 0.4, 0.0, 1.0) * 0.6)
    if rung_obj is not None:                                                      # boots keep the rungs bright in the middle
        paint(rung_obj, wear=lambda co, n: 0.6 * np.exp(-((co[:, 0] - xl) / 0.11) ** 2) * (co[:, 2] > tz + 0.2))

    socket("waterline", "attach", (0.0, -0.5, tz), role="water_level")
