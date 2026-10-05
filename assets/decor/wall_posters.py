"""A cluster of 2-6 old paper bills posted over one another on a wall: a framed proclamation with a
wood-type headline and a big emblem, a coloured 'banner' bill with a second line of text, and smaller
bills, notices and playbills with real display words.  The bills are clustered the way years of posting
leave them (staggered, tilted, one half buried under a later one, a small notice tacked over the lot).
Every sheet is its own mesh grid that is torn (jagged corner tears, bites, ripped-off strips), deckled,
rolled up at free corners and peeled along edges (true cylindrical curl showing the unprinted back),
pasted with pinched wrinkle ridges, paste bubbles and fold creases, or hung from tacks so that it bellies
off the wall and its lower edge lifts.  Upper sheets drape over the ones below.  The print (rules,
frames, a line of real small type and at most two greeked lines per bill, emblem, headlines) is separate
'ink' geometry clipped to the paper triangles
so it rides 0.3 mm proud of the paper through every wrinkle and curl.  Three draw calls: paper, ink /
coloured stock, iron fixings - paper tones and ink colours are told apart by the weathering masks.
"""
from tk import *
import bisect
import math
import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
import bpy
try:
    from tk.geo import _font as _tk_font
except Exception:
    _tk_font = None

META = {"name": "Pasted wall posters", "class": "prop_small", "mount": "wall", "mount_height": 1.55,
        "focus": [0.0, 0.0, 0.0], "tri_budget": 7500}
PARAMS = {
    "count": {"default": 4, "min": 2, "max": 6, "desc": "number of bills in the group"},
    "spread": {"default": 1.05, "min": 0.5, "max": 1.8, "desc": "maximum width of the group (m); the bills always overlap in a cluster"},
    "size": {"default": 0.58, "min": 0.4, "max": 0.8, "desc": "height of the main bill (m); the others scale with it"},
    "palette": {"default": "parchment", "choices": ["parchment", "royal", "festival", "faded"], "desc": "paper tint and the colour of the second ink / coloured stock"},
    "headline": {"default": "KEEP THE LAMPS LIT", "desc": "headline printed on the main bill"},
    "subline": {"default": "A BETTER TOMORROW", "desc": "text printed on the second (banner) bill"},
    "notices": {"default": "NOTICE,GRAND CONCERT,TO LET,TONIGHT,FOR SALE,MARKET DAY,LOST,AUCTION", "desc": "comma-separated display lines printed on the smaller bills"},
    "emblem": {"default": "crown", "choices": ["crown", "star", "moon", "sun", "mixed"], "desc": "big printed emblem shape"},
    "tattered": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "torn corners, bites, ripped strips, faded and rubbed print, dirt and damp (0 = freshly posted)"},
    "curl": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "how much free corners and lower edges roll away from the wall"},
    "tabs": {"default": True, "desc": "tear-off strips cut into the bottom edge of the small notice"},
    "pinned": {"default": 0.4, "min": 0.0, "max": 1.0, "desc": "share of the bills hung from tacks / nails instead of pasted"},
}
VARIANTS = {
    "royal_pair": {"count": 2, "spread": 0.8, "size": 0.72, "palette": "royal", "headline": "BY ORDER OF THE CROWN",
                   "subline": "CURFEW AT THE TENTH BELL", "tattered": 0.2, "curl": 0.4, "pinned": 0.0, "seed": 4},
    "festival_six": {"count": 6, "spread": 1.25, "size": 0.5, "palette": "festival", "headline": "LANTERN FAIR",
                     "subline": "MUSIC AND DANCING", "notices": "TONIGHT,GRAND PARADE,FIREWORKS,ALL WELCOME,SWEETS,MASQUE,PUPPETS,FREE CIDER",
                     "emblem": "mixed", "tattered": 0.35, "curl": 0.6, "pinned": 0.5, "seed": 11},
    "tattered_old": {"count": 5, "spread": 1.2, "size": 0.6, "palette": "faded", "headline": "REWARD", "subline": "LOST CAT",
                     "notices": "WANTED,STOLEN,BEWARE,LAST SEEN,NO ENTRY,MISSING,DEBTS,PLAGUE", "emblem": "moon", "tattered": 1.0, "curl": 0.9,
                     "pinned": 0.3, "seed": 23},
    "notice_trio": {"count": 3, "spread": 0.7, "size": 0.44, "palette": "parchment", "headline": "TOWN MEETING",
                    "subline": "ROOMS TO LET", "notices": "NOTICE,TO LET,LOST,FOUND", "emblem": "star", "tattered": 0.1, "curl": 0.3,
                    "pinned": 1.0, "seed": 31},
}

PALETTES = {
    #            paper tint, coloured stock = second ink (black ink is the same material driven dark by the grime mask)
    "parchment": ("#f4ead2", "#30446c"),
    "royal": ("#f1e6cc", "#5c1722"),
    "festival": ("#f6e7c0", "#1d5a50"),
    "faded": ("#d9d0b8", "#4b5361"),
}
GAP = 0.0009        # one layer: paper + paste film
INK_OFF = 0.0003    # ink stands 0.3 mm proud
BACK_T = 0.0005     # paper thickness where the back is visible
FALL = 0.03         # distance over which an upper sheet drapes down past the edge of a lower one


def _sstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------------ 2D helpers
def _rect(x0, z0, x1, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def _frame(x0, z0, x1, z1, t):
    return [_rect(x0, z0, x1, z0 + t), _rect(x0, z1 - t, x1, z1), _rect(x0, z0 + t, x0 + t, z1 - t), _rect(x1 - t, z0 + t, x1, z1 - t)]


def _disc(cx, cz, r, n=20):
    return [(cx + r * math.cos(2 * math.pi * i / n), cz + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _seal(cx, cz, r, rng, n=22):
    """A pressed wax seal / ink stamp: round-ish blob with an irregular squeezed-out rim."""
    ph = [rng.u(0, 6.28) for _ in range(3)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1.0 + 0.085 * math.sin(3 * a + ph[0]) + 0.055 * math.sin(5 * a + ph[1]) + 0.03 * math.sin(8 * a + ph[2]))
        pts.append((cx + rr * math.cos(a), cz + rr * math.sin(a)))
    return pts


def _area(poly):
    a = 0.0
    for i in range(len(poly)):
        x0, z0 = poly[i - 1]
        x1, z1 = poly[i]
        a += x0 * z1 - x1 * z0
    return 0.5 * a


def _tess(pts):
    """Concave outline -> list of CCW triangles."""
    tris = []
    for (a, b, c) in tessellate_polygon([[Vector((x, z, 0.0)) for (x, z) in pts]]):
        t = [pts[a], pts[b], pts[c]]
        ar = _area(t)
        if abs(ar) < 1e-10:
            continue
        tris.append(t if ar > 0 else t[::-1])
    return tris


def _emblem(kind, cx, cz, s):
    """Emblem of overall size s centred on (cx, cz) -> list of polygons."""
    def T(pts):
        return [(cx + x * s, cz + z * s) for (x, z) in pts]
    out = []
    if kind == "crown":
        out += _tess(T([(-0.40, -0.30), (0.40, -0.30), (0.50, 0.26), (0.25, -0.02), (0.0, 0.40), (-0.25, -0.02), (-0.50, 0.26)]))
        out.append(T(_rect(-0.40, -0.45, 0.40, -0.36)))
        for (x, z) in ((0.50, 0.31), (0.0, 0.45), (-0.50, 0.31)):
            out.append(T(_disc(x, z, 0.058, 20 if s > 0.07 else 10)))
    elif kind == "star":
        pts = []
        for i in range(10):
            r = 0.5 if i % 2 == 0 else 0.2
            a = math.pi / 2 + i * math.pi / 5
            pts.append((r * math.cos(a), r * math.sin(a) - 0.03))
        out += _tess(T(pts))
    elif kind == "moon":
        st = 12 if s > 0.07 else 24
        pts = [(0.5 * math.cos(math.radians(a)) - 0.08, 0.5 * math.sin(math.radians(a))) for a in range(40, 321, st)]
        pts += [(0.25 + 0.347 * math.cos(math.radians(a)) - 0.08, 0.347 * math.sin(math.radians(a))) for a in range(270, 89, -st - 2)]
        out += _tess(T(pts))
        out += _tess(T([(0.30 + 0.11 * math.cos(math.pi / 2 + i * math.pi / 5) * (1.0 if i % 2 == 0 else 0.42),
                         0.05 + 0.11 * math.sin(math.pi / 2 + i * math.pi / 5) * (1.0 if i % 2 == 0 else 0.42)) for i in range(10)]))
    else:  # sun
        out.append(T(_disc(0.0, 0.0, 0.25, 22 if s > 0.07 else 12)))
        for i in range(10):
            a = i * math.pi / 5
            da = 0.17
            out.append(T([(0.30 * math.cos(a - da), 0.30 * math.sin(a - da)), (0.5 * math.cos(a), 0.5 * math.sin(a)),
                          (0.30 * math.cos(a + da), 0.30 * math.sin(a + da))]))
    return out


def _clip(subj, tri):
    """Sutherland-Hodgman: polygon `subj` clipped to the convex CCW polygon `tri`."""
    out = subj
    m = len(tri)
    for i in range(m):
        ax, az = tri[i]
        bx, bz = tri[(i + 1) % m]
        ex, ez = bx - ax, bz - az
        inp = out
        out = []
        if len(inp) < 3:
            return []
        px, pz = inp[-1]
        dp = ex * (pz - az) - ez * (px - ax)
        for (qx, qz) in inp:
            dq = ex * (qz - az) - ez * (qx - ax)
            if dq >= 0.0:
                if dp < 0.0:
                    t = dp / (dp - dq)
                    out.append((px + (qx - px) * t, pz + (qz - pz) * t))
                out.append((qx, qz))
            elif dp >= 0.0:
                t = dp / (dp - dq)
                out.append((px + (qx - px) * t, pz + (qz - pz) * t))
            px, pz, dp = qx, qz, dq
    return out


_TT = {}


def _text_tris(body, font, res=3):
    """A text line as flat 2D triangles (cap height ~1).  Built as an un-extruded font curve at a low
    curve resolution: the print only needs the front faces, and 5 cm letters do not need 12-gon serifs."""
    key = (body, font, res)
    if key in _TT:
        return _TT[key]
    try:
        cu = bpy.data.curves.new("hl_tmp", 'FONT')
        cu.body = body
        cu.size = 1.0
        cu.extrude = 0.0
        cu.align_x = 'CENTER'
        cu.resolution_u = res
        fnt = _tk_font(font) if _tk_font else None
        if fnt is not None:
            cu.font = fnt
        tmp = bpy.data.objects.new("hl_tmp_o", cu)
        bpy.context.scene.collection.objects.link(tmp)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg), depsgraph=dg)
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.curves.remove(cu)
        polys = [[(me.vertices[i].co.x, me.vertices[i].co.y) for i in poly.vertices] for poly in me.polygons]
        bpy.data.meshes.remove(me)
    except Exception as e:                      # fall back to the toolkit text() and keep its front faces
        print("DBG text fallback", e)
        o = text("hl_tmp", body, size=1.0, depth=0.002, rot=(0, 0, 0), font=font)
        me = o.data
        polys = [[(me.vertices[i].co.x, me.vertices[i].co.y) for i in poly.vertices] for poly in me.polygons
                 if poly.normal.z > 0.5 and poly.center.z > 0.001]
        delete([o])
    tris = []
    for vs in polys:
        for i in range(1, len(vs) - 1):
            t = [vs[0], vs[i], vs[i + 1]]
            ar = _area(t)
            if abs(ar) > 1e-9:
                tris.append(t if ar > 0 else t[::-1])
    _TT[key] = tris
    return tris


def _mesh(name, verts, tris, uvs):
    o = mesh_object(name, verts, tris)
    me = o.data
    uvl = me.uv_layers.new(name="UVMap")
    for li, loop in enumerate(me.loops):
        uvl.data[li].uv = uvs[loop.vertex_index]
    uv_keep(o)
    smooth(o, 50)
    return o


def _lookup(vals, ref):
    """Mask callable returning per-vertex values made for `ref` (falls back to nearest vertex)."""
    vals = np.asarray(vals, dtype=np.float64)

    def fn(co, nrm):
        if len(co) == len(vals):
            return vals
        out = np.zeros(len(co))
        for s in range(0, len(co), 512):
            d = ((co[s:s + 512, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
            out[s:s + 512] = vals[np.argmin(d, axis=1)]
        return out
    return fn


def _groups(tris):
    """Connected components (shared vertices) of a triangle / quad list -> lists of face indices."""
    par = {}

    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for t in tris:
        r0 = find(t[0])
        for v in t[1:]:
            par[find(v)] = r0
    comp = {}
    for ti, t in enumerate(tris):
        comp.setdefault(find(t[0]), []).append(ti)
    return list(comp.values())


# ------------------------------------------------------------------------------------------------ a sheet of paper
def _curl_pen(c, x, z):
    """How far (m) the point lies inside the rolled-up zone of curl c (<= 0: outside)."""
    cx, cz, nx, nz, L, R, th, xr, lobe = c
    if xr and not (xr[0] - 1e-4 <= x <= xr[1] + 1e-4):
        return -1.0
    if lobe:
        q = (-(x - cx) * nz + (z - cz) * nx - lobe[0]) / lobe[1]
        if abs(q) >= 1.0:
            return -1.0
        L = L * (1.0 - q * q) ** 2
    return L - ((x - cx) * nx + (z - cz) * nz)


class Sheet:
    def __init__(self, k, kind, w, h):
        self.k, self.kind, self.w, self.h = k, kind, w, h
        self.cx = self.cz = 0.0
        self.ca, self.sa = 1.0, 0.0
        self.tears = []      # ('line', cx, cz, nx, nz, T, jag) | ('bite', ex, ez, r, jag)
        self.curls = []      # (cx, cz, nx, nz, L, R, thmax, x-range, lobe)
        self.pins = []       # (x, z)
        self.ridges = []     # broad undulations (px, pz, dx, dz, length, sigma, amp)
        self.pinch = []      # crisp pinched wrinkle ridges (axis, c, t0, tlen, amp, half width)
        self.bubbles = []    # paste bubbles (x, z, r, amp)
        self.creases_x, self.creases_z = [], []
        self.pinned = False
        self.pin_style = None
        self.belly = self.hang = self.skew = 0.0
        self.edge_amp, self.eph = 0.0, (0.0, 0.0, 0.0)
        self.below, self.uppers = [], []
        self.tab_h = 0.0     # height of the tear-off strips along the bottom (0 = none)
        self.slits, self.tab_ranges, self.tab_gone = [], [], set()

    def set_pose(self, cx, cz, tilt):
        self.cx, self.cz = cx, cz
        self.ca, self.sa = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))

    def world(self, x, z):
        return self.cx + x * self.ca - z * self.sa, self.cz + x * self.sa + z * self.ca

    def local(self, X, Z):
        dx, dz = X - self.cx, Z - self.cz
        return dx * self.ca + dz * self.sa, -dx * self.sa + dz * self.ca

    def rect_dist(self, x, z):
        return math.hypot(max(abs(x) - self.w / 2, 0.0), max(abs(z) - self.h / 2, 0.0))

    def tear_f(self, x, z):
        f = 1.0
        for t in self.tears:
            if t[0] == 'line':
                _, cx, cz, nx, nz, T, jag = t
                d = (x - cx) * nx + (z - cz) * nz - T
                s = -(x - cx) * nz + (z - cz) * nx
            else:
                _, ex, ez, r, jag = t
                d = math.hypot(x - ex, z - ez) - r
                s = math.atan2(z - ez, x - ex) * r
            for (a, kf, ph) in jag:
                d += a * math.sin(kf * s + ph)
            if d < f:
                f = d
        return f

    def curl_s(self, x, z):
        """Largest penetration into any rolled-up zone (> 0 = inside a curl)."""
        m = -1.0
        for c in self.curls:
            m = max(m, _curl_pen(c, x, z))
        if self.tab_h and z < -self.h / 2 + self.tab_h + 0.012:
            m = 1.0                                   # slits / missing strips: never counts as solid cover
        return m

    def support(self, X, Z):
        x, z = self.local(X, Z)
        d = self.rect_dist(x, z)
        if d >= FALL:
            return 0.0
        a = 1.0 - d / FALL
        f = self.tear_f(x, z)
        if f < 0.0:
            a *= max(0.0, 1.0 + f / FALL)
        return a

    def free(self, X, Z):
        """1 where no later sheet lies on this one, falling to 0 towards / under a covering sheet."""
        f = 1.0
        for up in self.uppers:
            ux, uz = up.local(X, Z)
            d = up.rect_dist(ux, uz)
            if d < FALL + 0.07:
                f *= _sstep(FALL + 0.004, FALL + 0.07, d)
        return f

    def loose_raw(self, x, z):
        """Stand-off of a sheet that hangs from tacks: belly between the fixings, lower edge lifting."""
        if not self.pin_style or not self.pins:
            return 0.0
        w, h = self.w, self.h
        if abs(x) > w / 2 + 1e-6 or abs(z) > h / 2 + 1e-6:
            return 0.0
        sg = (0.42 if self.pin_style == "four" else 0.36) * min(w, h)
        pf = 1.0
        for (qx, qz) in self.pins:
            pf *= 1.0 - math.exp(-((x - qx) ** 2 + (z - qz) ** 2) / (sg * sg))
        u, t = x / w, 0.5 - z / h
        wob = 0.8 + 0.2 * math.sin(x * 21.0 + 1.7 * self.k) * math.sin(z * 17.0 + self.k)
        if self.pin_style == "four":
            v = self.belly * wob * (1.0 - 0.35 * (2 * u) ** 4) * (1.0 - 0.35 * (2 * (t - 0.5)) ** 4) + self.hang * 0.35 * max(0.0, t) ** 2
        else:
            v = self.belly * wob * 0.6 + self.hang * max(0.0, t) ** 1.5 * (1.0 + self.skew * 2.0 * u)
            if self.pin_style == "top1":
                v += self.hang * 0.45 * (2 * u) ** 2
        return max(0.0, pf * v)

    def solid(self, X, Z):
        """True where this sheet certainly hides whatever is underneath."""
        x, z = self.local(X, Z)
        if abs(x) > self.w / 2 - 0.02 or abs(z) > self.h / 2 - 0.02:
            return False
        mg = 0.02 + (2.2 * self.loose_raw(x, z) if self.pin_style else 0.0)
        if abs(x) > self.w / 2 - mg or abs(z) > self.h / 2 - mg:
            return False
        if self.tear_f(x, z) < mg - 0.004:
            return False
        return self.curl_s(x, z) < -0.012

    def open_f(self, x, z):
        """0 under a sheet pasted on top (it presses the fine relief flat), 1 out in the open."""
        if not self.uppers:
            return 1.0
        X, Z = self.world(x, z)
        f = 1.0
        for up in self.uppers:
            ux, uz = up.local(X, Z)
            d = up.rect_dist(ux, uz)
            if d < 0.03:
                f *= _sstep(0.004, 0.03, d)
        return f

    def edge_lift(self, x, z):
        """Stretches of a pasted edge that have come a few mm away from the wall."""
        if self.edge_amp <= 0.0:
            return 0.0
        e = min(self.w / 2 - abs(x), self.h / 2 - abs(z))
        if e >= 0.045:
            return 0.0
        a = math.atan2(z / self.h, x / self.w)
        nse = math.sin(3.0 * a + self.eph[0]) * math.sin(5.0 * a + self.eph[1]) + 0.35 * math.sin(11.0 * a + self.eph[2])
        if nse <= 0.1:
            return 0.0
        return self.edge_amp * min(1.0, (nse - 0.1) / 0.5) * (1.0 - max(e, 0.0) / 0.045) ** 2

    def wrinkle(self, x, z, of=None):
        if of is None:
            of = self.open_f(x, z)
        v = self.edge_lift(x, z) * of if of > 0.0 else 0.0
        for (px, pz, dx, dz, ln, sg, amp) in self.ridges:
            t = (x - px) * dx + (z - pz) * dz
            d = -(x - px) * dz + (z - pz) * dx
            v += amp * (0.35 + 0.65 * of) * math.exp(-(d / sg) ** 2 - (t / ln) ** 2)
        if of > 0.0:
            for (ax, c, t0, tl, amp, hw) in self.pinch:
                d, t = (x - c, z - t0) if ax == 'x' else (z - c, x - t0)
                if abs(d) < hw:
                    v += of * amp * (1.0 - abs(d) / hw) * math.exp(-(t / tl) ** 2)
            for (bx, bz, r, amp) in self.bubbles:
                q = ((x - bx) ** 2 + (z - bz) ** 2) / (1.3 * r) ** 2
                if q < 1.0:
                    v += of * amp * (1.0 - q) ** 2
        if self.pins:
            pf = 1.0
            for (qx, qz) in self.pins:
                pf *= 1.0 - math.exp(-((x - qx) ** 2 + (z - qz) ** 2) / (0.045 ** 2))
            v *= pf
        cr = 0.0011 * (0.45 + 0.55 * of)
        for c in self.creases_x:
            v += cr * max(0.0, 1.0 - abs(x - c) / 0.0045)
        for c in self.creases_z:
            v += cr * max(0.0, 1.0 - abs(z - c) / 0.0045)
        return v

    def under(self, X, Z):
        b = 0.0
        for s in self.below:
            a = s.support(X, Z)
            if a > 0.0:
                b = max(b, a * s.top(X, Z))
        return b

    def top(self, X, Z):
        x, z = self.local(X, Z)
        return self.under(X, Z) + GAP + self.wrinkle(x, z)

    def deform(self, x, z, pressed=False):
        """Flat sheet coords -> (world xyz, curl lift, loose stand-off).  pressed: the vertex belongs to a mesh
        cell that reaches under a sheet pasted on top - it carries no fine relief, so that the straight mesh
        edges between it and its neighbours can never come up through the covering sheet."""
        dx = dz = lift = 0.0
        for c in self.curls:
            s = _curl_pen(c, x, z)
            if s > 0.0:
                nx, nz, R, thm = c[2], c[3], c[5], c[6]
                th = min(s / R, thm)
                ex = s - R * th
                adv = R * math.sin(th) + ex * math.cos(th)
                lift += R * (1.0 - math.cos(th)) + ex * math.sin(th)
                dx += nx * (s - adv)
                dz += nz * (s - adv)
        X0, Z0 = self.world(x, z)
        X, Z = self.world(x + dx, z + dz)
        if pressed:
            return (X, -(self.under(X0, Z0) + GAP + self.wrinkle(x, z, 0.0) + lift), Z), lift, 0.0
        lo = self.loose_raw(x, z)
        if lo > 0.0:
            lo *= self.free(X0, Z0)
        return (X, -(self.top(X0, Z0) + lo + lift), Z), lift, lo


def _axis_nodes(L, creases, feats=()):
    band = [0.0, 0.012, 0.026, 0.044, 0.068] if L >= 0.3 else ([0.0, 0.012, 0.028, 0.05] if L >= 0.16 else [0.0, 0.012, 0.03])
    inner = L - 2 * band[-1]
    n = max(1, int(round(inner / 0.078)))
    nodes = [-L / 2 + b for b in band] + [-L / 2 + band[-1] + inner * i / n for i in range(1, n)] + [L / 2 - b for b in reversed(band)]
    for (c, d) in [(c, 0.0045) for c in creases] + list(feats):
        nodes = [v for v in nodes if abs(v - c) > d * 1.5 + 0.004 or abs(abs(v) - L / 2) < 1e-6]
        nodes += [c - d, c, c + d]
    return sorted(nodes)


# ------------------------------------------------------------------------------------------------ arrangement
def _arrange(items, spread, hmax, r, tries=170):
    """Greedy clustered placement.  items (in placement order) are dicts with w, h, tilt, layer, vmin (least
    visible share), key / kw (local zone that should stay uncovered, weight), target (wanted covered share or
    None), home (x it is drawn to or None), zr (vertical scatter), amin / amax (overlap with the others)."""
    placed = []
    for it in items:
        ca, sa = math.cos(math.radians(it["tilt"])), math.sin(math.radians(it["tilt"]))
        it["ca"], it["sa"] = ca, sa
        gx = np.linspace(-0.47, 0.47, 9) * it["w"]
        gz = np.linspace(-0.47, 0.47, 11) * it["h"]
        L = np.array([(x, z) for z in gz for x in gx])
        it["R"] = np.stack([L[:, 0] * ca - L[:, 1] * sa, L[:, 0] * sa + L[:, 1] * ca], axis=1)
        if it.get("key"):
            kx0, kx1, kz0, kz1 = it["key"]
            it["keym"] = (L[:, 0] >= kx0 * it["w"]) & (L[:, 0] <= kx1 * it["w"]) & (L[:, 1] >= kz0 * it["h"]) & (L[:, 1] <= kz1 * it["h"])
        else:
            it["keym"] = None
        cr = np.array([(sx * it["w"] / 2, sz * it["h"] / 2) for sx in (-1, 1) for sz in (-1, 1)])
        it["C"] = np.stack([cr[:, 0] * ca - cr[:, 1] * sa, cr[:, 0] * sa + cr[:, 1] * ca], axis=1)

    def inside(P, it):
        dx, dz = P[:, 0] - it["cx"], P[:, 1] - it["cz"]
        return (np.abs(dx * it["ca"] + dz * it["sa"]) < it["w"] / 2) & (np.abs(-dx * it["sa"] + dz * it["ca"]) < it["h"] / 2)

    def cost(cur, it):
        c = 0.0
        for a in cur:
            P = a["R"] + (a["cx"], a["cz"])
            cov = np.zeros(len(P), dtype=bool)
            for b in cur:
                if b["layer"] > a["layer"]:
                    cov |= inside(P, b)
            f = float(cov.mean())
            c += 40.0 * max(0.0, a["vmin"] - (1.0 - f))
            if a["keym"] is not None:
                c += a["kw"] * float(cov[a["keym"]].mean())
            if a["target"] is not None:
                c += 14.0 * abs(f - a["target"])
        cs = np.concatenate([a["C"] + (a["cx"], a["cz"]) for a in cur])
        c += 100.0 * max(0.0, float(cs[:, 0].max() - cs[:, 0].min()) - spread) + 40.0 * max(0.0, float(cs[:, 1].max() - cs[:, 1].min()) - hmax)
        P = it["R"] + (it["cx"], it["cz"])
        ov = np.zeros(len(P), dtype=bool)
        for b in cur:
            if b is not it:
                ov |= inside(P, b)
                if abs(it["cx"] - b["cx"]) < 0.4 * (it["w"] + b["w"]):
                    c += 4.0 * max(0.0, 0.7 * it["zr"] - abs(it["cz"] - b["cz"])) / (0.7 * it["zr"])   # no shared mid-line
        at = float(ov.mean())
        c += 30.0 * max(0.0, it["amin"] - at) / it["amin"] + 9.0 * max(0.0, at - it["amax"])
        if it["home"] is not None:
            c += 5.0 * abs(it["cx"] - it["home"])
        return c

    for it in items:
        if not placed:
            it["cx"], it["cz"] = it["home"] or 0.0, 0.0
            placed.append(it)
            continue
        cs = np.concatenate([a["C"] + (a["cx"], a["cz"]) for a in placed])
        bx0, bx1, bz0, bz1 = cs[:, 0].min(), cs[:, 0].max(), cs[:, 1].min(), cs[:, 1].max()
        best = None
        cur = placed + [it]
        for _ in range(tries):
            if it["home"] is not None:
                it["cx"], it["cz"] = it["home"] + r.u(-0.16, 0.16), r.u(-it["zr"], it["zr"])
            else:
                it["cx"], it["cz"] = r.u(bx0, bx1), r.u(bz0 - 0.02, bz1 + 0.02)
            c = cost(cur, it)
            if best is None or c < best[0]:
                best = (c, it["cx"], it["cz"])
        it["cx"], it["cz"] = best[1], best[2]
        it["cost"] = best[0]
        placed.append(it)
    return placed


# ------------------------------------------------------------------------------------------------ print layouts
FINE = ("ENQUIRE WITHIN", "AT THE TOWN HALL", "BY THE OLD GATE", "APPLY AT THE INN", "PAID IN COIN", "EVERY EVENING",
        "ASK FOR THE CLERK", "ON MARKET SQUARE", "BEFORE THE NINTH BELL")
FINE_PROC = ("BY ORDER OF THE COUNCIL", "GIVEN AT THE TOWN HALL", "LET IT BE KNOWN TO ALL")


class Ink:
    def __init__(self):
        self.polys = {}      # key -> [(polygon, rub)]  rub: 0 never rubbed off, 1 small greeked print, 2 small real type (only on wrecked bills)

    def add(self, key, polys, rub=False):
        if polys and isinstance(polys[0], tuple):
            polys = [polys]
        self.polys.setdefault(key, []).extend((q, rub) for q in polys)


def _greek_line(ink, key, x0, x1, z, hh, rng, fill=1.0):
    """A line of small text as word-length bars with word gaps and a ragged right edge."""
    xe = x0 + (x1 - x0) * fill * rng.u(0.86, 1.0)
    x = x0
    while x < xe - hh:
        wl = min(hh * rng.u(2.0, 7.5), xe - x)
        ink.add(key, _rect(x, z - hh, x + wl, z), rub=True)
        x += wl + hh * 1.0


def _headline(ink, key, body, x0, x1, z0, z1, font, res=3, maxlines=3, rub=False, align="c"):
    """Fit the text into the box as 1-3 justified lines (each line scaled to the measure)."""
    words = [w for w in str(body).split() if w]
    if not words:
        return False
    W, H = x1 - x0, z1 - z0
    best = None
    n = len(words)
    cuts = [()]
    if maxlines >= 2:
        cuts += [(i,) for i in range(1, n)]
    if maxlines >= 3:
        cuts += [(i, j) for i in range(1, n) for j in range(i + 1, n)]
    for c in cuts:
        idx = (0,) + c + (n,)
        lines = [" ".join(words[idx[i]:idx[i + 1]]) for i in range(len(idx) - 1)]
        sz = min(W / (0.74 * max(len(l) for l in lines)), H / (1.32 * len(lines) - 0.32))
        bal = sz - 0.002 * (max(len(l) for l in lines) - min(len(l) for l in lines))
        if best is None or bal > best[0]:
            best = (bal, sz, lines)
    _, sz, lines = best
    built = []
    for l in lines:
        tris = _text_tris(l, font, res)
        if not tris:
            continue
        xs = [q[0] for t in tris for q in t]
        zs = [q[1] for t in tris for q in t]
        bw, bh = max(xs) - min(xs), max(zs) - min(zs)
        s = min(W / bw, sz * 1.3 / max(bh, 1e-6))
        built.append((tris, s, (max(xs) + min(xs)) / 2, min(zs), bh * s, bw))
    if not built:
        return False
    gap = 0.3 * sz
    tot = sum(b[4] for b in built) + gap * (len(built) - 1)
    if tot > H:
        k = H / tot
        built = [(t, s * k, c, zb, bh * k, bw) for (t, s, c, zb, bh, bw) in built]
        gap *= k
        tot = H
    z = (z0 + z1) / 2 + tot / 2
    for (tris, s, c, zb, bh, bw) in built:
        z -= bh
        xc = (x0 + x1) / 2 if align == "c" else x0 + bw * s / 2
        ink.add(key, [[(xc + (px - c) * s, z + (pz - zb) * s) for (px, pz) in t] for t in tris], rub=rub)
        z -= gap
    return True


def _body(ink, key, x0, x1, zt, zb, k, rng, font, phrase, centre=False, fh=0.0125, lh=0.0075, pitch=0.0175, rows=2):
    """Small print: one line of real small type, then at most two greeked lines.  Returns the z below it."""
    z = zt
    if phrase and z - fh * k >= zb - 1e-6:
        _headline(ink, key, phrase, x0, x1, z - fh * k, z, font, res=2, maxlines=1, rub=2, align="c" if centre else "l")
        z -= fh * k + (pitch - lh) * k
    q = 0
    while q < rows and z - lh * k >= zb - 1e-6:
        if q == 0:
            _greek_line(ink, key, x0, x1, z, lh * k, rng)
        else:
            ww = (x1 - x0) * rng.u(0.4, 0.7)
            xa = (x0 + x1 - ww) / 2 if centre else x0
            _greek_line(ink, key, xa, xa + ww, z, lh * k, rng)
        z -= pitch * k
        q += 1
    return z


def _layout(sh, ink, p, rng, emblem, words):
    w, h = sh.w, sh.h
    fine = lambda: rng.pick(list(FINE))
    k = w / 0.40
    kind = sh.kind
    font = sh.font
    if kind == "proclaim":
        m = 0.055 * w
        x0, x1, z0, z1 = -w / 2 + m, w / 2 - m, -h / 2 + m, h / 2 - m
        ink.add("ink", _frame(x0, z0, x1, z1, 0.0048 * k))
        ink.add("ink", _frame(x0 + 0.0105 * k, z0 + 0.0105 * k, x1 - 0.0105 * k, z1 - 0.0105 * k, 0.0016 * k))
        px0, px1 = x0 + 0.03 * k, x1 - 0.03 * k
        zt = z1 - 0.03 * k
        hh = 0.25 * h
        _headline(ink, "ink", p.headline, px0, px1, zt - hh, zt, "serif")
        z = zt - hh - 0.016 * k
        d = 0.009 * k
        ink.add("ink", [_rect(px0, z - 0.0012 * k, -d * 1.6, z + 0.0012 * k), _rect(d * 1.6, z - 0.0012 * k, px1, z + 0.0012 * k),
                        [(-d, z), (0.0, z - d), (d, z), (0.0, z + d)]])
        es = min(0.35 * w, 0.25 * h)
        z -= 0.02 * k + es / 2
        ink.add("accent", _emblem(emblem, 0.0, z, es))
        z -= es / 2 + 0.018 * k
        ink.add("ink", _rect(px0 + 0.04 * k, z - 0.0016 * k, px1 - 0.04 * k, z))
        z -= 0.015 * k
        _body(ink, "ink", px0 + 0.015 * k, px1 - 0.015 * k, z, z0 + 0.088 * k, k, rng, "serif", rng.pick([f for f in FINE_PROC if f.split()[0] != (str(p.headline).upper().split() or [""])[0]]), centre=True,
              fh=0.0125, lh=0.0072, pitch=0.0165)
        sx, sz, sr = px1 - 0.035 * k, z0 + 0.064 * k, 0.02 * k
        for q in (-1, 1):                                                 # ribbon tails under the seal
            tail = [(sx + q * 0.55 * sr, sz - 0.7 * sr), (sx + q * 1.0 * sr, sz - 2.0 * sr), (sx + q * 0.4 * sr, sz - 1.75 * sr),
                    (sx + q * 0.05 * sr, sz - 0.9 * sr)]
            ink.add("accent", _tess(tail))
        ink.add("accent", _seal(sx, sz, sr, rng))
        ink.add("ink", _rect(px0, z0 + 0.03 * k, px0 + 0.11 * k, z0 + 0.0315 * k))       # signature line
    elif kind == "banner":
        m = 0.075 * w
        x0, x1, z0, z1 = -w / 2 + m, w / 2 - m, -h / 2 + m, h / 2 - m
        for (za, sg) in ((z1, -1.0), (z0, 1.0)):                          # head and foot rules (no side frame)
            ink.add("pale", _rect(x0, min(za, za + sg * 0.0042 * k), x1, max(za, za + sg * 0.0042 * k)))
            ink.add("pale", _rect(x0, min(za + sg * 0.0085 * k, za + sg * 0.0098 * k), x1, max(za + sg * 0.0085 * k, za + sg * 0.0098 * k)))
        px0, px1 = x0 + 0.02 * k, x1 - 0.02 * k
        es = 0.25 * w
        zt = z1 - 0.04 * k
        ink.add("pale", _emblem(emblem, 0.0, zt - es / 2, es))
        za = zt - es - 0.03 * k
        zb = z0 + 0.19 * h
        _headline(ink, "pale", p.subline, px0, px1, zb, za, "roman")
        zr = zb - 0.028 * k
        ink.add("pale", _rect(px0, zr - 0.002 * k, px1, zr))
        e2 = 0.12 * w
        zc = (zr + z0 + 0.0098 * k) / 2 - 0.002 * k
        ink.add("pale", _emblem(emblem, px0 + e2 * 0.55, zc, e2))
        ink.add("pale", _emblem(emblem, px1 - e2 * 0.55, zc, e2))
        _greek_line(ink, "pale", -0.22 * (px1 - px0), 0.22 * (px1 - px0), zc + 0.012 * k, 0.008 * k, rng)
        _greek_line(ink, "pale", -0.15 * (px1 - px0), 0.15 * (px1 - px0), zc - 0.006 * k, 0.008 * k, rng)
    elif kind == "bill":
        m = 0.07 * w
        x0, x1, z0, z1 = -w / 2 + m, w / 2 - m, -h / 2 + m, h / 2 - m
        bh = 0.05 * h
        ink.add("accent", _rect(x0, z1 - bh, x1, z1))
        z = z1 - bh - 0.03 * k
        dh = 0.17 * h
        _headline(ink, "ink", words(), x0, x1, z - dh, z, font, res=2, maxlines=2)
        z -= dh + 0.03 * k
        fs = min(0.5 * w, 0.3 * h)
        ink.add("ink", _frame(-fs / 2, z - fs, fs / 2, z, 0.004 * k))
        ink.add("accent", _emblem(emblem, 0.0, z - fs / 2, fs * 0.66))
        for sx in (-1, 1):   # printer's ornaments either side of the box
            xa, xb = (x0, -fs / 2 - 0.02 * k) if sx < 0 else (fs / 2 + 0.02 * k, x1)
            if xb - xa > 0.03:
                xm, zm, d = (xa + xb) / 2, z - fs / 2, min(0.012 * k, 0.3 * (xb - xa))
                ink.add("ink", [[(xm - d, zm), (xm, zm - d), (xm + d, zm), (xm, zm + d)]])
                for q in (-1, 1):
                    ink.add("ink", _rect(xa, zm + q * 2.6 * d - 0.0008 * k, xb, zm + q * 2.6 * d + 0.0008 * k))
        z -= fs + 0.028 * k
        zf = z0 + 0.1 * h
        _body(ink, "ink", x0, x1, z, zf + 0.014 * k, k, rng, font, fine(), centre=True, fh=0.0135, lh=0.008, pitch=0.021)
        ink.add("ink", _rect(x0, zf, x1, zf + 0.002 * k))
        _headline(ink, "accent", words(), x0 + 0.1 * (x1 - x0), x1 - 0.1 * (x1 - x0), z0, zf - 0.012 * k, font, res=2, maxlines=1)
    elif kind == "notice":
        m = 0.08 * w
        x0, x1, z0, z1 = -w / 2 + m, w / 2 - m, -h / 2 + m, h / 2 - m
        if sh.tab_h:
            z0 = -h / 2 + sh.tab_h + 0.25 * m
            ink.add("ink", _rect(-w / 2 + 0.01, -h / 2 + sh.tab_h + 0.004, w / 2 - 0.01, -h / 2 + sh.tab_h + 0.0052))
            for (ta, tb_) in sh.tab_ranges:
                tc = (ta + tb_) / 2
                ink.add("ink", _rect(tc - 0.0052, -h / 2 + 0.009, tc - 0.0012, -h / 2 + sh.tab_h - 0.008))
                ink.add("ink", _rect(tc + 0.002, -h / 2 + 0.016, tc + 0.0052, -h / 2 + sh.tab_h - 0.014))
        th = (0.26 if sh.wide else 0.15) * h
        _headline(ink, "ink", words(), x0, x1 if not sh.wide else x0 + 0.62 * (x1 - x0), z1 - th, z1, font, res=2, maxlines=1 if sh.wide else 2)
        z = z1 - th - 0.024 * k
        ink.add("accent", _rect(x0, z - 0.004 * k, x1, z))
        ink.add("ink", _rect(x0, z - 0.008 * k, x1, z - 0.0068 * k))
        z -= 0.026 * k
        zs = z0 + 0.1 * h                                                 # the seal's row
        extra = (z - zs) - (0.015 + 0.014 + 0.0225 + 0.0085) * k
        if extra > 0.05 * k:                                              # room for a second display line
            dh = min(extra - 0.022 * k, 0.17 * h)
            _headline(ink, "accent", words(), x0, x1, z - dh, z, font, res=2, maxlines=1)
            z -= dh + 0.022 * k
        _body(ink, "ink", x0, x1, z, zs, k, rng, font, fine(), fh=0.015, lh=0.0085, pitch=0.0225)
        ink.add("accent", _seal(x1 - 0.045 * k, z0 + 0.05 * h, 0.028 * k, rng, 20))
        ink.add("ink", _rect(x0, z0 + 0.035 * h, x0 + (x1 - x0) * 0.42, z0 + 0.035 * h + 0.0013 * k))   # signature line
    elif kind == "playbill":   # stacked lines of display type between rules
        m = 0.08 * w
        x0, x1, z0, z1 = -w / 2 + m, w / 2 - m, -h / 2 + m, h / 2 - m
        ink.add("ink", _frame(x0, z0, x1, z1, 0.003 * k))
        px0, px1 = x0 + 0.03 * k, x1 - 0.03 * k
        pw = px1 - px0
        z = z1 - 0.035 * k
        _greek_line(ink, "ink", px0 + pw * 0.18, px1 - pw * 0.18, z, 0.008 * k, rng)
        z -= 0.03 * k
        dh = 0.2 * h
        _headline(ink, "ink", words(), px0, px1, z - dh, z, font, res=2, maxlines=2)
        z -= dh + 0.028 * k
        ink.add("ink", _rect(px0, z - 0.004 * k, px1, z))
        z -= 0.032 * k
        es = min(0.5 * w, 0.17 * h)
        ink.add("accent", _emblem(emblem, 0.0, z - es / 2, es))
        z -= es + 0.03 * k
        dh = 0.13 * h
        _headline(ink, "accent", words(), px0, px1, z - dh, z, font, res=2, maxlines=2)
        z -= dh + 0.028 * k
        ink.add("ink", _rect(px0 + pw * 0.15, z - 0.0018 * k, px1 - pw * 0.15, z))
        z -= 0.03 * k
        zend = z0 + 0.03 * k
        extra = (z - zend) - (0.03 + 0.016 + 0.009) * k
        if extra > 0.07 * h:                                              # a third line of display type
            dh = min(extra - 0.034 * k, 0.11 * h)
            _headline(ink, "ink", words(), px0, px1, z - dh, z, font, res=2, maxlines=1)
            z -= dh + 0.02 * k
            ink.add("ink", _rect(px0, z - 0.003 * k, px1, z))
            z -= 0.026 * k
        _body(ink, "ink", px0, px1, z, zend, k, rng, font, fine(), centre=True, fh=0.03, lh=0.009, pitch=0.025, rows=1)
    else:  # remnant: the head of a bill that was ripped off under its tacks
        m = 0.06 * w
        x0, x1, z1 = -w / 2 + m, w / 2 - m, h / 2 - 0.022
        ink.add("ink", _rect(x0, z1 - 0.003, x1, z1))
        dh = min(0.05, 0.36 * h)
        _headline(ink, "ink", words(), x0 + 0.05 * w, x1 - 0.05 * w, z1 - 0.012 - dh, z1 - 0.012, font, res=2, maxlines=1)
        z = z1 - 0.03 - dh
        for q in range(2):
            if z > -h / 2:
                _greek_line(ink, "ink", x0, x1, z, 0.0065, rng)
            z -= 0.017


# ------------------------------------------------------------------------------------------------ build
def build(p, rng):
    T, C = float(p.tattered), float(p.curl)
    paper_hex, stock_hex = PALETTES[p.palette]
    n = int(p.count)
    H = float(p.size)

    # ---- which bills, how big, in which order they were posted
    nominal = {"proclaim": (0.70, 1.0), "banner": (0.60, 0.86), "bill": (0.52, 0.70), "notice": (0.42, 0.50),
               "playbill": (0.37, 0.92), "notice_wide": (0.58, 0.40)}
    r = rng.fork("pick")
    others = sorted(["bill", "notice", "playbill", "notice_wide"], key=lambda _: r.u(0, 1))
    kinds = (["proclaim", "banner"] + others)[:n]
    late = kinds.index("notice") if "notice" in kinds else (kinds.index("notice_wide") if (n >= 5 and "notice_wide" in kinds) else None)
    rest = [i for i in range(2, n) if i != late]
    buried = rest[0] if rest else None
    order = rest + [1, 0] + ([late] if late is not None else [])          # bottom -> top
    layer_of = {i: q for q, i in enumerate(order)}
    mid = n // 2 if n != 2 else r.pick([0, 1])
    rest_slots = sorted([s for s in range(n) if s != mid], key=lambda _: r.u(0, 1))
    slot_of = {0: mid}
    for i, s in zip(range(1, n), rest_slots):
        slot_of[i] = s
    n_pin = int(round(float(p.pinned) * n))
    pin_rank = sorted(range(n), key=lambda i: (0 if i == late else 1, nominal[kinds[i]][0] * nominal[kinds[i]][1], i))
    pinned_set = set(pin_rank[:n_pin])
    dims, rss = {}, {}
    for i in range(n):
        rss[i] = rng.fork(f"sheet{i}")
        dims[i] = (nominal[kinds[i]][0] * H * rss[i].u(0.95, 1.05), nominal[kinds[i]][1] * H * rss[i].u(0.96, 1.04))
    phrases = [s.strip() for s in str(p.notices).split(",") if s.strip()] or ["NOTICE"]
    phrases = sorted(phrases, key=lambda _: r.u(0, 1))
    pcount = [0]

    def words():
        pcount[0] += 1
        return phrases[(pcount[0] - 1) % len(phrases)]

    # ---- arrangement: a left-to-right run of 'home' columns squeezed into a cluster, then every bill is
    # dropped near its column at a staggered height and tilt so that the overlaps come out as wanted
    seq = sorted(range(n), key=lambda i: slot_of[i])
    sumw = sum(dims[i][0] for i in range(n))
    S = min(float(p.spread), 0.6 * sumw + 0.08)
    want = (sumw - S) / (n - 1)
    xl, x = {}, 0.0
    for q, i in enumerate(seq):
        if q > 0:
            j = seq[q - 1]
            x -= max(0.03, min(want, 0.5 * min(dims[i][0], dims[j][0])))
        xl[i] = x
        x += dims[i][0]
    home = {i: xl[i] + dims[i][0] / 2 - x / 2 for i in range(n)}
    for i in range(n):                                                  # a narrow spread squeezes the columns together
        lim = max(float(p.spread) / 2 - dims[i][0] / 2 - 0.01, 0.0)
        if abs(home[i]) > lim:
            home[i] = math.copysign(lim, home[i])
    zr = min(0.24, 0.13 + 0.035 * (n - 2)) * H / 0.58
    ra = rng.fork("arrange")
    items = {}
    for i in range(n):
        w, h = dims[i]
        rs = rss[i]
        pinned = i in pinned_set
        pin_style = rs.weighted([("top2", 3), ("four", 2), ("top1", 1.5 if w < 0.3 else 0.0)]) if pinned else None
        sgn = rs.pick([-1, 1])
        if i == 0:
            tilt = sgn * rs.u(1.5, 4.0)
        elif i == 1:
            tilt = sgn * rs.u(3.5, 7.0)
        else:
            tilt = sgn * (rs.u(4.0, 9.0) if pin_style != "top1" else rs.u(6.0, 11.0))
        it = {"i": i, "w": w, "h": h, "tilt": tilt, "layer": layer_of[i], "home": home[i], "zr": zr, "pin_style": pin_style,
              "vmin": 0.55, "key": (-0.46, 0.46, 0.12, 0.47), "kw": 8.0, "target": None, "amin": 0.1, "amax": 0.4}
        if i == 0:
            it.update(vmin=0.86, key=(-0.43, 0.43, -0.15, 0.47), kw=70.0, amax=0.5)
        elif i == 1:
            it.update(vmin=0.62, key=(-0.4, 0.4, -0.24, 0.2), kw=30.0, amin=0.2)
            if buried is None:                                           # few bills: the banner is the one pasted over
                it.update(vmin=0.55, kw=6.0, target=0.36, amin=0.3, amax=0.6)
        if i == buried:
            it.update(vmin=0.42, kw=4.0, target=0.5, amax=0.7, amin=0.3)
        if i == late:
            it.update(amin=0.2, amax=0.75)
        items[i] = it
    rem = []
    n_rem = 0 if T < 0.4 else 1
    for q in range(n_rem):
        rs = rng.fork(f"remnant{q}")
        rem.append({"i": -1 - q, "w": H * rs.u(0.4, 0.54), "h": rs.u(0.13, 0.18) * (0.6 + 0.4 * H / 0.58),
                    "tilt": rs.pick([-1, 1]) * rs.u(3.0, 8.0), "layer": -1 - q, "home": None, "zr": zr, "pin_style": "top2",
                    "vmin": 0.6, "key": (-0.5, 0.5, 0.05, 0.5), "kw": 30.0, "target": None, "amin": 0.14, "amax": 0.4, "free_w": True, "rs": rs})
    placed = _arrange([items[i] for i in range(n)] + rem, float(p.spread), H * 1.04 + 1.8 * zr, ra)

    sheets = []
    for it in sorted(placed, key=lambda a: a["layer"]):
        i = it["i"]
        if i >= 0:
            kd, rs = kinds[i], rss[i]
        else:
            kd, rs = "remnant", it["rs"]
        sh = Sheet(len(sheets), kd.replace("_wide", ""), it["w"], it["h"])
        sh.idx = i
        sh.wide = kd.endswith("_wide")
        sh.pin_style = it["pin_style"]
        sh.pinned = sh.pin_style is not None
        sh.set_pose(it["cx"], it["cz"], it["tilt"])
        sh.below = list(sheets)
        sh.rs = rs
        sh.font = "serif" if i == 0 else rs.pick(["sans_narrow", "serif", "roman", "sans", "sans_narrow"])
        sheets.append(sh)
    for sh in sheets:
        sh.uppers = sheets[sh.k + 1:]
    # centre the group on the fixing point
    cor = [sh.world(sx * sh.w / 2, sz * sh.h / 2) for sh in sheets for sx in (-1, 1) for sz in (-1, 1)]
    ox = (min(c[0] for c in cor) + max(c[0] for c in cor)) / 2
    oz = (min(c[1] for c in cor) + max(c[1] for c in cor)) / 2
    for sh in sheets:
        sh.cx -= ox
        sh.cz -= oz
    ext = (min(c[0] for c in cor) - ox, max(c[0] for c in cor) - ox, min(c[1] for c in cor) - oz, max(c[1] for c in cor) - oz)

    # paper stock of every sheet: the banner is printed on coloured stock, the others are cream / bleached / straw
    tones = sorted(["white", "straw", "cream"], key=lambda _: r.u(0, 1))
    tq = 0
    for sh in sorted(sheets, key=lambda s: s.idx):
        if sh.idx == 0:
            sh.tone = "cream"
        elif sh.idx == 1 or (p.palette == "festival" and sh.kind == "playbill"):
            sh.tone = "stock"
        else:
            sh.tone = tones[tq] if tq < 3 else ("cream", "white")[tq % 2]
            tq += 1

    def covered_near(sh, x, z, margin=0.03):
        X, Z = sh.world(x, z)
        for up in sh.uppers:
            ux, uz = up.local(X, Z)
            if up.rect_dist(ux, uz) < margin:
                return True
        return False

    def zone_blocked(sh, cx, cz, nx, nz, L, span=1.4):
        """Is the zone within L of the fold line touched by a sheet lying on top?"""
        tx, tz = -nz, nx
        pts = [(cx, cz), (cx + nx * L * 0.5, cz + nz * L * 0.5)]
        for s in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for f in (0.0, 1.0):
                px, pz = cx + nx * L * f + tx * L * s * span, cz + nz * L * f + tz * L * s * span
                pts.append((max(-sh.w / 2, min(sh.w / 2, px)), max(-sh.h / 2, min(sh.h / 2, pz))))
        if any(covered_near(sh, x, z) for (x, z) in pts):
            return True
        for up in sh.uppers:
            for sx in (-1, 1):
                for sz in (-1, 1):
                    X, Z = up.world(sx * up.w / 2, sz * up.h / 2)
                    x, z = sh.local(X, Z)
                    if sh.rect_dist(x, z) < 0.02 and L - ((x - cx) * nx + (z - cz) * nz) > -0.03:
                        return True
        return False

    # ---- per-sheet features: fixings, tears, curls, wrinkles, creases
    def jag(rs, amp, wl):
        return [(amp * rs.u(0.5, 1.0) / (i + 1), 2 * math.pi / (wl * rs.u(0.7, 1.3)) * (i * 1.7 + 1), rs.u(0, 6.28)) for i in range(3)]

    n_feat = 0
    for sh in sheets:
        rs = sh.rs.fork("feat")
        w, h = sh.w, sh.h
        ks = w / 0.40
        corners = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        pin_corners = set()
        if sh.pin_style == "top2":
            pin_corners = {(-1, 1), (1, 1)}
        elif sh.pin_style == "four":
            pin_corners = set(corners)
        for (sx, sz) in sorted(pin_corners):
            sh.pins.append((sx * (w / 2 - rs.u(0.013, 0.022)), sz * (h / 2 - rs.u(0.013, 0.022))))
        if sh.pin_style == "top1":
            sh.pins.append((rs.u(-0.02, 0.02), h / 2 - rs.u(0.014, 0.024)))
        if not sh.pinned and rs.chance(0.3):
            sh.pins.append((rs.u(-0.3, 0.3) * w, h / 2 - rs.u(0.015, 0.03)))      # a nail through a pasted bill
        sh.fixing = rs.pick(["tack", "nail"]) if sh.pinned else "nail"
        if sh.pinned:
            sh.belly = rs.u(0.008, 0.015) if sh.pin_style == "four" else rs.u(0.006, 0.011)
            sh.hang = rs.u(0.02, 0.03) * (0.6 + 0.5 * C) * min(1.0, h / 0.2)
            sh.skew = rs.u(-0.45, 0.45)
        if sh.kind == "remnant":
            sh.fixing = "tack"
            sh.belly, sh.hang = rs.u(0.003, 0.005), rs.u(0.008, 0.014)
            a = math.radians(rs.u(-7, 7))
            sh.tears.append(('line', 0.0, -h / 2, math.sin(a), math.cos(a), 0.36 * h, jag(rs, 0.2 * h, 0.085)))
            continue
        tabbed = bool(p.tabs) and sh.kind == "notice" and not sh.wide
        if tabbed:
            sh.tab_h = min(0.2 * h, 0.062)
            ntab = max(4, int(round(w / 0.027)))
            sh.slits = [-w / 2 + w * q / ntab for q in range(1, ntab)]
            edges_x = [-w / 2 - 1.0] + sh.slits + [w / 2 + 1.0]
            for q in range(ntab):
                ta, tb_ = edges_x[q] + 0.0012, edges_x[q + 1] - 0.0012
                if rs.chance(0.12 + 0.45 * T):
                    sh.tab_gone.add(q)                                      # somebody took the address
                    continue
                sh.tab_ranges.append((max(ta, -w / 2), min(tb_, w / 2)))
                if C > 0.02 and not covered_near(sh, (max(ta, -w / 2) + min(tb_, w / 2)) / 2, -h / 2, 0.035):
                    sh.curls.append((0.0, -h / 2, 0.0, 1.0, sh.tab_h * rs.u(0.5, 0.97), rs.u(0.025, 0.09),
                                     math.radians(rs.u(8, 25 + 50 * C)), (ta, tb_), None))
        # tears
        for (sx, sz) in corners:
            if (sx, sz) in pin_corners or (tabbed and sz < 0):
                continue
            if rs.chance(0.12 + 0.62 * T):
                a = math.radians(rs.u(28, 62))
                nx, nz = -sx * math.cos(a), -sz * math.sin(a)
                size = min(rs.u(0.012, 0.028 + 0.11 * T) * ks, 0.28 * min(w, h))
                sh.tears.append(('line', sx * w / 2, sz * h / 2, nx, nz, size, jag(rs, size * 0.2, max(0.02, size * 0.6))))
        for _ in range(int(T * 3.4 * rs.u(0.4, 1.0) + 0.3)):
            e = rs.pick([0, 1, 2, 3]) if not tabbed else rs.pick([0, 2, 3])
            t = rs.u(-0.36, 0.36) if not tabbed else rs.u(-0.1, 0.36)
            ex, ez = [(t * w, h / 2), (t * w, -h / 2), (-w / 2, t * h), (w / 2, t * h)][e]
            if any(math.hypot(ex - qx, ez - qz) < 0.07 for (qx, qz) in sh.pins):
                continue
            rr = rs.u(0.008, 0.012 + 0.034 * T) * ks
            sh.tears.append(('bite', ex, ez, rr, jag(rs, rr * 0.3, rr * 1.3)))
        if T > 0.5 and sh.pin_style != "four" and sh.idx not in (0, 1) and not tabbed and rs.chance((T - 0.35) * 1.2):
            a = math.radians(rs.u(-14, 14))                                   # the bottom strip ripped off
            hs = rs.u(0.08, 0.2) * h
            sh.tears.append(('line', 0.0, -h / 2, math.sin(a), math.cos(a), hs, jag(rs, 0.018 * ks, 0.07)))
        # curls: rolled corners ...
        for (sx, sz) in corners:
            if (sx, sz) in pin_corners or (sh.pin_style == "top1" and sz > 0) or (tabbed and sz < 0):
                continue
            ch = (0.42 + 0.58 * C) * (1.0 if sz < 0 else 0.7) + (0.2 if sh.pinned else 0.0)
            if C > 0.02 and rs.chance(ch):
                a = math.radians(rs.u(30, 60))
                nx, nz = -sx * math.cos(a), -sz * math.sin(a)
                thm = math.radians(rs.u(35, 55 + 85 * C))
                top_lift = 0.012 + 0.02 * C                                # a corner never stands further off the wall
                L = min(rs.u(0.035, 0.05 + 0.12 * C) * ks, 0.3 * min(w, h), top_lift * thm / (1.0 - math.cos(thm)))
                R = L / thm * rs.u(0.85, 1.0)
                if not zone_blocked(sh, sx * w / 2, sz * h / 2, nx, nz, L):
                    sh.curls.append((sx * w / 2, sz * h / 2, nx, nz, L, R, thm, None, None))
        # ... and a stretch of a pasted edge that has come away from the wall
        if not sh.pinned and not tabbed and C > 0.05:
            for _ in range(2):
                if not rs.chance(0.3 + 0.5 * C):
                    continue
                e = rs.weighted([(0, 3), (1, 1.5), (2, 1.5)])               # bottom, left, right
                ln = (w if e == 0 else h)
                tw = rs.u(0.25, 0.4) * ln
                t0 = rs.u(-0.5, 0.5) * (ln - 2 * tw) * 0.8
                cx, cz, nx, nz = [(0.0, -h / 2, 0.0, 1.0), (-w / 2, 0.0, 1.0, 0.0), (w / 2, 0.0, -1.0, 0.0)][e]
                L = min(rs.u(0.03, 0.04 + 0.05 * C), 0.22 * min(w, h))
                tx, tz = -nz, nx
                if any(abs(c[0] - cx) < 1e-6 and abs(c[1] - cz) < 1e-6 for c in sh.curls):
                    continue
                if zone_blocked(sh, cx + tx * t0, cz + tz * t0, nx, nz, L, span=tw / L):
                    continue
                if any(c[8] is None and c[7] is None and math.hypot(c[0] - (cx + tx * t0), c[1] - (cz + tz * t0)) < tw + c[4] * 1.5 for c in sh.curls):
                    continue                                              # keep clear of a rolled corner
                sh.curls.append((cx, cz, nx, nz, L, L * rs.u(0.9, 1.7), math.radians(rs.u(22, 30 + 30 * C)), None, (t0, tw)))
        # wrinkles: broad undulations, pinched ridges and paste bubbles on pasted sheets
        if not sh.pinned:
            sh.edge_amp = rs.u(0.002, 0.003) + 0.003 * C
            sh.eph = (rs.u(0, 6.28), rs.u(0, 6.28), rs.u(0, 6.28))
            for _ in range(2 + int(rs.u(0, 3))):
                a = math.radians(rs.u(-35, 35) + (90 if rs.chance(0.7) else 0))
                sh.ridges.append((rs.u(-0.4, 0.4) * w, rs.u(-0.4, 0.4) * h, math.cos(a), math.sin(a), rs.u(0.08, 0.22), rs.u(0.035, 0.055),
                                  rs.u(0.0012, 0.002 + 0.002 * T)))
            want_f = (2 if sh.idx == 0 else 1) if (n <= 4 or sh.idx in (0, 1)) else 0
            if min(w, h) > 0.24 and n_feat < 5:
                used_x, used_z = [], []
                for _ in range(want_f):
                    c = rs.u(-0.5, 0.5) * (w - 0.24)
                    if all(abs(c - u) > 0.05 for u in used_x):
                        used_x.append(c)
                        sh.pinch.append(('x', c, rs.u(-0.3, 0.3) * h, rs.u(0.06, 0.14), rs.u(0.0026, 0.0042), rs.u(0.007, 0.01)))
                        n_feat += 1
                for _ in range(want_f):
                    bx, bz = rs.u(-0.5, 0.5) * (w - 0.26), rs.u(-0.5, 0.5) * (h - 0.26)
                    if all(abs(bx - u) > 0.07 for u in used_x) and all(abs(bz - u) > 0.07 for u in used_z):
                        used_x.append(bx)
                        used_z.append(bz)
                        sh.bubbles.append((bx, bz, rs.u(0.02, 0.027), rs.u(0.0035, 0.005)))
                        n_feat += 1
        if sh.idx == 0 or rs.chance(0.45):
            cz_ = rs.u(-0.03, 0.03) * h
            if all(abs(cz_ - b[1]) > 0.05 for b in sh.bubbles):
                sh.creases_z.append(cz_)
            if sh.idx == 0 and rs.chance(0.6):
                cx_ = rs.u(-0.02, 0.02) * w
                if all(abs(cx_ - b[0]) > 0.05 for b in sh.bubbles) and all(abs(cx_ - q[1]) > 0.03 for q in sh.pinch):
                    sh.creases_x.append(cx_)


    # ---- materials: one paper, one coloured stock that is also the ink, one metal
    m_paper = mat("paper", paper_hex)
    m_ink = mat("paper", stock_hex)
    m_iron = mat("iron_wrought")

    # ---- meshes
    stats = {"paper": 0, "back": 0, "ink": 0, "text": 0}
    pin_mats = {"tack": [], "nail": []}
    for sh in sheets:
        rs = sh.rs.fork("mesh")
        w, h = sh.w, sh.h
        dark_paper = sh.tone == "stock"
        xs = _axis_nodes(w, sh.creases_x, [(q[1], q[5]) for q in sh.pinch if q[0] == 'x'] + [(b[0], b[2]) for b in sh.bubbles])
        zs = _axis_nodes(h, sh.creases_z, [(q[1], q[5]) for q in sh.pinch if q[0] == 'z'] + [(b[1], b[2]) for b in sh.bubbles])
        if sh.tab_h:
            for sl in sh.slits:
                xs = [v for v in xs if abs(v - sl) > 0.0045 or abs(abs(v) - w / 2) < 1e-6] + [sl - 0.0012, sl + 0.0012]
            zt = -h / 2 + sh.tab_h
            zs = [v for v in zs if abs(v - zt) > 0.007 or abs(abs(v) - h / 2) < 1e-6] + [zt]
            xs, zs = sorted(xs), sorted(zs)
        nx, nz = len(xs), len(zs)
        flat = []
        wear_v = []
        moved = set()
        for j in range(nz):
            for i in range(nx):
                x, z = xs[i], zs[j]
                wv = 0.0
                bx, bz = i in (0, nx - 1), j in (0, nz - 1)
                if bx or bz:
                    moved.add(len(flat))
                if (bx or bz) and not (bx and bz):
                    d = rs.u(0.0, 0.0022) * (0.3 + T)                       # deckle
                    if rs.chance(0.07 * T):
                        d = rs.u(0.004, 0.013)                              # nick
                        wv = 0.7
                    if bx:
                        x -= math.copysign(d, x)
                    else:
                        z -= math.copysign(d, z)
                if any(abs(x - c) < 1e-6 for c in sh.creases_x) or any(abs(z - c) < 1e-6 for c in sh.creases_z):
                    wv = max(wv, 0.62)
                flat.append([x, z])
                wear_v.append(wv)
        # tears: drop cells that are completely gone, pull the remaining rim onto the tear line
        fv = [sh.tear_f(x, z) for (x, z) in flat]
        quads = []
        for j in range(nz - 1):
            for i in range(nx - 1):
                q = (j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i)
                if sh.tab_h and zs[j + 1] <= -h / 2 + sh.tab_h + 1e-6:
                    xm = (xs[i] + xs[i + 1]) / 2
                    if any(abs(xm - sl) < 0.001 for sl in sh.slits):
                        continue                                            # the scissor cut
                    if int((xm + w / 2) / w * (len(sh.slits) + 1)) in sh.tab_gone:
                        continue
                if any(fv[v] > 0.0015 for v in q):
                    quads.append(q)
        if not quads:
            continue
        qbox = {q: (xs[q[0] % nx], xs[q[1] % nx], zs[q[0] // nx], zs[q[3] // nx]) for q in quads}

        def quad_area(q):
            b = qbox[q]
            return (b[1] - b[0]) * (b[3] - b[2])

        def holds(qx, qz, qs):
            return any(qbox[q][0] - 0.004 <= qx <= qbox[q][1] + 0.004 and qbox[q][2] - 0.004 <= qz <= qbox[q][3] + 0.004 for q in qs)

        def pin_in(qs):
            return any(holds(qx, qz, qs) for (qx, qz) in sh.pins)
        # a torn-off piece only stays on the wall if a tack still holds it
        comps = _groups(quads)
        if len(comps) > 1:
            comps.sort(key=lambda c: -sum(quad_area(quads[qi]) for qi in c))
            keep = set(comps[0])
            for c in comps[1:]:
                if sum(quad_area(quads[qi]) for qi in c) > 8e-4 and pin_in([quads[qi] for qi in c]):
                    keep |= set(c)
            quads = [q for qi, q in enumerate(quads) if qi in keep]
        used = sorted({v for q in quads for v in q})
        for v in used:
            if fv[v] < 0.0:
                x, z = flat[v]
                for _ in range(5):
                    f0 = sh.tear_f(x, z)
                    gx = (sh.tear_f(x + 2e-4, z) - sh.tear_f(x - 2e-4, z)) / 4e-4
                    gz = (sh.tear_f(x, z + 2e-4) - sh.tear_f(x, z - 2e-4)) / 4e-4
                    g2 = gx * gx + gz * gz
                    if g2 < 1e-8 or abs(f0) < 1e-5:
                        break
                    st = max(-0.03, min(0.03, -f0 / g2))
                    x += gx * st
                    z += gz * st
                flat[v] = [x, z]
                wear_v[v] = 0.95
                moved.add(v)
        # deform
        press = set()
        for up in sh.uppers:                                                # cells that reach under a later sheet
            ew, eh = up.w / 2 + 0.006, up.h / 2 + 0.006
            na, nb = int(2 * ew / 0.008) + 1, int(2 * eh / 0.008) + 1
            smp = [(-ew + 2 * ew * a / na, sg * eh) for a in range(na + 1) for sg in (-1, 1)]
            smp += [(sg * ew, -eh + 2 * eh * b / nb) for b in range(nb + 1) for sg in (-1, 1)]
            na, nb = int(2 * ew / 0.025) + 1, int(2 * eh / 0.025) + 1
            smp += [(-ew + 2 * ew * a / na, -eh + 2 * eh * b / nb) for a in range(1, na) for b in range(1, nb)]
            for (ux, uz) in smp:
                x, z = sh.local(*up.world(ux, uz))
                if abs(x) > w / 2 + 0.004 or abs(z) > h / 2 + 0.004:
                    continue
                i = min(max(bisect.bisect_right(xs, x) - 1, 0), nx - 2)
                j = min(max(bisect.bisect_right(zs, z) - 1, 0), nz - 2)
                press.update((j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i))
        pos, lift, loose, opn = {}, {}, {}, {}
        for v in used:
            pos[v], lift[v], loose[v] = sh.deform(flat[v][0], flat[v][1], v in press)
            opn[v] = 0.0 if v in press else sh.open_f(flat[v][0], flat[v][1])
        # triangulate: along the contour of a curl; in flat areas along the 'valley' diagonal, so that a
        # bilinear patch over the quad (used for the print, see below) never dips under the paper
        tris, tri_q, bil = [], [], set()
        for qi, (a, b, c, d) in enumerate(quads):
            if not (lift[a] or lift[b] or lift[c] or lift[d]) and not (a in moved or b in moved or c in moved or d in moved):
                pa, pb, pc, pd = pos[a], pos[b], pos[c], pos[d]
                tw = math.sqrt(sum((pa[i] + pc[i] - pb[i] - pd[i]) ** 2 for i in range(3))) / 4
                ac = pa[1] + pc[1] >= pb[1] + pd[1]
                if tw < 0.0006:
                    bil.add(qi)
            else:
                ac = abs(lift[a] - lift[c]) <= abs(lift[b] - lift[d])
            for t in ([(a, b, c), (a, c, d)] if ac else [(a, b, d), (b, c, d)]):
                if _area([flat[v] for v in t]) > 2e-7:
                    tris.append(t)
                    tri_q.append(qi)
        # hidden-face removal: triangles completely under an upper sheet
        ups = sh.uppers
        if ups:
            keep, keep_q = [], []
            for t, qi in zip(tris, tri_q):
                pts = [sh.world(*flat[v]) for v in t]
                pts.append((sum(q[0] for q in pts) / 3, sum(q[1] for q in pts) / 3))
                pts += [((pts[i][0] + pts[(i + 1) % 3][0]) / 2, (pts[i][1] + pts[(i + 1) % 3][1]) / 2) for i in range(3)]
                if not any(all(up.solid(X, Z) for (X, Z) in pts) for up in ups):
                    keep.append(t)
                    keep_q.append(qi)
            tris, tri_q = keep, keep_q
            # what still shows must read as part of a bill: slivers and specks peeking out between the
            # sheets on top are dropped unless a tack sits on them
            comps = _groups(tris)
            keep = set()
            for c in comps:
                ar = sum(_area([flat[v] for v in tris[ti]]) for ti in c)
                vs = [flat[v] for ti in c for v in tris[ti]]
                dg = math.hypot(max(q[0] for q in vs) - min(q[0] for q in vs), max(q[1] for q in vs) - min(q[1] for q in vs))
                if (ar > max(30e-4, 0.12 * w * h) and ar / max(dg, 1e-6) > 0.02) or (ar > 6e-4 and pin_in([quads[tri_q[ti]] for ti in c])):
                    keep |= set(c)
            tris = [t for ti, t in enumerate(tris) if ti in keep]
            tri_q = [q for ti, q in enumerate(tri_q) if ti in keep]
        if not tris:
            continue
        kept_q = [quads[qi] for qi in sorted(set(tri_q))]
        used = sorted({v for t in tris for v in t})
        remap = {v: i for i, v in enumerate(used)}
        P = np.array([pos[v] for v in used], dtype=np.float64)
        F2 = np.array([flat[v] for v in used], dtype=np.float64)
        TI = np.array([[remap[v] for v in t] for t in tris], dtype=np.int64)
        fnrm = np.cross(P[TI[:, 1]] - P[TI[:, 0]], P[TI[:, 2]] - P[TI[:, 0]])
        N = np.zeros_like(P)
        for c in range(3):
            np.add.at(N, TI[:, c], fnrm)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        uo = (rs.u(0, 0.5), rs.u(0, 0.5))
        uvs = [(float(F2[i, 0]) + uo[0] + w, float(F2[i, 1]) + uo[1] + h) for i in range(len(used))]
        paper = _mesh(f"paper{sh.k}", P.tolist(), TI.tolist(), uvs)
        assign(paper, m_ink if dark_paper else m_paper)
        stats["paper"] += len(tris)

        # art-directed dirt: rain streaks from the top edge, grubby edges, damp blotches low down, pale fibres at
        # tears / folds.  The masks switch the look near 0.5, so dirt is kept in the 'patchy' range and off
        # everything that stands away from the wall (lifted paper stays pale and catches the light).
        ph = [rs.u(0, 6.28) for _ in range(10)]

        def blot_f(x, z):
            return 0.5 + 0.25 * (np.sin(x * 17.0 + ph[2]) * np.sin(z * 13.0 + ph[3]) + np.sin((x + z) * 9.0 + ph[4]) * np.sin((x - z) * 11.0 + ph[5]))

        def rub_f(x, z):
            return 0.5 + 0.25 * (np.sin(x * 37.0 + ph[6]) * np.sin(z * 29.0 + ph[7]) + np.sin((x + z) * 21.0 + ph[8]) * np.sin((x - z) * 25.0 + ph[9]))

        def crease_f(x, z):
            v = np.zeros_like(x)
            for c in sh.creases_x:
                v = np.maximum(v, np.exp(-np.abs(x - c) / 0.012))
            for c in sh.creases_z:
                v = np.maximum(v, np.exp(-np.abs(z - c) / 0.012))
            return v

        def fade_f(x, z):
            """0..1: how far the print has faded / rubbed off here."""
            edge = np.minimum(w / 2 - np.abs(x), h / 2 - np.abs(z))
            return np.clip(1.25 * T - 0.22 + (rub_f(x, z) - 0.5) * 1.3 * (0.35 + 0.65 * T) + 0.5 * T * np.exp(-np.maximum(edge, 0) / 0.03) +
                           0.45 * T * crease_f(x, z), 0, 1)

        Lc = np.array([lift[v] for v in used])
        Lo = np.array([loose[v] for v in used])
        xn, zn = F2[:, 0], (F2[:, 1] + h / 2) / h
        edge = np.minimum(w / 2 - np.abs(F2[:, 0]), h / 2 - np.abs(F2[:, 1]))
        streak = np.clip(np.sin(xn * 53.0 + ph[0]) * np.sin(xn * 21.0 + ph[1]) * 1.6 - 0.25, 0, 1) * (0.35 + 0.65 * zn)
        blot = blot_f(F2[:, 0], F2[:, 1])
        up_f = 1.0 - np.clip(Lc / 0.003, 0, 1)
        g_arr = np.clip(0.02 + (0.1 + 0.2 * T) * np.exp(-np.maximum(edge, 0) / 0.03) + (0.06 + 0.2 * T) * streak + 0.08 * T * blot, 0, 0.47)
        g_arr *= up_f * (1.0 - 0.5 * np.clip(Lo / 0.01, 0, 1))
        s_arr = np.clip((blot - 0.52) * 2.6, 0, 1) * (0.1 + 0.55 * T) * (1.0 - 0.6 * zn) * up_f
        w_arr = np.clip(np.array([wear_v[v] for v in used]) * (0.5 + 0.5 * T), 0, 1)
        if sh.tone == "white":
            w_arr = np.maximum(w_arr, 0.84 + 0.16 * blot)
            s_arr *= 0.8
        elif sh.tone == "straw":
            s_arr = np.maximum(s_arr, 0.8 + 0.2 * blot)
        elif sh.tone == "stock":
            s_arr *= 0.45
            g_arr *= 0.5
            w_arr *= 0.5

        # visible paper back where the sheet stands off the wall near a free edge (always the pale unprinted side)
        ed = np.minimum(edge, np.array([max(sh.tear_f(float(a), float(b)), 0.0) for (a, b) in F2]))
        El = np.array([sh.edge_lift(float(a), float(b)) for (a, b) in F2]) * np.array([opn[v] for v in used])
        SO = Lc + Lo + El
        g_arr *= 1.0 - 0.5 * np.clip(El / 0.003, 0, 1)
        paint(paper, grime=_lookup(g_arr, P), wear=_lookup(w_arr, P), special=_lookup(s_arr, P), mode="set")
        bt = [t for t in TI.tolist() if max(SO[t[0]], SO[t[1]], SO[t[2]]) > 0.0015 and
              (max(Lc[t[0]], Lc[t[1]], Lc[t[2]]) > 0.0012 or min(ed[t[0]], ed[t[1]], ed[t[2]]) < 0.06)]
        if bt:
            bu = sorted({v for t in bt for v in t})
            bm = {v: i for i, v in enumerate(bu)}
            BP = P[bu] - N[bu] * BACK_T
            back = _mesh(f"back{sh.k}", BP.tolist(), [(bm[a], bm[c], bm[b]) for (a, b, c) in bt], [uvs[v] for v in bu])
            assign(back, m_paper)
            paint(back, grime=0.0, wear=0.3, special=0.12 * T, mode="set")
            stats["back"] += len(bt)

        # ---- print
        ink = Ink()
        em = p.emblem if p.emblem != "mixed" else sh.rs.fork("emblem").pick(["crown", "star", "moon", "sun"])
        _layout(sh, ink, p, sh.rs.fork("ink"), em, words)
        # clip cells: whole quads where the paper is (nearly) flat, single triangles elsewhere
        nq = {}
        for qi in tri_q:
            nq[qi] = nq.get(qi, 0) + 1
        cells = []
        for qi in sorted(q for q in nq if q in bil and nq[q] == 2):
            cells.append([remap[v] for v in quads[qi]])
        cells += [list(t) for t, qi in zip(TI.tolist(), tri_q) if not (qi in bil and nq[qi] == 2)]
        cpoly = [[(float(F2[v, 0]), float(F2[v, 1])) for v in c] for c in cells]
        tb = np.array([[min(q[0] for q in c), max(q[0] for q in c), min(q[1] for q in c), max(q[1] for q in c)] for c in cpoly])
        PO = P + N * INK_OFF
        # rubbed-off print: small type goes in patches, everything cracks off along the folds
        gx_, gz_ = np.meshgrid(np.linspace(-w / 2, w / 2, 14), np.linspace(-h / 2, h / 2, 14))
        rub_thr = float(np.quantile(rub_f(gx_, gz_), 1.0 - min(0.6, 0.32 * T * T + 0.06 * T))) if T > 0.02 else 2.0
        cph = sh.rs.fork("crack").u(0, 6.28)
        def cell_w(ci, qx, qz):
            cp = cpoly[ci]
            if len(cp) == 4:
                u = (qx - cp[0][0]) / (cp[1][0] - cp[0][0])
                v = (qz - cp[0][1]) / (cp[3][1] - cp[0][1])
                return ((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v)
            (ax, az), (bx, bz), (cx, cz) = cp
            den = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
            l0 = ((bz - cz) * (qx - cx) + (cx - bx) * (qz - cz)) / den
            l1 = ((cz - az) * (qx - cx) + (ax - cx) * (qz - cz)) / den
            return (l0, l1, 1.0 - l0 - l1)

        def surf(ci, qx, qz):
            return sum(PO[i] * wt for i, wt in zip(cells[ci], cell_w(ci, qx, qz)))

        def cell_at(qx, qz):
            for ci in np.nonzero((tb[:, 0] <= qx + 1e-7) & (tb[:, 1] >= qx - 1e-7) & (tb[:, 2] <= qz + 1e-7) & (tb[:, 3] >= qz - 1e-7))[0]:
                if min(cell_w(ci, qx, qz)) > -1e-4:
                    return int(ci)
            return None

        def whole(poly):
            """Cells of the corners if the small polygon can ride on the paper without being cut at the cell
            borders (the paper under it is flat to within the ink offset), else None."""
            cis = [cell_at(qx, qz) for (qx, qz) in poly]
            if any(c is None for c in cis) or len(set(cis)) == 1:
                return None
            p3 = [surf(c, qx, qz) for c, (qx, qz) in zip(cis, poly)]
            m = len(poly)
            tests = [((poly[i][0] + poly[(i + 1) % m][0]) / 2, (poly[i][1] + poly[(i + 1) % m][1]) / 2, (p3[i] + p3[(i + 1) % m]) / 2) for i in range(m)]
            if m == 3:
                tests.append((sum(q[0] for q in poly) / 3, sum(q[1] for q in poly) / 3, sum(p3) / 3))
            for (sx_, sz_, pl) in tests:
                c = cell_at(sx_, sz_)
                if c is None:
                    return None
                d = pl[1] - surf(c, sx_, sz_)[1]              # > 0: the paper bulges in front of the flat piece
                if d > 0.00013 or d < -0.0005:
                    return None
            return cis

        for key, polys in ink.polys.items():
            iv, ivf, ivuv, ifc, vmap = [], [], [], [], {}
            for (poly, rub) in polys:
                if _area(poly) < 0:
                    poly = poly[::-1]
                pxs = [q[0] for q in poly]
                pzs = [q[1] for q in poly]
                pieces = None
                if len(poly) == 3 and max(pxs) - min(pxs) < 0.035 and max(pzs) - min(pzs) < 0.035:
                    cis = whole(poly)
                    if cis is not None:
                        pieces = [(poly, cis)]
                if pieces is None:
                    pieces = []
                    cand = np.nonzero((tb[:, 0] < max(pxs)) & (tb[:, 1] > min(pxs)) & (tb[:, 2] < max(pzs)) & (tb[:, 3] > min(pzs)))[0]
                    for ci in cand:
                        piece = _clip(poly, cpoly[ci])
                        if len(piece) >= 3 and _area(piece) >= 3e-8:
                            pieces.append((piece, [int(ci)] * len(piece)))
                for (piece, pcs) in pieces:
                    mx = sum(q[0] for q in piece) / len(piece)
                    mz = sum(q[1] for q in piece) / len(piece)
                    if rub and (rub == 1 or T > 0.6) and float(rub_f(np.float64(mx), np.float64(mz))) > rub_thr:
                        continue
                    crk = [mz for c in sh.creases_x if abs(mx - c) < 0.0044] + [mx for c in sh.creases_z if abs(mz - c) < 0.0044]
                    if crk and T > 0.12 and 0.5 + 0.5 * math.sin(crk[0] * 55.0 + cph) * math.sin(crk[0] * 17.0 + 2.0 * cph) < (T - 0.1) * 0.9:
                        continue
                    face = []
                    for (qx, qz), ci in zip(piece, pcs):
                        kq = (int(round(qx * 5e4)), int(round(qz * 5e4)))
                        vi = vmap.get(kq)
                        if vi is None:
                            vi = len(iv)
                            vmap[kq] = vi
                            iv.append(surf(ci, qx, qz).tolist())
                            ivf.append((qx, qz))
                            ivuv.append((qx + uo[1] + 2 * w, qz + uo[0] + 2 * h))
                        if not face or (vi != face[-1] and vi != face[0]):
                            face.append(vi)
                    if len(face) >= 3:
                        ifc.append(face)
                        stats["ink"] += len(face) - 2
                        if not rub:
                            stats["text"] += len(face) - 2
            if not ifc:
                continue
            io = _mesh(f"ink{sh.k}_{key}", iv, ifc, ivuv)
            IF = np.array(ivf, dtype=np.float64)
            IP = np.array(iv, dtype=np.float64)
            fade = fade_f(IF[:, 0], IF[:, 1])
            if key == "pale" or dark_paper:                 # pale print on coloured stock: bare paper colour
                assign(io, m_paper)
                paint(io, grime=0.0, wear=0.25, special=_lookup(np.clip(fade - 0.35, 0, 0.62), IP), mode="set")
            elif key == "accent":                           # second colour: fades to pale patches
                assign(io, m_ink)
                paint(io, grime=0.1 * T, wear=_lookup(np.clip(np.maximum(fade * 1.6 - 0.76, (T - 0.55) * 1.3), 0, 0.8), IP), special=0.0, mode="set")
            else:                                           # black: the stock colour driven dark; fades browner and lighter
                assign(io, m_ink)
                paint(io, grime=1.0, wear=_lookup(fade, IP), special=0.0, mode="set")

        # ---- fixings (only where the head and the paper it holds can be seen)
        for (qx, qz) in sh.pins:
            X, Z = sh.world(qx, qz)
            if any(up.support(X, Z) > 0.0 for up in ups) or sh.tear_f(qx, qz) < 0.01:
                continue
            if not holds(qx, qz, kept_q):
                continue
            y = -(sh.top(X, Z) + 0.0002)
            pin_mats[sh.fixing].append(trs((X, y, Z), (90 + rs.u(-9, 9), rs.u(-9, 9), rs.u(0, 360)), rs.u(0.9, 1.15)))

    if pin_mats["tack"]:
        tack = lathe("tack", [(0.0072, 0.0), (0.0072, 0.0008), (0.006, 0.0022), (0.0035, 0.0035), (0.0, 0.004)], segments=10)
        o = instances_to_mesh("tacks", tack, pin_mats["tack"], keep_base=False)
        assign(o, m_iron)
        paint(o, special=0.15 + 0.3 * T, wear=0.55)
    if pin_mats["nail"]:
        nail = lathe("nail", [(0.0055, 0.0), (0.0058, 0.0014), (0.0042, 0.003), (0.0018, 0.0038), (0.0, 0.004)], segments=6)
        o = instances_to_mesh("nails", nail, pin_mats["nail"], keep_base=False)
        assign(o, m_iron)
        paint(o, special=0.45 + 0.4 * T)
    print(f"DBG sheets={len(sheets)} tris paper={stats['paper']} back={stats['back']} ink~{stats['ink']} (type/emblem {stats['text']}) "
          f"ext x {ext[0]:.2f}..{ext[1]:.2f} z {ext[2]:.2f}..{ext[3]:.2f} mats=3 "
          f"layout cost {sum(a.get('cost', 0.0) for a in placed):.1f}")
