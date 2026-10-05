"""Stack of 2-6 wooden shipping crates left on the square: every crate is built board by board
(sheathing boards with end grain, outer cleats / picture-frame battens with a diagonal brace, skids,
nailed lids, rose-head nails at every crossing, hemp rope beckets through the end panels). Three
constructions are mixed - battened shipping case, slatted produce crate (burlap-wrapped goods show
through the gaps) and a vertically boarded box. The stacking is skewed from the seed; the top crate
has its lid prised off and slid back over packing straw with wine bottles, stencilled shipping marks
on the faces and a crowbar leaning where the carter left it.
"""
from tk import *
import numpy as np
from math import sin, cos, tan, radians, degrees, atan2, hypot, pi
from mathutils import Matrix, Vector, Euler

# up to six individually boarded and nailed crates + straw: needs a little more than a single prop
META = {"name": "Wooden crates", "class": "prop", "mount": "ground", "tri_budget": 25000, "focus": 0.62, "lod": "far"}
PARAMS = {
    "count": {"default": 4, "min": 2, "max": 6, "desc": "number of crates in the stack"},
    "base": {"default": 0.8, "min": 0.5, "max": 1.1, "desc": "width of the largest (bottom) crate (m)"},
    "style": {"default": "mixed", "choices": ["mixed", "slatted", "boarded"], "desc": "crate construction mix"},
    "stencil": {"default": "ALDER & FINCH", "desc": "stencilled consignee mark on the big crate"},
    "stencil2": {"default": "FRAGILE", "desc": "second stencilled handling mark (with this-way-up arrows)"},
    "ink": {"default": "#17130f", "desc": "stencil paint colour"},
    "straw": {"default": True, "desc": "packing straw (and bottles) under the lid that is ajar"},
    "tone": {"default": "#c9a878", "desc": "wood tone (tint of the weathered boards)"},
    "skew": {"default": 0.6, "min": 0.0, "max": 1.0, "desc": "how carelessly the crates are stacked"},
    "age": {"default": 0.5, "min": 0.0, "max": 1.0, "desc": "dirt, damp, worn stencils, missing slats"},
    "crowbar": {"default": True, "desc": "crowbar leaning against the bottom crate"},
    "coil": {"default": True, "desc": "coil of hemp rope left on a free crate top"},
}
VARIANTS = {
    "pair_slatted": {"count": 2, "base": 0.72, "style": "slatted", "tone": "#e2c79a", "stencil": "PEARS", "stencil2": "No 12",
                     "skew": 0.9, "age": 0.25, "crowbar": False, "seed": 4},
    "dock_six": {"count": 6, "base": 1.0, "style": "boarded", "tone": "#8f6d4c", "stencil": "PORT NOCTURNE", "stencil2": "KEEP DRY",
                 "ink": "#e9dfc8", "skew": 0.8, "age": 0.9, "seed": 11},
    "tidy_three": {"count": 3, "base": 0.7, "tone": "#b98a5a", "stencil": "TEA", "stencil2": "CEYLON", "ink": "#6a1712",
                   "straw": False, "skew": 0.15, "age": 0.15, "crowbar": False, "seed": 23},
    "market_five": {"count": 5, "base": 0.68, "style": "mixed", "tone": "#d8c9a8", "stencil": "VINO ROSSO", "stencil2": "THIS SIDE UP",
                    "ink": "#1d2f45", "skew": 0.5, "age": 0.6, "seed": 37},
}


def _T(loc=(0, 0, 0), rot=(0, 0, 0), s=1.0):
    M = Matrix.Translation(Vector(loc)) @ Euler((radians(rot[0]), radians(rot[1]), radians(rot[2])), 'XYZ').to_matrix().to_4x4()
    if s != 1.0:
        M = M @ Matrix.Diagonal((s, s, s, 1.0))
    return M


def _put(o, M):
    loc, q, sc = M.decompose()
    e = q.to_euler('XYZ')
    rot = (degrees(e.x), degrees(e.y), degrees(e.z))
    if abs(sc.x - 1.0) > 1e-4:
        set_loc_rot(o, tuple(loc), rot, tuple(sc))
    else:
        set_loc_rot(o, tuple(loc), rot)
    return o


def _prism(L, Wd, T, c, m):
    """Board as a chamfered 8-gon prism along X with two extra vertex rings `m` from the ends, so the
    baked AO / grime under the cleats stays local instead of darkening the whole board."""
    m = min(m, L * 0.3)
    xs = [-L / 2, -L / 2 + m, L / 2 - m, L / 2]
    sec = [(-Wd / 2 + c, -T / 2), (Wd / 2 - c, -T / 2), (Wd / 2, -T / 2 + c), (Wd / 2, T / 2 - c),
           (Wd / 2 - c, T / 2), (-Wd / 2 + c, T / 2), (-Wd / 2, T / 2 - c), (-Wd / 2, -T / 2 + c)]
    vs = [(x, y, z) for x in xs for (y, z) in sec]
    fs = []
    for i in range(len(xs) - 1):
        for k in range(8):
            a, b = i * 8 + k, i * 8 + (k + 1) % 8
            fs.append((a, b, b + 8, a + 8))
    fs.append(tuple(range(7, -1, -1)))
    fs.append(tuple(range((len(xs) - 1) * 8, len(xs) * 8)))
    return mesh_object("board", vs, fs)


def _fquad(name, L, Wd):
    """Far level: a board / panel / cleat as the one face that shows (in XY, looking +Z, grain along X)."""
    return mesh_object(name, [(-L / 2, -Wd / 2, 0), (L / 2, -Wd / 2, 0), (L / 2, Wd / 2, 0), (-L / 2, Wd / 2, 0)], [(0, 1, 2, 3)])


def _fpanel(name, L, Wd, e0, e1):
    """Far level: the sheathing of one side as a panel of three fields along X, split where the boards of lod 0 have
    their own vertices next to the cleats (e0 / e1 from the two ends) - the shade under and beside the cleats stays
    there and the open middle keeps its tone."""
    xs = [-L / 2, -L / 2 + e0, L / 2 - e1, L / 2] if L - e0 - e1 > 0.03 else [-L / 2, L / 2]
    vs = [(x, y, 0.0) for x in xs for y in (-Wd / 2, Wd / 2)]
    return mesh_object(name, vs, [(2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1) for k in range(len(xs) - 1)])


def _fglyphs(o, k=0.6):
    """Far level: lettering as one block per glyph (the extent of every run of faces along the line, shrunk to
    about the inked share of it) - reads as a stencilled line of text from 20 m and beyond."""
    me = o.data
    if not len(me.polygons):
        return None
    co = np.array([tuple(v.co) for v in me.vertices])
    iv = []
    for f in me.polygons:
        q = co[list(f.vertices)]
        iv.append((q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()))
    iv.sort()
    runs = [list(iv[0])]
    for a, b, c, d in iv[1:]:
        r = runs[-1]
        if a <= r[1] + 1e-5:
            r[1], r[2], r[3] = max(r[1], b), min(r[2], c), max(r[3], d)
        else:
            runs.append([a, b, c, d])
    z = float(co[:, 2].max())
    vs, fs = [], []
    for a, b, c, d in runs:
        cx, cy, hw, hh = (a + b) / 2, (c + d) / 2, (b - a) / 2 * k, (d - c) / 2 * k
        n = len(vs)
        vs += [(cx - hw, cy - hh, z), (cx + hw, cy - hh, z), (cx + hw, cy + hh, z), (cx - hw, cy + hh, z)]
        fs.append((n, n + 1, n + 2, n + 3))
    return uv_box(mesh_object("stencil", vs, fs))


def _ext(w, d, a):
    ca, sa = abs(cos(radians(a))), abs(sin(radians(a)))
    return w * ca + d * sa, w * sa + d * ca


def build(p, rng):
    # lod 2 (META "lod": "far", docs/LOD.md) is modelled here: every crate as a closed case of one panel per side
    # with its cleats, slats, beckets, stencil blocks, lid and straw as single faces in front of it; nails, skids,
    # board joints, strapping, corks and bale cords (all well under a pixel from 25 m) are left out
    far = lod() == 2
    n = int(p.count)
    B = float(p.base)
    skew, age = float(p.skew), float(p.age)
    tl = hex_rgb(p.tone)
    W = [mat("wood_weathered", tl), mat("wood_oak", tuple(lerp(c, 1.0, 0.5) for c in tl)),
         mat("wood_oak", tuple(c * 0.62 for c in tl))]         # weathered deal, fresh pale boards, dark stained
    endg = mat("wood_endgrain", tuple(lerp(c, 1.0, 0.25) for c in tl))
    iron = mat("iron_rusty")
    rope = mat("rope", "#cfb37c")
    sack = mat("burlap", "#c9b27c")
    ink = mat("wood_painted", p.ink)
    glass = mat("glass_clear", "#86b890", opacity=0.55)

    # ------------------------------------------------------------------ layout (sizes, tiers, skew)
    r = rng.fork("layout")

    def mk(w, d, h, tag):
        return dict(w=w, d=d, h=h, x=0.0, y=0.0, z=0.0, yaw=0.0, loaded=False, tag=tag)

    pick4 = r.pick(["u1", "t"])
    pick5 = r.chance(0.5)
    plan = {2: ["u0"], 3: ["g1", "u0"], 4: ["g1", "u0", pick4],
            5: ["g1", "g2", "u0", "u1"] if pick5 else ["g1", "u0", "u1", "t"],
            6: ["g1", "g2", "u0", "u1", "t"]}[max(2, min(6, n))]
    g0 = mk(B, B * r.u(0.62, 0.76), B * r.u(0.56, 0.72), "g0")
    g0["yaw"] = r.u(-5, 5) * skew
    byt = {"g0": g0}
    crates = [g0]
    e0 = _ext(g0["w"], g0["d"], g0["yaw"])

    def upper(s, tag):
        w = max(0.26, s["w"] * r.u(0.62, 0.84))
        d = max(0.2, min(s["d"] * r.u(0.72, 0.9), w * 0.92))
        h = max(0.17, w * r.u(0.48, 0.78))
        c = mk(w, d, h, tag)
        yr = r.pick([-1, 1]) * r.u(4, 18) * skew
        ex, ey = _ext(w, d, yr)
        mx = max(0.0, (s["w"] - ex) / 2) + 0.02
        my = max(0.0, (s["d"] - ey) / 2) + 0.02
        ox = r.u(-1, 1) * mx * (0.3 + 0.7 * skew)
        oy = r.u(-1, 1) * my * (0.3 + 0.7 * skew)
        a = radians(s["yaw"])
        c["x"] = s["x"] + ox * cos(a) - oy * sin(a)
        c["y"] = s["y"] + ox * sin(a) + oy * cos(a)
        c["z"] = s["z"] + s["h"]
        c["yaw"] = s["yaw"] + yr
        s["loaded"] = True
        return c

    for tag in plan:
        if tag == "g1":
            w = B * r.u(0.68, 0.84)
            c = mk(w, w * r.u(0.68, 0.85), w * r.u(0.6, 0.85), tag)
            c["yaw"] = r.u(-9, 9) * skew
            e1 = _ext(c["w"], c["d"], c["yaw"])
            c["x"] = e0[0] / 2 + 0.075 + r.u(0, 0.05) + e1[0] / 2
            c["y"] = (e0[1] - e1[1]) / 2 - r.u(0, 0.08) * skew
        elif tag == "g2":
            w = B * r.u(0.5, 0.62)
            c = mk(w, w * r.u(0.7, 0.9), w * r.u(0.5, 0.8), tag)
            c["yaw"] = r.u(-28, 28) * skew
            g1 = byt["g1"]
            e1 = _ext(g1["w"], g1["d"], g1["yaw"])
            e2 = _ext(c["w"], c["d"], c["yaw"])
            c["y"] = g1["y"] - e1[1] / 2 - 0.075 - r.u(0, 0.05) - e2[1] / 2
            c["x"] = max(g1["x"] + r.u(-0.12, 0.08), e0[0] / 2 + 0.075 + e2[0] / 2)
        elif tag == "u0":
            c = upper(g0, tag)
        elif tag == "u1":
            c = upper(byt["g1"], tag)
        else:
            c = upper(byt["u0"], tag)
        byt[tag] = c
        crates.append(c)

    # recentre the footprint on the origin
    xs, ys = [], []
    for c in crates:
        ex, ey = _ext(c["w"], c["d"], c["yaw"])
        xs += [c["x"] - ex / 2, c["x"] + ex / 2]
        ys += [c["y"] - ey / 2, c["y"] + ey / 2]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for c in crates:
        c["x"] -= cx
        c["y"] -= cy

    # construction kind per crate; the highest free crate is the one with its lid ajar
    for i, c in enumerate(crates):
        if p.style == "slatted":
            c["kind"] = "slat"
        elif p.style == "boarded":
            c["kind"] = "frame" if i == 0 else r.pick(["frame", "box", "box"])
        else:
            c["kind"] = "frame" if i == 0 else ("slat" if i == 1 else r.weighted([("slat", 3), ("box", 3), ("frame", 2)]))
    free = [c for c in crates if not c["loaded"]]
    ajar = max(free, key=lambda c: c["z"] + c["h"])
    marks = {0: p.stencil}
    marks[1] = p.stencil2

    def surface_z(xw, yw):
        """Height of the surface a fallen straw would land on (None = on an edge / against a crate)."""
        best = 0.0
        for c in crates:
            a = radians(-c["yaw"])
            dx, dy = xw - c["x"], yw - c["y"]
            lx, ly = dx * cos(a) - dy * sin(a), dx * sin(a) + dy * cos(a)
            top = c["z"] + c["h"]
            if abs(lx) < c["w"] / 2 - 0.012 and abs(ly) < c["d"] / 2 - 0.012:
                if c is ajar or not c["loaded"]:
                    return None
                best = max(best, top)
            elif abs(lx) < c["w"] / 2 + 0.04 and abs(ly) < c["d"] / 2 + 0.04:
                return None
        for c in crates:                      # anything standing on that surface there?
            if abs(c["z"] - best) < 1e-6 and best > 0:
                a = radians(-c["yaw"])
                dx, dy = xw - c["x"], yw - c["y"]
                lx, ly = dx * cos(a) - dy * sin(a), dx * sin(a) + dy * cos(a)
                if abs(lx) < c["w"] / 2 + 0.04 and abs(ly) < c["d"] / 2 + 0.04:
                    return None
        return best

    coil_done = []
    nails = []          # world matrices of every nail head
    inked = []          # stencil meshes
    ropes = []
    fallen = []         # far level: the straws lying on the paving / on a lid (ends)
    lying = []          # ... and those lying in the open case (ends, length)
    I4 = Matrix.Identity(4)

    # ------------------------------------------------------------------ stencil lettering / arrows
    def stencil(M, body, cxp, cyp, availw, maxh, rr, arrows=False):
        body = str(body).strip()
        if not body or availw < 0.08 or maxh < 0.025:
            return
        words = body.split()
        lines = [body]
        if len(words) >= 2 and len(body) > 8 and maxh > 0.14:
            k = (len(words) + 1) // 2
            lines = [" ".join(words[:k]), " ".join(words[k:])]
        nl = len(lines)
        size = min(0.1, maxh / (nl * 1.5))
        aw = availw - (size * 1.5 if arrows else 0.0)
        for attempt in range(2):
            objs, wmax = [], 0.0
            for ln in lines:
                o = text("stencil", ln, size=size, depth=0.0008, rot=(0, 0, 0), font="sans")
                co = np.array([tuple(v.co) for v in o.data.vertices]) if len(o.data.vertices) else np.zeros((1, 3))
                objs.append((o, co.min(axis=0), co.max(axis=0)))
                wmax = max(wmax, co[:, 0].max() - co[:, 0].min())
            if wmax <= aw or attempt == 1:
                break
            delete([o for o, _, _ in objs])
            size *= aw / wmax * 0.98
        lh = size * 1.5
        skw = rr.u(-1.6, 1.6)
        ox = -size * 0.75 if arrows else 0.0
        Ms = M @ _T((cxp + ox + rr.u(-0.01, 0.01), cyp + rr.u(-0.006, 0.006), 0.0003), (0, 0, skw))
        for i, (o, mn, mx) in enumerate(objs):
            yc = (nl - 1) / 2 * lh - i * lh
            Mo = Ms @ _T((-(mn[0] + mx[0]) / 2, yc - (mn[1] + mx[1]) / 2, 0))
            if far:
                q = _fglyphs(o)
                delete(o)
                if q is None:
                    continue
                o = q
                Mo = Mo @ _T((0, 0, 0.006))
            _put(o, Mo)
            assign(o, ink)
            inked.append(o)
        if arrows:
            s = min(size * nl * 1.5, maxh * 0.95)
            a = s * 0.11
            pts = [(-a, -s / 2), (a, -s / 2), (a, s * 0.08), (a * 2.6, s * 0.08), (0, s / 2), (-a * 2.6, s * 0.08), (-a, s * 0.08)]
            for k in range(2):
                if far:
                    o = uv_box(mesh_object("arrow", [(q[0], q[1], 0.0068) for q in pts], [(0, 1, 2, 6), (3, 4, 5)]))
                else:
                    o = shape_extrude("arrow", pts, depth=0.0008)
                _put(o, Ms @ _T((wmax / 2 + size * 0.55 + k * a * 6.2, 0, 0)))
                assign(o, ink)
                inked.append(o)

    # ------------------------------------------------------------------ one crate
    def crate(c, i):
        rr = rng.fork("crate%d" % i)
        w, d, h, kind = c["w"], c["d"], c["h"], c["kind"]
        t = clamp(w * 0.02, 0.011, 0.019)                # board thickness
        ct = t + 0.004                                   # cleat thickness
        cw = clamp(w * 0.085, 0.04, 0.075)               # cleat width
        sk = 0.024 if w > 0.42 else 0.017                # skid height
        C = _T((c["x"], c["y"], c["z"]), (0, 0, c["yaw"]))
        wb = W[i % 3]
        wc = W[(i + 1 + (1 if rr.chance(0.3) else 0)) % 3] if (kind == "frame" or rr.chance(0.5)) else wb
        zb, zt = sk, h - t
        H = zt - zb
        is_ajar = c is ajar
        parts = []

        def brd(F, L, Wd, T, x, y, z, ang=0.0, m=wb, bv=0.003, seg=0.0, ty=0.0, fk=False, lip=0, both=False, past=0.0):
            if far:
                if not fk:
                    return None
                o = _fquad("board", L + 2 * past, Wd)       # far level: the outer face only (`past`: run on by this at both ends)
                # ... and of an upright cleat the edge towards the middle of its side: the slot behind the face is closed
                # there, a view along the side ends on wood (`both`: the edge at the corner too, where the next side has
                # slats and no panel that runs past it)
                for s_ in ((lip, -lip) if both else (lip,)) if lip else ():
                    yl = s_ * Wd / 2
                    e = mesh_object("lip", [(-L / 2, yl, 0), (L / 2, yl, 0), (L / 2, yl, -T), (-L / 2, yl, -T)],
                                    [(0, 1, 2, 3) if s_ > 0 else (3, 2, 1, 0)])
                    o = join([o, e])
                uv_grain(o, 'X')
                assign(o, m)
                _put(o, C @ F @ _T((x, y, z + T / 2), (0, 0, ang)))
                parts.append(o)
                return o
            if seg > 0:
                o = _prism(L, Wd, T, min(bv, T * 0.3), seg)
            else:
                o = box("board", (L, Wd, T), bevel=min(bv, T * 0.3, Wd * 0.3), segments=1)
            uv_grain(o, 'X')
            assign(o, m)
            if bv > 0:
                assign_faces(o, endg, lambda cc, nn: abs(nn.x) > 0.9)
            _put(o, C @ F @ _T((x, y, z), (0, ty, ang)))
            parts.append(o)
            return o

        def nail(F, x, y, z):
            if rr.chance(0.04 + 0.05 * age):
                return                                    # the odd nail has gone
            nails.append(C @ F @ _T((x + rr.u(-0.0025, 0.0025), y + rr.u(-0.0025, 0.0025), z - 0.0002),
                                    (rr.u(-10, 10), rr.u(-10, 10), rr.u(0, 60)), rr.u(0.85, 1.2)))

        sides = [("front", _T((0, -d / 2, 0), (90, 0, 0)), w, False),
                 ("right", _T((w / 2, 0, 0), (90, 0, 90)), d - 2 * t, True),
                 ("back", _T((0, d / 2, 0), (90, 0, 180)), w, False),
                 ("left", _T((-w / 2, 0, 0), (90, 0, 270)), d - 2 * t, True)]
        missing_done = False
        text_body = marks.get(i)
        ztop = zt if is_ajar else h                       # far level: the side panels run from the ground to here
        for sname, F, pw, is_end in sides:
            slats = kind == "slat" and not is_end
            vertical = kind == "box" and not is_end
            ycs, bw = [], 0.0
            # far level: every side runs past the corner by the thickness of the cleats on the next side (2 mm where
            # that has none), and so do lid and packing - the slot between a panel and the cleat faces in front of
            # it is closed at its ends and from above, nobody looks along it at the back of a cleat
            xt = 0.002 if (is_end and kind == "slat") else ct
            if far and slats:                               # behind the slats: one face at their inner side, which takes
                o = _fquad("backing", pw + 2 * xt, ztop)    # the shade of the gaps (closed, nothing shows through)
                assign(uv_grain(o, 'X'), wb)
                _put(o, C @ F @ _T((0, ztop / 2, -t)))
                parts.append(o)
            if far and not slats:                           # the sheathing of this side as one panel (ends: over the corners)
                PW = pw + (2 * t if is_end else 0.0) + 2 * xt
                e_ = cw + 0.03
                o = _fpanel("panel", ztop, PW, zb + e_, ztop - zt + e_) if vertical else _fpanel("panel", PW, ztop, e_ + xt, e_ + xt)
                assign(uv_grain(o, 'X'), wb)
                _put(o, C @ F @ _T((0, ztop / 2, 0), (0, 0, 90 if vertical else 0)))
                parts.append(o)
                # ... and its inner face: what lod 0 has inside the case (buried, dark) is taken up by this one when
                # the masks are carried over, not by the panel that shows
                o = _fquad("lining", (pw if is_end else pw - 2 * t) - 0.004, ztop - sk - t)
                assign(uv_grain(o, 'X'), wb)
                _put(o, C @ F @ _T((0, (sk + ztop - t) / 2, -t), (180, 0, 0)))
                parts.append(o)
            if vertical:
                nv = max(3, int(round(pw / 0.11)))
                bwv = (pw - (nv - 1) * 0.002) / nv
                xcs = [-pw / 2 + bwv / 2 + j * (bwv + 0.002) for j in range(nv)]
                for xc in xcs:
                    brd(F, H - rr.u(0, 0.003), bwv * rr.u(0.975, 1.0), t, xc, zb + H / 2, -t / 2 - rr.u(0, 0.0008), 90 + rr.u(-0.2, 0.2), seg=cw + 0.03)
                for yk in (zt - cw / 2, zb + cw / 2):       # horizontal cleats top and bottom
                    brd(F, pw, cw, ct, 0, yk, ct / 2, rr.u(-0.15, 0.15), wc, fk=True)
                    for k, xc in enumerate(xcs):
                        nail(F, xc + bwv * 0.18 * (1 if k % 2 else -1), yk + cw * 0.2 * (1 if k % 2 else -1), ct)
                for q in ((0.3, 0.7) if H > 0.2 else (0.5,)):   # hoop-iron strapping nailed across the boards
                    ys_ = zb + cw + (H - 2 * cw) * q + rr.u(-0.006, 0.006)
                    sa_ = rr.u(-0.5, 0.5)
                    if not far:
                        o = box("strap", (pw + 0.003, 0.024, 0.002))
                        assign(o, iron)
                        _put(o, C @ F @ _T((0, ys_, 0.001), (0, 0, sa_)))
                        parts.append(o)
                    for k, xc in enumerate(xcs):
                        nail(F, xc + bwv * 0.2 * (1 if k % 2 else -1), ys_, 0.002)
            else:
                g = clamp(H * 0.05, 0.012, 0.028) if slats else 0.002
                nb = max(2, int(round(H / (0.1 if slats else 0.125))))
                bw = (H - (nb - 1) * g) / nb
                ycs = [zb + bw / 2 + j * (bw + g) for j in range(nb)]
                gone = -1
                if slats and not missing_done and nb >= 4 and age > 0.3 and rr.chance(0.5 * age) and text_body is None:
                    gone = 1 + int(rr.u(0, nb - 2.001))
                    missing_done = True
                for j, yc in enumerate(ycs):
                    if j == gone:
                        if far:                             # the wrapped bale shows where the slat is missing
                            o = _fquad("bundle", pw - 0.006, bw + g)
                            assign(uv_box(o), sack)
                            _put(o, C @ F @ _T((0, yc, -t + 0.004)))
                        continue
                    brd(F, pw - rr.u(0, 0.003), bw * rr.u(0.965, 1.0), t, rr.u(-0.0015, 0.0015), yc, -t / 2 - rr.u(0, 0.0008),
                        rr.u(-0.25, 0.25) * (2.0 if slats else 1.0), seg=cw + 0.03, ty=rr.u(-0.45, 0.45) if slats else 0.0, fk=slats,
                        past=0.005)                         # far level: a slat's face reaches the corners whatever its play
                    if slats:                               # nailed into the edges of the end panels
                        for sx in (-1, 1):
                            for sy in (-1, 1):
                                nail(F, sx * (pw / 2 - t / 2), yc + sy * bw * 0.24, 0)
                        if pw > 0.55:
                            nail(F, 0.006 * (1 if j % 2 else -1), yc, 0)
                if slats and pw > 0.55:                     # inner centre batten the slats are nailed to
                    brd(F, H, 0.04, 0.02, 0, zb + H / 2, -t - 0.0105, 90)
            # ---- outer cleats
            if is_end or kind == "frame":
                xo = pw / 2 + (t if is_end else 0.0) - cw / 2
                for sx in (-1, 1):
                    brd(F, H, cw, ct, sx * xo, zb + H / 2, ct / 2, 90 + rr.u(-0.2, 0.2), wc, fk=True, lip=sx, both=kind == "slat")
                    for j, yc in enumerate(ycs):
                        for sy in (-1, 1):
                            nail(F, sx * xo + sy * cw * 0.2, yc + sy * bw * 0.22, ct)
                inner = 2 * xo - cw - 0.002                 # clear length between the two upright cleats
                if kind == "frame" and not is_end:
                    for yk in (zt - cw / 2, zb + cw / 2):
                        brd(F, inner, cw, ct, 0, yk, ct / 2, rr.u(-0.1, 0.1), wc, fk=True)
                        nn = max(2, int(inner / 0.14))
                        for k in range(nn):
                            nail(F, -inner / 2 + inner * (k + 0.5) / nn, yk + cw * 0.18 * (1 if k % 2 else -1), ct)
                    if not (sname == "front" and text_body):
                        Wi, Hi = inner, H - 2 * cw
                        a = atan2(Hi, Wi)
                        sg = rr.pick([-1, 1])
                        L = hypot(Wi, Hi) + cw * 0.45
                        brd(F, L - (cw * 1.4 if far else 0.0), cw * 0.9, ct * 0.8, 0, zb + H / 2, ct * 0.4, sg * degrees(a), wc, fk=True)
                        for q in (-0.38, -0.14, 0.14, 0.38):
                            nail(F, q * L * cos(a), zb + H / 2 + sg * q * L * sin(a), ct * 0.8)
            # ---- rope becket through the end panels
            if is_end:
                rail = kind == "frame"
                yh = zb + H * (0.62 if rail else 0.68)
                z0 = 0.0
                if rail:
                    inner = pw + 2 * t - 2 * cw - 0.002
                    brd(F, inner, cw * 0.9, ct, 0, yh, ct / 2, rr.u(-0.2, 0.2), wc, fk=True)
                    for sx in (-1, 1):
                        nail(F, sx * (inner / 2 - 0.02), yh + cw * 0.15 * sx, ct)
                    z0 = ct
                hs = clamp(pw * 0.2, 0.045, 0.075)
                rad = 0.0058
                drop = min(hs * rr.u(0.85, 1.15), yh - zb - 0.03)
                sw = rr.u(-0.012, 0.012)
                zl = rad + 0.0015
                pts = [(-hs, yh, z0 - 0.012), (-hs, yh, z0 + rad * 0.8), (-hs * 0.98, yh - drop * 0.25, (z0 + zl) if not rail else z0 + zl),
                       (-hs * 0.72 + sw, yh - drop * 0.78, zl + z0 * (0.0 if drop > cw * 0.9 else 1.0)),
                       (sw, yh - drop, zl + z0 * (0.0 if drop > cw * 0.9 else 1.0)),
                       (hs * 0.72 + sw, yh - drop * 0.78, zl + z0 * (0.0 if drop > cw * 0.9 else 1.0)),
                       (hs * 0.98, yh - drop * 0.25, z0 + zl), (hs, yh, z0 + rad * 0.8), (hs, yh, z0 - 0.012)]
                Mw = C @ F
                if far:                                    # the becket as a flat strip of three legs
                    zq, rq, y1 = z0 + zl + 0.003, rad * 1.25, yh - drop * 0.78
                    x1, x2 = -hs * 0.72 + sw, hs * 0.72 + sw
                    k7 = rq * 0.7
                    vs = [(-hs - rq, yh), (-hs + rq, yh), (x1 - k7, y1 - k7), (x1 + k7, y1 + k7),
                          (x2 + k7, y1 - k7), (x2 - k7, y1 + k7), (hs + rq, yh), (hs - rq, yh)]
                    vs = [tuple(Mw @ Vector((q[0], q[1], zq))) for q in vs]
                    o = uv_box(mesh_object("becket", vs, [(0, 2, 3, 1), (2, 4, 5, 3), (4, 6, 7, 5)]))
                    assign(o, rope)
                    ropes.append(o)
                else:
                    o = sweep("becket", [tuple(Mw @ Vector(q)) for q in pts], rad, sides=5, smooth_path=True, step=0.02)
                    assign(o, rope)
                    ropes.append(o)
            # ---- stencilled marks on the front
            if sname == "front" and text_body:
                arrows = i != 0
                if kind == "slat":
                    j = min(len(ycs) - 1, len(ycs) // 2 + (1 if len(ycs) > 2 else 0))
                    stencil(C @ F, text_body, 0, ycs[j], pw * 0.86, bw * 1.12, rr, arrows)
                elif kind == "frame":
                    stencil(C @ F, text_body, 0, zb + H * 0.5, pw - 2 * cw - 0.07, (H - 2 * cw) * 0.62, rr, arrows)
                else:
                    stencil(C @ F, text_body, 0, zb + H * 0.5, pw * 0.86, (H - 2 * cw) * 0.5, rr, arrows)

        # ---- floor and skids
        brd(I4, w - 2 * t - 0.002, d - 2 * t - 0.002, t, 0, 0, sk + t / 2, 0, wb, bv=0.0)
        xs_ = w / 2 - max(0.05, w * 0.14)
        for sx in (-1, 1):
            brd(I4, d - rr.u(0, 0.006), 0.052, sk, sx * xs_, rr.u(-0.003, 0.003), sk / 2, 90 + rr.u(-0.4, 0.4), wc, bv=0.004)
        if far:                                             # the underside of the floor: shows where a crate overhangs
            brd(_T(rot=(180, 0, 0)), w - 0.004, d - 0.004, 0.0, 0, 0, -(sk - 0.001), fk=True)   # the one below

        # ---- lid
        lw, ld = w + 0.012, d + 0.012
        tilt, sy = 0.0, 0.0
        if is_ajar:
            sy = d * rr.u(0.27, 0.36)
            tilt = rr.u(3.5, 6.0) if p.straw else 0.0
            TL = _T((0, d / 2, zt)) @ _T(rot=(-tilt, 0, 0)) @ _T((rr.u(-0.035, 0.035), -d / 2 + sy, 0), (0, 0, rr.u(4, 10) * rr.pick([-1, 1])))
        else:
            TL = _T((rr.u(-0.004, 0.004), rr.u(-0.004, 0.004), zt), (0, 0, rr.u(-1.2, 1.2) * skew))
        gl = 0.008 if kind == "slat" else 0.003
        nl = max(2, int(round(ld / 0.115)))
        lbw = (ld - (nl - 1) * gl) / nl
        lys = [-ld / 2 + lbw / 2 + j * (lbw + gl) for j in range(nl)]
        top_cleats = not c["loaded"]
        xc_ = lw / 2 - cw * 0.95
        if far and is_ajar:                                 # the slid-back lid as a closed slab, the open case covered
            o = box("lid", (lw, ld, t))                     # at its rim by the packing
            uv_grain(o, 'X')
            assign(o, wb)
            _put(o, C @ TL @ _T((0, 0, t / 2)))
            parts.append(o)
        elif far:
            brd(I4, w + 2 * ct, d + (0.004 if kind == "slat" else 2 * ct), 0.0, 0, 0, h, fk=True)
        for j, yl in enumerate(lys):
            brd(TL, lw - rr.u(0, 0.005), lbw * rr.u(0.97, 1.0), t, rr.u(-0.0025, 0.0025), yl, t / 2 + rr.u(0, 0.0008), rr.u(-0.25, 0.25))
            for sx in (-1, 1):
                for q in (-1, 1):
                    nail(TL, sx * (w / 2 - t / 2), yl + q * lbw * 0.24, t)
                    if top_cleats:
                        nail(TL, sx * xc_ + q * cw * 0.17, yl + q * lbw * 0.2, t + ct)
        if top_cleats:
            for sx in (-1, 1):
                brd(TL, ld - 0.024, cw * 0.8, ct, sx * xc_, 0, t + ct / 2, 90 + rr.u(-0.3, 0.3), wc, fk=True)

        if p.coil and top_cleats and not is_ajar and not coil_done and text_body is None:
            rc = 0.0085
            Rc = min(0.115, xc_ - cw * 0.4 - rc - 0.012, ld / 2 - 0.03)
            if Rc > 0.06:
                coil_done.append(1)
                ox = rr.u(-1, 1) * max(0.0, xc_ - cw * 0.4 - rc - 0.012 - Rc)
                oy = rr.u(-0.3, 1) * max(0.0, ld / 2 - 0.03 - Rc)
                pth, turns = [], 3.5
                N = int(turns * 16)
                for k in range(N + 1):
                    a = 2 * pi * turns * k / N
                    q = Rc * (1.0 - 0.06 * (a / (2 * pi)) / turns + 0.03 * sin(a * 0.37 + i))
                    pth.append((ox + q * cos(a), oy + q * sin(a), h + rc + a / (2 * pi) * rc * 1.75))
                xe, ye, ze = pth[-1]
                yf = -d / 2 - ct - rc - 0.003
                hang = min(0.2, h * 0.5)
                pth += [(xe - 0.004, ye - Rc * 0.8, h + rc + 0.012), (xe + 0.004, -ld / 2 + 0.03, h + rc + 0.001), (xe + 0.006, yf + 0.004, h - 0.012),
                        (xe + 0.012, yf, h - hang * 0.6), (xe + 0.008, yf, h - hang)]
                if far:                                    # the coil as a six-sided puck, its tail as a strip over the edge
                    Rk, z1 = Rc + rc * 0.6, h + rc * 7.4
                    ring = [(ox + Rk * cos(k * pi / 3), oy + Rk * sin(k * pi / 3)) for k in range(6)]
                    vs = [(x, y, h) for x, y in ring] + [(x, y, z1) for x, y in ring]
                    fs = [(k, (k + 1) % 6, (k + 1) % 6 + 6, k + 6) for k in range(6)] + [(6, 7, 8, 9), (6, 9, 10, 11)]
                    y1 = -d / 2 - 0.0005
                    vs += [(xe - rc, y1, h + 2 * rc), (xe + rc, y1, h + 2 * rc), (xe - rc, yf - rc, h + rc), (xe + rc, yf - rc, h + rc),
                           (xe - rc + 0.008, yf - rc, h - hang), (xe + rc + 0.008, yf - rc, h - hang)]
                    fs += [(12, 14, 15, 13), (14, 16, 17, 15)]
                    o = uv_box(mesh_object("coil", [tuple(C @ Vector(v)) for v in vs], fs))
                else:
                    o = sweep("coil", [tuple(C @ Vector(v)) for v in pth], rc, sides=5, smooth_path=True, step=0.036)
                assign(o, rope)
                ropes.append(o)

        body = join(parts, "crate%d" % i)
        z0w = c["z"]
        sd = p.seed * 13 + i * 7

        def grime(co, nrm):
            a = height_ramp(0.0, 0.32)(co, nrm) * (0.35 + 0.5 * age)
            b = height_ramp(z0w, z0w + 0.1)(co, nrm) * 0.35
            n_ = noise_mask(0.22, seed=sd, lo=0.42, hi=0.8)(co, nrm) * (0.15 + 0.5 * age)
            return np.maximum(np.maximum(a, b), n_)

        if not far:
            paint(body, grime=grime,
                  wear=mask_mul(noise_mask(0.1, seed=sd + 3, lo=0.5, hi=0.85), 0.25 + 0.35 * age),
                  special=mask_mul(height_ramp(0.0, 0.16), noise_mask(0.12, seed=sd + 5, lo=0.45, hi=0.8), age * 0.55))

        # ---- contents
        wi, di = w - 2 * t - 0.004, d - 2 * t - 0.004
        if far:
            if is_ajar:                                     # packing straw / wrapped bale: closes the open case at its rim
                o = _fquad("packing", w + 2 * ct, d + (0.0 if kind == "slat" else 2 * ct))
                assign(uv_box(o), sack)
                _put(o, C @ _T((0, 0, zt)))
        elif kind == "slat" or (is_ajar and not p.straw):
            inset = 0.05 if kind == "slat" else 0.02
            lh_ = H - t - (0.05 if is_ajar else 0.02)
            o = box("bundle", (wi - 0.02, di - inset, lh_), bevel=min(0.03, lh_ * 0.3), segments=2)
            densify(o, 0.11)
            displace_noise(o, strength=0.007, scale=0.16, seed=sd)
            smooth(o, 60)
            assign(o, sack)
            _put(o, C @ _T((0, 0, sk + t + lh_ / 2 + 0.001)))
            paint(o, grime=mask_mul(noise_mask(0.15, seed=sd + 1), 0.4))
            if is_ajar:                                    # cord tied round the wrapped bale
                for k, xk in enumerate((-wi * 0.22, wi * 0.2)):
                    hx, hy, zc_ = 0.0, (di - inset) / 2 + 0.004, sk + t + 0.001
                    loop = [(xk, -hy, zc_ + 0.02), (xk, -hy, zc_ + lh_ - 0.02), (xk, -hy + 0.03, zc_ + lh_ + 0.004), (xk + 0.004, 0, zc_ + lh_ + 0.007),
                            (xk, hy - 0.03, zc_ + lh_ + 0.004), (xk, hy, zc_ + lh_ - 0.02), (xk, hy, zc_ + 0.02)]
                    q = sweep("cord", [tuple(C @ Vector(v)) for v in loop], 0.004, sides=5, smooth_path=True, step=0.03)
                    assign(q, rope)
                    ropes.append(q)
        if is_ajar and p.straw:
            straw_fill(c, C, rr, w, d, t, zt, sk, sy, tilt, wi, di)
        if c is g0 and p.crowbar:
            crowbar(C, sides[3][1], rr, (d - 2 * t) / 2 + t - cw / 2, h, ct)
        if not c["loaded"] and not is_ajar:
            socket("top_%s" % c["tag"], "attach", (c["x"], c["y"], c["z"] + h + ct), rot=(0, 0, c["yaw"]), role="prop_top")

    # ------------------------------------------------------------------ straw, bottles
    strands = {}

    def straw_fill(c, C, rr, w, d, t, zt, sk, sy, tilt, wi, di):
        ylf = -d / 2 - 0.006 + sy                         # front edge of the slid-back lid
        ph = [rr.u(0, 6.28) for _ in range(4)]
        tt = tan(radians(tilt))

        def mz(x, y):
            e = smoothstep(0.0, 0.05, min(wi / 2 - abs(x), di / 2 - abs(y)))
            fb = 1.0 - smoothstep(ylf - 0.02, ylf + 0.1, y)
            nz = 0.5 * sin(x * 23 + ph[0]) * cos(y * 19 + ph[1]) + 0.5 * sin(x * 41 + y * 37 + ph[2])
            z = zt - 0.03 + e * (0.022 + 0.03 * fb + 0.016 * nz)
            if y > ylf - 0.03:
                z = min(z, zt + (d / 2 - y) * tt - 0.008)
            return z

        if not far:
            m = surface("straw_bed", lambda u, v: ((u - 0.5) * wi, (v - 0.5) * di, mz((u - 0.5) * wi, (v - 0.5) * di)), 13, 11)
            smooth(m, 70)
            assign(m, sack)
            _put(m, C)
            paint(m, grime=mask_mul(noise_mask(0.06, seed=p.seed + 9), 0.55))

        def add(kind_, M):
            strands.setdefault(kind_, []).append(M)

        y_lo, y_hi = -d / 2 + t + 0.008, min(ylf + 0.06, d / 2 - t - 0.01)
        dens = clamp(w / 0.5, 0.6, 1.6)
        for _ in range(int(55 * dens)):                   # lying in the exposed strip
            L = rr.u(0.09, 0.2)
            yaw = rr.u(-32, 32) + rr.pick([0, 180])
            x = rr.u(-1, 1) * max(0.0, wi / 2 - L / 2 * 0.95)
            lo = y_lo + L / 2 * abs(sin(radians(yaw)))
            y = rr.u(lo, max(lo, y_hi))
            kd = rr.pick(["arc", "kink", "flat"])
            Ml = C @ _T((x, y, mz(x, y) + rr.u(0.003, 0.02)), (rr.u(0, 360), rr.u(-16, 16), yaw), L)
            add(kd, Ml)
            if far:
                Ci = C.inverted() @ Ml
                qa, qb = Ci @ Vector((-0.5, 0, 0.03)), Ci @ Vector((0.5, 0, 0.03))
                qa.z, qb.z = max(qa.z, zt + 0.006), max(qb.z, zt + 0.006)   # above the packing
                lying.append((C @ qa, C @ qb, L))
        for _ in range(int(40 * dens)):                   # spilling over the front rim
            L = rr.u(0.13, 0.25)
            x = rr.u(-1, 1) * wi / 2 * 0.94
            add("droop", C @ _T((x, -d / 2 - rr.u(-0.012, 0.03), zt + rr.u(0.006, 0.024)), (rr.u(-25, 25), rr.u(-12, 14), -90 + rr.u(-38, 38)), L))
        for sx in (-1, 1):                                # ... and over the two ends
            for _ in range(int(9 * dens)):
                L = rr.u(0.12, 0.2)
                y = rr.u(y_lo + 0.01, max(y_lo + 0.011, ylf - 0.01))
                add("droop", C @ _T((sx * (w / 2 + rr.u(-0.012, 0.02)), y, zt + rr.u(0.006, 0.02)),
                                    (rr.u(-25, 25), rr.u(-12, 14), (0 if sx > 0 else 180) + rr.u(-35, 35)), L))
        for _ in range(int(45 * dens)):                   # fallen on the crate below / on the paving
            L = rr.u(0.09, 0.2)
            pl = C @ Vector((rr.u(-w / 2 - 0.14, w / 2 + 0.14), -d / 2 - rr.u(0.03, 0.42), 0))
            yaw = rr.u(0, 360)
            dx, dy = L / 2 * cos(radians(yaw)), L / 2 * sin(radians(yaw))
            zs = [surface_z(pl.x + k * dx, pl.y + k * dy) for k in (-1, 0, 1)]
            if None in zs or max(zs) - min(zs) > 1e-4:
                continue
            kd = rr.pick(["arc", "flat"])
            Mf = _T((pl.x, pl.y, zs[0] + 0.0028), (90 * rr.pick([-1, 1]), 0, yaw), L)
            add(kd, Mf)
            fallen.append((Vector((pl.x - dx, pl.y - dy, zs[0])), Vector((pl.x + dx, pl.y + dy, zs[0]))))

        # wine bottles bedded in the straw
        strip = ylf - (-d / 2 + t)
        if strip > 0.085:
            nbt = max(1, min(4, int((wi - 0.04) / 0.1)))
            prof = [(0.0, 0.0), (0.033, 0.0), (0.037, 0.005), (0.037, 0.15), (0.034, 0.175), (0.022, 0.205), (0.0135, 0.228), (0.0125, 0.268),
                    (0.0155, 0.27), (0.0155, 0.281), (0.011, 0.283), (0.0, 0.283)]
            for k in range(nbt):
                if rr.chance(0.25) and nbt > 2:
                    continue
                x = -wi / 2 + 0.05 + (wi - 0.1) * ((k + 0.5) / nbt) + rr.u(-0.012, 0.012)
                y = -d / 2 + t + rr.u(0.045, max(0.046, strip - 0.045))
                top = mz(x, y) + rr.u(0.085, 0.13)
                bs = min(1.0, (top - sk - t - 0.004) / 0.283)
                Mb = C @ _T((x, y, top - 0.283 * bs), (rr.u(-11, 11), rr.u(-11, 11), rr.u(0, 360)), bs)
                if far:                                    # three-sided: shoulder and neck above the straw
                    vs = [(rb * cos(a_), rb * sin(a_), zb_) for zb_, rb in ((0.10, 0.047), (0.212, 0.027)) for a_ in (0.5, 2.6, 4.7)]
                    o = uv_box(mesh_object("bottle", vs + [(0.0, 0.0, 0.289)],
                                           [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5), (3, 4, 6), (4, 5, 6), (5, 3, 6)]))
                    assign(o, glass)
                    _put(o, Mb)
                    continue
                o = lathe("bottle", prof, segments=12)
                assign(o, glass)
                _put(o, Mb)
                ck = cylinder("cork", 0.0098, 0.014, loc=(0, 0, 0.287), verts=8, bevel=0.0015, segments=1)
                assign(ck, endg)
                _put(ck, Mb @ _T((0, 0, 0.287)))

    # ------------------------------------------------------------------ crowbar
    def crowbar(C, F, rr, xb, h, zc):
        rad, L, f = 0.0105, 0.6, rr.u(0.17, 0.24)
        foot = Vector((0.004, zc + f))                    # (up, out) in the plane of the lean
        ytop = (L * L - f * f) ** 0.5
        if ytop + rad < h - 0.02:
            tip = Vector((ytop, zc + rad + 0.001))
        else:                                             # taller than the crate: rests on the lid arris
            tip = Vector((h + rad * 0.75, 0.006 + rad * 0.75))
        u = (tip - foot).normalized()
        end = foot + u * L
        nn = Vector((-u.y, u.x))
        if nn.y < 0:
            nn = -nn
        R = 0.024
        cen = end + nn * R
        pts2 = [foot, foot + u * (L * 0.5), end - u * 0.03]
        for k in range(0, 6):
            a = radians(k * 19.0)
            pts2.append(cen + (-nn * cos(a) + u * sin(a)) * R)
        a = radians(95.0)
        tan_ = (nn * sin(a) + u * cos(a)).normalized()
        Mw = C @ F
        path = [tuple(Mw @ Vector((xb, q.x, q.y))) for q in pts2]
        if far:                                            # a three-sided bar, the gooseneck as one short leg
            rf = rad * 1.25
            o = sweep("crowbar", [path[0], path[2], path[-1]], [(rf, 0.0), (-rf / 2, rf * 0.866), (-rf / 2, -rf * 0.866)], smooth_path=False)
            assign(o, iron)
            return
        o = sweep("crowbar", path, rad, sides=6, smooth_path=True,
                  scale_fn=lambda s: (0.45 + 0.55 * smoothstep(0.0, 0.1, s)) * (1.0 - 0.5 * smoothstep(0.9, 1.0, s)))
        bar = [o]
        # forged flats: chisel toe on the paving and the split claw at the gooseneck
        for (cp, dr, ln, wd) in ((foot + u * 0.026, u, 0.062, 0.027), (pts2[-1] + tan_ * 0.024, tan_, 0.07, 0.036)):
            M3 = Matrix(((0.0, 1.0, 0.0, xb), (dr.x, 0.0, dr.y, cp.x), (dr.y, 0.0, -dr.x, cp.y), (0.0, 0.0, 0.0, 1.0)))
            q = box("claw", (ln, wd, 0.0075), bevel=0.002, segments=1)
            _put(q, Mw @ M3)
            bar.append(q)
        for q in bar:
            assign(q, iron)
            paint(q, wear=0.6, grime=0.15)

    for i, c in enumerate(crates):
        crate(c, i)

    # ------------------------------------------------------------------ stamped small parts
    if far:
        # straw: a flat pointed blade for every second strand that hangs over the rim or lies in the open case (twice
        # as wide), one for every fallen straw (they touch one another in lod 0, a thinned set would float in pieces);
        # nails are left out
        up = Vector((0, 0, 1))
        vs, fs = [], []

        def blade(a, b, hw, both=False):
            sd_ = up.cross(b - a)
            if sd_.length < 1e-6:
                return
            sd_ = sd_.normalized() * hw
            nv = len(vs)
            vs.extend([tuple(a + sd_), tuple(a - sd_), tuple(b)])
            fs.append((nv, nv + 1, nv + 2))
            if both:                                        # hangs free or lies tilted: seen from behind and below as well
                vs.extend([tuple(a + sd_), tuple(a - sd_), tuple(b)])
                fs.append((nv + 4, nv + 3, nv + 5))

        dr = strands.get("droop", [])
        tips = [M @ Vector((0.5, 0.0, -0.34)) for M in dr]
        keep = set(range(0, len(dr), 2))
        if dr:
            ks = range(len(dr))
            keep |= {min(ks, key=lambda k: tips[k].x), max(ks, key=lambda k: tips[k].x),
                     min(ks, key=lambda k: tips[k].y), max(ks, key=lambda k: tips[k].y)}
        for k in sorted(keep):
            blade(dr[k] @ Vector((-0.12, 0.0, 0.03)), tips[k], 0.009, True)
        for k in range(0, len(lying), 2):
            a, b, L = lying[k]
            blade(a, b, 0.048 * L, True)
        lift = Vector((0, 0, 0.006))
        for a, b in fallen:
            blade(a + lift, b + lift, 0.024 * (b - a).length)
        if fs:
            assign(uv_box(mesh_object("straw", vs, fs)), rope)
        return
    if nails:
        rn, hh = 0.0042, 0.0018
        vs = [(rn * cos(k * pi / 3), rn * sin(k * pi / 3), 0.0) for k in range(6)] + [(0.0, 0.0, hh)]
        nb_ = mesh_object("nail", vs, [(k, (k + 1) % 6, 6) for k in range(6)])
        assign(nb_, iron)
        instances_to_mesh("nails", nb_, nails, keep_base=False)
    shapes = {"arc": [(-0.5, 0, 0), (-0.25, 0, 0.05), (0, 0, 0.07), (0.25, 0, 0.05), (0.5, 0, 0)],
              "droop": [(-0.5, 0, 0.02), (-0.2, 0, 0.03), (0.1, 0, 0.0), (0.32, 0, -0.13), (0.5, 0, -0.34)],
              "kink": [(-0.5, 0, 0), (-0.1, 0, 0.04), (0.06, 0.03, 0.03), (0.5, 0.1, -0.04)],
              "flat": [(-0.5, 0, 0), (0, 0.02, 0.008), (0.5, 0, 0)]}
    for k, Ms in strands.items():
        b = sweep("strand_" + k, shapes[k], 0.012, sides=3, smooth_path=True, step=0.34, cap=False)
        assign(b, rope)
        instances_to_mesh("straw_" + k, b, Ms, keep_base=False, seed=p.seed)
    for o in inked:
        paint(o, wear=mask_mul(noise_mask(0.035, seed=p.seed + 2, lo=0.45, hi=0.75), 0.15 + 0.4 * age), special=0.15 + 0.4 * age)
    for o in ropes:
        paint(o, grime=0.25 + 0.3 * age)
