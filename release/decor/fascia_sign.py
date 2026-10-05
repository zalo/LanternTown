"""Painted timber shop fascia: horizontal V-jointed boards (butt-jointed over wall battens, one of them a later
replacement) inside a mitred moulded frame, a dentilled cornice dressed with lead (drip edge, wall upstand,
lap rolls), a sill bead, scrolled console brackets carrying panelled, moulded-capped console boxes at each
end, individually cut chamfered letters in gold leaf with a signwriter's gilt line, an optional segmental
pediment tablet and optional vitreous-enamel swan-neck sign lamps fixed to the wall above, washing the board.
Wall plane y = 0, origin at the bottom centre; the fascia itself fills width x height (lamps sit above).
"""
from tk import *
from math import sin, cos, pi, radians, asin, sqrt
import numpy as np

META = {"name": "Painted fascia sign board", "class": "component", "mount": "wall", "mount_height": 2.85,
        "focus": [0.0, -0.05, 0.36]}
PARAMS = {
    "text": {"default": "Luggage", "desc": "sign lettering"},
    "width": {"default": 3.0, "min": 1.2, "max": 6.0, "desc": "overall width (m)"},
    "height": {"default": 0.62, "min": 0.4, "max": 1.0, "desc": "overall height of the fascia (m)"},
    "caps": {"default": True, "desc": "set the lettering in capitals"},
    "subtitle": {"default": "", "desc": "small gilt text in the pediment tablet (pediment frame only; empty = sunburst)"},
    "font": {"default": "serif", "choices": ["serif", "roman", "sans", "sans_narrow", "script", "noto_serif"], "desc": "letter face"},
    "board": {"default": "#6e1a18", "desc": "board (ground) paint colour"},
    "trim": {"default": "#24473a", "desc": "frame / cornice / console paint colour"},
    "letter": {"default": "#d4a84b", "desc": "letter colour (gold leaf tone when gilt)"},
    "gilt": {"default": True, "desc": "letters and ornaments in gold leaf (False = painted in the letter colour)"},
    "frame": {"default": "moulded", "choices": ["moulded", "plain", "pediment"], "desc": "frame style"},
    "consoles": {"default": True, "desc": "scrolled console brackets and capped console boxes at the ends"},
    "lamps": {"default": 2, "min": 0, "max": 3, "desc": "number of swan-neck sign lamps above"},
    "shade": {"default": "#16362b", "desc": "lamp shade enamel colour"},
    "lamp_reach": {"default": 0.5, "min": 0.3, "max": 0.6, "desc": "how far the lamps stand off the wall (m)"},
    "intensity": {"default": 2.6, "min": 0.0, "max": 30.0, "desc": "light per lamp (cd); dimmed automatically for pale paint"},
    "age": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "weathering: rain streaks, flaking paint, sun fade"},
}
VARIANTS = {
    "accessories_long": {"text": "Accessories", "width": 4.8, "height": 0.72, "board": "#1b2c46", "trim": "#cfc3a2",
                         "lamps": 3, "font": "roman", "age": 0.3, "shade": "#14161a", "seed": 4},
    "dress_suit_pediment": {"text": "Dress Suit", "caps": False, "font": "script", "frame": "pediment", "subtitle": "Est. 1887",
                            "width": 3.6, "height": 0.95, "board": "#15171b", "trim": "#5a1616", "lamps": 2, "shade": "#14161a", "seed": 11},
    "items_plain": {"text": "Items", "width": 1.7, "height": 0.46, "frame": "plain", "consoles": False, "lamps": 1,
                    "board": "#1f4a3a", "trim": "#16191a", "gilt": False, "letter": "#e9dcb8", "font": "sans", "age": 0.6,
                    "shade": "#5a1616", "lamp_reach": 0.42, "seed": 23},
    "old_teal": {"text": "Luggage & Trunks", "width": 4.0, "height": 0.56, "frame": "plain", "board": "#2f5e63", "trim": "#d9cfb4",
                 "gilt": False, "letter": "#f3ead0", "lamps": 0, "age": 1.0, "font": "noto_serif", "seed": 31},
}

# ------------------------------------------------------------------ local helpers
def _arc(cx, cy, r, a0, a1, n):
    return [(cx + r * cos(radians(a0 + (a1 - a0) * i / n)), cy + r * sin(radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def _strip(name, rings, hint=(0, -1, 0)):
    """Quad strip through rings of 3D points (same count per ring), wound so the net normal follows `hint`."""
    n = len(rings[0])
    verts = [tuple(v) for r in rings for v in r]
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n - 1):
            faces.append((i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j))
    V = np.array(verts)
    tot = np.zeros(3)
    for f in faces:
        tot += np.cross(V[f[2]] - V[f[0]], V[f[3]] - V[f[1]])
    if float(tot.dot(np.array(hint))) < 0:
        faces = [f[::-1] for f in faces]
    return mesh_object(name, verts, faces)


def _between(r0, r1, step):
    """Rings from r0 to r1 with intermediate rings every ~step metres (vertex-colour resolution)."""
    a, b = np.array(r0), np.array(r1)
    k = max(1, int(round(float(np.linalg.norm(b[0] - a[0])) / step)))
    return [(a + (b - a) * (i / k)).tolist() for i in range(k + 1)]


def _prism_x(name, poly, x0, x1):
    """Closed solid: polygon [(depth, z)] (depth = distance in front of the wall) extruded from x0 to x1."""
    n = len(poly)
    verts = [(x0, -d, z) for d, z in poly] + [(x1, -d, z) for d, z in poly]
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    o = mesh_object(name, verts, faces)
    recalc_normals(o)
    return o


def _moulding(name, prof, x0, x1, dface, returns, step=0.3):
    """Horizontal moulding of profile [(projection, z)] on the face at depth dface between x0..x1; with
    `returns` it is mitred round both ends and returned to the wall."""
    out = []
    if returns:
        ra = [(x0 - q, 0.0, z) for q, z in prof]
        rb = [(x0 - q, -(dface + q), z) for q, z in prof]
        rc = [(x1 + q, -(dface + q), z) for q, z in prof]
        rd = [(x1 + q, 0.0, z) for q, z in prof]
        f = _strip(name, _between(rb, rc, step))
        l = _strip(name + "_retl", [ra, rb], hint=(-1, 0, 0))
        r = _strip(name + "_retr", [rc, rd], hint=(1, 0, 0))
        uv_grain(l, "Y")
        uv_grain(r, "Y")
        out = [f, l, r]
    else:
        rb = [(x0, -(dface + q), z) for q, z in prof]
        rc = [(x1, -(dface + q), z) for q, z in prof]
        out = [_strip(name, _between(rb, rc, step))]
    uv_grain(out[0], "X")
    for o in out:
        smooth(o, 38)
    return out


def _cornice_profile(style, z0, ch, cp):
    if style == "plain":
        pts = [(0, 0), (0.06, 0), (0.06, 0.10)] + _arc(0.40, 0.10, 0.34, 180, 90, 4)[1:]
        pts += [(0.47, 0.44), (0.47, 0.52), (0.9, 0.52), (0.9, 0.86), (1.0, 0.9), (1.0, 1.0)]
    else:
        pts = [(0, 0), (0.04, 0), (0.04, 0.05), (0.04, 0.30), (0.20, 0.30)] + _arc(0.20, 0.37, 0.07, -90, 0, 2)[1:]
        pts += [(0.27, 0.39), (0.62, 0.39), (0.62, 0.61), (0.66, 0.61), (0.66, 0.65)]
        for i in range(1, 6):
            t = i / 5
            pts.append((0.66 + 0.30 * (t - 0.155 * sin(2 * pi * t)), 0.65 + 0.28 * t))
        pts += [(1.0, 0.93), (1.0, 1.0)]
    return [(q * cp, z0 + z * ch) for q, z in pts]


def _frame_profile(style, fw, ft):
    if style == "plain":
        pts = [(0, 0), (0, 0.6), (0.07, 0.7), (0.66, 0.7), (1.0, 0.36), (1.0, 0)]
    else:
        pts = [(0, 0), (0, 0.8), (0.03, 0.95), (0.08, 1.0), (0.21, 1.0), (0.25, 0.82)]
        for i in range(1, 6):
            t = i / 5
            pts.append((0.25 + 0.6 * t, 0.82 - 0.55 * (t - 0.15 * sin(2 * pi * t))))
        pts += [(0.85, 0.2)] + _arc(0.925, 0.2, 0.075, 180, 0, 3)[1:] + [(1.0, 0)]
    return [(u * fw, v * ft) for u, v in pts]


def _notch_rect(w, h, rc, inset=0.0, n=4):
    """Rectangle with concave quarter-round corners (signwriter's line), CCW; `inset` shrinks it evenly."""
    hw, hh = w / 2, h / 2
    r = rc + inset
    dl = degrees(asin(min(0.99, inset / r))) if inset > 0 else 0.0
    pts = []
    for (cx, cy, a0) in ((hw, -hh, 180), (hw, hh, 270), (-hw, hh, 360), (-hw, -hh, 90)):
        pts += _arc(cx, cy, r, a0 - dl, a0 - 90 + dl, n)
    return pts


def _open_box(name, w, q, h):
    """Dentil block: front, two cheeks and soffit only (back and top are buried in the cornice)."""
    x, y, z = w / 2, q / 2, h / 2
    v = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    return mesh_object(name, v, [(0, 1, 5, 4), (1, 2, 6, 5), (3, 0, 4, 7), (0, 3, 2, 1)])


def _facet_gem(name, pts, h, table=0.38):
    """Faceted boss on the XY plane: outline pts (CCW) rising to a small flat table (prismatic gilt ornament)."""
    n = len(pts)
    v = [(x, y, 0.0) for x, y in pts] + [(x * table, y * table, h) for x, y in pts]
    f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n, 2 * n))]
    return mesh_object(name, v, f)


def build(p, rng):
    W, H = float(p.width), float(p.height)
    style = p.frame
    ped = style == "pediment"
    cstyle = "plain" if style == "plain" else "moulded"
    cons = bool(p.consoles)
    age = float(p.age)
    sd = int(p.seed)
    old = clamp((age - 0.35) / 0.65)                 # 0 below a cared-for sign, 1 = long neglected

    # sun and rain chalk the paint: the tint itself shifts with age (the vertex masks add the local stories)
    board_col = _mix(p.board, "#a9a69a", 0.36 * old ** 1.3)
    trim_col = _mix(p.trim, "#a39f93", 0.22 * old ** 1.3)
    m_board = mat("wood_painted", board_col, rough=0.8 + 0.45 * age, scale=3.4)
    m_board_new = mat("wood_painted", _mix(p.board, "#ffffff", 0.05), rough=0.5, scale=3.2)   # the replaced board: fresher, glossier
    m_trim = mat("wood_painted", trim_col, rough=0.85 + 0.35 * age, scale=2.4)
    m_gilt = mat("metal_plain", p.letter, rough=0.4) if p.gilt else mat("wood_painted", p.letter, rough=0.8, scale=3.2)
    m_lead = mat("lead")
    m_dark = mat("wood_painted", _mix(p.board, "#050403", 0.72))          # blocking shadow / primed backs and battens

    # ---- layout (depths are distances in front of the wall)
    d_bat, d_b = 0.02, 0.045                         # batten face, board face
    d_c = d_b + 0.012                                # cornice / sill bed face
    rise = clamp(0.3 * H, 0.14, 0.27) if ped else 0.0
    cap_up = 0.046 if cons else 0.03
    Hk = H - max(rise, cap_up)                       # top of the cornice
    ch = clamp(0.2 * H, 0.08, 0.15)
    cp = ch * (0.6 if cstyle == "plain" else 0.88)
    Hc = Hk - ch                                     # cornice bed = top of the sign panel
    hb = clamp(0.075 * H, 0.032, 0.06)               # sill bead height
    bp = hb * 0.75
    cw = clamp(0.26 * H, 0.12, 0.2) if cons else 0.0
    capov = 0.024
    xi = (W / 2 - cw) if cons else (W / 2 - cp - 0.004)   # half width of the panel / cornice run
    dbx = d_c + cp + 0.014                           # console box face
    ph = Hc - hb
    fw = clamp(0.12 * ph, 0.034, 0.062)
    ft = clamp(0.5 * fw, 0.018, 0.03)
    trim, low_trim, corn_trim, gilt_objs = [], [], [], []
    fine = 0.12 if age > 0.5 else 0.6                # weathered signs get vertex resolution where the paint fails

    # ---- wall battens (the grounds the fascia is nailed to) - seen from behind / at the ends
    nbat = max(3, int(round(2 * xi / 0.55)) + 1)
    rb = rng.fork("battens")
    bat_x = []
    for i in range(nbat):
        x = -xi + 0.06 + (2 * xi - 0.12) * i / (nbat - 1) + rb.jitter(0, 0.015)
        b = box("batten", (0.045, d_bat, Hk - 0.03), loc=(x, -d_bat / 2, (Hk - 0.03) / 2 + 0.01))
        uv_grain(b, "Z")
        assign(b, m_dark)
        bat_x.append(x)

    # ---- board: horizontal V-jointed planks, butt-jointed over a batten on long runs; one piece is a later repair
    xb = xi if cons else xi - 0.018
    nb = max(2, int(round(ph / 0.15)))
    bh = ph / nb
    rp = rng.fork("planks")
    boards, new_boards = [], []
    ncol = max(4, int(round(2 * xb / clamp(W / 55, 0.05, 0.085))))
    xcol = [-xb + 2 * xb * j / ncol for j in range(ncol + 1)]
    bk = _strip("board_back", [[(-xb, -d_bat, hb), (-xb, -d_bat, Hc)], [(xb, -d_bat, hb), (xb, -d_bat, Hc)]], hint=(0, 1, 0))
    bk2 = _strip("board_back", [[(-xb, -d_bat - 0.001, hb), (-xb, -d_bat - 0.001, Hc)], [(xb, -d_bat - 0.001, hb), (xb, -d_bat - 0.001, Hc)]])
    for q in (bk, bk2):
        uv_grain(q, "X")
        assign(q, m_dark)
    odd_row = int(rp.u(0, nb - 1e-6)) if age > 0.12 else -1
    odd_side = 0 if rp.chance(0.5) else 1
    jprev = -1
    c = 0.003
    for i in range(nb):
        za, zb_ = hb + bh * i + 0.0006, hb + bh * (i + 1) - 0.0006
        # the face is a real grid (not two long triangles) so the baked contact shadow round the raised
        # letters, the rain streaks and the flaking have vertices to live on; top / bottom rows fold back as the V-joint
        nseg = max(2, int(round((zb_ - za - 2 * c) / 0.065)))
        jc = -1
        if 2 * xb > 2.3:
            cand = [j for j in (min(range(ncol + 1), key=lambda j_: abs(xcol[j_] - x)) for x in bat_x)
                    if 3 <= j <= ncol - 3 and abs(xcol[j]) < 0.6 * xb and j != jprev]
            if cand:
                jc = rp.pick(cand)
        jprev = jc
        segs = [(0, ncol)] if jc < 0 else [(0, jc), (jc, ncol)]
        for k, (j0, j1) in enumerate(segs):
            is_new = i == odd_row and (jc < 0 or k == odd_side)
            df = d_b + rp.jitter(0, 0.0005) - (0.0007 if is_new else 0.0)      # boards never sit perfectly flush
            rows = [(df - c, za)] + [(df, za + c + (zb_ - za - 2 * c) * q / nseg) for q in range(nseg + 1)] + [(df - c, zb_)]
            rings = []
            for j in range(j0, j1 + 1):
                x = xcol[j]
                if jc >= 0 and j == jc:                                          # eased butt end, 2 mm open joint
                    sgn = -1 if k == 0 else 1
                    end = [(x + sgn * 0.0004, -(d - c), z) for d, z in rows]
                    face = [(x + sgn * 0.003, -d, z) for d, z in rows]
                    rings += [face, end] if k == 0 else [end, face]
                else:
                    rings.append([(x, -d, z) for d, z in rows])
            o = _strip("board", rings)
            uv_grain(o, "X")
            assign(o, m_board_new if is_new else m_board)
            (new_boards if is_new else boards).append(o)
    # nail heads where every board is fixed to the battens (painted over)
    nail = mesh_object("nail", [(0.0036, 0, 0), (0, 0.0036, 0), (-0.0036, 0, 0), (0, -0.0036, 0), (0, 0, 0.0015)], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])
    nm_ = []
    for x in bat_x:
        if abs(x) > xb - fw - 0.02:
            continue
        for i in range(nb):
            for fz in (0.22, 0.78):
                nm_.append(trs((x + rp.jitter(0, 0.006), -(d_b + 0.0012), hb + bh * (i + fz) + rp.jitter(0, 0.006)), (90, 0, 0)))
    nails = instances_to_mesh("nails", nail, nm_, keep_base=False, uv_jitter=False) if nm_ else None
    if nails:
        assign(nails, m_board)
        paint(nails, wear=0.5 * age, grime=0.6)
    else:
        delete([nail])
    if not cons:
        for sx in (-1, 1):
            e = box("end_stile", (0.018, d_b + 0.004, ph), loc=(sx * (xi - 0.009), -(d_b + 0.004) / 2, hb + ph / 2), bevel=0.003)
            uv_grain(e, "Z")
            trim.append(e)

    # ---- mitred frame round the panel
    fprof = _frame_profile(cstyle, fw, ft)
    ox, z0f, z1f = xb, hb, Hc

    def corner(sx, sz):
        return [(sx * (ox - u), -(d_b + v), (z0f + u) if sz < 0 else (z1f - u)) for u, v in fprof]

    bl, br, tr_, tl = corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)
    for nm, a, b_, ax in (("frame_b", bl, br, "X"), ("frame_r", br, tr_, "Z"), ("frame_t", tr_, tl, "X"), ("frame_l", tl, bl, "Z")):
        o = _strip(nm, _between(a, b_, fine if nm == "frame_b" else max(fine, 0.4)))
        uv_grain(o, ax)
        smooth(o, 38)
        (low_trim if nm == "frame_b" else trim).append(o)

    # ---- cornice: backing, moulded run, dentils, lead dressing
    core = box("cornice_core", (2 * xi, d_c, ch - 0.002), loc=(0, -d_c / 2, Hc + ch / 2 - 0.001))
    uv_grain(core, "X")
    trim.append(core)
    cprof = _cornice_profile(cstyle, Hc, ch, cp)
    corn_trim += _moulding("cornice", cprof, -xi, xi, d_c, returns=not cons, step=min(0.6, fine * 1.5))
    if cstyle != "plain":
        dw = clamp(0.24 * ch, 0.022, 0.034)
        nd = max(3, int((2 * xi - dw) / (dw * 2.15)))
        pitch = (2 * xi - dw) / nd
        dz, dq = 0.2 * ch, 0.15 * cp
        dent = _open_box("dentil", dw, dq + 0.004, dz)
        rd = rng.fork("dentils")
        mats = []
        for i in range(nd + 1):
            if 0 < i < nd and rd.chance(0.07 * old):            # the odd dentil has dropped off an old sign
                continue
            mats.append(trs((-xi + dw / 2 + i * pitch, -(d_c + 0.04 * cp + dq / 2 - 0.002), Hc + 0.185 * ch + rd.jitter(0, 0.0012 * (0.3 + age)))))
        trim.append(instances_to_mesh("dentils", dent, mats, keep_base=False))
    Dtop = d_c + cp
    lx = xi if cons else W / 2
    # lead: turned up the wall behind, falling to the front, dressed down over the crown fillet with a drip
    drop = (0.10 if cstyle == "plain" else 0.07) * ch + 0.003
    lead = _prism_x("lead", [(0, Hk), (Dtop, Hk), (Dtop, Hk - drop), (Dtop + 0.0045, Hk - drop - 0.003), (Dtop + 0.007, Hk - drop + 0.001),
                             (Dtop + 0.007, Hk + 0.001), (Dtop + 0.003, Hk + 0.005), (0.007, Hk + 0.012), (0.004, Hk + 0.028), (0, Hk + 0.028)], -lx, lx)
    densify(lead, 0.26)
    displace_noise(lead, strength=0.001, scale=0.14, seed=sd, axis_mask=(0, 0.3, 1))
    transform_verts(lead, lambda co: Vector((co.x, min(co.y, 0.0), co.z)))
    smooth(lead, 50)
    leads = [lead]
    rl_ = rng.fork("lead")
    nlap = max(1, int(round(2 * lx / 1.4)) - 1)
    for i in range(nlap):                                    # wood-cored rolls where the sheets lap
        x = -lx + 2 * lx * (i + 1) / (nlap + 1) + rl_.jitter(0, 0.12)
        if ped and abs(x) < 0.5 * min(clamp(0.44 * W, 0.7, 2.0), 2 * xi - 0.1) + 0.03:
            continue
        leads.append(box("lead_roll", (0.024, Dtop + 0.011, 0.013), loc=(x, -(Dtop + 0.011) / 2, Hk + 0.0085), rot=(-2.5, 0, rl_.jitter(0, 1.5)), bevel=0.004, segments=1))
    nm_lead = noise_mask(0.2, seed=sd + 5, lo=0.3, hi=0.8)
    for o in leads:
        assign(o, m_lead)
        paint(o, grime=lambda co, n: np.clip(0.16 + 0.34 * nm_lead(co, n) + 0.5 * (n[:, 1] < -0.5), 0.0, 1.0))

    # ---- sill bead under the panel
    sprof = [(0, 0), (0.45 * bp, 0)] + _arc(0.45 * bp, 0.5 * hb, 0.5 * hb, -90, 0, 3)[1:]
    sprof = [(q * (bp / (0.45 * bp + 0.5 * hb)), z) for q, z in sprof]
    sprof += [(bp, 0.72 * hb), (bp - 0.004, 0.8 * hb), (0.3 * bp, hb + 0.002), (0, hb + 0.004)]
    low_trim += _moulding("sill", sprof, -xi, xi, d_b, returns=not cons, step=fine)
    score = box("sill_core", (2 * xi, d_b, hb), loc=(0, -d_b / 2, hb / 2))
    uv_grain(score, "X")
    trim.append(score)

    # ---- console ends: scrolled bracket carrying a panelled, capped console box the cornice dies into
    if cons:
        zb = Hc - 0.5 * hb                            # underside of the console box
        Hs = zb
        bx0, bx1 = xi, W / 2 - capov
        bw = bx1 - bx0
        D1 = dbx - 0.008
        r1 = min(0.24 * Hs, (D1 - d_b) * 0.5)
        D2 = d_b + 0.55 * (D1 - d_b)
        r2 = min(0.13 * Hs, (D2 - 0.02) * 0.5)
        c1, c2 = D1 - r1, D2 - r2
        z1c, z2c = Hs - r1, r2 + 0.004
        if cstyle == "plain":
            outline = [(0, 0.004), (d_b + 0.012, 0.004), (d_b + 0.012, 0.03)] + \
                      bezier2((d_b + 0.012, 0.03), (d_b + 0.02, Hs * 0.55), (D1 - 0.03, Hs * 0.62), (D1, Hs * 0.78), 8)[1:] + [(D1, Hs), (0, Hs)]
            bead = None
        else:
            arc_b = _arc(c2, z2c, r2, -90, 60, 5)
            arc_t = _arc(c1, z1c, r1, -60, 90, 6)
            P0, P3 = arc_b[-1], arc_t[0]
            k = 0.42 * sqrt((P3[0] - P0[0]) ** 2 + (P3[1] - P0[1]) ** 2)
            P1 = (P0[0] - sin(radians(60)) * k, P0[1] + cos(radians(60)) * k)
            P2 = (P3[0] - sin(radians(60)) * k, P3[1] - cos(radians(60)) * k)
            neck = bezier2(P0, P1, P2, P3, 6)
            outline = [(0, 0.004)] + arc_b + neck[1:-1] + arc_t + [(0, Hs)]
            # raised S-bead on the cheeks: volute - neck - volute
            ins = 0.009
            bead = []
            for i in range(9):
                t = i / 8
                a = radians(60 - 400 * (1 - t))
                r = (r2 - ins) * (0.22 + 0.78 * t)
                bead.append((c2 + r * cos(a), z2c + r * sin(a)))
            for i in range(1, len(neck) - 1):
                tx, tz = neck[i + 1][0] - neck[i - 1][0], neck[i + 1][1] - neck[i - 1][1]
                L = sqrt(tx * tx + tz * tz)
                bead.append((neck[i][0] - tz / L * ins, neck[i][1] + tx / L * ins))
            for i in range(11):
                t = i / 10
                a = radians(-60 + 450 * t)
                r = (r1 - ins) * (1.0 - 0.8 * t)
                bead.append((c1 + r * cos(a), z1c + r * sin(a)))
        hv = Hk - 0.004 - zb                             # visible height of the box face below its cap
        zf = zb + hv / 2
        for sx in (-1, 1):
            xa, xb_ = sorted((sx * (bx0 + 0.012), sx * (bx1 - 0.012)))
            sc = _prism_x("console", outline, xa, xb_)
            bevel(sc, 0.003, segments=1, angle=50)
            smooth(sc, 40)
            uv_grain(sc, "Z")
            trim.append(sc)
            if bead:
                for xs in (xa - 0.001, xb_ + 0.001):
                    s = sweep("console_bead", [(xs, -d, z) for d, z in bead], 0.0056, sides=3, smooth_path=False,
                              scale_fn=lambda t: 0.55 + 1.8 * min(t, 1 - t, 0.25))
                    trim.append(s)
                # three reeds running down the front of the scroll
                front = arc_b[1:] + neck[1:-1] + arc_t[:-1]
                nf = len(front)
                for k3 in (-1, 0, 1):
                    xr = (xa + xb_) / 2 + k3 * (xb_ - xa) * 0.27
                    rings = []
                    for q in range(nf):
                        a_, b2 = front[max(q - 1, 0)], front[min(q + 1, nf - 1)]
                        td, tz = b2[0] - a_[0], b2[1] - a_[1]
                        L = sqrt(td * td + tz * tz)
                        nd_, nz_ = tz / L, -td / L              # outward normal of the scroll face
                        t = q / (nf - 1)
                        sc_ = 0.3 + 2.8 * min(t, 1 - t, 0.25)
                        d0, z0r = front[q][0] - 0.0008 * nd_, front[q][1] - 0.0008 * nz_
                        rings.append([(xr - 0.0085 * sc_, -d0, z0r), (xr, -(front[q][0] + nd_ * 0.0062 * sc_), front[q][1] + nz_ * 0.0062 * sc_),
                                      (xr + 0.0085 * sc_, -d0, z0r)])
                    rd_ = _strip("console_reed", rings)
                    uv_grain(rd_, "Z")
                    trim.append(rd_)
            # console box: body, planted panel mould round a sunk field, moulded timber cap under a lead weathering
            xc = sx * (bx0 + bx1) / 2
            bxo = box("console_box", (bw, dbx, Hk + 0.006 - zb), loc=(xc, -dbx / 2, (zb + Hk + 0.006) / 2), bevel=0.003)
            densify(bxo, 0.1)
            uv_grain(bxo, "Z")
            trim.append(bxo)
            ow, oh = bw / 2 - 0.009, hv / 2 - 0.009
            pm = clamp(0.11 * min(bw, hv), 0.01, 0.02)
            pan = shape_extrude("console_panel", [(-ow, -oh), (ow, -oh), (ow, oh), (-ow, oh)],
                                [[(-ow + pm, -oh + pm), (ow - pm, -oh + pm), (ow - pm, oh - pm), (-ow + pm, oh - pm)]],
                                depth=0.007, loc=(xc, -dbx, zf), rot=(90, 0, 0), bevel=0.0025)
            uv_grain(pan, "X")
            trim.append(pan)
            zc0 = Hk - 0.004
            cprof2 = [(0, 0), (0.004, 0), (0.004, 0.005), (0.006, 0.011), (0.011, 0.016), (0.017, 0.019), (0.020, 0.022), (0.022, 0.026), (0.022, 0.035)]
            cprof2 = [(q, zc0 + z) for q, z in cprof2]
            trim += _moulding("console_cap", cprof2, xc - bw / 2, xc + bw / 2, dbx, returns=True)
            ztop = zc0 + 0.035
            topb = _prism_x("console_top", [(0, ztop - 0.002), (dbx + 0.024, ztop - 0.002), (dbx + 0.026, ztop + 0.003), (dbx * 0.4, ztop + 0.011), (0, ztop + 0.015)],
                            xc - bw / 2 - 0.024, xc + bw / 2 + 0.024)
            assign(topb, m_lead)
            paint(topb, grime=_add(_mul(noise_mask(0.1, seed=sd + 9, lo=0.3, hi=0.8), 0.4), 0.14))
            # carved patera in the sunk field (plain frames get a faceted stud)
            pr = max(0.012, min(bw, hv) / 2 - 0.009 - pm - 0.006)
            if cstyle == "plain":
                g = _facet_gem("stud", [(pr * 0.8, 0), (0, pr * 0.8), (-pr * 0.8, 0), (0, -pr * 0.8)], pr * 0.4, table=0.3)
                set_loc_rot(g, loc=(xc, -dbx, zf), rot=(90, 0, 0))
                gilt_objs.append(g)
            else:
                gilt_objs.append(lathe("patera", [(pr, 0.0), (pr * 1.0, 0.004), (pr * 0.93, 0.0075), (pr * 0.85, 0.004), (pr * 0.84, 0.0)],
                                       segments=12, loc=(xc, -dbx, zf), rot=(90, 0, 0)))
                gilt_objs.append(lathe("patera_boss", [(pr * 0.3, 0.0), (pr * 0.3, 0.004), (pr * 0.22, 0.011), (pr * 0.1, 0.0145), (0.0, 0.0155)],
                                       segments=8, loc=(xc, -dbx, zf), rot=(90, 0, 0)))
                pv, pf = [], []
                for q in range(8):                        # eight raised petals, each a ridged leaf
                    a = radians(22.5 + 45 * q)
                    ca, sa = cos(a), sin(a)
                    leaf = [(0.3, 0.0, 0.002), (0.56, -0.16, 0.0), (0.84, 0.0, 0.001), (0.56, 0.16, 0.0), (0.56, 0.0, 0.0075)]
                    b0 = len(pv)
                    pv += [(pr * (u * ca - v * sa), pr * (u * sa + v * ca), w) for u, v, w in leaf]
                    pf += [(b0, b0 + 1, b0 + 4), (b0 + 1, b0 + 2, b0 + 4), (b0 + 2, b0 + 3, b0 + 4), (b0 + 3, b0, b0 + 4)]
                pet = mesh_object("patera_petals", pv, pf)
                set_loc_rot(pet, loc=(xc, -dbx, zf), rot=(90, 0, 0))
                gilt_objs.append(pet)

    # ---- pediment tablet with lead-capped segmental top
    wp = 0.0
    if ped:
        wp = min(clamp(0.44 * W, 0.7, 2.0), 2 * xi - 0.1)
        R = (wp * wp / 4 + rise * rise) / (2 * rise)
        zc = H - R
        ph0 = asin(wp / 2 / R)
        Rt = R - 0.016
        top = [(Rt * sin(ph0 - 2 * ph0 * i / 16), zc + Rt * cos(ph0 - 2 * ph0 * i / 16)) for i in range(17)]
        outer = [(-wp / 2 + 0.01, Hk - 0.002), (wp / 2 - 0.01, Hk - 0.002)] + [q for q in top if q[1] > Hk]
        tab = shape_extrude("tablet", outer, depth=d_c, loc=(0, 0, 0), rot=(90, 0, 0), bevel=0.0)
        uv_grain(tab, "X")
        trim.append(tab)
        Rm = R - 0.011
        path = [(Rm * sin(-ph0 + 2 * ph0 * i / 18), -(d_c + 0.03) / 2, zc + Rm * cos(-ph0 + 2 * ph0 * i / 18)) for i in range(19)]
        capm = sweep("tablet_cap", path, rounded_rect(0.022, d_c + 0.03, 0.006, 2), smooth_path=False, up=(0, -1, 0))
        trim.append(capm)
        # inner bead following the arch
        Rb = R - 0.05
        phb = asin(min(0.999, (wp / 2 - 0.06) / Rb))
        zlo = Hk + 0.02
        bpath = [(Rb * sin(-phb + 2 * phb * i / 16), -d_c, zc + Rb * cos(-phb + 2 * phb * i / 16)) for i in range(17)]
        bpath = [q for q in bpath if q[2] > zlo]
        if len(bpath) > 3:
            gilt_objs.append(sweep("tablet_bead", bpath, 0.005, sides=5, smooth_path=False))
        if str(p.subtitle).strip():
            st = _fit_text("subtitle", str(p.subtitle), "serif", wp * 0.55, rise * 0.34, 0.005, (0, -d_c, Hk + rise * 0.42), res=18.0)
            gilt_objs.append(st)
        else:
            oz = (Hk + 0.012) - zc
            ray = _facet_gem("ray", [(-0.5, -0.006), (0.5, -0.004), (0.5, 0.004), (-0.5, 0.006)], 0.005, table=0.45)
            mats = []
            for i in range(11):
                a = radians(15 + 15 * i)
                sa = sin(a)
                Rr = R - 0.07
                t = -oz * sa + sqrt(max(1e-6, (oz * sa) ** 2 - (oz * oz - Rr * Rr)))
                r0 = 0.05
                L = max(0.02, (t - r0) * (1.0 if i % 2 == 0 else 0.72))
                cr = r0 + L / 2
                mats.append(trs((cr * cos(a), -d_c, Hk + 0.012 + cr * sa), (90, 0, 0), 1.0) @ trs((0, 0, 0), (0, 0, degrees(a)), (L, 1.0, 1.0)))
            gilt_objs.append(instances_to_mesh("sunburst", ray, mats, keep_base=False))
            gilt_objs.append(lathe("sun", [(0.0, 0.0), (0.04, 0.0), (0.042, 0.004), (0.03, 0.009), (0.0, 0.012)], segments=16,
                                   loc=(0, -d_c, Hk + 0.012), rot=(90, 0, 0), angle=360.0))

    # ---- lettering, gilt line and ornaments
    ix, iz0, iz1 = xb - fw, hb + fw, Hc - fw
    ih = iz1 - iz0
    cz = (iz0 + iz1) / 2
    li = clamp(0.075 * ih, 0.011, 0.026)             # line inset from the frame
    lw = clamp(0.03 * ih, 0.005, 0.009)
    rc = clamp(0.15 * ih, 0.02, 0.048)
    lwid, lhei = 2 * (ix - li), ih - 2 * li
    line = shape_extrude("gilt_line", _notch_rect(lwid, lhei, rc), [_notch_rect(lwid, lhei, rc, inset=lw)], depth=0.0014,
                         loc=(0, -(d_b + 0.0006), cz), rot=(90, 0, 0))
    body = str(p.text).strip() or " "
    if p.caps:
        body = body.upper()
    upper = body == body.upper()
    ldepth = clamp(0.06 * ih, 0.008, 0.018)
    max_w = lwid - 2 * lw - 2 * rc - 0.03
    # the word is set out to fill the board: letter-spaced and drawn a little extended (or condensed) like a signwriter would
    letters, sh = _fit_text("letters", body, p.font, max_w, ih * (0.6 if upper else 0.74), ldepth, (0, -d_b, cz), fill_w=0.66 * 2 * ix,
                            shadow=True, rng=rng.fork("letters"), tilt=0.9 + 1.5 * (0.4 + 0.6 * age), chamfer=clamp(0.02 * ih, 0.002, 0.0038), res=9.0 if len(body) <= 8 else (11.5 if len(body) <= 11 else 13.0))
    mn, mx = world_bbox([letters])
    # painted drop shadow (signwriter's blocking) behind the raised letters
    off = 0.05 * (mx.z - mn.z)
    set_loc_rot(sh, loc=(letters.location.x + off, letters.location.y - 0.0009, letters.location.z - off))
    assign(sh, m_dark)
    # lozenge-and-dots ornaments where there is room either side of the word - they shrink with the gap and go when it closes
    gap = (lwid / 2 - lw - rc * 0.4) - (mx.x - mn.x) / 2
    if gap > 0.09:
        s = min(clamp(0.26 * ih, 0.03, 0.085), 0.5 * gap)
        dia = _facet_gem("lozenge", [(0.36 * s, 0), (0, 0.5 * s), (-0.36 * s, 0), (0, -0.5 * s)], ldepth * 0.55)
        dot = lathe("dot", [(0.0, 0.0), (0.15 * s, 0.0), (0.15 * s, 0.002), (0.09 * s, ldepth * 0.42), (0.0, ldepth * 0.5)], segments=6)
        xo = (mx.x - mn.x) / 2 + gap * 0.5
        dm, om = [], []
        for sx in (-1, 1):
            dm.append(trs((sx * xo, -d_b, cz), (90, 0, 0)))
            if gap > 0.3:
                for k2 in (-1, 1):
                    om.append(trs((sx * xo + k2 * min(0.5 * s + 0.05, gap * 0.3), -d_b, cz), (90, 0, 0)))
        gilt_objs.append(instances_to_mesh("lozenges", dia, dm, keep_base=False, uv_jitter=False))
        if om:
            gilt_objs.append(instances_to_mesh("dots", dot, om, keep_base=False, uv_jitter=False))
        else:
            delete([dot])

    # gold leaf: bright burnished faces, dark gold-size and dirt where the leaf meets the paint, tarnish breakup
    nm_tarn = noise_mask(0.09, seed=sd + 21, lo=0.4, hi=0.85)
    nm_burn = noise_mask(0.05, seed=sd + 22, lo=0.35, hi=0.75)

    def gild(o, plane, depth):
        assign(o, m_gilt)
        if p.gilt:
            def recess(co, n, plane=plane, depth=depth):
                proud = np.clip((-co[:, 1] - plane) / max(depth, 1e-4), 0.0, 1.0)      # 0 at the board, 1 on the face
                return np.clip((1.0 - proud) ** 1.5 * 0.9 + nm_tarn(co, n) * (0.12 + 0.3 * age), 0.0, 1.0)
            paint(o, grime=recess, wear=_mul(nm_burn, 0.7))
        else:
            paint(o, wear=_mul(noise_mask(0.04, seed=sd + 3, lo=0.5, hi=0.8), 0.7 * age), grime=0.1 * age)

    gild(letters, d_b, ldepth)
    gild(line, d_b - 0.004, 0.004)
    for o in gilt_objs:
        mn_, mx_ = world_bbox([o])
        gild(o, -mx_.y, max(0.004, mx_.y - mn_.y))

    # ---- paint weathering stories
    nm_chip = noise_mask(0.05, seed=sd + 1, lo=0.52, hi=0.78)
    nm_flake = noise_mask(0.17, seed=sd + 4, lo=0.36, hi=0.6)
    nm_fade = noise_mask(0.5, seed=sd + 2, lo=0.25, hi=0.8)
    up = facing((0, 0, 1))

    def rain(co):
        x = co[:, 0]
        return (0.5 + 0.5 * np.sin(x * 61.0 + 2.5 * np.sin(x * 13.7 + sd) + sd * 1.3)) ** 3

    def streaks(co, n):                               # rain off the lead runs down the board; dust lies on the bottom rail
        z = co[:, 2]
        g = np.clip(1.0 - (Hc - z) / max(0.8 * ph, 1e-3), 0.0, 1.0)
        dust = np.clip(1.0 - (z - hb - fw) / max(0.22 * ph, 1e-3), 0.0, 1.0)
        return np.clip(rain(co) * g * (0.12 + 0.75 * age * age) * (0.4 + nm_fade(co, n)) + dust * 0.45 * age, 0.0, 1.0)

    def low_flake(co, n):                             # paint lets go first along the bottom where the wet sits
        g = np.clip(1.0 - (co[:, 2] - hb) / max(0.5 * ph, 1e-3), 0.0, 1.0)
        return np.clip(nm_chip(co, n) * (0.25 + 0.75 * g) * 1.1 * age + nm_flake(co, n) * g * g * 1.5 * old ** 1.5, 0.0, 1.0)

    for o in boards:
        paint(o, grime=streaks, wear=low_flake, special=_mul(nm_fade, 0.8 * age))
    for o in new_boards:                              # the repair is younger: cleaner, hardly chipped, not faded
        paint(o, grime=_mul(streaks, 0.4), wear=_mul(nm_chip, 0.25 * age))
    for o in trim + low_trim + corn_trim:
        assign(o, m_trim)
    for o in trim:
        paint(o, grime=_mul(up, 0.3 + 0.45 * age), wear=_mul(nm_chip, 0.7 * age), special=_mul(nm_fade, 0.6 * age))
    for o in low_trim:                                # bottom rail and sill: flaking back to grey timber
        paint(o, grime=_mul(up, 0.35 + 0.5 * age),
              wear=lambda co, n: np.clip(nm_chip(co, n) * 0.8 * age + nm_flake(co, n) * 1.6 * old ** 1.5, 0.0, 1.0),
              special=_mul(nm_fade, 0.6 * age))
    for o in corn_trim:                               # cornice: dirty streaks under the lead drip
        paint(o, grime=lambda co, n: np.clip(up(co, n) * (0.3 + 0.45 * age) + rain(co + 0.37) * (0.1 + 0.7 * age * age) *
                                             np.clip((co[:, 2] - Hc) / ch, 0.0, 1.0), 0.0, 1.0),
              wear=_mul(nm_chip, 0.7 * age), special=_mul(nm_fade, 0.6 * age))

    # ---- swan-neck sign lamps on the wall above
    n = int(p.lamps)
    if n > 0:
        m_brass = mat("brass")
        m_shade = mat("ceramic_glazed", p.shade)
        m_white = mat("ceramic_glazed", "#ddd6c4")
        m_bulb = mat("emissive", "#ffd9a0", glow=12)
        Rm_, hs = 0.125, 0.105
        reach = float(p.lamp_reach)
        # pale trim and painted (non-gilt) pale letters bounce far more of the lamp back: fit a smaller bulb
        k_pale = lerp(1.0, 0.5, clamp(_lum(p.trim) * 1.6)) * (1.0 if p.gilt else lerp(1.0, 0.7, clamp(_lum(p.letter) * 1.3)))
        cd = float(p.intensity) * k_pale
        rl = rng.fork("lamps")
        roses = []
        for i in range(n):
            x = -xi + 2 * xi * (i + 0.5) / n + rl.jitter(0, 0.025)
            th = 20.0 + rl.jitter(0, 3.5)                        # shade tilt towards the board
            ry = rl.jitter(0, 5.0)                               # ...and a few degrees out of true sideways
            top = Hk + 0.028
            if ped and abs(x) < wp / 2 + 0.05:
                top = max(top, zc + sqrt(max(0.0, R * R - min(abs(x), wp / 2) ** 2)))
            zw = top + 0.2 + rl.jitter(0, 0.012)
            ct, st_, cr_, sr_ = cos(radians(th)), sin(radians(th)), cos(radians(ry)), sin(radians(ry))
            dv = (-ct * sr_, st_, -ct * cr_)                     # shade axis: down, leaning back at the board
            rot = (180 + th, ry, 0)
            # The shade hangs clear above the cornice so the nearest paint is the lead, not the crown mould.
            S = (x + rl.jitter(0, 0.014), -reach, top + 0.085 + hs * ct + rl.jitter(0, 0.01))

            def along(d_, S=S, dv=dv):
                return (S[0] + dv[0] * d_, S[1] + dv[1] * d_, S[2] + dv[2] * d_)

            objs = []
            objs.append(lathe("lamp_plate", [(0.05, 0.0), (0.05, 0.006), (0.034, 0.009), (0.023, 0.019), (0.019, 0.03), (0.0, 0.032)],
                              segments=10, loc=(x, 0, zw), rot=(90, 0, 0)))
            bt = cylinder("lamp_bolt", 0.0065, 0.005, loc=(0, 0, 0.0025), verts=5)
            apply_transform(bt)
            objs.append(instances_to_mesh("lamp_bolts", bt, [trs((x + 0.04 * cos(radians(90 + 120 * k)), -0.007, zw + 0.04 * sin(radians(90 + 120 * k))),
                                                                 (90, 0, 0)) for k in range(3)], keep_base=False))
            neck = bezier_path((x, -0.03, zw), (x, -0.03 - reach * 0.25, zw + 0.3), along(-0.28), along(-0.012), n=12)
            objs.append(sweep("lamp_neck", neck, 0.009, sides=6, smooth_path=False))
            objs.append(sphere("lamp_knuckle", 0.016, loc=neck[0], segments=6, rings=3))
            objs.append(cylinder("lamp_ferrule", 0.0125, 0.03, loc=along(-0.012), rot=rot, verts=8))
            for o in objs:                                       # the wall rose is dull tarnished brass, the neck handled brighter
                assign(o, m_brass)
                paint(o, special=_mul(noise_mask(0.05, seed=sd + 13 + i, lo=0.45, hi=0.8), 0.25 + 0.5 * age),
                      grime=0.7 if o.name.startswith(("lamp_plate", "lamp_bolt")) else 0.3)
            # spun steel shade, vitreous enamel: gallery, shoulder, flared dish, rolled rim; white inside
            prof = [(0.0, 0.0), (0.019, 0.0), (0.023, 0.004), (0.023, 0.027), (0.029, 0.032), (0.038, 0.035)]
            for q in range(1, 5):
                t = q / 4
                prof.append((0.038 + (Rm_ - 0.038) * t ** 0.85, 0.035 + (hs - 0.004 - 0.035) * t ** 1.7))
            prof += [(Rm_ + 0.004, hs - 0.003), (Rm_ + 0.005, hs + 0.001), (Rm_ + 0.002, hs + 0.005), (Rm_ - 0.002, hs + 0.002)]
            for q in (0.55,):
                prof.append((0.038 + (Rm_ - 0.038) * q ** 0.85 - 0.002, 0.035 + (hs - 0.004 - 0.035) * q ** 1.7 + 0.004))
            prof += [(0.03, 0.041), (0.0, 0.041)]
            shade = lathe("lamp_shade", prof, segments=16, smooth_angle=62.0)
            assign(shade, m_shade)
            assign_faces(shade, m_white, lambda c_, nn: c_.z > 0.038 and ((nn.x * c_.x + nn.y * c_.y) < -1e-5 or (nn.z > 0.6 and (c_.x * c_.x + c_.y * c_.y) < 0.0016)))
            set_loc_rot(shade, loc=S, rot=rot)

            def rim(co, nn, S=S, dv=dv, i=i):                    # enamel only chips where the rim gets knocked
                v = co - np.array(S)
                ax = v @ np.array(dv)
                r = np.linalg.norm(v - ax[:, None] * np.array(dv)[None, :], axis=1)
                return np.clip((r - (Rm_ - 0.012)) / 0.012, 0.0, 1.0) * noise_mask(0.025, seed=sd + 17 + i, lo=0.5, hi=0.72)(co, nn) * (0.35 + 0.65 * age)

            paint(shade, wear=rim, grime=_mul(up, 0.22))
            hold = cylinder("lamp_holder", 0.015, 0.024, loc=along(0.05), rot=rot, verts=8)
            assign(hold, m_white)
            bulb = lathe("lamp_bulb", [(0.012, 0.0), (0.02, 0.026), (0.027, 0.047), (0.021, 0.066), (0.0, 0.076)],
                         segments=8, loc=along(0.058), rot=rot)
            assign(bulb, m_bulb)
            # The emitter hangs 3 cm below the rim and the whole fitting (shade, neck, rose) is kept out of the shadow
            # pass, so the wash fades softly up the wall: no hard rim line, no magnified neck shadow up the wall.
            Lp = along(hs + 0.03)
            far = max(sqrt(sum((a_ - b2) ** 2 for a_, b2 in zip(Lp, q))) for q in list(neck) + [(x, 0.0, zw + 0.05), (x, 0.0, zw - 0.05)])
            face = sqrt((Lp[1] + d_b + ldepth) ** 2 + (Lp[2] - iz1) ** 2)          # nearest lettering: must keep casting
            light(f"lamp{i}", Lp, color="#ffbf74", intensity=cd, range=3.4, flicker=0.03, shadow=True, shadow_near=round(min(far + 0.02, 0.97 * face), 3))
            roses.append((x, zw))
        # lead-sheathed flex clipped along the wall from rose to rose, then down behind the fascia
        for (x0, z0_), (x1, z1_) in zip(roses[:-1], roses[1:]):
            c_ = sweep("lamp_flex", [(q[0], -0.006, q[2]) for q in catenary((x0 + 0.05, 0, z0_), (x1 - 0.05, 0, z1_), sag=0.02, n=6)], 0.0035, sides=3, smooth_path=False)
            assign(c_, m_lead)
        x0, z0_ = roses[0]
        zt = Hk + 0.03
        if ped and abs(x0) < wp / 2:
            zt = max(zt, zc + sqrt(max(0.0, R * R - x0 * x0)))
        c_ = sweep("lamp_flex", [(x0 - 0.05, -0.006, z0_), (x0 - 0.075, -0.006, z0_ - 0.02), (x0 - 0.082, -0.006, z0_ - 0.07), (x0 - 0.08, -0.006, zt - 0.01)],
                   0.0035, sides=3, smooth_path=False)
        assign(c_, m_lead)

    socket("awning", "attach", (0, -d_b, 0.0), role="awning")
    socket("centre_top", "attach", (0, -0.02, H), role="lantern")


# ------------------------------------------------------------------ small utilities
def degrees(a):
    return a * 180.0 / pi


def bezier2(p0, p1, p2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def _mul(fn, k):
    return lambda co, n: np.clip(fn(co, n) * k, 0.0, 1.0)


def _add(fn, k):
    return lambda co, n: np.clip(fn(co, n) + k, 0.0, 1.0)


def _mix(hex_a, hex_b, t):
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _lum(hx):
    r, g, b = hex_rgb(hx)[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _measure(body, font, spacing=1.0):
    probe = text("probe", body, size=0.1, depth=0.0, font=font, spacing=spacing)
    mn, mx = world_bbox([probe])
    delete([probe])
    return max(mx.x - mn.x, 1e-4), max(mx.z - mn.z, 1e-4)


def _fit_text(name, body, font, max_w, max_h, depth, centre, fill_w=0.0, shadow=False, rng=None, tilt=0.0, chamfer=0.0, res=9.0):
    """Raised cut-out lettering (chamfered face, splayed sides, no hidden back) set out to the board: the cap height
    is max_h, the word is letter-spaced / extended towards fill_w or condensed and shrunk to stay inside max_w, and
    centred on `centre` on the plane y = centre.y. With `rng` every letter is pinned on slightly out of true (tilt in
    degrees). With shadow=True also returns the flat outline for a painted drop shadow."""
    import bmesh
    w0, h0 = _measure(body, font)
    size = 0.1 * max_h / h0
    w = w0 * size / 0.1
    spacing, sx = 1.0, 1.0
    if w > max_w:
        sx = max(0.84, max_w / w)
        if w * sx > max_w:
            size *= max_w / (w * sx)
    elif min(fill_w, max_w) > w and len(body) > 1:
        goal = min(fill_w, max_w)
        if font != "script":                                    # joined script cannot be letter-spaced
            w1 = _measure(body, font, 1.25)[0] * size / 0.1
            if w1 > w + 1e-5:
                f = clamp((goal - w) / (w1 - w))
                spacing = 1.0 + 0.25 * f
                w = w + (w1 - w) * f
        sx = clamp(goal / w, 1.0, 1.2)
    flat_ = text(name + "_flat", body, size=size, depth=0.0, font=font, spacing=spacing)
    bm = bmesh.new()
    bm.from_mesh(flat_.data)
    delete([flat_])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.normal_update()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.z < 0], context="FACES")
    bmesh.ops.dissolve_limit(bm, angle_limit=radians(res), verts=bm.verts[:], edges=bm.edges[:])
    for v in bm.verts:
        v.co.x *= sx
    if rng is not None and tilt > 0:
        # group the outline islands into glyphs (dots / accents ride with their letter) and knock each one out of true
        seen, isl = set(), []
        for v in bm.verts:
            if v in seen:
                continue
            seen.add(v)
            stack, comp = [v], []
            while stack:
                a = stack.pop()
                comp.append(a)
                for e in a.link_edges:
                    b = e.other_vert(a)
                    if b not in seen:
                        seen.add(b)
                        stack.append(b)
            xs = [q.co.x for q in comp]
            isl.append([min(xs), max(xs), comp])
        isl.sort(key=lambda q: q[0])
        groups = []
        for x0, x1, comp in isl:
            g = groups[-1] if groups else None
            if g and (min(x1, g[1]) - max(x0, g[0])) > 0.5 * min(x1 - x0, g[1] - g[0]):
                g[1] = max(g[1], x1)
                g[2] += comp
            else:
                groups.append([x0, x1, list(comp)])
        tl = tilt * (0.35 if font == "script" else 1.0)
        odd = int(rng.u(0, len(groups) - 1e-6))
        for gi, (x0, x1, comp) in enumerate(groups):
            a = radians(rng.jitter(0, tl) + (rng.pick((-1, 1)) * tl * 1.3 if gi == odd else 0.0))
            dx, dy = rng.jitter(0, 0.022 * size), rng.jitter(0, 0.022 * size)
            ys = [q.co.y for q in comp]
            cx, cy = (x0 + x1) / 2, (min(ys) + max(ys)) / 2
            ca, sa = cos(a), sin(a)
            for q in comp:
                ux, uy = q.co.x - cx, q.co.y - cy
                q.co.x, q.co.y = cx + ux * ca - uy * sa + dx, cy + ux * sa + uy * ca + dy
    bm.normal_update()
    out = []
    if shadow:
        sh = from_bmesh(name + "_shadow", bm.copy())
        out.append(sh)
    for v in bm.verts:
        v.co.z = depth
    slope = 0.45 * depth
    acc = {}
    loops = []
    for e in bm.edges:
        if not e.is_boundary:
            continue
        l = e.link_loops[0]
        v0, v1 = l.vert, l.link_loop_next.vert
        d = v1.co - v0.co
        L = max((d.x * d.x + d.y * d.y) ** 0.5, 1e-9)
        nx, ny = d.y / L, -d.x / L
        for v in (v0, v1):
            a = acc.setdefault(v, [0.0, 0.0])
            a[0] += nx
            a[1] += ny
        loops.append((v0, v1))
    low, mid = {}, {}
    for v, (ax, ay) in acc.items():
        L = max((ax * ax + ay * ay) ** 0.5, 1e-6)
        k = min(1.6, 2.0 / max(L, 0.5))
        nx, ny = ax / L * k, ay / L * k
        low[v] = bm.verts.new((v.co.x + nx * slope, v.co.y + ny * slope, 0.0))
        if chamfer > 0:                                         # shoulder on the outline, face inset: a crisp gilder's chamfer
            mid[v] = bm.verts.new((v.co.x, v.co.y, depth - 0.8 * chamfer))
            v.co.x -= nx * chamfer
            v.co.y -= ny * chamfer
    for v0, v1 in loops:
        if chamfer > 0:
            bm.faces.new((v1, v0, mid[v0], mid[v1]))
            bm.faces.new((mid[v1], mid[v0], low[v0], low[v1]))
        else:
            bm.faces.new((v1, v0, low[v0], low[v1]))
    t = from_bmesh(name, bm)
    out.insert(0, t)
    mn, mx = world_bbox([t])
    for o in out:
        set_loc_rot(o, loc=(centre[0] - (mn.x + mx.x) / 2, centre[1], 0.0), rot=(90, 0, 0))
    mn, mx = world_bbox([t])
    for o in out:
        set_loc_rot(o, loc=(o.location.x, centre[1], centre[2] - (mn.z + mx.z) / 2))
        smooth(o, 30)
    return out if shadow else t
