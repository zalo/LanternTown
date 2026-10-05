"""Deeply furrowed grey-brown bark of a mature broadleaf trunk (ash / elm / oak type). Ridges run along V.

How real furrowed bark is built: the trunk keeps growing inside a dead cork jacket that cannot
stretch, so the jacket tears lengthwise into a NET of staggered, lens-shaped ridges (25-45 mm wide,
12-30 cm long) that pinch out between their neighbours.  So the layout here is built from units, not
contours: one elongated, staggered Voronoi cell per ridge, the cell borders being the furrows.
Furrows are V-shaped (floor only 2-4 mm wide, flanks ramping over 10-14 mm) and the flanks expose the
stacked cork sheets as 3 mm terraces.  Every ridge has its own width, crown level, bulge and tone.
Ridge tops are plated: overlapping cork scales with lifted lower edges, broader flakes sitting at
slightly different levels, a few second-generation fissures and cross checks that snap a ridge into
blocks.  Colour stays restrained - the darkness of the furrows comes from the derived AO, not paint."""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.014, "ao_strength": 0.9}


def _terrace(x, softness=0.35):
    f = np.floor(x)
    return f + smoothstep(1.0 - softness, 1.0, x - f)


def _vor(t, X, Y, nx, ny, jit, seed, stagger=False, q=1.0, rx=2, ry=2):
    """Periodic Voronoi on an nx x ny seed grid evaluated at (possibly warped) positions X, Y.
    q > 1 weights vertical distance more (pointed, lens-shaped cells on a staggered grid).
    Returns id, exact physical distance to the cell border (m), metric distance to the seed
    (cell units) and the local offsets from the seed (m)."""
    rng = np.random.default_rng(seed)
    cw, ch = t.W / nx, t.H / ny
    sx = (np.arange(nx)[None, :] + 0.5 + (rng.random((ny, nx)) - 0.5) * jit).astype(F)
    sy = (np.arange(ny)[:, None] + 0.5 + (rng.random((ny, nx)) - 0.5) * jit).astype(F)
    if stagger:
        sy = sy + 0.5 * (np.arange(nx)[None, :] % 2).astype(F)
    xs, ys = (X / cw).astype(F), (Y / ch).astype(F)
    gx = np.floor(xs).astype(np.int64)
    gy = np.floor(ys).astype(np.int64)
    offs = [(ox, oy) for oy in range(-ry, ry + 1) for ox in range(-rx, rx + 1)]

    def seedpos(ox, oy):
        cx, cy = gx + ox, gy + oy
        wx, wy = cx % nx, cy % ny
        return sx[wy, wx] + (cx - wx), sy[wy, wx] + (cy - wy), wy * nx + wx

    d1 = np.full(X.shape, 1e9, dtype=F)
    p1x = np.zeros(X.shape, dtype=F)
    p1y = np.zeros(X.shape, dtype=F)
    id1 = np.zeros(X.shape, dtype=np.int64)
    for ox, oy in offs:
        qx, qy, qi = seedpos(ox, oy)
        d = (qx - xs) ** 2 + ((qy - ys) * q) ** 2
        c = d < d1
        d1 = np.where(c, d, d1)
        p1x = np.where(c, qx, p1x)
        p1y = np.where(c, qy, p1y)
        id1 = np.where(c, qi, id1)
    dist = np.full(X.shape, 1e9, dtype=F)
    for ox, oy in offs:
        qx, qy, qi = seedpos(ox, oy)
        ex, ey = qx - p1x, (qy - p1y) * q
        el = np.sqrt(ex * ex + ey * ey)
        ok = el > 1e-5
        el = np.maximum(el, 1e-5)
        ds = (((p1x + qx) * 0.5 - xs) * ex + ((p1y + qy) * 0.5 - ys) * q * ey) / el
        nrm = np.sqrt((ex / el / cw) ** 2 + (ey / el * q / ch) ** 2)
        dp = ds / nrm
        dist = np.where(ok & (dp < dist), dp, dist)
    return dict(id=id1, d=np.maximum(dist, 0.0).astype(F), rs=np.sqrt(d1).astype(F),
                lx=((xs - p1x) * cw).astype(F), ly=((ys - p1y) * ch).astype(F), n=nx * ny, cw=cw, ch=ch)


def _rnd(cell, seed):
    return np.random.default_rng(seed).random(cell["n"]).astype(F)[cell["id"]]


def _ridge_net(t, X, Y, nx, ny, jit, seed, q, gnx, gny, lipn, Dp, Ds, wp, ws, Rp, Rs, crown_fn):
    """The ridge net.  One staggered, lens-shaped Voronoi cell per ridge strand; strands are bundled
    into ridges (groups).  Borders between ridges are primary furrows (deep, absolute floor at -Dp);
    borders between strands of one ridge are secondary fissures (floor Ds below the ridge level).
    Every border is an exact V groove: floor half-width w, flank ramp R, combined by a smooth max of
    the per-border deficits.  Returns height (m, crown ~ 0), crown height C, ids and distances."""
    rng = np.random.default_rng(seed)
    cw, ch = t.W / nx, t.H / ny
    sx = (np.arange(nx)[None, :] + 0.5 + (rng.random((ny, nx)) - 0.5) * jit)
    sy = (np.arange(ny)[:, None] + 0.5 + (rng.random((ny, nx)) - 0.5) * jit) + 0.5 * (np.arange(nx)[None, :] % 2)
    # bundle strands: nearest seed of a coarser staggered lattice (periodic)
    gx_ = (np.arange(gnx)[None, :] + 0.5 + (rng.random((gny, gnx)) - 0.5) * 0.9) * (nx / gnx)
    gy_ = (np.arange(gny)[:, None] + 0.5 + (rng.random((gny, gnx)) - 0.5) * 0.9 + 0.5 * (np.arange(gnx)[None, :] % 2)) * (ny / gny)
    ddx = np.abs(sx.reshape(-1, 1) - gx_.reshape(1, -1))
    ddx = np.minimum(ddx, nx - ddx)
    ddy = np.abs(np.mod(sy.reshape(-1, 1) - gy_.reshape(1, -1), ny))
    ddy = np.minimum(ddy, ny - ddy)
    grp = np.argmin(ddx ** 2 + (ddy * q) ** 2, axis=1).astype(np.int64)
    sx, sy = sx.astype(F), sy.astype(F)

    xs, ys = (X / cw).astype(F), (Y / ch).astype(F)
    gx = np.floor(xs).astype(np.int64)
    gy = np.floor(ys).astype(np.int64)
    offs = [(ox, oy) for oy in range(-2, 3) for ox in range(-3, 4)]

    def seedpos(ox, oy):
        cx, cy = gx + ox, gy + oy
        wx, wy = cx % nx, cy % ny
        return sx[wy, wx] + (cx - wx), sy[wy, wx] + (cy - wy), wy * nx + wx

    d1 = np.full(X.shape, 1e9, dtype=F)
    p1x = np.zeros(X.shape, dtype=F)
    p1y = np.zeros(X.shape, dtype=F)
    id1 = np.zeros(X.shape, dtype=np.int64)
    for ox, oy in offs:
        qx, qy, qi = seedpos(ox, oy)
        d = (qx - xs) ** 2 + ((qy - ys) * q) ** 2
        c = d < d1
        d1 = np.where(c, d, d1)
        p1x = np.where(c, qx, p1x)
        p1y = np.where(c, qy, p1y)
        id1 = np.where(c, qi, id1)
    g1 = grp[id1]
    rs = np.sqrt(d1)
    C, Cg = crown_fn(id1, g1, rs, nx * ny, gnx * gny)       # strand crown height, ridge (group) level
    acc = np.zeros(X.shape, dtype=F)
    dmin = np.full(X.shape, 1e9, dtype=F)
    dpri = np.full(X.shape, 1e9, dtype=F)
    for ox, oy in offs:
        qx, qy, qi = seedpos(ox, oy)
        ex, ey = qx - p1x, (qy - p1y) * q
        el = np.sqrt(ex * ex + ey * ey)
        ok = el > 1e-5
        el = np.maximum(el, 1e-5)
        ds = (((p1x + qx) * 0.5 - xs) * ex + ((p1y + qy) * 0.5 - ys) * q * ey) / el
        dp = ds / np.sqrt((ex / el / cw) ** 2 + (ey / el * q / ch) ** 2)
        dp = np.where(ok, dp, 1e9)
        same = grp[qi] == g1
        dmin = np.minimum(dmin, dp)
        dpri = np.minimum(dpri, np.where(same, 1e9, dp))
        dl = dp + lipn
        xx = sat((dl - np.where(same, ws, wp)) / np.where(same, Rs, Rp))
        full = np.where(same, np.maximum(C - Cg + Ds, 0.0004), np.maximum(C + Dp, 0.0004))
        dfc = full * (1 - xx) ** 1.25
        acc += (dfc * 100.0) ** 6
    h = C - acc ** (1.0 / 6.0) / 100.0
    return dict(h=h.astype(F), C=C.astype(F), id=id1, grp=g1, d=dmin, dpri=dpri, rs=rs, n=nx * ny, gn=gnx * gny)


def build(t):
    # ------------------------------------------------------------------ layout: a net of lens-shaped ridges
    # ridges wander a little as they climb the trunk (domain warp), the tear lines are ragged
    wx = ((t.noise(0.16, seed=1, stretch=(1, 2.2)) - 0.5) * 2 * 0.020 + (t.noise(0.050, seed=2, stretch=(1, 2.0)) - 0.5) * 2 * 0.012
          + (t.noise(0.018, seed=42, stretch=(1, 1.6)) - 0.5) * 2 * 0.003)
    wy = (t.noise(0.15, seed=3) - 0.5) * 2 * 0.012 + (t.noise(0.035, seed=43) - 0.5) * 2 * 0.014
    lip = t.fbm(0.012, 3, 0.6, seed=6, stretch=(1, 2.5))           # torn, irregular furrow lips
    lip2 = t.fbm(0.030, 2, 0.5, seed=46, stretch=(1, 2.0))
    lipn = (lip - 0.5) * 0.0034 + (lip2 - 0.5) * 0.0034
    wn = t.fbm(0.10, 3, 0.5, seed=5, stretch=(1, 2))
    wv = t.noise(0.05, seed=41, stretch=(1, 3))
    # primary furrows: 11-16 mm deep, floor 2.5-5 mm wide, flanks ramping over 10.5-14 mm
    Dp = 0.0125 * (0.85 + 0.30 * t.noise(0.11, seed=8))
    wp = 0.0012 + 0.0014 * wv
    Rp = 0.0105 + 0.0035 * wn
    # secondary fissures inside a ridge: 2-8 mm, hairline floor, shallowing out so strands fuse
    gate = 0.40 + 0.60 * smoothstep(0.30, 0.55, t.noise(0.034, seed=7, stretch=(1, 2.6)))
    Ds = 0.0095 * gate * (0.8 + 0.4 * t.noise(0.09, seed=9))
    ws = 0.0003 + 0.0006 * wv
    Rs = 0.0075 + 0.0040 * wn

    def crown(cid, gid, rs, n, gn):
        r = np.random.default_rng(5150)
        g_lvl = (r.random(gn).astype(F)[gid] - 0.5) * 0.0070          # each ridge at its own level (+-4 mm)
        c_lvl = (r.random(n).astype(F)[cid] - 0.5) * 0.0022
        bul = (0.0015 + 0.0040 * r.random(gn).astype(F)[gid] ** 2) * (1 - sat(rs) ** 2)   # old ridges bulge
        return g_lvl + c_lvl + bul, g_lvl

    rg = _ridge_net(t, t.X + wx, t.Y + wy, 26, 6, 0.85, 4101, 2.0, 16, 6, lipn, Dp, Ds, wp, ws, Rp, Rs, crown)
    C = rg["C"]
    swell = (t.fbm(0.28, 2, 0.5, seed=10) - 0.5) * 0.005
    rgen = np.random.default_rng(6001)
    r_tone = rgen.random(rg["gn"]).astype(F)[rg["grp"]]
    r_hue = rgen.random(rg["gn"]).astype(F)[rg["grp"]]
    s_tone = rgen.random(rg["n"]).astype(F)[rg["id"]]

    # ------------------------------------------------------------------ flanks expose the cork sheets
    sheet = 0.0038
    hh = rg["h"] + Dp                                              # height above the primary floor
    tn_ = t.fbm(0.026, 3, 0.5, seed=11, stretch=(1, 4.0))
    tq = hh / sheet + (tn_ - 0.5) * 2.0
    terr = (_terrace(tq, 0.50) - (tn_ - 0.5) * 2.0) * sheet
    step_i = np.floor(tq)
    body = mix(hh, terr, 0.85)

    dep = C - rg["h"]                                              # depth below the strand's own crown
    prof = 1 - sat(dep / np.maximum(C + Dp, 0.004))                # 0 primary floor .. 1 crown
    top = 1 - smoothstep(0.0012, 0.0050, dep)                      # ridge crown
    flr = 1 - smoothstep(0.0, 0.22, prof)                          # furrow floor
    wall = sat(1 - top - flr)

    # ------------------------------------------------------------------ ridge tops: stacked cork plates
    # broad flakes (26 x 65 mm) sitting at slightly different levels
    fwx = (t.noise(0.040, seed=51) - 0.5) * 2 * 0.012
    fwy = (t.noise(0.040, seed=52) - 0.5) * 2 * 0.012
    fl = _vor(t, t.X + fwx, t.Y + fwy, 23, 18, 0.95, 4102)
    flake = (_rnd(fl, 21) - 0.5) * 0.0030
    fl_gap = (1 - smoothstep(0.0003, 0.0016, fl["d"])) * smoothstep(0.35, 0.65, _rnd(fl, 22))
    # small scales (12 x 40 mm) overlapping like shingles: lower edge lifted off the one below
    swx = (t.noise(0.025, seed=53) - 0.5) * 2 * 0.006
    swy = (t.noise(0.030, seed=54) - 0.5) * 2 * 0.008
    sc = _vor(t, t.X + swx, t.Y + swy, 56, 28, 0.9, 4103, stagger=True, q=1.4)
    sc_r = _rnd(sc, 23)
    sc_lift = sat(0.5 - sc["ly"] / sc["ch"]) * (0.0009 + 0.0013 * _rnd(sc, 24))    # higher towards the lower edge
    scale_h = (sc_r - 0.5) * 0.0011 + sc_lift
    sc_gap = 1 - smoothstep(0.0002, 0.0011, sc["d"])
    plates = t.blur(flake + scale_h, 0.0004)

    # second-generation fissures up the ridge tops (few, of mixed width) and cross checks
    f1 = t.scratches(density=300, length=0.060, width=0.0010, seed=12, angle=90, angle_spread=22)
    f2 = t.scratches(density=150, length=0.075, width=0.0024, seed=13, angle=90, angle_spread=18)
    f1 = t.blur(t.warp(f1, 0.004, 0.05, seed=14), 0.0005)
    f2 = t.blur(t.warp(f2, 0.005, 0.06, seed=15), 0.0009)
    c1 = t.scratches(density=330, length=0.020, width=0.0011, seed=16, angle=0, angle_spread=40)
    c2 = t.scratches(density=170, length=0.028, width=0.0026, seed=17, angle=0, angle_spread=30)
    c1 = t.blur(t.warp(c1, 0.003, 0.03, seed=18), 0.0005)
    c2 = t.blur(t.warp(c2, 0.004, 0.03, seed=19), 0.0009)
    on_ridge = 1 - smoothstep(0.003, 0.008, dep)
    fiss_d = (sat(f1 * 1.5) * 0.0016 + sat(f2 * 1.4) * 0.0034) * on_ridge
    chk_d = (sat(c1 * 1.5) * 0.0015 + sat(c2 * 1.4) * 0.0040) * on_ridge
    crack = sat((fiss_d + chk_d) / 0.0030)

    # micro: fibrous inner bark in the furrows, corky grit and pits on the crowns
    fibre = t.fbm(0.0050, 3, 0.6, seed=31, stretch=(1, 10))
    grit = t.fbm(0.0040, 3, 0.6, seed=32)
    pits = t.speckle(density=9000, size=0.0034, seed=33)

    height = (body - Dp + swell
              + plates * (1 - smoothstep(0.002, 0.008, dep))
              - (fl_gap * 0.0012 + sc_gap * 0.0002) * top
              - fiss_d - chk_d
              + (fibre - 0.5) * (0.0007 + 0.0009 * (1 - prof))
              + (grit - 0.5) * 0.0006 - pits * 0.0005 * top)

    # ------------------------------------------------------------------ masks from the height
    cav = t.cavity(height, 0.0035, 0.0012)
    edge = t.edges(height, 0.0045, 0.0014)

    # ------------------------------------------------------------------ colour
    macro = t.fbm(0.30, 3, 0.5, seed=21)
    mott = t.fbm(0.035, 4, 0.55, seed=22, stretch=(1, 2.2))
    tone = sat(0.47 + (r_tone - 0.5) * 0.40 + (s_tone - 0.5) * 0.14 + (mott - 0.5) * 0.34 + (_rnd(fl, 25) - 0.5) * 0.20
               + (sc_r - 0.5) * 0.16 + (fibre - 0.5) * 0.12 + (macro - 0.5) * 0.12)
    crown_c = ramp(tone, [(0.0, "#5a5146"), (0.3, "#6d6457"), (0.55, "#7f7668"),
                          (0.8, "#8f8779"), (1.0, "#9f988b")])
    # some ridges a little warmer / browner, some cooler grey
    crown_c = crown_c * mix(hexlin("#f2e6d6") * 1.06, hexlin("#e6e8e4") * 1.05, r_hue[..., None])
    # flanks: browner, and each exposed cork sheet a slightly different tone
    lay_t = (np.sin(step_i * 2.4 + 1.3) * 0.5 + 0.5)
    wall_c = ramp(sat(0.25 + tone * 0.45 + (lay_t - 0.5) * 0.30 + (tn_ - 0.5) * 0.30),
                  [(0.0, "#4c3d30"), (0.4, "#5f4d3c"), (0.75, "#72604b"), (1.0, "#84735d")])
    floor_c = ramp(sat(fibre * 0.9 + (grit - 0.5) * 0.4), [(0.0, "#3a2e24"), (0.5, "#4a3b2d"), (1.0, "#5a4836")])
    wmix = smoothstep(0.0025, 0.0100, dep + (mott - 0.5) * 0.004) * 0.85
    albedo = mix(crown_c, wall_c, wmix)
    albedo = mix(albedo, floor_c, 1 - smoothstep(0.0, 0.25, prof))
    albedo = mix(albedo, hexlin("#33291f"), sat(crack * 0.9 + fl_gap * 0.4 * top) * 0.25)        # open cracks
    albedo = mix(albedo, albedo * hexlin("#8f7a66") * 1.25, (sat(cav) * 0.35)[..., None])        # dust in recesses
    albedo = mix(albedo, albedo * 1.12 + 0.010, (edge * top * 0.25)[..., None])                  # bleached arrises
    # a breath of grey-green algal dust on some crowns (real moss comes from the engine)
    lich = smoothstep(0.60, 0.85, t.fbm(0.07, 4, 0.6, seed=24)) * top * (0.4 + 0.6 * grit)
    albedo = mix(albedo, hexlin("#858a77"), lich * 0.30)
    albedo = albedo * (0.95 + 0.10 * macro)[..., None]
    albedo = np.clip(albedo, 0.025, 1.0)

    # ------------------------------------------------------------------ roughness
    # matt everywhere; weathered cork crowns a little smoother than fibrous, dusty furrows
    rough = (0.90 - 0.10 * top + 0.05 * flr + 0.03 * wall + 0.07 * (grit - 0.5) + 0.05 * (tone - 0.5) * top
             + 0.05 * lich + 0.04 * sat(cav) - 0.05 * edge * top)

    return dict(albedo=albedo, height=height, rough=sat(rough), metal=0.0)
