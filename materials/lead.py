"""Old sheet lead (flashing, cames, cistern linings, finials): soft dull blue-grey metal.

Built in the order the real thing gets its surface:
  1. milled sheet with a faint orange-peel skin, sagging a little between its fixings;
  2. bossed and dressed by hand: overlapping soft oval mallet flats (no facets), and broad swathes
     where the dresser was dragged across it - 30-50 mm wide bands of very fine striations, every
     swathe at its own angle, slightly bowed, each line with its own length and weight. They mostly
     burnish (smoother, a touch lighter); only a few lines cut through to bright metal;
  3. decades of knocks: a handful of broad shallow irregular dishes, a scatter of small dings, two or
     three soft creases, a couple of dozen single scratches of all lengths, slightly curved;
  4. patina: a tight dull blue-grey oxide skin - dielectric, metal = 0 - cloudy, with darker smoother
     rain-washed / handled patches;
  5. white lead-carbonate bloom: a thin powdery haze everywhere, concentrated by logic - tide lines
     where water dried in the dishes (against the downhill lip), plugs in dings, lines in creases,
     and a few short wandering run-off veils below the dishes. Powdery granular edges.
Everything is scattered with a minimum spacing so density is even and nothing is countable.
"""
import math
from tk.tex import *

META = {"normal_strength": 3.2}
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=F)


def ovals(t, spacing, r_lo, r_hi, seed, aspect=1.6, plateau=0.3, nb=2):
    """Overlapping oval marks on a jittered grid, summed (each 1 on its plateau falling softly to 0 at
    its rim, random orientation / size / strength). r_hi must stay below nb * spacing."""
    rng = np.random.default_rng(t.seed * 1000 + 7000 + seed)
    nx, ny = max(3, round(t.W / spacing)), max(3, round(t.H / spacing))
    cw, ch = t.W / nx, t.H / ny
    sh = (ny, nx)
    px, py = rng.random(sh).astype(F), rng.random(sh).astype(F)
    an = (rng.random(sh) * math.pi).astype(F)
    rr = (r_lo + (r_hi - r_lo) * rng.random(sh)).astype(F)
    asp = (1.0 + (aspect - 1.0) * rng.random(sh)).astype(F)
    amp = (0.4 + 0.6 * rng.random(sh)).astype(F)
    gx, gy = t.X / cw, t.Y / ch
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = (gx - ix).astype(F), (gy - iy).astype(F)
    acc = t.zeros()
    for oy in range(-nb, nb + 1):
        for ox in range(-nb, nb + 1):
            jx, jy = (ix + ox) % nx, (iy + oy) % ny
            dx = (fx - ox - px[jy, jx]) * cw
            dy = (fy - oy - py[jy, jx]) * ch
            ca, sa = np.cos(an[jy, jx]), np.sin(an[jy, jx])
            a, b = dx * ca + dy * sa, (-dx * sa + dy * ca) * asp[jy, jx]
            d = np.sqrt(a * a + b * b) / rr[jy, jx]
            acc += amp[jy, jx] * (1 - smoothstep(plateau, 1.0, d))
    return acc.astype(F)


def scatter(rng, t, n, sep):
    """n points kept at least `sep` apart (wrapping): even density, no grid."""
    pts, tries = [], 0
    while len(pts) < n and tries < 8000:
        tries += 1
        x, y = rng.random() * t.W, rng.random() * t.H
        if all(math.hypot((x - qx + t.W * 0.5) % t.W - t.W * 0.5, (y - qy + t.H * 0.5) % t.H - t.H * 0.5) >= sep
               for qx, qy in pts):
            pts.append((x, y))
    return pts


def local(t, cx, cy, ang=0.0, X=None, Y=None):
    """Wrapped coordinates (along, across) in metres about a point, rotated by ang."""
    X = t.X if X is None else X
    Y = t.Y if Y is None else Y
    dx = (X - cx + t.W * 0.5) % t.W - t.W * 0.5
    dy = (Y - cy + t.H * 0.5) % t.H - t.H * 0.5
    ca, sa = math.cos(ang), math.sin(ang)
    return dx * ca + dy * sa, -dx * sa + dy * ca


def dishes(t, seed, n=7, r_lo=0.021, r_hi=0.036, sep=0.125):
    """Broad shallow dents: irregular (warped) ellipses, evenly scattered.
    Returns (dm: 0 at the rim .. 1 at the centre, per-dish random, downhill weight -1..1, centres)."""
    rng = np.random.default_rng(t.seed * 1000 + 5000 + seed)
    wx = (t.noise(0.05, seed=seed * 7 + 1) - 0.5) * 0.022                   # irregular outlines
    wy = (t.noise(0.05, seed=seed * 7 + 2) - 0.5) * 0.022
    dm, rv, low = t.zeros(), t.zeros(), t.zeros()
    pts = scatter(rng, t, n, sep)
    info = []
    for (x, y) in pts:
        r, asp, ang, v = r_lo + (r_hi - r_lo) * rng.random(), 1.0 + 0.6 * rng.random(), rng.random() * math.pi, rng.random()
        a, b = local(t, x, y, ang, t.X + wx, t.Y + wy)
        m = 1 - sat(np.sqrt(a * a + (b * asp) ** 2) / r)
        _, dyv = local(t, x, y)
        up = m > dm
        dm = np.where(up, m, dm)
        rv = np.where(up, v, rv)
        low = np.where(up, sat(-dyv / (r * 0.8) * 0.5 + 0.5) * 2 - 1, low)
        info.append((x, y, r, v))
    return dm.astype(F), rv.astype(F), low.astype(F), info


def dings(t, seed, n=17, sep=0.085):
    """Small dings 3-8 mm, each its own size / depth / slightly oval. Returns (profile, per-ding random)."""
    rng = np.random.default_rng(t.seed * 1000 + 5500 + seed)
    out, rv = t.zeros(), t.zeros()
    for (x, y) in scatter(rng, t, n, sep):
        r, v = 0.0015 + 0.0028 * rng.random(), rng.random()
        a, b = local(t, x, y, rng.random() * math.pi)
        d = np.sqrt(a * a + (b * (1.0 + 0.7 * rng.random())) ** 2) / r
        m = np.exp(-1.6 * d * d)
        up = m > out
        out = np.where(up, m, out)
        rv = np.where(up, v, rv)
    return out.astype(F), rv.astype(F)


def creases(t, seed, n=3):
    """Soft creases where the sheet was folded round something: 60-150 mm, slight bow."""
    rng = np.random.default_rng(t.seed * 1000 + 6000 + seed)
    out = t.zeros()
    for (x, y) in scatter(rng, t, n, 0.2):
        L = 0.06 + 0.09 * rng.random()
        a, b = local(t, x, y, rng.random() * math.pi)
        b = b - (rng.random() - 0.5) * 0.8 * (a * a - L * L / 12) / L
        w = 0.0016 + 0.0012 * rng.random()
        out = np.maximum(out, np.exp(-(b / w) ** 2) * smoothstep(0.0, 0.4, (L * 0.5 - np.abs(a)) / L) * (0.6 + 0.4 * rng.random()))
    return out.astype(F)


def swathes(t, seed, n=14, sep=0.10):
    """Dresser drag swathes. Each: its own angle (any), 90-180 mm long, 30-50 mm wide, bowed, with a
    slowly wandering direction; filled with fine striations (1.5-2.6 mm pitch) whose every line has
    its own length and weight. Later swathes overwrite earlier ones.
    Returns (striation -0.5..0.5 * envelope, envelope 0..1, slow per-line selector 0..1)."""
    rng = np.random.default_rng(t.seed * 1000 + 3000 + seed)
    stri_tex = t.noise(0.0009, seed=seed + 1, stretch=(70.0, 1.0), bandwidth=0.55)
    line_tex = t.noise(0.0014, seed=seed + 2, stretch=(400.0, 1.0))        # constant along a line
    sel_tex = t.noise(0.0045, seed=seed + 3, stretch=(30.0, 1.0))
    wob = (t.noise(0.16, seed=seed + 4) - 0.5) * 0.012                      # direction wanders +-8 deg
    out, cover, sel = t.zeros(), t.zeros(), t.zeros()
    for (x, y) in scatter(rng, t, n, sep):
        ang = rng.random() * math.pi
        L, wd = 0.09 + 0.09 * rng.random(), 0.030 + 0.020 * rng.random()
        a, b = local(t, x, y, ang)
        b = b + wob * (0.6 + 0.8 * rng.random()) + (rng.random() - 0.5) * 0.10 * a * a / L
        pitch = 0.85 + 0.6 * rng.random()
        oa, ob = rng.random() * t.W, rng.random() * t.H
        s = t.sample(stri_tex, a + oa, b / pitch + ob)
        ln = t.sample(line_tex, a * 0.02 + oa, b / pitch + ob)
        sl = t.sample(sel_tex, a + oa, b / pitch + ob)
        ea = smoothstep(0.0, 0.30, (L * 0.5 - np.abs(a)) / L + (ln - 0.5) * 0.9)
        eb = smoothstep(0.0, 0.30, (wd * 0.5 - np.abs(b)) / wd + (ln - 0.5) * 0.25)
        env = ea * eb * (0.75 + 0.25 * rng.random())
        k = sat(env * 1.5)
        out = out * (1 - k) + (s - 0.5) * env
        sel = sel * (1 - k) + sl * k
        cover = np.maximum(cover * (1 - k), env)
    return out.astype(F), cover.astype(F), sel.astype(F)


def scratches(t, seed, n=22, sep=0.07):
    """Single isolated scratches: 10-120 mm, slightly curved, soft-edged (never thinner than ~2 px),
    pressure fading towards one end. Returns (profile 0..1, per-scratch random)."""
    rng = np.random.default_rng(t.seed * 1000 + 2000 + seed)
    out, rv = t.zeros(), t.zeros()
    for (x, y) in scatter(rng, t, n, sep):
        L = 0.010 + 0.110 * rng.random() ** 1.8
        a, b = local(t, x, y, rng.random() * math.pi)
        b = b - (rng.random() - 0.5) * 0.5 * (a * a) / L
        w = max(0.00038 + 0.00030 * rng.random(), 0.85 * t.px)
        u = a / L + 0.5                                                    # 0..1 along it
        env = smoothstep(0.0, 0.08, u) * (1 - smoothstep(0.35 + 0.4 * rng.random(), 1.0, u))
        m = np.exp(-(b / w) ** 2) * env * (0.55 + 0.45 * rng.random())
        v = rng.random()
        up = m > out
        out = np.where(up, m, out)
        rv = np.where(up, v, rv)
    return out.astype(F), rv.astype(F)


def veils(t, seed, n=4):
    """Run-off veils: carbonate carried down the slope (-V) from a drip point. Short (0.1-0.3 m),
    wandering, 5-25 mm wide, widening a little and fading out; stratified in X, heads spread in Y.
    Returns a soft 0..1 wash."""
    rng = np.random.default_rng(t.seed * 1000 + 4000 + seed)
    out = t.zeros()
    wand = (t.noise(0.05, seed=seed + 1, stretch=(1.0, 2.0)) - 0.5) * 0.030      # +-5 mm wander
    ys = rng.permutation(n)
    for i in range(n):
        x = (i + 0.15 + 0.7 * rng.random()) * t.W / n
        y0 = (ys[i] + rng.random()) * t.H / n
        L = 0.10 + 0.20 * rng.random()
        w0 = 0.0025 + 0.006 * rng.random()
        s = ((y0 - t.Y) % t.H) / L                                          # 0 at the head, 1 at the foot
        dx = (t.X + wand - x + t.W * 0.5) % t.W - t.W * 0.5
        w = w0 * (1.0 + 1.1 * np.clip(s, 0, 1))
        env = smoothstep(0.0, 0.12, s) * (1 - smoothstep(0.1, 1.0, s)) ** 1.2 * (s < 1.0)
        out = np.maximum(out, env * np.exp(-(np.abs(dx) / w) ** 1.6) * (0.6 + 0.4 * rng.random()))
    return out.astype(F)


def build(t):
    # --- 1/2/3. the sheet: sagged, bossed, dressed and knocked about --------------------------------
    sag = t.fbm(0.22, 2, 0.45, seed=1)                                     # creep between fixings
    boss = t.blur(ovals(t, 0.045, 0.022, 0.040, 2, aspect=1.7, plateau=0.45), 0.003)   # 45-80 mm mallet flats
    boss2 = t.blur(ovals(t, 0.03, 0.014, 0.024, 3, aspect=1.5, plateau=0.0), 0.003)
    dm, dent_r, dent_l, dinfo = dishes(t, 4)
    dm = sat(t.blur(dm, 0.003) * 1.05 - 0.02)
    dent = smoothstep(0.0, 0.9, dm)                                       # soft, broad dish profile
    ding_m, ding_r = dings(t, 7)
    ding_m = t.warp(ding_m, 0.0008, 0.006, seed=6)
    crease = t.warp(creases(t, 8), 0.002, 0.05, seed=9)
    stri, scover, ssel = swathes(t, 40)
    scr, scr_r = scratches(t, 11)
    peel = t.fbm(0.0032, 3, 0.6, seed=10)                                  # skin grain
    pock = t.speckle(density=26000, size=0.0013, seed=13, soft=0.8) * smoothstep(0.45, 0.7, t.fbm(0.04, 3, 0.55, seed=14))   # corrosion pocks, in drifts

    form = ((sag - 0.5) * 0.0021 - boss * 0.00040 - boss2 * 0.00015
            - dent * (0.00065 + 0.00050 * dent_r) - ding_m * (0.00012 + 0.00030 * ding_r) - crease * 0.00028)
    h = (form + stri * 0.00012 + (peel - 0.5) * 0.00005 - pock * 0.00005
         - scr * (0.00004 + 0.00006 * scr_r)
         + sat(t.blur(scr, 0.0009) * 1.8 - scr) * 0.00003)                 # a scratch in lead throws up a burr

    low = sat(dent + ding_m * 0.7 + crease * 0.6)
    lowb = t.blur(low, 0.003)
    cv = t.curv(form, 0.018)
    hollow, proud = sat(-cv / 0.00025), sat(cv / 0.00025)                  # powder stays in hollows, rubs off the proud

    # --- 4. patina: dull blue-grey oxide skin (dielectric) ------------------------------------------
    m1 = t.fbm(0.09, 4, 0.55, seed=20)
    m2 = t.fbm(0.02, 3, 0.6, seed=21)
    pt = sat(0.38 + 0.5 * (m1 - 0.5) + 0.25 * proud - 0.15 * hollow + 0.3 * (m2 - 0.5) + 0.25 * (peel - 0.5))
    base = ramp(pt, [(0.0, "#676b6f"), (0.3, "#6f7377"), (0.55, "#787c80"), (0.8, "#82868a"), (1.0, "#8a8d90")])
    cast = t.fbm(0.12, 3, 0.5, seed=22)
    base = mix(base, base * hexlin("#a9b0bd") * 1.78, smoothstep(0.45, 0.85, cast) * 0.45)    # bluer
    base = mix(base, base * hexlin("#b4b0a6") * 1.75, smoothstep(0.45, 0.15, cast) * 0.22)    # dirtier, neutral
    wash_n = t.fbm(0.09, 3, 0.5, seed=23, stretch=(1.0, 1.5))
    washed = smoothstep(0.52, 0.72, wash_n)                                # rain-washed / handled: darker, smoother
    base = mix(base, hexlin("#5f6468"), washed * 0.5)
    base = mix(base, base * 0.88, lowb * 0.5)                              # dirt settles in the hollows
    base = mix(base, hexlin("#4c5054"), pock * 0.55)
    burn = scover * (0.5 + 1.3 * stri)                                    # dresser burnish: thinner, lighter skin
    base = base * (1.0 + 0.42 * burn)[..., None]

    # --- 5. white carbonate bloom -------------------------------------------------------------------
    gran = t.fbm(0.0022, 2, 0.6, seed=30)                                  # 1-2 mm powder grain
    gran2 = t.fbm(0.006, 3, 0.65, seed=31)                                 # clumps
    cloud = t.fbm(0.045, 3, 0.5, seed=32)                                  # 20-60 mm haze
    brk = t.fbm(0.03, 2, 0.5, seed=34)
    ind = smoothstep(0.03, 0.12, dm)
    lvl = 0.26 + 0.3 * ((dent_r * 7.31) % 1.0)                             # each dish dried at its own level
    cres = smoothstep(-0.5, 0.5, dent_l + (brk - 0.5) * 1.6)               # sheet slopes: against the downhill lip
    ring = np.exp(-((dm - lvl) / 0.11) ** 2) * ind * cres * (0.55 + 0.45 * smoothstep(0.3, 0.7, brk))
    ring2 = 0.6 * np.exp(-((dm - lvl * 1.7 - 0.08) / 0.05) ** 2) * ind * cres * (dent_r > 0.35)   # an earlier line
    pool = smoothstep(lvl, lvl + 0.25, dm) * ind * (0.5 + 0.5 * cres)      # thin residue on the dish floor
    plug = smoothstep(0.25, 0.8, ding_m) * (ding_r > 0.3)
    fold = t.cavity(form, 0.004, 0.00016) * sat(crease * 3.0)
    vl = veils(t, 77)
    vfil = t.fbm(0.004, 3, 0.55, seed=33, stretch=(1.0, 10.0))             # filaments inside a veil
    field = np.maximum.reduce([1.0 * ring, 0.6 * ring2, 0.9 * plug, 0.8 * fold])
    dep = smoothstep(0.25, 0.95, field + 0.6 * (gran - 0.5) + 0.4 * (gran2 - 0.5))   # powdery granular edge
    haze = np.clip(0.45 + 1.0 * (cloud - 0.5) + 0.9 * hollow - 0.45 * proud + 0.6 * pool + 2.2 * vl * (0.35 + 1.1 * vfil), 0, 2.2) * (0.55 + 0.9 * gran2) * (0.7 + 0.6 * vfil)
    cut = 1 - sat(scr * 1.6)                                               # scratches cut through it
    bmix = sat(dep * (0.30 + 0.3 * (gran - 0.5)) + haze * 0.17 * (1 - dep)) * cut * (1 - 0.6 * washed * (1 - dep))
    bcol = ramp(sat(0.5 + 1.2 * (gran - 0.5) + 0.3 * dep), [(0.0, "#b4b6b2"), (0.5, "#c2c4bf"), (1.0, "#cfd0cb")])
    albedo = mix(base, bcol, bmix)
    h += dep * (0.00002 + (gran - 0.5) * 0.00005)                          # a thin powdery crust

    # --- 6. bright metal: fresh scratches and the few dresser lines that cut through ----------------
    sline = smoothstep(0.58, 0.72, 0.5 + stri) * smoothstep(0.52, 0.62, ssel) * sat(scover * 1.4)
    rubn = t.fbm(0.02, 3, 0.6, seed=41)
    rub = sat(t.curv(form, 0.006) / 0.00010) * (0.35 + 1.2 * rubn)        # skin rubbed thin on proud rims
    rub = smoothstep(0.30, 1.25, t.blur(rub, 0.0015) + 0.55 * (gran - 0.5) + 0.35 * (peel - 0.5)) * (1 - dep)
    fresh = smoothstep(0.42, 0.5, scr_r)
    bare = sat(sline * 0.9 + smoothstep(0.25, 0.6, scr) * fresh) * (1 - 0.7 * dep)
    old = smoothstep(0.2, 0.6, scr) * (1 - fresh)                          # healed-over scratch: darker line
    albedo = mix(albedo, hexlin("#585c60"), old * 0.6)
    albedo = mix(albedo, hexlin("#a7abb0"), np.maximum(bare, rub * 0.32))

    # --- PBR ------------------------------------------------------------------------------------------
    macro = t.fbm(0.25, 2, 0.5, seed=50)                                   # damp vs dry areas
    rough = 0.60 + 0.16 * (macro - 0.5) + 0.10 * (pt - 0.5) + 0.08 * (m2 - 0.5) + 0.05 * low
    rough = mix(rough, 0.50 + 0.06 * (m2 - 0.5), washed * 0.85)
    rough = rough - 0.11 * sat(burn)
    rough = mix(rough, 0.89 + 0.06 * (gran - 0.5), sat(dep + haze * 0.35 * cut))
    rough = mix(rough, 0.36 + 0.08 * (peel - 0.5), bare)
    rough = mix(rough, 0.44, rub * 0.75 * (1 - bare))
    metal = np.maximum(bare, rub * 0.5)
    lum = albedo @ LUMA
    pl = t.blur(lum, 0.06)
    print("DBG lead lum %.3f dep %.3f haze %.3f bare %.4f swathe %.3f | patch lum %.3f..%.3f" % (
        float(lum.mean()), float(dep.mean()), float(haze.mean()), float(metal.mean()), float(scover.mean()),
        float(pl.min()), float(pl.max())))
    return dict(albedo=albedo, height=h, rough=sat(rough), metal=sat(metal))
