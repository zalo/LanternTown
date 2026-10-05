"""Oak end grain: the cross-cut end of a log / boxed-heart beam, hand-sawn, grey and dry. Tintable.

A cross-cut shows the tree head-on and everything on it is organised by two directions - round the
pith (rings) and away from it (rays, checks):
  * ONE pith sits inside the tile, off-centre, and the growth rings are closed, slightly lobed and
    eccentric rounds about it (wide years on one side, tight on the other): juvenile wood of 4-5 mm
    rings at the heart, 2-3.5 mm outside, in runs of fat and lean years. Out to ~115 mm that is dark
    heartwood; beyond lies the sapwood - pale, soft, grey, worm-eaten, its rings faint - and that is
    where the tile meets its neighbour: the ring field of the next repeat is merged in smoothly (a
    wide soft minimum of the two radii plus a domain warp), so no kink or cusp exists anywhere;
  * oak is ring-porous: each year starts with about a millimetre of open earlywood vessels (soft,
    grey, eroded back up to 0.45 mm) and densifies to a dark line where the year ends. A few
    compression rings are darker, a few fat years carry a false ring. Colour, height and gloss come
    from that one profile;
  * medullary rays run along the ring normal, so they radiate from the pith: hair-fine spindles and a
    few broad oak rays in broken segments, hard, pale, slightly proud;
  * drying checks follow the rays: a small star at the heart, four wedge-shaped seasoning checks of
    similar rank spread round the face (2 mm open at the outer end, hairline at the inner, with a
    shoulder that falls towards the opening), a handful of medium ones, swarms of hairlines, a few
    turning into a ring shake;
  * the cut and its age: an uneven hand-saw face (broad steps where the saw wandered, straight oblique
    kerf that ignores the rings, burnished streaks along the kerf and torn fuzzy patches), grey
    weathering in clouds that ignore the rings, tide marks, beetle flight holes in the sapwood.
"""
from tk.tex import *

META = {"normal_strength": 1.0}

N_TAB = 130                                                                       # ring table (pith -> out)
PITH = (0.44, 0.41)                                                               # in tile units
R_HEART = 0.112                                                                   # heartwood radius (m)


def build(t):
    X, Y, W, H = t.X, t.Y, t.W, t.H
    TAU = 2 * np.pi
    f32 = np.float32
    u, v = X / W, Y / H
    px, py = t.px, t.py
    rng = np.random.default_rng(1851)

    def unit(a):                                                                  # t.noise -> zero mean, unit variance
        return (a - 0.5) / 0.18

    def field(a, x, y):
        return a[int(np.floor((H - y) / py)) % t.h, int(np.floor(x / px)) % t.w]

    def dir_noise(sig_a, sig_c, ang, seed):
        """Unit-variance noise with gaussian features sig_a long along direction `ang` (rad, Y up), sig_c across."""
        wn = np.random.default_rng(seed).standard_normal((t.h, t.w))
        fx = np.fft.fftfreq(t.w, d=px)[None, :]
        fy = -np.fft.fftfreq(t.h, d=py)[:, None]                                  # rows run downwards
        fa = fx * np.cos(ang) + fy * np.sin(ang)
        fc = -fx * np.sin(ang) + fy * np.cos(ang)
        filt = np.exp(-2 * np.pi ** 2 * ((fa * sig_a) ** 2 + (fc * sig_c) ** 2))
        filt[0, 0] = 0
        out = np.fft.ifft2(np.fft.fft2(wn) * filt).real
        return (out / (out.std() + 1e-12)).astype(f32)

    # ------------------------------------------------------------------ ring radius: closed rounds about one pith
    ph = rng.random(12) * TAU
    th0 = 0.85                                                                    # the side of the fat years
    wx = unit(t.noise(0.12, seed=1)) * 0.0040 + unit(t.noise(0.05, seed=102)) * 0.0012   # the trunk is not a circle
    wy = unit(t.noise(0.12, seed=101)) * 0.0040 + unit(t.noise(0.05, seed=103)) * 0.0012
    kb = 1.0 / 0.019                                                              # width of the merge with the next repeat
    acc = np.zeros((t.h, t.w), dtype=np.float64)
    for i in range(-2, 3):
        for j in range(-2, 3):
            dx = X + wx - (PITH[0] + i) * W
            dy = Y + wy - (PITH[1] + j) * H
            r = np.sqrt(dx * dx + dy * dy)
            th = np.arctan2(dy, dx)
            s = (1 + 0.13 * np.cos(th - th0) + 0.055 * np.cos(2 * th + ph[1]) + 0.035 * np.cos(3 * th + ph[2])
                 + 0.02 * np.cos(5 * th + ph[3]))
            acc += np.exp(-kb * r / s)
    rho = np.maximum(-np.log(acc) / kb, 0.0).astype(f32)                          # metres of growth from the pith

    gx, gy = t.slope(t.blur(rho, 0.0012))
    g = np.sqrt(gx ** 2 + gy ** 2)
    NX, NY = (gx / np.maximum(g, 1e-6)).astype(f32), (gy / np.maximum(g, 1e-6)).astype(f32)   # radial (rays, checks)
    g = np.clip(g, 0.12, 1.6).astype(f32)                                         # growth per metre of surface
    sapz = unit(t.noise(0.045, seed=104)) * 0.004
    sap = smoothstep(R_HEART - 0.006, R_HEART + 0.010, rho + sapz).astype(f32)    # sapwood (boundary follows a ring)
    fade = smoothstep(R_HEART - 0.014, R_HEART + 0.030, rho + sapz).astype(f32)   # rings relax towards the repeat
    fade = np.maximum(fade, 0.0) * 0.72 + 0.28 * smoothstep(0.150, 0.172, rho + sapz).astype(f32)
    # growth stops: beyond ~175 mm the last few years stretch out and end, so no small rounds close on the far corner
    R_END, SPAN = 0.169, 0.007
    rho_r = np.where(rho < R_END, rho, R_END + SPAN * np.tanh((rho - R_END) / SPAN)).astype(f32)
    g_r = (g * np.where(rho < R_END, 1.0, 1.0 - np.tanh((rho - R_END) / SPAN) ** 2)).astype(f32)
    ringy = smoothstep(0.04, 0.30, g_r).astype(f32)                               # 0 where there are no years left
    heart = np.exp(-(rho / 0.030) ** 2).astype(f32)                               # juvenile core
    pith = np.exp(-(rho / 0.0021) ** 2).astype(f32)

    # ------------------------------------------------------------------ helpers working along the radial direction
    dcol, drow = NX / px, -NY / py                                                # px per metre travelled outwards
    J, I = np.meshgrid(np.arange(t.h, dtype=f32), np.arange(t.w, dtype=f32), indexing='ij')

    def tap(a, s):
        ii = np.rint(I + s * dcol).astype(np.int32) % t.w
        jj = np.rint(J + s * drow).astype(np.int32) % t.h
        return a[jj, ii]

    def trace_mean(a, half, step):
        n = int(round(half / step))
        acc_ = t.zeros()
        wsum = 0.0
        for k in range(-n, n + 1):
            wk = 1.0 - abs(k) / (n + 1.0)
            acc_ += tap(a, k * step) * wk
            wsum += wk
        acc_ /= wsum
        return (acc_ - acc_.mean()) / (acc_.std() + 1e-9)

    def splat_rays(n, lmin, lmax, wmin, wmax, taps, seed, seg=None):
        """Medullary rays: n spindles marched along the radial field, anti-aliased splats. With `seg` a ray is a
        chain of segments about `seg` m long with gaps between them."""
        r = np.random.default_rng(seed)
        x = r.random(n) * W
        y = r.random(n) * H
        L = lmin + (lmax - lmin) * r.random(n) ** 1.6
        wd = wmin + (wmax - wmin) * r.random(n)
        amp = np.clip(wd / (px * len(taps)), 0.0, 1.0) * 0.5 * (0.55 + 0.45 * r.random(n))
        nseg = (L / seg) * (0.7 + 0.6 * r.random(n)) if seg else None
        phs = r.random(n)
        step = 0.5 * px
        idx, wts = [], []
        for k in range(int(lmax / step) + 1):
            s = k * step / L
            act = s < 1.0
            if not act.any():
                break
            ii = np.floor(x / px).astype(np.int64) % t.w
            jj = np.floor((H - y) / py).astype(np.int64) % t.h
            ax, ay = NX[jj, ii], NY[jj, ii]
            env = np.sin(np.pi * np.clip(s, 0, 1)) ** 0.8                         # fades out at both ends
            if seg:
                env = env * smoothstep(-0.45, 0.25, np.sin(TAU * (nseg * s + phs)))
            wgt = amp * env * act
            for o in taps:
                fx = (x + ay * o * wd) / px - 0.5
                fy = (H - (y - ax * o * wd)) / py - 0.5
                i0 = np.floor(fx).astype(np.int64)
                j0 = np.floor(fy).astype(np.int64)
                tx, ty = fx - i0, fy - j0
                for dj, di, ww in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)), (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
                    idx.append(((j0 + dj) % t.h) * t.w + (i0 + di) % t.w)
                    wts.append(wgt * ww)
            x = x + ax * step
            y = y + ay * step
        acc_ = np.bincount(np.concatenate(idx), weights=np.concatenate(wts), minlength=t.h * t.w)
        return np.clip(acc_.reshape(t.h, t.w), 0, 1).astype(f32)

    # ------------------------------------------------------------------ rays: radiate from the pith, short and broken
    rayfade = (1 - 0.45 * fade) * (1 - pith)
    n_fine = int(W * H / (0.0024 * 0.0085))
    ray_f = splat_rays(n_fine, 0.004, 0.016, 0.00014, 0.00030, (0.0,), 71) * rayfade
    n_broad = int(W * H / (0.015 * 0.040))
    ray_b = splat_rays(n_broad, 0.020, 0.075, 0.00045, 0.00080, (-0.33, 0.0, 0.33), 72, seg=0.012) * rayfade

    # ------------------------------------------------------------------ the years: widths and characters
    kk = np.arange(N_TAB)
    base_w = 0.0028 + 0.0014 * np.exp(-kk / 7.0)                                  # juvenile 4.2 mm -> mature 2.8 mm
    z = rng.standard_normal(N_TAB)
    ker = (1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1)
    zs = sum(np.roll(z, k - 5) * wk for k, wk in enumerate(ker))
    zs = zs / zs.std()
    wr = np.exp(0.46 * zs + 0.24 * rng.standard_normal(N_TAB))                    # fat and lean years in runs of 5-15
    wr = np.clip(wr / wr.mean(), 0.55, 1.70)
    wrm = base_w * wr / wr.mean()                                                 # ring widths in metres of growth
    cb = np.concatenate([[0.0], np.cumsum(wrm)])

    fluting = unit(t.noise(0.027, seed=2, bandwidth=0.45)) * 0.00055              # +-0.4 ring, ~55 mm
    jit = unit(t.noise(0.006, seed=3)) * 0.00010
    phi1 = rho_r + (fluting + jit) * smoothstep(0.004, 0.02, rho) - 0.0005 * sat(t.blur(ray_b, 0.0006) * 2.4)

    def ring_fields(phi):
        pm = np.clip(phi, 0.0, cb[-1] - 1e-6)
        rid = np.clip(np.searchsorted(cb, pm, side='right') - 1, 0, N_TAB - 1)
        f = ((pm - cb[rid]) / wrm[rid]).astype(f32)                               # 0 = start of the year (pith side)
        return rid, f

    def per_ring():
        return rng.random(N_TAB).astype(f32)

    ew_abs = 0.00080 + 0.00075 * per_ring()                                       # pore band of this year (m)
    dk_ring = 0.78 + 0.44 * per_ring()                                            # how dark its latewood gets
    eg_ring = 0.90 + 0.20 * per_ring()                                            # how pale its earlywood is
    er_ring = 0.65 + 0.60 * per_ring()                                            # how far it has eroded
    z2 = rng.standard_normal(N_TAB)
    dec = sum(np.roll(z2, k - 3) * wk for k, wk in enumerate((1, 2, 3, 3, 3, 2, 1)))
    tone_ring = (0.032 * dec / dec.std() + 0.040 * (per_ring() - 0.5)).astype(f32)  # single years and a little of the decades
    z3 = rng.standard_normal(N_TAB)
    soft_ring = sum(np.roll(z3, k - 1) * wk for k, wk in enumerate((1, 2, 1)))
    soft_ring = sat(0.5 + 0.5 * soft_ring / soft_ring.std()).astype(f32)         # groups of 2-4 soft years
    comp = np.zeros(N_TAB, dtype=f32)                                             # compression years: dense and dark
    comp[(4 + np.arange(5) * 9 + rng.integers(0, 9, 5)) % N_TAB] = 0.6 + 0.4 * rng.random(5)
    dk_ring = dk_ring + 0.50 * comp
    false_pos = np.zeros(N_TAB, dtype=f32)                                        # false rings in fat years
    fat = np.flatnonzero((wrm[:55] > 0.0034))
    for k in rng.choice(fat, min(4, len(fat)), replace=False):
        false_pos[k] = 0.50 + 0.22 * rng.random()

    rid1, _f1 = ring_fields(phi1)

    # ------------------------------------------------------------------ radial checks: a hierarchy
    check = t.zeros()
    depth = t.zeros()
    major = t.zeros()
    gape = t.zeros()                                                              # how far the check has opened (0..1)
    ends = []

    def draw_path(P, hw, hwmax, dmax, big):
        nonlocal check, depth, major, gape
        for c0 in range(0, len(P) - 1, 40):
            Pc, hc = P[c0:c0 + 41], hw[c0:c0 + 41]
            m = hwmax + 2.5 * px
            i0, i1 = int(np.floor((Pc[:, 0].min() - m) / px)), int(np.ceil((Pc[:, 0].max() + m) / px))
            j0, j1 = int(np.floor((H - Pc[:, 1].max() - m) / py)), int(np.ceil((H - Pc[:, 1].min() + m) / py))
            ii = np.arange(i0, i1 + 1)
            jj = np.arange(j0, j1 + 1)
            xc = ((ii + 0.5) * px)[None, None, :]
            yc = (H - (jj + 0.5) * py)[None, :, None]
            ax_, ay_ = Pc[:-1, 0][:, None, None], Pc[:-1, 1][:, None, None]
            ex, ey = (Pc[1:, 0] - Pc[:-1, 0])[:, None, None], (Pc[1:, 1] - Pc[:-1, 1])[:, None, None]
            l2 = np.maximum(ex * ex + ey * ey, 1e-18)
            tt = np.clip(((xc - ax_) * ex + (yc - ay_) * ey) / l2, 0, 1)
            dd = np.sqrt((xc - ax_ - tt * ex) ** 2 + (yc - ay_ - tt * ey) ** 2)
            hwl = hc[:-1][:, None, None] * (1 - tt) + hc[1:][:, None, None] * tt
            thin = np.clip(2 * hwl / px + 0.10, 0, 1)
            cov = np.clip((hwl - dd) / px + 0.5, 0, 1) * thin
            # V-shaped in section: steep weathered walls falling to a narrow floor
            dep = dmax * (hwl / hwmax) ** 0.6 * np.clip(1 - dd / np.maximum(hwl, 0.35 * px), 0, 1) ** 0.75 * thin
            gp = cov * np.clip(hwl / 0.0010, 0, 1)
            cov = cov.max(axis=0).astype(f32)
            dep = dep.max(axis=0).astype(f32)
            gp = gp.max(axis=0).astype(f32)
            ix = np.ix_(jj % t.h, ii % t.w)
            check[ix] = np.maximum(check[ix], cov)
            depth[ix] = np.maximum(depth[ix], dep)
            gape[ix] = np.maximum(gape[ix], gp)
            if big:
                major[ix] = np.maximum(major[ix], cov * big)

    def wedge(S, peak, tail=0.0):                                                 # opens slowly, closes quickly
        prof = np.where(S < peak, (S / peak) ** 0.9, (np.maximum(1 - S, 0) / (1 - peak)) ** 0.6)
        wob = 0.80 + 0.40 * np.interp(S, np.linspace(0, 1, 9), rng.random(9))
        return np.maximum(prof * wob, tail * np.sin(np.pi * S) ** 0.5)

    def add_check(cx, cy, L, hwmax, dmax, pjump, peak, big=0.0, flip=False):
        step = 0.75 * px
        n = max(4, int(L / step))
        x = cx - field(NX, cx, cy) * L * 0.5
        y = cy - field(NY, cx, cy) * L * 0.5
        pts = [(x, y)]
        ss = [0.0]
        off = 0.0
        last = field(rid1, x, y)
        for k in range(1, n + 1):
            ax, ay = field(NX, x, y), field(NY, x, y)
            x, y = x + ax * step, y + ay * step
            cur = field(rid1, x, y)
            if cur != last:
                last = cur
                if rng.random() < pjump and 3 < k < n - 3:
                    j = (0.0004 + 0.0007 * rng.random()) * (-1.0 if (off > 0) == (rng.random() < 0.8) else 1.0)
                    off += j
                    pts.append((x, y))
                    ss.append(k / n)
                    x, y = x + ay * j, y - ax * j                                 # sideways along the ring boundary
            pts.append((x, y))
            ss.append(k / n)
        P = np.asarray(pts, dtype=np.float64)
        S = np.asarray(ss, dtype=np.float64)
        prof = wedge(1 - S if flip else S, peak)
        draw_path(P, hwmax * prof, hwmax, dmax, big)
        if big:
            k = int(len(P) * (0.3 + 0.4 * rng.random()))
            ends.append((P[k, 0], P[k, 1], hwmax))

    def pdist(c, q):
        dx = abs(c[0] - q[0])
        dy = abs(c[1] - q[1])
        return np.hypot(min(dx, W - dx), min(dy, H - dy))

    cdens = smoothstep(0.32, 0.68, t.noise(0.10, seed=40))                        # checks come in loose groups
    radial_ok = smoothstep(0.45, 0.75, g) * smoothstep(0.010, 0.024, rho)         # where "radial" means something
    placed = []

    def place(n, ncand, lo=0.10, zone=None):
        zone = radial_ok if zone is None else zone
        out = []
        for _ in range(n):
            best, bd = None, -1.0
            k = 0
            while k < ncand:
                c = (rng.random() * W, rng.random() * H)
                if rng.random() > (lo + (1 - lo) * field(cdens, c[0], c[1])) * field(zone, c[0], c[1]):
                    continue
                k += 1
                dmin = min([pdist(c, q) for q in placed] or [9.0])
                if dmin > bd:
                    best, bd = c, dmin
            placed.append(best)
            out.append(best)
        return out

    def polar(rr, th):                                                            # tile position at growth radius rr
        s = 1 + 0.13 * np.cos(th - th0) + 0.055 * np.cos(2 * th + ph[1]) + 0.035 * np.cos(3 * th + ph[2])
        return ((PITH[0] * W + rr * s * np.cos(th)) % W, (PITH[1] * H + rr * s * np.sin(th)) % H)

    # four seasoning checks of similar rank, a quarter turn apart: wedges that open towards the bark
    tha = rng.random() * TAU
    for m in range(4):
        th = tha + m * TAU / 4 + (rng.random() - 0.5) * 1.05
        L = 0.030 + 0.016 * rng.random()
        c = polar((0.050, 0.086, 0.064, 0.096)[m] + 0.008 * (rng.random() - 0.5), th)
        placed.append(c)
        add_check(c[0], c[1], L, 0.00085 + 0.00030 * rng.random(), 0.0018 + 0.0004 * rng.random(), 0.16,
                  0.70 + 0.10 * rng.random(), big=1.0)
    # the star at the heart: short checks that start at the pith
    ths = tha + TAU / 8 + rng.random() * 0.4
    for m in range(3):
        th = ths + m * TAU / 3 + (rng.random() - 0.5) * 0.7
        L = 0.011 + 0.010 * rng.random()
        c = polar(0.0025 + L * 0.5, th)
        placed.append(c)
        add_check(c[0], c[1], L, 0.00028 + 0.00012 * rng.random(), 0.0010, 0.08, 0.35, big=0.6, flip=False)
    for i, (cx, cy) in enumerate(place(7, 40)):                                   # medium: 20-38 mm, 0.5-0.9 mm
        s = rng.random()
        add_check(cx, cy, 0.020 + 0.018 * s, 0.00024 + 0.00020 * s, 0.0011 + 0.0007 * s, 0.14, 0.55 + 0.25 * rng.random(),
                  big=0.7, flip=(i % 4 == 3))
    for (cx, cy) in place(10, 14):                                                # small: 12-25 mm, 0.25-0.35 mm
        add_check(cx, cy, 0.012 + 0.013 * rng.random(), 0.00012 + 0.00005 * rng.random(), 0.0006 + 0.0003 * rng.random(),
                  0.10, 0.45 + 0.35 * rng.random(), flip=rng.random() < 0.3)
    hair_ok = smoothstep(0.30, 0.55, g) * smoothstep(0.006, 0.02, rho)
    for (cx, cy) in place(40, 4, lo=0.04, zone=hair_ok):                          # hairlines: 8-25 mm, 0.15-0.25 mm
        add_check(cx, cy, 0.008 + 0.017 * rng.random(), 0.000075 + 0.00005 * rng.random(), 0.00025 + 0.0001 * rng.random(),
                  0.06, 0.4 + 0.3 * rng.random(), flip=rng.random() < 0.3)
    for (cx, cy) in place(80, 2, lo=0.03, zone=hair_ok):                          # short hairlines
        add_check(cx, cy, 0.003 + 0.006 * rng.random(), 0.00007 + 0.00003 * rng.random(), 0.0002,
                  0.04, 0.4 + 0.3 * rng.random())

    # rings dip slightly where a large check crosses them
    pinch = sat(t.blur(major, 0.0030) * 5.0)
    phi = phi1 + 0.0007 * pinch
    rid, f = ring_fields(phi)
    wloc = (wrm[rid] / np.maximum(g_r, 0.02)).astype(f32)                                             # this year's width on the face (m)
    fw = px / wloc                                                                # one pixel, in ring fractions

    # ring shakes: a check turns and runs along the soft earlywood of one ring
    for k in rng.choice(len(ends), min(3, len(ends)), replace=False):
        ex, ey, hwm = ends[k]
        rk = int(field(rid, ex, ey))
        tx_, ty_ = -float(field(NY, ex, ey)), float(field(NX, ex, ey))            # along the ring
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        Lk = 0.010 + 0.014 * rng.random()
        ddx = (X - ex + W / 2) % W - W / 2
        ddy = (Y - ey + H / 2) % H - H / 2
        sd = (ddx * tx_ + ddy * ty_) * sgn / Lk                                   # 0 at the check, 1 at the tip
        near = (ddx * ddx + ddy * ddy) < 0.03 ** 2
        hw = np.minimum(hwm, 0.00040) * 0.55 * np.clip(1 - sd, 0, 1) ** 0.8 * (sd > -0.05)
        dd = np.abs(f * wloc - 0.00028)
        thin = np.clip(2 * hw / px + 0.10, 0, 1)
        cov = np.clip((hw - dd) / px + 0.5, 0, 1) * thin * (rid == rk) * (hw > 0) * near
        check = np.maximum(check, cov.astype(f32))
        depth = np.maximum(depth, (cov * 0.0008 * np.clip(1 - sd, 0, 1) ** 0.5).astype(f32))

    lips = sat(t.blur(check, 0.0007) * 2.6) * (1 - check)                         # 1-2 mm weathered margin
    shoulder = sat(t.blur(gape, 0.0016) * 2.6)                                    # 2-3 mm of face falling to an open check
    halo = sat(t.blur(major, 0.0035) * 3.2)                                       # weather soaks in beside the open checks
    dish = t.blur(major, 0.0070)                                                  # ... and the face has dished towards them

    # ------------------------------------------------------------------ ring anatomy: ONE profile for colour / height / gloss
    rc = ((1 - 0.70 * fade) * (0.25 + 0.75 * ringy)).astype(f32)                                            # ring contrast: faint in the sapwood
    s0 = np.clip(0.75 * fw, 0.035, 0.22)                                          # anti-aliased year boundary
    fwrap = np.where(f > 0.5, f - 1.0, f)
    rise = smoothstep(-s0, s0, fwrap)
    ew = np.clip(ew_abs[rid] * np.clip(np.sqrt(wloc / 0.0032), 1.0, 2.4) / wloc, 0.05, 0.58)   # lean years are mostly pores, fat ones have 2-3 rows
    fall = 1 - smoothstep(ew * 0.55, ew * 1.25 + s0, f)
    early = np.where(f > 0.78, rise, rise * fall).astype(f32) * (1 - pith) * ringy        # earlywood band
    lr = sat((f - ew) / (1 - ew))
    late = (1 - early)
    cr = comp[rid]
    term = smoothstep(1 - np.clip(0.00045 / wloc, 0.08, 0.3), 1.0, f)             # the last dense cells of the year
    fpos = false_pos[rid]
    falser = (np.exp(-((f - fpos) / (0.045 + s0 * 0.5)) ** 2) * (fpos > 0)).astype(f32)   # a soft dark line, no pores
    dark = late * (0.08 + 0.66 * lr ** (1.1 - 0.5 * cr) + 0.26 * term) * dk_ring[rid]
    dark = (dark + 0.40 * falser * late).astype(f32)                              # latewood densifies to the year's end
    pn = t.noise(max(0.00048, 1.6 * px), seed=4, bandwidth=0.6)
    pz = t.noise(0.004, seed=5)
    pores = smoothstep(0.50, 0.66, pn + (pz - 0.5) * 0.07)                        # big earlywood vessels, 0.25-0.35 mm
    pore_m = pores * early
    flame = smoothstep(0.3, 1.3, trace_mean(t.noise(0.0022, seed=6), 0.004, 0.0006))   # radial flames of small latewood pores
    flame = flame * late * (1 - lr * 0.7)
    ray_f = ray_f * (1 - 0.5 * early)                                             # rays dim across the pore band
    ray_b = ray_b * (1 - 0.4 * early)
    rad_m = (1 - smoothstep(0.136, 0.160, rho)).astype(f32)                       # nothing radiates from the far corner
    stain = trace_mean(t.noise(0.008, seed=31), 0.040, 0.0030) * 0.5 * rad_m      # radial water / tannin staining
    grp = (zs / 2.0).astype(f32)[rid]                                             # > 0 in runs of fat years
    rfib = trace_mean(t.noise(max(0.0011, 2.5 * px), seed=32), 0.006, 0.0006)     # fine radial fibre: micro-checks, ray flecks
    rfib = np.clip(rfib, -2, 2) * (1 - pith) * (1 - 0.5 * early) * rad_m

    # ------------------------------------------------------------------ the saw: broad steps, fine kerf, patches
    def kerf(a, b, nteeth, seed):
        r = np.random.default_rng(seed)
        n = 4096
        s = (np.arange(n) + 0.5) / n
        pos = (np.arange(nteeth) + r.random(nteeth) * 0.7) / nteeth
        prof = np.zeros(n)
        for k in range(nteeth):
            dd = (s - pos[k] + 0.5) % 1.0 - 0.5
            wdt = (0.10 + 0.10 * r.random()) / nteeth
            dp = 0.35 + 0.65 * r.random()
            prof -= dp * np.clip(1 - np.abs(dd) / wdt, 0, 1) ** 1.5              # the score left by a tooth
            nxt = (pos[(k + 1) % nteeth] - pos[k]) % 1.0
            ramp_ = np.where((dd > 0) & (dd < nxt), dd / max(nxt, 1e-6), 0.0)
            prof += 0.45 * dp * (ramp_ - 0.5) * ((dd > 0) & (dd < nxt))          # slight step between strokes
        sc = np.mod(a * u + b * v, 1.0)
        return np.interp(sc, s, prof, period=1.0).astype(f32)

    def steps(a, b, nst, seed, soft):
        """Where the saw wandered: a few irregular shallow steps across the whole face."""
        r = np.random.default_rng(seed)
        n = 4096
        s = (np.arange(n) + 0.5) / n
        pos = np.sort((np.arange(nst) + 0.05 + r.random(nst) * 0.9) / nst)
        hgt = 0.35 + 0.65 * r.random(nst)
        prof = np.zeros(n)
        for k in range(nst):
            nxt = (pos[(k + 1) % nst] - pos[k]) % 1.0
            dd = (s - pos[k]) % 1.0
            prof += np.where(dd < nxt, hgt[k] * (dd / nxt - 0.5), 0.0)
        kf = np.fft.fftfreq(n, d=1.0 / n)
        prof = np.fft.ifft(np.fft.fft(prof) * np.exp(-2 * (np.pi * kf * soft) ** 2)).real
        sc = np.mod(a * u + b * v, 1.0)
        return np.interp(sc, s, prof, period=1.0).astype(f32)

    ang1 = np.arctan2(-5.0, 9.0)                                                  # direction of the kerf lines (5u + 9v = c)
    ang2 = np.arctan2(-4.0, 11.0)
    k1 = kerf(5, 9, 10, 91)                                                       # 29 deg, ~2.9 mm pitch
    k2 = kerf(4, 11, 8, 92)                                                       # 20 deg, ~3.2 mm pitch
    kz = dir_noise(0.050, 0.016, ang1, 9)                                         # each stroke leaves a band along itself
    ka = smoothstep(-0.25, 0.45, kz)
    kb_ = smoothstep(-0.45, -1.05, kz) * smoothstep(-0.6, 0.2, dir_noise(0.045, 0.020, ang2, 19))
    saw = k1 * ka + k2 * kb_                                                      # partly weathered away
    saw_amt = sat(ka + kb_)
    stp = steps(2, 3, 3, 93, 0.020) * (0.40 + 0.60 * smoothstep(0.3, 0.7, t.noise(0.09, seed=23)))
    stp = stp + 0.7 * steps(1, 4, 3, 94, 0.024) * (0.30 + 0.70 * smoothstep(0.35, 0.7, t.noise(0.08, seed=24)))

    mz = dir_noise(0.034, 0.0075, ang1, 17)                                       # streaks along the stroke
    burn = smoothstep(0.20, 0.90, mz) * (1 - 0.6 * sap)                           # where the saw burnished the dense wood
    burn = (burn * (0.55 + 0.45 * late)).astype(f32)                              # ... the hard latewood takes the polish
    fz = t.fbm(0.065, 3, 0.5, seed=17)
    fuzz = sat(smoothstep(0.50, 0.30, fz) * smoothstep(0.3, 0.6, t.noise(0.05, seed=27)) + 0.45 * sap) * (1 - burn)
    torn = t.fbm(max(0.0014, 3 * px), 3, 0.6, seed=10) - 0.5                      # torn fibre stubble
    nap = t.fbm(max(0.0009, 2.2 * px), 2, 0.6, seed=28) - 0.5                     # raised fuzzy nap

    # ------------------------------------------------------------------ beetle flight holes: loose clusters in the sapwood
    hole = t.zeros()
    hdep = t.zeros()
    frass = t.zeros()
    centres = []
    for _ in range(3):
        best, bd = None, -1.0
        nc = 0
        while nc < 30:
            c = (rng.random() * W, rng.random() * H)
            if field(sap, c[0], c[1]) < 0.9:
                continue
            nc += 1
            dmin = min([pdist(c, q) for q in centres] or [9.0])
            if dmin > bd:
                best, bd = c, dmin
        centres.append(best)
    hpts = []
    for ci, (cx, cy) in enumerate(centres):
        for _ in range((5, 4, 4)[ci]):
            for _try in range(40):
                hx, hy = cx + rng.standard_normal() * 0.016, cy + rng.standard_normal() * 0.016
                if all(np.hypot(hx - q[0], hy - q[1]) > 0.006 for q in hpts) and field(sap, hx % W, hy % H) > 0.3:
                    break
            hpts.append((hx, hy))
            r0 = 0.00050 + 0.00040 * rng.random()
            p1, p2, p3 = rng.random(3) * TAU
            m = r0 * 1.4 + 0.0012
            ii = np.arange(int(np.floor((hx - m) / px)), int(np.ceil((hx + m) / px)) + 1)
            jj = np.arange(int(np.floor((H - hy - m) / py)), int(np.ceil((H - hy + m) / py)) + 1)
            xx = ((ii + 0.5) * px)[None, :] - hx
            yy = (H - (jj + 0.5) * py)[:, None] - hy
            rr = np.sqrt(xx * xx + yy * yy)
            th = np.arctan2(yy, xx)
            rad = r0 * (1 + 0.11 * np.cos(2 * th + p1) + 0.07 * np.cos(3 * th + p2) + 0.04 * np.cos(5 * th + p3))
            cov = np.clip((rad - rr) / px + 0.5, 0, 1).astype(f32)
            dp = (np.clip(1 - (rr / rad) ** 2, 0, 1) ** 0.5).astype(f32)
            ix = np.ix_(jj % t.h, ii % t.w)
            hole[ix] = np.maximum(hole[ix], cov)
            hdep[ix] = np.maximum(hdep[ix], dp)
            if rng.random() < 0.36:
                rim = np.exp(-((rr - rad - 0.00030) / 0.00030) ** 2) * (0.6 + 0.4 * np.cos(th + p1))
                frass[ix] = np.maximum(frass[ix], (np.clip(rim, 0, 1) * (1 - cov)).astype(f32))

    pin = t.speckle(density=26000.0, size=0.0009, seed=51, soft=0.6) * sap * smoothstep(0.35, 0.6, t.noise(0.03, seed=52))
    pin = pin.astype(f32)                                                         # powder-post pinholes in the sapwood

    # ------------------------------------------------------------------ height (metres)
    wth = smoothstep(0.34, 0.66, t.fbm(0.13, 3, 0.5, seed=14))                    # the weathered side of the face
    erode = (0.72 + 0.48 * wth) * er_ring[rid] * (1 - 0.45 * burn) * (1 - 0.50 * fade) * (1 - 0.35 * heart) * (1 - 0.6 * smoothstep(0.150, 0.172, rho))
    macro = (t.fbm(0.15, 3, 0.5, seed=13) - 0.5) * 0.0010 + (t.fbm(0.045, 2, 0.5, seed=12) - 0.5) * 0.0006   # the cut is not a plane
    ringh = t.blur((-early * 0.00042 * erode + dark * 0.00010 * erode).astype(f32), 0.00044)   # weather-rounded troughs
    lean = t.blur((sat(-grp) * rc).astype(f32), 0.0012)                           # runs of lean years are soft: worn lower
    wash = t.blur((soft_ring[rid] * rc).astype(f32), 0.0011) - 0.5 * rc           # groups of soft years weather into a washboard
    height = macro + stp * 0.00100 + ringh - lean * 0.00024 - wash * (0.00034 + 0.00024 * wth) * (1 - 0.5 * burn) * (1 - 0.4 * heart) - pore_m * 0.00008 - flame * 0.00003 + rfib * 0.000025
    height = height + ray_f * 0.00004 + ray_b * 0.00009 + saw * (0.00015 + 0.00010 * burn)
    height = height + torn * 0.00010 * (0.35 + 0.65 * early + 0.9 * sap) * (0.7 + 0.5 * wth) + nap * 0.00020 * fuzz
    height = height - t.blur(sap, 0.004) * 0.00030                                # the soft sapwood has weathered back
    height = height - dish * 0.0008 - shoulder * 0.00034 - lips * 0.00014
    height = height - depth - pith * 0.0004 - hole * 0.0003 - hdep * 0.0009 + frass * 0.00004 - pin * 0.00035
    height = height.astype(f32)

    # ------------------------------------------------------------------ colour
    cav = t.cavity(height, 0.0012, 0.00025)
    tone = t.fbm(0.12, 3, 0.5, seed=15)                                           # clouds that ignore the rings
    val = (0.55 + 0.15 * (tone - 0.5) + tone_ring[rid] * rc + 0.025 * np.clip(grp, -1, 1) * rc + 0.11 * torn
           + 0.045 * np.clip(stain, -1, 1) + 0.020 * rfib - 0.025 * heart + 0.02 * sap)
    albedo = ramp(sat(val), [(0.0, "#5f4835"), (0.35, "#7c5f46"), (0.7, "#967757"), (1.0, "#ab8e69")])
    base_l = luminance(albedo)[..., None]
    dvar = 0.75 + 0.5 * t.noise(0.012, seed=21)
    albedo = albedo * (1.08 - 0.40 * rc * np.minimum(dark * dvar, 1.35) + 0.08 * flame)[..., None]   # latewood darkens to the year's end
    ewc = base_l * (hexlin("#a39680") / luminance(hexlin("#a39680")[None, None, :])[0, 0]) * 1.16
    albedo = mix(albedo, ewc * eg_ring[rid][..., None], early * 0.75 * (0.35 + 0.65 * rc))   # earlywood: paler, greyer
    albedo = mix(albedo, albedo * 0.66, pore_m * 0.50)                            # open vessels hold dirt
    albedo = albedo * (1.0 + 0.12 * ray_f + 0.17 * ray_b)[..., None]
    albedo = mix(albedo, hexlin("#ab9c84"), sat(ray_f * 0.05 + ray_b * 0.18))
    albedo = albedo * (1.0 + 0.16 * saw * saw_amt + 0.04 * stp + 0.05 * burn)[..., None]   # dirt in the scores, rubbed crests
    # weathering: warm grey in clouds that ignore the rings - also in the sapwood, beside the checks
    gz = smoothstep(0.30, 0.78, t.fbm(0.10, 4, 0.55, seed=16))
    gm = sat(0.20 + gz * (0.50 + 0.30 * early) + halo * 0.5 + lips * 0.5 + sap * 0.22 + fuzz * 0.20 - burn * 0.20)
    wg = hexlin("#8a8074")
    grey = luminance(albedo)[..., None] * (wg / luminance(wg[None, None, :])[0, 0])
    albedo = mix(albedo, grey * 0.97, gm * 0.56)
    albedo = albedo * (1.0 - 0.07 * gz - 0.08 * halo)[..., None]                  # end grain wicks water and dirt
    albedo = mix(albedo, grey * 1.12, lips * 0.40)                                # bleached, fibrous lips of the checks
    tn = t.fbm(0.085, 3, 0.5, seed=41)                                            # tide marks of old wettings
    tide_in = smoothstep(0.585, 0.63, tn)
    tide_rim = np.exp(-((tn - 0.60) / 0.016) ** 2) * smoothstep(0.35, 0.6, t.noise(0.04, seed=42))
    tn2 = t.fbm(0.06, 3, 0.5, seed=43)
    tide_rim = tide_rim + 0.7 * np.exp(-((tn2 - 0.33) / 0.014) ** 2) * smoothstep(0.4, 0.65, t.noise(0.05, seed=44))
    albedo = albedo * (1.0 - 0.035 * tide_in - 0.055 * sat(tide_rim))[..., None] * np.array([1.0, 0.99, 0.97], dtype=f32)
    albedo = mix(albedo, albedo * 0.84, sat(cav * 0.4))
    albedo = mix(albedo, hexlin("#a8977c"), frass * 0.40)
    cdeep = sat(depth / 0.0014)
    albedo = mix(albedo, hexlin("#4a3a2c"), pith * 0.8)                           # the pith: a dark soft fleck
    albedo = mix(albedo, hexlin("#57483a"), check * (0.40 + 0.40 * cdeep))        # hairlines only darken; walls are weathered wood
    albedo = mix(albedo, hexlin("#40342a"), check * cdeep * 0.80)                 # dry oak in shade, not a black slot
    albedo = mix(albedo, hexlin("#3d3128"), hole * 0.90)
    albedo = mix(albedo, hexlin("#4a3d31"), pin * 0.55)
    target = np.array([0.231, 0.161, 0.101], dtype=f32)                           # grey-brown weathered oak, luminance ~0.17
    gain = np.clip(target / albedo.reshape(-1, 3).mean(0), 0.75, 1.3)
    albedo = albedo * gain[None, None, :]
    albedo = np.maximum(albedo, hexlin("#3a2f26")[None, None, :])

    # ------------------------------------------------------------------ roughness
    rough = 0.65 - 0.19 * burn + 0.08 * (soft_ring[rid] - 0.5) * rc + 0.09 * fuzz - 0.03 * dark * rc + 0.05 * gz + 0.04 * sap
    rough = mix(rough, 0.90, early * 0.9 * (0.5 + 0.5 * rc))
    rough = rough + 0.05 * pore_m + 0.04 * flame + 0.08 * torn - 0.05 * saw * saw_amt + 0.04 * gm * late
    rough = mix(rough, 0.60, sat(ray_f * 0.8 + ray_b * 0.9))
    rough = mix(rough, 0.90, sat(lips * 0.85 + shoulder * 0.4))
    rough = mix(rough, 0.94, sat(check + hole + pith + pin))
    rough = mix(rough, 0.88, frass * 0.6)
    rough = np.clip(rough, 0.46, 0.96)

    ao = t.ao(t.blur(height + 0.45 * depth, 0.00032), radius=0.004, strength=0.62)               # soft: no black dashes in the pore bands
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0, ao=ao)
