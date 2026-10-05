"""Rough-hewn / split stone face (no joints): rock-faced grey limestone, pitted and point-chiselled.

Built in the order the quarryman and the mason make it:
  1. the block is SPLIT and pitched: the face is a mosaic of flake scars 7-12 cm across.  Every
     scar is an individually tilted, slightly dished fracture facet; where two scars meet the
     surface is the LOWER of the two, so they meet in crisp arrises, and where a scar ends short
     it climbs out in a broad inclined riser broken into two or three hackle steps;
  2. smaller secondary flakes (3-5 cm) are knocked off the proud arrises: the same construction
     one scale down, fresher and paler; then the hackly 1-2 cm chips of the break itself;
  3. every fracture facet carries its own fracture marks: conchoidal ribs concentric about the
     point of percussion on its edge and plumose hackle radiating from that same point;
  4. the proud stone is then taken down with the POINT: one continuous, loosely spaced field of
     individual furrows (25-40 mm long, V-section, deep at the strike, bursting out in a small
     spall at the far end), each driven roughly the way its neighbours are, never in tidy patches;
  5. punch / weather pits, crystalline grain, dark mineral flecks, a little dirt in the hollows.
"""
from tk.tex import *
import math

META = {"ao_strength": 0.55}
DEBUG = False


def _win(t, cx, cy, half):
    """Wrapping pixel window around (cx, cy): (index, Xlocal, Ylocal)."""
    ix, iy = int(math.floor(cx / t.px)), int(math.floor((t.H - cy) / t.py))
    hw = min(int(half / t.px) + 2, t.w // 2 - 1)
    xs = np.arange(ix - hw, ix + hw + 1)
    ys = np.arange(iy - hw, iy + hw + 1)
    Xl = ((xs[None, :] + 0.5) * t.px - cx).astype(F)
    Yl = ((t.H - (ys[:, None] + 0.5) * t.py) - cy).astype(F)
    return np.ix_(ys % t.h, xs % t.w), Xl, Yl


def _at(t, a, cx, cy):
    return float(a[int((t.H - cy) / t.py) % t.h, int(cx / t.px) % t.w])


def _dots(t, density, size, seed, soft=0.6, keep=0.5):
    """Speckle with a 4x size range.  One candidate per jittered grid cell, of which only a
    fraction `keep` exists, so spacing is irregular (no halftone).  (mask 0..1, per-dot random)."""
    cell = max(1.0 / math.sqrt(density), t.px * 3)
    nx, ny = max(2, int(round(t.W / cell))), max(2, int(round(t.H / cell)))
    cw, ch = t.W / nx, t.H / ny
    rng = np.random.default_rng(t.seed * 1000 + 4000 + seed)
    jx, jy, rr, kk = (rng.random((ny, nx)).astype(F) for _ in range(4))
    gx = np.floor(t.X / cw).astype(np.int64)
    gy = np.floor(t.Y / ch).astype(np.int64)
    out = np.zeros(t.X.shape, dtype=F)
    rv = np.zeros(t.X.shape, dtype=F)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = gx + ox, gy + oy
            wx, wy = cx % nx, cy % ny
            dx = (cx + jx[wy, wx]) * cw - t.X
            dy = (cy + jy[wy, wx]) * ch - t.Y
            r = size * 0.5 * (0.35 + 1.3 * rr[wy, wx] ** 2) * (kk[wy, wx] < keep)
            m = 1 - smoothstep(r * (1 - soft), r + 1e-9, np.sqrt(dx * dx + dy * dy))
            upd = m > out
            out = np.where(upd, m, out)
            rv = np.where(upd, rr[wy, wx], rv)
    return out, rv


def scars(t, pitch, jitter, seed, wx, wy, level, frac=1.0, rise=(0.0008, 0.002), slope=(0.03, 0.09),
          wgt=0.3, wall=(0.4, 0.9), ripple=0.0003, lam=(0.006, 0.011), hackle=0.0004,
          step=0.6, start=None, lvl_jit=0.0, ring=2):
    """Fracture facets.  The face is partitioned into irregular convex-ish plates (a weighted
    Voronoi partition of jittered seeds, bent by the shared warp field).  Every plate is an
    individually tilted, slightly dished fracture plane carrying conchoidal ribs + plumose
    hackle about its percussion point.  Beyond its own border a plate climbs in an inclined,
    hackle-stepped riser, and the surface is the LOWEST of all plates: so two plates meet in a
    plain arris where their levels agree and in a broad riser where they do not - a step whose
    height changes along its length and dies out, as on a real cleft face.  Never a closed pan.
    `level(cx, cy)` gives the plate datum; with `start` the plates only bite where they are
    lower than the existing surface (flakes struck off proud stone).
    Returns (height, plate id per pixel (-1 = start surface survives), n, riser mask 0..1)."""
    rng = np.random.default_rng(t.seed * 1000 + 7000 + seed)
    n = max(3, int(round(t.W / pitch)))
    m = max(3, int(round(t.H / pitch)))
    cw, ch = t.W / n, t.H / m
    R = 0.5 * math.sqrt(cw * ch)
    out = np.full((t.h, t.w), 1e3, dtype=F) if start is None else start.astype(F).copy()
    fid = np.full((t.h, t.w), -1, dtype=np.int64)
    ris = np.zeros((t.h, t.w), dtype=F)
    rr = rng.random((m, n, 24))
    px = (np.arange(n)[None, :] + 0.5 + (rr[:, :, 1] - 0.5) * jitter) * cw
    py = (np.arange(m)[:, None] + 0.5 + (rr[:, :, 2] - 0.5) * jitter) * ch
    pw = (rr[:, :, 5] - 0.5) * 2 * (wgt * R) ** 2 * 4            # power-diagram weights: uneven plate sizes
    k = 0
    for j in range(m):
        for i in range(n):
            r = rr[j, i]
            k += 1
            if r[0] > frac:
                continue
            cx, cy = px[j, i] % t.W, py[j, i] % t.H
            D = rise[0] + (rise[1] - rise[0]) * r[6]
            S = wall[0] + (wall[1] - wall[0]) * r[7]
            s = slope[0] + (slope[1] - slope[0]) * r[8]
            ta = r[9] * 2 * math.pi
            gx, gy = s * math.cos(ta), s * math.sin(ta)
            c = level(cx, cy) + (r[10] - 0.5) * lvl_jit
            # percussion point: on the plate's edge, on its low (deep) side
            pa = ta + math.pi + (r[11] - 0.5) * 1.6
            ox, oy = math.cos(pa) * R * 1.7, math.sin(pa) * R * 1.7
            lm = lam[0] + (lam[1] - lam[0]) * r[12]
            ar = ripple * (0.15 + 0.85 * r[13] ** 1.5)
            nh = int(26 + 30 * r[14])                    # hackle steps per turn
            ah = hackle * (0.25 + 0.95 * r[15])
            mh, ph, ph2 = int(2 + 5 * r[16]), r[17] * 6.283, r[18] * 6.283
            wst = 0.0045 + 0.004 * r[19]                 # tread of the hackle steps on the riser
            mst = step * (0.35 + 0.65 * r[20])
            idx, Xl, Yl = _win(t, cx, cy, R * 2.7)
            X2, Y2 = Xl + wx[idx], Yl + wy[idx]
            # signed distance outside this plate = deepest violation of a neighbour bisector
            e = None
            for dj in range(-ring, ring + 1):
                for di in range(-ring, ring + 1):
                    if di == 0 and dj == 0:
                        continue
                    jj, ii = (j + dj) % m, (i + di) % n
                    qx = px[jj, ii] + ((i + di) - ii) * cw - px[j, i]
                    qy = py[jj, ii] + ((j + dj) - jj) * ch - py[j, i]
                    ql = math.sqrt(qx * qx + qy * qy) + 1e-9
                    ek = (X2 * qx + Y2 * qy) / ql - 0.5 * ql - (pw[j, i] - pw[jj, ii]) / (2 * ql)
                    e = ek if e is None else np.maximum(e, ek)
            ee = np.clip(e, 0, None)
            rad2 = (X2 * X2 + Y2 * Y2) / (R * R)
            # riser: inclined face (always steeper than the plate's own tilt), broken into two or
            # three hackle steps parallel to the scarp that die out up the slope
            if mst > 0:
                riser = (S + s) * (ee + mst * wst / 6.283 * np.sin(6.283 * ee / wst) * np.exp(-ee / (2.6 * wst)))
            else:
                riser = (S + s) * ee
            h = c + gx * X2 + gy * Y2 + D * np.minimum(rad2, 4.0) + riser
            if ar > 0 or ah > 0:
                dx, dy = X2 - ox, Y2 - oy
                rho = np.sqrt(dx * dx + dy * dy)
                th = np.arctan2(dy * math.cos(pa) - dx * math.sin(pa), -(dx * math.cos(pa) + dy * math.sin(pa)))
                # conchoidal ribs: concentric about the percussion point, opening out and dying away
                fade = np.clip(1 - rho / (3.2 * R), 0, 1) * smoothstep(0.5 * R, 1.1 * R, rho)
                h = h + ar * np.sin(2 * math.pi * (rho / lm) ** 0.9 + ph2 + 1.2 * np.sin(th * 3 + ph)) * fade ** 1.5
                # plumose hackle: little steps radiating from the same point, growing away from it
                sw = nh * th / (2 * math.pi) + 0.45 * np.sin(mh * th + ph) + 0.25 * np.sin(rho / R * 5 + ph2)
                sw = sw - np.floor(sw)
                saw = np.minimum(sw / 0.8, (1 - sw) / 0.2)   # asymmetric tooth: no vertical cliffs
                h = h + ah * (saw - 0.5) * smoothstep(0.7 * R, 1.8 * R, rho)
            o = out[idx]
            win = h < o
            out[idx] = np.where(win, h, o)
            fid[idx] = np.where(win, k, fid[idx])
            ris[idx] = np.where(win, sat(ee / 0.004), ris[idx])
    return out, fid, k + 1, ris


def point_field(t, gate, ang, wx, wy, seed):
    """Point-chisel work as ONE continuous field of individual furrows.  Positions: dart throwing
    with an elliptical exclusion zone laid along the local driving direction (so marks keep an
    irregular 30-40 mm pitch along and 15-20 mm across, never a lattice); the zone grows where
    `gate` is weak, so the work thins out instead of ending at an outline.
    Each mark: a V furrow, steep where the point bit, running out where the chip burst into a
    shallow spall, with a small crushed burr at the far end.
    Returns (cut <= 0, burr >= 0, bruise halo 0..1, spall mask 0..1)."""
    rng = np.random.default_rng(t.seed * 1000 + 7100 + seed)
    cut, lip, halo, spall = t.zeros(), t.zeros(), t.zeros(), t.zeros()
    La, Lc = 0.040, 0.022
    nb = max(3, int(t.W / 0.075))
    mb = max(3, int(t.H / 0.075))
    buckets = {}
    marks = []
    for _ in range(int(t.W * t.H / (La * Lc) * 12)):
        x, y, ja = rng.random() * t.W, rng.random() * t.H, rng.random()
        g = _at(t, gate, x, y)
        if g < 0.12:
            continue
        a = _at(t, ang, x, y)
        ca, sa = math.cos(a), math.sin(a)
        grow = 1.0 + 1.3 * (1 - g) ** 1.5
        bi, bj = int(x / t.W * nb) % nb, int(y / t.H * mb) % mb
        ok = True
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                for (xj, yj) in buckets.get(((bi + di) % nb, (bj + dj) % mb), ()):
                    ddx = (xj - x + t.W * 0.5) % t.W - t.W * 0.5
                    ddy = (yj - y + t.H * 0.5) % t.H - t.H * 0.5
                    al, ac = ddx * ca + ddy * sa, -ddx * sa + ddy * ca
                    if (al / (La * grow)) ** 2 + (ac / (Lc * grow)) ** 2 < 1.0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        buckets.setdefault((bi, bj), []).append((x, y))
        marks.append((x, y, a + math.radians((ja - 0.5) * 34), g))
    for (cx, cy, a2, g) in marks:
        q = rng.random(10)
        c2, s2 = math.cos(a2), math.sin(a2)
        L = 0.022 + 0.028 * q[0] ** 1.2
        Wd = (0.0105 + 0.0045 * q[1]) * (0.8 + 0.2 * min(1.0, L / 0.035))
        D = (0.0023 + 0.0014 * q[2]) * (0.65 + 0.35 * g)
        idx, Xl, Yl = _win(t, cx, cy, L * 0.8 + 0.014)
        X2, Y2 = Xl + wx[idx] * 0.35, Yl + wy[idx] * 0.35
        al = X2 * c2 + Y2 * s2
        ac = -X2 * s2 + Y2 * c2 + (q[3] - 0.5) * 0.25 * (al * al) / L       # never ruler straight
        u = al / L + 0.5                                   # 0 strike .. 1 run-out
        v = np.abs(ac)
        prof = smoothstep(0.0, 0.14, u) * (1 - smoothstep(0.38, 1.0, u)) ** 0.8
        w = Wd * 0.5 * (0.5 + 0.5 * prof)
        gcut = -D * prof * np.clip(1 - v / w, 0, 1)
        # the chip that burst off ahead of the point: a shallow, wider spall scar at the run-out
        if q[4] < 0.7:
            Rb = 0.0065 + 0.005 * q[5]
            Db = (0.0005 + 0.0007 * q[6]) * (0.6 + 0.4 * g)
            rb2 = ((al - (0.30 + 0.12 * q[7]) * L) / (Rb * 1.25)) ** 2 + ((ac - (q[8] - 0.5) * 0.006) / Rb) ** 2
            sp = np.clip(1 - rb2, 0, 1)
            gcut = np.minimum(gcut, -Db * sp ** 0.7)
            spall[idx] = np.maximum(spall[idx], (sp > 0).astype(F))
        cut[idx] = np.minimum(cut[idx], gcut.astype(F))
        bump = 0.00050 * np.exp(-(((u - 1.02) / 0.10) ** 2 + (v / (Wd * 0.75)) ** 2))
        lip[idx] = np.maximum(lip[idx], bump.astype(F))
        hl = np.exp(-(((u - 0.5) / 0.7) ** 2 + (v / (Wd * 1.3)) ** 2) * 1.2)
        halo[idx] = np.maximum(halo[idx], hl.astype(F))
    if DEBUG:
        print('DBG point marks', len(marks))
    return cut, lip, halo, spall


def build(t):
    # ================================================================== height (metres)
    # ragged-edge field shared by all fracture geometry (so arrises are never ruler straight)
    wx = (t.noise(0.11, seed=1) - 0.5) * 0.010 + (t.fbm(0.03, 3, 0.5, seed=2) - 0.5) * 0.007
    wy = (t.noise(0.11, seed=3) - 0.5) * 0.010 + (t.fbm(0.03, 3, 0.5, seed=4) - 0.5) * 0.007
    lump = (t.fbm(0.30, 2, 0.5, seed=5) - 0.5) * 0.002

    # 1. primary fracture plates, 8-13 cm: tilted facets stepping down to one another over broad,
    #    hackle-stepped risers
    h1, id1, n1, ris1 = scars(t, 0.105, 0.85, 1, wx, wy, lambda x, y: _at(t, lump, x, y), lvl_jit=0.0038,
                              rise=(-0.0004, 0.0012), slope=(0.07, 0.18), wgt=0.30, wall=(0.22, 0.34),
                              ripple=0.00100, lam=(0.009, 0.015), hackle=0.00050, step=0.55)
    md = float(np.median(h1))
    h1 = md + 0.010 * np.tanh((h1 - md) / 0.010)                  # no freak peaks / holes
    # 2. secondary flakes, 3-6 cm, struck off the proud arrises (datum = local mean level)
    bl = t.blur(h1, 0.016)
    h2, id2, n2, ris2 = scars(t, 0.050, 0.95, 2, wx * 0.6, wy * 0.6,
                              lambda x, y: _at(t, bl, x, y) + 0.0004, lvl_jit=0.0022, frac=0.55,
                              rise=(0.0002, 0.0010), slope=(0.04, 0.16), wgt=0.35, wall=(0.30, 0.46),
                              ripple=0.00035, lam=(0.006, 0.009), hackle=0.00024, step=0.45, start=h1)
    fresh = t.blur((id2 >= 0).astype(F), 0.0015)
    # 2b. the hackly break itself: tiny angular chips (1-2 cm) all over, a fraction of a mm deep
    bl3 = t.blur(h2, 0.006)
    h3, id3, n3, ris3 = scars(t, 0.021, 1.0, 5, wx * 0.3, wy * 0.3,
                              lambda x, y: _at(t, bl3, x, y) + 0.00015, lvl_jit=0.0012, frac=0.55,
                              rise=(0.0, 0.0003), slope=(0.04, 0.15), wgt=0.35, wall=(0.28, 0.48),
                              ripple=0.0, hackle=0.0, step=0.0, start=h2, ring=1)

    def per(ids, n, k, seed):
        return np.random.default_rng(seed * 100 + k).random(n + 1).astype(F)[ids + 1]

    # 2c. lumpiness of the faces themselves (the break is never a clean plane)
    lmp = (t.fbm(0.045, 2, 0.5, seed=12) - 0.5) * 0.0021 + (t.ridged(0.016, 2, 0.5, seed=13) - 0.5) * 0.0007
    hs = h3 + lmp

    # 3. point work: continuous field, soft gate biased to the proud stone, smooth direction field
    prd = (hs - t.blur(hs, 0.045)) / 0.004
    gsrc = (t.fbm(0.30, 2, 0.5, seed=14) - 0.5) * 2.2 + t.blur(prd, 0.012) * 0.55 + (t.noise(0.09, seed=15) - 0.5) * 1.2
    lo, hi = np.percentile(gsrc, 42), np.percentile(gsrc, 74)
    gate = smoothstep(lo, hi, gsrc)
    ang = math.radians(58) + (t.noise(0.38, seed=16) - 0.5) * 5.2 + (t.noise(0.14, seed=17) - 0.5) * 1.2
    cut, lip, halo, spall = point_field(t, gate, ang, wx, wy, 3)
    tooled = sat(-cut / 0.0010)

    # 4. punch / weather pits of mixed size, in loose flurries
    pg = 0.25 + 0.75 * smoothstep(0.42, 0.62, t.fbm(0.13, 2, 0.5, seed=41))
    pit, pr = _dots(t, 950, 0.0150, 40, soft=0.85, keep=0.5)
    pit = t.warp(pit * pg * (0.55 + 0.45 * pr), 0.0012, 0.006, seed=44)
    mp, _ = _dots(t, 14000, 0.0032, 42, soft=0.8, keep=0.45)
    mp = mp * (0.3 + 0.7 * smoothstep(0.45, 0.65, t.fbm(0.07, 2, 0.5, seed=43)))

    # 5. crystalline grain (crisp) and granular break-out
    g1 = t.fbm(0.0038, 3, 0.68, seed=6)
    g2 = t.fbm(0.014, 3, 0.6, seed=7)
    xt = np.abs(g1 - 0.5) * 2                                     # sugary: sharp little crests

    height = t.blur(hs + cut + lip, 0.0008)
    height = (height + (g1 - 0.5) * 0.00030 - xt * 0.00010 + (g2 - 0.5) * 0.00050 * (1 - 0.5 * tooled)
              + (t.fbm(0.0045, 2, 0.6, seed=10) - 0.5) * 0.0005 * tooled          # crushed furrow floors
              - pit * 0.0026 - mp * 0.00040)

    # ================================================================== masks
    # fracture geometry alone (before the point and the pits): its creases and arrises
    hf = t.blur(hs, 0.0008)
    crease = t.cavity(hf, 0.003, 0.00045)
    arris = t.edges(hf, 0.003, 0.00045)
    cav_s = t.cavity(height, 0.004, 0.0012)
    cv = t.curv(height, 0.028)
    cav_l = smoothstep(0.0010, 0.0055, -cv)                        # broad hollows (soft)
    proud = smoothstep(0.0010, 0.0055, cv)                         # broad proud stone (soft)
    riser = sat(t.blur(np.maximum(ris1, ris2 * 0.8), 0.0015))
    rub = t.blur(proud, 0.004)                                     # rubbed proud stone: no hard outline
    grit = sat(0.55 + (g1 - 0.5) * 2.2)
    bruise = smoothstep(0.0003, 0.0011, -cut) * (0.55 + 0.45 * grit)   # crushed pale powder in the furrows

    # ================================================================== colour
    # per-plate character, blended over a few mm so nothing reads as a cut-out patch
    f_val = t.blur(per(id1, n1, 1, 11), 0.006)
    f_hue = t.blur(per(id1, n1, 2, 11), 0.008)
    f_age = t.blur(per(id1, n1, 3, 11), 0.005)
    c_val = per(id2, n2, 4, 12)
    # weathered skin: plates of the old face that neither a later flake nor the point has touched
    skin = smoothstep(0.38, 0.62, f_age) * (1 - fresh) * (1 - sat(tooled + halo * 0.6))
    skin = skin * (0.45 + 0.55 * smoothstep(0.38, 0.58, t.fbm(0.04, 3, 0.55, seed=19)))
    lam = t.fbm(0.022, 3, 0.5, seed=20, stretch=(9.0, 1.0))        # bedding laminae crossing the break
    mott = t.fbm(0.045, 4, 0.58, seed=22)
    tone = sat(0.5 + (f_val - 0.5) * 0.16 + (t.fbm(0.42, 2, 0.5, seed=21) - 0.5) * 0.10 + (mott - 0.5) * 0.40
               + (lam - 0.5) * 0.26 + (g1 - 0.5) * 0.40 + (g2 - 0.5) * 0.26 + fresh * (c_val - 0.35) * 0.14)
    alb = ramp(tone, [(0.0, "#8d8a82"), (0.25, "#9c978c"), (0.5, "#aaa498"), (0.75, "#b5ae9f"), (1.0, "#c0b9aa")])
    alb = overlay_tint(alb, "#d9c6a4", smoothstep(0.55, 0.9, f_hue) * 0.07)                  # warmer plates
    alb = overlay_tint(alb, "#bcc4c8", smoothstep(0.45, 0.1, f_hue) * 0.07)                  # cooler plates
    # skin: a touch darker and warmer (buff patina), peppered with grey-black micro-lichen / soot specks
    alb = alb * (1 - skin[..., None] * (1 - np.array([0.90, 0.875, 0.82], dtype=F)))
    lk, lr = _dots(t, 9000, 0.0050, 29, soft=0.7, keep=0.4)
    lk = t.warp(lk, 0.0012, 0.005, seed=31) * smoothstep(0.35, 0.6, t.fbm(0.06, 3, 0.55, seed=32))
    alb = mix(alb, hexlin("#6a665c"), lk * (0.10 + 0.30 * lr) * skin)
    # iron staining: small weak warm blooms following the laminae, ~5 % of the face
    iron = smoothstep(0.70, 0.92, t.fbm(0.07, 3, 0.55, seed=23, stretch=(2.2, 1.0))) * (0.45 + 0.55 * lam)
    alb = overlay_tint(alb, "#c39a66", iron * 0.24 * (1 - 0.7 * fresh) * (1 - tooled))
    # minerals: dark flecks of mixed size + irregular pale crystal faces
    fk, fr = _dots(t, 6000, 0.0036, 24, soft=0.6, keep=0.28)
    fk = t.warp(fk * smoothstep(0.40, 0.62, t.fbm(0.05, 3, 0.55, seed=27)), 0.0010, 0.004, seed=28)
    fk2 = smoothstep(0.80, 0.90, t.fbm(0.0055, 3, 0.6, seed=25))
    alb = mix(alb, hexlin("#56524b"), sat(fk * (0.15 + 0.5 * fr) + fk2 * 0.30))
    xl = smoothstep(0.76, 0.90, t.fbm(0.0048, 3, 0.62, seed=26))
    alb = mix(alb, hexlin("#d0cbbf"), xl * 0.30)
    # fresh small flakes: cleaner, paler, slightly cooler stone
    alb = mix(alb, (alb * 1.075) * np.array([0.985, 1.0, 1.02], dtype=F), fresh * 0.9)
    # bruised stone: the point crushes the furrow to pale powder; its spall and burr are paler too
    bru = sat(bruise + spall * 0.22 * grit + halo * 0.06 + lip * 1200)
    alb = mix(alb, (alb * 1.17 + 0.006) * np.array([0.98, 1.0, 1.025], dtype=F), bru)
    # chipped arrises and rubbed proud points: paler
    alb = mix(alb, alb * 1.10 + 0.004, sat(arris * 0.75 * (0.5 + 0.5 * grit) + proud * 0.12) * (1 - bru))
    # dirt settled at the foot of risers, in creases, hollows and pits: darker, warmer (the engine adds real grime)
    dirt = sat(crease * 0.85 * (1 - tooled) + riser * 0.22 + cav_l * 0.28 + pit * 0.7 + mp * 0.22)
    alb = mix(alb, alb * np.array([0.78, 0.74, 0.66], dtype=F), dirt)
    # the same weather as the dressed stone beside it (one family): a soft grey film in patches on
    # the old skin, loose colonies of pale crustose lichen with a few ochre rosettes
    film = smoothstep(0.45, 0.8, t.fbm(0.26, 4, 0.55, seed=45)) * (1 - fresh) * (1 - bru)
    alb = overlay_tint(alb, "#a5a29b", film * 0.22)
    lgate = smoothstep(0.50, 0.70, t.fbm(0.33, 3, 0.5, seed=46)) * (1 - fresh) * (1 - tooled)
    lich = t.speckle(density=420, size=0.022, seed=47, soft=0.4) * lgate
    lich = t.warp(lich, 0.002, 0.012, seed=48) * (0.55 + 0.45 * grit)
    lcol = mix(hexlin("#c3c3b9"), hexlin("#b09c62"), smoothstep(0.62, 0.72, t.noise(0.13, seed=49)))
    alb = mix(alb, lcol * (0.85 + 0.3 * g1)[..., None], lich * 0.40)
    alb = desaturate(alb, 0.10)
    alb = alb * (np.array([0.372, 0.356, 0.318], dtype=F) / alb.reshape(-1, 3).mean(0))

    # ================================================================== roughness
    # split stone is matt; hands, boots and rain have rubbed the proud points and arrises smoother;
    # crushed furrows and dirt-filled creases are the roughest
    rvar = t.fbm(0.035, 3, 0.55, seed=30)
    rough = 0.81 + (rvar - 0.5) * 0.16 + (g1 - 0.5) * 0.10
    rough = rough + fresh * 0.04 + bruise * 0.08 - skin * 0.09
    rough = rough - arris * 0.12 * (1 - tooled) - rub * (0.16 + 0.14 * grit) - xl * 0.18
    rough = rough + dirt * 0.10 + lich * 0.08 + film * 0.03

    if DEBUG:
        globals()['_dbg'] = dict(cav_s=cav_s, cav_l=cav_l, arris=arris, proud=proud, crease=crease, riser=riser, tooled=tooled, skin=skin, fresh=fresh, dirt=dirt, bru=bru, gate=gate)
        gx, gy = t.slope(height)
        sl = np.sqrt(gx * gx + gy * gy)
        print('DBG slope p50 %.2f p90 %.2f p99 %.2f p99.9 %.2f max %.2f' % tuple(np.percentile(sl, [50, 90, 99, 99.9, 100])))
        print('DBG tooled %.2f gate %.2f fresh %.2f skin %.2f iron %.3f dirt %.2f' % (
            float((tooled > 0.3).mean()), float(gate.mean()), float(fresh.mean()), float(skin.mean()),
            float((iron > 0.3).mean()), float(dirt.mean())))
        print('DBG rough p1 %.2f p50 %.2f p99 %.2f' % tuple(np.percentile(rough, [1, 50, 99])))
    return dict(albedo=alb, height=height, rough=np.clip(rough, 0.42, 0.97), metal=0.0)
