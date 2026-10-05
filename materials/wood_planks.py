"""Aged tongue-and-groove V-jointed softwood board cladding: 8 boards (~125 mm) per 1.0 m, along +U.

Construction logic:
  * every board is its own piece of a log: ring number = distance from that log's pith axis (below /
    beside the board, drifting along it) / that tree's ring width -> flat-sawn boards show wide
    cathedral arches, boards cut off to the side show tight straight rift grain. Nothing continues
    across a seam. Latewood is a thin hard line with a sharp outer edge and stands proud (the soft
    earlywood has weathered back).
  * boards were machined with a small chamfer on each edge, so a seam is a V-groove. They have since
    shrunk, bowed and cupped: the gap at the bottom of the V opens and closes along the length, the
    seam line wanders a millimetre, the arrises are rubbed pale and chipped out in places, most
    boards are cupped edges-forward (one or two the other way), board ends have twisted.
  * the cladding is face-nailed to studs at 400 mm centres: two rose-head nails per board per stud,
    sunk a little, rusted, half of them bleeding a grey iron stain along the grain; a butt joint
    always lands on a stud, never on the same stud as a joint in the nearby rows.
  * knots are what the tree gave: sound red knots, black checked knots, a dropped knot hole, spike
    knots running in from a board edge.
Grey-brown aged pine, tintable (variation is mostly lighter / darker / greyer).
"""
from tk.tex import *

META = {"normal_strength": 1.5}

N_BOARDS = 8
N_STUDS = 4
WIDTHS = (0.125, 0.117, 0.133, 0.121, 0.130, 0.114, 0.136, 0.124)          # metres, sum = 1.0
JOINT_STUD = {0: 3, 2: 0, 3: 1, 5: 2, 7: 1}                                 # row -> stud of its butt joint
# knots: (row, x fraction along the board, v fraction of the half width, radius m, type)
#   type 0 = sound red knot, 1 = black checked knot, 2 = dropped knot hole, 3 = spike knot from the edge
KNOTS = ((0, 0.31, 0.25, 0.012, 0), (1, 0.66, -0.45, 0.0055, 1), (1, 0.17, 1.0, 0.0090, 3),
         (3, 0.52, 0.10, 0.0075, 2), (4, 0.80, -0.20, 0.017, 0), (4, 0.36, 0.55, 0.0045, 1),
         (5, 0.24, -0.35, 0.0085, 1), (6, 0.60, -1.0, 0.0100, 3), (6, 0.09, 0.40, 0.0050, 0),
         (7, 0.72, 0.30, 0.0095, 0))


def _hash(i, k=0.0):
    x = np.sin(i * 12.9898 + k * 78.233) * 43758.5453
    return (x - np.floor(x)).astype(np.float32)


def _wrapd(a, L):
    return (a + L / 2) % L - L / 2


def build(t):
    X, Y, W, H = t.X, t.Y, t.W, t.H
    NB = N_BOARDS
    rng = np.random.default_rng(4243)
    sp = W / N_STUDS
    stud0 = sp * 0.5
    col = np.broadcast_to(np.arange(t.w)[None, :], X.shape)
    x1 = X[0]

    def prof(n, kmax, power=0.8):
        """n periodic 1-D noise profiles along the boards, each scaled to -1..1."""
        out = np.zeros((n, t.w), dtype=np.float32)
        for i in range(n):
            for k in range(1, kmax + 1):
                out[i] += rng.standard_normal() / k ** power * np.sin(2 * np.pi * (k * x1 / W + rng.random()))
            out[i] /= np.abs(out[i]).max() + 1e-9
        return out

    # ------------------------------------------------------------------ board layout: bowed seam lines
    c = np.concatenate([[0.0], np.cumsum(WIDTHS)]) * (H / float(sum(WIDTHS)))
    lines = c[:NB, None] + prof(NB, 3) * (0.0009 + 0.0008 * rng.random(NB))[:, None]      # seam i = lower edge of board i
    row = -1 + sum((Y >= lines[i][None, :]).astype(np.int64) for i in range(NB))
    Yu = np.where(row < 0, Y + H, Y)
    row = np.where(row < 0, NB - 1, row)
    lo = lines[row, col]
    hi = np.where(row == NB - 1, lines[0, col] + H, lines[(row + 1) % NB, col])
    bw = (hi - lo).astype(np.float32)                                # local board width
    vloc = (Yu - (lo + hi) / 2).astype(np.float32)                    # metres across the board, 0 = centre
    de = (bw / 2 - np.abs(vloc)).astype(np.float32)                   # metres to the board edge
    vn = vloc / (bw / 2)
    sidx = np.where(vloc < 0, row, (row + 1) % NB)                    # nearest seam

    def per(vals):
        return np.asarray(vals, dtype=np.float32)[row]

    def spread(lo_, hi_, jit=0.5):
        """One value per board, evenly spread over lo..hi (shuffled) so the boards really differ."""
        q = (rng.permutation(NB) + 0.5 + (rng.random(NB) - 0.5) * jit) / NB
        return lo_ + (hi_ - lo_) * q

    jointed = np.array([i in JOINT_STUD for i in range(NB)])
    xj = np.array([stud0 + JOINT_STUD.get(i, 0) * sp for i in range(NB)]) + (rng.random(NB) - 0.5) * 0.016
    xl = np.mod(X - xj[row], W).astype(np.float32)                    # metres along the board from its end
    db = np.where(jointed[row], np.minimum(xl, W - xl), 1.0)          # metres to the butt joint
    jsign = np.where(_wrapd(X - xj[row], W) >= 0, 1.0, -1.0)

    # ------------------------------------------------------------------ per-board character
    rift = np.array([0, 1, 0, 0, 1, 0, 1, 0], dtype=bool)             # cut off to the side of the log: straight tight grain
    v0a = np.where(rift, (0.11 + rng.random(NB) * 0.08) * np.where(rng.random(NB) < 0.5, -1, 1), (rng.random(NB) - 0.5) * 0.14)
    d0a = np.where(rift, 0.012 + rng.random(NB) * 0.03, 0.016 + rng.random(NB) * 0.055)
    rwa = np.where(rift, 0.0062 + rng.random(NB) * 0.0016, spread(0.0060, 0.0100))
    grey_a = np.array([0.22, 0.95, 0.10, 0.32, 0.80, 0.15, 0.95, 0.28]) * (0.85 + 0.15 * rng.random(NB))
    tone_a = spread(0.0, 1.0)
    cup_a = (0.6 + 0.5 * rng.random(NB)) * np.array([1, 1, -1, 1, 1, 1, -0.8, 1])
    v0, d0, rw = per(v0a), per(d0a), per(rwa)
    sl = per((rng.random(NB) - 0.5) * 0.05 * jointed)                 # taper: pith depth drifts along the board
    offx, offy = per(rng.random(NB) * W), per(rng.random(NB) * H)
    b_tone, b_grey, b_cup = per(tone_a), per(grey_a), per(cup_a)
    b_val = per(spread(0.88, 1.10))
    b_lvl = per(spread(-1.0, 1.0))
    b_tilt = per(rng.random(NB) - 0.5)
    b_rgh = per(spread(-0.08, 0.08))
    b_tw = per((rng.random(NB) - 0.5) * 2)
    b_saw = per(np.array([0, 0, 1, 0, 0, 1, 0, 1], dtype=np.float32) * (0.6 + 0.4 * rng.random(NB)))   # boards still showing saw marks
    b_ph = per(rng.random(NB) * 7)

    # ------------------------------------------------------------------ knots
    DV = t.zeros()
    krad = t.zeros() + 9.0
    kang = t.zeros()
    khalo = t.zeros()
    ktype = np.zeros(X.shape, dtype=np.int64)
    for (ki, kx, kv, r, kt) in KNOTS:
        m = (row == ki)
        dx = _wrapd(xl - kx * W, W)
        dy = vloc - kv * bw / 2
        ax, ay = (0.75, 3.8) if kt == 3 else (1.15, 1.0)
        fx, fy = (2.5, 6.0) if kt == 3 else (8.0, 3.0)
        g = np.exp(-((dx / (r * fx)) ** 2 + (dy / (r * fy)) ** 2)) * m
        DV += dy * (0.5 if kt == 3 else 0.94) * g
        rr = np.where(m, np.sqrt((dx / ax) ** 2 + (dy / ay) ** 2) / r, 9.0)
        near = rr < krad
        kang = np.where(near, np.arctan2(dy, dx), kang)
        ktype = np.where(near, kt, ktype)
        krad = np.minimum(krad, rr)
        khalo = np.maximum(khalo, np.exp(-((dx / (r * fx * 0.5)) ** 2 + (dy / (r * fy * 0.6)) ** 2)) * m)
    kr = krad + (t.noise(0.007, seed=40) - 0.5) * 0.40
    kmask = 1 - smoothstep(0.85, 1.0, kr)
    k_red, k_blk, k_hole, k_spk = [(ktype == i).astype(np.float32) for i in range(4)]
    vs = vloc - DV
    sx = xl + offx                                                    # per-board sample coordinates
    sy = Y - DV + offy

    # ------------------------------------------------------------------ ring phase per board
    wobn = t.noise(0.18, seed=1, stretch=(5.0, 1.0))
    wob2 = t.noise(0.05, seed=2, stretch=(6.0, 1.0))
    jit = t.noise(0.016, seed=3, stretch=(8.0, 1.0))
    d = d0 + sl * (xl - W / 2) + (t.sample(wobn, sx, sy) - 0.5) * 0.040
    d = np.maximum(d, 0.006)
    R = np.sqrt((vs - v0) ** 2 + d ** 2)
    phi = R / rw + (t.sample(wob2, sx, sy) - 0.5) * 1.4 + (t.sample(jit, sx, sy) - 0.5) * 0.3 + b_ph
    dens = np.sqrt(((vs - v0) / (R * rw)) ** 2 + (d * (np.abs(sl) + 0.03) / (R * rw)) ** 2) * (1 - 0.85 * khalo) + 4.0
    foot = dens * t.px                                                # rings per pixel
    aa = 1 - smoothstep(0.22, 0.40, foot)
    rid = np.floor(phi)
    ry = _hash(rid + row * 31.0, 1.0)
    # latewood: a thin dense line ending in a hard edge at the ring boundary (box-filtered over the pixel)
    late = t.zeros()
    crest = t.zeros()
    for k in range(5):
        f = np.mod(phi + (k - 2) / 5.0 * foot, 1.0).astype(np.float32)
        late += smoothstep(0.72 - 0.08 * ry, 0.92, f) * (1 - smoothstep(0.975, 1.0, f))
        crest += (smoothstep(0.45, 0.93, f) ** 1.6) * (1 - smoothstep(0.975, 1.0, f))
    late = late / 5.0 * aa + 0.16 * (1 - aa)
    crest = crest / 5.0 * aa + 0.25 * (1 - aa)

    # ------------------------------------------------------------------ fibres, saw marks, drying checks
    fib = t.sample(t.fibres(0.0042, 30.0, seed=4), sx, sy)
    fib2 = t.sample(t.fbm(0.014, 3, 0.55, seed=5, stretch=(16.0, 1.0)), sx, sy)
    streak = t.sample(t.fbm(0.045, 3, 0.5, seed=6, stretch=(10.0, 1.0)), sx, sy)
    resin = smoothstep(0.80, 0.94, t.sample(t.noise(0.005, seed=7, stretch=(7.0, 1.0)), sx, sy)) * (1 - late)  # resin canals
    # pit-saw marks: slightly raked strokes across the board, only on a few boards and only in patches
    sawn = t.sample(t.noise(0.011, seed=8, stretch=(1.0, 30.0), bandwidth=0.45), sx + 0.22 * sy, sy)
    saw = (sawn - 0.5) * 2 * smoothstep(0.45, 0.75, t.sample(t.fbm(0.28, 2, 0.5, seed=9, stretch=(2.0, 1.0)), sx, sy)) * b_saw

    cn = t.noise(0.016, seed=20, stretch=(40.0, 1.0))
    _, cgy = t.slope(cn)
    cgy = np.abs(t.sample(cgy, sx, sy))
    cn = t.sample(cn, sx, sy)
    valid = smoothstep(0.12 / 0.016, 0.40 / 0.016, cgy)
    gate = smoothstep(0.66 - 0.10 * b_grey, 0.88, t.sample(t.noise(0.05, seed=21, stretch=(5.0, 1.0)), sx, sy))
    wd = 0.0017 * gate * valid
    check = sat(1 - smoothstep(wd * 0.3, wd + 1e-6, np.abs(cn - 0.5) / np.maximum(cgy, 0.12 / 0.016))) * (gate > 0.01)
    check = check * (1 - kmask) * smoothstep(0.004, 0.010, de)

    # ------------------------------------------------------------------ nails on the stud lines
    si = np.floor(X / sp).astype(np.int64) % N_STUDS
    side = np.where(vloc >= 0, 1.0, -1.0).astype(np.float32)
    at_butt = (np.array([JOINT_STUD.get(i, -1) for i in range(NB)])[row] == si)
    nid = row * 17.0 + si * 5.0 + (side > 0) * 2.0 + ((jsign > 0) & at_butt) * 0.5
    dxs = _wrapd(X - (stud0 + si * sp), W)
    nx = np.where(at_butt, _wrapd(X - xj[row], W) - jsign * (0.016 + 0.008 * _hash(nid, 9.0)),
                  dxs - (_hash(nid, 3.0) - 0.5) * 0.018)
    nv = vloc - side * (bw / 2 - 0.027 + (_hash(nid, 4.0) - 0.5) * 0.012)
    nr = np.sqrt(nx * nx + nv * nv)
    hr = 0.0035 + 0.0010 * _hash(nid, 8.0)                            # rose-head radius
    head = 1 - smoothstep(hr - 0.0010, hr + 0.0004, nr)
    proud = (_hash(nid, 7.0) < 0.045).astype(np.float32)              # the odd nail never driven home / backed out
    dimple = np.exp(-(nr / (hr * 1.7)) ** 2)
    hdome = np.sqrt(sat(1 - (nr / hr) ** 2))
    skew = (nx * np.cos(_hash(nid, 11.0) * 6.283) + nv * np.sin(_hash(nid, 11.0) * 6.283)) / hr * head
    stained = (_hash(nid, 5.0) < 0.55).astype(np.float32)
    slen = 0.010 + 0.014 * _hash(nid, 12.0)                           # e-fold: visible streak 20-50 mm along the grain
    nstain = sat(1.5 * np.exp(-np.abs(nx) / slen)) * np.exp(-(nv / (hr * 1.1)) ** 2) * (0.5 + 0.2 * _hash(nid, 13.0)) * stained
    nstain = nstain + 0.30 * np.exp(-(nr / (hr * 1.6)) ** 2)          # every head has a small dark halo
    nstain = sat(nstain * (0.7 + 0.6 * fib)) * (1 - head)
    # split from an end nail to the board end
    has_split = at_butt & (_hash(nid, 6.0) < 0.40)
    between = (np.abs(nx) < 0.026) & (nx * jsign < 0)                 # between the nail and the joint
    nsplit = (1 - smoothstep(0.0005, 0.0015, np.abs(nv + (fib2 - 0.5) * 0.0025))) * between * has_split * (1 - head)

    # ------------------------------------------------------------------ seams: V-joint, gap, damaged arrises
    chw = (0.0038 + 0.0018 * _hash(row * 3.0 + (vloc > 0), 21.0))     # chamfer width of this edge
    chipn = t.noise(0.008, seed=30, stretch=(3.5, 1.0)) + 0.10 * (t.noise(0.004, seed=31) - 0.5)
    chip = smoothstep(0.705, 0.76, chipn) * (1 - smoothstep(0.012, 0.016, de))     # ~10 % of the edge length
    chw_c = chw + chip * (0.004 + 0.006 * t.noise(0.02, seed=32))
    cham = 1 - sat(de / chw_c)
    cham = cham ** (1.0 - 0.3 * chip)                                 # broken-out chips are hollow, not planar
    gprof = prof(NB, 6)[sidx, col]
    gapw = (0.0008 + 0.0013 * np.asarray(rng.random(NB), dtype=np.float32)[sidx]) * np.clip(1.25 + 1.1 * gprof, 0.0, 2.0)
    gapw = gapw * (gprof > -0.72)                                     # squeezed shut in places
    gap = (1 - smoothstep(gapw * 0.5 - 0.0004, gapw * 0.5 + 0.0009, de)) * (gapw > 0.0002)
    bgw = 0.0008 + 0.0014 * per(rng.random(NB))
    butt_gap = 1 - smoothstep(bgw * 0.5, bgw * 0.5 + 0.0012, db)
    butt_ease = 1 - smoothstep(0.0, 0.0035, db)
    seam = sat(gap + butt_gap)

    # knot bodies
    krings = 0.5 + 0.5 * np.sin(kr * 5.0 * 2 * np.pi + t.noise(0.012, seed=41) * 6)
    kstar = smoothstep(0.84, 0.97, np.abs(np.sin(kang * 1.5 + 0.4 + (t.noise(0.02, seed=42) - 0.5) * 3)))
    kstar = kstar * smoothstep(0.05, 0.25, kr) * (1 - smoothstep(0.7, 0.95, kr)) * kmask * k_blk
    krim = np.exp(-((kr - 1.0) / 0.12) ** 2)
    hole = (1 - smoothstep(0.62, 0.92, kr)) * k_hole

    # ------------------------------------------------------------------ height (metres)
    cup = b_cup * 0.0022 * (vn * vn - 0.33)                           # cupped across the width
    twist = b_tw * 0.0011 * np.exp(-db / 0.14) * vn * jsign           # board ends have wound: one corner proud
    bow = (t.sample(t.fbm(0.35, 2, 0.5, seed=10, stretch=(3.0, 1.0)), sx, sy) - 0.5) * 0.0010
    board = cup + b_lvl * 0.0010 + b_tilt * 0.0016 * vn + twist + bow
    raise_ = 1.0 + 0.5 * b_grey                                       # weathered boards: grain stands prouder
    grain = (crest - 0.25) * 0.00045 * raise_ + (fib - 0.5) * 0.00035 + (fib2 - 0.5) * 0.00022 + saw * 0.00014 - resin * 0.00015
    height = board + grain
    height = height - check * 0.0013 - nsplit * 0.0013
    height = height + kmask * (1 - k_hole) * (0.00030 + 0.00022 * (krings - 0.5)) - kstar * 0.0009 - krim * (1 - kmask) * 0.0003
    height = height - hole * 0.0021 + krim * k_hole * 0.00015
    height = height - dimple * 0.00055 * (1 - proud) + head * (hdome * 0.00030 + proud * (0.0009 + 0.0005 * skew))
    height = height - cham * (0.0027 + 0.0006 * chip) - butt_ease * 0.0010
    height = height + chip * (1 - cham) * (t.noise(0.003, seed=33) - 0.6) * 0.0006
    height = np.where(seam > 0, mix(height, -0.0065, seam), height)

    # ------------------------------------------------------------------ colour
    tone = t.fbm(0.45, 3, 0.5, seed=11)
    v = 0.5 + 0.82 * (b_tone - 0.5) + 0.24 * (streak - 0.5) + 0.14 * (tone - 0.5) + 0.22 * (fib - 0.5) + 0.16 * (fib2 - 0.5)
    albedo = ramp(sat(v), [(0.0, "#6b5440"), (0.35, "#80664c"), (0.65, "#94795a"), (1.0, "#a88d6b")])
    albedo = albedo * b_val[..., None]
    greyz = sat(b_grey * (0.55 + 0.9 * t.sample(t.fbm(0.25, 3, 0.5, seed=15, stretch=(4.0, 1.0)), sx, sy)))
    albedo = mix(albedo, desaturate(albedo, 0.55) * hexlin("#bab5ac") * 2.05, greyz)                 # silvered, weathered boards
    albedo = mix(albedo, albedo * hexlin("#a58866") * 1.62, late * (0.62 + 0.2 * ry))                # latewood: darker, redder
    albedo = mix(albedo, albedo * 0.80, resin * 0.5)
    albedo = albedo * (1.0 + 0.09 * saw)[..., None]
    kcol = mix(ramp(krings, [(0, "#5c4030"), (1, "#7b563c")]), ramp(krings, [(0, "#3f2e23"), (1, "#5a4230")]), k_blk[..., None])
    kcol = mix(kcol, desaturate(kcol, 0.45) * 0.95, greyz)
    albedo = mix(albedo, albedo * 0.80, khalo * (1 - kmask) * 0.5)
    albedo = mix(albedo, kcol, kmask * 0.92)
    albedo = mix(albedo, hexlin("#33271e"), sat(krim * (1 - kmask) * (0.25 + 0.4 * k_blk + 0.3 * k_hole) + kstar))
    albedo = mix(albedo, hexlin("#2c231c"), hole * 0.92)
    arris = smoothstep(0.4, 0.95, de / chw) * (1 - smoothstep(1.0, 2.2, de / chw)) * (1 - chip)      # rubbed chamfer edge
    albedo = mix(albedo, albedo * 1.20 + 0.012, arris * 0.55)
    albedo = mix(albedo, desaturate(albedo, 0.12) * 0.86, chip * smoothstep(0.0, 0.45, cham) * 0.8)              # torn fibre in the chips
    stain_c = desaturate(albedo, 0.6) * hexlin("#8d8780") * 1.45
    albedo = mix(albedo, stain_c, sat(nstain * 1.3))                                               # grey iron stain along the grain
    edge_dirt = (1 - smoothstep(0.0, 1.0, de / chw_c)) * 0.50 + (1 - smoothstep(0.0, 0.010, db)) * 0.30
    edge_dirt = sat(edge_dirt * (0.6 + 0.8 * t.fbm(0.06, 3, 0.5, seed=12)))
    albedo = mix(albedo, albedo * hexlin("#7a6a58") * 1.3 * 0.58, edge_dirt)
    cav = t.cavity(height, 0.003, 0.0004)
    albedo = mix(albedo, desaturate(albedo, 0.4) * 0.78, cav * 0.3 * (1 - seam))
    albedo = mix(albedo, hexlin("#30261e"), sat(check * 0.75 + nsplit * 0.85))
    rust = ramp(t.noise(0.004, seed=13), [(0.0, "#3c2a1f"), (0.5, "#4a3324"), (1.0, "#5d3f2a")])
    albedo = mix(albedo, rust, head)
    albedo = mix(albedo, hexlin("#2e251e"), seam)

    # ------------------------------------------------------------------ roughness
    polish = t.sample(t.fbm(0.20, 4, 0.5, seed=14, stretch=(5.0, 1.0)), sx, sy)
    rough = 0.78 + b_rgh + 0.06 * greyz - 0.10 * smoothstep(0.40, 0.85, polish) * (1 - 0.6 * greyz)
    rough = rough + 0.08 * (fib - 0.5) - 0.10 * late + 0.05 * np.abs(saw)
    rough = rough + 0.10 * cav + 0.12 * edge_dirt - 0.12 * kmask * (1 - k_hole) - 0.07 * arris + 0.12 * chip * (1 - cham)
    rough = rough + 0.14 * sat(check + nsplit) + 0.06 * sat(nstain)
    rough = mix(rough, 0.80, head)
    rough = mix(rough, 0.95, sat(seam + hole))
    rough = np.clip(rough, 0.42, 0.97)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
