"""Candle wax / tallow: hand-dipped ivory body, cloudy bloom, runs of overflowed wax that set as
rounded rivulets ending in fat beads (drips run DOWN the tile, towards V = 0), thin early sheets
underneath them, pin-hole bubbles, handling nicks. Tintable: variation is value / warmth only."""
from tk.tex import *

META = {"normal_strength": 1.0, "ao_radius": 0.004, "ao_strength": 0.8}


def _runs(t, rng, n, w_rng, h_rng, len_rng, p, bead, wander=1.0, cluster=0):
    """Rasterise `n` wax runs. Each run is a ribbon following a slightly wandering path downwards
    with half-width hw(s) and crest height H(s) (s = metres from its top), closing in a bead.
    Returns (height, s_norm, crest) where crest = 1 on the centre line."""
    hmap = t.zeros()
    age = t.zeros()                       # per-run random (colour / gloss variation)
    px, py = t.px, t.py
    cols = (np.arange(n) + rng.random(n) * 0.8 + 0.1) / n * t.W
    rows = (rng.permutation(n) + rng.random(n)) / n * t.H
    if cluster:
        # overflow happens at a few low points of the rim: later runs follow (and pile on) earlier ones
        par = np.concatenate([rng.permutation(n), rng.permutation(n)])[:cluster]
        cols = np.concatenate([cols, cols[par] + rng.normal(0, 0.005, cluster)])
        rows = np.concatenate([rows, rows[par] + rng.normal(0, 0.045, cluster)])
        n = n + cluster
    for i in range(n):
        x0, ytop = cols[i], rows[i]
        w0 = rng.uniform(*w_rng) * 0.5                       # half width (m)
        H0 = rng.uniform(*h_rng)
        L = rng.uniform(*len_rng) * (0.3 + 0.7 * rng.random() ** 1.3)
        rb = w0 * (bead[0] + (bead[1] - bead[0]) * rng.random())     # bead radius
        Hb = H0 * rng.uniform(1.35, 1.75) if bead[1] > 1.05 else H0
        Lt = min(L * 0.45, rng.uniform(0.02, 0.05))                  # run-in length
        a1, l1, f1 = rng.uniform(0.0008, 0.0028) * wander, rng.uniform(0.07, 0.16), rng.random() * 6.283
        a2, l2, f2 = rng.uniform(0.0002, 0.0007) * wander, rng.uniform(0.018, 0.04), rng.random() * 6.283
        rp, rf, ra = rng.uniform(0.005, 0.011), rng.random() * 6.283, rng.uniform(0.02, 0.08)
        wv, wf = rng.uniform(0.02, 0.05), rng.random() * 6.283
        nr = int(L / py) + 2
        s = (np.arange(nr, dtype=F) + 0.5) * py                      # metres below the top
        cx = x0 + a1 * np.sin(6.283 * s / l1 + f1) + a2 * np.sin(6.283 * s / l2 + f2)
        tin = smoothstep(0.0, Lt, s)
        hw = w0 * (0.72 + 0.28 * tin) * (1.0 + 0.13 * np.sin(6.283 * s / wv + wf))
        Hs = H0 * tin * (1.0 + ra * np.sin(6.283 * s / rp + rf))        # the run melts out of the body
        sc = L - rb
        k = smoothstep(sc - 3.2 * rb, sc - 0.2 * rb, s)
        hw = mix(hw, rb, k)
        Hs = mix(Hs, Hb, k)
        endc = np.sqrt(np.clip(1.0 - ((s - sc) / rb) ** 2, 0.0, 1.0))
        tail = s > sc
        hw = np.where(tail, rb * endc, hw)
        Hs = np.where(tail, Hb * endc ** 0.85, Hs)
        hwmax = float(hw.max()) + abs(a1) + abs(a2) + 2 * px
        nc = int(2 * hwmax / px) + 3
        ix0 = int(np.floor((x0 - hwmax) / px))
        xs = (ix0 + np.arange(nc, dtype=F) + 0.5) * px
        q = np.abs(xs[None, :] - cx[:, None]) / np.maximum(hw[:, None], 1e-6)
        prof = np.clip(1.0 - q ** p, 0.0, 1.0)
        hh = (Hs[:, None] * prof).astype(F)
        r0 = int((t.H - ytop) / py)
        yy = (r0 + np.arange(nr)) % t.h
        xx = (ix0 + np.arange(nc)) % t.w
        sel = np.ix_(yy, xx)
        old = hmap[sel]
        hmap[sel] = np.maximum(old, hh) + 0.4 * np.minimum(old, hh)   # later wax rides over earlier
        age[sel] = np.where(hh > old, rng.random(), age[sel])
    return hmap, age


def build(t):
    rng = np.random.default_rng(4711)

    # ---- candle body: hand dipped, never truly flat; faint vertical dipping streaks
    body = (t.fbm(0.09, 3, 0.5, seed=1) - 0.5) * 0.00045
    body += (t.fbm(0.03, 3, 0.5, seed=8) - 0.5) * 0.00030
    body += (t.fbm(0.012, 3, 0.5, seed=2, stretch=(1.0, 0.12)) - 0.5) * 0.00012
    body += (t.fbm(0.004, 3, 0.55, seed=3) - 0.5) * 0.00003

    # ---- wax runs: thin wide early sheets, then proper rivulets with beads, then a few thin late ones
    sheet, sheet_age = _runs(t, rng, 10, (0.012, 0.026), (0.00016, 0.00032), (0.10, 0.26), 4.0, (0.85, 1.0), 1.6)
    sheet = t.warp(sheet, 0.003, 0.02, seed=40)
    drip, drip_age = _runs(t, rng, 13, (0.0036, 0.0085), (0.0011, 0.0021), (0.06, 0.21), 2.6, (1.2, 1.55), cluster=11)
    fine, fine_age = _runs(t, rng, 9, (0.0020, 0.0032), (0.0005, 0.0009), (0.03, 0.11), 2.4, (1.15, 1.5), cluster=4)
    sheet = t.blur(sheet, 0.0016)
    runs = np.maximum(drip, fine) + 0.4 * np.minimum(drip, fine)
    runs = t.blur(runs, 0.00038)
    age = np.where(drip >= fine, drip_age, fine_age)

    # ---- small defects: pin-hole bubbles, handling nicks, a few stuck wax crumbs
    pin = t.speckle(density=3800, size=0.0011, seed=5, soft=0.7) * smoothstep(0.42, 0.62, t.fbm(0.04, 3, seed=15))
    nick = t.scratches(density=420, length=0.014, width=0.00045, seed=6)
    nick = t.blur(nick, 0.00025)
    crumb = t.speckle(density=900, size=0.0016, seed=7, soft=0.8)
    crumb = t.blur(crumb, 0.0003)

    height = body + sheet + runs - pin * 0.00016 - nick * 0.00007 + crumb * 0.00022
    on_run = smoothstep(0.00008, 0.0005, runs)
    on_sheet = smoothstep(0.00004, 0.00018, sheet) * (1 - on_run)
    thick = sat(runs / 0.0022)

    # ---- masks from the relief
    foot = t.cavity(body + runs, 0.0016, 0.00022) * (1 - on_run * 0.7)   # concave foot of every run: dust sits here
    foot = t.blur(foot, 0.0004)
    crest = t.edges(height, 0.0020, 0.00035)

    # ---- colour: ivory tallow. cloudy opaque bloom vs. slightly deeper translucent wax
    cloud = t.fbm(0.07, 3, 0.45, seed=10)
    fine_c = t.fbm(0.006, 3, 0.55, seed=11)
    streak = t.fbm(0.016, 3, 0.5, seed=12, stretch=(1.0, 0.1))
    tone = sat(0.50 + (cloud - 0.5) * 0.42 + (fine_c - 0.5) * 0.14 + (streak - 0.5) * 0.22)
    albedo = ramp(tone, [(0.0, "#d0c4a2"), (0.3, "#d9ceb0"), (0.6, "#e0d7bd"), (1.0, "#e8e1cc")])
    # runs: thick wax is deeper / warmer in its body (light gets lost inside), paler along the chilled crest
    run_col = ramp(sat(0.25 + 0.5 * age + (fine_c - 0.5) * 0.3), [(0.0, "#d6caa6"), (0.5, "#ddd2b2"), (1.0, "#e5dcc2")])
    albedo = mix(albedo, run_col, on_run * 0.8)
    albedo = mix(albedo, albedo * hexlin("#f6ecd0"), thick * 0.5)
    albedo = mix(albedo, hexlin("#ece4cd"), crest * on_run * 0.22)
    albedo = mix(albedo, albedo * hexlin("#f6f0de"), on_sheet * 0.35)
    # dust / soot held in the feet of the runs and in pin holes; rare soot flecks from the wick
    albedo = mix(albedo, albedo * hexlin("#b3a488"), sat(foot * 0.6))
    albedo = mix(albedo, albedo * hexlin("#9a8c74"), pin * 0.6)
    soot = t.speckle(density=1500, size=0.0008, seed=13, soft=0.6) * smoothstep(0.45, 0.7, t.fbm(0.05, 2, seed=14))
    albedo = mix(albedo, hexlin("#4a4238"), soot * 0.7)
    albedo = mix(albedo, albedo * 1.05, nick * 0.5)                    # scuffs bruise wax pale

    # ---- roughness: fresh runs are glossy, bloom and handling are matte, dust is rough
    bloom = smoothstep(0.3, 0.95, cloud + (fine_c - 0.5) * 0.15)
    rough = 0.33 + 0.20 * bloom + (fine_c - 0.5) * 0.05 + (streak - 0.5) * 0.06
    rough = mix(rough, 0.24 + 0.10 * age, on_run)
    rough = mix(rough, 0.33, on_sheet * 0.6)
    rough = rough + 0.28 * sat(foot) + 0.2 * nick + 0.25 * pin + 0.3 * soot
    rough = np.clip(rough, 0.16, 0.9)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
