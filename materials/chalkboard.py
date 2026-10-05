"""Honed black slate chalkboard: wiped chalk-dust swaths, dust ridges at the eraser edges, ghosted menu writing."""
from tk.tex import *

META = {"normal_strength": 2.0}

# ---------------------------------------------------------------------------- chalk hand lettering
# Single-stroke capitals on a 4 x 6 grid (polylines), the way a cafe board is actually lettered.
_P = [(0, 0), (0, 6), (3, 6), (4, 5), (4, 3.8), (3, 2.8), (0, 2.8)]
_O = [(1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)]
_SIX = [(3.6, 6), (1.5, 5), (0, 2.5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.4), (3, 3.3), (1, 3.3), (0, 2.2)]
GLYPHS = {
    "A": [[(0, 0), (2, 6), (4, 0)], [(1, 2.2), (3, 2.2)]],
    "B": [[(0, 0), (0, 6), (2.5, 6), (3.5, 5.2), (3.5, 3.8), (2.5, 3), (0, 3)], [(2.5, 3), (4, 2.2), (4, 0.8), (3, 0), (0, 0)]],
    "C": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)]],
    "D": [[(0, 0), (0, 6), (2.5, 6), (4, 4.5), (4, 1.5), (2.5, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3), (3, 3)]],
    "G": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.8), (2.2, 2.8)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "I": [[(2, 0), (2, 6)]],
    "J": [[(3.5, 6), (3.5, 1), (2.5, 0), (1, 0), (0, 1)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 2.5), (4, 6), (4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [_O],
    "P": [_P],
    "R": [_P, [(2, 2.8), (4, 0)]],
    "S": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3.1), (3, 2.9), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "0": [_O],
    "1": [[(1, 4.8), (2.2, 6), (2.2, 0)]],
    "2": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 3.8), (0, 0), (4, 0)]],
    "3": [[(0, 5.2), (1, 6), (3, 6), (4, 5), (4, 3.8), (3, 3), (1.5, 3)], [(3, 3), (4, 2.2), (4, 1), (3, 0), (1, 0), (0, 0.8)]],
    "4": [[(3, 0), (3, 6), (0, 2), (4, 2)]],
    "5": [[(4, 6), (0.3, 6), (0, 3.3), (2.8, 3.6), (4, 2.6), (4, 1), (3, 0), (1, 0), (0, 0.8)]],
    "6": [_SIX],
    "7": [[(0, 6), (4, 6), (1.5, 0)]],
    "8": [[(1, 3.1), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3.1), (1, 3.1), (0, 2.1), (0, 1), (1, 0), (3, 0),
           (4, 1), (4, 2.1), (3, 3.1)]],
    "9": [[(4 - x, 6 - y) for x, y in _SIX]],
    ",": [[(2.1, 0.5), (1.6, -0.9)]],
    "-": [[(0.6, 3), (3.4, 3)]],
    ".": [[(1.9, 0.0), (2.1, 0.3)]],
}
WORDS = ["CAFE", "THE", "VIN ROUGE", "SOUPE DU JOUR", "TARTE", "PAIN", "BIERE", "PLAT DU JOUR", "FROMAGE", "SALADE",
         "OMELETTE", "JAMBON", "CHOCOLAT", "POTAGE", "GALETTE", "CIDRE", "MOULES", "DESSERT", "ROTI", "LAPIN"]
PRICES = ["3,50", "2,-", "4,80", "12,-", "1,90", "6,50", "9,-", "7,20", "5,-", "8,40"]


def _chaikin(pts):
    """One corner-cutting pass: a hand does not draw sharp polygon corners."""
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(len(pts) - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        out.append((0.8 * x0 + 0.2 * x1, 0.8 * y0 + 0.2 * y1))
        out.append((0.2 * x0 + 0.8 * x1, 0.2 * y0 + 0.8 * y1))
    out.append(pts[-1])
    return out


def _lettering(t, seed, n_lines, cap, stroke_sigma):
    """Rasterise lines of hand-lettered menu text that wrap round the tile.
    Returns (stroke mask 0..1, per-word strength image 0..1)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((t.h, t.w), dtype=np.float64)
    pitch = t.H / n_lines
    step = min(t.px, t.py) * 0.5
    for li in range(n_lines):
        base_y = (li + 0.28 + 0.12 * rng.random()) * pitch
        unit = cap / 6.0 * (0.85 + 0.3 * rng.random())                     # each line lettered a bit differently
        drift_a = (rng.random() - 0.5) * 0.012                              # baseline wanders
        drift_p = rng.random() * 6.28
        slant = 0.08 + 0.12 * rng.random()
        x = rng.random() * t.W
        x_end = x + t.W - 0.03                                               # one wrap, no overlap
        first = True
        while True:
            is_price = (not first) and rng.random() < 0.4
            word = PRICES[rng.integers(len(PRICES))] if is_price else WORDS[rng.integers(len(WORDS))]
            wlen = sum((2.6 if ch == " " else (2.2 if ch in "I1,." else 5.3)) for ch in word) * unit * 0.8
            if x + wlen > x_end:
                break
            strength = 0.25 + 0.75 * rng.random() ** 1.5                     # how much survived the wiping
            for ch in word:
                if ch == " ":
                    x += 2.6 * unit * 0.8
                    continue
                gw = 2.2 if ch in "I1,." else 5.3
                gs = unit * (0.93 + 0.14 * rng.random())
                gy = base_y + (rng.random() - 0.5) * 0.003 + drift_a * np.sin(drift_p + x / t.W * 6.283)
                rot = (rng.random() - 0.5) * 0.10
                xoff = -1.2 if ch in "I1,." else 0.0
                for stroke in GLYPHS[ch]:
                    pts = [(px_ + xoff + (rng.random() - 0.5) * 0.35, py_ + (rng.random() - 0.5) * 0.35) for px_, py_ in stroke]
                    pts = _chaikin(pts)
                    press = 0.7 + 0.3 * rng.random()
                    for i in range(len(pts) - 1):
                        (ax, ay), (bx, by) = pts[i], pts[i + 1]
                        seg = np.hypot(bx - ax, by - ay) * gs
                        n = max(2, int(seg / step))
                        s = np.linspace(0, 1, n, endpoint=False)
                        ux = (ax + (bx - ax) * s) * 0.8
                        uy = ay + (by - ay) * s
                        ux = ux + slant * uy
                        wx = x + (ux * np.cos(rot) - uy * np.sin(rot)) * gs
                        wy = gy + (ux * np.sin(rot) + uy * np.cos(rot)) * gs
                        cols = np.floor(wx / t.px).astype(np.int64) % t.w
                        rows = np.floor((t.H - wy) / t.py).astype(np.int64) % t.h
                        np.add.at(acc, (rows, cols), press * strength * seg / n)
                x += gw * unit * 0.8
            x += (0.035 + 0.09 * rng.random()) if not is_price else (0.06 + 0.1 * rng.random())
            first = False
    m = t.blur(acc.astype(np.float32), stroke_sigma)
    peak = np.percentile(m, 99.7) + 1e-12
    return sat(m / (0.55 * peak))


def _swaths(t, n, seed, axis):
    """Eraser swaths: n wavy bands along `axis`. Every band is a separate stroke, so the dust it
    left varies along its length independently of its neighbours, and a thin ridge of pushed dust
    marks one edge. Returns (residue 0..1, edge line 0..1, streak 0..1)."""
    rng = np.random.default_rng(1000 + seed)
    offs = (rng.random(n) * (t.W if axis == 'x' else t.H)).astype(np.float32)
    bias = (rng.random(n).astype(np.float32) - 0.5) * 0.35
    wob = (t.fbm(0.4, 2, 0.5, seed=seed) - 0.5) * 0.11 + (t.fbm(0.11, 2, 0.5, seed=seed + 1) - 0.5) * 0.02
    if axis == 'x':                                                          # bands run along X
        ph = n * (t.Y + wob) / t.H
        along = t.fbm(0.22, 3, 0.5, seed=seed + 4, stretch=(2.0, 1.0))
        soft = t.fbm(0.03, 3, 0.5, seed=seed + 2, stretch=(8.0, 1.0))
        fine = t.fbm(0.0035, 2, 0.6, seed=seed + 5, stretch=(22.0, 1.0))
        hair = t.fbm(0.0016, 2, 0.6, seed=seed + 6, stretch=(28.0, 1.0))
        streak = t.sample(0.62 * soft + 0.25 * fine + 0.13 * hair, t.X, t.Y + wob)
        gate = t.fbm(0.12, 2, 0.5, seed=seed + 3, stretch=(3.0, 1.0))
    else:
        ph = n * (t.X + wob) / t.W
        along = t.fbm(0.22, 3, 0.5, seed=seed + 4, stretch=(1.0, 2.0))
        soft = t.fbm(0.03, 3, 0.5, seed=seed + 2, stretch=(1.0, 8.0))
        fine = t.fbm(0.0035, 2, 0.6, seed=seed + 5, stretch=(1.0, 22.0))
        hair = t.fbm(0.0016, 2, 0.6, seed=seed + 6, stretch=(1.0, 28.0))
        streak = t.sample(0.62 * soft + 0.25 * fine + 0.13 * hair, t.X + wob, t.Y)
        gate = t.fbm(0.12, 2, 0.5, seed=seed + 3, stretch=(1.0, 3.0))
    k = np.floor(ph).astype(np.int64)
    f = (ph - k).astype(np.float32)

    def band(kk):
        o = offs[kk % n]
        v = t.sample(along, t.X + o, t.Y) if axis == 'x' else t.sample(along, t.X, t.Y + o)
        return sat(v + bias[kk % n])

    res = mix(band(k), band(k + 1), smoothstep(0.72, 1.0, f))
    width = (t.H if axis == 'x' else t.W) / n
    edge = np.exp(-(1 - f) * width / 0.0035) + 0.5 * np.exp(-f * width / 0.006)
    edge = edge * smoothstep(0.45, 0.7, gate)
    return res.astype(np.float32), sat(edge).astype(np.float32), level(streak)


def _strokes(t, n, seed):
    """The last few passes of a felt eraser: n curved bands 6-11 cm wide that took most of the
    dust off, with a thin line of pushed dust along each side and a heap where the stroke stopped.
    Returns (clean 0..1, dust line 0..1). Wraps (each stroke is shorter than half a tile)."""
    rng = np.random.default_rng(4000 + seed)
    clean = t.zeros()
    dust = t.zeros()
    g = int(np.ceil(np.sqrt(n)))
    cells = [(i, j) for i in range(g) for j in range(g)]
    rng.shuffle(cells)
    for k in range(n):
        i, j = cells[k]
        cx = (i + rng.random()) / g * t.W
        cy = (j + rng.random()) / g * t.H
        ang = np.radians(rng.normal(0, 28)) + (np.pi / 2 if rng.random() < 0.25 else 0.0)
        L = 0.22 + 0.13 * rng.random()
        wd = 0.03 + 0.025 * rng.random()
        bend = (rng.random() - 0.5) * 5.0
        strength = 0.45 + 0.4 * rng.random()
        dx = (t.X - cx + t.W / 2) % t.W - t.W / 2
        dy = (t.Y - cy + t.H / 2) % t.H - t.H / 2
        al = dx * np.cos(ang) + dy * np.sin(ang)
        ac = -dx * np.sin(ang) + dy * np.cos(ang) - bend * al * al
        ends = smoothstep(0.0, 0.012, L / 2 - al) * smoothstep(0.0, 0.07, L / 2 + al)   # sharp stop, soft start
        inside = (1 - smoothstep(wd * 0.86, wd, np.abs(ac))) * ends
        # the felt is uneven: parallel lanes of more / less dust left behind
        lanes = 0.5 + 0.2 * np.sin(ac * rng.uniform(300, 500) + rng.random() * 6.3) \
            + 0.18 * np.sin(ac * rng.uniform(800, 1300) + rng.random() * 6.3) \
            + 0.12 * np.sin(ac * rng.uniform(1800, 2600) + rng.random() * 6.3)
        clean = np.maximum(clean, inside * strength * (0.45 + 0.55 * lanes))
        side = np.exp(-((np.abs(ac) - wd) / 0.0022) ** 2) * ends * (0.5 + 0.5 * rng.random())
        heap = np.exp(-((al - L / 2) / 0.0035) ** 2) * (1 - smoothstep(wd * 0.8, wd * 1.05, np.abs(ac)))
        dust = np.maximum(dust, np.maximum(side * 0.6, heap) * strength)
    return clean.astype(np.float32), dust.astype(np.float32)


def build(t):
    # ------------------------------------------------------------------ slate (meso/micro)
    # Honed slate: nearly flat, faint cleavage undulation, a slight silky grain, pin pits and the
    # scoring left by years of chalk and grit.
    cleave = t.fbm(0.09, 4, 0.5, seed=1, stretch=(2.2, 1.0))
    grain = t.fbm(0.004, 3, 0.6, seed=2, stretch=(4.0, 1.0))
    tooth = t.fbm(0.0018, 2, 0.6, seed=3)
    pits = t.speckle(density=5000, size=0.0018, seed=4, soft=0.5)
    flake = smoothstep(0.80, 0.86, t.fbm(0.05, 4, 0.55, seed=5)) * 1.0      # shallow spalled cleavage flakes
    flake = t.blur(flake, 0.0012)
    scr = np.maximum(t.scratches(250, 0.06, 0.0007, seed=6), t.scratches(500, 0.02, 0.0007, seed=7))
    riven = t.ridged(0.035, 4, 0.5, seed=17, stretch=(3.5, 1.0))            # faint cleavage ledges left by honing
    height = (cleave - 0.5) * 0.00055 + (riven - 0.5) * 0.00007 + (grain - 0.5) * 0.00005 + (tooth - 0.5) * 0.00003 \
        - pits * 0.00012 - flake * 0.00016 - scr * 0.00006

    slate_tone = t.fbm(0.2, 4, 0.5, seed=8)
    slate = ramp(slate_tone, [(0.0, "#292b2d"), (0.5, "#2f3133"), (1.0, "#353738")])
    slate = slate * (0.92 + 0.16 * grain)[..., None]
    vein = smoothstep(0.86, 0.97, t.ridged(0.25, 3, 0.5, seed=9, stretch=(3.0, 1.0)))   # faint pale mineral bands
    slate = mix(slate, hexlin("#46484a"), vein * 0.35)

    # ------------------------------------------------------------------ chalk dust: wiped swaths
    resx, edgex, stx = _swaths(t, 9, 20, 'x')
    resy, edgey, sty = _swaths(t, 8, 30, 'y')
    # Two generations of wiping: an older up-down pass left a soft haze; the latest side-to-side
    # pass took most of it off again and left its own streaks and pushed-dust edges.
    r_new = smoothstep(0.2, 0.9, resx)
    r_old = smoothstep(0.2, 0.9, resy)
    cloud = t.fbm(0.06, 4, 0.55, seed=11)
    broad = t.fbm(0.4, 2, 0.5, seed=10)
    keep = 0.5 + 0.5 * r_new
    haze = (0.025 + 0.15 * r_old * (0.35 + 0.65 * sty)) * keep
    haze = haze + 0.12 * r_new * (0.2 + 0.8 * stx) + 0.03 * cloud * stx + 0.03 * (broad - 0.3)
    edge = np.maximum(edgex, edgey * 0.75 * keep)
    haze = haze + 0.13 * edge * (0.4 + 0.6 * t.fbm(0.01, 2, 0.6, seed=12))
    damp = smoothstep(0.52, 0.78, t.fbm(0.2, 3, 0.5, seed=13)) * (1 - 0.7 * r_new)   # gone over with a wet rag
    haze = sat(haze) * (1 - 0.6 * damp)
    wiped, pushed = _strokes(t, 6, 3)                                        # the most recent eraser strokes
    brk = 0.35 + 0.65 * t.fbm(0.008, 2, 0.6, seed=19)
    haze = haze * (1 - 0.85 * wiped) + 0.02 * wiped + 0.17 * pushed * brk
    streak = stx
    zone = t.zeros()

    # ------------------------------------------------------------------ ghosted writing
    text = _lettering(t, 5, 9, 0.040, 0.0017)
    old = t.shift(_lettering(t, 6, 6, 0.060, 0.0022), 0.11, 0.037)           # an older, larger layer
    chalk_grain = 0.45 + 0.55 * smoothstep(0.3, 0.7, t.fbm(0.0022, 2, 0.6, seed=14))   # chalk skips on the tooth
    smear_x = t.blur(text, 0.008, 0.0012)
    smear_y = t.blur(text, 0.0012, 0.008)
    smear = mix(smear_x, smear_y, zone)
    survive = smoothstep(0.41, 0.78, t.fbm(0.09, 3, 0.55, seed=15))          # patchy wiping: fragments remain
    frag = smoothstep(0.3, 0.6, t.fbm(0.02, 3, 0.55, seed=16))
    ghost = (text * chalk_grain * 0.34 * (0.25 + 0.75 * frag) + smear * 0.20) * (0.06 + 0.94 * survive) * (0.5 + 0.5 * streak)
    ghost = ghost + old * chalk_grain * 0.08 * (1 - survive * 0.5) * smoothstep(0.35, 0.65, t.fbm(0.12, 2, 0.5, seed=18))
    ghost = ghost * (1 - 0.6 * damp) * (1 - 0.75 * wiped)

    # dust also lodges in pits, scratches and spalled flakes
    lodged = sat(pits * 0.7 + scr * 0.55 + flake * 0.35) * (0.4 + 0.6 * cloud)
    cover = sat(haze + ghost + lodged * 0.3)

    chalk_col = hexlin("#b4b2aa")
    albedo = mix(slate, chalk_col, cover * 0.62)
    albedo = np.clip(albedo, 0.012, 0.8)
    height = height + cover * 0.000025 * (0.5 + chalk_grain)

    # ------------------------------------------------------------------ roughness
    # Clean honed slate has a soft sheen; chalk dust is dead matt; damp-wiped areas are the glossiest.
    rough = 0.50 + 0.10 * (grain - 0.5) + 0.05 * (slate_tone - 0.5) - 0.10 * damp
    rough = mix(rough, 0.95, sat(cover * 3.0) ** 0.8)
    rough = np.clip(rough + 0.1 * pits, 0.3, 0.98)
    return dict(albedo=albedo, height=height, rough=rough, metal=0.0)
