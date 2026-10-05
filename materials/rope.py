"""Three-strand hawser-laid hemp rope, ~9.5 mm diameter. U runs ALONG the rope, V around it.

Tile 60 x 30 mm: V is one circumference (3 strands), U holds 5 strand crossings (lay 36 mm, ~40 deg).
Construction, the way a ropewalk does it: fibres are spun right-handed into yarns, ten or so visible
yarns are twisted left-handed into each strand, three strands are laid right-handed into the rope.
The opposing twists are why the yarns on the surface run almost parallel to the rope axis while the
strands spiral across them. Everything is analytic in (strand phase, yarn phase, fibre phase) with
integer winding numbers so the helices close across both tile edges."""
from tk.tex import *

META = {"normal_strength": 1.0}

NSTR, NLAY = 3, 5        # strands round V, strand crossings along U
NY = 10                  # yarns visible across one strand
AY = 40                  # yarn winding number along U  (yarns ~parallel to the rope axis)
KOFF = 20                # yarn-index offset per strand (so the three strands differ)
PY = 30                  # period of every per-yarn function   (AY - 5*KOFF and 3*KOFF are 0 mod PY)
FQ, MQ, PQ = 4, -60, 60  # fibre phase q = FQ*phase + MQ*U/W, period PQ


def _rnd(n, k):
    return np.random.default_rng(1307 + 7919 * k).random(n).astype(F)


def build(t):
    pi = math.pi
    U, V = t.X / t.W, t.Y / t.H

    # --- strands: helical phase, slightly uneven lay (hand-laid rope is never a perfect screw)
    lay = (t.noise(0.020, seed=1) - 0.5) * 0.16 + (t.noise(0.007, seed=2) - 0.5) * 0.05
    ps = NSTR * V - NLAY * U + lay
    k = np.floor(ps)
    s = ps - k - 0.5                                             # -0.5..0.5 across a strand
    sx = np.clip(s * 1.60, -0.9999, 0.9999)                      # visible arc of a strand = +-53 deg
    cth = np.sqrt(1 - sx * sx)                                   # cos of the angle round the strand
    r_str = 0.0043 * (0.94 + 0.12 * t.noise(0.016, seed=3))
    strand = r_str * (cth - 1.0)                                 # 0 on the crown .. -2.1 mm in the groove
    groove = smoothstep(0.34, 0.50, np.abs(s))

    # --- yarns: helices round the strand, opposite hand to the lay
    gs = np.arcsin(sx) / (2 * math.asin(0.80))                        # -0.5..0.5, yarns foreshorten at the edges
    ywob = (t.noise(0.0045, seed=4, stretch=(5, 1)) - 0.5) * 0.50 + (t.noise(0.0018, seed=5, stretch=(5, 1)) - 0.5) * 0.08
    phase = NY * gs + AY * U + KOFF * k + ywob
    iy = np.floor(phase)
    fy = phase - iy - 0.5
    yid = iy.astype(np.int64) % PY
    y_proud = _rnd(PY, 1)[yid]
    y_tone = _rnd(PY, 2)[yid]
    y_hue = _rnd(PY, 3)[yid]
    y_fat = _rnd(PY, 4)[yid]
    ysp = t.H / PY                                               # yarn space: 1 yarn = 1 mm
    ymid0 = 1 - smoothstep(0.30, 0.5, np.abs(fy))
    y_tone = 0.5 + (y_tone - 0.5) * ymid0                        # yarn identity fades into the shared seam
    y_hue = 0.5 + (y_hue - 0.5) * ymid0
    yn = t.sample(t.fbm(0.0026, 2, 0.5, seed=6, stretch=(7, 1)), t.X, phase * ysp)      # varies along each yarn
    yb = 0.78 * sat(np.cos(pi * fy)) + 0.22 * np.sqrt(np.maximum(1 - (2 * fy * 0.93) ** 2, 0.0))             # round yarn crown
    ygap = sat(1 - yb / 0.55) ** 1.2 * (1 - groove * 0.7)
    yamp = 0.00042 * (0.35 + 0.65 * cth) * (1 - groove * 0.6)
    ymid = 1 - smoothstep(0.26, 0.5, np.abs(fy))                 # per-yarn offsets die out at the seams (no cliffs)
    yarn = yamp * (yb - 1.0) * (0.85 + 0.3 * y_fat * ymid) + ((y_proud - 0.5) * 0.00018 * ymid + (yn - 0.5) * 0.00022) * (1 - groove)

    # --- fibres: bast fibre bundles spiralling in each yarn (same hand as the lay), 0.1-0.2 mm
    q = FQ * phase + MQ * U
    qsp = t.H / PQ
    fibA = t.sample(t.fbm(0.00042, 3, 0.62, seed=7, stretch=(22, 1)), t.X, q * qsp)
    fibB = t.sample(t.fbm(0.0011, 2, 0.55, seed=8, stretch=(14, 1)), t.X, q * qsp + 0.011)
    fibre = 0.6 * fibA + 0.4 * fibB
    grit = t.fbm(0.0006, 2, 0.6, seed=9)

    height = strand + yarn + ((fibre - 0.5) * 0.00010 * (0.4 + 0.6 * yb) + (grit - 0.5) * 0.00003) * (1 - groove * 0.8)
    # nothing in a rope is knife-sharp: bed the yarn seams and the groove bottom a little
    height = mix(height, t.blur(height, 0.00010), 0.75)

    # --- whiskers: broken fibre ends standing off the yarns, mostly lying along the rope
    hr = np.maximum(t.scratches(density=420000, length=0.0040, width=0.00012, seed=10, angle=6, angle_spread=24),
                    t.scratches(density=200000, length=0.0070, width=0.00013, seed=11, angle=3, angle_spread=14))
    hr = sat(t.blur(t.warp(hr, 0.0003, 0.006, seed=12), 0.00006) * 1.5) * (1 - groove * 0.6)
    hair_top = np.maximum(t.blur(height, 0.0007) + 0.00012, height + 0.00003)
    height = mix(height, np.maximum(height, hair_top), hr)

    # --- masks
    crown = smoothstep(0.72, 1.0, cth) * yb                      # what a hand or a bollard touches
    cav = sat(groove * 0.9 + ygap * 0.5)

    # --- colour: weathered hemp, grey-tan; yarn to yarn drift; dark grime and old tar in the lay
    macro = t.fbm(0.03, 3, 0.5, seed=13)
    blot = t.fbm(0.009, 3, 0.55, seed=14)
    tn = sat(0.47 + (y_tone - 0.5) * 0.42 + (yn - 0.5) * 0.40 + (fibre - 0.5) * 0.55 * (1 - groove * 0.7) + (macro - 0.5) * 0.20)
    base = ramp(tn, [(0.0, "#6b5a40"), (0.25, "#85724f"), (0.5, "#9b8862"),
                     (0.75, "#ad9b76"), (1.0, "#c0b08c")])
    base = mix(base, desaturate(base, 0.28) * 0.97, sat((0.30 - y_hue) * 3.0 + (blot - 0.6) * 1.5)[..., None])   # greyed
    base = mix(base, base * hexlin("#ecd2b2") * 1.13, sat((y_hue - 0.72) * 3.0)[..., None])                        # redder
    base = base * (1.0 - 0.42 * ygap)[..., None]
    base = mix(base, base * hexlin("#7a6750") * 1.05, (groove * (0.65 + 0.5 * blot))[..., None] * 0.85)
    base = mix(base, hexlin("#3a2f23"), smoothstep(0.455, 0.50, np.abs(s)) * 0.55)
    base = mix(base, base * 1.10 + 0.012, crown * (0.35 + 0.4 * macro))            # rubbed pale on the crowns
    hair_col = ramp(t.noise(0.004, seed=15), [(0.2, "#a8966f"), (0.8, "#d0c19c")])
    albedo = mix(base, hair_col, hr * 0.6)
    albedo = albedo * (0.95 + 0.10 * macro)[..., None]

    # --- roughness: dry fibre; crowns handled smoother, grooves dusty
    rough = 0.88 - 0.12 * crown + 0.08 * groove + 0.05 * ygap + 0.10 * (fibre - 0.5) + 0.05 * (macro - 0.5)
    rough = mix(rough, 0.74, hr * 0.6)

    ao = sat(1.0 - 0.62 * smoothstep(0.25, 0.5, np.abs(s)) ** 1.5 - 0.22 * ygap * (1 - groove))
    return dict(albedo=sat(albedo), height=height, rough=sat(rough), metal=0.0, ao=ao)
