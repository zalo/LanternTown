"""Riven blue-black slate: the face left when the block is split along its cleavage - thin laminae
breaking off in stepped, feather-edged terraces, a fine lineation along the grain (V), satin sheen
on the cleavage planes, dust and rust blooms in the lee of the steps."""
from tk.tex import *
import math


def terraces(t, n, levels, riser_w=0.002, ragged=0.002, seed=0):
    """Quantise a 0..1 field into `levels` flat treads separated by crisp risers of constant
    physical width `riser_w` (m).  The step lines are torn by a fine domain warp (`ragged` m) so
    their raggedness is bounded in distance wherever the field is flat.
    Returns (stepped 0..1, tread index)."""
    if ragged > 0:
        n = t.warp(n, ragged, 0.012, seed=900 + seed)
        n = t.warp(n, ragged * 2.0, 0.04, seed=950 + seed)
    f = n * levels
    gx, gy = t.slope(f)
    g = np.sqrt(gx * gx + gy * gy)
    k = np.floor(f)
    fr = f - k
    w = np.clip(riser_w * g, 0.01, 0.6)
    st = (k + smoothstep(0.0, w, fr)) / levels
    return st.astype(F), k


def hash01(k, s):
    v = np.sin(k * 12.9898 + s * 78.233) * 43758.5453
    return (v - np.floor(v)).astype(F)


def build(t):
    # ------------------------------------------------------------------ height (metres)
    # broad lay of the split: laminae peel in long tongues running with the grain (along V)
    n1 = t.fbm(0.42, 3, 0.40, seed=3, stretch=(1.0, 2.2))
    n1 = t.warp(n1, 0.012, 0.15, seed=4)
    st1, k1 = terraces(t, n1, 4, 0.0022, 0.0022, 1)
    n2 = t.fbm(0.20, 3, 0.45, seed=5, stretch=(1.0, 2.0))
    st2, k2 = terraces(t, n2, 3, 0.0018, 0.0018, 2)
    # thin feather flakes: very shallow and small
    n3 = t.fbm(0.075, 3, 0.5, seed=7, stretch=(1.0, 1.7))
    st3, k3 = terraces(t, n3, 2, 0.0014, 0.0012, 3)

    und = (t.fbm(0.4, 3, 0.45, seed=8) - 0.5) * 0.0012         # the slab is never dead flat
    line = t.fibres(0.0026, 16.0, seed=9, axis='y')           # cleavage lineation
    line2 = t.fbm(0.010, 3, 0.55, seed=10, stretch=(1.0, 9.0))
    ripple = t.fbm(0.035, 2, 0.5, seed=11, stretch=(2.5, 1.0))  # faint cross ripples (plumose hackle)
    grain = t.fbm(0.004, 3, 0.6, seed=12)

    wav = (t.fbm(0.09, 3, 0.5, seed=14, stretch=(1.0, 1.6)) - 0.5) * 0.0013   # each lamina is gently wavy
    height = (und + wav + (st1 - 0.5) * 0.0034 + (st2 - 0.5) * 0.0016 + (st3 - 0.5) * 0.00036
              + (line - 0.5) * 0.00008 + (line2 - 0.5) * 0.00020 + (ripple - 0.5) * 0.00055
              + (grain - 0.5) * 0.00005)
    # pin-holes where pyrite cubes have weathered out
    pyr = t.speckle(density=300, size=0.0032, seed=13, soft=0.35)
    height = height - pyr * 0.0004

    # ------------------------------------------------------------------ masks
    lip = t.edges(height, 0.0025, 0.0003)                    # top edge of each riser: scuffed pale
    foot = t.cavity(height, 0.0035, 0.0004)                  # sheltered foot of each riser: dust
    foot_w = t.cavity(height, 0.012, 0.0008)

    # ------------------------------------------------------------------ colour
    # every lamina is a slightly different bed: tone steps with the terraces
    lam = (hash01(k1, 1.0) - 0.5) * 0.5 + (hash01(k2, 2.0) - 0.5) * 0.35 + (hash01(k3, 3.0) - 0.5) * 0.15
    cloud = t.fbm(0.3, 4, 0.5, seed=20)
    mott = t.fbm(0.04, 4, 0.55, seed=21, stretch=(1.0, 2.0))
    tone = sat(0.5 + lam * 0.30 + (cloud - 0.5) * 0.45 + (mott - 0.5) * 0.40 + (line2 - 0.5) * 0.10)
    alb = ramp(tone, [(0.0, "#2f343b"), (0.3, "#3a4048"), (0.55, "#444a52"), (0.8, "#4e545c"), (1.0, "#595e65")])
    # hue drifts between beds: bluer, greener, a touch of plum
    alb = overlay_tint(alb, "#b7c4d6", smoothstep(0.5, 0.85, t.fbm(0.2, 3, 0.5, seed=22)) * 0.30)
    alb = overlay_tint(alb, "#c3cbb9", smoothstep(0.55, 0.9, t.fbm(0.17, 3, 0.5, seed=23)) * 0.25)
    alb = overlay_tint(alb, "#cfc0c9", smoothstep(0.6, 0.9, t.fbm(0.23, 3, 0.5, seed=24)) * 0.20)
    # pale silty ribbons crossing the grain at a low angle (bedding seen on the cleavage)
    rib = smoothstep(0.62, 0.9, t.fbm(0.09, 2, 0.5, seed=25, stretch=(4.0, 1.0)))
    alb = mix(alb, alb * 1.16 + 0.004, rib * 0.45)
    # weathering bloom: exposed treads go slightly chalky; dust and rust gather under the steps
    bloom = smoothstep(0.5, 0.85, t.fbm(0.12, 4, 0.5, seed=26)) * (1 - foot_w)
    alb = mix(alb, hexlin("#6c737a"), bloom * 0.14)
    alb = mix(alb, hexlin("#8f959b"), lip * 0.14)
    alb = mix(alb, alb * hexlin("#8a7f72") * 1.05, sat(foot * 0.6 + foot_w * 0.3))
    rust_g = smoothstep(0.55, 0.85, t.fbm(0.13, 3, 0.5, seed=27))
    rust = sat(t.blur(pyr, 0.004) * 5.0) * 0.6 + sat(foot * rust_g * 0.6)
    alb = mix(alb, hexlin("#5e4633"), rust * 0.45)
    alb = alb * (0.975 + 0.05 * line)[..., None]
    alb = alb * (0.95 + 0.10 * ripple)[..., None]
    alb = desaturate(alb, 0.18)

    # ------------------------------------------------------------------ roughness
    rvar = t.fbm(0.08, 4, 0.55, seed=30)
    rough = 0.47 + (rvar - 0.5) * 0.22 + (line - 0.5) * 0.10 + lam * 0.22 + (ripple - 0.5) * 0.10
    rough = rough + bloom * 0.14 + sat(foot + foot_w * 0.5) * 0.25 + rust * 0.15 + lip * 0.08
    return dict(albedo=alb, height=height, rough=sat(rough), metal=0.0)
