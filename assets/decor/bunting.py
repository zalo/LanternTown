"""Festival bunting: a three-strand hemp line slung between two fixing points with a row of sewn canvas
pennants. Every pennant is a limp piece of cloth with real thickness whose top is folded into a tape
sleeve (teardrop hem) threaded on the line; optional bound edges or woven stripes. Each end of the line
passes through a forged screw-eye on a rose plate, is turned back and seized with twine, and leaves a
knotted, frayed tail. Wind lifts, bellies and ripples the pennants, twist / curl come from the seed, age
fades, soils and tears them. Origin = first fixing point, second fixing point = (span, 0, 0).
"""
from tk import *
import numpy as np
from math import sin, cos, pi, radians, sqrt, atan2
from mathutils import Vector

META = {"name": "Bunting string", "class": "component", "mount": "hang", "mount_height": 3.4,
        "focus": [1.1, 0.0, -0.22]}
PALETTES = {
    "festival": "#a8292b,#e6d8b2,#3565a4,#dba02e,#3a8257",
    "heraldic": "#8a2230,#d4a843,#2c4f85",
    "harvest": "#b5532a,#d9a441,#6f7f3a,#e6d7b3",
    "pastel": "#d9a29a,#ebdfc4,#9cb8c9,#b7c69c,#e3c98a",
    "royal": "#5a429c,#d2a942",
    "tricolour": "#b0282e,#ece4d0,#3463a6",
    "linen": "#e2d6bb,#b9a57c",
}
PARAMS = {
    "span": {"default": 5.0, "min": 2.0, "max": 14.0, "desc": "distance between the two fixing points (m); second point is at (span, 0, 0)"},
    "sag": {"default": 0.38, "min": 0.03, "max": 2.5, "desc": "dip of the line at mid-span (m, limited to 0.3 x span)"},
    "flag_width": {"default": 0.23, "min": 0.12, "max": 0.5, "desc": "pennant width along the line (m)"},
    "flag_length": {"default": 0.31, "min": 0.12, "max": 0.8, "desc": "pennant drop (m)"},
    "gap": {"default": 0.055, "min": 0.0, "max": 0.6, "desc": "clear line between neighbouring pennants (m)"},
    "shape": {"default": "triangle", "choices": ["triangle", "swallowtail", "scallop", "square", "mixed"], "desc": "pennant cut; mixed alternates triangle / swallow-tail"},
    "palette": {"default": "festival", "desc": "preset (festival|heraldic|harvest|pastel|royal|tricolour|linen) or a comma separated list of up to 5 '#rrggbb' colours, used alternately"},
    "pattern": {"default": "bordered", "choices": ["plain", "bordered", "striped", "diamond"], "desc": "plain cloth, edges bound in the trim tape, woven stripes, or an applique lozenge in the trim colour"},
    "trim": {"default": "#e8dcc0", "desc": "colour of the sleeve tape / edge binding / stripes"},
    "wind": {"default": 0.35, "min": 0.0, "max": 1.0, "desc": "breeze: lifts and ripples the pennants, bows the line"},
    "twist": {"default": 0.45, "min": 0.0, "max": 1.0, "desc": "amount of random twist and curl per pennant"},
    "age": {"default": 0.3, "min": 0.0, "max": 1.0, "desc": "0 = new, 1 = sun-faded, grimy, torn tips and the odd missing pennant"},
    "rope_diameter": {"default": 0.009, "min": 0.005, "max": 0.02, "desc": "line diameter (m)"},
    "lamps": {"default": 3, "min": 0, "max": 6, "desc": "number of glass fairy lamps (oil wick in a glass cup on a wire bail) hung in place of pennants"},
    "lamp_glass": {"default": "#ffb658", "desc": "glass colour of the fairy lamps"},
    "lamp_intensity": {"default": 1.3, "min": 0.5, "max": 6.0, "desc": "light of one fairy lamp (cd)"},
    "fixing": {"default": "eye_plate", "choices": ["eye_plate", "ring", "none"], "desc": "end fitting: screw-eye on a rose plate, a loose forged ring, or just the tied loop"},
}
VARIANTS = {
    "swallowtail_heraldic": {"span": 7.0, "sag": 0.7, "shape": "swallowtail", "palette": "heraldic", "pattern": "striped",
                             "trim": "#d9c58c", "flag_width": 0.26, "flag_length": 0.42, "gap": 0.09, "wind": 0.45, "lamps": 0, "seed": 7},
    "faded_scallop": {"span": 4.0, "sag": 0.3, "shape": "scallop", "palette": "harvest", "pattern": "plain", "age": 0.9,
                      "flag_width": 0.2, "flag_length": 0.24, "gap": 0.04, "wind": 0.1, "twist": 0.8, "lamps": 0, "fixing": "ring", "seed": 12},
    "long_mixed": {"span": 12.0, "sag": 1.3, "shape": "mixed", "palette": "tricolour", "pattern": "bordered", "wind": 0.6,
                   "flag_width": 0.3, "flag_length": 0.4, "gap": 0.12, "rope_diameter": 0.012, "lamps": 0, "seed": 23},
    "royal_lamps": {"span": 3.2, "sag": 0.2, "shape": "square", "palette": "royal", "pattern": "diamond", "trim": "#efe6cc",
                    "flag_width": 0.2, "flag_length": 0.26, "gap": 0.07, "wind": 0.15, "age": 0.1, "lamps": 3, "fixing": "none", "seed": 31},
}

TIERS = [(10, 12), (8, 12), (8, 10), (8, 8), (6, 8), (6, 6), (4, 6), (4, 5)]      # pennant grid by triangle budget


def _tris(objs):
    return sum(sum(len(pl.vertices) - 2 for pl in o.data.polygons) for o in objs)


def _palette(spec):
    spec = PALETTES.get(str(spec).strip().lower(), str(spec))
    cols = [c.strip() for c in spec.split(",") if c.strip().startswith("#") and len(c.strip()) == 7]
    return (cols or PALETTES["festival"].split(","))[:5]


def build(p, rng):
    span = p.span
    sag = min(p.sag, span * 0.3)
    rr = p.rope_diameter * 0.5
    wind, twist, age = p.wind, p.twist, p.age
    W, H = p.flag_width, p.flag_length
    n_lamps = int(p.lamps)
    cols = _palette(p.palette)[:4 if n_lamps else 5]

    rope_m = mat("rope", "#c9b48a")
    iron_m = mat("iron_wrought")
    trim_m = mat("canvas", p.trim)
    thread_m = mat("canvas", "#4b3b2a")
    if p.pattern == "striped":
        flag_ms = [mat("canvas", c, stripe=max(0.03, W / 3.5), tint2=p.trim) for c in cols]
    else:
        flag_ms = [mat("canvas", c) for c in cols]

    # ------------------------------------------------------------------ end fittings and tied loops
    wire = 0.0032
    if p.fixing == "eye_plate":
        ring_r, ring_c = 0.013, 0.012 + 0.013
    elif p.fixing == "ring":
        ring_r, ring_c = 0.02, 0.02 + wire
    else:
        ring_r, ring_c = 0.0, 0.0
    rho = wire + rr + 0.0006                      # rope centre-line radius around the ring wire
    far = ring_c + ring_r if p.fixing != "none" else rho
    xk = far + 0.04 + 3 * rr                      # where the turned-back end is seized to the line
    iron, ropes = [], []

    def fitting(sx, fx_, r):
        X = lambda lx: fx_ + sx * lx
        if p.fixing == "eye_plate":
            iron.append(lathe("rose", [(0.0, 0.0), (0.023, 0.0), (0.0235, 0.0018), (0.021, 0.0036), (0.012, 0.0046), (0.008, 0.0075),
                                       (0.0062, 0.012), (0.0, 0.012)], segments=12, loc=(fx_, 0, 0), rot=(0, 90 * sx, 0)))
            for a in (35, 215):
                iron.append(cylinder("screw", 0.0034, 0.0022, loc=(X(0.0044), 0.0165 * cos(radians(a)), 0.0165 * sin(radians(a))),
                                     rot=(0, 90 * sx, 0), verts=7, radius2=0.0022))
        if p.fixing != "none":
            iron.append(torus("eye", ring_r, wire, loc=(X(ring_c), 0, 0), rot=(90, 0, r.jitter(0, 6 if p.fixing == "ring" else 0)),
                              major_seg=14, minor_seg=5))
        # line through the eye, turned back on itself
        s1 = rr * 1.02
        loop = [(xk, s1), (far + 0.03, s1 + 0.001), (far + 0.014, rho * 1.25), (far + 0.004, rho * 1.08), (far - rho * 0.72, rho * 0.72),
                (far - rho, 0.0), (far - rho * 0.72, -rho * 0.72), (far + 0.004, -rho * 1.08), (far + 0.014, -rho * 1.25),
                (far + 0.03, -s1 - 0.001), (xk, -s1)]
        ropes.append(sweep("loop", [(X(lx), ly, 0.0) for lx, ly in loop], rr, sides=5, smooth_path=False))
        # seizing: tight turns of thin cord round both strands
        turns, hp = 5, []
        for k in range(turns * 5 + 1):
            a = k / 5 * 2 * pi
            hp.append((X(xk - 2.4 * rr + k / 5 * rr * 0.8), (2.15 * rr) * cos(a), (1.4 * rr) * sin(a)))
        ropes.append(sweep("seizing", hp, rr * 0.42, sides=3, smooth_path=False))
        # free tail drooping from the seizing, overhand stopper knot and a frayed end
        tl = r.u(0.09, 0.17)
        ty = -s1 + r.jitter(0.0, 0.012)
        tail = bezier_path((X(xk + 0.5 * rr), -s1, -0.1 * rr), (X(xk + 0.03), -s1 - 0.002, 0.002),
                           (X(xk + 0.042), ty, -tl * 0.2), (X(xk + 0.04 + r.jitter(0, 0.012)), ty * 1.4, -tl), n=8)
        ropes.append(sweep("tail", tail, rr, sides=5, smooth_path=False))
        ropes.append(sphere("stopper", rr * 1.8, loc=tuple(Vector(tail[-3])), segments=7, rings=4, scale=(1.0, 1.0, 0.9)))
        e0, e1 = Vector(tail[-2]), Vector(tail[-1])
        dv = (e1 - e0).normalized()
        ropes.append(sweep("fray", [tuple(e1 - dv * 0.004), tuple(e1 + dv * 0.016)], rr, sides=5,
                           scale_fn=lambda t: 0.95 + 0.55 * t, smooth_path=False))

    fitting(1, 0.0, rng.fork("fit0"))
    fitting(-1, span, rng.fork("fit1"))

    # ------------------------------------------------------------------ the line
    bow = wind * min(0.22 * sag, 0.25)
    x0, x1 = xk - rr, span - xk + rr
    T = np.linspace(0.0, 1.0, 241)
    Q = 1.0 - (2.0 * T - 1.0) ** 2
    PX, PY, PZ = x0 + (x1 - x0) * T, bow * Q, -sag * Q
    S = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(PX) ** 2 + np.diff(PY) ** 2 + np.diff(PZ) ** 2))])
    L = float(S[-1])

    def P(s):
        return Vector((float(np.interp(s, S, PX)), float(np.interp(s, S, PY)), float(np.interp(s, S, PZ))))

    nseg = int(clamp(L / 0.08, 36, 96))
    line = sweep("line", [tuple(P(L * i / nseg)) for i in range(nseg + 1)], rr, sides=6, smooth_path=False)
    ropes.append(line)

    # ------------------------------------------------------------------ layout: pennant slots, some taken by lamps
    margin = 0.14
    usable = max(L - 2 * margin, W)
    n = int(clamp(int((usable + p.gap) / (W + p.gap)), 1, 64))
    pitch = (usable - W) / (n - 1) if n > 1 else 0.0
    s_first = (L - pitch * (n - 1)) * 0.5
    m = min(n_lamps, n // 3)
    lamp_slots = sorted({int((k + 0.5) / m * n) for k in range(m)}) if m else []

    # ------------------------------------------------------------------ fairy lamps: oil wick in a glass cup hung on a wire bail
    lamp_iron, lamp_glass, lamp_flame = [], [], []
    for k, slot in enumerate(lamp_slots):
        li0, lg0, lf0 = len(lamp_iron), len(lamp_glass), len(lamp_flame)
        r = rng.fork(f"lamp{k}")
        b = P(s_first + slot * pitch)
        rk = rr + 0.0024
        stem, hb, rc = r.u(0.085, 0.12), 0.05, 0.0288
        zc = b.z - stem - hb                       # wire collar under the cup rim
        zb = zc - 0.054                            # underside of the cup
        yaw = r.jitter(0.0, 25.0)
        ca, sa = cos(radians(yaw)), sin(radians(yaw))
        hook = [(b.x, b.y + rk * sin(radians(a)), b.z + rk * cos(radians(a))) for a in (-120, -75, -30, 15, 60, 105, 140)]
        hook += [(b.x, b.y + rk * 0.4, b.z - rk * 1.6), (b.x, b.y, b.z - rk * 2.8), (b.x, b.y, b.z - stem)]
        lamp_iron.append(sweep("lamp_hook", hook, 0.0021, sides=4, smooth_path=False))
        bail = [(b.x + rc * cos(t) * ca, b.y + rc * cos(t) * sa, zc + hb * sin(t) ** 0.8) for t in np.linspace(0.0, pi, 9)]
        lamp_iron.append(sweep("lamp_bail", bail, 0.0016, sides=3, smooth_path=False))
        lamp_iron.append(torus("lamp_collar", rc, 0.0015, loc=(b.x, b.y, zc), major_seg=10, minor_seg=3))
        lamp_glass.append(lathe("lamp_cup", [(0.0, 0.0), (0.017, 0.0), (0.0225, 0.007), (0.0262, 0.05), (0.0295, 0.057), (0.0298, 0.062),
                                             (0.0272, 0.062), (0.0245, 0.05), (0.021, 0.009), (0.0, 0.006)], segments=10, loc=(b.x, b.y, zb)))
        lamp_iron.append(lathe("lamp_wick", [(0.0, 0.0), (0.011, 0.0), (0.0125, 0.004), (0.0045, 0.0065), (0.003, 0.013), (0.0, 0.013)],
                               segments=6, loc=(b.x, b.y, zb + 0.0065)))
        lamp_flame.append(lathe("lamp_flame", [(0.0, 0.0), (0.0034, 0.004), (0.004, 0.009), (0.0022, 0.016), (0.0, 0.023)], segments=6,
                                loc=(b.x, b.y, zb + 0.019)))
        light(f"lamp{k}", (b.x, b.y, zb + 0.03), color="#ffb45e", intensity=p.lamp_intensity, range=3.5, flicker=0.45, shadow_near=0.06)
        part(lamp_iron[li0:] + lamp_glass[lg0:] + lamp_flame[lf0:], f"lamp{k}", pivot=(b.x, b.y, b.z + rk), motion="swing", axis="X",
             amp=round(1.5 + 5.0 * wind, 2), speed=round(r.u(0.35, 0.5), 2))

    # ------------------------------------------------------------------ pennants
    n_flags = max(n - len(lamp_slots), 1)
    overhead = _tris(iron + ropes + lamp_iron + lamp_glass + lamp_flame)
    ns = 2 if n <= 40 else 1                      # sleeve segments along the line
    hem_tris = (ns + 2) * 8 * 2
    n_st = int(W / 0.021) if n_flags <= 34 else 0   # running stitches per pennant face
    per = (11500.0 - overhead) / n_flags - hem_tris - 4 * n_st - (96 if p.pattern == "diamond" else 0)
    nu, nv = next((t for t in TIERS if 4 * t[0] * t[1] + 4 * (t[0] + t[1]) <= per), TIERS[-1])
    hem_h = clamp(0.085 * H, 0.016, 0.03)
    bind = clamp(0.065 * W, 0.009, 0.022)
    print(f"DBG pennants={n_flags} lamps={len(lamp_slots)} grid={nu}x{nv} pitch={pitch:.3f} line={L:.2f}m overhead={overhead}")

    def depth(co):                                 # 0 at the line .. 1 at the pennant tips
        t = np.clip((co[:, 0] - x0) / max(x1 - x0, 1e-6), 0.0, 1.0)
        return np.clip((-sag * (1.0 - (2.0 * t - 1.0) ** 2) - co[:, 2]) / H, 0.0, 1.0)

    fade_n = noise_mask(0.45, seed=p.seed + 11, lo=0.2, hi=0.8)
    blot_n = noise_mask(0.07, seed=p.seed + 17, lo=0.45, hi=0.8)

    def grime_fn(co, nrm):                         # dirt and damp gather towards the tips, blotchy
        return np.clip((0.04 + 0.8 * age * age + 0.25 * age) * (0.25 + 0.75 * depth(co) ** 2) * (0.5 + 0.9 * blot_n(co, nrm)), 0.0, 1.0)

    def fade_fn(co, nrm):                          # sun-fade, uneven from pennant to pennant
        return np.clip((0.1 + 0.9 * age) * (0.25 + 0.75 * fade_n(co, nrm)), 0.0, 1.0)

    def fray_fn(co, nrm):                          # threadbare tips
        return np.clip(age * 0.6 * depth(co) ** 3, 0.0, 1.0)

    # Every pennant (with its sleeve and stitching) is its own 'flutter' part pivoting on the line, so it can
    # never leave the line; long strings group neighbours to keep the part count <= 24.
    gsize = max(1, -(-n_flags // 24))
    grp = {"flags": [], "hems": [], "v": [], "f": [], "piv": [], "k": 0}

    def flush():
        if not grp["flags"]:
            return
        k = grp["k"]
        fr = rng.fork(f"flutter{k}")
        fl = join(grp["flags"], f"cloth{k}") if len(grp["flags"]) > 1 else grp["flags"][0]
        paint(fl, grime=grime_fn, special=fade_fn, wear=fray_fn)
        hm = join(grp["hems"], f"sleeve{k}") if len(grp["hems"]) > 1 else grp["hems"][0]
        assign(hm, trim_m)
        paint(hm, grime=0.15 + 0.5 * age, special=fade_fn)
        objs = [fl, hm]
        if grp["f"]:
            st = mesh_object(f"stitches{k}", grp["v"], grp["f"])
            assign(st, thread_m)
            objs.append(st)
        pv = sum(grp["piv"], Vector((0, 0, 0))) / len(grp["piv"])
        part(objs, f"pennant{k}", pivot=tuple(pv), motion="flutter", axis="X",
             amp=round((3.0 + 14.0 * wind) * fr.u(0.75, 1.25), 2), speed=round((0.6 + 0.9 * wind) * fr.u(0.8, 1.25), 2),
             phase=round(fr.u(0.0, 1.0), 3))
        grp.update(flags=[], hems=[], v=[], f=[], piv=[], k=k + 1)

    for i in range(n):
        r = rng.fork(f"flag{i}")
        sc = s_first + i * pitch
        missing = r.chance(max(0.0, age - 0.45) * 0.3) and n > 3
        shape = p.shape if p.shape != "mixed" else ("triangle", "swallowtail")[i % 2]
        torn = r.chance(age * 0.45)
        Hf = H * r.vary(1.0, 0.03) * (r.u(0.72, 0.9) if torn else 1.0)
        rag = [(r.u(0.0, 0.07 + 0.1 * age) * Hf if torn else r.u(0.0, 0.02 * age) * Hf) for _ in range(nu + 1)]
        tap = {"triangle": 0.97 if not torn else r.u(0.7, 0.85), "swallowtail": 0.1, "scallop": 0.0, "square": 0.0}[shape]
        notch = r.vary(0.36, 0.06)
        bordered = p.pattern == "bordered"
        if missing:                                # torn off: a ragged stub is left in the sleeve
            shape, tap, bordered = "square", 0.0, False
            Hf = H * r.u(0.14, 0.26)
            rag = [r.u(0.0, 0.55) * Hf for _ in range(nu + 1)]
        th0 = radians(wind * r.u(12.0, 30.0) + r.jitter(0.0, 4.0 + 9.0 * twist))        # swing about the line
        kc = radians(wind * r.u(10.0, 50.0) + r.jitter(0.0, 10.0 + 55.0 * twist))       # curl towards the tip
        phi = radians(r.jitter(0.0, 10.0 + 65.0 * twist))                               # twist about the centre line
        drift = r.jitter(wind * 0.06, 0.03 + 0.1 * twist)
        rip = (0.005 + 0.016 * wind + 0.008 * twist) * r.u(0.7, 1.3) * (W / 0.23)
        lam = W * r.u(0.8, 1.4)
        ph = r.u(0.0, 2 * pi)
        rip2 = (0.003 + 0.007 * twist + 0.006 * wind) * r.u(0.5, 1.2) * (W / 0.23)
        ang2 = radians(r.u(25.0, 65.0) * r.pick([-1, 1]))
        lam2 = W * r.u(0.9, 1.5)
        ph2 = r.u(0.0, 2 * pi)
        ecurl = r.jitter(0.0, 0.07 * twist) + 0.03 * wind
        if i in lamp_slots:
            continue
        if len(grp["piv"]) >= gsize:
            flush()
        grp["piv"].append(P(sc))
        st_v, st_f = grp["v"], grp["f"]

        def lenf(g, shape=shape, notch=notch):
            g = clamp(abs(g), 0.0, 1.0)
            if shape == "swallowtail":
                return 1.0 - notch * (1.0 - g)
            if shape == "scallop":
                return 0.6 + 0.4 * sqrt(max(0.0, 1.0 - g * g))
            return 1.0

        def flat(u, v, shape=shape, tap=tap, Hf=Hf, rag=rag, bordered=bordered, lenf=lenf):
            ci, rj = int(round(u * nu)), int(round(v * nv))
            btm = bordered and shape != "triangle"
            if btm:
                vf = 1.0 if rj == nv else (rj / (nv - 1)) * (1.0 - bind / Hf)
            else:
                vf = rj / nv
            hw = 0.5 * W * (1.0 - tap * (rj / nv))
            if bordered:
                inner = max(hw - bind, hw * 0.3)
                a = -hw if ci == 0 else hw if ci == nu else -inner + 2 * inner * (ci - 1) / (nu - 2)
            else:
                a = -hw + 2 * hw * ci / nu
            d = vf * Hf * lenf(a / max(hw, 1e-6))
            if rj == nv or (rj == nv - 1 and btm):
                d -= rag[ci]
            return (a, 0.0, -d)

        def hang(co, sc=sc, Hf=Hf, th0=th0, kc=kc, phi=phi, drift=drift, rip=rip, lam=lam, ph=ph, ecurl=ecurl,
                 rip2=rip2, ang2=ang2, lam2=lam2, ph2=ph2):
            a, d, yt = co.x, -co.z, co.y
            s = clamp(d / Hf, 0.0, 1.2)
            psi = th0 + kc * s
            if abs(kc) < 1e-4:
                nc, zc = d * sin(th0), d * cos(th0)
            else:
                nc, zc = Hf * (cos(th0) - cos(psi)) / kc, Hf * (sin(psi) - sin(th0)) / kc
            pt = phi * s ** 1.3
            env = min(1.0, s * 2.4)
            off = nc + a * sin(pt)
            off += rip * sin(2 * pi * a / lam + ph + 2.6 * s) * env
            off += rip2 * sin(2 * pi * (a * cos(ang2) + d * sin(ang2)) / lam2 + ph2) * env
            off += ecurl * Hf * (a / (0.5 * W)) ** 2 * s
            b = P(sc + a * cos(pt) + drift * d * (1.0 + 0.5 * s))
            return Vector((b.x, b.y + off + yt * cos(psi), b.z - zc + yt * sin(psi)))

        f = surface(f"pennant{i}", flat, nu, nv, thickness=0.0013)
        assign(f, flag_ms[i % len(flag_ms)])
        if bordered:
            f.data.materials.append(trim_m)
            for ci in range(nu):
                for rj in range(nv):
                    if ci == 0 or ci == nu - 1 or (shape != "triangle" and rj == nv - 1):
                        f.data.polygons[ci * nv + rj].material_index = 1
        parts = [f]
        if p.pattern == "diamond" and not missing:
            # applique lozenge stitched on, a little proud of the cloth on both faces
            re_ = W * (0.2 if shape == "triangle" else 0.25)
            dc = Hf * {"triangle": 0.34, "swallowtail": 0.36, "scallop": 0.46, "square": 0.5}[shape]
            e = surface(f"lozenge{i}", lambda u, v, re_=re_, dc=dc: ((u - v) * re_, 0.0, -(dc + (u + v - 1.0) * re_ * 1.3)), 4, 4, thickness=0.0075)
            assign(e, trim_m)
            parts.append(e)
        for o in parts:
            uv_planar(o, (1, 0, 0), (0, 0, 1))
            realize(o)
            uv_keep(o)
            transform_verts(o, hang)
        grp["flags"] += parts

        # tape sleeve folded over the line and stitched below it (teardrop section), pinched shut at the ends
        R = rr + 0.0016
        prof = [(0.0017, hem_h)] + [(R * sin(radians(a)), -R * cos(radians(a))) for a in (140, 84, 28, -28, -84, -140)] + [(-0.0017, hem_h)]
        us = [-0.5 * W - 0.004, -0.5 * W - 0.0015] + [-0.5 * W + W * k / ns for k in range(1, ns)] + [0.5 * W + 0.0015, 0.5 * W + 0.004]
        nuh = len(us) - 1

        def hem(u, v, sc=sc, th0=th0, us=us, prof=prof, nuh=nuh):
            ci = int(round(u * nuh))
            pn, pd = prof[int(round(v * len(prof))) % len(prof)]
            if ci == 0 or ci == nuh:
                k = min(1.0, (rr * 0.8) / max(sqrt(pn * pn + pd * pd), 1e-6))
                pn, pd = pn * k, pd * k
            b = P(sc + us[ci])
            return (b.x, b.y + pn * cos(th0) + pd * sin(th0), b.z - pd * cos(th0) + pn * sin(th0))

        h = surface(f"hem{i}", hem, nuh, len(prof), closed_v=True)
        recalc_normals(h)
        grp["hems"].append(h)

        # running stitch that closes the sleeve, on both faces
        pd0, pd1 = hem_h * 0.66, hem_h * 0.66 + 0.0026
        for face in (-1.0, 1.0):
            for q in range(n_st):
                a0 = -0.5 * W + (q + 0.25 + r.jitter(0.0, 0.06)) * W / n_st
                a1 = a0 + 0.55 * W / n_st
                pn = face * 0.0031
                quad = []
                for (aa, pd) in ((a0, pd0), (a1, pd0), (a1, pd1), (a0, pd1)):
                    b = P(sc + aa)
                    quad.append((b.x, b.y + pn * cos(th0) + pd * sin(th0), b.z - pd * cos(th0) + pn * sin(th0)))
                if face * cos(th0) < 0:            # keep the quad facing away from the cloth
                    quad.reverse()
                k0 = len(st_v)
                st_v.extend(quad)
                st_f.append((k0, k0 + 1, k0 + 2, k0 + 3))

    flush()

    # ------------------------------------------------------------------ materials and weathering of the static parts
    for o in iron + lamp_iron:
        assign(o, iron_m)
        paint(o, special=mask_mul(noise_mask(0.02, seed=p.seed + 3, lo=0.35, hi=0.7), 0.35 + 0.6 * age), grime=0.2 + 0.3 * age)
    for o in ropes:
        assign(o, rope_m)
    if lamp_glass:
        glass_m, flame_m = mat("glass_lantern", p.lamp_glass, glow=2.6), mat("emissive", "#ffb45e", glow=9)
        for o in lamp_glass:
            assign(o, glass_m)
        for o in lamp_flame:
            assign(o, flame_m)
    paint(line, grime=0.2 + 0.4 * age)
    socket("start", "attach", (0, 0, 0), rot=(0, 0, 90), role="fixing")
    socket("end", "attach", (span, 0, 0), rot=(0, 0, -90), role="fixing")
    socket("mid", "attach", (span * 0.5, bow, -sag - rr), role="lantern")
