#!/usr/bin/env python3
"""What a baked tier draws, everywhere a visitor can stand - without a browser.

    python3 tools/perf_sim.py [tier=phone|low] [near=..] [mid=..] [tris=..] [top=ring|near] [budget=view|all] [detail=..] [reach=..] [maxcut=..] [hyst=1] [town=town] [aspect=1.778] [fov=62] [step=2] [pitch=0] [pitches=0,15,..]

Replays the level choice of web/engine/lite_v1.js (BakedTown.update: near / mid radius, nearest first inside the
triangle budget) and three's frustum culling (bounding spheres) on the bake's manifest, for the review shots of the town and
for a grid over the walkable regions x 24 headings. Prints triangles / draw calls in view and resident triangles:
the shots, then mean / 95 % / worst over the grid, and what a visitor who walks everywhere downloads. The numbers of
a tier (BAKED_TIERS in lite.js) are the defaults; hyst=1 takes the radii a cell is kept to once it is in (the worst
a walk can leave standing). Use it to pick tier numbers; the real figures come from tools/perf_bench.py.
"""
import sys, os, re, json, math, gzip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    town = kw.get("town", "town")
    T = json.load(open(os.path.join(ROOT, "out", town, "town.json")))
    MP = os.path.join(ROOT, "out/web", kw.get("bake") or ("_bake" if town == "town" else "_bake_" + town), "manifest.json")  # (bake=_next/lean: another bake's folder)
    M = json.load(open(MP))
    tier = kw.get("tier", "phone")
    m = re.search(r"\b%s: \{([^}]*)\}" % re.escape(tier), open(os.path.join(ROOT, "web/engine/lite_v1.js")).read())
    if not m:
        sys.exit("perf_sim: no tier %r in BAKED_TIERS of web/engine/lite_v1.js" % tier)
    D = dict((k.strip(), v.strip().strip("'")) for k, v in (kv.split(":") for kv in m.group(1).split(",")))
    R_near, R_mid, CAP = float(kw.get("near", D["near"])), float(kw.get("mid", D["mid"])), float(kw.get("tris", D["tris"]))
    TOP, TEX = kw.get("top", D.get("top", "near")), D["tex"]
    BUDGET = kw.get("budget", D.get("budget", "all"))
    DETAIL, REACH = float(kw.get("detail", D.get("detail", 0.0055))), min(R_mid, float(kw.get("reach", D.get("reach", R_mid))))
    LADDER = [float(v) for v in kw.get("ladder", "1,1.6,2.5,4,6.3,10,16").split(",")]
    RES, R0 = float(kw.get("resident", D.get("resident", 0))), (0, R_mid, R_near)
    if kw.get("hyst") == "1":
        hold = float(D.get("hold", 1e9))
        R_near, R_mid = R_near and R_near + min(R_near * 0.2 + 2, hold), R_mid and R_mid + min(R_mid * 0.2 + 2, hold)
    aspect, fov, step = float(kw.get("aspect", 16 / 9)), float(kw.get("fov", 62)), float(kw.get("step", 2))
    chunks = []
    for c in M["chunks"]:
        lv = [c["lo"], c.get("mid"), c.get(TOP)]
        box = [c["lo"]["bmin"][0], c["lo"]["bmin"][2], c["lo"]["bmax"][0], c["lo"]["bmax"][2]]
        for m in lv[1:]:
            if m:
                box = [min(box[0], m["bmin"][0]), min(box[1], m["bmin"][2]), max(box[2], m["bmax"][0]), max(box[3], m["bmax"][2])]
        sph = []
        for m in lv:
            if not m:
                sph.append(None)
                continue
            ce = [(a + b) / 2 for a, b in zip(m["bmin"], m["bmax"])]
            sph.append((ce, 0.5 * math.dist(m["bmin"], m["bmax"]), m["ni"] // 3, len(m["sub"])))
        # the top level as the page draws it: [sphere, box on the ground, steps (part size, triangles down to it)] per
        # cluster of its opaque range (each a mesh of its own), and the other ranges as one more with the cell's sphere
        top = []
        if lv[2]:
            m = lv[2]
            for sub, steps, b in m.get("parts", []):
                top.append((([(b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2], 0.5 * math.dist(b[:3], b[3:])), (b[0], b[2], b[3], b[5]), [(size, n // 3) for size, n in steps], 1))
            rest = m["ni"] // 3 - sum(t[2][-1][1] for t in top)
            if rest:
                top.append(((sph[2][0], sph[2][1]), None, [(0, rest)], len(m["sub"]) - len(top)))
        chunks.append((box, sph, top))

    th = math.tan(math.radians(fov) / 2)
    tw = th * aspect
    kw_, kh_ = 1 / math.sqrt(1 + tw * tw), 1 / math.sqrt(1 + th * th)

    def frustum(pos, fwd):
        # -> test(centre, radius): does the sphere reach into the view (three's culling test; no far plane)
        fx, fy, fz = fwd
        l = math.sqrt(fx * fx + fy * fy + fz * fz)
        fx, fy, fz = fx / l, fy / l, fz / l
        rx, ry, rz = -fz, 0.0, fx  # right = fwd x up
        rl = math.hypot(rx, rz) or 1
        rx, rz = rx / rl, rz / rl
        ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx  # up = right x fwd

        def test(c, r):
            dx, dy, dz = c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]
            zc = dx * fx + dy * fy + dz * fz
            xc = dx * rx + dy * ry + dz * rz
            yc = dx * ux + dy * uy + dz * uz
            return not (zc < -r or (zc * tw - abs(xc)) * kw_ < -r or (zc * th - abs(yc)) * kh_ < -r)

        return test

    lo_tris = sum(sph[0][2] for box, sph, top in chunks)
    HW = math.atan(tw)  # half the horizontal field of view
    MAXCUT = float(kw.get("maxcut", D.get("maxcut", 0.3)))
    RELAX = kw.get("relax", "1") == "1"
    pm = re.search(r"const VIEW_PITCH = \[([^\]]*)\]", open(os.path.join(ROOT, "web/engine/lite_v1.js")).read())
    PITCHES = [math.radians(float(v.replace(" ", ""))) for v in kw.get("pitches", pm.group(1)).split(",")]  # the pitches the budget is kept for
    PITCH = math.radians(float(kw.get("pitch", 0)))  # of the grid views (the budget itself assumes a level camera)

    def levels(pos):
        x, z = pos[0], pos[2]
        order = []
        want = [0] * len(chunks)
        for i, (box, sph, top) in enumerate(chunks):
            d = math.hypot(max(box[0] - x, 0, x - box[2]), max(box[1] - z, 0, z - box[3]))
            w = 2 if sph[2] and R_near > 0 and d < R_near else 1 if sph[1] and R_mid > 0 and d < R_mid else 0
            if w:
                want[i] = w
                order.append((d, i))
        order.sort()
        first, order0 = list(want), order  # as asked by distance, before any ceiling
        if RES:
            # the ceiling on what is held (BakedTown.update): cells past their radius go first, then the farthest top cells
            held = [lo_tris + sum(chunks[i][1][want[i]][2] for d, i in order)]

            def down(i):
                held[0] -= chunks[i][1][want[i]][2]
                want[i] -= 1
                while want[i] > 0 and not chunks[i][1][want[i]]:
                    want[i] -= 1
                if want[i] > 0:
                    held[0] += chunks[i][1][want[i]][2]
            for d, i in reversed(order):
                while want[i] > 0 and held[0] > RES and d >= R0[want[i]]:
                    down(i)
            for d, i in reversed(order):
                if want[i] == 2 and held[0] > RES:
                    down(i)
            order = [(d, i) for d, i in order if want[i]]
        tris = lo_tris  # the budget is on everything shown
        cost = {}
        miss = [0, R_mid, 0, R_mid, []]  # top-level cells the budget refused, distance of the nearest cell it left at lo, step of the ladder, nearest cell refused the top level, steps of the top cells close by
        if BUDGET == "view":
            # BakedTown._viewBudget of web/engine/lite_v1.js: the budget is on what any one view from here can hold: 64
            # headings x the pitches of VIEW_PITCH. A sphere is in a view when it passes the side planes (heading within
            # its bearing +- the half field of view at that pitch + the angle it takes up, or around the back for one
            # overhead), the near and the upper plane and, looking up or down past the edge of the picture, the plane that
            # then cuts off what is straight ahead; each arc half a heading step wider
            HEADS = 64
            S = HEADS * len(PITCHES)
            st = 2 * math.pi / HEADS
            hs = st / 2
            sw, sh = math.sqrt(1 + tw * tw), math.sqrt(1 + th * th)

            def mask(sp):
                dx, dz = sp[0][0] - x, sp[0][2] - z
                d, r = max(math.hypot(dx, dz), 1e-3), sp[1]
                a = math.atan2(dx, dz)
                B = []
                for h in range(HEADS):
                    b = abs(h * st - a) % (2 * math.pi)
                    B.append(2 * math.pi - b if b > math.pi else b)
                out = []
                for pk in PITCHES:
                    p_ = abs(pk)
                    sp_, cp = math.sin(p_), math.cos(p_)
                    dy = pos[1] - sp[0][1] if pk < 0 else sp[0][1] - pos[1]
                    A = tw * cp
                    phi, rho = math.atan(A), (r * sw + tw * dy * sp_) / (d * math.sqrt(1 + A * A))
                    w1, w2 = 4, 0
                    if rho <= -1:
                        w1, w2 = -1, 4
                    elif rho < 1:
                        t = math.asin(rho)
                        w1, w2 = phi + t, math.pi + phi - t
                    cn = -(r + dy * sp_) / (d * cp)
                    wn = -1 if cn >= 1 else 4 if cn <= -1 else math.acos(cn)
                    q = sp_ - th * cp
                    cl = (dy * (cp + th * sp_) + r * sh) / (d * q) if q > 0 else 1
                    wl = 0 if cl >= 1 else 4 if cl <= -1 else math.acos(cl)
                    ct = (dy * (cp - th * sp_) - r * sh) / (d * (sp_ + th * cp))
                    wn = min(wn, -1 if ct >= 1 else 4 if ct <= -1 else math.acos(ct))
                    out += [(b <= w1 + hs or b >= w2 - hs) and b <= wn + hs and b >= wl - hs for b in B]
                return out

            # per cell: what each heading sees of it at lo / at mid, and the clusters of its top level (headings, distance)
            see = [[[sph[0][2] if v else 0 for v in mask(sph[0])], None, None] for box, sph, top in chunks]
            base = [sum(row[0][h] for row in see) for h in range(S)]
            dist = {}
            for d, i in order:
                box, sph, top = chunks[i]
                dist[i] = d
                if sph[1]:
                    see[i][1] = [sph[1][2] if v else 0 for v in mask(sph[1])]
                if want[i] == 2:
                    see[i][2] = [(mask(sp), 0.0 if b is None else math.hypot(max(b[0] - x, 0, x - b[2]), max(b[1] - z, 0, z - b[3]))) for sp, b, steps, draws in top]
            ask = list(want)
            far = [i for d, i in reversed(order) if ask[i] == 2 and d > 0]  # top cells the last step may give up, farthest first

            # the top level of cell i at step f of the ladder: triangles per cluster (each down to the part size that
            # shows at its distance x f), and what each heading sees of them
            def top_at(i, f):
                ns = []
                nw = [0] * S
                for (sp, b, steps, draws), (mk, d) in zip(chunks[i][2], see[i][2]):
                    size = min(MAXCUT, d * DETAIL * f)
                    n = next((n for sz, n in steps if sz <= size), steps[-1][1])
                    ns.append(n)
                    for h in range(S):
                        if mk[h]:
                            nw[h] += n
                return ns, nw

            def solve(f, drop):
                load = list(base)
                cur = [row[0] for row in see]

                # what the headings see of cell i becomes nw, if all that see more of it have the room
                def put(i, nw):
                    c0 = cur[i]
                    if any(nw[h] > c0[h] and load[h] + nw[h] - c0[h] > CAP for h in range(S)):
                        return False
                    for h in range(S):
                        load[h] += nw[h] - c0[h]
                    cur[i] = nw
                    return True

                want = list(ask)
                step, drawn = {}, {}
                for i in far[:drop]:
                    want[i] = 1
                for d, i in order:
                    if want[i] == 2:
                        ns, nw = top_at(i, f)
                        if put(i, nw):
                            step[i], drawn[i] = f, ns
                        else:
                            want[i] = 1
                refused = sum(1 for i in far if want[i] == 1 and i not in far[:drop])
                gap = R_mid
                for d, i in order:
                    if want[i] == 1 and not (see[i][1] and put(i, see[i][1])):
                        want[i] = 0
                        gap = min(gap, d)
                return want, step, drawn, refused, gap, put

            for f in LADDER:
                want, step, drawn, refused, gap, put = solve(f, 0)
                if not refused and gap >= REACH:
                    break
            else:
                for drop in range(1, len(far) + 1):
                    if gap >= REACH:
                        break
                    want, step, drawn, refused, gap, put = solve(f, drop)
            # what the tightest heading asked of all is given back where that heading does not look: nearest first, each
            # top cell as fine as it now fits
            if RELAX:
                for d, i in order:
                    if ask[i] != 2:
                        continue
                    for g in LADDER:
                        if g >= step.get(i, 1e9):
                            break
                        ns, nw = top_at(i, g)
                        if put(i, nw):
                            want[i], step[i], drawn[i] = 2, g, ns
                            break
            lost = [d for d, i in order0 if first[i] == 2 and want[i] < 2 and d < R0[2]]
            close = [(step.get(i, 0) if want[i] == 2 else 1e9) for d, i in order0 if first[i] == 2 and d < 8]
            return want, [refused, gap, f, min(lost) if lost else R_mid, close], drawn
        for d, i in order:  # mid for all that fit, nearest first ...
            sph = chunks[i][1]
            cost[i] = 0
            if sph[1] and tris + sph[1][2] - sph[0][2] <= CAP:
                cost[i] = sph[1][2]
                tris += cost[i] - sph[0][2]
            elif sph[1] or want[i] < 2:
                want[i] = 0
                miss[1] = min(miss[1], d)
        for d, i in order:  # ... then near
            if want[i] < 2:
                continue
            n = chunks[i][1][2][2] - (cost[i] or chunks[i][1][0][2])
            if tris + n <= CAP:
                tris += n
            else:
                want[i] = 1 if cost[i] else 0
                miss[0] += 1
        return want, miss, {}

    def view(test, lv_):
        want, _, drawn = lv_
        tris = draws = res = 0
        lv = [0, 0, 0]
        for i, (box, sph, top) in enumerate(chunks):
            s = sph[want[i]]
            res += s[2] + (sph[0][2] if want[i] else 0)  # (the lo level stays on the GPU under the others)
            lv[want[i]] += 1
            if i in drawn:  # the top level in clusters
                for (sp, b, steps, dr), n in zip(top, drawn[i]):
                    if n and test(sp[0], sp[1]):
                        tris += n
                        draws += dr
            elif test(s[0], s[1]):
                tris += s[2]
                draws += s[3]
        return tris, draws, res, lv

    b2t = lambda p: (p[0], p[2] if len(p) > 2 else 0.0, -p[1])
    print(f"tier: {TOP} {R_near} m, mid {R_mid} m, budget {CAP:.0f} tris shown; fov {fov}, aspect {aspect:.3f}")
    sp = T["spawn"]
    yaw = math.radians(sp.get("yaw", 0))
    shots = [("spawn", b2t(sp["pos"]), (-math.sin(yaw), 0.0, -math.cos(yaw)))]
    for s in T["shots"]:
        p, t = b2t(s["pos"]), b2t(s["target"])
        shots.append((s["name"], p, (t[0] - p[0], t[1] - p[1], t[2] - p[2])))
    for name, p, f in shots:
        w = levels(p)
        tris, draws, res, lv = view(frustum(p, f), w)
        print(f"  {name:24s} {tris / 1e3:6.0f}k tris  {draws:4d} draws   resident {res / 1e3:6.0f}k  levels lo/mid/near {lv}" + (f"  step {w[1][2]:g}" if w[1][2] else ""))
    rows = []
    spots = []
    for r in T["regions"]:
        x0, y0, x1, y1 = r["rect"]
        nx, ny = max(1, round((x1 - x0) / step)), max(1, round((y1 - y0) / step))
        for i in range(nx):
            for j in range(ny):
                x, y = x0 + (i + 0.5) * (x1 - x0) / nx, y0 + (j + 0.5) * (y1 - y0) / ny
                p = b2t([x, y, (r.get("z") or 0) + 1.65])
                w = levels(p)
                spots.append(w[1])
                for k in range(24):
                    a = (k + 0.5) * math.pi / 12  # between the headings the view budget is checked for
                    tris, draws, res, lv = view(frustum(p, (math.sin(a) * math.cos(PITCH), math.sin(PITCH), math.cos(a) * math.cos(PITCH))), w)
                    rows.append((tris, draws, res, r["name"], round(x, 1), round(y, 1), k * 15 + 7.5))
    if kw.get("quiet"):
        rows = [r[:3] + ("", 0, 0, 0) for r in rows]
    for col, name in ((0, "tris in view"), (1, "draws"), (2, "resident tris")):
        v = sorted(r[col] for r in rows)
        worst = max(rows, key=lambda r: r[col])
        print(f"  grid {name:14s} mean {sum(v) / len(v):9.0f}  95% {v[int(len(v) * 0.95)]:8d}  worst {v[-1]:8d}  ({worst[3]} {worst[4]},{worst[5]} heading {worst[6]})")
    print(f"  ({len(rows)} views; draws exclude the page's own ~8: sky, sprites, post)")
    short = sorted(m[1] for m in spots)
    print(f"  budget: the top level is refused to a cell in reach at {sum(1 for m in spots if m[0]) / len(spots) * 100:.1f} % of the spots; "
          f"mid reaches {sum(short) / len(short):.0f} m on average, {short[len(short) // 20]:.0f} m at the tightest 5 %, {short[0]:.0f} m at worst (asked: {R_mid:g} m)")
    ref = sorted(m[3] for m in spots if m[3] < R_mid)
    if ref:
        print(f"  a cell in reach is left without the top level at {len(ref) / len(spots) * 100:.0f} % of the spots (nearer than 8 m: {sum(1 for d in ref if d < 8) / len(spots) * 100:.0f} %, "
              f"than 4 m: {sum(1 for d in ref if d < 4) / len(spots) * 100:.0f} %); the nearest such cell: {ref[0]:.1f} m at worst, {ref[len(ref) // 2]:.1f} m median")
    if BUDGET == "view":
        close = [f for m in spots for f in m[4]]
        share = lambda lo, hi: sum(1 for f in close if lo < f <= hi) / max(1, len(close)) * 100
        print(f"  top cells within 8 m: {share(0, 1):.0f} % whole (parts down to 3 px), {share(1, 2.5):.0f} % down to 8 px, {share(2.5, 6.3):.0f} % to 19 px, "
              f"{share(6.3, 1e8):.0f} % coarser (parts under {MAXCUT:g} m at most), {share(1e8, 1e10):.0f} % at the mid level")
    # the files of the levels this tier ever shows (a visitor who has been everywhere), as stored / as a gzip server sends them
    files = [M["base"]["file"]] + [name + ".webp" for name in M["tex"][TEX]["pages"] + M["tex"][TEX].get("hpages", [])]
    files += [c[l]["file"] for c in M["chunks"] for l in (R_mid > 0 and "mid", R_near > 0 and TOP) if l and c.get(l)]
    raw = gz = 0
    for f in files:
        b = open(os.path.join(os.path.dirname(MP), f), "rb").read()
        raw += len(b)
        gz += len(b) if f.endswith(".webp") else min(len(b), len(gzip.compress(b, 6)))
    print(f"  everything this tier can fetch: {len(files)} files, {raw / 1e6:.1f} MB ({gz / 1e6:.1f} MB gzip), the page's scripts aside")


if __name__ == "__main__":
    main()
