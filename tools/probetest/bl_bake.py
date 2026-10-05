"""Route A: Cycles as the light-probe baker for three.js' LightProbeGrid.

    blender --background --factory-startup --python tools/probetest/bl_bake.py -- scratch=<dir> [key=value ...]

    out=probes_blender   output stem in out/probetest/ (<stem>.bin + <stem>.json)
    w=64 h=32            equirect capture size per probe
    spp=64               Cycles samples per pixel
    denoise=0            OptiX denoiser on the capture (not needed: the SH projection is itself a 2048-pixel average)
    seed=0
    res=8,4,8            override the probe counts of scene.json's grid (same box)
    limit=0              only the first N probes (timing tests)
    sheet=1              also write <stem>_captures.png: the captures of one horizontal probe layer

One Cycles animation, one frame per probe: the panoramic camera is keyframed to every probe position, persistent
data keeps the scene on the GPU, every frame is written as a small multilayer EXR (radiance, depth, back-face AOV)
into scratch, then numpy projects it to 9 x RGB L2 SH coefficients in three's basis order.

File format (<stem>.bin): float32 little endian, [probe][coefficient 0..8][r, g, b], probes in TEXTURE order
(index = ix + iy * nx + iz * nx * ny in three space, X fastest). Coefficients are RADIANCE SH (not convolved): three's
shader applies the cosine convolution itself. Probe (ix, iy, iz) sits at box.min + i * size / (res - 1).
"""
import bpy, sys, os, json, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as C

A = dict(a.split("=", 1) for a in sys.argv[sys.argv.index("--") + 1:])
SCRATCH = A["scratch"]
STEM = A.get("out", "probes_blender")
W, H = int(A.get("w", 64)), int(A.get("h", 32))
SPP = int(A.get("spp", 64))
DENOISE = A.get("denoise", "0") == "1"
SEED = int(A.get("seed", 0))
LIMIT = int(A.get("limit", 0))
BF_INVALID = float(A.get("bf", 0.25))      # a probe that sees more back faces than this share of the sphere is inside something

t_start = time.time()
S = C.load_scene_json()
G = S["grid"]
res = [int(v) for v in A["res"].split(",")] if "res" in A else G["res"]
nx, ny, nz = res
sc = C.setup_cycles(SCRATCH, samples=SPP, denoise=DENOISE, seed=SEED)
info = C.build(sc, S)
t_built = time.time()

# probe positions, texture order
pos_three = []
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            idx = (ix, iy, iz)
            pos_three.append([G["center"][a] - G["size"][a] / 2 + idx[a] * G["size"][a] / (res[a] - 1) for a in range(3)])
N = len(pos_three)
n_bake = min(N, LIMIT) if LIMIT else N

cd = bpy.data.cameras.new("probe")
cd.type = "PANO"
cd.panorama_type = "EQUIRECTANGULAR"
cd.clip_start = 0.01
cd.clip_end = 1000.0
cam = bpy.data.objects.new("probe", cd)
cam.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
sc.collection.objects.link(cam)
sc.camera = cam
for i in range(n_bake):
    cam.location = C.t2b(pos_three[i])
    cam.keyframe_insert("location", frame=i + 1)
for fc in cam.animation_data.action.fcurves:
    for k in fc.keyframe_points:
        k.interpolation = "CONSTANT"
sc.frame_start, sc.frame_end = 1, n_bake
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
sc.render.image_settings.color_depth = "32"
sc.render.image_settings.exr_codec = "NONE"
exr_dir = os.path.join(SCRATCH, "exr")
os.makedirs(exr_dir, exist_ok=True)
sc.render.filepath = os.path.join(exr_dir, STEM + "_")

stamps = []
bpy.app.handlers.render_post.append(lambda *a: stamps.append(time.time()))
t0 = time.time()
bpy.ops.render.render(animation=True)
t_render = time.time() - t0
steps = np.diff(np.array(stamps)) if len(stamps) > 2 else np.array([t_render])

# ---- project
t0 = time.time()
db, dt, dw = C.equirect_dirs(W, H)
B = (C.sh_basis_three(dt) * dw[..., None]).reshape(-1, 9)          # (pixels, 9)
sh = np.zeros((N, 9, 3), dtype=np.float32)
bf = np.zeros(N, dtype=np.float32)
dmin = np.zeros(N, dtype=np.float32)
dmean = np.zeros(N, dtype=np.float32)
sheet = None
for i in range(n_bake):
    p = os.path.join(exr_dir, "%s_%04d.exr" % (STEM, i + 1))
    ch = C.exr_channels(p)
    rgb = np.stack([C.pick(ch, "Combined.R"), C.pick(ch, "Combined.G"), C.pick(ch, "Combined.B")], axis=-1)
    z = np.minimum(C.pick(ch, "Depth"), 100.0)
    b = C.pick(ch, ".bf.")
    sh[i] = B.T @ rgb.reshape(-1, 3)
    bf[i] = float((b * dw).sum() / (4 * math.pi))
    dmin[i] = float(z.min())
    dmean[i] = float((z * dw).sum() / (4 * math.pi))
    if A.get("sheet", "1") == "1":
        ix, iy, iz = i % nx, (i // nx) % ny, i // (nx * ny)
        if iy == min(1, ny - 1):
            if sheet is None:
                sheet = np.zeros((nz * H, nx * W, 3), dtype=np.float32)
            # image row 0 = far -Z in three = north in the courtyard, so the sheet reads like a map seen from above
            sheet[iz * H:(iz + 1) * H, ix * W:(ix + 1) * W] = rgb
    os.remove(p)
t_project = time.time() - t0

# ---- probes inside geometry: mark invalid, fill from valid neighbours so they cannot leak black (or lit window backs)
valid = bf <= BF_INVALID
valid[n_bake:] = False
raw = sh.copy()
filled = valid.copy()
grid_idx = lambda ix, iy, iz: ix + iy * nx + iz * nx * ny
rounds = 0
while not filled.all() and filled.any():
    rounds += 1
    new = filled.copy()
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                i = grid_idx(ix, iy, iz)
                if filled[i]:
                    continue
                acc, wsum = np.zeros((9, 3)), 0.0
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            jx, jy, jz = ix + dx, iy + dy, iz + dz
                            if (dx or dy or dz) and 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz and filled[grid_idx(jx, jy, jz)]:
                                wgt = 1.0 / (dx * dx + dy * dy + dz * dz)
                                acc += wgt * sh[grid_idx(jx, jy, jz)]
                                wsum += wgt
                if wsum > 0:
                    sh[i] = acc / wsum
                    new[i] = True
    filled = new

out_dir = os.path.join(C.ROOT, "out", "probetest")
sh.astype("<f4").tofile(os.path.join(out_dir, STEM + ".bin"))
raw.astype("<f4").tofile(os.path.join(out_dir, STEM + "_raw.bin"))
if sheet is not None:
    s8 = np.clip(sheet * 2.0, 0, 1) ** (1 / 2.2)
    img = bpy.data.images.new("sheet", sheet.shape[1], sheet.shape[0], alpha=False)
    rgba = np.concatenate([s8[::-1], np.ones(s8.shape[:2] + (1,), dtype=np.float32)], axis=-1)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    img.filepath_raw = os.path.join(out_dir, STEM + "_captures.png")
    img.file_format = "PNG"
    img.save()

meta = {
    "format": "float32 LE [probe][coef 0..8][rgb]; radiance L2 SH, three.js basis order/signs; probes in texture order ix + iy*nx + iz*nx*ny (three space)",
    "grid": {"center": G["center"], "size": G["size"], "res": res},
    "probes": N, "baked": n_bake, "bytes_per_probe": 108,
    "capture": {"w": W, "h": H, "spp": SPP, "denoise": DENOISE, "seed": SEED, "max_bounces": sc.cycles.max_bounces,
                "clamp_indirect": sc.cycles.sample_clamp_indirect},
    "valid": [bool(v) for v in valid], "invalid_count": int((~valid[:n_bake]).sum()), "bf_threshold": BF_INVALID, "dilation_rounds": rounds,
    "backface": [round(float(v), 4) for v in bf], "dist_min": [round(float(v), 3) for v in dmin], "dist_mean": [round(float(v), 3) for v in dmean],
    "scene_tris": info["tris"],
    "timing": {"build_s": round(t_built - t_start, 2), "render_s": round(t_render, 2), "project_s": round(t_project, 2),
               "s_per_probe_render_total": round(t_render / n_bake, 4), "s_per_probe_steady_median": round(float(np.median(steps)), 4),
               "first_frame_s": round(stamps[0] - (t0 - t_project) + 0.0, 2) if False else None,
               "s_per_probe_all_in": round((time.time() - t_start) / n_bake, 4)},
}
with open(os.path.join(out_dir, STEM + ".json"), "w") as f:
    json.dump(meta, f)
print("BAKE " + json.dumps({k: meta[k] for k in ("grid", "probes", "baked", "invalid_count", "dilation_rounds", "scene_tris", "timing", "capture")}))
print("BAKE bf>thr at", [i for i in range(n_bake) if not valid[i]])
