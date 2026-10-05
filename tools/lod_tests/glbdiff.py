#!/usr/bin/env python3
"""Compare two GLBs attribute by attribute (needs numpy: run with Blender's python). usage: glbdiff.py a.glb b.glb
geohash(): order-independent geometry hash used to prove lod 0 unchanged; tri_diff(): the same with a tolerance."""
import sys, json, struct
import numpy as np
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
def load(p):
    b = open(p, "rb").read()
    jl = struct.unpack_from("<I", b, 12)[0]
    j = json.loads(b[20:20 + jl])
    bin_ = b[20 + jl + 8:]
    def acc(i):
        a = j["accessors"][i]; bv = j["bufferViews"][a["bufferView"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = a["count"] * NC[a["type"]]
        return np.frombuffer(bin_, dtype=CT[a["componentType"]], count=n, offset=off).reshape(a["count"], -1)
    out = {}
    for mi, m in enumerate(j["meshes"]):
        for pi, pr in enumerate(m["primitives"]):
            key = f"{m.get('name')}/{j['materials'][pr['material']]['name'] if 'material' in pr else pi}"
            d = {k: acc(v) for k, v in pr["attributes"].items()}
            d["idx"] = acc(pr["indices"])
            out[key] = d
    return out, j
def geohash(p, q=1e-4):
    """order-independent geometry hash: per primitive the sorted set of quantised triangle corner positions"""
    import hashlib
    out, j = load(p)
    h = hashlib.sha1(); tris = 0
    for k in sorted(out):
        d = out[k]; P = np.round(d["POSITION"].astype(np.float64) / q).astype(np.int64)
        T = P[d["idx"].ravel()].reshape(-1, 9)
        T = T[np.lexsort(T.T[::-1])]
        h.update(k.encode()); h.update(T.tobytes()); tris += len(T)
    return h.hexdigest()[:16], tris
def tri_diff(a, b, q=1e-3):
    """Triangles of a that have no partner in b (and back), per material, corner positions rounded to q metres.
    For assets that are not reproducible run to run: compare the count against two runs of the SAME code."""
    A, B = load(a)[0], load(b)[0]
    res = {}
    for k in sorted(set(A) | set(B)):
        s = []
        for M in (A, B):
            d = M.get(k)
            T = np.zeros((0, 9), np.int64) if d is None else np.round(d["POSITION"].astype(np.float64)[d["idx"].ravel()].reshape(-1, 9) / q).astype(np.int64)
            s.append((len(T), set(r.tobytes() for r in T)))
        n = len(s[0][1] ^ s[1][1])
        if n or s[0][0] != s[1][0]:
            res[k.split("/", 1)[1]] = (n // 2, s[0][0], s[1][0])
    return res
if __name__ == "__main__":
    A, ja = load(sys.argv[1]); B, jb = load(sys.argv[2])
    print("prims", len(A), len(B), "nodes equal", ja["nodes"] == jb["nodes"], "materials equal", ja["materials"] == jb["materials"])
    for k in A:
        if k not in B: print("missing in b", k); continue
        for an in A[k]:
            x, y = A[k][an], B[k].get(an)
            if y is None or x.shape != y.shape: print(k, an, "shape", x.shape, None if y is None else y.shape); continue
            if not np.array_equal(x, y):
                d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                print(f"{k:40} {an:10} differs: {int((d.max(1) > 0).sum())}/{len(x)} rows, max abs {d.max():.3g}")
    print("geohash", geohash(sys.argv[1]), geohash(sys.argv[2]))
    print("triangles without a partner at 1 mm (material: n, tris a, tris b):", tri_diff(sys.argv[1], sys.argv[2]))
