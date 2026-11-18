"""Verify BAGSFit label/parameter correspondence using geometry.

For each visible instance i (ins==i), check that its noiseless 'scan' points
lie on the surface defined by prim row i -> confirms row index == instance id,
and derive the cls value -> type mapping from majority vote.
"""
import numpy as np

def parse_prims(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            t = parts[0]
            vals = [float(x) for x in parts[1:]]
            rows.append((t, vals))
    return rows

def dist_to(pts, t, v):
    if t == "Plane":
        p0, n = np.array(v[0:3]), np.array(v[3:6])
        n = n / np.linalg.norm(n)
        return np.abs((pts - p0) @ n)
    if t == "Sphere":
        c, r = np.array(v[0:3]), v[3]
        return np.abs(np.linalg.norm(pts - c, axis=1) - r)
    if t == "Cylinder":
        p0, u, r = np.array(v[0:3]), np.array(v[3:6]), v[6]
        u = u / np.linalg.norm(u)
        d = pts - p0
        perp = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
        return np.abs(perp - r)
    if t == "Cone":
        apex, u, half = np.array(v[0:3]), np.array(v[3:6]), v[6]
        u = u / np.linalg.norm(u)
        d = pts - apex
        proj = d @ u
        perp = np.sqrt(np.maximum(np.sum(d*d, axis=1) - proj*proj, 0))
        return np.abs(perp - proj * np.tan(half))
    return np.full(len(pts), np.nan)

for scene in ["000", "001"]:
    for frame in ["00000", "00001"]:
        base = f"D:/download/pcl_eval/data/bagsfit_data/extracted/{scene}/{frame}"
        z = np.load(base + ".npz", allow_pickle=True)
        scan = z["scan"].reshape(3, -1).T      # (Npix, 3)
        ins = z["ins"].ravel()
        cls = z["cls"].ravel()
        prims = parse_prims(base + ".prim")
        print(f"=== {scene}/{frame}: {len(prims)} prim rows, "
              f"visible ins {sorted(set(ins[ins>=0].tolist()))}")
        for i in sorted(set(ins[ins >= 0].tolist())):
            mask = ins == i
            pts = scan[mask]
            if i >= len(prims):
                print(f"  ins {i}: NO prim row (>{len(prims)}) cls_maj="
                      f"{np.bincount(cls[mask][cls[mask]>=0], minlength=4).argmax()}")
                continue
            t, v = prims[i]
            d = dist_to(pts, t, v)
            cls_maj = np.bincount(cls[mask][cls[mask] >= 0],
                                  minlength=4).argmax()
            print(f"  ins {i:2d} type={t:8s} n={mask.sum():6d} "
                  f"dist_med={np.median(d):.4f} dist_p95={np.percentile(d,95):.4f} "
                  f"cls_maj={cls_maj} cls_frac={np.mean(cls[mask]==cls_maj):.3f}")
