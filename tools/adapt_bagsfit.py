"""BAGSFit TEST-20s -> unified sample format adapter.

Verified format facts (see verify_bagsfit.py, all 20 scenes checked):
  npz: depth-image grid (480x640); fields data(3,H,W noisy xyz), scan(3,H,W
       noiseless), ins(H,W instance id, -1 background), cls(H,W class id)
       cls: 0=plane 1=sphere 2=cylinder 3=cone (-1 background)
  prim: one row per scene primitive "Type x y z [nx ny nz] [r|theta]"
       row index == instance id (0-based). Rows 20/21 may be referenced by
       ins without having parameters (large support planes) -> has_params
       False, type taken from cls.
  KNOWN SOURCE BUG: Sphere center y-coordinate is sign-flipped in the .prim
       files. Verified on 36/36 sphere instances: flipping y makes the
       noiseless scan points lie on the sphere (median dist < 0.005 m vs
       0.7-1.9 m without the fix). We correct it here and record the fix in
       metadata. Plane/cone/cylinder parameters are consistent as-is.

Output per frame: points.txt / gt_labels.csv / gt_primitives.json / metadata.json
Point order: raster order of the depth image, background pixels removed.
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

CLS_TYPE = {0: "plane", 1: "sphere", 2: "cylinder", 3: "cone"}
TYPE_BY_LEN = {"Plane": 6, "Sphere": 4, "Cylinder": 7, "Cone": 7}


def parse_prims(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            t = parts[0]
            vals = [float(x) for x in parts[1:]]
            assert len(vals) == TYPE_BY_LEN[t], f"unexpected prim row: {line}"
            rows.append((t, vals))
    return rows


def convert_frame(npz_path, prim_path, out_dir, use_noisy=True):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(npz_path, allow_pickle=True)
    if use_noisy:
        pts = z["data"].reshape(3, -1).T
    else:
        pts = z["scan"].reshape(3, -1).T
    ins = z["ins"].ravel()
    cls = z["cls"].ravel()
    prims = parse_prims(prim_path)

    keep = ins >= 0
    pts, ins, cls = pts[keep], ins[keep], cls[keep]

    # sanity: labels and points must stay 1:1
    assert len(pts) == len(ins) == len(cls)

    prim_by_inst = {}
    for i, (t, v) in enumerate(prims):
        prim_by_inst[i] = (t, v)

    prims_out = []
    for i in sorted(set(ins.tolist())):
        sel = ins == i
        c_vals = cls[sel]
        c_major = Counter(c_vals.tolist()).most_common(1)[0][0]
        # instances 20/21 (support surfaces) carry cls=-1: unclassified,
        # no prim row. Type is 'none' (non-target), points are KEPT.
        t_cls = CLS_TYPE.get(int(c_major), "none") if c_major >= 0 else "none"
        p = {"instance_id": int(i), "is_target": t_cls in
             ("plane", "sphere", "cylinder"),
             "type": t_cls, "source": "bagsfit_prim"}
        if i in prim_by_inst:
            t, v = prim_by_inst[i]
            # cross-check prim type against cls
            expected = {"Plane": "plane", "Sphere": "sphere",
                       "Cylinder": "cylinder", "Cone": "cone"}[t]
            if expected != t_cls:
                raise RuntimeError(f"cls/prim type mismatch inst {i}")
            p["has_params"] = True
            if t == "Plane":
                p["point"] = v[0:3]
                p["normal"] = v[3:6]
            elif t == "Sphere":
                p["center"] = [v[0], -v[1], v[2]]   # y sign fix (see header)
                p["center_raw"] = v[0:3]
                p["radius"] = v[3]
                p["param_fix"] = "sphere center y sign-flipped (verified vs scan)"
            elif t == "Cylinder":
                p["axis_point"] = v[0:3]
                p["axis_direction"] = v[3:6]
                p["radius"] = v[6]
            elif t == "Cone":
                p["apex"] = v[0:3]
                p["axis_direction"] = v[3:6]
                p["half_angle"] = v[6]
        else:
            # referenced by ins but no prim row (support planes): type from
            # cls only, parameters unknown -> must not be used as param GT
            p["has_params"] = False
            p["param_fix"] = "no prim row in source; type from cls only"
        prims_out.append(p)

    with open(os.path.join(out_dir, "points.txt"), "w") as f:
        for x, y, zz in pts:
            f.write(f"{x:.8f} {y:.8f} {zz:.8f}\n")
    with open(os.path.join(out_dir, "gt_labels.csv"), "w", newline="") as f:
        f.write("point_index,instance_id,primitive_type\n")
        for i2, (lab, c) in enumerate(zip(ins, cls)):
            t = CLS_TYPE.get(int(c), "none") if int(c) >= 0 else "none"
            f.write(f"{i2},{int(lab)},{t}\n")
    with open(os.path.join(out_dir, "gt_primitives.json"), "w") as f:
        json.dump({"primitives": prims_out}, f, indent=2)
    type_counts = Counter(p["type"] for p in prims_out)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "dataset": "bagsfit_test20s",
            "source_file": os.path.basename(npz_path),
            "variant": "noisy" if use_noisy else "clean",
            "num_points": int(len(pts)),
            "num_instances": len(prims_out),
            "type_counts": dict(type_counts),
            "bbox": {a: [float(pts[:, k].min()), float(pts[:, k].max())]
                     for k, a in enumerate("xyz")},
            "units": "metres (Blensor simulated range camera)",
            "param_fixes": ["sphere center y sign-flipped to match scan "
                           "geometry (source bug, verified 36/36 instances)"],
            "no_param_instances": [p["instance_id"] for p in prims_out
                                   if not p["has_params"]],
        }, f, indent=2)
    z.close()
    return {"num_points": int(len(pts)), "primitives": prims_out,
            "bbox": [float(pts.min()), float(pts.max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted TEST-20s dir")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--frames", default="00000",
                    help="comma list of frame ids per scene, e.g. 00000,00018")
    ap.add_argument("--scenes", default="", help="comma list, empty = all")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    frames = args.frames.split(",")
    scenes = args.scenes.split(",") if args.scenes else None
    report = []
    for sc in sorted(os.listdir(args.root)):
        if not os.path.isdir(os.path.join(args.root, sc)):
            continue
        if scenes and sc not in scenes:
            continue
        for fr in frames:
            npz = os.path.join(args.root, sc, fr + ".npz")
            prim = os.path.join(args.root, sc, fr + ".prim")
            if not os.path.exists(npz):
                continue
            out = os.path.join(args.out_root, f"{sc}_{fr}")
            r = convert_frame(npz, prim, out, use_noisy=not args.clean)
            tc = Counter(p["type"] for p in r["primitives"])
            report.append({"sample": f"{sc}_{fr}", **{k: v for k, v in
                                                       r.items()
                                                       if k != "primitives"},
                           "type_counts": dict(tc)})
            print(f"{sc}_{fr}: {r['num_points']} pts, {len(tc)} instances "
                  f"{dict(tc)}")
    with open(os.path.join(args.out_root, "adapt_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("report ->", os.path.join(args.out_root, "adapt_report.json"))


if __name__ == "__main__":
    main()
