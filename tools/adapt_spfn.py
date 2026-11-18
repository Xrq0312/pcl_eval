"""SPFN TraceParts (h5) -> unified sample format adapter.

Source format (verified from SPFN repo spfn/lib/dataset.py and fitters):
  - one .h5 per CAD shape; official split lists in train/test_models.csv
  - noisy_points (N,3)  : noisy point cloud (8k points)
  - gt_points   (N,3)   : noiseless points (same order? NO - dataset.py
                          shuffles at load; we do NOT shuffle, we keep the
                          raw h5 order which is the supervision order)
  - gt_normals  (N,3)
  - gt_labels   (N,)    : per-point instance id, 0-based, consecutive
  - {name}_soup_{i} groups: per-instance gt_points + attrs['meta'] =
    a pickled dict with keys 'type' and parameters:
      plane:   location_x/y/z, axis_x/y/z        (n.p = n.location)
      sphere:  location_x/y/z, radius
      cylinder:location_x/y/z, axis_x/y/z, radius
      cone:    location_x/y/z, axis_x/y/z, plus angle fields
  - type id order in configs is [sphere, plane, cylinder, cone] but we never
    rely on numeric ids; the pickled meta carries the type string.

Target classes: plane, sphere, cylinder. cone is kept as non-target.
Instance numbering in gt_labels is preserved as instance_id.
"""
import argparse
import csv
import json
import os
import pickle
import sys

import h5py
import numpy as np

NON_TARGET = {"cone"}


def read_split(csv_path, h5_root, limit):
    with open(csv_path, newline="") as f:
        rows = [r[0] for r in csv.reader(f) if r and r[0].strip()]
    base = os.path.abspath(h5_root) if h5_root else os.path.dirname(os.path.abspath(csv_path))
    out = []
    for r in rows:
        p = r if os.path.isabs(r) else os.path.join(base, r)
        out.append(p)
    return out[:limit] if limit > 0 else out


def convert_one(h5_path, out_dir, use_noisy=True):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(h5_path))[0]
    with h5py.File(h5_path, "r") as f:
        pts = f["noisy_points"][()] if use_noisy else f["gt_points"][()]
        gt_pts = f["gt_points"][()]
        labels = f["gt_labels"][()]
        keys = {}
        import re
        for k in f.keys():
            m = re.match(r"(.*)_soup_([0-9]+)$", k)
            if m:
                keys[int(m.group(2))] = k
        inst_params = {}
        for i in sorted(keys):
            meta = pickle.loads(f[keys[i]].attrs["meta"])
            inst_params[i] = meta
    pts = np.asarray(pts, dtype=float)
    labels = np.asarray(labels).astype(int)
    if len(pts) != len(labels):
        raise RuntimeError(f"{h5_path}: points {len(pts)} != labels {len(labels)}")

    # primitives JSON
    prims = []
    for i, meta in sorted(inst_params.items()):
        t = meta.get("type", "unknown")
        p = {"instance_id": int(i), "type": t,
             "is_target": t not in NON_TARGET,
             "has_params": True, "source": "spfn_meta_pickle"}
        loc = [float(meta.get(f"location_{a}", 0.0)) for a in "xyz"]
        if t == "plane":
            p["normal"] = [float(meta.get(f"axis_{a}", 0.0)) for a in "xyz"]
            p["point"] = loc          # n.(q - point) = 0
        elif t == "sphere":
            p["center"] = loc
            p["radius"] = float(meta.get("radius", 0.0))
        elif t == "cylinder":
            p["axis_point"] = loc
            p["axis_direction"] = [float(meta.get(f"axis_{a}", 0.0)) for a in "xyz"]
            p["radius"] = float(meta.get("radius", 0.0))
        else:
            p["location"] = loc       # cone / others: keep raw, non-target
        prims.append(p)

    type_of = {p["instance_id"]: p["type"] for p in prims}
    with open(os.path.join(out_dir, "points.txt"), "w") as f:
        for x, y, z in pts:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    with open(os.path.join(out_dir, "gt_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point_index", "instance_id", "primitive_type"])
        for i, lab in enumerate(labels):
            t = type_of.get(int(lab), "none")
            w.writerow([i, int(lab), t])
    with open(os.path.join(out_dir, "gt_primitives.json"), "w") as f:
        json.dump({"primitives": prims}, f, indent=2)
    ranges = {a: [float(pts[:, k].min()), float(pts[:, k].max())]
              for k, a in enumerate("xyz")}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({
            "dataset": "spfn_traceparts",
            "source_file": os.path.basename(h5_path),
            "variant": "noisy" if use_noisy else "clean",
            "num_points": int(len(pts)),
            "num_instances": len(prims),
            "type_counts": {t: sum(1 for p in prims if p["type"] == t)
                            for t in sorted({p["type"] for p in prims})},
            "bbox": ranges,
            "units": "raw CAD units (unconverted)",
            "notes": "type ids taken from pickled meta dicts, not numeric ids",
        }, f, indent=2)
    return name, len(pts), prims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", help="single h5 file")
    ap.add_argument("--split-csv", help="official split csv (e.g. test_models.csv)")
    ap.add_argument("--h5-root", help="directory containing split-listed h5 files")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--clean", action="store_true", help="use gt_points instead of noisy")
    args = ap.parse_args()

    files = []
    if args.h5:
        files = [args.h5]
    elif args.split_csv:
        files = read_split(args.split_csv, args.h5_root, args.limit)
    else:
        ap.error("need --h5 or --split-csv")
    report = []
    for p in files:
        if not os.path.exists(p):
            print("MISSING", p)
            continue
        name, n, prims = convert_one(p, os.path.join(args.out_root, name_safe(p)),
                                     use_noisy=not args.clean)
        counts = {}
        for pr in prims:
            counts[pr["type"]] = counts.get(pr["type"], 0) + 1
        report.append((name, n, counts))
        print(f"{name}: {n} points, {len(prims)} primitives {counts}")
    with open(os.path.join(args.out_root, "adapt_report.json"), "w") as f:
        json.dump([{"sample": n, "points": p, "types": t} for n, p, t in report],
                  f, indent=2)


def name_safe(p):
    return os.path.splitext(os.path.basename(p))[0]


if __name__ == "__main__":
    main()
