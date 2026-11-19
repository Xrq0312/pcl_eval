"""HPNet ABCParts HDF5 -> pcl_eval unified sample format.

Verified against HPNet's ``dataloader/ABCDataset.py`` and
``utils/process_abc.py``. Each preprocessed object is one HDF5 file with:

  points (N,3), normals (N,3), labels (N,), prim (N,), T_param (N,22)

Primitive IDs follow ParSeNet: 1 plane, 3 cone, 4 cylinder, 5 sphere; all
other IDs are spline/extra surfaces. T_param blocks are sphere [0:4], plane
[4:8], cylinder [8:15], cone [15:22]. Parameters are repeated per point.

Unlike HPNet training, this adapter does not discard instances with <=100
points: those instances are essential for small-primitive evaluation. The
HPNet preprocessing did not fit parameters for such instances, so their
``has_params`` value is false when the corresponding parameter block is zero.
"""
import argparse
import csv
import hashlib
import json
import os

import h5py
import numpy as np


TYPE_ID = {
    0: "closed_spline", 1: "plane", 2: "open_spline", 3: "cone",
    4: "cylinder", 5: "sphere", 6: "closed_spline",
    7: "closed_spline", 8: "open_spline", 9: "closed_spline",
}
TARGET = {"plane", "sphere", "cylinder"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def surface_distances(points, primitive):
    p = np.asarray(points, dtype=float)
    if primitive["type"] == "plane":
        n = unit(primitive["normal"])
        return np.abs(p @ n - primitive["d"])
    if primitive["type"] == "sphere":
        return np.abs(np.linalg.norm(p - primitive["center"], axis=1)
                      - primitive["radius"])
    if primitive["type"] == "cylinder":
        axis = unit(primitive["axis_direction"])
        delta = p - primitive["axis_point"]
        radial = delta - np.outer(delta @ axis, axis)
        return np.abs(np.linalg.norm(radial, axis=1) - primitive["radius"])
    return np.full(len(points), np.nan)


def parameter_block(type_name, rows):
    median = np.median(np.asarray(rows, dtype=float), axis=0)
    out = {}
    if type_name == "sphere":
        out = {"center": median[0:3].tolist(), "radius": float(median[3])}
        raw = median[0:4]
    elif type_name == "plane":
        out = {"normal": unit(median[4:7]).tolist(), "d": float(median[7])}
        raw = median[4:8]
    elif type_name == "cylinder":
        out = {
            "axis_direction": unit(median[8:11]).tolist(),
            "axis_point": median[11:14].tolist(),
            "radius": float(median[14]),
        }
        raw = median[8:15]
    elif type_name == "cone":
        out = {
            "axis_direction": unit(median[15:18]).tolist(),
            "apex": median[18:21].tolist(),
            "half_angle": float(median[21]),
        }
        raw = median[15:22]
    else:
        return {}, False
    finite = np.isfinite(raw).all()
    nonzero = np.linalg.norm(raw) > 1e-12
    radius_ok = type_name not in {"sphere", "cylinder"} or out["radius"] > 0
    direction_ok = type_name not in {"plane", "cylinder", "cone"} or np.linalg.norm(
        out["normal"] if type_name == "plane" else out["axis_direction"]) > 0.9
    return out, bool(finite and nonzero and radius_ok and direction_ok)


def convert_one(h5_path, out_dir):
    with h5py.File(h5_path, "r") as handle:
        required = {"points", "normals", "labels", "prim", "T_param"}
        missing = required - set(handle.keys())
        if missing:
            raise RuntimeError(f"{h5_path}: missing fields {sorted(missing)}")
        points = np.asarray(handle["points"][()], dtype=float)
        normals = np.asarray(handle["normals"][()], dtype=float)
        labels = np.asarray(handle["labels"][()], dtype=int).reshape(-1)
        types = np.asarray(handle["prim"][()], dtype=int).reshape(-1)
        params = np.asarray(handle["T_param"][()], dtype=float)
    n = len(points)
    if points.shape != (n, 3) or normals.shape != (n, 3):
        raise RuntimeError(f"{h5_path}: points/normals must be Nx3")
    if len(labels) != n or len(types) != n or params.shape != (n, 22):
        raise RuntimeError(f"{h5_path}: inconsistent array lengths or T_param shape")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise RuntimeError(f"{h5_path}: non-finite point or normal")

    primitives = []
    point_type = np.full(n, "none", dtype=object)
    validation = []
    for instance_id in sorted(np.unique(labels).tolist()):
        mask = labels == instance_id
        type_values, counts = np.unique(types[mask], return_counts=True)
        type_id = int(type_values[np.argmax(counts)])
        if len(type_values) != 1:
            raise RuntimeError(
                f"{h5_path}: instance {instance_id} has mixed types {type_values}")
        type_name = TYPE_ID.get(type_id, f"unknown_{type_id}")
        point_type[mask] = type_name
        values, has_params = parameter_block(type_name, params[mask])
        primitive = {
            "instance_id": int(instance_id), "type": type_name,
            "source_type_id": type_id, "is_target": type_name in TARGET,
            "has_params": has_params,
            "parameter_source": "HPNet T_param (fitted during preprocessing)",
            **values,
        }
        if has_params and type_name in TARGET:
            distances = surface_distances(points[mask], primitive)
            primitive["source_param_p95_residual"] = float(
                np.percentile(distances, 95))
            validation.append(primitive["source_param_p95_residual"])
        primitives.append(primitive)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "points.txt"), "w") as stream:
        for x, y, z in points:
            stream.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
    with open(os.path.join(out_dir, "gt_labels.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "instance_id", "primitive_type"])
        for i, (instance_id, type_name) in enumerate(zip(labels, point_type)):
            writer.writerow([i, int(instance_id), type_name])
    with open(os.path.join(out_dir, "gt_primitives.json"), "w") as stream:
        json.dump({"primitives": primitives}, stream, indent=2)
    with open(os.path.join(out_dir, "metadata.json"), "w") as stream:
        json.dump({
            "dataset": "abcparts_hpnet", "source_file": os.path.basename(h5_path),
            "source_sha256": sha256_file(h5_path), "num_points": n,
            "num_instances": len(primitives),
            "type_counts": {name: sum(p["type"] == name for p in primitives)
                            for name in sorted({p["type"] for p in primitives})},
            "parameter_gt_note": (
                "T_param contains HPNet preprocessing fits, not original CAD "
                "analytic parameters; <=100-point instances may have no params."),
            "max_source_param_p95_residual": max(validation) if validation else None,
            "small_instances_preserved": True,
        }, stream, indent=2)
    return {"sample": os.path.splitext(os.path.basename(h5_path))[0],
            "points": n, "instances": len(primitives)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HDF5 file or directory")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if os.path.isdir(args.input):
        files = sorted(os.path.join(args.input, name)
                       for name in os.listdir(args.input) if name.endswith(".h5"))
    else:
        files = [args.input]
    if args.limit > 0:
        files = files[:args.limit]
    report = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        result = convert_one(path, os.path.join(args.out_root, name))
        report.append(result)
        print(f"{name}: {result['points']} points, {result['instances']} instances")
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "adapt_report.json"), "w") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()

