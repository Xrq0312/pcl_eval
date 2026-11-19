"""BA Primitive Fitting CSV -> unified samples with explicit derived pseudo-GT.

The upstream files contain xyz, normals and a constant zero label, but no
primitive parameters or part/instance masks. This adapter therefore refuses to
run unless --allow-derived-gt is supplied. It derives references only from the
paired clean file and transfers the row-aligned mask to noisy variants:

* sphere: all points, algebraic sphere fit;
* cylinder: side points selected from clean normals; caps remain type ``none``;
* box: six faces grouped by their clean oriented normals and fitted as planes.

These outputs are useful for controlled noise ablations, never as official BA
instance ground truth.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path):
    values = np.loadtxt(path, delimiter=",")
    if values.shape != (1000, 7) or not np.isfinite(values).all():
        raise RuntimeError(f"{path}: expected finite 1000x7 CSV")
    if not np.allclose(values[:, 6], values[0, 6]):
        raise RuntimeError(f"{path}: unexpected nonconstant source label column")
    normals = values[:, 3:6]
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths < 1e-8):
        raise RuntimeError(f"{path}: zero normal")
    return values[:, :3], normals / lengths[:, None]


def canonical_rows(normals):
    result = normals.copy()
    for row in result:
        pivot = np.flatnonzero(np.abs(row) > 1e-8)
        if len(pivot) and row[pivot[0]] < 0:
            row *= -1
    return result


def fit_plane(points, normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    return {"point": points.mean(axis=0).tolist(), "normal": normal.tolist()}


def fit_sphere(points):
    matrix = np.column_stack((2.0 * points, np.ones(len(points))))
    solution, *_ = np.linalg.lstsq(matrix, np.sum(points * points, axis=1), rcond=None)
    center = solution[:3]
    radius = np.sqrt(max(0.0, solution[3] + center @ center))
    return {"center": center.tolist(), "radius": float(radius)}


def fit_cylinder(points, normals):
    scatter = normals.T @ normals
    _, vectors = np.linalg.eigh(scatter)
    axis = vectors[:, 0]
    pivot = np.flatnonzero(np.abs(axis) > 1e-8)
    if len(pivot) and axis[pivot[0]] < 0:
        axis *= -1
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.8 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    xy = np.column_stack((points @ first, points @ second))
    matrix = np.column_stack((2.0 * xy, np.ones(len(xy))))
    solution, *_ = np.linalg.lstsq(matrix, np.sum(xy * xy, axis=1), rcond=None)
    radius = np.sqrt(max(0.0, solution[2] + solution[:2] @ solution[:2]))
    origin = solution[0] * first + solution[1] * second
    origin -= axis * (origin @ axis)
    return {"axis_point": origin.tolist(), "axis_direction": axis.tolist(),
            "radius": float(radius)}


def derive_clean(clean_points, clean_normals, category):
    if category == "spheres":
        return np.zeros(len(clean_points), dtype=int), [("sphere", fit_sphere(clean_points))]
    if category == "cylinders":
        canonical = canonical_rows(clean_normals)
        rounded = np.round(canonical, 6)
        unique, counts = np.unique(rounded, axis=0, return_counts=True)
        cap_axis = unique[np.argmax(counts)]
        cap_axis /= np.linalg.norm(cap_axis)
        side = np.abs(clean_normals @ cap_axis) < 0.5
        if side.sum() < 100 or (~side).sum() < 20:
            raise RuntimeError("cannot separate cylinder side from caps")
        labels = np.full(len(clean_points), -1, dtype=int)
        labels[side] = 0
        return labels, [("cylinder", fit_cylinder(clean_points[side], clean_normals[side]))]
    if category == "boxes":
        rounded = np.round(clean_normals, 5)
        unique, inverse, counts = np.unique(rounded, axis=0, return_inverse=True,
                                            return_counts=True)
        order = np.argsort(-counts)
        if len(unique) != 6 or counts.min() < 100:
            raise RuntimeError(f"expected six box faces, got {len(unique)} normal groups")
        remap = np.empty(6, dtype=int)
        remap[order] = np.arange(6)
        labels = remap[inverse]
        primitives = []
        for output_id, source_id in enumerate(order):
            mask = inverse == source_id
            primitives.append(("plane", fit_plane(clean_points[mask], unique[source_id])))
        return labels, primitives
    raise RuntimeError(f"unsupported BA category: {category}")


def convert_one(source_path, clean_path, output_dir, allow_derived_gt=False):
    if not allow_derived_gt:
        raise RuntimeError("BA has no official instance/parameter GT; pass --allow-derived-gt")
    source_path, clean_path = Path(source_path), Path(clean_path)
    category = source_path.parent.name
    points, _ = load_csv(source_path)
    clean_points, clean_normals = load_csv(clean_path)
    if points.shape != clean_points.shape:
        raise RuntimeError("noisy and clean files are not row aligned")
    labels, fitted = derive_clean(clean_points, clean_normals, category)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "points.txt", points, fmt="%.10g")
    with open(output_dir / "gt_labels.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "instance_id", "primitive_type"])
        types = {index: primitive[0] for index, primitive in enumerate(fitted)}
        for index, instance in enumerate(labels):
            writer.writerow([index, int(instance), types.get(int(instance), "none")])
    primitives = []
    for instance, (primitive_type, parameters) in enumerate(fitted):
        primitive = {"instance_id": instance, "type": primitive_type,
                     "is_target": True, "has_params": True,
                     "source": "derived_from_paired_clean_ba_csv",
                     "annotation_status": "derived_pseudo_gt"}
        primitive.update(parameters)
        primitives.append(primitive)
    with open(output_dir / "gt_primitives.json", "w", encoding="utf-8") as stream:
        json.dump({"annotation_status": "derived_pseudo_gt", "primitives": primitives},
                  stream, indent=2)
    metadata = {
        "dataset": "ba_primitive_fitting", "source_file": str(source_path),
        "clean_reference_file": str(clean_path), "category": category,
        "variant": source_path.parents[1].name, "num_points": len(points),
        "annotation_status": "derived_pseudo_gt",
        "official_point_instance_gt": False, "official_parameter_gt": False,
        "source_sha256": sha256_file(source_path),
        "clean_reference_sha256": sha256_file(clean_path),
        "warning": "Do not report derived masks/parameters as official BA ground truth."
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return {"points": len(points), "instances": len(primitives),
            "assigned": int(np.sum(labels >= 0)), "category": category}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="BA repository or shapes directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["clean", "error", "error_double"],
                        default="error")
    parser.add_argument("--categories", default="boxes,spheres,cylinders")
    parser.add_argument("--limit", type=int, default=0, help="limit per category")
    parser.add_argument("--allow-derived-gt", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    shapes = root / "shapes" if (root / "shapes").is_dir() else root
    converted = []
    for category in [value.strip() for value in args.categories.split(",") if value.strip()]:
        files = sorted((shapes / args.variant / category).glob("*.csv"))
        if args.limit:
            files = files[:args.limit]
        for source in files:
            clean = shapes / "clean" / category / source.name
            name = f"{args.variant}_{category[:-1]}_{source.stem}"
            converted.append(convert_one(source, clean, Path(args.output) / name,
                                         args.allow_derived_gt))
    print(json.dumps({"samples": len(converted), "details": converted}, indent=2))


if __name__ == "__main__":
    main()
