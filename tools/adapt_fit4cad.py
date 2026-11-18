"""Convert an official Fit4CAD test subset to the unified evaluation format.

Fit4CAD may assign boundary points to more than one primitive.  Such points
are deliberately labelled instance -1 / type ``none`` instead of resolving
the ambiguity by primitive order.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


POINT_RE = re.compile(
    r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]")
MEMBERS_RE = re.compile(r"Primitive(\d+)\s*:?=\s*\[([^\]]*)\]\s*;?")
MODEL_RE = re.compile(
    r"Primitive(\d+)\s*:?=\s*\[([A-Za-z]+)\s*,\s*\[([^\]]*)\]\s*\]\s*;?")
TYPE_MAP = {"plane": "plane", "sphere": "sphere", "cylinder": "cylinder",
            "cone": "cone", "torus": "torus"}


def numbers(text):
    return [float(value) for value in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)]


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def normalized(vector):
    length = norm(vector)
    if not math.isfinite(length) or length <= 1e-12:
        return None
    return [value / length for value in vector]


def parse_models(parametric_path, implicit_path):
    parametric = {}
    for match in MODEL_RE.finditer(parametric_path.read_text(encoding="utf-8")):
        parametric[int(match.group(1))] = (
            TYPE_MAP.get(match.group(2).lower(), match.group(2).lower()),
            numbers(match.group(3)))
    implicit = {}
    for match in MODEL_RE.finditer(implicit_path.read_text(encoding="utf-8")):
        implicit[int(match.group(1))] = TYPE_MAP.get(
            match.group(2).lower(), match.group(2).lower())
    models = []
    for primitive_id in sorted(set(parametric) | set(implicit)):
        primitive_type, coefficients = parametric.get(
            primitive_id, (implicit[primitive_id], []))
        model = {"instance_id": primitive_id - 1, "type": primitive_type,
                 "is_target": primitive_type in {"plane", "sphere", "cylinder"},
                 "has_params": False, "source_primitive_id": primitive_id}
        if primitive_type == "plane" and len(coefficients) == 9:
            normal = normalized(cross(coefficients[0:3], coefficients[3:6]))
            if normal:
                model.update({"point": coefficients[6:9], "normal": normal,
                              "has_params": True})
        elif primitive_type == "cylinder" and len(coefficients) == 12:
            axis = normalized(coefficients[6:9])
            radius = 0.5 * (norm(coefficients[0:3]) + norm(coefficients[3:6]))
            if axis and math.isfinite(radius) and radius > 0:
                model.update({"axis_point": coefficients[9:12],
                              "axis_direction": axis, "radius": radius,
                              "has_params": True})
        elif primitive_type == "sphere" and len(coefficients) == 12:
            radius = sum(norm(coefficients[offset:offset + 3])
                         for offset in (0, 3, 6)) / 3.0
            if math.isfinite(radius) and radius > 0:
                model.update({"center": coefficients[9:12], "radius": radius,
                              "has_params": True})
        models.append(model)
    return models


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_one(source_root, output_root, pc_id, source_commit):
    stem = f"PC{pc_id}"
    point_path = source_root / "pointCloud" / f"{stem}.txt"
    member_path = source_root / "pointCloud_primitives" / f"{stem}_primitives.txt"
    parameter_path = source_root / "pointCloud_parametric" / f"{stem}_parametric.txt"
    implicit_path = source_root / "pointCloud_implicit" / f"{stem}_implicit.txt"
    paths = (point_path, member_path, parameter_path, implicit_path)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"incomplete Fit4CAD model {stem}")
    points = [tuple(map(float, match.groups()))
              for match in POINT_RE.finditer(point_path.read_text(encoding="utf-8"))]
    if not points:
        raise ValueError(f"no points parsed from {point_path}")
    memberships = [[] for _ in points]
    primitive_ids = set()
    text = member_path.read_text(encoding="utf-8")
    for match in MEMBERS_RE.finditer(text):
        primitive_id = int(match.group(1))
        primitive_ids.add(primitive_id)
        for source_index in (int(value) for value in re.findall(r"\d+", match.group(2))):
            index = source_index - 1  # official files use MATLAB one-based indices
            if index < 0 or index >= len(points):
                raise ValueError(f"{stem}: point index {source_index} out of range")
            memberships[index].append(primitive_id)
    models = parse_models(parameter_path, implicit_path)
    model_by_source_id = {model["source_primitive_id"]: model for model in models}
    if primitive_ids != set(model_by_source_id):
        raise ValueError(f"{stem}: membership/model primitive ids differ")
    labels = []
    ambiguous = unlabeled = 0
    for assigned in memberships:
        unique = sorted(set(assigned))
        if len(unique) == 1:
            model = model_by_source_id[unique[0]]
            labels.append((model["instance_id"], model["type"]))
        else:
            labels.append((-1, "none"))
            ambiguous += len(unique) > 1
            unlabeled += not unique
    output = output_root / stem
    output.mkdir(parents=True, exist_ok=True)
    with (output / "points.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for point in points:
            handle.write("{:.12g} {:.12g} {:.12g}\n".format(*point))
    with (output / "gt_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("point_index", "instance_id", "primitive_type"))
        for index, label in enumerate(labels):
            writer.writerow((index, *label))
    with (output / "gt_primitives.json").open("w", encoding="utf-8") as handle:
        json.dump({"primitives": models}, handle, indent=2)
    xs, ys, zs = zip(*points)
    metadata = {
        "dataset": "Fit4CAD", "split": "official_test", "source_model": stem,
        "source_repository": "https://github.com/chiararomanengo/Fit4CAD",
        "source_commit": source_commit, "source_coordinate_units": "unspecified",
        "num_points": len(points), "num_primitives": len(models),
        "target_primitives": sum(model["is_target"] for model in models),
        "ambiguous_multi_primitive_points": ambiguous,
        "unlabeled_points": unlabeled,
        "label_policy": "one membership -> source primitive; zero or multiple -> none",
        "bbox": {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)],
                 "z": [min(zs), max(zs)]},
        "source_sha256": {path.name: sha256(path) for path in paths},
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ids", nargs="+", type=int,
                        default=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 35])
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    summaries = [convert_one(args.source_root, args.output_root, pc_id,
                             args.source_commit) for pc_id in args.ids]
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"selection": "12 IDs evenly spaced over official test IDs 1..35",
                   "ids": args.ids, "source_commit": args.source_commit,
                   "samples": summaries}, handle, indent=2)
    print(f"converted {len(summaries)} Fit4CAD test samples to {args.output_root}")


if __name__ == "__main__":
    main()
