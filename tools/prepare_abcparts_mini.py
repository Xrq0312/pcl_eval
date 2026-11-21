"""Select and adapt a reproducible mini set from ParSeNet ABCParts HDF5.

The monolithic ParSeNet files contain official points, normals, instance labels,
and primitive type labels, but no analytic primitive parameters.  Parameters
written by this tool are explicitly marked as derived references fitted from
the official per-instance points/normals; they are not original CAD ground
truth and are never used to select samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


TYPE_ID = {
    0: "closed_spline", 1: "plane", 2: "open_spline", 3: "cone",
    4: "cylinder", 5: "sphere", 6: "closed_spline",
    7: "revolution", 8: "extrusion", 9: "closed_spline",
}
TARGET = {"plane", "sphere", "cylinder"}
SELECTION_VERSION = "abcparts_train_mini_stratified_v2"


def instance_rows(labels: np.ndarray, types: np.ndarray) -> list[dict]:
    rows = []
    for instance in np.unique(labels):
        mask = labels == instance
        values, counts = np.unique(types[mask], return_counts=True)
        type_id = int(values[np.argmax(counts)])
        rows.append({
            "instance_id": int(instance),
            "type_id": type_id,
            "type": TYPE_ID.get(type_id, f"unknown_{type_id}"),
            "points": int(mask.sum()),
            "mixed_type_points": int(mask.sum() - counts.max()),
        })
    return rows


def sample_stats(index: int, labels: np.ndarray, types: np.ndarray) -> dict:
    instances = instance_rows(labels, types)
    type_counts: dict[str, int] = {}
    for row in instances:
        type_counts[row["type"]] = type_counts.get(row["type"], 0) + 1
    target = [row for row in instances if row["type"] in TARGET]
    return {
        "source_index": index,
        "instances": len(instances),
        "target_instances": len(target),
        "non_target_instances": len(instances) - len(target),
        "small_target_instances_le100": sum(row["points"] <= 100 for row in target),
        "minimum_target_points": min((row["points"] for row in target), default=0),
        "type_counts": type_counts,
    }


def scan_stats(handle: h5py.File, chunk_size: int = 128) -> list[dict]:
    labels_ds = handle["labels"]
    types_ds = handle["prim"]
    result = []
    for start in range(0, labels_ds.shape[0], chunk_size):
        stop = min(labels_ds.shape[0], start + chunk_size)
        labels = labels_ds[start:stop]
        types = types_ds[start:stop]
        for offset in range(stop - start):
            result.append(sample_stats(start + offset, labels[offset], types[offset]))
        print(f"scan {stop}/{labels_ds.shape[0]}", flush=True)
    return result


def deterministic_rank(index: int, seed: int) -> int:
    value = f"{SELECTION_VERSION}:{seed}:{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def select_samples(stats: list[dict], count: int, seed: int) -> tuple[list[dict], dict]:
    eligible = [row for row in stats if row["target_instances"] > 0]
    complexities = sorted(row["instances"] for row in eligible)
    simple_max = complexities[max(0, math.floor(0.33 * len(complexities)) - 1)]
    medium_max = complexities[max(0, math.floor(0.67 * len(complexities)) - 1)]
    for row in eligible:
        row["complexity"] = ("simple" if row["instances"] <= simple_max else
                             "medium" if row["instances"] <= medium_max else "complex")
    quotas = {
        "has_plane": round(0.80 * count),
        "has_sphere": round(0.20 * count),
        "has_cylinder": round(0.50 * count),
        "has_small_target": round(0.20 * count),
        "has_non_target": round(0.50 * count),
        "is_complex": round(0.40 * count),
        "has_multiple_targets": round(0.50 * count),
    }

    def flags(row: dict) -> dict[str, bool]:
        types = row["type_counts"]
        return {
            "has_plane": types.get("plane", 0) > 0,
            "has_sphere": types.get("sphere", 0) > 0,
            "has_cylinder": types.get("cylinder", 0) > 0,
            "has_small_target": row["small_target_instances_le100"] > 0,
            "has_non_target": row["non_target_instances"] > 0,
            "is_complex": row["complexity"] == "complex",
            "has_multiple_targets": row["target_instances"] >= 4,
        }

    availability = {name: sum(flags(row)[name] for row in eligible) for name in quotas}
    quotas = {name: min(value, availability[name]) for name, value in quotas.items()}
    achieved = {name: 0 for name in quotas}
    remaining = {row["source_index"]: row for row in eligible}
    selected = []
    base = count // 5
    stratum_targets = {"simple": base, "medium": 2 * base,
                       "complex": count - 3 * base}
    schedule = []
    while any(value > 0 for value in stratum_targets.values()):
        for stratum in ("simple", "medium", "complex", "medium", "complex"):
            if stratum_targets[stratum] > 0:
                schedule.append(stratum)
                stratum_targets[stratum] -= 1
    for desired_stratum in schedule:
        best = None
        best_key = None
        for row in remaining.values():
            if row["complexity"] != desired_stratum:
                continue
            row_flags = flags(row)
            unmet_score = sum(
                (1.0 + (quotas[name] - achieved[name]) / max(1, quotas[name]))
                for name in quotas if achieved[name] < quotas[name] and row_flags[name]
            )
            diversity = 0.02 * len(row["type_counts"]) + 0.001 * row["instances"]
            key = (unmet_score + diversity,
                   -deterministic_rank(row["source_index"], seed))
            if best_key is None or key > best_key:
                best, best_key = row, key
        assert best is not None
        selected.append(best)
        for name, present in flags(best).items():
            achieved[name] += int(present)
        del remaining[best["source_index"]]
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} eligible samples for requested {count}")
    return selected, {
        "complexity_thresholds": {"simple_max_instances": simple_max,
                                  "medium_max_instances": medium_max},
        "selected_complexity_counts": {
            name: sum(row["complexity"] == name for row in selected)
            for name in ("simple", "medium", "complex")},
        "requested_quotas": quotas,
        "achieved": achieved,
        "eligible_samples": len(eligible),
        "availability": availability,
    }


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else vector


def fit_plane(points: np.ndarray, normals: np.ndarray) -> dict:
    moment = normals.T @ normals
    values, vectors = np.linalg.eigh(moment)
    normal = unit(vectors[:, int(np.argmax(values))])
    offset = float(np.median(points @ normal))
    residual = np.abs(points @ normal - offset)
    return {"normal": normal.tolist(), "d": offset,
            "source_param_p95_residual": float(np.percentile(residual, 95))}


def fit_sphere(points: np.ndarray) -> dict:
    design = np.column_stack((2.0 * points, np.ones(len(points))))
    parameters, *_ = np.linalg.lstsq(design, np.sum(points * points, axis=1), rcond=None)
    center = parameters[:3]
    radius_sq = float(center @ center + parameters[3])
    if radius_sq <= 0 or not np.isfinite(radius_sq):
        return {}
    radius = math.sqrt(radius_sq)
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    return {"center": center.tolist(), "radius": radius,
            "source_param_p95_residual": float(np.percentile(residual, 95))}


def fit_cylinder(points: np.ndarray, normals: np.ndarray) -> dict:
    values, vectors = np.linalg.eigh(normals.T @ normals)
    axis = unit(vectors[:, int(np.argmin(values))])
    e1 = unit(np.cross(axis, np.array([1.0, 0.0, 0.0])))
    if np.linalg.norm(e1) < 0.5:
        e1 = unit(np.cross(axis, np.array([0.0, 1.0, 0.0])))
    e2 = unit(np.cross(axis, e1))
    uv = np.column_stack((points @ e1, points @ e2))
    design = np.column_stack((2.0 * uv, np.ones(len(uv))))
    parameters, *_ = np.linalg.lstsq(design, np.sum(uv * uv, axis=1), rcond=None)
    center_uv = parameters[:2]
    radius_sq = float(center_uv @ center_uv + parameters[2])
    if radius_sq <= 0 or not np.isfinite(radius_sq):
        return {}
    radius = math.sqrt(radius_sq)
    axial = float(np.mean(points @ axis))
    axis_point = center_uv[0] * e1 + center_uv[1] * e2 + axial * axis
    residual = np.abs(np.linalg.norm(uv - center_uv, axis=1) - radius)
    return {"axis_direction": axis.tolist(), "axis_point": axis_point.tolist(),
            "radius": radius,
            "source_param_p95_residual": float(np.percentile(residual, 95))}


def sample_digest(points: np.ndarray, normals: np.ndarray,
                  labels: np.ndarray, types: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (points, normals, labels, types):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def adapt_one(handle: h5py.File, row: dict, order: int, output_root: Path) -> dict:
    index = row["source_index"]
    points = np.asarray(handle["points"][index])
    normals = np.asarray(handle["normals"][index])
    labels = np.asarray(handle["labels"][index]).reshape(-1)
    types = np.asarray(handle["prim"][index]).reshape(-1)
    name = f"mini_{order:03d}_train_{index:05d}"
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)

    np.savetxt(output / "points.txt", points, fmt="%.9g")
    np.savetxt(output / "normals.txt", normals, fmt="%.9g")
    point_types = np.empty(len(points), dtype=object)
    primitives = []
    for instance in instance_rows(labels, types):
        instance_id = instance["instance_id"]
        type_name = instance["type"]
        mask = labels == instance_id
        point_types[mask] = type_name
        params = {}
        if mask.sum() >= 6:
            if type_name == "plane":
                params = fit_plane(points[mask], normals[mask])
            elif type_name == "sphere":
                params = fit_sphere(points[mask])
            elif type_name == "cylinder":
                params = fit_cylinder(points[mask], normals[mask])
        residual_limit = 0.01
        radius_ok = (type_name not in {"sphere", "cylinder"} or
                     (params and 0 < params.get("radius", 0) <= 10.0))
        residual_ok = (params and
                       params.get("source_param_p95_residual", math.inf) <= residual_limit)
        reliable_params = bool(params and radius_ok and residual_ok)
        primitives.append({
            **instance, "is_target": type_name in TARGET,
            "has_params": reliable_params,
            "parameter_source": (
                "derived reference fitted from official ParSeNet points/normals; "
                "not original CAD analytic GT" if reliable_params else
                "derived fit unavailable or rejected by quality gate"),
            "parameter_quality_gate": {
                "p95_residual_max": residual_limit,
                "radius_max": 10.0 if type_name in {"sphere", "cylinder"} else None,
                "accepted": reliable_params,
            },
            **params,
        })

    with (output / "gt_labels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "instance_id", "primitive_type"])
        writer.writerows((i, int(labels[i]), point_types[i]) for i in range(len(points)))
    with (output / "gt_primitives.json").open("w", encoding="utf-8") as stream:
        json.dump({"primitives": primitives}, stream, indent=2)
    complexity = row.get("complexity", "unclassified")
    metadata = {
        "dataset": "abcparts_parsenet_train_mini",
        "official_split": "train",
        "source_index": index,
        "mini_order": order,
        "num_points": len(points),
        "num_instances": len(primitives),
        "complexity": complexity,
        "source_sample_sha256": sample_digest(points, normals, labels, types),
        "coordinates": "official HDF5 values, unchanged by adapter",
        "official_gt": ["points", "normals", "instance labels", "primitive types"],
        "derived_reference": ["plane", "sphere", "cylinder parameters"],
        "source_id_note": (
            "The supplied HDF5 has no train_ids.txt; source_index is the only "
            "recoverable stable sample identifier."),
        "type_counts": row["type_counts"],
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return {"name": name, **row, "complexity": complexity,
            "sample_sha256": metadata["source_sample_sha256"]}


def validate_schema(handle: h5py.File) -> dict:
    required = {"points", "normals", "labels", "prim"}
    missing = required - set(handle.keys())
    if missing:
        raise RuntimeError(f"missing HDF5 fields: {sorted(missing)}")
    shapes = {name: list(handle[name].shape) for name in required}
    if handle["points"].shape != handle["normals"].shape:
        raise RuntimeError("points/normals shape mismatch")
    if handle["labels"].shape != handle["prim"].shape:
        raise RuntimeError("labels/prim shape mismatch")
    if handle["points"].shape[:2] != handle["labels"].shape:
        raise RuntimeError("point/label shape mismatch")
    return {name: {"shape": shapes[name], "dtype": str(handle[name].dtype),
                   "compression": handle[name].compression}
            for name in sorted(required)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--smoke-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--reuse-manifest", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as handle:
        schema = validate_schema(handle)
        if args.reuse_manifest:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            if manifest["selection_version"] != SELECTION_VERSION:
                raise RuntimeError("manifest selection version does not match this tool")
            if manifest["source"]["file_size"] != args.input.stat().st_size:
                raise RuntimeError("source HDF5 size differs from frozen manifest")
            selected = manifest["samples"]
        else:
            stats = scan_stats(handle)
            selected, selection = select_samples(stats, args.count, args.seed)
            manifest = {
                "schema_version": 1,
                "selection_version": SELECTION_VERSION,
                "source": {
                    "path": str(args.input),
                    "file_size": args.input.stat().st_size,
                    "official_split": "train",
                    "format": schema,
                    "source_project": "ParSeNet ABCParts",
                    "source_url": "https://github.com/Hippogriff/parsenet-codebase",
                    "parameter_note": "No T_param field in supplied HDF5.",
                },
                "seed": args.seed,
                "count": args.count,
                "smoke_count": args.smoke_count,
                "selection": selection,
                "samples": selected,
            }
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if args.out_root:
            adapted = []
            expected_hashes = {
                int(row["source_index"]): row["sample_sha256"]
                for row in manifest.get("adapted_samples", [])
            }
            for order, row in enumerate(selected):
                adapted.append(adapt_one(handle, row, order, args.out_root))
                expected = expected_hashes.get(int(row["source_index"]))
                if expected is not None and expected != adapted[-1]["sample_sha256"]:
                    raise RuntimeError(
                        f"sample hash mismatch for source index {row['source_index']}")
                print(f"adapt {order + 1}/{len(selected)}: {adapted[-1]['name']}", flush=True)
            manifest["adapted_samples"] = adapted
            manifest["smoke_sample_names"] = [row["name"] for row in adapted[:args.smoke_count]]
            args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
