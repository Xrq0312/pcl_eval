"""Frozen native-v2.7 versus PCL-D4 audit on the Fit4CAD test subset."""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluate import HAVE_SCIPY, evaluate_sample
from generalization_audit import METRICS, sha256, validate_output


DISTANCE_DIAGONAL_RATIO = 0.004


def geometry(sample_dir):
    with (sample_dir / "metadata.json").open() as handle:
        metadata = json.load(handle)
    bbox = metadata["bbox"]
    diagonal = math.sqrt(sum((bbox[axis][1] - bbox[axis][0]) ** 2
                             for axis in ("x", "y", "z")))
    distance = DISTANCE_DIAGONAL_RATIO * diagonal
    return {
        "bbox_diagonal": diagonal,
        "distance": distance,
        "min_radius": 0.5 * distance,
        "max_radius": 500.0 * distance,
        "merge_distance": 5.0 * distance,
        "merge_max_gap": 10.0 * distance,
    }


def command(engine, executable, points, prefix, scale):
    common = {
        "distance": scale["distance"], "min_points": 80,
        "smooth": 12, "curvature": 0.05,
        "min_radius": scale["min_radius"],
        "max_radius": scale["max_radius"],
        "merge_distance": scale["merge_distance"], "merge_angle": 5,
    }
    if engine == "native_v27":
        return [str(executable), str(points), str(prefix),
                f"--distance={common['distance']:.17g}",
                f"--min-points={common['min_points']}",
                f"--smooth-deg={common['smooth']}",
                f"--curvature={common['curvature']}",
                f"--min-radius={common['min_radius']:.17g}",
                f"--max-radius={common['max_radius']:.17g}",
                f"--merge-distance={common['merge_distance']:.17g}",
                f"--merge-angle-deg={common['merge_angle']}",
                f"--merge-max-gap={scale['merge_max_gap']:.17g}",
                "--normal-k=300", "--region-neighbors=30", "--threads=16"]
    return [str(executable), str(points), str(prefix),
            f"{common['distance']:.17g}", str(common["min_points"]),
            str(common["smooth"]), str(common["curvature"]),
            f"{common['min_radius']:.17g}", f"{common['max_radius']:.17g}",
            f"{common['merge_distance']:.17g}", str(common["merge_angle"]),
            "--normalK=300", "--regionNeighbors=30", "--maxIterations=2000",
            "--cylinderIterations=4000", "--minInlierRatio=0.3",
            "--maxRmseFactor=0.75", "--ablation=D4",
            f"--fragmentMaxGap={scale['merge_max_gap']:.17g}"]


def parameter_summary(rows):
    result = {}
    fields = {
        "plane": ("normal_angle_deg", "offset_dist"),
        "sphere": ("center_dist", "radius_abs_err", "radius_rel_err"),
        "cylinder": ("axis_angle_deg", "axis_line_dist", "radius_abs_err",
                     "radius_rel_err"),
    }
    for primitive_type, names in fields.items():
        matches = [item for row in rows for item in row["parameter_errors"]
                   if item["type"] == primitive_type]
        entry = {"matched_instances": len(matches)}
        for name in names:
            values = [float(item[name]) for item in matches
                      if item.get(name) is not None and math.isfinite(float(item[name]))]
            entry[f"mean_{name}"] = statistics.mean(values) if values else None
        result[primitive_type] = entry
    return result


def main():
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required for exact one-to-one matching")
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--native-exe", required=True, type=Path)
    parser.add_argument("--pcl-d4-exe", required=True, type=Path)
    parser.add_argument("--samples", nargs="*")
    args = parser.parse_args()
    samples = tuple(args.samples or sorted(
        path.name for path in args.samples_root.iterdir()
        if path.is_dir() and (path / "gt_labels.csv").is_file()))
    if not samples:
        parser.error("no unified samples found")
    for executable in (args.native_exe, args.pcl_d4_exe):
        if not executable.is_file():
            parser.error(f"missing executable: {executable}")
    protocol = {
        "name": "fit4cad_test12_frozen_cross_dataset_audit_v1",
        "samples": samples,
        "distance_rule": "0.004 * bbox diagonal",
        "derived_rules": {"min_radius": "0.5 * distance",
                          "max_radius": "500 * distance",
                          "merge_distance": "5 * distance",
                          "merge_max_gap": "10 * distance"},
        "fixed_counts": {"min_points": 80, "normal_k": 300,
                         "region_neighbors": 30},
        "note": "Rules fixed before fitting; identical geometry/count settings for both engines; no GT-dependent tuning.",
        "executables": {
            "native_v27": {"path": str(args.native_exe.resolve()),
                            "sha256": sha256(args.native_exe)},
            "pcl_d4": {"path": str(args.pcl_d4_exe.resolve()),
                       "sha256": sha256(args.pcl_d4_exe)},
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")
    records, failures, evaluations = [], [], {"native_v27": [], "pcl_d4": []}
    for sample in samples:
        sample_dir = args.samples_root / sample
        scale = geometry(sample_dir)
        for engine, executable in (("native_v27", args.native_exe),
                                   ("pcl_d4", args.pcl_d4_exe)):
            output = args.output_root / engine / sample
            output.mkdir(parents=True, exist_ok=True)
            prefix = output / "out"
            invocation = command(engine, executable, sample_dir / "points.txt",
                                 prefix, scale)
            started = time.perf_counter()
            process = subprocess.run(invocation, capture_output=True, text=True,
                                     timeout=600)
            wall = time.perf_counter() - started
            (output / "run_log.txt").write_text(
                process.stdout + process.stderr, encoding="utf-8")
            if process.returncode not in (0, 2):
                failures.append({"sample": sample, "engine": engine,
                                 "returncode": process.returncode})
                print(f"{sample} {engine} FAILED rc={process.returncode}", flush=True)
                continue
            try:
                evaluation = evaluate_sample(str(sample_dir), str(prefix))
                integrity = validate_output(prefix, evaluation)
            except Exception as error:
                failures.append({"sample": sample, "engine": engine,
                                 "error": str(error)})
                print(f"{sample} {engine} INVALID {error}", flush=True)
                continue
            evaluation["dataset"] = "Fit4CAD official test subset"
            evaluation["wall_seconds"] = wall
            (output / "evaluation.json").write_text(
                json.dumps(evaluation, indent=2), encoding="utf-8")
            prf = evaluation.get("point_type_prf", {})
            instance = evaluation["final_instance"]
            row = {"sample": sample, "engine": engine, "wall": wall,
                   **scale, **integrity,
                   "num_gt_target": int(instance["num_gt_target"]),
                   "detected_target": int(round(instance["detection_target_rate"] *
                                                instance["num_gt_target"])),
                   "detected_typed": int(round(instance["detection_typed_rate"] *
                                               instance["num_gt_target"])),
                   "target_iou_sum": float(instance["miou_target"] *
                                           instance["num_gt_target"]),
                   **{key: float(accessor(evaluation))
                      for key, accessor in METRICS.items()},
                   **{f"{kind}_f1": prf.get(kind, {}).get("f1")
                      for kind in ("plane", "sphere", "cylinder")}}
            records.append(row)
            evaluations[engine].append(evaluation)
            print(f"{sample} {engine} det={row['det']:.3f} "
                  f"typed={row['typed']:.3f} miou={row['miou']:.3f} "
                  f"acc={row['point_acc']:.3f} wall={wall:.2f}s", flush=True)
            (args.output_root / "progress.json").write_text(
                json.dumps({"records": records, "failures": failures}, indent=2),
                encoding="utf-8")
    summary = {"protocol": protocol, "records": records, "failures": failures,
               "mean": {}, "instance_weighted": {}, "parameter_errors": {}}
    mean_keys = (*METRICS, "plane_f1", "sphere_f1", "cylinder_f1", "wall",
                 "models")
    for engine in ("native_v27", "pcl_d4"):
        rows = [row for row in records if row["engine"] == engine]
        summary["mean"][engine] = {}
        for key in mean_keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            summary["mean"][engine][key] = statistics.mean(values) if values else None
        total_gt = sum(row["num_gt_target"] for row in rows)
        summary["instance_weighted"][engine] = {
            "num_gt_target": total_gt,
            "det": sum(row["detected_target"] for row in rows) / total_gt,
            "typed": sum(row["detected_typed"] for row in rows) / total_gt,
            "miou": sum(row["target_iou_sum"] for row in rows) / total_gt,
        } if total_gt else {}
        summary["parameter_errors"][engine] = parameter_summary(evaluations[engine])
    paired = []
    for sample in samples:
        pair = {row["engine"]: row for row in records if row["sample"] == sample}
        if len(pair) == 2:
            paired.append({"sample": sample,
                           **{key: pair["native_v27"][key] - pair["pcl_d4"][key]
                              for key in mean_keys
                              if pair["native_v27"].get(key) is not None
                              and pair["pcl_d4"].get(key) is not None}})
    summary["paired_deltas_native_minus_pcl_d4"] = paired
    quality_keys = ("det", "typed", "miou", "point_acc")
    summary["paired_wins"] = {
        key: {"native": sum(row[key] > 1e-12 for row in paired),
              "tie": sum(abs(row[key]) <= 1e-12 for row in paired),
              "pcl_d4": sum(row[key] < -1e-12 for row in paired)}
        for key in quality_keys
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"mean": summary["mean"],
                      "instance_weighted": summary["instance_weighted"],
                      "paired_wins": summary["paired_wins"],
                      "parameter_errors": summary["parameter_errors"],
                      "failures": failures}, indent=2), flush=True)
    return 0 if not failures and len(paired) == len(samples) else 1


if __name__ == "__main__":
    sys.exit(main())
