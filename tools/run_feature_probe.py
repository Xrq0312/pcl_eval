"""Run the independent geometry feature probe over adapted sample directories.

This is an oracle-instance experiment: GT labels define each input patch.  It
measures feature/initialization quality and must not be reported as detector
instance recall or IoU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path


def sample_dirs(root: Path) -> list[Path]:
    required = ("points.txt", "gt_labels.csv", "gt_primitives.json")
    return sorted(
        path for path in root.rglob("*")
        if path.is_dir() and all((path / name).is_file() for name in required)
    )


def run_probe(executable: Path, sample: Path, normal_k: int,
              aso_k: int, max_points: int, min_points: int) -> dict:
    command = [
        str(executable), "--sample-dir", str(sample),
        "--normal-k", str(normal_k), "--aso-k", str(aso_k),
        "--max-points", str(max_points),
        "--min-points", str(min_points),
    ]
    completed = subprocess.run(
        command, check=True, text=True, encoding="utf-8",
        errors="replace", capture_output=True,
    )
    return json.loads(completed.stdout)


def classify_pair(record: dict) -> str:
    sphere = record["sphere_low_rank"]
    cylinder = record["cylinder_low_rank"]
    pairs = record["point_pairs"]
    if not sphere["valid"] and not cylinder["valid"]:
        return "plane"
    if pairs["normal_change"] < 0.10:
        return "plane"
    if pairs["sphere_residual"] < 0.10:
        return "sphere"
    return "cylinder"


def classify_aso(record: dict) -> str:
    aso = record.get("aso", {})
    if not aso.get("valid"):
        return "cylinder"
    if aso.get("valid") and aso["scaled_abs_k1"] < 0.07:
        return "plane"
    if aso["abs_curvature_ratio"] > 0.70:
        return "sphere"
    return "cylinder"


def classify_combined(record: dict) -> str:
    sphere = record["sphere_low_rank"]
    cylinder = record["cylinder_low_rank"]
    pairs = record["point_pairs"]
    aso = record.get("aso", {})
    if not sphere["valid"] and not cylinder["valid"]:
        return "plane"
    if aso.get("valid") and aso["scaled_abs_k1"] < 0.07:
        return "plane"
    if (aso.get("valid") and aso["abs_curvature_ratio"] > 0.70
            and pairs["sphere_residual"] < 0.30):
        return "sphere"
    return "cylinder"


def summarize(records: list[dict]) -> dict:
    confusion: Counter[tuple[str, str]] = Counter()
    parameter_errors: dict[str, list[float]] = {}
    ablation_correct = {"point_pair": 0, "aso": 0, "combined": 0}
    for record in records:
        predictions = {
            "point_pair": classify_pair(record),
            "aso": classify_aso(record),
            "combined": classify_combined(record),
        }
        record["frozen_rule_predictions"] = predictions
        prediction = predictions["combined"]
        confusion[(record["type"], prediction)] += 1
        for name, predicted_type in predictions.items():
            if predicted_type == record["type"]:
                ablation_correct[name] += 1
        for name, value in record.get("parameter_errors", {}).items():
            if isinstance(value, (int, float)):
                parameter_errors.setdefault(name, []).append(float(value))
    total = len(records)
    correct = sum(count for (truth, prediction), count in confusion.items()
                  if truth == prediction)
    sphere_records = [r for r in records if r["type"] == "sphere"
                      and "sphere_radius_relative" in r.get("parameter_errors", {})]
    cylinder_records = [r for r in records if r["type"] == "cylinder"
                        and "cylinder_radius_relative" in r.get("parameter_errors", {})]

    def rate(items: list[dict], predicate) -> dict:
        successes = sum(bool(predicate(item)) for item in items)
        return {
            "successes": successes,
            "total": len(items),
            "rate": successes / len(items) if items else None,
        }

    reliability = {
        "sphere_estimated_normals": rate(sphere_records, lambda r:
            r["parameter_errors"]["sphere_radius_relative"] < 0.05
            and r["parameter_errors"]["sphere_center_relative"] < 0.10),
        "sphere_oracle_normals": rate(
            [r for r in sphere_records if "oracle_normal_low_rank" in r], lambda r:
            r["oracle_normal_low_rank"]["radius_relative_error"] < 0.05
            and r["oracle_normal_low_rank"]["center_relative_error"] < 0.10),
        "cylinder_estimated_normals": rate(cylinder_records, lambda r:
            r["parameter_errors"]["cylinder_radius_relative"] < 0.05
            and r["parameter_errors"]["cylinder_axis_degrees"] < 2.0
            and r["parameter_errors"]["cylinder_axis_offset_relative"] < 0.10),
        "cylinder_oracle_normals": rate(
            [r for r in cylinder_records if "oracle_normal_low_rank" in r], lambda r:
            r["oracle_normal_low_rank"]["radius_relative_error"] < 0.05
            and r["oracle_normal_low_rank"]["axis_degrees"] < 2.0
            and r["oracle_normal_low_rank"]["axis_offset_relative"] < 0.10),
    }
    return {
        "instances": total,
        "classification_correct": correct,
        "classification_accuracy": correct / total if total else None,
        "classification_ablation": {
            name: {"correct": count, "total": total,
                   "accuracy": count / total if total else None}
            for name, count in ablation_correct.items()
        },
        "confusion": [
            {"truth": truth, "prediction": prediction, "count": count}
            for (truth, prediction), count in sorted(confusion.items())
        ],
        "low_rank_reliability": reliability,
        "parameter_error_means": {
            name: statistics.fmean(values)
            for name, values in sorted(parameter_errors.items()) if values
        },
        "parameter_error_medians": {
            name: statistics.median(values)
            for name, values in sorted(parameter_errors.items()) if values
        },
        "mean_normal_ms": statistics.fmean(
            record["normal_ms"] for record in records) if records else None,
        "mean_feature_ms": statistics.fmean(
            record["feature_ms"] for record in records) if records else None,
        "mean_aso_ms": statistics.fmean(
            record["aso_ms"] for record in records) if records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--exe", type=Path,
        default=Path("build-native/Release/geometry_feature_probe.exe"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-k", type=int, default=64)
    parser.add_argument("--aso-k", type=int, default=64)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--min-points", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    samples = sample_dirs(args.root)
    if args.max_samples:
        samples = samples[: args.max_samples]
    records: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        try:
            result = run_probe(
                args.exe.resolve(), sample.resolve(), args.normal_k,
                args.aso_k, args.max_points, args.min_points,
            )
            for record in result["records"]:
                record["sample_path"] = str(sample)
                record["normal_k"] = args.normal_k
                records.append(record)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            failures.append({"sample": str(sample), "error": str(error)})
        print(f"[{index}/{len(samples)}] {sample.name}")

    payload = {
        "schema_version": 1,
        "evaluation_mode": "oracle_instance_patch_not_detection",
        "root": str(args.root),
        "normal_k": args.normal_k,
        "aso_k": args.aso_k,
        "max_points_per_instance": args.max_points,
        "samples": len(samples),
        "wall_seconds": time.perf_counter() - started,
        "summary": summarize(records),
        "failures": failures,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
