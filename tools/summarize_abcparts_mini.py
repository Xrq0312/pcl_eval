"""Stratified ABCParts-Mini report for scene detection and oracle fitting."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


TARGET = {"plane", "sphere", "cylinder"}


def size_bin(points: int) -> str:
    if points <= 100:
        return "small_le100"
    if points <= 500:
        return "medium_101_500"
    return "large_gt500"


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def group_metrics(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result = {}
    for name, values in sorted(groups.items()):
        result[name] = {
            "instances": len(values),
            "recall_iou50": ratio(sum(row["detected"] for row in values), len(values)),
            "typed_recall_iou50": ratio(sum(row["typed"] for row in values), len(values)),
            "miou": statistics.fmean(row["iou"] for row in values),
        }
    return result


def dataset_summary(samples_root: Path) -> dict:
    type_counts: dict[str, int] = defaultdict(int)
    size_counts: dict[str, int] = defaultdict(int)
    complexity_counts: dict[str, int] = defaultdict(int)
    reliable: dict[str, int] = defaultdict(int)
    all_instances = 0
    target_instances = 0
    for sample in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
        primitives = json.loads((sample / "gt_primitives.json").read_text(
            encoding="utf-8"))["primitives"]
        complexity_counts[metadata["complexity"]] += 1
        for primitive in primitives:
            all_instances += 1
            type_counts[primitive["type"]] += 1
            if primitive["type"] in TARGET:
                target_instances += 1
                size_counts[size_bin(primitive["points"])] += 1
                if primitive.get("has_params"):
                    reliable[primitive["type"]] += 1
    return {
        "samples": sum(complexity_counts.values()),
        "points_per_sample": 10000,
        "instances": all_instances,
        "target_instances": target_instances,
        "type_counts": dict(sorted(type_counts.items())),
        "target_size_counts": dict(sorted(size_counts.items())),
        "sample_complexity_counts": dict(sorted(complexity_counts.items())),
        "reliable_derived_parameter_instances": dict(sorted(reliable.items())),
    }


def summarize_native(samples_root: Path, results_root: Path, stage: str) -> dict:
    rows = []
    parameter_rows = []
    predicted_by_type: dict[str, int] = defaultdict(int)
    typed_detected_by_type: dict[str, int] = defaultdict(int)
    runtimes = []
    walls = []
    non_target_absorptions = 0
    wrong_type_detections = 0
    unmatched_predictions = 0
    for evaluation_path in sorted((results_root / stage).glob("*/evaluation.json")):
        sample_name = evaluation_path.parent.name
        sample = samples_root / sample_name
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        primitives = json.loads((sample / "gt_primitives.json").read_text(
            encoding="utf-8"))["primitives"]
        metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
        pairs = {pair["gt"]: pair for pair in evaluation["final_instance"]["matched_pairs"]}
        for primitive in primitives:
            if primitive["type"] not in TARGET:
                continue
            pair = pairs.get(primitive["instance_id"])
            iou = float(pair["iou"]) if pair else 0.0
            detected = iou >= 0.5
            typed = detected and pair["pred_type"] == primitive["type"]
            rows.append({
                "sample": sample_name, "type": primitive["type"],
                "points": primitive["points"], "size": size_bin(primitive["points"]),
                "complexity": metadata["complexity"], "iou": iou,
                "detected": detected, "typed": typed,
            })
            wrong_type_detections += int(detected and not typed)
        output = json.loads((evaluation_path.parent / "out_primitives.json").read_text(
            encoding="utf-8"))
        for model in output.get("models", []):
            if model["type"] in TARGET:
                predicted_by_type[model["type"]] += 1
        for pair in evaluation["final_instance"]["matched_pairs"]:
            if pair["iou"] >= 0.5 and pair["gt_type"] == pair["pred_type"] \
                    and pair["gt_type"] in TARGET:
                typed_detected_by_type[pair["gt_type"]] += 1
            if pair["iou"] >= 0.5 and pair["gt_type"] not in TARGET \
                    and pair["pred_type"] in TARGET:
                non_target_absorptions += 1
        unmatched_predictions += len(evaluation["final_instance"]["pred_unmatched"])
        parameter_rows.extend(row for row in evaluation.get("parameter_errors", [])
                              if row["type"] == "cylinder")
        timing = evaluation.get("timing_seconds", {}).get("total")
        if timing is not None:
            runtimes.append(float(timing))
        run_path = evaluation_path.parent / "run.json"
        if run_path.exists():
            walls.append(float(json.loads(run_path.read_text(encoding="utf-8"))[
                "wall_seconds"]))

    detected = sum(row["detected"] for row in rows)
    typed = sum(row["typed"] for row in rows)
    total_predictions = sum(predicted_by_type.values())
    cylinder_metrics = {}
    for key in ("axis_angle_deg", "axis_line_dist", "radius_abs_err", "radius_rel_err"):
        values = [float(row[key]) for row in parameter_rows if key in row]
        cylinder_metrics[key] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    return {
        "stage": stage,
        "samples": len({row["sample"] for row in rows}),
        "gt_target_instances": len(rows),
        "predicted_target_instances": total_predictions,
        "detected_instances_iou50": detected,
        "typed_detected_instances_iou50": typed,
        "micro_recall_iou50": ratio(detected, len(rows)),
        "micro_precision_iou50": ratio(detected, total_predictions),
        "typed_micro_recall_iou50": ratio(typed, len(rows)),
        "typed_micro_precision_iou50": ratio(typed, total_predictions),
        "miou": statistics.fmean(row["iou"] for row in rows),
        "by_type": group_metrics(rows, "type"),
        "by_instance_size": group_metrics(rows, "size"),
        "by_sample_complexity": group_metrics(rows, "complexity"),
        "typed_precision_by_predicted_type": {
            name: ratio(typed_detected_by_type[name], count)
            for name, count in sorted(predicted_by_type.items())
        },
        "cylinder_parameter_errors_reliable_derived_reference_only": cylinder_metrics,
        "mean_algorithm_seconds": statistics.fmean(runtimes) if runtimes else None,
        "mean_wall_seconds": statistics.fmean(walls) if walls else None,
        "failure_counts": {
            "missed_gt_instances": len(rows) - detected,
            "wrong_type_at_iou50": wrong_type_detections,
            "predictions_absorbing_non_target_instance_iou50": non_target_absorptions,
            "unmatched_predictions_iou50": unmatched_predictions,
        },
    }


def summarize_oracle(samples_root: Path, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    metadata_cache = {}
    for record in records:
        sample = Path(record["sample_path"])
        if record["sample"] not in metadata_cache:
            metadata_cache[record["sample"]] = json.loads(
                (sample / "metadata.json").read_text(encoding="utf-8"))
        record["complexity"] = metadata_cache[record["sample"]]["complexity"]
        record["size"] = size_bin(record["source_points"])
    variants = {}
    for field in ("model_competition_without_stability", "model_competition",
                  "model_competition_soft_geometry", "model_competition_combined"):
        correct = sum(record[field]["selected"] == record["type"] for record in records)
        rejected = sum(record[field]["rejected"] for record in records)
        variants[field] = {"correct": correct, "total": len(records),
                           "accuracy": ratio(correct, len(records)),
                           "rejected": rejected,
                           "rejection_rate": ratio(rejected, len(records))}

    def oracle_group(field: str) -> dict:
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            groups[str(record[field])].append(record)
        return {
            name: {"instances": len(values),
                   "accuracy": ratio(sum(value[
                       "model_competition_without_stability"]["selected"] == value["type"]
                       for value in values), len(values)),
                   "rejection_rate": ratio(sum(value[
                       "model_competition_without_stability"]["rejected"]
                       for value in values), len(values))}
            for name, values in sorted(groups.items())
        }
    return {
        "evaluation_mode": payload["evaluation_mode"],
        "instances": len(records),
        "competition_ablation": variants,
        "pure_bic_by_type": oracle_group("type"),
        "pure_bic_by_instance_size": oracle_group("size"),
        "pure_bic_by_sample_complexity": oracle_group("complexity"),
        "robust_candidate_reliability": payload["summary"][
            "robust_candidate_reliability"],
        "mean_normal_ms": payload["summary"]["mean_normal_ms"],
        "mean_candidate_and_competition_ms": payload["summary"][
            "mean_candidate_and_competition_ms"],
        "warning": "GT defines each input patch; these are not detector recall/precision/IoU.",
    }


def summarize_voting(samples_root: Path, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gt_rows = []
    candidates = []
    runtimes = []
    for record in payload["records"]:
        metadata = json.loads((samples_root / record["sample"] / "metadata.json").read_text(
            encoding="utf-8"))
        best: dict[int, float] = defaultdict(float)
        for candidate in record["candidates"]:
            candidates.append(candidate)
            instance = int(candidate.get("best_gt_instance", -1))
            if instance >= 0:
                best[instance] = max(best[instance], float(candidate["best_gt_iou"]))
        for gt in record["gt_cylinders"]:
            points = int(gt["sampled_points"])
            gt_rows.append({"sample": record["sample"], "type": "cylinder",
                            "points": points, "size": size_bin(points),
                            "complexity": metadata["complexity"],
                            "iou": best[int(gt["instance_id"])],
                            "detected": best[int(gt["instance_id"])] >= 0.5,
                            "typed": best[int(gt["instance_id"])] >= 0.5})
        runtimes.append(float(record["runtime_ms"]))
    hits30 = sum(float(candidate.get("best_gt_iou", 0)) >= 0.30 for candidate in candidates)
    hits50 = sum(float(candidate.get("best_gt_iou", 0)) >= 0.50 for candidate in candidates)
    return {
        "evaluation_mode": payload["evaluation_mode"],
        "scenes": len(payload["records"]),
        "gt_cylinders": len(gt_rows),
        "candidates": len(candidates),
        "recall_iou30": ratio(sum(row["iou"] >= 0.30 for row in gt_rows), len(gt_rows)),
        "precision_iou30": ratio(hits30, len(candidates)),
        "recall_iou50": ratio(sum(row["detected"] for row in gt_rows), len(gt_rows)),
        "precision_iou50": ratio(hits50, len(candidates)),
        "miou": statistics.fmean(row["iou"] for row in gt_rows),
        "by_instance_size_iou50": group_metrics(gt_rows, "size"),
        "by_sample_complexity_iou50": group_metrics(gt_rows, "complexity"),
        "mean_runtime_ms": statistics.fmean(runtimes),
        "warning": "Cylinder-only proposal experiment; not a plane/sphere/cylinder segmenter.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--voting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "dataset": dataset_summary(args.samples_root),
        "scene_detection": {
            stage: summarize_native(args.samples_root, args.native_root, stage)
            for stage in ("global_ransac", "joint_opt")
        },
        "oracle_instance_fitting": summarize_oracle(args.samples_root, args.oracle),
        "point_only_cylinder_voting": summarize_voting(args.samples_root, args.voting),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
