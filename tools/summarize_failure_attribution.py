"""Aggregate per-instance proposal lifecycle failure attribution."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LIFECYCLE = (
    ("screened_proposals", 0.30),
    ("proposals", 0.30),
    ("selected_models", 0.30),
    ("premerge", 0.50),
    ("final_typed", 0.50),
)


def size_bin(points: int) -> str:
    if points <= 100:
        return "small_le100"
    if points <= 500:
        return "medium_101_500"
    return "large_gt500"


def summarize(records: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    lifecycle = {name: 0 for name, _ in LIFECYCLE}
    for record in records:
        counts[record["outcome"]] += 1
        best = record["best_typed_iou"]
        for name, threshold in LIFECYCLE:
            lifecycle[name] += int(float(best.get(name, 0.0)) >= threshold)
    total = len(records)
    return {
        "instances": total,
        "failure_counts": dict(sorted(counts.items())),
        "failure_rates": {
            key: value / total if total else None
            for key, value in sorted(counts.items())
        },
        "lifecycle_recall": {
            name: value / total if total else None
            for name, value in lifecycle.items()
        },
        "dominant_outcome": max(counts, key=counts.get) if counts else None,
    }


def grouped(records: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: summarize(values) for name, values in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {"schema_version": 1, "stages": {}}
    for stage_dir in sorted(path for path in args.results_root.iterdir()
                            if path.is_dir()):
        records = []
        scene_rows = []
        unavailable = []
        for evaluation_path in sorted(stage_dir.glob("*/evaluation.json")):
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            final = payload["final_instance"]
            scene_rows.append({
                "gt": int(final["num_gt_target"]),
                "pred": int(final["num_pred_target"]),
                "detected": int(final.get("num_detected_target", 0)),
                "typed": int(final.get("num_detected_typed", 0)),
                "miou_sum": float(final["miou_target"]) * int(final["num_gt_target"]),
                "seconds": payload.get("timing_seconds", {}).get("total"),
            })
            attribution = payload.get("failure_attribution", {})
            if not attribution.get("available"):
                unavailable.append(evaluation_path.parent.name)
                continue
            metadata_path = args.samples_root / evaluation_path.parent.name / "metadata.json"
            metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata_path.exists() else {})
            for record in attribution["records"]:
                record = dict(record)
                record["sample"] = evaluation_path.parent.name
                record["size"] = size_bin(int(record["points"]))
                record["complexity"] = metadata.get("complexity", "unknown")
                records.append(record)
        total_gt = sum(row["gt"] for row in scene_rows)
        total_pred = sum(row["pred"] for row in scene_rows)
        measured_seconds = [float(row["seconds"]) for row in scene_rows
                            if row["seconds"] is not None]
        result["stages"][stage_dir.name] = {
            "scene_detection_micro": {
                "samples": len(scene_rows), "gt_target_instances": total_gt,
                "predicted_target_instances": total_pred,
                "recall_iou50": (sum(row["detected"] for row in scene_rows) / total_gt
                                 if total_gt else None),
                "precision_iou50": (sum(row["detected"] for row in scene_rows) / total_pred
                                    if total_pred else None),
                "typed_recall_iou50": (sum(row["typed"] for row in scene_rows) / total_gt
                                       if total_gt else None),
                "typed_precision_iou50": (sum(row["typed"] for row in scene_rows) / total_pred
                                          if total_pred else None),
                "miou": (sum(row["miou_sum"] for row in scene_rows) / total_gt
                         if total_gt else None),
                "mean_algorithm_seconds": (sum(measured_seconds) / len(measured_seconds)
                                           if measured_seconds else None),
            },
            "overall": summarize(records),
            "by_type": grouped(records, "type"),
            "by_instance_size": grouped(records, "size"),
            "by_sample_complexity": grouped(records, "complexity"),
            "unavailable_samples": unavailable,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
