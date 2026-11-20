"""Summarize the independent robust-candidate synthetic experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def candidate_success(row: dict[str, str]) -> bool:
    if row["valid"] != "1":
        return False
    if float(row["radius_abs_error"]) >= 0.05 or float(row["location_error"]) >= 0.10:
        return False
    return row["truth"] != "cylinder" or float(row["axis_degrees"]) < 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["family"] in {"sphere", "cylinder"}:
            grouped[(row["truth"], row["suite"], row["estimator"])].append(row)

    candidate_summary = []
    for (truth, suite, estimator), items in sorted(grouped.items()):
        successes = sum(candidate_success(item) for item in items)
        candidate_summary.append({
            "truth": truth,
            "suite": suite,
            "estimator": estimator,
            "successes": successes,
            "total": len(items),
            "recall": successes / len(items),
            "median_radius_error": statistics.median(
                float(item["radius_abs_error"]) for item in items),
            "median_location_error": statistics.median(
                float(item["location_error"]) for item in items),
            "median_axis_degrees": statistics.median(
                float(item["axis_degrees"]) for item in items),
            "mean_runtime_ms": statistics.fmean(
                float(item["runtime_ms"]) for item in items),
        })

    # One competition record per generated patch; candidate rows repeat it.
    cases: dict[str, dict[str, str]] = {}
    for row in rows:
        cases.setdefault(row["seed"], row)
    competition_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in cases.values():
        competition_groups[(row["truth"], row["suite"])].append(row)
    competition_summary = []
    for (truth, suite), items in sorted(competition_groups.items()):
        correct = sum(item["competition_selected"] == truth for item in items)
        rejected = sum(item["rejected"] == "1" for item in items)
        competition_summary.append({
            "truth": truth,
            "suite": suite,
            "correct": correct,
            "total": len(items),
            "accuracy_with_rejection_as_error": correct / len(items),
            "rejected": rejected,
            "rejection_rate": rejected / len(items),
        })

    payload = {
        "schema_version": 1,
        "evaluation_mode": "synthetic_known_instance_patch_not_scene_detection",
        "candidate_success_thresholds": {
            "radius_abs_error": 0.05,
            "location_error": 0.10,
            "cylinder_axis_degrees": 2.0,
        },
        "rows": len(rows),
        "cases": len(cases),
        "candidate_summary": candidate_summary,
        "competition_summary": competition_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
