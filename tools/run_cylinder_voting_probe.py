"""Run the xyz-only cylinder voting prototype on complete labeled scenes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def sample_dirs(root: Path) -> list[Path]:
    required = ("points.txt", "gt_labels.csv", "gt_primitives.json")
    return sorted(path for path in root.rglob("*")
                  if path.is_dir() and all((path / name).is_file() for name in required))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exe", type=Path,
                        default=Path("build-native/Release/cylinder_voting_scene_probe.exe"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--directions", type=int, default=192)
    parser.add_argument("--hypotheses", type=int, default=256)
    parser.add_argument("--minimum-support", type=int, default=80)
    args = parser.parse_args()

    samples = sample_dirs(args.root)
    if args.max_samples:
        samples = samples[:args.max_samples]
    records = []
    failures = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        command = [str(args.exe.resolve()), "--sample-dir", str(sample.resolve()),
                   "--max-points", str(args.max_points),
                   "--directions", str(args.directions),
                   "--hypotheses", str(args.hypotheses),
                   "--minimum-support", str(args.minimum_support)]
        try:
            completed = subprocess.run(command, check=True, text=True, encoding="utf-8",
                                       errors="replace", capture_output=True)
            record = json.loads(completed.stdout)
            record["sample_path"] = str(sample)
            records.append(record)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            failures.append({"sample": str(sample), "error": str(error)})
        print(f"[{index}/{len(samples)}] {sample.name}")

    gt_total = sum(len(record["gt_cylinders"]) for record in records)
    candidates = [candidate for record in records for candidate in record["candidates"]]
    recalled_iou: set[tuple[str, int]] = set()
    recalled_strict: set[tuple[str, int]] = set()
    for record in records:
        for candidate in record["candidates"]:
            instance = candidate.get("best_gt_instance", -1)
            key = (record["sample_path"], instance)
            if instance >= 0 and candidate.get("best_gt_iou", 0.0) >= 0.30:
                recalled_iou.add(key)
                if (candidate.get("radius_relative_error", float("inf")) < 0.10
                        and candidate.get("axis_degrees", float("inf")) < 5.0
                        and candidate.get("axis_offset_relative", float("inf")) < 0.20):
                    recalled_strict.add(key)
    false_positives = sum(candidate.get("best_gt_iou", 0.0) < 0.10
                          for candidate in candidates)
    summary = {
        "scenes": len(records),
        "gt_cylinders": gt_total,
        "candidates": len(candidates),
        "iou30_recalled": len(recalled_iou),
        "iou30_candidate_recall": len(recalled_iou) / gt_total if gt_total else None,
        "strict_parameter_recalled": len(recalled_strict),
        "strict_parameter_recall": len(recalled_strict) / gt_total if gt_total else None,
        "false_positive_candidates_iou_below_010": false_positives,
        "false_positive_fraction": false_positives / len(candidates) if candidates else None,
        "mean_runtime_ms": statistics.fmean(
            record["runtime_ms"] for record in records) if records else None,
    }
    payload = {
        "schema_version": 1,
        "evaluation_mode": "full_scene_point_only_candidate_detection",
        "leakage_guard": "GT is read only after xyz-only detection for matching",
        "thresholds": {"candidate_iou": 0.30, "false_positive_iou": 0.10,
                       "radius_relative_error": 0.10, "axis_degrees": 5.0,
                       "axis_offset_relative": 0.20},
        "root": str(args.root), "max_points": args.max_points,
        "directions": args.directions, "hypotheses": args.hypotheses,
        "minimum_support": args.minimum_support,
        "wall_seconds": time.perf_counter() - started,
        "summary": summary, "failures": failures, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
