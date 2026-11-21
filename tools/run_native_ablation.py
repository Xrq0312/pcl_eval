"""Reproducible runner for the independent native primitive pipeline."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def metric_row(evaluation, sample, stage, wall):
    final = evaluation["final_instance"]
    premerge = evaluation["premerge_instance"]
    timing = evaluation.get("timing_seconds", {})
    cylinders = [value for value in evaluation.get("parameter_errors", [])
                 if value.get("type") == "cylinder"]
    mean = lambda key: (sum(float(value[key]) for value in cylinders) / len(cylinders)
                        if cylinders else None)
    return {
        "sample": sample,
        "stage": stage,
        "detection_target": final["detection_target_rate"],
        "precision_target": final.get("detection_target_precision"),
        "detection_typed": final["detection_typed_rate"],
        "precision_typed": final.get("detection_typed_precision"),
        "miou_target": final["miou_target"],
        "premerge_detection_target": premerge["detection_target_rate"],
        "premerge_miou_target": premerge["miou_target"],
        "point_accuracy": evaluation["point_type_accuracy_target"],
        "unassigned_rate": evaluation["failures"]["unassigned_ratio"],
        "models": final["num_pred"],
        "cylinder_matches": len(cylinders),
        "cylinder_axis_deg": mean("axis_angle_deg"),
        "cylinder_axis_line_dist": mean("axis_line_dist"),
        "cylinder_radius_abs": mean("radius_abs_err"),
        "cylinder_radius_rel": mean("radius_rel_err"),
        "algorithm_seconds": timing.get("total"),
        "wall_seconds": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--stages", default="global_local,joint_opt")
    parser.add_argument("--distance", type=float, default=0.02)
    parser.add_argument("--min-points", type=int, default=80)
    parser.add_argument("--small-min-points", type=int, default=20)
    parser.add_argument("--adaptive-local-anchors", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=("all", "development", "audit"),
                        default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-failure-attribution", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    samples_root = Path(args.samples)
    results_root = Path(args.results)
    stages = [value.strip() for value in args.stages.split(",") if value.strip()]
    samples = sorted(path for path in samples_root.iterdir()
                     if path.is_dir() and (path / "gt_labels.csv").exists())
    if args.split != "all":
        if not args.manifest:
            raise ValueError("--manifest is required when --split is not all")
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        development = set(manifest["smoke_sample_names"])
        if args.split == "development":
            samples = [path for path in samples if path.name in development]
        else:
            samples = [path for path in samples if path.name not in development]
    if args.limit:
        samples = samples[:args.limit]
    rows = []
    for sample in samples:
        point_file = sample / "points.txt"
        if not point_file.exists():
            pcd = sample / "points.pcd"
            if not pcd.exists():
                raise FileNotFoundError(f"no points.txt or points.pcd in {sample}")
            point_file = pcd
        for stage in stages:
            output_dir = results_root / stage / sample.name
            output_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_dir / "out"
            evaluation_file = output_dir / "evaluation.json"
            if args.force or not evaluation_file.exists():
                command = [args.exe, "--input", str(point_file),
                           "--output-prefix", str(prefix), "--stage", stage,
                           "--distance", str(args.distance), "--min-points",
                           str(args.min_points), "--small-min-points",
                           str(args.small_min_points), "--adaptive-local-anchors",
                           str(args.adaptive_local_anchors)]
                pixel_map = sample / "pixel_map.txt"
                if pixel_map.exists():
                    command += ["--pixel-map", str(pixel_map)]
                start = time.perf_counter()
                subprocess.run(command, cwd=root, check=True)
                wall = time.perf_counter() - start
                evaluate_command = [
                    sys.executable, str(root / "tools" / "evaluate.py"),
                    "--sample-dir", str(sample), "--pred-prefix", str(prefix),
                    "--out", str(evaluation_file), "--dataset", samples_root.name]
                if args.skip_failure_attribution:
                    evaluate_command.append("--skip-failure-attribution")
                subprocess.run(evaluate_command,
                               cwd=root, check=True)
                with open(output_dir / "run.json", "w", encoding="utf-8") as stream:
                    json.dump({"command": command, "wall_seconds": wall}, stream, indent=2)
            else:
                run_file = output_dir / "run.json"
                wall = json.load(open(run_file, encoding="utf-8")).get("wall_seconds") \
                    if run_file.exists() else None
            with open(evaluation_file, encoding="utf-8") as stream:
                rows.append(metric_row(json.load(stream), sample.name, stage, wall))

    results_root.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with open(results_root / "rows.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for stage in stages:
        selected = [row for row in rows if row["stage"] == stage]
        summary[stage] = {}
        for key in columns:
            if key in {"sample", "stage", "wall_seconds"}:
                continue
            values = [float(row[key]) for row in selected if row[key] is not None]
            summary[stage][key] = (sum(values) if key == "cylinder_matches"
                                   else sum(values) / len(values)) if values else None
            if len(values) != len(selected):
                summary[stage][key + "_samples"] = len(values)
        summary[stage]["samples"] = len(selected)
        summary[stage]["wall_seconds"] = sum(
            float(row["wall_seconds"] or 0.0) for row in selected)
    with open(results_root / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
