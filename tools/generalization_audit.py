"""Frozen-scene, no-tuning comparison of native v2.7 and local PCL-D4.

The scene IDs deliberately exclude the ten BAGSFit scenes used to select
native v2.7. Results are an audit set, not a development/tuning set.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluate import HAVE_SCIPY, evaluate_sample


HOLDOUT = tuple(f"{scene:03d}_{frame:05d}"
                for scene in (10, 12, 14, 16, 18) for frame in (0, 18))
METRICS = {
    "det": lambda r: r["final_instance"]["detection_target_rate"],
    "typed": lambda r: r["final_instance"]["detection_typed_rate"],
    "miou": lambda r: r["final_instance"]["miou_target"],
    "point_acc": lambda r: r["point_type_accuracy_target"],
    "unassigned": lambda r: r["failures"]["unassigned_ratio"],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(engine, exe, points, prefix):
    if engine == "native_v27":
        return [str(exe), str(points), str(prefix), "--threads=16"]
    return [str(exe), str(points), str(prefix), "0.02", "80", "12",
            "0.05", "0.01", "10", "0.1", "5", "--normalK=300",
            "--regionNeighbors=30", "--maxIterations=2000",
            "--cylinderIterations=4000", "--minInlierRatio=0.3",
            "--maxRmseFactor=0.75", "--ablation=D4"]


def validate_output(prefix, evaluation):
    with Path(str(prefix) + "_primitives.json").open() as handle:
        output = json.load(handle)
    models = output["models"]
    finite = all(all(math.isfinite(float(c)) for c in m["coefficients"])
                 for m in models)
    owned = sum(int(m["points"]) for m in models)
    unassigned = int(evaluation["failures"]["unassigned_points"])
    points = int(output["num_points"])
    duplicates = int(output.get("unassigned_stats", {}).get("duplicate_hits", 0))
    if not finite or owned + unassigned != points or duplicates != 0:
        raise RuntimeError(f"output integrity failed: {prefix}")
    if evaluation["failures"]["nonfinite_param_models"]:
        raise RuntimeError(f"evaluation found non-finite model: {prefix}")
    return {"points": points, "models": len(models), "finite": finite,
            "duplicate_hits": duplicates}


def main():
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required for exact one-to-one matching")
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-exe", type=Path, required=True)
    parser.add_argument("--pcl-d4-exe", type=Path, required=True)
    parser.add_argument("--samples", nargs="*", choices=HOLDOUT,
                        default=HOLDOUT)
    args = parser.parse_args()
    samples = tuple(args.samples)
    if not samples:
        parser.error("at least one sample is required")
    for exe in (args.native_exe, args.pcl_d4_exe):
        if not exe.is_file():
            parser.error(f"missing executable: {exe}")
    manifest = {
        "protocol": "frozen_v2.7_cross_scene_audit_v1",
        "selection": "BAGSFit TEST-20s scene IDs 010/012/014/016/018; frames 00000/00018",
        "development_scene_ids": ["000", "001", "002", "004", "006"],
        "samples": samples,
        "executables": {"native_v27": {"path": str(args.native_exe.resolve()),
                                      "sha256": sha256(args.native_exe)},
                        "pcl_d4": {"path": str(args.pcl_d4_exe.resolve()),
                                   "sha256": sha256(args.pcl_d4_exe)}},
        "note": "Same dataset family, unseen scene IDs; not independent sensor/domain validation.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "protocol.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    records = []
    failures = []
    for sample in samples:
        sample_dir = args.samples_root / sample
        points = sample_dir / "points.txt"
        if not points.is_file():
            raise RuntimeError(f"missing sample: {points}")
        for engine, exe in (("native_v27", args.native_exe),
                            ("pcl_d4", args.pcl_d4_exe)):
            output = args.output_root / engine / sample
            output.mkdir(parents=True, exist_ok=True)
            prefix = output / "out"
            command = command_for(engine, exe, points, prefix)
            started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True,
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
            evaluation["dataset"] = "BAGSFit unseen scenes"
            evaluation["wall_seconds"] = wall
            with (output / "evaluation.json").open("w") as handle:
                json.dump(evaluation, handle, indent=2)
            row = {"sample": sample, "engine": engine, "wall": wall,
                   **integrity,
                   **{key: float(accessor(evaluation))
                      for key, accessor in METRICS.items()}}
            records.append(row)
            print(f"{sample} {engine} det={row['det']:.3f} "
                  f"typed={row['typed']:.3f} miou={row['miou']:.3f} "
                  f"wall={wall:.2f}s", flush=True)
            with (args.output_root / "progress.json").open("w") as handle:
                json.dump({"records": records, "failures": failures},
                          handle, indent=2)
    summary = {"protocol": manifest, "records": records,
               "failures": failures, "mean": {}}
    for engine in ("native_v27", "pcl_d4"):
        rows = [r for r in records if r["engine"] == engine]
        summary["mean"][engine] = {
            key: statistics.mean(r[key] for r in rows) if rows else None
            for key in (*METRICS, "wall")}
    paired = []
    for sample in samples:
        pair = {r["engine"]: r for r in records if r["sample"] == sample}
        if len(pair) == 2:
            paired.append({"sample": sample,
                           **{key: pair["native_v27"][key] - pair["pcl_d4"][key]
                              for key in (*METRICS, "wall")}})
    summary["paired_deltas_native_minus_pcl_d4"] = paired
    with (args.output_root / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"mean": summary["mean"], "failures": failures},
                     indent=2), flush=True)
    return 0 if not failures and len(paired) == len(samples) else 1


if __name__ == "__main__":
    sys.exit(main())
