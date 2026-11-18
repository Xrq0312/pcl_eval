"""Reproduce the frozen native-v2.7 synthetic scale/density/noise audit."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from evaluate import HAVE_SCIPY, evaluate_sample
from generalization_audit import METRICS, sha256, validate_output
from make_stress_cases import generate_cases


RUNS = (
    ("native_default", "nominal", ()),
    ("native_default", "scale_0p1", ()),
    ("native_default", "scale_10", ()),
    ("native_default", "density_25", ()),
    ("native_default", "density_10", ()),
    ("native_default", "noise_2cm", ()),
    ("native_scale_calibrated", "scale_0p1",
     ("--distance=0.002", "--merge-distance=0.01", "--min-radius=0.001",
      "--max-radius=1", "--merge-max-gap=0.02")),
    ("native_scale_calibrated", "scale_10",
     ("--distance=0.2", "--merge-distance=1", "--min-radius=0.1",
      "--max-radius=100", "--merge-max-gap=2")),
    ("native_density_calibrated", "density_25",
     ("--normal-k=75", "--region-neighbors=8", "--min-points=20")),
    ("native_density_calibrated", "density_10",
     ("--normal-k=30", "--region-neighbors=3", "--min-points=8")),
)


def main():
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required for exact one-to-one matching")
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-exe", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not args.native_exe.is_file():
        parser.error(f"missing executable: {args.native_exe}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    generate_cases(args.output_root / "inputs")
    rows = []
    for protocol, case, options in RUNS:
        sample = args.output_root / "inputs" / case
        output = args.output_root / protocol / case
        output.mkdir(parents=True, exist_ok=True)
        prefix = output / "out"
        command = [str(args.native_exe.resolve()), str(sample / "points.txt"),
                   str(prefix), "--threads=16", *options]
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=600)
        wall = time.perf_counter() - start
        (output / "run_log.txt").write_text(result.stdout + result.stderr,
                                            encoding="utf-8")
        if result.returncode not in (0, 2):
            raise RuntimeError(f"{protocol}/{case} failed rc={result.returncode}")
        evaluation = evaluate_sample(str(sample), str(prefix))
        integrity = validate_output(prefix, evaluation)
        with (output / "evaluation.json").open("w") as handle:
            json.dump(evaluation, handle, indent=2)
        row = {"protocol": protocol, "case": case, "args": options,
               "wall": wall, **integrity,
               **{key: float(accessor(evaluation))
                  for key, accessor in METRICS.items()}}
        rows.append(row)
        print(f"{protocol}/{case}: det={row['det']:.3f} "
              f"miou={row['miou']:.3f}", flush=True)
    with (args.output_root / "summary.json").open("w") as handle:
        json.dump({"protocol": "native_v27_synthetic_stress_v1",
                   "native_executable": str(args.native_exe.resolve()),
                   "native_sha256": sha256(args.native_exe),
                   "generator_seed": 20261008,
                   "runs": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
