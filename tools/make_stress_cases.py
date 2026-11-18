"""Deterministic perturbations for a synthetic primitive-fitting stress audit.

These are geometry/scale/density tests, not an independent real-world dataset.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from gen_synth_sample import main as make_base


def read_case(case):
    with (case / "points.txt").open() as handle:
        points = [tuple(map(float, line.split())) for line in handle]
    with (case / "gt_labels.csv").open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    with (case / "gt_primitives.json").open() as handle:
        primitives = json.load(handle)
    return points, labels, primitives


def write_case(case, points, labels, primitives, note):
    case.mkdir(parents=True, exist_ok=True)
    with (case / "points.txt").open("w") as handle:
        for x, y, z in points:
            handle.write(f"{x:.9f} {y:.9f} {z:.9f}\n")
    with (case / "gt_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=("point_index", "instance_id",
                                            "primitive_type"))
        writer.writeheader()
        for index, row in enumerate(labels):
            writer.writerow({"point_index": index,
                             "instance_id": row["instance_id"],
                             "primitive_type": row["primitive_type"]})
    with (case / "gt_primitives.json").open("w") as handle:
        json.dump(primitives, handle, indent=2)
    with (case / "metadata.json").open("w") as handle:
        json.dump({"dataset": "synthetic_stress", "num_points": len(points),
                   "note": note}, handle, indent=2)


def scaled_primitives(original, factor):
    result = json.loads(json.dumps(original))
    for primitive in result["primitives"]:
        for key in ("point", "center", "axis_point"):
            if key in primitive:
                primitive[key] = [factor * x for x in primitive[key]]
        for key in ("radius", "d"):
            if key in primitive:
                primitive[key] *= factor
    return result


def generate_cases(out):
    base = out / "nominal"
    make_base(str(base), seed=20261008)
    points, labels, primitives = read_case(base)
    for name, factor in (("scale_0p1", 0.1), ("scale_10", 10.0)):
        write_case(out / name,
                   [tuple(factor * x for x in p) for p in points], labels,
                   scaled_primitives(primitives, factor),
                   f"isotropic scale={factor}; same topology and point count")
    for name, fraction in (("density_25", 0.25), ("density_10", 0.10)):
        rng = random.Random(9301)
        indices = [i for i in range(len(points)) if rng.random() < fraction]
        write_case(out / name, [points[i] for i in indices],
                   [labels[i] for i in indices], primitives,
                   f"Bernoulli density={fraction}; unchanged physical scale")
    rng = random.Random(9302)
    noisy = [tuple(x + rng.gauss(0.0, 0.02) for x in p) for p in points]
    write_case(out / "noise_2cm", noisy, labels, primitives,
               "additional independent Gaussian coordinate noise sigma=0.02")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate_cases(args.out)


if __name__ == "__main__":
    main()
