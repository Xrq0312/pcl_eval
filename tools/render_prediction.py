"""Render deterministic orthographic previews from native prediction outputs."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--only-type", choices=["plane", "sphere", "cylinder"])
    args = parser.parse_args()

    prefix = Path(args.prefix)
    with open(str(prefix) + "_primitives.json", encoding="utf-8") as stream:
        metadata = json.load(stream)
    count = int(metadata["num_points"])
    stride = max(1, math.ceil(count / args.max_points))
    points, labels = [], []
    with open(str(prefix) + "_labels.csv", newline="", encoding="utf-8") as stream:
        for row_number, row in enumerate(csv.DictReader(stream)):
            if row_number % stride:
                continue
            points.append([float(row["x"]), float(row["y"]), float(row["z"])])
            labels.append(int(row["primitive_id"]))
    points = np.asarray(points)
    labels = np.asarray(labels)
    models = metadata.get("models", [])
    if args.only_type:
        keep = {int(model["id"]) for model in models if model["type"] == args.only_type}
        labels = np.asarray([label if label in keep else -1 for label in labels])
    colors = np.full((len(points), 4), [0.35, 0.35, 0.35, 0.20])
    cmap = plt.get_cmap("turbo")
    denominator = max(1, len(models) - 1)
    assigned = labels >= 0
    colors[assigned] = np.asarray([cmap((label * 0.61803398875) % 1.0)
                                   for label in labels[assigned]])
    colors[assigned, 3] = 0.75

    views = [(20, -65, "perspective"), (90, -90, "top"), (0, -90, "front")]
    fig = plt.figure(figsize=(18, 6), dpi=160)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    half = max((points.max(axis=0) - points.min(axis=0)).max() / 2.0, 1e-6)
    for index, (elevation, azimuth, title) in enumerate(views, 1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors,
                     s=0.18, linewidths=0, depthshade=False, rasterized=True)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_zlim(center[2] - half, center[2] + half)
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(title)
        axis.set_axis_off()
    type_counts = {}
    for model in models:
        type_counts[model["type"]] = type_counts.get(model["type"], 0) + 1
    assigned_ratio = float(np.mean(labels >= 0))
    timing = metadata.get("timing_seconds", {}).get("total", 0.0)
    distance = metadata.get("config", {}).get("distance", 0.0)
    selection = f" | showing {args.only_type}" if args.only_type else ""
    fig.suptitle(
        f"{prefix.parent.name}: {count:,} points | models {type_counts} | "
        f"sampled assigned {assigned_ratio:.1%}{selection} | distance {distance:.5g} | {timing:.2f}s",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
