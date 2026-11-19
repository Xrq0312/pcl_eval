"""Report GT residual and estimated-normal compatibility by primitive instance."""
import argparse
import csv
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--pred-prefix", required=True)
    args = parser.parse_args()

    labels = {}
    types = {}
    with open(args.sample_dir + "/gt_labels.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[int(row["point_index"])] = int(row["instance_id"])
            types[int(row["instance_id"])] = row["primitive_type"]
    points, normals, instance = [], [], []
    with open(args.pred_prefix + "_labels.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            index = int(row["point_index"])
            points.append([float(row[k]) for k in ("x", "y", "z")])
            normals.append([float(row[k]) for k in ("normal_x", "normal_y", "normal_z")])
            instance.append(labels[index])
    points = np.asarray(points)
    normals = np.asarray(normals)
    instance = np.asarray(instance)
    with open(args.sample_dir + "/gt_primitives.json") as stream:
        primitives = json.load(stream)["primitives"]
    print("id,type,n,residual_p50,residual_p95,normal_dot_p50,normal_dot_p05")
    for primitive in primitives:
        primitive_id = primitive["instance_id"]
        mask = instance == primitive_id
        p, n = points[mask], normals[mask]
        if primitive["type"] == "plane":
            direction = np.asarray(primitive["normal"], dtype=float)
            direction /= np.linalg.norm(direction)
            origin = np.asarray(primitive["point"], dtype=float)
            residual = np.abs((p - origin) @ direction)
            alignment = np.abs(n @ direction)
        elif primitive["type"] == "sphere":
            center = np.asarray(primitive["center"], dtype=float)
            radial = p - center
            distance = np.linalg.norm(radial, axis=1)
            residual = np.abs(distance - float(primitive["radius"]))
            alignment = np.abs(np.sum(n * radial, axis=1) / np.maximum(distance, 1e-12))
        elif primitive["type"] == "cylinder":
            origin = np.asarray(primitive["axis_point"], dtype=float)
            axis = np.asarray(primitive["axis_direction"], dtype=float)
            axis /= np.linalg.norm(axis)
            delta = p - origin
            radial = delta - np.outer(delta @ axis, axis)
            distance = np.linalg.norm(radial, axis=1)
            residual = np.abs(distance - float(primitive["radius"]))
            alignment = np.abs(np.sum(n * radial, axis=1) / np.maximum(distance, 1e-12))
        else:
            continue
        print(f"{primitive_id},{primitive['type']},{mask.sum()},"
              f"{np.quantile(residual, .5):.6g},{np.quantile(residual, .95):.6g},"
              f"{np.quantile(alignment, .5):.6g},{np.quantile(alignment, .05):.6g}")


if __name__ == "__main__":
    main()
