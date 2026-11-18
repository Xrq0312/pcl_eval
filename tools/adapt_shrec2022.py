"""Convert an official SHREC2022 test subset to the unified audit format."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


TYPE_NAMES = {1: "plane", 2: "cylinder", 3: "sphere", 4: "cone", 5: "torus"}
ARTIFACTS = {
    "A0": "clean",
    "A1": "uniform_noise",
    "A2": "gaussian_noise",
    "A3": "undersampling",
    "A4": "missing_parts",
    "A5": "uniform_noise+undersampling",
    "A6": "gaussian_noise+undersampling",
    "A7": "uniform_noise+missing_parts",
    "A8": "gaussian_noise+missing_parts",
    "A9": "local_deformation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_from_info(values: list[float]) -> str:
    uniform = values[0] != 0
    gaussian = values[3] != 0
    undersampled = values[6] != 0
    missing = values[8] != 0
    deformed = values[10] != 0
    if deformed:
        return "A9"
    key = (uniform, gaussian, undersampled, missing)
    mapping = {
        (False, False, False, False): "A0",
        (True, False, False, False): "A1",
        (False, True, False, False): "A2",
        (False, False, True, False): "A3",
        (False, False, False, True): "A4",
        (True, False, True, False): "A5",
        (False, True, True, False): "A6",
        (True, False, False, True): "A7",
        (False, True, False, True): "A8",
    }
    if key not in mapping:
        raise ValueError(f"unsupported SHREC artifact flags: {key}")
    return mapping[key]


def primitive_from_gt(values: list[float]) -> dict:
    code = int(values[0])
    primitive_type = TYPE_NAMES[code]
    result = {"instance_id": 0, "type": primitive_type,
              "is_target": code in (1, 2, 3), "has_params": code in (1, 2, 3)}
    if code == 1:
        if len(values) != 7:
            raise ValueError(f"plane GT needs 7 values, got {len(values)}")
        result.update(normal=values[1:4], point=values[4:7])
    elif code == 2:
        if len(values) != 8:
            raise ValueError(f"cylinder GT needs 8 values, got {len(values)}")
        result.update(radius=values[1], axis_direction=values[2:5], axis_point=values[5:8])
    elif code == 3:
        if len(values) != 5:
            raise ValueError(f"sphere GT needs 5 values, got {len(values)}")
        result.update(radius=values[1], center=values[2:5])
    return result


def convert(source: Path, output: Path, commit: str) -> dict:
    point_files = sorted((source / "pointCloud").glob("pointCloud*.txt"),
                         key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)))
    report = []
    for points_path in point_files:
        sample_id = int(re.search(r"(\d+)$", points_path.stem).group(1))
        gt_path = source / "GTpointCloud" / f"GTpointCloud{sample_id}.txt"
        info_path = source / "infoPointCloud" / f"infoPointCloud{sample_id}.txt"
        gt = [float(line) for line in gt_path.read_text().splitlines() if line.strip()]
        info = [float(line) for line in info_path.read_text().splitlines() if line.strip()]
        if len(info) not in (12, 13):
            raise ValueError(f"{info_path}: expected 12 or 13 values, got {len(info)}")
        if len(info) == 12:
            info.append(0.0)
        primitive = primitive_from_gt(gt)
        if not primitive["is_target"]:
            continue
        artifact = artifact_from_info(info)
        points = []
        for line_number, line in enumerate(points_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            xyz = [float(value) for value in line.replace(",", " ").split()]
            if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
                raise ValueError(f"{points_path}:{line_number}: invalid point")
            points.append(xyz)
        name = f"pointCloud{sample_id:04d}_{artifact}_{primitive['type']}"
        sample = output / name
        sample.mkdir(parents=True, exist_ok=True)
        (sample / "points.txt").write_text(
            "".join("{:.17g} {:.17g} {:.17g}\n".format(*point) for point in points),
            encoding="utf-8")
        with (sample / "gt_labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["point_index", "instance_id", "primitive_type"])
            writer.writerows((index, 0, primitive["type"]) for index in range(len(points)))
        (sample / "gt_primitives.json").write_text(
            json.dumps({"primitives": [primitive]}, indent=2) + "\n", encoding="utf-8")
        bbox = {axis: [min(point[i] for point in points), max(point[i] for point in points)]
                for i, axis in enumerate("xyz")}
        metadata = {
            "dataset": "SHREC2022",
            "split": "official_test",
            "source_id": sample_id,
            "source_repository": "https://github.com/chiararomanengo/SHREC2022",
            "source_commit": commit,
            "artifact": artifact,
            "artifact_description": ARTIFACTS[artifact],
            "primitive_type": primitive["type"],
            "num_points": len(points),
            "num_primitives": 1,
            "target_primitives": 1,
            "bbox": bbox,
            "info_vector": info,
            "source_sha256": {
                points_path.name: sha256(points_path),
                gt_path.name: sha256(gt_path),
                info_path.name: sha256(info_path),
            },
        }
        (sample / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        report.append({"sample": name, "source_id": sample_id,
                       "type": primitive["type"], "artifact": artifact,
                       "points": len(points),
                       "source_sha256": metadata["source_sha256"]})
    (output / "adapt_report.json").write_text(
        json.dumps({
            "dataset": "SHREC2022 official test set",
            "source_repository": "https://github.com/chiararomanengo/SHREC2022",
            "source_commit": commit,
            "selection": "median numeric source ID in each plane/cylinder/sphere x A0-A9 stratum, fixed before fitting",
            "samples": report,
        }, indent=2) + "\n",
        encoding="utf-8")
    print(f"converted {len(report)} samples, {sum(row['points'] for row in report)} points")
    return {"samples": report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    convert(args.source_root, args.output_root, args.source_commit)


if __name__ == "__main__":
    main()
