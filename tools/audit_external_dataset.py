"""Inventory datasets that lack official point-level instance ground truth.

This tool intentionally does not emit unified gt_labels.csv files. It records
what is actually present so GCCao detections are not mistaken for annotations
and Primitive Object pose labels are not mistaken for point masks.
"""
import argparse
import json
from pathlib import Path


def relative(path, root):
    return path.relative_to(root).as_posix()


def ply_header(path):
    result = {"path": str(path), "bytes": path.stat().st_size}
    with open(path, "rb") as stream:
        lines = []
        for _ in range(200):
            line = stream.readline()
            if not line:
                raise RuntimeError(f"{path}: truncated PLY header")
            text = line.decode("ascii", errors="strict").strip()
            lines.append(text)
            if text == "end_header":
                break
        else:
            raise RuntimeError(f"{path}: PLY header exceeds 200 lines")
    if not lines or lines[0] != "ply":
        raise RuntimeError(f"{path}: not a PLY file")
    result["format"] = next((line.split(maxsplit=1)[1] for line in lines
                             if line.startswith("format ")), None)
    result["vertices"] = next((int(line.split()[2]) for line in lines
                               if line.startswith("element vertex ")), None)
    result["header"] = lines
    return result


def pcd_header(path):
    values = {"path": str(path), "bytes": path.stat().st_size}
    with open(path, "rb") as stream:
        lines = []
        for _ in range(200):
            line = stream.readline()
            if not line:
                raise RuntimeError(f"{path}: missing PCD DATA header")
            text = line.decode("ascii", errors="strict").strip()
            lines.append(text)
            if text.upper().startswith("DATA "):
                break
        else:
            raise RuntimeError(f"{path}: PCD header exceeds 200 lines")
    def field(name):
        prefix = name + " "
        return next((line[len(prefix):] for line in lines
                     if line.upper().startswith(prefix)), None)
    values.update({"fields": field("FIELDS"), "points": int(field("POINTS") or 0),
                   "width": int(field("WIDTH") or 0), "height": int(field("HEIGHT") or 0),
                   "data": field("DATA")})
    return values


def audit_gccao(root):
    ply_files = sorted(root.rglob("*.ply"))
    mat_files = sorted(root.rglob("*.mat"))
    text_files = sorted(root.rglob("*.txt"))
    return {
        "dataset": "gccao_cylinders_v2", "root": str(root.resolve()),
        "raw_ply": [{**ply_header(path), "path": relative(path, root)} for path in ply_files],
        "mat_files": [{"path": relative(path, root), "bytes": path.stat().st_size,
                       "status": "algorithm_output_or_intermediate_not_independent_gt"}
                      for path in mat_files],
        "text_files": [{"path": relative(path, root), "bytes": path.stat().st_size}
                       for path in text_files],
        "official_point_instance_gt": False,
        "official_primitive_parameter_gt": False,
        "quantitative_status": "blocked_pending_separate_manual_annotation",
    }


def parse_pose_label(path):
    records = []
    with open(path, encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            tokens = line.replace(",", " ").split()
            if not tokens:
                continue
            if len(tokens) != 11:
                raise RuntimeError(f"{path}:{line_number}: expected type + 10 pose/dimension values")
            try:
                numbers = [float(value) for value in tokens[1:]]
            except ValueError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid numeric label") from error
            records.append({"type": tokens[0], "position_xyz": numbers[0:3],
                            "quaternion_xyzw": numbers[3:7], "dimensions_xyz": numbers[7:10]})
    return records


def audit_primitive_object(root):
    pcd_files = sorted(root.rglob("*.pcd"))
    text_by_stem = {path.stem: path for path in root.rglob("*.txt")}
    samples, missing = [], []
    for pcd in pcd_files:
        label = text_by_stem.get(pcd.stem)
        entry = pcd_header(pcd)
        entry["path"] = relative(pcd, root)
        if label:
            entry["label_path"] = relative(label, root)
            entry["objects"] = parse_pose_label(label)
        else:
            entry["objects"] = []
            missing.append(relative(pcd, root))
        samples.append(entry)
    return {
        "dataset": "primitive_object_osf", "root": str(root.resolve()),
        "samples": samples, "pcd_count": len(pcd_files),
        "missing_pose_labels": missing,
        "official_object_pose_and_dimensions": True,
        "official_point_instance_gt": False,
        "point_iou_status": "not_applicable_without_declared_mask_protocol",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gccao", "primitive_object"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise RuntimeError(f"dataset root does not exist: {root}")
    result = audit_gccao(root) if args.dataset == "gccao" else audit_primitive_object(root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"dataset": result["dataset"], "output": str(output),
                      "files": len(result.get("samples", result.get("raw_ply", [])))}, indent=2))


if __name__ == "__main__":
    main()
