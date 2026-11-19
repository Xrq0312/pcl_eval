"""Write a compact model table from native primitive JSON output."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.prefix + "_primitives.json", encoding="utf-8") as stream:
        metadata = json.load(stream)
    rows = []
    for model in sorted(metadata.get("models", []), key=lambda item: -item["points"]):
        coefficients = model["coefficients"]
        row = {"id": model["id"], "type": model["type"], "points": model["points"],
               "rmse": model["rmse"], "fit_threshold": model.get("fit_threshold"),
               "radius": "", "axis_x": "", "axis_y": "", "axis_z": ""}
        if model["type"] == "cylinder":
            row.update({"radius": coefficients[6], "axis_x": coefficients[3],
                        "axis_y": coefficients[4], "axis_z": coefficients[5]})
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
