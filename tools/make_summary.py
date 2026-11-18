"""Final ablation summary for C / D / D1..D5 with the corrected metric
conventions (target-only detection denominators, one-to-one matching,
non-target statistics, rollback counts)."""
import csv
import glob
import json
import os
import sys

import numpy as np

TARGET = ("plane", "sphere", "cylinder")
MODES = ["C", "D1", "D2", "D3", "D4", "D5", "D"]


def collect(results_root="results/bagsfit_dxd"):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_root, "*", "*",
                                            "evaluation.json"))):
        parts = p.replace("\\", "/").split("/")
        sample, mode = parts[-3], parts[-2]
        r = json.load(open(p))
        f = r["final_instance"]
        prf = r.get("point_type_prf", {})
        nt = r.get("non_target", {})
        rollbacks = 0
        absorbed = 0
        opfile = os.path.join(os.path.dirname(p), "out_operations.csv")
        if os.path.exists(opfile):
            for o in csv.DictReader(open(opfile)):
                if o["operation_type"] == "ROLLBACK" and o["accepted"] == "0":
                    rollbacks += 1
                if o["operation_type"] in ("ABSORB", "FRAGMENT_ABSORB") \
                        and o["accepted"] == "1":
                    absorbed += int(o["absorbed_points"] or 0)
        rows.append(dict(
            sample=sample, mode=mode,
            det=f["detection_target_rate"],
            typed=f["detection_typed_rate"],
            miou=f["miou_target"],
            ptacc=r["point_type_accuracy_target"],
            pF1=prf.get("plane", {}).get("f1"),
            sF1=prf.get("sphere", {}).get("f1"),
            cF1=prf.get("cylinder", {}).get("f1"),
            unass=r["failures"]["unassigned_ratio"],
            cone_pts=nt.get("cone_assigned_ratio"),
            cone_inst=nt.get("cone_instances_detected_as_target"),
            none_pts=nt.get("none_assigned_ratio"),
            wall=r.get("wall_seconds", 0),
            rollbacks=rollbacks, absorbed=absorbed,
            params=r.get("parameter_errors", []),
        ))
    return rows


def avg(sel, key):
    v = [r[key] for r in sel if r[key] is not None]
    return sum(v) / len(v) if v else float("nan")


def main():
    rows = collect()
    out_dir = os.path.dirname(rows[0]["params"] and "." or ".")
    with open("results/bagsfit_dxd_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys()
                                           if k != "params"])
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "params"})
    # drop rows missing header fix: rewrite properly
    keys = [k for k in rows[0].keys() if k != "params"]
    with open("results/bagsfit_dxd_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "params"})

    print("=== 10-sample means (all samples incl. tuning sample 000_00000) ===")
    print("mode  det   typed miou   ptacc  pF1  sF1  cF1  unass  cone% coneInst "
          "none%  rollbacks absPts  wall")
    for m in MODES:
        sel = [r for r in rows if r["mode"] == m]
        if not sel:
            continue
        print(f"{m:5s} {avg(sel,'det'):.2f}  {avg(sel,'typed'):.2f} "
              f"{avg(sel,'miou'):.3f}  {avg(sel,'ptacc'):.2f}  "
              f"{avg(sel,'pF1'):.2f} {avg(sel,'sF1'):.2f} {avg(sel,'cF1'):.2f}  "
              f"{avg(sel,'unass'):.2f}   {avg(sel,'cone_pts'):.2f} "
              f"{avg(sel,'cone_inst'):.1f}   {avg(sel,'none_pts'):.2f}  "
              f"{sum(r['rollbacks'] for r in sel):3d}    "
              f"{sum(r['absorbed'] for r in sel):5d}   {avg(sel,'wall'):.1f}s")

    # independent subset
    print("\n=== independent 9 samples (excl. tuning sample 000_00000) ===")
    print("mode  det   typed miou   ptacc  unass")
    for m in MODES:
        sel = [r for r in rows if r["mode"] == m and r["sample"] != "000_00000"]
        if not sel:
            continue
        print(f"{m:5s} {avg(sel,'det'):.2f}  {avg(sel,'typed'):.2f} "
              f"{avg(sel,'miou'):.3f}  {avg(sel,'ptacc'):.2f}  "
              f"{avg(sel,'unass'):.2f}")

    # per-GT transitions C -> D5 (target only, one-to-one)
    print("\n=== target GT transitions C -> D5 (one-to-one IoU>=0.5) ===")
    def load_mode(sample, mode):
        p = os.path.join("results/bagsfit_dxd", sample, mode,
                         "evaluation.json")
        if not os.path.exists(p):
            return None
        r = json.load(open(p))
        return {(m["gt"], m["gt_type"]): m for m in
                r["final_instance"]["matched_pairs"]}
    trans = {"lost": [], "gained": [], "kept": 0}
    for sample in sorted(set(r["sample"] for r in rows)):
        c = load_mode(sample, "C")
        d5 = load_mode(sample, "D5")
        if not c or not d5:
            continue
        gtp = os.path.join("samples/bagsfit", sample, "gt_primitives.json")
        prims = {p["instance_id"]: p for p in json.load(open(gtp))["primitives"]}
        c_map = {g: m for (g, t), m in c.items() if t in TARGET}
        d5_map = {g: m for (g, t), m in d5.items() if t in TARGET}
        for g in c_map:
            cdet = c_map[g]["iou"] >= 0.5
            ddet = d5_map.get(g, {}).get("iou", 0) >= 0.5
            if cdet and not ddet:
                trans["lost"].append((sample, g, prims[g]["type"],
                                      c_map[g]["iou"],
                                      c_map[g]["pred_type"],
                                      d5_map.get(g, {}).get("iou", 0),
                                      d5_map.get(g, {}).get("pred_type", "-")))
            elif cdet and ddet:
                trans["kept"] += 1
        for g in d5_map:
            if d5_map[g]["iou"] >= 0.5 and c_map.get(g, {}).get("iou", 0) < 0.5:
                trans["gained"].append((sample, g, prims[g]["type"],
                                        c_map.get(g, {}).get("iou", 0),
                                        c_map.get(g, {}).get("pred_type", "-"),
                                        d5_map[g]["iou"],
                                        d5_map[g]["pred_type"]))
    print(f"kept={trans['kept']}  lost={len(trans['lost'])}  "
          f"gained={len(trans['gained'])}")
    for x in trans["lost"]:
        print(f"  LOST  {x[0]}/GT{x[1]} {x[2]}: C {x[3]:.3f}[{x[4]}] -> "
              f"D5 {x[5]:.3f}[{x[6]}]")
    for x in trans["gained"]:
        print(f"  GAIN  {x[0]}/GT{x[1]} {x[2]}: C {x[3]:.3f}[{x[4]}] -> "
              f"D5 {x[5]:.3f}[{x[6]}]")

    # focus cases: 002_00000/GT0, 002_00018/GT8, 006_00000/GT9
    print("\n=== focus cases (C / D3 / D4 / D5 best one-to-one IoU) ===")
    for sample, g in [("002_00000", 0), ("002_00018", 8), ("006_00000", 9)]:
        line = [f"{sample}/GT{g}"]
        for mode in ["C", "D3", "D4", "D5"]:
            mm = load_mode(sample, mode)
            hit = [m for (gg, t), m in (mm or {}).items() if gg == g]
            iou = hit[0]["iou"] if hit else 0.0
            pt = hit[0]["pred_type"] if hit else "-"
            line.append(f"{mode}={iou:.3f}[{pt}]")
        print("  " + " ".join(line))


if __name__ == "__main__":
    main()
