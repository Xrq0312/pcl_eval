"""Unit tests for the evaluation pipeline. Run: python test_evaluate.py"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import (iou_matrix, hungarian_match, parameter_errors,
                      line_line_distance, angle_between_deg, evaluate_sample)

FAILS = []


def check(name, cond, info=""):
    if cond:
        print(f"PASS {name}")
    else:
        FAILS.append(name)
        print(f"FAIL {name} {info}")


def test_iou_and_hungarian():
    gt = np.array([0, 0, 0, 1, 1, 1, 1, -1])
    pred = np.array([2, 2, 2, 2, 5, 5, 5, 5])
    g_ids, p_ids, iou, _, _ = iou_matrix(gt, pred)
    # GT0 {0,1,2}, pred2 {0,1,2,3} -> iou 3/4=0.75 ; GT1 {3..6} pred2/pred5
    check("iou_00", abs(iou[0, 0] - 0.75) < 1e-9, iou[0, 0])
    check("iou_gt1_pred5", abs(iou[1, 1] - 0.6) < 1e-9, iou[1, 1])
    pairs = hungarian_match(iou)
    check("hung_pairs", len(pairs) == 2)


def test_parameter_errors():
    # plane: n ~ -n equivalence; offset must be sign-aligned
    gt = {"type": "plane", "normal": [0, 0, 1], "d": 2.0}
    # plane z=2 flipped normal: (n,d)=(0,0,1,-2) -> (-n,-d)=(0,0,-1,2)
    e = parameter_errors(gt, "plane", [0, 0, -1, 2.0])   # same plane, flipped
    check("plane_sign_equiv_angle", e["normal_angle_deg"] < 1e-6, e)
    check("plane_sign_equiv_offset", e["offset_dist"] < 1e-6, e)
    e2 = parameter_errors(gt, "plane", [0, 0, 1, -2.2])
    check("plane_offset_0.2", abs(e2["offset_dist"] - 0.2) < 1e-6)
    # sphere
    gt_s = {"type": "sphere", "center": [1, 2, 3], "radius": 0.5}
    e3 = parameter_errors(gt_s, "sphere", [1.02, 2, 3, 0.48])
    check("sphere_center", abs(e3["center_dist"] - 0.02) < 1e-9)
    check("sphere_radius_rel", abs(e3["radius_rel_err"] - 0.04) < 1e-9)
    # cylinder: axis ~ -axis, any axis point valid
    gt_c = {"type": "cylinder", "axis_point": [0, 0, 0],
            "axis_direction": [0, 0, 2], "radius": 1.0}
    e4 = parameter_errors(gt_c, "cylinder", [0, 0, 5, 0, 0, -1, 1.01])
    check("cyl_axis_equiv_angle", e4["axis_angle_deg"] < 1e-6)
    check("cyl_axis_line_dist", e4["axis_line_dist"] < 1e-9, e4)
    check("cyl_radius", abs(e4["radius_abs_err"] - 0.01) < 1e-9)


def test_line_line_distance():
    d = line_line_distance([0, 0, 0], [0, 0, 1], [1, 0, 3], [0, 0, 1])
    check("parallel_lines_d1", abs(d - 1.0) < 1e-9, d)
    d2 = line_line_distance([0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0])
    check("skew_lines_d1", abs(d2 - 1.0) < 1e-9, d2)


def make_tmp_sample(tmp, npts=200):
    os.makedirs(tmp, exist_ok=True)
    # GT: plane z=0 (60 pts), plane z=2 (60 pts), cylinder r=1 z-axis (60),
    #     cone (20 pts -> non-target), none (no param support, 20 pts)
    rows, pts = [], []
    def add(inst, typ, gen, k):
        for _ in range(k):
            p = gen()
            pts.append(p)
            rows.append((len(pts) - 1, inst, typ))
    add(0, "plane", lambda: (np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0), 60)
    add(1, "plane", lambda: (np.random.uniform(-1, 1), np.random.uniform(-1, 1), 2.0), 60)
    def cyl():
        a = np.random.uniform(0, 2 * np.pi)
        return (np.cos(a), np.sin(a), np.random.uniform(0, 5))
    add(2, "cylinder", cyl, 60)
    def cone():
        t = np.random.uniform(0.5, 1.5)
        a = np.random.uniform(0, 2 * np.pi)
        return (t * np.cos(a), t * np.sin(a), t)
    add(3, "cone", cone, 20)
    add(4, "none", lambda: (np.random.uniform(5, 6), 0, 0), 20)
    with open(os.path.join(tmp, "points.txt"), "w") as f:
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    with open(os.path.join(tmp, "gt_labels.csv"), "w") as f:
        f.write("point_index,instance_id,primitive_type\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")
    prims = [
        {"instance_id": 0, "type": "plane", "normal": [0, 0, 1], "d": 0.0,
         "has_params": True},
        {"instance_id": 1, "type": "plane", "normal": [0, 0, 1], "d": 2.0,
         "has_params": True},
        {"instance_id": 2, "type": "cylinder", "axis_point": [0, 0, 0],
         "axis_direction": [0, 0, 1], "radius": 1.0, "has_params": True},
        {"instance_id": 3, "type": "cone", "has_params": False},
        {"instance_id": 4, "type": "none", "has_params": False},
    ]
    with open(os.path.join(tmp, "gt_primitives.json"), "w") as f:
        json.dump({"primitives": prims}, f)
    with open(os.path.join(tmp, "metadata.json"), "w") as f:
        json.dump({"dataset": "unit-test"}, f)
    return [p for p, _ in zip(pts, rows)], rows


def make_fake_pred(tmp_prefix, pts, rows):
    """near-perfect prediction: target GTs covered by their own models,
    non-target points left unassigned (-1)"""
    n = len(pts)
    with open(tmp_prefix + "_labels.csv", "w") as f:
        f.write("point_index,x,y,z,region_id,premerge_primitive_id,primitive_id\n")
        for i, p in enumerate(pts):
            inst = rows[i][1]
            pid = inst if inst <= 2 else -1
            f.write(f"{i},{p[0]},{p[1]},{p[2]},0,{pid},{pid}\n")
    models = [
        {"id": 0, "type": "plane", "points": 60, "regions": 1, "rmse": 0.0,
         "inlier_ratio": 1.0, "coefficients": [0, 0, 1, 0.0]},
        {"id": 1, "type": "plane", "points": 60, "regions": 1, "rmse": 0.0,
         "inlier_ratio": 1.0, "coefficients": [0, 0, 1, -2.0]},
        {"id": 2, "type": "cylinder", "points": 60, "regions": 1, "rmse": 0.0,
         "inlier_ratio": 1.0, "coefficients": [0, 0, 0, 0, 0, 1, 1.0]},
    ]
    meta = {"config": {"ablation": "D"}, "num_points": n, "num_regions": 1,
            "timing_seconds": {}, "premerge_models": models, "models": models,
            "failed_regions": []}
    with open(tmp_prefix + "_primitives.json", "w") as f:
        json.dump(meta, f)


def test_full_eval():
    np.random.seed(1)
    tmp = tempfile.mkdtemp(prefix="evtest_")
    try:
        pts, rows = make_tmp_sample(tmp)
        pref = os.path.join(tmp, "out")
        make_fake_pred(pref, pts, rows)
        res = evaluate_sample(tmp, pref)
        f = res["final_instance"]
        check("perfect_detection_target",
              abs(f["detection_target_rate"] - 1.0) < 1e-9,
              f["detection_target_rate"])
        check("perfect_typed", abs(f["detection_typed_rate"] - 1.0) < 1e-9)
        check("perfect_precision_target",
              abs(f["detection_target_precision"] - 1.0) < 1e-9)
        check("perfect_precision_typed",
              abs(f["detection_typed_precision"] - 1.0) < 1e-9)
        check("perfect_miou", abs(f["miou_target"] - 1.0) < 1e-9,
              f["miou_target"])
        check("perfect_ptacc", res["point_type_accuracy_target"] > 0.99)
        # typed must not exceed target detection rate
        check("typed_le_target",
              f["detection_typed_rate"] <= f["detection_target_rate"] + 1e-12)
        # best-IoU diag must be >= one-to-one target detection
        check("diag_ge_target",
              f["best_iou_target_diag_rate"] >=
              f["detection_target_rate"] - 1e-12)
        nt = res["non_target"]
        check("cone_not_detected",
              nt["cone_instances_detected_as_target"] == 0, nt)
        # parameter errors ~ 0 for perfect fits
        pe = res["parameter_errors"]
        check("param_errs_zero", len(pe) == 3 and all(
            all(abs(v) < 1e-6 for k, v in e.items()
                if isinstance(v, float) and k != "iou")
            for e in pe), pe)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_typed_le_target_imperfect():
    np.random.seed(2)
    tmp = tempfile.mkdtemp(prefix="evtest2_")
    try:
        pts, rows = make_tmp_sample(tmp)
        pref = os.path.join(tmp, "out")
        make_fake_pred(pref, pts, rows)
        # swap cylinder model type to sphere -> type-mismatched detection
        with open(pref + "_primitives.json") as f:
            meta = json.load(f)
        meta["models"][2]["type"] = "sphere"
        meta["models"][2]["coefficients"] = [0, 0, 2.5, 2.0]
        with open(pref + "_primitives.json", "w") as f:
            json.dump(meta, f)
        res = evaluate_sample(tmp, pref)
        f = res["final_instance"]
        check("mismatch_target_ge_typed",
              f["detection_target_rate"] > f["detection_typed_rate"],
              (f["detection_target_rate"], f["detection_typed_rate"]))
        check("mismatch_precision_target_ge_typed",
              f["detection_target_precision"] > f["detection_typed_precision"],
              (f["detection_target_precision"], f["detection_typed_precision"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_iou_and_hungarian()
    test_parameter_errors()
    test_line_line_distance()
    test_full_eval()
    test_typed_le_target_imperfect()
    print("\n" + ("ALL PASSED" if not FAILS else f"FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)
