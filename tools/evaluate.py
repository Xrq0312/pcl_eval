"""Unified evaluation for the PCL primitive fitting pipeline.

Compares predictions (labels CSV + primitives JSON produced by
primitive_fit) against ground truth in the unified sample format:

sample/
  points.pcd (or .txt)
  gt_labels.csv     point_index,instance_id,primitive_type
  gt_primitives.json  instance parameter ground truth
  metadata.json

Outputs evaluation.json per sample, or an aggregate CSV over many samples.

Type conventions:
  target types:   plane, sphere, cylinder
  non-target:     cone (and any other type string) - tracked but excluded
                  from target metrics; points must NOT be silently dropped.
  no GT instance: instance_id = -1 (e.g. background / unknown surfaces)

Sign/parameter equivalences handled explicitly:
  plane normals   n ~ -n            (compare via |n1.n2|, offset |c1-c2|)
  cylinder axes   v ~ -v            (angle via |v1.v2|)
  cylinder axis   any point on the axis is valid (use line-line distance)
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

TARGET_TYPES = ("plane", "sphere", "cylinder")
IOU_DETECT = 0.5          # instance detection threshold
EPS = 1e-12


# ---------------------------------------------------------------- GT loading
def load_gt(sample_dir):
    labels_path = os.path.join(sample_dir, "gt_labels.csv")
    prim_path = os.path.join(sample_dir, "gt_primitives.json")
    rows = []
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["point_index"]), int(r["instance_id"]),
                         r["primitive_type"]))
    rows.sort(key=lambda t: t[0])
    gt_inst = np.array([r[1] for r in rows], dtype=int)
    gt_type = [r[2] for r in rows]
    with open(prim_path) as f:
        gt_prims = json.load(f)["primitives"]
    return gt_inst, gt_type, gt_prims


# ------------------------------------------------------------ prediction load
def load_pred(prefix):
    """prefix: path passed to primitive_fit (its *_labels.csv etc.)."""
    labels = {}
    with open(prefix + "_labels.csv", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            i = int(r["point_index"])
            labels[i] = (int(r["region_id"]),
                         int(r["premerge_primitive_id"]),
                         int(r["primitive_id"]))
    n = max(labels) + 1
    region = np.full(n, -1, dtype=int)
    premerge = np.full(n, -1, dtype=int)
    final = np.full(n, -1, dtype=int)
    for i, (a, b, c) in labels.items():
        region[i], premerge[i], final[i] = a, b, c
    with open(prefix + "_primitives.json") as f:
        meta = json.load(f)
    return region, premerge, final, meta


def load_pred_observations(prefix):
    """Read detector coordinates/normals used to reproduce proposal support."""
    points, normals = {}, {}
    with open(prefix + "_labels.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"x", "y", "z"}.issubset(fields):
            return None, None
        have_normals = {"normal_x", "normal_y", "normal_z"}.issubset(fields)
        for row in reader:
            index = int(row["point_index"])
            points[index] = [float(row["x"]), float(row["y"]), float(row["z"])]
            if have_normals:
                normals[index] = [float(row["normal_x"]), float(row["normal_y"]),
                                  float(row["normal_z"])]
    if not points:
        return None, None
    ordered_points = np.asarray([points[index] for index in range(max(points) + 1)])
    ordered_normals = None
    if len(normals) == len(points):
        ordered_normals = np.asarray(
            [normals[index] for index in range(max(points) + 1)])
    return ordered_points, ordered_normals


def pred_type_of(meta, stage):
    key = "models" if stage == "final" else "premerge_models"
    return [m["type"] for m in meta.get(key, [])]


# ------------------------------------------------------------------- geometry
def v3(x):
    a = np.asarray(x, dtype=float)
    return a / max(np.linalg.norm(a), EPS)


def plane_nc(p):
    """primitive dict -> unit normal, offset c (n.p = c)."""
    n = v3(p["normal"])
    if "d" in p:
        c = float(p["d"])
        if "point" in p:  # d stored as plane offset from point along normal
            c = float(np.dot(n, np.asarray(p["point"], float))) + float(p["d"])
    else:
        c = float(np.dot(n, np.asarray(p["point"], float)))
    return n, c


def point_surface_distances(pts, p):
    """Signed abs distance from Nx3 points to GT primitive p."""
    if p["type"] == "plane":
        n, c = plane_nc(p)
        return np.abs(pts @ n - c)
    if p["type"] == "sphere":
        ctr = np.asarray(p["center"], float)
        r = float(p["radius"])
        return np.abs(np.linalg.norm(pts - ctr, axis=1) - r)
    if p["type"] == "cylinder":
        p0 = np.asarray(p["axis_point"], float)
        u = v3(p["axis_direction"])
        r = float(p["radius"])
        d = pts - p0
        perp = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
        return np.abs(perp - r)
    return np.full(len(pts), np.nan)


def pred_surface_distances(pts, ptype, coeff):
    c = np.asarray(coeff, dtype=float)
    if ptype == "plane":
        n = v3(c[:3])
        return np.abs(pts @ n + c[3])
    if ptype == "sphere":
        return np.abs(np.linalg.norm(pts - c[:3], axis=1) - c[3])
    if ptype == "cylinder":
        u = v3(c[3:6])
        d = pts - c[:3]
        perp = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
        return np.abs(perp - c[6])
    return np.full(len(pts), np.nan)


def proposal_support(model, points, normals, config):
    """Reconstruct the detector's geometric/normal support for a model."""
    ptype = model["type"]
    coeff = np.asarray(model["coefficients"], dtype=float)
    threshold = float(model.get("fit_threshold") or config.get("distance", 0.0))
    if threshold <= 0:
        return np.zeros(len(points), dtype=bool)
    support = pred_surface_distances(points, ptype, coeff) <= threshold
    if normals is None:
        return support
    normal_norm = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(normal_norm) & (normal_norm > EPS)
    unit_normals = normals / np.maximum(normal_norm[:, None], EPS)
    if ptype == "plane":
        expected = np.tile(v3(coeff[:3]), (len(points), 1))
    elif ptype == "sphere":
        expected = points - coeff[:3]
    elif ptype == "cylinder":
        axis = v3(coeff[3:6])
        delta = points - coeff[:3]
        expected = delta - np.outer(delta @ axis, axis)
    else:
        return np.zeros(len(points), dtype=bool)
    expected_norm = np.linalg.norm(expected, axis=1)
    expected_unit = expected / np.maximum(expected_norm[:, None], EPS)
    alignment = np.abs(np.sum(expected_unit * unit_normals, axis=1))
    minimum = math.cos(math.radians(float(config.get("normal_angle_deg", 25.0))))
    support &= (~valid) | ((expected_norm > EPS) & (alignment >= minimum))
    return support


def best_typed_model_iou(gt_mask, gt_type, models, points, normals, config):
    best = 0.0
    for model in models:
        if model.get("type") != gt_type:
            continue
        mask = proposal_support(model, points, normals, config)
        intersection = int(np.count_nonzero(gt_mask & mask))
        union = int(np.count_nonzero(gt_mask | mask))
        if union:
            best = max(best, intersection / union)
    return best


def best_label_iou(gt_mask, gt_type, labels, models, typed=True):
    best = 0.0
    for model_id, model in enumerate(models):
        if typed and model.get("type") != gt_type:
            continue
        mask = labels == model_id
        intersection = int(np.count_nonzero(gt_mask & mask))
        union = int(np.count_nonzero(gt_mask | mask))
        if union:
            best = max(best, intersection / union)
    return best


def attribute_failures(gt_inst, gt_by_id, premerge, final, meta, prefix,
                       proposal_iou=0.30, detection_iou=0.50):
    diagnostics = meta.get("diagnostics")
    if not diagnostics:
        return {"available": False, "reason": "detector did not export lifecycle diagnostics"}
    points, normals = load_pred_observations(prefix)
    if points is None or len(points) != len(gt_inst):
        return {"available": False, "reason": "prediction observations unavailable"}
    config = meta.get("config", {})
    stage_keys = ("screened_proposals", "refit_failed_proposals",
                  "filtered_proposals", "proposals", "selected_models")
    stages = {key: diagnostics.get(key, []) for key in stage_keys}
    premerge_models = meta.get("premerge_models", [])
    final_models = meta.get("models", [])
    records = []
    counts = {}
    for instance_id in sorted(set(gt_inst.tolist()) - {-1}):
        primitive = gt_by_id[instance_id]
        if primitive["type"] not in TARGET_TYPES:
            continue
        gt_mask = gt_inst == instance_id
        best = {
            key: best_typed_model_iou(gt_mask, primitive["type"], models,
                                      points, normals, config)
            for key, models in stages.items()
        }
        best["premerge"] = best_label_iou(
            gt_mask, primitive["type"], premerge, premerge_models)
        best["final_typed"] = best_label_iou(
            gt_mask, primitive["type"], final, final_models)
        best["final_any_type"] = best_label_iou(
            gt_mask, primitive["type"], final, final_models, typed=False)
        if best["final_typed"] >= detection_iou:
            outcome = "detected"
        elif best["premerge"] >= detection_iou:
            outcome = "final_merge_or_connectivity_failure"
        elif best["selected_models"] >= proposal_iou:
            outcome = "assignment_or_refit_failure"
        elif best["proposals"] >= proposal_iou:
            outcome = "candidate_selection_failure"
        elif best["screened_proposals"] >= proposal_iou:
            outcome = ("candidate_refit_failure" if
                       best["refit_failed_proposals"] >= proposal_iou else
                       "candidate_filter_failure")
        else:
            outcome = "candidate_not_generated"
        if (outcome != "detected" and best["final_any_type"] >= detection_iou and
                best["final_typed"] < detection_iou):
            outcome = "wrong_type_competition"
        counts[outcome] = counts.get(outcome, 0) + 1
        records.append({
            "instance_id": int(instance_id), "type": primitive["type"],
            "points": int(np.count_nonzero(gt_mask)), "outcome": outcome,
            "best_typed_iou": {key: round(float(value), 6)
                               for key, value in best.items()},
        })
    return {
        "available": True,
        "proposal_iou_threshold": proposal_iou,
        "detection_iou_threshold": detection_iou,
        "counts": dict(sorted(counts.items())),
        "records": records,
        "note": ("Proposal support is reconstructed from exported coefficients, "
                 "distance thresholds and detector normals; GT is evaluation-only."),
    }


def line_line_distance(p1, u1, p2, u2):
    """Shortest distance between lines (p1,u1) and (p2,u2), unit dirs."""
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    w = p2 - p1
    cross = np.cross(u1, u2)
    nc = np.linalg.norm(cross)
    if nc < 1e-6:  # (near-)parallel: perpendicular component
        return float(np.linalg.norm(w - np.dot(w, u1) * u1))
    return float(abs(np.dot(w, cross)) / nc)


def angle_between_deg(a, b):
    d = abs(float(np.dot(v3(a), v3(b))))
    return math.degrees(math.acos(min(1.0, d)))


def parameter_errors(gt, ptype, coeff):
    """Error metrics between one GT primitive and one fitted model."""
    out = {}
    if ptype == "plane":
        gn, gc = plane_nc(gt)
        pn = v3(np.asarray(coeff[:3], float))
        pc = -float(coeff[3])          # PCL: n.p + d = 0
        # sign alignment: n ~ -n is the same plane, flip prediction so that
        # the normals agree before comparing offsets
        if float(np.dot(gn, pn)) < 0:
            pn, pc = -pn, -pc
        out["normal_angle_deg"] = angle_between_deg(gn, pn)
        out["offset_dist"] = abs(gc - pc)
    elif ptype == "sphere":
        gc = np.asarray(gt["center"], float)
        out["center_dist"] = float(np.linalg.norm(np.asarray(coeff[:3]) - gc))
        gr = float(gt["radius"])
        out["radius_abs_err"] = abs(float(coeff[3]) - gr)
        out["radius_rel_err"] = abs(float(coeff[3]) - gr) / max(gr, EPS)
    elif ptype == "cylinder":
        gu = v3(gt["axis_direction"])
        gp = np.asarray(gt["axis_point"], float)
        pu = v3(np.asarray(coeff[3:6], float))
        out["axis_angle_deg"] = angle_between_deg(gu, pu)
        out["axis_line_dist"] = line_line_distance(coeff[:3], pu, gp, gu)
        gr = float(gt["radius"])
        out["radius_abs_err"] = abs(float(coeff[6]) - gr)
        out["radius_rel_err"] = abs(float(coeff[6]) - gr) / max(gr, EPS)
    return out


# ------------------------------------------------------------------- matching
def iou_matrix(gt_inst, pred_labels):
    g_ids = sorted(set(gt_inst.tolist()) - {-1})
    p_ids = sorted(set(pred_labels.tolist()) - {-1})
    iou = np.zeros((len(g_ids), len(p_ids)))
    g_index = {g: i for i, g in enumerate(g_ids)}
    p_index = {p: i for i, p in enumerate(p_ids)}
    g_sets = {g: (gt_inst == g) for g in g_ids}
    p_sets = {p: (pred_labels == p) for p in p_ids}
    for g in g_ids:
        gs = g_sets[g]
        for p in p_ids:
            ps = p_sets[p]
            inter = np.count_nonzero(gs & ps)
            union = np.count_nonzero(gs | ps)
            if union:
                iou[g_index[g], p_index[p]] = inter / union
    return g_ids, p_ids, iou, g_sets, p_sets


def hungarian_match(iou):
    if iou.size == 0:
        return []
    if not HAVE_SCIPY:  # greedy fallback
        pairs = []
        used_g, used_p = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None),
                                           iou.shape))[0]
        for gi, pi in order:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            pairs.append((int(gi), int(pi)))
        return pairs
    gi, pi = linear_sum_assignment(-iou)
    return list(zip(gi.tolist(), pi.tolist()))


# ---------------------------------------------------------------- evaluation
def evaluate_sample(sample_dir, pred_prefix, iou_detect=IOU_DETECT):
    gt_inst, gt_type, gt_prims = load_gt(sample_dir)
    region, premerge, final, meta = load_pred(pred_prefix)
    if len(gt_inst) != len(final):
        raise RuntimeError(f"label length mismatch: gt {len(gt_inst)} vs pred {len(final)}")

    gt_by_id = {p["instance_id"]: p for p in gt_prims}
    n = len(final)
    res = {"sample": os.path.basename(sample_dir.rstrip("/\\")),
           "num_points": n,
           "timing_seconds": meta.get("timing_seconds", {}),
           "config": meta.get("config", {})}
    res["failure_attribution"] = attribute_failures(
        gt_inst, gt_by_id, premerge, final, meta, pred_prefix,
        detection_iou=iou_detect)

    # ------- stage-wise instance metrics (premerge = B, final = D)
    for stage, labels in (("premerge", premerge), ("final", final)):
        types = pred_type_of(meta, stage)
        g_ids, p_ids, iou, g_sets, p_sets = iou_matrix(gt_inst, labels)
        pairs = hungarian_match(iou)
        matched = [(g_ids[gi], p_ids[pi], iou[gi, pi]) for gi, pi in pairs
                   if iou[gi, pi] > 0]
        g_type = {g: gt_by_id[g]["type"] for g in g_ids}

        # detection: IoU >= threshold; report both with/without type agreement
        det_all = [g for g, p, i in matched if i >= iou_detect]
        det_typed = [g for g, p, i in matched
                     if i >= iou_detect and g_type[g] == types[p]
                     and g_type[g] in TARGET_TYPES]
        # one-to-one detection restricted to target GTs (main metric):
        det_target = [g for g, p, i in matched
                      if i >= iou_detect and g_type[g] in TARGET_TYPES]
        n_target_gt = sum(1 for g in g_ids if g_type[g] in TARGET_TYPES)
        n_target_pred = sum(1 for p in p_ids if types[p] in TARGET_TYPES)

        # mIoU over GT instances (unmatched counts as 0) - target types only
        best_iou = {g: 0.0 for g in g_ids}
        for g, p, i in matched:
            best_iou[g] = max(best_iou[g], i)
        # best-IoU DIAGNOSTIC (not one-to-one): max IoU over all predictions
        best_diag = {g: 0.0 for g in g_ids}
        for gj, g in enumerate(g_ids):
            for pj in range(len(p_ids)):
                best_diag[g] = max(best_diag[g], iou[gj, pj])
        res[f"{stage}_instance"] = {
            "num_gt": len(g_ids), "num_pred": len(p_ids),
            "num_gt_target": n_target_gt, "num_pred_target": n_target_pred,
            # legacy rate over ALL GTs (incl. cone / none), kept for history
            f"detection_rate_iou{iou_detect}": len(det_all) / max(len(g_ids), 1),
            # main one-to-one target metrics
            "detection_target_rate": len(det_target) / max(n_target_gt, 1),
            "detection_typed_rate": len(det_typed) / max(n_target_gt, 1),
            "detection_target_precision": len(det_target) / max(n_target_pred, 1),
            "detection_typed_precision": len(det_typed) / max(n_target_pred, 1),
            "num_detected_target": len(det_target),
            "num_detected_typed": len(det_typed),
            # diagnostic (not one-to-one)
            "best_iou_target_diag_rate": (
                sum(1 for g in g_ids if g_type[g] in TARGET_TYPES
                    and best_diag[g] >= iou_detect) / max(n_target_gt, 1)),
            "miou_target": (sum(best_iou[g] for g in g_ids
                                if g_type[g] in TARGET_TYPES) /
                            max(n_target_gt, 1)),
            "mean_iou_matched": (sum(i for _, _, i in matched) /
                                 max(len(matched), 1)),
            "gt_unmatched": [g for g in g_ids if best_iou[g] < iou_detect],
            "gt_unmatched_target": [g for g in g_ids
                                    if g_type[g] in TARGET_TYPES
                                    and best_iou[g] < iou_detect],
            "pred_unmatched": [p for p in p_ids
                               if max((iou[g_ids.index(g), p_ids.index(p)]
                                      for g in g_ids), default=0.0) < iou_detect],
            "matched_pairs": [
                {"gt": g, "pred": p, "iou": round(i, 4),
                 "gt_type": g_type[g], "pred_type": types[p]}
                for g, p, i in matched],
        }

    # ------- merge analysis: how initial regions map onto GT instances
    g_ids, _, _, g_sets, _ = iou_matrix(gt_inst, region)
    gt_region_count = {}
    for g in g_ids:
        gt_region_count[g] = int(len(set(region[g_sets[g]].tolist()) - {-1}))
    overseg = {g: c for g, c in gt_region_count.items() if c > 1}
    res["merge"] = {
        "gt_regions_count": gt_region_count,
        "oversegmented_gt": overseg,
        "premerge_miou_target": res["premerge_instance"]["miou_target"],
        "final_miou_target": res["final_instance"]["miou_target"],
        "premerge_detection_target":
            res["premerge_instance"]["detection_target_rate"],
        "final_detection_target":
            res["final_instance"]["detection_target_rate"],
    }
    # GT instances whose best final match is shared with another GT instance
    # (multiple GT merged into one prediction) - detect via many-to-one
    final_types = pred_type_of(meta, "final")
    g_ids2, p_ids2, iou2, _, _ = iou_matrix(gt_inst, final)
    multi = {}
    for gj, g in enumerate(g_ids2):
        strong = [p_ids2[pj] for pj in range(len(p_ids2))
                  if iou2[gj, pj] >= iou_detect]
        if len(strong) > 1:
            multi[g] = strong
    res["merge"]["gt_merged_into_multi"] = multi

    # ------- per-point type metrics (targets + non-target tracking)
    gt_type_arr = np.array(gt_type)
    pred_pt_type = np.array(
        [final_types[p] if p >= 0 else "none" for p in final])
    conf = {}
    for g in ["plane", "sphere", "cylinder", "cone", "none"]:
        row = {}
        for p in ["plane", "sphere", "cylinder", "none"]:
            row[p] = int(np.count_nonzero((gt_type_arr == g) &
                                          (pred_pt_type == p)))
        conf[g] = row
    res["point_type_confusion"] = conf
    per_type = {}
    for t in TARGET_TYPES:
        tp = conf[t][t]
        fp = sum(conf[o][t] for o in ["plane", "sphere", "cylinder", "cone", "none"]
                 if o != t and o != "cone") + 0  # target columns only
        fp += conf["cone"][t]  # cone pts wrongly typed as target
        fn = sum(conf[t][o] for o in ["plane", "sphere", "cylinder", "none"]
                 if o != t)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_type[t] = {"precision": prec, "recall": rec,
                       "f1": 2 * prec * rec / max(prec + rec, EPS),
                       "points": int(np.count_nonzero(gt_type_arr == t))}
    res["point_type_prf"] = per_type
    target_mask = np.isin(gt_type_arr, TARGET_TYPES)
    correct = sum(conf[t][t] for t in TARGET_TYPES)
    res["point_type_accuracy_target"] = correct / max(int(np.count_nonzero(target_mask)), 1)
    n_cone = int(np.count_nonzero(gt_type_arr == "cone"))
    n_none = int(np.count_nonzero(gt_type_arr == "none"))
    cone_pt_types = {t: int(np.count_nonzero((gt_type_arr == "cone") &
                                             (pred_pt_type == t)))
                     for t in ["plane", "sphere", "cylinder", "none"]}
    none_pt_types = {t: int(np.count_nonzero((gt_type_arr == "none") &
                                             (pred_pt_type == t)))
                     for t in ["plane", "sphere", "cylinder", "none"]}
    # cone GT instances detected as a target primitive (best-IoU diagnostic,
    # prediction type must be one of the target types)
    cone_gt_ids = [g for g in set(gt_inst.tolist()) - {-1}
                   if gt_by_id[g]["type"] == "cone"]
    _cd = []
    for m in res["final_instance"]["matched_pairs"]:
        if m["gt"] in cone_gt_ids and m["iou"] >= iou_detect \
                and m["pred_type"] in TARGET_TYPES:
            _cd.append(m["gt"])
    cone_pt_assigned = sum(v for k, v in cone_pt_types.items() if k != "none")
    res["non_target"] = {
        "cone_points": n_cone,
        "cone_assigned_ratio": cone_pt_assigned / max(n_cone, 1),
        "cone_point_types": cone_pt_types,
        "cone_instances_detected_as_target": len(_cd),
        "cone_instance_ids_detected_as_target": _cd,
        "none_points": n_none,
        "none_assigned_ratio": (n_none - none_pt_types["none"]) / max(n_none, 1),
        "none_point_types": none_pt_types,
        "note": ("cone mis-recognition is counted separately from the "
                 "unclassified support surfaces (type 'none'); support "
                 "surfaces may legitimately be planes but carry no "
                 "parameter GT"),
    }

    # ------- instance-level type accuracy (on matched target pairs)
    matched_typed = [m for m in res["final_instance"]["matched_pairs"]
                     if m["gt_type"] in TARGET_TYPES and m["iou"] >= iou_detect]
    if matched_typed:
        res["instance_type_accuracy"] = (
            sum(1 for m in matched_typed if m["gt_type"] == m["pred_type"]) /
            len(matched_typed))
    else:
        res["instance_type_accuracy"] = None

    # ------- geometric parameter errors on matched, same-type pairs
    param_rows = []
    coeff_by_pred = {m["type"]: None for m in meta.get("models", [])}
    final_models = meta.get("models", [])
    for m in res["final_instance"]["matched_pairs"]:
        if m["iou"] < iou_detect or m["gt_type"] != m["pred_type"]:
            continue
        if m["gt_type"] not in TARGET_TYPES:
            continue
        gt_p = gt_by_id[m["gt"]]
        if not gt_p.get("has_params", True):
            continue
        model = final_models[m["pred"]]
        e = parameter_errors(gt_p, m["pred_type"], model["coefficients"])
        # surface RMSE both directions
        gmask = gt_inst == m["gt"]
        gpts = None
        if "points_path" in gt_p:  # optional: not used by default
            pass
        param_rows.append({
            "gt": m["gt"], "pred": m["pred"], "type": m["gt_type"],
            "iou": m["iou"], "gt_points": int(np.count_nonzero(gmask)),
            "pred_points": model["points"],
            "pred_rmse": model.get("rmse"), **e})
    res["parameter_errors"] = param_rows

    # ------- failure / robustness stats
    failed = meta.get("failed_regions", [])
    reasons = {}
    for fr in failed:
        reasons[fr["reason"]] = reasons.get(fr["reason"], 0) + 1
    bad_params = sum(1 for m in final_models
                     if any(not math.isfinite(x) for x in m["coefficients"]))
    res["failures"] = {
        "failed_region_count": len(failed),
        "failed_region_reasons": reasons,
        "unassigned_points": int(np.count_nonzero(final < 0)),
        "unassigned_ratio": float(np.count_nonzero(final < 0)) / max(n, 1),
        "nonfinite_param_models": bad_params,
        "exit_code": meta.get("exit_code"),
    }
    return res


# --------------------------------------------------------------- aggregation
AGG_FIELDS = [
    "dataset", "sample", "mode", "num_gt_target", "num_pred_target",
    "detection", "detection_typed", "miou", "type_acc_point", "type_acc_inst",
    "plane_f1", "sphere_f1", "cylinder_f1", "unassigned_ratio",
]


def aggregate(results_root, out_csv):
    rows = []
    for rj in sorted(glob.glob(os.path.join(results_root, "**",
                                             "evaluation.json"),
                                recursive=True)):
        with open(rj) as f:
            r = json.load(f)
        for mode in ("premerge", "final"):
            st = r[mode + "_instance"]
            prf = r.get("point_type_prf", {})
            rows.append({
                "dataset": r.get("dataset", "?"),
                "sample": r["sample"],
                "mode": mode,
                "num_gt_target": st["num_gt_target"],
                "num_pred_target": st["num_pred_target"],
                "detection_target": st.get("detection_target_rate"),
                "detection_typed": st.get("detection_typed_rate"),
                "best_iou_diag": st.get("best_iou_target_diag_rate"),
                "miou": st["miou_target"],
                "type_acc_point": r.get("point_type_accuracy_target"),
                "type_acc_inst": r.get("instance_type_accuracy"),
                "plane_f1": prf.get("plane", {}).get("f1"),
                "sphere_f1": prf.get("sphere", {}).get("f1"),
                "cylinder_f1": prf.get("cylinder", {}).get("f1"),
                "unassigned_ratio": r["failures"]["unassigned_ratio"],
                "cone_assigned_ratio": r.get("non_target", {}).get(
                    "cone_assigned_ratio"),
                "cone_instances_as_target": r.get("non_target", {}).get(
                    "cone_instances_detected_as_target"),
                "none_assigned_ratio": r.get("non_target", {}).get(
                    "none_assigned_ratio"),
            })
    if not rows:
        print("no evaluation.json found under", results_root)
        return
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} with {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir")
    ap.add_argument("--pred-prefix")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default="?")
    ap.add_argument("--aggregate", nargs=2, metavar=("RESULTS_ROOT", "CSV"))
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.aggregate[0], args.aggregate[1])
        return
    res = evaluate_sample(args.sample_dir, args.pred_prefix)
    res["dataset"] = args.dataset
    out = args.out or os.path.join(os.path.dirname(args.pred_prefix) or ".",
                                   "evaluation.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
