"""Authoritative C vs D per-GT audit, recomputed from raw label CSVs.

Rules:
- each row keyed by (full sample id, gt instance id) - no shorthand
- one-to-one Hungarian matching (from evaluation.json matched_pairs) AND
  best-IoU diagnostic (recomputed from labels) are reported separately
- target metrics count only plane/sphere/cylinder GT primitives
- type correctness = predicted model type equals GT type
"""
import csv as csvmod
import glob
import json
import os

import numpy as np

from scipy.optimize import linear_sum_assignment

TARGET = ("plane", "sphere", "cylinder")
IOU = 0.5


def load_labels(path):
    with open(path, newline="") as f:
        return np.array([int(r["primitive_id"]) for r in csvmod.DictReader(f)],
                        dtype=int)


def load_gt(sample):
    with open(os.path.join(sample, "gt_labels.csv"), newline="") as f:
        rd = list(csvmod.DictReader(f))
    inst = np.array([int(r["instance_id"]) for r in rd], dtype=int)
    typ = [r["primitive_type"] for r in rd]
    with open(os.path.join(sample, "gt_primitives.json")) as f:
        prims = {p["instance_id"]: p for p in json.load(f)["primitives"]}
    return inst, typ, prims


def best_iou_matches(gt_inst, pred):
    out = {}
    for g in sorted(set(gt_inst.tolist()) - {-1}):
        gm = gt_inst == g
        best = (0.0, -1)
        for k in sorted(set(pred[gm].tolist()) - {-1}):
            pm = pred == k
            inter = np.count_nonzero(gm & pm)
            union = np.count_nonzero(gm | pm)
            i = inter / union if union else 0
            if i > best[0]:
                best = (i, k)
        out[g] = best
    return out


def hungarian(gt_inst, pred, pred_types):
    g_ids = sorted(set(gt_inst.tolist()) - {-1})
    p_ids = sorted(set(pred.tolist()) - {-1})
    iou = np.zeros((len(g_ids), len(p_ids)))
    for a, g in enumerate(g_ids):
        gm = gt_inst == g
        for b, p in enumerate(p_ids):
            pm = pred == p
            inter = np.count_nonzero(gm & pm)
            union = np.count_nonzero(gm | pm)
            iou[a, b] = inter / union if union else 0
    if iou.size == 0:
        return {}
    gi, pi = linear_sum_assignment(-iou)
    return {g_ids[a]: (iou[a, b], p_ids[b], pred_types[p_ids[b]])
            for a, b in zip(gi, pi)}


def main():
    rows = []
    for sample in sorted(glob.glob("samples/bagsfit/*")):
        if not os.path.exists(os.path.join(sample, "gt_labels.csv")):
            continue
        name = os.path.basename(sample)
        gt_inst, gt_type, prims = load_gt(sample)
        per_mode = {}
        for mode in ("C", "D"):
            pred = load_labels(
                os.path.join("results", "bagsfit", name, mode,
                             "out_labels.csv"))
            with open(os.path.join("results", "bagsfit", name, mode,
                                  "out_primitives.json")) as f:
                models = json.load(f)["models"]
            pt = {i: m["type"] for i, m in enumerate(models)}
            bi = best_iou_matches(gt_inst, pred)
            hg = hungarian(gt_inst, pred, pt)
            per_mode[mode] = (bi, hg)

        for g in sorted(set(gt_inst.tolist()) - {-1}):
            t = prims[g]["type"]
            npts = int(np.count_nonzero(gt_inst == g))
            cb, _ = per_mode["C"][0][g]
            db, _ = per_mode["D"][0][g]
            ch = per_mode["C"][1].get(g)
            dh = per_mode["D"][1].get(g)
            ch_iou, ch_pred, ch_type = ch if ch else (0.0, -1, "-")
            dh_iou, dh_pred, dh_type = dh if dh else (0.0, -1, "-")
            c_det = ch_iou >= IOU
            d_det = dh_iou >= IOU
            c_tdet = c_det and ch_type == t and t in TARGET
            d_tdet = d_det and dh_type == t and t in TARGET
            status = ("lost" if c_det and not d_det
                      else "gained" if d_det and not c_det
                      else "kept" if c_det else "never")
            rows.append(dict(sample=name, gt=g, gt_type=t, points=npts,
                             c_best=cb, d_best=db,
                             c_hung=ch_iou, c_ptype=ch_type,
                             d_hung=dh_iou, d_ptype=dh_type,
                             c_typed_det=c_tdet, d_typed_det=d_tdet,
                             status=status))

    with open("results/cd_audit_raw.csv", "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tgt = [r for r in rows if r["gt_type"] in TARGET]
    print("=== TARGET GT primitives only (hungarian, IoU>=0.5) ===")
    print(f"C typed-det: {sum(r['c_typed_det'] for r in tgt)}/{len(tgt)}"
          f"  D typed-det: {sum(r['d_typed_det'] for r in tgt)}/{len(tgt)}")
    for st in ("lost", "gained", "kept", "never"):
        sel = [r for r in tgt if r["status"] == st]
        print(f"{st:6s}: {len(sel)}")
        for r in sel:
            flag = " [type-correct]" if r["c_typed_det"] or r["d_typed_det"] else ""
            print(f"   {r['sample']}/GT{r['gt']} {r['gt_type']}"
                  f" ({r['points']} pts): C {r['c_hung']:.3f}"
                  f" [{r['c_ptype']}] -> D {r['d_hung']:.3f}"
                  f" [{r['d_ptype']}]{flag}")
    print("\n=== non-target transitions ===")
    for r in rows:
        if r["gt_type"] not in TARGET and r["status"] in ("lost", "gained"):
            print(f"   {r['sample']}/GT{r['gt']} {r['gt_type']}"
                  f" ({r['points']} pts): C {r['c_hung']:.3f}"
                  f" [{r['c_ptype']}] -> D {r['d_dung' if False else 'd_hung']:.3f}"
                  f" [{r['d_ptype']}] ({r['status']})")


if __name__ == "__main__":
    main()
