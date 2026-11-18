"""C->D transition analysis: per GT primitive, which detections were lost /
kept / gained, where the points went, and how each surviving model changed
in type / parameters / point set between the two ablation stages.

C = region growing + fit + merge (no refit, no absorption)
D = C + merge-refit + normal-constrained global re-absorption
Both runs share the same labels CSV / primitives JSON output format, so we
pair D models with their C predecessors by maximal point-set containment.
"""
import glob
import json
import os
import sys

import numpy as np

TARGET = ("plane", "sphere", "cylinder")
IOU_DET = 0.5


def load(prefix):
    import csv
    ids = []
    with open(prefix + "_labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            ids.append(int(r["primitive_id"]))
    with open(prefix + "_primitives.json") as f:
        meta = json.load(f)
    return np.array(ids, dtype=int), meta["models"]


def gt_arrays(sample_dir):
    import csv as _csv
    inst = []
    typ = []
    with open(os.path.join(sample_dir, "gt_labels.csv"), newline="") as f:
        for r in _csv.DictReader(f):
            inst.append(int(r["instance_id"]))
            typ.append(r["primitive_type"])
    with open(os.path.join(sample_dir, "gt_primitives.json")) as f:
        prims = {p["instance_id"]: p for p in json.load(f)["primitives"]}
    return np.array(inst, dtype=int), typ, prims


def iou(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def param_str(m):
    c = m["coefficients"]
    if m["type"] == "plane":
        n = np.array(c[:3])
        n = n / np.linalg.norm(n)
        return f"plane n={np.round(n,3)} d={-c[3]/np.linalg.norm(c[:3]):.4f}"
    if m["type"] == "sphere":
        return f"sphere c={np.round(c[:3],3)} r={c[3]:.4f}"
    u = np.array(c[3:6])
    u = u / np.linalg.norm(u)
    return f"cyl p={np.round(c[:3],3)} u={np.round(u,3)} r={c[6]:.4f}"


def param_delta(a, b):
    """readable delta between predecessor params a and successor b"""
    if a["type"] != b["type"]:
        return "TYPE CHANGED"
    ca, cb = np.array(a["coefficients"]), np.array(b["coefficients"])
    if a["type"] == "plane":
        na, nb = ca[:3] / np.linalg.norm(ca[:3]), cb[:3] / np.linalg.norm(cb[:3])
        ang = np.degrees(np.arccos(min(1, abs(na @ nb))))
        da, db = -ca[3] / np.linalg.norm(ca[:3]), -cb[3] / np.linalg.norm(cb[:3])
        return f"n-angle {ang:.3f}deg, offset {abs(da-db):.5f} m"
    if a["type"] == "sphere":
        return (f"center {np.linalg.norm(ca[:3]-cb[:3]):.5f} m, "
                f"radius {abs(ca[3]-cb[3]):.5f} m")
    ua, ub = ca[3:6] / np.linalg.norm(ca[3:6]), cb[3:6] / np.linalg.norm(cb[3:6])
    ang = np.degrees(np.arccos(min(1, abs(ua @ ub))))
    return (f"axis-angle {ang:.3f}deg, line-dist "
            f"{np.linalg.norm(np.cross(cb[:3]-ca[:3], ua)):.5f} m, "
            f"radius {abs(ca[6]-cb[6]):.5f} m")
    return "?"


def main(results_root="results/bagsfit", samples_root="samples/bagsfit",
         out_path="results/cd_diff_report.md"):
    lines = ["# C->D transition analysis (per GT primitive)",
             "",
             "C = grow+fit+merge(no refit); D = C + merge-refit + "
             "normal-constrained re-absorption.",
             "Legend: det = IoU>=0.5; 'lost' = detected in C, not in D.",
             ""]
    n_lost_total = n_kept = n_gained = 0
    for sample_dir in sorted(glob.glob(os.path.join(samples_root, "*"))):
        name = os.path.basename(sample_dir)
        cdir = os.path.join(results_root, name, "C")
        ddir = os.path.join(results_root, name, "D")
        if not (os.path.exists(os.path.join(cdir, "out_labels.csv"))
                and os.path.exists(os.path.join(ddir, "out_labels.csv"))):
            continue
        c_ids, c_models = load(os.path.join(cdir, "out"))
        d_ids, d_models = load(os.path.join(ddir, "out"))
        gt_inst, gt_type, gt_prims = gt_arrays(sample_dir)
        n = len(gt_inst)

        lines.append(f"\n## sample {name}")
        lines.append(f"C models: {len(c_models)}, D models: {len(d_models)}")

        # pair each D model with its C predecessor (max containment)
        pred_of = {}
        for j, dm in enumerate(d_models):
            dj = d_ids == j
            best, best_ov = -1, 0.0
            for i, cm in enumerate(c_models):
                ov = np.count_nonzero(dj & (c_ids == i)) / max(np.count_nonzero(dj), 1)
                if ov > best_ov:
                    best_ov, best = ov, i
            pred_of[j] = (best, best_ov)

        # per-model evolution table
        lines.append("\n### D-model evolution (predecessor -> successor)")
        lines.append("| D# | type | pts C->D | absorbed(new) | left C | "
                     "predecessor C# | C type | param delta |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for j, dm in enumerate(d_models):
            dj = d_ids == j
            i, ov = pred_of[j]
            if i < 0:
                lines.append(f"| {j} | {dm['type']} | ?->{dm['points']} | ? | ? | "
                             f"none | - | new model |")
                continue
            ci = c_ids == i
            absorbed = np.count_nonzero(dj & ~ci)
            kept_pts = np.count_nonzero(dj & ci)
            left = np.count_nonzero(ci & ~dj)
            src = c_models[i]["type"]
            delta = param_delta(c_models[i], dm)
            lines.append(f"| {j} | {dm['type']} | {kept_pts}+{absorbed}="
                         f"{kept_pts+absorbed} | "
                         f"{absorbed} | {left} | {i} | {src} | {delta} |")
        lines.append("")
        for j, dm in enumerate(d_models):
            i, ov = pred_of[j]
            if i >= 0:
                lines.append(f"  D#{j} [{param_str(dm)}]  <- C#{i} "
                             f"[{param_str(c_models[i])}]  "
                             f"(containment {ov:.2f})")

        # GT-level kept/lost/gained
        gt_ids = sorted(set(gt_inst.tolist()) - {-1})
        rows = []
        for g in gt_ids:
            gm = gt_inst == g
            t = gt_type[int(np.flatnonzero(gm)[0])] if gm.any() else None
            c_best = max(((iou(gm, c_ids == i), i)
                          for i in range(len(c_models))), default=(0, -1))
            d_best = max(((iou(gm, d_ids == j), j)
                          for j in range(len(d_models))), default=(0, -1))
            c_type = c_models[c_best[1]]["type"] if c_best[1] >= 0 else "-"
            d_type = d_models[d_best[1]]["type"] if d_best[1] >= 0 else "-"
            status = ("kept" if c_best[0] >= IOU_DET and d_best[0] >= IOU_DET
                      else "lost" if c_best[0] >= IOU_DET
                      else "gained" if d_best[0] >= IOU_DET
                      else "none-both")
            if t not in TARGET and status in ("none-both",):
                pass  # keep non-target rows too, marked
            rows.append((g, t, status, c_best[0], c_best[1], c_type,
                         d_best[0], d_best[1], d_type))
        lines.append("\n### GT-level C vs D (best-IoU match)")
        lines.append("| gt | type | status | C IoU (pred, type) | "
                     "D IoU (pred, type) |")
        lines.append("|---|---|---|---|---|")
        for g, t, s, ci_, cpi, ct, di_, dpi, dt in rows:
            lines.append(f"| {g} | {t} | **{s}** | {ci_:.3f} "
                         f"(#{cpi},{ct}) | {di_:.3f} (#{dpi},{dt}) |")
        kept = sum(1 for r in rows if r[2] == "kept")
        lost = sum(1 for r in rows if r[2] == "lost")
        gained = sum(1 for r in rows if r[2] == "gained")
        n_kept += kept
        n_lost_total += lost
        n_gained += gained

        # lost primitives: point-by-point transfer C->D
        for g, t, s, ci_, cpi, ct, di_, dpi, dt in rows:
            if s != "lost":
                continue
            gm = gt_inst == g
            transfer = {}
            for x in np.flatnonzero(gm):
                transfer.setdefault((int(c_ids[x]), int(d_ids[x])), 0)
                transfer[(int(c_ids[x]), int(d_ids[x]))] += 1
            lines.append(f"\n#### lost GT {g} ({t}, {int(gm.sum())} pts)")
            lines.append("C best pred #%d (%s, IoU %.3f); D best pred #%d "
                         "(%s, IoU %.3f)" % (cpi, ct, ci_, dpi, dt, di_))
            lines.append("point transfer (C pred id -> D pred id), "
                         "-1 = unassigned:")
            for (cx, dx), k in sorted(transfer.items(),
                                      key=lambda kv: -kv[1]):
                lines.append(f"  {cx} -> {dx}: {k}")
            # what absorbed those points
            for (cx, dx), k in sorted(transfer.items(),
                                      key=lambda kv: -kv[1]):
                if dx == -1 or cx == dx:
                    continue
                i, _ = pred_of.get(dx, (-1, 0))
                lines.append(f"  (points {cx}->{dx}: into D#{dx} "
                             f"[{d_models[dx]['type']}] "
                             f"[{param_str(d_models[dx])}], C predecessor "
                             f"#{i})")
    lines.insert(3, f"\nTotal across samples: kept={n_kept}, lost="
                    f"{n_lost_total}, gained={n_gained}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:24]))
    print("...")
    print(f"report -> {out_path} (kept={n_kept}, lost={n_lost_total}, "
          f"gained={n_gained})")


def absrobed_txt(a):
    return str(a)


if __name__ == "__main__":
    main(*sys.argv[1:])
