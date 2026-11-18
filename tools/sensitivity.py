"""One-at-a-time (OAT) parameter sensitivity analysis for the frozen D4
pipeline. Baseline = frozen BAGSFit-validated configuration; each parameter
is perturbed independently on a validation subset (non-tuning scenes).
No full grid search, no test-set tuning."""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_sample  # noqa: E402

BASE = dict(distance=0.02, min_points=80, smooth=12, curvature=0.05,
            min_radius=0.01, max_radius=10, merge_distance=0.1,
            merge_angle=5, normalK=300, minInlierRatio=0.3)

SWEEPS = {
    "normalK":        [50, 100, 300, 500],
    "smooth_deg":     [8, 12, 20],
    "curvature":      [0.03, 0.05, 0.10],
    "distance":       [0.015, 0.02, 0.03],
    "minInlierRatio": [0.2, 0.3, 0.5],
    "min_points":     [80, 300],
    "merge_distance": [0.06, 0.1, 0.2],
    "merge_angle":    [3, 5, 10],
}


def build_args(cfg, mode="D4"):
    pos = [str(cfg["distance"]), str(cfg["min_points"]), str(cfg["smooth"]),
           str(cfg["curvature"]), str(cfg["min_radius"]), str(cfg["max_radius"]),
           str(cfg["merge_distance"]), str(cfg["merge_angle"])]
    opts = [f"--normalK={cfg['normalK']}",
            f"--minInlierRatio={cfg['minInlierRatio']}",
            f"--ablation={mode}"]
    return pos, opts


def run_config(exe, samples, out_root, tag, cfg, dll_path, mode="D4"):
    pos, opts = build_args(cfg, mode)
    results = []
    for s in samples:
        out_dir = os.path.join(out_root, tag, os.path.basename(s))
        done = os.path.join(out_dir, "evaluation.json")
        if os.path.exists(done):
            with open(done) as f:
                results.append(json.load(f))
            continue
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, "out")
        env = dict(os.environ)
        env["PATH"] = dll_path + os.pathsep + env.get("PATH", "")
        cmd = [exe, os.path.join(s, "points.txt"), prefix] + pos + opts
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=600)
        wall = time.time() - t0
        if p.returncode not in (0, 2):
            print(f"  {tag}/{os.path.basename(s)} CRASH rc={p.returncode}")
            continue
        try:
            r = evaluate_sample(s, prefix)
            r["wall_seconds"] = wall
            with open(done, "w") as f:
                json.dump(r, f, indent=2)
            results.append(r)
        except Exception as e:
            print(f"  {tag}/{os.path.basename(s)} EVAL FAIL {e}")
    return results


def summarize(results):
    if not results:
        return None
    def mean(k, sub):
        v = [r[k] for r in sub if r.get(k) is not None]
        return sum(v)/len(v) if v else float("nan")
    f = [r["final_instance"] for r in results]
    prf = [r.get("point_type_prf", {}) for r in results]
    return {
        "n": len(results),
        "det": mean(0, [dict(x) for x in [None]]) if False else
               sum(x["detection_target_rate"] for x in f)/len(f),
        "typed": sum(x["detection_typed_rate"] for x in f)/len(f),
        "miou": sum(x["miou_target"] for x in f)/len(f),
        "ptacc": sum(r["point_type_accuracy_target"] for r in results)/len(results),
        "unassigned": sum(r["failures"]["unassigned_ratio"] for r in results)/len(results),
        "plane_f1": sum(p.get("plane", {}).get("f1", 0) for p in prf)/len(prf),
        "sphere_f1": sum(p.get("sphere", {}).get("f1", 0) for p in prf)/len(prf),
        "cyl_f1": sum(p.get("cylinder", {}).get("f1", 0) for p in prf)/len(prf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--samples-root", required=True)
    ap.add_argument("--subset", help="comma list of sample dir names")
    ap.add_argument("--out-root", default="results/sensitivity")
    ap.add_argument("--dll-path", default="D:/vcpkg/installed/x64-windows/bin")
    ap.add_argument("--mode", default="D4")
    args = ap.parse_args()

    if args.subset:
        names = args.subset.split(",")
    else:
        names = sorted(os.listdir(args.samples_root))
    samples = [os.path.join(args.samples_root, n) for n in names]
    print(f"validation subset: {len(samples)} samples")

    rows = []
    # baseline
    res = run_config(args.exe, samples, args.out_root, "baseline",
                     BASE, args.dll_path, args.mode)
    s0 = summarize(res)
    rows.append(("BASELINE", "-", s0))
    print(f"BASELINE det={s0['det']:.3f} typed={s0['typed']:.3f} "
          f"miou={s0['miou']:.3f} ptacc={s0['ptacc']:.3f}")

    for pname, levels in SWEEPS.items():
        for lv in levels:
            if lv == BASE.get(pname):
                continue
            cfg = dict(BASE)
            cfg[pname] = lv
            tag = f"{pname}={lv}"
            res = run_config(args.exe, samples, args.out_root,
                             tag.replace("=", "_"), cfg, args.dll_path,
                             args.mode)
            s = summarize(res)
            rows.append((pname, lv, s))
            if s:
                print(f"{tag:24s} det={s['det']:.3f} ({s['det']-s0['det']:+.3f}) "
                      f"typed={s['typed']:.3f} miou={s['miou']:.3f} "
                      f"ptacc={s['ptacc']:.3f} unass={s['unassigned']:.3f}")

    # write table
    import csv
    with open(os.path.join(args.out_root, "sensitivity_table.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value", "n", "det", "det_delta", "typed",
                    "miou", "ptacc", "unassigned", "plane_f1", "sphere_f1",
                    "cyl_f1"])
        base = rows[0][2]
        for pname, lv, s in rows:
            if s is None:
                continue
            w.writerow([pname, lv, s["n"], f"{s['det']:.4f}",
                        f"{s['det']-base['det']:+.4f}", f"{s['typed']:.4f}",
                        f"{s['miou']:.4f}", f"{s['ptacc']:.4f}",
                        f"{s['unassigned']:.4f}", f"{s['plane_f1']:.4f}",
                        f"{s['sphere_f1']:.4f}", f"{s['cyl_f1']:.4f}"])
    print("\n->", os.path.join(args.out_root, "sensitivity_table.csv"))


if __name__ == "__main__":
    main()
