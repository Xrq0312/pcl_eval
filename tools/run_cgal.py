"""Run cgal_primitive_fit over the 10-sample BAGSFit set (+ evaluate).

Tuning policy (per task): parameters chosen ONLY on 000_00000; the other 9
samples are evaluated independently. Caches finished runs."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--samples", default="samples/bagsfit")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--opts", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dll", default="D:/vcpkg/installed/x64-windows/bin")
    args = ap.parse_args()

    samples = sorted([d for d in glob.glob(os.path.join(args.samples, "*"))
                      if os.path.isdir(d)
                      and os.path.exists(os.path.join(d, "gt_labels.csv"))])
    if args.limit:
        samples = samples[:args.limit]
    for s in samples:
        name = os.path.basename(s)
        out_dir = os.path.join(args.results, args.tag, name)
        done = os.path.join(out_dir, "evaluation.json")
        if os.path.exists(done):
            print(f"{name}: cached")
            continue
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, "out")
        env = dict(os.environ)
        env["PATH"] = args.dll + os.pathsep + env.get("PATH", "")
        cmd = [args.exe, os.path.join(s, "points.txt"), prefix]
        if args.opts:
            cmd += args.opts.split()
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=1800)
        wall = time.time() - t0
        with open(os.path.join(out_dir, "run_log.txt"), "w") as f:
            f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
        if p.returncode not in (0, 2):
            print(f"{name}: CRASH rc={p.returncode}")
            print(p.stderr[:500])
            continue
        with open(prefix + "_primitives.json") as f:
            meta = json.load(f)
        meta["exit_code"] = p.returncode
        meta["wall_seconds"] = wall
        with open(prefix + "_primitives.json", "w") as f:
            json.dump(meta, f, indent=2)
        try:
            r = evaluate_sample(s, prefix)
            r["dataset"] = "bagsfit"
            r["wall_seconds"] = wall
            r["method"] = args.tag
            with open(done, "w") as f:
                json.dump(r, f, indent=2)
            st = r["final_instance"]
            print(f"{name}: wall={wall:.1f}s gt={st['num_gt_target']} "
                  f"pred={st['num_pred_target']} "
                  f"det={st['detection_target_rate']:.2f} "
                  f"typed={st['detection_typed_rate']:.2f} "
                  f"miou={st['miou_target']:.3f} "
                  f"ptacc={r['point_type_accuracy_target']:.2f}")
        except Exception as e:
            print(f"{name}: EVAL FAIL {e}")


if __name__ == "__main__":
    main()
