"""Batch test runner: run primitive_fit (ablation modes) + evaluator
over every unified sample in a root directory.

Usage:
  python batch_test.py --samples samples/bagsfit --results results/bagsfit \
      --exe D:/.../primitive_fit.exe --distance 0.02 --modes ABCD --dll-path ...
Per sample S and mode M: results/bagsfit/S/M/{out_labels.csv,...,evaluation.json}
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_sample  # noqa: E402


def run_one(exe, sample_dir, out_dir, distance, mode, extra_args, dll_path,
            opts=""):
    os.makedirs(out_dir, exist_ok=True)
    pts = os.path.join(sample_dir, "points.txt")
    if not os.path.exists(pts):
        pts = os.path.join(sample_dir, "points.pcd")
    prefix = os.path.join(out_dir, "out")
    cmd = [exe, pts, prefix, str(distance)] + [str(a) for a in extra_args]
    if opts:
        cmd += opts.split()
    cmd += [f"--ablation={mode}"]
    env = dict(os.environ)
    if dll_path:
        env["PATH"] = dll_path + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=3600)
    wall = time.time() - t0
    with open(os.path.join(out_dir, "run_log.txt"), "w") as f:
        f.write(proc.stdout)
        f.write("\n--- stderr ---\n")
        f.write(proc.stderr)
    if proc.returncode not in (0, 2):
        print(f"  CRASH/ERROR rc={proc.returncode}")
        return None, wall
    # record exit code into primitives.json for the evaluator
    pj = os.path.join(out_dir, "out_primitives.json")
    if os.path.exists(pj):
        with open(pj) as f:
            meta = json.load(f)
        meta["exit_code"] = proc.returncode
        meta["wall_seconds"] = wall
        with open(pj, "w") as f:
            json.dump(meta, f, indent=2)
    return prefix, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--distance", type=float, default=0.02)
    ap.add_argument("--min-points", type=int, default=80)
    ap.add_argument("--modes", default="ABCD")
    ap.add_argument("--extra", default="",
                    help="extra fixed positional args for primitive_fit, e.g. '0.05,10,0.01'")
    ap.add_argument("--opts", default="",
                    help="extra --key=value options passed through, e.g. '--normalK=100 --minInlierRatio=0.5'")
    ap.add_argument("--dll-path", default="D:/vcpkg/installed/x64-windows/bin")
    ap.add_argument("--dataset", default="bagsfit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--filter", default="",
                    help="only sample dirs whose name contains this prefix "
                         "(e.g. '002_' for one scene)")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    samples = sorted([d for d in glob.glob(os.path.join(args.samples, "*"))
                      if os.path.isdir(d)
                      and os.path.exists(os.path.join(d, "gt_labels.csv"))])
    if args.filter:
        samples = [d for d in samples
                   if os.path.basename(d).startswith(args.filter)]
    if args.limit:
        samples = samples[:args.limit]
    extra = []
    if args.extra:
        extra = [x for x in args.extra.split(",") if x]
    total_wall = 0.0
    for s in samples:
        name = os.path.basename(s)
        for mode in (args.modes.split(",") if "," in args.modes
                     else list(args.modes)):
            out_dir = os.path.join(args.results, name, mode)
            done = os.path.join(out_dir, "evaluation.json")
            if os.path.exists(done):
                print(f"{name}/{mode}: cached")
                continue
            prefix, wall = run_one(args.exe, s, out_dir, args.distance,
                                   mode, extra, args.dll_path, args.opts)
            if prefix is None:
                continue
            total_wall += wall
            try:
                res = evaluate_sample(s, prefix)
                res["dataset"] = args.dataset
                res["wall_seconds"] = wall
                with open(done, "w") as f:
                    json.dump(res, f, indent=2)
                st = res["final_instance"]
                print(f"{name}/{mode}: wall={wall:.1f}s gt={st['num_gt_target']} "
                      f"pred={st['num_pred_target']} "
                      f"det={st['detection_rate_iou0.5']:.2f} "
                      f"miou={st['miou_target']:.3f}")
            except Exception as e:
                print(f"{name}/{mode}: EVAL FAIL {e}")
    print(f"total fit wall time: {total_wall:.0f}s")


if __name__ == "__main__":
    main()
