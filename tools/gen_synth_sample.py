"""Generate one unified-format synthetic sample with exact GT labels,
used to validate the evaluation pipeline before real datasets arrive."""
import argparse
import json
import math
import os
import random


def main(out_dir, seed=20260924):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    xyz, labels = [], []          # labels: (instance_id, type)
    prims = []

    def put(x, y, z, inst, typ, noise=0.001):
        xyz.append((x + random.gauss(0, noise), y + random.gauss(0, noise),
                    z + random.gauss(0, noise)))
        labels.append((inst, typ))

    # inst 0/1: two coplanar z=0 patches (same plane geometry)
    for xmin, xmax in ((-2.3, -1.3), (-0.7, 0.3)):
        for _ in range(2000):
            put(random.uniform(xmin, xmax), random.uniform(-0.5, 0.5), 0,
                0 if xmin < -1 else 1, "plane")
    prims.append({"instance_id": 0, "type": "plane", "normal": [0, 0, 1],
                  "d": 0.0, "has_params": True})
    prims.append({"instance_id": 1, "type": "plane", "normal": [0, 0, 1],
                  "d": 0.0, "has_params": True})

    # inst 2: sphere
    for _ in range(2500):
        u = random.uniform(-1, 1)
        phi = random.uniform(0, 2 * math.pi)
        rr = 0.35 * math.sqrt(max(0, 1 - u * u))
        put(0.9 + rr * math.cos(phi), rr * math.sin(phi), 0.65 + 0.35 * u,
            2, "sphere")
    prims.append({"instance_id": 2, "type": "sphere",
                  "center": [0.9, 0.0, 0.65], "radius": 0.35,
                  "has_params": True})

    # inst 3: cylinder lateral surface + caps as separate planes (4/5)
    for _ in range(3000):
        phi = random.uniform(0, 2 * math.pi)
        put(2.0 + 0.25 * math.cos(phi), 0.25 * math.sin(phi),
            random.uniform(0.2, 1.1), 3, "cylinder")
    prims.append({"instance_id": 3, "type": "cylinder",
                  "axis_point": [2.0, 0.0, 0.5], "axis_direction": [0, 0, 1],
                  "radius": 0.25, "has_params": True})
    for z, inst in ((1.1, 4), (0.2, 5)):
        for _ in range(1200):
            phi = random.uniform(0, 2 * math.pi)
            rr = 0.25 * math.sqrt(random.uniform(0, 1))
            put(2.0 + rr * math.cos(phi), rr * math.sin(phi), z, inst, "plane")
        prims.append({"instance_id": inst, "type": "plane",
                      "normal": [0, 0, 1], "d": z, "has_params": True})

    # background noise: no instance
    for _ in range(100):
        put(random.uniform(-2.4, 2.4), random.uniform(-0.8, 0.8),
            random.uniform(-0.1, 1.2), -1, "none", noise=0)

    with open(os.path.join(out_dir, "points.txt"), "w") as f:
        for x, y, z in xyz:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    with open(os.path.join(out_dir, "gt_labels.csv"), "w") as f:
        f.write("point_index,instance_id,primitive_type\n")
        for i, (inst, typ) in enumerate(labels):
            f.write(f"{i},{inst},{typ}\n")
    with open(os.path.join(out_dir, "gt_primitives.json"), "w") as f:
        json.dump({"primitives": prims}, f, indent=2)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"dataset": "synthetic", "units": "meters",
                   "num_points": len(xyz),
                   "note": "two coplanar patches (inst 0,1), sphere, "
                           "cylinder + 2 cap planes, uniform noise"}, f, indent=2)
    print(f"wrote sample with {len(xyz)} points to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()
    main(a.out_dir, a.seed)
