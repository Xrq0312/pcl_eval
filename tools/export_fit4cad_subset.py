"""Export a deterministic Fit4CAD test subset from an official Git clone.

The upstream tree contains macOS ``Icon\r`` entries that Git for Windows
cannot check out. This exporter reads only legal requested blobs and records
their Git object IDs and SHA-256 checksums.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


DEFAULT_IDS = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 35)


def git(repo, *arguments, binary=False):
    output = subprocess.check_output(
        ["git", "-C", str(repo), *arguments], stderr=subprocess.DEVNULL)
    return output if binary else output.decode("utf-8").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ids", nargs="+", type=int, default=DEFAULT_IDS)
    args = parser.parse_args()
    commit = git(args.repo, "rev-parse", "HEAD")
    paths = ["README.md", "licence.txt"]
    for pc_id in args.ids:
        paths.extend((
            f"dataset/test_set/pointCloud/PC{pc_id}.txt",
            f"dataset/test_set/pointCloud_primitives/PC{pc_id}_primitives.txt",
            f"dataset/test_set/pointCloud_parametric/PC{pc_id}_parametric.txt",
            f"dataset/test_set/pointCloud_implicit/PC{pc_id}_implicit.txt",
        ))
    files = []
    for relative in paths:
        spec = f"{commit}:{relative}"
        blob = git(args.repo, "rev-parse", spec)
        content = git(args.repo, "cat-file", "blob", blob, binary=True)
        target = args.output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files.append({"path": relative, "git_blob": blob, "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {
        "dataset": "Fit4CAD", "official_repository": args.repo.resolve().as_posix(),
        "upstream_url": "https://github.com/chiararomanengo/Fit4CAD",
        "source_commit": commit,
        "selection": "12 IDs evenly spaced over official test IDs 1..35",
        "ids": args.ids, "files": files,
        "license_file": "licence.txt",
    }
    (args.output_root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"exported {len(files)} official blobs at {commit} to {args.output_root}")


if __name__ == "__main__":
    main()
