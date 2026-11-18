"""Acquire and validate a deterministic SPFN/CPFN 128K test subset.

The official release is one uncompressed 121 GB tar.  This tool deliberately
keeps acquisition separate from extraction: the archive can be resumed, then
only the selected official-test members are extracted.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


OFFICIAL_URL = (
    "https://shapenet.cs.stanford.edu/media/minhyuk/spfn/data/"
    "spfn_traceparts_high_res.tar"
)
EXPECTED_POINTS = 131_072
OFFICIAL_SPLIT = (
    "erictuanle/CPFN@74635b02acf8cb3b935fd463db5cc6ce461fa6b8:"
    "Dataset/test_models.csv"
)


def read_split(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [row[0].strip() for row in csv.reader(stream) if row and row[0].strip()]


def evenly_spaced(items: list[str], count: int) -> list[str]:
    if count <= 0 or count > len(items):
        raise ValueError(f"count must be in [1, {len(items)}]")
    if count == 1:
        return [items[0]]
    indices = [(i * (len(items) - 1)) // (count - 1) for i in range(count)]
    return [items[i] for i in indices]


def write_manifest(split: Path, output: Path, count: int, split_reference: str) -> list[str]:
    all_names = read_split(split)
    selected = evenly_spaced(all_names, count)
    payload = {
        "dataset": "SPFN TraceParts high-resolution / CPFN TraceParts_v2",
        "source_url": OFFICIAL_URL,
        "official_split": split_reference,
        "official_split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "selection": "evenly spaced over official CPFN test_models.csv order",
        "test_split_size": len(all_names),
        "subset_size": len(selected),
        "expected_points_per_shape": EXPECTED_POINTS,
        "files": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return selected


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    have = output.stat().st_size if output.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "native-primitive-fit-audit/1"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        if have and status != 206:
            raise RuntimeError("server ignored Range; refusing to overwrite partial archive")
        mode = "ab" if have else "wb"
        with output.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_destination(root: Path, member_name: str) -> Path:
    parts = PurePosixPath(member_name).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise RuntimeError(f"unsafe tar member: {member_name}")
    destination = root.joinpath(*parts).resolve()
    if root.resolve() not in destination.parents:
        raise RuntimeError(f"tar member escapes output root: {member_name}")
    return destination


def extract_subset(archive: Path, manifest: Path, output: Path) -> None:
    wanted = set(load_manifest(manifest)["files"])
    found: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            basename = PurePosixPath(member.name).name
            if member.isfile() and basename in wanted:
                if basename in found:
                    raise RuntimeError(f"duplicate archive basename: {basename}")
                found[basename] = member
        missing = sorted(wanted - found.keys())
        if missing:
            raise RuntimeError(f"archive misses {len(missing)} selected files: {missing[:3]}")
        output.mkdir(parents=True, exist_ok=True)
        for basename in sorted(wanted):
            member = found[basename]
            destination = safe_destination(output, basename)
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def verify(manifest: Path, h5_root: Path) -> None:
    import h5py

    names = load_manifest(manifest)["files"]
    errors = []
    required = {"noisy_points", "gt_points", "gt_normals", "gt_labels"}
    for name in names:
        path = h5_root / name
        if not path.is_file():
            errors.append(f"{name}: missing")
            continue
        try:
            with h5py.File(path, "r") as h5:
                missing = sorted(required - set(h5.keys()))
                counts = {key: len(h5[key]) for key in required if key in h5}
                if missing:
                    errors.append(f"{name}: missing keys {missing}")
                elif any(value != EXPECTED_POINTS for value in counts.values()):
                    errors.append(f"{name}: expected {EXPECTED_POINTS}, got {counts}")
        except Exception as exc:  # report every bad member in one pass
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("high-resolution validation failed:\n  " + "\n  ".join(errors))
    print(f"verified {len(names)} files x {EXPECTED_POINTS} points")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select")
    select.add_argument("--split", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--count", type=int, default=12)
    select.add_argument("--split-reference", default=OFFICIAL_SPLIT)

    fetch = sub.add_parser("download")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--url", default=OFFICIAL_URL)

    extract = sub.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)

    check = sub.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--h5-root", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "select":
        names = write_manifest(args.split, args.output, args.count, args.split_reference)
        print(f"selected {len(names)} files -> {args.output}")
    elif args.command == "download":
        download(args.url, args.output)
    elif args.command == "extract":
        extract_subset(args.archive, args.manifest, args.output)
    else:
        verify(args.manifest, args.h5_root)


if __name__ == "__main__":
    main()
