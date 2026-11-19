"""Dependency-light adapter regression tests. Run via tools/run_python.ps1."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_abcparts import convert_one
from adapt_ba import convert_one as convert_ba
from adapt_spfn import decode_meta
from audit_external_dataset import audit_gccao, audit_primitive_object


def test_spfn_meta_safety():
    assert decode_meta("{'type': 'plane'}") == {"type": "plane"}
    try:
        decode_meta(np.void(b"not trusted"))
    except RuntimeError as error:
        assert "--trust-pickle" in str(error)
    else:
        raise AssertionError("pickle metadata must require explicit trust")


def test_abcparts_conversion():
    temp = tempfile.mkdtemp(prefix="abc_adapter_")
    try:
        source = os.path.join(temp, "00000.h5")
        output = os.path.join(temp, "out")
        rng = np.random.RandomState(7)
        plane = np.column_stack((rng.uniform(-1, 1, 160),
                                 rng.uniform(-1, 1, 160),
                                 np.full(160, 0.25)))
        angles = rng.uniform(0, 2 * np.pi, 180)
        cylinder = np.column_stack((0.4 * np.cos(angles),
                                    0.4 * np.sin(angles),
                                    rng.uniform(-1, 1, 180)))
        points = np.vstack((plane, cylinder))
        normals = np.vstack((np.tile([0, 0, 1.0], (160, 1)),
                             np.column_stack((np.cos(angles),
                                              np.sin(angles),
                                              np.zeros(180)))))
        labels = np.r_[np.zeros(160, dtype=int), np.ones(180, dtype=int)]
        types = np.r_[np.ones(160, dtype=int), np.full(180, 4, dtype=int)]
        params = np.zeros((340, 22), dtype=float)
        params[:160, 4:8] = [0, 0, 1, 0.25]
        params[160:, 8:15] = [0, 0, 1, 0, 0, 0, 0.4]
        with h5py.File(source, "w") as handle:
            handle["points"] = points
            handle["normals"] = normals
            handle["labels"] = labels
            handle["prim"] = types
            handle["T_param"] = params
        result = convert_one(source, output)
        assert result["instances"] == 2
        with open(os.path.join(output, "gt_primitives.json")) as stream:
            primitives = json.load(stream)["primitives"]
        assert [item["type"] for item in primitives] == ["plane", "cylinder"]
        assert max(item["source_param_p95_residual"] for item in primitives) < 1e-10
        with open(os.path.join(output, "gt_labels.csv")) as stream:
            assert sum(1 for _ in stream) == 341
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def test_ba_requires_explicit_pseudo_gt():
    temp = tempfile.mkdtemp(prefix="ba_adapter_")
    try:
        source_dir = os.path.join(temp, "shapes", "clean", "spheres")
        os.makedirs(source_dir)
        source = os.path.join(source_dir, "s_000.csv")
        rng = np.random.RandomState(3)
        normals = rng.normal(size=(1000, 3))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        np.savetxt(source, np.column_stack((normals, normals, np.zeros(1000))), delimiter=",")
        try:
            convert_ba(source, source, os.path.join(temp, "denied"))
        except RuntimeError as error:
            assert "--allow-derived-gt" in str(error)
        else:
            raise AssertionError("derived BA GT must require explicit opt-in")
        result = convert_ba(source, source, os.path.join(temp, "allowed"), True)
        assert result["instances"] == 1 and result["assigned"] == 1000
        metadata = json.load(open(os.path.join(temp, "allowed", "metadata.json")))
        assert metadata["annotation_status"] == "derived_pseudo_gt"
        assert metadata["official_point_instance_gt"] is False
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def test_unlabeled_dataset_audits():
    temp = tempfile.mkdtemp(prefix="external_audit_")
    try:
        gccao = os.path.join(temp, "gccao")
        os.makedirs(gccao)
        with open(os.path.join(gccao, "scan.ply"), "wb") as stream:
            stream.write(b"ply\nformat ascii 1.0\nelement vertex 1\n"
                         b"property float x\nproperty float y\nproperty float z\n"
                         b"end_header\n0 0 0\n")
        open(os.path.join(gccao, "detected.mat"), "wb").write(b"not ground truth")
        result = audit_gccao(Path(gccao))
        assert result["raw_ply"][0]["vertices"] == 1
        assert result["official_point_instance_gt"] is False

        primitive = os.path.join(temp, "primitive")
        os.makedirs(primitive)
        with open(os.path.join(primitive, "real_001.pcd"), "wb") as stream:
            stream.write(b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\n"
                         b"SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                         b"WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n0 0 0\n")
        with open(os.path.join(primitive, "real_001.txt"), "w") as stream:
            stream.write("cylinder 0 0 1 0 0 0 1 0.2 0.2 1\n")
        result = audit_primitive_object(Path(primitive))
        assert result["pcd_count"] == 1
        assert result["samples"][0]["objects"][0]["type"] == "cylinder"
        assert result["official_point_instance_gt"] is False
    finally:
        shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    test_spfn_meta_safety()
    test_abcparts_conversion()
    test_ba_requires_explicit_pseudo_gt()
    test_unlabeled_dataset_audits()
    print("ALL ADAPTER TESTS PASSED")
