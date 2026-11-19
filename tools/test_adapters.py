"""Dependency-light adapter regression tests. Run via tools/run_python.ps1."""
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_abcparts import convert_one
from adapt_spfn import decode_meta


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


if __name__ == "__main__":
    test_spfn_meta_safety()
    test_abcparts_conversion()
    print("ALL ADAPTER TESTS PASSED")
