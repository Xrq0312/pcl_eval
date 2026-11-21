# Dataset integration status

Audit date: 2026-11-20. Machine-readable facts and license caveats live in
`datasets/registry.json`; this page describes the executable integration path.
No download runs implicitly.

## Tier A: official point-instance supervision

### SPFN TraceParts

The SPFN 8k release is an approximately 9.4 GB ZIP (about 12 GB extracted).
Each HDF5 shape carries noisy/clean points, normals, per-point instance labels
and per-instance primitive metadata. Metadata in the official files may be a
Python pickle, so the adapter refuses to deserialize it unless the user has
verified the archive and passes `--trust-pickle`.

```powershell
./tools/run_python.ps1 tools/adapt_spfn.py --help
```

The CPFN high-resolution TraceParts release has 131072 points per object and a
reported archive size of 121 GB. It is intentionally lower priority until the
8k pipeline is stable; it is the appropriate later audit for multi-scale small
primitive recall and CPU scaling.

### ABCParts (HPNet preprocessing)

The HPNet README distributes an approximately 69 GB preprocessed HDF5 set.
Verified fields are `points`, `normals`, `labels`, `prim`, and `T_param`.
`T_param` contains parameters fitted during HPNet preprocessing, not original
CAD-native parameters, and may be zero for missing/small instances. The adapter
preserves small labeled instances, validates parameter residuals, and records
missing parameters instead of dropping their points.

```powershell
./tools/run_python.ps1 tools/adapt_abcparts.py --help
```

### ABCParts-Mini100 (ParSeNet train-derived)

The user-supplied `data/train_data.h5` is the ParSeNet monolithic training
split: 24,000 objects with 10,000 points each and official point, normal,
instance-label, and primitive-type arrays. It has no `T_param` field or
`train_ids.txt`. A fixed seed and checked-in manifest select 100 HDF5 rows,
with the first 20 frozen as a smoke subset. Plane, sphere, and cylinder
parameters are quality-gated derived references, not original CAD analytic GT.

```powershell
./tools/run_python.ps1 tools/prepare_abcparts_mini.py `
  --input data/train_data.h5 `
  --manifest datasets/abcparts_mini100_manifest.json `
  --out-root samples/abcparts_mini100 --reuse-manifest
```

This set is a reproducible train-derived stress benchmark, not an official
ABCParts test score. The full protocol and results are in
`docs/ABCPARTS_MINI_REPORT.md`.

## Tier B: controlled derived references

### BA Primitive Fitting Dataset

The verified Git revision contains 1773 headerless CSVs: 591 shapes in each of
`clean`, `error`, and `error_double`; every file has 1000 rows of
`x,y,z,nx,ny,nz,label`. The label column is constant and no primitive
parameters are stored. Boxes contain six faces, and cylinder files contain the
side plus planar caps. Consequently this is not official point-instance GT.

For controlled noise tests, `adapt_ba.py` derives pseudo-GT from the paired
clean file, transfers only row-aligned masks to the noisy version, and stamps
every output with `annotation_status=derived_pseudo_gt`. It requires explicit
acknowledgement:

```powershell
./tools/run_python.ps1 tools/adapt_ba.py `
  --root data/upstream/BA-Primitive_Fitting_Dataset `
  --output data/ba_eval --variant error --allow-derived-gt
```

Sphere points form one instance; box faces are grouped by clean oriented
normals; only the cylinder side is assigned to the cylinder and caps remain
`none`. These references must never be described as official BA annotations.

## Tier C: no official point-instance supervision

### GCCao Cylinders_detection_Cao_V2

The repository contains two industrial pipeline PLY scans, MATLAB code, and
MAT/TXT intermediate or detected-cylinder outputs. The detected models are not
independent annotations. Use the audit tool to inventory a populated checkout;
quantitative precision/recall remains blocked until a separately versioned
manual annotation protocol exists.

```powershell
./tools/run_python.ps1 tools/audit_external_dataset.py `
  --dataset gccao --root data/upstream/Cylinders_detection_Cao_V2 `
  --output results/audits/gccao.json
```

### Primitive Object Dataset

The OSF release described by DOI `10.17605/OSF.IO/4AW5U` contains 8000 virtual
and 200 RealSense SR300 PCD scenes. TXT records provide primitive type,
position xyz, quaternion xyzw and dimensions xyz, but no per-point object mask;
real scenes include table/background. The audit validates PCD headers and pose
record shape without generating fake `gt_labels.csv` files.

```powershell
./tools/run_python.ps1 tools/audit_external_dataset.py `
  --dataset primitive_object --root data/primitive_object `
  --output results/audits/primitive_object.json
```

Pose/dimension error and runtime are valid native metrics. Point IoU is not an
official metric unless a declared, independently versioned masking protocol is
added later.

## Regression tests

`tools/test_adapters.py` covers safe SPFN metadata handling, ABCParts schema
conversion, the BA explicit pseudo-GT gate, and both unlabeled-dataset audit
paths. It uses synthetic fixtures and requires no dataset download.
