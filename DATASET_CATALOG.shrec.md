# Primitive-fitting dataset catalog

## Available locally

### BAGSFit TEST-20s

- Location: `D:/download/pcl_eval/samples/bagsfit_full`
- Available samples: 720
- Domain: simulated range-camera scenes
- Development subset previously used by native v2.7: ten frames from scene
  IDs `000`, `001`, `002`, `004`, `006`
- Frozen unseen-scene audit subset: ten frames from scene IDs `010`, `012`,
  `014`, `016`, `018`

The remaining BAGSFit samples are useful for broader same-dataset testing but
are not a different acquisition or shape domain.

### Fit4CAD official test subset

- Official repository: `https://github.com/chiararomanengo/Fit4CAD`
- Source commit: `abe2693789b0021f097aab210f354e8cf95a70c7`
- Raw official subset: `D:/download/pcl_eval/data/Fit4CAD_test12`
- Unified samples: `D:/download/pcl_eval/samples/fit4cad_test12`
- Selected official test IDs: `1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 35`
- Selection rule: 12 IDs evenly spaced over the official test IDs `1..35`,
  fixed before running the fitter
- Total: 12 point clouds, 225,835 points, 272 primitive instances; 266 are
  plane/sphere/cylinder targets
- Domain: clean CAD-object surface samples with varying object scale and point
  count; this differs materially from BAGSFit range images

The official repository contains 225 point clouds split into 190 training and
35 test samples. Only 12 official test samples were exported for this first
cross-dataset audit. Each raw sample retains its point cloud, primitive point
memberships, parametric representation, and implicit representation.

Fit4CAD boundary points can belong to multiple primitives. The unified adapter
labels points with exactly one membership by that primitive and marks zero- or
multi-membership points as `instance_id=-1, primitive_type=none`. Across the
12 samples, 11,875 points are multi-membership boundary points and one point is
unlabelled. This policy avoids arbitrary order-dependent ground truth.

`PC31_parametric.txt` is an empty blob in the official source. Its instance
types and segmentation remain available from the implicit and membership
files, but its primitive parameters are marked unavailable.

Parameter conversion was checked geometrically. For the other 11 samples,
the normalized p95 distance between uniquely assigned target points and their
converted plane/sphere/cylinder ground-truth surfaces is below `1e-6`.

The source license is preserved at
`data/Fit4CAD_test12/licence.txt`. It permits research use under the stated
GPL terms and requires a separate agreement for commercial use. Review the
original license before redistribution or commercial use.

Reproduction:

```powershell
git clone --depth 1 --no-checkout https://github.com/chiararomanengo/Fit4CAD.git data/Fit4CAD
python tools/export_fit4cad_subset.py --repo data/Fit4CAD --output-root data/Fit4CAD_test12
python tools/adapt_fit4cad.py --source-root data/Fit4CAD_test12/dataset/test_set --output-root samples/fit4cad_test12 --source-commit abe2693789b0021f097aab210f354e8cf95a70c7
```

The upstream tree contains filenames ending in a carriage return, which Git
for Windows cannot check out. `export_fit4cad_subset.py` reads only the legal
requested blobs directly from the downloaded Git object database. The raw
`SOURCE_MANIFEST.json` records every upstream Git blob ID, byte count and
SHA-256 checksum.

### SHREC2022 stratified test subset

- Official repository: `https://github.com/chiararomanengo/SHREC2022`
- Source commit: `da2b5f4f5c69896c69ac2777db2d2d499e32f03d`
- Raw subset: `D:/download/pcl_eval/data/SHREC2022_test30`
- Unified samples: `D:/download/pcl_eval/samples/shrec2022_test30`
- Results: `D:/download/pcl_eval/results/shrec2022_audit_v27`
- Total: 30 single-primitive point clouds and 89,922 points

The subset takes the median numeric source ID in every stratum of the three
supported types (plane, cylinder and sphere) and the ten official artifact
classes A0--A9. Selection was frozen from GT type and artifact metadata before
running either fitter. All downloaded point-cloud files match the Git blobs at
the recorded source commit. The exact sample list and SHA-256 hashes are in
`tools/manifests/shrec2022_test30.json`. The upstream GPL-3.0 `LICENSE` and
README are retained with the raw subset.

Unlike BAGSFit and Fit4CAD, this benchmark isolates one primitive segment per
cloud and systematically exercises clean data, uniform/Gaussian noise,
undersampling, missing parts, combined corruptions and local deformation. See
`SHREC2022_COMPARISON.md` for the native v2.7 versus PCL-D4 result.

## Prepared dataset pending upstream recovery

### SPFN / CPFN TraceParts high resolution

- Official SPFN repository: `https://github.com/lingxiaoli94/SPFN`
- Official CPFN repository and test split: `https://github.com/erictuanle/CPFN`
- Intended release: `spfn_traceparts_high_res.tar` (approximately 121 GB)
- Resolution: 131,072 points per object
- Official CPFN test split size: 3,366 objects
- Frozen audit subset: 12 names evenly spaced over the official test-file
  order; see `tools/manifests/spfn_highres_test12.json`

The high-resolution release was selected instead of the 8k-point release so
that this audit exercises dense-cloud behavior and small geometric details.
The sole URL published by both projects was unreachable over HTTP and HTTPS
on 2026-10-08. This is also an upstream-known problem: the SPFN repository has
an open broken-link report from 2023 and CPFN has another from 2025. No
verifiable author-maintained mirror was found, so no low-resolution substitute
or synthetic upsampling is used.

`tools/prepare_spfn_highres.py` provides resumable acquisition, deterministic
subset selection, safe extraction of only the selected files from the large
tar, and validation of the four required HDF5 arrays at exactly 131,072 rows.
Once the official archive or a checksum-verifiable mirror is reachable:

```powershell
python tools/prepare_spfn_highres.py download --output data/spfn_traceparts_high_res.tar
python tools/prepare_spfn_highres.py extract --archive data/spfn_traceparts_high_res.tar --manifest tools/manifests/spfn_highres_test12.json --output data/TraceParts_v2_test12
python tools/prepare_spfn_highres.py verify --manifest tools/manifests/spfn_highres_test12.json --h5-root data/TraceParts_v2_test12
python tools/adapt_spfn.py --split-csv repos/CPFN/Dataset/test_models.csv --h5-root data/TraceParts_v2_test12 --out-root samples/spfn_highres_test12
```

Acquisition remains pending; therefore no TraceParts comparison score is
reported yet.

## Candidate datasets not downloaded

### ParSeNet / ABCParts

This offers a richer CAD surface distribution, including spline patches, but
published preprocessed releases are tens of gigabytes and include primitive
types outside the current plane/sphere/cylinder scope. It was not selected for
the first supplemental download.

## Evaluation status

The Fit4CAD and SHREC2022 subsets have been downloaded, checksummed, converted
and validated, but have not been used to change native v2.7. They are frozen
cross-dataset audit sets. Any future algorithm revision evaluated during
development on them must report them as validation data rather than untouched
test data.
