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

## Candidate datasets not downloaded

### SPFN TraceParts

The official SPFN release provides 8k-point CAD shapes with primitive
membership and parameters, but its standard archive is approximately 9.4 GB
compressed and 12 GB extracted. The high-resolution release is approximately
121 GB. It remains a useful later audit source but is disproportionate for a
small initial download.

### ParSeNet / ABCParts

This offers a richer CAD surface distribution, including spline patches, but
published preprocessed releases are tens of gigabytes and include primitive
types outside the current plane/sphere/cylinder scope. It was not selected for
the first supplemental download.

## Evaluation status

The Fit4CAD subset has been downloaded, checksummed, converted and validated,
but has not been used to change native v2.7. It should remain a frozen
cross-dataset audit set. Any future algorithm revision evaluated during
development on these 12 samples must report them as validation data rather
than untouched test data.
