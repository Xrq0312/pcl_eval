# Native point-cloud primitive fitting

C++17 implementation of plane, sphere and cylinder segmentation without
CGAL and without PCL feature/search/segmentation/sample-consensus operators.
The algorithmic core uses only STL and Eigen. PCL `common`/`io` is confined
to the CLI adapter for PCD loading and writing.

## Build

```powershell
cmake -S . -B build-native -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-native -j
ctest --test-dir build-native --output-on-failure
```

## Run

```powershell
build-native/native_primitive_fit.exe points.txt results/out
build-native/native_primitive_fit.exe cloud.pcd results/out --distance=0.02 --normal-k=300
```

Run `native_primitive_fit --help` for all options. Outputs retain the former
labels CSV, models CSV, primitives JSON, segmented PCD and unassigned PCD
schemas so `D:/download/pcl_eval/tools/evaluate.py` can read them directly.
The hybrid build additionally writes `<prefix>_operations.csv` and
`<prefix>_lineage.json`. Models keep separate `ownedPoints` (exclusive point
ownership) and `coreInliers` (strict-threshold fit support); absorbed points
extend ownership without moving the fitted parameters.

Normal estimation and per-region fitting run in parallel. `--threads=0`
(the default) uses the machine's hardware concurrency; set `--threads=1` for
single-threaded benchmarking or a positive value to cap CPU usage. Results
remain deterministic for a fixed seed regardless of thread count. Region fits
use a largest-first dynamic task queue, but every random stream is derived from
the region id rather than the worker thread.

## Native D4 replica

The model-management design is adapted from the local D4/D5 reference at
commit `45df91a` without importing its PCL algorithms or Git history. The one
default pipeline reproduces D4's important behavior:

- iteratively merge same-type, parameter-compatible fragments within a bounded
  spatial gap (`--merge-max-gap`, default `10*distance`);
- robustly refit the fixed primitive type while retaining the complete union as
  ownership and keeping strict-distance points as the core;
- globally absorb unassigned points using surface distance and normal agreement
  without letting absorbed points move the parameters.

Merge hypotheses are ranked on a deterministic 4096-point subset with at most
32 trials, then re-estimated and evaluated on the complete owned set. This is
the main speed optimization over a literal full-cloud D4 refit. The default
absorption normal tolerance is 35 degrees. The legacy
`--fragment-max-gap` spelling remains an alias for `--merge-max-gap`; D5
neighbor-support options remain accepted for command compatibility but are not
hard gates in the D4 pipeline.

There is no public `fast`, `d4`, or `d5` mode: the executable has one
deterministic default pipeline.

## Scene decomposition (v2.7)

Region growing is an adjacency proposal, not a promise that a region contains
only one primitive. Each region now extracts up to three disjoint robust-fit
hypotheses (`--max-region-models=3`). The first hypothesis follows the original
pipeline unchanged. Secondary hypotheses are admitted only after primary
merging and global absorption, and may claim only still-unassigned points.
This recovers surfaces hidden inside mixed regions without changing existing
primary ownership. `--max-region-models=1` disables residual extraction.

## Normal-aware cylinder refinement

The selected cylinder support is refined with a native bounded LM step over
five independent parameters: two axis-line offsets, two axis-direction
increments, and radius. Its objective combines signed radial point distance
with sign-invariant normal/radial angle, matching the useful behavior of
PCL's normal-aware cylinder SAC without calling PCL sample-consensus code.
`--cylinder-normal-weight=0.10` controls the angular term and
`--cylinder-refine-iterations=2` controls the bounded iteration count (zero
disables this final refinement). The optimizer runs only after robust support
selection and commits atomically only when strict core support is preserved.
The value `0.10` follows the explicitly configured local PCL-D4 reference and
PCL cylinder example; it is not the PCL model class's hard-coded default.

## Plane Merge v2

After global ownership absorption, remaining plane models are consolidated by
a deterministic best-first merge. Each plane caches sufficient statistics
`N`, `sum(p)` and `sum(p*p^T)`, a local point-spacing estimate, an observed
normal, and boundary cells from a projected 2D grid. Candidate edges use a
gap normalized by local spacing rather than only a fixed physical distance.
Every accepted merge must pass observed/candidate normal, spatial-gap and
boundary-contact gates. It then needs one of two evidence routes: a global
refit that preserves the configured fraction of each source core
(`--plane-merge-source-retention=0.98`) and passes p90 residual, or local
continuity between source normals and boundary tangent crossings (p90 at most
`3*distance`). Global-evidence edges are preferred. The local route permits a
mildly bent or nonuniformly noisy plane to remain one instance without
relaxing the absolute `--merge-max-gap` safety ceiling; spatially separate
coplanar instances still cannot merge. Accepted merges record
`accepted_global` or `accepted_local` in the operations CSV.

`tools/fragmentation_oracle.py` is an offline GT-only diagnostic. It reports
largest-fragment recall/IoU, same-type and all-type oracle merge IoU, fragment
count at 90% coverage, and classifies plane misses as fragmentation, type,
coverage, or mixed errors. GT data is never used by the runtime fitter.

## Coverage diagnostics

Version v2.6a adds read-only stage tracing without changing model selection.
Each run also writes:

- `<prefix>_diagnostics.csv`: per-point finite/normal/region state, whether the
  region produced a plane candidate, whether the point was a strict candidate
  inlier, and its premerge, post-absorption and final owner type;
- `<prefix>_region_diagnostics.csv`: candidate validity, support, RMSE, inlier
  ratio and score for all three primitive types in every region;
- `<prefix>_plane_merge_rejections.csv`: the terminal unmerged plane pairs,
  their first failed gate, and the measured angle/gap/contact/retention/p90
  values.

The oracle combines these files with GT only after fitting. Its aggregate JSON
contains a coverage funnel, per-instance ownership confusion, small/medium/
large plane summaries, coverage subclasses, and a one-primary-pair rejection
histogram for the remaining fragmentation cases. `plane_candidate_inlier` is
the strict funnel stage; `region_plane_candidate` only says that the containing
region produced some plane candidate and can overstate support in mixed regions.

For reproducible timing, run (sample names may be supplied explicitly):

```powershell
python tools/benchmark_stability.py `
  --executable build-native/native_primitive_fit.exe `
  --samples-root D:/download/pcl_eval/samples/bagsfit_full `
  --output-root D:/download/pcl_eval/results/native-v2.7-benchmark `
  --runs 3 --threads 16
```

Historical CGAL source archives and build artifacts were removed during the
2026-10-08 workspace cleanup. They were never included, linked or used by the
native target; previous tracked source states remain available from local Git.
