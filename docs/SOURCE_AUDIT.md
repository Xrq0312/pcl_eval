# Upstream source and reproducibility audit

Audit date: 2026-11-19

## Finding

The clean clone of `https://github.com/Xrq0312/pcl_eval` at commit
`85c50e733fc8e14683c43e4ae61f9cb7841ec654` is an experiment/artifact
repository, not the complete C++ algorithm repository.

- `worktrees/native_v28/` is empty.
- No `CMakeLists.txt`, `primitive_core.cpp`, `primitive_core.hpp`, `main.cpp`
  or C++ tests are present in the Git tree.
- `repos/BAGSFit`, `repos/CPFN`, `repos/SPFN` and `repos/SHREC2022` are gitlink
  entries, but the repository has no `.gitmodules`, so Git cannot initialize
  them from a clean clone.
- `repos/Fit4CAD` contains a bare/partial Git object database rather than a
  normal checkout.
- Reports refer to machine-local paths such as
  `D:/download/cgal_primitive_fit` and `D:/download/pcl_region_primitive`.
  Those paths are not part of this repository.
- Public searches for the documented algorithm version
  `native-v2.7-scene-decomposition` and source commit `10911c3` did not locate
  a public source repository.

Consequently, existing reports cannot be rebuilt from this clone.  The
checked-in prediction files under `samples/synth/synth_00/out_*` are retained
as an immutable artifact baseline.  New code is an independent, auditable
implementation and must not be presented as the missing original source.

## Frozen baseline verification

The existing evaluator was rerun against the shipped synthetic prediction.

| Stage | Target detection | Typed detection | Target mIoU |
|---|---:|---:|---:|
| pre-merge | 1.000 | 1.000 | 0.977 |
| final | 0.833 | 0.833 | 0.727 |

The final stage merged two disjoint coplanar plane instances.  This is a useful
regression case for boundary/connectivity-aware model management.  The point
type accuracy remains 0.978, showing why point classification alone is not a
sufficient instance metric.

## Recovery options

If the author later supplies the original C++ repository, import it as a
separate `legacy/` subtree (or a proper submodule pinned by commit), build it
unchanged, and regenerate the frozen baseline before comparing new stages.
Do not rewrite the new implementation to impersonate the missing history.

