# Native primitive fitting implementation report

Date: 2026-09-30

## Delivered implementation

- Target: `D:\download\cgal_primitive_fit`
- Executable: `build-native\native_primitive_fit.exe`
- Core dependencies: C++17 STL and Eigen3
- PCL use: `common`/`io` boundary only, for PCD loading and writing
- Algorithms implemented locally: balanced KD tree, PCA normals, region
  growing, plane/sphere/cylinder robust fitting, geometric merge and
  normal-constrained absorption
- Historical CGAL files remain present but are not included or linked.
- Original wrapper files were copied to
  `legacy_cgal_wrapper_20260928` before replacement.

## Validation

CTest: 1 test executable, all checks passed. Coverage includes KD-tree KNN
and radius queries, plane/sphere/cylinder least-squares recovery, degenerate
geometry, invalid normals/configuration, plane/cylinder sign equivalences,
and deterministic labels.

Synthetic integration (`samples/synth/synth_00`):

| Metric | Native |
|---|---:|
| target detection | 1.000 |
| typed detection | 1.000 |
| target mIoU | 0.980 |
| target point-type accuracy | 0.984 |
| unassigned ratio | 0.022 |

BAGSFit 10-sample comparison, identical evaluator and samples:

| Metric (mean) | Old CGAL | Native |
|---|---:|---:|
| target detection | 0.000 | 0.565 |
| typed detection | 0.000 | 0.510 |
| target mIoU | 0.047 | 0.535 |
| target point-type accuracy | 0.417 | 0.816 |
| unassigned ratio | 0.029 | 0.149 |
| wall seconds/sample | 9.70 | 25.27 |
| non-finite models | 0 | 0 |

The native implementation meets the four quality acceptance metrics. Its
higher unassigned ratio is expected from conservative distance/normal gates;
the old CGAL result assigned 97% of points but split them into many incorrect
models and detected no target instances.

Raw outputs are under `results/native_migration/baseline_cgal` and
`results/native_migration/native_default`.

## Efficiency optimization

The second pass parallelizes normal estimation and independent region fits
with deterministic per-region random streams. Dense radius adjacency checks
now stop at the first marked neighbor instead of collecting and sorting every
point in the radius. `--threads=0` uses hardware concurrency and
`--threads=N` provides an explicit cap.

BAGSFit 10-sample means before/after optimization:

| Metric | Initial native | Optimized native |
|---|---:|---:|
| wall seconds/sample | 25.27 | 6.08 |
| normal estimation seconds | 15.53 | 1.01 |
| fitting seconds | 5.19 | 3.48 |
| merge seconds | 3.23 | 0.41 |
| target detection | 0.565 | 0.579 |
| typed detection | 0.510 | 0.543 |
| target mIoU | 0.535 | 0.551 |
| target point-type accuracy | 0.816 | 0.812 |

Overall wall time improved by 4.16x. On `000_00000`, 20-thread and
single-thread runs produced byte-identical labels and identical model data;
the measured algorithm phase time was 4.06 s versus 19.37 s (4.77x).
Optimized raw outputs are under `results/native_migration/native_optimized`.

## D4 hybrid model-management experiment

The model-management ideas were independently reimplemented from the read-only
local D4/D5 repository at commit `45df91a` (`v1.0-d4d5`). No PCL search,
feature, segmentation, sample-consensus code, or Git history was copied. The
new output contract separates exclusive `ownedPoints` from strict
`coreInliers`, performs absorption without moving parameters, and emits
`_operations.csv` plus `_lineage.json`.

The requested staged selection was run on the same ten BAGSFit samples. The
full conservative candidate included transactional RMSE/parameter/retention
checks, hard KNN support, and adjacent plane-fragment recovery. A second
candidate used KNN support only to choose between eligible model owners.

| 10-sample mean | Native v2.1 | PCL-D4 reference | Dual-set stage | Full conservative | Support-priority |
|---|---:|---:|---:|---:|---:|
| target detection | 0.579 | 0.654 | 0.579 | 0.515 | 0.570 |
| typed detection | 0.543 | 0.636 | 0.543 | 0.479 | 0.534 |
| target mIoU | 0.551 | 0.614 | 0.551 | 0.528 | 0.552 |
| point-type accuracy | 0.812 | 0.811 | 0.812 | 0.755 | 0.812 |
| unassigned ratio | 0.153 | 0.181 | 0.153 | 0.224 | 0.153 |
| wall seconds/sample | 6.03 | 7.54 | 6.31 | 6.14 | 6.53 |
| non-finite models | 0 | 0 | 0 | 0 | 0 |

Only the dual-set stage meets the quality/throughput selection rule (values are
rounded to the specified three-decimal acceptance precision). It is therefore
the sole default in `native-v2.2-d4-hybrid`; the conservative candidates are
retained as evaluation evidence, not as runtime modes. Raw outputs are under
`results/native_migration/d4_stage2_dual`,
`d4_stage3_transactional`, and `d4_stage2_supported`.

The selected synthetic run reached target detection `1.000`, typed detection
`1.000`, target mIoU `0.980`, point-type accuracy `0.984`, and had no non-finite
model. All ten BAGSFit samples completed without crashes or duplicate point
ownership.

## Native D4 replica

The follow-up implementation reproduces the quality-positive D4 behavior more
directly instead of applying D5's conservative gates. Compatible fragments are
merged iteratively, their fixed primitive type is robustly refitted, the full
point union remains owned, and only the strict-distance subset becomes the new
core. A final global distance/normal pass absorbs unassigned points without
changing model parameters. Spatial connectivity remains bounded at
`10*distance`, so separated coplanar instances are not merged.

For efficiency, each merge ranks at most 32 hypotheses on a deterministic
uniform subset of at most 4096 owned points. The selected hypothesis is then
refitted and evaluated on the complete union. This reduced the initial literal
native D4 experiment from 9.73 seconds/sample to 6.89 seconds/sample.

| 10-sample mean | Native v2.2 | Native D4 replica | PCL-D4 reference |
|---|---:|---:|---:|
| target detection | 0.579 | 0.589 | 0.654 |
| typed detection | 0.543 | 0.543 | 0.636 |
| target mIoU | 0.551 | 0.579 | 0.614 |
| point-type accuracy | 0.812 | 0.819 | 0.811 |
| unassigned ratio | 0.153 | 0.140 | 0.181 |
| wall seconds/sample | 6.31 | 6.89 | 7.54 |
| non-finite models | 0 | 0 | 0 |

The replica improves every native quality metric except typed detection, which
is unchanged, while remaining below the seven-second requirement. It closes
about 45% of the mIoU gap from native v2.2 to PCL-D4 without adding PCL
algorithm dependencies. Raw outputs are under
`results/native_migration/d4_native_replica_cap32_final`.

## Normal-aware joint cylinder refinement

Version `native-v2.4-cylinder-joint` adds the cylinder-specific behavior that
was still missing from the native D4 replica: after RANSAC selects support, a
native bounded LM optimizer jointly updates the axis line, axis direction, and
radius from radial point residuals and sign-invariant normal-angle residuals.
The default normal weight is `0.10`, matching the PCL-D4 reference setting.
For efficiency, only the selected support is optimized, at most 2048
deterministically sampled points are used, and the default is two iterations.
A transaction gate rejects a result if it loses any strict core support or its
native score grows by more than 10 percent.

| 10-sample mean | Native D4 replica | Cylinder joint |
|---|---:|---:|
| target detection | 0.589 | 0.602 |
| typed detection | 0.543 | 0.555 |
| target mIoU | 0.579 | 0.582 |
| point-type accuracy | 0.819 | 0.823 |
| unassigned ratio | 0.140 | 0.138 |
| executable wall seconds/sample | 6.89 | 6.95 |
| non-finite models | 0 | 0 |

Matched-cylinder parameter means also improved: axis angle `1.627° → 1.246°`,
axis-line distance `0.00832 → 0.00557`, and absolute radius error
`0.00900 → 0.00742`. The mixed synthetic scene retained detection and typed
detection `1.000`, reached mIoU `0.983`, point-type accuracy `0.987`, and had
no non-finite model. Thread counts 1 and 4 produced byte-identical labels,
models, and operation logs. Raw outputs are under
`results/native_migration/cylinder_joint_final10` and
`results/native_migration/cylinder_joint_final_synth`.

## Plane fragmentation diagnosis and Plane Merge v2

The v2.5 work began with a GT-only oracle rather than merge-threshold tuning.
Prediction fragments are assigned to their majority GT instance only when
their purity is at least 0.5. For each plane the diagnostic reports largest
fragment recall/IoU, a same-type oracle union, an all-type oracle union, and
the number of fragments required for 90% coverage.

On the v2.4 ten-sample output, the 54 target planes decomposed as follows:

| Classification | Planes |
|---|---:|
| already detected | 25 |
| fragmentation error | 13 |
| type error | 2 |
| coverage error | 14 |

For the 13 pure fragmentation cases, mean largest-fragment IoU was only 0.372
while mean same-type oracle merge IoU was 0.810. This confirmed that instance
assembly, rather than point recall alone, was the main actionable bottleneck.

Plane Merge v2 runs after global absorption, when the ownership evidence is
available. It performs deterministic best-first agglomeration using projected
2D boundary cells, local-spacing-normalized gaps, observed-normal agreement,
source-core retention, and p90 residual checks. Plane sufficient statistics
allow every trial refit to use one 3x3 eigendecomposition. The absolute gap
remains a safety ceiling. A bounded-occlusion synthetic case merges, while the
existing spatially separated coplanar case remains separate.

| 10-sample mean | v2.4 cylinder joint | v2.5 plane merge |
|---|---:|---:|
| target detection | 0.602 | 0.632 |
| typed detection | 0.555 | 0.586 |
| target mIoU | 0.582 | 0.597 |
| point-type accuracy | 0.823 | 0.823 |
| unassigned ratio | 0.138 | 0.138 |
| final target models | 55.3 | 51.8 |
| executable wall seconds/sample | 6.95 | 3.39 |
| non-finite models | 0 | 0 |

The point metrics are unchanged because this stage only unions existing
exclusive ownership. The oracle classification after v2.5 contains 28
detected, 10 remaining fragmentation, 2 type, and 14 coverage planes.

The speedup comes from deterministic largest-first dynamic region scheduling.
Every region keeps its id-derived RNG stream, so 1-, 4-, and 16-thread
synthetic outputs are byte-identical; the dense `001_00000` labels, models,
and operations are also byte-identical between 1 and 4 threads. Two-level
hypothesis verification and adaptive RANSAC were benchmarked but rejected:
although faster, they changed model selection and reduced detection.

Raw outputs are under
`results/native_migration/v25_plane_merge_contact_final10`;
the oracle report is `fragmentation_oracle.json` in that directory.

## v2.6a coverage diagnosis

Version v2.6a adds observation only; candidate generation, scoring, merging,
absorption, model order, and labels are unchanged. With matched synthetic
settings, v2.5 and v2.6a labels and model CSV files are byte-identical.
Runtime outputs now expose the full plane coverage funnel and terminal merge
gate measurements, while all GT joins remain in the offline oracle.

The BAGSFit 10 quality metrics therefore remain target detection `0.632197`,
typed detection `0.585833`, mIoU `0.597446`, point-type accuracy `0.822767`,
and unassigned ratio `0.138229`. The 54 GT planes still split into 28 detected,
10 fragmentation, 2 type, and 14 coverage errors.

The stricter funnel counts a point as candidate-covered only if it belongs to
that region's plane candidate support. This matters for large mixed regions:
merely saying that a region produced a plane candidate substantially
overestimated coverage. The resulting diagnosis is:

| Coverage diagnosis | Planes |
|---|---:|
| front end has <50% strict plane-candidate support | 12 |
| candidate exists but final plane ownership is insufficient | 2 |
| candidate lost/rejected after >=50% strict support | 0 |
| predominantly owned by a wrong primitive type | 0 |

Coverage error is strongly size-concentrated: 11 are in the smallest GT-size
tertile (at most 1471 points), 3 are medium, and none are large. Two tiny GT
annotations contain only 1 and 4 points; these cannot satisfy the configured
80-point support and should not drive a global threshold change. The two
large-looking medium failures (8511 and 11210 points) sit inside mixed regions,
but their plane candidates strictly explain only 229 and 1892 GT points. This
points to region separation / local candidate recovery, not broader global
post-merge absorption, as the next high-value experiment.

For each of the ten fragmentation errors, the rejection histogram uses only
the two largest same-type fragments, avoiding the combinatorial bias of all
fragment pairs. Seven primary pairs fail source-core retention and three fail
the local-spacing-normalized gap. Boundary contact and p90 are not the primary
blocker for these ten cases, so globally weakening every merge gate is not
supported by this diagnosis.

The final stability benchmark used fixed 16 threads and three runs over the
same ten samples. Across the three per-run ten-sample means, executable wall
time was median `4.320s`, p10 `4.318s`, and p90 `4.347s`; algorithm-core time
was median `2.592s`. The cold-run mean was `4.318s`, and every individual
sample remained below seven seconds (maximum `6.648s`). Raw outputs are under
`results/native_migration/v26a_diagnostics_benchmark_final`; `summary.json`,
`aggregate.csv`, and `coverage_oracle.json` contain the benchmark, quality,
and diagnosis respectively.

## v2.7 scene decomposition and local continuity

The v2.6a coverage funnel showed that candidate generation, rather than
post-fit absorption, was the dominant plane failure: 12 of 14 coverage misses
had less than 50% strict plane-candidate support. A region-growing component
may contain several surfaces, so choosing only one primitive per region is a
structural bottleneck. v2.7 fits at most three disjoint hypotheses per region.
The primary hypothesis retains its original random stream and entire original
merge/absorption path. Secondary residual hypotheses are admitted only after
that path completes and only for still-unassigned points. This preserves the
primary models while recovering previously hidden surfaces.

The remaining fragmentation problem was different: global refit retention
rejected continuous but mildly bent or nonuniformly noisy large planes. Plane
Merge v2 therefore retains its shared spatial, boundary-contact, observed-
normal and candidate-normal gates, then accepts either global fit evidence
(source-core retention and merged p90) or local continuity evidence (source
normal agreement and p90 boundary tangent-crossing residual within
`3*distance`). The global route receives priority in best-first ranking.
The absolute spatial-gap ceiling still prevents merging separated coplanar
patches. Both routes preserve full ownership and exclusive labels.

| BAGSFit 10 mean | Native v2.6a | PCL-D4 reference | Native v2.7 |
|---|---:|---:|---:|
| target detection | 0.632197 | 0.654 | 0.727576 |
| typed detection | 0.585833 | 0.636 | 0.681212 |
| target mIoU | 0.597446 | 0.614 | 0.655049 |
| point-type accuracy | 0.822767 | 0.811 | 0.850540 |
| unassigned ratio | 0.138229 | 0.181 | 0.060943 |
| executable wall seconds/sample | 4.320 | 7.54 | 5.058 |

The native times above use fixed 16-thread, three-run medians of the ten-
sample mean, rather than a single hot/cold run. v2.7's per-run mean executable
time had median `5.057524s`, p10 `5.052175s` and p90 `5.061401s`; median
algorithm-core time was `3.248562s`. One individual sample took about `8.11s`,
so the seven-second result is a **ten-sample mean**, not a per-sample bound.
All ten samples ran without crashes, non-finite coefficients or duplicate
ownership; label, model and operation CSVs were byte-identical across three
runs. Every sample's Det, Typed Det and mIoU was at least its v2.6a value.

The offline 54-plane oracle now classifies 37 as detected, 2 fragmentation,
2 type, 11 coverage and 2 mixed; v2.6a had 28 detected, 10 fragmentation,
2 type and 14 coverage. The remaining 11 coverage misses are all in the
smallest GT-size tertile. On matched, correctly typed instances, the mean
plane normal-angle error changed `1.725° -> 1.558°` (28 vs 37 matches),
the sphere center error stayed `0.02443` (16 matches), and cylinder axis-angle
error changed `1.246° -> 1.203°` (18 vs 19 matches). These are conditional
means over different matched sets, not paired parameter improvements.

Rejected experiments were immediate secondary-model commitment (typed Det
`0.573`, a regression), five models per region (no Det/Typed gain over three),
extra generic surface-normal score weight (no gain), core-only plane statistics
(regression), and simple source-retention relaxation (smaller gain than local
continuity). The final mechanism therefore changes region decomposition and
geometric continuity evidence, not a globally looser threshold.

Tests cover residual extraction, unchanged primary proposals and ownership,
bounded bent-plane continuity, and separated coplanar instances. The mixed
synthetic scene reached Det `1.0`, Typed Det `1.0`, mIoU `0.983` and point
accuracy `0.987`. Raw repeat outputs, quality aggregate, timing summary and
GT-only oracle are in
`results/native_migration/v27_scene_decomposition_final3`.

## Dependency audit

Searches over active source, CMake, and generated Ninja link rules found no
CGAL references and no PCL features/search/segmentation/sample-consensus
headers, symbols, or libraries.
