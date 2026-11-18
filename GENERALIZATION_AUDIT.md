# Native v2.7 generalization audit (2026-10-08)

This audit freezes `native-v2.7-scene-decomposition` at Git merge commit
`10911c3`; neither code nor defaults were changed in response to these cases.
The local PCL-D4 reference is commit `45df91a`. Both executables were run on
the same point order and evaluated with `tools/evaluate.py` (one-to-one target
matching at IoU 0.5). Executable SHA-256 hashes and exact scene IDs are saved
in `results/generalization_audit_v27_full/protocol.json`.

## Unseen-scene BAGSFit audit

The predeclared ten frames are `00000` and `00018` from scene IDs `010`,
`012`, `014`, `016`, `018`. The ten frames used to develop v2.7 came from
different scene IDs (`000`, `001`, `002`, `004`, `006`). These are five unseen
scenes within the **same simulated-range-camera dataset**, not an independent
sensor or geometry distribution. Two frames per scene are correlated, so the
table must not be read as ten independent scene trials.

| Mean over ten frames | Native v2.7 | PCL-D4 | Native minus D4 |
|---|---:|---:|---:|
| target detection | 0.7736 | 0.7527 | +0.0209 |
| typed detection | 0.7345 | 0.7064 | +0.0282 |
| target mIoU | 0.6941 | 0.6503 | +0.0437 |
| point-type accuracy | 0.8819 | 0.8263 | +0.0556 |
| unassigned ratio | 0.0681 | 0.1624 | -0.0943 |
| process wall seconds | 5.33 | 9.63 | -4.29 |

All 20 runs completed, with finite model coefficients and model point totals
consistent with the exclusive label assignments. Native used 16 threads; wall
times are one run per sample, not a repeated controlled throughput benchmark.
The paired mIoU difference is positive on eight frames and negative on both
`016` frames. On `016_00000`, GT plane 7 has native IoU `0.4818` and is
matched to a cylinder, while D4 reaches plane IoU `0.8572`. On `016_00018`,
GT plane 8 is matched to a sphere at native IoU `0.2408`, versus D4 plane
IoU `0.8179`. Thus the positive mean hides a real plane competition/ownership
regression; no threshold was changed after observing it.

Per-run outputs and evaluations are under
`D:/download/pcl_eval/results/generalization_audit_v27_full`. The frozen
protocol runner is `D:/download/pcl_eval/tools/generalization_audit.py`.

## Deterministic synthetic stress audit

`tools/make_stress_cases.py` generated a fresh-seed (`20261008`) 12,000-point
synthetic scene and then applied isotropic scale changes, Bernoulli thinning,
or additional 2 cm Gaussian coordinate noise. This probes invariances but is
**not independent real-world evidence**: topology matches the synthetic scene
used during development. Unless stated otherwise, native v2.7 used unchanged
defaults.

| Case | Det | Typed | mIoU | Point acc. | Unassigned |
|---|---:|---:|---:|---:|---:|
| nominal | 1.000 | 1.000 | 0.964 | 0.967 | 0.040 |
| scale x0.1, default thresholds | 0.500 | 0.500 | 0.542 | 0.966 | 0.038 |
| scale x10, default thresholds | 0.667 | 0.667 | 0.686 | 0.929 | 0.078 |
| scale x0.1, distance/radius/gap x0.1 | 1.000 | 1.000 | 0.964 | 0.966 | 0.040 |
| scale x10, distance/radius/gap x10 | 1.000 | 1.000 | 0.964 | 0.967 | 0.040 |
| 25% density, unchanged defaults | 0.667 | 0.667 | 0.615 | 0.524 | 0.478 |
| 10% density, unchanged defaults | 0.167 | 0.167 | 0.084 | 0.331 | 0.673 |
| 25% density, count parameters x0.25 | 1.000 | 1.000 | 0.959 | 0.965 | 0.038 |
| 10% density, count parameters x0.10 | 0.667 | 0.667 | 0.579 | 0.796 | 0.120 |
| +2 cm coordinate noise, unchanged defaults | 0.667 | 0.667 | 0.511 | 0.867 | 0.127 |

The scale-calibrated runs scaled `distance`, `merge-distance`, `min-radius`,
`max-radius`, and `merge-max-gap` by the known coordinate factor. The
density-calibrated runs scaled `normal-k`, `region-neighbors`, and
`min-points` by the known retention fraction, rounded to valid integers.
These are **controlled sensitivity checks using known transforms**, not a
deployable automatic estimator. The 10%-density case still degrades sharply
after calibration, and extra noise also hurts. Reproduced raw outputs,
evaluations and exact option lists are under
`D:/download/pcl_eval/results/generalization_stress_v27_repro`.

The audit scripts run with Python 3, NumPy and SciPy. From the project root,
the two commands are:

```powershell
python tools/generalization_audit.py --samples-root D:/download/pcl_eval/samples/bagsfit_full --output-root D:/download/pcl_eval/results/generalization_audit_v27_full --native-exe build-native/native_primitive_fit.exe --pcl-d4-exe D:/download/pcl_region_primitive/pcl_region_primitive/build/primitive_fit.exe
python tools/run_stress_audit.py --native-exe build-native/native_primitive_fit.exe --output-root D:/download/pcl_eval/results/generalization_stress_v27_repro
```

## Implications for the next algorithm version

1. Keep v2.7 frozen. Do not tune global constants against the ten audit
   frames or stress cases and then count them as independent validation.
2. Replace fixed-count neighborhoods/support with local physical support:
   estimate local point spacing and normal uncertainty, choose a neighborhood
   whose spatial extent is meaningful, and test whether a region contains
   enough *effective surface area* rather than a fixed number of samples.
   A global density fraction is not sufficient for nonuniform scans.
3. Estimate a robust scene or local distance/noise scale, then express
   geometric gates in that scale. The exact scale-transform recovery shows
   the current geometry is near scale-equivariant **when units are supplied**;
   the unchanged defaults are not scale invariant.
4. Investigate the `016` plane-versus-curved-model failures using candidate
   and ownership traces. The test should ask whether plane evidence was never
   generated or was generated but lost competition; avoid a scene-specific
   merge or score threshold change.
5. Require a genuinely independent labelled dataset (for example, a CAD
   surface benchmark such as SPFN or user-provided real scans) before claiming
   broad superiority to PCL-D4. No such second labelled dataset is currently
   available in this workspace.

The current evidence supports better **same-dataset cross-scene** mean quality
than PCL-D4, not universal generality.
