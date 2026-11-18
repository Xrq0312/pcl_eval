# Fit4CAD cross-dataset comparison

Date: 2026-10-08

## Protocol

This is a frozen comparison of native
`native-v2.7-scene-decomposition` against the local PCL-D4 reference at
commit `45df91a`. The 12 Fit4CAD official-test samples were selected and
converted before either fitter was run. No result-dependent parameter search
was performed.

Fit4CAD source coordinates have unspecified and strongly varying physical
scale. Both engines therefore received exactly the same deterministic scale
conversion per sample:

- `distance = 0.004 * bbox_diagonal`
- `min_radius = 0.5 * distance`
- `max_radius = 500 * distance`
- `merge_distance = 5 * distance`
- `merge_max_gap = 10 * distance`

The BAGSFit-selected count/angular configuration was otherwise frozen:
`min_points=80`, `normal_k=300`, `region_neighbors=30`, smooth angle 12
degrees, curvature 0.05, merge angle 5 degrees, minimum inlier ratio 0.3.
Native used 16 threads. Times are a single process-wall observation per
sample and are descriptive, not a controlled repeated throughput benchmark.

The exact commands, executable SHA-256 hashes and per-sample derived values
are stored in
`D:/download/pcl_eval/results/fit4cad_audit_v27_final/protocol.json`.

## Results

All 24 fitter runs completed without crashes, duplicate ownership, inconsistent
point counts, or non-finite model coefficients.

| 12-sample macro mean | Native v2.7 | PCL-D4 | Native - D4 |
|---|---:|---:|---:|
| target detection | 0.1768 | 0.1971 | -0.0203 |
| typed detection | 0.1744 | 0.1949 | -0.0205 |
| target mIoU | 0.1846 | 0.1686 | +0.0159 |
| target point-type accuracy | 0.4269 | 0.4037 | +0.0233 |
| unassigned ratio | 0.5585 | 0.5988 | -0.0402 |
| plane point F1 | 0.5050 | 0.4628 | +0.0422 |
| sphere point F1 | 0.0783 | 0.0783 | 0.0000 |
| cylinder point F1 | 0.2390 | 0.2535 | -0.0145 |
| process wall seconds | 0.2600 | 0.6237 | -0.3637 |
| predicted models | 5.33 | 4.00 | +1.33 |

When weighted by the number of GT target instances rather than giving each
sample equal weight (264 evaluable target instances), Native/D4 detection is
`0.1515/0.1553`, typed detection is `0.1439/0.1515`, and mIoU is
`0.1498/0.1280`.

Paired wins over 12 samples:

| Metric | Native wins | Ties | D4 wins |
|---|---:|---:|---:|
| detection | 2 | 7 | 3 |
| typed detection | 1 | 9 | 2 |
| mIoU | 7 | 3 | 2 |
| point-type accuracy | 8 | 3 | 1 |

A deterministic 20,000-resample paired bootstrap gives descriptive 95%
intervals for Native minus D4 of `[-0.0571, 0.00895]` for Det,
`[-0.0556, 0.00758]` for Typed Det, `[0.00274, 0.0304]` for mIoU, and
`[0.00830, 0.0406]` for point accuracy. With only 12 deliberately spaced
samples, these intervals are sensitivity summaries, not population claims.

| Sample | N Det | D4 Det | N Typed | D4 Typed | N mIoU | D4 mIoU | N Acc | D4 Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PC1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| PC10 | 0.333 | 0.500 | 0.333 | 0.500 | 0.492 | 0.514 | 0.777 | 0.774 |
| PC13 | 0.105 | 0.105 | 0.105 | 0.105 | 0.080 | 0.073 | 0.182 | 0.168 |
| PC16 | 0.171 | 0.171 | 0.171 | 0.171 | 0.170 | 0.150 | 0.509 | 0.483 |
| PC19 | 0.188 | 0.312 | 0.188 | 0.312 | 0.282 | 0.279 | 0.496 | 0.478 |
| PC22 | 0.200 | 0.200 | 0.200 | 0.200 | 0.222 | 0.178 | 0.651 | 0.634 |
| PC25 | 0.132 | 0.158 | 0.132 | 0.132 | 0.123 | 0.126 | 0.467 | 0.474 |
| PC28 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| PC31 | 0.409 | 0.364 | 0.409 | 0.364 | 0.343 | 0.274 | 0.707 | 0.635 |
| PC35 | 0.125 | 0.125 | 0.125 | 0.125 | 0.120 | 0.120 | 0.556 | 0.556 |
| PC4 | 0.400 | 0.400 | 0.400 | 0.400 | 0.317 | 0.282 | 0.555 | 0.502 |
| PC7 | 0.059 | 0.029 | 0.029 | 0.029 | 0.066 | 0.026 | 0.224 | 0.140 |

Parameter errors are conditional on different detected/matched sets and must
not be treated as paired accuracy. Native has 23 matched planes, one sphere,
and five cylinders; D4 has 24, one, and seven respectively. Both produce very
accurate parameters for the clean instances they do match. The full conditional
means remain in `summary.json`.

## Failure diagnosis

The cross-dataset result is poor for both engines. It should not be summarized
as a clean win for either implementation:

- Native has slightly lower Det/Typed Det but higher mIoU, point accuracy,
  plane F1, assignment coverage, and lower wall time.
- Both engines detect no target instance in PC1 and PC28.
- Of 264 target instances with at least one uniquely assigned GT point, 89
  (33.7%) contain fewer than the frozen `min_points=80`; ten contain fewer
  than 20 points. Median target support is only 247.5 points.
- Native creates 18,619 regions across the subset and rejects 18,556 as too
  small. D4 creates 18,826 and rejects 18,763 as too small. This means the
  dominant failure occurs before RANSAC and model merging.
- `normal_k=300` and a fixed absolute support count were selected for dense
  BAGSFit depth scenes. They are not appropriate general defaults for CAD
  objects with 5,000-46,364 points and many small patches.

The scale-normalized distance gates alone are therefore insufficient for
generality. The next principled experiment should replace fixed-count normal
neighborhoods and minimum support with local-spacing/uncertainty and effective
surface-area criteria. Any such mechanism developed using these 12 samples
must treat them as validation data and use the remaining 23 official Fit4CAD
test objects, or another dataset, for untouched confirmation.

## Artifacts

- Summary: `D:/download/pcl_eval/results/fit4cad_audit_v27_final/summary.json`
- Per-run predictions/evaluations:
  `D:/download/pcl_eval/results/fit4cad_audit_v27_final/{native_v27,pcl_d4}`
- Reproducible runner: `D:/download/pcl_eval/tools/fit4cad_audit.py`
