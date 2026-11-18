# SHREC2022 cross-dataset comparison

## Protocol

This audit compares native v2.7 and the local PCL-D4 reference on a frozen
subset of the official SHREC2022 test set at source commit
`da2b5f4f5c69896c69ac2777db2d2d499e32f03d`.

The subset contains one median-ID sample from every Cartesian stratum of the
three supported primitive types (plane, cylinder and sphere) and all ten
official artifact classes A0--A9. It therefore contains 30 single-primitive
segments and 89,922 points. Selection was fixed from GT type and artifact
metadata before either fitter was run. All 30 downloaded point-cloud blobs
match their Git objects.

Both engines use the same scale rule and count thresholds:

- `distance = 0.004 * bbox diagonal`
- `min_radius = 0.5 * distance`
- `max_radius = 500 * distance`
- `merge_distance = 5 * distance`
- `merge_max_gap = 10 * distance`
- `min_points = 80`, `normal_k = 300`, `region_neighbors = 30`

No SHREC result was used to change or tune native v2.7.

## Aggregate result

| Metric | Native v2.7 | PCL-D4 | Native - D4 |
|---|---:|---:|---:|
| Detection | 0.7667 | 0.7667 | 0.0000 |
| Typed detection | 0.7667 | 0.7667 | 0.0000 |
| mIoU | 0.6877 | 0.6867 | +0.0010 |
| Point-type accuracy | 0.7480 | 0.6867 | +0.0612 |
| Unassigned rate | 0.2502 | 0.3133 | -0.0631 |
| Mean wall time | 0.1121 s | 0.1260 s | -0.0138 s |
| Mean model count | 1.4333 | 0.8333 | +0.6000 |

Detection and typed detection are identical on all 30 samples: 23 successes
and seven misses for each engine. Native wins ten, ties eighteen and loses two
of the paired mIoU comparisons. It is about 11% faster in this small-segment
workload.

The point-accuracy gain must not be read as an equally large instance-fitting
gain. SHREC has one GT primitive per cloud. Native often assigns additional
same-type fragments, so those points count as type-correct while mIoU matches
only one predicted instance. The small mIoU gain is the more conservative
measure of single-model coverage.

## Robustness by artifact

| Artifact | Description | Native Det | D4 Det | Native mIoU | D4 mIoU |
|---|---|---:|---:|---:|---:|
| A0 | clean | 1.000 | 1.000 | 1.000 | 1.000 |
| A1 | uniform noise | 0.667 | 0.667 | 0.534 | 0.531 |
| A2 | Gaussian noise | 0.667 | 0.667 | 0.491 | 0.491 |
| A3 | undersampling | 1.000 | 1.000 | 1.000 | 1.000 |
| A4 | missing parts | 1.000 | 1.000 | 1.000 | 1.000 |
| A5 | uniform noise + undersampling | 0.333 | 0.333 | 0.405 | 0.404 |
| A6 | Gaussian noise + undersampling | 0.667 | 0.667 | 0.496 | 0.493 |
| A7 | uniform noise + missing parts | 0.667 | 0.667 | 0.431 | 0.429 |
| A8 | Gaussian noise + missing parts | 0.667 | 0.667 | 0.544 | 0.543 |
| A9 | local deformation | 1.000 | 1.000 | 0.977 | 0.976 |

Both methods are robust to clean undersampling, missing regions and the chosen
local-deformation cases. Noise combined with undersampling is the clearest
shared failure mode.

## Result by primitive type

| Type | Native Det | D4 Det | Native mIoU | D4 mIoU |
|---|---:|---:|---:|---:|
| Plane | 0.800 | 0.800 | 0.731 | 0.729 |
| Sphere | 0.900 | 0.900 | 0.734 | 0.734 |
| Cylinder | 0.600 | 0.600 | 0.598 | 0.597 |

Cylinders remain the main generalization weakness. Four of the seven misses
are cylinders, especially when noise is combined with missing data or
undersampling.

## Parameter accuracy on matched instances

PCL-D4 is more accurate after a primitive has been detected:

| Error | Native v2.7 | PCL-D4 |
|---|---:|---:|
| Plane normal angle | 0.0318 deg | 0.0283 deg |
| Plane offset | 0.00958 | 0.00649 |
| Sphere center | 0.00172 | 0.00061 |
| Sphere radius absolute | 0.00115 | 0.00035 |
| Cylinder axis angle | 0.1361 deg | 0.0748 deg |
| Cylinder axis-line distance | 0.00492 | 0.00444 |
| Cylinder radius absolute | 0.00581 | 0.00418 |

This indicates that the next principled improvement should target robust
single-primitive estimation under noisy sparse support, especially cylinder
axis estimation. Increasing absorption or producing more fragments would
improve point-type coverage without solving that parameter-estimation gap.

## Reproduction

The raw subset is under `D:/download/pcl_eval/data/SHREC2022_test30`, unified
samples under `D:/download/pcl_eval/samples/shrec2022_test30`, and complete
outputs under `D:/download/pcl_eval/results/shrec2022_audit_v27`.

```powershell
python tools/adapt_shrec2022.py `
  --source-root D:/download/pcl_eval/data/SHREC2022_test30 `
  --output-root D:/download/pcl_eval/samples/shrec2022_test30 `
  --source-commit da2b5f4f5c69896c69ac2777db2d2d499e32f03d

python tools/shrec2022_audit.py `
  --samples-root D:/download/pcl_eval/samples/shrec2022_test30 `
  --output-root D:/download/pcl_eval/results/shrec2022_audit_v27 `
  --native-exe build-native/Release/native_primitive_fit.exe `
  --pcl-d4-exe D:/download/pcl_region_primitive/pcl_region_primitive/build/primitive_fit.exe
```
