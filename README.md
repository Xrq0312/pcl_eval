# pcl_eval

Reproducible evaluation and staged development of CPU geometric-primitive
fitting for point clouds.  The target primitives are planes, spheres and
cylinders.  The repository keeps the original published evaluation artifacts
as a frozen baseline and develops new algorithms additively.

## Important source audit result

The upstream `main` commit (`85c50e7`) does **not** contain the C++ sources
named in `DEVELOPMENT_GUIDE.md` (`primitive_core.cpp`, `main.cpp`, tests or the
CMake project).  `worktrees/native_v28` is empty after a clean clone, and four
entries under `repos/` are gitlinks without a `.gitmodules` file.  Therefore,
the checked-in `samples/synth/synth_00/out_*` files and reports are the only
reproducible original-algorithm baseline in this clone.  See
[`docs/SOURCE_AUDIT.md`](docs/SOURCE_AUDIT.md).

New implementation work is isolated under `src/`, `include/` and `tests/`.
It must never overwrite the frozen `samples/**/out_*` artifacts.

## Local prerequisites

- Visual Studio 2022 with C++
- CMake 3.20+
- `C:\vcpkg\vcpkg\vcpkg.exe`
- vcpkg x64-windows packages: Eigen3, PCL, FLANN, nlohmann-json
- Python 3 with NumPy/SciPy for evaluation

The project always configures CMake with
`C:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake`.  Python entry points should
be run through `tools/run_python.ps1`; it also fixes the DLL search path for a
local Anaconda installation.

```powershell
cmake --preset windows-vcpkg
cmake --build --preset release
ctest --preset release --output-on-failure

./tools/run_python.ps1 tools/test_evaluate.py
./tools/run_python.ps1 tools/evaluate.py `
  --sample-dir samples/synth/synth_00 `
  --pred-prefix samples/synth/synth_00/out `
  --out results/frozen/synth_00.json

# Reproducible native ablation; pixel_map.txt is used automatically when present.
./tools/run_python.ps1 tools/run_native_ablation.py `
  --exe build-native/Release/native_primitive_fit.exe `
  --samples samples/bagsfit_v28 `
  --results results/native_ablation_10 `
  --stages 'global_local,joint_opt' --distance 0.02 --min-points 80 --limit 10
```

`native_primitive_fit` accepts unorganized TXT/PCD input.  For range-camera
data, `--pixel-map` optionally supplies image topology; no label or GT data is
read by the fitter.  Available ablation stages are `global_ransac`,
`global_local`, and `joint_opt`.

## Development policy

Each algorithmic idea is introduced as a separately selectable stage and is
accepted only after a fixed-seed ablation.  The primary metrics are target
instance detection at IoU 0.5, typed detection, target mIoU, cylinder axis and
radius error, and wall-clock time.  Dataset-specific limitations are recorded
in [`datasets/registry.json`](datasets/registry.json); missing point-level GT
must not be silently synthesized or reported as official GT.
The current measured results and their limitations are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

