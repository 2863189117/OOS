# OOS — Operator-based Optical Solver

OOS builds deterministic optical-transport operators for closed, labeled,
multi-medium geometries. It reports absolute efficiency for sensitive
channels and supports likelihood-based reconstruction from channel counts.

The production runtime is C++17/OpenMP with an optional CUDA backend. Geometry
uses Embree, persistent arrays use HDF5, and scenes use YAML. Structured
analytic geometry can replace mesh-count-dependent source integration while
retaining the same surface physics. The installed command-line tools are
`oos`, `oos-efficiency`, and `oos-regress`.

## Repository boundary

`oos_core` is detector-independent. A synthetic dual-phase TPC and PET-4x4
live under `examples/` and consume only public file formats, command-line
tools, and the plugin ABI. Examples and the optional LXe plugin are not built
or installed by default.

Generated data, simulation results, post-processing, paper material, and
historical research code are outside this repository. See `PROVENANCE.md` and
run `python3 scripts/check_release_scope.py` before publishing.

## Build

The required packages are OpenMP, Embree 4.3, HDF5 1.14, yaml-cpp 0.8,
nlohmann-json 3.11, and Catch2 3.7 when tests are enabled.

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOOS_ENABLE_CUDA=OFF \
  -DOOS_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/install"
```

Pinned dependencies can instead be built under `.deps/`:

```bash
./scripts/bootstrap_deps.sh
# Source the environment.sh path printed by the script, then configure OOS.
```

Enable the dual-phase TPC LXe plugin explicitly when needed:

```bash
cmake -S . -B build-tpc -G Ninja \
  -DOOS_BUILD_LXE_PLUGIN=ON \
  -DOOS_ENABLE_CUDA=OFF
cmake --build build-tpc --target oos_lxe_plugin
```

## Stable command-line interfaces

```bash
oos-efficiency precompute operators.h5 --output effective.h5

oos-efficiency calculate operators.h5 --scene scene.yaml \
  --sources sources.yaml --output response.h5 --device cpu

oos-regress grid operators.h5 --precomputed effective.h5 \
  --scene scene.yaml --output grid.h5

oos-regress fit --hits hits.h5 --grid grid.h5 --output regression.h5
```

`precompute` constructs the bounded response exclusively through terminal-
basis adjoint propagation. Function plugins must export
`oos_get_function_operator_v2` with paired linear forward and adjoint actions.
V1 function plugins and `oos.effective-bounded-response.v1` files are not
accepted.

Version 0.2 writes `oos.effective-adjoint-response.v3` and
`oos.response-grid.v2`. These files carry required semantic fingerprints;
older response and grid schemas are intentionally rejected.

See `docs/formats.md`, `docs/architecture.md`,
`examples/dual-phase-tpc/README.md`, and `examples/pet-4x4/README.md`.

## Release status

The source/example boundary is enforced in the repository. See
`RELEASE_CHECKLIST.md` for the checks required before publishing.
