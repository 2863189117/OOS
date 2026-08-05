# Synthetic dual-phase TPC example

This example models a generic dual-phase xenon time-projection chamber while
remaining independent of `oos_core`. Its runtime plugin and all input
generators live inside this directory and are disabled in the default build.

The geometry is intentionally fictional. It uses a 2 m active diameter, a
2 m liquid depth, a 60 mm gas region, and circular 2-inch (50.8 mm) PMT
apertures. PMT centers are generated row by row on a regular hexagonal
lattice with 64 mm pitch and sequential public-example channel IDs. There is
no imported channel map, survey coordinate, vendor solid model, detector
database, or reference-geometry input.

The scene retains the features needed to demonstrate a general dual-phase
TPC calculation:

- gaseous and liquid xenon domains separated by a nonlocal interface;
- deterministic Rayleigh first-flight and diffusion return in the liquid;
- a factorized Fourier--Bessel response evaluated by an optional plugin;
- quartz windows, sensitive photocathodes, and reflective boundaries;
- analytic and triangle-derived surface-basis options;
- an explicitly declared `structured_analytic` source backend for the gas
  track, including exact directional-disk integration at the liquid surface.

`generator/build_geometry.py` constructs the validation mesh and hexagonal PMT
layout. `generator/build_analytic_geometry.py` adds an adaptive analytic
transport basis. `generator/build_lxe_function_block.py` produces the
geometry-independent liquid response used by `liboos_lxe_plugin.so`.
Generated HDF5 files are runtime outputs and must not be committed.
The system response is materialized by the C++ adjoint precompute; the Python
generators only build geometry and the intrinsic LXe factorized block.

Build the optional plugin from the OOS repository root:

```bash
cmake -S . -B build-tpc -G Ninja \
  -DOOS_BUILD_LXE_PLUGIN=ON \
  -DOOS_ENABLE_CUDA=OFF
cmake --build build-tpc --target oos oos-efficiency oos_lxe_plugin
```

Install the example-owned Python dependencies and run a smoke profile:

```bash
python3 -m pip install -r examples/dual-phase-tpc/requirements.txt
python3 examples/dual-phase-tpc/run_example.py \
  --oos /path/to/build-tpc/oos \
  --oos-efficiency /path/to/build-tpc/oos-efficiency \
  --lxe-plugin /path/to/build-tpc/examples/dual-phase-tpc/plugin/liboos_lxe_plugin.so \
  --cache-dir /tmp/oos-tpc-cache \
  --output /tmp/oos-tpc-run \
  --geometry-profile smoke
```

The runner always uses analytic geometry because the shipped source contract
explicitly requires `structured_analytic`. Triangle-only geometry can still
be generated separately for generic-BVH research comparisons.
The `smoke` profile passes `--test` to the LXe generator. This reduces the
liquid phase grid, collision sampling, and modal expansion while preserving
the geometry-owned 40 x 80 liquid-surface egress basis. It therefore exercises
generation, forward/adjoint plugin coupling, CUDA precompute, source injection,
and HDF5 output without paying production precompute cost. The other profiles
retain the production LXe defaults.

An LXe block is keyed by the liquid radius, depth, and liquid-surface element
contract rather than by the complete GXe geometry hash. Changing PMT or gas
geometry therefore reuses the block when the liquid contract is unchanged.
Both generated and externally supplied blocks are rejected if their liquid
radius or depth disagrees with the selected detector geometry.

The example can be copied outside the source tree after the binaries and
plugin are built. It locates no core headers or source files at runtime.
