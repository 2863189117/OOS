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

The practical production collision profile uses
`--collision-sample-power 11` (2048 Sobol samples). This is the default because
it keeps a full production-shaped LXe build within an ordinary interactive run
while preserving the same phase-space axes, collision order, modal truncation,
and egress basis. Use `--collision-sample-power 12` for a higher-precision
comparison. Production acceptance relies on numerical checks that affect the
physics result: finite and nonnegative values, probability closure, compatible
geometry and axes, and forward/adjoint consistency. Rebuilding artifacts solely
to prove byte identity or to defend against deliberate file replacement is not
part of this example's runtime contract.

The production egress representation is `--egress-angular-layout
surface_marginal`. It keeps the converged liquid-surface position response and
applies the escaping P1 angular quadrature at runtime. The optional
`joint_surface_angle` layout retains the early-exit position/angle correlation,
but is experimental: increasing its angular grid from 4 x 8 to 8 x 16 did not
improve the external localization benchmark and increased the one-time adjoint
precompute cost. Select it explicitly for controlled studies rather than for
the default example artifacts.

An LXe block is keyed by the liquid radius, depth, and liquid-surface element
contract rather than by the complete GXe geometry hash. Changing PMT or gas
geometry therefore reuses the block when the liquid contract is unchanged.
Both generated and externally supplied blocks are rejected if their liquid
radius or depth disagrees with the selected detector geometry.

For detector layouts whose PMT pitch is finer than the default 40 x 80 LXe
surface coupling, use the optional ragged-ring v2 basis.  Each radial cell is
assigned enough azimuth points to bound the arc length at the cell's outer
radius; `--lxe-surface-min-phi` must also be at least `2*M+1` for the retained
Fourier order `M`.  The expensive liquid collision/diffusion solve is reused:
only the geometry-dependent egress quadrature is rebuilt.

```bash
python3 examples/dual-phase-tpc/generator/build_analytic_geometry.py \
  --output /path/to/geometry-ragged.h5 \
  --analytic-profile production \
  --lxe-surface-target-arc-mm 20 \
  --lxe-surface-min-phi 65 \
  --lxe-surface-phi-multiple 6 \
  --force

python3 examples/dual-phase-tpc/generator/resample_lxe_function_block.py \
  --input /path/to/lxe-factorized-v1.h5 \
  --geometry /path/to/geometry-ragged.h5 \
  --output /path/to/lxe-factorized-v2.h5 \
  --ragged-rings
```

Ragged resampling defaults to the unfiltered modal series.  Individual phase
basis kernels can therefore retain small Fourier undershoots, but production
responses are accepted only after every final, physical-source PMT probability
has been certified finite and nonnegative.  This source-restricted contract
avoids imposing an approximately one-percent smoothing bias on the physical
S2 manifold merely to make arbitrary basis vectors probabilistic.  The output
still records a dense-azimuth diagnostic.  Caratheodory--Fejer, Fejer,
generalized Cesaro, and ringwise adaptive contraction remain available for
explicit comparisons.  The plugin independently checks ring areas,
retained-mode quadrature moments, and the `m=0` return integral when it loads
v2.  Resampling an existing v2 block is rejected to prevent accidental double
filtering.  The current resampler also fails closed unless the analytic LXe
disk is origin-centered with canonical axes; transformed disks require an
explicit local-frame implementation rather than implicit world-coordinate use.

### Accelerated LXe block generation

The explicit-collision and Fourier--Bessel projection stages use a CUDA GPU
through CuPy when one is available. Install the CuPy wheel matching the host
CUDA toolkit; `--compute-backend auto` is the default:

```bash
python3 -m pip install cupy-cuda12x
python3 examples/dual-phase-tpc/generator/build_lxe_function_block.py \
  --geometry /path/to/geometry.h5 \
  --output /path/to/lxe-factorized-block.h5 \
  --compute-backend auto \
  --processes 1
```

Auto mode falls back to the NumPy/multiprocessing implementation when CuPy
cannot open a device. The CUDA backend uses one host process, keeps all public
arrays in NumPy, and writes the same HDF5 contract as the CPU backend. Use
`--compute-backend cpu --processes N` for an explicit CPU-only run.

The example can be copied outside the source tree after the binaries and
plugin are built. It locates no core headers or source files at runtime.
