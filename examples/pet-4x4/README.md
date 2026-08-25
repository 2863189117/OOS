# PET-4x4 example

This detector-specific example is independent of `oos_core`. It generates a
PET-4x4 geometry from source, invokes installed OOS command-line tools, and
writes every generated artifact beneath the requested output directory.

The example contains no response matrices, hit datasets, Geant4 sources,
plots, Fisher-information analysis, or post-processing scripts.

## Model

The source lies in the liquid-xenon neighborhood around one of 68 lower anode
lines. The public solver API represents this as a generic `parallel_lines`
search domain and a `rectangular_line_neighborhood` spatial distribution; all
PET dimensions and material choices remain local to this example.

The scene contains the lower anode grid, a 4x4 SiPM channel array, quartz,
silicon, photocathodes, the bottom cathode, and PTFE. The upper wire layer and
liquid-xenon Rayleigh scattering are intentionally omitted from this example.

Geometry profiles select the diffuse-surface basis:

- `smoke`: 20 mm maximum PTFE patch size;
- `coarse`: 10 mm;
- `production`: 5 mm;
- `fine`: 2.5 mm convergence reference.

## Run independently

Build and install OOS first. Then copy this directory anywhere outside the OOS
source tree and run it using only the installed binary directory:

```bash
python3 -m pip install -r requirements.txt
python3 run_example.py \
  --binary-dir /path/to/oos/install/bin \
  --output /tmp/oos-pet4x4 \
  --geometry-profile smoke \
  --cycles 8 \
  --grid-spacing-mm 8
```

For the reproducible production baseline, use:

| setting | value |
| --- | ---: |
| diffuse-basis profile / maximum patch | `production` / 5 mm |
| bounded-response cycles | 112 |
| adjoint-precompute batch size | 64 |
| response-grid spacing along each wire | 2 mm |
| transverse source samples | 32 |
| isotropic angular rule | 32 x 128 (`mu` x `phi`) |
| source-intersection backend | `generic_bvh` |
| response-grid batch size | 16 |
| compute backend | CPU |
| Lambertian quadrature (`mu^2` x `phi`) | 8 x 32 |
| geometry / energy / Neumann tolerances | `1e-8 mm` / `1e-10` / `1e-9` |
| maximum specular hits / diffuse bounces | 64 / 512 |
| ray-origin offset | `0` (geometry-derived float-safe value) |

The runner passes `--source-backend generic_bvh` explicitly. OOS 0.2 defaults
to `auto`, which may consume structured analytic source declarations when a
geometry provides them. `generic_bvh` is the explicit fallback and matches the
pre-0.2 intersection path used to establish this PET example; keeping it in
the command prevents a future geometry declaration from silently changing the
source-intersection algorithm. The fixed-ray isotropic product still traces
all subsequent optical branches through the ordinary Embree geometry.

Run that baseline with:

```bash
python3 run_example.py \
  --binary-dir /path/to/oos/install/bin \
  --output /tmp/oos-pet4x4-production \
  --geometry-profile production \
  --cycles 112 \
  --grid-spacing-mm 2 \
  --source-transverse-count 32 \
  --source-mu-order 32 \
  --source-phi-count 128
```

These settings are computationally expensive and are not implied to be a
detector-performance result. The effective-response HDF5 records the 112
cycles and batch size; the response-grid HDF5 records `generic_bvh`, spatial
spacing, and angular/source quadratures, so downstream results can audit the
configuration without relying only on this README.

The library default of seven cycles is not sufficient for this highly
reflective detector. In the original 5 mm PET production convergence audit,
the operator had 1,736 diffuse states and 945,048 nonzero transition entries;
112 cycles reduced the maximum unresolved source weight to `1.27e-10` and the
maximum normalized-channel L1 difference from the direct solve to `2.13e-11`.
OOS 0.2 changes precompute construction to an adjoint implementation but not
the bounded-cycle definition, so the detector-specific 112-cycle requirement
is retained.

The workflow generates geometry, validates the scene, builds operators,
calculates direct and adjoint-precomputed bounded responses, and materializes
a parallel-line response grid. Python orchestrates the installed C++ tools;
it contains no independent response-precompute implementation. Generated
HDF5 files are outputs and must not be committed.
