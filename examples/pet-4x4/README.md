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

For a higher-resolution run, use the `production` profile, 112 response
cycles, and 2 mm grid spacing. Those settings are computationally expensive
and are not implied to be a detector-performance result.

The workflow generates geometry, validates the scene, builds operators,
calculates direct and bounded responses, and materializes a parallel-line
response grid. Generated HDF5 files are outputs and must not be committed.
