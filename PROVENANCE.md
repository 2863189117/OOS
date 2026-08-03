# Provenance and release scope

OOS contains the reusable optical-transport implementation and two source
examples. The core is not linked to or configured by either example.

The dual-phase TPC example is synthetic and owns its geometry generators,
intrinsic LXe model, and optional LXe runtime plugin. Its dimensions, regular
hexagonal PMT layout, and sequential channel IDs are generated solely for the
public example. The PET-4x4 example owns its geometry generator and detector
parameters. Both communicate with the solver through installed command-line
interfaces, the public plugin ABI, YAML scenes, and documented HDF5 formats.

The following research material is intentionally outside this repository:

- generated geometry, operators, response matrices, hit data, and caches;
- paper sources, figures, tables, and benchmark summaries;
- Fisher-information, plotting, and other post-processing scripts;
- Geant4 comparison sources, generated datasets, and analysis scripts;
- historical development implementations and internal cluster scripts.

No generated research artifact is required at checkout time. Example inputs
are generated from source into a user-selected output directory.
