# Changelog

## 0.2.0

- Add the reusable `SourceTraceRuntime` and the detector-independent
  `structured_analytic` source-integration backend.
- Add explicit analytic visibility, projected-aperture, and directional-disk
  geometry contracts with validated triangle ownership.
- Use an embedded 31 x 64 disk rule by default, with a reused lower-order
  error estimate.
- Stream file SHA-256 calculations and persist required semantic fingerprints
  in effective-response v3 and response-grid v2 files.
- Enable the structured backend in the synthetic dual-phase TPC example.
- Add a reusable liquid-only LXe asset identity and a fast `--test` grid that
  retains the geometry egress and forward/adjoint contracts.
- Add explicit `accurate` and `fast` regression modes. Accurate fits sample
  every refinement candidate through the effective response and fit a guarded
  quadratic peak only after the finest sampled grid; fast fits use only the
  persisted regular response grid and are the default.
- Support optional emitted-photon counts and absolute multinomial regression,
  including the no-hit category, while retaining conditional-pattern fits
  when emitted counts are unavailable.
- Remove loaders and tracing entry points that existed only for pre-release
  compatibility.
