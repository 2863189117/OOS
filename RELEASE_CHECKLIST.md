# Public release checklist

- [x] BSD-3-Clause license with copyright held by Fan Junhan.
- [x] Core source builds without either example.
- [x] The dual-phase TPC uses a synthetic 2 m geometry, 2-inch PMTs, and a
  generated hexagonal layout with no imported channel map.
- [x] The LXe nonlocal plugin is an explicit optional component.
- [x] PET-specific public interfaces are generic parallel-line and
  rectangular-obstacle primitives; permission to publish is confirmed.
- [x] Generated data, results, papers, post-processing, Geant4 comparisons,
  historical code, and internal execution scripts are excluded.
- [x] Repository scope is checked by `scripts/check_release_scope.py`.
- [ ] Configure the GitHub remote and required branch protections.
