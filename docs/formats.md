# Scene and data formats

`scene.yaml` is schema version 1. Every surface must set `two_sided: true`
and provide every model-specific probability. There are no implicit optical
defaults. Triangle orientation defines the minus-to-plus domain normal.
The built-in `specular_reflector` model is an opaque mirror with an explicit
`reflectivity`; its remaining weight is absorbed. The built-in `lambertian`
model uses the scene-level `lambertian_mu2_order` and
`lambertian_phi_count` deterministic quadrature controls.
Sensitive surfaces accept `absorb`, `reflect_specular`,
`reflect_lambertian`, or `transmit` as the action for the undetected
remainder.
Every custom surface also declares a right-handed orthonormal `local_frame`
with `origin_mm`, `x_axis`, `y_axis`, and `z_axis`. Plugin hit callbacks
receive positions, directions, and polarization reference axes in this frame,
plus the ordinal and barycentric coordinates of the hit element; they never
receive a global primitive ID or domain ID.

`geometry.h5` contains:

```text
/geometry/vertices          float64 [Nv,3]
/geometry/triangles         uint32  [Nt,3]
/geometry/surface_id        uint32  [Nt]
/geometry/surface_basis_id  uint32  [Nt]
/geometry/minus_domain_id   int32   [Nt]
/geometry/plus_domain_id    int32   [Nt]
/geometry/channel_id        int32   [Nt]
```

`surface_basis_id` is local to a surface. Geometry triangles with the same
`(surface_id, surface_basis_id, primary-domain side)` share one diffuse
radiance state. A missing dataset is the legacy exact-leaf convention: one
state per reflective triangle. Geometry and basis are deliberately separate;
refining the Embree mesh therefore does not force a larger transport matrix.
When a state contains several triangles, incident power is accumulated into
that state and outgoing Lambertian quadrature is distributed over its member
triangles in proportion to physical area.

An optional nested hierarchy is stored alongside the fixed geometry:

```text
/surface_basis/patch_size_mm
/surface_basis/state_count
/surface_basis/levels/<l>/triangle_basis_id
/surface_basis/levels/<l>/area_mm2
/surface_basis/transfers/<l>_to_<l+1>/restriction   CSR
/surface_basis/transfers/<l>_to_<l+1>/prolongation CSR
```

State vectors represent power. Restriction sums child power into its parent;
prolongation distributes parent power to children by area. Both are
nonnegative, prolongation columns sum to one, and `restriction *
prolongation` is the identity on the parent level. The active fixed basis is
copied to `/geometry/surface_basis_id`, so the native runtime does not need
Python or hierarchy traversal.

Tile-frozen adaptive runs may instead use a compact overlay:

```text
/surface_basis_selection/triangle_basis_id uint32 [Nt]
```

Pass it as `oos build ... --surface-basis selection.h5` and repeat the same
option for `oos solve ... --scene scene.yaml`. The overlay hash participates
in the operator cache key. It is normally a compressed few-megabyte file, not
a duplicate of the full geometry.

`sources.yaml` supports `point`, `line_segment`, `disk`, `cylinder`, `box`,
`tetrahedron`, and `rectangular_line_neighborhood` spatial quadratures.
`rectangular_line_neighborhood` produces a deterministic equal-weight
low-discrepancy sample of the active-medium portion of the two-dimensional
cross-section whose
Euclidean distance from a rectangular line is no more than
`maximum_distance_mm`. It takes `line_center_mm`, `obstacle_half_width_mm`,
`obstacle_half_thickness_mm`, `medium_z_max_mm`, and `count`; the longitudinal
coordinate is fixed by `line_center_mm`.

Angular quadratures are `isotropic`,
`isotropic_product`, `isotropic_surface_shape_factor`, `cosine`, `cone`, or
an explicit weighted direction table. `isotropic_product` uses
Gauss-Legendre nodes in direction cosine and uniform midpoint nodes in
azimuth, configured by `mu_order` and `phi_count`.

`isotropic_surface_shape_factor` does not emit a fixed direction grid. The
core integrates the exact solid angle of each boundary triangle, follows its
centroid ray through the ordinary surface behavior, and recursively
subdivides triangles until the parent/children response L1 difference meets
`relative_tolerance`. It accepts `maximum_subdivision_depth`; zero performs
the mandatory parent-to-four-children assessment without further recursion.
The depth is a hard cap: a non-converged branch accepts its child sum and
reports the residual parent/child L1 difference in
`/response/source_integration_l1_error_estimate`. The accepted source row is
normalized to its declared weight; the L1 size of that normalization
correction is included in the same estimate.

Before evaluating four children, the core applies a contribution selector.
Smooth elements are refined only when their weighted solid-angle fraction is
at least `minimum_refinement_solid_angle_fraction`. Elements touching a
surface/domain junction or a crease sharper than
`feature_dihedral_degrees` use
`minimum_feature_solid_angle_fraction`; only children that still touch the
original feature edge inherit that lower threshold. Thus distant smooth
elements retain exact solid-angle weights without paying for unnecessary
child rays. The global L1
error budget is divided across top-level boundary triangles and redistributed
within each child tree according to accepted child weight. This mode currently
requires an unpolarized isotropic source in a declared closed domain.

Spatial and angular or shape-factor weights are normalized independently.

The operator cache stores three CSR matrices:

```text
T = /operators/transition
D = /operators/detection
L = /operators/losses
```

Each CSR group has `shape`, `indptr`, `indices`, and `data`. `/metadata`
records the scene, geometry, dependency lock, plugin binary, code commit,
energy, aggregate cache hashes, and `/metadata/state_labels_json`. State
labels make the ordinary/plugin state offsets inspectable without affecting
the numerical runtime.

`oos build` treats this file as a content-addressed cache.  The cache key
includes the bytes of the scene, geometry, dependency lock, solver commit,
every custom plugin binary, and every referenced precomputed custom-surface
block.  A compatible file is reused; a missing, malformed, or stale file is
rebuilt through a temporary file and atomically replaced.  `--force` bypasses
the compatibility check.

A `custom_nonlocal` plugin always emits its surface-relative egress basis:

```text
/nonlocal/egress/surface_element    uint64 [Ne]
/nonlocal/egress/barycentric        float64 [Ne,3]
/nonlocal/egress/side               uint64 [Ne]
/nonlocal/egress/direction_local    float64 [Ne,3]
/nonlocal/egress/stokes             optional float64 [Ne,4]
/nonlocal/egress/reference_axis_local optional float64 [Ne,3]
```

An explicit implementation additionally emits:

```text
/nonlocal/internal_transition       CSR [Ns,Ns]
/nonlocal/emission                  CSR [Ns,Ne]
/nonlocal/internal_losses           CSR [Ns,Nl]
```

Its metadata contains
`{"locality":"nonlocal","loss_names":[...]}`. A function implementation
instead emits metadata
`{"locality":"nonlocal","execution":"function","state_count":Ns,
"loss_names":[...]}` and exports `oos_get_function_operator_v2`. The
function consumes a row-major batch `[B,Ns]` and returns retained state
`[B,Ns]`, egress coefficients `[B,Ne]`, and intrinsic losses `[B,Nl]`. It
also consumes adjoints of those three outputs and returns the Euclidean
adjoint in `[B,Ns]`. The two actions must satisfy their inner-product identity.
They represent the deterministic linear physical map and must not perform
input-dependent clipping or renormalization. A truncated spectral basis may
contain signed intermediate coefficients. The implementation may internally
use factorized arrays, modal coefficients, FFTs, or a matrix, but those
details are outside the core schema.

The scene config must name `nonlocal_domain_id`; every triangle in the
surface group must separate that domain from another declared medium.
`surface_element` is the ordinal within that surface group, never a global
primitive ID. `side=0/1` selects the triangle minus/plus side. The local
direction uses a deterministic tangent/bitangent/side-normal frame and must
have positive local z. Optional Stokes vectors have unit intensity; the
function output coefficient supplies their scalar weight. The optional
reference axis uses the same tangent frame and is projected perpendicular to
the ray. If omitted, egress is unpolarized and uses the triangle tangent as
its reference. The core reconstructs each ray, traces it with Embree, then
composes

```text
T_surface = internal_transition + emission * K_state
D_surface = emission * K_detection
L_surface = internal_losses + emission * K_loss
```

where `K` is geometry-dependent and belongs to the operator cache, not to the
surface behavior file. External primitive IDs and sensitive channel IDs are
forbidden in a nonlocal payload. Every row and every incident deposition
stencil must be nonnegative and probability-conserving.

For a functional block, the operator cache stores the plugin identity and
configuration together with geometry-only egress coupling:

```text
/operators/function_blocks/<i>/descriptor
/operators/function_blocks/<i>/egress_to_transition
/operators/function_blocks/<i>/egress_to_detection
/operators/function_blocks/<i>/egress_to_losses
/operators/function_blocks/<i>/intrinsic_loss_columns
```

The intrinsic function is evaluated at solve time; its factorized assets are
part of the cache key. No dense or explicit `Ns x Ne` matrix is required.

`response.h5` contains absolute sensitive-channel efficiency, named losses,
unresolved state weight, input weight, and the signed energy-balance error
for every source. Runtime metadata records the backend, hardware, iteration
count, wall time, and resident-device bytes.

The built-in loss channels distinguish physical and numerical termination:

```text
escape                       declared transmission to the exterior
float32_intersection_miss    no Embree hit inside a validated enclosed domain
```

Embree traces float32 rays even though the canonical mesh and optical weights
are float64. Grazing seams can therefore produce a small nonzero
`float32_intersection_miss`. It is audited rather than forced to zero. Since
the solver is passive, the cumulative missed weight is a conservative
absolute upper bound on the sensitive efficiency that those rays could have
contributed. The per-source value is written as:

```text
/response/float32_efficiency_loss_upper_bound float64 [Ns]
/response/source_integration_l1_error_estimate float64 [Ns]
```

Thus, for every source,

```text
0 <= true_efficiency - reported_efficiency
   <= float32_efficiency_loss_upper_bound
```

for the part of the numerical error caused specifically by float32
intersection misses. Other quadrature, discretization, and model errors are
not included in this bound. The source-integration dataset is the accumulated
parent/child L1 difference of the accepted shape-factor quadrature. It is an
a-posteriori convergence diagnostic rather than a rigorous upper bound.

## Adjoint bounded effective response

`oos-efficiency precompute` writes:

```text
/effective/state_to_detection  float64 [Nstate,Nchannel]
/effective/state_to_losses     float64 [Nstate,Nloss]
/effective/state_unresolved    float64 [Nstate]
/effective/channel_id          int32   [Nchannel]
/metadata/cycles               uint32  [1]
/metadata/build_batch_size     uint64  [1]
/metadata/operator_tolerance   float64 [1]
/metadata/construction_method  string: adjoint_linear
/metadata/operator_cache_key_sha256
```

The schema is `oos.effective-adjoint-response.v2`. The default is seven
complete bounded Neumann cycles. Terminal channel and loss bases are
propagated through the complete adjoint operator, including matrix-free
function blocks. The file is invalidated when the originating `operators.h5`
cache key changes. The former forward-basis v1 response schema is deliberately
unsupported.

## Regression input and response grids

The canonical hit input is:

```text
/hits/counts       uint64 [Nevent,Nchannel]
/hits/channel_id   int32  [Nchannel]
/hits/truth_xy_mm  optional float64 [Nevent,2]
```

`oos-regress grid` writes:

```text
/grid/xy_mm                        float64 [Npoint,2]
/grid/conditional_log_probability float32 [Npoint,Nchannel]
/grid/top_efficiency               float64 [Npoint]
/grid/channel_id                   int32   [Nchannel]
/metadata/domain_shape             string: disk, rectangle, or parallel_lines
/metadata/radius_mm
/metadata/half_x_mm                zero for disk grids
/metadata/half_y_mm                zero for disk grids
/metadata/line_y_start_mm          zero unless parallel_lines
/metadata/line_pitch_mm            zero unless parallel_lines
/metadata/line_count               zero unless parallel_lines
/metadata/spacing_mm
/metadata/source_angular_mode
/metadata/source_z_mm
/metadata/source_thickness_mm      maximum line distance for line sources
/metadata/source_transverse_count  transverse samples per line/x grid point
/metadata/obstacle_half_width_mm
/metadata/obstacle_half_thickness_mm
/metadata/source_medium_z_max_mm
/metadata/source_mu_order
/metadata/source_phi_count
```

Version-1 grid readers treat files without `domain_shape` as legacy disk
grids. Rectangle grids persist their independent half-widths so continuous
refinement clips candidates to the same source plane used by the coarse grid.
`parallel_lines` grids restrict the transverse coordinate to
`line_y_start_mm + line_id * line_pitch_mm` and refine only the discrete line
identifier plus the longitudinal coordinate. The source metadata records the
near-line volume and angular quadrature used to marginalize the unobserved
emission offset when each response-matrix row was built.

The default Cartesian spacing is 10 mm and may be changed with
`--spacing-mm`. `oos-regress fit --adjoint` additionally writes
`/regression/full_plane_log_likelihood [Nevent,Npoint]`; without `--adjoint`
only fitted coordinates and best likelihoods are retained.
Parallel-line fits additionally write
`/regression/fitted_line_id [Nevent]` and
`/regression/fitted_line_x_mm [Nevent]`.
