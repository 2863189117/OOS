# Architecture

The solver separates intrinsic surface behavior from geometric transport.
A behavior receives an incident state in the local surface frame and returns
only local branches: reflection side, transmission side, diffuse reflection,
detection, or absorption. It cannot name a destination primitive, diffuse
state, or sensitive channel. The geometry assembler uses Embree to map these
branches to the surrounding scene and is the only component that owns global
state and channel indices.

This rule applies equally to built-in Fresnel, opaque specular, Lambertian,
sensitive, local-plugin, and nonlocal-plugin surfaces. Translating or rotating
a surface changes the geometric coupling kernel, not its material behavior.

Ray-intersection primitives are not transport states. The fixed validated fine
triangulation supplies topology and the generic fallback. Exact disks,
annuli, finite cylinders, perforated disks, and boxes own production
intersections when declared. Diffuse surface brightness is represented by an
independent piecewise-constant basis: several fine triangles may share one
state, and one fine triangle may be selected at the exact leaf level. This
permits basis convergence studies without changing geometry, visibility,
surface behavior, or sensitive-channel ownership.

Adaptive surface bases are frozen by spatial source tile. Refinement proceeds
only by replacing a hierarchy parent with all of its children, so active nodes
remain a non-overlapping partition of every reflective surface. Every
candidate position evaluated inside one tile uses the same cached operator.
Changing the basis at each likelihood candidate is forbidden because it would
introduce nonphysical discontinuities into the objective.

For source batch `B`, the absolute response is

```text
E = B (I + T + T² + ...) D
```

The CPU and CUDA backends evaluate the same bounded Neumann series.
The CUDA implementation uses native cuSPARSE and cuBLAS in float64; it never
calls Python or CuPy.

The production efficiency path materializes the bounded seven-cycle response

```text
R_D = sum(n=0..6) T^n D
R_L = sum(n=0..6) T^n L
u_7 = T^7 1
```

including matrix-free nonlocal blocks. Precomputation propagates batches of
sensitive-channel and loss terminal bases backward through the Euclidean
adjoint of the complete transport operator; it never propagates one identity
source per transport state. Operators and function assets remain resident on
the GPU. The resulting `effective.h5` is bound to the operator cache key.
Source geometry remains continuous: a newly traced source batch `B` is
evaluated as `direct + B R_D`, with corresponding losses and unresolved
weight.

Regression normalizes each response across sensitive channels because emitted
light yield is a nuisance parameter. A response grid stores
`log p(channel | x,y)`. Full-plane likelihood is the adjoint dense product

```text
logL[event,xy] = counts[event,channel] * logp[xy,channel]^T
```

implemented with native cuBLAS. Without a grid, or after a grid supplies the
global initial maximum, continuous candidates are rebuilt by the analytic
shape-factor source integrator and evaluated through `effective.h5`.
`SourceTraceRuntime` owns the reusable Embree scene, validated operator view,
surface/plugin runtimes, domain-boundary candidates, and feature-edge cache.
Clients construct it once and reuse it across source batches.
The optional `structured_analytic` source backend is selected by geometry
contracts rather than detector names. Declared direct/aperture elements avoid
first-hit BVH queries, while a declared directional disk replaces an entire
mapped validation fan with exact solid-angle coordinates. Recursive optical
branches still use the normal surface models and may fall back to Embree at
undeclared seams, so this is an alternate integration backend rather than a
detector-specific physics implementation.

Custom surfaces are explicitly divided into two contracts:

- `custom_local` evaluates a probability-conserving interaction at the hit
  point. It may produce specular reflection/transmission, Lambertian
  reflection, detection, and absorption. It does not add hidden transport
  states.
- `custom_nonlocal` declares a finite input state basis. Transmitted light is
  conservatively deposited into that basis. The plugin declares intrinsic
  losses and a distribution over rays expressed relative to its own surface
  group. Its intrinsic linear map may be supplied either as explicit CSR
  matrices or through `oos_function_operator_v2`. The latter permits
  factorized, modal, FFT, or other matrix-free implementations without
  changing the core contract. The core traces the declared egress rays and
  composes their geometry-dependent destinations into the global operator.

Both contracts use `oos_surface_plugin_v3`; scene locality must match the
plugin ABI declaration. A functional nonlocal implementation additionally
exports `oos_get_function_operator_v2`. V2 requires paired linear forward and
Euclidean-adjoint callbacks on every supported backend. Input-dependent
clipping and renormalization are outside the ABI. Its assembled state layout is

```text
[ordinary Lambertian/local-plugin states,
 nonlocal surface 0 states,
 nonlocal surface 1 states, ...]
```

No plugin may replace the complete scene operator or embed another surface's
primitive/channel identifiers. In particular, a functional block consumes
only its own input state weights and produces:

```text
retained intrinsic state + surface-relative egress + intrinsic losses
```

The LXe finite-cylinder plugin is nonlocal: GXe-to-LXe Fresnel transmission is
deposited into its entry phase basis, while Fresnel reflection remains a
generic local behavior. Its explicit-seven collision/diffusion calculation
is naturally factorized and therefore uses the functional contract instead
of expanding a prohibitively large CSR emission matrix. It produces rays
relative to the LXe surface. Embree then maps those rays to PTFE, sensitive
surfaces, another nonlocal interface, or a named loss.

The image-source method is not part of the generic solver. Explicit Fresnel
branches are the default. A future planar image accelerator must prove
equivalence to explicit tracing and clip finite planes at their real boundary.
