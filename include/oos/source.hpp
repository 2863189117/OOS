#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "oos/scene.hpp"
#include "oos/solver.hpp"

namespace oos {

struct SourceRay {
  Ray ray;
  Stokes stokes;
  Vec3 reference_axis{1.0, 0.0, 0.0};
  std::int32_t domain{-1};
};

struct SourcePoint {
  Vec3 position;
  Stokes stokes;
  std::int32_t domain{-1};
};

enum class SourceIntegration {
  discrete_rays,
  isotropic_surface_shape_factor,
};

enum class ShapeFactorBackend {
  automatic,
  generic_bvh,
  structured_analytic,
};

struct ShapeFactorOptions {
  // `automatic` uses explicit structured-visibility metadata where available
  // and falls back element-by-element. `structured_analytic` requires every
  // outward source-boundary candidate to carry a supported declaration.
  ShapeFactorBackend backend{ShapeFactorBackend::automatic};
  // The adaptive estimate is the L1 difference between one centroid
  // evaluation and the sum over four child triangles. The requested global
  // tolerance is divided across top-level boundary triangles; each child
  // tree redistributes its budget in proportion to measured child errors.
  double relative_tolerance{1.0e-5};
  // Hard recursion cap. Zero permits the mandatory parent-to-four-children
  // assessment but no further recursion. At the cap the child sum is accepted
  // and its parent/child L1 difference is reported instead of throwing.
  std::uint32_t maximum_subdivision_depth{8};
  // Smooth elements below this weighted fraction of 4*pi use one centroid
  // behavior evaluation and are not subdivided.
  double minimum_refinement_solid_angle_fraction{1.0e-5};
  // Elements touching a material/domain junction or a geometric crease use a
  // lower threshold. Only descendants that still touch the feature edge
  // inherit this threshold.
  double minimum_feature_solid_angle_fraction{1.0e-8};
  double feature_dihedral_degrees{5.0};
  // Triangles whose centroid projected-area estimate is at most this
  // fraction of 4*pi use that estimate directly. Larger triangles fall back
  // to the exact atan2 solid-angle formula. Zero preserves exact-only
  // weighting.
  double maximum_approximate_solid_angle_fraction{1.0e-5};
  // Finite polar cells that project across a circular aperture use this
  // Gauss-Legendre order in azimuth.  Cells below the weight threshold retain
  // the center-visibility decision used by the optimized path.
  std::uint32_t aperture_edge_phi_order{8};
  double aperture_edge_weight_threshold{1.0e-7};
  // Order 31 selects an embedded Gauss-Kronrod 15/31 rule in direction
  // cosine and an embedded periodic 32/64 rule in azimuth, so the error row
  // reuses the full-resolution ray traces. Other orders use independent
  // full- and half-resolution Gauss-Legendre passes.
  std::uint32_t structured_disk_mu_order{31};
  std::uint32_t structured_disk_phi_count{64};
};

struct SourceQuadrature {
  std::string id;
  SourceIntegration integration{SourceIntegration::discrete_rays};
  std::vector<SourceRay> rays;
  std::vector<SourcePoint> points;
  ShapeFactorOptions shape_factor;
};

// Reusable source-tracing state. Constructing this object validates the
// operator dimensions, builds the ray-intersection scene, loads surface
// plug-ins, and indexes source-boundary candidates once. Reuse one runtime
// across likelihood/grid batches to keep those fixed costs out of every
// source call.
class SourceTraceRuntime {
 public:
  SourceTraceRuntime(const Scene& scene, const OperatorSet& operators);
  ~SourceTraceRuntime();

  SourceTraceRuntime(const SourceTraceRuntime&) = delete;
  SourceTraceRuntime& operator=(const SourceTraceRuntime&) = delete;
  SourceTraceRuntime(SourceTraceRuntime&&) noexcept;
  SourceTraceRuntime& operator=(SourceTraceRuntime&&) noexcept;

  SourceBatch trace(
      const std::vector<SourceQuadrature>& quadratures) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::vector<Vec3> uniform_rectangular_line_neighborhood_samples(
    const Vec3& line_center_mm, double obstacle_half_width_mm,
    double obstacle_half_thickness_mm, double maximum_distance_mm,
    double medium_z_max_mm, std::uint32_t count);

std::vector<SourceQuadrature> load_sources_yaml(
    const std::filesystem::path& path, const Scene& scene);

}  // namespace oos
