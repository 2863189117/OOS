#pragma once

#include <cstdint>
#include <filesystem>
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

struct ShapeFactorOptions {
  // The adaptive estimate is the L1 difference between one centroid
  // evaluation and the sum over four child triangles. The requested global
  // tolerance is divided across top-level boundary triangles; each child
  // tree redistributes its budget in proportion to measured child errors.
  double relative_tolerance{1.0e-5};
  // Hard recursion cap. Zero permits the mandatory parent-to-four-children
  // assessment but no further recursion. At the cap the child sum is accepted
  // and its parent/child L1 difference is reported instead of throwing.
  std::uint32_t maximum_subdivision_depth{8};
  // Smooth elements below this fraction of 4*pi use their exact solid angle
  // with one centroid behavior evaluation and are not subdivided.
  double minimum_refinement_solid_angle_fraction{1.0e-5};
  // Elements touching a material/domain junction or a geometric crease use a
  // lower threshold. Only descendants that still touch the feature edge
  // inherit this threshold.
  double minimum_feature_solid_angle_fraction{1.0e-8};
  double feature_dihedral_degrees{5.0};
  // Finite polar cells that project across a circular aperture use this
  // Gauss-Legendre order in azimuth.  Cells below the weight threshold retain
  // the center-visibility decision used by the optimized path.
  std::uint32_t aperture_edge_phi_order{8};
  double aperture_edge_weight_threshold{1.0e-7};
};

struct SourceQuadrature {
  std::string id;
  SourceIntegration integration{SourceIntegration::discrete_rays};
  std::vector<SourceRay> rays;
  std::vector<SourcePoint> points;
  ShapeFactorOptions shape_factor;
};

std::vector<Vec3> uniform_rectangular_line_neighborhood_samples(
    const Vec3& line_center_mm, double obstacle_half_width_mm,
    double obstacle_half_thickness_mm, double maximum_distance_mm,
    double medium_z_max_mm, std::uint32_t count);

std::vector<SourceQuadrature> load_sources_yaml(
    const std::filesystem::path& path, const Scene& scene);

SourceBatch trace_source_quadratures(
    const Scene& scene, const OperatorSet& operators,
    const std::vector<SourceQuadrature>& quadratures);

}  // namespace oos
