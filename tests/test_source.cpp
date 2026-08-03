#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "oos/source.hpp"

TEST_CASE("built-in source quadratures are normalized") {
  const auto path =
      std::filesystem::temp_directory_path() / "oos-source-test.yaml";
  {
    std::ofstream output(path);
    output << R"(schema_version: 1
sources:
  - id: point
    spatial: {type: point, position_mm: [0, 0, 0]}
    angular: {type: isotropic, count: 16}
  - id: line
    spatial:
      {type: line_segment, start_mm: [0, 0, 0], end_mm: [0, 0, 1], count: 4}
    angular: {type: cone, axis: [0, 0, 1], half_angle_deg: 30, count: 8}
  - id: disk
    spatial:
      {type: disk, center_mm: [0, 0, 0], axis: [0, 0, 1],
       radius_mm: 2, count: 7}
    angular: {type: cosine, axis: [0, 0, 1], count: 9}
  - id: cylinder
    spatial:
      {type: cylinder, center_mm: [0, 0, 0], axis: [0, 0, 1],
       radius_mm: 2, length_mm: 3, count: 7}
    angular: {type: isotropic, count: 5}
  - id: box
    spatial: {type: box, min_mm: [-1, -1, -1], max_mm: [1, 1, 1], count: 7}
    angular: {type: isotropic, count: 5}
  - id: tetra
    spatial:
      type: tetrahedron
      vertices_mm: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
      count: 7
    angular:
      type: explicit
      directions:
        - {direction: [0, 0, 1], weight: 2}
        - {direction: [0, 1, 0], weight: 1}
)";
  }
  oos::Scene scene;
  scene.primary_domain = 4;
  const auto sources = oos::load_sources_yaml(path, scene);
  REQUIRE(sources.size() == 6);
  for (const auto& source : sources) {
    double total = 0.0;
    for (const auto& ray : source.rays) {
      total += ray.stokes.i;
      REQUIRE(ray.domain == 4);
    }
    REQUIRE(total == Catch::Approx(1.0).margin(1e-14));
  }
  std::filesystem::remove(path);
}

TEST_CASE("line sources support Gauss-Legendre spatial quadrature") {
  const auto directory =
      std::filesystem::temp_directory_path() / "oos-gauss-line-source";
  std::filesystem::create_directories(directory);
  const auto path = directory / "sources.yaml";
  {
    std::ofstream stream(path);
    stream << R"(schema_version: 1
sources:
  - id: gauss-line
    domain: 0
    spatial:
      type: line_segment
      start_mm: [0, 0, 0]
      end_mm: [0, 0, 5]
      count: 4
      quadrature: gauss_legendre
    angular:
      type: explicit
      directions:
        - direction: [0, 0, 1]
          weight: 1
)";
  }
  oos::Scene scene;
  scene.primary_domain = 0;
  const auto sources = oos::load_sources_yaml(path, scene);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].rays.size() == 4);
  const std::array<double, 4> expected_z{
      0.347159225553966, 1.65004769808296, 3.34995230191704,
      4.65284077444603};
  const std::array<double, 4> expected_weight{
      0.173927422568727, 0.326072577431273, 0.326072577431273,
      0.173927422568727};
  for (std::size_t i = 0; i < expected_z.size(); ++i) {
    CHECK(sources[0].rays[i].ray.origin.z == Catch::Approx(expected_z[i]));
    CHECK(sources[0].rays[i].stokes.i ==
          Catch::Approx(expected_weight[i]));
  }
  std::filesystem::remove_all(directory);
}

TEST_CASE("isotropic product quadrature matches the legacy rule") {
  const auto path =
      std::filesystem::temp_directory_path() /
      "oos-isotropic-product-source.yaml";
  {
    std::ofstream stream(path);
    stream << R"(schema_version: 1
sources:
  - id: product
    spatial: {type: point, position_mm: [0, 0, 0]}
    angular: {type: isotropic_product, mu_order: 8, phi_count: 48}
)";
  }
  oos::Scene scene;
  scene.primary_domain = 0;
  const auto sources = oos::load_sources_yaml(path, scene);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].rays.size() == 8 * 48);
  double total = 0.0;
  double mean_z = 0.0;
  double mean_z2 = 0.0;
  for (const auto& ray : sources[0].rays) {
    total += ray.stokes.i;
    mean_z += ray.stokes.i * ray.ray.direction.z;
    mean_z2 += ray.stokes.i * ray.ray.direction.z *
               ray.ray.direction.z;
  }
  CHECK(total == Catch::Approx(1.0).margin(1e-14));
  CHECK(mean_z == Catch::Approx(0.0).margin(1e-15));
  CHECK(mean_z2 == Catch::Approx(1.0 / 3.0).margin(1e-14));
  std::filesystem::remove(path);
}

TEST_CASE("line-neighborhood sources are uniform deterministic samples") {
  const auto path =
      std::filesystem::temp_directory_path() /
      "oos-line-neighborhood-source.yaml";
  {
    std::ofstream stream(path);
    stream << R"(schema_version: 1
sources:
  - id: line
    domain: 0
    spatial:
      type: rectangular_line_neighborhood
      line_center_mm: [2, -0.5, 17.499]
      obstacle_half_width_mm: 0.0025
      obstacle_half_thickness_mm: 0.0005
      maximum_distance_mm: 0.006
      medium_z_max_mm: 17.5
      count: 32
    angular:
      type: explicit
      directions:
        - direction: [0, 0, -1]
          weight: 1
)";
  }
  oos::Scene scene;
  scene.primary_domain = 0;
  const auto sources = oos::load_sources_yaml(path, scene);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].rays.size() == 32);
  double total = 0.0;
  for (const auto& ray : sources[0].rays) {
    const double dy =
        std::max(std::abs(ray.ray.origin.y + 0.5) - 0.0025, 0.0);
    const double dz =
        std::max(std::abs(ray.ray.origin.z - 17.499) - 0.0005, 0.0);
    CHECK(std::hypot(dy, dz) <= 0.006);
    CHECK(ray.ray.origin.z < 17.5);
    total += ray.stokes.i;
  }
  CHECK(total == Catch::Approx(1.0).margin(1e-14));
  std::filesystem::remove(path);
}

TEST_CASE("isotropic surface shape-factor sources retain spatial quadrature") {
  const auto path =
      std::filesystem::temp_directory_path() /
      "oos-isotropic-shape-factor-source.yaml";
  {
    std::ofstream stream(path);
    stream << R"(schema_version: 1
sources:
  - id: shape-factor
    domain: 3
    spatial:
      type: line_segment
      start_mm: [0, 0, 0]
      end_mm: [0, 0, 5]
      count: 4
      quadrature: gauss_legendre
    angular:
      type: isotropic_surface_shape_factor
      relative_tolerance: 2.5e-6
      maximum_subdivision_depth: 0
      minimum_refinement_solid_angle_fraction: 3.0e-5
      minimum_feature_solid_angle_fraction: 4.0e-8
      feature_dihedral_degrees: 7.5
      aperture_edge_phi_order: 12
      aperture_edge_weight_threshold: 6.0e-8
)";
  }
  oos::Scene scene;
  scene.primary_domain = 0;
  const auto sources = oos::load_sources_yaml(path, scene);
  REQUIRE(sources.size() == 1);
  CHECK(sources[0].integration ==
        oos::SourceIntegration::isotropic_surface_shape_factor);
  REQUIRE(sources[0].rays.empty());
  REQUIRE(sources[0].points.size() == 4);
  double total = 0.0;
  for (const auto& point : sources[0].points) {
    total += point.stokes.i;
    CHECK(point.domain == 3);
  }
  CHECK(total == Catch::Approx(1.0).margin(1.0e-14));
  CHECK(sources[0].shape_factor.relative_tolerance ==
        Catch::Approx(2.5e-6));
  CHECK(sources[0].shape_factor.maximum_subdivision_depth == 0);
  CHECK(sources[0]
            .shape_factor.minimum_refinement_solid_angle_fraction ==
        Catch::Approx(3.0e-5));
  CHECK(sources[0].shape_factor.minimum_feature_solid_angle_fraction ==
        Catch::Approx(4.0e-8));
  CHECK(sources[0].shape_factor.feature_dihedral_degrees ==
        Catch::Approx(7.5));
  CHECK(sources[0].shape_factor.aperture_edge_phi_order == 12);
  CHECK(sources[0].shape_factor.aperture_edge_weight_threshold ==
        Catch::Approx(6.0e-8));
  std::filesystem::remove(path);
}
